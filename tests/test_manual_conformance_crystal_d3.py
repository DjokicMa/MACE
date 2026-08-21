"""CRYSTAL23 manual-conformance regression tests for the D3 generators.

Every assertion here is anchored to the CRYSTAL23 User's Manual:

* BAND (manual 24241-24264): the segment extremes I1,I2,I3 / J1,J2,J3 are
  INTEGERS expressed in units of 1/ISS, where ISS is the shrinking factor on
  the same record. Fractional coordinates in those fields are not legal input.
* ECHG (manual 25570-25583) + MAPNET (manual 26755-26826): IDER is 0 or 1,
  "MAPNET" is a dummy keyword that is never written, the block is
  NPY / <unit> / COORDINA / A / B / C / END, and nBC is chosen by CRYSTAL.
  FRACTION/ANGSTROM/BOHR set the unit for SUBSEQUENT input (manual 3403-3415,
  25964-25966), so they must precede the coordinates.
* POTC (manual 27370-27397, 27425-27426): ICA is 0, 2 or 3 (1 is "not
  implemented"), ZD is read only for ICA=3, and NPU < 0 is the flag that makes
  CRYSTAL read the points from POTC.INP.

Where a real deck exists under test/ it is used as the oracle rather than a
hand-written fixture.
"""
import builtins
import sys
from pathlib import Path

import pytest

from conftest import find_data

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen
import d3_interactive as d3int
import d3_kpoints


# ---------------------------------------------------------------- helpers ---

def _generator(out_file, calc_type, tmp_path, monkeypatch):
    """D3Generator on a REAL CRYSTAL output, writing into tmp_path.

    The wavefunction copy is stubbed out: it only moves a fort.9 around and
    would otherwise prompt when one is absent.
    """
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    return d3gen.D3Generator(str(out_file), calc_type, output_dir=str(tmp_path))


def _written_deck(tmp_path):
    decks = sorted(tmp_path.glob("*.d3"))
    assert len(decks) == 1, f"expected exactly one D3 file, got {decks}"
    return decks[0].read_text()


def _band_segment_lines(deck):
    """Coordinate records of a BAND deck (everything after the NLINE header)."""
    lines = deck.strip().splitlines()
    return [ln for ln in lines[3:] if ln.strip() and ln.strip() != "END"]


def _all_tokens_integer(segment_lines):
    for ln in segment_lines:
        for token in ln.split():
            if "." in token:
                return False
            try:
                int(token)
            except ValueError:
                return False
    return True


class _Prompts:
    """Scripted stand-in for the menu_nav readers used by d3_interactive.

    Answers are matched on a substring of the prompt so the same script drives
    the pre-fix and post-fix menus (whose wording differs); unmatched prompts
    take their default.
    """

    def __init__(self, answers):
        self.answers = answers
        self.seen = []

    def _lookup(self, prompt):
        self.seen.append(prompt)
        for fragment, value in self.answers.items():
            if fragment in prompt:
                return value
        return None

    def read(self, prompt="", valid_set=None):
        value = self._lookup(prompt)
        return "" if value is None else str(value)

    def nav_int(self, prompt="", default=None, choices=None):
        value = self._lookup(prompt)
        return int(default if default is not None else 0) if value is None else int(value)

    def nav_float(self, prompt="", default=None):
        value = self._lookup(prompt)
        return float(default if default is not None else 0.0) if value is None else float(value)

    def install(self, monkeypatch):
        monkeypatch.setattr(d3int, "_nav_read", self.read)
        monkeypatch.setattr(d3int, "_nav_int", self.nav_int)
        monkeypatch.setattr(d3int, "_nav_float", self.nav_float)
        return self


def _stub_input(monkeypatch, answers):
    """builtins.input stub keyed on prompt substrings (writer-side prompts)."""

    def fake_input(prompt=""):
        for fragment, value in answers.items():
            if fragment in prompt:
                return value
        raise AssertionError(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr(builtins, "input", fake_input)


# ------------------------------------------------- BAND k-point scaling -----

def test_reference_band_deck_is_the_integer_shrink_unit_oracle():
    """The shipped BAND deck confirms the manual's ISS-unit convention.

    Guards the oracle used by the fallback tests below: a real, CRYSTAL-accepted
    deck carries integer segment extremes and a non-zero ISS on the header.
    """
    ref = find_data("BAND/*_band.d3", must_contain="BAND")
    deck = ref.read_text()
    header = deck.strip().splitlines()[2].split()
    iss = int(header[1])
    assert iss > 0, "reference deck uses coordinate mode, so ISS must be > 0"
    assert _all_tokens_integer(_band_segment_lines(deck))


def test_seekpath_fallback_scales_segments_by_shrink(tmp_path, monkeypatch):
    """SeeK-path unavailable -> the fallback wrote FRACTIONAL k-points.

    Real failure prevented: with seekpath missing, CRYSTALOptToD3 emitted
    "0.5 0.0 0.5  0.0 0.0 0.0" records against a default ISS=16 header. CRYSTAL
    reads those fields as integers, so every band structure produced on a node
    without seekpath was silently computed along the wrong path.
    """
    out_file = find_data("BAND/1_dia*_band.out")
    monkeypatch.setattr(d3_kpoints, "get_seekpath_full_kpath", lambda *a, **k: None)
    monkeypatch.setattr(d3_kpoints, "get_literature_kpath_vectors", lambda *a, **k: [])

    gen = _generator(out_file, "BAND", tmp_path, monkeypatch)
    used = gen.generate_d3(shared_config={"path_method": "coordinates",
                                          "auto_path": True,
                                          "seekpath_full": True,
                                          "n_points": 1000})

    deck = _written_deck(tmp_path)
    header = deck.strip().splitlines()[2].split()
    segments = _band_segment_lines(deck)
    assert _all_tokens_integer(segments), deck
    assert int(header[1]) == used["shrink"], "header ISS must be the scaling shrink"
    assert any(int(tok) != 0 for ln in segments for tok in ln.split()), deck


def test_literature_fallback_scales_segments_by_shrink(tmp_path, monkeypatch):
    """Literature k-path unavailable -> same unscaled-coordinate failure.

    Real failure prevented: the literature_path branch had its own bare
    get_kpoint_coordinates_from_labels fallback that stored fractional vectors
    and left config["shrink"] untouched.
    """
    out_file = find_data("BAND/1_dia*_band.out")
    monkeypatch.setattr(d3_kpoints, "get_literature_kpath_vectors", lambda *a, **k: [])

    gen = _generator(out_file, "BAND", tmp_path, monkeypatch)
    used = gen.generate_d3(shared_config={"path_method": "coordinates",
                                          "auto_path": True,
                                          "literature_path": True,
                                          "n_points": 1000})

    deck = _written_deck(tmp_path)
    header = deck.strip().splitlines()[2].split()
    assert _all_tokens_integer(_band_segment_lines(deck)), deck
    assert int(header[1]) == used["shrink"]


def test_interactive_literature_fallback_scales_segments(monkeypatch):
    """d3_interactive's literature fallback stored fractional segments.

    Real failure prevented: option 3 (literature path with vectors) on a space
    group with no literature entry handed CRYSTALOptToD3 a config whose
    "segments" were fractional, producing the same corrupt deck.
    """
    out_file = find_data("BAND/1_dia*_band.out")
    monkeypatch.setattr(d3int, "get_literature_kpath_vectors", lambda *a, **k: [])
    _Prompts({"Select method": "1", "Select format": "3"}).install(monkeypatch)

    config = d3int.configure_band_calculation(str(out_file))

    segments = config["segments"]
    assert segments, "fallback must still produce a path"
    assert all(isinstance(v, int) for seg in segments for v in seg), segments
    assert isinstance(config["shrink"], int) and config["shrink"] > 0


def test_interactive_seekpath_fallback_scales_segments(monkeypatch):
    """d3_interactive's SeeK-path fallback stored fractional segments.

    Real failure prevented: option 4 (SeeK-path full path) with the library
    absent fell back to unscaled fractional vectors.
    """
    out_file = find_data("BAND/1_dia*_band.out")
    monkeypatch.setattr(d3int, "get_seekpath_full_kpath", lambda *a, **k: None)
    _Prompts({"Select method": "1", "Select format": "4"}).install(monkeypatch)

    config = d3int.configure_band_calculation(str(out_file))

    segments = config["segments"]
    assert segments, "fallback must still produce a path"
    assert all(isinstance(v, int) for seg in segments for v in seg), segments
    assert isinstance(config["shrink"], int) and config["shrink"] > 0


# ------------------------------------------------------ ECHG / MAPNET -------

def _echg_deck(tmp_path, monkeypatch, coord_type):
    out_file = find_data("SP/*MOLECULE*sp*.out", must_contain="MOLECULAR CALCULATION")
    gen = _generator(out_file, "ECHG", tmp_path, monkeypatch)
    _stub_input(monkeypatch, {
        "Coordinate type": coord_type,
        "Point A": "0.0 0.0 0.0",
        "Point B": "1.0 0.0 0.0",
        "Point C": "0.0 1.0 0.0",
        "Number of points": "50",
    })
    return gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "need_map_points": True})


def test_echg_mapnet_block_matches_manual_record_order(tmp_path, monkeypatch):
    """The ECHG map block was structurally invalid and died on the first read.

    Real failure prevented: the deck opened the MAPNET block with the literal
    string "MAPNET" where CRYSTAL performs a free-format integer read of NPY,
    never wrote NPY or the COORDINA keyword, invented "CARTESIAN", appended a
    non-existent "n_ab n_bc" record and never terminated the block.
    """
    deck = _echg_deck(tmp_path, monkeypatch, "F")

    assert deck.splitlines() == [
        "ECHG",
        "0",
        "50",
        "FRACTION",
        "COORDINA",
        "0.0 0.0 0.0",
        "1.0 0.0 0.0",
        "0.0 1.0 0.0",
        "END",
        "END",
    ], deck


def test_echg_unit_keyword_precedes_the_coordinates(tmp_path, monkeypatch):
    """FRACTION governs SUBSEQUENT input, so it cannot follow the points.

    Real failure prevented: a unit keyword written after XA/XB/XC cannot
    retroactively reinterpret records CRYSTAL has already read - the map plane
    would silently be built from Angstrom values with no error.
    """
    lines = _echg_deck(tmp_path, monkeypatch, "F").splitlines()

    assert lines.index("FRACTION") < lines.index("COORDINA")
    assert lines.index("COORDINA") < lines.index("0.0 0.0 0.0")


def test_echg_cartesian_map_uses_the_angstrom_default(tmp_path, monkeypatch):
    """Cartesian input must not emit the bogus "CARTESIAN" keyword.

    Real failure prevented: "CARTESIAN" is not in MAPNET's keyword set, so
    CRYSTAL aborted on it; Angstrom is the documented default and needs no
    keyword at all.
    """
    deck = _echg_deck(tmp_path, monkeypatch, "C")

    assert "CARTESIAN" not in deck
    assert "MAPNET" not in deck
    assert deck.splitlines() == [
        "ECHG",
        "0",
        "50",
        "COORDINA",
        "0.0 0.0 0.0",
        "1.0 0.0 0.0",
        "0.0 1.0 0.0",
        "END",
        "END",
    ], deck


def test_echg_derivative_menu_rejects_undefined_order(monkeypatch):
    """The menu offered a "Laplacian" IDER=2 that CRYSTAL does not define.

    Real failure prevented: choosing option 2 wrote IDER=2 into the deck; the
    manual defines IDER 0 and 1 only ("order of the derivative - < 2").
    """
    _Prompts({"Select type": "2", "Select order": "2"}).install(monkeypatch)

    config = d3int.configure_charge_density_calculation()

    assert config["type"] == "ECHG"
    assert config["derivative_order"] == 0


def test_echg_declining_the_map_plane_warns(monkeypatch, capsys):
    """Answering "no" to the map plane silently produced an invalid deck.

    Real failure prevented: MAPNET records are unconditional for ECHG, so a
    deck of ECHG / IDER / END makes CRYSTAL read END as the integer NPY and
    abort. The escape hatch is kept (it is the only non-interactive route) but
    it now says so.
    """
    _Prompts({"Select type": "2", "Select order": "0",
              "map plane": "n"}).install(monkeypatch)

    config = d3int.configure_charge_density_calculation()

    assert config["need_map_points"] is False
    assert "MAPNET" in capsys.readouterr().out


# --------------------------------------------------------------- POTC -------

def _potc_generator(tmp_path, monkeypatch):
    out_file = find_data("SP/*MOLECULE*sp*.out", must_contain="MOLECULAR CALCULATION")
    return _generator(out_file, "POTC", tmp_path, monkeypatch)


def test_potc_read_from_file_keeps_npu_negative(tmp_path, monkeypatch):
    """A config carrying a negative NPU was negated back to positive.

    Real failure prevented: the interactive layer stored -N for "read from
    file" and the writer negated it again, so the deck promised N coordinate
    records that were never written and CRYSTAL consumed the following END as
    an X,Y,Z triple.
    """
    gen = _potc_generator(tmp_path, monkeypatch)

    deck = gen._write_potential_d3({"type": "POTC", "ica": 0, "n_points": -12,
                                    "read_file": True})

    assert deck.splitlines() == ["POTC", "0 -12 0", "END"], deck


def test_potc_read_from_file_end_to_end(tmp_path, monkeypatch):
    """The configured POTC.INP deck must reach CRYSTAL with NPU < 0.

    Real failure prevented: the double negation above, exercised through the
    menu that actually produces the config.
    """
    _Prompts({"Select type": "2", "Select mode": "0", "Select option": "3",
              "Number of points in file": "12"}).install(monkeypatch)
    config = d3int.configure_potential_calculation()
    assert config["read_file"] is True

    gen = _potc_generator(tmp_path, monkeypatch)
    deck = gen._write_potential_d3(config)

    assert "0 -12 0" in deck, deck


def test_potc_volume_average_is_ica3_and_carries_zd(tmp_path, monkeypatch):
    """ZD belongs to ICA=3, not ICA=2.

    Real failure prevented: the writer attached the half-thickness record to
    the plane average (ICA=2), where CRYSTAL reads it as the next properties
    keyword, and omitted it from the volume average (ICA=3), where it is
    mandatory.
    """
    gen = _potc_generator(tmp_path, monkeypatch)
    base = {"type": "POTC", "z_range": (0.0, 10.0), "n_planes": 100,
            "slice_thickness": 0.25}

    volume = gen._write_potential_d3(dict(base, ica=3))
    plane = gen._write_potential_d3(dict(base, ica=2))

    assert volume.splitlines() == ["POTC", "3 100 0", "0.0 10.0", "0.25", "END"], volume
    assert plane.splitlines() == ["POTC", "2 100 0", "0.0 10.0", "END"], plane


def test_potc_menu_maps_volume_average_to_ica3(monkeypatch):
    """The menu's "volume-averaged" option wrote ICA=2 (a plane average).

    Real failure prevented: the 0/1/2 menu was off by one against the manual's
    0/2/3, so "plane-averaged" selected the unimplemented ICA=1 and
    "volume-averaged" selected the plane average.
    """
    _Prompts({"Select type": "2", "Select mode": "3", "Z minimum": "0",
              "Z maximum": "10", "Number of planes": "100",
              "thickness": "0.25"}).install(monkeypatch)

    config = d3int.configure_potential_calculation()

    assert config["ica"] == 3
    assert config["slice_thickness"] == 0.25


def test_potc_menu_refuses_unimplemented_ica1(monkeypatch):
    """ICA=1 is "not implemented" in CRYSTAL and must never reach the deck."""
    _Prompts({"Select type": "2", "Select mode": "1",
              "Select option": "1"}).install(monkeypatch)

    config = d3int.configure_potential_calculation()

    assert config["ica"] == 0


# ------------------------------------------- CHARGE+POTENTIAL combiner ------

def test_combined_deck_keeps_the_mapnet_terminator(tmp_path, monkeypatch):
    """rstrip('\\nEND') is a character-SET strip and ate the MAPNET END.

    Real failure prevented: an ECHG charge block ends "...\\nEND\\nEND" (MAPNET
    terminator + deck terminator). rstrip removed both, leaving the MAPNET
    block unterminated in the combined CHARGE+POTENTIAL deck.
    """
    out_file = find_data("SP/*MOLECULE*sp*.out", must_contain="MOLECULAR CALCULATION")
    gen = _generator(out_file, "CHARGE+POTENTIAL", tmp_path, monkeypatch)
    _stub_input(monkeypatch, {
        "Coordinate type": "F",
        "Point A": "0.0 0.0 0.0",
        "Point B": "1.0 0.0 0.0",
        "Point C": "0.0 1.0 0.0",
        "Number of points": "50",
    })
    config = {
        "charge_config": {"type": "ECHG", "derivative_order": 0,
                          "need_map_points": True},
        "potential_config": {"type": "POT3", "n_points": 100, "itol": 5,
                             "use_range": False, "scale": 3},
    }
    gen.generate_d3(shared_config=config)

    deck = _written_deck(tmp_path)
    assert "\nEND\nPOT3\n" in deck, deck
    assert deck.strip().endswith("END")
    assert deck.strip().splitlines().count("END") == 2, deck


def test_combined_ech3_pot3_still_matches_the_reference_deck(tmp_path, monkeypatch):
    """Preservation guard for the combiner rewrite (passes before and after).

    The shipped ECH3+POT3 deck under test/ECH3POT3 survived the old character-set
    rstrip only because its last charge record ends in a digit. It is locked
    here so the suffix-strip replacement cannot change a deck that is already
    correct.
    """
    out_file = find_data(
        "SP/1LiFSI-1DEC-conf1_MOLECULE*sp_HSESOL3C_optimized.out",
        must_contain="MOLECULAR CALCULATION")
    ref = find_data("ECH3POT3/1LiFSI-1DEC-conf1_MOLECULE*charge+potential.d3")
    expected = ref.read_text().splitlines()[:10]

    gen = _generator(out_file, "CHARGE+POTENTIAL", tmp_path, monkeypatch)
    config = {
        "charge_config": {"type": "ECH3", "n_points": 100, "use_range": False,
                          "scale": 3},
        "potential_config": {"type": "POT3", "n_points": 100, "itol": 5,
                             "use_range": False, "scale": 3},
    }
    gen.generate_d3(shared_config=config)

    assert _written_deck(tmp_path).splitlines() == expected
