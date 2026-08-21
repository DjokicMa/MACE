"""Non-interactive hardening of the D3 writers (CHARGE / POTENTIAL paths).

Every assertion is anchored to the CRYSTAL23 User's Manual (plain-text dump
line numbers, PDF page in brackets):

* ECHG [p.323], manual 25570-25573: the record table is "IDER" followed by
  "insert MAPNET input records" with NO conditional - the map-plane records are
  unconditional, so a deck of ECHG / IDER / END cannot run.
* MAPNET [p.344], manual 26755-26814: NPY ("number of points on the B-A
  segment", 26772), a keyword choosing the coordinate type, COORDINA, then
  XA,YA,ZA / XB,YB,ZB / XC,YC,ZC, and END.  ANGSTROM is the documented default
  and FRACTION/BOHR are the alternatives; the unit keyword governs SUBSEQUENT
  input (manual 3404-3405, 25964-25966), hence it precedes the coordinates.
  FRACTION is fractional only along the periodic directions (manual 3404-3412:
  x,y,z for 3D; x,y for 2D with z in Angstrom/bohr; x for 1D).
* ECH3 [p.322], manual 25470-25500 and POT3 [p.351], manual 27316-27347: after
  RANGE come all the minima on one record and all the maxima on the next -
  ZMIN/ZMAX (2D), YMIN,ZMIN / YMAX,ZMAX (1D), XMIN,YMIN,ZMIN / XMAX,YMAX,ZMAX
  (0D) - as a "boundary for non-periodic dimensions (au)" (manual 25490), i.e.
  bohr.
* POTC [p.352], manual 27382-27386: "if NPU > 0 insert NPU records" of
  "point coordinates (cartesian, bohr)"; NPU < 0 reads them from POTC.INP.

Real CRYSTAL outputs under test/ are the oracles: a 0D molecular SP, a 2D slab
OPT and a 3D diamond SP.  No synthetic CRYSTAL output is used.
"""
import builtins
import json
import shutil
import sys
from pathlib import Path

import pytest

from conftest import find_data

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen
import d3_config
import d3_interactive as d3int


# ---------------------------------------------------------------- helpers ---

def _generator(out_file, calc_type, tmp_path, monkeypatch):
    """D3Generator on a REAL CRYSTAL output, writing into tmp_path."""
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    return d3gen.D3Generator(str(out_file), calc_type, output_dir=str(tmp_path))


def _molecule_out():
    return find_data("SP/*MOLECULE*sp*.out", must_contain="MOLECULAR CALCULATION")


def _slab_out():
    return find_data("OPT/4LG_2x2_AA_opt*optimized.out", must_contain="SLAB CALCULATION")


def _no_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)


def _terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


def _stub_input(monkeypatch, answers):
    """builtins.input stub keyed on prompt substrings (writer-side prompts)."""

    def fake_input(prompt=""):
        for fragment, value in answers.items():
            if fragment in prompt:
                return value
        raise AssertionError(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr(builtins, "input", fake_input)


def _no_input(monkeypatch):
    """Any input() call is a test failure (proves nothing prompted)."""

    def boom(prompt=""):
        raise AssertionError(f"prompted with no terminal: {prompt!r}")

    monkeypatch.setattr(builtins, "input", boom)


def _messages(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


# ------------------------------------ item 4: characterization guards -------
# These two pass BEFORE and AFTER the change: they lock the SCALE branch and the
# terminal RANGE prompts that the RANGE consolidation must not disturb.

def test_ech3_scale_records_are_unchanged(tmp_path, monkeypatch):
    """Preservation guard: SCALE arity per dimensionality (manual 25492-25500).

    Passes before and after the RANGE change - the SCALE branch is untouched and
    must stay byte-identical (2D: ZSCALE; 0D: XSCALE,YSCALE,ZSCALE).
    """
    slab = _generator(_slab_out(), "CHARGE", tmp_path, monkeypatch)
    molecule = _generator(_molecule_out(), "CHARGE", tmp_path, monkeypatch)
    config = {"type": "ECH3", "n_points": 100, "use_range": False, "scale": 3}

    assert slab._write_charge_d3(config).splitlines() == [
        "ECH3", "100", "SCALE", "3", "END"]
    assert molecule._write_charge_d3(config).splitlines() == [
        "ECH3", "100", "SCALE", "3 3 3", "END"]


def test_range_prompts_still_work_at_a_terminal(tmp_path, monkeypatch):
    """Preservation guard: the interactive RANGE prompts and their record order.

    Passes before and after the change.  Manual 25496-25500: all minima on one
    record, all maxima on the next, so the 0D prompt order is X/Y/Z min then
    X/Y/Z max.
    """
    _terminal(monkeypatch)
    _stub_input(monkeypatch, {"X min": "-5.0", "Y min": "-6.0", "Z min": "-7.0",
                              "X max": "5.0", "Y max": "6.0", "Z max": "7.0"})
    gen = _generator(_molecule_out(), "CHARGE", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECH3", "n_points": 100,
                                 "use_range": True})

    assert deck.splitlines() == [
        "ECH3", "100", "RANGE", "-5.0 -6.0 -7.0", "5.0 6.0 7.0", "END"], deck


def test_potc_plane_average_unaffected(tmp_path, monkeypatch):
    """Preservation guard for the POTC point-list change (passes before/after).

    ICA=2 takes ZM,ZP and no ZD (manual 27390-27396); the custom-point handling
    must not touch it.
    """
    gen = _generator(_molecule_out(), "POTC", tmp_path, monkeypatch)

    deck = gen._write_potential_d3({"type": "POTC", "ica": 2,
                                    "z_range": (0.0, 10.0), "n_planes": 100})

    assert deck.splitlines() == ["POTC", "2 100 0", "0.0 10.0", "END"], deck


# --------------------------------------- item 1: wavefunction / main() ------

def test_missing_wavefunction_without_a_terminal_returns_false(tmp_path, monkeypatch):
    """No fort.9 and no terminal raised EOFError out of the yes/no prompt.

    Fails against the old behaviour (EOFError from yes_no_prompt), passes after:
    the guard reports and returns False, matching the D12 precedent at
    Crystal_d12/NewCifToD12.py:957-965.
    """
    out_file = tmp_path / "no_wavefunction_here.out"
    shutil.copy(_molecule_out(), out_file)
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)

    gen = d3gen.D3Generator(str(out_file), "CHARGE", output_dir=str(tmp_path))

    assert gen._copy_wavefunction() is False


def test_single_file_run_exits_nonzero_when_no_deck_is_written(tmp_path, monkeypatch, capsys):
    """main() discarded generate_d3()'s return and exited 0 with no .d3 file.

    Fails against the old behaviour (EOFError propagated out of main), passes
    after: the single-file path exits 1, which is the only signal the workflow
    executor gates on.
    """
    out_file = tmp_path / "no_wavefunction_here.out"
    shutil.copy(_molecule_out(), out_file)
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD3.py", "--input", str(out_file),
                                      "--calc-type", "CHARGE",
                                      "--output-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as exc:
        d3gen.main()

    assert exc.value.code == 1
    assert not list(tmp_path.glob("*.d3")), "no deck must be left behind"


def test_batch_run_exits_nonzero_when_every_file_fails(tmp_path, monkeypatch):
    """The batch loop discarded every return value and always exited 0.

    Fails against the old behaviour (EOFError out of the first file's prompt),
    passes after: with no wavefunction anywhere, nothing is written and the run
    exits 1 instead of reporting a silent success.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for name in ("one.out", "two.out"):
        shutil.copy(_molecule_out(), batch_dir / name)
    config_file = tmp_path / "charge.json"
    config_file.write_text(json.dumps({
        "version": "1.0", "type": "d3_configuration", "calculation_type": "CHARGE",
        "configuration": {"calculation_type": "CHARGE", "type": "ECH3",
                          "n_points": 100, "scale": 3, "use_range": False}}))
    monkeypatch.chdir(batch_dir)
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    # --shared-settings: without it the batch path asks "use same configuration
    # for all files?" even when a config file was supplied, which is a separate
    # (pre-existing) non-interactive hole outside this change.
    monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD3.py", "--batch",
                                      "--calc-type", "CHARGE",
                                      "--shared-settings",
                                      "--config-file", str(config_file),
                                      "--output-dir", str(batch_dir)])

    with pytest.raises(SystemExit) as exc:
        d3gen.main()

    assert exc.value.code == 1
    assert not list(batch_dir.glob("*.d3"))


# ------------------------------------------- item 2: ECHG map plane ---------

_PLANE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_echg_map_points_in_the_config_emit_the_mapnet_block(tmp_path, monkeypatch, capsys):
    """A config carrying map_points but not need_map_points dropped MAPNET.

    Fails against the old behaviour (deck was "ECHG / 0 / END"), passes after.
    The MAPNET records are unconditional for ECHG (manual 25573), so the old
    deck made CRYSTAL read END as the integer NPY and abort.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "map_points": _PLANE, "map_coord_type": "F",
                                 "n_points": 50})

    assert deck.splitlines() == [
        "ECHG", "0", "50", "FRACTION", "COORDINA",
        "0.0 0.0 0.0", "1.0 0.0 0.0", "0.0 1.0 0.0", "END", "END"], deck


def test_echg_config_plane_in_angstrom_writes_no_unit_keyword(tmp_path, monkeypatch):
    """ANGSTROM is MAPNET's default (manual 26810), so it needs no keyword.

    Fails against the old behaviour (the plane was ignored entirely), passes
    after.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "map_points": _PLANE,
                                 "map_coord_type": "ANGSTROM", "n_points": 50})

    assert "FRACTION" not in deck
    assert deck.splitlines()[:3] == ["ECHG", "0", "50"], deck
    assert deck.splitlines()[3] == "COORDINA", deck


def test_echg_map_points_without_a_coord_type_are_refused(tmp_path, monkeypatch, capsys):
    """A plane with no unit is a silent lattice-parameter-sized error.

    Fails against the old behaviour (no such concept; the writer prompted or
    ignored the points), passes after: the writer refuses and names both units.
    FRACTION and ANGSTROM differ by the lattice parameter and CRYSTAL cannot
    detect the mistake (manual 3404-3405, 26810-26812).
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "map_points": _PLANE, "n_points": 50})

    assert deck is None
    messages = _messages(capsys)
    assert "map_coord_type" in messages
    assert "FRACTION" in messages and "ANGSTROM" in messages


def test_echg_unrecognised_coord_type_is_refused(tmp_path, monkeypatch, capsys):
    """A typo must not silently fall back to one of the two units."""
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "map_points": _PLANE, "map_coord_type": "X",
                                 "n_points": 50})

    assert deck is None
    assert "FRACTION" in _messages(capsys)


def test_echg_wrong_number_of_map_points_is_refused(tmp_path, monkeypatch, capsys):
    """MAPNET reads exactly three points A, B, C (manual 26773-26776)."""
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "map_points": _PLANE[:2],
                                 "map_coord_type": "F", "n_points": 50})

    assert deck is None
    assert "three" in _messages(capsys)


def test_echg_without_points_or_terminal_writes_no_deck(tmp_path, monkeypatch, capsys):
    """ECHG with no plane and no terminal crashed with EOFError mid-deck.

    Fails against the old behaviour (EOFError at the "Coordinate type" prompt),
    passes after: generate_d3 reports, returns None and leaves no .d3 behind -
    a half-written deck is worse than none because the caller gates on the file.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "CHARGE", tmp_path, monkeypatch)

    result = gen.generate_d3(shared_config={"type": "ECHG",
                                            "derivative_order": 0,
                                            "need_map_points": True})

    assert result is None
    assert not list(tmp_path.glob("*.d3"))
    assert "map_points" in _messages(capsys)


def test_typed_fractional_word_selects_the_fraction_keyword(tmp_path, monkeypatch):
    """Typing "Fractional" used to produce an ANGSTROM plane.

    Fails against the old behaviour (exact `== "F"` test, so anything longer
    than a bare F fell through to the Angstrom default and the map plane was
    built in the wrong unit), passes after: the answer is matched against a word
    set.
    """
    _terminal(monkeypatch)
    _stub_input(monkeypatch, {"Coordinate type": " Fractional ",
                              "Point A": "0.0 0.0 0.0",
                              "Point B": "1.0 0.0 0.0",
                              "Point C": "0.0 1.0 0.0",
                              "Number of points": "50"})
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "need_map_points": True})

    assert "FRACTION" in deck.splitlines(), deck


def test_typed_coordinate_type_typo_is_refused(tmp_path, monkeypatch, capsys):
    """An unrecognised typed unit silently meant Angstrom.

    Fails against the old behaviour (any non-"F" answer produced an Angstrom
    plane), passes after: the writer refuses rather than guessing the unit.
    """
    _terminal(monkeypatch)
    _stub_input(monkeypatch, {"Coordinate type": "cartesain",
                              "Point A": "0.0 0.0 0.0",
                              "Point B": "1.0 0.0 0.0",
                              "Point C": "0.0 1.0 0.0",
                              "Number of points": "50"})
    gen = _generator(_molecule_out(), "ECHG", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECHG", "derivative_order": 0,
                                 "need_map_points": True})

    assert deck is None
    assert "ANGSTROM" in _messages(capsys)


# ------------------------------------------------ item 3: POTC points -------

def test_potc_points_from_config_are_written(tmp_path, monkeypatch):
    """A config carrying the coordinates implies custom points.

    Fails against the old behaviour (custom_points defaults to False, so the
    header promised NPU records and none followed - CRYSTAL then read the deck's
    END as an X,Y,Z triple), passes after.  Manual 27382-27385: "if NPU > 0
    insert NPU records" of cartesian bohr coordinates.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "POTC", tmp_path, monkeypatch)

    deck = gen._write_potential_d3({"type": "POTC", "ica": 0, "n_points": 2,
                                    "points": [[0.0, 0.0, 0.0],
                                               [0.0, 0.0, 2.5]]})

    assert deck.splitlines() == ["POTC", "0 2 0", "0.0 0.0 0.0",
                                 "0.0 0.0 2.5", "END"], deck


def test_potc_point_count_must_match_npu(tmp_path, monkeypatch, capsys):
    """A header/record disagreement is fatal and must not be guessed away.

    Fails against the old behaviour (points were unreachable from a config),
    passes after: the writer refuses and names both counts, because CRYSTAL
    reads exactly NPU records (manual 27382).
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "POTC", tmp_path, monkeypatch)

    deck = gen._write_potential_d3({"type": "POTC", "ica": 0, "n_points": 3,
                                    "points": [[0.0, 0.0, 0.0]]})

    assert deck is None
    messages = _messages(capsys)
    assert "3" in messages and "1" in messages


def test_potc_custom_points_without_a_terminal_write_no_deck(tmp_path, monkeypatch, capsys):
    """custom_points with no terminal crashed with EOFError.

    Fails against the old behaviour (EOFError at "Point 1 (x y z)"), passes
    after: reported, no deck written.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "POTENTIAL", tmp_path, monkeypatch)

    result = gen.generate_d3(shared_config={"type": "POTC", "ica": 0,
                                            "n_points": 2,
                                            "custom_points": True})

    assert result is None
    assert not list(tmp_path.glob("*.d3"))
    assert "points" in _messages(capsys)


# -------------------------------------------------- item 4: RANGE ----------

def test_range_from_config_needs_no_terminal(tmp_path, monkeypatch):
    """RANGE was unreachable without a terminal - it always prompted.

    Fails against the old behaviour (EOFError at "X min (bohr)"), passes after.
    Manual 25498-25500: 0D takes XMIN,YMIN,ZMIN then XMAX,YMAX,ZMAX in bohr.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "POTENTIAL", tmp_path, monkeypatch)

    deck = gen._write_potential_d3({"type": "POT3", "n_points": 100, "itol": 5,
                                    "use_range": True,
                                    "range_min": [-5.0, -6.0, -7.0],
                                    "range_max": [5.0, 6.0, 7.0]})

    assert deck.splitlines() == ["POT3", "100", "5", "RANGE",
                                 "-5.0 -6.0 -7.0", "5.0 6.0 7.0", "END"], deck


def test_range_arity_must_match_dimensionality(tmp_path, monkeypatch, capsys):
    """A 2D slab takes ZMIN/ZMAX only (manual 25494-25495).

    Fails against the old behaviour (RANGE could not be configured at all),
    passes after: three values for a slab are refused instead of being written
    as a record CRYSTAL would misread.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_slab_out(), "CHARGE", tmp_path, monkeypatch)

    deck = gen._write_charge_d3({"type": "ECH3", "n_points": 100,
                                 "use_range": True,
                                 "range_min": [-5.0, -6.0, -7.0],
                                 "range_max": [5.0, 6.0, 7.0]})

    assert deck is None
    assert "Z" in _messages(capsys)


def test_range_without_a_terminal_writes_no_deck(tmp_path, monkeypatch, capsys):
    """use_range with no boundaries and no terminal crashed with EOFError.

    Fails against the old behaviour, passes after: reported, no deck written.
    """
    _no_terminal(monkeypatch)
    _no_input(monkeypatch)
    gen = _generator(_molecule_out(), "CHARGE", tmp_path, monkeypatch)

    result = gen.generate_d3(shared_config={"type": "ECH3", "n_points": 100,
                                            "use_range": True})

    assert result is None
    assert not list(tmp_path.glob("*.d3"))
    assert "range_min" in _messages(capsys)


# ------------------------------------- item 5 salvage: saved ECHG config ----

def test_saved_echg_config_survives_reload_validation(monkeypatch):
    """A saved ECHG configuration could never be loaded again.

    Fails against the old behaviour: configure_charge_density_calculation's ECHG
    branch never set n_points, and validate_d3_config requires it for CHARGE
    (d3_config.py:257), so `--config-file` on a saved ECHG config exited 1.
    n_points is MAPNET's NPY, "number of points on the B-A segment"
    (manual 26772).
    """
    answers = {"Select type": "2", "Select order": "0",
               "Number of points along the B-A segment": "50"}

    def read(prompt="", valid_set=None):
        for fragment, value in answers.items():
            if fragment in prompt:
                return value
        return ""

    monkeypatch.setattr(d3int, "_nav_read", read)
    monkeypatch.setattr(d3int, "_nav_int",
                        lambda prompt="", default=None, choices=None:
                        int(read(prompt) or (0 if default is None else default)))
    monkeypatch.setattr(d3int, "yes_no_prompt", lambda *a, **k: True)

    config = d3int.configure_d3_calculation("CHARGE")

    assert config["type"] == "ECHG"
    assert config["n_points"] == 50
    assert d3_config.validate_d3_config(config) == (True, [])
