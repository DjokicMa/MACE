"""CRYSTAL23 manual-conformance guards for the FREQCALC deck emitter.

Four deck-breaking defects are covered:

* ``write_frequency_section`` refused to emit IRSPEC unless a dielectric
  tensor/constant was supplied, so the ``ir_spectrum`` and ``ir_raman``
  templates silently produced a deck that computes no IR spectrum at all.
  The manual (line 16877) says "If the dielectric tensor is omitted, only
  the raw absorption spectrum is computed" — omitting it degrades the
  output, it does not invalidate the block.
* SCELPHONO was written *inside* the FREQCALC block, after DISPERSION.
  SCELPHONO is a geometry-block keyword (manual sec. 4.21, worked example
  at lines 4838-4851) and FREQCALC "must be the last keyword in the
  geometry input block" (manual 15506-15507), so the supercell has to
  precede it. SCELPHONO appears in neither the FREQCALC option list nor
  the DISPERSION sub-keyword table (manual 17123-17175).
* The 3-element ``scelphono`` branch wrote a single "2 2 2" line, but the
  expansion matrix E is read "by rows: 9 reals (3D)" (manual 4822-4826).
* A Raman-only config emitted NOINTENS immediately followed by INTRAMAN.
  The manual (16226-16227) states "the INTRAMAN should be always used
  together with the INTENS keyword", and RAMSPEC additionally requires
  "prior calculation of Raman intensities" (16932-16935).

Where a real reference deck exists it is used as the oracle rather than a
synthetic fixture, so the layout is pinned to input CRYSTAL has actually
accepted.
"""
import io
import json

import pytest

import d12_calc_freq
from d12_calc_freq import FREQ_TEMPLATES, write_frequency_section

from conftest import REPO_ROOT, find_data

# phonon_bands resolves an AUTO band path from symmetry, so a crystal system
# and space group must be supplied or d3_kpoints raises TypeError.
CUBIC = ("cubic", 227)


def _deck(settings, crystal_system=CUBIC[0], space_group=CUBIC[1]):
    """Emitted FREQCALC deck for ``settings`` as a list of stripped lines."""
    buf = io.StringIO()
    write_frequency_section(buf, dict(settings), crystal_system, space_group)
    return [ln.strip() for ln in buf.getvalue().splitlines()]


def _template(name):
    return _deck(FREQ_TEMPLATES[name])


# --------------------------------------------------------------------------
# IRSPEC without a dielectric tensor/constant
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ir_spectrum", "ir_raman"])
def test_ir_templates_emit_irspec(name):
    """The IR templates must actually request an IR spectrum.

    Prevents the regression where every ir_spectrum/ir_raman deck ran a
    frequency calculation but never wrote IRSPEC, so no IRSPEC.DAT was
    produced and the user got silence instead of a spectrum.
    """
    assert "IRSPEC" in _template(name)


def test_irspec_block_is_closed():
    """IRSPEC opens a block that must be closed by END (manual 16744-16745)."""
    deck = _template("ir_spectrum")
    i = deck.index("IRSPEC")
    assert "END" in deck[i:], "IRSPEC block was left unterminated"


def test_irspec_without_dielectric_matches_real_accepted_deck():
    """Oracle: a real deck CRYSTAL accepted has IRSPEC and no DIELTENS/DIELISO.

    The reference deck is a completed FREQ run whose IRSPEC.DAT exists on
    disk, proving the gate this test guards was rejecting input CRYSTAL is
    perfectly happy with.
    """
    ref = find_data("FREQ/*temp.d12", must_contain="IRSPEC")
    lines = [ln.strip() for ln in ref.read_text(errors="ignore").splitlines()]

    assert "IRSPEC" in lines
    assert not [ln for ln in lines if ln.startswith(("DIELTENS", "DIELISO"))], (
        f"{ref.name} was expected to exercise the no-dielectric path"
    )

    # The emitter, given the same intent, must produce the same two keywords.
    deck = _deck({"intensities": True, "ir_method": "CPHF", "irspec": True,
                  "spec_range": [0.0, 4000.0]})
    assert "IRSPEC" in deck
    assert not [ln for ln in deck if ln.startswith(("DIELTENS", "DIELISO"))]


def test_irspec_still_emitted_with_dielectric():
    """Supplying a dielectric constant must keep DIELISO *and* IRSPEC.

    Guards the previously-correct path: the fix removed a gate, it must not
    have removed the dielectric emission that LO/TO splitting depends on.
    """
    deck = _deck({"intensities": True, "irspec": True, "dielectric_constant": 2.5})
    assert "DIELISO" in deck
    assert "IRSPEC" in deck


def test_dielectric_tensor_still_emitted():
    """DIELTENS + its 9-value row must survive the IRSPEC gate removal."""
    deck = _deck({"intensities": True, "irspec": True,
                  "dielectric_tensor": [2.5, 0, 0, 0, 2.5, 0, 0, 0, 2.5]})
    i = deck.index("DIELTENS")
    assert len(deck[i + 1].split()) == 9


def test_irspec_without_dielectric_warns_but_still_emits(capsys):
    """The user is told the spectrum will be absorption-only (manual 16877).

    The advisory goes to stdout; the deck itself must be unaffected.
    """
    deck = _deck({"intensities": True, "irspec": True})
    out = capsys.readouterr().out
    assert "IRSPEC" in deck
    assert "raw absorption" in out
    assert "skipped" not in out


def test_refrind_dielfun_warn_without_dielectric(capsys):
    """REFRIND/DIELFUN cannot produce output without a dielectric (16877-16880)."""
    _deck({"intensities": True, "irspec": True, "spec_refrind": True})
    assert "REFRIND/DIELFUN" in capsys.readouterr().out


def test_irspec_warns_when_intensities_missing(capsys):
    """IRSPEC needs a prior INTENS (manual 16742)."""
    _deck({"irspec": True})
    out = capsys.readouterr().out
    assert "IRSPEC requires IR intensities" in out


# --------------------------------------------------------------------------
# SCELPHONO placement and expansion-matrix shape
# --------------------------------------------------------------------------

def test_scelphono_precedes_freqcalc():
    """SCELPHONO is a geometry-block keyword; FREQCALC must come last.

    Prevents the regression where every phonon-dispersion deck put
    SCELPHONO inside the FREQCALC block, where it is not a valid
    sub-keyword (manual 15506-15507, 17123-17175).
    """
    deck = _template("phonon_bands")
    assert "SCELPHONO" in deck, "phonon_bands must still request a supercell"
    assert deck.index("SCELPHONO") < deck.index("FREQCALC")


def test_scelphono_not_inside_dispersion_block():
    """SCELPHONO must not reappear after DISPERSION inside FREQCALC."""
    deck = _template("phonon_bands")
    assert deck.count("SCELPHONO") == 1
    assert deck.index("SCELPHONO") < deck.index("DISPERSION")


def test_scelphono_expansion_factors_become_nine_reals():
    """[2,2,2] must expand to a 3x3 matrix by rows, not a bare "2 2 2" line.

    The expansion matrix E is "IDIMxIDIM elements, input by rows: 9 reals
    (3D)" (manual 4822-4826), so a single 3-value record left CRYSTAL
    reading the following keyword lines as matrix data.
    """
    deck = _deck({"dispersion": True, "scelphono": [2, 2, 2]})
    i = deck.index("SCELPHONO")
    rows = deck[i + 1:i + 4]
    assert rows == ["2 0 0", "0 2 0", "0 0 2"]
    assert "2 2 2" not in deck


def test_scelphono_non_uniform_factors_expand_diagonally():
    """Anisotropic expansion factors land on the matrix diagonal."""
    deck = _deck({"dispersion": True, "scelphono": [1, 2, 3]})
    i = deck.index("SCELPHONO")
    assert deck[i + 1:i + 4] == ["1 0 0", "0 2 0", "0 0 3"]


def test_scelphono_layout_matches_real_supercell_deck():
    """Oracle: a real accepted deck lays the matrix out as 3 rows before FREQCALC.

    The reference uses SUPERCEL, SCELPHONO's sibling geometry-block keyword
    (manual sec. 4.21), and is the shape our emitted block must mirror.
    """
    ref = find_data("FREQ/*supercel222.d12")
    lines = [ln.strip() for ln in ref.read_text(errors="ignore").splitlines()]

    i = lines.index("SUPERCEL")
    ref_rows = lines[i + 1:i + 4]
    assert ref_rows == ["2 0 0", "0 2 0", "0 0 2"]
    assert i < lines.index("FREQCALC"), "reference puts the supercell before FREQCALC"

    deck = _deck({"dispersion": True, "scelphono": [2, 2, 2]})
    j = deck.index("SCELPHONO")
    assert deck[j + 1:j + 4] == ref_rows
    assert j < deck.index("FREQCALC")


def test_scelphono_full_matrix_preserved():
    """A 9-element transformation matrix keeps its existing row layout."""
    deck = _deck({"dispersion": True,
                  "scelphono": [1, 1, 0, 1, -1, 0, 0, 0, 2]})
    i = deck.index("SCELPHONO")
    assert deck[i + 1:i + 4] == ["1 1 0", "1 -1 0", "0 0 2"]


def test_no_scelphono_without_dispersion():
    """Supercell emission stays gated on dispersion, as before the move."""
    deck = _deck({"scelphono": [2, 2, 2]})
    assert "SCELPHONO" not in deck


def test_dispersion_still_emits_bands_subkeywords():
    """Moving SCELPHONO out must not disturb the rest of the DISPERSION block."""
    deck = _template("phonon_bands")
    assert "DISPERSION" in deck
    assert "BANDS" in deck
    assert deck.index("DISPERSION") < deck.index("BANDS")


# --------------------------------------------------------------------------
# INTENS / INTRAMAN / RAMSPEC consistency
# --------------------------------------------------------------------------

def test_raman_never_emits_nointens():
    """A Raman-only config must not write NOINTENS then INTRAMAN.

    "the INTRAMAN should be always used together with the INTENS keyword"
    (manual 16226-16227) — the old deck contradicted itself in two
    consecutive lines.
    """
    deck = _deck({"raman": True})
    assert "NOINTENS" not in deck
    assert "INTENS" in deck
    assert deck.index("INTENS") < deck.index("INTRAMAN")


def test_minimal_raman_deck_matches_manual_example():
    """Manual 16230-16236 gives FREQCALC / INTENS / INTRAMAN / INTCPHF / END / END."""
    deck = [ln for ln in _deck({"raman": True}) if ln]
    assert deck[:4] == ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF"]


@pytest.mark.parametrize("name", ["raman_spectrum", "ir_raman"])
def test_raman_templates_pair_intens_with_intraman(name):
    """Every shipped Raman template keeps the INTENS/INTRAMAN pairing."""
    deck = _template(name)
    assert "NOINTENS" not in deck
    assert deck.index("INTENS") < deck.index("INTRAMAN")


def test_ramspec_warns_without_raman_intensities(capsys):
    """RAMSPEC requires INTENS+INTRAMAN (manual 16932-16935).

    Advisory only: the deck is unchanged so an intentional hand-built
    config is never silently dropped.
    """
    deck = _deck({"intensities": True, "ramspec": True})
    assert "RAMSPEC" in deck
    assert "RAMSPEC requires Raman intensities" in capsys.readouterr().out


def test_ramspec_silent_when_raman_requested(capsys):
    """No spurious warning on the normal path."""
    _deck(FREQ_TEMPLATES["raman_spectrum"])
    assert "RAMSPEC requires" not in capsys.readouterr().out


def test_nointens_still_written_for_plain_frequency_run():
    """Templates that genuinely want no intensities must keep NOINTENS."""
    assert "NOINTENS" in _template("phonon_bands")


# --------------------------------------------------------------------------
# 3c composite example config
# --------------------------------------------------------------------------

def test_3c_composite_config_uses_pbeh3c_basis():
    """PBEh-3c is defined on def2-mSVP, not MINIX.

    "EPBEh-3c = EPBEh/def2-mSVP" and def2-mSVP "should be always used when
    performing PBEh-3c calculations" (manual 11765-11812). MINIX belongs to
    HF-3c (manual 11675-11737). The example config paired PBEH3C with
    MINIX, so every deck generated from it used the wrong orbital basis and
    the reported energies were not PBEh-3c energies.
    """
    cfg_path = REPO_ROOT / "Crystal_d12" / "example_configs" / "3c_composite.json"
    cfg = json.loads(cfg_path.read_text())["configuration"]

    assert cfg["functional"] == "PBEH3C"
    assert cfg["basis_set"] == "def2-mSVP"


def test_3c_composite_config_agrees_with_constants_table():
    """The example config must match the in-tree functional->basis mapping."""
    from d12_constants import FUNCTIONAL_CATEGORIES

    cfg_path = REPO_ROOT / "Crystal_d12" / "example_configs" / "3c_composite.json"
    cfg = json.loads(cfg_path.read_text())["configuration"]

    required = FUNCTIONAL_CATEGORIES["3C"]["basis_requirements"]
    assert cfg["basis_set"] == required[cfg["functional"]]
