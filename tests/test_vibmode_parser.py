"""Phase 1: correctness fixes for the CRYSTAL23 FREQ vibmode parser.

Fixes (from MACE_PLOTTING_INTEGRATION_PLAN.md, combined per the plan note):
  2.1 freq-format  — recover trailing <6-column eigenvector blocks; parse
                     imaginary (negative cm-1) frequencies and tag them.
  2.2 vib-degenerate — expand degenerate MODES-table ranges (e.g. ``7-  9``)
                     into one entry per mode; map eigenvector columns to modes
                     by positional cursor (not freq proximity) so degenerate
                     partners keep distinct displacements; carry irrep through.
  2.5 element-ecp  — heavy-element symbols (PB/TI/AG/SE/...) present in the
                     four element dicts so crystalline FREQ files don't render
                     as DEFAULT.

Validated against the real corpus (per project policy). TiPbO3_mp-19845 is a
cubic PM3M perovskite: 5 atoms / 15 modes in 5 triply-degenerate ranges with an
imaginary soft mode and a 6+6+3 eigenvector layout — it exercises all three
fixes at once.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import find_data

# vibmode_viewer.py lives in the (not-yet-wired) plotting scripts dir.
_VIB_DIR = Path(__file__).resolve().parent.parent / "test" / "AddedPlottingFunctionalty"
if str(_VIB_DIR) not in sys.path:
    sys.path.insert(0, str(_VIB_DIR))

vib = pytest.importorskip("vibmode_viewer")

# Cubic perovskite: 5 atoms, 15 modes, degenerate ranges, imaginary soft mode,
# 6+6+3 eigenvector blocks.
_TIPBO3_CUBIC = "FREQ/TiPbO3_mp-19845_sg221_*optimized_freq_*optimized.out"
# Tetragonal perovskite with Pb/Ti (element coverage).
_TIPBO3_TET = "FREQ/TiPbO3_mp-20459_sg99_*optimized_freq_*optimized.out"
# Molecular conformer with an imaginary mode (-12.70 cm-1).
_SULFOLANE = "FREQ/1LiFSI-1Sulfolane-conf3_*freq_*temp.out"


def _parse(pattern, must_contain=None):
    return vib.Crystal23FreqParser(str(find_data(pattern, must_contain)))


# ---------------- 2.2: degenerate range expansion ----------------

def test_degenerate_ranges_expand_to_full_mode_count():
    """Cubic TiPbO3 has 15 modes printed as 5 degenerate ranges; all 15 must
    appear as individual entries (pre-fix only 4 survived)."""
    p = _parse(_TIPBO3_CUBIC)
    assert p.n_atoms == 5
    assert len(p.modes) == 3 * p.n_atoms == 15
    assert [m["mode"] for m in p.modes] == list(range(1, 16))


def test_degenerate_partners_keep_distinct_eigenvectors():
    """The triply-degenerate 116.07 cm-1 set (modes 7,8,9) are three distinct
    Cartesian polarizations — pre-fix they collapsed onto one mode."""
    p = _parse(_TIPBO3_CUBIC)
    d7, d8, d9 = (p.get_displacement(m) for m in (7, 8, 9))
    for d in (d7, d8, d9):
        assert d is not None and d.shape == (5, 3)
    assert not np.allclose(d7, d8)
    assert not np.allclose(d7, d9)
    assert not np.allclose(d8, d9)


def test_irrep_labels_carried_through():
    p = _parse(_TIPBO3_CUBIC)
    irreps = {m.get("irrep") for m in p.modes}
    assert "F1u" in irreps and "F2u" in irreps


# ---------------- 2.1a: trailing <6-column block recovery ----------------

def test_trailing_block_eigenvectors_recovered():
    """The final 3-column block (modes 13,14,15 at 508.98 cm-1) is dropped by
    the fixed-6-column regex; after the fix all three have displacements."""
    p = _parse(_TIPBO3_CUBIC)
    for m in (13, 14, 15):
        d = p.get_displacement(m)
        assert d is not None and d.shape == (5, 3), f"mode {m} displacement missing"


def test_every_mode_has_a_displacement():
    p = _parse(_TIPBO3_CUBIC)
    assert len(p.displacements) == len(p.modes) == 15


# ---------------- 2.1c: imaginary frequencies ----------------

def test_imaginary_modes_are_parsed_and_tagged():
    """Cubic TiPbO3's soft mode (-164.23 cm-1) must be captured and flagged;
    pre-fix the negative-frequency table rows were rejected outright."""
    p = _parse(_TIPBO3_CUBIC)
    imaginary = [m for m in p.modes if m["freq"] < -1.0]
    assert len(imaginary) == 3  # the F1u soft-mode triplet
    assert all(m.get("imaginary") is True for m in imaginary)


def test_molecular_imaginary_mode_parsed():
    p = _parse(_SULFOLANE)
    assert any(m["freq"] < -1.0 and m.get("imaginary") for m in p.modes)


# ---------------- 2.5: heavy-element coverage ----------------

@pytest.mark.parametrize("sym", ["PB", "TI", "AG", "SE"])
def test_heavy_elements_present_in_all_dicts(sym):
    assert sym in vib.ELEMENT_COLORS
    assert sym in vib.ELEMENT_OUTLINE_COLORS
    assert sym in vib.COVALENT_RADII
    assert sym in vib.DISPLAY_SIZES


def test_heavy_element_color_is_not_default():
    p = _parse(_TIPBO3_TET)
    assert {"PB", "TI"} <= set(p.get_elements())
    assert vib.ELEMENT_COLORS["PB"] != vib.ELEMENT_COLORS["DEFAULT"]
    assert vib.ELEMENT_COLORS["TI"] != vib.ELEMENT_COLORS["DEFAULT"]


# ---------------- regression: molecules still parse ----------------

def test_molecular_file_still_parses():
    p = _parse("FREQ/1LiFSI-1DEC-conf3_*freq_*temp.out")
    assert p.n_atoms == 28
    # most modes animatable (was 79/80 pre-fix; must not regress)
    assert len(p.displacements) >= 80


# ---------------- whole-corpus regression net ----------------

def test_all_freq_outputs_parse_without_error():
    """Every real FREQ .out must parse cleanly, and within a file all
    displacement arrays must be the same (k, 3) shape (internal consistency).
    Broad guard for the combined regex/encoding change across 0D molecules and
    3D crystals (incl. degenerate, imaginary, heavy-element, ECP cases)."""
    freq_dir = Path(__file__).resolve().parent.parent / "test" / "FREQ"
    if not freq_dir.is_dir():
        pytest.skip("test/FREQ corpus not present")
    outs = sorted(freq_dir.glob("*.out"))
    if not outs:
        pytest.skip("no FREQ .out files")

    failures = []
    parsed_with_modes = 0
    for f in outs:
        try:
            p = vib.Crystal23FreqParser(str(f))
        except Exception as e:  # parsing must never raise
            failures.append((f.name, repr(e)))
            continue
        if p.modes:
            parsed_with_modes += 1
        shapes = {d.shape for d in p.displacements.values()}
        if len(shapes) > 1:
            failures.append((f.name, f"ragged displacement shapes {shapes}"))
        for d in p.displacements.values():
            if d.ndim != 2 or d.shape[1] != 3:
                failures.append((f.name, f"bad displacement shape {d.shape}"))
                break

    assert not failures, f"{len(failures)}/{len(outs)} FREQ files problematic: {failures[:3]}"
    assert parsed_with_modes >= len(outs) // 2  # most files have a modes table


# ---------------- render guard: refuse mismatched displacement ----------------

def test_show_mode_refuses_displacement_atomcount_mismatch():
    """On supercell / high-symmetry FREQ outputs the parsed coordinates are the
    asymmetric/primitive cell while eigenvectors span the full cell, so
    displacement.shape[0] != n_atoms. The real render entry point (show_mode,
    used by `--mode`) must refuse with a clear message rather than failing in a
    raw numpy broadcast (which is also a ValueError, hence the message match)."""
    p = _parse("FREQ/1_dia_opt_rev1_freq_*supercel222.out")
    anim = vib.VibModeAnimator(p)
    bad = [m for m, d in p.displacements.items() if d.shape[0] != anim.n_atoms]
    if not bad:
        pytest.skip("no atom-count mismatch in this corpus build")
    with pytest.raises(ValueError, match="refusing to render"):
        anim.show_mode(bad[0])


def test_all_modes_html_skips_fully_mismatched_file(tmp_path):
    """`--all` on a file where every mode mismatches the parsed structure must
    skip gracefully (return None), not crash on an empty mode list."""
    p = _parse("FREQ/1_dia_opt_rev1_freq_*supercel222.out")
    anim = vib.VibModeAnimator(p)
    if all(d.shape[0] == anim.n_atoms for d in p.displacements.values()):
        pytest.skip("no mismatch in this corpus build")
    assert anim.create_all_modes_html(str(tmp_path / "all.html")) is None


def test_create_molecule_traces_guards_mismatch():
    """Defense-in-depth: the trace builder itself also refuses a mismatch."""
    p = _parse("FREQ/1_dia_opt_rev1_freq_*supercel222.out")
    anim = vib.VibModeAnimator(p)
    bad = [d for d in p.displacements.values() if d.shape[0] != anim.n_atoms]
    if not bad:
        pytest.skip("no atom-count mismatch in this corpus build")
    with pytest.raises(ValueError, match="refusing to render"):
        anim._create_molecule_traces(anim.coords, displacement=bad[0], show_arrows=True)
