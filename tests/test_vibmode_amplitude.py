"""FREQ normal-mode amplitudes (1.1.6), checked against real CRYSTAL outputs.

CRYSTAL prints NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES (IN BOHR):
every atom at the classical turning point of the vibrational ground state,
x = sqrt(hbar / (omega mu)) (CRYSTAL23 manual, SCANMODE). Mass-weighting the
printed vectors gives exactly hbar/omega, so they are physical displacements
and "amplitude 1" can draw them as they are. The CLI viewer now works in
Angstrom (it added bohr to Angstrom before), draws a periodic output's
fractional coordinates as Cartesian, and repeats cells (--supercell) for modes
at Gamma. Corpus files are read in place (skipped without the test/ corpus).
"""
import argparse
import re

import numpy as np
import pytest

from conftest import find_data

vib = pytest.importorskip("mace.plotting.engines.vibmode_viewer")

AMU = 1822.888486          # electron masses per atomic mass unit
CM = 219474.6313705        # cm^-1 per hartree
MOLECULE = ("FREQ/DEC_MOLECULE_*freq*.out", "NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES")
CRYSTAL = ("FREQ/3_dia3_opt_rev1_freq_*.out", "COORDINATES OF THE EQUIVALENT ATOMS (FRACTIONAL UNITS)")


def masses(path):
    text = path.read_text(errors="replace")
    i = text.index("ATOMS ISOTOPIC MASS (AMU) FOR FREQUENCY CALCULATION")
    block = text[i:i + 200000].split("\n\n")[1]
    return np.array([float(x) for x in re.findall(r"\d+\s+[A-Z][A-Za-z]?\s+([\d.]+)", block)])


@pytest.mark.parametrize("pattern", [MOLECULE, CRYSTAL], ids=["molecule", "crystal"])
def test_printed_modes_are_zero_point_classical_amplitudes(pattern):
    """sum_i m_i |u_i|^2 = hbar / omega for every real mode (atomic units), to the print's 4 decimals."""
    f = find_data(*pattern)
    m, p = masses(f), vib.Crystal23FreqParser(str(f))
    ratios = []
    for mode in p.modes:
        d = p.displacements.get(mode["mode"])
        if d is None or abs(mode["freq"]) < 50 or len(d) != len(m):
            continue
        ratios.append(float(np.sum(m[:, None] * AMU * np.asarray(d) ** 2)) * abs(mode["freq"]) / CM)
    assert len(ratios) > 5
    assert np.allclose(ratios, 1.0, atol=0.01), (min(ratios), max(ratios))


def test_cli_displacements_are_angstrom_and_unscaled():
    f = find_data(*MOLECULE)
    p = vib.Crystal23FreqParser(str(f))
    a = vib.VibModeAnimator(p)
    mode = next(x["mode"] for x in p.modes if x["freq"] > 50)
    d = a._validated_displacement(mode)
    assert np.allclose(d, np.asarray(p.get_displacement(mode)) * vib.BOHR_TO_ANG)
    assert 0.005 < np.linalg.norm(d, axis=1).max() < 0.3          # tenths of an Angstrom at most: the true size


def test_cli_periodic_coordinates_are_cartesian():
    """Fractional coordinates were drawn as Angstrom (atoms 0.1 A apart); now they sit in the cell."""
    f = find_data(*CRYSTAL)
    p = vib.Crystal23FreqParser(str(f))
    assert p.periodic_axes() == [0, 1, 2] and p.get_lattice() is not None
    a = vib.VibModeAnimator(p)
    dist = np.linalg.norm(a.coords[:, None] - a.coords[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() > 1.0
    # the raw parse is unchanged (other callers read it)
    assert np.allclose(p.get_coordinates() + np.mean([[x[1], x[2], x[3]] for x in p.atoms], axis=0),
                       [[x[1], x[2], x[3]] for x in p.atoms])


def test_cli_supercell_repeats_atoms_and_modes():
    f = find_data(*CRYSTAL)
    p = vib.Crystal23FreqParser(str(f))
    a = vib.VibModeAnimator(p, supercell=(2, 2, 1))
    assert len(a.images) == 4 and a.n_atoms == 4 * a.n_base
    mode = next(x["mode"] for x in p.modes if x["freq"] > 50)
    d = a._validated_displacement(mode)
    assert d.shape == (a.n_atoms, 3)
    assert np.allclose(d[:a.n_base], d[a.n_base:2 * a.n_base])     # Gamma: every image moves as its atom
    # a molecule has nothing to repeat; modes away from Gamma would need phases
    with pytest.raises(ValueError, match="periodic"):
        vib.VibModeAnimator(vib.Crystal23FreqParser(str(find_data(*MOLECULE))), supercell=(2, 1, 1))
    p._frame = (p._frame[0], p._frame[1], True)
    with pytest.raises(ValueError, match="Gamma"):
        vib.VibModeAnimator(p, supercell=(2, 1, 1))


def test_handler_passes_supercell(monkeypatch, tmp_path):
    from mace.plotting.handlers import freq as freq_h
    ns = argparse.Namespace(mode=3, all_modes=False, list_modes=False, amplitude=1.0, gif=False, gif_fps=20,
                            static=False, normalize=False, frames=None, freq_supercell=[2, 2, 1])
    cfg = freq_h._freq_config_from_args(ns)
    assert cfg["supercell"] == [2, 2, 1]
    ns.freq_supercell = None
    assert freq_h._freq_config_from_args(ns)["supercell"] == [1, 1, 1]
    seen = []
    monkeypatch.setattr(freq_h, "_run_engine", lambda argv: seen.append(argv) or True)
    freq_h.plot_freq([str(find_data(*CRYSTAL))], {**cfg, "mode": 5}, output_dir=str(tmp_path))
    i = seen[0].index("--supercell")
    assert seen[0][i + 1:i + 4] == ["2", "2", "1"]


def test_gui_freq_section_says_classical():
    """freq.py tells the page whether the output printed classical amplitudes (the Physical scale's meaning)."""
    from mace.gui import freq as gf
    f = find_data(*MOLECULE)
    parsed = gf.parse_job(path=str(f), start=0, end=f.stat().st_size)
    assert parsed["classical"] is True
    k = next(m["mode"] for m in parsed["modes"] if m["freq"] > 50)
    row = gf.mode_payload(parsed, k)
    assert row["classical"] is True and row["units"] == "angstrom"
