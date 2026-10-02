"""`mace phonopy collect --nac-from FREQ.out` (needs phonopy).

The FREQ output is phonopy's NaCl-CRYSTAL crystal.o (CRYSTAL17, Gamma FREQCALC
with INTENS + INTCPHF), trimmed to the blocks phonopy-crystal-born reads. The
CRYSTAL23 outputs in the group's corpus print the same headers ("FR.(eV) COMP.
ALPHA(Re,Im) EPSILON(Re,Im)", "ATOMIC BORN CHARGE TENSOR") and the same
asymmetric-unit list, so this excerpt stands for both.

phonopy-crystal-born writes Born tensors for CRYSTAL's asymmetric ("T") atoms
and phonopy spreads them over ITS idea of the asymmetric unit, in cartesian
axes - so each way the two can disagree must be refused, not silently used.
"""
import json
import shutil

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from conftest import REPO_ROOT  # noqa: E402

from mace.phonopy_bridge import cli  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "NaCl"


@pytest.fixture
def setdir(tmp_path, monkeypatch):
    for f in ("parent.out", "parent.d12", "crystal.o", "crystal.d12"):
        shutil.copy(FIX / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "parent.out", "--dim", "4 4 4"]) == 0
    d = tmp_path / "phonopy_parent_444"
    man = json.loads((d / "mace_phonopy.json").read_text())
    for i, job in enumerate(man["jobs"], start=1):
        shutil.copy(FIX / f"supercell-{i:03d}.o", d / f"{job}.out")
    return d


def _freq_cm(d, q):
    ph = phonopy.load(str(d / "phonopy_params.yaml"))
    ph.run_qpoints([q])
    return ph.qpoints.frequencies[0] * 33.35641


def test_born_written_and_lo_to_split(setdir):
    d = setdir
    assert cli.main(["collect", str(d), "--nac-from", "crystal.o", "--band", "none"]) == 0
    born = (d / "BORN").read_text().split("\n")
    assert born[0] == "default"
    assert [float(x) for x in born[2].split()][0] == pytest.approx(1.0238)     # Na Z*
    f = _freq_cm(d, [0.001, 0.0, 0.001])
    assert f[3] == pytest.approx(204.24, abs=0.05)                         # TO
    assert f[5] > f[3] + 50                                                 # LO pushed up
    # Without NAC there is no splitting.
    d2 = d.parent / "nonac"
    shutil.copytree(d, d2)
    (d2 / "BORN").unlink()
    assert cli.main(["collect", str(d2), "--band", "none"]) == 0
    g = _freq_cm(d2, [0.001, 0.0, 0.001])
    assert g[5] - g[3] < 0.5


def _refused(d, text, capsys, name="bad.o", d12=True):
    p = d.parent / name
    p.write_text(text)
    if d12:
        shutil.copy(d.parent / "crystal.d12", p.with_suffix(".d12"))
    rc = cli.main(["collect", str(d), "--nac-from", str(p), "--band", "none"])
    err = capsys.readouterr().err
    assert rc == 1, err
    assert "NAC refused" in err
    assert not (d / "FORCE_SETS").exists() or not (d / "BORN").exists()
    return err


def _good():
    return (FIX / "crystal.o").read_text()


def test_no_intcphf_refused(setdir, capsys):
    t = _good()
    t = t[:t.index(" ATOMIC BORN CHARGE TENSOR")]
    assert "INTENS and INTCPHF" in _refused(setdir, t, capsys)


def test_atom_count_mismatch_refused(setdir, capsys):
    t = _good().replace("ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2",
                        "ATOMS IN THE ASYMMETRIC UNIT    3 - ATOMS IN THE UNIT CELL:    3")
    row = [ln for ln in t.splitlines() if ln.strip().startswith("2 T  17 CL")][0]
    t = t.replace(row + "\n", row + "\n      3 T  11 NA    2.500000000000E-01  2.500000000000E-01  "
                  "2.500000000000E-01\n", 1)
    assert "do not match the phonopy unit cell" in _refused(setdir, t, capsys)


def test_rotated_lattice_refused(setdir, capsys):
    t = _good()
    i = t.index("DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)")
    lines = t[i:].splitlines()
    # swap the x and y columns of the three lattice vectors: same cell, other orientation
    new = []
    for ln in lines[2:5]:
        x, y, z = ln.split()
        new.append(f"   {y}   {x}   {z}")
    t = t[:i] + "\n".join(lines[:2] + new + lines[5:]) + "\n"
    assert "orientation" in _refused(setdir, t, capsys)


def test_asymmetric_unit_mismatch_refused(setdir, capsys):
    row = [ln for ln in _good().splitlines() if ln.strip().startswith("2 T  17 CL")][0]
    t = _good().replace(row, row.replace(" T ", " F ", 1))
    assert "asymmetric unit" in _refused(setdir, t, capsys)


def test_empty_born_refused(setdir, capsys):
    t = _good().replace("FR.(eV) COMP.   ALPHA(Re,Im)", "FR.(eV) COMP.  ALPHA(Re,Im)")
    assert "wrote nothing" in _refused(setdir, t, capsys)


def test_different_method_refused(setdir, capsys):
    p = setdir.parent / "other.o"
    p.write_text(_good())
    p.with_suffix(".d12").write_text(
        (setdir.parent / "crystal.d12").read_text().replace("PBE0", "B3LYP"))
    rc = cli.main(["collect", str(setdir), "--nac-from", str(p), "--band", "none"])
    assert rc == 1 and "functional/basis records differ" in capsys.readouterr().err


def test_scelphono_run_refused(setdir, capsys):
    t = _good().replace(" CRYSTAL CALCULATION", " CRYSTAL CALCULATION\nSCELPHONO", 1)
    assert "SCELPHONO" in _refused(setdir, t, capsys)


def test_scelphono_deck_refused(setdir, capsys):
    """CRYSTAL does not echo its input, so the deck next to the output is read too."""
    p = setdir.parent / "scel.o"
    p.write_text(_good())
    p.with_suffix(".d12").write_text(
        (setdir.parent / "crystal.d12").read_text().replace("FREQCALC\n", "FREQCALC\nSCELPHONO\n2 0 0\n0 2 0\n0 0 2\nDISPERSI\n"))
    rc = cli.main(["collect", str(setdir), "--nac-from", str(p), "--band", "none"])
    assert rc == 1 and "SCELPHONO" in capsys.readouterr().err
