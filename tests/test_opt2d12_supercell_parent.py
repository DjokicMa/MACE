"""opt2d12 children of a SUPERCEL / SUPERCON parent describe the parent's supercell.

After SUPERCEL or SUPERCON (manual sec. 4.21, pp.74-77) CRYSTAL computes the
supercell, keeping only the operators the supercell keeps. The space group it
printed, and the parent deck's group record, are the primitive host's. The
child used to be that group with the supercell's lattice and asymmetric
atoms: `opt2d12 --calc-type SP` on the corpus diamond 2x2x2 SUPERCEL FREQ wrote
SG 227, a = 5.01312 and 5 atoms (not diamond). The child now lists every atom
of the supercell in P1 - the cell the parent computed, atoms in the same order.

Real outputs: the corpus diamond FREQ (skipped without the corpus) and two MgO
defect runs made on HPCC with `mace defects` (CRYSTAL23, SUPERCON 2x2x2 of the
conventional cell; one with ATOMSUBS + DOPING, one with GHOSTS), stored as
gzipped excerpts without run-host data (tests/test_supercell_parent_excerpts.py).
"""
import gzip
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
DATA = REPO_ROOT / "tests" / "data" / "supercell_parents"
DIAMOND = "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222"


def _stage(tmp_path, stem):
    shutil.copy(DATA / f"{stem}.d12", tmp_path)
    with gzip.open(DATA / f"{stem}.out.gz", "rt") as src:
        (tmp_path / f"{stem}.out").write_text(src.read())
    return tmp_path


def _opt2d12(cwd, stem, calc_type):
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{stem}.out",
         "--calc-type", calc_type, "--non-interactive"],
        cwd=cwd, input="", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    decks = sorted(cwd.glob(f"{stem}_{calc_type.lower()}_*_optimized.d12"))
    assert len(decks) == 1, sorted(p.name for p in cwd.iterdir())
    return decks[0].read_text().splitlines(), result.stdout + result.stderr


def _geometry(lines):
    """(group, cell record, atom lines) of a CRYSTAL deck."""
    assert lines[1] == "CRYSTAL"
    n = int(lines[5].split()[0])
    return int(lines[3]), lines[4].split(), lines[6:6 + n]


@pytest.mark.parametrize("calc_type", ["SP", "OPT", "FREQ"])
def test_defect_supercell_child_is_the_supercell_in_p1(tmp_path, calc_type):
    _stage(tmp_path, "rs_qm1")
    lines, log = _opt2d12(tmp_path, "rs_qm1", calc_type)
    group, cell, atoms = _geometry(lines)
    assert group == 1
    assert [float(x) for x in cell] == [8.424, 8.424, 8.424, 90.0, 90.0, 90.0]
    assert len(atoms) == 64
    # ATOMSUBS turned atom 1 (O) into N; the child keeps it, once.
    assert atoms[0].split()[0] == "7" and atoms[0].split()[-1] == "N"
    assert sum(a.split()[-1] == "N" for a in atoms) == 1
    assert sum(a.split()[-1] == "O" for a in atoms) == 31
    assert sum(a.split()[-1] == "Mg" for a in atoms) == 32
    # The cell's charge is the parent's: the same cell, so no "check it" warning.
    assert "DOPING" in lines and lines[lines.index("DOPING") + 1].strip() == "1.0"
    assert "copied unchanged into a deck" not in log


@pytest.mark.parametrize("calc_type", ["SP", "OPT"])
def test_corpus_supercel_child_is_diamond_supercell(tmp_path, calc_type):
    src = TEST_DATA / "FREQ" / f"{DIAMOND}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    lines, _ = _opt2d12(tmp_path, DIAMOND, calc_type)
    group, cell, atoms = _geometry(lines)
    assert group == 1
    assert [round(float(x), 5) for x in cell] == [5.01312, 5.01312, 5.01312, 60.0, 60.0, 60.0]
    assert len(atoms) == 16 and all(a.split()[0] == "6" for a in atoms)


@pytest.mark.parametrize("calc_type", ["SP", "OPT"])
def test_ghost_parent_child_keeps_ghosts_on_the_same_atom(tmp_path, calc_type):
    """GHOSTS (basis-set input, p.82) names atom 1 of the supercell; the P1
    child lists the supercell's atoms in CRYSTAL's order, so atom 1 is the
    same O and the record is kept as the parent wrote it."""
    _stage(tmp_path, "rg_q0")
    lines, log = _opt2d12(tmp_path, "rg_q0", calc_type)
    group, _cell, atoms = _geometry(lines)
    assert group == 1 and len(atoms) == 64 and atoms[0].split()[-1] == "O"
    at = lines.index("GHOSTS")
    assert lines[at:at + 3] == ["GHOSTS", "1", "1"]
    assert lines[at - 2:at] == ["BASISSET", "POB-TZVP-REV2"]
    assert "kept the parent's GHOSTS" in log
