"""Decks derived from a GHOSTS parent keep its ghost atoms.

GHOSTS (basis-set input, manual p.82) turns atoms into ghosts: their basis
functions stay, their nucleus and electrons go (a vacancy with its basis, as
`mace defects --ghost` writes). Real output: an MgO V_O ghost run made on
HPCC with `mace defects` (CRYSTAL23, SUPERCON 2x2x2, GHOSTS on atom 1), stored as a
gzipped excerpt.

* opt2d12 keeps the record (tests/test_opt2d12_supercell_parent.py).
* opt2d3 DOSS: CRYSTAL lists the ghost's functions under its element ("1 O"
  in LOCAL ATOMIC FUNCTIONS; "1 XX" only in the neighbour table), so the
  element projection "#O" took in the ghost's empty functions. The ghost is
  now projected on its own ("#O_GHOST") and "#O" holds the 31 real O atoms.
"""
import gzip
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

DATA = REPO_ROOT / "tests" / "data" / "supercell_parents"
CONFIG = REPO_ROOT / "Crystal_d3" / "example_configs" / "doss_element_orbital_auto.json"


def _projections(tmp_path):
    shutil.copy(DATA / "rg_q0.d12", tmp_path)
    with gzip.open(DATA / "rg_q0.out.gz", "rt") as src:
        (tmp_path / "rg_q0.out").write_text(src.read())
    (tmp_path / "rg_q0.f9").write_bytes(b"")
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "--input", "rg_q0.out",
         "--calc-type", "DOSS", "--config-file", str(CONFIG), "--output-dir", str(tmp_path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    proj = {}
    for line in (tmp_path / "rg_q0_doss.d3").read_text().splitlines():
        if "#" in line:
            nums, label = line.split("#")
            ints = [int(x) for x in nums.split()]
            assert ints[0] == len(ints) - 1
            proj[label.strip()] = ints[1:]
    return proj


def test_doss_projects_the_ghost_apart_from_its_element(tmp_path):
    proj = _projections(tmp_path)
    # 18 functions per O (POB-TZVP-REV2), 19 per Mg; NUMBER OF AO 1184.
    assert proj["O_GHOST all"] == list(range(1, 19))
    assert len(proj["O all"]) == 31 * 18 and min(proj["O all"]) == 19
    assert len(proj["MG all"]) == 32 * 19
    every = proj["O_GHOST all"] + proj["O all"] + proj["MG all"]
    assert sorted(every) == list(range(1, 1185))


def test_ghost_labels_are_read_from_the_output():
    from d3_interactive import ghost_atom_labels
    text = gzip.open(DATA / "rg_q0.out.gz", "rt").read()
    assert ghost_atom_labels(text) == {1}
    assert ghost_atom_labels(" * ATOMS TRANSFORMED INTO GHOSTS :   3\n"
                             "   1(  8)  17(  8)\n  40( 12)\n SYMMETRY MAY BE BROKEN\n") == {1, 17, 40}
    assert ghost_atom_labels("no ghosts here") == set()


def test_d3_reads_the_electron_count(tmp_path):
    """CRYSTAL prints "N. OF ELECTRONS PER CELL", which the D3 reader's
    "PER UNIT CELL" pattern never matched (n_electrons was always 0)."""
    from CRYSTALOptToD3 import D3Generator
    out = tmp_path / "rg_q0.out"
    out.write_text(gzip.open(DATA / "rg_q0.out.gz", "rt").read())
    assert D3Generator(str(out), "DOSS").structure_info["n_electrons"] == 632
