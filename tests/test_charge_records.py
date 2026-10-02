"""A derived deck keeps its parent's electron-count records (CHEMOD, CHARGED, DOPING).

Real case: the group's Li/FSI/EC clusters (HSEsol-3c, BASISSET SOLDEF2MSVP) set
Li+ and FSI- with CHEMOD and the net charge with DOPING (DOPING -1 for Li+ with
no FSI). opt2d12 wrote their SP and FREQ children without either record, so the
children silently ran a different charge state.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))
from charge_records import carry_charge_records, extract_charge_records  # noqa: E402

PARENT = """motif_Li003_FSI0_EC6
MOLECULE
1
3
3 0.0 0.0 0.0 Li
8 2.0 0.0 0.0 O
6 3.2 0.0 0.0 C
END
BASISSET
SOLDEF2MSVP
CHEMOD
1
1     2.0 0.0 0.0 0.0 0.0
DFT
HSESOL3C
XLGRID
ENDDFT
DOPING
-1
TOLINTEG
7 7 7 7 14
SCFDIR
PPAN
END
"""

CHILD = """motif_Li003_FSI0_EC6_sp
MOLECULE
1
3
3 0.0 0.0 0.1 Li
8 2.0 0.0 0.1 O
6 3.2 0.0 0.1 C
END
BASISSET
SOLDEF2MSVP
DFT
HSESOL3C
XLGRID
ENDDFT
TOLINTEG
7 7 7 7 14
SCFDIR
PPAN
END
"""


def test_records_are_read_from_the_parent():
    rec = extract_charge_records(PARENT)
    assert rec["CHEMOD"] == ["CHEMOD", "1", "1     2.0 0.0 0.0 0.0 0.0"]
    assert rec["DOPING"] == ["DOPING", "-1"]
    assert rec["CHARGED"] == []


def test_the_title_is_never_a_record():
    rec = extract_charge_records("DOPING\nMOLECULE\n1\n0\nEND\n")
    assert rec["DOPING"] == []


def test_child_gets_chemod_after_the_basis_name_and_doping_before_the_last_end():
    out, notes = carry_charge_records(CHILD, PARENT)
    lines = out.splitlines()
    b = lines.index("SOLDEF2MSVP")
    assert lines[b + 1:b + 4] == ["CHEMOD", "1", "1     2.0 0.0 0.0 0.0 0.0"]
    assert lines[-3:] == ["DOPING", "-1", "END"]
    assert out.endswith("\n")
    assert not any(n.startswith("WARNING") for n in notes)


def test_explicit_basis_gets_records_after_99_0():
    parent = PARENT.replace("BASISSET\nSOLDEF2MSVP\n", "3 2\n0 0 1 2.0 1.0\n 1.0 1.0\n99 0\nCHARGED\n")
    child = CHILD.replace("BASISSET\nSOLDEF2MSVP\n", "3 2\n0 0 1 2.0 1.0\n 1.0 1.0\n99 0\nEND\n")
    out, _ = carry_charge_records(child, parent)
    lines = out.splitlines()
    i = lines.index("99 0")
    assert lines[i + 1:i + 5] == ["CHEMOD", "1", "1     2.0 0.0 0.0 0.0 0.0", "CHARGED"]


def test_a_different_atom_count_keeps_doping_but_not_chemod():
    child = CHILD.replace("3\n3 0.0 0.0 0.1 Li", "4\n3 0.0 0.0 0.1 Li\n1 5.0 0.0 0.0 H")
    out, notes = carry_charge_records(child, PARENT)
    assert "CHEMOD" not in out.splitlines()
    assert out.splitlines()[-3:] == ["DOPING", "-1", "END"]
    assert any(n.startswith("WARNING") and "CHEMOD was NOT copied" in n for n in notes)


def test_records_the_child_already_has_are_left_alone():
    child = CHILD.replace("ENDDFT\n", "ENDDFT\nDOPING\n1\n")
    out, _ = carry_charge_records(child, PARENT)
    assert out.splitlines().count("DOPING") == 1
    assert "1" in out.splitlines()[out.splitlines().index("DOPING") + 1]


def test_a_neutral_parent_leaves_the_child_byte_identical():
    parent = PARENT.replace("CHEMOD\n1\n1     2.0 0.0 0.0 0.0 0.0\n", "").replace("DOPING\n-1\n", "")
    out, notes = carry_charge_records(CHILD, parent)
    assert out == CHILD and notes == []


SP_PARENT = TEST_DATA / "SP" / "1LiFSI-1DEC-conf2_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"


@pytest.mark.skipif(not SP_PARENT.with_suffix(".out").exists(), reason="needs the test/ corpus")
@pytest.mark.parametrize("calc_type", ["SP", "FREQ"])
def test_opt2d12_children_of_a_charged_cluster_keep_its_records(tmp_path, calc_type):
    """The real invocation path: mace opt2d12 on a parent deck carrying CHEMOD + DOPING."""
    shutil.copy(SP_PARENT.with_suffix(".out"), tmp_path / "parent.out")
    deck = SP_PARENT.with_suffix(".d12").read_text()
    deck = deck.replace("SOLDEF2MSVP\n", "SOLDEF2MSVP\nCHEMOD\n1\n10     2.0 0.0 0.0 0.0 0.0\n", 1)
    deck = deck.replace("ENDDFT\n", "ENDDFT\nDOPING\n-1\n", 1)
    (tmp_path / "parent.d12").write_text(deck)
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", str(tmp_path / "parent.out"),
         "--calc-type", calc_type, "--non-interactive", "--output-dir", str(tmp_path / "out")],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    child = next((tmp_path / "out").glob("*.d12")).read_text().splitlines()
    i = child.index("CHEMOD")
    assert child[i + 1:i + 3] == ["1", "10     2.0 0.0 0.0 0.0 0.0"]
    j = child.index("DOPING")
    assert child[j + 1].strip() == "-1"
