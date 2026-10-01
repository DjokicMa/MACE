"""Chained SCF steps restart from the predecessor's density matrix - opt-in.

With ``execution_settings.guessp_restart: true`` in the workflow plan, the
engine copies the predecessor's ``.f9`` to ``<job>.f20`` in the follow-up's
directory (the file submitcrystal23.sh stages as fort.20) and adds GUESSP to
the follow-up deck - only when both decks have the same symmetry, atom list,
basis set and spin treatment (CRYSTAL23 manual, GUESSP, pp. 114-115: "same
symmetry, and same number of atoms, basis functions and shells ... The program
does not check the 1:1 old-new correspondence").

Without the option the engine writes exactly what it wrote before.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.workflow import guessp_chain
from mace.workflow.engine import WorkflowEngine

WF_ID = "workflow_20260930_120000"
LOW_DIM = REPO_ROOT / "tests" / "data" / "low_dim_groups"

EXT_BASIS = """8 2
0 0 6 2.0 1.0
  27032.382631      0.00021726302465
  4052.3871392      0.00168386621990
0 2 1 0.0 1.0
  0.3565870100      1.00000000000000
22 2
0 0 1 2.0 1.0
  211575.690250      0.00023318151011
0 3 1 0.0 1.0
  0.33183757000      1.00000000000000
99 0
END
"""

HAM = "DFT\nSPIN\nB3LYP-D3\nXLGRID\nENDDFT\n"
SCF = ("TOLINTEG\n7 7 7 7 14\nTOLDEE\n7\nSHRINK\n8 16\nSCFDIR\nMAXCYCLE\n800\n"
       "FMIXING\n30\nDIIS\nHISTDIIS\n100\nPPAN\nEND\n")


def _geom(a="3.94649838", atoms=("22 0.5 0.5 0.5", "8 0.0 0.5 0.5"), sg="221",
          block=""):
    return (f"CRYSTAL\n0 0 0\n{sg}\n{a}\n{len(atoms)}\n" + "\n".join(atoms) + "\n"
            + block + "END\n")


OPTGEOM = "OPTGEOM\nFULLOPTG\nMAXCYCLE\n800\nENDOPT\n"
FREQCALC = "FREQCALC\nNUMDERIV\n2\nEND\n"

OPT_DECK = "pto_opt\n" + _geom(block=OPTGEOM) + EXT_BASIS + HAM + SCF
# What opt2d12 writes for the follow-ups: the optimised lattice parameter,
# the same symmetry, atoms and basis.
SP_DECK = "pto_sp\n" + _geom(a="3.95012345") + EXT_BASIS + HAM + SCF
FREQ_DECK = "pto_freq\n" + _geom(a="3.95012345", block=FREQCALC) + EXT_BASIS + HAM + SCF
OPT2_DECK = "pto_opt2\n" + _geom(a="3.95012345", block=OPTGEOM) + EXT_BASIS + HAM + SCF

F9 = b"\x00\x01DENSITY-MATRIX-OF-THE-PARENT\x02"


class _FakeDB:
    def __init__(self, calc):
        self.calc = calc
        self.created = []

    def get_calculation(self, cid):
        return self.calc

    def get_calculations_by_status(self, *a, **k):
        return []

    def create_calculation(self, **kw):
        self.created.append(kw)
        return f"calc_{len(self.created)}"

    def update_calculation_status(self, *a, **k):
        pass


def _plan(tmp_path, execution_settings):
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir(exist_ok=True)
    plan = {"workflow_id": WF_ID,
            "workflow_sequence": ["OPT", "SP", "FREQ", "OPT2", "SP2"],
            "step_configurations": {}}
    if execution_settings is not None:
        plan["execution_settings"] = execution_settings
    (cfg / f"workflow_plan_{WF_ID.replace('workflow_', '')}.json").write_text(json.dumps(plan))


def _engine(tmp_path, source_type="OPT", source_deck=OPT_DECK, f9=F9,
            execution_settings=None, source_out="fake CRYSTAL output\n"):
    src = tmp_path / "parent"
    src.mkdir()
    (src / "parent.d12").write_text(source_deck)
    (src / "parent.out").write_text(source_out)
    if f9 is not None:
        (src / "parent.f9").write_bytes(f9)
    _plan(tmp_path, execution_settings)
    calc = {"calc_id": "c1", "material_id": "pto", "status": "completed",
            "calc_type": source_type, "output_file": str(src / "parent.out"),
            "input_file": str(src / "parent.d12"), "work_dir": str(src),
            "settings_json": json.dumps({"workflow_id": WF_ID, "workflow_step": 1})}
    eng = WorkflowEngine(db_path=str(tmp_path / "m.db"), base_work_dir=str(tmp_path),
                         auto_submit=False)
    eng.db = _FakeDB(calc)
    return eng


def _fake_opt2d12(eng, deck_text):
    """Stand-in for CRYSTALOptToD12: write the follow-up deck it would write."""
    def run(script_path, work_dir, args=None, input_data=None):
        target = next(a for a in (args or []) if a.endswith("_temp_config.json"))
        base = json.loads(Path(target).read_text())["calculation_type"]
        (Path(work_dir) / f"pto_{base}_generated.d12").write_text(deck_text)
        return True, "", ""
    eng.run_script_in_isolated_directory = run


def _generate(eng, target, deck_text):
    _fake_opt2d12(eng, deck_text)
    calc_id = eng.generate_numbered_calculation("c1", target)
    assert calc_id, "the engine generated no step"
    created = eng.db.created[-1]
    deck = Path(created["input_file"])
    return deck, Path(created["work_dir"]), created["settings"]


def _staged(step_dir):
    return sorted(p.name for p in step_dir.glob("*.f20"))


# ---------------------------------------------------------------------------
# Default: off, and the engine's output is exactly what it was
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("settings", [None, {}, {"max_concurrent_jobs": 5},
                                      {"guessp_restart": False},
                                      {"guessp_restart": "yes"}],
                         ids=["no-settings", "empty", "other-keys", "false", "not-a-bool"])
@pytest.mark.parametrize("target,deck", [("SP", SP_DECK), ("FREQ", FREQ_DECK),
                                         ("OPT2", OPT2_DECK)],
                         ids=["SP", "FREQ", "OPT2"])
def test_without_the_option_the_deck_is_untouched_and_nothing_is_staged(
        tmp_path, settings, target, deck):
    eng = _engine(tmp_path, execution_settings=settings)
    path, step_dir, calc_settings = _generate(eng, target, deck)
    assert path.read_bytes() == deck.encode()
    assert _staged(step_dir) == []
    assert "guessp_from" not in calc_settings


# ---------------------------------------------------------------------------
# On: the same-geometry, same-basis follow-ups restart from the matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source_type,target,deck", [
    ("OPT", "SP", SP_DECK),       # SP at the optimised geometry
    ("OPT", "FREQ", FREQ_DECK),   # FREQCALC's reference point is that geometry
    ("OPT", "OPT2", OPT2_DECK),   # OPT2 starts where OPT ended
    ("SP", "SP2", SP_DECK),
    ("SP", "OPT2", OPT2_DECK),
], ids=["OPT-SP", "OPT-FREQ", "OPT-OPT2", "SP-SP2", "SP-OPT2"])
def test_same_geometry_and_basis_restart_from_the_predecessor(tmp_path, source_type,
                                                              target, deck):
    eng = _engine(tmp_path, source_type=source_type,
                  execution_settings={"guessp_restart": True})
    path, step_dir, calc_settings = _generate(eng, target, deck)
    job = path.stem
    assert _staged(step_dir) == [f"{job}.f20"]
    assert (step_dir / f"{job}.f20").read_bytes() == F9
    # The deck gains exactly one record, GUESSP right before SCFDIR, where
    # the d12 writer puts it; nothing else changes.
    expected = deck.replace("SCFDIR\n", "GUESSP\nSCFDIR\n", 1)
    assert path.read_text() == expected
    assert calc_settings["guessp_from"] == "c1"


def test_a_different_functional_still_restarts(tmp_path):
    """The manual allows different "computational conditions" (p. 115); the
    density matrix is only a starting guess in the same basis."""
    deck = SP_DECK.replace("B3LYP-D3", "PBE0")
    eng = _engine(tmp_path, execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP", deck)
    assert "GUESSP" in path.read_text().splitlines()
    assert _staged(step_dir)


def test_a_deck_that_already_asks_for_guessp_is_not_given_a_second_one(tmp_path):
    """opt2d12 carries a parent's GUESSP; the matrix is still staged."""
    deck = SP_DECK.replace("SCFDIR\n", "GUESSP\nSCFDIR\n")
    eng = _engine(tmp_path, execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP", deck)
    assert path.read_text() == deck
    assert _staged(step_dir)


# ---------------------------------------------------------------------------
# On, but the matrix is not a valid guess: the deck is left exactly as written
# ---------------------------------------------------------------------------
NOT_ELIGIBLE = {
    "basis name changed": (
        "pto_sp\n" + _geom(a="3.95") .replace("END\n", "BASISSET\nPOB-TZVP-REV2\n")
        + HAM + SCF),
    "basis exponent block changed": (
        "pto_sp\n" + _geom(a="3.95") + EXT_BASIS.replace("0.3565870100", "0.2000000000")
        + HAM + SCF),
    "basis shell added": (
        "pto_sp\n" + _geom(a="3.95")
        + EXT_BASIS.replace("99 0", "0 3 1 0.0 1.0\n  0.8 1.0\n99 0") + HAM + SCF),
    "space group changed": (
        "pto_sp\n" + _geom(a="3.95", sg="1") + EXT_BASIS + HAM + SCF),
    "atoms reordered": (
        "pto_sp\n" + _geom(a="3.95", atoms=("8 0.0 0.5 0.5", "22 0.5 0.5 0.5"))
        + EXT_BASIS + HAM + SCF),
    "atom added": (
        "pto_sp\n" + _geom(a="3.95", atoms=("22 0.5 0.5 0.5", "8 0.0 0.5 0.5",
                                             "82 0.0 0.0 0.0"))
        + EXT_BASIS + HAM + SCF),
    "symmetry edited": (
        "pto_sp\n" + _geom(a="3.95", block="MODISYMM\n1\n1 1\n") + EXT_BASIS + HAM + SCF),
    "closed shell from spin-polarized": (
        "pto_sp\n" + _geom(a="3.95") + EXT_BASIS + HAM.replace("SPIN\n", "") + SCF),
    "UHF from DFT SPIN": (
        "pto_sp\n" + _geom(a="3.95") + EXT_BASIS + "UHF\n" + SCF),
    "ATOMSPIN": (
        "pto_sp\n" + _geom(a="3.95") + EXT_BASIS + HAM
        + SCF.replace("SCFDIR\n", "ATOMSPIN\n1\n1 1\nSCFDIR\n")),
    "two-component": (
        "pto_sp\n" + _geom(a="3.95") + EXT_BASIS + HAM
        + SCF.replace("SCFDIR\n", "TWOCOMPON\nSOC\nEND\nSCFDIR\n")),
    "EXTERNAL geometry": (
        "pto_sp\nEXTERNAL\nEND\n" + EXT_BASIS + HAM + SCF),
}


@pytest.mark.parametrize("case", sorted(NOT_ELIGIBLE))
def test_a_mismatched_follow_up_is_left_alone(tmp_path, case):
    deck = NOT_ELIGIBLE[case]
    eng = _engine(tmp_path, execution_settings={"guessp_restart": True})
    path, step_dir, calc_settings = _generate(eng, "SP", deck)
    assert path.read_bytes() == deck.encode()
    assert _staged(step_dir) == []
    assert "guessp_from" not in calc_settings


@pytest.mark.parametrize("f9", [None, b""])
def test_no_matrix_means_no_guessp(tmp_path, f9):
    """A run CRYSTAL aborted copies back an EMPTY fort.9 (measured); GUESSP
    with nothing to read stops CRYSTAL."""
    eng = _engine(tmp_path, f9=f9, execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP", SP_DECK)
    assert path.read_text() == SP_DECK
    assert _staged(step_dir) == []


def test_a_freq_step_is_never_the_source(tmp_path):
    eng = _engine(tmp_path, source_type="FREQ", source_deck=FREQ_DECK,
                  execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP2", SP_DECK)
    assert path.read_text() == SP_DECK
    assert _staged(step_dir) == []


def test_an_old_job_script_without_f20_staging_gets_no_guessp(tmp_path):
    """A workflow planned before the GUESSP staging existed keeps its old
    workflow_scripts/ templates, which neither stage fort.20 nor strip GUESSP
    - CRYSTAL would stop with "COPY OF WAVEFUNCTION FILE fort.20 CAN NOT BE
    FOUND"."""
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    (scripts / "submitcrystal23_sp_2.sh").write_text(
        "#!/bin/bash --login\n#SBATCH -J $1\nexport JOB=$1\nexport DIR=$SLURM_SUBMIT_DIR\n"
        "cp $DIR/$JOB.d12 $scratch/$JOB/INPUT\nPcrystal\ncp fort.9 ${DIR}/${JOB}.f9\n")
    eng = _engine(tmp_path, execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP", SP_DECK)
    assert path.read_text() == SP_DECK
    assert _staged(step_dir) == []


# ---------------------------------------------------------------------------
# The generated job script picks the staged matrix up (fake SLURM run)
# ---------------------------------------------------------------------------
def test_the_generated_job_script_stages_the_predecessors_matrix(tmp_path):
    eng = _engine(tmp_path, execution_settings={"guessp_restart": True})
    path, step_dir, _ = _generate(eng, "SP", SP_DECK)
    job = path.stem
    script = (step_dir / f"{job}.sh").read_text()
    start = script.index("# GUESSP restart")
    block = script[start:script.index("\nfi\n", start) + 4]
    scratch = tmp_path / "scratch"
    (scratch / job).mkdir(parents=True)
    # The script copies the deck to INPUT before this block runs.
    shutil.copy(path, scratch / job / "INPUT")
    prelude = f'DIR="{step_dir}"\nJOB={job}\nscratch="{scratch}"\nRESTART_KEEPS_FORT20=""\n'
    run = subprocess.run(["bash", "-c", prelude + block], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (scratch / job / "fort.20").read_bytes() == F9
    assert "GUESSP" in (scratch / job / "INPUT").read_text().splitlines()
    assert f"GUESSP: staged {job}.f20 as fort.20" in run.stdout


# ---------------------------------------------------------------------------
# The comparison on decks the real CRYSTALOptToD12 writes
# ---------------------------------------------------------------------------
def _real_child(tmp_path, name, calc_type):
    shutil.copy(LOW_DIM / f"{name}.out", tmp_path)
    shutil.copy(LOW_DIM / f"{name}.d12", tmp_path)
    args = [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file",
            f"{name}.out", "--d12-file", f"{name}.d12", "--non-interactive",
            "--calc-type", calc_type]
    result = subprocess.run(args, cwd=tmp_path, input="", capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    kids = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(kids) == 1, sorted(p.name for p in tmp_path.iterdir())
    return kids[0].read_text()


@pytest.mark.parametrize("name", ["graphene_lg80", "graphene_lg47", "polyyne_rg51"])
@pytest.mark.parametrize("calc_type", ["SP", "FREQ"])
def test_real_follow_up_decks_are_recognised_as_the_same_system(tmp_path, name, calc_type):
    parent = (LOW_DIM / f"{name}.d12").read_text()
    child = _real_child(tmp_path, name, calc_type)
    assert guessp_chain.refusal(parent, child) is None
    with_guessp = guessp_chain.add_guessp(child).splitlines()
    assert with_guessp.count("GUESSP") == 1
    assert with_guessp[with_guessp.index("GUESSP") + 1] == "SCFDIR"
    assert [l for l in with_guessp if l != "GUESSP"] == child.splitlines()


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------
def test_basis_numbers_compare_as_numbers():
    """opt2d12 may rewrite a parent's basis with other number formatting."""
    reformatted = SP_DECK.replace("0 0 6 2.0 1.0", "0 0 6 2. 1.").replace(
        "  27032.382631", " 27032.3826310")
    assert guessp_chain.refusal(OPT_DECK, reformatted) is None


def test_add_guessp_keeps_crlf_line_endings():
    deck = SP_DECK.replace("\n", "\r\n")
    out = guessp_chain.add_guessp(deck)
    assert "\r\nGUESSP\r\nSCFDIR\r\n" in out
    assert out.replace("GUESSP\r\n", "", 1) == deck


def test_the_legacy_executor_path_says_it_does_not_add_guessp(tmp_path, capsys):
    """WorkflowExecutor.generate_inputs_with_crystal_opt (the executor's own
    step generator, reached only from its monitor loop, which nothing starts)
    does not apply guessp_restart. With the option on it must say so rather
    than silently writing cold-start decks."""
    from mace.workflow.executor import WorkflowExecutor

    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.outputs_dir = tmp_path
    (tmp_path / WF_ID).mkdir()
    ex.active_workflows = {WF_ID: {
        "plan": {"workflow_sequence": ["OPT", "SP"],
                 "execution_settings": {"guessp_restart": True}},
        "submitted_jobs": {}}}
    ex.generate_inputs_with_crystal_opt(WF_ID, 1, "SP", {})
    out = capsys.readouterr()
    assert "guessp_restart" in out.out + out.err
    assert "engine" in out.out + out.err

    ex.active_workflows[WF_ID]["plan"]["execution_settings"] = {}
    ex.generate_inputs_with_crystal_opt(WF_ID, 1, "SP", {})
    out = capsys.readouterr()
    assert "guessp_restart" not in out.out + out.err
