"""Characterization tests for the pure/mockable logic kernels in
mace/workflow/engine.py (WorkflowEngine).

These pin the CURRENT, validated behavior of the calc-type parsing, DFT
dependency resolution, next-step selection, material-name/suffix derivation,
SLURM memory math, and workflow-path parsing — the logic whose silent breakage
corrupts the workflow DAG or mis-resources jobs. Expected values were captured
from the real code (no SLURM, no DB, no 12GB corpus).

WorkflowEngine.__init__ builds a sqlite MaterialDatabase and scans the
filesystem, so every test instantiates via __new__ to exercise the method under
test in isolation (per the standard harness for this class).
"""
import pytest

from mace.workflow.engine import WorkflowEngine


@pytest.fixture
def eng():
    return WorkflowEngine.__new__(WorkflowEngine)


@pytest.mark.parametrize("raw,expected", [
    ("OPT", ("OPT", 1)),
    ("OPT2", ("OPT", 2)),
    ("OPT_2", ("OPT", 2)),
    ("SP", ("SP", 1)),
    ("BAND3", ("BAND", 3)),
    ("BAND_3", ("BAND", 3)),
    ("CHARGE+POTENTIAL", ("CHARGE+POTENTIAL", 1)),
    ("CHARGE+POTENTIAL_2", ("CHARGE+POTENTIAL", 2)),
    ("TRANSPORT", ("TRANSPORT", 1)),
    ("opt", ("opt", 1)),   # lowercase falls through the [A-Z]+ regex -> (input, 1)
    ("", ("", 1)),
])
def test_parse_calc_type(eng, raw, expected):
    assert eng._parse_calc_type(raw) == expected


@pytest.mark.parametrize("calc,seq,expected", [
    ("SP", ["OPT", "SP"], "OPT"),
    ("BAND", ["OPT", "SP", "BAND", "DOSS"], "SP"),     # not DOSS — the SP wavefunction
    ("FREQ", ["OPT", "SP", "FREQ"], "OPT"),
    ("OPT", ["OPT", "SP"], None),                       # first step has no dependency
    ("OPT2", ["OPT", "OPT2"], "OPT"),
    ("OPT2", ["OPT", "FREQ", "OPT2"], "OPT"),           # skips FREQ
    ("DOSS", ["OPT", "SP"], None),                      # not in sequence
])
def test_find_dependency_in_sequence(eng, calc, seq, expected):
    assert eng._find_dependency_in_sequence(calc, seq) == expected


@pytest.mark.parametrize("calc,seq,completed,failed,expected", [
    ("OPT", ["OPT", "SP"], set(), set(), (True, None)),         # first step
    ("SP", ["OPT", "SP"], {"OPT"}, set(), (True, None)),        # dep completed
    ("SP", ["OPT_1", "SP"], {"OPT"}, set(), (True, None)),      # base-type match
    ("SP", ["OPT", "SP"], set(), set(), (False, "OPT")),        # dep not done
    ("BAND", ["SP", "BAND"], set(), {"SP"}, (False, "SP")),     # optional-failed D3 still blocks
])
def test_check_dependencies_met(eng, calc, seq, completed, failed, expected):
    assert eng._check_dependencies_met(calc, "m", seq, completed, failed) == expected


@pytest.mark.parametrize("idx,seq,expected", [
    (1, ["OPT", "SP", "BAND", "DOSS"], ["BAND", "DOSS"]),   # BAND+DOSS bundled
    (1, ["OPT", "SP", "TRANSPORT"], ["TRANSPORT"]),         # TRANSPORT not paired
    (3, ["OPT", "SP", "BAND", "DOSS"], []),                 # at end
    (0, [], []),                                            # empty
])
def test_get_next_steps_from_sequence(eng, idx, seq, expected):
    assert eng._get_next_steps_from_sequence(idx, seq, "SP") == expected


def test_get_next_unstarted_steps_bundles_and_skips(eng):
    # The only side-effecting collaborator is _calculation_already_exists.
    existing = set()
    eng._calculation_already_exists = lambda mid, ct: ct in existing
    assert eng._get_next_unstarted_steps(1, ["OPT", "SP", "BAND", "DOSS"], "m") == ["BAND", "DOSS"]
    assert eng._get_next_unstarted_steps(3, ["OPT", "SP", "BAND", "DOSS"], "m") == []
    existing = {"BAND"}
    eng._calculation_already_exists = lambda mid, ct: ct in existing
    assert eng._get_next_unstarted_steps(1, ["OPT", "SP", "BAND", "DOSS"], "m") == ["DOSS"]


@pytest.mark.parametrize("raw,expected", [
    ("1_dia_opt", "1_dia"),
    ("1_dia_sp2", "1_dia"),
    ("test2_sp", "test2"),                                          # name digit preserved
    ("mat_CRYSTAL_OPTGEOM_PBE-D3_POB-TZVP", "mat"),
    ("quartz_sp.d12", "quartz"),                                    # stem stripped first
    ("3,4^2T10-CA_rev1_sp_B3LYP-D3", "3,4^2T10-CA_rev1_sp"),        # real validated name
])
def test_extract_core_material_name(eng, raw, expected):
    assert eng.extract_core_material_name(raw) == expected


@pytest.mark.parametrize("value,unit,expected", [
    (1024, "M", 1.0),
    (1, "G", 1.0),
    (1048576, "K", 1.0),
    (8, "", 8.0),
])
def test_convert_to_gb(eng, value, unit, expected):
    assert eng._convert_to_gb(value, unit) == expected


def test_fix_memory_reporting(eng):
    out = eng._fix_memory_reporting("#SBATCH --mem-per-cpu=4G\n#SBATCH --ntasks=8\n")
    assert "4G per CPU" in out and "32.0GB total" in out   # 4G * 8 ntasks
    assert "64.0GB" in eng._fix_memory_reporting("#SBATCH --mem=64G\n")
    assert eng._fix_memory_reporting("echo hi\n") == "echo hi\n"  # unchanged when no mem


@pytest.mark.parametrize("path,expected", [
    ("/x/workflow_outputs/workflow_20260613_120000/step_002_SP/mat/file.d12",
     ("workflow_20260613_120000", "step_002_SP")),
    ("/x/random/file.d12", None),
    # short 'workflow_1' name fails the length guard -> no match
    ("/y/workflow_outputs/workflow_1/step_001_OPT/m/f.d12", None),
])
def test_find_workflow_context(eng, path, expected):
    assert eng.find_workflow_context(path) == expected


def test_get_next_calc_suffix_increments_from_disk(eng, tmp_path):
    assert eng.get_next_calc_suffix("1_dia", "SP", tmp_path) == "_sp"
    (tmp_path / "step_002_SP" / "1_dia_sp").mkdir(parents=True)
    assert eng.get_next_calc_suffix("1_dia", "SP", tmp_path) == "_sp2"
    (tmp_path / "step_004_SP" / "1_dia_sp2").mkdir(parents=True)
    assert eng.get_next_calc_suffix("1_dia", "SP", tmp_path) == "_sp3"


def test_get_next_step_number_fs_fallback(eng, tmp_path):
    # Force the filesystem fallback (real plan lookup returns >=1).
    eng.get_workflow_step_number = lambda wid, ct: 0
    assert eng._get_next_step_number(tmp_path, "OPT") == 1   # empty base
    (tmp_path / "step_001_OPT").mkdir()
    (tmp_path / "step_002_SP").mkdir()
    assert eng._get_next_step_number(tmp_path, "OPT") == 1   # existing exact base type
    assert eng._get_next_step_number(tmp_path, "OPT2") == 3  # new numbered -> max+1
    assert eng._get_next_step_number(tmp_path, "DOSS") == 3  # new base type -> max+1
