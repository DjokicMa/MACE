"""One error classification for completion_checker, the recovery detector and
the queue manager (mace.utils.crystal_errors).

The three used to disagree on real failures: completion_checker had no OOM or
Fermi patterns and filed an OOM kill and a diverged SCF as "Segmentation
fault/runtime error"; the detector matched line by line, so the BAD
TERMINATION banner before "KILLED BY SIGNAL: 9" made an OOM kill a
system_error with no recovery, and it called a NEIGHB stop a basis-set linear
dependence. The classes are now matched in one priority order (an OOM kill
first, then CRYSTAL's own errors, then the generic MPI/SLURM lines).
"""

import contextlib
import io
from pathlib import Path

import pytest

from conftest import find_data
from mace.completion_checker import categorize_output_file
from mace.recovery.detector import CrystalErrorDetector
from mace.utils import crystal_errors as ce


@pytest.fixture
def det(tmp_path):
    return CrystalErrorDetector(base_dir=str(tmp_path), db_path=str(tmp_path / "x.db"),
                                enable_tracking=False)


@pytest.fixture
def mgr():
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)


# corpus output -> (completion_checker bucket, detector type, manager type)
REAL = [
    ("FAILED_QA/3_dia3_opt_killed_signal9_oom.out", ("memory", "memory_error", "memory_error")),
    ("FAILED_QA/T7_sp_killed_signal9_scf_converged.out",
     ("memory", "memory_error", "memory_error")),
    ("FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out",
     ("too_many_scf", "convergence_error", "convergence_error")),
    ("FAILED_QA/3_dia3_neighb_atoms_too_close.out",
     ("geometry_small_dist", "geometry_error", "geometry_error")),
]


@pytest.mark.parametrize("name, expected", REAL)
def test_the_three_classifiers_agree_on_real_failures(name, expected, det, mgr):
    out = find_data(name)
    with contextlib.redirect_stdout(io.StringIO()):
        bucket = categorize_output_file(out)[0]
        found = det.analyze_output_file(Path(out))
        managed = mgr.analyze_calculation_error({"output_file": str(out)})[0]
    assert (bucket, found["error_type"], managed) == expected


def test_an_oom_kill_outranks_the_bad_termination_banner_before_it():
    text = ("=   BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES\n"
            "=   RANK 3 PID 1 RUNNING AT node\n"
            "=   KILLED BY SIGNAL: 9 (Killed)\n")
    assert ce.classify(text).key == "oom"
    assert ce.classify(text).error_type == "memory_error"


def test_a_crystal_error_outranks_the_mpi_abort_after_it():
    text = (" ERROR **** ZERO **** FERMI ENERGY NOT IN INTERVAL\n"
            "Abort(1) on node 3 (rank 3 in comm 0): application called MPI_Abort\n")
    assert ce.classify(text).error_type == "convergence_error"


def test_the_generic_lines_alone_are_a_runtime_error_with_no_recovery():
    match = ce.classify("Abort(1) on node 3 (rank 3 in comm 0): application called MPI_Abort\n")
    assert match.key == "runtime" and match.bucket == "potential"
    assert match.no_recovery


def test_a_clean_output_matches_nothing():
    assert ce.classify("    TOTAL CPU TIME =    12.3\n") is None


def test_every_class_has_a_handler_or_says_there_is_none():
    """A class the recovery engine has a handler for is recovered; every other
    one carries the reason it is not, for the manager to print."""
    from mace.recovery.recovery import ErrorRecoveryEngine
    eng = ErrorRecoveryEngine.__new__(ErrorRecoveryEngine)
    eng.config_path = None
    handled = set(eng.load_recovery_config()["error_recovery"])
    for cls in ce.ERROR_CLASSES:
        if cls.error_type in handled:
            assert not cls.no_recovery, cls.key
        else:
            assert cls.no_recovery, cls.key
