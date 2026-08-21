"""The basis-set compatibility check must reflect what CRYSTAL23 actually has.

Origin: a user's hybrid lead perovskites all died with
``ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL`` under HSE-3c. The cells were
neutral; CRYSTAL's own def2-mSVP library is broken for Pb. MACE emitted those
decks silently because check_basis_set_compatibility guarded on
``if basis_set in INTERNAL_BASIS_SETS`` and the 3c basis sets have no entry
there -- so the element loop never ran and every element was "compatible".

The expectations below are measured, not assumed: tests/basis_coverage/*.csv
are the raw results of running CRYSTAL/23-intel-2023a on a one-atom cell for
every element 1-99, produced by tests/basis_coverage/scan_basis.sh.
"""
import csv
import glob
import os
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_constants import (  # noqa: E402
    VERIFIED_INTERNAL_BASIS_ELEMENTS,
    check_basis_set_compatibility,
)

COVERAGE_DIR = REPO_ROOT / "tests" / "basis_coverage"


def _measured(basis):
    path = COVERAGE_DIR / f"cov_{basis}.csv"
    with open(path) as fh:
        return {int(r["Z"]) for r in csv.DictReader(fh) if r["status"] == "PRESENT"}


def test_the_reported_bug_is_now_caught():
    """A lead perovskite under HSE-3c/def2-mSVP must be refused, naming Pb."""
    ok, missing = check_basis_set_compatibility("def2-mSVP", [82, 53, 6, 1, 7])
    assert not ok, "Pb in def2-mSVP must be reported incompatible"
    assert missing == [82], f"only Pb is unusable here, got {missing}"


def test_the_working_alternative_still_passes():
    """POB-TZVP-REV2 ran this exact composition to convergence on HPCC."""
    ok, missing = check_basis_set_compatibility("POB-TZVP-REV2", [82, 53, 6, 1, 7])
    assert ok, f"POB-TZVP-REV2 must accept Pb/I, got missing={missing}"


def test_iodine_is_not_falsely_blamed():
    """LiI converged under def2-mSVP -- iodine is present; only Pb is broken."""
    ok, _ = check_basis_set_compatibility("def2-mSVP", [3, 53])
    assert ok


@pytest.mark.parametrize("basis", [os.path.basename(p)[4:-4]
                                   for p in sorted(glob.glob(str(COVERAGE_DIR / "cov_*.csv")))])
def test_table_matches_measured_crystal_behavior(basis):
    """Every element the table claims must really load, and vice versa."""
    measured = _measured(basis)
    claimed = set(VERIFIED_INTERNAL_BASIS_ELEMENTS[basis])
    overclaimed = sorted(claimed - measured)
    underclaimed = sorted(measured - claimed)
    assert not overclaimed, (
        f"{basis} claims elements CRYSTAL rejects: {overclaimed} -- these would "
        f"produce a deck that dies at basis load")
    assert not underclaimed, (
        f"{basis} omits elements CRYSTAL supports: {underclaimed} -- users would "
        f"be blocked from valid calculations")


@pytest.mark.parametrize("basis", ["def2-mSVP", "MINIX", "mTZVP",
                                   "SOLDEF2MSVP", "SOLMINIX"])
def test_3c_basis_sets_are_no_longer_a_free_pass(basis):
    """The 3c sets are absent from INTERNAL_BASIS_SETS; they must still be
    checked, or the original bug returns for every composite method."""
    ok, missing = check_basis_set_compatibility(basis, [99])
    assert not ok and missing == [99], (
        f"{basis} accepted element 99; the coverage check is not being applied")


def test_unknown_basis_is_permitted_not_rejected():
    """An unrecognised name means 'cannot verify', not 'everything missing' --
    users may supply custom basis sets we have not mapped."""
    ok, missing = check_basis_set_compatibility("SOME-CUSTOM-SET", [82, 6])
    assert ok and missing == []


def test_basis_name_matching_is_case_insensitive():
    """CRYSTAL keywords are case-insensitive and the 3c names are mixed case."""
    ok, missing = check_basis_set_compatibility("DEF2-MSVP", [82])
    assert not ok and missing == [82]


def test_external_missing_file_is_caught():
    """read_basis_file returns '' for a missing element file, silently dropping
    that basis block. The check must catch it first."""
    sys.path.insert(0, str(REPO_ROOT))
    import mace_config

    # Xe (54) has no file in the shipped external directories.
    ok, missing = check_basis_set_compatibility(
        mace_config.DEFAULT_TZ_PATH, [54], "EXTERNAL")
    assert not ok and missing == [54]

    ok, _ = check_basis_set_compatibility(
        mace_config.DEFAULT_TZ_PATH, [6, 82, 53], "EXTERNAL")
    assert ok, "C/Pb/I all have external basis files and must pass"


def test_external_nondirectory_does_not_blame_elements():
    """A bad basis directory is not the elements' fault -- don't mis-report."""
    ok, missing = check_basis_set_compatibility(
        "/nonexistent/basis/dir", [6, 82], "EXTERNAL")
    assert ok and missing == []
