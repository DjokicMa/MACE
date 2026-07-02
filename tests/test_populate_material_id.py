"""Regression test for bug_005 (cloud review): the completed-jobs scanner used by
the queue manager must derive material_id with the canonical
create_material_id_from_file, so continuation/numbered output filenames dedup
against engine-created records instead of registering duplicate material rows.

These are self-contained: a .out only needs the 'TERMINATION' marker the scanner
checks for, so a tiny tmp file exercises the real derivation path.

Imports are done lazily inside the tests (not at module top): importing the
database stack at collection time pulls in the pandas->pyarrow chain, which can
collide with other test modules that import xarray/pyarrow at collection (a known
pyarrow "type extension already defined" double-registration). Keeping these
imports local avoids perturbing unrelated test modules' collection order.
"""
import pytest


@pytest.mark.parametrize("fname,expected_calc", [
    ("mat_opt2.out", "OPT"),                                   # numbered calc type
    ("mat_opt_B3LYP-D3_optimized.out", "OPT"),                 # OPT continuation
    ("1_dia_opt_rev1_sp_B3LYP-D3_optimized.out", "SP"),        # SP continuation
])
def test_scan_material_id_matches_canonical(tmp_path, fname, expected_calc):
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    from mace.database.materials import create_material_id_from_file
    out = tmp_path / fname
    out.write_text("CRYSTAL run\n OPT END - CONVERGENCE\n TTTTTT TERMINATION TTTTTT\n")
    calcs = scan_for_completed_calculations(tmp_path)
    assert len(calcs) == 1
    c = calcs[0]
    # The whole point: scan id == canonical id (was a divergent strip-loop result).
    assert c["material_id"] == create_material_id_from_file(fname)
    assert c["calc_type"] == expected_calc


def test_scan_continuation_id_is_not_the_full_stem(tmp_path):
    """Pin the concrete divergence the bug caused: the continuation stem must NOT
    leak through as the material id (that created the duplicate material rows)."""
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    out = tmp_path / "mat_opt_B3LYP-D3_optimized.out"
    out.write_text("CRYSTAL run\n TTTTTT TERMINATION TTTTTT\n")
    c = scan_for_completed_calculations(tmp_path)[0]
    assert c["material_id"] == "mat"
    assert c["material_id"] != "mat_opt_B3LYP-D3_optimized"


@pytest.mark.parametrize("fname,expected_id", [
    # transport / charge+potential are calc-type tokens too: without them in the
    # suffix regex, simple-named outputs of the newly scanned TRANSPORT and
    # CHARGE+POTENTIAL types registered as brand-new (duplicate) materials.
    ("mat_transport.out", "mat"),
    ("mat_transport2.out", "mat"),
    ("mat_charge+potential.out", "mat"),
    ("mat_potential.out", "mat"),
    # must match the ID its sibling SP produces (both 'C1-RCSR-ana_optimized')
    ("C1-RCSR-ana_optimized_TRANSPORT.out", "C1-RCSR-ana_optimized"),
])
def test_transport_chargepot_tokens_canonicalized(fname, expected_id):
    from mace.database.materials import create_material_id_from_file
    assert create_material_id_from_file(fname) == expected_id


def test_transport_id_matches_sibling_sp_id():
    """The dedup property that actually matters: a TRANSPORT output and its
    parent SP output must derive the SAME material id."""
    from mace.database.materials import create_material_id_from_file as cmif
    assert (cmif("C1-RCSR-ana_optimized_TRANSPORT.out")
            == cmif("C1-RCSR-ana_optimized_sp_B3LYP-D3_optimized.out"))
