"""The bundled MgO defect outputs are excerpts with no run-host data.

tests/data/supercell_parents/rg_q0.out.gz and rs_qm1.out.gz were complete
CRYSTAL23 outputs of two runs: MPI start-up lines, the process count, start
and termination dates, per-step wall/CPU timings and per-node CPU times. They
are now trimmed to what the tests read (header and title, input and supercell
geometry, basis listing, GHOSTS/substitution records, the cell's AO and
electron counts, the SCF result), with every run-specific line dropped.

tests/data/low_dim_groups/polyyne_rg4.out (the rod group 4 polymer parent of
test_band_polymer_path.py) is held to the same rule.
"""
import gzip
import re

import pytest

from conftest import REPO_ROOT

DATA = REPO_ROOT / "tests" / "data" / "supercell_parents"
LOW_DIM = REPO_ROOT / "tests" / "data" / "low_dim_groups"
HOST = re.compile(r"MPI startup|PROCESS +\d+ +OF +\d+ +WORKING|DATE +\d\d \d\d \d{4}|"
                  r"TELAPSE|TCPU|NODE +\d+ +CPU TIME|TOTAL CPU TIME|/(home|mnt|scratch|tmp|mnt)/")


@pytest.mark.parametrize("stem", ["rg_q0", "rs_qm1"])
def test_excerpt_has_no_run_host_data(stem):
    text = gzip.open(DATA / f"{stem}.out.gz", "rt").read()
    assert not HOST.findall(text)
    assert len(text.splitlines()) < 600          # an excerpt, not the whole run
    assert "CRYSTAL23" in text and "== SCF ENDED" in text


@pytest.mark.parametrize("stem", ["polyyne_rg4"])
def test_low_dim_excerpt_has_no_run_host_data(stem):
    text = (LOW_DIM / f"{stem}.out").read_text()
    assert not HOST.findall(text)
    assert len(text.splitlines()) < 450          # the run was 565 lines
    assert "CRYSTAL23" in text and "== SCF ENDED" in text
