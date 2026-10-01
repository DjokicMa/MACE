"""The shipped DOSS example configs must give decks CRYSTAL accepts.

Measured on HPCC (CRYSTAL23 properties, diamond SP fort.9), decks written
from these configs before the fix stopped with:
  - "DOSS FORMAT ERROR IN INPUT DECK": NPR=1 with no prtrec record after it;
  - "DENSIM BAND RANGE NOT ALLOWED": band range 0 999 (bands are 1..NAO).
They also asked for IPLO 0 (no DOSS.DAT written), and "Total DOS only"
came out as per-orbital projections because the writer ignored
"projection_type". The fixed deck (bands 1 36, NPR 1 + "105 -1") ran.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

PARENT = TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"
CONFIGS = REPO_ROOT / "Crystal_d3" / "example_configs"
N_AO = 36  # diamond, this basis: "NUMBER OF AO 36" in the parent output

pytestmark = pytest.mark.skipif(
    not PARENT.with_suffix(".out").exists(), reason="needs the test/ corpus")


def _doss_deck(tmp_path, config_name):
    for suffix in (".out", ".f9", ".d12"):
        shutil.copy(PARENT.with_suffix(suffix), tmp_path)
    out = tmp_path / PARENT.with_suffix(".out").name
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "--input", str(out),
         "--calc-type", "DOSS", "--config-file", str(CONFIGS / config_name),
         "--output-dir", str(tmp_path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    deck = (tmp_path / (PARENT.name + "_doss.d3")).read_text().splitlines()
    i = deck.index("DOSS")
    return deck, [int(x) for x in deck[i + 1].split()], deck[i + 2:]


@pytest.mark.parametrize("config_name", [
    "doss_total_only.json",
    "doss_orbital_projections.json",
    "doss_element_orbital_auto.json",
])
def test_example_config_writes_a_valid_doss_record(tmp_path, config_name):
    deck, (npro, npt, inzb, ifnb, iplo, npol, npr), rest = _doss_deck(tmp_path, config_name)
    assert iplo == 2, "the example should write DOSS.DAT for plotting"
    if inzb > 0:
        assert 1 <= inzb <= ifnb <= N_AO
    else:
        assert inzb < 0 and ifnb < 0  # energy window: BMI/BMA record follows
    assert rest[-1] == "END"
    if npr:
        # the prtrec record is the last record before END
        assert rest[-2] == "105 -1"
    body = rest[:-2] if npr else rest[:-1]
    if inzb < 0:
        body = body[1:]
    assert len(body) == npro


def test_total_only_example_has_no_projections(tmp_path):
    _, record, _ = _doss_deck(tmp_path, "doss_total_only.json")
    assert record[0] == 0
