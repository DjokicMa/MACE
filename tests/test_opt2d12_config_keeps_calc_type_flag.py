"""`opt2d12 --calc-type X --config-file C` writes an X deck.

A config without calculation_type (a tolerances-only template, say) left the
parent's own type in place: `--calc-type SP --config-file {tolerances only}
--non-interactive` on an OPT parent wrote ..._opt_..._optimized.d12 with
OPTGEOM. The explicit flag wins. Real parents: a graphene SP run (in the
repo) and the corpus diamond OPT.
"""
import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
LOW_DIM = REPO_ROOT / "tests" / "data" / "low_dim_groups"


def _run(cwd, stem, calc_type, extra=()):
    config = cwd / "tolerances.json"
    config.write_text(json.dumps({"tolerances": {"TOLDEE": 9}}))
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{stem}.out",
         "--calc-type", calc_type, "--config-file", str(config), "--non-interactive", *extra],
        cwd=cwd, input="", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    decks = sorted(p.name for p in cwd.glob(f"{stem}_*_optimized.d12"))
    assert len(decks) == 1, decks
    assert decks[0].startswith(f"{stem}_{calc_type.lower()}_"), decks
    return (cwd / decks[0]).read_text().splitlines()[1:]


@pytest.mark.parametrize("calc_type,block", [("OPT", "OPTGEOM"), ("FREQ", "FREQCALC")])
def test_flag_wins_over_an_sp_parent(tmp_path, calc_type, block):
    for suffix in (".out", ".d12"):
        shutil.copy(LOW_DIM / f"graphene_lg80{suffix}", tmp_path)
    body = _run(tmp_path, "graphene_lg80", calc_type)
    assert block in body
    assert body[body.index("TOLDEE") + 1].strip() == "9"


def test_opt_type_flag_is_honored_with_a_config(tmp_path):
    for suffix in (".out", ".d12"):
        shutil.copy(LOW_DIM / f"graphene_lg80{suffix}", tmp_path)
    body = _run(tmp_path, "graphene_lg80", "OPT", ("--opt-type", "ATOMONLY"))
    assert body[body.index("OPTGEOM") + 1] == "ATOMONLY"


def test_sp_flag_wins_over_an_opt_parent(tmp_path):
    src = TEST_DATA / "OPT" / "1_dia_opt_BULK_OPTGEOM.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    body = _run(tmp_path, "1_dia_opt_BULK_OPTGEOM", "SP")
    assert "OPTGEOM" not in body and "FREQCALC" not in body
