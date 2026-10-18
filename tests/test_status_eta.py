"""The ETA column and totals footer of `mace monitor --status` (1.1.6): the
same progress()/totals functions as `mace cluster job`, one cached squeue for
the table, blank cells when nothing is known."""
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest

from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace

from mace.cluster import status_eta as S

REPO = Path(__file__).resolve().parents[1]
OPT = REPO / "test" / "OPT" / "Ag1Cl5_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"


def _total(lo, hi, conf="rough", ended=None):
    return {"remaining_s": {"low": lo, "high": hi}, "confidence": conf, "info": {"ended": ended} if ended else {}}


def test_eta_cell_formats():
    assert S.eta_cell(_total(1.4 * 3600, 3.2 * 3600, "fair")) == "~2.1 h (1.4-3.2 h)"
    assert S.eta_cell(_total(1.4 * 3600, 3.2 * 3600)) == "~2.1 h (1.4-3.2 h) rough"
    assert S.eta_cell(_total(900, 2400, "fair")) == "~24 min (15-40 min)"      # the geometric mean
    assert S.eta_cell(_total(40 * 60, 5400, "fair")) == "~1.0 h (40 min-1.5 h)"
    assert S.eta_cell(_total(5000, None, "unknown")) == ">= 1.4 h"
    assert S.eta_cell(_total(None, None, "unknown")) == ""
    assert S.eta_cell(_total(None, None, "unknown"), stalled=True) == "possibly stalled"
    assert S.eta_cell(_total(1.4 * 3600, 3.2 * 3600), stalled=True) == "~2.1 h (1.4-3.2 h) rough, possibly stalled"
    assert S.eta_cell(_total(0, 0, "fair", ended="completed")) == "finished"
    assert S.eta_cell(None) == ""


def _calc(cid, jid, out, status="running"):
    return {"calc_id": cid, "calc_type": "OPT", "slurm_job_id": jid, "status": status, "material_id": "m",
            "output_file": str(out), "work_dir": str(out.parent)}


def test_job_rows_running_pending_and_unknown(tmp_path):
    if not OPT.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    lines = OPT.read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (tmp_path / "a.out").write_text("".join(lines[:heads[5]]))
    shutil.copy(OPT.with_suffix(".d12"), tmp_path / "a.d12")
    (tmp_path / "c.out").write_text("not a CRYSTAL output\n")
    soon = datetime.fromtimestamp(time.time() + 3 * 3600).strftime("%Y-%m-%dT%H:%M:%S")
    sq = [{"id": "11", "state": "RUNNING", "elapsed": "0:28:00", "time_limit": "1-00:00:00"},
          {"id": "12", "state": "PENDING", "start": soon, "reason": "Priority"},
          {"id": "13", "state": "RUNNING", "elapsed": "0:01:00"}]
    rows = S.job_rows([_calc("a", "11", tmp_path / "a.out"), _calc("b", "12", tmp_path / "b.out", "submitted"),
                       _calc("c", "13", tmp_path / "c.out"), _calc("d", "", tmp_path / "d.out")], sq)
    assert [r["calc_id"] for r in rows] == ["a", "b", "c"]                       # no SLURM id: not listed
    a, b, c = rows
    assert re.match(r"(~|>= )\d", a["eta"]) and a["total"]["text"].startswith("Total ")
    assert b["eta"].startswith("starts in ~") and b["eta"].endswith("(SLURM)")
    assert c["eta"] == ""                                                        # nothing known: blank, no error
    lines_ = S.table_lines(rows)
    assert lines_[0].startswith("Jobs (ETA") and lines_[1].startswith("  a ") and lines_[3].rstrip().endswith("RUNNING")


def test_monitor_status_prints_eta_column(tmp_path):
    if not OPT.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    shutil.copy(OPT.with_suffix(".d12"), tmp_path / "mat1.d12")
    p = _mace(tmp_path, env, "submit", "--track", "mat1.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    rec = fake.records()[0]
    lines = OPT.read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (tmp_path / "mat1.out").write_text("".join(lines[:heads[5]]))
    fake.set_record(rec["id"], state="RUNNING", elapsed="0:28:00", time_limit="1-00:00:00")
    out = _mace(tmp_path, env, "monitor", "--status").stdout
    m = re.search(r"Jobs \(ETA.*\n(.*)\n", out)
    assert m, out[-3000:]
    row = m.group(1)
    assert rec["id"] in row and "RUNNING" in row and re.search(r"(~|>= )\d", row), row
    # the summary view stays as it was
    assert "Jobs (ETA" not in _mace(tmp_path, env, "monitor", "--status", "--summary").stdout


def test_status_lists_no_superseded_resubmitted_parent(tmp_path):
    """After `mace recover` the failed parent row is 'resubmitted' (its job ended
    long ago; its successor is the live job): the Jobs table shows only the
    successor, and the old rows do not pile up with each recovery."""
    import subprocess
    import sys
    from test_recover_resubmits import _env as rec_env, _seed, _recover, rows, MACE_CLI
    fake = FakeSlurm(tmp_path / "fs")
    env = rec_env(tmp_path, fake)
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    assert "Resubmitted 1 of 1" in _recover(env, work, "--yes").stdout
    st = {v["status"]: v["slurm_job_id"] for v in rows(work / "materials.db").values()}
    assert "resubmitted" in st
    old_job = st["resubmitted"]
    for cmd in (["status"], ["monitor", "--status"]):
        s = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *cmd], cwd=str(work), env=env,
                           capture_output=True, text=True, timeout=300)
        table = s.stdout.split("Jobs (ETA", 1)
        assert len(table) == 2, s.stdout[-2000:] + s.stderr[-1000:]
        assert "RESUBMITTED" not in table[1] and f" {old_job} " not in table[1], table[1]
