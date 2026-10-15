"""A database build files properties only under materials it has a row for
(bug hunt 1.1.5, #3).

A build with --register extracted every .out it saw: failed runs (which the
populate scan rightly does not register) and MACE's ``X_temp.out`` copy of a
run whose ``X.out`` sits beside it got rows under a file-name material id
with no materials row (the Database view's material page 404'd, the export
left them out), and the CLI and GUI summaries counted those ids and files:
"Materials processed: 10" and "11 new" for a folder of 5 materials.

Now a build skips an output with no calculation record when it is a failed
or unfinished run, or the _temp copy of an output beside it; an untracked
completed output (Register off) gets a materials row; and the CLI and the GUI
word the summary with one function. Real outputs, real CLI subprocess."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

EC_FREQ = "FREQ/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized"
FILES = ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12",
         "FAILED_QA/3_dia3_opt_killed_signal9_oom.out", "FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out",
         EC_FREQ + ".out", EC_FREQ + "_temp.out", EC_FREQ + "_temp.d12"]


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _folder(tmp_path, files=FILES):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp_path / "runs"
    for r in files:
        dest = d / Path(r).parent.name.replace("FAILED_QA", "FAILED") / Path(r).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEST_DATA / r, dest)
    return d


def _build(d, db, *extra):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze",
                        "--extract-properties", str(d), "--db-path", str(db), *extra],
                       capture_output=True, text=True, env=_env(), cwd=str(d.parent), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _q(db, sql, *a):
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def _lines(p):
    return [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]


def test_every_property_has_its_material_and_the_summaries_agree(tmp_path):
    d = _folder(tmp_path)
    db = tmp_path / "materials.db"
    p = _build(d, db, "--register", "--progress-json")
    orphans = _q(db, "SELECT DISTINCT material_id FROM properties WHERE material_id NOT IN "
                     "(SELECT material_id FROM materials)")
    assert orphans == [], orphans
    recs = {Path(r["file"]).name: r for r in _lines(p) if "i" in r}
    for failed in ("3_dia3_opt_killed_signal9_oom.out", "1_dia_b3lyp_fermi_not_in_interval.out"):
        assert recs[failed]["status"] == "skipped", recs[failed]
    temp = Path(EC_FREQ).name + "_temp.out"
    assert recs[temp]["status"] == "skipped" and recs[temp]["class"] == "skipped"
    assert recs[Path(EC_FREQ).name + ".out"]["status"] == "extracted"
    # one FREQ calculation for EC, recorded with the run's own output
    calcs = _q(db, "SELECT output_file FROM calculations WHERE material_id = 'EC'")
    assert [Path(c[0]).name for c in calcs] == [Path(EC_FREQ).name + ".out"]
    done = [r for r in _lines(p) if r.get("done")][-1]
    n_mat = _q(db, "SELECT count(*) FROM materials")[0][0]
    assert done["materials"] == n_mat == 2 and done["materials_new"] == 2
    # the GUI's summary of the same lines is the CLI's
    from mace.gui import dbbuild
    from mace.utils import db_build
    state = {"files": [dict(r, rel=Path(r["file"]).name) for r in _lines(p) if "i" in r],
             "done": done, "backup": None, "pruned": []}
    gui = dbbuild.summary(state, ["analyze", "x", "--progress-json"])
    assert gui["message"] == db_build.summary_message(done) == done["message"]
    assert gui["new"] == 2 and gui["skipped"] == 3 and gui["failed"] == []

    assert "Database updated: 2 new, 0 changed, 0 failed outputs; 2 materials (2 new); 3 skipped" == gui["message"]
    # the plain run prints the same sentence
    p2 = _build(d, db, "--register")
    # (both outputs extracted again, unchanged: counted as re-extracted)
    assert ("Database updated: 0 new, 0 changed, 2 re-extracted, 0 failed outputs; 2 materials (0 new); "
            "3 skipped") in p2.stdout
    # a skipped output says why, as the dry run does
    temp = Path(EC_FREQ).name + "_temp.out"
    assert f"{temp}: a copy of {Path(EC_FREQ).name}.out" in p2.stdout
    assert "failed or unfinished run with no calculation record" in p2.stdout


def test_an_old_temp_registration_is_kept_and_the_run_output_files_under_it(tmp_path):
    """A database that registered the _temp copy (1.1.5 before this fix):
    the next build files X.out under the same calculation, no new material."""
    d = _folder(tmp_path, [EC_FREQ + "_temp.out", EC_FREQ + "_temp.d12"])
    db = tmp_path / "materials.db"
    _build(d, db, "--register")
    assert [r[0] for r in _q(db, "SELECT material_id FROM materials")] == ["EC"]
    shutil.copyfile(TEST_DATA / (EC_FREQ + ".out"), d / "FREQ" / (Path(EC_FREQ).name + ".out"))
    _build(d, db, "--register")
    assert [r[0] for r in _q(db, "SELECT material_id FROM materials")] == ["EC"]
    assert _q(db, "SELECT count(*) FROM calculations")[0][0] == 1
    assert _q(db, "SELECT DISTINCT material_id FROM properties") == [("EC",)]


def test_untracked_outputs_of_a_build_get_a_materials_row(tmp_path):
    """Register off (the GUI's checkbox): no calculation rows, but the rows
    the build writes are under a material the Database view can open."""
    d = _folder(tmp_path, ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12",
                           "FAILED_QA/3_dia3_opt_killed_signal9_oom.out"])
    db = tmp_path / "materials.db"
    p = _build(d, db, "--progress-json")
    assert _q(db, "SELECT count(*) FROM calculations")[0][0] == 0
    # the failed run too is extracted (as a plain run extracts it), under its own row
    mats = _q(db, "SELECT material_id, formula, dimensionality FROM materials ORDER BY 1")
    assert mats[0] == ("1_dia", "C", "CRYSTAL") and [m[0] for m in mats] == ["1_dia", "3_dia3"]
    assert _q(db, "SELECT DISTINCT material_id FROM properties ORDER BY 1") == [("1_dia",), ("3_dia3",)]
    recs = {Path(r["file"]).name: r for r in _lines(p) if "i" in r}
    assert recs["3_dia3_opt_killed_signal9_oom.out"]["status"] == "extracted"


def test_dry_run_lists_what_the_build_skips(tmp_path):
    d = _folder(tmp_path)
    db = tmp_path / "materials.db"
    p = _build(d, db, "--register", "--dry-run", "--json")
    doc = json.loads(p.stdout)
    cls = {Path(f["file"]).name: f["class"] for f in doc["files"]}
    assert cls["3_dia3_opt_killed_signal9_oom.out"] == "skipped"
    assert cls[Path(EC_FREQ).name + "_temp.out"] == "skipped"
    assert cls[Path(EC_FREQ).name + ".out"] == "new"
    assert doc["counts"]["skipped"] == 3


def test_without_register_a_partial_run_is_extracted_as_a_plain_run_does(tmp_path):
    """Register off: a failed or unfinished output is extracted (under a
    materials row), by the build (--progress-json, as the GUI runs it) as by
    a plain extraction; only --register skips it (it registers none)."""
    d = _folder(tmp_path, ["FAILED_QA/3_dia3_opt_killed_signal9_oom.out"])
    built, plain = tmp_path / "built.db", tmp_path / "plain.db"
    p = _build(d, built, "--progress-json")
    rec = [r for r in _lines(p) if "i" in r][0]
    assert rec["status"] == "extracted" and rec["class"] == "incomplete", rec
    _build(d, plain)
    q = "SELECT material_id, property_name FROM properties ORDER BY 1, 2"
    assert _q(built, q) == _q(plain, q) != []
    assert not _q(built, "SELECT material_id FROM properties WHERE material_id NOT IN "
                         "(SELECT material_id FROM materials)")
    dry = _build(d, tmp_path / "dry.db", "--dry-run", "--json")
    assert [f["class"] for f in json.loads(dry.stdout)["files"]] == ["incomplete"]
