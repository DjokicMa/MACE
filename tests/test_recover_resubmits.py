"""`mace recover` classifies failed tracked runs, shows the plan, and after a
confirmation applies the fix and resubmits through the queue manager's own
recovery path.

Real `mace_cli` subprocesses, tests/fake_slurm.py on PATH, and copies of the
real failed outputs in the test/ corpus (FAILED_QA): an OOM kill of an OPT
(3_dia3), an OOM kill of an SP (T7), an SCF stop (diamond, FERMI ENERGY NOT
IN INTERVAL) and a geometry stop (NEIGHB, atoms too close; no automatic
recovery).
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from array_helpers import callback, rows
from conftest import REPO_ROOT, find_data
from fake_slurm import FakeSlurm

MACE_CLI = REPO_ROOT / "mace_cli"

# (case, failed output, deck, job script, calc type, what the fix must leave in the submitted job)
CASES = {
    "oom_opt": ("FAILED_QA/3_dia3_opt_killed_signal9_oom.out", "OPT/3_dia3_opt_rev1.d12",
                "OPT/3_dia3_opt_rev1.sh", "OPT", "memory_error"),
    "oom_sp": ("FAILED_QA/T7_sp_killed_signal9_scf_converged.out",
               "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12",
               "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.sh", "SP", "memory_error"),
    "scf": ("FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out",
            "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12", None, "SP", "convergence_error"),
    "close": ("FAILED_QA/3_dia3_neighb_atoms_too_close.out", "OPT/3_dia3_opt_rev1.d12", None, "OPT",
              "geometry_error"),
}


def _env(tmp_path, fake, **extra):
    env = dict(fake.env)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=str(home), USER="tester", MACE_HOSTNAME="dev-amd20.hpcc.msu.edu")
    for k in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM", "NO_COLOR", "MACE_RECOVER_MAX_AGE_DAYS"):
        env.pop(k, None)
    env.update(extra)
    return env


def _seed(work, case, name=None, job_id="12345", db=None, material=None, status="failed", settings=None):
    """A failed tracked calculation: the corpus deck, its job script and its
    real failed output, copied into ``work``."""
    out, deck, script, calc_type, _ = CASES[case]
    work.mkdir(parents=True, exist_ok=True)
    name = name or Path(deck).stem
    shutil.copy(find_data(deck), work / f"{name}.d12")
    if script:
        text = find_data(script).read_text().replace(Path(script).stem, name)
        (work / f"{name}.sh").write_text(text)
    shutil.copy(find_data(out), work / f"{name}.out")
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    mdb = MaterialDatabase(str(db or work / "materials.db"))
    material = material or name
    if not mdb.get_material(material):
        mdb.create_material(material, "C")
    cid = mdb.create_calculation(material_id=material, calc_type=calc_type, input_file=str(work / f"{name}.d12"),
                                 work_dir=str(work), settings=settings)
    mdb.update_calculation_status(cid, status, slurm_job_id=job_id, output_file=str(work / f"{name}.out"))
    return cid


def _recover(env, cwd, *args, stdin="", timeout=600):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "recover", *args], cwd=str(cwd),
                          env=env, input=stdin, capture_output=True, text=True, timeout=timeout)


def _ok(p):
    assert p.returncode == 0, p.stdout[-6000:] + p.stderr[-3000:]
    return p.stdout


def _row(db, cid):
    return rows(db)[cid]


def _snapshot(folder):
    return {str(p.relative_to(folder)): (p.stat().st_mtime_ns, p.read_bytes())
            for p in sorted(folder.rglob("*")) if p.is_file() and ".queue_locks" not in p.parts}


@pytest.mark.parametrize("case", ["oom_opt", "oom_sp", "scf"])
def test_each_recoverable_corpus_class_is_fixed_and_resubmitted(tmp_path, case):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, case)
    out = _ok(_recover(env, work, "--yes"))
    expected = CASES[case][4]
    klass = {"memory_error": "oom", "convergence_error": "scf"}[expected]
    assert re.search(rf"{cid}\s+{klass} \({expected}\)\s+0/2, total 0/3", out), out
    assert "Resubmitted 1 of 1 calculation(s)." in out, out
    (sb,) = fake.calls("sbatch")
    db = rows(work / "materials.db")
    parent = db[cid]
    new = [r for r in db.values() if r["calc_id"] != cid]
    assert parent["status"] == "resubmitted" and parent["error_type"] == expected
    assert len(new) == 1 and new[0]["status"] == "submitted" and new[0]["slurm_job_id"] == "5000"
    assert new[0]["job_script"] == sb["script"]
    if expected == "memory_error":
        # the bumped copy of the job's own script, per-cpu kept
        assert re.search(r"_recovery_\d{8}_\d{6}\.sh$", sb["script"])
        assert "#SBATCH --mem-per-cpu=7GB" in sb["script_text"]
        assert "#SBATCH --mem-per-cpu=5G\n" in (work / f"{Path(CASES[case][1]).stem}.sh").read_text()
    else:
        # a new deck with the SCF settings raised, submitted with a script written from the template
        deck = Path(new[0]["input_file"])
        assert re.search(r"_recovery_\d{8}_\d{6}\.d12$", deck.name)
        assert "MAXCYCLE\n2600\n" in deck.read_text() and "FMIXING\n20\n" in deck.read_text()
        assert f"export JOB={deck.stem}" in sb["script_text"] or deck.stem in sb["script_text"]
    # the next run sees a lineage with one recovery spent: nothing failed is left
    again = _ok(_recover(env, work, "--yes"))
    assert "No failed calculations to recover." in again
    assert len(fake.calls("sbatch")) == 1


def test_non_recoverable_class_is_reported_with_its_reason(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "close")
    out = _ok(_recover(env, work, "--yes"))
    assert re.search(rf"{cid}\s+atoms_too_close \(geometry_error\)\s+-\s+no automatic recovery: the geometry "
                     r"has to be fixed \(or SMALLDIST set\) by a person\s+\[skip: needs a person\]", out), out
    assert "0 to resubmit; 1 without automatic recovery." in out
    assert not fake.calls("sbatch")
    assert _row(work / "materials.db", cid)["status"] == "failed"


def test_dry_run_shows_the_diffs_and_the_sbatch_command_and_changes_nothing(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    _seed(work, "oom_opt", job_id="12345")
    _seed(work, "scf", name="dia_sp", job_id="12346")
    before = _snapshot(work)
    out = _ok(_recover(env, work, "--dry-run"))
    assert "Dry run: nothing is changed or submitted." in out
    # the OOM fix: a new job script, with the memory line diffed
    assert re.search(rf"would write {re.escape(str(work))}/3_dia3_opt_rev1_recovery_\d{{8}}_\d{{6}}\.sh", out), out
    assert "      -#SBATCH --mem-per-cpu=5G\n      +#SBATCH --mem-per-cpu=7GB\n" in out
    assert re.search(rf"would run: cd {re.escape(str(work))} && sbatch {re.escape(str(work))}/"
                     r"3_dia3_opt_rev1_recovery_\d{8}_\d{6}\.sh", out), out
    # the SCF fix: a new deck diffed against the old, its script from the template, then sbatch
    assert re.search(rf"would write {re.escape(str(work))}/dia_sp_recovery_\d{{8}}_\d{{6}}\.d12", out), out
    assert "       MAXCYCLE\n      -1600\n      +2600\n       FMIXING\n      -30\n      +20\n" in out
    assert re.search(r"would run: cd \S+ && bash \S+/mace/submission/submitcrystal23\.sh "
                     r"dia_sp_recovery_\d{8}_\d{6}   \(writes dia_sp_recovery_\d{8}_\d{6}\.sh\)", out), out
    assert "/tmp/mace-recover-preview" not in out and "mace-recover-preview" not in out
    # nothing written, nothing submitted, the rows untouched
    assert _snapshot(work) == before
    assert not fake.calls("sbatch")
    assert {r["status"] for r in rows(work / "materials.db").values()} == {"failed"}
    # --json carries the same text and the plan
    p = _recover(env, work, "--dry-run", "--json")
    doc = json.loads(_ok(p))
    assert doc["dry_run"] is True and len(doc["entries"]) == 2
    assert all(e["state"] == "ready" and e["preview"]["commands"] for e in doc["entries"])
    assert "Dry run: nothing is changed or submitted." in doc["text"]


def test_scoping_flags_and_no_answer_submits_nothing(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    db = work / "materials.db"
    a = _seed(work / "a", "oom_opt", job_id="1", db=db)
    b = _seed(work / "b", "scf", name="dia_sp", job_id="2", db=db)
    c = _seed(work / "b", "close", name="close", job_id="3", db=db)

    def listed(*args):
        out = _ok(_recover(env, work, *args))      # no --yes, stdin empty: nothing submitted
        return {x for x in (a, b, c) if x in out}, out

    got, out = listed()
    assert got == {a, b, c}
    assert "Nothing submitted. Run again with --yes" in out
    assert listed("--calc-id", a)[0] == {a}
    assert listed("--calc-id", f"{a},{c}")[0] == {a, c}
    assert listed("--errors", "scf")[0] == {b}
    assert listed("--errors", "memory_error,atoms_too_close")[0] == {a, c}
    assert listed("--dir", str(work / "b"), "--db", str(db))[0] == {b, c}
    assert listed("--dir", str(work / "a"), "--errors", "scf", "--db", str(db))[0] == set()
    # the answer No (or no answer at all) changes nothing
    _ok(_recover(env, work, stdin="n\n"))
    assert not fake.calls("sbatch")
    assert {r["status"] for r in rows(db).values()} == {"failed"}
    # a scoped run resubmits only its calculation
    _ok(_recover(env, work, "--calc-id", b, "--yes"))
    assert len(fake.calls("sbatch")) == 1
    st = {k: v["status"] for k, v in rows(db).items()}
    assert st[a] == "failed" and st[b] == "resubmitted" and st[c] == "failed"


def test_a_job_already_in_the_queue_is_not_submitted_again(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt")
    # a recovery of this deck (by another `mace recover`, or by hand) waits in the queue
    fake.add_record(id="7001", name="3_dia3_opt_rev1_recovery_20261001_120000", state="PENDING", user="tester",
                    workdir=str(work))
    out = _ok(_recover(env, work, "--yes"))
    assert "skip: job 7001 (3_dia3_opt_rev1_recovery_20261001_120000) of this deck is already in the queue" in out
    assert not fake.calls("sbatch")
    # its own job still listed (squeue lags the database)
    fake.remove_record("7001")
    fake.add_record(id="12345", name="other_name", state="RUNNING", user="tester", workdir=str(work))
    out = _ok(_recover(env, work, "--yes"))
    assert "skip: its job 12345 is still in the queue (RUNNING)" in out
    assert not fake.calls("sbatch")
    assert _row(work / "materials.db", cid)["status"] == "failed"


def test_two_concurrent_recovers_submit_once(tmp_path):
    """Both see the failed row; the queue lock and the re-check under it let
    exactly one of them resubmit it."""
    fake = FakeSlurm(tmp_path / "fs", {"sbatch_delay": 1.5})
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt")
    argv = [sys.executable, str(MACE_CLI), "--no-banner", "recover", "--yes"]
    procs = [subprocess.Popen(argv, cwd=str(work), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=600)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert len(fake.calls("sbatch")) == 1, outs
    joined = "\n".join(outs)
    assert joined.count("Resubmitted 1 of 1") == 1, joined
    assert re.search(r"Not submitted: (its status is now 'resubmitted'|a newer run is submitted|"
                     r"job 5000 \S+ of this deck is already in the queue)", joined), joined
    db = rows(work / "materials.db")
    assert db[cid]["status"] == "resubmitted" and len(db) == 2


def test_retry_limit_is_honored_and_reported(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt", job_id="3")
    # two earlier recoveries of this OPT already ran out of memory (recovery_config max_retries 2)
    db = work / "materials.db"
    con = sqlite3.connect(db)
    for i, jid in enumerate(("1", "2")):
        con.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, slurm_job_id, created_at, "
                    "error_type, input_file, work_dir) VALUES (?, ?, 'OPT', 'resubmitted', ?, ?, 'memory_error', ?, ?)",
                    (f"old{i}", "3_dia3_opt_rev1", jid, f"2026-01-0{i + 1}T00:00:00", str(work / "x.d12"), str(work)))
    con.commit()
    con.close()
    out = _ok(_recover(env, work, "--yes"))
    assert re.search(rf"{cid}\s+oom \(memory_error\)\s+2/2, total 2/3\s+.*\[skip: retry limit reached\]", out), out
    assert "0 to resubmit; 1 at the retry limit." in out
    assert not fake.calls("sbatch")
    # a lower lineage limit binds first
    out = _ok(_recover(env, work, "--max-recovery-attempts", "1", "--dry-run"))
    assert "2/2, total 2/1" in out and "retry limit reached" in out


def test_a_stale_job_is_resubmitted_only_after_its_own_confirmation(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"sacct": {"12345": [["12345", "OUT_OF_MEMORY", "2026-01-02T03:04:05"]]}})
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt")
    out = _ok(_recover(env, work, "--yes"))
    assert re.search(r"\[ask: ended 2026-01-02 03:04 \(over 3 days ago\)\]", out), out
    assert "0 to resubmit; 1 ended over 3 days ago (asked separately)." in out
    assert "Resubmit them too? [y/N] n (--yes)" in out and "add --include-stale" in out
    assert not fake.calls("sbatch")
    # asked and answered No
    _ok(_recover(env, work, stdin="n\n"))
    assert not fake.calls("sbatch")
    # a longer limit: not stale
    out = _ok(_recover(dict(env, MACE_RECOVER_MAX_AGE_DAYS="100000"), work, "--dry-run"))
    assert "[resubmit]" in out
    # the explicit Yes resubmits it
    out = _ok(_recover(env, work, stdin="y\n"))
    assert "Resubmitted 1 of 1" in out, out
    assert len(fake.calls("sbatch")) == 1
    assert _row(work / "materials.db", cid)["status"] == "resubmitted"


def test_include_stale_with_yes(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"sacct": {"12345": [["12345", "OUT_OF_MEMORY", "2026-01-02T03:04:05"]]}})
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    out = _ok(_recover(env, work, "--yes", "--include-stale"))
    assert "Resubmit them too? [y/N] y (--yes)" in out and "Resubmitted 1 of 1" in out
    assert len(fake.calls("sbatch")) == 1


def test_the_existing_job_script_is_reused_unless_overwrite_sh(tmp_path):
    """A fix that leaves the job script alone (disk cleanup) submits the
    <job>.sh beside the deck as it is: a script the user edited is never
    written again from the template without --overwrite-sh."""
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt")
    sh = work / "3_dia3_opt_rev1.sh"
    sh.write_text(sh.read_text() + "# my own edit\n")
    con = sqlite3.connect(work / "materials.db")
    con.execute("UPDATE calculations SET error_type='disk_space_error'")
    con.commit()
    con.close()
    out_file = work / "3_dia3_opt_rev1.out"
    out_file.write_text(" forrtl: severe (38): error during write, unit 9, file fort.9\n")
    out = _ok(_recover(env, work, "--dry-run"))
    assert "disk (disk_space_error)" in out, out
    assert f"sbatch {sh}   (the existing job script, reused)" in out
    _ok(_recover(env, work, "--yes"))
    (sb,) = fake.calls("sbatch")
    assert sb["script"] == str(sh) and "# my own edit" in sb["script_text"]
    assert "# my own edit" in sh.read_text()
    assert _row(work / "materials.db", cid)["status"] == "resubmitted"


def test_a_recovered_workflow_step_keeps_its_workflow_and_the_chain_goes_on(tmp_path):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    d = tmp_path / "decks"
    d.mkdir()
    shutil.copy(find_data("OPT/1_dia_opt_rev1.d12"), d / "dia0.d12")
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "--progress", "opt_sp", "decks"],
                       cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    _ok(p)
    (db,) = d.glob(".mace_context_*/materials.db")
    (opt,) = rows(db).values()
    wf = json.loads(opt["settings_json"])["workflow_id"]
    # the deck runs in its step folder (workflow_outputs/<id>/step_001_OPT/dia0/dia0_opt.d12)
    job_out = Path(opt["input_file"]).with_suffix(".out")
    assert job_out.parent == d / "workflow_outputs" / wf / "step_001_OPT" / "dia0"
    # the OPT ran out of memory
    shutil.copy(find_data("FAILED_QA/3_dia3_opt_killed_signal9_oom.out"), job_out)
    fake.remove_record(opt["slurm_job_id"])
    con = sqlite3.connect(db)
    con.execute("UPDATE calculations SET status='failed', output_file=? WHERE calc_id=?",
                (str(job_out), opt["calc_id"]))
    con.commit()
    con.close()
    out = _ok(_recover(env, tmp_path, "--db", str(db), "--yes"))
    assert f"keeps workflow" not in out          # (the dry run says it; the run just resubmits)
    r = rows(db)
    new = [x for x in r.values() if x["calc_id"] != opt["calc_id"]]
    assert len(new) == 1 and new[0]["calc_type"] == "OPT" and new[0]["status"] == "submitted", out
    s = json.loads(new[0]["settings_json"])
    assert s["workflow_id"] == wf and s["workflow_step"] == 1
    assert r[opt["calc_id"]]["status"] == "resubmitted"
    dry = _ok(_recover(env, tmp_path, "--db", str(db), "--dry-run"))
    assert "No failed calculations to recover." in dry
    # the recovered OPT completes: the job callback goes on to the plan's SP
    shutil.copy(find_data("OPT/1_dia_opt_rev1.out"), job_out)
    fake.remove_record(new[0]["slurm_job_id"])
    fake.update(sacct={new[0]["slurm_job_id"]: [[new[0]["slurm_job_id"], "COMPLETED", "2026-10-14T12:00:00"]]})
    cb = callback(env, job_out.parent, db, mode="completion")
    assert cb.returncode == 0, cb.stdout[-4000:] + cb.stderr[-2000:]
    r = rows(db)
    assert r[new[0]["calc_id"]]["status"] == "completed", cb.stdout[-4000:]
    sp = [x for x in r.values() if x["calc_type"].startswith("SP")]
    assert len(sp) == 1 and sp[0]["status"] == "submitted", cb.stdout[-4000:]
    assert json.loads(sp[0]["settings_json"])["workflow_id"] == wf


def test_dry_run_of_a_workflow_step_names_its_workflow(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    _seed(work, "oom_opt", settings={"workflow_id": "workflow_20261014_000000_abcd", "workflow_step": 1})
    out = _ok(_recover(env, work, "--dry-run"))
    assert "keeps workflow workflow_20261014_000000_abcd: the workflow goes on when the new run completes" in out


def test_help_describes_the_resubmission():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "PYTHONPATH")}
    h = subprocess.run([sys.executable, str(MACE_CLI), "recover", "--help"], capture_output=True, text=True,
                       timeout=120, env=env, stdin=subprocess.DEVNULL)
    text = " ".join((h.stdout + h.stderr).split())
    for flag in ("--dry-run", "--yes", "--calc-id", "--dir", "--errors", "--include-stale", "--prepare-only"):
        assert flag in text, flag
    assert "resubmits them through the queue manager" in text


def test_dry_run_copies_next_to_the_database_not_into_the_system_temp(tmp_path):
    """The dry run's copies of the deck, job script and output go to a
    folder next to the database ($TMPDIR when it is set), removed after."""
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    for k in ("TMPDIR", "TEMP", "TMP"):
        env.pop(k, None)
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    log = tmp_path / "copies.log"
    code = ("import runpy, shutil, sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "orig = shutil.copy2\n"
            "def cp(src, dst, *a, **k):\n"
            f"    open({str(log)!r}, 'a').write(str(dst) + '\\n')\n"
            "    return orig(src, dst, *a, **k)\n"
            "shutil.copy2 = cp\n"
            f"sys.argv = [{str(MACE_CLI)!r}, '--no-banner', 'recover', '--dry-run']\n"
            f"runpy.run_path({str(MACE_CLI)!r}, run_name='__main__')\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=str(work), env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0 and "Dry run" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
    dests = [Path(x) for x in log.read_text().split()]
    assert dests and all(str(d).startswith(str(work) + os.sep) for d in dests), dests
    assert not list(work.glob(".mace-recover-preview-*"))          # removed after

    env["TMPDIR"] = str(tmp_path / "mytmp")
    (tmp_path / "mytmp").mkdir()
    log.write_text("")
    p = subprocess.run([sys.executable, "-c", code], cwd=str(work), env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    dests = [Path(x) for x in log.read_text().split()]
    assert dests and all(str(d).startswith(str(tmp_path / "mytmp") + os.sep) for d in dests), dests


def test_json_without_yes_or_dry_run_asks_nothing_and_prints_pure_json(tmp_path):
    """--json answers no questions: without --yes or --dry-run it refuses (exit 2)
    with a JSON document on stdout and the reason on stderr; nothing is submitted."""
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt")
    p = _recover(env, work, "--json", stdin="y\n")
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    doc = json.loads(p.stdout)
    assert doc["ok"] is False and "--yes" in doc["error"] and "--dry-run" in doc["error"]
    assert "--yes" in p.stderr and "[y/N]" not in p.stdout + p.stderr
    assert not fake.calls("sbatch") and _row(work / "materials.db", cid)["status"] == "failed"
    # with --yes it is one JSON document as well
    p = _recover(env, work, "--json", "--yes")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert json.loads(p.stdout)["results"][0]["status"] == "submitted"


def test_a_read_only_database_is_reported_without_a_traceback(tmp_path):
    import os
    import stat
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    modes = {p: p.stat().st_mode for p in work.iterdir()}
    for p in work.iterdir():
        os.chmod(p, 0o444)
    os.chmod(work, 0o555)
    try:
        for flag in ("--dry-run", "--yes"):
            p = _recover(env, work, flag)
            assert p.returncode == 1, p.stdout[-2000:] + p.stderr[-2000:]
            assert "Traceback" not in p.stdout + p.stderr
            assert "read-only" in p.stdout + p.stderr
        assert not fake.calls("sbatch")
    finally:
        os.chmod(work, 0o755)
        for p, m in modes.items():
            os.chmod(p, stat.S_IMODE(m))


def test_a_busy_queue_lock_gives_a_nonzero_exit(tmp_path, monkeypatch):
    """The worker timed out on the queue lock and submitted nothing: exit 1, so a
    script running mace recover --yes can retry."""
    from mace.recovery import recovery, resubmit
    fake = FakeSlurm(tmp_path / "fs")
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    for k, v in _env(tmp_path, fake).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(work)
    monkeypatch.setattr(resubmit, "run_worker",
                        lambda e, calc, args: {"status": "busy", "why": "another MACE process held the queue lock"})
    monkeypatch.setattr(sys, "argv", ["recovery.py", "--yes"])
    with pytest.raises(SystemExit) as ex:
        recovery.main()
    assert ex.value.code == 1


def test_dry_run_reads_the_scratch_record_the_job_left(tmp_path, monkeypatch):
    """The OPT-timeout preview reads <submit dir>/.<JOB>.scratch (the scratch the
    job really used) as the real run does, not only the script's scratch line."""
    from mace.recovery.resubmit import Planner
    import argparse
    fake = FakeSlurm(tmp_path / "fs")
    for k, v in _env(tmp_path, fake).items():
        monkeypatch.setenv(k, v)
    work = tmp_path / "w"
    cid = _seed(work, "oom_opt", name="job")
    text = find_data("OPT/1_dia_opt_rev1.out").read_text(errors="ignore")
    lines = text.splitlines(keepends=True)
    cut = next(i for i, l in enumerate(lines) if "OPTIMIZATION - POINT    3" in l)
    (work / "job.out").write_text("".join(lines[:cut]))
    shutil.copy(find_data("OPT/1_dia_opt_rev1.d12"), work / "job.d12")
    used = tmp_path / "fallback_scratch" / "job"
    used.mkdir(parents=True)
    (used / "OPTINFO.DAT").write_text("x")
    (work / ".job.scratch").write_text(str(used) + "\n")
    empty = tmp_path / "login_scratch"                 # what $SCRATCH says here: no OPTINFO.DAT
    (empty / "crys23" / "job").mkdir(parents=True)
    monkeypatch.setenv("SCRATCH", str(empty))
    args = argparse.Namespace(max_jobs=250, reserve=30, max_recovery_attempts=3, config=None)
    planner = Planner(work / "materials.db", args)
    calc = planner.db.get_calculation(cid)
    entry = {"handler": "timeout_handler", "error_type": "timeout_error", "message": "time limit"}
    prev = planner.preview(calc, entry)
    notes = "\n".join(prev["notes"])
    assert "Added RESTART" in notes, notes


def test_recover_uses_the_queue_budget_not_115s_250_limit(tmp_path):
    """1.1.5's recover refused every resubmission once 220 (250 - 30) jobs
    were queued; 1.1.6 keeps the user-wide budget, as mace manager does."""
    fake = FakeSlurm(tmp_path / "fs")
    for i in range(230):
        fake.add_record(id=str(900000 + i), state="RUNNING", user="tester", name=f"other{i}")
    (tmp_path / "site.toml").write_text("[queue]\nbudget = 900\n")
    env = _env(tmp_path, fake, MACE_SITE_CONFIG=str(tmp_path / "site.toml"),
               MACE_QUEUE_STATE_DIR=str(tmp_path / "qs"))
    work = tmp_path / "w"
    _seed(work, "oom_opt")
    out = _ok(_recover(env, work, "--yes"))
    assert "Queue nearly full" not in out and "Resubmitted 1 of 1 calculation(s)." in out, out
    # an explicit --max-jobs still throttles
    work2 = tmp_path / "w2"
    _seed(work2, "oom_opt")
    out = _recover(env, work2, "--yes", "--max-jobs", "240", "--reserve", "20").stdout
    assert "Queue nearly full (231/240)" in out, out


def test_recover_reserve_without_max_jobs_warns_that_it_is_ignored(monkeypatch, capsys):
    """As mace manager does: in 1.1.5 --reserve alone lowered the throttle."""
    from types import SimpleNamespace

    from mace.queue import budget
    from mace.recovery import resubmit
    monkeypatch.setattr(budget, "resolve", lambda quiet=True: SimpleNamespace(budget=10, reserve=2))
    assert resubmit.queue_limits(None, 8) == (12, 2)
    assert "--reserve is ignored without --max-jobs" in capsys.readouterr().err
    assert resubmit.queue_limits(None, None) == (12, 2)
    assert capsys.readouterr().err == ""
