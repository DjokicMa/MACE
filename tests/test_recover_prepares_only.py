"""`mace recover --prepare-only` prepares; it never submits.

(1.1.6: plain `mace recover` now applies the fixes and resubmits, after a
confirmation - tests/test_recover_resubmits.py. --prepare-only keeps the
1.1.5 behavior tested here.)

It writes the fixed deck or the bumped job script and a 'pending' recovery
row, but only the queue manager resubmits. 1.1.5's menu, docs and help said
"fix ... and resubmit them", the action printed a resubmit_delay nobody
honors, and the unsubmitted rows counted toward max_retries. Now the text
says what happens, the action ends with the exact command that submits each
prepared recovery, a recovery prepared but never submitted is reused (not
prepared again) and does not count as a retry.

Through the real mace_cli on a copy of a real out-of-memory run (test/ corpus:
FAILED_QA/3_dia3_opt_killed_signal9_oom.out with the deck and job script of
OPT/3_dia3_opt_rev1).
"""
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
OOM_OUT = TEST_DATA / "FAILED_QA" / "3_dia3_opt_killed_signal9_oom.out"
DECK = TEST_DATA / "OPT" / "3_dia3_opt_rev1.d12"
SCRIPT = TEST_DATA / "OPT" / "3_dia3_opt_rev1.sh"

needs_corpus = pytest.mark.skipif(not (OOM_OUT.exists() and DECK.exists() and SCRIPT.exists()),
                                  reason="needs the test/ corpus")


def _seed(work):
    work.mkdir()
    shutil.copy(DECK, work / DECK.name)
    shutil.copy(SCRIPT, work / SCRIPT.name)
    shutil.copy(OOM_OUT, work / "3_dia3_opt_rev1.out")
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from mace.database.materials import MaterialDatabase
db = MaterialDatabase({str(work / 'materials.db')!r})
db.create_material("3_dia3", "C")
cid = db.create_calculation(material_id="3_dia3", calc_type="OPT", input_file={str(work / DECK.name)!r},
                            work_dir={str(work)!r})
db.update_calculation_status(cid, "failed", slurm_job_id="12345",
                             output_file={str(work / '3_dia3_opt_rev1.out')!r},
                             error_type="memory_error", error_message="KILLED BY SIGNAL: 9 (Killed)")
print(cid)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=_env(work))
    assert r.returncode == 0, r.stderr
    return r.stdout.split()[-1]


def _env(work):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH") and not k.startswith("SLURM_")}
    fake = work.parent / "fakebin"
    fake.mkdir(exist_ok=True)
    for tool in ("sbatch", "squeue", "sacct", "scontrol"):
        p = fake / tool
        p.write_text(f'#!/bin/bash\necho "{tool} $*" >> "{work.parent}/slurm_calls.log"\nexit 1\n')
        p.chmod(0o755)
    env["PATH"] = f"{fake}:{env.get('PATH', '/usr/bin:/bin')}"
    return env


def _recover(work):
    return subprocess.run([sys.executable, str(MACE_CLI), "recover", "--action", "recover",
                           "--prepare-only", "--db", str(work / "materials.db")],
                          cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
                          env=_env(work))


def _rows(work):
    c = sqlite3.connect(work / "materials.db")
    try:
        return c.execute("SELECT calc_id, status, slurm_job_id, settings_json FROM calculations "
                         "ORDER BY rowid").fetchall()
    finally:
        c.close()


def _retry_count(work, calc_id):
    code = (f"import sys, contextlib, io; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    from mace.recovery.recovery import ErrorRecoveryEngine\n"
            f"    e = ErrorRecoveryEngine({str(work / 'materials.db')!r})\n"
            f"print(e.get_retry_count({calc_id!r}))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=work, capture_output=True, text=True, timeout=300,
                       env=_env(work))
    assert r.returncode == 0, r.stderr
    return int(r.stdout.split()[-1])


@needs_corpus
def test_recover_prepares_prints_the_command_and_submits_nothing(tmp_path):
    work = tmp_path / "rec"
    cid = _seed(work)
    r = _recover(work)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    scripts = sorted(work.glob("3_dia3_opt_rev1_recovery_*.sh"))
    assert len(scripts) == 1, out
    assert "#SBATCH --mem-per-cpu=7GB" in scripts[0].read_text()
    # honest text, the exact next command, no promised delay
    assert "it submits nothing" in out
    assert f"  cd {work} && sbatch {scripts[0].name}" in out.splitlines()
    assert "delayed" not in out
    assert not (tmp_path / "slurm_calls.log").exists()          # never submitted
    rows = _rows(work)
    assert [(r[1], r[2]) for r in rows] == [("failed", "12345"), ("pending", None)]
    assert _retry_count(work, cid) == 0                           # unsubmitted: no retry spent

    # again: the prepared files are reused, not prepared a second time
    r2 = _recover(work)
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2
    assert "already prepared" in out2
    assert sorted(work.glob("3_dia3_opt_rev1_recovery_*.sh")) == scripts
    assert len(_rows(work)) == 2
    assert f"  cd {work} && sbatch {scripts[0].name}" in out2.splitlines()

    # once the prepared row is submitted it is a retry
    c = sqlite3.connect(work / "materials.db")
    c.execute("UPDATE calculations SET status='submitted', slurm_job_id='777' WHERE status='pending'")
    c.commit()
    c.close()
    assert _retry_count(work, cid) == 1


def test_help_names_prepare_only():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "PYTHONPATH")}
    h = subprocess.run([sys.executable, str(MACE_CLI), "recover", "--help"], capture_output=True, text=True,
                       timeout=120, env=env, stdin=subprocess.DEVNULL)
    text = " ".join((h.stdout + h.stderr).split())
    assert "--prepare-only Only write the fixed inputs and job scripts and print the commands that submit them" in text


def test_next_command_shapes(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine as E
    deck = tmp_path / "a b.d12"
    assert E.next_command({"work_dir": str(tmp_path), "fixed_input_file": deck,
                           "fixed_job_script": None}) == f"cd {tmp_path} && mace submit 'a b.d12'"
    # the deck's own job script: mace submit keeps it
    assert E.next_command({"work_dir": str(tmp_path), "fixed_input_file": deck,
                           "fixed_job_script": deck.with_suffix(".sh")}) == f"cd {tmp_path} && mace submit 'a b.d12'"
    assert E.next_command({"work_dir": str(tmp_path), "fixed_input_file": deck,
                           "fixed_job_script": tmp_path / "x_recovery_1.sh"}) == \
        f"cd {tmp_path} && sbatch x_recovery_1.sh"
