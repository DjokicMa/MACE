"""The queue lock is exclusive for as long as its holder runs, whatever
lock_timeout says, and one lock covers every callback mode.

Before 1.1.5 a waiter treated a lock older than lock_timeout (300 s) as
expired, deleted it and took its own; the first holder, still running,
then deleted the new holder's lock file on release. A long monitoring
cycle could so run beside a completion callback, and both resubmitted
through recovery. The lock was also named per mode, so a completion
callback and `mace manager --callback-mode full_check` never excluded each
other at all. Real processes contend here (as on a cluster: separate
callbacks), with lock_timeout scaled down to 1 s."""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from mace.queue.queue_lock_manager import QueueLockManager
    lock_dir, name, hold, log, who = Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6]
    m = QueueLockManager(lock_dir=lock_dir, lock_timeout=1)
    def body():
        with open(log, 'a') as f:
            f.write(f"{time.time():.3f} {who} in\\n")
        print('held', flush=True)
        time.sleep(hold)
        with open(log, 'a') as f:
            f.write(f"{time.time():.3f} {who} out\\n")
    m.with_lock(name, body, timeout=60)
    """)


def _start(lock_dir, name, hold, log, who, kind=""):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), MACE_QUEUE_LOCK=kind)
    return subprocess.Popen([sys.executable, "-c", HOLDER, str(REPO_ROOT), str(lock_dir), name,
                             str(hold), str(log), who],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _intervals(log):
    spans, open_ = [], {}
    for line in Path(log).read_text().splitlines():
        t, who, what = line.split()
        if what == "in":
            open_[who] = float(t)
        else:
            spans.append((open_.pop(who), float(t), who))
    assert not open_, open_
    return sorted(spans)


def _assert_no_overlap(spans):
    for (a0, a1, a), (b0, b1, b) in zip(spans, spans[1:]):
        assert b0 >= a1 - 0.01, f"{a} {a0:.2f}-{a1:.2f} overlaps {b} {b0:.2f}-{b1:.2f}"


# "token": the owner-token fallback for filesystems without working locks
KINDS = pytest.mark.parametrize("kind", ["", "token"])


@KINDS
def test_a_holder_past_lock_timeout_keeps_the_lock(tmp_path, kind):
    """A holds 4 s (lock_timeout 1 s); B and then C wait. Nobody runs
    beside anybody: B does not break in once A passes lock_timeout, and A's
    release does not delete B's lock for C to walk in."""
    log = tmp_path / "log"
    a = _start(tmp_path / "locks", "queue_state", 4, log, "A", kind)
    assert a.stdout.readline().strip() == "held", a.stderr.read()
    b = _start(tmp_path / "locks", "queue_state", 3, log, "B", kind)
    time.sleep(2.5)
    c = _start(tmp_path / "locks", "queue_state", 1, log, "C", kind)
    for p in (a, b, c):
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
    spans = _intervals(log)
    assert len(spans) == 3
    _assert_no_overlap(spans)


@KINDS
def test_a_killed_holder_frees_the_lock(tmp_path, kind):
    """No expiry is needed: the lock goes with the process that held it."""
    log = tmp_path / "log"
    a = _start(tmp_path / "locks", "queue_state", 600, log, "A", kind)
    assert a.stdout.readline().strip() == "held", a.stderr.read()
    a.send_signal(signal.SIGKILL)
    a.wait()
    t0 = time.time()
    b = _start(tmp_path / "locks", "queue_state", 0, log, "B", kind)
    assert b.stdout.readline().strip() == "held", b.stderr.read()
    assert time.time() - t0 < 10
    b.wait(timeout=30)


def test_the_lock_file_stays_put(tmp_path):
    """Release never deletes the lock file: deleting it by path is what let
    a late release remove another holder's lock."""
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue.queue_lock_manager import QueueLockManager
    m = QueueLockManager(lock_dir=tmp_path, lock_timeout=1)
    assert m.acquire_lock("queue_state", timeout=1)
    m.release_lock("queue_state")
    assert (tmp_path / "queue_state.lock").exists()
    assert m.acquire_lock("queue_state", timeout=1)
    m.release_lock("queue_state")


@pytest.mark.parametrize("mode", ["completion", "status_check", "full_check", "submit_new",
                                  "early_failure"])
def test_every_callback_mode_takes_the_one_queue_lock(tmp_path, mode):
    """Completion callbacks and `mace manager` sweeps both resubmit through
    recovery: they take the same lock."""
    from array_helpers import callback
    from fake_slurm import FakeSlurm
    from mace.queue.queue_lock_manager import QUEUE_STATE_LOCK
    fake = FakeSlurm(tmp_path / "slurm")
    work = tmp_path / "w"
    work.mkdir()
    db = work / "materials.db"
    log = tmp_path / "log"
    holder = _start(work / ".queue_locks", QUEUE_STATE_LOCK, 120, log, "H")
    try:
        assert holder.stdout.readline().strip() == "held", holder.stderr.read()
        p = callback(fake.env, work, db, mode=mode, extra_env={"MACE_ARRAY_LOCK_SCALE": "0.05"})
    finally:
        holder.kill()
        holder.wait()
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert f"Could not acquire lock for {mode} callback" in out, out[-3000:]
    assert "=== Queue Manager Callback" not in out, out[-3000:]


def test_a_lock_file_this_user_cannot_write_falls_back_to_the_token_lock(tmp_path, capsys):
    """In a folder shared by a group the persistent lock file belongs to
    whoever created it; it was created 0644, so everyone else's sweep failed
    to open it (PermissionError) instead of waiting for or taking the lock."""
    if os.geteuid() == 0:
        pytest.skip("root can write a read-only file")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mace.queue.queue_lock_manager import QueueLockManager
    locks = tmp_path / ".queue_locks"
    locks.mkdir()
    blocked = locks / "queue_state.lock"
    blocked.write_text("")
    blocked.chmod(0o444)                       # another user's 0644 file, as seen by this one
    m = QueueLockManager(locks)
    assert m.acquire_lock("queue_state", timeout=2)
    assert "queue_state.lock" in capsys.readouterr().out
    m.release_lock("queue_state")


def test_the_lock_file_is_created_group_writable_under_the_umask(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mace.queue.queue_lock_manager import QueueLockManager
    old = os.umask(0o002)
    try:
        m = QueueLockManager(tmp_path / "l")
        assert m.acquire_lock("queue_state", timeout=2)
        m.release_lock("queue_state")
    finally:
        os.umask(old)
    assert (tmp_path / "l" / "queue_state.lock").stat().st_mode & 0o777 == 0o664
