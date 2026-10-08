"""F5: the throttled file scan and its cleanup suggestions (never deletes)."""
import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import REPO_ROOT, TEST_DATA
from mace.cluster import config, filescan


def _tree(root: Path, n_files=30000, per_dir=500):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        d = root / f"d{i // per_dir:03d}" / ("deep" if i % 3 == 0 else "")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"f{i}.txt").write_bytes(b"x" * (i % 7))
    return root


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    free = shutil.disk_usage("/tmp").free
    if free < 2 * 1024 ** 3:
        pytest.skip("less than 2 GB free in /tmp")
    root = _tree(tmp_path_factory.mktemp("scan") / "root")
    yield root
    shutil.rmtree(root, ignore_errors=True)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_counts_without_file_stats(big, monkeypatch):
    """F5-a: exact counts with sizes=off and no stat of files (lstat calls ==
    directories); sizes exact with sizes=all."""
    calls = []
    real_lstat, real_stat = os.lstat, os.stat
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: calls.append(p) or real_lstat(p, *a, **k))
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: calls.append(p) or real_stat(p, *a, **k))
    clk = FakeClock()
    r = filescan.scan(big, sizes="off", rate=10 ** 9, clock=clk, sleep=clk.sleep, temp_dirs=[])
    monkeypatch.setattr(os, "lstat", real_lstat)
    monkeypatch.setattr(os, "stat", real_stat)
    assert r["files"] == 30000 and r["complete"]
    file_calls = [c for c in calls if str(c).endswith(".txt")]
    assert file_calls == []
    inside = [str(c) for c in calls if str(c).startswith(str(big))]
    assert len(set(inside)) == r["dirs"]               # one lstat per directory, none per file
    assert len(inside) <= r["dirs"] + 3                # (+ resolving the root itself)
    r = filescan.scan(big, sizes="all", rate=10 ** 9, clock=clk, sleep=clk.sleep, temp_dirs=[])
    assert r["bytes"] == sum(i % 7 for i in range(30000))


def test_symlink_loop_and_device_boundary(tmp_path, monkeypatch):
    root = tmp_path / "r"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f").write_text("x")
    (root / "a" / "loop").symlink_to(root)
    (root / "other").mkdir()
    (root / "other" / "g").write_text("x")
    real = os.lstat

    class St:
        def __init__(self, s, dev):
            self._s, self.st_dev = s, dev

        def __getattr__(self, k):
            return getattr(self._s, k)
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: St(real(p), 999) if str(p).endswith("other") else real(p))
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[])
    assert r["files"] == 2                      # f and the symlink; nothing behind "other" or the loop


def test_throttle_and_caps(big):
    """F5-b: 1000 entries/s takes >= 27 s of clock for 30 000 entries;
    max_seconds stops early and says so; inconsistent caps are a load error."""
    clk = FakeClock()
    r = filescan.scan(big, sizes="off", rate=1000, clock=clk, sleep=clk.sleep, temp_dirs=[])
    assert r["complete"] and clk.t >= 27
    clk = FakeClock()
    r = filescan.scan(big, sizes="off", rate=1000, max_seconds=2, clock=clk, sleep=clk.sleep, temp_dirs=[])
    assert not r["complete"] and "stopped at 2 s" in r["coverage"] and "the rest is not counted" in r["coverage"]
    assert r["entries"] < 3000


def _planted(tmp_path, ended=True):
    need = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not need.exists():
        pytest.skip("test/ data corpus not present")
    root = tmp_path / "proj"
    job = root / "mat1"
    job.mkdir(parents=True)
    lines = need.read_text().splitlines(True)
    (job / "mat1_opt.out").write_text("".join(lines if ended else lines[:800]))
    scratch = tmp_path / "scr" / "mat1_opt"
    scratch.mkdir(parents=True)
    (scratch / "fort.9").write_text("x")
    (job / ".mat1_opt.scratch").write_text(str(scratch) + "\n")
    return root, scratch


def test_suggestions_by_provenance(tmp_path, monkeypatch):
    """F5-c."""
    root, scratch = _planted(tmp_path)
    home = tmp_path / "home"
    for name in ("tmp_results", "tmp", "tmpabcdefgh"):
        (home / name).mkdir(parents=True)
        (root / name).mkdir()
    monkeypatch.setenv("HOME", str(home))
    systmp = tmp_path / "systmp"
    (systmp / "tmpabcdefgh").mkdir(parents=True)
    (systmp / "pymp-abcd1234").mkdir()
    (systmp / "tmpnew12345").mkdir()
    old = time.time() - 3 * 86400
    os.utime(systmp / "tmpabcdefgh", (old, old))
    os.utime(systmp / "pymp-abcd1234", (old, old))
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[str(systmp)])
    paths = {s["path"]: s["rule"] for s in r["suggestions"] if s["path"]}
    assert paths.get(str(scratch)) == "R1"
    assert paths.get(str(systmp / "tmpabcdefgh")) == "R3" and paths.get(str(systmp / "pymp-abcd1234")) == "R3"
    assert str(systmp / "tmpnew12345") not in paths
    for name in ("tmp_results", "tmp", "tmpabcdefgh"):
        assert str(home / name) not in paths and str(root / name) not in paths
    # queued job of that name: not suggested
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[], queued_names=["mat1_opt"])
    assert str(scratch) not in {s["path"] for s in r["suggestions"]}


def test_unfinished_job_scratch_not_suggested(tmp_path):
    root, scratch = _planted(tmp_path, ended=False)
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[])
    assert str(scratch) not in {s["path"] for s in r["suggestions"]}


def test_module_never_deletes():
    """F5-d: no unlink, rmtree, remove, rename or rmdir call in the module."""
    tree = ast.parse(Path(filescan.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    assert not names & {"unlink", "rmtree", "remove", "rename", "rmdir", "removedirs", "truncate_path"}


def test_commands_are_quoted(tmp_path):
    """F5-f: a path with a space and ';' gives one quoted command."""
    root = tmp_path / "proj"
    job = root / "mat1"
    job.mkdir(parents=True)
    out = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not out.exists():
        pytest.skip("test/ data corpus not present")
    shutil.copy(out, job / "mat1_opt.out")
    scratch = tmp_path / "my run; echo x"
    scratch.mkdir()
    (job / ".mat1_opt.scratch").write_text(str(scratch))
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[])
    cmd = [s["command"] for s in r["suggestions"] if s["path"] == str(scratch)][0]
    assert cmd == f"rm -rf -- '{scratch}'"
    assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0
    assert shlex.split(cmd) == ["rm", "-rf", "--", str(scratch)]


def test_process_entry_and_lock(tmp_path):
    """The subprocess form writes the result and refuses a second scan."""
    root = _tree(tmp_path / "r", n_files=50, per_dir=10)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME")}
    env.update(XDG_CACHE_HOME=str(tmp_path / "c"), MACE_SITE_CONFIG="none", PATH="/usr/bin:/bin")
    out = tmp_path / "res.json"
    lock = tmp_path / "c" / "mace" / "cluster" / "scan.lock"
    lock.parent.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl,sys,time; f=open(sys.argv[1],'a'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('x', flush=True); time.sleep(20)", str(lock)],
                              stdout=subprocess.PIPE, text=True)
    try:
        holder.stdout.readline()
        p = subprocess.run([sys.executable, "-m", "mace.cluster.filescan", str(root), "--out", str(out)],
                           cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 5 and "A scan is already running" in p.stdout
    finally:
        holder.kill()
        holder.wait()
    p = subprocess.run([sys.executable, "-m", "mace.cluster.filescan", str(root), "--out", str(out)],
                       cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert json.loads(out.read_text())["files"] == 50


def test_pycache_rule_skips_installs_and_caps(tmp_path):
    root = tmp_path / "r"
    for i in range(14):
        d = root / f"pkg{i}" / "__pycache__"
        d.mkdir(parents=True)
        (d / "m.cpython-312.pyc").write_bytes(b"x")
    v = root / "venv" / "lib" / "python3.12" / "site-packages" / "x" / "__pycache__"
    v.mkdir(parents=True)
    (v / "a.pyc").write_bytes(b"x")
    mixed = root / "mixed" / "__pycache__"
    mixed.mkdir(parents=True)
    (mixed / "a.py").write_text("x")
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[])
    r4 = [s for s in r["suggestions"] if s["rule"] == "R4"]
    paths = [s["path"] for s in r4 if s["path"]]
    assert len(paths) == filescan.PYCACHE_MAX
    assert not [p for p in paths if "site-packages" in p or "mixed" in p]
    assert any("and 4 more __pycache__ folders" in s["reason"] for s in r4)


@pytest.mark.parametrize("where", ["jobdir", "parent", "root_parent", "scratch_root"])
def test_pointer_to_a_folder_holding_outputs_is_not_suggested(tmp_path, where):
    """A .<JOB>.scratch pointer that names the job folder itself, a folder
    above it, or a scratch root never yields an rm -rf suggestion."""
    root, scratch = _planted(tmp_path)
    job = root / "mat1"
    target = {"jobdir": job, "parent": root, "root_parent": tmp_path,
              "scratch_root": scratch.parent}[where]
    (job / ".mat1_opt.scratch").write_text(str(target) + "\n")
    protected = [str(scratch.parent)] if where == "scratch_root" else []
    r = filescan.scan(root, sizes="off", rate=10 ** 6, temp_dirs=[], protected=protected)
    assert not [s for s in r["suggestions"] if s["rule"] in ("R1", "R2")], r["suggestions"]
