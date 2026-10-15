"""F11 (and the subprocess halves of F2-F10): the real `mace_cli cluster`
under FakeSlurm, human and --json, its exit codes, and the standard-library-
only run of every subcommand with numpy/yaml/rich/... blocked."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import MACE_CLI, REPO_ROOT, SITE_QUOTA_TABLE, fake, res_text
from cluster_fixtures import TEST_DATA

# a Python 3.11 venv for the standard-library-only run: MACE_CIVENV311, else the
# Python running the tests when it is 3.11 (the 3.11 suite), else the test skips
CIVENV = os.environ.get("MACE_CIVENV311", "")
BLOCKED = ("numpy", "ase", "plotly", "spglib", "pandas", "scipy", "yaml", "rich", "matplotlib")


def mace(fs, *args, python=sys.executable, env_extra=None, cwd=None, timeout=120):
    env = dict(fs.env)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([python, str(MACE_CLI), *args], env=env, cwd=str(cwd or REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture
def fs(tmp_path):
    return fake(tmp_path)


def test_nodes_table_and_json(fs):
    """F2-c: --json equals summary(); the table has the documented columns."""
    rc, out, err = mace(fs, "cluster", "nodes", "--json")
    assert rc == 0, err
    data = json.loads(out)
    from mace.cluster import config, nodes
    from cluster_fixtures import NODES_TEXT
    ns = nodes.parse_nodes(NODES_TEXT)
    usable = set(data["usable_partitions"])
    assert usable == {"general-short", "general-long", "general-long-gpu", "group-a"}
    assert data["rows"] == json.loads(json.dumps(nodes.summary(ns, regex=config.DEFAULT_GROUP_REGEX,
                                                               only_partitions=usable)))
    rc, out, err = mace(fs, "cluster", "nodes")
    assert rc == 0
    header = out.splitlines()[1]
    for col in ("CPU type", "Nodes", "Idle", "Mixed", "Alloc", "Down/Drain", "Free cores", "Free mem", "Free GPUs"):
        assert col in header
    assert "Free now is a snapshot" in out
    rc, out, err = mace(fs, "cluster", "nodes", "--all-partitions")
    assert "visible-restricted: not usable by your accounts" in out


def test_fit_predict(fs):
    """F3-d: the fit table and the predicted start; one sbatch --test-only,
    no plain sbatch."""
    rc, out, err = mace(fs, "cluster", "fit", "--cpus", "24", "--mem-per-cpu", "4G", "--time", "1-00:00:00",
                        "--predict")
    assert rc == 0, err
    assert "fits now on" in out and "general-long" in out
    assert "SLURM predicts a start at 2030-01-01T21:38:00 on cpu-[001-002] in partition general-long." in out
    sb = fs.calls("sbatch")
    assert len(sb) == 1 and "--test-only" in sb[0]["argv"]
    rc, out, err = mace(fs, "cluster", "fit", "--cpus", "24", "--mem-per-cpu", "4G", "--time", "1-00:00:00",
                        "--predict", "--json")
    assert json.loads(out)["predict"]["accepted"] is True
    assert len(fs.calls("sbatch")) == 1                     # the cached answer


def test_fit_refused_and_reservation(fs):
    now = time.time()
    fmt = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    fs.update(test_only={"refuse": "allocation failure: Requested time limit is invalid (missing or exceeds some "
                                   "limit)"},
              res_text=res_text(fmt(now + 2 * 86400), fmt(now + 2 * 86400 + 3600)))
    rc, out, err = mace(fs, "cluster", "fit", "--cpus", "4", "--time", "3-00:00:00", "--predict")
    assert rc == 0
    assert "SLURM would refuse this request: Requested time limit is invalid" in out
    assert "Warning: Maintenance maint1 starts in 1 d 23 h" in out or "Warning: Maintenance maint1 starts in 2 d" in out
    rc, out, err = mace(fs, "cluster", "reservations")
    assert "maint1" in out and "maintenance" in out


def _site_tool(tmp_path, home):
    d = tmp_path / "site_bin"
    d.mkdir()
    t = d / "userquota.py"
    t.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({SITE_QUOTA_TABLE!r} + '\\n')\n")
    t.chmod(0o555)
    d.chmod(0o555)
    home.mkdir(exist_ok=True)
    (home / ".bash_profile").write_text(f"alias quota='NO_COLOR=1 {t}'\n")
    return d


def test_quota_from_site_alias(fs, tmp_path):
    """F4-d: the site tool (an alias in the login profile) gives the 87.1 %
    home row as a warning and the 91 % files row as critical; --json round-
    trips; warn_percent from a temp site.toml."""
    home = tmp_path / "home"
    d = _site_tool(tmp_path, home)
    try:
        rc, out, err = mace(fs, "cluster", "quota", env_extra={"HOME": str(home)})
        assert rc == 0, err
        assert re.search(r"home \(NFS\).*87\.1 %.*WARNING", out), out
        assert re.search(r"scratch .*91 %.*CRITICAL", out), out
        rc, out2, err = mace(fs, "cluster", "quota", "--json", env_extra={"HOME": str(home)})
        data = json.loads(out2)
        lv = {r["label"]: r["level"] for r in data["rows"]}
        assert lv["home (NFS)"] == "warning" and lv["scratch"] == "critical"
        site = tmp_path / "site.toml"
        site.write_text("[quota]\nwarn_percent = 85\n")
        rc, out3, err = mace(fs, "cluster", "quota", "--json", "--refresh",
                             env_extra={"HOME": str(home), "MACE_SITE_CONFIG": str(site), "MACE_GUI_TEST_HOOKS": "1",
                                        "MACE_CLUSTER_TEST_CLOCK": str(time.time() + 1000)})
        data = json.loads(out3)
        assert data["warn_percent"] == 85
        assert {r["label"]: r["level"] for r in data["rows"]}["home (NFS)"] == "warning"
        from mace.cluster import quotas
        assert quotas.levels({"space_pct": 84.0}, 85, 90) == "ok"
    finally:
        d.chmod(0o755)


def test_limits_check_submit(fs):
    """F6-d: the table, and the two warnings for 300 jobs x 32 cores; exit 0."""
    for i in range(12):
        fs.add_record(id=str(4100 + i), state="RUNNING", cpus=32, account="acct1")
    rc, out, err = mace(fs, "cluster", "limits", "--check-submit", "300", "--cpus", "32")
    assert rc == 0, err
    assert "This adds 300 jobs; 812 of 1000 submit slots are in use, so 112 more are refused by SLURM." in out
    assert "Running cores would exceed your QOS limit (1040): extra jobs wait with reason QOSMaxCpuPerUserLimit." in out
    assert "Max running" in out and "Fairshare" in out


def test_overview(fs):
    rc, out, err = mace(fs, "cluster")
    assert rc == 0, err
    assert out.startswith("Cluster cpu-cluster")
    assert "Usable partitions: general-long, general-long-gpu, general-short, group-a" in out
    assert "Maintenance: none scheduled" in out
    rc, out, err = mace(fs, "cluster", "--json")
    assert json.loads(out)["cluster"] == "cpu-cluster"


def _running_job(fs, tmp_path):
    src = sorted((TEST_DATA / "OPT").glob("Ag1Cl5_sym*.out")) if (TEST_DATA / "OPT").exists() else []
    if not src:
        pytest.skip("test/ data corpus not present")
    d = tmp_path / "proj"
    d.mkdir()
    lines = src[0].read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (d / "mat1_opt.out").write_text("".join(lines[:heads[5]]))
    shutil.copy(src[0].with_suffix(".d12"), d / "mat1_opt.d12")
    if src[0].with_suffix(".sh").exists():
        shutil.copy(src[0].with_suffix(".sh"), d / "mat1_opt.sh")
    (d / "recovery_config.yaml").write_text("error_recovery:\n  timeout_error:\n    max_retries: 1\n")
    fs.add_record(id="4001", state="RUNNING", name="mat1_opt", workdir=str(d), time_limit="0:30:00",
                  elapsed="0:28:00", cpus=32)
    fs.update(job_text={"4001": f"JobId=4001 JobName=mat1_opt UserId=user1(1001) JobState=RUNNING Reason=None "
                                f"RunTime=00:28:00 TimeLimit=00:30:00 Partition=general-long NodeList=cpu-001 "
                                f"NumNodes=1 NumCPUs=32 WorkDir={d} StdOut={d}/mat1_opt-4001.o Account=acct1"})
    return d


def test_job_detail_and_codes(fs, tmp_path):
    _running_job(fs, tmp_path)
    rc, out, err = mace(fs, "cluster", "job", "4001")
    assert rc == 0, err
    assert "Job 4001 · mat1_opt · RUNNING" in out
    assert "Progress (opt): Point 5" in out
    assert "This job is not tracked" in out
    rc, out, err = mace(fs, "cluster", "job", "999")
    assert rc == 4 and "Not one of your jobs" in err
    rc, out, err = mace(fs, "cluster", "job", "12_3;rm")
    assert rc == 2
    rc, out, err = mace(fs, "cluster", "nodes", "--bogus")
    assert rc == 2


def test_estimate_json_equals_function(fs, tmp_path):
    """F9-e."""
    folder = TEST_DATA / "SP"
    if not folder.is_dir():
        pytest.skip("test/ data corpus not present")
    hist = tmp_path / "hist"
    hist.mkdir()
    picked = 0
    for out in sorted(folder.glob("*.out")):
        if out.stat().st_size > 3_000_000 or not out.with_suffix(".d12").exists() or not out.with_suffix(".sh").exists():
            continue
        for suf in (".out", ".d12", ".sh"):
            shutil.copy(out.with_suffix(suf), hist / (out.stem + suf))
        picked += 1
        if picked >= 40:
            break
    deck = sorted(hist.glob("*.d12"))[0]
    rc, out, err = mace(fs, "cluster", "estimate", str(deck), "--history", str(hist), "--json")
    assert rc == 0, err
    data = json.loads(out)
    env = dict(fs.env)
    code = ("import json, sys; sys.path.insert(0, %r); from mace.cluster import history; "
            "print(json.dumps(history.estimate(%r, history=%r, use_sacct=True), default=str))"
            % (str(REPO_ROOT), str(deck), str(hist)))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert json.loads(r.stdout) == data
    rc, out, err = mace(fs, "cluster", "estimate", str(deck), "--history", str(hist))
    assert ("median" in out and "Suggested walltime" in out) or "Not enough similar runs" in out or "hidden" in out


def test_devnodes_pick(fs, tmp_path):
    """F10-b: --pick prints one host name only."""
    b = tmp_path / "sshbin"
    b.mkdir()
    (b / "ssh").write_text("#!/bin/bash\nfor a; do h=$a; done\nprintf '1.0 1 1 1/1 1\\n64\\nMemAvailable: 1000 kB\\n'\n")
    (b / "ssh").chmod(0o755)
    site = tmp_path / "site.toml"
    site.write_text('[devnodes]\nhosts = ["dev-node-1"]\n')
    rc, out, err = mace(fs, "cluster", "devnodes", "--pick",
                        env_extra={"PATH": f"{b}:{fs.env['PATH']}", "MACE_SITE_CONFIG": str(site)})
    assert rc == 0 and out == "dev-node-1\n", err
    rc, out, err = mace(fs, "cluster", "devnodes")
    assert "No interactive nodes are configured" in out


def test_files_subcommand(fs, tmp_path):
    root = tmp_path / "tree"
    for i in range(30):
        (root / f"d{i % 3}").mkdir(parents=True, exist_ok=True)
        (root / f"d{i % 3}" / f"f{i}").write_text("x")
    rc, out, err = mace(fs, "cluster", "files", str(root), "--json")
    assert rc == 0, err
    assert json.loads(out)["files"] == 30
    rc, out, err = mace(fs, "cluster", "files", str(root))
    assert "30 files in" in out and "nothing was deleted" not in out or "Cleanup" in out


def test_no_slurm(tmp_path):
    """F11-d: no SLURM on PATH: exit 3 with the sentence; quota still answers."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM", "NO_COLOR")}
    env.update(PATH="/usr/bin:/bin", XDG_CACHE_HOME=str(tmp_path / "c"), MACE_SITE_CONFIG="none",
               HOME=str(tmp_path))
    with_tools = [t for t in ("sinfo", "squeue") if shutil.which(t, path="/usr/bin:/bin")]
    if with_tools:
        pytest.skip("this machine has SLURM in /usr/bin")
    r = subprocess.run([sys.executable, str(MACE_CLI), "cluster", "nodes"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "No SLURM on this machine (sinfo not found)" in r.stderr
    r = subprocess.run([sys.executable, str(MACE_CLI), "cluster", "quota"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "Filesystem" in r.stdout
    # each note once (the view's notes are printed with the meta lines only)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(lines) == len(set(lines)), r.stdout


def _blocker(tmp_path, extra=()):
    d = tmp_path / "blocker"
    d.mkdir()
    names = list(BLOCKED) + list(extra)
    (d / "sitecustomize.py").write_text(
        "import sys\n"
        f"_BLOCK = {names!r}\n"
        "class _B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in _BLOCK:\n"
        "            raise ImportError(f'blocked for the test: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _B())\n")
    return d


def _civenv_python():
    if not CIVENV and sys.version_info[:2] == (3, 11):
        return sys.executable
    p = Path(CIVENV) / "bin" / "python"
    if not CIVENV or not p.exists():
        pytest.skip("no Python 3.11 venv (set MACE_CIVENV311)")
    return str(p)


def test_every_subcommand_stdlib_only(fs, tmp_path):
    """F11-b: under the import blocker (numpy, ase, plotly, spglib, pandas,
    scipy, yaml, rich, matplotlib) on the 3.11 venv every subcommand exits
    with its documented code and no traceback; `job` prints the walltime-risk
    sentence with the PyYAML note."""
    py = _civenv_python()
    d = _running_job(fs, tmp_path)
    blk = _blocker(tmp_path)
    env = {"PYTHONPATH": str(blk)}
    deck = d / "mat1_opt.d12"
    cases = [(("cluster", "--help"), 0), (("cluster",), 0), (("cluster", "nodes"), 0),
             (("cluster", "fit", "--cpus", "4", "--time", "1:00:00", "--predict"), 0), (("cluster", "reservations"), 0),
             (("cluster", "quota"), 0), (("cluster", "files", str(d)), 0), (("cluster", "limits"), 0),
             (("cluster", "job", "4001"), 0), (("cluster", "estimate", str(deck), "--cores", "32"), 0),
             (("cluster", "devnodes"), 0), (("cluster", "nodes", "--help"), 0)]
    for args, code in cases:
        rc, out, err = mace(fs, *args, python=py, env_extra=env)
        assert rc == code, (args, out, err)
        assert "Traceback" not in out + err and "ImportError" not in out + err, (args, err)
        if args[:2] == ("cluster", "job"):
            assert "PyYAML is not installed" in out, out
    # the blocker really blocks
    r = subprocess.run([py, "-c", "import yaml"], env={**fs.env, **env}, capture_output=True, text=True)
    assert r.returncode != 0


def test_no_fcntl_no_slurm(tmp_path):
    """F11-e: without fcntl (as on Windows) and without SLURM: exit 3 with
    the sentence, and query.run still works (no file lock)."""
    blk = _blocker(tmp_path, extra=("fcntl",))
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "COLORTERM", "TERM", "NO_COLOR")}
    env.update(PATH="/usr/bin:/bin", XDG_CACHE_HOME=str(tmp_path / "c"), MACE_SITE_CONFIG="none",
               PYTHONPATH=str(blk))
    if shutil.which("sinfo", path="/usr/bin:/bin"):
        pytest.skip("this machine has SLURM in /usr/bin")
    r = subprocess.run([sys.executable, str(MACE_CLI), "cluster"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "No SLURM on this machine" in r.stderr and "Traceback" not in r.stderr
    code = ("import sys; sys.path.insert(0, %r); from mace.cluster import query; "
            "r = query.run('x', ['echo', 'hi'], ttl=60); print(r.value.strip())" % str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.stdout.strip() == "hi", r.stderr


def test_help_flags_match_argparse():
    """F11-c: every flag of the argparse parser appears in `mace cluster
    --help`, and the curated help names no flag argparse lacks."""
    from mace.cluster.cli import build_parser
    ap = build_parser()
    flags = set()
    for a in ap._actions:
        flags.update(o for o in a.option_strings if o.startswith("--"))
    sub = [a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction"][0]
    for p in sub.choices.values():
        for a in p._actions:
            flags.update(o for o in a.option_strings if o.startswith("--"))
    r = subprocess.run([sys.executable, str(MACE_CLI), "cluster", "--help"], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME")})
    assert r.returncode == 0
    shown = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", r.stdout)) - {"--test-only"}   # sbatch's, named in prose
    assert flags - {"--help"} <= shown, flags - shown
    assert shown <= flags, shown - flags


def test_docs_name_every_flag():
    """F14-a: docs/cli.md's mace cluster section names every flag and
    subcommand of the parser (the help == docs check)."""
    from mace.cluster.cli import build_parser
    doc = (REPO_ROOT / "docs" / "cli.md").read_text()
    sec = doc[doc.index("## mace cluster"):]
    ap = build_parser()
    sub = [a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction"][0]
    for name, p in sub.choices.items():
        if name != "overview":
            assert f"mace cluster {name}" in sec, name
        for a in p._actions:
            for o in a.option_strings:
                if o.startswith("--") and o != "--help":
                    assert o in sec, (name, o)


def test_job_array_tasks(fs, tmp_path):
    """An array task's detail lists its array's tasks in the queue (from the
    cached squeue rows; no extra query)."""
    d = tmp_path / "arr"
    d.mkdir()
    for t, st in ((1, "RUNNING"), (2, "RUNNING"), (3, "PENDING"), (4, "PENDING")):
        fs.add_record(id=f"5000_{t}", array="5000", task=t, state=st, name="arr_sp", workdir=str(d), cpus=4)
    fs.add_record(id="5100", state="RUNNING", name="single", workdir=str(d), cpus=4)
    fs.update(job_text={"5000_2": "JobId=5002 ArrayJobId=5000 ArrayTaskId=2 JobName=arr_sp UserId=user1(1001) "
                                  f"JobState=RUNNING Reason=None RunTime=00:01:00 TimeLimit=01:00:00 WorkDir={d} "
                                  "NumNodes=1 NumCPUs=4 Partition=general-short Account=acct1"})
    rc, out, err = mace(fs, "cluster", "job", "5000_2", "--json")
    assert rc == 0, err
    arr = json.loads(out)["array_tasks"]
    assert arr["array_job"] == "5000" and arr["total_in_queue"] == 4
    assert arr["counts"] == {"RUNNING": 2, "PENDING": 2}
    assert [t["id"] for t in arr["tasks"]] == ["5000_1", "5000_2", "5000_3", "5000_4"]
    rc, out, err = mace(fs, "cluster", "job", "5000_2")
    assert "Array 5000: 4 task(s) in the queue (2 pending, 2 running)" in out
    rc, out, err = mace(fs, "cluster", "job", "5100", "--json")
    assert rc == 0 and json.loads(out)["array_tasks"] is None


# ------------------------------------------------------------------ nested ETA levels (polish I)
def test_job_text_prints_levels(fs, tmp_path):
    """`mace cluster job ID` prints the ETA as nested levels: the total,
    the optimization's points, the next point's SCF, then the Why line."""
    _running_job(fs, tmp_path)
    rc, out, err = mace(fs, "cluster", "job", "4001")
    assert rc == 0, err
    lines = out.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Progress (opt): Point 5"))
    assert lines[i + 1].startswith("  Total ")
    assert lines[i + 2].startswith("    Optimization point 5 of about ")
    assert "(+ final restart 1-7)" in lines[i + 2]
    assert lines[i + 3].startswith("      Thresholds: ")
    assert lines[i + 4].startswith("      Next point's SCF: about ")
    assert any(l.startswith("  Why: ") and "left in total" in l for l in lines)
    assert not any(l.startswith("ETA") for l in lines)
    tl = [l for l in out.splitlines() if "Thresholds:" in l]       # only the deck's non-default SCF TOLDEE
    assert tl == ["      Thresholds: SCF energy change 1e-9 (tighter than the optimization default 1e-7)"], out
    assert "Warning: Job 4001 " in out and "left in total" in out     # 2 min of walltime left


def test_job_json_has_levels(fs, tmp_path):
    _running_job(fs, tmp_path)
    rc, out, err = mace(fs, "cluster", "job", "4001", "--json")
    assert rc == 0, err
    pr = json.loads(out)["progress"]
    assert pr["levels_version"] == 1
    assert [lv["id"] for lv in pr["levels"]][:2] == ["total", "opt.points"]
    assert "low_s" in pr["eta"] and pr["eta"]["low_s"] == pr["levels"][0]["remaining_s"]["low"]
    assert pr["eta"]["low_points"] is not None and pr["eta"]["stage_basis"]
    assert all(lv["text"] for lv in pr["levels"])


def test_job_text_error_progress_unchanged():
    """A progress dict without levels (the reader failed) prints as before."""
    from mace.cluster.cli import job_text
    v = {"id": "1", "name": "x", "state": "RUNNING", "elapsed_s": 10, "limit_s": 100, "efficiency": {},
         "scontrol_command": "scontrol show job 1",
         "progress": {"mode": "unknown", "error": "OSError: boom", "summary": "",
                      "eta": {"basis": "fewer than 3 cycles", "low_s": None}}}
    out = job_text(v)
    assert "Progress (unknown): " in out and "ETA unknown: fewer than 3 cycles" in out
    assert "Total" not in out


def test_job_text_cpu_caveat_once_after_the_cpu_line():
    """The busy-wait caveat follows the CPU line once; the advice stays."""
    from mace.cluster import job_efficiency as je
    from mace.cluster.cli import job_text
    rows = {"7100": {"JobID": "7100", "State": "COMPLETED", "Elapsed": "10:00:00", "NCPUS": "32",
                     "AllocTRES": "cpu=32,mem=64G,node=1"},
            "7100.batch": {"JobID": "7100.batch", "TotalCPU": "3-08:00:00", "MaxRSS": "9813656K", "NTasks": "1"}}
    eff = je.summarize(rows, "7100")
    v = {"id": "7100", "name": "x", "state": "COMPLETED", "elapsed_s": 36000, "limit_s": 86400,
         "efficiency": eff, "scontrol_command": "scontrol show job 7100"}
    lines = job_text(v).splitlines()
    i = lines.index(eff["cpu_text"])
    assert lines[i + 1] == je.CPU_CAVEAT and lines.count(je.CPU_CAVEAT) == 1
    assert "You could request fewer cores (about 8 were busy)." in lines


def test_job_text_thresholds_line_only_when_non_default(fs, tmp_path):
    """A tighter max gradient in the .out header joins the Thresholds line
    under the optimization level (the gradient defaults show none: previous
    test)."""
    d = _running_job(fs, tmp_path)
    out_f = d / "mat1_opt.out"
    text = out_f.read_text()
    assert "MAXIMUM GRADIENT COMPONENT   0.00045" in text
    out_f.write_text(text.replace("MAXIMUM GRADIENT COMPONENT   0.00045", "MAXIMUM GRADIENT COMPONENT   0.00015"))
    rc, out, err = mace(fs, "cluster", "job", "4001")
    assert rc == 0, err
    tl = [l for l in out.splitlines() if "Thresholds:" in l]
    assert tl == ["      Thresholds: max gradient 1.5e-4 (tighter than the default 4.5e-4), SCF energy change "
                  "1e-9 (tighter than the optimization default 1e-7)"], out


# ------------------------------------------------------------------ live HPCC check (polish I)
def test_job_elapsed_moves_on_with_the_cache_age(fs, tmp_path):
    """A running job's elapsed time is scontrol's RunTime plus the age of the
    cached query: refreshed every few seconds inside the cache's minute, the
    job's clock moves on instead of standing still (seen live on a SLURM
    cluster: "Elapsed 33 s" three times over 50 s)."""
    _running_job(fs, tmp_path)
    t0 = time.time()
    env = {"MACE_GUI_TEST_HOOKS": "1", "MACE_CLUSTER_TEST_CLOCK": str(t0)}
    rc, out, err = mace(fs, "cluster", "job", "4001", "--json", env_extra=env)
    assert rc == 0, err
    assert json.loads(out)["elapsed_s"] == pytest.approx(28 * 60, abs=1)
    env["MACE_CLUSTER_TEST_CLOCK"] = str(t0 + 40)             # inside the 60 s cache: no new scontrol
    rc, out, err = mace(fs, "cluster", "job", "4001", "--json", env_extra=env)
    assert rc == 0, err
    v = json.loads(out)
    assert v["detail_cache"]["source"] == "cache"
    assert v["elapsed_s"] == pytest.approx(28 * 60 + 40, abs=1)
    env["MACE_CLUSTER_TEST_CLOCK"] = str(t0 + 50)
    rc, out, err = mace(fs, "cluster", "job", "4001", env_extra=env)
    assert "Elapsed 28 min of 30 min" in out


def test_job_text_no_why_line_that_repeats_the_total(fs, tmp_path):
    """While the first point runs, the Total line says why the time left is
    unknown; the Why line said the same twice more and is left out. The first
    point's SCF, which CRYSTAL writes into the output itself, gets its own line."""
    src = sorted((TEST_DATA / "OPT").glob("Ag1Cl5_sym*.out")) if (TEST_DATA / "OPT").exists() else []
    if not src:
        pytest.skip("test/ data corpus not present")
    d = _running_job(fs, tmp_path)
    lines = src[0].read_text(errors="replace").splitlines(True)
    cyc = [i for i, l in enumerate(lines) if re.match(r"\s*CYC\s+\d+\s+ETOT", l)]
    (d / "mat1_opt.out").write_text("".join(lines[:cyc[3] + 1]))
    rc, out, err = mace(fs, "cluster", "job", "4001")
    assert rc == 0, err
    assert "  Total: unknown (0 points done; an estimate needs 3)" in out
    assert "      First point's SCF: cycle 3 (|DE| " in out, out
    assert "Why:" not in out, out


def test_ended_job_forgotten_by_scontrol_is_not_running(fs, tmp_path):
    """Live: two test jobs ended (TIMEOUT, COMPLETED), scontrol forgot them
    within minutes ("Invalid job id"), and `mace cluster job` kept showing
    the cached "RUNNING" record, its elapsed time growing past the real one.
    Once the queue no longer lists the job and sacct says it ended, the state
    and elapsed time are sacct's."""
    d = _running_job(fs, tmp_path)
    t0 = time.time()
    env = {"MACE_GUI_TEST_HOOKS": "1", "MACE_CLUSTER_TEST_CLOCK": str(t0)}
    rc, out, err = mace(fs, "cluster", "job", "4001", "--json", env_extra=env)
    assert rc == 0 and json.loads(out)["state"] == "RUNNING", err
    fs.remove_record("4001")
    st = fs.load()
    st["job_text"].pop("4001")
    st["sacct_rows"] = {"4001": ["4001|mat1_opt|COMPLETED|00:29:10|00:30:00|32|4G||00:00:00|cpu=32|cpu-001|0:0|"
                                 f"2030-01-01T00:00:00|2030-01-01T00:29:10|{d}"]}
    fs.save(st)
    env["MACE_CLUSTER_TEST_CLOCK"] = str(t0 + 600)            # the scontrol cache is stale; its refresh fails
    for _ in range(2):
        rc, out, err = mace(fs, "cluster", "job", "4001", "--json", env_extra=env)
        assert rc == 0, err
        v = json.loads(out)
        assert v["state"] == "COMPLETED", (v["state"], v["detail_cache"])
        assert v["elapsed_s"] == pytest.approx(29 * 60 + 10)
    rc, out, err = mace(fs, "cluster", "job", "4001", env_extra=env)
    assert "Job 4001 · mat1_opt · COMPLETED" in out and "RUNNING" not in out, out
    assert "Elapsed 29 min of 30 min" in out


def test_job_killed_at_its_walltime_reads_stopped(fs, tmp_path):
    """Live: an IR/Raman job hit its walltime inside the IR tensors; CRYSTAL
    wrote no last line, so the output alone still read "IR/Raman tensors:
    running". With SLURM's TIMEOUT the running stage reads "stopped here",
    the total says the job ended, and nothing is estimated or warned."""
    d = _running_job(fs, tmp_path)
    fs.remove_record("4001")
    st = fs.load()
    st["job_text"].pop("4001")
    st["sacct_rows"] = {"4001": ["4001|mat1_opt|TIMEOUT|00:30:05|00:30:00|32|4G||00:00:00|cpu=32|cpu-001|0:0|"
                                 f"2030-01-01T00:00:00|2030-01-01T00:30:05|{d}"]}
    fs.save(st)
    rc, out, err = mace(fs, "cluster", "job", "4001", "--json")
    assert rc == 0, err
    v = json.loads(out)
    pr = v["progress"]
    assert v["state"] == "TIMEOUT" and pr["stopped"] == "TIMEOUT"
    assert pr["levels"][0]["text"] == "Total: the job ended (TIMEOUT) before CRYSTAL finished"
    assert [lv["text"] for lv in pr["levels"] if lv["status"] == "stopped"] == ["Optimization: stopped here (TIMEOUT)"]
    assert not any(lv["status"] == "running" for lv in pr["levels"])
    assert pr["risk"]["level"] is None and pr["eta"]["low_s"] is None
    rc, out, err = mace(fs, "cluster", "job", "4001")
    assert "Progress (opt): Stopped (TIMEOUT) in Optimization" in out, out
    assert "Why:" not in out and "Warning" not in out and "running" not in out.lower().split("progress")[1], out


def test_stopped_progress_marks_stages_not_reached():
    """The stages after the one SLURM stopped never ran: "not reached", no
    time; finished stages keep their times; the memo's dict is not changed."""
    from mace.cluster import jobinfo
    from mace.cluster import progress as P
    lv = [P._level("total", 0, "Total", "running", unknown_reason="x", basis="x"),
          P._level("freq.cphf", 1, "IR/Raman (CPHF/CPKS)", "done", done_s=60.0),
          P._level("freq.tensors", 1, "IR/Raman tensors", "running", basis="b"),
          P._level("freq.output", 1, "Frequencies and output", "pending", low_s=0.0, high_s=30.0, basis="c")]
    for x in lv:
        x["text"] = P.level_text(x)
    prog = {"mode": "freq", "summary": "IR/Raman tensors running (time unknown)", "levels": lv,
            "eta": {"basis": "x"}, "risk": {"level": "tight"}}
    got = jobinfo.stopped_progress(prog, "TIMEOUT")
    assert got["summary"] == "Stopped (TIMEOUT) in IR/Raman tensors"
    assert [x["text"] for x in got["levels"]] == [
        "Total: the job ended (TIMEOUT) before CRYSTAL finished", "IR/Raman (CPHF/CPKS, first order): done in 1 min",
        "IR/Raman tensors: stopped here (TIMEOUT)", "Frequencies and output: not reached"]
    assert got["levels"][3]["remaining_s"] == {"low": None, "high": None}
    assert prog["levels"][2]["status"] == "running" and prog["risk"]["level"] == "tight"
