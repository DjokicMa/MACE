"""`mace update` (mace/update.py) on fake git repositories: a bare "origin"
with tags v1.1.4, v1.1.5-preview, v1.1.5, v1.1.6-preview and main commits
after them; channel resolution, the refusals (local changes, local commits,
older targets), dry runs, the requirements message, --check, and --remote
through a fake ssh that runs the command in a temporary "cluster" checkout."""
import re
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mace import update as U

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not __import__("shutil").which("git"), reason="needs git")


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.edu", "-c",
                           "init.defaultBranch=main", "-c", "advice.detachedHead=false", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def commit(work, version, msg, req=None, tag=None, changelog=None):
    (work / "mace").mkdir(exist_ok=True)
    (work / "mace" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (work / "mace_cli").write_text(f'#!/usr/bin/env python3\nprint("MACE v{version}")\n')
    if req is not None:
        (work / "requirements.txt").write_text(req)
    if changelog:
        (work / "CHANGELOG.md").write_text(changelog)
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", msg)
    if tag:
        git(work, "tag", tag)


def changelog(*sections):
    return "# Changelog\n\n" + "".join(f"## [{s}]\n\n- change in {s}\n\n" for s in sections)


@pytest.fixture
def origin(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q")
    commit(work, "1.1.4", "v1.1.4", req="numpy\n", tag="v1.1.4", changelog=changelog("1.1.4"))
    commit(work, "1.1.5", "preview", tag="v1.1.5-preview", changelog=changelog("1.1.5-preview", "1.1.4"))
    commit(work, "1.1.5", "final", req="numpy\nase\n", tag="v1.1.5",
           changelog=changelog("1.1.5", "1.1.5-preview", "1.1.4"))
    commit(work, "1.1.6", "preview 6", tag="v1.1.6-preview",
           changelog=changelog("1.1.6-preview", "1.1.5", "1.1.5-preview", "1.1.4"))
    commit(work, "1.1.6", "main work", changelog=changelog("Unreleased", "1.1.6-preview", "1.1.5"))
    bare = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))
    return bare


def clone(origin, dest, at="v1.1.4"):
    git(dest.parent, "clone", "-q", str(origin), str(dest))
    git(dest, "reset", "-q", "--hard", at)          # the test's starting point: main behind origin/main
    return dest


def _run(home, *args, **kw):
    return subprocess.run([sys.executable, "-c", "import sys; from mace.update import main; sys.exit(main())",
                           "--home", str(home), *args], cwd=str(REPO), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=120, **kw)


def test_channels(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    g = U.Git(home)
    g.run("fetch", "--tags", "origin")
    assert U.channel_target(g, "release")[0] == "v1.1.5"
    assert U.channel_target(g, "preview")[0] == "v1.1.6-preview"
    assert U.channel_target(g, "latest")[0] == "origin/main"
    # a final is newer than its own preview
    assert U.tag_key("v1.1.5") > U.tag_key("v1.1.5-preview") > U.tag_key("v1.1.4")
    git(home, "tag", "-d", "v1.1.6-preview")
    git(origin, "tag", "-d", "v1.1.6-preview")
    g.run("fetch", "--tags", "--prune", "--prune-tags", "origin")
    assert U.channel_target(g, "preview")[0] == "v1.1.5"


def test_release_fast_forwards_main_and_prints_the_pip_line(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    r = _run(home, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "MACE in" in out and ": 1.1.4 (v1.1.4, on main)" in out
    assert "Target (release): v1.1.5 = 1.1.5 (v1.1.5)" in out
    assert "## [1.1.5]" in out and "## [1.1.5-preview]" in out and "## [1.1.4]" not in out
    assert "Will run: git merge --ff-only v1.1.5   (on main; git pull keeps working)" in out
    # a system/module Python (GitHub's runner) gets --user
    assert "requirements.txt changed" in out and re.search(
        rf"-m pip install (--user )?-r {re.escape(str(home))}/requirements.txt", out), out
    assert "Check: MACE v1.1.5" in out
    assert git(home, "describe", "--tags") == "v1.1.5" and git(home, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    git(home, "pull", "-q", "--ff-only")                          # still a normal git pull checkout
    assert git(home, "rev-parse", "HEAD") == git(origin, "rev-parse", "main")


def test_dry_run_changes_nothing(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    before = git(home, "rev-parse", "HEAD")
    r = _run(home, "--channel", "preview", "--dry-run")
    assert r.returncode == 0, r.stdout
    assert "Would run: git merge --ff-only v1.1.6-preview" in r.stdout and "Dry run: nothing changed" in r.stdout
    assert "## [1.1.6-preview]" in r.stdout
    assert git(home, "rev-parse", "HEAD") == before


def test_refuses_local_changes(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mace_cli").write_text("edited\n")
    (home / "untracked.out").write_text("output files do not count\n")
    before = git(home, "rev-parse", "HEAD")
    r = _run(home, "--yes")
    assert r.returncode == 2
    assert "these files have local changes" in r.stdout and " M mace_cli" in r.stdout
    assert "untracked.out" not in r.stdout and f"git -C {home} stash" in r.stdout
    assert git(home, "rev-parse", "HEAD") == before and (home / "mace_cli").read_text() == "edited\n"


def test_refuses_local_commits_and_older_targets(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mine.txt").write_text("x")
    git(home, "add", "mine.txt")
    git(home, "commit", "-q", "-m", "my own change")
    head = git(home, "rev-parse", "HEAD")
    r = _run(home, "--yes")
    assert r.returncode == 2 and "not a fast-forward" in r.stdout and "my own change" in r.stdout
    assert git(home, "rev-parse", "HEAD") == head


def test_a_checkout_ahead_of_the_newest_release_is_not_a_failure(origin, tmp_path):
    """main after a git pull is newer than the newest release: nothing to do,
    exit 0, and the channel that would move it."""
    newer = clone(origin, tmp_path / "newer", at="origin/main")
    head = git(newer, "rev-parse", "HEAD")
    r = _run(newer, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout
    assert "newer than v1.1.5" in r.stdout and "mace update --channel latest" in r.stdout
    assert git(newer, "rev-parse", "HEAD") == head


def test_detached_checkout_gets_a_branch_and_latest_returns_to_main(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    git(home, "checkout", "-q", "--detach", "v1.1.4")
    r = _run(home, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout
    assert "git checkout -b mace-v1.1.5 v1.1.5   (return later with: git checkout main && git pull)" in r.stdout
    assert git(home, "rev-parse", "--abbrev-ref", "HEAD") == "mace-v1.1.5"
    r = _run(home, "--channel", "latest", "--yes")
    assert r.returncode == 0, r.stdout
    assert git(home, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(home, "rev-parse", "HEAD") == git(origin, "rev-parse", "main")
    assert "## [Unreleased]" in r.stdout


def test_needs_yes_without_a_terminal_and_up_to_date(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    r = _run(home)
    assert r.returncode == 2 and "pass --yes" in r.stdout
    up = clone(origin, tmp_path / "up", at="v1.1.5")
    assert "nothing to do" in _run(up, "--yes").stdout


def test_check_reports_each_channel(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    r = _run(home, "--check")
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    assert any(ln.startswith("  release  v1.1.5 (1.1.5): available (2 commits ahead)") for ln in lines), r.stdout
    assert any(ln.startswith("  preview  v1.1.6-preview (1.1.6): available") for ln in lines)
    assert any(ln.startswith("  latest   origin/main (1.1.6): available") for ln in lines)
    assert git(home, "rev-parse", "HEAD") == git(home, "rev-parse", "v1.1.4")


def test_queue_warning(origin, tmp_path, monkeypatch):
    fb = tmp_path / "bin"
    fb.mkdir()
    (fb / "squeue").write_text("#!/bin/sh\nprintf 'RUNNING\\nPENDING\\nPENDING\\n'\n")
    (fb / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "netid")
    w = U.busy_warnings(Path("/x/MACE"))
    assert w[0].startswith("Warning: 3 SLURM job(s) of yours are in the queue (1 running, 2 waiting).")
    assert "run the code in /x/MACE: update between batches" in w[0]


FAKE_SSH = """#!/usr/bin/env bash
# runs the remote command locally, as the "cluster": a clean environment (ssh passes none of this
# computer's, e.g. MACE_HOME), HOME=$FAKE_CLUSTER_HOME
echo "$*" >> "$FAKE_SSH_LOG"
while [ $# -gt 1 ]; do case "$1" in -J|-o) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
cd "$FAKE_CLUSTER_HOME" && exec env -i HOME="$FAKE_CLUSTER_HOME" PATH=/usr/bin:/bin bash -c "$1"
"""


def test_remote_update_runs_in_the_clusters_checkout(origin, tmp_path):
    cluster_home = tmp_path / "cluster"
    cluster_home.mkdir()
    remote = clone(origin, cluster_home / "MACE")
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    from mace.gui import connect as C
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "hpcc"},
                     "profiles": {"hpcc": {"gateway": "me@gw.example.edu", "node": "dev-node-01"}}},
                    cfg / "mace" / "connect.toml")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(PATH=f"{fb}:{env['PATH']}", XDG_CONFIG_HOME=str(cfg), FAKE_CLUSTER_HOME=str(cluster_home),
               FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote", "--remote-only",
                        "--dry-run"], env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--- the cluster (hpcc: me@gw.example.edu) ---" in r.stdout
    assert f"MACE in {remote}: 1.1.4" in r.stdout and "Would run: git merge --ff-only v1.1.5" in r.stdout
    log = (tmp_path / "ssh.log").read_text()
    assert "BatchMode=yes" in log and "me@gw.example.edu" in log and "dev-node-01" not in log
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote", "hpcc",
                        "--remote-only", "--yes", "--channel", "preview"], env=env, capture_output=True, text=True,
                       timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert git(remote, "describe", "--tags") == "v1.1.6-preview" and "Check: MACE v1.1.6" in r.stdout
    # the same refusals over there
    (remote / "mace_cli").write_text("edited\n")
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote", "--remote-only",
                        "--yes", "--channel", "latest"], env=env, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 2 and "these files have local changes" in r.stdout


# ---- the standalone connect scripts: --update-remote / -UpdateRemote, the version line, checksums

SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"


def test_scripts_update_remote_parity():
    import re
    sh, ps = SH.read_text(), PS1.read_text()
    assert re.search(r"^\s+--update-remote\)", sh, re.M) and "$UpdateRemote" in ps
    for word in ("update --help", "--remote-only", "else exit 3; fi", "MACE on ", "older than 1.1.6 and has no update "
                 "command: update it once by hand", "stopped without updating", "you said no",
                 "mace update did not run"):
        assert word in sh and word in ps, word


def test_bash_update_remote_runs_mace_update_first(tmp_path):
    fb = tmp_path / "fb"
    fb.mkdir()
    (fb / "ssh").write_text('#!/bin/sh\n[ "$1" = -G ] && exit 0\nprintf "%s\\n" "$*" >> "$LOG"\nexit 255\n')
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BROWSER"}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "c"),
               LOG=str(tmp_path / "log"))
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--no-open", "--update-remote",
                        "preview"], env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    first = (tmp_path / "log").read_text().splitlines()[0]
    assert "BatchMode=yes" in first and " me@gw exec bash -lc " in first and "n1" not in first
    assert 'then "$M" update --channel preview --yes; else exit 3; fi' in first
    assert "Updating MACE on me@gw (preview) ..." in r.stdout
    assert "ssh to me@gw failed (see above), so mace update did not run" in r.stderr
    assert "has no update command" not in r.stderr


SH_UPDATE_SSH = """#!/bin/sh
[ "$1" = -G ] && { printf '%s' "$FAKE_G"; exit 0; }
printf '%s\\n' "$*" >> "$LOG"
case "$*" in *"exec bash -lc"*"update --help"*) exit "$FAKE_URC" ;; esac
exit 255
"""


@pytest.mark.parametrize("urc,want", [
    ("3", "the MACE on me@gw is older than 1.1.6 and has no update command: update it once by hand"),
    ("2", "mace update on me@gw stopped without updating (its reason is above)"),
    ("1", "the MACE on me@gw was not updated (you said no)"),
    ("0", None),
])
def test_bash_update_remote_tells_a_refusal_from_an_old_mace(tmp_path, urc, want):
    fb = tmp_path / "fb"
    fb.mkdir()
    (fb / "ssh").write_text(SH_UPDATE_SSH)
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BROWSER"}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "c"),
               LOG=str(tmp_path / "log"), FAKE_URC=urc, FAKE_G="user me\n")
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--no-open", "--update-remote",
                        "--mace", "python3 ~/M&CE/mace_cli"], env=env, capture_output=True, text=True, timeout=60,
                       stdin=subprocess.DEVNULL)
    first = (tmp_path / "log").read_text().splitlines()[0]
    # each word of a multi-word mace command quoted as connect's remote command quotes it
    assert "then python3 \"$HOME\"/'\"'\"'M&CE/mace_cli'\"'\"' update --channel release --yes;" in first, first
    if want:
        assert want in r.stderr, r.stderr
    else:
        assert "mace update" not in r.stderr and "MACE on me@gw" not in r.stderr


def test_bash_update_remote_keeps_the_config_timeout(tmp_path):
    fb = tmp_path / "fb"
    fb.mkdir()
    (fb / "ssh").write_text(SH_UPDATE_SSH)
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BROWSER"}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "c"),
               LOG=str(tmp_path / "log"), FAKE_URC="0", FAKE_G="user me\nconnecttimeout 90\n")
    subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--no-open", "--update-remote"], env=env,
                   capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    log = (tmp_path / "log").read_text().splitlines()
    assert "ConnectTimeout=20" not in log[0]                       # the update's ssh
    assert len(log) > 1 and "ConnectTimeout=20" not in log[1]      # the connection's


def test_published_checksums_match_the_scripts():
    import hashlib
    sums = dict(reversed(ln.split()) for ln in (REPO / "scripts" / "SHA256SUMS").read_text().splitlines() if ln)
    for name in ("mace-gui-connect.sh", "mace-gui-connect.ps1"):
        assert sums[name] == hashlib.sha256((REPO / "scripts" / name).read_bytes()).hexdigest(), \
            f"regenerate scripts/SHA256SUMS: cd scripts && sha256sum mace-gui-connect.sh mace-gui-connect.ps1"


# ---- after an update that changes requirements: run pip or not (pip is always a fake here)

def fake_python(tmp_path, kind):
    """<prefix>/bin/python that logs its arguments (never a real pip); a venv, a conda env or a plain prefix."""
    prefix = tmp_path / ("envs" if kind == "conda" else "") / f"py-{kind}"
    (prefix / "bin").mkdir(parents=True)
    if kind == "venv":
        (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n")
    if kind == "conda":
        (prefix / "conda-meta").mkdir()
    py = prefix / "bin" / "python"
    py.write_text('#!/bin/sh\necho "$@" >> "$(dirname "$0")/pip.log"\nexit "${FAKE_PIP_RC:-0}"\n')
    py.chmod(0o755)
    return py


def _pip_log(py):
    log = py.parent / "pip.log"
    return log.read_text() if log.exists() else ""


def _interactive(home, answers, capsys, **kw):
    it, prompts = iter(answers), []
    code = U.update(home, "release", ask=lambda prompt: prompts.append(prompt) or next(it), tty=True, **kw)
    return code, capsys.readouterr().out + "\n".join(prompts)


@pytest.mark.parametrize("kind", ["venv", "conda", "system"])
def test_pip_question_choice_1_runs_it(origin, tmp_path, capsys, monkeypatch, kind):
    home = clone(origin, tmp_path / "me")
    py = fake_python(tmp_path, kind)
    monkeypatch.setenv("MACE_PYTHON", str(py))
    code, out = _interactive(home, ["y", "1"], capsys)
    assert code == 0, out
    assert "requirements.txt changed:" in out and "  + ase" in out
    user = ["--user"] if kind == "system" else []
    assert _pip_log(py).split() == ["-m", "pip", "install", *user, "-r", str(home / "requirements.txt")]
    assert "The requirements are installed." in out and "Check: MACE v1.1.5" in out
    if kind == "conda":
        assert f"the conda environment py-conda ({py.parent.parent})" in out
    if kind == "system":
        assert "uses --user" in out and "(default 2)" in out
    else:
        assert "(default 1)" in out and "--user" not in out


def test_pip_question_choice_2_and_system_default(origin, tmp_path, capsys, monkeypatch):
    home = clone(origin, tmp_path / "me")
    py = fake_python(tmp_path, "system")
    monkeypatch.setenv("MACE_PYTHON", str(py))
    code, out = _interactive(home, ["y", ""], capsys)          # Enter: the default, 2 for a system Python
    assert code == 0 and _pip_log(py) == ""
    assert f"{py} -m pip install --user -r {home}/requirements.txt" in out
    assert "python3 -m venv ~/mace-env" in out and "conda create -n mace" in out
    venv = clone(origin, tmp_path / "v")
    vpy = fake_python(tmp_path, "venv")
    monkeypatch.setenv("MACE_PYTHON", str(vpy))
    code, out = _interactive(venv, ["y", "2"], capsys)
    assert code == 0 and _pip_log(vpy) == "" and f"{vpy} -m pip install -r {venv}/requirements.txt" in out


def test_yes_and_no_terminal_never_run_pip(origin, tmp_path, monkeypatch):
    py = fake_python(tmp_path, "venv")
    monkeypatch.setenv("MACE_PYTHON", str(py))
    r = _run(clone(origin, tmp_path / "me"), "--yes")
    assert r.returncode == 0 and _pip_log(py) == "" and "not done for you" in r.stdout
    out = []
    monkeypatch.setattr(U, "say", lambda t="": out.append(t))
    U.update(clone(origin, tmp_path / "me2"), "release", yes=True, ask=lambda p: "1", tty=True)
    assert _pip_log(py) == "" and not any("[1] run it now" in ln for ln in out)


def test_pip_and_no_pip_flags(origin, tmp_path, monkeypatch):
    py = fake_python(tmp_path, "venv")
    monkeypatch.setenv("MACE_PYTHON", str(py))
    r = _run(clone(origin, tmp_path / "a"), "--yes", "--pip")
    assert r.returncode == 0 and "-m pip install -r" in _pip_log(py) and "The requirements are installed." in r.stdout
    (py.parent / "pip.log").unlink()
    out = []
    monkeypatch.setattr(U, "say", lambda t="": out.append(t))
    U.update(clone(origin, tmp_path / "b"), "release", ask=lambda prompt: "y", tty=True, pip=False)
    assert _pip_log(py) == "" and not any("[1]" in ln for ln in out)
    assert any("not done for you" in ln for ln in out)


def test_failed_pip_keeps_the_update(origin, tmp_path, monkeypatch):
    py = fake_python(tmp_path, "venv")
    monkeypatch.setenv("MACE_PYTHON", str(py))
    monkeypatch.setenv("FAKE_PIP_RC", "1")
    home = clone(origin, tmp_path / "me")
    r = _run(home, "--yes", "--pip")
    assert r.returncode == 0, r.stdout
    assert "pip failed (exit 1). The MACE code IS updated, but its packages are NOT" in r.stdout
    assert f"  {py} -m pip install -r {home}/requirements.txt" in r.stdout
    assert git(home, "describe", "--tags") == "v1.1.5" and "Check: MACE v1.1.5" in r.stdout


def test_remote_passes_pip_through(origin, tmp_path):
    cluster_home = tmp_path / "cluster"
    cluster_home.mkdir()
    clone(origin, cluster_home / "MACE")
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    from mace.gui import connect as C
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "hpcc"}, "profiles": {"hpcc": {"gateway": "me@gw.example.edu"}}},
                    cfg / "mace" / "connect.toml")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(PATH=f"{fb}:{env['PATH']}", XDG_CONFIG_HOME=str(cfg), FAKE_CLUSTER_HOME=str(cluster_home),
               FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    for flag in ("--pip", "--no-pip"):
        r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote-only",
                            "--dry-run", flag], env=env, capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (tmp_path / "ssh.log").read_text().splitlines()[-1].endswith(f"--channel release --dry-run {flag}'")


# ---- a release retagged on origin (v1.1.5 was the preview, now the final)

def _retagged(origin, tmp_path):
    final = git(origin, "rev-parse", "v1.1.5^{commit}")
    git(origin, "tag", "-f", "v1.1.5", "v1.1.5-preview")          # what users cloned: v1.1.5 = the preview
    home = clone(origin, tmp_path / "me")
    git(origin, "tag", "-f", "v1.1.5", final)                      # then moved to the final commit
    return home, final


def test_moved_release_tag_is_followed(origin, tmp_path):
    home, final = _retagged(origin, tmp_path)
    assert git(home, "rev-parse", "v1.1.5^{commit}") != final        # the stale local tag
    r = _run(home, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout
    assert "Note: v1.1.5 moved on origin (it was the commit now tagged v1.1.5-preview); updated the local tag " \
           "v1.1.5." in r.stdout
    assert git(home, "rev-parse", "HEAD") == final == git(home, "rev-parse", "v1.1.5^{commit}")
    assert git(home, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    git(home, "pull", "-q", "--ff-only")
    assert git(home, "rev-parse", "HEAD") == git(origin, "rev-parse", "main")


def test_moved_release_tag_in_check(origin, tmp_path):
    home, final = _retagged(origin, tmp_path)
    git(home, "reset", "-q", "--hard", "v1.1.5-preview")              # on the old v1.1.5
    r = _run(home, "--check")
    assert r.returncode == 0, r.stdout
    assert "Note: v1.1.5 moved on origin" in r.stdout
    assert any(ln.startswith("  release  v1.1.5 (1.1.5): available (1 commit ahead)") for ln in r.stdout.splitlines())
    assert git(home, "rev-parse", "HEAD") == git(home, "rev-parse", "v1.1.5-preview^{commit}")   # --check moved nothing


def test_moved_release_tag_on_the_cluster(origin, tmp_path):
    final = git(origin, "rev-parse", "v1.1.5^{commit}")
    git(origin, "tag", "-f", "v1.1.5", "v1.1.5-preview")
    cluster_home = tmp_path / "cluster"
    cluster_home.mkdir()
    remote = clone(origin, cluster_home / "MACE")
    git(origin, "tag", "-f", "v1.1.5", final)
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    from mace.gui import connect as C
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "hpcc"}, "profiles": {"hpcc": {"gateway": "me@gw.example.edu"}}},
                    cfg / "mace" / "connect.toml")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(PATH=f"{fb}:{env['PATH']}", XDG_CONFIG_HOME=str(cfg), FAKE_CLUSTER_HOME=str(cluster_home),
               FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote-only", "--yes",
                        "--no-pip"], env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Note: v1.1.5 moved on origin" in r.stdout
    assert git(remote, "rev-parse", "HEAD") == final


def test_remote_host_is_never_read_as_an_ssh_option():
    """A profile's gateway that starts with "-" is the host, not an option: "--" ends ssh's options."""
    argv = U.remote_argv({"gateway": "-oProxyCommand=touch /tmp/x"}, [], False)
    assert argv[-3:-1] == ["--", "-oProxyCommand=touch /tmp/x"]
    argv = U.remote_argv({"gateway": "me@gw.example.edu", "jump": "me@bastion"}, ["--check"], True)
    assert argv[0] == "ssh" and argv[-3:-1] == ["--", "me@gw.example.edu"] and "-J" in argv[:argv.index("--")]


# ---- 1.1.6 review: requirement files, Python 3.6 clusters, read-only checkouts, --remote --local

def _release(origin, tmp_path, tag, files):
    """A new release ``tag`` on origin that changes ``files`` ({name: text})."""
    work = tmp_path / ("w-" + tag)
    git(tmp_path, "clone", "-q", str(origin), str(work))
    for name, text in files.items():
        (work / name).write_text(text)
    (work / "mace" / "__init__.py").write_text(f'__version__ = "{tag[1:]}"\n')
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", tag)
    git(work, "tag", tag)
    git(work, "push", "-q", "origin", "HEAD:main", tag)


def test_test_requirement_files_never_reach_pip(origin, tmp_path):
    _release(origin, tmp_path, "v1.1.7", {"requirements-gui-test.txt": "selenium\naxe-selenium-python\n",
                                          "requirements-test.txt": "pytest\n",
                                          "requirements-optional.txt": "ase\nphonopy\n"})
    home = clone(origin, tmp_path / "me", at="v1.1.6-preview")
    r = _run(home, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout
    assert "requirements-gui-test" not in r.stdout and "requirements-test" not in r.stdout
    assert "pip install" not in r.stdout                   # requirements.txt did not change
    (line,) = [ln for ln in r.stdout.splitlines() if "requirements-optional.txt" in ln]
    assert "changed" in line and "-r" not in line.split("requirements-optional.txt")[0].split()[-1:]


def test_only_requirements_txt_is_in_the_pip_command(origin, tmp_path):
    _release(origin, tmp_path, "v1.1.7", {"requirements.txt": "numpy\nase\nscipy\n",
                                          "requirements-gui-test.txt": "selenium\n",
                                          "requirements-optional.txt": "phonopy\n"})
    home = clone(origin, tmp_path / "me", at="v1.1.6-preview")
    r = _run(home, "--channel", "release", "--yes")
    assert r.returncode == 0, r.stdout
    pip = [ln for ln in r.stdout.splitlines() if "pip install" in ln]
    assert pip and all("requirements.txt" in ln and "requirements-" not in ln for ln in pip), r.stdout
    assert "+ scipy" in r.stdout and "selenium" not in r.stdout
    assert sum("requirements-optional.txt" in ln for ln in r.stdout.splitlines()) == 1


def test_update_py_runs_on_python_36():
    """--remote runs this file with the cluster's python3, 3.6 on RHEL/Rocky 8."""
    import ast
    src = Path(U.__file__).read_text()
    assert "from __future__ import annotations" not in src and "shlex.join" not in src
    assert "capture_output" not in src and "text=True" not in src
    tree = ast.parse(src, feature_version=(3, 6))
    assert not any(isinstance(n, ast.NamedExpr) for n in ast.walk(tree))


def test_remote_payload_is_small(origin, tmp_path):
    """One argv entry on Windows (CreateProcess: 32,767 characters in all)."""
    prof = {"gateway": "me@gw.example.edu", "mace": "python3 ~/MACE/mace_cli", "setup": "module load Python"}
    argv = U.remote_argv(prof, ["--channel", "release", "--yes"], False)
    total = len(" ".join(argv))
    assert total < 16000, total
    assert "zlib" in argv[-1]


FAKE_NEW_MACE = """#!/bin/sh
case "$*" in
  "update --help") echo "usage: mace update [--remote-only] ..." ;;
  update*) echo "CLUSTER MACE UPDATE $*" ;;
  *) echo "unexpected $*"; exit 9 ;;
esac
"""


def test_remote_prefers_the_clusters_own_mace_update(origin, tmp_path):
    cluster_home = tmp_path / "cluster"
    (cluster_home / "bin").mkdir(parents=True)
    (cluster_home / "bin" / "mace").write_text(FAKE_NEW_MACE)
    (cluster_home / "bin" / "mace").chmod(0o755)
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    from mace.gui import connect as C
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "c"},
                     "profiles": {"c": {"gateway": "me@gw.example.edu", "mace": "~/bin/mace"}}},
                    cfg / "mace" / "connect.toml")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(PATH=f"{fb}:{env['PATH']}", XDG_CONFIG_HOME=str(cfg), FAKE_CLUSTER_HOME=str(cluster_home),
               FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote-only",
                        "--dry-run"], env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CLUSTER MACE UPDATE update --channel release --dry-run" in r.stdout, r.stdout


def _read_only(path, on):
    for p in [path, *path.rglob("*")]:
        if p.is_symlink():
            continue
        m = p.stat().st_mode
        p.chmod((m & ~0o222) if on else (m | 0o200))


def test_read_only_checkout(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    _read_only(home / ".git", True)
    try:
        r = _run(home, "--check")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "not writable by you" in r.stdout
        assert any(ln.startswith("  release  v1.1.5") and "available" in ln for ln in r.stdout.splitlines()), r.stdout
        r = _run(home, "--yes")
        assert r.returncode == 2 and "not writable by you" in r.stdout and "FETCH_HEAD" not in r.stdout
    finally:
        _read_only(home / ".git", False)
    assert git(home, "describe", "--tags") == "v1.1.4"


def test_remote_and_local_says_when_a_local_refusal_skips_the_cluster(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mace_cli").write_text("edited\n")
    from mace.gui import connect as C
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "c"}, "profiles": {"c": {"gateway": "me@gw.example.edu"}}},
                    cfg / "mace" / "connect.toml")
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    env = dict(os.environ, PATH=f"{fb}:{os.environ['PATH']}", XDG_CONFIG_HOME=str(cfg),
               FAKE_CLUSTER_HOME=str(tmp_path), FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    r = _run(home, "--remote", "--local", "--yes", env=env)
    assert r.returncode == 2
    assert "these files have local changes" in r.stdout
    assert "the cluster (c) was not updated" in r.stdout
    assert not (tmp_path / "ssh.log").exists()


def _pwsh():
    return __import__("shutil").which("pwsh")


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
def test_powershell_manual_does_not_drop_the_update(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BROWSER"}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), APPDATA=str(tmp_path / "c"))
    r = subprocess.run([_pwsh(), "-NoProfile", "-File", str(PS1), "-Gateway", "me@gw", "-Node", "n1", "-Manual",
                        "-UpdateRemote", "preview"], env=env, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "Updating MACE on me@gw (preview) ..." in out
    assert out.index("Updating MACE on") < out.index("1. In one terminal")


def test_powershell_update_ssh_has_a_timeout_and_t_only_with_a_console():
    ps = PS1.read_text()
    body = ps[ps.index("function Invoke-UpdateRemote"):ps.index("\nif ($Manual) {")]
    assert "Ct-Setup $uhost" in body and "@($ct.Opts)" in body
    assert 'if ($interactive) { $ua += @("-t") } else { $ua += @("-o", "BatchMode=yes") }' in body
    assert "HomePath $_" in body                     # each word of the mace command quoted
