"""`mace update --check --json` (what MACE GUI's update notice reads): one
JSON object on stdout for one channel, with the current and the new version,
the CHANGELOG sections in between, the refusal `mace update` would print
(word for word), local changes with the git stash line, the requirements
change with its pip command, and the queue counts. On the bare-origin fixture
of test_update.py."""
import json
import os
import subprocess
import sys

import pytest

from mace import update as U
from test_update import REPO, clone, git, origin  # noqa: F401 - the fixture

pytestmark = pytest.mark.skipif(not __import__("shutil").which("git"), reason="needs git")


def _json(home, *args):
    r = subprocess.run([sys.executable, "-c", "import sys; from mace.update import main; sys.exit(main())",
                        "--home", str(home), "--check", "--json", *args], cwd=str(REPO), capture_output=True,
                       text=True, stdin=subprocess.DEVNULL, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)          # the whole stdout is the one object


def test_available_release_with_changelog_and_requirements(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    d = _json(home, "--channel", "release")
    assert d["channel"] == "release" and d["state"] == "available" and d["available"] is True
    assert d["current"]["version"] == "1.1.4" and d["current"]["branch"] == "main"
    assert d["target"]["ref"] == "v1.1.5" and d["target"]["version"] == "1.1.5" and d["target"]["commits"] == 2
    text = "\n".join(d["changelog"])
    assert "## [1.1.5]" in text and "## [1.1.5-preview]" in text and "## [1.1.4]" not in text
    assert d["refusal"] is None and d["writable"] is True and d["local_changes"] == []
    req = d["requirements"]
    assert req["files"] == ["requirements.txt"] and "ase" in req["changes"]
    assert f"-m pip install" in req["pip"] and f"-r {home}/requirements.txt" in req["pip"]
    assert req["python_kind"] in ("venv", "conda", "system")
    assert d["how"].startswith("git merge --ff-only v1.1.5")
    assert git(home, "rev-parse", "HEAD") == git(home, "rev-parse", "v1.1.4")     # nothing moved


def test_preview_and_latest_channels(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    assert _json(home, "--channel", "preview")["target"]["ref"] == "v1.1.6-preview"
    d = _json(home, "--channel", "latest")
    assert d["target"]["ref"] == "origin/main" and "## [Unreleased]" in "\n".join(d["changelog"])


def test_current_and_newer(origin, tmp_path):
    up = clone(origin, tmp_path / "up", at="v1.1.5")
    d = _json(up, "--channel", "release")
    assert d["state"] == "current" and d["available"] is False and d["changelog"] == []
    newer = clone(origin, tmp_path / "newer", at="origin/main")
    d = _json(newer, "--channel", "release")
    assert d["state"] == "newer" and d["available"] is False
    assert d["message"].startswith("This checkout is newer than v1.1.5 (1.1.5); nothing to do")


def test_local_changes_give_the_exact_refusal_and_the_stash_line(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mace_cli").write_text("edited\n")
    d = _json(home)
    assert d["available"] is True and d["refusal_kind"] == "local_changes"
    with pytest.raises(U.Refused) as exc:
        U.plan(U.Git(home), "release")
    assert d["refusal"] == str(exc.value)
    assert d["local_changes"] == [" M mace_cli"] and d["stash"] == f"git -C {home} stash"


def test_local_commits_refusal(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mine.txt").write_text("x")
    git(home, "add", "mine.txt")
    git(home, "commit", "-q", "-m", "my own change")
    d = _json(home)
    assert d["state"] == "diverged" and d["refusal_kind"] == "not_ff"
    assert "not a fast-forward" in d["refusal"] and "my own change" in d["refusal"]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anything")
def test_read_only_checkout_says_ask_the_maintainer(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    gd = home / ".git"
    modes = {}
    for p in [gd, gd / "objects", gd / "refs", gd / "HEAD", gd / "FETCH_HEAD"]:
        if p.exists():
            modes[p] = p.stat().st_mode
            p.chmod(p.stat().st_mode & ~0o222)
    try:
        d = _json(home)
    finally:
        for p, m in modes.items():
            p.chmod(m)
    assert d["writable"] is False and d["refusal_kind"] == "not_writable"
    assert d["refusal"] == U.not_writable(home)
    assert d["available"] is True and d["target"]["ref"] == "v1.1.5"


def test_moved_tag_note_stays_out_of_stdout(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    git(home, "fetch", "-q", "--tags", "origin")
    # origin retags v1.1.5 (a preview renamed to the final, as for 1.1.5)
    git(origin, "tag", "-f", "v1.1.5", "v1.1.5-preview")
    d = _json(home)
    assert any("v1.1.5 moved on origin" in n for n in d["notes"])


def test_json_needs_check(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    r = subprocess.run([sys.executable, "-c", "import sys; from mace.update import main; sys.exit(main())",
                        "--home", str(home), "--json"], cwd=str(REPO), capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=60)
    assert r.returncode == 2 and "--json goes with --check" in r.stderr


def test_no_git_checkout_is_json_too(tmp_path):
    d = _json(tmp_path)
    assert d["refusal_kind"] == "not_git" and "not a git checkout" in d["refusal"]
