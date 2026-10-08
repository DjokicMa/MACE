"""Commits the tests read with git (oracle trees, frozen files) must exist in
every clone.

The GUI's development history is squashed for the release, so a test that
runs `git show <commit>` on a development commit skips silently in a public
clone. Every commit a test names must be the 1.1.5 base (daa5468e) or one
of its ancestors; content that only exists later is frozen under tests/
instead (tests/gui/frozen).
"""
import re
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

RELEASE_BASE = "daa5468e"
QUOTED_HEX = re.compile(r"""["']([0-9a-f]{7,40})(?:[\^:~][^"']*)?["']""")


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def test_every_commit_a_test_names_is_reachable_from_the_release_base():
    if shutil.which("git") is None or _git("cat-file", "-e", RELEASE_BASE + "^{commit}").returncode:
        pytest.skip("no git history here")
    bad = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        if "frozen" in path.parts:
            continue
        for h in set(QUOTED_HEX.findall(path.read_text(errors="replace"))):
            if _git("cat-file", "-e", h + "^{commit}").returncode:
                continue                       # not a commit of this repository
            if _git("merge-base", "--is-ancestor", h, RELEASE_BASE).returncode:
                bad.append(f"{path.relative_to(REPO_ROOT)}: {h}")
    assert not bad, bad
