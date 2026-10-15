"""The local test corpus (test/, 12+ GB) is never committed, whether it is a
folder or, as in most worktrees, a symbolic link to the one copy. Git's
`test/` pattern matches only a directory, so a symlinked corpus showed up in
`git status`; `/test` covers it."""
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    root = tmp_path / "repo"
    root.mkdir()
    assert _git(root, "init", "-q").returncode == 0
    shutil.copy(REPO_ROOT / ".gitignore", root / ".gitignore")
    _git(root, "add", ".gitignore")
    return root


def _untracked(root):
    out = _git(root, "status", "--porcelain", "--untracked-files=all").stdout
    return [l[3:] for l in out.splitlines() if l.startswith("??")]


def test_symlinked_corpus_is_ignored(repo, tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "OPT").mkdir(parents=True)
    (corpus / "OPT" / "x.out").write_text("output\n")
    (repo / "test").symlink_to(corpus, target_is_directory=True)
    assert _untracked(repo) == []
    assert _git(repo, "check-ignore", "-q", "test").returncode == 0


def test_corpus_folder_is_ignored(repo):
    (repo / "test" / "OPT").mkdir(parents=True)
    (repo / "test" / "OPT" / "x.out").write_text("output\n")
    assert _untracked(repo) == []


def test_tests_folder_is_not_ignored(repo):
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("")
    assert _untracked(repo) == ["tests/test_x.py"]
