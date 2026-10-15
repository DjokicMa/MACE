"""dirwatch.py, the Explorer's refresh: a check is one stat() per watched
folder (no listing) until something changed; it sees a file added or removed
in any listed folder, a new subfolder's files after that, and leaves out what
the Explorer does not list; at most max_dirs folders, the shallowest first."""
import os
import sys

import guiharness as H

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import dirwatch  # noqa: E402
from mace.gui import server  # noqa: E402


_BUMPS = [0]


def _bump(path):
    """A folder's mtime set to a value it never had (two changes in one clock
    tick of the filesystem would otherwise give the same mtime)."""
    _BUMPS[0] += 1
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + _BUMPS[0] * 10_000_000_000))


def _watch(root, **kw):
    return dirwatch.DirWatch(root, server.MAX_DEPTH, server._list_skip, **kw)


def test_a_check_without_changes_lists_nothing(tmp_path, monkeypatch):
    for d in ("a/b/c", "d"):
        (tmp_path / d).mkdir(parents=True)
    w = _watch(tmp_path)
    assert len(w.stamps) == 5
    calls = []
    real = os.scandir
    monkeypatch.setattr(dirwatch.os, "scandir", lambda *a: calls.append(a) or real(*a))
    assert not w.changed() and not w.changed()
    assert calls == []


def test_added_and_removed_files_anywhere_listed(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    w = _watch(tmp_path)
    (tmp_path / "a" / "b" / "x.out").write_text("x")
    _bump(tmp_path / "a" / "b")
    assert w.changed() and not w.changed()
    (tmp_path / "a" / "new").mkdir()                    # a new folder: watched from then on
    _bump(tmp_path / "a")
    assert w.changed()
    (tmp_path / "a" / "new" / "y.out").write_text("y")
    _bump(tmp_path / "a" / "new")
    assert w.changed()
    (tmp_path / "a" / "b" / "x.out").unlink()
    _bump(tmp_path / "a" / "b")
    assert w.changed()


def test_unlisted_folders_are_not_watched(tmp_path):
    for d in (".git/objects", "__pycache__", ".hidden", "ok"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "ok")
    deep = tmp_path.joinpath(*[f"l{i}" for i in range(server.MAX_DEPTH + 2)])
    deep.mkdir(parents=True)
    w = _watch(tmp_path)
    names = {os.path.relpath(d, tmp_path) for d in w.stamps}
    assert not any(n.startswith((".git", "__pycache__", ".hidden", "link")) for n in names), names
    assert max(n.count(os.sep) for n in names if n.startswith("l0")) == server.MAX_DEPTH - 1
    _bump(tmp_path / ".hidden")
    _bump(deep)
    assert not w.changed()


def test_at_most_max_dirs_shallowest_first(tmp_path):
    for i in range(30):
        (tmp_path / f"d{i:02d}" / "sub").mkdir(parents=True)
    w = _watch(tmp_path, max_dirs=20)
    assert len(w.stamps) == 20
    assert all(os.path.dirname(d) == str(tmp_path) for d in w.stamps if d != str(tmp_path))
