"""The named folders the GUI shortens paths with (shared rule S4 of the 1.1.5
polish plans; Open Folder's allowed roots use the same table): HOME as
written and resolved, and $SCRATCH, $WORK, $PSCRATCH when set, non-empty and
existing. Plus the page's alias-aware abbreviation, and /api/info."""
import json
import os
import subprocess
import sys
import urllib.request

import pytest

from mace.gui.paths import root_aliases

# (case, env builder -> env, expected [(name, path)] builder)
def _dirs(tmp_path):
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    link = tmp_path / "home-link"
    link.symlink_to(real_home)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    work = scratch / "work"           # nested inside SCRATCH
    work.mkdir()
    return real_home, link, scratch, work


ALIAS_TABLE = [
    # name, env, expected (name, path) list
    ("home literal and resolved", lambda d: {"HOME": str(d[1])},
     lambda d: [("HOME", str(d[1])), ("HOME", str(d[0]))]),
    ("each variable set and existing", lambda d: {"HOME": str(d[0]), "SCRATCH": str(d[2]), "WORK": str(d[3]),
                                                   "PSCRATCH": str(d[2] / "p")},
     lambda d: [("HOME", str(d[0])), ("SCRATCH", str(d[2])), ("WORK", str(d[3]))]),
    ("set but missing", lambda d: {"HOME": str(d[0]), "SCRATCH": str(d[2] / "nope")},
     lambda d: [("HOME", str(d[0]))]),
    ("set to empty", lambda d: {"HOME": str(d[0]), "SCRATCH": "", "WORK": "  "},
     lambda d: [("HOME", str(d[0]))]),
    ("two variables, one folder", lambda d: {"HOME": str(d[0]), "SCRATCH": str(d[2]), "PSCRATCH": str(d[2])},
     lambda d: [("HOME", str(d[0])), ("SCRATCH", str(d[2]))]),
    ("relative path ignored", lambda d: {"HOME": str(d[0]), "WORK": "work"},
     lambda d: [("HOME", str(d[0]))]),
]


@pytest.mark.parametrize("case,env,want", ALIAS_TABLE, ids=[c[0] for c in ALIAS_TABLE])
def test_root_aliases_table(tmp_path, case, env, want):
    d = _dirs(tmp_path)
    got = [(a["name"], a["path"]) for a in root_aliases(env(d))]
    assert got == want(d), case


def test_info_sends_only_home_without_scratch_variables(tmp_path):
    """Smoke through the real entry point: /api/info has `aliases`, HOME only."""
    sys.path.insert(0, os.path.dirname(__file__))
    import guiharness as H
    env_extra = {}
    for k in ("SCRATCH", "WORK", "PSCRATCH"):
        os.environ.pop(k, None)
    root = tmp_path / "d"
    root.mkdir()
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env_extra)
    try:
        req = urllib.request.Request(base + "api/info", headers={"X-Mace-Token": H.TOKEN})
        info = json.loads(urllib.request.urlopen(req, timeout=20).read())
        assert info["aliases"] and {a["name"] for a in info["aliases"]} == {"HOME"}
        assert "home" in info
    finally:
        proc.terminate()
        proc.wait(timeout=15)
