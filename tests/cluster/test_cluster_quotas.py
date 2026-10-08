"""F4: quota parsing for each source, the alias safety rule, once-a-day
discovery and the 80/90 levels."""
import os
import stat
import sys
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import (DF_I, DF_K, LFS_QUOTA, QUOTA_SW, SITE_QUOTA_TABLE, fake, use_env)

from mace.cluster import config, query, quotas

GB = 1024 ** 3


def test_site_table_numbers():
    """F4-a: ANSI colors, box drawing, 1.0M/342.0K/87.1G parse to exact
    numbers; the printed percentages agree with the numbers."""
    rows = {r["label"]: r for r in quotas.table_parser(SITE_QUOTA_TABLE)}
    h = rows["home (NFS)"]
    assert h["space_used_b"] == pytest.approx(87.1 * GB) and h["space_limit_b"] == 100 * GB
    assert h["files_used"] == 342_000 and h["files_limit"] == 1_000_000
    assert h["space_pct"] == 87.1 and h["files_pct"] == 34.2
    for r in (rows["home (NFS)"], rows["scratch"]):
        assert abs(100 * r["space_used_b"] / r["space_limit_b"] - r["space_pct"]) <= 0.1
        assert abs(100 * r["files_used"] / r["files_limit"] - r["files_pct"]) <= 0.1
    assert quotas.levels(h, 80, 90) == "warning"
    assert quotas.levels(rows["scratch"], 80, 90) == "critical"
    assert quotas.levels(rows["project_space"], 80, 90) == "ok"


def test_other_sources():
    """F4-b: lfs (with the * over-quota mark), quota -s -w, df -P -k/-i."""
    l = quotas.lfs_parser(LFS_QUOTA)[0]
    assert l["space_used_b"] == 9_500_000 * 1024 and l["space_pct"] > 100 and l["grace"] == "6d23h"
    q = quotas.quota_parser(QUOTA_SW)[0]
    assert q["space_used_b"] == 89716 * 1024 ** 2 and q["space_limit_b"] == 100 * GB
    assert q["files_used"] == 342_000 and q["files_limit"] == 1_000_000
    d = quotas.df_parser(DF_K, DF_I)[0]
    assert d["kind"] == "filesystem" and "not your quota" in d["label"]
    assert d["space_pct"] == 87.0 and d["files_pct"] == 40.0
    mm = quotas.mmlsquota_parser(
        "                         Block Limits                                    |     File Limits\n"
        "Filesystem Fileset    type             KB      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks\n"
        "scr1       root       USR          900000    1000000    1100000          0     none |   95000  100000   110000        0     none\n")
    assert mm[0]["space_pct"] == 90.0 and mm[0]["files_pct"] == 95.0


@pytest.fixture
def tool(tmp_path):
    """A site quota tool in a read-only folder outside HOME."""
    d = tmp_path / "site" / "bin"
    d.mkdir(parents=True)
    t = d / "userquota.py"
    t.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({SITE_QUOTA_TABLE!r} + '\\n')\n")
    t.chmod(0o555)
    d.chmod(0o555)
    yield t
    d.chmod(0o755)


def test_alias_safety(tool, tmp_path):
    """F4-c."""
    home = tmp_path / "home"
    home.mkdir()
    ok = quotas.check_alias(f"NO_COLOR=1 {tool}", home=str(home))
    assert ok[0] == [str(tool)] and ok[1] == {"NO_COLOR": "1"} and ok[2] is None
    # clearing LD_LIBRARY_PATH (a real site's alias) is allowed: it can only remove code
    ok = quotas.check_alias(f'LD_LIBRARY_PATH="" {tool}', home=str(home))
    assert ok[0] == [str(tool)] and ok[1] == {"LD_LIBRARY_PATH": ""}
    assert "may change which code runs" in quotas.check_alias(f"LD_PRELOAD=x {tool}", home=str(home))[2]
    assert "may change which code runs" in quotas.check_alias(f"PYTHONPATH=x {tool}", home=str(home))[2]
    assert "add it to [quota] alias_env_allow" in quotas.check_alias(f"FOO=1 {tool}", home=str(home))[2]
    assert quotas.check_alias(f"FOO=1 {tool}", home=str(home), extra_allow=["FOO"])[2] is None
    assert "not an absolute path" in quotas.check_alias("tool.py", home=str(home))[2]
    assert "shell syntax" in quotas.check_alias(f"{tool}; rm -rf x", home=str(home))[2]
    assert "shell syntax" in quotas.check_alias(f"$(echo {tool})", home=str(home))[2]
    mine = home / "q.py"
    mine.write_text("x")
    mine.chmod(0o555)
    assert "inside your home folder" in quotas.check_alias(str(mine), home=str(home))[2]
    w = tmp_path / "w.py"
    w.write_text("x")
    assert "writable by you" in quotas.check_alias(str(w), home=str(home))[2]
    # a listed refused name is a site.toml load error
    p = tmp_path / "site.toml"
    p.write_text('[quota]\nalias_env_allow = ["PYTHONPATH"]\n')
    assert config.load({"MACE_SITE_CONFIG": str(p)}).errors


def _fake_bash(tmp_path, log):
    b = tmp_path / "fakebash"
    b.mkdir()
    (b / "bash").write_text(f"#!/bin/bash\necho call >> {log}\nexec /bin/bash \"$@\"\n")
    (b / "bash").chmod(0o755)
    return b


def test_discovery_once_a_day(tool, tmp_path, monkeypatch):
    """F4-e: discovery runs bash -lc once; later reads use the cached argv."""
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text(f"alias quota='NO_COLOR=1 {tool}'\n")
    log = tmp_path / "bash.log"
    b = _fake_bash(tmp_path, log)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    clockf = tmp_path / "clock"
    clockf.write_text("2000000")
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    monkeypatch.setenv("MACE_CLUSTER_TEST_CLOCK", str(clockf))
    query._which_cache.clear()
    for i in range(100):
        clockf.write_text(str(2000000 + i * 601))         # past the 10 min cache each time, within the day
        r = quotas.read()
        assert "site tool" in r.value["sources"], r.value
    assert log.read_text().count("call") == 1
    lv = {x["label"]: x["level"] for x in r.value["rows"]}
    assert lv["home (NFS)"] == "warning" and lv["scratch"] == "critical"
    # after a day: one more discovery
    clockf.write_text(str(2000000 + 86400 + 700))
    quotas.read()
    assert log.read_text().count("call") == 2


def test_discovery_timeout_falls_through(tool, tmp_path, monkeypatch):
    """F4-e: a profile that prints and sleeps: discovery times out and the
    chain falls through to quota/df with the note."""
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text("echo hello from profile\nsleep 30\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(quotas, "DISCOVERY_TIMEOUT", 1.0)
    query._which_cache.clear()
    data = quotas.collect()
    assert any("timed out" in n for n in data["notes"]), data
    assert "site tool" not in data["sources"]
    assert "filesystem" in data["sources"]


def test_warn_levels_from_site_toml(tmp_path):
    cfgp = tmp_path / "site.toml"
    cfgp.write_text("[quota]\nwarn_percent = 85\n")
    cfg = config.load({"MACE_SITE_CONFIG": str(cfgp)})
    assert quotas.levels({"space_pct": 87.1}, cfg.get("quota", "warn_percent"), 90) == "warning"
    assert quotas.levels({"space_pct": 84.0}, cfg.get("quota", "warn_percent"), 90) == "ok"
