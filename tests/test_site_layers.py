"""Site settings in layers (PLAN_G G0): group files, the user's site.toml and
the environment, per key; permissions of group files; fail closed when the
settings cannot be read (Python without tomllib).

Precedence: env > ~/.config/mace/site.toml > group files (/etc/mace/site.toml,
$XDG_CONFIG_DIRS/mace/site.toml with the first directory winning, or the
files MACE_SITE_GROUP_CONFIG names instead) > built-in."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from test_site_jobscript import SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site

def _group(tmp, text, name="lab", mode=0o644, dmode=0o755):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "site.toml"
    f.write_text(text)
    f.chmod(mode)
    d.chmod(dmode)
    return f


def test_precedence_env_user_group(tmp_path):
    g = _group(tmp_path, '[slurm]\naccount = "grp_group"\npartition = "gpart"\n'
                         '[modules]\ncrystal = "CRYSTAL/23-group"\n')
    _toml(tmp_path, '[slurm]\npartition = "upart"\n')
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g), MACE_SLURM_ACCOUNT="grp_env")
    r = _mace(tmp_path, env, "site", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(r.stdout)
    assert (d["account"]["value"], d["account"]["source"]) == ("grp_env", "env MACE_SLURM_ACCOUNT")
    assert (d["partition"]["value"], d["partition"]["source"]) == ("upart", "site.toml")
    assert d["partition"]["origin"] == str(tmp_path / "xdg" / "mace" / "site.toml")
    assert d["crystal"]["preferred"] == ["CRYSTAL/23-group"] and d["crystal"]["preferred_source"] == "group"
    assert d["config"]["group"] == [{"path": str(g), "owner": site._owner(os.getuid())}]
    text = _mace(tmp_path, env, "site").stdout
    assert f"Group file:  {g}" in text and f"Partition:   upart  (site.toml: {tmp_path}" in text


def test_group_values_reach_the_job_script(tmp_path):
    g = _group(tmp_path, '[slurm]\naccount = "grp_group"\n[resources]\nntasks = 48\n')
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g))
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    sh = (d / "m.sh").read_text()
    assert "#SBATCH -A grp_group" in sh and "#SBATCH --ntasks=48" in sh


def test_site_config_none_ignores_group_files_too(tmp_path):
    g = _group(tmp_path, '[slurm]\naccount = "grp_group"\n')
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g), MACE_SITE_CONFIG="none")
    s = site.load(env)
    assert s.account == "mendoza_q" and s.sources["account"] == "built-in"
    assert site.config_layers(env) == []


def test_xdg_config_dirs_first_wins_and_system_file_lowest(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, acct, part in ((a, "grp_a", None), (b, "grp_b", "pb")):
        (d / "mace").mkdir(parents=True)
        body = f'[slurm]\naccount = "{acct}"\n' + (f'partition = "{part}"\n' if part else "")
        (d / "mace" / "site.toml").write_text(body)
        (d / "mace" / "site.toml").chmod(0o644)
        (d / "mace").chmod(0o755)
    env = _env(tmp_path, XDG_CONFIG_DIRS=f"{a}:{b}")
    env.pop("MACE_SITE_GROUP_CONFIG")
    sysf = tmp_path / "etc" / "site.toml"
    sysf.parent.mkdir()
    sysf.write_text('[slurm]\naccount = "grp_sys"\npartition = "psys"\nqos = "qsys"\n')
    monkeypatch.setattr(site, "SYSTEM_FILE", sysf)
    layers = site.config_layers(env)
    assert [p for _, p, _ in layers][:3] == [sysf, b / "mace" / "site.toml", a / "mace" / "site.toml"]
    s = site.load(env)
    assert (s.account, s.partition, s.qos) == ("grp_a", "pb", "qsys")       # per key
    assert s.sources["account"] == "group" and s.origins["account"] == str(a / "mace" / "site.toml")


def test_foreign_tables_give_no_warning_and_load_tables_merges(tmp_path):
    g = _group(tmp_path, '[cluster]\nnode_query = "auto"\n[quota]\nwarn = 80\n')
    _toml(tmp_path, '[cluster]\nnode_query = "sinfo"\n[devnodes]\nnames = ["dev1"]\n')
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g))
    s = site.load(env)
    assert s.warnings == []
    data, path, text = site.load_tables(env)
    assert data == {"cluster": {"node_query": "sinfo"}, "quota": {"warn": 80}, "devnodes": {"names": ["dev1"]}}
    assert path == tmp_path / "xdg" / "mace" / "site.toml" and "devnodes" in text


# ---------------------------------------------------------------- permissions (D6.3)
def test_group_writable_group_file_is_refused(tmp_path):
    g = _group(tmp_path, '[slurm]\naccount = "grp_group"\n', mode=0o664)
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g))
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 2, p.stdout + p.stderr
    assert str(g) in p.stdout and "0664" in p.stdout
    assert not (d / "m.sh").exists()


def test_group_writable_directory_is_refused(tmp_path):
    g = _group(tmp_path, '[slurm]\naccount = "grp_group"\n', dmode=0o775)
    env = _env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g))
    with pytest.raises(site.SiteConfigError, match="its directory .* is writable by group or others"):
        site.load(env)
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 2 and "refused" in r.stdout


def test_another_users_file_named_explicitly_is_accepted(tmp_path, monkeypatch):
    g = _group(tmp_path, '[slurm]\naccount = "grp_pi"\n')
    real = Path.stat

    def fake_stat(self, *a, **k):
        st = real(self, *a, **k)
        if self == g:
            vals = list(st)
            vals[stat.ST_UID] = 4242
            return os.stat_result(vals)
        return st
    monkeypatch.setattr(Path, "stat", fake_stat)
    s = site.load(_env(tmp_path, MACE_SITE_GROUP_CONFIG=str(g)))
    assert s.account == "grp_pi" and s.group_owners[str(g)] == site._owner(4242)
    # the same file from an implicit location ($XDG_CONFIG_DIRS) is refused
    xdg = tmp_path / "xdgdirs"
    (xdg / "mace").mkdir(parents=True)
    implicit = xdg / "mace" / "site.toml"
    implicit.write_text('[slurm]\naccount = "grp_pi"\n')
    implicit.chmod(0o644)
    real2 = Path.stat

    def fake_stat2(self, *a, **k):
        st = real2(self, *a, **k)
        if self == implicit:
            vals = list(st)
            vals[stat.ST_UID] = 4242
            return os.stat_result(vals)
        return st
    monkeypatch.setattr(Path, "stat", fake_stat2)
    env = _env(tmp_path, XDG_CONFIG_DIRS=str(xdg))
    env.pop("MACE_SITE_GROUP_CONFIG")
    with pytest.raises(site.SiteConfigError, match="must belong to root or to you"):
        site.load(env)


def test_group_writable_user_file_only_warns(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "grp_mine"\n')
    f = tmp_path / "xdg" / "mace" / "site.toml"
    f.chmod(0o664)
    env = _env(tmp_path)
    s = site.load(env)
    assert s.account == "grp_mine"
    assert any("writable by group or others (mode 0664)" in w for w in s.warnings)
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 0 and "Warning:" in r.stdout and "0664" in r.stdout


def test_missing_named_group_file_warns(tmp_path):
    s = site.load(_env(tmp_path, MACE_SITE_GROUP_CONFIG=str(tmp_path / "nope.toml")))
    assert any("does not exist" in w for w in s.warnings) and s.account == "mendoza_q"


# ---------------------------------------------------------------- D7: no tomllib
def _no_tomllib(tmp, env, cwd, *args):
    code = ("import sys, runpy; sys.modules['tomllib'] = None; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            f"sys.argv = ['mace_cli', '--no-banner', *{list(args)!r}]; "
            f"runpy.run_path({str(REPO_ROOT / 'mace_cli')!r}, run_name='__main__')")
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_unreadable_settings_fail_closed(tmp_path):
    env = _env(tmp_path)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\n')
    p = _no_tomllib(tmp_path, env, d, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "needs Python 3.11 or newer to read site settings" in p.stdout
    assert str(tmp_path / "xdg" / "mace" / "site.toml") in p.stdout
    assert not (d / "m.sh").exists()
    # no settings file: today's bytes
    (tmp_path / "xdg" / "mace" / "site.toml").unlink()
    p = _no_tomllib(tmp_path, env, d, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m.sh").read_bytes() == _template_output(tmp_path, SCF, "m")
