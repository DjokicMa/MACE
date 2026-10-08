"""`mace site` (PLAN_submit D33) and the invalid-account hint (D35), through
the real mace_cli with the fake SLURM on PATH."""
import json
import os
import subprocess
import sys

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import SP_D12

MACE_CLI = REPO_ROOT / "mace_cli"


def _env(tmp, fake=None, **extra):
    src = fake.env if fake else os.environ
    env = {k: v for k, v in src.items()
           if not k.startswith(("MACE_SLURM_", "MACE_CRYSTAL", "MACE_SITE", "MODULE", "LMOD"))
           and k not in ("MACE_HOME", "PYTHONPATH", "BASH_ENV", "SLURM_JOB_ID")}
    if fake is None:       # no SLURM tools at all
        env["PATH"] = "/usr/bin:/bin"
    (tmp / "home").mkdir(parents=True, exist_ok=True)
    env.update({"XDG_CONFIG_HOME": str(tmp / "xdg"), "XDG_CACHE_HOME": str(tmp / "cache"), "HOME": str(tmp / "home"),
                "MACE_MODULES_NO_PROFILE": "1", "USER": "tester", "MACE_SITE_GROUP_CONFIG": "none", **extra})
    return env


def _site(tmp, env, *args):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "site", *args], cwd=tmp, env=env,
                          capture_output=True, text=True, timeout=120)


def test_show_json_and_sources(tmp_path):
    env = _env(tmp_path, MACE_SLURM_ACCOUNT="grp_env")
    r = _site(tmp_path, env)
    assert r.returncode == 0, r.stderr
    assert "Account:     grp_env  (env MACE_SLURM_ACCOUNT)" in r.stdout
    assert "CRYSTAL/23-intel-2023a" in r.stdout and "not checked: no module system here" in r.stdout
    assert "account not checked" in r.stdout
    d = json.loads(_site(tmp_path, env, "--json").stdout)
    assert {k: d["account"][k] for k in ("value", "source")} == {"value": "grp_env", "source": "env MACE_SLURM_ACCOUNT"}
    assert d["crystal"]["module"] == "CRYSTAL/23-intel-2023a" and d["crystal"]["source"] == "unchecked"
    assert _site(tmp_path, env, "--refresh").returncode == 0


def test_init_writes_once_and_changes_nothing(tmp_path):
    env = _env(tmp_path)
    r = _site(tmp_path, env, "--init")
    path = tmp_path / "xdg" / "mace" / "site.toml"
    assert r.returncode == 0 and path.is_file() and oct(path.stat().st_mode & 0o777) == "0o644"
    import tomllib
    data = tomllib.loads(path.read_text())
    assert set(data) == {"slurm", "resources", "mpi", "modules", "crystal", "paths", "queue", "local_run"}
    assert data["resources"] == {"properties": {}}
    assert all(v == {} for k, v in data.items() if k != "resources")      # no active key
    d = json.loads(_site(tmp_path, env, "--json").stdout)
    assert d["config"]["exists"] and all(d[k]["source"] == "built-in" for k in ("account", "partition", "python"))
    r2 = _site(tmp_path, env, "--init")
    assert r2.returncode == 1 and "already exists" in r2.stdout


def test_account_check(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"assoc": ["grp_a", "grp_b"]})
    r = _site(tmp_path, _env(tmp_path, fake))
    # not a member of the built-in account: it is left out, SLURM uses the default one
    assert "Account:     (your SLURM default account)" in r.stdout
    assert "Account check: no #SBATCH -A line; SLURM charges your default account (yours: grp_a, grp_b)" in r.stdout
    (tmp_path / "xdg" / "mace").mkdir(parents=True)
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[slurm]\naccount = "grp_c"\n')
    r = _site(tmp_path, _env(tmp_path, fake))
    assert ("Account grp_c is not one of your SLURM accounts (yours: grp_a, grp_b). Set [slurm] account in "
            f"{tmp_path / 'xdg' / 'mace' / 'site.toml'} or MACE_SLURM_ACCOUNT.") in r.stdout
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[slurm]\naccount = "grp_a"\n')
    r = _site(tmp_path, _env(tmp_path, fake))
    assert "grp_a is one of your SLURM accounts" in r.stdout and "not one of" not in r.stdout
    assert any(c["argv"][:3] == ["sacctmgr", "-nP", "show"] for c in fake.calls("sacctmgr"))


def test_invalid_config_exits_2(tmp_path):
    (tmp_path / "xdg" / "mace").mkdir(parents=True)
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[slurm]\naccount = "a b"\n')
    r = _site(tmp_path, _env(tmp_path))
    assert r.returncode == 2 and "Error:" in r.stdout and "account" in r.stdout


def test_listed_in_help(tmp_path):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--help"], env=_env(tmp_path), capture_output=True, text=True)
    assert "  site        Show the site settings for job scripts" in r.stdout
    r = _site(tmp_path, _env(tmp_path), "--help")
    assert r.returncode == 0 and "usage: mace site" in r.stdout and "--init" in r.stdout


def test_invalid_account_hint_once(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"reject_account": True})
    d = tmp_path / "decks"
    d.mkdir()
    (d / "m.d12").write_text(SP_D12)
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "m.d12"], cwd=d,
                       env=_env(tmp_path, fake), capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert "Invalid account or account/partition combination" in out
    assert out.count("Set your cluster account in") == 1 and "`mace site` shows what MACE uses." in out
    # a submit that works prints no hint
    fake2 = FakeSlurm(tmp_path / "fs2")
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "m.d12"], cwd=d,
                       env=_env(tmp_path, fake2), capture_output=True, text=True, timeout=120)
    assert "Set your cluster account" not in r.stdout + r.stderr


def test_docs_match_help(tmp_path):
    """docs/cli.md's Site settings section names every option of `mace site
    --help`, every environment variable and every site.toml key."""
    import re
    doc = (REPO_ROOT / "docs" / "cli.md").read_text()
    sec = doc[doc.index("## Site settings"):]
    help_text = _site(tmp_path, _env(tmp_path), "--help").stdout
    assert "--json" in help_text
    from mace.submission.site_cli import build_parser
    for opt in [o for o in build_parser()._option_string_actions if o.startswith("--")]:
        if opt != "--help":
            assert f"mace site {opt}" in sec or f"mace site check {opt}" in sec, opt
    from mace.submission import site
    for var in ("MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION", "MACE_CRYSTAL_MODULE", "MACE_SITE_CONFIG"):
        assert var in sec, var
    for section, keys in site.KNOWN.items():
        assert f"[{section}]" in sec
        for k in keys:
            assert re.search(rf"^{k}\s*=", sec, re.M), k


def test_invalid_account_hint_for_arrays(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"reject_account": True})
    d = tmp_path / "decks"
    d.mkdir()
    for n in ("a", "b"):
        (d / f"{n}.d12").write_text(SP_D12)
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "--array", "decks"], cwd=tmp_path,
                       env=_env(tmp_path, fake), capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and out.count("Set your cluster account in") == 1, out[-2000:]


def test_init_text_names_every_key():
    """`mace site --init` lists every site.toml key, commented out."""
    import re
    import tomllib
    from mace.submission import site
    text = site.INIT_TEXT
    for section, keys in site.KNOWN.items():
        assert f"[{section}]" in text, section
        for k in keys - {"properties"} if section == "resources" else keys:
            assert re.search(rf"^# {k}\s*=", text, re.M), (section, k)
    assert "[resources.properties]" in text
    data = tomllib.loads(text)
    assert all(not v or v == {"properties": {}} for v in data.values())


def test_docs_example_site_toml_loads(tmp_path):
    """INSTALLATION.md's generic-cluster site.toml parses and passes load()."""
    import re
    import tomllib
    from mace.submission import site
    doc = (REPO_ROOT / "INSTALLATION.md").read_text()
    sec = doc[doc.index("A site.toml for a generic cluster"):]
    block = re.search(r"```toml\n(.*?)```", sec, re.S).group(1)
    s = site.Site()
    site._apply_toml(s, tomllib.loads(block), tmp_path / "site.toml")
    assert s.warnings == [] and s.launcher == "srun" and s.effective_resources("prop")["mem"] == "120G"
    flat = " ".join(doc.split())
    assert "SLURM only" in flat and "MACE 1.2 or later" in flat
    cli = (REPO_ROOT / "docs" / "cli.md").read_text()
    sec = cli[cli.index("## Site settings"):]
    block = re.search(r"```toml\n(.*?)```", sec, re.S).group(1)
    data = tomllib.loads(block)
    data["resources"].pop("mem", None)
    s = site.Site()
    site._apply_toml(s, data, tmp_path / "site.toml")
    assert s.warnings == []
    help_text = _site(tmp_path, _env(tmp_path), "--help").stdout
    assert "SLURM only" in " ".join(help_text.split()) and "MACE 1.2 or later" in " ".join(help_text.split())


def test_describe_blocks_for_the_gui(tmp_path):
    """The GUI's Site row reads describe(): the new blocks are there."""
    from mace.submission import site
    d = site.describe(_env(tmp_path))
    for key in ("resources", "mpi", "scratch", "crystal_install", "slurm_extra", "modules_purge", "local_run"):
        assert key in d, key
    assert d["config"]["group"] == []
    json.dumps(d, default=str)


def test_init_appends_detected_suggestions_as_comments(tmp_path):
    import tomllib
    fake = FakeSlurm(tmp_path / "fs", {"partitions": "PartitionName=main MaxTime=1-00:00:00 State=UP Default=YES\n",
                                       "sinfo": "".join(f"n{i}|idle|2|24|1|48|192000|x|(null)|main*\n"
                                                        for i in range(2))})
    (tmp_path / "scratch" / "tester").mkdir(parents=True)
    env = _env(tmp_path, fake, MACE_SCRATCH_PROBE_ROOT=str(tmp_path))
    r = _site(tmp_path, env, "--init")
    assert r.returncode == 0, r.stdout + r.stderr
    text = (tmp_path / "xdg" / "mace" / "site.toml").read_text()
    assert "# Detected on this cluster" in text and "# [resources] ntasks = 48" in text
    assert f'# scratch = "{tmp_path}/scratch/$USER"' in text
    data = tomllib.loads(text)
    assert all(not v or v == {"properties": {}} for v in data.values())
