"""Site settings are validated before anything is generated (PLAN_submit D30):
site.toml values and the MACE_SLURM_ACCOUNT / MACE_SLURM_PARTITION /
MACE_CRYSTAL_MODULE variables end up inside single-quoted ``echo '...'``
lines of the job-script generators, so quotes, ``$``, spaces and newlines
are refused with exit code 2 and no job script is written."""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from test_array_default_path_golden import SP_D12

from mace.submission import site

MACE_CLI = REPO_ROOT / "mace_cli"


def _env(tmp, **extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MACE_SLURM_", "MACE_CRYSTAL", "MACE_SITE", "MODULE", "LMOD"))
           and k not in ("MACE_HOME", "PYTHONPATH", "BASH_ENV", "SLURM_JOB_ID")}
    env.update(XDG_CONFIG_HOME=str(tmp / "xdg"), XDG_CACHE_HOME=str(tmp / "cache"), MACE_MODULES_NO_PROFILE="1",
               MACE_NO_BANNER="1", **extra)
    return env


def _toml(tmp, text):
    d = tmp / "xdg" / "mace"
    d.mkdir(parents=True, exist_ok=True)
    (d / "site.toml").write_text(text)
    return d / "site.toml"


@pytest.mark.parametrize("text,needle", [
    ('[slurm]\naccount = "a b"\n', "account"),
    ("[slurm]\naccount = \"x'y\"\n", "account"),
    ('[slurm]\naccount = "$(id)"\n', "account"),
    ('[slurm]\npartition = "a;b"\n', "partition"),
    ('[slurm]\npartition = "a,,b"\n', "partition"),
    ('[modules]\ncrystal = "CRYSTAL/23 x"\n', "module name"),
    ("[modules]\ncrystal = \"CRYSTAL/23'x\"\n", "module name"),
    ('[modules]\ncrystal = ["CRYSTAL/23-a", "foss/2023a"]\n', "last entry must start with CRYSTAL/"),
    ('[modules]\ncrystal_minimum = "24"\ncrystal_maximum = "23"\n', "crystal_minimum"),
    ('[local_run]\ngateway_message = "go {away}"\n', "placeholders"),
    ("[local_run]\nmodule_message = \"it's {module}\"\n", "single quote"),
])
def test_bad_values_are_refused(tmp_path, text, needle):
    path = _toml(tmp_path, text)
    with pytest.raises(site.SiteConfigError) as err:
        site.load(_env(tmp_path))
    assert needle in str(err.value) and str(path) in str(err.value)


def test_good_values_and_partition_list(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "grp_example"\npartition = "a,b"\n'
                    '[modules]\ncrystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]\npython = []\n')
    s = site.load(_env(tmp_path))
    assert (s.account, s.partition, s.crystal, s.python) == ("grp_example", "a,b",
                                                             ["foss/2023a", "CRYSTAL/23-foss-2023a"], [])
    assert s.window == (23, 23) and s.sources["account"] == "site.toml"


def test_malformed_toml_names_the_line(tmp_path):
    path = _toml(tmp_path, '[slurm]\naccount = "x"\n\naccount = = 3\n')
    with pytest.raises(site.SiteConfigError) as err:
        site.load(_env(tmp_path))
    assert f"{path}:4:" in str(err.value)


def test_unknown_keys_only_warn(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "grp_a"\nqueue = "x"\n[other]\nk = 1\n')
    s = site.load(_env(tmp_path))
    assert s.account == "grp_a" and "slurm.queue" in s.warnings[0] and "other" in s.warnings[0]


def test_mace_clusters_tables_are_not_unknown(tmp_path):
    """One site.toml for both: `mace cluster`'s [cluster], [quota], [scan] and
    [devnodes] tables give the job-script settings no warning, and each
    reader keeps its own values."""
    _toml(tmp_path, '[slurm]\naccount = "acct1"\n\n[quota]\nwarn_percent = 75\n\n[cluster]\nnode_query = "auto"\n')
    env = _env(tmp_path)
    s = site.load(env)
    assert s.account == "acct1"
    assert not any("unknown" in w for w in s.warnings), s.warnings
    from mace.cluster import config
    cfg = config.load(env)
    assert cfg.values["quota"]["warn_percent"] == 75 and not cfg.errors and not cfg.warnings, (cfg.errors, cfg.warnings)


def test_env_wins_over_toml(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "grp_a"\n')
    s = site.load(_env(tmp_path, MACE_SLURM_ACCOUNT="grp_b"))
    assert s.account == "grp_b" and s.sources["account"] == "env MACE_SLURM_ACCOUNT"


def test_site_config_none_ignores_the_file(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "a b"\n')
    assert site.load(_env(tmp_path, MACE_SITE_CONFIG="none")).account == site.DEFAULTS["account"]


@pytest.mark.parametrize("var,value", [("MACE_SLURM_ACCOUNT", "x'y"), ("MACE_SLURM_PARTITION", "a;b"),
                                       ("MACE_CRYSTAL_MODULE", "CRYSTAL/23 x"),
                                       ("MACE_SLURM_ACCOUNT", "a\nb")])
def test_bad_env_stops_submit_before_any_script(tmp_path, var, value):
    d = tmp_path / "decks"
    d.mkdir()
    (d / "m.d12").write_text(SP_D12)
    before = sorted(p.name for p in d.iterdir())
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "--nosubmit", "m.d12"], cwd=d,
                       env=_env(tmp_path, **{var: value}), capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stdout + p.stderr
    assert var in p.stdout + p.stderr and repr(value) in p.stdout + p.stderr
    assert sorted(x.name for x in d.iterdir()) == before          # no .sh
    # nothing was written into the install either (the -i/walltime temporary generator)
    assert not list((REPO_ROOT / "mace" / "submission").glob("*_custom_*.sh"))


def test_bad_toml_stops_submit_with_walltime(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "a b"\n')
    d = tmp_path / "decks"
    d.mkdir()
    (d / "m.d12").write_text(SP_D12)
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "--nosubmit", "--walltime", "1:00:00",
                        "m.d12"], cwd=d, env=_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and not (d / "m.sh").exists()
    assert not list((REPO_ROOT / "mace" / "submission").glob("*_custom_*.sh"))


def test_bad_env_stops_mace_run(tmp_path):
    d = tmp_path / "decks"
    d.mkdir()
    (d / "m.d12").write_text(SP_D12)
    p = subprocess.run([sys.executable, str(MACE_CLI), "run", "--dry-run", "m.d12"], cwd=d,
                       env=_env(tmp_path, MACE_SLURM_PARTITION="a;b"), capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "MACE_SLURM_PARTITION" in p.stderr
    assert not (d / ".mace_local").exists()
