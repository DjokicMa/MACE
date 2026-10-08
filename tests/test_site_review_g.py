"""Review of the 1.1.5 site settings (Area G): values that slipped past the
checks, group files reached another way, and messages that named a removed
folder."""
import os
import subprocess
import sys

import pytest

from mace.submission import site, site_check
from test_site_jobscript import MACE_CLI, _env


def _load(tmp_path, text):
    p = tmp_path / "site.toml"
    p.write_text(text)
    p.chmod(0o644)
    return site.load({"MACE_SITE_CONFIG": str(p), "MACE_SITE_GROUP_CONFIG": "none", "HOME": str(tmp_path),
                      "USER": "tester"})


# ---------------------------------------------------------------- [paths] scratch
@pytest.mark.parametrize("value", ["/scratch/$USERNAME", "/s/$SLURM_JOB_IDS", "/s/$USER_x", "$HOMEDIR/x",
                                   "/s/$HOME2"])
def test_scratch_names_longer_than_a_token_are_refused(tmp_path, value):
    # bash reads "$USERNAME" as the variable USERNAME, not $USER then NAME
    with pytest.raises(site.SiteConfigError, match="scratch is not a valid directory"):
        _load(tmp_path, f'[paths]\nscratch = "{value}"\n')


@pytest.mark.parametrize("value", ["/scratch/$USER", "/s/${USER}x", "$HOME/scr", "${HOME}/a/$SLURM_JOB_ID",
                                   "/s/$USER/b-$SLURM_JOB_ID.x", "/s/$USER.tmp"])
def test_scratch_tokens_still_accepted(tmp_path, value):
    assert _load(tmp_path, f'[paths]\nscratch = "{value}"\n').scratch == value


def test_scratch_env_variable_is_checked_the_same_way(tmp_path):
    with pytest.raises(site.SiteConfigError):
        site.load({"MACE_SITE_CONFIG": "none", "MACE_SCRATCH_BASE": "/scratch/$USERNAME"})


# ---------------------------------------------------------------- [slurm] extra
@pytest.mark.parametrize("item,hit", [("--tim=1:00:00", "--time"), ("--part=x", "--partition"),
                                      ("--acc=x", "--account"), ("--qo=x", "--qos"), ("--job=x", "--job-name"),
                                      ("--arr=1-3", "--array"), ("--cons=a", "--constraint")])
def test_extra_refuses_abbreviations_of_the_lines_mace_writes(tmp_path, item, hit):
    # sbatch takes any unambiguous prefix of a long option
    with pytest.raises(site.SiteConfigError, match=f"MACE writes {hit} itself"):
        _load(tmp_path, f'[slurm]\nextra = ["{item}"]\n')


@pytest.mark.parametrize("item", ["--exclusive", "--mem-bind=local", "--tmp=10G", "--mail-type=END",
                                  "--acctg-freq=30", "--open-mode=append", "--export=NONE", "--gpus-per-node=2",
                                  "--nodelist=c[001-004]", "--parsable", "--ntasks-per-node=4"])
def test_extra_keeps_other_options(tmp_path, item):
    assert _load(tmp_path, f'[slurm]\nextra = ["{item}"]\n').extra == [item]


# ---------------------------------------------------------------- [mpi] env
@pytest.mark.parametrize("name", ["CDPATH", "GLOBIGNORE", "BASHOPTS", "POSIXLY_CORRECT"])
def test_env_names_that_change_how_bash_runs_are_refused(tmp_path, name):
    with pytest.raises(site.SiteConfigError, match=f"{name} cannot be set from site settings"):
        _load(tmp_path, f'[mpi]\nenv = ["{name}=/tmp"]\n')


def test_srun_mpi_without_srun_is_a_warning(tmp_path):
    s = _load(tmp_path, '[mpi]\nsrun_mpi = "pmix"\n')
    assert s.launcher == "mpirun" and any("srun_mpi" in w and "ignored" in w for w in s.warnings)
    assert not _load(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix"\n').warnings


# ---------------------------------------------------------------- group files
def _group(tmp_path, folder_mode=0o755):
    g = tmp_path / "lab"
    g.mkdir()
    (g / "site.toml").write_text('[slurm]\naccount = "lab"\n[cluster]\nname = "c"\n')
    (g / "site.toml").chmod(0o644)
    g.chmod(folder_mode)
    return g / "site.toml"


def test_load_tables_refuses_a_group_file_others_can_change(tmp_path):
    f = _group(tmp_path, 0o775)
    env = {"MACE_SITE_GROUP_CONFIG": str(f), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    with pytest.raises(site.SiteConfigError, match="writable by group or others"):
        site.load(env)
    with pytest.raises(site.SiteConfigError, match="writable by group or others"):
        site.load_tables(env)


def test_load_tables_reads_a_safe_group_file(tmp_path):
    f = _group(tmp_path)
    data, _, _ = site.load_tables({"MACE_SITE_GROUP_CONFIG": str(f), "XDG_CONFIG_HOME": str(tmp_path / "xdg")})
    assert data["cluster"]["name"] == "c"


def test_a_symlinked_group_file_is_checked_where_it_points(tmp_path):
    target = _group(tmp_path, 0o777)                  # anyone can swap the file the link names
    safe = tmp_path / "etc"
    safe.mkdir()
    (safe / "site.toml").symlink_to(target)
    safe.chmod(0o755)
    env = {"MACE_SITE_GROUP_CONFIG": str(safe / "site.toml"), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    with pytest.raises(site.SiteConfigError, match=f"its directory {tmp_path / 'lab'} is writable"):
        site.load(env)
    target.parent.chmod(0o755)
    assert site.load(env).account == "lab"


# ---------------------------------------------------------------- mace site check
def test_check_without_slurm_names_no_removed_folder(tmp_path):
    env = _env(tmp_path, PATH="/usr/bin:/bin", TMPDIR=str(tmp_path))
    rep = site_check.run(env, kinds=("scf",))
    job = next(i for i in rep["items"] if i["item"] == "Job script")
    assert job["status"] == "ok" and "bash -n passes" in job["detail"]
    assert str(tmp_path) not in job["detail"] and rep["kept"] is None
    assert not list(tmp_path.glob("mace-site-check-*"))


# ---------------------------------------------------------------- help text
def test_curated_run_help_states_the_cap_of_any_cluster(tmp_path):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "run", "--help"], env=_env(tmp_path),
                       capture_output=True, text=True, timeout=120)
    flat = " ".join(r.stdout.split())
    assert "ulimit -t" in flat and "1h55m on MSU HPCC" in flat and "[local_run] max_time" in flat


def test_scratch_suggestion_replaces_only_the_user_segment(tmp_path):
    (tmp_path / "scratch" / "at").mkdir(parents=True)
    env = {"USER": "at", "HOME": str(tmp_path / "home"), "MACE_SCRATCH_PROBE_ROOT": str(tmp_path)}
    got = site_check.scratch_suggestion(env)
    assert got == f"{tmp_path}/scratch/$USER"
    assert site._valid_scratch(got, "test") == got


def test_fit_counts_every_cpu_of_a_task_for_mem_per_cpu(monkeypatch):
    s = site.Site()
    s.partition = "p"
    s.res_raw[""] = {"ntasks": 16, "cpus_per_task": 4, "mem_per_cpu": "4G"}
    shape = {"cores": 64, "max_cores": 64, "threads": 1, "mem_mb": 200 * 1024, "max_mem_mb": 200 * 1024,
             "nodes": 4, "features": [], "gres": []}
    monkeypatch.setattr(site, "node_shapes", lambda part, env=None, refresh=False: {"p": shape})
    f = site.fit(s, "scf", {})
    # 64 CPUs x 4G = 256G: more than the node's 200G
    assert f["request"] == {"cores_per_node": 64, "mem_mb_per_node": 64 * 4096} and not f["ok"]
    assert f["suggest"]["ntasks"] == 16                  # a node of 64 cores holds 16 tasks of 4 CPUs


def test_curated_run_help_interpolates_the_parser_timeout_help(tmp_path):
    """`mace run --help` (curated page) and the parser's --timeout help are one text."""
    from mace.submission.local_run import TIMEOUT_HELP
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "run", "--help"], env=_env(tmp_path),
                       capture_output=True, text=True, timeout=120)
    flat = " ".join(r.stdout.split())
    want = " ".join(TIMEOUT_HELP.split())
    assert r.returncode == 0 and "{RUN_TIMEOUT_HELP}" not in r.stdout
    assert (want[:1].upper() + want[1:]) in flat
