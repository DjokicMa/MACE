"""`mace site set KEY VALUE` / `mace site unset KEY`: one key of the user's
site.toml, every comment and other value kept, the value checked before
anything is written, and the value the job scripts now get printed."""
import os
import tomllib

import pytest

from test_site_jobscript import _env, _mace

from mace.submission import site, site_edit

MINE = """# my cluster
[slurm]
account = "lab"   # the PI's account

[resources]
ntasks = 32       # our nodes have 32 cores
mem_per_cpu = "3900M"

[mpi]
launcher = "srun"
"""


def _path(tmp):
    return tmp / "xdg" / "mace" / "site.toml"


def _write(tmp, text):
    p = _path(tmp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _site(tmp, env, *args):
    return _mace(tmp, env, "site", *args)


def test_set_creates_the_file_from_init_and_uncomments_the_example(tmp_path):
    env = _env(tmp_path)
    r = _site(tmp_path, env, "set", "resources.ntasks", "16")
    assert r.returncode == 0, r.stdout + r.stderr
    text = _path(tmp_path).read_text()
    assert text.startswith(site.INIT_TEXT.splitlines()[0])
    assert "Created " in r.stdout and "Set [resources] ntasks = 16 in" in r.stdout
    assert "Now: CRYSTAL jobs --ntasks=16 -N 1 --cpus-per-task=1 --mem-per-cpu=5G -t 7-00:00:00" in r.stdout
    assert "     properties jobs --ntasks=16 " in r.stdout
    # the commented example line is the one set, with its comment kept
    line = next(l for l in text.splitlines() if l.startswith("ntasks"))
    assert line.startswith("ntasks        = 16 ") and "# #SBATCH --ntasks=..." in line
    assert tomllib.loads(text) == {**{k: {} for k in site.KNOWN}, "resources": {"ntasks": 16, "properties": {}}}
    # every other line is as --init wrote it
    init = site.INIT_TEXT.splitlines()
    assert [l for l in text.splitlines() if not l.startswith("ntasks ")][:len(init) - 1] == \
        [l for l in init if not l.startswith("# ntasks        =")]


def test_set_then_unset_gives_the_file_back(tmp_path):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    for key, value in (("resources.FREQ.ntasks", "64"), ("resources.charge+potential.walltime", "4:00:00"),
                       ("slurm.qos", "normal"), ("queue.max_jobs", "100"), ("resources.properties.mem", "120G")):
        r = _site(tmp_path, env, "set", key, value)
        assert r.returncode == 0, (key, r.stdout + r.stderr)
    data = tomllib.loads(p.read_text())
    assert data["resources"]["FREQ"] == {"ntasks": 64}
    assert data["resources"]["CHARGE+POTENTIAL"] == {"walltime": "4:00:00"}
    assert data["slurm"] == {"account": "lab", "qos": "normal"} and data["queue"] == {"max_jobs": 100}
    assert '[resources."CHARGE+POTENTIAL"]' in p.read_text()
    for key in ("resources.properties.mem", "queue.max_jobs", "slurm.qos", "resources.CHARGE+POTENTIAL.walltime",
                "resources.FREQ.ntasks"):
        r = _site(tmp_path, env, "unset", key)
        assert r.returncode == 0, (key, r.stdout + r.stderr)
        assert "Removed " in r.stdout
    assert p.read_text() == MINE


def test_set_replaces_a_value_and_keeps_its_comment(tmp_path):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    r = _site(tmp_path, env, "set", "slurm.account", "other_lab")
    assert r.returncode == 0, r.stdout + r.stderr
    assert p.read_text() == MINE.replace('account = "lab"   #', 'account = "other_lab" #')
    assert "Now: other_lab  (site.toml)" in r.stdout
    # unset of a line with its own comment comments it out (the comment stays)
    r = _site(tmp_path, env, "unset", "resources.ntasks")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "# ntasks = 32       # our nodes have 32 cores" in p.read_text()
    assert "ntasks" not in tomllib.loads(p.read_text())["resources"]
    # ... and a set uncomments it again
    r = _site(tmp_path, env, "set", "resources.ntasks", "24")
    assert "ntasks = 24       # our nodes have 32 cores" in p.read_text()


def test_one_memory_form_per_table(tmp_path):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    r = _site(tmp_path, env, "set", "resources.mem", "250G")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mem_per_cpu removed" in r.stdout
    assert tomllib.loads(p.read_text())["resources"] == {"ntasks": 32, "mem": "250G"}
    assert "--mem=250G" in r.stdout


@pytest.mark.parametrize("args,why", [
    (("set", "resources.cores", "4"), "is not a site setting"),
    (("set", "resources.PHONON.ntasks", "4"), "is not a site setting"),
    (("set", "cluster.name", "x"), "is not a site setting"),
    (("set", "resources.ntasks", "lots"), "whole number"),
    (("set", "resources.ntasks", "0"), "ntasks"),
    (("set", "resources.SP.walltime", "12:00"), "walltime"),
    (("set", "slurm.account", "a b"), "account"),
    (("set", "modules.purge", "maybe"), "true or false"),
    (("set", "resources.ntasks"), "usage"),
])
def test_invalid_input_writes_nothing(tmp_path, args, why):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    r = _site(tmp_path, env, *args)
    assert r.returncode == 2, r.stdout + r.stderr
    assert why in r.stderr, r.stderr
    assert p.read_text() == MINE
    assert not list(p.parent.glob(".site.toml.*"))


def test_unset_of_a_key_that_is_not_set(tmp_path):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    r = _site(tmp_path, env, "unset", "resources.OPT.walltime")
    assert r.returncode == 1 and "is not set" in r.stderr
    assert p.read_text() == MINE
    p.unlink()
    r = _site(tmp_path, env, "unset", "resources.ntasks")
    assert r.returncode == 1 and not p.exists()


def test_a_form_it_cannot_edit_is_left_alone(tmp_path):
    env = _env(tmp_path)
    text = '[resources]\nOPT = { ntasks = 8 }\n'
    p = _write(tmp_path, text)
    r = _site(tmp_path, env, "set", "resources.OPT.ntasks", "16")
    assert r.returncode == 2 and "edit the file by hand" in r.stderr, r.stdout + r.stderr
    assert p.read_text() == text


def test_the_environment_still_wins_and_mace_site_says_so(tmp_path):
    env = _env(tmp_path, MACE_SLURM_NTASKS="8")
    _write(tmp_path, MINE)
    r = _site(tmp_path, env, "set", "resources.SP.ntasks", "12")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Now: SP jobs --ntasks=8 " in r.stdout and "MACE_SLURM_NTASKS=8 is set in this shell" in r.stdout


def test_list_values_and_a_value_that_looks_like_an_option(tmp_path):
    env = _env(tmp_path)
    p = _write(tmp_path, MINE)
    assert _site(tmp_path, env, "set", "slurm.extra", "--exclusive").returncode == 0
    assert _site(tmp_path, env, "set", "modules.python", '["Python/3.12", "SciPy/1"]').returncode == 0
    data = tomllib.loads(p.read_text())
    assert data["slurm"]["extra"] == ["--exclusive"] and data["modules"]["python"] == ["Python/3.12", "SciPy/1"]


def test_no_site_toml_to_edit(tmp_path):
    env = _env(tmp_path, MACE_SITE_CONFIG="none")
    r = _site(tmp_path, env, "set", "resources.ntasks", "16")
    assert r.returncode == 2 and "MACE_SITE_CONFIG=none" in r.stderr


def test_edit_text_keeps_the_comment_column():
    text = '[resources]\n# ntasks = 28        # the cores\n'
    new, _ = site_edit.edit_text(text, ("resources",), "ntasks", 1024)
    assert new == '[resources]\nntasks = 1024      # the cores\n'
    new, _ = site_edit.edit_text('[slurm]\naccount = "a#b"  # x\n', ("slurm",), "account", "c")
    assert new == '[slurm]\naccount = "c"    # x\n'


def test_a_symlinked_site_toml_stays_a_link(tmp_path):
    """A site.toml kept elsewhere (a dotfiles repository) and linked in:
    mace site set writes through the link, never replacing it with a copy."""
    env = _env(tmp_path)
    real = tmp_path / "dotfiles" / "mace-site.toml"
    real.parent.mkdir()
    real.write_text(MINE)
    link = _path(tmp_path)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    r = _site(tmp_path, env, "set", "slurm.qos", "normal")
    assert r.returncode == 0, r.stdout + r.stderr
    assert link.is_symlink() and link.resolve() == real.resolve()
    assert tomllib.loads(real.read_text())["slurm"]["qos"] == "normal"
    assert not list(link.parent.glob(".*.tmp")) and not list(real.parent.glob(".*.tmp"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only folder")
def test_a_read_only_folder_is_named_not_the_temp_file(tmp_path):
    # review fix: the error showed the raw temporary file path
    env = _env(tmp_path)
    p = _write(tmp_path, "[resources]\nntasks = 8\n")
    p.parent.chmod(0o555)
    try:
        r = _site(tmp_path, env, "set", "resources.ntasks", "16")
    finally:
        p.parent.chmod(0o755)
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"cannot write {p}" in r.stderr and "read-only" in r.stderr
    assert ".tmp" not in r.stderr
    assert "ntasks = 8" in p.read_text()
