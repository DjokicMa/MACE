"""A project folder's walltime (mace-folder.toml [submit]) among the
[resources] walltime levels of the site settings (1.1.6).

The folder's walltime is a [resources] level of its own: it wins over
site.toml's [resources] and [resources.properties] walltime, a
[resources.<TYPE>] table and MACE_SLURM_WALLTIME still win over it, and it is
never above the partition's MaxTime (as folder_walltime caps it), while a
walltime someone set in site.toml is written as set.
"""
from test_folder_settings import _proj
from test_site_jobscript import _env, _toml

from mace.submission import site
from mace.utils import folder_settings as F

SITE = '[resources]\nwalltime = "1-00:00:00"\n[resources.FREQ]\nwalltime = "3-00:00:00"\n'


def _load(tmp_path, folder, maxtime=None, **extra):
    env = _env(tmp_path, **extra)
    _toml(tmp_path, SITE)
    d = _proj(tmp_path, folder)
    env = dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME))
    if maxtime is None:
        return site.load(env)
    from test_walltime_partition_maxtime import slurm_maxtime
    return site.load_resolved(slurm_maxtime(tmp_path, env, maxtime))


def test_the_folder_walltime_wins_over_the_site_level(tmp_path):
    s = _load(tmp_path, '[submit]\nwalltime = "5-00:00:00"\n')
    assert s.walltime("scf") == s.walltime("scf", "SP") == "5-00:00:00"
    assert s.walltime("prop", "BAND") == "5-00:00:00"          # the plain [submit] walltime covers properties jobs
    assert s.walltime("scf", "FREQ") == "3-00:00:00"           # a [resources.<TYPE>] table still wins
    assert site.overlay_resources({"walltime": "7-00:00:00"}, s, "scf", "OPT")["walltime"] == "5-00:00:00"


def test_its_own_properties_walltime(tmp_path):
    s = _load(tmp_path, '[submit]\nwalltime = "5-00:00:00"\n[submit.properties]\nwalltime = "3:00:00"\n')
    assert (s.walltime("scf"), s.walltime("prop")) == ("5-00:00:00", "0-03:00:00")


def test_the_partition_caps_only_the_folder_walltime(tmp_path):
    s = _load(tmp_path, '[submit]\nwalltime = "5-00:00:00"\n', maxtime="2-00:00:00")
    assert s.walltime("scf", "SP") == "2-00:00:00"
    assert s.walltime("scf", "FREQ") == "3-00:00:00"           # site.toml's own value: written as set


def test_the_environment_wins(tmp_path):
    s = _load(tmp_path, '[submit]\nwalltime = "5-00:00:00"\n', maxtime="1:00:00", MACE_SLURM_WALLTIME="12:00:00")
    assert s.walltime("scf") == s.walltime("scf", "FREQ") == s.walltime("prop") == "12:00:00"


def test_no_folder_file_keeps_the_site_levels(tmp_path):
    s = _load(tmp_path, '[viewer]\nshow_cell = false\n')
    assert (s.walltime("scf"), s.walltime("scf", "FREQ"), s.walltime("prop")) == ("1-00:00:00", "3-00:00:00", "1-00:00:00")


def test_the_plain_folder_walltime_leaves_a_site_properties_walltime_alone(tmp_path):
    """As with ntasks: the folder's plain [submit] values are the [resources]
    level; site.toml's [resources.properties] still wins for properties jobs
    (only the folder's own [submit.properties] beats it)."""
    env = _env(tmp_path)
    _toml(tmp_path, '[resources]\nwalltime = "1-00:00:00"\n[resources.properties]\nwalltime = "3:00:00"\nntasks = 4\n')
    d = _proj(tmp_path, '[submit]\nwalltime = "2-00:00:00"\nntasks = 48\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.walltime("scf", "OPT") == "2-00:00:00"
    assert s.walltime("prop", "BAND") == "3:00:00"
    assert s.effective_resources("prop", "BAND")["ntasks"] == 4
    assert site.folder_walltime(s, "prop") is None
    d = _proj(tmp_path, '[submit]\nwalltime = "2-00:00:00"\n[submit.properties]\nwalltime = "5:00:00"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.walltime("prop", "BAND") == "0-05:00:00"


def test_folder_help_texts_say_what_each_walltime_covers():
    assert "properties" in F.KEYS["submit.walltime"]["help"]
