"""The partition's MaxTime caps a project folder's walltime (mace-folder.toml
[submit]) and a size-aware suggestion on every path, whatever MaxTime is.

Before, the resolved site kept the partition's MaxTime only when it was
under the templates' 7 days, and the `-i` defaults, the --exclude-* scripts,
the workflow planner and the quick-start overlaid the walltime on a site
that was never resolved. A folder walltime of 5 days on a 2-day partition,
or of 14 days on a 7- or 10-day one, then reached the job script and SLURM
refused the job or left it pending (PartitionTimeLimit).

Every test drives a fake `scontrol` (tests/fake_slurm.py) with MaxTime below
and at or above 7 days; none sets the cap by hand. MSU HPCC's job scripts are
pinned by the goldens (test_site_g_goldens.py, test_site_generic_cluster.py).
"""
import os
import re

import pytest

from fake_slurm import FakeSlurm
from test_folder_settings import _write
from test_site_generic_cluster import PARTITIONS, _cluster, _fresh  # noqa: F401 (autouse fixture)
from test_site_jobscript import _deck, _mace, _toml
from test_array_default_path_golden import BAND_D3, SP_D12

from mace.submission import site
from mace.utils import folder_settings as F

# (the default partition's MaxTime, the folder's walltime, what the job asks for)
CASES = [("2-00:00:00", "5-00:00:00", "2-00:00:00"),
         ("7-00:00:00", "14-00:00:00", "7-00:00:00"),
         ("10-00:00:00", "14-00:00:00", "10-00:00:00")]


def slurm_maxtime(tmp, env, maxtime):
    """``env`` with a fake ``scontrol`` whose default partition allows ``maxtime``
    (for tests that resolve a site in the process: site.resolve_limit)."""
    fake = FakeSlurm(tmp / ("fs-maxtime-" + maxtime.replace(":", "")), {"partitions": f"PartitionName=p MaxTime={maxtime} State=UP Default=YES\n"})
    site._SLURM_MEMO.clear()
    return dict(env, PATH=f"{fake.bindir}:{env.get('PATH', os.environ.get('PATH', ''))}",
                FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"], FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])


def _setup(tmp_path, maxtime, folder_text, site_toml=None):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(partitions=PARTITIONS.replace("MaxTime=2-00:00:00", f"MaxTime={maxtime}"))
    if site_toml:
        _toml(tmp_path, site_toml)
    proj = tmp_path / "home" / "proj"
    _write(proj, folder_text)
    return env, proj


def _t(text):
    return re.search(r"(?m)^#SBATCH -t (\S+)", text).group(1)


def _submit(proj, env, deck, *args, stdin=""):
    _deck(proj, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (proj / deck.replace(".d3", ".f9")).write_text("fort9\n")
    sh = proj / (deck.rsplit(".", 1)[0] + ".sh")
    sh.unlink(missing_ok=True)
    p = _mace(proj, env, "submit", "--nosubmit", *args, deck, stdin=stdin)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return _t(sh.read_text()), p.stdout + p.stderr


@pytest.mark.parametrize("maxtime,folder,want", CASES)
@pytest.mark.parametrize("args,stdin", [
    ([], ""),                                   # the plain generator path
    (["-i"], "n\n"),                            # -i, the defaults kept
    (["-i"], "y\n\n\n\n\nn\n"),                 # -i, every prompt left blank
    (["--exclude-nodes", "node9"], ""),         # the customized generator without -i
])
def test_submit_caps_the_folder_walltime(tmp_path, maxtime, folder, want, args, stdin):
    env, proj = _setup(tmp_path, maxtime, f'[submit]\nwalltime = "{folder}"\n')
    t, out = _submit(proj, env, "m.d12", *args, stdin=stdin)
    assert t == want, out


@pytest.mark.parametrize("maxtime,folder,want", CASES)
def test_properties_jobs_are_capped_too(tmp_path, maxtime, folder, want):
    env, proj = _setup(tmp_path, maxtime, f'[submit.properties]\nwalltime = "{folder}"\n')
    t, out = _submit(proj, env, "m_band.d3")
    assert t == want, out
    t, out = _submit(proj, env, "m_band.d3", "-i", stdin="n\n")
    assert t == want, out


def _in_process(monkeypatch, env, proj):
    for k in list(os.environ):
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(F.ENV, str(proj / F.FILENAME))
    monkeypatch.chdir(proj)


@pytest.mark.parametrize("maxtime,folder,want", CASES)
def test_every_overlay_path_is_capped(tmp_path, monkeypatch, maxtime, folder, want):
    env, proj = _setup(tmp_path, maxtime, f'[submit]\nwalltime = "{folder}"\n')
    _in_process(monkeypatch, env, proj)
    assert site.default_walltime("scf", "7-00:00:00", out=lambda t: None) == want
    from mace.workflow import planner
    assert planner._folder_walltime("scf") == want
    assert planner._site_overlay({"walltime": "7-00:00:00"}, "scf", "OPT")["walltime"] == want
    from mace.submission import crystal, properties
    assert crystal.get_default_d12_resources("OPT")["walltime"] == want
    assert properties.get_default_d3_resources("BAND")["walltime"] == want
    from mace import run_mace
    assert run_mace._get_quick_slurm_config("OPT", 1)["scripts"]["submitcrystal23.sh"]["resources"]["walltime"] == want


@pytest.mark.parametrize("maxtime", ["2-00:00:00", "7-00:00:00"])
def test_a_suggested_walltime_is_capped(tmp_path, monkeypatch, maxtime):
    env, proj = _setup(tmp_path, maxtime, "[plots]\n")
    _in_process(monkeypatch, env, proj)
    s = site.resolve(site.load(), out=lambda t: None)
    assert s.walltime("scf", "OPT", {"walltime": "30-00:00:00"}) == maxtime
    assert site.overlay_resources({}, s, "scf", "OPT", {"walltime": "30-00:00:00"})["walltime"] == maxtime


def test_a_site_toml_walltime_is_written_as_set(tmp_path, monkeypatch):
    """Only the folder's and the suggestion's walltime are capped: the user's
    own site.toml value is written as set (mace site warns about it)."""
    env, proj = _setup(tmp_path, "7-00:00:00", "[plots]\n", '[resources]\nwalltime = "10-00:00:00"\n')
    _in_process(monkeypatch, env, proj)
    s = site.resolve(site.load(), out=lambda t: None)
    assert s.walltime("scf") == "10-00:00:00"


@pytest.mark.parametrize("maxtime", ["2-00:00:00", "7-00:00:00", "10-00:00:00"])
def test_mace_site_warns_about_a_walltime_above_maxtime(tmp_path, maxtime):
    env, proj = _setup(tmp_path, maxtime, "[plots]\n", '[resources]\nwalltime = "12-00:00:00"\n')
    p = _mace(proj, env, "site")
    out = p.stdout + p.stderr
    assert f'walltime = "12-00:00:00" is above the {maxtime}' in out, out


def test_mace_site_is_quiet_within_maxtime(tmp_path):
    env, proj = _setup(tmp_path, "14-00:00:00", "[plots]\n", '[resources]\nwalltime = "12-00:00:00"\n')
    p = _mace(proj, env, "site")
    assert "is above the" not in p.stdout + p.stderr


def test_mace_site_says_a_folder_walltime_is_capped(tmp_path):
    env, proj = _setup(tmp_path, "7-00:00:00", '[submit]\nwalltime = "14-00:00:00"\n')
    p = _mace(proj, env, "site")
    out = p.stdout + p.stderr
    assert "14-00:00:00" in out and "7-00:00:00" in out and "SLURM will refuse" not in out, out
