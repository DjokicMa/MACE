"""Per-calculation-type resources ([resources.OPT], [resources.FREQ], ...) and
the site's default walltime ([resources] walltime), on every path that offers
or writes a job size.

Per key: an answer for one job > the environment > [resources.<TYPE>] >
[resources.properties] (properties types) > [resources] > built-in. With
none of it set, nothing changes (test_site_default_cores.py,
test_site_jobscript.py::test_default_bytes, test_array_default_path_golden.py).
"""
import json

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_default_cores import OPT_D12, _ntasks, _planner_offer, in_env  # noqa: F401 (fixture)
from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site

FREQ_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n", "6 0.125 0.125 0.125\nFREQCALC\nEND\nEND\n", 1)
DOSS_D3 = "NEWK\n4 4\n1 0\nDOSS\n0 100 -1 -1 1 12 0\nEND\n"
TYPED = """[resources]
ntasks = 16
walltime = "24:00:00"
[resources.properties]
ntasks = 8
[resources.FREQ]
ntasks = 64
mem_per_cpu = "7G"
walltime = "3-00:00:00"
[resources.DOSS]
ntasks = 4
walltime = "1:30:00"
"""


def _walltime(text):
    return sorted({l.split("'")[1] if "'" in l else l for l in text.splitlines() if l.startswith("#SBATCH -t ")
                   or l.startswith("echo '#SBATCH -t ")})


def _sbatch(text, flag):
    return sorted({l for l in text.splitlines() if l.startswith(f"#SBATCH {flag}")})


def _load(tmp_path, text, **env):
    path = tmp_path / "site.toml"
    path.write_text(text)
    return site.load({"MACE_SITE_CONFIG": str(path), "MACE_SITE_GROUP_CONFIG": "none", **env})


# ---------------------------------------------------------------- the precedence matrix
@pytest.mark.parametrize("calc,want_ntasks,want_wall", [
    ("OPT", 16, "24:00:00"), ("SP", 16, "24:00:00"), ("FREQ", 64, "3-00:00:00"),
    ("BAND", 8, "24:00:00"), ("DOSS", 4, "1:30:00"), ("TRANSPORT", 8, "24:00:00"),
    ("CHARGE+POTENTIAL", 8, "24:00:00"), ("sp2", 16, "24:00:00"), ("FREQ3", 64, "3-00:00:00")])
def test_type_beats_properties_beats_top_level(tmp_path, calc, want_ntasks, want_wall):
    s = _load(tmp_path, TYPED)
    kind = site.type_kind(site.base_calc_type(calc))
    assert s.effective_resources(kind, calc)["ntasks"] == want_ntasks
    assert s.walltime(kind, calc) == want_wall


def test_kind_levels_ignore_type_tables(tmp_path):
    s = _load(tmp_path, TYPED)
    assert s.effective_resources("scf")["ntasks"] == 16 and s.effective_resources("prop")["ntasks"] == 8
    # a type of the other kind never applies (a BAND table to a Pcrystal job)
    assert s.effective_resources("scf", "DOSS")["ntasks"] == 16
    assert s.typed("scf") == ["FREQ"] and s.typed("prop") == ["DOSS"]
    # memory: the type's form wins, the others keep theirs
    assert s.effective_resources("scf", "FREQ")["mem"] == "7G"
    assert s.effective_resources("prop", "DOSS")["mem_form"] == "mem"


def test_environment_beats_every_table(tmp_path):
    s = _load(tmp_path, TYPED, MACE_SLURM_NTASKS="12", MACE_SLURM_WALLTIME="2-00")
    for calc in site.RESOURCE_TYPES:
        kind = site.type_kind(calc)
        assert s.effective_resources(kind, calc)["ntasks"] == 12, calc
        assert s.walltime(kind, calc) == "2-00", calc
    # a memory form from the environment clears the type's other form too
    s = _load(tmp_path, TYPED, MACE_SLURM_MEM="100G")
    assert s.effective_resources("scf", "FREQ")["mem_form"] == "mem"
    assert s.effective_resources("scf", "FREQ")["mem"] == "100G"
    assert s.effective_resources("scf", "FREQ")["ntasks"] == 64


def test_user_file_beats_group_file_per_key(tmp_path):
    group = tmp_path / "g" / "site.toml"
    group.parent.mkdir()
    group.write_text('[resources.OPT]\nntasks = 48\nwalltime = "4-00:00:00"\n')
    group.chmod(0o644)
    group.parent.chmod(0o755)
    user = tmp_path / "site.toml"
    user.write_text("[resources.OPT]\nntasks = 24\n")
    s = site.load({"MACE_SITE_CONFIG": str(user), "MACE_SITE_GROUP_CONFIG": str(group)})
    assert s.effective_resources("scf", "OPT")["ntasks"] == 24
    assert s.walltime("scf", "OPT") == "4-00:00:00"


@pytest.mark.parametrize("bad", ["12:00", "90", "0:00:00", "1-25", "8-00:61:00", "400-00", "tomorrow", 24])
def test_invalid_walltime_is_refused(tmp_path, bad):
    with pytest.raises(site.SiteConfigError, match="walltime"):
        _load(tmp_path, f"[resources.SP]\nwalltime = {json.dumps(bad)}\n")


def test_invalid_walltime_from_the_environment(tmp_path):
    with pytest.raises(site.SiteConfigError, match="MACE_SLURM_WALLTIME"):
        _load(tmp_path, "", MACE_SLURM_WALLTIME="30")


def test_invalid_type_values_and_unknown_tables(tmp_path):
    with pytest.raises(site.SiteConfigError, match=r"\[resources.FREQ\] ntasks"):
        _load(tmp_path, "[resources.FREQ]\nntasks = 0\n")
    with pytest.raises(site.SiteConfigError, match="mem or mem_per_cpu"):
        _load(tmp_path, '[resources.BAND]\nmem = "1G"\nmem_per_cpu = "1G"\n')
    s = _load(tmp_path, "[resources.PHONON]\nntasks = 4\n[resources.OPT]\ncores = 4\n")
    assert any("resources.OPT.cores" in w and "resources.PHONON" in w for w in s.warnings), s.warnings


# ---------------------------------------------------------------- nothing set: nothing changes
def test_without_settings_mace_site_shows_no_new_lines(tmp_path):
    env = _env(tmp_path)
    r = _mace(tmp_path, env, "site")
    assert "By calculation type" not in r.stdout and "Walltime:" not in r.stdout
    d = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert not any(t["own"] or t["set"] for t in d["resources_by_type"].values())
    assert d["walltime"]["scf"] == {"value": "7-00:00:00", "set": False, "source": "built-in", "origin": None}


def test_without_settings_offers_and_scripts_keep_their_bytes(tmp_path, in_env):
    env = in_env()
    offers = {c: _planner_offer(c) for c in ("OPT", "SP", "FREQ", "BAND", "DOSS")}
    assert {c: o["walltime"] for c, o in offers.items()} == {
        "OPT": "7-00:00:00", "SP": "3-00:00:00", "FREQ": "7-00:00:00", "BAND": "2:00:00", "DOSS": "2:00:00"}
    assert offers["SP"]["memory_per_cpu"] == "4G" and offers["FREQ"]["memory_per_cpu"] == "7G"
    d = tmp_path / "decks"
    _deck(d, "m.d12", FREQ_D12)
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m.sh").read_bytes() == _template_output(tmp_path, SCF, "m")
    _deck(d, "m_doss.d3", DOSS_D3)
    (d / "m_doss.f9").write_text("fort9\n")
    p = _mace(d, env, "submit", "--nosubmit", "m_doss.d3")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m_doss.sh").read_bytes() == _template_output(tmp_path, PROP, "m_doss")


# ---------------------------------------------------------------- consumers
def test_planner_offers_each_types_values_unscaled(tmp_path, in_env):
    _toml(tmp_path, TYPED + '[resources.SP]\nmem_per_cpu = "5G"\n')
    in_env()
    o = {c: _planner_offer(c) for c in ("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT")}
    assert {c: v["ntasks"] for c, v in o.items()} == {"OPT": 16, "SP": 16, "FREQ": 64, "BAND": 8, "DOSS": 4,
                                                      "TRANSPORT": 8}
    assert {c: v["walltime"] for c, v in o.items()} == {
        "OPT": "24:00:00", "SP": "24:00:00", "FREQ": "3-00:00:00", "BAND": "24:00:00", "DOSS": "1:30:00",
        "TRANSPORT": "24:00:00"}
    # a memory someone set is offered as set (the planner's per-type scaling is for built-in values)
    assert o["SP"]["memory_per_cpu"] == "5G" and o["FREQ"]["memory_per_cpu"] == "7G"


def test_submit_folder_of_mixed_decks_gets_each_types_size(tmp_path, in_env):
    _toml(tmp_path, TYPED)
    env = in_env()
    d = tmp_path / "decks"
    _deck(d, "a_opt.d12", OPT_D12)
    _deck(d, "b_sp.d12", SP_D12)
    _deck(d, "c_freq.d12", FREQ_D12)
    p = _mace(tmp_path, env, "submit", "--nosubmit", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    got = {n: (d / f"{n}.sh").read_text() for n in ("a_opt", "b_sp", "c_freq")}
    assert {n: _ntasks(t) for n, t in got.items()} == {
        "a_opt": ["#SBATCH --ntasks=16"], "b_sp": ["#SBATCH --ntasks=16"], "c_freq": ["#SBATCH --ntasks=64"]}
    assert {n: _walltime(t) for n, t in got.items()} == {
        "a_opt": ["#SBATCH -t 24:00:00"], "b_sp": ["#SBATCH -t 24:00:00"], "c_freq": ["#SBATCH -t 3-00:00:00"]}
    assert _sbatch(got["c_freq"], "--mem-per-cpu") == ["#SBATCH --mem-per-cpu=7G"]
    assert _sbatch(got["a_opt"], "--mem-per-cpu") == ["#SBATCH --mem-per-cpu=5G"]
    assert not list(d.glob(".submitcrystal23_site*")), "the rewritten generators are removed"
    # properties decks, a folder of their own
    d = tmp_path / "props"
    _deck(d, "m_band.d3", BAND_D3)
    _deck(d, "m_doss.d3", DOSS_D3)
    for n in ("m_band", "m_doss"):
        (d / f"{n}.f9").write_text("fort9\n")
    p = _mace(tmp_path, env, "submit", "--nosubmit", "props")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    band, doss = (d / "m_band.sh").read_text(), (d / "m_doss.sh").read_text()
    assert _ntasks(band) == ["#SBATCH --ntasks=8"] and _ntasks(doss) == ["#SBATCH --ntasks=4"]
    assert _walltime(band) == ["#SBATCH -t 24:00:00"] and _walltime(doss) == ["#SBATCH -t 1:30:00"]


def test_walltime_option_and_interactive_answers_win(tmp_path, in_env):
    _toml(tmp_path, TYPED)
    env = in_env()
    d = tmp_path / "decks"
    _deck(d, "c_freq.d12", FREQ_D12)
    p = _mace(d, env, "submit", "--nosubmit", "--walltime", "5:00:00", "c_freq.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sh = (d / "c_freq.sh").read_text()
    assert _walltime(sh) == ["#SBATCH -t 5:00:00"] and _ntasks(sh) == ["#SBATCH --ntasks=64"]
    (d / "c_freq.sh").unlink()
    # -i: the FREQ offer is 64 cores and 3 days; typed answers are written
    _deck(d, "a_opt.d12", OPT_D12)
    # answers for each group, in the order they are asked (on a pipe only with the opt-in)
    p = _mace(tmp_path, {**env, "MACE_SUBMIT_PIPED_GROUPS": "1"}, "submit", "--nosubmit", "-i", "decks",
              stdin="y\n20\n\n\n\nn\ny\n40\n\n6:00:00\n\nn\n")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "FREQ decks (1: c_freq.d12); defaults from [resources.FREQ]" in p.stdout
    assert "Cores (ntasks) [16]" in p.stdout and "Cores (ntasks) [64]" in p.stdout
    assert "Walltime [3-00:00:00]" in p.stdout
    opt, freq = (d / "a_opt.sh").read_text(), (d / "c_freq.sh").read_text()
    assert _ntasks(opt) == ["#SBATCH --ntasks=20"] and _walltime(opt) == ["#SBATCH -t 24:00:00"]
    assert _ntasks(freq) == ["#SBATCH --ntasks=40"] and _walltime(freq) == ["#SBATCH -t 6:00:00"]
    assert not list(d.glob("*custom*")) and not list(d.parent.glob("*custom*"))


def test_job_array_gets_each_types_size(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, TYPED)
    d = tmp_path / "decks"
    _deck(d, "a_sp.d12", SP_D12)
    _deck(d, "b_freq.d12", FREQ_D12)
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sorted(tuple(_ntasks(s)) for s in sent) == [("#SBATCH --ntasks=16",), ("#SBATCH --ntasks=64",)], \
        [_ntasks(s) for s in sent]


def test_progress_plan_and_next_steps_get_each_types_values(tmp_path, monkeypatch):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, TYPED + "[resources.SP]\nnodes = 2\n")
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", OPT_D12)
    p = _mace(tmp_path, env, "submit", "--progress", "full_electronic", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    plan = json.loads(next((d / "workflow_configs").glob("workflow_plan_*.json")).read_text())
    res = {step: list(cfg["slurm_config"]["scripts"].values())[0]["resources"]
           for step, cfg in plan["step_configurations"].items()}
    assert {s: r["ntasks"] for s, r in res.items()} == {"OPT_1": 16, "SP_2": 16, "BAND_3": 8, "DOSS_4": 4}
    assert {s: r["walltime"] for s, r in res.items()} == {
        "OPT_1": "24:00:00", "SP_2": "24:00:00", "BAND_3": "24:00:00", "DOSS_4": "1:30:00"}
    assert res["SP_2"]["site_resources"]["scf"] == {"nodes": 2} and "site_resources" not in res["OPT_1"]
    scripts = {c.name: c.read_text() for c in sorted((d / "workflow_scripts").glob("*.sh"))}
    doss = [t for n, t in scripts.items() if "doss" in n]
    assert doss and all("echo '#SBATCH -t 1:30:00' >> $1.sh" in t for t in doss)
    sp = [t for n, t in scripts.items() if "_sp_" in n]
    assert sp and all("echo '#SBATCH -N 2' >> $1.sh" in t for t in sp)
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sent and all(_walltime(s) == ["#SBATCH -t 24:00:00"] for s in sent), [_walltime(s) for s in sent]
    # the next step in a folder without workflow_scripts/: the generator mace submit uses
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    site._SLURM_MEMO.clear()
    from types import SimpleNamespace
    from mace.workflow.engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "materials.db").touch()
    eng.base_work_dir, eng.db = bare, SimpleNamespace(db_path=bare / "materials.db")
    for step, calc, ext, want, wall in ((2, "FREQ", "d12", "64", "3-00:00:00"), (3, "DOSS", "d3", "4", "1:30:00"),
                                        (4, "BAND", "d3", "8", "24:00:00")):
        cd = bare / f"step_00{step}_{calc}" / f"m_{calc.lower()}"
        cd.mkdir(parents=True)
        (cd / f"m_{calc.lower()}.{ext}").write_text("dummy\n")
        script = eng._create_slurm_script_for_calculation(cd, f"m_{calc.lower()}", calc, step, "wf_t")
        text = script.read_text()
        assert _ntasks(text) == [f"#SBATCH --ntasks={want}"] and _walltime(text) == [f"#SBATCH -t {wall}"], calc


def test_quick_start_config_gets_each_types_values(tmp_path, in_env):
    _toml(tmp_path, TYPED)
    in_env()
    from mace.run_mace import _get_quick_slurm_config
    got = {c: list(_get_quick_slurm_config(c, 1)["scripts"].values())[0]["resources"] for c in ("OPT", "FREQ", "DOSS")}
    assert {c: (r["ntasks"], r["walltime"]) for c, r in got.items()} == {
        "OPT": (16, "24:00:00"), "FREQ": (64, "3-00:00:00"), "DOSS": (4, "1:30:00")}


def test_tracked_submission_gets_each_types_size(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, TYPED)
    d = tmp_path / "decks"
    _deck(d, "c_freq.d12", FREQ_D12)
    p = _mace(d, env, "submit", "--track", "c_freq.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sent and _ntasks(sent[-1]) == ["#SBATCH --ntasks=64"] and _walltime(sent[-1]) == ["#SBATCH -t 3-00:00:00"]


def test_mace_site_and_the_gui_show_each_type_and_its_source(tmp_path, in_env):
    _toml(tmp_path, TYPED)
    env = in_env(MACE_SLURM_NODES="1")
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 0, r.stdout + r.stderr
    path = str(tmp_path / "xdg" / "mace" / "site.toml")
    assert f"Walltime:    24:00:00, every job  (site.toml: {path})" in r.stdout
    assert "By calculation type" in r.stdout
    line = next(l for l in r.stdout.splitlines() if l.strip().startswith("FREQ "))
    assert "--ntasks=64 -N 1 --cpus-per-task=1 --mem-per-cpu=7G -t 3-00:00:00" in line
    assert "ntasks, mem, walltime from [resources.FREQ]" in line and "nodes from env MACE_SLURM_NODES" in line
    line = next(l for l in r.stdout.splitlines() if l.strip().startswith("BAND "))
    assert "ntasks from [resources.properties]" in line and "walltime from [resources]" in line
    from mace.gui import submit_routes
    submit_routes._SITE_CACHE.clear()
    d = submit_routes.site_describe()
    assert d["resources_by_type"]["FREQ"]["value"]["ntasks"] == 64 and d["resources_by_type"]["FREQ"]["own"]
    assert d["resources_by_type"]["DOSS"]["walltime"] == "1:30:00"
    assert d["walltime"]["prop"]["set"] and d["walltime"]["prop"]["value"] == "24:00:00"


def test_walltime_above_the_partition_limit_is_written_and_warned(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[resources]\nwalltime = "30-00:00:00"\n')
    d = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert any("30-00:00:00" in w and "partition" in w for w in d["warnings"]), d["warnings"]
    dd = tmp_path / "decks"
    _deck(dd, "m.d12", SP_D12)
    p = _mace(dd, env, "submit", "m.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert _walltime(fake.calls("sbatch")[-1]["script_text"]) == ["#SBATCH -t 30-00:00:00"]
