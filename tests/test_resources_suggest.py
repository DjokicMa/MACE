"""Size-aware job sizes ([resources] auto, mace/submission/resources_suggest.py).

One function sizes a job from what MACE knows before it runs (AOs,
irreducible k-points, spins, symmetry, type, the site's limits); `mace site
suggest`, the GUI and, with ``auto = true``, every path that offers or writes
a job size use it. Per key: a value chosen for one job > the environment >
the values set in site.toml > the suggestion > the built-in value. With auto
off (the default) nothing changes, byte for byte.
"""
import json

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_default_cores import OPT_D12, _ntasks, _planner_offer, in_env  # noqa: F401 (fixture)
from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import resources_suggest as rs
from mace.submission import site

FREQ_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n", "6 0.125 0.125 0.125\nFREQCALC\nEND\nEND\n", 1)
SPIN_D12 = SP_D12.replace("B3LYP\nEND\n", "B3LYP\nSPIN\nEND\n", 1)
SUPER_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                           "6 0.125 0.125 0.125\nSUPERCEL\n2 0 0\n0 2 0\n0 0 2\nEND\n", 1)
DOSS_D3 = "NEWK\n4 4\n1 0\nDOSS\n0 100 -1 -1 1 12 0\nEND\n"
# the start of a CRYSTAL output: what sizing reads (resources_suggest.read_out_size)
OUT_HEAD = """ CRYSTAL CALCULATION
 NUMBER OF SYMMETRY OPERATORS         :    4
 N. OF ATOMS PER CELL       120  COULOMB OVERLAP TOL         (T1) 10**   -7
 NUMBER OF AO              1800  EXCHANGE OVERLAP TOL        (T3) 10**   -7
 TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL
 SHRINK. FACT.(MONKH.)    4  4  4  NUMBER OF K POINTS IN THE IBZ     24
 CYC   0 ETOT(AU) -7.685930554264E+01 DETOT -7.69E+01 tst  0.00E+00 PX  1.00E+00
"""
AUTO = "[resources]\nauto = true\n"


def _walltime(text):
    return sorted({l for l in text.splitlines() if l.startswith("#SBATCH -t ")})


def _mem(text):
    return sorted({l for l in text.splitlines() if l.startswith("#SBATCH --mem")})


def _load(tmp_path, text, **env):
    path = tmp_path / "site.toml"
    path.write_text(text)
    return site.load({"MACE_SITE_CONFIG": str(path), "MACE_SITE_GROUP_CONFIG": "none", **env})


# ---------------------------------------------------------------- the rule
SMALL = rs.SizeInfo(nao=10, nk=29, nspin=1, nsym=48, natoms=2, nirr=1)
BIG = rs.SizeInfo(nao=1800, nk=24, nspin=2, nsym=4, natoms=120, nirr=30)


def test_small_jobs_ask_for_few_cores_and_large_ones_a_node():
    s = rs.suggest_resources("SP", SMALL)
    assert (s["ntasks"], s["mem_per_cpu"], s["walltime"]) == (1, "2G", "0-02:00:00")
    b = rs.suggest_resources("SP", BIG)
    assert b["ntasks"] == 32 and b["walltime"] > s["walltime"]
    assert b["reason"].startswith("~1,800 AOs, 24 k-points x 2 spins -> 32 cores")
    # the node size MACE knows bounds it, and so does the useful parallelism
    wide = rs.suggest_resources("SP", BIG, {"node_cores": 128})
    assert 32 < wide["ntasks"] <= 113 and "113 cores useful" in wide["reason"]          # AO/16
    assert rs.suggest_resources("SP", BIG, {"node_cores": 16})["ntasks"] == 16
    # a FREQ is many SCFs: longer than an SP of the same cell, never above the cap
    f = rs.suggest_resources("FREQ", BIG, {"max_walltime_s": 3 * 86400})
    assert f["walltime"] == "3-00:00:00" and "the most allowed" in f["reason"]
    assert "90 displacements" in f["reason"]


def test_properties_jobs_are_small_and_use_total_memory():
    for t in ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"):
        s = rs.suggest_resources(t, SMALL)
        assert s["ntasks"] == 1 and s["walltime"] == "0-01:00:00" and s["mem"] == "4G", t
    b = rs.suggest_resources("DOSS", BIG)
    assert 1 < b["ntasks"] <= 18 and int(b["mem"].rstrip("G")) >= 4


def test_memory_is_conservative_and_follows_a_chosen_core_count():
    s = rs.suggest_resources("OPT", BIG)
    # never below 2 GB per core; the envelope covers every peak of the calibration data
    assert int(s["mem_per_cpu"].rstrip("G")) >= 2
    assert s["estimate"]["mem_mb_per_task"] >= 2.6 * 428
    chosen = rs.suggest_resources("OPT", BIG, ntasks=8)
    assert chosen["ntasks"] == 8 and "8 cores (chosen)" in chosen["reason"]
    assert chosen["walltime"] >= s["walltime"]           # fewer cores: at least as long
    # a node with too little memory for every core: fewer cores, said why
    tight = rs.suggest_resources("FREQ", BIG, {"node_cores": 32, "node_mem_mb": 40 * 1024})
    assert tight["ntasks"] < 32 and "fit the node's memory" in tight["reason"]


def test_unknown_types_and_sizes_give_no_suggestion():
    assert rs.suggest_resources("NEB", SMALL) is None
    assert rs.suggest_resources("SP", rs.SizeInfo()) is None
    assert rs.suggest_resources("sp2", SMALL)["calc_type"] == "SP"
    assert rs.suggest_resources("SP", SMALL.as_dict())["ntasks"] == 1


# ---------------------------------------------------------------- the inputs
def test_deck_sizing_counts_aos_atoms_and_kpoints(tmp_path):
    pytest.importorskip("ase")                                              # the cell is built with ase
    _deck(tmp_path, "m.d12", SP_D12)
    i = rs.deck_size(tmp_path / "m.d12")
    # 2 C atoms x (s + sp) = 10 AOs; SHRINK 8 on diamond: 29 irreducible k-points; Oh: 48 operators
    assert (i.nao, i.natoms, i.nk, i.nsym, i.nspin, i.nirr) == (10, 2, 29, 48, 1, 1)
    _deck(tmp_path, "x.d12", SUPER_D12)
    x = rs.deck_size(tmp_path / "x.d12")
    assert (x.nao, x.natoms) == (80, 16)


def test_deck_sizing_counts_spin(tmp_path):
    _deck(tmp_path, "s.d12", SPIN_D12)
    assert rs.size_for(tmp_path / "s.d12").nspin == 2


def test_internal_basis_sets_count_from_the_bundled_libraries():
    assert rs.element_aos("POB-TZVP-REV2", 6) == (18, True)
    assert rs.element_aos("POB-TZVP", 6) == (18, True)
    assert rs.element_aos("POB-DZVP-REV2", 6) == (14, True)
    assert rs.element_aos("SOLDEF2MSVP", 6)[0] == 15 and rs.element_aos("SOLDEF2MSVP", 1)[0] == 2
    assert rs.element_aos("STO-3G", 8)[0] == 5
    assert rs.record_aos(["6 2", "0 0 2 2. 1.", "8.0 1.0", "2.0 1.0", "0 1 1 4. 1.", "1.0 1.0 1.0"])[:2] == (6, 5)


def test_the_parent_output_sizes_its_children(tmp_path):
    (tmp_path / "mof_opt.out").write_text(OUT_HEAD)
    _deck(tmp_path, "mof_opt_sp.d12", SP_D12)
    _deck(tmp_path, "mof_opt_band.d3", BAND_D3)
    assert rs.find_size_out(tmp_path / "mof_opt_sp.d12").name == "mof_opt.out"
    assert rs.find_size_out(tmp_path / "mof_opt_band.d3").name == "mof_opt.out"
    i = rs.size_for(tmp_path / "mof_opt_sp.d12")
    # the output's AOs, k-points and symmetry; the deck's own spin
    assert (i.nao, i.nk, i.nsym, i.natoms, i.nspin) == (1800, 24, 4, 120, 1)
    assert i.source == "out: mof_opt.out"
    assert rs.size_for(tmp_path / "mof_opt_band.d3").nspin == 2
    # a .d3 with no output to read: not sized
    _deck(tmp_path / "lone", "x_band.d3", BAND_D3)
    assert rs.suggest_for_deck(tmp_path / "lone" / "x_band.d3") is None


# ---------------------------------------------------------------- precedence
def test_auto_is_off_by_default_and_per_type(tmp_path):
    assert not _load(tmp_path, "").auto("scf", "SP")
    s = _load(tmp_path, AUTO + "[resources.OPT]\nauto = false\n[resources.properties]\nauto = false\n"
                              "[resources.BAND]\nauto = true\n")
    assert s.auto_types("scf") == ["SP", "FREQ", "ELASTIC", "EOS", "QHA", "CPHF"]
    assert s.auto_types("prop") == ["BAND"]
    with pytest.raises(site.SiteConfigError, match="auto"):
        _load(tmp_path, '[resources]\nauto = "sometimes"\n')


def test_set_values_and_the_environment_win_over_the_suggestion(tmp_path):
    auto = rs.suggest_resources("SP", BIG)
    s = _load(tmp_path, AUTO)
    eff = s.effective_resources("scf", "SP", auto)
    assert (eff["ntasks"], eff["mem"], s.walltime("scf", "SP", auto)) == (32, "2G", auto["walltime"])
    # a value set in site.toml wins per key
    s = _load(tmp_path, '[resources]\nauto = true\nntasks = 16\n[resources.SP]\nwalltime = "12:00:00"\n')
    eff = s.effective_resources("scf", "SP", auto)
    assert eff["ntasks"] == 16 and eff["mem"] == "2G" and s.walltime("scf", "SP", auto) == "12:00:00"
    # ... and the environment over both
    s = _load(tmp_path, AUTO, MACE_SLURM_NTASKS="8", MACE_SLURM_MEM_PER_CPU="3G")
    eff = s.effective_resources("scf", "SP", auto)
    assert (eff["ntasks"], eff["mem"]) == (8, "3G")
    # the suggestion's memory takes the template's form (properties: --mem)
    p = rs.suggest_resources("DOSS", BIG)
    assert _load(tmp_path, AUTO).effective_resources("prop", "DOSS", p)["mem_form"] == "mem"
    # without a suggestion, exactly the values of before
    assert s.resources_set("scf", "SP", None) == s.resources_set("scf", "SP")


def test_a_set_core_count_resizes_memory_and_walltime(tmp_path, in_env):
    _toml(tmp_path, '[resources]\nauto = true\nntasks = 4\n')
    in_env()
    (tmp_path / "m_opt.out").write_text(OUT_HEAD)
    _deck(tmp_path, "m_opt_sp.d12", SP_D12)
    a = site.auto_for(tmp_path / "m_opt_sp.d12")
    assert a["ntasks"] == 4 and "4 cores (chosen)" in a["reason"]
    assert a["walltime"] == rs.suggest_resources("SP", rs.size_for(tmp_path / "m_opt_sp.d12"), {}, ntasks=4)["walltime"]


def test_walltime_never_above_the_partition_limit(tmp_path):
    from test_walltime_partition_maxtime import slurm_maxtime
    path = tmp_path / "site.toml"
    path.write_text(AUTO)
    env = slurm_maxtime(tmp_path, {"MACE_SITE_CONFIG": str(path), "MACE_SITE_GROUP_CONFIG": "none",
                                   "XDG_CACHE_HOME": str(tmp_path / "cache")}, "5:00:00")
    s = site.load_resolved(env)
    assert s.walltime("scf", "OPT", {"walltime": "5-00:00:00"}) == "0-05:00:00"


# ---------------------------------------------------------------- off: byte for byte
@pytest.mark.parametrize("text", ["", "[resources]\nauto = false\n", "[resources.BAND]\nauto = true\n"])
def test_off_keeps_every_script_byte_for_byte(tmp_path, text):
    _toml(tmp_path, text)
    env = _env(tmp_path)
    d = tmp_path / "decks"
    _deck(d, "m.d12", FREQ_D12)
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m.sh").read_bytes() == _template_output(tmp_path, SCF, "m")
    assert "suggested" not in p.stdout
    # auto on, but a deck MACE cannot size (no parent output): the values of before
    _deck(d, "m_doss.d3", DOSS_D3)
    (d / "m_doss.f9").write_text("fort9\n")
    p = _mace(d, env, "submit", "--nosubmit", "m_doss.d3")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m_doss.sh").read_bytes() == _template_output(tmp_path, PROP, "m_doss")


def test_off_keeps_planner_and_quick_start_offers(tmp_path, in_env):
    _toml(tmp_path, "[resources]\nauto = false\n")
    in_env()
    from mace.run_mace import _get_quick_slurm_config
    from mace.workflow.planner import WorkflowPlanner
    _deck(tmp_path, "m.d12", SP_D12)
    p = WorkflowPlanner.__new__(WorkflowPlanner)
    p.auto_inputs = [tmp_path / "m.d12"]
    assert p.get_default_resources("submitcrystal23.sh", "SP") == _planner_offer("SP")
    got = _get_quick_slurm_config("SP", 1, [tmp_path / "m.d12"])
    assert got == _get_quick_slurm_config("SP", 1)


# ---------------------------------------------------------------- every consumer
def test_submit_writes_each_decks_suggestion_and_says_why(tmp_path, in_env):
    _toml(tmp_path, AUTO)
    env = in_env()
    d = tmp_path / "decks"
    (d).mkdir()
    (d / "big_opt.out").write_text(OUT_HEAD)
    _deck(d, "big_opt_sp.d12", SP_D12)        # sized from big_opt.out: 1,800 AOs
    _deck(d, "small_sp.d12", SP_D12)          # sized from the deck: 10 AOs
    _deck(d, "small_freq.d12", FREQ_D12)
    p = _mace(tmp_path, env, "submit", "--nosubmit", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    big, small, freq = ((d / f"{n}.sh").read_text() for n in ("big_opt_sp", "small_sp", "small_freq"))
    want_big = rs.suggest_for_deck(d / "big_opt_sp.d12")
    assert _ntasks(big) == [f"#SBATCH --ntasks={want_big['ntasks']}"] == ["#SBATCH --ntasks=32"]
    assert _mem(big) == [f"#SBATCH --mem-per-cpu={want_big['mem_per_cpu']}"]
    assert _walltime(big) == [f"#SBATCH -t {want_big['walltime']}"]
    assert _ntasks(small) == ["#SBATCH --ntasks=1"] and _walltime(small) == ["#SBATCH -t 0-02:00:00"]
    assert _ntasks(freq) == ["#SBATCH --ntasks=1"]
    assert "~1,800 AOs, 24 k-points x 1 spins" not in p.stdout          # the deck has no SPIN
    assert "~1,800 AOs, 24 k-points -> 32 cores" in p.stdout and "([resources] auto)" in p.stdout
    assert not list(d.glob(".submitcrystal23_site*")), "the rewritten generators are removed"


def test_submit_properties_sized_from_the_parent_output(tmp_path, in_env):
    _toml(tmp_path, "[resources.DOSS]\nauto = true\n")
    env = in_env()
    d = tmp_path / "props"
    d.mkdir()
    (d / "m.out").write_text(OUT_HEAD)
    _deck(d, "m_doss.d3", DOSS_D3)
    _deck(d, "m_band.d3", BAND_D3)
    for n in ("m_doss", "m_band"):
        (d / f"{n}.f9").write_text("fort9\n")
    p = _mace(tmp_path, env, "submit", "--nosubmit", "props")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    want = rs.suggest_for_deck(d / "m_doss.d3")
    doss, band = (d / "m_doss.sh").read_text(), (d / "m_band.sh").read_text()
    assert _ntasks(doss) == [f"#SBATCH --ntasks={want['ntasks']}"] and _mem(doss) == [f"#SBATCH --mem={want['mem']}"]
    # BAND's suggestions are off: the built-in properties cores
    from mace.workflow.common.constants import PROPERTIES_NTASKS
    assert _ntasks(band) == [f"#SBATCH --ntasks={PROPERTIES_NTASKS}"]


def test_interactive_offers_the_suggestion_and_answers_win(tmp_path, in_env):
    _toml(tmp_path, AUTO)
    env = in_env()
    d = tmp_path / "decks"
    _deck(d, "a_sp.d12", SP_D12)
    p = _mace(tmp_path, env, "submit", "--nosubmit", "-i", "decks", stdin="y\n6\n\n\n\nn\n")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "defaults suggested from their size ([resources] auto)" in p.stdout
    assert "Cores (ntasks) [1]" in p.stdout and "Walltime [0-02:00:00]" in p.stdout
    sh = (d / "a_sp.sh").read_text()
    assert _ntasks(sh) == ["#SBATCH --ntasks=6"] and _walltime(sh) == ["#SBATCH -t 0-02:00:00"]


def test_job_array_and_tracked_runs_get_the_suggestion(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, AUTO)
    d = tmp_path / "decks"
    d.mkdir()
    (d / "big_opt.out").write_text(OUT_HEAD)
    _deck(d, "big_opt_sp.d12", SP_D12)
    _deck(d, "small_sp.d12", SP_D12)
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sorted(tuple(_ntasks(s)) for s in sent) == [("#SBATCH --ntasks=1",), ("#SBATCH --ntasks=32",)]
    t = tmp_path / "tracked"
    t.mkdir()
    (t / "big_opt.out").write_text(OUT_HEAD)
    _deck(t, "big_opt_sp.d12", SP_D12)
    p = _mace(t, env, "submit", "--track", "big_opt_sp.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert _ntasks(fake.calls("sbatch")[-1]["script_text"]) == ["#SBATCH --ntasks=32"]


def test_planner_quick_start_and_progress_plans_store_the_largest_inputs_size(tmp_path, in_env, monkeypatch):
    _toml(tmp_path, AUTO)
    in_env()
    d = tmp_path / "in"
    d.mkdir()
    (d / "big_opt.out").write_text(OUT_HEAD)
    _deck(d, "big_opt_sp.d12", SP_D12)
    _deck(d, "small.d12", SP_D12)
    inputs = [d / "big_opt_sp.d12", d / "small.d12"]
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner.__new__(WorkflowPlanner)
    p.auto_inputs = inputs
    offer = p.get_default_resources("submitcrystal23.sh", "SP")
    assert offer["ntasks"] == 32 and p.last_auto["reason"].startswith("largest of 2 inputs (big_opt_sp.d12)")
    assert offer["memory_per_cpu"] == p.last_auto["mem_per_cpu"] and offer["walltime"] == p.last_auto["walltime"]
    from mace.run_mace import _get_quick_slurm_config
    q = _get_quick_slurm_config("SP", 2, inputs)["resources"]
    assert (q["ntasks"], q["walltime"]) == (offer["ntasks"], offer["walltime"])
    band = _get_quick_slurm_config("BAND", 3, inputs)["resources"]
    assert band["ntasks"] == rs.suggest_for_inputs(inputs, "BAND")["ntasks"]


def test_progress_plan_and_the_next_step_in_a_plain_folder(tmp_path, monkeypatch):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, AUTO)
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", OPT_D12)
    p = _mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    plan = json.loads(next((d / "workflow_configs").glob("workflow_plan_*.json")).read_text())
    res = {step: list(cfg["slurm_config"]["scripts"].values())[0]["resources"]
           for step, cfg in plan["step_configurations"].items()}
    assert {s: r["ntasks"] for s, r in res.items()} == {"OPT_1": 1, "SP_2": 1}
    assert res["SP_2"]["walltime"] == "0-02:00:00"
    assert _ntasks(fake.calls("sbatch")[-1]["script_text"]) == ["#SBATCH --ntasks=1"]
    # the next step of a --track run in a folder without workflow_scripts/
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
    cd = bare / "step_002_SP" / "m_sp"
    cd.mkdir(parents=True)
    (cd / "m_opt.out").write_text(OUT_HEAD)
    (cd / "m_opt_sp.d12").write_text(SP_D12)
    text = eng._create_slurm_script_for_calculation(cd, "m_opt_sp", "SP", 2, "wf_t").read_text()
    assert _ntasks(text) == ["#SBATCH --ntasks=32"]


# ---------------------------------------------------------------- mace site
def test_mace_site_set_suggest_and_show(tmp_path):
    env = _env(tmp_path)
    r = _mace(tmp_path, env, "site", "set", "resources.auto", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("Now: size-aware suggestions on for OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, "
            "TRANSPORT, CHARGE+POTENTIAL") in r.stdout
    r = _mace(tmp_path, env, "site", "set", "resources.band.auto", "false")
    assert r.returncode == 0 and "BAND" not in r.stdout.split("on for", 1)[1].split("\n")[0]
    assert _mace(tmp_path, env, "site", "set", "resources.auto", "maybe").returncode == 2
    text = (tmp_path / "xdg" / "mace" / "site.toml").read_text()
    assert "auto = true" in text and "[resources.BAND]" in text
    shown = _mace(tmp_path, env, "site").stdout
    assert "Suggestions: on for OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, DOSS, TRANSPORT, CHARGE+POTENTIAL" in shown
    d = tmp_path / "d"
    d.mkdir()
    (d / "big_opt.out").write_text(OUT_HEAD)
    _deck(d, "big_opt_sp.d12", SP_D12)
    _deck(d, "x_band.d3", BAND_D3)
    r = _mace(d, env, "site", "suggest", "big_opt_sp.d12")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Suggested:  --ntasks=32 --mem-per-cpu=" in r.stdout and "Why:        ~1,800 AOs" in r.stdout
    assert "Sized from: out: big_opt.out" in r.stdout and "Job script: --ntasks=32" in r.stdout
    r = _mace(d, env, "site", "suggest", "--json", "--type", "FREQ", "big_opt_sp.d12", "x_band.d3")
    assert r.returncode == 1                       # x_band.d3 cannot be sized
    got = json.loads(r.stdout)
    assert got[0]["calc_type"] == "FREQ" and got[0]["suggestion"]["calc_type"] == "FREQ"
    assert got[1]["suggestion"] is None and "could not size" in got[1]["error"]


def test_init_file_documents_auto(tmp_path):
    env = _env(tmp_path)
    assert _mace(tmp_path, env, "site", "--init").returncode == 0
    text = (tmp_path / "xdg" / "mace" / "site.toml").read_text()
    assert "# auto          = true" in text
    assert not _load(tmp_path, text).auto("scf", "OPT")


# ---------------------------------------------------------------- the GUI's function == the CLI's
def test_gui_route_and_cli_return_the_same_suggestions(tmp_path, in_env):
    _toml(tmp_path, AUTO + "[resources.FREQ]\nntasks = 12\n")
    env = in_env()
    d = tmp_path / "proj"
    d.mkdir()
    (d / "big_opt.out").write_text(OUT_HEAD)
    (d / "m.out").write_text(OUT_HEAD)
    decks = {"big_opt_sp.d12": SP_D12, "small_sp.d12": SP_D12, "f_freq.d12": FREQ_D12, "o_opt.d12": OPT_D12,
             "spin.d12": SPIN_D12, "super.d12": SUPER_D12, "m_doss.d3": DOSS_D3, "m_band.d3": BAND_D3}
    for n, t in decks.items():
        _deck(d, n, t)
    r = _mace(d, env, "site", "suggest", "--json", *decks)
    assert r.returncode == 0, r.stdout + r.stderr
    cli = json.loads(r.stdout)

    class Handler:
        headers = {}

        def __init__(self):
            from types import SimpleNamespace
            self.state = SimpleNamespace(root=d, resolve=lambda p: (d / p).resolve(),
                                         rel=lambda p: p.relative_to(d).as_posix())

        def _json(self, obj, status=200):
            self.out = (status, json.loads(json.dumps(obj, default=str)))
            return self.out
    from mace.gui import submit_routes
    h = Handler()
    submit_routes.get(h, "submit/suggest", {"path": list(decks)})
    status, gui = h.out
    assert status == 200
    gui = gui["suggestions"]
    assert [g["deck"] for g in gui] == list(decks)
    for c, g in zip(cli, gui):
        assert c["suggestion"] == g["suggestion"], c["deck"]
        assert (c["applies"], c["auto"], c["calc_type"], c["error"]) == (g["applies"], g["auto"], g["calc_type"],
                                                                        g["error"]), c["deck"]
        assert c["suggestion"] is not None, c["deck"]
    by = {g["deck"]: g for g in gui}
    assert by["f_freq.d12"]["applies"]["ntasks"] == 12                     # set for FREQ: it wins
    assert by["big_opt_sp.d12"]["applies"]["ntasks"] == 32


def test_gui_preview_stages_the_parent_output_head(tmp_path, in_env):
    _toml(tmp_path, AUTO)
    in_env()
    from mace.gui import submit as sub
    d = tmp_path / "proj"
    d.mkdir()
    big = OUT_HEAD + " CYC   1 ...\n" + "x" * 10000 + "\n"
    (d / "big_opt.out").write_text(big)
    _deck(d, "big_opt_sp.d12", SP_D12)
    sel = sub.selection(d, [d / "big_opt_sp.d12"])
    items, size = sub.stage_plan(sel, d)
    assert (d / "big_opt.out", "outhead") in items and size < len(big) + len(SP_D12) + 100
    mirror = tmp_path / "mirror"
    sub.stage(sel, d, mirror)
    staged = (mirror / "big_opt.out").read_text()
    assert staged.startswith(OUT_HEAD.split(" CYC ")[0]) and "x" * 100 not in staged
    assert rs.suggest_for_deck(mirror / "big_opt_sp.d12") == {**rs.suggest_for_deck(d / "big_opt_sp.d12")}
    _toml(tmp_path, "")
    assert (d / "big_opt.out", "outhead") not in sub.stage_plan(sel, d)[0]


# ---------------------------------------------------------------- the node shape it sizes for (1.1.6 review)
SHAPES = {"small": {"cores": 32, "max_cores": 32, "mem_mb": 128000, "max_mem_mb": 128000},
          "big": {"cores": 128, "max_cores": 128, "mem_mb": 1000000, "max_mem_mb": 1000000},
          "mid": {"cores": 64, "max_cores": 64, "mem_mb": 256000, "max_mem_mb": 256000}}


def _cached_shapes(tmp_path, default):
    from fake_slurm import FakeSlurm
    from mace.utils import envmodules
    parts = "".join(f"PartitionName={p} MaxTime=7-00:00:00 State=UP{' Default=YES' if p == default else ''}\n"
                    for p in SHAPES)
    fake = FakeSlurm(tmp_path / "fs-shapes", {"partitions": parts})
    env = {"XDG_CACHE_HOME": str(tmp_path / "cache"), "PATH": f"{fake.bindir}:/usr/bin:/bin",
           "FAKE_SLURM_STATE": fake.env["FAKE_SLURM_STATE"], "FAKE_SLURM_LOG": fake.env["FAKE_SLURM_LOG"]}
    envmodules.cache_put(json.dumps(["SLURM-shape", "", "sinfo"]), {"shapes": SHAPES, "name": "SLURM"}, env)
    site._SLURM_MEMO.clear()
    return env


def test_the_suggestion_sizes_for_the_default_partition_not_the_largest(tmp_path):
    env = _cached_shapes(tmp_path, "mid")
    lim = rs.limits_for(site.Site(), "scf", env)
    assert (lim["node_cores"], lim["node_mem_mb"]) == (64, 256000)


def test_without_a_default_partition_the_smallest_node(tmp_path):
    env = _cached_shapes(tmp_path, None)
    lim = rs.limits_for(site.Site(), "scf", env)
    assert (lim["node_cores"], lim["node_mem_mb"]) == (32, 128000)


def test_a_set_partition_still_wins(tmp_path):
    env = _cached_shapes(tmp_path, "mid")
    s = site.Site()
    s.partition = "big"
    assert rs.limits_for(s, "scf", env)["node_cores"] == 128
