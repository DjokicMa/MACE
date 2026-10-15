"""The default cores per job ([resources] ntasks), on every path that offers
or writes a job size.

A user who wants 16 cores for every job sets it once in site.toml (or
MACE_SLURM_NTASKS); the workflow planner offers it, `mace submit` writes it,
the GUI's Submit dialog shows it, a job array and the next workflow step get
it, and an answer typed for one job still wins. With no setting the built-in
32 (CRYSTAL) and PROPERTIES_NTASKS (properties, 8) stay, byte for byte
(test_site_jobscript.py::test_default_bytes, test_array_default_path_golden.py)."""
import json
import os

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site

OPT_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n", "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1)


def _ntasks(text):
    return sorted({l for l in text.splitlines() if "--ntasks=" in l and "#SBATCH" in l})


def _submit(tmp, env, deck, *args, stdin=""):
    d = tmp / "decks"
    _deck(d, deck, BAND_D3 if deck.endswith(".d3") else SP_D12)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    (d / (deck.rsplit(".", 1)[0] + ".sh")).unlink(missing_ok=True)
    p = _mace(d, env, "submit", "--nosubmit", *args, deck, stdin=stdin)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return (d / (deck.rsplit(".", 1)[0] + ".sh")).read_text()


@pytest.fixture
def in_env(tmp_path, monkeypatch):
    """os.environ as _env(tmp_path) builds it (the in-process paths read it)."""
    def apply(**extra):
        env = _env(tmp_path, **extra)
        for k in list(os.environ):
            if k not in env:
                monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        site._SLURM_MEMO.clear()
        return env
    return apply


def _planner_offer(calc_type):
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner.__new__(WorkflowPlanner)
    script = p.get_required_scripts(calc_type)[0]
    return p.get_default_resources(script, calc_type)


# ---------------------------------------------------------------- no setting
def test_without_a_setting_the_built_in_sizes_stay(tmp_path, in_env):
    env = in_env()
    assert {c: _planner_offer(c)["ntasks"] for c in ("OPT", "SP", "FREQ", "BAND", "DOSS")} == \
        {"OPT": 32, "SP": 32, "FREQ": 32, "BAND": 8, "DOSS": 8}
    scf = _submit(tmp_path, env, "m.d12")
    assert scf.encode() == _template_output(tmp_path, SCF, "m")
    prop = _submit(tmp_path, env, "m_band.d3")
    assert prop.encode() == _template_output(tmp_path, PROP, "m_band")
    d = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert d["resources"]["scf"]["value"]["ntasks"] == 32 and d["resources"]["prop"]["value"]["ntasks"] == 8
    assert d["resources"]["scf"]["source"] == "built-in"


# ---------------------------------------------------------------- [resources] ntasks = 16
def test_user_setting_reaches_planner_submit_and_site(tmp_path, in_env):
    _toml(tmp_path, "[resources]\nntasks = 16\n")
    env = in_env()
    assert {c: _planner_offer(c)["ntasks"] for c in ("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT")} == \
        dict.fromkeys(("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT"), 16)
    assert _ntasks(_submit(tmp_path, env, "m.d12")) == ["#SBATCH --ntasks=16"]
    assert _ntasks(_submit(tmp_path, env, "m_band.d3")) == ["#SBATCH --ntasks=16"]
    # mace site (and the GUI's Site row, which shows mace site --json)
    r = _mace(tmp_path, env, "site")
    path = str(tmp_path / "xdg" / "mace" / "site.toml")
    assert f"Resources:   --ntasks=16 -N 1 --cpus-per-task=1 --mem-per-cpu=5G  (site.toml: {path})" in r.stdout
    assert f"Properties:  --ntasks=16 -N 1 --cpus-per-task=1 --mem=80G  (site.toml: {path})" in r.stdout
    from mace.gui import submit_routes
    submit_routes._SITE_CACHE.clear()
    gui = submit_routes.site_describe()["resources"]
    assert gui["scf"]["value"]["ntasks"] == 16 and gui["prop"]["value"]["ntasks"] == 16
    assert gui["scf"]["source"] == "site.toml"


def test_properties_built_in_has_one_source(tmp_path, in_env):
    """Every place that offers or writes the properties built-in takes it from
    workflow.common.constants.PROPERTIES_NTASKS (the template line included)."""
    from mace.workflow.common.constants import PROPERTIES_NTASKS, SLURM_RESOURCE_DEFAULTS
    from mace.submission import properties
    from mace import run_mace
    in_env()
    assert PROPERTIES_NTASKS == 8
    for c in ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"):
        assert SLURM_RESOURCE_DEFAULTS[c]["ntasks"] == PROPERTIES_NTASKS
        assert _planner_offer(c)["ntasks"] == PROPERTIES_NTASKS
        cfg = run_mace._get_quick_slurm_config(c, 1)
        assert cfg["scripts"]["submit_prop.sh"]["resources"]["ntasks"] == PROPERTIES_NTASKS
    assert site.TEMPLATE_RESOURCES["prop"]["ntasks"] == PROPERTIES_NTASKS
    assert site.RESOURCE_LINES["prop"]["ntasks"] in PROP.read_text().splitlines()
    assert properties.get_default_d3_resources()["ntasks"] == PROPERTIES_NTASKS
    assert f"properties {PROPERTIES_NTASKS})" in site.INIT_TEXT
    assert f"# ntasks = {PROPERTIES_NTASKS} " in site.INIT_TEXT
    # the CRYSTAL jobs keep 32
    assert {c: SLURM_RESOURCE_DEFAULTS[c]["ntasks"] for c in ("OPT", "SP", "FREQ")} == dict.fromkeys(("OPT", "SP", "FREQ"), 32)


def test_properties_memory_is_one_total_on_every_path(tmp_path, in_env):
    """Quick start, the planner, mace submit and the executor write the same
    --mem (PROPERTIES_MEMORY), and the site's [resources.properties] mem wins
    on each of them (the planner does not scale it)."""
    from mace.workflow.common.constants import PROPERTIES_MEMORY, SLURM_RESOURCE_DEFAULTS
    from mace.workflow.executor import WorkflowExecutor
    from mace.submission import properties
    from mace import run_mace
    in_env()
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path
    template = PROP.read_text()
    assert f"echo '#SBATCH --mem={PROPERTIES_MEMORY}' >> $1.sh" in template.splitlines()
    assert properties.get_default_d3_resources()["memory"] == PROPERTIES_MEMORY
    for c in ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "BAND2"):
        base = c.rstrip("0123456789")
        assert SLURM_RESOURCE_DEFAULTS[base]["memory"] == PROPERTIES_MEMORY
        offer = _planner_offer(c)
        assert offer["memory"] == PROPERTIES_MEMORY and "memory_per_cpu" not in offer
        res = run_mace._get_quick_slurm_config(c, 1)["scripts"]["submit_prop.sh"]["resources"]
        assert res["memory"] == PROPERTIES_MEMORY and "memory_per_cpu" not in res
        text = ex.apply_script_customizations(template, {"resources": res, "workflow_id": "w"})
        assert [l for l in text.splitlines() if "--mem" in l] == [f"echo '#SBATCH --mem={PROPERTIES_MEMORY}' >> $1.sh"]
    _toml(tmp_path, '[resources.properties]\nmem = "100G"\n')
    env = in_env()
    assert _planner_offer("BAND")["memory"] == "100G"
    res = run_mace._get_quick_slurm_config("DOSS", 1)["scripts"]["submit_prop.sh"]["resources"]
    assert res["memory"] == "100G"
    text = ex.apply_script_customizations(template, {"resources": res, "workflow_id": "w"})
    assert [l for l in text.splitlines() if "--mem" in l] == ["echo '#SBATCH --mem=100G' >> $1.sh"]
    assert "#SBATCH --mem=100G" in _submit(tmp_path, env, "m_band.d3")
    _toml(tmp_path, '[resources.properties]\nmem_per_cpu = "3G"\n')
    in_env()
    res = run_mace._get_quick_slurm_config("BAND", 1)["scripts"]["submit_prop.sh"]["resources"]
    text = ex.apply_script_customizations(template, {"resources": res, "workflow_id": "w"})
    assert [l for l in text.splitlines() if "--mem" in l] == ["echo '#SBATCH --mem-per-cpu=3G' >> $1.sh"]


def test_site_properties_level_restores_the_old_size(tmp_path, in_env):
    """[resources.properties] ntasks = 28 gives back the size before 1.1.5."""
    _toml(tmp_path, "[resources.properties]\nntasks = 28\n")
    env = in_env()
    assert _planner_offer("BAND")["ntasks"] == 28 and _planner_offer("OPT")["ntasks"] == 32
    assert _ntasks(_submit(tmp_path, env, "m_band.d3")) == ["#SBATCH --ntasks=28"]
    assert _ntasks(_submit(tmp_path, env, "m.d12")) == ["#SBATCH --ntasks=32"]
    env = in_env(MACE_SLURM_NTASKS="12")          # the environment still wins
    assert _ntasks(_submit(tmp_path, env, "m_band.d3")) == ["#SBATCH --ntasks=12"]


def test_properties_level_overrides_the_global_one(tmp_path, in_env):
    _toml(tmp_path, "[resources]\nntasks = 16\n[resources.properties]\nntasks = 14\n")
    env = in_env()
    assert _planner_offer("OPT")["ntasks"] == 16 and _planner_offer("BAND")["ntasks"] == 14
    assert _ntasks(_submit(tmp_path, env, "m.d12")) == ["#SBATCH --ntasks=16"]
    assert _ntasks(_submit(tmp_path, env, "m_band.d3")) == ["#SBATCH --ntasks=14"]


def test_environment_variable_sets_it_too(tmp_path, in_env):
    env = in_env(MACE_SLURM_NTASKS="16")
    assert _planner_offer("SP")["ntasks"] == 16 and _planner_offer("BAND")["ntasks"] == 16
    assert _ntasks(_submit(tmp_path, env, "m.d12")) == ["#SBATCH --ntasks=16"]
    assert _ntasks(_submit(tmp_path, env, "m_band.d3")) == ["#SBATCH --ntasks=16"]


def test_an_answer_for_one_job_wins(tmp_path, in_env):
    _toml(tmp_path, "[resources]\nntasks = 16\n")
    env = in_env()
    # mace submit -i: the offer is 16, a typed 24 is written
    sh = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n24\n\n\n\nn\n")
    assert _ntasks(sh) == ["#SBATCH --ntasks=24"]
    sh = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n\n\n\n\nn\n")
    assert _ntasks(sh) == ["#SBATCH --ntasks=16"]
    # the environment wins over site.toml
    env = in_env(MACE_SLURM_NTASKS="8")
    assert _ntasks(_submit(tmp_path, env, "m.d12")) == ["#SBATCH --ntasks=8"]


# ---------------------------------------------------------------- workflows and arrays
def test_workflow_steps_and_next_step_get_it(tmp_path, monkeypatch):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, "[resources]\nntasks = 16\n[resources.properties]\nntasks = 14\n")
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", OPT_D12)
    p = _mace(tmp_path, env, "submit", "--progress", "full_electronic", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    scripts = {c.name: c.read_text() for c in sorted((d / "workflow_scripts").glob("*.sh"))}
    assert scripts
    for name, text in scripts.items():
        want = "14" if name.startswith("submit_prop") else "16"
        assert _ntasks(text) == [f"echo '#SBATCH --ntasks={want}' >> $1.sh"], (name, _ntasks(text))
    # the plan stores each step's size (what the next steps are submitted with)
    plan = json.loads(next((d / "workflow_configs").glob("workflow_plan_*.json")).read_text())
    sizes = {step: [s["resources"]["ntasks"] for s in cfg["slurm_config"]["scripts"].values()]
             for step, cfg in plan["step_configurations"].items()}
    assert sizes == {"OPT_1": [16], "SP_2": [16], "BAND_3": [14], "DOSS_4": [14]}
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sent and all(_ntasks(s) == ["#SBATCH --ntasks=16"] for s in sent)
    # the next step in a folder without workflow_scripts/: the generator mace submit uses
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from mace.workflow.engine import WorkflowEngine
    from types import SimpleNamespace
    eng = WorkflowEngine.__new__(WorkflowEngine)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "materials.db").touch()
    eng.base_work_dir, eng.db = bare, SimpleNamespace(db_path=bare / "materials.db")
    for step, calc, ext, want in ((2, "SP", "d12", "16"), (3, "BAND", "d3", "14")):
        cd = bare / f"step_00{step}_{calc}" / f"m_{calc.lower()}"
        cd.mkdir(parents=True)
        (cd / f"m_{calc.lower()}.{ext}").write_text("dummy\n")
        script = eng._create_slurm_script_for_calculation(cd, f"m_{calc.lower()}", calc, step, "wf_t")
        assert _ntasks(script.read_text()) == [f"#SBATCH --ntasks={want}"], calc


def test_job_array_gets_it(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, "[resources]\nntasks = 16\n")
    d = tmp_path / "decks"
    for n in ("a", "b"):
        _deck(d, f"{n}_sp.d12", SP_D12)
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sent and all(_ntasks(s) == ["#SBATCH --ntasks=16"] for s in sent), [_ntasks(s) for s in sent]


def test_workflow_quick_start_gets_it(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, "[resources]\nntasks = 16\n")
    d = tmp_path / "decks"
    _deck(d, "a_opt.d12", OPT_D12)
    p = _mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks", "--workflow", "opt_sp")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    sent = [c["script_text"] for c in fake.calls("sbatch")]
    assert sent and all(_ntasks(s) == ["#SBATCH --ntasks=16"] for s in sent), [_ntasks(s) for s in sent]
