"""The workflow builder's server side (GUI_WORKFLOW_DESIGN.md sections 3-6,
Phase 4) on a real `mace gui` with the logging fake SLURM: the GUI writes a
workflow spec and runs the CLI (`mace workflow --spec SPEC --write-plan`,
then the launch command it prints); the preview runs the same commands in a
staged copy (SLURM's state changes recorded, not made) and the real run's
job scripts equal the preview's. Preview ids are single use; a changed
folder generation is 409; a network-bound server refuses the run; the same
structures cannot be launched twice while the first launch's jobs are live.
"""
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

import guiharness as H
import submit_util as SU
from submit_util import norm

TEST_DATA = H.REPO_ROOT / "test"
CKSUM = re.compile(r"cksum of the lines above: \d+ \d+")
CIF = TEST_DATA / "CIFs" / "1_dia_opt_rev1.cif"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    if not CIF.exists():
        pytest.skip("test/ data corpus not present")
    s = SU.Served(tmp_path_factory.mktemp("wfbuild"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _cifs(srv, folder, names=("A", "B")):
    d = srv.root / folder
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(CIF, d / f"{n}.cif")
    return d


def _gen(srv):
    return srv.req("api/info")[2]["root_gen"]


def _draft(folder, names=("A", "B"), **extra):
    d = {"inputs": {"type": "cif", "files": [f"{folder}/{n}.cif" for n in names]}, "label": "e2e",
         "template": "full_electronic", "sequence": ["OPT", "SP", "BAND", "DOSS"],
         "steps": {"SP_2": {"method": {"functional": "PBE0", "dispersion": False}},
                   "BAND_3": {"resources": {"memory": "48G"}}},
         "execution": {}}
    d.update(extra)
    return d


def _task(srv, route, body, timeout=600):
    return srv.task_result(srv.req(f"api/{route}", {**body, "wait": 60}), timeout=timeout)


def _preview(srv, draft, **extra):
    return _task(srv, "workflow/preview", {"root_gen": _gen(srv), "draft": draft, **extra})


def _run(srv, pid, **extra):
    r = srv.req("api/workflow/run", {"root_gen": _gen(srv), "preview_id": pid, **extra})
    if r[0] in (400, 403, 409):
        return r[0], r[2]
    return srv.task_result(r)


def _state_changes(calls):
    return [c for c in calls if c["tool"] in ("sbatch", "scancel")
            or (c["tool"] == "scontrol" and c["argv"][1:2] != ["show"])]


# ---------------------------------------------------------------- read-only routes
def test_templates_and_calc_types(srv):
    st, _, r = srv.req("api/workflow/templates")
    assert st == 200, r
    names = {t["name"]: t for t in r["templates"]}
    assert names["full_electronic"]["sequence"] == ["OPT", "SP", "BAND", "DOSS"]
    assert "custom" not in names and all(t["help"] for t in r["templates"])
    assert r["calc_types"]["BAND"]["kind"] == "d3" and r["calc_types"]["SP"]["kind"] == "d12"
    assert r["calc_types"]["SP"]["help"]


def test_inputs_scan_folder_and_files(srv):
    _cifs(srv, "scan")
    st, _, r = srv.req("api/workflow/inputs", {"targets": ["scan"]})
    assert st == 200, r
    assert r["type"] == "cif" and r["folder_rel"] == "scan"
    assert r["files"] == ["scan/A.cif", "scan/B.cif"]
    assert [m["material_id"] for m in r["materials"]] == ["A", "B"]
    st, _, r = srv.req("api/workflow/inputs", {"targets": ["scan/B.cif"]})
    assert st == 200 and r["files"] == ["scan/B.cif"]
    (srv.root / "scan" / "C.d12").write_text(SU.OPT_D12)
    st, _, r = srv.req("api/workflow/inputs", {"targets": ["scan/A.cif", "scan/C.d12"]})
    assert st == 400 and "CIF" in r["error"]


def test_a_symlink_out_of_the_folder_is_skipped_by_inputs_and_defaults(srv):
    """A folder holding a symlink to a CIF outside the served folder: the
    inputs and defaults routes skip it (never read, never named), the
    folder's other files and a symlink that stays inside still work, and the
    outside file named directly is refused."""
    d = _cifs(srv, "links", names=("A",))
    outside = srv.root.parent / "outside_wf"
    outside.mkdir(exist_ok=True)
    shutil.copy(CIF, outside / "X.cif")
    (d / "X.cif").symlink_to(outside / "X.cif")
    _cifs(srv, "links_src", names=("B",))
    (d / "B.cif").symlink_to(srv.root / "links_src" / "B.cif")
    st, _, r = srv.req("api/workflow/inputs", {"targets": ["links"]})
    assert st == 200, r
    assert [m["material_id"] for m in r["materials"]] == ["A", "B"], r
    assert str(outside) not in json.dumps(r) and "X" not in json.dumps(r["materials"])
    draft = _draft("links", names=("A",), sequence=["OPT", "SP"], template=None, steps={})
    draft["inputs"]["files"] = ["links"]
    st, r = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, r
    assert str(outside) not in json.dumps(r)
    st, _, r = srv.req("api/workflow/inputs", {"targets": ["links/X.cif"]})
    assert st in (400, 403) and str(outside) not in json.dumps(r), r


@H.needs_ase
def test_defaults_show_values_and_origins(srv):
    _cifs(srv, "defs")
    st, r = _task(srv, "workflow/defaults", {"draft": _draft("defs")})
    assert st == 200, r
    steps = r["steps"]
    assert list(steps) == ["OPT_1", "SP_2", "BAND_3", "DOSS_4"]
    assert steps["BAND_3"]["resources"]["memory"] == "48G" and steps["BAND_3"]["origins"]["memory"] == "spec"
    assert steps["DOSS_4"]["origins"]["memory"] != "spec"
    assert steps["SP_2"]["settings_origin"].startswith("spec")
    assert steps["OPT_1"]["settings"]["dft_functional"] == "B3LYP"
    draft = _draft("defs")
    draft["steps"]["OPT_1"] = {"answers": {"functional": "PBE0", "dispersion": "No"}}
    st, r = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, r
    o1 = r["steps"]["OPT_1"]
    assert o1["settings"]["dft_functional"] == "PBE0" and o1["settings"]["use_dispersion"] is False
    assert o1["settings_origin"].startswith("spec") and o1["settings"]["basis_set"] == "POB-TZVP-REV2"
    assert steps["OPT_1"]["parent"] is None and steps["BAND_3"]["parent"] == "SP_2"
    assert r["argv"][:3] == ["mace", "workflow", "--spec"]
    # the check wrote nothing in the folder
    assert sorted(p.name for p in (srv.root / "defs").iterdir()) == ["A.cif", "B.cif"]


@H.needs_ase
def test_step_forms_reuse_calculation_setup(srv):
    _cifs(srv, "forms")
    draft = _draft("forms")
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {"functional": "PBE0"}})
    assert st == 200, r
    assert r["mode"] == "opt2d12" and r["parent"] == "OPT_1" and set(r["editable"]) == {"functional", "dispersion"}
    form = {f["id"]: f for f in r["form"]}
    assert form["functional"]["value"] == "PBE0" and form["functional"]["changed"]
    assert {"PBE0", "HSE06", "B3LYP"} <= {o["head"] for o in form["functional"]["options"]}
    assert form["dft_grid"]["changed"] is False
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "BAND_3"})
    assert st == 200, r
    assert r["mode"] == "opt2d3" and r["parent"] == "SP_2" and r["json_text"] and r["deck"]
    # a CIF start's first step: the workflow's own conversion settings, not mace convert's defaults
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT_1"})
    assert st == 200 and r["mode"] == "cif2d12" and r["deck"], r
    form = {f["id"]: f for f in r["form"]}
    assert form["functional"]["value"] == "B3LYP" and r["base_values"]["functional"] == "B3LYP"
    assert "B3LYP-D3" in r["deck"] and r["overrides"] == {}
    assert "functional" in r["editable"] and "calc_type" not in r["editable"]
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT_1", "answers": {"functional": "PBE0"}})
    assert st == 200, r
    assert r["overrides"]["dft_functional"] == "PBE0" and "PBE0-D3" in r["deck"]
    assert r["base_values"]["functional"] == "B3LYP" and {f["id"]: f for f in r["form"]}["functional"]["value"] == "PBE0"
    # the SP form is built from the first step's deck as changed
    draft2 = dict(draft, steps={**draft["steps"], "OPT_1": {"answers": {"functional": "PBE0"}}})
    st, r = _task(srv, "workflow/step-form", {"draft": draft2, "step": "SP_2"})
    assert st == 200 and {f["id"]: f for f in r["form"]}["functional"]["value"] == "PBE0", r.get("error")
    # TRANSPORT and CHARGE+POTENTIAL: opt2d3's form, its own questions, on the stand-in SP
    tr = dict(draft, template="transport_analysis", sequence=["OPT", "SP", "TRANSPORT"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": tr, "step": "TRANSPORT_3"})
    assert st == 200 and r["mode"] == "opt2d3" and r["parent"] == "SP_2" and r["standin"], r
    rows = {f["label"]: f for f in r["form"]}
    assert rows["Minimum temperature (K) [100]"]["value"] == "100" and "BOLTZTRA" in r["deck"]
    tmin = rows["Minimum temperature (K) [100]"]["id"]
    st, r = _task(srv, "workflow/step-form", {"draft": tr, "step": "TRANSPORT_3", "answers": {tmin: "200"}})
    assert st == 200 and json.loads(r["json_text"])["configuration"]["temperature_range"][0] == 200.0, r
    cp = dict(draft, template=None, sequence=["OPT", "SP", "CHARGE+POTENTIAL"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": cp, "step": "CHARGE+POTENTIAL_3"})
    assert st == 200 and r["mode"] == "opt2d3" and "ECH3" in r["deck"] and "POT3" in r["deck"], r
    assert len([f for f in r["form"] if f["id"].endswith("#2")]) >= 2
    # a later d12 step inherits its parent's settings: only the method can change
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {"tolerance": "Tight"}})
    assert st == 400 and "inherit" in r["error"]


# ---------------------------------------------------------------- preview == run
@H.needs_ase
def test_preview_equals_run(srv):
    _cifs(srv, "pr")
    srv.clear_calls()
    st, prev = _preview(srv, _draft("pr"))
    assert st == 200 and prev["code"] == 0, prev.get("cli_log")
    # the preview reached SLURM only with read-only queries
    assert not _state_changes(srv.calls())
    assert not (srv.root / "pr" / "workflow_configs").exists()
    assert [f["material_id"] for f in prev["first_step"]] == ["A", "B"]
    assert all("OPT" in f["deck_name"] or f["deck_text"] for f in prev["first_step"])
    proj = {p["step"]: p for p in prev["projected"]}
    assert sorted(proj) == ["BAND_3", "DOSS_4", "SP_2"]
    assert "PBE0" in proj["SP_2"]["deck_text"] and proj["SP_2"]["projected"] is True
    assert "#SBATCH --mem=48G" in proj["BAND_3"]["script_text"]
    assert prev["commands"][0].startswith("cd pr && mace workflow --spec workflow_configs/gui_specs/")
    assert prev["commands"][1].startswith("mace workflow --execute ")
    assert prev["plan"]["text"] and prev["spec"]["text"]
    pid = prev["preview_id"]

    srv.clear_calls()
    st, res = _run(srv, pid)
    assert st == 200 and res["code"] == 0, res
    assert res["script_drift"] == [], res["script_drift"]
    real = _state_changes(srv.calls())
    assert sorted(norm(c["argv"], srv.root) for c in real) == \
        sorted(norm(a, srv.root) for a in prev["intercepted"])
    assert len(real) == 2
    assert len(res["job_ids"]) == 2
    wid = res["workflow_id"]
    assert prev["scripts"]
    for rel, text in prev["scripts"].items():
        real_rel = rel.replace(prev["workflow_id_preview"], wid)
        assert norm(text, srv.root) == norm(CKSUM.sub("cksum of the lines above: <CKSUM>",
                                                      (srv.root / real_rel).read_text()), srv.root), rel
    plan = json.loads((srv.root / res["plan_rel"]).read_text())
    assert plan["workflow_id"] == wid and plan["label"] == "e2e"
    st, _, lst = srv.req("api/workflow/list")
    assert any(w["workflow_id"] == wid and w["folder_rel"] == "pr" for w in lst["workflows"])
    # the spec the GUI wrote is where the commands said
    assert (srv.root / "pr" / prev["spec"]["rel"].split("pr/", 1)[1]).is_file()

    # single use
    st, again = _run(srv, pid)
    assert st == 409 and "already used" in again["error"]
    # the same structures again while their first steps are queued: refused
    st, prev2 = _preview(srv, _draft("pr"))
    assert st == 200 and prev2["blocked"], prev2
    assert "A" in prev2["blocked"] and prev2["code"] == 0
    st, res2 = _run(srv, prev2["preview_id"])
    assert st == 409 and "unfinished" in res2["error"]
    assert len(_state_changes(srv.calls())) == 2


@H.needs_ase
def test_root_gen_changed_is_409(srv):
    _cifs(srv, "gen")
    st, _, r = srv.req("api/workflow/preview", {"root_gen": int(_gen(srv)) + 7, "draft": _draft("gen")})
    assert st == 409 and "folder changed" in r["error"]
    st, _, r = srv.req("api/workflow/preview", {"draft": _draft("gen")})
    assert st == 400 and "root_gen" in r["error"]
    st, prev = _preview(srv, _draft("gen"))
    assert st == 200 and prev["code"] == 0, prev.get("cli_log")
    srv.req("api/test/root_gen", {})
    st, res = _run(srv, prev["preview_id"])
    assert st == 409 and "folder changed" in res["error"]
    assert not (srv.root / "gen" / "workflow_configs").exists()


@H.needs_ase
def test_inputs_changed_since_preview_is_409(srv):
    d = _cifs(srv, "chg")
    st, prev = _preview(srv, _draft("chg"))
    assert st == 200 and prev["code"] == 0
    (d / "A.cif").write_text((d / "A.cif").read_text() + "\n")
    st, res = _run(srv, prev["preview_id"])
    assert st == 409 and "changed since the preview" in res["error"]


def test_refused_on_network_bind(tmp_path):
    import re
    import subprocess
    import sys
    if not CIF.exists():
        pytest.skip("test/ data corpus not present")
    s = SU.Served.__new__(SU.Served)
    s.tmp = tmp_path
    s.root = tmp_path / "proj"
    s.root.mkdir()
    s.fake = SU.FakeSlurm(tmp_path / "fs")
    s.env = SU.fake_env(s.fake, tmp_path)
    s.env.pop("XDG_CACHE_HOME")
    s.env["MACE_SITE_CONFIG"] = "none"
    port = SU.H.free_port()
    env = SU.H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", **s.env)
    proc = SU.H.ServerProcess([sys.executable, str(SU.MACE_CLI), "--no-banner", "gui", "--dir", str(s.root), "--port",
                               str(port), "--token", SU.H.TOKEN, "--host", "0.0.0.0", "--insecure-network",
                               "--allow-host", "127.0.0.1"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    try:
        base = None
        for _ in range(60):
            m = re.search(r"(http://[^/\s]+:\d+/s/[A-Za-z0-9_-]+/)", proc.stdout.readline())
            if m:
                base = m.group(1).replace("0.0.0.0", "127.0.0.1")
                break
        assert base
        s.base, s.proc = base, proc
        _cifs(s, "net")
        st, prev = _preview(s, _draft("net"))
        assert st == 200 and "Submitting is turned off while MACE GUI is open to the network" in prev["warnings"]
        st, res = _run(s, prev["preview_id"])
        assert st == 409 and "open to the network" in res["error"]
        assert not [c for c in s.fake.calls() if c["tool"] == "sbatch"]
        assert not (s.root / "net" / "workflow_configs").exists()
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_d12_start_preview_equals_run(srv):
    d = srv.root / "decks12"
    d.mkdir()
    for n in ("m0", "m1"):
        (d / f"{n}.d12").write_text(SU.OPT_D12)
    draft = {"inputs": {"type": "d12", "files": ["decks12/m0.d12", "decks12/m1.d12"]}, "template": "opt_sp",
             "sequence": ["OPT", "SP"], "steps": {"OPT_1": {"resources": {"ntasks": 12}}},
             "execution": {"job_arrays": {"first_step": True, "max_running": 2}}}
    srv.clear_calls()
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, prev.get("cli_log")
    assert prev["commands"][1].startswith("mace submit --progress-plan ")
    assert not _state_changes(srv.calls())
    st, res = _run(srv, prev["preview_id"])
    assert st == 200 and res["code"] == 0 and res["script_drift"] == [], res
    (call,) = [c for c in srv.calls("sbatch")]
    assert "%2" in json.dumps(call)
    assert "--ntasks=12" in "".join(prev["scripts"].values())
    db = d / f".mace_context_{res['workflow_id']}" / "materials.db"
    con = sqlite3.connect(db)
    try:
        n = con.execute("SELECT count(*) FROM calculations").fetchone()[0]
    finally:
        con.close()
    assert n == 2
