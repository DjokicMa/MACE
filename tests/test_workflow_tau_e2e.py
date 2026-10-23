"""The TAU step end to end, through the real CLI and job callback with fake
SLURM: spec (TRANSPORT relaxation_time auto) -> plan with a TAU step -> SP ->
BAND, DOSS and the strained single points -> their completions in any order
-> the fit (masses from the workflow's own BAND and DOSS) -> TRANSPORT with
RELAXTIM = tau. The SP, BAND and DOSS outputs are the corpus diamond's
(test/), the strained outputs the HPCC runs of the same structure
(tests/data/hpcc_116/strain), standing in for the members' outputs."""
import json
import re
import shutil
from pathlib import Path

import pytest

from array_helpers import callback, rows
from conftest import REPO_ROOT, find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace

STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"
SP = "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"
DIA_BAND = "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
DIA_DOSS = "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss"


def _ok(p):
    assert p.returncode == 0, p.stdout[-6000:] + p.stderr[-4000:]
    return p


def _exports(sh):
    return dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)="([^"]*)"',
                           Path(sh).read_text(), re.M))


def _finish(env, fake, db, row, out_src, extra=(), state="COMPLETED"):
    """The job of ``row`` ends: its output (and files) beside its deck, then its callback."""
    deck = Path(row["input_file"])
    if out_src is not None:
        shutil.copy(out_src, deck.with_suffix(".out"))
    for src, suffix in extra:
        shutil.copy(src, deck.with_name(deck.stem + suffix))
    fake.set_record(row["slurm_job_id"], state=state)
    return _ok(callback(env, deck.parent, db, mode="completion",
                        extra_env=dict(_exports(deck.with_suffix(".sh")), SLURM_JOB_ID=row["slurm_job_id"])))


def _of(db, calc_type):
    return [r for r in rows(db).values() if r["calc_type"] == calc_type]


def _member(row):
    return json.loads(row["settings_json"])["tau_member"]


def _strained_out(member):
    """The HPCC output of a member (x_m1pct -> dia_sp_dp_x_m1pct.out)."""
    return STRAIN / f"dia_sp_dp_{member}.out"


@pytest.fixture
def launched(tmp_path):
    sp_deck = find_data(SP + ".d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "wf"
    root.mkdir()
    shutil.copy(sp_deck, root / sp_deck.name)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "label": "tau", "inputs": {"type": "d12", "files": [sp_deck.name]},
        "sequence": ["SP", "BAND", "DOSS", "TRANSPORT"],
        "steps": {"SP_1": {"resources": {"walltime": "2:00:00"}},
                  "TRANSPORT_4": {"overrides": {"relaxation_time": "auto"}}}}))
    w = json.loads(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    plan = json.loads(Path(w["plan_file"]).read_text())
    _ok(_mace(root, env, *w["launch"][1:]))
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    return {"root": root, "env": env, "fake": fake, "db": db, "wid": w["workflow_id"], "plan": plan}


def _sp_done(L):
    (sp,) = _of(L["db"], "SP")
    sp_out = find_data(SP + ".out")
    deck = Path(sp["input_file"])
    shutil.copy(sp_out.with_suffix(".f9"), deck.with_suffix(".f9"))
    return _finish(L["env"], L["fake"], L["db"], sp, sp_out)


def test_plan_has_the_tau_step_with_the_sp_resources(launched):
    plan = launched["plan"]
    assert plan["workflow_sequence"] == ["SP", "BAND", "DOSS", "TAU", "TRANSPORT"]
    tau = plan["step_configurations"]["TAU_4"]
    assert tau["tau"]["axes"] == "auto" and tau["tau"]["strains"] == [-0.01, -0.005, 0.005, 0.01]
    assert tau["slurm_config"]["resources"]["walltime"] == "3:00:00"          # the SP's 2:00:00 x 1.5
    script = launched["root"] / f".mace_context_{launched['wid']}" / "workflow_scripts" / "submitcrystal23_tau_4.sh"
    assert "#SBATCH -t 3:00:00" in script.read_text()


def test_tau_group_fit_and_transport_relaxtim(launched):
    L = launched
    db, env, fake = L["db"], L["env"], L["fake"]
    p = _sp_done(L)
    members = _of(db, "TAU")
    assert len(members) == 5, p.stdout[-6000:]                      # cubic: x only, 4 strains + unstrained
    assert sorted(_member(r) for r in members) == ["x_m0_5pct", "x_m1pct", "x_p0_5pct", "x_p0pct", "x_p1pct"]
    folder = Path(members[0]["work_dir"])
    assert folder.parent.name == "step_004_TAU" and all(Path(r["work_dir"]) == folder for r in members)
    manifest = json.loads((folder / "tau_group.json").read_text())
    assert manifest["axes"] == ["x"] and "cubic (space group 227)" in manifest["axes_reason"]
    m1 = folder / next(m["deck"] for m in manifest["members"] if m["member"] == "x_m1pct")
    assert "ELASTIC\n-2\n-0.010000 0.0 0.0" in m1.read_text()
    assert "#SBATCH -t 3:00:00" in m1.with_suffix(".sh").read_text()
    assert _of(db, "BAND") and _of(db, "DOSS") and not _of(db, "TRANSPORT")

    # the members end in any order; BAND and DOSS are still running, so the fit waits for the masses
    order = ["x_p1pct", "x_m0_5pct", "x_p0pct", "x_m1pct", "x_p0_5pct"]
    by = {_member(r): r for r in members}
    for i, m in enumerate(order):
        p = _finish(env, fake, db, by[m], _strained_out(m))
        assert not _of(db, "TRANSPORT")
        assert not (folder / "tau_summary.json").exists(), p.stdout[-4000:]
    assert "waiting for BAND and DOSS" in p.stdout
    st = json.loads(_ok(_mace(L["root"], env, "workflow", "--status", L["wid"], "--json")).stdout)
    tau_node = st["materials"][0]["nodes"][3]
    assert tau_node["group"]["done"] == 5 and tau_node["group"]["total"] == 5 and tau_node["state"] == "waiting"
    assert st["materials"][0]["nodes"][4]["state"] == "waiting"

    (band,) = _of(db, "BAND")
    bsrc = find_data(DIA_BAND + ".BAND.DAT")
    _finish(env, fake, db, band, bsrc.with_name(bsrc.name.replace(".BAND.DAT", ".out")),
            extra=[(bsrc, ".BAND.DAT")])
    assert not _of(db, "TRANSPORT")
    (doss,) = _of(db, "DOSS")
    dsrc = find_data(DIA_DOSS + ".DOSS.DAT")
    p = _finish(env, fake, db, doss, dsrc.with_name(dsrc.name.replace(".DOSS.DAT", ".out")),
                extra=[(dsrc, ".DOSS.DAT")])

    summary = json.loads((folder / "tau_summary.json").read_text())
    assert summary["fallback"] is False and summary["carrier"] == "electron" and summary["axis"] == "x"
    assert "MACE has no doping" in summary["carrier_reason"]
    assert summary["low_confidence"] is True              # diamond's CBM along x: R^2 0.54 (valleys split)
    el = summary["results"]["electron"]["x"]
    assert "BAND.DAT" in el["m_c_source"] and el["C_unit"] == "GPa"
    assert (folder / "tau_electron_x.json").is_file() and (folder / "tau_hole_x.json").is_file()
    # the same numbers as `mace tau fit` (same outputs, the same BAND and DOSS files)
    from mace.transport import strain
    from mace.transport.cli import resolve_masses
    from mace.transport import dpt
    pts = [strain.read_point(Path(r["input_file"]).with_suffix(".out")) for r in _of(db, "TAU")]
    ft = strain.fit(pts, "electron")
    band_dat = Path(band["input_file"]).with_name(Path(band["input_file"]).stem + ".BAND.DAT")
    doss_dat = Path(doss["input_file"]).with_name(Path(doss["input_file"]).stem + ".DOSS.DAT")
    m, md, _, _ = resolve_masses("auto", "electron", "x", 3, band=band_dat, dos=doss_dat)
    tau = dpt.estimate(3, m, ft["E1_eV"], ft["C"], 300.0, None, carrier="electron", axis="x", m_dos=md)["tau_fs"]
    assert summary["computed_fs"] == pytest.approx(tau, rel=1e-9)

    (tr,) = _of(db, "TRANSPORT")
    d3 = Path(tr["input_file"]).read_text()
    assert f"RELAXTIM\n{summary['used_fs']}" in d3, d3
    assert summary["used_fs"] == float(f"{tau:.4g}")
    rt = json.loads(tr["settings_json"])["relaxation_time"]
    assert rt["relaxation_time_fs"] == summary["used_fs"] and rt["low_confidence"] and not rt["fallback"]
    assert rt["json"].endswith("tau_electron_x.json")

    st = json.loads(_ok(_mace(L["root"], env, "workflow", "--status", L["wid"], "--json")).stdout)
    nodes = st["materials"][0]["nodes"]
    assert nodes[3]["state"] == "completed" and nodes[3]["tau"]["used_fs"] == summary["used_fs"]
    assert nodes[4]["relaxation_time"]["relaxation_time_fs"] == summary["used_fs"]
    text = _ok(_mace(L["root"], env, "workflow", "--status", L["wid"])).stdout
    assert "Relaxation time:" in text and "RELAXTIM" in text and "low confidence" in text


def test_a_failed_member_is_recovered_and_the_last_failure_still_fits(launched):
    """A member that fails recoverably is resubmitted as the same member; one
    that fails for good is left out, and the fit (with the rest) runs from
    that failure's own callback."""
    L = launched
    db, env, fake = L["db"], L["env"], L["fake"]
    _sp_done(L)
    by = {_member(r): r for r in _of(db, "TAU")}
    for kind, suffix in (("BAND", ".BAND.DAT"), ("DOSS", ".DOSS.DAT")):
        (row,) = _of(db, kind)
        src = find_data((DIA_BAND if kind == "BAND" else DIA_DOSS) + suffix)
        _finish(env, fake, db, row, src.with_name(src.name.replace(suffix, ".out")), extra=[(src, suffix)])
    for m in ("x_p0pct", "x_m1pct", "x_p1pct"):
        _finish(env, fake, db, by[m], _strained_out(m))
    # x_m0_5pct: SCF did not converge (a recoverable error): resubmitted as the same member
    bad = _strained_out("x_m0_5pct").read_text()
    cut = bad.index(" == SCF ENDED")
    broken = bad[:cut] + " CYCLE  80 ETOT  -7.6E+01\n TOO MANY CYCLES\n"
    tmp = Path(by["x_m0_5pct"]["work_dir"]) / "broken.txt"
    tmp.write_text(broken)
    _finish(env, fake, db, by["x_m0_5pct"], tmp, state="FAILED")
    again = [r for r in _of(db, "TAU") if _member(r) == "x_m0_5pct" and r["calc_id"] != by["x_m0_5pct"]["calc_id"]]
    folder = Path(by["x_p0pct"]["work_dir"])
    assert again, "the failed member was not recovered"
    assert rows(db)[by["x_m0_5pct"]["calc_id"]]["status"] == "resubmitted"
    assert Path(again[-1]["work_dir"]) == folder                    # its own deck name, in the step's folder
    assert Path(again[-1]["input_file"]).name.startswith("1_dia_sp_dp_x_m0_5pct_recovery_")
    assert not (folder / "tau_summary.json").exists()
    _finish(env, fake, db, again[-1], _strained_out("x_m0_5pct"))
    # x_p0_5pct fails for good (no output at all): the group finishes with 4 of 5
    p = _finish(env, fake, db, by["x_p0_5pct"], None, state="FAILED")
    summary = json.loads((folder / "tau_summary.json").read_text())
    assert summary["fallback"] is False, p.stdout[-4000:]
    assert summary["members"]["failed"] == ["x_p0_5pct"] and summary["members"]["completed"] == 4
    assert any("1 of 5 strained runs failed" in w for w in summary["warnings"])
    (tr,) = _of(db, "TRANSPORT")
    assert f"RELAXTIM\n{summary['used_fs']}" in Path(tr["input_file"]).read_text()
