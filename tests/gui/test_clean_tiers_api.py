"""Clean's tiers through the real `mace gui` (plan: Clean tiers, T7):
springs unchanged (A1), the methods route and its hints (A2), the charged
and geometry-edited deck guards with the CLI's words (A8, A13b), the fake ML
worker (one model load, cancel with grace and kill, nice level: A9, A12,
A21), provenance and the FF/ML marks on Accept, Save CIF and Setup (A17),
and the weighted CPU slots (A21)."""
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402

CHARGED = F.REPO / "tests" / "preopt"


def _deck(edits="", scf=""):
    import test_preopt_cli as T   # the synthetic MgO deck of the CLI tests
    return T.MGO_DECK.format(edits=edits, basis_extra="", scf_extra=scf)


def _fake_xtb(d: Path, version: str):
    d.mkdir(parents=True, exist_ok=True)
    p = d / "xtb"
    p.write_text(f'#!/bin/sh\necho "      * xtb version {version} (fake) compiled by x on 2024"\n')
    p.chmod(0o755)
    return p


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cleantiers")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    (root / "decks").mkdir()
    (root / "decks" / "charged.d12").write_text(_deck(scf="CHARGED\n"))
    (root / "decks" / "defect.d12").write_text(_deck(edits="SUPERCEL\n2 0 0\n0 2 0\n0 0 2\nATOMREMO\n1\n1\n"))
    xtbdir = tmp / "fakebin"
    _fake_xtb(xtbdir, "6.6.1")
    py = F.make_fake_python(tmp / "venv")
    ptmp = tmp / "ptmp"
    ptmp.mkdir()
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_MAX_WORKERS="2", MACE_MLIP_PYTHON=str(py),
                                   FAKE_HELLO_LOG=str(tmp / "hello.log"), FAKE_MODE_FILE=str(tmp / "mode"),
                                   PATH=f"{xtbdir}:/usr/bin:/bin", TMPDIR=str(ptmp),
                                   XDG_DATA_HOME=str(tmp / "data"))
    end = time.time() + 60
    while time.time() < end and U.req(base, "api/clean/methods")[2].get("checking"):
        time.sleep(0.2)
    yield {"base": base, "root": root, "tmp": tmp, "proc": proc, "xtbdir": xtbdir, "ptmp": ptmp}
    proc.terminate()
    proc.wait(timeout=20)
    left = list((ptmp / f"mace-preopt-{os.getuid()}").glob("*")) if (ptmp / f"mace-preopt-{os.getuid()}").exists() else []
    assert not left, left


def session_of(base, atoms, name, **kw):
    st, _, s = U.post(base, "api/edit/from_structure", {"name": name, "cell": atoms.cell.array.tolist(),
                                                       "frac": atoms.get_scaled_positions(wrap=False).tolist(),
                                                       "numbers": atoms.numbers.tolist(), **kw})
    assert st == 200, s
    return s


def result_of(base, st, r, timeout=120):
    if st == 200:
        return r
    assert st == 202, (st, r)
    tid = r["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        st2, _, r2 = U.req(base, f"api/tasks/{tid}/result")
        if st2 == 200:
            return r2
        assert st2 == 409, (st2, r2)
        if r2.get("task", {}).get("state") in ("failed", "canceled"):
            raise AssertionError(r2)
        time.sleep(0.1)
    raise AssertionError("the clean did not finish")


def clean(base, sid, rev, **kw):
    st, _, r = U.post(base, "api/clean", {"session": sid, "base_rev": rev, **kw})
    if st >= 400:
        return st, r
    return 200, result_of(base, st, r)


def hellos(tmp):
    p = tmp / "hello.log"
    return len(p.read_text().splitlines()) if p.exists() else 0


# --- A1: springs exactly as before ------------------------------------------------------

@pytest.mark.parametrize("name", ["MgO_rattled", "FSI_box"])
def test_springs_route_is_unchanged(served, name):
    from mace.gui import clean as C
    base = served["base"]
    atoms = F.t0_structures()[name]
    s = session_of(base, atoms, f"a1_{name}")
    for limit in (60, 120):                                   # a non-default limit, Run Longer's doubled one
        st, r = clean(base, s["session"], s["rev"], time_limit=limit)
        assert st == 200 and r["report"]["converged"] and r["report"]["steps"] == F.T0[name][0]
        assert "method_ran" not in r and r["summary"].startswith("max shift ")
        ref = C.clean(atoms, time_limit=60)["atoms"]
        cart = np.asarray(r["displacements"]) + np.asarray(r["before_cart"])
        d = np.linalg.solve(ref.cell.array.T, (cart - ref.positions).T).T
        d -= np.round(d)                                      # the same atom, maybe one cell over
        assert np.abs(d @ ref.cell.array).max() < 1e-8
    st, r = clean(base, s["session"], s["rev"], time_limit=500)
    assert r["time_limit"] == 120                             # the springs clamp is still 120 s


def test_metal_nothing_to_clean_only_for_springs(served):
    base = served["base"]
    s = session_of(base, F.copper(), "cu_t")
    st, r = clean(base, s["session"], s["rev"])
    assert r["nothing_to_clean"]
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)
    assert st == 200 and r["method_ran"] == "mlip:chgnet-0.3.0", r      # all-metal Cu reaches the ML tier


# --- A2: methods, hints, recheck -----------------------------------------------------------

def test_methods_route_and_recheck(served):
    base = served["base"]
    st, _, r = U.req(base, "api/clean/methods")
    assert st == 200 and not r["checking"]
    m = {x["id"]: x for x in r["methods"]}
    assert list(m) == ["springs", "gfnff", "mlip:acesuit-mace-mpa-0-medium", "mlip:chgnet-0.3.0", "auto"]
    assert m["springs"]["available"] and m["springs"]["max_time_limit"] == 120
    g = m["gfnff"]
    assert not g["available"] and g["status"] == "too_old"
    assert g["reason"] == "Found xtb 6.6.1; GFN-FF for periodic cells needs 6.7+"
    assert 'conda install -c conda-forge "xtb>=6.7"' in g["hint"] and g["max_time_limit"] == 600
    mpa = m["mlip:acesuit-mace-mpa-0-medium"]
    assert mpa["label"] == "ML potential: MACE-MPA-0 (ACEsuit mace-torch)" and not mpa["available"]
    assert mpa["status"] == "needs_fetch" and mpa["hint"] == "mace mlip fetch acesuit-mace-mpa-0-medium"
    # each command of a hint on its own, so the dialog copies a runnable line, not the prose around it
    assert g["commands"] == ['conda install -c conda-forge "xtb>=6.7"', "export MACE_XTB=/path/to/xtb"]
    assert mpa["commands"] == ["mace mlip fetch acesuit-mace-mpa-0-medium"]
    assert "unrelated to, this MACE tool" in mpa["unrelated_note"]
    assert m["mlip:chgnet-0.3.0"]["available"] and m["mlip:chgnet-0.3.0"]["training_data"]
    s = session_of(base, F.wurtzite(), "zno_g")
    st, r = clean(base, s["session"], s["rev"], method="gfnff")
    assert st == 400 and "needs 6.7+" in r["error"] and r["hint"]
    try:
        _fake_xtb(served["xtbdir"], "6.7.1")
        st, _, r = U.post(base, "api/clean/methods/recheck", {})
        g = {x["id"]: x for x in r["methods"]}["gfnff"]
        assert g["available"] and g["label"] == "GFN-FF (xtb 6.7.1)"
    finally:
        _fake_xtb(served["xtbdir"], "6.6.1")
        U.post(base, "api/clean/methods/recheck", {})


def test_no_asl_or_meta_method_ids():
    import re
    from mace.gui import server as S
    words = re.compile(r"\b(omat|fairchem|uma|grace|asl)\b", re.I)
    for mid in S._clean_method_ids():
        assert not words.search(mid.replace("-", " ").replace(":", " ")), mid


# --- A8, A13b: decks, charge, refusals, GUI == CLI ------------------------------------------

def test_charged_deck_needs_a_charge_like_the_cli(served):
    base = served["base"]
    st, _, s = U.post(base, "api/edit/open", {"path": "decks/charged.d12"})
    assert st == 200, s
    st, _, state = U.req(base, f"api/edit/state?session={s['session']}")
    assert state["doc_meta"]["deck"]["charge_records"] == ["CHARGED"]
    for method in ("gfnff", "mlip:chgnet-0.3.0", "auto"):
        st, r = clean(base, s["session"], s["rev"], method=method)
        assert st == 400 and r["error"] == "this deck is charged (CHARGED found): enter the charge", (method, r)
    st, r = clean(base, s["session"], s["rev"])
    assert st == 200 and "proposal_id" in r                      # springs never need it
    for bad in (1.5, True, "1.5", "one"):                          # never truncated to a whole number
        st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", charge=bad)
        assert st == 400 and r["error"] == "charge and uhf are whole numbers", (bad, r)
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", charge=1, time_limit=60)
    assert st == 200 and r["report"]["charge"] == 1
    assert r["warnings"].count("ML potentials take no charge input; they see a neutral cell") == 1


def test_geometry_edited_deck_refused_with_the_cli_text(served, tmp_path):
    from mace.preopt import deckscan
    base = served["base"]
    st, _, s = U.post(base, "api/edit/open", {"path": "decks/defect.d12"})
    assert st == 200, s
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", charge=0)
    assert st == 400 and r["error"] == deckscan.refusal_text("defect.d12", ["SUPERCEL", "ATOMREMO"])
    rc = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "preopt",
                         str(served["root"] / "decks" / "defect.d12"), "--method", "auto", "-o", str(tmp_path)],
                        env={k: v for k, v in os.environ.items() if not k.startswith("PYTHON")},
                        cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert rc.returncode == 1 and rc.stderr.strip() == r["error"]
    st, r = clean(base, s["session"], s["rev"])
    assert st == 200 and "proposal_id" in r


def test_gui_and_cli_give_the_same_coordinates(served, tmp_path):
    """A13(b): same input and options, springs and the fake ML worker."""
    import ase.io
    base = served["base"]
    a = F.wurtzite()
    ase.io.write(str(tmp_path / "zno.cif"), a, format="cif")
    a = ase.io.read(str(tmp_path / "zno.cif"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["MACE_MLIP_PYTHON"] = str(served["tmp"] / "venv" / "bin" / "python")
    for method, extra in (("springs", ["--time-limit", "60"]), ("mlip:chgnet-0.3.0", ["--time-limit", "60"])):
        out = tmp_path / method.replace(":", "_")
        r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "preopt", str(tmp_path / "zno.cif"),
                            "--method", method, "-o", str(out), *extra], env=env, cwd=str(tmp_path),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        cli = ase.io.read(str(out / "zno_preopt.cif"))
        s = session_of(base, a, f"x_{method.replace(':', '_')}")
        st, g = clean(base, s["session"], s["rev"], method=method, time_limit=60)
        assert st == 200 and g["report"]["converged"]
        gui = np.asarray(g["displacements"]) + np.asarray(g["before_cart"])
        fa = cli.get_scaled_positions()
        fb = np.linalg.solve(a.cell.array.T, gui.T).T % 1.0
        d = (fa - fb + 0.5) % 1.0 - 0.5
        assert np.abs(d @ a.cell.array).max() < 1e-8, method


# --- A12, A9, A21: the server's one ML worker -----------------------------------------------

def test_three_ml_cleans_one_model_load_and_nice_workers(served):
    base, tmp = served["base"], served["tmp"]
    n0 = hellos(tmp)
    s = session_of(base, F.wurtzite(), "ml3")
    for _ in range(3):
        st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)
        assert st == 200 and r["method_ran"] == "mlip:chgnet-0.3.0", r
        assert r["ran"].startswith("Ran: ML potential: CHGNet 0.3.0, converged in")
    assert hellos(tmp) - n0 <= 1
    rep = U.req(base, "api/test/mlip")[2]
    assert rep["running"] and rep["owner"] is None
    with open(f"/proc/{rep['pid']}/stat") as fh:
        nice = int(fh.read().rsplit(")", 1)[1].split()[16])
    assert nice >= 10
    assert r["report"]["worker_nice"] >= 10


def test_cancel_a_hung_ml_call_kills_the_worker_and_the_next_job_reloads(served):
    base, tmp = served["base"], served["tmp"]
    s = session_of(base, F.wurtzite(), "mlhang")
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)   # a loaded worker
    rep0 = U.req(base, "api/test/mlip")[2]
    n0 = hellos(tmp)
    (tmp / "mode").write_text("hang")
    try:
        st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"],
                                              "method": "mlip:chgnet-0.3.0", "time_limit": 60})
        assert st == 202, r
        tid = r["task"]["id"]
        time.sleep(3.0)                                      # the springs baseline, then the hung call
        U.post(base, f"api/tasks/{tid}/cancel", {})
        end = time.time() + 20
        while time.time() < end and U.req(base, "api/test/mlip")[2]["kills"] == rep0["kills"]:
            time.sleep(0.2)
        rep = U.req(base, "api/test/mlip")[2]
        assert rep["kills"] == rep0["kills"] + 1 and not rep["running"]
        assert not os.path.exists(f"/proc/{rep0['pid']}")      # reaped, not a zombie
    finally:
        (tmp / "mode").write_text("")
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)
    assert st == 200 and r["method_ran"] == "mlip:chgnet-0.3.0"
    assert hellos(tmp) == n0 + 1                             # the model was loaded again


def test_ml_crash_keeps_the_springs_with_the_reason(served):
    base, tmp = served["base"], served["tmp"]
    s = session_of(base, F.wurtzite(), "mlcrash")
    (tmp / "mode").write_text("crash")
    try:
        st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)
    finally:
        (tmp / "mode").write_text("")
    assert st == 200 and r["method_ran"] == "springs" and r["fell_back"]
    assert r["attempts"][-1]["status"] == "failed" and r["attempts"][-1]["reason"].startswith("crash:")
    assert r["ran"].startswith("Ran: ML chgnet-0.3.0 failed (crash:")


# --- A17: Accept, Save CIF, Setup ------------------------------------------------------------

def test_accept_records_provenance_and_marks_the_document(served):
    base, root = served["base"], served["root"]
    s = session_of(base, F.wurtzite(), "zno_acc", locked=False)
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=60)
    assert r["citation"].startswith("B. Deng et al.") and "Training data: MPtrj" in r["citation"]
    assert r["starting_geometry_note"].startswith("Starting geometry only")
    st, _, a = U.post(base, "api/edit/clean_accept", {"session": s["session"], "base_rev": s["rev"],
                                                      "proposal_id": r["proposal_id"]})
    assert st == 200 and a["history"][-1]["label"] == "Clean Geometry"
    meta = U.req(base, f"api/edit/state?session={s['session']}")[2]["doc_meta"]
    p = meta["provenance"]["params"]
    assert meta["preopt"] == {"method_ran": "mlip:chgnet-0.3.0", "model": "chgnet-0.3.0"}
    assert p["model"] == "chgnet-0.3.0" and p["code_license"] == "BSD-3-Clause-LBNL" and p["training_data"]
    assert p["citation"] and p["charge"] == 0
    st, _, sv = U.post(base, "api/structure/save", {"session": s["session"], "path": "zno_acc.cif", "format": "cif"})
    assert st == 200, sv
    text = (root / "zno_acc.cif").read_text()
    assert text.startswith("# mace-preopt: method=mlip:chgnet-0.3.0 model=chgnet-0.3.0 version=1\n")
    st, _, o = U.post(base, "api/edit/open", {"path": "zno_acc.cif"})
    st, _, state = U.req(base, f"api/edit/state?session={o['session']}")
    assert state["doc_meta"]["preopt"]["method_ran"] == "mlip:chgnet-0.3.0"
    # a springs-only document saves exactly as before: no marker
    s2 = session_of(base, F.wurtzite(), "zno_plain")
    U.post(base, "api/structure/save", {"session": s2["session"], "path": "zno_plain.cif", "format": "cif"})
    assert "mace-preopt" not in (root / "zno_plain.cif").read_text()


def test_setup_warns_for_a_preopt_geometry_unless_opt(served):
    from mace.gui import server as S

    class St:
        pass
    st = St()
    st.sessions = type("SS", (), {"get": staticmethod(lambda sid: type("Sess", (), {
        "doc_meta": {"preopt": {"method_ran": "gfnff", "model": None}}})())})()
    body = {"source": {"session": "x"}}
    for calc in ("SP", "FREQ", "BAND", "DOSS"):
        w = S._setup_preopt_warning(st, body, None, calc)
        assert w.startswith("This geometry is an FF/ML pre-optimization (gfnff). Build an OPT first;"), w
    assert S._setup_preopt_warning(st, body, None, "OPT") is None
    st.sessions = type("SS", (), {"get": staticmethod(lambda sid: type("Sess", (), {"doc_meta": {}})())})()
    assert S._setup_preopt_warning(st, body, None, "SP") is None


def test_legacy_clean_route_is_springs_only(served):
    base = served["base"]
    s = session_of(base, F.wurtzite(), "legacy")
    st, _, r = U.post(base, "api/edit/clean", {"session": s["session"], "base_rev": s["rev"], "method": "gfnff"})
    assert st == 400 and "use Clean Geometry (POST /api/clean)" in json.dumps(r)


def test_draft_round_trip_keeps_only_known_deck_and_preopt_values(served):
    base = served["base"]
    s = session_of(base, F.wurtzite(), "draft", meta={"deck": {"geometry_edits": ["SUPERCEL", "rm -rf"],
                                                               "charge_records": ["CHARGED", 3]},
                                                      "preopt": {"method_ran": "gfnff", "model": None},
                                                      "evil": {"x": 1}})
    meta = U.req(base, f"api/edit/state?session={s['session']}")[2]["doc_meta"]
    assert meta["deck"] == {"geometry_edits": ["SUPERCEL"], "charge_records": ["CHARGED"]}
    assert meta["preopt"] == {"method_ran": "gfnff", "model": None} and "evil" not in meta
    s = session_of(base, F.wurtzite(), "draft2", meta={"preopt": {"method_ran": "springs; rm"}})
    assert "preopt" not in (U.req(base, f"api/edit/state?session={s['session']}")[2].get("doc_meta") or {})


def test_time_limit_clamp_follows_the_method(served):
    base = served["base"]
    s = session_of(base, F.wurtzite(), "clamp")
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=480)
    assert st == 200 and r["time_limit"] == 480 and r["max_time_limit"] == 600
    st, r = clean(base, s["session"], s["rev"], method="mlip:chgnet-0.3.0", time_limit=5000)
    assert r["time_limit"] == 600


# --- A21: the weighted CPU budget ------------------------------------------------------------

def test_gate_weight_is_clamped_and_released():
    from mace.gui.compute import ComputeGate
    g = ComputeGate(2)
    with g.slot(weight=8):
        assert g.running == 2
    assert g.running == 0


def test_gate_resize_while_a_weighted_waiter_waits():
    from mace.gui.compute import ComputeGate
    g = ComputeGate(4)
    done = threading.Event()
    hold = g.slot(weight=3)
    hold.__enter__()

    def waiter():
        with g.slot(weight=4):
            done.set()
    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.3)
    g.resize(2)                       # smaller than both: the waiter now needs 2
    hold.__exit__(None, None, None)
    assert done.wait(5)
    t.join(5)
    assert g.running == 0


def test_gate_small_jobs_do_not_starve_a_weighted_waiter(monkeypatch):
    from mace.gui import compute
    monkeypatch.setattr(compute, "RESERVE_AFTER_S", 0.3)
    g = compute.ComputeGate(4)
    stop = threading.Event()
    got = threading.Event()

    def small():
        while not stop.is_set():
            with g.slot():
                time.sleep(0.05)

    smalls = [threading.Thread(target=small, daemon=True) for _ in range(6)]
    for t in smalls:
        t.start()
    time.sleep(0.2)

    def big():
        with g.slot(weight=4):
            got.set()
    threading.Thread(target=big, daemon=True).start()
    assert got.wait(5)
    stop.set()
    for t in smalls:
        t.join(5)
    assert g.running == 0


def test_gate_two_weighted_waiters_do_not_wait_for_each_other(monkeypatch):
    """Both reserve while a small job runs; each used to leave room for the
    other's reservation, so neither ever ran. First come, first served."""
    from mace.gui import compute
    monkeypatch.setattr(compute, "RESERVE_AFTER_S", 0.2)
    g = compute.ComputeGate(4)
    small = g.slot()
    small.__enter__()
    done = []

    def big(tag):
        with g.slot(weight=4):
            done.append(tag)
            time.sleep(0.05)
    ts = [threading.Thread(target=big, args=(k,), daemon=True) for k in "ab"]
    for t in ts:
        t.start()
        time.sleep(0.05)
    time.sleep(0.6)                   # both past RESERVE_AFTER_S: both reserved
    assert len(g.reserved) == 2
    small.__exit__(None, None, None)
    for t in ts:
        t.join(5)
    assert sorted(done) == ["a", "b"] and g.running == 0 and not g.reserved


def test_a_task_queued_on_the_ml_host_holds_no_slots():
    from mace.gui import compute, tasks
    gate_before = compute.GATE.running
    gate = threading.Event()

    def acquire(t):
        while not gate.is_set():
            if t.cancel_requested:
                raise compute.Canceled("canceled")
            time.sleep(0.05)
        return lambda: None
    m = tasks.TaskManager(lambda mgr: None)
    t = m.submit("clean", "x", "builders.clean_job", {}, inline=lambda ctx: {"ok": 1}, weight=3, acquire=acquire)
    time.sleep(0.3)
    assert t.state == "queued" and compute.GATE.running == gate_before
    m.cancel(t.id)
    assert t.done.wait(5) and t.state == "canceled"


def test_ml_host_lease_survives_a_worker_start_that_raises(monkeypatch):
    """A start failing with anything but MlipError (a temp-dir error, say)
    used to leave the lease owned by a finished task: every later ML clean
    then waited "queued: ML worker busy" forever."""
    from types import SimpleNamespace
    from mace.gui import mlip_host
    from mace.preopt import detect, mlip_client, proc

    def boom(*a, **k):
        raise RuntimeError("no private temp directory")
    monkeypatch.setattr(mlip_client, "WorkerProcess", boom)
    monkeypatch.setattr(detect, "configured_python", lambda: sys.executable)
    monkeypatch.setattr(detect, "weights_status", lambda m: {"status": "bundled", "path": None})
    monkeypatch.setattr(proc, "mem_available_gb", lambda: 16.0)
    h = mlip_host.MlipHost()
    try:
        for n in range(2):
            t = SimpleNamespace(id=f"t{n}", payload={}, cancel_requested=False, message="", log="")
            release = h.acquire(t, "chgnet-0.3.0", 1)
            assert t.payload["mlip_lease"]["error"] == "exception: no private temp directory"
            assert h.owner == t.id
            release()
            assert h.owner is None
    finally:
        h.shutdown()
