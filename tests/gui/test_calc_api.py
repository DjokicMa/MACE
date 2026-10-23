"""The Calculators panel's routes through the real `mace gui` (mace calc in
the GUI): the backend list, a UFF4MOF optimization that opens as a new
document and saves the CLI's files next to the CIF, EQeq charges, refusals
(an unavailable backend with its setup commands, a bad task), and an ML
potential through the server's one ML worker (the fake worker of the preopt
tests: no torch) on device auto, which must give the worker GPU 0 only."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calcapi")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    py = F.make_fake_python(tmp / "venv")
    ptmp = tmp / "ptmp"
    ptmp.mkdir()
    env = dict(MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2", MACE_MLIP_PYTHON=str(py),
               FAKE_ENV_LOG=str(tmp / "env.json"), FAKE_MODE_FILE=str(tmp / "fake_mode"), PATH="/usr/bin:/bin",
               TMPDIR=str(ptmp),
               XDG_DATA_HOME=str(tmp / "data"), CUDA_VISIBLE_DEVICES=None)
    env = {k: v for k, v in env.items() if v is not None}
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        proc, base, _ = H.start_server(root, xdg, **env)
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    end = time.time() + 60
    while time.time() < end and U.req(base, "api/calc/backends")[2].get("checking"):
        time.sleep(0.2)
    yield {"base": base, "root": root, "tmp": tmp, "proc": proc}
    proc.terminate()
    proc.wait(timeout=20)


def result_of(base, st, r, timeout=180):
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
    raise AssertionError("the calculation did not finish")


def run(base, body):
    st, _, r = U.post(base, "api/calc/run", body)
    if st >= 400:
        return st, r
    return 200, result_of(base, st, r)


def test_backends_listed(served):
    st, _, r = U.req(served["base"], "api/calc/backends")
    assert st == 200 and not r["checking"]
    names = [b["name"] for b in r["backends"]]
    assert names == ["uff", "eqeq", "xtb", "dftb", "mlip"]
    uff = next(b for b in r["backends"] if b["name"] == "uff")
    assert uff["available"] and uff["default_method"] == "uff4mof"
    ml = next(b for b in r["backends"] if b["name"] == "mlip")
    labels = " ".join(m["label"] for m in ml["methods"])
    assert "mace-torch" in labels and "orb-models" in labels and "mattersim" in labels
    assert not any("off23" in m["id"] or "ASL" in m["license"] or "restricted" in m for m in ml["methods"])
    xtb = next(b for b in r["backends"] if b["name"] == "xtb")
    assert not xtb["available"] and xtb["methods"][0]["commands"]


def test_uff_opt_opens_and_saves(served):
    base, root = served["base"], served["root"]
    body = {"source": {"path": "structures/MgO.cif"}, "backend": "uff", "method": "uff4mof", "task": "opt",
            "steps": 50, "fmax": 0.05, "save": True}
    st, r = run(base, body)
    assert st == 200, r
    assert r["calc_task"] == "opt" and r["energy_eV"] is not None
    assert r["open"]["session"] or r["open"]["held"]
    files = r["files"]
    assert set(files) >= {"report", "structure", "trajectory"}
    cif = root / files["structure"]
    assert cif.is_file() and "# mace-preopt: method=calc:uff:uff4mof" in cif.read_text()
    rep = json.loads((root / files["report"]).read_text())
    assert rep["provenance"]["tool"] == "mace calc" and rep["backend"] == "uff"
    st, r2 = run(base, body)                       # the same files again: refused without overwrite
    assert st == 200 and "already exist" in r2["save_error"]


def test_eqeq_charges(served):
    st, r = run(served["base"], {"source": {"path": "structures/NaCl.cif"}, "backend": "eqeq", "task": "charges"})
    assert st == 200, r
    q, sym = r["charges"], r["symbols"]
    assert abs(sum(q)) < 1e-6
    assert all(x > 0 for x, s in zip(q, sym) if s == "Na") and all(x < 0 for x, s in zip(q, sym) if s == "Cl")


def test_refusals(served):
    base = served["base"]
    st, r = run(base, {"source": {"path": "structures/Si.cif"}, "backend": "xtb", "method": "gfn2", "task": "sp"})
    assert st == 400 and r.get("unavailable") and r.get("commands")
    st, r = run(base, {"source": {"path": "structures/Si.cif"}, "backend": "eqeq", "task": "md"})
    assert st == 400 and "does not do" in r["error"]
    st, r = run(base, {"source": {"path": "structures/Si.cif"}, "backend": "uff", "task": "sp", "device": "gpu7"})
    assert st == 400


def test_mlip_auto_device_is_gpu0(served):
    """The fake ML environment has chgnet (weights in the package): the model
    loads in the server's ML worker on device auto, and the worker sees
    CUDA_VISIBLE_DEVICES=0 (never a second GPU)."""
    st, r = run(served["base"], {"source": {"path": "structures/Si.cif"}, "backend": "mlip",
                                 "method": "chgnet-0.3.0", "task": "sp", "device": "auto"})
    assert st == 200, r
    assert r["energy_eV"] is not None
    assert "GPU 0" in r["report"]["device_text"]
    env = json.loads((served["tmp"] / "env.json").read_text())
    assert env["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "--device" in env["argv"] and env["argv"][env["argv"].index("--device") + 1] == "auto"


def test_backend_failure_is_a_clean_error(served):
    """The ML worker crashes during the call: the route answers 422 with what failed and the backend's own
    words (no 500, no traceback); a detached task fails the same way. Last here: the crash ends the worker."""
    mode = served["tmp"] / "fake_mode"
    mode.write_text("crash")
    try:
        st, _, r = U.post(served["base"], "api/calc/run", {"source": {"path": "structures/Si.cif"}, "backend": "mlip",
                                                           "method": "chgnet-0.3.0", "task": "sp", "device": "cpu"})
        if st == 202:
            tid = r["task"]["id"]
            end = time.time() + 120
            while time.time() < end:
                st2, _, r2 = U.req(served["base"], f"api/tasks/{tid}/result")
                if r2.get("task", {}).get("state") in ("failed", "done", "canceled"):
                    break
                time.sleep(0.1)
            rec = r2["task"]
            assert rec["state"] == "failed" and rec["traceback"] is None, rec
            text = rec["error"]
        else:
            assert st == 422, (st, r)
            assert r["task"]["traceback"] is None and r["kind"] == "crash", r
            text = r["error"]
        assert text.startswith("CHGNet 0.3.0 failed during the single point: crash: the ML worker "), text
    finally:
        mode.unlink()
