"""The ML worker protocol with a fake worker (no torch): launch argv and
environment (A11), hello checks, errors, timeouts, cancel with grace and
kill, both carriers (A9, A12), and the name-clash self-check of the real
worker."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ase")

import preopt_fixtures as F  # noqa: E402
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)


def model(mid="chgnet-0.3.0"):
    from mace.preopt import registry
    return registry.get(mid)


def worker(tmp_path, monkeypatch, mode="", use_socket=False, **kw):
    from mace.preopt import mlip_client
    py = F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("FAKE_MODE", mode)
    monkeypatch.setenv("FAKE_HELLO_LOG", str(tmp_path / "hello.log"))
    monkeypatch.setenv("FAKE_ENV_LOG", str(tmp_path / "env.json"))
    return mlip_client.WorkerProcess(str(py), model(), threads=2, use_socket=use_socket, **kw)


def calc_msg(atoms):
    return {"op": "calc", "numbers": atoms.numbers.tolist(), "cell": atoms.cell.array.tolist(),
            "positions": atoms.positions.tolist(), "pbc": [True] * 3, "properties": ["energy", "forces", "stress"]}


def reaped_and_gone(p):
    try:
        p.wait(timeout=10)                        # reap first: a zombie still has a process group
    except subprocess.TimeoutExpired:
        return False
    try:
        os.killpg(p.pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("device,visible,wdev", [("cpu", "", "cpu"), ("cuda:0", "0", "cuda:0"),
                                                 ("cuda:1", "1", "cuda:0")])
def test_launch_argv_and_env(tmp_path, monkeypatch, device, visible, wdev):
    from mace.preopt import mlip_client
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.setenv("MACE_HOME", "/m")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    argv, env, text = mlip_client.launch("/venv/bin/python", model("acesuit-mace-mpa-0-medium"), weights="/w.model",
                                         threads=3, device=device, workdir=tmp_path)
    assert argv[:3] == ["/venv/bin/python", "-I", str(mlip_client.WORKER)]
    assert argv[argv.index("--device") + 1] == wdev and argv[argv.index("--threads") + 1] == "3"
    assert argv[argv.index("--dtype") + 1] == "float32" and argv[argv.index("--weights") + 1] == "/w.model"
    assert env["CUDA_VISIBLE_DEVICES"] == visible and env["OMP_NUM_THREADS"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"          # GPU numbers as nvidia-smi's
    assert env["MACE_MLIP_NO_DOWNLOAD"] == "1" and env["HF_HUB_OFFLINE"] == "1" and env["TMPDIR"] == str(tmp_path)
    assert env["LD_LIBRARY_PATH"] == "/opt/lib"
    assert not [k for k in env if k.startswith("PYTHON")] and [k for k in env if k.startswith("MACE_")] == ["MACE_MLIP_NO_DOWNLOAD"]
    if device == "cuda:1":
        assert text == "cuda:1 (visible to the worker as cuda:0)"


def test_device_mapping_counts_within_an_inherited_gpu_list():
    """Polish A7: cuda:N inside a job with CUDA_VISIBLE_DEVICES set indexes that
    list (SLURM, a site wrapper): never a GPU the job was not given."""
    from mace.preopt.mlip_client import device_mapping
    # unset: today's behavior
    assert device_mapping("cuda:2", {})[:2] == ("2", "cuda:0")
    for env in ({}, {"CUDA_VISIBLE_DEVICES": "3,5"}, {"CUDA_VISIBLE_DEVICES": ""}):
        assert device_mapping("cpu", env) == ("", "cpu", "cpu")
    env = {"CUDA_VISIBLE_DEVICES": "3,5"}
    assert device_mapping("cuda:0", env)[:2] == ("3", "cuda:0")
    vis, wdev, text = device_mapping("cuda:1", env)
    assert (vis, wdev) == ("5", "cuda:0") and "cuda:1 = physical 5" in text
    with pytest.raises(ValueError, match=r"can see 2 GPU\(s\).*cuda:0\.\.cuda:1"):
        device_mapping("cuda:2", env)
    for none in ("", "-1"):
        with pytest.raises(ValueError, match="no GPU is visible to this job"):
            device_mapping("cuda:0", {"CUDA_VISIBLE_DEVICES": none})
    with pytest.raises(ValueError, match="use cpu or cuda:N"):
        device_mapping("cuda:x", env)


def test_launch_maps_the_device_in_the_environment_it_is_given(tmp_path):
    from mace.preopt import mlip_client
    argv, env, text = mlip_client.launch("/venv/bin/python", model("acesuit-mace-mpa-0-medium"), weights=None,
                                         threads=1, device="cuda:1", workdir=tmp_path,
                                         environ={"CUDA_VISIBLE_DEVICES": "3,5"})
    assert env["CUDA_VISIBLE_DEVICES"] == "5" and argv[argv.index("--device") + 1] == "cuda:0"


def test_preopt_cli_refuses_a_gpu_outside_the_job(tmp_path):
    """The real invocation: `mace preopt --device cuda:3` in a job that sees one
    GPU fails with the reason on stderr, before any worker starts."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM")}
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    work = tmp_path / "work"
    work.mkdir()
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "preopt", str(F.EXAMPLES / "MgO.cif"),
                        "--method", "mlip:acesuit-mace-mpa-0-medium", "--device", "cuda:3"],
                       capture_output=True, text=True, env=env, timeout=120, cwd=work)
    assert r.returncode != 0, r.stdout
    assert "this job can see 1 GPU(s)" in r.stderr, r.stderr
    assert not list(work.iterdir())


def test_pipe_round_trip_and_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(F.REPO))
    w = worker(tmp_path, monkeypatch)
    try:
        assert w.hello["protocol"] == 1 and w.hello["model"] == "chgnet-0.3.0" and w.hello["threads"] == 2
        a = F.demo("NaCl.cif")
        r = w.request(calc_msg(a), 10)
        assert r["ok"] and len(r["forces"]) == len(a) and len(r["stress"]) == 6
        assert w.ping()
        logged = json.loads((tmp_path / "env.json").read_text())
        assert logged["cwd"] == str(w.workdir) and logged["flags_isolated"] == 1
        assert "PYTHONPATH" not in logged["env"] and logged["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert logged["env"]["OMP_NUM_THREADS"] == "2"
    finally:
        w.close()
    assert not w.workdir.exists() and reaped_and_gone(w.p)


def test_floats_round_trip(tmp_path, monkeypatch):
    from mace.preopt import mlip_client
    w = worker(tmp_path, monkeypatch)
    try:
        a = F.rattled_mgo()
        c = mlip_client.MlipCalculator(w, owner=w)
        a.calc = c
        f = a.get_forces()
        ref = F.load_fake_calc()
        b = a.copy()
        b.calc = ref
        assert f.tobytes() == b.get_forces().tobytes()                 # repr floats: bit for bit
    finally:
        w.close()


@pytest.mark.parametrize("mode,kind", [("nan", "nan"), ("nanenergy", "nan"), ("bigforce", "bigforce"),
                                       ("crash", "crash"), ("oom", "oom")])
def test_errors_raise_never_fall_back_silently(tmp_path, monkeypatch, mode, kind):
    from mace.preopt import mlip_client
    w = worker(tmp_path, monkeypatch, mode)
    try:
        a = F.demo("NaCl.cif")
        a.calc = mlip_client.MlipCalculator(w, owner=w)
        with pytest.raises(mlip_client.MlipError) as e:
            a.get_potential_energy()
        assert e.value.kind == kind, (e.value.kind, str(e.value))
        if mode == "crash":
            assert "exited (code 9) during a call" in str(e.value)
    finally:
        w.close()


def test_hello_checks(tmp_path, monkeypatch):
    from mace.preopt import mlip_client
    with pytest.raises(mlip_client.MlipError) as e:
        worker(tmp_path, monkeypatch, "badhello")
    assert e.value.kind == "protocol" and "protocol 2" in str(e.value)
    py = F.make_fake_python(tmp_path / "venv2")
    monkeypatch.setenv("FAKE_MODE", "")
    m = dict(model(), id="chgnet-0.3.0")
    w = mlip_client.WorkerProcess(str(py), m)
    try:
        with pytest.raises(mlip_client.MlipError):
            mlip_client._check_hello(w.hello, "acesuit-mace-mpa-0-medium")       # the model id must match
        with pytest.raises(mlip_client.MlipError) as e:
            mlip_client._check_elements({"z": [1, 8]}, {1, 8, 92})
        assert e.value.kind == "elements" and "U" in str(e.value)
    finally:
        w.close()
    with pytest.raises(mlip_client.MlipTimeout):
        worker(tmp_path, monkeypatch, "slowload", hello_timeout=1.0)


def test_cancel_during_the_hello_wait_kills_the_worker(tmp_path, monkeypatch):
    from mace.preopt import mlip_client, proc
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    t0 = time.monotonic()
    with pytest.raises(proc.Canceled):
        worker(tmp_path, monkeypatch, "slowload", canceled=flag.is_set)
    assert time.monotonic() - t0 < 4


def test_stderr_tail_in_errors(tmp_path, monkeypatch):
    from mace.preopt import mlip_client
    py = tmp_path / "bad" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\necho 'Traceback: torch not found' >&2\nexit 7\n")
    py.chmod(0o755)
    with pytest.raises(mlip_client.MlipError) as e:
        mlip_client.WorkerProcess(str(py), model())
    assert "exited (code 7)" in str(e.value) and "torch not found" in str(e.value)


def test_library_output_on_stdout_does_not_break_the_protocol(tmp_path, monkeypatch):
    """mace_mp prints while it loads; the worker keeps stdout for the protocol."""
    w = worker(tmp_path, monkeypatch, "chatty")
    try:
        assert w.hello["op"] == "hello" and w.request(calc_msg(F.demo("NaCl.cif")), 10)["ok"]
        assert "Using Materials Project MACE" in w.log.read_text()
    finally:
        w.close()


def test_stdin_eof_exits_the_worker(tmp_path, monkeypatch):
    w = worker(tmp_path, monkeypatch)
    w.p.stdin.close()
    assert w.p.wait(timeout=10) == 0
    w.close()


def test_cancel_in_flight_grace_keeps_the_worker(tmp_path, monkeypatch):
    """slowcall 1 s: the cancel lets the call finish, the reply is dropped,
    the same worker answers afterwards and no second model load happens."""
    from mace.preopt import mlip_client, proc
    monkeypatch.setenv("FAKE_SLOWCALL", "1")
    w = worker(tmp_path, monkeypatch, "slowcall")
    try:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        with pytest.raises(proc.Canceled):
            w.request(calc_msg(F.demo("NaCl.cif")), 30, flag.is_set)
        assert w.alive() and w.ping(timeout=10)
        assert len((tmp_path / "hello.log").read_text().splitlines()) == 1
    finally:
        w.close()


def test_cancel_past_the_grace_kills_the_group(tmp_path, monkeypatch):
    from mace.preopt import mlip_client, proc
    monkeypatch.setattr(mlip_client, "CANCEL_GRACE", 1.0)
    w = worker(tmp_path, monkeypatch, "hang")
    flag = threading.Event()
    threading.Timer(0.2, flag.set).start()
    t0 = time.monotonic()
    with pytest.raises(proc.Canceled):
        w.request(calc_msg(F.demo("NaCl.cif")), 30, flag.is_set)
    assert time.monotonic() - t0 < 0.2 + 1.0 + 2.5
    assert reaped_and_gone(w.p) and not w.alive()
    w.close()


def test_call_timeout_kills(tmp_path, monkeypatch):
    from mace.preopt import mlip_client
    w = worker(tmp_path, monkeypatch, "hang")
    with pytest.raises(mlip_client.MlipTimeout):
        w.request(calc_msg(F.demo("NaCl.cif")), 1.0)
    assert reaped_and_gone(w.p)
    w.close()
    assert mlip_client.call_timeout(100) == 35.0


def test_socket_carrier_and_a_client_closing_mid_call(tmp_path, monkeypatch):
    from mace.preopt import mlip_client, proc
    monkeypatch.setenv("FAKE_SLOWCALL", "0.5")
    w = worker(tmp_path, monkeypatch, "slowcall", use_socket=True)
    try:
        assert (os.stat(w.socket_path).st_mode & 0o777) == 0o600
        assert (os.stat(w.workdir).st_mode & 0o777) == 0o700
        c = mlip_client.SocketClient(w.socket_path)
        flag = threading.Event()
        threading.Timer(0.1, flag.set).start()
        with pytest.raises(proc.Canceled):
            c.request(calc_msg(F.demo("NaCl.cif")), 30, flag.is_set)    # closes mid-call
        c2 = mlip_client.SocketClient(w.socket_path)
        r = c2.request(calc_msg(F.demo("NaCl.cif")), 30)
        assert r["ok"] and r["id"] == 1
        c2.close()
        assert len((tmp_path / "hello.log").read_text().splitlines()) == 1
    finally:
        w.close()


def test_long_socket_path_goes_through_the_directory_fd(tmp_path, monkeypatch):
    from mace.preopt import mlip_client, proc
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    monkeypatch.setattr(__import__("tempfile"), "tempdir", str(deep))
    w = worker(tmp_path, monkeypatch, use_socket=True)
    try:
        assert len(w.socket_path) > 108
        c = mlip_client.SocketClient(w.socket_path)
        assert c.request({"op": "ping"}, 10) == {"op": "pong"}
        c.close()
    finally:
        w.close()


def test_self_check_refuses_a_foreign_mace_package(tmp_path):
    """mace_mp backend: a 'mace' first on the path that the mace-torch
    distribution does not own (this MACE tool, say) -> exit 3."""
    from mace.preopt import mlip_client
    site = tmp_path / "site"
    (site / "mace").mkdir(parents=True)
    (site / "mace" / "__init__.py").write_text("")
    other = tmp_path / "elsewhere" / "mace"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("")
    di = site / "mace_torch-0.3.16.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: mace-torch\nVersion: 0.3.16\n")
    (di / "RECORD").write_text("../elsewhere/mace/__init__.py,,\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(site)
    r = subprocess.run([sys.executable, str(mlip_client.WORKER), "--model-id", "x", "--backend", "mace_mp"],
                       env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "package name clash" in r.stderr, r.stderr
    r = subprocess.run([sys.executable, "-I", str(mlip_client.WORKER), "--model-id", "x", "--backend", "mace_mp"],
                       env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "package name clash" in r.stderr       # -I: not even on the path


def test_preoptimize_with_the_fake_ml_tier(tmp_path, monkeypatch):
    """The whole pipeline through make_calculator (CLI route: a worker of its own)."""
    from mace.preopt import core
    py = F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("MACE_MLIP_PYTHON", str(py))
    monkeypatch.setenv("FAKE_HELLO_LOG", str(tmp_path / "hello.log"))
    r = core.preoptimize(F.wurtzite(), "mlip:chgnet-0.3.0", time_limit=60)
    assert r.method_ran == "mlip:chgnet-0.3.0" and r.report["converged"], r.attempts
    p = core.provenance(r)
    assert p["model"] == "chgnet-0.3.0" and p["versions"]["torch"] == "0.0-fake" and p["device"] == "cpu"
    assert p["training_data"] and p["citation"]
    monkeypatch.setenv("FAKE_MODE", "crash")
    r = core.preoptimize(F.wurtzite(), "mlip:chgnet-0.3.0", time_limit=60)
    assert r.fell_back and r.attempts[-1]["status"] == "failed" and r.attempts[-1]["reason"].startswith("crash:")
