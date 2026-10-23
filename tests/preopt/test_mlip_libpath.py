"""An ML interpreter built from a cluster's module Python finds its libpython
only through LD_LIBRARY_PATH (the module sets it): MACE finds that folder,
records it per interpreter in mlip.json and puts it in front of the child's
LD_LIBRARY_PATH whenever it starts that interpreter (the probe, the worker,
pip in the installer), so the environment runs where the module is not
loaded. The stand-in "module Python" here is a shell script that fails with
the loader's message unless its lib folder is on LD_LIBRARY_PATH."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="LD_LIBRARY_PATH is a POSIX loader's")

LIB = "libpython3.99.so.1.0"
GATE = """#!/bin/sh
# a module Python: starts only with {lib} on LD_LIBRARY_PATH, as the module sets it
case ":$LD_LIBRARY_PATH:" in
  *":{lib}:"*) exec {target} "$@" ;;
esac
echo "$0: error while loading shared libraries: {name}: cannot open shared object file: No such file or directory" >&2
exit 127
"""
OTHER = "/opt/other-module/lib"           # an unrelated folder already on LD_LIBRARY_PATH: kept, after ours


def gate(path: Path, target: Path, libdir: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GATE.format(lib=libdir, target=target, name=LIB))
    path.chmod(0o755)
    return path


def module_python(root: Path, with_lib: bool = True):
    """root/sw/Python/bin/python3.99 (the module's interpreter, wrapping the fake ML python)
    and root/sw/Python/lib/libpython3.99.so.1.0; returns (interpreter, lib folder)."""
    real = F.make_fake_python(root / "real")
    prefix = root / "sw" / "Python"
    libdir = prefix / "lib"
    libdir.mkdir(parents=True)
    if with_lib:
        (libdir / LIB).write_bytes(b"")
    return gate(prefix / "bin" / "python3.99", real, libdir), libdir


def venv_of(base: Path, d: Path, copies: bool = False) -> Path:
    """A venv's shape: bin/python links to (or is a copy of) the base, pyvenv.cfg names its home."""
    (d / "bin").mkdir(parents=True)
    (d / "pyvenv.cfg").write_text(f"home = {base.parent}\ninclude-system-site-packages = false\nversion = 3.99.0\n")
    py = d / "bin" / "python"
    if copies:
        shutil.copy(base, py)
    else:
        py.symlink_to(base)
    return py


@pytest.fixture
def no_module(monkeypatch):
    """A shell without the module loaded: LD_LIBRARY_PATH has only an unrelated folder."""
    monkeypatch.setenv("LD_LIBRARY_PATH", OTHER)
    from mace.preopt import detect
    detect._LIBDIRS_MEM.clear()
    yield
    detect._LIBDIRS_MEM.clear()


def record():
    from mace.preopt import detect
    p = detect.config_path()
    return json.loads(p.read_text()) if p.exists() else {}


def test_find_library_dirs_beside_the_real_path_or_the_venv_home(tmp_path, no_module):
    from mace.preopt import detect
    base, libdir = module_python(tmp_path)
    linked = venv_of(base, tmp_path / "venv")
    copied = venv_of(base, tmp_path / "venv-copies", copies=True)
    for py in (base, linked, copied):
        assert subprocess.run([str(py), "-c", "pass"], capture_output=True).returncode == 127
        assert detect.find_library_dirs(str(py)) == {"ok": True, "dirs": [str(libdir)], "reason": ""}
    plain = F.make_fake_python(tmp_path / "plain")                 # starts as it is: nothing needed
    assert detect.find_library_dirs(str(plain)) == {"ok": True, "dirs": [], "reason": ""}
    # the library is nowhere MACE looks: not ok, and the reason names it
    nolib, _ = module_python(tmp_path / "nolib", with_lib=False)
    r = detect.find_library_dirs(str(nolib))
    assert not r["ok"] and r["missing"] == LIB and LIB in r["reason"]


def test_a_folder_on_the_current_library_path_is_found_too(tmp_path, monkeypatch):
    """A Python whose module put its libpython elsewhere: found among the current LD_LIBRARY_PATH's folders."""
    from mace.preopt import detect
    real = F.make_fake_python(tmp_path / "real")
    elsewhere = tmp_path / "elsewhere" / "lib"
    elsewhere.mkdir(parents=True)
    (elsewhere / LIB).write_bytes(b"")
    py = gate(tmp_path / "opt" / "bin" / "python3.99", real, elsewhere)
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{OTHER}:{elsewhere}")
    assert detect.find_library_dirs(str(py))["dirs"] == [str(elsewhere)]


def test_with_library_dirs_prepends_once_and_keeps_the_rest():
    from mace.preopt import detect
    env = detect.with_library_dirs({"LD_LIBRARY_PATH": f"/a:{OTHER}:/a"}, ["/m/lib", "/a"])
    assert env["LD_LIBRARY_PATH"] == f"/m/lib:/a:{OTHER}"
    assert detect.with_library_dirs({}, ["/m/lib"]) == {"LD_LIBRARY_PATH": "/m/lib"}
    assert detect.with_library_dirs({"X": "1"}, []) == {"X": "1"}


def test_the_probe_finds_records_and_then_uses_the_folder(tmp_path, no_module):
    """A recorded environment (made where the module was loaded) probed where it is not: the probe
    finds the folder, probes again, records it; the next process uses the record."""
    from mace.preopt import detect
    base, libdir = module_python(tmp_path)
    py = str(venv_of(base, tmp_path / "venv"))
    detect.save_python(py)
    pr = detect.probe(py, recheck=True)
    assert pr["status"] == "ok" and pr["library_dirs"] == [str(libdir)]
    assert record()["python"] == py and record()["library_path"] == {py: {"dirs": [str(libdir)]}}
    detect._LIBDIRS_MEM.clear()                                       # another process: only the record
    detect._MEM.clear()
    assert detect.library_dirs(py) == [str(libdir)]
    pr = detect.probe(py, recheck=True)
    assert pr["status"] == "ok" and "library_dirs" not in pr          # started with the record, no search
    st = detect.mlip_status(detect_model(), recheck=True)
    assert st["status"] != "import_failed", st
    env = detect.interpreter_env(py)
    assert env["LD_LIBRARY_PATH"] == f"{libdir}:{OTHER}" and "PYTHONPATH" not in env


def detect_model():
    from mace.preopt import registry
    return registry.get("chgnet-0.3.0")


def test_a_library_found_nowhere_gives_the_module_hint(tmp_path, no_module):
    from mace.preopt import detect
    base, _ = module_python(tmp_path, with_lib=False)
    py = str(venv_of(base, tmp_path / "venv"))
    detect.save_python(py)
    st = detect.mlip_status(detect_model(), recheck=True)
    assert st["status"] == "import_failed" and "error while loading shared libraries" in st["reason"]
    assert LIB in st["reason"] and "load the module its Python came from" in st["hint"]
    assert f"mace mlip setup --python {py}" in detect.hint_commands(st["hint"])
    assert "library_path" not in record()


def test_the_worker_starts_with_the_recorded_folder(tmp_path, no_module, monkeypatch):
    pytest.importorskip("ase")                      # the fake worker's calculator is an ase Calculator
    from mace.preopt import detect, mlip_client
    base, libdir = module_python(tmp_path)
    py = str(venv_of(base, tmp_path / "venv"))
    detect.save_library_dirs(py, [str(libdir)], "Python/3.99.0-GCCcore-99")
    assert detect.recorded_library(py) == {"dirs": [str(libdir)], "module": "Python/3.99.0-GCCcore-99"}
    argv, env, _ = mlip_client.launch(py, detect_model(), weights=None, threads=1, workdir=tmp_path)
    assert env["LD_LIBRARY_PATH"] == f"{libdir}:{OTHER}"
    monkeypatch.setenv("FAKE_ENV_LOG", str(tmp_path / "env.json"))
    w = mlip_client.WorkerProcess(py, detect_model())
    try:
        assert w.hello["model"] == "chgnet-0.3.0" and w.ping()
    finally:
        w.close()
    assert json.loads((tmp_path / "env.json").read_text())["env"]["LD_LIBRARY_PATH"] == f"{libdir}:{OTHER}"
    # forgetting it again: the record goes, and the worker no longer starts here
    detect.save_library_dirs(py, [])
    assert "library_path" not in record()
    with pytest.raises(mlip_client.MlipError):
        mlip_client.WorkerProcess(py, detect_model(), hello_timeout=20)


def mace(*args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "MACE_"))}
    e.update(env or {})
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", *args], env=e,
                       cwd=os.environ["TMPDIR"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_mlip_setup_python_records_the_folder_through_mace_cli(tmp_path, no_module):
    from mace.preopt import detect
    base, libdir = module_python(tmp_path)
    py = str(venv_of(base, tmp_path / "venv"))
    rc, out = mace("mlip", "setup", "--python", py)
    assert rc == 0 and "Recorded in" in out and f"It needs {libdir} on LD_LIBRARY_PATH" in out, out
    assert record() == {"python": py, "library_path": {py: {"dirs": [str(libdir)]}}}
    # an interpreter that starts as it is: recorded, nothing for its library path (a stale entry goes)
    plain = str(F.make_fake_python(tmp_path / "plain"))
    detect.save_library_dirs(plain, ["/stale/lib"])
    rc, out = mace("mlip", "setup", "--python", plain)
    assert rc == 0 and "LD_LIBRARY_PATH" not in out, out
    assert record() == {"python": plain, "library_path": {py: {"dirs": [str(libdir)]}}}


def test_mace_calc_runs_a_recorded_module_environment_without_the_module(tmp_path, no_module):
    """`mace calc` through mace_cli with the environment recorded where the module was loaded (no library
    record yet): the probe records the folder and the worker starts with it."""
    pytest.importorskip("ase")                      # mace calc reads the CIF and the worker computes with ase
    f = tmp_path / "my.model"
    f.write_bytes(b"a model file the user supplies")
    cif = tmp_path / "NaCl.cif"
    shutil.copy(F.EXAMPLES / "NaCl.cif", cif)
    base, libdir = module_python(tmp_path)
    py = str(venv_of(base, tmp_path / "venv"))
    from mace.preopt import detect
    detect.save_python(py)
    log = tmp_path / "env.json"
    rc, out = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", str(f), "--model-backend", "mace-torch",
                   "--model-license-ok", "--device", "cpu", "-o", str(tmp_path / "out"), env={"FAKE_ENV_LOG": str(log)})
    assert rc == 0, out
    assert json.loads(log.read_text())["env"]["LD_LIBRARY_PATH"] == f"{libdir}:{OTHER}"
    assert record()["library_path"] == {py: {"dirs": [str(libdir)]}}


# --- the installer, from a module Python that does not start without its folder ------------------

BASE = r'''#!{exe}
import os, sys
args = sys.argv[1:]
if "-c" in args:
    print("3 12 2"); sys.exit(0)
if args[:2] == ["-m", "venv"]:
    d = args[2]
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    open(os.path.join(d, "pyvenv.cfg"), "w").write("home = {home}\ninclude-system-site-packages = false\n")
    os.symlink({venv_target!r}, os.path.join(d, "bin", "python"))
    print("created", d); sys.exit(0)
sys.exit(3)
'''


def test_install_from_a_module_python_runs_every_step_with_its_folder_and_records_it(tmp_path, no_module,
                                                                                       monkeypatch):
    from mace.preopt import detect, mlip_install as MI
    prefix = tmp_path / "sw" / "Python"
    libdir = prefix / "lib"
    libdir.mkdir(parents=True)
    (libdir / LIB).write_bytes(b"")
    real = tmp_path / "real"
    real.mkdir()
    venv_real = real / "venv_py"
    venv_real.write_text(F.VENV_PY.format(exe=sys.executable))
    venv_real.chmod(0o755)
    venv_gate = gate(prefix / "bin" / "python3.99-env", venv_real, libdir)
    base_real = real / "base_py"
    base_real.write_text(BASE.format(exe=sys.executable, home=prefix / "bin", venv_target=str(venv_gate)))
    base_real.chmod(0o755)
    base = gate(prefix / "bin" / "python3.99", base_real, libdir)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "pip.log"))
    monkeypatch.setattr(MI, "gpu_present", lambda: False)
    plan = MI.make_plan(str(tmp_path / "env"), backends=["chgnet"], python=str(base))
    assert plan["problems"] == [] and plan["python_version"] == "3.12.2" and plan["library_dirs"] == [str(libdir)]
    lines = []
    res = MI.install(plan, log=lines.append)
    py = str(tmp_path / "env" / "bin" / "python")
    calls = [json.loads(x) for x in (tmp_path / "pip.log").read_text().splitlines()]
    assert [c["argv"][:2] for c in calls] == [["install", "--upgrade"], ["install", "torch"],
                                              ["install", "chgnet==0.4.2"], ["check"]]
    assert res["ok"] and record() == {"python": py, "library_path": {py: {"dirs": [str(libdir)]}}}
    assert any("Recorded its library folder" in x for x in lines), lines
    detect._LIBDIRS_MEM.clear()
    detect._MEM.clear()
    pr = detect.probe(py, recheck=True)
    assert pr["status"] == "ok" and "library_dirs" not in pr
    # the same install without the folder would not even make the venv: the base does not start
    assert MI._version(str(base)) is None


# --- a module Python whose standard library needs another module's library (MSU HPCC's 3.11: libssl 1.1) ---

TOOLCHAIN_PY = r'''#!{exe}
# starts only with {libdir} on LD_LIBRARY_PATH; its _ssl loads only with {ssldir} there too
import json, os, sys
ld = os.environ.get("LD_LIBRARY_PATH", "").split(":")
if {libdir!r} not in ld:
    sys.stderr.write(sys.argv[0] + ": error while loading shared libraries: {lib}: cannot open shared object file: "
                     "No such file or directory\n"); sys.exit(127)
ssl = {ssldir!r} in ld
code = sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else ""
if "MACE_STDLIB_LIB" in code:
    if not ssl:
        sys.stderr.write("MACE_STDLIB_LIB libssl.so.1.1: cannot open shared object file: No such file or directory\n")
        sys.exit(3)
    sys.exit(0)
if code:
    ok = {{"ok": True, "version": "1.0-fake"}}
    mt = ok if ssl else {{"ok": False, "version": "0.3.16",
                          "error": "ImportError: libssl.so.1.1: cannot open shared object file: No such file or directory"}}
    print("MACE_PROBE_JSON " + json.dumps({{"python": "3.11.3", "prefix": sys.prefix, "packages": {{
        "torch": ok, "ase": ok, "mace-torch": mt}}}}))
    sys.exit(0)
sys.exit(3)
'''


def toolchain_python(root: Path):
    prefix = root / "sw" / "Python"
    libdir, ssldir = prefix / "lib", root / "sw" / "OpenSSL" / "lib"
    for d, name in ((libdir, LIB), (ssldir, "libssl.so.1.1")):
        d.mkdir(parents=True)
        (d / name).write_bytes(b"")
    base = prefix / "bin" / "python3.99"
    base.parent.mkdir(parents=True)
    base.write_text(TOOLCHAIN_PY.format(exe=sys.executable, libdir=str(libdir), ssldir=str(ssldir), lib=LIB))
    base.chmod(0o755)
    return venv_of(base, root / "venv"), libdir, ssldir


def test_a_standard_library_module_needing_another_module_folder(tmp_path, monkeypatch, no_module):
    from mace.preopt import detect, registry
    py, libdir, ssldir = toolchain_python(tmp_path)
    py = str(py)
    # where the module is loaded (its folders on LD_LIBRARY_PATH): both are found and recorded
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{OTHER}:{ssldir}:{libdir}")
    assert detect.find_library_dirs(py) == {"ok": True, "dirs": [str(libdir), str(ssldir)], "reason": ""}
    rc, out = mace("mlip", "setup", "--python", py, env={"LD_LIBRARY_PATH": f"{OTHER}:{ssldir}:{libdir}"})
    assert rc == 0 and f"It needs {libdir}:{ssldir} on LD_LIBRARY_PATH" in out, out
    assert record()["library_path"] == {py: {"dirs": [str(libdir), str(ssldir)]}}
    detect._LIBDIRS_MEM.clear()
    monkeypatch.setenv("LD_LIBRARY_PATH", OTHER)                        # then without the module: it runs
    assert detect.probe(py, recheck=True)["data"]["packages"]["mace-torch"]["ok"]
    # recorded before (no library record) and first probed without the module: libpython's folder is
    # found beside it, OpenSSL's is not; it starts, and the hint says to record it where the module is loaded
    detect.save_library_dirs(py, [])
    detect._LIBDIRS_MEM.clear()
    r = detect.find_library_dirs(py)
    assert r["ok"] and r["dirs"] == [str(libdir)] and r["missing"] == "libssl.so.1.1"
    st = detect.mlip_status(registry.get("acesuit-mace-mpa-0-medium"), recheck=True)
    assert st["status"] == "import_failed" and "libssl.so.1.1" in st["reason"]
    assert "load the module its Python came from" in st["hint"] and f"mace mlip setup --python {py}" in st["hint"]
    assert record()["library_path"] == {py: {"dirs": [str(libdir)]}}
