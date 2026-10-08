"""GFN-FF without xtb (A3, A4, A8, A9, A10): the parsers on hand-written
Turbomole blocks, the stress formula against finite differences, the version
parser, detection with fake ``xtb`` programs on a temp PATH, the child
environment, the temp-directory sweeper and cancel/timeout of a hung xtb."""
import os
import stat
import subprocess
import sys
import textwrap
import time

import pytest

from preopt_fixtures import preopt_sandbox  # noqa: F401  (autouse: temp root, XDG dirs)

np = pytest.importorskip("numpy")

GRADIENT = """$grad
  cycle =      1    SCF energy =   -10.00000000000   |dE/xyz| =  0.100000
    0.00000000000000      0.00000000000000      0.00000000000000      si
    2.56530000000000      2.56530000000000      2.56530000000000      si
   1.0000000000000D-01   2.0000000000000D-02  -3.0000000000000D-03
  -1.0000000000000D-01  -2.0000000000000D-02   3.0000000000000D-03
  cycle =      2    SCF energy =   -10.50000000000   |dE/xyz| =  0.010000
    0.00000000000000      0.00000000000000      0.00000000000000      si
    2.56530000000000      2.56530000000000      2.56530000000000      si
   1.5000000000000D-02  -2.5000000000000E-03   0.0000000000000D+00
  -1.5000000000000D-02   2.5000000000000E-03   0.0000000000000D+00
$end
"""


def test_parse_turbomole_block_takes_the_last_cycle():
    from mace.preopt import gfnff
    g = gfnff.parse_turbomole_block(GRADIENT, 2)
    assert np.allclose(g, [[0.015, -0.0025, 0.0], [-0.015, 0.0025, 0.0]])
    with pytest.raises(gfnff.GFNFFError):
        gfnff.parse_turbomole_block("$grad\n$end\n", 2)
    with pytest.raises(gfnff.GFNFFError):
        gfnff.parse_turbomole_block(GRADIENT, 3)


def test_parse_total_energy_and_error_lines():
    from mace.preopt import gfnff
    out = textwrap.dedent("""\
         | TOTAL ENERGY              -12.000000000000 Eh   |
        -1- Could not read topology file
         | TOTAL ENERGY              -12.345678901234 Eh   |
        -2- gfnff_setup: too many atoms with extreme high CN
        """)
    assert gfnff.parse_total_energy(out) == -12.345678901234
    assert gfnff.error_lines(out) == ["-2- gfnff_setup: too many atoms with extreme high CN"]
    with pytest.raises(gfnff.GFNFFError):
        gfnff.parse_total_energy("nothing")


def test_stress_from_gradlatt_matches_finite_differences():
    """sigma = sym(L^T G)/V against the strain derivative of an analytic E(L):
    the spring model's energy at fixed fractional coordinates."""
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.gui import clean as C
    from mace.preopt import gfnff
    a = bulk("NaCl", "rocksalt", a=5.5).repeat((1, 1, 2))
    a.set_cell(a.cell.array @ np.array([[1.0, 0.03, 0.0], [0.0, 1.0, -0.02], [0.01, 0.0, 1.0]]), scale_atoms=True)
    frac = a.get_scaled_positions()
    bonds = C.bonded_pairs(a)

    def energy(L):
        b = a.copy()
        b.set_cell(L, scale_atoms=False)
        b.set_scaled_positions(frac)
        b.calc = C.SpringModel(bonds)
        return b.get_potential_energy()
    L0, h = a.cell.array.copy(), 1e-5
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Lp, Lm = L0.copy(), L0.copy()
            Lp[i, j] += h
            Lm[i, j] -= h
            G[i, j] = (energy(Lp) - energy(Lm)) / (2 * h)
    V = abs(np.linalg.det(L0))
    sigma = gfnff.stress_from_gradlatt(L0, G, V)
    ref = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e = np.zeros((3, 3))
            e[i, j] = e[j, i] = h / 2 if i != j else h
            ref[i, j] = (energy(L0 @ (np.eye(3) + e)) - energy(L0 @ (np.eye(3) - e))) / (2 * h) / V
    ref_v = np.array([ref[0, 0], ref[1, 1], ref[2, 2], ref[1, 2], ref[0, 2], ref[0, 1]])
    assert np.abs(sigma - ref_v).max() < 1e-6, (sigma, ref_v)


@pytest.mark.parametrize("text,want", [
    ("      * xtb version 6.6.1 (8d0f1dd) compiled by 'conda@1efc2f54142f' on 2023-08-01", (6, 6, 1)),
    ("xtb version 6.7.0 (unknown) compiled by 'x' on 2024-02-28", (6, 7, 0)),
    ("   * xtb version 6.7.1 (edcfbbe) compiled by 'conda@3d2fa0d1a6c3' on 2024-07-22", (6, 7, 1)),
    ("xtb version 6.7.2-dev (a1b2c3d) compiled", (6, 7, 2)),
    ("normal termination of xtb", None), ("", None)])
def test_parse_xtb_version(text, want):
    from mace.preopt import detect
    assert detect.parse_xtb_version(text) == want


def fake_exe(d, name, body, py=False):
    p = d / name
    p.write_text((f"#!{sys.executable}\n" if py else "#!/bin/sh\n") + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


@pytest.mark.parametrize("case,status,reason", [
    ("missing", "not_found", "Not found: xtb was not found on PATH"),
    ("661", "too_old", "Found xtb 6.6.1; GFN-FF for periodic cells needs 6.7+"),
    ("671", "ready", ""),
    ("garbage", "broken", "xtb --version failed (no version in its output)"),
    ("rc1", "broken", "xtb --version failed (exit code 1)"),
    ("hang", "broken", "xtb --version gave no answer within 0.5 s")])
def test_xtb_detection_matrix(tmp_path, monkeypatch, case, status, reason):
    from mace.preopt import detect
    d = tmp_path / "bin"
    d.mkdir()
    body = {"661": 'echo "      * xtb version 6.6.1 (8d0f1dd) compiled by x on 2023"\n',
            "671": 'echo "      * xtb version 6.7.1 (edcfbbe) compiled by x on 2024"\n',
            "garbage": "echo hello\n", "rc1": 'echo "xtb version 6.7.1"; exit 1\n', "hang": "sleep 30\n"}.get(case)
    if body:
        fake_exe(d, "xtb", body)
    monkeypatch.setenv("PATH", f"{d}:/usr/bin:/bin")
    monkeypatch.setattr(detect, "XTB_TIMEOUT", 0.5)
    t0 = time.monotonic()
    st = detect.xtb_status()
    assert st["status"] == status and st["reason"] == reason, st
    assert time.monotonic() - t0 < 5
    if status != "ready":
        assert 'conda install -c conda-forge "xtb>=6.7"' in st["hint"] and "MACE_XTB" in st["hint"]
    if case == "671":                         # cached by (path, realpath, mtime, size)
        (d / "xtb").write_text("#!/bin/sh\nexit 3\n")
        assert detect.xtb_status()["status"] == "broken"     # a changed file is looked at again


def test_mace_xtb_overrides_path(tmp_path, monkeypatch):
    from mace.preopt import detect
    exe = fake_exe(tmp_path, "my-xtb", 'echo "xtb version 6.7.1 (x)"\n')
    monkeypatch.setenv("MACE_XTB", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert detect.xtb_status()["status"] == "ready" and detect.xtb_path() == str(exe)
    monkeypatch.setenv("MACE_XTB", str(tmp_path / "nope"))
    st = detect.xtb_status()
    assert st["status"] == "not_found" and "MACE_XTB=" in st["reason"]


def test_hpcc_wrapper_from_the_docs_is_a_working_mace_xtb(tmp_path, monkeypatch):
    """On MSU HPCC the xtb module can't be loaded beside MACE's Python module (Lmod refuses
    gfbf/2024a next to gfbf/2023a), and its binary fails without the module's libraries. The
    HPCC notes in docs/preopt.md write a wrapper that sets that library path; built with the
    docs' own printf line, the wrapper is what MACE_XTB finds."""
    import re
    import shutil
    from mace.preopt import detect
    from preopt_fixtures import REPO
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    doc = (REPO / "docs" / "preopt.md").read_text()
    line = next(x.strip() for x in doc.splitlines() if x.strip().startswith("printf '#!/bin/sh"))
    line = re.sub(r"\s*&&$", "", line).replace("~/bin/xtb-gfnff", '"$OUT"')
    lib, mod = tmp_path / "module" / "lib", tmp_path / "module" / "bin"
    lib.mkdir(parents=True)
    mod.mkdir(parents=True)
    # the module's xtb: it runs only with the module's library directory on LD_LIBRARY_PATH
    fake_exe(mod, "xtb", f'case ":$LD_LIBRARY_PATH:" in *":{lib}:"*) ;;\n'
                         '  *) echo "xtb: error while loading shared libraries: libflexiblas.so.3" >&2; exit 127 ;;\nesac\n'
                         'echo "      * xtb version 6.7.1 (edcfbbe) compiled by x on 2024"\n')
    wrapper = tmp_path / "xtb-gfnff"
    # "module load xtb/...": the module's bin on PATH and its libraries on LD_LIBRARY_PATH
    env = {"PATH": f"{mod}:/usr/bin:/bin", "LD_LIBRARY_PATH": f"{lib}:/opt/other/lib", "OUT": str(wrapper)}
    subprocess.run(["bash", "-c", line], env=env, check=True, timeout=30)
    wrapper.chmod(0o755)
    # back in MACE's environment (the Python module's library path, no xtb on PATH)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/python/lib")
    monkeypatch.setenv("MACE_XTB", str(mod / "xtb"))
    st = detect.xtb_status(recheck=True)
    assert st["status"] == "broken" and "127" in st["reason"], st       # the bare module binary
    monkeypatch.setenv("MACE_XTB", str(wrapper))
    st = detect.xtb_status(recheck=True)
    assert st["status"] == "ready" and st["version"] == "6.7.1" and st["path"] == str(wrapper), st
    assert "export MACE_XTB=~/bin/xtb-gfnff" in doc


def test_xtb_hint_never_offers_the_module_load_that_fails_on_hpcc():
    from mace.preopt import detect
    assert "module load" not in detect.XTB_HINT
    assert not [c for c in detect.hint_commands(detect.XTB_HINT) if c.startswith("module")]
    assert "HPCC notes in docs/preopt.md" in detect.XTB_HINT


def test_child_env_inherits_then_denies(monkeypatch):
    from mace.preopt import proc
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.setenv("PYTHONHOME", "/x")
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setenv("MACE_HOME", "/m")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
    monkeypatch.setenv("MODULEPATH", "/mods")
    env = proc.child_env(OMP_NUM_THREADS=1, TMPDIR="/t")
    assert not [k for k in env if k.startswith(("PYTHON", "MACE_"))] and "VIRTUAL_ENV" not in env
    assert env["LD_LIBRARY_PATH"] == "/opt/lib" and env["MODULEPATH"] == "/mods"
    assert env["OMP_NUM_THREADS"] == "1" and env["TMPDIR"] == "/t"


FAKE_XTB = r'''
import os, sys, time
log = os.environ.get("FAKE_XTB_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\n")
mode = os.environ.get("FAKE_XTB_MODE", "ok")
if mode == "sleep":
    with open(os.environ["FAKE_XTB_PID"], "w") as fh:
        fh.write(str(os.getpid()))
    time.sleep(30)
if mode == "fail":
    print("-1- Could not read topology file")
    print("-2- gfnff_setup: too many atoms with extreme high CN")
    sys.exit(1)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
open(os.path.join(os.environ["TMPDIR"], "litter"), "w").write("x")
open("gfnff_topo", "w").write("topology")
lines = open("POSCAR").read().splitlines()
n = sum(int(x) for x in lines[6].split())
print("          | TOTAL ENERGY              -5.000000000000 Eh   |")
with open("gradient", "w") as fh:
    fh.write("$grad\n  cycle =      1    SCF energy =    -5.0  |dE/xyz| =  0.0\n")
    for i in range(n):
        fh.write("  0.0  0.0  0.0  x\n")
    for i in range(n):
        fh.write("  0.0D+00  0.0D+00  0.0D+00\n")
    fh.write("$end\n")
'''


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    d = tmp_path / "fx"
    d.mkdir()
    exe = fake_exe(d, "xtb", FAKE_XTB, py=True)
    monkeypatch.setenv("FAKE_XTB_LOG", str(tmp_path / "argv.log"))
    return exe


def test_gfnff_calculator_charge_topology_and_cleanup(fake_xtb, tmp_path, preopt_sandbox):
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.preopt import gfnff
    calc = gfnff.GFNFF(xtb=str(fake_xtb), charge=-1, uhf=0)
    wd = calc.workdir
    a = bulk("Si")
    a.calc = calc
    assert abs(a.get_potential_energy() + 5.0 * 27.211386) < 1e-3
    a.positions[0] += 0.01
    a.get_forces()
    argv = (tmp_path / "argv.log").read_text().splitlines()
    assert argv == ["POSCAR --gfnff --grad --chrg -1 --uhf 0"] * 2
    assert "timeout" in calc.argv_seen[0] and calc.argv_seen[1] == "--kill-after=2"
    assert (wd / "tmp" / "litter").exists()                 # xtb's own litter lands in the private dir
    calc.close()
    assert not wd.exists()


def test_gfnff_failure_message(fake_xtb, monkeypatch):
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.preopt import gfnff
    monkeypatch.setenv("FAKE_XTB_MODE", "fail")
    a = bulk("MgO", "rocksalt", a=4.2)
    calc = gfnff.GFNFF(xtb=str(fake_xtb))
    a.calc = calc
    with pytest.raises(gfnff.GFNFFError) as e:
        a.get_potential_energy()
    msg = str(e.value)
    assert msg.startswith("GFN-FF can't type this structure (common for MgO-like oxides and halides)")
    assert "too many atoms with extreme high CN" in msg and "topology file" not in msg
    calc.close()


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_cancel_stops_a_hung_xtb_within_2_5_s(fake_xtb, tmp_path, monkeypatch):
    pytest.importorskip("ase")
    import threading
    from ase.build import bulk
    from mace.preopt import gfnff, proc
    pidf = tmp_path / "xtb.pid"
    monkeypatch.setenv("FAKE_XTB_MODE", "sleep")
    monkeypatch.setenv("FAKE_XTB_PID", str(pidf))
    flag = threading.Event()
    calc = gfnff.GFNFF(xtb=str(fake_xtb), canceled=flag.is_set)
    a = bulk("Si")
    a.calc = calc
    threading.Timer(1.0, flag.set).start()
    t0 = time.monotonic()
    with pytest.raises(proc.Canceled):
        a.get_potential_energy()
    took = time.monotonic() - t0
    assert took < 1.0 + 2.5, took
    pid = int(pidf.read_text())
    assert _gone(pid)
    calc.close()


def test_an_interrupt_stops_xtb_too(fake_xtb, tmp_path, monkeypatch):
    """Ctrl-C in `mace preopt` reaches only the CLI (xtb runs in its own
    session): the wait must still stop xtb, which would otherwise keep
    running in a work directory that close() has removed."""
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.preopt import gfnff
    pidf = tmp_path / "xtb.pid"
    monkeypatch.setenv("FAKE_XTB_MODE", "sleep")
    monkeypatch.setenv("FAKE_XTB_PID", str(pidf))
    t0 = time.monotonic()

    def interrupt():
        if pidf.exists() and time.monotonic() - t0 > 0.5:
            raise KeyboardInterrupt
        return False
    calc = gfnff.GFNFF(xtb=str(fake_xtb), canceled=interrupt)
    a = bulk("Si")
    a.calc = calc
    with pytest.raises(KeyboardInterrupt):
        a.get_potential_energy()
    assert time.monotonic() - t0 < 0.5 + 2.5
    assert _gone(int(pidf.read_text()))
    calc.close()


def test_per_call_timeout_kills_xtb(fake_xtb, tmp_path, monkeypatch):
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.preopt import gfnff
    monkeypatch.setenv("FAKE_XTB_MODE", "sleep")
    monkeypatch.setenv("FAKE_XTB_PID", str(tmp_path / "xtb.pid"))
    calc = gfnff.GFNFF(xtb=str(fake_xtb), per_call_timeout=1.0)
    a = bulk("Si")
    a.calc = calc
    t0 = time.monotonic()
    with pytest.raises(gfnff.GFNFFError):
        a.get_potential_energy()
    assert time.monotonic() - t0 < 6.5
    assert _gone(int((tmp_path / "xtb.pid").read_text()))
    calc.close()


def test_orphaned_xtb_is_bounded_by_the_timeout_wrapper_and_the_sweeper_waits(fake_xtb, tmp_path, monkeypatch):
    """A parent killed with SIGKILL leaves xtb running under `timeout`; the
    sweeper keeps its live directory, and removes it once xtb has stopped."""
    pytest.importorskip("ase")
    from mace.preopt import proc
    pidf = tmp_path / "xtb.pid"
    script = textwrap.dedent(f"""
        import sys, tempfile
        sys.path.insert(0, {repr(os.getcwd())})
        tempfile.tempdir = {repr(tempfile_dir())}
        from ase.build import bulk
        from mace.preopt import gfnff
        c = gfnff.GFNFF(xtb={repr(str(fake_xtb))}, per_call_timeout=3.0)
        a = bulk("Si"); a.calc = c
        a.get_potential_energy()
    """)
    env = dict(os.environ, FAKE_XTB_MODE="sleep", FAKE_XTB_PID=str(pidf))
    env.pop("PYTHONPATH", None)
    parent = subprocess.Popen([sys.executable, "-c", script], env=env)
    end = time.monotonic() + 20
    while not pidf.exists() and time.monotonic() < end:
        time.sleep(0.05)
    xtb_pid = int(pidf.read_text())
    parent.kill()
    parent.wait()
    root = proc.temp_root()
    assert proc.sweep() == 0 and len(list(root.iterdir())) == 1      # xtb still runs in it
    end = time.monotonic() + 10
    while not _gone(xtb_pid) and time.monotonic() < end:
        time.sleep(0.1)
    assert _gone(xtb_pid)                                            # timeout 3 s + kill-after 2 s
    time.sleep(0.2)
    assert proc.sweep() == 1 and not list(root.iterdir())


def tempfile_dir():
    import tempfile
    return tempfile.gettempdir()


def test_sweeper_removes_dead_dirs_and_keeps_live_ones():
    from mace.preopt import proc
    root = proc.temp_root()
    dead = subprocess.Popen(["true"])
    dead.wait()
    (root / f"{dead.pid}-gfnff-abc").mkdir()
    mine = proc.private_dir("gfnff")
    assert proc.sweep() == 1 and mine.exists()
    proc.remove(mine)


def test_temp_root_is_private():
    from mace.preopt import proc
    r = proc.temp_root()
    assert (r.stat().st_mode & 0o777) == 0o700 and r.name == f"mace-preopt-{os.getuid()}"
    d = proc.private_dir("x y")
    assert (d.stat().st_mode & 0o777) == 0o700 and d.name.startswith(f"{os.getpid()}-x_y-")
    proc.remove(d)


def test_hint_commands_are_whole_runnable_lines():
    from mace.preopt import detect, registry
    assert detect.hint_commands(detect.XTB_HINT) == [
        'conda install -c conda-forge "xtb>=6.7"', "export MACE_XTB=/path/to/xtb"]
    mpa = registry.get(registry.default_id())
    assert detect.hint_commands(detect.setup_hint(mpa)) == [
        "mace mlip setup", "mace mlip setup --python DIR/bin/python", f"mace mlip fetch {mpa['id']}"]
    assert detect.hint_commands(f"install it in the ML environment: {mpa['code']['pin']}") == [
        f"DIR/bin/pip install {mpa['code']['pin']}"]
    assert detect.hint_commands("") == []
