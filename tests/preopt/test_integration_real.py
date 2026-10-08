"""The real tools, when they are installed (A5-A9, A11, A13b, A20): skipped,
with the reason, when xtb >= 6.7 or the ML environment is missing.

They run with the tools the suite was started with: MACE_XTB (else xtb on
PATH), MACE_MLIP_PYTHON and MACE_MLIP_DIR (the MACE-MPA-0 weights fetched
with `mace mlip fetch`). Every structure is synthetic. Timings are printed
(run with -s to see them)."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

import preopt_fixtures as F  # noqa: E402
from preopt_fixtures import REAL, preopt_sandbox  # noqa: E402,F401

MACE_CLI = F.REPO / "mace_cli"


def _xtb():
    path = REAL.get("MACE_XTB") or shutil.which("xtb")
    if not path:
        pytest.skip("xtb is not installed (set MACE_XTB or put xtb >= 6.7 on PATH)")
    from mace.preopt import detect
    os.environ["MACE_XTB"] = path
    st = detect.xtb_status(recheck=True)
    if st["status"] != "ready":
        pytest.skip(f"xtb is not usable for GFN-FF: {st['reason']}")
    return path


def _mlip(mid):
    if not REAL.get("MACE_MLIP_PYTHON"):
        pytest.skip("no ML environment (set MACE_MLIP_PYTHON)")
    os.environ["MACE_MLIP_PYTHON"] = REAL["MACE_MLIP_PYTHON"]
    if REAL.get("MACE_MLIP_DIR"):
        os.environ["MACE_MLIP_DIR"] = REAL["MACE_MLIP_DIR"]
    from mace.preopt import detect, registry
    st = detect.mlip_status(registry.get(mid))
    if st["status"] != "ready":
        pytest.skip(f"{mid} is not ready: {st['status']} {st.get('reason', '')}")
    return REAL["MACE_MLIP_PYTHON"]


def cli(*args, cwd, env=None, timeout=600):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(env or {})
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "preopt", *args], env=e, cwd=str(cwd),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr, time.monotonic() - t0


def write_cif(path, atoms):
    import ase.io
    ase.io.write(str(path), atoms, format="cif")
    return path


def sg(atoms, prec=1e-3):
    from mace.preopt import core
    g = core.spacegroup(atoms, prec)
    return g["number"] if g else None


# --- GFN-FF ------------------------------------------------------------------------------

def test_gfnff_topology_is_rebuilt_every_call():
    """A5: the same Si with one atom moved by a whole lattice vector gives the
    fresh energy (xtb's own topology reuse gave +30 eV here)."""
    path = _xtb()
    from ase.build import bulk
    from mace.preopt import gfnff
    a = bulk("Si", "diamond", a=5.43, cubic=True)
    a.positions += np.random.default_rng(1).normal(0, 0.02, a.positions.shape)
    c = gfnff.GFNFF(xtb=path)
    try:
        a.calc = c
        e1 = a.get_potential_energy()
        b = a.copy()
        b.positions[0] += a.cell.array[0]
        b.calc = c
        e2 = b.get_potential_energy()
        fresh = gfnff.GFNFF(xtb=path)
        b2 = b.copy()
        b2.calc = fresh
        e3 = b2.get_potential_energy()
        fresh.close()
    finally:
        c.close()
    assert abs(e1 - e2) < 1e-6 and abs(e2 - e3) < 1e-6, (e1, e2, e3)


@pytest.mark.parametrize("name,ideal_sg", [("Si", 227), ("NaCl", 225)])
def test_gfnff_relaxes_rattled_cells_downhill_to_ideal(name, ideal_sg):
    path = _xtb()
    from mace.gui import clean as C
    from mace.preopt import gfnff
    ideal = F.demo(f"{name}.cif")
    rattled = ideal.copy()
    rattled.positions += np.random.default_rng(5).normal(0, 0.05, rattled.positions.shape)
    C.ADAPTERS["gfnff-test"] = lambda atoms, **kw: gfnff.GFNFF(xtb=path)
    try:
        t0 = time.monotonic()
        r = C.clean(rattled, method="gfnff-test", keep_symmetry=False, fmax=0.005, max_steps=500, time_limit=120)
        dt = time.monotonic() - t0
        ref = ideal.copy()
        ref.calc = gfnff.GFNFF(xtb=path)
        e_ideal = ref.get_potential_energy()
        ref.calc.close()
    finally:
        C.ADAPTERS.pop("gfnff-test", None)
    print(f"\n[timing] GFN-FF rattled {name}: {r['steps']} steps, {dt:.1f} s")
    assert r["converged"] and r["energy_after"] < r["energy_before"]
    assert abs(r["energy_after"] - e_ideal) / len(ideal) < 1e-3, (r["energy_after"], e_ideal)
    assert sg(r["atoms"], 1e-2) == ideal_sg


def test_gfnff_mgo_fails_cleanly_and_writes_the_springs(tmp_path):
    """A6: MgO can't be typed by GFN-FF: exit 3 within 30 s, the reason, the spring CIF."""
    _xtb()
    shutil.copy(F.EXAMPLES / "MgO.cif", tmp_path / "MgO.cif")
    rc, out, err, dt = cli("MgO.cif", "--method", "gfnff", cwd=tmp_path)
    print(f"\n[timing] GFN-FF MgO refusal path: {dt:.1f} s")
    assert rc == 3 and dt < 30, (rc, out, err)
    assert "GFN-FF can't type this structure (common for MgO-like oxides and halides)" in out
    assert (tmp_path / "MgO_preopt.cif").exists()
    assert json.loads((tmp_path / "MgO_preopt.json").read_text())["method_ran"] == "springs"


def test_gfnff_forced_cell_relax_on_nacl_is_never_accepted(tmp_path):
    """A7: a bad relax (GFN-FF cell relaxation of NaCl) is rejected or fails."""
    _xtb()
    shutil.copy(F.EXAMPLES / "NaCl.cif", tmp_path / "NaCl.cif")
    rc, out, err, dt = cli("NaCl.cif", "--method", "gfnff", "--relax-cell", cwd=tmp_path)
    p = json.loads((tmp_path / "NaCl_preopt.json").read_text())
    att = p["attempts"][-1]
    print(f"\n[result] GFN-FF NaCl --relax-cell: {att['status']} ({att['reason']}), {dt:.1f} s")
    if p["method_ran"] == "gfnff":           # accepted only if every check passed (|dV| <= 10 %, ...)
        assert abs(att["dV_pct"]) <= 10 and att["converged"]
    else:
        assert rc == 3 and att["status"] in ("rejected", "failed")


def test_gfnff_fsi_box_with_charge_converges(tmp_path):
    """A8/A20: the synthetic FSI- with --charge -1 converges; the charge is recorded."""
    _xtb()
    write_cif(tmp_path / "fsi.cif", F.fsi_box())
    rc, out, err, dt = cli("fsi.cif", "--method", "gfnff", "--charge", "-1", cwd=tmp_path)
    p = json.loads((tmp_path / "fsi_preopt.json").read_text())
    print(f"\n[timing] GFN-FF FSI- (charge -1): {p['steps_taken']} steps, {dt:.1f} s; {p['summary']}")
    assert rc == 0, (out, err)
    assert p["method_ran"] == "gfnff" and p["charge"] == -1 and p["converged"]
    assert p["uhf"].endswith("(recorded only; GFN-FF ignores it)")


def test_gfnff_li_fsi_pair_converges_under_200_steps(tmp_path):
    _xtb()
    from ase import Atoms
    a = F.fsi_box()
    a.set_cell([17.0] * 3)                   # 15 Å leaves 9.5 Å between the pair's images: under GFN-FF's 10 Å
    a.positions += 1.0
    mid = (a.positions[3] + a.positions[4]) / 2                # Li+ on the two O of one SO2 group
    li = Atoms("Li", positions=[2 * mid - a.positions[1]])
    pair = a + li
    write_cif(tmp_path / "lifsi.cif", pair)
    rc, out, err, dt = cli("lifsi.cif", "--method", "gfnff", "--charge", "0", "--steps", "200", cwd=tmp_path)
    p = json.loads((tmp_path / "lifsi_preopt.json").read_text())
    print(f"\n[timing] GFN-FF Li+FSI- pair: {p['steps_taken']} steps, {dt:.1f} s; {p['summary']}")
    assert rc == 0 and p["method_ran"] == "gfnff" and p["converged"] and p["steps_taken"] < 200, p["summary"]


def test_gfnff_cancel_stops_xtb_and_leaves_nothing():
    """A9/A10 with the real xtb: a cancel mid-run, no xtb left, no temp dir."""
    _xtb()
    from mace.preopt import core, proc
    big = F.demo("NaCl.cif").repeat((3, 3, 3))
    big.positions += np.random.default_rng(2).normal(0, 0.05, big.positions.shape)
    flag = threading.Event()
    threading.Timer(12.0, flag.set).start()            # after the 10 s springs baseline, inside GFN-FF
    t0 = time.monotonic()
    with pytest.raises(proc.Canceled):
        core.preoptimize(big, "gfnff", keep_symmetry=False, canceled=flag.is_set, time_limit=120)
    assert time.monotonic() - t0 < 12.0 + 2.5 + 1.0
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "--gfnff --grad"], capture_output=True, text=True)
    mine = [p for p in out.stdout.split() if p.strip()]
    assert not mine, mine


def test_gui_and_cli_gfnff_coordinates_agree(tmp_path):
    """A13(b): GUI POST /api/clean and `mace preopt` give the same GFN-FF result
    within 1e-6 Å (the FSI- box, charge -1)."""
    path = _xtb()
    import ase.io
    sys.path.insert(0, str(F.REPO / "tests" / "gui"))
    import guiharness as H
    import p4util as U
    write_cif(tmp_path / "fsi.cif", F.fsi_box())
    a = ase.io.read(str(tmp_path / "fsi.cif"))
    rc, out, err, _ = cli("fsi.cif", "--method", "gfnff", "--charge", "-1", "--time-limit", "120", cwd=tmp_path)
    assert rc == 0, (out, err)
    cli_atoms = ase.io.read(str(tmp_path / "fsi_preopt.cif"))
    root, xdg = H.examples_copy(tmp_path / "srv"), tmp_path / "xdg"
    proc_, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_XTB=path)
    try:
        end = time.time() + 60
        while time.time() < end and U.req(base, "api/clean/methods")[2].get("checking"):
            time.sleep(0.2)
        st, _, s = U.post(base, "api/edit/from_structure", {"name": "fsi", "cell": a.cell.array.tolist(),
                                                           "frac": a.get_scaled_positions().tolist(),
                                                           "numbers": a.numbers.tolist()})
        st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], "method": "gfnff",
                                              "charge": -1, "time_limit": 120})
        import test_clean_tiers_api as A
        r = A.result_of(base, st, r, timeout=300)
    finally:
        proc_.terminate()
        proc_.wait(timeout=20)
    assert r["method_ran"] == "gfnff", r.get("ran")
    gui = np.asarray(r["displacements"]) + np.asarray(r["before_cart"])
    fb = np.linalg.solve(a.cell.array.T, gui.T).T
    d = cli_atoms.get_scaled_positions() - fb
    d -= np.round(d)
    assert np.abs(d @ a.cell.array).max() < 1e-6


# --- ML potentials ------------------------------------------------------------------------

MPA = "acesuit-mace-mpa-0-medium"
CHG = "chgnet-0.3.0"


def _lattice_fcc(atoms):
    return (4 * atoms.get_volume() / (len(atoms) / 2)) ** (1 / 3)


@pytest.mark.parametrize("mid,a_ref", [(MPA, 4.257), (CHG, 4.255)])
def test_ml_mgo_cell_relax(tmp_path, mid, a_ref):
    """A20: MgO with --relax-cell: a = a_ref ± 0.01 Å, Fm-3m. It starts from the
    experimental 4.21 Å: from 4.08 Å the volume grows 13.6 %, which the |dV| <=
    10 % check rejects (test_ml_mgo_large_volume_change_is_rejected)."""
    _mlip(mid)
    from ase.build import bulk
    write_cif(tmp_path / "mgo.cif", bulk("MgO", "rocksalt", a=4.21))
    rc, out, err, dt = cli("mgo.cif", "--method", f"mlip:{mid}", "--relax-cell", "--threads", "4",
                           cwd=tmp_path, timeout=900)
    p = json.loads((tmp_path / "mgo_preopt.json").read_text())
    import ase.io
    res = ase.io.read(str(tmp_path / "mgo_preopt.cif"))
    a = _lattice_fcc(res)
    print(f"\n[timing] {mid} MgO --relax-cell: a = {a:.4f} Å, {p['steps_taken']} steps, {dt:.1f} s total; {p['summary']}")
    assert rc == 0, (out, err)
    assert abs(a - a_ref) < 0.01 and sg(res) == 225
    assert p["model"] == mid and p["versions"] and p["device"] == "cpu"


def test_ml_mgo_large_volume_change_is_rejected(tmp_path):
    """From a = 4.08 Å the ML cell relaxation reaches 4.257 Å: +13.6 % in volume,
    over the 10 % limit, so the spring result is kept (exit 3) and the reason shown."""
    _mlip(MPA)
    from ase.build import bulk
    write_cif(tmp_path / "mgo.cif", bulk("MgO", "rocksalt", a=4.08))
    rc, out, err, dt = cli("mgo.cif", "--method", f"mlip:{MPA}", "--relax-cell", cwd=tmp_path, timeout=900)
    p = json.loads((tmp_path / "mgo_preopt.json").read_text())
    att = p["attempts"][-1]
    print(f"\n[result] MPA-0 MgO from 4.08 Å: {att['status']} ({att['reason']}), {dt:.1f} s")
    assert rc == 3 and att["status"] == "rejected" and att["reason"].startswith("volume changed by +13")


def test_ml_rattled_nacl_symmetrize_first(tmp_path):
    _mlip(MPA)
    nacl = F.demo("NaCl.cif")
    r = nacl.copy()
    r.set_cell(nacl.cell.array * 1.03, scale_atoms=True)
    r.positions += np.random.default_rng(3).normal(0, 0.03, r.positions.shape)
    write_cif(tmp_path / "r.cif", r)
    rc, out, err, dt = cli("r.cif", "--method", f"mlip:{MPA}", "--symmetrize", "0.3", cwd=tmp_path, timeout=900)
    p = json.loads((tmp_path / "r_preopt.json").read_text())
    print(f"\n[timing] MPA-0 rattled NaCl, symmetrize 0.3: {dt:.1f} s; {p['summary']}")
    assert rc == 0, (out, err)
    assert p["space_group"]["reference"] == "Fm-3m" and p["space_group"]["result"] == "Fm-3m"
    assert "by symmetrize-first (0.3 Å)" in p["symmetrize"]["text"]


def test_ml_na_cl_pair_stays_bound(tmp_path):
    _mlip(MPA)
    from ase import Atoms
    pair = Atoms("NaCl", positions=[[7.5, 7.5, 6.0], [7.5, 7.5, 9.0]], cell=[15.0] * 3, pbc=True)
    write_cif(tmp_path / "pair.cif", pair)
    rc, out, err, dt = cli("pair.cif", "--method", f"mlip:{MPA}", cwd=tmp_path, timeout=900)
    import ase.io
    res = ase.io.read(str(tmp_path / "pair_preopt.cif"))
    d = res.get_distance(0, 1, mic=True)
    print(f"\n[result] MPA-0 Na-Cl pair from 3.0 Å: {d:.3f} Å, {dt:.1f} s")
    assert rc == 0 and 2.2 < d < 3.0


def test_ml_tilted_perovskite_never_gains_symmetry(tmp_path):
    """A20: an I4/mcm (a0a0c-) SrTiO3-like cell keeps its group: a result that
    untilted it to a higher group would be rejected, never written."""
    _mlip(MPA)
    from ase.spacegroup import crystal
    sto = crystal(["Sr", "Ti", "O", "O"], basis=[(0, 0.5, 0.25), (0, 0, 0), (0, 0, 0.25), (0.235, 0.735, 0)],
                  spacegroup=140, cellpar=[5.52, 5.52, 7.81, 90, 90, 90])
    write_cif(tmp_path / "sto.cif", sto)
    rc, out, err, dt = cli("sto.cif", "--method", f"mlip:{MPA}", cwd=tmp_path, timeout=900)
    import ase.io
    res = ase.io.read(str(tmp_path / "sto_preopt.cif"))
    p = json.loads((tmp_path / "sto_preopt.json").read_text())
    print(f"\n[result] MPA-0 tilted perovskite: {p['summary']}, {dt:.1f} s")
    assert sg(res) == 140


@pytest.mark.parametrize("mid", [MPA, CHG])
def test_ml_stress_matches_finite_differences(mid):
    """A20: the worker's stress (units and sign) against strain derivatives of its energy."""
    py = _mlip(mid)
    from ase.build import bulk
    from mace.preopt import detect, mlip_client, registry
    m = registry.get(mid)
    ws = detect.weights_status(m)
    w = mlip_client.WorkerProcess(py, m, weights=ws.get("path"), threads=4, dtype="float64")
    try:
        c = mlip_client.MlipCalculator(w)
        a = bulk("MgO", "rocksalt", a=4.15)
        a.set_cell(a.cell.array @ np.array([[1.0, 0.02, 0], [0, 1.0, 0.01], [0, 0, 1.0]]), scale_atoms=True)
        a.calc = c
        s = a.get_stress(voigt=False)
        V, h = a.get_volume(), 1e-3             # CHGNet computes in float32: a larger step
        L0 = a.cell.array.copy()
        fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                e = np.zeros((3, 3))
                e[i, j] = e[j, i] = h / 2 if i != j else h
                es = []
                for sign in (1, -1):
                    b = a.copy()
                    b.set_cell(L0 @ (np.eye(3) + sign * e), scale_atoms=True)
                    b.calc = c
                    es.append(b.get_potential_energy())
                fd[i, j] = (es[0] - es[1]) / (2 * h) / V
    finally:
        w.close()
    print(f"\n[result] {mid} stress vs FD: max |diff| {np.abs(s - fd).max():.2e} eV/Å³")
    assert np.abs(s - fd).max() < 1e-3


def test_ml_worker_imports_mace_torch_even_with_the_repo_on_pythonpath(monkeypatch):
    """A11: the parent's PYTHONPATH (this repo) never reaches the worker."""
    py = _mlip(MPA)
    from mace.preopt import detect, mlip_client, registry
    monkeypatch.setenv("PYTHONPATH", str(F.REPO))
    m = registry.get(MPA)
    w = mlip_client.WorkerProcess(py, m, weights=detect.weights_status(m)["path"], threads=2)
    try:
        mod = w.hello["module"]
        assert mod and str(F.REPO) not in mod and "site-packages" in mod
        r = subprocess.run([py, "-I", "-c", "import importlib.metadata as md, os, sys;"
                            "d = md.distribution('mace-torch');"
                            "print(any(os.path.realpath(str(d.locate_file(f))) == sys.argv[1] for f in d.files))", mod],
                           capture_output=True, text=True, timeout=120)
        assert r.stdout.strip() == "True"
    finally:
        w.close()
