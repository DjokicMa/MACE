"""The orbital picker's server side (mace/gui/orbitals.py, spec 4.4) on the
repo's own CRYSTAL23 ORBITALS fixtures (tests/data/orbitals).

The jobs: the index is ``api.describe`` plus the picker's extras (energy
reference, occupation words, default supercell); a grid is ``run.grid`` in
the Volume document's ``{meta, grid}`` shape with the reader's 90 % isovalue
as its level; a save is ``api.make_cube`` in ``<job>_orbitals/``. The routes
keep the server's path and token rules, run in the pool, and never write a
.molden file."""
import hashlib
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
DATA = Path(__file__).resolve().parents[1] / "data" / "orbitals"
H2O = str(DATA / "h2o" / "h2o_c.molden")
SIG = str(DATA / "si_g" / "si_g_00001_K000_real.molden")
SIF = str(DATA / "si_f" / "si_f_00003_K200_real.molden")
SIF_K = str(DATA / "si_f" / "si_f_00004_K110_real.molden")
PY = str(DATA / "polyyne" / "py_f_00001_K000_real.molden")
FE = str(DATA / "fe" / "fe_f_00001_K000_real.molden")
O2 = str(DATA / "o2" / "o2_c.molden")


def test_molden_is_the_orbitals_kind():
    from mace.gui import filekind
    assert filekind.kind_by_name("h2o_c.molden") == "orbitals"
    assert filekind.kind_by_name("si_orb_00001_K000_real.MOLDEN") == "orbitals"
    # the kinds listed before are unchanged
    assert filekind.kind_by_name("x_DENS.CUBE") == "cube" and filekind.kind_by_name("x.out") == "out"


def test_index_k_list_levels_and_words():
    from mace.gui import orbitals as O
    ix = O.index_job(path=SIF)
    assert ix["selected"] == 3 and ix["dimensionality"] == 3 and ix["default_cells"] == [2, 2, 2]
    ks = {e["index"]: e for e in ix["kpoints"]}
    assert ks[3]["usable"] and ks[3]["real"]
    assert not ks[4]["usable"] and not ks[4]["real"] and "complex k point in a 2D/3D system" in ks[4]["reason"]
    sp = ix["k"]["spins"][0]
    assert sp["spin"] == "ALPHA+BETA" and sp["ref_kind"] == "HOMO" and sp["default_mo"] == sp["homo"] == 14
    lv = {l["n"]: l for l in sp["levels"]}
    assert lv[14]["rel_ev"] == 0.0 and lv[15]["rel_ev"] > 0
    # periodic closed shell: two electrons per band in words, never "1.0"
    assert lv[14]["occ_text"] == "2 e (closed shell)" and lv[15]["occ_text"] == "empty"
    assert lv[13]["degenerate"] == lv[14]["degenerate"] == 2 and lv[13]["group"] == lv[14]["group"]
    # the file of the unusable k point opens at the usable one, saying why
    ix = O.index_job(path=SIF_K, k=4)
    assert ix["selected"] == 3 and ix["asked"]["k"] == 4 and "orthonormality" in ix["asked"]["reason"]


def test_index_open_shell_metal_and_polymer():
    from mace.gui import orbitals as O
    ix = O.index_job(path=O2)
    assert [s["spin"] for s in ix["k"]["spins"]] == ["ALPHA", "BETA"] and ix["default_cells"] == [1, 1, 1]
    a, b = ix["k"]["spins"]
    assert a["homo"] == 9 and b["homo"] == 7
    assert next(l for l in a["levels"] if l["n"] == 9)["occ_text"] == "1 e"
    fe = O.index_job(path=FE)
    s = fe["k"]["spins"][0]
    assert s["metallic"] and s["ref_kind"] == "E_F" and abs(s["ref_ev"] - fe["efermi"] * O.HARTREE_EV) < 1e-9
    assert s["below_ef"] is not None
    py = O.index_job(path=PY)
    assert py["default_cells"] == [4, 1, 1]
    assert [e["usable"] for e in py["kpoints"]] == [True, True, True] and not py["kpoints"][1]["real"]


def test_grid_payload_is_the_volume_shape():
    from mace.gui import orbitals as O
    from mace.orbitals import api
    from mace.orbitals.dataset import default_isovalue
    r = O.grid_job(path=H2O, k=1, mos=["HOMO"], part="re", cells=[1, 1, 1], quality="preview")
    m, g = r["meta"], np.frombuffer(r["grid"], "<f4")
    assert m["type"] == "orbital" and m["signed"] and m["orbital"]["mos"] == [5] and m["orbital"]["part"] == "re"
    assert g.size == int(np.prod(m["shape"])) and m["nbytes"] == g.nbytes and m["stride"] == 1
    assert not m["periodic"] and m["pbc"] == [False, False, False] and m["atoms"]["symbols"] == ["O", "H", "H"]
    # the level is the reader's 90 % isovalue of the same data
    run = api.open_run(H2O)
    grid, data, meta = run.grid(1, "ALPHA+BETA", [5], part="re", quality="preview", max_points=O.QUALITY_POINTS["preview"])
    assert np.allclose(g.reshape(m["shape"]), data, atol=1e-6)
    assert abs(m["level"] - default_isovalue(data, True)) < 1e-12 and m["level"] == m["orbital"]["isovalue"]
    assert m["voxels"][0][0] == pytest.approx(grid.vecs[0][0] * O.BOHR, abs=1e-9)
    assert "MO 5 (HOMO)" in m["name"] and "Re ψ" in m["name"]


def test_degenerate_sum_is_abs2_and_the_box_holds_images():
    from mace.gui import orbitals as O
    r = O.grid_job(path=SIG, k=1, mos=[13], part="re", cells=[2, 2, 2], quality="preview", degenerate_sum=True)
    m, g = r["meta"], np.frombuffer(r["grid"], "<f4")
    o = m["orbital"]
    assert o["mos"] == [12, 13, 14] and o["part"] == "abs2" and not m["signed"] and o["degenerate_sum"]
    assert g.min() >= 0 and o["cells"] == [2, 2, 2] and m["periodic"] and m["pbc"] == [True, True, True]
    # atoms and their images inside the 2x2x2 box (2 atoms per primitive cell: at least 16)
    assert len(m["atoms"]["numbers"]) >= 16 and set(m["atoms"]["symbols"]) == {"Si"}
    # the cell drawn is the supercell box: twice each primitive vector
    from mace.orbitals import api
    A = api.open_run(SIG).load(1).basis.lattice_axes() * O.BOHR
    assert np.allclose(np.asarray(m["lattice"]), 2 * A, atol=1e-4)
    # one member alone: signed, with the gauge note
    r = O.grid_job(path=SIG, k=1, mos=[13], part="re", cells=[1, 1, 1], quality="preview")
    assert r["meta"]["signed"] and r["meta"]["orbital"]["gauge_note"]


def test_complex_k_im_and_refusals():
    from mace.gui import orbitals as O
    from mace.orbitals.dataset import OrbitalsError
    r = O.grid_job(path=PY, k=2, mos=["HOMO"], part="im", cells=[4, 1, 1], quality="preview")
    m = r["meta"]
    assert m["signed"] and m["orbital"]["part"] == "im" and m["orbital"]["kpoint"]["real"] is False
    assert m["pbc"] == [True, False, False] and m["orbital"]["cells"] == [4, 1, 1]
    with pytest.raises(OrbitalsError, match="Im psi is zero"):
        O.grid_job(path=H2O, k=1, mos=[5], part="im", cells=[1, 1, 1], quality="preview")
    with pytest.raises(OrbitalsError):
        O.grid_job(path=SIF, k=4, mos=[14], part="re", cells=[1, 1, 1], quality="preview")
    for bad in (dict(part="x"), dict(cells=[0, 1, 1]), dict(cells=[9, 1, 1]), dict(quality="ultra"), dict(mos=[]), dict(k="a")):
        kw = dict(k=1, spin=None, mos=[5], part="re", cells=[1, 1, 1], quality="preview", degenerate_sum=False)
        kw.update(bad)
        with pytest.raises(ValueError):
            O.check_request(**kw)


def test_worker_keeps_runs_until_files_change(tmp_path):
    from mace.gui import orbitals as O
    d = tmp_path / "h2o"
    shutil.copytree(DATA / "h2o", d)
    a = O.open_run(str(d / "h2o_c.molden"))
    assert O.open_run(str(d / "h2o_c.molden")) is a
    p = d / "h2o_c.molden"
    p.write_bytes(p.read_bytes())
    t = time.time() + 5
    import os
    os.utime(p, (t, t))
    assert O.open_run(str(p)) is not a
    # a run whose file leads out of the served folder (a symlink) is refused
    inside = tmp_path / "served"
    inside.mkdir()
    (inside / "x.molden").symlink_to(p)
    with pytest.raises(PermissionError):
        O.open_run(str(inside / "x.molden"), root=str(inside))


# ---------------------------------------------------------------- the routes, through the real server
@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orbitals_api")
    root = H.examples_copy(tmp)
    shutil.copytree(DATA, root / "orb")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="2")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)


def result(base, path, data=None, timeout=180):
    st, _, r = U.post(base, path, data) if data is not None else U.req(base, path)
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                return r
            time.sleep(0.1)
        raise AssertionError("the task never finished")
    assert st == 200, (st, r)
    return r


def test_routes_index_grid_and_cache(served):
    root, base, _ = served
    ix = result(base, "api/orbitals?path=orb/si_g/si_g_00001_K000_real.molden&wait=60")
    assert ix["path"] == "orb/si_g/si_g_00001_K000_real.molden" and ix["k"]["spins"][0]["homo"] == 14
    q = "api/orbitals/grid?path=orb/si_g/si_g_00001_K000_real.molden&k=1&mo=12&sum=1&cells=2x2x2&quality=preview&wait=60"
    m = result(base, q)
    assert m["orbital"]["mos"] == [12, 13, 14] and not m["signed"] and m["grid_key"]
    st, h, raw = U.req(base, f"api/volume/grid?key={m['grid_key']}")
    assert st == 200 and len(raw) == 4 * int(np.prod(m["shape"]))
    # the same request again: the cached meta and grid, no task
    st, _, again = U.req(base, q)
    assert st == 200 and again["grid_key"] == m["grid_key"]
    # Im at a polymer's complex k
    m = result(base, "api/orbitals/grid?path=orb/polyyne/py_f_00001_K000_real.molden&k=2&mo=HOMO&part=im&cells=4x1x1&quality=preview&wait=60")
    assert m["signed"] and m["orbital"]["part"] == "im"


def test_routes_refuse_bad_paths_and_values(served):
    root, base, _ = served
    st, _, r = U.req(base, "api/orbitals?path=../etc/passwd")
    assert st in (403, 404), (st, r)
    st, _, r = U.req(base, "api/orbitals?path=orb/h2o/h2o_c.out")
    assert st == 400 and "Molden" in r["error"]
    st, _, r = U.req(base, "api/orbitals/grid?path=orb/h2o/h2o_c.molden&k=1&mo=5&part=xx")
    assert st == 400
    st, _, r = U.req(base, "api/orbitals/grid?path=orb/h2o/h2o_c.molden&k=1&mo=5&cells=9x1x1")
    assert st == 400
    st, _, r = U.req(base, "api/orbitals?path=orb/h2o/h2o_c.molden", token=False)
    assert st in (401, 403)
    # an unusable k point: the task fails with the reader's reason
    st, _, r = U.req(base, "api/orbitals/grid?path=orb/si_f/si_f_00003_K200_real.molden&k=4&mo=14&wait=60")
    assert st == 500 and "orthonormality" in r["error"], (st, r)


def test_save_cube_in_the_orbitals_folder(served):
    root, base, _ = served
    molden = root / "orb/h2o/h2o_c.molden"
    before = hashlib.md5(molden.read_bytes()).hexdigest()
    body = {"path": "orb/h2o/h2o_c.molden", "k": 1, "mos": ["HOMO"], "part": "re", "cells": "1x1x1", "quality": "preview"}
    st, _, r = U.post(base, "api/orbitals/save", body)
    assert st == 200, r
    assert r["path"] == "orb/h2o/h2o_c_orbitals/h2o_c_ab_MO5_re.cube" and not r["cached"] and r["warning"] is None
    cube = root / r["path"]
    from mace.orbitals.cube import cube_meta, read_cube
    g, data, Z, pos, c1, c2 = read_cube(cube)
    assert cube_meta(c2)["mo"] == "5" and list(Z) == [8, 1, 1]
    stamp = cube.stat().st_mtime_ns
    st, _, r = U.post(base, "api/orbitals/save", body)
    assert st == 200 and r["cached"] and cube.stat().st_mtime_ns == stamp
    assert hashlib.md5(molden.read_bytes()).hexdigest() == before
    # a summed set: the cube of the set's |psi|^2
    st, _, r = U.post(base, "api/orbitals/save", {"path": "orb/si_g/si_g_00001_K000_real.molden", "k": 1, "mos": [12],
                                                  "degenerate_sum": True, "cells": [1, 1, 1], "quality": "preview"})
    assert st == 200 and r["path"].endswith("si_g_k1_ab_MO12-13-14_abs2.cube"), r


def test_save_refuses_a_symlinked_cube_folder(served, tmp_path):
    root, base, _ = served
    out = tmp_path / "elsewhere"
    out.mkdir()
    link = root / "orb/o2/o2_c_orbitals"
    link.symlink_to(out, target_is_directory=True)
    try:
        st, _, r = U.post(base, "api/orbitals/save", {"path": "orb/o2/o2_c.molden", "k": 1, "mos": [9], "cells": "1x1x1"})
        assert st == 403 and not list(out.iterdir()), (st, r)
    finally:
        link.unlink()


def test_quota_warning_counts_the_folder(tmp_path, monkeypatch):
    from mace.gui import orbitals as O
    from mace.orbitals import api
    d = tmp_path / "h2o"
    shutil.copytree(DATA / "h2o", d)
    folder = d / "h2o_c_orbitals"
    folder.mkdir()
    for i in range(api.WARN_CUBE_FILES):
        (folder / f"old_{i}.cube").write_text("x")
    r = O.save_job(path=str(d / "h2o_c.molden"), k=1, mos=[5], part="re", cells=[1, 1, 1], quality="preview")
    assert r["folder_files"] == api.WARN_CUBE_FILES + 1 and "1M-file quota" in r["warning"]
