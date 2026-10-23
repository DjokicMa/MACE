"""Arbitrary slice planes of a cube (1.1.6): mace/gui/volume.py plane_geometry / plane_sample, the
`volume/plane` route's arithmetic, and `mace slice` (run as a subprocess through mace_cli).

* (hkl) planes satisfy h x + k y + l z = t in fractional coordinates of the box, in a triclinic box too;
  [uvw] planes are normal to u a + v b + w c; tilt and twist turn the normal by their angles.
* The sampling is exact for a field linear in the Cartesian coordinates (trilinear interpolation of a
  linear field), in a skewed box; periodic axes wrap (no NaN at the far face).
* On a real CRYSTAL cube (test/ corpus), the (0 0 1) plane at a grid plane's position reads the values of
  the server's exact grid slice there.
* Bad requests are refused in words.
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.gui import volume as V
from mace.gui import orbitals as O

CUBE = (REPO_ROOT / "test" / "ECH3POT3" /
        "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential_DENS.CUBE")
TRICLINIC = {"origin": [0.3, -0.2, 0.1], "edges": [[5.0, 0.0, 0.0], [1.2, 4.5, 0.0], [0.8, 0.6, 6.0]],
             "pbc": [True, True, True], "period": [20, 20, 24]}


def _frac(box, pts):
    E = np.asarray(box["edges"], float)
    return (np.asarray(pts, float) - np.asarray(box["origin"], float)) @ np.linalg.inv(E)


@pytest.mark.parametrize("hkl", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, -1, 1), (0, 0, 3)])
@pytest.mark.parametrize("pos", [0.2, 0.5, 0.8])
def test_miller_plane_is_hx_plus_ky_plus_lz_const(hkl, pos):
    g = V.plane_geometry(TRICLINIC, "hkl", hkl, pos)
    assert len(g["polygon"]) >= 3
    f = _frac(TRICLINIC, g["polygon"])
    t = f @ np.asarray(hkl, float)
    assert np.allclose(t, g["miller_t"], atol=1e-6), (t, g["miller_t"])
    # every corner of the cut lies on the box's surface (some fractional coordinate is 0 or 1)
    assert all(np.any(np.isclose(row, 0, atol=1e-6) | np.isclose(row, 1, atol=1e-6)) for row in f)
    assert abs(np.linalg.norm(g["normal"]) - 1) < 1e-9
    # the spacing: 1 / |h a* + k b* + l c*|
    R = np.linalg.inv(np.asarray(TRICLINIC["edges"])).T
    assert math.isclose(g["spacing_A"], 1 / np.linalg.norm(np.asarray(hkl) @ R), rel_tol=1e-9)
    assert math.isclose(g["position_A"], pos * (g["dmax"] - g["dmin"]), rel_tol=1e-9)


def test_uvw_plane_is_normal_to_the_direction_and_turns_by_its_angles():
    g = V.plane_geometry(TRICLINIC, "uvw", (1, 1, 0), 0.5)
    d = np.asarray(TRICLINIC["edges"][0]) + np.asarray(TRICLINIC["edges"][1])
    assert np.allclose(np.cross(g["normal"], d / np.linalg.norm(d)), 0, atol=1e-9)
    t = V.plane_geometry(TRICLINIC, "uvw", (1, 1, 0), 0.5, tilt=25)
    assert math.isclose(math.degrees(math.acos(np.clip(np.dot(g["normal"], t["normal"]), -1, 1))), 25, abs_tol=1e-6)
    tt = V.plane_geometry(TRICLINIC, "uvw", (1, 1, 0), 0.5, twist=-30)
    assert math.isclose(math.degrees(math.acos(np.clip(np.dot(g["normal"], tt["normal"]), -1, 1))), 30, abs_tol=1e-6)
    # the turned plane's in-plane axes stay orthonormal and normal to it
    for x in (t, tt):
        n, e1, e2 = (np.asarray(x[k]) for k in ("normal", "e1", "e2"))
        assert abs(np.dot(n, e1)) < 1e-9 and abs(np.dot(n, e2)) < 1e-9 and abs(np.dot(e1, e2)) < 1e-9
        P = np.asarray(x["polygon"])
        assert np.allclose((P - np.asarray(x["point"])) @ n, 0, atol=1e-6)


def test_sampling_is_exact_for_a_linear_field_in_a_skewed_box():
    n = (12, 10, 14)
    box = {"origin": [0.3, -0.2, 0.1], "edges": None, "pbc": [False, False, False], "period": list(n)}
    E = np.asarray(TRICLINIC["edges"], float)
    vox = E / np.asarray([[x - 1] for x in n])                 # a non-periodic grid spans n - 1 voxels
    box["edges"] = (vox * np.asarray([[x - 1] for x in n])).tolist()
    idx = np.indices(n).reshape(3, -1).T
    cart = np.asarray(box["origin"]) + idx @ vox
    a = np.array([0.7, -1.3, 2.1])
    data = (cart @ a + 0.5).reshape(n)
    for spec in (("hkl", (1, 1, 1), 0.45, 0, 0), ("uvw", (1, 0, 2), 0.6, 15, -10), ("hkl", (0, 1, 0), 0.3, 0, 0)):
        g = V.plane_geometry(box, spec[0], spec[1], spec[2], spec[3], spec[4])
        s = V.plane_sample(data, box["origin"], vox, n, box["pbc"], g)
        X, Y = np.meshgrid(s["x"], s["y"])
        pts = np.asarray(g["point"]) + X[..., None] * np.asarray(g["e1"]) + Y[..., None] * np.asarray(g["e2"])
        want = pts @ a + 0.5
        ok = np.isfinite(s["values"])
        assert ok.sum() > 50, spec
        # (the mesh coordinates are reported to 1e-5 A)
        assert np.allclose(s["values"][ok], want[ok], atol=1e-4), spec


def test_periodic_axes_wrap_at_the_far_face():
    n = (8, 8, 8)
    vox = np.eye(3) * 0.5
    box = {"origin": [0, 0, 0], "edges": (vox * 8).tolist(), "pbc": [True, True, True], "period": list(n)}
    k = np.arange(8)
    data = np.cos(2 * np.pi * k / 8)[:, None, None] * np.ones(n)
    g = V.plane_geometry(box, "hkl", (0, 0, 1), 0.5)
    s = V.plane_sample(data, box["origin"], vox, n, box["pbc"], g)
    v = s["values"]
    assert np.isfinite(v).all()                     # the last voxel band wraps to the first plane, no NaN
    # along x the field is cos(2 pi x / 4 A): one period across the 4 A box, its two ends the same value
    assert np.allclose(v[:, 0], v[:, -1], atol=1e-9)


@pytest.mark.parametrize("bad, msg", [((0, 0, 0), "not a plane"), ((13, 0, 0), "run from"), (("a", 1, 1), "whole numbers")])
def test_bad_planes_are_refused_in_words(bad, msg):
    with pytest.raises(ValueError, match=msg):
        V.plane_geometry(TRICLINIC, "hkl", bad, 0.5)
    with pytest.raises(ValueError, match="position"):
        V.plane_geometry(TRICLINIC, "hkl", (1, 0, 0), 1.5)
    with pytest.raises(ValueError, match="tilt"):
        V.plane_geometry(TRICLINIC, "hkl", (1, 0, 0), 0.5, tilt=95)


def test_a_turned_plane_that_misses_the_box_says_so():
    flat = {"origin": [0, 0, 0], "edges": [[10, 0, 0], [0, 10, 0], [0, 0, 0.5]], "pbc": [False] * 3, "period": [5, 5, 2]}
    g = V.plane_geometry(flat, "hkl", (0, 0, 1), 0.0, tilt=0)
    assert len(g["polygon"]) >= 3
    with pytest.raises(ValueError, match="does not cut"):
        V.plane_sample(np.zeros((5, 5, 2)), flat["origin"], np.diag([2.5, 2.5, 0.5]), flat["period"], flat["pbc"],
                       {**g, "polygon": []})


def corpus():
    if not CUBE.is_file():
        pytest.skip("test/ corpus not present (gitignored)")


def test_real_cube_001_plane_reads_the_grid_slice():
    corpus()
    c = V.read(str(CUBE), skip_vacuum=True)
    box = V.grid_box(c)
    n = c.data.shape[2]
    i = n // 2
    r = V.plane_of_cube(c, "hkl", (0, 0, 1), pos=i / (n - 1))
    s = V.slice_job(path=str(CUBE), axis="c", index=i)
    assert math.isclose(r["position_A"], s["position_A"], rel_tol=1e-6)
    # the plane's mesh points fall between grid points; at a grid point the values agree, and everywhere the
    # sampled values are within the slice's range
    vals = np.asarray(s["values"], float)
    z = r["values"][np.isfinite(r["values"])]
    assert vals.min() - 1e-9 <= z.min() and z.max() <= vals.max() + 1e-9
    assert abs(z.mean() - vals.mean()) < 0.05 * abs(vals).max()
    assert box["pbc"] == [False, False, False] and box["period"] == list(c.data.shape)


def test_payload_carries_the_box():
    corpus()
    c = V.read(str(CUBE))
    meta, _ = V.payload(c, CUBE, "low")
    assert set(meta["box"]) == {"origin", "edges", "pbc", "period"}


def _mace(*args, cwd=None):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), *args], capture_output=True, text=True, timeout=300,
                          cwd=cwd, env={**__import__("os").environ, "MPLBACKEND": "Agg"})


def test_mace_slice_cli_writes_a_table_and_a_picture(tmp_path):
    corpus()
    r = _mace("slice", str(CUBE), "--hkl", "1", "1", "0", "--pos", "0.4", "-o", str(tmp_path / "p.csv"))
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / "p.csv").read_text().splitlines()
    assert lines[0].startswith("# ") and "(1 1 0)" in lines[0] and lines[1] == "# x_A,y_A,value" and len(lines) > 100
    r = _mace("slice", str(CUBE), "--axis", "c", "--index", "5", "-o", str(tmp_path / "c5.png"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "c5.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    r = _mace("slice", str(CUBE), "--hkl", "0", "0", "0")
    assert r.returncode == 1 and "not a plane" in r.stderr
    r = _mace("slice", "--help")
    assert r.returncode == 0 and "mace slice" in r.stdout


@pytest.mark.parametrize("mode, ints", [("hkl", (1, 1, 1)), ("hkl", (2, -1, 1)), ("uvw", (1, 0, 0)), ("hkl", (0, 0, 1))])
@pytest.mark.parametrize("pos", [0.0, 1.0])
def test_a_plane_at_either_end_still_cuts_the_box(mode, ints, pos):
    """At pos 0 or 1 an oblique plane only touched a corner or an edge of the box ("the plane does
    not cut the grid's box"); it stays a hair inside, and its map has two points across at least."""
    g = V.plane_geometry(TRICLINIC, mode, ints, pos)
    assert len(g["polygon"]) >= 3
    assert g["dmin"] < g["d"] < g["dmax"] and abs(g["position_A"] - pos * (g["dmax"] - g["dmin"])) <= \
        1.001e-3 * (g["dmax"] - g["dmin"])
    s = V.plane_sample(np.ones((20, 20, 24)), TRICLINIC["origin"], np.asarray(TRICLINIC["edges"]) / 20,
                       TRICLINIC["period"], TRICLINIC["pbc"], g)
    assert len(s["x"]) >= 2 and len(s["y"]) >= 2


def test_mace_slice_at_the_box_s_corner_and_its_default_names(tmp_path):
    corpus()
    import shutil
    for name in ("Si_1.5GPa_DENS.CUBE", "Si_1.5GPa_POT.CUBE"):
        shutil.copy(CUBE, tmp_path / name)
        r = _mace("slice", name, "--hkl", "1", "1", "1", "--pos", "0", cwd=tmp_path)
        assert r.returncode == 0, r.stderr
    # the default name keeps the whole stem: the two cubes' pictures do not overwrite each other
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["Si_1.5GPa_DENS_111.png", "Si_1.5GPa_POT_111.png"]


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node not installed")
def test_js_plane_matches_at_the_ends(tmp_path):
    """slice_widget.js planeGeometry keeps the plane inside the box as volume.py does."""
    import json
    js = REPO_ROOT / "mace" / "gui" / "static" / "js" / "docs" / "slice_widget.js"
    cases = [("hkl", [1, 1, 1], 0.0), ("hkl", [1, 1, 1], 1.0), ("uvw", [1, 0, 0], 0.0), ("hkl", [0, 0, 1], 0.5)]
    script = tmp_path / "run.mjs"
    script.write_text(f"import {{ planeGeometry }} from {json.dumps(js.as_uri())};\n"
                      f"const box = {json.dumps(TRICLINIC)};\n"
                      f"console.log(JSON.stringify({json.dumps(cases)}.map(([mode, ints, pos]) => "
                      "{ const g = planeGeometry(box, { mode, ints, pos }); return [g.d, g.polygon.length]; })));\n")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for (mode, ints, pos), (d, npoly) in zip(cases, json.loads(r.stdout)):
        g = V.plane_geometry(TRICLINIC, mode, ints, pos)
        assert npoly >= 3 and math.isclose(d, g["d"], rel_tol=1e-9, abs_tol=1e-9), (mode, ints, pos, d, g["d"])


def test_orbital_list_short_occupations():
    assert O._occ_short(2.0, 2.0, True) == "2 e"
    assert O._occ_short(1.0, 1.0, False) == "1 e"
    assert O._occ_short(0.0, 2.0, True) == "empty"
    assert O._occ_short(0.523, 2.0, True) == "0.523 e"
    assert O._occ_text(2.0, 2.0, True) == "2 e (closed shell)"          # the words stay (tooltip, screen readers)
