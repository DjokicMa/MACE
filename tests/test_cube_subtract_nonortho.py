"""Issue 2.3: subtract_cubes must interpolate with the FULL voxel matrix so
different-grid subtraction is correct on non-orthogonal (triclinic) cells, not
just axis-aligned ones.

Strategy (known answer): the SAME physical Gaussian sampled on two different
grids of the same lattice -> a correct A - interp(B) equals the independent
full-affine reference (and ~0). Pre-fix the diagonal-only grid injects a large
residual on triclinic cells; orthogonal and same-grid must stay correct.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parent.parent / "test" / "AddedPlottingFunctionalty"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

sc = pytest.importorskip("subtract_cubes")
pytest.importorskip("scipy")
from scipy.ndimage import map_coordinates  # noqa: E402

ORTHO = [[13.0, 0, 0], [0, 12.0, 0], [0, 0, 13.0]]
TRICLINIC = [[13.0, 0, 0], [5.0, 12.0, 0], [2.0, 1.0, 13.0]]  # strong skew


def _grid_r(origin, vox, nvox):
    vox = np.asarray(vox, float)
    nx, ny, nz = nvox
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return (np.asarray(origin, float)
            + ii[..., None] * vox[0] + jj[..., None] * vox[1] + kk[..., None] * vox[2])


def _gauss(R, center, sigma=1.6):
    return np.exp(-((R - np.asarray(center)) ** 2).sum(-1) / (2 * sigma ** 2))


def _cube(origin, cell, nvox, center):
    cell = np.asarray(cell, float)
    vox = cell / np.asarray(nvox)[:, None]
    data = _gauss(_grid_r(origin, vox, nvox), center)
    return {"title1": "t", "title2": "t", "natoms": 1,
            "origin": np.asarray(origin, float), "nvoxels": np.asarray(nvox),
            "voxel_vectors": vox, "atoms": [[6, 6.0, *center]], "data": data}


def _reference(large, small):
    """Correct large - interp(small) via the full affine transform."""
    Ms = np.asarray(small["voxel_vectors"]).T
    invMs = np.linalg.inv(Ms)
    rL = _grid_r(large["origin"], large["voxel_vectors"], large["nvoxels"])
    frac = (rL - small["origin"]) @ invMs.T
    interp = map_coordinates(small["data"],
                             [frac[..., 0].ravel(), frac[..., 1].ravel(), frac[..., 2].ravel()],
                             order=1, mode="constant", cval=0.0).reshape(large["data"].shape)
    return large["data"] - interp


def _case(cell):
    P = 0.5 * np.asarray(cell, float).sum(0)          # fractional (.5,.5,.5)
    large = _cube([0, 0, 0], cell, (44, 44, 44), P)
    small = _cube([1.0, 0.8, 0.6], cell, (36, 36, 36), P)  # same blob, shifted/coarser grid
    res = sc.subtract_cubes(large, small, verbose=False)
    ref = _reference(large, small)
    peak = large["data"].max()
    return np.abs(res["data"] - ref).max() / peak


def test_subtract_triclinic_diff_grid_matches_affine_reference():
    # pre-fix: ~0.16 (diagonal-only). post-fix: ~1e-6.
    assert _case(TRICLINIC) < 0.01


def test_subtract_orthogonal_diff_grid_matches_reference():
    assert _case(ORTHO) < 0.01


def test_subtract_same_grid_is_exact():
    cell = TRICLINIC
    large = _cube([0, 0, 0], cell, (30, 30, 30), 0.5 * np.asarray(cell, float).sum(0))
    small = _cube([0, 0, 0], cell, (30, 30, 30), 0.4 * np.asarray(cell, float).sum(0))
    res = sc.subtract_cubes(large, small, verbose=False)
    assert np.allclose(res["data"], large["data"] - small["data"], atol=1e-6)
