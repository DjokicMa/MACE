"""Issue 2.3 (slice fidelity): plot_slice_plotly must preserve the full 2D
(skewed) Cartesian grid for non-orthogonal cells. The old go.Heatmap call
collapsed X/Y to 1D (X[0,:], Y[:,0]), rendering a rectangle instead of the true
parallelogram. Fixed by rendering with go.Surface on the full 2D affine grid.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parent.parent / "test" / "AddedPlottingFunctionalty"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

ccp = pytest.importorskip("crystal_cubeviz_plotly")


def _grid_r(vox, nvox):
    vox = np.asarray(vox, float)
    nx, ny, nz = nvox
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return ii[..., None] * vox[0] + jj[..., None] * vox[1] + kk[..., None] * vox[2]


def _write_cube(path, vox, nvox, center, sigma=1.4):
    data = np.exp(-((_grid_r(vox, nvox) - np.asarray(center)) ** 2).sum(-1) / (2 * sigma ** 2))
    with open(path, "w") as f:
        f.write("test\nGAUSSIAN CUBE FORMAT\n")
        f.write(f"{1:5d}{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        for n, v in zip(nvox, vox):
            f.write(f"{n:5d}{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}\n")
        f.write(f"{6:5d}{6.0:12.6f}{center[0]:12.6f}{center[1]:12.6f}{center[2]:12.6f}\n")
        flat = data.ravel(order="C")
        for i in range(0, len(flat), 6):
            f.write("".join(f"{v:13.5e}" for v in flat[i:i + 6]) + "\n")


TRI_VOX = [[0.30, 0, 0], [0.10, 0.28, 0], [0.05, 0.04, 0.30]]   # off-diagonals
ORTHO_VOX = [[0.30, 0, 0], [0, 0.30, 0], [0, 0, 0.30]]
NVOX = (18, 18, 18)
CENTER = (2.7, 2.7, 2.7)


def test_zslice_preserves_skew_via_surface(tmp_path):
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=False)
    tr = fig.data[0]
    assert tr.type == "surface", f"slice still rectilinear ({tr.type}); skew lost"
    x = np.asarray(tr.x)
    assert x.ndim == 2, f"slice x collapsed to {x.ndim}D; skew lost"
    # off-diagonal v2x means x must vary across the j-direction (rows differ)
    assert not np.allclose(x[0, :], x[-1, :]), "rows identical; parallelogram not rendered"


def test_orthogonal_slice_still_renders(tmp_path):
    f = tmp_path / "ortho_DENS.cube"
    _write_cube(f, ORTHO_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=False)
    assert fig.data[0].type == "surface"
    assert np.asarray(fig.data[0].x).ndim == 2


def test_slice_with_atoms_renders(tmp_path):
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=True)
    # surface + atom overlay both present, no exception
    assert any(t.type == "surface" for t in fig.data)


# ---- --slice-all (grid of subplots) ----

def test_slice_all_uses_surface(tmp_path):
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_all_slices_plotly(cube, "z", show_atoms=False)
    assert any(t.type == "surface" for t in fig.data)
    assert not any(t.type == "heatmap" for t in fig.data), "still rectilinear (skew lost)"


# ---- --slice-browse (slider) ----

def test_slice_browse_triclinic_uses_surface(tmp_path):
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_browser_plotly(cube, "z", show_atoms=False, max_frames=5)
    assert any(t.type == "surface" for t in fig.data)
    assert fig.frames and all(any(tr.type == "surface" for tr in fr.data) for fr in fig.frames)


def test_slice_browse_orthogonal_keeps_rich_contour_path(tmp_path):
    f = tmp_path / "ortho_DENS.cube"
    _write_cube(f, ORTHO_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_browser_plotly(cube, "z", show_atoms=False, max_frames=5)
    types = {tr.type for fr in fig.frames for tr in fr.data}
    assert "contour" in types or "heatmap" in types  # rich orthogonal path preserved
