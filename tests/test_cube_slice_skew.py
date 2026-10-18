"""Issue 2.3 (slice fidelity): plot_slice_plotly must preserve the full 2D
(skewed) Cartesian grid for non-orthogonal cells. The old go.Heatmap call
collapsed X/Y to 1D (X[0,:], Y[:,0]), rendering a rectangle instead of the true
parallelogram. Fixed by rendering with go.Surface on the full 2D affine grid.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Cube engine ships in the package (relocated from gitignored test/); import it
# from there so this test runs on a fresh clone.
ccp = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly")


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


def test_zslice_preserves_skew_as_a_flat_2d_map(tmp_path):
    """A single slice is a flat 2D map (no 3D scene: its edge-on axis strip showed in PNGs),
    resampled on Cartesian in-plane axes so a skewed cell keeps its parallelogram."""
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=False)
    tr = fig.data[0]
    assert tr.type == "heatmap"
    assert not any(t.type in ("surface", "scatter3d", "mesh3d") for t in fig.data)
    z = np.asarray(tr.z, float)
    # the parallelogram leaves NaN corners: its bottom row starts at x = 0, its top row is shifted
    nan_bottom = np.isnan(z[0]).argmin()
    nan_top = np.isnan(z[-1]).argmin()
    assert np.isnan(z).any() and nan_top > nan_bottom, "skew lost"
    assert fig.layout.xaxis.scaleanchor == "y"


def test_orthogonal_slice_still_renders(tmp_path):
    f = tmp_path / "ortho_DENS.cube"
    _write_cube(f, ORTHO_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=False)
    z = np.asarray(fig.data[0].z, float)
    assert fig.data[0].type == "heatmap" and not np.isnan(z).any()
    assert fig.layout.xaxis.title.text == "X (Å)" and fig.layout.yaxis.title.text == "Y (Å)"
    assert np.nanmax(z) > np.nanmin(z)


def test_slice_with_atoms_renders(tmp_path):
    f = tmp_path / "tri_DENS.cube"
    _write_cube(f, TRI_VOX, NVOX, CENTER)
    cube = ccp.CubeFile(str(f))
    fig = ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=True)
    assert any(t.type == "heatmap" for t in fig.data)
    assert any(t.type == "scatter" and t.mode == "markers+text" for t in fig.data)
    assert "e/bohr³" in fig.data[0].colorbar.title.text


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
