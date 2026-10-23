"""mace plotting's static cube and orbital figures (E-M8, F4-5): one surface (a +/- pair for
signed data) with transparency as the GUI, orbital lobes + blue / - red as the GUI, molecules
labeled and drawn with bonds, the box framed, and slices as flat 2D maps."""
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")                       # these tests draw; the engine imports without plotly
ccp = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly")
from mace.orbitals import api  # noqa: E402

H2O = Path(__file__).parent / "data" / "orbitals" / "h2o"


@pytest.fixture
def h2o_orbital_cube(tmp_path):
    for f in H2O.iterdir():
        shutil.copy(f, tmp_path / f.name)
    run = api.open_run(tmp_path / "h2o_c.molden")
    path, _ = api.make_cube(run, 1, "ab", ["HOMO"], part="re", quality="preview")
    return path


def _crystal_0d_cube(path):
    """A CRYSTAL ECH3 cube of a molecule: the second comment line is a zero cell."""
    n = 16
    r = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - n / 2) * 0.3
    data = np.exp(-(r ** 2).sum(-1))
    with open(path, "w") as f:
        f.write(" Charge density - 3D GRID - GAUSSIAN CUBE FORMAT ./mol\n")
        f.write("     0.00000000     0.00000000     0.00000000     0.000000   0.000000   0.000000\n")
        f.write(f"    2 {-n * 0.15:12.6f} {-n * 0.15:12.6f} {-n * 0.15:12.6f}\n")
        for i in range(3):
            v = [0.0, 0.0, 0.0]
            v[i] = 0.3
            f.write(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        f.write(f"    1     1.000000     0.000000     0.000000    -0.700000\n")
        f.write(f"    1     1.000000     0.000000     0.000000     0.700000\n")
        flat = data.ravel()
        for i in range(0, len(flat), 6):
            f.write("".join(f"{v:13.5E}" for v in flat[i:i + 6]) + "\n")
    return path


def test_orbital_cube_is_a_molecule_with_a_blue_red_lobe_pair(h2o_orbital_cube):
    cube = ccp.CubeFile(str(h2o_orbital_cube))
    assert cube.is_molecule and cube.structure_label() == "molecule"
    fig = ccp.plot_isosurface_plotly(cube, opacity=0.5)
    meshes = [t for t in fig.data if t.type == "mesh3d"]
    assert len(meshes) == 2                        # one +/- pair, not four nested surfaces
    by_sign = {t.name[0]: t.color for t in meshes}
    assert by_sign["+"] == ccp.ISO["plus"] == "#2563eb"      # + blue, as the GUI's --vol-plus
    assert by_sign["−"] == ccp.ISO["minus"]
    assert all(t.opacity == 0.5 for t in meshes)
    assert any(t.type == "scatter3d" and t.mode == "lines" for t in fig.data)   # bonds
    eye = fig.layout.scene.camera.eye
    assert min(eye.x, eye.y) >= 1.5                 # far enough out to keep the box in frame


def test_crystal_molecule_density_gets_one_surface(tmp_path):
    cube = ccp.CubeFile(str(_crystal_0d_cube(tmp_path / "mol_DENS.CUBE")))
    assert cube.is_molecule and not cube.is_2d
    assert len(ccp.default_iso_values(cube)) == 1
    fig = ccp.plot_isosurface_plotly(cube)
    assert len([t for t in fig.data if t.type == "mesh3d"]) == 1


def test_gui_and_cli_share_the_default_level(tmp_path):
    from mace.gui import volume
    cube = ccp.CubeFile(str(_crystal_0d_cube(tmp_path / "mol_DENS.CUBE")))
    assert volume.default_level(cube) == ccp.default_iso_level(cube) == ccp.default_iso_values(cube)[0]


def test_slice_png_has_no_3d_scene_and_a_unit(tmp_path):
    cube = ccp.CubeFile(str(_crystal_0d_cube(tmp_path / "mol_POT.CUBE")))
    fig = ccp.plot_slice_plotly(cube, "z", position=8)
    assert {t.type for t in fig.data} <= {"heatmap", "scatter"}
    assert "Ha e<sup>−1</sup>" in fig.data[0].colorbar.title.text
