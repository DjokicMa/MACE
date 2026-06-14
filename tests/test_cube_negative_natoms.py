"""Issue 2.6 (defensive): Gaussian-cube multi-dataset convention. A NEGATIVE
atom count means an extra "m id1..idm" line follows the atom block and the
volumetric data interleaves m values per grid point. The readers used
abs(natoms) for atoms but then slurped the DSET line into the data and crashed
on reshape. CRYSTAL emits positive natoms (m=1); this is foreign-cube interop.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Cube engine ships in the package (relocated from gitignored test/) so this
# import — and the CubeFile tests below — run on a fresh clone.
ccp = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly")

# subtract_cubes is a DEPRECATED standalone diagnostic intentionally left under
# gitignored test/ (the shipped cube engine's own diff path supersedes it). Its
# tests skip on a clone; they must not skip the CubeFile tests above.
_SC_DIR = Path(__file__).resolve().parent.parent / "test" / "AddedPlottingFunctionalty"
if str(_SC_DIR) not in sys.path:
    sys.path.insert(0, str(_SC_DIR))
try:
    import subtract_cubes as sc
except ImportError:
    sc = None
_needs_sc = pytest.mark.skipif(sc is None, reason="subtract_cubes (deprecated diagnostic) not on disk")

NV = (6, 6, 6)
VOX = [[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]]


def _header(f, natoms):
    f.write("test\nGAUSSIAN CUBE FORMAT\n")
    f.write(f"{natoms:5d}{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
    for n, v in zip(NV, VOX):
        f.write(f"{n:5d}{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}\n")
    for z in (6, 8):
        f.write(f"{z:5d}{float(z):12.6f}{1.0:12.6f}{1.0:12.6f}{1.0:12.6f}\n")


def _write_multidataset(path):
    with open(path, "w") as f:
        _header(f, -2)                      # negative -> multi-dataset
        f.write(f"{2:5d}{1:5d}{2:5d}\n")     # m=2, ids 1,2
        vals = []
        for _ in range(NV[0] * NV[1] * NV[2]):
            vals += [1.0, 2.0]              # dataset0=1.0, dataset1=2.0 interleaved
        for i in range(0, len(vals), 6):
            f.write("".join(f"{v:13.5e}" for v in vals[i:i + 6]) + "\n")


def _write_single(path, fill=1.0):
    with open(path, "w") as f:
        _header(f, 2)
        vals = [fill] * (NV[0] * NV[1] * NV[2])
        for i in range(0, len(vals), 6):
            f.write("".join(f"{v:13.5e}" for v in vals[i:i + 6]) + "\n")


def test_cubefile_reads_negative_natoms_multidataset(tmp_path):
    f = tmp_path / "multi_DENS.cube"
    _write_multidataset(f)
    cube = ccp.CubeFile(str(f), skip_vacuum_detection=True)
    assert cube.n_datasets == 2
    assert tuple(cube.data.shape) == NV
    assert np.allclose(cube.data, 1.0)                 # active = first dataset
    assert cube.data_all is not None and cube.data_all.shape == (*NV, 2)
    assert np.allclose(cube.data_all[..., 1], 2.0)


def test_cubefile_reads_positive_natoms_unchanged(tmp_path):
    f = tmp_path / "single_DENS.cube"
    _write_single(f)
    cube = ccp.CubeFile(str(f), skip_vacuum_detection=True)
    assert cube.n_datasets == 1
    assert tuple(cube.data.shape) == NV
    assert cube.data_all is None


@_needs_sc
def test_subtract_reader_handles_negative_natoms(tmp_path):
    f = tmp_path / "multi_DENS.cube"
    _write_multidataset(f)
    cube = sc.read_cube_file(str(f))
    assert tuple(cube["data"].shape) == NV
    assert np.allclose(cube["data"], 1.0)


@_needs_sc
def test_subtract_output_natoms_positive_roundtrip(tmp_path):
    a, b, out = tmp_path / "a.cube", tmp_path / "b.cube", tmp_path / "out.CUBE"
    _write_single(a, 2.0)
    _write_single(b, 0.5)
    res = sc.subtract_cubes(sc.read_cube_file(str(a)), sc.read_cube_file(str(b)), verbose=False)
    sc.write_cube_file(str(out), res)
    rt = sc.read_cube_file(str(out))
    assert rt["natoms"] > 0
    assert tuple(rt["data"].shape) == NV
    assert np.allclose(rt["data"], 1.5)
