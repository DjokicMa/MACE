"""The plotly figures `mace plotting` writes (cube isosurface / slice / slice grid,
FREQ mode animation) keep their legends, colorbars and annotation boxes off the
data, the tick labels, each other, the controls and the figure edge: the HTML
is opened in Chromium at its own size and at the PNG/SVG export size
(1800 x 1350), and measured with tests/legend_geometry.py, the rule every MACE
plot is held to. Cubes come from the repository fixtures, plus the corpus's
potential and spin cubes when it is present.
"""
import gzip
import shutil
import sys
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from legend_geometry import PLOTLY_BOXES, plotly_problems  # noqa: E402

FIX = H.REPO_ROOT / "tests" / "fixtures" / "gui_parity"
CORPUS = H.REPO_ROOT / "test"


def _inputs(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    cubes = {}
    (src / "mgo_dens_DENS.CUBE").write_bytes(gzip.decompress((FIX / "mgo_dens" / "DENS_CUBE.DAT.gz").read_bytes()))
    cubes["dens"] = src / "mgo_dens_DENS.CUBE"
    (src / "mgo_WAN.CUBE").write_bytes(gzip.decompress((FIX / "mgo_wan" / "1_WAN.CUBE.gz").read_bytes()))
    cubes["orbital"] = src / "mgo_WAN.CUBE"
    syn = CORPUS / "_synthetic_cubes" / "cubic3D_POT.cube"
    if syn.is_file():
        cubes["potential"] = Path(shutil.copy(syn, src / "cub_POT.cube"))
    spin = sorted((CORPUS / "ECH3POT3").glob("*conf2*_SPIN.CUBE")) if (CORPUS / "ECH3POT3").is_dir() else []
    if spin:
        cubes["spin"] = Path(shutil.copy(spin[0], src / "mol_SPIN.CUBE"))
    (src / "water_freq.out").write_bytes(gzip.decompress((FIX / "water_freq.out.gz").read_bytes()))
    return cubes, src / "water_freq.out"


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    from mace.plotting.handlers import cube as hc, freq as hf
    tmp = tmp_path_factory.mktemp("cliplots")
    cubes, freq = _inputs(tmp / "src")
    for name, f in cubes.items():
        for view in ("iso", "slice", "slice-all"):
            cfg = hc.configure_cube_plot(interactive=False)
            cfg.update(view=view, formats=["html"])
            hc.plot_cube([str(f)], cfg, str(tmp / f"cube_{name}_{view}"))
    hf.plot_freq([str(freq)], hf.configure_freq_plot(interactive=False), str(tmp / "freq"))
    files = sorted(tmp.rglob("*.html"))
    assert len(files) >= 7, files
    return files


def test_cli_plotly_figures_keep_their_boxes_off_the_data(outputs):
    drv = H.chromium(1366, 900)
    failures = []
    try:
        for f in outputs:
            drv.get(f.as_uri())
            H.wait(drv, "const g = document.querySelector('.plotly-graph-div'); return !!(g && g._fullLayout)", 60)
            time.sleep(0.8)
            for size in (None, (1800, 1350)):
                if size:
                    drv.execute_script("Plotly.relayout(document.querySelector('.plotly-graph-div'), {width: arguments[0], height: arguments[1]}); return 1", *size)
                    time.sleep(0.6)
                st = drv.execute_script(PLOTLY_BOXES, drv.find_element("css selector", ".plotly-graph-div"))
                failures += [f"{f.name} {size or 'html'}: {p}" for p in plotly_problems(st)]
    finally:
        drv.quit()
    assert not failures, "\n".join(failures)
