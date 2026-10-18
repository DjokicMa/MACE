"""mace plotting's band/DOS figure size and resolution (1.1.6 review D-4, D-5, D-20).

The plotly PNGs carried no resolution (kaleido writes no pHYs chunk), so LaTeX and Word placed a 7.9 in
figure 25-33 in wide; the figure size could not be set. They now say their dpi and take
--width/--height (inches) and --dpi, as do the matplotlib figures.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
FE = REPO_ROOT / "tests" / "data" / "spin_fe"
plotly_ok = pytest.mark.skipif(not __import__("importlib").util.find_spec("kaleido"), reason="kaleido not installed")


def _run(cwd, backend, *args):
    env = {**os.environ, "MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": backend}
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "plotting", *args], cwd=str(cwd),
                       capture_output=True, text=True, timeout=300, env=env)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    return p


@pytest.fixture
def fe(tmp_path):
    for f in FE.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    return tmp_path


def _png(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size, tuple(round(v) for v in im.info.get("dpi", (0, 0)))


@plotly_ok
def test_plotly_png_says_its_dpi_and_opens_at_the_figure_size(fe):
    _run(fe, "plotly", "--band", "--format", "png")
    (w, h), dpi = _png(fe / "fe.BANDS.png")
    assert dpi == (300, 300)
    assert w / 300 == pytest.approx(760 / 96, abs=0.01) and h / 300 == pytest.approx(420 / 96, abs=0.01)


@plotly_ok
def test_plotly_width_height_and_dpi(fe):
    _run(fe, "plotly", "--dos", "--format", "png", "--width", "3.5", "--height", "4", "--dpi", "600")
    (w, h), dpi = _png(fe / "fe.DOSS.png")
    assert dpi == (600, 600) and (w, h) == (pytest.approx(2100, abs=3), pytest.approx(2400, abs=3))


def test_matplotlib_width_height_and_dpi(fe):
    _run(fe, "matplotlib", "--band", "--format", "png", "--width", "3", "--height", "2.5", "--dpi", "200")
    (w, h), dpi = _png(fe / "fe.BANDS.png")
    assert dpi == (200, 200) and h == 500 and w >= 600       # matplotlib grows a figure for its legend (legend_layout)
    _run(fe, "matplotlib", "--dos", "--format", "png", "--width", "3", "--height", "4", "--dpi", "100")
    (w, h), dpi = _png(fe / "fe.DOSS.png")
    assert dpi == (100, 100) and h == 400 and w >= 300


def test_set_png_dpi_keeps_the_pixels(tmp_path):
    from PIL import Image
    from mace.plotting import plotly_bands as PB
    p = tmp_path / "x.png"
    Image.new("RGB", (7, 5), (10, 20, 30)).save(p, dpi=(72, 72))
    PB.set_png_dpi(p, 300)
    with Image.open(p) as im:
        assert im.size == (7, 5) and tuple(round(v) for v in im.info["dpi"]) == (300, 300)
        assert im.getpixel((3, 2)) == (10, 20, 30)
