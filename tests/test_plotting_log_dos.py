"""mace plotting --log-dos (1.1.6): the DOS axis only, as mace gui's Log Y on a
DOS plot and Log DOS on a band + DOS plot. Band energies are signed, so their
axis stays linear and --band --log-dos is refused. Through mace_cli, on
polyacetylene of the bundled example suite."""
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "examples" / "suite" / "4-polymer-1d" / "polyacetylene"
FILES = {"BAND": ("polyacetylene_band.BAND.DAT", "polyacetylene_band.d3", "polyacetylene_band.out"),
         "DOSS": ("polyacetylene_doss.DOSS.DAT", "polyacetylene_doss.d3", "polyacetylene_doss.out")}


def unpack(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for sub, names in FILES.items():
        for name in names:
            xz, plain = RUN / sub / (name + ".xz"), RUN / sub / name
            if xz.is_file():
                (dst / name).write_bytes(lzma.decompress(xz.read_bytes()))
            elif plain.is_file():
                (dst / name).write_bytes(plain.read_bytes())
            else:
                pytest.skip("the bundled example suite is not here")


def mace(cwd, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300, env=env)


def figure_of(html: Path):
    """The figure's traces and layout from plotly's HTML (Plotly.newPlot(id, data, layout, config))."""
    text = html.read_text()
    i = text.rindex("Plotly.newPlot(")          # the call (the bundled plotly.js names it first)
    dec = json.JSONDecoder()
    pos = text.index("[", i)
    data, pos = dec.raw_decode(text, pos)
    pos = text.index("{", pos)
    layout, _ = dec.raw_decode(text, pos)
    return data, layout


def layout_of(html: Path) -> dict:
    return figure_of(html)[1]


@pytest.fixture
def run_dir(tmp_path):
    pytest.importorskip("plotly")
    unpack(tmp_path / "both")
    return tmp_path


@pytest.mark.parametrize("flag,name,dos_axis", [("--band-dos", "polyacetylene_band.BAND.DAT", "xaxis2"),
                                                ("--dos", "polyacetylene_doss.DOSS.DAT", "xaxis")])
def test_log_dos_is_the_dos_axis(run_dir, flag, name, dos_axis):
    for log in (False, True):
        out = run_dir / f"out{int(log)}{flag}"
        r = mace(run_dir, flag, f"both/{name}", *(["--log-dos"] if log else []), "--backend", "plotly",
                 "--format", "html", "-o", str(out))
        assert r.returncode == 0, r.stdout + r.stderr
        data, lay = figure_of(next(out.glob("*.html")))
        assert lay[dos_axis].get("type", "linear") == ("log" if log else "linear"), lay[dos_axis]
        # review 1.1.6 E3: a log DOS total is not filled (the fill was joined across the band gap)
        fills = {t.get("fill", "none") for t in data if t.get("meta") == "dos"}
        assert fills == {"none"} if log else "tozerox" in fills, fills
        energy = lay["yaxis"]
        assert energy.get("type", "linear") == "linear", energy


def test_band_log_dos_is_refused(run_dir):
    r = mace(run_dir, "--band", "both/polyacetylene_band.BAND.DAT", "--log-dos")
    assert r.returncode == 2 and "--log-dos is for --dos and --band-dos" in r.stderr, r.stderr[-400:]


def test_matplotlib_refuses_log_dos(run_dir):
    r = mace(run_dir, "--band-dos", "both/polyacetylene_band.BAND.DAT", "--log-dos", "--backend", "matplotlib",
             "--format", "png", "-o", str(run_dir / "mpl"))
    assert r.returncode != 0 and "cannot draw" in r.stdout and "--log-dos" in r.stdout, r.stdout[-600:]
