"""mace plotting's band/DOS backends (1.1.6), run as the user runs them (``mace_cli`` in a
subprocess) on the real Fe ferromagnet of tests/data/spin_fe (its DOSS deck keeps MACE's
"105 -1" print record):

* plotly (the default where plotly and kaleido can export): --band, --dos and the new
  --band-dos write the GUI's figure under the old file names; --format picks one format;
* matplotlib: --backend, [plots] backend, MACE_PLOT_BACKEND, and the fallback when kaleido
  cannot be imported (said once);
* the bugs: the DOS total is "Total" (was "105 -1"), --dos-colors total draws it, the DOS
  takes --spin-up-color/--spin-down-color, --help lists the style options.
"""
import os
import shutil
import struct
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
FE = REPO_ROOT / "tests" / "data" / "spin_fe"

plotly_ok = pytest.mark.skipif(not __import__("importlib").util.find_spec("kaleido"), reason="kaleido not installed")


def _run(cwd, *args, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k != "MACE_PLOT_BACKEND"}
    e.update({"MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", **(env or {})})
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "plotting", *args], cwd=str(cwd),
                       capture_output=True, text=True, timeout=timeout, env=e)
    out = p.stdout + p.stderr
    assert "Traceback (most recent call last)" not in out, out[-3000:]
    return p, out


@pytest.fixture
def fe(tmp_path):
    for f in FE.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    return tmp_path


def _png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


@plotly_ok
def test_plotly_band_dos_and_combined(fe):
    p, out = _run(fe, "--band")
    assert p.returncode == 0, out
    assert (fe / "fe.BANDS.png").is_file() and (fe / "fe.BANDS.svg").is_file()
    w, h = _png_size(fe / "fe.BANDS.png")
    assert (w, h) == (round(760 * 300 / 96), round(420 * 300 / 96))          # 300 dpi
    svg = (fe / "fe.BANDS.svg").read_text()
    assert "spin up (α)" in svg and "spin down (β)" in svg and "H|P" in svg
    assert "rgb(31, 78, 156)" in svg and "rgb(192, 57, 43)" in svg           # blue/red, the default
    p, out = _run(fe, "--dos", "--format", "svg")
    assert p.returncode == 0, out
    assert (fe / "fe.DOSS.svg").is_file() and not (fe / "fe.DOSS.png").exists()
    dos = (fe / "fe.DOSS.svg").read_text()
    assert "Total" in dos and "105 -1" not in dos and "DOS (states/eV)" in dos
    p, out = _run(fe, "--band-dos", "fe_doss.DOSS.DAT", "--format", "pdf")
    assert p.returncode == 0, out
    assert (fe / "fe.BAND_DOS.pdf").read_bytes()[:4] == b"%PDF"


@plotly_ok
def test_plotly_dos_colors_total_is_not_empty(fe):
    p, out = _run(fe, "--dos", "--dos-colors", "total", "--format", "svg")
    assert p.returncode == 0, out
    svg = (fe / "fe.DOSS.svg").read_text()
    # the total and its fill: the spin pair's two lines
    assert "Total" in svg and "rgb(31, 78, 156)" in svg and "rgb(192, 57, 43)" in svg


@plotly_ok
def test_all_writes_the_combined_figure(fe):
    p, out = _run(fe, "--all", "--format", "svg")
    assert p.returncode == 0, out
    for name in ("fe.BANDS.svg", "fe.DOSS.svg", "fe.BAND_DOS.svg"):
        assert (fe / name).is_file(), (name, out[-2000:])


def test_matplotlib_backend_flag_folder_and_env(fe):
    p, out = _run(fe, "--band", "--backend", "matplotlib", "--format", "svg")
    assert p.returncode == 0 and "Saved:" in out, out             # ipBANDS_V2's own line
    assert "<!-- Created with matplotlib" in (fe / "fe.BANDS.svg").read_text() or \
        "matplotlib" in (fe / "fe.BANDS.svg").read_text()[:2000]
    (fe / "mace-folder.toml").write_text('[plots]\nbackend = "matplotlib"\n')
    p, out = _run(fe, "--dos", "--format", "svg", env={"MACE_FOLDER_SETTINGS": ""})
    assert p.returncode == 0 and "Saved:" in out, out
    (fe / "mace-folder.toml").unlink()
    p, out = _run(fe, "--dos", "--format", "svg", env={"MACE_PLOT_BACKEND": "matplotlib"})
    assert p.returncode == 0 and "Saved:" in out, out


def test_matplotlib_combined_figure(fe):
    p, out = _run(fe, "--band-dos", "--backend", "matplotlib", "--format", "png")
    assert p.returncode == 0, out
    assert (fe / "fe.BAND_DOS.png").stat().st_size > 10000


def test_falls_back_to_matplotlib_without_kaleido(fe, tmp_path_factory):
    shim = tmp_path_factory.mktemp("shim")
    (shim / "kaleido").mkdir()
    (shim / "kaleido" / "__init__.py").write_text("raise ImportError('no kaleido here')\n")
    env = {"PYTHONPATH": str(shim) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p, out = _run(fe, "--band", "--format", "svg", env=env)
    assert p.returncode == 0, out
    assert out.count("drawn with matplotlib") == 1 and "Saved:" in out, out
    # without plotly (CI's base jobs) the import of plotly.io stops it first; the fallback is the same
    import importlib.util
    why = "ImportError: no kaleido here" if importlib.util.find_spec("plotly") else "No module named 'plotly'"
    assert why in out and 'pip install "plotly>=5.0.0,<6.2" "kaleido==0.2.1"' in out, out
    assert (fe / "fe.BANDS.svg").is_file()


def test_plotly_7_with_kaleido_0_2_1_says_why_and_how_to_fix(monkeypatch):
    """1.1.6 review H1: plotly 7 refuses kaleido 0.2.1 with a message that starts with a newline, so
    the note read "(RuntimeError: )". It now names the error, the cause and the install line.
    plotly and kaleido are faked, so this runs where neither is installed."""
    import types
    from mace.plotting import plotly_bands as PB
    plotly, pio = types.ModuleType("plotly"), types.ModuleType("plotly.io")
    plotly.io = pio
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.io", pio)
    monkeypatch.setitem(sys.modules, "kaleido", types.ModuleType("kaleido"))

    def refuse(*a, **k):
        raise RuntimeError("\nImage export requires the Kaleido package, v1.0.0 or greater, which can be installed using pip:\n"
                           "    $ pip install --upgrade kaleido\n")
    pio.to_image = refuse
    plotly.__version__ = "7.1.0"
    monkeypatch.setattr(PB, "_KALEIDO_OK", None)
    ok, why = PB.available()
    assert not ok
    assert why.startswith("RuntimeError: Image export requires the Kaleido package, v1.0.0 or greater")
    assert "plotly 7.1.0 needs kaleido 1" in why and why.endswith('pip install "plotly>=5.0.0,<6.2" "kaleido==0.2.1"')
    monkeypatch.setattr(PB, "_KALEIDO_OK", None)


def test_requirements_pin_a_plotly_that_kaleido_0_2_1_can_export_with():
    """plotly 7 cannot export through kaleido 0.2.1 (a fresh install got 7.1.0 and silently fell back to
    matplotlib); the pin keeps plotly at the versions tested with it (5.24.1 to 6.1.2)."""
    from packaging.requirements import Requirement
    from mace.plotting import plotly_bands as PB
    reqs = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            r = Requirement(line)
            reqs[r.name.lower()] = r.specifier
    assert str(reqs["kaleido"]) == "==0.2.1"
    for good in ("5.24.1", "6.0.1", "6.1.2"):
        assert reqs["plotly"].contains(good), good
    for bad in ("6.2.0", "7.0.0", "7.1.0"):
        assert not reqs["plotly"].contains(bad), bad
    assert Requirement(PB.PLOTLY_PIN).specifier == reqs["plotly"]


def test_matplotlib_dos_takes_the_spin_colors_and_draws_the_total(fe):
    """Bugs (a) and (b): the total is there (and alone with --dos-colors total), in the spin pair given."""
    p, out = _run(fe, "--dos", "--backend", "matplotlib", "--format", "svg", "--dos-colors", "total",
                  "--spin-up-color", "#123456", "--spin-down-color", "#abcdef")
    assert p.returncode == 0, out
    svg = (fe / "fe.DOSS.svg").read_text()
    assert "#123456" in svg and "#abcdef" in svg
    assert "105 -1" not in svg


def test_help_lists_the_style_options():
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "plotting", "--help"],
                       capture_output=True, text=True, timeout=120, env={**os.environ, "COLUMNS": "1000"})
    h = p.stdout
    for flag in ("--band-dos", "--backend", "--style", "--spin-colors", "--spin-up-color", "--spin-down-color",
                 "--spin-down-style", "--spin-down-solid", "--dos-colors", "--dos-spin", "--no-dos-total", "--units"):
        assert flag in h, flag
    assert "0.3" not in h.split("--alpha", 1)[1].splitlines()[0]


def test_band_dos_is_discovered_only_with_a_partner(fe, tmp_path_factory):
    from mace.plotting import detect
    from mace.plotting.registry import PlotKind
    found = detect.discover(str(fe))
    assert [p.split("/")[-1] for p in found[PlotKind.BANDDOS]] == ["fe_band.BAND.DAT"]
    alone = tmp_path_factory.mktemp("alone")
    shutil.copy2(fe / "fe_band.BAND.DAT", alone / "fe_band.BAND.DAT")
    assert detect.discover(str(alone))[PlotKind.BANDDOS] == []
    # a positional band file is still a band file (the band entry comes first)
    assert detect.classify_file(str(fe / "fe_band.BAND.DAT")) == PlotKind.BAND


@plotly_ok
def test_interactive_menu_offers_band_dos(fe):
    e = {k: v for k, v in os.environ.items() if k != "MACE_PLOT_BACKEND"}
    e.update({"MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none"})
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "plotting"], cwd=str(fe), input="3\n\n\n\n\n",
                       capture_output=True, text=True, timeout=300, env=e)
    out = p.stdout + p.stderr
    assert "band structure + DOS" in out and "Traceback" not in out, out[-3000:]
    assert (fe / "fe.BAND_DOS.png").is_file(), out[-3000:]
