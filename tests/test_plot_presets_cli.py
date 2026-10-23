"""mace plotting's plot styles and the 1.1.6 plotting bug fixes, run as a user runs them
(``python mace_cli plotting ...`` as a subprocess) on real runs from the test/ corpus.

* --style, --dos-colors, --spin-style, --spin-dos and --no-dos-total reach the figure;
* B1: a bare G that is the zone center reads Γ (MACE's default paths), GAMMA's G stays G;
* B2: one rule for the spin DOS: a closed-shell run one-sided, a spin-polarized run
  mirrored, in mace plotting and in the GUI's data alike;
* B6: spin and difference volumes draw + blue and − red, and the captions say so;
* B8: -o puts band and DOS figures where it says; without it they stay beside the data;
* T8: a band/DOS figure is 3.5 x 5 in without a title;
* a folder's [plots] style is mace plotting's default.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data

MACE_CLI = REPO_ROOT / "mace_cli"
AA = "4LG_2x2_AA_opt_HSESOL3C_optimized"            # spin-polarized (alpha/beta up to 130 meV apart)
DIA = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"    # closed shell
pytestmark = pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")


def _plotting(args, cwd, env=None, timeout=300):
    # the matplotlib plotters' figures (tests/test_plotting_backends.py checks plotly's)
    e = {**os.environ, "MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": "matplotlib", **(env or {})}
    p = subprocess.run([sys.executable, str(MACE_CLI), "plotting", *args], cwd=str(cwd), capture_output=True,
                       text=True, timeout=timeout, env=e)
    out = p.stdout + p.stderr
    assert "Traceback (most recent call last)" not in out, out[-3000:]
    return p, out


def _copy(dest, *patterns):
    dest.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        src = find_data(pat)
        shutil.copy(src, dest / Path(src).name)
    return dest


@pytest.fixture
def aa(tmp_path):
    return _copy(tmp_path / "aa", f"BAND/{AA}_band.BAND.DAT", f"BAND/{AA}_band.d3",
                 f"DOSS/{AA}_doss.DOSS.DAT", f"DOSS/{AA}_doss.d3")


@pytest.fixture
def dia(tmp_path):
    return _copy(tmp_path / "dia", f"DOSS/{DIA}_doss.DOSS.DAT", f"DOSS/{DIA}_doss.d3", f"DOSS/{DIA}_doss.out")


def _svg_colors(path):
    return set(re.findall(r"stroke:\s*(#[0-9a-f]{6})", Path(path).read_text()))


def test_o_puts_band_and_dos_figures_there(aa, tmp_path):
    """B8: -o DIR is where the figures go; without -o they stay beside the data."""
    out = tmp_path / "figs"
    p, log = _plotting(["--band", f"{AA}_band.BAND.DAT", "-o", str(out), "--format", "svg"], aa)
    assert p.returncode == 0, log[-2000:]
    assert (out / f"{AA}.BANDS.svg").is_file() and not list(aa.glob("*.BANDS.*")), sorted(os.listdir(aa))
    p, log = _plotting(["--dos", f"{AA}_doss.DOSS.DAT"], aa)
    assert p.returncode == 0 and (aa / f"{AA}.DOSS.png").is_file(), log[-2000:]


def test_spin_dos_one_rule(aa, dia, tmp_path):
    """B2: closed shell one-sided, spin-polarized mirrored (and --spin-dos overrides), as the GUI's data say."""
    from mace.gui import results
    assert results.dos_data(dia / f"{DIA}_doss.DOSS.DAT")["spin_polarized"] is False
    assert results.dos_data(aa / f"{AA}_doss.DOSS.DAT")["spin_polarized"] is True
    _, log = _plotting(["--dos", f"{DIA}_doss.DOSS.DAT", "-o", str(tmp_path / "a")], dia)
    assert "Spin: spin-polarized, alpha = beta (non-magnetic), total alpha + beta drawn, drawn one-sided" in log
    _, log = _plotting(["--dos", f"{AA}_doss.DOSS.DAT", "-o", str(tmp_path / "b")], aa)
    assert "Spin: spin-polarized, drawn mirrored" in log
    _, log = _plotting(["--dos", f"{AA}_doss.DOSS.DAT", "-o", str(tmp_path / "c"), "--spin-dos", "overlaid"], aa)
    assert "drawn overlaid" in log
    _, log = _plotting(["--dos", f"{DIA}_doss.DOSS.DAT", "-o", str(tmp_path / "d"), "--spin-dos", "mirrored"], dia)
    assert "Spin: spin-polarized, alpha = beta (non-magnetic), drawn mirrored" in log


def test_style_and_spin_flags_reach_the_band_figure(aa, tmp_path):
    """--style publication: black up / vermilion down; default: the interim blue / red pair;
    --spin-up-color / --spin-down-color override; --spin-style changes spin down's dashes."""
    from mace.plotting import presets
    runs = {"default": [], "publication": ["--style", "publication"],
            "custom": ["--spin-up-color", "#123456", "--spin-down-color", "#654321", "--spin-style", "dot"]}
    for name, extra in runs.items():
        p, log = _plotting(["--band", f"{AA}_band.BAND.DAT", "-o", str(tmp_path / name), "--format", "svg", *extra], aa)
        assert p.returncode == 0, log[-2000:]
    c = {n: _svg_colors(tmp_path / n / f"{AA}.BANDS.svg") for n in runs}
    assert {presets.SPIN_COLORS["light"]["up"], presets.SPIN_COLORS["light"]["down"]} <= c["default"]
    assert {"#000000", "#d55e00"} <= c["publication"] and "#1f4e9c" not in c["publication"]
    assert {"#123456", "#654321"} <= c["custom"]
    dashes = lambda n: set(re.findall(r"stroke-dasharray:\s*([0-9.,\s]+)", (tmp_path / n / f"{AA}.BANDS.svg").read_text()))  # noqa: E731
    assert dashes("custom") != dashes("default")


def test_dos_colors_flag_picks_the_lines(dia, tmp_path):
    """--dos-colors element draws the element's line only (named "C"), orbital its s/p/d ("C s"...),
    total the total; --no-dos-total leaves the total out."""
    import matplotlib
    matplotlib.use("Agg")
    for mode in ("total", "element", "orbital"):
        p, log = _plotting(["--dos", f"{DIA}_doss.DOSS.DAT", "-o", str(tmp_path / mode), "--format", "svg",
                            "--dos-colors", mode], dia)
        assert p.returncode == 0, log[-2000:]
    from mace.plotting import presets
    assert presets.ORBITAL_COLORS["p"] in _svg_colors(tmp_path / "orbital" / f"{DIA}.DOSS.svg")
    assert presets.ORBITAL_COLORS["p"] not in _svg_colors(tmp_path / "total" / f"{DIA}.DOSS.svg")
    p, log = _plotting(["--dos", f"{DIA}_doss.DOSS.DAT", "-o", str(tmp_path / "nt"), "--format", "svg",
                        "--dos-colors", "orbital", "--no-dos-total"], dia)
    assert p.returncode == 0
    # the total's light neutral fill (1.1.6 N1: the total's color at presets.neutral_fill's opacity; the total
    # is in the Band style's color, Auto = Single = the spin pair's first color, as the GUI draws it)
    fills = lambda n: set(re.findall(r"fill:\s*(#[0-9a-f]{6}); fill-opacity:\s*([0-9.]+)",  # noqa: E731
                                     (tmp_path / n / f"{DIA}.DOSS.svg").read_text()))
    first = presets.SPIN_COLORS["light"]["up"]
    want = (first, presets.neutral_fill(first, "default").rsplit(",", 1)[1].rstrip(")"))
    assert want in fills("orbital") and want not in fills("nt")


def test_figures_are_a_column_wide_without_a_title(monkeypatch, dia):
    """T8: 3.5 x 5 in, no title, 11 pt axis labels, 10 pt ticks (DOS; the band plot shares the layout)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipDOS_V2 as ipd
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.chdir(dia)
    ipd.ipDOS(DIA, -6, 6, "orbital", formats="png", output_dir=str(dia))
    fig = saved[-1]
    ax = fig.axes[0]
    assert ax.get_title() == "" and ax.xaxis.label.get_size() == 11 and fig.get_size_inches()[1] == pytest.approx(5)
    assert ax.get_xlim()[0] == 0                                    # B2: one-sided
    names = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "C p" in names and "C (p)" not in names                  # the GUI's names


def test_bare_g_reads_gamma_only_at_the_zone_center(tmp_path):
    """B1: MACE's default path writes G for 0 0 0 -> Γ; seekpath's GAMMA paths keep a bare G (tI2, mC2)."""
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipBANDS_V2 as ipb
    real = find_data(f"BAND/{AA}_band.d3")
    labels = ipb.parse_kpoint_path_and_segments(str(real))[0]
    assert labels == ["V", "Y", "Γ", "Z", "T", "R", "Γ", "X", "U", "Γ"], labels
    d3 = tmp_path / "bct_band.d3"
    d3.write_text("BAND\nbct - Band Structure - SeeKPath - GAMMA-X-G\n2 4 100 1 10 1 0\n0 0 0  2 0 0\n2 0 0  1 1 1\nEND\n")
    assert ipb.parse_kpoint_path_and_segments(str(d3))[0] == ["Γ", "X", "G"]
    d3.write_text("BAND\nx - Band Structure - default - X-G-M\n2 4 100 1 10 1 0\n2 0 0  1 1 1\n1 1 1  2 2 0\nEND\n")
    assert ipb.parse_kpoint_path_and_segments(str(d3))[0] == ["X", "G", "M"]          # G is not 0 0 0 here
    d3.write_text("BAND\nx - Band Structure - default - X-G-M\n2 4 100 1 10 1 0\n2 0 0  0 0 0\n0 0 0  2 2 0\nEND\n")
    assert ipb.parse_kpoint_path_and_segments(str(d3))[0] == ["X", "Γ", "M"]
    # the GUI's band data carry the same label
    from mace.gui import results
    shutil.copy(find_data(f"BAND/{AA}_band.BAND.DAT"), tmp_path / f"{AA}_band.BAND.DAT")
    shutil.copy(real, tmp_path / f"{AA}_band.d3")
    assert "Γ" in results.band_data(tmp_path / f"{AA}_band.BAND.DAT")["labels"]


def test_signed_volumes_plus_blue_and_captions_say_so():
    """B6: spin and difference isosurfaces + blue / − red (the GUI's 3D view and slices draw the same);
    the captions name those colors; a potential in chemistry colors (the default) the same, physics swaps
    them; --style vesta: yellow / cyan."""
    E = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly", reason="the cube engine imports plotly")
    E.use_style("default")
    E.use_potential_colors("chemistry")
    assert E.get_iso_color("spin", 1) == "#2563eb" and E.get_iso_color("spin", -1) == "#dc2626"
    assert E.get_iso_color("density_difference", 1) == "#2563eb"
    assert E.get_iso_color("potential", 1) == "#2563eb" and E.get_iso_color("potential", -1) == "#dc2626"
    assert E.get_plotly_colorscale("spin") == "RdBu" and E.get_plotly_colorscale("potential") == "RdBu"
    E.use_potential_colors("physics")
    assert E.get_iso_color("potential", 1) == "#dc2626" and E.get_plotly_colorscale("potential") == "RdBu_r"
    assert E.get_iso_color("spin", 1) == "#2563eb" and E.get_plotly_colorscale("spin") == "RdBu"
    E.use_potential_colors("chemistry")
    src = (REPO_ROOT / "mace" / "plotting" / "engines" / "crystal_cubeviz_plotly.py").read_text()
    assert "<b>Red (+):</b> Spin" not in src and "{PW} (+):</b> Spin-up excess" in src
    E.use_style("vesta")
    assert (E.ISO["plus"], E.ISO["plus_word"]) == ("#e6c200", "Yellow")
    E.use_style("default")


def _pot_cube(path):
    """A small signed potential cube (8³, + on one side, − on the other)."""
    import math
    n = 8
    vals = [math.sin(2 * math.pi * i / n) * 0.05 for i in range(n) for _ in range(n) for _ in range(n)]
    head = ["MACE test potential", "POT3", "    1    0.000000    0.000000    0.000000"]
    head += [f"    {n}    " + "    ".join("0.600000" if j == k else "0.000000" for j in range(3)) for k in range(3)]
    head += ["    8    8.000000    2.100000    2.100000    2.100000"]
    rows = [" ".join(f"{v: .5E}" for v in vals[i:i + 6]) for i in range(0, len(vals), 6)]
    path.write_text("\n".join(head + rows) + "\n")
    return path


def _low_color(fig):
    """The (r, g, b) of the colorscale's low end of the figure's first heatmap/contour trace."""
    import plotly.colors as pc
    t = next(t for t in fig.data if t.type in ("heatmap", "contour", "surface"))
    cs = pc.get_colorscale(t.colorscale) if isinstance(t.colorscale, str) else list(t.colorscale)
    low = cs[-1][1] if t.reversescale else cs[0][1]
    return pc.unlabel_rgb(pc.convert_colors_to_same_type(low, "rgb")[0][0])


def test_potential_colors_chemistry_default_and_physics(tmp_path):
    """Electrostatic potential: chemistry colors by default (red = negative, blue = positive) in the
    slices and isosurfaces, captions naming what is drawn; --potential-colors physics: red = positive."""
    pytest.importorskip("plotly")                   # the engine imports without plotly; the figures need it
    E = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly", reason="the cube engine imports plotly")
    import contextlib
    import io
    E.use_style("default")
    cube_path = _pot_cube(tmp_path / "x_POT.CUBE")
    with contextlib.redirect_stdout(io.StringIO()):
        cube = E.CubeFile(str(cube_path))
    assert cube.data_type == "potential"
    for scheme, low_red in (("chemistry", True), ("physics", False)):
        E.use_potential_colors(scheme)
        with contextlib.redirect_stdout(io.StringIO()):
            fig = E.plot_slice_plotly(cube, "x", 2)
        r, g, b = _low_color(fig)
        assert (r > b) == low_red, (scheme, r, g, b)
        cap = E._potential_caption()
        assert cap.startswith("<b>Red (−):</b> Negative" if low_red else "<b>Blue (−):</b> Negative"), cap
        assert ("<b>Blue (+):</b> Positive" in cap) == low_red
    E.use_potential_colors("chemistry")
    # the engine's flag and mace plotting's handler argv
    from mace.plotting.handlers import cube as H
    cfg = {"view": "slice", "style": "default", "potential_colors": "physics"}
    argv = H._common_argv(cfg)
    assert argv[argv.index("--potential-colors") + 1] == "physics"
    assert "--potential-colors" not in H._common_argv({**cfg, "potential_colors": "chemistry"})


def test_folder_potential_colors_key_and_old_name(tmp_path):
    """[plots] potential_colors is mace plotting's --potential-colors default and the GUI's Potential colors;
    1.1.6's [viewer] potential_colors = "cli" (red = positive) reads as physics."""
    from mace.utils import folder_settings as fs
    pytest.importorskip("tomllib")
    f = tmp_path / "mace-folder.toml"
    f.write_text('[plots]\npotential_colors = "physics"\n')
    s = fs.parse(f.read_text(), f)
    assert fs.plot_defaults(s)["potential_colors"] == "physics" and s.gui()["volume.mepColors"] == "physics"
    f.write_text('[viewer]\npotential_colors = "cli"\n')
    s = fs.parse(f.read_text(), f)
    assert s.values == {"plots.potential_colors": "physics"} and any("is now plots.potential_colors" in w for w in s.warnings)


def test_gallery_command_carries_the_style():
    from mace.plotting import gallery
    a = gallery.command("dos", ["x.DOSS.DAT"], 3, "g.svg", style="grayscale", dos_colors="element")
    assert a[a.index("--style") + 1] == "grayscale" and a[a.index("--dos-colors") + 1] == "element"
    assert "--style" not in gallery.command("dos", ["x.DOSS.DAT"], 3, "g.svg")


def test_folder_plots_style_is_the_default(aa, tmp_path):
    """A folder's mace-folder.toml [plots] style is mace plotting's --style unless one is given."""
    (aa / "mace-folder.toml").write_text('[plots]\nstyle = "publication"\nspin_down_style = "dot"\n')
    p, log = _plotting(["--band", f"{AA}_band.BAND.DAT", "-o", str(tmp_path / "f"), "--format", "svg"], aa,
                       env={"MACE_FOLDER_SETTINGS": str(aa / "mace-folder.toml")})
    assert p.returncode == 0, log[-2000:]
    assert "--style publication" in log
    assert {"#000000", "#d55e00"} <= _svg_colors(tmp_path / "f" / f"{AA}.BANDS.svg")
