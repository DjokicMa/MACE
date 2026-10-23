"""--band-style reaches every renderer: the GUI and the plotly CLI figure (always did), the matplotlib
plotters and the gallery (once they drew the old ink look whatever the style said)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("/mnt/iscsi/UsefulScripts/Codebase/reorganization/test")


def _panel(metal=False):
    x = [0.0, 1.0, 2.0]
    return {"series": [{"x": x, "y": [-1.0, -0.5, -1.0], "name": "bands", "role": "up"}], "xlabel": "", "ylabel": "E",
            "xlim": [0, 2], "ylim": [-3, 3], "energy": "y", "fermi": 0.0, "eflines": [0.0], "ticks": [0, 2], "ticklabels": ["A", "B"],
            "width": 3, "closed": True, "metal": metal}


def _colors(look, metal=False):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mace.plotting import gallery
    fig, ax = plt.subplots()
    gallery._draw_panel(ax, _panel(metal), None, False, "default", look)
    out = [(ln.get_color(), ln.get_linestyle()) for ln in ax.get_lines() if len(ln.get_xdata()) == 3]
    plt.close(fig)
    return out


def test_style_argv_forwards_band_style():
    from mace.plotting.handlers import legacy
    assert legacy._style_argv({"band_style": "overlay"}, dos=False)[-2:] == ["--band-style", "overlay"]
    assert "--band-style" not in legacy._style_argv({"band_style": "auto"}, dos=False)
    assert "--band-style" not in legacy._style_argv({}, dos=True)


def test_gallery_command_carries_band_style():
    from mace.plotting import gallery
    assert "--band-style=ink" in gallery.command("band", ["d"], 3, "o.svg", band_style="ink")
    assert not any(a.startswith("--band-style") for a in gallery.command("band", ["d"], 3, "o.svg"))


def test_gallery_draws_the_band_style():
    from mace.plotting import presets
    pair = presets.spin_colors("default", "band")
    single = _colors({"band_style": "single"})
    ink = _colors({"band_style": "ink"})
    overlay = _colors({"band_style": "overlay"})
    auto_gap = _colors({"band_style": "auto"})
    auto_metal = _colors({"band_style": "auto"}, metal=True)
    assert single[0][0] == pair["up"] and len(single) == 1
    assert ink[0][0] != pair["up"] and len(ink) == 1
    assert len(overlay) == 2 and overlay[1][0] == pair["down"] and overlay[1][1] != "-"
    assert auto_gap == single                      # Auto: single with a gap
    assert len(auto_metal) == 2                    # Auto: overlay for a metal
    assert _colors({"band_style": "single", "band_color": "#ff8800"})[0][0] == "#ff8800"


def test_gallery_spin_look_has_the_default_single():
    # no band_style at all (the CLI's default None): Auto, single with a gap, never the old ink look
    from mace.plotting import presets
    assert _colors({})[0][0] == presets.spin_colors("default", "band")["up"]


def test_argparse_accepts_it_for_the_plotters():
    for script in ("ipBANDS_V2.py", "ipDOS_V2.py"):
        text = (ROOT / "Plotting" / script).read_text()
        assert "band_style" in text and "--band-style" in text


@pytest.mark.skipif(not (CORPUS / "BAND").is_dir(), reason="real CRYSTAL corpus not present")
def test_matplotlib_band_follows_band_style(tmp_path):
    src = CORPUS / "BAND"
    stem = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
    for ext in (".BAND.DAT", ".d3", ".out"):
        (tmp_path / (stem + ext)).write_bytes((src / (stem + ext)).read_bytes())
    sizes = {}
    for st in ("single", "ink"):
        out = tmp_path / st
        r = subprocess.run([sys.executable, str(ROOT / "mace_cli"), "--no-banner", "plotting", "--band", str(tmp_path / (stem + ".BAND.DAT")),
                            "--backend", "matplotlib", "--format", "png", "--band-style", st, "-o", str(out)],
                           capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"}, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        pngs = list(out.glob("*.png"))
        assert pngs, r.stdout
        sizes[st] = pngs[0].read_bytes()
    assert sizes["single"] != sizes["ink"]
