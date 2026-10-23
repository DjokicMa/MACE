"""The 1.1.6 DOS layout in the matplotlib figure (Plotting/ipDOS_V2.py, mace plotting --backend
matplotlib): mirrored spin halves with "↑ up" / "down ↓" labels and magnitude ticks, the totals filled
in the spin colors, spin down dashed in each projection's own color, a closed-shell total as a neutral
fill under an ink line, the grouped legend (Spin, Projections), a dotted E_F, the 2-4 largest
projections by default and the opt-in smoothing (the plotly figure's rules, presets.py helpers)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FE = REPO_ROOT / "tests" / "data" / "spin_fe"
DIA = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"
DIA_DIR = REPO_ROOT / "test" / "DOSS"

from mace.utils.plot_labels import fermi_key, mathtext  # noqa: E402

FERMI_KEY = mathtext(fermi_key(0))     # the dotted E_F line's legend entry (E$_{\mathregular{F}}$)

sys.path.insert(0, str(REPO_ROOT / "Plotting"))


def _figure(monkeypatch, work, src, material, **kw):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import ipDOS_V2 as ip
    work.mkdir(parents=True, exist_ok=True)
    for ext in ("_doss.DOSS.DAT", "_doss.d3"):
        shutil.copy(src / f"{material}{ext}", work / f"{material}{ext}")
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(ip.plt, "close", lambda *a, **k: None)
    monkeypatch.chdir(work)
    ip.ipDOS(material, -6, 6, "both", formats="png", output_dir=str(work), dos_colors="auto", **kw)
    assert saved
    return ip, saved[-1].axes[0]


def test_spin_dos_is_mirrored_with_labels_fills_and_groups(monkeypatch, tmp_path):
    from mace.plotting import presets as P
    from mace.plotting.plotly_bands import tokens
    ip, ax = _figure(monkeypatch, tmp_path, FE, "fe")
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(-hi) and hi > 0                         # symmetric range
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks and all(not t.startswith(("-", "−")) for t in ticks)  # magnitudes on both sides
    pair = P.spin_colors("default")
    labels = {t.get_text(): t.get_color() for t in ax.texts}
    assert labels["↑ up"] == pair["up"] and labels["down ↓"] == pair["down"]
    # the totals lightly filled in their spin's color (presets.spin_fills)
    fills = P.spin_fills(pair, "default")
    faces = [tuple(np.round(c.get_facecolor()[0], 3)) for c in ax.collections]
    for want in (fills["up"], fills["down"]):
        assert tuple(np.round(ip._mpl_color(want), 3)) in faces, (want, faces)
    # each drawn projection: solid on the right, dashed on the left, in one color clear of the spin pair
    by_color = {}
    for ln in ax.get_lines():
        if ln.get_xydata().size > 4:
            by_color.setdefault(ln.get_color(), []).append(ln.get_linestyle())
    proj = [c for c in by_color if c not in (pair["up"], pair["down"])]
    assert 2 <= len(proj) <= 4                                         # the largest projections only
    for c in proj:
        assert sorted(by_color[c])[0] != sorted(by_color[c])[-1] and "-" in by_color[c]
    # the grouped legend: Spin, the pair, Projections, swatches (Total first), then E_F
    leg = ax.get_legend()
    names = [t.get_text() for t in leg.get_texts()]
    assert names[:5] == ["Spin", "spin up (α)", "spin down (β)", "Projections", "Total"] and names[-1] == FERMI_KEY
    assert leg.get_texts()[0].get_color() == tokens("default")["--text-2"]
    fermi = next(ln for ln in ax.get_lines() if ln.get_label() == FERMI_KEY)
    # dotted (presets.MPL_DASH["dot"]): dashed reads as spin down
    assert tuple(fermi._unscaled_dash_pattern[1]) == tuple(P.MPL_DASH["dot"][1])


def test_smoothing_is_drawn_and_labeled(monkeypatch, tmp_path):
    _ip, plain = _figure(monkeypatch, tmp_path / "plain", FE, "fe")
    _ip, smooth = _figure(monkeypatch, tmp_path / "smooth", FE, "fe", dos_smooth=0.1)
    assert any("σ = 0.1 eV" in t.get_text() for t in smooth.texts)
    assert not any("σ" in t.get_text() for t in plain.texts)
    peak = lambda ax: np.max(np.abs(next(ln for ln in ax.get_lines() if ln.get_label() == "Total").get_xdata()))  # noqa: E731
    assert peak(smooth) < peak(plain)                                  # display only: the peak softened


@pytest.mark.skipif(not (DIA_DIR / f"{DIA}_doss.DOSS.DAT").is_file(), reason="test/ data corpus not present")
def test_closed_shell_total_is_a_neutral_fill_under_an_ink_line(monkeypatch, tmp_path):
    from mace.plotting import presets as P
    from mace.plotting.plotly_bands import tokens
    # Band style Ink (1.1.6: Auto is Single, the spin pair's first color, as the GUI draws it)
    ip, ax = _figure(monkeypatch, tmp_path, DIA_DIR, DIA, band_style="ink")
    ink = tokens("default")["--plot-ink"]
    assert ax.get_xlim()[0] == 0                                       # one-sided
    total = next(ln for ln in ax.get_lines() if ln.get_label() == "Total")
    assert total.get_color() == ink
    faces = [tuple(np.round(c.get_facecolor()[0], 3)) for c in ax.collections]
    assert tuple(np.round(ip._mpl_color(P.neutral_fill(ink, "default")), 3)) in faces
    names = [t.get_text() for t in ax.get_legend().get_texts()]
    assert names[:2] == ["Spin", "α = β (closed shell)"] and "spin up (α)" not in names
    assert not {"↑ up", "down ↓"} & {t.get_text() for t in ax.texts}


def test_custom_x_scale_draws_the_given_range(monkeypatch, tmp_path):
    """custom:NEG:POS left xlimit unset (UnboundLocalError, no figure; 1.1.5 too)."""
    ip, ax = _figure(monkeypatch, tmp_path, FE, "fe", x_scale_mode="custom:5:5")
    assert ax.get_xlim() == pytest.approx((-5, 5))
