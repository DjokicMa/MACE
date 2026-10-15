"""k-path tick labels in mace plotting --band: each at its own k, never overlapping.

A seekpath oA2/oC2 path (Γ-Y-F0|Δ0-Γ-Z-B0|G0-T-Y|Γ-S-R-Z-T, a slipped AB
graphite cell) has two segments 14x shorter than the others. The plot spread
a segment's labels evenly, so Y sat half-way between Γ and F0 instead of
next to F0; at their real positions Y and F0 would overlap, so the second
drops to a row below.
"""
import sys
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT
from test_gui_results import SLIP_TICKS, write_band_crystal, write_band_path

sys.path.insert(0, str(REPO_ROOT / "Plotting"))


def _cli_axes(monkeypatch, path, no_gaps=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import ipBANDS_V2 as ipb
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(ipb.plt, "close", lambda *a, **k: None)
    args = SimpleNamespace(list_segments=False, segments=None, path=None, no_gaps=no_gaps, gap_width=0.02,
                           E_lower=-10, E_upper=10, fixed_width=None, fixed_height=None, alpha=1.0,
                           spin_up_color=ipb.DEFAULT_SPIN_UP_COLOR, spin_down_color=ipb.DEFAULT_SPIN_DOWN_COLOR,
                           auto_width=True, formats="png")          # what mace plotting passes
    d3 = path.with_name(path.name.replace(".BAND.DAT", ".d3"))
    ipb.ipBANDS(str(path), "m", {"d3": str(d3)}, args, str(path.parent))
    fig = saved[-1]
    fig.canvas.draw()
    return fig, fig.axes[0]


def _overlaps(fig, ax):
    """Pairs of printed tick labels whose boxes overlap (a dropped label's box
    also covers its empty first line)."""
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    boxes = []
    for t in ax.get_xticklabels():
        if not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        if t.get_text().startswith("\n"):            # the ink: one line's height at the bottom
            text = t.get_text()
            t.set_text(text.lstrip("\n"))
            h = t.get_window_extent(r).height
            t.set_text(text)
            b = Bbox([[b.x0, b.y0], [b.x1, b.y0 + h]])
        boxes.append(b)
    return [(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]
            if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5]


@pytest.mark.parametrize("no_gaps", [False, True])
def test_cli_labels_sit_at_their_k_and_never_overlap(monkeypatch, tmp_path, no_gaps):
    fig, ax = _cli_axes(monkeypatch, write_band_path(tmp_path), no_gaps)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    pos = list(ax.get_xticks())
    g, y = pos[labels.index("Γ")], pos[[lab.strip() for lab in labels].index("Y")]
    f0 = pos[next(i for i, lab in enumerate(labels) if "F_{0}" in lab)]
    # Y at its own k: 0.534 of the 0.573 to F0 (it was spread evenly, at 0.5)
    assert (y - g) / (f0 - g) == pytest.approx(SLIP_TICKS[1] / SLIP_TICKS[2], abs=0.02)
    assert any(lab.startswith("\n") for lab in labels)       # one of a crowded pair drops a row
    assert not _overlaps(fig, ax)


def test_cli_roomy_path_keeps_one_row(monkeypatch, tmp_path):
    fig, ax = _cli_axes(monkeypatch, write_band_crystal(tmp_path)[0], no_gaps=True)   # one "X|K" label
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert not any(lab.startswith("\n") for lab in labels) and not _overlaps(fig, ax)


def test_stagger_tick_labels_rows():
    import ipBANDS_V2 as ipb
    out = ipb.stagger_tick_labels([0, 0.5, 0.52, 1.0], ["Γ", "Y", "$F_{0}$", "Z"], 0, 1, 4.0)
    assert out == ["Γ", "Y", "\n$F_{0}$", "Z"]
    assert ipb.stagger_tick_labels([0, 0.5, 1.0], ["Γ", "Y", "Z"], 0, 1, 4.0) == ["Γ", "Y", "Z"]
