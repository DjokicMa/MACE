"""mace plotting --band / --dos: the saved figure's legend sits outside the axes.

The band and DOS plotters placed their legend INSIDE the axes, in the corner
with the fewest points (or the band gap): a metal or a busy DOS has no empty
corner, so the legend covered bands or DOS curves (and the DOS's band-gap box
sat over the curves too). Both now put the legend right of the axes and of
everything drawn beside them (the DOS's tick labels and axis label are on the
right), widening the figure when the legend needs more room than the right
margin, the axes keeping their size and place in inches so band and DOS
figures still line up. The GUI's band + DOS plot does the same
(tests/gui/test_plot_legend_layout_e2e.py).
"""
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from test_dos_spin_window import write_doss_spin
from test_gui_results import write_band_crystal

sys.path.insert(0, str(REPO_ROOT / "Plotting"))

MANY = ["Pb (s)", "Pb (p)", "Pb (d)", "I (s)", "I (p)", "I (d)", "Cs (s)", "Cs (p)", "Cs (d)", "Cs (f)"]


def write_doss_many(directory, material="mgo", n_e=80):
    """A closed-shell DOSS.DAT with ten labeled projections and the total."""
    shells = {"S": 1, "P": 3, "D": 5, "F": 7}
    rows, start = [], 1
    for lab in MANY:
        el, sh = lab.split(" (")
        n = shells[sh[0].upper()]
        rows.append(f"{n}  {' '.join(str(start + i) for i in range(n))}  #{el} {sh[0].upper()}")
        start += n
    (directory / f"{material}_doss.d3").write_text(
        f"NEWK\n8 8\n1 0\nDOSS\n{len(MANY)} 400 -1 -1 2 14 0\n-0.5 0.5\n" + "\n".join(rows) + "\nEND\n")
    e = np.linspace(-0.26, 0.26, n_e)
    lines = [f"# NEPTS {n_e:5d} NPROJ {len(MANY) + 1:5d} NSPIN     1", "#", '@ XAXIS LABEL "E-EFERMI (HARTREE)"']
    for x in e:
        cols = [5 + 4 * np.sin((k + 3) * x) ** 2 for k in range(len(MANY))]
        lines.append(" ".join(f"{c: .4E}" for c in [x, *cols, sum(cols)]))
    lines += ["# EFERMI (HARTREE)   -0.10828", "&"]
    (directory / f"{material}_doss.DOSS.DAT").write_text("\n".join(lines) + "\n")
    return directory / f"{material}_doss.DOSS.DAT"


def _saved(monkeypatch, module):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(module.plt, "close", lambda *a, **k: None)
    return saved


def _band_figure(monkeypatch, path, **kw):
    from types import SimpleNamespace
    import ipBANDS_V2 as ipb
    saved = _saved(monkeypatch, ipb)
    args = SimpleNamespace(list_segments=False, segments=None, path=None, no_gaps=False, gap_width=0.02,
                           E_lower=-10, E_upper=10, fixed_width=None, fixed_height=None, alpha=1.0,
                           spin_up_color=ipb.DEFAULT_SPIN_UP_COLOR, spin_down_color=ipb.DEFAULT_SPIN_DOWN_COLOR,
                           auto_width=False, formats="png")
    for k, v in kw.items():
        setattr(args, k, v)
    d3 = path.with_name(path.name.replace(".BAND.DAT", ".d3"))
    ipb.ipBANDS(str(path), "m", {"d3": str(d3)}, args, str(path.parent))
    return saved[-1]


def _dos_figure(monkeypatch, directory, material, proj="orbital"):
    import ipDOS_V2 as ip
    saved = _saved(monkeypatch, ip)
    monkeypatch.chdir(directory)
    ip.ipDOS(material, -10, 10, proj, formats="png", output_dir=str(directory))
    return saved[-1]


def _overlaps(fig):
    """What the legend covers: the axes (data area), a tick label, an axis label, the
    figure edge. Boxes in display pixels from the drawn figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None
    lb = leg.get_window_extent(r)
    out = []
    if lb.overlaps(ax.get_window_extent(r)):
        out.append(("axes", lb.bounds, ax.get_window_extent(r).bounds))
    texts = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label, ax.title]
    for t in texts:
        if t.get_visible() and t.get_text().strip() and lb.overlaps(t.get_window_extent(r)):
            out.append(("label", t.get_text()))
    fb = fig.bbox
    if lb.x0 < fb.x0 - 0.5 or lb.x1 > fb.x1 + 0.5 or lb.y0 < fb.y0 - 0.5 or lb.y1 > fb.y1 + 0.5:
        out.append(("figure edge", lb.bounds, fb.bounds))
    return out


@pytest.mark.parametrize("kind", ["identical", "magnetic", "closed"])
def test_band_legend_outside_the_axes(monkeypatch, tmp_path, kind):
    if kind == "closed":
        path, _, _ = write_band_crystal(tmp_path)
    else:
        path, _, _ = write_band_crystal(tmp_path, noise=1e-4 if kind == "identical" else 0.0,
                                        beta_shift=0.002 if kind == "magnetic" else 0.0)
    fig = _band_figure(monkeypatch, path)
    assert not _overlaps(fig), _overlaps(fig)


def test_band_axes_keep_their_size_in_inches(monkeypatch, tmp_path):
    """A wider figure for the legend leaves the axes where they were, so band and DOS figures line up."""
    import ipBANDS_V2 as ipb
    path, _, _ = write_band_crystal(tmp_path, beta_shift=0.002)
    fig = _band_figure(monkeypatch, path)
    w, h = fig.get_size_inches()
    pos = fig.axes[0].get_position()
    assert h == pytest.approx(ipb.DEFAULT_FIG_HEIGHT)
    assert pos.y0 == pytest.approx(ipb.SUBPLOT_BOTTOM) and pos.y1 == pytest.approx(ipb.SUBPLOT_TOP)
    # left margin : axes width as the plotter set them, whatever the figure grew by on the right
    assert pos.x0 / pos.width == pytest.approx(ipb.SUBPLOT_LEFT / (ipb.SUBPLOT_RIGHT - ipb.SUBPLOT_LEFT))


@pytest.mark.parametrize("kind", ["spin", "magnetic", "many"])
def test_dos_legend_outside_the_axes_and_its_labels(monkeypatch, tmp_path, kind):
    if kind == "many":
        write_doss_many(tmp_path, material="m")
    else:
        write_doss_spin(tmp_path, material="m", magnetic=kind == "magnetic")
    fig = _dos_figure(monkeypatch, tmp_path, "m")
    assert not _overlaps(fig), _overlaps(fig)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    if kind == "many":
        assert all(texts.count(n) == 1 for n in MANY), texts
    # the band-gap note is no box over the curves: it heads the legend
    assert not [t for t in fig.axes[0].texts if t.get_text().startswith(("$E_g", "Conductive", "Semimetal"))]
