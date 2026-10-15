"""Every matplotlib figure `mace plotting` saves keeps its legend and boxed notes
off the data, the tick labels, the axis labels, each other and the figure edge
(tests/legend_geometry.py: the one rule the GUI's plots are held to as well).

Kinds: band (non-magnetic, magnetic, closed shell), DOS (spin, magnetic, ten
projections), IR, and Raman total / parallel-perpendicular / all six
single-crystal directions, from the repository's fixtures (no corpus). A new
matplotlib plotter is added to KINDS.
"""
import gzip
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from legend_geometry import mpl_problems
from test_dos_spin_window import write_doss_spin
from test_gui_results import write_band_crystal
from test_plot_legend_outside import _band_figure, _dos_figure, write_doss_many

sys.path.insert(0, str(REPO_ROOT / "Plotting"))
FIX = REPO_ROOT / "tests" / "fixtures" / "gui_parity"


def _spectra(monkeypatch, tmp_path, render, name, **kw):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from mace.plotting.handlers import spectra_api as sa
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    src = tmp_path / name
    src.write_bytes(gzip.decompress((FIX / f"{name}.gz").read_bytes()))
    data = sa.read_spectrum(str(src))
    if name.endswith("RAMSPEC.DAT") and data.shape[1] < 10:
        # a molecule's RAMSPEC has the polycrystalline columns only: the six directions from them
        data = np.hstack([data, *[data[:, 1:2] * (0.2 + 0.1 * k) for k in range(10 - data.shape[1])]])
    getattr(sa, render)(data, str(tmp_path / "out"), **kw)
    assert saved
    return saved[-1]


KINDS = {
    "band identical": lambda mp, tp: _band_figure(mp, write_band_crystal(tp, noise=1e-4)[0]),
    "band magnetic": lambda mp, tp: _band_figure(mp, write_band_crystal(tp, beta_shift=0.002)[0]),
    "band closed": lambda mp, tp: _band_figure(mp, write_band_crystal(tp)[0]),
    "dos spin": lambda mp, tp: (write_doss_spin(tp, material="m"), _dos_figure(mp, tp, "m"))[1],
    "dos magnetic": lambda mp, tp: (write_doss_spin(tp, material="m", magnetic=True), _dos_figure(mp, tp, "m"))[1],
    "dos many": lambda mp, tp: (write_doss_many(tp, material="m"), _dos_figure(mp, tp, "m"))[1],
    "ir": lambda mp, tp: _spectra(mp, tp, "render_ir", "water_freq_IRSPEC.DAT"),
    "raman total": lambda mp, tp: _spectra(mp, tp, "render_raman", "water_freq_RAMSPEC.DAT", mode="total"),
    "raman par_perp": lambda mp, tp: _spectra(mp, tp, "render_raman", "water_freq_RAMSPEC.DAT", mode="par_perp"),
    "raman all": lambda mp, tp: _spectra(mp, tp, "render_raman", "water_freq_RAMSPEC.DAT", mode="all"),
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_cli_figure_legends_stay_off_the_data(monkeypatch, tmp_path, kind):
    fig = KINDS[kind](monkeypatch, tmp_path)
    assert any(ax.get_legend() for ax in fig.axes), "no legend drawn"
    probs = mpl_problems(fig)
    assert not probs, probs
