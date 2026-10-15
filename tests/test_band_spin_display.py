"""Spin-up and spin-down bands: one set when they agree, both when they differ.

A non-magnetic spin-polarized run (graphite with SPIN on) writes beta bands
that differ from alpha only in CRYSTAL's last printed digit (1e-4 Ha, 2.7 meV).
Drawn as a dotted beta set over the solid alpha set, the dotted copies of
degenerate bands run out of phase and read as one solid beta line, hiding
alpha. Both plotters now draw such a run once and say so; a magnetic run keeps
both sets. The GUI (mace/gui/results.py) and the CLI (Plotting/ipBANDS_V2.py)
share ipBANDS_V2.spin_state.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT, find_data
from test_gui_results import HA, write_band, write_band_crystal

sys.path.insert(0, str(REPO_ROOT / "Plotting"))


def test_spin_state_closed_identical_polarized():
    import ipBANDS_V2 as ipb
    a = np.array([[-1.0, 2.0], [-1.1, 2.1]])
    assert ipb.spin_state(a, np.zeros_like(a)) == ("closed", 0.0)
    assert ipb.spin_state(a, a.copy()) == ("closed", 0.0)
    state, diff = ipb.spin_state(a, a + 0.0027)
    assert state == "identical" and diff == pytest.approx(0.0027)
    assert ipb.spin_state(a, a + 0.03)[0] == "polarized"


def test_gui_sends_a_non_magnetic_run_once(tmp_path):
    from mace.gui import results
    path, alpha, _ = write_band_crystal(tmp_path, noise=1e-4)       # CRYSTAL's last digit
    b = results.band_data(path)
    assert b["spin_polarized"] is False and b["spin_identical"] is True and b["down"] is None
    assert 0 < b["spin_max_diff"] <= 0.01
    assert b["up"][0][0] == pytest.approx(alpha[0, 0] * HA)


def test_gui_keeps_both_spins_of_a_magnetic_run_and_closed_shell_is_plain(tmp_path):
    from mace.gui import results
    path, _, _ = write_band_crystal(tmp_path, beta_shift=0.002)     # 54 meV
    b = results.band_data(path)
    assert b["spin_polarized"] is True and b["spin_identical"] is False and b["down"]
    (tmp_path / "closed").mkdir()
    path, _, _ = write_band(tmp_path / "closed", stem="c_band")
    b = results.band_data(path)
    assert b["spin_polarized"] is False and b["spin_identical"] is False


def test_real_runs_magnetic_and_closed_shell():
    """4LG AA (2x2) is magnetic (bands split by up to 0.13 eV); diamond is closed-shell."""
    from mace.gui import results
    mag = results.band_data(find_data("BAND/4LG_2x2_AA_opt_*_band.BAND.DAT"))
    assert mag["spin_polarized"] is True and mag["down"] and mag["spin_max_diff"] > 0.05
    closed = results.band_data(find_data("BAND/1_dia_opt_rev1_sp_*_band.BAND.DAT"))
    assert closed["spin_polarized"] is False and closed["spin_identical"] is False


def _cli_figure(monkeypatch, path):
    """Run ipBANDS on one BAND.DAT and hand back the figure it saved."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import ipBANDS_V2 as ipb
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(ipb.plt, "close", lambda *a, **k: None)
    args = SimpleNamespace(list_segments=False, segments=None, path=None, no_gaps=False, gap_width=0.02,
                           E_lower=-10, E_upper=10, fixed_width=None, fixed_height=None, alpha=1.0,
                           spin_up_color=ipb.DEFAULT_SPIN_UP_COLOR, spin_down_color=ipb.DEFAULT_SPIN_DOWN_COLOR,
                           auto_width=False, formats="png")
    d3 = path.with_name(path.name.replace(".BAND.DAT", ".d3"))
    ipb.ipBANDS(str(path), "m", {"d3": str(d3)}, args, str(path.parent))
    assert saved
    ax = saved[-1].axes[0]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--" and len(ln.get_xdata())]
    return legend, dashed


def test_cli_draws_a_non_magnetic_run_once(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path, noise=1e-4)
    legend, dashed = _cli_figure(monkeypatch, path)
    assert legend[0] == "Spin ↑ = ↓ (non-magnetic)" and "Spin ↓" not in legend
    assert not dashed


def test_cli_keeps_both_spins_of_a_magnetic_run(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path, beta_shift=0.002)
    legend, dashed = _cli_figure(monkeypatch, path)
    assert legend[:2] == ["Spin ↑", "Spin ↓"] and dashed


def test_cli_closed_shell_is_one_set(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path)
    legend, dashed = _cli_figure(monkeypatch, path)
    assert legend[0] == "Bands" and not dashed
