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
from mace.gui import results

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
    # (matplotlib names every custom dash pattern "--", the dotted Fermi line's too)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--" and len(ln.get_xdata())
              and not ln.get_label().startswith("$E_F")]
    fermi = [ln for ln in ax.get_lines() if ln.get_label().startswith("$E_F")]
    _cli_figure.fermi_dashes = [ln._unscaled_dash_pattern for ln in fermi]
    colors = {ln.get_color() for ln in ax.get_lines() if len(ln.get_xdata()) > 2}      # the bands (not k ticks)
    _cli_figure.colors = colors
    return legend, dashed


def test_cli_draws_a_non_magnetic_run_once(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path, noise=1e-4)
    legend, dashed = _cli_figure(monkeypatch, path)
    # one set, in the plot's ink, with no spin legend (Marcus 2026-10-16: not spin up's pink)
    assert not any("spin" in t or "α" in t for t in legend)
    assert not dashed and _cli_figure.colors == {"#000000"}


def test_cli_keeps_both_spins_of_a_magnetic_run(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path, beta_shift=0.002)
    legend, dashed = _cli_figure(monkeypatch, path)
    assert legend[:2] == ["spin up (α)", "spin down (β)"] and dashed
    # the Fermi line is dotted, as in the DOS and the GUI (dashed is spin down)
    from mace.plotting.presets import MPL_DASH
    assert _cli_figure.fermi_dashes == [MPL_DASH["dot"]]


def test_cli_closed_shell_is_one_set(monkeypatch, tmp_path):
    path, _, _ = write_band_crystal(tmp_path)
    legend, dashed = _cli_figure(monkeypatch, path)
    assert not any("spin" in t or "bands" in t for t in legend)
    assert not dashed and _cli_figure.colors == {"#000000"}


BAND_4LG = REPO_ROOT / "test" / "BAND" / "4LG_FSI_2x2_AA_opt_HSESOL3C_optimized_band.BAND.DAT"


@pytest.mark.skipif(not BAND_4LG.is_file(), reason="test/ corpus not present")
def test_a_wide_cli_window_keeps_every_band_in_it():
    """1.1.6 review F4-4: mace plotting's band data (a requested window) keep every band that enters
    it, no 400 cap (the 4LG cell lost 5 bands in -20..10 eV); the GUI's capped data keep the bands
    that come nearest E_F, so every band crossing the default +/-6 eV window is still there."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipBANDS_V2 as ipb
    _E, B, _Eb, _Bb, _t, maxV, _k, _n = ipb.read_band_data(str(BAND_4LG))
    ef = float(maxV)

    def inside(lo, hi):
        return {tuple(np.round(B[:, j], 6)) for j in range(B.shape[1]) if B[:, j].max() > ef + lo and B[:, j].min() < ef + hi}

    def sent(b):
        n = b["n_kpoints"]
        cols = np.array([[v for v in col if v is not None][:n] for col in b["up"]])
        return {tuple(np.round(c, 6)) for c in cols}

    cli = results.band_data(BAND_4LG, window=(-20.0, 10.0))
    assert not cli["truncated"] and inside(-20, 10) <= sent(cli)
    assert len(inside(-20, 10)) > 250                   # a large cell: the old cap bit here
    gui = results.band_data(BAND_4LG)
    assert gui["truncated"] and len(gui["up"]) == results.MAX_BANDS
    assert inside(-6, 6) <= sent(gui)
