"""`mace gui` Results: band/DOS arrays for the interactive plots.

Synthetic BAND.DAT / DOSS.DAT files in tmp_path, in CRYSTAL's layout: two
spin blocks, tick specs, an optional POTC vacuum reference.
"""

import numpy as np
import pytest

from test_ipdos_readers import write_doss

HA = 27.2114


def write_band(directory, stem="mgo_band", spin=False, d3=True):
    """Path GAMMA-X|K-GAMMA: two panels, the second starting at x = 1 again."""
    x = np.r_[np.linspace(0, 1, 5), np.linspace(1, 2, 5)]
    alpha = np.c_[-0.2 + 0.05 * x, 0.1 + 0.02 * x]
    beta = alpha + (0.01 if spin else 0.0)
    out = ["# NKPT    10 NBND     2 NSPIN     2", "# NPANEL     2",
           "#      1   G", "#      5   X", "#     10   G"]

    def block(bands):
        lines = ["@ XAXIS TICK SPEC     3"]
        for k, t in enumerate((0.0, 1.0, 2.0)):
            lines += [f"@ XAXIS TICK     {k},     {t:.5f}", f'@ XAXIS TICKLABEL    {k},"G"']
        lines += [f"{xi:.5f} " + " ".join(f"{e:.6f}" for e in row) for xi, row in zip(x, bands)]
        return lines + ["# EFERMI (HARTREE)   -0.08000"]

    out += block(alpha) + block(beta)
    (directory / f"{stem}.BAND.DAT").write_text("\n".join(out) + "\n")
    if d3:
        (directory / f"{stem}.d3").write_text(
            "BAND\nmgo - Band Structure - SeeKPath - GAMMA-X|K-GAMMA\n2 8 10 1 2 1 0\n")
    return directory / f"{stem}.BAND.DAT", alpha, beta


def test_closed_shell_bands_send_one_set_with_k_labels_and_breaks(tmp_path):
    from mace.gui import results
    path, alpha, _ = write_band(tmp_path)
    b = results.band_data(path)
    assert b["spin_polarized"] is False and b["down"] is None
    assert b["labels"] == ["Γ", "X|K", "Γ"] and b["ticks"] == [0.0, 1.0, 2.0]
    assert b["fermi"] == 0 and b["reference"] == "fermi"
    # a null between the two points at x = 1 (the U|K-style jump)
    assert b["x"][5] is None and b["x"][4] == b["x"][6] == pytest.approx(1.0)
    assert len(b["up"]) == 2 and b["up"][0][5] is None
    assert b["up"][0][0] == pytest.approx(alpha[0, 0] * HA)


def test_spin_polarized_bands_send_both_spins(tmp_path):
    from mace.gui import results
    path, _, beta = write_band(tmp_path, spin=True)
    b = results.band_data(path)
    assert b["spin_polarized"] is True
    assert b["down"][1][0] == pytest.approx(beta[0, 1] * HA)


def test_potc_puts_the_fermi_level_on_the_vacuum_scale(tmp_path):
    from mace.gui import results
    path, alpha, _ = write_band(tmp_path)
    (tmp_path / "mgo_POTC.POTC.DAT").write_text("# z V\n0.0 0.1\n1.0 0.1\n")
    (tmp_path / "mgo_POTC.out").write_text(" FERMI ENERGY      -0.2\n")
    b = results.band_data(path)
    shift = -(0.1 + 0.2) * HA
    assert b["reference"] == "vacuum" and b["fermi"] == pytest.approx(shift)
    assert b["up"][0][0] == pytest.approx(alpha[0, 0] * HA + shift)


def test_bands_without_a_d3_still_plot(tmp_path):
    from mace.gui import results
    path, _, _ = write_band(tmp_path, d3=False)
    b = results.band_data(path)
    assert b["labels"] == ["", "", ""] and len(b["ticks"]) == 3


def test_dos_closed_shell_and_spin_and_no_d3(tmp_path):
    from mace.gui import results
    m = write_doss(tmp_path)
    d = results.dos_data(tmp_path / f"{m}_doss.DOSS.DAT")
    assert [p["label"] for p in d["projections"]] == ["Mg (all)", "O (s)", "O (p)", "Total"]
    assert d["spin_polarized"] is False and len(d["energy"]) == 60   # beta = -alpha
    (tmp_path / f"{m}_doss.d3").unlink()
    d = results.dos_data(tmp_path / f"{m}_doss.DOSS.DAT")
    assert [p["label"] for p in d["projections"]][-1] == "Total" and len(d["projections"]) == 4


def test_band_and_dos_routes_pair_the_run(tmp_path):
    from test_gui_examples import _json, _start
    write_band(tmp_path)
    write_doss(tmp_path, material="mgo")
    proc, base, _ = _start(tmp_path)
    try:
        status, b = _json(base, "api/bands?path=mgo_band.BAND.DAT")
        assert status == 200, b
        assert b["partner"] == "mgo_doss.DOSS.DAT"
        status, d = _json(base, "api/dos?path=" + b["partner"])
        assert status == 200 and d["partner"] == "mgo_band.BAND.DAT"
        assert _json(base, "api/bands?path=../x.BAND.DAT")[0] in (403, 404)
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def write_band_crystal(directory, stem="gra_band", beta_shift=0.0, noise=0.0):
    """CRYSTAL's own layout at a path break (Γ-X|K-Γ): the tick's point (X)
    ends the first panel and the next point, one step further along x, is
    already K. No x repeats, so nothing but the tick marks the break.
    ``noise`` (Ha) jitters beta the way CRYSTAL's last printed digit does."""
    step = 0.25
    x = np.round(np.arange(9) * step, 5)            # ticks at 0, 1 (point 4) and 2 (point 8)
    alpha = np.c_[np.r_[np.full(5, -0.30), np.full(4, -0.10)],    # jumps at the break
                  np.r_[np.full(5, 0.20), np.full(4, 0.05)]]
    rng = np.random.default_rng(0)
    beta = alpha + beta_shift + (rng.uniform(-noise, noise, alpha.shape) if noise else 0.0)
    out = ["# NKPT     9 NBND     2 NSPIN     2", "# NPANEL     2",
           "#      1   A", "#      5   A", "#      9   A"]

    def block(bands):
        lines = ["@ XAXIS TICK SPEC     3"]
        for k, t in enumerate((0.0, 1.0, 2.0)):
            lines += [f"@ XAXIS TICK     {k},     {t:.5f}", f'@ XAXIS TICKLABEL    {k},"A"']
        lines += [f"{xi:.5f} " + " ".join(f"{e:.7f}" for e in row) for xi, row in zip(x, bands)]
        return lines + ["# EFERMI (HARTREE)   -0.08000"]

    out += block(alpha) + block(beta)
    (directory / f"{stem}.BAND.DAT").write_text("\n".join(out) + "\n")
    (directory / f"{stem}.d3").write_text(
        "BAND\ngra - Band Structure - SeeKPath - GAMMA-X-|-K-GAMMA\n2 8 9 1 2 1 0\n")
    return directory / f"{stem}.BAND.DAT", alpha, beta


def test_crystal_path_break_without_a_repeated_point_gets_a_gap(tmp_path):
    """CRYSTAL does not repeat the x at "X|K": the gap goes after the tick's point."""
    from mace.gui import results
    path, alpha, _ = write_band_crystal(tmp_path)
    b = results.band_data(path)
    assert b["labels"] == ["Γ", "X|K", "Γ"]
    assert b["x"][5] is None and b["x"][4] == pytest.approx(1.0) and b["x"][6] == pytest.approx(1.25)
    assert [col[5] for col in b["up"]] == [None, None]
    assert b["up"][0][4] == pytest.approx(alpha[4, 0] * HA) and b["up"][0][6] == pytest.approx(alpha[5, 0] * HA)
    assert b["x"].count(None) == 1                    # the continuous ticks stay joined


def test_path_break_indices_both_layouts():
    import sys
    from conftest import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipBANDS_V2 as ipb
    crystal = [0, 0.5, 1.0, 1.5, 2.0, 2.5]           # tick 1.0 is point 2
    assert ipb.path_break_indices(crystal, [0, 1.0, 2.5], ["G", "X|K", "G"]) == [2]
    repeated = [0, 0.5, 1.0, 1.0, 1.5, 2.0]          # the tick's x written twice
    assert ipb.path_break_indices(repeated, [0, 1.0, 2.0], ["G", "X|K", "G"]) == [2]
    assert ipb.path_break_indices(crystal, [0, 1.0, 2.5], ["G", "X", "G"]) == []
    assert ipb.path_break_indices(crystal, [0, 1.0001, 2.5], ["G", "X|K", "G"]) == [2]   # 5-digit ticks


def test_real_crystal_band_break_is_not_joined():
    """A real CRYSTAL BAND.DAT with a "U|K" break: one gap, at the tick, no repeated x."""
    from conftest import find_data
    from mace.gui import results
    path = find_data("BAND/1_dia_opt_rev1_sp_*_band.BAND.DAT")
    b = results.band_data(path)
    k = b["labels"].index("U|K")
    gaps = [i for i, v in enumerate(b["x"]) if v is None]
    assert len(gaps) == 1
    i = gaps[0]
    assert b["x"][i - 1] == pytest.approx(b["ticks"][k], abs=1e-3) and b["x"][i + 1] > b["x"][i - 1]


# seekpath's oA2/oC2 path as MACE writes it for a slipped AB graphite cell: three breaks,
# and two segments (Y-F0, B0|G0-T) 14x shorter than their neighbors
SLIP_PATH = "GAMMA-Y-F_0-|-DELTA_0-GAMMA-Z-B_0-|-G_0-T-Y-|-GAMMA-S-R-Z-T"
SLIP_TICKS = (0.0, 0.53414, 0.57266, 0.98978, 1.38128, 1.79454, 1.83691, 2.2284, 2.58203,
              2.97352, 3.32331, 3.85746)


def write_band_path(directory, stem="slip_band", path=SLIP_PATH, ticks=SLIP_TICKS, step=0.0039):
    """A closed-shell BAND.DAT in CRYSTAL's layout for any path: one point per
    ``step`` along x, each tick's point ending its panel (no repeated x)."""
    ticks = list(ticks)
    xs = [0.0]
    tick_idx = [1]
    for a, b in zip(ticks, ticks[1:]):
        n = max(2, int(round((b - a) / step)))
        xs += list(np.linspace(xs[-1] + step, b, n))
        tick_idx.append(len(xs))
    x = np.round(np.array(xs), 5)
    bands = np.c_[-0.2 + 0.05 * np.sin(5 * x), 0.1 + 0.03 * np.cos(4 * x)]
    out = [f"# NKPT {len(x):5d} NBND     2 NSPIN     2", f"# NPANEL {len(ticks) - 1:5d}"]
    out += [f"# {k:6d}   A" for k in tick_idx]

    def block():
        lines = [f"@ XAXIS TICK SPEC    {len(ticks)}"]
        for k, t in enumerate(ticks):
            lines += [f"@ XAXIS TICK    {k:2d},     {t:.5f}", f'@ XAXIS TICKLABEL   {k:2d},"A"']
        lines += [f"{xi:.5f} " + " ".join(f"{e:.7f}" for e in row) for xi, row in zip(x, bands)]
        return lines + ["# EFERMI (HARTREE)   -0.10500"]

    out += block() + block()
    (directory / f"{stem}.BAND.DAT").write_text("\n".join(out) + "\n")
    (directory / f"{stem}.d3").write_text(f"BAND\nslip - Band Structure - SeeKPath (w.I) - {path}\n"
                                          f"{len(ticks) - 1} 8 1000 1 2 1 0\n")
    return directory / f"{stem}.BAND.DAT"


def test_seekpath_oa2_path_three_breaks_and_subscripts(tmp_path):
    from mace.gui import results
    b = results.band_data(write_band_path(tmp_path))
    assert b["labels"] == ["Γ", "Y", "F<sub>0</sub>|Δ<sub>0</sub>", "Γ", "Z", "B<sub>0</sub>|G<sub>0</sub>",
                           "T", "Y|Γ", "S", "R", "Z", "T"]
    assert b["spin_polarized"] is False and b["spin_identical"] is False
    gaps = [i for i, v in enumerate(b["x"]) if v is None]
    assert len(gaps) == 3
    for i, t in zip(gaps, (0.57266, 1.79454, 2.2284)):
        assert b["x"][i - 1] == pytest.approx(t, abs=1e-4) and b["x"][i + 1] > b["x"][i - 1]
