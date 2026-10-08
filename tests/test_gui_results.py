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
