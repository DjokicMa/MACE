"""DOS spin blocks and the edges of the computed DOS window.

``write_doss_spin`` writes a spin-polarized DOSS.DAT in CRYSTAL's layout whose
beta block is not exactly the negative of alpha (so both spins are sent), and,
with ``pileup``, the second record of each block holds the large value CRYSTAL
puts there when the DOS window cuts through bands (graphite: 3615 states/Ha in
one bin, against about 30 elsewhere).
"""
import numpy as np

HA = 27.2114


def write_doss_spin(directory, material="gra", n_e=80, pileup=True, magnetic=False):
    d3 = ("NEWK\n8 8\n1 0\nDOSS\n2 400 -1 -1 2 14 0\n-0.5 0.5\n"
          "4  1 2 3 4  #C S\n9  5 6 7 8 9 10 11 12 13  #C P\nEND\n")
    (directory / f"{material}_doss.d3").write_text(d3)
    e = np.linspace(-0.26, 0.26, n_e)                # Ha, about -7 to +7 eV
    lines = [f"# NEPTS {n_e:5d} NPROJ     3 NSPIN     2", "#",
             '@ XAXIS LABEL "E-EFERMI (HARTREE)"']
    for block in range(2):
        sign = 1 if block == 0 else -1
        for k, x in enumerate(e):
            s = 10 + 8 * np.sin(9 * x) ** 2
            p = 20 + 15 * np.cos(7 * x) ** 2
            if magnetic and block == 1:
                s, p = 0.6 * s, 1.3 * p
            if k == 0 or k == n_e - 1:
                s = p = 0.0                         # CRYSTAL's zero end points
            elif k == 1 and pileup:
                s, p = 900.0, 2700.0                 # the states below the window, in one bin
            cols = [x, sign * s, sign * p, sign * (s + p) * (1 + (1e-3 if block else 0))]
            lines.append(" ".join(f"{c: .4E}" for c in cols))
        lines += ["# EFERMI (HARTREE)   -0.10828", "&"]
    (directory / f"{material}_doss.DOSS.DAT").write_text("\n".join(lines) + "\n")
    return directory / f"{material}_doss.DOSS.DAT"


def test_slightly_asymmetric_spin_blocks_send_both_spins(tmp_path):
    from mace.gui import results
    d = results.dos_data(write_doss_spin(tmp_path, pileup=False))
    assert d["spin_polarized"] is True
    assert [p["label"] for p in d["projections"]] == ["C (s)", "C (p)", "Total"]
    tot = d["projections"][-1]
    assert max(tot["up"]) > 0 and min(tot["down"]) < 0


def _ipdos():
    import sys
    from conftest import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT / "Plotting"))
    import ipDOS_V2
    return ipDOS_V2


def test_suppress_edge_pileup_zeroes_the_bin_in_both_spin_blocks():
    ip = _ipdos()
    e = list(range(20))
    up = [0, 500] + [10 + (k % 3) for k in range(17)] + [0]
    cols, n = ip.suppress_edge_pileup([e + e, up + up, [-v for v in up] * 2], 2)
    assert n == 1 and cols[1][1] == 0 and cols[1][21] == 0 and cols[2][1] == 0
    assert cols[1][2:19] == up[2:19] and cols[0] == e + e
    # a smooth edge, a top-edge pile-up, a block too short to judge
    assert ip.suppress_edge_pileup([e, [0, 11] + up[2:]], 1)[1] == 0
    top = [0, 11] + up[2:-2] + [400, 0]
    cols, n = ip.suppress_edge_pileup([e, top], 1)
    assert n == 1 and cols[1][-2] == 0
    assert ip.suppress_edge_pileup([[0, 1, 2], [0, 99, 1]], 1) == ([[0, 1, 2], [0, 99, 1]], 0)


def test_gui_dos_leaves_out_the_pileup_and_sends_the_window(tmp_path):
    from mace.gui import results
    d = results.dos_data(write_doss_spin(tmp_path))
    tot = d["projections"][-1]
    assert d["edge_pileup"] == 1 and d["spin_polarized"] is True
    assert max(map(abs, tot["up"])) < 3 and max(map(abs, tot["down"])) < 3     # 132 states/eV before
    assert d["window"] == [d["energy"][0], d["energy"][-1]]
    assert abs(d["window"][0] + 0.26 * HA) < 1e-6
    clean = results.dos_data(write_doss_spin(tmp_path, material="b", pileup=False))
    assert clean["edge_pileup"] == 0


def test_real_dos_runs_pileup_and_spin():
    from conftest import find_data
    from mace.gui import results
    dia = results.dos_data(find_data("DOSS/1_dia_opt_rev1_sp_*_doss.DOSS.DAT"))
    assert dia["edge_pileup"] == 1 and dia["spin_polarized"] is False
    assert max(map(abs, dia["projections"][-1]["up"])) < 5          # 37.7 states/eV in the edge bin before
    mag = results.dos_data(find_data("DOSS/4LG_2x2_AA_opt_*_doss.DOSS.DAT"))
    assert mag["spin_polarized"] is True and mag["edge_pileup"] == 1
    assert max(map(abs, mag["projections"][-1]["down"])) < 100     # 4955 before
    mol = results.dos_data(find_data("DOSS/EC_MOLECULE_OPT_*_doss.DOSS.DAT"))
    assert mol["edge_pileup"] == 0


def test_cli_dos_axis_ignores_the_pileup(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    ip = _ipdos()
    write_doss_spin(tmp_path)
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(ip.plt, "close", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    ip.ipDOS("gra", -10, 10, "orbital", formats="png", output_dir=str(tmp_path))
    lo, hi = saved[-1].axes[0].get_xlim()
    assert hi < 5 and lo > -5            # +-146 with the edge bin
