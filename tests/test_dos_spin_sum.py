"""An unrestricted run with alpha = beta: the one-sided DOS is the total, alpha + |beta| (1.1.6 review F4-1).

MACE's SPIN default makes nearly every run unrestricted. Each block of an unrestricted DOSS.DAT holds
one spin, so a one-sided figure that drew the first block only showed half of the DOS and called it
"Total" (Fe: each block alone integrates to its own spin's count; the two together give 8 e). A real
restricted run writes one block (NSPIN 1, CRYSTAL's IUHF) and is drawn as it is.
"""
import shutil

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.gui import results
from mace.plotting import plotly_bands as PB

HA = 27.2114
trapz = getattr(np, "trapezoid", None) or np.trapz       # numpy 2 renamed it
CORPUS = REPO_ROOT / "test"
DIA = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss"
FE = REPO_ROOT / "tests" / "data" / "spin_fe"
need_corpus = pytest.mark.skipif(not (CORPUS / "DOSS" / f"{DIA}.DOSS.DAT").is_file(), reason="test/ corpus not present")


def _blocks(path):
    """The raw alpha and beta blocks (states/eV), last column = the total projection."""
    rows = [list(map(float, ln.split())) for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith(("#", "@", "&"))]
    a = np.array(rows)
    half = len(a) // 2
    return a[:half, -1] / HA, a[half:, -1] / HA


def _figure(dos, look=None):
    opts = PB.opts_for()
    opts.update(look or {})
    tk = PB.tokens("default")
    env = {"css": lambda n: tk.get(n, ""), "look": PB.look_of(opts, {}), "tone": "light", "paper": {},
           "element_colors": {}, "width": 520, "marginR": 14, "fermiKey": True}
    return PB.band_figure(None, dos, opts, env)


@need_corpus
def test_alpha_equals_beta_dos_is_the_spin_sum(tmp_path):
    for ext in ("DOSS.DAT", "d3", "out"):
        shutil.copy(CORPUS / "DOSS" / f"{DIA}.{ext}", tmp_path)
    src = tmp_path / f"{DIA}.DOSS.DAT"
    alpha, beta = _blocks(src)
    assert np.allclose(alpha, -beta)                         # unrestricted, alpha = beta
    d = results.dos_data(src)
    assert d["spin_polarized"] is False and d["spin_summed"] is True
    total = np.array(next(p["up"] for p in d["projections"] if p["label"] == "Total"))
    want = alpha + np.abs(beta)
    nz = slice(2, -2)                                        # the edge bins (CRYSTAL's zero end, the pile-up)
    assert np.allclose(total[nz], want[nz], rtol=1e-6, atol=1e-9)

    # plotly (and bandfig.js, by the parity test): one-sided = the total, mirrored = one spin a side
    fig = _figure(d)
    tr = next(t for t in fig["traces"] if t.get("meta") == "dos" and t["name"] == "Total")
    assert np.allclose(np.array(tr["x"])[nz], want[nz], rtol=1e-6, atol=1e-9)
    fig = _figure(d, {"dos_spin": "mirrored"})
    up = next(t for t in fig["traces"] if t.get("meta") == "dos" and t["name"] == "Total ↑")
    down = next(t for t in fig["traces"] if t.get("meta") == "dos" and t["name"] == "Total ↓")
    assert np.allclose(np.array(up["x"])[nz], alpha[nz], rtol=1e-6, atol=1e-9)
    assert np.allclose(np.array(down["x"])[nz], beta[nz], rtol=1e-6, atol=1e-9)
    rng = fig["layout"]["xaxis"]["range"]
    one = _figure(d)["layout"]["xaxis"]["range"]
    assert one[1] == pytest.approx(2 * rng[1], rel=1e-9)     # the one-sided range holds both spins


def test_magnetic_fe_keeps_both_spins_and_counts_eight_electrons():
    d = results.dos_data(FE / "fe_doss.DOSS.DAT")
    assert d["spin_polarized"] is True and d["spin_summed"] is False
    e = np.array(d["energy"])
    tot = next(p for p in d["projections"] if p["label"] == "Total")
    occ = e <= 0
    n = trapz(np.array(tot["up"])[occ], e[occ]) + trapz(np.abs(np.array(tot["down"]))[occ], e[occ])
    assert n == pytest.approx(8.0, abs=0.1)


def test_restricted_single_block_is_drawn_as_written(tmp_path):
    """NSPIN 1 (a restricted run): one block, nothing summed."""
    e = np.linspace(-0.2, 0.2, 40)
    lines = ["# NEPTS    40 NPROJ     1 NSPIN     1", "#", '@ XAXIS LABEL "E-EFERMI (HARTREE)"']
    lines += [f"{x: .4E} {5 + np.cos(5 * x): .4E}" for x in e]
    p = tmp_path / "x_doss.DOSS.DAT"
    p.write_text("\n".join(lines) + "\n# EFERMI (HARTREE)   -0.1\n")
    d = results.dos_data(p)
    assert d["spin_summed"] is False and d["spin_polarized"] is False
    assert np.allclose(np.array(d["projections"][0]["up"])[2:-2], ((5 + np.cos(5 * e)) / HA)[2:-2], rtol=1e-3)
