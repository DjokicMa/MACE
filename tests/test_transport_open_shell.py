"""Open-shell (spin-polarized) BOLTZTRA files: alpha and beta are summed.

CRYSTAL23 manual, BOLTZTRA (p. 315): for an open-shell calculation SIGMA.DAT,
SIGMAS.DAT and KAPPA.DAT hold an alpha block and then a beta block, each with
the header "# 1 [Alpha ...]" / "# 1 [Beta ...]" (0 for closed shell, 1 for
open shell), while "the Seebeck coefficient, SEEBECK.DAT file, is not spin
dependent". The two spin channels conduct in parallel, so the totals are
sigma = sigma_a + sigma_b and kappa0 = kappa0_a + kappa0_b (Eqs. 14.1 and
14.3 are sums over bands). The rows used to be keyed on (T, mu) alone, so the
beta block overwrote the alpha one: sigma and kappa0 were the beta channel's,
ZT_e survived by luck (the ratio cancels) and ZT_total did not (kappa_L is
not halved).

There is no open-shell transport run in the corpus (all four are "# 0"), so
the documented layout is built from the real afi run, split between the spin
channels: alpha = beta = half, and an unequal 0.7 / 0.3 split. Either way the
numbers must be the closed-shell file's.
"""
import re
import shutil
from pathlib import Path

import pytest

from conftest import TEST_DATA, find_data
from mace.utils import transport_zt as Z
from mace.utils.property_extractor import CrystalPropertyExtractor

AFI = "TRANSPORT/C1-RCSR-afi_optimized_TRANSPORT"
ADDITIVE = ("SIGMA", "SIGMAS", "KAPPA")


def _scaled(lines, f):
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out.append(ln)
            continue
        v = s.split()
        out.append("  ".join(v[:2] + [f"{float(x) * f:.10E}" for x in v[2:]]))
    return out


def _open_shell(src_stem, dst_dir, split=(0.5, 0.5), seebeck_split=None):
    """The documented open-shell layout of a closed-shell run: each additive
    file as an alpha block then a beta block; SEEBECK as one block, or (when
    seebeck_split is given) per spin with S_a, S_b whose sigma-weighted mean is S."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = Path(src_stem).name
    shutil.copy(f"{src_stem}.out", dst_dir / f"{name}.out")
    for q in ADDITIVE + ("SEEBECK",):
        lines = Path(f"{src_stem}.{q}.dat").read_text().splitlines()
        head = lines[0]
        assert head.startswith("# 0 [")
        body = lines[1:]
        if q == "SEEBECK" and seebeck_split is None:
            text = [head.replace("# 0 [", "# 1 [", 1)] + body
        elif q == "SEEBECK":
            fa, fb = seebeck_split            # S_a = fa*S, S_b = fb*S, with split[0]*fa + split[1]*fb = 1
            text = ([head.replace("# 0 [", "# 1 [Alpha ", 1)] + _scaled_s(body, fa)
                    + [head.replace("# 0 [", "# 1 [Beta ", 1)] + _scaled_s(body, fb))
        else:
            a, b = split
            text = ([head.replace("# 0 [", "# 1 [Alpha ", 1)] + _scaled(body, a)
                    + [head.replace("# 0 [", "# 1 [Beta ", 1)] + _scaled(body, b))
        (dst_dir / f"{name}.{q}.dat").write_text("\n".join(text) + "\n")
    return dst_dir / f"{name}.out"


def _scaled_s(lines, f):
    """Scale S only (the N column is the run's)."""
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out.append(ln)
            continue
        v = s.split()
        out.append("  ".join(v[:3] + [f"{float(x) * f:.10E}" for x in v[3:]]))
    return out


def _props(out, kappa_l=None):
    x = CrystalPropertyExtractor(enable_tracking=False)
    x.kappa_lattice = Z.parse_kappa_lattice(kappa_l) if kappa_l is not None else None
    x.tau_fs = None
    return x.extract_all_properties(out)


KEYS = ("transport_zt_total_max_avg", "transport_zt_electronic_max_avg", "transport_power_factor_max_avg",
        "transport_kappa0_at_zt_max", "transport_kappa_electronic_at_zt_max", "transport_seebeck_max_uv_per_k",
        "transport_zt_total_kappa_electronic_at_max")


@pytest.mark.parametrize("split", [(0.5, 0.5), (0.7, 0.3)], ids=["half", "0.7-0.3"])
def test_open_shell_sums_alpha_and_beta(tmp_path, split):
    closed = find_data(AFI + ".out")
    out = _open_shell(str(closed)[:-4], tmp_path / "os", split)
    cs, os_ = _props(closed, "1.5"), _props(out, "1.5")
    for k in KEYS:
        assert os_[k] == pytest.approx(cs[k], rel=1e-8), k
    for k in ("transport_zt_total_max_temperature_k", "transport_zt_total_max_mu_ev"):
        assert os_[k] == cs[k], k
    assert os_["transport"]["spin_channels"] == "alpha+beta"
    assert cs["transport"]["spin_channels"] == "closed shell"


def test_per_spin_seebeck_is_sigma_weighted(tmp_path):
    """Not the documented layout (SEEBECK is not spin dependent), but if a file
    has per-spin S the total is (sigma_a S_a + sigma_b S_b) / (sigma_a + sigma_b)."""
    closed = find_data(AFI + ".out")
    # sigma split 0.7/0.3; S_a = 1.2 S, S_b = (1 - 0.7*1.2)/0.3 S, so the weighted mean is S
    out = _open_shell(str(closed)[:-4], tmp_path / "os", (0.7, 0.3), (1.2, (1 - 0.84) / 0.3))
    cs, os_ = _props(closed, "1.5"), _props(out, "1.5")
    for k in KEYS:
        assert os_[k] == pytest.approx(cs[k], rel=1e-7), k


def test_gui_and_cli_agree_on_an_open_shell_run(tmp_path):
    from mace.gui import results
    closed = find_data(AFI + ".out")
    out = _open_shell(str(closed)[:-4], tmp_path / "os", (0.7, 0.3))
    p = _props(out, "1.5")
    for q in ("SEEBECK", "SIGMA", "KAPPA"):
        z = results.transport_zt(out.with_name(out.stem + f".{q}.dat"), "1.5", None)
        assert z["zt_total_max"]["value"] == p["transport_zt_total_max_avg"]
        assert z["zt_e_max"]["value"] == p["transport_zt_electronic_max_avg"]
    cz = results.transport_zt(Path(str(closed)[:-4] + ".SEEBECK.dat"), "1.5", None)
    assert z["zt_total_max"]["value"] == pytest.approx(cz["zt_total_max"]["value"], rel=1e-8)


def test_gui_plot_of_an_open_shell_sigma_is_the_total(tmp_path):
    from mace.gui import results
    closed = find_data(AFI + ".out")
    out = _open_shell(str(closed)[:-4], tmp_path / "os", (0.7, 0.3))
    for q in ("SIGMA", "KAPPA"):
        a = results.transport_data(Path(str(closed)[:-4] + f".{q}.dat"))
        b = results.transport_data(out.with_name(out.stem + f".{q}.dat"))
        assert [x["T"] for x in a["blocks"]] == [x["T"] for x in b["blocks"]]
        for x, y in zip(a["blocks"], b["blocks"]):
            assert y["mu"] == x["mu"]
            assert y["avg"] == pytest.approx(x["avg"], rel=1e-8)
            for c in ("xx", "yy", "zz"):
                assert y[c] == pytest.approx(x[c], rel=1e-8), c


def test_parser_reads_the_spin_headers(tmp_path):
    closed = find_data(AFI + ".out")
    out = _open_shell(str(closed)[:-4], tmp_path / "os", (0.7, 0.3))
    unit, rows = CrystalPropertyExtractor._parse_transport_dat(out.with_name(out.stem + ".SIGMA.dat"))
    _, cs_rows = CrystalPropertyExtractor._parse_transport_dat(Path(str(closed)[:-4] + ".SIGMA.dat"))
    assert unit == "1/Ohm/m" and len(rows) == len(cs_rows)
    for r, c in zip(rows, cs_rows):
        assert r[:2] == c[:2]
        assert r[3] == pytest.approx(c[3], rel=1e-8)
        assert r[2] == pytest.approx(c[2], rel=1e-8)          # carriers: alpha + beta


def test_lut_refuses_rows_with_the_same_point_twice():
    rows = [(300.0, 0.1, 1.0, 2.0), (300.0, 0.1, 1.0, 3.0)]
    with pytest.raises(ValueError, match="twice"):
        Z._lut(rows)


def test_a_real_closed_shell_file_is_read_as_before():
    if not (TEST_DATA / "TRANSPORT").is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    for f in sorted((TEST_DATA / "TRANSPORT").glob("*.dat")):
        if re.search(r"\.(SEEBECK|SIGMA|SIGMAS|KAPPA)\.dat$", f.name):
            assert f.read_text().startswith("# 0 [")
            _, rows = CrystalPropertyExtractor._parse_transport_dat(f)
            assert rows and len({(round(T, 3), round(mu, 4)) for T, mu, *_ in rows}) == len(rows)
