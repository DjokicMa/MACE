"""The electronic ZT of a TRANSPORT run uses kappa_e, not KAPPA.dat's kappa0.

KAPPA.dat holds CRYSTAL's zero-field K coefficient kappa0 (header "K
coefficient"; CRYSTAL23 manual Eq. 14.3); the electronic thermal
conductivity is kappa_e = kappa0 - T*sigma*S^2. ZT_e was computed as
PF*T/kappa0 = x, which is below 1 always; it is PF*T/kappa_e = x/(1-x).
"""
import re
import shutil

import pytest

from conftest import find_data
from mace.utils.property_extractor import CrystalPropertyExtractor

BCU = "TRANSPORT/C1-RCSR-bcu-f_optimized_TRANSPORT"


def _props(out):
    return CrystalPropertyExtractor(enable_tracking=False).extract_all_properties(out)


def test_zt_peak_by_hand():
    # the peak (300 K, mu -2.610 eV), from the three files' rows by hand,
    # diagonal averages (xx + yy + zz)/3:
    S = (2 * -3.2416492E-04 - 3.2727810E-04) / 3            # SEEBECK.dat, V/K
    sigma = (2 * 5.6830186E+03 + 1.4938241E+03) / 3         # SIGMA.dat, 1/(Ohm m)
    kappa0 = (2 * 1.9225154E-01 + 5.1793779E-02) / 3        # KAPPA.dat, W/m/K
    x = S * S * sigma * 300.0 / kappa0                      # 0.93515: the old ZT
    p = _props(find_data(BCU + ".out"))
    t = p["transport"]
    assert t["zt_electronic_max_at"] == {"temperature_k": 300.0, "mu_ev": -2.61}
    assert p["transport_zt_max"] == pytest.approx(x / (1 - x), rel=1e-9) == pytest.approx(14.4214, abs=1e-4)
    assert p["transport_kappa0_at_zt_max"] == pytest.approx(kappa0, rel=1e-12)
    assert p["transport_kappa_electronic_at_zt_max"] == pytest.approx(kappa0 - S * S * sigma * 300.0, rel=1e-9)
    assert t["kappa_electronic_at_zt_max_w_per_m_k"] == p["transport_kappa_electronic_at_zt_max"]
    assert CrystalPropertyExtractor._get_property_unit(None, "transport_kappa_electronic_at_zt_max") == "W/(m·K)"


def test_points_with_nonpositive_kappa_e_store_nothing(tmp_path, capsys):
    # KAPPA.dat values set to 1e-300: T*sigma*S^2 >= kappa0 at every point
    # with a Seebeck coefficient
    out = find_data(BCU + ".out")
    for f in out.parent.glob(out.stem + ".*"):
        shutil.copy(f, tmp_path / f.name)
    kap = tmp_path / (out.stem + ".KAPPA.dat")
    rows = []
    for line in kap.read_text().splitlines():
        if line.startswith("#"):
            rows.append(line)
        else:
            v = line.split()
            rows.append("  ".join(v[:3] + ["1.0E-300"] * len(v[3:])))
    kap.write_text("\n".join(rows) + "\n")
    p = _props(tmp_path / out.name)
    assert "transport_zt_max" not in p and "transport_kappa_electronic_at_zt_max" not in p
    assert p["transport"]["zt_electronic_max"] is None
    assert p["transport"]["zt_points_skipped_kappa_e_nonpositive"] > 0
    assert re.search(r"kappa_e <= 0\) left out of ZT", capsys.readouterr().out)
    assert p["transport_power_factor_max"] > 0      # PF does not need kappa


def test_gui_labels_kappa_dat_as_kappa0():
    from mace.gui.results import TRANSPORT_QUANTITIES
    assert "κ₀" in TRANSPORT_QUANTITIES["KAPPA"][0]


def test_zt_is_named_electrons_only():
    p = _props(find_data(BCU + ".out"))
    assert p["transport_zt_electronic_max"] == p["transport_zt_max"]
    assert "no lattice kappa" in p["transport"]["units"]["zt"]
    assert CrystalPropertyExtractor._get_property_unit(None, "transport_power_factor_max") == "W/(m·K²)"
