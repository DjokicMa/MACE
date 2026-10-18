"""Band gaps are stored from the last SCF of a run, never an earlier cycle.

property_extractor took the last DIRECT and the last INDIRECT gap line
separately, so an SP whose cycle 0 (the guess) printed an INDIRECT gap of
3.4768 eV and that converged to a DIRECT gap of 4.3946 eV was stored with
indirect_band_gap 3.4768 and band_gap_type "indirect". Only the last SCF's
records count now: per spin channel (ALPHA then BETA, or one record for
both), the gap kind and value of that cycle, and a gap of 0 with no kind
when it ended conducting. A spin-polarized run's gap is the fundamental one,
min(CBM) - max(VBM) over both channels from the TOP OF VALENCE / BOTTOM OF
VIRTUAL BANDS lines, which is smaller than both channel gaps when the VBM is
in one channel and the CBM in the other; without those lines it is the
smaller channel gap.
"""
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA


def _extractor():
    from mace.utils.property_extractor import CrystalPropertyExtractor
    return CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)


def _cycle(n, *records):
    out = [f" CYC {n:3d} ETOT(AU) -1.0E+02 DETOT -1.0E-03 tst 0.0E+00 PX 1.0E+00"]
    for chan, rec in records:
        if chan:
            out.append(f"    {chan:<10s} ELECTRONS")
        if rec == "conducting":
            out.append(" POSSIBLY CONDUCTING STATE - EFERMI(AU) -1.0E-01 (RES. CHARGE  1.0E-12;IT.  6)")
        else:
            kind, value = rec
            out += [" INSULATING STATE", f" {kind} ENERGY BAND GAP:   {value:.4f} eV"]
    return "\n".join(out) + "\n"


def _gaps(text):
    keys = ("band_gap", "direct_band_gap", "indirect_band_gap", "band_gap_type", "alpha_band_gap",
            "beta_band_gap", "spin_polarized")
    p = _extractor()._band_gap_properties(text)
    return {k: p[k] for k in keys if k in p}


def test_cycle_zero_indirect_then_converged_direct():
    text = (_cycle(0, (None, ("INDIRECT", 3.4768))) + _cycle(1, (None, ("DIRECT", 4.2))) +
            _cycle(2, (None, ("DIRECT", 4.3946))) + " == SCF ENDED - CONVERGENCE ON ENERGY\n")
    assert _gaps(text) == {"band_gap": 4.3946, "direct_band_gap": 4.3946, "band_gap_type": "direct",
                           "spin_polarized": False}


def test_spin_channels_of_the_last_cycle():
    text = (_cycle(0, ("ALPHA", ("DIRECT", 0.5)), ("BETA", ("DIRECT", 0.5))) +
            _cycle(1, ("ALPHA", ("INDIRECT", 1.2)), ("BETA", ("DIRECT", 2.0))))
    g = _gaps(text)
    assert g["band_gap"] == 1.2 and g["indirect_band_gap"] == 1.2 and g["band_gap_type"] == "indirect"
    assert "direct_band_gap" not in g


def test_spin_summary_is_the_last_one():
    text = (_cycle(0, ("ALPHA", ("DIRECT", 0.5)), ("BETA", ("DIRECT", 0.6))) +
            "   ALPHA BAND GAP:                0.5000  eV \n   BETA BAND GAP:                 0.6000  eV \n" +
            _cycle(1, ("ALPHA", ("DIRECT", 0.8)), ("BETA", ("DIRECT", 0.9))) +
            "   ALPHA BAND GAP:                0.8000  eV \n   BETA BAND GAP:                 0.9000  eV \n")
    g = _gaps(text)
    assert (g["alpha_band_gap"], g["beta_band_gap"], g["direct_band_gap"]) == (0.8, 0.9, 0.8)


def test_conducting_at_the_end_has_no_gap_of_an_earlier_cycle():
    text = (_cycle(0, ("ALPHA", ("INDIRECT", 0.16)), ("BETA", ("INDIRECT", 0.16))) +
            _cycle(1, (None, "conducting")))
    g = _gaps(text)
    assert g["band_gap"] == 0.0
    assert not {"direct_band_gap", "indirect_band_gap", "band_gap_type"} & set(g)


def test_conducting_throughout_stores_gap_zero():
    # 0 as for a run that ended conducting after gapped cycles (and a
    # multi-point OPT ending conducting); no gap kind
    g = _gaps(_cycle(0, (None, "conducting")) + _cycle(1, (None, "conducting")))
    assert g["band_gap"] == 0.0
    assert not {"direct_band_gap", "indirect_band_gap", "band_gap_type"} & set(g)


def _spin_block(chan, kind, gap, tops, bottoms):
    """A channel's SCF block with its band edges, (eig_au, k) each."""
    out = [f"    {chan:<10s} ELECTRONS", " INSULATING STATE"]
    out += [f" TOP OF VALENCE BANDS -    BAND     44; K {k:4d}; EIG {e:.8E} AU" for e, k in tops]
    out.append(f" BOTTOM OF VIRTUAL BANDS - BAND     45; K {bottoms[0][1]:4d}; EIG {bottoms[0][0]:.8E} AU")
    out.append(f" {kind} ENERGY BAND GAP:   {gap:.4f} eV")
    # CRYSTAL can print a BOTTOM line after the gap line (the direct gap's)
    out += [f" BOTTOM OF VIRTUAL BANDS - BAND     45; K {k:4d}; EIG {e:.8E} AU" for e, k in bottoms[1:]]
    return "\n".join(out) + "\n"


def test_spin_gap_is_cbm_minus_vbm_across_channels():
    # VBM in ALPHA (-0.30 at k 1), CBM in BETA (-0.25 at k 4): 0.05 Ha = 1.3606 eV,
    # smaller than both channel gaps (ALPHA 2.7211, BETA 2.1769)
    text = (" CYC   9 ETOT(AU) -1.0E+02\n" +
            _spin_block("ALPHA", "INDIRECT", 2.7211, [(-0.30, 1)], [(-0.20, 3), (-0.19, 1)]) +
            _spin_block("BETA", "DIRECT", 2.1769, [(-0.33, 4), (-0.331, 1)], [(-0.25, 4)]))
    g = _gaps(text)
    assert g == {"band_gap": 1.3606, "indirect_band_gap": 1.3606, "band_gap_type": "indirect",
                 "spin_polarized": True}



def test_spin_summary_run_reports_the_fundamental_gap_everywhere():
    # With CRYSTAL's ALPHA/BETA BAND GAP summary, band_gap is not stored, but the
    # gap other properties use must be the same fundamental gap that
    # direct/indirect_band_gap report, not the smaller channel (an upper bound)
    text = (" CYC   9 ETOT(AU) -1.0E+02\n" +
            _spin_block("ALPHA", "INDIRECT", 2.7211, [(-0.30, 1)], [(-0.20, 3)]) +
            _spin_block("BETA", "DIRECT", 2.1769, [(-0.33, 4)], [(-0.25, 4)]) +
            " ALPHA BAND GAP:   2.7211 eV\n BETA BAND GAP:   2.1769 eV\n")
    x = _extractor()
    g = _gaps(text)
    assert (g["alpha_band_gap"], g["beta_band_gap"], g["spin_polarized"]) == (2.7211, 2.1769, True)
    assert g["indirect_band_gap"] == 1.3606
    assert x._final_gap_value(text) == 1.3606


def test_closed_shell_run_is_not_spin_polarized():
    g = _gaps(_cycle(9, (None, ("DIRECT", 3.2))))
    assert (g["band_gap"], g["spin_polarized"]) == (3.2, False)

def test_spin_gap_across_channels_at_one_k_point_is_direct():
    text = (_spin_block("ALPHA", "DIRECT", 2.7211, [(-0.30, 2)], [(-0.20, 2)]) +
            _spin_block("BETA", "DIRECT", 2.1769, [(-0.33, 2)], [(-0.25, 2)]))
    g = _gaps(text)
    assert (g["band_gap"], g["band_gap_type"], g["direct_band_gap"]) == (1.3606, "direct", 1.3606)


def test_spin_gap_in_one_channel_is_its_printed_gap():
    text = (_spin_block("ALPHA", "INDIRECT", 1.1974, [(-0.35411914, 1)], [(-0.31011466, 9), (-0.31011212, 1)]) +
            _spin_block("BETA", "INDIRECT", 8.7402, [(-0.35983717, 9), (-0.35983974, 1)], [(-0.038639115, 3)]))
    assert _gaps(text)["band_gap"] == 1.1974


def test_spin_edges_that_overlap_are_conducting():
    # ALPHA's VBM (-0.24) above BETA's CBM (-0.25): a half metal, no gap
    text = (_spin_block("ALPHA", "DIRECT", 2.7211, [(-0.24, 1)], [(-0.14, 1)]) +
            _spin_block("BETA", "DIRECT", 2.1769, [(-0.33, 1)], [(-0.25, 1)]))
    g = _gaps(text)
    assert g["band_gap"] == 0.0
    assert not {"direct_band_gap", "indirect_band_gap", "band_gap_type"} & set(g)


def test_spin_gap_without_band_edges_is_the_smaller_channel():
    # no TOP/BOTTOM lines: the smaller channel gap, as before
    text = _cycle(1, ("ALPHA", ("INDIRECT", 1.2)), ("BETA", ("DIRECT", 2.0)))
    assert _gaps(text)["band_gap"] == 1.2


SP = "SP/3.4^9T2_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1_sp_B3LYP-D3-D3_optimized"
CASES = {
    # cycle 0 INDIRECT 0.1600 (spin channels), converged DIRECT 0.4205
    SP + ".out": {"band_gap": 0.4205, "direct_band_gap": 0.4205, "band_gap_type": "direct"},
    # open shell, one extra beta electron: ALPHA 3.3875, BETA 8.7278 eV; the beta HOMO
    # (-0.35230394 Ha) is above the alpha HOMO, so the gap is alpha LUMO - beta HOMO
    "SP/FSI_MOLECULE_relaxed_optimized_sp.out": {"band_gap": 3.0356, "direct_band_gap": 3.0356,
                                                  "band_gap_type": "direct"},
    # open shell, VBM and CBM both in ALPHA (indirect 1.1974; BETA 8.7402): the alpha gap
    "SP/4LG_FSI_2x2_AA_FSI_opt_sp.out": {"band_gap": 1.1974, "indirect_band_gap": 1.1974,
                                         "band_gap_type": "indirect"},
    # spin run with identical channels, and closed shell: as CRYSTAL prints them
    "SP/Ag2Se_mp-568936_sg19_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_"
    "sp_B3LYP-D3-D3_optimized.out": {"band_gap": 2.228, "direct_band_gap": 2.228, "band_gap_type": "direct"},
    # multi-point OPTs: band_gap from CRYSTAL's closing BAND GAP summary; the
    # last SCF's gap kind (6.003 / 6.8399 eV indirect) is point 1's, not stored
    "OPT/1_dia_opt_BULK_OPTGEOM.out": {"band_gap": 6.001},
    "OPT/3_dia3_opt_BULK_OPTGEOM.out": {"band_gap": 6.8348},
    # gapped early cycles, conducting at the end
    "SP/Ti19O30_mp-760075_sg1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_"
    "sp_B3LYP-D3-D3_optimized.out": {"band_gap": 0.0},
}


@pytest.mark.parametrize("rel", sorted(CASES))
def test_corpus_outputs(rel):
    path = TEST_DATA / rel
    if not path.exists():
        pytest.skip("needs the test/ corpus")
    g = _gaps(path.read_text(errors="replace"))
    want = CASES[rel]
    assert {k: g.get(k) for k in want} == want
    for k in {"direct_band_gap", "indirect_band_gap", "band_gap_type"} - set(want):
        assert k not in g, (k, g)


def test_extraction_through_the_cli_stores_the_converged_gap(tmp_path):
    src = TEST_DATA / (SP + ".out")
    if not src.exists():
        pytest.skip("needs the test/ corpus")
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copyfile(src, runs / src.name)
    shutil.copyfile(src.with_name(src.name[:-4] + ".d12"), runs / (src.name[:-4] + ".d12"))
    db = tmp_path / "materials.db"
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        str(runs), "--db-path", str(db)], capture_output=True, text=True, env=env, cwd=tmp_path,
                       timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    c = sqlite3.connect(db)
    rows = dict(c.execute("SELECT property_name, COALESCE(property_value_text, property_value) FROM properties "
                          "WHERE property_name IN ('band_gap', 'direct_band_gap', 'indirect_band_gap', "
                          "'band_gap_type')").fetchall())
    c.close()
    assert rows["band_gap_type"] == "direct"
    assert float(rows["direct_band_gap"]) == 0.4205 and float(rows["band_gap"]) == 0.4205
    assert "indirect_band_gap" not in rows
