"""The BAND.DAT HOMO column is offset by the first band written.

A BAND run writes bands band_start..band_end, so BAND.DAT column n is band
band_start + n - 1; the occupied-band count is absolute (from the electron
count). The gap, band edges and direct/indirect flag read column num_occ,
which is the HOMO only for band_start = 1. The test writes the diamond BAND
run (bands 1-36) as one of bands 3-36: the .out's FROM BAND line and the
BAND.DAT without its first two band columns, and expects the same results.
"""
import shutil

import pytest

from conftest import find_data

STEM = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
KEYS = ("band_dat_vbm_ev", "band_dat_cbm_ev", "band_direct_gap_ev",
        "band_fundamental_gap_ev", "band_vbm_kpoint_index",
        "band_cbm_kpoint_index", "band_gap_is_direct")


def _band_props(extractor, out):
    return extractor._extract_band_structure_properties(out.read_text(errors="replace"), out)


def _drop_first_bands(src, dst, drop):
    lines = []
    for line in src.read_text().splitlines():
        if line.startswith("# NKPT"):
            tok = line.split()
            i = tok.index("NBND") + 1
            line = line.replace(f"NBND {int(tok[i]):5d}", f"NBND {int(tok[i]) - drop:5d}")
        elif line.strip() and line[0] not in "#@":
            vals = line.split()
            line = "".join(f" {float(v): .5E}" for v in [vals[0]] + vals[1 + drop:])
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def test_band_start_one_is_unchanged(extractor):
    p = _band_props(extractor, find_data(f"BAND/{STEM}.out"))
    assert "band_start" not in p          # band 1 is not stored (1.1.6)
    assert p["band_fundamental_gap_ev"] == pytest.approx(5.997, abs=0.01)


def test_band_start_three_reads_the_same_homo(extractor, tmp_path):
    out = find_data(f"BAND/{STEM}.out")
    ref = _band_props(extractor, out)
    text = out.read_text(errors="replace").replace("FROM BAND   1 TO BAND  36", "FROM BAND   3 TO BAND  36")
    new_out = tmp_path / out.name
    new_out.write_text(text)
    shutil.copy(out.with_suffix(".d3"), tmp_path / f"{STEM}.d3")
    _drop_first_bands(out.parent / f"{STEM}.BAND.DAT", tmp_path / f"{STEM}.BAND.DAT", 2)
    p = _band_props(extractor, new_out)
    assert p["band_start"] == 3
    for k in KEYS:
        assert p[k] == pytest.approx(ref[k], abs=2e-3), k
    # band numbers stay absolute (HOMO = band 6 of diamond)
    for edge, band in (("vbm", 6), ("cbm", 7)):
        assert p["band_structure"][edge]["band_index"] == ref["band_structure"][edge]["band_index"] == band
    assert p["band_structure"]["occupied_bands"] == ref["band_structure"]["occupied_bands"] == 6
