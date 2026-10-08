"""BAND.DAT rows wrap after 1000 fields, so NBND >= 1000 must be stitched.

CRYSTAL writes at most 1000 fields per data line (the k-path abscissa plus
999 bands) and continues the same k-point on the next line. The band
readers used to stitch only when NBND > 1000, so NBND == 1000 (a row one
value short) read the continuation line as the next k-point, and the two
line-per-k-point readers never stitched at all.
"""
import importlib.util

import numpy as np
import pytest

from conftest import REPO_ROOT

NKPT = 3


def _value(i, j):
    return round(-0.5 + 0.001 * i + 1e-4 * j, 6)


def _write_band_dat(path, nbnd):
    lines = [f"# NKPT {NKPT} NBND {nbnd} NSPIN 1", "# NPANEL 1", "#     0 G", f"#     {NKPT - 1} X"]
    for i in range(NKPT):
        fields = [f"{0.1 * i:.6f}"] + [f"{_value(i, j):.6f}" for j in range(nbnd)]
        for start in range(0, len(fields), 1000):
            lines.append(" ".join(fields[start:start + 1000]))
    path.write_text("\n".join(lines) + "\n")


def _expected(nbnd):
    return np.array([[_value(i, j) for j in range(nbnd)] for i in range(NKPT)])


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("nbnd", [999, 1000, 1001, 1500, 2100])
def test_plotting_reader_stitches_wrapped_rows(tmp_path, nbnd):
    ip = _load("ipBANDS_V2_wrap", "Plotting/ipBANDS_V2.py")
    f = tmp_path / "x_band.BAND.DAT"
    _write_band_dat(f, nbnd)
    E, BANDS = ip.read_band_data(str(f))[:2]
    assert len(E) == NKPT
    np.testing.assert_allclose(BANDS, _expected(nbnd) * 27.2114, atol=1e-6)


@pytest.mark.parametrize("nbnd", [999, 1000, 1500, 2100])
def test_dat_file_processor_stitches_wrapped_rows(tmp_path, nbnd):
    from mace.utils.dat_file_processor import DatFileProcessor
    f = tmp_path / "x_band.BAND.DAT"
    _write_band_dat(f, nbnd)
    res = DatFileProcessor()._parse_band_dat_content(f.read_text())
    assert res["num_bands"] == nbnd
    assert len(res["eigenvalues"]) == NKPT
    np.testing.assert_allclose(np.array(res["eigenvalues"]), _expected(nbnd), atol=1e-9)


@pytest.mark.parametrize("nbnd", [999, 1000, 1500])
def test_electronic_analyzer_stitches_wrapped_rows(tmp_path, nbnd):
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    f = tmp_path / "x_band.BAND.DAT"
    _write_band_dat(f, nbnd)
    _, _, k, energies = AdvancedElectronicAnalyzer().read_band_data(f)
    assert len(k) == NKPT
    np.testing.assert_allclose(energies, _expected(nbnd), atol=1e-9)
