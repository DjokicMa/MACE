"""IR and Raman spectra broadened from a FREQ output's modes (plan E4), against
the curves CRYSTAL itself wrote (IRSPEC.DAT / RAMSPEC.DAT) for the same runs.

With CRYSTAL's default width (DAMPFAC = 8 cm-1, a full width at half
maximum) the strongest isolated peaks sit where CRYSTAL's do (2 cm-1) and
are as wide (20 %). Real corpus outputs, read in place (read only); skipped
without the gitignored corpus."""
from pathlib import Path

import pytest

import guiharness as H

np = pytest.importorskip("numpy")
pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
CORPUS = H.REPO_ROOT / "test" / "FREQ"
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present (gitignored)")


def _fwhm(x, y, x0):
    """Width at half the local maximum near x0, half-max crossings interpolated."""
    i = int(np.argmin(abs(x - x0)))
    lo, hi = max(0, i - 8), min(len(x), i + 9)
    i = lo + int(np.argmax(y[lo:hi]))
    h = y[i] / 2
    left = i
    while left > 0 and y[left] > h:
        left -= 1
    right = i
    while right < len(y) - 1 and y[right] > h:
        right += 1
    xl = x[left] + (h - y[left]) * (x[left + 1] - x[left]) / (y[left + 1] - y[left])
    xr = x[right - 1] + (h - y[right - 1]) * (x[right] - x[right - 1]) / (y[right] - y[right - 1])
    return x[i], xr - xl


def _pairs():
    outs = []
    for out in sorted(CORPUS.glob("*.out")):
        ir, ra = out.with_name(out.stem + "_IRSPEC.DAT"), out.with_name(out.stem + "_RAMSPEC.DAT")
        if ir.is_file() and ra.is_file():
            outs.append(out)
    return outs


@needs_corpus
def test_broadened_peaks_match_crystal_spec_files():
    from mace.gui import spectra
    from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
    from mace.plotting.handlers.spectra_api import read_spectrum
    checked = 0
    for out in _pairs()[:6]:
        parsed = {"modes": Crystal23FreqParser(str(out)).modes}
        for kind, suffix, col in (("ir", "_IRSPEC.DAT", 2), ("raman", "_RAMSPEC.DAT", 1)):
            dat = read_spectrum(str(out.with_name(out.stem + suffix)))
            fwhm = spectra.dampfac(out, kind) or spectra.DEFAULT_FWHM
            r = spectra.from_modes(parsed, kind, fwhm=fwhm)
            x, y = np.array(r["x"]), np.array(next(iter(r["columns"].values())))
            peaks = r["peaks"]
            freqs = np.array([p["freq"] for p in peaks])
            isolated = [p for p in peaks if (np.abs(freqs - p["freq"]) < 2 * fwhm).sum() == 1]
            isolated.sort(key=lambda p: -p["intens"])
            if len(isolated) < 3:
                continue
            for p in isolated[:5]:
                xs_ref, w_ref = _fwhm(dat[:, 0], dat[:, col], p["freq"])
                xs_us, w_us = _fwhm(x, y, p["freq"])
                assert abs(xs_ref - p["freq"]) <= 2.0, (out.name[-40:], kind, p, xs_ref)
                assert abs(xs_us - xs_ref) <= 2.0, (out.name[-40:], kind, xs_us, xs_ref)
                assert abs(w_us - w_ref) <= 0.2 * w_ref, (out.name[-40:], kind, p["mode"], w_us, w_ref)
            checked += 1
    assert checked >= 6, checked


def test_default_width_and_shapes(tmp_path):
    """FWHM: the run's DAMPFAC (from <stem>.d12 or MACE's <stem>_temp.d12), else 8; the
    curve's peaks have that full width; Raman's default column is the powder total."""
    from mace.gui import spectra
    out = tmp_path / "run.out"
    out.write_text("")
    assert spectra.dampfac(out, "ir") is None
    (tmp_path / "run_temp.d12").write_text("t\nMOLECULE\n1\nEND\nFREQCALC\nINTENS\nIRSPEC\nDAMPFAC\n12.5\nEND\nRAMSPEC\nEND\nEND\n")
    assert spectra.dampfac(out, "ir") == 12.5 and spectra.dampfac(out, "raman") is None
    parsed = {"modes": [
        {"mode": 1, "freq": 0.0, "irrep": "A", "imaginary": False, "ir_active": True, "ir_intens": 0.0, "raman_active": True, "raman_intens": 0.0},
        {"mode": 7, "freq": 1000.0, "irrep": "B2", "imaginary": False, "ir_active": True, "ir_intens": 100.0, "raman_active": True, "raman_intens": 1000.0},
        {"mode": 8, "freq": 1500.0, "irrep": "A1", "imaginary": False, "ir_active": False, "ir_intens": 0.0, "raman_active": True, "raman_intens": 500.0}]}
    for kind, shape in (("ir", "lorentzian"), ("ir", "gaussian"), ("raman", "lorentzian"), ("raman", "pseudo-voigt")):
        r = spectra.from_modes(parsed, kind, shape=shape, fwhm=8.0, eta=0.5)
        x, y = np.array(r["x"]), np.array(next(iter(r["columns"].values())))
        top, w = _fwhm(x, y, 1000.0)
        assert abs(top - 1000.0) < 0.6 and abs(w - 8.0) < 0.25, (kind, shape, top, w)
        # each line has unit area times its intensity, as CRYSTAL's curves
        area = np.trapezoid(y[(x > 900) & (x < 1100)], x[(x > 900) & (x < 1100)]) if hasattr(np, "trapezoid") else \
            np.trapz(y[(x > 900) & (x < 1100)], x[(x > 900) & (x < 1100)])
        want = 100.0 if kind == "ir" else 1000.0
        assert abs(area - want) < 0.1 * want, (kind, shape, area)
    r = spectra.from_modes(parsed, "raman")
    assert list(r["columns"]) == ["total"] and [p["mode"] for p in r["peaks"]] == [7, 8]
    assert r["ylabel"] == "Raman intensity (relative, max 1000)"
    with pytest.raises(ValueError, match="no IR or Raman intensities"):
        spectra.from_modes({"modes": [dict(parsed["modes"][0])]}, "ir")


def _modes_table(rows):
    """mace plotting's parser's expansion of CRYSTAL rows (first, last, freq, ir intensity, raman):
    one entry per mode, every mode of a set carrying the row's intensity."""
    modes = []
    for first, last, freq, ir, ra in rows:
        for n in range(first, last + 1):
            modes.append({"mode": n, "freq": freq, "irrep": "T1u" if last > first else "A1", "imaginary": False,
                          "degeneracy": last - first + 1, "ir_active": ir > 0, "ir_intens": ir,
                          "raman_active": ra > 0, "raman_intens": ra if n == first else 0.0})
    return {"modes": modes}


def test_a_degenerate_set_counts_once():
    """CRYSTAL's INTENS of a degenerate set already holds the degeneracy (INTENS_n = ... d_n ...):
    a triply degenerate row of 300 km/mol is one 300 km/mol peak, as tall as a single mode's 300."""
    from mace.gui import spectra
    parsed = _modes_table([(1, 3, 0.0, 0.0, 0.0), (4, 6, 400.0, 300.0, 0.0), (7, 7, 900.0, 300.0, 50.0)])
    peaks = spectra.active_peaks(parsed, "ir")
    assert [(p["mode"], p["degeneracy"], p["intens"]) for p in peaks] == [(4, 3, 300.0), (7, 1, 300.0)]
    d = spectra.from_modes(parsed, "ir")
    x, y = np.asarray(d["x"]), np.asarray(d["columns"]["absorbance"])
    h4, h7 = y[np.argmin(abs(x - 400))], y[np.argmin(abs(x - 900))]
    assert abs(h4 / h7 - 1) < 1e-3, (h4, h7)
    assert [p["mode"] for p in spectra.active_peaks(parsed, "raman")] == [7]


@needs_corpus
def test_a_degenerate_set_counts_once_on_a_real_crystal(tmp_path):
    """AgCl rock salt: one IR-active Fu set (modes 4-6); its broadened area is the row's INTENS."""
    from mace.gui import freq, spectra
    from mace.gui.outindex import build
    out = next(CORPUS.glob("Ag1Cl1_sym_*freq*.out"), None)
    if out is None:
        pytest.skip("no AgCl FREQ output in the corpus")
    start, end = build(out).freq
    parsed = freq.parse_job(path=str(out), start=start, end=end, tmpdir=str(tmp_path))
    rows = [m for m in parsed["modes"] if m["ir_active"] and m["ir_intens"] > 0 and m["freq"] > 1]
    assert len(rows) == 3 and len({m["ir_intens"] for m in rows}) == 1
    peaks = spectra.active_peaks(parsed, "ir")
    assert len(peaks) == 1 and peaks[0]["degeneracy"] == 3
    d = spectra.from_modes(parsed, "ir", fwhm=8.0, xmax=3000.0)
    x, y = np.asarray(d["x"]), np.asarray(d["columns"]["absorbance"])
    area = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))
    # the Lorentzian's area between 0 and 3000 cm-1 (a 62 cm-1 peak loses its tail below 0)
    f0, half = rows[0]["freq"], 4.0
    kept = (np.arctan((3000 - f0) / half) + np.arctan(f0 / half)) / np.pi
    assert abs(area / (rows[0]["ir_intens"] * kept) - 1) < 2e-3, (area, kept)


def test_broadening_grid_is_bounded():
    from mace.gui import spectra
    parsed = _modes_table([(1, 1, 500.0, 10.0, 0.0)])
    for bad in (1e9, float("nan"), float("inf"), 1.0):
        with pytest.raises(ValueError, match="xmax"):
            spectra.from_modes(parsed, "ir", xmax=bad)
    with pytest.raises(ValueError, match="eta"):
        spectra.from_modes(parsed, "raman", shape="pseudo-voigt", eta=float("nan"))
    assert spectra.from_modes(parsed, "ir", xmax=4000.0)["x"][-1] <= 4001
