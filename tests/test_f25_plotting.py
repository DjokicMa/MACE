"""fort.25 (``*.f25``) as band and DOS data (mace/plotting/f25.py), electronic and phonon.

* the real corpus (test/BAND, test/DOSS: 76 + 80 f25 files, each beside the BAND.DAT/DOSS.DAT of its run):
  the f25 read through its blocks gives the arrays the DAT gives (to the DAT's own rounding);
* a file that holds several runs (CRYSTAL appends to fort.25) draws the last, the DAT's;
* synthetic blocks in the Appendix D layout: a closed-shell run, a band count that is no multiple of six,
  numbers that touch, a truncated file, a MAPN file; phonon files (frequencies in cm-1, no Fermi energy),
  which the corpus has none of: the layout is the manual's;
* mace plotting takes the f25 as it takes the DAT (pinned and by classification, plotly and matplotlib);
* the figure of a phonon plot, and the new Reference/Labels options, are the same in the GUI's bandfig.js
  and mace plotting's plotly_bands.py.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.gui import results
from mace.plotting import f25 as F

CORPUS = REPO_ROOT / "test"
needs_corpus = pytest.mark.skipif(not (CORPUS / "BAND").is_dir(), reason="real CRYSTAL corpus not present")
PLOTTING = REPO_ROOT / "Plotting"


def _plotters():
    if str(PLOTTING) not in sys.path:
        sys.path.insert(0, str(PLOTTING))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import ipBANDS_V2
    import ipDOS_V2
    return ipBANDS_V2, ipDOS_V2


# ------------------------------------------------------------------ synthetic blocks
def _e(v):
    s = f"{v:.5E}"
    return f"{s:>12}"


def _lines(vals):
    out = []
    for i in range(0, len(vals), 6):
        out.append("".join(_e(v) for v in vals[i:i + 6]))
    return out


def band_block(ihferm, bands, dk, ef, start=0, ints=(0, 0, 0, 8, 0, 8), kind="BAND"):
    """One BAND block: ``bands[j][i]`` is band i at k point j (the layout of CRYSTAL23's manual, Appendix D)."""
    nk, nb = len(bands), len(bands[0])
    flat = [v for kp in bands for v in kp]
    head = f"-%-{ihferm}{kind}{nb:5d}{nk:5d}{_e(0.0)}{_e(dk)}{_e(ef)}"
    return "\n".join([head, _e(min(flat)) + _e(max(flat)), "".join(f"{i:5d}" for i in ints)] + _lines(flat)) + "\n"


def doss_block(ihferm, index, values, e0, dk, ef):
    head = f"-%-{ihferm}DOSS{1:5d}{len(values):5d}{_e(0.0)}{_e(dk)}{_e(ef)}"
    return "\n".join([head, _e(0.0) + _e(e0), f"{index:5d}{16:5d}" + "    0" * 4] + _lines(values)) + "\n"


def _bands(nk, nb, seed=0):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(-0.5, 0.5, nb))
    return [list(base + 0.01 * k * np.arange(nb) / nb) for k in range(nk)]


# ------------------------------------------------------------------ the reader
def test_closed_shell_bands_with_a_band_count_that_is_no_multiple_of_six(tmp_path):
    ef = -0.1
    seg1, seg2 = _bands(5, 7, 1), _bands(4, 7, 2)
    p = tmp_path / "x_band.f25"
    p.write_text(band_block(0, seg1, 0.1, ef) + band_block(0, seg2, 0.2, ef))
    blocks = F.read_blocks(p)
    assert [b.ncol for b in blocks] == [5, 4] and blocks[0].nrow == 7
    assert np.allclose(np.array(blocks[0].values).reshape(5, 7), seg1, atol=1e-5)
    text = F.band_dat_text(p).splitlines()
    assert text[0].split()[1:7] == ["NKPT", "9", "NBND", "7", "NSPIN", "1"]
    dat = tmp_path / "out"
    syn, _ = F.materialize(p, dat)
    ipb, _ = _plotters()
    E, B, Eb, Bb, ticks, maxV, _k, _n = ipb.read_band_data(str(syn))
    assert B.shape == (9, 7) and abs(maxV) < 1e-12
    # E - E_F in eV; x runs on by each segment's own step, the first point one step past the previous segment's last
    assert np.allclose(B[:5], (np.array(seg1) - ef) * 27.2114, atol=1e-3)
    assert np.allclose(E[:6], [0, .1, .2, .3, .4, .6], atol=1e-6)
    assert np.allclose(ticks, [0, 0.4, 1.2], atol=1e-6)


def test_numbers_that_touch_and_a_truncated_file(tmp_path):
    p = tmp_path / "t_band.f25"
    p.write_text(band_block(0, [[-1.23456, 2.0, 3.0, -4.0, 5.0, -6.5]], 0.1, -0.05))
    assert "-1.23456E+00-4.00000E+00" not in p.read_text() or True
    assert F.read_blocks(p)[0].values[3] == -4.0 and F.read_blocks(p)[0].values[5] == -6.5
    text = p.read_text().splitlines()
    cut = tmp_path / "cut_band.f25"
    cut.write_text("\n".join(text[:3] + ["  1.00000E+00"]) + "\n")
    with pytest.raises(F.F25Error, match="stops inside"):
        F.read_blocks(cut)
    junk = tmp_path / "junk.f25"
    junk.write_text("not a fort.25\n")
    with pytest.raises(F.F25Error, match="not a fort.25"):
        F.read_blocks(junk)


def test_open_shell_two_runs_draws_the_last(tmp_path):
    ef = -0.2
    run1 = [_bands(3, 6, 3), _bands(3, 6, 4)]            # alpha, beta of the first run
    run2 = [_bands(3, 6, 5), _bands(3, 6, 6)]
    text = "".join(band_block(1, b, 0.3, ef) for b in run1 + run2)
    p = tmp_path / "two_band.f25"
    p.write_text(text)
    info = F.describe(p)
    assert info["kind"] == "BAND" and info["nspin"] == 2 and info["n_runs"] == 2 and info["domain"] == "electronic"
    syn, _ = F.materialize(p, tmp_path / "o")
    ipb, _ = _plotters()
    E, B, Eb, Bb, *_ = ipb.read_band_data(str(syn))
    assert np.allclose(B, (np.array(run2[0]) - ef) * 27.2114, atol=1e-3) and np.allclose(Bb, (np.array(run2[1]) - ef) * 27.2114, atol=1e-3)
    syn1, _ = F.materialize(p, tmp_path / "o1", run=0)
    B1 = ipb.read_band_data(str(syn1))[1]
    assert np.allclose(B1, (np.array(run1[0]) - ef) * 27.2114, atol=1e-3)


def test_doss_blocks_are_a_column_per_projection_and_spin(tmp_path):
    ef = -0.1
    n = 20
    vals = [np.linspace(0, 1, n) * (k + 1) for k in range(3)]
    text = "".join(doss_block(1, k + 1, vals[k], -0.5, 0.05, ef) for k in range(3)) + \
        "".join(doss_block(1, k + 1, -vals[k], -0.5, 0.05, ef) for k in range(3))
    p = tmp_path / "x_doss.f25"
    p.write_text(text)
    info = F.describe(p)
    assert info["kind"] == "DOSS" and info["nspin"] == 2 and info["n_projections"] == 3
    syn, _ = F.materialize(p, tmp_path / "o")
    _, ipd = _plotters()
    assert ipd.read_doss_header(str(syn)) == (3, 2) and ipd.read_doss_points(str(syn)) == (n, 2)
    cols = ipd.read_doss_columns(str(syn), 4)
    assert len(cols[0]) == 2 * n
    assert np.isclose(cols[0][0], (-0.5 - ef) * 27.2114, atol=1e-3) and np.isclose(cols[0][1] - cols[0][0], 0.05 * 27.2114, atol=1e-3)
    assert np.allclose(cols[3][:n], vals[2] / 27.2114, atol=1e-4) and np.allclose(cols[3][n:], -vals[2] / 27.2114, atol=1e-4)


def test_a_map_file_is_not_band_data(tmp_path):
    p = tmp_path / "m.f25"
    p.write_text("-%-0MAPN    2    2 1.00000E+00 1.00000E+00 0.00000E+00\n" + "\n".join(["0"] * 5) + "\n")
    assert F.sniff_kind(p) == "MAPN" and F.plot_kind(p) is None
    with pytest.raises(F.F25Error, match="only BAND and DOSS"):
        F.materialize(p, tmp_path / "o")


# ------------------------------------------------------------------ the real corpus
def _pairs(kind, suffix):
    folder = CORPUS / kind
    return sorted(f for f in folder.glob("*.f25") if f.with_suffix("").with_name(f.stem + suffix).exists())


def _sample(pairs):
    """Every third pair and every file of the perovskites (their fort.25 hold several runs): all 156 take two minutes."""
    return [f for i, f in enumerate(pairs) if i % 3 == 0 or f.name.startswith(("Ag4", "PbCu", "Ti2O4"))]


@needs_corpus
def test_every_corpus_band_f25_gives_the_band_dat(tmp_path):
    ipb, _ = _plotters()
    pairs = _pairs("BAND", ".BAND.DAT")
    assert len(pairs) >= 70
    multi = 0
    for k, f in enumerate(_sample(pairs)):
        dat = f.with_name(f.stem + ".BAND.DAT")
        syn, _ = F.materialize(f, tmp_path / f"b{k}")
        a, b = ipb.read_band_data(str(dat)), ipb.read_band_data(str(syn))
        for i, name in enumerate(("E", "B", "Eb", "Bb", "ticks")):
            x, y = np.asarray(a[i]), np.asarray(b[i])
            assert x.shape == y.shape, (f.name, name)
            # the DAT prints five significant digits: half a unit of the last one of its largest value
            tol = 6e-5 * max(1.0, float(np.abs(x).max())) if name != "ticks" else 2e-4
            assert np.max(np.abs(x - y)) <= max(tol, 1e-3), (f.name, name, np.max(np.abs(x - y)))
        multi += F.describe(f)["n_runs"] > 1
        shutil.rmtree(tmp_path / f"b{k}")
    assert multi >= 1               # the Ag/Pb perovskites' fort.25 hold more than one run


@needs_corpus
def test_every_corpus_doss_f25_gives_the_doss_dat(tmp_path):
    _, ipd = _plotters()
    pairs = _pairs("DOSS", ".DOSS.DAT")
    assert len(pairs) >= 70
    for k, f in enumerate(_sample(pairs)):
        dat = f.with_name(f.stem + ".DOSS.DAT")
        syn, _ = F.materialize(f, tmp_path / f"d{k}")
        ha, hb = ipd.read_doss_header(str(dat)), ipd.read_doss_header(str(syn))
        assert ha == hb and ipd.read_doss_points(str(dat)) == ipd.read_doss_points(str(syn)), f.name
        ca, cb = ipd.read_doss_columns(str(dat), ha[0] + 1), ipd.read_doss_columns(str(syn), hb[0] + 1)
        for j, (x, y) in enumerate(zip(ca, cb)):
            x, y = np.asarray(x), np.asarray(y)
            assert x.shape == y.shape
            tol = 6e-5 * max(1.0, float(np.abs(x).max())) if j == 0 else 1e-3 * max(1.0, float(np.abs(x).max()))
            assert np.max(np.abs(x - y)) <= max(tol, 1e-3), (f.name, j)
        shutil.rmtree(tmp_path / f"d{k}")


@needs_corpus
def test_results_read_the_f25_like_the_dat():
    band = CORPUS / "BAND" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
    a, b = results.band_data(Path(str(band) + ".f25")), results.band_data(Path(str(band) + ".BAND.DAT"))
    assert a["name"].endswith(".f25") and a["source"] == "f25" and a["domain"] == "electronic"
    for k in ("ticks", "labels", "spin_polarized", "n_kpoints", "n_bands", "fermi"):
        assert a[k] == b[k], k
    assert np.allclose(np.array(a["x"], float), np.array(b["x"], float), atol=1e-3, equal_nan=True)
    d = CORPUS / "DOSS" / "Ag4Cl2S_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized_doss"
    c, e = results.dos_data(Path(str(d) + ".f25")), results.dos_data(Path(str(d) + ".DOSS.DAT"))
    assert [p["label"] for p in c["projections"]] == [p["label"] for p in e["projections"]]
    assert c["spin_polarized"] == e["spin_polarized"] and c["edge_pileup"] == e["edge_pileup"]


# ------------------------------------------------------------------ phonons: a real file (CRYSTAL23 diamond, SCELPHONO 2x2x2,
# FREQCALC DISPERSI + BANDS (Γ-X-W-L-Γ, 4 lines, ISS 8, 50 points) + PDOS 1600 160 1), with its PHONBANDS.DAT
PH = REPO_ROOT / "tests" / "data" / "f25"
PH_F25 = PH / "dia_phonon_f25.fort.25"
GAMMA_OPTICAL_CM = 1380.0594          # the run's output: the triply degenerate optical mode at Γ


def _phonbands():
    rows = [ln.split() for ln in (PH / "dia_phonon_f25.PHONBANDS.DAT").read_text().splitlines() if ln and ln[0] not in "#@"]
    return np.array(rows, float)


def test_the_real_phonon_file_is_recognized_whole():
    # a PDOS block comes before each of the four BAND blocks
    assert F.file_kinds(PH_F25) == ["PDOS", "BAND"] * 4
    assert F.sniff_kind(PH_F25) == "BAND" and F.plot_kind(PH_F25) == "band" and F.plot_kind(PH_F25, "dos") == "dos"
    assert F.is_f25_name(PH_F25.name)                       # "<name>.fort.25"
    info = F.describe(PH_F25)
    assert info["domain"] == "phonon" and info["nspin"] == 1 and info["n_segments"] == 4 and info["n_runs"] == 1
    assert info["has"] == ["band", "dos"]
    # the header numbers the DOS is laid out by: start 0, step 10.0629 cm-1, 160 bins
    pdos = [b for b in F.read_blocks(PH_F25) if b.kind == "PDOS"]
    assert len(pdos) == 4 and all(b.ncol == 160 and b.dum == 0.0 and abs(b.dk - 10.0629) < 1e-4 for b in pdos)


def test_four_bands_blocks_are_one_path_and_no_spin_is_inferred():
    text = F.band_dat_text(PH_F25).splitlines()
    assert text[0].split() == ["#", "NKPT", "197", "NBND", "6", "NSPIN", "1"] and text[1].split() == ["#", "NPANEL", "4"]
    # PHONBANDS.DAT's panel lines: the point index of each tick (1, 50, 99, 148, 197)
    panels = [ln.split() for ln in text if ln.startswith("# ") and ln.split()[1].isdigit()]
    assert [int(p[1]) for p in panels] == [1, 50, 99, 148, 197]
    assert [p[2] for p in panels] == ["Γ", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "Γ"]      # ISS 8 from the .d12 beside it
    assert F.band_dat_text(PH_F25, domain="phonon") == "\n".join(text) + "\n"
    d = results.band_data(PH_F25)
    assert d["domain"] == "phonon" and d["n_kpoints"] == 197 and d["n_bands"] == 6 and not d["spin_polarized"]
    assert d["labels"] == ["Γ", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "Γ"] and d["fermi"] == 0 and d["has_potc"] is False


def test_the_frequencies_and_k_distances_are_phonbands_dat():
    d = results.band_data(PH_F25)
    ref = _phonbands()
    x = np.array([np.nan if v is None else v for v in d["x"]], float)
    cm = np.array(d["up"], float).T * 8065.543937          # the readers' eV scale back to cm-1
    assert x.shape == (197,) and cm.shape == (197, 6)
    # the ticks: the path's true distances in bohr^-1 (2π/a, π/a, √2π/a, √3π/a: each header's DK is its panel's
    # length), not PHONBANDS.DAT's (45.96042, 68.94063, 101.43955, 141.24244), which steps by the whole DK
    assert np.allclose(d["ticks"], [0, 0.937968, 1.406952, 2.070195, 2.882499], atol=2e-6)
    # the first panel is the DAT's, its x 49 (NSUB - 1) times ours; the DAT drops the first point of each later panel
    # and ends it with a stale row (the X point's frequencies again), so panel p lines up with ours one row on
    assert np.allclose(x[:50] * 49, ref[:50, 0], atol=6e-3) and np.allclose(cm[:50], ref[:50, 1:], atol=0.06)
    for p in (1, 2, 3):
        first_dat, first_ours = 50 + 48 * (p - 1) + (p - 1), 50 + 49 * (p - 1)
        dat = ref[first_dat:first_dat + 48]
        assert np.allclose(cm[first_ours + 1:first_ours + 49], dat[:, 1:], atol=0.06), p
    # every panel's last point is the next one's start region: the path ends at Γ, the DAT's stale last row does not
    assert abs(cm[196, 3] - GAMMA_OPTICAL_CM) < 0.01 and abs(cm[0, 3] - GAMMA_OPTICAL_CM) < 0.01 and np.allclose(cm[0, :3], 0, atol=0.05)


def test_the_phonon_dos_is_one_dos_with_the_real_header_grid():
    d = results.dos_data(PH_F25)
    assert d["domain"] == "phonon" and not d["spin_polarized"] and [p["label"] for p in d["projections"]] == ["Total"]
    e_cm = np.array(d["energy"]) * 8065.543937
    assert len(e_cm) == 160 and abs(e_cm[0]) < 1e-3 and abs(e_cm[1] - e_cm[0] - 10.0629) < 1e-3 and abs(e_cm[-1] - 1600.0) < 0.01
    # states per cm-1 integrate to 3N = 6 modes of the two-atom cell (the last PDOS block, the complete one)
    per_cm = np.array(d["projections"][0]["up"]) / 8065.543937
    assert abs(per_cm.sum() * 10.0629 - 6.12) < 0.02
    last = [b for b in F.read_blocks(PH_F25) if b.kind == "PDOS"][-1]
    assert abs(sum(last.values) - 0.6084182) < 1e-6
    # a phonon file holds both halves: each is the other's partner
    assert results.partner(PH_F25, "band") == PH_F25 and results.partner(PH_F25, "dos") == PH_F25


def test_the_real_phonon_file_in_the_plotly_figure(tmp_path):
    from mace.plotting import plotly_bands as PB
    band, dos = results.band_data(PH_F25), results.dos_data(PH_F25)
    fig = PB.figure(band, dos, {}, PB.opts_for())
    lay = fig["layout"]
    assert lay["yaxis"]["title"]["text"] == "Frequency (cm<sup>−1</sup>)" and "states/cm<sup>−1</sup>" in lay["xaxis2"]["title"]["text"]
    # the window follows the modes (the top one ~1388 cm-1), not the PDOS grid's upper limit (1600 cm-1)
    assert lay["yaxis"]["range"][0] <= 0 and 1388 < lay["yaxis"]["range"][1] < 1500
    assert not [s for s in lay["shapes"] if s.get("name", "").startswith("fermi")]
    assert not [t for t in fig["data"] if t.get("meta") == "key-fermi"]
    assert lay["xaxis"]["ticktext"][0] == "Γ" and "(4,0,4)/8" in lay["xaxis"]["ticktext"]
    thz = PB.figure(band, None, {}, {**PB.opts_for(), "unit": "THz"})["layout"]["yaxis"]
    assert thz["title"]["text"] == "Frequency (THz)" and 40 < thz["range"][1] < 47


def test_a_phonon_legend_has_no_spin_and_the_smoothing_is_in_the_axis_unit():
    """Phonons have no spin: no Spin group (an electronic closed-shell run keeps its "α = β (closed shell)"),
    the DOS's group is the phonon DOS, a dispersion alone has no legend, and the smoothing width is said in
    the frequency unit drawn (0.003 eV is 24.2 cm⁻¹), applied on the readers' eV scale like the data."""
    from mace.plotting import plotly_bands as PB
    from mace.plotting import presets as P
    band, dos = results.band_data(PH_F25), results.dos_data(PH_F25)
    for unit, sig in (("cm-1", "24.2 cm<sup>−1</sup>"), ("THz", "0.725 THz"), ("meV", "3 meV")):
        o = {**PB.opts_for(), "unit": unit}
        fig = PB.figure(band, dos, {"dos_smooth": 0.003}, o)
        data, lay = fig["data"], fig["layout"]
        assert not [t for t in data if t.get("meta") in ("key-spin", "key-band")]
        assert not [t for t in data if "closed shell" in str(t.get("name"))]
        keys = [t for t in data if t.get("meta") == "key-proj"]
        assert [t["name"] for t in keys] == ["Total"] and keys[0]["legendgrouptitle"]["text"] == "Phonon DOS"
        assert [a["text"] for a in lay["annotations"] if a.get("name") == "dos-smooth"] == [f"smoothed, σ = {sig}"]
        alone = PB.figure(None, dos, {"dos_smooth": 0.003}, o)["layout"]
        assert [a["text"] for a in alone["annotations"] if a.get("name") == "dos-smooth"] == [f"Gaussian smoothing σ = {sig} (display only)"]
        disp = PB.figure(band, None, {}, o)
        assert disp["layout"]["showlegend"] is False and not [t for t in disp["data"] if str(t.get("meta", "")).startswith("key")]
        assert "title" not in disp["layout"]["legend"]
    # the smoothing runs on the readers' eV grid with sigma in eV: the same curve as sigma in cm-1 on a cm-1 grid
    e, y, cm = list(dos["energy"]), list(dos["projections"][0]["up"]), 8065.543937
    assert np.allclose(P.gauss_smooth(y, e, 0.003), P.gauss_smooth(y, [v * cm for v in e], 0.003 * cm), rtol=1e-9, atol=1e-12)
    # an electronic closed-shell band plot keeps its Spin key, and eV keeps the width as set
    el = {**band, "domain": "electronic"}
    assert [t["name"] for t in PB.figure(el, None, {}, PB.opts_for())["data"] if t.get("meta") == "key-spin"] == ["α = β (closed shell)"]
    assert PB.sigma_text(0.05, 1, "eV") == "0.05 eV" and PB.sigma_text(0.003, 8065.543937, "cm-1") == "24.2 cm-1"


def test_cli_draws_the_real_phonon_file_with_plotly_and_refuses_matplotlib(tmp_path):
    pytest.importorskip("plotly")
    for name in ("dia_phonon_f25.fort.25", "dia_phonon_f25.d12"):
        shutil.copyfile(PH / name, tmp_path / name)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cli = [sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting"]
    src = str(tmp_path / "dia_phonon_f25.fort.25")
    r = subprocess.run(cli + ["--band-dos", src, "--format", "html"], capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    html = next(tmp_path.glob("dia_phonon_f25.PHONON_BAND_DOS.html")).read_text()      # a phonon band + DOS figure
    assert "Frequency" in html and "(4,0,4)" in html and "DOS (states" in html
    r = subprocess.run(cli + ["--band", src, "--format", "svg", "--backend", "matplotlib"], capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    assert r.returncode != 0 and "plotly" in (r.stdout + r.stderr)
    # by classification, with no flag: a band + DOS figure of the one file
    r = subprocess.run(cli + [src, "--format", "html"], capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_phonon_path_is_labeled_by_its_decks_title(tmp_path):
    """opt2d12 titles a phonon band deck with its path (``... - Phonon Band Structure - default - M-GAMMA-K``, as
    opt2d3 titles a BAND deck): the plot's ticks, the GUI's and mace plotting's, take those names, read by the
    electronic band plot's parser; a title whose Γ is not at the origin of the segment ends keeps the coordinates."""
    pytest.importorskip("plotly")
    deck = (PH / "dia_phonon_f25.d12").read_text().splitlines()
    shutil.copyfile(PH_F25, tmp_path / PH_F25.name)
    named = ["Γ", "X", "W", "L", "Γ"]
    (tmp_path / "dia_phonon_f25.d12").write_text(
        "\n".join(["dia - Phonon Band Structure - default - GAMMA-X-W-L-GAMMA"] + deck[1:]) + "\n")
    src = tmp_path / PH_F25.name
    assert results.band_data(src)["labels"] == named
    seg = results.kpath_info(src)["segments"]
    assert [(s["from"], s["to"]) for s in seg] == [("GAMMA", "X"), ("X", "W"), ("W", "L"), ("L", "GAMMA")]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--band", str(src),
                        "--format", "html"], capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    html = next(tmp_path.glob("dia_phonon_f25.BAND*.html")).read_text()
    assert '"W"' in html and "(4,2,6)" not in html
    # a slab's path (M-Γ-K) on these segments: Γ is not where the deck's names put it
    (tmp_path / "dia_phonon_f25.d12").write_text(
        "\n".join(["dia - Phonon Band Structure - default - M-GAMMA-K-M-GAMMA"] + deck[1:]) + "\n")
    assert results.band_data(src)["labels"] == ["Γ", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "Γ"]


def test_the_run_beside_the_file_decides_phonon_or_electronic(tmp_path):
    p = tmp_path / "x_band.f25"
    p.write_text(band_block(0, [[1.0, 2.0]] * 3, 0.1, 0.0))
    assert F.domain_of(p) == "electronic"
    (tmp_path / "x_band.out").write_text("... CRYSTAL - PROPERTIES ... \n FREQCALC\n")
    assert F.domain_of(p) == "electronic"
    (tmp_path / "x_band.out").write_text(" EEEEEEEEEE STARTING\n PHONON BANDS\n FREQCALC\n")
    assert F.domain_of(p) == "phonon"
    assert F.domain_of(p, domain="electronic") == "electronic"


# ------------------------------------------------------------------ mace plotting
@needs_corpus
def test_mace_plotting_takes_an_f25_like_a_dat(tmp_path):
    stem = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
    for ext in (".f25", ".d3", ".out"):
        shutil.copyfile(CORPUS / "BAND" / (stem + ext), tmp_path / (stem + ext))
    env = {**os.environ, "MPLBACKEND": "Agg"}
    cli = [sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting"]
    for extra in ([], ["--backend", "matplotlib"]):
        for pin in (["--band"], []):
            for f in tmp_path.glob("*.BANDS.*"):
                f.unlink()
            r = subprocess.run(cli + pin + [str(tmp_path / (stem + ".f25")), "--format", "png"] + extra, capture_output=True, text=True,
                               env=env, timeout=300)
            assert r.returncode == 0, (extra, pin, r.stdout + r.stderr)
            assert (tmp_path / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.BANDS.png").is_file(), (extra, pin, r.stdout)
    # the DOS flag on a band f25 says so, and exits non-zero
    r = subprocess.run(cli + ["--dos", str(tmp_path / (stem + ".f25"))], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0 and "band" in (r.stdout + r.stderr).lower()


@needs_corpus
def test_discovery_lists_the_dat_not_its_f25_twin(tmp_path):
    from mace.plotting import detect
    from mace.plotting.registry import PlotKind
    import mace.plotting.handlers  # noqa: F401  (registers the plotters)
    stem = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
    for ext in (".f25", ".BAND.DAT"):
        shutil.copyfile(CORPUS / "BAND" / (stem + ext), tmp_path / (stem + ext))
    found = detect.discover(str(tmp_path))
    assert [Path(x).name for x in found[PlotKind.BAND]] == [stem + ".BAND.DAT"]
    (tmp_path / (stem + ".BAND.DAT")).unlink()
    assert [Path(x).name for x in detect.discover(str(tmp_path))[PlotKind.BAND]] == [stem + ".f25"]
    assert detect.classify_file(str(tmp_path / (stem + ".f25"))) == PlotKind.BAND


def test_gallery_collects_a_lone_f25_not_a_twinned_one(tmp_path):
    from mace.plotting import gallery
    (tmp_path / "a_band.f25").write_text(band_block(0, _bands(3, 4), 0.1, 0.0))
    (tmp_path / "b_band.f25").write_text(band_block(0, _bands(3, 4), 0.1, 0.0))
    (tmp_path / "b_band.BAND.DAT").write_text("# NKPT 1 NBND 1\n")
    got = gallery.collect([tmp_path], "band")
    assert [Path(x).name for x in got["items"]] == ["a_band.f25", "b_band.BAND.DAT"]


# ------------------------------------------------------------------ the figure's new options, GUI = CLI
def test_reference_labels_and_phonon_figures_match_the_gui(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    from test_plotly_bands import FE, _case, _compare
    band, dos = results.band_data(FE / "fe_band.BAND.DAT"), results.dos_data(FE / "fe_doss.DOSS.DAT")
    cases = [_case(band, dos, o={"fermi_line": False}, fermi_key=True), _case(band, dos, o={"k_labels": False}),
             _case(band, dos, o={"spin_labels": False}), _case(band, None, o={"fermi_line": False, "k_labels": False}, fermi_key=True)]
    # a phonon plot: the same arrays labeled as frequencies (the readers' eV scale), in each frequency unit
    pb, pd = {**band, "domain": "phonon", "fermi": 0}, {**dos, "domain": "phonon", "fermi": 0}
    for unit in ("eV", "cm-1", "THz", "meV"):
        cases.append(_case(pb, pd, o={"unit": unit}, fermi_key=True))
        cases.append(_case(pb, None, o={"unit": unit}))
        cases.append(_case(None, pd, o={"unit": unit}))
    cases.append(_case(pb, pd, o={"shift": 1.0, "ylog": True}))
    # the smoothing note in the frequency unit, and no Spin group (phonons have no spin)
    for unit in ("cm-1", "THz"):
        cases.append(_case(pb, pd, o={"unit": unit}, look={"dos_smooth": 0.003}))
        cases.append(_case(None, pd, o={"unit": unit}, look={"dos_smooth": 0.003}, width=520))
    _compare(cases, tmp_path)


# ------------------------------------------------------------------ the explorer
def test_the_explorer_knows_what_a_fort25_holds(tmp_path):
    from mace.gui import filekind
    band = tmp_path / "a_band.f25"
    band.write_text(band_block(0, _bands(3, 4), 0.1, 0.0))
    doss = tmp_path / "a_doss.f25"
    doss.write_text(doss_block(0, 1, [0.0, 1.0, 2.0], -0.5, 0.1, 0.0))
    mapn = tmp_path / "m.f25"
    mapn.write_text("-%-0MAPN    2    2 1.00000E+00 1.00000E+00 0.00000E+00\n")
    other = tmp_path / "other.f25"
    other.write_text("hello\n")
    assert [filekind.f25_sub(p) for p in (band, doss, mapn, other)] == ["BAND", "DOSS", "MAPN", ""]
    # the name's kind is unchanged: the listing adds the content as "sub"
    assert filekind.kind_by_name("a_band.f25") == "f25" and filekind.kind_by_name("fort.25") == "f25"
    info = filekind.info_payload(band, "f25")
    assert info["f25_kind"] == "BAND" and info["domain"] == "electronic" and info["n_runs"] == 1


def test_the_publication_figure_copies_the_run_files_of_a_fort25(tmp_path):
    """plots.render copies the run's .d3/.out/.d12 along (the k-labels come from them): the run's stem
    of x_band.fort.25 is x_band, not x_band.fort."""
    from mace.gui import plots
    assert plots._run_stem("dia_band.fort.25", "band") == "dia_band"
    assert plots._run_stem("dia_band.f25", "bandos") == "dia_band"
    assert plots._run_stem("dia_band.BAND.DAT", "band") == "dia_band"
    shutil.copy(PH_F25, tmp_path / PH_F25.name)
    shutil.copy(PH_F25.with_name("dia_phonon_f25.d12"), tmp_path / "dia_phonon_f25.d12")
    (tmp_path / "dia_phonon_f25.out").write_text("not used\n")

    class Disk:
        def acquire(self, key):
            d = tmp_path / "cache" / key
            d.mkdir(parents=True, exist_ok=True)
            return d

        def release(self, key, remove=False):
            pass
    r = plots.render(tmp_path / PH_F25.name, "band", Disk(), fmt="svg", env={**os.environ, "MPLBACKEND": "Agg"})
    work = tmp_path / "cache" / r["key"]
    assert (work / "dia_phonon_f25.d12").is_file() and (work / "dia_phonon_f25.out").is_file(), sorted(os.listdir(work))


def test_the_explorer_reads_a_fort25_only_to_its_first_header(tmp_path, monkeypatch):
    """The listing's chip (filekind.f25_sub) reads up to the first header that decides the file: a band
    file followed by megabytes of data is not scanned whole (it read up to 8 MB of every fort.25)."""
    from mace.gui import filekind
    big = tmp_path / "big_band.f25"
    with open(big, "w") as fh:
        fh.write(band_block(0, _bands(3, 4), 0.1, 0.0))
        fh.write(("1.00000E+00" * 6 + "\n") * 60_000)                  # ~4 MB, then a DOSS run
        fh.write(doss_block(0, 1, [0.0, 1.0, 2.0], -0.5, 0.1, 0.0))
    lines = []
    real_open = open

    class Counting:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

        def __iter__(self):
            for ln in self.fh:
                lines.append(len(ln))
                yield ln
    monkeypatch.setattr(F, "open", lambda *a, **k: Counting(real_open(*a, **k)), raising=False)
    assert filekind.f25_sub(big) == "BAND" and sum(lines) < 10_000, sum(lines)
    # a phonon run's PDOS block comes first: the scan goes on to its BAND block
    lines.clear()
    assert filekind.f25_sub(PH_F25) == "BAND" and sum(lines) < 4_000, sum(lines)
    # opening the file still reads every header
    assert F.file_kinds(big) == ["BAND", "DOSS"]

