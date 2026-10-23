"""A phonon dispersion and a phonon DOS from two folders in one figure (1.1.6): mace
plotting --phonon-band-dos (--partner), as mace gui's Add Phonon DOS… and Add
Phonon Bands…, through the electronic pair's code (results.pair_check,
bandos_cli.plot). The halves share zero frequency and one unit (no Fermi level, no
DOS shift); an electronic half with a phonon one, another system or another
supercell is refused. On the bundled example suite's real CRYSTAL23 runs: diamond
(3D) and h-BN (2D, a slightly imaginary acoustic mode at Γ)."""
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "examples" / "suite"
DIAMOND = SUITE / "2-dense-3d" / "diamond"
HBN = SUITE / "3-slab-2d" / "hBN"
CM_PER_EV = 8065.543937


def put(src_dir, names, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        xz, plain = src_dir / (name + ".xz"), src_dir / name
        if xz.is_file():
            (dst_dir / name).write_bytes(lzma.decompress(xz.read_bytes()))
        elif plain.is_file():
            (dst_dir / name).write_bytes(plain.read_bytes())
        else:
            pytest.skip("the bundled example suite is not here")


def run_files(material, kind):
    return tuple(f"{material}_{kind}{ext}" for ext in (".f25", ".out", ".d12"))


def mace(cwd, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300, env=env)


def layout_of(html: Path):
    text = html.read_text()
    dec = json.JSONDecoder()
    pos = text.index("[", text.rindex("Plotly.newPlot("))
    data, pos = dec.raw_decode(text, pos)
    layout, _ = dec.raw_decode(text, text.index("{", pos))
    return data, layout


@pytest.fixture
def runs(tmp_path):
    for src, material in ((DIAMOND, "diamond"), (HBN, "hBN")):
        put(src / "PHONON_BAND", run_files(material, "phonon_band"), tmp_path / material / "PHONON_BAND")
        put(src / "PHONON_DOS", run_files(material, "phonon_dos"), tmp_path / material / "PHONON_DOS")
    put(DIAMOND / "BAND", ("diamond_band.f25", "diamond_band.out", "diamond_band.d3"), tmp_path / "diamond" / "BAND")
    put(DIAMOND / "DOSS", ("diamond_doss.DOSS.DAT", "diamond_doss.out", "diamond_doss.d3"), tmp_path / "diamond" / "DOSS")
    return tmp_path


def _pair(runs, material):
    return (runs / material / "PHONON_BAND" / f"{material}_phonon_band.f25",
            runs / material / "PHONON_DOS" / f"{material}_phonon_dos.f25")


@pytest.mark.parametrize("material,cell", [("diamond", "2×2×2"), ("hBN", "2×2×1")])
def test_a_phonon_pair_is_one_system_on_zero_frequency(runs, material, cell):
    pytest.importorskip("ase")          # the runs' outputs are read with it (results.pair_check)
    from mace.gui import results
    band, dos = _pair(runs, material)
    r = results.pair_check(band, dos)
    assert r["error"] is None and r["domain"] == "phonon", r
    # no Fermi level: the SCF's E_F in the outputs never moves the DOS (it did: -7.45 eV for h-BN)
    assert r["dos_shift"] == 0.0 and not r["warnings"], r
    assert r["band"]["fermi_ev"] is None and r["dos"]["fermi_ev"] is None, r
    assert any("zero frequency" in n and f"{cell} supercell" in n for n in r["notes"]), r["notes"]


@pytest.mark.parametrize("band,dos", [("diamond/BAND/diamond_band.f25", "diamond/PHONON_DOS/diamond_phonon_dos.f25"),
                                      ("diamond/PHONON_BAND/diamond_phonon_band.f25", "diamond/DOSS/diamond_doss.DOSS.DAT")])
def test_an_electronic_half_never_pairs_with_a_phonon_half(runs, band, dos):
    pytest.importorskip("ase")          # the runs' outputs are read with it (results.pair_check)
    from mace.gui import results
    r = results.pair_check(runs / band, runs / dos)
    assert r["error"] and "electronic bands and phonons are never one figure" in r["error"], r


def test_an_electronic_pair_of_a_spin_locked_scf_reads_the_derived_fermi_level(runs):
    # diamond's spin-locked SCF prints CRYSTAL's -1.0 Ha placeholder: both halves take the derived E_F
    # (was "The Fermi levels differ by -25.09 eV", the DOSS output's placeholder taken as a value)
    pytest.importorskip("ase")
    from mace.gui import results
    r = results.pair_check(runs / "diamond/BAND/diamond_band.f25", runs / "diamond/DOSS/diamond_doss.DOSS.DAT")
    assert r["error"] is None and not r["warnings"], r
    assert r["band"]["fermi_ev"] == pytest.approx(r["dos"]["fermi_ev"], abs=0.01)
    assert r["band"]["fermi_ev"] == pytest.approx(-2.122, abs=0.01) and r["dos_shift"] == 0.0


def test_another_supercell_is_refused(runs):
    pytest.importorskip("ase")          # the runs' outputs are read with it (results.pair_check)
    from mace.gui import results
    band, dos = _pair(runs, "diamond")
    out = dos.with_suffix(".out")
    text = out.read_text()
    assert " E1     2.000     0.000     0.000" in text
    out.write_text(text.replace(" E1     2.000     0.000     0.000", " E1     3.000     0.000     0.000", 1))
    r = results.pair_check(band, dos)
    assert r["error"] and "2×2×2 supercell" in r["error"] and "3×2×2" in r["error"], r
    # without an output, the run's deck says (SCELPHONO)
    out.unlink()
    deck = dos.with_suffix(".d12")
    deck.write_text(deck.read_text().replace("SCELPHONO\n2 0 0", "SCELPHONO\n4 0 0"))
    r = results.pair_check(band, dos)
    assert r["error"] and "4×2×2" in r["error"], r


def test_the_listing_marks_phonon_files(runs):
    from mace.gui import filekind
    for f in _pair(runs, "diamond") + _pair(runs, "hBN"):
        assert filekind.f25_phonon(f), f
    assert not filekind.f25_phonon(runs / "diamond" / "BAND" / "diamond_band.f25")


def test_one_frequency_axis_with_the_imaginary_mode_and_the_modes_window(runs):
    """Both panels on the band panel's y axis; h-BN's slightly imaginary mode inside the window on both sides; the
    window follows the modes, not the PDOS grid's upper limit (2000 cm⁻¹ in these decks)."""
    pytest.importorskip("plotly")
    from mace.gui import results
    from mace.plotting import plotly_bands as PB
    for material in ("diamond", "hBN"):
        band_f, dos_f = _pair(runs, material)
        band, dos = results.band_data(band_f), results.dos_data(dos_f)
        fig = PB.figure(band, dos, {}, PB.opts_for())
        lay = fig["layout"]
        assert "yaxis2" not in lay and {t.get("yaxis", "y") for t in fig["data"]} == {"y"}
        assert {t.get("xaxis") for t in fig["data"]} >= {"x", "x2"}
        assert lay["yaxis"]["title"]["text"] == "Frequency (cm<sup>−1</sup>)"
        lo = min(v for c in band["up"] for v in c if v is not None) * CM_PER_EV
        hi = max(v for c in band["up"] for v in c if v is not None) * CM_PER_EV
        assert lay["yaxis"]["range"][0] < min(lo, 0) and hi < lay["yaxis"]["range"][1] < 1.2 * hi, (material, lay["yaxis"]["range"], hi)
        if material == "hBN":
            assert lo < 0          # the real run's imaginary acoustic branch near Γ


def test_cli_phonon_band_dos_from_two_folders(runs):
    pytest.importorskip("plotly")
    cwd = runs / "diamond" / "PHONON_BAND"
    r = mace(cwd, "--phonon-band-dos", "diamond_phonon_band.f25", "--partner", "../PHONON_DOS/diamond_phonon_dos.f25",
             "--units", "THz", "--style", "publication", "--format", "html", "-o", "out")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2×2×2 supercell" in r.stdout and "Warning" not in r.stdout, r.stdout
    figs = sorted(p.name for p in (cwd / "out").iterdir())
    assert figs == ["diamond.PHONON_BAND_DOS.html"], figs
    data, lay = layout_of(cwd / "out" / figs[0])
    assert lay["yaxis"]["title"]["text"] == "Frequency (THz)" and "xaxis2" in lay and "yaxis2" not in lay
    assert "states/THz" in lay["xaxis2"]["title"]["text"]
    # the reverse: a phonon DOS with the dispersion from another folder (a 2D slab)
    cwd = runs / "hBN" / "PHONON_DOS"
    r = mace(cwd, "--phonon-band-dos", "hBN_phonon_dos.f25", "--partner", "../PHONON_BAND/hBN_phonon_band.f25",
             "--format", "png", "-o", "out")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cwd / "out" / "hBN.PHONON_BAND_DOS.png").stat().st_size > 10000


@pytest.mark.parametrize("args,why", [
    (("diamond_phonon_band.f25", "--partner", "../../hBN/PHONON_DOS/hBN_phonon_dos.f25"), "not the same system"),
    (("diamond_phonon_band.f25", "--partner", "../DOSS/diamond_doss.DOSS.DAT"), "never one figure"),
    (("../BAND/diamond_band.f25", "--partner", "../DOSS/diamond_doss.DOSS.DAT"), "is an electronic band structure: --phonon-band-dos"),
])
def test_cli_refuses_a_mismatched_pair(runs, args, why):
    pytest.importorskip("plotly")
    cwd = runs / "diamond" / "PHONON_BAND"
    r = mace(cwd, "--phonon-band-dos", *args, "--format", "html", "-o", "out")
    assert r.returncode != 0 and why in r.stdout, r.stdout[-800:]
    assert not (cwd / "out").exists() or not list((cwd / "out").iterdir())


def test_cli_phonon_band_dos_is_plotly_only(runs):
    cwd = runs / "diamond" / "PHONON_BAND"
    r = mace(cwd, "--phonon-band-dos", "diamond_phonon_band.f25", "--partner", "../PHONON_DOS/diamond_phonon_dos.f25",
             "--backend", "matplotlib", "--format", "png", "-o", "out")
    assert r.returncode != 0 and "drawn by the plotly backend" in r.stdout, r.stdout[-600:]


def test_gui_command_is_the_cli_flag():
    """The GUI's Publication Figure and Copy CLI Command run mace plotting --phonon-band-dos (kind phbandos)."""
    from mace.gui import plots
    from mace.plotting.main import create_parser
    assert plots.KINDS["phbandos"][0] == "--phonon-band-dos" and "phbandos" in plots.PAIR_KINDS
    band = REPO / "examples" / "x" / "PHONON_BAND" / "x_phonon_band.f25"
    cmd = plots.command(band, "phbandos", partner=band.parent.parent / "PHONON_DOS" / "x_phonon_dos.f25")
    assert cmd == "mace plotting --phonon-band-dos x_phonon_band.f25 --partner ../PHONON_DOS/x_phonon_dos.f25", cmd
    args = create_parser().parse_args(cmd.split()[2:])
    assert args.phonon_band_dos and args.partner == "../PHONON_DOS/x_phonon_dos.f25"
