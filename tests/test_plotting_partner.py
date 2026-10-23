"""mace plotting --band-dos --partner (1.1.6): a band file and a DOS file of one
system from two folders in one band + DOS figure, as mace gui's Add DOS… and
Add Band…. The pair is checked (another system is refused) and the DOS goes
on the band run's Fermi level when the two runs' differ. Through mace_cli, on
the bundled example suite's layout (BAND/ and DOSS/ folders)."""
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "examples" / "suite"
POLY = SUITE / "4-polymer-1d" / "polyacetylene"
GRAPHENE = SUITE / "3-slab-2d" / "graphene"


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


def mace(cwd, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300, env=env)


def layout_of(html: Path) -> dict:
    text = html.read_text()
    dec = json.JSONDecoder()
    pos = text.index("[", text.rindex("Plotly.newPlot("))
    _, pos = dec.raw_decode(text, pos)
    layout, _ = dec.raw_decode(text, text.index("{", pos))
    return layout


@pytest.fixture
def runs(tmp_path):
    pytest.importorskip("plotly")
    put(POLY / "BAND", ("polyacetylene_band.BAND.DAT", "polyacetylene_band.d3", "polyacetylene_band.out"), tmp_path / "BAND")
    put(POLY / "DOSS", ("polyacetylene_doss.DOSS.DAT", "polyacetylene_doss.d3", "polyacetylene_doss.out"), tmp_path / "DOSS")
    put(GRAPHENE / "DOSS", ("graphene_doss.DOSS.DAT", "graphene_doss.d3", "graphene_doss.out"), tmp_path / "graphene")
    return tmp_path


def test_without_partner_there_is_no_dos_beside_it(runs):
    r = mace(runs / "BAND", "--band-dos", "polyacetylene_band.BAND.DAT", "--backend", "plotly", "--format", "html", "-o", "out")
    assert r.returncode != 0 and "No DOS of the same run beside it" in r.stdout, r.stdout[-400:]


@pytest.mark.parametrize("file,partner", [("polyacetylene_band.BAND.DAT", "../DOSS/polyacetylene_doss.DOSS.DAT"),
                                          ("polyacetylene_doss.DOSS.DAT", "../BAND/polyacetylene_band.BAND.DAT")])
def test_partner_from_another_folder(runs, file, partner):
    cwd = runs / ("BAND" if "band" in file else "DOSS")
    r = mace(cwd, "--band-dos", file, "--partner", partner, "--backend", "plotly", "--format", "html", "-o", "out")
    assert r.returncode == 0, r.stdout + r.stderr
    figs = list((cwd / "out").glob("*.BAND_DOS.html"))
    assert [f.name for f in figs] == ["polyacetylene.BAND_DOS.html"], r.stdout
    lay = layout_of(figs[0])
    assert "xaxis2" in lay, sorted(lay)              # the bands and the DOS on one energy axis


def test_another_system_is_refused(runs):
    r = mace(runs / "BAND", "--band-dos", "polyacetylene_band.BAND.DAT", "--partner", "../graphene/graphene_doss.DOSS.DAT",
             "--backend", "plotly", "--format", "html", "-o", "out")
    assert r.returncode != 0 and "not the same system" in r.stdout, r.stdout[-600:]
    assert not list((runs / "BAND" / "out").glob("*.html")) if (runs / "BAND" / "out").exists() else True


def test_partner_needs_band_dos(runs):
    r = mace(runs / "BAND", "--band", "polyacetylene_band.BAND.DAT", "--partner", "../DOSS/polyacetylene_doss.DOSS.DAT")
    assert r.returncode == 2 and "--partner is for --band-dos" in r.stderr, r.stderr[-300:]


def test_the_dos_moves_onto_the_band_runs_fermi_level(runs):
    """E_F 0.2 eV apart (an edited output): warned, and the DOS drawn 0.2 eV up, on the bands' E_F."""
    sys.path.insert(0, str(REPO))
    from mace.gui import results
    out = runs / "DOSS" / "polyacetylene_doss.out"
    text = out.read_text()
    ef = -0.143 + 0.2 / results.HARTREE_EV
    out.write_text(text.replace("FERMI ENERGY              -0.143E+00", f"FERMI ENERGY              {ef:.6E}"))
    r = results.pair_check(runs / "BAND" / "polyacetylene_band.BAND.DAT", runs / "DOSS" / "polyacetylene_doss.DOSS.DAT")
    assert r["error"] is None and r["band"]["formula"] == r["dos"]["formula"] == "CH", r
    assert abs(r["dos_shift"] - (ef * results.HARTREE_EV - r["band"]["fermi_ev"])) < 1e-6 and r["dos_shift"] > 0.1, r
    assert any("Fermi levels differ" in w for w in r["warnings"]), r
    p = mace(runs / "BAND", "--band-dos", "polyacetylene_band.BAND.DAT", "--partner", "../DOSS/polyacetylene_doss.DOSS.DAT",
             "--backend", "plotly", "--format", "html", "-o", "out")
    assert p.returncode == 0 and "Warning: The Fermi levels differ" in p.stdout, p.stdout[-600:]
