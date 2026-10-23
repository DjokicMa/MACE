"""ECH3/POT3 grids of flat slabs, polymers and molecules use RANGE, not SCALE.

CRYSTAL23 manual p.322 (ECH3) and p.351 (POT3): for a non-3D system the grid
along the non-periodic directions is either SCALE (the atoms' extent along each
of them, multiplied) or RANGE (ZMIN / ZMAX for a slab, YMIN,ZMIN / YMAX,ZMAX for
a polymer, XMIN,YMIN,ZMIN / XMAX,YMAX,ZMAX for a molecule, in bohr). A flat
structure has no extent to scale: monolayer graphene stopped CRYSTAL with
"GRID3D **** USE RANGE NOT SCALE - SYSTEM HAS ZERO EXTENT IN Z", and planar
polyacetylene ended normally with cubes one point thick along z (the HPCC runs
of the bundled example suite). The writer now uses RANGE there (the atoms + 5 A
on each side) and keeps SCALE, byte for byte, for every structure with depth.

The parents are real CRYSTAL outputs: graphene and polyyne in
tests/data/low_dim_groups, the MgO slab in tests/data/orbitals and planar
polyacetylene from the bundled example suite.
"""
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

_D3 = str(REPO_ROOT / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen

DATA = Path(__file__).parent / "data"
GRAPHENE = DATA / "low_dim_groups" / "graphene_lg80.out"        # monolayer: z extent 0
POLYYNE = DATA / "low_dim_groups" / "polyyne_rg4.out"           # linear chain: y and z extent 0
MGO_SLAB = DATA / "orbitals" / "scf" / "mgoslab.out"            # four atoms in two layers: depth along z
POLYACETYLENE = REPO_ROOT / "examples" / "suite" / "4-polymer-1d" / "polyacetylene" / "SP" / "polyacetylene_sp.out.xz"
SCRIPT = REPO_ROOT / "Crystal_d3" / "CRYSTALOptToD3.py"

ECH3 = {"type": "ECH3", "n_points": 100, "scale": 3, "use_range": False}
POT3 = {"type": "POT3", "n_points": 100, "itol": 5, "scale": 3, "use_range": False}
PAD_BOHR = 5.0 / 0.52917721067


def _generator(out_file, tmp_path, monkeypatch):
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    return d3gen.D3Generator(str(out_file), "CHARGE+POTENTIAL", output_dir=str(tmp_path))


def _polyacetylene(tmp_path):
    if not POLYACETYLENE.is_file():
        pytest.skip("the bundled example suite is not here")
    out = tmp_path / "polyacetylene_sp.out"
    out.write_bytes(lzma.decompress(POLYACETYLENE.read_bytes()))
    return out


def _nums(line):
    return [float(x) for x in line.split()]


def test_monolayer_slab_gets_range_along_z(tmp_path, monkeypatch):
    g = _generator(GRAPHENE, tmp_path, monkeypatch)
    for deck, head in ((g._write_charge_d3(ECH3), ["ECH3", "100"]), (g._write_potential_d3(POT3), ["POT3", "100", "5"])):
        lines = deck.splitlines()
        assert lines[:len(head)] == head
        rest = lines[len(head):]
        assert rest[0] == "RANGE" and rest[-1] == "END" and len(rest) == 4, deck
        (zmin,), (zmax,) = _nums(rest[1]), _nums(rest[2])
        assert zmin == pytest.approx(-PAD_BOHR, abs=1e-3) and zmax == pytest.approx(PAD_BOHR, abs=1e-3)


def test_planar_chain_gets_range_with_its_depth_in_the_plane(tmp_path, monkeypatch):
    g = _generator(_polyacetylene(tmp_path), tmp_path, monkeypatch)
    lines = g._write_charge_d3(ECH3).splitlines()
    assert lines[2] == "RANGE", lines
    (ymin, zmin), (ymax, zmax) = _nums(lines[3]), _nums(lines[4])
    # the chain is centered at z = 5 A (flat) and spans y 3.58-6.42 A (H to H): both padded by 5 A, in bohr
    assert (zmin, zmax) == (pytest.approx(0.0, abs=1e-3), pytest.approx(10.0 / 0.52917721067, abs=1e-3))
    assert ymin == pytest.approx((3.579705 - 5.0) / 0.52917721067, abs=1e-3)
    assert ymax == pytest.approx((6.420295 + 5.0) / 0.52917721067, abs=1e-3)
    assert g._write_potential_d3(POT3).splitlines()[3:6] == ["RANGE", lines[3], lines[4]]


def test_linear_chain_gets_range_along_y_and_z(tmp_path, monkeypatch):
    g = _generator(POLYYNE, tmp_path, monkeypatch)
    lines = g._write_charge_d3(ECH3).splitlines()
    assert lines[2] == "RANGE" and len(_nums(lines[3])) == 2 and len(_nums(lines[4])) == 2, lines


def test_slab_with_depth_keeps_scale_byte_for_byte(tmp_path, monkeypatch):
    g = _generator(MGO_SLAB, tmp_path, monkeypatch)
    assert g._write_charge_d3(ECH3) == "ECH3\n100\nSCALE\n3\nEND"
    assert g._write_potential_d3(POT3) == "POT3\n100\n5\nSCALE\n3\nEND"


def test_explicit_range_still_wins(tmp_path, monkeypatch):
    g = _generator(GRAPHENE, tmp_path, monkeypatch)
    deck = g._write_charge_d3({**ECH3, "use_range": True, "range_min": [-12.0], "range_max": [12.0]})
    assert deck.splitlines() == ["ECH3", "100", "RANGE", "-12.0", "12.0", "END"]


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MPLBACKEND"] = "Agg"
    return env


def _stage(tmp_path, src):
    d = tmp_path / "parent"
    d.mkdir()
    out = d / src.name
    shutil.copy(src, out)
    (d / (src.stem + ".f9")).write_text("wavefunction placeholder\n")
    return out


def _deck(out_dir):
    decks = sorted(Path(out_dir).glob("*.d3"))
    assert len(decks) == 1, decks
    return decks[0].read_text()


def test_workflow_and_enter_route_write_the_same_flat_deck(tmp_path):
    """The workflow engine's CHARGE+POTENTIAL default (mace/workflow/engine.py) and opt2d3 with Enter on stdin
    (the GUI's Setup runs the same CLI) go through the same writer: the same RANGE deck for monolayer graphene."""
    out = _stage(tmp_path, GRAPHENE)
    cfg = tmp_path / "cp.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "CHARGE+POTENTIAL",
                               "configuration": {"calculation_type": "CHARGE+POTENTIAL",
                                                 "charge_config": dict(ECH3), "potential_config": dict(POT3)}}))
    wf = subprocess.run([sys.executable, str(SCRIPT), "--input", str(out), "--calc-type", "CHARGE+POTENTIAL",
                         "--output-dir", str(tmp_path / "wf"), "--config-file", str(cfg)],
                        capture_output=True, text=True, env=_env(), timeout=120, stdin=subprocess.DEVNULL)
    assert wf.returncode == 0, wf.stdout[-2000:] + wf.stderr[-2000:]
    man = subprocess.run([sys.executable, str(SCRIPT), "--input", str(out), "--calc-type", "CHARGE+POTENTIAL",
                          "--output-dir", str(tmp_path / "man")],
                         input="\n" * 30, capture_output=True, text=True, env=_env(), timeout=120)
    assert man.returncode == 0, man.stdout[-2000:] + man.stderr[-2000:]
    deck = _deck(tmp_path / "wf")
    assert "SCALE" not in deck and deck.count("RANGE") == 2, deck
    assert _deck(tmp_path / "man") == deck
