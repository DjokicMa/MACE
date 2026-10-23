"""DOSS defaults by periodic dimensionality (1.1.6).

CRYSTAL needs a DOSS energy window's bounds (BMI, BMA) in band gaps: on the MSU HPCC (CRYSTAL23,
2026-10-23) the example suite's 1D polyacetylene at E_F -20..+15 eV, whose BMA fell inside a band,
stopped with "double free or corruption" (with NPRO 0 too), while a BMA in a gap and the same bands by
index ran. Above 5000 points over every band:

* 3D and 2D: the window E_F -20..+15 eV, its bounds moved outward (never inward) into the nearest gaps
  when a BAND run of the same SCF is beside it (at most 5 eV: a 3D material's conduction bands are a
  continuum), the plain window without one;
* 1D: the bands by index (INZB, IFNB), from the first band above the core (CRYSTAL's core count, plus an
  ECP atom's semicore orbitals) to the conduction bands: --doss-conduction-bands N, opt2d3's question,
  Setup's Conduction bands row, the planner's question;
* after the run, a band-index DOS that stops below E_F + 15 eV gets a note (mace analyze, the DOS plot).
"""
import builtins
import json
import lzma
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_config as C  # noqa: E402

pytest.importorskip("numpy")

SUITE = REPO_ROOT / "examples" / "suite"
HA = 27.211386245988
ENTRIES = {"SOD": "1-porous", "diamond": "2-dense-3d", "hBN": "3-slab-2d", "graphene": "3-slab-2d",
           "polyacetylene": "4-polymer-1d"}


def _put(src: Path, dst: Path):
    xz = src.with_name(src.name + ".xz")
    dst.write_bytes(lzma.decompress(xz.read_bytes()) if xz.is_file() else src.read_bytes())


def _run_folder(root: Path, name: str, band=True) -> Path:
    """The suite's SP of ``name`` (and its BAND run beside it) in a run folder; its .f9 a stand-in."""
    root.mkdir(parents=True)
    entry = SUITE / ENTRIES[name] / name
    _put(entry / "SP" / f"{name}_sp.out", root / f"{name}_sp.out")
    (root / f"{name}_sp.f9").write_text("wavefunction placeholder\n")
    if band:
        for n in (f"{name}_band.out", f"{name}_band.BAND.DAT"):
            _put(entry / "BAND" / n, root / n)
    return root / f"{name}_sp.out"


def _opt2d3(parent, *args, answers=""):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    out_dir = parent.parent / f"o{len(list(parent.parent.glob('o*')))}"
    flags = () if answers else ("--non-interactive",)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", str(parent), "-t", "DOSS",
                        "-o", str(out_dir), *flags, *args], cwd=str(parent.parent), input=answers,
                       capture_output=True, text=True, env=env, timeout=180)
    decks = list(out_dir.glob("*_doss.d3"))
    if not decks:
        return r, None, None
    lines = decks[0].read_text().splitlines()
    rec = lines[lines.index("DOSS") + 1].split()
    window = [float(x) for x in lines[lines.index("DOSS") + 2].split()] if int(rec[2]) < 0 else None
    return r, rec[1:4], window


# ------------------------------------------------------------------ the counts and the band range
def test_crystal_core_counts_agree_with_its_outputs():
    """CRYSTAL's "CORE ELECTRONS PER CELL": the noble-gas core, plus a filled d shell after a transition series
    (all-electron Br: 28) and 4f from Hf on; an ECP atom's semicore orbitals are added to it."""
    assert [C.crystal_core_electrons(z) for z in (1, 6, 14, 22, 35, 47, 82)] == [0, 2, 10, 18, 28, 36, 78]
    assert [C.valence_shell_orbitals(z) for z in (1, 6, 14, 22, 35, 47, 64)] == [1, 4, 4, 9, 4, 9, 16]
    ag = ("   1 T 247 AG    0.0 0.0 0.0\n ATOMIC NUMBER  47, NUCLEAR CHARGE 19.000, GIVEN IN INPUT\n"
          " NUMBER OF AO                35\n N. OF ELECTRONS PER CELL    19\n CORE ELECTRONS PER CELL      0\n")
    counts = C.doss_band_counts(ag)
    assert (counts["core"], counts["semicore"], counts["occupied"]) == (4, 4, 10)      # Ag 4s 4p: core-like
    assert C.doss_band_index_range(counts)[0] == 5
    assert C.doss_band_counts("no counts here") is None


@pytest.mark.parametrize("name, counts, bands", [("polyacetylene", (2, 7, 34), (3, 12)), ("diamond", (2, 6, 36), (3, 10)),
                                                 ("hBN", (2, 6, 30), (3, 10)), ("graphene", (2, 6, 30), (3, 10)),
                                                 ("SOD", (42, 90, 288), (43, 138))])
def test_band_index_range_of_the_suite(name, counts, bands):
    """valence + as many conduction bands: what reaches E_F + 15 eV in every suite material (BWIDTH, HPCC
    2026-10-23: bands 12, 10, 10, 10 and 111 hold it)."""
    text = lzma.decompress((SUITE / ENTRIES[name] / name / "SP" / f"{name}_sp.out.xz").read_bytes()).decode()
    c = C.doss_band_counts(text)
    assert (c["core"], c["occupied"], c["n_ao"]) == counts
    assert C.doss_band_index_range(c) == bands
    assert C.doss_band_index_range(c, 8) == (bands[0], min(counts[1] + 8, counts[2]))
    assert C.doss_band_index_range(c, 10 ** 6)[1] == counts[2]                            # never past the last band


def test_snapping_moves_bounds_outward_into_gaps():
    ev = 1 / HA
    bands = [(-30 * ev, -25 * ev), (-22 * ev, -1 * ev), (2 * ev, 13 * ev), (14 * ev, 40 * ev)]
    # -20 inside the valence bands: 1 eV below their bottom (the gap below is 3 eV); +15 inside a continuum
    # reaching +40 (no gap within 5 eV): kept
    lo, hi, why = C.doss_snap_window(-20 * ev, 15 * ev, bands)
    assert lo == pytest.approx(-23 * ev) and hi == pytest.approx(15 * ev)
    assert "moved down to E_F-23.00 eV" in why[0] and "kept" in why[1]
    # +12 inside 2..13, and 13..14 is a 1 eV gap: half way into it; never inward
    lo, hi, why = C.doss_snap_window(-24 * ev, 12 * ev, bands)
    assert lo == pytest.approx(-24 * ev) and hi == pytest.approx(13.5 * ev) and len(why) == 1
    # bounds in gaps: as given
    assert C.doss_snap_window(-23.5 * ev, 1 * ev, bands) == (-23.5 * ev, 1 * ev, [])
    # gaps narrower than twice the margin are not gaps
    lo, _, _ = C.doss_snap_window(-10 * ev, 1 * ev, [(-12 * ev, -8 * ev), (-7.9 * ev, -3 * ev)])
    assert lo == pytest.approx(-13 * ev)


# ------------------------------------------------------------------ opt2d3's decks
def test_1d_takes_its_bands_by_index_on_every_route(tmp_path):
    """polyacetylene: bands 3-12 (2 core, 7 occupied); NPT from the BAND run's span of them (36.1 eV + 1 eV),
    else the 50 eV bound; Enter, --non-interactive and a settings file agree."""
    sp = _run_folder(tmp_path / "run", "polyacetylene")
    r, rec, window = _opt2d3(sp)
    assert rec == ["1855", "3", "12"] and window is None, (r.stdout + r.stderr)[-2000:]
    assert "the deck covers bands 3 to 12 instead (a 1D material: by band index" in (r.stdout + r.stderr)
    assert "reach E_F+16.6 eV" in (r.stdout + r.stderr)
    r, enter, _ = _opt2d3(sp, answers="\n" * 40)
    assert enter == rec and "Use bands 3 to 12 by index instead? [Y/n]: " in (r.stdout + r.stderr)
    assert "Conduction bands above the highest occupied band [5]: " in (r.stdout + r.stderr)
    cfg = tmp_path / "doss.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "DOSS",
                               "configuration": {"calculation_type": "DOSS", "projection_type": 3,
                                                 "project_orbital_types": True, "n_points": "auto"}}))
    assert _opt2d3(sp, "--config-file", str(cfg))[1] == rec
    alone = _run_folder(tmp_path / "solo" / "alone", "polyacetylene", band=False)     # no BAND run beside it
    assert _opt2d3(alone)[1] == [str(math.ceil(C.DOSS_BAND_INDEX_SPAN_EV / 0.02)), "3", "12"]


def test_1d_conduction_bands_and_the_opt_outs(tmp_path):
    sp = _run_folder(tmp_path / "run", "polyacetylene")
    assert _opt2d3(sp, "--doss-conduction-bands", "8")[1][1:] == ["3", "15"]
    assert _opt2d3(sp, answers="\n\n\n\n8\n" + "\n" * 40)[1][1:] == ["3", "15"]          # opt2d3's question
    cfg = tmp_path / "doss.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "DOSS",
                               "configuration": {"calculation_type": "DOSS", "projection_type": 3,
                                                 "n_points": "auto", "conduction_bands": 8}}))
    assert _opt2d3(sp, "--config-file", str(cfg))[1][1:] == ["3", "15"]
    # every band: No, or --doss-window all
    assert _opt2d3(sp, "--doss-window", "all")[1][1:] == ["1", "34"]
    assert _opt2d3(sp, answers="\n\n\nn\n" + "\n" * 40)[1][1:] == ["1", "34"]
    # a window as given, with the warning: its upper bound is inside a band (the crash)
    r, rec, window = _opt2d3(sp, "--doss-window", "-20", "15")
    assert rec[1:] == ["-1", "-1"] and "is inside the bands" in (r.stdout + r.stderr) and "double free" in (r.stdout + r.stderr)
    r, _, _ = _opt2d3(sp, "--doss-conduction-bands", "0")
    assert r.returncode == 2 and "--doss-conduction-bands" in r.stderr


@pytest.mark.parametrize("name", ["hBN", "SOD"])
def test_3d_2d_window_moves_into_gaps_with_a_band_run(tmp_path, name):
    sp = _run_folder(tmp_path / "run", name)
    r, rec, (bmi, bma) = _opt2d3(sp)
    from mace.utils.fermi_level import sibling_band_extents
    extents = sibling_band_extents(sp)[2]
    assert "The window's bounds moved outward into band gaps" in (r.stdout + r.stderr)
    assert all(not (lo < bmi < hi or lo < bma < hi) for lo, hi in extents if hi - lo > 0) or "kept" in (r.stdout + r.stderr)
    alone = _run_folder(tmp_path / "solo" / "alone", name, band=False)
    r, rec2, (bmi2, bma2) = _opt2d3(alone)
    assert rec2 == ["1750", "-1", "-1"] and "not checked against the bands" in (r.stdout + r.stderr)
    assert bmi < bmi2 and bma >= bma2 and int(rec[0]) > 1750                             # never narrowed
    assert "--doss-conduction-bands" not in (r.stdout + r.stderr)
    r, rec3, _ = _opt2d3(alone, "--doss-conduction-bands", "8")
    assert rec3 == rec2 and "does not use it" in (r.stdout + r.stderr)


# ------------------------------------------------------------------ after the run
def _band_index_run(folder: Path, top_ev: float, first=3, last=12) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "x_doss.d3").write_text(f"NEWK\n18 36\n1 0\nDOSS\n0 100 {first} {last} 2 14 0\nEND\n")
    rows = "\n".join(f" {e / HA:.5E}  1.0000E+00" for e in (-20.0, 0.0, top_ev))
    dat = folder / "x_doss.DOSS.DAT"
    dat.write_text(f"# NEPTS     3 NPROJ     1 NSPIN     1\n#\n@ XAXIS LABEL \"E-EFERMI (HARTREE)\"\n{rows}\n")
    return dat


def test_the_reach_note(tmp_path):
    from mace.utils import doss_reach
    low = _band_index_run(tmp_path / "low", 9.0)
    note = doss_reach.reach_note(low)
    assert "stops at E_F+9.0 eV, below E_F+15 eV" in note and "--doss-conduction-bands" in note
    assert doss_reach.reach_note(_band_index_run(tmp_path / "high", 16.6)) is None
    win = _band_index_run(tmp_path / "win", 9.0, -1, -1)
    assert doss_reach.reach_note(win) is None                                             # a window: as asked
    data = doss_reach.add_note({"notes": []}, low)
    assert data["reach_note"] == note and data["notes"] == [note]


def test_the_reach_note_is_stored_and_drawn(tmp_path):
    """DB <-> GUI parity: the extractor's doss_reach_note (a property key) and the GUI DOS plot's note."""
    from mace.utils.property_keys import key
    assert key("doss_reach_note").category == "density_of_states"
    entry = SUITE / "4-polymer-1d" / "polyacetylene"
    folder = tmp_path / "DOSS"
    dat = _band_index_run(folder, 9.0)
    (folder / "x_doss.out").write_text(lzma.decompress((entry / "SP" / "polyacetylene_sp.out.xz").read_bytes()).decode()
                                       + "\n TOTAL AND PROJECTED DENSITY OF STATES - FOURIER LEGENDRE METHOD\n")
    from mace.utils.property_extractor import CrystalPropertyExtractor
    props = CrystalPropertyExtractor()._extract_dos_properties((folder / "x_doss.out").read_text(),
                                                                folder / "x_doss.out")
    assert props["doss_reach_note"].startswith("the DOS stops at E_F+9.0 eV")
    from mace.gui import results
    assert results.dos_data(dat)["reach_note"] == props["doss_reach_note"]


# ------------------------------------------------------------------ the workflow planner
def test_planner_asks_the_conduction_bands(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner
    for answers, want in (([], {}), (["2", "", "", "", "8"], {"conduction_bands": 8}),
                          (["2", "", "", "", "x", "6"], {"conduction_bands": 6})):
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda prompt="", it=it: next(it, ""))
        got = WorkflowPlanner(work_dir=tmp_path)._ask_doss_resolution()
        assert got == {"energy_resolution": "standard", "n_points": "auto", **want}, answers
    assert C.validate_d3_config({"calculation_type": "DOSS", "projection_type": 3, "n_points": "auto",
                                 "conduction_bands": 0})[0] is False
