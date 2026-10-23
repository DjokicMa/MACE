"""E_F-relative ranges of the properties decks need a trustworthy Fermi energy (1.1.6).

An SCF that ends with its spins locked (SPINLOCK, MACE's spin-polarized default for a closed-shell
system: the example suite's diamond) computes no Fermi level: its output prints none, and the
wavefunction holds CRYSTAL's -1.0 Hartree placeholder. CRYSTAL's DOSS window (BMI BMA, with INZB and
IFNB negative; manual p.320) and BOLTZTRA's MURANGE and TDFRANGE are absolute energies, so opt2d3
places them with mace.utils.fermi_level.scf_fermi: the SCF's printed E_F, else a DOSS or BAND run of
the same SCF beside it (same title, opt2d3's names), else none. Without one a DOSS window is not
written (the deck covers all bands) and a TRANSPORT deck relative to E_F is refused; before, the
diamond TRANSPORT deck was written at -2..+2 eV absolute, all inside diamond's gap (σ ~ 0).

The DOSS window around E_F: the default when a step over every band needs more than 5000 points
(Marcus, 2026-10-23), -20..+15 eV around E_F (diamond's whole valence band, about 21 eV wide), on every route; --doss-window LO HI or all; Setup's
Window around E_F row. Its bounds move outward into band gaps when a BAND run of the SCF is beside it; a 1D
material takes its bands by index instead (tests/test_doss_dimensionality.py).
A probe on the HPCC (diamond, 2026-10-23) found the window's DOS equal to the all-bands DOS inside
it (bands that cross BMI or BMA are included: FROM BAND 3 TO BAND 9 for -15..+10 eV).
"""
import builtins
import json
import lzma
import os
import shutil
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
DIAMOND_EF = -0.077970          # the top of the occupied bands of the suite's diamond BAND run (= its DOSS run's)


def _put(src: Path, dst: Path):
    xz = src.with_name(src.name + ".xz")
    dst.write_bytes(lzma.decompress(xz.read_bytes()) if xz.is_file() else src.read_bytes())


def _diamond(root: Path, band=True) -> Path:
    """The suite's spin-locked diamond SP (and its BAND run beside it) in a run folder; its .f9 a stand-in."""
    root.mkdir(parents=True)
    _put(SUITE / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out", root / "diamond_sp.out")
    (root / "diamond_sp.f9").write_text("wavefunction placeholder\n")
    if band:
        for n in ("diamond_band.out", "diamond_band.BAND.DAT"):
            _put(SUITE / "2-dense-3d" / "diamond" / "BAND" / n, root / n)
    return root / "diamond_sp.out"


def _opt2d3(parent, calc, *args, answers=""):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    out_dir = parent.parent / f"o{len(list(parent.parent.glob('o*')))}"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", str(parent), "-t", calc,
                        "-o", str(out_dir), *args], cwd=str(parent.parent), input=answers, capture_output=True,
                       text=True, env=env, timeout=180)
    decks = list(out_dir.glob("*.d3"))
    return r, (decks[0].read_text().splitlines() if decks else None)


def _record(lines, key, n=1):
    i = lines.index(key)
    return lines[i + 1:i + 1 + n]


# ------------------------------------------------------------------ the shared E_F
def test_a_spin_locked_scf_takes_e_f_from_its_band_run(tmp_path):
    from mace.utils import fermi_level as F
    out = _diamond(tmp_path / "run")
    f = F.scf_fermi(out, None)
    assert f.value_au == pytest.approx(DIAMOND_EF, abs=1e-6) and f.derived and "diamond_band.BAND.DAT" in f.note
    assert F.scf_fermi(out, -1.0).value_au == pytest.approx(DIAMOND_EF, abs=1e-6)     # the placeholder: not used
    assert F.scf_fermi(out, -0.2).value_au == -0.2                                      # a printed one is the SCF's
    # a DOSS run of it comes first (its NEWK computes E_F)
    (tmp_path / "run" / "diamond_doss.DOSS.DAT").write_text("# EFERMI (HARTREE)  -0.07800\n")
    shutil.copy(tmp_path / "run" / "diamond_band.out", tmp_path / "run" / "diamond_doss.out")
    assert F.scf_fermi(out, None).value_au == -0.078


def test_no_band_run_or_another_scfs_band_run_gives_no_e_f(tmp_path):
    from mace.utils import fermi_level as F
    out = _diamond(tmp_path / "alone", band=False)
    f = F.scf_fermi(out, -1.0)
    assert f.value_au is None and "SPINLOCK" in f.note and "run BAND or DOSS first" in f.note
    # a BAND run of another SCF (another title) beside it is not used
    other = _diamond(tmp_path / "other")
    band_out = other.with_name("diamond_band.out")
    band_out.write_text(band_out.read_text().replace("diamond_opt_sp_B3LYP-D3_optimized", "something_else", 1))
    assert F.scf_fermi(other, None).value_au is None
    # a run folder layout: <entry>/SP beside <entry>/BAND (the example suite, a workflow's step folders)
    ent = tmp_path / "entry"
    sp = _diamond(ent / "SP", band=False)
    (ent / "BAND").mkdir()
    for n in ("diamond_band.out", "diamond_band.BAND.DAT"):
        _put(SUITE / "2-dense-3d" / "diamond" / "BAND" / n, ent / "BAND" / n)
    assert F.scf_fermi(sp, None).value_au == pytest.approx(DIAMOND_EF, abs=1e-6)


# ------------------------------------------------------------------ the DOSS window
def test_cap_and_window_constants():
    assert C.DOSS_MAX_NPTS == 99997 and C.DOSS_NPTS_RANGE == (1000, 99997)
    assert C.DOSS_WINDOW_OFFER_NPTS == 5000 and C.DOSS_DEFAULT_WINDOW_EV == (-20.0, 15.0)
    assert C.doss_window_offer(890.0, 44500, 0.02) == ("This DOS needs 44500 points for a 0.02 eV step over all "
                                                       "bands (~890 eV, including core levels)")
    assert C.doss_window_offer(890.0, 4450, 0.2) is None and C.doss_window_offer(None, 99997, 0.02) is None


def _window(lines):
    npt, bmi_bma = _record(lines, "DOSS", 2)
    return npt.split()[1:4], [float(x) for x in bmi_bma.split()]


def test_the_window_is_the_default_on_every_route(tmp_path):
    """Above 5000 points over every band (diamond at Standard: 39552) the deck covers E_F -20..+15 eV: Enter
    (the [Y/n] question), --non-interactive and a settings file write the same deck (Marcus, 2026-10-23). With
    the BAND run of the SCF beside it, the lower bound inside diamond's valence band (from E_F-23.16 eV) moves
    1 eV below it, into the gap above the core; the upper one, inside the conduction bands' continuum (no gap
    within 5 eV), is kept. --doss-window -20 15 is written as given."""
    sp = _diamond(tmp_path / "run")
    r, enter = _opt2d3(sp, "DOSS", answers="\n" * 40)
    out = r.stdout
    assert "This DOS needs 39552 points for a 0.02 eV step over all bands (~791 eV, including core levels)." in out
    assert "Use a window around E_F (-20 to +15 eV) instead? [Y/n]: " in out
    assert "The window's bounds moved outward into band gaps" in out + r.stderr
    from mace.utils.fermi_level import sibling_band_extents
    bottom = min(lo for lo, _ in sibling_band_extents(sp)[2][2:])            # band 3: the valence band's bottom
    head, (bmi, bma) = _window(enter)
    assert head == ["1959", "-1", "-1"]                              # 39.2 eV / 0.02 eV, BMI/BMA follow
    assert bmi == pytest.approx(bottom - 1 / HA, abs=1e-9) and bmi < DIAMOND_EF - 20 / HA
    assert bma == pytest.approx(DIAMOND_EF + 15 / HA, abs=1e-9)
    r, quiet = _opt2d3(sp, "DOSS", "--non-interactive")
    assert quiet == enter and "--doss-window all for every band" in r.stdout + r.stderr
    cfg = tmp_path / "doss.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "DOSS",
                               "configuration": {"calculation_type": "DOSS", "projection_type": 3,
                                                 "project_orbital_types": True, "n_points": "auto"}}))
    assert _opt2d3(sp, "DOSS", "--config-file", str(cfg))[1] == enter
    _, given = _opt2d3(sp, "DOSS", "--doss-window", "-20", "15", "--non-interactive")
    head, (bmi, bma) = _window(given)
    assert head == ["1750", "-1", "-1"] and bmi == pytest.approx(DIAMOND_EF - 20 / HA, abs=1e-9)


def test_other_bounds_and_the_opt_out(tmp_path):
    sp = _diamond(tmp_path / "run")
    # projection, range: Enter; resolution: Standard; the window: Enter; its bounds: -15 and 10
    _, lines = _opt2d3(sp, "DOSS", answers="\n\n\n\n-15\n10\n" + "\n" * 40)
    head, (bmi, bma) = _window(lines)
    assert head == ["1250", "-1", "-1"] and bma == pytest.approx(DIAMOND_EF + 10 / HA, abs=1e-9)
    assert _opt2d3(sp, "DOSS", "--doss-window", "-15", "10", "--non-interactive")[1] == lines
    # No, or --doss-window all: every band
    r, lines = _opt2d3(sp, "DOSS", answers="\n\n\nn\n" + "\n" * 40)
    assert _record(lines, "DOSS")[0].split()[1:4] == ["39552", "1", "36"]
    assert _opt2d3(sp, "DOSS", "--doss-window", "all", "--non-interactive")[1] == lines
    r, _ = _opt2d3(sp, "DOSS", "--doss-window", "5", "10")
    assert r.returncode == 2 and "LO < 0 < HI" in r.stderr


def test_no_window_around_a_placeholder(tmp_path):
    sp = _diamond(tmp_path / "alone", band=False)
    for args, answers in ((("--non-interactive",), ""), (("--doss-window", "-15", "10", "--non-interactive"), ""),
                          ((), "\n" * 40)):
        r, lines = _opt2d3(sp, "DOSS", *args, answers=answers)
        assert _record(lines, "DOSS")[0].split()[1:4] == ["39552", "1", "36"], args   # never around -1.0
        assert "No window around E_F can be written" in r.stdout + r.stderr or \
            "no energy window around E_F" in r.stdout + r.stderr


def test_planner_asks_the_window(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner
    for answers, want in (([], {"energy_resolution": "standard", "n_points": "auto"}),
                          (["2", "", "", "10"], {"energy_resolution": "standard", "n_points": "auto",
                                                 "energy_window": pytest.approx([-20 / HA, 10 / HA])}),
                          (["2", "n"], {"energy_resolution": "standard", "n_points": "auto", "all_bands": True})):
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda prompt="", it=it: next(it, ""))
        assert WorkflowPlanner(work_dir=tmp_path)._ask_doss_resolution() == want, answers


# ------------------------------------------------------------------ TRANSPORT
def _transport_config(tmp_path):
    from mace.workflow.common.constants import opt2d3_default_transport_config
    cfg = tmp_path / "transport.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "TRANSPORT",
                               "configuration": opt2d3_default_transport_config()}))
    return cfg


def test_transport_ranges_are_around_e_f(tmp_path):
    sp = _diamond(tmp_path / "run")
    r, lines = _opt2d3(sp, "TRANSPORT", "--config-file", str(_transport_config(tmp_path)))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    ef = DIAMOND_EF * HA
    mu = [float(x) for x in _record(lines, "MURANGE")[0].split()]
    tdf = [float(x) for x in _record(lines, "TDFRANGE")[0].split()]
    assert mu == pytest.approx([ef - 2, ef + 2, 0.01], abs=1e-6)
    assert tdf == pytest.approx([ef - 5, ef + 5, 0.01], abs=1e-6)       # TDFRANGE is absolute too


def test_transport_without_e_f_is_refused_or_asks_for_absolute_energies(tmp_path):
    sp = _diamond(tmp_path / "alone", band=False)
    r, lines = _opt2d3(sp, "TRANSPORT", "--config-file", str(_transport_config(tmp_path)))
    assert lines is None and "no TRANSPORT deck" in r.stdout + r.stderr and "SPINLOCK" in r.stdout + r.stderr
    # interactively: absolute energies, which have no default
    r, lines = _opt2d3(sp, "TRANSPORT", answers="\n\n\n-4.1\n-0.1\n\n" + "\n" * 30)
    assert "Minimum μ (eV, absolute): " in r.stdout
    assert [float(x) for x in _record(lines, "MURANGE")[0].split()] == [-4.1, -0.1, 0.01]


@pytest.mark.parametrize("entry", ["2-dense-3d/diamond", "1-porous/SOD", "3-slab-2d/hBN", "3-slab-2d/graphene",
                                   "4-polymer-1d/polyacetylene"])
def test_the_suites_transport_runs_cover_their_chemical_potentials(entry):
    """Each example TRANSPORT run's TDF covers its μ window, and σ is not zero in it. With TDFRANGE written as
    absolute -5..+5 eV, SOD's and h-BN's μ windows (E_F ±2 eV, E_F -6.75 and -5.63 eV) lay below the TDF, and
    their σ was zero at every point."""
    name = entry.split("/")[1]
    run = SUITE / entry / "TRANSPORT"
    lines = (run / f"{name}_transport.d3").read_text().splitlines()
    mu_lo, mu_hi, _ = map(float, _record(lines, "MURANGE")[0].split())
    tdf_lo, tdf_hi, _ = map(float, _record(lines, "TDFRANGE")[0].split())
    assert tdf_lo < mu_lo - 2 and tdf_hi > mu_hi + 2
    sigma = lzma.decompress((run / f"{name}_transport.SIGMA.DAT.xz").read_bytes()).decode()
    vals = [abs(float(x)) for ln in sigma.splitlines() if ln.strip() and not ln.startswith("#")
            for x in ln.split()[3:4]]
    assert max(vals) > 1.0
