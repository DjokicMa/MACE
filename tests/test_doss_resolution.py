"""DOSS energy resolution (1.1.6): one resolver, Crystal_d3/d3_config.doss_npts, for every route.

The setting is the energy step of the DOS grid: Screening 0.2 eV, Standard 0.02 eV (the default,
DOSS_DEFAULT_STEP_EV), Fine 0.005 eV, another step, or a number of points written as given (a
screening run of many structures may want 1000). With a step the deck gets ceil(span / step)
points within 1000-99997 (CRYSTAL23's manual sets no limit for NPT, DOSS p.320; DOSS.DAT's header
prints NPT + 2 in five digits, so 99997 is the real maximum). An all-bands
range's span is first printed by the DOSS run, so it is estimated from the atoms (the deepest core
level to ~20 Hartree); the example suite's runs check the estimate. A number of points is honored up
to 99997 (DOSS.DAT's header prints NPT + 2 in five digits). A coarse grid (over 0.05 eV) is
warned about: the DOS effective masses and band-edge character need a finer one.

Routes: mace opt2d3 (Enter, the menu, --doss-step / --doss-points, --non-interactive, a settings
file), the workflow planner, and Setup (tests/gui/test_setup_doss_resolution.py).
"""
import builtins
import json
import lzma
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_config as C  # noqa: E402

SUITE = REPO_ROOT / "examples" / "suite"
# the suite's all-bands DOSS runs: (entry folder, stem)
RUNS = [("1-porous/SOD", "SOD"), ("2-dense-3d/diamond", "diamond"), ("3-slab-2d/hBN", "hBN"),
        ("3-slab-2d/graphene", "graphene"), ("4-polymer-1d/polyacetylene", "polyacetylene")]


def _text(path):
    xz = path.with_name(path.name + ".xz")
    return lzma.decompress(xz.read_bytes()).decode() if xz.is_file() else path.read_text()


# ------------------------------------------------------------------ the resolver
def test_the_standard_step_is_the_one_default():
    assert C.DOSS_DEFAULT_STEP_EV == 0.02
    assert C.DOSS_STEP_PRESETS == {"screening": 0.2, "standard": C.DOSS_DEFAULT_STEP_EV, "fine": 0.005}
    assert C.doss_step_ev({}) == C.doss_step_ev({"energy_resolution": "standard"}) == C.DOSS_DEFAULT_STEP_EV
    assert C.doss_step_ev({"energy_resolution": "screening"}) == 0.2
    assert C.doss_step_ev({"energy_resolution": "step", "energy_step_ev": 0.1}) == 0.1
    # a preset names its step: a stale energy_step_ev beside it does not count
    assert C.doss_step_ev({"energy_resolution": "fine", "energy_step_ev": 0.3}) == 0.005


def test_points_follow_the_step_within_the_range():
    assert C.doss_npts(100.0) == 5000                    # 100 eV / 0.02 eV
    assert C.doss_npts(100.0, step=0.2) == 1000          # 500, raised to the least
    assert C.doss_npts(10.0) == 1000
    assert C.doss_npts(890.0) == 44500
    assert C.doss_npts(890.0, step=0.005) == 99997       # 178000, the most
    assert C.doss_npts(890.0, step=0.2) == 4450
    assert C.doss_npts(30.0, step=0.015) == 2000         # exact multiples stay exact
    assert C.doss_npts(None) == C.doss_npts(float("nan")) == C.doss_npts(0) == 99997   # no span: the most
    assert C.DOSS_NPTS_RANGE == (1000, C.DOSS_MAX_NPTS) == (1000, 99997)


def test_explicit_points_are_honored_as_given():
    assert C.doss_npts(890.0, 1000) == 1000              # screening: however coarse
    assert C.doss_npts(890.0, 500) == 500                # below the least of the step rule too
    assert C.doss_npts(5.0, 60000) == 60000              # above the most
    assert C.doss_npts(890.0, "auto") == 44500
    assert C.doss_explicit_points({"n_points": 1000}) == 1000
    assert C.doss_explicit_points({"n_points": "auto"}) is None and C.doss_explicit_points({}) is None


def test_a_coarse_grid_is_warned_about_once_with_its_reason():
    line, warn = C.doss_grid_note(890.0, 1000, explicit=True, estimated=True)
    assert line == "1000 energy points, ~0.89 eV apart over ~890 eV"
    assert warn.startswith("the 1000 points given give a ~0.89 eV energy step")
    assert "DOS effective masses and band-edge character" in warn and "0.05 eV" in warn
    _, warn = C.doss_grid_note(9000.0, 99997, explicit=False, estimated=True)
    assert warn.startswith("99997 points, the most MACE writes, give")
    _, warn = C.doss_grid_note(890.0, 4450, explicit=False, estimated=False)
    assert warn.startswith("the chosen step gives a 0.2 eV")
    assert C.doss_grid_note(300.0, 20000, explicit=False, estimated=True)[1] is None
    assert C.doss_grid_note(None, 20000, explicit=False, estimated=True) == ("20000 energy points", None)


# the all-bands DOSS runs of the suite's SPs (HPCC, 2026-10-23, CRYSTAL23): "FROM BAND 1 TO BAND N ENERGY
# RANGE lo hi" (Hartree). The suite now ships window runs, so the spans are kept here.
ALL_BANDS_RANGE_HA = {"SOD": (-66.210, 25.561), "diamond": (-10.091, 22.619), "hBN": (-14.313, 19.434),
                      "graphene": (-10.219, 15.740), "polyacetylene": (-10.183, 15.310)}


@pytest.mark.parametrize("folder, stem", RUNS)
def test_the_span_estimate_is_close_to_the_runs(folder, stem):
    """The estimate from the SP's atoms against the span an all-bands DOSS run of it printed."""
    sp = _text(SUITE / folder / "SP" / f"{stem}_sp.out")
    lo, hi = ALL_BANDS_RANGE_HA[stem]
    real = (hi - lo) * 27.211386245988
    est = C.doss_span_estimate_ev(C.doss_atomic_numbers(sp))
    assert est == pytest.approx(real, rel=0.15), (est, real)


def test_atoms_and_ecp_atoms_from_the_geometry_listing():
    text = ("      1 T 247 AG   -2.5E-01 -2.5E-01 -2.5E-01\n      2 T  35 BR    0.0E+00  0.0E+00  0.0E+00\n"
            "      1 T 247 AG   -2.5E-01 -2.5E-01 -2.5E-01\n")
    assert C.doss_atomic_numbers(text) == [247, 35]      # the first listing only
    assert C.doss_core_level_ha(247) == C.DOSS_ECP_BOTTOM_HA
    assert C.doss_core_level_ha(6) == pytest.approx(-9.1, abs=0.1)     # diamond's band 1: -10.09 Hartree
    assert C.doss_core_level_ha(106) == C.doss_core_level_ha(6)        # CRYSTAL's 1xx: carbon, another basis
    assert C.doss_span_estimate_ev([]) is None
    win = {"energy_window": (-0.3675, 0.7350)}           # -10..+20 eV
    assert C.doss_span(win, [6]) == (pytest.approx(30.0, abs=0.01), False)


def test_settings_are_validated():
    ok, _ = C.validate_d3_config({"calculation_type": "DOSS", "projection_type": 3, "n_points": "auto",
                                  "energy_resolution": "fine"})
    assert ok
    # DOSS.DAT's header prints NPT + 2 in five digits: 200000 points wrote "NEPTS *****" (HPCC)
    assert C.DOSS_MAX_NPTS == 99997
    for bad in ({"n_points": 1}, {"n_points": 100000}, {"n_points": "many"}, {"energy_resolution": "coarse"}, {"energy_step_ev": 0}):
        ok, errors = C.validate_d3_config({"calculation_type": "DOSS", "projection_type": 3, "n_points": "auto",
                                           **bad})
        assert not ok, bad


# ------------------------------------------------------------------ mace opt2d3
@pytest.fixture()
def parent(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    out = d / "diamond_sp.out"
    out.write_text(_text(SUITE / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out"))
    (d / "diamond_sp.f9").write_text("wavefunction placeholder\n")
    return out


def _opt2d3(parent, *args, answers=""):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    out_dir = parent.parent / f"o{len(list(parent.parent.glob('o*')))}"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", str(parent), "-t", "DOSS",
                        "-o", str(out_dir), *args], cwd=str(parent.parent), input=answers, capture_output=True,
                       text=True, env=env, timeout=180)
    decks = list(out_dir.glob("*_doss.d3"))
    npt = None
    if decks:
        lines = decks[0].read_text().splitlines()
        npt = int(lines[lines.index("DOSS") + 1].split()[1])
    return r, npt


def test_enter_is_standard_and_the_menu_shows_each_choices_grid(parent):
    r, npt = _opt2d3(parent, answers="\n" * 40)
    assert r.returncode == 0 and npt == 39552, r.stdout[-1500:]
    menu = r.stdout[r.stdout.index("Energy resolution (the energy step"):]
    assert "1: Screening - 0.2 eV step (3956 points, ~0.2 eV apart)" in menu
    assert "2: Standard - 0.02 eV step (39552 points, ~0.02 eV apart)" in menu
    assert "3: Fine - 0.005 eV step (99997 points" in menu
    assert "Select energy resolution (1-5) [2]: " in menu
    assert "Energy grid: 39552 energy points" in r.stdout
    # no window (the default above 5000 points) without a trustworthy E_F: the SP's spins are locked and no
    # BAND or DOSS is beside it
    assert "This DOS needs 39552 points for a 0.02 eV step over all bands (~791 eV, including core levels)." in menu
    assert "No window around E_F can be written" in menu and "SPINLOCK" in menu


@pytest.mark.parametrize("answer, extra, want", [("1", "", 3956), ("4", "0.1\n", 7911), ("5", "1000\n", 1000)])
def test_the_menus_other_answers(parent, answer, extra, want):
    # projection, range (Enter), the resolution and its follow-up, then Enter
    _, npt = _opt2d3(parent, answers="\n\n" + answer + "\n" + extra + "\n" * 40)
    assert npt == want


def test_flags_and_non_interactive(parent):
    r, npt = _opt2d3(parent, "--non-interactive", "--doss-points", "1000")
    assert r.returncode == 0 and npt == 1000
    assert "the 1000 points given give a ~0.79 eV energy step" in r.stdout + r.stderr
    r, npt = _opt2d3(parent, "--non-interactive")
    # warned; no window can be offered without a trustworthy E_F (tests/test_fermi_relative_ranges.py)
    assert npt == 39552 and "No window around E_F can be written" in r.stdout + r.stderr
    assert _opt2d3(parent, "--non-interactive", "--doss-step", "screening")[1] == 3956
    assert _opt2d3(parent, "--doss-step", "0.1", answers="\n" * 40)[1] == 7911
    r, npt = _opt2d3(parent, "--doss-step", "0.1", "--doss-points", "1000")
    assert r.returncode == 2 and npt is None and "not both" in r.stderr
    r, _ = _opt2d3(parent, "--doss-step", "-1")
    assert r.returncode == 2 and "--doss-step" in r.stderr
    r, _ = _opt2d3(parent, "--doss-points", "100000")
    assert r.returncode == 2 and "five digits" in r.stderr


def test_a_settings_file_keeps_its_points_and_a_flag_overrides_it(parent, tmp_path):
    cfg = tmp_path / "doss.json"
    base = {"calculation_type": "DOSS", "projection_type": 3, "project_orbital_types": True}
    for conf, want in (({**base, "n_points": 1000}, 1000),                 # a preview-era file: as given
                       ({**base, "n_points": "auto", "energy_resolution": "screening"}, 3956),
                       ({**base, "n_points": "auto"}, 39552)):
        cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "DOSS",
                                   "configuration": conf}))
        assert _opt2d3(parent, "--config-file", str(cfg))[1] == want, conf
    assert _opt2d3(parent, "--config-file", str(cfg), "--doss-points", "1500")[1] == 1500


# ------------------------------------------------------------------ the workflow planner
def _answers(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it, ""))


@pytest.mark.parametrize("answers, want", [([], {"energy_resolution": "standard", "n_points": "auto"}),
                                           (["1"], {"energy_resolution": "screening", "n_points": "auto"}),
                                           (["4", "0.05"], {"energy_resolution": "step", "energy_step_ev": 0.05,
                                                            "n_points": "auto"}),
                                           (["5", "1000"], {"energy_resolution": "points", "n_points": 1000})])
def test_planner_asks_opt2d3s_resolution(tmp_path, monkeypatch, answers, want):
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner(work_dir=tmp_path)
    _answers(monkeypatch, answers)
    assert planner._ask_doss_resolution() == want


def test_every_workflow_route_inherits_the_standard_step(tmp_path, monkeypatch):
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.planner import WorkflowPlanner
    from mace.workflow.common.constants import opt2d3_default_doss_config
    _answers(monkeypatch, [])
    planner = WorkflowPlanner(work_dir=tmp_path)
    for conf in (WorkflowEngine.__new__(WorkflowEngine)._get_default_d3_config("DOSS"),
                 planner._get_basic_d3_config("DOSS"), planner._get_advanced_d3_config("DOSS"),
                 opt2d3_default_doss_config()):
        assert C.doss_step_ev(conf) == C.DOSS_DEFAULT_STEP_EV and C.doss_explicit_points(conf) is None
