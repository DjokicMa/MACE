"""/api/related (1.1.6, the panel context): the run a plot, spectrum, volume,
properties deck or job log came from, whose structure the side panels show
read-only. Repo data only: examples/gui and the CRYSTAL23 parity fixtures
(tests/fixtures/gui_parity), copied into a temp folder."""
import shutil

import pytest

import guiharness as H
import p4util as U
from test_parity_fixtures import unpack

pytest.importorskip("numpy")
pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("related")
    root = H.examples_copy(tmp)
    unpack(root / "parity")
    # a band file beside two SCF runs of its material: ambiguous
    amb = root / "amb"
    amb.mkdir()
    shutil.copyfile(root / "outputs" / "Si_sp.out", amb / "Si_sp.out")
    shutil.copyfile(root / "outputs" / "Si_sp.out", amb / "Si_opt.out")
    (amb / "Si_band.BAND.DAT").write_text("# a band file\n")
    # a band file beside one SCF run in a workflow's step folder beside it
    wf = root / "wf"
    (wf / "step_002_SP").mkdir(parents=True)
    (wf / "step_003_BAND").mkdir()
    shutil.copyfile(root / "outputs" / "Si_sp.out", wf / "step_002_SP" / "Si_sp.out")
    (wf / "step_003_BAND" / "Si_band.BAND.DAT").write_text("# a band file\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def related(base, path, pick=None):
    q = f"api/related?path={path}" + (f"&pick={pick}" if pick else "")
    st, _, r = U.req(base, q)
    assert st == 200, (st, r)
    return r


@H.needs_ase
def test_a_dos_plot_shows_its_properties_runs_structure_and_projections(served):
    root, base = served
    r = related(base, "parity/mgo_doss.DOSS.DAT")
    assert r["state"] == "found" and r["source"] == "parity/mgo_doss.out", r
    s = r["structure"]
    assert s["symbols"] == ["Mg", "O"] and abs(s["lattice"]["a"] - 2.97692) < 1e-4, s["lattice"]
    assert s["symmetry"]["number"] == 225
    # the .d3's projections, by element: Mg s/p/d/all on atom 1, O on atom 2
    pj = {(p["element"], p["orbital"]): p["atoms"] for p in r["projections"]}
    assert pj[("Mg", "s")] == [0] and pj[("O", "all")] == [1], pj


@H.needs_ase
@pytest.mark.parametrize("path, source", [
    ("parity/mgo_dens/DENS_CUBE.DAT", "parity/mgo_dens/mgo_dens.out"),     # CRYSTAL's generic name: the folder is the run
    ("parity/mgo_echg.f25", "parity/mgo_echg.out"),
    ("parity/mgo_doss.d3", "parity/mgo_doss.out"),
    ("parity/water_freq_IRSPEC.DAT", "parity/water_freq.out"),             # the FREQ run, not the optimization beside it
    ("parity/water_freq.out", "parity/water_freq.out"),
    ("parity/water_opt-123456.o", "parity/water_opt.out"),                 # a job log: its run
    ("wf/step_003_BAND/Si_band.BAND.DAT", "wf/step_002_SP/Si_sp.out"),     # a workflow's SP step beside the BAND step
])
def test_the_run_a_file_came_from(served, path, source):
    root, base = served
    r = related(base, path)
    assert r["state"] == "found" and r["source"] == source, r
    assert r["structure"]["numbers"] and r["structure"]["path"] == source


@H.needs_ase
def test_two_candidate_runs_are_ambiguous_until_one_is_picked(served):
    root, base = served
    r = related(base, "amb/Si_band.BAND.DAT")
    assert r["state"] == "ambiguous" and r["structure"] is None, r
    assert sorted(c["path"] for c in r["candidates"]) == ["amb/Si_opt.out", "amb/Si_sp.out"]
    r = related(base, "amb/Si_band.BAND.DAT", pick="amb/Si_opt.out")
    assert r["state"] == "found" and r["source"] == "amb/Si_opt.out" and r["structure"]["symbols"] == ["Si", "Si"], r
    # a pick that is not a candidate is not read
    r = related(base, "amb/Si_band.BAND.DAT", pick="structures/NaCl.cif")
    assert r["state"] == "ambiguous" and r["source"] is None, r


def test_a_file_with_no_run_has_none(served):
    root, base = served
    assert related(base, "README.md")["state"] == "none"
    assert related(base, "structures/NaCl.cif")["state"] == "none"


def test_paths_outside_the_folder_are_refused(served):
    root, base = served
    st, _, _ = U.req(base, "api/related?path=../../etc/passwd")
    assert st in (403, 404), st
    st, _, _ = U.req(base, "api/related?path=nope/missing.DOSS.DAT")
    assert st == 404, st
