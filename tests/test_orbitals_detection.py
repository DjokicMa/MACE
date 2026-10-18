"""ORBITALS runs are told apart, and complete only with their Molden files.

An ORBITALS deck starts with NEWK, as a DOSS deck does, so the type is read
before DOSS (by the deck and by the .out). CRYSTAL can end an ORBITALS run
normally after an input error (serial properties exits 0 on FORMAT ERROR IN
INPUT DECK): a finished .out is complete only when its <stem>_orb*.molden
files are next to it.

The outputs in tests/data/orbitals/props/ are real ORBITALS runs (CRYSTAL23,
HPCC): a 3D Si run (8 k points) and an H2O molecule.
"""
import shutil
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.utils import calc_detection as cd

PROPS = REPO_ROOT / "tests" / "data" / "orbitals" / "props"
SCF = REPO_ROOT / "tests" / "data" / "orbitals" / "scf"
DECK = "NEWK\n2 12\n1 0\nORBITALS\nsi_orbitals_orb\n1\n0\nEND"
DOSS_DECK = "NEWK\n8 16\n1 0\nDOSS\n0 1000 -1 -1 1 14 0\n-0.5 0.5\nEND"


# ------------------------------------------------------------------ decks
def test_orbitals_deck_type_before_doss():
    assert cd.d3_calc_type(cd.deck_records(DECK, is_d3=True)) == "ORBITALS"
    assert cd.d3_calc_type(cd.deck_records(DOSS_DECK, is_d3=True)) == "DOSS"
    assert cd.d3_calc_type({"NEWK", "ORBITALS", "DOSS"}) == "ORBITALS"
    assert "ORBITALS" in cd.PROPERTY_CALC_TYPES


def test_the_root_record_is_free_text():
    # a root named like a keyword is not read as one
    records = cd.deck_records("NEWK\n1 0\nORBITALS\nDOSS\n0\n0\nEND", is_d3=True)
    assert "DOSS" not in records and cd.d3_calc_type(records) == "ORBITALS"


@pytest.mark.parametrize("name,expected", [
    ("si_orbitals.d3", "ORBITALS"), ("si_orbitals2.out", "ORBITALS"), ("x_sp_orb.d3", "ORBITALS"),
    ("si_orbital.d3", None), ("si_orbitals_orb_00001_K000_real", "ORBITALS"), ("si_doss.d3", "DOSS")])
def test_filename_tokens(name, expected):
    assert cd.calc_type_from_filename(name) == expected


# ------------------------------------------------------------------ outputs
def test_orbitals_outputs_are_told_apart():
    for name in ("si_orbitals.out", "h2o_orbitals.out"):
        text = (PROPS / name).read_text()
        assert cd.is_orbitals_output(text)
        assert not cd.is_doss_output(text) and not cd.is_band_output(text)
        assert not cd.is_charge_potential_output(text) and not cd.is_transport_output(text)
    for name in ("si.out", "h2o.out", "fe.out"):
        assert not cd.is_orbitals_output((SCF / name).read_text())


def test_orbitals_output_without_deck_by_content(tmp_path):
    from mace.completion_checker import determine_completed_subtype
    p = tmp_path / "x_band.out"        # the name says BAND; the content wins
    shutil.copy(PROPS / "si_orbitals.out", p)
    lines = p.read_text().splitlines(keepends=True)
    assert determine_completed_subtype(p, lines) == "ORBITALS"


# ------------------------------------------------------------------ completion
def _orbitals_run(tmp_path, molden=True, deck=True, stem="si_orbitals"):
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROPS / "si_orbitals.out", d / f"{stem}.out")
    if deck:
        (d / f"{stem}.d3").write_text(DECK.replace("si_orbitals", stem))
    if molden:
        for k in ("00001_K000_real", "00002_K100_real", "00002_K100_complex"):
            (d / f"{stem}_orb_{k}.molden").write_text("[Molden Format]\n")
    return d / f"{stem}.out"


def test_complete_with_its_molden_files(tmp_path):
    from mace.completion_checker import CALC_TYPE_TO_BUCKET, categorize_output_file
    out = _orbitals_run(tmp_path)
    assert categorize_output_file(out) == ("completeorbitals", "si_orbitals")
    assert CALC_TYPE_TO_BUCKET["ORBITALS"] == "completeorbitals"
    assert [p.name for p in cd.orbitals_molden_files(out)][0] == "si_orbitals_orb_00001_K000_real.molden"
    assert cd.orbitals_missing_reason(out) is None


@pytest.mark.parametrize("deck", [True, False])
def test_not_complete_without_its_molden_files(tmp_path, deck):
    from mace.completion_checker import ERROR_DESCRIPTIONS, categorize_output_file
    out = _orbitals_run(tmp_path, molden=False, deck=deck)
    assert categorize_output_file(out) == ("orbitals_no_files", "si_orbitals")
    assert "Molden" in ERROR_DESCRIPTIONS["orbitals_no_files"]
    why = cd.orbitals_missing_reason(out)
    assert "si_orbitals_orb*.molden" in why and "FORMAT ERROR" in why


def test_other_runs_files_do_not_count(tmp_path):
    out = _orbitals_run(tmp_path, molden=False)
    (out.parent / "other_orbitals_orb.molden").write_text("x")
    (out.parent / "si_orbitals_orb.txt").write_text("x")
    assert cd.orbitals_missing_reason(out)


def test_molecule_file_counts(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    shutil.copy(PROPS / "h2o_orbitals.out", d / "h2o_orbitals.out")
    assert cd.orbitals_missing_reason(d / "h2o_orbitals.out")
    (d / "h2o_orbitals_orb.molden").write_text("[Molden Format]\n")
    from mace.completion_checker import categorize_output_file
    assert categorize_output_file(d / "h2o_orbitals.out")[0] == "completeorbitals"


def test_recovery_detector_needs_the_files(tmp_path):
    from mace.recovery.detector import CrystalErrorDetector
    det = CrystalErrorDetector(base_dir=str(tmp_path), enable_tracking=False)
    res = det.analyze_output_file(_orbitals_run(tmp_path / "a"))
    assert (res["status"], res["calc_type"]) == ("completed", "ORBITALS")
    res = det.analyze_output_file(_orbitals_run(tmp_path / "b", molden=False))
    assert (res["status"], res["error_type"], res["recoverable"]) == ("error", "orbitals_no_files", False)


def test_organize_moves_the_molden_files(tmp_path, monkeypatch):
    from mace.completion_checker import DEFAULT_EXTENSIONS, organize_files
    out = _orbitals_run(tmp_path)
    (out.parent / "si_orbitals.f9").write_text("x")
    (out.parent / "si_orbitals_orbX.d3").write_text("x")      # another deck's: not a .molden
    assert "_orb*.molden" in DEFAULT_EXTENSIONS
    monkeypatch.chdir(out.parent)
    organize_files({"completeorbitals": ["si_orbitals"]}, target_dir="sorted")
    moved = sorted(p.name for p in (out.parent / "sorted" / "completeorbitals").iterdir())
    assert moved == ["si_orbitals.d3", "si_orbitals.f9", "si_orbitals.out",
                     "si_orbitals_orb_00001_K000_real.molden", "si_orbitals_orb_00002_K100_complex.molden",
                     "si_orbitals_orb_00002_K100_real.molden"]
    assert (out.parent / "si_orbitals_orbX.d3").exists()


def test_queue_manager_reports_the_missing_files(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager as QM
    out = _orbitals_run(tmp_path, molden=False)
    qm = QM.__new__(QM)
    error_type, message = QM.analyze_calculation_error(qm, {"output_file": str(out), "calc_id": "x"})
    assert error_type == "orbitals_no_files" and "si_orbitals_orb*.molden" in message


# ------------------------------------------------------------------ the root name
def test_set_orbitals_root():
    assert cd.orbitals_root("mat_orbitals") == "mat_orbitals_orb"
    renamed = cd.set_orbitals_root(DECK, "mat_orbitals2")
    assert renamed.split("\n")[4] == "mat_orbitals2_orb"
    assert renamed.replace("mat_orbitals2_orb", "si_orbitals_orb") == DECK
    assert cd.set_orbitals_root(DOSS_DECK, "x") == DOSS_DECK


def test_material_id_of_an_orbitals_output():
    from mace.database.materials import create_material_id_from_file
    assert create_material_id_from_file("mat_orbitals.out") == create_material_id_from_file("mat_sp.out")


# ------------------------------------------------------------------ other readers of the type
def test_settings_extractor_and_cluster_progress():
    from mace.utils.settings_extractor import _extract_property_parameters
    assert "ORBITALS" in _extract_property_parameters(DECK)["property_types"]
    from mace.cluster.progress import properties_plan
    assert properties_plan(DECK, "") == ["NEWK", "ORBITALS"]
    assert properties_plan("", (PROPS / "si_orbitals.out").read_text()) == ["NEWK", "ORBITALS"]


def test_local_run_orders_an_orbitals_deck_after_its_parent():
    from mace.submission.local_run import parent_matches
    assert parent_matches("si_sp", "si_orbitals") and parent_matches("si", "si_orbitals2")


# ------------------------------------------------------------------ exact names
# The files of job X are X_orb.molden or X_orb_<n>_K<ijk>_(real|complex).molden.
# A prefix glob X_orb*.molden also caught the files of another job whose name
# starts with X_orb (for job "si": si_orbitals_orb_...), the standard naming.
@pytest.mark.parametrize("name,ok", [
    ("si_orb.molden", True), ("si_orb_00001_K000_real.molden", True),
    ("si_orb_00002_K100_complex.molden", True),
    ("si_orbitals_orb_00001_K000_real.molden", False), ("si_orbX.molden", False),
    ("si_orb_00001_K000_imag.molden", False), ("si_orb_1x_K0_real.molden", False),
    ("si_orb.molden.bak", False)])
def test_molden_name_is_matched_exactly(name, ok):
    assert cd.is_orbitals_molden_name(name, "si") is ok


def test_another_jobs_files_do_not_count(tmp_path):
    out = _orbitals_run(tmp_path, molden=False, stem="si")
    _orbitals_run(tmp_path, stem="si_orbitals")               # si_orbitals_orb_*.molden beside it
    assert cd.orbitals_molden_files(out) == []
    assert cd.orbitals_missing_reason(out)


def test_organize_leaves_another_jobs_molden_files(tmp_path, monkeypatch):
    from mace.completion_checker import organize_files
    sp = _orbitals_run(tmp_path, molden=False, deck=False, stem="si")
    (sp.parent / "si.d12").write_text("x")
    orb = _orbitals_run(tmp_path, stem="si_orbitals")
    monkeypatch.chdir(orb.parent)
    organize_files({"completesp": ["si"], "completeorbitals": ["si_orbitals"]}, target_dir="sorted")
    sp_files = sorted(p.name for p in (orb.parent / "sorted" / "completesp").iterdir())
    assert sp_files == ["si.d12", "si.out"]
    orb_files = sorted(p.name for p in (orb.parent / "sorted" / "completeorbitals").iterdir())
    assert len([n for n in orb_files if n.endswith(".molden")]) == 3


def test_file_records_index_only_this_runs_molden_files(tmp_path):
    # review fix: a folder-wide *.molden gave every calculation a row for
    # every ORBITALS job's files
    from mace.queue.manager import EnhancedCrystalQueueManager as QM
    out = _orbitals_run(tmp_path, stem="si")
    _orbitals_run(tmp_path, stem="si_orbitals")
    rows = []

    class DB:
        def add_file_record(self, **kw):
            rows.append(kw)
    qm = QM.__new__(QM)
    qm.enable_tracking, qm.db = True, DB()
    QM.update_file_records(qm, {"calc_id": "c1", "work_dir": str(out.parent), "calc_type": "ORBITALS",
                                "output_file": str(out)})
    molden = sorted(r["file_name"] for r in rows if r["file_type"] == "orbitals")
    assert molden == [f"si_orb_{k}.molden" for k in ("00001_K000_real", "00002_K100_complex", "00002_K100_real")]
    rows.clear()
    QM.update_file_records(qm, {"calc_id": "c2", "work_dir": str(out.parent), "calc_type": "SP",
                                "output_file": str(out.parent / "x.out")})
    assert not [r for r in rows if r["file_type"] == "orbitals"]


# ------------------------------------------------------------------ mace submit refuses a failing deck
@pytest.mark.parametrize("mode", [[], ["--track"], ["--array"], ["--track", "--array"]])
def test_submit_refuses_a_deck_preflight_fails(tmp_path, mode):
    # review fix: only `mace preflight` ran orbitals_deck_problem, so a deck
    # with ICAR and ILOC on one line was queued and wrote no file
    from fake_slurm import FakeSlurm
    from test_site_jobscript import _env, _mace
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    d = tmp_path / "d"
    d.mkdir()
    (d / "x_orbitals.d3").write_text("NEWK\n2 12\n1 0\nORBITALS\nx_orbitals_orb\n1 0\nEND\n")
    (d / "x_orbitals.f9").write_text("fort9\n")
    p = _mace(d, env, "submit", *mode, "x_orbitals.d3")
    out = p.stdout + p.stderr
    assert p.returncode != 0, out
    if mode != ["--track"]:      # the tracked loop's progress bar captures the manager's output
        assert "ICAR and ILOC are on one line" in out
    assert not fake.calls("sbatch")
    # a good deck still goes
    (d / "x_orbitals.d3").write_text(DECK.replace("si_orbitals", "x_orbitals") + "\n")
    p = _mace(d, env, "submit", *mode, "x_orbitals.d3")
    assert p.returncode == 0, p.stdout + p.stderr
    assert fake.calls("sbatch")
