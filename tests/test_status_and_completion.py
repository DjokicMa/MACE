"""`mace status` and `mace completion`: what they report for finished,
failed, killed and still-running calculations.

`mace status` is `mace monitor --status` over the database (the workflow's
isolated context database when there is one). `mace completion` sorts the
.out files of a directory by what CRYSTAL printed (mace/completion_checker.py,
typing finished runs with mace/utils/calc_detection.py), finds jobs that
finished but still hold a SLURM allocation, and can move files into
per-category folders.

The outputs are the real CRYSTAL23 outputs under tests/data; the one completed
OPT is the real killed tqb_pto run with CRYSTAL's own closing lines appended.
SLURM (squeue/scancel) is stubbed.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace import completion_checker as cc

DATA = Path(__file__).resolve().parent / "data"
MACE_CLI = REPO_ROOT / "mace_cli"

# CRYSTAL's closing lines of a converged optimization, in its own format.
OPT_END = (" * OPT END - CONVERGED * E(AU):  -1.268344904941E+03  POINTS    5 *\n"
           " EEEEEEEEEE TERMINATION  DATE 25 09 2026 TIME 12:53:46.2\n"
           "    TOTAL CPU TIME =     1812.401\n")

# A MACE BAND deck: title, then NLINE ISS NK INZB IFNB IPRINT IPLOT, one segment.
BAND_D3 = "BAND\ntqb_pto_band\n1 12 1000 1 52 1 0\n0 0 0 6 6 6\nEND\n"


def real(name):
    return (DATA / name).read_text(errors="ignore")


@pytest.fixture
def calc_dir(tmp_path):
    """One directory holding one of each kind of run, as a user's would."""
    d = tmp_path / "calcs"
    d.mkdir()
    # finished SP (SLAB graphene, layer group 37)
    shutil.copy(DATA / "low_dim_groups" / "graphene_lg37.out", d / "graphene_lg37.out")
    shutil.copy(DATA / "low_dim_groups" / "graphene_lg37.d12", d / "graphene_lg37.d12")
    # SCF that ran out of cycles (it still printed TOTAL CPU TIME)
    shutil.copy(DATA / "low_dim_groups" / "graphene_lg80.out", d / "graphene_lg80.out")
    # OPT killed by the walltime: the .out just stops
    (d / "tqb_pto.out").write_text(real("opt_restart/tqb_pto.out.timeout1"))
    (d / "tqb_pto.d12").write_text(real("opt_restart/tqb_pto.d12"))
    # a RESTART rerun that died in MPI_Abort
    (d / "tqb_abort.out").write_text(real("opt_restart/tqb_pto.out"))
    # the same killed OPT, had it converged
    (d / "tqb_done.out").write_text(real("opt_restart/tqb_pto.out.timeout1") + OPT_END)
    # a properties run: an SCF-style output with its BAND deck beside it
    (d / "tqb_band.out").write_text(real("low_dim_groups/polyyne_rg28.out"))
    (d / "tqb_band.d3").write_text(BAND_D3)
    return d


# ------------------------------------------------------------ categorising

@pytest.mark.parametrize("name, expected", [
    ("graphene_lg37.out", "completesp"),
    ("graphene_lg80.out", "too_many_scf"),
    ("tqb_pto.out", "ongoing"),
    ("tqb_abort.out", "potential"),
    ("tqb_done.out", "complete"),
    ("tqb_band.out", "completeband"),
])
def test_each_real_output_lands_in_its_category(calc_dir, name, expected):
    assert cc.categorize_output_file(calc_dir / name) == (expected, Path(name).stem)


def test_an_error_line_wins_over_the_completion_lines(tmp_path):
    """graphene_lg80 printed TOTAL CPU TIME after TOO MANY CYCLES: it failed."""
    text = real("low_dim_groups/graphene_lg80.out")
    assert "    TOTAL CPU TIME =" in text and "TOO MANY CYCLES" in text
    out = tmp_path / "x.out"
    out.write_text(text)
    assert cc.categorize_output_file(out)[0] == "too_many_scf"


@pytest.mark.parametrize("line, expected", [
    ("slurmstepd: error: *** JOB 42 ON amr-026 CANCELLED AT 2026-09-25T12:53:46 "
     "DUE TO TIME LIMIT ***", "time"),
    ("slurmstepd: error: Detected 1 oom_kill event in StepId=42.batch. Some of the step "
     "tasks have been OOM Killed. out-of-memory handler", "memory"),
    (" ERROR **** NEIGHB **** DISTANCE TOO SMALL", "geometry_small_dist"),
    (" ERROR **** CHOLSK **** BASIS SET LINEARLY DEPENDENT", "linear_basis"),
    ("ERROR: no writable scratch directory (tried /mnt/scratch/u/crys23)", "scratch"),
    (" ERROR *** some other stop", "unknown"),
])
def test_error_messages_pick_the_category(tmp_path, line, expected):
    """A killed run's .out with the message that stopped it appended."""
    out = tmp_path / "job.out"
    out.write_text(real("opt_restart/tqb_pto.out.timeout1") + line + "\n")
    assert cc.categorize_output_file(out)[0] == expected


def test_unreadable_output_is_unknown(tmp_path):
    missing = tmp_path / "gone.out"
    assert cc.categorize_output_file(missing) == ("unknown", "gone")


def test_title_words_do_not_type_a_finished_run(tmp_path):
    """A finished SCF named *_band / *_doss is still an SP: the name only
    decides for a properties run that stopped before printing its tell."""
    text = real("low_dim_groups/polyyne_rg28.out")
    for name in ("MoS2_band.out", "Fe_doss.out"):
        (tmp_path / name).write_text(text)
        assert cc.categorize_output_file(tmp_path / name)[0] == "completesp"


# ------------------------------------------------------------ scanning

def test_scan_finds_every_output_and_maps_names_to_paths(calc_dir):
    buckets, paths = cc.scan_directory(calc_dir)
    assert set(buckets) == set(cc.initialize_buckets())
    got = {name: cat for cat, names in buckets.items() for name in names}
    assert got == {"graphene_lg37": "completesp", "graphene_lg80": "too_many_scf",
                   "tqb_pto": "ongoing", "tqb_abort": "potential", "tqb_done": "complete",
                   "tqb_band": "completeband"}
    assert paths["tqb_pto"] == calc_dir / "tqb_pto.out"


def test_recursive_scan_reaches_workflow_step_folders(calc_dir):
    step = calc_dir / "workflow_outputs" / "step_001_OPT" / "dia"
    step.mkdir(parents=True)
    shutil.copy(DATA / "low_dim_groups" / "polyyne_rg51.out", step / "polyyne_rg51.out")
    flat, _ = cc.scan_directory(calc_dir)
    deep, paths = cc.scan_directory(calc_dir, recursive=True)
    assert "polyyne_rg51" not in flat["completesp"]
    assert "polyyne_rg51" in deep["completesp"]
    assert paths["polyyne_rg51"] == step / "polyyne_rg51.out"


def test_empty_directory_reports_nothing(tmp_path, capsys):
    buckets, paths = cc.scan_directory(tmp_path)
    assert paths == {} and not any(buckets.values())
    cc.print_summary(buckets)
    assert "No .out files found" in capsys.readouterr().out


def test_summary_counts(calc_dir, capsys):
    buckets, _ = cc.scan_directory(calc_dir)
    cc.print_summary(buckets, detailed=True)
    out = capsys.readouterr().out
    assert "Total files scanned: 6" in out
    assert "COMPLETED: 3 calculation(s)" in out
    assert "Geometry optimization (OPT END): 1" in out
    assert "Single point energy (SP): 1" in out
    assert "Band structure (D3 BAND): 1" in out
    assert "ERRORS: 2 calculation(s)" in out
    assert "SCF convergence failure: 1" in out
    assert "ONGOING/INCOMPLETE: 1 calculation(s)" in out
    assert "• tqb_pto" in out                       # the detailed listing


# ------------------------------------------------------------ organising

def test_move_completed_takes_only_finished_runs_and_their_files(calc_dir, monkeypatch):
    monkeypatch.chdir(calc_dir)
    buckets, _ = cc.scan_directory(".")
    cc.organize_completed(buckets, "completed")
    done = calc_dir / "completed"
    assert sorted(p.relative_to(done).as_posix() for p in done.rglob("*") if p.is_file()) == [
        "complete/tqb_done.out",
        "completeband/tqb_band.d3", "completeband/tqb_band.out",
        "completesp/graphene_lg37.d12", "completesp/graphene_lg37.out"]
    # failed, killed and aborted runs stay where they are
    for name in ("graphene_lg80.out", "tqb_pto.out", "tqb_pto.d12", "tqb_abort.out"):
        assert (calc_dir / name).is_file()


def test_organize_sorts_everything_and_honours_extensions(calc_dir, monkeypatch):
    monkeypatch.chdir(calc_dir)
    buckets, _ = cc.scan_directory(".")
    cc.organize_files(buckets, "sorted", extensions=[".out"])
    assert (calc_dir / "sorted" / "ongoing" / "tqb_pto.out").is_file()
    assert (calc_dir / "sorted" / "too_many_scf" / "graphene_lg80.out").is_file()
    assert (calc_dir / "sorted" / "potential" / "tqb_abort.out").is_file()
    # only .out was asked for: the decks stay
    assert (calc_dir / "tqb_pto.d12").is_file() and (calc_dir / "tqb_band.d3").is_file()


def test_nothing_completed_moves_nothing(tmp_path, monkeypatch, capsys):
    (tmp_path / "tqb_pto.out").write_text(real("opt_restart/tqb_pto.out.timeout1"))
    monkeypatch.chdir(tmp_path)
    buckets, _ = cc.scan_directory(".")
    cc.organize_completed(buckets)
    assert "No completed calculations to organize." in capsys.readouterr().out
    assert not (tmp_path / "completed").exists()


# ------------------------------------------------------------ zombie jobs

@pytest.mark.parametrize("log, job_id", [
    ("tqb_pto-17781763.o", "17781763"),        # the job script's -o $JOB-%J.o
    ("tqb_pto.o17781763", "17781763"),
    ("x_tqb_pto_17781763.o", "17781763"),
])
def test_job_id_comes_from_the_slurm_log_name(tmp_path, log, job_id):
    (tmp_path / "tqb_pto.out").write_text("")
    (tmp_path / log).write_text("")
    assert cc.find_slurm_job_id(tmp_path / "tqb_pto.out") == job_id


def test_no_log_no_job_id(tmp_path):
    (tmp_path / "tqb_pto.out").write_text("")
    assert cc.find_slurm_job_id(tmp_path / "tqb_pto.out") is None


class FakeSlurm:
    """squeue lists `running`; scancel records what it was asked to cancel."""

    def __init__(self, running):
        self.running = running
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, "".join(f"{j}\n" for j in self.running), "")
        if cmd[0] == "scancel":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(cmd)


@pytest.fixture
def zombies(calc_dir, monkeypatch):
    for stem, jid in (("tqb_done", "101"), ("graphene_lg80", "102"),
                      ("tqb_pto", "103"), ("graphene_lg37", "104")):
        (calc_dir / f"{stem}-{jid}.o").write_text("")
    fake = FakeSlurm(running=["101", "102", "103", "999"])      # 104 has left the queue
    monkeypatch.setattr(cc.subprocess, "run", fake)
    buckets, paths = cc.scan_directory(calc_dir)
    return buckets, paths, fake


def test_finished_jobs_still_in_the_queue_are_zombies(zombies):
    buckets, paths, _ = zombies
    found = {z["name"]: (z["job_id"], z["category"]) for z in cc.detect_zombie_jobs(buckets, paths)}
    assert found == {"tqb_done": ("101", "complete"),
                     "graphene_lg80": ("102", "too_many_scf"),
                     "tqb_pto": ("103", "ongoing")}


def test_nothing_running_means_no_zombies(calc_dir, monkeypatch):
    monkeypatch.setattr(cc.subprocess, "run", FakeSlurm(running=[]))
    buckets, paths = cc.scan_directory(calc_dir)
    assert cc.detect_zombie_jobs(buckets, paths) == []


def test_squeue_missing_means_no_zombies(calc_dir, monkeypatch, capsys):
    def no_squeue(cmd, **kw):
        raise FileNotFoundError("squeue")
    monkeypatch.setattr(cc.subprocess, "run", no_squeue)
    assert cc.get_running_jobs() == set()
    assert "Could not query squeue" in capsys.readouterr().out


def test_cancelling_zombies_asks_twice_and_cancels_what_was_agreed(zombies, monkeypatch, capsys):
    buckets, paths, fake = zombies
    answers = iter(["y", "n"])
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or next(answers))
    cc.remove_zombie_jobs(buckets, paths)
    assert len(prompts) == 2
    assert "Cancel 1 completed + 1 failed zombie job(s)?" in prompts[0]
    assert "Also cancel 1 ongoing job(s)" in prompts[1]
    cancels = [c for c in fake.calls if c[0] == "scancel"]
    assert cancels == [["scancel", "101", "102"]]
    assert "No ongoing jobs cancelled." in capsys.readouterr().out


def test_declining_cancels_nothing(zombies, monkeypatch):
    buckets, paths, fake = zombies
    monkeypatch.setattr("builtins.input", lambda p="": "")
    cc.remove_zombie_jobs(buckets, paths)
    assert not [c for c in fake.calls if c[0] == "scancel"]


# ------------------------------------------------------------ the CLI

def _env():
    env = dict(os.environ)
    env["MACE_NO_BANNER"] = "1"
    for var in ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        env.pop(var, None)
    return env


def _mace(cwd, *args):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args], cwd=str(cwd),
                       env=_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_completion_command_reports_and_moves(calc_dir, tmp_path):
    out = _mace(tmp_path, "completion", "-d", str(calc_dir), "--move-completed")
    assert "COMPLETED: 3 calculation(s)" in out
    assert (calc_dir / "completed" / "complete" / "tqb_done.out").is_file()
    assert (calc_dir / "tqb_pto.out").is_file()


def _seed(db_path):
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(db_path))
    db.create_material("tqb_pto", "PbTiO3")
    opt = db.create_calculation("tqb_pto", "OPT")
    db.update_calculation_status(opt, "failed", error_type="timeout_error",
                                 error_message="SLURM state TIMEOUT")
    sp = db.create_calculation("tqb_pto", "SP")
    db.update_calculation_status(sp, "completed")
    db.create_calculation("tqb_pto", "BAND")
    return opt, sp


def test_status_reads_the_database_in_the_current_directory(tmp_path):
    opt, sp = _seed(tmp_path / "materials.db")
    out = _mace(tmp_path, "status")
    assert "'mace status' has been merged into 'mace monitor --status'" in out
    for line in ("Materials: 1", "Calculations: 3", "Pending: 1", "Running: 0",
                 "Completed: 1", "Failed: 1", f"{opt} (tqb_pto): failed",
                 f"{sp} (tqb_pto): completed"):
        assert line in out, line
    assert "SLURM state TIMEOUT" not in out           # errors only in --detailed


def test_status_detailed_and_summary(tmp_path):
    _seed(tmp_path / "materials.db")
    detailed = _mace(tmp_path, "status", "--detailed")
    assert "Calculation Types:" in detailed and "OPT: 1" in detailed
    assert "Error: SLURM state TIMEOUT" in detailed
    summary = _mace(tmp_path, "monitor", "--summary")
    assert "Calculations: 3" in summary
    assert "Pending:" not in summary and "Recent Activity" not in summary


def test_status_reads_the_workflows_isolated_database(tmp_path):
    """With one isolated workflow context present, `mace status` shows its
    database, not the (empty) materials.db beside it."""
    wid = "workflow_20260925_125346"
    ctx = tmp_path / f".mace_context_{wid}"
    ctx.mkdir()
    _seed(ctx / "materials.db")
    out = _mace(tmp_path, "status")
    assert f"MACE Status (Workflow: {wid})" in out
    assert "Isolation Mode: isolated" in out
    assert "Calculations: 3" in out
    # A second context makes the choice ambiguous unless --workflow-id picks one.
    (tmp_path / ".mace_context_workflow_20260926_000000").mkdir()
    out = _mace(tmp_path, "status", "--workflow-id", wid)
    assert f"(Workflow: {wid})" in out and "Calculations: 3" in out
