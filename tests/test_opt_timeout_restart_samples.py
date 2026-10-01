"""The timeout RESTART path on the committed real runs, so it is tested where
the test/ corpus is absent (CI).

test_opt_timeout_restart.py and test_opt_restart_abort.py cover the same path
on the corpus OPT/1_dia_opt_rev1 run and skip without it. The runs used here
are HPCC runs that took exactly this path (tests/data/opt_restart):

  * tqb_pto (PbTiO3, P m-3m, cell-only OPT): the first run was killed by the
    walltime during point 5 (tqb_pto.out.timeout1 - the name the recovery
    gives a timed-out output); its rerun used tqb_pto.d12, the original deck
    plus RESTART. The original deck, as submitted, is
    tests/data/samples/ecp_decks/TiPbO3_mp-19845_...d12.
  * tqc_agbr (Ag2Br3, R -3 c): the first run was killed during point 4
    (tqc_agbr.out.timeout1); the RESTART rerun (tqc_agbr.d12,
    tqc_agbr.out) resumed at point 3 and was cut off during point 7.

SLURM is faked twice: with the in-process runner the recovery accepts
(FakeSlurm from test_opt_timeout_restart), and, end to end, with sbatch /
squeue / sacct / scontrol / sacctmgr executables on PATH.
"""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mace.recovery import opt_restart
from test_opt_timeout_restart import FakeSlurm, cut_before, job_script, new_time
from test_opt_restart_abort import _as_before_this_change, _generated_script

DATA = Path(__file__).resolve().parent / "data"
RUNS = DATA / "opt_restart"
PTO_ORIGINAL = (DATA / "samples" / "ecp_decks" /
                "TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
                "triplezeta_opt_B3LYP-D3-D3_optimized.d12")
AGBR_ECP = (DATA / "samples" / "ecp_decks" /
            "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.d12")


def run(name):
    return (RUNS / name).read_text(errors="ignore")


def killed_first_run(material="tqb_pto"):
    return run(f"{material}.out.timeout1")


def original_deck(material="tqb_pto"):
    """The deck the first (killed) run used."""
    if material == "tqb_pto":
        return PTO_ORIGINAL.read_text()
    return opt_restart.remove_optgeom_restart(run(f"{material}.d12"))[0]


# ------------------------------------------------------------- output markers

@pytest.mark.parametrize("material", ["tqb_pto", "tqc_agbr"])
def test_killed_first_runs_had_progress(material):
    text = killed_first_run(material)
    assert "GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT" in text
    assert opt_restart.optimization_has_progress(text)
    assert not opt_restart.ended_normally(text)


@pytest.mark.parametrize("material", ["tqb_pto", "tqc_agbr"])
def test_killed_inside_the_first_scf_has_no_progress(material):
    text = killed_first_run(material)
    killed = cut_before(text, "== SCF ENDED")
    assert "OPTIMIZATION - POINT    1" in killed
    assert not opt_restart.optimization_has_progress(killed)
    # after the first SCF, in the first gradient, before OPTINFO.DAT was written
    killed = cut_before(text, "GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT")
    assert not opt_restart.optimization_has_progress(killed)


def test_killed_after_the_first_cycle_has_progress():
    text = killed_first_run()
    assert opt_restart.optimization_has_progress(cut_before(text, "OPTIMIZATION - POINT    2"))
    assert opt_restart.optimization_has_progress(cut_before(text, "OPTIMIZATION - POINT    3"))


@pytest.mark.parametrize("material, first_point", [("tqb_pto", 4), ("tqc_agbr", 3)])
def test_rerun_killed_before_its_own_first_cycle_still_has_progress(material, first_point):
    """A RESTART rerun numbers from where it resumed and never prints the
    OPTINFO line; its history is the OPTINFO.DAT it started from."""
    text = run(f"{material}.out")
    assert "RESTARTING FROM A PREVIOUS GEOMETRY OPTIMIZATION RUN" in text
    assert "INFORMATION STORED IN OPTINFO.DAT" not in text
    points = [int(n) for n in re.findall(r"OPTIMIZATION - POINT\s+(\d+)", text)]
    assert points[0] == first_point
    # killed right after announcing its first point, before finishing a cycle
    lines = text.splitlines(keepends=True)
    first = next(i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l)
    killed = "".join(lines[:first + 1])
    assert not re.search(r"POINT\s+%d\b" % (first_point + 1), killed)
    assert opt_restart.optimization_has_progress(killed)


def test_markers_agree_with_point_counts_on_every_committed_output():
    checked = 0
    for out in sorted(DATA.rglob("*.out*")):
        text = out.read_text(errors="ignore")
        points = [int(n) for n in re.findall(r"OPTIMIZATION - POINT\s+(\d+)", text)]
        restarted = "RESTARTING FROM A PREVIOUS GEOMETRY OPTIMIZATION RUN" in text
        assert opt_restart.optimization_has_progress(text) == (len(points) >= 2 or restarted), out
        checked += bool(points)
    assert checked >= 4


# ------------------------------------------------------------ RESTART in deck

@pytest.mark.parametrize("material", ["tqb_pto", "tqc_agbr"])
def test_restart_makes_exactly_the_deck_the_rerun_used(material):
    """Adding RESTART to the killed run's deck gives, byte for byte, the deck
    the successful HPCC rerun was submitted with."""
    new, changed = opt_restart.add_optgeom_restart(original_deck(material))
    assert changed
    assert new == run(f"{material}.d12")
    lines = new.splitlines()
    i = lines.index("RESTART")
    assert lines[i - 2:i] == ["OPTGEOM", "FULLOPTG"]


def test_the_committed_original_is_the_rerun_deck_without_restart():
    assert PTO_ORIGINAL.read_text() == opt_restart.remove_optgeom_restart(run("tqb_pto.d12"))[0]


def test_restart_is_never_added_twice_to_a_real_deck():
    real = run("tqb_pto.d12")
    assert opt_restart.has_optgeom_restart(real)
    assert opt_restart.add_optgeom_restart(real) == (real, False)


def test_ecp_deck_gets_restart_in_its_optgeom_only():
    """A real ECP deck (Ag as 247): the basis-set records after OPTGEOM's END
    stay untouched."""
    deck = AGBR_ECP.read_text()
    new, changed = opt_restart.add_optgeom_restart(deck)
    assert changed and new.count("RESTART") == 1
    assert new.replace("FULLOPTG\nRESTART\n", "FULLOPTG\n", 1) == deck
    assert opt_restart.remove_optgeom_restart(new) == (deck, True)


# ------------------------------------------------- the recovery engine

@pytest.fixture
def engine(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))


def killed_job(tmp_path, monkeypatch, out_text, material="tqb_pto", optinfo=True,
               scratch_env=True, walltime="3-00:00:00"):
    """<material>.d12/.out/.sh as the killed run left them, and its scratch."""
    d12 = tmp_path / f"{material}.d12"
    d12.write_text(original_deck(material))
    (tmp_path / f"{material}.out").write_text(out_text)
    scratch = tmp_path / "scratch"
    (scratch / "crys23" / material).mkdir(parents=True, exist_ok=True)
    if optinfo:
        (scratch / "crys23" / material / "OPTINFO.DAT").write_text("steps")
    monkeypatch.setenv("SCRATCH", str(scratch) if scratch_env else "")
    return d12, job_script(tmp_path, walltime, name=material)


def recover(engine, script, runner=None, d12=None):
    engine.slurm_runner = runner or FakeSlurm()
    d12 = d12 or script.with_suffix(".d12")
    calc = {"calc_id": "T", "material_id": "M", "calc_type": "OPT",
            "input_file": str(d12), "work_dir": str(d12.parent),
            "job_script": str(script), "error_type": "timeout_error"}
    return engine.attempt_recovery(calc, create_record=False)


@pytest.mark.parametrize("material", ["tqb_pto", "tqc_agbr"])
def test_walltime_killed_opt_is_resubmitted_as_the_real_rerun(engine, tmp_path, monkeypatch,
                                                                material):
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run(material), material)
    res = recover(engine, script)
    assert res["fixed_input_file"] == d12                 # same deck, same $JOB, same scratch
    assert d12.read_text() == run(f"{material}.d12")
    assert (tmp_path / f"{material}.out.timeout1").read_text() == killed_first_run(material)
    assert new_time(res) == "6-00:00:00"
    assert script.read_text().count("#SBATCH -t 3-00:00:00") == 1   # the original is kept


def test_first_scf_kill_is_resubmitted_from_the_start(engine, tmp_path, monkeypatch, capsys):
    d12, script = killed_job(tmp_path, monkeypatch,
                             cut_before(killed_first_run(), "== SCF ENDED"), optinfo=False)
    res = recover(engine, script)
    assert d12.read_text() == original_deck()
    assert new_time(res) == "6-00:00:00"
    assert "before the first optimization step completed" in capsys.readouterr().out


def test_first_scf_kill_at_the_queue_limit_is_not_resubmitted(engine, tmp_path, monkeypatch,
                                                               capsys):
    d12, script = killed_job(tmp_path, monkeypatch,
                             cut_before(killed_first_run(), "== SCF ENDED"), optinfo=False,
                             walltime="7-00:00:00")
    assert recover(engine, script) is None
    assert "would time out again" in capsys.readouterr().out
    assert d12.read_text() == original_deck()
    assert not list(tmp_path.glob("tqb_pto_recovery_*.sh"))


def test_progress_at_the_queue_limit_restarts_at_the_same_walltime(engine, tmp_path, monkeypatch,
                                                                    capsys):
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run(), walltime="7-00:00:00")
    res = recover(engine, script)
    assert new_time(res) == "7-00:00:00"
    assert d12.read_text() == run("tqb_pto.d12")
    out = capsys.readouterr().out
    assert "already at the queue limit 7-00:00:00" in out
    assert "optimization continues (restart)" in out


def test_restart_without_optinfo_starts_fresh_from_the_best_point(engine, tmp_path, monkeypatch):
    """Scratch visible but OPTINFO.DAT gone: no RESTART; the deck is rewritten
    to the lowest-energy geometry the killed run reached (point 2: a =
    3.92491474) and the original kept as .orig."""
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run(), optinfo=False)
    res = recover(engine, script)
    assert res and new_time(res) == "6-00:00:00"
    assert not opt_restart.has_optgeom_restart(d12.read_text())
    assert d12.read_text() == original_deck().replace("\n3.94649838\n", "\n3.92491474\n")
    assert (tmp_path / "tqb_pto.d12.orig").read_text() == original_deck()


def test_unseen_scratch_trusts_the_output(engine, tmp_path, monkeypatch):
    """$SCRATCH empty where the recovery runs: the output decides (the job
    script re-checks OPTINFO.DAT on the compute node)."""
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run(), scratch_env=False)
    recover(engine, script)
    assert d12.read_text() == run("tqb_pto.d12")


def test_the_rerun_timing_out_too_keeps_one_restart(engine, tmp_path, monkeypatch):
    """tqc_agbr as it happened: killed, rerun with RESTART, and the rerun cut
    off in its turn."""
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run("tqc_agbr"), "tqc_agbr",
                             walltime="1-00:00:00")
    first = recover(engine, script)
    (tmp_path / "tqc_agbr.out").write_text(run("tqc_agbr.out"))
    second = recover(engine, first["fixed_job_script"], d12=d12)
    assert d12.read_text() == run("tqc_agbr.d12")
    assert d12.read_text().count("RESTART") == 1
    assert (tmp_path / "tqc_agbr.out.timeout1").read_text() == killed_first_run("tqc_agbr")
    assert (tmp_path / "tqc_agbr.out.timeout2").read_text() == run("tqc_agbr.out")
    assert (new_time(first), new_time(second)) == ("2-00:00:00", "4-00:00:00")


def test_enable_restart_false_turns_it_off(engine, tmp_path, monkeypatch):
    d12, script = killed_job(tmp_path, monkeypatch, killed_first_run())
    engine.config["error_recovery"]["timeout_error"]["enable_restart"] = False
    recover(engine, script)
    assert d12.read_text() == original_deck()
    assert not (tmp_path / "tqb_pto.out.timeout1").exists()


def test_an_old_job_script_copy_gets_the_restart_staging(engine, tmp_path, monkeypatch):
    (tmp_path / "testmat.d12").write_text(original_deck())
    (tmp_path / "testmat.out").write_text(killed_first_run())
    monkeypatch.setenv("SCRATCH", "")
    old = _as_before_this_change(_generated_script(tmp_path)).replace(
        "#SBATCH -t 7-00:00:00", "#SBATCH -t 1-00:00:00")
    (tmp_path / "testmat.sh").write_text(old)
    res = recover(engine, tmp_path / "testmat.sh")
    text = res["fixed_job_script"].read_text()
    assert "# OPTGEOM RESTART" in text and opt_restart._NEW_GUESSP_IF in text
    assert (tmp_path / "testmat.sh").read_text() == old
    assert subprocess.run(["bash", "-n", str(res["fixed_job_script"])]).returncode == 0
    assert opt_restart.has_optgeom_restart((tmp_path / "testmat.d12").read_text())


# ------------------------------------------- errors only in scratch fort.87

@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                       enable_tracking=True, organize_outputs=False)


def _classify(mgr, tmp_path, monkeypatch, out_text, fort87, stale=False):
    scratch = tmp_path / "scratch" / "crys23" / "tqb_pto"
    scratch.mkdir(parents=True)
    (scratch / "INPUT").write_text(original_deck())
    (scratch / "fort.87").write_text(fort87)
    if stale:
        old = time.time() - 3600
        os.utime(scratch / "fort.87", (old, old))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "tqb_pto.out").write_text(out_text)
    script = job_script(tmp_path, "1-00:00:00", name="tqb_pto")
    return mgr.analyze_calculation_error({"output_file": str(tmp_path / "tqb_pto.out"),
                                          "input_file": str(tmp_path / "tqb_pto.d12"),
                                          "job_script": str(script)})


def _aborted_in_first_scf():
    return (cut_before(killed_first_run(), "== SCF ENDED") +
            "Abort(1) on node 3 (rank 3 in comm 0): application called "
            "MPI_Abort(MPI_COMM_WORLD, 1) - process 3\n")


def test_fort87_error_gives_the_real_error_type(mgr, tmp_path, monkeypatch):
    assert _classify(mgr, tmp_path, monkeypatch, _aborted_in_first_scf(),
                     " ERROR **** NEIGHB **** DISTANCE TOO SMALL\n")[0] == "geometry_error"


def test_unmatched_fort87_error_is_reported_in_crystals_words(mgr, tmp_path, monkeypatch):
    killed = cut_before(killed_first_run(), "== SCF ENDED")
    assert _classify(mgr, tmp_path, monkeypatch, killed, " ERROR **** SOMETHING **** NEW\n") == \
        ("crystal_error", "fort.87: ERROR **** SOMETHING **** NEW")


def test_stale_fort87_is_ignored(mgr, tmp_path, monkeypatch):
    killed = cut_before(killed_first_run(), "== SCF ENDED")
    assert _classify(mgr, tmp_path, monkeypatch, killed, " ERROR **** SOMETHING **** OLD\n",
                     stale=True)[0] == "unknown_error"


# ------------------------------- end to end, SLURM as executables on PATH

FAKE_SLURM = r'''#!{python}
"""Fake SLURM command: MSU HPCC's mendoza_q (7-day MaxTime). Logs each call."""
import json, os, re, sys
state = os.environ["FAKE_SLURM_DIR"]
cmd, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(os.path.join(state, "calls.jsonl"), "a") as f:
    f.write(json.dumps([cmd] + args) + "\n")

def seconds(t):
    d, _, hms = t.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]

if cmd == "sbatch":
    if "--test-only" in args:
        if seconds(args[args.index("-t") + 1]) > 7 * 86400:
            sys.stderr.write("sbatch: error: Batch job submission failed: Requested time "
                             "limit is invalid (missing or exceeds some limit)\n")
            sys.exit(1)
        sys.stderr.write("sbatch: Job 17781763 to start at 2026-10-10T23:17:35 using 32 "
                         "processors on nodes vim-001 in partition mendoza_q\n")
        sys.exit(0)
    counter = os.path.join(state, "next_id")
    n = int(open(counter).read()) if os.path.exists(counter) else 777001
    open(counter, "w").write(str(n + 1))
    print(f"Submitted batch job {n}")
elif cmd == "squeue":
    print("JOBID,STATE,START")            # the job has left the queue
elif cmd == "sacct":
    print("TIMEOUT ")
elif cmd == "scontrol":
    if "-a" not in args:
        sys.stderr.write(f"Partition {args[-2]} not found\n")
        sys.exit(1)
    print(f"PartitionName={args[-2]} AllowAccounts={args[-2]} MaxTime=7-00:00:00 MinNodes=0")
elif cmd == "sacctmgr":
    print("")
else:
    sys.exit(2)
'''


@pytest.fixture
def slurm_on_path(tmp_path, monkeypatch):
    state = tmp_path / "slurm"
    bindir = state / "bin"
    bindir.mkdir(parents=True)
    body = FAKE_SLURM.replace("{python}", sys.executable)
    for name in ("sbatch", "squeue", "sacct", "scontrol", "sacctmgr"):
        p = bindir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))

    def calls():
        log = state / "calls.jsonl"
        return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    return calls


def test_walltime_kill_is_resubmitted_with_restart_through_real_slurm_calls(
        tmp_path, monkeypatch, slurm_on_path):
    """Submit through the real generator and sbatch; the job then vanishes
    from squeue and sacct says TIMEOUT; the status sweep keeps the killed
    output, adds RESTART, doubles the walltime within the queue's limit and
    sbatches the bumped script - all via the SLURM commands on PATH."""
    from mace.queue.manager import EnhancedCrystalQueueManager

    work = tmp_path / "work"
    work.mkdir()
    (work / "tqb_pto.d12").write_text(original_deck())
    scratch = tmp_path / "scr"
    (scratch / "crys23" / "tqb_pto").mkdir(parents=True)
    (scratch / "crys23" / "tqb_pto" / "OPTINFO.DAT").write_text("steps")
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.chdir(work)
    for var in ("SLURM_JOB_ID", "MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)

    mgr = EnhancedCrystalQueueManager(d12_dir=str(work), db_path=str(work / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    mgr.walltime_override = "3-00:00:00"
    calc_id = mgr.submit_calculation(work / "tqb_pto.d12")
    assert calc_id
    assert mgr.db.get_calculation(calc_id)["slurm_job_id"] == "777001"
    assert "#SBATCH -t 3-00:00:00" in (work / "tqb_pto.sh").read_text()

    # The job runs out of time during point 5.
    (work / "tqb_pto.out").write_text(killed_first_run())
    mgr.check_queue_status()

    old = mgr.db.get_calculation(calc_id)
    assert old["status"] == "resubmitted", old
    assert (work / "tqb_pto.d12").read_text() == run("tqb_pto.d12")
    assert (work / "tqb_pto.out.timeout1").read_text() == killed_first_run()

    calls = slurm_on_path()
    submits = [c for c in calls if c[0] == "sbatch" and "--test-only" not in c]
    assert len(submits) == 2
    assert Path(submits[0][-1]).name == "tqb_pto.sh"
    resub = Path(submits[1][-1])
    assert re.fullmatch(r"tqb_pto_recovery_\d{8}_\d{6}\.sh", resub.name)
    assert "#SBATCH -t 6-00:00:00" in resub.read_text()
    assert "# OPTGEOM RESTART" in resub.read_text()
    assert ["sacct", "-j", "777001", "--format=State", "-n", "-X"] in calls
    assert ["sbatch", "--test-only", "-t", "6-00:00:00", str(work / "tqb_pto.sh")] in calls
    new = [c for c in mgr.db.get_all_calculations() if c["calc_id"] != calc_id]
    assert [c["slurm_job_id"] for c in new] == ["777002"]
