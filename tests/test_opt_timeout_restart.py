"""A walltime-killed OPT continues from its last completed step (OPTGEOM
RESTART), and the doubled walltime never exceeds what the job's queue allows.

CRYSTAL23 manual 7.4.3: RESTART reads earlier optimization steps from
OPTINFO.DAT (written after each optimization cycle) with the SCF guess from
fort.20 = the fort.9 of the last successful SCF, and the deck must otherwise be
the one the first run used. A job killed inside its first SCF has no
OPTINFO.DAT and is resubmitted from the start.

Output markers are checked against the real OPT corpus; truncated copies of a
real output stand in for the two kinds of kill. SLURM is faked with the
limits measured on MSU HPCC (7 days on mendoza_q, 14 on mendoza_q_long with
-A mendoza_q_long).
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import TEST_DATA
from mace.recovery import opt_restart, slurm_limits

SUBMIT_SH = Path(__file__).resolve().parent.parent / "mace" / "submission" / "submitcrystal23.sh"

DIA_OUT = "OPT/1_dia_opt_rev1.out"           # 4-point cell OPT, converged
DIA_D12 = "OPT/1_dia_opt_rev1.d12"
RESTARTED = "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized"   # a real RESTART run (starts at POINT 30)


def corpus(rel):
    p = TEST_DATA / rel
    if not p.is_file():
        pytest.skip(f"test/ corpus file {rel} not present")
    return p


def cut_before(text, marker, occurrence=1):
    """The output as it stood when the job was killed just before the
    `occurrence`-th line containing `marker`."""
    lines = text.splitlines(keepends=True)
    seen = 0
    for i, line in enumerate(lines):
        if marker in line:
            seen += 1
            if seen == occurrence:
                return "".join(lines[:i])
    raise AssertionError(f"{marker!r} #{occurrence} not in output")


# ------------------------------------------------------------- output markers

def test_completed_real_opt_shows_progress():
    assert opt_restart.optimization_has_progress(corpus(DIA_OUT).read_text(errors="ignore"))


def test_real_output_killed_inside_first_scf_has_no_progress():
    text = corpus(DIA_OUT).read_text(errors="ignore")
    killed = cut_before(text, "== SCF ENDED")          # still iterating SCF #1
    assert "OPTIMIZATION - POINT    1" in killed
    assert not opt_restart.optimization_has_progress(killed)
    # Also killed in the first gradient, after the SCF but before OPTINFO.DAT.
    killed = cut_before(text, "GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT")
    assert not opt_restart.optimization_has_progress(killed)


def test_real_output_killed_mid_optimization_has_progress():
    text = corpus(DIA_OUT).read_text(errors="ignore")
    # Killed during point 3: two cycles are in OPTINFO.DAT.
    killed = cut_before(text, "OPTIMIZATION - POINT    3") + " CELL OPTIMIZATION - POINT    3\n"
    assert opt_restart.optimization_has_progress(killed)
    # Killed right after the first cycle was stored, before point 2 printed.
    killed = cut_before(text, "OPTIMIZATION - POINT    2")
    assert opt_restart.optimization_has_progress(killed)


def test_restarted_run_killed_before_its_own_first_cycle_still_has_progress():
    """A RESTART run numbers from where it resumed and never prints the
    OPTINFO line; its history is the OPTINFO.DAT it started from."""
    text = corpus(RESTARTED + ".out").read_text(errors="ignore")
    assert "GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT" not in text
    killed = cut_before(text, "COORDINATE AND CELL OPTIMIZATION - POINT   31")
    assert opt_restart.optimization_has_progress(killed)


def test_markers_agree_with_point_counts_across_the_corpus():
    if not (TEST_DATA / "OPT").is_dir():
        pytest.skip("test/ corpus not present")
    checked = 0
    for out in sorted((TEST_DATA / "OPT").glob("*.out")):
        text = out.read_text(errors="ignore")
        points = [int(n) for n in re.findall(r"OPTIMIZATION - POINT\s+(\d+)", text)]
        assert opt_restart.optimization_has_progress(text) == (len(points) >= 2), out.name
        checked += bool(points)
    assert checked > 20


# ------------------------------------------------------------ RESTART in deck

def test_restart_goes_after_the_type_line_and_nothing_else_changes():
    text = corpus(DIA_D12).read_text()
    new, changed = opt_restart.add_optgeom_restart(text)
    assert changed
    old_lines, new_lines = text.splitlines(), new.splitlines()
    i = new_lines.index("RESTART")
    assert new_lines[i - 2:i] == ["OPTGEOM", "FULLOPTG"]
    assert new_lines[:i] + new_lines[i + 1:] == old_lines


def test_restart_is_never_added_twice():
    text = corpus(DIA_D12).read_text()
    once, _ = opt_restart.add_optgeom_restart(text)
    twice, changed = opt_restart.add_optgeom_restart(once)
    assert not changed and twice == once
    # The real restarted deck already carries it.
    real = corpus(RESTARTED + ".d12").read_text()
    assert opt_restart.has_optgeom_restart(real)
    assert opt_restart.add_optgeom_restart(real) == (real, False)


def test_restart_placement_without_type_line_and_other_blocks_untouched():
    deck = ("title\nCRYSTAL\n0 0 0\n227\n3.56\n1\n6 0 0 0\nOPTGEOM\nMAXCYCLE\n50\nENDOPT\n"
            "END\nBASISSET\nPOB-TZVP-REV2\nFREQCALC\nRESTART\nEND\nEND\n")
    assert not opt_restart.has_optgeom_restart(deck)      # FREQCALC's RESTART is not OPTGEOM's
    new, changed = opt_restart.add_optgeom_restart(deck)
    assert changed
    assert "OPTGEOM\nRESTART\nMAXCYCLE\n50\nENDOPT" in new
    assert new.count("RESTART") == 2


def test_crlf_deck_keeps_its_line_endings():
    deck = "t\r\nCRYSTAL\r\nOPTGEOM\r\nCELLONLY\r\nENDOPT\r\nEND\r\n"
    new, _ = opt_restart.add_optgeom_restart(deck)
    assert new == "t\r\nCRYSTAL\r\nOPTGEOM\r\nCELLONLY\r\nRESTART\r\nENDOPT\r\nEND\r\n"


def test_non_opt_deck_is_left_alone():
    deck = "title\nCRYSTAL\n0 0 0\n1\n2.0\n1\n6 0 0 0\nEND\nBASISSET\nX\nEND\n"
    assert not opt_restart.has_optgeom(deck)
    assert opt_restart.add_optgeom_restart(deck) == (deck, False)


def test_scratch_dir_comes_from_the_job_script(monkeypatch, tmp_path):
    script = "export JOB=mat_opt\nexport DIR=$SLURM_SUBMIT_DIR\nexport scratch=$SCRATCH/crys23\n"
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert opt_restart.job_name(script) == "mat_opt"
    assert opt_restart.job_scratch_dir(script, "mat_opt") == tmp_path / "crys23" / "mat_opt"
    # $SCRATCH empty on this node (the agx-000 case): unknown, not "/crys23".
    monkeypatch.setenv("SCRATCH", "")
    assert opt_restart.job_scratch_dir(script, "mat_opt") is None
    monkeypatch.delenv("SCRATCH")
    assert opt_restart.job_scratch_dir(script, "mat_opt") is None


# ------------------------------------------------------------ fake SLURM

HPCC_PARTITIONS = {"mendoza_q": "7-00:00:00", "mendoza_q_long": "14-00:00:00",
                   "general-long": "7-00:00:00", "general-short": "04:00:00"}


class FakeSlurm:
    """sbatch --test-only / scontrol / sacctmgr as measured on MSU HPCC: the
    account routes the job (-A mendoza_q_long -> mendoza_q_long, otherwise
    mendoza_q), and a walltime over the partition's MaxTime is refused."""

    def __init__(self, assoc_maxwall=""):
        self.calls = []
        self.assoc_maxwall = assoc_maxwall

    def __call__(self, cmd, cwd=None):
        self.calls.append(cmd)
        if cmd[0] == "sbatch":
            t = cmd[cmd.index("-t") + 1]
            script = Path(cmd[-1]).read_text()
            acct = slurm_limits.script_account(script) or "general"
            part = "mendoza_q_long" if acct == "mendoza_q_long" else "mendoza_q"
            if slurm_limits.parse_walltime(t) > slurm_limits.parse_walltime(HPCC_PARTITIONS[part]):
                return subprocess.CompletedProcess(cmd, 1, "", "allocation failure: Requested "
                                                   "time limit is invalid (missing or exceeds some limit)\n")
            return subprocess.CompletedProcess(
                cmd, 0, "", f"sbatch: Job 17781763 to start at 2026-10-10T23:17:35 a using 32 "
                            f"processors on nodes vim-001 in partition {part}\n")
        if cmd[0] == "scontrol":
            if "-a" not in cmd:     # mendoza_q* are hidden partitions on HPCC
                return subprocess.CompletedProcess(cmd, 1, "", f"Partition {cmd[-2]} not found\n")
            p = cmd[-2]
            return subprocess.CompletedProcess(
                cmd, 0, f"PartitionName={p} AllowAccounts={p} Default=NO DefaultTime=00:01:00 "
                        f"MaxTime={HPCC_PARTITIONS[p]} MinNodes=0\n", "")
        if cmd[0] == "sacctmgr":
            return subprocess.CompletedProcess(cmd, 0, self.assoc_maxwall + "\n", "")
        raise AssertionError(cmd)


def unreachable(cmd, cwd=None):
    return None


def job_script(tmp_path, walltime, account="mendoza_q # or general", name="job"):
    p = tmp_path / f"{name}.sh"
    p.write_text("#!/bin/bash --login\n#SBATCH -J job\n#SBATCH -o job-%J.o\n"
                 f"#SBATCH --ntasks=32\n#SBATCH -A {account}\n#SBATCH -N 1\n"
                 f"#SBATCH -t {walltime}\n#SBATCH --mem-per-cpu=5G\nexport JOB={name}\n"
                 "export DIR=$SLURM_SUBMIT_DIR\nexport scratch=$SCRATCH/crys23\n")
    return p


@pytest.fixture
def engine(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))


def recover_timeout(engine, tmp_path, script, d12_text="t\nCRYSTAL\nEND\n", runner=None):
    d12 = tmp_path / f"{script.stem}.d12"
    if not d12.exists():
        d12.write_text(d12_text)
    engine.slurm_runner = runner or FakeSlurm()
    calc = {"calc_id": "T", "material_id": "M", "calc_type": "OPT",
            "input_file": str(d12), "work_dir": str(tmp_path),
            "job_script": str(script), "error_type": "timeout_error"}
    res = engine.attempt_recovery(calc, create_record=False)
    assert res is not None
    return res


def new_time(res):
    return re.search(r"#SBATCH -t (\S+)", res["fixed_job_script"].read_text()).group(1)


def test_seven_day_job_on_mendoza_q_stays_at_seven_days(engine, tmp_path, capsys, monkeypatch):
    # An OPT that can continue (a non-OPT at the limit is not resubmitted at
    # all - see test_opt_restart_abort.py).
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    _killed_opt(tmp_path, killed, monkeypatch=monkeypatch)
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00"))
    assert new_time(res) == "7-00:00:00"
    assert "already at the queue limit 7-00:00:00" in capsys.readouterr().out


def test_doubling_that_fits_is_taken_as_is(engine, tmp_path):
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert new_time(res) == "6-00:00:00"


def test_doubling_past_the_limit_is_cut_to_the_limit(engine, tmp_path):
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "5-00:00:00"))
    assert new_time(res) == "7-00:00:00"


def test_mendoza_q_long_may_go_to_fourteen_days(engine, tmp_path):
    fake = FakeSlurm()
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00", "mendoza_q_long"),
                          runner=fake)
    assert new_time(res) == "14-00:00:00"
    # Never moved to another account or partition.
    assert "#SBATCH -A mendoza_q_long" in res["fixed_job_script"].read_text()
    assert not any("-p" in c or "-A" in c for c in fake.calls if c[0] == "sbatch")


def test_mendoza_q_long_caps_at_fourteen_days(engine, tmp_path):
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "10-00:00:00", "mendoza_q_long"))
    assert new_time(res) == "14-00:00:00"


def test_slurm_unreachable_falls_back_to_the_configured_maximum(engine, tmp_path, capsys):
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "5-00:00:00"), runner=unreachable)
    assert new_time(res) == "7-00:00:00"
    assert "SLURM not reachable" in capsys.readouterr().out
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "1-00:00:00", name="b"),
                          runner=unreachable)
    assert new_time(res) == "2-00:00:00"


def test_association_maxwall_tightens_the_partition_limit(engine, tmp_path):
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "5-00:00:00"),
                          runner=FakeSlurm(assoc_maxwall="6-00:00:00"))
    assert new_time(res) == "6-00:00:00"


def test_account_and_partition_directives_parse():
    s = "#SBATCH -A mendoza_q # or general\n#SBATCH --partition=a,b\n"
    assert slurm_limits.script_account(s) == "mendoza_q"
    assert slurm_limits.script_partitions(s) == ["a", "b"]
    assert slurm_limits.script_account("#SBATCH --account mendoza_q_long\n") == "mendoza_q_long"
    assert slurm_limits.parse_walltime("UNLIMITED") is None
    assert slurm_limits.parse_walltime("14-00:00:00") == 14 * 86400


# ------------------------------------------------- the whole timeout handler

def _killed_opt(tmp_path, out_text, name="job", optinfo=True, scratch_env=True, monkeypatch=None):
    d12 = tmp_path / f"{name}.d12"
    d12.write_text(corpus(DIA_D12).read_text())
    (tmp_path / f"{name}.out").write_text(out_text)
    scratch = tmp_path / "scratch"
    (scratch / "crys23" / name).mkdir(parents=True, exist_ok=True)
    if optinfo:
        (scratch / "crys23" / name / "OPTINFO.DAT").write_text("x")
    if scratch_env:
        monkeypatch.setenv("SCRATCH", str(scratch))
    else:
        monkeypatch.setenv("SCRATCH", "")
    return d12


def test_mid_optimization_timeout_resubmits_with_restart(engine, tmp_path, monkeypatch):
    text = corpus(DIA_OUT).read_text(errors="ignore")
    killed = cut_before(text, "OPTIMIZATION - POINT    3")
    d12 = _killed_opt(tmp_path, killed, monkeypatch=monkeypatch)
    before = d12.read_text()
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert res["fixed_input_file"] == d12                 # same deck, same $JOB, same scratch
    assert opt_restart.has_optgeom_restart(d12.read_text())
    assert d12.read_text().replace("RESTART\n", "", 1) == before
    assert (tmp_path / "job.out.timeout1").read_text() == killed
    assert new_time(res) == "6-00:00:00"


def test_first_scf_timeout_resubmits_without_restart(engine, tmp_path, monkeypatch, capsys):
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "== SCF ENDED")
    d12 = _killed_opt(tmp_path, killed, optinfo=False, monkeypatch=monkeypatch)
    before = d12.read_text()
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert d12.read_text() == before
    assert res["fixed_job_script"] is not None
    assert "before the first optimization step completed" in capsys.readouterr().out


def test_missing_optinfo_in_scratch_means_no_restart(engine, tmp_path, monkeypatch):
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    d12 = _killed_opt(tmp_path, killed, optinfo=False, monkeypatch=monkeypatch)
    recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert not opt_restart.has_optgeom_restart(d12.read_text())


def test_unseen_scratch_trusts_the_output(engine, tmp_path, monkeypatch):
    """$SCRATCH empty where the recovery runs: the output decides, and the job
    script re-checks OPTINFO.DAT on the compute node."""
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    d12 = _killed_opt(tmp_path, killed, scratch_env=False, monkeypatch=monkeypatch)
    recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert opt_restart.has_optgeom_restart(d12.read_text())


def test_restart_that_times_out_again_keeps_one_restart(engine, tmp_path, monkeypatch):
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    d12 = _killed_opt(tmp_path, killed, monkeypatch=monkeypatch)
    recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    # The restarted run is killed too, before finishing a cycle of its own.
    again = cut_before(corpus(RESTARTED + ".out").read_text(errors="ignore"),
                       "COORDINATE AND CELL OPTIMIZATION - POINT   31")
    (tmp_path / "job.out").write_text(again)
    recover_timeout(engine, tmp_path, job_script(tmp_path, "6-00:00:00"))
    assert d12.read_text().count("RESTART") == 1
    assert (tmp_path / "job.out.timeout1").read_text() == killed
    assert (tmp_path / "job.out.timeout2").read_text() == again


def test_non_opt_timeout_is_unchanged(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    res = recover_timeout(engine, tmp_path, job_script(tmp_path, "1-00:00:00"))
    assert (tmp_path / "job.d12").read_text() == "t\nCRYSTAL\nEND\n"
    assert not list(tmp_path.glob("*.timeout*"))
    assert new_time(res) == "2-00:00:00"


# ------------------------------------------------- job-script staging (bash)

def _staging_blocks(tmp_path):
    """The RESTART and GUESSP staging of a REALLY generated job script, from
    the RESTART block up to `cd $scratch/$JOB` (the template is a generator;
    asserting on its text would not prove what runs - a stray apostrophe in a
    comment there once turned the generator itself into a broken script)."""
    gen = tmp_path / "submitcrystal23.sh"
    shutil.copy(SUBMIT_SH, gen)
    gen.write_text(gen.read_text().replace("\nsbatch ", "\n#sbatch "))
    subprocess.run(["bash", str(gen), "testmat"], cwd=tmp_path, capture_output=True, text=True)
    script = (tmp_path / "testmat.sh").read_text()
    assert subprocess.run(["bash", "-n", str(tmp_path / "testmat.sh")]).returncode == 0
    start = script.index("# OPTGEOM RESTART")
    # Before GUESSP staging, which must not overwrite the killed run's fort.20.
    assert start < script.index("# GUESSP restart")
    end = script.index("\ncd $scratch/$JOB", start)
    return script[start:end + 1]


DECK = ("title\nCRYSTAL\n0 0 0\n1\n2.0\n1\n6 0 0 0\nOPTGEOM\nFULLOPTG\nRESTART\nMAXCYCLE\n"
        "800\nENDOPT\nEND\nBASISSET\nPOB-TZVP-REV2\nFREQCALC\nRESTART\nEND\nEND\n")


def _run_blocks(tmp_path, tag, deck, scratch_files, dir_files=None):
    block = _staging_blocks(tmp_path)
    d = tmp_path / tag
    scratch = d / "scratch" / "testmat"
    scratch.mkdir(parents=True)
    (d / "testmat.d12").write_text(deck)
    (scratch / "INPUT").write_text(deck)
    for name, content in scratch_files.items():
        (scratch / name).write_text(content)
    for name, content in (dir_files or {}).items():
        (d / name).write_text(content)
    prelude = f'DIR="{d}"\nJOB=testmat\nscratch="{d}/scratch"\n'
    r = subprocess.run(["bash", "-c", prelude + block], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    f20 = scratch / "fort.20"
    return (scratch / "INPUT").read_text(), (f20.read_text() if f20.exists() else None), r.stdout


def test_killed_runs_own_fort20_is_kept(tmp_path):
    """Measured on HPCC: a killed OPT leaves fort.9 EMPTY and the last SCF's
    matrix in fort.20 - copying fort.9 over it would destroy the guess."""
    deck, f20, out = _run_blocks(tmp_path, "ok", DECK,
                                 {"OPTINFO.DAT": "steps", "fort.9": "", "fort.20": "LAST-SCF"})
    assert f20 == "LAST-SCF"
    assert deck == DECK
    assert "RESTART: continuing" in out


def test_guessp_staging_stands_aside_for_a_restart(tmp_path):
    guessp = DECK.replace("END\nBASISSET", "END\nBASISSET", 1).replace(
        "POB-TZVP-REV2\n", "POB-TZVP-REV2\nGUESSP\n")
    deck, f20, out = _run_blocks(tmp_path, "gp", guessp,
                                 {"OPTINFO.DAT": "steps", "fort.20": "LAST-SCF"},
                                 {"testmat.f9": "EARLIER-RUN"})
    assert f20 == "LAST-SCF"
    assert "GUESSP" in deck and "GUESSP:" not in out


def test_nonempty_fort9_is_used_when_fort20_is_missing(tmp_path):
    deck, f20, out = _run_blocks(tmp_path, "f9", DECK, {"OPTINFO.DAT": "s", "fort.9": "FINAL"})
    assert f20 == "FINAL"


def test_job_script_drops_restart_without_optinfo(tmp_path):
    deck, f20, out = _run_blocks(tmp_path, "cold", DECK, {"fort.9": "X"})
    assert f20 is None
    assert "OPTGEOM\nFULLOPTG\nMAXCYCLE" in deck               # OPTGEOM's RESTART gone
    assert "FREQCALC\nRESTART\nEND" in deck                     # FREQCALC's untouched
    assert "no OPTINFO.DAT" in out


def test_job_script_ignores_decks_without_optgeom_restart(tmp_path):
    plain = DECK.replace("FULLOPTG\nRESTART\n", "FULLOPTG\n")
    deck, f20, out = _run_blocks(tmp_path, "plain", plain, {"OPTINFO.DAT": "s", "fort.9": "X"})
    assert deck == plain and f20 is None and out == ""


# ------------------------------------------- the queue manager's real path

def test_walltime_killed_opt_is_recovered_with_restart_by_the_manager(monkeypatch, tmp_path):
    """End to end through the queue manager's own status sweep: sacct says
    TIMEOUT, the .out is merely cut off (as on HPCC - the time-limit notice
    goes to the -o log, not the .out), and the job is resubmitted with the
    bumped script and RESTART in the same deck."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    shutil.copy2(corpus(DIA_D12), tmp_path / "job.d12")
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    scratch = tmp_path / "scr"
    (scratch / "crys23" / "job").mkdir(parents=True)
    (scratch / "crys23" / "job" / "OPTINFO.DAT").write_text("steps")
    monkeypatch.setenv("SCRATCH", str(scratch))

    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted = []
    mgr.submit_to_slurm = (lambda input_file, work_dir, calc_type, submit_script_override=None:
                           submitted.append((input_file, submit_script_override)) or
                           str(777001 + len(submitted) - 1))
    calc_id = mgr.submit_calculation(tmp_path / "job.d12")
    job_script(tmp_path, "3-00:00:00")                    # the generated job.sh
    (tmp_path / "job.out").write_text(killed)
    (tmp_path / "job-777001.o").write_text(
        "[2026-09-25T12:53:46.235] error: *** JOB 777001 ON amr-026 CANCELLED AT "
        "2026-09-25T12:53:46 DUE TO TIME LIMIT ***\n")
    mgr.error_recovery_engine.slurm_runner = FakeSlurm()

    def fake_run(cmd, capture_output=True, text=True, **kw):
        out = "JOBID,STATE,START\n" if cmd[0] == "squeue" else "TIMEOUT \n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(qm.subprocess, "run", fake_run)

    mgr.check_queue_status()

    old = mgr.db.get_calculation(calc_id)
    assert old["status"] == "resubmitted", old
    assert len(submitted) == 2
    resub_input, resub_script = submitted[1]
    assert Path(resub_input).name == "job.d12"
    assert "#SBATCH -t 6-00:00:00" in Path(resub_script).read_text()
    assert opt_restart.has_optgeom_restart((tmp_path / "job.d12").read_text())
    assert (tmp_path / "job.out.timeout1").read_text() == killed


def test_time_limit_log_alone_marks_a_timeout(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    calc = {"slurm_job_id": "42", "work_dir": str(tmp_path)}
    assert mgr._walltime_kill_evidence(calc, "NOT_IN_QUEUE") is None
    (tmp_path / "job-42.o").write_text("*** JOB 42 ON x CANCELLED AT t DUE TO TIME LIMIT ***\n")
    assert "TIME LIMIT" in mgr._walltime_kill_evidence(calc, "NOT_IN_QUEUE")
    assert mgr._walltime_kill_evidence({}, "TIMEOUT")


def test_enable_restart_false_turns_it_off(engine, tmp_path, monkeypatch):
    killed = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3")
    d12 = _killed_opt(tmp_path, killed, monkeypatch=monkeypatch)
    engine.config["error_recovery"]["timeout_error"]["enable_restart"] = False
    recover_timeout(engine, tmp_path, job_script(tmp_path, "3-00:00:00"))
    assert not opt_restart.has_optgeom_restart(d12.read_text())
