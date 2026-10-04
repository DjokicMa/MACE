"""An optimization that FINALRUN keeps re-starting between two geometries.

Measured on CRYSTAL23 (Si, B3LYP-D3, POB-TZVP-REV2, TOLINTEG 7 7 7 7 14):
with the default FINALRUN 4 the optimization converges, FINALRUN re-starts it
("FINALRUN OPTION: RE-STARTING OPTIMIZATION"), the reference geometry is
changed, the energy jumps by about 2.2E-4 Ha - alternately down and up - and
it converges again to the other of two states, forever inside one job. With
OPTGEOM MAXCYCLE reached the .out ends "* OPT END - FAILED * ... POINTS n"
followed by MPI_Abort. TOLINTEG 8 8 8 9 24 converges in 4 points; SPIN plays
no part. The outputs below are synthetic excerpts of that run's form.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from mace.completion_checker import categorize_output_file
from mace.recovery import opt_restart
from test_opt_timeout_restart import job_script

TITLE = "si_loop_test"
STAR = " " + "*" * 79

DECK = f"""{TITLE}
CRYSTAL
0 0 0
227
5.43100000
1
14 0.0000000000 0.0000000000 0.0000000000 Biso 1.000000 Si
OPTGEOM
FULLOPTG
MAXCYCLE
800
TOLDEG
0.0003
TOLDEX
0.0012
TOLDEE
7
ENDOPT
BASISSET
POB-TZVP-REV2
DFT
SPIN
B3LYP-D3
XLGRID
ENDDFT
TOLINTEG
7 7 7 7 14
TOLDEE
7
SHRINK
0 16
8 8 8
SCFDIR
MAXCYCLE
800
FMIXING
30
DIIS
HISTDIIS
100
PPAN
END
"""


def _geometry(a):
    p = a / 2 ** 0.5
    return f"""{STAR}
 LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM
 PRIMITIVE CELL - CENTRING CODE 5/0 VOLUME=    40.000000 - DENSITY  2.300 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     {p:.8f}     {p:.8f}     {p:.8f}    60.000000  60.000000  60.000000
{STAR}
 ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:    2
     ATOM                 X/A                 Y/B                 Z/C
{STAR}
      1 T  14 SI    0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
      2 F  14 SI    2.500000000000E-01  2.500000000000E-01  2.500000000000E-01

{STAR}
 CRYSTALLOGRAPHIC CELL (VOLUME=        160.00000000)
         A              B              C           ALPHA      BETA       GAMMA
     {a:.8f}     {a:.8f}     {a:.8f}    90.000000  90.000000  90.000000

 COORDINATES IN THE CRYSTALLOGRAPHIC CELL
     ATOM                 X/A                 Y/B                 Z/C
{STAR}
      1 T  14 SI    0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
      2 F  14 SI    2.500000000000E-01 -2.500000000000E-01 -2.500000000000E-01

 T = ATOM BELONGING TO THE ASYMMETRIC UNIT
"""


HEADER = f"""{STAR}
 {TITLE}
 CRYSTAL CALCULATION
 SPACE GROUP (CENTROSYMMETRIC)        :  F D 3 M

 GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    3
 (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)
{_geometry(5.431)}
 ****   48 SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS
 INFORMATION **** NEW DEFAULT **** FINALRUN 4 FOR GEOMETRY OPTIMIZATIONS
{STAR}
 *                             OPTIMIZATION STARTS                             *
{STAR}
"""

# The two states the measured run alternated between, and its energy steps.
A_LOW, E_LOW = 5.43660, -578.99031   # reached after a step DOWN by ~2.2E-4
A_HIGH, E_HIGH = 5.43600, -578.99008
SATISFIED = "\n CONVERGENCE TESTS SATISFIED AFTER {n:3d} ENERGY AND GRADIENT CALCULATIONS\n"
RESTART = (" " + "+" * 79 + "\n FINALRUN OPTION: RE-STARTING OPTIMIZATION\n"
           + " " + "+" * 79 + "\n")
ABORT = ("Abort(1) on node 2 (rank 2 in comm 0): application called "
         "MPI_Abort(MPI_COMM_WORLD, 1) - process 2\n")


def _point(n, a, energy, de, new_reference=False):
    ref = "\n REFERENCE GEOMETRY WILL BE CHANGED\n" if new_reference else ""
    return (f"{STAR}\n CELL OPTIMIZATION - POINT {n:4d}\n{_geometry(a)}{ref}\n"
            f" TOTAL ENERGY(DFT)(AU)(  3) {energy:.13E}     DE (AU)   {de:.3E}\n\n"
            " MAX GRADIENT      0.001861  THRESHOLD              0.000450 CONVERGED NO\n")


def loop_output(restarts=5, end="maxcycle", jumps=None, title=True):
    """A run FINALRUN re-started `restarts` times. `jumps` are the energy
    changes right after each re-start (default: the measured alternation).
    `end`: "maxcycle" (OPT END - FAILED, MPI_Abort), "running" (still
    writing), "killed" (cut off mid-line by the walltime) or "converged"."""
    jumps = jumps or [(-2.223e-4, 2.252e-4)[k % 2] for k in range(restarts)]
    restarts = len(jumps)
    out = HEADER if title else HEADER.replace(TITLE, "another_deck")
    out += (f"{STAR}\n CELL OPTIMIZATION - POINT    1\n"
            " TOTAL ENERGY(DFT)(AU)(  6) -5.7898784078044E+02 DE-2.3E-09 tester 2.4E-12\n")
    out += _point(2, A_HIGH, E_HIGH, -2.776e-6) + _point(3, A_HIGH, E_HIGH - 1e-7, -7.7e-8)
    n = 3
    for k in range(restarts):
        out += SATISFIED.format(n=n) + RESTART
        low = jumps[k] < 0
        a, e = (A_LOW, E_LOW) if low else (A_HIGH, E_HIGH)
        out += _point(n + 1, a, e, jumps[k], new_reference=True)
        out += _point(n + 2, a, e - 1e-6, -1.0e-6)
        n += 2
    if end == "maxcycle":
        out += (f" CONVERGENCE TESTS UNSATISFIED AFTER {n:3d} ENERGY AND GRADIENT CALCULATIONS\n\n"
                " " + "*" * 66 + "\n"
                f" * OPT END - FAILED    * E(AU):  {E_LOW:.12E}  POINTS {n:4d} *\n"
                " " + "*" * 66 + "\n\n EIGENVECTORS IN FORTRAN UNIT  8\n" + ABORT)
    elif end == "killed":
        out += " SYMMETRY ALLOWED FORC"
    elif end == "converged":
        out += (SATISFIED.format(n=n) + "\n * OPT END - CONVERGED * E(AU):  "
                f"{E_LOW:.12E}  POINTS {n:4d} *\n EEEEEEEEEE TERMINATION\n")
    return out


def unconverged_output():
    """MAXCYCLE reached without any FINALRUN re-start."""
    out = HEADER + f"{STAR}\n CELL OPTIMIZATION - POINT    1\n"
    out += _point(2, A_HIGH, E_HIGH, -2.0e-5) + _point(3, A_LOW, E_HIGH - 3e-5, -3.0e-5)
    return out + (" CONVERGENCE TESTS UNSATISFIED AFTER   3 ENERGY AND GRADIENT CALCULATIONS\n\n"
                  f" * OPT END - FAILED    * E(AU):  {E_HIGH - 3e-5:.12E}  POINTS    3 *\n\n"
                  + ABORT)


# ------------------------------------------------------------ detection

@pytest.mark.parametrize("end", ["maxcycle", "running", "killed"])
def test_loop_is_recognized_however_the_run_ended(end):
    why = opt_restart.finalrun_loop(loop_output(end=end))
    assert why and "5 FINALRUN re-starts" in why


def test_two_restarts_are_not_a_loop():
    assert opt_restart.finalrun_loop(loop_output(restarts=2)) is None


def test_restarts_that_do_not_alternate_are_not_a_loop():
    assert opt_restart.finalrun_loop(loop_output(jumps=[-2e-4, -2e-4, -2e-4])) is None
    assert opt_restart.finalrun_loop(loop_output(jumps=[-2e-4, 2e-6, -2e-4])) is None


def test_a_loop_that_finally_converged_is_not_an_error():
    assert opt_restart.finalrun_loop(loop_output(end="converged")) is None


def test_loop_found_in_its_trailing_restarts():
    # A first, ordinary re-start, then the alternation.
    why = opt_restart.finalrun_loop(loop_output(jumps=[-5e-6, -2.2e-4, 2.2e-4, -2.2e-4]))
    assert why and "last 3" in why


def _detector(tmp_path):
    from mace.recovery.detector import CrystalErrorDetector
    return CrystalErrorDetector(base_dir=str(tmp_path), db_path=str(tmp_path / "d.db"),
                                enable_tracking=False)


@pytest.mark.parametrize("end", ["maxcycle", "running", "killed"])
def test_detector_reports_the_loop(tmp_path, end):
    out = tmp_path / "si.out"
    out.write_text(loop_output(end=end))
    res = _detector(tmp_path).analyze_output_file(out)
    assert res["status"] == "error"
    assert res["error_type"] == "finalrun_loop"
    assert res["recoverable"] is True


def test_detector_reports_an_unconverged_optimization(tmp_path):
    out = tmp_path / "si.out"
    out.write_text(unconverged_output())
    res = _detector(tmp_path).analyze_output_file(out)
    assert res["error_type"] == "opt_unconverged_error"
    assert res["status"] == "error"


# ------------------------------------------------- OPT END - FAILED

@pytest.mark.parametrize("text", [unconverged_output(), loop_output()], ids=["plain", "loop"])
def test_opt_end_failed_is_an_unconverged_optimization_not_a_runtime_error(tmp_path, text):
    out = tmp_path / "si.out"
    out.write_text(text)
    assert categorize_output_file(out)[0] == "opt_unconverged"


def test_mace_completion_reports_it_as_unconverged(tmp_path):
    import os
    import sys
    (tmp_path / "si.out").write_text(unconverged_output())
    cli = Path(__file__).resolve().parent.parent / "mace_cli"
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(cli), "completion"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Geometry optimization not converged (OPT END - FAILED): 1" in r.stdout
    assert "runtime error" not in r.stdout


def test_converged_optimization_is_still_complete(tmp_path):
    out = tmp_path / "si.out"
    out.write_text(loop_output(end="converged"))
    assert categorize_output_file(out)[0] == "complete"


@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)


def _classify(mgr, tmp_path, text):
    (tmp_path / "si.out").write_text(text)
    return mgr.analyze_calculation_error({"output_file": str(tmp_path / "si.out"),
                                          "input_file": str(tmp_path / "si.d12")})


def test_manager_classifies_the_loop_and_the_unconverged_run(mgr, tmp_path):
    assert _classify(mgr, tmp_path, loop_output())[0] == "finalrun_loop"
    assert _classify(mgr, tmp_path, unconverged_output())[0] == "opt_unconverged_error"


def test_walltime_kill_of_a_loop_stays_a_loop(mgr, tmp_path):
    """Resubmitting a looping run with more walltime (the timeout recovery)
    would only loop longer."""
    (tmp_path / "si.out").write_text(loop_output(end="killed"))
    cid = mgr.db.create_calculation(material_id="si", calc_type="OPT",
                                    input_file=str(tmp_path / "si.d12"), work_dir=str(tmp_path))
    mgr.db.update_calculation_status(cid, "failed", output_file=str(tmp_path / "si.out"))
    mgr.enable_error_recovery = False
    mgr.handle_failed_calculation(cid, "TIMEOUT")
    assert mgr.db.get_calculation(cid)["error_type"] == "finalrun_loop"


def test_both_are_recoverable_with_their_own_retry_budget(mgr):
    cfg = mgr.error_recovery_engine.config["error_recovery"]
    assert cfg["finalrun_loop"]["handler"] == "finalrun_loop_handler"
    assert cfg["opt_unconverged_error"]["handler"] == "opt_fresh_start_handler"
    shipped = Path(__file__).resolve().parent.parent / "mace" / "config" / "recovery_config.yaml"
    import yaml
    data = yaml.safe_load(shipped.read_text())["error_recovery"]
    for key in ("finalrun_loop", "opt_unconverged_error"):
        assert data[key]["handler"] == cfg[key]["handler"]
        assert data[key]["max_retries"] == cfg[key]["max_retries"]


# --------------------------------------------------------- TOLINTEG

@pytest.mark.parametrize("before, after", [
    ("7 7 7 7 14", "8 8 8 9 24"),      # Standard -> Tight
    ("8 8 7 7 14", "8 8 8 9 24"),      # Coulomb-only tightening -> Tight
    ("8 8 8 9 24", "9 9 9 11 38"),     # Tight -> Very tight
    ("7 7 7 7 30", "8 8 8 9 30"),      # nothing is ever lowered
    ("6 6 6 6 12", "7 7 7 7 14"),      # CRYSTAL's default -> Standard
])
def test_tolinteg_goes_up_one_preset_level(before, after):
    deck = DECK.replace("7 7 7 7 14", before)
    new, old, raised = opt_restart.raise_tolinteg(deck)
    assert (old, raised) == (before, after)
    assert new == DECK.replace("7 7 7 7 14", after)


def test_deck_without_tolinteg_gets_the_standard_level():
    deck = DECK.replace("TOLINTEG\n7 7 7 7 14\n", "")
    new, old, raised = opt_restart.raise_tolinteg(deck)
    assert old is None and raised == "7 7 7 7 14"
    assert new == deck.replace("PPAN\nEND\n", "PPAN\nTOLINTEG\n7 7 7 7 14\nEND\n")


def test_nothing_above_very_tight():
    deck = DECK.replace("7 7 7 7 14", "9 9 9 11 38")
    new, old, raised = opt_restart.raise_tolinteg(deck)
    assert new == deck and raised is None


# ------------------------------------------------------------ recovery

@pytest.fixture
def engine(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))


def _looped_job(tmp_path, deck=DECK, out=None):
    (tmp_path / "si.d12").write_text(deck)
    (tmp_path / "si.out").write_text(out if out is not None else loop_output())
    return tmp_path / "si.d12", job_script(tmp_path, "1-00:00:00", name="si")


def _recover(engine, d12, script, error_type="finalrun_loop"):
    calc = {"calc_id": "L", "material_id": "si", "calc_type": "OPT",
            "input_file": str(d12), "work_dir": str(d12.parent),
            "job_script": str(script), "error_type": error_type}
    return engine.attempt_recovery(calc, create_record=False)


def test_loop_restarts_fresh_from_the_best_point_with_tighter_tolinteg(engine, tmp_path, capsys):
    d12, script = _looped_job(tmp_path)
    res = _recover(engine, d12, script)
    assert res["fixed_input_file"] == d12 and res["fixed_job_script"] == script
    # The lowest-energy point's cell and TOLINTEG one level up; every other
    # record (SPIN, TOLDEE, SHRINK, FMIXING, OPTGEOM ...) as it was.
    assert d12.read_text() == DECK.replace("\n5.43100000\n", f"\n{A_LOW:.8f}\n").replace(
        "7 7 7 7 14", "8 8 8 9 24")
    assert (tmp_path / "si.d12.orig").read_text() == DECK
    assert (tmp_path / "si.out.finalrunloop1").read_text() == loop_output()
    assert "TOLINTEG 7 7 7 7 14 -> 8 8 8 9 24" in capsys.readouterr().out


def test_loop_at_the_deck_geometry_still_tightens_tolinteg(engine, tmp_path):
    # No point carries over (the run is of another deck): the deck keeps its
    # geometry, loses RESTART, and only TOLINTEG changes.
    deck = DECK.replace("FULLOPTG\n", "FULLOPTG\nRESTART\n")
    d12, script = _looped_job(tmp_path, deck=deck, out=loop_output(title=False))
    assert _recover(engine, d12, script)
    assert d12.read_text() == DECK.replace("7 7 7 7 14", "8 8 8 9 24")
    assert (tmp_path / "si.d12.orig").read_text() == deck


def test_loop_at_very_tight_is_not_resubmitted(engine, tmp_path, capsys):
    deck = DECK.replace("7 7 7 7 14", "9 9 9 11 38")
    d12, script = _looped_job(tmp_path, deck=deck)
    assert _recover(engine, d12, script) is None
    assert d12.read_text() == deck
    assert "no higher TOLINTEG level" in capsys.readouterr().out


def test_unconverged_optimization_continues_from_the_best_point(engine, tmp_path):
    d12, script = _looped_job(tmp_path, out=unconverged_output())
    assert _recover(engine, d12, script, "opt_unconverged_error")
    assert d12.read_text() == DECK.replace("\n5.43100000\n", f"\n{A_LOW:.8f}\n")


def test_loop_is_recovered_by_the_manager(monkeypatch, tmp_path):
    """End to end through the status sweep: sacct says COMPLETED, the .out
    ends OPT END - FAILED after five FINALRUN re-starts - the job goes back
    in with the best geometry and TOLINTEG raised, the script unchanged."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    (tmp_path / "si.d12").write_text(DECK)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted = []
    mgr.submit_to_slurm = (lambda input_file, work_dir, calc_type, submit_script_override=None:
                           submitted.append((input_file, submit_script_override)) or
                           str(999001 + len(submitted) - 1))
    calc_id = mgr.submit_calculation(tmp_path / "si.d12")
    script = job_script(tmp_path, "1-00:00:00", name="si")
    (tmp_path / "si.out").write_text(loop_output())

    def fake_run(cmd, capture_output=True, text=True, **kw):
        out = "JOBID,STATE,START\n" if cmd[0] == "squeue" else "COMPLETED \n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(qm.subprocess, "run", fake_run)

    mgr.check_queue_status()

    old = mgr.db.get_calculation(calc_id)
    assert old["status"] == "resubmitted", old
    assert old["error_type"] == "finalrun_loop"
    assert len(submitted) == 2
    assert Path(submitted[1][0]).name == "si.d12"
    assert Path(submitted[1][1]) == script
    new = (tmp_path / "si.d12").read_text()
    assert "\nTOLINTEG\n8 8 8 9 24\n" in new and "\nSPIN\n" in new
