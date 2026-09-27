"""An OPTGEOM RESTART rerun that aborts on a collapsed step size continues as a
fresh optimization from the best geometry reached; the original deck is kept.

Measured on HPCC (tests/data/opt_restart, trimmed copies of the real files):
tqb_pto (PbTiO3, P m-3m, cell-only OPT) timed out after four points; its
RESTART rerun re-evaluated the best point (point 2's geometry), saw
DE -1.7E-08, set the trust radius to 0.000E+00 and died in MPI_Abort. The
.out ends there; "ERROR **** BFGS_ **** PXK TOO SMALL" is only in the scratch
fort.87, and sacct says COMPLETED - so it was classified unknown_error and the
deck kept RESTART forever. tqc_agbr (Ag2Br3, R -3 c in hexagonal axes)
supplies a centred-cell RESTART run whose best point is in the EARLIER run.

The geometry transfer is also checked against every 3D OPT pair of the real
corpus and against the SP decks MACE's CRYSTALOptToD12 made from them.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import TEST_DATA
from mace.recovery import opt_geometry, opt_restart
from test_opt_timeout_restart import (FakeSlurm, SUBMIT_SH, corpus, cut_before, job_script,
                                      DIA_D12, DIA_OUT)

DATA = Path(__file__).resolve().parent / "data" / "opt_restart"


def data(name):
    return (DATA / name).read_text()


def _same_atoms(a, b, tol=1e-9):
    return len(a) == len(b) and all(
        x[0] == y[0] and all(abs(p - q) <= tol for p, q in zip(x[1:], y[1:]))
        for x, y in zip(a, b))


def opt_pairs():
    """Every real 3D (CRYSTAL keyword) OPT output with its deck."""
    if not (TEST_DATA / "OPT").is_dir():
        pytest.skip("test/ corpus not present")
    pairs = []
    for out in sorted((TEST_DATA / "OPT").glob("*.out")):
        d12 = out.with_suffix(".d12")
        if not d12.is_file():
            continue
        deck = d12.read_text(errors="ignore")
        text = out.read_text(errors="ignore")
        if "OPTIMIZATION - POINT" in text and deck.splitlines()[1].strip() == "CRYSTAL":
            pairs.append((out, deck, text))
    return pairs


# ---------------------------------------------------------------- detection

def test_restart_abort_is_recognised_from_the_output_alone():
    msg = opt_restart.trust_radius_abort(data("tqb_pto.out"))
    assert msg and "RESTART" in msg and "trust radius 0" in msg


def test_fort87_names_the_real_error():
    msg = opt_restart.trust_radius_abort(data("tqb_pto.out"), data("tqb_pto.fort.87"))
    assert "PXK TOO SMALL" in msg


def test_trust_radius_information_alone_is_not_an_abort():
    """"TOO SMALL TRUST RADIUS" is printed by converged runs too, and killed or
    finished runs are not aborts: no corpus output and none of the other HPCC
    runs count."""
    assert "TOO SMALL TRUST RADIUS" in corpus(
        "OPT/TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_"
        "B3LYP-D3-D3_optimized.out").read_text(errors="ignore")
    for name in ("tqb_pto.out.timeout1", "tqc_agbr.out", "tqc_agbr.out.timeout1"):
        assert opt_restart.trust_radius_abort(data(name)) is None, name
    hits = [p.name for p in TEST_DATA.rglob("*.out")
            if opt_restart.trust_radius_abort(p.read_text(errors="ignore"))]
    assert hits == []


def _scratch_job(tmp_path, monkeypatch, name, fort87, fort87_is_stale=False):
    scratch = tmp_path / "scratch" / "crys23" / name
    scratch.mkdir(parents=True)
    (scratch / "INPUT").write_text("deck")
    (scratch / "fort.87").write_text(fort87)
    if fort87_is_stale:           # left by an earlier run: older than this run's INPUT
        old = time.time() - 3600
        os.utime(scratch / "fort.87", (old, old))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    return scratch


@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)


def _classify(mgr, tmp_path, name, out_text):
    (tmp_path / f"{name}.out").write_text(out_text)
    script = job_script(tmp_path, "1-00:00:00", name=name)
    return mgr.analyze_calculation_error({"output_file": str(tmp_path / f"{name}.out"),
                                          "input_file": str(tmp_path / f"{name}.d12"),
                                          "job_script": str(script)})


def test_manager_classifies_the_abort(mgr, tmp_path, monkeypatch):
    _scratch_job(tmp_path, monkeypatch, "tqb_pto", data("tqb_pto.fort.87"))
    error_type, msg = _classify(mgr, tmp_path, "tqb_pto", data("tqb_pto.out"))
    assert error_type == "opt_trust_radius_error"
    assert "PXK TOO SMALL" in msg


def test_fort87_error_gets_a_real_error_type(mgr, tmp_path, monkeypatch):
    """An .out that stops at MPI_Abort with the error only in fort.87."""
    truncated = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "== SCF ENDED") + \
        "Abort(1) on node 3 (rank 3 in comm 0): application called MPI_Abort\n"
    _scratch_job(tmp_path, monkeypatch, "job", " ERROR **** NEIGHB **** DISTANCE TOO SMALL\n")
    assert _classify(mgr, tmp_path, "job", truncated)[0] == "geometry_error"


def test_unmatched_fort87_error_is_reported_in_crystals_words(mgr, tmp_path, monkeypatch):
    truncated = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "== SCF ENDED")
    _scratch_job(tmp_path, monkeypatch, "job", " ERROR **** SOMETHING **** NEW\n")
    assert _classify(mgr, tmp_path, "job", truncated) == \
        ("crystal_error", "fort.87: ERROR **** SOMETHING **** NEW")


def test_stale_fort87_is_ignored(mgr, tmp_path, monkeypatch):
    truncated = cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "== SCF ENDED")
    _scratch_job(tmp_path, monkeypatch, "job", " ERROR **** SOMETHING **** OLD\n",
                 fort87_is_stale=True)
    assert _classify(mgr, tmp_path, "job", truncated)[0] == "unknown_error"


# ------------------------------------------------------ the best geometry

def test_point_one_energy_is_what_point_two_was_measured_from():
    pts = opt_geometry.optimization_points(data("tqb_pto.out.timeout1"))
    assert [p.number for p in pts] == [1, 2, 3, 4]          # point 5 was killed
    p1, p2 = pts[0], pts[1]
    assert p1.geometry is None
    assert p1.energy == pytest.approx(-1.2683449049240E+03 + 3.622E-04, abs=1e-9)
    assert p2.geometry.cell[0] == 3.92491474


def test_best_point_across_the_killed_and_the_aborted_run():
    pts = (opt_geometry.optimization_points(data("tqb_pto.out"), "tqb_pto.out") +
           opt_geometry.optimization_points(data("tqb_pto.out.timeout1"), "timeout1"))
    best = opt_geometry.best_point(pts)
    # The RESTART run's re-evaluation of point 2's geometry: -1268.3449049414.
    assert (best.source, best.number, best.energy) == ("tqb_pto.out", 4, -1.2683449049414E+03)
    assert best.geometry.cell[:3] == (3.92491474,) * 3


def test_every_real_3d_opt_transfers_its_last_point_exactly():
    """For every 3D OPT in the corpus: the header ties deck and output frames
    together, re-writing the header geometry reproduces the deck's numbers,
    only the cell line and atom lines change, and the last point equals
    CRYSTAL's FINAL OPTIMIZED GEOMETRY (covers P, C, F and R lattices, origin
    choices, comments after the cell parameters, 1/2/3/4-parameter cells)."""
    pairs = opt_pairs()
    assert len(pairs) > 30
    for out, deck, text in pairs:
        header = opt_geometry.header_geometry(text)
        points = [p for p in opt_geometry.optimization_points(text) if p.geometry]
        if not points:
            continue
        same = opt_geometry.rewrite_geometry(deck, header, header)
        assert _same_atoms(opt_geometry.parse_deck_geometry(same.splitlines()).atoms,
                           opt_geometry.parse_deck_geometry(deck.splitlines()).atoms), out.name
        new = opt_geometry.rewrite_geometry(deck, header, points[-1].geometry)
        g = opt_geometry.parse_deck_geometry(deck.splitlines())
        changed = {i for i, (a, b) in enumerate(zip(deck.splitlines(), new.splitlines())) if a != b}
        assert changed <= {g.cell_line, *g.atom_lines}, out.name
        assert len(new.splitlines()) == len(deck.splitlines())
        if "FINAL OPTIMIZED GEOMETRY" in text:
            lines = text.splitlines()
            i = next(k for k, l in enumerate(lines) if "FINAL OPTIMIZED GEOMETRY" in l)
            final = opt_geometry.read_geometry(lines, i, len(lines))
            assert final.cell == points[-1].geometry.cell, out.name
            assert final.atoms == points[-1].geometry.atoms, out.name


def test_transferred_geometry_matches_maces_own_sp_decks():
    """CRYSTALOptToD12 wrote an SP deck from the final geometry of many corpus
    OPTs; carrying the last point into the OPT deck gives the same cell and
    atoms (up to lattice/centring images)."""
    compared = 0
    for out, deck, text in opt_pairs():
        sps = sorted((TEST_DATA / "SP").glob(out.stem + "_sp_*.d12"))
        points = [p for p in opt_geometry.optimization_points(text) if p.geometry]
        if not sps or not points:
            continue
        header = opt_geometry.header_geometry(text)
        new = opt_geometry.parse_deck_geometry(
            opt_geometry.rewrite_geometry(deck, header, points[-1].geometry).splitlines())
        sp = opt_geometry.parse_deck_geometry(sps[0].read_text().splitlines())
        assert new.cell == pytest.approx(sp.cell, abs=1e-6), out.name
        images = opt_geometry._CENTRING_TRANSLATIONS[header.lattice]
        for a, b in zip(new.atoms, sp.atoms):
            assert opt_geometry._nearest_image(a[1:], b[1:], images)[1] < 1e-6, out.name
        compared += 1
    assert compared >= 20


def test_non_3d_or_foreign_decks_are_refused():
    header = opt_geometry.header_geometry(data("tqb_pto.out"))
    molecule = "title\nMOLECULE\n1\n3\n8 0 0 0\n1 0 0 1\n1 1 0 0\nOPTGEOM\nENDOPT\nEND\n"
    with pytest.raises(opt_geometry.GeometryTransferError):
        opt_geometry.rewrite_geometry(molecule, header, header)
    other = data("tqb_pto.d12").replace("3.94649838", "4.10000000")
    with pytest.raises(opt_geometry.GeometryTransferError):
        opt_geometry.rewrite_geometry(other, header, header)


# ----------------------------------------------------- the fallback handler

@pytest.fixture
def engine(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))


def _aborted_job(tmp_path, name, runs):
    for fname in [f"{name}.d12", *runs]:
        shutil.copy(DATA / fname, tmp_path / fname)
    script = job_script(tmp_path, "0-00:24:00", name=name)
    return tmp_path / f"{name}.d12", script


def _recover_abort(engine, d12, script):
    calc = {"calc_id": "A", "material_id": "M", "calc_type": "OPT",
            "input_file": str(d12), "work_dir": str(d12.parent),
            "job_script": str(script), "error_type": "opt_trust_radius_error"}
    return engine.attempt_recovery(calc, create_record=False)


def test_fallback_deck_restarts_fresh_from_the_best_point(engine, tmp_path):
    d12, script = _aborted_job(tmp_path, "tqb_pto", ["tqb_pto.out", "tqb_pto.out.timeout1"])
    original = d12.read_text()
    res = _recover_abort(engine, d12, script)
    assert res["fixed_input_file"] == d12                  # same deck, same job name
    assert res["fixed_job_script"] == script               # same script, same walltime
    new = d12.read_text()
    assert (tmp_path / "tqb_pto.d12.orig").read_text() == original
    assert not opt_restart.has_optgeom_restart(new)
    # Only the cell line differs (cubic: a) and RESTART is gone; every other
    # record is byte-identical.
    expected = original.replace("\n3.94649838\n", "\n3.92491474\n").replace(
        "FULLOPTG\nRESTART\n", "FULLOPTG\n")
    assert new == expected
    assert (tmp_path / "tqb_pto.out.optabort1").read_text() == data("tqb_pto.out")


def test_centred_cell_takes_the_earlier_runs_best_point(engine, tmp_path):
    d12, script = _aborted_job(tmp_path, "tqc_agbr", ["tqc_agbr.out", "tqc_agbr.out.timeout1"])
    original = d12.read_text()
    (tmp_path / "tqc_agbr.out").write_text(data("tqc_agbr.out") +
                                           "Abort(1) on node 1: MPI_Abort\n")
    assert _recover_abort(engine, d12, script)
    new = d12.read_text().splitlines()
    old = original.splitlines()
    # Point 3 of the first run (-16032.1872931170) beats the rerun's points.
    assert new[4] == "6.86462653 18.56008267"
    assert new[6].split()[1:4] == ["-1.323597806577E-17", "-2.083102576397E-17",
                                   "1.587231170349E-01"]
    assert new[7].split()[1:4] == ["3.333333333333E-01", "-8.848585668743E-03",
                                   "-8.333333333333E-02"]
    assert new[6].split()[4:] == old[6].split()[4:]     # Biso 1.000000 X kept
    rest = [l for i, l in enumerate(old) if i not in (4, 6, 7) and l != "RESTART"]
    assert [l for i, l in enumerate(new) if i not in (4, 6, 7)] == rest


def test_backups_are_never_overwritten(engine, tmp_path):
    d12, script = _aborted_job(tmp_path, "tqb_pto", ["tqb_pto.out", "tqb_pto.out.timeout1"])
    original = d12.read_text()
    assert _recover_abort(engine, d12, script)
    first = d12.read_text()
    # The fresh run is itself killed; its RESTART rerun aborts again, somewhere
    # lower in energy (same geometry print, lower energy).
    shutil.copy(DATA / "tqb_pto.out", tmp_path / "tqb_pto.out")
    lower = data("tqb_pto.out").replace("-1.2683449049414E+03", "-1.2683449059414E+03").replace(
        "3.92491474     3.92491474     3.92491474", "3.92500000     3.92500000     3.92500000")
    (tmp_path / "tqb_pto.out").write_text(
        lower.replace("3.94649838     3.94649838     3.94649838",
                      "3.92491474     3.92491474     3.92491474"))
    d12.write_text(opt_restart.add_optgeom_restart(first)[0])
    assert _recover_abort(engine, d12, script)
    assert (tmp_path / "tqb_pto.d12.orig").read_text() == original
    assert (tmp_path / "tqb_pto.d12.orig2").read_text() == opt_restart.add_optgeom_restart(first)[0]
    assert "\n3.92500000\n" in d12.read_text()
    assert (tmp_path / "tqb_pto.out.optabort2").is_file()


def test_nothing_to_carry_over_means_no_resubmission(engine, tmp_path, capsys):
    """Only the aborted RESTART run with no point of its own completed and no
    earlier output: nothing better than repeating the abort."""
    d12, script = _aborted_job(tmp_path, "tqb_pto", [])
    head = cut_before(data("tqb_pto.out"), "CELL OPTIMIZATION - POINT") + "Abort(1)\n"
    (tmp_path / "tqb_pto.out").write_text(head)
    before = d12.read_text()
    assert _recover_abort(engine, d12, script) is None
    assert d12.read_text() == before
    assert not (tmp_path / "tqb_pto.d12.orig").exists()


# ------------------------------------------------- timeouts at the limit

def _recover_timeout(engine, tmp_path, script, runner=None):
    engine.slurm_runner = runner or FakeSlurm()
    calc = {"calc_id": "T", "material_id": "M", "calc_type": "OPT",
            "input_file": str(tmp_path / f"{script.stem}.d12"), "work_dir": str(tmp_path),
            "job_script": str(script), "error_type": "timeout_error"}
    return engine.attempt_recovery(calc, create_record=False)


def test_non_opt_at_the_limit_is_not_resubmitted(engine, tmp_path, capsys):
    (tmp_path / "job.d12").write_text("t\nCRYSTAL\nEND\n")
    assert _recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00")) is None
    out = capsys.readouterr().out
    assert "already at the queue walltime limit 7-00:00:00" in out
    assert "would time out again" in out
    assert not list(tmp_path.glob("job_recovery_*.sh"))


def test_first_scf_opt_at_the_limit_is_not_resubmitted(engine, tmp_path, monkeypatch):
    shutil.copy(corpus(DIA_D12), tmp_path / "job.d12")
    (tmp_path / "job.out").write_text(
        cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "== SCF ENDED"))
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert _recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00")) is None


def test_opt_with_progress_at_the_limit_restarts(engine, tmp_path, monkeypatch):
    shutil.copy(corpus(DIA_D12), tmp_path / "job.d12")
    (tmp_path / "job.out").write_text(
        cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3"))
    (tmp_path / "scratch" / "crys23" / "job").mkdir(parents=True)
    (tmp_path / "scratch" / "crys23" / "job" / "OPTINFO.DAT").write_text("x")
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    res = _recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00"))
    assert res and "#SBATCH -t 7-00:00:00" in res["fixed_job_script"].read_text()
    assert opt_restart.has_optgeom_restart((tmp_path / "job.d12").read_text())


def test_timeout_without_optinfo_starts_from_the_best_point(engine, tmp_path, monkeypatch):
    """Killed after four points, but its scratch directory is gone: at the
    limit it is still worth resubmitting, as a fresh OPT from point 2."""
    shutil.copy(DATA / "tqb_pto.d12", tmp_path / "tqb_pto.d12")
    deck = opt_restart.remove_optgeom_restart(data("tqb_pto.d12"))[0]
    (tmp_path / "tqb_pto.d12").write_text(deck)
    (tmp_path / "tqb_pto.out").write_text(data("tqb_pto.out.timeout1"))
    (tmp_path / "scratch" / "crys23").mkdir(parents=True)       # visible, no job dir
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    res = _recover_timeout(engine, tmp_path, job_script(tmp_path, "7-00:00:00", name="tqb_pto"))
    assert res
    assert (tmp_path / "tqb_pto.d12").read_text() == deck.replace("\n3.94649838\n",
                                                                  "\n3.92491474\n")
    assert (tmp_path / "tqb_pto.d12.orig").read_text() == deck
    assert (tmp_path / "tqb_pto.out.timeout1").is_file()


# ------------------------------------------------------ absent scratch

def test_visible_scratch_without_the_job_dir_is_absent(monkeypatch, tmp_path):
    script = "export JOB=mat_opt\nexport scratch=$SCRATCH/crys23\n"
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    d = opt_restart.job_scratch_dir(script, "mat_opt")
    assert opt_restart.optinfo_state(d) == "absent"              # $SCRATCH there, crys23 not
    (tmp_path / "crys23").mkdir()
    assert opt_restart.optinfo_state(d) == "absent"
    monkeypatch.setenv("SCRATCH", str(tmp_path / "not" / "mounted"))
    assert opt_restart.optinfo_state(opt_restart.job_scratch_dir(script, "mat_opt")) == "unknown"


# ------------------------------------------------ the per-error retry cap

def test_per_error_max_retries_is_honoured(mgr, capsys):
    db = mgr.db
    db.create_material(material_id="m", formula="C")
    for i in range(2):
        cid = db.create_calculation(material_id="m", calc_type="OPT",
                                    input_file=f"/x/in{i}.d12", work_dir="/x")
        db.update_calculation_status(cid, "failed", error_type="timeout_error")
        db.update_calculation_status(cid, "resubmitted")
    third = db.create_calculation(material_id="m", calc_type="OPT",
                                  input_file="/x/in2.d12", work_dir="/x")
    db.update_calculation_status(third, "failed")
    mgr.max_recovery_attempts = 10
    reached = []
    mgr.error_recovery_engine.attempt_recovery = lambda *a, **k: reached.append(1)
    assert mgr.error_recovery_engine.config["error_recovery"]["timeout_error"]["max_retries"] == 2
    ok = mgr.attempt_error_recovery({"calc_id": third, "material_id": "m"},
                                    "timeout_error", "SLURM state TIMEOUT")
    assert ok is False and not reached
    assert "max_retries 2" in capsys.readouterr().out
    # Another error type still has its own budget.
    mgr.attempt_error_recovery({"calc_id": third, "material_id": "m"},
                               "opt_trust_radius_error", "collapsed")
    assert reached


# --------------------------------------- old job scripts (bash staging)

def _generated_script(tmp_path):
    gen = tmp_path / "submitcrystal23.sh"
    shutil.copy(SUBMIT_SH, gen)
    gen.write_text(gen.read_text().replace("\nsbatch ", "\n#sbatch "))
    subprocess.run(["bash", str(gen), "testmat"], cwd=tmp_path, capture_output=True, text=True)
    return (tmp_path / "testmat.sh").read_text()


def _as_before_this_change(script):
    """The same script as generated before the RESTART staging existed."""
    start = script.index("# OPTGEOM RESTART")
    end = script.index("# GUESSP restart")
    old = script[:start] + script[end:]
    return old.replace(opt_restart._NEW_GUESSP_IF, opt_restart._OLD_GUESSP_IF)


def test_old_script_gets_exactly_the_current_staging(tmp_path):
    new = _generated_script(tmp_path)
    old = _as_before_this_change(new)
    assert "# OPTGEOM RESTART" not in old and "RESTART_KEEPS_FORT20" not in old
    refreshed, what = opt_restart.refresh_restart_staging(old)
    assert refreshed == new, what
    assert opt_restart.refresh_restart_staging(new) == (new, "already has the RESTART staging")


def test_old_script_with_other_fort20_handling_is_left_alone(tmp_path):
    old = _as_before_this_change(_generated_script(tmp_path))
    odd = old.replace(opt_restart._OLD_GUESSP_IF, 'if [ -f "$DIR/$JOB.f9" ]; then')
    assert opt_restart.refresh_restart_staging(odd)[0] == odd


def test_timeout_recovery_refreshes_an_old_script_copy(engine, tmp_path, monkeypatch, capsys):
    shutil.copy(corpus(DIA_D12), tmp_path / "testmat.d12")
    (tmp_path / "testmat.out").write_text(
        cut_before(corpus(DIA_OUT).read_text(errors="ignore"), "OPTIMIZATION - POINT    3"))
    monkeypatch.setenv("SCRATCH", "")
    old = _as_before_this_change(_generated_script(tmp_path)).replace(
        "#SBATCH -t 7-00:00:00", "#SBATCH -t 1-00:00:00")
    (tmp_path / "testmat.sh").write_text(old)
    res = _recover_timeout(engine, tmp_path, tmp_path / "testmat.sh")
    text = res["fixed_job_script"].read_text()
    assert "# OPTGEOM RESTART" in text and opt_restart._NEW_GUESSP_IF in text
    assert (tmp_path / "testmat.sh").read_text() == old          # the original is untouched
    assert subprocess.run(["bash", "-n", str(res["fixed_job_script"])]).returncode == 0


def test_empty_f9_is_not_staged_as_a_guess(tmp_path):
    """An aborted OPT copies its EMPTY fort.9 back as $JOB.f9 (measured:
    tqb_pto.f9, 0 bytes); GUESSP must not stage it."""
    script = _generated_script(tmp_path)
    block = script[script.index("# OPTGEOM RESTART"):script.index("\ncd $scratch/$JOB") + 1]
    d = tmp_path / "run"
    (d / "scratch" / "testmat").mkdir(parents=True)
    deck = "t\nCRYSTAL\nEND\nGUESSP\nEND\n"
    (d / "scratch" / "testmat" / "INPUT").write_text(deck)
    (d / "testmat.f9").write_text("")
    r = subprocess.run(["bash", "-c", f'DIR="{d}"\nJOB=testmat\nscratch="{d}/scratch"\n' + block],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (d / "scratch" / "testmat" / "fort.20").exists()
    assert "GUESSP" not in (d / "scratch" / "testmat" / "INPUT").read_text()


# ------------------------------------------- the queue manager's real path

def test_restart_abort_is_recovered_by_the_manager(monkeypatch, tmp_path):
    """End to end through the status sweep: sacct says COMPLETED, the .out
    stops at MPI_Abort, fort.87 holds PXK TOO SMALL - the job goes back in
    with the rewritten deck and the script that ran."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    for f in ("tqb_pto.d12", "tqb_pto.out.timeout1"):
        shutil.copy(DATA / f, tmp_path / f)
    original = (tmp_path / "tqb_pto.d12").read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted = []
    mgr.submit_to_slurm = (lambda input_file, work_dir, calc_type, submit_script_override=None:
                           submitted.append((input_file, submit_script_override)) or
                           str(888001 + len(submitted) - 1))
    calc_id = mgr.submit_calculation(tmp_path / "tqb_pto.d12")
    script = job_script(tmp_path, "0-00:24:00", name="tqb_pto")
    shutil.copy(DATA / "tqb_pto.out", tmp_path / "tqb_pto.out")
    _scratch_job(tmp_path, monkeypatch, "tqb_pto", data("tqb_pto.fort.87"))

    def fake_run(cmd, capture_output=True, text=True, **kw):
        out = "JOBID,STATE,START\n" if cmd[0] == "squeue" else "COMPLETED \n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(qm.subprocess, "run", fake_run)

    mgr.check_queue_status()

    old = mgr.db.get_calculation(calc_id)
    assert old["status"] == "resubmitted", old
    assert old["error_type"] == "opt_trust_radius_error"
    assert len(submitted) == 2
    assert Path(submitted[1][0]).name == "tqb_pto.d12"
    assert Path(submitted[1][1]) == script
    assert (tmp_path / "tqb_pto.d12.orig").read_text() == original
    new = (tmp_path / "tqb_pto.d12").read_text()
    assert "\n3.92491474\n" in new and not opt_restart.has_optgeom_restart(new)
