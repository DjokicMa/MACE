"""A recovered job keeps the resources the user set in its job script.

When recovery fixes a deck (SCF convergence: a new ``*_recovery_*`` deck) the
queue manager generates that deck's job script from the template, which used
to drop what the user had changed in the original ``<job>.sh`` (ntasks,
memory, walltime, partition, exclude, account). Drives the real
submit_to_slurm and its generator against a fake sbatch that keeps a copy of
every script it is given.
"""
import os
import re
import stat

import pytest

D12 = """\
TITLE
CRYSTAL
0 0 0
1
2.0
1
6 0.0 0.0 0.0
OPTGEOM
MAXCYCLE
50
ENDOPT
END
BASISSET
POB-TZVP-REV2
DFT
PBE
ENDDFT
TOLDEE
9
MAXCYCLE
100
END
"""

CUSTOM = {
    "ntasks": "#SBATCH --ntasks=8",
    "mem": "#SBATCH --mem-per-cpu=9G",
    "time": "#SBATCH -t 3:00:00",
    "partition": "#SBATCH -p mypart",
    "exclude": "#SBATCH --exclude=amr-001",
    "account": "#SBATCH -A mylab",
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from mace.queue.manager import EnhancedCrystalQueueManager
    from mace.recovery.recovery import ErrorRecoveryEngine

    work = tmp_path / "work"
    work.mkdir()
    deck = work / "job.d12"
    deck.write_text(D12)
    cap = tmp_path / "sbatched"
    cap.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text(f'#!/bin/bash\ncp "$1" "{cap}/$(basename "$1")"\necho "Submitted batch job 4242"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for var in ("MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION"):
        monkeypatch.delenv(var, raising=False)

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(work), db_path=str(tmp_path / "mgr.db"),
        enable_tracking=True, enable_error_recovery=False, organize_outputs=False)
    mgr.is_workflow_context = False
    engine = ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))
    return mgr, engine, work, deck, cap


def _customize(script):
    """The user's edits: every resource line replaced with their own value."""
    out = []
    for line in script.read_text().splitlines():
        if re.match(r"#SBATCH\s+(--ntasks|--mem|-t\b|--time|-p\b|--partition|--exclude|-A\b|--account)", line):
            continue
        out.append(line)
    at = max(i for i, l in enumerate(out) if l.startswith("#SBATCH")) + 1
    out[at:at] = list(CUSTOM.values())
    script.write_text("\n".join(out) + "\n")


def _sbatch_lines(text):
    return [l for l in text.splitlines() if l.startswith("#SBATCH")]


def _original(mgr, work, deck):
    assert mgr.submit_to_slurm(deck, work, "OPT") == "4242"
    script = work / "job.sh"
    assert script.is_file()
    _customize(script)
    return script


def test_scf_recovery_keeps_custom_resources(setup):
    mgr, engine, work, deck, cap = setup
    _original(mgr, work, deck)
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT", "input_file": str(deck),
            "work_dir": str(work), "error_type": "convergence_error"}
    rec = engine.attempt_recovery(calc, create_record=False)
    assert rec and rec["fixed_job_script"] is None
    assert mgr.resubmit_fixed_calculation(calc, fixed_input=rec["fixed_input_file"])

    stem = rec["fixed_input_file"].stem
    sent = (cap / f"{stem}.sh").read_text()
    lines = _sbatch_lines(sent)
    for line in CUSTOM.values():
        assert lines.count(line) == 1, (line, lines)
    assert not any(re.match(r"#SBATCH\s+--mem(-per-cpu)?=", l) and l != CUSTOM["mem"] for l in lines)
    assert f"#SBATCH -J {stem}" in lines                    # the recovery deck's own name
    from mace.submission.job_script_age import generated_unmodified
    assert generated_unmodified(work / f"{stem}.sh")        # still MACE's own script


def test_oom_recovery_raises_memory_from_the_custom_value(setup):
    mgr, engine, work, deck, cap = setup
    script = _original(mgr, work, deck)
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT", "input_file": str(deck),
            "work_dir": str(work), "job_script": str(script), "error_type": "memory_error"}
    rec = engine.attempt_recovery(calc, create_record=False)
    assert rec and rec["fixed_job_script"] is not None
    assert mgr.resubmit_fixed_calculation(calc, fixed_input=rec["fixed_input_file"],
                                          fixed_job_script=rec["fixed_job_script"])
    lines = _sbatch_lines((cap / rec["fixed_job_script"].name).read_text())
    assert "#SBATCH --mem-per-cpu=13GB" in lines             # 9G x 1.5, not the template's 5G
    for key in ("ntasks", "time", "partition", "exclude", "account"):
        assert CUSTOM[key] in lines


def test_unreadable_original_keeps_the_template(setup, capsys):
    mgr, engine, work, deck, cap = setup
    (work / "job.sh").write_text("#!/bin/bash\nsrun my_own_thing\n")
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT", "input_file": str(deck),
            "work_dir": str(work), "error_type": "convergence_error"}
    rec = engine.attempt_recovery(calc, create_record=False)
    capsys.readouterr()
    assert mgr.resubmit_fixed_calculation(calc, fixed_input=rec["fixed_input_file"])
    out = capsys.readouterr().out
    assert out.count("is not a job script MACE can read") == 1
    lines = _sbatch_lines((cap / f"{rec['fixed_input_file'].stem}.sh").read_text())
    assert "#SBATCH --ntasks=32" in lines                    # the template's


def test_carry_resources_unit():
    from mace.submission.job_resources import carry_resources
    gen = ("#!/bin/bash --login\n#SBATCH -J new\n#SBATCH -o new.o%j\n#SBATCH --ntasks=32\n"
           "#SBATCH -t 7-00:00:00\n#SBATCH --mem-per-cpu=5G\nexport JOB=new\n")
    orig = ("#!/bin/bash --login\n#SBATCH -J old\n#SBATCH -o old.o%j\n#SBATCH -n 4\n"
            "#SBATCH --time=1:00:00\n#SBATCH --mem=40G\n#SBATCH --qos=short\nexport JOB=old\n")
    text, carried = carry_resources(orig, gen)
    assert _sbatch_lines(text) == ["#SBATCH -J new", "#SBATCH -o new.o%j", "#SBATCH -n 4",
                                   "#SBATCH --time=1:00:00", "#SBATCH --mem=40G", "#SBATCH --qos=short"]
    assert "export JOB=new" in text and carried == ["mem", "ntasks", "qos", "time"]
    assert carry_resources("#!/bin/bash\nsrun x\n", gen) == (None, [])
