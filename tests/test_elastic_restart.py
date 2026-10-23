"""RESTART / RESTART2 of elastic and EOS runs through MACE.

CRYSTAL's RESTART (partial) and RESTART2 (complete) of ELASTCON, ELAPIEZO,
PIEZOCON, PHOTOELA, EOS and QHA read ELASINFO.DAT or EOSINFO.DAT in the run
directory (manual pp. 253, 284, 290). The job script never copied the file back
or staged it, a RESTART deck (then from the elastic generate command) had a new job name
(a new scratch folder, without the file), and the parent deck's own module
settings were never read (d12_calc_elastic.parse_block had no caller).

Now the job script copies the file back as $JOB.ELASINFO.DAT /
$JOB.EOSINFO.DAT and stages it for a RESTART/RESTART2 deck; opt2d12 puts the
parent run's file next to the new deck under the new name (or refuses when it
has none), and a follow-up of the same module keeps the parent's settings.

Real files: the HPCC decks and outputs of tests/data/hpcc_116/mechanics, with
the ELASINFO.DAT and EOSINFO.DAT the diamond ELAPIEZO and EOS runs wrote.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_parsers import CrystalInputParser  # noqa: E402

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
SUBMIT_SH = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def _mace(*args, cwd):
    env = dict(os.environ, MACE_NO_BANNER="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _restart(cwd, name, mode, record):
    """The RESTART deck the 1.1.6 elastic generate --restart wrote (its flags gave PRINT)."""
    calc = mode if mode in ("EOS", "QHA", "CPHF") else "ELASTIC"
    return _mace("opt2d12", "--out-file", f"{name}.out", "--calc-type", calc, "--non-interactive",
                 "--elastic-settings", json.dumps({"mode": mode, "restart": record, "print": True}), cwd=cwd)


def _parent(tmp_path, name, info=True):
    for ext in ("out", "d12"):
        shutil.copy(MECH / f"{name}.{ext}", tmp_path / f"{name}.{ext}")
    kind = "EOSINFO.DAT" if "eos" in name else "ELASINFO.DAT"
    if info:
        shutil.copy(MECH / f"{name}.{kind}", tmp_path / f"{name}.{kind}")
    return kind


def _block(lines, key):
    i = lines.index(key)
    return lines[i:lines.index("END", i) + 1]


# ------------------------------------------------------------------ the job script
def _script(tmp_path):
    gen = tmp_path / "gen" / "submitcrystal23.sh"
    gen.parent.mkdir()
    shutil.copy(SUBMIT_SH, gen)
    subprocess.run(["bash", str(gen), "job"], cwd=gen.parent, capture_output=True, text=True)
    text = (gen.parent / "job.sh").read_text()
    assert subprocess.run(["bash", "-n", str(gen.parent / "job.sh")]).returncode == 0
    return text


def _stage_block(text):
    start = text.index("# ELASTCON/ELAPIEZO/PIEZOCON/PHOTOELA and EOS/QHA RESTART")
    return text[start:text.index("\nfi\n", start) + 4]


def _copy_back_block(text):
    start = text.index("# The restart file of an elastic or EOS/QHA run")
    return text[start:text.index("\ndone\n", start) + 6]


def _run(block, d, scratch):
    prelude = f'DIR="{d}"\nJOB=job\nscratch="{scratch}"\n'
    r = subprocess.run(["bash", "-c", prelude + block], cwd=scratch / "job", capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    return r.stdout


def test_the_blocks_sit_before_and_after_crystal(tmp_path):
    text = _script(tmp_path)
    stage = text.index("# ELASTCON/ELAPIEZO/PIEZOCON/PHOTOELA and EOS/QHA RESTART")
    back = text.index("# The restart file of an elastic or EOS/QHA run")
    assert text.index('cp "$DIR/$JOB.d12" "$scratch/$JOB/INPUT"') < stage < text.index("bin/Pcrystal") < back
    assert back < text.index("# SCRATCH FALLBACK CLEANUP")


@pytest.mark.parametrize("record", ["RESTART", "RESTART2"])
def test_a_restart_deck_gets_its_file(tmp_path, record):
    d, scratch = tmp_path / "d", tmp_path / "s"
    (scratch / "job").mkdir(parents=True)
    d.mkdir()
    lines = (MECH / "dia_elapiezo.d12").read_text().splitlines()
    lines.insert(lines.index("PRINT") + 1, record)
    (scratch / "job" / "INPUT").write_text("\n".join(lines) + "\n")
    shutil.copy(MECH / "dia_elapiezo.ELASINFO.DAT", d / "job.ELASINFO.DAT")
    out = _run(_stage_block(_script(tmp_path)), d, scratch)
    assert "RESTART: staged job.ELASINFO.DAT as ELASINFO.DAT" in out
    assert (scratch / "job" / "ELASINFO.DAT").read_bytes() == (MECH / "dia_elapiezo.ELASINFO.DAT").read_bytes()
    assert not (scratch / "job" / "EOSINFO.DAT").exists()


def test_no_restart_record_stages_nothing_and_a_newer_scratch_file_is_kept(tmp_path):
    d, scratch = tmp_path / "d", tmp_path / "s"
    (scratch / "job").mkdir(parents=True)
    d.mkdir()
    shutil.copy(MECH / "dia_eos.EOSINFO.DAT", d / "job.EOSINFO.DAT")
    shutil.copy(MECH / "dia_eos.d12", scratch / "job" / "INPUT")
    block = _stage_block(_script(tmp_path))
    assert _run(block, d, scratch) == "" and not (scratch / "job" / "EOSINFO.DAT").exists()
    lines = (MECH / "dia_eos.d12").read_text().splitlines()
    lines.insert(lines.index("PRINT") + 1, "RESTART")
    (scratch / "job" / "INPUT").write_text("\n".join(lines) + "\n")
    killed = scratch / "job" / "EOSINFO.DAT"          # a killed run of this job left its own
    killed.write_text("newer\n")
    os.utime(d / "job.EOSINFO.DAT", (1, 1))
    assert _run(block, d, scratch) == "" and killed.read_text() == "newer\n"


def test_a_run_copies_its_file_back(tmp_path):
    d, scratch = tmp_path / "d", tmp_path / "s"
    (scratch / "job").mkdir(parents=True)
    d.mkdir()
    run = scratch / "job"
    (run / ".mace_run_start").write_text("")
    os.utime(run / ".mace_run_start", (1000, 1000))
    shutil.copy(MECH / "dia_eos.EOSINFO.DAT", run / "EOSINFO.DAT")       # written by this run
    (run / "ELASINFO.DAT").write_text("from an earlier run\n")
    os.utime(run / "ELASINFO.DAT", (10, 10))
    _run(_copy_back_block(_script(tmp_path)), d, scratch)
    assert (d / "job.EOSINFO.DAT").read_bytes() == (MECH / "dia_eos.EOSINFO.DAT").read_bytes()
    assert not (d / "job.ELASINFO.DAT").exists()


def _freq_block(text):
    start = text.index("# What a FREQ run writes beside its output")
    return text[start:text.index("\n\n", start) + 1]


def test_a_freq_run_copies_its_spectra_and_phonon_bands_back(tmp_path):
    """The example suite's diamond FREQ on HPCC: IRSPEC.DAT, RAMSPEC.DAT,
    PHONBANDS.DAT, PHONDOS.DAT and fort.25 stayed in scratch, so mace submit
    users lost the spectra and the phonon bands and DOS. A file an earlier run
    left is not copied."""
    d, scratch = tmp_path / "d", tmp_path / "s"
    run = scratch / "job"
    run.mkdir(parents=True)
    d.mkdir()
    (run / ".mace_run_start").write_text("")
    os.utime(run / ".mace_run_start", (1000, 1000))
    for name in ("IRSPEC.DAT", "RAMSPEC.DAT", "PHONBANDS.DAT", "PHONDOS.DAT", "fort.25"):
        (run / name).write_text(f"{name} of this run\n")
    (run / "RAMSPEC.DAT").write_text("from an earlier run\n")
    os.utime(run / "RAMSPEC.DAT", (10, 10))
    _run(_freq_block(_script(tmp_path)), d, scratch)
    assert (d / "job_IRSPEC.DAT").read_text() == "IRSPEC.DAT of this run\n"
    assert (d / "job.PHONBANDS.DAT").read_text() == "PHONBANDS.DAT of this run\n"
    assert (d / "job.PHONDOS.DAT").read_text() == "PHONDOS.DAT of this run\n"
    assert (d / "job.f25").read_text() == "fort.25 of this run\n"
    assert not (d / "job_RAMSPEC.DAT").exists()


# ------------------------------------------------------------------ the parent's settings
def test_the_input_parser_reads_the_module_block():
    got = {n: CrystalInputParser(str(MECH / f"{n}.d12")).parse().get("elastic_settings")
           for n in ("dia_elastcon", "bn_elapiezo", "dia_eos", "dia_qha", "dia_cphf")}
    assert got["dia_elastcon"] == {"mode": "ELASTCON", "nuchess": True, "hessnum2": True, "print": True}
    assert got["bn_elapiezo"] == {"mode": "ELAPIEZO", "print": True}      # its RESTART is its own run's
    assert got["dia_eos"] == {"mode": "EOS", "print": True}
    assert got["dia_qha"]["mode"] == "QHA" and got["dia_cphf"] == {"mode": "CPHF", "fourth": True}
    sp = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si" / "parent.d12"
    assert "elastic_settings" not in CrystalInputParser(str(sp)).parse()


def test_a_follow_up_of_the_same_module_keeps_the_parents_settings(tmp_path):
    _parent(tmp_path, "dia_elastcon", info=False)
    r = _mace("opt2d12", "--out-file", "dia_elastcon.out", "--non-interactive", "--calc-type", "ELASTIC",
              "--output-dir", "o", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next((tmp_path / "o").glob("*.d12")).read_text().splitlines()
    assert _block(deck, "ELASTCON") == ["ELASTCON", "NUCHESS", "HESSNUM2", "PRINT", "END"]


def test_the_parents_restart_record_is_not_carried(tmp_path):
    """bn_elapiezo.d12 is the RESTART deck of its third job: a new deck from
    it restarts nothing unless asked."""
    _parent(tmp_path, "bn_elapiezo", info=False)
    r = _mace("opt2d12", "--out-file", "bn_elapiezo.out", "--non-interactive", "--calc-type", "ELASTIC",
              "--output-dir", "o", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next((tmp_path / "o").glob("*.d12")).read_text().splitlines()
    assert _block(deck, "ELAPIEZO") == ["ELAPIEZO", "PRINT", "END"]


# ------------------------------------------------------------------ opt2d12 with "restart" in --elastic-settings
@pytest.mark.parametrize("name, mode, record, block", [
    ("dia_elastcon", "ELASTCON", "RESTART2", ["ELASTCON", "NUCHESS", "HESSNUM2", "PRINT", "RESTART2", "END"]),
    ("dia_elapiezo", "ELAPIEZO", "RESTART", ["ELAPIEZO", "PRINT", "RESTART", "END"]),
    ("dia_eos", "EOS", "RESTART2", ["EOS", "PRINT", "RESTART2", "END"]),
])
def test_restart_stages_the_parent_runs_file_under_the_new_name(tmp_path, name, mode, record, block):
    if name == "dia_elastcon":          # the ELASTCON run's own file: same format as ELAPIEZO's
        for ext in ("out", "d12"):
            shutil.copy(MECH / f"{name}.{ext}", tmp_path / f"{name}.{ext}")
        shutil.copy(MECH / "dia_elapiezo.ELASINFO.DAT", tmp_path / f"{name}.ELASINFO.DAT")
        kind = "ELASINFO.DAT"
    else:
        kind = _parent(tmp_path, name)
    r = _restart(tmp_path, name, mode, record)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(p for p in tmp_path.glob(f"{name}_*.d12"))
    assert _block(deck.read_text().splitlines(), mode) == block
    staged = deck.with_suffix(f".{kind}")
    assert staged.read_bytes() == (tmp_path / f"{name}.{kind}").read_bytes()
    assert f"staged {name}.{kind} as {staged.name}" in r.stdout.replace(str(tmp_path) + "/", "")


def test_restart_from_the_parent_jobs_scratch_folder(tmp_path):
    """A run killed at its walltime copied nothing back: its file is in the
    scratch folder its job script recorded (.<JOB>.scratch)."""
    _parent(tmp_path, "dia_elapiezo", info=False)
    scratch = tmp_path / "scratch" / "dia_elapiezo"
    scratch.mkdir(parents=True)
    shutil.copy(MECH / "dia_elapiezo.ELASINFO.DAT", scratch / "ELASINFO.DAT")
    (tmp_path / ".dia_elapiezo.scratch").write_text(str(scratch) + "\n")
    r = _restart(tmp_path, "dia_elapiezo", "ELAPIEZO", "RESTART")
    assert r.returncode == 0, r.stdout + r.stderr
    staged = next(tmp_path.glob("dia_elapiezo_*.ELASINFO.DAT"))
    assert staged.read_bytes() == (MECH / "dia_elapiezo.ELASINFO.DAT").read_bytes()


def test_restart_without_the_file_is_refused(tmp_path):
    _parent(tmp_path, "dia_eos", info=False)
    r = _restart(tmp_path, "dia_eos", "EOS", "RESTART")
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "RESTART reads EOSINFO.DAT of the run it continues, and none was found" in out
    assert "dia_eos.EOSINFO.DAT" in out
    assert not list(tmp_path.glob("dia_eos_*.d12"))
