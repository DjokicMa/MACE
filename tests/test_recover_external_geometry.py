"""`mace recover` on an EXTERNAL deck keeps its geometry. The SCF fix writes a
new deck, <stem>_recovery_<time>.d12, and the job script MACE writes for it
stages <that stem>.ext as fort.34 (submitcrystal23.sh); the recovered deck
used to go to the queue with no .ext beside it, and CRYSTAL stopped with no
geometry. Real `mace_cli` subprocesses with tests/fake_slurm.py on PATH, from
the corpus' SCF stop (FERMI ENERGY NOT IN INTERVAL) on an SP deck whose
geometry is made EXTERNAL."""
import re
from pathlib import Path

import pytest

from array_helpers import rows
from conftest import find_data
from fake_slurm import FakeSlurm
from test_recover_resubmits import CASES, _env, _ok, _recover, _seed

FORT34 = """3 1 1
 0.00000000000E+00 0.17724056000E+01 0.17724056000E+01
 0.17724056000E+01 0.00000000000E+00 0.17724056000E+01
 0.17724056000E+01 0.17724056000E+01 0.00000000000E+00
1
 1.00000000000E+00 0.00000000000E+00 0.00000000000E+00
 0.00000000000E+00 1.00000000000E+00 0.00000000000E+00
 0.00000000000E+00 0.00000000000E+00 1.00000000000E+00
 0.00000000000E+00 0.00000000000E+00 0.00000000000E+00
2
 6 0.44310140000E+00 0.44310140000E+00 0.44310140000E+00
 6 -0.44310140000E+00 -0.44310140000E+00 -0.44310140000E+00
"""


def _external(work, name):
    """The seeded deck with its geometry moved out to a fort.34 file."""
    deck = work / f"{name}.d12"
    lines = deck.read_text().splitlines(keepends=True)
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "END")
    deck.write_text(lines[0] + "EXTERNAL\n" + "".join(lines[end:]))
    return deck


def _recovered(work, cid):
    new = [r for r in rows(work / "materials.db").values() if r["calc_id"] != cid]
    assert len(new) == 1 and new[0]["status"] == "submitted", new
    deck = Path(new[0]["input_file"])
    assert re.search(r"_recovery_\d{8}_\d{6}\.d12$", deck.name)
    assert deck.read_text().splitlines()[1] == "EXTERNAL"
    return deck


def test_the_recovered_deck_takes_its_ext_along(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    name = Path(CASES["scf"][1]).stem
    cid = _seed(work, "scf")
    deck = _external(work, name)
    deck.with_suffix(".ext").write_text(FORT34)
    out = _ok(_recover(env, work, "--yes"))
    assert "Resubmitted 1 of 1 calculation(s)." in out, out
    new = _recovered(work, cid)
    assert new.with_suffix(".ext").read_text() == FORT34
    (sb,) = fake.calls("sbatch")
    # the job MACE wrote for it stages <its stem>.ext as fort.34
    assert new.stem in sb["script_text"] and '"$DIR/$JOB.ext"' in sb["script_text"]


def test_a_geometry_its_own_job_script_staged_goes_along_too(tmp_path):
    """The failed job's own script copied another file to fort.34
    (external_geometry.fort34_source): the recovered deck's .ext is that file."""
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    name = Path(CASES["scf"][1]).stem
    cid = _seed(work, "scf")
    _external(work, name)
    (work / "relaxed.gui").write_text(FORT34)
    script = find_data(CASES["oom_sp"][2]).read_text().replace(Path(CASES["oom_sp"][2]).stem, name)
    (work / f"{name}.sh").write_text(script + '\ncp "$DIR/relaxed.gui" "$scratch/$JOB/fort.34"\n')
    _ok(_recover(env, work, "--yes"))
    new = _recovered(work, cid)
    assert new.with_suffix(".ext").read_text() == FORT34


def test_dry_run_names_the_geometry_copy_and_writes_nothing(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    work = tmp_path / "w"
    name = Path(CASES["scf"][1]).stem
    _seed(work, "scf")
    deck = _external(work, name)
    deck.with_suffix(".ext").write_text(FORT34)
    before = sorted(p.name for p in work.iterdir())
    out = _ok(_recover(env, work, "--dry-run"))
    n = re.escape(name)
    assert re.search(rf"would write \S*{n}_recovery_\d{{8}}_\d{{6}}\.ext\n +\(a copy of {n}\.ext: ", out), out
    assert sorted(p.name for p in work.iterdir() if p.name != ".queue_locks") == \
        [n for n in before if n != ".queue_locks"]
    assert not fake.calls("sbatch")
