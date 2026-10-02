"""EXTERNAL geometry: the job script stages ``$JOB.ext`` as fort.34.

A deck that starts with EXTERNAL reads its geometry from fort.34 (manual p.21;
file format pp.450-451). ``mace phonopy generate`` writes one such deck per
displaced supercell, with the geometry next to it as ``<job>.ext``. Without the
staging step CRYSTAL would find no fort.34 in the scratch directory, or worse a
fort.34 left behind by an earlier run of the same job name.

The block must be a no-op for every existing deck (none has a .ext), must sit
outside the OPTGEOM-restart/GUESSP region that recovery/opt_restart.py copies
into old scripts, and must run before Pcrystal.
"""
import os
import shutil
import subprocess

from conftest import REPO_ROOT

SUBMIT_SH = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"
WORKFLOW_SH = REPO_ROOT / "mace" / "workflow" / "submitcrystal23.sh"


def _generate(tmp_path):
    gen = tmp_path / "submitcrystal23.sh"
    shutil.copy(SUBMIT_SH, gen)
    gen.write_text(gen.read_text().replace("\nsbatch ", "\n#sbatch "))
    subprocess.run(["bash", str(gen), "testmat"], cwd=tmp_path,
                   capture_output=True, text=True)
    script = tmp_path / "testmat.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, \
        "generated job script is not valid shell"
    return script.read_text()


def _external_block(script):
    start = script.index("# EXTERNAL geometry")
    return script[start:script.index("\nfi\n", start) + 4]


def test_block_sits_after_guessp_and_before_crystal_runs(tmp_path):
    script = _generate(tmp_path)
    ext = script.index("# EXTERNAL geometry")
    assert script.index("# GUESSP restart") < ext
    # Outside the region opt_restart.py lifts into old scripts ...
    assert script.index("# OPTGEOM RESTART") < script.index("# GUESSP restart") < ext
    # ... and before the job changes into scratch and starts CRYSTAL.
    assert ext < script.index("cd $scratch/$JOB") < script.index("bin/Pcrystal")


def test_workflow_copy_is_still_a_symlink_to_the_submission_script():
    assert WORKFLOW_SH.is_symlink()
    assert os.readlink(WORKFLOW_SH) == "../submission/submitcrystal23.sh"


def _run_block(tmp_path, tag, ext_bytes):
    block = _external_block(_generate(tmp_path))
    d = tmp_path / tag
    scratch = d / "scratch" / "testmat"
    scratch.mkdir(parents=True)
    (d / "testmat.d12").write_text("title\nEXTERNAL\nEND\n")
    if ext_bytes is not None:
        (d / "testmat.ext").write_bytes(ext_bytes)
    prelude = f'DIR="{d}"\nJOB=testmat\nscratch="{d}/scratch"\n'
    res = subprocess.run(["bash", "-c", prelude + block],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    fort34 = scratch / "fort.34"
    return res.stdout, (fort34.read_bytes() if fort34.exists() else None)


def test_ext_is_staged_byte_for_byte_as_fort34(tmp_path):
    ext = b"3 1 1\n  5.0 0.0 0.0\n  0.0 5.0 0.0\n  0.0 0.0 5.0\n1\n"
    out, staged = _run_block(tmp_path, "with", ext)
    assert staged == ext
    assert "EXTERNAL: staged testmat.ext as fort.34" in out


def test_nothing_happens_without_an_ext(tmp_path):
    out, staged = _run_block(tmp_path, "without", None)
    assert staged is None
    assert "EXTERNAL:" not in out
