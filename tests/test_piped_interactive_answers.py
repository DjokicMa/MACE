"""Answers piped into an interactive cif2d12 / opt2d3 run are read again.

The isatty() guards added for batch and workflow runs (no one to ask) also
caught interactive runs whose answers come from a pipe: cif2d12 answered the
per-file space-group question itself and ignored the piped answer, and opt2d3
ECHG refused with "no terminal to ask" after reading the earlier answers
from the same pipe. As in 1.1.1, a run that asks its setup questions on stdin
reads the later ones there too; if the pipe ends first, the question takes
its unattended handling (cif2d12: the default; opt2d3: no deck, exit 1)
instead of an EOFError. --batch / --config-file runs are unchanged.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"

# Rock salt in its primitive rhombohedral cell, declared P1: spglib finds
# Fm-3m, so cif2d12 asks which group to use (default 3: keep the CIF's).
PRIM_CIF = """data_prim
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a 3.98808
_cell_length_b 3.98808
_cell_length_c 3.98808
_cell_angle_alpha 60
_cell_angle_beta 60
_cell_angle_gamma 60
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
Cl1 Cl 0.5 0.5 0.5
"""
# Use the defaults, basis set type, internal basis, save the options
SETUP = "\n\n\n\n"


def _convert(tmp_path, stdin, *extra):
    pytest.importorskip("ase", reason="CIF parsing needs ase")
    pytest.importorskip("spglib", reason="the group check needs spglib")
    cif_dir = tmp_path / "cif"
    cif_dir.mkdir(exist_ok=True)
    (cif_dir / "prim.cif").write_text(PRIM_CIF)
    out = tmp_path / "out"
    shutil.rmtree(out, ignore_errors=True)
    kwargs = {"stdin": subprocess.DEVNULL} if stdin is None else {"input": stdin}
    r = subprocess.run([sys.executable, str(MACE_CLI), "convert", "--no-banner",
                        "--cif_dir", str(cif_dir), "--output_dir", str(out), *extra],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300, **kwargs)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    decks = list(out.glob("*.d12"))
    assert len(decks) == 1, log[-3000:]
    return decks[0].read_text().splitlines(), log


def test_cif2d12_reads_the_piped_space_group_answer(tmp_path):
    deck, log = _convert(tmp_path, SETUP + "2\n")
    assert "Proceeding with spglib space group 225" in log
    assert deck[3] == "225"


def test_cif2d12_takes_the_default_when_the_pipe_ends(tmp_path):
    deck, log = _convert(tmp_path, SETUP)
    assert "Error during symmetry analysis" not in log
    assert "end of input" in log
    assert deck[3] == "1"


def test_cif2d12_batch_without_stdin_is_unchanged(tmp_path):
    _convert(tmp_path, SETUP + "\n")  # saves the options file
    options = tmp_path / "cif2d12_options.json"
    assert options.exists()
    deck, log = _convert(tmp_path, None, "--batch", "--options_file", str(options))
    assert "keeping the CIF space group" in log
    assert deck[3] == "1"


# ------------------------------------------------------------------ opt2d3

def _opt2d3(tmp_path, stdin):
    parent = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not parent.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    shutil.copy(parent, tmp_path / "p.out")
    (tmp_path / "p.f9").write_bytes(b"x")
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d3", "--input", "p.out",
                           "--calc-type", "CHARGE"],
                          cwd=tmp_path, input=stdin, capture_output=True, text=True,
                          timeout=300)


def test_opt2d3_echg_reads_the_piped_map_plane(tmp_path):
    # ECHG, density only, 50 points, define the plane, fractional A B C
    r = _opt2d3(tmp_path, "2\n0\n50\ny\nF\n0 0 0\n1 0 0\n0 1 0\n" + "\n" * 20)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    deck = (tmp_path / "p_charge.d3").read_text().split()
    assert deck[0] == "ECHG"
    assert "no terminal" not in log


def test_opt2d3_echg_without_the_plane_answers_writes_nothing(tmp_path):
    r = _opt2d3(tmp_path, "2\n0\n50\n")
    log = r.stdout + r.stderr
    assert r.returncode == 1, log[-3000:]
    assert "Traceback" not in log
    assert not list(tmp_path.glob("*.d3"))
