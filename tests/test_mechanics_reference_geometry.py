"""The structure of an ELASTIC, EOS or QHA output is its reference cell.

These runs optimize every strained cell or volume to its own FINAL OPTIMIZED
GEOMETRY. opt2d12, opt2cif and a RESTART deck took the
first of them as the structure: diamond came out with a = 3.44764 (EOS, 92 %
of the volume) or 3.49164 (ELAPIEZO strain 1, under group 227 though the cell
is not cubic) instead of 3.54481. The reference is the input geometry block,
where the central point is computed; a run that pre-optimized its structure
(PREOPTGEOM) is refused as a parent.

The outputs are the HPCC CRYSTAL23 runs of tests/data/hpcc_116/mechanics; the
*_strained.out ones keep the central point and the first strained cell of the
full runs (README there).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_parsers import CrystalOutputParser  # noqa: E402
from mace.utils.calc_detection import mechanics_reference_problem  # noqa: E402

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
DIAMOND_A = "3.54481120"
REFERENCE = {"dia_elastcon": DIAMOND_A, "dia_elapiezo": DIAMOND_A, "dia_photoela": DIAMOND_A,
             "dia_eos": DIAMOND_A, "dia_qha": DIAMOND_A, "dia_eos_strained": DIAMOND_A,
             "dia_elapiezo_strained": DIAMOND_A, "bn_elapiezo": "3.60083734"}


def _mace(*args, cwd):
    env = dict(os.environ, MACE_NO_BANNER="1")
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _parent(tmp_path, name, text=None):
    """The output (or ``text``) and its deck under one stem in tmp_path."""
    out = tmp_path / f"{name}.out"
    if text is None:
        shutil.copy(MECH / f"{name}.out", out)
    else:
        out.write_text(text)
    shutil.copy(MECH / f"{name.replace('_strained', '')}.d12", tmp_path / f"{name}.d12")
    return out


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_the_parser_reads_the_reference_cell(name):
    text = (MECH / f"{name}.out").read_text()
    assert mechanics_reference_problem(text) is None
    data = CrystalOutputParser(str(MECH / f"{name}.out")).parse()
    a = REFERENCE[name]
    assert data["conventional_cell"] == [a, a, a, "90.000000", "90.000000", "90.000000"]
    assert data["spacegroup"] == (216 if name.startswith("bn") else 227)
    assert [c["x"] for c in data["coordinates"]][0] in ("1.250000000000E-01", "0.000000000000E+00")


def test_the_strained_cells_are_in_the_outputs():
    """What the parser used to take: the first FINAL OPTIMIZED GEOMETRY."""
    eos = (MECH / "dia_eos_strained.out").read_text()
    ela = (MECH / "dia_elapiezo_strained.out").read_text()
    assert "FINAL OPTIMIZED GEOMETRY" in eos and "3.44764376     3.44764376     3.44764376" in eos
    assert "FINAL OPTIMIZED GEOMETRY" in ela and "3.49163903     3.54481120     3.54481120" in ela


@pytest.mark.parametrize("name", ["dia_eos_strained", "dia_elapiezo_strained"])
def test_opt2d12_writes_the_reference_cell(tmp_path, name):
    out = _parent(tmp_path, name)
    r = _mace("opt2d12", "--out-file", out.name, "--non-interactive", "--calc-type", "SP",
              "--output-dir", "o", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next((tmp_path / "o").glob("*.d12")).read_text().splitlines()
    i = deck.index("227")
    assert deck[i + 1] == DIAMOND_A
    assert deck[i + 3].split()[1:4] == ["1.250000000000E-01"] * 3


def test_opt2cif_writes_the_reference_cell(tmp_path):
    out = _parent(tmp_path, "dia_eos_strained")
    r = _mace("opt2cif", out.name, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    cif = (tmp_path / "dia_eos_strained.cif").read_text()
    assert "_cell_length_a    2.506560" in cif          # the primitive cell, as for the OPT parent


def test_a_module_deck_from_an_elastic_parent_keeps_the_reference_cell(tmp_path):
    """A RESTART continues the elastic run: its deck is written from the run's own output."""
    out = _parent(tmp_path, "dia_elapiezo_strained")
    r = _mace("opt2d12", "--out-file", out.name, "--calc-type", "ELASTIC", "--non-interactive",
              "--elastic-settings", '{"mode": "ELAPIEZO", "print": true}', "--output-dir", "o", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next((tmp_path / "o").glob("*.d12")).read_text().splitlines()
    assert deck[deck.index("227") + 1] == DIAMOND_A


def test_a_preoptimized_run_is_refused(tmp_path):
    """PREOPTGEOM optimizes cell and atoms (FULLOPTG) before the strains: the
    reference is that structure, which the output does not print where MACE
    could tell it apart. The parent is refused with what to use instead."""
    text = (MECH / "dia_eos_strained.out").read_text()
    head, sep, tail = text.partition(" EQUATION OF STATE CALCULATION\n")
    pre = head + sep + " INFORMATION **** FULLOPTG **** FULL OPTIMIZATION (ATOMS AND CELL PARAMETERS)\n" + tail
    why = mechanics_reference_problem(pre)
    assert "PREOPTGEOM" in why and "OPT .out" in why
    with pytest.raises(ValueError, match="PREOPTGEOM"):
        CrystalOutputParser(str(_parent(tmp_path, "dia_eos_strained", pre))).parse()
    r = _mace("opt2d12", "--out-file", "dia_eos_strained.out", "--non-interactive", "--calc-type", "SP",
              "--output-dir", "o", cwd=tmp_path)
    assert r.returncode != 0
    assert "PREOPTGEOM" in r.stdout + r.stderr
    assert not list((tmp_path / "o").glob("*.d12"))
    r = _mace("opt2cif", "dia_eos_strained.out", cwd=tmp_path)
    assert "PREOPTGEOM" in r.stdout + r.stderr
    assert not (tmp_path / "dia_eos_strained.cif").exists()


def test_a_central_point_off_the_input_cell_is_refused():
    """The central point at another volume than the input block's: the run
    started from a structure it did not print as its input."""
    text = (MECH / "dia_eos_strained.out").read_text()
    moved = text.replace(" VOLUME OF PRIMITIVE CELL (A^3)=          11.135746510281",
                         " VOLUME OF PRIMITIVE CELL (A^3)=          11.201746510281")
    assert moved != text
    assert "11.2017" in mechanics_reference_problem(moved)


def test_other_outputs_are_not_affected():
    sp = (REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain" / "dia_sp_dp_x_p1pct.out").read_text()
    assert mechanics_reference_problem(sp) is None
    assert mechanics_reference_problem((MECH / "dia_cphf.out").read_text()) is None
