"""2-component (SOC) matrix dumps: accept the full ones, refuse the silent ones.

MEASURED on HPCC (round-4 facts) on a real 2c-SOC Bi2 fort.9:

* the development ``properties`` printed the full 2c matrices: 124
  ``FOCK MATRIX (REAL PART)`` + 124 ``(IMAG PART)`` blocks under
  ``ALPHA_ALPHA`` / ``ALPHA_BETA`` / ``BETA_BETA`` headers;
* the STOCK ``properties`` on the same fort.9 ran WITHOUT error and printed
  only 62 scalar ``FOCK MATRIX - CELL`` blocks - a silently incomplete dump.

So a dump whose parent SCF was 2c but which holds only scalar Fock blocks is
refused, and ``mace opt2d3 --calc-type MATDUMP`` warns up front when the
parent is 2c. A parent is 2c when its .d12 opens a TWOCOMPON block (manual
sec. 6.2, p.170) or its .out prints the per-cycle magnetization components or
the spinor count only a 2c SCF prints (stock CRYSTAL23, fcc Au 2c-SCF).
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

import mace.wannier.driver as driver
from mace.wannier.driver import Lcao2WannierUnavailable, convert

MACE_CLI = REPO_ROOT / "mace_cli"

_OVERLAP = " OVERLAP MATRIX - CELL N.   1(  0  0  0)"
_SCALAR = [_OVERLAP, " FOCK MATRIX - CELL N.   1(  0  0  0)"]
_TWO_C = [_OVERLAP,
          " ALPHA_ALPHA ELECTRONS",
          " FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)",
          " FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)",
          " ALPHA_BETA ELECTRONS",
          " FOCK MATRIX (REAL PART) - CELL N.   1(  0  0  0)",
          " FOCK MATRIX (IMAG PART) - CELL N.   1(  0  0  0)"]

_SCF_1C = " TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL\n TOTAL ATOMIC SPINS\n"
_SCF_2C = (" TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL\n"
           " TOTAL X-COMP MAGNETIZATION    0.00000001\n"
           " TOTAL Y-COMP MAGNETIZATION   -0.00000001\n"
           " TOTAL Z-COMP MAGNETIZATION    0.00000000\n")
_SCF_2C_SPINORS = " - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70\n"

_DECK_1C = "Bi2 title\nSLAB\n72\n4.5\n1\n83 0 0 0\nEND\nEND\nSHRINK\n8 8\nEND\n"
_DECK_2C = _DECK_1C.replace("SHRINK", "TWOCOMPON\nSOC\nEND\nSHRINK")


def _material(tmp_path, dump_lines, scf_out=None, scf_deck=None, stem="bi2"):
    dump = tmp_path / f"{stem}_matdump.out"
    dump.write_text("\n".join(dump_lines) + "\n")
    if scf_out is not None:
        (tmp_path / f"{stem}_sp.out").write_text(scf_out)
    if scf_deck is not None:
        (tmp_path / f"{stem}_sp.d12").write_text(scf_deck)
    return dump


class _Completed:
    returncode, stdout, stderr = 0, "", ""


@pytest.fixture
def runs(monkeypatch):
    calls = []
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Completed())
    return calls


@pytest.mark.parametrize("scf_out, scf_deck", [
    (_SCF_2C, None),                     # output markers alone
    (_SCF_1C + _SCF_2C_SPINORS, None),   # the spinor count alone
    (_SCF_1C, _DECK_2C),                 # the deck's TWOCOMPON block alone
])
def test_a_scalar_dump_from_a_two_component_parent_is_refused(
        tmp_path, runs, scf_out, scf_deck):
    dump = _material(tmp_path, _SCALAR, scf_out, scf_deck)
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(dump, output_dir=tmp_path / "out")
    message = str(exc.value)
    assert "2-component" in message
    assert "FOCK MATRIX - CELL" in message
    assert "development" in message
    assert "bi2_sp" in message
    assert runs == [], "the conversion must not run on an incomplete dump"


def test_a_full_two_component_dump_is_accepted(tmp_path, runs):
    dump = _material(tmp_path, _TWO_C, _SCF_2C, _DECK_2C)
    convert(dump, output_dir=tmp_path / "out")
    assert len(runs) == 2


def test_a_two_component_dump_without_its_parent_is_accepted(tmp_path, runs):
    convert(_material(tmp_path, _TWO_C), output_dir=tmp_path / "out")
    assert len(runs) == 2


def test_a_scalar_dump_from_a_collinear_parent_is_accepted(tmp_path, runs):
    dump = _material(tmp_path, _SCALAR, _SCF_1C, _DECK_1C)
    convert(dump, output_dir=tmp_path / "out")
    assert len(runs) == 2


def test_real_and_imag_parts_without_spinor_labels_are_incomplete(tmp_path, runs):
    lines = [line for line in _TWO_C if "ELECTRONS" not in line]
    dump = _material(tmp_path, lines, _SCF_2C)
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(dump, output_dir=tmp_path / "out")
    assert "ALPHA_ALPHA" in str(exc.value)
    assert runs == []


def test_twocompon_in_the_title_line_is_not_a_two_component_deck(tmp_path, runs):
    deck = "TWOCOMPON\n" + _DECK_1C.split("\n", 1)[1]
    dump = _material(tmp_path, _SCALAR, _SCF_1C, deck)
    convert(dump, output_dir=tmp_path / "out")
    assert len(runs) == 2


def test_an_explicit_parent_elsewhere_is_checked(tmp_path, runs):
    elsewhere = tmp_path / "scf"
    elsewhere.mkdir()
    scf = elsewhere / "anything.out"
    scf.write_text(_SCF_2C)
    dump = _material(tmp_path, _SCALAR)
    with pytest.raises(Lcao2WannierUnavailable):
        convert(dump, output_dir=tmp_path / "out", scf_output=scf)
    assert runs == []


def test_a_scalar_dump_with_no_parent_found_says_so(tmp_path, runs):
    notes = []
    convert(_material(tmp_path, _SCALAR), output_dir=tmp_path / "out",
            progress=notes.append)
    assert len(runs) == 2
    assert any("--scf-output" in note for note in notes)


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_the_cli_refuses_with_the_reason_and_exit_2(tmp_path):
    dump = _material(tmp_path, _SCALAR, _SCF_2C)
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "wannier",
         "--input", str(dump), "--output-dir", str(tmp_path / "out")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "Traceback" not in combined
    assert "FOCK MATRIX - CELL" in combined


# --------------------------------------------------------------------------
# opt2d3 --calc-type MATDUMP warns when the parent is 2c
# --------------------------------------------------------------------------


def _scf_parent(tmp_path, out_extra="", deck=None, stem="dsoc_sp"):
    out = tmp_path / f"{stem}.out"
    out.write_text("\n".join([
        " TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL",
        " NO.OF VECTORS CREATED 6999 STARS 1021 RMAX   101.51010 BOHR",
        " NUMBER OF AO                36  EXCHANGE OVERLAP TOL        (T3) 10**   -8",
        " MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS 321",
    ]) + "\n" + out_extra)
    if deck is not None:
        (tmp_path / f"{stem}.d12").write_text(deck)
    (tmp_path / f"{stem}.f9").write_bytes(b"\x00" * 64)
    return out


def _opt2d3(tmp_path, out):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MACE_PROPERTIES_BINARY", "EBROOTCRYSTAL")}
    env["PATH"] = str(tmp_path / "empty") + os.pathsep + os.path.dirname(sys.executable)
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "opt2d3",
         "--input", str(out), "--calc-type", "MATDUMP"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=300)


@pytest.mark.parametrize("out_extra, deck", [
    ("", _DECK_2C),
    (" TOTAL X-COMP MAGNETIZATION    0.00000001\n", None),
])
def test_opt2d3_warns_that_a_two_component_parent_needs_the_development_build(
        tmp_path, out_extra, deck):
    proc = _opt2d3(tmp_path, _scf_parent(tmp_path, out_extra, deck))
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert (tmp_path / "dsoc_matdump.d3").exists()   # a warning, not a refusal
    assert "spin treatment: soc" in combined
    assert "FOCK MATRIX - CELL" in combined
    assert "development" in combined


def test_opt2d3_says_nothing_for_a_collinear_parent(tmp_path):
    proc = _opt2d3(tmp_path, _scf_parent(tmp_path, deck=_DECK_1C))
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "development" not in combined
