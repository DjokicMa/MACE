"""CRYSTAL decks with IFSO > 1 carry an origin-shift record (manual p.19).

``IFLAG IFHR IFSO`` with IFSO > 1 puts an ``IX IY IZ`` record after the space
group line. charge_records._geometry_atom_count and
ion_charges.read_deck_geometry skipped it, so `mace ions` refused valid decks
with confusing errors and the CHEMOD carry of opt2d12 dropped CHEMOD; and
opt2d12 kept the parent's ``0 0 2`` without the record, writing a deck
CRYSTAL misreads. Each path must behave as for the IFSO = 0 control.
"""
import os
import shutil
import subprocess
import sys

import pytest

import charge_records as cr
import ion_charges as ic
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"

P1 = """LiClO4 P1
CRYSTAL
0 0 {ifso}
1
{shift}12.0 12.0 12.0 90.0 90.0 90.0
6
8   0.5693 0.5693 0.5693
3   0.1000 0.1000 0.1000
17  0.5000 0.5000 0.5000
8   0.4307 0.4307 0.5693
8   0.4307 0.5693 0.4307
8   0.5693 0.4307 0.4307
END
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
SHRINK
2 2
END
"""

NACL = """NaCl
CRYSTAL
0 0 {ifso}
225
{shift}5.64
2
11 0.0 0.0 0.0
17 0.5 0.5 0.5
END
BASISSET
POB-TZVP-REV2
DFT
PBE
END
SHRINK
8 8
END
"""


def _decks(template):
    return template.format(ifso=0, shift=""), template.format(ifso=2, shift="0 0 0\n")


def _strip_origin(text):
    lines = text.splitlines()
    i = lines.index("CRYSTAL")
    flags = lines[i + 1]
    out = lines[:i + 1] + lines[i + 2:i + 3]
    rest = lines[i + 3:]
    if cr.has_origin_shift(flags):
        rest = rest[1:]
    return out + rest


def _run(args, cwd):
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    return subprocess.run([sys.executable, str(MACE_CLI)] + args, cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("template", [P1, NACL], ids=["P1", "NaCl"])
def test_atom_count_and_geometry_read_past_the_shift_record(template):
    control, shifted = _decks(template)
    assert cr._geometry_atom_count(shifted.splitlines()) == cr._geometry_atom_count(control.splitlines())
    a, b = ic.read_deck_geometry(control), ic.read_deck_geometry(shifted)
    assert (b.symbols, b.xyz, b.cell) == (a.symbols, a.xyz, a.cell)


def test_mace_ions_net_charge_on_a_shifted_deck_matches_the_control(tmp_path):
    control, shifted = _decks(NACL)
    (tmp_path / "ifso0.d12").write_text(control)
    (tmp_path / "ifso2.d12").write_text(shifted)
    for name in ("ifso0", "ifso2"):
        run = _run(["ions", f"{name}.d12", "--net-charge", "-1", "--non-interactive", "-o", "out"], tmp_path)
        assert run.returncode == 0, run.stdout + run.stderr
    a = (tmp_path / "out" / "ifso0.d12").read_text()
    b = (tmp_path / "out" / "ifso2.d12").read_text()
    assert "DOPING" in b
    assert _strip_origin(b) == _strip_origin(a)


def test_mace_ions_chemod_on_a_shifted_p1_deck_matches_the_control(tmp_path):
    control, shifted = _decks(P1)
    (tmp_path / "ifso0.d12").write_text(control)
    (tmp_path / "ifso2.d12").write_text(shifted)
    for name in ("ifso0", "ifso2"):
        run = _run(["ions", f"{name}.d12", "--ion-charges", "auto", "--chemod", "yes", "--non-interactive",
                    "-o", "out"], tmp_path)
        assert run.returncode == 0, run.stdout + run.stderr
    a = (tmp_path / "out" / "ifso0.d12").read_text()
    b = (tmp_path / "out" / "ifso2.d12").read_text()
    assert "CHEMOD" in b
    assert _strip_origin(b) == _strip_origin(a)


def test_carry_keeps_chemod_of_a_shifted_parent():
    control, shifted = _decks(NACL)
    chemod = "POB-TZVP-REV2\nCHEMOD\n1\n1 2 8 8 0 0 0 0\n"
    for parent in (control, shifted):
        out, notes = cr.carry_charge_records(control, parent.replace("POB-TZVP-REV2\n", chemod))
        assert "CHEMOD" in out.splitlines(), notes
        assert any("kept the parent's CHEMOD" in n for n in notes)


def test_origin_shift_record():
    assert cr.origin_shift_record([0.0, 6.0, 12.0]) == "0 6 12"
    with pytest.raises(ValueError, match="origin-shift record"):
        cr.origin_shift_record(None)
    assert not cr.has_origin_shift("0 0 1") and cr.has_origin_shift("0 0 2")


AGCL = "Ag1Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"
AGCL_OUT = TEST_DATA / "SP" / f"{AGCL}.out"
AGCL_D12 = TEST_DATA / "SP" / f"{AGCL}.d12"


@pytest.mark.skipif(not AGCL_OUT.exists(), reason="needs the test/ corpus")
@pytest.mark.parametrize("shift", ["0 0 0", "6 6 6"])
def test_opt2d12_child_of_a_shifted_chemod_parent_matches_the_control(tmp_path, shift):
    """The real invocation path: opt2d12 on a CHEMOD parent with IFSO = 2."""
    deck = AGCL_D12.read_text().replace(
        "99 0\nEND\n", "99 0\nCHEMOD\n1\n2 2.0 8.0 2.0 8.0 8.0 0.0 0.0\nEND\n", 1)
    assert "CHEMOD" in deck
    children = {}
    for name, text in (("control", deck),
                       ("shifted", deck.replace("CRYSTAL\n0 0 0\n221\n", f"CRYSTAL\n0 0 2\n221\n{shift}\n", 1))):
        d = tmp_path / name
        d.mkdir()
        shutil.copy(AGCL_OUT, d / "parent.out")
        (d / "parent.d12").write_text(text)
        run = _run(["opt2d12", "--out-file", "parent.out", "--calc-type", "SP", "--non-interactive",
                    "--output-dir", "o"], d)
        assert run.returncode == 0, run.stdout + run.stderr
        assert "kept the parent's CHEMOD" in run.stdout + run.stderr
        children[name] = next((d / "o").glob("*.d12")).read_text()
    a, b = children["control"], children["shifted"]
    lines = b.splitlines()
    i = lines.index("CRYSTAL")
    # the child keeps the parent's origin: its flag and its shift record
    assert lines[i + 1:i + 4] == ["0 0 2", "221", shift]
    assert _strip_origin(b) == _strip_origin(a)
    assert "CHEMOD" in b.splitlines()
