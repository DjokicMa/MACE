"""A derived deck with a different basis set, or an odd electron count, keeps the parent's charge state.

Three bugs in carrying CHEMOD / CHARGED / DOPING into a child deck
(charge_records.carry_charge_records):

* E2: a parent with an explicit basis carried its charge in CHEMOD + CHARGED
  (measured on HPCC: with an explicit basis a non-neutral CHEMOD sets the
  electron count). A child loading an internal BASISSET got the same CHEMOD +
  CHARGED, which with an internal library only reshape the guess (measured:
  the guess is renormalized and the run stays neutral), so the child silently
  ran neutral. The charge becomes DOPING.
* E5: CHEMOD was copied whenever both decks listed the same number of atoms,
  although its shell charges belong to the parent's basis (CRYSTAL wants one
  per shell of the atom's basis, manual p.80). It is now copied only with the
  same basis. With another basis, a CHEMOD that MACE's ion charges wrote for
  the parent is worked out again for the new basis (decided D2); any other is
  left out with a warning. The net charge is always kept.
* E3: an odd electron count in a closed-shell child was written without a
  spin state. It gets the parent's SPIN/SPINLOCK, or is refused.
"""
import subprocess
import sys

import pytest

from basis_library import read_record
from charge_records import ChargeCarryError, carry_charge_records
from conftest import REPO_ROOT
from ion_fixtures import molecule_deck, mulliken_block, synth_out

TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"
LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]
LI_PLUS = "1 2.0 0.0 0.0 0.0 0.0"
# Li+ with its two electrons moved to the second shell: not what MACE's ion
# charges write, so MACE cannot say what it means in another basis
HAND_MADE = "1 1.0 1.0 0.0 0.0 0.0"


def _explicit(zs=(1, 3, 8)):
    return "END\n" + "".join(read_record(str(TZ), z) for z in zs) + "99 0\n"


def explicit_parent(records="CHEMOD\n1\n" + LI_PLUS + "\nCHARGED\n"):
    return molecule_deck(LI_WATER, title="p", basis=_explicit() + records + "END\n", dft="PBE0\nXLGRID\n")


def internal(name="POB-TZVP-REV2", records="", spin=False, tail=""):
    return molecule_deck(LI_WATER, title="c", basis=f"BASISSET\n{name}\n{records}", dft="PBE0\nXLGRID\n",
                         spin=spin, tail=tail)


def _lines(text):
    return text.splitlines()


# --------------------------------------------------------------------- E2

def test_explicit_chemod_charge_becomes_doping_in_an_internal_basis_child():
    """MACE's own Li+ CHEMOD is worked out again for the library (D2): its
    charge, set by CHEMOD + CHARGED in the explicit basis, is DOPING there."""
    out, notes = carry_charge_records(internal(), explicit_parent())
    lines = _lines(out)
    assert "CHARGED" not in lines
    assert lines[lines.index("CHEMOD") + 1:lines.index("CHEMOD") + 3] == ["1", LI_PLUS]
    assert lines[-3:] == ["DOPING", "-1", "END"]
    assert any("worked out again for the new basis" in n for n in notes)


def test_a_hand_made_explicit_chemod_is_dropped_and_its_charge_becomes_doping():
    parent = explicit_parent("CHEMOD\n1\n" + HAND_MADE + "\nCHARGED\n")
    out, notes = carry_charge_records(internal(), parent)
    lines = _lines(out)
    assert "CHEMOD" not in lines and "CHARGED" not in lines
    assert lines[-3:] == ["DOPING", "-1", "END"]
    assert any("CHEMOD was NOT copied" in n and "not the CHEMOD MACE's ion charges write" in n
               for n in notes)


def test_the_parent_mulliken_count_sets_the_doping():
    atoms = [("LI", 3), ("O", 8), ("H", 1), ("H", 1)]
    out_text = mulliken_block(atoms, [2.0, 8.8, 0.6, 0.6], 12.0)
    out, _ = carry_charge_records(internal(), explicit_parent(), out_text)
    assert _lines(out)[-3:] == ["DOPING", "-1", "END"]


def test_same_explicit_basis_still_keeps_chemod_and_charged():
    child = molecule_deck(LI_WATER, title="c", basis=_explicit() + "END\n", dft="PBE0\nXLGRID\n")
    out, _ = carry_charge_records(child, explicit_parent())
    lines = _lines(out)
    i = lines.index("99 0")
    assert lines[i + 1:i + 5] == ["CHEMOD", "1", LI_PLUS, "CHARGED"]
    assert "DOPING" not in lines


# --------------------------------------------------------------------- E5

def test_internal_chemod_is_worked_out_again_for_another_library():
    """POB-DZVP-REV2 gives Li one shell fewer than SOLDEF2MSVP: the copied
    record would have had one charge too many (CRYSTAL wants one per shell)."""
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + LI_PLUS + "\n", tail="DOPING\n-1\n")
    out, notes = carry_charge_records(internal("POB-DZVP-REV2"), parent)
    lines = _lines(out)
    assert lines[lines.index("CHEMOD") + 1:lines.index("CHEMOD") + 3] == ["1", "1 2.0 0.0 0.0 0.0"]
    assert lines[-3:] == ["DOPING", "-1", "END"]


def test_hand_made_internal_chemod_is_not_copied_into_another_library():
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + HAND_MADE + "\n", tail="DOPING\n-1\n")
    out, notes = carry_charge_records(internal("POB-TZVP-REV2"), parent)
    lines = _lines(out)
    assert "CHEMOD" not in lines
    assert lines[-3:] == ["DOPING", "-1", "END"]
    assert any("different one" in n for n in notes)


def test_a_library_mace_has_no_shells_for_drops_chemod_and_says_why():
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + LI_PLUS + "\n", tail="DOPING\n-1\n")
    out, notes = carry_charge_records(internal("DEF2-TZVP"), parent)
    lines = _lines(out)
    assert "CHEMOD" not in lines and lines[-3:] == ["DOPING", "-1", "END"]
    assert any("CHEMOD was NOT copied" in n and "DEF2-TZVP" in n for n in notes)


def test_same_library_keeps_chemod():
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + LI_PLUS + "\n", tail="DOPING\n-1\n")
    out, _ = carry_charge_records(internal("SOLDEF2MSVP"), parent)
    assert "CHEMOD" in _lines(out)


def test_internal_chemod_enters_an_explicit_child_as_its_own():
    """In an explicit basis MACE's CHEMOD sets the count (with CHARGED), so
    the charge is no longer DOPING."""
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + LI_PLUS + "\n", tail="DOPING\n-1\n")
    child = molecule_deck(LI_WATER, title="c", basis=_explicit() + "END\n", dft="PBE0\nXLGRID\n")
    out, _ = carry_charge_records(child, parent)
    lines = _lines(out)
    i = lines.index("99 0")
    assert lines[i + 1:i + 5] == ["CHEMOD", "1", LI_PLUS, "CHARGED"]
    assert "DOPING" not in lines


def test_a_different_explicit_basis_drops_a_hand_made_chemod_and_keeps_the_charge():
    blocks = _explicit().replace("0 0 1 0.0 1.0\n  0.1200746200", "0 0 1 0.0 1.0\n  0.1300000000")
    child = molecule_deck(LI_WATER, title="c", basis=blocks + "END\n", dft="PBE0\nXLGRID\n")
    out, _ = carry_charge_records(child, explicit_parent("CHEMOD\n1\n" + HAND_MADE + "\nCHARGED\n"))
    lines = _lines(out)
    assert "CHEMOD" not in lines and lines[-3:] == ["DOPING", "-1", "END"]


def test_a_different_explicit_basis_gets_maces_chemod_again():
    blocks = _explicit().replace("0 0 1 0.0 1.0\n  0.1200746200", "0 0 1 0.0 1.0\n  0.1300000000")
    child = molecule_deck(LI_WATER, title="c", basis=blocks + "END\n", dft="PBE0\nXLGRID\n")
    out, _ = carry_charge_records(child, explicit_parent())
    lines = _lines(out)
    i = lines.index("99 0")
    assert lines[i + 1:i + 5] == ["CHEMOD", "1", LI_PLUS, "CHARGED"]
    assert "DOPING" not in lines


def test_doping_into_a_different_atom_count_warns():
    parent = internal("SOLDEF2MSVP", tail="DOPING\n-1\n")
    bigger = molecule_deck(LI_WATER + [("H", (6.0, 0, 0)), ("H", (6.74, 0, 0))], title="c",
                           basis="BASISSET\nSOLDEF2MSVP\n", dft="PBE0\nXLGRID\n")
    out, notes = carry_charge_records(bigger, parent)
    assert _lines(out)[-3:] == ["DOPING", "-1", "END"]
    assert any("copied unchanged into a deck with 6 atoms" in n for n in notes)


# --------------------------------------------------------------------- E3

ODD_TAIL = "DOPING\n-2\n"  # Li + H2O is 13 electrons; -2 leaves 11


def test_odd_count_child_gets_the_parents_spin_state():
    parent = internal("SOLDEF2MSVP", spin=True, tail=ODD_TAIL + "SPINLOCK\n1 30\n")
    out, notes = carry_charge_records(internal("SOLDEF2MSVP"), parent)
    lines = _lines(out)
    assert lines[lines.index("ENDDFT") - 1] == "SPIN"
    assert lines[-5:] == ["DOPING", "-2", "SPINLOCK", "1 30", "END"]
    assert any("SPINLOCK 1 30" in n for n in notes)


def test_odd_count_child_without_a_parent_spin_state_is_refused():
    parent = internal("SOLDEF2MSVP", tail=ODD_TAIL)
    with pytest.raises(ChargeCarryError, match="11 electrons"):
        carry_charge_records(internal("SOLDEF2MSVP"), parent)


def test_even_count_child_is_left_alone():
    parent = internal("SOLDEF2MSVP", tail="DOPING\n-1\n")
    out, notes = carry_charge_records(internal("SOLDEF2MSVP"), parent)
    assert "SPIN" not in _lines(out)


# ------------------------------------------------- the real opt2d12 command

def _opt2d12(tmp_path, parent_deck, parent_out, *extra):
    if parent_deck is not None:
        (tmp_path / "parent.d12").write_text(parent_deck)
    (tmp_path / "parent.out").write_text(parent_out)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "parent.out",
         "--calc-type", "SP", "--non-interactive", "--output-dir", "o", *extra],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)


def test_opt2d12_refuses_an_odd_closed_shell_child_and_writes_nothing(tmp_path):
    run = _opt2d12(tmp_path, internal("SOLDEF2MSVP", tail=ODD_TAIL), synth_out(LI_WATER))
    assert run.returncode != 0
    assert "odd" in run.stdout + run.stderr
    assert not list((tmp_path / "o").glob("*.d12")) if (tmp_path / "o").exists() else True


def test_odd_periodic_child_is_only_warned_about():
    cell = "CRYSTAL\n0 0 0\n1\n10.0 10.0 10.0 90.0 90.0 90.0\n"
    rows = "4\n3 0.0 0.0 0.0\n8 0.195 0.0 0.0\n1 0.254 0.076 0.0\n1 0.254 -0.076 0.0\n"
    tail = "END\nBASISSET\nSOLDEF2MSVP\nDFT\nPBE0\nENDDFT\nSHRINK\n1 1\n{}END\n"
    parent = "p\n" + cell + rows + tail.format(ODD_TAIL)
    child = "c\n" + cell + rows + tail.format("")
    out, notes = carry_charge_records(child, parent)
    assert _lines(out)[-3:] == ["DOPING", "-2", "END"]
    assert any("half-filled band" in n for n in notes)


def test_opt2d12_basis_change_writes_the_chemod_for_the_new_basis(tmp_path):
    """The real command: a basis change in an SP child of MACE's charged Li+
    parent (D2: regenerate CHEMOD, keep the net charge)."""
    (tmp_path / "c.json").write_text('{"calculation_type": "SP", "basis_set": "POB-DZVP-REV2", '
                                     '"basis_set_type": "INTERNAL"}')
    parent = internal("SOLDEF2MSVP", "CHEMOD\n1\n" + LI_PLUS + "\n", tail="DOPING\n-1\n")
    run = _opt2d12(tmp_path, parent, synth_out(LI_WATER), "--d12-file", "parent.d12",
                   "--config-file", "c.json")
    assert run.returncode == 0, (run.stdout + run.stderr)[-2000:]
    (deck,) = (tmp_path / "o").glob("*.d12")
    lines = deck.read_text().splitlines()
    i = lines.index("POB-DZVP-REV2")
    assert lines[i + 1:i + 4] == ["CHEMOD", "1", "1 2.0 0.0 0.0 0.0"]
    assert lines[-3:] == ["DOPING", "-1", "END"]
