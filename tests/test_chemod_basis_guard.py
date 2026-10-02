"""A parent's CHEMOD is not carried into a child whose basis set differs.

CHEMOD gives one charge per shell of an atom's basis (CRYSTAL23 manual,
p.80), so it fits only the basis it was written for. full.basis.triplezeta
and its rev2022 update have different Rb-I and La-Lu records; a child moved
from one to the other (or to another internal set) would get the parent's
shell charges on the wrong shells. opt2d12 refuses such a child by name.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

from charge_records import chemod_basis_change, deck_basis  # noqa: E402
import basis_library as BL  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "data" / "opt_restart"
TZ = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta") + "/"
REV = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta.rev2022") + "/"

INTERNAL = """t
MOLECULE
1
1
3 0.0 0.0 0.0 Li
END
BASISSET
{basis}
{chemod}DFT
PBE
ENDDFT
END
"""
CHEMOD = "CHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\n"


def internal(basis, chemod=""):
    return INTERNAL.format(basis=basis, chemod=chemod)


def explicit(records, chemod=""):
    head = "t\nCRYSTAL\n0 0 0\n225\n5.0\n2\n237 0 0 0\n53 .5 .5 .5\nEND\n"
    return head + "".join(records) + "99 0\n" + chemod + "END\nDFT\nPBE\nENDDFT\nEND\n"


def test_internal_basis_change_is_named():
    reason = chemod_basis_change(internal("POB-DZVP-REV2"), internal("SOLDEF2MSVP", CHEMOD))
    assert "BASISSET SOLDEF2MSVP, this deck: BASISSET POB-DZVP-REV2" in reason
    assert chemod_basis_change(internal("SOLDEF2MSVP"), internal("SOLDEF2MSVP", CHEMOD)) is None


def test_no_chemod_or_the_childs_own_chemod_is_not_judged():
    assert chemod_basis_change(internal("POB-DZVP-REV2"), internal("SOLDEF2MSVP")) is None
    assert chemod_basis_change(internal("POB-DZVP-REV2", CHEMOD),
                               internal("SOLDEF2MSVP", CHEMOD)) is None


def test_tz_to_rev2022_is_refused_and_spacing_is_not_a_change():
    tz, rev = BL.load_family("full.basis.triplezeta"), BL.load_family(
        "full.basis.triplezeta.rev2022")
    parent = explicit([tz.record("237"), tz.record("253")], CHEMOD)
    reason = chemod_basis_change(explicit([rev.record("237"), rev.record("253")]), parent)
    assert "atom type 237: same 8 shells, different exponents or coefficients" in reason
    respaced = [" ".join(line.split()) + "\n" for line in
                (tz.record("237") + tz.record("253")).splitlines()]
    assert chemod_basis_change(explicit(respaced), parent) is None


def test_a_different_shell_count_is_named():
    tz, rev = BL.load_family("full.basis.triplezeta"), BL.load_family(
        "full.basis.triplezeta.rev2022")
    parent = explicit([tz.record("257")], CHEMOD)
    reason = chemod_basis_change(explicit([rev.record("257")]), parent)
    assert "atom type 257: 9 shells in the parent, 11 in this deck" in reason
    assert BL.shell_count("full.basis.triplezeta", "257") == 9
    assert BL.shell_count("full.basis.triplezeta.rev2022", "257") == 11


def test_internal_against_explicit_is_a_change():
    reason = chemod_basis_change(explicit([BL.load_family(
        "full.basis.triplezeta").record("237")]), internal("POB-TZVP-REV2", CHEMOD))
    assert "parent: internal basis, this deck: external basis" in reason


def test_the_basis_block_of_a_real_deck_is_read():
    kind, records = deck_basis((SAMPLE / "tqc_agbr.d12").read_text())
    assert kind == "EXTERNAL" and sorted(records) == [35, 247]


def _opt2d12(tmp_path, options):
    work = tmp_path / "work"
    work.mkdir()
    (work / "p.out").write_bytes((SAMPLE / "tqc_agbr.out").read_bytes())
    deck = (SAMPLE / "tqc_agbr.d12").read_text().replace(
        "99 0\n", "99 0\nCHEMOD\n1\n1 2.0 2.0 8.0 2.0 8.0 10.0 0.0 0.0 0.0 0.0 0.0\n", 1)
    (work / "p.d12").write_text(deck)
    (work / "opts.json").write_text(json.dumps(options))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "p.out",
         "--d12-file", "p.d12", "--calc-type", "SP", "--non-interactive",
         "--config-file", "opts.json", "--output-dir", "out"],
        cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300), work


@pytest.mark.parametrize("path, refused", [(TZ, False), (REV, True)])
def test_opt2d12_refuses_a_chemod_child_moved_to_another_basis(tmp_path, path, refused):
    """Real CLI. The Ag2Br3 parent's own records are pob-TZVP 2018 for Ag:
    rewriting them from full.basis.triplezeta keeps CHEMOD; switching to the
    rev2022 family is refused by name and leaves no deck."""
    result, work = _opt2d12(tmp_path, {
        "basis_set_type": "EXTERNAL", "basis_set_path": path, "basis_set": path,
        "use_original_external_basis": False})
    log = result.stdout + result.stderr
    decks = list((work / "out").glob("*.d12")) if (work / "out").exists() else []
    if refused:
        assert "atom type 247: same 11 shells, different exponents" in log, log[-2000:]
        assert decks == []
        assert result.returncode != 0
    else:
        assert result.returncode == 0, log[-2000:]
        (deck,) = decks
        assert "CHEMOD" in deck.read_text()
