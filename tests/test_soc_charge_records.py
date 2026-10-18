"""SOC decks and the charged-system records (1.1.6).

The charge records (CHEMOD, CHARGED, DOPING, SPINLOCK) go into a deck after
soc_deck wrote it: cif2d12's and opt2d12's charge options, or a parent's
records carried over. A SOC deck keeps only what soc_deck keeps of a deck's
own (CHEMOD, and not with soc_basis); CHARGED, DOPING and SPINLOCK are not in
the two-component chapter (p. 166), so the deck is refused with that reason.
Before, these records reached a TWOCOMPON deck unchecked.
"""
import pytest

import soc_ecp as E
from test_soc_decks import opt2d12  # noqa: F401  (the fixture)

SOC = """t
CRYSTAL
0 0 0
225
6.462
2
282 0. 0. 0.
252 .5 .5 .5
END
282 1
INPSOC
INTERNAL 1.0
99 0
{basis_input}END
DFT
SPIN
PBE0
ENDDFT
SHRINK
8 16
TWOCOMPON
SOC
END
SCFDIR
FMIXING
50
{scf}END
"""


def _deck(basis_input="", scf=""):
    return SOC.format(basis_input=basis_input, scf=scf)


def test_a_soc_deck_without_charge_records_and_a_scalar_deck_are_fine():
    assert E.charge_records_refusal(_deck()) is None
    assert E.charge_records_refusal(_deck(scf="DOPING\n-1\n").replace("TWOCOMPON\nSOC\nEND\n", "")) is None


def test_chemod_is_kept_but_not_with_soc_basis():
    deck = _deck(basis_input="CHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\n")
    assert E.charge_records_refusal(deck) is None
    why = E.charge_records_refusal(deck, "pob-tzvp-rev2-soc")
    assert why.startswith("CHEMOD gives charges for the shells")


def test_charged_is_refused_by_name():
    why = E.charge_records_refusal(_deck(basis_input="CHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\nCHARGED\n"))
    assert why.startswith("CHARGED in the basis-set input (after '99 0')") and "p. 166" in why


@pytest.mark.parametrize("scf, names", [("DOPING\n-1\n", "DOPING"), ("SPINLOCK\n1 30\n", "SPINLOCK"),
                                        ("DOPING\n-1\nSPINLOCK\n1 30\n", "DOPING and SPINLOCK")])
def test_doping_and_spinlock_are_refused_by_name(scf, names):
    why = E.charge_records_refusal(_deck(scf=scf))
    assert why.startswith(f"{names} in the SCF input (the charged-system settings)") and "p. 166" in why


def test_opt2d12_refuses_charge_options_on_a_soc_deck(opt2d12, capsys):  # noqa: F811
    status, decks = opt2d12(["pbte"], soc=True, net_charge=2)
    err = capsys.readouterr()
    assert status != 0
    assert not decks["pbte"].exists()
    said = " ".join((err.out + err.err).split())
    assert "the charged-system options (ion_charges, net_charge, site_states) are not written into a SOC deck" in said
    status, decks = opt2d12(["pbte"], net_charge=2)          # scalar: DOPING as before
    assert status == 0 and "DOPING\n-2\n" in decks["pbte"].read_text()


def test_convert_refuses_charged_in_a_soc_deck_and_keeps_it_without_soc(tmp_path):
    pytest.importorskip("ase")
    import test_ion_charges_cli as T
    from conftest import REPO_ROOT
    basis = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta") + "/"
    # Li+(EC)2 with an I2 molecule beside it: an ECP atom for SOC, CHEMOD + CHARGED for the Li+
    cif = T.molecule_cif(T.li_ec2() + [("I", (20.0, 0.0, 0.0)), ("I", (22.67, 0.0, 0.0))], name="li_ec2")
    opts = dict(T.MOLECULE, dft_functional="PBE0", basis_set=basis, basis_set_path=basis, basis_set_type="EXTERNAL")
    (tmp_path / "a").mkdir()
    run, deck = T._convert(tmp_path / "a", "li_ec2", cif, opts, "--ion-charges", "auto")
    assert run.returncode == 0 and "CHARGED" in deck.splitlines()
    (tmp_path / "b").mkdir()
    run, deck = T._convert(tmp_path / "b", "li_ec2", cif, dict(opts, soc=True), "--ion-charges", "auto")
    said = " ".join((run.stdout + run.stderr).split())
    assert run.returncode != 0 and deck is None
    assert "Not writing li_ec2_MOLECULE_SP_P1_PBE0_SOC_full.basis.triplezeta.d12: CHARGED in the basis-set input" in said
