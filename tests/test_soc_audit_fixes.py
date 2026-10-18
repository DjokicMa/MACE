"""SOC deck fixes from the scientific audit of the SOC branch (B1-B6).

B1  an all-electron atom gets no spin-orbit term (the SOC operator lives in
    the ECP, manual p. 175): the default INPSOC/STUTSC path says so, naming
    the atoms, and warns for Z >= 31, where the lost splitting is large.
B2  basis-input records between '99 0' and its END (CHEMOD, GHOSTS,
    CHARGED) were taken for block 3 and refused as "END". CHEMOD, which the
    manual pairs with the 2c atomic guess (p. 172), is kept; GHOSTS and
    CHARGED, which chapter 6 never mentions, are refused by name.
B3  an ECP atom labeled above 300 (382: Pb, a distinct label) is Z = 82.
B4  the official SOC basis is read through basis_library.soc_basis_record;
    no second loader or fetch script.
B5  full.basis.triplezeta.rev2022 is one of the library families.
B6  sopseud/219.mol (K) lists its d terms twice: refused, not converted.
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import soc_ecp as S  # noqa: E402
from test_soc_decks import PB, TE, _deck, _library_block  # noqa: E402

BR = _library_block(35)      # all-electron Br in full.basis.triplezeta
O_AE = _library_block(8)


def _two_atom_deck(first, second, block2=(), labels=("282", "252")):
    return "\n".join(["PbX", "CRYSTAL", "0 0 0", "225", "6.462", "2",
                      f"{labels[0]} 0.0 0.0 0.0", f"{labels[1]} 0.5 0.5 0.5", "END",
                      *first, *second, "99 0", *block2, "END",
                      "DFT", "PBE0", "ENDDFT", "TOLINTEG", "7 7 7 7 14", "TOLDEE", "7",
                      "SHRINK", "8 8", "SCFDIR", "MAXCYCLE", "800", "FMIXING", "30",
                      "PPAN", "END"]) + "\n"


# --- B1 ----------------------------------------------------------------------

def test_an_all_electron_4p_atom_is_named_with_a_warning():
    notes = []
    S.soc_deck(_two_atom_deck(PB, BR, labels=("282", "35")), log=notes.append)
    warned = [n for n in notes if "Br" in n and n.startswith("Warning")]
    assert warned, notes
    text = warned[0]
    assert "no spin-orbit" in text and "all-electron" in text
    assert "stuttgart" in text          # Ga-Se: the stuttgart ECP10 records match
    assert "p. 175" in text


def test_a_light_all_electron_atom_gets_a_note_not_a_warning():
    notes = []
    S.soc_deck(_two_atom_deck(PB, O_AE, labels=("282", "8")), log=notes.append)
    about_o = [n for n in notes if " O" in n and "all-electron" in n]
    assert about_o and not any(n.startswith("Warning") for n in about_o), notes


def test_the_stutsc_path_names_all_electron_atoms_too():
    notes = []
    S.soc_deck(_two_atom_deck(PB, BR, labels=("282", "35")), log=notes.append,
               ecp_form="stutsc")
    assert any(n.startswith("Warning") and "Br" in n for n in notes), notes


def test_ecp_only_decks_get_no_all_electron_note():
    notes = []
    S.soc_deck(_deck(PB + TE), log=notes.append)
    assert not any("all-electron" in n for n in notes), notes


# --- B2 ----------------------------------------------------------------------

def _chemod_for_pb():
    charges = [ln.split()[3] for ln in PB[3 + 14:] if len(ln.split()) == 5]
    assert len(charges) == int(PB[0].split()[1])
    return ["CHEMOD", "1", "1 " + " ".join(charges)]


def test_chemod_after_99_0_is_kept_and_the_deck_is_written():
    chemod = _chemod_for_pb()
    soc = S.soc_deck(_two_atom_deck(PB, TE, block2=chemod), log=lambda m: None).split("\n")
    i = soc.index("99 0")
    assert soc[i + 1:i + 1 + len(chemod)] == chemod
    assert soc[i + 1 + len(chemod)] == "END"
    j = soc.index("SCFDIR")
    assert soc[j - 3:j] == ["TWOCOMPON", "SOC", "END"]


@pytest.mark.parametrize("records,name", [
    (["GHOSTS", "1", "2"], "GHOSTS"),
    (["CHARGED"], "CHARGED"),
])
def test_ghosts_and_charged_are_refused_by_name(records, name):
    with pytest.raises(S.SocError) as err:
        S.soc_deck(_two_atom_deck(PB, TE, block2=records), log=lambda m: None)
    message = str(err.value)
    assert name in message and "basis-set input" in message
    assert ": END" not in message


# --- B3 ----------------------------------------------------------------------

def test_an_ecp_label_above_300_is_its_element():
    pb382 = ["382 " + PB[0].split()[1]] + PB[1:]
    soc = S.soc_deck(_two_atom_deck(pb382, TE, labels=("382", "252")),
                     log=lambda m: None).split("\n")
    i = soc.index(pb382[0])
    assert soc[i + 1:i + 4] == ["INPSOC", "INTERNAL 1.0", "22. 0 2 4 4 2 2"]


def test_an_ecp_label_above_300_with_stutsc():
    pb382 = ["382 " + PB[0].split()[1]] + PB[1:]
    soc = S.soc_deck(_two_atom_deck(pb382, TE, labels=("382", "252")),
                     log=lambda m: None, ecp_form="stutsc").split("\n")
    assert soc[soc.index(pb382[0]) + 1] == "STUTSC"


# --- B4 ----------------------------------------------------------------------

def test_the_official_soc_basis_is_read_through_basis_library():
    import basis_library
    sets = S.load_soc_basis("pob-tzvp-rev2-soc")
    assert sorted(sets) == basis_library.soc_basis_elements()
    for z, lines in sets.items():
        expected = [ln.rstrip() for ln in basis_library.soc_basis_record(z).split("\n")
                    if ln.strip()]
        assert lines == expected


def test_no_second_loader_or_fetch_script():
    assert not (REPO_ROOT / "Crystal_d12" / "basis_sets" / "fetch_soc_basis.py").exists()
    source = (REPO_ROOT / "Crystal_d12" / "soc_ecp.py").read_text()
    assert "TODO" not in source
    assert "SOC_BASIS_DIR" not in source


# --- B5 ----------------------------------------------------------------------

def test_rev2022_is_a_library_family():
    assert "full.basis.triplezeta.rev2022" in S.LIBRARY_FAMILIES
    import basis_library
    record = basis_library.load_family("full.basis.triplezeta.rev2022").record("237")
    lines = [ln for ln in record.splitlines() if ln.strip()]
    end = 3 + sum(int(c) for c in lines[2].split()[1:])
    assert [ln.split() for ln in lines[end:]] in S._library_shells(37)


# --- B6 ----------------------------------------------------------------------

def test_only_k_lists_a_term_twice():
    flagged = {z: S.duplicated_terms(S.load_so_ecp(z)) for z in S.available_so_ecps()}
    assert [z for z, d in flagged.items() if d] == [19]
    assert all(block.startswith("d") for block in flagged[19])


def test_k_is_refused_not_converted():
    with pytest.raises(S.SocError, match="K.*twice"):
        S.soc_ecp_records(19, ["INPUT", "9. 0 0 0 0 0 0"])


# --- every MACE writer refuses 2c OPT/FREQ and ANDERSON/BROYDEN (p. 166) ------

from test_soc_decks import cif2d12, opt2d12  # noqa: E402,F401


@pytest.mark.parametrize("record", ["ANDERSON", "BROYDEN\n0.0001 50 2"])
def test_soc_deck_refuses_anderson_and_broyden(record):
    deck = _deck(PB + TE).replace("DIIS\n", record + "\n")
    with pytest.raises(S.SocError, match=record.split()[0]):
        S.soc_deck(deck, log=lambda m: None)


@pytest.mark.parametrize("extra,why", [
    ({"calculation_type": "OPT"}, "OPTGEOM"),
    ({"calculation_type": "FREQ"}, "FREQCALC"),
    ({"scf_method": "ANDERSON"}, "ANDERSON"),
    ({"scf_method": "BROYDEN"}, "BROYDEN"),
])
def test_cif2d12_writes_no_soc_deck_crystal_cannot_run(cif2d12, capsys, extra, why):
    ok, out = cif2d12("external", soc=True, **extra)
    captured = capsys.readouterr()
    assert ok is not True and not out.exists()
    assert why in captured.out + captured.err


@pytest.mark.parametrize("extra,why", [
    ({"calculation_type": "OPT"}, "OPTGEOM"),
    ({"calculation_type": "FREQ"}, "FREQCALC"),
    ({"scf_settings": {"method": "ANDERSON"}}, "ANDERSON"),
    ({"scf_settings": {"method": "BROYDEN"}}, "BROYDEN"),
])
def test_opt2d12_writes_no_soc_deck_crystal_cannot_run(opt2d12, capsys, extra, why):
    status, decks = opt2d12(["pbte"], soc=True, **extra)
    captured = capsys.readouterr()
    assert status not in (0, None)
    assert not any(p.exists() for p in decks.values())
    assert why in captured.out + captured.err
