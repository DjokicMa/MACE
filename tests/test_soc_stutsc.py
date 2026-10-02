"""CRYSTAL's STUTSC spin-orbit ECP in place of MACE's INPSOC ("soc_ecp").

STUTSC is CRYSTAL's internal Stuttgart spin-orbit ECP library, written as one
keyword line where INPSOC takes its records (manual p. 84). Measured on HPCC
(FACTS): for Te, Au, Pb and Bi CRYSTAL prints the same pseudopotential for
STUTSC as for MACE's INPSOC conversion, with the same core, and the energies
agree. "soc_ecp": "stutsc" writes it for those elements; the default stays
INPSOC.
"""
import json
import sys

import pytest

import CRYSTALOptToD12 as M
import soc_ecp as S
from test_soc_decks import (PB, PBTE, SOC_PARENT, TE, _basis, _deck, _NoTTY,  # noqa: F401
                            _Parser, cif2d12)

QUIET = dict(log=lambda m: None)


def _swap_inpsoc_for_stutsc(deck):
    lines = deck.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i] == "INPSOC":
            n = sum(int(c) for c in lines[i + 2].split()[1:])
            out.append("STUTSC")
            i += 3 + n
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def test_stutsc_is_the_inpsoc_deck_with_the_records_swapped():
    inpsoc = S.soc_deck(PBTE, **QUIET)
    notes = []
    stutsc = S.soc_deck(PBTE, log=notes.append, ecp_form="stutsc")
    assert stutsc == _swap_inpsoc_for_stutsc(inpsoc)
    assert stutsc.count("STUTSC") == 2 and "INPSOC" not in stutsc
    assert sum("gets CRYSTAL's STUTSC" in n for n in notes) == 2


def test_the_default_and_inpsoc_write_the_same_deck():
    assert S.soc_deck(PBTE, **QUIET) == S.soc_deck(PBTE, ecp_form="inpsoc", **QUIET)
    assert S.soc_deck(PBTE, ecp_form="STUTSC", **QUIET) == \
        S.soc_deck(PBTE, ecp_form="stutsc", **QUIET)


@pytest.mark.parametrize("bad", ["colusc", "", 1, True])
def test_an_unknown_soc_ecp_is_refused(bad):
    with pytest.raises(S.SocError, match='soc_ecp must be "inpsoc"'):
        S.soc_deck(PBTE, ecp_form=bad, **QUIET)


def test_an_element_whose_stutsc_was_not_compared_is_refused(monkeypatch):
    monkeypatch.setattr(S, "STUTSC_CORES", {82: 60})
    with pytest.raises(S.SocError, match=r"Te \(Z=52\): CRYSTAL's STUTSC spin-orbit ECP was "
                                         r"compared with MACE's INPSOC conversion only for Pb"
                                         r".*the default INPSOC"):
        S.soc_deck(PBTE, ecp_form="stutsc", **QUIET)


def test_a_core_that_is_not_the_measured_one_is_refused(monkeypatch):
    monkeypatch.setattr(S, "STUTSC_CORES", {52: 28, 82: 78})
    with pytest.raises(S.SocError, match=r"Pb \(Z=82\).*only for Te, Pb"):
        S.soc_deck(PBTE, ecp_form="stutsc", **QUIET)


def test_stutsc_with_a_soscale_is_refused():
    with pytest.raises(S.SocError, match="STUTSC has no scale factor.*SOSCALE 0.5"):
        S.soc_deck(PBTE, soscale=0.5, ecp_form="stutsc", **QUIET)


def test_the_pairing_check_still_comes_first():
    """An ECP that is not the spin-orbit ECP's scalar part is refused as
    before, STUTSC or not."""
    from conftest import REPO_ROOT
    au = [ln.rstrip("\n") for ln in open(REPO_ROOT / "Crystal_d12" / "basis_sets"
                                         / "stuttgart" / "279") if ln.strip()]
    with pytest.raises(S.SocError, match="not the scalar part"):
        S.soc_deck(_deck(au), ecp_form="stutsc", **QUIET)


# --- a deck derived from a SOC deck -------------------------------------------

STUTSC_PARENT = S.soc_deck(PBTE, ecp_form="stutsc", **QUIET)


def _again(deck, **kw):
    notes = []
    return S.soc_deck(deck.replace("TWOCOMPON\nSOC\nEND\n", ""), log=notes.append, **kw), notes


def test_a_stutsc_block_mace_would_write_is_kept():
    again, notes = _again(STUTSC_PARENT)
    assert _basis(again) == _basis(STUTSC_PARENT)
    assert sum("keeps CRYSTAL's STUTSC" in n for n in notes) == 2
    # with "stutsc" too
    assert _again(STUTSC_PARENT, ecp_form="stutsc")[0] == again


def test_a_kept_inpsoc_becomes_stutsc_with_stutsc():
    again, notes = _again(SOC_PARENT, ecp_form="stutsc")
    assert _basis(again) == _basis(STUTSC_PARENT)
    assert sum("INPSOC records become CRYSTAL's STUTSC" in n for n in notes) == 2


def test_a_stutsc_block_on_a_foreign_valence_basis_is_refused():
    """STUTSC on a valence basis that is not the one published with that
    potential (here Pb's first primitive changed) is not taken over."""
    deck = STUTSC_PARENT.replace("TWOCOMPON\nSOC\nEND\n", "")
    assert deck.count(PB[18]) == 1
    deck = deck.replace(PB[18], PB[18].replace(PB[18].split()[0], "999.0", 1))
    with pytest.raises(S.SocError, match=r"Pb \(Z=82\): the deck's STUTSC ECP sits on a "
                                         r"valence basis that is neither"):
        S.soc_deck(deck, **QUIET)


def test_a_kept_stutsc_with_a_soscale_is_refused():
    with pytest.raises(S.SocError, match="STUTSC has no scale factor"):
        _again(STUTSC_PARENT, soscale=2.0)


def test_stutsc_for_an_element_outside_the_table_is_refused_when_kept(monkeypatch):
    monkeypatch.setattr(S, "STUTSC_CORES", {82: 60})
    with pytest.raises(S.SocError, match=r"Te \(Z=52\): the deck's STUTSC ECP sits on"):
        _again(STUTSC_PARENT)


def test_explicit_inpsoc_on_a_kept_stutsc_is_refused():
    with pytest.raises(S.SocError, match='"soc_ecp": "inpsoc" would change the potential'):
        _again(STUTSC_PARENT, ecp_form="inpsoc")


def test_other_ecp_keywords_are_refused_naming_what_is_accepted():
    deck = _deck(["282 1", "STUTLC", "0 0 1 2.0 1.0", "1.0 1.0"])
    with pytest.raises(S.SocError) as exc:
        S.soc_deck(deck, **QUIET)
    msg = str(exc.value)
    assert "not STUTLC" in msg and "INPUT" in msg and "INPSOC" in msg and "STUTSC" in msg


# --- through opt2d12 and cif2d12 ------------------------------------------------

@pytest.fixture
def opt2d12_parent(tmp_path, monkeypatch):
    """opt2d12 on a real parent .d12 ``text`` (the .out is stubbed)."""
    monkeypatch.setattr(M, "CrystalOutputParser",
                        lambda p: _Parser(str(p).replace("socpbte", "pbte"), "out"))
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.chdir(tmp_path)

    def run(text, **template):
        (tmp_path / "socpbte.out").write_text("")
        (tmp_path / "socpbte.d12").write_text(text)
        base = {"calculation_type": "SP", "basis_set": M.PARENT_BASIS_MARKER,
                "basis_set_type": "EXTERNAL", "use_original_external_basis": True,
                "has_original_external_basis": True}
        base.update(template)
        (tmp_path / "t.json").write_text(json.dumps(base))
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", "--out-file", "socpbte.out",
                                          "--config-file", "t.json", "--output-dir", "sp",
                                          "--yes"])
        try:
            M.main()
            status = 0
        except SystemExit as e:
            status = e.code
        decks = list((tmp_path / "sp").glob("*.d12")) if (tmp_path / "sp").exists() else []
        return status, decks
    return run


def test_an_opt2d12_child_of_a_stutsc_parent_keeps_stutsc(opt2d12_parent, capsys):
    status, decks = opt2d12_parent(STUTSC_PARENT)
    assert status == 0 and len(decks) == 1
    text = decks[0].read_text()
    # the parent's basis records come back as they were (stripped, as the
    # parser keeps every external basis)
    assert _basis(text) == [ln.strip() for ln in _basis(STUTSC_PARENT)]
    assert "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in text
    assert "keeps CRYSTAL's STUTSC" in capsys.readouterr().out


def test_an_opt2d12_template_soc_ecp_writes_stutsc(opt2d12_parent):
    status, decks = opt2d12_parent(SOC_PARENT, soc_ecp="stutsc")
    assert status == 0
    assert _basis(decks[0].read_text()) == [ln.strip() for ln in _basis(STUTSC_PARENT)]


def test_a_scalar_child_of_a_stutsc_parent_warns_about_its_stutsc(opt2d12_parent, capsys):
    status, decks = opt2d12_parent(STUTSC_PARENT, soc=False)
    assert status == 0
    err = capsys.readouterr().err
    assert "STUTSC" in err and "untested" in err


def test_cif2d12_soc_ecp_stutsc(cif2d12):
    ok, out = cif2d12("external", soc=True, soc_ecp="stutsc")
    text = out.read_text()
    assert ok is True and "279 " in text and "\nSTUTSC\n" in text and "INPSOC" not in text


# --- the measured table -------------------------------------------------------

def test_every_core_in_the_table_is_the_spin_orbit_ecps_own():
    assert {z: S.load_so_ecp(z).ncore for z in S.STUTSC_CORES} == S.STUTSC_CORES


@pytest.mark.parametrize("z", [72, 57, 64])
def test_hf_and_the_lanthanides_are_not_given_stutsc(z):
    """H1 (HPCC): Hf's STUTSC has another local-term exponent; for La-Lu
    CRYSTAL stops after the PSEUDOPOTENTIAL INFORMATION header."""
    from test_soc_decks import _library_block
    deck = _deck(_library_block(200 + z))
    assert "INPSOC" in S.soc_deck(deck, **QUIET)
    with pytest.raises(S.SocError, match="the default INPSOC"):
        S.soc_deck(deck, ecp_form="stutsc", **QUIET)
