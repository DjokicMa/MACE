"""The official pob-TZVP-rev2-SOC basis sets for SOC decks ("soc_basis").

X_pob_TZVP_rev2_SOC on crystal.unito.it is CRYSTAL's STUTSC spin-orbit ECP
with the valence basis published for it ("To be used with CRYSTAL23"; Laun &
Bredow, J. Comput. Chem. 43 (2022) 839; Desmarais, Flament & Erba, Phys. Rev. B
101 (2020) 235142). "soc_basis": "pob-tzvp-rev2-soc" replaces every ECP atom's
block whole by the set's block for its element; all-electron atoms keep theirs.
"""
import hashlib
import importlib.util
import json
import sys

import pytest

from conftest import REPO_ROOT

import CRYSTALOptToD12 as M
import soc_ecp as S
from basis_file_check import basis_block_problem
from test_soc_decks import (I_AE, PBTE, SOC_PARENT, _basis, _deck, _library_block,  # noqa: F401
                            cif2d12)
from test_soc_stutsc import opt2d12_parent  # noqa: F401

FAMILY = "pob-tzvp-rev2-soc"
QUIET = dict(log=lambda m: None)
COVERED = [37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56,
           *range(72, 84)]


def test_the_set_covers_rb_to_i_but_tc_cs_ba_and_hf_to_bi():
    sets = S.load_soc_basis(FAMILY)
    assert sorted(sets) == COVERED
    for z, lines in sets.items():
        assert lines[0].split()[0] == str(200 + z) and lines[1] == "STUTSC"
        assert basis_block_problem("\n".join(lines)) is None


def test_the_bi_block_is_pinned_and_is_maces_bi_valence_basis():
    bi = S.load_soc_basis(FAMILY)[83]
    assert hashlib.sha256("\n".join(bi).encode()).hexdigest() == \
        "edb85b9126f944fc376ad5288027a81f4c39040ef77ea6b73d7e326d22f3a1c9"
    # Hf-Bi: the set's valence basis is pob-TZVP-rev2's own (as in
    # full.basis.triplezeta), with STUTSC in place of the INPUT ECP
    tz = _library_block(283)
    end = 3 + sum(int(c) for c in tz[2].split()[1:])
    assert bi[0] == tz[0] and bi[2:] == tz[end:]


def test_the_library_records_where_each_set_came_from():
    text = (REPO_ROOT / "Crystal_d12" / "basis_sets" / f"{FAMILY}.lib").read_text()
    assert text.count("#%record ") == len(COVERED)
    assert text.count("#%  doi 10.1002/jcc.26839") == len(COVERED)
    assert "#%  url https://www.crystal.unito.it/Basis_Sets/bismuth.html#Bi_pob_TZVP_rev2_SOC" in text
    assert "#%missing 243 " in text


def test_a_record_that_breaks_the_grammar_is_refused(tmp_path, monkeypatch):
    bad = "282 3\nSTUTSC\n0 0 1 2.0 1.0\n1.0 1.0\n"
    (tmp_path / f"{FAMILY}.lib").write_text(
        f"#%MACE-BASIS-LIBRARY 1\n#%family {FAMILY}\n#%record 282 bytes={len(bad)}\n{bad}\n")
    monkeypatch.setattr(S, "SOC_BASIS_DIR", str(tmp_path))
    monkeypatch.setattr(S, "_SOC_BASIS_CACHE", {})
    with pytest.raises(S.SocError, match="record 282 .*announces 3 shells but holds 1"):
        S.load_soc_basis(FAMILY)


# --- the fetch tool's parser, on a synthetic page ---------------------------------

def _fetcher():
    path = REPO_ROOT / "Crystal_d12" / "basis_sets" / "fetch_soc_basis.py"
    spec = importlib.util.spec_from_file_location("fetch_soc_basis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAGE = """<html><body>
<p><a name="Xx_pob_TZVP_rev2"></a>Xx_pob_TZVP_rev2</p>
<pre>283 1
INPUT
1. 0 0 0 0 0 0
0 0 1 2.0 1.0
  1.0  1.0
</pre>
<p><a name="Bi_pob_TZVP_rev2_SOC"></a><font size="+2">Bi_pob_TZVP_rev2_SOC</font></p>
<pre>283 2
STUTSC
0 0 2 2 1.0
  23.2398550290      -0.068255758685
  6.64742550000       0.978880464710
0 2 1 0 1.0
  0.42288904000       1.000000000000

J. Laun, T. Bredow J. Comput. Chem.  (2022), 43:839-846
"BSSE-corrected ...", DOI: 10.1002/jcc.26839

Note: To be used with CRYSTAL23
</pre>
</body></html>"""


def test_the_fetch_parser_cuts_the_record_from_the_reference():
    f = _fetcher()
    anchor, record, trailing = f.get_set(PAGE, 83)
    assert anchor == "Bi_pob_TZVP_rev2_SOC"
    assert record == ["283 2", "STUTSC", "0 0 2 2 1.0",
                      "  23.2398550290      -0.068255758685",
                      "  6.64742550000       0.978880464710",
                      "0 2 1 0 1.0", "  0.42288904000       1.000000000000"]
    assert trailing.startswith("J. Laun") and "To be used with CRYSTAL23" in trailing
    text = f.library_text({83: ("https://x/bismuth.html", anchor, record, trailing)},
                          {43: "Tc: none"}, "2026-10-01")
    assert "#%  doi 10.1002/jcc.26839" in text and "#%  note To be used with CRYSTAL23" in text
    assert "#%missing 243 Tc: none" in text
    with pytest.raises(LookupError):
        f.get_set(PAGE.replace("Bi_pob", "Bx_pob"), 83)


def test_the_fetch_parser_refuses_a_block_that_is_not_stutsc():
    f = _fetcher()
    with pytest.raises(ValueError, match="STUTSC"):
        f.get_set(PAGE.replace("STUTSC", "INPUT"), 83)


# --- soc_deck -------------------------------------------------------------------

def test_both_ecp_blocks_of_pbte_are_replaced_whole():
    notes = []
    soc = S.soc_deck(PBTE, log=notes.append, basis=FAMILY)
    sets = S.load_soc_basis(FAMILY)
    assert _basis(soc) == sets[82] + sets[52]
    assert "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in soc and "INPSOC" not in soc
    # the geometry keeps its atom numbers
    assert "282 0.0 0.0 0.0\n252 0.5 0.5 0.5\n" in soc
    assert any("Pb, Te get the official pob-tzvp-rev2-soc basis" in n for n in notes)
    assert any("manual p. 174" in n and "delete the SOC line" in n for n in notes)
    # the block after the basis is the same as with MACE's INPSOC
    default = S.soc_deck(PBTE, **QUIET)
    assert soc.split("99 0\n")[1] == default.split("99 0\n")[1]


def test_all_electron_atoms_keep_their_block_with_a_note():
    notes = []
    soc = S.soc_deck(_deck(_library_block(282) + I_AE), log=notes.append, basis=FAMILY)
    assert "\n".join(I_AE) + "\n99 0\n" in soc
    assert any("atom 53 is all-electron" in n and "p. 175" in n for n in notes)


def test_tc_stays_all_electron_with_the_note():
    notes = []
    tc = ["43 1", "0 0 1 2.0 1.0", "1.0 1.0"]
    soc = S.soc_deck(_deck(_library_block(282) + tc), log=notes.append, basis=FAMILY)
    assert "\n".join(tc) + "\n99 0\n" in soc
    assert any("MACE's Tc is all-electron and stays so" in n for n in notes)


@pytest.mark.parametrize("deck,kw,why", [
    (_deck(_library_block(257)), {}, r"has no basis set for La \(Z=57\)"),
    (PBTE, {"ecp_form": "inpsoc"}, 'cannot be combined with "soc_ecp": "inpsoc"'),
    (PBTE, {"soscale": 0.5}, "STUTSC has no scale factor"),
    (PBTE, {"basis": "pob-tzvp-rev2"}, "soc_basis must be one of pob-tzvp-rev2-soc"),
    (PBTE.replace("END\n282 ", "BASISSET\nPOB-TZVP-REV2\n282 ", 1), {}, "internal basis-set library"),
], ids=["La", "inpsoc", "soscale", "unknown", "BASISSET"])
def test_what_soc_basis_cannot_do_is_refused(deck, kw, why):
    kw = dict(kw)
    kw.setdefault("basis", FAMILY)
    with pytest.raises(S.SocError, match=why):
        S.soc_deck(deck, **QUIET, **kw)


def test_with_stutsc_the_deck_is_the_same():
    assert S.soc_deck(PBTE, basis=FAMILY, ecp_form="stutsc", **QUIET) == \
        S.soc_deck(PBTE, basis=FAMILY, **QUIET)


def test_a_child_of_a_soc_basis_deck_keeps_the_official_blocks():
    parent = S.soc_deck(PBTE, basis=FAMILY, **QUIET)
    notes = []
    again = S.soc_deck(parent.replace("TWOCOMPON\nSOC\nEND\n", ""), log=notes.append)
    assert _basis(again) == _basis(parent)
    assert sum("official pob-TZVP-rev2-SOC block" in n for n in notes) == 2


def test_without_the_key_nothing_changes():
    assert S.soc_deck(PBTE, basis=None, **QUIET) == S.soc_deck(PBTE, **QUIET)


# --- through opt2d12 and cif2d12 ------------------------------------------------

def test_an_opt2d12_template_soc_basis_reaches_the_deck(opt2d12_parent):
    status, decks = opt2d12_parent(SOC_PARENT, soc_basis=FAMILY)
    assert status == 0
    sets = S.load_soc_basis(FAMILY)
    assert _basis(decks[0].read_text()) == sets[82] + sets[52]


def test_cif2d12_soc_basis(cif2d12):
    ok, out = cif2d12("external", soc=True, soc_basis=FAMILY)
    assert ok is True
    assert _basis(out.read_text()) == S.load_soc_basis(FAMILY)[79]
