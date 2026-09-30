"""Two-component SOC decks, written only on request ("soc": true).

soc_ecp.soc_deck turns a finished single-point deck into a 2c-SCF SOC deck:
each INPUT ECP that is the scalar part of a spin-orbit ECP becomes INPSOC
(manual pp. 84-87), and a TWOCOMPON block holding SOC (manual sec. 6.2,
pp. 169-175) goes into the SCF input. Anything chapter 6 does not support is
refused and no deck is written. Without the option every deck is byte for
byte what it was: the hashes below were taken from decks written by main
before the option existed.
"""
import copy
import hashlib
import json
import os
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import CRYSTALOptToD12 as M  # noqa: E402
import soc_ecp as S  # noqa: E402

TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"


def _library_block(z):
    return [ln.rstrip("\n") for ln in open(TZ / str(z)) if ln.strip()]


PB = _library_block(282)
TE = _library_block(252)
I_AE = ["53 1", "0 0 1 2.0 1.0", "1.0 1.0"]      # an all-electron atom


def _deck(basis, block3=None):
    block3 = block3 or ["DFT", "PBE0", "ENDDFT", "TOLINTEG", "7 7 7 7 14", "TOLDEE", "7",
                        "SHRINK", "8 16", "SCFDIR", "MAXCYCLE", "800", "FMIXING", "30",
                        "DIIS", "HISTDIIS", "100", "PPAN", "END"]
    return "\n".join(["PbTe", "CRYSTAL", "0 0 0", "225", "6.462", "2",
                      "282 0.0 0.0 0.0", "252 0.5 0.5 0.5", "END",
                      *basis, "99 0", "END", *block3]) + "\n"


PBTE = _deck(PB + TE)


def test_ecps_become_inpsoc_and_twocompon_goes_before_scfdir():
    soc = S.soc_deck(PBTE).split("\n")
    plain = PBTE.split("\n")
    # Pb: header kept, INPUT -> INPSOC + INTERNAL, valence shells unchanged
    i = soc.index("282 " + PB[0].split()[1])
    assert soc[i + 1:i + 4] == ["INPSOC", "INTERNAL 1.0", "22. 0 2 4 4 2 2"]
    assert soc[i + 3:i + 4 + 14] == S.inpsoc_records(S.load_so_ecp(82), 82)[2:]
    pb_shells = PB[3 + 14:]
    assert soc[i + 4 + 14:i + 4 + 14 + len(pb_shells)] == pb_shells
    # Te too: its library ECP is the scalar part of sopseud/252.mol
    assert "252 " + TE[0].split()[1] in soc and soc.count("INPSOC") == 2
    # SCF input: the only other change is the TWOCOMPON block before SCFDIR
    j = soc.index("SCFDIR")
    assert soc[j - 3:j] == ["TWOCOMPON", "SOC", "END"]
    assert soc[soc.index("END", soc.index("99 0")) + 1:j - 3] == \
        ["8 8" if ln == "8 16" else ln   # the Gilat net follows the Monkhorst net
         for ln in plain[plain.index("END", plain.index("99 0")) + 1:plain.index("SCFDIR")]]
    assert soc[j:] == ["50" if ln == "30" else ln   # FMIXING 30 -> 50
                       for ln in plain[plain.index("SCFDIR"):]
                       if ln not in ("DIIS", "HISTDIIS", "100")]
    # geometry untouched
    assert soc[:soc.index("END")] == plain[:plain.index("END")]


def test_diis_and_histdiis_are_left_out():
    """A DIIS record in a 2c deck stops the stock build with "ERROR **** DIIS
    **** DIIS NOT COMPATIBLE WITH 2-COMP SCF" (HPCC), although the manual says
    TWOCOMPON deactivates DIIS (p. 170)."""
    soc = S.soc_deck(PBTE).split("\n")
    assert "DIIS" not in soc and "HISTDIIS" not in soc
    assert soc[soc.index("FMIXING") + 2:] == ["PPAN", "END", ""]


@pytest.mark.parametrize("before,after", [
    (["SHRINK", "8 16"], ["SHRINK", "8 8"]),
    (["SHRINK", "12 12"], ["SHRINK", "12 12"]),
    (["SHRINK", "0 20", "10 10 1"], ["SHRINK", "0 10", "10 10 1"]),   # SLAB
    (["SHRINK", "0 24", "12 8 6"], ["SHRINK", "0 12", "12 8 6"]),     # P1
])
def test_the_gilat_net_is_the_monkhorst_net(before, after):
    """fcc Au in 2c with SHRINK 12 24 never converged (charge normalization
    factor 1.35-1.69); 12 12 converged (HPCC)."""
    deck = PBTE.replace("SHRINK\n8 16\n", "\n".join(before) + "\n")
    notes = []
    soc = S.soc_deck(deck, log=notes.append).split("\n")
    i = soc.index("SHRINK")
    assert soc[i:i + len(after)] == after
    assert any("SHRINK" in n for n in notes) == (before != after)


def _scf(deck):
    lines = deck.split("\n")
    return {k: lines[lines.index(k) + 1] for k in ("FMIXING", "MAXCYCLE") if k in lines}


@pytest.mark.parametrize("fmixing,maxcycle,expected,warned", [
    ("30", "800", {"FMIXING": "50", "MAXCYCLE": "800"}, False),   # MACE's defaults
    ("50", "800", {"FMIXING": "50", "MAXCYCLE": "800"}, False),
    ("85", "800", {"FMIXING": "50", "MAXCYCLE": "800"}, False),   # replaced, noted
    ("60", "800", {"FMIXING": "50", "MAXCYCLE": "800"}, False),   # replaced, noted
    ("30", "100", {"FMIXING": "50", "MAXCYCLE": "100"}, True),    # set: kept, warned
    ("30", "300", {"FMIXING": "50", "MAXCYCLE": "300"}, False),
])
def test_fmixing_50_and_enough_cycles(fmixing, maxcycle, expected, warned):
    """FMIXING scan on HPCC (TOLDEE 7): 50 converged fastest for the Bi2
    bilayer (12 cycles), PbTe (9) and fcc Au at 12 12 (25), with the energies
    85 gave; 30 aborted Bi2 in cycle 1. So every SOC deck gets FMIXING 50,
    and a different FMIXING in the deck is named when it is replaced."""
    deck = PBTE.replace("MAXCYCLE\n800\nFMIXING\n30\n",
                        f"MAXCYCLE\n{maxcycle}\nFMIXING\n{fmixing}\n")
    notes = []
    assert _scf(S.soc_deck(deck, log=notes.append)) == expected
    assert any(n.startswith("Warning") for n in notes) == warned
    replaced = [n for n in notes if "FMIXING" in n]
    if fmixing == "50":
        assert replaced == []
    else:
        assert replaced == [f"SOC deck: FMIXING {fmixing} replaced by 50 (the SOC default; "
                            f"\"soc_fmixing\" in the options file or template sets another)"]


@pytest.mark.parametrize("deck_fmixing,asked,written,noted", [
    ("30", 70, "70", True),     # MACE's default replaced by the asked value
    ("60", 70, "70", True),     # a parent/template value too: it was asked for
    ("70", 70, "70", False),    # already there: nothing to say
    (None, 85, "85", True),     # no FMIXING in the deck: added
    ("30", 30, "30", False),    # 30 asked for on purpose: kept
])
def test_soc_fmixing_sets_the_fmixing(deck_fmixing, asked, written, noted):
    deck = PBTE.replace("FMIXING\n30\n", f"FMIXING\n{deck_fmixing}\n" if deck_fmixing else "")
    notes = []
    soc = S.soc_deck(deck, log=notes.append, fmixing=asked)
    assert _scf(soc)["FMIXING"] == written and soc.count("FMIXING") == 1
    assert any("soc_fmixing" in n for n in notes) == noted
    assert not any(n.startswith("Warning") and "FMIXING" in n for n in notes)


def test_the_default_soc_fmixing_is_the_module_constant(monkeypatch):
    monkeypatch.setattr(S, "SOC_FMIXING", 77)
    assert _scf(S.soc_deck(PBTE, log=lambda m: None))["FMIXING"] == "77"


@pytest.mark.parametrize("bad", [-1, 101, "high", 8.5, True])
def test_a_soc_fmixing_that_is_not_a_percentage_is_refused(bad):
    with pytest.raises(S.SocError, match="soc_fmixing"):
        S.soc_deck(PBTE, log=lambda m: None, fmixing=bad)


def test_missing_fmixing_and_maxcycle_are_added():
    deck = PBTE.replace("MAXCYCLE\n800\nFMIXING\n30\n", "")
    soc = S.soc_deck(deck, log=lambda m: None)
    assert _scf(soc) == {"FMIXING": "50", "MAXCYCLE": "200"}
    assert soc.endswith("PPAN\nFMIXING\n50\nMAXCYCLE\n200\nEND\n")


def test_all_electron_atoms_keep_their_basis():
    soc = S.soc_deck(_deck(PB + I_AE))
    assert "\n".join(I_AE) + "\n99 0\n" in soc


def test_the_soc_deck_reads_back_record_for_record():
    """Every ECP number in the SOC deck is the sopseud file's own."""
    soc = S.soc_deck(PBTE).split("\n")
    i = soc.index("INPSOC")
    znuc, local, per_l = S.parse_inpsoc(soc[i:i + 3 + 14])
    ecp = S.load_so_ecp(82)
    assert znuc == 22.0 and local == []
    assert [(a, c, n) for a, _b, c, n in per_l[1] if c] == \
        [t.values() for t in ecp.spin_orbit[1]]


@pytest.mark.parametrize("change,why", [
    (lambda d: d.replace("PPAN\n", "OPTGEOM\nENDOPT\nPPAN\n"), "geometry optimization"),
    (lambda d: d.replace("PPAN\n", "FREQCALC\nEND\nPPAN\n"), "frequency calculation"),
    (lambda d: d.replace("PBE0\n", "HSE06\n"), "HSE06"),
    (lambda d: d.replace("PBE0\n", "HSE06-D3\n"), "HSE06-D3"),
    (lambda d: d.replace("PBE0\n", "M06-D3\n"), "M06-D3"),
    (lambda d: d.replace("PPAN\n", "DFTD3\nEND\nPPAN\n"), "DFTD3"),
    (lambda d: d.replace("DIIS\n", "BROYDEN\n0.0001 50 2\n"), "BROYDEN"),
    (lambda d: d.replace("SCFDIR\n", "GUESSP\nSCFDIR\n"), "GUESSP"),
    (lambda d: d.replace("SCFDIR\n", "TWOCOMPON\nEND\nSCFDIR\n"), "already has a TWOCOMPON"),
])
def test_what_chapter_6_does_not_support_is_refused(change, why):
    with pytest.raises(S.SocError, match=why):
        S.soc_deck(change(PBTE))


@pytest.mark.parametrize("deck,why", [
    (_deck(I_AE), "no atom in the deck carries an ECP"),
    (_deck(["282 1", "HAYWSC", "0 0 1 2.0 1.0", "1.0 1.0"]), "only an ECP entered with INPUT"),
    (_deck([ln.rstrip("\n") for ln in open(REPO_ROOT / "Crystal_d12" / "basis_sets" / "stuttgart" / "279")
            if ln.strip()]), "not the scalar part"),
    (_deck(PB[:5]), "could not read the basis-set input"),
    (PBTE.replace("END\n282 ", "BASISSET\nPOB-TZVP-REV2\n282 ", 1), "internal basis-set library"),
])
def test_basis_input_soc_cannot_use_is_refused(deck, why):
    with pytest.raises(S.SocError, match=why):
        S.soc_deck(deck)


@pytest.mark.parametrize("dft", ["SPIN\nPBE0", "PBE0-D3", "B3LYP-D3", "SPIN\nB3LYP-D3",
                                 "PBE-D3", "BLYP-D3", "PW1PW-D3"])
def test_spin_and_d3_are_accepted(dft):
    """On HPCC a 2c run accepts SPIN (without effect), and PBE0-D3 and
    B3LYP-D3 print "DFT-D3(BJ) WITH AUTOMATIC PARAMETER SETUP" and a D3
    energy. The other -D3 keywords are the manual's (p. 150) whose
    functional chapter 6 allows."""
    soc = S.soc_deck(PBTE.replace("PBE0\n", dft + "\n"), log=lambda m: None)
    assert f"DFT\n{dft}\nENDDFT\n" in soc and "TWOCOMPON\nSOC\nEND\n" in soc


def test_exchange_and_correlation_keywords_from_the_list_are_accepted():
    deck = PBTE.replace("PBE0\n", "EXCHANGE\nPBE\nCORRELAT\nPBE\nHYBRID\n25\nXLGRID\n")
    assert "TWOCOMPON" in S.soc_deck(deck)
    with pytest.raises(S.SocError, match="EXCHANGE SCAN"):
        S.soc_deck(deck.replace("EXCHANGE\nPBE\n", "EXCHANGE\nSCAN\n"))


# --- through opt2d12, as a user runs it ----------------------------------------

def _out(atoms):
    return {
        "dimensionality": "CRYSTAL", "spacegroup": 225, "origin_setting": "0 0 0",
        "conventional_cell": [6.462, 6.462, 6.462, 90.0, 90.0, 90.0],
        "coordinates": [{"atom_number": str(z), "x": f"{i / 2:.6f}", "y": f"{i / 2:.6f}",
                         "z": f"{i / 2:.6f}", "is_unique": True} for i, z in enumerate(atoms)],
        "functional": "PBE0", "calculation_type": "SP",
    }


def _parent_deck(records):
    return {"functional": "PBE0", "dispersion": False, "dft_grid": "XLGRID",
            "calculation_type": "SP", "k_points": "8 16",
            "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
            "basis_set_type": "EXTERNAL", "external_basis_data": records}


PARENTS = {
    "pbte": (_out([282, 252]), _parent_deck(PB + TE)),
    "pbi": (_out([282, 53]), _parent_deck(PB + I_AE)),
    "nacl": (_out([11, 17]), _parent_deck(["11 1", "0 0 1 2.0 1.0", "1.0 1.0",
                                           "17 1", "0 0 1 2.0 1.0", "1.0 1.0"])),
}


class _Parser:
    def __init__(self, path, kind):
        self.stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self.kind = kind

    def parse(self):
        out, deck = PARENTS[self.stem]
        return copy.deepcopy(out if self.kind == "out" else deck)


class _NoTTY:
    def isatty(self):
        return False


@pytest.fixture
def opt2d12(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CrystalOutputParser", lambda p: _Parser(str(p), "out"))
    monkeypatch.setattr(M, "CrystalInputParser", lambda p: _Parser(str(p), "d12"))
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.chdir(tmp_path)
    for stem in PARENTS:
        (tmp_path / f"{stem}.out").write_text("")
        (tmp_path / f"{stem}.d12").write_text("")

    def run(stems, _yes=True, **template):
        base = {"calculation_type": "SP", "functional": "PBE0", "dispersion": False,
                "dft_grid": "XLGRID", "basis_set": M.PARENT_BASIS_MARKER,
                "basis_set_type": "EXTERNAL", "use_original_external_basis": True,
                "has_original_external_basis": True}
        base.update(template)
        (tmp_path / "t.json").write_text(json.dumps(base))
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", "--out-file",
                                          *[f"{s}.out" for s in stems], "--config-file",
                                          "t.json", "--output-dir", "sp"]
                            + (["--yes"] if _yes else []))
        try:
            M.main()
            status = 0
        except SystemExit as e:
            status = e.code
        return status, {s: tmp_path / "sp" / f"{s}_sp_PBE0_optimized.d12" for s in stems}
    return run


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Written by main (9fcdb9c2) from the same parents and template.
MAIN_SHA = {
    "pbte": "a816e774a163a29f6c5117fdf45d97cdd3bf79f975cd2db70eb26c325852f85a",
    "pbi": "322b662e8cacb0b7bbf850ddbe33bafb173e1643455389840b01c366e86ef0d4",
    "nacl": "19a47a987caa03630ac0bb75f1db6cc5966787277fe0acb5913fcf252d637091",
}


@pytest.mark.parametrize("template", [{}, {"soc": False}], ids=["no key", "soc false"])
def test_without_soc_opt2d12_writes_what_main_wrote(opt2d12, template):
    status, decks = opt2d12(["pbte", "pbi", "nacl"], **template)
    assert status == 0
    assert {s: _sha(p) for s, p in decks.items()} == MAIN_SHA


def test_soc_true_writes_the_2c_deck(opt2d12):
    status, decks = opt2d12(["pbte"], soc=True)
    assert status == 0
    text = decks["pbte"].read_text()
    assert text.count("INPSOC") == 2 and "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in text


def test_soc_refusal_writes_no_deck_and_fails_the_file(opt2d12, capsys):
    status, decks = opt2d12(["nacl", "pbte"], soc=True)
    err = capsys.readouterr().err
    assert status == 1
    assert "no atom in the deck carries an ECP" in err
    assert not decks["nacl"].exists()
    assert decks["pbte"].exists()
    assert not any(n.startswith(".") for n in os.listdir(decks["pbte"].parent))


def test_opt2d12_template_soc_fmixing_reaches_the_deck(opt2d12):
    status, decks = opt2d12(["pbte"], soc=True, soc_fmixing=60)
    assert status == 0
    assert "FMIXING\n60\n" in decks["pbte"].read_text()


def test_soc_with_dispersion_writes_the_d3_functional(opt2d12):
    status, decks = opt2d12(["pbte"], soc=True, dispersion=True)
    assert status == 0
    deck = decks["pbte"].with_name("pbte_sp_PBE0-D3_optimized.d12").read_text()
    assert "DFT\nPBE0-D3\n" in deck and "TWOCOMPON\nSOC\nEND\n" in deck


# --- a SOC parent deck, read by the real parser ----------------------------------

# The parent: the SOC deck soc_deck writes for PbTe (INPSOC records, TWOCOMPON).
SOC_PARENT = S.soc_deck(PBTE, log=lambda m: None)


def _basis(deck):
    lines = deck.split("\n")
    end = lines.index("END")
    return lines[end + 1:lines.index("99 0")]


def test_the_parser_reads_a_soc_deck_and_keeps_its_inpsoc_basis(tmp_path):
    from d12_parsers import CrystalInputParser
    path = tmp_path / "pbte.d12"
    path.write_text(SOC_PARENT)
    data = CrystalInputParser(str(path)).parse()
    assert data["soc"] is True and data["two_component"] is True
    assert data["basis_set_type"] == "EXTERNAL"
    assert data["external_basis_data"] == [ln.strip() for ln in _basis(SOC_PARENT) if ln.strip()]
    assert data["external_basis_data"].count("INPSOC") == 2
    assert data["k_points"] == "8 8" and data["scf_settings"]["fmixing"] == 50
    # a scalar deck gets neither key
    path.write_text(PBTE)
    data = CrystalInputParser(str(path)).parse()
    assert "soc" not in data and "two_component" not in data


def test_the_output_parser_marks_a_two_component_run(tmp_path, monkeypatch):
    from d12_parsers import CrystalOutputParser
    monkeypatch.setattr(CrystalOutputParser, "_extract_geometry", lambda self, c: None)
    out = tmp_path / "x.out"
    # lines of a real fcc Au 2c-SCF (HPCC)
    out.write_text(" CHARGE NORMALIZATION FACTOR   1.00000000\n"
                   " TOTAL X-COMP MAGNETIZATION    0.00000001\n"
                   " - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70\n")
    assert CrystalOutputParser(str(out)).parse()["two_component"] is True
    # a real 1c run
    out.write_text(METAL_OUT)
    assert "two_component" not in CrystalOutputParser(str(out)).parse()


def test_soc_deck_keeps_inpsoc_records_it_wrote():
    notes = []
    again = S.soc_deck(SOC_PARENT.replace("TWOCOMPON\nSOC\nEND\n", ""), log=notes.append)
    assert _basis(again) == _basis(SOC_PARENT)
    assert sum("keeps the INPSOC" in n for n in notes) == 2


@pytest.mark.parametrize("change", [
    lambda d: d.replace("INTERNAL 1.0", "INTERNAL 2.0", 1),
    lambda d: d.replace("INTERNAL 1.0", "COLUMBUS 1.0", 1),
    lambda d: d.replace(S.inpsoc_records(S.load_so_ecp(82), 82)[3],
                        S.inpsoc_records(S.load_so_ecp(82), 82)[3].replace("0.000000", "0.100000", 1), 1),
], ids=["soscale", "convention", "numbers"])
def test_an_inpsoc_mace_did_not_write_is_refused(change):
    deck = change(SOC_PARENT.replace("TWOCOMPON\nSOC\nEND\n", ""))
    with pytest.raises(S.SocError, match="INPSOC ECP is not the spin-orbit ECP"):
        S.soc_deck(deck, log=lambda m: None)


@pytest.fixture
def opt2d12_soc_parent(tmp_path, monkeypatch):
    """opt2d12 on a real SOC parent .d12 (the .out is stubbed)."""
    monkeypatch.setattr(M, "CrystalOutputParser",
                        lambda p: _Parser(str(p).replace("socpbte", "pbte"), "out"))
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "socpbte.out").write_text("")
    (tmp_path / "socpbte.d12").write_text(SOC_PARENT)

    def run(**template):
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


def test_a_child_of_a_soc_parent_is_a_soc_deck(opt2d12_soc_parent, capsys):
    status, decks = opt2d12_soc_parent()
    assert status == 0 and len(decks) == 1
    text = decks[0].read_text()
    assert "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in text and "DIIS" not in text
    # the INPSOC records round-trip (the parser strips each record, as for any
    # external basis)
    assert _basis(text) == [ln.strip() for ln in _basis(SOC_PARENT)]
    out = capsys.readouterr().out
    assert "the parent is a two-component SOC run, so this deck is one too" in out


def test_soc_false_in_the_template_writes_a_scalar_child_with_a_warning(opt2d12_soc_parent, capsys):
    status, decks = opt2d12_soc_parent(soc=False)
    assert status == 0 and len(decks) == 1
    text = decks[0].read_text()
    assert "TWOCOMPON" not in text
    err = capsys.readouterr().err
    assert "the parent is a two-component run, but this deck is scalar" in err
    assert "INPSOC" in err and "untested" in err


def test_an_optimisation_of_a_soc_parent_is_refused(opt2d12_soc_parent, capsys):
    status, decks = opt2d12_soc_parent(calculation_type="OPT")
    assert status == 1 and decks == []
    assert "geometry optimization (OPTGEOM) is not available" in capsys.readouterr().err


def test_a_frequency_run_of_a_soc_parent_is_refused(opt2d12_soc_parent, capsys):
    """No frequency calculation in the two-component spinor basis (p. 166)."""
    status, decks = opt2d12_soc_parent(calculation_type="FREQ")
    assert status == 1 and decks == []
    assert "frequency calculation (FREQCALC) is not available" in capsys.readouterr().err


def test_a_parent_known_as_2c_only_from_its_out_gives_a_scalar_deck_with_a_warning(
        opt2d12, monkeypatch, capsys):
    out, deck = PARENTS["pbte"]
    monkeypatch.setitem(PARENTS, "pbte", (dict(out, two_component=True), deck))
    status, decks = opt2d12(["pbte"])
    assert status == 0 and "TWOCOMPON" not in decks["pbte"].read_text()
    err = capsys.readouterr().err
    assert "the parent is a two-component run, but this deck is scalar" in err
    assert "no SOC could be read from a parent .d12" in err


# --- through cif2d12 -----------------------------------------------------------

AU_CIF = """data_Au
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 4.0782
_cell_length_b 4.0782
_cell_length_c 4.0782
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Au1 Au 0 0 0
"""

MAIN_SHA_CIF = {"internal": "93f9103bc90650c1c627fe11a467837c2807154318325e2bef3b01803a5f78de", "external": "38712af78b13cd9c5b57fd7916e1ab8b7610da860f3fdb858a0b8f42ef6e7197"}


@pytest.fixture
def cif2d12(tmp_path):
    pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")
    import NewCifToD12
    from d12_constants import DEFAULT_TOLERANCES
    (tmp_path / "Au.cif").write_text(AU_CIF)
    cif = NewCifToD12.parse_cif(str(tmp_path / "Au.cif"))

    def run(name, **extra):
        opts = dict(dimensionality="CRYSTAL", calculation_type="SP", method="DFT",
                    dft_functional="PBE0", is_spin_polarized=False,
                    tolerances=DEFAULT_TOLERANCES, scf_method="DIIS",
                    symmetry_handling="CIF", basis_set_type="EXTERNAL",
                    basis_set=str(TZ) + "/")
        opts.update(extra)
        out = tmp_path / "Au.d12"
        ok = NewCifToD12.create_d12_file(copy.deepcopy(cif), str(out), opts)
        return ok, out
    return run


@pytest.mark.parametrize("extra", [{}, {"soc": False}], ids=["no key", "soc false"])
@pytest.mark.parametrize("basis", ["internal", "external"])
def test_without_soc_cif2d12_writes_what_main_wrote(cif2d12, extra, basis):
    if basis == "internal":
        extra = dict(extra, basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2")
    ok, out = cif2d12(basis, **extra)
    assert ok is True
    assert _sha(out) == MAIN_SHA_CIF[basis]


def test_cif2d12_soc_true_writes_the_2c_deck(cif2d12):
    ok, out = cif2d12("external", soc=True)
    text = out.read_text()
    assert ok is True and "INPSOC\nINTERNAL 1.0\n19. 0 2 4 4 2 2\n" in text
    assert "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in text
    assert "SHRINK\n10 10\n" in text and "FMIXING\n50\n" in text
    assert "DIIS" not in text


def test_cif2d12_soc_fmixing_reaches_the_deck(cif2d12):
    ok, out = cif2d12("external", soc=True, soc_fmixing=60)
    assert ok is True and "FMIXING\n60\n" in out.read_text()


def test_cif2d12_soc_with_an_internal_basis_writes_nothing(cif2d12):
    ok, out = cif2d12("internal", soc=True, basis_set_type="INTERNAL",
                      basis_set="POB-TZVP-REV2")
    assert ok is False and not out.exists()


# --- SMEAR for metals ------------------------------------------------------------
#
# fcc Au in a 2c-SCF aborted on SHRINK 10 10 and 11 11 and converged at 12 12,
# or at 10 10 with SMEAR 0.005 (HPCC). SMEAR is written only when asked for:
# a "smear" key, or an answer at the prompt opt2d12 shows when the parent's
# .out looks metallic.

DATA = REPO_ROOT / "tests" / "data" / "low_dim_groups"
# The last SCF cycle of a 1c diamond run (spin-polarized) - an insulator.
DIAMOND_LAST_CYCLE = """ CYC   5 ETOT(AU) -7.619411780627E+01 DETOT -7.64E-08 tst  3.28E-07 PX  6.85E-05
 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT FDIK        TELAPSE      142.86 TCPU      142.73

    ALPHA      ELECTRONS
 INSULATING STATE
 TOP OF VALENCE BANDS -    BAND      6; K    1; EIG -7.7969146E-02 AU
 BOTTOM OF VIRTUAL BANDS - BAND      7; K   35; EIG  1.4257519E-01 AU
 INDIRECT ENERGY BAND GAP:   6.0013 eV
 BOTTOM OF VIRTUAL BANDS - BAND      7; K    1; EIG  1.9340298E-01 AU

    BETA       ELECTRONS
 INSULATING STATE
 TOP OF VALENCE BANDS -    BAND      6; K    1; EIG -7.7969146E-02 AU
 BOTTOM OF VIRTUAL BANDS - BAND      7; K   35; EIG  1.4257519E-01 AU
 INDIRECT ENERGY BAND GAP:   6.0013 eV
 BOTTOM OF VIRTUAL BANDS - BAND      7; K    1; EIG  1.9340298E-01 AU
 CYC   6 ETOT(AU) -7.619411779907E+01 DETOT  7.20E-09 tst  4.41E-10 PX  6.85E-05
 == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -7.6194117799070E+01 CYCLES   6
"""
# A metallic last cycle: graphene (layer group 80), a real 1c run.
METAL_OUT = (DATA / "graphene_lg80.out").read_text()


@pytest.mark.parametrize("text,why", [
    (METAL_OUT, '"POSSIBLY CONDUCTING STATE"'),
    ((DATA / "polyyne_rg28.out").read_text(), '"POSSIBLY CONDUCTING STATE"'),
    ((DATA / "graphene_lg37.out").read_text(), "a band gap of 0.0024 eV"),
    (DIAMOND_LAST_CYCLE, None),
    # the last cycle decides: conducting earlier, insulating at the end
    (" CYC   1 ETOT(AU) -1.0\n POSSIBLY CONDUCTING STATE - EFERMI(AU) -1.9E-01 (RES.)\n"
     + DIAMOND_LAST_CYCLE, None),
    ("", None),
], ids=["graphene-lg80", "polyyne", "graphene-lg37-gap", "diamond", "last-cycle", "empty"])
def test_what_a_parent_output_says_about_a_metal(text, why):
    evidence = S.parent_metal_evidence(text)
    assert (evidence is None) if why is None else (why in evidence)


def test_smear_writes_smear_after_shrink_with_a_note():
    notes = []
    soc = S.soc_deck(PBTE, log=notes.append, smear=0.005)
    assert "SHRINK\n8 8\nSMEAR\n0.005000\nTWOCOMPON\nSOC\nEND\n" in soc
    assert "SOC deck: SMEAR 0.005000 added (smear)" in notes
    # the directional SHRINK form keeps its two records together
    slab = PBTE.replace("SHRINK\n8 16\n", "SHRINK\n0 20\n10 10 1\n")
    assert "SHRINK\n0 10\n10 10 1\nSMEAR\n0.005000\n" in S.soc_deck(slab, log=lambda m: None,
                                                                    smear="0.005")
    # a SMEAR the deck has gets the width asked for, noted; the same width: silent
    has = PBTE.replace("SHRINK\n8 16\n", "SHRINK\n8 16\nSMEAR\n0.010000\n")
    notes = []
    soc = S.soc_deck(has, log=notes.append, smear=0.005)
    assert soc.count("SMEAR") == 1 and "SMEAR\n0.005000\n" in soc
    assert "SOC deck: SMEAR 0.010000 replaced by 0.005000 (smear)" in notes
    notes = []
    S.soc_deck(has, log=notes.append, smear=0.01)
    assert not any("SMEAR" in n for n in notes)


def test_no_smear_asked_for_means_none_written():
    assert "SMEAR" not in S.soc_deck(PBTE, log=lambda m: None)


@pytest.mark.parametrize("bad", [0, -0.01, "wide", True, 1e-9, float("nan")])
def test_a_smear_that_is_not_a_width_is_refused(bad):
    with pytest.raises(S.SocError, match="smear must be a width in hartree"):
        S.soc_deck(PBTE, log=lambda m: None, smear=bad)


def _metal_parent(tmp_path, stem="pbte"):
    (tmp_path / f"{stem}.out").write_text(METAL_OUT)


def test_batch_metal_parent_warns_in_the_summary_and_leaves_the_deck(opt2d12, tmp_path, capsys):
    status, decks = opt2d12(["pbte"], soc=True)
    plain = decks["pbte"].read_text()
    capsys.readouterr()
    _metal_parent(tmp_path)
    status, decks = opt2d12(["pbte"], soc=True)
    assert status == 0
    assert decks["pbte"].read_text() == plain and "SMEAR" not in plain
    err = capsys.readouterr().err
    assert 'the parent looks metallic (its last SCF cycle reports "POSSIBLY CONDUCTING STATE")' in err
    summary = err[err.rindex("SOC decks written without SMEAR from a parent that looks metallic"):]
    assert "  pbte.out" in summary


def test_batch_of_several_lists_the_metallic_parents_in_the_summary(opt2d12, tmp_path, capsys):
    _metal_parent(tmp_path)
    status, decks = opt2d12(["pbte", "pbi"], soc=True)
    assert status == 0 and decks["pbte"].exists() and decks["pbi"].exists()
    err = capsys.readouterr().err
    assert "pbte.out: wrote" in err and "WARNING: SOC deck without SMEAR" in err
    summary = err[err.rindex("SOC decks written without SMEAR from a parent that looks metallic"):]
    assert "  pbte.out" in summary and "pbi.out" not in summary


def test_an_insulating_parent_gets_no_smear_warning(opt2d12, tmp_path, capsys):
    (tmp_path / "pbte.out").write_text(DIAMOND_LAST_CYCLE)
    status, _ = opt2d12(["pbte"], soc=True)
    assert status == 0 and "metallic" not in capsys.readouterr().err


def test_a_scalar_deck_of_a_metal_parent_is_not_touched(opt2d12, tmp_path, capsys):
    _metal_parent(tmp_path)
    status, decks = opt2d12(["pbte", "pbi", "nacl"])
    assert status == 0
    assert {s: _sha(p) for s, p in decks.items()} == MAIN_SHA
    assert "metallic" not in capsys.readouterr().err


def test_template_smear_writes_smear_and_needs_no_warning(opt2d12, tmp_path, capsys):
    _metal_parent(tmp_path)
    status, decks = opt2d12(["pbte"], soc=True, smear=0.005)
    assert status == 0
    assert "SHRINK\n8 8\nSMEAR\n0.005000\nTWOCOMPON\n" in decks["pbte"].read_text()
    out = capsys.readouterr()
    assert "SOC deck: SMEAR 0.005000 added (smear)" in out.out
    assert "metallic" not in out.err


class _TTY:
    def isatty(self):
        return True


@pytest.mark.parametrize("answers,smear", [
    ([""], None),                          # Enter: no SMEAR
    (["wide", "-1", "0.005"], "0.005000"),  # asked again until a width
])
def test_at_a_terminal_the_user_is_asked_for_smear(opt2d12, tmp_path, monkeypatch, capsys,
                                                   answers, smear):
    _metal_parent(tmp_path)
    monkeypatch.setattr(sys, "stdin", _TTY())
    asked = []
    replies = iter(answers)

    def fake_input(prompt=""):
        asked.append(prompt)
        if prompt.startswith("SMEAR width"):
            return next(replies)
        raise AssertionError(f"unexpected question {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(M, "yes_no_prompt", lambda *a, **k: True)
    # one file, no --yes: someone is at the terminal
    monkeypatch.setattr(M, "stdin_is_terminal", lambda: True)
    status, decks = opt2d12(["pbte"], soc=True, _yes=False)
    assert status == 0
    assert sum(p.startswith("SMEAR width") for p in asked) == len(answers)
    text = decks["pbte"].read_text()
    if smear:
        assert f"SHRINK\n8 8\nSMEAR\n{smear}\nTWOCOMPON\n" in text
    else:
        assert "SMEAR" not in text
    err = capsys.readouterr().err
    assert "the parent looks metallic" in err
    assert "SOC decks written without SMEAR" not in err   # answered, not noted


@pytest.fixture
def fresh_cif_warning(monkeypatch):
    monkeypatch.setattr(S, "_cif_metal_warned", [])


def test_cif2d12_soc_without_smear_warns_once(cif2d12, capsys, fresh_cif_warning):
    for _ in range(2):
        ok, out = cif2d12("external", soc=True)
        assert ok is True and "SMEAR" not in out.read_text()
    err = capsys.readouterr().err
    assert err.count("MACE cannot tell from a CIF whether a system is a metal") == 1


def test_cif2d12_smear_key_writes_smear_without_the_warning(cif2d12, capsys, fresh_cif_warning):
    ok, out = cif2d12("external", soc=True, smear=0.005)
    assert ok is True and "SHRINK\n10 10\nSMEAR\n0.005000\nTWOCOMPON\n" in out.read_text()
    assert "cannot tell from a CIF" not in capsys.readouterr().err


def test_cif2d12_scalar_decks_get_no_warning(cif2d12, capsys, fresh_cif_warning):
    ok, out = cif2d12("external")
    assert ok is True and _sha(out) == MAIN_SHA_CIF["external"]
    assert "cannot tell from a CIF" not in capsys.readouterr().err
