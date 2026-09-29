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
        plain[plain.index("END", plain.index("99 0")) + 1:plain.index("SCFDIR")]
    assert soc[j:] == plain[plain.index("SCFDIR"):]
    # geometry untouched
    assert soc[:soc.index("END")] == plain[:plain.index("END")]


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
    (lambda d: d.replace("PBE0\n", "B3LYP-D3\n"), "B3LYP-D3"),
    (lambda d: d.replace("PBE0\n", "HSE06\n"), "HSE06"),
    (lambda d: d.replace("PBE0\n", "SPIN\nPBE0\n"), "SPIN"),
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

    def run(stems, **template):
        base = {"calculation_type": "SP", "functional": "PBE0", "dispersion": False,
                "dft_grid": "XLGRID", "basis_set": M.PARENT_BASIS_MARKER,
                "basis_set_type": "EXTERNAL", "use_original_external_basis": True,
                "has_original_external_basis": True}
        base.update(template)
        (tmp_path / "t.json").write_text(json.dumps(base))
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", "--out-file",
                                          *[f"{s}.out" for s in stems], "--config-file",
                                          "t.json", "--output-dir", "sp", "--yes"])
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


def test_soc_with_dispersion_is_refused(opt2d12, capsys):
    status, decks = opt2d12(["pbte"], soc=True, dispersion=True)
    assert status == 1 and not decks["pbte"].exists()
    assert "not supported in a two-component SCF" in capsys.readouterr().err


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


def test_cif2d12_soc_with_an_internal_basis_writes_nothing(cif2d12):
    ok, out = cif2d12("internal", soc=True, basis_set_type="INTERNAL",
                      basis_set="POB-TZVP-REV2")
    assert ok is False and not out.exists()
