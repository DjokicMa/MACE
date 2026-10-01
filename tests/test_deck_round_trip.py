"""A deck MACE wrote, read by CrystalInputParser and written back by opt2d12's
deck writer (CRYSTALOptToD12.write_d12_file), is the same deck byte for byte.

What the parse lost before, measured on the decks cif2d12 writes in this
module (811 decks: every dimensionality, internal and EXTERNAL basis,
SP/OPT/FREQ, and the option variants; 80 came back identical) and on the 401
real decks of the test corpus:

* coordinates were kept only as floats, so ``1.250000000000E-01`` or
  ``0.2500000000`` came back as ``0.125`` / ``0.25``: the parse now keeps
  each atom's x, y, z as written, and the record's last word (MACE writes the
  element symbol, or X when it was handed an ECP number such as 247);
* an EXTERNAL basis came back with its records' spacing stripped (a basis
  file's ``  13575.349682      0.00022245814352``): the records are also
  kept as written;
* a FREQCALC block was read for NUMDERIV only (see
  test_freqcalc_parent_settings.py);
* an RHF deck (no Hamiltonian keyword, "RHF [default]", manual p. 123) and
  the HF-3c / HFsol-3c decks (manual sec. 5.3.1, pp. 158, 162) were read with
  no functional, and written back with a DFT block;
* SMEAR was read under a key the writer does not take.

All of them now come back identical. The title is the one record the parse
cannot give the writer: write_d12_file names the deck after its file, so a
deck is written to a file named after its title. The corpus decks' titles are
``./<name>`` (an older MACE took the title from the path), which no file name
gives; they are compared from the second record on.
"""
import contextlib
import io
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import CRYSTALOptToD12 as opt2d12
from conftest import REPO_ROOT, TEST_DATA
from d12_constants import DEFAULT_TOLERANCES
from d12_parsers import CrystalInputParser

DATA = Path(__file__).parent / "data"
MACE_CLI = REPO_ROOT / "mace_cli"
EXTERNAL_BASIS = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta") + "/"


def parse(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return CrystalInputParser(str(path)).parse()


def rebuild(parsed, out_dir):
    """The deck write_d12_file writes from the parse, as text.

    Only the parse is used. The geometry goes in the form opt2d12 hands the
    writer (cell parameters, one record per atom); each atom gets the number
    the writer was given when it wrote the deck: the number as written, or
    the plain atomic number when the record ends in its element symbol
    (the writer adds 200 for an ECP element of an EXTERNAL basis itself).
    """
    cp = parsed.get("cell_parameters")
    cell = None
    if cp:
        cell = [cp["a"], cp["b"] if cp["b"] is not None else cp["a"],
                cp["c"] if cp["c"] is not None else 500.0,
                cp["alpha"], cp["beta"], 90.0 if cp["gamma"] is None else cp["gamma"]]
    coords = []
    for atom, xyz, label in zip(parsed["atoms"], parsed["atom_coordinate_text"],
                                parsed["atom_labels"]):
        number = atom["atom_number"] if label in (None, "X") else atom["atomic_number"]
        x, y, z = xyz
        coords.append({"atom_number": str(number), "x": x, "y": y, "z": z})
    settings = dict(parsed)
    settings.setdefault("calculation_type", "SP")
    settings["write_only_unique"] = False
    if parsed["dimensionality"] in ("SLAB", "POLYMER"):
        settings["spacegroup"] = parsed.get("layer_group") or parsed.get("rod_group")
    basis = parsed.get("external_basis_text") or None
    if basis:
        settings["use_original_external_basis"] = True
    name = Path(parsed["title"]).name
    out = Path(out_dir) / f"{name}.d12"
    with contextlib.redirect_stdout(io.StringIO()):
        assert opt2d12.write_d12_file(
            str(out), {"conventional_cell": cell, "coordinates": coords,
                       "crystallographic_coordinates": coords},
            settings, external_basis_data=basis,
            parent_k_points=parsed.get("k_points"), ask=False)
    return out.read_text()


def round_trip(path, out_dir):
    return rebuild(parse(path), out_dir)


# ------------------------------------------------------------ cif2d12 decks

def _cif2d12():
    pytest.importorskip("ase", reason="NewCifToD12 reads CIFs with ase")
    import NewCifToD12
    return NewCifToD12


def _structures(writer):
    dia = writer.parse_cif(str(DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif"))
    dia3 = writer.parse_cif(str(DATA / "3_dia3_opt_BULK_OPTGEOM_symm.cif"))
    cell = dict(alpha=90.0, beta=90.0, gamma=90.0)
    return {
        "dia": ("CRYSTAL", dia, {}),
        "dia3": ("CRYSTAL", dia3, {}),
        # Ag is an ECP element of the external sets: written as 247
        "agbr": ("CRYSTAL", dict(a=5.77, b=5.77, c=5.77, spacegroup=225, **cell,
                                 atomic_numbers=[47, 35], symbols=["Ag", "Br"],
                                 positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), {}),
        "graphene": ("SLAB", dict(a=2.47, b=2.47, c=20.0, alpha=90.0, beta=90.0, gamma=120.0,
                                  spacegroup=191, atomic_numbers=[6], symbols=["C"],
                                  positions=[[-1.0 / 3.0, 1.0 / 3.0, 0.0]]),
                     {"layer_group": 80}),
        "sn_chain": ("POLYMER", dict(a=4.431, b=10.0, c=10.0, spacegroup=1, **cell,
                                     atomic_numbers=[16, 7], symbols=["S", "N"],
                                     positions=[[0.0, 0.5, 0.5], [0.14160054, 0.567, 0.5]]),
                     {"rod_group": 1}),
        "water": ("MOLECULE", dict(a=20.0, b=20.0, c=20.0, spacegroup=1, **cell,
                                   atomic_numbers=[8, 1, 1], symbols=["O", "H", "H"],
                                   positions=[[0.5, 0.5, 0.5], [0.5379, 0.5, 0.5233],
                                              [0.4621, 0.5, 0.5233]]), {}),
    }


def _options(dim, **overrides):
    opts = dict(dimensionality=dim, calculation_type="SP", basis_set_type="INTERNAL",
                basis_set="POB-TZVP-REV2", method="DFT", dft_functional="PBE",
                is_spin_polarized=False, tolerances=dict(DEFAULT_TOLERANCES),
                scf_method="DIIS", symmetry_handling="CIF", write_only_unique=True)
    opts.update(overrides)
    return opts


METHODS = {
    "PBE": dict(dft_functional="PBE"),
    "B3LYP-D3": dict(dft_functional="B3LYP", use_dispersion=True, dft_grid="XLGRID"),
    "HSE06-spin": dict(dft_functional="HSE06", is_spin_polarized=True, dft_grid="LGRID"),
    "PBESOL": dict(dft_functional="PBESOL"),
    "PWGGA": dict(dft_functional="PWGGA"),
    "RHF": dict(method="HF", hf_method="RHF"),
    "UHF": dict(method="HF", hf_method="UHF", is_spin_polarized=True),
    # composite methods: internal basis only
    "HSESOL3C": dict(dft_functional="HSESOL3C", basis_set="SOLDEF2MSVP"),
    "PBEH3C": dict(dft_functional="PBEH3C", basis_set="MINIX"),
    "HF3C": dict(method="HF", hf_method="HF3C", basis_set="MINIX"),
}
OPT = {"MAXCYCLE": 800, "TOLDEG": 0.0003, "TOLDEX": 0.0012, "TOLDEE": 7}
CALCS = {
    "SP": dict(calculation_type="SP"),
    "OPT": dict(calculation_type="OPT", optimization_type="FULLOPTG",
                optimization_settings=dict(OPT)),
    "OPT-ATOMONLY": dict(calculation_type="OPT", optimization_type="ATOMONLY",
                         optimization_settings={"MAXCYCLE": 500, "TOLDEG": 0.0001,
                                                "TOLDEX": 0.0004, "TOLDEE": 8}),
    "FREQ": dict(calculation_type="FREQ",
                 freq_settings={"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}),
    "FREQ-NUMDERIV": dict(calculation_type="FREQ", freq_settings={"numderiv": 2}),
    "FREQ-IR": dict(calculation_type="FREQ",
                    freq_settings={"intensities": True, "ir_method": "CPHF"}),
    "FREQ-RAMAN": dict(calculation_type="FREQ",
                       freq_settings={"intensities": True, "raman": True, "irspec": True,
                                      "ramspec": True, "spec_range": [0, 4000]}),
}


def _write_and_round_trip(writer, cif, name, options, tmp_path):
    written = tmp_path / "written"
    again = tmp_path / "again"
    written.mkdir(exist_ok=True)
    again.mkdir(exist_ok=True)
    deck = written / f"{name}.d12"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert writer.create_d12_file(dict(cif), str(deck), options, interactive=False), name
    return deck.read_text(), round_trip(deck, again)


@pytest.mark.parametrize("basis", ["INTERNAL", "EXTERNAL"])
@pytest.mark.parametrize("structure", ["dia", "dia3", "agbr", "graphene", "sn_chain", "water"])
def test_cif2d12_decks_round_trip(tmp_path, structure, basis):
    writer = _cif2d12()
    dim, cif, extra = _structures(writer)[structure]
    different = []
    n = 0
    for method, method_opts in METHODS.items():
        if basis == "EXTERNAL" and "basis_set" in method_opts:
            continue
        for calc, calc_opts in CALCS.items():
            opts = _options(dim, **extra, **method_opts, **calc_opts)
            if basis == "EXTERNAL":
                opts.update(basis_set_type="EXTERNAL", basis_set=EXTERNAL_BASIS)
            name = f"{structure}_{basis}_{method}_{calc}"
            original, rebuilt = _write_and_round_trip(writer, cif, name, opts, tmp_path)
            n += 1
            if rebuilt != original:
                different.append(name)
    assert n >= 28
    assert not different, different


# One option at a time on the diamond deck: the SCF, OPTGEOM and FREQCALC
# records cif2d12 can write, and the functionals of its menus.
VARIANTS = {
    "smear": dict(use_smearing=True, smearing_width=0.005),
    "spinlock": dict(is_spin_polarized=True, spinlock=2, spinlock_cycles=30),
    "anderson": dict(scf_method="ANDERSON"),
    "broyden": dict(scf_method="BROYDEN"),
    "maxcycle-fmixing": dict(scf_maxcycle=1600, fmixing=50),
    "tolerances": dict(tolerances={"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}),
    "no-grid": dict(dft_grid=None),
    "oldgrid": dict(dft_grid="OLDGRID"),
    "guessp": dict(guessp=True),
    "p1": dict(symmetry_handling="P1", write_only_unique=False),
    "hfsol3c": dict(method="HF", hf_method="HFSOL3C", basis_set="SOLMINIX"),
    "uhf-external": dict(method="HF", hf_method="UHF", is_spin_polarized=True,
                         basis_set_type="EXTERNAL", basis_set=EXTERNAL_BASIS),
    "cellonly": dict(calculation_type="OPT", optimization_type="CELLONLY",
                     optimization_settings=dict(OPT)),
    "itatocel": dict(calculation_type="OPT", optimization_type="ITATOCEL",
                     optimization_settings=dict(OPT)),
    "cvolopt": dict(calculation_type="OPT", optimization_type="CVOLOPT",
                    optimization_settings=dict(OPT)),
    "maxtradius": dict(calculation_type="OPT", optimization_type="FULLOPTG",
                       optimization_settings=dict(OPT, MAXTRADIUS=0.25)),
    "freq-temperature": dict(calculation_type="FREQ",
                             freq_settings={"temprange": (20, 0, 400), "pressrange": (5, 0, 10)}),
    "freq-stepsize": dict(calculation_type="FREQ",
                          freq_settings={"stepsize": 0.001, "numderiv": 1}),
    "freq-isotopes": dict(calculation_type="FREQ", freq_settings={"isotopes": {1: 13.003}}),
    "freq-modes": dict(calculation_type="FREQ",
                       freq_settings={"mode_selection": "IR", "analysis": True,
                                      "eckart": False, "print_modes": False}),
    "freq-wannier": dict(calculation_type="FREQ",
                         freq_settings={"intensities": True, "ir_method": "WANNIER",
                                        "relocalize_wannier": True}),
    "freq-dieliso": dict(calculation_type="FREQ",
                         freq_settings={"intensities": True, "dielectric_constant": 5.7,
                                        "born_tensor_norm": True}),
    "freq-dieltens": dict(calculation_type="FREQ",
                          freq_settings={"intensities": True,
                                         "dielectric_tensor": [5.7, 0, 0, 0, 5.7, 0, 0, 0, 5.7]}),
    "freq-cphf": dict(calculation_type="FREQ",
                      freq_settings={"intensities": True, "ir_method": "CPHF",
                                     "cphf_settings": {"fmixing": 60, "tolalpha": 4,
                                                       "maxcycle": 150}}),
    "freq-raman-cphf": dict(calculation_type="FREQ",
                            freq_settings={"intensities": True, "raman": True,
                                           "cphf_settings": {"fmixing2": 60, "tolgamma": 3,
                                                             "maxcycle2": 150},
                                           "ramanexp": (298, 532), "norenorm": True}),
    "freq-irspec": dict(calculation_type="FREQ",
                        freq_settings={"intensities": True, "irspec": True,
                                       "spec_range": [0, 4000], "spec_step": 1.0,
                                       "spec_dampfac": 8.0, "spec_gaussian": True,
                                       "spec_angle": 20.0, "spec_refrind": True,
                                       "spec_dielfun": True, "dielectric_constant": 5.7}),
    "freq-ramspec": dict(calculation_type="FREQ",
                         freq_settings={"intensities": True, "raman": True, "ramspec": True,
                                        "spec_range": [100, 3500], "spec_step": 2.0,
                                        "raman_voigt": 0.5, "raman_dampfac": 4.0}),
    "freq-preoptgeom": dict(calculation_type="FREQ",
                            freq_settings={"preoptgeom": True,
                                           "optgeom_settings": {"fulloptg": True,
                                                                "toldeg": 0.0001,
                                                                "toldex": 0.0004}}),
    "freq-fragment": dict(calculation_type="FREQ", freq_settings={"fragment": [1, 2]}),
    "freq-misc": dict(calculation_type="FREQ",
                      freq_settings={"neglectfreq": 3, "multitask": 4, "freqscale": 0.97,
                                     "noanalysis": True}),
    "freq-dispersion": dict(calculation_type="FREQ",
                            freq_settings={"dispersion": True, "scelphono": [2, 2, 2],
                                           "interphess": {"expand": [4, 4, 4], "print": 0},
                                           "bands": {"shrink": 16, "npoints": 100,
                                                     "path": [[0, 0, 0, 8, 0, 8],
                                                              [8, 0, 8, 8, 4, 12]]},
                                           "pdos": {"max_freq": 2000, "nbins": 200,
                                                    "projected": True},
                                           "ins": {"max_freq": 3000, "nbins": 300,
                                                   "neutron_type": 2}}),
}
FUNCTIONALS = ["PBE0", "B3LYP", "HSE06", "HSEsol", "PBESOL0", "M06", "SCAN", "LC-wPBE",
               "mPW1PW91", "SOGGA", "VBH", "WCGGA", "BLYP", "SVWN", "B3PW", "wB97X",
               "r2SCAN", "M062X", "B97H", "HSE3C", "B973C", "PBESOL03C"]


def test_cif2d12_option_variants_round_trip(tmp_path):
    writer = _cif2d12()
    _dim, dia, _ = _structures(writer)["dia"]
    variants = dict(VARIANTS)
    for fn in FUNCTIONALS:
        variants[f"{fn}"] = dict(dft_functional=fn)
        variants[f"{fn}-D3"] = dict(dft_functional=fn, use_dispersion=True)
        variants[f"{fn}-spin"] = dict(dft_functional=fn, is_spin_polarized=True)
    different = []
    for name, overrides in variants.items():
        opts = _options("CRYSTAL", dft_grid="XLGRID")
        opts.update(overrides)
        original, rebuilt = _write_and_round_trip(writer, dia, name, opts, tmp_path)
        if rebuilt != original:
            different.append(name)
    assert not different, different


# ------------------------------------------------------------- opt2d12 decks

LOW_DIM = DATA / "low_dim_groups"
OPT_RESTART = DATA / "opt_restart"


def _opt2d12_child(tmp_path, out, deck, calc_type):
    shutil.copy(out, tmp_path / out.name)
    shutil.copy(deck, tmp_path / deck.name)
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", out.name,
         "--d12-file", deck.name, "--non-interactive", "--calc-type", calc_type],
        cwd=tmp_path, input="", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    kids = [p for p in tmp_path.glob("*.d12") if p.name != deck.name]
    assert len(kids) == 1, sorted(p.name for p in tmp_path.iterdir())
    return kids[0]


@pytest.mark.parametrize("calc_type", ["SP", "OPT", "FREQ"])
@pytest.mark.parametrize("parent", [LOW_DIM / "graphene_lg80", LOW_DIM / "polyyne_rg28",
                                    OPT_RESTART / "tqc_agbr"], ids=lambda p: p.name)
def test_opt2d12_children_round_trip(tmp_path, parent, calc_type):
    """SLAB, POLYMER and an EXTERNAL-basis CRYSTAL with an ECP atom (247)."""
    work = tmp_path / "work"
    work.mkdir()
    child = _opt2d12_child(work, parent.with_suffix(".out"), parent.with_suffix(".d12"),
                           calc_type)
    assert round_trip(child, tmp_path) == child.read_text()


# ---------------------------------------------------------- real HPCC decks

def test_real_slab_deck_round_trips(tmp_path):
    deck = LOW_DIM / "graphene_lg80.d12"
    assert round_trip(deck, tmp_path) == deck.read_text()


@pytest.mark.parametrize("deck", sorted((DATA / "ecp_decks").glob("*.d12")),
                         ids=lambda p: p.name[:6])
def test_real_external_ecp_decks_round_trip_after_the_title(tmp_path, deck):
    """Decks opt2d12 wrote and CRYSTAL ran: EXTERNAL basis read verbatim from
    full.basis.triplezeta (its own spacing), Pb and Ag as ECP atoms 282/247
    labelled X. Their title is "./<name>" (see the module docstring).
    (tests/data/opt_restart holds the same kind of deck with a RESTART that
    mace/recovery put into OPTGEOM, a record the deck writer has no input for.)"""
    original = deck.read_text().split("\n")
    assert original[0].startswith("./")
    assert round_trip(deck, tmp_path).split("\n")[1:] == original[1:]


# ------------------------------------------------------ what the parse reads

def _parse_text(tmp_path, text):
    deck = tmp_path / "deck.d12"
    deck.write_text(textwrap.dedent(text).lstrip("\n"))
    return parse(deck)


DIAMOND = """
    dia
    CRYSTAL
    0 0 0
    227
    3.56700000
    1
    6 1.250000000000E-01 1.250000000000E-01 1.250000000000E-01 Biso 1.000000 C
    {block}BASISSET
    {basis}
    {hamiltonian}TOLINTEG
    7 7 7 7 14
    TOLDEE
    7
    SHRINK
    8 16
    SCFDIR
    MAXCYCLE
    800
    FMIXING
    30
    DIIS
    HISTDIIS
    100
    PPAN
    END
"""


def _diamond(block="", basis="POB-TZVP-REV2", hamiltonian="DFT\nPBE\nENDDFT\n"):
    return textwrap.dedent(DIAMOND).lstrip("\n").format(
        block=block, basis=basis, hamiltonian=hamiltonian)


def test_coordinates_and_labels_are_kept_as_written(tmp_path):
    p = _parse_text(tmp_path, _diamond())
    assert p["atom_coordinate_text"] == [["1.250000000000E-01"] * 3]
    assert p["atom_labels"] == ["C"]
    assert p["title"] == "dia"


@pytest.mark.parametrize("hamiltonian,functional", [
    ("", "RHF"),
    ("UHF\n", "UHF"),
    ("HF3C\nEND\n", "HF3C"),
    ("HFSOL3C\nEND\n", "HFSOL3C"),
])
def test_hartree_fock_decks_read_their_hamiltonian(tmp_path, hamiltonian, functional):
    p = _parse_text(tmp_path, _diamond(basis="MINIX", hamiltonian=hamiltonian))
    assert p["functional"] == functional
    assert p["method"] == "HF"


def test_smear_is_read_under_the_writers_key(tmp_path):
    p = _parse_text(tmp_path, _diamond().replace("SCFDIR\n", "SMEAR\n0.005000\nSCFDIR\n"))
    assert p["smearing"] is True and p["smearing_width"] == 0.005


def test_external_basis_records_are_kept_with_their_spacing(tmp_path):
    text = _diamond().replace("BASISSET\nPOB-TZVP-REV2\n",
                              "END\n6 1\n0 0 1 2.0 1.0\n  0.1644071000   1.00000000000000\n"
                              "99 0\nEND\n")
    p = _parse_text(tmp_path, text)
    assert p["external_basis_data"] == ["6 1", "0 0 1 2.0 1.0",
                                        "0.1644071000   1.00000000000000"]
    assert p["external_basis_text"] == ["6 1", "0 0 1 2.0 1.0",
                                        "  0.1644071000   1.00000000000000"]


# ------------------------------------------- what reaches an opt2d12 child

def test_record_keys_never_reach_a_childs_settings():
    """Title, verbatim records and coordinates describe the parent deck; a
    child's are its own (its file name, the .out's geometry)."""
    from d12_parsers import DECK_GEOMETRY_KEYS, DECK_TEXT_KEYS
    assert {"title", "external_basis_text"} <= set(DECK_TEXT_KEYS)
    assert {"atom_coordinate_text", "atom_labels"} <= set(DECK_GEOMETRY_KEYS)


# ------------------------------------------------------------ the corpus

def _hand_edited(text):
    """Records MACE's writer never writes, in decks edited after it wrote them:
    SUPERCEL; a RESTART in OPTGEOM (mace/recovery adds it to a deck's text);
    GUESSP moved below MAXCYCLE; RAMSPEC ahead of IRSPEC; comments."""
    return ("SUPERCEL" in text or "#" in text
            or re.search(r"\nOPTGEOM\n\w+\nRESTART\n", text)
            or re.search(r"\nMAXCYCLE\n\d+\nGUESSP\n", text)
            or ("RAMSPEC" in text and "IRSPEC" in text
                and text.index("RAMSPEC") < text.index("IRSPEC")))


def test_every_corpus_deck_mace_wrote_round_trips(tmp_path):
    decks = sorted(TEST_DATA.glob("*/*.d12")) if TEST_DATA.is_dir() else []
    if not decks:
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    different, checked = [], 0
    for deck in decks:
        text = deck.read_text()
        if _hand_edited(text):
            continue
        checked += 1
        if round_trip(deck, tmp_path).split("\n")[1:] != text.split("\n")[1:]:
            different.append(deck.name)
    assert checked >= 300
    assert not different, different
