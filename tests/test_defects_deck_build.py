"""mace.defects.deck_edit: defect deck sets are the host deck plus records.

The reference decks (tests/defects_data) are the ones run on HPCC with
CRYSTAL23 for the round-6 defect facts: they converged, so a builder that
reproduces them byte for byte (titles aside) writes decks CRYSTAL accepts and
places DOPING / SPINLOCK where the measured runs had them.
"""
import json
import subprocess
import sys

import pytest

from _defects_fixtures import DATA, deck, edits, hbn_host, mgo_host, same_but_title, strip_records
from conftest import REPO_ROOT

from mace.defects import DefectError, NeedsPristineOutput
from mace.defects import deck_edit as de

MGO_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def _build(host_text, site, keyword="SUPERCON", matrix=MGO_222, charges=(0,), pristine_n=None,
           name="mgo", **opts):
    host, _ = de.classify_host(host_text, insert=site.kind == "interstitial",
                               host_spin=opts.get("host_spin", "refuse"))
    mult = de.supercell_multiplicity(host, keyword, matrix)
    electrons = de.supercell_electrons(host, mult, pristine_n)
    o = de.SetOptions(charges=list(charges), **opts)
    decks, msgs, added = de.build_supercell_set(host, site, keyword, matrix, o, electrons, name)
    return {d.name: d for d in decks}, msgs, added


# --- 1-2: MgO V_O ------------------------------------------------------------

def test_mgo_vacancy_set_reproduces_the_measured_decks():
    decks, msgs, _ = _build(mgo_host(), de.DefectSite("vacancy", "O", label=1), charges=(0, 1, 2))
    assert same_but_title(decks["mgo_pris"].text, deck("mgo_pris"))
    assert same_but_title(decks["mgo_q0"].text, deck("mgo_v0"))
    assert same_but_title(decks["mgo_qp1"].text, deck("mgo_v1"))
    assert same_but_title(decks["mgo_qp2"].text, deck("mgo_v2"))
    # every choice is printed
    assert any("q=+1" in m and "odd" in m and "SPINLOCK 1 30" in m for m in msgs)
    assert any("q=+2" in m and "even" in m and "DOPING -2.0" in m for m in msgs)


def test_only_whitelisted_records_are_added():
    host = mgo_host()
    decks, _, _ = _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0, 1, 2))
    allowed = {"SUPERCON", "2 0 0", "0 2 0", "0 0 2", "ATOMREMO", "1", "DOPING", "-1.0", "-2.0",
               "SPIN", "SPINLOCK", "1 30"}
    for d in decks.values():
        ins, dele, rep = edits(host, d.text)
        assert not dele and not rep, d.name
        assert set(ins) <= allowed, (d.name, set(ins) - allowed)


def test_doping_value_and_tags():
    assert de.doping_value(1) == "-1.0"
    assert de.doping_value(-1) == "1.0"
    assert de.doping_value(2.0) == "-2.0"
    assert [de.q_tag(q) for q in (0, 1, -1, 2)] == ["q0", "qp1", "qm1", "qp2"]


# --- 3: hBN sizes ------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_hbn_sizes_reproduce_the_measured_decks(n):
    # 12 electrons per primitive hBN cell (N + B); odd sizes need the pristine count
    decks, _, _ = _build(hbn_host(), de.DefectSite("vacancy", "N", label=1), keyword="SUPERCEL",
                         matrix=[[n, 0], [0, n]], charges=(-1, 0, 1), pristine_n=12 * n * n,
                         name=f"hbn{n}")
    for tag in ("pris", "q0", "qp1", "qm1"):
        assert same_but_title(decks[f"hbn{n}_{tag}"].text, deck(f"hbn{n}_{tag}")), tag


def test_odd_multiplicity_without_pristine_output_waits_for_it():
    with pytest.raises(NeedsPristineOutput):
        _build(hbn_host(), de.DefectSite("vacancy", "N", label=1), keyword="SUPERCEL",
               matrix=[[3, 0], [0, 3]], charges=(0,))
    # an even multiplicity fixes the parity without any output
    decks, _, _ = _build(hbn_host(), de.DefectSite("vacancy", "N", label=1), keyword="SUPERCEL",
                         matrix=[[4, 0], [0, 4]], charges=(0,), name="hbn4")
    assert same_but_title(decks["hbn4_q0"].text, deck("hbn4_q0"))


def test_supercon_multiplicity_counts_the_centring():
    host, _ = de.classify_host(mgo_host())
    assert de.supercell_multiplicity(host, "SUPERCON", MGO_222) == 32   # F: 4 per cell
    assert de.supercell_multiplicity(host, "SUPERCEL", MGO_222) == 8


# --- 4: substitution ----------------------------------------------------------

def test_n_on_o_substitution_parity():
    decks, msgs, _ = _build(mgo_host(), de.DefectSite("substitution", "O", new_element="N", label=1),
                            charges=(-1, 0))
    q0 = decks["mgo_q0"].text
    assert "ATOMSUBS\n1\n1 7\n" in q0
    assert "DFT\nSPIN\nPBE0" in q0 and "SPINLOCK\n1 30\nPPAN\nEND" in q0      # 639 electrons
    qm1 = decks["mgo_qm1"].text
    assert "DOPING\n1.0\nPPAN\nEND" in qm1 and "SPIN" not in qm1             # 640
    assert decks["mgo_q0"].odd and not decks["mgo_qm1"].odd


# --- 5: interstitial -------------------------------------------------------------

def test_interstitial_record_and_too_close_refusal():
    from mace.defects import structure as st
    geom = st.parse_geometry((DATA / "mgo_pris_geometry.out").read_text())
    host, _ = de.classify_host(mgo_host(), insert=True)
    point = st.insertion_point(host, geom, (0.25, 0.25, 0.25), False, "SUPERCON", MGO_222)
    assert point.tolist() == pytest.approx([1.053, 1.053, 1.053])
    site = de.DefectSite("interstitial", "O", cart=tuple(point))
    decks, msgs, _ = _build(mgo_host(), site, charges=(0, -2))
    assert "ATOMINSE\n1\n8 1.053000 1.053000 1.053000\nBASISSET" in decks["mgo_q0"].text
    assert "DOPING\n2.0\nPPAN" in decks["mgo_qm2"].text
    assert any("untested" in m for m in msgs)
    with pytest.raises(DefectError, match="closer than 0.8"):
        st.insertion_point(host, geom, (0.05, 0, 0), False, "SUPERCON", MGO_222)
    with pytest.raises(DefectError, match="closer than 0.8"):
        st.insertion_point(host, geom, (0.3, 0, 0), True, "SUPERCON", MGO_222)


def test_insert_refuses_unit_changing_block1_keywords():
    host = mgo_host().replace("12 0.5 0.5 0.5\n", "12 0.5 0.5 0.5\nFRACTION\n")
    with pytest.raises(DefectError, match="FRACTION"):
        de.classify_host(host, insert=True)
    de.classify_host(host, insert=False)


# --- 10: refusals ----------------------------------------------------------------

SLAB_HOST = hbn_host()


def _refused(text, match, **kw):
    with pytest.raises(DefectError, match=match):
        de.classify_host(text, **kw)


def test_refusals_name_their_reason():
    m = mgo_host()
    _refused("t\nMOLECULE\n1\n1\n8 0 0 0\nEND\n", "MOLECULE host")
    _refused("t\nPOLYMER\n1\n2.5\n1\n6 0 0 0\nEND\n", "POLYMER")
    _refused("t\nEXTERNAL\nBASISSET\nPOB-TZVP-REV2\nEND\n", "EXTERNAL")
    _refused(m.replace("BASISSET", "OPTGEOM\nFULLOPTG\nEND\nBASISSET"), "opt2d12 --calc-type SP")
    _refused(m.replace("BASISSET", "FREQCALC\nEND\nBASISSET"), "frequency")
    _refused(m.replace("PPAN", "DOPING\n-1.0\nPPAN"), "already charged")
    _refused(m.replace("BASISSET", "SUPERCEL\n2 0 0\n0 2 0\n0 0 2\nBASISSET"), "already has a supercell")
    _refused(m.replace("BASISSET", "KEEPSYMM\nBASISSET"), "KEEPSYMM in block 1")
    _refused(m.replace("PPAN", "SPINLOCK\n0 30\nPPAN"), "--host-spin keep")
    de.classify_host(m.replace("PPAN", "SPINLOCK\n0 30\nPPAN"), host_spin="keep")


def test_supercon_is_refused_for_2d_hosts():
    host, _ = de.classify_host(SLAB_HOST)
    with pytest.raises(DefectError, match="SUPERCON"):
        de.supercell_records(host.dimensionality, "SUPERCON", [[2, 0], [0, 2]])
    slabcut = mgo_host().replace("BASISSET", "SLABCUT\n0 0 1\n1 3\nBASISSET")
    h2, _ = de.classify_host(slabcut)
    assert h2.dimensionality == 2 and h2.slabcut
    with pytest.raises(DefectError, match="SUPERCON"):
        de.supercell_records(2, "SUPERCON", [[2, 0], [0, 2]])
    with pytest.raises(DefectError, match="2x2"):
        de.supercell_records(2, "SUPERCEL", MGO_222)
    with pytest.raises(DefectError, match="singular"):
        de.supercell_records(3, "SUPERCEL", [[1, 0, 0], [1, 0, 0], [0, 0, 1]])


def test_slabcut_host_gets_its_supercell_after_the_slabcut_records():
    slabcut = mgo_host().replace("BASISSET", "SLABCUT\n0 0 1\n1 3\nBASISSET")
    decks, _, _ = _build(slabcut, de.DefectSite("vacancy", "O", label=1), keyword="SUPERCEL",
                         matrix=[[2, 0], [0, 2]], charges=(0,), name="s")
    assert "SLABCUT\n0 0 1\n1 3\nSUPERCEL\n2 0\n0 2\nATOMREMO\n1\n1\nBASISSET" in decks["s_q0"].text


def test_spin_state_parity_and_override():
    with pytest.raises(DefectError, match="NSPIN must be odd"):
        _build(mgo_host(), de.DefectSite("vacancy", "O", label=1), charges=(1,), spin_states={1: 2})
    decks, msgs, _ = _build(mgo_host(), de.DefectSite("vacancy", "O", label=1), charges=(0,),
                            spin_states={0: 2})
    assert "SPINLOCK\n2 30\nPPAN" in decks["mgo_q0"].text      # a triplet asked for explicitly
    decks, _, _ = _build(mgo_host(), de.DefectSite("vacancy", "O", label=1), charges=(1,),
                         spinlock_cycles=50)
    assert "SPINLOCK\n1 50\n" in decks["mgo_qp1"].text


def test_host_spin_keep_needs_a_spin_state_for_every_charge():
    host = mgo_host().replace("PPAN", "SPINLOCK\n0 40\nPPAN").replace("DFT\nPBE0", "DFT\nSPIN\nPBE0")
    with pytest.raises(DefectError, match="--spin-state for every charge"):
        _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0, 1), host_spin="keep",
               spin_states={1: 1})
    decks, _, _ = _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0, 1),
                         host_spin="keep", spin_states={0: 0, 1: 1})
    assert decks["mgo_qp1"].text.count("SPINLOCK") == 1
    assert "SPINLOCK\n1 40\n" in decks["mgo_qp1"].text


def test_missing_element_in_an_explicit_basis_is_refused():
    explicit = _explicit_host()
    with pytest.raises(DefectError, match="F is not in the host's explicit basis"):
        _build(explicit, de.DefectSite("substitution", "O", new_element="F", label=1), charges=(0,))


def test_missing_element_in_an_internal_library_is_refused():
    # POB-TZVP-REV2 lacks the noble gases (measured coverage, d12_constants)
    with pytest.raises(DefectError, match="no Ar"):
        _build(mgo_host(), de.DefectSite("substitution", "O", new_element="Ar", label=1), charges=(0,))


# --- 12-13: spin hosts ----------------------------------------------------------

def test_a_spin_host_is_kept_and_only_odd_states_get_spinlock():
    host = mgo_host().replace("DFT\nPBE0", "DFT\nSPIN\nPBE0")
    decks, msgs, _ = _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0, 1, 2))
    for name in ("mgo_q0", "mgo_qp2"):
        ins, dele, rep = edits(host, decks[name].text)
        assert not dele and not rep
        assert "SPINLOCK" not in ins and "SPIN" not in ins
    ins, _, _ = edits(host, decks["mgo_qp1"].text)
    assert "SPINLOCK" in ins and "SPIN" not in ins
    assert decks["mgo_qp1"].text.count("SPIN\n") == 1
    assert any("host SPIN kept" in m for m in msgs)


def test_hf_host_goes_open_shell():
    rhf = mgo_host().replace("DFT\nPBE0\nEND\n", "RHF\n")
    decks, msgs, _ = _build(rhf, de.DefectSite("vacancy", "O", label=1), charges=(1,))
    assert "RHF" not in decks["mgo_qp1"].text and "UHF\n" in decks["mgo_qp1"].text
    plain = mgo_host().replace("DFT\nPBE0\nEND\n", "")
    decks, msgs, _ = _build(plain, de.DefectSite("vacancy", "O", label=1), charges=(1,))
    assert "BASISSET\nPOB-TZVP-REV2\nUHF\nTOLINTEG" in decks["mgo_qp1"].text
    assert any("added UHF" in m for m in msgs)


# --- 15: PPAN --------------------------------------------------------------------

def test_ppan_is_added_to_every_deck_when_the_host_has_none():
    host = mgo_host().replace("PPAN\n", "")
    decks, msgs, added = _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0, 1))
    assert added == ["PPAN"]
    for d in decks.values():
        assert d.text.rstrip().endswith("PPAN\nEND")
    assert "DOPING\n-1.0\nSPINLOCK\n1 30\nPPAN\nEND" in decks["mgo_qp1"].text
    assert any("Mulliken" in m for m in msgs)


def test_shrink_replacement_handles_both_forms():
    host = mgo_host()
    decks, _, added = _build(host, de.DefectSite("vacancy", "O", label=1), charges=(0,), shrink=[2])
    assert "SHRINK\n2 4\nTOLDEE" in decks["mgo_pris"].text and "SHRINK 2" in added[0]
    directional = host.replace("SHRINK\n4 4\n", "SHRINK\n0 8\n4 4 4\n")
    decks, _, _ = _build(directional, de.DefectSite("vacancy", "O", label=1), charges=(0,), shrink=[2, 2])
    assert "SHRINK\n2 2\nTOLDEE" in decks["mgo_q0"].text and "4 4 4" not in decks["mgo_q0"].text
    h, _ = de.classify_host(host)
    assert de.suggest_shrink(h, MGO_222) == "2"


def test_block1_terminator_and_atom_count_cross_check():
    lines = mgo_host().splitlines()
    assert lines[de.block1_insert_index(lines)] == "BASISSET"
    bad = mgo_host().replace("\n2\n8  0.0", "\n3\n8  0.0")
    with pytest.raises(DefectError):
        de.classify_host(bad)
    # an IFSO > 1 origin-shift record is skipped, not read as the lattice
    shifted = mgo_host().replace("0 0 0\n225\n", "0 0 2\n225\n0 0 0\n")
    h, _ = de.classify_host(shifted)
    assert h.lattice_params == [4.212] and len(h.atoms) == 2


# --- helpers -----------------------------------------------------------------------

def _explicit_host():
    """The measured explicit-basis MgO deck (mgo_g1x) with its defect records taken out."""
    text = deck("mgo_g1x")
    text = strip_records(text, "SUPERCON", 3)
    text = strip_records(text, "GHOSTS", 2)
    text = strip_records(text, "CHEMOD", 2)
    text = strip_records(text, "CHARGED", 0)
    text = strip_records(text, "SPINLOCK", 1)
    return text.replace("DFT\nSPIN\nPBE0", "DFT\nPBE0")


# --- 14: decks written by MACE's own writer --------------------------------------------

MGO_CIF = """data_MgO
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
_cell_length_a   4.212
_cell_length_b   4.212
_cell_length_c   4.212
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1 O 0.0 0.0 0.0 1
Mg1 Mg 0.5 0.5 0.5 1
"""


def _convert(tmp_path, **changes):
    """A host deck from `mace convert --batch` (NewCifToD12), spin-polarised SP as MACE writes it."""
    pytest.importorskip("ase")
    from d12_constants import scf_tolerances
    opts = {"symmetry_handling": "CIF", "write_only_unique": True, "dimensionality": "CRYSTAL",
            "calculation_type": "SP", "method": "DFT", "dft_functional": "PBE0",
            "use_dispersion": False, "basis_set_type": "INTERNAL", "basis_set": "POB-TZVP-REV2",
            "dft_grid": "XLGRID", "is_spin_polarized": True, "tolerances": scf_tolerances("2"),
            "scf_method": "DIIS", "scf_maxcycle": 800, "fmixing": 30}
    opts.update(changes)
    work = tmp_path / ("conv_" + str(abs(hash(json.dumps(changes, sort_keys=True)))))
    (work / "cifs").mkdir(parents=True)
    (work / "cifs" / "MgO.cif").write_text(MGO_CIF)
    (work / "opts.json").write_text(json.dumps(opts))
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "convert", "--no-banner", "--batch",
                        "--options_file", str(work / "opts.json"), "--cif_dir", str(work / "cifs"),
                        "--output_dir", str(work / "out")], cwd=work, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    (deck_path,) = list((work / "out").glob("*.d12"))
    return deck_path.read_text()


ALLOWED_RECORDS = {"SUPERCON", "SUPERCEL", "2 0 0", "0 2 0", "0 0 2", "2 0", "0 2", "ATOMREMO", "ATOMSUBS",
                   "1", "1 7", "DOPING", "-1.0", "-2.0", "1.0", "SPINLOCK", "1 30", "SPIN", "UHF",
                   "GHOSTS", "-0.5", "-1.5"}


@pytest.mark.parametrize("variant", ["internal", "explicit", "slabcut", "3c", "hf"])
def test_real_writer_hosts_keep_every_host_line(tmp_path, variant):
    changes = {
        "internal": {},
        "explicit": {"basis_set_type": "EXTERNAL", "basis_set": "full.basis.triplezeta"},
        "slabcut": {"slabcut": {"miller": [0, 0, 1], "isup": 1, "nlayers": 3}},
        "3c": {"dft_functional": "HSESOL3C", "basis_set": "SOLDEF2MSVP"},
        "hf": {"method": "HF", "hf_method": "RHF", "is_spin_polarized": False},
    }[variant]
    host_text = _convert(tmp_path, **changes)
    if variant == "slabcut":
        kw, mat, site, charges = "SUPERCEL", [[2, 0], [0, 2]], de.DefectSite("vacancy", "O", label=1), (0, 1)
    elif variant == "3c":
        kw, mat, site, charges = "SUPERCON", MGO_222, de.DefectSite("substitution", "O", new_element="N",
                                                                   label=1), (0, -1)
    else:
        kw, mat, site, charges = "SUPERCON", MGO_222, de.DefectSite("vacancy", "O", label=1), (0, 1, 2)
    decks, msgs, added = _build(host_text, site, keyword=kw, matrix=mat, charges=charges)
    assert added == []                       # MACE decks already carry PPAN
    for d in decks.values():
        ins, dele, rep = edits(host_text, d.text)
        assert not rep, (d.name, rep)
        assert not dele, (d.name, dele)
        assert set(ins) <= ALLOWED_RECORDS, (d.name, set(ins) - ALLOWED_RECORDS)
    if variant == "hf":
        assert "UHF" in decks["mgo_qp1"].text
    if variant == "slabcut":
        assert "SLABCUT\n0 0 1\n1 3\nSUPERCEL\n2 0\n0 2\n" in decks["mgo_pris"].text
    if variant == "3c":
        assert "SPINLOCK\n1 30" in decks["mgo_q0"].text and "SPINLOCK" not in decks["mgo_qm1"].text
