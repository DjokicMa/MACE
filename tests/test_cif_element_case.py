"""All-caps (and lower-case) element symbols in CIFs: "BI" is bismuth.

ase.io.cif reads the element with "[A-Z][a-z]?", so a CIF that writes
bismuth as "BI" next to boron "B" came out as two borons. Every MACE path that
reads a CIF goes through mace.utils.cif_symbols, checked here.
"""

from pathlib import Path

import pytest

pytest.importorskip("ase", reason="CIF parsing needs ase")

from mace.utils.cif_symbols import element_symbol, read_cif  # noqa: E402

HEAD = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
loop_
"""
SITES = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]


def _cif(tmp_path: Path, rows, with_type=True, name="t.cif") -> Path:
    cols = ["_atom_site_label"] + (["_atom_site_type_symbol"] if with_type else []) + [
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    lines = [HEAD + "\n".join(cols)]
    for row, xyz in zip(rows, SITES):
        lines.append(" ".join(list(row) + [str(c) for c in xyz]))
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.mark.parametrize("token,want", [
    ("BI", "Bi"), ("Bi", "Bi"), ("bi", "Bi"), ("bI", "Bi"), ("BI1", "Bi"), ("Bi1a", "Bi"),
    ("BI3+", "Bi"), ("Bi3+", "Bi"), ("B", "B"), ("B2", "B"), ("b", "B"),
    ("CL", "Cl"), ("CL1", "Cl"), ("C", "C"), ("SN", "Sn"), ("SN4+", "Sn"), ("S", "S"),
    ("O2-", "O"), ("O", "O"), ("Ow", "O"), ("HA1", "H"), ("CO1", "Co"), ("D", "D"),
    ("FE", "Fe"), ("NB", "Nb"), ("XYZ1", None), ("123", None), ("", None),
])
def test_element_symbol(token, want):
    assert element_symbol(token) == want


ROWS = [("BI1", "BI"), ("B2", "B"), ("CL1", "CL"), ("SN1", "SN4+")]


def test_type_symbol_all_caps(tmp_path):
    assert read_cif(_cif(tmp_path, ROWS)).get_chemical_symbols() == ["Bi", "B", "Cl", "Sn"]


def test_type_symbol_lower_and_mixed_case(tmp_path):
    rows = [("x1", "bi"), ("x2", "b"), ("x3", "Bi"), ("x4", "cl")]
    assert read_cif(_cif(tmp_path, rows)).get_chemical_symbols() == ["Bi", "B", "Bi", "Cl"]


def test_labels_only_all_caps(tmp_path):
    rows = [("BI1",), ("B2",), ("CL1",), ("SN1",)]
    atoms = read_cif(_cif(tmp_path, rows, with_type=False))
    assert atoms.get_chemical_symbols() == ["Bi", "B", "Cl", "Sn"]


def test_labels_only_charged_and_suffixed(tmp_path):
    rows = [("Bi1a",), ("O2-",), ("Bi3+",), ("B1",)]
    atoms = read_cif(_cif(tmp_path, rows, with_type=False))
    assert atoms.get_chemical_symbols() == ["Bi", "O", "Bi", "B"]


def test_type_symbol_beats_ambiguous_label(tmp_path):
    """Label "CO1" alone reads as cobalt; a type symbol "C" says carbon."""
    rows = [("CO1", "C"), ("CO2", "CO"), ("CA1", "C"), ("O1", "O")]
    atoms = read_cif(_cif(tmp_path, rows))
    assert atoms.get_chemical_symbols() == ["C", "Co", "C", "O"]


def test_proper_case_unchanged(tmp_path):
    """A CIF ase already reads right gives exactly ase's atoms."""
    import ase.io
    rows = [("Bi1", "Bi3+"), ("B1", "B"), ("Cl1", "Cl-"), ("Sn1", "Sn")]
    p = _cif(tmp_path, rows)
    mine, theirs = read_cif(p), ase.io.read(str(p), format="cif")
    assert mine.get_chemical_symbols() == theirs.get_chemical_symbols() == ["Bi", "B", "Cl", "Sn"]
    assert (mine.get_scaled_positions() == theirs.get_scaled_positions()).all()
    assert mine.info["spacegroup"].no == theirs.info["spacegroup"].no


def test_deuterium_kept(tmp_path):
    rows = [("D1", "D"), ("O1", "O"), ("BI1", "BI"), ("B1", "B")]
    atoms = read_cif(_cif(tmp_path, rows))
    assert atoms.get_chemical_symbols() == ["H", "O", "Bi", "B"]
    assert atoms.get_masses()[0] > 2.0


def test_cif2d12_parse_cif(tmp_path):
    """The `mace convert` / cif2d12 path: Z=83 for BI, not boron's 5."""
    import NewCifToD12
    data = NewCifToD12.parse_cif(str(_cif(tmp_path, ROWS)), interactive=False)
    assert list(data["atomic_numbers"]) == [83, 5, 17, 50]
    assert list(data["symbols"]) == ["Bi", "B", "Cl", "Sn"]


def test_cif2d12_parse_cif_labels_only(tmp_path):
    import NewCifToD12
    rows = [("BI1",), ("B2",), ("CL1",), ("SN1",)]
    data = NewCifToD12.parse_cif(str(_cif(tmp_path, rows, with_type=False)), interactive=False)
    assert list(data["atomic_numbers"]) == [83, 5, 17, 50]


def test_gui_read_cif(tmp_path):
    """The GUI's CIF reader, both its P1 fast path and the ase path."""
    from mace.gui import structures
    p = _cif(tmp_path, ROWS)
    assert list(structures.read_cif(p)["numbers"]) == [83, 5, 17, 50]
    assert list(structures.read_any(p, "cif")["numbers"]) == [83, 5, 17, 50]
    sym = p.read_text().replace("_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n"
                                "loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n",
                                "_symmetry_space_group_name_H-M 'P -1'\n_symmetry_Int_Tables_number 2\n")
    p2 = tmp_path / "pm1.cif"
    p2.write_text(sym)
    nums = list(structures.read_cif(p2)["numbers"])  # P-1: the ase path, not the P1 fast path
    assert {83, 5, 17} <= set(nums) and 6 not in nums


def test_preopt_read_input(tmp_path):
    from mace.preopt.cli import read_input
    atoms, _, _, _ = read_input(_cif(tmp_path, ROWS))
    assert atoms.get_chemical_symbols() == ["Bi", "B", "Cl", "Sn"]


# --------------------------------------------------------------- POSCAR

POSCAR = """{comment}
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
{species}
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


@pytest.mark.parametrize("species,want", [
    ("BI S", ["Bi", "S"]), ("bi s", ["Bi", "S"]), ("B BI", ["B", "Bi"]),
    ("Bi S", ["Bi", "S"]), ("CL SN ! comment", ["Cl", "Sn"]),
])
def test_gui_reads_poscar_species_any_case(tmp_path, species, want):
    from ase.data import chemical_symbols
    from mace.gui import structures
    p = tmp_path / "POSCAR"
    p.write_text(POSCAR.format(comment="test", species=species))
    got = structures.read_any(p, "poscar")
    assert [chemical_symbols[n] for n in got["numbers"]] == want
    assert [chemical_symbols[n] for n in structures.read_structure(p)["numbers"]] == want


def test_poscar_normalize_leaves_correct_files_alone():
    from mace.utils.cif_symbols import normalize_poscar_species
    good = POSCAR.format(comment="BI S", species="Bi S")
    assert normalize_poscar_species(good) == good
    vasp4 = POSCAR.format(comment="Bi S", species="1 1").replace("1 1\nDirect", "Direct", 1)
    assert normalize_poscar_species(vasp4) == vasp4
    fixed = normalize_poscar_species(POSCAR.format(comment="BI S", species="BI S"))
    assert fixed.splitlines()[0] == "BI S" and fixed.splitlines()[5].split() == ["Bi", "S"]


# ------------------------------------------- cif2d12's manual fallback


def test_cif2d12_manual_fallback(tmp_path, monkeypatch):
    """When ase cannot read the CIF (here an H-M name it does not know),
    parse_cif falls back to its own reader; that reader used an undefined
    `Element` and stopped with NameError."""
    import NewCifToD12
    text = _cif(tmp_path, ROWS).read_text()
    text = text.replace("'P 1'", "'X 1'").replace("_cell_length_a 5.0", "_cell_length_a 5.0(1)")
    text = text.replace("loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n", "")
    text = text.replace("_symmetry_Int_Tables_number 1\n", "")
    p = tmp_path / "bad_hm.cif"
    p.write_text(text)
    data = NewCifToD12.parse_cif(str(p), interactive=False)
    assert data["atomic_numbers"] == [83, 5, 17, 50]
    assert data["symbols"] == ["Bi", "B", "Cl", "Sn"]
    assert data["spacegroup"] is None          # no number in the file
    assert (data["a"], data["b"], data["gamma"]) == (5.0, 5.0, 90.0)
    assert data["positions"][1] == [0.5, 0.5, 0.5]
    assert data["cif_atom_symbols"] == data["symbols"]


def test_cif2d12_raw_records_first_letter_run(tmp_path):
    """A label-only "C1A" is carbon: the letters after the digit are a suffix."""
    import NewCifToD12
    rows = [("C1A",), ("BI1",), ("O2-",), ("Bi1a",)]
    syms, _ = NewCifToD12._parse_cif_atom_site_records(str(_cif(tmp_path, rows, with_type=False)))
    assert syms == ["C", "Bi", "O", "Bi"]
