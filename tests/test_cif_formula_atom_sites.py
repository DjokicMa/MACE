"""formula_extractor.extract_formula_from_cif without _chemical_formula_sum
reads the atom-site loop (bug hunt 1.1.5, #3 note).

The fallback regex ``_atom_site_type_symbol\\s+(\\w+)`` matched the header
and took the NEXT loop tag as the element, so a CIF with no formula line got
the formula "_atom_site_fract_x"."""
from mace.utils.formula_extractor import extract_formula_from_cif

CIF = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 Ti4+ 0.0 0.0 0.0
Pb1 Pb 0.5 0.5 0.5
O1 O2- 0.5 0.0 0.0
O2 O 0.0 0.5 0.0
O3 O 0.0 0.0 0.5
"""


def test_atom_site_loop_gives_the_elements(tmp_path):
    p = tmp_path / "x.cif"
    p.write_text(CIF)
    assert extract_formula_from_cif(p) == "TiPbO3"


def test_label_column_when_there_is_no_type_symbol(tmp_path):
    p = tmp_path / "y.cif"
    p.write_text(CIF.replace("_atom_site_type_symbol\n", "").replace(" Ti4+ ", " ").replace(" Pb ", " ")
                 .replace(" O2- ", " ").replace("O2 O ", "O2 ").replace("O3 O ", "O3 "))
    assert extract_formula_from_cif(p) == "TiPbO3"
