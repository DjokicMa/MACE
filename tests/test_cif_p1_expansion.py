"""A P1 deck from a CIF holds the cell the symmetrised deck describes.

A CIF without a symmetry-operator loop names its group by number only, and
ASE then expands the atom records in origin choice 1. Diamond given in origin
choice 2, C at (1/8, 1/8, 1/8) - CRYSTAL's own diamond example (manual p. 373,
"0 0 0" deck) and what the symmetrised deck declares - became 16 atoms at
Fd-3m 16c, a different structure with C-C contacts of 1.26 A instead of
diamond's 8 atoms at 1.545 A.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase", reason="CIF parsing needs ase")
import NewCifToD12  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES  # noqa: E402

DATA = Path(__file__).parent / "data"

CIF = """data_diamond
_symmetry_space_group_name_H-M    'F d -3 m'
_symmetry_Int_Tables_number       227
_cell_length_a                    3.5670
_cell_length_b                    3.5670
_cell_length_c                    3.5670
_cell_angle_alpha                 90.0000
_cell_angle_beta                  90.0000
_cell_angle_gamma                 90.0000
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1   C     {x}   {x}   {x}   1.00
"""


def _options(symmetry_handling):
    return dict(dimensionality="CRYSTAL", calculation_type="SP",
                basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2", method="DFT",
                dft_functional="PBE", is_spin_polarized=False,
                tolerances=DEFAULT_TOLERANCES, scf_method="DIIS",
                symmetry_handling=symmetry_handling)


def _p1_atoms(tmp_path, cif_text=None, cif_file=None):
    src = tmp_path / "src"
    src.mkdir()
    if cif_file:
        shutil.copy(cif_file, src)
    else:
        (src / "dia.cif").write_text(cif_text)
    out = tmp_path / "out"
    written, found = NewCifToD12.process_cifs(str(src), _options("P1"), str(out),
                                              interactive=False)
    assert (written, found) == (1, 1)
    lines = next(out.glob("*.d12")).read_text().splitlines()
    assert lines[3] == "1"
    a = float(lines[4].split()[0])
    n = int(lines[5])
    frac = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[6:6 + n]])
    return a, frac


def _shortest_contact(a, frac):
    best = np.inf
    for i in range(len(frac)):
        d = frac - frac[i]
        d -= np.round(d)
        r = np.linalg.norm(d * a, axis=1)
        r[i] = np.inf
        best = min(best, r.min())
    return best


@pytest.mark.parametrize("x", ["0.12500", "0.00000"])
def test_diamond_without_symops_is_eight_atoms(tmp_path, x):
    a, frac = _p1_atoms(tmp_path, CIF.format(x=x))
    assert len(frac) == 8
    assert _shortest_contact(a, frac) == pytest.approx(a * np.sqrt(3) / 4, abs=1e-6)


def test_diamond_with_symops_is_unchanged(tmp_path):
    a, frac = _p1_atoms(tmp_path, cif_file=DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif")
    assert len(frac) == 8
    assert _shortest_contact(a, frac) == pytest.approx(a * np.sqrt(3) / 4, abs=1e-6)
