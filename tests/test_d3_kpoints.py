"""Regression test for crystal-d3#F2: a duplicate 'orthorhombic_bc' key in
normalize_lattice_type's centering map silently overrode the body-centered ('I')
mapping with C-centered ('C'). The fix removes the stray duplicate."""
import sys
from pathlib import Path

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

from d3_kpoints import normalize_lattice_type


def test_body_centered_orthorhombic_maps_to_I():
    assert normalize_lattice_type("orthorhombic_bc") == "I"


def test_c_centered_orthorhombic_variants_still_C():
    assert normalize_lattice_type("orthorhombic_ab") == "C"
    assert normalize_lattice_type("orthorhombic_ac") == "C"


def test_bare_centering_letters_passthrough():
    assert normalize_lattice_type("I") == "I"
    assert normalize_lattice_type("P") == "P"
