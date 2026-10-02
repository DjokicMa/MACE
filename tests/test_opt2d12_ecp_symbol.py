"""The geometry comment of an ECP atom in an opt2d12 deck names its element.

opt2d12 writes each atom as 'NAT x y z Biso 1.000000 <symbol>'. CRYSTAL reads
only NAT and the coordinates on these lines; the rest is a comment. An
EXTERNAL-basis parent keeps its conventional atom numbers (282 for Pb with an
ECP, manual p. 22), and the symbol was looked up with that number, so every
ECP atom was labelled "X".
"""
import sys

import pytest

import CRYSTALOptToD12 as M

TE_PB_BASIS = ["282 1", "HAYWSC", "0 0 1 2.0 1.0", "1.0 1.0",
               "252 1", "HAYWSC", "0 0 1 2.0 1.0", "1.0 1.0"]


class _NoTTY:
    def isatty(self):
        return False


def _settings(dimensionality):
    cell = [6.462, 6.462, 6.462, 90.0, 90.0, 90.0]
    geometry = {
        "conventional_cell": cell,
        "coordinates": [{"atom_number": "282", "x": "0.0", "y": "0.0", "z": "0.0"},
                        {"atom_number": "252", "x": "0.5", "y": "0.5", "z": "1.5"},
                        {"atom_number": "8", "x": "0.25", "y": "0.25", "z": "0.5"}],
    }
    settings = {
        "dimensionality": dimensionality, "spacegroup": 1, "origin_setting": "0 0 0",
        "calculation_type": "SP", "functional": "PBE0", "dispersion": False,
        "dft_grid": "XLGRID", "k_points": "8 8 8" if dimensionality == "CRYSTAL" else "8 8 1",
        "basis_set_type": "EXTERNAL", "basis_set": M.PARENT_BASIS_MARKER,
        "use_original_external_basis": True,
    }
    return geometry, settings


@pytest.mark.parametrize("dimensionality", ["CRYSTAL", "SLAB"])
def test_ecp_atoms_are_labelled_with_their_element(tmp_path, monkeypatch, dimensionality):
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    geometry, settings = _settings(dimensionality)
    out = tmp_path / "child.d12"
    assert M.write_d12_file(str(out), geometry, settings, TE_PB_BASIS) is True
    atoms = [ln.split() for ln in out.read_text().split("\n") if " Biso " in ln]
    # atom numbers and coordinates as before; the comment names the element
    assert [a[0] for a in atoms] == ["282", "252", "8"]
    assert [a[-1] for a in atoms] == ["Pb", "Te", "O"]
