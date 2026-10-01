"""Choosing spglib's space group writes spglib's cell and atoms with it.

When the CIF's group and spglib's disagree, option 2 of
verify_and_reduce_to_asymmetric_unit ("Proceed with spglib space group")
declared spglib's group number but kept the CIF's cell and atoms. CRYSTAL
applies that group's operators in their standard setting, so a cell in any
other setting described a different crystal. Here rock salt and diamond are
given in their primitive rhombohedral cells (a = b = c, 60 degrees) as P1:
spglib finds Fm-3m and Fd-3m, whose decks must carry the conventional cube.
"""
import numpy as np
import pytest

spglib = pytest.importorskip("spglib")
pytest.importorskip("ase", reason="NewCifToD12 imports ase")
import NewCifToD12  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES  # noqa: E402


def _primitive_fcc(a_conv, symbols, numbers, positions):
    a = a_conv / np.sqrt(2)
    return {
        "a": a, "b": a, "c": a, "alpha": 60.0, "beta": 60.0, "gamma": 60.0,
        "spacegroup": 1, "symbols": symbols, "atomic_numbers": numbers,
        "positions": positions, "name": "prim",
    }


def _choose_spglib(monkeypatch, cif_data):
    monkeypatch.setattr(NewCifToD12, "get_user_input", lambda *a, **k: "2")
    return NewCifToD12.verify_and_reduce_to_asymmetric_unit(
        cif_data, 1e-5, interactive=True)


def _expand(result):
    """Every atom the written group generates from the written atoms."""
    from ase.spacegroup import crystal
    setting = 2 if result["spacegroup"] in NewCifToD12.MULTI_ORIGIN_SPACEGROUPS else 1
    return crystal(result["symbols"], result["positions"],
                   spacegroup=result["spacegroup"], setting=setting,
                   cellpar=[result[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")])


def test_rock_salt_gets_the_conventional_cube(monkeypatch):
    result = _choose_spglib(monkeypatch, _primitive_fcc(
        5.64, ["Na", "Cl"], [11, 17], [[0, 0, 0], [0.5, 0.5, 0.5]]))
    assert result["spacegroup"] == 225
    assert (result["a"], result["b"], result["c"]) == pytest.approx((5.64,) * 3)
    assert (result["alpha"], result["beta"], result["gamma"]) == pytest.approx((90,) * 3)
    atoms = _expand(result)
    assert len(atoms) == 8
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(2.82)


def test_diamond_deck_matches_its_origin(monkeypatch, tmp_path):
    result = _choose_spglib(monkeypatch, _primitive_fcc(
        3.567, ["C", "C"], [6, 6], [[0, 0, 0], [0.25, 0.25, 0.25]]))
    assert result["spacegroup"] == 227
    assert len(result["positions"]) == 1
    atoms = _expand(result)
    assert len(atoms) == 8
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(3.567 * np.sqrt(3) / 4)

    out = tmp_path / "dia.d12"
    options = dict(dimensionality="CRYSTAL", calculation_type="SP",
                   basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2", method="DFT",
                   dft_functional="PBE", is_spin_polarized=False,
                   tolerances=DEFAULT_TOLERANCES, scf_method="DIIS",
                   symmetry_handling="SPGLIB")
    assert NewCifToD12.create_d12_file(result, str(out), options, interactive=False)
    lines = out.read_text().splitlines()
    # "0 0 0" is origin choice 2, where diamond's atoms sit on odd eighths
    # (8a, as in the manual's example on p. 373, or the equivalent 8b)
    assert lines[2:4] == ["0 0 0", "227"]
    assert float(lines[4].split()[0]) == pytest.approx(3.567)
    assert lines[5] == "1"
    coords = [float(v) for v in lines[6].split()[1:4]]
    assert all(abs(v * 8 - round(v * 8)) < 1e-6 and round(v * 8) % 2 == 1 for v in coords)
