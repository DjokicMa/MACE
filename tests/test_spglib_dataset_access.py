"""spglib's dataset is read by attribute, with the same symmetry results.

spglib 2.5 made get_symmetry_dataset return a SpglibDataset and deprecated
dict-style access (dataset["rotations"]); older spglib, which
requirements.txt still allows, returns a plain dict. NewCifToD12 now wraps
the dataset with spglib_compat.attribute_dataset and reads attributes only.

These pin, on four known structures, what the asymmetric-unit reduction
reports: the space group, the number of symmetry operations and the reduced
atom list. The same values must come out whether spglib hands back a
SpglibDataset or an old-style dict.
"""
import sys
import warnings

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from spglib_compat import attribute_dataset  # noqa: E402


def test_old_spglib_dict_gets_attribute_access():
    ds = attribute_dataset({"number": 227, "rotations": [1, 2],
                            "equivalent_atoms": [0, 0]})
    assert ds.number == 227
    assert ds.rotations == [1, 2]
    assert ds.equivalent_atoms == [0, 0]


def test_failed_analysis_stays_none():
    assert attribute_dataset(None) is None


def test_non_dict_dataset_is_returned_unchanged():
    marker = object()
    assert attribute_dataset(marker) is marker


# --- the reduction itself, on real spglib ------------------------------------

def _cell(a, b, c, alpha, beta, gamma, sg, atoms):
    symbols = [s for s, _ in atoms]
    numbers = {"C": 6, "Na": 11, "Cl": 17, "Zn": 30, "O": 8, "Si": 14, "H": 1}
    return {
        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma,
        "spacegroup": sg,
        "symbols": symbols,
        "atomic_numbers": [numbers[s] for s in symbols],
        "positions": [list(p) for _, p in atoms],
    }


_FCC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]

DIAMOND = _cell(3.567, 3.567, 3.567, 90, 90, 90, 227,
                [("C", p) for p in _FCC]
                + [("C", (x + 0.25, y + 0.25, z + 0.25)) for x, y, z in _FCC])

ROCK_SALT = _cell(5.64, 5.64, 5.64, 90, 90, 90, 225,
                  [("Na", p) for p in _FCC]
                  + [("Cl", ((x + 0.5) % 1, y, z)) for x, y, z in _FCC])

P1_CELL = _cell(5.1, 6.3, 7.2, 81.0, 95.0, 103.0, 1,
                [("Si", (0.11, 0.23, 0.37)), ("O", (0.52, 0.18, 0.71)),
                 ("H", (0.83, 0.64, 0.29))])

_U = 0.382
WURTZITE = _cell(3.25, 3.25, 5.207, 90, 90, 120, 186,
                 [("Zn", (1 / 3, 2 / 3, 0.0)), ("Zn", (2 / 3, 1 / 3, 0.5)),
                  ("O", (1 / 3, 2 / 3, _U)), ("O", (2 / 3, 1 / 3, 0.5 + _U))])

# (structure, space group, symmetry operations, reduced symbols, reduced positions)
CASES = {
    "diamond_Fd-3m": (DIAMOND, 227, 192, ["C"], [(0, 0, 0)]),
    "rock_salt_Fm-3m": (ROCK_SALT, 225, 192, ["Na", "Cl"],
                        [(0, 0, 0), (0.5, 0, 0)]),
    "triclinic_P1": (P1_CELL, 1, 1, ["Si", "O", "H"],
                     [(0.11, 0.23, 0.37), (0.52, 0.18, 0.71), (0.83, 0.64, 0.29)]),
    "wurtzite_P6_3mc": (WURTZITE, 186, 12, ["Zn", "O"],
                        [(1 / 3, 2 / 3, 0.0), (1 / 3, 2 / 3, _U)]),
}


@pytest.fixture
def new_cif():
    pytest.importorskip("ase", reason="NewCifToD12 needs ase (absent in CI)")
    pytest.importorskip("spglib", reason="symmetry reduction needs spglib (absent in CI)")
    import NewCifToD12
    return NewCifToD12


def _reduce(module, structure):
    """Run the reduction on a copy and return (result, reported op count)."""
    reported = []
    real_print = module.ui.print

    def record(msg="", *a, **k):
        text = str(msg)
        if "Symmetry operations:" in text:
            reported.append(int(text.rsplit(":", 1)[1]))
        return real_print(msg, *a, **k)

    module.ui.print = record
    try:
        result = module.verify_and_reduce_to_asymmetric_unit(
            {k: (list(v) if isinstance(v, list) else v) for k, v in structure.items()})
    finally:
        module.ui.print = real_print
    return result, reported


def _check(result, reported, sg, n_ops, symbols, positions):
    assert result["spacegroup"] == sg
    assert reported == [n_ops]
    assert result["symbols"] == symbols
    assert len(result["positions"]) == len(positions)
    for got, want in zip(result["positions"], positions):
        assert got == pytest.approx(list(want), abs=1e-6)


@pytest.mark.parametrize("name", list(CASES))
def test_reduction_results_are_unchanged(new_cif, name):
    structure, sg, n_ops, symbols, positions = CASES[name]
    with warnings.catch_warnings():
        # The dict interface is gone from NewCifToD12: any use of it is an error.
        warnings.filterwarnings("error", message="dict interface is deprecated")
        result, reported = _reduce(new_cif, structure)
    _check(result, reported, sg, n_ops, symbols, positions)


@pytest.mark.parametrize("name", list(CASES))
def test_old_spglib_dict_dataset_gives_the_same_results(new_cif, monkeypatch, name):
    """Simulate spglib < 2.5, whose get_symmetry_dataset returns a dict."""
    real = new_cif.spglib.get_symmetry_dataset

    def as_dict(*a, **k):
        ds = real(*a, **k)
        if ds is None or isinstance(ds, dict):
            return ds
        return {f: getattr(ds, f) for f in ds.__dataclass_fields__}

    monkeypatch.setattr(new_cif.spglib, "get_symmetry_dataset", as_dict)
    structure, sg, n_ops, symbols, positions = CASES[name]
    result, reported = _reduce(new_cif, structure)
    _check(result, reported, sg, n_ops, symbols, positions)
