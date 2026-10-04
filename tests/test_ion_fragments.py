"""Fragments by connectivity, the ion table, and the Mulliken parsers (mace/utils/ion_fragments).

The group's add_chemod_crystal.py assumed a fixed atom order (Li first, then
FSI and EC blocks) and silently skipped two clusters whose order differed.
Here fragments come from bonds, so atom order does not matter. All geometries
are synthetic (tests/ion_fixtures.py).
"""
import math

import pytest

from ion_fixtures import (IONS, acetate, benzene, carbonate, cluster, dmso, ec, fsi, hno3, hydroxide, li_near,
                          mulliken_block, otf, shuffled, so2f2, sulfate, water)
from mace.utils.ion_fragments import (COVALENT_RADII, ION_TABLE, IonChargeError, cell_matrix, classify,
                                      find_fragments, format_fragment_report, free_valence, mulliken_atoms,
                                      mulliken_electron_count, valence_charges)


def _split(atoms):
    return [s for s, _ in atoms], [p for _, p in atoms]


def _ions(frags, symbols):
    return sorted((f["ion"] or "neutral", f["formula"]) for f in frags)


# ------------------------------------------------------------------ tables

def test_ion_table_is_pinned():
    assert {k: (v["formula"], v["charge"], v.get("terminal")) for k, v in ION_TABLE.items()} == {
        "Li+": ({"Li": 1}, 1, None), "Na+": ({"Na": 1}, 1, None), "K+": ({"K": 1}, 1, None),
        "Mg2+": ({"Mg": 1}, 2, None), "Ca2+": ({"Ca": 1}, 2, None), "Zn2+": ({"Zn": 1}, 2, None),
        "FSI-": ({"F": 2, "N": 1, "O": 4, "S": 2}, -1, ("O", "S", 4)),
        "TFSI-": ({"C": 2, "F": 6, "N": 1, "O": 4, "S": 2}, -1, ("O", "S", 4)),
        "OTf-": ({"C": 1, "F": 3, "O": 3, "S": 1}, -1, ("O", "S", 3)),
        "ClO4-": ({"Cl": 1, "O": 4}, -1, ("O", "Cl", 4)),
        "NO3-": ({"N": 1, "O": 3}, -1, ("O", "N", 3)),
        "PF6-": ({"F": 6, "P": 1}, -1, ("F", "P", 6)),
        "BF4-": ({"B": 1, "F": 4}, -1, ("F", "B", 4)),
    }


def test_covalent_radii_are_cordero():
    for el, r in {"H": 0.31, "Li": 1.28, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05,
                  "Cl": 1.02, "P": 1.07, "B": 0.84, "Zn": 1.22, "Pb": 1.46}.items():
        assert COVALENT_RADII[el] == r


# ------------------------------------------------------------- molecules

def li_fsi_ec():
    anion = fsi()
    return cluster((anion, (0.0, 0.0, 0.0)), (ec(), (8.0, 0.0, 0.0))) + [li_near(anion, 2)]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_shuffled_cluster_gives_the_same_fragments(seed):
    syms, xyz = _split(shuffled(li_fsi_ec(), seed))
    frags = classify(syms, xyz)
    assert _ions(frags, syms) == [("FSI-", "F2NO4S2"), ("Li+", "Li"), ("neutral", "C3H4O3")]
    fsi_frag = next(f for f in frags if f["ion"] == "FSI-")
    assert sorted(syms[i] for i in fsi_frag["sites"]) == ["O"] * 4


def test_li_never_merges_with_the_oxygen_it_sits_on():
    atoms = water() + [("Li", (0.0, -1.95, 0.0))]
    syms, xyz = _split(atoms)
    assert math.dist(xyz[0], xyz[3]) == pytest.approx(1.95)
    frags = classify(syms, xyz)
    assert _ions(frags, syms) == [("Li+", "Li"), ("neutral", "H2O")]


@pytest.mark.parametrize("name", sorted(IONS))
def test_every_table_anion_is_recognized(name):
    syms, xyz = _split(shuffled(IONS[name]() + [("Li", (9.0, 9.0, 9.0))]))
    frags = classify(syms, xyz)
    anion = next(f for f in frags if f["ion"] != "Li+")
    assert anion["ion"] == name
    spec = ION_TABLE[name]
    assert len(anion["sites"]) == spec["terminal"][2]
    assert {syms[i] for i in anion["sites"]} == {spec["terminal"][0]}


@pytest.mark.parametrize("cation", ["Li", "Na", "K", "Mg", "Ca", "Zn"])
def test_every_table_cation_is_recognized(cation):
    syms, xyz = _split(water() + [(cation, (0.0, -2.1, 0.0))])
    frags = classify(syms, xyz)
    assert [f["ion"] for f in frags if f["formula"] != "H2O"] == [f"{cation}{'+' if cation in ('Li', 'Na', 'K') else '2+'}"]


@pytest.mark.parametrize("builder", [hno3, so2f2], ids=["HNO3", "SO2F2"])
def test_look_alike_neutral_molecules_stay_neutral(builder):
    syms, xyz = _split(builder())
    frags = classify(syms, xyz)
    assert [f["ion"] for f in frags] == [None]


def test_otf_sites_are_its_three_sulfonyl_oxygens():
    syms, xyz = _split(otf())
    (frag,) = classify(syms, xyz)
    assert frag["ion"] == "OTf-"
    assert all(syms[i] == "O" for i in frag["sites"]) and len(frag["sites"]) == 3


def test_report_lists_each_kind_once():
    syms, xyz = _split(li_fsi_ec() + [("Li", (30.0, 0.0, 0.0))])
    lines = format_fragment_report(classify(syms, xyz), syms)
    assert "2 x Li: Li+ (+1 each)" in lines
    assert "1 x F2NO4S2: FSI- (-1 each)" in lines
    assert "1 x C3H4O3: neutral" in lines


# --------------------------------------------------------------- periodic

def test_fsi_across_a_cell_boundary_is_one_ion():
    a = 12.0
    cell = cell_matrix(a, a, a, 90, 90, 90)
    # center the anion on the x = 0 face: half its atoms wrap to x ~ 12
    atoms = [(s, ((p[0]) % a, p[1] + 6.0, p[2] + 6.0)) for s, p in fsi()]
    syms, xyz = _split(atoms)
    assert any(x > 10 for x, _, _ in xyz) and any(x < 2 for x, _, _ in xyz)
    frags = classify(syms, xyz, cell, (0, 1, 2))
    assert [(f["ion"], f["extended"]) for f in frags] == [("FSI-", False)]


def test_a_molecule_bonded_to_its_own_image_is_extended():
    cell = cell_matrix(2.2, 8.0, 8.0, 90, 90, 90)
    syms, xyz = _split(water())
    frags = find_fragments(syms, [(x + 1.0, y + 4, z + 4) for x, y, z in xyz], cell, (0,))
    assert [f["extended"] for f in frags] == [True]
    # the same molecule in a big cell is not
    big = cell_matrix(10.0, 10.0, 10.0, 90, 90, 90)
    assert [f["extended"] for f in find_fragments(syms, xyz, big, (0, 1, 2))] == [False]


def test_a_framework_is_extended_and_neutral():
    a = 3.567
    cell = cell_matrix(a, a, a, 90, 90, 90)
    frac = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
            (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
    xyz = [(x * a, y * a, z * a) for x, y, z in frac]
    frags = classify(["C"] * 8, xyz, cell, (0, 1, 2))
    assert [(f["extended"], f["ion"], f["charge"]) for f in frags] == [(True, None, 0)]


def _rocksalt(metal, a):
    cell = cell_matrix(a, a, a, 90, 90, 90)
    fcc = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    atoms = [(metal, tuple(v * a for v in f)) for f in fcc]
    atoms += [("O", tuple((v + .5) % 1 * a for v in f)) for f in fcc]
    return cell, atoms


@pytest.mark.parametrize("metal,a", [("Mg", 4.21), ("Zn", 4.28)])
def test_an_ionic_solid_is_refused_in_auto_mode(metal, a):
    cell, atoms = _rocksalt(metal, a)
    syms, xyz = _split(atoms)
    with pytest.raises(IonChargeError, match="this looks like an ionic solid"):
        classify(syms, xyz, cell, (0, 1, 2))
    # the fragments themselves are still reported without auto
    frags = classify(syms, xyz, cell, (0, 1, 2), auto=False)
    assert sum(1 for f in frags if f["ion"]) == 4


def test_a_table_metal_inside_a_framework_is_refused():
    a = 3.567
    cell = cell_matrix(a, a, a, 90, 90, 90)
    frac = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
            (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
    atoms = [("C", tuple(v * a for v in f)) for f in frac] + [("Li", (a / 2, a / 2, a / 2))]
    syms, xyz = _split(atoms)
    with pytest.raises(IonChargeError, match="extended framework"):
        classify(syms, xyz, cell, (0, 1, 2))


# ----------------------------------------------------------------- parsers

ATOMS = [("LI", 3), ("O", 8), ("H", 1), ("H", 1)]


def test_electron_count_skips_the_spin_block():
    text = (mulliken_block(ATOMS, [2.0, 8.6, 0.7, 0.7], 12.0, "ALPHA+BETA ELECTRONS")
            + mulliken_block(ATOMS, [0.0, 1.0, 0.0, 0.0], 1.0, "ALPHA-BETA ELECTRONS"))
    assert mulliken_electron_count(text) == pytest.approx(12.0)


def test_mulliken_atoms_reads_both_tables():
    text = (mulliken_block(ATOMS, [2.0, 8.6, 0.7, 0.7], 12.0, "ALPHA+BETA ELECTRONS")
            + mulliken_block(ATOMS, [0.0, 0.9, 0.05, 0.05], 1.0, "ALPHA-BETA ELECTRONS"))
    pops, spins = mulliken_atoms(text)
    assert [(a["atom"], a["element"], a["z"], a["value"]) for a in pops] == [
        (1, "Li", 3, 2.0), (2, "O", 8, 8.6), (3, "H", 1, 0.7), (4, "H", 1, 0.7)]
    assert [a["value"] for a in spins] == [0.0, 0.9, 0.05, 0.05]


def test_mulliken_atoms_closed_shell_has_no_spin_table():
    pops, spins = mulliken_atoms(mulliken_block(ATOMS, [2.0, 8.6, 0.7, 0.7], 12.0))
    assert len(pops) == 4 and spins == []


def test_valence_charges():
    assert valence_charges(" ATOMIC NUMBER  82, NUCLEAR CHARGE 22.000, GIVEN IN INPUT") == {82: 22.0}


def test_corpus_mulliken_atoms_match_the_count():
    from pathlib import Path

    from conftest import find_data
    path = find_data("BAND/1LiFSI-1Sulfolane-conf3_*_sp_*.out", "ALPHA-BETA ELECTRONS")
    pops, spins = mulliken_atoms(Path(path).read_text(errors="ignore"))
    assert len(pops) == len(spins) == 25
    assert sum(a["value"] for a in pops) == pytest.approx(156.0, abs=0.05)


# ------------------------------------------- anions outside the ion table

@pytest.mark.parametrize("builder,free", [(carbonate, 2), (hydroxide, 1), (sulfate, 2), (acetate, 1)],
                         ids=["CO3", "OH", "SO4", "CH3COO"])
def test_anions_outside_the_table_have_free_valences(builder, free):
    syms, xyz = _split(builder())
    (frag,) = find_fragments(syms, xyz)
    assert free_valence(frag, syms) == free


@pytest.mark.parametrize("builder", [ec, water, so2f2, hno3, dmso, benzene],
                         ids=["EC", "H2O", "SO2F2", "HNO3", "DMSO", "C6H6"])
def test_neutral_molecules_have_no_free_valence(builder):
    syms, xyz = _split(builder())
    (frag,) = find_fragments(syms, xyz)
    assert free_valence(frag, syms) == 0


def test_free_valence_is_unknown_for_elements_without_valences():
    syms, xyz = ["Fe", "O", "O"], [(0.0, 0.0, 0.0), (1.62, 0.0, 0.0), (-1.62, 0.0, 0.0)]
    (frag,) = find_fragments(syms, xyz)
    assert free_valence(frag, syms) is None
