"""CrystalInputParser reads a deck's whole geometry input.

The deck parser stopped at the space group: no cell record, no atoms, and no
way to tell a SLAB's layer group from a 3D space group. It now reads the group
records, the cell record (only the free parameters the group leaves, expanded
to the full cell) and the atoms, so the geometry input can be rebuilt from the
parse. The acceptance test rebuilds every geometry deck of the corpus with
MACE's own writer and compares the records.

The new keys never reach opt2d12's settings: a before/after comparison of
every OPT/SP corpus parent's SP/OPT/FREQ children (1458 decks, CLI and config
paths) is byte-identical.

Corpus tests run the real files and skip when ``test/`` is absent.
"""
import io
import contextlib
import math
import textwrap

import pytest

import CRYSTALOptToD12 as opt2d12
from conftest import TEST_DATA
from d12_parsers import CrystalInputParser


def _parse_text(tmp_path, text):
    deck = tmp_path / "parent.d12"
    deck.write_text(textwrap.dedent(text).lstrip("\n"))
    return CrystalInputParser(str(deck)).parse()


# ----------------------------------------------------------- deck geometry

def test_cubic_crystal_with_comments_and_crlf(tmp_path):
    deck = tmp_path / "dia.d12"
    deck.write_bytes(b"dia\r\nCRYSTAL\r\n0 0 0\r\n227\r\n"
                     b"3.544300 #a=b=c cubic\r\n1\r\n"
                     b"6   0.125000  0.125000  0.125000   Biso 1.000000 C\r\n"
                     b"END\r\nBASISSET\r\nPOB-TZVP-REV2\r\nEND\r\n")
    p = CrystalInputParser(str(deck)).parse()
    assert p["spacegroup"] == 227 and p["cell_record"] == [3.5443]
    assert p["cell_parameters"] == {"a": 3.5443, "b": 3.5443, "c": 3.5443,
                                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    assert p["n_atoms"] == 1
    assert p["atoms"] == [{"atom_number": 6, "atomic_number": 6,
                           "x": 0.125, "y": 0.125, "z": 0.125}]
    assert p["rhombohedral_axes"] is False


@pytest.mark.parametrize("group,cell,expected", [
    (2, "5.0 6.0 7.0 80.0 85.0 70.0", (5.0, 6.0, 7.0, 80.0, 85.0, 70.0)),
    (14, "5.0 6.0 7.0 100.0", (5.0, 6.0, 7.0, 90.0, 100.0, 90.0)),
    (62, "5.0 6.0 7.0", (5.0, 6.0, 7.0, 90.0, 90.0, 90.0)),
    (139, "4.0 9.0", (4.0, 4.0, 9.0, 90.0, 90.0, 90.0)),
    (166, "2.7 13.7", (2.7, 2.7, 13.7, 90.0, 90.0, 120.0)),
    (194, "2.5 6.7", (2.5, 2.5, 6.7, 90.0, 90.0, 120.0)),
])
def test_crystal_cell_record_expands_per_crystal_system(tmp_path, group, cell, expected):
    p = _parse_text(tmp_path, f"""
        t
        CRYSTAL
        0 0 0
        {group}
        {cell}
        1
        6 0.0 0.0 0.0
        END
        """)
    cp = p["cell_parameters"]
    assert tuple(cp[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")) == expected
    assert p["cell_record"] == [float(v) for v in cell.split()]


def test_rhombohedral_axes_record(tmp_path):
    p = _parse_text(tmp_path, """
        t
        CRYSTAL
        0 1 0
        166
        3.2 60.5
        1
        6 0.0 0.0 0.0
        END
        """)
    assert p["rhombohedral_axes"] is True
    assert p["cell_parameters"] == {"a": 3.2, "b": 3.2, "c": 3.2,
                                    "alpha": 60.5, "beta": 60.5, "gamma": 60.5}


@pytest.mark.parametrize("group,cell,expected", [
    (1, "8.4 8.5 119.9", (8.4, 8.5, 119.9)),     # oblique: a, b, gamma
    (37, "3.3 4.6", (3.3, 4.6, 90.0)),           # rectangular: a, b
    (55, "3.9", (3.9, 3.9, 90.0)),               # square: a
    (80, "2.4612", (2.4612, 2.4612, 120.0)),     # hexagonal: a
])
def test_slab_layer_group_and_cell(tmp_path, group, cell, expected):
    p = _parse_text(tmp_path, f"""
        t
        SLAB
        {group}
        {cell}
        2
        6 0.33 0.67 0.0 Biso 1.000000 C
        206 0.67 0.33 1.5
        BASISSET
        SOLDEF2MSVP
        """)
    assert p["layer_group"] == group == p["spacegroup"]
    cp = p["cell_parameters"]
    assert (cp["a"], cp["b"], cp["gamma"]) == expected and cp["c"] is None
    assert [a["atomic_number"] for a in p["atoms"]] == [6, 6]
    assert [a["atom_number"] for a in p["atoms"]] == [6, 206]
    assert p["atoms"][1]["z"] == 1.5


def test_a_3d_number_in_a_slab_deck_is_not_a_layer_group(tmp_path):
    p = _parse_text(tmp_path, """
        t
        SLAB
        191
        2.4612
        1
        6 0.33 0.67 0.0
        """)
    assert "layer_group" not in p and "cell_parameters" not in p


def test_polymer_rod_group_and_molecule_point_group(tmp_path):
    poly = _parse_text(tmp_path, """
        t
        POLYMER
        75
        1.28
        1
        6 0.0 0.0 0.0
        """)
    assert poly["rod_group"] == 75 and poly["cell_record"] == [1.28]
    assert poly["cell_parameters"]["a"] == 1.28 and poly["cell_parameters"]["b"] is None
    mol = _parse_text(tmp_path, """
        t
        MOLECULE
        1
        2
        1 0.0 0.0 0.37
        1 0.0 0.0 -0.37
        """)
    assert mol["point_group"] == 1 and mol["n_atoms"] == 2
    assert "cell_parameters" not in mol and mol["spacegroup"] is None


def _corpus_decks():
    if not TEST_DATA.is_dir():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    decks = sorted(TEST_DATA.glob("*/*.d12"))
    if not decks:
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    return decks


def _records(lines):
    """Geometry records of a deck: dimensionality through the atom records,
    each as its tokens (comments and text after an atom's x y z dropped)."""
    lines = [l.rstrip("\r\n") for l in lines]
    i = next(k for k in range(1, len(lines))
             if lines[k].strip() in ("CRYSTAL", "SLAB", "POLYMER", "MOLECULE"))
    dim = lines[i].strip()
    n_before_atoms = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 2, "MOLECULE": 1}[dim]
    recs = [[dim]] + [lines[i + 1 + k].split("#")[0].split() for k in range(n_before_atoms)]
    j = i + 1 + n_before_atoms
    nat = int(lines[j].split()[0])
    recs.append([str(nat)])
    recs += [lines[j + 1 + k].split()[:4] for k in range(nat)]
    return recs


def _same_number(orig, new):
    """Same record value: equal text, equal integer, or equal to the precision
    the writer printed (e.g. a deck's 3.544300 against the writer's 3.54430000)."""
    if orig == new:
        return True
    if orig.lstrip("-").isdigit():
        return new.lstrip("-").isdigit() and int(orig) == int(new)
    if "e" in new.lower() or "." not in new:
        decimals = 12
    else:
        decimals = len(new.split(".")[1])
    return math.isclose(float(orig), float(new), rel_tol=0,
                        abs_tol=max(1e-9, 0.5 * 10 ** -decimals + 1e-12))


def _rebuild(tmp_path, parsed):
    """The geometry input written back by opt2d12's own deck writer."""
    dim = parsed["dimensionality"]
    group = {"CRYSTAL": parsed.get("spacegroup"), "SLAB": parsed.get("layer_group"),
             "POLYMER": parsed.get("rod_group"), "MOLECULE": 1}[dim]
    cp = parsed.get("cell_parameters")
    cell = None
    if cp:
        cell = [cp["a"], cp["b"] if cp["b"] is not None else cp["a"],
                cp["c"] if cp["c"] is not None else 500.0,
                cp["alpha"], cp["beta"], cp["gamma"] if cp["gamma"] is not None else 90.0]
    coords = [{"atom_number": str(a["atomic_number"]), "x": repr(a["x"]),
               "y": repr(a["y"]), "z": repr(a["z"]), "is_unique": True}
              for a in parsed["atoms"]]
    settings = {k: v for k, v in parsed.items()}
    settings.update(calculation_type="SP", write_only_unique=False, spacegroup=group,
                    tolerances={}, k_points=None)
    ext = parsed.get("external_basis_data") or None
    if ext:
        settings.update(has_original_external_basis=True, use_original_external_basis=True)
    out = tmp_path / "rebuilt.d12"
    with contextlib.redirect_stdout(io.StringIO()):
        assert opt2d12.write_d12_file(
            str(out), {"conventional_cell": cell, "coordinates": coords,
                       "crystallographic_coordinates": coords}, settings,
            external_basis_data=ext)
    return out.read_text().splitlines()


def test_every_corpus_geometry_deck_rebuilds_from_the_parse(tmp_path):
    """Acceptance: for each of the corpus' CRYSTAL and SLAB decks, the parse
    holds everything the geometry input needs - the writer rebuilds the same
    records (whitespace and float formatting normalised)."""
    counts = {"CRYSTAL": 0, "SLAB": 0, "POLYMER": 0}
    failures = []
    for deck in _corpus_decks():
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = CrystalInputParser(str(deck)).parse()
        dim = parsed["dimensionality"]
        if dim not in counts:
            continue
        counts[dim] += 1
        assert parsed.get("cell_parameters"), deck.name
        assert len(parsed["atoms"]) == parsed["n_atoms"], deck.name
        orig = _records(deck.read_text().splitlines())
        new = _records(_rebuild(tmp_path, parsed))
        same = len(orig) == len(new) and all(
            len(o) == len(n) and all(_same_number(a, b) for a, b in zip(o, n))
            for o, n in zip(orig, new))
        if not same:
            failures.append(deck.name)
    assert sum(counts.values()) >= 137, counts
    assert not failures, failures


def test_every_corpus_slab_deck_carries_its_layer_group():
    for deck in _corpus_decks():
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = CrystalInputParser(str(deck)).parse()
        if parsed["dimensionality"] == "SLAB":
            assert parsed["layer_group"] == parsed["spacegroup"], deck.name
            assert 1 <= parsed["layer_group"] <= 80
