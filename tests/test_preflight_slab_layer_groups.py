"""SLAB layer groups / POLYMER rod groups: the record CRYSTAL actually reads.

The record after the SLAB keyword is a LAYER group (IGR 1-80, manual Appendix
A.2, page 421) and the record after POLYMER a ROD group (IGR 1-99, Appendix
A.3, pages 422-424). ``writer.create_d12_file`` wrote the 3D space-group
number into both, which fails in two different ways:

* SLAB - anything above 80 is out of range. A hexagonal slab (space group 191)
  wrote "191", and CRYSTAL MPI_Aborted right after the title without writing
  any fort.87 at all, so MACE's fort.87-based error classification saw nothing.
  Appendix A.2 shows 191 is layer group 80.
* POLYMER - a wrong number is usually still inside 1-99, so it is accepted and
  silently builds a different chain. Nothing fails; the answer is just wrong.

Every test below fails against the old code and passes against the new one,
except those marked COMPATIBILITY GUARD, which pin behaviour that must NOT
change (the production P1 slab decks under ``test/``, and 3D decks, which the
new resolution block must leave alone).

The appendix rows themselves are transcribed from the PDF - the plain-text dump
interleaves A.2's two printed columns - so the first block of tests re-derives
the manual's own invariants from the transcription rather than trusting it.

``NewCifToD12`` is imported lazily inside the writer tests, not at module
level: it imports ``ase.io``, and a module-level ``pytest.importorskip`` would
take the pure-table tests down with it on a machine without ase.
"""
import os
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_constants  # noqa: E402
from d12_constants import (  # noqa: E402
    DEFAULT_TOLERANCES,
    LAYER_GROUP_CANDIDATES,
    LAYER_GROUP_FROM_SPACEGROUP,
    LAYER_GROUP_ROWS,
    LAYER_GROUPS_POLAR_IN_Z,
    ROD_GROUP_CANDIDATES,
    ROD_GROUP_FROM_SPACEGROUP,
    ROD_GROUP_ROWS,
    ROD_GROUPS_FREE_OF_AXIS_OPERATIONS,
    check_layer_group_cell,
    generate_unit_cell_line,
    layer_group_lattice,
)


def _writer():
    """Import NewCifToD12 on demand (it pulls in ase.io at import time)."""
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    import NewCifToD12

    return NewCifToD12

# A real production slab deck: layer group 1, hexagonal-looking cell.
REFERENCE_SLAB = TEST_DATA / "SP" / "4LG_FSI_2x2_AA_opt_sp.d12"


# ============================================================
# The appendix transcription (Appendix A.2 / A.3)
# ============================================================


def test_appendix_row_counts_match_the_manual():
    """Manual L434: "230 space groups, 80 layer groups, 99 rod groups, 45 point
    groups are available (Appendix A)".

    A.3 in particular runs to 99 - pages 423-424 carry IGR 32-99, all of them
    unparenthesised. Stopping at 31 (where page 422 breaks) would give
    confidently wrong answers for every space group from 75 to 194.
    """
    assert [row[0] for row in LAYER_GROUP_ROWS] == list(range(1, 81))
    assert [row[0] for row in ROD_GROUP_ROWS] == list(range(1, 100))


def test_a_mapped_space_group_has_exactly_one_unparenthesised_appendix_row():
    """The invariant the map is built on, asserted directly.

    A.2/A.3 header: "The number of the space group is written in parentheses
    when the orientation of the symmetry operators does not correspond to the
    first setting in the I. T." So a parenthesised row cannot be inverted. But
    being unparenthesised is not on its own enough: the appendix may print
    several rows for one number, and a mapped space group must have exactly
    ONE candidate row in total, that one unparenthesised.

    Checking only "no space group has two unparenthesised rows" is the wrong
    invariant - it is true of the appendix as printed and would keep passing
    while the real ambiguity (one unparenthesised row plus parenthesised
    siblings) went unnoticed.
    """
    for rows, table, candidates in [
        (LAYER_GROUP_ROWS, LAYER_GROUP_FROM_SPACEGROUP, LAYER_GROUP_CANDIDATES),
        (ROD_GROUP_ROWS, ROD_GROUP_FROM_SPACEGROUP, ROD_GROUP_CANDIDATES),
    ]:
        by_igr = {row[0]: row for row in rows}
        for sg, igr in table.items():
            assert candidates[sg] == (igr,), (sg, candidates[sg])
            assert not by_igr[igr][3].startswith("("), (sg, igr)
        # And every candidate row really came from the transcription.
        for sg, igrs in candidates.items():
            assert [i for i in igrs] == sorted(igrs)
            for igr in igrs:
                assert int(by_igr[igr][3].strip("()")) == sg

    assert len(LAYER_GROUP_FROM_SPACEGROUP) == 45
    assert len(ROD_GROUP_FROM_SPACEGROUP) == 75


def test_parenthesis_only_space_groups_are_refused():
    """Space groups the appendices print only in parentheses.

    The tempting alternative - "take the single candidate whenever there is
    exactly one" - would map space group 4 to layer group 9 (P2_111, the 2_1
    along x) against ITA #4's 2_1 along b. Refusing is the only defensible
    answer.
    """
    for sg in (4, 5, 27, 29, 30, 31, 38, 39, 49, 53, 54, 57):
        assert len(LAYER_GROUP_CANDIDATES[sg]) == 1
        assert sg not in LAYER_GROUP_FROM_SPACEGROUP
    for sg in (4, 11):
        assert len(ROD_GROUP_CANDIDATES[sg]) == 1
        assert sg not in ROD_GROUP_FROM_SPACEGROUP


def test_same_type_different_orientation_siblings_are_refused():
    """The case the parenthesis flag alone does NOT catch.

    A.2 rows 23 and 27 are both C2v^1, printed "25" and "(25)": Pmm2 has its
    2-fold along the surface normal, P2mm in the plane, and a symmetric
    two-sided slab is the second. Taking the unparenthesised row would silently
    pick Pmm2. Same shape at N = 28 (rows 24 Pma2 / 30 P2mb, both C2v^4) and
    N = 51 (rows 39 Pmma / 38 Pmam, both D2h^5); A.3 has it at N = 25 (rows 20,
    24, 26, all C2v^1) and N = 26 (rows 21, 22, both C2v^2).

    check_layer_group_cell cannot rescue any of these - both members of each
    A.2 pair are rectangular - so the map itself has to refuse.
    """
    assert LAYER_GROUP_CANDIDATES[25] == (23, 27)
    assert LAYER_GROUP_CANDIDATES[28] == (24, 30)
    assert LAYER_GROUP_CANDIDATES[51] == (38, 39)
    for sg in (25, 28, 51):
        assert sg not in LAYER_GROUP_FROM_SPACEGROUP
        assert all(layer_group_lattice(g) == "rectangular"
                   for g in LAYER_GROUP_CANDIDATES[sg])

    assert ROD_GROUP_CANDIDATES[25] == (20, 24, 26)
    assert ROD_GROUP_CANDIDATES[26] == (21, 22)
    for sg in (25, 26):
        assert sg not in ROD_GROUP_FROM_SPACEGROUP


def test_the_measured_hexagonal_case_maps_to_layer_group_80():
    """The deck that MPI_Aborted: space group 191 (P6/mmm).

    A.2 row 80 is "P6/mmm D6h^1 191", unparenthesised. 191 is not a legal layer
    group number at all, which is why CRYSTAL died before writing anything.
    """
    assert LAYER_GROUP_FROM_SPACEGROUP[191] == 80
    assert 191 not in range(1, 81)


def test_manual_rod_group_examples_are_transcribed_correctly():
    """The manual's three POLYMER decks name their rod groups.

    (SN)x is "POLYMER / 4 / 4.431" (L29216-29218), the water polymer
    "POLYMER / 1 / 4.965635" (L29230-29232) and formamide "POLYMER / 4 / 8.774"
    (L29251-29253). Rod group 4 is P2_111 and rod group 1 is P1.
    """
    rows = {row[0]: row for row in ROD_GROUP_ROWS}
    assert rows[1][1] == "P1"
    assert rows[4][1] == "P2_111"
    # The column carried here is A.3's "polymer" symbol (x direction), which
    # for row 78 reads P6_3, consistent with Schoenflies C6^6 and space group
    # 173. (A.3's separate Hermann-Mauguin (z direction) column prints "P6_6"
    # on that row, which reads as a slip; it is not transcribed.)
    assert rows[78][1:] == ("P6_3", "C6^6", "173")
    # A.3 runs past the page break with a non-monotonic space-group column:
    # IGR 59-64 are 149, 151, 153, 150, 152, 154 (PDF page 423).
    assert [rows[i][3] for i in range(59, 65)] == [
        "149", "151", "153", "150", "152", "154",
    ]


# ============================================================
# Lattice class, and the drift guard against generate_unit_cell_line
# ============================================================


def test_layer_group_lattice_does_not_drift_from_the_cell_line():
    """The SLAB cell record is "a,[b],[gamma]" - b "for rectangular lattices
    only", the angle for "triclinic lattices only" (manual L999-1002) - and
    A.2's four lattice headings partition the groups 1-7 / 8-48 / 49-64 / 65-80.

    layer_group_lattice and generate_unit_cell_line encode that partition
    separately, so pin them together the way
    test_the_two_opt_types_copies_do_not_drift pins the two OPT_TYPES copies.
    """
    expected_values = {"oblique": 3, "rectangular": 2, "square": 1, "hexagonal": 1}
    for igr in range(1, 81):
        line = generate_unit_cell_line(
            igr, [2.5, 2.5, 20.0, 90.0, 90.0, 120.0], "SLAB"
        )
        assert len(line.split()) == expected_values[layer_group_lattice(igr)], igr


@pytest.mark.parametrize("igr", [0, 81, -1])
def test_layer_group_lattice_rejects_out_of_range(igr):
    with pytest.raises(ValueError):
        layer_group_lattice(igr)


def test_hexagonal_layer_group_refuses_a_ninety_degree_cell():
    """A hexagonal layer group prints a alone, so CRYSTAL supplies gamma = 120.

    Handing it a gamma = 90 cell writes a deck that runs to completion on a
    structure the user never described. The manual's diamond (100) deck
    (L29193-29196) prints "2.52437 2.52437" - two values with a == b - which is
    why the value count cannot be inferred from the numbers instead.
    """
    assert check_layer_group_cell(80, 2.47, 2.47, 90.0) is not None
    assert check_layer_group_cell(61, 2.47, 2.47, 120.0) is not None  # square
    assert check_layer_group_cell(37, 2.47, 3.10, 120.0) is not None  # rectangular
    # Oblique constrains nothing.
    assert check_layer_group_cell(1, 2.47, 3.10, 77.0) is None


def test_real_slab_decks_pass_the_lattice_check():
    """COMPATIBILITY GUARD - passes before and after.

    Asserts only that the real corpus is not refused. Note what this does NOT
    cover: every production slab deck under test/ is layer group 1, which is
    OBLIQUE, and check_layer_group_cell returns None unconditionally for
    oblique - so this test never reads either tolerance constant. Setting them
    to 1e-9 and 1e-12 leaves all of these decks passing. The tolerances are
    pinned by test_lattice_tolerances_are_pinned_by_a_nonoblique_group below;
    do not treat this test as covering them.
    """
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    decks = [
        p
        for p in sorted(TEST_DATA.rglob("*.d12"))
        if p.read_text(errors="ignore").splitlines()[1:2] == ["SLAB"]
    ]
    if not decks:
        pytest.skip("no SLAB decks in the corpus")
    for deck in decks:
        lines = deck.read_text(errors="ignore").splitlines()
        igr = int(lines[2])
        values = [float(v) for v in lines[3].split()]
        a, b, gamma = values if len(values) == 3 else (values[0], values[0], 90.0)
        assert check_layer_group_cell(igr, a, b, gamma) is None, deck


def test_two_sided_layer_groups_are_not_auto_mapped():
    """z is Cartesian and non-periodic in a SLAB deck (manual L1021-1022,
    L29023-29024), measured from the layer group's own origin.

    34 of the 45 auto-mappable layer groups contain an operation reversing z,
    and CRYSTAL builds the other half of the slab from z = 0 - the manual's
    diamond (100) deck lists five atoms at z = 0.44625 .. 4.01625 for what its
    title calls a "ten layers slab" (L29193-29207). A converter working from
    fractional coordinates has no way to know where that plane is, and the
    coordinates cannot reveal it either, since an asymmetric unit legitimately
    sits entirely on one side of it.
    """
    auto = set(LAYER_GROUP_FROM_SPACEGROUP.values())
    assert sorted(auto & LAYER_GROUPS_POLAR_IN_Z) == [
        1, 25, 26, 49, 55, 56, 65, 69, 70, 73, 77,
    ]
    assert len(auto - LAYER_GROUPS_POLAR_IN_Z) == 34
    # Only P1 leaves the rod axis free: rod group 2 already has an inversion
    # centre on it.
    assert ROD_GROUPS_FREE_OF_AXIS_OPERATIONS == frozenset({1})


# ============================================================
# The writer (writer.create_d12_file)
# ============================================================

@pytest.fixture
def writer():
    """NewCifToD12, imported per-test - see the module docstring."""
    return _writer()


def slab_options(**overrides):
    opts = dict(
        dimensionality="SLAB",
        calculation_type="SP",
        basis_set_type="INTERNAL",
        basis_set="POB-TZVP-REV2",
        method="DFT",
        dft_functional="PBE",
        is_spin_polarized=False,
        tolerances=DEFAULT_TOLERANCES,
        scf_method="DIIS",
        symmetry_handling="CIF",
    )
    opts.update(overrides)
    return opts


# Manual Test05, graphite 2D (L29025-29030): "SLAB / 77 / 2.47 / 1 /
# 6 -0.33333333333 0.33333333333 0." - one atom, on the z = 0 plane. Given the
# 3D space group of a single hexagonal sheet in a vacuum cell (191), the
# appendix answer is layer group 80.
def graphene_cif(**overrides):
    cif = dict(
        a=2.47,
        b=2.47,
        c=20.0,
        alpha=90.0,
        beta=90.0,
        gamma=120.0,
        spacegroup=191,
        atomic_numbers=[6],
        symbols=["C"],
        positions=[[-1.0 / 3.0, 1.0 / 3.0, 0.0]],
    )
    cif.update(overrides)
    return cif


def test_slab_writes_a_layer_group_not_the_space_group(writer, tmp_path):
    """Old behaviour: "SLAB / 191 / 2.47000000 2.47000000 120.000000".

    191 is not a layer group, and the three-value cell line is the oblique
    form. With the layer group named explicitly (the caller thereby also
    asserting that z = 0 is the mirror plane) the deck becomes the manual's own
    shape: the group, then a alone because the lattice is hexagonal.
    """
    out = tmp_path / "graphene.d12"
    assert writer.create_d12_file(
        graphene_cif(), str(out), slab_options(layer_group=80)
    )
    lines = out.read_text().splitlines()
    assert lines[1] == "SLAB"
    assert lines[2] == "80"
    assert lines[3] == "2.47000000"


def test_hexagonal_slab_from_a_space_group_is_refused_not_mirrored(writer, tmp_path):
    """THE headline case, and the reason this is not a plain "write 80" fix.

    Old behaviour wrote "191" and CRYSTAL aborted with no output. Simply
    substituting 80 would be worse: layer group 80 has sigma_h, so a slab
    sitting at fractional z ~ 0.5 (what a pymatgen/VASP slab CIF carries) is
    mirrored to -c/2 and the deck silently describes a bilayer separated by c.
    Loud failure must not become quiet corruption, so the automatic map refuses
    every layer group whose z origin it cannot know.
    """
    out = tmp_path / "bilayer.d12"
    cif = graphene_cif(positions=[[-1.0 / 3.0, 1.0 / 3.0, 0.5]])
    assert writer.create_d12_file(cif, str(out), slab_options()) is False
    assert not out.exists(), "a refused deck must leave nothing on disk"


def test_space_group_without_a_first_setting_layer_group_is_refused(writer, tmp_path):
    """Space group 13 appears in A.2 only as "(13)", on rows 7 and 17.

    Both orientations exist and the appendix does not say which the number
    means, so there is no answer to give.
    """
    out = tmp_path / "sg13.d12"
    cif = graphene_cif(spacegroup=13)
    assert writer.create_d12_file(cif, str(out), slab_options()) is False
    assert not out.exists()


@pytest.mark.parametrize("bad", [0, 81, 191, "eighty"])
def test_out_of_range_layer_group_leaves_no_truncated_deck(writer, tmp_path, bad):
    """generate_unit_cell_line raises ValueError outside 1-80.

    That call sits inside the ``with open(...)`` block, so an unvalidated
    override would abort with the title already written - the exact failure
    tests/test_d12_abort_no_truncated_deck.py exists to prevent. Validate
    before the file is opened.
    """
    out = tmp_path / "bad.d12"
    assert (
        writer.create_d12_file(
            graphene_cif(), str(out), slab_options(layer_group=bad)
        )
        is False
    )
    assert not out.exists()


def test_auto_mapped_layer_group_is_refused_when_the_cell_contradicts_it(
    writer, tmp_path
):
    """The lattice class comes from the layer group, not from the numbers.

    Space group 183 maps to layer group 77 (P6mm), which is hexagonal, so the
    cell record is "a" alone and CRYSTAL supplies gamma = 120. Handing it a
    gamma = 90 cell would write a deck that runs to completion on a structure
    nobody described. Nothing upstream puts the cell into the first setting, so
    for an automatically chosen group this is the only check that ever compares
    the group against the cell: refuse.
    """
    out = tmp_path / "autowrongcell.d12"
    cif = graphene_cif(spacegroup=183, gamma=90.0)
    assert writer.create_d12_file(cif, str(out), slab_options()) is False
    assert not out.exists()


def test_explicit_layer_group_downgrades_the_cell_check_to_a_warning(
    writer, tmp_path, capsys
):
    """Naming the group asserts the group; the deck is still written.

    A relaxed slab whose gamma is a tenth of a degree outside
    LAYER_GROUP_ANGLE_TOL_DEG is still the group its author says it is, and a
    refusal would leave no route through at all. So an explicit
    options["layer_group"] warns instead - loudly, naming the lattice class -
    and writes the deck.
    """
    out = tmp_path / "wrongcell.d12"
    cif = graphene_cif(gamma=90.0)
    assert writer.create_d12_file(cif, str(out), slab_options(layer_group=80))
    captured = capsys.readouterr()
    assert "hexagonal" in (captured.out + captured.err)
    lines = out.read_text().splitlines()
    assert lines[1:4] == ["SLAB", "80", "2.47000000"]


def test_real_p1_slab_deck_is_reproduced_record_for_record(writer, tmp_path):
    """COMPATIBILITY GUARD - passes before and after.

    Layer group 1 is oblique, so the cell line keeps all three values and every
    production slab deck is untouched by the change. Asserted against a real
    deck rather than a fixture: this project has been burned by fixtures that
    encoded the bug.

    c is set to 1.0 so that the deck's own Cartesian z is the fractional z; c
    enters a SLAB deck only through that conversion.
    """
    if not REFERENCE_SLAB.is_file():
        pytest.skip(f"test/ corpus file not present: {REFERENCE_SLAB.name}")
    deck = REFERENCE_SLAB.read_text().splitlines()
    a, b, gamma = [float(v) for v in deck[3].split()]
    natoms = int(deck[4])
    rows = [deck[5 + i].split() for i in range(natoms)]
    cif = dict(
        a=a,
        b=b,
        c=1.0,
        alpha=90.0,
        beta=90.0,
        gamma=gamma,
        spacegroup=1,
        atomic_numbers=[int(r[0]) for r in rows],
        symbols=[r[-1] for r in rows],
        positions=[[float(r[1]), float(r[2]), float(r[3])] for r in rows],
    )

    out = tmp_path / "roundtrip.d12"
    assert writer.create_d12_file(cif, str(out), slab_options())
    lines = out.read_text().splitlines()

    assert lines[1:5] == deck[1:5], "SLAB / layer group / cell / natoms must match"
    for i in range(natoms):
        written = lines[5 + i].split()
        original = deck[5 + i].split()
        assert written[0] == original[0] and written[-1] == original[-1]
        for j in (1, 2, 3):
            assert float(written[j]) == pytest.approx(float(original[j]), abs=1e-9)


# --- POLYMER ---------------------------------------------------------------


def polymer_cif(**overrides):
    """(SN)x-like chain: manual L29216-29218 gives "POLYMER / 4 / 4.431"."""
    cif = dict(
        a=4.431,
        b=10.0,
        c=10.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        spacegroup=1,
        atomic_numbers=[16, 7],
        symbols=["S", "N"],
        positions=[[0.0, 0.5, 0.5], [0.14160054, 0.567, 0.5]],
    )
    cif.update(overrides)
    return cif


def test_polymer_writes_a_rod_group_not_the_space_group(writer, tmp_path):
    """Old behaviour wrote the space group, here 75.

    75 is inside 1-99, so CRYSTAL accepts it as rod group 75 (P6_5) and
    silently builds a completely different chain - nothing fails. A.3 row 32 is
    "P4 C4^1 P4 75", so 75 is rod group 32.
    """
    out = tmp_path / "rod.d12"
    cif = polymer_cif(spacegroup=75)
    assert writer.create_d12_file(
        cif, str(out), slab_options(dimensionality="POLYMER", rod_group=32)
    )
    lines = out.read_text().splitlines()
    assert lines[1] == "POLYMER"
    assert lines[2] == "32"
    assert lines[3] == "4.43100000"
    assert ROD_GROUP_FROM_SPACEGROUP[75] == 32


def test_polymer_with_wrapped_fractional_coordinates_is_refused(writer, tmp_path):
    """y and z are Cartesian distances from the ROD AXIS (manual L1019-1020).

    NewCifToD12 writes positions[i][1] * b and positions[i][2] * c, which for
    wrapped fractional coordinates is a chain running at roughly (b/2, c/2) -
    off the axis every rod group but P1 pins at y = z = 0. The manual's own
    decks carry signed values about it ("16 0.0 -0.844969 0.0", L29220). So
    the automatic map refuses anything but rod group 1.
    """
    out = tmp_path / "offaxis.d12"
    cif = polymer_cif(spacegroup=75)
    assert (
        writer.create_d12_file(
            cif, str(out), slab_options(dimensionality="POLYMER")
        )
        is False
    )
    assert not out.exists()


def test_out_of_range_rod_group_is_refused(writer, tmp_path):
    """POLYMER has no downstream backstop: generate_unit_cell_line's POLYMER
    branch returns f"{a:.8f}" for any integer and never range-checks, so an
    unvalidated rod_group is written verbatim into the deck."""
    out = tmp_path / "badrod.d12"
    assert (
        writer.create_d12_file(
            polymer_cif(), str(out), slab_options(dimensionality="POLYMER", rod_group=100)
        )
        is False
    )
    assert not out.exists()


def test_polymer_record_carries_exactly_one_lattice_value():
    """Pins the RECORD SHAPE, not rod-group handling.

    A polymer has one lattice vector, so the record after the rod group is one
    value - the manual's three POLYMER decks print "4.431" (L29218), "4.965635"
    (L29240) and "8.774" (L29262). generate_unit_cell_line's POLYMER branch
    ignores its group argument entirely, so this holds for every IGR, including
    numbers that are not rod groups at all; it is not coverage of the map.
    """
    for igr in (1, 4, 50, 99):
        assert (
            len(
                generate_unit_cell_line(
                    igr, [4.431, 10.0, 10.0, 90.0, 90.0, 90.0], "POLYMER"
                ).split()
            )
            == 1
        )


def test_polymer_p1_is_unchanged(writer, tmp_path):
    """COMPATIBILITY GUARD - passes before and after.

    Space group 1 maps to rod group 1, and the POLYMER cell line was already
    f"{a:.8f}", so a P1 chain is byte-identical.
    """
    out = tmp_path / "p1.d12"
    assert writer.create_d12_file(
        polymer_cif(), str(out), slab_options(dimensionality="POLYMER")
    )
    lines = out.read_text().splitlines()
    assert lines[1:4] == ["POLYMER", "1", "4.43100000"]


def test_no_stray_state_between_slab_and_crystal_decks(writer, tmp_path):
    """CRYSTAL decks must not be touched by the new resolution block.

    ``layer_group``/``rod_group`` are local to create_d12_file and only set for
    2D/1D, but the CRYSTAL branch shares the same function, so pin that a 3D
    deck still writes its space group.
    """
    cif = dict(
        a=3.567,
        b=3.567,
        c=3.567,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        spacegroup=227,
        atomic_numbers=[6],
        symbols=["C"],
        positions=[[0.125, 0.125, 0.125]],
    )
    out = tmp_path / "dia.d12"
    assert writer.create_d12_file(
        cif, str(out), slab_options(dimensionality="CRYSTAL")
    )
    lines = out.read_text().splitlines()
    assert lines[1] == "CRYSTAL"
    assert lines[3] == "227"


def test_sibling_ambiguity_is_refused_with_the_siblings_named(
    writer, tmp_path, capsys
):
    """Space group 25 is A.2 rows 23 (Pmm2) and 27 (P2mm), both C2v^1.

    The unparenthesised row is Pmm2 - 2-fold along the surface normal - but a
    symmetric two-sided slab is P2mm, and the number cannot say which. The
    refusal has to name both, otherwise the user has nothing to act on.
    """
    out = tmp_path / "sg25.d12"
    cif = graphene_cif(spacegroup=25, gamma=90.0, b=3.10)
    assert writer.create_d12_file(cif, str(out), slab_options()) is False
    assert not out.exists()
    message = capsys.readouterr()
    message = message.out + message.err
    assert "23" in message and "27" in message


def test_rod_sibling_ambiguity_is_refused(writer, tmp_path):
    """A.3 prints three rows for space group 25 (IGR 20 P2mm, 24 Pm2m, 26
    Pmm2, all C2v^1) and two for 26 (IGR 21, 22, both C2v^2).

    This is the dangerous direction: a rod group number is almost always inside
    1-99, so a wrong pick is accepted by CRYSTAL and quietly builds a different
    chain.
    """
    for sg in (25, 26):
        out = tmp_path / f"rodsg{sg}.d12"
        cif = polymer_cif(spacegroup=sg)
        assert (
            writer.create_d12_file(
                cif, str(out), slab_options(dimensionality="POLYMER")
            )
            is False
        )
        assert not out.exists()


# --- SLABCUT (opt-in) -------------------------------------------------------


# Manual, "Optimization of surface" (L30416-30436): the alpha-Al2O3 (001)
# surface, "CRYSTAL / 0 0 0 / 167 / 4.7602 12.9933 / 2 / 13 0. 0. 0.35216 /
# 8 0.30624 0. 0.25 / SLABCUT / 0 0 1 / 1 6 / OPTGEOM / ...".
def corundum_cif(**overrides):
    cif = dict(
        a=4.7602,
        b=4.7602,
        c=12.9933,
        alpha=90.0,
        beta=90.0,
        gamma=120.0,
        spacegroup=167,
        atomic_numbers=[13, 8],
        symbols=["Al", "O"],
        positions=[[0.0, 0.0, 0.35216], [0.30624, 0.0, 0.25]],
    )
    cif.update(overrides)
    return cif


def crystal_options(**overrides):
    return slab_options(dimensionality="CRYSTAL", **overrides)


def test_slabcut_records_follow_the_manual_example(writer, tmp_path):
    """The manual's own alpha-Al2O3 (001) deck, record for record.

    "SLABCUT / 0 0 1 / 1 6" sits straight after the coordinates with no END in
    between, and ISUP and NL share one record (manual L4729-4735, L30416-30436).
    Fails against the old code, which had no SLABCUT route at all.
    """
    out = tmp_path / "corundum.d12"
    assert writer.create_d12_file(
        corundum_cif(),
        str(out),
        crystal_options(slabcut={"miller": [0, 0, 1], "isup": 1, "nlayers": 6}),
    )
    lines = out.read_text().splitlines()
    assert lines[1] == "CRYSTAL"
    assert lines[3] == "167"
    cut = lines.index("SLABCUT")
    assert lines[cut : cut + 3] == ["SLABCUT", "0 0 1", "1 6"]
    # Straight after the last coordinate record, no END in between; the
    # geometry block is still closed downstream (BASISSET for an internal
    # basis, END for an external one).
    assert lines[cut - 1].split()[0] == "8"
    assert lines[cut + 3] == "BASISSET"


def test_slabcut_probe_deck_asks_slabinfo_and_stops(writer, tmp_path):
    """Run 1 of the two the manual requires (L4763-4771).

    ISUP is unknowable before this run - "The surface layer ISUP may be found
    from an analysis of the information printed by the SLABINFO option"
    (L4759-4760) - so a probe deck deliberately carries neither 'isup' nor
    'nlayers', and TESTGEOM stops the program after the geometry block
    (L4761-4762).
    """
    out = tmp_path / "probe.d12"
    assert writer.create_d12_file(
        corundum_cif(),
        str(out),
        crystal_options(slabcut={"miller": [0, 0, 1], "probe": True}),
    )
    lines = out.read_text().splitlines()
    cut = lines.index("SLABINFO")
    assert lines[cut : cut + 3] == ["SLABINFO", "0 0 1", "TESTGEOM"]
    assert lines[cut - 1].split()[0] == "8"
    assert "SLABCUT" not in lines


def test_slabcut_warns_that_the_shrinking_factors_are_from_the_3d_cell(
    writer, tmp_path, capsys
):
    """SLABCUT note 4 (L4755-4758): CRYSTAL picks new 2D vectors by minimal
    cell area, shortest vectors and minimum |cos(gamma)|.

    IS3 is forced to 1 for a 2D system (L1772, L9356), but IS1/IS2 here are
    still derived from the 3D a and b, and SLABINFO note 5 warns that "The
    shape of the new cell may be very different, computational parameters must
    be carefully checked."
    """
    out = tmp_path / "warn.d12"
    assert writer.create_d12_file(
        corundum_cif(),
        str(out),
        crystal_options(slabcut={"miller": [1, 1, 0], "isup": 3, "nlayers": 8}),
    )
    message = capsys.readouterr()
    assert "SHRINK" in (message.out + message.err)


@pytest.mark.parametrize(
    "bad",
    [
        {"miller": [0, 0, 1]},                       # no isup/nlayers
        {"miller": [0, 0, 1], "isup": 0, "nlayers": 6},
        {"miller": [0, 0, 1], "isup": 1, "nlayers": 0},
        {"miller": [0, 0, 0], "isup": 1, "nlayers": 6},
        {"miller": [0, 1], "isup": 1, "nlayers": 6},
        {"isup": 1, "nlayers": 6},                   # no miller
        "0 0 1",
    ],
)
def test_bad_slabcut_leaves_no_truncated_deck(writer, tmp_path, bad):
    """Validation happens before the file is opened.

    Same rule as the layer group: a refusal must leave nothing on disk, which
    is what tests/test_d12_abort_no_truncated_deck.py exists to protect.
    """
    out = tmp_path / "badcut.d12"
    assert (
        writer.create_d12_file(corundum_cif(), str(out), crystal_options(slabcut=bad))
        is False
    )
    assert not out.exists()


def test_slabcut_is_refused_on_a_slab_deck(writer, tmp_path):
    """SLABCUT is geometry editing for a 3D structure - it "allows the creation
    of a slab (2D) of given thickness from the 3D perfect lattice" (manual note
    16, L1228-1230). There is nothing to cut in a deck that is already 2D."""
    out = tmp_path / "double.d12"
    assert (
        writer.create_d12_file(
            graphene_cif(),
            str(out),
            slab_options(
                layer_group=80, slabcut={"miller": [0, 0, 1], "isup": 1, "nlayers": 2}
            ),
        )
        is False
    )
    assert not out.exists()


def test_deck_without_slabcut_carries_no_slabcut_records(writer, tmp_path):
    """COMPATIBILITY GUARD - passes before and after.

    The whole SLABCUT route is gated on options["slabcut"] being present, so a
    deck that does not ask for it must gain nothing.
    """
    plain = tmp_path / "plain.d12"
    assert writer.create_d12_file(corundum_cif(), str(plain), crystal_options())
    text = plain.read_text()
    assert "SLABCUT" not in text and "SLABINFO" not in text
    assert "TESTGEOM" not in text


def test_module_constants_are_importable_from_the_writer(writer):
    """The writer imports the appendix maps by name; a rename would break the
    standalone ``python NewCifToD12.py`` entry point at import time."""
    assert writer.LAYER_GROUP_FROM_SPACEGROUP is (
        d12_constants.LAYER_GROUP_FROM_SPACEGROUP
    )
    assert writer.ROD_GROUP_FROM_SPACEGROUP is (
        d12_constants.ROD_GROUP_FROM_SPACEGROUP
    )
    assert writer.LAYER_GROUP_CANDIDATES is d12_constants.LAYER_GROUP_CANDIDATES
    assert writer.ROD_GROUP_CANDIDATES is d12_constants.ROD_GROUP_CANDIDATES
    assert os.path.basename(writer.__file__) == "NewCifToD12.py"


def test_lattice_tolerances_are_pinned_by_a_nonoblique_group():
    """The two tolerance constants are justified at length in d12_constants but
    nothing exercised them: the whole test/ corpus is layer group 1 (oblique),
    for which check_layer_group_cell returns None before reading either one.

    Layer group 77 (P6mm, manual Appendix A.2) is hexagonal, so it does read
    them. Without this test both constants could be tightened to 1e-12 with a
    green suite, and real hexagonal slab decks would then start being refused.
    """
    from d12_constants import (
        LAYER_GROUP_ANGLE_TOL_DEG,
        LAYER_GROUP_LENGTH_RTOL,
        check_layer_group_cell,
    )

    HEX = 77
    # A cell within tolerance must be accepted. The corpus really does carry
    # angles a few thousandths off (4LG_FSI_2x2_AA_opt_sp.d12: gamma=120.003561),
    # which is what the angle tolerance exists to absorb.
    assert check_layer_group_cell(HEX, 2.47, 2.47, 120.003561) is None
    assert LAYER_GROUP_ANGLE_TOL_DEG >= 0.005, (
        "angle tolerance too tight to absorb real optimised slab cells")
    assert LAYER_GROUP_LENGTH_RTOL >= 1e-4, (
        "length tolerance too tight to absorb real optimised slab cells")

    # Genuinely wrong lattices must still be refused, or the guard is inert.
    for a, b, gamma, why in [
        (2.47, 2.47, 90.0, "gamma=90 is not a hexagonal lattice"),
        (2.47, 3.90, 120.0, "a != b is not a hexagonal lattice"),
    ]:
        msg = check_layer_group_cell(HEX, a, b, gamma)
        assert msg is not None, f"should have refused: {why}"
        assert "hexagonal" in msg and "a = b and gamma = 120" in msg, msg


def _refusal_text(dimensionality, spacegroup, capsys):
    """Drive create_d12_file to an origin-freedom refusal and return what the
    user actually sees."""
    import NewCifToD12 as M

    cif = {
        "a": 2.47, "b": 2.47, "c": 20.0,
        "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        "spacegroup": spacegroup,
        "atomic_numbers": [6], "symbols": ["C"], "positions": [[0.1, 0.1, 0.1]],
    }
    opts = {
        "dimensionality": dimensionality, "calculation_type": "SP",
        "method": "DFT", "functional": "PBE", "dft_functional": "PBE",
        "basis_set": "POB-TZVP-REV2", "basis_set_type": "INTERNAL",
        "dft_grid": "XLGRID", "use_dispersion": False, "dispersion": False,
        "is_spin_polarized": False, "spin_polarized": False,
        "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
        "shrink": [4, 4], "k_points": 4, "scf_maxcycle": 100, "fmixing": 30,
        "scf_method": "DIIS", "symmetry_handling": "CIF",
        "optimization_settings": {}, "freq_settings": {},
    }
    capsys.readouterr()
    result = M.create_d12_file(cif, "/tmp/_refusal_probe.d12", opts)
    captured = capsys.readouterr()
    return result, captured.out + captured.err


def test_refusal_messages_are_not_malformed(capsys, tmp_path, monkeypatch):
    """Guards the prose of the refusal, not just the return value.

    A doubled "or" shipped in the SLAB refusal twice - it was fixed once and
    came back when the surrounding block was rewritten, because every test
    asserted on the return value and none on the text the user reads. This is
    the cheapest possible guard against that class.
    """
    monkeypatch.chdir(tmp_path)
    # Space group 191 maps to layer group 80, which is NOT in
    # LAYER_GROUPS_POLAR_IN_Z, so this reaches the origin-freedom refusal.
    # 191 is also the deck that originally MPI_Aborted inside CRYSTAL.
    # Asserting the refusal really fires keeps this test from going vacuous the
    # way test_real_slab_decks_pass_the_lattice_check silently did.
    result, text = _refusal_text("SLAB", 191, capsys)
    assert result is False, (
        "space group 191 must reach the origin-freedom refusal; if it stops "
        "doing so this test is no longer checking any message at all")

    lowered = text.lower()
    for dup in (" or or ", " the the ", " a a ", " and and ", " in in "):
        assert dup not in lowered, f"refusal contains '{dup.strip()}': {text}"
    # The message must still offer both escape routes it promises.
    assert "p1" in lowered, text
    assert "slabcut" in lowered, text
