"""Deck-level conformance of d12_constants/d12_calc_basic with the CRYSTAL23 manual.

Three defects are pinned here, all of the "CRYSTAL reads the deck, but reads
something other than what was meant" kind:

* the optimization-type menu offered keywords CRYSTAL does not have (INTONLY,
  ITATOCELL) -- manual sec. 7.3.1 lists ATOMONLY / FULLOPTG / CELLONLY /
  ITATOCEL / INTREDUN and nothing else;
* the SLAB cell line always printed three values, but the layer group's 2D
  lattice type fixes how many the minimal set has (manual SLAB record
  "a,[b],[gamma]", b for rectangular only, angle for oblique only; Appendix A.2
  partitions layer groups 1-7 oblique, 8-48 rectangular, 49-64 square,
  65-80 hexagonal);
* a rhombohedral space group written in rhombohedral axes needs "a alpha"
  (IFHR=1), not "a c" (IFHR=0) -- manual CRYSTAL record, IFHR, and note 13.

Plus the internal basis-set menu, which filtered on a guess (max_z <= 36 or a
name whitelist) and therefore hid POB-TZVP-REV2, the only internal set that
carries Pb -- the same lead-perovskite bug tests/test_basis_coverage.py was
written for, one layer higher up.

Manual line numbers below refer to the plain-text CRYSTAL23 manual. Where a real
deck exists under ``test/`` it is asserted against in preference to a fixture;
the manual's own worked examples are used only where the corpus has no such
structure (no rhombohedral-axes deck exists in it).
"""
import re
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_calc_basic  # noqa: E402
import d12_constants  # noqa: E402
from d12_constants import (  # noqa: E402
    RHOMBOHEDRAL_SPACEGROUPS,
    generate_unit_cell_line,
)


def _deck(relpath):
    """Real deck from the gitignored ``test/`` corpus, or skip."""
    path = TEST_DATA / relpath
    if not path.is_file():
        pytest.skip(f"test/ corpus file not present: {relpath}")
    return path.read_text(errors="ignore").splitlines()


# ============================================================
# Optimization type (manual sec. 7.3.1, lines 14275-14296)
# ============================================================

# Manual 14275-14296 enumerates the type-of-optimization keywords: ATOMONLY
# ("Only atomic coordinates are optimized. This was the default before
# Crystal14"), FULLOPTG, CELLONLY, ITATOCEL, INTREDUN. Note that the keyword
# index block at manual 2257-2268 is NOT exhaustive -- it omits ATOMONLY,
# because ATOMONLY is the historical default -- so it must not be used to argue
# that a keyword does not exist.
EXPECTED_OPT_TYPES = {
    "1": "FULLOPTG",
    "2": "CELLONLY",
    "3": "ATOMONLY",
    "4": "ITATOCEL",
    "5": "CVOLOPT",
}


def test_opt_types_are_real_crystal_keywords():
    """Slot 3 wrote INTONLY (d12_constants) / ITATOCELL (d12_calc_basic).

    Neither string appears in the CRYSTAL23 manual. CRYSTAL stops on the
    unrecognised keyword inside OPTGEOM, so every "optimize internal
    coordinates only" deck the menu produced was dead on arrival.
    """
    assert d12_constants.OPT_TYPES == EXPECTED_OPT_TYPES
    assert d12_calc_basic.OPT_TYPES == EXPECTED_OPT_TYPES


def test_the_two_opt_types_copies_do_not_drift():
    """d12_calc_basic.OPT_TYPES shadows d12_constants.OPT_TYPES.

    The live menu (d12_calc_basic:195) reads the local copy, so the two
    disagreeing is how slot 3 stayed broken in one file after being "fixed" in
    the other. Pin them together.
    """
    assert d12_calc_basic.OPT_TYPES == d12_constants.OPT_TYPES


def test_opt_menu_labels_name_the_keyword_they_write():
    """The printed menu is the only place the user sees the keyword.

    Slot 3 advertised "ITATOCELL - Optimize only internal coordinates
    iteratively", which is both a non-keyword and the wrong scope: ATOMONLY
    relaxes atoms at fixed cell (manual 14275-14277), while an iterative
    atoms-cell procedure is ITATOCEL (manual 14287-14289).
    """
    import inspect

    source = inspect.getsource(d12_calc_basic._configure_optimization_impl)
    for number, keyword in d12_calc_basic.OPT_TYPES.items():
        assert f"{number}. {keyword}" in source, (
            f"menu line for option {number} does not name {keyword}"
        )
    assert "fixed cell" in source, "ATOMONLY's scope must be stated in the menu"


def test_no_phantom_optimization_keywords_remain():
    """Guard against either invented keyword coming back in live code.

    Comments are stripped: both names are still mentioned there, deliberately,
    to say why they are gone.
    """
    for module in (d12_constants, d12_calc_basic):
        code = "\n".join(
            line.split("#")[0] for line in open(module.__file__).read().splitlines()
        )
        assert "INTONLY" not in code, f"{module.__name__} still names INTONLY"
        assert "ITATOCELL" not in code, f"{module.__name__} still names ITATOCELL"


def test_slot3_keyword_survives_a_deck_round_trip(tmp_path):
    """OPT -> OPT2/SP progression re-reads the deck it just wrote.

    d12_parsers only recognises FULLOPTG/CVOLOPT/CELLONLY/ATOMONLY, so a slot-3
    deck written with INTONLY/ITATOCELL (or INTREDUN) is silently dropped on the
    next step of the chain and falls back to a full optimization -- the cell
    moves in a step the user pinned it for.
    """
    from d12_parsers import CrystalInputParser

    deck = tmp_path / "atomonly.d12"
    with open(deck, "w") as f:
        print("test", file=f)
        print("CRYSTAL", file=f)
        print("0 0 0", file=f)
        print("1", file=f)
        print("5.00000000 5.00000000 5.00000000 90.000000 90.000000 90.000000",
              file=f)
        print("1", file=f)
        print("6 0.0 0.0 0.0", file=f)
        settings = dict(d12_calc_basic.DEFAULT_OPT_SETTINGS)
        settings["type"] = d12_calc_basic.OPT_TYPES["3"]
        d12_calc_basic.write_optimization_section(
            f, settings["type"], settings
        )
        print("END", file=f)

    parsed = CrystalInputParser(str(deck)).parse()
    assert parsed["optimization_settings"].get("type") == "ATOMONLY"


# ============================================================
# SLAB cell line (manual 999-1006; Appendix A.2, 33477-33697)
# ============================================================


def test_hexagonal_layer_group_takes_only_a():
    """Manual test05, graphite 2D (29026-29029): layer group 77, cell line "2.47".

    The old code printed "2.47000000 2.47000000 120.000000"; CRYSTAL then reads
    the atom-count line as the tail of the cell record and the whole deck
    shifts by a line.
    """
    line = generate_unit_cell_line(77, [2.47, 2.47, 20.0, 90, 90, 120], "SLAB")
    assert line == "2.47000000"


def test_square_layer_group_takes_only_a():
    """Manual 29130-29136, CO on MgO(001): layer group 55, cell line "2.97692"."""
    line = generate_unit_cell_line(55, [2.97692, 2.97692, 20.0, 90, 90, 90], "SLAB")
    assert line == "2.97692000"


def test_rectangular_layer_group_takes_a_and_b():
    """Manual 29099-29101, MgO(110) 2 layers: layer group 40, "4.21 2.97692".

    Manual 999-1006: "(b for rectangular lattices only)" and the angle is
    "triclinic lattices only", so no gamma here.
    """
    line = generate_unit_cell_line(40, [4.21, 2.97692, 20.0, 90, 90, 90], "SLAB")
    assert line == "4.21000000 2.97692000"


def test_oblique_layer_group_keeps_all_three():
    """Manual 29079-29082, Corundum 110 slab: layer group 7 (Oblique),
    "5.129482 6.997933 95.8395" -- three values, the only case that has them."""
    line = generate_unit_cell_line(
        7, [5.129482, 6.997933, 20.0, 90, 90, 95.8395], "SLAB"
    )
    assert line == "5.12948200 6.99793300 95.839500"


def test_layer_group_out_of_range_is_refused():
    """Only 80 layer groups exist (Appendix A.2). A 3D space group number
    arriving here is a bug upstream, not a cell line to guess at."""
    with pytest.raises(ValueError):
        generate_unit_cell_line(194, [2.47, 2.47, 20.0, 90, 90, 120], "SLAB")


@pytest.mark.parametrize(
    "relpath",
    [
        "SP/4LG_FSI_2x2_AA_opt_sp.d12",
        "SP/4LG_FSI_2x2_ABAB_opt_sp.d12",
        "OPT/4LG_FSI_TopMiddle_2x2_ABAB_opt.d12",
    ],
)
def test_real_slab_decks_are_byte_identical(relpath):
    """Every production slab deck in test/ uses layer group 1 (oblique), which
    still takes a, b and gamma. The layer-group dispatch must not perturb them."""
    lines = _deck(relpath)
    assert lines[1].strip() == "SLAB"
    layer_group = int(lines[2].strip())
    values = [float(v) for v in lines[3].split()]
    assert len(values) == 3, f"{relpath} is not the oblique 3-value form"
    a, b, gamma = values

    regenerated = generate_unit_cell_line(
        layer_group, [a, b, 500.0, 90.0, 90.0, gamma], "SLAB"
    )
    assert regenerated == lines[3].strip()


# ============================================================
# Rhombohedral cell line / IFHR (manual 885-935, note 13 at 1202-1206)
# ============================================================


def test_rhombohedral_axes_emit_a_and_alpha():
    """Manual 28869-28876, corundum in the rhombohedral representation:
    IFHR=1, space group 167, cell line "5.12948 55.29155" = a, alpha.

    The old code returned "5.12948000 5.12948000" -- it printed a where CRYSTAL
    reads alpha, so the deck describes a 5.13 degree cell. That is a silently
    wrong structure, not a crash.
    """
    line = generate_unit_cell_line(
        167,
        [5.12948, 5.12948, 5.12948, 55.29155, 55.29155, 55.29155],
        "CRYSTAL",
    )
    assert line == "5.12948000 55.291550"


def test_hexagonal_axes_of_a_rhombohedral_group_are_untouched():
    """Manual 28855-28861, the same corundum with IFHR=0: "4.7602 12.9933" = a, c.

    The hexagonal test runs first, exactly as in
    NewCifToD12.detect_trigonal_setting, so this path is unchanged.
    """
    line = generate_unit_cell_line(
        167, [4.7602, 4.7602, 12.9933, 90.0, 90.0, 120.0], "CRYSTAL"
    )
    assert line == "4.76020000 12.99330000"


def test_axes_can_be_stated_explicitly():
    """AUTO conversions know their own IFHR; they must be able to say so rather
    than have it re-inferred from the numbers."""
    params = [5.12948, 5.12948, 5.12948, 55.29155, 55.29155, 55.29155]
    assert generate_unit_cell_line(
        167, params, "CRYSTAL", use_rhombohedral_axes=True
    ) == "5.12948000 55.291550"
    hexagonal = [4.7602, 4.7602, 12.9933, 90.0, 90.0, 120.0]
    assert generate_unit_cell_line(
        167, hexagonal, "CRYSTAL", use_rhombohedral_axes=False
    ) == "4.76020000 12.99330000"


def test_only_rhombohedral_groups_can_switch_axes():
    """Manual 885-935 introduces IFHR as "meaningless for non-rhombohedral
    crystals"; note 13 (1202-1206) names 146-148-155-160-161-166-167. A trigonal
    but non-rhombohedral group such as 156 (P3m1) has the hexagonal cell only,
    so its a=b=c-looking cell must not be re-read as a, alpha."""
    assert RHOMBOHEDRAL_SPACEGROUPS == [146, 148, 155, 160, 161, 166, 167]
    assert 156 not in RHOMBOHEDRAL_SPACEGROUPS
    line = generate_unit_cell_line(
        156, [3.0, 3.0, 3.0, 80.0, 80.0, 80.0], "CRYSTAL"
    )
    assert line == "3.00000000 3.00000000"


@pytest.mark.parametrize(
    "relpath",
    [
        "SP/Ag2Br3_mp-862982_sg167_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
        "triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized.d12",
        "SP/2_dia2_opt_rev1_sp_B3LYP-D3-D3_optimized.d12",
        "OPT/Ti6O_mp-554098_sg163_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
        "triplezeta_opt_B3LYP-D3-D3_optimized.d12",
    ],
)
def test_real_trigonal_decks_are_byte_identical(relpath):
    """The corpus's trigonal decks (space groups 166, 167 and 163) are all
    IFHR=0 hexagonal cells with a two-value "a c" line. They must stay so."""
    lines = _deck(relpath)
    assert lines[1].strip() == "CRYSTAL"
    assert lines[2].split()[1] == "0", "expected IFHR=0 (hexagonal cell)"
    spacegroup = int(lines[3].strip())
    a, c = [float(v) for v in lines[4].split()]

    regenerated = generate_unit_cell_line(
        spacegroup, [a, a, c, 90.0, 90.0, 120.0], "CRYSTAL"
    )
    assert regenerated == lines[4].strip()


def test_rhombohedral_deck_matches_the_output_it_came_from():
    """End-to-end on real data: the CRYSTALLOGRAPHIC CELL block of a finished
    space-group-167 run, fed through the cell-line generator, must reproduce the
    .d12 that MACE actually wrote for the follow-on SP."""
    stem = (
        "SP/Ag2Br3_mp-862982_sg167_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
        "triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"
    )
    deck_lines = _deck(stem + ".d12")
    out_path = TEST_DATA / (stem + ".out")
    if not out_path.is_file():
        pytest.skip("companion .out not present")

    text = out_path.read_text(errors="ignore")
    match = re.search(
        r"CRYSTALLOGRAPHIC CELL \(VOLUME=[^)]*\)\s*\n\s*A\s+B\s+C\s+ALPHA\s+"
        r"BETA\s+GAMMA\s*\n\s*(.+)",
        text,
    )
    assert match, "no CRYSTALLOGRAPHIC CELL block in the reference output"
    params = [float(v) for v in match.group(1).split()[:6]]

    assert generate_unit_cell_line(167, params, "CRYSTAL") == deck_lines[4].strip()


# ============================================================
# Internal basis-set menu
# ============================================================


def _select_internal(elements, monkeypatch):
    """Drive select_basis_set as a user who picks INTERNAL and presses Enter.

    Returns (basis_config, menu_options, offered_default).
    """
    seen = {}

    def fake_get_user_input(prompt, options, default):
        if "internal basis set" in prompt.lower():
            seen["options"] = dict(options)
            seen["default"] = default
            return default  # pressing Enter, exactly as get_user_input would
        return "2"  # basis set type: INTERNAL

    monkeypatch.setattr(d12_constants, "get_user_input", fake_get_user_input)
    config = d12_constants.select_basis_set(elements)
    return config, seen.get("options", {}), seen.get("default")


def test_lead_structure_is_offered_the_basis_that_covers_lead(monkeypatch, capsys):
    """The menu filtered on max_z <= 36 or a POB-DZVP/POB-TZVP name whitelist.

    For a Pb compound that leaves exactly POB-DZVP and POB-TZVP -- neither of
    which has Pb (measured: VERIFIED_INTERNAL_BASIS_ELEMENTS) -- while hiding
    POB-TZVP-REV2, the one internal set that does. Same lead-perovskite failure
    as tests/test_basis_coverage.py, one layer up in the UI.
    """
    _config, options, _default = _select_internal([6, 8, 53, 82], monkeypatch)
    offered = list(options.values())
    assert "POB-TZVP-REV2" in offered
    for name in ("POB-DZVP", "POB-TZVP", "STO-3G"):
        assert name not in offered, f"{name} does not cover Pb but was offered"


def test_filtered_menu_default_is_a_valid_option(monkeypatch, capsys):
    """The default was the literal "7".

    get_user_input returns the default verbatim on an empty line without
    checking membership (d12_constants, "if choice == ... and default"), so on
    any menu shorter than seven entries the caller raised KeyError and the whole
    conversion aborted.
    """
    config, options, default = _select_internal([6, 8, 53, 82], monkeypatch)
    assert len(options) < 7, "this case must produce a short menu"
    assert default in options
    assert config["basis_set"] == options[default]


def test_common_case_menu_is_unchanged(monkeypatch, capsys):
    """An organic (H/C/O) structure must see the same 13-entry menu and the same
    default as before: this fix may not move the common path."""
    config, options, default = _select_internal([1, 6, 8], monkeypatch)
    assert list(options.values()) == list(d12_constants.INTERNAL_BASIS_SETS)
    assert default == "7"
    assert config["basis_set"] == "POB-TZVP-REV2"
    assert config["basis_set_type"] == "INTERNAL"


def test_no_internal_basis_falls_back_to_external(monkeypatch, capsys):
    """Francium (87) is outside every internal set's measured coverage.

    Previously the filter still produced a two-entry menu of sets that do not
    have it, then died on the "7" default. Point the user at EXTERNAL instead.
    """
    config, options, _default = _select_internal([6, 87], monkeypatch)
    assert options == {}, "no internal menu should be offered"
    assert config["basis_set_type"] == "EXTERNAL"
    assert config["basis_set"]
    assert config["basis_set"] == config["basis_set_path"]
    assert "No internal basis set" in capsys.readouterr().out
