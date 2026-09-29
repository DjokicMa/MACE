"""Invariants of the preset and lookup tables in Crystal_d12/d12_constants.py.

The deck writers, menus and planner all read these tables, and a wrong entry
ends up in a CRYSTAL23 deck without any error. The corpus tests would catch
some of that by comparing against real decks, but they skip in CI. These
check the tables against each other and against the writer, with no data
files. The exact OPT/SCF preset values are pinned in
tests/test_convergence_presets.py; this file checks how the tables fit
together.
"""
import io
import re
from collections import Counter

import pytest

import d12_constants as C
from d12_writer import write_dft_section


def _dft_keyword(functional, use_dispersion):
    buf = io.StringIO()
    write_dft_section(buf, functional, use_dispersion, "XLGRID", False)
    return buf.getvalue().splitlines()[1]


# --------------------------------------------------------------------------
# Convergence presets
# --------------------------------------------------------------------------

def test_preset_levels_are_the_same_three_menu_keys():
    assert list(C.OPT_CONVERGENCE_PRESETS) == ["1", "2", "3"]
    assert list(C.SCF_TOLERANCE_PRESETS) == ["1", "2", "3"]
    assert C.FREQ_SCF_LEVEL in C.SCF_TOLERANCE_PRESETS


def test_opt_presets_tighten_with_each_level():
    p = [C.OPT_CONVERGENCE_PRESETS[k] for k in ("1", "2", "3")]
    assert p[0]["toldeg"] > p[1]["toldeg"] > p[2]["toldeg"]
    assert p[0]["toldex"] > p[1]["toldex"] > p[2]["toldex"]
    assert p[0]["toldee"] < p[1]["toldee"] < p[2]["toldee"]
    # TOLDEX is 4 x TOLDEG at every level (CRYSTAL's own default ratio)
    for preset in p:
        assert preset["toldex"] == pytest.approx(4 * preset["toldeg"])


def test_scf_presets_tighten_with_each_level():
    ints = [list(map(int, C.SCF_TOLERANCE_PRESETS[k]["TOLINTEG"].split()))
            for k in ("1", "2", "3")]
    for values in ints:
        assert len(values) == 5          # TOLINTEG takes five integers
    for looser, tighter in zip(ints, ints[1:]):
        assert all(t >= l for l, t in zip(looser, tighter))
        assert tighter != looser
    toldee = [C.SCF_TOLERANCE_PRESETS[k]["TOLDEE"] for k in ("1", "2", "3")]
    assert toldee == sorted(toldee) and len(set(toldee)) == 3


@pytest.mark.parametrize("level,expected", [
    ("1", "TOLDEG=0.0003, TOLDEX=0.0012, TOLDEE=7, MAXCYCLE=800"),
    ("2", "TOLDEG=0.0001, TOLDEX=0.0004, TOLDEE=8, MAXCYCLE=800"),
    ("3", "TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=9, MAXCYCLE=800"),
])
def test_describe_opt_preset(level, expected):
    assert C.describe_opt_preset(level) == expected


@pytest.mark.parametrize("level,expected", [
    ("1", "TOLINTEG: 7 7 7 7 14, TOLDEE: 7"),
    ("2", "TOLINTEG: 8 8 8 9 24, TOLDEE: 9"),
    ("3", "TOLINTEG: 9 9 9 11 38, TOLDEE: 11"),
])
def test_describe_scf_preset(level, expected):
    assert C.describe_scf_preset(level) == expected


@pytest.mark.parametrize("value,expected", [
    (0.0003, "0.0003"), (0.00003, "0.00003"), (0.0012, "0.0012"),
    (1.0, "1"), (0.5, "0.5"), (0.0000001, "0"),
])
def test_format_opt_tolerance(value, expected):
    assert C.format_opt_tolerance(value) == expected


def test_opt_and_scf_helpers_return_fresh_dicts():
    a = C.scf_tolerances("1")
    a["TOLDEE"] = 99
    assert C.SCF_TOLERANCE_PRESETS["1"]["TOLDEE"] == 7
    b = C.opt_convergence("1")
    b["toldee"] = 99
    assert C.OPT_CONVERGENCE_PRESETS["1"]["toldee"] == 7


def test_freq_default_tolerances_copies_the_parent_values():
    parent = {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}
    got = C.freq_default_tolerances("FREQ", parent)
    assert got == parent and got is not parent


# --------------------------------------------------------------------------
# Default settings
# --------------------------------------------------------------------------

def test_default_settings_aliases_agree():
    d = C.DEFAULT_SETTINGS
    assert d["method"] == d["method_type"] == "DFT"
    assert d["functional"] == d["dft_functional"]
    assert d["dispersion"] is d["use_dispersion"]
    assert d["spin_polarized"] is d["is_spin_polarized"]
    assert d["maxcycle"] == d["scf_maxcycle"]
    assert d["optimization_type"] == C.DEFAULT_OPT_SETTINGS["type"]
    assert d["optimization_settings"] is C.DEFAULT_OPT_SETTINGS
    assert d["tolerances"] is C.DEFAULT_TOLERANCES


def test_default_settings_name_valid_table_entries():
    d = C.DEFAULT_SETTINGS
    assert d["dft_grid"] in C.DFT_GRIDS.values()
    assert d["basis_set_type"] == "INTERNAL"
    assert d["basis_set"] in C.INTERNAL_BASIS_SETS
    assert d["optimization_type"] in C.OPT_TYPES.values()
    assert d["calculation_type"] in ("OPT", "SP", "FREQ")
    # the default functional with the default dispersion is a real keyword
    assert d["functional"] in C.D3_FUNCTIONALS
    assert C.crystal23_functional_keyword(_dft_keyword(d["functional"], d["dispersion"])) == "HSE06-D3"
    assert C.DEFAULT_OPT_SETTINGS["convergence"] == C.OPT_CONVERGENCE_PRESETS["1"]["name"]


# --------------------------------------------------------------------------
# Grid and optimization keyword tables
# --------------------------------------------------------------------------

def test_dft_grids_table():
    assert C.DFT_GRIDS == {
        "1": "OLDGRID", "2": "DEFAULT", "3": "LGRID", "4": "XLGRID",
        "5": "XXLGRID", "6": "XXXLGRID", "7": "HUGEGRID",
    }


def test_opt_types_are_exactly_the_optgeom_type_keywords():
    assert set(C.OPT_TYPES.values()) == set(C.OPTGEOM_TYPE_KEYWORDS)
    assert len(C.OPTGEOM_TYPE_KEYWORDS) == len(set(C.OPTGEOM_TYPE_KEYWORDS))
    assert C.DEFAULT_OPT_SETTINGS["type"] == C.OPT_TYPES["1"] == "FULLOPTG"


def test_freq_template_copies_agree():
    """d12_calc_freq keeps its own FREQ_TEMPLATES, which the FREQ menu reads.
    The d12_constants copy must name the same templates, and every setting it
    holds must match; the d12_calc_freq copy only adds band-path keys."""
    import d12_calc_freq

    assert set(C.FREQ_TEMPLATES) == set(d12_calc_freq.FREQ_TEMPLATES)
    for name, template in C.FREQ_TEMPLATES.items():
        other = d12_calc_freq.FREQ_TEMPLATES[name]
        for key, value in template.items():
            if isinstance(value, dict):
                assert value.items() <= other[key].items(), (name, key)
            else:
                assert other[key] == value, (name, key)


# --------------------------------------------------------------------------
# Functional tables
# --------------------------------------------------------------------------

ALL_MENU_FUNCTIONALS = [f for cat in C.FUNCTIONAL_CATEGORIES.values() for f in cat["functionals"]]
DFT_MENU_FUNCTIONALS = [f for name, cat in C.FUNCTIONAL_CATEGORIES.items() if name != "HF"
                        for f in cat["functionals"]]

# The menus accept B97 only with D3: CRYSTAL23 rejects a bare B97 keyword.
D3_ONLY = {"B97"}


def test_every_menu_functional_is_listed_once_and_described():
    assert not [f for f, n in Counter(ALL_MENU_FUNCTIONALS).items() if n > 1]
    for name, cat in C.FUNCTIONAL_CATEGORIES.items():
        assert set(cat["functionals"]) == set(cat["descriptions"]), name


def test_hf_category_is_the_crystal23_hf_methods():
    assert C.FUNCTIONAL_CATEGORIES["HF"]["functionals"] == C.CRYSTAL23_HF_METHODS


@pytest.mark.parametrize("category", ["3C", "HF"])
def test_basis_requirements_name_only_that_categorys_methods(category):
    cat = C.FUNCTIONAL_CATEGORIES[category]
    assert set(cat["basis_requirements"]) <= set(cat["functionals"])


def test_every_3c_method_has_a_required_basis():
    cat = C.FUNCTIONAL_CATEGORIES["3C"]
    assert set(cat["basis_requirements"]) == set(cat["functionals"])


@pytest.mark.parametrize("functional", [f for f in DFT_MENU_FUNCTIONALS if f not in D3_ONLY])
def test_menu_functional_is_written_as_a_crystal23_keyword(functional):
    """Without D3, the writer's DFT keyword line for every menu functional is
    either a keyword CRYSTAL23 accepts or the start of an EXCHANGE/CORRELAT pair."""
    keyword = _dft_keyword(functional, use_dispersion=False)
    assert keyword == "EXCHANGE" or C.crystal23_functional_keyword(keyword) is not None, keyword


@pytest.mark.parametrize("functional", C.D3_FUNCTIONALS)
def test_d3_functional_is_written_as_a_crystal23_d3_keyword(functional):
    assert functional in ALL_MENU_FUNCTIONALS
    keyword = _dft_keyword(functional, use_dispersion=True)
    assert keyword.endswith("-D3")
    assert C.crystal23_functional_keyword(keyword) == keyword


def test_d3_functionals_cover_every_crystal23_d3_base():
    """CRYSTAL23's PW1PW-D3 is offered as mPW1PW91 with D3."""
    written = {_dft_keyword(f, True)[:-3] for f in C.D3_FUNCTIONALS}
    assert written == set(C.CRYSTAL23_D3_KEYWORD_BASES)


@pytest.mark.parametrize("functional", ALL_MENU_FUNCTIONALS)
def test_mace_functional_name_round_trips_case_insensitively(functional):
    assert C.mace_functional_name(functional.lower()) == functional
    assert C.mace_functional_name(functional.upper() + "-d3") == functional + "-D3"


def test_crystal23_functional_keyword_spellings():
    assert C.crystal23_functional_keyword(" hsesol-d3 ") == "HSEsol-D3"
    assert C.crystal23_functional_keyword("pw1pw-d3") == "PW1PW-D3"
    assert C.crystal23_functional_keyword("mPW1PW91-D3") is None
    assert C.crystal23_functional_keyword("") is None
    assert C.crystal23_functional_keyword(None) is None
    for method in C.CRYSTAL23_HF_METHODS:
        assert C.crystal23_functional_keyword(method) is None
        assert C.crystal23_functional_keyword(method.lower(), allow_hf=True) == method


def test_standalone_keyword_list_has_no_duplicates():
    upper = [k.upper() for k in C.CRYSTAL23_STANDALONE_FUNCTIONALS]
    assert len(upper) == len(set(upper))


def test_custom_functional_label_is_not_a_keyword():
    assert C.crystal23_functional_keyword(C.CUSTOM_FUNCTIONAL, allow_hf=True) is None
    assert C.mace_functional_name(C.CUSTOM_FUNCTIONAL) is None


def test_describe_custom_functional_pairs_records_with_their_values():
    records = ["EXCHANGE", "PBE", "CORRELAT", "PBE", "HYBRID", "25", "NONLOCAL", "0.1 0.1"]
    assert C.describe_custom_functional(records) == (
        "custom: EXCHANGE PBE / CORRELAT PBE / HYBRID 25 / NONLOCAL 0.1 0.1")
    assert C.describe_custom_functional(["PBE0"]) == "custom: PBE0"
    assert C.describe_custom_functional(None) == "custom: "


def test_unrecognised_functional_fallback_is_a_keyword():
    fb = C.UNRECOGNISED_FUNCTIONAL_FALLBACK
    assert C.crystal23_functional_keyword(fb) == fb
    assert C.mace_functional_name(fb) == fb


# --------------------------------------------------------------------------
# Basis-set tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(C.INTERNAL_BASIS_SETS))
def test_internal_basis_all_electron_and_ecp_partition_the_elements(name):
    entry = C.INTERNAL_BASIS_SETS[name]
    ae, ecp = set(entry["all_electron"]), set(entry["ecp_elements"])
    assert not ae & ecp
    assert ae | ecp == set(entry["elements"])
    assert all(1 <= z <= 118 for z in entry["elements"])


def test_ecp_elements_external_table():
    ecp = C.ECP_ELEMENTS_EXTERNAL
    assert ecp == sorted(set(ecp))
    assert min(ecp) == 37 and max(ecp) == 99
    # full-core in the external sets, per the table's own note
    assert not {36, 43, 54, 86} & set(ecp)


# --------------------------------------------------------------------------
# Element and space-group tables
# --------------------------------------------------------------------------

def test_element_symbols_cover_1_to_118_and_invert():
    assert sorted(C.ELEMENT_SYMBOLS) == list(range(1, 119))
    assert len(set(C.ELEMENT_SYMBOLS.values())) == 118
    assert all(C.SYMBOL_TO_NUMBER[s] == z for z, s in C.ELEMENT_SYMBOLS.items())
    assert (C.ELEMENT_SYMBOLS[6], C.ELEMENT_SYMBOLS[82], C.ELEMENT_SYMBOLS[118]) == ("C", "Pb", "Og")


def test_spacegroup_symbols_cover_1_to_230_and_invert():
    assert sorted(C.SPACEGROUP_SYMBOLS) == list(range(1, 231))
    assert len(set(C.SPACEGROUP_SYMBOLS.values())) == 230
    assert all(C.SPACEGROUP_SYMBOL_TO_NUMBER[s] == n for n, s in C.SPACEGROUP_SYMBOLS.items())
    assert C.SPACEGROUP_SYMBOLS[227] == "Fd-3m"


def test_every_spacegroup_has_a_crystal_output_spelling():
    """CRYSTAL prints the space group with spaces ('F D 3 M'); every number
    must be reachable from that form."""
    spaced = {n for s, n in C.SPACEGROUP_ALTERNATIVES.items() if " " in s}
    assert spaced == set(range(1, 231))


def test_alternatives_point_at_valid_space_groups():
    assert all(1 <= n <= 230 for n in C.SPACEGROUP_ALTERNATIVES.values())


def test_alternatives_never_contradict_the_canonical_symbol():
    clashes = {s: (n, C.SPACEGROUP_SYMBOL_TO_NUMBER[s])
               for s, n in C.SPACEGROUP_ALTERNATIVES.items()
               if s in C.SPACEGROUP_SYMBOL_TO_NUMBER and C.SPACEGROUP_SYMBOL_TO_NUMBER[s] != n}
    assert clashes == {}


def test_rhombohedral_groups_are_the_r_centred_groups():
    r_groups = sorted(n for n, s in C.SPACEGROUP_SYMBOLS.items() if s.startswith("R"))
    assert C.RHOMBOHEDRAL_SPACEGROUPS == r_groups


def test_multi_origin_table_entries_match_the_symbol_table():
    for n, info in C.MULTI_ORIGIN_SPACEGROUPS.items():
        assert info["name"].replace("_", "") == C.SPACEGROUP_SYMBOLS[n], n
        assert info["crystal_code"] == "0 0 0"
        assert re.fullmatch(r"[01] [01] [01]", info["alt_crystal_code"]), n
        assert info["alt_crystal_code"] != info["crystal_code"]


def test_spacegroup_to_path_covers_every_group_with_a_known_path():
    assert sorted(C.SPACEGROUP_TO_PATH) == list(range(1, 231))
    assert set(C.SPACEGROUP_TO_PATH.values()) <= set(C.HIGH_SYMMETRY_PATHS)
    assert C.SPACEGROUP_TO_PATH[225] == "cubic_fc"
    assert C.SPACEGROUP_TO_PATH[229] == "cubic_bc"
    assert C.SPACEGROUP_TO_PATH[221] == "cubic_simple"


def test_cubic_path_follows_the_lattice_centring():
    for n in range(195, 231):
        letter = C.SPACEGROUP_SYMBOLS[n][0]
        expected = {"F": "cubic_fc", "I": "cubic_bc", "P": "cubic_simple"}[letter]
        assert C.SPACEGROUP_TO_PATH[n] == expected, (n, C.SPACEGROUP_SYMBOLS[n])


# --------------------------------------------------------------------------
# High-symmetry path tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", [k for k, p in C.HIGH_SYMMETRY_PATHS.items()
                                 if "coord_path" in p])
def test_coord_path_is_the_label_path_on_one_integer_scale(key):
    """coord_path is label_path with each point's fractional coordinates
    multiplied by one common factor (the IS that DISPERSION uses)."""
    p = C.HIGH_SYMMETRY_PATHS[key]
    assert len(p["coord_path"]) == len(p["label_path"])
    factors = set()
    for segment, ints in zip(p["label_path"], p["coord_path"]):
        start, end = segment.split()
        frac = p["coordinates"][start] + p["coordinates"][end]
        for f, i in zip(frac, ints):
            assert isinstance(i, int)
            if f == 0:
                assert i == 0
            else:
                factors.add(i / f)
    assert len(factors) == 1, factors


@pytest.mark.parametrize("key", [k for k in C.HIGH_SYMMETRY_PATHS if k != "cubic_simple"])
def test_labels_chain_is_the_label_path(key):
    p = C.HIGH_SYMMETRY_PATHS[key]
    assert [f"{a} {b}" for a, b in zip(p["labels"], p["labels"][1:])] == p["label_path"]


def test_simple_cubic_label_path_adds_the_m_r_branch():
    p = C.HIGH_SYMMETRY_PATHS["cubic_simple"]
    chain = [f"{a} {b}" for a, b in zip(p["labels"], p["labels"][1:])]
    assert p["label_path"] == chain + ["M R"]


@pytest.mark.parametrize("key", list(C.HIGH_SYMMETRY_PATHS))
def test_every_path_label_has_coordinates(key):
    p = C.HIGH_SYMMETRY_PATHS[key]
    missing = {lab for seg in p["label_path"] for lab in seg.split()} - set(p["coordinates"])
    assert missing == set()
    assert p["coordinates"]["G"] == [0.0, 0.0, 0.0]
