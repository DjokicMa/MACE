"""Corpus-free deck-writing tests for Crystal_d12/d12_writer.py.

The corpus tests pin the writer against real decks under ``test/``, which CI
does not have. These pin the same emitters with inline inputs, asserting the
exact lines and their order, so a change to what goes into a CRYSTAL23 deck
fails in CI too. A full deck assembled from the emitters is read back with
the project's own input parser, so written and read settings must agree.

write_method_block and write_scf_block are not covered here.
"""
import io

import pytest

from d12_writer import (
    DEFAULT_BROYDEN_IMIX,
    DEFAULT_BROYDEN_ISTART,
    DEFAULT_BROYDEN_W0,
    DEFAULT_SPINLOCK_CYCLES,
    write_basis_block,
    write_dft_section,
    write_frequency_block,
    write_k_points,
    write_minimal_raman_section,
    write_optimization_block,
    write_print_options,
    write_properties_block,
    write_scf_section,
    write_smearing_settings,
    write_spin_settings,
)

STANDARD_TOL = {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}


def _lines(writer, *args, **kwargs):
    buf = io.StringIO()
    writer(buf, *args, **kwargs)
    return buf.getvalue().splitlines()


def _scf(**kwargs):
    params = dict(
        tolerances=dict(STANDARD_TOL), k_points=(8, 8, 8), dimensionality="CRYSTAL",
        use_smearing=False, smearing_width=0.0, scf_method="DIIS", scf_maxcycle=800,
        fmixing=30, num_atoms=2, spacegroup=227,
    )
    params.update(kwargs)
    return _lines(write_scf_section, **params)


def _dft(functional, use_dispersion=False, dft_grid="DEFAULT", spin=False, **kwargs):
    return _lines(write_dft_section, functional, use_dispersion, dft_grid, spin, **kwargs)


# --------------------------------------------------------------------------
# write_scf_section: record order and the branches not pinned elsewhere
# --------------------------------------------------------------------------

def test_scf_section_full_record_order():
    """Every optional record at once, in the order the writer emits them:
    SPINLOCK leads, then tolerances, SHRINK, SMEAR, GUESSP, SCFDIR, buffers,
    MAXCYCLE/FMIXING, the method and its HISTDIIS, LEVSHIFT, PPAN, END."""
    lines = _scf(spinlock=2, guessp=True, use_smearing=True, smearing_width=0.005,
                 num_atoms=8, levshift=(5, 1), spacegroup=1, k_points=(4, 6, 8),
                 scf_maxcycle=200, fmixing=40)
    assert lines == [
        "SPINLOCK", f"2 {DEFAULT_SPINLOCK_CYCLES}",
        "TOLINTEG", "7 7 7 7 14",
        "TOLDEE", "7",
        "SHRINK", "0 16", "4 6 8",
        "SMEAR", "0.005000",
        "GUESSP",
        "SCFDIR",
        "BIPOSIZE", "110000000",
        "EXCHSIZE", "110000000",
        "MAXCYCLE", "200",
        "FMIXING", "40",
        "DIIS",
        "HISTDIIS", "100",
        "LEVSHIFT", "5 1",
        "PPAN",
        "END",
    ]


def test_spinlock_cycles_are_caller_overridable():
    assert _scf(spinlock=-1, spinlock_cycles=30)[:2] == ["SPINLOCK", "-1 30"]


def test_smear_width_is_written_with_six_decimals_after_shrink():
    lines = _scf(use_smearing=True, smearing_width=0.01)
    i = lines.index("SMEAR")
    assert lines[i + 1] == "0.010000"
    assert lines[i - 2:i] == ["SHRINK", "8 16"]
    assert lines[i + 2] == "SCFDIR"


def test_no_smear_record_without_smearing():
    assert "SMEAR" not in _scf(use_smearing=False, smearing_width=0.01)


def test_preformatted_kpoint_string_is_written_after_a_fixed_0_24_line():
    """A string mesh is written verbatim as the second SHRINK line, and the
    first line is always '0 24' whatever the string holds."""
    lines = _scf(k_points="12 12 12")
    i = lines.index("SHRINK")
    assert lines[i:i + 3] == ["SHRINK", "0 24", "12 12 12"]


@pytest.mark.parametrize("k_points,dimensionality", [
    ((4, 4, 4), "MOLECULE"),
    (None, "CRYSTAL"),
    ((), "CRYSTAL"),
])
def test_no_shrink_for_molecules_or_without_a_mesh(k_points, dimensionality):
    lines = _scf(k_points=k_points, dimensionality=dimensionality)
    assert "SHRINK" not in lines
    assert lines[:5] == ["TOLINTEG", "7 7 7 7 14", "TOLDEE", "7", "SCFDIR"]


@pytest.mark.parametrize("dimensionality,k_points,spacegroup,expected", [
    # P1 always takes the two-line directional form, ISP = 2 * max
    ("CRYSTAL", (8, 8, 8), 1, ["0 16", "8 8 8"]),
    ("CRYSTAL", (4, 6, 8), 1, ["0 16", "4 6 8"]),
    ("SLAB", (6, 4, 9), 1, ["0 18", "6 4 1"]),
    ("POLYMER", (6, 3, 3), 1, ["0 12", "6 1 1"]),
    # non-P1 slab / polymer keep the directional form, the periodic-only mesh
    ("SLAB", (6, 4, 1), 11, ["0 12", "6 4 1"]),
    ("POLYMER", (6, 6, 6), 4, ["0 12", "6 1 1"]),
    # non-P1 bulk with a uniform mesh: the one-line IS ISP form
    ("CRYSTAL", (6, 6, 6), 225, ["6 12"]),
])
def test_shrink_forms_by_dimensionality_and_symmetry(dimensionality, k_points,
                                                     spacegroup, expected):
    lines = _scf(dimensionality=dimensionality, k_points=k_points, spacegroup=spacegroup)
    i = lines.index("SHRINK")
    assert lines[i + 1:i + 1 + len(expected)] == expected
    assert lines[i + 1 + len(expected)] == "SCFDIR"


def test_non_uniform_bulk_mesh_is_made_uniform_and_says_so(capsys):
    lines = _scf(k_points=(8, 8, 4), spacegroup=139)
    i = lines.index("SHRINK")
    assert lines[i + 1:i + 3] == ["8 16", "SCFDIR"]
    out = capsys.readouterr().out
    assert "Converting non-uniform k-points (8,8,4) to uniform (8,8,8) for space group 139" in out


@pytest.mark.parametrize("tolerances", [
    {},
    {"TOLINTEG": None, "TOLDEE": None},
])
def test_missing_or_none_tolerances_fall_back_to_standard(tolerances):
    assert _scf(tolerances=tolerances)[:4] == ["TOLINTEG", "7 7 7 7 14", "TOLDEE", "7"]


def test_tight_tolerances_are_written_verbatim():
    assert _scf(tolerances={"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11})[:4] == [
        "TOLINTEG", "9 9 9 11 38", "TOLDEE", "11"]


@pytest.mark.parametrize("num_atoms,present", [(5, False), (6, True)])
def test_buffer_sizes_switch_on_above_five_atoms(num_atoms, present):
    lines = _scf(num_atoms=num_atoms)
    assert ("BIPOSIZE" in lines) is present
    assert ("EXCHSIZE" in lines) is present


def test_parent_exchsize_alone_suppresses_the_default_biposize():
    lines = _scf(num_atoms=10, exchsize=200)
    i = lines.index("SCFDIR")
    assert lines[i:i + 4] == ["SCFDIR", "EXCHSIZE", "200", "MAXCYCLE"]
    assert "BIPOSIZE" not in lines


@pytest.mark.parametrize("histdiis", [None, 0])
def test_falsy_histdiis_writes_no_record(histdiis):
    lines = _scf(histdiis=histdiis)
    assert lines[lines.index("DIIS") + 1] == "PPAN"


def test_histdiis_is_only_written_for_diis():
    lines = _scf(scf_method="ANDERSON", histdiis=100)
    assert "HISTDIIS" not in lines
    assert lines[-5:] == ["FMIXING", "30", "ANDERSON", "PPAN", "END"]


def test_broyden_default_record_matches_module_constants():
    lines = _scf(scf_method="BROYDEN")
    assert lines[lines.index("BROYDEN") + 1] == (
        f"{DEFAULT_BROYDEN_W0} {DEFAULT_BROYDEN_IMIX} {DEFAULT_BROYDEN_ISTART}")
    assert (DEFAULT_BROYDEN_W0, DEFAULT_BROYDEN_IMIX, DEFAULT_BROYDEN_ISTART) == (0.0001, 50, 2)


def test_levshift_uses_only_the_first_two_values():
    lines = _scf(levshift=(8, 0, 99))
    assert lines[lines.index("LEVSHIFT") + 1] == "8 0"


# --------------------------------------------------------------------------
# write_dft_section: branches not pinned elsewhere
# --------------------------------------------------------------------------

def test_spin_follows_dft_and_precedes_the_functional():
    assert _dft("B3LYP", use_dispersion=True, dft_grid="XLGRID", spin=True) == [
        "DFT", "SPIN", "B3LYP-D3", "XLGRID", "ENDDFT"]


def test_dispersion_flag_is_ignored_for_functionals_without_d3_parameters():
    assert _dft("SVWN", use_dispersion=True) == ["DFT", "SVWN", "ENDDFT"]
    assert _dft("PBESOL", use_dispersion=True, dft_grid=None) == ["DFT", "PBESOLXC", "ENDDFT"]


@pytest.mark.parametrize("grid", ["DEFAULT", None, ""])
def test_default_or_empty_grid_writes_no_grid_record(grid):
    assert _dft("HSE06", use_dispersion=True, dft_grid=grid) == ["DFT", "HSE06-D3", "ENDDFT"]


def test_mpw1pw91_is_written_as_pw1pw_d3_only_with_dispersion():
    assert _dft("mPW1PW91", use_dispersion=True, dft_grid="LGRID") == [
        "DFT", "PW1PW-D3", "LGRID", "ENDDFT"]
    assert _dft("mPW1PW91", use_dispersion=False, dft_grid="LGRID") == [
        "DFT", "mPW1PW91", "LGRID", "ENDDFT"]


def test_d3_suffixed_names_are_written_as_given():
    assert _dft("PBE0-D3", dft_grid="XLGRID") == ["DFT", "PBE0-D3", "XLGRID", "ENDDFT"]
    # Not a D3 base: passed through untouched (the menus refuse it upstream).
    assert _dft("SCAN-D3", dft_grid="XLGRID") == ["DFT", "SCAN-D3", "XLGRID", "ENDDFT"]


@pytest.mark.parametrize("functional", ["PBEH3C", "HSE3C", "B973C", "PBESOL03C"])
def test_3c_methods_take_no_d3_suffix_and_skip_a_default_grid(functional):
    assert _dft(functional, use_dispersion=True, spin=True) == [
        "DFT", "SPIN", functional, "ENDDFT"]
    assert _dft(functional, dft_grid="XXLGRID") == ["DFT", functional, "XXLGRID", "ENDDFT"]


def test_parent_dftd3_block_replaces_the_d3_suffix_after_enddft():
    assert _dft("HSE06", use_dispersion=True, dft_grid="XLGRID",
                custom_dftd3=["DFTD3", "END"]) == [
        "DFT", "HSE06", "XLGRID", "ENDDFT", "DFTD3", "END"]


def test_parent_dftd3_block_inside_the_dft_block():
    assert _dft("HSE06", use_dispersion=True, dft_grid="XLGRID",
                custom_dftd3=["DFTD3", "END"], custom_dftd3_in_dft=True) == [
        "DFT", "HSE06", "XLGRID", "DFTD3", "END", "ENDDFT"]


def test_custom_functional_without_records_is_refused():
    from d12_constants import CUSTOM_FUNCTIONAL
    with pytest.raises(ValueError, match="EXCHANGE/CORRELAT"):
        _dft(CUSTOM_FUNCTIONAL, custom_functional=[])


def test_custom_functional_writes_records_then_grid():
    from d12_constants import CUSTOM_FUNCTIONAL
    records = ["EXCHANGE", "PBE", "CORRELAT", "PBE", "HYBRID", "25"]
    assert _dft(CUSTOM_FUNCTIONAL, dft_grid="XLGRID", spin=True,
                custom_functional=records) == ["DFT", "SPIN", *records, "XLGRID", "ENDDFT"]


# --------------------------------------------------------------------------
# A whole deck from the emitters, read back by the input parser
# --------------------------------------------------------------------------

MGO_GEOMETRY = (
    "MgO_rocksalt\n"
    "CRYSTAL\n"
    "0 0 0\n"
    "225\n"
    "4.21\n"
    "2\n"
    "12 0.0 0.0 0.0\n"
    "8 0.5 0.5 0.5\n"
    "END\n"
)


def _mgo_deck():
    buf = io.StringIO()
    buf.write(MGO_GEOMETRY)
    write_basis_block(buf, {"basis_set_type": "INTERNAL", "basis_set": "POB-TZVP-REV2"}, {})
    write_dft_section(buf, "HSE06", True, "XLGRID", True)
    write_scf_section(buf, {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}, (6, 6, 6), "CRYSTAL",
                      True, 0.005, "DIIS", 200, 40, 2, 225, levshift=(5, 1))
    return buf.getvalue()


def test_assembled_deck_is_byte_exact():
    assert _mgo_deck() == MGO_GEOMETRY + (
        "BASISSET\nPOB-TZVP-REV2\n"
        "DFT\nSPIN\nHSE06-D3\nXLGRID\nENDDFT\n"
        "TOLINTEG\n8 8 8 9 24\nTOLDEE\n9\n"
        "SHRINK\n6 12\n"
        "SMEAR\n0.005000\n"
        "SCFDIR\n"
        "MAXCYCLE\n200\nFMIXING\n40\n"
        "DIIS\nHISTDIIS\n100\n"
        "LEVSHIFT\n5 1\n"
        "PPAN\nEND\n"
    )


def test_assembled_deck_reads_back_with_the_settings_written(tmp_path):
    from d12_parsers import CrystalInputParser

    deck = tmp_path / "MgO_rocksalt.d12"
    deck.write_text(_mgo_deck())
    d = CrystalInputParser(str(deck)).parse()

    assert d["dimensionality"] == "CRYSTAL"
    assert d["spacegroup"] == 225
    assert d["basis_set_type"] == "INTERNAL"
    assert d["basis_set"] == "POB-TZVP-REV2"
    assert d["method"] == "DFT"
    assert d["functional"] == "HSE06-D3"
    assert d["dispersion"] is True
    assert d["dft_grid"] == "XLGRID"
    assert d["spin_polarized"] is True
    assert d["tolerances"] == {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}
    assert d["k_points"] == "6 12"
    assert d["use_smearing"] is True
    assert d["smearing_width"] == pytest.approx(0.005)
    assert d["scf_method"] == "DIIS"
    assert d["scf_maxcycle"] == 200
    assert d["fmixing"] == 40
    assert d["scf_settings"]["histdiis"] == 100
    assert tuple(d["scf_settings"]["levshift"]) == (5, 1)


# --------------------------------------------------------------------------
# Smaller emitters
# --------------------------------------------------------------------------

def test_minimal_raman_section_is_exactly_the_cphf_block():
    assert _lines(write_minimal_raman_section) == [
        "FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "ENDCPHF", "ENDFREQ"]


def test_internal_basis_block_names_the_set():
    assert _lines(write_basis_block, {"basis_set_type": "INTERNAL", "basis_set": "STO-3G"},
                  {"elements": [6, 1]}) == ["BASISSET", "STO-3G"]
    # defaults: an internal POB-TZVP
    assert _lines(write_basis_block, {}, {}) == ["BASISSET", "POB-TZVP"]


def test_external_basis_block_concatenates_files_in_z_order_and_ends_with_99_0(tmp_path):
    """External files are named by atomic number, ECP elements (Z >= 37) by
    Z + 200; each unique element's file is written once, in Z order."""
    (tmp_path / "1").write_text("1 2\n0 0 3 1.0 1.0\n")
    (tmp_path / "8").write_text("8 3\n0 0 6 2.0 1.0\n")
    (tmp_path / "282").write_text("282 5\nINPUT\n")
    lines = _lines(write_basis_block,
                   {"basis_set_type": "EXTERNAL", "basis_set": str(tmp_path)},
                   {"elements": [82, 8, 1, 8, 1]})
    assert lines == ["1 2", "0 0 3 1.0 1.0", "8 3", "0 0 6 2.0 1.0",
                     "282 5", "INPUT", "99 0"]


def test_external_basis_block_skips_a_missing_element_file(tmp_path, capsys):
    (tmp_path / "6").write_text("6 4\n")
    lines = _lines(write_basis_block,
                   {"basis_set_type": "EXTERNAL", "basis_set": str(tmp_path)},
                   {"elements": [6, 9]})
    assert lines == ["6 4", "99 0"]
    assert "element 9" in capsys.readouterr().out


@pytest.mark.parametrize("k_points,dimensionality,expected", [
    (8, "CRYSTAL", ["SHRINK", "8 8"]),
    (8, "SLAB", ["SHRINK", "0 8", "8 8 1"]),
    ([6, 12], "CRYSTAL", ["SHRINK", "6 12"]),
    ([12, 6, 4], "SLAB", ["SHRINK", "0 12", "6 4 1"]),
    (8, "MOLECULE", []),
    ([6, 12, 3], "CRYSTAL", []),        # wrong arity for the dimensionality
])
def test_write_k_points(k_points, dimensionality, expected):
    assert _lines(write_k_points, k_points, dimensionality) == expected


@pytest.mark.xfail(strict=True, reason=(
    "write_k_points writes a one-number SHRINK record for POLYMER, but CRYSTAL's "
    "SHRINK always takes IS and ISP on its first record"))
@pytest.mark.parametrize("k_points", [8, [8]])
def test_write_k_points_polymer_has_both_is_and_isp(k_points):
    lines = _lines(write_k_points, k_points, "POLYMER")
    assert lines[0] == "SHRINK"
    assert len(lines[1].split()) == 2


@pytest.mark.parametrize("level,expected", [
    (0, ["PRINTOUT", "PRINT", "0"]),
    (1, []),
    (2, ["PRINTOUT", "PRINT", "2"]),
    (3, ["PRINTOUT", "PRINT", "3"]),
])
def test_write_print_options(level, expected):
    assert _lines(write_print_options, level) == expected


def test_write_spin_settings():
    assert _lines(write_spin_settings, False, 2) == []
    assert _lines(write_spin_settings, True) == ["UHF"]
    assert _lines(write_spin_settings, True, 0) == ["UHF", "SPINLOCK", "0"]
    assert _lines(write_spin_settings, True, 2) == ["UHF", "SPINLOCK", "2"]


def test_write_smearing_settings():
    assert _lines(write_smearing_settings, {}) == []
    assert _lines(write_smearing_settings, {"enabled": False, "width": 0.02}) == []
    assert _lines(write_smearing_settings, {"enabled": True}) == ["SMEAR", "0.01"]
    assert _lines(write_smearing_settings, {"enabled": True, "width": 0.005}) == ["SMEAR", "0.005"]


@pytest.mark.parametrize("calc_type,options,expected", [
    ("SP", {"calculate_bands": True, "calculate_dos": True, "mulliken_analysis": True},
     ["BAND", "DOSS", "PPAN"]),
    ("OPT", {"calculate_bands": True, "calculate_dos": True}, []),
    ("OPT", {"mulliken_analysis": True}, ["PPAN"]),
    ("FREQ", {}, []),
])
def test_write_properties_block(calc_type, options, expected):
    assert _lines(write_properties_block, calc_type, options) == expected


def test_write_frequency_block_delegates_to_the_freq_writer():
    from d12_calc_freq import write_frequency_calculation
    settings = {"mode": "GAMMA", "intensities": False}
    assert _lines(write_frequency_block, settings) == _lines(write_frequency_calculation, settings)
    assert _lines(write_frequency_block, {}) == ["FREQCALC", "NOINTENS", "ENDFREQ"]


@pytest.mark.xfail(strict=True, raises=ModuleNotFoundError, reason=(
    "write_optimization_block imports write_optimization_section from "
    "d12_calc_opt, which does not exist (the function lives in d12_calc_basic)"))
def test_write_optimization_block_writes_an_optgeom_block():
    lines = _lines(write_optimization_block, {"type": "FULLOPTG"})
    assert lines[0] == "OPTGEOM"

