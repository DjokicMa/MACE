"""X_pob_TZVP_rev2_SOC basis records: data and loader only.

The sets on crystal.unito.it pair the pob-TZVP-rev2 valence shells with
CRYSTAL's internal STUTSC library, the fully relativistic
Stuttgart-Cologne small-core ECP (CRYSTAL23 manual p.84), for two-component
SCF with spin-orbit coupling (pp.169-175). They are not an ordinary external
basis: no path, menu entry or mace_config key leads to them, only
basis_library.soc_basis_record(z). The SOC deck code uses them.
"""
import pytest

import basis_library as BL
from conftest import REPO_ROOT

STATED = list(range(37, 43)) + list(range(44, 54)) + [55, 56] + list(range(72, 84))


def test_coverage_is_the_stated_set():
    lib = BL.load_family(BL.SOC_FAMILY)
    assert sorted(BL.soc_basis_elements() + [int(z) for z in lib.missing]) == STATED


@pytest.mark.parametrize("z", STATED)
def test_each_record_uses_stutsc_and_cites_the_paper(z):
    if z not in BL.soc_basis_elements():
        pytest.skip(BL.load_family(BL.SOC_FAMILY).missing[str(z)])
    record = BL.soc_basis_record(z)
    lines = record.split("\n")
    assert lines[0].split()[0] == str(200 + z)
    assert lines[1] == "STUTSC"
    assert BL.cut_record(record) == (record, "")
    lib = BL.load_family(BL.SOC_FAMILY)
    key = str(200 + z)
    assert "10.1002/jcc.26839" in lib.field(key, "doi")
    assert "10.1002/jcc.26839" in lib.field(key, "reference")[0]
    assert lib.field(key, "source")[0].lower().endswith("_pob_tzvp_rev2_soc")


def test_the_valence_shells_are_the_rev2_ones_on_stutsc():
    """Each SOC record is the rev2 record's shells (rev2022 family: Rb-I from
    2022, Cs/Ba/Hf-Bi as in full.basis.triplezeta) with the scalar INPUT ECP
    replaced by STUTSC - equal numbers, checked for every element."""
    rev = BL.load_family("full.basis.triplezeta.rev2022")
    for z in BL.soc_basis_elements():
        soc = BL.numeric_tokens(BL.soc_basis_record(z))
        ref = BL.numeric_tokens(rev.record(str(200 + z)))
        assert ref[1] == ["INPUT"]
        n_ecp = 2 + int(sum(ref[2][1:]))           # INPUT, header, M+M0+..+M4 terms
        assert soc[0] == ref[0] and soc[2:] == ref[1 + n_ecp:], z


def test_it_is_never_an_ordinary_external_basis():
    import mace_config
    from d12_config import EXTERNAL_BASIS_FAMILIES

    path = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / BL.SOC_FAMILY)
    assert BL.bundled_family_for(path) is None
    assert BL.SOC_FAMILY not in BL.FAMILIES
    assert not BL.path_exists(path)
    assert BL.SOC_FAMILY not in EXTERNAL_BASIS_FAMILIES.values()
    assert not [n for n in dir(mace_config) if "SOC" in n.upper()]
    assert mace_config.get_basis_path("soc") == mace_config.BASIS_DOUBLEZETA_DIR  # the default


def test_an_element_without_a_record_is_refused():
    with pytest.raises(KeyError):
        BL.soc_basis_record(43)     # Tc: no X_pob_TZVP_rev2_SOC
    with pytest.raises(KeyError):
        BL.soc_basis_record(84)     # Po is on the site but not in MACE's TZ set
