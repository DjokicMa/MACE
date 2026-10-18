"""Property-extractor fixes from the 1.1.6 pre-release review (reviewer C, F3).

Real corpus outputs throughout; each test skips when the gitignored test/
corpus is missing.
"""
import pytest

from conftest import find_data


# --- V1: closed-shell Mulliken populations ---------------------------------

CLOSED_SHELL = "OPT/1_dia_opt_BULK_OPTGEOM.out"   # RESTRICTED CLOSED SHELL, PPAN


def test_closed_shell_mulliken_keeps_continuation_lines(extractor):
    content = find_data(CLOSED_SHELL).read_text(errors="replace")
    assert "RESTRICTED CLOSED SHELL" in content
    m = extractor._extract_general_mulliken_section(content)
    assert m["total_electrons"] == 12.0
    assert [a["atom_number"] for a in m["atoms"]] == [1, 2]
    c1 = m["atoms"][0]
    assert c1["element"] == "C" and c1["mulliken_population"] == 6.0
    # 8 on the row, 8 on the first continuation line, 2 on the second
    assert c1["orbital_populations"] == [
        0.853, 1.120, 0.754, 0.391, 0.451, 0.451, 0.451, -0.012,
        -0.012, -0.012, 0.506, 0.506, 0.506, 0.004, 0.012, 0.012,
        0.004, 0.012]


def test_closed_shell_mulliken_takes_the_last_print(extractor):
    content = find_data(CLOSED_SHELL).read_text(errors="replace")
    i = content.index("MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS")
    j = content.index("OVERLAP POPULATION", i) if "OVERLAP POPULATION" in content[i:] else len(content)
    block = content[i:j]
    # an earlier print (e.g. an earlier geometry) with other charges, before the real one
    earlier = block.replace("6.000  0.853", "5.900  0.753")
    assert earlier != block
    m = extractor._extract_general_mulliken_section(earlier + "\n OVERLAP POPULATION\n" + content)
    assert m["atoms"][0]["mulliken_population"] == 6.0
    assert m["atoms"][0]["orbital_populations"][0] == 0.853


def test_closed_shell_parsers_agree(extractor):
    # CRYSTAL23 heads a closed-shell PPAN with "ALPHA+BETA ELECTRONS" too, so
    # the stored rows come from the spin-section parser; the general parser
    # (outputs without that header) must give the same atoms and populations
    content = find_data(CLOSED_SHELL).read_text(errors="replace")
    props = extractor._extract_population_analysis(content)
    stored = props["mulliken_alpha_plus_beta"]["atoms"]
    general = extractor._extract_general_mulliken_section(content)["atoms"]
    assert general == stored


# --- V2: spin_polarized of an unrestricted run that ends conducting --------

@pytest.mark.parametrize("pattern", [
    "SP/4LG_FSI_TopMiddle_2x2_ABAB_graphene_opt_sp.out",
    "SP/Ti19O30_mp-760075_*_sp_B3LYP-D3-D3_optimized.out",
])
def test_unrestricted_run_ending_conducting_is_spin_polarized(extractor, pattern):
    content = find_data(pattern).read_text(errors="replace")
    assert "UNRESTRICTED OPEN SHELL" in content
    props = extractor._band_gap_properties(content)
    assert props.get("spin_polarized") is True


# --- V3: validation accepts the classification the extractor writes -------
# (1.1.6: conductivity_type, metallic/non_metallic, is merged into
# electronic_classification)

def test_validation_accepts_extractor_classifications():
    from mace.database.utils.validation import PropertyValidator
    v = PropertyValidator()
    for value in ("metal", "semimetal", "semiconductor", "insulator"):
        ok, errors = v.validate_property("electronic_classification", value)
        assert ok, errors
    assert not v.validate_property("electronic_classification", "conductor")[0]
    issues = v._validate_cross_properties({"band_gap": 2.0, "electronic_classification": "metal"})
    assert any("Metal with non-zero band gap" in i for i in issues)
    issues = v._validate_cross_properties({"band_gap": 1.0, "electronic_classification": "insulator"})
    assert any("Insulator with small band gap" in i for i in issues)
    # values with their unit, as validate_database passes them
    issues = v._validate_cross_properties({"band_gap": {"value": 3.0, "unit": "eV"},
                                           "electronic_classification": "insulator"})
    assert any("Insulator with small band gap" in i for i in issues)
    assert not v._validate_cross_properties({"band_gap": 3.0, "electronic_classification": "semiconductor"})


def test_extractor_classification_is_valid(extractor):
    from mace.database.utils.validation import PropertyValidator
    content = find_data("SP/4LG_FSI_TopMiddle_2x2_ABAB_graphene_opt_sp.out").read_text(errors="replace")
    props = extractor._band_gap_properties(content)
    value = extractor._classify_electronic_properties(props, content)["electronic_classification"]
    assert value == "metal"
    assert PropertyValidator().validate_property("electronic_classification", value)[0]


# --- V4: fermi_energy is stored in Hartree ----------------------------------

def test_fermi_energy_default_unit_is_hartree(extractor):
    from mace.database.utils.units import get_default_unit
    assert get_default_unit("fermi_energy") == "hartree"
    assert get_default_unit("fermi_energy_au") == "hartree"   # the 1.1.6 name


# --- V8: the DOS at E_F (dos_at_fermi_total since 1.1.6) is both spins -----

def _data(name):
    from pathlib import Path
    return Path(__file__).parent / "data" / "spin_fe" / name


def test_fe_dos_total_at_fermi_holds_both_spins(extractor):
    from mace.utils.dat_file_processor import DatFileProcessor
    dat = _data("fe_doss.DOSS.DAT")
    info = DatFileProcessor().process_doss_dat_file(dat)
    summary, scalars = extractor._build_dos_summary(info, "", dat.with_name("fe_doss.out"), dat)
    alpha, beta = scalars["dos_at_fermi_alpha"], scalars["dos_at_fermi_beta"]
    # ferromagnetic Fe: both spins have states at E_F, so the total is well
    # above the alpha channel the key used to hold (reviewer C, F3-V8)
    assert alpha > 0.1 and beta > 0.1
    assert scalars["dos_at_fermi_total"] == pytest.approx(alpha + beta, abs=1e-9)
    assert summary["total_dos_at_fermi"] == scalars["dos_at_fermi_total"]


# --- V6: no effective masses or mobilities for a metal ----------------------

def test_fe_metal_gets_no_effective_masses():
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    r = AdvancedElectronicAnalyzer().analyze_material(_data("fe_band.BAND.DAT"),
                                                       _data("fe_doss.DOSS.DAT"))
    assert r["electronic_classification"] in ("metal", "semimetal")
    assert r["electron_effective_mass"] is None and r["hole_effective_mass"] is None
    assert r["has_real_effective_mass"] is False
    assert "electron_mobility_estimate" not in r and "hole_mobility_estimate" not in r


def test_band_crossing_without_dos_gets_no_effective_masses():
    # BAND.DAT alone: the bands overlap E_F (CBM below VBM), so no masses even
    # though there is no DOS to name it metal or semimetal
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    r = AdvancedElectronicAnalyzer().analyze_material(_data("fe_band.BAND.DAT"), None)
    assert r["electron_effective_mass"] is None and r["hole_effective_mass"] is None


def test_semiconductor_keeps_effective_masses():
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    band = find_data("BAND/1_dia_opt_rev1_sp*_band.BAND.DAT")
    r = AdvancedElectronicAnalyzer().analyze_material(band, None)
    assert r["electron_effective_mass"] == pytest.approx(1.61, abs=0.05)
    assert r["hole_effective_mass"] == pytest.approx(0.48, abs=0.05)


# --- V5: a BAND.DAT is read once per extraction -----------------------------

def test_band_dat_read_once_per_extraction(monkeypatch, tmp_path):
    import builtins
    import shutil
    from mace.utils import dat_text
    from mace.utils.property_extractor import CrystalPropertyExtractor
    out = find_data("BAND/1_dia_opt_rev1_sp*_band.out")
    # a fresh copy: the read cache is keyed by path
    for src in out.parent.glob(out.stem + "*"):
        if src.is_file():
            shutil.copy(src, tmp_path / src.name)
    dat_text._text.cache_clear()
    reads = []
    real_open = builtins.open

    def counting_open(f, *a, **k):
        if str(f).endswith(".BAND.DAT"):
            reads.append(str(f))
        return real_open(f, *a, **k)

    monkeypatch.setattr(builtins, "open", counting_open)
    props = CrystalPropertyExtractor(db_path=None, enable_tracking=False) \
        .extract_all_properties(tmp_path / out.name)
    assert props.get("band_dat_exists") is True
    assert len(reads) == 1, reads


# --- S2: the effective mass is fitted in the spin block that holds the edge --

def test_hole_mass_uses_the_block_holding_the_vbm():
    import numpy as np
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    a = AdvancedElectronicAnalyzer()
    nk = 41
    k = np.linspace(-0.2, 0.2, nk)
    # alpha VBM 0.1 eV (0.0037 Ha) below beta's, with another curvature
    alpha_vb = -0.0037 - k ** 2 / (2 * 2.0)      # m* = 2.0
    beta_vb = -k ** 2 / (2 * 0.5)                # m* = 0.5, the VBM
    cb = 0.2 + k ** 2 / (2 * 0.3)
    alpha = np.column_stack([alpha_vb, cb])
    beta = np.column_stack([beta_vb, cb])
    energies = np.vstack([alpha, beta])
    kk = np.concatenate([k, k])
    a._band_layout = {"nkpt": nk, "nspin": 2, "panel_ends": [1, nk], "nrows": 2 * nk}
    m = a._calculate_band_curvature(kk, energies, float(beta_vb.max()), "hole")
    assert m == pytest.approx(0.5, rel=1e-3)
