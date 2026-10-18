"""A multi-point OPT stores no value of its first geometry as final.

CRYSTAL prints an OPT's SCF in full only at point 1: the per-cycle
INSULATING/CONDUCTING blocks with their DIRECT/INDIRECT gap lines, the
EFERMI lines of a conducting state and the ``+++ ENERGIES IN A.U. +++``
components. From point 2 on it prints only the energy, and after OPT END a
summary for the final geometry (``BAND GAP`` or ``ALPHA/BETA BAND GAP``, or
``FERMI ENERGY``). The extractor took direct/indirect_band_gap,
band_gap_type, fermi_energy and the energy components from point 1 of a
multi-point OPT and stored them as the run's. Only what follows the last
optimization point counts now; a one-point OPT and an SP are unchanged.
"""
from conftest import find_data


def _props(extractor, pattern):
    content = find_data(pattern).read_text(errors="ignore")
    return {**extractor._band_gap_properties(content),
            **extractor._extract_computational_properties(content),
            **extractor._extract_energy_components(content)}


GAP_KIND = ("direct_band_gap", "indirect_band_gap", "band_gap_type")


def test_spin_multi_point_opt_keeps_only_the_final_summary(extractor):
    p = _props(extractor, "OPT/3.4^2T137_BULK_OPTGEOM_TZ.out")
    # point 1's INDIRECT 2.6 eV gap is not the final geometry's
    assert not any(k in p for k in GAP_KIND)
    assert (p["alpha_band_gap"], p["beta_band_gap"]) == (2.5678, 2.5678)
    assert "kinetic_energy_au" not in p


def test_closed_shell_multi_point_opt_gap_from_summary(extractor):
    p = _props(extractor, "OPT/1_dia_opt_BULK_OPTGEOM.out")
    assert p["band_gap"] == 6.0010
    assert not any(k in p for k in GAP_KIND)


def test_conducting_multi_point_opt_has_no_point_one_fermi(extractor):
    p = _props(extractor, "OPT/Ti11O14_mp-759754_sg1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
                          "triplezeta_opt_B3LYP-D3-D3_optimized.out")
    # its EFERMI lines (last 1.5519687E-02) are point 1's; none after OPT END
    assert "fermi_energy_au" not in p


def test_multi_point_opt_final_fermi_energy_kept(extractor):
    p = _props(extractor, "OPT/3,4^2T1-CA_BULK_OPTGEOM_TZ.out")
    assert p["fermi_energy_au"] == -1.1373763E-01


def test_one_point_opt_keeps_its_scf_values(extractor):
    p = _props(extractor, "OPT/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_"
                          "HSESOL3C_optimized.out")
    assert p["direct_band_gap"] == 7.8402 and p["band_gap_type"] == "direct"
    assert "kinetic_energy_au" in p


def test_multi_point_opt_ending_conducting_after_gapped_point_one(extractor):
    # point 1 had gapped cycles, the final geometry is conducting (FERMI ENERGY
    # after OPT END): 0, as for a run that ended conducting
    p = _props(extractor, "OPT/Ag1Cl5_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_"
                          "B3LYP-D3-D3_optimized.out")
    # a spin run (ALPHA/BETA ELECTRONS blocks; its last point-1 block is common)
    assert p["band_gap"] == 0.0 and p["spin_polarized"] is True
    assert p["fermi_energy_au"] == -2.5299136E-01
