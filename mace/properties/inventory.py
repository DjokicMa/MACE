"""The property inventory: every key the database stores (the extractor's own,
mace/utils/property_keys.py, and the derived properties of the registry), by
calculation type, with its unit, where it comes from and its database key;
and what the GUI computes or shows that the database does not have.

``mace analyze --list-properties`` prints it; docs/properties.md is its
markdown (``mace analyze --list-properties --update-doc docs/properties.md``
rewrites the generated part; tests/test_properties_inventory.py checks the
document is current).

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.utils import property_keys

SCF = ("OPT", "SP", "FREQ", "ELASTIC", "EOS", "QHA", "CPHF")
MECHANICS = ("ELASTIC", "EOS", "QHA")
CALC_TYPES = ("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS",
              "ELASTIC", "EOS", "QHA", "CPHF")
EVERY = CALC_TYPES

# category -> (calculation types, where the value comes from)
_BY_CATEGORY = {
    "lattice": (SCF, ".out: the final geometry (an OPT's last point; the input geometry otherwise)"),
    "structural": (SCF, ".out: geometry and NEIGHBORS tables"),
    "crystallographic": (SCF, ".out: SPACE GROUP / layer group lines"),
    "energy": (SCF, ".out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.)"),
    "population_analysis": (SCF, ".out: Mulliken population tables (PPAN)"),
    "optimization": (("OPT",), ".out: OPTGEOM convergence"),
    "frequency": (("FREQ",), ".out: FREQCALC mode table"),
    "thermodynamics": (("FREQ",), ".out: the first THERMODYNAMIC FUNCTIONS block (AT T, P)"),
    "band_structure": (("BAND",), "<run>.BAND.DAT (+ the .out and .d3: k path, Fermi level)"),
    "density_of_states": (("DOSS",), "<run>.DOSS.DAT (+ the .out: projections, Fermi level)"),
    "transport": (("TRANSPORT",), "SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM)"),
    "charge_potential": (("CHARGE+POTENTIAL",), ".out and the ECH3/POT3 .CUBE files"),
    "orbitals": (("ORBITALS",), "the ORBITALS .molden files (+ the .out)"),
    "scf_settings": (SCF, ".out: the SCF and DFT settings echoed"),
    "computational": (EVERY, ".out: timings and the run's type"),
    "mechanical": (MECHANICS, ".out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis"),
    "electronic": (SCF, ".out: the last SCF"),
}

# per-key exceptions (calculation types, source)
_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "fermi_energy_au": {"calc_types": SCF + ("BAND", "DOSS", "TRANSPORT"),
                        "source": ".out: the last SCF's EFERMI, else the FERMI ENERGY a BAND or DOSS run prints"},
    "fermi_energy_note": {"calc_types": ("BAND", "DOSS"),
                          "source": "mace.utils.fermi_level, when the .out prints CRYSTAL's -1.0 placeholder: the band "
                                    "data's derived E_F, else the DOSS.DAT's"},
    "spin_polarized": {"calc_types": EVERY, "source": ".out: UNRESTRICTED OPEN SHELL, alpha/beta tables, SPIN cube"},
    "electronic_classification": {"calc_types": SCF + ("BAND", "DOSS", "TRANSPORT"),
                                  "source": "computed: metal / semiconductor / insulator from the gap "
                                            "(property_keys.classify_gap); a gapless BAND run with a DOSS run "
                                            "beside it: metal or semimetal by N(E_F) per atom "
                                            "(property_keys.classify_gapless)"},
    "band_gap": {"source": ".out: the last SCF's gap (BAND.DAT/DOSS.DAT gaps are band_fundamental_gap_ev, "
                           "dos_band_gap_from_dos_ev)"},
    "enthalpy_au": {"source": "computed: H = G + TS from the printed columns"},
    "enthalpy_ev": {"source": "computed: H = G + TS from the printed columns"},
    "enthalpy_kj_mol": {"source": "computed: H = G + TS from the printed columns"},
    "zero_point_energy_au": {"source": ".out: E0 of the harmonic thermodynamics"},
    "zero_point_energy_ev": {"source": ".out: E0 of the harmonic thermodynamics"},
    "zero_point_energy_kj_mol": {"source": ".out: E0 of the harmonic thermodynamics"},
    "electron_count": {"source": "computed: electrons of a charged run (ion_fragments)"},
    "fragment_charges": {"source": "computed: Mulliken charge and spin per molecular fragment (ion_fragments)"},
    "dielectric_cphf": {"calc_types": ("CPHF", "ELASTIC", "FREQ"), "source": ".out: CPHF/CPKS tensors"},
    "dielectric_constant_electronic": {"calc_types": ("CPHF", "ELASTIC", "FREQ"), "source": ".out: CPHF/CPKS tensors"},
    "refractive_index_electronic": {"calc_types": ("CPHF", "ELASTIC", "FREQ"), "source": ".out: CPHF/CPKS tensors"},
    "hyperpolarizability_beta": {"calc_types": ("CPHF",), "source": ".out: CPHF2 tensors"},
    "hyperpolarizability_gamma": {"calc_types": ("CPHF",), "source": ".out: CPHF2/CPHF4 tensors"},
    "qha_thermal_properties": {"calc_types": ("QHA",)},
    "qha_volume_300k_ang3": {"calc_types": ("QHA",)},
    "qha_bulk_modulus_300k_gpa": {"calc_types": ("QHA",)},
    "qha_thermal_expansion_300k_per_k": {"calc_types": ("QHA",)},
    "eos_fits": {"calc_types": ("EOS", "QHA")}, "eos_points": {"calc_types": ("EOS", "QHA")},
    "eos_bulk_modulus_gpa": {"calc_types": ("EOS", "QHA")}, "eos_bulk_modulus_prime": {"calc_types": ("EOS", "QHA")},
    "eos_equilibrium_volume_ang3": {"calc_types": ("EOS", "QHA")},
    "advanced_analysis_note": {"calc_types": ("BAND", "DOSS"), "source": "why a BAND.DAT/DOSS.DAT was not analyzed"},
    "advanced_analysis_error": {"calc_types": ("BAND", "DOSS")},
    "extraction_provenance": {"calc_types": EVERY, "source": "the registry: which derived properties ran, "
                                                             "their versions and files"},
}
_PROPS_RUNS = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL")
for _n in ("magnetic_classification", "classification_gap_source", "scf_gilat_shrink",
           "scf_shrink_k1", "scf_shrink_k2", "scf_shrink_k3"):
    _OVERRIDES.setdefault(_n, {})["calc_types"] = SCF + _PROPS_RUNS
for _n in ("band_gap", "band_gap_type", "direct_band_gap", "indirect_band_gap"):
    _OVERRIDES.setdefault(_n, {})["calc_types"] = SCF + ("DOSS", "TRANSPORT")
_OVERRIDES["hyperpolarizability_beta"]["calc_types"] = ("CPHF", "FREQ")
# the band analysis of a BAND run reads the DOSS run of the same parent too (metal or semimetal)
for _n in ("dos_threshold", "dos_mean", "doss_file_analyzed"):
    _OVERRIDES.setdefault(_n, {})["calc_types"] = ("BAND", "DOSS")
for _n in property_keys.KEYS:
    if _n.startswith(("two_component_",)):
        _OVERRIDES.setdefault(_n, {"calc_types": SCF, "source": ".out: TWOCOMPON SCF magnetization"})
    if _n.startswith("processed_"):
        _OVERRIDES.setdefault(_n, {"source": "computed: population_analysis_processor on the Mulliken tables"})
    if _n.startswith("transport_zt_total"):
        _OVERRIDES.setdefault(_n, {"source": "computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL)"})
    if _n.startswith(("electronic_", "dos_at_fermi", "alpha_", "beta_", "direct_", "indirect_")) and \
            property_keys.KEYS[_n].category == "electronic":
        _OVERRIDES.setdefault(_n, {"calc_types": SCF})

# What the GUI computes or shows from a run, and where it stands in the
# database. status: "stored" (the database has it), "derived" (now a
# registered derived property), "interactive" (Save to Database), "gap" (not
# in the database), "not a property" (a view, not a value).
GUI: List[Dict[str, Any]] = [
    {"what": "Run details card: energy, energy per atom, functional, basis set, CRYSTAL version, MPI processes",
     "where": "Run Analysis › card (runinfo.meta_job)", "calc_types": "every .out",
     "keys": "total_energy_au, energy_per_atom_ev, dft_functional, basis_set, crystal_version, mpi_processes",
     "status": "derived", "note": "energy_per_atom and run_info: the card reads the same functions"},
    {"what": "Run details card: start and end dates, wall and CPU time, shrink, TOLINTEG, spin",
     "where": "Run Analysis › card", "calc_types": "every .out",
     "keys": "wall_time_s, cpu_time_total_s, scf_shrink_k*, scf_tolinteg*, spin_polarized",
     "status": "stored", "note": "the dates are the calculations table's started_at/completed_at for tracked runs"},
    {"what": "Band-edge effective masses m_c, m_d (electrons, holes): the accurate ones, to use",
     "where": "Relaxation Time dialog (m* auto), mace tau", "calc_types": "BAND",
     "keys": "effective_mass_{electron,hole}_{mc,md}, effective_masses", "status": "derived",
     "note": "stored with the BAND run (no transport axis): every degenerate band and valley, several directions, "
             "R² checked. The extractor's real_{electron,hole,average}_effective_mass is a quick estimate (one "
             "band along the path, AdvancedElectronicAnalyzer) and gives other numbers"},
    {"what": "Deformation-potential relaxation time and mobility of a strain series",
     "where": "Relaxation Time dialog › Fit", "calc_types": "SP (the unstrained run)",
     "keys": "relaxation_time_dpt_{carrier}_{axis}_fs, mobility_dpt_{carrier}_{axis}_cm2_vs, relaxation_time_dpt_{carrier}_{axis}",
     "status": "interactive", "note": "Save to Database in the dialog"},
    {"what": "Thermodynamics at every TEMPERAT temperature (G, H, S, Cv vs T)",
     "where": "FREQ output (the first block only was stored)", "calc_types": "FREQ",
     "keys": "thermodynamics_vs_temperature, thermodynamic_n_points", "status": "derived", "note": ""},
    # reference energies (mace/properties/defs/reference_energies.py)
    {"what": "Reference energies: formation, cohesive, surface, adsorption and reaction energies (ΔE; ΔH and ΔG "
             "with FREQ thermodynamics for every run)",
     "where": "Properties tab › Add Reference… (and the database's own lookup)", "calc_types": "OPT, SP, FREQ",
     "keys": "formation_*, cohesive_*, surface_*, adsorption_*, reaction_*, reference_runs", "status": "derived",
     "note": "the references chosen are saved with the run (reference_runs, kept on re-extraction)"},
    {"what": "Transport peaks: |S|max, PF max and ZTe max (average and per axis), carrier type",
     "where": "Transport plot", "calc_types": "TRANSPORT", "keys": "transport_*", "status": "stored", "note": ""},
    {"what": "Transport curves S, σ, κ, PF, ZTe, κe, L vs μ and T",
     "where": "Transport plot (results.transport_data)", "calc_types": "TRANSPORT", "keys": "",
     "status": "gap", "note": "curves; the peaks are stored"},
    {"what": "Total ZT with a lattice κL (and τ)",
     "where": "Transport plot › ZT with κL", "calc_types": "TRANSPORT", "keys": "transport_zt_total*",
     "status": "gap", "note": "stored by the CLI (--kappa-lattice); the GUI's value is not saved"},
    {"what": "Elastic: Voigt/Reuss/Hill moduli, Young's modulus, Poisson's ratio, A^U, Debye T, Born stability",
     "where": "Mechanical Properties Results", "calc_types": "ELASTIC", "keys": "bulk_modulus_*, shear_modulus_*, ...",
     "status": "stored", "note": ""},
    # --- phase 2b (charge/potential, FREQ, SP/OPT, mechanics): one block
    {"what": "Elastic: directional extremes of E, G, ν and the linear compressibility, sound and group velocity "
             "extremes, anisotropy indices, Pugh, Cauchy, Chen/Tian hardness, Clarke/Cahill κmin, piezoelectric "
             "extremes, the 2D anisotropy",
     "where": "Mechanical Properties Results (elastic/views.py)", "calc_types": "ELASTIC",
     "keys": "youngs_modulus_{min,max}, shear_modulus_*, poisson_ratio_*, sound_velocity_q*_*, pugh_ratio, "
             "cauchy_*, vickers_hardness_*, min_thermal_conductivity_*, piezo_*, mechanics_derived",
     "status": "derived", "note": "mechanics_derived: mace.elastic.properties.all_properties, the document's numbers"},
    {"what": "Elastic: the directions of the extremes; each Born criterion",
     "where": "Mechanical Properties Results › surfaces (elastic/views.py)", "calc_types": "ELASTIC",
     "keys": "", "status": "gap", "note": "the extremes are stored, not their directions; elastic_born_stable is"},
    {"what": "EOS fit residuals; QHA S, Cv, Cp, γ near 300 K; CPHF principal n, birefringence, 2V, SHG d_eff; "
             "Born charges, ε0, LO-TO of FREQ with INTCPHF",
     "where": "Mechanical Properties Results", "calc_types": "EOS, QHA, CPHF, FREQ",
     "keys": "eos_*_residual_*, formula_units, qha_*_300k, refractive_index_principal_*, birefringence*, "
             "optic_axial_angle_deg, shg_d_*, born_charge_*, dielectric_constant_static, lo_to_splitting_max",
     "status": "derived", "note": "mechanics_derived"},
    {"what": "QHA harmonic heat capacity per volume; CPHF dielectric spectrum",
     "where": "Mechanical Properties Results", "calc_types": "QHA, CPHF", "keys": "", "status": "gap",
     "note": "curves (qha_thermal_properties and dielectric_cphf are stored)"},
    {"what": "Planar and macroscopic average of the potential; vacuum levels, potential step and work functions "
             "of a slab", "where": "Properties tab", "calc_types": "CHARGE+POTENTIAL",
     "keys": "potential_planar_average, vacuum_level_{top,bottom}_ev, potential_vacuum_step_ev, work_function*_ev",
     "status": "derived", "note": "potential_profile (POT cube)"},
    {"what": "Cell dipole (slab: from the vacuum step, with its dipole correction; molecule, polymer: from the "
             "density)", "where": "Properties tab", "calc_types": "CHARGE+POTENTIAL",
     "keys": "cell_dipole_*_debye, cell_dipole_correction_ev, cell_dipole", "status": "derived",
     "note": "cell_dipole"},
    {"what": "Bader charges, basin volumes and spin moments", "where": "Properties tab",
     "calc_types": "CHARGE+POTENTIAL", "keys": "bader_charge_{min,max}, bader_spin_*, bader_charges",
     "status": "derived", "note": "bader_charges: MACE's own grid algorithm (DENS + SPIN cubes)"},
    {"what": "Phonon Debye temperatures (DOS moments), acoustic velocities and their Debye temperature; "
             "mean-square displacements of a molecule", "where": "Properties tab", "calc_types": "FREQ",
     "keys": "debye_temperature_phonon*_k, sound_velocity_phonon_*, debye_temperature_sound_k, msd_*",
     "status": "derived", "note": "phonon_debye_temperature, phonon_sound_velocities (fort.25), "
                                  "mean_square_displacement (0D)"},
    {"what": "Mulliken charge and spin per element, total magnetization; printed dipole moment; packing fractions",
     "where": "Properties tab", "calc_types": "SP, OPT, FREQ",
     "keys": "mulliken_charge_*, mulliken_spin_*, mulliken_by_element, dipole_moment_*, packing_fraction_*",
     "status": "derived", "note": "mulliken_summary, dipole_moment, packing_fraction"},
    # --- end of phase 2b block
    {"what": "IR and Raman spectra (broadened)", "where": "IR/Raman plot (spectra.py)", "calc_types": "FREQ",
     "keys": "vibrational_frequencies, raman_intensities", "status": "gap",
     "note": "the modes and intensities are stored, the broadened curves are not"},
    {"what": "Band and DOS plots: gaps, band edges, DOS at E_F", "where": "Band and DOS plots",
     "calc_types": "BAND, DOSS", "keys": "band_*, dos_*", "status": "stored", "note": ""},
    # -- electronic derived properties (mace/properties/defs: band_edges, dos_analysis, orbital_composition)
    {"what": "Band gap character: direct or indirect, the k points of the VBM, CBM and smallest direct gap, "
             "the gap of each spin channel; metal or semimetal of a gapless run",
     "where": "Properties tab", "calc_types": "BAND",
     "keys": "band_path_gap_type, band_path_{vbm,cbm,direct_gap}_k, band_path_gap_{alpha,beta}_ev, "
             "band_path_fermi_dos_per_atom, band_path_semimetal_threshold, band_edges",
     "status": "derived", "note": "along the band path (BAND.DAT via mace.transport.mass.read_band); a gapless run "
                                  "is a semimetal when N(E_F) of the DOSS run is below 0.05 states/eV/atom"},
    {"what": "Valence band width and the dispersion of the VBM and CBM bands",
     "where": "Properties tab", "calc_types": "BAND",
     "keys": "valence_band_width_ev, vbm_band_dispersion_ev, cbm_band_dispersion_ev", "status": "derived",
     "note": "the conduction manifold's width is in band_edges only (an LCAO basis gives it no top)"},
    {"what": "Spin splitting of the band edges (spin-polarized runs)", "where": "Properties tab",
     "calc_types": "BAND", "keys": "vbm_spin_splitting_mev, cbm_spin_splitting_mev, band_max_spin_splitting_mev",
     "status": "derived", "note": ""},
    {"what": "Fermi velocities of a metal or semimetal at each E_F crossing of the path", "where": "Properties tab",
     "calc_types": "BAND",
     "keys": "fermi_velocity_{mean,min,max}_m_s, fermi_level_crossings, fermi_velocity_crossings",
     "status": "derived", "note": ""},
    {"what": "Sommerfeld coefficient γ from N(E_F)", "where": "Properties tab", "calc_types": "DOSS",
     "keys": "sommerfeld_gamma_mj_mol_k2, sommerfeld_gamma_per_atom_mj_mol_k2", "status": "derived",
     "note": "N(E_F) itself is the extractor's dos_at_fermi_total"},
    {"what": "d-band center (to E_F + 10 eV, and occupied only), filling and width of transition-metal d "
             "projections", "where": "Properties tab",
     "calc_types": "DOSS", "keys": "d_band_center_ev, d_band_center_occupied_ev, d_band_filling, d_band_width_ev, "
                                  "d_band_element, d_band",
     "status": "derived", "note": "the projections as the DOS plot reads and labels them (results.dos_data)"},
    {"what": "Orbital and atom character of the band edges (of E_F in a metal) from the projected DOS",
     "where": "Properties tab", "calc_types": "DOSS",
     "keys": "{vbm,cbm,fermi_level}_character(_fraction), band_edge_composition", "status": "derived", "note": ""},
    {"what": "Carrier concentrations against the Fermi level at 300 K; intrinsic level and n_i",
     "where": "Properties tab", "calc_types": "DOSS",
     "keys": "carrier_concentration_300k, intrinsic_fermi_level_300k_ev, "
             "intrinsic_carrier_concentration_300k_cm{3,2,1}", "status": "derived", "note": ""},
    {"what": "Density-of-states effective masses from the DOS (rough)", "where": "Properties tab",
     "calc_types": "DOSS",
     "keys": "dos_effective_mass_{electron,hole}, dos_effective_masses", "status": "derived",
     "note": "mace.transport.mass.dos_mass, the DOS fit of mace tau, on CRYSTAL's smoothed DOS (NPOL, grid); "
             "the BAND run's effective_mass_*_md is the accurate m_d"},
    {"what": "Mulliken composition of the HOMO and LUMO (per k point and spin)", "where": "Properties tab",
     "calc_types": "ORBITALS", "keys": "{homo,lumo}_character(_fraction), orbital_composition",
     "status": "derived", "note": "the coefficients and overlap of the orbital viewer (mace/orbitals)"},
    # -- end of the electronic derived properties
    {"what": "Volume statistics (min, max, integral of a cube)", "where": "Volume view (volume.stats)",
     "calc_types": "CHARGE+POTENTIAL", "keys": "", "status": "gap", "note": ""},
    {"what": "Convergence: SCF cycles per point, gradients and displacements vs thresholds",
     "where": "Run Analysis › Convergence", "calc_types": "OPT", "keys": "scf_cycles, optimization_*",
     "status": "gap", "note": "per-point curves; the final values are stored"},
    {"what": "Problems (imaginary modes, warnings, tolerances)", "where": "Problems tab", "calc_types": "every .out",
     "keys": "num_imaginary_frequencies", "status": "not a property", "note": ""},
    {"what": "Structure Analysis: porosity, powder pattern, topology, polyhedra, morphology, planes",
     "where": "Analysis tab", "calc_types": "structures (and a run's geometry)", "keys": "",
     "status": "gap", "note": "computed per structure for the session; not stored"},
    {"what": "Coordination: CN, ECoN, neighbor shells", "where": "Coordination tab", "calc_types": "structures, runs",
     "keys": "neighbor_analysis, final_neighbor_analysis", "status": "stored", "note": "ECoN is not stored"},
    {"what": "Phonopy bands, DOS and thermal properties", "where": "Phonopy results", "calc_types": "phonopy folders",
     "keys": "", "status": "gap", "note": "not a CRYSTAL output"},
]


def _rule(name: str) -> Dict[str, Any]:
    k = property_keys.key(name)
    types, source = _BY_CATEGORY.get(k.category if k else "", (EVERY, ".out"))
    o = _OVERRIDES.get(name, {})
    return {"calc_types": tuple(o.get("calc_types", types)), "source": o.get("source", source)}


def stored() -> List[Dict[str, Any]]:
    """The extractor's own keys: name, unit, type, category, group, label,
    calc types, source, description."""
    from mace.properties import display
    out = []
    for name in sorted(property_keys.KEYS):
        k = property_keys.KEYS[name]
        d = display.describe(name)
        r = _rule(name)
        out.append({"name": name, "unit": k.unit, "type": k.type, "category": k.category, "group": d["group"],
                    "label": d["label_plain"], "calc_types": list(r["calc_types"]), "source": r["source"],
                    "description": k.description, "derived": None, "blob": property_keys.is_blob(name),
                    **_unit_facts(name)})
    return out


def _unit_facts(name: str) -> Dict[str, Any]:
    """The Properties tab's units of a key (mace/properties/units.py): its
    dimension, the units it can be shown in, extensive or not, and the key
    whose row shows it when the database has its quantity in several units."""
    from mace.properties import units
    i = units.info(name)
    return {"dimension": i["dim"], "display_units": units.describe_units(name), "extensive": i["extensive"],
            "shown_as": i["same_as"]}


def derived_rows() -> List[Dict[str, Any]]:
    """One row per output key of each registered derived property."""
    from mace.properties import registry, display
    out = []
    for p in registry.properties():
        for o in p.outputs:
            d = display.describe(o.name)
            out.append({"name": o.name, "unit": o.unit, "type": o.type, "category": d["category"],
                        "group": d["group"], "label": d["label_plain"], "calc_types": list(p.calc_types),
                        "source": p.source, "description": d["description"], "derived": p.name,
                        "version": p.version, "interactive": p.interactive, "blob": o.blob,
                        "siblings": [s.kind + ("" if s.required else " (optional)") for s in p.siblings],
                        "references": [r.name + ("" if r.required else " (optional)") for r in p.references],
                        **_unit_facts(o.name)})
    return out


def everything() -> List[Dict[str, Any]]:
    return stored() + derived_rows()


def _types_text(types: Sequence[str]) -> str:
    t = tuple(types)
    if "*" in t or t == EVERY:
        return "every run"
    if t == SCF:
        return "SCF"
    return ", ".join(t)


def _applies(row: Dict[str, Any], calc_type: Optional[str]) -> bool:
    if not calc_type:
        return True
    t = row["calc_types"]
    return "*" in t or calc_type.upper() in t


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def counts() -> Dict[str, int]:
    rows = stored()
    d = derived_rows()
    gui = {}
    for g in GUI:
        gui[g["status"]] = gui.get(g["status"], 0) + 1
    return {"stored_keys": len(rows), "derived_keys": len(d),
            "derived_properties": len({r["derived"] for r in d}),
            "interactive_keys": sum(1 for r in d if r.get("interactive")),
            **{f"gui_{k.replace(' ', '_')}": v for k, v in gui.items()}}


def markdown() -> str:
    """The generated part of docs/properties.md."""
    from mace.properties import registry
    lines: List[str] = []
    c = counts()
    lines += [f"The database stores **{c['stored_keys']} keys** from the extractor's own parsing and "
              f"**{c['derived_keys']} keys** of **{c['derived_properties']} derived properties** "
              f"({c['interactive_keys']} of these keys belong to interactive properties, saved from the GUI). Of the "
              f"{len(GUI)} things the GUI computes or shows from a run, "
              f"{c.get('gui_stored', 0) + c.get('gui_derived', 0) + c.get('gui_interactive', 0)} are in the "
              f"database and **{c.get('gui_gap', 0)} are gaps** (marked **gap** below).", ""]
    lines += ["SCF in the calculation-type column means OPT, SP, FREQ, ELASTIC, EOS, QHA and CPHF (an ELASTIC, "
              "EOS or QHA run's structure and energies are its reference cell's).", ""]
    # derived
    lines += ["### Derived properties", "",
              "| Property | Keys | Unit | Calculation types | Needs | Version | Source |",
              "|---|---|---|---|---|---|---|"]
    for p in registry.properties():
        needs = [s.kind + ("" if s.required else " (optional)") for s in p.siblings]
        needs += [f"reference run {r.name}" for r in p.references]
        needs += list(p.requires)
        keys = ", ".join(f"`{o.name}`" for o in p.outputs)
        if len(p.outputs) > 6:
            keys = f"{len(p.outputs)} keys: " + ", ".join(f"`{o.name}`" for o in p.outputs[:3]) + ", ..."
        units = sorted({o.unit for o in p.outputs})
        lines.append(f"| {p.name}{' (interactive)' if p.interactive else ''} | {keys} | {_md_escape(', '.join(units))} | "
                     f"{_types_text(p.calc_types)} | {_md_escape(', '.join(needs)) or '-'} | {p.version} | "
                     f"{_md_escape(p.source)} |")
    lines.append("")
    lines += _units_markdown()
    # gui
    lines += ["### What the GUI computes or shows", "",
              "| What | Where | Calculation types | Database keys | Status | Note |", "|---|---|---|---|---|---|"]
    for g in GUI:
        st = "**gap**" if g["status"] == "gap" else g["status"]
        lines.append(f"| {_md_escape(g['what'])} | {_md_escape(g['where'])} | {_md_escape(g['calc_types'])} | "
                     f"{_md_escape(g['keys']) or '-'} | {st} | {_md_escape(g['note']) or '-'} |")
    lines.append("")
    # stored, by calc type
    rows = stored()
    lines += ["### Stored keys by calculation type", "",
              "Each key once, under the first calculation type it applies to; the last column lists all of them.",
              ""]
    seen = set()
    sections = [("Every run", lambda r: tuple(r["calc_types"]) == EVERY),
                ("SCF runs (OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF)", lambda r: tuple(r["calc_types"]) == SCF)]
    sections += [(ct, (lambda c: lambda r: c in r["calc_types"])(ct)) for ct in CALC_TYPES]
    for title, pick in sections:
        sel = [r for r in rows if r["name"] not in seen and pick(r)]
        if not sel:
            continue
        lines += [f"#### {title}", "", "| Key | Unit | Source | Calculation types | Meaning |", "|---|---|---|---|---|"]
        for r in sel:
            seen.add(r["name"])
            lines.append(f"| `{r['name']}` | {_md_escape(r['unit'])} | {_md_escape(r['source'])} | "
                         f"{_types_text(r['calc_types'])} | {_md_escape(r['description'])} |")
        lines.append("")
    rest = [r for r in rows if r["name"] not in seen]
    if rest:
        lines += ["#### Other", "", "| Key | Unit | Source | Calculation types | Meaning |", "|---|---|---|---|---|"]
        for r in rest:
            lines.append(f"| `{r['name']}` | {_md_escape(r['unit'])} | {_md_escape(r['source'])} | "
                         f"{_types_text(r['calc_types'])} | {_md_escape(r['description'])} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _units_markdown() -> List[str]:
    """The Properties tab's display units (mace/properties/units.py)."""
    from mace.properties import units
    lines = ["### Display units", "",
             "The units the Properties tab can show a value in (display only: the database keeps each key in its "
             "stored unit). A preset sets every dimension listed for it; a dimension it does not name keeps the "
             "stored unit.", "",
             "| Dimension | Units | " + " | ".join(p["label"] for p in units.PRESETS[1:]) + " |",
             "|---|---|" + "---|" * (len(units.PRESETS) - 1)]
    for dim, (title, us) in units.dimensions().items():
        if len(us) < 2:
            continue
        lines.append(f"| {title} | {_md_escape(', '.join(u for u, _, _ in us))} | "
                     + " | ".join(_md_escape(p["units"].get(dim, "as stored")) for p in units.PRESETS[1:]) + " |")
        if dim == "energy":
            lines.append("| Energy levels (a gap, a band edge, E<sub>F</sub>, a work function) | the same | "
                         + " | ".join(_md_escape(p["units"].get("energy_level", "as stored")) for p in units.PRESETS[1:])
                         + " |")
    groups = units.same_quantity_groups()
    lines += ["", f"**One row for one quantity**: {len(groups)} quantities are stored under several keys that "
              "differ only by unit; the tab shows each once, under its first key below (the most precise), and "
              "converts it to the unit picked:", ""]
    lines.append(", ".join(f"`{g[0]}` ({', '.join(f'`{n}`' for n in g[1:])})" for g in groups.values()) + ".")
    ext = sorted(n for n in list(property_keys.KEYS) + list(property_keys.EXTRA)
                 if units.is_extensive(n) and not units.same_as(n))
    lines += ["", "**Per cell (extensive)**, also shown per atom and per formula unit (with the keys of the same "
              "quantity in other units): "
              + ", ".join(f"`{n}`" for n in ext) + ".", ""]
    return lines


BEGIN = "<!-- BEGIN GENERATED: mace analyze --list-properties --update-doc docs/properties.md -->"
END = "<!-- END GENERATED -->"


def update_doc(path: Path) -> bool:
    """Rewrite the generated part of ``path``; True when it changed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ValueError(f"{path} has no generated section ({BEGIN} ... {END})")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n\n" + markdown() + "\n" + END + tail
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def generated_section(text: str) -> str:
    head, rest = text.split(BEGIN, 1)
    return rest.split(END, 1)[0].strip() + "\n"


def _table(rows: List[Dict[str, Any]], width: int = 120) -> str:
    from mace.properties import display
    cols = [("Key", lambda r: r["name"]), ("Unit", lambda r: display.unit_text(r["unit"]) or "-"),
            ("Calc types", lambda r: _types_text(r["calc_types"])),
            ("Kind", lambda r: ("derived: " + r["derived"] + (" (interactive)" if r.get("interactive") else ""))
             if r.get("derived") else "stored"),
            ("Source", lambda r: r["source"])]
    data = [[f(r) for _, f in cols] for r in rows]
    w = [max([len(h)] + [len(d[i]) for d in data]) for i, (h, _) in enumerate(cols)]
    w = [min(x, 44) for x in w]

    def cut(s, n):
        return s if len(s) <= n else s[: n - 1] + "…"
    out = ["  ".join(h.ljust(w[i]) for i, (h, _) in enumerate(cols)).rstrip()]
    out.append("  ".join("-" * w[i] for i in range(len(cols))))
    for d in data:
        out.append("  ".join(cut(x, w[i]).ljust(w[i]) for i, x in enumerate(d)).rstrip())
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="mace analyze --list-properties",
                                 description="List every property the database stores (by calculation type) and "
                                             "what the GUI computes that it does not")
    ap.add_argument("--calc-type", help="Only the keys of this calculation type (OPT, SP, FREQ, BAND, DOSS, ...)")
    ap.add_argument("--derived", action="store_true", help="Only the derived properties (mace/properties)")
    ap.add_argument("--gui", action="store_true", help="What the GUI computes or shows, and the database's gaps")
    ap.add_argument("--units", action="store_true",
                    help="The display units the Properties tab converts between, its presets and the quantities "
                         "stored in several units")
    ap.add_argument("--json", action="store_true", help="JSON on stdout")
    ap.add_argument("--markdown", action="store_true", help="The markdown of docs/properties.md's generated part")
    ap.add_argument("--update-doc", metavar="FILE", help="Rewrite the generated part of FILE (docs/properties.md)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.update_doc:
        changed = update_doc(Path(a.update_doc))
        print(f"{a.update_doc}: {'updated' if changed else 'already current'}")
        return 0
    if a.markdown:
        sys.stdout.write(markdown())
        return 0
    if a.gui:
        if a.json:
            print(json.dumps(GUI, indent=1, ensure_ascii=False))
            return 0
        for g in GUI:
            print(f"[{g['status']}] {g['what']}")
            print(f"    where: {g['where']}; runs: {g['calc_types']}" + (f"; keys: {g['keys']}" if g['keys'] else ""))
            if g["note"]:
                print(f"    {g['note']}")
        return 0
    if a.units:
        from mace.properties import units
        if a.json:
            print(json.dumps({**units.catalog(), "same_quantity": units.same_quantity_groups()}, indent=1,
                             ensure_ascii=False))
            return 0
        print("\n".join(_units_markdown()))
        return 0
    rows = derived_rows() if a.derived else everything()
    rows = [r for r in rows if _applies(r, a.calc_type)]
    if a.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return 0
    print(_table(rows))
    c = counts()
    print(f"\n{len(rows)} keys shown ({c['stored_keys']} stored by the extractor, {c['derived_keys']} by "
          f"{c['derived_properties']} derived properties); `--gui` lists the GUI's values and the gaps; "
          "docs/properties.md has the full inventory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
