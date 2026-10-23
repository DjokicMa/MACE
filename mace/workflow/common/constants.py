"""
Common constants for the CRYSTAL workflow system.

This module contains shared constants used across multiple workflow components.
Extracted during refactoring to avoid duplication and ensure consistency.
"""

# Calculation types and their properties
CALC_TYPES = {
    "OPT": {
        "name": "Geometry Optimization",
        "depends_on": [],
        "generates": ["optimized_geometry"],
    },
    "SP": {
        "name": "Single Point Energy",
        "depends_on": ["OPT"],
        "generates": ["wavefunction"],
    },
    "FREQ": {
        "name": "Frequency Calculation",
        "depends_on": ["OPT"],
        "generates": ["frequencies", "ir_intensities"],
    },
    "BAND": {
        "name": "Band Structure",
        "depends_on": ["SP", "OPT"],
        "generates": ["band_structure"],
    },
    "DOSS": {
        "name": "Density of States",
        "depends_on": ["SP", "OPT"],
        "generates": ["dos"],
    },
    "TRANSPORT": {
        "name": "Transport Properties",
        "depends_on": ["SP", "OPT"],
        "generates": ["conductivity", "seebeck_coefficient"],
    },
    # the TRANSPORT step's relaxation time (RELAXTIM) from deformation
    # potentials: strained single points of the SP, then a fit
    # (mace/workflow/tau_step.py)
    "TAU": {
        "name": "Relaxation Time (deformation potentials)",
        "depends_on": ["SP"],
        "generates": ["relaxation_time"],
    },
    "CHARGE+POTENTIAL": {
        "name": "Charge Density & Potential",
        "depends_on": ["SP", "OPT"],
        "generates": ["charge_density", "electrostatic_potential"],
    },
    # opt2d12's response-property modules (Crystal_d12/d12_calc_elastic.py):
    # one deck each, at the geometry of the OPT before it (the SP's when the
    # chain has no OPT), with that run's method; nothing is built from them
    "ELASTIC": {
        "name": "Elastic Constants (ELASTIC)",
        "depends_on": ["OPT", "SP"],
        "generates": ["elastic_constants"],
    },
    "EOS": {
        "name": "Equation of State (EOS)",
        "depends_on": ["OPT", "SP"],
        "generates": ["equation_of_state"],
    },
    "QHA": {
        "name": "Quasi-Harmonic Approximation (QHA)",
        "depends_on": ["OPT", "SP"],
        "generates": ["thermal_expansion"],
    },
    "CPHF": {
        "name": "Dielectric Response (CPHF)",
        "depends_on": ["OPT", "SP"],
        "generates": ["polarizability", "dielectric_tensor"],
    },
    "ORBITALS": {
        "name": "Orbitals (Molden files)",
        "depends_on": ["SP", "OPT"],
        "generates": ["orbitals"],
    },
}

# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "basic_opt": ["OPT"],
    "opt_sp": ["OPT", "SP"],
    "opt_sp_freq": ["OPT", "SP", "FREQ"],
    "full_electronic": ["OPT", "SP", "BAND", "DOSS"],
    "double_opt": ["OPT", "OPT2", "SP"],
    "complete": ["OPT", "SP", "BAND", "DOSS", "FREQ"],
    "transport_analysis": ["OPT", "SP", "TRANSPORT"],
    "charge_analysis": ["OPT", "SP", "CHARGE+POTENTIAL"],
    "combined_analysis": ["OPT", "SP", "BAND", "DOSS", "TRANSPORT"],
}

# The order of the planner's template menu (`mace workflow` asks "Select workflow template"): "custom"
# follows these, then any template added later. The menu keeps the numbers it always had (piped answers
# and the default "3" stay valid); the GUI shows the same numbers.
PLANNER_TEMPLATE_MENU = ("basic_opt", "opt_sp", "full_electronic", "double_opt", "complete",
                         "transport_analysis", "charge_analysis", "combined_analysis")


def planner_template_numbers():
    """{template name: the number the planner's menu gives it} (custom and interactive included)."""
    order = list(PLANNER_TEMPLATE_MENU) + ["custom"] + [k for k in WORKFLOW_TEMPLATES if k not in PLANNER_TEMPLATE_MENU]
    return {k: i for i, k in enumerate(order, 1)}


# One line about each template (`mace workflow --show-templates`, the GUI).
WORKFLOW_TEMPLATE_DESCRIPTIONS = {
    "basic_opt": "Basic geometry optimization only",
    "opt_sp": "Geometry optimization followed by single point calculation",
    "opt_sp_freq": "Optimization, single point, then vibrational frequencies",
    "full_electronic": "Complete electronic structure characterization",
    "double_opt": "Two-stage optimization for difficult convergence cases",
    "complete": "Complete characterization including vibrational analysis",
    "transport_analysis": "Transport properties calculation (conductivity, Seebeck)",
    "charge_analysis": "Charge density and electrostatic potential analysis",
    "combined_analysis": "Electronic structure with transport properties",
}

# The built-in MPI ranks of a properties job (BAND, DOSS, TRANSPORT,
# CHARGE+POTENTIAL, ORBITALS; Pproperties): the one place this number lives. The job
# scripts (submit_prop.sh), the site settings, the planner, quick start and
# `mace submit` all take it from here. Measured on MSU HPCC: BAND and DOSS
# runs on 28 ranks kept 0.4-0.9 cores busy (1-3 %), and a work-based sizing
# of 160+ properties runs asked for 8 or fewer ranks for most of them (12-19
# only for the largest slabs). Pproperties does spread k-points over ranks,
# so it is not 1; their memory is a total (--mem), not per rank, and stays.
PROPERTIES_NTASKS = 8
# The built-in memory of a properties job: a total (#SBATCH --mem), the same
# for every properties type and every path (template, mace submit, planner,
# quick start, workflow steps). Measured peaks stay under 0.6 GB per rank.
PROPERTIES_MEMORY = "80G"

# Default SLURM resource allocations: MACE's built-in values. A user's or
# group's site settings replace them ([resources] ntasks, mem_per_cpu, ... in
# ~/.config/mace/site.toml, or MACE_SLURM_NTASKS); `mace site` shows what
# applies (docs/cli.md, "Change the default cores per job").
SLURM_RESOURCE_DEFAULTS = {
    "OPT": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "5G",
        "account": "mendoza_q",
    },
    "SP": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "3-00:00:00",
        "memory_per_cpu": "4G",
        "account": "mendoza_q",
    },
    "FREQ": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "7G",
        "account": "mendoza_q",
    },
    # the strained single points: the SP's job, 1.5x its walltime (a strained
    # cell has less symmetry, so more k points)
    "TAU": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "4-12:00:00",
        "memory_per_cpu": "4G",
        "account": "mendoza_q",
    },
    # ELASTIC, EOS, QHA and CPHF: one job of many SCFs each (52x, 22x, 50x
    # and 19x the SP's work on the HPCC diamond runs, resources_suggest.py),
    # so the OPT's walltime; strained (ELASTIC) and displaced (QHA) cells lose
    # the symmetry, so FREQ's memory
    "ELASTIC": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "7G",
        "account": "mendoza_q",
    },
    "EOS": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "5G",
        "account": "mendoza_q",
    },
    "QHA": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "7G",
        "account": "mendoza_q",
    },
    "CPHF": {
        "ntasks": 32,
        "nodes": 1,
        "walltime": "7-00:00:00",
        "memory_per_cpu": "5G",
        "account": "mendoza_q",
    },
    "BAND": {
        "ntasks": PROPERTIES_NTASKS,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": PROPERTIES_MEMORY,  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "DOSS": {
        "ntasks": PROPERTIES_NTASKS,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": PROPERTIES_MEMORY,  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "TRANSPORT": {
        "ntasks": PROPERTIES_NTASKS,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": PROPERTIES_MEMORY,  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "CHARGE+POTENTIAL": {
        "ntasks": PROPERTIES_NTASKS,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": PROPERTIES_MEMORY,  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    # ORBITALS itself takes under a second (HPCC, 7 systems); the NEWK before
    # it costs what any NEWK on that mesh costs. Γ or IS 2 by default.
    "ORBITALS": {
        "ntasks": PROPERTIES_NTASKS,
        "nodes": 1,
        "walltime": "1:00:00",
        "memory": PROPERTIES_MEMORY,  # Total memory, not per CPU
        "account": "mendoza_q",
    },
}

# Module and scratch directory settings
MODULE_DEFAULTS = {
    "CRYSTAL23": "CRYSTAL/23-intel-2023a",
    "Python": "Python/3.11.3-GCCcore-12.3.0",
    "Python_bundle": "Python-bundle-PyPI/2023.06-GCCcore-12.3.0",
}

SCRATCH_DIR_DEFAULTS = {
    "OPT": "$SCRATCH/crys23",
    "SP": "$SCRATCH/crys23",
    "FREQ": "$SCRATCH/crys23",
    "BAND": "$SCRATCH/crys23/prop",
    "DOSS": "$SCRATCH/crys23/prop",
    "TRANSPORT": "$SCRATCH/crys23/prop",
    "TAU": "$SCRATCH/crys23",
    "ELASTIC": "$SCRATCH/crys23",
    "EOS": "$SCRATCH/crys23",
    "QHA": "$SCRATCH/crys23",
    "CPHF": "$SCRATCH/crys23",
    "CHARGE+POTENTIAL": "$SCRATCH/crys23/prop",
    "ORBITALS": "$SCRATCH/crys23/prop",
}

# Calculation type categories
D12_CALC_TYPES = ["OPT", "SP", "FREQ", "ELASTIC", "EOS", "QHA", "CPHF"]  # Use .d12 input files
# The opt2d12 module types a workflow step can be (written by CRYSTALOptToD12
# with the step's "elastic_settings"); terminal: no step is built from them
MODULE_STEP_TYPES = ("ELASTIC", "EOS", "QHA", "CPHF")

#: a stand-in output's Fermi level (Hartree): previews place E_F-relative ranges (TRANSPORT's MURANGE and
#: TDFRANGE, a DOSS window) around it; the real step places them around its parent's E_F
STAND_IN_FERMI_AU = -0.2
D3_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS"]  # Use .d3 input files

# Calculation Setup and the GUI's Calculation menu (1.1.6): every calculation type a file can be set
# up for, in two groups numbered as the CLI's own menus number them. Regular calculations are the
# .d12 runs of opt2d12's "Select calculation type" menu (Crystal_d12/d12_interactive.py), phonopy's
# displaced supercells (a folder of decks) its choice 8. Properties calculations are the .d3 runs of opt2d3's menu
# (Crystal_d3/CRYSTALOptToD3.py). Each entry: (type, menu number, the menu's words).
SETUP_CALC_GROUPS = (
    ("regular", "Regular Calculations", "mace opt2d12", (
        ("SP", 1, "Single Point (SP)"), ("OPT", 2, "Optimization (OPT)"), ("FREQ", 3, "Frequency (FREQ)"),
        ("ELASTIC", 4, "Elastic, piezoelectric or photoelastic constants (ELASTIC)"),
        ("EOS", 5, "Equation of state (EOS)"), ("QHA", 6, "Quasi-harmonic approximation (QHA)"),
        ("CPHF", 7, "Dielectric response, CPHF/KS (CPHF)"),
        ("PHONOPY", 8, "Finite-displacement phonons with phonopy (PHONOPY)"))),
    ("properties", "Properties Calculations", "mace opt2d3", (
        ("BAND", 1, "Electronic band structure (BAND)"), ("DOSS", 2, "Density of states (DOSS)"),
        ("TRANSPORT", 3, "Transport properties (TRANSPORT)"), ("CHARGE", 4, "Charge density (CHARGE)"),
        ("POTENTIAL", 5, "Electrostatic potential (POTENTIAL)"),
        ("CHARGE+POTENTIAL", 6, "Combined calculation (CHARGE+POTENTIAL)"),
        ("ORBITALS", 7, "Orbitals as Molden files (ORBITALS)"))),
)
# A follow-up that reads a computed run starts from a finished run of one of these
# (mace opt2d12 refuses the others for choices 4-8: phonopy_bridge.parent.run_refusals)
FINISHED_RUN_TYPES = ("OPT", "SP")
# CHARGE and POTENTIAL are the halves of CHARGE+POTENTIAL; PHONOPY is no workflow step, and
# needs what the modules need (a finished OPT or SP: phonopy_bridge.parent.run_refusals)
_SETUP_REQUIREMENT_OF = {"CHARGE": "CHARGE+POTENTIAL", "POTENTIAL": "CHARGE+POTENTIAL", "PHONOPY": "ELASTIC"}


def setup_requirement(calc_type: str) -> str:
    """What the file in front needs for Calculation Setup to set up calc_type, from the
    workflow's own prerequisites (CALC_TYPES depends_on): "run" (a finished OPT or SP output)
    for a type that depends on an SP, whose wavefunction or optimized geometry and settings it
    reads; "structure" (a CIF, a deck or an output) for the others."""
    deps = CALC_TYPES[_SETUP_REQUIREMENT_OF.get(calc_type, calc_type)]["depends_on"]
    return "run" if "SP" in deps else "structure"

# Optional calculation types (workflow continues if these fail)
OPTIONAL_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS", "FREQ", "TAU",
                       "ELASTIC", "EOS", "QHA", "CPHF"]

# File extensions
CRYSTAL_INPUT_EXTENSIONS = {
    "d12": [".d12", ".D12"],
    "d3": [".d3", ".D3"],
    "cif": [".cif", ".CIF"],
}

CRYSTAL_OUTPUT_EXTENSIONS = {
    "output": [".out", ".output", ".log"],
    "wavefunction": [".f9", ".F9", "fort.9"],
    "phonon": [".f25", ".F25", "fort.25"],
}

# Error recovery settings
DEFAULT_MAX_RECOVERY_ATTEMPTS = 3
DEFAULT_RECOVERY_WAIT_TIME = 60  # seconds

# Queue manager settings
DEFAULT_MAX_JOBS = 250
DEFAULT_RESERVE_JOBS = 30
DEFAULT_MAX_SUBMIT = 5

def opt2d3_default_doss_config():
    """The DOSS configuration opt2d3 builds when every question is answered
    with Enter: projected DOS by element and orbital type with element
    totals (menu answer 3), all bands, the Standard energy resolution (the
    deck's points follow from its step and the span, Crystal_d3/d3_config
    doss_npts), DOSS.DAT output (format 2), NPOL 14, no integrated DOS.

    The workflow routes that choose a DOSS configuration for the user use
    this. They carried {projection_type: 0, energy_range: [-20, 20]}, which
    the writer does not read (it takes projection_type 1-4 and
    energy_window), so the plan described total DOS in a window while the
    deck was this one. A fresh dict is returned on every call.
    """
    return {
        "calculation_type": "DOSS",
        "projection_type": 3,
        "project_orbital_types": True,
        "element_only": False,
        "include_element_totals": True,
        "energy_resolution": "standard",
        "n_points": "auto",
        "output_format": 2,
        "npol": 14,
        "print_integrated": False,
    }


def opt2d3_default_band_config():
    """The BAND configuration opt2d3 builds when every question is answered
    with Enter: the automatic path as SeeK-path full paths (format 4: the
    HPKOT path of the cell, k-point vectors in CRYSTAL's reciprocal basis),
    1000 k points along the path, all bands, no eigenvalue print.

    SeeK-path is the default k-path of a 3D crystal's BAND step (the seekpath
    package is a required dependency). The workflow routes that choose a BAND
    configuration for the user (the engine default, the planner's Basic
    answer, quick-start / --progress) use this, so they write the deck the
    manual route writes. A fresh dict is returned on every call. A slab or a
    polymer gets the high-symmetry label path instead ("kpath_default"), as
    opt2d3's Enter gives it there.
    """
    return {
        "calculation_type": "BAND",
        "path_method": "coordinates",
        "shrink": "auto",
        "path": "auto",
        "labels": "auto",
        "auto_path": True,
        "seekpath_full": True,
        # the default, not a chosen format: a slab or polymer gets the label
        # path (Crystal_d3/d3_kpoints.default_band_format)
        "kpath_default": True,
        "n_points": 1000,
        "bands": "auto",
        "auto_bands": True,
        "first_band": 1,
        "last_band": None,
        "plot": True,
        "print_eigenvalues": False,
    }


def opt2d3_default_transport_config():
    """The TRANSPORT configuration the workflow routes use: opt2d3's
    answers when Enter is pressed through it, as opt2d3 holds them (the
    temperatures are floats there, so TRANGE is written "100.0 800.0 50.0"
    by every route). A fresh dict is returned on every call."""
    return {
        "calculation_type": "TRANSPORT",
        "temperature_range": [100.0, 800.0, 50.0],
        "mu_range": [-2.0, 2.0, 0.01],
        "mu_reference": "fermi",
        "mu_range_type": "auto_fermi",
        "tdf_range": [-5.0, 5.0, 0.01],
        # BOLTZTRA's k mesh: the Standard spacing (Crystal_d3/d3_config.transport_mesh)
        "k_resolution": "standard",
        "relaxation_time": 10
    }
