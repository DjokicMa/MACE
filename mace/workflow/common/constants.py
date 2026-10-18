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
    "CHARGE+POTENTIAL": {
        "name": "Charge Density & Potential",
        "depends_on": ["SP", "OPT"],
        "generates": ["charge_density", "electrostatic_potential"],
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
    "CHARGE+POTENTIAL": "$SCRATCH/crys23/prop",
    "ORBITALS": "$SCRATCH/crys23/prop",
}

# Calculation type categories
D12_CALC_TYPES = ["OPT", "SP", "FREQ"]  # Use .d12 input files
D3_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS"]  # Use .d3 input files

# Optional calculation types (workflow continues if these fail)
OPTIONAL_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS", "FREQ"]

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
    totals (menu answer 3), all bands, 10000 energy points, DOSS.DAT output
    (format 2), NPOL 14, no integrated DOS.

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
        "n_points": 10000,
        "output_format": 2,
        "npol": 14,
        "print_integrated": False,
    }


def opt2d3_default_band_config():
    """The BAND configuration opt2d3 builds when every question is answered
    with Enter: the automatic path in high-symmetry label format (shrink 0),
    1000 k points along the path, all bands, no eigenvalue print.

    The workflow routes that choose a BAND configuration for the user (the
    engine default, the planner's Basic answer, quick-start / --progress)
    use this, so they write the deck the manual route writes. A fresh dict
    is returned on every call. For a parent CRYSTAL23 reads no labels for
    (triclinic, any slab) the writer puts the same path in integer
    coordinates, the deck opt2d3's k-point vector format writes.
    """
    return {
        "calculation_type": "BAND",
        "path_method": "labels",
        "path": "auto",
        "labels": "auto",
        "auto_path": True,
        "shrink": 0,
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
        "relaxation_time": 10
    }
