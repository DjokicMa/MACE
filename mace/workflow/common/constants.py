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
}

# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "basic_opt": ["OPT"],
    "opt_sp": ["OPT", "SP"],
    "full_electronic": ["OPT", "SP", "BAND", "DOSS"],
    "double_opt": ["OPT", "OPT2", "SP"],
    "complete": ["OPT", "SP", "BAND", "DOSS", "FREQ"],
    "transport_analysis": ["OPT", "SP", "TRANSPORT"],
    "charge_analysis": ["OPT", "SP", "CHARGE+POTENTIAL"],
    "combined_analysis": ["OPT", "SP", "BAND", "DOSS", "TRANSPORT"],
}

# Default SLURM resource allocations
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
        "ntasks": 28,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": "48G",  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "DOSS": {
        "ntasks": 28,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": "48G",  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "TRANSPORT": {
        "ntasks": 28,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": "80G",  # Total memory, not per CPU
        "account": "mendoza_q",
    },
    "CHARGE+POTENTIAL": {
        "ntasks": 28,
        "nodes": 1,
        "walltime": "2:00:00",
        "memory": "80G",  # Total memory, not per CPU
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
}

# Calculation type categories
D12_CALC_TYPES = ["OPT", "SP", "FREQ"]  # Use .d12 input files
D3_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]  # Use .d3 input files

# Optional calculation types (workflow continues if these fail)
OPTIONAL_CALC_TYPES = ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "FREQ"]

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