# D12 Generation and Ghost Atom Insertion Scripts

This folder contains Python scripts for automating `.d12` input file generation and ghost atom insertion for **CRYSTAL23** slab simulations.

---

## Features

- Convert `.cif` files to `.d12` format  
- Insert ghost atoms above and below slab surfaces  
- Automatically or manually calculate ghost spacing  
- Prepare CRYSTAL input files from optimized geometries  

---

## Scripts Overview

### `NewCifToD12.py`

**Purpose:** Converts `.cif` files to `.d12` format.

**Features:**

- Supports SLAB, MOLECULE, and CRYSTAL input modes  
- Interactive or batch conversion  
- Customizable DFT functional, SCF, symmetry, and basis options  
- **NEW**: Comprehensive frequency calculation support including:
  - IR intensities (Berry phase, Wannier functions, CPHF methods)
  - Raman intensities with CPHF
  - Phonon dispersion and density of states
  - Spectral generation (IR/Raman spectra)
  - Anharmonic calculations (ANHARM, VSCF, VCI)
  - Temperature-dependent thermodynamic properties  

**Usage:**

```bash
python NewCifToD12.py --cif_dir /path/to/cif/files
```

**Inputs:** `.cif`  
**Outputs:** `*.d12`  
**Libraries:** `numpy`, `ase`, `spglib`

**Integration:** Fully integrated with `run_workflow.py` for automated workflow execution. Input settings are automatically extracted and stored in the materials database for complete calculation provenance.  

---

### `create_d12_w-ghosts.py`

**Purpose:** Automatically adds ghost atoms using geometry from `_bulk.out` and `_slab.out`.

**Features:**

- Parses coordinates  
- Computes spacing from slab/bulk output  
- Injects ghost atoms top and bottom  

**Usage:**

```bash
python create_d12_w-ghosts.py
```

**Inputs:** `*_slab.out`, `*_bulk.out`, `*_slab.d12`  
**Outputs:** `*_ghostatoms_slab.d12`  
**Libraries:** `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`  

---

### `manual_create_d12_w-ghosts.py`

**Purpose:** Manually inserts ghost atoms with user-defined spacing.

**Features:**

- Prompts for spacing value  
- Duplicates atoms symmetrically  

**Usage:**

```bash
python manual_create_d12_w-ghosts.py
```

**Input:** `*_slab.d12`  
**Output:** `_<spacing>A_ghosts_slab.d12`  
**Libraries:** `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`  

---

### `CRYSTALOptToD12.py`

**Purpose:** Extracts optimized geometry from CRYSTAL output and regenerates `.d12`.

**Features:**

- Modular and cleaner replacement for `get_optimized2.py`  
- Parses final geometry section  
- **NEW**: Full support for comprehensive frequency calculations
- Generates SP, FREQ, and other calculation types from optimized geometries
- Complete integration with workflow system  

**Usage:**

```bash
python CRYSTALOptToD12.py --input optimized_output.out
```

**Input:** Optimized CRYSTAL `.out`  
**Output:** `.d12`  
**Libraries:** `os`, `re`, `argparse`, `pathlib`  

---

## Installation

```bash
pip install numpy matplotlib ase spglib
```

---

## Outputs

All scripts generate CRYSTAL23-compatible `.d12` files:

- From `.cif` via `NewCifToD12.py`  
- With ghost atoms via `create_d12_w-ghosts.py` or `manual_create_d12_w-ghosts.py`  
- Post-optimization via `CRYSTALOptToD12.py`  

---

## Shared Module: `d12creation.py`

**Purpose:** Shared constants, utilities, and functions for creating D12 input files.

**Key Features:**

- Element class with atomic numbers
- Basis set definitions (internal and external)
- DFT functional library with dispersion support
- **NEW**: Comprehensive `write_frequency_section()` function supporting:
  - Multiple calculation modes (GAMMA, DISPERSION, CUSTOM)
  - IR intensity methods (Berry phase, Wannier functions, CPHF)
  - Raman intensity calculation
  - Spectral generation with customizable parameters
  - Phonon dispersion and DOS calculations
  - Elastic constants
  - Anharmonic corrections (ANHARM, VSCF, VCI)
  - Temperature/pressure-dependent properties
  - Advanced settings (restart, print levels, Hessian storage)

**Frequency Calculation Examples:**

```python
# Simple gamma point calculation
freq_settings = {
    "mode": "GAMMA",
    "numderiv": 2,
    "temperatures": [298.15]
}

# IR and Raman with CPHF
freq_settings = {
    "mode": "GAMMA",
    "intensities": True,
    "ir_method": "CPHF",
    "raman": True,
    "ir_spectrum": True,
    "raman_spectrum": True
}

# Phonon dispersion
freq_settings = {
    "mode": "DISPERSION",
    "n_kpoints": 30,
    "phdos": True
}
```

---

## Notes

- Only **P1 symmetry** is supported for ghost atom injection  
- `_slab.out`, `_bulk.out`, and `_slab.d12` must be present in the working directory  
- Customize basis paths in `NewCifToD12.py` as needed  
- Frequency calculations now support all CRYSTAL23 features through interactive configuration

---
