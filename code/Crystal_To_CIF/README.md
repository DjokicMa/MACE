# D12 Generation and Ghost Atom Insertion Scripts

This folder contains Python scripts for automating `.d12` input file generation and ghost atom insertion for CRYSTAL23 slab simulations. These tools help:

* Convert CIF files to `.d12` inputs
* Compute or manually define ghost atom spacing
* Insert ghost atoms above and below slab surfaces

## Scripts Overview

### `NewCifToD12.py`

Converts `.cif` files to `.d12` input format for CRYSTAL23.

**Features**:

* Interactive and batch modes
* Supports multiple dimensionalities (CRYSTAL, SLAB, MOLECULE, etc.)
* Configurable DFT functional, basis set, symmetry handling, SCF controls
* Supports external and internal basis sets

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `ase`, `spglib`
* Input Files: `.cif`
* Output File: `*.d12`

**Usage**:

```bash
python NewCifToD12.py --cif_dir /path/to/cif/files
```

---

### `create_d12_w-ghosts.py`

Automatically computes interlayer spacing using `_bulk.out` and `_slab.out`, and injects ghost atoms into a slab `.d12` file.

**Features**:

* Parses atomic coordinates from output files
* Calculates spacing from bulk/surface geometry
* Adds ghost atoms symmetrically on top and bottom of the slab

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`
* Input Files: `*_slab.out`, `*_bulk.out`, `*_slab.d12`
* Output File: `*_ghostatoms_slab.d12`

**Usage**:

```bash
python create_d12_w-ghosts.py
```

---

### `manual_create_d12_w-ghosts.py`

Inserts ghost atoms into slab `.d12` based on user-provided spacing.

**Features**:

* Prompts for ghost atom separation manually
* Duplicates atoms above and below based on the given spacing

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`
* Input Files: `*_slab.d12`
* Output File: `_<spacing>A_ghosts_slab.d12`

**Usage**:

```bash
python manual_create_d12_w-ghosts.py
```

---

## Requirements

* Python 3.x
* `numpy`
* `matplotlib`
* `ase`
* `spglib`

Install with:

```bash
pip install numpy matplotlib ase spglib
```
### `CRYSTALOptToD12.py`

* Language: Python 3
* Required Libraries: `os`, `re`, `argparse`, `pathlib`
* Purpose: Similar to `get_optimized2.py`, but enhanced and modular.
* Function: Parses the final geometry in a CRYSTAL output and creates a fresh `.d12` for follow-up calculations.
* Improvement: Intended to replace `get_optimized2.py` with better reliability and clearer structure.

---

## Outputs

### `.d12` Files

* Converted from `.cif` files (from `NewCifToD12.py`)
* Enhanced with ghost atoms (from `create_d12_w-ghosts.py` or `manual_create_d12_w-ghosts.py`)

These files are compatible with CRYSTAL23 and useful for simulating surfaces, vacuum levels, and electronic properties with ghost atom augmentation.

---

## Notes

* Only P1 symmetry is supported for ghost atom injection.
* Ensure necessary `_slab.out`, `_bulk.out`, and `_slab.d12` files are in the working directory.
* Customize paths in `NewCifToD12.py` for local basis sets.
