# D12 Generation and Ghost Atom Insertion Tools

This repository contains Python scripts to automate and customize the generation of `.d12` input files for **CRYSTAL23** quantum chemical calculations. These tools enable CIF-to-D12 conversion and automated/manual insertion of ghost atoms for slab geometries.

---

## 📜 Scripts Overview

### `NewCifToD12.py`

Converts `.cif` files to CRYSTAL23-compatible `.d12` input files.

**Features:**
- Supports `CRYSTAL`, `SLAB`, `POLYMER`, `MOLECULE` dimensionality
- Symmetry detection via CIF or `spglib`
- DFT functionals, basis sets, SCF/grid/optimization controls
- Batch mode and interactive mode available

**Usage:**
```bash
python NewCifToD12.py --cif_dir /path/to/cif/files

create_d12_w-ghosts.py

Automatically inserts ghost atoms above and below a slab using spacing computed from _bulk.out and _slab.out.

Usage:

python create_d12_w-ghosts.py

Input:

    *_bulk.out, *_slab.out, *_slab.d12

Output:

    *_ghostatoms_slab.d12

manual_create_d12_w-ghosts.py

Inserts ghost atoms with a user-defined spacing.

Usage:

python manual_create_d12_w-ghosts.py

Input:

    *_slab.d12

Output:

    _<spacing>A_ghosts_slab.d12

📦 Requirements

    Python 3.x

    numpy

    matplotlib

    ase

    spglib (optional)

Install with:

pip install numpy matplotlib ase spglib

📂 Outputs

    .d12 files with embedded ghost atoms for slab geometries

    Customized .d12 input files from CIFs ready for CRYSTAL23

⚠️ Notes

    Only P1 symmetry is supported for ghost atom insertion.

    Make sure the directory contains the required _bulk.out, _slab.out, and .d12 files.

    Modify NewCifToD12.py paths for basis sets to match your environment.
