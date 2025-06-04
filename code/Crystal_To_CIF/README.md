D12 Generation and Ghost Atom Insertion Scripts

This folder contains Python scripts for automating .d12 input file generation and ghost atom insertion for CRYSTAL23 slab simulations. These tools help:

    Convert CIF files to .d12 inputs

    Compute or manually define ghost atom spacing

    Insert ghost atoms above and below slab surfaces

Scripts Overview
NewCifToD12.py

Converts .cif files to .d12 input format for CRYSTAL23.

Features:

    Interactive and batch modes

    Supports multiple dimensionalities (CRYSTAL, SLAB, MOLECULE, etc.)

    Configurable DFT functional, basis set, symmetry handling, SCF controls

    Supports external and internal basis sets

Requirements:

    Language: Python 3

    Libraries: numpy, ase, spglib

    Input: .cif files

    Output: *.d12 files

Usage:

python NewCifToD12.py --cif_dir /path/to/cif/files

create_d12_w-ghosts.py

Automatically computes interlayer spacing using _bulk.out and _slab.out, and injects ghost atoms into a slab .d12 file.

Features:

    Parses atomic coordinates from output files

    Calculates spacing from bulk/surface geometry

    Adds ghost atoms symmetrically on top and bottom of the slab

Requirements:

    Language: Python 3

    Libraries: numpy, math, matplotlib, glob, os, csv, warnings

    Input: *_slab.out, *_bulk.out, *_slab.d12

    Output: *_ghostatoms_slab.d12

Usage:

python create_d12_w-ghosts.py

manual_create_d12_w-ghosts.py

Inserts ghost atoms into slab .d12 based on user-provided spacing.

Features:

    Prompts for ghost atom separation manually

    Duplicates atoms above and below based on the given spacing

Requirements:

    Language: Python 3

    Libraries: numpy, math, matplotlib, glob, os, csv, warnings

    Input: *_slab.d12

    Output: _<spacing>A_ghosts_slab.d12

Usage:

python manual_create_d12_w-ghosts.py

CRYSTALOptToD12.py

Parses the final geometry in a CRYSTAL output and creates a fresh .d12 for follow-up calculations.

Features:

    Improved modular replacement for get_optimized2.py

    Reads final optimized geometry from .out

    Generates clean .d12 input for next-step calculations

Requirements:

    Language: Python 3

    Libraries: os, re, argparse, pathlib

    Input: CRYSTAL output file

    Output: .d12 input file

Requirements

All scripts require:

pip install numpy matplotlib ase spglib

Outputs
.d12 Files

    Generated from .cif files using NewCifToD12.py

    Enhanced with ghost atoms using create_d12_w-ghosts.py or manual_create_d12_w-ghosts.py

    Post-optimized .d12 regenerated via CRYSTALOptToD12.py

These .d12 files are CRYSTAL23-ready and suitable for slab models, vacuum simulations, and ghost-augmented surface calculations.
Notes

    Only P1 symmetry is supported for ghost atom injection

    Ensure necessary _slab.out, _bulk.out, and _slab.d12 files are present in the working directory

    Adjust paths inside NewCifToD12.py if using local basis sets
