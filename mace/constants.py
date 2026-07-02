"""Canonical physical constants for MACE (single source of truth).

Pure literals, zero imports — any layer can import this without creating cycles.

These full-precision (CODATA 2018) values are the single definition shared by
the mace package (units.py, dat_file_processor.py, property_extractor.py all
import from here; the extractor's former truncated ``27.2114`` literals were
replaced with HARTREE_TO_EV in the v1.1.0 extraction wave — a ~2e-6 relative
shift in derived ``*_ev`` values, covered by the extraction regression tests).
Truncated copies remain ONLY in the frozen legacy scripts under
``code/NewPlotting_Scripts/`` — those are preserved-for-compatibility tools and
are deliberately not modified.
"""

# Hartree <-> electron-volt
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Bohr <-> Angstrom
BOHR_TO_ANGSTROM = 0.52917721067
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
