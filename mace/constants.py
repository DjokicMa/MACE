"""Canonical physical constants for MACE (single source of truth).

Pure literals, zero imports — any layer can import this without creating cycles.

These full-precision (CODATA 2018) values were ALREADY used, byte-identical, by
the canonical sites (mace/database/utils/units.py, mace/utils/dat_file_processor.py)
and the regression tests; this module just removes the duplicate literals so they
share one definition. Truncated copies elsewhere (e.g. ``27.2114`` inside the
validated property_extractor parser, and the d3/plotting scripts) are deliberately
NOT changed here — replacing those alters computed output and is a separate,
regression-pinned precision fix, not this byte-identical consolidation.
"""

# Hartree <-> electron-volt
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Bohr <-> Angstrom
BOHR_TO_ANGSTROM = 0.52917721067
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
