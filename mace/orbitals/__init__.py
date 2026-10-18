"""CRYSTAL23 ORBITALS support: read the Molden files CRYSTAL writes and put orbitals on grids.

CRYSTAL's ORBITALS keyword (properties) writes molecular orbitals (0D) or
Bloch orbitals at the NEWK k points (1D/2D/3D) in a "modified Molden"
format. This package reads those files, checks them (orthonormality on
load), evaluates the orbitals with the Bloch lattice sum on a grid, and
writes Gaussian cubes for the existing cube pipeline.
"""
