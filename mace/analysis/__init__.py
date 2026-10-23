"""Structure analysis (1.1.6): porosity (native grid + Zeo++), powder
diffraction (simulation, peaks, indexing, Pawley/Rietveld), topology (nets,
RCSR matching, interpenetration) and VESTA/Materials Studio-style tools
(coordination polyhedra, bond and angle distributions, morphology,
Miller planes and cleavage).

Only numpy/scipy (and ase for element data) are needed; Zeo++ is used when
installed. Each module is imported on first use, so ``import mace.analysis``
costs nothing.

The command line is ``mace analyze <tool> FILE``; the GUI is Analysis ›
Structure Analysis.
"""

TOOLS = ("porosity", "pxrd", "peaks", "index", "refine", "topology", "rcsr", "polyhedra",
         "bonds", "morphology", "planes", "zeopp")
