"""Finite-displacement phonons with phonopy, driven from a MACE parent.

``mace phonopy generate`` reads a finished CRYSTAL23 parent (deck + output),
asks phonopy for the displaced supercells and writes one EXTERNAL deck per
displacement (geometry in ``<job>.ext``, staged as fort.34 by the job script).
``mace phonopy collect`` checks every output, reads the forces and writes
phonopy's band structure, DOS and thermal properties.

The package is named ``phonopy_bridge`` so that it can never shadow the real
``phonopy`` package. Only :mod:`.parent` works without phonopy installed.
"""
