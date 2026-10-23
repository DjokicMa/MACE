"""Finite-displacement phonons with phonopy, driven from a MACE parent.

``mace opt2d12`` (choice 8 / ``--calc-type PHONOPY``,
Crystal_d12/d12_calc_phonopy.py) reads a finished CRYSTAL23 parent (deck +
output), asks phonopy for the displaced supercells (:mod:`.structure`) and
writes one EXTERNAL deck per displacement with its own deck writer (geometry
in ``<job>.ext``, staged as fort.34 by the job script).
``mace analyze phonopy DIR`` (:mod:`.cli`, :mod:`.collect`) checks every
output, reads the forces and writes phonopy's band structure, DOS and thermal
properties; the completion callback of a set's last job runs it too.

The package is named ``phonopy_bridge`` so that it can never shadow the real
``phonopy`` package. Only :mod:`.parent` works without phonopy installed.
"""
