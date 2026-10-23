"""Force fields, tight binding, charge models and ML potentials (``mace calc``).

One package, one backend per module, one interface (``base.Backend``):

* ``uff``: UFF and UFF4MOF, native (numpy);
* ``eqeq``: EQeq partial charges for periodic structures, native;
* ``xtb``: GFN2-xTB, GFN1-xTB and GFN-FF through the external ``xtb``
  program (the same detection and runner as Clean's GFN-FF tier), or the
  tblite Python package when it is installed;
* ``dftb``: DFTB+ through ase's calculator (Slater-Koster files or DFTB+'s
  own xTB Hamiltonian);
* ``mlip``: the ML potentials of ``mace/preopt``'s registry (MACE-MP from
  ACEsuit's mace-torch, which is unrelated to this MACE tool; ORB-v3;
  MatterSim; CHGNet), run in their own interpreter by the same
  worker the GUI's Clean uses, on GPU 0 when there is one.

``tasks.run(atoms, spec)`` does a single point, an optimization, finite-
difference frequencies, molecular dynamics or partial charges with any of
them. Importing this package imports numpy only; every optional code is
loaded when a calculation needs it.
"""
from .base import CalcError, CalcSpec, Canceled, TASKS  # noqa: F401
