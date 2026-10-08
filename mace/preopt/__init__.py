"""Pre-optimization: a starting geometry for CRYSTAL, never a result.

Three tiers share one optimizer loop (``mace.gui.clean.clean``: FIRE,
FixSymmetry, an optional cell filter, the cancel hook):

* ``springs``: the GUI's spring model, always available, the default;
* ``gfnff``: GFN-FF through an external ``xtb`` program (6.7 or newer);
* ``mlip:<id>``: a universal ML potential (MACE-MPA-0 from ACEsuit's
  mace-torch, or CHGNet), run out of process in its own interpreter.

``core.preoptimize`` is the one pipeline the CLI (``mace preopt``) and the
GUI's Clean dialog both call. Nothing here imports torch or xtb, and the
package imports nothing heavy at import time.
"""
