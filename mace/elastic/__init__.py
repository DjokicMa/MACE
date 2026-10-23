"""Mechanical properties of CRYSTAL23 runs: ELASTCON, ELAPIEZO, PIEZOCON,
PIEZOCP, PHOTOELA and EOS.

* ``analysis`` reads the tensors and the equation-of-state fits a run
  prints and derives the moduli (Voigt-Reuss-Hill, Young's modulus,
  Poisson's ratio, anisotropy, Born stability, Debye temperature).
* ``cli`` is ``mace analyze elastic``.

The decks themselves are written by ``Crystal_d12/d12_calc_elastic.py``
through ``mace opt2d12`` (choices 4-7 / ``--calc-type ELASTIC|EOS|QHA|CPHF``).
"""
