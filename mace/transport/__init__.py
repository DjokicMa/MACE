"""The relaxation time tau of a BOLTZTRA run, estimated from simulation.

CRYSTAL23's Boltzmann transport (manual 14.5, p. 314) works in the
constant relaxation-time approximation: sigma and the electronic thermal
conductivity scale with RELAXTIM (default 10 fs), the Seebeck coefficient
does not. This package gives a physically documented tau instead of a
guessed one:

* ``dpt`` - deformation-potential theory (acoustic-phonon scattering,
  Bardeen & Shockley 1950) for 3D, 2D and 1D carriers, the energy-resolved
  tau(E, T) of 3D parabolic bands, and tau from a measured mobility;
* ``strain`` - the strained single-point decks (CRYSTAL's ELASTIC
  deformation, manual p. 47) and the fits that give the deformation
  potential E1 and the stretching modulus C from their outputs;
* ``cli`` - ``mace tau estimate|strain|fit``.

Every result carries its provenance: the method, the formula, the
reference, and where each input came from.
"""
