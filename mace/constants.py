"""Canonical physical constants for MACE (single source of truth).

Pure literals, zero imports — any layer can import this without creating cycles.

These full-precision CODATA 2018 values are the single definition shared by
the mace package (units.py, dat_file_processor.py, property_extractor.py and
every module that converts Hartree, Bohr, J, kJ/mol or cm⁻¹ import from here;
1.1.6 corrected the Bohr radius, which was the CODATA 2014 0.52917721067, a
4e-10 relative change of every Bohr <-> Å conversion; the extractor's former truncated ``27.2114`` literals were
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
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# SI defining constants (exact since the 2019 SI redefinition; CODATA 2018):
# the Properties tab's unit conversions (mace/properties/units.py) derive
# every other factor from these and the two values above
ELEMENTARY_CHARGE_C = 1.602176634e-19
AVOGADRO_PER_MOL = 6.02214076e23
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299792458.0
BOLTZMANN_J_K = 1.380649e-23

# Derived from the values above (CODATA 2018)
HARTREE_TO_J = HARTREE_TO_EV * ELEMENTARY_CHARGE_C                      # 4.3597447222071e-18
EV_TO_KJ_MOL = ELEMENTARY_CHARGE_C * AVOGADRO_PER_MOL / 1000.0           # 96.48533212331002
HARTREE_TO_KJ_MOL = HARTREE_TO_EV * EV_TO_KJ_MOL                        # 2625.4996394799
CALORIE_J = 4.184                                                        # thermochemical calorie (exact)
HARTREE_TO_KCAL_MOL = HARTREE_TO_KJ_MOL / CALORIE_J                     # 627.5094740631
EV_TO_CM1 = ELEMENTARY_CHARGE_C / (PLANCK_J_S * SPEED_OF_LIGHT_M_S * 100.0)  # 8065.543937349212
HARTREE_TO_CM1 = HARTREE_TO_EV * EV_TO_CM1                              # 219474.6313632
HARTREE_TO_THZ = HARTREE_TO_EV * ELEMENTARY_CHARGE_C / PLANCK_J_S / 1e12  # 6579.683920502
THZ_TO_CM1 = 1e12 / (SPEED_OF_LIGHT_M_S * 100.0)                        # 33.35640951981520
