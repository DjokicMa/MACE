"""An EOS output in CRYSTAL23's format (the lines mace.elastic reads: the atoms of the cell, SORTING
VOLUMES/ENERGIES and FITTING USING ALL POINTS) for a made-up phase, to compare enthalpies against a
real run. The energies of its points are the Birch-Murnaghan curve of the parameters given."""
GPA_A3_HA = 1e-21 / 4.3597447222071e-18


def bm_energy(v, v0, e0, b0, bp):
    x = (v0 / v) ** (2 / 3)
    return e0 + 9 * v0 * b0 * GPA_A3_HA / 16 * ((x - 1) ** 3 * bp + (x - 1) ** 2 * (6 - 4 * x))


def eos_output(v0, e0, b0, bp, atoms=(("6", "C"), ("6", "C")), n=7, lo=0.92, hi=1.08):
    rows = "\n".join(f"      {k + 1} T {z:>3} {el:<2}    {0.125 * k:.12E}  {0.125 * k:.12E}  {0.125 * k:.12E}"
                     for k, (z, el) in enumerate(atoms))
    vs = [v0 * (lo + (hi - lo) * k / (n - 1)) for k in range(n)]
    pts = "\n".join(f"      {v:9.6f}   {bm_energy(v, v0, e0, b0, bp):.12E}" for v in vs)
    fits = "\n".join(f" {name:<30}{v0:8.4f}   {e0:15.8f}  {b0:6.2f}     {bp:4.2f}"
                     for name in ("MURNAGHAN 1944", "BIRCH-MURNAGHAN 1947", "POIRIER-TARANTOLA 1998", "VINET 1987"))
    return f""" CRYSTAL CALCULATION
 ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:    {len(atoms)}
     ATOM                 X/A                 Y/B                 Z/C
 *******************************************************************************
{rows}

 PRIMITIVE CELL - CENTRING CODE 5/0 VOLUME=    {v0:.6f} - DENSITY  3.700 g/cm^3
 N. OF ATOMS PER CELL         {len(atoms)}  COULOMB OVERLAP TOL         (T1) 10**   -7

 SORTING VOLUMES/ENERGIES

     VOLUME (A^3)    ENERGY (a.u.)

{pts}

 +++++++ FITTING USING ALL POINTS +++++++

 EQUATION OF STATE             VOL(A^3)        E(AU)       BM(GPa)   BM PRIME
 -------------------------------------------------------------------------------
{fits}

 POLYNOMIAL FITTING            VOL(A^3)        E(AU)       BM(GPa)
"""
