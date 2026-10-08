"""Minimal synthetic CRYSTAL optimization outputs (generic, diamond-like) for
the convergence parser's stop cases that the real corpus lacks (manual
p.208 alternate stops, MAXCYCLE, OPT END - FAILED, atom-only runs). Only the
lines the parsers read are written; numbers are made up."""
from typing import List, Optional, Sequence

HEADER = """ CRYSTAL23 synthetic test output
 INFORMATION **** NEW DEFAULT **** FINALRUN 4 FOR GEOMETRY OPTIMIZATIONS
{units}
 *******************************************************************************
 {control} OPTIMIZATION CONTROL

 MAXIMUM GRADIENT COMPONENT   0.00045 MAXIMUM DISPLACEMENT COMPONENT     0.00180
 R.M.S. OF GRADIENT COMPONENT 0.00030 R.M.S. OF DISPLACEMENT COMPONENTS  0.00120
 THRESHOLD ON ENERGY CHANGE 0.100E-06 EXTRAPOLATING POLYNOMIAL ORDER           2
 MAXIMUM ALLOWED NUMBER OF STEPS {maxcycle:4d} SORTING OF ENERGY POINTS:               NO
 *******************************************************************************

 MAX NUMBER OF SCF CYCLES     800  CONVERGENCE ON DELTAP        10**-16
 WEIGHT OF F(I) IN F(I+1)      30% CONVERGENCE ON ENERGY        10**- 7
"""

UNITS = """ IMPORTANT INFORMATION:
 THE COORDINATE SYSTEM ADOPTED IN THE OPTIMIZATION IS NOT IN
 GENERAL HOMOGENEOUS UNITS. FOR THIS REASON THE GRADIENT AND
 HESSIAN MATRIX IN TERMS OF THIS SYSTEM ARE GIVEN WITHOUT
 UNITS. FOR MORE DETAILS SEE THE MANUAL."""


def block(a: float = 2.5, frac: Sequence[float] = (0.125, 0.125, 0.125), frac2: Optional[Sequence[float]] = None) -> str:
    f2 = frac2 if frac2 is not None else [-x for x in frac]
    vol = a ** 3 / 2 ** 0.5
    return f""" LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM
 PRIMITIVE CELL - CENTRING CODE 5/0 VOLUME=    {vol:.6f} - DENSITY  3.543 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     {a:.8f}     {a:.8f}     {a:.8f}    60.000000  60.000000  60.000000
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2
     ATOM                 X/A                 Y/B                 Z/C
 *******************************************************************************
      1 T   6 C     {frac[0]: .12E} {frac[1]: .12E} {frac[2]: .12E}
      2 T   6 C     {f2[0]: .12E} {f2[1]: .12E} {f2[2]: .12E}

"""


def crit(vals: Sequence[float], yes: Sequence[bool], disp: bool = True) -> str:
    names = [("MAX GRADIENT", 0.00045), ("RMS GRADIENT", 0.0003), ("MAX DISPLAC.", 0.0018), ("RMS DISPLAC.", 0.0012)]
    out = []
    for k, ((nm, thr), v, y) in enumerate(zip(names, vals, yes)):
        if k >= 2 and not disp:
            break
        out.append(f" {nm:<13}{v:12.6f}  THRESHOLD {thr:20.6f} CONVERGED {'YES' if y else 'NO'}")
    return "\n".join(out) + "\n"


def opt_output(energies: List[float], des: List[Optional[float]], crits: List[tuple], *, end: Optional[str] = "CONVERGED",
               units: bool = True, maxcycle: int = 800, warning: Optional[str] = None, cells: Optional[List[float]] = None,
               final: bool = True, cpu: bool = True, restart_after: Sequence[int] = (), control: str = "COMBINED CELL/ATOM") -> str:
    """An OPT output with one POINT per energy. ``crits[k]`` = (values, yes-flags)."""
    cells = cells or [2.5 + 0.001 * k for k in range(len(energies))]
    text = HEADER.format(units=UNITS if units else "", maxcycle=maxcycle, control=control)
    text += block(cells[0])
    for k, e in enumerate(energies):
        p = k + 1
        text += f" *******************************************************************************\n COORDINATE AND CELL OPTIMIZATION - POINT {p:4d}\n" \
                " *******************************************************************************\n"
        if k:
            text += block(cells[k])
        if k == 0:
            text += " CYC   0 ETOT(AU) -7.600000000000E+01 DETOT -7.60E+01 tst  0.00E+00 PX  1.00E+00\n"
            text += f" CYC   1 ETOT(AU) {e:.12E} DETOT -1.00E-08 tst  1.00E-09 PX  1.00E-04\n"
            text += f" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {e:.13E} CYCLES   1\n"
            text += f" TOTAL ENERGY(DFT)(AU)(  1) {e:.13E} DE 1.0E-08 tester 1.0E-09\n"
        else:
            text += f" TOTAL ENERGY(DFT)(AU)(  4) {e:.13E}     DE (AU) {des[k]:10.3E}\n"
        vals, yes = crits[k]
        text += "\n" + crit(vals, yes, disp=k > 0) + "\n"
        if k == 0:
            text += " - INFORMATION ON SCF MOVED TO SCFOUT.LOG\n"
        if p in restart_after:
            text += " FINALRUN OPTION: RE-STARTING OPTIMIZATION\n"
    if warning:
        text += "\n WARNINGWARNINGWARNINGWARNINGWARNING\n\n " + warning + "\n\n WARNINGWARNINGWARNINGWARNINGWARNING\n\n"
    if end:
        text += f" ******************************************************************\n" \
                f" * OPT END - {end} * E(AU): {energies[-1]:.12E}  POINTS {len(energies):4d} *\n" \
                f" ******************************************************************\n"
        if final and end == "CONVERGED":
            text += " FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)\n"
            text += block(cells[-1])
    if cpu:
        text += " EEEEEEEEEE STARTING  DATE 01 01 2026 TIME 00:00:00.0\n    TOTAL CPU TIME =     1.000\n" if end else ""
    return text
