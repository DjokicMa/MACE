"""Synthetic CRYSTAL23 output text for the k-path tests.

The layout is copied from real CRYSTAL23 outputs: the input echo
("LATTICE PARAMETERS ... - CONVENTIONAL CELL" / "- PRIMITIVE CELL"), the
SUPERCEL banner with its "EXPANSION MATRIX OF PRIMITIVE CELL", and the
computed geometry blocks ("GEOMETRY FOR WAVE FUNCTION", optimisation steps,
"FINAL OPTIMIZED GEOMETRY") with their cell, atoms, transformation matrix
and crystallographic cell.
"""

STARS = " " + "*" * 79


def echo(sg_symbol, conventional, primitive=None, centro=True):
    kind = "CENTROSYMMETRIC" if centro else "NONCENTROSYMMETRIC"
    text = (f" SPACE GROUP ({kind})        :  {sg_symbol}\n\n"
            " LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL\n"
            "        A           B           C        ALPHA        BETA       GAMMA\n"
            + "".join(f" {v:11.5f}" for v in conventional) + "\n\n")
    if primitive is not None:
        text += (" LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL\n"
                 "       A          B          C        ALPHA      BETA     GAMMA        VOLUME\n"
                 + "".join(f" {v:10.5f}" for v in primitive) + "     11.135747\n\n")
    return text


def supercel(rows, scelphono=False):
    text = (STARS + "\n * SUPERCELL OPTION\n" + STARS + "\n\n\n"
            "        EXPANSION MATRIX OF PRIMITIVE CELL\n"
            + "".join(f" E{i + 1}" + "".join(f" {v:9.3f}" for v in row) + "\n"
                      for i, row in enumerate(rows))
            + "\n NUMBER OF ATOMS PER SUPERCELL    16\n\n"
            " **************************** SUPERCELL GENERATED ****************************\n\n")
    if scelphono:
        text += (" **** ATOMS IN THE SUPERCELL REORDERED FOR PHONON CALCULATION\n"
                 "      ATOMS IN THE SMALL CELL ON TOP\n\n")
    return text


def block(params, atoms, header="GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    3",
          n_atoms=None, w=None, conventional=None, centering="1/0"):
    """A computed geometry block. ``atoms``: (Z, symbol, x, y, z) in the primitive cell."""
    n = len(atoms) if n_atoms is None else n_atoms
    text = (f" {header}\n"
            " (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)\n"
            + STARS + "\n"
            " LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n"
            f" PRIMITIVE CELL - CENTRING CODE {centering} VOLUME=    11.135747 - DENSITY  3.579 g/cm^3\n"
            "         A              B              C           ALPHA      BETA       GAMMA \n"
            "  " + "".join(f" {v:14.8f}" for v in params[:3])
            + "".join(f" {v:10.6f}" for v in params[3:]) + "\n"
            + STARS + "\n"
            f" ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL: {n:4d}\n"
            "     ATOM                 X/A                 Y/B                 Z/C    \n"
            + STARS + "\n")
    for i, (z, sym, x, y, zz) in enumerate(atoms):
        text += f" {i + 1:6d} {'T' if i == 0 else 'F'} {z:3d} {sym:<2s} {x:20.12E}{y:20.12E}{zz:20.12E}\n"
    text += "\n"
    if w is not None:
        text += (" TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC CELL\n"
                 + "".join(f" {v:7.4f}" for row in w for v in row) + "\n\n")
    if conventional is not None:
        text += (STARS + "\n CRYSTALLOGRAPHIC CELL (VOLUME=         44.54298617)\n"
                 "         A              B              C           ALPHA      BETA       GAMMA \n"
                 "  " + "".join(f" {v:14.8f}" for v in conventional[:3])
                 + "".join(f" {v:10.6f}" for v in conventional[3:]) + "\n\n"
                 " COORDINATES IN THE CRYSTALLOGRAPHIC CELL\n")
    text += "\n T = ATOM BELONGING TO THE ASYMMETRIC UNIT\n\n"
    return text


def final_block(params, atoms, **kw):
    return block(params, atoms, header="FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3",
                 **kw)


def opt_step(params, atoms, **kw):
    return (" CRYSTALLOGRAPHIC CELL NOT OPTIMIZED - OPT STEP\n\n"
            + block(params, atoms, header="GEOMETRY AT THIS STEP", **kw))
