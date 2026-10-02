"""Synthetic ion/solvent geometries, decks and CRYSTAL output excerpts for the charge tests.

Every coordinate here is made up from textbook bond lengths (S-O 1.43, S-N 1.61,
C-H 1.09 A, ...); nothing comes from a calculation or from the group's data.
The geometries only have to be chemically sensible enough that bonding by
covalent radii finds the right molecules.
"""
import math
import random

# --------------------------------------------------------------- geometries

_TETRA = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
_OCTA = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _add(a, b, s=1.0):
    return tuple(x + s * y for x, y in zip(a, b))


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _star(centre, ligand, dirs, r):
    return [(centre, (0.0, 0.0, 0.0))] + [(ligand, tuple(r * x for x in _unit(d))) for d in dirs]


def _sulfonyl(s, sign, tail):
    """O2S-X on a sulfur at ``s``; ``sign`` mirrors x; tail is 'F' or 'CF3'."""
    atoms = [("S", s),
             ("O", _add(s, (sign * 0.31, 0.62, 1.25))),
             ("O", _add(s, (sign * 0.31, 0.62, -1.25)))]
    d = _unit((sign * 1.0, 1.22, 0.0))
    if tail == "F":
        atoms.append(("F", _add(s, d, 1.58)))
    else:
        c = _add(s, d, 1.82)
        atoms.append(("C", c))
        e1 = (-d[1], d[0], 0.0)
        for k in range(3):
            phi = 2 * math.pi * k / 3
            perp = tuple(math.cos(phi) * a + math.sin(phi) * b for a, b in zip(e1, (0.0, 0.0, 1.0)))
            atoms.append(("F", _add(c, tuple(0.334 * x + 0.943 * y for x, y in zip(d, perp)), 1.33)))
    return atoms


def fsi():
    return ([("N", (0.0, 0.0, 0.0))] + _sulfonyl((1.37, 0.85, 0.0), 1, "F")
            + _sulfonyl((-1.37, 0.85, 0.0), -1, "F"))


def tfsi():
    return ([("N", (0.0, 0.0, 0.0))] + _sulfonyl((1.37, 0.85, 0.0), 1, "CF3")
            + _sulfonyl((-1.37, 0.85, 0.0), -1, "CF3"))


def otf():
    atoms = [("S", (0.0, 0.0, 0.0)), ("C", (0.0, 0.0, 1.82))]
    for k in range(3):
        phi = 2 * math.pi * k / 3
        atoms.append(("F", (1.33 * 0.943 * math.cos(phi), 1.33 * 0.943 * math.sin(phi), 1.82 + 1.33 * 0.334)))
        atoms.append(("O", (1.44 * 0.943 * math.cos(phi + math.pi / 3),
                            1.44 * 0.943 * math.sin(phi + math.pi / 3), -1.44 * 0.334)))
    return atoms


def clo4():
    return _star("Cl", "O", _TETRA, 1.44)


def bf4():
    return _star("B", "F", _TETRA, 1.40)


def pf6():
    return _star("P", "F", _OCTA, 1.60)


def no3():
    return [("N", (0.0, 0.0, 0.0))] + [
        ("O", (1.25 * math.cos(2 * math.pi * k / 3), 1.25 * math.sin(2 * math.pi * k / 3), 0.0)) for k in range(3)]


def hno3():
    """Nitric acid: the NO3 core with an H on one O - a different formula, so neutral."""
    return [("N", (0.0, 0.0, 0.0)), ("O", (1.40, 0.0, 0.0)), ("H", (1.65, 0.94, 0.0)),
            ("O", (-0.605, 1.048, 0.0)), ("O", (-0.605, -1.048, 0.0))]


def so2f2():
    return [("S", (0.0, 0.0, 0.0)), ("O", tuple(1.40 * x for x in _unit(_TETRA[0]))),
            ("O", tuple(1.40 * x for x in _unit(_TETRA[1]))),
            ("F", tuple(1.53 * x for x in _unit(_TETRA[2]))),
            ("F", tuple(1.53 * x for x in _unit(_TETRA[3])))]


def ec():
    """Ethylene carbonate, C3H4O3 (planar ring, idealised)."""
    return [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 1.195, 0.0)),
            ("O", (1.111, -0.746, 0.0)), ("O", (-1.111, -0.746, 0.0)),
            ("C", (0.766, -2.145, 0.0)), ("C", (-0.766, -2.145, 0.0)),
            ("H", (1.150, -2.650, 0.890)), ("H", (1.150, -2.650, -0.890)),
            ("H", (-1.150, -2.650, 0.890)), ("H", (-1.150, -2.650, -0.890))]


def water():
    return [("O", (0.0, 0.0, 0.0)), ("H", (0.757, 0.586, 0.0)), ("H", (-0.757, 0.586, 0.0))]


IONS = {"FSI-": fsi, "TFSI-": tfsi, "OTf-": otf, "ClO4-": clo4, "NO3-": no3,
        "PF6-": pf6, "BF4-": bf4}
CATIONS = {"Li+": "Li", "Na+": "Na", "K+": "K", "Mg2+": "Mg", "Ca2+": "Ca", "Zn2+": "Zn"}


def place(atoms, shift):
    return [(s, _add(p, shift)) for s, p in atoms]


def cluster(*parts, spacing=6.0):
    """Molecules side by side along x, ``spacing`` A apart; a str part is a lone atom.

    A tuple part ``(atoms, shift)`` is placed at that shift instead.
    """
    out = []
    for k, part in enumerate(parts):
        if isinstance(part, tuple):
            atoms, shift = part
        else:
            atoms, shift = ([(part, (0.0, 0.0, 0.0))] if isinstance(part, str) else part), (k * spacing, 0.0, 0.0)
        out += place(atoms, shift)
    return out


def li_near(atoms, index, distance=1.95):
    """A Li placed ``distance`` A from atom ``index`` of ``atoms``, away from the molecule's centre."""
    centre = tuple(sum(p[i] for _, p in atoms) / len(atoms) for i in range(3))
    p = atoms[index][1]
    d = _unit(_add(p, centre, -1.0))
    return ("Li", _add(p, d, distance))


def shuffled(atoms, seed=7):
    """The same atoms in a scrambled order (fixed seed)."""
    order = list(range(len(atoms)))
    random.Random(seed).shuffle(order)
    return [atoms[i] for i in order]


Z = {"H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "P": 15,
     "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Zn": 30}


def molecule_deck(atoms, title="cluster", basis="BASISSET\nSOLDEF2MSVP\n", dft="HSESOL3C\nXLGRID\n",
                  spin=False, tail=""):
    """A MOLECULE deck in the shape cif2d12 writes."""
    rows = "".join(f"{Z[s]} {x:.6f} {y:.6f} {z:.6f} Biso 1.000000 {s}\n" for s, (x, y, z) in atoms)
    return (f"{title}\nMOLECULE\n1\n{len(atoms)}\n{rows}{basis}DFT\n" + ("SPIN\n" if spin else "")
            + f"{dft}ENDDFT\n{tail}TOLINTEG\n7 7 7 7 14\nTOLDEE\n9\nSCFDIR\nMAXCYCLE\n800\nFMIXING\n30\nDIIS\nPPAN\nEND\n")


def molecule_cif(atoms, box=30.0, name="cluster"):
    """A P1 box CIF holding ``atoms`` (Cartesian A) around the box centre."""
    lines = [f"data_{name}", f"_cell_length_a {box:.6f}", f"_cell_length_b {box:.6f}",
             f"_cell_length_c {box:.6f}", "_cell_angle_alpha 90.000000", "_cell_angle_beta 90.000000",
             "_cell_angle_gamma 90.000000", "_symmetry_space_group_name_H-M 'P 1'",
             "_symmetry_Int_Tables_number 1", "loop_", "_symmetry_equiv_pos_as_xyz", "  'x, y, z'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z"]
    for k, (s, (x, y, z)) in enumerate(atoms):
        lines.append(f"{s}{k + 1} {s} {0.5 + x / box:.6f} {0.5 + y / box:.6f} {0.5 + z / box:.6f}")
    return "\n".join(lines) + "\n"


def p1_cif(cell, frac_atoms, name="crystal"):
    a, b, c, al, be, ga = cell
    lines = [f"data_{name}", f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}",
             f"_cell_length_c {c:.6f}", f"_cell_angle_alpha {al:.6f}", f"_cell_angle_beta {be:.6f}",
             f"_cell_angle_gamma {ga:.6f}", "_symmetry_space_group_name_H-M 'P 1'",
             "_symmetry_Int_Tables_number 1", "loop_", "_symmetry_equiv_pos_as_xyz", "  'x, y, z'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z"]
    for k, (s, (x, y, z)) in enumerate(frac_atoms):
        lines.append(f"{s}{k + 1} {s} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ CRYSTAL output

def mulliken_rows(atoms, populations):
    return "\n".join(f"{i + 1:4d} {s.upper():<2s} {z:3d} {p:6.3f}  {p:6.3f}"
                     for i, ((s, z), p) in enumerate(zip(atoms, populations)))


def mulliken_block(atoms, populations, count, label=""):
    """One CRYSTAL Mulliken print: ``atoms`` = [(symbol, Z column)], ``populations`` the CHARGE column."""
    return (" MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n"
            + (f"     {label}\n" if label else "")
            + f" MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS  {count:10.6f}\n\n"
            + "  ATOM    Z CHARGE  A.O. POPULATION\n\n"
            + mulliken_rows(atoms, populations) + "\n\n")


def synth_out(atoms, title="cluster", header_electrons=None, mulliken=None, extra=""):
    """A MOLECULE SCF .out in CRYSTAL23's layout, with the lines MACE reads.

    ``atoms`` = [(symbol, (x, y, z))]. ``mulliken`` is the text of the Mulliken
    print(s) (see mulliken_block), or None for a run without PPAN.
    """
    n = len(atoms)
    rows = "\n".join(f"{i + 1:7d} T {Z[s]:3d} {s.upper():<2s}    {x: .12E} {y: .12E} {z: .12E}"
                     for i, (s, (x, y, z)) in enumerate(atoms))
    ne = header_electrons if header_electrons is not None else sum(Z[s] for s, _ in atoms)
    return f"""
 *******************************************************************************
 *                               CRYSTAL23                                     *
 *******************************************************************************
 EEEEEEEEEE STARTING  DATE 01 10 2026 TIME 10:00:00.0
 {title}

 MOLECULAR CALCULATION
 POINT  GROUP  N.  1                  :  1      OR  C1

 GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    0
 (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT {n:4d} - ATOMS IN THE UNIT CELL: {n:4d}
     ATOM             X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)
 *******************************************************************************
{rows}

 T = ATOM BELONGING TO THE ASYMMETRIC UNIT

 N. OF ELECTRONS PER CELL  {ne:4d}  EXCHANGE PSEUDO OVP (F(G))  (T4) 10**   -9
{extra}
 == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -1.0000000000000E+02 CYCLES  10
{mulliken or ''}
 EEEEEEEEEE TERMINATION  DATE 01 10 2026 TIME 10:01:00.0
"""
