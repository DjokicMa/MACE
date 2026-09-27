"""The best geometry an interrupted optimization reached, written back into its
own deck.

When a geometry optimization cannot simply continue with OPTGEOM RESTART (the
RESTART rerun itself aborted, or its OPTINFO.DAT is gone), the next best thing
is a fresh optimization that starts from the lowest-energy point already
reached, instead of from the original geometry. This module finds that point
in the killed runs' outputs and rewrites ONLY the geometry records of the deck.

Why the .out and not the optc<N> / fort.34 files in scratch:
  * the .out sits in the job's own directory; scratch can be invisible from
    where the recovery runs ($SCRATCH empty on some nodes) and is purged;
  * every optimization point prints its geometry in exactly the frame the
    deck uses - for a centred lattice CRYSTAL adds the
    "CRYSTALLOGRAPHIC CELL" and "COORDINATES IN THE CRYSTALLOGRAPHIC CELL"
    after the primitive cell - followed by that point's energy, so geometry
    and energy come from the same place;
  * optc files are EXTERNAL-format (primitive vectors, Cartesian atoms, the
    symmetry operators), would have to be converted back to the deck's
    space-group form, and are not one per point (a real scratch directory
    holds optc001-005, 007, 009... with the rejected steps missing).
MACE's own CrystalOutputParser does not help here: without
"FINAL OPTIMIZED GEOMETRY" it falls back to "GEOMETRY FOR WAVE FUNCTION",
which is the INPUT geometry of the run.

What an output shows (checked on the real OPT corpus and the HPCC runs):
  * "GEOMETRY FOR WAVE FUNCTION" near the top is the deck's own geometry, even
    in a RESTART run;
  * "<TYPE> OPTIMIZATION - POINT N" starts each point; from point 2 on (and at
    the first point of a RESTART run) the point prints its geometry, then
    "TOTAL ENERGY(...)(AU)( n) E  DE (AU) d", where d is E minus the energy of
    the reference (best) point so far;
  * point 1 prints no geometry (it is the deck's) and no DE, so its energy is
    E(point 2) - DE(point 2).

Safety: the deck and the output are tied together before anything is written.
The output's own header geometry must reproduce the deck's cell parameters and
atoms (same atomic numbers, same order, coordinates equal up to one common
origin shift and whole lattice translations). The same offsets are then
applied to the best point, whose atoms must stay close to the deck's. Anything
else - another dimensionality, EXTERNAL geometry, a cell the deck's parameter
line cannot express - is refused, and the caller does not guess.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_POINT_RE = re.compile(r'OPTIMIZATION - POINT\s+(\d+)')
_ENERGY_RE = re.compile(
    r'TOTAL ENERGY\([^)]*\)\(AU\)\(\s*\d+\)\s*(-?\d+\.\d+E[+-]\d+)\s+DE \(AU\)\s+(-?\d+\.\d+E[+-]\d+)')
_CENTRING_RE = re.compile(r'PRIMITIVE CELL - CENTRING CODE\s+(\d+)')
_ATOM_RE = re.compile(
    r'^\s*\d+\s+([TF])\s+(\d+)\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
_FLOAT = r'-?\d+\.\d+(?:[EeDd][+-]?\d+)?'

# Atoms may move this far (fractional) between the deck and the best point
# before the match is distrusted - an optimization step moves atoms by a few
# hundredths; a different representative of the atom's orbit jumps further.
MAX_FRACTIONAL_MOVE = 0.25


class GeometryTransferError(Exception):
    """The best point cannot be written into this deck safely."""


@dataclass
class Geometry:
    cell: Tuple[float, ...]                      # a b c alpha beta gamma (crystallographic)
    atoms: List[Tuple[int, float, float, float]]  # asymmetric unit, in print order
    lattice: str = 'P'                           # lattice letter of the space group


@dataclass
class Point:
    number: int
    energy: float
    geometry: Optional[Geometry]   # None: the run's input geometry (point 1)
    source: str = ''


def _floats(line: str) -> List[float]:
    return [float(x.replace('D', 'E').replace('d', 'e')) for x in re.findall(_FLOAT, line)]


def _read_atoms(lines: Sequence[str], i: int) -> Tuple[List[Tuple[int, float, float, float]], int]:
    """T atoms of an atom table starting after the header at line i."""
    j = i + 1
    while j < len(lines) and '*****' not in lines[j]:
        j += 1
    j += 1
    atoms = []
    while j < len(lines):
        m = _ATOM_RE.match(lines[j])
        if not m:
            break
        if m.group(1) == 'T':
            atoms.append((int(m.group(2)), float(m.group(3)), float(m.group(4)),
                          float(m.group(5))))
        j += 1
    return atoms, j


def _cell_after(lines: Sequence[str], i: int) -> Optional[Tuple[float, ...]]:
    """The six numbers on the line after the 'A B C ALPHA BETA GAMMA' header
    that follows line i."""
    for j in range(i + 1, min(i + 4, len(lines))):
        if 'ALPHA' in lines[j] and 'GAMMA' in lines[j] and j + 1 < len(lines):
            vals = _floats(lines[j + 1])
            return tuple(vals[:6]) if len(vals) >= 6 else None
    return None


def read_geometry(lines: Sequence[str], start: int, stop: int) -> Optional[Geometry]:
    """The geometry printed in lines[start:stop], in the crystallographic frame.

    A primitive (centring code 1) cell is its own crystallographic cell; for a
    centred one the crystallographic block must be there, or None.
    """
    centring = None
    prim_cell = prim_atoms = cryst_cell = cryst_atoms = None
    i = start
    while i < stop:
        line = lines[i]
        m = _CENTRING_RE.search(line)
        if m and prim_cell is None:
            centring = int(m.group(1))
            prim_cell = _cell_after(lines, i)
        elif 'ATOMS IN THE ASYMMETRIC UNIT' in line and prim_atoms is None:
            prim_atoms, i = _read_atoms(lines, i + 1)
            continue
        elif 'CRYSTALLOGRAPHIC CELL (VOLUME=' in line and cryst_cell is None:
            cell = _floats(lines[i + 2]) if i + 2 < len(lines) else []
            cryst_cell = tuple(cell[:6]) if len(cell) >= 6 else None
        elif 'COORDINATES IN THE CRYSTALLOGRAPHIC CELL' in line and cryst_atoms is None:
            cryst_atoms, i = _read_atoms(lines, i)
            continue
        elif 'TOTAL ENERGY' in line or _POINT_RE.search(line):
            break
        i += 1
    if cryst_cell and cryst_atoms:
        return Geometry(cryst_cell, cryst_atoms)
    if centring == 1 and prim_cell and prim_atoms:
        return Geometry(prim_cell, prim_atoms)
    return None


def header_geometry(out_text: str) -> Optional[Geometry]:
    """The run's input geometry ("GEOMETRY FOR WAVE FUNCTION")."""
    lines = out_text.splitlines()
    m = re.search(r'^\s*SPACE GROUP[^:]*:\s*([A-Z])', out_text, re.M)
    for i, line in enumerate(lines):
        if 'GEOMETRY FOR WAVE FUNCTION' in line:
            stop = next((j for j in range(i + 1, len(lines)) if _POINT_RE.search(lines[j])
                         or 'SYMMOPS' in lines[j]), len(lines))
            geom = read_geometry(lines, i, stop)
            if geom is not None and m:
                geom.lattice = m.group(1)
            return geom
    return None


def optimization_points(out_text: str, source: str = '') -> List[Point]:
    """Every optimization point of one run that has both a geometry and an
    energy. Point 1 of a fresh run (the deck's geometry) is returned with
    geometry None and the energy the optimizer compared point 2 against."""
    lines = out_text.splitlines()
    starts = [(i, int(m.group(1))) for i, l in enumerate(lines)
              for m in [_POINT_RE.search(l)] if m]
    points = []
    for k, (i, n) in enumerate(starts):
        stop = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        energy = de = None
        e_line = None
        for j in range(i, stop):
            m = _ENERGY_RE.search(lines[j])
            if m:
                energy, de, e_line = float(m.group(1)), float(m.group(2)), j
                break
        if energy is None:
            continue
        geom = read_geometry(lines, i + 1, e_line)
        if geom is None:
            continue
        points.append(Point(n, energy, geom, source))
        # The point before it, when that was point 1 of this very run: its
        # energy is what this DE was measured from.
        if k == 1 and starts[0][1] == 1 and not any(p.number == 1 for p in points):
            points.insert(0, Point(1, energy - de, None, source))
    return points


def best_point(points: Sequence[Point]) -> Optional[Point]:
    return min(points, key=lambda p: p.energy) if points else None


# ------------------------------------------------------------------ the deck

@dataclass
class DeckGeometry:
    cell_line: int                    # index of the cell-parameter line
    atom_lines: List[int]             # indices of the atom lines
    cell: List[float]
    atoms: List[Tuple[int, float, float, float]]
    extra: Dict = field(default_factory=dict)


def parse_deck_geometry(lines: Sequence[str]) -> DeckGeometry:
    """Locate the geometry records of a 3D CRYSTAL deck given by space group.

    Layout (CRYSTAL23 manual, geometry input): title / CRYSTAL /
    IFLAG IFHR IFSO / space group / [origin shift when IFSO > 1] /
    cell parameters / number of atoms / one line per atom.
    """
    if len(lines) < 7 or lines[1].strip().upper() != 'CRYSTAL':
        raise GeometryTransferError('not a 3D CRYSTAL deck (only those are rewritten)')
    flags = lines[2].split()
    if len(flags) < 3 or not all(f.lstrip('-').isdigit() for f in flags[:3]):
        raise GeometryTransferError(f'unexpected IFLAG IFHR IFSO line: {lines[2].strip()!r}')
    i = 4
    if int(flags[2]) > 1:
        i += 1
    cell_line = i
    try:
        cell = _leading_numbers(lines[cell_line])
        natoms = int(lines[cell_line + 1].split()[0])
    except (ValueError, IndexError):
        raise GeometryTransferError('cell parameters / atom count not where expected')
    if not cell or len(cell) > 6 or natoms < 1:
        raise GeometryTransferError('cell parameters / atom count not where expected')
    atom_lines = list(range(cell_line + 2, cell_line + 2 + natoms))
    atoms = []
    for k in atom_lines:
        parts = lines[k].split() if k < len(lines) else []
        try:
            atoms.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except (ValueError, IndexError):
            raise GeometryTransferError(f'atom line {k + 1} is not "Z x y z ..."')
    return DeckGeometry(cell_line, atom_lines, cell, atoms)


def _leading_numbers(line: str) -> List[float]:
    """The numbers a record starts with; CRYSTAL reads only those, so a
    trailing comment ('3.5443 #a=b=c cubic') is not part of the cell."""
    vals = []
    for tok in line.split():
        try:
            vals.append(float(tok))
        except ValueError:
            break
    return vals


# Which crystallographic parameters (a b c alpha beta gamma = 0..5) a deck's
# cell line holds, by how many it has. Several readings are possible for 2 and
# 4 values; the one that reproduces the deck from the output's header wins.
_CELL_READINGS = {
    1: [(0,)],
    2: [(0, 2), (0, 3)],           # a c (tetragonal, hexagonal) or a alpha (rhombohedral axes)
    3: [(0, 1, 2)],
    4: [(0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 3)],   # monoclinic: beta, gamma or alpha unique
    6: [(0, 1, 2, 3, 4, 5)],
}


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _cell_reading(deck: DeckGeometry, header: Geometry) -> Tuple[int, ...]:
    for reading in _CELL_READINGS.get(len(deck.cell), []):
        if all(_close(deck.cell[k], header.cell[idx], 2e-5 if idx < 3 else 2e-4)
               for k, idx in enumerate(reading)):
            return reading
    raise GeometryTransferError(
        f'the output header cell {header.cell} does not reproduce the deck cell {deck.cell}')


# Translations that map a centred lattice onto itself (crystallographic
# frame, International Tables). CRYSTAL prints each atom as any one of these
# images - the corpus has C 2/m and F m-3 decks whose atoms come back shifted
# by (1/2,1/2,0) or (1/2,0,1/2) - so they are all "the same atom".
_CENTRING_TRANSLATIONS = {
    'P': [(0, 0, 0)],
    'A': [(0, 0, 0), (0, .5, .5)],
    'B': [(0, 0, 0), (.5, 0, .5)],
    'C': [(0, 0, 0), (.5, .5, 0)],
    'I': [(0, 0, 0), (.5, .5, .5)],
    'F': [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)],
    'R': [(0, 0, 0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],   # hexagonal axes
}


def _nearest_image(pos, target, translations):
    """The image of `pos` under the lattice (and centring) translations that
    lies closest to `target`, and its largest fractional distance from it."""
    best, best_d = None, None
    for t in translations:
        cand = [pos[c] + t[c] for c in range(3)]
        cand = [cand[c] - round(cand[c] - target[c]) for c in range(3)]
        d = max(abs(cand[c] - target[c]) for c in range(3))
        if best_d is None or d < best_d:
            best, best_d = cand, d
    return best, best_d


def _origin_shift(deck: DeckGeometry, header: Geometry) -> Tuple[float, float, float]:
    """The one translation taking the output's frame to the deck's: every
    header atom plus it must be a lattice/centring image of its deck atom,
    or the frames do not agree."""
    if [a[0] for a in deck.atoms] != [a[0] for a in header.atoms]:
        raise GeometryTransferError(
            'the output header atoms are not the deck atoms (count, order or atomic number)')
    translations = _CENTRING_TRANSLATIONS.get(header.lattice)
    if translations is None:
        raise GeometryTransferError(f'unknown lattice type {header.lattice!r}')
    shift = tuple(deck.atoms[0][c] - header.atoms[0][c] for c in (1, 2, 3))
    for d, h in zip(deck.atoms, header.atoms):
        _, dist = _nearest_image([h[c] + shift[c - 1] for c in (1, 2, 3)], d[1:], translations)
        if dist > 1e-4:
            raise GeometryTransferError(
                'the output header coordinates differ from the deck by more than an '
                'origin shift and lattice translations')
    return shift


def _fmt_like(original: str, value: float) -> str:
    """Format as the deck wrote the number it replaces."""
    if 'E' in original.upper():
        mantissa = original.upper().split('E')[0]
        decimals = len(mantissa.split('.')[1]) if '.' in mantissa else 12
        return f'{value:.{decimals}E}'
    decimals = len(original.split('.')[1]) if '.' in original else 8
    return f'{value:.{max(decimals, 8)}f}'


def _replace_tokens(line: str, replacements: Dict[int, float]) -> str:
    """Replace whitespace-separated tokens by index, keeping indentation and
    the line ending."""
    body = line.rstrip('\r\n')
    ending = line[len(body):]
    indent = body[:len(body) - len(body.lstrip())]
    tokens = body.split()
    for k, v in replacements.items():
        tokens[k] = _fmt_like(tokens[k], v)
    return indent + ' '.join(tokens) + ending


def rewrite_geometry(d12_text: str, header: Geometry, point: Geometry) -> str:
    """The deck with its cell parameters and atom coordinates taken from
    `point`; every other byte stays as it was. `header` is the geometry the
    same deck produced in an output (its "GEOMETRY FOR WAVE FUNCTION"), which
    fixes how deck and output frames correspond."""
    lines = d12_text.splitlines(keepends=True)
    deck = parse_deck_geometry([l.rstrip('\r\n') for l in lines])
    reading = _cell_reading(deck, header)
    shift = _origin_shift(deck, header)
    if [a[0] for a in point.atoms] != [a[0] for a in deck.atoms]:
        raise GeometryTransferError('the best point lists different atoms than the deck')
    translations = _CENTRING_TRANSLATIONS[header.lattice]

    new_atoms = []
    for (z, *xyz), (_, *dxyz) in zip(point.atoms, deck.atoms):
        # CRYSTAL may print another lattice/centring image of the atom than
        # the deck has ((-.5,-.5,-.5) then (.5,.5,.5) at a later point): take
        # the image next to the deck's atom, so the deck keeps its convention.
        moved, dist = _nearest_image([xyz[c] + shift[c] for c in range(3)], dxyz, translations)
        if dist > MAX_FRACTIONAL_MOVE:
            raise GeometryTransferError(
                f'atom Z={z} would move more than {MAX_FRACTIONAL_MOVE} (fractional) - '
                f'not trusted as the same atom')
        new_atoms.append(moved)

    lines[deck.cell_line] = _replace_tokens(
        lines[deck.cell_line], {k: point.cell[idx] for k, idx in enumerate(reading)})
    for k, xyz in zip(deck.atom_lines, new_atoms):
        lines[k] = _replace_tokens(lines[k], {1: xyz[0], 2: xyz[1], 3: xyz[2]})
    return ''.join(lines)


def collect_points(outputs: Sequence[Path]) -> List[Point]:
    points = []
    for p in outputs:
        try:
            text = Path(p).read_text(errors='ignore')
        except OSError:
            continue
        points.extend(optimization_points(text, source=Path(p).name))
    return points
