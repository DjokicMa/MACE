#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin-orbit ECPs and two-component SOC decks for CRYSTAL23
---------------------------------------------------------
CRYSTAL23 puts spin-orbit coupling into a 2c-SCF through the SOREP part of an
effective core potential (manual sec. 6.2, keyword SOC, p. 175). An ECP with a
SOREP is entered by hand with INPSOC instead of INPUT (manual sec. 3.2,
pp. 84-87):

    INPSOC
    PSONAME SOSCALE              convention for the SOREP coefficients, scale X
    ZNUC M M0 M1 M2 M3 M4        as for INPUT
    ALFKL CGKL CSOGKL NKL        M+M0+...+M4 rows: exponent, AREP and SOREP
                                 coefficients, power of r

This module converts the Stuttgart spin-orbit ECPs kept in MOLPRO format in
Crystal_d3/Archived/basis/sopseud/2NN.mol into those records, and turns a
finished MACE deck into a two-component SOC deck. An atom's INPUT ECP is
given its spin-orbit part only when it is, number for number, the scalar part
of that spin-orbit ECP, so the valence basis beside it - published with that
potential - stays with it; any other ECP is refused, never re-paired.

The conversion follows the manual's own worked example (p. 87), which enters
the Stuttgart ECP28MWB potential for Eu - the same numbers as sopseud/263.mol:
  - the first MOLPRO block is the l = L term (the M record); a term with a
    zero coefficient there is left out (the example's "zero l = L = 5 terms");
  - the next blocks are l = 0 .. L-1 (the M0 .. M4 records);
  - the spin-orbit blocks, for l = 1 .. lso, give CSOGKL on the row with the
    same exponent and power, with PSONAME INTERNAL and SOSCALE 1.0: the MOLPRO
    SO coefficients already carry the 2/(2l+1) factor of eq. 3.26b;
  - a MOLPRO power n means r^(n-2), so NKL = n - 2 (as the manual's note on
    Hay-Wadt tables, p. 86).

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))

from basis_file_check import basis_block_problem, shell_problem  # noqa: E402

# Telling a two-component run from its deck or .out. Defined once in
# mace/utils/calc_detection.py (standard library only); a standalone run loads
# it by path, and a flat copy without the mace package uses the same tests
# inline.
try:
    from mace.utils.calc_detection import (is_spin_orbit_deck, is_two_component_deck,
                                           is_two_component_output, staged_2c_guess,
                                           two_component_guess)
except ImportError:
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_mace_calc_detection",
            os.path.join(_HERE, "..", "mace", "utils", "calc_detection.py"))
        _calc_detection = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_calc_detection)
        is_spin_orbit_deck = _calc_detection.is_spin_orbit_deck
        is_two_component_deck = _calc_detection.is_two_component_deck
        is_two_component_output = _calc_detection.is_two_component_output
        staged_2c_guess = _calc_detection.staged_2c_guess
        two_component_guess = _calc_detection.two_component_guess
    except Exception:  # pragma: no cover - flat copy without the mace package
        def _twocompon_records(content):
            records = None
            for line in content.splitlines()[1:]:
                record = line.strip().upper()
                if records is None:
                    records = set() if record == "TWOCOMPON" else None
                elif record in ("END", "ENDTWO"):
                    return records
                else:
                    records.add(record)
            return records

        def is_two_component_deck(content):
            return _twocompon_records(content) is not None

        def is_spin_orbit_deck(content):
            return "SOC" in (_twocompon_records(content) or ())

        def is_two_component_output(content):
            return re.search(r"^[ \t]*(?:TOTAL [XY]-COMP MAGNETIZATION[ \t]"
                             r"|- NUMBER OF FULLY OCCUPIED/TOTAL SPINORS[ \t])",
                             content, re.MULTILINE) is not None

        def two_component_guess(content):
            block = _twocompon_records(content) or ()
            return next((r for r in ("GUESSPSO", "GUESSPNOSO") if r in block), None)

        def staged_2c_guess(deck_path):
            try:
                with open(deck_path, errors="replace") as fh:
                    record = two_component_guess(fh.read())
            except OSError:
                return None
            return None if record is None else (record, os.path.splitext(str(deck_path))[0] + ".f20")

SOPSEUD_DIR = os.path.join(_HERE, "..", "Crystal_d3", "Archived", "basis", "sopseud")

# INPSOC takes M0..M4, so semi-local terms run up to l = 4 and the l = L term
# is at most L = 5 (manual p. 84 and eq. 3.24).
MAX_LOCAL_L = 5


class SocError(ValueError):
    """A SOC deck or SO-ECP that MACE must not write."""


@dataclass
class Term:
    """One Gaussian term: coefficient * r^power * exp(-exponent r^2).

    ``exponent`` and ``coefficient`` keep the digits they were read with, so
    writing them back copies them exactly. ``power`` is CRYSTAL's NKL."""
    exponent: str
    coefficient: str
    power: int

    def values(self) -> Tuple[float, float, int]:
        return (float(self.exponent), float(self.coefficient), self.power)


@dataclass
class SoEcp:
    symbol: str
    ncore: int
    lmax: int                    # L: the l of the local (M) term
    lso: int                     # highest l with a spin-orbit term
    local: List[Term]
    semilocal: List[List[Term]]  # index l = 0 .. L-1
    spin_orbit: Dict[int, List[Term]] = field(default_factory=dict)  # l = 1 .. lso


# --- MOLPRO format ---------------------------------------------------------

def parse_molpro_ecp(text: str) -> SoEcp:
    """Read a MOLPRO ``ECP,El,ncore,lmax,lso;`` definition with its SO blocks."""
    tokens = [t.strip() for t in text.replace("\n", " ").split(";")]
    tokens = [t for t in tokens if t]
    if not tokens or not tokens[0].upper().startswith("ECP,"):
        raise SocError("not a MOLPRO ECP definition (no 'ECP,El,ncore,lmax,lso;' header)")
    head = [h.strip() for h in tokens[0].split(",")]
    if len(head) != 5:
        raise SocError(f"ECP header needs element, ncore, lmax and lso: {tokens[0]!r}")
    symbol, ncore, lmax, lso = head[1], int(head[2]), int(head[3]), int(head[4])

    blocks, pos = [], 1
    for _ in range(lmax + 1 + lso):
        if pos >= len(tokens):
            raise SocError(f"{symbol}: expected {lmax + 1 + lso} ECP blocks, found {len(blocks)}")
        count = int(tokens[pos])
        terms = []
        for tok in tokens[pos + 1: pos + 1 + count]:
            parts = [p.strip() for p in tok.split(",")]
            if len(parts) != 3:
                raise SocError(f"{symbol}: ECP term must be 'n,exponent,coefficient': {tok!r}")
            terms.append(Term(parts[1], parts[2], int(parts[0]) - 2))
        if len(terms) != count:
            raise SocError(f"{symbol}: block announces {count} terms, has {len(terms)}")
        blocks.append(terms)
        pos += 1 + count
    if pos != len(tokens):
        raise SocError(f"{symbol}: unexpected text after the ECP blocks: {tokens[pos:]}")

    return SoEcp(symbol, ncore, lmax, lso,
                 local=blocks[0],
                 semilocal=blocks[1:lmax + 1],
                 spin_orbit={l: blocks[lmax + l] for l in range(1, lso + 1)})


def load_so_ecp(z: int) -> SoEcp:
    path = os.path.join(SOPSEUD_DIR, f"{200 + z}.mol")
    if not os.path.exists(path):
        raise SocError(f"no spin-orbit ECP for Z={z} (no {200 + z}.mol in sopseud)")
    with open(path) as fh:
        return parse_molpro_ecp(fh.read())


def available_so_ecps() -> List[int]:
    """Atomic numbers with a spin-orbit ECP in sopseud."""
    return sorted(int(n[:-4]) - 200 for n in os.listdir(SOPSEUD_DIR)
                  if re.fullmatch(r"2\d\d\.mol", n))


# --- CRYSTAL INPSOC records -------------------------------------------------

def _is_zero(value: str) -> bool:
    return float(value) == 0.0


def _merge(scalar: List[Term], so: List[Term]) -> List[Tuple[str, str, str, int]]:
    """Rows (exponent, C_AREP, C_SOREP, NKL) for one l.

    A SO term goes on the scalar row with the same exponent and power; one with
    no such row gets a row of its own with a zero AREP coefficient."""
    rows = [[t.exponent, t.coefficient, "0.000000", t.power] for t in scalar]
    used = [False] * len(rows)
    for s in so:
        for i, t in enumerate(scalar):
            if not used[i] and float(t.exponent) == float(s.exponent) and t.power == s.power:
                rows[i][2] = s.coefficient
                used[i] = True
                break
        else:
            rows.append([s.exponent, "0.000000", s.coefficient, s.power])
    return [tuple(r) for r in rows]


def inpsoc_records(ecp: SoEcp, z: int, soscale: float = 1.0) -> List[str]:
    """The INPSOC lines for ``ecp`` on element ``z`` (manual pp. 84-87)."""
    if ecp.lmax > MAX_LOCAL_L:
        raise SocError(f"{ecp.symbol}: L = {ecp.lmax}, but INPSOC takes semi-local terms only up to l = 4")
    if ecp.lso >= ecp.lmax or ecp.lso > 4:
        raise SocError(f"{ecp.symbol}: spin-orbit terms up to l = {ecp.lso} do not fit L = {ecp.lmax}")
    if z - ecp.ncore <= 0:
        raise SocError(f"{ecp.symbol}: {ecp.ncore} core electrons leave no effective charge for Z={z}")

    local = [(t.exponent, t.coefficient, "0.000000", t.power)
             for t in ecp.local if not _is_zero(t.coefficient)]
    per_l = []
    for l in range(5):
        scalar = ecp.semilocal[l] if l < ecp.lmax else []
        per_l.append(_merge(scalar, ecp.spin_orbit.get(l, [])))

    lines = ["INPSOC", f"INTERNAL {soscale}",
             f"{z - ecp.ncore}. {len(local)} " + " ".join(str(len(r)) for r in per_l)]
    for exp, carep, csorep, nkl in local + [row for rows in per_l for row in rows]:
        lines.append(f"{exp} {carep} {csorep} {nkl}")
    return lines


def parse_inpsoc(lines: List[str]) -> Tuple[float, List[tuple], List[List[tuple]]]:
    """Read INPSOC records back: (ZNUC, local rows, rows per l = 0..4).

    Each row is (exponent, C_AREP, C_SOREP, NKL) as numbers."""
    if lines[0].strip() != "INPSOC":
        raise SocError("INPSOC records must start with INPSOC")
    name, _scale = lines[1].split()
    if name != "INTERNAL":
        raise SocError(f"only the INTERNAL convention is read back, not {name}")
    head = lines[2].split()
    znuc, counts = float(head[0]), [int(c) for c in head[1:]]
    rows = [(float(a), float(b), float(c), int(d))
            for a, b, c, d in (ln.split() for ln in lines[3:3 + sum(counts)])]
    if len(rows) != sum(counts):
        raise SocError("fewer INPSOC rows than the header announces")
    local, pos, per_l = rows[:counts[0]], counts[0], []
    for c in counts[1:]:
        per_l.append(rows[pos:pos + c])
        pos += c
    return znuc, local, per_l


# --- CRYSTAL scalar INPUT ECPs (the stuttgart/ library) ---------------------

def parse_input_ecp(lines: List[str]) -> Tuple[float, List[tuple], List[List[tuple]]]:
    """Read an INPUT ECP (ZNUC, local rows, rows per l = 0..4) - rows are
    (exponent, coefficient, NKL) - from the lines that start with INPUT."""
    if lines[0].strip() != "INPUT":
        raise SocError("scalar ECP records must start with INPUT")
    head = lines[1].split()
    znuc, counts = float(head[0]), [int(c) for c in head[1:]]
    counts += [0] * (6 - len(counts))
    rows = [(float(a), float(b), int(c))
            for a, b, c in (ln.split()[:3] for ln in lines[2:2 + sum(counts)])]
    local, pos, per_l = rows[:counts[0]], counts[0], []
    for c in counts[1:]:
        per_l.append(rows[pos:pos + c])
        pos += c
    return znuc, local, per_l


_L_LETTERS = "spdfgh"


def duplicated_terms(ecp: SoEcp) -> List[str]:
    """The blocks of ``ecp`` that list one term (exponent, coefficient and
    power) more than once, as "d (scalar)" or "d (spin-orbit)". A published
    potential has no such repeat: entered as it stands, the term would count
    twice. Only sopseud/219.mol (K) has them, in both of its d blocks."""
    found = []
    blocks = [(ecp.lmax, "scalar", ecp.local)]
    blocks += [(l, "scalar", terms) for l, terms in enumerate(ecp.semilocal)]
    blocks += [(l, "spin-orbit", terms) for l, terms in sorted(ecp.spin_orbit.items())]
    for l, kind, terms in blocks:
        values = [t.values() for t in terms]
        if len(set(values)) != len(values):
            found.append(f"{_L_LETTERS[l] if l < len(_L_LETTERS) else f'l={l}'} ({kind})")
    return found


def scalar_part(ecp: SoEcp, z: int) -> Tuple[float, List[tuple], List[List[tuple]]]:
    """The SO-ECP's scalar (AREP) part in the form parse_input_ecp returns."""
    local = [t.values() for t in ecp.local if not _is_zero(t.coefficient)]
    per_l = [[t.values() for t in ecp.semilocal[l]] if l < ecp.lmax else [] for l in range(5)]
    return float(z - ecp.ncore), local, per_l


def _library_record(family: str, key) -> Optional[str]:
    """Record ``key`` of a bundled basis family (basis_sets/<family>.lib),
    as stored, or None when the family has none."""
    import basis_library
    lib = basis_library.load_family(family)
    return lib.record(key) if key in lib else None


def read_stuttgart(z: int) -> Optional[Tuple[List[str], List[str]]]:
    """(ECP lines from INPUT on, valence shell lines) of stuttgart/<200+Z>
    (the stuttgart family of the basis library), or None if there is no such
    record or it holds no INPUT ECP. Raises
    SocError for a file whose valence basis CRYSTAL could not read as it
    stands (see _stuttgart_shell_problem)."""
    text = _library_record("stuttgart", 200 + z)
    if text is None:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or lines[1].strip() != "INPUT":
        return None
    counts = [int(c) for c in lines[2].split()[1:]]
    end = 3 + sum(counts)
    problem = basis_block_problem("\n".join(lines))
    if problem:
        raise SocError(f"stuttgart/{200 + z} (Z={z}) cannot be used: {problem}; "
                       f"the file needs repairing from the published basis set")
    return lines[1:end], lines[end:]


def _stuttgart_shell_problem(header: str, shells: List[str]) -> Optional[str]:
    """What stops CRYSTAL reading these valence shells, or None. The first
    line gives NAT and NSHL, and NSHL shell records follow the ECP, each
    'ITYB LAT NG CHE SCAL' with a numeric formal charge CHE and, for ITYB 0,
    NG primitive lines (manual pp. 25-27). Some library files hold a
    placeholder where NSHL belongs, "X" for CHE, or fewer shells than NSHL;
    which shells are missing or how the electrons are shared among them is
    not in the file, so it is refused rather than guessed. The check is
    basis_file_check.shell_problem, which every basis reader now uses."""
    return shell_problem(header, shells)


def same_scalar_potential(ecp: SoEcp, z: int, input_lines: List[str]) -> bool:
    """Whether the INPUT ECP ``input_lines`` is exactly this SO-ECP's scalar part."""
    return parse_input_ecp(input_lines) == scalar_part(ecp, z)


def soc_ecp_records(z: int, input_lines: List[str], soscale: float = 1.0) -> List[str]:
    """INPSOC records to replace the INPUT ECP ``input_lines`` of element ``z``.

    Only an ECP whose scalar part is, number for number, the spin-orbit ECP's
    own is replaced: the valence basis beside it was published with that
    potential, so it stays as it is. Anything else is refused rather than
    paired with a basis built for another potential."""
    ecp = load_so_ecp(z)
    if ecp.lso == 0:
        raise SocError(f"{ecp.symbol}: {200 + z}.mol has no spin-orbit terms (lso = 0)")
    repeated = duplicated_terms(ecp)
    if repeated:
        raise SocError(
            f"{ecp.symbol}: {200 + z}.mol lists the same terms twice in its "
            f"{' and '.join(repeated)} block{'s' if len(repeated) > 1 else ''}; entered "
            f"as it stands each would count twice, so it is not used until it is checked "
            f"against the published potential")
    if not same_scalar_potential(ecp, z, input_lines):
        raise SocError(
            f"{ecp.symbol}: the deck's ECP is not the scalar part of the spin-orbit ECP "
            f"{200 + z}.mol (different core or parameters), so its valence basis does not "
            f"belong to that potential; use a basis set built on it")
    return inpsoc_records(ecp, z, soscale)


# --- CRYSTAL's internal STUTSC library -------------------------------------
#
# Instead of the inline INPSOC records, an atom's ECP can be CRYSTAL's own
# Stuttgart spin-orbit library, the one-line keyword STUTSC (manual p. 84).
# STUTSC_CORES lists the elements for which CRYSTAL's STUTSC was measured to be
# the same potential as MACE's INPSOC conversion of sopseud/<200+Z>.mol, with
# the core size of both. Measured on HPCC, CRYSTAL23 module of the intel-2023a toolchain:
#   - Te, Au, Pb, Bi (FACTS): bulk/slab SOC runs, the same printed
#     PSEUDOPOTENTIAL INFORMATION and energies (Bi2, Au identical; PbTe 2.8e-7 Ha).
#   - the others (round 6, H1): one atom as MOLECULE, INPSOC and STUTSC decks
#     differing only in that record, MAXCYCLE 2. The PSEUDOPOTENTIAL INFORMATION
#     tables print the same nuclear charge and the same numbers, and the
#     cycle-1 energies are identical, except where CRYSTAL's own data has one
#     digit less (Pb 9.9406230 vs 9.9406210, In 9.1034300 vs 9.1034330;
#     In's energy still identical, Pb's 2.8e-7 Ha apart).
# Not in the table, also from H1 and the HPCC follow-up runs: Hf (see
# STUTSC_NOT_THE_PUBLISHED_ECP below); La-Lu, where CRYSTAL stops (MPI_Abort) right after printing
# the PSEUDOPOTENTIAL INFORMATION header for STUTSC; Tc and Yb, which MACE gives
# no spin-orbit ECP. For an element outside the table the default INPSOC is
# written.
STUTSC_CORES = {
    37: 28, 38: 28, 39: 28, 40: 28, 41: 28, 42: 28,             # Rb-Mo, measured (H1)
    44: 28, 45: 28, 46: 28, 47: 28, 48: 28, 49: 28, 50: 28,     # Ru-Sn, measured (H1)
    51: 28, 53: 28,                                             # Sb, I, measured (H1)
    52: 28,                                                     # Te, measured (FACTS, H1)
    55: 46, 56: 46,                                             # Cs, Ba, measured (H1)
    73: 60, 74: 60, 75: 60, 76: 60, 77: 60, 78: 60,             # Ta-Pt, measured (H1)
    80: 60, 81: 60,                                             # Hg, Tl, measured (H1)
    79: 60, 82: 60, 83: 60,                                     # Au, Pb, Bi, measured (FACTS, H1)
}
# Elements whose STUTSC is NOT the published potential, so MACE writes its
# INPSOC instead, even for the official pob-TZVP-rev2-SOC set (soc_basis),
# whose record names STUTSC. Hf: MACE's INPUT/INPSOC (sopseud/272.mol) is the
# published ECP60MDF (D. Figgen, K. A. Peterson, M. Dolg, H. Stoll, J. Chem.
# Phys. 130 (2009) 164108); CRYSTAL's internal STUTSC gives the r^2 term of
# the local potential the exponent of its first term, 10.2488560, where
# ECP60MDF has 10.7291270. Measured on HPCC (CRYSTAL/23-intel-2023a, HfO2,
# INPUT, INPSOC and STUTSC decks otherwise identical): STUTSC is 9.67e-5 Ha
# above INPUT/INPSOC.
STUTSC_NOT_THE_PUBLISHED_ECP = {
    72: ("CRYSTAL's internal STUTSC for Hf is not the published ECP60MDF (Figgen et al. "
         "2009) that MACE's INPSOC is: it gives the r^2 term of the local potential the "
         "exponent of the first term, 10.2488560 instead of 10.7291270, which costs "
         "+9.67e-5 Ha (measured on HPCC, HfO2)"),
}
ECP_FORMS = ("inpsoc", "stutsc")
# Library families whose ECP atoms soc_deck can give a spin-orbit part.
LIBRARY_FAMILIES = ("full.basis.triplezeta", "full.basis.doublezeta",
                    "full.basis.triplezeta.rev2022", "stuttgart")


def _checked_ecp_form(value) -> Optional[str]:
    """A "soc_ecp" value, "inpsoc" or "stutsc", or None when none was given."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in ECP_FORMS:
        raise SocError(f'soc_ecp must be "inpsoc" (MACE\'s INPSOC records, the default) '
                       f'or "stutsc" (CRYSTAL\'s STUTSC library, manual p. 84), not {value!r}')
    return value.strip().lower()


_ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
             "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
             "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
             "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es").split()


def _symbol(z: int) -> str:
    if 0 < z < len(_ELEMENTS):
        return _ELEMENTS[z]
    return f"Z={z}"


def _atom_z(number: int) -> int:
    """Z of a conventional atomic number: number modulo 100 (manual p. 25), so
    282 and 382 are both Pb with an ECP, 124 an all-electron Cr label."""
    return number % 100 if number > 100 else number


def _is_ecp_label(number: int) -> bool:
    """Whether a conventional atomic number asks for an ECP (NAT > 200)."""
    return 200 < number < 1000


# A spin-orbit splitting of a 4p shell (Ga-Br) is about 0.3-0.5 eV, so from
# Z = 31 on an all-electron atom without its SO term is warned about by name.
AE_WARN_Z = 31


# Ga-Se: the stuttgart library's ECP10 records (231-234) carry the SO term.
# Measured on HPCC (InAs, PBE0, CRYSTAL/23-intel-2023a): the split-off gap
# Delta0 at Gamma was about 0.07 eV with the all-electron As of
# full.basis.triplezeta and about 0.35 eV with the stuttgart ECP10 As + INPSOC
# (experiment 0.39 eV), while the total energy moved by only about 0.6 mHa.
AE_ECP10_Z = range(31, 35)
AE_MEASURED = (" Measured on HPCC (InAs): with all-electron As the valence-band spin-orbit "
               "splitting Delta0 was about 0.07 eV, with the stuttgart ECP10 As and INPSOC "
               "about 0.35 eV (experiment 0.39 eV) - about 80% of the splitting is lost - "
               "while the total energy moved by only about 0.6 mHa, so the energy does not "
               "show it. When the band structure matters, MACE strongly recommends the "
               "ECP10 records for Ga-Se")


def _stuttgart_matches(z: int) -> bool:
    """Whether the stuttgart family's ECP for ``z`` is the scalar part of
    sopseud/<200+Z>.mol (so a deck in that family gets its SO term)."""
    try:
        found = read_stuttgart(z)
        return bool(found) and same_scalar_potential(load_so_ecp(z), z, found[0])
    except (SocError, ValueError, KeyError, IndexError, OSError):
        return False


def _official_soc_elements():
    """Atomic numbers the official pob-TZVP-rev2-SOC set covers."""
    try:
        import basis_library
        return set(int(z) for z in basis_library.soc_basis_elements())
    except Exception:
        return set()


def _atom_name(number: int) -> str:
    """'Cl (atomic number 17)', or for a label that is not the bare atomic
    number (CRYSTAL's conventional atomic number, manual p. 26)
    'C (atom label 106)': these name an element, not an atom's position."""
    z = _atom_z(number)
    return (f"{_symbol(z)} (atomic number {z})" if number == z
            else f"{_symbol(z)} (atom label {number})")


def _all_electron_notes(numbers: List[int], log) -> None:
    """Name the all-electron atoms of a SOC deck: they get no spin-orbit term,
    since CRYSTAL puts SOC only in the ECP (manual p. 175). From Z = 31 on
    (AE_WARN_Z) it is a warning with what would give them one."""
    light, heavy = [], []
    for number in dict.fromkeys(numbers):
        z = _atom_z(number)
        (heavy if z >= AE_WARN_Z else light).append((number, z))
    if light:
        log(f"SOC deck: {', '.join(_atom_name(n) for n, z in light)} "
            f"{'is' if len(light) == 1 else 'are'} all-electron and keep{'s' if len(light) == 1 else ''} "
            f"the basis as written; there is no spin-orbit term on an all-electron atom "
            f"(CRYSTAL23 manual p. 175), which matters less the lighter the atom")
    for number, z in heavy:
        split = " (about 0.3-0.5 eV for its 4p shell)" if 31 <= z <= 36 else ""
        if z in _official_soc_elements():
            fix = (f"The official pob-TZVP-rev2-SOC set has {_symbol(z)}: give the atom an "
                   f"ECP in the parent (atom number {200 + z} with an ECP basis record, e.g. "
                   f"full.basis.triplezeta's), and \"soc_basis\" writes the SOC set for it")
        elif _stuttgart_matches(z):
            fix = (f"Record {200 + z} of MACE's stuttgart library is an ECP10 potential for "
                   f"{_symbol(z)} that does get its SO term (its provenance is not verified). "
                   f"No MACE writer picks it - below Rb they use the all-electron record - so "
                   f"enter it by hand: atom number {200 + z} in the geometry and that record "
                   f"as its basis, then convert the deck again")
        else:
            fix = (f"MACE has no ECP for {_symbol(z)} that carries a spin-orbit term (not in "
                   f"stuttgart either), and the official pob-TZVP-rev2-SOC set starts at Rb")
        log(f"Warning: SOC deck: {_atom_name(number)} is all-electron here, so it gets "
            f"no spin-orbit term - CRYSTAL puts SOC only in the ECP (CRYSTAL23 manual "
            f"p. 175) - and its spin-orbit splitting{split} is missing from this deck. "
            f"{fix}.{AE_MEASURED if z in AE_ECP10_Z else ''}")


# La-Lu and the actinides: an open-shell 4f (5f) 2c-SCF did not converge on
# HPCC. EuO (CRYSTAL/23-intel-2023a, PBE0, INPSOC) from a cold start stopped
# with "FERMI ENERGY NOT IN INTERVAL"; with SMEAR 0.005 or 0.01 the energy
# swung by about +-8 Ha from cycle to cycle until MAXCYCLE (80). The route the
# manual offers for such a start, a 1c spin-polarized parent read with
# GUESSPNOSO (p. 173), was not run.
F_ELEMENTS = set(range(57, 72)) | set(range(89, 104))


def _f_element_warning(numbers: List[int], guess: Optional[str], log) -> None:
    """Warn once when a SOC deck has an f element (La-Lu, Ac-Lr), whose
    open-shell 2c-SCF did not converge from a cold start (EuO, HPCC)."""
    found = list(dict.fromkeys(_symbol(_atom_z(n)) for n in numbers
                               if _atom_z(n) in F_ELEMENTS))
    if not found:
        return
    route = ("This deck restarts with GUESSPNOSO, that route; it is untested"
             if guess == "guesspnoso" else
             "A route that may work, untested: a one-component SPIN (unrestricted) parent "
             "run without symmetry, then this deck from it with GUESSPNOSO "
             "(\"soc_guess\": \"guesspnoso\" in an opt2d12 template, manual p. 173)")
    log(f"Warning: SOC deck: {', '.join(found)} {'is an f element' if len(found) == 1 else 'are f elements'}; "
        f"if the f shell is open, the 2c-SCF may not converge. Measured on HPCC: EuO "
        f"(open-shell 4f) from a cold start stopped with \"FERMI ENERGY NOT IN INTERVAL\", "
        f"and with SMEAR its energy swung by about +-8 Ha per cycle until it ran out of "
        f"cycles. {route}.")


def _stutsc_elements() -> str:
    return ", ".join(_symbol(z) for z in sorted(STUTSC_CORES))


def _no_scale_for_stutsc(soscale: float, what: str) -> None:
    if float(soscale) != 1.0:
        raise SocError(f"{what}: CRYSTAL's STUTSC has no scale factor for its spin-orbit "
                       f"terms (INPSOC's SOSCALE, manual p. 84), so SOSCALE {soscale} "
                       f"would be dropped; use the INPSOC records (\"soc_ecp\": \"inpsoc\")")


def stutsc_records(z: int, soscale: float = 1.0) -> List[str]:
    """["STUTSC"] for element ``z``, whose deck ECP has already been found to
    be the scalar part of sopseud/<200+Z>.mol (soc_ecp_records), when
    CRYSTAL's STUTSC was measured to be that same potential (STUTSC_CORES)."""
    ecp = load_so_ecp(z)
    if z in STUTSC_NOT_THE_PUBLISHED_ECP:
        raise SocError(f"{ecp.symbol} (Z={z}): {STUTSC_NOT_THE_PUBLISHED_ECP[z]}; leave out "
                       f"\"soc_ecp\": \"stutsc\" for the default INPSOC, the published potential")
    if z not in STUTSC_CORES or ecp.ncore != STUTSC_CORES[z]:
        raise SocError(
            f"{ecp.symbol} (Z={z}): CRYSTAL's STUTSC spin-orbit ECP was compared with "
            f"MACE's INPSOC conversion only for {_stutsc_elements()} (the same potential, "
            f"measured on HPCC); for {ecp.symbol} the default INPSOC (\"soc_ecp\": "
            f"\"inpsoc\") works")
    _no_scale_for_stutsc(soscale, f"{ecp.symbol} (Z={z})")
    return ["STUTSC"]


def _tokens(lines: List[str]) -> List[List[str]]:
    return [ln.split() for ln in lines if ln.strip()]


def _library_shells(z: int) -> List[List[List[str]]]:
    """The valence shells (as tokens) of MACE's library files for 200+Z whose
    INPUT ECP is the scalar part of sopseud/<200+Z>.mol."""
    found = []
    ecp = load_so_ecp(z)
    for family in LIBRARY_FAMILIES:
        text = _library_record(family, 200 + z)
        if text is None:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        try:
            if lines[1].strip() != "INPUT":
                continue
            end = 3 + sum(int(c) for c in lines[2].split()[1:])
            if same_scalar_potential(ecp, z, lines[1:end]):
                found.append(_tokens(lines[end:]))
        except (IndexError, ValueError):
            continue
    return found


# --- the official pob-TZVP-rev2-SOC basis sets --------------------------------
#
# X_pob_TZVP_rev2_SOC of the CRYSTAL basis set library (crystal.unito.it,
# "To be used with CRYSTAL23"): CRYSTAL's STUTSC spin-orbit ECP with the valence
# basis published for it (J. Laun, T. Bredow, J. Comput. Chem. 43 (2022) 839,
# doi 10.1002/jcc.26839; J. K. Desmarais, J. P. Flament, A. Erba, Phys. Rev. B
# 101 (2020) 235142). Rb-I except Tc, Cs, Ba and Hf-Bi; the site has no such set
# for Tc, La-Lu, the actinides or Po onwards. The records, with their source
# and retrieval date, are in basis_sets/pob-tzvp-rev2-soc.lib (written by
# basis_sets/build_library.py, read with basis_library.soc_basis_record). Their
# valence shells are numerically those of pob-TZVP-rev2 (checked when the
# library is built); only the ECP keyword differs.
SOC_BASIS_FAMILIES = ("pob-tzvp-rev2-soc",)


def _checked_soc_basis(value) -> Optional[str]:
    """A "soc_basis" value, or None when none was given."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in SOC_BASIS_FAMILIES:
        raise SocError(f"soc_basis must be one of {', '.join(SOC_BASIS_FAMILIES)}, "
                       f"not {value!r}")
    return value.strip().lower()


_SOC_BASIS_CACHE: Dict[str, Dict[int, List[str]]] = {}


def load_soc_basis(family: str) -> Dict[int, List[str]]:
    """{Z: record lines} of an official SOC basis family, read through
    basis_library (basis_library.soc_basis_record). Every record must pass
    basis_block_problem and have NAT = 200 + Z with STUTSC on its second
    line. Raises SocError otherwise."""
    import basis_library
    family = _checked_soc_basis(family)
    if family in _SOC_BASIS_CACHE:
        return _SOC_BASIS_CACHE[family]
    name = family + ".lib"
    try:
        elements = basis_library.soc_basis_elements()
    except (OSError, KeyError, ValueError) as exc:
        raise SocError(f"no data for soc_basis {family} ({exc})")
    records: Dict[int, List[str]] = {}
    for z in elements:
        text = basis_library.soc_basis_record(z)
        lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
        problem = basis_block_problem(text)
        if problem or not lines or lines[0].split()[0] != str(200 + z) \
                or len(lines) < 2 or lines[1].strip().upper() != "STUTSC":
            raise SocError(f"{name}: record {200 + z} is not an ECP block "
                           f"with STUTSC that CRYSTAL can read ({problem or lines[:2]})")
        records[z] = lines
    if not records:
        raise SocError(f"{name}: no records")
    _SOC_BASIS_CACHE[family] = records
    return records


def official_soc_block(z: int) -> Optional[List[str]]:
    """The official pob-TZVP-rev2-SOC block for element ``z`` (NAT 200+Z,
    STUTSC, shells), or None when MACE has none for it."""
    try:
        return load_soc_basis(SOC_BASIS_FAMILIES[0]).get(z)
    except SocError:
        return None


def _soc_basis_atoms(lines: List[str], geom_end: int, end99: int, family: str,
                     ecp_form: Optional[str], soscale: float, log) -> List[str]:
    """The basis input (lines geom_end+1 .. end99) with every ECP atom's
    block (200 < NAT < 300) replaced whole by the ``family`` block for its
    element; all-electron atoms keep theirs. Atom numbers never change."""
    if ecp_form == "inpsoc":
        raise SocError(f"soc_basis {family} is written with CRYSTAL's STUTSC ECP (\"To be "
                       f"used with CRYSTAL23\"), so it cannot be combined with \"soc_ecp\": "
                       f"\"inpsoc\"; leave out soc_ecp")
    _no_scale_for_stutsc(soscale, f"soc_basis {family}")
    sets = load_soc_basis(family)
    out, i, missing, replaced, all_electron = [], geom_end + 1, [], [], []
    while i < end99:
        try:
            number = int(lines[i].split()[0])
            _s, _e, nxt = _basis_block_span(lines, i)
        except (ValueError, IndexError):
            raise SocError(f"could not read the basis-set input at line {i + 1}: {lines[i]!r}")
        if _is_ecp_label(number):
            z = _atom_z(number)
            if z not in sets:
                missing.append(f"{_symbol(z)} (Z={z})")
            elif z in STUTSC_NOT_THE_PUBLISHED_ECP:
                # the official valence shells on MACE's INPSOC, the published potential
                out += ([" ".join([str(number)] + sets[z][0].split()[1:])]
                        + inpsoc_records(load_so_ecp(z), z, soscale) + sets[z][2:])
                log(f"SOC deck: {_symbol(z)} gets the official {family} valence shells "
                    f"with MACE's INPSOC in place of the record's STUTSC: "
                    f"{STUTSC_NOT_THE_PUBLISHED_ECP[z]}")
            else:
                # the deck's own label (382 for a second Pb, say) stays
                out += [" ".join([str(number)] + sets[z][0].split()[1:])] + sets[z][1:]
                replaced.append(_symbol(z))
        else:
            out += lines[i:nxt]
            all_electron.append(number)
            if _atom_z(number) == 43:
                log("SOC deck: MACE's Tc is all-electron and stays so: the set has no Tc")
        i = nxt
    # the same notes as the default path, with the warning from Ga (Z = 31) on
    if all_electron:
        _all_electron_notes(all_electron, log)
    if missing:
        raise SocError(f"soc_basis {family} has no basis set for "
                       f"{', '.join(dict.fromkeys(missing))} (the official set covers "
                       f"Rb-I except Tc, Cs, Ba and Hf-Bi; not La-Lu, the actinides or "
                       f"Po onwards)")
    if replaced:
        log(f"SOC deck: {', '.join(dict.fromkeys(replaced))} get the official {family} "
            f"basis (CRYSTAL's STUTSC ECP with its own valence basis), replacing the "
            f"deck's block")
        log(f"Note: with soc_basis the valence basis differs from a scalar parent's, so "
            f"its energy is not the reference for the SOC contribution: that needs a "
            f"two-component run without SOC in this same basis (manual p. 174), which "
            f"MACE does not write - copy this deck and delete the SOC line from its "
            f"TWOCOMPON block")
    return out


def _kept_stutsc(z: int, block: List[str], soscale: float, ecp_form: Optional[str],
                 log) -> List[str]:
    """The STUTSC record of a deck derived from a SOC deck (``block`` is the
    atom's whole basis block), kept when MACE would have written that block
    itself: its shells are MACE's library basis for an element whose STUTSC
    core was measured to match (STUTSC_CORES), or the block is the official
    pob-TZVP-rev2-SOC one. Anything else is refused, as for INPSOC: MACE does
    not take over a potential it did not write."""
    what = f"{_symbol(z)} (Z={z})"
    official = official_soc_block(z)
    if z in STUTSC_NOT_THE_PUBLISHED_ECP and official is not None \
            and _tokens(block[1:]) == _tokens(official[1:]) \
            and block[0].split()[1:] == official[0].split()[1:]:
        if ecp_form == "stutsc":
            stutsc_records(z, soscale)          # raises, saying why
        log(f"SOC deck: {_symbol(z)}'s STUTSC becomes MACE's INPSOC, on the same official "
            f"valence shells: {STUTSC_NOT_THE_PUBLISHED_ECP[z]}")
        return inpsoc_records(load_so_ecp(z), z, soscale)
    if ecp_form == "inpsoc":
        raise SocError(f"{what}: the deck has CRYSTAL's STUTSC ECP, and \"soc_ecp\": "
                       f"\"inpsoc\" would change the potential its valence basis was "
                       f"kept with; leave out soc_ecp to keep STUTSC")
    _no_scale_for_stutsc(soscale, what)
    if official is not None and _tokens(block[1:]) == _tokens(official[1:]) \
            and block[0].split()[1:] == official[0].split()[1:]:
        log(f"SOC deck: {_symbol(z)} keeps CRYSTAL's STUTSC spin-orbit ECP it already has "
            f"(the official pob-TZVP-rev2-SOC block)")
        return [block[1]]
    try:
        library = (z in STUTSC_CORES and load_so_ecp(z).ncore == STUTSC_CORES[z]
                   and _tokens(block[2:]) in _library_shells(z))
    except SocError:
        library = False
    if not library:
        raise SocError(
            f"{what}: the deck's STUTSC ECP sits on a valence basis that is neither MACE's "
            f"library basis for an element whose STUTSC core was measured to match "
            f"({_stutsc_elements()}) nor the official pob-TZVP-rev2-SOC block; MACE does "
            f"not take over a potential it did not write")
    log(f"SOC deck: {_symbol(z)} keeps CRYSTAL's STUTSC spin-orbit ECP it already has")
    return [block[1]]


# Keywords the 2c-SCF accepts, from manual chapter 6 (pp. 166-169): anything
# not listed there "is not supported for calculations in the two-component
# spinor basis" (p. 166).
TWOC_FUNCTIONALS = {
    "SVWN", "BLYP", "PBEXC", "PBESOLXC", "SOGGAXC",
    "B3PW", "B3LYP", "PBE0", "PBESOL0", "B1WC", "WC1LYP", "B97H", "PBE0-13",
    "MPW1PW91", "MPW1K",
}
TWOC_EXCHANGE = {"LDA", "VBH", "BECKE", "PBE", "PBESOL", "MPW91", "PWGGA", "SOGGA", "WCGGA"}
TWOC_CORRELATION = {"PZ", "VBH", "VWN", "LYP", "P86", "PBE", "PBESOL", "PWGGA", "PWLSD", "WL"}
# "<functional>-D3" keywords (manual sec. 5.1, p. 150) whose functional the
# 2c-SCF accepts (ch. 6). On HPCC, PBE0-D3 and B3LYP-D3 2c runs printed
# "DFT-D3(BJ) WITH AUTOMATIC PARAMETER SETUP" and a D3 energy. PW1PW is the
# D3 keyword of mPW1PW91; PBE-D3 is PBE exchange and correlation.
TWOC_D3_FUNCTIONALS = {"BLYP-D3", "PBE-D3", "B3LYP-D3", "PBE0-D3", "PW1PW-D3"}
TWOC_DFT_KEYWORDS = {
    # SPIN is not in ch. 6, but a 2c run accepts it (HPCC), and without
    # effect: the 2c-SCF treats spin itself.
    "SPIN",
    "HYBRID", "COLLINEAR", "NONCOLC", "NONCOLSF", "TOLM",
    "ANGULAR", "RADIAL", "BECKE", "SAVIN", "OLDGRID", "LGRID", "XLGRID", "XXLGRID",
    "RADSAFE", "TOLLDENS", "TOLLGRID", "BATCHPNT", "CHUNKS", "DISTGRID", "LIMBEK",
    "RADIUS", "FCHARGE",
}
TWOC_SCF_KEYWORDS = {
    "BIPOLAR", "BIPOSIZE", "EXCHSIZE", "EXCHPERM", "GUESSPAT", "ILASIZE", "INTGPACK",
    "MADELIND", "NOBIPCOU", "NOBIPEXCH", "NOBIPOLA", "POLEORDR", "TOLINTEG", "TOLPSEUD",
    "SCFDIR", "EIGS", "FIXINDEX", "FDAOSYM", "FDAOCCUP", "FMIXING", "LEVSHIFT",
    "MAXCYCLE", "SMEAR", "TOLDEE", "MAXNEIGHB", "NEIGHBOR", "QVRSGDIM", "SETINF",
    "TESTPDIM", "TEST", "TESTRUN", "POSTSCF", "PPAN",
    # SHRINK is not in the chapter 6 list, but a periodic SCF cannot run
    # without it (and the 2c reference outputs report their SHRINK mesh).
    "SHRINK",
}
# Records soc_deck leaves out, with how many argument lines follow each. The
# manual says TWOCOMPON deactivates DIIS (p. 170), but a DIIS record left in
# the deck stops the stock build: "ERROR **** DIIS **** DIIS NOT COMPATIBLE
# WITH 2-COMP SCF" (measured on HPCC).
_STRIPPED = {"DIIS": 0, "HISTDIIS": 1}
# Keywords that open a block in block 3 closed by END (or END<name>).
_BLOCK3_OPENERS = {"DFT", "SCDFT", "SDFT"}
# Present anywhere, these ask for something 2c cannot do (p. 166).
UNSUPPORTED_ANYWHERE = {
    "OPTGEOM": "geometry optimization", "FREQCALC": "frequency calculation",
    "ELASTCON": "elastic constants", "ELAPIEZO": "elastic/piezoelectric tensors",
    "PIEZOCON": "piezoelectric tensor", "EOS": "equation of state",
    "CPHF": "response properties", "CPKS": "response properties",
}

# What CRYSTAL itself does with some of these, measured on HPCC
# (CRYSTAL/23-intel-2023a), added to MACE's refusal.
MEASURED_REFUSALS = {
    # CRYSTAL accepts ANDERSON in a TWOCOMPON deck without a word: on PbTe
    # the SCF diverged and stopped with "FERMI ENERGY NOT IN INTERVAL", where
    # the same deck without ANDERSON converged in 9 cycles.
    "ANDERSON": ". CRYSTAL does not refuse it: measured on HPCC, a 2c PbTe deck with "
                "ANDERSON diverged and stopped with \"ERROR **** ZERO **** FERMI ENERGY NOT "
                "IN INTERVAL\", where the same deck without it converged in 9 cycles. "
                "Use DIIS instead (MACE leaves the DIIS record out of a SOC deck, since the "
                "2c-SCF has its own mixing)",
    # A TWOCOMPON deck with OPTGEOM stops at input; the .out ends without a
    # reason, which is printed only in fort.87.
    "OPTGEOM": ". CRYSTAL23 stops such a deck at input (measured on HPCC), with the "
               "reason only in fort.87: \"TWO-COMPONENT SCF IS NOT COMPATIBLE WITH "
               "OPTGEOM\"",
}


def unsupported_message(key: str) -> str:
    """The refusal soc_deck gives a deck with ``key`` (UNSUPPORTED_ANYWHERE),
    word for word; the GUI shows it on the calculations it grays out."""
    return (f"{UNSUPPORTED_ANYWHERE[key]} ({key}) is not available in a two-component SCF "
            f"(CRYSTAL23 manual p. 166); SOC decks are single points"
            + MEASURED_REFUSALS.get(key, ""))

def _is_number_line(line: str) -> bool:
    try:
        [float(x) for x in line.split()]
        return bool(line.split())
    except ValueError:
        return False


def _basis_block_span(lines: List[str], i: int) -> Tuple[int, int, int]:
    """(start, end) of the ECP records and the end of the atom basis block
    that starts at lines[i]; start == end when the atom has no ECP input."""
    number, nshell = (int(x) for x in lines[i].split()[:2])
    i += 1
    ecp_start = ecp_end = i
    if 200 < number < 1000:
        key = lines[i].strip().upper()
        if key == "INPUT":
            counts = [int(c) for c in lines[i + 1].split()[1:]]
            ecp_end = i + 2 + sum(counts)
        elif key == "INPSOC":
            counts = [int(c) for c in lines[i + 2].split()[1:]]
            ecp_end = i + 3 + sum(counts)
        else:
            ecp_end = i + 1   # an internal ECP library keyword (HAYWSC, ...)
        i = ecp_end
    for _ in range(nshell):
        ityb, _lat, ng = (int(x) for x in lines[i].split()[:3])
        i += 1 + (ng if ityb == 0 else 0)
    return ecp_start, ecp_end, i


SINGLE_POINT_REMEDY = (" - make it a single point (\"calculation_type\": \"SP\", or "
                       "opt2d12 --calc-type SP)")
BASISSET_REFUSAL = (
    "the deck uses an internal basis-set library (BASISSET); a SOC deck needs explicit "
    "per-element basis input for its spin-orbit ECPs - set \"basis_set_type\": \"EXTERNAL\" "
    "with \"basis_set\": \"full.basis.triplezeta\" (or \"stuttgart\"), MACE's bundled "
    "libraries with ECPs for the heavy elements")
FUNCTIONAL_REMEDY = (
    " - set the functional (\"dft_functional\" in a mace convert options file, "
    "\"functional\" in an opt2d12 template) to one of " + ", ".join(sorted(TWOC_FUNCTIONALS))
    + "; the -D3 forms of BLYP, PBE, B3LYP, PBE0 and PW1PW (\"PW1PW\" is mPW1PW91) "
    "work with dispersion on (\"use_dispersion\" / \"dispersion\": true), e.g. PBE0 or "
    "B3LYP with dispersion")


def _block3_start(upper: List[str]) -> Optional[int]:
    """Index of the first line of the deck's block 3 (Hamiltonian and SCF),
    or None when it cannot be found."""
    if "BASISSET" in upper:
        return upper.index("BASISSET") + 2
    if "99 0" in upper:
        end99 = upper.index("99 0")
        return next((j + 1 for j in range(end99 + 1, len(upper)) if upper[j] == "END"), None)
    return None


def _block3_problems(rest: List[str]) -> Tuple[List[str], Optional[int], List[str]]:
    """(the block-3 keywords a 2c-SCF does not take, index of SCFDIR, those
    of them in the DFT block)."""
    problems, inside, scfdir, dft = [], None, None, []
    for j, ln in enumerate(rest):
        key = ln.strip().upper()
        if not key or _is_number_line(key):
            continue
        word = key.split()[0]
        if inside is not None:
            if word in ("END", "ENDDFT", f"END{inside}"):
                inside = None
            elif inside in _BLOCK3_OPENERS:
                if word in ("EXCHANGE", "CORRELAT"):
                    allowed = TWOC_EXCHANGE if word == "EXCHANGE" else TWOC_CORRELATION
                    arg = rest[j + 1].strip().upper() if j + 1 < len(rest) else ""
                    if arg not in allowed:
                        problems.append(f"{word} {arg}")
                        dft.append(f"{word} {arg}")
                elif word in TWOC_EXCHANGE | TWOC_CORRELATION and rest[j - 1].strip().upper() in ("EXCHANGE", "CORRELAT"):
                    pass
                elif word not in TWOC_FUNCTIONALS | TWOC_D3_FUNCTIONALS | TWOC_DFT_KEYWORDS:
                    problems.append(word)
                    dft.append(word)
            continue
        if word in _BLOCK3_OPENERS:
            inside = word
        elif word == "SCFDIR" and scfdir is None:
            scfdir = j
        elif word == "END" and j == len(rest) - 1:
            pass
        elif word in _STRIPPED:
            pass
        elif word not in TWOC_SCF_KEYWORDS:
            problems.append(word)
    return problems, scfdir, dft


def _block3_refusals(problems: List[str], dft_words=()) -> List[str]:
    """The refusal, with its remedy, for each block-3 problem (dft_words:
    those found in the DFT block, which get the functional remedy)."""
    out = []
    problems = list(dict.fromkeys(problems))
    if "GUESSP" in problems:
        # The 2c restarts are GUESSPSO (from a 2c run) and GUESSP[NOSO] (from
        # an unrestricted 1c run without symmetry, SYMMREMO), both inside the
        # TWOCOMPON block (manual p. 173). They are written only on request
        # ("soc_guess"), from a parent whose .f9 is staged as fort.20.
        out.append(
            "GUESSP (a restart) is not a 2c-SCF keyword. A 2c run restarts with GUESSPSO "
            "(from an earlier 2c run) or GUESSPNOSO (from an unrestricted 1c run without "
            "symmetry) inside TWOCOMPON (CRYSTAL23 manual p. 173): set \"soc_guess\": "
            "\"guesspso\" or \"guesspnoso\" in an opt2d12 template, which checks the "
            "parent and stages its .f9 as fort.20")
    for misplaced in ("GUESSPSO", "GUESSPNOSO"):
        if misplaced in problems:
            out.append(
                f"{misplaced} is outside the TWOCOMPON block, and CRYSTAL stops on it there at "
                f"input (\"ERROR **** READM2 **** KEYWORD {misplaced} NOT ALLOWED\", measured on "
                f"HPCC). MACE writes the 2c restart inside TWOCOMPON itself: set \"soc_guess\": "
                f"\"{misplaced.lower()}\" in an opt2d12 template instead")
            break
    if "EIGENVAL" in problems:
        out.append(
            "PRINTOUT EIGENVAL is not written in a SOC deck: measured on HPCC, a 2c run with "
            "it printed garbage symmetry labels, split the eigenvalues into alpha and beta "
            "sets and then died with a segfault after the SCF. Leave out the PRINTOUT "
            "block (the band structure of a SOC run comes from properties, BAND)")
    if "ANDERSON" in problems:
        out.append(
            "ANDERSON is not a 2c-SCF keyword (CRYSTAL23 manual ch. 6 lists every keyword "
            "it accepts, p. 166)" + MEASURED_REFUSALS["ANDERSON"])
    handled = {"GUESSP", "GUESSPSO", "GUESSPNOSO", "EIGENVAL", "ANDERSON"}
    rest = [p for p in problems if p not in handled]
    if out and rest == ["PRINTOUT"] and "EIGENVAL" in problems:
        rest = []
    if rest:
        dft = [p for p in rest if p in dft_words]
        out.append(
            "not supported in a two-component SCF (CRYSTAL23 manual ch. 6 lists every "
            "keyword it accepts, p. 166): " + ", ".join(rest)
            + (FUNCTIONAL_REMEDY if dft else ""))
    return out


def _joined_refusals(refusals: List[str]) -> str:
    """One refusal as it is; several as a list, so all are fixed in one go."""
    if len(refusals) == 1:
        return refusals[0]
    return (f"a SOC deck needs {len(refusals)} changes:\n"
            + "\n".join(f"  - {r}" for r in refusals))


def soc_deck(deck: str, soscale: float = 1.0, log=print,
             fmixing: Optional[int] = None, smear: Optional[float] = None,
             ecp_form: Optional[str] = None, basis: Optional[str] = None,
             guess: Optional[str] = None) -> str:
    """Turn a finished single-point MACE deck into a 2c-SCF SOC deck.

    Every atom whose basis block carries an INPUT ECP (conventional atomic
    number 200+Z) has it replaced by the INPSOC form of the spin-orbit ECP for
    Z - only if its scalar part is that very potential - and keeps its valence
    shells. All-electron atoms keep their block, since the SOC operator lives in
    the ECP (p. 175).
    A TWOCOMPON block holding SOC goes into the SCF input, before SCFDIR (a
    placement the stock build accepts, HPCC); DIIS is left out, SHRINK gets
    IS = ISP, and FMIXING/MAXCYCLE are set for a 2c-SCF (see the helpers
    below). ``fmixing`` (the "soc_fmixing" setting) asks for that FMIXING
    whatever the deck had. ``smear`` (the "smear" setting, a width in
    hartree) writes SMEAR with that width after the SHRINK records, or gives
    the deck's own SMEAR that width; SMEAR is never added otherwise.
    ``ecp_form`` (the "soc_ecp" setting): "stutsc" writes CRYSTAL's STUTSC
    record in place of the INPSOC records, for the elements in STUTSC_CORES
    only and with SOSCALE 1; "inpsoc" or None writes INPSOC. A deck derived
    from a SOC deck keeps INPSOC records MACE wrote (converted to STUTSC with
    "stutsc") and a STUTSC record on a basis MACE would have written it with
    (_kept_stutsc). ``basis`` (the "soc_basis" setting, "pob-tzvp-rev2-soc")
    replaces every ECP atom's block whole by the official SOC basis set for
    its element (STUTSC and its own valence basis); all-electron atoms keep
    theirs (_soc_basis_atoms). ``guess`` (the "soc_guess" setting,
    "guesspso" or "guesspnoso") writes that restart record in the TWOCOMPON
    block (manual p. 173); whether the deck may restart from a given parent
    is for the caller to check (restart_mismatch), and the parent's .f9 has
    to be staged as <deck stem>.f20. Both get FMIXING SOC_RESTART_FMIXING
    unless ``fmixing`` is given. Notes and warnings go to
    ``log``. Raises SocError, and returns no deck, for anything the manual
    says the 2c-SCF does not support."""
    fmixing = _checked_fmixing(fmixing)
    guess = _checked_guess(guess)
    smear = checked_smear(smear)
    ecp_form = _checked_ecp_form(ecp_form)
    basis = _checked_soc_basis(basis)
    lines = deck.rstrip("\n").split("\n")
    upper = [ln.strip().upper() for ln in lines]

    if "TWOCOMPON" in upper:
        raise SocError("the deck already has a TWOCOMPON block")
    # Everything about the deck's settings that a SOC deck cannot have, all
    # at once and each with its remedy, so one run says every change to make.
    blockers = [unsupported_message(key) + SINGLE_POINT_REMEDY
                for key in UNSUPPORTED_ANYWHERE if key in upper]
    if "BASISSET" in upper:
        blockers.append(BASISSET_REFUSAL)
    start3 = _block3_start(upper)
    if start3 is not None:
        found, _, dft_words = _block3_problems(lines[start3:])
        blockers += _block3_refusals(found, dft_words)
    if blockers:
        raise SocError(_joined_refusals(blockers))

    try:
        end99 = upper.index("99 0")
    except ValueError:
        raise SocError("no explicit basis-set input (no '99 0' record) found in the deck")
    geom_end = upper.index("END")
    if geom_end > end99:
        raise SocError("could not find the end of the geometry input")

    # Basis section: replace ECP atoms' blocks.
    out = lines[:geom_end + 1]
    i, ecp_atoms, all_electron, atoms = geom_end + 1, 0, [], []
    if basis is not None:
        replaced = _soc_basis_atoms(lines, geom_end, end99, basis, ecp_form, soscale, log)
        lines = lines[:geom_end + 1] + replaced + lines[end99:]
        upper = [ln.strip().upper() for ln in lines]
        end99 = geom_end + 1 + len(replaced)
    while i < end99:
        try:
            number = int(lines[i].split()[0])
            ecp_start, ecp_end, nxt = _basis_block_span(lines, i)
        except (ValueError, IndexError):
            raise SocError(f"could not read the basis-set input at line {i + 1}: {lines[i]!r}")
        key = lines[ecp_start].strip().upper() if ecp_end > ecp_start else None
        z = _atom_z(number)
        atoms.append(number)
        if key == "INPSOC" and _is_ecp_label(number):
            # A deck derived from a SOC deck carries the INPSOC records already
            # (or soc_basis has just written them: no note).
            records = _kept_inpsoc(z, lines[ecp_start:ecp_end], soscale,
                                   log if basis is None else (lambda m: None))
            if ecp_form == "stutsc":
                records = stutsc_records(z, soscale)
                log(f"SOC deck: {_symbol(z)}'s INPSOC records become CRYSTAL's "
                    f"STUTSC (soc_ecp)")
            out += [lines[i]] + records + lines[ecp_end:nxt]
            ecp_atoms += 1
        elif key == "STUTSC" and _is_ecp_label(number):
            # A deck derived from a SOC deck written with STUTSC.
            # (a block soc_basis has just written is not "kept": no note)
            out += [lines[i]] + _kept_stutsc(z, lines[i:nxt], soscale, ecp_form,
                                             log if basis is None else (lambda m: None))
            out += lines[ecp_end:nxt]
            ecp_atoms += 1
        elif key is not None:
            if key != "INPUT" or not _is_ecp_label(number):
                raise SocError(
                    f"atom {number}: only an ECP entered with INPUT can be given its "
                    f"spin-orbit part, not {lines[ecp_start].strip()} (a SOC deck takes "
                    f"INPUT, which MACE converts; INPSOC records as MACE writes them; or "
                    f"CRYSTAL's STUTSC on MACE's library basis for "
                    f"{_stutsc_elements()} or on the official pob-TZVP-rev2-SOC block)")
            records = soc_ecp_records(z, lines[ecp_start:ecp_end], soscale)
            if ecp_form == "stutsc":
                records = stutsc_records(z, soscale)
                log(f"SOC deck: {_symbol(z)} gets CRYSTAL's STUTSC spin-orbit "
                    f"ECP (soc_ecp), the same potential as MACE's INPSOC (measured)")
            out += [lines[i]] + records + lines[ecp_end:nxt]
            ecp_atoms += 1
        else:
            out += lines[i:nxt]
            all_electron.append(number)
        i = nxt
    if i != end99:
        raise SocError("could not read the basis-set input: its blocks do not end at '99 0'")
    if ecp_atoms == 0:
        raise SocError("no atom in the deck carries an ECP, so there is no spin-orbit "
                       "(SOREP) operator for SOC to include (CRYSTAL23 manual p. 175)")
    if all_electron and basis is None:      # _soc_basis_atoms names them itself
        _all_electron_notes(all_electron, log)
    _f_element_warning(atoms, guess, log)

    # The rest of the basis-set input: records after '99 0', up to its END.
    try:
        block2_end = next(j for j in range(end99 + 1, len(lines))
                          if lines[j].strip().upper() == "END")
    except StopIteration:
        raise SocError("could not find the END of the basis-set input after '99 0'")
    _check_basis_input_records(lines[end99 + 1:block2_end], basis)

    # Block 3 (after that END): check it, then add TWOCOMPON.
    block3_start = block2_end + 1
    out += lines[end99:block3_start]
    rest = lines[block3_start:]
    problems, scfdir, dft_words = _block3_problems(rest)
    refusals = _block3_refusals(problems, dft_words)
    if refusals:
        raise SocError(_joined_refusals(refusals))
    if scfdir is None:
        raise SocError("no SCFDIR record to place the TWOCOMPON block before")

    twocompon = ["TWOCOMPON", "SOC"] + ([guess.upper()] if guess else []) + ["END"]
    rest = rest[:scfdir] + twocompon + rest[scfdir:]
    rest = _monkhorst_equals_gilat(_strip_records(rest), log)
    if smear is not None:
        rest = _with_smear(rest, smear, log)
    out += _scf_convergence(rest, log, fmixing, guess)
    return "\n".join(out) + "\n"


# Basis-set input records (after '99 0') a SOC deck keeps. CHEMOD: the manual
# pairs it with the 2c atomic guess, for a different guess on atoms of the same
# element (p. 172). The others (GHOSTS, CHARGED, ...) are not mentioned in
# chapter 6, which says anything it does not mention is not supported (p. 166).
_BASIS_INPUT_KEPT = {"CHEMOD"}


def _check_basis_input_records(records: List[str], basis: Optional[str]) -> None:
    """Refuse basis-set input records (between '99 0' and its END) that a SOC
    deck cannot keep, naming them; CHEMOD is kept, except with soc_basis,
    which replaces the blocks whose shells CHEMOD gives charges for."""
    words = [ln.split()[0].upper() for ln in records
             if ln.strip() and not _is_number_line(ln.strip())]
    refused = [w for w in dict.fromkeys(words) if w not in _BASIS_INPUT_KEPT]
    if refused:
        names = ", ".join(refused)
        raise SocError(
            f"{names} in the basis-set input (after '99 0'): the two-component chapter of "
            f"the CRYSTAL23 manual (ch. 6, pp. 165-175) does not mention "
            f"{'it' if len(refused) == 1 else 'them'}, and anything it does not mention is "
            f"not supported in a 2c-SCF (p. 166), so MACE does not write a SOC deck with "
            f"{'it' if len(refused) == 1 else 'them'}. CHEMOD is kept (the manual pairs it "
            f"with the 2c guess, p. 172)")
    if "CHEMOD" in words and basis is not None:
        raise SocError(
            f"CHEMOD gives charges for the shells of the deck's basis, which soc_basis "
            f"{basis} replaces with other shells; write the SOC deck without soc_basis, or "
            f"without CHEMOD")


# The SCF records of the charged-system settings (--net-charge, --spinlock, or
# a parent's records carried over). Not in the chapter 6 list (p. 166).
_CHARGE_SCF_RECORDS = ("DOPING", "SPINLOCK")


def charge_records_refusal(deck: str, basis: Optional[str] = None) -> Optional[str]:
    """Why a SOC deck (one with TWOCOMPON) cannot keep the charged-system
    records added after soc_deck wrote it (cif2d12's and opt2d12's charge
    options, or records carried from a parent), checked as soc_deck checks a
    deck's own: CHARGED and the other basis-set input records but CHEMOD,
    CHEMOD with soc_basis, and DOPING / SPINLOCK in the SCF input. None when
    the deck is not a SOC deck or has none of them."""
    lines = deck.splitlines()
    words = [ln.split()[0].upper() if ln.split() else "" for ln in lines]
    if "TWOCOMPON" not in words:
        return None
    end99 = next((i for i, ln in enumerate(lines) if ln.split() == ["99", "0"]), None)
    if end99 is not None:
        block2_end = next((j for j in range(end99 + 1, len(lines)) if words[j] == "END"), None)
        if block2_end is not None:
            try:
                _check_basis_input_records(lines[end99 + 1:block2_end], basis)
            except SocError as exc:
                return str(exc)
    found = [w for w in _CHARGE_SCF_RECORDS if w in words]
    if found:
        it = "it" if len(found) == 1 else "them"
        return (f"{' and '.join(found)} in the SCF input (the charged-system settings): the "
                f"two-component chapter of the CRYSTAL23 manual (ch. 6, pp. 165-175) does not "
                f"mention {it}, and anything it does not mention is not supported in a 2c-SCF "
                f"(p. 166), so MACE does not write a SOC deck with {it}")
    return None


def _kept_inpsoc(z: int, records: List[str], soscale: float, log) -> List[str]:
    """The INPSOC records of a deck derived from a SOC deck, kept as they are
    when they are the ones soc_ecp_records writes for ``z`` - same numbers,
    INTERNAL convention, same SOSCALE. Anything else is refused: MACE does not
    take over an INPSOC potential it did not write."""
    ecp = load_so_ecp(z)
    try:
        name, scale = records[1].split()[:2]
        same = (name.upper() == "INTERNAL" and float(scale) == soscale
                and parse_inpsoc(records) == parse_inpsoc(inpsoc_records(ecp, z, soscale)))
    except (ValueError, IndexError, SocError):
        same = False
    if not same:
        raise SocError(
            f"{ecp.symbol} (Z={z}): the deck's INPSOC ECP is not the spin-orbit ECP "
            f"{200 + z}.mol as MACE writes it (INTERNAL, SOSCALE {soscale}); MACE does "
            f"not take over an INPSOC potential it did not write")
    log(f"SOC deck: {ecp.symbol} keeps the INPSOC spin-orbit ECP it already has")
    return list(records)


def _monkhorst_equals_gilat(block3: List[str], log) -> List[str]:
    """block3 with the SHRINK Gilat net (ISP) set equal to the Monkhorst net.

    A 2c-SCF of fcc Au with SHRINK 12 24 never converged (charge
    normalization factor 1.35-1.69) and converged with 12 12 (HPCC). The
    directional form "0 ISP" / "IS1 IS2 IS3" (every SLAB, POLYMER and P1 deck)
    keeps its own mesh and gets ISP = the largest ISi."""
    out = list(block3)
    for i, ln in enumerate(out):
        if ln.strip().upper() != "SHRINK":
            continue
        record = out[i + 1].split()
        if record[0] == "0":
            isp = str(max(int(k) for k in out[i + 2].split()))
            new = f"0 {isp}"
        else:
            new = f"{record[0]} {record[0]}"
        if out[i + 1].strip() != new:
            log(f"SOC deck: SHRINK {out[i + 1].strip()} -> {new} "
                f"(Gilat net equal to the Monkhorst net)")
            out[i + 1] = new
    return out


# FMIXING and MAXCYCLE for a 2c-SCF, measured on HPCC (TOLDEE 7, only FMIXING
# changed): FMIXING 50 converged fastest in all three systems tried - Bi2
# bilayer 12 cycles, PbTe 9, fcc Au (12 12) 25 - with the energies FMIXING 85
# gave (Bi2, PbTe within 3e-7 Ha; Au within 5e-6 Ha), while FMIXING 30 aborted
# Bi2 in cycle 1. The manual warns that above 50% mixing can "force the
# stabilization of the total energy value, without a real self consistency"
# (p. 110). SOC_FMIXING is written in every SOC deck, whatever FMIXING the deck
# had, unless a "soc_fmixing" key in a cif2d12 options file or an opt2d12
# template asks for another value (the soc_deck ``fmixing`` argument). A value
# replaced is named in the log.
SOC_FMIXING = 50
SOC_MIN_MAXCYCLE = 200
# FMIXING for a 2c restart (soc_guess), measured on a Bi2 bilayer only (HPCC):
# GUESSPSO from a 2c matrix diverged at 50 (charges 19.7/26.3) and converged
# in 5 cycles at 80 (3 cycles from a 6-cycle parent); GUESSPNOSO from a P1
# unrestricted 1c matrix diverged at 50 (E -420.87 Ha by cycle 70) and
# converged in 29 cycles at 80, to the energy a cold SOC start reaches in 12
# cycles at 50. "soc_fmixing" wins.
SOC_RESTART_FMIXING = 80
SOC_GUESSES = ("guesspso", "guesspnoso")


def _checked_guess(value) -> Optional[str]:
    """A "soc_guess" value, "guesspso" or "guesspnoso", or None."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in SOC_GUESSES:
        raise SocError(f'soc_guess must be "guesspso" (restart from a 2c run) or '
                       f'"guesspnoso" (restart from an unrestricted 1c run without '
                       f'symmetry), manual p. 173, not {value!r}')
    return value.strip().lower()


def _checked_fmixing(value) -> Optional[int]:
    """A "soc_fmixing" value as an int, or None when none was given.

    FMIXING's IPMIX is a percentage (manual p. 110), so 0 to 100."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SocError(f"soc_fmixing must be a whole number from 0 to 100, not {value!r}")
    try:
        number = int(value)
    except ValueError:
        raise SocError(f"soc_fmixing must be a whole number from 0 to 100, not {value!r}")
    if not 0 <= number <= 100:
        raise SocError(f"soc_fmixing must be a whole number from 0 to 100 "
                       f"(FMIXING is a percentage, manual p. 110), not {value!r}")
    return number


def _scf_convergence(block3: List[str], log, fmixing: Optional[int] = None,
                     guess: Optional[str] = None) -> List[str]:
    """block3 with FMIXING SOC_FMIXING (or ``fmixing``, the "soc_fmixing"
    setting) and at least SOC_MIN_MAXCYCLE cycles.

    The FMIXING is written whatever the deck had; a different value in the
    deck (MACE's 30, or the parent's or template's own) is replaced, and the
    log says which. A MAXCYCLE below the minimum is kept, with a warning."""
    out = list(block3)
    keys = [ln.strip().upper() for ln in out]
    end = len(out) - 1 - keys[::-1].index("END")

    target = SOC_FMIXING if fmixing is None else fmixing
    why = ("soc_fmixing" if fmixing is not None else
           f"the SOC default; \"soc_fmixing\" in the options file or template sets another")
    if fmixing is None and guess == "guesspso":
        target = SOC_RESTART_FMIXING
        why = ("the GUESSPSO restart default, measured on Bi2 only: 50 diverged (charges "
               "19.7/26.3), 80 converged in 5 cycles; \"soc_fmixing\" sets another")
    elif fmixing is None and guess == "guesspnoso":
        target = SOC_RESTART_FMIXING
        why = ("the GUESSPNOSO restart default, measured on Bi2 only: 50 diverged, 80 "
               "converged in 29 cycles (a cold SOC start took 12); \"soc_fmixing\" sets "
               "another")
    if "FMIXING" in keys:
        i = keys.index("FMIXING")
        value = out[i + 1].split()[0]
        if int(value) != target:
            out[i + 1] = str(target)
            log(f"SOC deck: FMIXING {value} replaced by {target} ({why})")
    else:
        out[end:end] = ["FMIXING", str(target)]
        end += 2
        log(f"SOC deck: FMIXING {target} added ({why})")

    if "MAXCYCLE" in keys:
        i = keys.index("MAXCYCLE")
        value = int(out[i + 1].split()[0])
        if value < SOC_MIN_MAXCYCLE:
            log(f"Warning: SOC deck keeps MAXCYCLE {value} as set; 2c-SCF runs "
                f"can need {SOC_MIN_MAXCYCLE} or more cycles")
    else:
        out[end:end] = ["MAXCYCLE", str(SOC_MIN_MAXCYCLE)]
    return out


def _strip_records(block3: List[str]) -> List[str]:
    """block3 without the _STRIPPED records and their argument lines."""
    kept, skip = [], 0
    for ln in block3:
        if skip:
            skip -= 1
            continue
        key = ln.strip().upper()
        if key in _STRIPPED:
            skip = _STRIPPED[key]
            continue
        kept.append(ln)
    return kept


# --- SMEAR for metals ---------------------------------------------------------
#
# fcc Au in a 2c-SCF aborted on MACE's default SHRINK 10 10, and on 11 11, in
# its first cycles: "POSSIBLY CONDUCTING STATE - EFERMI(AU) ..." and then
# "ERROR **** ZERO **** FERMI ENERGY NOT IN INTERVAL". 12 12 converged, and so
# did 10 10 with SMEAR 0.005 (HPCC). FMIXING does not cure it (10 10 with
# FMIXING 50 aborts as with 85), and for a metal the mixing value can change
# which SCF solution the run settles in (Au at 12 12: FMIXING 50-85 all near
# -135.86080 Ha, FMIXING 30 7 meV lower after 128 cycles). SMEAR, or a denser
# mesh, is the fix. SMEAR takes one record, WIDTH = k_B T in hartree (manual
# p. 126), and is among the SCF keywords the 2c-SCF accepts (p. 169).
#
# MACE cannot know beforehand that a system is a metal, but a parent's .out
# can show it. After each Fermi-level search CRYSTAL prints either
#     POSSIBLY CONDUCTING STATE - EFERMI(AU) -1.4886526E-01 (RES. CHARGE ...)
# or the band edges and the gap
#     TOP OF VALENCE BANDS -    BAND     12; K    5; EIG -1.4892012E-01 AU
#     DIRECT ENERGY BAND GAP:   0.0024 eV      (INDIRECT ENERGY BAND GAP: ...)
# (1c graphene and polyyne runs in tests/data/low_dim_groups; the Au 2c run
# prints the first form). The report of the parent's last SCF cycle decides:
# a conducting state, or a gap below NEAR_ZERO_GAP_EV, counts as metallic.
# The threshold is MACE's choice, not a measured limit.
NEAR_ZERO_GAP_EV = 0.1
SOC_METAL_SMEAR_EXAMPLE = 0.005
_ELECTRONIC_STATE_LINE = re.compile(
    r"^[ \t]*(?:(POSSIBLY CONDUCTING STATE) - EFERMI\(AU\)"
    r"|(?:IN)?DIRECT ENERGY BAND GAP:[ \t]*(-?\d+(?:\.\d*)?)[ \t]*EV)",
    re.MULTILINE | re.IGNORECASE)
_SCF_CYCLE_LINE = re.compile(r"^[ \t]*CYC[ \t]+\d+[ \t]+ETOT", re.MULTILINE)


def parent_metal_evidence(content: str) -> Optional[str]:
    """Why a CRYSTAL .out looks metallic, or None.

    Reads the electronic-state report of the last SCF cycle (every report
    after the last "CYC n ETOT" line before it, so both spins of an
    unrestricted run): "POSSIBLY CONDUCTING STATE", or a direct/indirect band
    gap below NEAR_ZERO_GAP_EV."""
    reports = list(_ELECTRONIC_STATE_LINE.finditer(content))
    if not reports:
        return None
    cycles = [m.start() for m in _SCF_CYCLE_LINE.finditer(content, 0, reports[-1].start())]
    last = [m for m in reports if m.start() >= (cycles[-1] if cycles else 0)]
    if any(m.group(1) for m in last):
        return 'its last SCF cycle reports "POSSIBLY CONDUCTING STATE"'
    gap = min(float(m.group(2)) for m in last)
    if gap < NEAR_ZERO_GAP_EV:
        return f"its last SCF cycle reports a band gap of {gap:g} eV"
    return None


def checked_smear(value) -> Optional[float]:
    """A "smear" value (SMEAR's WIDTH, hartree, manual p. 126) as a float, or
    None when none was given. It must be above 0 and show in the 6 decimals
    MACE writes."""
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        width = float(value)
    except (TypeError, ValueError):
        raise SocError(f"smear must be a width in hartree above 0 (SMEAR, manual "
                       f"p. 126), not {value!r}")
    if not width >= 0.0000005 or width != width or width == float("inf"):
        raise SocError(f"smear must be a width in hartree above 0 (SMEAR, manual "
                       f"p. 126), not {value!r}")
    return width


def deck_has_smear(deck: str) -> bool:
    """Whether a deck has a SMEAR record (line 1, the title, is not a record)."""
    return any(ln.strip().upper() == "SMEAR" for ln in deck.split("\n")[1:])


def _with_smear(block3: List[str], width: float, log) -> List[str]:
    """block3 with SMEAR ``width``: the deck's own SMEAR gets that width, or a
    SMEAR record goes right after the SHRINK records (where MACE's writers put
    it). Noted either way."""
    out = list(block3)
    keys = [ln.strip().upper() for ln in out]
    text = f"{width:.6f}"
    if "SMEAR" in keys:
        j = keys.index("SMEAR")
        if float(out[j + 1].split()[0]) != float(text):
            log(f"SOC deck: SMEAR {out[j + 1].strip()} replaced by {text} (smear)")
            out[j + 1] = text
        return out
    if "SHRINK" not in keys:
        raise SocError("smear was asked for, but the deck has no SHRINK (no k-points), so "
                       "there is no Fermi surface to smear")
    i = keys.index("SHRINK")
    at = i + (3 if out[i + 1].split()[0] == "0" else 2)
    out[at:at] = ["SMEAR", text]
    log(f"SOC deck: SMEAR {text} added (smear)")
    return out


def metal_parent_warning(evidence: str) -> str:
    """The warning for a SOC deck without SMEAR whose parent looks metallic."""
    return (f"Warning: the parent looks metallic ({evidence}) and this SOC deck has no "
            f"SMEAR. A 2c-SCF of fcc Au aborted in its first cycles on SHRINK 10 10 and "
            f"11 11 (\"FERMI ENERGY NOT IN INTERVAL\") and converged at 12 12, or at 10 10 "
            f"with SMEAR {SOC_METAL_SMEAR_EXAMPLE} (HPCC); for a metal the FMIXING value can "
            f"also change the SCF solution reached. SMEAR (\"smear\": <width in hartree> "
            f"in the template) or a denser mesh is the fix.")


CIF_METAL_WARNING = (
    "Warning: SOC decks here have no SMEAR, and MACE cannot tell from a CIF whether a "
    "system is a metal. A metal may need SMEAR (\"smear\": <width in hartree> in the "
    "options file) or a denser mesh: fcc Au in a 2c-SCF aborted on SHRINK 10 10 and "
    f"11 11 and converged at 12 12, or at 10 10 with SMEAR {SOC_METAL_SMEAR_EXAMPLE} (HPCC).")
_cif_metal_warned = []


def cif_metal_warning_once(log=print) -> None:
    """Print CIF_METAL_WARNING, the first time only in this run: it holds for
    every SOC deck written from a CIF without SMEAR alike."""
    if not _cif_metal_warned:
        _cif_metal_warned.append(True)
        log(CIF_METAL_WARNING)


# --- 2c restarts: may this deck start from that parent's density matrix? ------
#
# GUESSPSO / GUESSPNOSO read the parent's density matrix (fort.20) as the guess
# (manual p. 173). As for GUESSP (pp. 114-115) CRYSTAL does not check that the
# matrix fits, so the two decks must have the same symmetry frame and the same
# atoms, shells and basis functions in the same order. The ECP records are left
# out of the comparison (GUESSPNOSO goes from INPUT to INPSOC/STUTSC, a GUESSPSO
# child may go from INPSOC to STUTSC), and the child is compared as written,
# after soc_deck (soc_basis replaces shells). These mirror
# mace/workflow/guessp_chain.geometry_frame without importing mace.

_GEOMETRY_STOP = {"OPTGEOM", "FREQCALC", "END", "ENDG", "ENDGEOM", "BASISSET"}


def _norm(line: str) -> Tuple[str, ...]:
    """Tokens of a record, numbers compared as numbers (2.0 == 2.)."""
    out = []
    for tok in line.split():
        try:
            out.append(repr(float(tok)))
        except ValueError:
            out.append(tok.upper())
    return tuple(out)


def geometry_frame(deck: str) -> Optional[tuple]:
    """(dimensionality, symmetry header, NAT tuple, extra geometry records) of
    a deck, or None when it cannot be read."""
    lines = [ln.strip() for ln in deck.splitlines()]
    try:
        dim = lines[1].upper()
        i = 2
        if dim == "CRYSTAL":
            flags = lines[i].split()
            header = [_norm(lines[i]), _norm(lines[i + 1])]
            i += 2
            if len(flags) >= 3 and int(flags[2]) > 1:
                header.append(_norm(lines[i]))
                i += 1
            i += 1
        elif dim in ("SLAB", "POLYMER"):
            header = [_norm(lines[i])]
            i += 2
        elif dim == "MOLECULE":
            header = [_norm(lines[i])]
            i += 1
        else:
            return None
        natoms = int(lines[i].split()[0])
        atoms = tuple(int(lines[i + 1 + k].split()[0]) for k in range(natoms))
        i += 1 + natoms
        extra = []
        while i < len(lines) and lines[i].upper() not in _GEOMETRY_STOP:
            if lines[i]:
                extra.append(_norm(lines[i]))
            i += 1
        if i >= len(lines):
            return None
    except (IndexError, ValueError):
        return None
    return (dim, tuple(header), natoms, atoms, tuple(extra))


def shell_signature(deck: str) -> Optional[tuple]:
    """((NAT, shell records), ...) per atom of a deck's explicit basis input,
    ECP records left out, or None when there is none to read."""
    lines = deck.rstrip("\n").split("\n")
    upper = [ln.strip().upper() for ln in lines]
    if "BASISSET" in upper or "99 0" not in upper:
        return None
    end99 = upper.index("99 0")
    try:
        start = max(k for k in range(end99) if upper[k] == "END") + 1
    except ValueError:
        return None
    blocks, i = [], start
    try:
        while i < end99:
            if not lines[i].strip():
                i += 1
                continue
            _s, ecp_end, nxt = _basis_block_span(lines, i)
            blocks.append((int(lines[i].split()[0]),
                           tuple(_norm(ln) for ln in lines[ecp_end:nxt] if ln.strip())))
            i = nxt
    except (ValueError, IndexError):
        return None
    return tuple(blocks) if i == end99 else None


def restart_mismatch(parent_deck: str, child_deck: str) -> Optional[str]:
    """Why ``child_deck`` (as written) cannot restart from ``parent_deck``'s
    density matrix, or None."""
    parent_frame, child_frame = geometry_frame(parent_deck), geometry_frame(child_deck)
    if parent_frame is None or child_frame is None:
        return "the symmetry and atom list of the parent deck and this deck could not be read"
    # SYMMREMO (p. 78) only stops CRYSTAL using the symmetry the frame
    # describes, and opening TWOCOMPON does that for the 2c child (p. 169);
    # GUESSPNOSO asks for a parent run so (p. 173). Whether the atoms then come
    # in the same order is not measured; the caller warns.
    parent_frame, child_frame = (f[:4] + (tuple(r for r in f[4] if r != ("SYMMREMO",)),)
                                 for f in (parent_frame, child_frame))
    if parent_frame != child_frame:
        return "the symmetry or the atom list differs from the parent's"
    parent_shells, child_shells = shell_signature(parent_deck), shell_signature(child_deck)
    if parent_shells is None or child_shells is None:
        return "the explicit basis input of the parent deck and this deck could not be read"
    if parent_shells != child_shells:
        return ("the basis shells differ from the parent's (soc_basis, or another basis, "
                "changes the basis functions)")
    return None


# The keys of a cif2d12 options file or opt2d12 template that tune a SOC deck,
# and the soc_deck argument each one sets.
SOC_OPTION_KEYS = {"soc_fmixing": "fmixing", "smear": "smear", "soc_ecp": "ecp_form",
                   "soc_basis": "basis", "soc_guess": "guess"}

# Why a SOC deck with soc_guess has no parent to restart from (cif2d12).
NO_PARENT_FOR_GUESS = ("soc_guess restarts from a parent run's density matrix (manual p. 173), "
                       "and a deck written from a CIF has no parent; use soc_guess in an "
                       "opt2d12 template")


def soc_deck_options(options) -> Dict[str, object]:
    """soc_deck keyword arguments from the SOC keys of ``options``."""
    options = options or {}
    return {arg: options[key] for key, arg in SOC_OPTION_KEYS.items()
            if options.get(key) is not None}


class SocDeckBuffer(io.StringIO):
    """Collects a deck being written so soc_deck can rewrite it before it
    reaches the atomic_deck file; ``discard`` is passed through. ``options``
    are the writer's options or settings, read for the SOC_OPTION_KEYS."""

    def __init__(self, deck_file, log=print, options=None, restart_from=None):
        super().__init__()
        self._deck = deck_file
        self._log = log
        self._kwargs = soc_deck_options(options)
        # The parent deck a "soc_guess" deck restarts from (its text).
        self._restart_from = restart_from
        self.written = None

    def _restart_refusal(self, deck: str) -> Optional[str]:
        if self._kwargs.get("guess") is None:
            return None
        if self._restart_from is None:
            return NO_PARENT_FOR_GUESS
        mismatch = restart_mismatch(self._restart_from, deck)
        if mismatch:
            return (f"{self._kwargs['guess'].upper()} cannot restart from the parent: "
                    f"{mismatch} (CRYSTAL does not check that the matrix fits, manual "
                    f"pp. 114-115 and 173)")
        return None

    def discard(self):
        self._deck.discard()

    def will_have_smear(self) -> bool:
        """Whether the SOC deck will carry SMEAR: the deck written so far has
        it, or a "smear" setting asks for it."""
        return self._kwargs.get("smear") is not None or deck_has_smear(self.getvalue())

    def set_smear(self, width) -> None:
        """Ask for SMEAR ``width`` (hartree), as a "smear" setting does."""
        self._kwargs["smear"] = width

    def refusal(self, soscale: float = 1.0) -> Optional[str]:
        """Why finish would refuse the deck written so far, or None; writes
        and prints nothing, so a refusal can come before any warning."""
        try:
            deck = soc_deck(self.getvalue(), soscale, log=lambda message: None, **self._kwargs)
        except SocError as exc:
            return str(exc)
        return self._restart_refusal(deck)

    def finish(self, soscale: float = 1.0) -> Optional[str]:
        """Write the SOC deck; on refusal discard it and return the reason.
        ``written`` then holds the deck written."""
        try:
            deck = soc_deck(self.getvalue(), soscale, log=self._log, **self._kwargs)
        except SocError as exc:
            self._deck.discard()
            return str(exc)
        refusal = self._restart_refusal(deck)
        if refusal:
            self._deck.discard()
            return refusal
        self.written = deck
        self._deck.write(deck)
        return None
