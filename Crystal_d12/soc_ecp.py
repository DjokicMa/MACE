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
SOPSEUD_DIR = os.path.join(_HERE, "..", "Crystal_d3", "Archived", "basis", "sopseud")
STUTTGART_DIR = os.path.join(_HERE, "basis_sets", "stuttgart")

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


def scalar_part(ecp: SoEcp, z: int) -> Tuple[float, List[tuple], List[List[tuple]]]:
    """The SO-ECP's scalar (AREP) part in the form parse_input_ecp returns."""
    local = [t.values() for t in ecp.local if not _is_zero(t.coefficient)]
    per_l = [[t.values() for t in ecp.semilocal[l]] if l < ecp.lmax else [] for l in range(5)]
    return float(z - ecp.ncore), local, per_l


def read_stuttgart(z: int) -> Optional[Tuple[List[str], List[str]]]:
    """(ECP lines from INPUT on, valence shell lines) of stuttgart/<200+Z>,
    or None if there is no such file or it holds no INPUT ECP."""
    path = os.path.join(STUTTGART_DIR, str(200 + z))
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if len(lines) < 3 or lines[1].strip() != "INPUT":
        return None
    counts = [int(c) for c in lines[2].split()[1:]]
    end = 3 + sum(counts)
    return lines[1:end], lines[end:]


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
    if not same_scalar_potential(ecp, z, input_lines):
        raise SocError(
            f"{ecp.symbol}: the deck's ECP is not the scalar part of the spin-orbit ECP "
            f"{200 + z}.mol (different core or parameters), so its valence basis does not "
            f"belong to that potential; use a basis set built on it")
    return inpsoc_records(ecp, z, soscale)


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
TWOC_DFT_KEYWORDS = {
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


def soc_deck(deck: str, soscale: float = 1.0, log=print) -> str:
    """Turn a finished single-point MACE deck into a 2c-SCF SOC deck.

    Every atom whose basis block carries an INPUT ECP (conventional atomic
    number 200+Z) has it replaced by the INPSOC form of the spin-orbit ECP for
    Z - only if its scalar part is that very potential - and keeps its valence
    shells. All-electron atoms keep their block, since the SOC operator lives in
    the ECP (p. 175).
    A TWOCOMPON block holding SOC goes into the SCF input, before SCFDIR, where
    the stock 2c test deck has it. Raises SocError, and returns no deck, for
    anything the manual says the 2c-SCF does not support."""
    lines = deck.rstrip("\n").split("\n")
    upper = [ln.strip().upper() for ln in lines]

    for key, what in UNSUPPORTED_ANYWHERE.items():
        if key in upper:
            raise SocError(f"{what} ({key}) is not available in a two-component SCF "
                           f"(CRYSTAL23 manual p. 166); SOC decks are single points")
    if "TWOCOMPON" in upper:
        raise SocError("the deck already has a TWOCOMPON block")
    if "BASISSET" in upper:
        raise SocError("the deck uses an internal basis-set library (BASISSET); a SOC "
                       "deck needs explicit per-element basis input for its spin-orbit ECPs")

    try:
        end99 = upper.index("99 0")
    except ValueError:
        raise SocError("no explicit basis-set input (no '99 0' record) found in the deck")
    geom_end = upper.index("END")
    if geom_end > end99:
        raise SocError("could not find the end of the geometry input")

    # Basis section: replace ECP atoms' blocks.
    out = lines[:geom_end + 1]
    i, ecp_atoms = geom_end + 1, 0
    while i < end99:
        try:
            number = int(lines[i].split()[0])
            ecp_start, ecp_end, nxt = _basis_block_span(lines, i)
        except (ValueError, IndexError):
            raise SocError(f"could not read the basis-set input at line {i + 1}: {lines[i]!r}")
        if ecp_end > ecp_start:
            if lines[ecp_start].strip().upper() != "INPUT" or not 200 < number < 300:
                raise SocError(
                    f"atom {number}: only an ECP entered with INPUT can be given its "
                    f"spin-orbit part, not {lines[ecp_start].strip()}")
            out += [lines[i]] + soc_ecp_records(number - 200, lines[ecp_start:ecp_end], soscale)
            out += lines[ecp_end:nxt]
            ecp_atoms += 1
        else:
            out += lines[i:nxt]
        i = nxt
    if i != end99:
        raise SocError("could not read the basis-set input: its blocks do not end at '99 0'")
    if ecp_atoms == 0:
        raise SocError("no atom in the deck carries an ECP, so there is no spin-orbit "
                       "(SOREP) operator for SOC to include (CRYSTAL23 manual p. 175)")

    # Block 3 (from the END after '99 0' on): check it, then add TWOCOMPON.
    block3_start = end99 + 2
    out += lines[end99:block3_start]
    rest = lines[block3_start:]
    problems, inside, scfdir = [], None, None
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
                elif word in TWOC_EXCHANGE | TWOC_CORRELATION and rest[j - 1].strip().upper() in ("EXCHANGE", "CORRELAT"):
                    pass
                elif word not in TWOC_FUNCTIONALS and word not in TWOC_DFT_KEYWORDS:
                    problems.append(word)
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
    if problems:
        raise SocError(
            "not supported in a two-component SCF (CRYSTAL23 manual ch. 6 lists every "
            "keyword it accepts, p. 166): " + ", ".join(dict.fromkeys(problems)))
    if scfdir is None:
        raise SocError("no SCFDIR record to place the TWOCOMPON block before")

    rest = rest[:scfdir] + ["TWOCOMPON", "SOC", "END"] + rest[scfdir:]
    out += _scf_convergence(_monkhorst_equals_gilat(_strip_records(rest), log), log)
    return "\n".join(out) + "\n"


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


# FMIXING and MAXCYCLE for a 2c-SCF. On HPCC, FMIXING 30 let Bi2 diverge and
# left Au unconverged after 60 cycles; FMIXING 85 converged Bi2 in 34 cycles.
SOC_FMIXING = 85
SOC_MIN_MAXCYCLE = 200
# What MACE writes when nobody chose a value (write_scf_section's callers).
# The deck cannot tell it from the same number chosen on purpose, so it is
# read as "not set"; any other value was set by the parent or template.
# (MACE's default MAXCYCLE, 800, is already above SOC_MIN_MAXCYCLE.)
_DEFAULT_FMIXING = 30


def _scf_convergence(block3: List[str], log) -> List[str]:
    """block3 with FMIXING 85 and at least 200 cycles, unless set otherwise.

    A value other than MACE's default was set by the parent or template: it
    is kept, with a warning when it is not what 2c runs need."""
    out = list(block3)
    keys = [ln.strip().upper() for ln in out]
    end = len(out) - 1 - keys[::-1].index("END")

    if "FMIXING" in keys:
        i = keys.index("FMIXING")
        value = int(out[i + 1].split()[0])
        if value == _DEFAULT_FMIXING:
            out[i + 1] = str(SOC_FMIXING)
            log(f"SOC deck: FMIXING {value} -> {SOC_FMIXING}")
        elif value != SOC_FMIXING:
            log(f"Warning: SOC deck keeps FMIXING {value} as set; 2c-SCF runs have "
                f"diverged with FMIXING 30 and converged with {SOC_FMIXING}")
    else:
        out[end:end] = ["FMIXING", str(SOC_FMIXING)]
        end += 2

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


class SocDeckBuffer(io.StringIO):
    """Collects a deck being written so soc_deck can rewrite it before it
    reaches the atomic_deck file; ``discard`` is passed through."""

    def __init__(self, deck_file, log=print):
        super().__init__()
        self._deck = deck_file
        self._log = log

    def discard(self):
        self._deck.discard()

    def finish(self, soscale: float = 1.0) -> Optional[str]:
        """Write the SOC deck; on refusal discard it and return the reason."""
        try:
            self._deck.write(soc_deck(self.getvalue(), soscale, log=self._log))
            return None
        except SocError as exc:
            self._deck.discard()
            return str(exc)
