"""Charged systems and ion clusters: CHEMOD, DOPING, CHARGED and SPIN for a deck.

What the records do (CRYSTAL23 manual, with what was measured on HPCC):

    CHEMOD  (basis-set input, p.80)  "NC", then NC records "LA CH(1..NS)": the
            shell charges of atom LA for the starting guess. NS must equal the
            number of shells of that atom's basis. The geometric symmetry is
            checked with the new configuration and CRYSTAL stops if it would be
            lowered ("No automatic reduction of the symmetry is allowed").
    CHARGED (basis-set input, p.80)  allow a non-neutral cell; a uniform
            background neutralises it. "Do not use for total energy comparison."
    DOPING  (SCF input, p.103)       add (positive) or remove (negative)
            electrons; a compensating background is added when needed.

Measured (CRYSTAL/23-intel-2023a, see the README section "Charged systems and
ion clusters"):

* With an internal BASISSET library, CHEMOD (+ CHARGED) changes nothing in the
  electron count: the guess is renormalised ("CHARGE NORMALIZATION FACTOR"),
  so the count comes from DOPING and CHEMOD only shapes the guess.
* With an explicit basis, a non-neutral CHEMOD does set the count (with
  CHARGED it matches DOPING to 1e-9 Ha); CHEMOD + DOPING then counts the
  charge twice (Li+/EC, 47 electrons instead of 48), so DOPING is never added.
  A periodic cell needs CHARGED (else "UNIT CELL NOT NEUTRAL").
* DOPING prints nothing and leaves the header's electron count unchanged; the
  real count is the Mulliken "NO. OF ELECTRONS" line.
* GHOSTS + DOPING with an odd count stops ("NALPHA-NBETA NOT COMPATIBLE WITH
  NUMBER OF ELECTRONS", parity checked before doping).

So the records are:

    internal BASISSET            DOPING -q; CHEMOD (optional) only shapes the guess
    explicit basis, CHEMOD, q!=0 CHEMOD + CHARGED, no DOPING; CHEMOD must remove q
    explicit basis, no CHEMOD    DOPING -q
    q = 0                        CHEMOD only (never "DOPING 0")

A periodic explicit-basis cell gets DOPING only unless CHEMOD is asked for
(CHARGED adds a background). Standard library only.
"""

import importlib.util
import math
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from mace.utils import ion_fragments as frag
except ImportError:
    try:
        import ion_fragments as frag  # copied flat by copy_dependencies
    except ImportError:
        _spec = importlib.util.spec_from_file_location(
            "_mace_ion_fragments",
            Path(__file__).resolve().parent.parent / "mace" / "utils" / "ion_fragments.py")
        frag = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(frag)

from charge_records import _basis_insert_index, extract_charge_records
from d12_writer import DEFAULT_SPINLOCK_CYCLES

IonChargeError = frag.IonChargeError

# Neutral shell charges of CRYSTAL23's internal basis libraries, in the order
# CRYSTAL lists the shells, with the shell types (S, L = SP, P, D, F).
# Measured on HPCC (CRYSTAL/23-intel-2023a, 2026-10-01): the shell types from
# the "LOCAL ATOMIC FUNCTIONS BASIS SET" listing, the occupations from the
# atomic-guess blocks ("NUMBER OF CLOSED SHELLS" / "OPEN SHELL OCCUPATION" per
# symmetry species) filled in listing order. That assignment reproduces the
# CHE values of the explicit full.basis.triplezeta files (= POB-TZVP-REV2) and
# of an explicit SOLDEF2MSVP deck, and the group's accepted ion vectors (Li+
# "2.0 0.0 0.0 0.0 0.0", FSI- O "2.0 2.0 0.0 0.0 4.25 0.0 0.0"). SOLDEF2MSVP
# H C N O F S Li were first read from the corpus outputs the same way.
SHELL_TABLE = {
    ("SOLDEF2MSVP", "Li"): ((2, 1, 0, 0, 0), "SSSPP"),
    ("SOLDEF2MSVP", "Na"): ((2, 2, 1, 0, 6, 0, 0), "SSSSPPD"),
    ("SOLDEF2MSVP", "K"): ((2, 2, 2, 1, 0, 6, 6, 0, 0, 0), "SSSSSPPPDD"),
    ("SOLDEF2MSVP", "Mg"): ((2, 2, 2, 0, 6, 0, 0, 0), "SSSSPPPD"),
    ("SOLDEF2MSVP", "Ca"): ((2, 2, 2, 2, 0, 6, 6, 0, 0, 0), "SSSSSPPPDD"),
    ("SOLDEF2MSVP", "Zn"): ((2, 2, 2, 2, 0, 6, 6, 10, 0, 0, 0), "SSSSSPPDDPF"),
    ("SOLDEF2MSVP", "O"): ((2, 2, 0, 0, 4, 0, 0), "SSSSPPD"),
    ("SOLDEF2MSVP", "F"): ((2, 2, 0, 0, 5, 0, 0), "SSSSPPD"),
    ("SOLDEF2MSVP", "N"): ((2, 2, 0, 0, 3, 0, 0), "SSSSPPD"),
    ("SOLDEF2MSVP", "C"): ((2, 2, 0, 0, 2, 0, 0), "SSSSPPD"),
    ("SOLDEF2MSVP", "H"): ((1, 0), "SS"),
    ("SOLDEF2MSVP", "S"): ((2, 2, 2, 0, 6, 4, 0, 0), "SSSSPPPD"),
    ("SOLDEF2MSVP", "Cl"): ((2, 2, 2, 0, 6, 5, 0, 0), "SSSSPPPD"),
    ("SOLDEF2MSVP", "P"): ((2, 2, 2, 0, 6, 3, 0, 0), "SSSSPPPD"),
    ("SOLDEF2MSVP", "B"): ((2, 2, 0, 0, 1, 0, 0), "SSSSPPD"),
    ("POB-TZVP-REV2", "Li"): ((2, 1, 0, 0, 0), "SSSSP"),
    ("POB-TZVP-REV2", "Na"): ((2, 2, 1, 0, 0, 6, 0, 0, 0), "SSSSSPPPD"),
    ("POB-TZVP-REV2", "K"): ((2, 2, 2, 1, 0, 0, 6, 6, 0, 0, 0), "SSSSSSPPPPD"),
    ("POB-TZVP-REV2", "Mg"): ((2, 2, 2, 0, 0, 6, 0, 0, 0), "SSSSSPPPD"),
    ("POB-TZVP-REV2", "Ca"): ((2, 2, 2, 2, 0, 0, 6, 6, 0, 0, 0), "SSSSSSPPPPD"),
    ("POB-TZVP-REV2", "Zn"): ((2, 2, 2, 2, 0, 0, 6, 6, 0, 0, 10, 0, 0, 0), "SSSSSSPPPPDDDF"),
    ("POB-TZVP-REV2", "O"): ((2, 2, 0, 0, 4, 0, 0, 0), "SSSSPPPD"),
    ("POB-TZVP-REV2", "F"): ((2, 2, 0, 0, 5, 0, 0, 0), "SSSSPPPD"),
    ("POB-TZVP-REV2", "N"): ((2, 2, 0, 0, 3, 0, 0, 0), "SSSSPPPD"),
    ("POB-TZVP-REV2", "C"): ((2, 2, 0, 0, 2, 0, 0, 0), "SSSSPPPD"),
    ("POB-TZVP-REV2", "H"): ((1, 0, 0, 0), "SSSP"),
    ("POB-TZVP-REV2", "S"): ((2, 2, 2, 0, 0, 6, 4, 0, 0), "SSSSSPPPP"),
    ("POB-TZVP-REV2", "Cl"): ((2, 2, 2, 0, 0, 6, 5, 0, 0, 0), "SSSSSPPPPD"),
    ("POB-TZVP-REV2", "P"): ((2, 2, 2, 0, 0, 6, 3, 0, 0, 0), "SSSSSPPPPD"),
    ("POB-TZVP-REV2", "B"): ((2, 2, 0, 0, 1, 0, 0, 0), "SSSSPPPD"),
    ("POB-DZVP-REV2", "Li"): ((2, 1, 0, 0), "SSSP"),
    ("POB-DZVP-REV2", "Na"): ((2, 2, 1, 0, 6, 0, 0), "SSSSPPD"),
    ("POB-DZVP-REV2", "K"): ((2, 2, 2, 1, 0, 6, 6, 0, 0), "SSSSSPPPD"),
    ("POB-DZVP-REV2", "Mg"): ((2, 2, 2, 0, 6, 0, 0), "SSSSPPD"),
    ("POB-DZVP-REV2", "Ca"): ((2, 2, 2, 2, 0, 6, 6, 0, 0), "SSSSSPPPD"),
    ("POB-DZVP-REV2", "Zn"): ((2, 2, 2, 2, 0, 6, 6, 0, 10, 0, 0), "SSSSSPPPDDF"),
    ("POB-DZVP-REV2", "O"): ((2, 2, 0, 4, 0, 0), "SSSPPD"),
    ("POB-DZVP-REV2", "F"): ((2, 2, 0, 5, 0, 0), "SSSPPD"),
    ("POB-DZVP-REV2", "N"): ((2, 2, 0, 3, 0, 0), "SSSPPD"),
    ("POB-DZVP-REV2", "C"): ((2, 2, 0, 2, 0, 0), "SSSPPD"),
    ("POB-DZVP-REV2", "H"): ((1, 0, 0), "SSP"),
    ("POB-DZVP-REV2", "S"): ((2, 2, 2, 0, 6, 4, 0, 0), "SSSSPPPD"),
    ("POB-DZVP-REV2", "Cl"): ((2, 2, 2, 0, 6, 5, 0, 0), "SSSSPPPD"),
    ("POB-DZVP-REV2", "P"): ((2, 2, 2, 0, 6, 3, 0, 0), "SSSSPPPD"),
    ("POB-DZVP-REV2", "B"): ((2, 2, 0, 1, 0, 0), "SSSPPD"),
}
SHELL_TABLE_PROVENANCE = ("measured on HPCC (CRYSTAL/23-intel-2023a): LOCAL ATOMIC FUNCTIONS listing "
                          "and atomic-guess shell occupations")

_LAT_TYPE = {0: "S", 1: "L", 2: "P", 3: "D", 4: "F", 5: "G"}
_CAPACITY = {"S": 2, "L": 8, "P": 6, "D": 10, "F": 14, "G": 18}
ECP_KEYWORDS = {"HAYWLC", "HAYWSC", "BARTHE", "DURAND"}


def _norm_basis(name: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(name).upper())


def _table_entry(library: str, element: str):
    key = _norm_basis(library)
    for (lib, el), value in SHELL_TABLE.items():
        if _norm_basis(lib) == key and el == element:
            return value
    return None


# --------------------------------------------------------------- basis text

@dataclass
class AtomBasis:
    """One atom type's basis: shell types ('S','L','P','D','F') and CHE values."""
    label: int
    types: List[str]
    che: List[Decimal]
    ecp: bool = False

    @property
    def valence(self) -> Decimal:
        return sum(self.che, Decimal(0))


def parse_basis_blocks(text: str) -> Dict[int, AtomBasis]:
    """The explicit basis-set blocks of a deck (or of read_basis_file text).

    Reads "Z NSHELL", the pseudopotential header of an ECP atom (INPUT with
    its ZNUC M M0..M4 record and data lines, or a HAYWLC/HAYWSC/BARTHE/DURAND
    keyword), then NSHELL "ITYB LAT NG CHE SCAL" records with NG primitive
    lines when ITYB is 0. Stops at "99 0". When the text is a whole deck, the
    blocks after the geometry's END are read.
    """
    lines = text.splitlines()
    start = 0
    if lines and not re.fullmatch(r"\s*\d+\s+\d+\s*", lines[0]):
        # a deck: blocks begin after the END that closes the geometry
        for i, line in enumerate(lines):
            if line.strip().upper() in ("END", "ENDGEOM") and i + 1 < len(lines) \
                    and re.fullmatch(r"\s*\d+\s+\d+\s*", lines[i + 1]):
                start = i + 1
                break
        else:
            return {}
    blocks: Dict[int, AtomBasis] = {}
    i = start
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) < 2:
            i += 1
            continue
        try:
            z, nshell = int(tok[0]), int(tok[1])
        except ValueError:
            break
        if z == 99 and nshell == 0:
            break
        i += 1
        ecp = False
        if i < len(lines) and lines[i].strip().upper() in ECP_KEYWORDS:
            ecp = True
            i += 1
        elif i < len(lines) and lines[i].strip().upper() == "INPUT":
            ecp = True
            nums = lines[i + 1].split()
            n_data = sum(int(float(x)) for x in nums[1:7])
            i += 2 + n_data
        types, che = [], []
        for _ in range(nshell):
            rec = lines[i].split()
            ityb, lat, ng = int(rec[0]), int(rec[1]), int(float(rec[2]))
            types.append(_LAT_TYPE.get(lat, "?"))
            che.append(Decimal(rec[3]))
            i += 1 + (ng if ityb == 0 else 0)
        blocks[z] = AtomBasis(z, types, che, ecp)
    return blocks


def basis_shells_from_output(text: str) -> Dict[str, List[str]]:
    """{element: shell types} from an .out's "LOCAL ATOMIC FUNCTIONS BASIS SET" listing."""
    lines = text.splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if "LOCAL ATOMIC FUNCTIONS BASIS SET" in l) + 4
    except StopIteration:
        return {}
    found: Dict[str, List[str]] = {}
    cur = None
    for line in lines[i:]:
        if line.startswith(" ****"):
            break
        m = re.match(r"\s*\d+\s+([A-Z]{1,2})\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$", line)
        if m:
            el = frag.element_symbol(m.group(1))
            cur = None if el in found else el
            if cur:
                found[cur] = []
            continue
        m = re.match(r"\s+(?:\d+-)?\s*\d+\s+(S|SP|P|D|F|G)\s*$", line)
        if m and cur:
            found[cur].append("L" if m.group(1) == "SP" else m.group(1))
    return found


# ------------------------------------------------------------------ numbers

def fmt_charge(x) -> str:
    """At least one decimal, at most six: 2 -> '2.0', 4.25 -> '4.25', 13/3 -> '4.333333'."""
    d = Decimal(str(x)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    s = format(d, "f").rstrip("0")
    if s.endswith("."):
        s += "0"
    if s in ("-0.0", "-0"):
        s = "0.0"
    return s


def _dec(x) -> Decimal:
    return Decimal(str(x))


def fmt_n(n) -> str:
    """An electron count for messages: 95, 31.5."""
    n = _dec(n)
    return str(int(n)) if n == n.to_integral_value() else fmt_charge(n)


def fmt_q(q) -> str:
    """A net charge for messages: 0, +1, -2, -0.5."""
    q = _dec(q)
    if q == q.to_integral_value():
        return "0" if q == 0 else f"{int(q):+d}"
    return ("+" if q > 0 else "") + fmt_charge(q)


# ------------------------------------------------------------------ the plan

@dataclass
class ChargeSetup:
    """The records a charge set-up writes, and what to tell the user."""
    chemod: List[Tuple[int, List[Decimal]]] = field(default_factory=list)
    doping: Optional[Decimal] = None
    charged: bool = False
    spin: bool = False                       # SPIN (DFT) needed
    spinlock: Optional[Tuple[int, int]] = None
    net_charge: Decimal = Decimal(0)
    electrons: Optional[Decimal] = None
    fragments: List[dict] = field(default_factory=list)
    report: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def chemod_lines(self) -> List[str]:
        if not self.chemod:
            return []
        return (["CHEMOD", str(len(self.chemod))]
                + [f"{label} " + " ".join(fmt_charge(v) for v in vec) for label, vec in self.chemod])

    def doping_lines(self) -> List[str]:
        if self.doping is None:
            return []
        d = self.doping
        return ["DOPING", str(int(d)) if d == d.to_integral_value() else fmt_charge(d)]


@dataclass
class BasisInfo:
    """Which basis a deck uses: an internal library name, or explicit blocks."""
    internal: Optional[str] = None
    blocks: Dict[int, AtomBasis] = field(default_factory=dict)

    @property
    def explicit(self) -> bool:
        return self.internal is None


@dataclass
class MethodInfo:
    """What the deck's Hamiltonian can carry: DFT (with or without SPIN) or HF."""
    method: str = "DFT"            # DFT or HF
    hf_method: str = "RHF"         # RHF, UHF, HF3C, HFSOL3C (HF only)
    spin: bool = False             # SPIN already in the DFT block / UHF
    spinlock: Optional[Tuple[int, int]] = None   # an existing SPINLOCK record
    ghosts: bool = False


def _atom_shells(i: int, z_label: int, element: str, basis: BasisInfo, out_shells: Dict[str, List[str]]):
    """(neutral CHE vector, types) of atom i for its basis, or raise with the reason."""
    if basis.explicit:
        block = basis.blocks.get(z_label) or basis.blocks.get(z_label % 100)
        if block is None:
            raise IonChargeError(f"atom {i + 1} ({element}, Z={z_label}) has no basis block in the deck")
        return list(block.che), list(block.types)
    entry = _table_entry(basis.internal, element)
    if entry is None:
        raise IonChargeError(
            f"basis {basis.internal} is internal; MACE does not know {element}'s shell occupations "
            f"in it (measured only for {', '.join(sorted({lib for lib, _ in SHELL_TABLE}))} / "
            f"{', '.join(sorted({el for _, el in SHELL_TABLE}))}). Give the atom's shell charges "
            "with site_states (and the parent .out to check the shell count), or use chemod 'no' "
            "to set the charge with DOPING alone")
    vec, types = entry
    return [_dec(v) for v in vec], list(types)


def _valence_index(types: List[str], che: List[Decimal], kind: str) -> Optional[int]:
    """Last occupied s (or sp) shell for kind 's'; last occupied p (or sp) shell for kind 'p'."""
    want = ("S", "L") if kind == "s" else ("P", "L")
    idx = [k for k, t in enumerate(types) if t in want and che[k] > 0]
    return idx[-1] if idx else None


def _check_capacity(i, element, types, vec):
    for k, (t, v) in enumerate(zip(types, vec)):
        cap = _CAPACITY.get(t)
        if v < 0 or (cap is not None and v > cap):
            raise IonChargeError(f"atom {i + 1} ({element}): shell {k + 1} ({t}) would hold "
                                 f"{fmt_charge(v)} electrons (allowed 0-{cap})")


def plan_charge_setup(symbols: Sequence[str], z_labels: Sequence[int], xyz, *, cell=None,
                      periodic_dirs: Sequence[int] = (), symmetric: bool = False,
                      basis: BasisInfo, method: MethodInfo,
                      ion_charges: Optional[str] = None, net_charge=None,
                      site_states: Optional[List[dict]] = None, chemod: str = "auto",
                      fragment_bond_scale: float = 1.15,
                      spinlock: Optional[int] = None, spinlock_cycles: Optional[int] = None,
                      parent_spinlock: Optional[Tuple[int, int]] = None,
                      parent_out_text: Optional[str] = None,
                      interactive: bool = False,
                      ask: Optional[Callable[[str], str]] = None) -> Optional[ChargeSetup]:
    """Work out the records for a charged system (see the module docstring).

    Returns None when nothing was asked for. Raises IonChargeError with the
    reason for anything MACE will not write.
    """
    if not ion_charges and net_charge is None and not site_states:
        return None
    if ion_charges not in (None, "", "auto", "none"):
        raise IonChargeError(f"ion_charges must be 'auto' (got {ion_charges!r})")
    if ion_charges == "none":
        ion_charges = None
    if site_states and net_charge is None:
        raise IonChargeError("site_states set the starting guess only; give the net charge too "
                             "(net_charge), since it is what fixes the electron count")
    if chemod not in ("auto", "yes", "no"):
        raise IonChargeError(f"chemod must be auto, yes or no (got {chemod!r})")

    syms = [frag.element_symbol(s) for s in symbols]
    n = len(syms)
    periodic = bool(periodic_dirs)
    setup = ChargeSetup()
    out_shells = basis_shells_from_output(parent_out_text) if parent_out_text else {}

    # 1. fragments and their formal charges
    q_auto = 0
    mods: Dict[int, List[Decimal]] = {}
    types_of: Dict[int, List[str]] = {}
    if ion_charges == "auto":
        frags = frag.classify(syms, xyz, cell, periodic_dirs, fragment_bond_scale, auto=True)
        setup.fragments = frags
        setup.report += frag.format_fragment_report(frags, syms)
        q_auto = sum(f["charge"] for f in frags)
        if periodic and q_auto and any(f["extended"] for f in frags):
            setup.warnings.append(
                "WARNING: the cell has an extended (framework) fragment, which auto mode leaves "
                "neutral; a charge it carries is not counted - check the net charge")
        for f in frags:
            if not f["ion"]:
                continue
            charge = f["charge"]
            sites = f["sites"]
            if charge > 0:
                for i in sites:
                    vec, types = _atom_shells(i, z_labels[i], syms[i], basis, out_shells)
                    k = _valence_index(types, vec, "s")
                    if k is None or vec[k] < charge:
                        raise IonChargeError(f"atom {i + 1} ({syms[i]}): no valence s shell holding "
                                             f"{charge} electrons in this basis")
                    vec[k] -= charge
                    _check_capacity(i, syms[i], types, vec)
                    mods[i], types_of[i] = vec, types
            else:
                extra = Decimal(-charge)
                share = (extra / len(sites)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
                for n_site, i in enumerate(sites):
                    vec, types = _atom_shells(i, z_labels[i], syms[i], basis, out_shells)
                    k = _valence_index(types, vec, "p")
                    if k is None:
                        raise IonChargeError(f"atom {i + 1} ({syms[i]}): no occupied p shell in this basis")
                    add = share if n_site < len(sites) - 1 else extra - share * (len(sites) - 1)
                    vec[k] += add
                    _check_capacity(i, syms[i], types, vec)
                    mods[i], types_of[i] = vec, types

    # 2. explicit per-site states override auto for the same atom
    for state in site_states or []:
        try:
            i = int(state["atom"]) - 1
            values = [_dec(v) for v in state["shell_charges"]]
        except (KeyError, TypeError, ValueError):
            raise IonChargeError(f"a site_states entry needs 'atom' (1-based) and 'shell_charges': {state!r}")
        if not 0 <= i < n:
            raise IonChargeError(f"site_states atom {i + 1} is not in the structure (1-{n})")
        if basis.explicit or _table_entry(basis.internal, syms[i]):
            vec, types = _atom_shells(i, z_labels[i], syms[i], basis, out_shells)
        elif syms[i] in out_shells:
            types = out_shells[syms[i]]
            vec = None
        else:
            raise IonChargeError(
                f"basis {basis.internal} is internal; MACE does not know {syms[i]}'s shell occupations "
                "and cannot check the site_states shell count - give the parent .out "
                "(its LOCAL ATOMIC FUNCTIONS listing names the shells)")
        if len(values) != len(types):
            raise IonChargeError(f"site_states atom {i + 1} ({syms[i]}): {len(values)} shell charges, but "
                                 f"its basis has {len(types)} shells ({''.join(types)})")
        _check_capacity(i, syms[i], types, values)
        mods[i], types_of[i] = values, types

    # 3. the net charge
    q = _dec(net_charge) if net_charge is not None else Decimal(q_auto)
    setup.net_charge = q
    if net_charge is not None and ion_charges == "auto" and q != q_auto:
        setup.warnings.append(f"WARNING: net_charge {fmt_q(q)} differs from the ions found "
                              f"({q_auto:+d}); the net charge sets the electron count")

    # 4. CHEMOD or not
    explicit_request = chemod == "yes" or bool(site_states)
    write_chemod = bool(mods) and chemod != "no"
    if write_chemod and basis.explicit and periodic and not explicit_request:
        write_chemod = False
        setup.report.append("periodic cell with an explicit basis: the charge is set by DOPING alone "
                            "(CHEMOD would need CHARGED, which adds a background - manual p.80); "
                            "ask for chemod 'yes' to write it")
    if write_chemod and symmetric:
        raise IonChargeError(
            "CHEMOD needs a deck without symmetry: CRYSTAL checks the symmetry with the new "
            "configuration and stops rather than lowering it (manual p.80). Write the structure "
            "in P1 (cif2d12 --symmetry P1) or set the charge with DOPING alone (chemod 'no')")
    if write_chemod:
        setup.chemod = [(i + 1, mods[i]) for i in sorted(mods)]

    # 5. records by case
    if basis.explicit and write_chemod:
        deficit = Decimal(0)
        for i in mods:
            neutral, _ = _atom_shells(i, z_labels[i], syms[i], basis, out_shells)
            deficit += sum(neutral, Decimal(0)) - sum(mods[i], Decimal(0))
        if abs(deficit - q) > Decimal("1e-9"):
            raise IonChargeError(
                f"with an explicit basis CHEMOD sets the electron count (measured), so its shell "
                f"charges must remove exactly the net charge: they remove {fmt_charge(deficit)}, "
                f"the net charge is {fmt_q(q)}")
        if q != 0:
            setup.charged = True
            if periodic:
                setup.warnings.append("WARNING: CHARGED adds a uniform background to the charged cell; "
                                      "do not compare its total energy with other charge states "
                                      "(manual p.80)")
    elif q != 0:
        setup.doping = -q

    # 6. electron count and spin
    if basis.explicit:
        zsum = Decimal(0)
        for z in z_labels:
            block = basis.blocks.get(z) or basis.blocks.get(z % 100)
            zsum += block.valence if block else Decimal(z % 100)
    else:
        # all-electron count; pseudopotential cores hold an even number of
        # electrons, so the parity is that of the valence count
        zsum = Decimal(sum(int(z) % 100 for z in z_labels))
    electrons = zsum - q
    setup.electrons = electrons
    integral = electrons == electrons.to_integral_value()
    odd = integral and int(electrons) % 2 == 1
    setup.report.append(f"net charge {fmt_q(q)}: {fmt_n(electrons)} electrons"
                        + ("" if basis.internal is None else " (all-electron count)")
                        + (" - odd, open shell" if odd else ""))
    if not integral:
        setup.warnings.append("WARNING: a fractional electron count; no spin state is set for it")

    if odd:
        if method.ghosts and setup.doping is not None:
            raise IonChargeError(
                "GHOSTS with DOPING and an odd electron count stops CRYSTAL (\"NALPHA-NBETA NOT "
                "COMPATIBLE WITH NUMBER OF ELECTRONS\": the parity is checked before the doping). "
                "Use an explicit basis with CHEMOD + CHARGED on a far atom instead")
        if method.method == "HF" and method.hf_method != "UHF":
            raise IonChargeError(f"{method.hf_method} is closed shell and cannot hold an odd number "
                                 f"of electrons ({fmt_n(electrons)}); use UHF")
        value, cycles = spinlock, spinlock_cycles
        source = "given"
        if not value and method.spinlock and method.spinlock[0]:
            value, cycles = method.spinlock
            source = "the deck's"
        if not value and parent_spinlock and parent_spinlock[0]:
            value, cycles = parent_spinlock
            source = "the parent's"
        if not value:
            if not interactive:
                raise IonChargeError(
                    f"{fmt_n(electrons)} electrons is odd, so the run needs a spin state "
                    "(SPIN + SPINLOCK); none was given and there is no terminal to ask - set "
                    "spinlock (e.g. 1) and spinlock_cycles")
            ask = ask or input
            reply = ask(f"Odd electron count ({fmt_n(electrons)}): SPINLOCK value nalpha-nbeta [1]: ").strip()
            value = int(reply) if reply else 1
            reply = ask(f"SPINLOCK cycles [{DEFAULT_SPINLOCK_CYCLES}]: ").strip()
            cycles = int(reply) if reply else DEFAULT_SPINLOCK_CYCLES
            source = "answered"
        if int(value) % 2 == 0:
            raise IonChargeError(f"SPINLOCK {value} is even, but {fmt_n(electrons)} electrons "
                                 "need an odd nalpha-nbeta")
        setup.spinlock = (int(value), int(cycles or DEFAULT_SPINLOCK_CYCLES))
        setup.spin = method.method == "DFT"
        setup.report.append(f"SPINLOCK {setup.spinlock[0]} {setup.spinlock[1]} ({source})")
    elif spinlock and integral and int(spinlock) % 2 != int(electrons) % 2:
        raise IonChargeError(f"SPINLOCK {spinlock} does not fit {fmt_n(electrons)} electrons "
                             "(nalpha-nbeta must have the parity of the electron count)")

    # 7. the CHEMOD sums are exact
    for label, vec in setup.chemod:
        target = sum(vec, Decimal(0))
        written = sum((Decimal(fmt_charge(v)) for v in vec), Decimal(0))
        assert abs(written - target) <= Decimal("1e-9"), (label, vec)
    if setup.chemod:
        setup.report.append(f"CHEMOD for {len(setup.chemod)} atom(s)"
                            + (" + CHARGED" if setup.charged else ""))
    if setup.doping is not None:
        setup.report.append(f"DOPING {setup.doping_lines()[1]}")
    return setup


# ------------------------------------------------------------- writing it

def _final_end(lines: List[str]) -> Optional[int]:
    return max((i for i, l in enumerate(lines) if l.strip().upper() == "END"), default=None)


def apply_charge_setup(text: str, setup: ChargeSetup) -> str:
    """The deck with the set-up's records added.

    Refuses a deck that already has CHEMOD, DOPING or CHARGED (strip them
    first, or leave the deck alone). SPIN goes before ENDDFT and SPINLOCK
    before the final END, each only when the deck lacks it.
    """
    have = extract_charge_records(text)
    if any(have.values()):
        names = ", ".join(k for k, v in have.items() if v)
        raise IonChargeError(f"the deck already has {names}; remove it first")
    trailing = text.endswith("\n")
    lines = text.splitlines()
    basis_add = setup.chemod_lines() + (["CHARGED"] if setup.charged else [])
    if basis_add:
        at = _basis_insert_index(lines)
        if at is None:
            raise IonChargeError("no basis-set block (BASISSET name or '99 0') to put CHEMOD after")
        lines[at:at] = basis_add
    upper = [l.strip().upper() for l in lines]
    if setup.spin and "SPIN" not in upper:
        if "ENDDFT" not in upper:
            raise IonChargeError("an odd electron count needs SPIN, but the deck has no DFT block")
        lines.insert(upper.index("ENDDFT"), "SPIN")
    tail = setup.doping_lines()
    if setup.spinlock and "SPINLOCK" not in [l.strip().upper() for l in lines]:
        tail = tail + ["SPINLOCK", f"{setup.spinlock[0]} {setup.spinlock[1]}"]
    if tail:
        end = _final_end(lines)
        if end is None:
            raise IonChargeError("no final END to put DOPING/SPINLOCK before")
        lines[end:end] = tail
    return "\n".join(lines) + ("\n" if trailing else "")


def strip_charge_records(text: str) -> str:
    """The deck without its CHEMOD, CHARGED and DOPING records."""
    lines = text.splitlines()
    out = [lines[0]] if lines else []
    i = 1
    while i < len(lines):
        key = lines[i].strip().upper()
        if key == "CHEMOD" and i + 1 < len(lines):
            try:
                i += 2 + int(lines[i + 1].split()[0])
                continue
            except (ValueError, IndexError):
                pass
        if key == "CHARGED":
            i += 1
            continue
        if key == "DOPING" and i + 1 < len(lines):
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def symmetric_from_deck_group(dimensionality: str, group) -> bool:
    """True unless the deck's space/layer/rod/point group is 1."""
    try:
        return int(str(group).split()[0]) != 1
    except (ValueError, IndexError):
        return True


def parse_site_states(value) -> Optional[List[dict]]:
    """site_states from a JSON string, a file path, or an already-parsed list."""
    if value is None or isinstance(value, list):
        return value
    import json
    import os
    if isinstance(value, str) and os.path.exists(value):
        with open(value) as f:
            value = f.read()
    try:
        data = json.loads(value)
    except (TypeError, ValueError) as e:
        raise IonChargeError(f"site_states is not valid JSON: {e}")
    if not isinstance(data, list):
        raise IonChargeError("site_states must be a list of {\"atom\": i, \"shell_charges\": [...]}")
    return data
