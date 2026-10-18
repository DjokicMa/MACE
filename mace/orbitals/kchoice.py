"""Which k points an ORBITALS step writes: the NEWK mesh of its deck.

CRYSTAL's ORBITALS writes one Molden file per k point of the NEWK mesh that
precedes it (two for a complex point), and NEWK takes a whole Monkhorst-Pack
mesh, not single points. So the choice is the smallest shrinking factor IS
whose mesh holds the k points wanted:

* ``gamma``: IS 1, the Γ point only (one file).
* ``band_edges`` (the default): Γ plus the real (time-reversal invariant)
  k points where the band edges of the parent SCF are. IS 1 when both edges
  are at Γ, else IS 2, whose points are Γ and the zone-boundary points (all of
  them real). An edge at a point IS 2 does not hold (Si: the CBM near X) is
  reported in a note; the nearest real points are written.
* ``all_real``: IS 2, every real k point of the mesh.
* ``custom``: an IS given by hand. IS 3 and up has complex points, which are
  usable in 1D but not in 2D/3D (their coefficients fail the orthonormality
  check, spec sec. 2.4), so a 2D/3D custom IS above 2 is refused unless asked
  for explicitly.

The band edges are the "TOP OF VALENCE BANDS" / "BOTTOM OF VIRTUAL BANDS"
lines of the last SCF cycle (alpha and beta blocks for a spin run); their K
is an index into the SCF's "K POINTS COORDINATES (OBLIQUE COORDINATES IN UNITS
OF IS = n)" list, whose entries read ``13-R(  3  3  0)``.

ISP (the Gilat net, used for the Fermi level) is the parent's: CRYSTAL rounds
it to a multiple of IS itself.

Standard library only (``mace.defects.outputs`` is too), so the deck writer
can load this module by path when it runs as a standalone script.
"""

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

try:
    from mace.defects.outputs import band_edges as _band_edges, _TOP, _BOTTOM
except ImportError:  # pragma: no cover - loaded by path, without the mace package
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path
    _spec = _ilu.spec_from_file_location(
        "_mace_defects_outputs", _Path(__file__).resolve().parent.parent / "defects" / "outputs.py")
    _outputs = _ilu.module_from_spec(_spec)
    _sys.modules[_spec.name] = _outputs     # its dataclasses look their module up
    _spec.loader.exec_module(_outputs)
    _band_edges, _TOP, _BOTTOM = _outputs.band_edges, _outputs._TOP, _outputs._BOTTOM

MODES = ("band_edges", "gamma", "all_real", "custom")
MODE_LABELS = {
    "band_edges": "Γ and the real k points of the band edges (default)",
    "gamma": "Γ only (IS 1, one file)",
    "all_real": "all real k points (IS 2)",
    "custom": "a shrinking factor IS of your choice",
}

# Thresholds of the file count/size warning (HPCC: 1M files per user).
MANY_FILES = 100
BIG_TOTAL_BYTES = 1 << 30
# Measured: Si (nbf 36) 33 KB per file, MgO slab (58) 81 KB, Fe (31 per spin) 48 KB.
BYTES_PER_COEFFICIENT = 22

# Measured on HPCC (spec sec. 6): in a two-component run NEWK segfaults for a
# molecule and for any k net but the SCF's, and on the SCF net ORBITALS writes
# a header and an empty [MO] block, then dies reading fort.8. CRYSTAL never
# refuses it cleanly, so MACE does, for good (not a placeholder).
SOC_REFUSAL = ("ORBITALS is not available for two-component (TWOCOMPON/SOC) runs: CRYSTAL23 "
               "cannot export two-component spinors (NEWK and ORBITALS crash in 2c). For orbital "
               "pictures, run the same system one-component (without TWOCOMPON) with the same "
               "basis set: its ORBITALS files are valid.")

WANNIER_ROUTE = ("Wannier functions are not written through ORBITALS (ILOC 1: the files are "
                 "not usable, spec sec. 2.4); use the Wannier90 / lcao2wannier route, which "
                 "writes _WAN.CUBE files")
COMPLEX_K_REFUSAL = ("complex k points are not usable in {dim}D: CRYSTAL's 2D/3D complex-k "
                     "Molden coefficients fail the orthonormality check (and are all zeros for "
                     "some systems). Use IS 1 (Γ) or IS 2 (all real k points)")

_KLIST_HEAD = re.compile(r"K POINTS COORDINATES \(OBLIQUE COORDINATES IN UNITS OF IS =\s*(\d+)\)")
_KENTRY = re.compile(r"(\d+)-([RC])\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\)")
_MONKH = re.compile(r"SHRINK\.\s*FACT\.\(MONKH\.\)\s*(\d+)\s+(\d+)\s+(\d+)")
_GILAT = re.compile(r"SHRINKING FACTOR\(GILAT NET\)\s*(\d+)")
_CHANNEL = re.compile(r"^\s*(ALPHA|BETA)\s+ELECTRONS\s*$")
# printed in a cycle with no band gap: a metal, or a locked spin
_NO_GAP = re.compile(r"CONDUCTING STATE|NO ENERGY GAP COMPUTED")


class KChoiceError(ValueError):
    """The k choice asked for cannot be written; the message says why."""


@dataclass(frozen=True)
class KPoint:
    """A k point of the SCF's list: index, R(eal)/C(omplex), oblique coordinates
    in units of the SCF's IS."""
    index: int
    kind: str
    oblique: Tuple[int, int, int]
    is_scf: int

    @property
    def frac(self) -> Tuple[Fraction, Fraction, Fraction]:
        """Fractional coordinates (of the reciprocal lattice vectors)."""
        return tuple(Fraction(n, self.is_scf) for n in self.oblique)

    @property
    def is_gamma(self) -> bool:
        return all(n == 0 for n in self.oblique)

    @property
    def is_real(self) -> bool:
        return self.kind == "R"

    def in_mesh(self, is_: int) -> bool:
        """Whether the NEWK mesh of shrinking factor ``is_`` holds this point."""
        return all((n * is_) % self.is_scf == 0 for n in self.oblique)

    def label(self) -> str:
        return f"{self.index}-{self.kind}({' '.join(str(n) for n in self.oblique)})/{self.is_scf}"


@dataclass
class BandEdge:
    """A band edge of the last SCF cycle."""
    edge: str                 # "VBM" or "CBM"
    band: int
    k_index: int
    energy_ha: float
    channel: Optional[str] = None     # "alpha"/"beta" for a spin run
    kpoint: Optional[KPoint] = None   # None when the k list is not in the output

    @property
    def oblique(self) -> Optional[Tuple[int, int, int]]:
        return self.kpoint.oblique if self.kpoint else None

    @property
    def frac(self) -> Optional[Tuple[Fraction, Fraction, Fraction]]:
        return self.kpoint.frac if self.kpoint else None

    @property
    def kind(self) -> Optional[str]:
        """"R" (real) or "C" (complex)."""
        return self.kpoint.kind if self.kpoint else None

    def describe(self) -> str:
        who = f"{self.channel} " if self.channel else ""
        where = (f"K {self.k_index} = {self.kpoint.label()}" if self.kpoint
                 else f"K {self.k_index}")
        return f"{who}{self.edge} band {self.band} at {where}"


@dataclass
class KChoice:
    """The NEWK mesh of an ORBITALS deck and why it was chosen."""
    is_: Optional[int]         # None for a molecule: NEWK has no mesh line
    isp: Optional[int]
    mode: str                  # band_edges / gamma / all_real / custom / molecule
    dimensionality: int
    edges: List[BandEdge] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def newk_lines(self) -> List[str]:
        """The record(s) between NEWK and "IFE NPR": "IS ISP", none for 0D."""
        if self.is_ is None:
            return []
        return [f"{self.is_} {self.isp or self.is_}"]

    @property
    def has_complex_points(self) -> bool:
        return self.is_ is not None and has_complex_points(self.is_, self.dimensionality)


@dataclass
class FileEstimate:
    """How many Molden files an ORBITALS run writes and how big they are."""
    n_files: int
    n_real: int
    n_complex: int
    exact: bool                        # False: an upper bound (symmetry not applied)
    bytes_per_file: Optional[int]      # None when the basis size is not known
    total_bytes: Optional[int]
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ parsing
def dimensionality_of(text: str) -> Optional[int]:
    """0-3 from a CRYSTAL .out banner, else from a .d12's geometry keyword, else None."""
    for word, dim in (("MOLECULAR CALCULATION", 0), ("POLYMER CALCULATION", 1),
                      ("SLAB CALCULATION", 2), ("CRYSTAL CALCULATION", 3)):
        if re.search(rf"^\s*{word}\s*$", text, re.MULTILINE):
            return dim
    for line in text.splitlines()[1:12]:
        word = line.strip().upper()
        if word in ("MOLECULE", "POLYMER", "SLAB", "CRYSTAL", "EXTERNAL"):
            return {"MOLECULE": 0, "POLYMER": 1, "SLAB": 2}.get(word, 3)
    return None


def parse_kpoints(text: str) -> Tuple[Optional[int], Dict[int, KPoint]]:
    """The SCF's IBZ k list (the last one printed): (IS, {index: KPoint})."""
    heads = list(_KLIST_HEAD.finditer(text))
    if not heads:
        return None, {}
    head = heads[-1]
    is_scf = int(head.group(1))
    points: Dict[int, KPoint] = {}
    for line in text[head.end():].splitlines()[1:]:
        found = _KENTRY.findall(line)
        if not found:
            if points:
                break
            continue
        for idx, kind, a, b, c in found:
            points[int(idx)] = KPoint(int(idx), kind, (int(a), int(b), int(c)), is_scf)
    return is_scf, points


def _last_edge_block(lines: List[str]) -> Optional[Tuple[int, int]]:
    """Start and end line of the last cycle's band-edge block (the lines
    between two timing lines that hold the last TOP line), None when the last
    cycles print no gap (a metal) or there is no TOP line."""
    last = None
    for i, line in enumerate(lines):
        if _TOP.search(line):
            last = i
    if last is None:
        return None
    if any(_NO_GAP.search(line) for line in lines[last + 1:]):
        return None
    start = last
    while start > 0 and "TTTTTTTTTT" not in lines[start - 1]:
        start -= 1
    end = last
    while end + 1 < len(lines) and "TTTTTTTTTT" not in lines[end + 1]:
        end += 1
    return start, end


def parse_band_edges(text: str) -> List[BandEdge]:
    """The VBM and CBM of the last SCF cycle, one pair per spin channel.

    A closed-shell run gives one pair (``mace.defects.outputs.band_edges``:
    the highest TOP and the lowest BOTTOM line of the cycle); a spin run one
    pair for the ALPHA and one for the BETA block. Empty for a metal, a
    locked spin with no gap, or a molecule-free output without edge lines.
    """
    lines = text.splitlines()
    block = _last_edge_block(lines)
    if block is None:
        return []
    _, points = parse_kpoints(text)
    start, end = block
    channels: Dict[Optional[str], Dict[str, BandEdge]] = {}
    channel: Optional[str] = None
    for line in lines[start:end + 1]:
        m = _CHANNEL.match(line)
        if m:
            channel = m.group(1).lower()
            continue
        for edge, rx, better in (("VBM", _TOP, lambda e, o: e > o), ("CBM", _BOTTOM, lambda e, o: e < o)):
            t = rx.search(line)
            if not t:
                continue
            e = float(t.group(3))
            have = channels.setdefault(channel, {}).get(edge)
            if have is None or better(e, have.energy_ha):
                k = int(t.group(2))
                channels[channel][edge] = BandEdge(edge, int(t.group(1)), k, e, channel, points.get(k))
    if set(channels) == {None}:
        # closed shell: the defect code's reading, which this must agree with
        be = _band_edges(text)
        out = []
        if be.vbm is not None:
            out.append(BandEdge("VBM", be.vbm_band, be.vbm_k, be.vbm, None, points.get(be.vbm_k)))
        if be.cbm is not None:
            out.append(BandEdge("CBM", be.cbm_band, be.cbm_k, be.cbm, None, points.get(be.cbm_k)))
        return out
    return [ch[e] for ch in channels.values() for e in ("VBM", "CBM") if e in ch]


def _shrink_record(d12_text: str) -> Optional[List[List[str]]]:
    """The one or two lines after a .d12's SHRINK record, split."""
    lines = d12_text.splitlines()
    for i, line in enumerate(lines[1:], 1):
        if line.strip().upper() == "SHRINK" and i + 1 < len(lines):
            return [lines[j].split() for j in range(i + 1, min(i + 3, len(lines)))]
    return None


def parent_isp(scf_text: str = "", d12_text: Optional[str] = None) -> Optional[int]:
    """The parent's ISP (Gilat net) from its .d12 SHRINK record, else its .out."""
    if d12_text:
        rec = _shrink_record(d12_text)
        try:
            if rec and rec[0]:
                if len(rec[0]) >= 2:
                    return int(rec[0][1])
                return int(rec[0][0])
        except ValueError:
            pass
    g = _GILAT.search(scf_text or "")
    if g:
        return int(g.group(1))
    m = _MONKH.search(scf_text or "")
    if m:
        return max(int(m.group(i)) for i in (1, 2, 3))
    return None


# ------------------------------------------------------------------ choosing
def has_complex_points(is_: int, dimensionality: int) -> bool:
    """Whether a NEWK mesh of shrinking factor ``is_`` has complex k points
    (every point of IS 1 and IS 2 is real)."""
    return dimensionality >= 1 and is_ > 2


def custom_is_problem(is_: int, dimensionality: int) -> Optional[str]:
    """Why a custom IS cannot be used for this dimensionality, or None."""
    if not isinstance(is_, int) or isinstance(is_, bool) or is_ < 1:
        return f"IS must be a whole number of at least 1 (got {is_!r})"
    if dimensionality >= 2 and has_complex_points(is_, dimensionality):
        return f"IS {is_} has complex k points, and " + COMPLEX_K_REFUSAL.format(dim=dimensionality)
    return None


def choose_kmesh(scf_text: str, mode: str = "band_edges", dimensionality: Optional[int] = None,
                 custom_is: Optional[int] = None, d12_text: Optional[str] = None) -> KChoice:
    """The NEWK mesh for an ORBITALS step from its parent SCF output.

    Args:
        scf_text: the parent SCF (or OPT) .out.
        mode: "band_edges" (default), "gamma", "all_real" or "custom".
        dimensionality: 0-3; read from ``scf_text`` (else ``d12_text``) when None.
        custom_is: the IS for mode "custom".
        d12_text: the parent's .d12, for its SHRINK ISP.

    Raises:
        KChoiceError: an unknown mode, a bad custom IS, or complex points in 2D/3D.
    """
    if mode not in MODES:
        raise KChoiceError(f"unknown ORBITALS k mode {mode!r}: one of {', '.join(MODES)}")
    if dimensionality is None:
        dimensionality = dimensionality_of(scf_text)
        if dimensionality is None and d12_text:
            dimensionality = dimensionality_of(d12_text)
        if dimensionality is None:
            dimensionality = 3
    if dimensionality == 0:
        return KChoice(None, None, "molecule", 0, notes=[
            "molecule: NEWK has no k mesh, and ORBITALS writes one file with every molecular orbital"])

    isp_parent = parent_isp(scf_text, d12_text)
    notes: List[str] = []
    edges: List[BandEdge] = []

    if mode == "gamma":
        is_ = 1
    elif mode == "all_real":
        is_ = 2
    elif mode == "custom":
        if custom_is is None:
            raise KChoiceError("mode custom needs an IS")
        try:
            is_ = int(custom_is)
        except (TypeError, ValueError):
            raise KChoiceError(f"IS must be a whole number of at least 1 (got {custom_is!r})")
        if is_ < 1:
            raise KChoiceError(f"IS must be a whole number of at least 1 (got {custom_is!r})")
        problem = custom_is_problem(is_, dimensionality)
        if problem:
            # CRYSTAL23's complex-k data in 2D/3D is incomplete: such a deck only makes
            # files nobody can use, so it is never written
            raise KChoiceError(problem)
        if has_complex_points(is_, dimensionality):
            notes.append(f"IS {is_} has complex k points: in 1D they were validated "
                         f"(Re and Im parts, |ψ|² uniform along the chain).")
    else:  # band_edges
        edges = parse_band_edges(scf_text)
        if not edges:
            is_ = 1
            notes.append("No band edges in the last SCF cycle (a metal, or a locked spin with "
                         "no gap computed): Γ only (IS 1). Choose all real k points (IS 2) "
                         "for the zone-boundary points as well.")
        else:
            located = [e for e in edges if e.kpoint is not None]
            if len(located) < len(edges):
                missing = sorted({e.k_index for e in edges if e.kpoint is None})
                notes.append(f"K {', '.join(map(str, missing))} is not in the k list of the "
                             f"output; IS 2 (every real k point) is used.")
                is_ = 2
            elif all(e.kpoint.is_gamma for e in located):
                is_ = 1
            else:
                is_ = 2
            off = [e for e in located if not e.kpoint.in_mesh(2)]
            if off and is_ == 2:
                where = "; ".join(e.describe() for e in off)
                notes.append(f"{where}: not a point NEWK IS 2 holds (only Γ and the zone-boundary "
                             f"points are real); the nearest real points of the IS 2 mesh are "
                             f"written instead.")

    isp = isp_parent if isp_parent else is_
    return KChoice(is_, isp, mode, dimensionality, edges, notes)


# ------------------------------------------------------------------ files
def estimate_orbital_files(is_: Optional[int], dimensionality: int, nbf: Optional[int] = None,
                           spin: bool = False, scf_text: Optional[str] = None) -> FileEstimate:
    """How many Molden files ORBITALS writes for a NEWK mesh, and their size.

    A periodic run writes one file per real k point of the irreducible zone
    and two per complex one (real and imaginary parts). Without symmetry the
    count is at most IS^dim (every point of the full mesh, time reversal
    pairing the complex ones); with the parent's k list at the same IS it is
    exact. Each file holds every band: about nbf² × 22 bytes (× 2 for spin).
    """
    if is_ is None or dimensionality == 0 or is_ == 1:
        n_real, n_complex, exact = 1, 0, True
    else:
        exact = False
        n_real = n_complex = 0
        if scf_text:
            is_scf, points = parse_kpoints(scf_text)
            if is_scf == is_ and points:
                n_real = sum(1 for p in points.values() if p.is_real)
                n_complex = len(points) - n_real
                exact = True
        if not exact:
            total = is_ ** dimensionality
            n_real = (2 ** dimensionality) if is_ % 2 == 0 else 1
            n_real = min(n_real, total)
            n_complex = (total - n_real) // 2
    n_files = n_real + 2 * n_complex
    per = None if not nbf else (2 if spin else 1) * nbf * nbf * BYTES_PER_COEFFICIENT
    total_bytes = None if per is None else per * n_files
    warnings: List[str] = []
    bound = "" if exact else "up to "
    if n_files > MANY_FILES:
        warnings.append(f"ORBITALS writes {bound}{n_files} Molden files with this mesh (a file per "
                        f"real k point, two per complex one); HPCC has a 1M-file quota.")
    if total_bytes is not None and total_bytes > BIG_TOTAL_BYTES:
        warnings.append(f"ORBITALS writes {bound}{_size(total_bytes)} of Molden files "
                        f"({_size(per)} each: every band, nbf {nbf}).")
    return FileEstimate(n_files, n_real, n_complex, exact, per, total_bytes, warnings)


def _size(n: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"
