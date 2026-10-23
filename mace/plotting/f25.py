"""CRYSTAL's fort.25 (``*.f25``) as band structure and DOS data.

A properties run writes its plot data twice: the xmgrace-style ``BAND.DAT`` /
``DOSS.DAT`` files and the fixed-width "Crgra" blocks of ``fort.25`` (CRYSTAL23
manual, Appendix D).  This module reads the second form and turns it into the
first, so the one set of readers that ``mace plotting`` and the GUI already use
(``ipBANDS_V2.read_band_data``, ``ipDOS_V2.read_doss_columns``) draws an f25
exactly as it draws the DAT beside it.

The layout of one block (a path segment of a BAND run, or one projection of a
DOSS run; ``format A3,I1,A4,2I5,3E12.5``)::

    -%-,IHFERM,TYPE,NROW,NCOL,DUM,DK,EF
    EMIN,EMAX                        6E12.5
    I1,I2,I3,J1,J2,J3                6I3 (segment ends, in ISS units)   [BAND]
    ((RDAT(I,J),I=1,NROW),J=1,NCOL)  6E12.5, one run of values over the lines

* BAND: NROW bands by NCOL k points per segment, eigenvalues in hartree, one
  block per path segment and (open shell, IHFERM 1 or 3) all the alpha segments
  followed by all the beta ones.  The k axis is the cumulative distance: the
  first point of a segment lies one DK past the last point of the one before it.
  A phonon BAND block (FREQCALC BANDS) breaks the manual's meaning of DK: there
  it is the segment's LENGTH (bohr^-1), not the step (``phonon_path_k``).
* DOSS: one block per projection (the total last), each NCOL values; the energy
  of point i is ``EMAX + i*DK`` (hartree, absolute), which the DAT file gives as
  ``E - E_F``.  Open shell: all projections for alpha, then for beta.

Phonon runs (FREQCALC with BANDS or PDOS) write the same blocks.  CRYSTAL does
not label the quantity in the block: the numbers are frequencies in cm-1 with
EF = 0.  ``domain_of`` decides from the run's output beside the file (or the
name); ``--f25-domain`` / ``domain=`` overrides it.  A phonon file is converted
into the DAT form with its frequencies in hartree (cm-1 / 219474.63) and a
``# DOMAIN PHONON`` line, so the readers' eV scale holds them on the energy axis
and the plot shows them as frequencies (cm-1, THz, meV).
"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.constants import HARTREE_TO_CM1  # noqa: E402
HARTREE_PER_CM = 1.0 / HARTREE_TO_CM1
DOMAINS = ("auto", "electronic", "phonon")
KINDS = ("BAND", "DOSS", "PDOS", "MAPN")      # PDOS: a phonon DOS (FREQCALC with PDOS), laid out like DOSS
GLOBS = ("*.f25", "fort.25", "*.fort.25")
MAX_VALUES = 40_000_000          # one file's values: 320 MB of floats; a larger file is not read

_HEADER = re.compile(r"^-%-(\d)(\w{4})")
_FLOAT = re.compile(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+(?:[EeDd][-+]?\d+)")


class F25Error(ValueError):
    """The file is not a fort.25 this module can read (what is wrong is in the message)."""


@dataclass
class Block:
    ihferm: int
    kind: str
    nrow: int
    ncol: int
    dum: float
    dk: float
    ef: float
    emin: float = 0.0
    emax: float = 0.0
    ints: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    @property
    def open_shell(self) -> bool:
        return self.ihferm in (1, 3)


# ------------------------------------------------------------------ the file
def is_f25_name(name: str) -> bool:
    low = Path(name).name.lower()
    return low.endswith((".f25", "fort.25"))


def file_kinds(path, first: bool = False) -> List[str]:
    """The types of the blocks in file order (BAND, DOSS, PDOS, MAPN), by their header lines only.
    ``first``: stop at the first header that decides the file (any but a phonon run's leading PDOS)."""
    out: List[str] = []
    seen = 0
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                seen += len(raw)
                if seen > 8_000_000:          # the headers of a plot file come early; a huge map is not scanned whole
                    break
                if raw.startswith(b"-%-"):
                    m = _HEADER.match(raw[:12].decode("ascii", "replace"))
                    if m and m.group(2).upper() in KINDS:
                        out.append(m.group(2).upper())
                        if first and out[-1] != "PDOS":
                            break
    except OSError:
        return []
    return out


def sniff_kind(path, first: bool = False) -> Optional[str]:
    """What the file is for plotting: BAND when it holds band blocks, else DOSS (DOSS or PDOS blocks), else MAPN
    (a fort.25 of a phonon run holds both, the PDOS block first), None for anything else. Header lines only;
    ``first``: up to the first header that decides it (the Explorer's listing, which reads every fort.25)."""
    kinds = set(file_kinds(path, first))
    if "BAND" in kinds:
        return "BAND"
    if kinds & {"DOSS", "PDOS"}:
        return "DOSS"
    return "MAPN" if "MAPN" in kinds else None


def _first_kind(path) -> Optional[str]:
    """BAND, DOSS or MAPN from the first block header (16 bytes read), else None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16).decode("ascii", "replace")
    except OSError:
        return None
    m = _HEADER.match(head)
    return m.group(2).upper() if m and m.group(2).upper() in KINDS else None


def _numbers(line: str) -> List[float]:
    """The values of one record: 12-character fields (they can touch: ``1.2E-03-8.0E-02``), else
    whitespace-separated (an ADDSPACE file)."""
    s = line.rstrip("\r\n").rstrip()
    if not s.strip():
        return []
    try:
        if len(s) % 12 == 0 or len(s) <= 72:
            vals = [float(s[k:k + 12]) for k in range(0, len(s), 12)]
            if len(s) % 12 == 0 or all(abs(v) < 1e300 for v in vals):
                return vals
    except ValueError:
        pass
    return [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT.findall(s)]


def _fixed(s: str, width: int, n: int) -> List[float]:
    out = []
    for k in range(n):
        piece = s[k * width:(k + 1) * width].strip()
        if not piece:
            break
        out.append(float(piece))
    return out


def _ints(line: str) -> List[int]:
    parts = line.split()
    try:
        return [int(x) for x in parts]
    except ValueError:
        pass
    # touching integers (6I3 with negative numbers): 3-character fields
    try:
        return [int(line[k:k + 3]) for k in range(0, len(line.rstrip()), 3)]
    except ValueError:
        return []


def read_blocks(path, kinds: Sequence[str] = ("BAND", "DOSS", "PDOS"), fermi: bool = True) -> List[Block]:
    """Every block of the wanted kinds, in file order.  Raises F25Error for a file that is not a
    fort.25 or stops inside a block (a copy still in progress).  A BAND block whose EF is CRYSTAL's
    "no Fermi level" placeholder (-1.0: the SCF ended with its spins locked) gets the Fermi level
    mace.utils.fermi_level derives from the bands (``fermi=False``: the file's own)."""
    path = Path(path)
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        raise F25Error(f"{path.name}: {exc}") from exc
    blocks: List[Block] = []
    total = 0
    i, n = 0, len(lines)
    seen_header = False
    while i < n:
        m = _HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        seen_header = True
        kind = m.group(2).upper()
        ln = lines[i]
        try:
            nrow, ncol = int(ln[8:13]), int(ln[13:18])
            dum, dk, ef = (_fixed(ln[18:], 12, 3) + [0.0, 0.0, 0.0])[:3]
        except ValueError:
            i += 1
            continue
        if kind not in kinds:
            # a block of another kind (a MAPN grid): skipped by its own size
            i += 1
            continue
        if i + 3 >= n:
            raise F25Error(f"{path.name} stops inside a {kind} block (is the run still writing it?)")
        rec2 = _fixed(lines[i + 1], 12, 2)
        if len(rec2) < 2:
            raise F25Error(f"{path.name}: unreadable second record of a {kind} block (line {i + 2})")
        ints = _ints(lines[i + 2])
        want = nrow * ncol
        total += want
        if want <= 0 or total > MAX_VALUES:
            raise F25Error(f"{path.name}: a {kind} block of {nrow} x {ncol} values is not readable here")
        vals: List[float] = []
        j = i + 3
        while len(vals) < want and j < n:
            if lines[j].startswith("-%-"):
                break
            vals.extend(_numbers(lines[j]))
            j += 1
        if len(vals) < want:
            raise F25Error(f"{path.name} stops inside a {kind} block: {len(vals)} of {want} values "
                           f"(is the run still writing it?)")
        blocks.append(Block(int(m.group(1)), kind, nrow, ncol, dum, dk, ef, rec2[0], rec2[1], ints, vals[:want]))
        i = j
    if not seen_header:
        raise F25Error(f"{path.name} is not a fort.25: no '-%-' block header")
    if fermi and any(b.kind == "BAND" for b in blocks):
        from mace.utils import fermi_level
        if any(b.kind == "BAND" and fermi_level.is_placeholder(b.ef) for b in blocks):
            fl = fermi_level.band_fermi(path)
            if fl.derived:
                for b in blocks:
                    if b.kind == "BAND" and fermi_level.is_placeholder(b.ef):
                        b.ef = fl.value_au
    return blocks


# ------------------------------------------------------------------ electronic or phonon
_PHONON_OUT = re.compile(r"PHONON BANDS|PHONON DISPERSION|CALCULATION OF PHONON FREQUENCIES|FREQCALC|"
                         r"PHONON DENSITY OF STATES|PHONON DOS|PDOS", re.I)
_PHONON_NAME = re.compile(r"(phonon|phon[_-]?(band|dos)|pdos)", re.I)


def domain_of(path, blocks: Optional[List[Block]] = None, domain: str = "auto") -> str:
    """"electronic" or "phonon".  An explicit ``domain`` wins.  Otherwise the output of the run beside the file
    decides (the .out of the same stem or of its material): a CRYSTAL PROPERTIES output is electronic, a
    CRYSTAL output of a FREQCALC run with BANDS or PDOS is phonon.  With no output beside it, a name that
    says phonon/pdos, or blocks with no Fermi energy and frequency-sized values (more than 50 in magnitude:
    no band energy in hartree reaches it), mean phonon."""
    if domain in ("electronic", "phonon"):
        return domain
    if blocks and any(b.kind == "PDOS" for b in blocks):
        return "phonon"                        # only FREQCALC's PDOS writes this block type
    path = Path(path)
    stem = _stem(path)
    base = re.sub(r"[._-](band|doss)$", "", stem, flags=re.I)
    for cand in dict.fromkeys((path.with_name(stem + ".out"), path.with_name(base + ".out"))):
        try:
            if cand.is_file():
                with open(cand, "rb") as fh:
                    head = fh.read(400_000).decode("utf-8", "replace")
                if "CRYSTAL - PROPERTIES" in head:
                    return "electronic"
                if _PHONON_OUT.search(head):
                    return "phonon"
        except OSError:
            continue
    if _PHONON_NAME.search(stem):
        return "phonon"
    if blocks:
        b = blocks[0]
        if b.kind == "BAND" and b.ef == 0.0 and max(abs(b.emin), abs(b.emax)) > 50.0:
            return "phonon"
    return "electronic"


def _first_head(path) -> Optional[Block]:
    """The first BAND/DOSS/PDOS block's two header records (no values): enough for ``domain_of``'s last test."""
    try:
        with open(path, "rb") as fh:
            for _ in range(2000):
                raw = fh.readline()
                if not raw:
                    return None
                m = _HEADER.match(raw[:12].decode("ascii", "replace"))
                if not m or m.group(2).upper() not in ("BAND", "DOSS", "PDOS"):
                    continue
                ln = raw.decode("ascii", "replace")
                dum, dk, ef = (_fixed(ln[18:], 12, 3) + [0.0, 0.0, 0.0])[:3]
                rec2 = _fixed(fh.readline().decode("ascii", "replace"), 12, 2) + [0.0, 0.0]
                return Block(int(m.group(1)), m.group(2).upper(), int(ln[8:13]), int(ln[13:18]), dum, dk, ef, rec2[0], rec2[1])
    except (OSError, ValueError):
        return None
    return None


def quick_domain(path, run_output: bool = True) -> str:
    """``domain_of`` from the header lines only (a band + DOS pair is checked by it, and the Explorer's listing
    marks a phonon fort.25 with it): a PDOS block is phonon, else the run's output beside the file
    (``run_output``; the listing leaves it out: one small read per fort.25) or the name, else the first block."""
    if "PDOS" in file_kinds(path, first=True):
        return "phonon"
    head = _first_head(path)
    if run_output:
        return domain_of(path, [head] if head else None)
    if head is not None and head.kind == "BAND" and head.ef == 0.0 and max(abs(head.emin), abs(head.emax)) > 50.0:
        return "phonon"
    return "phonon" if _PHONON_NAME.search(_stem(Path(path))) else "electronic"


def describe(path, domain: str = "auto") -> Dict[str, Any]:
    """What a fort.25 holds, for the explorer and the GUI: {kind (BAND, or DOSS), has (the plot kinds in it),
    domain, nspin, n_runs, n_segments / n_projections}."""
    path = Path(path)
    kind = sniff_kind(path)
    out: Dict[str, Any] = {"kind": kind, "domain": None}
    if kind not in ("BAND", "DOSS"):
        return out
    allb = read_blocks(path)
    out["has"] = [k for k, w in (("band", "BAND"), ("dos", "DOSS")) if any(b.kind == w or (w == "DOSS" and b.kind == "PDOS") for b in allb)]
    out["domain"] = domain_of(path, allb, domain)
    blocks = [b for b in allb if (b.kind == "BAND") == (kind == "BAND") and b.kind != "MAPN"]
    blocks, nspin, nruns = select_run(blocks, domain=out["domain"])
    out["nspin"] = nspin
    out["n_runs"] = nruns
    if kind == "BAND":
        out["n_segments"] = len(blocks) // nspin
    else:
        out["n_projections"] = len(blocks) // nspin
    return out


def _nspin(blocks: List[Block]) -> int:
    if blocks and blocks[0].open_shell and len(blocks) % 2 == 0:
        return 2
    return 1


def _sig(b: Block):
    return (b.kind, b.nrow, b.ncol, round(b.dk, 9), round(b.ef, 7), tuple(b.ints[1:] if b.kind == "DOSS" else b.ints))


def _phonon_runs(blocks: List[Block]) -> List[List[Block]]:
    """A phonon file has no spin: the BAND blocks one after the other are the panels of one path (a new run starts
    where the first panel's segment comes again), and the PDOS blocks (CRYSTAL writes one before each path
    segment, each a snapshot of the same sum) are one DOS, the last of them being the complete one; a PDOS set is
    its projections, the index 1 starting it."""
    if not blocks:
        return []
    if blocks[0].kind == "BAND":
        runs: List[List[Block]] = []
        for b in blocks:
            if not runs or (runs[-1] and b.ints == runs[-1][0].ints and b.ncol == runs[-1][0].ncol and b.dk == runs[-1][0].dk):
                runs.append([])
            runs[-1].append(b)
        return runs
    sets: List[List[Block]] = []
    for b in blocks:
        if not sets or (b.ints and b.ints[0] == 1):
            sets.append([])
        sets[-1].append(b)
    return [sets[-1]]


def runs_of(blocks: List[Block], domain: str = "electronic") -> List[List[Block]]:
    """The runs a fort.25 holds.  CRYSTAL appends to an existing fort.25, so a folder whose properties
    step ran more than once has several runs one after the other (the BAND.DAT is overwritten, the f25
    is not): a run is its alpha blocks followed by its beta ones (open shell).  DOSS: a projection
    index of 1 starts a set; BAND: a set ends where the blocks repeat as the beta set."""
    if not blocks:
        return []
    if domain == "phonon":
        return _phonon_runs(blocks)
    spins = 2 if blocks[0].open_shell else 1
    out: List[List[Block]] = []
    if blocks[0].kind == "DOSS":
        sets: List[List[Block]] = []
        for b in blocks:
            if not sets or (b.ints and b.ints[0] == 1 and len(sets[-1]) > 0):
                sets.append([])
            sets[-1].append(b)
        for k in range(0, len(sets), spins):
            out.append([b for st in sets[k:k + spins] for b in st])
        return out
    sigs = [_sig(b) for b in blocks]
    if spins == 1:
        # a closed-shell file is one run, or the same run written again and again
        n = len(blocks)
        m = next((m for m in range(1, n + 1) if n % m == 0 and sigs == sigs[:m] * (n // m)), n)
        return [blocks[k:k + m] for k in range(0, n, m)]
    rest = blocks
    while rest:
        sg = [_sig(b) for b in rest]
        m = next((m for m in range(1, len(rest) // 2 + 1) if sg[m:2 * m] == sg[:m]), max(1, len(rest) // 2))
        take = min(2 * m, len(rest))
        out.append(rest[:take])
        rest = rest[take:]
    return out


def select_run(blocks: List[Block], run: int = -1, domain: str = "electronic") -> Tuple[List[Block], int, int]:
    """(the blocks of one run, its spin count, how many runs the file holds); ``run`` -1 is the last
    (the one a BAND.DAT/DOSS.DAT of the same folder holds)."""
    runs = runs_of(blocks, domain)
    if not runs:
        return [], 1, 0
    chosen = runs[run if -len(runs) <= run < len(runs) else -1]
    return chosen, (1 if domain == "phonon" else _nspin(chosen)), len(runs)


# ------------------------------------------------------------------ to the DAT form
def _row(values: Sequence[float], width: int = 15, prec: int = 7) -> str:
    return "".join(f"{v:{width}.{prec}E}" for v in values)


def _deck_of(path: Path) -> Optional[str]:
    """The text of the phonon run's deck beside the file (``<stem>.d12``, or the run's name without _band/_doss)."""
    stem = _stem(path)
    base = re.sub(r"[._-](band|doss)$", "", stem, flags=re.I)
    for cand in dict.fromkeys((path.with_name(stem + ".d12"), path.with_name(base + ".d12"))):
        try:
            if cand.is_file() and cand.stat().st_size < 2_000_000:
                text = cand.read_text(errors="replace")
                if re.search(r"^\s*BANDS\s*$", text, re.M):
                    return text
        except OSError:
            continue
    return None


def _iss_of(path: Path) -> Optional[int]:
    """The ISS (shrinking factor the segment ends are integers of) of a phonon BANDS record, from the deck beside
    the file (``BANDS`` then ``ISS NSUB``); None without one."""
    m = re.search(r"^\s*BANDS\s*\n\s*(\d+)\s+\d+", _deck_of(path) or "", re.M)
    return int(m.group(1)) if m else None


#: a k-point label of a deck title's path (GAMMA, X, S_0, Y2, ...)
_TITLE_LABEL = re.compile(r"^(?:GAMMA|[A-Z][A-Za-z0-9_']*)$")


def _title_labels(deck: Optional[str], pts: List[Tuple[int, ...]]) -> Optional[List[str]]:
    """The labels of the path the deck's title names (opt2d12's phonon band title ends ``- M-GAMMA-K``, as
    opt2d3's BAND title does), read by the electronic band plot's own parser (ipBANDS_V2.kpoint_path_labels):
    one per tick, Γ exactly where the segment ends are the origin; else None."""
    if not deck or not pts:
        return None
    title = deck.splitlines()[0].strip() if deck.strip() else ""
    names = title.split(" - ")[-1].strip() if " - " in title else ""
    tokens = [t.strip() for t in re.split(r"[-|]", names)]
    if not names or not all(_TITLE_LABEL.match(t) for t in tokens):
        return None
    plotting = Path(__file__).resolve().parents[2] / "Plotting"
    if str(plotting) not in sys.path:
        sys.path.insert(0, str(plotting))
    import ipBANDS_V2
    labels = ipBANDS_V2.kpoint_path_labels(title)[0] or []
    if len(labels) != len(pts):
        return None
    for lab, p in zip(labels, pts):
        if (lab.split("|")[0] == "Γ") != (not any(p)):
            return None
    return labels


def _phonon_labels(path, segs: List[Block]) -> List[str]:
    """The k-point label of each tick of a phonon path: the first segment's start, then each segment's end. The
    names of the deck's title (as the electronic BAND plot reads its .d3's) when they fit the segment ends, else
    (i1,i2,i3)/ISS (Γ for the origin). PHONBANDS.DAT writes its panel labels as coordinates except that its first
    one repeats the first segment's END, which is not where the path starts."""
    iss = _iss_of(Path(path))
    pts = []
    if segs and len(segs[0].ints) >= 6:
        pts.append(tuple(segs[0].ints[:3]))
        pts += [tuple(b.ints[3:6]) for b in segs if len(b.ints) >= 6]
    named = _title_labels(_deck_of(Path(path)), pts)
    if named:
        return named

    def lab(p):
        if not any(p):
            return "Γ"
        return f"({p[0]},{p[1]},{p[2]})" + (f"/{iss}" if iss else "")
    return [lab(p) for p in pts]


def phonon_path_k(segs: List[Block]) -> Tuple[List[List[float]], List[float]]:
    """(the k distance of every point, per segment; the ticks) of one phonon path, in bohr^-1.

    FREQCALC BANDS writes a BAND block's DK as the segment's length, not the step the manual gives for a
    PROPERTIES run (checked on CRYSTAL23 diamond: 0.937968, 0.468984, 0.663243, 0.812304 bohr^-1 are
    Γ-X, X-W, W-L and L-Γ, i.e. 2π/a, π/a, √2π/a and √3π/a).  Each line is NSUB points end to end (the
    first block's NCOL), so the step is DK / (NSUB - 1); a later block that holds NSUB - 1 points has dropped
    its start (the previous end, which the output lists again), so its first point is one step in."""
    if not segs:
        return [], [0.0]
    nsub = max(b.ncol for b in segs)
    pos: List[List[float]] = []
    ticks = [0.0]
    for i, b in enumerate(segs):
        step = b.dk / (nsub - 1) if nsub > 1 else 0.0
        lead = 1 if i and b.ncol == nsub - 1 else 0
        pos.append([ticks[-1] + step * (lead + j) for j in range(b.ncol)])
        ticks.append(ticks[-1] + b.dk)
    return pos, ticks


def band_dat_text(path, domain: str = "auto", run: int = -1) -> str:
    """The BAND.DAT text of an f25's BAND blocks (xmgrace layout; the energies E - E_F in hartree)."""
    allb = read_blocks(path)
    blocks = [b for b in allb if b.kind == "BAND"]
    if not blocks:
        raise F25Error(f"{Path(path).name} holds no BAND block (it is a {sniff_kind(path) or 'different'} file)")
    dom = domain_of(path, allb, domain)
    blocks, nspin, _nruns = select_run(blocks, run, dom)
    nseg = len(blocks) // nspin
    if nseg * nspin != len(blocks):
        raise F25Error(f"{Path(path).name}: {len(blocks)} BAND blocks do not split into {nspin} spin(s) of whole segments")
    nbnd = blocks[0].nrow
    if any(b.nrow != nbnd for b in blocks):
        raise F25Error(f"{Path(path).name}: the BAND blocks do not hold the same number of bands")
    nkpt = sum(b.ncol for b in blocks[:nseg])
    scale = HARTREE_PER_CM if dom == "phonon" else 1.0
    out = [f"# NKPT {nkpt:5d} NBND {nbnd:5d} NSPIN {nspin:5d}", f"# NPANEL {nseg:5d}"]
    labels = _phonon_labels(path, blocks[:nseg]) if dom == "phonon" else None
    if labels:                    # PHONBANDS.DAT's panel lines: the k point index of each tick and its label
        idx = [1]
        for b in blocks[:nseg]:
            idx.append(idx[-1] + b.ncol - (0 if len(idx) > 1 else 1))
        out += [f"# {i:6d}   {lab}" for i, lab in zip(idx, labels)]
    if dom == "phonon":
        out.append("# DOMAIN PHONON")
    for s in range(nspin):
        seg = blocks[s * nseg:(s + 1) * nseg]
        ef = seg[0].ef
        # the k axis: the cumulative distance, the first point of a panel one of its own steps past the last point of the
        # panel before it (CRYSTAL's BAND.DAT puts every point at this x; the tick of a panel's end is its last point).
        # A phonon path's DK is the panel's length, not its step (phonon_path_k); CRYSTAL's PHONBANDS.DAT steps by
        # the whole DK and is not followed
        if dom == "phonon":
            xs, ticks = phonon_path_k(seg)
        else:
            x = 0.0
            ticks = [0.0]
            xs = []
            for b in seg:
                col: List[float] = []
                for j in range(b.ncol):
                    if xs or col:
                        x += b.dk
                    col.append(x)
                xs.append(col)
                ticks.append(x)
        rows: List[str] = []
        for b, col in zip(seg, xs):
            for j, x in enumerate(col):
                e = b.values[j * b.nrow:(j + 1) * b.nrow]
                rel = [(v - (0.0 if dom == "phonon" else ef)) * scale for v in e]
                rows.append(f"{x:14.7E}" + _row(rel))
        out.append(f'@ WITH G{s}')
        out.append(f'@ G{s} ON')
        out.append('@ TITLE "' + ("PHONON BANDS" if dom == "phonon" else "ALPHA" if s == 0 else "BETA") + '"')
        out.append(f"@ XAXIS TICK SPEC {len(ticks):5d}")
        for k, t in enumerate(ticks):
            out.append(f"@ XAXIS TICK {k:5d}, {t:12.5f}")
            out.append(f'@ XAXIS TICKLABEL {k:5d},"{(labels[k] if labels and k < len(labels) else "A"):<9}"')
        out.extend(rows)
        out.append(f"# EFERMI (HARTREE) {(0.0 if dom == 'phonon' else ef):10.5f}")
    return "\n".join(out) + "\n"


def doss_dat_text(path, domain: str = "auto", run: int = -1) -> str:
    """The DOSS.DAT text of an f25's DOSS blocks: the energy column E - E_F in hartree and a column per
    projection, one spin block after the other."""
    allb = read_blocks(path)
    blocks = [b for b in allb if b.kind in ("DOSS", "PDOS")]
    if not blocks:
        raise F25Error(f"{Path(path).name} holds no DOSS block (it is a {sniff_kind(path) or 'different'} file)")
    dom = domain_of(path, allb, domain)
    blocks, nspin, _nruns = select_run(blocks, run, dom)
    nproj = len(blocks) // nspin
    if nproj * nspin != len(blocks):
        raise F25Error(f"{Path(path).name}: {len(blocks)} DOSS blocks do not split into {nspin} spin(s) of whole projections")
    npts = blocks[0].ncol * blocks[0].nrow
    if any(b.ncol * b.nrow != npts for b in blocks):
        raise F25Error(f"{Path(path).name}: the DOSS blocks do not hold the same number of energy points")
    out = [f"# NEPTS {npts:5d} NPROJ {nproj:5d} NSPIN {nspin:5d}"]
    if dom == "phonon":
        out.append("# DOMAIN PHONON")
    out.append("#")
    for s in range(nspin):
        seg = blocks[s * nproj:(s + 1) * nproj]
        first = seg[0]
        ef = first.ef
        if dom == "phonon":
            e0, scale = first.dum, HARTREE_PER_CM          # PDOS: the header's first number is the start, DK the step (cm-1)
            ef_use = 0.0
        else:
            e0, scale, ef_use = first.emax, 1.0, ef
        for i in range(npts):
            e = (e0 + i * first.dk - ef_use) * scale
            vals = [b.values[i] for b in seg]
            if dom == "phonon":
                # states/cm-1 -> states/hartree, so the readers' states/eV is states per frequency unit
                vals = [v / HARTREE_PER_CM for v in vals]
            out.append(f"{e:14.7E}" + _row(vals))
        out.append(f"# EFERMI (HARTREE) {ef_use:10.5f}")
        if s + 1 < nspin:
            out.append("&")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ files beside it
_SIBLING_SUFFIXES = (".d3", ".out", ".d12")
_SIBLING_MAX = 20 * 1024 * 1024


def _stem(path: Path) -> str:
    low = path.name.lower()
    if low.endswith(".f25"):
        return path.name[:-4]
    if low.endswith(".fort.25"):
        return path.name[:-8]
    return path.stem


def _material(stem: str, tag: str) -> str:
    return re.sub(rf"[._]{tag}$", "", stem, flags=re.I)


def materialize(path, dest, kind: Optional[str] = None, domain: str = "auto", inside=None, run: int = -1) -> Tuple[Path, List[Path]]:
    """Write the DAT form of ``path`` into the folder ``dest`` and copy the small files of the run beside
    it (the .d3 with the k-path or the projections, the .out, the POTC pair), under the DAT's names, so the
    readers find what they look for.  Returns (the DAT file, [the files copied]).

    ``inside``: whether a sibling may be read (the GUI: inside the served folder)."""
    path = Path(path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    kind = kind or sniff_kind(path)
    if kind == "PDOS":
        kind = "DOSS"
    if kind not in ("BAND", "DOSS"):
        raise F25Error(f"{path.name} is a {kind or 'different'} fort.25: only BAND and DOSS files are plotted as bands and DOS")
    inside = inside or (lambda f: True)
    stem = _stem(path)
    tag = "band" if kind == "BAND" else "doss"
    material = _material(stem, tag)
    # the DAT's name: <material>_band.BAND.DAT / <material>_doss.DOSS.DAT (the names MACE's own runs write)
    dat = dest / (f"{material}_{tag}.BAND.DAT" if kind == "BAND" else f"{material}_{tag}.DOSS.DAT")
    dat.write_text(band_dat_text(path, domain, run) if kind == "BAND" else doss_dat_text(path, domain, run))
    copied: List[Path] = []
    for sib in path.parent.iterdir():
        if not (sib.is_file() and sib.suffix in _SIBLING_SUFFIXES and inside(sib) and sib.stat().st_size < _SIBLING_MAX):
            continue
        s_stem = sib.name[: -len(sib.suffix)]
        if s_stem == stem or s_stem == material or s_stem in (f"{material}_band", f"{material}_doss"):
            name = sib.name if s_stem != stem else f"{material}_{tag}{sib.suffix}"
            if not (dest / name).exists():
                shutil.copy2(sib, dest / name)
                copied.append(dest / name)
    if dat_domain(dat) != "phonon":        # a phonon plot has no vacuum scale
        for name in (f"{material}_POTC.POTC.DAT", f"{material}_POTC.out"):
            sib = path.parent / name
            if sib.is_file() and inside(sib) and not (dest / name).exists():
                shutil.copy2(sib, dest / name)
                copied.append(dest / name)
    return dat, copied


def dat_labels(dat_path) -> List[str]:
    """The panel labels of a DAT written by ``materialize`` (its ``#   index   label`` lines), else []."""
    out: List[str] = []
    try:
        with open(dat_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                m = re.match(r"#\s+(\d+)\s+(\S+)\s*$", line)
                if m:
                    out.append(m.group(2))
    except OSError:
        return []
    return out


def dat_domain(dat_path) -> str:
    """"phonon" for a DAT file written by ``materialize`` from a phonon f25 (its ``# DOMAIN PHONON`` line)."""
    try:
        with open(dat_path, "rb") as fh:
            head = fh.read(400).decode("ascii", "replace")
    except OSError:
        return "electronic"
    return "phonon" if "# DOMAIN PHONON" in head else "electronic"


# ------------------------------------------------------------------ the other half of a run, staging
_BAND_DAT = (".BAND.DAT", ".band.band.dat", "_band.dat")
_DOSS_DAT = (".DOSS.DAT", "_doss.dat")


def plot_kind(path, want: Optional[str] = None) -> Optional[str]:
    """"band" or "dos" for a fort.25 that holds BAND or DOSS/PDOS blocks, else None (a MAPN grid, another file).
    A phonon run's file holds both: ``want`` asks for one of them (band, dos); without it, band."""
    kinds = set(file_kinds(path))
    has = {"band": "BAND" in kinds, "dos": bool(kinds & {"DOSS", "PDOS"})}
    if want:
        return want if has.get(want) else None
    return "band" if has["band"] else "dos" if has["dos"] else None


def has_dat_twin(path) -> bool:
    """A BAND.DAT/DOSS.DAT of the same run beside the f25 (the DAT holds the last run's data: discovery
    draws that one and leaves the f25 to be named explicitly)."""
    path = Path(path)
    stem = _stem(path)
    if plot_kind(path) is None:
        return False
    names = []
    if plot_kind(path, "band"):
        m = _material(stem, "band")
        names += [f"{m}_band.BAND.DAT", f"{m}.BAND.DAT", f"{m}_band.dat", f"{m}.band.band.dat"]
    if plot_kind(path, "dos"):
        m = _material(stem, "doss")
        names += [f"{m}_doss.DOSS.DAT", f"{m}.DOSS.DAT", f"{m}_doss.dat"]
    return any((path.parent / n).is_file() for n in names)


def partner_of(path) -> Optional[Path]:
    """The DOSS file of the same run as a BAND f25 (and the reverse): an f25 first, else a DAT."""
    path = Path(path)
    kind = plot_kind(path)
    if kind is None:
        return None
    if plot_kind(path, "band") and plot_kind(path, "dos"):
        return path                      # a phonon run's file holds both halves
    stem = _stem(path)
    material = _material(stem, "band" if kind == "band" else "doss")
    other = "doss" if kind == "band" else "band"
    names = [f"{material}_{other}.f25", f"{material}.{other}.f25"] + (
        [f"{material}_doss.DOSS.DAT", f"{material}.DOSS.DAT", f"{material}_doss.dat"] if kind == "band" else
        [f"{material}_band.BAND.DAT", f"{material}.BAND.DAT", f"{material}_band.dat", f"{material}.band.band.dat"])
    for n in names:
        c = path.parent / n
        if c.is_file():
            if is_f25_name(c.name) and plot_kind(c, "dos" if kind == "band" else "band") is None:
                continue
            return c
    return None


def stage(path, dest, kind: str, domain: str = "auto", inside=None, run: int = -1) -> Path:
    """Put one half of a run into the folder ``dest`` for the readers: an f25 is converted
    (``materialize``), a DAT copied with the small files beside it.  Returns the DAT now in ``dest``."""
    path = Path(path)
    if is_f25_name(path.name):
        return materialize(path, dest, "BAND" if kind == "band" else "DOSS", domain, inside, run)[0]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    inside = inside or (lambda f: True)
    shutil.copy2(path, dest / path.name)
    suffixes = _BAND_DAT if kind == "band" else _DOSS_DAT
    stem = next((path.name[: -len(s)] for s in suffixes if path.name.endswith(s)), path.stem)
    for sib in path.parent.iterdir():
        if (sib.is_file() and sib.name.startswith(stem) and sib.suffix in _SIBLING_SUFFIXES and inside(sib)
                and sib.stat().st_size < _SIBLING_MAX and not (dest / sib.name).exists()):
            shutil.copy2(sib, dest / sib.name)
    tag = "_band" if kind == "band" else "_doss"
    material = stem[: -len(tag)] if stem.endswith(tag) else stem
    for name in (f"{material}_POTC.POTC.DAT", f"{material}_POTC.out"):
        sib = path.parent / name
        if sib.is_file() and inside(sib) and not (dest / name).exists():
            shutil.copy2(sib, dest / name)
    return dest / path.name
