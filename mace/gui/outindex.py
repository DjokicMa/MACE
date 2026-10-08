"""A byte-offset index of a CRYSTAL output (spec section 8.4.6, plan B6).

``structures.read_out_frames`` reads the whole file and parses every
geometry block; a 34 MB optimization output takes seconds and holds every
frame in memory. This module scans the file once in binary (1 MB reads) and
records, for every geometry block, the byte range of its lines plus the six
lines before it (where "FINAL OPTIMIZED GEOMETRY" and "OPTIMIZATION - POINT"
appear), the SCF energy that follows it, the OPT END status and the
dimensionality banner.

Nothing here re-implements the block parser: a block is read by handing
only its lines to the same ``seekpath_interface._geometry_blocks`` and
``structures._block_structure`` that ``read_out_frames`` uses, so the
answers are the same (frame count, labels, energies, coordinates; a test
compares them on every example output and a generated 34 MB file).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import structures as S

CHUNK = 1 << 20
CONTEXT = 6                  # _geometry_blocks looks this many lines back for FINAL / OPT POINT
PROBE_LINES = 24             # enough lines after a block start to tell an empty block from a real one
LAST_CAP = 1 << 20           # the last block is read in windows from this size up (it ends at EOF)
_BLOCK_START = re.compile(rb"^\s*PRIMITIVE CELL(?:\s+- CENTRING CODE|\s*$)")
_ENERGY = re.compile(rb"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*\d+\)\s+(\S+)")
# CRYSTAL's corrected total (D3 and/or gCP): the energy the optimizer compares for the first point
_CORRECTED = re.compile(rb"TOTAL ENERGY \+ (?:DISP|GCP)(?: \+ (?:DISP|GCP))? \(AU\)\s+(\S+)")
_OPT_END_E = re.compile(rb"OPT END.*?E\(AU\):\s*(\S+)")
# version 2: the FREQ markers and the FREQ section's byte range (plan B5): a FREQ output is found,
# and its modes are read, without reading the whole file
# version 3: opt_energies (the optimizer's reference surface: the first point's
# dispersion/gCP-corrected total, the final frame's OPT END energy)
# version 4: molecules (dimensionality 0: no PRIMITIVE CELL block, Angstrom tables instead) have frames
# (the 1.1.5 pre-releases called either change version 3)
VERSION = 4
_MOL_START = re.compile(rb"^\s*(?:GEOMETRY FOR WAVE FUNCTION|FINAL OPTIMIZED GEOMETRY) - DIMENSIONALITY OF THE SYSTEM\s+0\s*$")


def _freq_markers():
    """mace plotting's own FREQ markers (detect.py), as bytes."""
    try:
        from mace.plotting.detect import _FREQ_SIGNATURE, _FREQ_MODES_MARKER
    except Exception:            # pragma: no cover - the plotting package is part of MACE
        _FREQ_SIGNATURE, _FREQ_MODES_MARKER = ("CALCULATION OF PHONON FREQUENCIES AT THE GAMMA POINT", "NORMAL MODES NORMALIZED")
    return _FREQ_SIGNATURE.encode(), _FREQ_MODES_MARKER.encode()


@dataclass
class Block:
    line: int                 # 0-based line of "PRIMITIVE CELL"
    ctx_start: int            # byte offset of the line CONTEXT lines before it
    ctx_lines: int            # how many context lines precede the block start in that range
    start: int                # byte offset of the block start line
    end: int                  # byte offset where the next block starts (or EOF)
    energy: Optional[float] = None
    opt_energy: Optional[float] = None   # the optimizer's energy surface (see OutIndex.opt_energies)


@dataclass
class OutIndex:
    path: str
    size: int
    mtime_ns: int
    dim: int
    status: Optional[str]
    blocks: List[Block] = field(default_factory=list)      # the frames, in order (empty blocks dropped)
    labels: List[str] = field(default_factory=list)
    freq_start: Optional[int] = None    # byte offset of the FREQ signature line
    freq_modes: Optional[int] = None    # byte offset of the first "NORMAL MODES NORMALIZED" line
    live: bool = False                  # still being written (no terminal marker, recent writes)
    pending_frame: bool = False         # the last geometry block is still being written
    molecule: bool = False              # the frames are molecule tables (dimensionality 0)

    @property
    def freq(self) -> Optional[Tuple[int, int]]:
        """(start, end) of the FREQ section when the output has normal modes, else None."""
        if self.freq_start is None or self.freq_modes is None:
            return None
        return self.freq_start, self.size

    @property
    def n_steps(self) -> int:
        return len(self.blocks)

    def energies(self) -> List[Optional[float]]:
        return [b.energy for b in self.blocks]

    def opt_energies(self) -> List[Optional[float]]:
        """The energies on the surface CRYSTAL's optimizer works on. The
        first point prints the bare SCF energy as ``TOTAL ENERGY(DFT)`` and
        the dispersion/gCP-corrected value on a later ``TOTAL ENERGY + DISP
        (+ GCP) (AU)`` line; from the second point on, ``TOTAL ENERGY(DFT)``
        already includes the correction (its ``DE (AU)`` matches only the
        corrected first value). The final block (after OPT END) takes the
        OPT END energy. ``energies()`` is unchanged."""
        return [b.opt_energy if b.opt_energy is not None else b.energy for b in self.blocks]

    # ------------------------------------------------------------ reading blocks
    def _lines(self, b: Block, fh=None) -> List[str]:
        own = fh is None
        fh = fh or open(self.path, "rb")
        try:
            fh.seek(b.ctx_start)
            total = b.end - b.ctx_start
            if total <= LAST_CAP or b is not self.blocks[-1]:
                raw = fh.read(total)
            else:
                # the last block runs to the end of the file (a FREQ or properties run after it can be most of
                # the file): read until its CARTESIAN COORDINATES table has ended, growing the window
                want = LAST_CAP
                while True:
                    fh.seek(b.ctx_start)
                    raw = fh.read(min(want, total))
                    k = raw.find(b"CARTESIAN COORDINATES - PRIMITIVE CELL")
                    if k >= 0:
                        m = re.search(rb"\n[ \t]*\n", raw[k + 200:])
                        if m:
                            raw = raw[:k + 200 + m.end()]
                            break
                    if want >= total:
                        break
                    want *= 4
        finally:
            if own:
                fh.close()
        return raw.decode("utf-8", "replace").split("\n")

    def block_structure(self, k: int) -> Dict[str, Any]:
        """The k-th frame's {cell, frac, cart, numbers, pbc[, origin]}, parsed
        from that block's lines only by the validated block parser."""
        from seekpath_interface import _geometry_blocks
        b = self.blocks[k]
        lines = self._lines(b)
        if self.dim == 0 and self.molecule:
            mol = [x for x in S.molecule_blocks(lines) if x["line"] == b.ctx_lines]
            s = S._block_structure(mol[0], 0) if mol else None
            if s is None:
                raise ValueError(f"geometry block {k} is not where the index says (the file changed?)")
            return s
        blocks = _geometry_blocks(lines)
        mine = [x for x in blocks if x["line"] == b.ctx_lines]
        if not mine:
            raise ValueError(f"geometry block {k} is not where the index says (the file changed?)")
        s = S._block_structure(mine[0], self.dim)
        if s is None:
            raise ValueError(f"geometry block {k} has no atoms")
        return s

    def frame(self, k: int) -> Dict[str, Any]:
        """{structure, energy_au, label, line} as read_out_frames gives it."""
        k = range(self.n_steps)[k]
        return {"structure": self.block_structure(k), "energy_au": self.blocks[k].energy,
                "label": self.labels[k], "line": self.blocks[k].line + 1}

    def payload(self, k: int, source: Dict[str, Any]) -> Dict[str, Any]:
        out = S.frame_payload(self.frame(k), source)
        out["n_frames"] = self.n_steps
        out["opt_status"] = self.status
        return out

    @property
    def is_opt(self) -> bool:
        return any(lab != "initial" for lab in self.labels)


def _is_real_block(lines: List[str], at: int, dim: int) -> Tuple[bool, bool, bool]:
    """(real, final, opt_point) for the block starting at line ``at`` of
    ``lines`` (the context lines, the start and up to PROBE_LINES after it):
    the validated parser decides, on that window only."""
    from seekpath_interface import _geometry_blocks
    blocks = [x for x in _geometry_blocks(lines) if x["line"] == at]
    if not blocks:
        return False, False, False
    b = blocks[0]
    real = bool(b.get("params")) and bool(b["numbers"])
    return real, bool(b["final"]), bool(b["opt_point"])


class _Scan:
    """The streaming pass's state, kept in the index so ``extend`` resumes
    where the last complete line ended (a live run costs only its new bytes)."""

    def __init__(self):
        self.starts: List[Tuple[int, int, int, int]] = []    # (line, offset, ctx_start, ctx_lines)
        self.energies: List[Tuple[int, float]] = []          # (line, value)
        self.mol_starts: List[Tuple[int, int]] = []          # (line, offset) of molecule geometry tables
        self.corrected: List[Tuple[int, float]] = []         # (line, value) of TOTAL ENERGY + DISP/GCP lines
        self.opt_end_e: Optional[float] = None
        self.status: Optional[str] = None
        self.freq_start: Optional[int] = None
        self.freq_modes: Optional[int] = None
        self.head: List[str] = []
        self.ring: List[int] = []                            # offsets of this line and up to CONTEXT before it
        self.line_no = 0
        self.pos = 0                                         # where the next unread line starts
        self.probes: Dict[int, Tuple[bool, bool, bool]] = {}  # block offset -> (real, final, opt_point)

    def clone(self) -> "_Scan":
        c = _Scan()
        for k, v in self.__dict__.items():
            setattr(c, k, list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
        return c

    def one(self, raw: bytes, freq_sig: bytes, freq_modes_marker: bytes):
        pos = self.pos
        self.ring.append(pos)
        if len(self.ring) > CONTEXT + 1:
            del self.ring[0]
        if self.line_no < 400:
            self.head.append(raw.decode("utf-8", "replace"))
        if b"PRIMITIVE CELL" in raw and _BLOCK_START.match(raw):
            self.starts.append((self.line_no, pos, self.ring[0], len(self.ring) - 1))
        elif b"TOTAL ENERGY(" in raw:
            m = _ENERGY.search(raw)
            if m:
                try:
                    self.energies.append((self.line_no, float(m.group(1))))
                except ValueError:
                    pass
        elif b"TOTAL ENERGY + " in raw:
            m = _CORRECTED.search(raw)
            if m:
                try:
                    self.corrected.append((self.line_no, float(m.group(1))))
                except ValueError:
                    pass
        elif b"OPT END" in raw:
            self.status = raw.decode("utf-8", "replace").strip()
            m = _OPT_END_E.search(raw)
            if m:
                try:
                    self.opt_end_e = float(m.group(1))
                except ValueError:
                    pass
        elif b"DIMENSIONALITY OF THE SYSTEM" in raw and _MOL_START.match(raw):
            self.mol_starts.append((self.line_no, pos))
        elif self.freq_start is None and freq_sig in raw:
            self.freq_start = pos
        elif self.freq_start is not None and self.freq_modes is None and freq_modes_marker in raw:
            self.freq_modes = pos
        self.pos = pos + len(raw) + 1
        self.line_no += 1

    def feed(self, path: Path, end: int, should_stop=None) -> bytes:
        """Complete lines of ``[self.pos, end)``; returns the trailing bytes
        without a newline (a line still being written), not consumed."""
        freq_sig, freq_modes_marker = _freq_markers()
        tail = b""
        left = end - self.pos
        with open(path, "rb") as fh:
            fh.seek(self.pos)
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if should_stop is not None:
                    should_stop()
                if not chunk:
                    break
                left -= len(chunk)
                lines = (tail + chunk).split(b"\n")
                tail = lines.pop()
                for raw in lines:
                    self.one(raw, freq_sig, freq_modes_marker)
        return tail


def _live(path: Path, size: int, mtime_ns: int) -> bool:
    """No terminal marker in the last 64 KB, and written within the live window."""
    try:
        from mace.utils import livefile
    except ImportError:          # pragma: no cover
        return False
    snap = livefile.Snap(size, mtime_ns, 0, 0, "", 0)
    try:
        return livefile.is_live(path, snap, livefile.tail_text(path, snap))
    except OSError:
        return False


def _block_complete(path: Path, ctx_start: int, ctx_lines: int, end: int) -> bool:
    """The last block's atom table is written: as many atom rows as the
    block's "ATOMS IN THE UNIT CELL" says (read from complete lines only, by
    the validated parser)."""
    from seekpath_interface import _geometry_blocks
    with open(path, "rb") as fh:
        fh.seek(ctx_start)
        raw = fh.read(max(0, min(end - ctx_start, 4 << 20)))
    lines = raw.decode("utf-8", "replace").split("\n")
    mine = [b for b in _geometry_blocks(lines) if b["line"] == ctx_lines]
    if not mine or not mine[0].get("n_atoms"):
        return False
    b = mine[0]
    # only complete lines were read: every row present is whole
    return bool(b.get("params")) and len(b["numbers"]) >= b["n_atoms"]


def build(path: Path, should_stop=None, size: Optional[int] = None, live: Optional[bool] = None) -> OutIndex:
    """One streaming pass over the file: block starts (with the byte offset of
    the CONTEXT lines before each), energies, OPT END and the dimensionality
    banner; then each candidate block is probed with the real parser on a
    small window, so the frame list is the one read_out_frames gives.

    Only ``[0, size)`` is read (the size of a stat taken before the read, so
    the cache key and the content agree). A file still being written
    (``live``; by default no terminal marker and written within the live
    window) has its trailing partial line ignored, and its last frame only
    once its coordinate table has ended (``pending_frame`` until then)."""
    path = Path(path)
    st = os.stat(path)
    size = st.st_size if size is None else min(size, st.st_size)
    if live is None:
        live = _live(path, size, st.st_mtime_ns)
    scan = _Scan()
    tail = scan.feed(path, size, should_stop)
    return _assemble(path, scan, tail, size, st.st_mtime_ns, live, should_stop)


def extend(idx: "OutIndex", path: Path, should_stop=None, size: Optional[int] = None,
           live: Optional[bool] = None) -> "OutIndex":
    """``build`` for a file that grew since ``idx``: the scan resumes at the
    last complete line. A replaced file (truncated by a resubmission, a new
    inode, a different head) is built again from the start."""
    path = Path(path)
    st = os.stat(path)
    scan = getattr(idx, "_scan", None)
    old = getattr(idx, "_snap", None)
    try:
        from mace.utils import livefile
        new = livefile.snapshot(path)
        same = old is not None and livefile.same_file(old, new, path)
    except (ImportError, OSError):
        same = False
    size = new.size if size is None else min(size, new.size) if same else size
    if scan is None or not same:
        return build(path, should_stop, size=size, live=live)
    if live is None:
        live = _live(path, size, new.mtime_ns)
    if size == idx.size and live == idx.live:
        return idx
    scan = scan.clone()
    tail = scan.feed(path, size, should_stop)
    return _assemble(path, scan, tail, size, new.mtime_ns, live, should_stop)


def _assemble(path: Path, scan: _Scan, tail: bytes, size: int, mtime_ns: int, live: bool, should_stop=None) -> "OutIndex":
    keep = scan
    if tail and not live:
        # a finished file without a final newline: its last line counts (as before)
        scan = scan.clone()
        freq_sig, freq_modes_marker = _freq_markers()
        scan.one(tail, freq_sig, freq_modes_marker)
    complete_end = scan.pos if not tail or not live else keep.pos
    starts, energies, corrected = scan.starts, scan.energies, scan.corrected
    line_no = scan.line_no
    idx = OutIndex(str(path), size, mtime_ns, S._out_dimensionality(scan.head), scan.status,
                   freq_start=scan.freq_start, freq_modes=scan.freq_modes)
    idx.live = bool(live)
    idx.pending_frame = False
    ei = ci = 0
    with open(path, "rb") as fh:
        for n, (line, off, ctx_start, ctx_lines) in enumerate(starts):
            if should_stop is not None and n % 64 == 0:
                should_stop()
            nxt = starts[n + 1][0] if n + 1 < len(starts) else line_no
            probe = keep.probes.get(off)
            if probe is None:
                fh.seek(ctx_start)
                window = b""
                want = ctx_lines + PROBE_LINES + 1
                limit = complete_end
                while window.count(b"\n") < want and fh.tell() < limit:
                    window += fh.read(min(16384, limit - fh.tell()))
                lines = [x.decode("utf-8", "replace") for x in window.split(b"\n")[:want]]
                probe = _is_real_block(lines, ctx_lines, idx.dim)
                if window.count(b"\n") >= want or not live:
                    keep.probes[off] = probe       # a whole window: its answer never changes
            real, final, opt_point = probe
            while ei < len(energies) and energies[ei][0] < line:
                ei += 1
            energy = energies[ei][1] if ei < len(energies) and energies[ei][0] < nxt else None
            opt_energy = None
            if final and scan.opt_end_e is not None:
                opt_energy = scan.opt_end_e          # the converged geometry: CRYSTAL's OPT END energy
            elif energy is not None:
                # the first corrected total after this block's energy line, before the next block
                while ci < len(corrected) and corrected[ci][0] < energies[ei][0]:
                    ci += 1
                if ci < len(corrected) and corrected[ci][0] < nxt:
                    opt_energy = corrected[ci][1]
            if live and n == len(starts) - 1 and real and not _block_complete(path, ctx_start, ctx_lines, complete_end):
                idx.pending_frame = True           # the job is still writing this block
                continue
            if not real:
                continue
            # the block's own lines end where the next block's context begins
            stop = starts[n + 1][1] if n + 1 < len(starts) else size
            idx.blocks.append(Block(line, ctx_start, ctx_lines, off, stop, energy, opt_energy))
            idx.labels.append("final" if final else "OPT point" if opt_point else "initial")
    if not idx.blocks and scan.mol_starts:
        # a molecule: its Angstrom tables, read from complete lines only while the run writes
        _molecule_frames(idx, path, complete_end if live else size, scan.mol_starts, energies, line_no, should_stop)
    idx._scan = keep
    try:
        # what this index read: the size it stopped at, and the head of those bytes
        from mace.utils import livefile
        st = os.stat(path)
        n = min(livefile.HEAD, size)
        idx._snap = livefile.Snap(size, mtime_ns, int(st.st_ino), int(st.st_dev), livefile._head_hash(path, n), n)
    except (ImportError, OSError):
        idx._snap = None
    return idx


def opt_energies_at(idx: OutIndex) -> Dict[str, float]:
    """``opt_energies()`` where it differs from ``energies()`` (frame -> value):
    the first point of a D3/gCP run and the final frame, so a long run's
    summary does not carry its energies twice (app.js expandSummary fills
    ``opt_energies_au`` from it)."""
    return {str(k): o for k, (e, o) in enumerate(zip(idx.energies(), idx.opt_energies())) if o != e}


def _molecule_frames(idx: OutIndex, path: Path, size: int, mol_starts, energies, n_lines: int, should_stop=None):
    """The frames of a molecule output: each whole Angstrom table (initial and
    final; CRYSTAL prints no per-point geometry for a molecule), read from a
    window that grows until the table has ended. A table cut by the end of the
    file (a run still writing it) is left out."""
    ei = 0
    with open(path, "rb") as fh:
        for n, (line, off) in enumerate(mol_starts):
            if should_stop is not None and n % 16 == 0:
                should_stop()
            want = 1 << 16
            while True:
                fh.seek(off)
                raw = fh.read(min(want, size - off))
                lines = raw.decode("utf-8", "replace").split("\n")
                complete = off + len(raw) >= size
                if not complete:
                    lines = lines[:-1]          # the last line of the window may be cut
                got = [b for b in S.molecule_blocks(lines) if b["line"] == 0]
                if (got and got[0]["end_line"] < len(lines)) or complete or not got:
                    break
                want *= 4
            if not got or not got[0]["complete"]:
                continue
            b = got[0]
            end = off + sum(len(x) + 1 for x in raw.split(b"\n")[:b["end_line"] + 1])
            nxt = mol_starts[n + 1][0] if n + 1 < len(mol_starts) else n_lines
            while ei < len(energies) and energies[ei][0] < line:
                ei += 1
            energy = energies[ei][1] if ei < len(energies) and energies[ei][0] < nxt else None
            idx.blocks.append(Block(line, off, 0, off, min(end, size), energy))
            idx.labels.append("final" if b["final"] else "initial")
    if idx.blocks:
        idx.dim = 0
        idx.molecule = True


def summary(idx: OutIndex, rel_path: str) -> Dict[str, Any]:
    """What the Trajectory document needs at open: every step's energy and
    label, the status, the first and last frames and their comparison. The
    other frames come one at a time (``step``)."""
    if not idx.n_steps:
        from .filekind import NoGeometry
        raise NoGeometry()
    source = {"kind": "out", "path": rel_path}
    first, last = idx.frame(0), idx.frame(-1)
    # compact for long runs (<=300 KB at 8000+ steps): the two frames sent by
    # index, the labels as runs; the page expands them (app.js expandSummary)
    runs: List[List[Any]] = []
    for lab in idx.labels:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])
    out = {"path": rel_path, "opt_status": idx.status, "is_opt": idx.is_opt,
           "dimensionality": S._DIM_NAMES[idx.dim], "n_steps": idx.n_steps,
           "frames_at": {"0": S.frame_payload(first, source), str(idx.n_steps - 1): S.frame_payload(last, source)},
           "energies_au": idx.energies(), "opt_energies_at": opt_energies_at(idx), "label_runs": runs, "summary": True, "index_version": VERSION,
           "live": idx.live, "pending_frame": idx.pending_frame,
           "freq": bool(idx.freq)}
    if idx.n_steps > 1:
        out["comparison"] = S.compare(first["structure"], last["structure"])
    return out
