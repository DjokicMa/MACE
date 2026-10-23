"""Structure editor sessions for the GUI (P1 and symmetry-aware edits).

A session holds an ``ase.Atoms`` and its undo/redo history. A slab, polymer or
molecule (1.1.6) keeps its ``pbc``: along a non-periodic direction the cell
vector is only a drawing box, so nothing wraps or has images there, and only
P1 editing applies (spglib's space groups are for 3D cells).
Every operation returns the new structure payload plus an ``index_map`` (old
atom index -> new index, or None when the atom is gone) so the browser can
keep its selection, and an ``op_summary`` sentence.

Symmetry-aware ("locked") mode captures the space group operations of the
structure (spglib, at the session tolerance) when it is switched on. An edit
of one atom is then applied to its whole orbit: adding an atom adds every
symmetry image, deleting or replacing an atom does so for all its
equivalents, and moving an atom moves each equivalent by the image of the
displacement. Moving an atom off a special position would grow its orbit
(4a -> 32f in Fm-3m) and change the composition, so that needs ``confirm``.
Supercell, primitive, symmetrize, a cell change and Clean can change the
operations; in locked mode they re-capture the group afterwards and say so.

The edited structure leaves the GUI only as a CIF (P1, or symmetrized with
spglib), which then goes through the normal cif2d12 path.
"""
from __future__ import annotations

import itertools
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import structures as S

MAX_SESSIONS = 32
MAX_HISTORY = 200
MAX_ATOMS = 5000
# Python's own cost of one history entry (dicts, small arrays' headers),
# measured with tracemalloc (tests/test_gui_editor.py)
ENTRY_OVERHEAD = 1500

# Every state of every session gets a revision from one counter, so a number
# is never reused, not even by a session re-created after it was evicted.
_REVS = itertools.count(1)
_REV_LOCK = threading.Lock()


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NO_LOCK = _NoLock()


def next_rev() -> int:
    with _REV_LOCK:
        return next(_REVS)


def _int_or_none(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class EditError(ValueError):
    """An edit that cannot be done (shown to the user)."""


class StaleRev(Exception):
    """The browser edited an older state than the session holds (HTTP 409)."""

    def __init__(self, rev: int):
        super().__init__("the structure changed since this edit was made")
        self.rev = rev


class Busy(RuntimeError):
    """Another operation (Clean) is running on this session."""


class NeedsConfirm(Exception):
    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details


def _number(element) -> int:
    from ase.data import atomic_numbers
    if isinstance(element, int) or (isinstance(element, str) and element.isdigit()):
        z = int(element)
    else:
        sym = str(element).strip()
        sym = sym[:1].upper() + sym[1:].lower()
        if sym not in atomic_numbers:
            raise EditError(f"unknown element {element!r}")
        z = atomic_numbers[sym]
    if not 1 <= z <= 103:
        raise EditError(f"element {element!r} is not a real atom (ghost and dummy "
                        "atoms cannot be edited or written to a CIF)")
    return z


def check_symprec(value) -> float:
    """A symmetry tolerance in Å, 1e-6 to 0.5. spglib crashes the whole
    process (segmentation fault) on a negative one, so nothing else reaches it."""
    try:
        s = float(value)
    except (TypeError, ValueError):
        raise EditError(f"tolerance {value!r} is not a number")
    if not (np.isfinite(s) and 1e-6 <= s <= 0.5):
        raise EditError("tolerance must be between 1e-6 and 0.5 Å")
    return s


def atoms_from_payload(p: Dict[str, Any]):
    """ase.Atoms from a read_structure() payload. A slab, polymer or molecule
    keeps its pbc and Cartesian positions (its non-periodic cell vectors are the
    drawing box; its origin and the file's own vacuum vectors ride in ``info``)."""
    from ase import Atoms
    numbers = [int(n) % 100 for n in p["numbers"]]   # CRYSTAL ECP numbers (247 = Ag)
    if any(n == 0 for n in numbers):
        raise EditError("the structure holds ghost or dummy atoms, which a CIF cannot carry")
    if all(p["pbc"]):
        return Atoms(numbers=numbers, cell=p["cell"], scaled_positions=p["frac"], pbc=True)
    a = Atoms(numbers=numbers, cell=p["cell"], positions=p["cart"], pbc=[bool(x) for x in p["pbc"]])
    a.info.update(S.nonperiodic_info(p))
    return a


def new_cell(cellpar: Sequence[float], sites: Sequence[Dict[str, Any]],
             spacegroup: Optional[int] = None, setting: int = 1, symprec: float = 1e-3):
    """A cell from lattice parameters; with a space group, ``sites`` is the
    asymmetric unit and ase.spacegroup.crystal expands it."""
    from ase import Atoms
    cellpar = [float(x) for x in cellpar]
    if len(cellpar) != 6 or min(cellpar[:3]) <= 0:
        raise EditError("give a, b, c (> 0) and alpha, beta, gamma")
    numbers = [_number(s["element"]) for s in sites]
    frac = [[float(x) for x in s["frac"]] for s in sites]
    if isinstance(spacegroup, str):
        # a number ("225") or a Hermann-Mauguin symbol ("F m -3 m")
        spacegroup = int(spacegroup) if spacegroup.strip().isdigit() else (spacegroup.strip() or None)
    if spacegroup:
        from ase.spacegroup import crystal
        try:
            return crystal(numbers, basis=frac or None, spacegroup=spacegroup if isinstance(spacegroup, str) else int(spacegroup),
                           setting=int(setting), cellpar=cellpar, onduplicates="error",
                           symprec=max(symprec, 1e-3))
        except Exception as exc:
            raise EditError(f"space group {spacegroup}: {_explain_crystal_error(exc, sites)}")
    from ase.geometry import cellpar_to_cell
    return Atoms(numbers=numbers, cell=cellpar_to_cell(cellpar), scaled_positions=frac or None,
                 pbc=True)


def _explain_crystal_error(exc, sites) -> str:
    import re
    msg = str(exc)
    m = re.search(r"scaled_positions (\d+) and (\d+) are equivalent", msg)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        name = lambda k: f"site {k + 1} ({sites[k]['element']} at {sites[k]['frac']})"  # noqa: E731
        return (f"{name(i)} and {name(j)} generate the same position; remove one of them "
                "or move it")
    if "overlap" in msg.lower() or "duplicate" in msg.lower():
        return f"two sites generate the same position ({msg}); remove one of them or move it"
    return msg


class EditSession:
    def __init__(self, atoms, name: str, symprec: float = 1e-3, label: Optional[str] = None,
                 locked: bool = False, undo_limit: int = MAX_HISTORY, wrap: bool = False):
        self.id = secrets.token_hex(8)
        self.name = name
        self.atoms = atoms.copy()
        self.symprec = check_symprec(symprec)
        self.locked = False
        self.ops = None            # (rotations, translations) while locked
        self.history: List[Dict[str, Any]] = []   # undo stack: {state, meta, nbytes}, oldest first
        self.future: List[Dict[str, Any]] = []    # redo stack: the next state is future[-1]
        self.lock = threading.Lock()
        self.busy = False
        self.undo_limit = max(1, int(undo_limit or MAX_HISTORY))
        self.wrap = bool(wrap)             # wrap moved atoms into the cell (Settings' wrapIntoCell)
        self.history_nbytes = 0
        self.trimmed = False               # oldest steps dropped since the last payload
        self.sticky = None                 # (locked, symprec) chosen without an undo step
        self.last_used = time.monotonic()
        self.ledger = None
        self._wrapped = 0
        self._label_info: Dict[str, Any] = {}
        self._ops_key = 0
        # what the document is beyond its atoms (Phase 4, plan B2): where a built or
        # dropped structure came from, and hints for Setup (a defect's charge)
        self.doc_meta: Dict[str, Any] = {}
        self.proposal: Optional[Dict[str, Any]] = None   # Clean's proposal, until Accept or Discard
        self.rev = next_rev()
        self.meta = {"label": label or f"Opened {name}", "t": time.time(), "n_atoms": len(self.atoms),
                     "rev": self.rev, "merge": None, "count": 1}
        if locked and all(self.atoms.pbc):  # the document's mode: not an undo step (P1 off 3D)
            self.locked = True
            self._capture()
            self.sticky = (True, self.symprec)

    # -- state ---------------------------------------------------------------
    # A history entry is a compact snapshot: the cell, Cartesian positions
    # (restored exactly) and atomic numbers as small numpy arrays, plus any
    # other per-atom arrays ase keeps, not a full ase.Atoms copy.
    def _state(self) -> Dict[str, Any]:
        a = self.atoms
        nums = a.numbers
        small = nums.size == 0 or int(nums.max()) < 256
        extra = {k: v.copy() for k, v in a.arrays.items() if k not in ("numbers", "positions")}
        return {"cell": a.cell.array.copy(), "positions": a.positions.copy(),
                "numbers": nums.astype(np.uint8) if small else nums.copy(), "pbc": a.pbc.copy(),
                "info": dict(a.info) if a.info else None, "extra": extra,
                "locked": self.locked, "ops": self.ops, "symprec": self.symprec}

    def _restore(self, st):
        from ase import Atoms
        a = Atoms(numbers=st["numbers"].astype(int), positions=st["positions"], cell=st["cell"],
                  pbc=st["pbc"])
        if st["info"]:
            a.info.update(st["info"])
        for k, v in st["extra"].items():
            a.new_array(k, v.copy())
        self.atoms = a
        self.locked, self.ops, self.symprec = st["locked"], st["ops"], st["symprec"]

    @staticmethod
    def _nbytes(st, older=None) -> int:
        """What a snapshot costs: its arrays plus a fixed overhead per entry;
        arrays it shares with the entry before it (the locked operations, an
        unchanged per-atom array) are counted once, there."""
        n = ENTRY_OVERHEAD + st["cell"].nbytes + st["positions"].nbytes + st["numbers"].nbytes
        n += st["pbc"].nbytes
        prev_extra = older["extra"] if older else {}
        for k, v in st["extra"].items():
            p = prev_extra.get(k)
            if p is not None and p.shape == v.shape and np.array_equal(p, v):
                st["extra"][k] = p          # share it
            else:
                n += v.nbytes
        if st["ops"] is not None and not (older and older["ops"] is st["ops"]):
            n += st["ops"][0].nbytes + st["ops"][1].nbytes
        return n

    def live_nbytes(self) -> int:
        a = self.atoms
        return ENTRY_OVERHEAD + a.positions.nbytes + a.numbers.nbytes + 9 * 8

    def _entry(self, st, meta, older=None) -> Dict[str, Any]:
        return {"state": st, "meta": meta, "nbytes": self._nbytes(st, older)}

    def _recount(self):
        self.history_nbytes = sum(e["nbytes"] for e in self.history) + sum(e["nbytes"] for e in self.future)

    def _trim(self):
        """Per session: at most undo_limit steps and 64 MB (spec §8.3.5),
        oldest first; then the shared budget (memory.MemoryLedger)."""
        from .memory import SESSION_HISTORY_CAP
        while self.history and (len(self.history) + len(self.future) > self.undo_limit
                                or self.history_nbytes > SESSION_HISTORY_CAP):
            self.history_nbytes -= self.history.pop(0)["nbytes"]
            self.trimmed = True
        led = self.ledger
        if led is None or led._sessions is None:
            return
        # the shared history cap: the ledger trims the other sessions first
        # (least recently used first); it cannot take this one's lock, which
        # this edit holds, so if the total is still over, this session's own
        # oldest steps go, and the cap is never passed, not even for a moment
        led.enforce(record=False)
        while self.history and led._sessions.history_bytes() > led.history_cap():
            self.history_nbytes -= self.history.pop(0)["nbytes"]
            self.trimmed = True
            led.freed()
        led.note()

    def drop_oldest_step(self) -> bool:
        """The ledger frees memory: the oldest undo step, else the farthest redo step."""
        if self.history:
            self.history_nbytes -= self.history.pop(0)["nbytes"]
        elif self.future:
            self.history_nbytes -= self.future.pop(0)["nbytes"]
        else:
            return False
        self.trimmed = True
        return True

    def check_rev(self, base_rev) -> None:
        """An edit made against an older state than this one (another tab, a
        request that overtook another) is refused, never applied out of order."""
        if base_rev is None:
            return
        try:
            b = int(base_rev)
        except (TypeError, ValueError):
            raise EditError(f"base_rev {base_rev!r} is not a revision number")
        if b != self.rev:
            raise StaleRev(self.rev)

    def history_rows(self) -> List[Dict[str, Any]]:
        metas = [e["meta"] for e in self.history] + [self.meta] + [e["meta"] for e in reversed(self.future)]
        return [{"label": m["label"], "t": round(m["t"], 3), "n_atoms": m["n_atoms"], "rev": m["rev"]}
                for m in metas]

    def payload(self, summary: str = "", index_map=None, extra=None, sym=None,
                known_ops_key=None) -> Dict[str, Any]:
        a = self.atoms
        if sym is None:
            sym = self.symmetry()
        p = S._payload(a.cell.array, a.get_scaled_positions(wrap=False), a.positions,
                       a.numbers, tuple(bool(x) for x in a.pbc), {"kind": "edit", "path": self.name},
                       symmetry_data=sym)
        if not all(a.pbc):
            p.update(S.nonperiodic_payload(a))
        p.update({"session": self.id, "name": self.name, "locked": self.locked,
                  "locked_group": self._locked_group(), "symprec": self.symprec,
                  "can_undo": bool(self.history), "can_redo": bool(self.future),
                  "op_summary": summary, "index_map": index_map,
                  "path": f"{self.name} (editing)", "rev": self.rev,
                  "history": self.history_rows(), "position": len(self.history),
                  "label": self.meta["label"], "wrapped": self._wrapped,
                  "history_trimmed": self.trimmed, "ops_key": self._ops_key if self.ops else None})
        if self.doc_meta:
            p["doc_meta"] = self.doc_meta
        self.trimmed = False
        if self.ops is not None and (known_ops_key is None or int(known_ops_key) != self._ops_key):
            p["ops"] = {"rotations": self.ops[0].tolist(), "translations": self.ops[1].tolist()}
        if extra:
            p.update(extra)
        return p

    def symmetry(self) -> Dict[str, Any]:
        """spglib's dataset for the current structure, through the shared
        compute gate (spec section 9.1)."""
        from .compute import GATE
        a = self.atoms
        from .compute import HEAVY_ATOMS
        return GATE.run(None, lambda: S.symmetry(a.cell.array, a.get_scaled_positions(), a.numbers,
                                                 tuple(bool(x) for x in a.pbc), symprec=self.symprec),
                        heavy=len(a) > HEAVY_ATOMS)

    def _locked_group(self):
        return None if not self.ops else len(self.ops[0])

    # -- symmetry helpers ------------------------------------------------------
    def _need_3d(self, what: str) -> None:
        if not all(self.atoms.pbc):
            dim = S._DIM_NAMES[int(sum(bool(x) for x in self.atoms.pbc))].lower()
            raise EditError(f"{what} needs a 3D crystal; this is a {dim} (edit it in P1)")

    def _capture(self):
        import spglib
        self._need_3d("Symmetry-aware editing")
        a = self.atoms
        sym = spglib.get_symmetry((a.cell.array, a.get_scaled_positions(), a.numbers),
                                  symprec=self.symprec)
        if sym is None:
            raise EditError("spglib found no symmetry at this tolerance")
        self.ops = (np.asarray(sym["rotations"]), np.asarray(sym["translations"]))
        self._ops_key = next_rev()

    def _tol(self) -> float:
        return max(2 * self.symprec, 1e-3)

    def _images(self, frac) -> np.ndarray:
        """The distinct images of one fractional position under the locked ops."""
        f = np.asarray(frac, float)
        if not self.locked or not self.ops:
            per = np.asarray(self.atoms.pbc, bool)
            return np.where(per, f % 1.0, f)[None, :]
        rot, tr = self.ops
        pts = (np.einsum("kij,j->ki", rot, f) + tr) % 1.0
        out: List[np.ndarray] = []
        for p in pts:
            if not any(self._dist(p, q) < self._tol() for q in out):
                out.append(p)
        return np.array(out)

    def _dist(self, f1, f2) -> float:
        d = np.asarray(f1, float) - np.asarray(f2, float)
        d -= np.where(np.asarray(self.atoms.pbc, bool), np.round(d), 0.0)
        return float(np.linalg.norm(d @ self.atoms.cell.array))

    def _find(self, frac, number=None) -> Optional[int]:
        sf = self.atoms.get_scaled_positions(wrap=False)
        best, bi = None, None
        for i, f in enumerate(sf):
            if number is not None and self.atoms.numbers[i] != number:
                continue
            d = self._dist(f, frac)
            if best is None or d < best:
                best, bi = d, i
        return bi if best is not None and best < self._tol() else None

    def _orbit(self, i: int) -> List[int]:
        f = self.atoms.get_scaled_positions(wrap=False)[i]
        idx = {i}
        for p in self._images(f):
            j = self._find(p, self.atoms.numbers[i])
            if j is not None:
                idx.add(j)
        return sorted(idx)

    def _expand(self, indices) -> List[int]:
        n = len(self.atoms)
        bad = [i for i in indices if not 0 <= int(i) < n]
        if bad or not indices:
            raise EditError("select atoms first" if not indices else f"no atom {bad[0] + 1}")
        out = set()
        for i in indices:
            out.update(self._orbit(int(i)) if self.locked else [int(i)])
        return sorted(out)

    # -- operations --------------------------------------------------------------
    def apply(self, op: str, args: Dict[str, Any], label: Optional[str] = None,
              merge: Optional[str] = None, merge_into=None, record: bool = True,
              base_rev=None, known_ops_key=None, merge_count: int = 1) -> Dict[str, Any]:
        """Run one edit. ``label`` names it in the History (a Title Case label
        per op by default). ``merge``/``merge_into``: the browser decides that
        a nudge extends the step with that revision (same key, within a second
        of the last one); it merges only if that step is still the newest.
        ``record=False`` (set_locked and set_symprec only) changes the mode
        without an undo step. ``base_rev``: the revision the browser holds.
        ``merge_count``: how many nudges this request carries (the browser
        sums the ones still waiting to be sent), so the label counts nudges."""
        self.check_rev(base_rev)
        self.last_used = time.monotonic()
        if self.busy:
            raise Busy("Clean is running on this structure")
        if op in ("undo", "redo"):
            return self.undo(known_ops_key) if op == "undo" else self.redo(known_ops_key)
        fn = getattr(self, "op_" + op, None)
        if fn is None:
            raise EditError(f"unknown edit {op!r}")
        if not record and op not in ("set_locked", "set_symprec"):
            raise EditError("only set_locked and set_symprec can skip the undo history")
        merging = bool(merge) and op == "move" and merge_into is not None and not self.future \
            and _int_or_none(merge_into) == self.rev and self.meta.get("merge") == merge
        before = self._state()
        self._wrapped = 0
        self._label_info = {}
        try:
            summary, index_map = fn(**args)
        except TypeError as exc:
            self._restore(before)
            raise EditError(f"{op}: {exc}")
        except Exception:
            self._restore(before)
            raise
        if len(self.atoms) > MAX_ATOMS:
            self._restore(before)
            raise EditError(f"more than {MAX_ATOMS} atoms: too big for the editor")
        sym = self.symmetry()
        if op in ("set_locked", "set_symprec") and (not record or self.sticky is not None):
            self.sticky = (self.locked, self.symprec)
        if not record:
            pass                                  # the structure is unchanged: same revision
        elif merging:
            self.meta = dict(self.meta, count=self.meta.get("count", 1) + max(1, int(merge_count)), t=time.time(),
                             n_atoms=len(self.atoms))
            self.meta["label"] = f"{self.meta['base']} (×{self.meta['count']})"
            self.rev = self.meta["rev"] = next_rev()
        else:
            self._commit(before, label or self._label(op, sym), merge if op == "move" else None,
                         max(1, int(merge_count)) if merge and op == "move" else 1)
        return self.payload(summary, index_map, sym=sym, known_ops_key=known_ops_key)

    def _commit(self, before, label: str, merge: Optional[str] = None, count: int = 1):
        """``before`` becomes an undo step; the current state gets a new revision."""
        older = self.history[-1]["state"] if self.history else None
        entry = self._entry(before, self.meta, older)
        # the new step and the trim that pays for it happen under the ledger's
        # lock: two sessions committing at once must not both add their step
        # before either trims, or the shared history cap is passed for a moment
        led = self.ledger
        with (led.lock if led is not None else _NO_LOCK):
            self.history.append(entry)
            self.history_nbytes += entry["nbytes"] - sum(e["nbytes"] for e in self.future)
            self.future.clear()
            self.rev = next_rev()
            self.meta = {"label": f"{label} (×{count})" if count > 1 else label, "base": label, "t": time.time(),
                         "n_atoms": len(self.atoms), "rev": self.rev, "merge": merge, "count": count}
            self._trim()

    def _label(self, op: str, sym: Dict[str, Any]) -> str:
        """The History row for an edit, e.g. "Replace Mg→Ca (4 atoms, orbit 4a)"."""
        info = self._label_info
        n = info.get("n", 0)
        atoms = f"{n} atom{'s' * (n != 1)}"
        orbit = ""
        idx = info.get("idx") or []
        if self.locked and idx and sym.get("available") and len(sym.get("wyckoffs", [])) > idx[0]:
            # the Wyckoff multiplicity of the conventional cell (4a in MgO,
            # whether the cell is conventional, primitive or a supercell)
            mult = (sym.get("multiplicity") or [])
            orbit = f", orbit {mult[idx[0]] if len(mult) > idx[0] else n}{sym['wyckoffs'][idx[0]]}"
        if op == "replace":
            return f"Replace {','.join(info.get('old', []))}→{info.get('new', '')} ({atoms}{orbit})"
        if op == "delete":
            return f"Delete {n} Atom{'s' * (n != 1)}"
        if op == "add":
            return f"Add {info.get('new', '')}" + (f" ({atoms}{orbit})" if n > 1 or orbit else "")
        if op == "move":
            return f"Move {n} Atom{'s' * (n != 1)}"
        return {"set_cell": "Edit Cell", "set_positions": "Set Coordinates",
                "supercell": "Supercell", "primitive": "Primitive Cell", "symmetrize": "Symmetrize",
                "set_symprec": "Symmetry Tolerance", "set_locked": "Symmetry Mode"}.get(op, op.replace("_", " ").title())

    def _go(self, index: int):
        """Move to history row ``index`` (0 = the oldest kept): one request
        for any distance; the rows after it stay redoable until a new edit."""
        rows = len(self.history) + 1 + len(self.future)
        if not 0 <= index < rows:
            raise EditError(f"no history step {index}")
        if index == len(self.history):
            return
        older = self.history[-1]["state"] if self.history else None
        cur = self._entry(self._state(), self.meta, older)
        seq = self.history + [cur] + list(reversed(self.future))
        target = seq[index]
        self.history = seq[:index]
        self.future = list(reversed(seq[index + 1:]))
        self._restore(target["state"])
        self.meta = target["meta"]
        self.rev = self.meta["rev"]
        self._recount()
        self._apply_sticky()

    def _apply_sticky(self):
        """A mode chosen without an undo step stays across undo and redo."""
        if self.sticky is None:
            return
        locked, symprec = self.sticky
        if self.locked == locked and self.symprec == symprec and (self.ops is not None) == locked:
            return
        self.symprec = symprec
        self.locked = locked
        self.ops = None
        if locked:
            try:
                self._capture()
            except EditError:
                self.locked = False

    def jump(self, index: int, base_rev=None, known_ops_key=None):
        self.check_rev(base_rev)
        self.last_used = time.monotonic()
        if self.busy:
            raise Busy("Clean is running on this structure")
        self._go(int(index))
        return self.payload("history", known_ops_key=known_ops_key)

    def undo(self, known_ops_key=None):
        if not self.history:
            raise EditError("nothing to undo")
        self._go(len(self.history) - 1)
        return self.payload("undone", known_ops_key=known_ops_key)

    def redo(self, known_ops_key=None):
        if not self.future:
            raise EditError("nothing to redo")
        self._go(len(self.history) + 1)
        return self.payload("redone", known_ops_key=known_ops_key)

    def _keep_map(self, removed) -> List[Optional[int]]:
        out, k = [], 0
        removed = set(removed)
        for i in range(len(self.atoms)):
            if i in removed:
                out.append(None)
            else:
                out.append(k)
                k += 1
        return out

    def op_add(self, element, frac=None, cart=None):
        from ase import Atom
        z = _number(element)
        if frac is None and cart is None:
            raise EditError("give fractional (frac) or Cartesian (cart) coordinates")
        f = (np.asarray(frac, float) if frac is not None
             else np.linalg.solve(self.atoms.cell.array.T, np.asarray(cart, float)))
        n0 = len(self.atoms)
        pts = self._images(f)
        for p in pts:
            j = self._find(p)
            if j is not None:
                raise EditError(f"an atom ({self.atoms.get_chemical_symbols()[j]}{j + 1}) "
                                "already sits there")
        for p in pts:
            self.atoms.append(Atom(z, p @ self.atoms.cell.array))
        sym = self.atoms.get_chemical_symbols()[-1]
        self._label_info = {"n": len(pts), "new": sym, "idx": list(range(n0, n0 + len(pts)))}
        return (f"added {len(pts)} {sym}" + (" (the whole orbit)" if len(pts) > 1 else ""),
                list(range(n0)))

    def op_delete(self, indices):
        idx = self._expand(indices)
        imap = self._keep_map(idx)
        self._label_info = {"n": len(idx)}
        del self.atoms[idx]
        return f"deleted {len(idx)} atom{'s' * (len(idx) != 1)}", imap

    def op_replace(self, indices, element):
        idx = self._expand(indices)
        z = _number(element)
        syms = self.atoms.get_chemical_symbols()
        old = list(dict.fromkeys(syms[i] for i in idx))
        nums = self.atoms.numbers.copy()
        nums[idx] = z
        self.atoms.numbers = nums
        self._label_info = {"n": len(idx), "old": old, "new": self.atoms.get_chemical_symbols()[idx[0]],
                            "idx": idx}
        return (f"replaced {len(idx)} atom{'s' * (len(idx) != 1)} with "
                f"{self.atoms.get_chemical_symbols()[idx[0]]}"), list(range(len(self.atoms)))

    def _wrap_into_cell(self, sf, moved, wrap):
        """Wrap the moved atoms' fractional coordinates into [0, 1) when asked
        (Settings' wrapIntoCell); counts the atoms that left the cell."""
        if not (self.wrap if wrap is None else wrap):
            return sf
        per = np.asarray(self.atoms.pbc, bool)
        for j in sorted(set(moved)):
            w = sf[j] - np.floor(sf[j] + 1e-12)
            w[np.abs(w - 1.0) < 1e-12] = 0.0
            w = np.where(per, w, sf[j])          # a non-periodic direction never wraps
            if np.any(np.abs(w - sf[j]) > 1e-12):
                self._wrapped += 1
            sf[j] = w
        return sf

    def _project(self, f0, df):
        """The part of a displacement that keeps a site's symmetry: the mean
        of R.df over the site's stabilizer {(R, t): R.f0 + t = f0 mod 1}."""
        rot, tr = self.ops
        stab = [R for R, t in zip(rot, tr) if self._dist(R @ f0 + t, f0) < self._tol()]
        if not stab:
            return df
        return np.mean([R @ df for R in stab], axis=0)

    def op_move(self, indices, delta=None, frac_delta=None, confirm=False, wrap=None, project=False):
        if delta is None and frac_delta is None:
            raise EditError("give a Cartesian (delta) or fractional (frac_delta) displacement")
        cell = self.atoms.cell.array
        df = (np.asarray(frac_delta, float) if frac_delta is not None
              else np.linalg.solve(cell.T, np.asarray(delta, float)))
        if df.shape != (3,) or not np.all(np.isfinite(df)):
            raise EditError("a displacement is three finite numbers")
        indices = [int(i) for i in indices]
        self._expand(indices)
        self._label_info = {"n": len(set(indices))}
        if not self.locked:
            sf = self.atoms.get_scaled_positions(wrap=False)
            for i in indices:
                sf[i] = sf[i] + df
            self.atoms.set_scaled_positions(self._wrap_into_cell(sf, indices, wrap))
            return f"moved {len(indices)} atom{'s' * (len(indices) != 1)}", \
                list(range(len(self.atoms)))
        # locked: move each picked orbit as a whole
        imap = list(range(len(self.atoms)))
        done = set()
        notes = []
        for i in indices:
            if i in done:
                continue
            orbit = self._orbit(i)
            done.update(orbit)
            f0 = self.atoms.get_scaled_positions(wrap=False)[i]
            dfi = self._project(f0, df) if project else df
            new = self._images(f0 + dfi)
            if len(new) != len(orbit):
                if not confirm:
                    raise NeedsConfirm(
                        f"moving {self.atoms.get_chemical_symbols()[i]}{i + 1} off its special "
                        f"position grows its orbit from {len(orbit)} to {len(new)} atoms "
                        "(the composition changes)",
                        {"old_multiplicity": len(orbit), "new_multiplicity": len(new)})
                z = self.atoms.numbers[i]
                removed = self._keep_map(orbit)
                del self.atoms[orbit]
                from ase import Atom
                for p in new:
                    self.atoms.append(Atom(int(z), p @ self.atoms.cell.array))
                imap = removed
                notes.append(f"orbit grew {len(orbit)} -> {len(new)}")
                break   # indices after this one have shifted
            rot, tr = self.ops
            sf = self.atoms.get_scaled_positions(wrap=False)
            moved = []
            for R, t in zip(rot, tr):
                j = self._find(R @ f0 + t, self.atoms.numbers[i])
                if j is not None:
                    sf[j] = R @ (f0 + dfi) + t
                    moved.append(j)
            self.atoms.set_scaled_positions(self._wrap_into_cell(sf, moved, wrap))
        self._label_info = {"n": len(done)}
        return "moved " + ", ".join(
            [f"{len(done)} symmetry-equivalent atoms"] + notes), imap

    def op_set_positions(self, indices, frac=None, cart=None, wrap=None, confirm=False,
                         project=False):
        """Absolute coordinates for the given atoms (one triple each). In P1
        they are set as given; in Sym each picked atom's orbit moves by the
        same displacement, through op_move (so growing an orbit asks first)."""
        if frac is None and cart is None:
            raise EditError("give fractional (frac) or Cartesian (cart) coordinates")
        indices = [int(i) for i in indices]
        self._expand(indices)
        target = np.asarray(frac if frac is not None else cart, float).reshape(-1, 3)
        if len(target) != len(indices) or not np.all(np.isfinite(target)):
            raise EditError("give one finite coordinate triple per atom")
        cell = self.atoms.cell.array
        tf = target if frac is not None else np.linalg.solve(cell.T, target.T).T
        sf = self.atoms.get_scaled_positions(wrap=False)
        if not self.locked:
            for k, i in enumerate(indices):
                sf[i] = tf[k]
            self.atoms.set_scaled_positions(self._wrap_into_cell(sf, indices, wrap))
            self._label_info = {"n": len(set(indices))}
            return (f"set the coordinates of {len(indices)} atom{'s' * (len(indices) != 1)}",
                    list(range(len(self.atoms))))
        done, imap, n = set(), list(range(len(self.atoms))), 0
        for k, i in enumerate(indices):
            if i in done:
                continue
            orbit = self._orbit(i)
            done.update(orbit)
            d = tf[k] - self.atoms.get_scaled_positions(wrap=False)[i]
            n0 = len(self.atoms)
            _, imap = self.op_move([i], frac_delta=d, confirm=confirm, wrap=wrap, project=project)
            n += len(orbit)
            if len(self.atoms) != n0 or imap != list(range(n0)):
                break      # an orbit grew: the indices after it have shifted
        self._label_info = {"n": n}
        return f"set the coordinates of {n} symmetry-equivalent atoms", imap

    def _set_cell(self, cell, scale_atoms):
        """set_cell; a slab, polymer or molecule keeps its atoms' heights along its non-periodic
        directions (that vector is only the drawing box: scaling it must not stretch the atoms)."""
        a = self.atoms
        per = np.asarray(a.pbc, bool)
        if not scale_atoms or per.all():
            a.set_cell(cell, scale_atoms=bool(scale_atoms))
            return
        old_c, old_p = a.cell.array.copy(), a.positions.copy()
        f = a.get_scaled_positions(wrap=False)
        new_c = np.asarray(cell, float)
        rec_old, rec_new = np.linalg.inv(old_c).T, np.linalg.inv(new_c).T
        pos = f[:, per] @ new_c[per] if per.any() else np.zeros_like(old_p)
        for k in np.nonzero(~per)[0]:
            nu_o = rec_old[k] / np.linalg.norm(rec_old[k])
            nu_n = rec_new[k] / np.linalg.norm(rec_new[k])
            fk = (old_p @ nu_o - pos @ nu_n) / float(new_c[k] @ nu_n)
            pos = pos + np.outer(fk, new_c[k])
        a.set_cell(new_c, scale_atoms=False)
        a.positions = pos

    def op_set_cell(self, cellpar=None, scale_atoms=True, matrix=None):
        from ase.geometry import cellpar_to_cell
        if matrix is not None:
            m = np.asarray(matrix, float)
            if m.shape != (3, 3) or not np.all(np.isfinite(m)) or abs(np.linalg.det(m)) < 1e-6:
                raise EditError("a cell matrix is three linearly independent vectors")
            self._set_cell(m, scale_atoms)
            return self._recaptured(
                "cell changed, atoms kept at their fractional positions" if scale_atoms else
                "cell changed, atoms kept at their Cartesian positions"), list(range(len(self.atoms)))
        if cellpar is None:
            raise EditError("give a, b, c (> 0) and alpha, beta, gamma")
        cp = [float(x) for x in cellpar]
        if len(cp) != 6 or min(cp[:3]) <= 0:
            raise EditError("give a, b, c (> 0) and alpha, beta, gamma")
        self._set_cell(cellpar_to_cell(cp), scale_atoms)
        # a new metric can lose operations (cubic -> Fmmm): re-capture when locked
        return self._recaptured(
            "cell changed, atoms kept at their fractional positions" if scale_atoms else
            "cell changed, atoms kept at their Cartesian positions"), list(range(len(self.atoms)))

    def _recaptured(self, text):
        if self.locked:
            self._capture()
            return text + f"; symmetry re-captured ({len(self.ops[0])} operations)"
        return text

    def op_supercell(self, n=(2, 2, 2), matrix=None, vacuum=None):
        """``n``: a diagonal supercell (ase's repeat, as before); ``matrix``
        (Phase 4, Build › Supercell): new vectors M @ cell for an integer M with
        a positive determinant, built like the New-tab result (builders.py)."""
        if matrix is not None:
            from . import builders
            from ase import Atoms
            try:
                r = builders.supercell(cell=self.atoms.cell.array, frac=self.atoms.get_scaled_positions(wrap=False),
                                       numbers=self.atoms.numbers, pbc=tuple(bool(p) for p in self.atoms.pbc), matrix=matrix,
                                       vacuum=vacuum)
            except ValueError as exc:
                raise EditError(str(exc))
            if len(r["numbers"]) > MAX_ATOMS:
                raise EditError(f"more than {MAX_ATOMS} atoms: too big for the editor")
            info = dict(self.atoms.info)
            self.atoms = Atoms(numbers=r["numbers"], cell=r["cell"], scaled_positions=r["frac"], pbc=r["pbc"])
            if not all(r["pbc"]):
                # a repeated axis: its vacuum vector becomes the new block's period (no copy on a copy next time)
                from . import structures as S
                self.atoms.info.update(S.repeated_info(info, r["info"]["repeats"], r["info"]["spacing"]))
            m = r["info"]["matrix"]
            diag = all(m[i][j] == 0 for i in range(3) for j in range(3) if i != j)
            what = f"{m[0][0]}x{m[1][1]}x{m[2][2]}" if diag else f"det {r['info']['det']}"
            return self._recaptured(f"{what} supercell"), None
        n = [int(x) for x in n]
        if min(n) < 1 or np.prod(n) > 64:
            raise EditError("supercell multiples must be 1-64 in total")
        if not all(self.atoms.pbc) and vacuum is not None:
            return self.op_supercell(matrix=np.diag(n).tolist(), vacuum=vacuum)
        if not all(self.atoms.pbc):
            from . import builders
            try:
                builders.check_periodic(np.diag(n), self.atoms.pbc)
            except ValueError as exc:
                raise EditError(str(exc))
        self.atoms = self.atoms.repeat(n)
        return self._recaptured(f"{n[0]}x{n[1]}x{n[2]} supercell"), None

    def op_add_many(self, atoms):
        """Paste and Duplicate (Phase 4): several atoms, each at its fractional
        position, as one step; in Sym mode each one brings its orbit, as Add
        Atom does. ``atoms``: [{element, frac}]."""
        from ase import Atom
        if not isinstance(atoms, (list, tuple)) or not atoms:
            raise EditError("nothing to paste")
        n0 = len(self.atoms)
        added = []
        for a in atoms:
            try:
                z = _number(a["element"])
                f = np.asarray(a["frac"], float).reshape(3)
            except (KeyError, TypeError, ValueError):
                raise EditError("each atom is {element, frac}")
            if not np.all(np.isfinite(f)):
                raise EditError("a position is three finite numbers")
            for p in self._images(f):
                j = self._find(p)
                if j is not None:
                    raise EditError(f"an atom ({self.atoms.get_chemical_symbols()[j]}{j + 1}) already sits there")
                self.atoms.append(Atom(z, p @ self.atoms.cell.array))
                added.append(len(self.atoms) - 1)
        self._label_info = {"n": len(added), "idx": added}
        return f"added {len(added)} atom{'s' * (len(added) != 1)}", list(range(n0))

    def op_defect(self, kind, index=None, element=None, frac=None):
        """One point defect in this cell (Build › Point Defect, This
        structure): on one atom, not its orbit, in Sym mode too (a single
        defect is the point; the group drops and Sym mode re-captures it)."""
        n = len(self.atoms)
        if kind in ("vacancy", "substitution", "antisite"):
            i = int(index) if index is not None else -1
            if not 0 <= i < n:
                raise EditError(f"no atom {i + 1}")
            if kind == "vacancy":
                imap = self._keep_map([i])
                del self.atoms[[i]]
                return self._recaptured(f"removed atom {i + 1}"), imap
            z = _number(element)
            if z == self.atoms.numbers[i]:
                raise EditError("Pick a different element")
            nums = self.atoms.numbers.copy()
            nums[i] = z
            self.atoms.numbers = nums
            return self._recaptured(f"atom {i + 1} is now {self.atoms.get_chemical_symbols()[i]}"), list(range(n))
        if kind == "interstitial":
            from ase import Atom
            if frac is None:
                raise EditError("an interstitial needs a position (fractional x y z)")
            f = np.asarray(frac, float).reshape(3)
            j = self._find(f)
            if j is not None:
                raise EditError(f"an atom ({self.atoms.get_chemical_symbols()[j]}{j + 1}) already sits there")
            self.atoms.append(Atom(_number(element), f @ self.atoms.cell.array))
            return self._recaptured(f"added {self.atoms.get_chemical_symbols()[-1]} at an interstitial site"), list(range(n))
        raise EditError("kind is vacancy, substitution, interstitial or antisite")

    def _standardize(self, primitive: bool):
        import spglib
        from ase import Atoms
        self._need_3d("Symmetrize and the cell transforms")
        a = self.atoms
        res = spglib.standardize_cell((a.cell.array, a.get_scaled_positions(), a.numbers),
                                      to_primitive=primitive, no_idealize=False,
                                      symprec=self.symprec)
        if res is None:
            raise EditError("spglib could not standardize this cell at this tolerance")
        lat, pos, nums = res
        self.atoms = Atoms(numbers=nums, cell=lat, scaled_positions=pos, pbc=True)

    def op_symmetrize(self, symprec=None):
        """spglib's refined cell at the session's tolerance, or (Phase 4,
        Build › Symmetrize) at ``symprec`` for this one step."""
        n0 = len(self.atoms)
        if symprec is None:
            self._standardize(primitive=False)
            return self._recaptured(f"symmetrized (spglib refine at {self.symprec} Å): "
                                    f"{n0} -> {len(self.atoms)} atoms"), None
        keep = self.symprec
        self.symprec = check_symprec(symprec)
        try:
            self._standardize(primitive=False)
        finally:
            self.symprec = keep
        return self._recaptured(f"symmetrized (spglib refine at {float(symprec):g} Å): "
                                f"{n0} -> {len(self.atoms)} atoms"), None

    def op_transform(self, kind):
        """Build › Transform Cell, This structure: primitive or conventional
        (spglib's standard cells, as op_primitive and op_symmetrize make
        them), or the Niggli-reduced cell with the atoms mapped into it."""
        n0 = len(self.atoms)
        if kind == "primitive":
            self._standardize(primitive=True)
        elif kind == "conventional":
            self._standardize(primitive=False)
        elif kind == "niggli":
            import spglib
            from ase import Atoms
            self._need_3d("A Niggli reduction")
            red = spglib.niggli_reduce(self.atoms.cell.array)
            if red is None:
                raise EditError("spglib could not Niggli-reduce this cell")
            red = np.asarray(red, float)
            f = self.atoms.positions @ np.linalg.inv(red)
            f -= np.floor(f + 1e-12)
            self.atoms = Atoms(numbers=self.atoms.numbers, cell=red, scaled_positions=f, pbc=True)
        else:
            raise EditError("kind is primitive, conventional or niggli")
        return self._recaptured(f"{kind} cell: {n0} -> {len(self.atoms)} atoms"), None

    def op_primitive(self):
        n0 = len(self.atoms)
        self._standardize(primitive=True)
        return self._recaptured(f"primitive cell: {n0} -> {len(self.atoms)} atoms"), None

    def op_set_symprec(self, symprec):
        s = check_symprec(symprec)
        self.symprec = s
        if self.locked:
            self._capture()
        return f"symmetry tolerance {s:g} Å", list(range(len(self.atoms)))

    def op_set_locked(self, locked):
        self.locked = bool(locked)
        if self.locked:
            self._capture()
            return (f"symmetry-aware editing on: {len(self.ops[0])} operations captured",
                    list(range(len(self.atoms))))
        self.ops = None
        return "P1 editing: edits touch only the picked atoms", list(range(len(self.atoms)))

    def apply_clean(self, new_atoms, summary, base_rev=None):
        self.check_rev(base_rev)
        before = self._state()
        keep = {k: v for k, v in (self.atoms.info or {}).items() if str(k).startswith("mace_")}
        self.atoms = new_atoms.copy()
        self.atoms.info.update(keep)              # a slab's drawing-box origin and file vacuum
        try:
            summary = self._recaptured(summary)   # Clean without FixSymmetry can lower it
        except Exception:
            self._restore(before)
            raise
        sym = self.symmetry()
        self._commit(before, "Clean Geometry")
        return self.payload(summary, list(range(len(self.atoms))), sym=sym)

    # -- output ---------------------------------------------------------------------
    def boxed_atoms(self):
        """A copy a 3D file format can carry: a slab, polymer or molecule in a
        box with S.SAVE_VACUUM Å of vacuum along each non-periodic direction."""
        return S.boxed(self.atoms) if not all(self.atoms.pbc) else self.atoms.copy()

    def cif_text(self, mode: str = "p1") -> str:
        """The structure as a CIF: ``p1`` (every atom) or ``symmetrized`` (spglib
        refined cell, space group and asymmetric unit)."""
        import io
        import warnings
        import ase.io
        a = S.boxed(self.atoms) if not all(self.atoms.pbc) else self.atoms.copy()   # (also for server._SessLike)
        if mode == "symmetrized":
            if not all(self.atoms.pbc):
                raise EditError("a symmetrized CIF needs a 3D crystal; save this one as P1")
            return _symmetrized_cif(a, self.symprec, self.name)
        if mode != "p1":
            raise EditError("CIF mode is p1 or symmetrized")
        buf = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ase.io.write(buf, a, format="cif")
        return buf.getvalue().decode()


def _symmetrized_cif(atoms, symprec: float, name: str) -> str:
    import spglib
    from ase.geometry import cell_to_cellpar
    ds = spglib.get_symmetry_dataset((atoms.cell.array, atoms.get_scaled_positions(),
                                      atoms.numbers), symprec=symprec)
    if ds is None:
        raise EditError("spglib found no symmetry at this tolerance")
    get = (lambda k: getattr(ds, k)) if hasattr(ds, "number") else ds.__getitem__
    lat, pos, nums = get("std_lattice"), get("std_positions"), get("std_types")
    if len(pos) < len(atoms):
        # a supercell (Phase 4 builds them): the refined cell would be the
        # space group's own, smaller cell, and the extra atoms would be lost
        k = len(atoms) / max(1, len(pos))
        raise EditError(f"this cell is {k:g}× the conventional cell of {get('international')} "
                        f"({len(atoms)} atoms, {len(pos)} in that cell): a symmetrized CIF would keep only "
                        "the smaller cell. Save as CIF (P1) to keep every atom")
    std = spglib.get_symmetry_dataset((lat, pos, nums), symprec=symprec)
    sget = (lambda k: getattr(std, k)) if hasattr(std, "number") else std.__getitem__
    a, b, c, al, be, ga = cell_to_cellpar(lat)
    from ase.data import chemical_symbols
    import re
    block = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "structure"
    lines = [f"data_{block}", f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}",
             f"_cell_length_c {c:.6f}", f"_cell_angle_alpha {al:.4f}",
             f"_cell_angle_beta {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
             f"_symmetry_space_group_name_H-M '{sget('international')}'",
             f"_symmetry_Int_Tables_number {int(sget('number'))}", "", "loop_",
             "_symmetry_equiv_pos_as_xyz"]
    for R, t in zip(sget("rotations"), sget("translations")):
        lines.append("  '" + _xyz(R, t) + "'")
    lines += ["", "loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
              "_atom_site_occupancy"]
    seen = set()
    for k, (eq, f, z) in enumerate(zip(sget("equivalent_atoms"), pos, nums)):
        if eq in seen:
            continue
        seen.add(eq)
        sym = chemical_symbols[int(z)]
        f = np.asarray(f) % 1.0
        lines.append(f"{sym}{len(seen)} {sym} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f} 1.0")
    return "\n".join(lines) + "\n"


def _xyz(R, t) -> str:
    from fractions import Fraction
    out = []
    for row, tt in zip(R, t):
        terms = ""
        for coef, axis in zip(row, "xyz"):
            if coef:
                terms += ("+" if coef > 0 else "-") + ("" if abs(coef) == 1 else str(abs(coef))) + axis
        tt = tt % 1.0
        if abs(tt) > 1e-6 and abs(tt - 1) > 1e-6:
            fr = Fraction(tt).limit_denominator(12)
            terms += f"+{fr.numerator}/{fr.denominator}"
        out.append(terms.lstrip("+") or "0")
    return ",".join(out)


class SessionStore:
    """The open edit sessions, at most MAX_SESSIONS: the least recently used
    one goes first. With a memory ledger attached, their undo history and
    live states count against the one cache budget (memory.MemoryLedger)."""

    def __init__(self):
        self._sessions: "OrderedDict[str, EditSession]" = OrderedDict()
        self._lock = threading.Lock()
        self.ledger = None

    def add(self, session: EditSession) -> EditSession:
        with self._lock:
            while len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)
            self._sessions[session.id] = session
        session.ledger = getattr(self, "ledger", None)
        return session

    def get(self, sid: str) -> EditSession:
        with self._lock:
            s = self._sessions.get(sid or "")
            if s is not None:
                self._sessions.move_to_end(s.id)
        if s is None:
            raise KeyError("edit session (reload the structure into the editor)")
        s.last_used = time.monotonic()
        return s

    def evict(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions)

    def _lru(self) -> List[EditSession]:
        with self._lock:
            return list(self._sessions.values())

    # -- what the memory ledger asks ------------------------------------------
    def history_bytes(self) -> int:
        return sum(s.history_nbytes for s in self._lru())

    def live_bytes(self) -> int:
        return sum(s.live_nbytes() for s in self._lru())

    def drop_oldest_history(self) -> bool:
        """The oldest undo step of the least recently used session that has one."""
        for s in self._lru():
            if not (s.history or s.future):
                continue
            if not s.lock.acquire(blocking=False):
                continue                    # busy right now: never wait on it
            try:
                if s.drop_oldest_step():
                    return True
            finally:
                s.lock.release()
        return False

    def drop_idle_session(self, used_before: float) -> bool:
        """The least recently used session not used since ``used_before``."""
        for s in self._lru():
            if s.last_used < used_before and not s.busy:
                return self.evict(s.id)
        return False
