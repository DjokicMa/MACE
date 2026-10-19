"""The structure a document without one of its own came from (1.1.6).

A band or DOS plot, a volume, an IR/Raman spectrum, a properties deck or a
job log has no atoms of its own, but it was made from a CRYSTAL run that has.
The GUI's side panels (Atoms, Coordination, the Inspector's Structure and
Cell) show that run's structure, read-only and labeled with where it came
from, instead of whatever structure was focused before.

The run is found by name, as the rest of MACE links a run's files
(``Crystal_d3/d3_bz.output_for``, ``results.partner``):

0. ``<stem>.out`` beside the file when it holds a geometry (the FREQ run of
   ``<stem>_IRSPEC.DAT``, the run of a job log ``<stem>-<jobid>.o``);
1. the material's SCF run (``<material>_sp.out``, ``_opt.out``, ``.out``) in
   the same folder, the folder above, or a folder beside it;
2. the properties run itself (``<stem>.out`` with no SCF geometry: CRYSTAL's
   properties program prints the cell and the atoms in Angstrom);
3. the material's deck (``<material>.d12``, ``_sp.d12``, ``_opt.d12``);
4. the one CRYSTAL output with a geometry in the file's folder;
5. a cube's own atoms (its header).

Two or more candidates at the first tier that has any is ambiguous: the
answer lists them and picks none (the page offers a choice) unless ``pick``
names one of them. Nothing here writes a file."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# a data file's name minus what the run's stem does not carry
_SUFFIXES = (".BAND.DAT", ".band.band.dat", "_band.dat", ".DOSS.DAT", "_doss.dat", "_IRSPEC.DAT", "_RAMSPEC.DAT",
             ".SEEBECK.dat", ".SIGMAS.dat", ".SIGMA.dat", ".KAPPA.dat", ".TDF.dat",
             "_DENS.CUBE", "_POT.CUBE", "_SPIN.CUBE", "_WAN.CUBE", "_DENS_CUBE.DAT", "_POT_CUBE.DAT", "_SPIN_CUBE.DAT",
             "_CUBE.DAT", ".CUBE", ".cube", ".f25", ".d3", ".DAT", ".dat", ".out", ".log", ".txt")
_JOBLOG = re.compile(r"^(?P<stem>.+)-\d+\.(?:o|e|out|err|log)$")
# a properties run's tag: <material>_<tag>
_TAGS = ("_band", "_doss", "_dos", "_dens", "_pot", "_spin", "_echg", "_ech3", "_pot3", "_pato", "_wan", "_freq",
         "_transport", "_boltz", "_charge+potential", "_charge", "_potential", "_bands", "_ir", "_raman", "_sp", "_opt")
_SCF = ("{m}_sp.out", "{m}_opt.out", "{m}_SP.out", "{m}_OPT.out", "{m}.out", "{m}_freq.out", "{m}_FREQ.out")
_DECKS = ("{m}.d12", "{m}_sp.d12", "{m}_opt.d12", "{m}_SP.d12", "{m}_OPT.d12")
_SIBLING_DIRS_MAX = 200
_CUBE_NAMES = re.compile(r"(\.cube|_CUBE\.DAT|\.CUBE)$", re.I)


def stems(path: Path) -> List[str]:
    """The run stems a file's name (and, for a generic name, its folder's) can carry, most specific first."""
    name = path.name
    out: List[str] = []
    m = _JOBLOG.match(name)
    if m:
        out.append(m.group("stem"))
    for suf in _SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            out.append(name[: -len(suf)])
            break
    # CRYSTAL's own generic names (DENS_CUBE.DAT, 1_WAN.CUBE, fort.25): the folder is the run
    out.append(path.parent.name)
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def material_of(stem: str) -> str:
    for tag in _TAGS:
        if stem.lower().endswith(tag) and len(stem) > len(tag):
            return stem[: -len(tag)]
    return stem


def _dirs(path: Path) -> List[Path]:
    """The file's folder, the one above, and the folders beside the file's folder (a workflow's steps)."""
    d = path.parent
    out = [d]
    if d.parent != d:
        out.append(d.parent)
        try:
            sib = sorted(x for x in d.parent.iterdir() if x.is_dir() and x != d and not x.name.startswith("."))
        except OSError:
            sib = []
        out.extend(sib[:_SIBLING_DIRS_MAX])
    return out


def _uniq(paths):
    seen, out = set(), []
    for p in paths:
        try:
            k = p.resolve()
        except OSError:
            continue
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


_PROPS_OUT_MAX = 48 * 1024 * 1024     # bytes (the 400 000-line guard, at ~120 bytes a line)


def _props_geometry(path: Path) -> Optional[Dict[str, Any]]:
    """(cell Å, cart Å, numbers, dim) from a CRYSTAL properties output: its last
    DIRECT LATTICE VECTOR COMPONENTS block and the atom table under it."""
    from .structures import _out_dimensionality
    # a properties output is small; a huge .out beside the data (a long SCF
    # log, 2 GB) is not read whole for the panel context (it took minutes)
    try:
        if path.stat().st_size > _PROPS_OUT_MAX:
            return None
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    if len(lines) > 400000:
        return None
    at = [k for k, ln in enumerate(lines) if "DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)" in ln]
    if not at:
        return None
    k = at[-1]
    try:
        cell = np.array([[float(x) for x in lines[k + 1 + i].split()[:3]] for i in range(3)])
    except (ValueError, IndexError):
        return None
    head = next((j for j in range(k, min(len(lines), k + 80)) if re.match(r"^\s*ATOM\s+N\.AT\.\s+SHELL\s+X\(A\)", lines[j])), None)
    if head is None:
        return None
    numbers, cart = [], []
    for ln in lines[head + 2:]:
        p = ln.split()
        if not p or ln.lstrip().startswith("*"):
            break
        try:
            numbers.append(int(p[1]) % 100)
            cart.append([float(p[4]), float(p[5]), float(p[6])])
        except (ValueError, IndexError):
            break
    if not numbers or abs(np.linalg.det(cell)) < 1e-6:
        return None
    dim = _out_dimensionality(lines)
    return {"cell": cell, "cart": np.array(cart), "numbers": numbers, "pbc": [i < dim for i in range(3)]}


def _cube_geometry(path: Path) -> Optional[Dict[str, Any]]:
    """A Gaussian cube's own atoms and cell (Bohr in the file), read from its header only."""
    bohr = 0.529177210903
    try:
        with open(path, errors="replace") as fh:
            head = [fh.readline() for _ in range(6)]
            n = abs(int(head[2].split()[0]))
            if n > 100000:
                return None
            vox = [[float(x) for x in head[3 + i].split()[:4]] for i in range(3)]
            rows = [fh.readline().split() for _ in range(n)]
    except (OSError, ValueError, IndexError):
        return None
    try:
        cell = np.array([[abs(v[0]) * v[1] * bohr, abs(v[0]) * v[2] * bohr, abs(v[0]) * v[3] * bohr] for v in vox])
        numbers = [int(float(r[0])) % 100 for r in rows]
        cart = np.array([[float(r[2]) * bohr, float(r[3]) * bohr, float(r[4]) * bohr] for r in rows])
    except (ValueError, IndexError):
        return None
    if not numbers or abs(np.linalg.det(cell)) < 1e-9:
        return None
    return {"cell": cell, "cart": cart, "numbers": numbers, "pbc": [True, True, True]}


def _geometry_payload(g: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    from .structures import _payload
    return _payload(g["cell"], None, g["cart"], g["numbers"], g["pbc"], source)


def projections(path: Path, symbols: List[str]) -> Optional[List[Dict[str, Any]]]:
    """A DOSS plot's projections and the atoms each covers: the .d3's ``#EL orbital``
    labels, by element (the atoms of that element). None when there is no DOSS deck."""
    name = path.name
    if not (name.endswith((".DOSS.DAT", "_doss.dat", ".d3", ".f25")) or "doss" in name.lower()):
        return None
    cands = []
    for s in stems(path):
        cands += [path.parent / f"{s}.d3", path.parent / f"{material_of(s)}_doss.d3"]
    deck = next((c for c in cands if c.is_file()), None)
    if deck is None:
        return None
    try:
        text = deck.read_text(errors="replace").splitlines()
    except OSError:
        return None
    k = next((i for i, ln in enumerate(text) if ln.strip().upper().startswith("DOSS")), None)
    if k is None:
        return None
    out = []
    lower = [s.lower() for s in symbols]
    for ln in text[k + 2:]:
        if ln.strip().upper().startswith("END"):
            break
        if "#" not in ln:
            continue
        label = ln[ln.find("#") + 1:].strip()
        if not label:
            continue
        parts = label.split()
        el = parts[0].capitalize()
        orb = " ".join(parts[1:]).lower() or "all"
        atoms = [i for i, s in enumerate(lower) if s == el.lower()]
        out.append({"label": f"{el} ({orb})", "element": el, "orbital": orb, "atoms": atoms})
    return out or None


def find(path: Path, *, inside: Callable[[Path], Optional[str]], has_geometry: Callable[[Path], bool],
         payload: Callable[[Path], Optional[Dict[str, Any]]], pick: Optional[str] = None) -> Dict[str, Any]:
    """What the side panels show for ``path``: ``{"state": "found"|"ambiguous"|"none",
    "source": rel, "via": why, "candidates": [{"path", "via"}], "structure": payload,
    "projections": [...]}``. ``inside(p)`` is p's served-relative path or None;
    ``has_geometry(p)`` whether a CRYSTAL output holds a geometry; ``payload(p)`` the
    GUI's structure payload of an output or deck (None when it has none)."""
    path = Path(path)
    tiers: List[List[Tuple[Path, str]]] = []
    st = stems(path)
    dirs = _dirs(path)

    def existing(names, where):
        out = []
        for d in where:
            for n in names:
                p = d / n
                if p.is_file() and p != path and inside(p) is not None:
                    out.append(p)
        return _uniq(out)

    # 0. the run of the same stem, beside the file
    t0 = [p for p in existing([f"{s}.out" for s in st], dirs[:2]) if has_geometry(p)]
    if path.suffix.lower() in (".out", ".log") and has_geometry(path):
        t0 = [path]
    tiers.append([(p, "its own run" if p == path else "the run of the same name") for p in t0])
    # 1. the material's SCF run
    mats = []
    for s in st:
        m = material_of(s)
        if m not in mats:
            mats.append(m)
    # (the nearest first: the file's folder, the one above, then the folders beside it)
    for where, why in ((dirs[:1], "the SCF run of the same material"), (dirs[1:2], "the SCF run of the same material, a folder up"),
                       (dirs[2:], "the SCF run of the same material, in a folder beside it")):
        t1 = [p for p in existing([n.format(m=m) for m in mats for n in _SCF], where) if has_geometry(p) and p not in t0]
        tiers.append([(p, why) for p in t1])
    # 2. the properties run itself: its cell and atoms
    t2 = [p for p in existing([f"{s}.out" for s in st], dirs[:2]) if not has_geometry(p)]
    t2 += [p for p in existing([n for n in (f"{path.parent.name}.out",)], [path.parent]) if not has_geometry(p) and p not in t2]
    if path.suffix.lower() == ".out" and not has_geometry(path):
        t2 = [path]                                     # a properties output: its own cell and atoms
    tiers.append([(p, "its own cell and atoms" if p == path else "the properties run that wrote it") for p in _uniq(t2)])
    # 3. the material's deck
    t3 = existing([n.format(m=m) for m in mats for n in _DECKS], dirs[:2])
    tiers.append([(p, "the deck of the same material") for p in t3])
    # 4. the one output in the folder (for CRYSTAL's own file names only: a README beside three runs is no run's)
    crystal_named = path.name.startswith("fort.") or path.suffix.lower() in (".dat", ".cube", ".f25", ".d3", ".o", ".e")
    outs: List[Path] = []
    if crystal_named:
        try:
            outs = sorted(p for p in path.parent.glob("*.out") if p != path and inside(p) is not None)[:20]
        except OSError:
            outs = []
    tiers.append([(p, "the output in the same folder") for p in outs if has_geometry(p)])

    def rel(p):
        return inside(p)

    def load(p: Path) -> Optional[Dict[str, Any]]:
        s = None
        if p.suffix.lower() == ".out" and not has_geometry(p):
            g = _props_geometry(p)
            if g:
                s = _geometry_payload(g, {"kind": "out", "path": str(p), "properties": True})
        else:
            try:
                s = payload(p)
            except Exception:  # noqa: BLE001 - an unreadable candidate is skipped, never an error here
                s = None
        if s is not None:
            s = dict(s)
            s["path"] = rel(p)
        return s

    def answer(state, p=None, via=None, cands=(), structure=None):
        out = {"path": inside(path), "state": state, "source": rel(p) if p is not None else None, "via": via,
               "candidates": [{"path": rel(c), "via": w} for c, w in cands], "structure": structure, "projections": None}
        if structure:
            out["projections"] = projections(path, structure.get("symbols") or [])
        return out

    if pick:
        for tier in tiers:
            for p, why in tier:
                if rel(p) == pick:
                    s = load(p)
                    if s:
                        return answer("found", p, why + " (picked)", [c for t in tiers for c in t], s)
    for tier in tiers:
        good = [(p, why) for p, why in tier]
        if not good:
            continue
        if len(good) > 1:
            return answer("ambiguous", cands=good)
        p, why = good[0]
        s = load(p)
        if s:
            return answer("found", p, why, good, s)
    # 5. a cube's own atoms
    if _CUBE_NAMES.search(path.name):
        g = _cube_geometry(path)
        if g:
            return answer("found", path, "the atoms in its header", [], _geometry_payload(g, {"kind": "cube", "path": str(path)}))
    return answer("none")


def _last_spin_text(p: Path) -> Optional[str]:
    """The text from the last "ALPHA-BETA ELECTRONS" line of an output to its end (memory-mapped:
    a large output is never read whole), or None when it has none."""
    import mmap
    try:
        with open(p, "rb") as fh:
            if p.stat().st_size == 0:
                return None
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # the block's own header: the line before "MULLIKEN POPULATION ANALYSIS" (SPINLOCK's
                # "ALPHA-BETA ELECTRONS LOCKED TO ..." is not one)
                end = len(mm)
                for _ in range(10_000):
                    i = mm.rfind(b"ALPHA-BETA ELECTRONS", 0, end)
                    if i < 0:
                        return None
                    if re.match(rb"ALPHA-BETA ELECTRONS[ \t]*\r?\n[ \t]*MULLIKEN POPULATION ANALYSIS", mm[i:i + 200]):
                        text = mm[i:i + 4_000_000].decode("utf-8", "replace")
                        nxt = text.find("ALPHA-BETA ELECTRONS", 20)
                        return text if nxt < 0 else text[:nxt]
                    end = i
                return None
    except (OSError, ValueError):
        return None


def spins(path: Path, *, inside: Callable[[Path], Optional[str]]) -> Dict[str, Any]:
    """/api/spins: the Mulliken spin moment of every atom (the Magnetic style), from the final
    ALPHA-BETA Mulliken block of the run: ``path`` itself when it is an output, else the output
    of the same stem beside it (as find's first tier). ``{"state": "ok", "source", "values",
    "symbols", "closed"}`` or ``{"state": "none", "reason"}``."""
    from mace.phonopy_bridge.parent import final_mulliken_spins
    path = Path(path)
    if path.suffix.lower() == ".out":
        cands = [path]
    else:
        names = [f"{s}.out" for s in stems(path)] + [f"{path.stem}.out"]
        cands = _uniq([path.parent / n for n in names if (path.parent / n).is_file() and inside(path.parent / n) is not None])
    if not cands:
        return {"state": "none", "reason": f"No CRYSTAL output named like {path.name} beside it: Magnetic needs the run's Mulliken spins"}
    src = cands[0]
    text = _last_spin_text(src)
    rows = final_mulliken_spins(text) if text else None
    if not rows:
        return {"state": "none", "source": inside(src),
                "reason": f"{src.name} has no ALPHA-BETA Mulliken analysis (a closed-shell run, or one that did not print it)"}
    rows = sorted(rows, key=lambda r: r[0])
    values = [round(r[2], 4) for r in rows]
    return {"state": "ok", "source": inside(src), "values": values, "symbols": [r[1].capitalize() for r in rows],
            "closed": all(abs(v) < 1e-3 for v in values)}
