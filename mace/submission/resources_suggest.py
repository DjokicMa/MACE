"""Size-aware job size suggestions: cores, memory and walltime per deck.

One pure function, :func:`suggest_resources`, turns what MACE can know
before a run into a job size and a one-line reason; :func:`suggest_for_deck`
gathers those inputs from a deck (and the CRYSTAL output of its parent run,
when there is one). ``mace site suggest``, the site's ``[resources] auto``
setting (``mace submit``, the planner, ``--progress`` plans, quick-start,
the next steps of a workflow) and the GUI's Submit dialog all call these two,
so they show the same numbers.

The inputs:

* atomic orbitals (``NUMBER OF AO``): from the parent's ``.out`` when there is
  one, else counted from the deck: an explicit basis shell by shell
  (s 1, sp 4, p 3, d 5, f 7, g 9), an internal one (``BASISSET``) from
  MACE's bundled basis libraries per element;
* the irreducible k-points and the symmetry operators: from the ``.out``, else
  from SHRINK and the expanded structure (spglib);
* spins (``SPIN``/``UHF`` doubles the k-point work), the calculation type and
  the atoms of the asymmetric unit (a FREQ job's displacements);
* the site's limits: cores and memory of a node (``mace site`` caches them),
  the partition's MaxTime.

The rules, and the coefficients below, were fitted to 680 completed CRYSTAL23
runs (the test corpus and the accounting of real cluster jobs) and checked on
a held-out 30 %; see docs/cli.md "Size-aware suggestions". In short:

* work: an SCF cycle costs ``exp(A) * AO**B * (k-points x spins)**C``
  core-seconds; an SP is ~14 cycles, an OPT ~18 (half of all OPT runs stop
  early; long ones get the wide walltime margin), a FREQ ~39 per
  displacement (3 x atoms in the asymmetric unit + 1); a properties job
  ``exp(a) * AO**b`` core-seconds per type;
* cores: Pcrystal spreads the integrals over every process and the k-space
  work over the k-points, so more cores help up to ``max(k-points x spins,
  AO/16)``; within that, the fewest cores that finish the typical estimate
  in an hour (properties: 15 minutes) - small jobs then ask for a few cores
  and start sooner, large ones a full node;
* memory: per process ``a + b*AO^2/symmetry + c*k*AO`` MB (SCF; properties
  ``a + b*AO^2``) times the largest ratio of observed peak to that formula,
  rounded up to whole GB, at least 2 GB per core;
* walltime: the typical estimate times the 95th percentile of observed /
  estimated for that type, times 1.5, rounded up to the hour, at least
  2 hours (properties 1 hour), at most the partition's MaxTime.

Nothing here talks to SLURM: the limits come in as arguments.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------ the model
# Fitted 2026-10-14 on the training 70 % (sha1 split of the output names);
# the evaluation of the held-out 30 % is in the docs.
MODEL_VERSION = "2026-10-14"
# ln(core-seconds per SCF cycle) = A + B ln(AO) + C ln(k-points x spins)
CYCLE = (-4.627, 1.661, 0.522)
# typical SCF cycles of a job (FREQ: per displacement, 3 x asymmetric atoms + 1)
EQUIV_CYCLES = {"SP": 14.1, "OPT": 17.8, "FREQ": 38.9}
# observed / typical at the 95th percentile: the walltime margin
WALL_FACTOR = {"SP": 4.7, "OPT": 27.5, "FREQ": 5.0}
# properties: ln(core-seconds) = a + b ln(AO), margin (the k-point net of a
# properties run is its own NEWK, not the SCF's, so the AOs alone)
PROP_WORK = {"BAND": (-1.380, 1.405, 15.7),
             "DOSS": (1.111, 1.307, 3.4),
             "TRANSPORT": (1.111, 1.307, 3.4),     # four runs only: DOSS's
             "CHARGE+POTENTIAL": (4.844, 0.797, 1.7),
             # no runs to fit: DOSS's, whose cost is its NEWK too (ORBITALS
             # itself takes under a second, and its mesh is at most IS 2)
             "ORBITALS": (1.111, 1.307, 3.4)}
# MB per process: a + b AO^2/nsym/1e6 + c k AO/1e6 (FREQ: nsym 1), times the envelope
MEM_SCF = (427.9, 150.76, 863.3, 2.6)
# MB per process: a + b AO^2/1e6, times the envelope
MEM_PROP = (276.0, 87.2, 1.35)
MIN_MEM_PER_CPU_GB = 2
SCF_TYPES = ("OPT", "SP", "FREQ")
PROP_TYPES = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS")
# the built-in job size (mace/submission/site.py TEMPLATE_RESOURCES): the
# most cores a suggestion asks for when the node size is not known
try:
    from mace.workflow.common.constants import PROPERTIES_NTASKS as _PROP_NTASKS
except ImportError:  # pragma: no cover - imported as a bare module
    _PROP_NTASKS = 8
BUILTIN_NTASKS = {"scf": 32, "prop": _PROP_NTASKS}
DEFAULT_WALL_CAP_S = {"scf": 7 * 86400, "prop": 86400}
TARGET_S = {"scf": 3600, "prop": 900}
MIN_WALL_S = {"scf": 2 * 3600, "prop": 3600}
# on top of the 95th-percentile margin: a job that times out loses all its work
WALL_SAFETY = 1.5
LADDER = (1, 2, 4, 8, 12, 16, 24, 28, 32, 48, 64, 96, 128, 192, 256)

# AOs per shell type (CRYSTAL: LAT 0 s, 1 sp, 2 p, 3 d, 4 f, 5 g; spherical d/f/g)
SHELL_AOS = {0: 1, 1: 4, 2: 3, 3: 5, 4: 7, 5: 9}
# internal basis sets -> the bundled library whose per-element counts stand for it
LIBRARY_FOR = {"POB-TZVP": "full.basis.triplezeta", "POB-TZVP-REV2": "full.basis.triplezeta.rev2022",
               "POB-DZVP": "full.basis.doublezeta", "POB-DZVP-REV2": "full.basis.doublezeta",
               "POB-DZVPP": "full.basis.doublezeta", "MTZVP": "full.basis.triplezeta"}
# split-valence (def2-mSVP, the 3c methods) and minimal sets: AOs per element by
# block of the periodic table, checked against the CRYSTAL23 printout where the
# corpus has the element (H 2, Li 9, C/N/O/F 15, S 18)
_SVP_BLOCKS = ((2, 2), (4, 9), (10, 15), (12, 13), (18, 18), (20, 21), (30, 24), (36, 23),
               (38, 21), (48, 23), (54, 23), (56, 21), (71, 30), (80, 23), (86, 23), (118, 30))
_MIN_BLOCKS = ((2, 1), (10, 5), (18, 9), (20, 13), (30, 18), (36, 18), (38, 22), (54, 27),
               (56, 31), (71, 38), (86, 36), (118, 40))
SVP_SETS = {"SOLDEF2MSVP", "DEF2-MSVP", "DEF2MSVP"}
MIN_SETS = {"STO-3G", "STO-6G", "MINIX", "SOLMINIX"}


def _block(table, z: int) -> int:
    for top, n in table:
        if z <= top:
            return n
    return table[-1][1]


@dataclass
class SizeInfo:
    """What MACE knows of a job's size before it runs."""
    calc_type: Optional[str] = None
    natoms: Optional[int] = None
    nao: Optional[int] = None
    nk: Optional[int] = None            # irreducible k-points
    nspin: int = 1
    nsym: int = 1                       # symmetry operators (memory)
    nirr: Optional[int] = None          # atoms of the asymmetric unit (FREQ)
    dim: str = "CRYSTAL"
    source: str = ""                    # "out: <name>" or "deck"
    approx: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ helpers
def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _crystal_d12():
    p = str(_repo() / "Crystal_d12")
    if p not in sys.path:
        sys.path.insert(0, p)


def base_type(name: Optional[str]) -> Optional[str]:
    t = str(name or "").strip().upper().rstrip("0123456789")
    return t if t in SCF_TYPES + PROP_TYPES else None


def kind_of(calc_type: str) -> str:
    return "prop" if calc_type in PROP_TYPES else "scf"


def _gb(mb: float) -> int:
    return max(1, int(math.ceil(mb / 1024.0)))


def walltime_text(seconds: float) -> str:
    """D-HH:MM:SS (the form the job templates use)."""
    s = int(seconds)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{sec:02d}"


def _human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, int(round(seconds)))} s"
    if seconds < 5400:
        return f"{int(round(seconds / 60))} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _snap(p: int, bound: int) -> int:
    """The first ladder value >= p (1, 2, 4, 8, 12, 16, 24, ...), or the
    bound itself when that is smaller."""
    p = max(1, min(int(p), bound))
    for v in LADDER:
        if v >= p:
            return min(v, bound)
    return bound


# ------------------------------------------------------------------ the rule
def estimate_work(calc_type: str, info: SizeInfo) -> Tuple[float, float]:
    """(typical core-seconds, walltime margin) of the whole job."""
    k = max(1, int(info.nk or 1)) * max(1, int(info.nspin or 1))
    nao = max(1, int(info.nao or 1))
    if calc_type in PROP_TYPES:
        a, b, margin = PROP_WORK[calc_type]
        return math.exp(a + b * math.log(nao)), margin
    a, b, c = CYCLE
    per_cycle = math.exp(a + b * math.log(nao) + c * math.log(k))
    n = EQUIV_CYCLES[calc_type]
    if calc_type == "FREQ":
        n *= 3 * max(1, int(info.nirr or info.natoms or 1)) + 1
    return per_cycle * n, WALL_FACTOR[calc_type]


def memory_per_task_mb(calc_type: str, info: SizeInfo) -> float:
    """The planned peak memory of one process (the envelope included)."""
    nao = max(1, int(info.nao or 1))
    if calc_type in PROP_TYPES:
        a, b, env = MEM_PROP
        return (a + b * nao * nao / 1e6) * env
    k = max(1, int(info.nk or 1)) * max(1, int(info.nspin or 1))
    nsym = 1 if calc_type == "FREQ" else max(1, int(info.nsym or 1))
    a, b, c, env = MEM_SCF
    return (a + b * nao * nao / nsym / 1e6 + c * k * nao / 1e6) * env


def suggest_resources(calc_type: str, info, limits: Optional[Dict[str, Any]] = None,
                      ntasks: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The job size for one deck: {ntasks, mem_per_cpu, mem, walltime,
    reason, inputs, estimate}, or None when the size is not known (no AO
    count). ``info`` is a SizeInfo (or its dict). ``limits``: node_cores,
    node_mem_mb, max_walltime_s (each optional). ``ntasks``: a core count
    chosen elsewhere (memory and walltime then follow it)."""
    ct = base_type(calc_type)
    if ct is None:
        return None
    if isinstance(info, dict):
        info = SizeInfo(**{k: v for k, v in info.items() if k in SizeInfo.__dataclass_fields__})
    if not info or not info.nao:
        return None
    limits = limits or {}
    kind = kind_of(ct)
    cap = int(limits.get("node_cores") or BUILTIN_NTASKS[kind])
    kpts = max(1, int(info.nk or 1)) * max(1, int(info.nspin or 1))
    work, margin = estimate_work(ct, info)
    per_task = memory_per_task_mb(ct, info)
    if kind == "scf":
        useful = max(kpts, int(math.ceil(info.nao / 16.0)))
        per_task_gb = max(MIN_MEM_PER_CPU_GB, _gb(per_task))
    else:
        useful = max(1, int(math.ceil(info.nao / 100.0)))
        per_task_gb = max(1, _gb(per_task))
    need = max(1, int(math.ceil(work / TARGET_S[kind])))
    p = _snap(min(useful, need), min(useful, cap))
    why_mem = ""
    node_mem = limits.get("node_mem_mb")
    if node_mem and per_task_gb * 1024 * p > 0.9 * node_mem:
        fit = max(1, int(0.9 * node_mem // (per_task_gb * 1024)))
        if fit < p:
            p, why_mem = fit, f"; {fit} cores fit the node's memory"
    chosen = int(ntasks) if ntasks else p
    wall_cap = int(limits.get("max_walltime_s") or DEFAULT_WALL_CAP_S[kind])
    typical_s = work / chosen
    wall = max(MIN_WALL_S[kind], math.ceil(typical_s * margin * WALL_SAFETY / 3600.0) * 3600)
    capped = wall > wall_cap
    wall = min(wall, wall_cap)
    mem_total_gb = per_task_gb * chosen if kind == "scf" else max(4, _gb(per_task * chosen) + 1)
    parts = [f"~{info.nao:,} AOs"]
    if kind == "scf":
        nk = max(1, int(info.nk or 1))
        kp = f"{nk} k-point{'s' if nk > 1 else ''}" if nk > 1 or info.dim != "MOLECULE" else "a molecule"
        parts.append(f"{kp} x {info.nspin} spins" if info.nspin > 1 else kp)
    if ct == "FREQ":
        parts.append(f"{3 * max(1, int(info.nirr or info.natoms or 1))} displacements")

    def cores(n):
        return f"{n} core{'s' if n != 1 else ''}"
    useful_txt = f"{cores(min(useful, cap))} useful" if useful <= cap else f"{cap}+ cores useful"
    if ntasks:
        head = f"{cores(chosen)} (chosen)"
    elif p < min(useful, cap) and not why_mem:
        head = f"{cores(p)} ({useful_txt}; {cores(p)} finish{'es' if p == 1 else ''} in ~{_human_time(work / p)})"
    else:
        head = f"{cores(p)} ({useful_txt}{why_mem})"
    reason = (f"{', '.join(parts)} -> {head}; ~{per_task_gb}G per core; "
              f"walltime {_human_time(wall)}{' (the most allowed)' if capped else ''} "
              f"(estimate ~{_human_time(typical_s)})")
    return {"calc_type": ct, "ntasks": chosen, "mem_per_cpu": f"{per_task_gb}G", "mem": f"{mem_total_gb}G",
            "walltime": walltime_text(wall), "reason": reason, "inputs": info.as_dict(),
            "estimate": {"core_hours": round(work / 3600.0, 3), "hours": round(typical_s / 3600.0, 3),
                         "mem_mb_per_task": round(per_task), "useful_cores": useful, "model": MODEL_VERSION}}


# ------------------------------------------------------------------ inputs: CRYSTAL output
_OUT_PAT = {
    "natoms": re.compile(r"N\. OF ATOMS PER CELL\s+(\d+)"),
    "nao": re.compile(r"NUMBER OF AO\s+(\d+)"),
    "nk": re.compile(r"NUMBER OF K POINTS IN THE IBZ\s+(\d+)"),
    "nsym": re.compile(r"NUMBER OF SYMMETRY OPERATORS\s*:\s*(\d+)"),
    "nirr": re.compile(r"NUMBER OF IRREDUCIBLE ATOMS IN THE CONVENTIONAL CELL:\s*(\d+)"),
}
_DIM_BANNERS = {"MOLECULE CALCULATION": "MOLECULE", "MOLECULAR CALCULATION": "MOLECULE",
                "POLYMER CALCULATION": "POLYMER", "SLAB CALCULATION": "SLAB", "CRYSTAL CALCULATION": "CRYSTAL"}


def read_out_size(path) -> Optional[SizeInfo]:
    """The size a CRYSTAL output printed before its first SCF cycle, or None."""
    found: Dict[str, int] = {}
    spin, dim = 1, None
    try:
        with open(path, errors="replace") as fh:
            for i, line in enumerate(fh):
                if line.startswith(" CYC ") or i > 200000:
                    break
                if "UNRESTRICTED OPEN SHELL" in line:
                    spin = 2
                s = line.strip()
                if dim is None and s in _DIM_BANNERS:
                    dim = _DIM_BANNERS[s]
                for k, r in _OUT_PAT.items():
                    if k not in found:
                        m = r.search(line)
                        if m:
                            found[k] = int(m.group(1))
                if "nk" in found and "nao" in found and "TTTTTTTT" in line and "INT_SCREEN" in line:
                    break
    except OSError:
        return None
    if "nao" not in found:
        return None
    return SizeInfo(natoms=found.get("natoms"), nao=found["nao"], nk=found.get("nk", 1), nspin=spin,
                    nsym=found.get("nsym", 1), nirr=found.get("nirr"), dim=dim or "CRYSTAL",
                    source=f"out: {Path(path).name}")


_D3_SUFFIXES = ("_band", "_doss", "_transport", "_charge+potential", "_charge", "_potential",
                "_ech3", "_pot3", "_boltz", "_orbitals")


def find_size_out(deck) -> Optional[Path]:
    """The CRYSTAL output this deck's size can be read from: the deck's own
    (an earlier run), else its parent run's in the same folder - the longest
    output name the deck's name starts with (``X_opt.out`` for
    ``X_opt_sp.d12``, ``X_sp.out`` for ``X_sp_band.d3``)."""
    deck = Path(deck)
    folder = deck.parent
    stem = deck.stem
    own = folder / f"{stem}.out"
    if own.is_file() and read_out_size(own):
        return own
    low = stem.lower()
    for suf in _D3_SUFFIXES:
        if low.endswith(suf):
            cand = folder / f"{stem[:-len(suf)]}.out"
            if cand.is_file() and read_out_size(cand):
                return cand
    try:
        outs = sorted((p for p in folder.glob("*.out") if stem.startswith(p.stem) and p.stem != stem),
                      key=lambda p: -len(p.stem))
    except OSError:
        outs = []
    for cand in outs[:5]:
        if read_out_size(cand):
            return cand
    return None


# ------------------------------------------------------------------ inputs: the deck
_LIB_CACHE: Dict[str, Dict[int, int]] = {}


def record_aos(lines: List[str], i: int = 0) -> Tuple[int, int, int]:
    """(Z, AOs, next line) of the basis record starting at ``lines[i]``
    (CRYSTAL23 manual 2.2.1 and the ECP section; Crystal_d12/basis_library.py
    cut_record reads the same grammar)."""
    nat, nshl = (int(t) for t in lines[i].split()[:2])
    i += 1
    if nat > 200:
        key = lines[i].strip().upper()
        i += 1
        if key in ("INPUT", "INPSOC"):
            if key == "INPSOC":
                i += 2
            counts = [float(x.replace("D", "E")) for x in lines[i].split()]
            i += 1 + int(sum(counts[1:]))
    n = 0
    for _ in range(nshl):
        f = lines[i].split()
        i += 1
        ityb, lat, ng = int(float(f[0])), int(float(f[1])), int(float(f[2]))
        n += SHELL_AOS.get(lat, 5)
        if ityb == 0:
            i += ng
    return nat % 100, n, i


def library_aos(family: str) -> Dict[int, int]:
    """{Z: AOs} of a bundled basis library (Crystal_d12/basis_sets)."""
    if family in _LIB_CACHE:
        return _LIB_CACHE[family]
    out: Dict[int, int] = {}
    try:
        _crystal_d12()
        import basis_library
        for _key, raw in basis_library.iter_records(family):
            text = raw.decode() if isinstance(raw, bytes) else raw
            try:
                z, n, _ = record_aos([l for l in text.split("\n") if l.strip()])
                out[z] = n
            except (ValueError, IndexError):
                continue
    except Exception:
        out = {}
    _LIB_CACHE[family] = out
    return out


def element_aos(basis: str, z: int) -> Tuple[int, bool]:
    """(AOs of element ``z`` in the internal basis set ``basis``, exact?)."""
    name = (basis or "").strip().upper()
    if name in SVP_SETS:
        return _block(_SVP_BLOCKS, z), False
    if name in MIN_SETS:
        return _block(_MIN_BLOCKS, z), False
    fam = LIBRARY_FOR.get(name, "full.basis.triplezeta")
    table = library_aos(fam)
    if z in table:
        return table[z], name in LIBRARY_FOR
    return _block(_SVP_BLOCKS, z) + 12, False


def _deck_text_basis(text: str) -> Dict[int, int]:
    """{Z: AOs} of an explicit basis in a deck (the records before ``99 0``)."""
    lines = [l for l in text.split("\n")]
    # the basis follows the END of the geometry block: find "99 0" and walk back
    end = None
    for j, l in enumerate(lines):
        if re.fullmatch(r"\s*99\s+0\s*", l):
            end = j
            break
    if end is None:
        return {}
    # the start: the first line after an END/ENDOPT/ENDFREQ... record that
    # parses as "NAT NSHL" and from which the records reach "99 0" exactly
    starts = [j + 1 for j, l in enumerate(lines[:end]) if l.strip().upper().startswith("END")]
    for s in starts:
        body = [l for l in lines[s:end] if l.strip()]
        out: Dict[int, int] = {}
        i = 0
        try:
            while i < len(body):
                z, n, i = record_aos(body, i)
                out[z] = n
        except (ValueError, IndexError):
            continue
        if out:
            return out
    return {}


def _shrink(text: str) -> Optional[List[int]]:
    lines = [l.split("#")[0].strip() for l in text.split("\n")]
    for j, l in enumerate(lines):
        if l.upper() == "SHRINK" and j + 1 < len(lines):
            first = lines[j + 1].split()
            try:
                vals = [int(x) for x in first]
            except ValueError:
                return None
            if vals and vals[0] == 0 and j + 2 < len(lines):
                try:
                    return [int(x) for x in lines[j + 2].split()[:3]]
                except ValueError:
                    return None
            return [vals[0]] * 3 if vals else None
    return None


def _supercell(text: str, dim: str) -> int:
    """|det| of a SUPERCEL / SCELPHONO expansion in the geometry block (1 without one)."""
    lines = [l.split("#")[0].strip() for l in text.split("\n")]
    size = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 1}.get(dim, 0)
    total = 1
    for j, l in enumerate(lines):
        if l.upper() in ("SUPERCEL", "SCELPHONO", "SUPERCON") and size:
            try:
                m = [[float(x) for x in lines[j + 1 + r].split()[:size]] for r in range(size)]
            except (ValueError, IndexError):
                continue
            if size == 1:
                det = m[0][0]
            elif size == 2:
                det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
            else:
                det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                       - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                       + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            total *= max(1, int(round(abs(det))))
    return total


def _spin(text: str) -> int:
    for l in text.split("\n"):
        w = l.split("#")[0].strip().upper()
        if w in ("SPIN", "UHF", "UKS", "ATOMSPIN", "SPINLOCK"):
            return 2
    return 1


_CENTERING = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}


def _structure(parsed: Dict[str, Any], deck: Path):
    """(numbers in the primitive cell, nsym, primitive cell or None, frac or None)."""
    dim = parsed.get("dimensionality", "CRYSTAL")
    asym = [int(a["atomic_number"]) % 100 for a in parsed.get("atoms") or []]
    if dim != "CRYSTAL":
        return asym, 1, None, None, True
    try:
        sys.path.insert(0, str(_repo()))
        from mace.gui.structures import d12_atoms
        atoms, sg = d12_atoms(deck)
    except Exception:
        return asym, 1, None, None, True
    numbers = list(atoms.numbers)
    try:
        import spglib
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        prim = spglib.find_primitive(cell, symprec=1e-3)
        ds = spglib.get_symmetry_dataset(prim or cell, symprec=1e-3)
        rots = getattr(ds, "rotations", None)
        if rots is None and isinstance(ds, dict):
            rots = ds.get("rotations")
        nsym = len({tuple(map(tuple, r)) for r in rots}) if rots is not None else 1
        if prim:
            return list(prim[2]), nsym, prim[0], prim[1], False
    except Exception:
        nsym = 1
    letter = str((atoms.info.get("spacegroup") or "P").symbol)[0] if atoms.info.get("spacegroup") else "P"
    mult = _CENTERING.get(letter, 1)
    return numbers[:max(1, len(numbers) // mult)], nsym, None, None, False


def _ibz(lattice, frac, numbers, mesh) -> Optional[int]:
    try:
        import numpy as np
        import spglib
        out = spglib.get_ir_reciprocal_mesh(list(mesh), (lattice, frac, numbers), is_shift=[0, 0, 0],
                                            is_time_reversal=True, symprec=1e-3)
        if out is None:
            return None
        return int(len(np.unique(out[0])))
    except Exception:
        return None


def deck_size(path, calc_type: Optional[str] = None) -> Optional[SizeInfo]:
    """The size of a .d12 deck from the deck alone (no output), or None."""
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    try:
        _crystal_d12()
        from d12_parsers import CrystalInputParser
        parsed = CrystalInputParser(str(path)).parse()
    except Exception:
        return None
    approx: List[str] = []
    numbers, nsym, lattice, frac, rough = _structure(parsed, path)
    if not numbers:
        return None
    if rough:
        approx.append("atoms")
    explicit = _deck_text_basis(text) if (parsed.get("basis_set_type") or "").upper() != "INTERNAL" else {}
    nao = 0
    basis = parsed.get("basis_set") or ""
    for z in numbers:
        if z in explicit:
            nao += explicit[z]
        else:
            n, exact = element_aos(basis, z)
            nao += n
            if not exact and "AOs" not in approx:
                approx.append("AOs")
    dim = parsed.get("dimensionality", "CRYSTAL") or "CRYSTAL"
    cells = _supercell(text, dim)
    natoms = len(numbers) * cells
    nao *= cells
    nk = 1
    if dim != "MOLECULE":
        mesh = _shrink(text)
        if mesh:
            mesh = (mesh + [1, 1, 1])[:3]
            if dim == "SLAB":
                mesh[2] = 1
            elif dim == "POLYMER":
                mesh[1] = mesh[2] = 1
            nk = _ibz(lattice, frac, numbers, mesh) if lattice is not None else None
            if not nk:
                full = mesh[0] * mesh[1] * mesh[2]
                nk = max(1, int(math.ceil(full / (2.0 * max(1, nsym)))) + 1)
                approx.append("k-points")
    return SizeInfo(calc_type=base_type(calc_type), natoms=natoms, nao=nao, nk=nk, nspin=_spin(text),
                    nsym=nsym, nirr=(len(parsed.get("atoms") or []) * cells) or None, dim=dim, source="deck",
                    approx=approx)


def _deck_type(path: Path) -> Optional[str]:
    try:
        from mace.submission.site import deck_calc_type
    except Exception:          # pragma: no cover
        return None
    return base_type(deck_calc_type(path))


def size_for(path, calc_type: Optional[str] = None) -> Optional[SizeInfo]:
    """The size of a deck's job: the parent's output when there is one (exact
    AOs, k-points, symmetry), with the deck's own spin and asymmetric atoms;
    else estimated from the deck."""
    path = Path(path)
    ct = base_type(calc_type) or _deck_type(path)
    out = find_size_out(path)
    info = read_out_size(out) if out else None
    deck_info = None
    if path.suffix.lower() == ".d12":
        if info is None or ct == "FREQ":
            deck_info = deck_size(path, ct)
    if info is None:
        info = deck_info
    elif deck_info is not None:
        info.nirr = deck_info.nirr or info.nirr
    if info is None:
        return None
    if path.suffix.lower() == ".d12":
        try:
            info.nspin = _spin(path.read_text(errors="replace"))
        except OSError:
            pass
    info.calc_type = ct
    return info


# ------------------------------------------------------------------ the site's limits
def limits_for(site=None, kind: str = "scf", env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """{node_cores, node_mem_mb, max_walltime_s}: the node shape `mace site`
    cached for the partition the jobs go to ([slurm] partition, else the
    cluster's default partition, scontrol's Default=YES), or the smallest
    cached partition's when that one is not known (a suggestion must fit
    every node it may land on, never only the largest), and the partition's
    MaxTime (``site.max_time``, set by site.resolve / resolve_limit)."""
    out: Dict[str, Any] = {"node_cores": None, "node_mem_mb": None, "max_walltime_s": None}
    try:
        from mace.submission import site as S
        from mace.submission.site import envmodules
        env = dict(os.environ if env is None else env)
        entries = envmodules._read_cache(env)
        part = ((getattr(site, "partition", None) or "").split(",")[0]) if site is not None else ""
        shapes: Dict[str, Dict[str, Any]] = {}
        for v in entries.values():
            if isinstance(v, dict) and isinstance(v.get("shapes"), dict):
                for name, sh in v["shapes"].items():
                    if isinstance(sh, dict):
                        shapes.setdefault(name, sh)
        if not part and shapes:
            part = S.default_partition(env) or ""
        if part in shapes:
            chosen = [shapes[part]]
        elif site is not None and getattr(site, "partition", None):
            chosen = []                 # the partition someone set: no cached shape, no guess
        else:
            chosen = list(shapes.values())
        cores = [sh["max_cores"] for sh in chosen if isinstance(sh.get("max_cores"), int)]
        mems = [sh["max_mem_mb"] for sh in chosen if isinstance(sh.get("max_mem_mb"), int) and sh["max_mem_mb"]]
        out["node_cores"] = min(cores) if cores else None
        out["node_mem_mb"] = min(mems) if mems else None
    except Exception:
        pass
    cap = (getattr(site, "max_time", None) or getattr(site, "walltime_cap", None)) if site is not None else None
    if cap:
        out["max_walltime_s"] = int(cap)
    return out


def suggest_for_deck(path, site=None, env: Optional[Dict[str, str]] = None, calc_type: Optional[str] = None,
                     ntasks: Optional[int] = None, limits: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The suggestion for one deck (None: the type or size is not known)."""
    path = Path(path)
    ct = base_type(calc_type) or _deck_type(path)
    if ct is None:
        return None
    info = size_for(path, ct)
    if info is None:
        return None
    if limits is None:
        limits = limits_for(site, kind_of(ct), env)
    out = suggest_resources(ct, info, limits, ntasks=ntasks)
    if out is not None:
        out["deck"] = path.name
        out["limits"] = limits
    return out


def suggestion_json(s: Optional[Dict[str, Any]]) -> str:
    return json.dumps(s, sort_keys=True, default=str)


# ------------------------------------------------------------------ inputs: a CIF (planner, quick-start)
def cif_size(path, calc_type: Optional[str] = None, basis: Optional[str] = None,
             nspin: int = 2) -> Optional[SizeInfo]:
    """The size of a CIF's job before MACE writes its deck: the primitive
    cell (spglib), ``basis`` per element (the planner's choice; the TZ library
    by default), the SHRINK MACE's CIF conversion picks (d12_constants
    generate_k_points), spin-polarized as MACE's decks are by default."""
    try:
        from ase.io import read
        import spglib
        atoms = read(str(path))
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        prim = spglib.find_primitive(cell, symprec=1e-3) or cell
        ds = spglib.get_symmetry_dataset(prim, symprec=1e-3)
        rots = getattr(ds, "rotations", None)
        if rots is None and isinstance(ds, dict):
            rots = ds.get("rotations")
        nsym = len({tuple(map(tuple, r)) for r in rots}) if rots is not None else 1
        equiv = getattr(ds, "equivalent_atoms", None)
        if equiv is None and isinstance(ds, dict):
            equiv = ds.get("equivalent_atoms")
        sg = getattr(ds, "number", None) or (ds.get("number") if isinstance(ds, dict) else 1)
        _crystal_d12()
        from d12_constants import generate_k_points
        a, b, c = atoms.cell.lengths()
        mesh = list(generate_k_points(a, b, c, "CRYSTAL", int(sg or 1)))
    except Exception:
        return None
    numbers = [int(z) for z in prim[2]]
    nao = 0
    for z in numbers:
        nao += element_aos(basis or "POB-TZVP-REV2", z)[0]
    nk = _ibz(prim[0], prim[1], prim[2], mesh) or max(1, (mesh[0] * mesh[1] * mesh[2]) // max(1, nsym))
    nirr = len(set(int(x) for x in equiv)) if equiv is not None else len(numbers)
    return SizeInfo(calc_type=base_type(calc_type), natoms=len(numbers), nao=nao, nk=nk, nspin=nspin, nsym=nsym,
                    nirr=nirr, source="cif", approx=["AOs", "k-points"])


def suggest_for_inputs(inputs, calc_type: str, site=None, env: Optional[Dict[str, str]] = None,
                       basis: Optional[str] = None, ntasks: Optional[int] = None,
                       limits: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """One suggestion that fits every input (decks or CIFs) of a plan step of
    ``calc_type``: the most cores, memory and walltime any of them gets, with
    the reason of the largest job. None when none of them can be sized."""
    ct = base_type(calc_type)
    if ct is None:
        return None
    if limits is None:
        limits = limits_for(site, kind_of(ct), env)
    found = []
    for p in inputs or ():
        p = Path(p)
        if p.suffix.lower() == ".cif":
            info = cif_size(p, ct, basis)
        else:
            info = size_for(p, ct)
        s = suggest_resources(ct, info, limits, ntasks=ntasks) if info else None
        if s:
            s["deck"] = p.name
            found.append(s)
    if not found:
        return None
    big = max(found, key=lambda s: s["estimate"]["core_hours"])
    out = dict(big)

    def gb(v):
        return int(str(v).rstrip("G"))

    def secs(w):
        d, rest = w.split("-", 1)
        h, m, sec = (int(x) for x in rest.split(":"))
        return int(d) * 86400 + h * 3600 + m * 60 + sec
    out["ntasks"] = max(s["ntasks"] for s in found)
    out["mem_per_cpu"] = f"{max(gb(s['mem_per_cpu']) for s in found)}G"
    out["mem"] = f"{max(gb(s['mem']) for s in found)}G"
    out["walltime"] = max((s["walltime"] for s in found), key=secs)
    if len(found) > 1:
        out["reason"] = f"largest of {len(found)} inputs ({big['deck']}): {big['reason']}"
    out["limits"] = limits
    return out
