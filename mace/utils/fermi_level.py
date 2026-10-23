"""A trustworthy Fermi level for a run's band data.

CRYSTAL stores the Fermi energy in the wavefunction (fort.9). An SCF that
ends with its spins still locked (SPINLOCK n m with m larger than the
cycles it took, MACE's spin-polarized default for a closed-shell system)
never computes one ("SPIN LOCKING: NO ENERGY GAP COMPUTED"): the
wavefunction keeps the placeholder -1.0 Hartree, every properties run
prints it ("N. OF SCF CYCLES ... FERMI ENERGY -0.100E+01"), and a BAND run
without NEWK writes its BAND.DAT / fort.25 energies as E - (-1.0) with
"EFERMI -1.00000" (the example suite's diamond: a "direct 249 eV gap", the
band plot 27 eV off). A DOSS run recomputes E_F (its NEWK) and is right.

:func:`band_fermi` gives the Fermi level of a BAND.DAT or a BAND fort.25:
the file's own, or, when that is the placeholder, the top of the occupied
manifold of its bands. The occupied bands per spin are the run's electron
count (the Mulliken count, else the header's, as the BAND gap's
band_occupied_count reads it) split by its alpha - beta difference (the
summed spin density the properties run prints, the value SPINLOCK locked):
E_F is the highest of the occupied bands, and the run is refused (no number)
when an occupied band reaches above a virtual one (a metal: its Fermi level
needs CRYSTAL's, from a run whose spins are not locked, or NEWK). Every
reader of band energies (the extractor and its derived properties, the
effective masses, the GUI's band plot and mace plotting, the fort.25
conversion) takes the shift from here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: what a wavefunction holds when its SCF computed no Fermi level (Hartree)
PLACEHOLDER_AU = -1.0

_SPIN_DENSITY = re.compile(r"SUMMED SPIN DENSITY\s+(-?\d+\.\d+)")
_LOCKED = re.compile(r"ALPHA-BETA ELECTRONS LOCKED TO\s+(-?\d+)")
_BAND_RANGE = re.compile(r"FROM BAND\s*(\d+)\s*TO BAND\s*(\d+)")
_DAT_EF = re.compile(r"^#\s*EFERMI\s*\(HARTREE\)\s*(-?\d+\.?\d*(?:[Ee][+-]?\d+)?)", re.M)


def is_placeholder(ef_au) -> bool:
    """CRYSTAL's "no Fermi level" value (-1.0 Hartree, printed -0.100E+01 or -1.00000)."""
    try:
        return ef_au is not None and abs(float(ef_au) - PLACEHOLDER_AU) < 5e-6
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Fermi:
    """The Fermi level of a band file: ``value_au`` (None when it cannot be known), the
    file's own ``file_au``, whether it was derived from the bands, and why (``note``)."""
    value_au: Optional[float]
    file_au: Optional[float]
    derived: bool = False
    note: str = ""

    @property
    def shift_au(self) -> float:
        """What to add to the file's E - E_F(file) to get E - E_F (0 when nothing is corrected)."""
        if self.value_au is None or self.file_au is None:
            return 0.0
        return float(self.file_au) - float(self.value_au)

    @property
    def known(self) -> bool:
        return self.value_au is not None


def scf_parent_of_band_output(output_file: Path) -> Optional[Path]:
    """The SCF .out a BAND .out was run from, when it sits beside it as <stem>.out."""
    stem = output_file.stem
    for suffix in ("_band", "_BAND", ".band", ".BAND"):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            candidate = output_file.with_name(stem[:-len(suffix)] + ".out")
            if candidate.exists() and candidate != output_file:
                return candidate
    return None


def spin_occupations(content: str, output_file: Optional[Path] = None) -> Optional[Tuple[int, int]]:
    """(alpha, beta) electrons of the run whose output ``content`` is: its count as the BAND gap's
    band_occupied_count reads it (Mulliken, the SCF parent's, else the header), split by the
    alpha - beta difference (SPINLOCK's locked value, else the summed spin density the properties
    header prints, else 0). None when they are not whole numbers."""
    from mace.utils.ion_fragments import mulliken_electron_count
    count = mulliken_electron_count(content)
    if count is None and output_file is not None:
        parent = scf_parent_of_band_output(Path(output_file))
        if parent is not None:
            try:
                count = mulliken_electron_count(parent.read_text(errors="ignore"))
            except OSError:
                count = None
    if count is None:
        header = re.findall(r"N\.?\s*OF ELECTRONS PER CELL\s*(\d+)", content)
        if not header:
            return None
        count = float(header[-1])
    locked = _LOCKED.findall(content)
    spins = _SPIN_DENSITY.findall(content)
    diff = float(locked[-1]) if locked else float(spins[-1]) if spins else 0.0
    n, d = round(count), round(diff)
    if abs(count - n) > 1e-3 or abs(diff - d) > 1e-3 or (n + d) % 2:
        return None
    return (n + d) // 2, (n - d) // 2


def derive(blocks: Sequence[np.ndarray], occupations: Tuple[int, int], first_band: int = 1) \
        -> Tuple[Optional[float], str]:
    """(E_F, why) from band energies (``blocks``: one (k, bands) array per spin, any common
    reference; E_F is returned in it): the top of the occupied manifold, the occupied bands per
    spin from ``occupations`` (alpha, beta) less the bands below ``first_band``. None and why
    when it cannot be known (too few bands, or an occupied band above a virtual one: a metal)."""
    if not blocks:
        return None, "no band energies"
    nocc = list(occupations) if len(blocks) == 2 else [occupations[0]]
    if len(blocks) == 1 and occupations[0] != occupations[1]:
        return None, "a spin-polarized occupation on one spin channel"
    tops, bottoms = [], []
    for blk, n_s in zip(blocks, nocc):
        n = int(n_s) - (int(first_band or 1) - 1)
        if n < 1:
            if n_s:
                return None, f"the file starts at band {first_band}, above the occupied bands"
            continue
        if n > blk.shape[1]:
            return None, f"the file holds {blk.shape[1]} bands from band {first_band}, fewer than the occupied {n_s}"
        tops.append(float(blk[:, :n].max()))
        if n < blk.shape[1]:
            bottoms.append(float(blk[:, n:].min()))
    if not tops:
        return None, "no occupied band in the file"
    vbm = max(tops)
    if bottoms and min(bottoms) < vbm:
        return None, ("an occupied band reaches above a virtual one (a metal): its Fermi level is not the top "
                      "of the occupied bands")
    return vbm, ""


def _sibling_out(path: Path) -> Optional[Path]:
    """The BAND run's .out beside its BAND.DAT / fort.25 (<stem>.out)."""
    name = path.name
    for suffix in (".BAND.DAT", ".band.band.dat", "_band.dat", ".fort.25", ".f25"):
        if name.lower().endswith(suffix.lower()):
            cand = path.with_name(name[: -len(suffix)] + ".out")
            if cand.is_file():
                return cand
    cand = path.with_suffix(".out")
    return cand if cand.is_file() else None


def resolve(file_au: Optional[float], blocks: Sequence[np.ndarray], output_file: Optional[Path],
            content: Optional[str] = None) -> Fermi:
    """The Fermi level of band energies ``blocks`` written relative to ``file_au`` (E - file_au, in
    Hartree): ``file_au`` unless it is CRYSTAL's placeholder, else derived (see :func:`derive`)
    from the run's .out (``output_file`` / ``content``: its electron count, spin and band range)."""
    if not is_placeholder(file_au):
        return Fermi(file_au, file_au)
    if content is None and output_file is not None:
        try:
            content = Path(output_file).read_text(errors="replace")
        except OSError:
            content = None
    if not content:
        return Fermi(None, file_au, note="CRYSTAL wrote no Fermi level (-1.0, SPINLOCK) and the run's .out is "
                                         "not beside the band file: E_F is not known")
    occ = spin_occupations(content, output_file)
    if occ is None:
        return Fermi(None, file_au, note="CRYSTAL wrote no Fermi level (-1.0, SPINLOCK) and the electron count "
                                         "per spin is not known: E_F is not known")
    rng = _BAND_RANGE.search(content)
    first = int(rng.group(1)) if rng else 1
    top, why = derive(blocks, occ, first)
    if top is None:
        return Fermi(None, file_au, note=f"CRYSTAL wrote no Fermi level (-1.0, SPINLOCK) and {why}")
    value = float(file_au) + top
    return Fermi(value, file_au, derived=True,
                 note=f"E_F derived: CRYSTAL wrote none (-1.0: the SCF ended with its spins locked, SPINLOCK); "
                      f"the top of the occupied bands ({occ[0]} alpha, {occ[1]} beta electrons), "
                      f"{value:.6f} Ha")


# ------------------------------------------------------------------ a BAND.DAT
def read_band_dat(path) -> Tuple[Optional[float], List[np.ndarray]]:
    """(the file's EFERMI in Hartree, [one (k, bands) array of E - EFERMI per spin]) of a BAND.DAT."""
    from mace.utils.dat_text import read_text
    text = read_text(path)
    head = re.search(r"NKPT\s+(\d+)\s+NBND\s+(\d+)(?:\s+NSPIN\s+(\d+))?", text)
    ef = _DAT_EF.search(text)
    if not head:
        return (float(ef.group(1)) if ef else None), []
    nk, nb, ns = int(head.group(1)), int(head.group(2)), int(head.group(3) or 1)
    nums = []
    for line in text.splitlines():
        if line and line[0] not in "#@" and line.strip():
            nums.extend(line.split())
    want = nk * ns * (nb + 1)
    if len(nums) < want:
        return (float(ef.group(1)) if ef else None), []
    arr = np.asarray(nums[:want], float).reshape(nk * ns, nb + 1)[:, 1:]
    return (float(ef.group(1)) if ef else None), [arr[s * nk:(s + 1) * nk] for s in range(ns)]


@lru_cache(maxsize=16)
def _band_fermi(path: str, size: int, mtime_ns: int) -> Fermi:
    p = Path(path)
    if p.name.lower().endswith((".f25", "fort.25")):
        from mace.plotting import f25
        bl = [b for b in f25.read_blocks(p, ("BAND",), fermi=False) if b.kind == "BAND"]
        if not bl:
            return Fermi(None, None)
        bl, nspin, _ = f25.select_run(bl, -1, "electronic")
        nseg = len(bl) // max(nspin, 1)
        file_au = bl[0].ef
        if not is_placeholder(file_au):
            return Fermi(file_au, file_au)
        blocks = [np.vstack([np.asarray(b.values, float).reshape(b.ncol, b.nrow) - b.ef
                             for b in bl[s * nseg:(s + 1) * nseg]]) for s in range(nspin)]
    else:
        file_au, blocks = read_band_dat(p)
        if not is_placeholder(file_au):
            return Fermi(file_au, file_au)
    return resolve(file_au, blocks, _sibling_out(p))


def band_fermi(path) -> Fermi:
    """The Fermi level of a BAND.DAT or a BAND fort.25 (see the module docstring); cached per file."""
    p = Path(path)
    try:
        st = os.stat(p)
    except OSError:
        return Fermi(None, None)
    return _band_fermi(os.path.abspath(p), st.st_size, st.st_mtime_ns)


# ------------------------------------------------------------------ a run's Fermi energy
def run_fermi(output_file, printed_au) -> Fermi:
    """The Fermi energy of a run whose .out prints ``printed_au``: that value, unless it is CRYSTAL's
    placeholder; then its band file's derived one (a BAND run), the DOSS.DAT's (a DOSS run: its NEWK
    computes one), else none (the note says why)."""
    if not is_placeholder(printed_au):
        return Fermi(printed_au, printed_au)
    out = Path(output_file)
    stem = out.with_suffix("")
    for cand in (Path(f"{stem}.BAND.DAT"), out.with_name("BAND.DAT"), Path(f"{stem}.f25")):
        if cand.is_file():
            try:
                fl = band_fermi(cand)
            except Exception:           # an unreadable band file: the next one, else no value
                continue
            if fl.value_au is not None or fl.note:
                return fl
    for cand in (Path(f"{stem}.DOSS.DAT"), out.with_name("DOSS.DAT")):
        if cand.is_file():
            from mace.utils.dat_text import read_text
            try:
                found = _DAT_EF.findall(read_text(cand))
            except OSError:
                found = []
            if found and not is_placeholder(found[-1]):
                value = float(found[-1])
                return Fermi(value, printed_au, derived=True,
                             note=f"E_F from {cand.name} (the DOSS run computes it): the .out prints the "
                                  f"wavefunction's -1.0, an SCF that ended with its spins locked (SPINLOCK)")
    return Fermi(None, printed_au, note="CRYSTAL wrote no Fermi level (-1.0: the SCF ended with its spins locked, "
                                        "SPINLOCK) and the run has no band data to derive it from")


# ------------------------------------------------------------------ an SCF parent's Fermi energy
_TITLE = re.compile(r"^ (\S[^\n]*?)\s*\n CRYSTAL - (?:SCF|PROPERTIES) - TYPE OF CALCULATION", re.M)
#: how far from an SCF output its properties runs are looked for: its folder, and the folders beside
#: it and one level in (the example suite's <entry>/SP and <entry>/BAND, a workflow's step_NNN_SP/<m>
#: and step_NNN_DOSS/<m>)
_SIBLING_KINDS = (".DOSS.DAT", ".BAND.DAT")


def _title(text: str) -> Optional[str]:
    m = _TITLE.search(text or "")
    return m.group(1).strip() if m else None


def _base(stem: str) -> str:
    """The name opt2d3 gives an SCF output's decks (x_sp, x_opt -> x)."""
    for suffix in ("_sp", "_SP", "_opt", "_OPT", "_scf", "_SCF"):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[:-len(suffix)]
    return stem


def sibling_property_runs(scf_out) -> List[Path]:
    """BAND.DAT / DOSS.DAT files of properties runs of this SCF: named after it (opt2d3's names) and
    run from a wavefunction with its title, in its folder or in the folders around it."""
    scf_out = Path(scf_out)
    try:
        title = _title(scf_out.read_text(errors="replace"))
    except OSError:
        return []
    if not title:
        return []
    names = {scf_out.stem, _base(scf_out.stem)}
    dirs = [scf_out.parent]
    up = scf_out.parent.parent
    try:
        for d in sorted(up.iterdir())[:200]:
            if d.is_dir() and d != scf_out.parent:
                dirs.append(d)
                dirs += [x for x in sorted(d.iterdir())[:200] if x.is_dir()]
    except OSError:
        pass
    found = []
    for d in dirs:
        for kind in _SIBLING_KINDS:
            for f in sorted(d.glob(f"*{kind}")):
                stem = f.name[: -len(kind)]
                if not any(stem.startswith(n + "_") or stem == n for n in names):
                    continue
                out = f.with_name(stem + ".out")
                try:
                    if out.is_file() and _title(out.read_text(errors="replace")) == title:
                        found.append(f)
                except OSError:
                    continue
    # a DOSS run first: its NEWK computes E_F; then a BAND run's derived one
    return sorted(set(found), key=lambda p: (not p.name.endswith(".DOSS.DAT"), str(p)))


def scf_fermi(scf_out, printed_au: Optional[float]) -> Fermi:
    """The Fermi energy (Hartree) of an SCF output for a properties deck's E_F-relative ranges (a DOSS
    window, TRANSPORT's chemical potentials): the one it prints unless that is missing or CRYSTAL's
    placeholder (an SCF that ended with its spins locked, SPINLOCK); then a properties run of the same
    SCF (a DOSS.DAT's E_F, else a BAND.DAT's, derived as band_fermi does); else none (the note says why)."""
    if printed_au is not None and not is_placeholder(printed_au):
        return Fermi(float(printed_au), float(printed_au))
    for f in sibling_property_runs(scf_out):
        if f.name.endswith(".DOSS.DAT"):
            from mace.utils.dat_text import read_text
            try:
                hits = _DAT_EF.findall(read_text(f))
            except OSError:
                continue
            if hits and not is_placeholder(hits[-1]):
                v = float(hits[-1])
                return Fermi(v, printed_au, derived=True, note=f"E_F from {f.name}, a DOSS run of this SCF")
        else:
            try:
                fl = band_fermi(f)
            except Exception:
                continue
            if fl.value_au is not None:
                return Fermi(fl.value_au, printed_au, derived=True,
                             note=f"E_F from {f.name}, a BAND run of this SCF ({fl.note or 'its own'})")
    try:
        locked = "SPIN LOCKING" in Path(scf_out).read_text(errors="replace")
    except OSError:
        locked = False
    why = ("the SCF ended with its spins locked (SPINLOCK), so CRYSTAL computed no Fermi level"
           if is_placeholder(printed_au) or locked else "")
    return Fermi(None, printed_au,
                 note=f"no trustworthy Fermi energy: {why or 'the output prints none'}, and no BAND or DOSS run of "
                      f"this SCF is beside it to take one from (run BAND or DOSS first)")


# ------------------------------------------------------------------ an SCF's band extents
def sibling_band_extents(scf_out) -> Optional[Tuple[Path, int, List[Tuple[float, float]]]]:
    """(the BAND.DAT, its first band, each band's (lowest, highest) energy in Hartree, absolute, over both
    spins) of a BAND run of this SCF beside it (sibling_property_runs), else None. A DOSS deck's energy
    window is moved into the gaps between them (Crystal_d3/d3_config.doss_snap_window)."""
    for f in sibling_property_runs(scf_out):
        if not f.name.endswith(".BAND.DAT"):
            continue
        try:
            file_au, blocks = read_band_dat(f)
        except Exception:               # an unreadable band file: the next one
            continue
        if file_au is None or not blocks:
            continue
        first = 1
        out = _sibling_out(f)
        if out is not None:
            try:
                rng = _BAND_RANGE.search(out.read_text(errors="replace"))
            except OSError:
                rng = None
            if rng:
                first = int(rng.group(1))
        lo = np.min([b.min(axis=0) for b in blocks], axis=0) + float(file_au)
        hi = np.max([b.max(axis=0) for b in blocks], axis=0) + float(file_au)
        return f, first, [(float(a), float(b)) for a, b in zip(lo, hi)]
    return None
