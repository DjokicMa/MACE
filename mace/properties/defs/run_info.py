"""The run card's values (the GUI's run details card, mace/gui/runinfo.py):
energy per atom, functional, basis set, CRYSTAL version and MPI processes.
The card reads them through these functions, and the extractor stores them,
so the card and the database say the same."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from mace.properties.registry import ALL, Output, Sibling, derived

from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402

_VERSION = re.compile(r"\*\s+(CRYSTAL\d{2})\s+\*")
_RELEASE = re.compile(r"\*\s+(public\s*:\s*[^*]+?)\s*\*")
_PROCS = re.compile(r"PROCESS\s+\d+\s+OF\s+(\d+)\s+WORKING")
_HEAD = 400_000          # the header: version, release, processes, the energy expression


#: DFT-block keywords that set the grid, the spin or a parameter, not the functional
_DFT_MODIFIERS = {"SPIN", "XLGRID", "LGRID", "XXLGRID", "OLDGRID", "RADIAL", "ANGULAR", "HYBRID", "NONLOCAL",
                  "TOLLGRID", "TOLLDENS", "LIMBEK", "BECKE", "SAVIN", "RADSAFE", "CHUNKS", "DISTGRID", "BATCHPNT",
                  "FCHARGE", "SMOOTH", "SPINLOCK"}


def _number(r: str) -> bool:
    try:
        [float(x.replace("D", "E")) for x in r.split()]
        return bool(r.split())
    except ValueError:
        return False


def _dft_block(recs) -> Optional[str]:
    """The functional of a deck's DFT block: its keyword (B3LYP, HSESOL3C,
    PBE0-D3 ...), else its EXCHANGE/CORRELAT pair as "X/C" ("PBE/PBE",
    "LDA/VWN"; an exchange-only functional "X/NONE"), with " HYBRID n" when
    the block mixes in n % of Hartree-Fock exchange."""
    i = recs.index("DFT")
    xc = {}
    hybrid = None
    named = None
    k = i + 1
    while k < len(recs):
        r = recs[k]
        if r in ("ENDDFT", "END"):
            break
        if r in ("EXCHANGE", "CORRELAT") and k + 1 < len(recs):
            xc[r] = recs[k + 1].split()[0] if recs[k + 1].split() else None
            k += 2
            continue
        if r == "HYBRID" and k + 1 < len(recs) and _number(recs[k + 1]):
            hybrid = recs[k + 1].split()[0]
            k += 2
            continue
        if r and named is None and r not in _DFT_MODIFIERS and not _number(r):
            named = r
        k += 1
    if named:
        return named
    if xc:
        out = f"{xc.get('EXCHANGE') or 'NONE'}/{xc.get('CORRELAT') or 'NONE'}"
        if hybrid is not None:
            try:
                hybrid = f"{float(hybrid):g}"
            except ValueError:
                pass
            out += f" HYBRID {hybrid}"
        return out
    return None


def functional(text: str, p: Path) -> Optional[str]:
    """The DFT functional of the run's deck (``<stem>.d12`` beside it): its
    keyword, or its EXCHANGE/CORRELAT pair ("PBE/PBE"); HF; else CRYSTAL's
    ENERGY EXPRESSION line."""
    d12 = Path(p).with_suffix(".d12")
    if d12.is_file():
        recs = [ln.strip().upper() for ln in d12.read_text(errors="replace").splitlines()[1:]]
        if "DFT" in recs:
            fn = _dft_block(recs)
            if fn:
                return fn
        if "HF" in recs or "RHF" in recs or "UHF" in recs:
            return "HF"
    m = re.search(r"ENERGY EXPRESSION=(.+)", text[:_HEAD])
    return m.group(1).strip()[:80] if m else None


def basis(p: Path) -> Optional[str]:
    """The BASISSET of the run's deck, "custom (in the deck)", or None (no deck)."""
    d12 = Path(p).with_suffix(".d12")
    if not d12.is_file():
        return None
    recs = [ln.strip() for ln in d12.read_text(errors="replace").splitlines()[1:]]
    up = [r.upper() for r in recs]
    if "BASISSET" in up:
        i = up.index("BASISSET")
        return recs[i + 1] if i + 1 < len(recs) else None
    return "custom (in the deck)"


def crystal_version(text: str) -> Optional[str]:
    """"CRYSTAL23 public : 1.0.1" from the output's banner."""
    head = text[:_HEAD]
    parts = [m.group(1).strip() for m in (_VERSION.search(head), _RELEASE.search(head)) if m]
    return " ".join(parts) or None


def mpi_processes(text: str) -> Optional[int]:
    procs = _PROCS.findall(text[:_HEAD])
    return max(int(v) for v in procs) if procs else None


def energy_per_atom_ev(energy_au, n_atoms) -> Optional[float]:
    """The SCF total energy (TOTAL ENERGY, without D3 and gCP) per atom of the cell, in eV."""
    if energy_au is None or not n_atoms:
        return None
    return float(energy_au) * HARTREE_EV / int(n_atoms)


@derived("energy_per_atom", calc_types=(ALL,), category="energy", group="Energy",
         requires=("total_energy_au", "atoms_in_unit_cell"), version="1",
         outputs=(Output("energy_per_atom_ev", "eV", label="E<sub>total</sub>/atom (without D3 and gCP)", precision=8,
                         description="total_energy_au / atoms_in_unit_cell, in eV: the SCF energy per atom without "
                                     "the D3 and gCP corrections (the GUI's run card). The corrected energy per atom "
                                     "is total_energy_corrected_au shown per atom (the Properties tab's Normalize)"),),
         source="total_energy_au and atoms_in_unit_cell of the .out")
def _energy_per_atom(ctx):
    """Total energy per atom of the cell."""
    return {"energy_per_atom_ev": energy_per_atom_ev(ctx.value("total_energy_au"), ctx.value("atoms_in_unit_cell"))}


@derived("run_info", calc_types=(ALL,), category="computational", group="Run",
         siblings=(Sibling("d12", required=False, description="the deck: functional and basis set"),), version="1",
         outputs=(Output("dft_functional", "category", "text", label="Functional",
                         description="DFT functional of the deck (HF; else CRYSTAL's ENERGY EXPRESSION)"),
                  Output("basis_set", "category", "text", label="Basis set",
                         description="BASISSET of the deck, or 'custom (in the deck)'"),
                  Output("crystal_version", "category", "text", label="CRYSTAL version",
                         description="CRYSTAL version and release of the output's banner"),
                  Output("mpi_processes", "dimensionless", "int", label="MPI processes", precision=6,
                         description="MPI processes the run used (PROCESS i OF N WORKING)")),
         source="the .out banner and the .d12 beside it (the GUI's run card)")
def _run_info(ctx):
    """The run card's settings: functional, basis set, CRYSTAL version, MPI processes."""
    return {"dft_functional": functional(ctx.content, ctx.output_file), "basis_set": basis(ctx.output_file),
            "crystal_version": crystal_version(ctx.content), "mpi_processes": mpi_processes(ctx.content)}
