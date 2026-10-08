"""Which records of a .d12 deck matter to a pre-optimization.

MACE reads a deck's geometry as its asymmetric unit expanded by the space
group (``structures.read_d12``); it does not apply the block-1 records that
edit that geometry (a supercell, removed or substituted atoms, a slab cut,
...) or the basis-set GHOSTS. A pre-optimization of such a deck would act on
the pristine host, not on the structure CRYSTAL builds, so ``mace preopt``
refuses it and the GUI flags it.

``GEOMETRY_EDITS`` is the one table of those records. It was checked
against the CRYSTAL23 manual's list of geometry keywords (pp. 35-37:
"Atoms and cell manipulation", "From crystals to slabs", "From slabs to
nanotubes", "From periodic structures to clusters", "Molecular crystals",
"BSSE correction", "From crystals to nanorods/nanocrystals", "Gibbs-Wulff
construction") plus GHOSTS (basis-set input) and the line-2 geometry kinds
other than CRYSTAL. ``CHARGE_RECORDS`` are the records that make a deck
charged (CHARGED, CHEMOD, DOPING); they are reported, not refused.
"""
from __future__ import annotations

from typing import Dict, List

GEOMETRY_EDITS = (
    # atoms and cell manipulation (manual p.35)
    "ATOMDISP", "ATOMINSE", "ATOMREMO", "ATOMROT", "ATOMSUBS", "ATOMORDE", "ELASTIC", "BREAKELAS",
    "POINTCHG", "SCELCONF", "SCELPHONO", "SUPERCEL", "SUPERCON", "USESAED", "ROTCRY",
    # 3D -> 2D, 2D -> 1D, -> 0D (p.36)
    "SLABINFO", "SLABCUT", "NANOTUBE", "SWCNT", "NANOMULTI", "CLUSTER", "CLUSTSIZE", "FULLE", "HYDROSUB",
    # molecular crystals, BSSE, nanorods, nanocrystals, Wulff (pp.36-37)
    "MOLECULE", "MOLEXP", "MOLSPLIT", "MOLEBSSE", "ATOMBSSE", "NANORODS", "NANOCRYSTAL", "WULFF",
    # basis-set input
    "GHOSTS",
)
CHARGE_RECORDS = ("CHARGED", "CHEMOD", "DOPING")
GEOMETRY_KINDS = ("CRYSTAL", "SLAB", "POLYMER", "HELIX", "MOLECULE", "EXTERNAL", "DLVINPUT")


def _kw(line: str) -> str:
    parts = line.strip().split()
    return parts[0].upper() if parts else ""


def scan(text: str) -> Dict[str, List[str]]:
    """{"kind", "geometry_edits", "charge_records"}: the known keywords found
    after the title (each once, in deck order). A line-2 geometry kind other
    than CRYSTAL is reported as an edit too."""
    lines = text.splitlines()
    kind = _kw(lines[1]) if len(lines) > 1 else ""
    edits: List[str] = []
    charges: List[str] = []
    if kind and kind != "CRYSTAL":
        edits.append(kind if kind in GEOMETRY_KINDS else f"{kind} (line 2)")
    for ln in lines[2:]:
        k = _kw(ln)
        if k in GEOMETRY_EDITS and k not in edits:
            edits.append(k)
        if k in CHARGE_RECORDS and k not in charges:
            charges.append(k)
    return {"kind": kind, "geometry_edits": edits, "charge_records": charges}


def refusal_text(name: str, edits: List[str]) -> str:
    return (f"{name} builds its geometry with {', '.join(edits)}: mace preopt reads only the asymmetric unit, "
            "so it would pre-optimize the pristine host, not this structure. Pre-optimize the expanded "
            "structure instead: open the .out of a run of this deck (or export a CIF from it) and pass that.")


def charged_text(records: List[str], gui: bool = False) -> str:
    return (f"this deck is charged ({'/'.join(records) if records else 'CHARGED/CHEMOD/DOPING'} found): "
            + ("enter the charge" if gui else "pass --charge Q"))


def host_warning(name: str, edits: List[str]) -> str:
    return (f"This document is the host geometry of {name} ({', '.join(edits)}): Clean acts on the host, "
            "not the structure CRYSTAL builds from the deck.")


def valid_meta(value) -> Dict[str, List[str]]:
    """``doc_meta["deck"]`` as a draft sends it back: known keywords only."""
    out = {"geometry_edits": [], "charge_records": []}
    if not isinstance(value, dict):
        return out
    known_e = set(GEOMETRY_EDITS) | set(GEOMETRY_KINDS)
    for key, known in (("geometry_edits", known_e), ("charge_records", set(CHARGE_RECORDS))):
        v = value.get(key)
        if isinstance(v, list):
            out[key] = [x for x in v if isinstance(x, str) and x in known][:40]
    if isinstance(value.get("name"), str):
        out["name"] = value["name"][:120]
    return out
