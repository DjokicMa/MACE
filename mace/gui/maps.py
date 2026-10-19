"""2D maps from a properties run's fort.25 (plan E6): ECHG charge (and spin)
density, POTM and CLAS potentials, written as "MAPN" blocks.

The layout is CRYSTAL's (manual, Appendix D, "CLAS - ECHG - POTM - EMDPDM"),
in fixed-width fields (numbers can touch: "1.10358E-03-8.01115E-02"):

    -%-,IHFERM,TYPE,NROW,NCOL,DX,DY,COSXY     A3,I1,A4,2I5,3E12.5
    XA,YA,ZA,XB,YB,ZB                         6E12.5   (Angstrom)
    XC,YC,ZC,NAF,LDIM                         3E12.5,4X,2I4
    ((RDAT(I,J),I=1,NROW),J=1,NCOL)           6E12.5 per line
    NAF lines: NAT,SYMBAT,XA,YA,ZA            I4,1X,A,1P,3E20.12
    3 lines: the cell vectors                  3E20.12

A spin-polarized ECHG writes a second block (the spin density) after the
first; POTM and CLAS write potentials in the same layout, so what a block
holds comes from the properties output beside it. BAND and DOSS blocks are
not maps and are skipped.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List

_E12 = 12


def _fixed(line: str, width: int, n: int) -> List[float]:
    vals = []
    for k in range(n):
        s = line[k * width:(k + 1) * width].strip()
        if not s:
            break
        vals.append(float(s))
    return vals


def read_f25(path: Path, max_values: int = 4_000_000) -> List[Dict[str, Any]]:
    """Every MAPN block of a fort.25, as {nrow, ncol, dx, dy, cosxy, A, B, C, ldim, values (ncol lists of
    nrow), atoms [{z, symbol, cart}], cell}."""
    lines = Path(path).read_text(errors="replace").splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if not ln.startswith("-%-"):
            i += 1
            continue
        kind = ln[4:8]
        try:
            nrow, ncol = int(ln[8:13]), int(ln[13:18])
        except ValueError:
            i += 1
            continue
        if kind != "MAPN":
            i += 1
            continue
        dx, dy, cosxy = _fixed(ln[18:], _E12, 3)
        ab = _fixed(lines[i + 1], _E12, 6)
        c = _fixed(lines[i + 2], _E12, 3)
        tail = lines[i + 2][36:].split()
        naf, ldim = (int(tail[0]), int(tail[1])) if len(tail) >= 2 else (0, 3)
        total = nrow * ncol
        if total > max_values:
            raise ValueError(f"a {nrow} x {ncol} map is too large to draw here")
        vals: List[float] = []
        j = i + 3
        while len(vals) < total and j < n:
            vals.extend(_fixed(lines[j], _E12, 6))
            j += 1
        if len(vals) < total:
            raise ValueError(f"the map stops after {len(vals)} of {total} values (a run still writing it?)")
        atoms = []
        for k in range(naf):
            if j + k >= n:
                break
            row = lines[j + k]
            m = re.match(r"\s*(\d+)\s+(\S+)", row)
            nums = re.findall(r"-?\d+\.\d+E[-+]\d+", row)
            if m and len(nums) >= 3:
                atoms.append({"z": int(m.group(1)) % 100, "symbol": m.group(2), "cart": [float(x) for x in nums[:3]]})
        j += naf
        cell = []
        for k in range(3):
            if j + k < n:
                nums = re.findall(r"-?\d+\.\d+E[-+]\d+", lines[j + k])
                if len(nums) == 3:
                    cell.append([float(x) for x in nums])
        j += len(cell)
        grid = [vals[col * nrow:(col + 1) * nrow] for col in range(ncol)]
        blocks.append({"nrow": nrow, "ncol": ncol, "dx": dx, "dy": dy, "cosxy": cosxy, "A": ab[:3], "B": ab[3:6], "C": c,
                       "ldim": ldim, "values": grid, "atoms": atoms, "cell": cell, "ihferm": ln[3]})
        i = j
    return blocks


_KEYWORD_LABEL = {"POTM": ("Electrostatic potential", "hartree/e"), "CLAS": ("Classical potential", "hartree/e")}


def map_kinds(f25: Path) -> List[Dict[str, str]]:
    """What each MAPN block holds, in order, from the properties output beside
    the fort.25 (<stem>.out): CRYSTAL echoes every block's header line there and
    closes each keyword with its "TTTT... ECHG / POTM / CLAS" timing line; for
    ECHG the "ALPHA+BETA" (charge) or "ALPHA-BETA" (spin) line before the
    header says which density. [] when there is no such output."""
    name = Path(f25).name
    stem = name[:-4] if name.lower().endswith(".f25") else None
    if not stem:
        return []
    out = Path(f25).with_name(stem + ".out")
    if not out.is_file() or out.stat().st_size > 64 * 1024 * 1024:
        return []
    slots: List[Dict[str, str]] = []
    open_from = 0
    density = ""
    with open(out, "r", errors="replace") as fh:
        for ln in fh:
            if "ALPHA+BETA ELECTRONS DENSITY" in ln:
                density = "charge"
            elif "ALPHA-BETA ELECTRONS DENSITY" in ln:
                density = "spin"
            elif ln.startswith("-%-") and ln[4:8] == "MAPN":
                slots.append({"keyword": "", "density": density})
                density = ""
            elif ln.lstrip().startswith("TTTTTTTTTT"):
                parts = ln.split()
                key = parts[1] if len(parts) > 1 else ""
                for s in slots[open_from:]:
                    s["keyword"] = key
                open_from = len(slots)
    return slots


def _label(kinds: List[Dict[str, str]], k: int, n: int):
    if k < len(kinds):
        kw = kinds[k]["keyword"]
        if kw == "ECHG":
            return ("Spin density", "e/bohr³") if kinds[k]["density"] == "spin" else ("Charge density", "e/bohr³")
        if kw in _KEYWORD_LABEL:
            return _KEYWORD_LABEL[kw]
    # no properties output beside it: CRYSTAL's fort.25 does not say what a map is
    return ("2D map" if n == 1 else f"Map {k + 1}"), "a.u."


def map_payload(path: Path) -> Dict[str, Any]:
    """The plot's data: each map's values on its in-plane axes (x along the rows' direction, y at the
    angle COSXY gives), the corner labels A, B, C and the extremes. Each map is named by the
    properties output beside it (map_kinds); without one it is "Map k"."""
    blocks = read_f25(path)
    if not blocks:
        raise ValueError(f"{Path(path).name} holds no 2D map (MAPN block)")
    kinds = map_kinds(path)
    out = []
    for k, b in enumerate(blocks):
        flat = [v for col in b["values"] for v in col]
        sin = math.sqrt(max(0.0, 1 - b["cosxy"] ** 2))
        label, unit = _label(kinds, k, len(blocks))
        # POTM / CLAS: drawn in the Potential colors (chemistry: red = negative)
        potential = k < len(kinds) and kinds[k]["keyword"] in _KEYWORD_LABEL
        out.append({"label": label, "unit": unit, "potential": potential,
                    "nrow": b["nrow"], "ncol": b["ncol"], "dx": b["dx"], "dy": b["dy"], "cosxy": b["cosxy"],
                    "x": [i * b["dx"] for i in range(b["nrow"])], "y": [j * b["dy"] * sin for j in range(b["ncol"])],
                    "values": b["values"], "min": min(flat), "max": max(flat), "A": b["A"], "B": b["B"], "C": b["C"],
                    "ldim": b["ldim"]})
    units = {m["unit"] for m in out}
    return {"name": Path(path).name, "maps": out, "unit": units.pop() if len(units) == 1 else "a.u.",
            "named_by": "output" if kinds else None}
