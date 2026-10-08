"""The pre-optimized CIF, in the input's own space group and setting.

``mace convert`` builds a deck in the CIF's own space group (cif2d12's
``symmetry_handling`` default is "CIF"), so a P1 CIF would make a P1 deck,
and spglib's standardized cell could change the setting. When the result
kept the input's group, ``write`` gives the asymmetric unit (one atom per
site of the reader's site map: ase's ``spacegroup_kinds``) with the input's
symmetry operations written out, then reads the file back with ase and
requires an atom-for-atom match with the result (1e-4 Å). Otherwise, or when
that check fails, it writes P1 and says why.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Optional

import numpy as np

MATCH_TOL = 1e-4


def _xyz(rot, trans) -> str:
    out = []
    for r, t in zip(np.asarray(rot), np.asarray(trans)):
        terms = []
        for c, v in zip(r, "xyz"):
            c = int(round(c))
            if c == 1:
                terms.append(f"+{v}")
            elif c == -1:
                terms.append(f"-{v}")
            elif c:
                terms.append(f"{c:+d}*{v}")
        f = Fraction(float(t) % 1.0).limit_denominator(12)
        if f:
            terms.append(f"+{f}")
        s = "".join(terms)
        out.append(s[1:] if s.startswith("+") else s)
    return ",".join(out)


def _header(lines: List[str]) -> str:
    return "".join(ln.rstrip() + "\n" for ln in lines)


def _cell_block(atoms) -> str:
    a, b, c, al, be, ga = atoms.cell.cellpar()
    return (f"_cell_length_a    {a:.8f}\n_cell_length_b    {b:.8f}\n_cell_length_c    {c:.8f}\n"
            f"_cell_angle_alpha {al:.8f}\n_cell_angle_beta  {be:.8f}\n_cell_angle_gamma {ga:.8f}\n")


def _atoms_block(symbols, frac) -> str:
    out = ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
           "_atom_site_fract_z", "_atom_site_occupancy"]
    count: Dict[str, int] = {}
    for s, f in zip(symbols, frac):
        count[s] = count.get(s, 0) + 1
        out.append(f"{s}{count[s]} {s} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f} 1.0")
    return "\n".join(out) + "\n"


def _render(atoms, name, header, sg, symbols, frac) -> str:
    text = _header(header) + f"data_{name}\n" + _cell_block(atoms)
    if sg is None:
        text += ("_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n"
                 "_space_group_name_H-M_alt 'P 1'\n_space_group_IT_number 1\n\nloop_\n"
                 "_space_group_symop_operation_xyz\n  'x,y,z'\n\n")
    else:
        sym = str(sg.symbol)
        text += (f"_symmetry_space_group_name_H-M '{sym}'\n_symmetry_Int_Tables_number {int(sg.no)}\n"
                 f"_space_group_name_H-M_alt '{sym}'\n_space_group_IT_number {int(sg.no)}\n\nloop_\n"
                 "_space_group_symop_operation_xyz\n")
        text += "".join(f"  '{_xyz(r, t)}'\n" for r, t in sg.get_symop()) + "\n"
    return text + _atoms_block(symbols, frac)


def _same(written_path, atoms) -> bool:
    import warnings
    import ase.io
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = ase.io.read(str(written_path), format="cif")
    if len(back) != len(atoms):
        return False
    if np.abs(back.cell.cellpar() - atoms.cell.cellpar()).max() > 1e-5:
        return False
    cell = atoms.cell.array
    fa = atoms.get_scaled_positions(wrap=True)
    fb = back.get_scaled_positions(wrap=True)
    used = np.zeros(len(atoms), bool)
    for i in range(len(back)):
        d = fa - fb[i]
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=1)
        dist[(atoms.numbers != back.numbers[i]) | used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] > MATCH_TOL:
            return False
        used[j] = True
    return True


def write(path, atoms, *, name: str, header: List[str], spacegroup=None, kinds=None,
          keep_group: bool = True, p1_reason: Optional[str] = None) -> Dict[str, Any]:
    """Write ``atoms`` to ``path``; returns {"mode": "input_group"|"p1", "reason", "group"}."""
    from pathlib import Path
    path = Path(path)
    reason = p1_reason
    if reason is None:
        if spacegroup is None or kinds is None:
            reason = "the input's space group and site map are not known (a P1 input?)"
        elif int(spacegroup.no) == 1:
            reason = "the input is P1"
        elif not keep_group:
            reason = "the space group changed (--no-symmetry)"
        elif len(kinds) != len(atoms):
            reason = "the atom count changed"
    if reason is None:
        kinds = np.asarray(kinds)
        first = [int(np.flatnonzero(kinds == k)[0]) for k in sorted(set(kinds.tolist()))]
        frac = atoms.get_scaled_positions(wrap=True)[first]
        syms = [atoms.get_chemical_symbols()[i] for i in first]
        path.write_text(_render(atoms, name, header, spacegroup, syms, frac))
        try:
            ok = _same(path, atoms)
        except Exception:
            ok = False
        if ok:
            return {"mode": "input_group", "reason": None, "group": f"{spacegroup.no} {spacegroup.symbol}"}
        reason = "re-expanding the input's space group did not give back every atom"
    path.write_text(_render(atoms, name, header, None, atoms.get_chemical_symbols(),
                            atoms.get_scaled_positions(wrap=True)))
    return {"mode": "p1", "reason": reason, "group": "1 P 1"}
