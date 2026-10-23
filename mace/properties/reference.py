"""Reference energies: what the properties that combine several runs share
(mace/properties/defs/reference_energies.py; the GUI's Add Reference…).

* :class:`Part`: one run taking part (the run itself or a reference): its
  energy (``total_energy_corrected_au``, the energy with D3 and gCP, else
  ``total_energy_au``), composition, dimensionality, cell size, settings
  (functional, basis set, CRYSTAL version, spin, D3, gCP) and its FREQ
  thermodynamics (its own, or a FREQ run of the same material in the
  database);
* :func:`check` and :func:`guard`: whether runs can be combined (what
  refuses, what only warns: docs/properties.md, "Reference energies");
* :func:`candidates`: the database's runs a lookup chooses from;
* :func:`forms`: ΔE, and ΔH and ΔG when every run has FREQ thermodynamics
  at one temperature (and the ΔH, ΔG curves when they share several);
* :func:`balance`: the stoichiometric coefficients of a reaction.

Standard library only.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: CODATA 2018: 1 Hartree in eV and in J; 1 eV in kJ/mol
from mace.constants import EV_TO_KJ_MOL as EV_KJ_MOL, HARTREE_TO_EV as HARTREE_EV, HARTREE_TO_J as HARTREE_J  # noqa: E402
#: the largest relative difference of two cells (a slab's area, a polymer's length) that are "the same cell"
CELL_TOL = 0.01
#: the energies a reference energy reads, in this order
ENERGY_KEYS = ("total_energy_corrected_au", "total_energy_au")
#: the calculation types a reference run can be (a standalone CPHF run is an SCF first:
#: its energy, cell and populations are an SP's)
REF_TYPES = ("OPT", "SP", "FREQ", "CPHF")
CUSTOM_BASIS = "custom (in the deck)"

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
_ELEMENTS = set(_SYMBOLS)

#: the composition's fallbacks: read only when the positions do not give it
_FALLBACK = ("initial_atomic_positions", "processed_atomic_charges")
#: the stored keys a reference needs (candidates() reads only these)
LIGHT = ("calculation_type", "total_energy_au", "total_energy_corrected_au", "d3_dispersion_energy_au",
         "gcp_energy_au", "atoms_in_unit_cell", "dft_functional", "basis_set", "crystal_version",
         "spin_polarized", "primitive_cell_area", "primitive_cell_volume", "primitive_cell_length",
         "initial_primitive_cell_area", "initial_primitive_cell_volume", "initial_primitive_cell_length",
         "atomic_positions", "initial_atomic_positions", "processed_atomic_charges", "gibbs_free_energy_au",
         "enthalpy_au", "electronic_energy_au", "thermodynamic_temperature_k", "thermodynamic_pressure_mpa",
         "thermodynamics_vs_temperature")


# ------------------------------------------------------------------ composition
def symbol(s: Any) -> Optional[str]:
    """An element symbol as written ("AG", "ag", "Ag") -> "Ag"; None for a ghost or anything else."""
    if s is None:
        return None
    t = str(s).strip()
    t = t[:1].upper() + t[1:].lower()
    return t if t in _ELEMENTS else None


def _from_z(z: Any) -> Optional[str]:
    try:
        z = int(z) % 100          # CRYSTAL adds 200 (and more) for an ECP
    except (TypeError, ValueError):
        return None
    return _SYMBOLS[z - 1] if 1 <= z <= len(_SYMBOLS) else None


_CELL_TABLE = re.compile(r"ATOMS IN THE UNIT CELL:\s*(\d+)\s*\n[^\n]*\n\s*\*+\s*\n"
                         r"((?:\s*\d+\s+[TF]\s+\d+\s+[A-Za-z]{1,2}\b[^\n]*\n)+)")
_CELL_ROW = re.compile(r"^\s*\d+\s+[TF]\s+(\d+)\s+([A-Za-z]{1,2})\b", re.MULTILINE)


def composition(props: Dict[str, Any], text: Optional[str] = None) -> Optional[Counter]:
    """{element: atoms in the cell} of a run: its atomic positions (final,
    else input), else its Mulliken table, else (``text``) the last ATOMS IN
    THE UNIT CELL table of the output. A list whose count is not
    atoms_in_unit_cell is not used."""
    n = props.get("atoms_in_unit_cell")
    for key in ("atomic_positions", "initial_atomic_positions", "processed_atomic_charges"):
        rows = props.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        c: Counter = Counter()
        for a in rows:
            if isinstance(a, dict):
                el = symbol(a.get("element")) or _from_z(a.get("atomic_number"))
                if el:
                    c[el] += 1
        if c and (not n or sum(c.values()) == int(n)):
            return c
    if text:
        tables = list(_CELL_TABLE.finditer(text))
        if tables:
            c = Counter()
            for z, sym in _CELL_ROW.findall(tables[-1].group(2)):
                el = symbol(sym) or _from_z(z)
                if el:
                    c[el] += 1
            if c and (not n or sum(c.values()) == int(n)):
                return c
    return None


def parse_formula(f: Optional[str]) -> Optional[Counter]:
    """"C3H4O3" -> Counter; None when it is not a formula ("Unknown")."""
    if not f:
        return None
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", str(f))
    if not parts or "".join(a + b for a, b in parts) != str(f).strip():
        return None
    c: Counter = Counter()
    for el, n in parts:
        if el not in _ELEMENTS:
            return None
        c[el] += int(n) if n else 1
    return c


def formula(comp: Optional[Counter]) -> str:
    """Hill order: C, H, then alphabetical (alphabetical without C)."""
    if not comp:
        return "?"
    els = sorted(comp)
    if "C" in comp:
        els = ["C"] + (["H"] if "H" in comp else []) + [e for e in els if e not in ("C", "H")]
    return "".join(f"{e}{comp[e] if comp[e] != 1 else ''}" for e in els)


def reduced(comp: Counter) -> Counter:
    g = 0
    for v in comp.values():
        g = math.gcd(g, int(v))
    return Counter({k: v // g for k, v in comp.items()}) if g else Counter(comp)


def ratio(a: Counter, b: Counter) -> Optional[float]:
    """n with a = n·b (the same elements in the same proportions), else None."""
    if not a or not b or set(a) != set(b):
        return None
    rs = [a[e] / b[e] for e in a]
    return rs[0] if all(abs(r - rs[0]) <= 1e-9 * max(1.0, rs[0]) for r in rs) else None


def dimensionality(props: Dict[str, Any]) -> Optional[int]:
    """3, 2, 1 or 0 from the cell the extractor read (volume, slab area, polymer length, none)."""
    for key, d in (("primitive_cell_volume", 3), ("primitive_cell_area", 2), ("primitive_cell_length", 1)):
        if props.get(key):
            return d
    return 0 if props.get("atoms_in_unit_cell") else None


_CELL_KEY = {3: "primitive_cell_volume", 2: "primitive_cell_area", 1: "primitive_cell_length"}
#: CRYSTAL's SUPERCELL / SCELPHONO (a phonon supercell) expansion of the input cell, printed before the SCF
_EXPANSION = re.compile(r"EXPANSION MATRIX OF PRIMITIVE CELL\s*\n((?:[ \t]*E[123](?:[ \t]+-?\d+\.\d+){3}[ \t]*\n){3})")
#: how much of an output a reference run's expansion is looked for in (it is in the geometry input)
_HEAD = 2_000_000


def supercell_multiple(text: str, dim: Optional[int]) -> int:
    """How many input cells the cell of a run's SCF holds: the determinant
    of the periodic part of its expansion matrix (SUPERCELL, or SCELPHONO
    for a phonon dispersion or DOS run), 1 without one. The atoms and the
    energy are the supercell's; the cell the extractor reads (the first one
    printed) is the input cell's, and so are a SCELPHONO run's FREQ
    thermodynamics (thermo_cells)."""
    m = _EXPANSION.search(text or "")
    if not m or not dim:
        return 1
    rows = [[float(v) for v in ln.split()[1:4]] for ln in m.group(1).strip().splitlines()]
    a = [r[:dim] for r in rows[:dim]]
    if dim == 1:
        det = a[0][0]
    elif dim == 2:
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    else:
        det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
               + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    n = int(round(abs(det)))
    return n if n >= 1 and abs(abs(det) - n) < 1e-6 else 1


# ------------------------------------------------------------------ a run taking part
class Part:
    """One run of a reference energy: the run itself (``of_run``) or a
    reference (``of_ref``)."""

    def __init__(self, label: str, props: Dict[str, Any], *, role: str = "run", calc_id: Optional[str] = None,
                 material_id: Optional[str] = None, output_file: Optional[Path] = None,
                 weight: Optional[float] = None, via: str = "", text: Optional[str] = None,
                 dim: Optional[int] = None, calc_type: Optional[str] = None):
        self.label, self.props, self.role = label, props or {}, role
        self.calc_id, self.material_id, self.output_file = calc_id, material_id, output_file
        self.weight, self.via, self._text, self._dim = weight, via, text, dim
        self.calc_type = calc_type or self.props.get("calculation_type")
        self._comp = self._settings = self._multiple = None
        self._comp_done = False

    @classmethod
    def of_run(cls, ctx) -> "Part":
        return cls(Path(ctx.output_file).name, ctx.properties, role="run", calc_id=ctx.calc_id,
                   material_id=ctx.material_id, output_file=Path(ctx.output_file), via="the run",
                   text=ctx.content, dim=ctx.dimensionality, calc_type=ctx.calc_type)

    @classmethod
    def of_ref(cls, ref, role: str, conn=None) -> "Part":
        mid, out, ct = ref.material_id, ref.output_file, ref.calc_type
        if ref.calc_id and conn is not None and (mid is None or out is None):
            row = calc_row(conn, ref.calc_id)
            if row:
                mid = mid or row[0]
                out = out or (Path(row[1]) if row[1] else None)
                ct = ct or row[2]
        label = Path(out).name if out else ref.describe()
        return cls(label, ref.properties, role=role, calc_id=ref.calc_id, material_id=mid,
                   output_file=Path(out) if out else None, weight=ref.weight, via=ref.via, calc_type=ct)

    def as_ref(self):
        from mace.properties.registry import RunRef
        return RunRef(output_file=self.output_file, material_id=self.material_id, calc_id=self.calc_id,
                      calc_type=self.calc_type, properties=self.props, via="database")

    # ---------------------------------------------------------------- what it is
    def text(self) -> str:
        if self._text is None:
            self._text = ""
            if self.output_file and Path(self.output_file).is_file():
                try:
                    self._text = Path(self.output_file).read_text(errors="replace")
                except OSError:
                    pass
        return self._text

    @property
    def comp(self) -> Optional[Counter]:
        if not self._comp_done:
            self._comp_done = True
            self._comp = composition(self.props)
            if self._comp is None and self.via != "database":
                # the output's own atom table (a lookup's candidates use their stored rows only)
                self._comp = composition(self.props, self.text())
        return self._comp

    @property
    def atoms(self) -> int:
        c = self.comp
        return int(sum(c.values())) if c else int(self.props.get("atoms_in_unit_cell") or 0)

    @property
    def dim(self) -> Optional[int]:
        return self._dim if self._dim is not None else dimensionality(self.props)

    @property
    def multiple(self) -> int:
        """Input cells in the cell of the run's SCF (supercell_multiple)."""
        if self._multiple is None:
            head = self._text
            if head is None and self.output_file and Path(self.output_file).is_file():
                try:
                    with open(self.output_file, encoding="utf-8", errors="replace") as fh:
                        head = fh.read(_HEAD)
                except OSError:
                    head = ""
            self._multiple = supercell_multiple(head or "", self.dim)
        return self._multiple

    @property
    def cell(self) -> Optional[float]:
        """The slab's area (Å²), the polymer's length (Å) or the volume (Å³)
        of the cell its atoms and energy are of: a supercell's (the input
        cell the extractor read times ``multiple``; an optimization's final
        cell is already the supercell's)."""
        k = _CELL_KEY.get(self.dim or 0)
        v = self.props.get(k) if k else None
        if not v:
            return None
        first = self.props.get("initial_" + k)
        final = first is not None and abs(float(first) - float(v)) > 1e-9 * abs(float(v))
        return float(v) * (1 if final else self.multiple)

    @property
    def energy(self) -> Optional[float]:
        for k in ENERGY_KEYS:
            if self.props.get(k) is not None:
                return float(self.props[k])
        return None

    @property
    def energy_key(self) -> Optional[str]:
        return next((k for k in ENERGY_KEYS if self.props.get(k) is not None), None)

    @property
    def settings(self) -> Dict[str, Any]:
        if self._settings is None:
            p = self.props
            fn, bs, ver = p.get("dft_functional"), p.get("basis_set"), p.get("crystal_version")
            if (fn is None or bs is None or ver is None) and self.output_file and Path(self.output_file).is_file():
                # a file the extractor read alone (no derived properties): the run card's own functions
                from mace.properties.defs import run_info
                head = self.text()[:400_000]
                fn = fn or run_info.functional(head, Path(self.output_file))
                bs = bs or run_info.basis(Path(self.output_file))
                ver = ver or run_info.crystal_version(head)
            self._settings = {"functional": fn, "basis_set": bs, "crystal_version": ver,
                              "spin_polarized": p.get("spin_polarized"),
                              "d3": p.get("d3_dispersion_energy_au") is not None,
                              "gcp": p.get("gcp_energy_au") is not None,
                              "components": p.get("total_energy_au") is not None}
        return self._settings

    def info(self) -> Dict[str, Any]:
        """What the property's JSON says about the run."""
        s = self.settings
        out = {"role": self.role, "run": self.label, "via": self.via, "calc_type": self.calc_type,
               "composition": formula(self.comp), "atoms": self.atoms, "energy_au": self.energy,
               "energy_key": self.energy_key, "functional": s["functional"], "basis_set": s["basis_set"]}
        if self.calc_id:
            out["calc_id"] = self.calc_id
        if self.output_file and self.role != "run":
            out["file"] = str(self.output_file)
        if self.weight is not None:
            out["weight"] = self.weight
        if self.cell:
            out["cell"] = self.cell
        return out


def calc_row(conn, calc_id: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """(material_id, output_file, calc_type) of a calculation, or None."""
    try:
        return conn.execute("SELECT material_id, output_file, calc_type FROM calculations WHERE calc_id = ?",
                            (calc_id,)).fetchone()
    except sqlite3.Error:
        return None


def tracked_calc(conn, path: Path) -> Optional[str]:
    """The calc_id recorded for exactly this output (its path in any
    spelling, or its deck's name in its folder); never a guess."""
    from mace.utils import db_build
    path = Path(path)
    outs = db_build.spellings(path)
    r = conn.execute(f"SELECT calc_id FROM calculations WHERE output_file IN ({','.join('?' * len(outs))}) "
                     "ORDER BY created_at DESC LIMIT 1", outs).fetchone()
    if r:
        return r[0]
    dirs = db_build.spellings(path.parent)
    for cid, inp in conn.execute(f"SELECT calc_id, input_file FROM calculations WHERE work_dir IN "
                                 f"({','.join('?' * len(dirs))}) ORDER BY created_at DESC", dirs):
        if inp and Path(inp).stem == path.stem:
            return cid
    return None


# ------------------------------------------------------------------ may they be combined?
#: CRYSTAL's shorthand keywords that are one EXCHANGE/CORRELAT pair (run_info.functional's "X/C")
_FN_SAME = {"PBEXC": "PBE/PBE", "PBESOLXC": "PBESOL/PBESOL", "SVWN": "LDA/VWN", "BLYP": "BECKE/LYP"}


def _fn_norm(f: str) -> Tuple[str, bool]:
    s = " ".join(str(f).upper().split())
    s = _FN_SAME.get(s, s)
    return s, bool(re.search(r"[*+]", s))          # True: CRYSTAL's ENERGY EXPRESSION, not a keyword


def _major(v: str) -> Optional[str]:
    m = re.search(r"CRYSTAL\s*(\d{2})", str(v).upper())
    return m.group(1) if m else None


def check(a: Part, b: Part) -> Tuple[List[str], List[str]]:
    """(refusals, warnings) of combining ``a`` and ``b``.

    Refused: different functionals (both named the same way), different
    named basis sets, different CRYSTAL versions (CRYSTAL17 and CRYSTAL23),
    D3 or gCP in one run only (both printing their energy terms), and a run
    with no energy. Warned: a setting one run does not say (no deck beside
    it), a basis set written in the deck (not compared), a functional named
    as a keyword in one run and as CRYSTAL's energy expression in the other,
    another release of the same CRYSTAL version, and spin-polarized against
    closed shell (valid only when the closed-shell run has no unpaired
    electrons)."""
    refuse: List[str] = []
    warn: List[str] = []
    sa, sb = a.settings, b.settings
    if b.energy is None:
        refuse.append(f"{b.label} has no energy (not in the database, or not a finished SCF run)")
        return refuse, warn
    fa, fb = sa["functional"], sb["functional"]
    if fa and fb:
        (na, ea), (nb, eb) = _fn_norm(fa), _fn_norm(fb)
        if ea == eb and na != nb:
            refuse.append(f"the functional differs: {a.label} {fa}, {b.label} {fb}")
        elif ea != eb:
            warn.append(f"the functional is not compared: {a.label} {fa}, {b.label} {fb} "
                        "(one named by its deck keyword, the other by CRYSTAL's energy expression)")
    else:
        warn.append(f"the functional of {(b if fa else a).label} is not known (no deck beside the output)")
    ba, bb = sa["basis_set"], sb["basis_set"]
    if ba and bb:
        if CUSTOM_BASIS in (ba, bb):
            if ba == bb:
                warn.append(f"{a.label} and {b.label} write their basis sets in the deck: not compared")
            else:
                warn.append(f"the basis set is not compared: {a.label} {ba}, {b.label} {bb}")
        elif ba.strip().upper() != bb.strip().upper():
            refuse.append(f"the basis set differs: {a.label} {ba}, {b.label} {bb}")
    else:
        warn.append(f"the basis set of {(b if ba else a).label} is not known (no deck beside the output)")
    va, vb = sa["crystal_version"], sb["crystal_version"]
    if va and vb:
        ma, mb = _major(va), _major(vb)
        if ma and mb and ma != mb:
            refuse.append(f"the CRYSTAL version differs: {a.label} CRYSTAL{ma}, {b.label} CRYSTAL{mb}")
        elif va != vb:
            warn.append(f"another CRYSTAL release: {a.label} {va}, {b.label} {vb}")
    else:
        warn.append(f"the CRYSTAL version of {(b if va else a).label} is not known")
    if sa["components"] and sb["components"]:
        for k, what in (("d3", "D3 dispersion"), ("gcp", "the gCP correction")):
            if sa[k] != sb[k]:
                has, lacks = (a, b) if sa[k] else (b, a)
                refuse.append(f"{what} is in {has.label} and not in {lacks.label}")
    pa, pb = sa["spin_polarized"], sb["spin_polarized"]
    if pa is not None and pb is not None and bool(pa) != bool(pb):
        op, cl = (a, b) if pa else (b, a)
        warn.append(f"{op.label} is spin-polarized and {cl.label} is not (valid only when {cl.label} "
                    "has no unpaired electrons)")
    return refuse, warn


def guard(parts: Sequence[Part]) -> List[str]:
    """Check every pair of runs; raise Skip naming each refusal, else the
    warnings (each once)."""
    from mace.properties.registry import Skip
    refuse: List[str] = []
    warn: List[str] = []
    for i, a in enumerate(parts):
        if a.energy is None:
            refuse.append(f"{a.label} has no energy (not in the database, or not a finished SCF run)")
        for b in parts[i + 1:]:
            r, w = check(a, b)
            refuse += [x for x in r if x not in refuse]
            warn += [x for x in w if x not in warn]
    if refuse:
        raise Skip("not combined: " + "; ".join(dict.fromkeys(refuse)))
    return warn


# ------------------------------------------------------------------ the database's runs
def light_values(conn, calc_ids: Sequence[str], names: Sequence[str] = LIGHT) -> Dict[str, Dict[str, Any]]:
    """{calc_id: {name: value}} of the stored rows ``names`` (JSON decoded)."""
    from mace.properties.registry import decode
    out: Dict[str, Dict[str, Any]] = {c: {} for c in calc_ids}
    if not calc_ids:
        return out
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                         "AND name IN ('properties', 'property_blobs')")]
    ids = list(calc_ids)
    for t in tables:
        num = "property_value" if t == "properties" else "NULL"
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            q = (f"SELECT calc_id, property_name, {num}, property_value_text FROM {t} "
                 f"WHERE calc_id IN ({','.join('?' * len(chunk))}) AND property_name IN ({','.join('?' * len(names))})")
            for cid, name, v, text in conn.execute(q, (*chunk, *names)):
                out[cid][name] = decode(v, text, name)
    return out


def candidates(conn, *, calc_types: Sequence[str] = REF_TYPES, dims: Optional[Sequence[str]] = None,
               formula_ok: Optional[Callable[[Optional[Counter]], bool]] = None, exclude: Iterable[str] = (),
               role: str = "", limit: int = 2000, material_id: Optional[str] = None) -> List[Part]:
    """Finished runs of the database (``calc_types``), the newest first, as
    Parts with their stored rows (:data:`LIGHT`): of materials whose
    ``dimensionality`` is one of ``dims`` (CRYSTAL, SLAB, POLYMER, MOLECULE)
    and whose formula passes ``formula_ok`` (a material with no formula
    passes both), or of one material. ``exclude``: calc_ids left out."""
    if conn is None:
        return []
    skip = set(exclude)
    mat = " AND c.material_id = ?" if material_id else ""
    rows = conn.execute(
        "SELECT c.calc_id, c.material_id, c.calc_type, c.output_file, m.formula, m.dimensionality "
        "FROM calculations c LEFT JOIN materials m ON m.material_id = c.material_id "
        f"WHERE c.status = 'completed' AND c.calc_type IN ({','.join('?' * len(calc_types))}){mat} "
        "ORDER BY COALESCE(c.completed_at, c.created_at) DESC LIMIT ?",
        (*calc_types, *([material_id] if material_id else []), int(limit))).fetchall()
    keep = []
    for cid, mid, ct, out, f, dim in rows:
        if cid in skip:
            continue
        if dims and dim and dim not in dims:
            continue
        comp = parse_formula(f)
        if formula_ok and comp is not None and not formula_ok(comp):
            continue
        keep.append((cid, mid, ct, out))
    # the Mulliken table (a large row) only for runs whose positions do not give the composition
    vals = light_values(conn, [k[0] for k in keep], [n for n in LIGHT if n not in _FALLBACK])
    again = [c for c, v in vals.items() if composition(v) is None]
    for c, v in light_values(conn, again, _FALLBACK).items():
        vals[c].update(v)
    parts = []
    for cid, mid, ct, out in keep:
        v = vals.get(cid) or {}
        if not any(v.get(k) is not None for k in ENERGY_KEYS):
            continue
        p = Path(out) if out else None
        parts.append(Part(p.name if p else cid, v, role=role, calc_id=cid, material_id=mid, output_file=p,
                          via="database", calc_type=ct))
    return parts


def consistent(run: Part, parts: Iterable[Part]) -> List[Part]:
    """The parts that may be combined with ``run`` (no refusal)."""
    return [p for p in parts if not check(run, p)[0]]


#: the largest relative difference of two input cells that are one structure's
SAME_CELL_TOL = 1e-4


def same_structure(a: Part, b: Part) -> bool:
    """Whether two periodic runs are one structure: per input cell, the same
    composition and cell (to SAME_CELL_TOL), as a phonon supercell
    (SCELPHONO) and the cell it expands are, or two runs of one cell under
    different names (its SP, CPHF and FREQ). Their energies differ only by
    the k-point sampling, not by phase."""
    if a.dim != b.dim or not a.dim or not (a.comp and b.comp):
        return False
    ma, mb = a.multiple, b.multiple
    n = ratio(a.comp, b.comp)
    ca, cb = a.cell, b.cell
    if n is None or abs(n - ma / mb) > 1e-9 * n or not (ca and cb):
        return False
    return abs(ca / ma - cb / mb) <= SAME_CELL_TOL * (ca / ma)


def describe_settings(run: Part) -> str:
    s = run.settings
    return f"{s['functional'] or 'functional unknown'}, {s['basis_set'] or 'basis set unknown'}"


# ------------------------------------------------------------------ thermodynamics
def _thermo_of(props: Dict[str, Any], text: str = "") -> Optional[Dict[str, Any]]:
    g, h = props.get("gibbs_free_energy_au"), props.get("enthalpy_au")
    if g is None or h is None:
        return None
    t, p = props.get("thermodynamic_temperature_k"), props.get("thermodynamic_pressure_mpa")
    pts = None
    tv = props.get("thermodynamics_vs_temperature")
    if isinstance(tv, dict) and isinstance(tv.get("points"), list):
        pts = tv["points"]
    elif text:
        got = _points(text)
        pts = got if len(got) >= 2 else None
    pts = [{"T": float(x["temperature_k"]), "P": float(x.get("pressure_mpa") or 0.0),
            "G": float(x["gibbs_free_energy_au"]), "H": float(x["enthalpy_au"])}
           for x in (pts or []) if x.get("gibbs_free_energy_au") is not None and x.get("enthalpy_au") is not None]
    return {"G": float(g), "H": float(h), "T": float(t) if t is not None else None,
            "P": float(p) if p is not None else 0.0, "points": pts or None}


_LAST_POINTS: List[Any] = [None, None]


def _points(text: str) -> List[Dict[str, Any]]:
    """thermodynamics.points of an output, the last one remembered (each
    reference energy of a FREQ run reads the same text)."""
    if _LAST_POINTS[0] is not text:
        from mace.properties.defs import thermodynamics
        _LAST_POINTS[:] = [text, thermodynamics.points(text)]
    return _LAST_POINTS[1]


def _borrowed(th: Dict[str, Any], energy: float, el: float, scale: float) -> Dict[str, Any]:
    """Another run's thermodynamics as this run's: its thermal correction
    (G - EL, H - EL at each temperature) times ``scale`` (this run's atoms
    over its atoms), added to this run's own energy."""
    def one(g, h):
        return energy + (g - el) * scale, energy + (h - el) * scale
    out = dict(th)
    out["G"], out["H"] = one(th["G"], th["H"])
    if th.get("points"):
        pts = []
        for x in th["points"]:
            g, h = one(x["G"], x["H"])
            pts.append({**x, "G": g, "H": h})
        out["points"] = pts
    return out


def thermo_cells(part: Part) -> int:
    """How many cells of its FREQ thermodynamics a run's energy is of: a
    phonon dispersion or DOS run on a supercell (SCELPHONO) prints them per
    input cell, its EL the supercell's energy over its input cells; a FREQ
    run on a SUPERCELL geometry (Γ only) prints them for the whole cell.
    The supercell's input cells (Part.multiple) when EL says per input
    cell, else 1."""
    el, e, m = part.props.get("electronic_energy_au"), part.energy, part.multiple
    if m <= 1 or el is None or e is None or not float(el):
        return 1
    r = e / float(el)
    return m if abs(r - m) < 1e-3 * m else 1


def thermo(part: Part, conn) -> Tuple[Optional[Dict[str, Any]], str]:
    """The run's FREQ thermodynamics: its own (a FREQ run), else the newest
    FREQ run of its material in the database that may be combined with it
    and whose composition is a multiple of the run's. A borrowed FREQ run
    lends only its thermal correction (G - EL, H - EL, EL its electronic
    energy), scaled by atoms(run)/atoms(FREQ run) and added to the run's
    own energy: a supercell's FREQ run serves its primitive cell. CRYSTAL
    prints a phonon supercell's (SCELPHONO) thermodynamics per input cell:
    they are scaled to the supercell its energy is of (thermo_cells).
    (thermo, "") or (None, why)."""
    # the output's text (every temperature it printed) for the run itself and a file; a stored run's curve is a row
    use_text = part.role == "run" or (not part.calc_id and part.output_file is not None)
    own = _thermo_of(part.props, part.text() if use_text else "")
    if own:
        n = thermo_cells(part)
        if n > 1:
            own = _borrowed(own, part.energy, float(part.props["electronic_energy_au"]), n)
            own["source"] = f"{part.label} (×{n} input cells)"
        else:
            own["source"] = part.label
        return own, ""
    if conn is not None and part.material_id:
        why = f"{part.label} has no FREQ thermodynamics (no FREQ run of {part.material_id} in the database)"
        if part.energy is None:
            return None, why
        for f in candidates(conn, calc_types=("FREQ",), exclude=[part.calc_id] if part.calc_id else (),
                            material_id=part.material_id):
            th = _thermo_of(f.props)
            if not th or check(part, f)[0]:
                continue
            scale = ratio(part.comp, f.comp) if part.comp and f.comp else None
            if scale is None:
                why = (f"{part.label} has no FREQ thermodynamics: the FREQ run {f.label} "
                       f"({formula(f.comp)}) is not a multiple of {formula(part.comp)}")
                continue
            # thermodynamics per input cell (a phonon supercell's) serve that many of its atoms
            cells = thermo_cells(f)
            scale *= cells
            el = f.props.get("electronic_energy_au")
            el = float(el) if el is not None else f.energy / cells
            th = _borrowed(th, part.energy, el, scale)
            th["source"] = f.label if abs(scale - 1) < 1e-12 else f"{f.label} (×{fmt_coef(scale)})"
            th["borrowed"] = {"from": f.label, "scale": scale, "electronic_energy_au": el}
            return th, ""
        return None, why
    return None, f"{part.label} has no FREQ thermodynamics"


def forms(parts: Sequence[Part], fn: Callable[[List[float]], Dict[str, float]], conn) \
        -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """{"energy": fn(E), "enthalpy": fn(H), "gibbs_energy": fn(G)} and what
    the property's JSON says about them. ``fn`` takes one value per part (in
    Hartree, the parts' order) and returns {unit suffix: value}. ΔH and ΔG
    only when every part has FREQ thermodynamics at one temperature and
    pressure; else "thermo_note" says why."""
    out = {"energy": fn([p.energy for p in parts])}
    detail: Dict[str, Any] = {}
    ths, why = [], []
    for p in parts:
        th, reason = thermo(p, conn)
        ths.append(th)
        if th is None:
            why.append(reason)
    if not why:
        temps = {round(t["T"], 2) if t["T"] is not None else None for t in ths}
        press = {round(t["P"], 6) for t in ths}
        if len(temps) == 1 and None not in temps and len(press) == 1:
            out["enthalpy"] = fn([t["H"] for t in ths])
            out["gibbs_energy"] = fn([t["G"] for t in ths])
            detail["thermo"] = {"temperature_k": ths[0]["T"], "pressure_mpa": ths[0]["P"],
                                "from": [t["source"] for t in ths],
                                "note": "H = G + TS and G = EL + E0 + ET + PV - TS of each FREQ run (CRYSTAL's "
                                        "harmonic thermodynamics); a run without its own takes the thermal "
                                        "correction (G - EL, H - EL) of a FREQ run of its material, scaled by "
                                        "their atoms; ΔE is from the energies above"}
        else:
            why.append("the FREQ runs are at different temperatures or pressures: " + ", ".join(
                f"{p.label} {t['T']} K, {t['P']} MPa" for p, t in zip(parts, ths)))
        sets = []
        for t in ths:
            pts = t["points"] or [{"T": t["T"], "P": t["P"], "G": t["G"], "H": t["H"]}]
            sets.append({(round(x["T"], 2), round(x["P"], 6)): x for x in pts if x["T"] is not None})
        common = sorted(set.intersection(*(set(s) for s in sets))) if sets else []
        if len(common) >= 2:
            detail["vs_temperature"] = [
                {"temperature_k": k[0], "pressure_mpa": k[1],
                 "enthalpy": fn([s[k]["H"] for s in sets]), "gibbs_energy": fn([s[k]["G"] for s in sets])}
                for k in common]
    if why:
        detail["thermo_note"] = "ΔE only: " + "; ".join(why)
    return out, detail


# ------------------------------------------------------------------ reactions
def balance(signed: Sequence[Tuple[int, Counter]], fixed: int = 0) -> Optional[List[Fraction]]:
    """Coefficients c (c[fixed] = 1, all > 0) with Σ sign·c·composition = 0,
    or None (not balanced in exactly one way). ``signed``: (+1 product,
    -1 reactant, composition) per participant."""
    els = sorted({e for _, c in signed for e in c})
    free = [j for j in range(len(signed)) if j != fixed]
    # A x = b over the free coefficients
    a = [[Fraction(signed[j][0] * signed[j][1].get(e, 0)) for j in free] for e in els]
    b = [Fraction(-signed[fixed][0] * signed[fixed][1].get(e, 0)) for e in els]
    rows, cols = len(a), len(free)
    m = [a[i] + [b[i]] for i in range(rows)]
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        m[r] = [x / m[r][c] for x in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        piv.append(c)
        r += 1
    if any(all(x == 0 for x in m[i][:cols]) and m[i][cols] != 0 for i in range(rows)):
        return None                     # inconsistent
    if len(piv) < cols:
        return None                     # more than one way
    x = [Fraction(0)] * cols
    for i, c in enumerate(piv):
        x[c] = m[i][cols]
    out = [Fraction(1)] * len(signed)
    for j, v in zip(free, x):
        out[j] = v
    return out if all(v > 0 for v in out) else None


def imbalance(signed: Sequence[Tuple[int, Counter]], coef: Sequence[float]) -> Dict[str, float]:
    """{element: atoms left over} (products minus reactants); {} when balanced."""
    tot: Counter = Counter()
    for (s, c), k in zip(signed, coef):
        for e, n in c.items():
            tot[e] += s * k * n
    return {e: v for e, v in tot.items() if abs(v) > 1e-6}


def fmt_coef(c: float) -> str:
    f = Fraction(c).limit_denominator(1000)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def json_safe(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))
