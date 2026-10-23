"""The derived-property registry: one place that says how a property MACE
computes from a run (rather than reads off its output) is computed, what it
needs, how it is shown and where it is stored.

A derived property is registered once, with :func:`derived` (a decorator on
its compute function) or :func:`register`:

* its **outputs**: the database keys it writes (``Output``: name, unit, value
  type, a label with plotly markup, ``<sub>``/``<sup>``, a display group and
  a precision); a curve or an array is a ``json`` output, and ``blob=True``
  keeps it in the ``property_blobs`` table;
* the **calculation types** it applies to (``"*"``: every run);
* what it **needs**: keys the extractor must have written (``requires``),
  files beside the run (``Sibling``: a BAND.DAT, a DOSS.DAT, the deck) and
  other runs (``Reference``: a bulk, elemental or clean-slab run the user
  names or the database finds);
* its **compute** function: ``compute(ctx) -> {output name: value}`` (an
  empty dict, None or :class:`Skip` stores nothing and says why).

``mace analyze --extract-properties`` runs every registered property that
applies after its own extraction (:func:`run_derived`) and stores the values
with the rest of the run's rows (``crystal_property_extractor.py`` rows:
re-extraction replaces them); which ones ran, from which files and at which
version is the run's ``extraction_provenance`` row. A property with
``interactive=True`` (a fit the user steers in the GUI) is not run on
extraction: the GUI computes it on request and its **Save to Database**
stores it (:func:`store_interactive`, rows by ``mace-gui``, which a
re-extraction keeps). The GUI's Properties tab shows the stored rows of a run,
or computes them with the same functions when the run is not in a database.

Adding a property is one module in ``mace/properties/defs/`` (every module
there is imported by :func:`load`); nothing else needs editing. See
docs/properties.md.

Standard library only at import: compute functions import what they use.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import pkgutil
import re
import sqlite3
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from mace.utils import property_keys

#: bumped when the provenance row's layout or the registry's semantics change
REGISTRY_VERSION = 1
#: extractor_script of the rows the extractor writes (derived ones included)
EXTRACTOR_SCRIPT = "crystal_property_extractor.py"
#: extractor_script of the rows an interactive Save to Database writes
GUI_SCRIPT = "mace-gui"
#: the run's provenance row (which derived properties ran, from which files)
PROVENANCE_KEY = "extraction_provenance"
#: the run's saved reference choices ({property: {reference: [{calc_id|file, weight}]}}):
#: what ``--reference`` or the GUI's Add Reference… chose, used by every later extraction
CHOICES_KEY = "reference_runs"
#: extractor_script of that row (a re-extraction keeps it, as the GUI's rows)
REFERENCE_SCRIPT = "mace-reference"
#: ``--reference NAME=auto``: forget the saved choice, look the run up in the database again
AUTO = "auto"

#: properties too slow for a job's completion callback or an MCP call (a Bader
#: partition of a dense cube, a packing-fraction grid): the queue and the MCP
#: results tool skip them unless MACE_DERIVED_EXPENSIVE=1
EXPENSIVE = ("bader_charges", "packing_fraction")
#: BLAS/OpenMP threads a derived property may use (MACE_DERIVED_THREADS overrides)
_THREADS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

ALL = "*"
_TYPES = ("float", "int", "bool", "text", "json")


class Skip(Exception):
    """Raised by a compute function: the property does not apply to this run,
    or an input is missing. Nothing is stored; the reason is recorded."""


class RegistryError(ValueError):
    """A registration that cannot work (a duplicate key, a bad unit, ...)."""


# ------------------------------------------------------------------ declarations
@dataclass(frozen=True)
class Output:
    """One database key a derived property writes.

    ``label`` may carry plotly markup (``m<sub>c</sub>``, ``Å<sup>3</sup>``);
    :func:`mace.properties.display.plain` gives the CSV form (``m_c``).
    ``unit`` and ``type`` follow mace/utils/property_keys.py (a ``bool`` is
    unit ``boolean``, a ``text`` unit ``category``, a curve ``json``).
    ``precision``: significant digits the GUI shows. ``blob``: a large JSON
    value kept in the ``property_blobs`` table. ``extensive``: the value is
    per cell (a total energy, a total moment), so the Properties tab offers
    it per atom and per formula unit too (mace/properties/units.py); an
    intensive value (a gap, an energy per atom) leaves it False.
    ``same_as``: another output of the same property holding the same
    quantity in another unit (``adsorption_energy_kj_mol`` is
    ``adsorption_energy_ev``): the Properties tab shows the two as one row
    with a unit picker; both stay in the database."""
    name: str
    unit: str
    type: str = "float"
    label: str = ""
    description: str = ""
    precision: int = 5
    group: str = ""
    category: str = ""
    blob: bool = False
    extensive: bool = False
    same_as: str = ""


@dataclass(frozen=True)
class Sibling:
    """A file beside the run the property reads.

    ``kind``: ``BAND`` or ``DOSS`` (the run's ``<stem>.<KIND>.DAT``, or the
    BAND/DOSS run of the same parent: the extractor's own rule,
    ``CrystalPropertyExtractor._dat_file_for``), ``d12`` or ``d3`` (the deck
    of the same stem), or any name with a ``find(output_file) -> Path|None``.
    A required sibling that is missing skips the property."""
    kind: str
    required: bool = True
    find: Optional[Callable[[Path], Optional[Path]]] = None
    description: str = ""


@dataclass(frozen=True)
class Reference:
    """Another run the property needs: a bulk for a surface energy, the
    elements for a formation energy, a clean slab and a molecule for an
    adsorption energy.

    Resolved in this order: the run(s) the user gives
    (``--reference NAME=FILE|CALC_ID`` on the command line, the GUI's picker:
    ``run_derived(references={NAME: ...})``), then the run's saved choice
    (:data:`CHOICES_KEY`: what an earlier ``--reference`` or the picker
    chose for a property that was stored), then ``find(ctx, conn)`` in the
    database (``conn`` a read-only sqlite3 connection, or None without a
    database; return a list of :class:`RunRef`, or raise :class:`Skip` to
    say why none fits). ``many``: several runs (the elements); else the
    first one is used. A required reference that is not found skips the
    property ("needs the reference run NAME")."""
    name: str
    calc_types: Tuple[str, ...] = ()
    required: bool = True
    many: bool = False
    find: Optional[Callable[["RunContext", Any], List["RunRef"]]] = None
    description: str = ""


@dataclass(frozen=True)
class DerivedProperty:
    name: str
    calc_types: Tuple[str, ...]
    outputs: Tuple[Output, ...]
    compute: Callable[["RunContext"], Optional[Dict[str, Any]]]
    category: str
    group: str = ""
    requires: Tuple[str, ...] = ()
    siblings: Tuple[Sibling, ...] = ()
    references: Tuple[Reference, ...] = ()
    version: str = "1"
    description: str = ""
    source: str = ""
    interactive: bool = False
    module: str = ""

    def applies_to(self, calc_type: Optional[str]) -> bool:
        return ALL in self.calc_types or (calc_type or "") in self.calc_types

    def output(self, name: str) -> Optional[Output]:
        return next((o for o in self.outputs if o.name == name), None)


@dataclass
class RunRef:
    """A run another property refers to: from the database (material_id,
    calc_id) or a file. ``properties`` are its stored rows (loaded when the
    reference is resolved), or the extractor's values of the file."""
    output_file: Optional[Path] = None
    material_id: Optional[str] = None
    calc_id: Optional[str] = None
    calc_type: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    via: str = ""
    #: a stoichiometric coefficient the user gave (``--reference reactants=2*FILE``), else None
    weight: Optional[float] = None

    def value(self, name: str, default: Any = None) -> Any:
        return self.properties.get(name, default)

    def describe(self) -> str:
        if self.output_file:
            return Path(self.output_file).name
        return self.calc_id or self.material_id or "?"

    def record(self) -> Dict[str, Any]:
        """How the reference is saved and reported: its calc_id or file, its
        coefficient, and where it came from (given, saved, database)."""
        out: Dict[str, Any] = {}
        if self.calc_id:
            out["calc_id"] = self.calc_id
        elif self.output_file:
            out["file"] = str(self.output_file)
        if self.weight is not None:
            out["weight"] = self.weight
        out["via"] = self.via
        return out


@dataclass
class RunContext:
    """What a compute function sees: the run's file and text, the values the
    extractor read from it, its siblings and references."""
    output_file: Path
    content: str
    properties: Dict[str, Any]
    calc_type: Optional[str]
    siblings: Dict[str, Path] = field(default_factory=dict)
    references: Dict[str, List[RunRef]] = field(default_factory=dict)
    dimensionality: Optional[int] = None
    #: an interactive property's inputs: what the user chose and the result
    #: the GUI computed (compute_interactive)
    inputs: Dict[str, Any] = field(default_factory=dict)
    #: the run's calculation and material in the database (None: untracked)
    calc_id: Optional[str] = None
    material_id: Optional[str] = None
    #: the read-only database connection references are looked up in (None without one)
    db: Any = None

    def value(self, name: str, default: Any = None) -> Any:
        return self.properties.get(name, default)

    def sibling(self, kind: str) -> Optional[Path]:
        return self.siblings.get(kind)

    def reference(self, name: str) -> Optional[RunRef]:
        refs = self.references.get(name) or []
        return refs[0] if refs else None


# ------------------------------------------------------------------ the registry
_REGISTRY: Dict[str, DerivedProperty] = {}
_OUTPUTS: Dict[str, str] = {}          # output key -> property name
_LOADED = False
_LOAD_LOCK = threading.RLock()


def _check(prop: DerivedProperty) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prop.name):
        raise RegistryError(f"{prop.name!r}: a property name is lower_snake_case")
    if prop.name in _REGISTRY and _REGISTRY[prop.name].module != prop.module:
        raise RegistryError(f"{prop.name!r} is registered twice ({_REGISTRY[prop.name].module}, {prop.module})")
    if not prop.outputs:
        raise RegistryError(f"{prop.name}: no outputs")
    if not prop.calc_types:
        raise RegistryError(f"{prop.name}: no calc_types (use '*' for every run)")
    if prop.category not in property_keys.CATEGORIES:
        raise RegistryError(f"{prop.name}: category {prop.category!r} is not one of property_keys.CATEGORIES")
    for o in prop.outputs:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", o.name):
            raise RegistryError(f"{prop.name}: output {o.name!r} is not lower_snake_case")
        if o.type not in _TYPES:
            raise RegistryError(f"{prop.name}.{o.name}: type {o.type!r} is not one of {_TYPES}")
        if o.type == "bool" and o.unit != "boolean":
            raise RegistryError(f"{prop.name}.{o.name}: a bool's unit is 'boolean'")
        if o.type == "text" and o.unit not in ("category", "code"):
            raise RegistryError(f"{prop.name}.{o.name}: a text value's unit is 'category' or 'code'")
        if o.blob and o.type != "json":
            raise RegistryError(f"{prop.name}.{o.name}: only a json output goes to property_blobs")
        if o.category and o.category not in property_keys.CATEGORIES:
            raise RegistryError(f"{prop.name}.{o.name}: category {o.category!r} is unknown")
        if o.name in property_keys.KEYS or property_keys.ALIASES.get(o.name) or o.name in property_keys.REMOVED:
            raise RegistryError(f"{prop.name}: {o.name!r} is a key the extractor itself writes (or wrote)")
        owner = _OUTPUTS.get(o.name)
        if owner and owner != prop.name:
            raise RegistryError(f"{prop.name}: output {o.name!r} is already written by {owner!r}")
        if o.name == PROVENANCE_KEY:
            raise RegistryError(f"{o.name!r} is the registry's own row")
        if o.same_as and (o.same_as == o.name or prop.output(o.same_as) is None):
            raise RegistryError(f"{prop.name}.{o.name}: same_as {o.same_as!r} is not another output of {prop.name}")
        if o.extensive and o.type != "float":
            raise RegistryError(f"{prop.name}.{o.name}: only a number is extensive")


def register(prop: DerivedProperty) -> DerivedProperty:
    """Register a derived property (its output keys become database keys:
    property_keys.key() knows their category, unit and type)."""
    _check(prop)
    _REGISTRY[prop.name] = prop
    for o in prop.outputs:
        _OUTPUTS[o.name] = prop.name
        property_keys.register_extra(
            o.name, property_keys.Key(o.category or prop.category, o.unit, o.type,
                                      o.description or prop.description or o.label or o.name),
            blob=o.blob)
    return prop


def unregister(name: str) -> None:
    """Remove a registered property and its keys (tests; a plugin that
    replaces one)."""
    prop = _REGISTRY.pop(name, None)
    if prop is None:
        return
    for o in prop.outputs:
        if _OUTPUTS.get(o.name) == name:
            del _OUTPUTS[o.name]
            property_keys.EXTRA.pop(o.name, None)
            property_keys.EXTRA_BLOBS.discard(o.name)


def derived(name: str, *, calc_types: Sequence[str], outputs: Sequence[Output], category: str,
            group: str = "", requires: Sequence[str] = (), siblings: Sequence[Sibling] = (),
            references: Sequence[Reference] = (), version: str = "1", description: str = "",
            source: str = "", interactive: bool = False):
    """Decorator: register ``fn(ctx) -> {output: value}`` as derived property ``name``.

    Example (mace/properties/defs/<area>.py)::

        @derived("energy_per_atom", calc_types=("SP", "OPT", "FREQ"), category="energy",
                 group="Energy", requires=("total_energy_au", "atoms_in_unit_cell"),
                 outputs=(Output("energy_per_atom_ev", "eV", label="E/atom"),),
                 source="total_energy_au and atoms_in_unit_cell of the .out")
        def energy_per_atom(ctx):
            return {"energy_per_atom_ev": ctx.value("total_energy_au") * HARTREE_TO_EV
                    / ctx.value("atoms_in_unit_cell")}
    """
    def wrap(fn):
        register(DerivedProperty(
            name=name, calc_types=tuple(calc_types), outputs=tuple(outputs), compute=fn, category=category,
            group=group, requires=tuple(requires), siblings=tuple(siblings), references=tuple(references),
            version=str(version), description=description or (fn.__doc__ or "").strip().split("\n")[0],
            source=source, interactive=interactive, module=getattr(fn, "__module__", "")))
        return fn
    return wrap


def load() -> Dict[str, DerivedProperty]:
    """Import every module of mace/properties/defs (each registers its
    properties) once; the registry by name. A module that fails to import
    is reported on stderr and skipped: one broken definition never stops
    an extraction. Thread-safe (the GUI's server threads): the registry is
    marked loaded only after every module is imported."""
    global _LOADED
    if _LOADED:
        return dict(_REGISTRY)
    with _LOAD_LOCK:
        if not _LOADED:
            from mace.properties import defs
            for m in sorted(pkgutil.iter_modules(defs.__path__), key=lambda m: m.name):
                if m.name.startswith("_"):
                    continue
                try:
                    importlib.import_module(f"mace.properties.defs.{m.name}")
                except Exception as e:                    # noqa: BLE001
                    print(f"warning: derived properties of mace/properties/defs/{m.name}.py not loaded: {e}",
                          file=sys.stderr)
            _LOADED = True
    return dict(_REGISTRY)


def properties(calc_type: Optional[str] = None, interactive: Optional[bool] = None) -> List[DerivedProperty]:
    """The registered properties (of a calculation type), by name."""
    reg = load()
    out = [p for p in reg.values()
           if (calc_type is None or p.applies_to(calc_type)) and (interactive is None or p.interactive == interactive)]
    return sorted(out, key=lambda p: p.name)


def get(name: str) -> Optional[DerivedProperty]:
    return load().get(name)


def owner_of(key: str) -> Optional[DerivedProperty]:
    """The derived property that writes database key ``key``, or None."""
    load()
    n = _OUTPUTS.get(key)
    return _REGISTRY.get(n) if n else None


def output_of(key: str) -> Optional[Output]:
    p = owner_of(key)
    return p.output(key) if p else None


# ------------------------------------------------------------------ siblings and references
def _deck(output_file: Path, suffix: str) -> Optional[Path]:
    p = Path(output_file).with_suffix(suffix)
    return p if p.is_file() else None


def find_sibling(output_file: Path, sib: Sibling) -> Optional[Path]:
    """The file ``sib`` names for this run, or None."""
    if sib.find is not None:
        return sib.find(Path(output_file))
    kind = sib.kind
    if kind in ("d12", "d3"):
        return _deck(output_file, "." + kind)
    from mace.utils.property_extractor import CrystalPropertyExtractor
    path, _ = CrystalPropertyExtractor._dat_file_for(Path(output_file), kind.upper())
    return path


def db_runs(conn, *, material_id: Optional[str] = None, calc_types: Sequence[str] = (),
            formula: Optional[str] = None, status: Optional[str] = "completed", limit: int = 50) -> List[RunRef]:
    """Runs of the database for a Reference's ``find``: by material, by
    calculation type and by the material's formula (the newest first), each a
    RunRef with its calc_id (its stored rows are loaded when the reference is
    resolved). ``conn`` None (no database): none."""
    if conn is None:
        return []
    where, args = [], []
    if material_id:
        where.append("c.material_id = ?")
        args.append(material_id)
    if calc_types:
        where.append(f"c.calc_type IN ({','.join('?' * len(calc_types))})")
        args += list(calc_types)
    if formula:
        where.append("m.formula = ?")
        args.append(formula)
    if status:
        where.append("c.status = ?")
        args.append(status)
    sql = ("SELECT c.calc_id, c.material_id, c.calc_type, c.output_file FROM calculations c "
           "LEFT JOIN materials m ON m.material_id = c.material_id"
           + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY COALESCE(c.completed_at, c.created_at) DESC LIMIT ?")
    return [RunRef(output_file=Path(o) if o else None, material_id=m, calc_id=c, calc_type=t, via="database")
            for c, m, t, o in conn.execute(sql, (*args, int(limit)))]


def _ref_from_given(value: Any) -> RunRef:
    if isinstance(value, RunRef):
        return value
    return given_ref(value)


_WEIGHT = re.compile(r"^\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*\*\s*(.+)$")


def given_ref(value: Any, *, exists: Optional[Callable[[str], bool]] = None) -> Any:
    """A reference the user gave (``--reference NAME=VALUE``, the GUI's
    picker): ``FILE``, ``CALC_ID`` or ``N*FILE``/``N*CALC_ID`` (a coefficient
    for a reaction), as a RunRef with via "given"; ``auto`` is :data:`AUTO`
    (forget the saved choice). A string that is not an existing file is a
    calc_id. A RunRef is returned as it is."""
    if isinstance(value, RunRef):
        return value
    s = str(value).strip()
    if s.lower() == AUTO:
        return AUTO
    weight = None
    m = _WEIGHT.match(s)
    if m:
        weight, s = float(m.group(1)), m.group(2).strip()
    is_file = exists(s) if exists is not None else Path(s).exists()
    if is_file:
        return RunRef(output_file=Path(s), via="given", weight=weight)
    return RunRef(calc_id=s, via="given", weight=weight)


def saved_choices(conn, calc_id: Optional[str]) -> Dict[str, Any]:
    """The run's saved reference choices (:data:`CHOICES_KEY`), {} when none."""
    if conn is None or not calc_id:
        return {}
    try:
        r = conn.execute("SELECT property_value_text FROM properties WHERE calc_id = ? AND property_name = ?",
                         (calc_id, CHOICES_KEY)).fetchone()
    except sqlite3.Error:
        return {}
    if not r or not r[0]:
        return {}
    try:
        v = json.loads(r[0])
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _ref_from_saved(entry: Dict[str, Any]) -> Optional[RunRef]:
    if not isinstance(entry, dict):
        return None
    w = entry.get("weight")
    if entry.get("calc_id"):
        return RunRef(calc_id=str(entry["calc_id"]), via="saved", weight=w)
    if entry.get("file"):
        return RunRef(output_file=Path(entry["file"]), via="saved", weight=w)
    return None


def merge_choices(current: Dict[str, Any], used: Dict[str, Dict[str, List[Dict[str, Any]]]],
                  forget: Dict[str, List[str]]) -> Dict[str, Any]:
    """The saved choices after an extraction: ``used`` ({property: {reference:
    [record]}}, the given references of the properties that stored a value)
    replace those references; ``forget`` ({property: [reference]}, the ones
    given as ``auto``) removes them."""
    out = {p: dict(v) for p, v in (current or {}).items() if isinstance(v, dict)}
    for p, roles in (forget or {}).items():
        for r in roles:
            out.get(p, {}).pop(r, None)
    for p, roles in (used or {}).items():
        for r, recs in roles.items():
            out.setdefault(p, {})[r] = [{k: v for k, v in rec.items() if k != "via"} for rec in recs]
    return {p: v for p, v in out.items() if v}


def _load_ref(ref: RunRef, conn) -> RunRef:
    """Fill a reference's properties: its stored rows (a calc_id in the
    database) or the extractor's values of its file."""
    if ref.properties:
        return ref
    if ref.calc_id and conn is not None:
        ref.properties = stored_values(conn, ref.calc_id)
        if ref.properties:
            return ref
    if ref.output_file and Path(ref.output_file).is_file():
        import contextlib
        import io
        from mace.utils.property_extractor import CrystalPropertyExtractor
        x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
        x.run_derived = False
        with contextlib.redirect_stdout(io.StringIO()):
            ref.properties = {k: v for k, v in x.extract_all_properties(Path(ref.output_file)).items()
                              if not k.startswith("_")}
        ref.calc_type = ref.calc_type or ref.properties.get("calculation_type")
    return ref


def stored_values(conn, calc_id: str) -> Dict[str, Any]:
    """{name: value} of a calculation's stored rows (JSON decoded)."""
    out: Dict[str, Any] = {}
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                          "AND name IN ('properties', 'property_blobs')")]
    for t in tables:
        num = "property_value" if t == "properties" else "NULL"
        for name, v, text in conn.execute(
                f"SELECT property_name, {num}, property_value_text FROM {t} WHERE calc_id = ?", (calc_id,)):
            out[name] = decode(v, text, name)
    return out


def decode(value: Any, text: Optional[str], name: str = "") -> Any:
    """A stored row's value: the number, else the text (JSON decoded for a
    json key, a bool key as a bool)."""
    load()                                    # the derived keys' types
    k = property_keys.key(name)
    if k is not None and k.type == "json" and text:
        try:
            return json.loads(text)
        except ValueError:
            return text
    if k is not None and k.type == "bool":
        if value is not None:
            return bool(value)
        return text in ("True", "true", "1")
    if value is not None:
        return int(value) if k is not None and k.type == "int" and float(value).is_integer() else value
    return text


def resolve_references(prop: DerivedProperty, ctx: RunContext, given: Dict[str, Any], conn) -> Dict[str, List[RunRef]]:
    """Each reference of ``prop``: the runs the user gave, else the run's
    saved choice (``reference_runs``, an earlier ``--reference`` or the GUI's
    picker), else the property's lookup in the database. A reference given
    as ``auto`` skips the saved choice. The given and saved references are
    in ``ctx.references`` before any lookup runs, and each lookup's result
    after it, so a lookup sees the others (the adsorbate that fits a
    surface the user gave)."""
    out: Dict[str, List[RunRef]] = {}
    ctx.references = out
    saved = (saved_choices(conn, ctx.calc_id) if conn is not None else {}).get(prop.name) or {}
    lookup = []
    for ref in prop.references:
        found: List[RunRef] = []
        v = given.get(ref.name)
        vals = [_ref_from_given(x) for x in (v if isinstance(v, (list, tuple)) else [v])] if v is not None else []
        auto = AUTO in vals
        vals = [x for x in vals if x != AUTO]
        if vals:
            found = vals
        elif not auto and saved.get(ref.name):
            found = [r for r in (_ref_from_saved(e) for e in saved[ref.name]) if r is not None]
        elif ref.find is not None:
            lookup.append(ref)
        found = [_load_ref(r, conn) for r in found]
        out[ref.name] = found if ref.many else found[:1]
    for ref in lookup:
        try:
            found = list(ref.find(ctx, conn) or [])
        except Skip as e:
            # the lookup says why nothing fits (no elemental run for Cl, ...)
            found = []
            ctx.properties.setdefault("_reference_skips", {})[ref.name] = str(e)
        except Exception as e:                        # noqa: BLE001
            # a lookup that fails (a locked or damaged database, a bug) is an
            # error of the property, not a reference that is missing
            found = []
            ctx.properties.setdefault("_reference_errors", {})[ref.name] = f"{type(e).__name__}: {e}"
        found = [_load_ref(r, conn) for r in found]
        out[ref.name] = found if ref.many else found[:1]
    return {r.name: out.get(r.name, []) for r in prop.references}


# ------------------------------------------------------------------ running
def _clean(prop: DerivedProperty, values: Dict[str, Any]) -> Dict[str, Any]:
    """Only the declared outputs, None and NaN dropped, types as declared."""
    out = {}
    for o in prop.outputs:
        if o.name not in values:
            continue
        v = values[o.name]
        if v is None:
            continue
        if o.type == "float":
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                continue
        elif o.type == "int":
            v = int(v)
        elif o.type == "bool":
            v = bool(v)
        elif o.type == "text":
            v = str(v)
        out[o.name] = v
    unknown = set(values) - {o.name for o in prop.outputs}
    if unknown:
        raise RegistryError(f"{prop.name} returned undeclared outputs: {sorted(unknown)}")
    return out


def thread_limit():
    """A context that caps the BLAS/OpenMP threads of what runs inside it
    (numpy's linear algebra of a packing fraction, a Bader partition): an
    extraction never takes every core. MACE_DERIVED_THREADS sets the cap
    (default: half the cores, at most 4). Uses threadpoolctl when it is
    installed (numpy already imported); before numpy is imported, the
    OMP/OpenBLAS/MKL environment variables are set when they are unset."""
    try:
        n = int(os.environ.get("MACE_DERIVED_THREADS") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        n = max(1, min(4, (os.cpu_count() or 2) // 2))
    if "numpy" not in sys.modules:
        for var in _THREADS_ENV:
            os.environ.setdefault(var, str(n))
    return _capped(n)


@contextlib.contextmanager
def _capped(n: int):
    with contextlib.ExitStack() as stack:
        try:
            from threadpoolctl import threadpool_limits
            stack.enter_context(threadpool_limits(limits=n))
        except Exception:                             # noqa: BLE001 (not installed, or no pools)
            pass
        mkl = sys.modules.get("mkl")
        if mkl is not None and hasattr(mkl, "set_num_threads_local"):
            # mkl-service's thread-local setting overrides the global one threadpoolctl sets
            try:
                stack.callback(mkl.set_num_threads_local, mkl.set_num_threads_local(n))
            except Exception:                         # noqa: BLE001
                pass
        yield


def expensive() -> Tuple[str, ...]:
    """The properties a job's completion callback and the MCP results tool
    skip (:data:`EXPENSIVE`), none when MACE_DERIVED_EXPENSIVE=1. ``mace
    analyze --extract-properties`` computes them."""
    on = os.environ.get("MACE_DERIVED_EXPENSIVE", "").strip().lower() in ("1", "true", "yes", "on")
    return () if on else EXPENSIVE


def reference_properties(calc_type: Optional[str] = None) -> List[DerivedProperty]:
    """The extraction properties that read other runs (a Reference): their
    values depend on what the database holds, so a bulk build computes them
    after every run's own rows are stored."""
    return [p for p in properties(calc_type, interactive=False) if p.references]


def reference_roles() -> Dict[str, List[str]]:
    """{reference name: [the properties that take it]} (``--reference NAME=...``)."""
    out: Dict[str, List[str]] = {}
    for p in properties(None):
        for r in p.references:
            out.setdefault(r.name, []).append(p.name)
    return out


def run_derived(output_file: Path, content: str, properties: Dict[str, Any], *,
                calc_type: Optional[str] = None, references: Optional[Dict[str, Any]] = None,
                conn=None, only: Optional[Iterable[str]] = None, interactive: bool = False,
                dimensionality: Optional[int] = None, calc_id: Optional[str] = None,
                material_id: Optional[str] = None,
                skip: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run every registered property that applies to this run.

    ``properties``: what the extractor read (not changed). ``references``:
    {reference name: file | RunRef | [..]} the user gave. ``conn``: a
    read-only connection to the database for reference lookups (or a
    function that opens one), or None. ``calc_id``/``material_id``: the
    run's calculation in that database (its saved reference choices).
    ``skip``: {property: why} not computed this time (turned off, or left
    for later); their entries are ``skipped`` with ``kept`` True (a stored
    value of theirs stays).
    Returns (values, report): the values to store, and {property: {status,
    version, inputs, reason, references}} (status ``stored``, ``skipped`` or
    ``error``; ``references``: {name: [RunRef.record()]}, which run each
    reference was and where it came from). A property that raises, or whose
    reference runs cannot be read, is reported as an error, never
    propagated."""
    calc_type = calc_type or properties.get("calculation_type")
    wanted = set(only) if only is not None else None
    skip = skip or {}
    values: Dict[str, Any] = {}
    report: Dict[str, Any] = {}
    opened: List[Any] = []

    def db():
        # ``conn`` may be a zero-argument callable: the database is opened
        # only when a property that applies has references
        if not opened:
            opened.append(conn() if callable(conn) else conn)
        return opened[0]
    todo = [p for p in properties_for(calc_type, interactive) if wanted is None or p.name in wanted]
    with thread_limit() if todo else contextlib.nullcontext():
        for prop in todo:
            entry: Dict[str, Any] = {"version": prop.version}
            report[prop.name] = entry
            if prop.name in skip:
                entry.update(status="skipped", reason=skip[prop.name], kept=True)
                continue
            try:
                got = _run_one(prop, entry, output_file, content, properties, references, db,
                               dimensionality, calc_type, calc_id, material_id)
            except Exception as e:                    # noqa: BLE001
                entry.update(status="error", reason=f"{type(e).__name__}: {e}")
                entry["trace"] = traceback.format_exc(limit=3)[-600:]
                continue
            if got:
                values.update(got)
    return values, report


def _run_one(prop: DerivedProperty, entry: Dict[str, Any], output_file: Path, content: str,
             properties: Dict[str, Any], references, db, dimensionality, calc_type, calc_id,
             material_id) -> Optional[Dict[str, Any]]:
    """One property of run_derived: fills ``entry``, returns its values."""
    missing = [k for k in prop.requires if properties.get(k) is None]
    if missing:
        entry.update(status="skipped", reason="the run has no " + ", ".join(missing))
        return None
    sibs: Dict[str, Path] = {}
    lacking = []
    for s in prop.siblings:
        try:
            p = find_sibling(output_file, s)
        except Exception:                             # noqa: BLE001
            p = None
        if p is not None:
            sibs[s.kind] = p
        elif s.required:
            lacking.append(s.kind)
    if lacking:
        entry.update(status="skipped", reason="no " + " or ".join(f"{k} file" for k in lacking) + " beside the run")
        return None
    ctx = RunContext(Path(output_file), content, dict(properties), calc_type, sibs,
                     dimensionality=dimensionality, calc_id=calc_id, material_id=material_id)
    ctx.db = db() if prop.references else None
    ctx.references = resolve_references(prop, ctx, references or {}, ctx.db)
    if prop.references:
        entry["references"] = {k: [r.record() for r in v] for k, v in ctx.references.items() if v}
    need = [r.name for r in prop.references if r.required and not ctx.references.get(r.name)]
    failed = ctx.properties.get("_reference_errors") or {}
    if any(n in failed for n in need):
        entry.update(status="error", reason="the lookup of the reference run failed: "
                     + "; ".join(f"{n}: {failed[n]}" for n in need if n in failed))
        return None
    if need:
        why = ctx.properties.get("_reference_skips") or {}
        told = list(dict.fromkeys(why[n] for n in need if n in why))
        entry.update(status="skipped", reason="; ".join(told) if told else
                     "needs the reference run " + ", ".join(need)
                     + " (--reference NAME=FILE|CALC_ID, or a run in the database)")
        return None
    inputs = [Path(output_file).name] + [p.name for p in sibs.values()]
    inputs += [r.describe() for refs in ctx.references.values() for r in refs]
    entry["inputs"] = inputs
    try:
        got = prop.compute(ctx)
        got = _clean(prop, got or {})
    except Skip as e:
        entry.update(status="skipped", reason=str(e))
        return None
    if not got:
        entry.update(status="skipped", reason="nothing to compute for this run")
        return None
    entry.update(status="stored", outputs=sorted(got))
    return got


def properties_for(calc_type: Optional[str], interactive: bool = False) -> List[DerivedProperty]:
    return [p for p in properties(None) if p.applies_to(calc_type) and p.interactive == interactive]


def provenance(report: Dict[str, Any], output_file: Path) -> Dict[str, Any]:
    """The run's ``extraction_provenance`` value."""
    try:
        from mace import __version__ as mace_version
    except Exception:                                 # noqa: BLE001
        mace_version = None
    return {"mace_version": mace_version, "registry_version": REGISTRY_VERSION,
            "extractor": EXTRACTOR_SCRIPT, "output_file": Path(output_file).name,
            "derived": {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in report.items()}}


# ------------------------------------------------------------------ interactive results
def compute_interactive(prop_name: str, output_file: Path, inputs: Dict[str, Any],
                        references: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The values of an interactive property for the run ``output_file``:
    its compute function on ``inputs`` (what the user chose and the result
    the GUI computed). Raises Skip or RegistryError."""
    prop = get(prop_name)
    if prop is None:
        raise RegistryError(f"no derived property {prop_name!r}")
    if not prop.interactive:
        raise RegistryError(f"{prop_name} is computed on extraction, not saved from the GUI")
    ctx = RunContext(Path(output_file), "", {}, None, inputs=dict(inputs))
    ctx.references = {k: [_ref_from_given(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
                      for k, v in (references or {}).items()}
    return _clean(prop, prop.compute(ctx) or {})


def store_interactive(db_path, prop_name: str, values: Dict[str, Any], *, material_id: str,
                      calc_id: Optional[str], script: str = GUI_SCRIPT) -> List[str]:
    """Store an interactive property's values (the GUI's Save to Database)
    with MaterialDatabase.store_material_property: one row per output (a blob
    output in property_blobs). The rows' extractor_script (``mace-gui``)
    and extracted_at say where they came from, and the property's JSON
    output carries its inputs. A re-extraction of the run keeps these rows
    (it replaces only its own). Returns the names stored."""
    prop = get(prop_name)
    if prop is None:
        raise RegistryError(f"no derived property {prop_name!r}")
    if not prop.interactive:
        raise RegistryError(f"{prop_name} is computed on extraction, not saved from the GUI")
    vals = _clean(prop, values)
    if not vals:
        raise RegistryError(f"{prop_name}: nothing to store")
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(db_path))
    stored = []
    for name, v in vals.items():
        o = prop.output(name)
        db.store_material_property(material_id, name, v, o.category or prop.category, calc_id=calc_id,
                                   property_unit=o.unit, extractor_script=script, blob=o.blob)
        stored.append(name)
    return stored


def now() -> str:
    return datetime.now().isoformat()
