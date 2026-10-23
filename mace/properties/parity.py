"""GUI/database property parity: what the GUI shows for a run is in the
database after the run is extracted, and the other way around.

Three checks per run (tests/test_property_parity.py runs them on every
output of the example suite):

* the Properties tab of a run in the database lists exactly the run's rows
  (``tab_names`` against ``db_names``);
* the values the tab computes for a run not in a database (the extractor and
  the derived-property registry, nothing written) are the keys and values
  the database holds after extraction (``compare``);
* every **producer** (a value the GUI computes elsewhere: the run card,
  the Relaxation Time dialog's automatic masses, the transport plot's ZT)
  maps to a database key with the same value.

A property the GUI starts to compute is registered as a derived property
(mace/properties/defs) and, when the GUI shows it outside the Properties
tab, gets a producer here (:func:`producer`), so the check covers it.
"""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Producer:
    """A GUI computation outside the Properties tab: ``values(output_file,
    calc_type) -> {database key: value}`` (empty when it does not apply)."""
    name: str
    calc_types: tuple
    values: Callable[[Path, Optional[str]], Dict[str, Any]]
    where: str = ""


PRODUCERS: List[Producer] = []


def producer(name: str, calc_types: Sequence[str], where: str = ""):
    """Decorator: register a GUI computation and the database keys it shows."""
    def wrap(fn):
        PRODUCERS.append(Producer(name, tuple(calc_types), fn, where))
        return fn
    return wrap


# ------------------------------------------------------------------ the GUI's own computations
@producer("run card", ("*",), "Run Analysis › card (mace/gui/runinfo.meta_job)")
def _run_card(out: Path, calc_type):
    from mace.gui import runinfo
    m = runinfo.meta_job(path=str(out))
    pairs = {"energy_au": "total_energy_au", "energy_ev_per_atom": "energy_per_atom_ev", "functional": "dft_functional",
             "basis": "basis_set", "crystal_version": "crystal_version", "mpi_processes": "mpi_processes",
             "atoms": "atoms_in_unit_cell", "cpu_s": "cpu_time_total_s"}
    return {db: m.get(card) for card, db in pairs.items()}


@producer("automatic masses", ("BAND",), "Relaxation Time dialog (m* auto) and mace tau --m-star auto")
def _auto_masses(out: Path, calc_type):
    from mace.properties.defs.effective_masses import _band_file
    from mace.transport import mass
    from mace.utils.property_extractor import CrystalPropertyExtractor
    band = _band_file(out)
    if band is None:
        return {}
    dos, _ = CrystalPropertyExtractor._dat_file_for(out, "DOSS")
    dim = CrystalPropertyExtractor._dimensionality(out.read_text(errors="replace")) or 3
    vals: Dict[str, Any] = {}
    for carrier in ("electron", "hole"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                am = mass.auto_masses(carrier, None, dim, band=band, dos=dos)
        except (mass.MassError, OSError, ValueError):
            vals[f"effective_mass_{carrier}_mc"] = None
            continue
        vals[f"effective_mass_{carrier}_mc"] = am["m_star"]
        vals[f"effective_mass_{carrier}_md"] = am["m_dos"]
    return vals


@producer("transport ZT plot", ("TRANSPORT",), "Transport plot › ZT (mace/gui/results.transport_zt)")
def _transport_zt(out: Path, calc_type):
    from mace.gui import results
    seebeck = [p for p in out.parent.iterdir() if p.name.upper().endswith("SEEBECK.DAT")
               and p.name.upper().startswith(out.stem.upper())]
    if not seebeck:
        return {}
    try:
        r = results.transport_zt(seebeck[0])
    except ValueError:
        return {}
    peak = r.get("zt_e_max")
    return {"transport_zt_electronic_max_avg": peak["value"] if peak else None}


@producer("mechanics results", ("ELASTIC", "EOS", "QHA", "CPHF", "FREQ"),
          "Mechanical Properties Results (mace/elastic/views.document)")
def _mechanics_results(out: Path, calc_type):
    from mace.elastic import properties as P
    from mace.elastic import views
    from mace.properties.defs import mechanics as M
    text = out.read_text(errors="replace")
    if calc_type == "FREQ" and "ATOMIC BORN CHARGE TENSOR" not in text:
        return {}
    doc = views.document(text, out)
    return M.keys_of(P.all_properties(doc["data"], text)) if doc["data"] else {}


# ------------------------------------------------------------------ comparisons
def same(a: Any, b: Any, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    """Equal values: numbers to ``rel``, JSON structures element by element."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) if isinstance(a, (bool, int, float)) and isinstance(b, (bool, int, float)) else a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rel, abs_) for x, y in zip(a, b))
    return a == b


# keys whose value records the extraction itself, not the run (compared by name only)
VOLATILE = {"extraction_provenance"}
# keys that hold a file path as the extraction spelled it (a scan's relative path, the GUI's
# absolute one): compared by the file's name
PATH_KEYS = {"band_file_analyzed", "doss_file_analyzed"}


def compare(gui: Dict[str, Any], db: Dict[str, Any], ignore: Iterable[str] = ()) -> List[str]:
    """What differs between the GUI's values and the database's for one run:
    keys only one side has, and values that differ."""
    skip = set(ignore)
    out = []
    for k in sorted(set(gui) - set(db) - skip):
        out.append(f"the GUI shows {k}, the database does not have it")
    for k in sorted(set(db) - set(gui) - skip):
        out.append(f"the database has {k}, the GUI does not show it")
    for k in sorted((set(gui) & set(db)) - skip - VOLATILE):
        if k in PATH_KEYS and isinstance(gui[k], str) and isinstance(db[k], str):
            if Path(gui[k]).name != Path(db[k]).name:
                out.append(f"{k}: GUI {gui[k]!r} != database {db[k]!r}")
            continue
        if not same(gui[k], db[k]):
            out.append(f"{k}: GUI {gui[k]!r} != database {db[k]!r}")
    return out


def producer_problems(out: Path, calc_type: Optional[str], db: Dict[str, Any]) -> List[str]:
    """Every producer that applies to the run: its values against the database's."""
    problems = []
    for p in PRODUCERS:
        if "*" not in p.calc_types and calc_type not in p.calc_types:
            continue
        for k, v in (p.values(Path(out), calc_type) or {}).items():
            if v is None and k not in db:
                continue
            if k not in db:
                problems.append(f"{p.name} shows {k} = {v!r}; the database does not have it")
            elif not same(v, db[k], rel=1e-9):
                problems.append(f"{p.name} shows {k} = {v!r}; the database has {db[k]!r}")
    return problems
