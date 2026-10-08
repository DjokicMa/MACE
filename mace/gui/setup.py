"""Calculation Setup: the CLI is the schema and the source of every default (plan B1-B3).

The GUI never writes calculation settings of its own. A deck is set up the
way a user sets it up at the prompt:

* ``mace convert`` (a CIF: cif2d12), ``mace opt2d12`` (a CRYSTAL .out, a
  follow-up SP/OPT/FREQ) or ``mace opt2d3`` (a .out, a properties run:
  BAND/DOSS/...) runs in a scratch folder with its "save the answers" switch;
* every field of the Setup form is one of the CLI's own questions (FIELDS):
  changing it re-runs the CLI with that question answered (answer_driver.py),
  so the settings the CLI works out from an answer (the grid, D3, the 3c
  basis, the FREQ block) are worked out by the CLI, never guessed here;
* the saved-answers JSON is the CLI's own file, kept as text on the server
  and handed to the browser verbatim (``json_text``): the browser never
  re-serializes it (0.0 stays 0.0, 1e-07 stays 1e-07);
* the deck preview runs the printed command itself (``--batch
  --options_file`` / ``--config-file --yes`` / ``--config-file``) on that
  file, in a mirror of the input's folder under the task's temp dir, so the
  preview is the deck the command writes (P == G).

A few settings no question sets (opt2d12's one-mesh-for-every-file key) are
written into the JSON by hand, as the CLI's documentation says: those are
``patch`` fields, applied in Python with the CLI's own JSON layout. The
cif2d12 k-point mesh is worked out from each cell by ``convert`` and is not
editable at all (a JSON ``k_points`` is ignored there).

Spike S6 (plan T0) classified each field by answering only its question on
the manual route: on cif2d12 and opt2d12 every question but the basis set
sits behind "use these default settings? / use the parent's settings?"
(answered No), and that route itself rewrites other keys (origin choice,
symmetry validation, the OPT block), so every asked field is derived by a
CLI run (``DERIVED_BY_CLI``); none is patched.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import clirun

REPO = clirun.REPO
import sys as _sys
for _d in ("Crystal_d12", "Crystal_d3"):
    if str(REPO / _d) not in _sys.path:
        _sys.path.insert(0, str(REPO / _d))
MODES = ("cif2d12", "opt2d12", "opt2d3")
CALC_TYPES = {"cif2d12": ("OPT", "SP", "FREQ"), "opt2d12": ("SP", "OPT", "FREQ"),
              "opt2d3": ("BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "CHARGE+POTENTIAL")}
DEFAULT_CALC = {"cif2d12": "OPT", "opt2d12": "SP", "opt2d3": "BAND"}
D12_GLOB_MAX = 20 * 1024 * 1024      # the .d12 files staged for opt2d3 (small text files)
OUT_SUFFIXES = (".out", ".log", ".o")


class SetupError(ValueError):
    """Setup can't make a deck from this (shown to the user, with the CLI's words)."""


# ---------------------------------------------------------------- modes and names
def mode_for(path: str, calc_type: Optional[str] = None) -> str:
    low = (path or "").lower()
    if low.endswith(".cif"):
        return "cif2d12"
    if low.endswith(OUT_SUFFIXES):
        return "opt2d3" if (calc_type or "").upper() in CALC_TYPES["opt2d3"] else "opt2d12"
    raise SetupError("A deck needs a CIF (a new calculation) or a CRYSTAL .out (a follow-up)")


def check_calc(mode: str, calc_type: Optional[str]) -> str:
    if mode not in MODES:
        raise SetupError(f"unknown mode {mode!r}")
    calc = (calc_type or DEFAULT_CALC[mode]).upper()
    if calc not in CALC_TYPES[mode]:
        raise SetupError(f"{mode} makes {', '.join(CALC_TYPES[mode])} decks, not {calc}")
    return calc


def d3_base(stem: str) -> str:
    """opt2d3's base name: the stem without one _opt/_sp/_OPT/_SP (CRYSTALOptToD3.D3Generator)."""
    for suffix in ("_opt", "_sp", "_OPT", "_SP", ".out", ".log"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def json_name(mode: str, calc: str, stem: str) -> str:
    if mode == "cif2d12":
        return f"{stem}_options.json"
    if mode == "opt2d12":
        return f"{stem}_{calc}_options.json"
    return f"{stem}_{calc.replace('+', '_')}_d3.json"


# ---------------------------------------------------------------- the wavefunction (opt2d3)
def wavefunction_candidates(out_name: str) -> List[str]:
    """The names opt2d3 looks for beside the .out, in its order
    (CRYSTALOptToD3.D3Generator._copy_wavefunction; mirrored here because the
    list is built inside that method, and a test checks it against the CLI)."""
    stem = Path(out_name).stem
    base = d3_base(stem)
    names = ["fort.9", "fort.98", f"{base}.f9", f"{base}.f98"]
    for suffix in ("sp", "opt", "freq", "soc"):
        names += [f"{base}_{suffix}.f9", f"{base}_{suffix}.f98"]
        for i in range(2, 100):
            names += [f"{base}_{suffix}{i}.f9", f"{base}_{suffix}{i}.f98"]
    if stem != base:
        names += [f"{stem}.f9", f"{stem}.f98"]
    return names


class StillRunning(Exception):
    """409: the run is still going; its wavefunction is copied when it ends."""


JOB_STATE_PROVIDER = None        # PLAN_submit's jobs.py: (out_path) -> "RUNNING" | "PENDING" | ... | None


def _f9_live(out_path: Path, f9: Path, st) -> bool:
    """Whether the run that writes ``f9`` is still going (PLAN_analysis C18.7),
    first rule that applies: the paired job is known queued or running; the
    paired .out has no end marker and is live; the job's scratch folder
    (``.<job>.scratch``) holds a fort.9 larger than the copy here (the
    job-end copy is still running); only when none of these is known, the
    .f9 changed in the last 10 s (30 s on a network file system)."""
    from mace.utils import livefile
    if JOB_STATE_PROVIDER is not None:
        try:
            js = JOB_STATE_PROVIDER(out_path)
            if js:
                return str(js).upper() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")
        except Exception:
            pass
    try:
        if out_path.is_file():
            snap = livefile.snapshot(out_path)
            tail = livefile.tail_text(out_path, snap)
            if not livefile.has_terminal_marker(tail):
                if livefile.is_live(out_path, snap, tail):
                    return True
    except OSError:
        pass
    ptr = out_path.parent / f".{out_path.stem}.scratch"
    try:
        scratch = Path(ptr.read_text().strip())
        sf9 = scratch / "fort.9"
        if sf9.is_file() and sf9.stat().st_size > st.st_size:
            return True
        if sf9.is_file():
            return False
    except (OSError, ValueError):
        pass
    young = 10.0
    try:
        from .dbsafety_loader import dbsafety
        if dbsafety().is_network(dbsafety().mount_of(f9)):
            young = 30.0
    except Exception:
        pass
    return time.time() - st.st_mtime < young


def find_wavefunction(out_path: Path) -> Optional[Dict[str, Any]]:
    """The wavefunction the CLI would copy, found by name and os.stat only
    (never opened): {name, bytes, live}, or None. ``live``: the run that
    writes it is still going (``_f9_live``)."""
    folder = out_path.parent
    for name in wavefunction_candidates(out_path.name):
        try:
            st = os.stat(folder / name)
        except OSError:
            continue
        return {"name": name, "bytes": int(st.st_size), "live": _f9_live(out_path, folder / name, st)}
    return None


def _anatase_siblings(out_path: Path) -> List[str]:
    """Sibling <base>_{opt,sp,OPT,SP}.{out,log} names: they change opt2d3's output names."""
    base = d3_base(out_path.stem)
    if base == out_path.stem:
        return []
    out = []
    for other in out_path.parent.glob(f"{base}_*"):
        if (other.suffix in (".out", ".log") and other.name != out_path.name
                and other.stem[len(base):] in ("_opt", "_sp", "_OPT", "_SP")):
            out.append(other.name)
    return sorted(out)


def _d12_globs(out_path: Path) -> List[Path]:
    """The .d12 files opt2d3 reads beside the .out: <base>*.d12 and <first stem token>*.d12."""
    base = d3_base(out_path.stem)
    first = out_path.stem.split("_")[0]
    seen, out = set(), []
    for pat in (f"{base}*.d12", f"{first}*.d12"):
        for p in sorted(out_path.parent.glob(pat)):
            if p.is_file() and p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


# ---------------------------------------------------------------- staging
class Staged:
    """A mirror, under ``tree``, of the files the printed command reads, at
    the same relative paths as in --dir; the command runs there unchanged."""

    def __init__(self, mode: str, calc: str, tree: Path, input_rel: str, json_rel: str, out_rel: str,
                 f9: Optional[Dict[str, Any]] = None, notes: Optional[List[str]] = None):
        self.mode, self.calc, self.tree = mode, calc, tree
        self.input_rel, self.json_rel, self.out_rel = input_rel, json_rel, out_rel
        self.f9 = f9
        self.notes = notes or []


def stage(mode: str, calc: str, work: Path, source_path: Optional[Path], rel: Optional[str],
          cif_bytes: Optional[bytes] = None, cif_name: Optional[str] = None) -> Staged:
    """Copy what the CLI reads into ``work/tree`` (plan B2):

    * cif2d12: the CIF alone, in its own folder setup/<stem>/ (the
      "Own folder" layout: ``convert`` reads a whole directory);
    * opt2d12: the .out and the .d12 of the same name (pair_out_files);
    * opt2d3: the .out, every .d12 the CLI globs for, zero-byte stand-ins
      for the sibling outputs whose names change its file names, and one
      zero-byte stand-in named as the wavefunction the CLI would copy:
      the deck never reads it, and opt2d3 copies the f9 even through a
      symlink (spike S10), so a preview of a 2 GB f9 costs nothing.
    Nothing outside ``work`` is written, and no symlink is made."""
    tree = work / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    if mode == "cif2d12":
        name = cif_name or (source_path.name if source_path else "structure.cif")
        stem = Path(name).stem
        folder = f"setup/{stem}"
        (tree / folder).mkdir(parents=True, exist_ok=True)
        if cif_bytes is not None:
            (tree / folder / name).write_bytes(cif_bytes)
        else:
            shutil.copyfile(source_path, tree / folder / name)
        return Staged(mode, calc, tree, f"{folder}/{name}", f"{folder}/{json_name(mode, calc, stem)}", folder)
    if source_path is None or rel is None:
        raise SetupError("a follow-up needs the CRYSTAL .out on disk")
    rel_dir = str(Path(rel).parent).replace("\\", "/")
    rel_dir = "" if rel_dir == "." else rel_dir
    dst = tree / rel_dir
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dst / source_path.name)
    stem = source_path.stem
    jrel = (f"{rel_dir}/" if rel_dir else "") + json_name(mode, calc, stem)
    if mode == "opt2d12":
        d12 = source_path.with_suffix(".d12")
        if d12.is_file():
            shutil.copyfile(d12, dst / d12.name)
        return Staged(mode, calc, tree, rel, jrel, rel_dir or ".")
    # opt2d3: name mirroring
    total = 0
    for p in _d12_globs(source_path):
        total += p.stat().st_size
        if total > D12_GLOB_MAX:
            raise SetupError(f"the .d12 files beside {source_path.name} are over {D12_GLOB_MAX // 2**20} MB")
        shutil.copyfile(p, dst / p.name)
    notes = []
    for sib in _anatase_siblings(source_path):
        (dst / sib).touch()
        notes.append(f"{sib} is beside {source_path.name}: the CLI names the files {stem}_*")
    f9 = find_wavefunction(source_path)
    if f9 is not None:
        (dst / f9["name"]).touch()          # an empty stand-in, never the real bytes
    return Staged(mode, calc, tree, rel, jrel, rel_dir or ".", f9, notes)


# ---------------------------------------------------------------- the commands
def batch_args(st: Staged) -> List[str]:
    """The printed command's arguments (paths relative to --dir)."""
    if st.mode == "cif2d12":
        return ["convert", "--batch", "--options_file", st.json_rel, "--cif_dir", st.out_rel,
                "--output_dir", st.out_rel]
    if st.mode == "opt2d12":
        return ["opt2d12", "--out-file", st.input_rel, "--config-file", st.json_rel, "--yes"]
    return ["opt2d3", "-i", st.input_rel, "-t", st.calc, "--config-file", st.json_rel]


def command_line(args: List[str]) -> str:
    import shlex
    return "mace " + " ".join(shlex.quote(a) for a in args)


def ask_args(st: Staged, json_out: Path, out_dir: Path) -> List[str]:
    """The interactive command with its "save the answers" switch."""
    if st.mode == "cif2d12":
        return ["convert", "--cif_dir", st.out_rel, "--output_dir", str(out_dir), "--save_options",
                "--options_file", str(json_out)]
    if st.mode == "opt2d12":
        return ["opt2d12", "--out-file", st.input_rel, "--calc-type", st.calc, "--save-options",
                "--options-file", str(json_out), "--output-dir", str(out_dir)]
    return ["opt2d3", "-i", st.input_rel, "-t", st.calc, "--save-config", "--options-file", str(json_out),
            "--output-dir", str(out_dir)]


# ---------------------------------------------------------------- the form: the CLI's questions
# The gates in front of the questions (answered No to reach them) and each field's question.
GATES = {
    "cif2d12": {"N": {"match": r"use these default settings\?", "answer": "n"},
                "ADV": {"match": r"use these default advanced settings\?", "answer": "n"},
                "SYM": {"match": r"use these default symmetry settings\?", "answer": "n"}},
    "opt2d12": {"N": {"match": r"use these exact settings for the new calculation", "answer": "n"}},
    "opt2d3": {},
}
# The answers a user gives to reach the defaults: opt2d12's follow-up keeps the parent's
# settings ("use these exact settings?" is [y/N]: the manual route answers y, spike S2).
DEFAULT_RULES = {
    "cif2d12": [],
    "opt2d12": [{"match": r"use these exact settings for the new calculation", "answer": "y"}],
    "opt2d3": [],
}


class Field:
    def __init__(self, id: str, label: str, section: str, kind: str, match: str, keys=(), gates=(),
                 modes=("cif2d12", "opt2d12"), calcs=None, unit: str = "", hint: str = ""):
        self.id, self.label, self.section, self.kind, self.match = id, label, section, kind, match
        self.keys, self.gates, self.modes, self.calcs, self.unit, self.hint = tuple(keys), tuple(gates), modes, calcs, unit, hint
        self.re = re.compile(match, re.I | re.S)

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        if self.kind == "menu":
            r = {"match": self.match, "choose": str(value)}
        elif self.kind == "yesno":
            r = {"match": self.match, "answer": "y" if value in (True, "y", "yes", "Yes", "true") else "n"}
        else:
            r = {"match": self.match, "answer": str(value)}
        return [r]


class FunctionalField(Field):
    """The functional: the CLI asks for its category, then for it."""

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        cat = functional_category(str(value))
        out = []
        if cat:
            out.append({"match": r"Select functional category", "choose": cat})
        out.append({"match": self.match, "choose": str(value)})
        return out


class SmearField(Field):
    """The smearing width: "Custom value" in the width menu, then the number."""

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        return [{"match": r"Smearing width options", "choose": "Custom value"},
                {"match": r"Enter smearing width in Hartree", "answer": str(value)}]


N, ADV, SYM = "N", "ADV", "SYM"
# id, label, section, kind, the question (regex on its text), JSON keys (display), gates, modes, calc types
FIELDS: List[Field] = [
    # cif2d12's calculation type is a question too (the other modes take it as --calc-type / -t)
    # the deck's dimensionality (plan E10): CRYSTAL, SLAB, POLYMER or MOLECULE; a molecule, slab or polymer is
    # oriented first (structures.standardize_for_dim), so the CLI's coordinate writer gets the cell it handles
    Field("dimensionality", "Dimensionality", "Structure", "menu", r"Select system dimensionality", ["dimensionality"], [N],
          ("cif2d12",), hint="CRYSTAL (3D), SLAB (2D, vacuum along z), POLYMER (1D, chain along x) or MOLECULE (0D)"),
    Field("calc_type", "Calculation", "Calculation", "menu", r"Select calculation type", ["calculation_type"], [N], ("cif2d12",)),
    Field("symmetry", "Symmetry handling", "Structure", "menu", r"Select symmetry handling", ["symmetry_handling"], [N, SYM],
          ("cif2d12",), hint="CIF keeps the file's space group; SPGLIB finds it; P1 writes every atom"),
    Field("opt_type", "Optimization type", "Optimization", "menu", r"Select optimization type", ["optimization_type", "optimization_settings.type"], [N],
          calcs=("OPT",)),
    Field("opt_convergence", "Convergence", "Optimization", "menu", r"Select convergence level", ["optimization_settings.convergence"], [N],
          calcs=("OPT",)),
    Field("freq_template", "Frequency calculation", "Frequencies", "menu", r"Select template", ["freq_settings.template"], [N],
          calcs=("FREQ",)),
    Field("numderiv", "NUMDERIV", "Frequencies", "menu", r"Numerical derivative method", ["freq_settings.numderiv"], [N], calcs=("FREQ",)),
    Field("ir", "IR intensities", "Frequencies", "yesno", r"Calculate IR intensities\?", ["freq_settings.intensities"], [N], calcs=("FREQ",)),
    Field("method", "Method", "Method", "menu", r"Select method type", ["method", "method_type"], [N]),
    FunctionalField("functional", "Functional", "Method", "menu", r"Select [\w -]*functional \(default", ["dft_functional", "functional"], [N]),
    Field("dispersion", "D3 dispersion", "Method", "yesno", r"Add D3 dispersion correction", ["dispersion", "use_dispersion"], [N]),
    Field("basis_keep", "Parent's basis set", "Method", "yesno", r"Keep the basis set from the original D12\?", ["use_original_external_basis"], [N],
          ("opt2d12",)),
    Field("basis_type", "Basis set type", "Method", "menu", r"Select basis set type", ["basis_set_type"], [], ("cif2d12",)),
    Field("basis_set", "Basis set", "Method", "menu", r"Select (internal|external) basis set", ["basis_set"], [], ("cif2d12",)),
    Field("spin", "Spin polarized", "Method", "yesno", r"Use spin-polarized calculation\?", ["spin_polarized", "is_spin_polarized"], [N, ADV]),
    Field("dft_grid", "DFT grid", "k-points and SCF", "menu", r"Select integration grid", ["dft_grid"], [N]),
    Field("tolerance", "SCF tolerances", "k-points and SCF", "menu", r"Select (tolerance level|SCF convergence level)", ["tolerances.TOLINTEG"], [N]),
    Field("maxcycle", "SCF max cycles", "k-points and SCF", "value", r"(^|\n)MAXCYCLE \[", ["maxcycle", "scf_maxcycle"], [N, ADV]),
    Field("fmixing", "FMIXING", "k-points and SCF", "value", r"(^|\n)FMIXING \[", ["fmixing"], [N, ADV], unit="%"),
    Field("smearing", "Fermi smearing", "k-points and SCF", "yesno", r"Use Fermi smearing\?", ["use_smearing"], [N, ADV]),
    SmearField("smear_width", "Smearing width", "k-points and SCF", "value", r"Smearing width options|Enter smearing width",
               ["smearing_width"], [N, ADV], unit="Ha"),
    # opt2d3
    Field("band_path", "Band path", "BAND", "menu", r"Band path definition", ["path_method"], [], ("opt2d3",), ("BAND",)),
    Field("band_format", "Path format", "BAND", "menu", r"Band path format", ["path_method"], [], ("opt2d3",), ("BAND",)),
    Field("n_points", "k-points along the path (NSUB)", "BAND", "value", r"Total number of k-points along path", ["n_points"], [],
          ("opt2d3",), ("BAND",)),
    Field("band_range", "Bands", "BAND", "menu", r"Band range selection", ["bands"], [], ("opt2d3",), ("BAND",)),
    Field("doss_projection", "Projections", "DOSS", "menu", r"DOS projection type", ["projection_type"], [], ("opt2d3",), ("DOSS",)),
    Field("doss_range", "Energy range", "DOSS", "menu", r"Energy/band range", ["energy_range"], [], ("opt2d3",), ("DOSS",)),
    Field("doss_points", "Energy points", "DOSS", "value", r"Number of energy points", ["n_points"], [], ("opt2d3",), ("DOSS",)),
    Field("npol", "NPOL", "DOSS", "value", r"NPOL value", ["npol"], [], ("opt2d3",), ("DOSS",)),
]
FIELD = {f.id: f for f in FIELDS}
# How each field is set (spike S6): "cli" = by answering the CLI's question (a CLI run);
# "patch" = written into the JSON (no question sets it); "readonly" = the CLI ignores it.
DERIVED_BY_CLI: Dict[str, str] = {f.id: "cli" for f in FIELDS}
PATCH_FIELDS = {
    # opt2d12: one k-point mesh for the follow-up (the CLI's documented hand-written key)
    "k_points_for_all_files": {"modes": ("opt2d12",), "label": "k-point mesh", "section": "k-points and SCF",
                               "hint": "written as k_points_for_all_files, the one key opt2d12 reads for a mesh"},
}
DERIVED_BY_CLI.update({k: "patch" for k in PATCH_FIELDS})
DERIVED_BY_CLI["k_points_cif2d12"] = "readonly"


def functional_category(name: str) -> Optional[str]:
    try:
        import d12_constants as C
    except ImportError:
        return None
    for cat in C.FUNCTIONAL_CATEGORIES.values():
        if name.upper() in (f.upper() for f in cat.get("functionals", [])):
            return cat["name"]
    return None


def schema() -> Dict[str, Any]:
    """What the form offers beyond the questions: the functionals by
    category and the internal basis sets with the elements each covers,
    read from the CLI's modules (never GUI literals)."""
    import d12_constants as C
    cats = [{"key": k, "name": v["name"], "functionals": list(v.get("functionals", [])),
             "basis_requirements": dict(v.get("basis_requirements", {}))}
            for k, v in C.FUNCTIONAL_CATEGORIES.items()]
    cover = {k: list(v) for k, v in getattr(C, "VERIFIED_INTERNAL_BASIS_ELEMENTS", {}).items()}
    return {"modes": list(MODES), "calc_types": {k: list(v) for k, v in CALC_TYPES.items()},
            "default_calc": DEFAULT_CALC, "functional_categories": cats,
            "internal_basis_sets": list(C.INTERNAL_BASIS_SETS), "basis_coverage": cover,
            "fields": [{"id": f.id, "label": f.label, "section": f.section, "kind": f.kind, "modes": list(f.modes),
                        "calcs": list(f.calcs) if f.calcs else None, "unit": f.unit, "hint": f.hint,
                        "how": DERIVED_BY_CLI[f.id]} for f in FIELDS],
            "patch_fields": {k: {**v, "modes": list(v["modes"])} for k, v in PATCH_FIELDS.items()},
            "derived_by_cli": DERIVED_BY_CLI, "cli_hash": clirun.cli_hash()}


def answer_rules(mode: str, answers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The answer script for these answers: the gates the fields need, then each field's answers."""
    gates, rules = [], []
    for fid, value in answers.items():
        if fid.startswith("q:"):                         # a question no field names ("Other questions")
            q = str(fid[2:])
            gates_needed = [N] if mode in ("cif2d12", "opt2d12") else []
            for g in gates_needed:
                if g not in gates:
                    gates.append(g)
            v = value if isinstance(value, dict) else {"answer": str(value)}
            rules.append({"match": re.escape(q), **{k: v[k] for k in ("answer", "choose") if k in v}})
            continue
        f = FIELD.get(fid)
        if f is None or mode not in f.modes:
            raise SetupError(f"{fid!r} is not a {mode} setting")
        for g in f.gates:
            if g not in gates:
                gates.append(g)
        rules += f.rules(mode, value)
    # the gates first (a rule found first wins), then the answers, then the defaults' own answers
    return [GATES[mode][g] for g in gates if g in GATES[mode]] + rules + DEFAULT_RULES[mode]


# ---------------------------------------------------------------- the CLI's JSON, kept as text
def dumps_like_cli(obj: Any) -> str:
    """json.dump(obj, f, indent=2) as every CLI saves its answers: no trailing newline."""
    return json.dumps(obj, indent=2)


def loads_ordered(text: str) -> Any:
    return json.loads(text, object_pairs_hook=OrderedDict)


def _cast_like(old: Any, value: Any) -> Any:
    """``value`` with the type the CLI's own value has (30.0 stays a float: 40 -> 40.0)."""
    if isinstance(old, bool):
        return value in (True, "true", "True", "y", "yes", 1, "1")
    if isinstance(old, float):
        return float(value)
    if isinstance(old, int):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(old, list) and isinstance(value, str):
        return [int(x) if x.lstrip("-").isdigit() else x for x in value.split()]
    return value


def apply_patch(json_text: str, patch: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Write ``patch`` ({key: value}, a dotted key for a nested one) into the
    CLI's JSON text, each value cast to the type the key already has, keys
    in the CLI's order; an empty patch gives the same bytes back."""
    if not patch:
        return json_text, []
    obj = loads_ordered(json_text)
    target = obj["configuration"] if isinstance(obj, dict) and isinstance(obj.get("configuration"), dict) \
        and obj.get("type") == "d3_configuration" else obj
    changed = []
    for key, value in patch.items():
        parts = key.split(".")
        d = target
        for p in parts[:-1]:
            d = d.setdefault(p, OrderedDict())
        last = parts[-1]
        new = _cast_like(d[last], value) if last in d and d[last] is not None else value
        if value is None:
            if last in d:
                del d[last]
                changed.append(key)
            continue
        if d.get(last, object()) != new or type(d.get(last)) is not type(new):
            d[last] = new
            changed.append(key)
    return dumps_like_cli(obj), changed


def json_value(obj: Any, key: str) -> Any:
    d = obj.get("configuration", obj) if isinstance(obj, dict) and obj.get("type") == "d3_configuration" else obj
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def validate(mode: str, json_text: str) -> List[str]:
    """Problems that stop the CLI from reading this JSON (empty = fine)."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return [f"not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["the settings file must be a JSON object"]
    if mode == "opt2d3":
        try:
            import d3_config
        except ImportError:
            return []
        conf = data.get("configuration", data)
        ok, errors = d3_config.validate_d3_config(conf) if isinstance(conf, dict) else (False, ["no configuration"])
        return [] if ok else list(errors)
    from . import decks
    return decks.validate(mode, data)


# ---------------------------------------------------------------- the transcript -> the form
_OPTION = re.compile(r"^\s*(\d+)\s*[:.)]\s*(\S.*?)\s*$", re.M)
_DEFAULT = re.compile(r"\(default:\s*([^)]*)\)|\[(?:1-\d+,\s*)?default=([^\]]*)\]|\[([^\]]*)\]\s*:?\s*$", re.I)


def question_options(text: str) -> List[Dict[str, str]]:
    """The menu lines of a question: the last block of numbered lines."""
    blocks, cur, last_end = [], [], -1
    for m in _OPTION.finditer(text):
        if cur and text.count("\n", last_end, m.start()) > 4:
            blocks.append(cur)
            cur = []
        cur.append(m)
        last_end = m.end()
    if cur:
        blocks.append(cur)
    if not blocks:
        return []
    # the block printed last is the one the answer picks from (a menu can be described first, then listed)
    best = blocks[-1]
    out, seen = [], set()
    for m in best:
        num, body = m.group(1), m.group(2)
        if num in seen:
            continue
        seen.add(num)
        head = re.split(r"\s+[-–]\s+", body, maxsplit=1)[0].strip()
        out.append({"num": num, "head": head, "text": body})
    return out


def question_default(text: str, prompt: str) -> Optional[str]:
    """The default the question shows: "(default: 7)", "[1-4, default=1]", "[800]", "[Y/n]";
    on the prompt line, else on the last line that shows one ("Select ... (default: 1):" then a
    menu, then "Enter your choice:")."""
    lines = [x.strip() for x in (text.strip().splitlines() + [prompt or ""]) if x.strip()]
    m = None
    for line in reversed(lines):
        for m2 in _DEFAULT.finditer(line):
            m = m2
        if m:
            tail = line
            break
    if not m:
        return None
    v = next((g for g in m.groups() if g is not None), "").strip()
    if v.upper() == "Y/N":
        return "y" if "[Y/n]" in tail else "n"
    return v


class Question:
    def __init__(self, raw: Dict[str, Any]):
        self.text = raw.get("question") or ""
        self.prompt = (raw.get("prompt") or "").strip()
        self.answer = raw.get("answer")
        self.options = question_options(self.text)
        self.default = question_default(self.text, self.prompt)
        lines = [x for x in self.text.strip().splitlines() if x.strip()]
        self.last = lines[-1].strip() if lines else self.prompt


def _matches(field: Field, q: Question) -> bool:
    return bool(field.re.search(q.text))


def _value_label(field: Field, q: Optional[Question], obj: Any) -> Tuple[Any, Optional[str]]:
    """The field's value now (from the JSON when it holds it) and the menu head it shows as."""
    val = None
    for k in field.keys:
        val = json_value(obj, k) if obj is not None else None
        if val is not None:
            break
    if field.kind == "yesno":
        if isinstance(val, bool):
            return val, "Yes" if val else "No"
        if q is not None and q.default in ("y", "n"):
            return q.default == "y", "Yes" if q.default == "y" else "No"
        return val, None
    if field.kind == "value":
        if val is None and q is not None:
            val = q.default
        return val, None if val is None else str(val)
    # a menu: the option the JSON's value names, else the question's default
    if q is not None and q.options:
        want = str(val).strip() if val is not None else None
        hit = None
        if want:
            from .answer_driver import pick
            num = pick("\n".join(f"{o['num']}: {o['text']}" for o in q.options), want)
            hit = next((o for o in q.options if o["num"] == num), None)
        if hit is None and q.default:
            hit = next((o for o in q.options if o["num"] == q.default), None)
        return val, hit["head"] if hit else (str(val) if val is not None else None)
    return val, None if val is None else str(val)


def build_form(mode: str, calc: str, survey: List[Dict[str, Any]], current: Optional[List[Dict[str, Any]]],
               obj: Any, defaults_obj: Any, answers: Dict[str, Any], patches: Dict[str, Any],
               schema_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The Setup form: one row per field whose question the CLI asks for this
    mode and calculation (the survey run, or this run when it went further),
    with the CLI's own options and default, the value the JSON holds, and
    whether it differs from the defaults."""
    qs = [Question(x) for x in (current if current else survey)]
    survey_qs = [Question(x) for x in survey]
    rows: List[Dict[str, Any]] = []
    used = set()
    for f in FIELDS:
        if mode not in f.modes or (f.calcs and calc not in f.calcs):
            continue
        q = next((x for x in qs if _matches(f, x)), None)
        if q is None:
            q = next((x for x in survey_qs if _matches(f, x)), None)
        if q is None:
            continue
        used.add(id(q))
        val, label = _value_label(f, q, obj)
        dval, dlabel = _value_label(f, q, defaults_obj)
        row = {"id": f.id, "label": f.label, "section": f.section, "kind": f.kind, "unit": f.unit, "hint": f.hint,
               "how": DERIVED_BY_CLI[f.id], "question": q.last[-160:], "default": dlabel, "value": label,
               "raw": val, "answered": f.id in answers}
        if f.id == "functional":
            cats = (schema_info or {}).get("functional_categories") or []
            row["options"] = [{"head": fn, "text": fn, "group": c["name"]} for c in cats if c["key"] not in ("HF",)
                              for fn in c["functionals"]]
        elif f.kind == "menu":
            row["options"] = [{"head": o["head"], "text": o["text"]} for o in q.options]
        elif f.kind == "yesno":
            row["options"] = [{"head": "Yes", "text": "Yes"}, {"head": "No", "text": "No"}]
        row["changed"] = f.id in answers and (str(answers[f.id]) != str(dlabel) if f.kind != "yesno"
                                              else (answers[f.id] in (True, "y", "yes", "Yes")) != (dlabel == "Yes"))
        rows.append(row)
    for key, spec in PATCH_FIELDS.items():
        if mode not in spec["modes"]:
            continue
        cur = json_value(obj, key) if obj is not None else None
        rows.append({"id": key, "label": spec["label"], "section": spec["section"], "kind": "value", "how": "patch",
                     "hint": spec["hint"], "value": None if cur is None else (" ".join(map(str, cur)) if isinstance(cur, list) else str(cur)),
                     "default": None, "changed": key in patches, "answered": key in patches, "unit": ""})
    # the questions no field names: "Other questions" (answerable by their own prompt)
    if current:
        for q in qs:
            if id(q) in used or not q.last or any(_matches(f, q) for f in FIELDS) or _is_gate(mode, q):
                continue
            fid = "q:" + q.last[-120:]
            kind = "menu" if q.options else "yesno" if re.search(r"\[[yY]/[nN]\]", q.last) else "value"
            row = {"id": fid, "label": q.last[-120:].rstrip(": "), "section": "Other questions", "kind": kind,
                   "how": "cli", "question": q.last[-160:], "default": q.default, "answered": fid in answers,
                   "value": answers.get(fid, q.answer if q.answer else q.default), "changed": fid in answers, "unit": ""}
            if kind == "menu":
                row["options"] = [{"head": o["head"], "text": o["text"]} for o in q.options]
            elif kind == "yesno":
                row["options"] = [{"head": "Yes", "text": "Yes"}, {"head": "No", "text": "No"}]
            rows.append(row)
    return rows


def _is_gate(mode: str, q: Question) -> bool:
    return any(re.search(g["match"], q.text, re.I | re.S) for g in GATES[mode].values()) or \
        bool(re.search(r"Save these options|save.*configuration", q.last, re.I))


# ---------------------------------------------------------------- one Setup run
def _decks_in(folder: Path, before: set) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".d12", ".d3")
                  and p.name not in before)


def _shrink_of(deck: str) -> Optional[str]:
    lines = deck.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "SHRINK" and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            more = lines[i + 2].strip() if nxt.split()[:1] == ["0"] and i + 2 < len(lines) else ""
            return (nxt + (" / " + more if more else "")).strip()
    return None


def survey_rules(mode: str) -> List[Dict[str, Any]]:
    """Every gate opened, every question answered with Enter: the list of questions and their menus."""
    return list(GATES[mode].values())


def ask(st: Staged, work: Path, rules: List[Dict[str, Any]], tag: str, ctx=None, extra_args=()) -> Dict[str, Any]:
    """Run the interactive command with these answers: its saved JSON, deck and questions."""
    d = work / f"ask_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rules_p, tr_p, json_p, out = d / "rules.json", d / "transcript.json", d / "answers.json", d / "out"
    out.mkdir(exist_ok=True)
    rules_p.write_text(json.dumps({"rules": rules, "max_prompts": 400}))
    args = ask_args(st, json_p, out) + list(extra_args)
    res = clirun.run(clirun.driver_argv(rules_p, tr_p, "--no-banner", *args), st.tree, ctx, timeout=180)
    try:
        tr = json.loads(tr_p.read_text())
    except (OSError, json.JSONDecodeError):
        tr = {"questions": [], "error": "no transcript"}
    decks = _decks_in(out, set())
    log = res.log.replace(str(st.tree) + os.sep, "").replace(str(d) + os.sep, "")
    return {"code": res.returncode, "json_text": json_p.read_text() if json_p.is_file() else None,
            "deck": decks[0].read_text(errors="replace") if decks else None, "deck_name": decks[0].name if decks else None,
            "questions": tr.get("questions", []), "error": tr.get("error"), "log": log, "elapsed": res.elapsed}


def run_printed(st: Staged, json_text: str, ctx=None, extra_args=()) -> Dict[str, Any]:
    """The printed command, unchanged, in the mirror: the preview deck (P == G)."""
    jp = st.tree / st.json_rel
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json_text)
    out = st.tree / st.out_rel
    before = {p.name for p in out.iterdir()} if out.is_dir() else set()
    args = batch_args(st) + list(extra_args)
    res = clirun.run(clirun.mace_argv("--no-banner", *args), st.tree, ctx, timeout=180)
    decks = _decks_in(out, before)
    log = res.log.replace(str(st.tree) + os.sep, "")
    return {"code": res.returncode, "decks": [{"name": p.name, "text": p.read_text(errors="replace")} for p in decks],
            "log": log, "elapsed": res.elapsed, "args": args}


P1_NOTE_MAX_ATOMS = 2000


def p1_note(cif: Path) -> Optional[str]:
    """Plan T13: a CIF that declares P1 (one operation) while spglib finds a
    higher group at 1e-3 Å, e.g. a structure built in the GUI and saved with
    every atom: cif2d12 keeps P1, which CRYSTAL runs more slowly. Nothing is
    changed; the note offers the symmetrized copy. None when it does not apply
    (or the structure is over 2000 atoms, where spglib is not worth a preview)."""
    try:
        from ase.io.cif import parse_cif
        blocks = [b for b in parse_cif(str(cif)) if b.has_structure()]
        if not blocks or len(blocks[-1].get_spacegroup(True).get_symop()) != 1:
            return None
        from . import structures
        p = structures.read_structure(cif)
        if len(p["numbers"]) > P1_NOTE_MAX_ATOMS:
            return None
        y = structures.symmetry(p["cell"], p["frac"], p["numbers"], p["pbc"], symprec=1e-3)
    except Exception:
        return None
    if not y.get("available") or int(y["number"]) <= 1:
        return None
    return (f"This CIF is P1; spglib finds {y['international']} ({y['number']}) at 1e-3 Å. "
            f"cif2d12 keeps P1 (a slower CRYSTAL run).")


_WARN_HEAD = re.compile(r"^(?:\[WARN\]|[^\w\s(\[]|\s)*(?:WARNING|Warning)?\s*:?\s*")


def _plain_warning(line: str) -> str:
    """A CLI warning line as a Problems message: the row's icon already says
    "warning", so the "[WARN]", the emoji and the "WARNING:" go."""
    return _WARN_HEAD.sub("", line).strip()


def problems_from(log: str, source: str = "Setup") -> List[Dict[str, Any]]:
    out = []
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("[WARN]") or s.startswith("WARNING") or s.startswith("Warning:"):
            msg = _plain_warning(s)
            if msg:
                out.append({"severity": "warning", "source": source, "message": msg[:400]})
    seen, uniq = set(), []
    for p in out:
        if p["message"] not in seen:
            seen.add(p["message"])
            uniq.append(p)
    return uniq[:20]


def refusal(log: str) -> str:
    """The CLI's own words when it wrote no deck: its error lines, else the log's end."""
    lines = [x.strip() for x in log.splitlines() if x.strip()]
    errs = [x for x in lines if re.match(r"^(\[ERR\]|\[ERROR\]|ERROR|Error|\[X\])", x) or "Aborting" in x or "requires" in x.lower()]
    return "\n".join(errs[-6:]) if errs else "\n".join(lines[-12:])


# ---------------------------------------------------------------- the whole run
class Request:
    """What one Setup run is asked for (from POST /api/setup/run)."""

    def __init__(self, body: Dict[str, Any]):
        src = body.get("source") or {}
        if not isinstance(src, dict):
            raise SetupError("source is {path} or {session, rev}")
        self.source = src
        self.calc_type = body.get("calc_type")
        self.mode = body.get("mode")
        answers = body.get("answers") or {}
        patches = body.get("patches") or {}
        if not isinstance(answers, dict) or not isinstance(patches, dict):
            raise SetupError("answers and patches are objects {field: value}")
        self.answers = OrderedDict((str(k), answers[k]) for k in answers)
        for k, v in self.answers.items():
            if not isinstance(v, (str, int, float, bool)) or (isinstance(v, str) and len(v) > 200):
                raise SetupError(f"the answer for {k} must be a short text, number or yes/no")
        self.patches = OrderedDict((str(k), patches[k]) for k in patches)
        for k in self.patches:
            if k not in PATCH_FIELDS:
                raise SetupError(f"{k} is not a setting written into the JSON by hand")
        hj = body.get("json_text")
        if hj is not None and (not isinstance(hj, str) or len(hj) > 1_000_000):
            raise SetupError("json_text must be the settings file's text")
        self.json_text = hj
        self.preview = body.get("preview", True) is not False
        # the CLI's own switches a Setup answer maps to (the defect builder's charge hint: --net-charge)
        flags = body.get("flags") or {}
        if not isinstance(flags, dict) or set(flags) - {"net_charge"}:
            raise SetupError("flags is {net_charge}")
        self.flags = {}
        if flags.get("net_charge") not in (None, ""):
            try:
                self.flags["net_charge"] = int(flags["net_charge"])
            except (TypeError, ValueError):
                raise SetupError("the net charge is a whole number")
        self.form = body.get("form", True) is not False


def source_key(mode: str, path: Path) -> Tuple:
    """What the staged input depends on: the file and the siblings the CLI reads, by stat."""
    def stamp(p: Path):
        st = os.stat(p)
        return (p.name, int(st.st_mtime_ns), int(st.st_size))
    parts = [stamp(path)]
    if mode == "opt2d12":
        d12 = path.with_suffix(".d12")
        if d12.is_file():
            parts.append(stamp(d12))
    elif mode == "opt2d3":
        parts += [stamp(p) for p in _d12_globs(path)]
        parts += [("sibling", n) for n in _anatase_siblings(path)]
        f9 = find_wavefunction(path)
        parts.append(("f9", f9["name"], f9["bytes"]) if f9 else ("f9", None))
    return (str(path.resolve()),) + tuple(parts)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_setup(req: Request, mode: str, calc: str, work: Path, source_path: Optional[Path], rel: Optional[str],
              src_key: Any, cache_get: Callable[[Any], Any], cache_put: Callable[[Any, Any, int], None], ctx=None,
              cif_bytes: Optional[bytes] = None, cif_name: Optional[str] = None,
              schema_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Stage, ask the CLI (its defaults, its questions, these answers), write
    the patches, then run the printed command on the result. Each CLI run's
    answer is cached by (input, mode, calc type, answers, CLI sources)."""
    t0 = time.monotonic()

    def progress(f, msg):
        if ctx is not None:
            ctx.progress(f, msg)

    progress(0.05, "Staging the input")
    st = stage(mode, calc, work, source_path, rel, cif_bytes, cif_name)
    base_key = (src_key, mode, calc, clirun.cli_hash())
    cmd_name = {"cif2d12": "mace convert", "opt2d12": "mace opt2d12", "opt2d3": "mace opt2d3"}[mode]

    def cached_ask(tag, rules, frac, msg, extra=()):
        key = ("setup-ask", base_key, json.dumps(rules, sort_keys=True), list(extra))
        hit = cache_get(key)
        if hit is not None:
            return hit
        progress(frac, msg)
        r = ask(st, work, rules, tag, ctx, extra)
        cache_put(key, r, len(json.dumps(r)))
        return r

    if mode == "cif2d12" and calc != DEFAULT_CALC[mode] and "calc_type" not in req.answers:
        # convert asks for the calculation type: SP and FREQ are answers, like any other
        req.answers = OrderedDict([("calc_type", calc)] + list(req.answers.items()))
    if req.json_text is not None and not req.form:
        defaults = survey = {"json_text": None, "questions": [], "log": ""}        # a hand-written file: the deck only
    else:
        defaults = cached_ask("defaults", list(DEFAULT_RULES[mode]), 0.15, f"Asking {cmd_name} for its defaults")
        survey = defaults if mode == "opt2d3" else cached_ask("survey", survey_rules(mode), 0.25,
                                                              f"Reading {cmd_name}'s questions")
    problems: List[Dict[str, Any]] = []
    current = None
    asked = None
    if req.json_text is not None:
        final = req.json_text
        source_of_json = "hand"
    elif req.answers or req.flags:
        extra = ["--net-charge", str(req.flags["net_charge"])] if "net_charge" in req.flags and mode != "opt2d3" else []
        asked = cached_ask("answers", answer_rules(mode, req.answers), 0.35, f"Answering {cmd_name}'s questions", extra)
        if asked.get("error"):
            raise SetupError(f"{cmd_name} did not ask what Setup expected: {asked['error']}. "
                             f"Set it through the CLI: {cmd_name} …")
        final = asked.get("json_text")
        current = asked.get("questions")
        source_of_json = "answers"
    else:
        final = defaults.get("json_text")
        source_of_json = "defaults"
    base_run = asked if asked is not None else defaults
    if final is None:
        why = refusal(base_run.get("log") or "")
        out = _refused(mode, calc, st, why, base_run, defaults, survey, schema_info, req)
        out["elapsed"] = round(time.monotonic() - t0, 3)
        return out
    final, changed = apply_patch(final, req.patches)
    bad = validate(mode, final)
    if bad:
        raise SetupError("; ".join(bad))
    try:
        obj = json.loads(final)
    except json.JSONDecodeError:
        obj = None
    try:
        defaults_obj = json.loads(defaults["json_text"]) if defaults.get("json_text") else None
    except json.JSONDecodeError:
        defaults_obj = None
    form = build_form(mode, calc, survey.get("questions") or [], current, obj, defaults_obj, req.answers, req.patches,
                      schema_info) if req.form else []
    args = batch_args(st)
    result: Dict[str, Any] = {
        "mode": mode, "calc_type": calc, "json_text": final, "json_sha256": _sha(final),
        "defaults_sha256": _sha(defaults["json_text"]) if defaults.get("json_text") else None,
        "defaults_json": defaults.get("json_text"),
        "json_source": source_of_json, "patched": changed, "form": form,
        "command": command_line(args), "argv": args, "json_rel": st.json_rel, "input_rel": st.input_rel,
        "f9": st.f9, "notes": st.notes, "problems": problems, "deck": None, "deck_name": None, "decks": [],
        "transcript": [{"q": Question(q).last[-160:], "a": q.get("answer")} for q in (current or [])][-60:],
    }
    problems += problems_from(base_run.get("log") or "")
    if mode == "opt2d3" and st.f9:
        problems.append({"severity": "info", "source": "Setup", "field": None,
                         "message": f"Preview used an empty stand-in for {st.f9['name']} ({_mb(st.f9['bytes'])}); the deck does not "
                                    f"read it. Running the command copies it to {d3_base(Path(st.input_rel).stem)}_"
                                    f"{calc.lower()}.f9."})
    for n in st.notes:
        problems.append({"severity": "info", "source": "Setup", "message": n})
    if mode == "cif2d12":
        note = cache_get(("setup-p1", base_key))
        if note is None:
            note = p1_note(st.tree / st.input_rel) or ""
            cache_put(("setup-p1", base_key), note, 256)
        if note:
            # the CLI's bare "Space group mismatch!" says the same thing, with less
            problems[:] = [p for p in problems if not p["message"].lower().startswith("space group mismatch")]
            problems.append({"severity": "info", "source": "Setup", "message": note, "action_id": "save_symmetrized"})
    if req.preview:
        key = ("setup-preview", base_key, _sha(final))
        p = cache_get(key)
        if p is None:
            progress(0.6, f"Running {command_line(args)}")
            p = run_printed(st, final, ctx)
            cache_put(key, p, len(json.dumps(p)))
        progress(0.9, "Reading the deck")
        if p["code"] != 0 or not p["decks"]:
            why = refusal(p["log"])
            result.update(deck=None, deck_name=None, log=p["log"][-6000:], refused=why)
            problems.insert(0, {"severity": "error", "source": "Setup", "message": f"The CLI wrote no deck: {why.splitlines()[-1] if why else ''}"})
        else:
            main = p["decks"][0]
            result.update(deck=main["text"], deck_name=main["name"], decks=p["decks"], log=p["log"][-6000:])
            if mode == "cif2d12":
                result["shrink"] = _shrink_of(main["text"])
            # the control route: the interactive run's own deck against the printed command's
            ctrl = base_run.get("deck")
            if source_of_json != "hand" and not req.patches and ctrl is not None:
                same = ctrl == main["text"]
                result["control_same"] = same
                if not same:
                    problems.append({"severity": "warning", "source": "Setup",
                                     "message": f"{cmd_name} writes a different deck when it asks than from its saved "
                                                "settings (a CLI finding: the saved-settings deck is shown)"})
    result["elapsed"] = round(time.monotonic() - t0, 3)
    return result


def _mb(n: int) -> str:
    return f"{n / 2**20:.0f} MB" if n >= 2**20 else f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


def _refused(mode, calc, st, why, base_run, defaults, survey, schema_info, req) -> Dict[str, Any]:
    """The CLI wrote no settings: its refusal is the answer (no f9, no seekpath, ...)."""
    msg = why.splitlines()[-1] if why else "the CLI stopped"
    if mode == "opt2d3" and st.f9 is None and re.search(r"wavefunction|fort\.9", base_run.get("log") or "", re.I):
        name = Path(st.input_rel).name
        stem = Path(st.input_rel).stem
        msg = (f"opt2d3 writes no {calc} deck without {d3_base(stem)}_sp.f9 (or fort.9) beside {name} "
               f"(the wavefunction of the SCF run)")
    args = batch_args(st)
    return {"mode": mode, "calc_type": calc, "json_text": None, "json_sha256": None, "form": [], "refused": why,
            "deck": None, "deck_name": None, "decks": [], "log": (base_run.get("log") or "")[-6000:],
            "command": command_line(args), "argv": args, "json_rel": st.json_rel, "input_rel": st.input_rel,
            "f9": st.f9, "notes": st.notes, "save_blocked": msg,
            "problems": [{"severity": "error", "source": "Setup", "message": msg}]}
