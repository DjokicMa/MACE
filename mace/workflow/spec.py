"""Workflow specs: a whole workflow, with per-step settings, as one JSON file.

The GUI's workflow builder writes a spec; the CLI turns it into the plan
today's code writes (``mace workflow --spec SPEC --write-plan``) and launches
it (``mace submit --progress-plan PLAN`` for decks, ``mace workflow
--execute PLAN`` for CIFs). The GUI never writes plans, scripts or decks.

Format (version 1)::

    {"version": 1,
     "label": "perovskites-b3lyp",                       # optional
     "inputs": {"type": "cif" | "d12",
                "files": ["inputs/A.cif", ...]           # or "directory": "inputs"
               },
     "template": "full_electronic",                      # or
     "sequence": ["OPT", "SP", "BAND", "DOSS"],
     "steps": {                                          # all optional
       "OPT_1": {"settings_file": "opt.json", "resources": {"ntasks": 32}},
       "SP_2":  {"overrides": {"method_modifications": {"new_functional": "HSE06"}}},
       "BAND_3": {"settings_file": "band.json", "resources": {"memory": "48G"}},
       "SP2_5": {"overrides": {"soc": true, "soc_ecp": "inpsoc", "smear": 0.005}}},
     "execution": {"max_jobs": 100, "walltime": "2:00:00",
                   "auto_progression": true,
                   "job_arrays": {"first_step": true, "max_running": 20}}}

Relative paths are relative to the spec file. A step without settings is the
template's step exactly: the plan, job scripts and decks are byte for byte
what ``mace submit --progress TEMPLATE`` / ``mace workflow --quick-start``
write (d12 follow-ups inherit their parent's settings, as always).

``settings_file`` is a Calculation Setup JSON: the cif2d12 options of the
first step of a CIF start (the plan's cif_conversion_config), else the
opt2d12 / opt2d3 config the engine passes to the generator
(``expert_config_file``; copied to ``workflow_configs/spec_<id>/``).
``overrides`` are merged into the step's plan entry; only keys the engine
reads are accepted, and not next to a ``settings_file`` (the engine
generates such a step from that file alone), except on a CIF start's first
step, whose overrides are merged into the conversion settings. ``resources`` set the step's job script.

Spin-orbit coupling is a setting of SP steps only (CRYSTAL23 runs a
two-component SCF as a single point, manual p. 166): the overrides ``soc``
(true, or false for a scalar step after a SOC parent), ``soc_ecp`` (inpsoc |
stutsc), ``soc_basis`` (pob-tzvp-rev2-soc), ``soc_guess`` (guesspso |
guesspnoso), ``smear`` (hartree) and ``soc_fmixing`` (the SOC deck's
FMIXING, 0-100), the keys of opt2d12's settings file,
checked as opt2d12 checks them. On a CIF start's first SP step they go into
the conversion settings (mace convert writes the SOC deck; no ``soc_guess``,
a CIF has no parent run). On any other step they are refused, with the
reason. A step that cannot start from a SOC run (soc_plan_problems) is
named in the check's ``warnings`` and by ``--write-plan``.
"""
import copy
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from mace.workflow.common.constants import CALC_TYPES, WORKFLOW_TEMPLATES

SPEC_VERSION = 1
D12_TYPES = ("OPT", "SP", "FREQ")
D3_TYPES = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS")
# plan step keys the engine reads (engine.py: _build_numbered_calc_config,
# generate_d3_calculation, the FREQ generator)
# spin-orbit coupling: the opt2d12 settings-file keys the engine passes on
# (SP steps only; Crystal_d12/soc_ecp.py checks them)
SOC_KEYS = ("soc", "soc_ecp", "soc_basis", "soc_guess", "smear", "soc_fmixing")
OVERRIDE_KEYS = {
    "OPT": ("optimization_settings", "optimization_type", "method_modifications"),
    "SP": ("method_modifications",) + SOC_KEYS,
    "FREQ": ("frequency_settings", "method_modifications"),
    "d3": ("d3_config", "d3_settings"),
}
METHOD_MODIFICATION_KEYS = ("new_functional", "use_dispersion")
RESOURCE_KEYS = ("ntasks", "walltime", "memory", "memory_per_cpu", "account", "constraint",
                 "partition", "exclude")
STEP_KEYS = ("settings_file", "overrides", "resources", "note")
EXECUTION_KEYS = ("max_jobs", "walltime", "auto_progression", "job_arrays")
TOP_KEYS = ("version", "label", "inputs", "template", "sequence", "steps", "execution", "description")
LABEL_RE = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")
WALLTIME_RE = re.compile(r"^(\d+-)?\d{1,3}(:\d{2}){0,2}$")
MEMORY_RE = re.compile(r"^\d+(\.\d+)?[KMGT]?B?$", re.I)
NAME_RE = re.compile(r"^[A-Za-z0-9._,\-\[\]&|()!*]{1,256}$")
STEP_TYPE_RE = re.compile(r"^(OPT|SP)(\d+)$")


class SpecError(ValueError):
    """The spec cannot be used; the message says why (nothing was written)."""


# ------------------------------------------------------------------ load
def load_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and validate a spec file; returns the normalized spec (absolute
    paths, ``sequence`` filled in from ``template``)."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except OSError as e:
        raise SpecError(f"cannot read the spec {path}: {e}")
    except ValueError as e:
        raise SpecError(f"{path} is not JSON: {e}")
    spec = validate_spec(raw, path.resolve().parent)
    spec["_source"] = str(path.resolve())
    return spec


def _is_step_type(t: str) -> bool:
    if t in CALC_TYPES:
        return True
    m = STEP_TYPE_RE.match(t)
    return bool(m and int(m.group(2)) >= 2)


def step_kind(calc_type: str) -> str:
    base = calc_type.rstrip("0123456789") or calc_type
    return "d3" if base in D3_TYPES else base


def validate_spec(raw: Any, base_dir: Union[str, Path]) -> Dict[str, Any]:
    """Check everything that can be checked without writing anything."""
    base_dir = Path(base_dir)
    if not isinstance(raw, dict):
        raise SpecError("a workflow spec is a JSON object")
    unknown = sorted(set(raw) - set(TOP_KEYS))
    if unknown:
        raise SpecError(f"unknown spec key(s): {', '.join(unknown)} (allowed: {', '.join(TOP_KEYS)})")
    if raw.get("version") != SPEC_VERSION:
        raise SpecError(f"spec version {raw.get('version')!r} is not supported (this MACE reads "
                        f"version {SPEC_VERSION})")
    spec: Dict[str, Any] = {"version": SPEC_VERSION}
    label = raw.get("label")
    if label is not None and not (isinstance(label, str) and LABEL_RE.match(label)):
        raise SpecError("label: letters, digits and . _ + - only, at most 64 characters")
    spec["label"] = label

    # inputs
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        raise SpecError('inputs: {"type": "cif" or "d12", "files": [...]} or "directory"')
    kind = inputs.get("type")
    if kind not in ("cif", "d12"):
        raise SpecError('inputs.type must be "cif" or "d12"')
    files: List[Path] = []
    if inputs.get("files") is not None:
        if not isinstance(inputs["files"], list) or not inputs["files"]:
            raise SpecError("inputs.files must be a non-empty list of paths")
        files = [(base_dir / str(f)).resolve() for f in inputs["files"]]
    elif inputs.get("directory") is not None:
        d = (base_dir / str(inputs["directory"])).resolve()
        if not d.is_dir():
            raise SpecError(f"inputs.directory {d} is not a directory")
        files = sorted(d.glob(f"*.{kind}"))
    else:
        raise SpecError("inputs needs files or directory")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SpecError(f"input file(s) not found: {', '.join(missing)}")
    wrong = [f.name for f in files if f.suffix.lower() != f".{kind}"]
    if wrong:
        raise SpecError(f"inputs.type is {kind}, but these are not .{kind} files: {', '.join(wrong)}")
    if not files:
        raise SpecError(f"no .{kind} files in inputs")
    if len({f for f in files}) != len(files):
        raise SpecError("inputs.files lists a file twice")
    if kind == "d12" and len({f.parent for f in files}) > 1:
        raise SpecError("the decks of a d12 start must be in one folder (they are submitted "
                        "together, from there)")
    spec["inputs"] = {"type": kind, "files": [str(f) for f in files]}

    # sequence
    if raw.get("template") is not None and raw.get("sequence") is not None:
        raise SpecError("give template or sequence, not both")
    if raw.get("template") is not None:
        if raw["template"] not in WORKFLOW_TEMPLATES:
            raise SpecError(f"unknown template {raw['template']!r} (one of: "
                            f"{', '.join(sorted(WORKFLOW_TEMPLATES))})")
        sequence = list(WORKFLOW_TEMPLATES[raw["template"]])
        spec["template"] = raw["template"]
    else:
        sequence = raw.get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise SpecError("sequence must be a non-empty list of calculation types")
        sequence = [str(s) for s in sequence]
        bad = [s for s in sequence if not _is_step_type(s)]
        if bad:
            raise SpecError(f"unknown calculation type(s) in sequence: {', '.join(bad)} "
                            f"(types: {', '.join(CALC_TYPES)}, or OPT2/SP2/...)")
        if len(set(sequence)) != len(sequence):
            raise SpecError("a calculation type appears twice in sequence; number repeats "
                            "(OPT, OPT2, ...)")
        spec["template"] = None
    if kind == "cif" and sequence[0] not in ("OPT", "SP"):
        raise SpecError("a CIF start begins with OPT or SP (the CIFs are converted to that deck)")
    for i, t in enumerate(sequence[1:], 1):
        deps = CALC_TYPES.get(t.rstrip("0123456789"), {}).get("depends_on") or []
        if deps and not any((s.rstrip("0123456789") or s) in deps for s in sequence[:i]):
            raise SpecError(f"{t} needs one of {', '.join(deps)} before it in sequence")
    spec["sequence"] = sequence

    # steps
    keys = [f"{t}_{i + 1}" for i, t in enumerate(sequence)]
    steps_raw = raw.get("steps") or {}
    if not isinstance(steps_raw, dict):
        raise SpecError("steps must be an object keyed by step (OPT_1, SP_2, ...)")
    unknown = [k for k in steps_raw if k not in keys]
    if unknown:
        raise SpecError(f"steps {', '.join(unknown)} are not in the sequence (its steps: {', '.join(keys)})")
    steps: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(keys):
        entry = steps_raw.get(key)
        if entry is None:
            continue
        steps[key] = _validate_step(key, sequence[i], i, entry, kind, base_dir)
    spec["steps"] = steps
    soc = two_component_orbitals_problem(spec)
    if soc:
        raise SpecError(soc)

    # execution
    ex = raw.get("execution") or {}
    if not isinstance(ex, dict):
        raise SpecError("execution must be an object")
    unknown = sorted(set(ex) - set(EXECUTION_KEYS))
    if unknown:
        raise SpecError(f"unknown execution key(s): {', '.join(unknown)} (allowed: {', '.join(EXECUTION_KEYS)})")
    out_ex: Dict[str, Any] = {"max_jobs": None, "walltime": None, "auto_progression": True,
                              "job_arrays": None}
    if "max_jobs" in ex:
        if not isinstance(ex["max_jobs"], int) or isinstance(ex["max_jobs"], bool) or ex["max_jobs"] < 1:
            raise SpecError("execution.max_jobs must be a whole number of 1 or more")
        out_ex["max_jobs"] = ex["max_jobs"]
    if ex.get("walltime") is not None:
        out_ex["walltime"] = _walltime(ex["walltime"], "execution.walltime")
    if "auto_progression" in ex:
        if not isinstance(ex["auto_progression"], bool):
            raise SpecError("execution.auto_progression must be true or false")
        out_ex["auto_progression"] = ex["auto_progression"]
    ja = ex.get("job_arrays")
    if ja:
        if not isinstance(ja, dict) or set(ja) - {"first_step", "max_running"}:
            raise SpecError('execution.job_arrays: {"first_step": true, "max_running": N}')
        mr = ja.get("max_running")
        if mr is not None and (not isinstance(mr, int) or isinstance(mr, bool) or mr < 1):
            raise SpecError("execution.job_arrays.max_running must be a whole number of 1 or more")
        if ja.get("first_step", True):
            out_ex["job_arrays"] = {"first_step": True, "max_running": mr}
    spec["execution"] = out_ex
    spec["description"] = raw.get("description")
    return spec


def _step_is_two_component(spec: Dict[str, Any], i: int) -> Optional[str]:
    """Why step ``i`` of a validated spec is a two-component (SOC) run, or None.

    Its settings file or its overrides ask for SOC ("soc": true; the GUI
    builder writes the overrides form, and on a CIF start's first step they
    are the conversion settings); "soc": false makes a scalar step; a step
    with neither inherits its parent's settings, so the parent decides; the
    first step of a d12 start is its decks (a TWOCOMPON block)."""
    from mace.utils.calc_detection import is_two_component_deck
    from mace.workflow.status import parent_index
    sequence = spec["sequence"]
    while i is not None:
        base = sequence[i].rstrip("0123456789") or sequence[i]
        if base not in ("SP",) and not (i == 0 and spec["inputs"]["type"] == "d12"):
            return None          # only a single point runs in 2c
        key = f"{sequence[i]}_{i + 1}"
        sf = (spec["steps"].get(key) or {}).get("settings_file")
        if sf:
            try:
                data = json.loads(Path(sf).read_text())
            except (OSError, ValueError):
                return None
            return f"{key} asks for SOC" if isinstance(data, dict) and data.get("soc") is True else None
        ov = (spec["steps"].get(key) or {}).get("overrides") or {}
        if isinstance(ov, dict) and "soc" in ov:
            return f"{key} asks for SOC" if ov.get("soc") is True else None
        if i == 0:
            if spec["inputs"]["type"] != "d12":
                return None
            names = []
            for f in spec["inputs"]["files"]:
                try:
                    if is_two_component_deck(Path(f).read_text(errors="replace")):
                        names.append(Path(f).name)
                except OSError:
                    pass
            if not names:
                return None
            more = f" and {len(names) - 3} more" if len(names) > 3 else ""
            return f"{key} runs the TWOCOMPON deck(s) {', '.join(names[:3])}{more}"
        i = parent_index(list(sequence), i)
    return None


def two_component_orbitals_problem(spec: Dict[str, Any]) -> Optional[str]:
    """Why an ORBITALS step of a validated spec cannot run (its wavefunction
    is from a two-component SCF; CRYSTAL23 crashes on it), or None."""
    from mace.orbitals.kchoice import SOC_REFUSAL
    from mace.workflow.status import parent_index
    sequence = spec["sequence"]
    for i, t in enumerate(sequence):
        if (t.rstrip("0123456789") or t) != "ORBITALS":
            continue
        j = parent_index(list(sequence), i)
        why = None if j is None else _step_is_two_component(spec, j)
        if why:
            return f"{t}_{i + 1} reads the wavefunction of {sequence[j]}_{j + 1}, and {why}: {SOC_REFUSAL}"
    return None


def _walltime(v: Any, where: str) -> str:
    v = str(v).strip()
    if not WALLTIME_RE.match(v):
        raise SpecError(f"{where}: {v!r} is not a SLURM time (D-HH:MM:SS, HH:MM:SS, MM)")
    return v


def _validate_step(key: str, calc_type: str, index: int, entry: Any, input_kind: str,
                   base_dir: Path) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise SpecError(f"steps.{key} must be an object")
    unknown = sorted(set(entry) - set(STEP_KEYS))
    if unknown:
        raise SpecError(f"steps.{key}: unknown key(s) {', '.join(unknown)} (allowed: {', '.join(STEP_KEYS)})")
    out: Dict[str, Any] = {}
    first_cif = index == 0 and input_kind == "cif"
    deck_start = index == 0 and input_kind == "d12"
    if deck_start and (entry.get("settings_file") or entry.get("overrides")):
        raise SpecError(f"steps.{key}: the first step of a d12 start runs the decks as they are; "
                        f"only its resources can be set")
    if entry.get("settings_file"):
        sf = (base_dir / str(entry["settings_file"])).resolve()
        try:
            data = json.loads(sf.read_text())
        except OSError:
            raise SpecError(f"steps.{key}.settings_file {sf} not found")
        except ValueError as e:
            raise SpecError(f"steps.{key}.settings_file {sf} is not JSON: {e}")
        if not isinstance(data, dict):
            raise SpecError(f"steps.{key}.settings_file must hold a JSON object")
        if first_cif and data.get("calculation_type") not in (None, calc_type):
            raise SpecError(f"steps.{key}.settings_file is for a {data.get('calculation_type')} deck, "
                            f"but this step is {calc_type}")
        if step_kind(calc_type) == "d3":
            # opt2d3's file names its run (the envelope, its configuration, or both): the
            # engine hands it to CRYSTALOptToD3 as this step's
            base = re.sub(r"\d+$", "", calc_type)
            conf = data.get("configuration") if isinstance(data.get("configuration"), dict) else {}
            for t in {data.get("calculation_type"), conf.get("calculation_type")} - {None}:
                if t != base:
                    raise SpecError(f"steps.{key}.settings_file is for a {t} step, but this step is {calc_type}")
        out["settings_file"] = str(sf)
    ov = entry.get("overrides")
    if ov:
        if not isinstance(ov, dict):
            raise SpecError(f"steps.{key}.overrides must be an object")
        _check_soc(f"steps.{key}.overrides", calc_type, ov, first_cif)
        if not first_cif and entry.get("settings_file"):
            # the engine generates the step from its settings file alone
            # (--config-file); overrides beside it would be shown by the
            # check and never used
            raise SpecError(f"steps.{key}: with a settings file the engine reads only that file for "
                            f"this step; put the overrides ({', '.join(sorted(ov))}) in the settings "
                            f"file, or drop the settings file")
        if not first_cif:
            allowed = OVERRIDE_KEYS.get(step_kind(calc_type), ())
            bad = sorted(set(ov) - set(allowed))
            if bad:
                raise SpecError(f"steps.{key}.overrides: {', '.join(bad)} are not read for a "
                                f"{calc_type} step (it reads: {', '.join(allowed) or 'nothing'})")
            mm = ov.get("method_modifications")
            if mm is not None:
                if not isinstance(mm, dict) or set(mm) - set(METHOD_MODIFICATION_KEYS):
                    raise SpecError(f"steps.{key}.overrides.method_modifications: "
                                    f"{', '.join(METHOD_MODIFICATION_KEYS)} only")
        out["overrides"] = copy.deepcopy(ov)
    res = entry.get("resources")
    if res:
        if not isinstance(res, dict):
            raise SpecError(f"steps.{key}.resources must be an object")
        bad = sorted(set(res) - set(RESOURCE_KEYS))
        if bad:
            raise SpecError(f"steps.{key}.resources: unknown key(s) {', '.join(bad)} "
                            f"(allowed: {', '.join(RESOURCE_KEYS)})")
        if "memory" in res and "memory_per_cpu" in res:
            raise SpecError(f"steps.{key}.resources: set memory or memory_per_cpu, not both")
        clean: Dict[str, Any] = {}
        for k, v in res.items():
            if k == "ntasks":
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    raise SpecError(f"steps.{key}.resources.ntasks must be a whole number of 1 or more")
                clean[k] = v
            elif k == "walltime":
                clean[k] = _walltime(v, f"steps.{key}.resources.walltime")
            elif k in ("memory", "memory_per_cpu"):
                if not MEMORY_RE.match(str(v)):
                    raise SpecError(f"steps.{key}.resources.{k}: {v!r} is not a memory size (e.g. 48G)")
                clean[k] = str(v)
            else:
                if not NAME_RE.match(str(v)):
                    raise SpecError(f"steps.{key}.resources.{k}: {v!r} has characters SLURM does not take")
                clean[k] = str(v)
        out["resources"] = clean
    if entry.get("note") is not None:
        out["note"] = str(entry["note"])
    return out


def _soc_ecp():
    """Crystal_d12/soc_ecp.py, the deck writer's own SOC checks."""
    import sys
    d = str(Path(__file__).resolve().parents[2] / "Crystal_d12")
    if d not in sys.path:
        sys.path.insert(0, d)
    import soc_ecp
    return soc_ecp


def soc_refusal(calc_type: str) -> Optional[str]:
    """Why a step of ``calc_type`` has no SOC setting (None for SP), in the
    deck writer's words where it has them."""
    base = calc_type.rstrip("0123456789") or calc_type
    if base == "SP":
        return None
    E = _soc_ecp()
    if base == "OPT":
        return E.unsupported_message("OPTGEOM")
    if base == "FREQ":
        return E.unsupported_message("FREQCALC")
    return ("A properties step has no SOC setting: it reads the wavefunction of the run it is "
            "built from, so it is a two-component run when that run is a SOC single point "
            "(CRYSTAL23 manual sec. 6.4, p. 183)")


def _check_soc(where: str, calc_type: str, ov: Dict[str, Any], first_cif: bool) -> None:
    """The SOC keys of a step's overrides: SP steps only, values as the
    deck writer checks them, the options only with "soc": true."""
    keys = [k for k in SOC_KEYS if k in ov]
    if not keys:
        return
    why = soc_refusal(calc_type)
    if why:
        raise SpecError(f"{where}: {', '.join(keys)}: {calc_type} has no spin-orbit (SOC) setting. {why}")
    soc = ov.get("soc")
    if "soc" in ov and not isinstance(soc, bool):
        raise SpecError(f"{where}.soc must be true or false, not {soc!r}")
    options = [k for k in keys if k != "soc"]
    if options and soc is not True:
        raise SpecError(f"{where}: {', '.join(options)} set up a SOC deck: only with \"soc\": true")
    E = _soc_ecp()
    checks = {"soc_ecp": E._checked_ecp_form, "soc_basis": E._checked_soc_basis,
              "soc_guess": E._checked_guess, "smear": E.checked_smear,
              "soc_fmixing": E._checked_fmixing}
    for k in options:
        try:
            if k == "smear" and isinstance(ov[k], bool):
                raise E.SocError(f"smear must be a width in hartree above 0 (SMEAR, manual p. 126), "
                                 f"not {ov[k]!r}")
            checks[k](ov[k])
        except E.SocError as e:
            raise SpecError(f"{where}.{k}: {e}")
    if first_cif and ov.get("soc_guess") is not None:
        raise SpecError(f"{where}.soc_guess: {E.NO_PARENT_FOR_GUESS}")


def soc_warnings(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """{step key: warnings}: the planner's own check (soc_plan_problems) of
    the steps that cannot start from a SOC single point of this spec."""
    seq = spec["sequence"]
    cfgs: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(seq):
        key = f"{t}_{i + 1}"
        ov = (spec["steps"].get(key) or {}).get("overrides") or {}
        if ov.get("soc") is True:
            cfgs[key] = {"soc": True}
    out: Dict[str, List[str]] = {f"{t}_{i + 1}": [] for i, t in enumerate(seq)}
    if not cfgs:
        return out
    from mace.workflow.planner import soc_plan_problems
    errors, warnings = soc_plan_problems(list(seq), cfgs)
    if errors:
        # what the planner would refuse at --execute (or the step's
        # generation would refuse) is refused by the check too
        raise SpecError("; ".join(errors))
    for w in warnings:
        step = w.split(" follows", 1)[0]
        for i, t in enumerate(seq):
            if t == step:
                out[f"{t}_{i + 1}"].append(w)
    return out


# ------------------------------------------------------------------ apply
def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def _set_resources(step_cfg: Dict[str, Any], res: Dict[str, Any]) -> None:
    slurm = step_cfg.get("slurm_config") or {}
    targets = []
    if isinstance(slurm.get("resources"), dict):
        targets.append(slurm["resources"])
    scripts = [s for s in (slurm.get("scripts") or {}).values() if isinstance(s, dict)]
    targets += [s["resources"] for s in scripts if isinstance(s.get("resources"), dict)
                and not any(s["resources"] is t for t in targets)]
    for t in targets:
        for k, v in res.items():
            if k == "exclude":
                continue
            if k == "memory":
                t.pop("memory_per_cpu", None)
                t["mem_form"] = "mem"
            elif k == "memory_per_cpu":
                t.pop("memory", None)
                t["mem_form"] = "mem_per_cpu"
            t[k] = v
    if "exclude" in res:
        for s in scripts:
            s["node_exclusion"] = res["exclude"]


def spec_dir(work_dir: Union[str, Path], workflow_id: str) -> Path:
    return Path(work_dir) / "workflow_configs" / f"spec_{workflow_id[len('workflow_'):]}"


def apply_spec(plan: Dict[str, Any], spec: Dict[str, Any], work_dir: Optional[Union[str, Path]] = None,
               workflow_id: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Put the spec's step settings and resources into ``plan`` (in place).
    With ``work_dir`` and ``workflow_id`` the settings files are copied to
    ``workflow_configs/spec_<id>/`` and the plan points there; without, the
    plan points at the spec's own files (a dry run). Returns the origins of
    what was set: {step: {"settings": origin, <resource key>: "spec"}}."""
    origins: Dict[str, Dict[str, str]] = {}
    seq = spec["sequence"]
    cif = spec["inputs"]["type"] == "cif"
    dest_dir = spec_dir(work_dir, workflow_id) if work_dir is not None and workflow_id else None
    for i, t in enumerate(seq):
        key = f"{t}_{i + 1}"
        entry = spec["steps"].get(key) or {}
        cfg = plan["step_configurations"][key]
        o: Dict[str, str] = {}
        if entry.get("settings_file"):
            src = Path(entry["settings_file"])
            if i == 0 and cif:
                plan["cif_conversion_config"] = json.loads(src.read_text())
            else:
                target = src
                if dest_dir is not None:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    target = dest_dir / f"{key}_config.json"
                    shutil.copyfile(src, target)
                cfg["expert_config_file"] = str(target)
            o["settings"] = "spec settings_file"
        if entry.get("overrides"):
            if i == 0 and cif:
                _merge(plan["cif_conversion_config"], entry["overrides"])
            else:
                _merge(cfg, entry["overrides"])
            o["settings"] = "spec overrides" if "settings" not in o else "spec settings_file + overrides"
        if entry.get("resources"):
            _set_resources(cfg, entry["resources"])
            for k in entry["resources"]:
                o[k] = "spec"
        origins[key] = o
    return origins


# ------------------------------------------------------------------ defaults
def _base_plan(spec: Dict[str, Any], work_dir: Union[str, Path]) -> Dict[str, Any]:
    from mace.run_mace import create_quick_workflow_plan
    files = [Path(f) for f in spec["inputs"]["files"]]
    ex = spec["execution"]
    args = SimpleNamespace(max_jobs=ex["max_jobs"], work_dir=str(work_dir),
                           array=bool(ex["job_arrays"]),
                           max_running=(ex["job_arrays"] or {}).get("max_running"))
    return create_quick_workflow_plan(files[0].parent, files, spec["inputs"]["type"],
                                      list(spec["sequence"]), args)


def _site_origin(site, key: str, kind: str, calc_type: str, value: Any) -> str:
    sources = getattr(site, "sources", {}) or {}
    raw = getattr(site, "res_raw", {}) or {}
    names = {"memory": "mem", "memory_per_cpu": "mem_per_cpu"}
    rk = names.get(key, key)
    typed = raw.get(calc_type) or {}
    if rk in typed and str(typed[rk]) == str(value):
        return f"site [resources.{calc_type}]"
    if key in ("account", "partition", "module", "modules"):
        tag = sources.get(key if key != "modules" else "crystal", "built-in")
        return "builtin" if tag == "built-in" else ("env" if str(tag).startswith("env") else "site")
    for level, label in ((("properties", "site [resources.properties]"),) if kind == "prop" else ()) + \
            (("", "site [resources]"),):
        tag = sources.get(f"resources.{level + '.' if level else ''}{rk}")
        if tag:
            return "env" if str(tag).startswith("env") else label
    return "builtin"


def resolve_defaults(spec: Dict[str, Any], work_dir: Union[str, Path] = ".") -> Dict[str, Any]:
    """Each step's settings and resources as the plan will have them, with
    where every value comes from: builtin, site ([resources],
    [resources.properties], [resources.<TYPE>] where a site sets one), env,
    parent (inherited from the step it is built from) or spec. Writes
    nothing."""
    from mace.submission import site as site_mod
    try:
        site = site_mod.load()
    except Exception:
        site = None
    plan = _base_plan(spec, work_dir)
    if spec["execution"]["walltime"]:
        from mace.workflow.manual_progress import _apply_walltime
        _apply_walltime(plan["step_configurations"], spec["execution"]["walltime"])
    spec_origins = apply_spec(plan, spec)
    cif = spec["inputs"]["type"] == "cif"
    warnings = soc_warnings(spec)
    out: Dict[str, Any] = {}
    for i, t in enumerate(spec["sequence"]):
        key = f"{t}_{i + 1}"
        cfg = plan["step_configurations"][key]
        kind = "prop" if step_kind(t) == "d3" else "scf"
        res = dict(((cfg.get("slurm_config") or {}).get("resources")) or {})
        res.pop("mem_form", None)
        # the account the job script gets: no key keeps the template's own
        # -A line (the built-in account); None means no -A line at all
        if "account" not in res:
            from mace.submission.site import DEFAULTS as _site_defaults
            res["account"] = _site_defaults["account"]
        so = spec_origins.get(key, {})
        origins = {}
        for k, v in res.items():
            if k in so:
                origins[k] = "spec"
            elif k == "walltime" and spec["execution"]["walltime"]:
                origins[k] = "spec (execution.walltime)"
            else:
                origins[k] = _site_origin(site, k, kind, t, v) if site is not None else "builtin"
        if "settings" in so:
            settings_origin = so["settings"]
        elif i == 0:
            settings_origin = "builtin (cif2d12 defaults)" if cif else "deck"
        elif step_kind(t) == "d3":
            settings_origin = "builtin (opt2d3 defaults)"
        else:
            settings_origin = "parent"
        settings = copy.deepcopy(plan["cif_conversion_config"]) if (i == 0 and cif) else \
            {k: v for k, v in cfg.items() if k != "slurm_config"}
        out[key] = {"calc_type": t, "settings": settings, "settings_origin": settings_origin,
                    "resources": res, "origins": origins, "warnings": warnings.get(key, [])}
    return out


# ------------------------------------------------------------------ build
def build_plan(spec: Dict[str, Any], work_dir: Union[str, Path], db_path: str = "materials.db"
               ) -> Tuple[Path, str, Dict[str, Any]]:
    """Write the spec's plan in ``work_dir`` (and, for decks, its job
    scripts, as ``mace submit --progress`` does). Returns (plan file,
    workflow id, plan)."""
    work_dir = Path(work_dir).resolve()
    ex = spec["execution"]
    files = [Path(f) for f in spec["inputs"]["files"]]
    meta = {"label": spec.get("label"), "spec_version": spec["version"],
            "template": spec.get("template"), "source": spec.get("_source")}

    for msgs in soc_warnings(spec).values():
        for m in msgs:
            print(f"Warning: {m}")

    def finish(plan: Dict[str, Any], workflow_id: str) -> None:
        apply_spec(plan, spec, work_dir, workflow_id)
        if not ex["auto_progression"]:
            plan.setdefault("execution_settings", {})["auto_progression"] = False

    if spec["inputs"]["type"] == "d12":
        from mace.workflow.manual_progress import build_progress_plan
        holder: Dict[str, Any] = {}

        def customize(plan, workflow_id):
            finish(plan, workflow_id)
            plan["spec"] = meta
            if spec.get("label"):
                plan["label"] = spec["label"]
            holder["id"] = workflow_id

        plan_file, workflow_id, _ = build_progress_plan(
            work_dir=work_dir, db_path=db_path, input_files=files,
            progress=spec.get("template") or "spec", max_jobs=ex["max_jobs"],
            walltime=ex["walltime"], job_arrays=ex["job_arrays"],
            sequence=list(spec["sequence"]), customize=customize)
    else:
        from mace.workflow.context import new_workflow_id
        from mace.workflow.manual_progress import _apply_walltime
        from mace.workflow.planner import WorkflowPlanner
        planner = WorkflowPlanner(str(work_dir), db_path)
        plan = _base_plan(spec, planner.work_dir)
        if ex["walltime"]:
            _apply_walltime(plan["step_configurations"], ex["walltime"])
        workflow_id = new_workflow_id(planner.work_dir)
        finish(plan, workflow_id)
        plan["workflow_id"] = workflow_id
        plan["spec"] = meta
        if spec.get("label"):
            plan["label"] = spec["label"]
        plan_file = planner.save_workflow_plan(plan)
    # the spec as launched, next to its settings copies
    d = spec_dir(work_dir, workflow_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps({k: v for k, v in spec.items() if not k.startswith("_")},
                                            indent=2) + "\n")
    plan = json.loads(Path(plan_file).read_text())
    return Path(plan_file), workflow_id, plan
