"""The workflow builder's server side (GUI_WORKFLOW_DESIGN.md sections 3-6,
Phase 4): a workflow made in the GUI is a **workflow spec** the CLI turns
into its plan, and it is launched by the CLI. The GUI never writes a plan,
a job script or a deck.

* **Spec.** The builder's draft (structures, steps, per-step settings and
  resources) becomes ``spec.json`` plus one settings file per step whose
  Calculation Setup form was changed, in
  ``<folder>/workflow_configs/gui_specs/<label>_<time>/``. ``<folder>`` is
  the folder of the structures: the CLI runs there.
* **CLI only.** ``mace workflow --spec SPEC --json`` (check only: each
  step's resources and settings with where they come from),
  ``mace workflow --spec SPEC --write-plan --json`` (the plan; it prints the
  launch command), then that command: ``mace workflow --execute PLAN``
  (CIFs) or ``mace submit --progress-plan PLAN <decks>`` (decks).
* **Preview == run.** The preview runs exactly those commands in a staged
  copy of the folder with ``slurm_preview_shim`` first on PATH (read-only
  SLURM queries reach the cluster, state changes are recorded), as Submit's
  preview does (submit.py). It shows the plan, the job-script templates,
  each structure's first deck and job script, and the later steps of one
  structure as the engine writes them from stand-in parents (projection.py:
  labeled "projected"). The run writes the same spec at the same place in
  the real folder and runs the same commands; its job scripts are compared
  with the preview's.
* **Once.** A run needs a fresh, unused preview id (HMAC, 15 minutes)
  whose inputs have not changed since; only one GUI launch at a time per
  folder (an O_EXCL lock file); structures with unfinished steps in this
  folder's workflows are refused; a network-bound server refuses the run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import shlex
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import clirun
from . import submit as sub
from . import slurm_preview_shim as shim

SPEC_DIR = "workflow_configs/gui_specs"
UI_DRAFT_NAME = "builder_draft.json"
LOCK_NAME = ".gui_launch.lock"
LOCK_STALE_S = 1800.0
MIRROR_MARKER = ".mace_preview_mirror"
MAX_INPUTS = 2000
D3_TYPES = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS")
METHOD_FIELDS = ("functional", "dispersion")
RESOURCE_KEYS = ("ntasks", "walltime", "memory", "memory_per_cpu", "account", "constraint", "partition", "exclude")
NO_TEMPLATES = ("custom", "interactive")
LABEL_RE = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")
STEP_KEY_RE = re.compile(r"^([A-Z+]+[0-9]*)_(\d+)$")
NOT_WRITING = re.compile(r"Not writing \S+?\.d12: (.+)$", re.M)

# Plain words for the builder (a student who never used the CLI reads these).
CALC_HELP = {
    "OPT": "Geometry optimization: moves the atoms (and the cell) to the lowest-energy structure.",
    "SP": "Single point: one SCF calculation on the optimized structure; it writes the wavefunction "
          "the property steps read.",
    "FREQ": "Vibrational frequencies at Γ (phonons; IR and Raman intensities when asked for).",
    "BAND": "Band structure along the high-symmetry path of the Brillouin zone.",
    "DOSS": "Density of states (total and projected).",
    "TRANSPORT": "Transport properties (conductivity, Seebeck coefficient) from the band structure.",
    "CHARGE+POTENTIAL": "Charge density and electrostatic potential maps.",
    "ORBITALS": "Every orbital as Molden files, at Γ and the band-edge k points (one file for a molecule), "
                "to draw single orbitals.",
}


class BuildError(sub.SubmitError):
    """A draft the builder must correct (HTTP 400)."""


# ------------------------------------------------------------------ templates
def calc_kind(calc_type: str) -> str:
    return "d3" if base_type(calc_type) in D3_TYPES else "d12"


def base_type(calc_type: str) -> str:
    return calc_type.rstrip("0123456789") or calc_type


def templates_info() -> Dict[str, Any]:
    from mace.workflow.common.constants import (CALC_TYPES, WORKFLOW_TEMPLATE_DESCRIPTIONS,
                                                WORKFLOW_TEMPLATES)
    temps = [{"name": n, "sequence": list(seq), "help": WORKFLOW_TEMPLATE_DESCRIPTIONS.get(n, " → ".join(seq))}
             for n, seq in WORKFLOW_TEMPLATES.items() if n not in NO_TEMPLATES]
    types = {t: {"name": v.get("name", t), "help": CALC_HELP.get(t, v.get("name", t)),
                 "depends_on": list(v.get("depends_on") or []), "kind": calc_kind(t),
                 "repeatable": t in ("OPT", "SP")}
             for t, v in CALC_TYPES.items()}
    return {"templates": temps, "calc_types": types, "first_types": {"cif": ["OPT", "SP"]}}


def parent_key(sequence: List[str], i: int) -> Optional[str]:
    from mace.workflow.status import parent_index
    j = parent_index(list(sequence), i)
    return None if j is None else f"{sequence[j]}_{j + 1}"


# ------------------------------------------------------------------ inputs
@dataclass
class Inputs:
    kind: str                     # cif | d12
    folder: Path                  # where the CLI runs (the structures' folder)
    files: List[Path]
    materials: List[str]


def _material_id(p: Path) -> str:
    try:
        from mace.database.materials import create_material_id_from_file
        return create_material_id_from_file(str(p))
    except Exception:
        return p.stem


def scan_inputs(st, targets: Any, kind: Optional[str] = None) -> Inputs:
    """The structures a workflow starts from: CIFs or decks (.d12) of one
    folder, given as files or as the folder."""
    if not isinstance(targets, list) or not targets or len(targets) > MAX_INPUTS:
        raise BuildError("Pick CIF or .d12 files, or one folder of them")
    paths = [st.resolve(str(t)) for t in targets]
    dirs = [p for p in paths if p.is_dir()]
    if dirs:
        if len(paths) > 1:
            raise BuildError("Pick one folder, or files of one folder")
        d = dirs[0]
        # a symlink that leads out of the served folder is skipped, as the
        # Explorer skips it (never read, never named); one inside works
        entries = []
        for x in d.iterdir():
            try:
                r = x.resolve()
            except OSError:
                continue
            if (r == st.root or st.root in r.parents) and r.is_file():
                entries.append(x)
        cifs = sorted(x for x in entries if x.suffix.lower() == ".cif")
        d12s = sorted(x for x in entries if x.suffix.lower() == ".d12")
        if kind == "cif" or (kind is None and cifs and not d12s):
            files = cifs
        elif kind == "d12" or (kind is None and d12s and not cifs):
            files = d12s
        elif cifs and d12s:
            raise BuildError(f"{st.rel(d) or 'This folder'} has CIF and .d12 files: pick the files to start from")
        else:
            raise BuildError(f"No CIF or .d12 files in {st.rel(d) or 'this folder'}")
        if not files:
            raise BuildError(f"No .{kind} files in {st.rel(d) or 'this folder'}")
    else:
        files = sorted(paths)
        for f in files:
            if not f.is_file():
                raise BuildError(f"{st.rel(f)} is not a file")
    kinds = {f.suffix.lower() for f in files}
    if kinds - {".cif", ".d12"}:
        bad = [f.name for f in files if f.suffix.lower() not in (".cif", ".d12")]
        raise BuildError(f"A workflow starts from CIF or .d12 files, not {', '.join(bad[:3])}")
    if len(kinds) > 1:
        raise BuildError("Start from CIF files or from .d12 decks, not both")
    if len({f.parent for f in files}) > 1:
        raise BuildError("Pick structures of one folder (the workflow runs in that folder)")
    for f in files:
        if not sub.path_ok(f):
            raise BuildError(f"File names with spaces or shell characters cannot run in a job; rename {f.name} "
                             f"(or its folder)")
    mats = [_material_id(f) for f in files]
    dup = sorted({m for m in mats if mats.count(m) > 1})
    if dup:
        raise BuildError(f"These files give the same material name: {', '.join(dup)} (rename one)")
    k = "cif" if ".cif" in kinds else "d12"
    if kind and kind != k:
        raise BuildError(f"The draft starts from {kind.upper()} files, these are {k.upper()}")
    return Inputs(k, files[0].parent, files, mats)


def inputs_view(st, inp: Inputs) -> Dict[str, Any]:
    out = {"type": inp.kind, "folder_rel": st.rel(inp.folder) if inp.folder != st.root else "",
           "files": [st.rel(f) for f in inp.files],
           "materials": [{"file": st.rel(f), "material_id": m} for f, m in zip(inp.files, inp.materials)]}
    if inp.kind == "d12":
        out["deck_types"] = sorted({sub.deck_calc_type(f) for f in inp.files})
        out["refused"] = input_refusals(inp)
    return out


def input_refusals(inp: Inputs) -> Dict[str, str]:
    """{step type: why} for the steps these inputs cannot have, which the
    builder grays out: ORBITALS when a deck is two-component (TWOCOMPON), on
    which CRYSTAL23 crashes (the spec check refuses it too)."""
    if inp.kind != "d12":
        return {}
    from mace.utils.calc_detection import is_two_component_run
    two_c = [f.name for f in inp.files if is_two_component_run(f)]
    if not two_c:
        return {}
    from mace.orbitals.kchoice import SOC_REFUSAL
    more = f" and {len(two_c) - 3} more" if len(two_c) > 3 else ""
    return {"ORBITALS": f"{', '.join(two_c[:3])}{more}: {SOC_REFUSAL}"}


# ------------------------------------------------------------------ the spec
_SCHEMA: Dict[str, Any] = {}


def setup_schema() -> Dict[str, Any]:
    """Calculation Setup's schema (the functional menus), read once."""
    if not _SCHEMA:
        from . import setup as gui_setup
        try:
            _SCHEMA.update(gui_setup.schema())
        except Exception:          # pragma: no cover - the form still works without the menus
            pass
    return _SCHEMA


def _schema_functionals() -> set:
    cats = setup_schema().get("functional_categories") or []
    if not cats:
        return set()
    return {str(f).upper() for c in cats for f in (c.get("functionals") or [])}


def draft_sequence(draft: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    from mace.workflow.common.constants import WORKFLOW_TEMPLATES
    seq = draft.get("sequence")
    if not isinstance(seq, list) or not seq or not all(isinstance(s, str) for s in seq) or len(seq) > 30:
        raise BuildError("Add at least one step")
    from mace.workflow.common.constants import CALC_TYPES
    bad = [s for s in seq if base_type(s) not in CALC_TYPES]
    if bad:
        raise BuildError(f"not a workflow step type: {', '.join(bad[:3])} (the types are {', '.join(CALC_TYPES)})")
    tmpl = draft.get("template")
    if tmpl and (tmpl not in WORKFLOW_TEMPLATES or list(WORKFLOW_TEMPLATES[tmpl]) != list(seq)):
        tmpl = None                   # edited after a template was picked: a custom sequence
    return list(seq), tmpl


# ------------------------------------------------------------------ spin-orbit coupling
SOC_KEYS = ("soc", "soc_ecp", "soc_basis", "soc_guess", "smear", "soc_fmixing")
SOC_NO_RESTART = ("The builder starts a SOC single point from scratch: a restart (GUESSPSO, GUESSPNOSO) needs a "
                  "parent run known to be two-component or spin-polarized without symmetry, which the step before "
                  "is not known to be until it has run. Set soc_guess in the workflow spec by hand "
                  "(mace workflow --spec) when it is.")


def soc_patches(s: Dict[str, Any]) -> Dict[str, Any]:
    """A draft step's SOC choices (``steps[key].soc``) as the CLI's settings-file
    keys: nothing when SOC is off, empty values left out."""
    soc = s.get("soc") if isinstance(s, dict) else None
    if not isinstance(soc, dict) or soc.get("soc") is not True:
        return {}
    return {k: soc[k] for k in SOC_KEYS if soc.get(k) not in (None, "")}


def check_soc(key: str, calc_type: str, patches: Dict[str, Any], deck_start: bool = False) -> None:
    """SOC on this step: an SP step only (the CLI's reason otherwise), values
    checked by the CLI's own checks, from scratch only."""
    keys = [k for k in SOC_KEYS if k in patches]
    if not keys:
        return
    if deck_start:
        raise BuildError(f"{key}: the first step runs the decks as they are; make a SOC deck in Calculation Setup")
    from mace.workflow.spec import soc_refusal
    why = soc_refusal(calc_type)
    if why:
        raise BuildError(f"{key}: no spin-orbit coupling here. {why}")
    from . import setup as gui_setup
    for k in keys:
        if k == "soc_fmixing":
            # the CLI's own check (mace workflow --spec reads it the same way)
            from mace.workflow.spec import SpecError, _check_soc
            try:
                _check_soc(key, calc_type, {"soc": True, k: patches[k]}, False)
            except SpecError as e:
                raise BuildError(str(e))
            continue
        try:
            gui_setup.check_soc_value(k, patches[k])
        except gui_setup.SetupError as e:
            raise BuildError(f"{key}: {e}")
    if patches.get("soc_guess") is not None:
        raise BuildError(f"{key}: {SOC_NO_RESTART}")


def soc_group(inp: Inputs, seq: List[str], i: int) -> Dict[str, Any]:
    """What the step form's Spin-orbit coupling group shows: whether this step
    takes SOC, else why not (the CLI's words), and the CLI's own tables."""
    from mace.workflow.spec import soc_refusal
    why = soc_refusal(seq[i])
    if why is None and i == 0 and inp.kind == "d12":
        why = ("The first step runs your decks as they are. Make a SOC deck in Calculation Setup "
               "(or mace opt2d12 / mace convert with \"soc\": true) and start from it.")
    info = setup_schema().get("soc") or {}
    stutsc = None
    if why is None:
        from . import setup as gui_setup
        stutsc = gui_setup.stutsc_refusals(inp.files)      # STUTSC grayed out as in Setup (soc_ecp.py's table)
    return {"applies": why is None, "reason": why, "restart_reason": SOC_NO_RESTART, "stutsc": stutsc,
            "metal_smear": info.get("metal_smear"), "ecp_forms": info.get("ecp_forms") or ["inpsoc", "stutsc"],
            "soc_basis_families": info.get("soc_basis_families") or ["pob-tzvp-rev2-soc"],
            "guesses": info.get("guesses") or ["guesspso", "guesspnoso"]}


def spec_files(inp: Inputs, draft: Dict[str, Any], spec_dir: Optional[Path] = None,
               step1_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """``spec.json`` and the step settings files for this draft. Input paths
    are relative to the spec's folder (``<folder>/workflow_configs/gui_specs/<x>/``),
    or absolute when ``spec_dir`` is None (a check in the GUI's scratch).

    A CIF start's first step gets ``step1_overrides`` (cif_overrides: the
    keys the CLI changed for the form's answers) merged into the workflow's
    own conversion settings; a later SCF step its method; a properties step
    the opt2d3 settings file its form made (``settings_text``)."""
    if not isinstance(draft, dict):
        raise BuildError("draft is an object")
    seq, tmpl = draft_sequence(draft)
    if spec_dir is None:
        files = [str(f) for f in inp.files]
    else:
        files = [os.path.relpath(f, spec_dir) for f in inp.files]
    spec: Dict[str, Any] = {"version": 1, "inputs": {"type": inp.kind, "files": files}}
    label = str(draft.get("label") or "").strip()
    if label:
        if not LABEL_RE.match(label):
            raise BuildError("Name: letters, digits and . _ + - only (no spaces), at most 64 characters")
        spec["label"] = label
    if tmpl:
        spec["template"] = tmpl
    else:
        spec["sequence"] = seq
    out: Dict[str, str] = {}
    steps_in = draft.get("steps") or {}
    if not isinstance(steps_in, dict):
        raise BuildError("steps is an object")
    keys = [f"{t}_{i + 1}" for i, t in enumerate(seq)]
    unknown = [k for k in steps_in if k not in keys]
    if unknown:
        raise BuildError(f"Settings for steps that are not in the chain: {', '.join(unknown)}")
    steps: Dict[str, Any] = {}
    functionals = None
    for i, key in enumerate(keys):
        s = steps_in.get(key) or {}
        if not isinstance(s, dict):
            raise BuildError(f"{key}: settings are an object")
        entry: Dict[str, Any] = {}
        text = s.get("settings_text")
        if text not in (None, ""):
            if not isinstance(text, str) or len(text) > 1_000_000:
                raise BuildError(f"{key}: settings_text is the settings file's text")
            if i == 0 and inp.kind == "d12":
                raise BuildError(f"{key}: the first step runs the decks as they are; only its resources can change")
            if i == 0 or calc_kind(seq[i]) != "d3":
                raise BuildError(f"{key}: a settings file is set here only on a properties step "
                                 f"({', '.join(D3_TYPES)})")
            named = _d3_calc_types(text) - {base_type(seq[i])}
            if named:
                raise BuildError(f"{key}: the settings file is for a {sorted(named)[0]} step")
            from . import setup as gui_setup
            bad = gui_setup.validate("opt2d3", text)
            if bad:
                raise BuildError(f"{key}: " + "; ".join(bad))
            name = f"{key.replace('+', '_')}_settings.json"
            out[name] = text if text.endswith("\n") else text + "\n"
            entry["settings_file"] = name
        if i == 0 and inp.kind == "cif" and step1_overrides:
            entry["overrides"] = copy.deepcopy(step1_overrides)
        m = s.get("method")
        if m:
            if i == 0 or calc_kind(seq[i]) != "d12":
                raise BuildError(f"{key}: the method is changed only on a later SCF step (SP, OPT2, FREQ)")
            if not isinstance(m, dict) or set(m) - set(METHOD_FIELDS):
                raise BuildError(f"{key}: method is {{functional, dispersion}}")
            fn = str(m.get("functional") or "").strip()
            if not fn:
                raise BuildError(f"{key}: name the functional")
            if functionals is None:
                functionals = _schema_functionals()
            base = re.sub(r"-D3$", "", fn, flags=re.I)
            if functionals and base.upper() not in functionals:
                raise BuildError(f"{key}: {fn} is not a functional MACE knows")
            mods = {"new_functional": base}
            if m.get("dispersion") is not None:
                mods["use_dispersion"] = bool(m.get("dispersion"))
            elif re.search(r"-D3$", fn, flags=re.I):
                mods["use_dispersion"] = True
            # else: no word on dispersion, the parent's D3 is kept (engine + opt2d12)
            entry["overrides"] = {"method_modifications": mods}
        # spin-orbit coupling: an SP step's overrides (a CIF start's first step gets them through its
        # conversion settings: step1_overrides, from the form's patches)
        soc = soc_patches(s)
        check_soc(key, seq[i], soc, deck_start=(i == 0 and inp.kind == "d12"))
        if soc and not (i == 0 and inp.kind == "cif"):
            entry.setdefault("overrides", {}).update(soc)
        res = s.get("resources")
        if res:
            if not isinstance(res, dict):
                raise BuildError(f"{key}: resources are an object")
            bad = sorted(set(res) - set(RESOURCE_KEYS))
            if bad:
                raise BuildError(f"{key}: unknown resources {', '.join(bad)}")
            clean = {}
            for k, v in res.items():
                if v in (None, ""):
                    continue
                if k == "ntasks":
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        raise BuildError(f"{key}: cores is a whole number")
                else:
                    v = str(v).strip()
                clean[k] = v
            if clean:
                entry["resources"] = clean
        if entry:
            steps[key] = entry
    if steps:
        spec["steps"] = steps
    ex = draft.get("execution") or {}
    if not isinstance(ex, dict):
        raise BuildError("execution is an object")
    execution: Dict[str, Any] = {}
    if ex.get("max_jobs") not in (None, ""):
        try:
            execution["max_jobs"] = int(ex["max_jobs"])
        except (TypeError, ValueError):
            raise BuildError("At most this many jobs queued: a whole number")
    if ex.get("job_arrays"):
        ja = ex["job_arrays"] if isinstance(ex["job_arrays"], dict) else {}
        arr: Dict[str, Any] = {"first_step": True}
        if ja.get("max_running") not in (None, ""):
            try:
                arr["max_running"] = int(ja["max_running"])
            except (TypeError, ValueError):
                raise BuildError("At most this many array tasks at once: a whole number")
        execution["job_arrays"] = arr
    if ex.get("auto_progression") is False:
        execution["auto_progression"] = False
    if execution:
        spec["execution"] = execution
    out["spec.json"] = json.dumps(spec, indent=2) + "\n"
    return out


def spec_folder_name(draft: Dict[str, Any]) -> str:
    label = str(draft.get("label") or "").strip()
    stem = label if label and LABEL_RE.match(label) else "workflow"
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


def write_files(d: Path, files: Dict[str, str], exclusive: bool = False) -> None:
    if exclusive and d.exists():
        raise sub.SubmitConflict(f"{d.name} exists already; preview again")
    d.mkdir(parents=True, exist_ok=not exclusive)
    for name, text in files.items():
        (d / name).write_text(text)


# ------------------------------------------------------------------ CLI helpers
def parse_json(stdout: str) -> Optional[Any]:
    """The JSON document a ``--json`` command printed (anything before it,
    such as a banner, is skipped)."""
    s = stdout.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
    return None


def cli_env() -> Dict[str, str]:
    return sub.submit_env()


def check_spec(root: Path, inp: Inputs, draft: Dict[str, Any], work: Path, ctx=None,
               step1_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``mace workflow --spec SPEC --json`` (nothing written) in the GUI's
    scratch: each step's settings and resources and where they come from,
    with the partition's time limit checked."""
    files = spec_files(inp, draft, None, step1_overrides)
    d = work / "check"
    write_files(d, files)
    wd = work / "wd"
    wd.mkdir(parents=True, exist_ok=True)
    argv = ["workflow", "--spec", str(d / "spec.json"), "--work-dir", str(wd), "--json"]
    r = clirun.run(clirun.mace_argv("--no-banner", *argv), work, ctx, timeout=300, env=cli_env())
    res = parse_json(r.stdout)
    if not isinstance(res, dict):
        raise BuildError("mace workflow --spec gave no answer: " + (r.stderr or r.stdout).strip()[-600:])
    if not res.get("ok"):
        raise BuildError(res.get("error") or "the CLI refused the spec")
    seq, _ = draft_sequence(draft)
    env = cli_env()
    steps = res.get("steps") or {}
    for i, key in enumerate(list(steps)):
        s = steps[key]
        s["parent"] = parent_key(seq, i)
        s["kind"] = calc_kind(seq[i])
        s["help"] = CALC_HELP.get(base_type(seq[i]), "")
        s["checks"] = resource_checks(s.get("resources") or {}, env)
    res["argv"] = ["mace", "workflow", "--spec", "SPEC", "--json"]
    res["steps"] = steps
    return res


def resource_checks(res: Dict[str, Any], env: Dict[str, str]) -> List[Dict[str, str]]:
    """What SLURM would refuse: a walltime over the partition's MaxTime."""
    from mace.submission import site
    out: List[Dict[str, str]] = []
    wt = res.get("walltime")
    if wt:
        secs = site.parse_walltime(str(wt))
        try:
            limit, part = site.partition_limit(res.get("partition") or None, env)
        except Exception:
            limit, part = None, None
        if secs and limit and secs > limit:
            out.append({"field": "walltime", "severity": "error",
                        "message": f"{wt} is longer than the {part or 'default'} partition allows "
                                   f"({_hms(limit)}): SLURM would refuse the job"})
    return out


def _hms(s: int) -> str:
    d, rem = divmod(int(s), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return (f"{d}-" if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"


# ------------------------------------------------------------------ step forms
def step_request(draft: Dict[str, Any], step: str) -> Tuple[List[str], int]:
    seq, _ = draft_sequence(draft)
    m = STEP_KEY_RE.match(str(step or ""))
    if not m or not (1 <= int(m.group(2)) <= len(seq)) or seq[int(m.group(2)) - 1] != m.group(1):
        raise BuildError(f"{step} is not a step of this chain")
    return seq, int(m.group(2)) - 1


def check_form_answers(inp: Inputs, seq: List[str], i: int, body: Dict[str, Any]) -> None:
    """A later SCF step inherits its parent's settings: only the method changes there."""
    if i == 0 and inp.kind == "d12":
        return
    check_soc(f"{seq[i]}_{i + 1}", seq[i], {k: v for k, v in (body.get("patches") or {}).items() if k in SOC_KEYS})
    if i > 0 and calc_kind(seq[i]) == "d12":
        answers = body.get("answers") or {}
        bad = sorted(set(answers) - set(METHOD_FIELDS)) + (["the settings file"] if body.get("json_text") else []) \
            + sorted(k for k in (body.get("patches") or {}) if k not in SOC_KEYS)
        if bad:
            raise BuildError(f"{seq[i]} inherits its parent's settings for each structure; only the functional "
                             f"and dispersion can change here (not {', '.join(bad)})")


class Builder:
    """What a step form or a preview works with: the served folder's state,
    the structures, the draft, a scratch folder, the task and Setup's cache."""

    def __init__(self, st, inp: Inputs, draft: Dict[str, Any], work: Path, ctx, cache_get, cache_put):
        self.st, self.inp, self.draft, self.work, self.ctx = st, inp, draft, work, ctx
        self.get, self.put = cache_get, cache_put
        self.seq, _ = draft_sequence(draft)

    def setup(self, req_body: Dict[str, Any], mode: str, calc: str, tag: str, source: Path, rel: str, key: Any,
              schema: bool = True) -> Dict[str, Any]:
        from . import setup as gui_setup
        return gui_setup.run_setup(gui_setup.Request(req_body), mode, calc, self.work / tag, source, rel, key,
                                   self.get, self.put, self.ctx, schema_info=setup_schema() if schema else None)

    # ---- a CIF start's first step: the workflow's own conversion settings
    def cif_base(self) -> Dict[str, Any]:
        """The conversion settings a workflow gives its CIFs when nothing is
        changed (the plan's cif_conversion_config), from the CLI itself."""
        key = ("wf-cifbase", self.seq[0], clirun.cli_hash())
        hit = self.get(key)
        if hit is None:
            first = Inputs(self.inp.kind, self.inp.folder, self.inp.files[:1], self.inp.materials[:1])
            res = check_spec(self.st.root, first, {"sequence": [self.seq[0]], "steps": {}}, self.work / "base",
                             self.ctx)
            hit = (res.get("steps") or {}).get(f"{self.seq[0]}_1", {}).get("settings") or {}
            self.put(key, hit, len(json.dumps(hit)))
        return copy.deepcopy(hit)

    def cif_overrides(self, answers: Dict[str, Any], patches: Dict[str, Any]) -> Dict[str, Any]:
        """The settings the form's answers change, as the CLI works them out:
        ``mace convert`` asked with and without the answers (Setup's own
        runs); the keys that differ, and each answered field's own keys."""
        from . import setup as gui_setup
        if not answers and not patches:
            return {}
        first = self.inp.files[0]
        rel, src = self.st.rel(first), gui_setup.source_key("cif2d12", first)
        calc = base_type(self.seq[0])
        d0 = self.setup({"answers": {}, "form": False, "preview": False}, "cif2d12", calc, "ov0", first, rel, src)
        d1 = self.setup({"answers": answers, "patches": patches, "form": False, "preview": False}, "cif2d12", calc,
                        "ov1", first, rel, src)
        if not d1.get("json_text"):
            raise BuildError("mace convert wrote no settings for these answers: "
                             + ((d1.get("refused") or "").strip().splitlines() or ["?"])[-1])
        a, d = json.loads(d1["json_text"]), json.loads(d0.get("json_text") or "{}")
        keys = {k for k in set(a) | set(d) if a.get(k) != d.get(k)}
        for fid in answers:
            f = gui_setup.FIELD.get(fid)
            if f is not None:
                keys |= {k.split(".")[0] for k in f.keys}
        keys -= {"calculation_type", "dimensionality"}
        return {k: a[k] for k in sorted(keys) if k in a}

    def step1_overrides(self) -> Optional[Dict[str, Any]]:
        if self.inp.kind != "cif":
            return None
        s = (self.draft.get("steps") or {}).get(f"{self.seq[0]}_1") or {}
        return self.cif_overrides(s.get("answers") or {}, {**(s.get("patches") or {}), **soc_patches(s)}) or None

    def cif_effective(self) -> Dict[str, Any]:
        from mace.workflow.spec import _merge
        cfg = self.cif_base()
        _merge(cfg, self.step1_overrides() or {})
        return cfg

    # ---- decks
    def parent_deck(self, i: int) -> str:
        """The deck step ``i`` writes for the first structure: a CIF start's
        conversion with the workflow's settings, a deck start's deck, a later
        SCF step from a stand-in of its own parent (with its method)."""
        seq = self.seq
        key = f"{seq[i]}_{i + 1}"
        s = (self.draft.get("steps") or {}).get(key) or {}
        first = self.inp.files[0]
        if i == 0:
            if self.inp.kind == "d12":
                return first.read_text(errors="replace")
            from . import setup as gui_setup
            res = self.setup({"json_text": json.dumps(self.cif_effective(), indent=2), "form": False, "preview": True},
                             "cif2d12", base_type(seq[0]), "p0", first, self.st.rel(first),
                             gui_setup.source_key("cif2d12", first), schema=False)
        else:
            pkey = parent_key(seq, i)
            pdeck = self.parent_deck(int(pkey.rsplit("_", 1)[1]) - 1)
            out_path = stand_in(self.work / f"pstandin_{i}", self.inp.materials[0], pdeck)
            m = s.get("method") or {}
            answers = {}
            if m.get("functional"):
                answers["functional"] = re.sub(r"-D3$", "", str(m["functional"]), flags=re.I)
                if m.get("dispersion") is not None:
                    answers["dispersion"] = "Yes" if m.get("dispersion") else "No"
            res = self.setup({"answers": answers, "patches": soc_patches(s), "form": False, "preview": True},
                             "opt2d12", base_type(seq[i]),
                             f"p{i}", out_path, f"projected/{out_path.name}",
                             ("wf-standin", hashlib.sha256(pdeck.encode()).hexdigest(), "opt2d12"), schema=False)
        if not res.get("deck"):
            raise BuildError(f"No {key} deck to build the next step's form from: " + refusal_line(res))
        return res["deck"]


def refusal_line(res: Dict[str, Any]) -> str:
    """The CLI's reason for writing no deck, in one line: its "Not writing <deck>: <reason>" (a SOC
    refusal), else the last line that says something (never the bare "[ERROR]" tag it ends with)."""
    from . import setup as gui_setup
    soc = (res.get("soc") or {}).get("refusal") or gui_setup.soc_refusal(res.get("log") or "")
    if soc:
        return soc
    words = [ln.strip() for ln in (res.get("refused") or "").splitlines()
             if re.sub(r"^\W*(\[?(ERROR|ERR|X)\]?)\W*", "", ln.strip())]
    return words[-1] if words else "the CLI wrote none"


# fields of a CIF start's first step that are not changed here (the structure's own; the chain's)
STEP1_FIXED = ("calc_type", "dimensionality")


def step_form(st, inp: Inputs, draft: Dict[str, Any], seq: List[str], i: int, body: Dict[str, Any],
              work: Path, ctx, cache_get, cache_put) -> Dict[str, Any]:
    """The Calculation Setup form of one step (gui/setup.py's run_setup,
    the CLI's own questions):

    * a CIF start's first step: the form of the workflow's own conversion
      settings (not ``mace convert``'s defaults: a workflow converts with
      the plan's cif_conversion_config), with the answers' changes merged in;
      ``base_values`` are the values with nothing changed;
    * a later SCF step: opt2d12 on a stand-in parent made from its parent
      step's deck; only the method can change (each structure keeps its own
      parent's settings);
    * a properties step: opt2d3 on a stand-in parent, every field.
    """
    from . import setup as gui_setup
    b = Builder(st, inp, draft, work, ctx, cache_get, cache_put)
    key = f"{seq[i]}_{i + 1}"
    calc = base_type(seq[i])
    first = inp.files[0]
    preview_deck = body.get("preview", True) is not False
    if i == 0 and inp.kind == "d12":
        return {"mode": "deck", "step": key, "calc_type": seq[0], "readonly": True, "form": [],
                "deck": first.read_text(errors="replace"), "deck_name": first.name, "parent": None,
                "note": "The first step runs your decks as they are.", "soc_group": soc_group(inp, seq, i)}
    if i == 0:
        answers, patches = body.get("answers") or {}, body.get("patches") or {}
        bad = [a for a in answers if a in STEP1_FIXED]
        if bad:
            raise BuildError(f"{', '.join(bad)} is set by the chain and the structures, not here")
        src = gui_setup.source_key("cif2d12", first)
        base = b.cif_base()
        ov = b.cif_overrides(answers, patches)
        from mace.workflow.spec import _merge
        eff = copy.deepcopy(base)
        _merge(eff, ov)
        res = b.setup({"json_text": json.dumps(eff, indent=2), "preview": preview_deck}, "cif2d12", calc, "s0",
                      first, st.rel(first), src)
        base_res = b.setup({"json_text": json.dumps(base, indent=2), "preview": False}, "cif2d12", calc, "s0b",
                           first, st.rel(first), src)
        res.update(step=key, parent=None, material_id=inp.materials[0], overrides=ov,
                   base_values={f["id"]: f.get("value") for f in base_res.get("form") or []},
                   editable=[f["id"] for f in res.get("form") or [] if f["id"] not in STEP1_FIXED
                             and not str(f["id"]).startswith("q:") and f.get("how") != "patch"],
                   soc_group=soc_group(inp, seq, i))
        return res
    pkey = parent_key(seq, i)
    pdeck = b.parent_deck(int(pkey.rsplit("_", 1)[1]) - 1)
    out_path = stand_in(work / f"standin_{i}", inp.materials[0], pdeck)
    mode = "opt2d3" if calc_kind(seq[i]) == "d3" else "opt2d12"
    res = b.setup({"answers": body.get("answers") or {}, "patches": body.get("patches") or {},
                   "json_text": body.get("json_text"), "preview": preview_deck}, mode, calc, f"s{i}", out_path,
                  f"projected/{out_path.name}", ("wf-standin", hashlib.sha256(pdeck.encode()).hexdigest(), mode))
    res.update(step=key, parent=pkey, material_id=inp.materials[0], standin=True,
               editable=list(METHOD_FIELDS) if mode == "opt2d12" else None, soc_group=soc_group(inp, seq, i))
    return res


def _d3_calc_types(text: str) -> set:
    """The calculation types an opt2d3 settings file names (its envelope, its configuration)."""
    try:
        data = json.loads(text)
    except ValueError:
        return set()
    if not isinstance(data, dict):
        return set()
    conf = data.get("configuration") if isinstance(data.get("configuration"), dict) else {}
    return {t for t in (data.get("calculation_type"), conf.get("calculation_type")) if isinstance(t, str)}


def stand_in(d: Path, material: str, deck_text: str) -> Path:
    """A stand-in parent: the deck, an output DummyFileCreator makes from it
    and an empty-ish wavefunction (as a finished job leaves them)."""
    from mace.workflow.dummy_file_creator import DummyFileCreator
    d.mkdir(parents=True, exist_ok=True)
    deck = d / f"{material}.d12"
    deck.write_text(deck_text)
    out = d / f"{material}.out"
    c = DummyFileCreator()
    c.create_dummy_out(out, c.extract_d12_settings(deck))
    c.create_dummy_f9(d / f"{material}.f9")
    return out


# ------------------------------------------------------------------ fingerprint
def active_rows(folder: Path, inp: Inputs) -> List[Dict[str, Any]]:
    """Unfinished steps (submitted, queued, running; pending with a job id)
    of these structures in this folder's databases."""
    import sqlite3
    mats = set(inp.materials)
    decks = {str(f) for f in inp.files} | {str(f.resolve()) for f in inp.files}
    out = []
    for db in sub.context_dbs([folder]):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        try:
            rows = con.execute("SELECT calc_id, material_id, calc_type, status, slurm_job_id, input_file "
                               "FROM calculations WHERE lower(status) IN (?, ?, ?, ?)", sub.QUEUED_STATES).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
        for calc_id, mid, ctype, status, job, inpf in rows:
            if not job and str(status or "").lower() == "pending":
                continue
            if mid in mats or (inpf and inpf in decks):
                out.append({"calc_id": calc_id, "material_id": mid, "calc_type": ctype, "status": status,
                            "job_id": job, "db": str(db)})
    return out


def fingerprint(root_gen: Any, inp: Inputs, files: Dict[str, str], env: Dict[str, str]) -> Dict[str, Any]:
    stamps = [sub._stamp(f) for f in inp.files]
    if inp.kind == "d12":
        for f in inp.files:
            sh = f.with_suffix(".sh")
            stamps.append(sub._stamp(sh) if sh.exists() else [str(sh), "absent"])
            out = f.with_suffix(".out")
            stamps.append([str(out), sub.out_bucket(out)] if out.exists() else [str(out), "absent"])
    blocked = sorted(json.dumps(r, sort_keys=True) for r in active_rows(inp.folder, inp))
    return {"root_gen": root_gen, "spec": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())},
            "inputs": stamps, "blocked": blocked, "site": sub.site_inputs(env), "module": sub.site_resolution(env)}


# ------------------------------------------------------------------ preview
@dataclass
class WfPreview:
    id: str
    created: float
    folder: Path
    inputs: Inputs
    spec_rel: str                       # the spec's folder, relative to the workflow folder
    files: Dict[str, str]
    fp: Dict[str, Any]
    result: Dict[str, Any]
    launch_shape: List[str] = field(default_factory=list)
    scripts: Dict[str, str] = field(default_factory=dict)   # root-relative path -> normalized text
    needs: Dict[str, bool] = field(default_factory=dict)
    used: bool = False


CKSUM_RE = re.compile(r"(# MACE-generated; cksum of the lines above: )\d+ \d+")


def normalize_script(text: str, mirror: Optional[Path] = None, root: Optional[Path] = None) -> str:
    """A job script with this run's ids, times and paths normalized (submit.normalize),
    and the trailing checksum of the lines above (which hold the workflow id) too."""
    return sub.normalize(CKSUM_RE.sub(r"\1<CKSUM>", text), mirror, root)


def _unmirror(text: str, mirror: Path, root: Path) -> str:
    return str(text).replace(str(mirror), str(root))


def _shape(argv: List[str]) -> List[str]:
    """A launch command with its plan's id and paths normalized."""
    return [sub.normalize(a) for a in argv]


def _display(argv: List[str], folder: Path) -> str:
    """The command as a user types it in ``folder``."""
    parts = []
    for a in argv:
        try:
            p = Path(a)
            if p.is_absolute():
                a = os.path.relpath(a, folder)
        except (ValueError, OSError):
            pass
        parts.append(shlex.quote(a))
    return " ".join(parts)


def preview(st, root: Path, root_gen: Any, inp: Inputs, draft: Dict[str, Any], work: Path, ctx=None,
            network: bool = False, cache_get=None, cache_put=None,
            ui_draft: Optional[str] = None) -> Tuple[WfPreview, Dict[str, Any]]:
    env = cli_env()
    folder = inp.folder
    folder_rel = sub._rel(root, folder)
    spec_name = spec_folder_name(draft)
    spec_rel = f"{SPEC_DIR}/{spec_name}"
    ov = Builder(st, inp, draft, work / "forms", ctx, cache_get or (lambda k: None),
                 cache_put or (lambda k, v, n: None)).step1_overrides()
    files = spec_files(inp, draft, folder / spec_rel, ov)
    if ui_draft:                      # the builder's own draft, for Clone (the CLI never reads it)
        files[UI_DRAFT_NAME] = ui_draft
    fp = fingerprint(root_gen, inp, files, env)
    if "error" in fp["module"]:
        raise BuildError(fp["module"]["error"])
    warnings: List[str] = []
    if network:
        warnings.append("Submitting is turned off while MACE GUI is open to the network")
    blocked_rows = active_rows(folder, inp)
    blocked = sorted({r["material_id"] for r in blocked_rows})
    # the staged copy: the structures, the folder's workflow state and databases
    mirror = work / "mirror"
    mirror.mkdir(parents=True, exist_ok=True)
    sel = sub.Selection(folder, ["."], list(inp.files), True)
    staged = sub.stage(sel, root, mirror)
    (mirror / MIRROR_MARKER).write_text("a MACE GUI workflow preview copy\n")
    mfolder = mirror / folder_rel if folder_rel else mirror
    write_files(mfolder / spec_rel, files)
    shim_dir, log_file = work / "shim", work / "shim.log"
    real = {t: sub.which(t, env) for t in shim.TOOLS}
    shim.install(shim_dir, real, log_file)
    if not real.get("sbatch"):
        warnings.append("No SLURM here: the preview uses default cluster limits")
    penv = dict(env)
    penv["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    before = sub.listing(mirror)
    logs: List[str] = []
    code = 0
    # 1. the plan
    write_argv = ["workflow", "--spec", f"{spec_rel}/spec.json", "--write-plan", "--json"]
    r = clirun.run(clirun.mace_argv("--no-banner", *write_argv), mfolder, ctx, timeout=sub.PREVIEW_TIMEOUT_S, env=penv)
    logs.append(r.log)
    w = parse_json(r.stdout)
    commands = [f"{'cd ' + shlex.quote(folder_rel) + ' && ' if folder_rel else ''}mace "
                + " ".join(shlex.quote(a) for a in write_argv[:-1])]
    seq, _ = draft_sequence(draft)
    n = len(inp.materials)
    title = (f"{str(draft.get('label') or '').strip() or 'workflow'}: {' → '.join(seq)} for "
             f"{n} structure{'' if n == 1 else 's'}")
    result: Dict[str, Any] = {"folder_rel": folder_rel, "inputs": inputs_view(st, inp), "warnings": warnings,
                              "title": title, "sequence": seq,
                              "blocked": blocked, "blocked_rows": blocked_rows, "commands": commands,
                              "spec": {"rel": f"{folder_rel + '/' if folder_rel else ''}{spec_rel}/spec.json",
                                       "text": files["spec.json"],
                                       "settings": {k: v for k, v in files.items() if k not in ("spec.json", UI_DRAFT_NAME)}},
                              "staged_bytes": staged, "first_step": [], "projected": [], "workflow_scripts": [],
                              "writes": [], "intercepted": [], "plan": None, "launch": None}
    launch: List[str] = []
    wid = None
    if r.returncode != 0 or not isinstance(w, dict) or not w.get("ok"):
        code = r.returncode or 1
        why = (w or {}).get("error") if isinstance(w, dict) else None
        result["refused"] = why or (r.stderr or r.stdout).strip()[-800:]
    else:
        wid = w["workflow_id"]
        launch = list(w["launch"])
        plan_file = Path(w["plan_file"])
        result["plan"] = {"rel": sub._rel(root, Path(_unmirror(plan_file, mirror, root))),
                          "text": _unmirror(plan_file.read_text(), mirror, root)}
        result["launch"] = [_unmirror(a, mirror, root) for a in launch]
        commands.append(_display([_unmirror(a, mirror, root) for a in launch], folder))
        # 2. the launch: the first step "submitted" through the recording shim
        r2 = clirun.run(clirun.mace_argv(*launch[1:]), mfolder, ctx, timeout=sub.PREVIEW_TIMEOUT_S, env=penv)
        logs.append(r2.log)
        code = r2.returncode
    after = sub.listing(mirror)
    calls = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()] if log_file.exists() else []
    if any(c.get("no_slurm") for c in calls) and "No SLURM here: the preview uses default cluster limits" not in warnings:
        warnings.append("No SLURM here: the preview uses default cluster limits")
    writes = sorted(k for k in after if k not in before or after[k][2] != before[k][2])
    result["writes"] = [w_ for w_ in writes if not w_.endswith(MIRROR_MARKER)]
    scripts: Dict[str, str] = {}
    for c in calls:
        c["argv"] = [_unmirror(a, mirror, root) for a in c["argv"]]
        c["cwd"] = _unmirror(c.get("cwd", ""), mirror, root)
    sbatches = [c for c in calls if c["tool"] == "sbatch" and c.get("intercepted")]
    from mace.submission.job_array import read_manifest
    for c in sbatches:
        text = c.get("script_text") or ""
        spath = Path(c.get("script") or "")
        if not spath.is_absolute():
            spath = Path(_unmirror(c.get("cwd") or "", root, mirror)) / spath
        try:
            srel = spath.relative_to(mirror).as_posix()
        except ValueError:
            srel = spath.name
        res = sub._resources(text)
        members = []
        if res.get("array"):
            for m in (spath.parent.glob("*manifest*") if spath.parent.exists() else []):
                try:
                    members = read_manifest(m)
                    break
                except OSError:
                    pass
        targets = [(Path(m["script"]), m["job"]) for m in members] if members else [(spath, spath.stem)]
        for sp, job in targets:
            try:
                stext = sp.read_text() if sp != spath else text
            except OSError:
                stext = text
            deck = sp.with_suffix(".d12")
            try:
                rel = sp.relative_to(mirror).as_posix()
            except ValueError:
                rel = sp.name
            scripts[rel] = normalize_script(stext, mirror, root)
            result["first_step"].append({
                "material_id": job, "script_rel": rel, "script_text": _unmirror(stext, mirror, root),
                "deck_rel": sub._rel(mirror, deck) if deck.exists() else None, "deck_name": deck.name,
                "deck_text": deck.read_text(errors="replace") if deck.exists() else None,
                "resources": sub._resources(stext), "array": bool(res.get("array")), "wrapper": srel})
    result["first_step"].sort(key=lambda x: x["material_id"])
    result["intercepted"] = [c["argv"] for c in calls if c.get("intercepted")]
    for rel in result["writes"]:
        if "/workflow_scripts/" in f"/{rel}" and rel.endswith(".sh"):
            text = (mirror / rel).read_text(errors="replace")
            scripts[rel] = normalize_script(text, mirror, root)
            result["workflow_scripts"].append({"rel": rel, "text": _unmirror(text, mirror, root)})
    # 3. the later steps of one structure, as the engine writes them from stand-in parents
    if code == 0 and wid and len(seq) > 1:
        pr = clirun.run([sys.executable, "-m", "mace.workflow.projection", "--folder", str(mfolder),
                         "--workflow-id", wid, "--json"], mfolder, ctx, timeout=sub.PREVIEW_TIMEOUT_S,
                        env=dict(penv, PYTHONPATH=str(clirun.REPO)))
        pj = parse_json(pr.stdout)
        if isinstance(pj, dict) and pj.get("ok"):
            for s in pj.get("steps") or []:
                deck = Path(s["input_file"])
                sh = Path(s["script"]) if s.get("script") else None
                result["projected"].append({
                    "step": s["step"], "calc_type": s["calc_type"], "material_id": s["material_id"],
                    "parent": s["parent"], "projected": True, "deck_name": deck.name,
                    "deck_text": deck.read_text(errors="replace") if deck.is_file() else None,
                    "script_text": _unmirror(sh.read_text(), mirror, root) if sh and sh.is_file() else None,
                    "resources": sub._resources(sh.read_text()) if sh and sh.is_file() else {}})
            result["not_projected"] = pj.get("not_projected") or []
            # a step the engine's deck writer refused (a SOC deck it cannot write): its reason, in its words
            for why in dict.fromkeys(m.group(1).strip() for m in NOT_WRITING.finditer(pr.stderr or "")):
                warnings.append(f"Projection: a later step's deck was not written: {why}")
            if pj.get("problems"):
                warnings.extend(f"Projection: {p}" for p in pj["problems"])
        else:
            warnings.append("The later steps could not be projected: "
                            + ((pj or {}).get("error") if isinstance(pj, dict) else (pr.stderr or "").strip()[-300:]))
    log = "\n".join(logs).replace(str(mirror) + os.sep, "").replace(str(mirror), ".")
    result["cli_log"] = log[-8000:]
    result["code"] = code
    result["workflow_id_preview"] = wid
    for wline in sub._cli_warnings(log):
        # Python's own notices about the CLI's code (a DeprecationWarning on stderr) are not the user's business
        if re.search(r"(Deprecation|Future|Pending\w*|Syntax|Resource)Warning", wline):
            continue
        if wline not in warnings:
            warnings.append(wline)
    result["n_jobs"] = len(result["first_step"])
    needs: Dict[str, bool] = {}
    if inp.kind == "d12":
        conf = sub.conflicts(root, sel, list(inp.files), None)
        result["conflicts"] = conf
        needs["overwrite"] = bool(conf["finished"] or conf["wavefunction"])
    p = WfPreview(id="", created=time.time(), folder=folder, inputs=inp, spec_rel=spec_rel, files=files, fp=fp,
                  result=result, launch_shape=_shape(result["launch"] or []), scripts=scripts, needs=needs)
    return p, result


def finalize(store: sub.Store, p: WfPreview) -> Dict[str, Any]:
    p.id = "wf-" + store.sign(p.fp)
    store.put(p)
    p.result["preview_id"] = p.id
    p.result["needs"] = dict(p.needs)
    return p.result


# ------------------------------------------------------------------ run
def check_run(store: sub.Store, pid: str, root: Path, root_gen: Any, ticks: Dict[str, bool]) -> WfPreview:
    p = store.take(pid)
    if p.fp.get("root_gen") != root_gen:
        raise sub.SubmitConflict("The folder changed; reload")
    env = cli_env()
    now = fingerprint(root_gen, p.inputs, p.files, env)
    if now["blocked"]:
        mats = sorted({json.loads(b)["material_id"] for b in now["blocked"]})
        raise sub.SubmitConflict(f"{', '.join(mats)} already have unfinished steps in a workflow in this folder; "
                                 f"wait for them or cancel that workflow first")
    if now["inputs"] != p.fp["inputs"] or now["spec"] != p.fp["spec"]:
        raise sub.SubmitConflict("The structures changed since the preview; preview again")
    if now["site"] != p.fp["site"]:
        raise sub.SubmitConflict("Site settings changed since the preview; preview again")
    if now["module"] != p.fp["module"]:
        raise sub.SubmitConflict("The CRYSTAL module MACE would load changed since the preview; preview again")
    if now["blocked"] != p.fp["blocked"]:
        raise sub.SubmitConflict("The queue changed since the preview; preview again")
    if p.needs.get("overwrite") and not ticks.get("confirm_overwrite"):
        raise sub.SubmitConflict("Tick “Overwrite finished results” to start from these decks", need="overwrite")
    if p.result.get("code"):
        raise sub.SubmitConflict("The preview failed; nothing can be submitted (see its log)")
    return p


def launched_job_ids(folder: Path, workflow_id: str) -> List[str]:
    """The SLURM job ids the launch recorded in the workflow's database."""
    from mace.workflow import status
    try:
        db = status.database_of(folder, workflow_id, status.load_plan(folder, workflow_id))
        rows = status.read_rows(db, workflow_id) if db is not None else []
    except Exception:
        return []
    out: List[str] = []
    for r in rows:
        j = str(r.get("slurm_job_id") or "").strip()
        if j and j not in out:
            out.append(j)
    return out


class LaunchLock:
    """One GUI launch at a time in a folder: ``workflow_configs/.gui_launch.lock``
    made with O_EXCL. It records the launching process and host: a lock of
    this host whose process is gone is a crashed launch's, and one of a
    running process holds however long the launch takes; a lock of another
    host (or an older one without the host) is a crashed launch's after half
    an hour."""

    def __init__(self, folder: Path):
        self.path = folder / "workflow_configs" / LOCK_NAME
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self.fd, f"{os.getpid()} {time.time():.0f} {socket.gethostname()}\n".encode())
                return self
            except FileExistsError:
                old = self._stale()
                if attempt == 0 and old:
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                raise sub.SubmitConflict("Another workflow is being launched from this folder; wait for it to finish")
        raise sub.SubmitConflict("Another workflow is being launched from this folder; wait for it to finish")

    def _stale(self) -> bool:
        try:
            parts = self.path.read_text().split()
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return True
        if len(parts) >= 3 and parts[2] == socket.gethostname():
            try:
                pid = int(parts[0])
            except ValueError:
                return age > LOCK_STALE_S
            if pid <= 0:
                return True
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except OSError:
                return False            # alive, another user's
            return False
        return age > LOCK_STALE_S

    def __exit__(self, *a):
        try:
            if self.fd is not None:
                os.close(self.fd)
            self.path.unlink()
        except OSError:
            pass
        return False


def run(session: sub.Store, p: WfPreview, root: Path, ctx=None, on_output=None) -> Dict[str, Any]:
    """The real run of a checked preview: the spec written where the preview
    wrote it, then the same two commands, in the folder, with the
    environment ``mace gui`` was started with (no time limit, no cancel).
    ``session`` is Submit's store: this session's job ids, and the count of
    launches a server shutdown waits for."""
    env = cli_env()
    t0 = time.time()
    folder = p.folder
    session.running += 1
    try:
        with LaunchLock(folder):
            write_files(folder / p.spec_rel, p.files, exclusive=True)
            write_argv = ["workflow", "--spec", f"{p.spec_rel}/spec.json", "--write-plan", "--json"]
            r = clirun.run(clirun.mace_argv("--no-banner", *write_argv), folder, ctx, timeout=None, env=env,
                           nice=False, cancelable=False, on_output=on_output)
            logs = [r.log]
            w = parse_json(r.stdout)
            if r.returncode != 0 or not isinstance(w, dict) or not w.get("ok"):
                why = (w or {}).get("error") if isinstance(w, dict) else (r.stderr or r.stdout).strip()[-800:]
                return {"code": r.returncode or 1, "log": "\n".join(logs)[-8000:], "error": why, "job_ids": [],
                        "script_drift": []}
            launch = list(w["launch"])
            drift = []
            if _shape(launch) != p.launch_shape:
                drift.append({"command": " ".join(launch), "diff": "the launch command differs from the preview's: "
                              + " ".join(p.launch_shape)})
            r2 = clirun.run(clirun.mace_argv(*launch[1:]), folder, ctx, timeout=None, env=env, nice=False,
                            cancelable=False, on_output=on_output)
            logs.append(r2.log)
        log = "\n".join(logs)
        ids = launched_job_ids(folder, w["workflow_id"]) or sub.job_ids_of(log, [folder], t0)
        for j in ids:
            session.session_jobs[j] = {"t": time.time(), "workdir": str(folder)}
        import difflib
        prev_wid = p.result.get("workflow_id_preview") or ""
        for prel, text in p.scripts.items():
            rel = prel.replace(prev_wid, w["workflow_id"]) if prev_wid else prel
            try:
                now = normalize_script((root / rel).read_text())
            except OSError:
                drift.append({"script": rel, "diff": "the preview wrote this job script; the run did not"})
                continue
            if now != text:
                diff = "".join(difflib.unified_diff(text.splitlines(True), now.splitlines(True),
                                                    "preview/" + rel, rel, n=1))
                drift.append({"script": rel, "diff": diff[-4000:]})
        return {"code": r2.returncode, "workflow_id": w["workflow_id"],
                "plan_rel": sub._rel(root, Path(w["plan_file"])), "folder_rel": sub._rel(root, folder),
                "job_ids": ids, "script_drift": drift, "log": log.replace(str(root) + os.sep, "")[-8000:],
                "commands": p.result.get("commands")}
    finally:
        session.running -= 1
