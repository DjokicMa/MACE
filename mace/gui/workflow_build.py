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
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import clirun
from . import submit as sub
from . import slurm_preview_shim as shim
# opt2d12's modules (Crystal_d12/d12_calc_elastic.py): steps after an OPT or SP, nothing built from them
from mace.constants import HARTREE_TO_EV
from mace.workflow.common.constants import MODULE_STEP_TYPES, STAND_IN_FERMI_AU

SPEC_DIR = "workflow_configs/gui_specs"
UI_DRAFT_NAME = "builder_draft.json"
LOCK_NAME = ".gui_launch.lock"
LOCK_STALE_S = 1800.0
MIRROR_MARKER = ".mace_preview_mirror"
MAX_INPUTS = 2000
D3_TYPES = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS")
METHOD_FIELDS = ("functional", "dispersion")
# A later SCF step's DFT grid and SCF block: each inherited from the structure's parent step unless set here
# (the spec's grid_modifications, tolerance_modifications and scf_modifications)
# (and its spin polarization, the spec's scf_modifications SPIN, and its k-point mesh, kpoint_modifications)
# The k-point mesh row is Calculation Setup's opt2d12 patch row (k_points_for_all_files, the key opt2d12 reads
# for a mesh): its answer goes to the CLI as that patch, and to the spec as kpoint_modifications.
KMESH = "k_points_for_all_files"
SCF_FIELDS = ("dft_grid", "tolerance", "maxcycle", "fmixing", "scf_method", "levshift", "levshift_values",
              "spinlock", "smearing", "smear_width", "spin", KMESH)
# What a later SCF step's form can change, per step type: the functional, the basis set, the DFT grid and the
# SCF block on every SCF step; OPT's type and convergence; FREQ's template, NUMDERIV and IR. Everything else
# stays what the structure's parent step had.
LATER_FIELDS = {
    "OPT": METHOD_FIELDS + ("basis_set",) + SCF_FIELDS + ("opt_type", "opt_convergence"),
    "SP": METHOD_FIELDS + ("basis_set",) + SCF_FIELDS,
    "FREQ": METHOD_FIELDS + ("basis_set",) + SCF_FIELDS + ("freq_template", "numderiv", "ir"),
}
RESOURCE_KEYS = ("ntasks", "walltime", "memory", "memory_per_cpu", "account", "constraint", "partition", "exclude")
NO_TEMPLATES = ("custom", "interactive")
LABEL_RE = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")
STEP_KEY_RE = re.compile(r"^([A-Z+]+[0-9]*)_(\d+)$")
NOT_WRITING = re.compile(r"Not writing \S+?\.d12: (.+)$", re.M)
# opt2d12's refusal of an ELASTIC/EOS/QHA/CPHF deck (d12_calc_elastic.resolve_for_opt2d12)
NOT_WRITING_MODULE = re.compile(r"Not writing the (?:ELASTIC|EOS|QHA|CPHF) deck: (.+)$", re.M)

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
    "TAU": "Relaxation time for TRANSPORT from deformation potentials: strained single points of the SP "
           "(4 per axis), fitted to E1 and C, with the masses from BAND and DOSS. TRANSPORT then runs with "
           "RELAXTIM = τ (10 fs, with the reason, when τ cannot be computed).",
    "ELASTIC": "Elastic constants (ELASTCON), with piezoelectric (ELAPIEZO, PIEZOCON, PIEZOCP) or photoelastic "
               "(PHOTOELA) constants: strained cells of the optimized structure, one CRYSTAL job.",
    "EOS": "Equation of state: energy against volume around the optimized cell, the bulk modulus and its "
           "pressure derivative (3D crystals).",
    "QHA": "Quasi-harmonic approximation: Γ phonons at several volumes, thermal expansion and thermodynamics "
           "against temperature (3D crystals).",
    "CPHF": "Dielectric response (CPHF/KS): polarizability and dielectric tensors, optionally the "
            "hyperpolarizabilities and frequency-dependent values.",
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
    from mace.workflow.common.constants import planner_template_numbers
    nums = planner_template_numbers()          # the number the CLI's template menu gives each
    temps = [{"name": n, "sequence": list(seq), "help": WORKFLOW_TEMPLATE_DESCRIPTIONS.get(n, " → ".join(seq)),
              "num": nums.get(n)}
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


def _hf_names() -> set:
    from . import setup as gui_setup
    return {n.upper() for n in gui_setup.hf_methods()}


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


# A SOC deck needs explicit basis input (no BASISSET: soc_ecp.BASISSET_REFUSAL). The external library a
# step's SOC deck is written with when its basis is an internal set, and the official SOC set that matches
# a library's valence shells (pob-TZVP-rev2-SOC: pob-TZVP-rev2's shells with CRYSTAL's STUTSC ECP).
SOC_EXTERNAL_FOR = {"POB-TZVP-REV2": "full.basis.triplezeta", "POB-DZVP-REV2": "full.basis.doublezeta"}
SOC_SET_FOR = {"full.basis.triplezeta": "pob-tzvp-rev2-soc", "full.basis.triplezeta.rev2022": "pob-tzvp-rev2-soc"}


def _soc_tables():
    """d12_constants, basis_library and soc_ecp: the deck writer's own tables."""
    from mace.workflow.spec import _d12_constants
    C = _d12_constants()
    import basis_library as BL
    import soc_ecp as E
    return C, BL, E


def soc_functionals() -> List[str]:
    """The functionals a two-component deck takes (soc_ecp.TWOC_FUNCTIONALS, and those whose -D3 form it
    takes), as the Functional row names them."""
    _, _, E = _soc_tables()
    return sorted(set(E.TWOC_FUNCTIONALS) | {f[:-3] for f in E.TWOC_D3_FUNCTIONALS})


def soc_basis_default(inp: Inputs, kind: Optional[str], name: Optional[str]) -> Dict[str, Any]:
    """What a SOC deck of a step whose basis is ``kind``/``name`` (the step's own or inherited) is written with
    by default: the external library (``basis_set``, a Basis set row option; None keeps the step's), the
    official SOC set for its heavy atoms (``soc_basis``; None: MACE's INPSOC ECP, the default route), and a
    note saying why, naming the elements that have no SOC basis."""
    from . import setup as gui_setup
    C, BL, E = _soc_tables()
    label = lambda fam: BL.FAMILY_LABELS.get(fam, fam)                     # noqa: E731
    out: Dict[str, Any] = {"parent_kind": kind, "parent": name, "basis_set": None, "soc_basis": None,
                           "missing": [], "note": ""}
    words = []
    if kind == "INTERNAL":
        fam = SOC_EXTERNAL_FOR.get(str(name or "").upper(), "full.basis.triplezeta")
        out["basis_set"] = label(fam)
        same = str(name or "").upper() in SOC_EXTERNAL_FOR
        out["parent"] = f"{name} (internal)"
        words.append(f"A SOC deck needs explicit basis input, and {name} is an internal set (BASISSET): the "
                     f"basis set becomes {label(fam)}, MACE's external library"
                     + (f" with the same shells as {name}." if same else
                        f" (no external library matches {name}; pick another in Basis set)."))
    else:
        fam = BL.bundled_family_for(name) if name else None
        out["parent"] = label(fam) if fam else (name or "the parent's own records")
    soc_set = SOC_SET_FOR.get(fam)
    elements, unread = set(), []
    for f in inp.files:
        z = gui_setup._atomic_numbers(f)
        if isinstance(z, str):
            unread.append(z)
        else:
            elements |= set(z)
    ecp = sorted(z for z in elements if z in C.ECP_ELEMENTS_EXTERNAL)
    covered = set(BL.soc_basis_elements())
    out["missing"] = [E._symbol(z) for z in ecp if z not in covered]
    if soc_set and ecp and not out["missing"] and not unread:
        out["soc_basis"] = soc_set
        words.append(f"The ECP atoms ({', '.join(E._symbol(z) for z in ecp)}) take the official "
                     f"pob-TZVP-rev2-SOC basis set (its STUTSC spin-orbit ECP), which matches "
                     f"{label(fam)}'s shells.")
    elif soc_set and out["missing"]:
        words.append(f"{', '.join(out['missing'])} {'has' if len(out['missing']) == 1 else 'have'} no "
                     f"pob-TZVP-rev2-SOC basis set (it covers Rb-I but Tc, Cs, Ba and Hf-Bi), so the ECPs get "
                     f"MACE's INPSOC spin-orbit records (the default route).")
    elif soc_set and unread:
        words.append("The elements could not be read (" + "; ".join(unread[:2]) + "): the ECPs get MACE's INPSOC "
                     "records; pick the official SOC basis set by hand if every heavy element has one.")
    elif not ecp and elements:
        words.append(f"No element of these structures has an ECP in {label(fam) if fam else 'this basis'}: "
                     f"CRYSTAL puts spin-orbit coupling only in the ECP (manual p. 175), so the CLI notes "
                     f"that the all-electron atoms get no spin-orbit term.")
    else:
        words.append("The ECPs get MACE's INPSOC spin-orbit records (the default route).")
    out["note"] = " ".join(words)
    return out


def spec_files(inp: Inputs, draft: Dict[str, Any], spec_dir: Optional[Path] = None,
               step1_overrides: Optional[Dict[str, Any]] = None,
               later_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
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
                from mace.workflow.spec import _a
                raise BuildError(f"{key}: the settings file is for {_a(sorted(named)[0])} step")
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
            if base.upper() in _hf_names():
                raise BuildError(f"{key}: {fn} is a Hartree-Fock method; the method (Hartree-Fock or DFT) is "
                                 f"set on the first step, and a later step keeps it")
            mods = {"new_functional": base}
            if m.get("dispersion") is not None:
                mods["use_dispersion"] = bool(m.get("dispersion"))
            elif re.search(r"-D3$", fn, flags=re.I):
                mods["use_dispersion"] = True
            # else: no word on dispersion, the parent's D3 is kept (engine + opt2d12)
            entry["overrides"] = {"method_modifications": mods}
        # a later SCF step's other settings (basis set, SCF convergence, OPT and FREQ settings): the CLI's own
        # answers to the form (Builder.later_overrides)
        extra = (later_overrides or {}).get(key)
        if extra:
            if i == 0 or calc_kind(seq[i]) != "d12":
                raise BuildError(f"{key}: these settings are changed only on a later SCF step (SP, OPT2, FREQ)")
            from mace.workflow.spec import _merge
            _merge(entry.setdefault("overrides", {}), copy.deepcopy(extra))
        # a TAU step's options, a TRANSPORT step's relaxation time (mace/workflow/tau_step.py)
        tau_opts = s.get("tau")
        if tau_opts:
            if base_type(seq[i]) != "TAU":
                raise BuildError(f"{key}: these are a TAU step's settings")
            from mace.workflow.tau_step import TauOptionError, check_options
            try:
                check_options(tau_opts, key)
            except TauOptionError as e:
                raise BuildError(str(e))
            entry.setdefault("overrides", {}).update(copy.deepcopy(tau_opts))
        # an ELASTIC/EOS/QHA/CPHF step's module block (spec override elastic_settings)
        mod = s.get("module")
        if mod:
            if base_type(seq[i]) not in MODULE_STEP_TYPES:
                raise BuildError(f"{key}: these are an ELASTIC, EOS, QHA or CPHF step's settings")
            from mace.workflow.spec import module_settings_problem
            why = module_settings_problem(seq[i], mod)
            if why:
                raise BuildError(f"{key}: {why}")
            entry.setdefault("overrides", {})["elastic_settings"] = copy.deepcopy(mod)
        rt = s.get("relaxation_time")
        if rt not in (None, "", "default"):                  # default: CRYSTAL's 10 fs, nothing written
            if base_type(seq[i]) != "TRANSPORT":
                raise BuildError(f"{key}: the relaxation time is a TRANSPORT step's setting")
            from mace.workflow.tau_step import TauOptionError, check_relaxation_time
            try:
                entry.setdefault("overrides", {})["relaxation_time"] = check_relaxation_time(rt, key)
            except TauOptionError as e:
                raise BuildError(str(e))
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
               step1_overrides: Optional[Dict[str, Any]] = None,
               later_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """``mace workflow --spec SPEC --json`` (nothing written) in the GUI's
    scratch: each step's settings and resources and where they come from,
    with the partition's time limit checked."""
    files = spec_files(inp, draft, None, step1_overrides, later_overrides)
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


def split_kmesh(answers: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A later SCF step's answers for Calculation Setup: (the answers to the CLI's questions, the patch of the
    k-point mesh row, which no question sets)."""
    ans = {k: v for k, v in (answers or {}).items() if k != KMESH}
    return ans, ({KMESH: str((answers or {})[KMESH]).strip()} if KMESH in (answers or {}) else {})


def kmesh_answer(key: str, value: Any) -> str:
    """The k-point mesh row's answer as the spec's kpoint_modifications new_k_points (BuildError if it is not
    one: d12_constants.check_kpoint_modifications)."""
    from mace.workflow.spec import _d12_constants
    try:
        return _d12_constants().check_kpoint_modifications({"new_k_points": str(value).strip()})["new_k_points"]
    except ValueError:
        raise BuildError(f"{key}: the k-point mesh is {_d12_constants().K_MESH_FORMS}, not {value!r}")


def check_form_answers(inp: Inputs, seq: List[str], i: int, body: Dict[str, Any]) -> None:
    """A later SCF step inherits its parent's settings: only the method changes there."""
    if i == 0 and inp.kind == "d12":
        return
    if base_type(seq[i]) == "TAU":
        # the step's options, checked as the CLI checks them (mace/workflow/tau_step.py)
        from mace.workflow.tau_step import TauOptionError, check_options
        key = f"{seq[i]}_{i + 1}"
        try:
            check_options(((body.get("draft") or {}).get("steps") or {}).get(key, {}).get("tau") or {}, key)
        except TauOptionError as e:
            raise BuildError(str(e))
    check_soc(f"{seq[i]}_{i + 1}", seq[i], {k: v for k, v in (body.get("patches") or {}).items() if k in SOC_KEYS})
    if i > 0 and calc_kind(seq[i]) == "d12":
        answers = body.get("answers") or {}
        allowed = LATER_FIELDS.get(base_type(seq[i]), METHOD_FIELDS)
        hf = str(answers.get("functional") or "").upper() in _hf_names()
        bad = sorted(set(answers) - set(allowed)) + (["the settings file"] if body.get("json_text") else []) \
            + sorted(k for k in (body.get("patches") or {}) if k not in SOC_KEYS) \
            + (["a Hartree-Fock method (set on the first step)"] if hf else [])
        if bad:
            raise BuildError(f"{seq[i]} inherits its parent's settings for each structure; only the functional, "
                             f"dispersion, basis set, DFT grid, SCF settings and k-point mesh"
                             f"{' (and the optimization settings)' if base_type(seq[i]) == 'OPT' else ''}"
                             f"{' (and the frequency settings)' if base_type(seq[i]) == 'FREQ' else ''} "
                             f"can change here (not {', '.join(bad)})")
        if KMESH in answers:
            kmesh_answer(f"{seq[i]}_{i + 1}", answers[KMESH])


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

    # ---- a later SCF step's answers and what they change
    def step_answers(self, key: str) -> Dict[str, Any]:
        """A later SCF step's answers for Calculation Setup: the form's, with the functional and
        dispersion of its method (the draft keeps both), limited to what that step type can change."""
        s = (self.draft.get("steps") or {}).get(key) or {}
        base = base_type(key.rsplit("_", 1)[0])
        allowed = LATER_FIELDS.get(base, METHOD_FIELDS)
        ans = {k: v for k, v in (s.get("answers") or {}).items() if k in allowed}
        if soc_patches(s):
            ans.pop("spin", None)        # a two-component step treats spin itself
        if ans.get("basis_set") in (None, "", "Keep the parent's basis set"):
            ans.pop("basis_set", None)
        m = s.get("method") or {}
        if m.get("functional"):
            ans["functional"] = re.sub(r"-D3$", "", str(m["functional"]), flags=re.I)
            if m.get("dispersion") is not None:
                ans["dispersion"] = "Yes" if m.get("dispersion") else "No"
        return ans

    def later_overrides(self) -> Dict[str, Dict[str, Any]]:
        """{step key: overrides} for the later SCF steps whose form changed more than the method: the
        spec keys the engine reads (basis_modifications, tolerance_modifications, optimization_settings,
        frequency_settings), worked out from the CLI's own saved settings for the answers (Setup run on a
        stand-in parent, as for the first step) and never written by the GUI."""
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(1, len(self.seq)):
            if calc_kind(self.seq[i]) != "d12":
                continue
            why = self.soc_basis_problem(i)
            if why:
                raise BuildError(why)
            key = f"{self.seq[i]}_{i + 1}"
            ans, kp = split_kmesh(self.step_answers(key))
            ov = {"kpoint_modifications": {"new_k_points": kmesh_answer(key, kp[KMESH])}} if kp else {}
            if set(ans) - set(METHOD_FIELDS):
                ov.update(self._derive_overrides(i, key, ans))
            if ov:
                out[key] = ov
        return out

    def _derive_overrides(self, i: int, key: str, ans: Dict[str, Any]) -> Dict[str, Any]:
        pkey = parent_key(self.seq, i)
        pdeck = self.parent_deck(int(pkey.rsplit("_", 1)[1]) - 1)
        out_path = stand_in(self.work / f"ovstandin_{i}", self.inp.materials[0], pdeck)
        res = self.setup({"answers": ans, "form": False, "preview": False}, "opt2d12", base_type(self.seq[i]),
                         f"ov{i}", out_path, f"projected/{out_path.name}",
                         ("wf-standin", hashlib.sha256(pdeck.encode()).hexdigest(), "opt2d12"), schema=False)
        if not res.get("json_text"):
            raise BuildError(f"{key}: the CLI wrote no settings for these answers: " + refusal_line(res))
        data = json.loads(res["json_text"])
        ov: Dict[str, Any] = {}
        if "basis_set" in ans:
            ov["basis_modifications"] = basis_modifications(key, data)
        if "tolerance" in ans and isinstance(data.get("tolerances"), dict):
            tol = {k: data["tolerances"][k] for k in ("TOLINTEG", "TOLDEE") if k in data["tolerances"]}
            if tol:
                ov["tolerance_modifications"] = {"custom_tolerances": tol}
        if "dft_grid" in ans and data.get("dft_grid"):
            ov["grid_modifications"] = {"new_grid": data["dft_grid"]}
        scf = scf_modifications(ans, data)
        if scf:
            ov["scf_modifications"] = scf
        if {"opt_type", "opt_convergence"} & set(ans):
            if data.get("optimization_type"):
                ov["optimization_type"] = data["optimization_type"]
            if isinstance(data.get("optimization_settings"), dict):
                ov["optimization_settings"] = data["optimization_settings"]
        if {"freq_template", "numderiv", "ir"} & set(ans) and isinstance(data.get("freq_settings"), dict):
            ov["frequency_settings"] = data["freq_settings"]
        return ov

    def step_basis(self, i: int) -> Tuple[Optional[str], Optional[str]]:
        """(INTERNAL | EXTERNAL, the set's name or library) of step ``i``'s deck for the first structure: its
        own Basis set answer, else its parent's (a CIF start's conversion settings, a deck start's deck)."""
        key = f"{self.seq[i]}_{i + 1}"
        if i > 0 and calc_kind(self.seq[i]) == "d12":
            ans = ((self.draft.get("steps") or {}).get(key) or {}).get("answers") or {}
            b = ans.get("basis_set")
            if b not in (None, "", "Keep the parent's basis set"):
                from . import setup as gui_setup
                _, BL, _ = _soc_tables()
                fam = {v: k for k, v in BL.FAMILY_LABELS.items()}.get(str(b))
                if fam or str(b) in gui_setup.external_basis_labels() or "/" in str(b):
                    return "EXTERNAL", fam or str(b)
                return "INTERNAL", str(b)
        if i == 0 or calc_kind(self.seq[i]) != "d12":
            if i == 0 and self.inp.kind == "cif":
                cfg = self.cif_effective()
                kind = "EXTERNAL" if str(cfg.get("basis_set_type") or "").upper() == "EXTERNAL" else "INTERNAL"
                return kind, cfg.get("basis_set_path") if kind == "EXTERNAL" and cfg.get("basis_set_path") \
                    else cfg.get("basis_set")
            if i == 0:
                m = re.search(r"^BASISSET\s*\n\s*(\S+)", self.inp.files[0].read_text(errors="replace"), re.M)
                return ("INTERNAL", m.group(1)) if m else ("EXTERNAL", None)
        pkey = parent_key(self.seq, i)
        return self.step_basis(int(pkey.rsplit("_", 1)[1]) - 1) if pkey else (None, None)

    def soc_basis_problem(self, i: int) -> Optional[str]:
        """Why step ``i``'s SOC deck cannot be written with its basis set (an internal set), with the external
        library to pick; None when SOC is off or the basis is external."""
        key = f"{self.seq[i]}_{i + 1}"
        s = (self.draft.get("steps") or {}).get(key) or {}
        if not soc_patches(s) or i == 0:
            return None
        kind, name = self.step_basis(i)
        if kind != "INTERNAL":
            return None
        d = soc_basis_default(self.inp, kind, name)
        return (f"{key}: a spin-orbit (SOC) deck needs an external basis set (explicit basis input for its "
                f"spin-orbit ECPs), and {name} is an internal one; pick {d['basis_set']} (or another external "
                f"set) in Basis set")

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
            answers, kp = split_kmesh(self.step_answers(key) if calc_kind(seq[i]) == "d12" else {})
            res = self.setup({"answers": answers, "patches": {**soc_patches(s), **kp}, "form": False, "preview": True},
                             "opt2d12", base_type(seq[i]),
                             f"p{i}", out_path, f"projected/{out_path.name}",
                             ("wf-standin", hashlib.sha256(pdeck.encode()).hexdigest(), "opt2d12"), schema=False)
        if not res.get("deck"):
            raise BuildError(f"No {key} deck to build the next step's form from: " + refusal_line(res))
        return res["deck"]


def basis_modifications(key: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """The spec's basis_modifications for the basis set the CLI saved (its settings file ``data``): an internal
    set by name, or an external library (MACE's bundled family name, else its directory), either way whatever
    the parent step had."""
    if data.get("basis_set_type") == "EXTERNAL":
        lib = data.get("basis_set_path") or data.get("basis_set")
        if not lib or str(lib).startswith("EXTERNAL (from"):
            raise BuildError(f"{key}: the CLI named no external basis library for this choice")
        try:
            from mace.workflow.spec import _d12_constants
            _d12_constants()
            import basis_library
            lib = basis_library.portable_path(str(lib))
        except ImportError:                                                  # pragma: no cover
            pass
        return {"new_basis": str(lib), "basis_set_type": "EXTERNAL"}
    if not data.get("basis_set"):
        raise BuildError(f"{key}: the CLI named no basis set for this choice")
    return {"new_basis": data["basis_set"]}


def scf_modifications(ans: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """The spec's scf_modifications for a later step's answered SCF fields: the values the CLI saved for
    them (its settings file ``data``), as the keys CRYSTALOptToD12 reads (d12_constants.SCF_MODIFICATION_KEYS)."""
    scf = data.get("scf_settings") if isinstance(data.get("scf_settings"), dict) else {}
    out: Dict[str, Any] = {}
    if "maxcycle" in ans and scf.get("maxcycle") is not None:
        out["MAXCYCLE"] = int(scf["maxcycle"])
    if "fmixing" in ans and scf.get("fmixing") is not None:
        out["FMIXING"] = int(scf["fmixing"])
    if "scf_method" in ans and scf.get("method"):
        out["METHOD"] = str(scf["method"]).upper()
    if {"levshift", "levshift_values"} & set(ans):
        lv = scf.get("levshift")
        out["LEVSHIFT"] = [int(lv[0]), int(lv[1])] if isinstance(lv, (list, tuple)) and len(lv) >= 2 else None
    if {"smearing", "smear_width"} & set(ans):
        width = data.get("smearing_width")
        out["SMEAR"] = float(width) if data.get("use_smearing") and width else None
    if "spin" in ans:
        out["SPIN"] = bool(data.get("spin_polarized", data.get("is_spin_polarized")))
    if "spinlock" in ans and out.get("SPIN") is not False:
        out["SPINLOCK"] = int(data.get("spinlock") or 0)
    return out


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


STAND_IN_KPATH_NOISE = re.compile(r"no computed geometry block found|seekpath library failed|"
                                  r"Shrink factor \d+ cannot represent the k-path points exactly")
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
      step's deck; the method, basis set, DFT grid and SCF settings (and OPT's
      or FREQ's own) can change, everything else is each structure's parent's
      (``editable``; an unanswered field shows the parent step's value);
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
    if calc == "TAU":
        return tau_form(inp, draft, seq, i, pkey, pdeck)
    if calc in MODULE_STEP_TYPES:
        return module_form(inp, draft, seq, i, pkey, pdeck, work, ctx, cache_get, cache_put)
    out_path = stand_in(work / f"standin_{i}", inp.materials[0], pdeck)
    mode = "opt2d3" if calc_kind(seq[i]) == "d3" else "opt2d12"
    # a stand-in output has no real structure for SeeK-path: its BAND form shows the static table path
    answers, kp = split_kmesh(body.get("answers") or {}) if mode == "opt2d12" else (body.get("answers") or {}, {})
    res = b.setup({"answers": answers, "patches": {**(body.get("patches") or {}), **kp},
                   "json_text": body.get("json_text"), "preview": preview_deck, "stand_in_kpath": mode == "opt2d3"},
                  mode, calc, f"s{i}", out_path,
                  f"projected/{out_path.name}", ("wf-standin", hashlib.sha256(pdeck.encode()).hexdigest(), mode))
    if mode == "opt2d3" and calc == "TRANSPORT":
        res["stand_in_fermi_note"] = STAND_IN_FERMI_NOTE.format(STAND_IN_FERMI_AU * HARTREE_TO_EV)
    if mode == "opt2d3" and calc == "BAND":
        # what the CLI says about the stand-in's missing structure is the stand-in's, not the user's to act on
        res["problems"] = [p for p in res.get("problems") or []
                           if not STAND_IN_KPATH_NOISE.search(str(p.get("message")))]
        res["static_kpath_note"] = ("The path in this deck is the static table path of the space group: the real step "
                                    "computes SeeK-path's path (the default) from the finished output.")
    res.update(step=key, parent=pkey, material_id=inp.materials[0], standin=True,
               editable=list(LATER_FIELDS.get(calc, METHOD_FIELDS)) if mode == "opt2d12" else None,
               soc_group=soc_group(inp, seq, i))
    if mode == "opt2d12":
        maxcycle_hint(res, calc)
        spin_hint(res, (body.get("patches") or {}).get("soc") is True)
        kmesh_hint(res, pdeck)
        g = res.get("soc_group") or {}
        if g.get("applies"):
            # inherited basis (the parent's) and what a SOC deck of this step is written with by default
            g["basis"] = soc_basis_default(inp, *b.step_basis(int(pkey.rsplit("_", 1)[1]) - 1))
            g["functionals"] = soc_functionals()
            if (body.get("patches") or {}).get("soc") is True:
                # this form's answers and SOC choices, over the draft's
                d2 = copy.deepcopy(draft)
                st2 = d2.setdefault("steps", {}).setdefault(key, {})
                st2["answers"] = dict(body.get("answers") or {})
                st2["soc"] = {k: v for k, v in (body.get("patches") or {}).items() if k in SOC_KEYS}
                soc_problems(res, Builder(st, inp, d2, work, ctx, cache_get, cache_put), i)
    if calc == "TRANSPORT":
        from mace.workflow.tau_step import tau_before
        tau = tau_before(seq, seq[i])
        rt = ((draft.get("steps") or {}).get(key) or {}).get("relaxation_time")
        res["relaxation"] = {"value": "auto" if tau else (rt if rt not in (None, "") else "default"),
                             "tau_step": f"{tau}_{seq.index(tau) + 1}" if tau else None, "default_fs": 10}
    return res


def soc_problems(res: Dict[str, Any], b: "Builder", i: int) -> None:
    """A SOC step's form: the CLI's refusal of an internal basis or of a functional the 2c-SCF does not take,
    said in the form's words (the row to change) instead of the settings-file keys the CLI names."""
    probs = [p for p in res.get("problems") or []
             if not (p.get("field") == "soc" and "internal basis-set library (BASISSET)" in str(p.get("message")))]
    why = b.soc_basis_problem(i)
    if why:
        probs.insert(0, {"severity": "error", "source": "Workflow", "field": "soc", "message": why.split(": ", 1)[1]})
    fn = next((f for f in res.get("form") or [] if f.get("id") == "functional"), None)
    name = re.sub(r"-D3$", "", str((fn or {}).get("value") or ""), flags=re.I)
    if name and name.upper() not in {x.upper() for x in soc_functionals()}:
        probs.insert(0, {"severity": "error", "source": "Workflow", "field": "soc",
                         "message": f"{name} has no two-component form in CRYSTAL23 (manual ch. 6): pick one of "
                                    f"{', '.join(soc_functionals())} in Functional (the -D3 forms of BLYP, PBE, "
                                    f"B3LYP, PBE0 and PW1PW too)"})
    res["problems"] = probs


def maxcycle_hint(res: Dict[str, Any], calc: str) -> None:
    """A FREQ step's SCF max cycles row says what the deck gets when it is left inherited (the parent's value,
    at least the floor; d12_constants.tight_scf_maxcycle) and that a value set here is written as given."""
    from mace.workflow.spec import _d12_constants
    C = _d12_constants()
    if calc not in C.TIGHT_SCF_CALC_TYPES:
        return
    for f in res.get("form") or []:
        if f.get("id") == "maxcycle":
            f["hint"] = (f"inherited: the parent's value, at least {C.TIGHT_SCF_MIN_MAXCYCLE} for {calc}'s tighter "
                         f"TOLDEE; a value set here is written as given")
            try:
                n = int(str(f.get("value")))
            except (TypeError, ValueError):
                n = None
            if f.get("answered") and n is not None and n < C.TIGHT_SCF_MIN_MAXCYCLE:
                f["note"] = (f"{n} is below the {C.TIGHT_SCF_MIN_MAXCYCLE} {C.a_calc(calc)} deck otherwise gets: "
                             f"its tighter TOLDEE may need more cycles")


def spin_hint(res: Dict[str, Any], soc: bool) -> None:
    """A later step's Spin polarized row: inherited from the step before, and a change starts the step from a
    fresh SCF guess (CRYSTAL cannot restart across a spin change: d12_constants.SPIN_CHANGE_GUESS). A SOC
    step treats spin itself: the row stays its parent's."""
    for f in res.get("form") or []:
        if f.get("id") != "spin":
            continue
        if soc:
            res["editable"] = [e for e in res.get("editable") or [] if e != "spin"]
            f["hint"] = "a two-component (SOC) step treats spin itself; set it on the step it starts from"
            return
        f["hint"] = ("inherited from the step before; a change starts this step from a fresh SCF guess, since "
                     "CRYSTAL cannot restart from a wavefunction of the other spin treatment")
        parent = str(f.get("default")) == "Yes"
        now = str(f.get("value")) == "Yes"
        if f.get("answered") and now != parent:
            f["note"] = ("starts from a fresh guess: the parent's wavefunction has no spin" if now else
                         "starts from a fresh guess: the parent's wavefunction is spin-polarized")


def _mesh_factors(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """The shrinking factors of a SHRINK record as Setup shows it ("8 16", "0 16 / 8 8 1") or of a mesh answer
    ("8 16", "8 8 4"): one-line IS ISP is IS along every direction."""
    if not text:
        return None
    nums = re.findall(r"\d+", str(text).split("/")[-1])
    if "/" not in str(text) and len(nums) == 2:
        nums = nums[:1] * 3
    return tuple(int(n) for n in nums) if len(nums) == 3 else None


def kmesh_hint(res: Dict[str, Any], pdeck: str) -> None:
    """A later step's k-point mesh row: the parent step's SHRINK record as inherited (``default_label``, the
    deck's own record; ``default`` and an unchanged row's value in the form an answer takes), "auto" offered
    beside it, and what the step's own deck gets when that is not what was typed (auto; a mesh whose
    symmetry-equivalent directions differ, raised to agree, d12_constants.symmetric_k_points). The SCF still
    restarts from the parent's density matrix: GUESSP is kept across a k-point change."""
    from . import setup as gui_setup
    for f in res.get("form") or []:
        if f.get("id") != KMESH:
            continue
        lines = pdeck.splitlines()
        dim = lines[1].strip().upper() if len(lines) > 1 else "CRYSTAL"
        shrink = gui_setup._shrink_of(pdeck)
        f["label"] = "k-point mesh (SHRINK)"
        if not shrink or dim == "MOLECULE":
            res["editable"] = [e for e in res.get("editable") or [] if e != KMESH]
            f["hint"] = "a molecule has no k points"
            return
        short = shrink.split(" / ", 1)[1] if " / " in shrink else shrink
        f.update(default=short, default_label=f"SHRINK {shrink}", suggestions=["auto", short])
        if not f.get("answered"):
            f["value"] = short
        f["hint"] = ("inherited from the step before. IS ISP (8 16), per axis IS1 IS2 IS3 (8 8 4; a slab's third "
                     "factor is 1, a polymer's second and third), or auto: MACE's mesh from each structure's cell. "
                     "The SCF still restarts from the parent's density matrix (GUESSP)")
        written = gui_setup._shrink_of(res.get("deck") or "")
        if not f.get("answered") or not written:
            return
        typed = str(f.get("value") or "").strip()
        n = {"SLAB": 2, "POLYMER": 1}.get(dim, 3)
        want, got = _mesh_factors(typed), _mesh_factors(written)
        if typed.lower() == "auto":
            f["note"] = f"auto: SHRINK {written} for {res.get('material_id')}, from its cell; each structure gets its own"
        elif want and got and want[:n] != got[:n]:
            f["note"] = (f"written as SHRINK {written}: directions the symmetry makes equivalent need the same "
                         f"factor (CRYSTAL stops on SHRINK BREAKS SYMMETRY)")


def tau_form(inp: Inputs, draft: Dict[str, Any], seq: List[str], i: int, pkey: str, pdeck: str) -> Dict[str, Any]:
    """A TAU step's settings: its options (the defaults filled in) and how many
    strained single points each structure gets, from the parent deck's symmetry."""
    from mace.workflow import tau_step as T
    key = f"{seq[i]}_{i + 1}"
    raw = ((draft.get("steps") or {}).get(key) or {}).get("tau") or {}
    try:
        opts = T.check_options(raw, key)
    except T.TauOptionError as e:
        raise BuildError(str(e))
    out = {"mode": "tau", "step": key, "calc_type": seq[i], "parent": pkey, "material_id": inp.materials[0],
           "options": opts, "defaults": T.check_options({}), "form": [], "soc_group": None,
           "masses": [s for s in seq if base_type(s) in ("BAND", "DOSS")],
           "transport": next((f"{s}_{j + 1}" for j, s in enumerate(seq)
                              if base_type(s) == "TRANSPORT" and T.tau_before(seq, s) == seq[i]), None)}
    try:
        dim, axes, why = T.symmetry_axes(pdeck)
        if opts["axes"] != "auto":
            axes, why = list(opts["axes"]), "the axes chosen here"
        out.update(dimensionality=dim, axes=axes, axes_reason=why, n_runs=T.member_count(axes, opts["strains"]))
    except ValueError as e:
        out["refusal"] = f"No strained decks for {inp.materials[0]}: {e}. TRANSPORT then runs with " \
                         f"{opts['fallback_fs']:g} fs."
    return out


def check_module_settings(draft: Dict[str, Any], seq: List[str], i: int) -> None:
    """An ELASTIC/EOS/QHA/CPHF step's module block, checked as the CLI checks
    a spec's elastic_settings (before its form is run)."""
    key = f"{seq[i]}_{i + 1}"
    raw = ((draft.get("steps") or {}).get(key) or {}).get("module")
    if not raw:
        return
    if base_type(seq[i]) not in MODULE_STEP_TYPES:
        raise BuildError(f"{key}: these are an ELASTIC, EOS, QHA or CPHF step's settings")
    from mace.workflow.spec import module_settings_problem
    why = module_settings_problem(seq[i], raw)
    if why:
        raise BuildError(f"{key}: {why}")


def module_form(inp: Inputs, draft: Dict[str, Any], seq: List[str], i: int, pkey: str, pdeck: str, work: Path,
                ctx, cache_get, cache_put) -> Dict[str, Any]:
    """An ELASTIC, EOS, QHA or CPHF step's settings: its module block (the
    draft's ``module``: mace opt2d12's --elastic-settings, Calculation
    Setup's module fields), the modules it can be (mace/gui/mechanics_routes.py)
    and the deck mace opt2d12 writes with it from a stand-in of the parent
    (or the CLI's reason for writing none)."""
    from mace.workflow.spec import _elastic
    from . import mechanics_routes
    E = _elastic()
    key = f"{seq[i]}_{i + 1}"
    calc = base_type(seq[i])
    check_module_settings(draft, seq, i)
    raw = ((draft.get("steps") or {}).get(key) or {}).get("module") or {}
    meta = mechanics_routes.modes()
    own = calc in E.OWN_TYPES
    modes = [m for m in meta["modes"] if (m["mode"] == calc if own else m["mode"] in E.ELASTIC_MODES)]
    norm = E.normalize(raw, calc)
    out = {"mode": "module", "step": key, "calc_type": seq[i], "parent": pkey, "material_id": inp.materials[0],
           "settings": raw, "module": norm["mode"], "describe": E.describe(norm), "modes": modes,
           "defaults": meta["defaults"], "toldee": E.QHA_TOLDEE if calc == "QHA" else E.MODULE_TOLDEE,
           "form": [], "soc_group": None}
    ck = ("wf-module", hashlib.sha256(pdeck.encode()).hexdigest(), calc, json.dumps(raw, sort_keys=True),
          clirun.cli_hash())
    hit = cache_get(ck)
    if hit is None:
        out_path = stand_in(work / f"mstandin_{i}", inp.materials[0], pdeck)
        dest = out_path.parent / "deck"
        dest.mkdir(exist_ok=True)
        # as the engine writes the step's deck: the step's config file (its type and module
        # block), so the stand-in parent is not checked as the run a deck asked for directly is
        step_cfg = out_path.parent / "module_step.json"
        step_cfg.write_text(json.dumps({"calculation_type": calc, **({"elastic_settings": raw} if raw else {})}))
        argv = clirun.mace_argv("--no-banner", "opt2d12", "--out-file", str(out_path), "--config-file",
                                str(step_cfg), "--non-interactive", "--output-dir", str(dest))
        r = clirun.run(argv, out_path.parent, ctx, timeout=180)
        decks = sorted(dest.glob("*.d12"))
        refusal = None
        if r.returncode != 0 or not decks:
            m = NOT_WRITING_MODULE.search(r.log)
            refusal = m.group(1).strip() if m else refusal_line({"refused": r.log})
        hit = {"deck": decks[0].read_text(errors="replace") if decks and r.returncode == 0 else None,
               "refusal": refusal}
        cache_put(ck, hit, len(json.dumps(hit)))
    out.update(deck=hit["deck"], standin=True)
    if hit["refusal"]:
        out["refusal"] = f"No {calc} deck for {inp.materials[0]}: {hit['refusal']}"
    return out


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


STAND_IN_FERMI_NOTE = ("The chemical potentials and the transport distribution range in this deck are placed around a "
                       "stand-in Fermi level ({:.2f} eV): the real step places them around its parent's E_F (the SCF's, "
                       "else a BAND or DOSS run of it), and refuses the deck without one.")


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
    # a stand-in Fermi level: a TRANSPORT step's E_F-relative ranges (and a DOSS window) need one to be
    # written at all; the real step places them around its parent's (STAND_IN_FERMI_NOTE)
    from mace.workflow.dummy_file_creator import append_stand_in_fermi
    append_stand_in_fermi(out)
    c.create_dummy_f9(d / f"{material}.f9")
    return out


# ------------------------------------------------------------------ fingerprint
def active_rows(folder: Path, inp: Inputs) -> List[Dict[str, Any]]:
    """Unfinished steps (submitted, queued, running, held; pending with a
    job id) of these structures in this folder's databases."""
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
                               f"FROM calculations WHERE lower(status) IN ({', '.join('?' * len(sub.QUEUED_STATES))})",
                               sub.QUEUED_STATES).fetchall()
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
            # a finished output is not run again (its job script is never read: the decks run in step folders)
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
    builder = Builder(st, inp, draft, work / "forms", ctx, cache_get or (lambda k: None),
                      cache_put or (lambda k, v, n: None))
    ov = builder.step1_overrides()
    files = spec_files(inp, draft, folder / spec_rel, ov, builder.later_overrides())
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
    # a deck whose own output finished is not run again: the CLI reads that output, so the copy has it
    finished: List[Dict[str, str]] = []
    if inp.kind == "d12":
        from mace.workflow.manual_progress import finished_output
        for f in inp.files:
            out = finished_output(f, sub.deck_calc_type(f))
            if out is not None:
                finished.append({"deck": f.name, "out_rel": sub._rel(root, out)})
                (mirror / sub._rel(root, out)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, mirror / sub._rel(root, out))
                staged += out.stat().st_size
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
        if code != 0:
            why = next((ln.strip() for ln in (r2.log or "").splitlines() if re.search(r"\bError:", ln)), "")
            result["refused"] = re.sub(r"^\W*(\[?ERROR\]?\W*)?(Error:\s*)?", "", why) or None
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
                "material_id": job, "script_rel": rel,
                "calc_type": sub.deck_calc_type(deck) if deck.exists() else None, "script_text": _unmirror(stext, mirror, root),
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
            if any(base_type(str(x.get("calc_type"))) == "BAND" for x in result["projected"]):
                result["projected_notes"] = ["The BAND k-path shown is the static table path of the deck's space group: "
                                             "the real step computes SeeK-path's path from the finished output."]
            if any(base_type(str(x.get("calc_type"))) == "TAU" for x in result["projected"]):
                result.setdefault("projected_notes", []).append(
                    "The TAU step's strained decks are written from a stand-in SP. Stand-in outputs give no relaxation "
                    "time, so the TRANSPORT deck shown has CRYSTAL's 10 fs: the real step runs with τ from the finished "
                    "strained runs (or 10 fs, with the reason, when τ cannot be computed).")
            # a step the engine's deck writer refused (a SOC deck it cannot write): its reason, in its words
            for why in dict.fromkeys(m.group(1).strip() for m in NOT_WRITING.finditer(pr.stderr or "")):
                warnings.append(f"Projection: a later step's deck was not written: {why}")
            for why in dict.fromkeys(m.group(1).strip() for m in NOT_WRITING_MODULE.finditer(pr.stderr or "")):
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
    # decks start in the workflow's own step folders (workflow_outputs/<id>/step_NNN_<TYPE>/<material>/), so
    # nothing of the folder is overwritten; a deck whose own output finished is not run again
    result["finished"] = finished
    p = WfPreview(id="", created=time.time(), folder=folder, inputs=inp, spec_rel=spec_rel, files=files, fp=fp,
                  result=result, launch_shape=_shape(result["launch"] or []), scripts=scripts, needs={})
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
            from .registry import pid_alive      # os.kill(pid, 0) would end the process on Windows
            return not pid_alive(pid)            # another user's live process counts as alive
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
