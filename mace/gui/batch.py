"""Batch Setup (PLAN_submit T6-T8): many decks at once, through the CLI's
own batch routes, each follow-up keeping its own parent's settings.

1. **Scan** a folder: CIFs (new calculations), completed runs that need a
   follow-up (with eligibility: not already beside, in an earlier
   ``batch_*/batch.json``, in a database or next in a workflow plan; FREQ only
   from a converged OPT), or failed runs (with what can be done).
2. **Plan**: the template is a set of Setup answers (only what the user
   changed), applied to each file on its own through ``setup.run_setup``
   (each parent staged and asked separately, so its functional, basis and
   settings stay its own); an override changes one file only. The SOC
   keys of the CLI's settings file ("soc", "soc_ecp", ...; Setup's SOC
   section) come as ``template_patches`` (and ``override_patches`` per file),
   written into each file's options as Setup writes them.
3. **Write** ``batch_<calc>_<date>/``: ``options/`` (the CLI's saved answers,
   one file per CIF group or per parent, plus ``template.json``), ``inputs/``
   (cif2d12's CIF groups), ``decks/`` (what the commands write),
   ``commands.sh`` (every command, shell-quoted: ``bash commands.sh``
   re-creates the decks) and ``batch.json`` (the GUI's record). Nothing
   outside the batch folder is written; a replaced deck keeps a ``.bak``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import clirun
from . import setup as S
from . import tasks as gui_tasks
from .submit import SubmitConflict, SubmitError, path_ok

NAME_RE = re.compile(r"^[A-Za-z0-9._+-]{1,80}$")
MAX_FILES = 5000
MAX_DEPTH = 3
FOLLOWUPS = ("SP", "OPT", "FREQ", "BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "ORBITALS")
D3_TYPES = ("BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "ORBITALS")
BUCKET_TYPE = {"complete": "OPT", "completesp": "SP", "completefreq": "FREQ", "completeband": "BAND",
               "completedoss": "DOSS", "completetransport": "TRANSPORT", "completecharge_potential": "CHARGE+POTENTIAL",
               "completeorbitals": "ORBITALS", "completeproperties": "PROPERTIES", "completeelastic": "ELASTIC",
               "completeeos": "EOS", "completeqha": "QHA", "completecphf": "CPHF"}
ERROR_BUCKETS = ("too_many_scf", "memory", "quota", "time", "geometry_small_dist", "shrink_error", "linear_basis",
                 "scratch", "opt_unconverged", "stopped", "potential", "orbitals_no_files", "cube_empty", "unknown",
                 "ongoing")
FAILED_ROUTE = {
    "time": "Timed out: set up a restart from its last geometry with opt2d12 (OPT), or submit again with a longer walltime",
    "opt_unconverged": "Optimization did not converge: set up an OPT from its last geometry with opt2d12",
    "too_many_scf": "SCF did not converge: change FMIXING / smearing in Setup and set it up again",
    "memory": "Out of memory: submit again with more memory per core (mace submit -i)",
    "ongoing": "No end marker: still running, or stopped without a message",
    "orbitals_no_files": "ORBITALS ended without writing its Molden files: check the ORBITALS records of its .d3",
    "cube_empty": "Charge/potential run wrote an empty cube file (e.g. ECH3 from a two-component molecule)",
}


class BatchError(SubmitError):
    pass


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


# ------------------------------------------------------------------ scan
def bucket(p: Path) -> str:
    from .submit import out_bucket
    return out_bucket(p)


def _walk(folder: Path, pattern: str, recursive: bool) -> List[Path]:
    out: List[Path] = []

    def rec(d: Path, depth: int):
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        for p in entries:
            if len(out) >= MAX_FILES:
                return
            if p.name.startswith(".") or p.is_symlink():
                continue
            if p.is_dir():
                if recursive and depth < MAX_DEPTH and not p.name.startswith("batch_"):
                    rec(p, depth + 1)
            elif p.match(pattern):
                out.append(p)
    rec(folder, 0)
    return out


# what opt2d3 refuses for a molecule (CRYSTALOptToD3.molecule_refusal; CRYSTAL23 manual pp.309, 320)
PERIODIC_ONLY = {"BAND": "CRYSTAL computes no bands for 0D", "DOSS": "CRYSTAL computes no DOSS for 0D",
                 "TRANSPORT": "transport comes from the bands"}
MOLECULE_HEAD = 4 << 20          # the dimensionality banner comes before the basis set and the SCF


def is_molecule(p: Path) -> bool:
    """A MOLECULE run: CRYSTAL prints "MOLECULAR CALCULATION" near the top of its output."""
    try:
        with open(p, "rb") as fh:
            head = fh.read(MOLECULE_HEAD)
    except OSError:
        return False
    return b"MOLECULAR CALCULATION" in head


def followup_mode(calc: str) -> str:
    return "opt2d3" if calc in D3_TYPES else "opt2d12"


def followup_names(out: Path, calc: str) -> List[str]:
    """Glob patterns of the deck/output a follow-up of ``calc`` gets beside
    its parent (spike S4: opt2d12 writes <stem>_<calc>_<functional>_optimized.d12,
    opt2d3 writes <base>_<calc>.d3)."""
    if calc in D3_TYPES:
        base = S.d3_base(out.stem)
        c = calc.lower()
        return [f"{base}_{c}.d3", f"{base}_{c}.out", f"{base}_{c}[0-9]*.d3"]
    c = calc.lower()
    return [f"{out.stem}_{c}_*.d12", f"{out.stem}_{c}_*.out"]


def earlier_batches(folder: Path, root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """{parent rel: [{calc, batch, deck_rel, state}]} from batch_*/batch.json below ``folder``."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for bj in list(folder.glob("batch_*/batch.json")) + list(folder.glob("*/batch_*/batch.json"))[:200]:
        try:
            data = json.loads(bj.read_text())
        except (OSError, ValueError):
            continue
        for parent, deck in (data.get("parents") or {}).items():
            deck_p = root / deck
            state = "written" if deck_p.exists() else "missing"
            if deck_p.with_suffix(".out").exists():
                state = "completed" if bucket(deck_p.with_suffix(".out")).startswith("complete") else "has an output"
            elif (data.get("job_ids") or {}).get(deck):
                state = f"submitted: {data['job_ids'][deck]}"
            out.setdefault(parent, []).append({"calc": data.get("calc"), "batch": bj.parent.name, "deck_rel": deck,
                                               "state": state})
    return out


def db_children(folder: Path) -> Dict[str, List[Tuple[str, str]]]:
    """{parent .out realpath: [(calc_type, status)]} of child calculations in
    the folder's materials.db and its workflow context databases (read-only):
    a row whose output is the parent names its material; every other row of
    that material is a child."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    dbs = [folder / "materials.db"] + [x for c in folder.glob(".mace_context_*") for x in c.rglob("*.db")]
    for db in dbs:
        if not db.is_file():
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            rows = con.execute("SELECT material_id, calc_type, status, output_file, input_file FROM calculations").fetchall()
            con.close()
        except sqlite3.Error:
            continue
        by_mat: Dict[str, List[Tuple[str, str, str, str]]] = {}
        for mat, ct, stt, outf, inf in rows:
            by_mat.setdefault(mat, []).append((ct or "", stt or "", outf or "", inf or ""))
        for mat, calcs in by_mat.items():
            for ct, stt, outf, inf in calcs:
                for cand in (outf, str(Path(inf).with_suffix(".out")) if inf else ""):
                    if cand:
                        key = os.path.realpath(cand)
                        out.setdefault(key, [])
                        out[key] += [(c2, s2) for c2, s2, o2, i2 in calcs if (o2 or i2) and (o2 != outf or i2 != inf)]
    return out


def workflow_next(folder: Path) -> Dict[str, Tuple[str, str]]:
    """{material stem: (workflow id, next step)} from workflow plans under
    ``folder`` (read-only; no context is activated): the step after the last
    one with an output in the plan's folder."""
    out: Dict[str, Tuple[str, str]] = {}
    # not the CIF-conversion copy, workflow_plan_<id>_updated.json
    plans = [p for p in folder.glob("workflow_configs/workflow_plan_*.json")
             if not p.stem.endswith("_updated")]
    for plan in sorted(plans)[-20:]:
        try:
            data = json.loads(plan.read_text())
        except (OSError, ValueError):
            continue
        seq = data.get("workflow_sequence") or []
        wid = data.get("workflow_id") or plan.stem
        files = ((data.get("input_files") or {}).get("d12") or [])
        for f in files:
            stem = Path(f).stem
            if len(seq) > 1:
                out[stem] = (wid, seq[1])
    return out


def scan(root: Path, folder: Path, filt: str, followup: Optional[str], recursive: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if filt == "cif":
        for p in _walk(folder, "*.cif", recursive):
            ok = path_ok(p)
            rows.append({"rel": _rel(root, p), "kind": "cif", "eligible": ok,
                         "why_not": "" if ok else f"File names with spaces or shell characters cannot be submitted; rename {p.name}"})
        return {"rows": rows, "truncated": len(rows) >= MAX_FILES}
    if filt == "d12":
        # decks as their own parents (mace opt2d12 --d12-file): the CLI's own rule says which can be
        from CRYSTALOptToD12 import deck_parent_refusal
        for p in _walk(folder, "*.d12", recursive):
            why = "" if path_ok(p) else f"File names with spaces or shell characters cannot be submitted; rename {p.name}"
            if not why:
                try:
                    why = deck_parent_refusal(str(p)) or ""
                except OSError as exc:
                    why = f"cannot read {p.name}: {exc}"
            rows.append({"rel": _rel(root, p), "kind": "d12", "eligible": not why, "why_not": why})
        return {"rows": rows, "truncated": len(rows) >= MAX_FILES}
    outs = _walk(folder, "*.out", recursive)
    if filt == "followup":
        calc = (followup or "SP").upper()
        if calc not in FOLLOWUPS:
            raise BatchError(f"follow-up type is one of {', '.join(FOLLOWUPS)}")
        earlier = earlier_batches(folder, root)
        children = db_children(folder)
        nxt = workflow_next(folder)
        for p in outs:
            b = bucket(p)
            ptype = BUCKET_TYPE.get(b)
            if ptype is None or ptype not in ("OPT", "SP"):
                continue
            rel = _rel(root, p)
            row = {"rel": rel, "kind": "out", "bucket": b, "calc_type": ptype, "eligible": True, "why_not": "",
                   "followups_present": []}
            why = ""
            if not path_ok(p):
                why = f"File names with spaces or shell characters cannot be submitted; rename {p.name}"
            elif calc in PERIODIC_ONLY and is_molecule(p):
                why = f"{calc} needs a periodic structure; this output is a molecule ({PERIODIC_ONLY[calc]})"
            elif calc == "ORBITALS" and S.parent_refusals(p).get(calc):
                # a two-component (SOC) run: CRYSTAL23 crashes on ORBITALS (kchoice.SOC_REFUSAL)
                why = S.parent_refusals(p)[calc]
            elif calc == "FREQ" and ptype != "OPT":
                why = "FREQ needs a converged OPT geometry (imaginary modes otherwise)"
            elif calc in D3_TYPES and S.find_wavefunction(p) is None:
                why = f"No wavefunction beside {p.name} ({S.d3_base(p.stem)}_sp.f9 or fort.9)"
            if not why:
                for pat in followup_names(p, calc):
                    hit = sorted(p.parent.glob(pat))
                    if hit:
                        row["followups_present"].append({"type": calc, "where": _rel(root, hit[0])})
                        why = f"{calc} already beside it: {hit[0].name}"
                        break
            if not why:
                for e in earlier.get(rel, []):
                    if (e.get("calc") or "").upper() == calc:
                        why = f"In {e['batch']} ({e['state']})"
                        break
            if not why:
                for ct, stt in children.get(os.path.realpath(str(p)), []):
                    if (ct or "").upper().rstrip("0123456789") == calc and (stt or "").lower() != "failed":
                        from mace.workflow.spec import _a
                        why = f"The database has {_a(calc)} of this material ({stt})"
                        break
            if not why:
                w = nxt.get(p.stem)
                if w and w[1].rstrip("0123456789") == calc:
                    why = f"Workflow {w[0]} will run {calc} next"
            row["eligible"], row["why_not"] = not why, why
            rows.append(row)
        return {"rows": rows, "truncated": len(outs) >= MAX_FILES}
    if filt == "failed":
        for p in outs:
            b = bucket(p)
            if b not in ERROR_BUCKETS:
                continue
            tracked = bool(db_children(p.parent).get(os.path.realpath(str(p))) is not None
                           and (p.parent / "materials.db").is_file())
            db_rel = _rel(root, p.parent / "materials.db") if tracked else None
            rows.append({"rel": _rel(root, p), "kind": "out", "bucket": b, "eligible": False,
                         "why_not": FAILED_ROUTE.get(b, "Failed: see its .out"),
                         "tracked": tracked,
                         "recover": (f"mace recover --db {shlex.quote(db_rel)}" if tracked else None)})
        return {"rows": rows, "truncated": len(outs) >= MAX_FILES}
    raise BatchError("filter is cif, d12, followup or failed")


# ------------------------------------------------------------------ plan
class Batch:
    def __init__(self, bid: str, root_gen, folder: Path, files: List[Path], mode: str, calc: str):
        self.id, self.root_gen, self.folder, self.files, self.mode, self.calc = bid, root_gen, folder, files, mode, calc
        self.template: Dict[str, Any] = {}
        self.flags: Dict[str, Any] = {}         # Setup's charged-system switches (setup.charge_flags), on every command
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.template_patches: Dict[str, Any] = {}
        self.override_patches: Dict[str, Dict[str, Any]] = {}
        self.items: Dict[str, Dict[str, Any]] = {}          # rel -> plan
        self.representative: Optional[str] = None
        self.created = time.time()
        self.written: Optional[Dict[str, Any]] = None


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.items: Dict[str, Batch] = {}

    def get(self, bid: str, root_gen) -> Batch:
        with self.lock:
            b = self.items.get(str(bid or ""))
        if b is None:
            raise SubmitConflict("No such batch; plan it again")
        if b.root_gen != root_gen:
            raise SubmitConflict("The folder changed; reload")
        return b

    def put(self, b: Batch) -> None:
        with self.lock:
            if len(self.items) > 20:
                for k in sorted(self.items, key=lambda k: self.items[k].created)[:5]:
                    del self.items[k]
            self.items[b.id] = b


def _setup_run(root: Path, p: Path, mode: str, calc: str, answers: Dict[str, Any], work: Path, cache, ctx, flags=None,
               patches: Optional[Dict[str, Any]] = None):
    rel = _rel(root, p)
    req = S.Request({"answers": answers, "patches": dict(patches or {}), "preview": True, "flags": flags or {}})
    get, put = cache
    return S.run_setup(req, mode, calc, work, p, rel, S.source_key(mode, p), get, put, ctx, schema_info=S.schema())


def check_patches(patches: Any, what: str) -> Dict[str, Any]:
    """Batch patches: the SOC keys of the CLI's settings file (Setup's SOC
    section), each checked by the CLI's own checks (S.Request)."""
    if patches is None:
        return {}
    if not isinstance(patches, dict):
        raise BatchError(f"{what} is an object")
    bad = [k for k in patches if k not in S.SOC_KEYS]
    if bad:
        raise BatchError(f"{what}: {bad[0]} is not a setting Batch Setup writes by hand (it writes {', '.join(S.SOC_KEYS)})")
    try:
        S.Request({"patches": patches})
    except S.SetupError as exc:
        raise BatchError(str(exc))
    return dict(patches)


def plan(root: Path, b: Batch, template: Dict[str, Any], overrides: Dict[str, Dict[str, Any]], disk, cache, ctx,
         flags: Optional[Dict[str, Any]] = None, template_patches: Optional[Dict[str, Any]] = None,
         override_patches: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Each file through Setup with template + its override (answers, and
    the settings keys written by hand: the SOC keys) and the charged-system
    switches, plus its default deck (no answers, no patches, no switches)
    for the diff."""
    b.flags = dict(flags or {})
    b.template = OrderedDict((k, v) for k, v in (template or {}).items())
    b.overrides = {k: OrderedDict(v) for k, v in (overrides or {}).items() if isinstance(v, dict)}
    b.template_patches = OrderedDict((template_patches or {}).items())
    b.override_patches = {k: OrderedDict(v) for k, v in (override_patches or {}).items() if isinstance(v, dict)}
    out = []
    n = len(b.files)
    for i, p in enumerate(b.files):
        if ctx is not None:
            ctx.check()
            ctx.progress(i / max(n, 1), f"{p.name} ({i + 1} of {n})")
        rel = _rel(root, p)
        answers = OrderedDict(b.template)
        answers.update(b.overrides.get(rel, {}))
        patches = OrderedDict(b.template_patches)
        patches.update(b.override_patches.get(rel, {}))
        key = "batch-" + secrets.token_hex(6)
        work = disk.acquire(key)
        try:
            try:
                r = _setup_run(root, p, b.mode, b.calc, answers, work / "a", cache, ctx, b.flags, patches)
                d = _setup_run(root, p, b.mode, b.calc, {}, work / "d", cache, ctx) if answers or b.flags or patches else r
            except S.SetupError as exc:
                r, d = {"deck": None, "refused": str(exc), "json_text": None}, None
        finally:
            disk.release(key, remove=True)
        item = {"rel": rel, "json_text": r.get("json_text"), "deck": r.get("deck"), "deck_name": r.get("deck_name"),
                "default_deck": (d or {}).get("deck"), "f9": r.get("f9"),
                "status": "ok" if r.get("deck") else "refused",
                "message": (r.get("refused") or "").strip().splitlines()[-1] if not r.get("deck") and r.get("refused") else "",
                "override": bool(b.overrides.get(rel) or b.override_patches.get(rel))}
        soc = r.get("soc") or {}
        if not r.get("deck") and soc.get("refusal"):
            item["message"] = soc["refusal"]            # the CLI's reason, as it said it
        # the CLI's SOC warnings for this file (all-electron atoms, f elements, metals), as it said them
        item["soc_warnings"] = [x["message"] for x in (r.get("problems") or [])
                                if x.get("field") == "soc" and x.get("severity") == "warning"]
        item["deck_sha"] = hashlib.sha256((item["deck"] or "").encode()).hexdigest()[:16] if item["deck"] else None
        if item["deck"] and item["default_deck"] is not None:
            diff = list(difflib.unified_diff(item["default_deck"].splitlines(), item["deck"].splitlines(), lineterm="", n=0))
            item["changed_lines"] = sum(1 for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---")))
        b.items[rel] = item
        out.append(item)
    # cif2d12: files with byte-identical settings share one convert --batch run
    groups: Dict[str, int] = {}
    for it in out:
        if it["status"] != "ok":
            it["group"] = None
            continue
        if b.mode == "cif2d12":
            k = hashlib.sha256((it["json_text"] or "").encode()).hexdigest()
            it["group"] = groups.setdefault(k, len(groups) + 1)
        else:
            it["group"] = None
    b.representative = b.representative or (out[0]["rel"] if out else None)
    return {"batch_id": b.id, "mode": b.mode, "calc": b.calc,
            "files": [{k: v for k, v in it.items() if k not in ("json_text", "deck", "default_deck")} for it in out],
            "groups": len(groups), "refused": sum(1 for it in out if it["status"] != "ok")}


def deck_text(b: Batch, rel: str) -> Dict[str, Any]:
    it = b.items.get(rel)
    if it is None:
        raise BatchError(f"{rel} is not in this batch")
    return {"rel": rel, "deck_name": it["deck_name"], "deck": it["deck"], "json_text": it["json_text"],
            "status": it["status"], "message": it["message"]}


def deck_diff(b: Batch, rel: str, against: str) -> Dict[str, Any]:
    it = b.items.get(rel)
    if it is None or not it.get("deck"):
        raise BatchError(f"{rel} has no deck in this batch")
    if against == "template":
        other = b.items.get(b.representative or "")
        base, label = (other or {}).get("deck") or "", f"template file ({b.representative})"
    else:
        base, label = it.get("default_deck") or "", "default"
    diff = "\n".join(difflib.unified_diff(base.splitlines(), it["deck"].splitlines(), label, it["deck_name"] or rel,
                                          lineterm=""))
    return {"rel": rel, "against": against, "diff": diff or "(identical)"}


# ------------------------------------------------------------------ write
def default_name(calc: str, folder: Path) -> str:
    base = f"batch_{calc.replace('+', '_')}_{datetime.now().strftime('%Y-%m-%d')}"
    name, k = base, 2
    while (folder / name).exists():
        name, k = f"{base}_{k}", k + 1
    return name


def check_name(name: str) -> str:
    if not NAME_RE.fullmatch(name or "") or name.startswith(".") or name in (".", ".."):
        raise BatchError("The batch folder name: 1 to 80 letters, digits, . _ + - (not starting with a dot)")
    return name


def move_new_files(stage: Path, out_dir: Path, st, prefix: str = "",
                   soc_f9: Optional[Path] = None) -> Tuple[List[Path], List[str]]:
    """Move what a CLI run wrote into ``stage`` to ``out_dir`` (Write decks
    now's rules: no symlink followed, a replaced deck keeps a .bak). A
    ``prefix`` (a same-name parent's subfolder) goes in front of each name.
    ``soc_f9``: the parent's .f9 a SOC restart copied as <deck>.f20, which
    never replaces a different .f20."""
    from . import saving
    made = sorted(x for x in stage.iterdir() if x.is_file() and not x.is_symlink())
    targets = [(x, out_dir / (prefix + x.name)) for x in made]
    for src, target in targets:
        # the opt2d3 wavefunction copy may be written as Write decks now writes it: when the
        # target is absent, or holds the same copy (same size and mtime as the copy just made)
        st.check_write(target, allow=("wavefunction",))
        if target.is_dir():
            raise PermissionError(f"{st.rel_new(target)} is a folder; not overwritten")
        if target.suffix.lower() in (".f9", ".f98") and target.exists():
            a, c = target.stat(), src.stat()
            if a.st_size != c.st_size or int(a.st_mtime) != int(c.st_mtime):
                raise saving.ProtectedOutput(f"{target.name} holds a finished run's wavefunction; MACE never writes "
                                             "over it. Write the batch into another folder, or move it first.")
        why = saving.f20_refusal(target, soc_f9) if soc_f9 is not None else None
        if why:
            raise saving.ProtectedOutput(why)
    written, backups = [], []
    for src, target in targets:
        deck = target.suffix.lower() in (".d12", ".d3")
        if target.exists():
            if deck:
                bak = saving.keep_backup(target)
                if bak["created"]:
                    backups.append(st.rel_new(bak["path"]))
            shutil.copymode(target, src)
        os.replace(src, target)
        written.append(target)
    return written, backups


def same_name_prefixes(folder: Path, root: Path, items: List[Dict[str, Any]]) -> Dict[str, str]:
    """rel -> prefix for parents whose file names repeat (``A/mat.out`` and
    ``B/mat.out``): the subfolder they sit in, so their settings files and
    decks do not overwrite each other in the batch's one options/ and decks/
    folder. Parents with a name of their own keep their names ("")."""
    by: Dict[str, List[str]] = {}
    for it in items:
        by.setdefault(Path(it["rel"]).stem, []).append(it["rel"])
    out = {it["rel"]: "" for it in items}
    for rels in by.values():
        if len(rels) < 2:
            continue
        for rel in rels:
            d = (root / rel).parent
            try:
                parts = d.relative_to(folder).parts
            except ValueError:
                parts = Path(rel).parent.parts
            out[rel] = "_".join(parts) + "_" if parts else ""
    return out


def check_unique(names: Dict[str, List[str]]) -> None:
    """Refuse (before anything is written) when two parents would still
    write the same file."""
    clash = {n: rels for n, rels in names.items() if len(rels) > 1}
    if clash:
        n, rels = sorted(clash.items())[0]
        raise BatchError(f"{' and '.join(rels)} would write the same name ({n}) in the batch folder; "
                         "rename one of them, or set them up in separate batches")


def write(st, b: Batch, name: str, overwrite: bool, ctx=None) -> Dict[str, Any]:
    root = st.root
    name = check_name(name or default_name(b.calc, b.folder))
    B = b.folder / name
    st.check_write(B / "x")
    if B.is_symlink():
        raise PermissionError(f"{st.rel_new(B)} is a symbolic link; nothing written")
    if B.exists() and not overwrite:
        raise SubmitConflict(f"{st.rel_new(B)} already exists (tick Overwrite to write into it)")
    ok = [it for it in b.items.values() if it["status"] == "ok"]
    if not ok:
        raise BatchError("No file of this batch has a deck to write")
    pre = same_name_prefixes(b.folder, root, ok)
    seen: Dict[str, List[str]] = {}
    for it in ok:
        if it.get("deck_name"):
            seen.setdefault("decks/" + pre[it["rel"]] + it["deck_name"], []).append(it["rel"])
        if b.mode == "cif2d12":
            seen.setdefault(f"inputs/g{it['group']}/" + pre[it["rel"]] + Path(it["rel"]).name, []).append(it["rel"])
        else:
            seen.setdefault("options/" + S.json_name(b.mode, b.calc, pre[it["rel"]] + Path(it["rel"]).stem),
                            []).append(it["rel"])
    check_unique(seen)
    from . import saving
    # the parent .f9 a SOC restart (soc_guess) copies as <deck>.f20
    soc_f9 = {it["rel"]: saving.soc_restart_matrix(it["json_text"] or "", root / it["rel"]) for it in ok} \
        if b.mode == "opt2d12" else {}
    soc_bytes = sum(f.stat().st_size for f in soc_f9.values() if f is not None)
    if b.mode == "opt2d3" or soc_bytes:
        need = sum((it.get("f9") or {}).get("bytes", 0) for it in ok) + soc_bytes + (1 << 30)
        free = shutil.disk_usage(b.folder).free
        if free < need:
            raise BatchError(f"The wavefunction copies need {need / 2**30:.1f} GB (plus 1 GB); "
                             f"{st.rel_new(b.folder) or '.'} has {free / 2**30:.1f} GB free")
    # an existing batch folder (Overwrite) may hold symbolic links: every path
    # written below is checked first, so nothing is written through one
    targets = [B / "options", B / "decks", B / "options" / "template.json", B / "commands.sh", B / "batch.json"]
    if b.mode == "cif2d12":
        for it in ok:
            targets += [B / "inputs", B / "inputs" / f"g{it['group']}",
                        B / "inputs" / f"g{it['group']}" / (pre[it["rel"]] + Path(it["rel"]).name),
                        B / "options" / f"g{it['group']}_options.json"]
    else:
        targets += [B / "options" / S.json_name(b.mode, b.calc, pre[it["rel"]] + Path(it["rel"]).stem) for it in ok]
    for t in targets:
        st.check_write(t)
    for sub_ in ("options", "decks"):
        (B / sub_).mkdir(parents=True, exist_ok=True)
    cwd = b.folder
    rel_b = _rel(cwd, B) if B != cwd else "."
    cmds: List[List[str]] = []
    cmd_pre: List[str] = []                 # follow-ups: the prefix each command's files get
    cmd_f9: List[Optional[Path]] = []      # follow-ups: the .f9 a SOC restart copies as <deck>.f20
    written_opts: List[str] = []
    parents: Dict[str, str] = {}
    calc = b.calc
    if b.mode == "cif2d12":
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for it in ok:
            groups.setdefault(it["group"], []).append(it)
        for g, items in sorted(groups.items()):
            inp = B / "inputs" / f"g{g}"
            inp.mkdir(parents=True, exist_ok=True)
            for it in items:
                shutil.copy2(root / it["rel"], inp / (pre[it["rel"]] + Path(it["rel"]).name))
            opt = B / "options" / f"g{g}_options.json"
            opt.write_text(items[0]["json_text"])
            written_opts.append(_rel(root, opt))
            cmds.append(["convert", "--batch", "--options_file", f"{rel_b}/options/{opt.name}", "--cif_dir",
                         f"{rel_b}/inputs/g{g}", "--output_dir", f"{rel_b}/decks"])
    else:
        for it in ok:
            p = root / it["rel"]
            jn = S.json_name(b.mode, calc, pre[it["rel"]] + p.stem)
            opt = B / "options" / jn
            opt.write_text(it["json_text"])
            written_opts.append(_rel(root, opt))
            prel = os.path.relpath(p, cwd)
            cmd_pre.append(pre[it["rel"]])
            cmd_f9.append(soc_f9.get(it["rel"]))
            if b.mode == "opt2d12":
                cmds.append(["opt2d12", S.source_flag(prel), prel, "--config-file", f"{rel_b}/options/{jn}", "--yes",
                             "--output-dir", f"{rel_b}/decks"])
            else:
                cmds.append(["opt2d3", "-i", prel, "-t", calc, "--config-file", f"{rel_b}/options/{jn}",
                             "--output-dir", f"{rel_b}/decks"])
    charge = S.flag_args(b.flags, b.mode)
    cmds = [c + charge for c in cmds]
    rep = b.items.get(b.representative or "") or ok[0]
    (B / "options" / "template.json").write_text(rep["json_text"])
    # run each command into a private stage inside decks/, then move the files in
    decks_dir = B / "decks"
    written, backups, logs, codes = [], [], [], []
    renames: List[List[Tuple[str, str]]] = [[] for _ in cmds]
    for i, args in enumerate(cmds):
        if ctx is not None:
            ctx.check()
            ctx.progress(i / max(len(cmds), 1), "mace " + " ".join(args[:1]) + f" ({i + 1} of {len(cmds)})")
        stage = Path(tempfile.mkdtemp(prefix=".mace-decks-", dir=str(decks_dir)))
        try:
            flag = "--output_dir" if args[0] == "convert" else "--output-dir"
            run_args = list(args)
            run_args[run_args.index(flag) + 1] = str(stage)
            r = clirun.run(clirun.mace_argv("--no-banner", *run_args), cwd, ctx, timeout=900)
            logs.append(r.log.replace(str(stage), f"{rel_b}/decks"))
            codes.append(r.returncode)
            prefix = cmd_pre[i] if i < len(cmd_pre) else ""
            if prefix:
                renames[i] = [(x.name, prefix + x.name) for x in sorted(stage.iterdir())
                              if x.is_file() and not x.is_symlink()]
            w, bk = move_new_files(stage, decks_dir, st, prefix, cmd_f9[i] if i < len(cmd_f9) else None)
            written += w
            backups += bk
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    # parent -> deck map (by name: the CLI's names, read back)
    deck_names = {x.name for x in written if x.suffix in (".d12", ".d3")}
    for it in ok:
        dn = pre[it["rel"]] + (it.get("deck_name") or "")
        if it.get("deck_name") and dn in deck_names:
            parents[it["rel"]] = _rel(root, decks_dir / dn)
    text = ["#!/bin/bash", f"# Re-create these decks with: bash commands.sh (run it in {st.rel_new(cwd) or 'the served folder'})",
            "# MACE Batch Setup: one command per line, as typed at the command line.", "set -e"]
    for args, ren in zip(cmds, renames):
        text.append("mace " + " ".join(shlex.quote(a) for a in args))
        # a same-name parent's files carry its subfolder's name (as Write did)
        text += [f"mv {shlex.quote(f'{rel_b}/decks/{a}')} {shlex.quote(f'{rel_b}/decks/{z}')}" for a, z in ren]
    (B / "commands.sh").write_text("\n".join(text) + "\n")
    os.chmod(B / "commands.sh", 0o644)
    from mace import __version__
    record = {"version": 1, "mace_version": __version__, "created_at": datetime.now().isoformat(timespec="seconds"),
              "folder": st.rel_new(cwd), "calc": calc, "mode": b.mode, "selection": [it["rel"] for it in b.items.values()],
              "template_answers": b.template, "flags": b.flags, "overrides": b.overrides,
              "template_patches": b.template_patches, "override_patches": b.override_patches, "parents": parents,
              "representative": b.representative, "commands": [" ".join(shlex.quote(a) for a in c) for c in cmds],
              "job_ids": {}}
    (B / "batch.json").write_text(json.dumps(record, indent=1) + "\n")
    out = {"folder_rel": st.rel_new(B), "decks_rel": st.rel_new(decks_dir),
           "written": sorted(st.rel_new(x) for x in written), "backups": backups,
           "commands_rel": st.rel_new(B / "commands.sh"), "batch_json_rel": st.rel_new(B / "batch.json"),
           "template_options_rel": st.rel_new(B / "options" / "template.json"),
           "commands": text[4:], "code": max(codes) if codes else 0,
           "log": "\n".join(logs).replace(str(root) + os.sep, "")[-6000:],
           "one_file_per_parent": None if b.mode == "cif2d12" else (
               "opt2d12 and opt2d3 read one settings file per parent so each follow-up keeps its own parent's "
               "functional, basis and settings; template.json is the shared part"),
           "parents": parents}
    b.written = out
    return out


def record_jobs(root: Path, decks_rel: str, job_ids: List[str]) -> None:
    """After a submit of a batch's decks: its batch.json names the job ids (best effort)."""
    bj = (root / decks_rel).parent / "batch.json"
    if bj.is_symlink() or bj.parent.is_symlink():
        return
    try:
        data = json.loads(bj.read_text())
    except (OSError, ValueError):
        return
    decks = sorted(data.get("parents", {}).values())
    data["job_ids"] = {d: j for d, j in zip(decks, job_ids)} if len(decks) == len(job_ids) else {"all": job_ids}
    try:
        bj.write_text(json.dumps(data, indent=1) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ routes
def _store(st) -> Store:
    s = getattr(st, "batches", None)
    if s is None:
        s = st.batches = Store()
    return s


def _folder(st, rel) -> Path:
    p = st.resolve(rel or ".", must_exist=True) if rel not in (None, "", ".") else st.root
    if not p.is_dir():
        raise BatchError("Batch Setup works on a folder")
    return p


def post_route(handler, route: str, body: Dict[str, Any]):
    st = handler.state
    from .server import cli_task
    gen = getattr(st, "root_gen", 0)
    if route == "batch/scan":
        folder = _folder(st, body.get("folder"))
        filt = str(body.get("filter") or "cif")
        fu = body.get("followup_type")
        rec = bool(body.get("recursive"))
        t = cli_task(st, "batch-scan", f"Batch Setup: scan {st.rel_new(folder) or '.'}",
                     lambda ctx: {**scan(st.root, folder, filt, fu, rec), "folder": st.rel_new(folder)},
                     client=handler.headers.get("X-Mace-Client"), supersede=("batch-scan", str(body.get("doc") or "")))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    if route == "batch/stutsc":
        # the files' elements whose STUTSC the CLI refuses (soc_ecp.py): the SOC section grays it out
        files = body.get("files") or []
        if not isinstance(files, list) or len(files) > MAX_FILES or not all(isinstance(f, str) for f in files):
            raise BatchError("files is a list of the folder's files")
        paths = [st.resolve(f) for f in files]
        t = cli_task(st, "batch-stutsc", f"Batch Setup: elements of {len(paths)} file(s)",
                     lambda ctx: S.stutsc_refusals(paths, name=st.rel),
                     client=handler.headers.get("X-Mace-Client"), supersede=("batch-stutsc", str(body.get("doc") or "")))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    if route == "batch/plan":
        folder = _folder(st, body.get("folder"))
        files = body.get("files") or []
        if not isinstance(files, list) or not files or len(files) > MAX_FILES:
            raise BatchError("files is a list of the folder's files")
        paths = [st.resolve(str(f)) for f in files]
        for p in paths:
            if not path_ok(p):
                raise BatchError(f"File names with spaces or shell characters cannot be submitted; rename {p.name}")
        calc = str(body.get("calc") or "").upper()
        if all(p.suffix.lower() == ".cif" for p in paths):
            mode = "cif2d12"
            calc = calc or "OPT"
        elif all(p.suffix.lower() in S.OUT_SUFFIXES for p in paths):
            calc = calc or "SP"
            mode = followup_mode(calc)
        elif all(S.is_deck(p.name) for p in paths):
            calc = calc or "SP"
            if calc in S.CALC_TYPES["opt2d3"]:
                raise BatchError(S.deck_properties_refusal(paths[0].name, calc))
            mode = "opt2d12"
        else:
            raise BatchError("A batch is all CIFs, all .out files or all .d12 decks")
        S.check_calc(mode, calc)
        bid = str(body.get("batch_id") or "")
        store = _store(st)
        if bid:
            b = store.get(bid, gen)
            b.files, b.mode, b.calc, b.folder = paths, mode, calc, folder
        else:
            b = Batch("b" + secrets.token_hex(6), gen, folder, paths, mode, calc)
        rep = body.get("representative")
        if rep:
            b.representative = st.rel(st.resolve(str(rep)))
        store.put(b)
        template = body.get("template_answers") or {}
        overrides = body.get("overrides") or {}
        if not isinstance(template, dict) or not isinstance(overrides, dict):
            raise BatchError("template_answers and overrides are objects")
        flags = S.charge_flags(body.get("flags"), S.BATCH_CHARGE_FLAGS)
        tpatches = check_patches(body.get("template_patches"), "template_patches")
        opatches = body.get("override_patches") or {}
        if not isinstance(opatches, dict):
            raise BatchError("override_patches is an object")
        opatches = {str(k): check_patches(v, f"override_patches[{k}]") for k, v in opatches.items()}

        def get(k):
            return st.parse_cache.get(("setup",) + (json.dumps(k, default=str),))

        def put(k, v, n):
            st.parse_cache.put(("setup",) + (json.dumps(k, default=str),), v, n)
            st.ledger.enforce()
        t = cli_task(st, "batch-plan", f"Batch Setup: {len(paths)} {calc} deck(s)",
                     lambda ctx: plan(st.root, b, template, overrides, st.disk, (get, put), ctx, flags, tpatches, opatches),
                     client=handler.headers.get("X-Mace-Client"), supersede=("batch-plan", b.id))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    if route == "batch/write":
        b = _store(st).get(str(body.get("batch_id") or ""), gen)
        name = str(body.get("name") or "") or default_name(b.calc, b.folder)
        check_name(name)
        B = b.folder / name
        st.check_write(B / "x")
        if B.exists() and not body.get("overwrite"):
            raise SubmitConflict(f"{st.rel_new(B)} already exists (tick Overwrite to write into it)")
        t = cli_task(st, "batch-write", f"Batch Setup: write {name}",
                     lambda ctx: write(st, b, name, bool(body.get("overwrite")), ctx),
                     client=handler.headers.get("X-Mace-Client"))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    raise BatchError(f"unknown batch route {route}")


def get_route(handler, route: str, query):
    st = handler.state
    q = lambda k, d=None: (query.get(k) or [d])[0]
    b = _store(st).get(q("batch_id", ""), getattr(st, "root_gen", 0))
    if route == "batch/deck":
        return handler._json(deck_text(b, q("rel", "")))
    if route == "batch/diff":
        against = q("against", "default")
        if against not in ("default", "template"):
            raise BatchError("against is default or template")
        return handler._json(deck_diff(b, q("rel", ""), against))
    raise BatchError(f"unknown batch route {route}")
