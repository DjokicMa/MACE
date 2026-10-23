"""Projected later steps of a workflow, for a preview.

The GUI's workflow builder previews a workflow by running the real CLI in a
staged copy of the folder (``mace workflow --spec SPEC --write-plan``, then
the launch command), with SLURM's state-changing commands recorded instead
of run. That shows the first step exactly. The later steps are written by
the engine when each parent finishes, from the parent's real output, so
they cannot be shown exactly before anything ran.

This module projects them: in the staged copy, for one material, it gives
each finished-to-be step a stand-in output (``DummyFileCreator``, made from
the step's own deck, with a stand-in wavefunction), marks it completed and
lets the real engine (``WorkflowEngine.execute_workflow_step``) write and
"submit" the next steps, as a job's completion callback would. The decks
and job scripts it writes are what the engine writes for those settings;
the geometry and anything else read from the parent's output come from the
stand-in, so the GUI labels them as projected.

It changes the database (steps marked completed) and writes stand-in
outputs, so it refuses to run anywhere but in a preview copy: a folder that
has, or whose parent folders have, a ``.mace_preview_mirror`` file.

Run as ``python -m mace.workflow.projection --folder DIR --workflow-id ID
[--material M] [--json]``.
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MARKER = ".mace_preview_mirror"
MAX_STEPS = 40


class ProjectionError(Exception):
    """Nothing was projected; the message says why."""


def in_preview_copy(folder: Path) -> bool:
    folder = Path(folder).resolve()
    return any((p / MARKER).is_file() for p in (folder, *folder.parents))


def _stand_in(deck: Path) -> Path:
    """A stand-in output and wavefunction beside ``deck`` (as a finished job leaves them)."""
    from mace.workflow.dummy_file_creator import DummyFileCreator
    creator = DummyFileCreator()
    out = deck.with_suffix(".out")
    if deck.suffix.lower() == ".d12":
        from mace.workflow.dummy_file_creator import append_stand_in_fermi
        creator.create_dummy_out(out, creator.extract_d12_settings(deck))
        append_stand_in_fermi(out)            # E_F-relative ranges (TRANSPORT) are placed around it
        creator.create_dummy_f9(deck.with_suffix(".f9"))
    else:
        out.write_text(" PROPERTIES STAND-IN OUTPUT\n EEEEEEEEEE TERMINATION\n")
    return out


def project(folder: Path, workflow_id: str, material: Optional[str] = None) -> Dict[str, Any]:
    """Write the later steps of ``material`` (default: the first material
    that has a first-step row) in the preview copy ``folder``."""
    from mace.workflow import status
    folder = Path(folder).resolve()
    if not in_preview_copy(folder):
        raise ProjectionError(f"{folder} is not a preview copy (no {MARKER}); projection marks steps "
                              f"completed and runs only in a preview")
    plan = status.load_plan(folder, workflow_id)
    if plan is None:
        raise ProjectionError(f"no plan of {workflow_id} in {folder}/workflow_configs")
    db_path = status.database_of(folder, workflow_id, plan)
    if db_path is None or not Path(db_path).is_file():
        raise ProjectionError(f"{workflow_id} has no database yet (was its first step submitted?)")
    sequence = list(plan.get("workflow_sequence") or [])
    keys = {t: f"{t}_{i + 1}" for i, t in enumerate(sequence)}
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(db_path))
    # the launch's rows not finished yet: the first steps, and the next step of a deck whose own output had
    # finished already (its row is recorded completed and its next step was submitted at launch)
    first = [r for r in db.get_calculations_by_status() if r.get("status") != "completed"]
    # a material that starts at the first step shows the whole chain
    starts = {r["material_id"] for r in first if r.get("calc_type") == sequence[0]}
    mats = sorted({r["material_id"] for r in first}, key=lambda m: (m not in starts, m))
    if not mats:
        raise ProjectionError(f"{workflow_id} has no submitted step in its database")
    mid = material if material is not None else mats[0]
    if mid not in mats:
        raise ProjectionError(f"{mid} has no submitted step (materials: {', '.join(mats)})")
    os.environ["MACE_WORKFLOW_ID"] = workflow_id
    # A stand-in output has no structure to run SeeK-path on (its atoms are the deck's unique ones), so a
    # BAND step's SeeK-path k-path is projected as the static table path of the deck's space group; the
    # real step computes it from the finished output.
    os.environ.setdefault("MACE_STATIC_KPATH", "1")
    os.environ.setdefault("MACE_STAND_IN_KPATH", "1")    # seekpath installed: still not run on the stand-in
    if Path(db_path).parent.name.startswith(".mace_context_"):
        os.environ["MACE_CONTEXT_DIR"] = str(Path(db_path).parent)
    from mace.workflow.engine import WorkflowEngine
    engine = WorkflowEngine(str(db_path), str(folder))
    todo = [r for r in first if r["material_id"] == mid]
    parent_of: Dict[str, str] = {}
    steps: List[Dict[str, Any]] = []
    problems: List[str] = []
    seen = set()
    while todo and len(seen) < MAX_STEPS:
        row = todo.pop(0)
        if row["calc_id"] in seen:
            continue
        seen.add(row["calc_id"])
        deck = Path(row.get("input_file") or "")
        if not deck.is_file():
            problems.append(f"{row['calc_type']}: its deck {deck} is missing")
            continue
        try:
            out = _stand_in(deck)
        except Exception as exc:       # a deck the stand-in maker cannot read: stop this branch
            problems.append(f"{row['calc_type']}: no stand-in output ({exc})")
            continue
        db.update_calculation_status(row["calc_id"], "completed", output_file=str(out))
        new_ids = engine.execute_workflow_step(mid, row["calc_id"]) or []
        for cid in new_ids:
            new = db.get_calculation(cid)
            if not new:
                continue
            parent_of[cid] = keys.get(row["calc_type"], row["calc_type"])
            inp = Path(new.get("input_file") or "")
            sh = inp.with_suffix(".sh")
            steps.append({"step": keys.get(new["calc_type"], new["calc_type"]), "calc_type": new["calc_type"],
                          "material_id": mid, "calc_id": cid, "parent": parent_of[cid],
                          "input_file": str(inp), "script": str(sh) if sh.is_file() else None})
            todo.append(new)
    missing = [keys[t] for t in sequence[1:] if keys[t] not in {s["step"] for s in steps}]
    return {"ok": True, "workflow_id": workflow_id, "material_id": mid, "steps": steps,
            "not_projected": missing, "problems": problems}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mace.workflow.projection", description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--workflow-id", required=True)
    ap.add_argument("--material")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr if a.json else sys.stdout):
            res = project(Path(a.folder), a.workflow_id, a.material)
    except ProjectionError as exc:
        if a.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        for s in res["steps"]:
            print(f"{s['step']:<10} {s['material_id']}: {s['input_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
