"""Pause, resume and cancel a workflow: its control file.

``mace workflow --pause|--resume|--cancel ID`` writes
``workflow_configs/workflow_control_<id without "workflow_">.json`` next to
the workflow's plan. Every completion callback reads it after it has found
the plan (``WorkflowEngine``) and before it recovers a failed job
(``EnhancedCrystalQueueManager``):

``paused``
    completions and failures are recorded; no next step and no recovery job
    is submitted. The completed calculations stay unprocessed, so resuming
    picks them up.
``canceled``
    as paused, for good (``--resume`` lifts it too, on purpose).
``active`` (or no file)
    progression as the plan says.

The file never changes the plan: plans stay what was launched.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

CONTROL_VERSION = 1
STATES = ("active", "paused", "canceled")
HOLD_STATES = ("paused", "canceled")
# A workflow id as minted today (workflow_<YYYYmmdd_HHMMSS>_<4 hex>) or by
# 1.1.4 and earlier (no suffix).
WORKFLOW_ID_RE = re.compile(r"^workflow_\d{8}_\d{6}(?:_[0-9a-f]{4})?$")


def control_name(workflow_id: str) -> str:
    return f"workflow_control_{workflow_id[len('workflow_'):]}.json"


def control_path(base_dir: Union[str, Path], workflow_id: str) -> Path:
    return Path(base_dir) / "workflow_configs" / control_name(workflow_id)


def read_control(base_dir: Union[str, Path], workflow_id: str) -> Optional[Dict[str, Any]]:
    """The control file of ``workflow_id`` in ``base_dir``, or None."""
    return _read(control_path(base_dir, workflow_id))


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def state_of(control: Optional[Dict[str, Any]]) -> str:
    state = str((control or {}).get("state") or "active").strip().lower()
    return state if state in STATES else "active"


def write_control(base_dir: Union[str, Path], workflow_id: str, state: str,
                  note: str = "") -> Dict[str, Any]:
    """Set the workflow's state (atomic replace); returns the new file's content."""
    if state not in STATES:
        raise ValueError(f"unknown workflow state {state!r} (one of {', '.join(STATES)})")
    path = control_path(base_dir, workflow_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read(path) or {"version": CONTROL_VERSION, "workflow_id": workflow_id, "history": []}
    now = datetime.now().isoformat(timespec="seconds")
    data["state"] = state
    data["updated"] = now
    entry = {"state": state, "at": now}
    if note:
        entry["note"] = note
    data.setdefault("history", []).append(entry)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
    return data


def candidate_dirs(start_dirs: Iterable[Union[str, Path, None]], levels: int = 4) -> List[Path]:
    """``start_dirs`` and their parents (``levels`` up), the way the engine
    looks for a plan's ``workflow_configs/``, without repeats."""
    out: List[Path] = []
    for d in start_dirs:
        if not d:
            continue
        p = Path(d)
        for _ in range(levels + 1):
            if p not in out:
                out.append(p)
            if p.parent == p:
                break
            p = p.parent
    return out


def held_state(workflow_id: Optional[str], start_dirs: Iterable[Union[str, Path, None]]) -> Optional[str]:
    """'paused' or 'canceled' when the first control file of ``workflow_id``
    found from ``start_dirs`` holds the workflow, else None. Reading never
    fails: an unreadable file holds nothing."""
    if not workflow_id:
        return None
    name = control_name(str(workflow_id))
    for d in candidate_dirs(start_dirs):
        path = d / "workflow_configs" / name
        if path.is_file():
            state = state_of(_read(path))
            return state if state in HOLD_STATES else None
    return None
