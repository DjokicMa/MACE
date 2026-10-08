"""The SLURM commands a Submit preview sees (PLAN_submit T2), used ONLY in previews.

A preview runs the real ``mace submit`` in a staged copy of the folder with a
folder of these launchers first on PATH. They split SLURM's commands in two:

* **read-only queries pass through** to the real tools (absolute paths,
  resolved before this folder went on PATH, so a launcher can never call
  itself) with the same argv and environment, output unchanged: ``squeue``,
  ``sinfo``, ``sacct``, ``scontrol show ...``, ``sacctmgr show|list ...`` and
  ``sbatch --test-only ...``. Live cluster state (node types for
  ``--exclude-type``, MaxArraySize for the array split, the submit limit)
  therefore shapes the preview exactly as it shapes the real submit. Where a
  tool does not exist (a workstation), an empty cluster answers and the
  preview says so ("no_slurm" in the log);
* **every state change is intercepted** and recorded (argv, working folder,
  script text): ``sbatch`` (answered with a fake id from 900000000 up),
  ``scancel``, ``scontrol`` with any subcommand but ``show``, ``sacctmgr``
  with anything but ``show``/``list``. Nothing reaches SLURM.

Run as ``python slurm_preview_shim.py <config.json> <tool> args...``; the
config names the real tools and the log. Standard library only.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

TOOLS = ("sbatch", "scancel", "scontrol", "squeue", "sinfo", "sacct", "sacctmgr")
FIRST_ID = 900000000
EMPTY_CONFIG = "MaxArraySize            = 1001\nSchedulerParameters     = (null)\n"


def install(shim_dir: Path, real: Dict[str, Optional[str]], log: Path, python: str = sys.executable) -> Path:
    """Write the launchers (mode 0700 folder) and the config; returns the config path."""
    shim_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(shim_dir, 0o700)
    cfg = shim_dir / "shim.json"
    cfg.write_text(json.dumps({"real": real, "log": str(log), "counter": str(shim_dir / "counter")}))
    me = Path(__file__).resolve()
    for tool in TOOLS:
        p = shim_dir / tool
        p.write_text(f"#!/bin/sh\nexec '{python}' -I '{me}' '{cfg}' {tool} \"$@\"\n")
        os.chmod(p, 0o700)
    return cfg


def _record(cfg: dict, entry: dict) -> None:
    with open(cfg["log"], "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(entry) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def _next_id(cfg: dict) -> int:
    path = cfg["counter"]
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        n = int(raw) + 1 if raw.isdigit() else FIRST_ID
        fh.seek(0)
        fh.truncate()
        fh.write(str(n))
        fcntl.flock(fh, fcntl.LOCK_UN)
    return n


def _script_of(argv: List[str]) -> Optional[str]:
    """The script argument of an sbatch command line (first non-option)."""
    with_value = {"-t", "-a", "--array", "-J", "-o", "-e", "-A", "-p", "-N", "-n", "-c", "--dependency", "-d",
                  "-D", "--chdir", "-w", "-x", "--exclude", "--mem", "--qos", "-q"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-"):
            if a in with_value and i + 1 < len(argv):
                i += 1
        else:
            return a
        i += 1
    return None


def passes_through(tool: str, argv: List[str]) -> bool:
    """A read-only query (True) or a state change (False)."""
    if tool in ("squeue", "sinfo", "sacct"):
        return True
    words = [a for a in argv if not a.startswith("-")]
    if tool == "scontrol":
        return bool(words) and words[0] == "show"
    if tool == "sacctmgr":
        return bool(words) and words[0] in ("show", "list")
    if tool == "sbatch":
        return "--test-only" in argv
    return False


def _empty_answer(tool: str, argv: List[str]) -> int:
    words = [a for a in argv if not a.startswith("-")]
    if tool == "scontrol" and words[1:2] == ["config"]:
        sys.stdout.write(EMPTY_CONFIG)
    if tool == "sbatch":                       # --test-only without SLURM
        sys.stderr.write("sbatch: error: no SLURM on this machine\n")
        return 1
    return 0


def main(argv: List[str]) -> int:
    cfg_path, tool, args = argv[0], argv[1], argv[2:]
    cfg = json.loads(Path(cfg_path).read_text())
    entry = {"tool": tool, "argv": [tool] + args, "cwd": os.getcwd()}
    if passes_through(tool, args):
        real = (cfg.get("real") or {}).get(tool)
        entry["passed"] = True
        if not real:
            entry["no_slurm"] = True
            _record(cfg, entry)
            return _empty_answer(tool, args)
        _record(cfg, entry)
        sys.stdout.flush()
        os.execv(real, [real] + args)
    entry["intercepted"] = True
    if tool == "sbatch":
        script = _script_of(args)
        text = ""
        if script:
            try:
                text = Path(script).read_text(errors="replace")
            except OSError:
                text = ""
            entry["script"] = str(Path(script).resolve()) if script else None
        entry["script_text"] = text
        if not script or not text:
            _record(cfg, entry)
            sys.stderr.write(f"sbatch: error: Unable to open file {script}\n")
            return 1
        job = _next_id(cfg)
        entry["job_id"] = str(job)
        _record(cfg, entry)
        print(f"{job}" if "--parsable" in args else f"Submitted batch job {job}")
        return 0
    _record(cfg, entry)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
