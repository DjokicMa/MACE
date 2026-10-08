"""``mace preflight`` on the previewed deck (plan T6), as a cancelable task.

The deck Setup previewed is written into the task's own folder and checked
with the real ``mace preflight`` (``-j 1``: one deck, one CPU slot, spec
section 9). Where the serial ``crystal`` is found the fast way (an explicit
path, $MACE_CRYSTAL_BIN, the site's [crystal] bin_dir and only it when set,
$EBROOTCRYSTAL/bin, PATH), CRYSTAL reads the deck
with TESTPDIM; otherwise the structure check runs alone (``--static-only``)
and Problems says so: the CLI's own "module load" probe can take a minute
on a workstation that has no CRYSTAL, so the GUI does not start it.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import clirun

_ERROR_LINE = re.compile(r"ERROR\s+\*{4}")


def fast_crystal(explicit: Optional[str] = None) -> Optional[str]:
    """The serial crystal binary, by the CLI's own search order minus the module probe."""
    def usable(p):
        return p if p and os.path.isfile(p) and os.access(p, os.X_OK) else None
    if explicit:
        return usable(os.path.expanduser(explicit))
    from mace.submission.preflight import _site_bin_dir_crystal
    if usable(os.environ.get("MACE_CRYSTAL_BIN")):
        return os.environ["MACE_CRYSTAL_BIN"]
    site_crystal = _site_bin_dir_crystal()
    if site_crystal is not None:
        # [crystal] bin_dir is the job scripts' CRYSTAL: its crystal or none
        return usable(site_crystal)
    for cand in (os.path.join(os.environ["EBROOTCRYSTAL"], "bin", "crystal") if os.environ.get("EBROOTCRYSTAL") else None,
                 shutil.which("crystal")):
        if usable(cand):
            return cand
    return None


def run(deck_name: str, deck_text: str, work: Path, ctx: Any = None, crystal_bin: Optional[str] = None) -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,200}\.d12", deck_name or ""):
        raise ValueError("preflight checks a .d12 deck")
    d = work / "decks"
    d.mkdir(parents=True, exist_ok=True)
    (d / deck_name).write_text(deck_text)
    keep = work / "keep"
    binary = fast_crystal(crystal_bin)
    args = ["--no-banner", "preflight", str(d / deck_name), "-j", "1", "--timeout", "120", "--keep", str(keep)]
    args += ["--crystal-bin", binary] if binary else ["--static-only"]
    shown = "mace preflight " + deck_name + (" --static-only" if not binary else "")
    if ctx is not None:
        ctx.progress(0.2, f"Running {shown}")
    r = clirun.run(clirun.mace_argv(*args), work, ctx, timeout=300)
    log = r.log.replace(str(d) + os.sep, "").replace(str(work) + os.sep, "")
    status = {0: "static" if not binary else "accepted", 1: "rejected"}.get(r.returncode, "not_checked")
    if binary and r.returncode == 0 and "structure-only" in log and " 1 passed" not in log:
        status = "static"
    excerpt, reason = "", ""
    out = next(keep.rglob("out.txt"), None) if keep.is_dir() else None
    if out is not None:
        text = out.read_text(errors="replace")
        err = next((ln.strip() for ln in text.splitlines() if _ERROR_LINE.search(ln)), "")
        reason = " ".join(err.split())
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        excerpt = "\n".join(lines[-20:])
    if not reason:
        m = re.search(rf"{re.escape(deck_name)}\s+(.+)", log)
        reason = m.group(1).strip() if m else ""
    problems: List[Dict[str, Any]] = []
    site_problem = None
    if not binary and not crystal_bin:
        from mace.submission.preflight import site_bin_dir_problem
        site_problem = site_bin_dir_problem()
    if status == "static" and site_problem:
        problems.append({"severity": "warning", "source": "Preflight",
                         "message": f"Ran the structure check only: {site_problem}."})
    elif status == "static":
        problems.append({"severity": "info", "source": "Preflight",
                         "message": "CRYSTAL not found on this machine; ran the structure check only. "
                                    f"Run Preflight on your cluster for the TESTPDIM check ({shown.replace(' --static-only', '')})."})
    elif status == "accepted":
        problems.append({"severity": "info", "source": "Preflight", "message": f"ACCEPTED: CRYSTAL read {deck_name} through the symmetry analysis (TESTPDIM)"})
    elif status == "rejected" and not binary:
        # --static-only: CRYSTAL never ran, so it is MACE's structure check that failed
        problems.append({"severity": "error", "source": "Preflight",
                         "message": f"Structure check failed (CRYSTAL not run): {reason or 'see the preflight log'}"})
    elif status == "rejected":
        problems.append({"severity": "error", "source": "Preflight", "message": f"REJECTED: {reason or 'CRYSTAL rejected the deck'}"})
    else:
        problems.append({"severity": "warning", "source": "Preflight", "message": f"Not checked: {reason or log.strip().splitlines()[-1] if log.strip() else 'no result'}"})
    return {"status": status, "reason": reason, "excerpt": excerpt[-3000:], "log": log[-4000:], "command": shown,
            "crystal": binary, "problems": problems, "code": r.returncode}
