"""``mace update --check --json``: the report MACE GUI's update notice
reads (one channel; the current and the new version, the CHANGELOG sections
in between, the refusal ``mace update`` would print word for word, local
changes with the git stash line, the requirements change with its pip
command, and the queue counts). Apart from mace/update.py, which runs as one
compressed file on a cluster (``--remote``) and stays small."""
import contextlib
import io
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

from mace.update import (MAIN, OPTIONAL_REQUIREMENTS, Git, Refused, busy_counts, changelog_between,
                         channel_target, describe, fetch, file_version, local_changes, mace_python, newer_message,
                         not_writable, pick_tag, pip_hint, plan, python_kind, remote_refs, requirement_changes,
                         requirements_changed, writable)


def check_report(home: Path, channel: str, runner: Callable = subprocess.run) -> Dict[str, Any]:
    """``--check --json``: one channel, everything an update notice shows
    (MACE GUI reads it). Changes nothing but origin's refs, like --check; the
    refusal is the text ``mace update`` would print, word for word."""
    r: Dict[str, Any] = {"home": str(home), "channel": channel, "state": "unknown", "available": False,
                         "writable": True, "owner_is_you": True, "current": {}, "target": {}, "how": "",
                         "changelog": [], "refusal": None, "refusal_kind": None, "message": "",
                         "local_changes": [], "stash": "", "requirements": None, "optional_changed": False,
                         "busy": {"jobs": 0, "running": 0, "waiting": 0}, "notes": []}
    notes = io.StringIO()

    def refuse(kind: str, text: str) -> Dict[str, Any]:
        r["refusal_kind"], r["refusal"] = kind, text
        r["notes"] = [ln for ln in notes.getvalue().splitlines() if ln.strip()]
        return r
    if not (home / ".git").exists():
        return refuse("not_git", f"{home} is not a git checkout (installed another way); update it the way it was "
                                 "installed.")
    git = Git(home, runner)
    try:
        r["owner_is_you"] = not hasattr(os, "getuid") or (home / ".git").stat().st_uid == os.getuid()
    except OSError:
        pass
    remote = None
    with contextlib.redirect_stdout(notes):
        try:
            if not git.ok("remote", "get-url", "origin"):
                return refuse("no_origin", f"{home} has no origin remote to update from")
            if writable(git):
                fetch(git)
            else:
                r["writable"] = False
                remote = remote_refs(git)
        except Refused as exc:
            return refuse("fetch", str(exc))
    r["notes"] = [ln for ln in notes.getvalue().splitlines() if ln.strip()]
    head = git.out("rev-parse", "HEAD")
    branch = git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    r["current"] = {"version": file_version(git, "HEAD"), "describe": describe(git), "branch": branch,
                    "commit": head}
    c = busy_counts(runner, gui=False)
    r["busy"] = {k: c[k] for k in ("jobs", "running", "waiting")}
    try:
        if remote is None:
            ref, target = channel_target(git, channel)
        elif channel == "latest":
            ref, target = f"origin/{MAIN}", remote[1]
            if not target:
                raise Refused(f"origin has no {MAIN} branch")
        else:
            ref = pick_tag(list(remote[0]), channel)
            target = remote[0][ref]
    except Refused as exc:
        return refuse("no_target", str(exc))
    r["target"] = {"ref": ref, "commit": target, "version": "?", "commits": None}
    if remote is not None and not git.rev(target):
        # origin has commits this read-only checkout never fetched
        r["target"]["version"] = ref[1:].replace("-preview", "") if ref.startswith("v") else "?"
        r["state"], r["available"] = "available", True
        return refuse("not_writable", not_writable(home))
    r["target"]["version"] = file_version(git, target)
    r["target"]["describe"] = describe(git, target)
    if target == head:
        r["state"] = "current"
    elif git.is_ancestor(head, target):
        r["state"], r["available"] = "available", True
        r["target"]["commits"] = len(git.out("rev-list", f"HEAD..{target}").split())
    elif git.is_ancestor(target, head):
        r["state"] = "newer"
        r["message"] = newer_message({"ref": ref, "new_version": r["target"]["version"]}, channel)
    else:
        r["state"] = "diverged"
    if r["state"] in ("available", "diverged"):
        r["changelog"] = changelog_between(git, r["current"]["version"], target, r["target"]["version"],
                                           channel == "latest")
        r["local_changes"] = local_changes(git)
        if r["local_changes"]:
            r["stash"] = f"git -C {shlex.quote(str(home))} stash"
        files = requirements_changed(git, head, target)
        if files:
            py = mace_python()
            kind, desc = python_kind(py)
            r["requirements"] = {"files": files, "changes": requirement_changes(git, head, target, files)[:20],
                                 "python": py, "python_kind": kind, "python_desc": desc,
                                 "pip": pip_hint(home, files, user=kind == "system"),
                                 "do_it_yourself": kind == "system"}
        r["optional_changed"] = bool(requirements_changed(git, head, target, OPTIONAL_REQUIREMENTS))
        if remote is not None:
            return refuse("not_writable", not_writable(home))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                p = plan(git, channel, do_fetch=False)
            r["how"] = p.get("how", "")
        except Refused as exc:
            text = str(exc)
            kind = ("local_changes" if r["local_changes"] and "local changes" in text else
                    "not_ff" if "fast-forward" in text else "other")
            return refuse(kind, text)
    return r
