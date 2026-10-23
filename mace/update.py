"""``mace update``: move a MACE git checkout to a release, a preview or the
latest main, safely, here and (``--remote``) on a cluster.

Channels (from the checkout's ``origin``, its tags fetched as they are now, a
retagged release included):

* ``release``: the newest final tag ``vX.Y.Z``;
* ``preview``: the newest ``vX.Y.Z-preview`` or final tag, whichever is newer
  (a final ``vX.Y.Z`` is newer than ``vX.Y.Z-preview``);
* ``latest``: the head of ``origin/main``.

The procedure never forces, never resets and never discards work:

1. refuse when tracked files have local changes (they are listed; ``git
   stash`` keeps them) or when the target is not a fast-forward of what is
   checked out (local commits, or a target older than the checkout);
2. show the current and the target version and the CHANGELOG sections in
   between, and ask (``--yes`` answers yes; without a terminal, ``--yes`` is
   required);
3. move: ``latest`` fast-forwards ``main`` (``git merge --ff-only``). A tag
   fast-forwards the current branch when the tag is on that branch's
   upstream (so ``git pull`` keeps working afterwards); otherwise it is
   checked out as a new branch ``mace-<tag>`` (``git checkout main`` and
   ``git pull`` return to main);
4. when requirements.txt changed, ask whether to run pip with the Python
   that runs MACE (``--user`` for a system or module Python) or leave it to
   you; without a terminal or with ``--yes`` only print the command
   (``--pip``/``--no-pip`` decide without asking); a changed
   requirements-optional.txt gets one line of its own (the test files are
   never mentioned); then check the result with ``mace_cli --version``.

It warns, but does not refuse, when SLURM jobs of yours are queued or
running (their callbacks run the code in this checkout) or a ``mace gui``
server runs.

``--remote [PROFILE]`` runs the same procedure in the cluster's checkout,
over ssh with a ``mace gui connect`` profile: the cluster's own ``mace
update`` when its MACE has the command, else this file, sent compressed with
the command (so a cluster MACE without ``mace update`` works too; the
cluster's checkout is then found from the profile's ``mace`` command,
``$MACE_HOME``, ``~/mace`` or ``~/MACE``); the output streams here. With a
terminal, ssh is interactive (2FA prompts work); without one,
``BatchMode=yes``.

Standard library only, and Python 3.6 or newer: it also runs as a single file
on the cluster, with its python3 (3.6 on RHEL/Rocky 8).
"""
import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

CHANNELS = ("release", "preview", "latest")
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(-preview)?$")
MAIN = "main"
CHANGELOG_LINES = 80          # at most this many CHANGELOG lines per update shown
REQUIREMENTS = "requirements.txt"                     # what MACE needs: the pip question
OPTIONAL_REQUIREMENTS = "requirements-optional.txt"   # optional features: one line, never pip
# what subprocess.run captures (its own capture and text arguments need Python 3.7)
CAPTURE = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "universal_newlines": True}


def shell_join(words: Sequence[str]) -> str:
    """The words as one shell line, each quoted (shlex's own join needs Python 3.8)."""
    return " ".join(shlex.quote(str(w)) for w in words)


def say(text: str = "") -> None:
    print(text, flush=True)


class Refused(Exception):
    """A reason not to update (printed, exit code 2)."""


# ----------------------------------------------------------------- git

class Git:
    def __init__(self, home: Path, runner: Callable = subprocess.run):
        self.home = Path(home)
        self.runner = runner

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = self.runner(["git", "-C", str(self.home)] + list(args), stdin=subprocess.DEVNULL, **CAPTURE)
        if check and r.returncode != 0:
            raise Refused(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
        return r

    def out(self, *args: str) -> str:
        return self.run(*args).stdout.strip()

    def ok(self, *args: str) -> bool:
        return self.run(*args, check=False).returncode == 0

    def rev(self, ref: str) -> Optional[str]:
        r = self.run("rev-parse", "--verify", "--quiet", ref + "^{commit}", check=False)
        return r.stdout.strip() or None

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.ok("merge-base", "--is-ancestor", a, b)


def tag_key(tag: str) -> Optional[Tuple[int, int, int, int]]:
    m = TAG_RE.match(tag)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), 0 if m.group(4) else 1


def version_key(v: str) -> Tuple[int, int, int, int]:
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)(-preview)?", v.strip())
    if not m:
        return (0, 0, 0, 0)
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), 0 if m.group(4) else 1


def tag_commits(git: Git) -> Dict[str, str]:
    """{tag: the commit it points to} for the local v* tags."""
    out = git.out("for-each-ref", "--format=%(refname:short) %(objectname) %(*objectname)", "refs/tags/v*")
    tags = {}
    for ln in out.splitlines():
        w = ln.split()
        if len(w) >= 2:
            tags[w[0]] = w[2] if len(w) > 2 else w[1]       # an annotated tag: the commit it peels to
    return tags


def fetch(git: Git) -> List[str]:
    """Fetch origin's branches and its tags as they are now. A tag that was
    moved on origin (a release retagged: v1.1.5 was the preview, now the
    final) is updated here too, with a note: a plain fetch keeps the stale
    local tag (and ``git fetch --tags`` refuses to "clobber" it). Only tags
    change; branches, the working tree and commits are left alone."""
    before = tag_commits(git)
    git.run("fetch", "--prune", "origin")
    git.run("fetch", "--force", "--tags", "origin")
    after = tag_commits(git)
    moved = sorted(t for t in before if t in after and before[t] != after[t])
    for t in moved:
        was = sorted(x for x, c in after.items() if c == before[t] and x != t)
        say(f"Note: {t} moved on origin" + (f" (it was the commit now tagged {', '.join(was)})" if was else "") +
            f"; updated the local tag {t}.")
    return moved


def pick_tag(tags: Sequence[str], channel: str) -> str:
    """The newest tag of a "release" or "preview" channel; Refused without one."""
    tags = [t for t in tags if tag_key(t)]
    if channel == "release":
        tags = [t for t in tags if tag_key(t)[3] == 1]
    if not tags:
        raise Refused(f"no {channel} tag (vX.Y.Z{'' if channel == 'release' else ' or vX.Y.Z-preview'}) on origin")
    return max(tags, key=tag_key)


def channel_target(git: Git, channel: str) -> Tuple[str, str]:
    """(ref name, commit) the channel points to (after a fetch)."""
    if channel == "latest":
        c = git.rev(f"origin/{MAIN}")
        if not c:
            raise Refused(f"origin has no {MAIN} branch")
        return f"origin/{MAIN}", c
    best = pick_tag(git.out("tag", "--list", "v*").split(), channel)
    return best, git.rev(best)


def remote_refs(git: Git) -> Tuple[Dict[str, str], Optional[str]]:
    """({tag: commit}, main's commit) on origin, from git ls-remote: asks origin
    without writing anything here (a read-only checkout)."""
    tags: Dict[str, str] = {}
    peeled: Dict[str, str] = {}
    main = None
    for ln in git.out("ls-remote", "--tags", "--heads", "origin").splitlines():
        w = ln.split()
        if len(w) != 2:
            continue
        sha, ref = w
        if ref == f"refs/heads/{MAIN}":
            main = sha
        elif ref.startswith("refs/tags/"):
            name = ref[len("refs/tags/"):]
            if name.endswith("^{}"):
                peeled[name[:-3]] = sha
            else:
                tags[name] = sha
    tags.update(peeled)                 # an annotated tag: the commit it peels to
    return tags, main


def writable(git: Git) -> bool:
    """This user can write the checkout's .git (fetch and move it)."""
    d = git.run("rev-parse", "--git-dir", check=False).stdout.strip()
    if not d:
        return True                     # git says why when it runs
    gd = Path(d) if os.path.isabs(d) else git.home / d
    return all(os.access(str(p), os.W_OK) for p in (gd, gd / "objects", gd / "refs") if p.exists()) and \
        all(os.access(str(p), os.W_OK) for p in (gd / "FETCH_HEAD", gd / "HEAD") if p.exists())


def not_writable(home: Path) -> str:
    return (f"{home} is not writable by you (its .git folder is read-only or another user's), so mace update cannot "
            "fetch or move it. Ask whoever installed it to update it, or use a clone of your own (git clone "
            f"$(git -C {shlex.quote(str(home))} remote get-url origin) ~/MACE).")


def file_version(git: Git, ref: str) -> str:
    r = git.run("show", f"{ref}:mace/__init__.py", check=False)
    m = re.search(r'__version__\s*=\s*"([^"]+)"', r.stdout or "")
    return m.group(1) if m else "?"


def describe(git: Git, ref: str = "HEAD") -> str:
    r = git.run("describe", "--tags", "--always", ref, check=False)
    return r.stdout.strip() or ref


def changelog_between(git: Git, cur: str, target_ref: str, target_version: str,
                      unreleased: bool = False) -> List[str]:
    """The CHANGELOG sections (from the target) newer than cur, up to the target."""
    text = git.run("show", f"{target_ref}:CHANGELOG.md", check=False).stdout or ""
    out: List[str] = []
    keep = False
    for line in text.splitlines():
        m = re.match(r"^## \[([^\]]+)\]", line)
        if m:
            name = m.group(1)
            if name.lower() == "unreleased":
                keep = unreleased                    # only main carries unreleased changes
            else:
                v = version_key(name)
                keep = version_key(cur) < v <= version_key(target_version)
        if keep:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    if len(out) > CHANGELOG_LINES:
        out = out[:CHANGELOG_LINES] + [f"... ({len(out) - CHANGELOG_LINES} more lines in CHANGELOG.md)"]
    return out


# ----------------------------------------------------------------- warnings

def busy_counts(runner: Callable = subprocess.run, gui: bool = True) -> Dict[str, int]:
    """{"jobs", "running", "waiting"}: your SLURM jobs; {"gui"}: mace gui processes here."""
    out = {"jobs": 0, "running": 0, "waiting": 0, "gui": 0}
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if shutil.which("squeue") and user:
        r = runner(["squeue", "-h", "-u", user, "-o", "%T"], stdin=subprocess.DEVNULL, **CAPTURE)
        states = [s.strip() for s in (r.stdout or "").splitlines() if s.strip()] if r.returncode == 0 else []
        run = sum(1 for s in states if s == "RUNNING")
        out.update(jobs=len(states), running=run, waiting=len(states) - run)
    if gui and shutil.which("pgrep"):
        r = runner(["pgrep", "-f", "mace_cli.* gui"], stdin=subprocess.DEVNULL, **CAPTURE)
        out["gui"] = len([x for x in (r.stdout or "").split() if x.strip() and x.strip() != str(os.getpid())])
    return out


def busy_warnings(home: Path, runner: Callable = subprocess.run, counts: Optional[Dict[str, int]] = None) -> List[str]:
    """Jobs of yours in SLURM and mace gui servers: they run this checkout's code."""
    c = counts if counts is not None else busy_counts(runner)
    out: List[str] = []
    if c["jobs"]:
        out.append(f"Warning: {c['jobs']} SLURM job(s) of yours are in the queue ({c['running']} running, "
                   f"{c['waiting']} waiting). Their completion callbacks run the code in {home}: "
                   "update between batches when you can.")
    if c.get("gui"):
        out.append(f"Warning: {c['gui']} mace gui process(es) run on this computer; restart them after the update.")
    return out


# ----------------------------------------------------------------- the procedure

def find_home(hint: str = "") -> Optional[Path]:
    """A MACE checkout: the one the mace command (hint) runs from, $MACE_HOME,
    ~/mace or ~/MACE (the first with mace_cli and .git)."""
    cands: List[Path] = []
    for w in shlex.split(hint or ""):
        p = os.path.expanduser(w) if "/" in w else shutil.which(w)       # never a name relative to the cwd
        if p and os.path.exists(p) and "python" not in os.path.basename(p):
            cands.append(Path(os.path.realpath(p)).parent)
    if os.environ.get("MACE_HOME"):
        cands.append(Path(os.environ["MACE_HOME"]))
    found = shutil.which("mace")
    if found:
        cands.append(Path(os.path.realpath(found)).parent)
    cands += [Path.home() / "mace", Path.home() / "MACE"]
    for c in cands:
        for d in (c, *c.parents):
            if (d / "mace_cli").is_file() and (d / ".git").exists():
                return d
    return None


def local_changes(git: Git) -> List[str]:
    # not git.out: its strip() would eat the first line's leading status column (" M")
    out = git.run("status", "--porcelain", "--untracked-files=no").stdout or ""
    return [ln.rstrip() for ln in out.splitlines() if ln.strip()]


def plan(git: Git, channel: str, do_fetch: bool = True) -> Dict[str, str]:
    """Fetch, then what moving to the channel would do; Refused when it must not."""
    if not git.ok("remote", "get-url", "origin"):
        raise Refused(f"{git.home} has no origin remote to update from")
    if not writable(git):
        raise Refused(not_writable(git.home))
    if do_fetch:
        say(f"Fetching from origin in {git.home} ...")
        fetch(git)
    ref, target = channel_target(git, channel)
    head = git.out("rev-parse", "HEAD")
    branch = git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    p = {"ref": ref, "target": target, "head": head, "branch": branch,
         "cur_version": file_version(git, "HEAD"), "new_version": file_version(git, target),
         "cur_desc": describe(git), "new_desc": describe(git, target)}
    if target == head:
        p["action"] = "none"
        return p
    if git.is_ancestor(target, head):
        p["action"] = "newer"           # e.g. main after a git pull, ahead of the newest release
        return p
    dirty = local_changes(git)
    if dirty:
        raise Refused("these files have local changes, so nothing was updated:\n  " + "\n  ".join(dirty) +
                      f"\nKeep them aside with: git -C {shlex.quote(str(git.home))} stash   (and git stash pop "
                      "after the update), then run mace update again.")
    if not git.is_ancestor(head, target):
        ahead = git.out("log", "--oneline", f"{target}..HEAD").splitlines()
        raise Refused(f"this checkout has {len(ahead)} commit(s) that {ref} does not have, so moving to it is not a "
                      "fast-forward:\n  " + "\n  ".join(ahead[:10]) +
                      "\nmace update never discards commits; merge or move them yourself.")
    upstream = git.run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).stdout.strip()
    if channel == "latest":
        if branch == MAIN:
            p["action"], p["how"] = "ff", f"git merge --ff-only {ref}"
        elif git.rev(f"refs/heads/{MAIN}") and not git.is_ancestor(f"refs/heads/{MAIN}", target):
            raise Refused(f"your local {MAIN} branch has commits that {ref} does not have; merge them yourself.")
        else:
            p["action"] = "switch-main"
            p["how"] = (f"git checkout {MAIN} && git merge --ff-only {ref}" if git.rev(f"refs/heads/{MAIN}")
                        else f"git checkout -b {MAIN} --track {ref}")
    elif branch and upstream and git.is_ancestor(target, upstream):
        p["action"], p["how"] = "ff", f"git merge --ff-only {ref}   (on {branch}; git pull keeps working)"
    else:
        name = f"mace-{ref}"
        if git.rev(f"refs/heads/{name}") and git.rev(f"refs/heads/{name}") != target:
            raise Refused(f"a branch {name} exists and is not at {ref}; rename or delete it yourself.")
        p["action"], p["branch_new"] = "branch", name
        p["how"] = (f"git checkout {name}" if git.rev(f"refs/heads/{name}") else f"git checkout -b {name} {ref}") + \
            f"   (return later with: git checkout {MAIN} && git pull)"
    return p


def apply(git: Git, p: Dict[str, str]) -> None:
    a = p["action"]
    ref = p["ref"]
    if a == "ff":
        git.run("merge", "--ff-only", p["target"])
    elif a == "switch-main":
        if git.rev(f"refs/heads/{MAIN}"):
            git.run("checkout", MAIN)
            git.run("merge", "--ff-only", p["target"])
        else:
            git.run("checkout", "-b", MAIN, "--track", ref)
    elif a == "branch":
        name = p["branch_new"]
        if git.rev(f"refs/heads/{name}"):
            git.run("checkout", name)
        else:
            git.run("checkout", "-b", name, p["target"])


def requirements_changed(git: Git, old: str, new: str, name: str = REQUIREMENTS) -> List[str]:
    """[name] when the update changed that requirements file, else [] (only
    requirements.txt is pip's: the -test files are for developers, and
    requirements-optional.txt gets a line of its own)."""
    r = git.run("diff", "--name-only", old, new, "--", name, check=False)
    return [f for f in (r.stdout or "").split() if f == name]


def mace_python() -> str:
    """The Python that runs MACE (MACE_PYTHON overrides it)."""
    return os.environ.get("MACE_PYTHON") or sys.executable or "python3"


def python_kind(py: str) -> Tuple[str, str]:
    """("venv" | "conda" | "system", a description) of the Python at py. A
    venv has pyvenv.cfg in its prefix, a conda environment conda-meta (named
    after its folder; "base" for conda's own); anything else is a system or
    module Python, whose site-packages are not yours to change."""
    prefix = Path(py).parent.parent             # <prefix>/bin/python, symlinks kept (a venv's python is one)
    if (prefix / "pyvenv.cfg").is_file():
        return "venv", f"the virtual environment {prefix}"
    if (prefix / "conda-meta").is_dir():
        name = "base" if (prefix / "envs").is_dir() or prefix.parent.name != "envs" else prefix.name
        return "conda", f"the conda environment {name} ({prefix})"
    return "system", "a system or module Python (not a virtual or conda environment)"


def requirement_changes(git: Git, old: str, new: str, files: Sequence[str]) -> List[str]:
    """The requirement lines added or changed (the + lines of the diff)."""
    r = git.run("diff", "--unified=0", old, new, "--", *files, check=False)
    return [ln[1:].strip() for ln in (r.stdout or "").splitlines()
            if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip()
            and not ln[1:].strip().startswith("#")]


def pip_argv(home: Path, files: Sequence[str], py: Optional[str] = None, user: bool = False) -> List[str]:
    argv = [py or mace_python(), "-m", "pip", "install"] + (["--user"] if user else [])
    for f in files:
        argv += ["-r", str(home / f)]
    return argv


def pip_hint(home: Path, files: Sequence[str], user: bool = False) -> str:
    return shell_join(pip_argv(home, files, user=user))


ENV_HINT = ("Your own environment instead: python3 -m venv ~/mace-env && ~/mace-env/bin/python -m pip install -r "
            "{req}   (or conda create -n mace python=3.11, conda activate mace, then pip install -r {req}); then "
            "run MACE with that Python.")
USER_WHY = ("This is a system or module Python: pip into its own site-packages needs administrator rights and can "
            "break other tools, so the command uses --user (packages go to your home folder).")


def requirements_step(git: Git, home: Path, old: str, new: str, files: Sequence[str], *, pip: Optional[bool],
                      yes: bool, tty: bool, ask: Optional[Callable[[str], str]], runner: Callable) -> Optional[bool]:
    """After an update that changed requirements: ask whether to run pip, run
    it, or print the command. True/False: pip ran and worked/failed; None: not run."""
    py = mace_python()
    kind, desc = python_kind(py)
    user = kind == "system"
    cmd = pip_hint(home, files, user=user)
    say(f"{', '.join(files)} changed:")
    for ln in requirement_changes(git, old, new, files)[:20]:
        say(f"  + {ln}")
    say(f"MACE runs with {py}, {desc}.")
    if user:
        say(USER_WHY)
    run = bool(pip)
    if pip is None and not yes and tty and ask is not None:
        default = "2" if user else "1"
        say(f"  [1] run it now: {cmd}")
        say("  [2] I'll do it myself, or set up my own environment")
        answer = ask(f"Install the new requirements? [1/2] (default {default}) ").strip() or default
        run = answer.startswith("1")
    if not run:
        say("Install them into the Python that runs MACE (not done for you):")
        say("  " + cmd)
        if pip is None and not yes and tty:
            say(ENV_HINT.format(req=shlex.quote(str(home / files[0]))))
        return None
    say(f"Running: {cmd}")
    r = runner(pip_argv(home, files, user=user))
    if r.returncode == 0:
        say("The requirements are installed.")
        return True
    say(f"pip failed (exit {r.returncode}). The MACE code IS updated, but its packages are NOT; fix the error above "
        f"and run it again:\n  {cmd}")
    return False


def update(home: Path, channel: str, *, dry_run: bool = False, yes: bool = False,
           ask: Optional[Callable[[str], str]] = None, runner: Callable = subprocess.run,
           pip: Optional[bool] = None, tty: Optional[bool] = None) -> int:
    if tty is None:
        tty = bool(sys.stdin and sys.stdin.isatty())
    git = Git(home, runner)
    try:
        p = plan(git, channel)
    except Refused as exc:
        say(f"mace update: {exc}")
        return 2
    say(f"MACE in {home}: {p['cur_version']} ({p['cur_desc']}" + (f", on {p['branch']}" if p["branch"] else
                                                               ", detached") + ")")
    if p["action"] == "none":
        say(f"Already at {p['ref']} ({p['new_version']}); nothing to do.")
        return 0
    if p["action"] == "newer":
        say(newer_message(p, channel))
        return 0
    say(f"Target ({channel}): {p['ref']} = {p['new_version']} ({p['new_desc']})")
    notes = changelog_between(git, p["cur_version"], p["target"], p["new_version"], channel == "latest")
    if notes:
        say("")
        say("Changes (CHANGELOG.md):")
        for ln in notes:
            say("  " + ln)
        say("")
    for w in busy_warnings(home, runner):
        say(w)
    say(f"Would run: {p['how']}" if dry_run else f"Will run: {p['how']}")
    if dry_run:
        say("Dry run: nothing changed (the fetch only updated origin's tags and branches).")
        return 0
    if not yes:
        if ask is None or not tty:
            say("mace update: no terminal to confirm on; pass --yes to update.")
            return 2
        if not ask(f"Update {home} to {p['ref']}? [y/N] ").strip().lower().startswith("y"):
            say("Not updated.")
            return 1
    try:
        apply(git, p)
    except Refused as exc:
        say(f"mace update: {exc}")
        return 2
    say(f"Updated {home} to {p['ref']} ({describe(git)}).")
    req = requirements_changed(git, p["head"], p["target"])
    pip_ok = None
    if req:
        pip_ok = requirements_step(git, home, p["head"], p["target"], req, pip=pip, yes=yes, tty=tty, ask=ask,
                                   runner=runner)
    if requirements_changed(git, p["head"], p["target"], OPTIONAL_REQUIREMENTS):
        say(f"{OPTIONAL_REQUIREMENTS} changed too: it lists optional features' packages (INSTALLATION.md); "
            "install from it only if you use them.")
    v = runner([sys.executable, str(home / "mace_cli"), "--version"], stdin=subprocess.DEVNULL, **CAPTURE)
    text = ((v.stdout or "") + (v.stderr or "")).strip().splitlines()
    if v.returncode == 0 and text:
        say(f"Check: {text[-1]}")
    else:
        say(f"Warning: mace_cli --version did not answer (exit {v.returncode}); "
            + (f"install the requirements above, then run {home / 'mace_cli'} --version" if req and not pip_ok else
               f"run {home / 'mace_cli'} --version to see why."))
    return 0


def check(home: Path, runner: Callable = subprocess.run) -> int:
    """Current vs available for each channel; changes nothing but origin's refs."""
    git = Git(home, runner)
    remote = None
    try:
        if not git.ok("remote", "get-url", "origin"):
            raise Refused(f"{home} has no origin remote")
        if writable(git):
            fetch(git)
        else:
            remote = remote_refs(git)   # ask origin, write nothing here
    except Refused as exc:
        say(f"mace update: {exc}")
        return 2
    head = git.out("rev-parse", "HEAD")
    branch = git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    say(f"MACE in {home}: {file_version(git, 'HEAD')} ({describe(git)}" + (f", on {branch}" if branch else
                                                                           ", detached") + ")")
    if remote is not None:
        say(f"Note: {home} is not writable by you: compared with origin by git ls-remote (nothing fetched); "
            "mace update cannot move this checkout.")
    for ch in CHANNELS:
        try:
            if remote is None:
                ref, target = channel_target(git, ch)
            elif ch == "latest":
                ref, target = f"origin/{MAIN}", remote[1]
                if not target:
                    raise Refused(f"origin has no {MAIN} branch")
            else:
                ref = pick_tag(list(remote[0]), ch)
                target = remote[0][ref]
        except Refused as exc:
            say(f"  {ch:<8} {exc}")
            continue
        if remote is not None and not git.rev(target):
            version = ref[1:] if ref.startswith("v") else "?"
            say(f"  {ch:<8} {ref} ({version}): available (not in this checkout yet)")
            continue
        if target == head:
            state = "current"
        elif git.is_ancestor(head, target):
            n = len(git.out("rev-list", f"HEAD..{target}").split())
            state = f"available ({n} commit{'s' if n != 1 else ''} ahead)" + \
                ("" if remote is not None else f": mace update --channel {ch}")
        elif git.is_ancestor(target, head):
            state = "older than this checkout"
        else:
            state = "not a fast-forward of this checkout"
        say(f"  {ch:<8} {ref} ({file_version(git, target)}): {state}")
    return 0


def newer_message(p: Dict[str, str], channel: str) -> str:
    return (f"This checkout is newer than {p['ref']} ({p['new_version']}); nothing to do (mace update only moves "
            "forward)." + ("" if channel == "latest" else " For the newest main: mace update --channel latest"))


# ----------------------------------------------------------------- remote

def remote_argv(profile: Dict, args_out: List[str], tty: bool, runner: Callable = subprocess.run) -> List[str]:
    """ssh to the profile's gateway (or its node, without a gateway) running
    the cluster's own ``mace update`` when its MACE has the command, else
    this file (zlib and base64: about 15 KB, well inside Windows' 32,767
    characters for a command line) with Python there. The login node shares
    your home (and so the MACE checkout) with the nodes, so the gateway is
    enough."""
    from mace.gui import connect as C
    host = str(profile.get("gateway") or profile.get("node") or "")
    if not host:
        raise Refused("the profile has no gateway or node")
    extra = [str(x) for x in profile.get("ssh_options") or []]
    argv = ["ssh"]
    if profile.get("jump"):
        argv += ["-J", str(profile["jump"])]
    argv += ["-t"] if tty else ["-o", "BatchMode=yes"]
    # through a jump host that may ask for a passphrase or 2FA, the gateway's
    # ConnectTimeout also covers that login: mace gui connect's longer wait
    prompt = C.prompt_wait(C.jump_chain(profile, "nested", runner=runner), runner)
    seconds = C.PROMPT_TIMEOUT_S if prompt else C.CONNECT_TIMEOUT_S
    argv += C._timeout_opts(extra, C.config_connect_timeout(host, runner), seconds) + extra
    src = base64.b64encode(zlib.compress(Path(__file__).read_bytes(), 9)).decode()
    words = shlex.split(str(profile.get("mace") or "mace")) or ["mace"]
    py = words[0] if "python" in os.path.basename(words[0]) else "python3"
    code = ("import base64,sys,zlib; exec(compile(zlib.decompress(base64.b64decode('" + src + "')), "
            "'mace_update.py', 'exec'))")
    sent = shell_join([py, "-c", code, "--find-home", " ".join(words)] + args_out)
    # the cluster's own mace update first (its MACE knows its checkout); a MACE
    # older than 1.1.6 has no update command, and this file runs instead
    find = C.mace_fallback(words)
    mace = '"$M"' if find else " ".join(C._home_path(w) for w in words)
    own = (f"if {mace} update --help 2>/dev/null | grep -q -e --remote-only; then exec {mace} update "
           f"{shell_join(args_out)}; fi; exec {sent}")
    if find:
        own = "{ " + find + "; } >/dev/null 2>&1; " + own
    parts = []
    setup = profile.get("setup")
    if isinstance(setup, (list, tuple)):
        setup = "; ".join(str(x) for x in setup)
    if setup and str(setup).strip():
        parts.append(str(setup).strip().rstrip(";"))
    if profile.get("conda"):
        parts.append("conda activate " + C._home_path(str(profile["conda"])))
    parts.append(own)
    # "--": a host that starts with "-" is still the host, never an ssh option
    return argv + ["--", host, "exec bash -lc " + shlex.quote("; ".join(parts))]


def run_remote(name: Optional[str], args_out: List[str], runner: Callable = subprocess.run) -> int:
    from mace.gui import connect as C
    data = C.load_profiles()
    profiles = data.get("profiles") or {}
    name = name or C.default_profile_name(data)
    if not name or name not in profiles:
        say(f"mace update: no connect profile {name or '(default)'}; save one with mace gui connect --save "
            "(mace gui connect --list shows them).")
        return 2
    prof = dict(profiles[name])
    tty = bool(sys.stdin and sys.stdin.isatty())
    argv = remote_argv(prof, args_out, tty)
    where = prof.get("gateway") or prof.get("node")
    say(f"--- the cluster ({name}: {where}) ---")
    chain = C.jump_chain(prof, "nested")
    if C.prompt_wait(chain):
        # ssh now waits PROMPT_TIMEOUT_S: a first jump host that does not answer fails fast here
        slow = C.first_hop_timed_out(chain)
        if slow:
            say(f"mace update: {C.unreachable_message(slow, 'timeout')}")
            say(f"mace gui connect {name} --check tests each step of the connection.")
            return 255
    r = runner(argv)
    if r.returncode == 255:
        say(f"mace update: ssh to {where} failed (see its message above); mace gui connect {name} --check tests "
            "the connection.")
    return r.returncode


# ----------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mace update",
        description="Update this MACE git checkout (and, with --remote, a cluster's) to the newest release, preview "
                    "or main, safely: no forced moves, no resets, refused with local changes or local commits.",
        epilog="Channels: release = newest vX.Y.Z tag; preview = newest vX.Y.Z-preview or vX.Y.Z tag; latest = "
               "origin/main. See docs/update.md.")
    p.add_argument("--channel", choices=CHANNELS, default="release", help="what to update to (default: release)")
    p.add_argument("--check", action="store_true", help="only report the current and the available versions")
    p.add_argument("--json", action="store_true", help="with --check: one JSON object for --channel on stdout "
                                                        "(what MACE GUI's update notice reads)")
    p.add_argument("--dry-run", action="store_true", help="show what would change; change nothing")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask before updating (and never run pip "
                                                              "unless --pip)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pip", dest="pip", action="store_const", const=True, default=None,
                   help="when requirements changed, run pip install with the Python that runs MACE without "
                        "asking (--user for a system or module Python)")
    g.add_argument("--no-pip", dest="pip", action="store_const", const=False,
                   help="when requirements changed, only print the pip command (no question)")
    p.add_argument("--local", action="store_true", help="update this computer's checkout (the default without "
                                                        "--remote; with --remote, both)")
    p.add_argument("--remote", nargs="?", const="", metavar="PROFILE",
                   help="update the MACE on the cluster of this mace gui connect profile (default: the default "
                        "profile); add --local to update this computer too")
    p.add_argument("--remote-only", action="store_true", help="only the cluster (the default profile unless "
                                                              "--remote names one)")
    p.add_argument("--home", help="the checkout to update (default: the one this mace runs from)")
    p.add_argument("--find-home", metavar="MACE_COMMAND", help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json:
        if not args.check or args.remote is not None or args.remote_only:
            print("mace update: --json goes with --check (this computer's checkout only)", file=sys.stderr)
            return 2
        home = Path(args.home).expanduser() if args.home else Path(__file__).resolve().parents[1]
        from mace.update_report import check_report      # never part of the --remote payload
        print(json.dumps(check_report(home, args.channel)), flush=True)
        return 0
    if args.remote_only and args.remote is None:
        args.remote = ""
    do_local = args.remote is None or (args.local and not args.remote_only)
    do_remote = args.remote is not None
    code = 0
    if do_local:
        if args.find_home is not None:
            home = find_home(args.find_home)
            if home is None:
                say("mace update: no MACE git checkout found here (looked at the mace command, $MACE_HOME, ~/mace "
                    "and ~/MACE). Pass --mace <path to mace_cli> to mace gui connect --save, or update by hand: "
                    "git -C <MACE folder> pull")
                return 2
        else:
            home = Path(args.home).expanduser() if args.home else Path(__file__).resolve().parents[1]
        if not (home / ".git").exists():
            say(f"mace update: {home} is not a git checkout (installed another way); update it the way it was "
                "installed.")
            return 2
        if args.check:
            code = check(home)
        else:
            code = update(home, args.channel, dry_run=args.dry_run, yes=args.yes, ask=input, pip=args.pip)
    if do_remote and code in (0, 1):
        out = ["--channel", args.channel] + (["--check"] if args.check else []) + \
              (["--dry-run"] if args.dry_run else []) + (["--yes"] if args.yes else []) + \
              (["--pip"] if args.pip else ["--no-pip"] if args.pip is False else [])
        rc = run_remote(args.remote or None, out)
        code = code or rc
    elif do_remote:
        say(f"mace update: the cluster ({remote_name(args.remote)}) was not updated, because the update here "
            "stopped (above). Fix that and run it again, or update only the cluster with --remote-only.")
    return code


def remote_name(name: Optional[str]) -> str:
    """The connect profile --remote names, or the default one."""
    if name:
        return name
    try:
        from mace.gui import connect as C
        return C.default_profile_name(C.load_profiles()) or "the default profile"
    except Exception:
        return "the default profile"


if __name__ == "__main__":     # also how the cluster runs it (sent by --remote)
    sys.exit(main())
