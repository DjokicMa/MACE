"""The folders MACE GUI may serve (File > Open Folder) and the user's scratch
folder: one function for both, so the window title's "Scratch" abbreviation
and Open Folder's "Scratch" chip agree (/api/info.scratch).

* ``home_aliases()``: ``$HOME`` as written and its real path.
* ``scratch_roots()``: ``$SCRATCH``, ``$WORK``, ``$PSCRATCH`` when set,
  non-empty and existing; then the first existing directory owned by you
  among common patterns (``/scratch/$USER``, ``/mnt/scratch/$USER``,
  ``/global/scratch/$USER``, ``/work/$USER``): many clusters set no
  ``$SCRATCH`` (MSU's HPCC is one). No site names.
* ``allowed_roots(cli_roots, env)``: ``--root DIR`` (repeatable) or
  ``MACE_GUI_ROOTS`` (os.pathsep list) replace the defaults; ``--root none``
  or ``MACE_GUI_ROOTS=`` (set, empty) turn Open Folder off (returns None).
  The defaults are your home folder and the scratch roots.

Standard library only.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

SCRATCH_VARS = ("SCRATCH", "WORK", "PSCRATCH")
SCRATCH_PATTERNS = ("/scratch/{user}", "/mnt/scratch/{user}", "/global/scratch/{user}", "/work/{user}")
# Their presence marks a home folder (holds_credentials), and they are never read.
CREDENTIAL_NAMES = (".ssh", ".gnupg", ".aws", ".kube", ".docker", ".netrc", ".git-credentials", ".pgpass")
CREDENTIAL_PAIRS = ((".config", "gh"),)
# Never read either, wherever they are (.docker/ and .kube/ are whole above),
# but they do not mark a home folder: a project's .env or .npmrc is common.
# Package-index and registry tokens, cloud logins, .env files, shell and
# interpreter histories, ssh private keys.
SECRET_NAMES = CREDENTIAL_NAMES + (".pypirc", ".npmrc", ".vault-token", ".azure", ".env",
                                   ".histfile", ".lesshst")
SECRET_PAIRS = CREDENTIAL_PAIRS + ((".config", "gcloud"),)
# .env.local, .bash_history/.zsh_history/.python_history/.Rhistory and the
# like, and ssh private keys by their key type (id_rsa, id_ed25519_work);
# id_*.pub is public. Only dot names for histories, so a "run_history"
# folder of outputs still lists.
SECRET_RE = re.compile(r"^(\.env\..+|\..*history|id_(rsa|dsa|ecdsa|ed25519|xmss)[A-Za-z0-9_-]*)$")


def _env(env):
    return os.environ if env is None else env


def _user(env: Mapping[str, str]) -> str:
    u = env.get("USER") or env.get("LOGNAME") or ""
    if not u:
        try:
            import getpass
            u = getpass.getuser()
        except Exception:
            u = ""
    return u


def home_aliases(env: Optional[Mapping[str, str]] = None) -> List[str]:
    e = _env(env)
    home = e.get("HOME") or os.path.expanduser("~")
    out: List[str] = []
    for p in (home, os.path.realpath(home)):
        p = p.rstrip("/") or "/"
        if p and p not in out:
            out.append(p)
    return out


def _owned_dir(p: Path) -> bool:
    try:
        st = p.stat()
    except OSError:
        return False
    if not p.is_dir():
        return False
    return not hasattr(os, "getuid") or st.st_uid == os.getuid()


def scratch_roots(env: Optional[Mapping[str, str]] = None,
                  patterns: Sequence[str] = SCRATCH_PATTERNS) -> List[Path]:
    e = _env(env)
    out: List[Path] = []

    def add(p: Path):
        r = Path(os.path.realpath(str(p)))
        if r not in out:
            out.append(r)
    for var in SCRATCH_VARS:
        v = (e.get(var) or "").strip()
        if v and os.path.isdir(v):
            add(Path(v))
    user = _user(e)
    if user:
        for pat in patterns:
            p = Path(pat.format(user=user))
            if _owned_dir(p):
                add(p)
                break
    return out


def parse_roots(value: str) -> List[str]:
    return [x for x in value.split(os.pathsep) if x.strip()]


def allowed_roots(cli_roots: Optional[Sequence[str]] = None,
                  env: Optional[Mapping[str, str]] = None) -> Optional[List[Path]]:
    """The roots Open Folder may switch within, or None when it is off."""
    e = _env(env)
    if cli_roots:
        if any(str(r).strip().lower() == "none" for r in cli_roots):
            return None
        raw = list(cli_roots)
    elif "MACE_GUI_ROOTS" in e:
        raw = parse_roots(e["MACE_GUI_ROOTS"])
        if not raw:
            return None
    else:
        home = e.get("HOME") or os.path.expanduser("~")
        return _dedupe([Path(home)] + scratch_roots(e))
    return _dedupe([Path(os.path.expanduser(str(r))) for r in raw])


def _dedupe(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        try:
            r = Path(os.path.realpath(str(p)))
        except OSError:
            continue
        if r.is_dir() and r not in out:
            out.append(r)
    return out


def root_label(p: Path, env: Optional[Mapping[str, str]] = None) -> str:
    e = _env(env)
    if str(p) in home_aliases(e):
        return "Home"
    if p in scratch_roots(e):
        return "Scratch"
    return p.name or str(p)


def credential_component(parts: Sequence[str]) -> Optional[str]:
    """The credential or secret folder or file named among these path
    components, if any: SECRET_NAMES, SECRET_PAIRS and SECRET_RE (``.env.*``,
    dot histories, ssh private keys).
    The one list for reads, the file listing and every download or export."""
    for i, name in enumerate(parts):
        if name in SECRET_NAMES or SECRET_RE.match(name):
            return name
        for a, b in SECRET_PAIRS:
            if name == a and i + 1 < len(parts) and parts[i + 1] == b:
                return f"{a}/{b}"
    return None


def secret_refusal(found: str) -> str:
    """What a read of a credential or secret file answers."""
    return (f"{found} is hidden for safety (it may hold credentials or other secrets); "
            "MACE GUI never opens it")


def holds_credentials(folder: Path) -> bool:
    """A folder with credential folders or files directly in it (a home folder)."""
    for name in CREDENTIAL_NAMES:
        if (folder / name).exists():
            return True
    return any((folder / a / b).exists() for a, b in CREDENTIAL_PAIRS)


def hidden_component(parts: Sequence[str]) -> Optional[str]:
    for name in parts:
        if name.startswith(".") and name not in (".", ".."):
            return name
    return None
