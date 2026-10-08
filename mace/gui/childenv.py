"""The environment and working directory of every Python child the GUI starts.

The CLI children (deck previews, plots, Setup, preflight) and the process
pool's workers run the same interpreter as the server. They must not pick up
modules from the MACE checkout by accident: ``mace/queue`` is a package, and
with ``mace/`` (or a working directory inside the checkout) on a child's
``sys.path`` it shadows the standard library's ``queue``, which
``multiprocessing`` and ``concurrent.futures`` import. So:

* ``clean_pythonpath`` drops every inherited ``PYTHONPATH`` entry that is
  empty or relative (the child's working directory) or is the ``mace/``
  package folder or below it (``mace/queue`` is the only module in the
  checkout that shadows the standard library). Every other entry is kept: on
  HPCC the module system puts its Python bundles on ``PYTHONPATH``, and the
  children need the same packages as the server.
* ``clean_sys_path`` does the same for a pool worker's ``sys.path`` (the
  spawn start method copies the server's): the checkout root and the folders
  the server added for the CLI's modules (``Crystal_d12``, ``Crystal_d3``)
  stay, since the worker's jobs import from them.
* ``neutral_cwd`` is a working directory outside the checkout for children
  that do not need a particular one.

The pre-optimization children (xtb, the ML worker and its probe) go further
and drop every ``PYTHON*`` variable (``mace/preopt/proc.py``): they run
another program or another interpreter.

``env_report`` and ``env_fingerprint`` are what the test hooks may show of an
environment: an allowlist, and a one-way digest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from typing import Dict, Iterable, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACE_PKG = os.path.join(REPO, "mace")     # holds mace/queue, which shadows the stdlib queue

TEST_ENV_ALLOWLIST = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MACE_GUI_MAX_WORKERS",
                      "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES")
_SECRET_NAME = re.compile(r"TOKEN|SECRET|KEY|PASS|^SSH_", re.I)


def inside(path: str, root: str = REPO) -> bool:
    """``path`` is ``root`` or below it (both absolute, symlinks resolved)."""
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def clean_pythonpath(value: Optional[str], pkg: str = MACE_PKG) -> Optional[str]:
    """``value`` without empty or relative entries and without ``mace/`` or
    any folder below it; None when nothing is left."""
    if not value:
        return None
    keep: List[str] = []
    for entry in value.split(os.pathsep):
        if not entry or not os.path.isabs(entry) or inside(entry, pkg):
            continue
        if entry not in keep:
            keep.append(entry)
    return os.pathsep.join(keep) or None


def clean_sys_path(paths: Iterable[str], pkg: str = MACE_PKG) -> List[str]:
    """A worker's sys.path: without ``mace/`` or any folder below it and
    without relative entries; everything else (the checkout root, which the
    worker imports from, Crystal_d12, ...) stays."""
    out: List[str] = []
    for p in paths:
        if not p or not os.path.isabs(p):
            continue
        if inside(p, pkg):
            continue
        if p not in out:
            out.append(p)
    return out


def python_child_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``base`` (default: this process's environment) with PYTHONPATH cleaned."""
    env = dict(os.environ if base is None else base)
    cleaned = clean_pythonpath(env.get("PYTHONPATH"))
    if cleaned is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = cleaned
    return env


def neutral_cwd() -> str:
    """A working directory outside the checkout: the temp directory, or the
    file system root when even that is inside it."""
    tmp = tempfile.gettempdir()
    return tmp if not inside(tmp) else os.sep


def env_report(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment as the test hooks may show it: the allowlist only,
    never a token, a secret, a key, a password or an SSH_* variable."""
    env = os.environ if env is None else env
    return {k: env[k] for k in TEST_ENV_ALLOWLIST if k in env and not _SECRET_NAME.search(k)}


def env_fingerprint(env: Optional[Dict[str, str]] = None, skip: Iterable[str] = ()) -> str:
    """A one-way digest of the whole environment (less ``skip``), so a test
    can tell whether two processes have the same environment without seeing it."""
    env = os.environ if env is None else env
    skip = set(skip)
    items = sorted((k, v) for k, v in env.items() if k not in skip)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


# ------------------------------------------------------------------ the login environment
LOGIN_ENV: Optional[Dict[str, str]] = None


def capture_login_env() -> Dict[str, str]:
    """Keep the environment ``mace gui`` was started with, before anything
    in this process changes it (mace_cli keeps the shell's own copy as
    ``MACE_LOGIN_ENV``; cli.main calls this first). Jobs submitted from the
    GUI get this environment, as from the shell."""
    global LOGIN_ENV
    main = sys.modules.get("__main__")
    base = getattr(main, "MACE_LOGIN_ENV", None)
    LOGIN_ENV = dict(base if isinstance(base, dict) else os.environ)
    return dict(LOGIN_ENV)


def login_env() -> Dict[str, str]:
    """The captured login environment (this process's own when none was captured)."""
    return dict(LOGIN_ENV if LOGIN_ENV is not None else os.environ)
