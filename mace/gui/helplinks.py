"""Which Git ref the GUI's Help links point at.

The Documentation link used to be `blob/v{__version__}/docs/gui.md`, which is a
404 whenever the version in `mace/__init__.py` has no tag on GitHub yet (every
development checkout between two releases, and every preview). A release tag
is used only when it is a final release version (`X.Y.Z`, no suffix) and that
tag exists in this checkout; everything else points at `main`, which always
exists.
"""
from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

_FINAL = re.compile(r"^\d+\.\d+\.\d+$")
_REPO = Path(__file__).resolve().parents[2]


def is_final(version: str) -> bool:
    return bool(_FINAL.match(version or ""))


@functools.lru_cache(maxsize=8)
def _tag_exists(tag: str, repo: str) -> bool:
    try:
        r = subprocess.run(["git", "-C", repo, "tag", "--list", tag], capture_output=True,
                           text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and tag in r.stdout.split()


def docs_ref(version: str, repo: Path | str | None = None) -> str:
    """`v{version}` for a final release whose tag exists here, else `main`."""
    if not is_final(version):
        return "main"
    tag = f"v{version}"
    return tag if _tag_exists(tag, str(repo or _REPO)) else "main"
