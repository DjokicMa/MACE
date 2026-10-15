"""MACE's temporary job-script generators: the copies of submitcrystal23.sh
and submit_prop.sh that `mace submit -i`, `--walltime` and `--exclude`
write with their resources (crystal.py and properties.py).

Each one goes to a folder of its own, readable only by the user, in the
user's temporary folder (TMPDIR, else /tmp), never into the install's
mace/submission/: a group or site-wide install is read-only to the people
using it, and two runs must not share a file.

The shipped generator pins the job's completion callback to the install
from its own location (MACE_GEN_DIR, "<install>/mace/submission"). A copy
elsewhere names the template's folder in that line instead, so the job
script it writes is byte for byte the one a copy beside the template wrote.
"""
import atexit
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Optional

PREFIX = "mace-generator-"
_GEN_DIR_LINE = re.compile(r'(?m)^MACE_GEN_DIR=\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]:-\$0\}"\)" 2>/dev/null && pwd\)$')


def write(template, text: str, name: str) -> Path:
    """Write ``text`` (an edited copy of ``template``) as an executable
    ``name`` in a new private temporary folder; removed by ``remove`` or, at
    the latest, when the process exits."""
    template = Path(template)
    pin = f"MACE_GEN_DIR=$(cd {shlex.quote(str(template.parent))} 2>/dev/null && pwd)"
    text = _GEN_DIR_LINE.sub(lambda _m: pin, text, count=1)
    folder = Path(tempfile.mkdtemp(prefix=PREFIX))
    path = folder / name
    try:
        path.write_text(text)
        path.chmod(0o755)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    atexit.register(remove, path, quiet=True)
    return path


def remove(path, quiet: bool = False) -> Optional[str]:
    """Remove a generator ``write`` made, and its folder. Returns the warning
    (also printed unless ``quiet``) when it could not be removed."""
    if not path:
        return None
    path = Path(path)
    folder = path.parent
    try:
        if path.exists():
            path.unlink()
        if folder.name.startswith(PREFIX) and folder.is_dir():
            folder.rmdir()
    except OSError as exc:
        msg = f"Warning: Could not remove temporary script {path}: {exc}"
        if not quiet:
            print(msg)
        return msg
    return None
