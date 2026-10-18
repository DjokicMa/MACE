"""The standalone connect scripts (scripts/mace-gui-connect.sh and .ps1)
with a profile filled into their DEFAULTS block, for ``mace gui connect
--export-script``. Standard library only."""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, List

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
FILES = {"bash": "mace-gui-connect.sh", "powershell": "mace-gui-connect.ps1"}
BEGIN = "# >>> mace-gui-connect defaults >>>"
END = "# <<< mace-gui-connect defaults <<<"
FIELDS = ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda", "local_port", "ssh_options",
          "roots", "open")


def _ps_str(v: str) -> str:
    """A PowerShell string literal for ``v``, in plain ASCII.

    Printable ASCII goes in a single-quoted string with ' doubled. Every other character
    (controls, and anything non-ASCII) is spelled [char]0xNNNN, one per UTF-16 code unit,
    so the literal cannot end early: PowerShell also ends a single-quoted string at the
    curly quotes U+2018-U+201B, and Windows PowerShell 5.1 reads a .ps1 without a BOM
    in the ANSI code page, where UTF-8 bytes such as the 0x91 in Ñ read as a quote."""
    parts: List[str] = []
    run = ""
    units = v.encode("utf-16-le")
    for i in range(0, len(units), 2):
        code = units[i] | units[i + 1] << 8
        if 0x20 <= code < 0x7F:
            run += "''" if code == 0x27 else chr(code)
            continue
        if run:
            parts.append(f"'{run}'")
            run = ""
        parts.append(f"[char]0x{code:04X}")
    if run or not parts:
        parts.append(f"'{run}'")
    if len(parts) == 1 and parts[0].startswith("'"):
        return parts[0]
    if parts[0].startswith("[char]"):
        parts.insert(0, "''")         # string + char, never char + char (that adds the code points)
    return "(" + " + ".join(parts) + ")"


def defaults_block(kind: str, prof: Dict[str, Any]) -> List[str]:
    hop = str(prof.get("hop") or "jump")
    hop = "jump" if hop == "auto" else hop
    vals = {"gateway": str(prof.get("gateway") or ""), "jump": str(prof.get("jump") or ""),
            "node": str(prof.get("node") or ""), "hop": hop, "dir": str(prof.get("dir") or ""),
            "mace": str(prof.get("mace") or "mace"), "setup": str(prof.get("setup") or ""),
            "conda": str(prof.get("conda") or ""), "local_port": int(prof.get("local_port") or 8765),
            "ssh_options": [str(x) for x in prof.get("ssh_options") or []],
            "roots": [str(x) for x in prof.get("roots") or []], "open": prof.get("open", True) is not False}
    out = [BEGIN]
    for k in FIELDS:
        v = vals[k]
        name = "D_" + k.upper()
        if kind == "bash":
            if isinstance(v, list):
                out.append(f"{name}=(" + " ".join(shlex.quote(x) for x in v) + ")")
            elif isinstance(v, bool):
                out.append(f'{name}="{1 if v else 0}"')
            else:
                out.append(f"{name}={shlex.quote(str(v))}" if str(v) else f'{name}=""')
        else:
            if isinstance(v, list):
                out.append(f"${name} = @(" + ", ".join(_ps_str(x) for x in v) + ")")
            elif isinstance(v, bool):
                out.append(f"${name} = {'$true' if v else '$false'}")
            elif isinstance(v, int):
                out.append(f"${name} = {v}")
            else:
                out.append(f"${name} = {_ps_str(v)}" if v else f'${name} = ""')
    out.append(END)
    return out


def export_script(kind: str, prof: Dict[str, Any]) -> str:
    text = (SCRIPTS / FILES[kind]).read_text(encoding="utf-8")
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(text):
        raise SystemExit(f"mace gui connect: {FILES[kind]} has no defaults block")
    return pat.sub(lambda m: "\n".join(defaults_block(kind, prof)), text, count=1)
