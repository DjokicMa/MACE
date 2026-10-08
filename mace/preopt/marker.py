"""The ``# mace-preopt:`` line that marks a CIF as an FF/ML starting geometry.

Written by ``mace preopt`` and by the GUI's Save CIF of a document accepted
from a GFN-FF/ML Clean; read by ``mace convert`` (a non-blocking warning
when the deck it builds is not an OPT), by the GUI when such a CIF is opened,
and by the Setup panel. Springs results are not marked: they are only tidied.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

PREFIX = "# mace-preopt:"
_LINE = re.compile(r"^#\s*mace-preopt:\s*(.*)$")
_SAFE = re.compile(r"[^A-Za-z0-9_.:+-]")


def line(method_ran: str, model: Optional[str] = None) -> str:
    m = _SAFE.sub("_", method_ran or "-")
    mid = _SAFE.sub("_", model) if model else "-"
    return f"{PREFIX} method={m} model={mid} version=1"


def comment(label: str, license_: Optional[str]) -> str:
    lic = f" ({license_})" if license_ else ""
    return (f"# Pre-optimized with {label}{lic}. Starting geometry only, not a CRYSTAL result.\n"
            "# Run a CRYSTAL OPT from this geometry before SP/FREQ/BAND/DOSS.")


def parse(text: str) -> Optional[Dict[str, str]]:
    """{method, model, version} from the first marker line in the first 60 lines."""
    for ln in text.splitlines()[:60]:
        m = _LINE.match(ln.strip())
        if m:
            out = {}
            for tok in m.group(1).split():
                k, _, v = tok.partition("=")
                if k in ("method", "model", "version"):
                    out[k] = v
            if out.get("method"):
                return out
    return None


def read(path) -> Optional[Dict[str, str]]:
    try:
        with open(path, errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(60))
    except OSError:
        return None
    return parse(head)


def is_ff_ml(info: Optional[Dict[str, str]]) -> bool:
    return bool(info) and info.get("method") not in (None, "", "springs")


def convert_warning(cif_name: str, info: Dict[str, str], calc_type: str) -> Optional[str]:
    """The `mace convert` / Setup warning, or None for an OPT."""
    if not is_ff_ml(info) or str(calc_type).upper() == "OPT":
        return None
    return (f"warning: {cif_name} is an FF/ML pre-optimized starting geometry ({info['method']}); "
            f"a {str(calc_type).upper()} calculation on it is not meaningful. Run a CRYSTAL OPT first.")
