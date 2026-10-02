"""The HSE-type screened-hybrid crash on odd-electron open-shell systems.

Measured on HPCC (CRYSTAL/23-intel-2023a, 2026-10-01/02): HSEsol-3c, HSE06
and HSEsol (HJS screened exchange) crash on any odd-electron open-shell
system, charged or not, as a molecule (0D) and in a 3D box alike: "DIIS TEST:
NaN" at SCF cycle 1-3, then every rank dies with a segfault, a malloc assert
or SIGKILL ("KILLED BY SIGNAL: 11 / 6 / 9"). The same decks with PBE0 or
PBEsol0 converged. Those signal lines read as an out-of-memory crash, but a
memory bump cannot help, and MACE never changes the functional itself - so
the run is reported as this error and left for the user to decide.
"""
import re
from typing import Optional

ERROR_TYPE = "screened_hybrid_odd_error"

_ELECTRONS = re.compile(r"N\. OF ELECTRONS PER CELL\s+(\d+)")
_FUNCTIONAL = re.compile(r"\(EXCHANGE\)\[CORRELATION\] FUNCTIONAL:\((HSE[^ )]*)", re.IGNORECASE)
_HJS = re.compile(r"\*\*\*\* RDDFTP \*\*\*\* (HSE\S*)-HJS ACTIVE", re.IGNORECASE)
_NAN = re.compile(r"DIIS TEST:\s+NaN", re.IGNORECASE)
_KILLED = re.compile(r"KILLED BY SIGNAL|Segmentation fault|malloc", re.IGNORECASE)


def odd_electron_crash(content: str) -> Optional[str]:
    """Why this .out is the screened-hybrid odd-electron crash, or None.

    All of: an HSE-type functional, an unrestricted open-shell SCF, an odd
    electron count, and an SCF that blew up (DIIS NaN) or ranks killed by a
    signal - and no finished run.
    """
    if "TOTAL CPU TIME" in content or "OPT END" in content:
        return None
    if "UNRESTRICTED OPEN SHELL" not in content:
        return None
    m = _ELECTRONS.search(content)
    if not m or int(m.group(1)) % 2 == 0:
        return None
    f = _HJS.search(content) or _FUNCTIONAL.search(content)
    if not f:
        return None
    if not (_NAN.search(content) or _KILLED.search(content)):
        return None
    functional = f.group(1)
    return (f"{functional} (HSE-type screened hybrid) on an odd-electron ({m.group(1)}) open-shell "
            "system: the SCF blew up (DIIS NaN) and the ranks crashed. On CRYSTAL23 HSEsol-3c, HSE06 "
            "and HSEsol crashed this way on every odd-electron open-shell system measured, 0D and "
            "3D, while PBE0 and PBEsol0 converged. More memory does not help, so MACE does not "
            "resubmit it and keeps the functional as asked; choosing PBE0 or PBEsol0 is up to you")
