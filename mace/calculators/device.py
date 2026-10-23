"""Which device an ML potential runs on.

``auto`` (the default of ``mace calc``) gives the ML worker one GPU, GPU 0,
unless CUDA_VISIBLE_DEVICES is already set (by the user, SLURM or a site
wrapper: then its first GPU); the worker uses it when torch sees it and the
CPU otherwise. A workstation's second GPU is never taken by default (it may
be busy with someone's training); ``--device cuda:1`` or
CUDA_VISIBLE_DEVICES=1 choose it explicitly. The mapping itself lives in
``mace.preopt.mlip_client.device_mapping`` (one rule for Clean and calc).
The force fields, EQeq, xtb and DFTB+ run on the CPU.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Optional

_GPUS: Optional[List[Dict[str, object]]] = None


def resolve(device: str = "auto", environ=None):
    """(CUDA_VISIBLE_DEVICES for the worker, the worker's --device, a text)."""
    from mace.preopt import mlip_client
    return mlip_client.device_mapping(device, environ)


def gpus(refresh: bool = False) -> List[Dict[str, object]]:
    """The NVIDIA GPUs nvidia-smi lists (index, name, memory in MiB); [] without one."""
    global _GPUS
    if _GPUS is not None and not refresh:
        return _GPUS
    out: List[Dict[str, object]] = []
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            p = subprocess.run([exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
            for line in p.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    out.append({"index": int(parts[0]), "name": parts[1],
                                "memory_mib": int(float(parts[2])) if parts[2].replace(".", "").isdigit() else None})
        except (OSError, subprocess.TimeoutExpired):
            out = []
    _GPUS = out
    return out


def default_text(environ=None) -> str:
    try:
        visible, wdev, text = resolve("auto", environ)
    except ValueError as exc:
        return f"auto: {exc}"
    g = gpus()
    if not g:
        return "auto: the CPU (no NVIDIA GPU found here)"
    if visible and visible.isdigit():
        named = next((x for x in g if x["index"] == int(visible)), None)
        if named:
            text += f" [{named['name']}]"
    return text
