"""How much of the machine ``mace gui`` may use (spec section 9).

The workstation it runs on is shared (training jobs on the GPUs, other
users on cluster dev nodes), so every limit here defaults low and is a
setting the user can raise:

* CPU: the process pool (pool.py) gets at most a quarter of the usable
  cores, 1 to 8, and no more than SLURM gave the job; its workers run at
  nice 10, and it shares that cap with the request threads (compute.GATE).
  ``max_workers`` in gui.json or ``MACE_GUI_MAX_WORKERS`` override it.
* RAM: the caches are capped by bytes, 512 MB by default (``cache_mb``,
  ``MACE_GUI_CACHE_MB``), and never above a quarter of physical memory.
* GPU: the server uses none today; ``gpu_device`` / ``MACE_GUI_GPU`` names
  the device MACE's own GPU features use, GPU 0 by default ("none" keeps
  everything on the CPU).

A value stored in gui.json is only checked for its type and wide bounds
when saved (the file is shared by every host the user logs into); the
host's own limits apply here, when the value is used. /api/resources adds
what the pool and the shared CPU budget hold right now.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

WORKER_NICENESS = 10
DEFAULT_CACHE_MB = 512
MIN_CACHE_MB = 64
MAX_AUTO_WORKERS = 8
DEFAULT_GPU = "0"


def cores() -> int:
    """The cores this process may run on (affinity, not the machine's count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def slurm_cpus() -> Optional[int]:
    v = os.environ.get("SLURM_CPUS_ON_NODE", "")
    return int(v) if v.isdigit() and int(v) > 0 else None


def _host_max() -> int:
    s = slurm_cpus()
    return min(cores(), s) if s else cores()


def worker_cap() -> int:
    """The default pool size: a quarter of the usable cores, 1 to 8, within SLURM's."""
    s = slurm_cpus()
    return max(1, min(cores() // 4, MAX_AUTO_WORKERS, s if s else MAX_AUTO_WORKERS))


def _int_env(name: str) -> Optional[int]:
    try:
        v = int(float(os.environ[name]))
    except (KeyError, ValueError, OverflowError):   # unset, not a number, or inf/1e400
        return None
    return v


def effective_workers(stored: Any = None) -> int:
    """MACE_GUI_MAX_WORKERS, else gui.json's max_workers if above 0, else
    the cap; a chosen value is clamped to what this host (and SLURM) has."""
    env = _int_env("MACE_GUI_MAX_WORKERS")
    if env is not None and env > 0:
        return max(1, min(env, _host_max()))
    if isinstance(stored, int) and not isinstance(stored, bool) and stored > 0:
        return max(1, min(stored, _host_max()))
    return worker_cap()


def physical_ram_mb() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (AttributeError, ValueError, OSError):
        return None


def _cache_ceiling_mb() -> int:
    ram = physical_ram_mb()
    return max(MIN_CACHE_MB, ram // 4) if ram else 65536


def cache_mb(stored: Any = None) -> int:
    """MACE_GUI_CACHE_MB, else gui.json's cache_mb, else 512; clamped to
    64 MB .. a quarter of physical RAM."""
    try:
        env = float(os.environ["MACE_GUI_CACHE_MB"])
    except (KeyError, ValueError):
        env = None
    if env is not None and env > 0:
        want = env
    elif isinstance(stored, (int, float)) and not isinstance(stored, bool) and stored > 0:
        want = stored
    else:
        want = DEFAULT_CACHE_MB
    return int(max(MIN_CACHE_MB, min(want, _cache_ceiling_mb())))


def cache_limit_bytes(stored: Any = None) -> int:
    return cache_mb(stored) * 1024 * 1024


def gpu_device(stored: Any = None, env: Optional[Dict[str, str]] = None) -> str:
    """MACE_GUI_GPU, else gui.json's gpu_device, else "0". Inside a SLURM job
    "0" is index 0 of the job's CUDA_VISIBLE_DEVICES (mlip_client maps it);
    a job given no GPU (none or an empty list) defaults to "none" (CPU),
    never the node's physical GPU 0."""
    e = os.environ if env is None else env
    pinned = e.get("MACE_GUI_GPU", "").strip()
    if pinned:
        return pinned
    if isinstance(stored, str) and stored:
        return stored
    if e.get("SLURM_JOB_ID"):
        return DEFAULT_GPU if (e.get("CUDA_VISIBLE_DEVICES") or "").strip() else "none"
    return DEFAULT_GPU


def pinned_by_env() -> list:
    """The settings an environment variable decides (the Settings dialog says so)."""
    out = []
    if (_int_env("MACE_GUI_MAX_WORKERS") or 0) > 0:
        out.append("max_workers")
    try:
        if float(os.environ["MACE_GUI_CACHE_MB"]) > 0:
            out.append("cache_mb")
    except (KeyError, ValueError):
        pass
    if os.environ.get("MACE_GUI_GPU", "").strip():
        out.append("gpu_device")
    return out


def report(stored: Dict[str, Any], cache_used_bytes: int = 0) -> Dict[str, Any]:
    """What /api/info shows: stored and effective values side by side."""
    return {"cores": cores(), "slurm_cpus": slurm_cpus(),
            "workers_cap": worker_cap(), "workers_stored": stored.get("max_workers", 0),
            "workers_effective": effective_workers(stored.get("max_workers")),
            "worker_niceness": WORKER_NICENESS, "pool_started": False,
            "cache_mb": cache_mb(stored.get("cache_mb")), "cache_mb_stored": stored.get("cache_mb", DEFAULT_CACHE_MB),
            "cache_used_mb": round(cache_used_bytes / (1024 * 1024), 1), "ram_mb": physical_ram_mb(),
            "gpu_device": gpu_device(stored.get("gpu_device")), "gpu_device_stored": stored.get("gpu_device", DEFAULT_GPU),
            "pinned_by_env": pinned_by_env()}
