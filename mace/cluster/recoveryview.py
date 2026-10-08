"""The timeout-recovery settings in effect for one job folder, read-only.

The queue manager runs in the job's folder, so the file that counts is
``<WorkDir>/recovery_config.yaml`` (else MACE's packaged one), laid over the
built-in defaults. This module is given that folder explicitly: it never
looks at the current directory (the GUI server's or the shell's).

* With full MACE installed (the GUI server) it uses the recovery engine's own
  merge (``ErrorRecoveryEngine.load_recovery_config``) for that path, without
  opening a database.
* With PyYAML but without the rest of MACE's dependencies it reads the file
  and lays its ``timeout_error`` entry over the built-in values.
* Without PyYAML (a bare module Python) it uses the built-in values and says
  so. A drift test keeps ``BUILTIN_TIMEOUT`` equal to the engine's defaults
  and the packaged file.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Optional

BUILTIN_TIMEOUT: Dict[str, Any] = {
    "handler": "timeout_handler",
    "walltime_factor": 2.0,
    "max_walltime": "7-00:00:00",
    "max_retries": 2,
    "resubmit_delay": 900,
    "enable_restart": True,
}
CONFIG_NAME = "recovery_config.yaml"
PACKAGED = Path(__file__).resolve().parent.parent / "config" / CONFIG_NAME
NO_YAML_NOTE = ("(using MACE's built-in recovery defaults; PyYAML is not installed here, so a "
                "recovery_config.yaml is not read)")


def config_path_for(job_dir: Optional[Path]) -> Optional[Path]:
    if job_dir is not None:
        p = Path(job_dir) / CONFIG_NAME
        if p.is_file():
            return p
    return PACKAGED if PACKAGED.is_file() else None


def _engine_timeout(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The engine's own merge for this path, or None when MACE's recovery
    module cannot be imported here."""
    try:
        import yaml  # noqa: F401  (the engine needs it; checked before importing the engine)
        import importlib.util
        import contextlib
        import io
        for dep in ("numpy",):
            if importlib.util.find_spec(dep) is None:
                return None
        with contextlib.redirect_stdout(io.StringIO()):      # import-time notices never reach a CLI's output
            from mace.recovery.recovery import ErrorRecoveryEngine
    except BaseException:                 # ImportError, or the module's sys.exit on a missing dependency
        return None
    shim = types.SimpleNamespace(config_path=path, known_handlers=ErrorRecoveryEngine.known_handlers)
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = ErrorRecoveryEngine.load_recovery_config(shim)
    except Exception:
        return None
    return dict((cfg.get("error_recovery") or {}).get("timeout_error") or {})


def timeout_config(job_dir: Optional[Path], use_engine: bool = True) -> Dict[str, Any]:
    """{values, path, source, note} for the job folder's timeout recovery."""
    path = config_path_for(job_dir)
    where = ("job folder" if path is not None and job_dir is not None and path.parent == Path(job_dir)
             else "MACE default" if path is not None else "built-in")
    if use_engine:
        vals = _engine_timeout(path)
        if vals is not None:
            return {"values": vals, "path": str(path) if path else None, "source": where, "note": None,
                    "via": "engine"}
    try:
        import yaml
    except ImportError:
        return {"values": dict(BUILTIN_TIMEOUT), "path": None, "source": "built-in", "note": NO_YAML_NOTE,
                "via": "builtin"}
    vals = dict(BUILTIN_TIMEOUT)
    if path is not None:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            entry = ((data.get("error_recovery") or {}).get("timeout_error") or {}) if isinstance(data, dict) else {}
            if isinstance(entry, dict):
                if entry.get("handler") == "manual_escalation":
                    entry = {**entry, "max_retries": 0, "handler": BUILTIN_TIMEOUT["handler"]}
                for k, v in entry.items():
                    if k in BUILTIN_TIMEOUT:
                        vals[k] = v
        except Exception as exc:
            return {"values": vals, "path": str(path), "source": where,
                    "note": f"(could not read {path.name}: {exc}; using the built-in values)", "via": "yaml"}
    return {"values": vals, "path": str(path) if path else None, "source": where, "note": None, "via": "yaml"}
