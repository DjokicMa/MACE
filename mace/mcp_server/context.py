"""What a tool call works with: the served folder, the confirmation rule,
cancellation and progress, safe file writes, JSON conversion and the CLI.

Paths: every path a client gives is read relative to the served folder
(``mace mcp --root``, the current folder by default) and must stay inside it,
symbolic links resolved, unless the server runs with
``--allow-outside-root``. Paths in results are relative to the folder when
they are inside it.

Writes: a new file is written atomically; replacing an existing file needs
``confirm: true`` and keeps the old version in the GUI's File History (the
same store the GUI uses, so it can be restored from there). Outputs, job
logs, wavefunctions, databases and the decks of queued jobs are never
written (``mace.gui.saving.protected_reason``).
"""
from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

MAX_RESULT_BYTES = 4 << 20          # a result above this is refused with a hint to ask for less
CLI_TIMEOUT_S = 600.0


class ToolError(Exception):
    """A failure the client should see as the tool's own answer (isError)."""

    def __init__(self, message: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.data = data or {}


class ConfirmRequired(ToolError):
    """The call would submit, cancel, replace or delete: nothing was done."""

    def __init__(self, action: str, preview: Optional[Dict[str, Any]] = None):
        super().__init__(f"Not done: {action}. Ask the user, then call again with \"confirm\": true.",
                         {"confirmation_required": True, "action": action, "preview": preview or {}})


class Canceled(ToolError):
    def __init__(self):
        super().__init__("canceled by the client")


@dataclass
class ServerConfig:
    root: Path
    read_only: bool = False
    allow_outside_root: bool = False
    max_atoms_listed: int = 500

    def __post_init__(self):
        self.root = Path(self.root).expanduser().resolve()


class ToolContext:
    """One call's context. Also the ``ctx`` MACE's builders and ``clirun``
    take: ``canceled``, ``check()``, ``progress(fraction, message)``,
    ``set_pid(pid)``."""

    def __init__(self, config: ServerConfig, confirmed: bool = False,
                 cancel_event: Optional[threading.Event] = None,
                 on_progress: Optional[Callable[[float, str], None]] = None):
        self.config = config
        self.confirmed = bool(confirmed)
        self._cancel = cancel_event or threading.Event()
        self._on_progress = on_progress
        self.pid: Optional[int] = None
        self.notes: List[str] = []

    # -------------------------------------------------- cancel / progress (builders' ctx protocol)
    @property
    def canceled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise Canceled()

    def progress(self, fraction: float, message: str = "", force: bool = False) -> None:
        if self._on_progress is not None:
            try:
                self._on_progress(max(0.0, min(1.0, float(fraction))), str(message or ""))
            except Exception:
                pass

    def set_pid(self, pid: Optional[int]) -> None:
        self.pid = pid

    # -------------------------------------------------- confirmation
    def require_confirm(self, action: str, preview: Optional[Dict[str, Any]] = None) -> None:
        if not self.confirmed:
            raise ConfirmRequired(action, preview)

    # -------------------------------------------------- paths
    @property
    def root(self) -> Path:
        return self.config.root

    def path(self, value: Any, must_exist: bool = True, kind: Optional[str] = None) -> Path:
        """The client's path, resolved inside the served folder."""
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            raise ToolError("a path is a non-empty string")
        raw = Path(os.path.expanduser(str(value).strip()))
        p = raw if raw.is_absolute() else self.root / raw
        try:
            p = p.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ToolError(f"{value}: {exc}")
        if not self.config.allow_outside_root and not self.inside(p):
            raise ToolError(f"{value} is outside the served folder {self.root} "
                            "(start the server with --root covering it, or --allow-outside-root)")
        if must_exist and not p.exists():
            raise ToolError(f"{self.show(p)} does not exist")
        if kind == "file" and p.exists() and not p.is_file():
            raise ToolError(f"{self.show(p)} is not a file")
        if kind == "dir" and p.exists() and not p.is_dir():
            raise ToolError(f"{self.show(p)} is not a folder")
        return p

    def inside(self, p: Path) -> bool:
        return p == self.root or self.root in p.parents

    def show(self, p: Any) -> str:
        """``p`` relative to the served folder when inside it, else absolute."""
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root).as_posix() or "."
        except (ValueError, OSError):
            return str(p)

    # -------------------------------------------------- writes
    def write_file(self, target: Path, data: bytes, reason: str = "Written by mace mcp") -> Dict[str, Any]:
        """Write ``data`` to ``target`` (inside the served folder): a new file
        directly; an existing one only with confirm, its old version kept in
        File History. Refuses the files MACE never writes."""
        if self.config.read_only:
            raise ToolError("the server runs with --read-only")
        target = Path(target)
        from mace.gui import saving
        why = saving.protected_reason(target)
        if why:
            raise ToolError(why)
        if target.is_symlink():
            raise ToolError(f"{self.show(target)} is a symbolic link; MACE does not write through links")
        existed = target.exists()
        if existed:
            if target.is_dir():
                raise ToolError(f"{self.show(target)} is a folder")
            self.require_confirm(f"replace the existing file {self.show(target)}",
                                 {"path": self.show(target), "size": target.stat().st_size, "new_size": len(data)})
        target.parent.mkdir(parents=True, exist_ok=True)
        from mace.gui import history
        if history.root() is None:
            history.configure(self.root)
        saving.write_atomic(target, data, reason=reason, route="mcp", kind="input")
        return {"path": self.show(target), "bytes": len(data), "replaced": existed}

    # -------------------------------------------------- the CLI
    def run_cli(self, args: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[float] = CLI_TIMEOUT_S,
                env: Optional[Dict[str, str]] = None, cancelable: bool = True,
                stdin_text: Optional[str] = None) -> Dict[str, Any]:
        """``mace <args>`` (this checkout's mace_cli) in a child process: nice
        10, one math thread, stdin closed, output captured (never on the
        protocol stream). Returns {code, stdout, stderr, argv}."""
        from mace.gui import clirun
        argv = clirun.mace_argv("--no-banner", *[str(a) for a in args])
        try:
            res = clirun.run(argv, Path(cwd or self.root), ctx=self, timeout=timeout, env=env,
                             cancelable=cancelable, stdin_text=stdin_text)
        except clirun.CliCanceled:
            raise Canceled()
        except clirun.CliTimeout as exc:
            raise ToolError(str(exc))
        return {"code": res.returncode, "stdout": res.stdout, "stderr": res.stderr,
                "argv": ["mace", *[str(a) for a in args]]}


# ---------------------------------------------------------------- JSON
def jsonable(obj: Any, depth: int = 0) -> Any:
    """``obj`` as plain JSON values: numpy arrays and scalars to lists and
    numbers, paths to strings, NaN and infinities to None (strict JSON)."""
    if depth > 64:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v, depth + 1) for v in obj]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(obj, np.ndarray):
            return jsonable(obj.tolist(), depth + 1)
        if isinstance(obj, np.generic):
            return jsonable(obj.item(), depth + 1)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__") and not callable(obj):
        return {k: jsonable(v, depth + 1) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def tail(text: str, limit: int = 6000) -> str:
    text = text or ""
    return text if len(text) <= limit else "...\n" + text[-limit:]
