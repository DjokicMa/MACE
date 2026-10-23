"""The MCP methods, independent of the transport.

``Dispatcher.handle(message)`` takes one decoded JSON-RPC message and returns
the response object (or None for a notification); ``call_tool`` is the part
both transports share with the official SDK adapter (``sdk.py``).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

from . import PROTOCOL_VERSIONS
from .context import MAX_RESULT_BYTES, ConfirmRequired, ServerConfig, ToolContext, ToolError, jsonable
from .registry import ArgumentError, Registry, check

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603
STRUCTURED_SINCE = "2025-06-18"     # structuredContent in tool results

INSTRUCTIONS = """MACE prepares, runs and reads CRYSTAL calculations.
Paths are relative to the served folder ({root}); tools never reach outside it.
Typical flow: files_list -> structure_inspect -> (structure_supercell / structure_slab /
structure_clean) -> setup_templates -> setup_d12 (preview, then output_dir to write) ->
deck_preflight -> job_submit -> job_status -> outputs_scan / output_summary /
output_properties -> db_search / db_material.
Tools that submit or cancel jobs, or replace files, do nothing until called again with
"confirm": true; ask the user first. Without confirm they describe what they would do.
A preopt/Clean geometry is a starting point, not a result: run a CRYSTAL OPT from it."""


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def build_registry(config: ServerConfig, log=None) -> Registry:
    """MACE's own tools, the plugin lines of plugins.py, then the site's own
    (``MACE_MCP_PLUGINS``: comma-separated ``module:function`` or
    ``name=module:function`` entries, on the server's PYTHONPATH)."""
    from .tools import register_all
    from . import plugins
    from .registry import load_plugins, plugin
    reg = Registry()
    register_all(reg)
    specs = list(plugins.PLUGINS)
    for entry in os.environ.get("MACE_MCP_PLUGINS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, target = entry.rpartition("=")
        name = name.strip() or target.rpartition(":")[2].strip().lower()
        specs.append(plugin(name, target.strip(), group="site plugins"))
    load_plugins(reg, specs, log=log)
    return reg


class Dispatcher:
    def __init__(self, config: ServerConfig, registry: Optional[Registry] = None, debug: bool = False, log=None):
        self.config = config
        self.registry = registry if registry is not None else build_registry(config, log=log)
        self.debug = debug
        self.protocol_version = PROTOCOL_VERSIONS[0]
        self.client_info: Dict[str, Any] = {}
        self.initialized = False
        self._lock = threading.Lock()

    # -------------------------------------------------------------- JSON-RPC
    def handle(self, msg: Any, cancel_event: Optional[threading.Event] = None,
               on_progress: Optional[Callable[[float, str], None]] = None) -> Optional[Dict[str, Any]]:
        """The response to one message (None for a notification or a response)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return error_response(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST,
                                  "not a JSON-RPC 2.0 message")
        method = msg.get("method")
        is_request = "id" in msg
        if method is None:
            return None                                      # a response to a request we never send
        if not isinstance(method, str):
            return error_response(msg.get("id"), INVALID_REQUEST, "method must be a string") if is_request else None
        params = msg.get("params")
        if params is not None and not isinstance(params, dict):
            return error_response(msg.get("id"), INVALID_PARAMS, "params must be an object") if is_request else None
        params = params or {}
        if not is_request:
            self.notify(method, params)
            return None
        try:
            result = self.request(method, params, cancel_event=cancel_event, on_progress=on_progress)
        except RpcError as exc:
            return error_response(msg["id"], exc.code, exc.message, exc.data)
        except Exception as exc:                              # a bug: the client gets a clean error
            self.log_exc(exc)
            return error_response(msg["id"], INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        if method == "notifications/initialized":
            self.initialized = True
        # the cancel notification is handled by the transport (it owns the running calls)

    def request(self, method: str, params: Dict[str, Any], cancel_event=None, on_progress=None) -> Dict[str, Any]:
        if method == "initialize":
            return self.initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise RpcError(INVALID_PARAMS, "tools/call needs a tool name")
            if self.tool(name) is None:
                raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}" + (
                    " (the server runs with --read-only)" if self.registry.get(name) is not None else ""))
            return self.call_tool(name, params.get("arguments"), cancel_event=cancel_event, on_progress=on_progress)
        if method == "resources/list":
            from . import resources
            return {"resources": resources.listing(self.config)}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "resources/read":
            from . import resources
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise RpcError(INVALID_PARAMS, "resources/read needs a uri")
            try:
                return {"contents": resources.read(self.config, uri)}
            except KeyError:
                raise RpcError(-32002, f"Resource not found: {uri}", {"uri": uri})
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        asked = params.get("protocolVersion")
        self.protocol_version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        self.client_info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        from mace import __version__
        return {"protocolVersion": self.protocol_version,
                "capabilities": {"tools": {"listChanged": False},
                                 "resources": {"subscribe": False, "listChanged": False}},
                "serverInfo": {"name": "mace", "title": "MACE - Mendoza Automated CRYSTAL Engine",
                               "version": __version__},
                "instructions": self.instructions()}

    def instructions(self) -> str:
        text = INSTRUCTIONS.format(root=self.config.root)
        if self.config.read_only:
            text += "\nThis server is read-only: tools that write files or touch jobs are not offered."
        return text

    # -------------------------------------------------------------- tools
    def tool(self, name: str):
        t = self.registry.get(name)
        if t is None or (self.config.read_only and t.writes):
            return None
        return t

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.listing() for t in self.registry.tools(read_only=self.config.read_only)]

    def call_tool(self, name: str, arguments: Any, cancel_event: Optional[threading.Event] = None,
                  on_progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """The tools/call result: the tool's answer, or its failure as
        ``isError`` (bad arguments included, so the client can correct them)."""
        tool = self.tool(name)
        if tool is None:
            return self.failure(f"Unknown tool: {name}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self.failure("arguments must be an object")
        try:
            args = check(tool.schema, dict(arguments))
        except ArgumentError as exc:
            return self.failure(f"Invalid arguments for {name}: {exc}")
        confirmed = args.pop("confirm", False) is True
        ctx = ToolContext(self.config, confirmed=confirmed, cancel_event=cancel_event, on_progress=on_progress)
        try:
            if tool.confirm == "always" and not confirmed and not tool.previews:
                raise ConfirmRequired(f"run {name}", {"arguments": jsonable(args)})
            result = jsonable(tool.handler(ctx, args))
        except ToolError as exc:
            return self.failure(str(exc), jsonable(exc.data) if exc.data else None)
        except (ValueError, KeyError, FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError,
                ImportError) as exc:
            self.log_exc(exc, quiet=True)
            return self.failure(_plain(exc))
        except Exception as exc:
            self.log_exc(exc)
            return self.failure(f"{type(exc).__name__}: {exc}")
        return self.success(result)

    def success(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, str):
            text, structured = result, None
        else:
            text = json.dumps(result, indent=1, ensure_ascii=False)
            structured = result if isinstance(result, dict) else {"result": result}
        if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
            return self.failure(f"the result is {len(text) // 1024} KB, over the {MAX_RESULT_BYTES >> 20} MB limit; "
                                "ask for less (a limit, a frame, fewer files, full=false)")
        out: Dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
        if structured is not None and self.protocol_version >= STRUCTURED_SINCE:
            out["structuredContent"] = structured
        return out

    def failure(self, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = message if not data else message + "\n" + json.dumps(data, indent=1, ensure_ascii=False)
        out: Dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": True}
        if data and self.protocol_version >= STRUCTURED_SINCE:
            out["structuredContent"] = {"error": message, **data}
        return out

    def log_exc(self, exc: BaseException, quiet: bool = False) -> None:
        if self.debug or not quiet:
            sys.stderr.write(f"mace mcp: {type(exc).__name__}: {exc}\n")
            if self.debug:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            sys.stderr.flush()


def throttle(emit: Callable[[float, str], None], min_interval: float = 0.25) -> Callable[[float, str], None]:
    """``emit`` at most every ``min_interval`` s and only for a larger fraction
    (the protocol wants progress to increase); 1.0 always goes out."""
    import time
    last = {"t": 0.0, "f": -1.0}

    def fn(fraction: float, message: str = "") -> None:
        now = time.monotonic()
        if fraction <= last["f"] or (now - last["t"] < min_interval and fraction < 1.0):
            return
        last["t"], last["f"] = now, fraction
        emit(fraction, message)
    return fn


def _plain(exc: BaseException) -> str:
    if isinstance(exc, KeyError):
        return f"not found: {exc.args[0] if exc.args else exc}"
    if isinstance(exc, ImportError):
        return f"{exc} (an optional package is missing; install it to use this tool)"
    return str(exc) or type(exc).__name__


def error_response(msg_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}
