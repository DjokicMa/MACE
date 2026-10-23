"""The transport through the official ``mcp`` Python package (optional).

``mace mcp --transport auto`` (the default) uses it when ``import mcp``
works, so the protocol versions and features that package follows (newer
protocol eras included) come with it; the tools, their checks and their
answers are the same Dispatcher as the built-in transport's. Both the 1.x
API (decorators) and the 2.x API (``on_*`` handlers) are supported; any
other shape, or a failed import, falls back to the built-in transport.
"""
from __future__ import annotations

import functools
import inspect
import io
import os
import sys
import threading
from typing import Any, Callable, Optional

from .server import STRUCTURED_SINCE, Dispatcher, throttle


class SdkUnavailable(ImportError):
    pass


def load():
    """(mcp.types, Server, stdio_server, api) where api is 1 or 2; SdkUnavailable otherwise."""
    try:
        import anyio  # noqa: F401
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
    except Exception as exc:
        raise SdkUnavailable(f"the mcp package is not usable here ({type(exc).__name__}: {exc})")
    params = inspect.signature(Server.__init__).parameters
    if "on_list_tools" in params and "on_call_tool" in params:
        return types, Server, stdio_server, 2
    if all(hasattr(Server, m) for m in ("list_tools", "call_tool", "list_resources", "read_resource")):
        return types, Server, stdio_server, 1
    raise SdkUnavailable("this version of the mcp package has an API MACE does not know")


def version() -> Optional[str]:
    try:
        from importlib.metadata import version as _v
        return _v("mcp")
    except Exception:
        return None


def _reporter(send) -> Optional[Callable[[float, str], None]]:
    """A progress callback for the worker thread: ``send(fraction, message)``
    (a coroutine function) run on the server's event loop, throttled."""
    if send is None:
        return None
    import anyio.from_thread

    def emit(fraction: float, message: str) -> None:
        try:
            anyio.from_thread.run(send, fraction, message)
        except Exception:
            pass                                        # progress is a courtesy; never fail the call
    return throttle(emit)


async def _threaded(fn: Callable[..., Any], *args, progress=None, **kwargs) -> Any:
    """``fn`` in a worker thread; a cancel from the client sets its cancel event."""
    import anyio
    cancel = threading.Event()
    call = functools.partial(fn, *args, cancel_event=cancel, on_progress=_reporter(progress), **kwargs)
    try:
        try:
            return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
        except TypeError:                                   # anyio < 4.1
            return await anyio.to_thread.run_sync(call, cancellable=True)
    except anyio.get_cancelled_exc_class():
        cancel.set()
        raise


def build(dispatcher: Dispatcher):
    types, Server, _stdio, api = load()
    from mace import __version__
    from . import resources
    cfg = dispatcher.config
    dispatcher.protocol_version = max(dispatcher.protocol_version, STRUCTURED_SINCE)

    def contents(uri: str):
        return resources.read(cfg, uri)

    if api == 2:
        async def on_list_tools(ctx, params):
            return types.ListToolsResult.model_validate({"tools": dispatcher.list_tools()})

        async def on_call_tool(ctx, params):
            session = getattr(ctx, "session", None)
            send = None
            if session is not None and hasattr(session, "report_progress"):
                async def send(fraction, message):
                    await session.report_progress(fraction, 1.0, message or None)
            res = await _threaded(dispatcher.call_tool, params.name, params.arguments or {}, progress=send)
            return types.CallToolResult.model_validate(res)

        async def on_list_resources(ctx, params):
            return types.ListResourcesResult.model_validate({"resources": resources.listing(cfg)})

        async def on_list_resource_templates(ctx, params):
            return types.ListResourceTemplatesResult.model_validate({"resourceTemplates": []})

        async def on_read_resource(ctx, params):
            return types.ReadResourceResult.model_validate({"contents": contents(str(params.uri))})

        return Server("mace", version=__version__, instructions=dispatcher.instructions(),
                      on_list_tools=on_list_tools, on_call_tool=on_call_tool, on_list_resources=on_list_resources,
                      on_list_resource_templates=on_list_resource_templates, on_read_resource=on_read_resource)

    try:
        server = Server("mace", version=__version__, instructions=dispatcher.instructions())
    except TypeError:
        server = Server("mace")

    @server.list_tools()
    async def _list_tools():
        return [types.Tool.model_validate(t) for t in dispatcher.list_tools()]

    try:
        deco = server.call_tool(validate_input=False)       # the Dispatcher checks the arguments itself
    except TypeError:
        deco = server.call_tool()

    @deco
    async def _call_tool(name, arguments):
        send = None
        try:
            rc = server.request_context
            token = getattr(rc.meta, "progressToken", None) if rc.meta is not None else None
            if token is not None:
                async def send(fraction, message):
                    await rc.session.send_progress_notification(token, fraction, 1.0, message=message or None)
        except Exception:
            send = None
        res = await _threaded(dispatcher.call_tool, name, arguments or {}, progress=send)
        return types.CallToolResult.model_validate(res)

    @server.list_resources()
    async def _list_resources():
        return [types.Resource.model_validate(r) for r in resources.listing(cfg)]

    @server.read_resource()
    async def _read_resource(uri):
        from mcp.server.lowlevel.helper_types import ReadResourceContents
        return [ReadResourceContents(content=c["text"], mime_type=c.get("mimeType")) for c in contents(str(uri))]

    return server


def serve(dispatcher: Dispatcher) -> int:
    """Serve on stdin/stdout through the official package until end of input."""
    import anyio
    _types, _Server, stdio_server, _api = load()
    server = build(dispatcher)
    # the same rule as the built-in transport: the protocol on a private copy of fd 1,
    # everything else that writes to fd 1 or sys.stdout lands on stderr
    try:
        sys.stdout.flush()
    except Exception:
        pass
    fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    proto_out = io.TextIOWrapper(os.fdopen(fd, "wb"), encoding="utf-8", newline="\n", line_buffering=True)
    proto_in = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="\n")
    sys.stdout = sys.stderr

    async def main():
        async with stdio_server(stdin=anyio.wrap_file(proto_in), stdout=anyio.wrap_file(proto_out)) as (r, w):
            await server.run(r, w, server.create_initialization_options())

    anyio.run(main)
    return 0
