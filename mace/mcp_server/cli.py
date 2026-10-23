"""``mace mcp``: serve MACE's tools to an MCP client over stdin/stdout.

The client starts this command and talks JSON-RPC to it; a person runs it
with --list-tools, --config or --check. Standard library only until a tool
runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

EPILOG = """examples:
  mace mcp                              serve the current folder (an MCP client starts this)
  mace mcp --root ~/calcs --read-only   only reading tools, inside ~/calcs
  mace mcp --list-tools                 the tools and what each one does
  mace mcp --config --root ~/calcs      the JSON block to add to an MCP client's server list
  mace mcp --check                      start the server in-process and list its tools (a self-test)

Tools that submit or cancel jobs, or replace files, do nothing until the client
calls them again with "confirm": true. Paths stay inside --root.
Guide: docs/mcp.md"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace mcp", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Serve MACE's tools (structures, decks, jobs, outputs, materials.db) "
                                            "to an MCP client over stdin/stdout (JSON-RPC 2.0).",
                                epilog=EPILOG, allow_abbrev=False)
    p.add_argument("--root", metavar="DIR", default=None,
                   help="the folder the tools work in; paths outside it are refused (default: the current folder)")
    p.add_argument("--read-only", action="store_true",
                   help="offer only the tools that read: no writes, no submit, no cancel")
    p.add_argument("--allow-outside-root", action="store_true",
                   help="accept paths outside --root (absolute paths anywhere you can read)")
    p.add_argument("--transport", choices=("auto", "builtin", "sdk"), default="auto",
                   help="auto: the official mcp package when it is installed, else the built-in one")
    p.add_argument("--workers", type=int, default=2, metavar="N",
                   help="built-in transport: tool calls run at once (default 2)")
    p.add_argument("--list-tools", action="store_true", help="print the tools and exit")
    p.add_argument("--json", action="store_true", help="with --list-tools: the tools/list JSON")
    p.add_argument("--config", action="store_true", help="print a client configuration block and exit")
    p.add_argument("--check", action="store_true", help="initialize the server in-process, list tools, exit")
    p.add_argument("--debug", action="store_true", help="tracebacks on stderr")
    return p


def _config(args):
    from .context import ServerConfig
    root = Path(args.root) if args.root else Path.cwd()
    if not root.is_dir():
        raise SystemExit(f"mace mcp: --root {root} is not a folder")
    return ServerConfig(root=root, read_only=args.read_only, allow_outside_root=args.allow_outside_root)


def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def client_config(cfg, read_only: bool) -> dict:
    cli = Path(__file__).resolve().parents[2] / "mace_cli"
    a = [str(cli), "mcp", "--root", str(cfg.root)]
    if read_only:
        a.append("--read-only")
    return {"mcpServers": {"mace": {"command": sys.executable, "args": a}}}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        _log("mace mcp: error: --workers must be at least 1")
        return 2
    cfg = _config(args)
    if args.config:
        print(json.dumps(client_config(cfg, args.read_only), indent=2))
        return 0
    from .server import Dispatcher
    disp = Dispatcher(cfg, debug=args.debug, log=_log)
    if args.list_tools:
        tools = disp.list_tools()
        if args.json:
            print(json.dumps({"tools": tools, "skipped": [{"name": n, "reason": r} for n, r in disp.registry.skipped]},
                             indent=2))
            return 0
        _print_tools(disp)
        return 0
    if args.check:
        return _check(disp, args)
    from mace import __version__
    n = len(disp.list_tools())
    try:
        if sys.stdin.isatty():
            _log("mace mcp: waiting for an MCP client on stdin (an MCP client starts this command; "
                 "Ctrl+D quits). For people: mace mcp --check, --list-tools, --config.")
    except (AttributeError, ValueError):
        pass
    if args.transport in ("auto", "sdk"):
        from . import sdk
        try:
            sdk.load()
        except sdk.SdkUnavailable as exc:
            if args.transport == "sdk":
                _log(f"mace mcp: error: {exc}; install it with: pip install mcp (or use --transport builtin)")
                return 1
        else:
            _log(f"mace mcp {__version__}: {n} tools for {cfg.root} (mcp package {sdk.version() or '?'}"
                 f"{', read-only' if cfg.read_only else ''})")
            return sdk.serve(disp)
    _log(f"mace mcp {__version__}: {n} tools for {cfg.root} (built-in transport"
         f"{', read-only' if cfg.read_only else ''})")
    from .stdio import serve
    return serve(disp, workers=args.workers)


def _print_tools(disp) -> None:
    rows = disp.registry.tools(read_only=disp.config.read_only)
    groups = {}
    for t in rows:
        groups.setdefault(t.group, []).append(t)
    for g, tools in groups.items():
        print(g)
        for t in tools:
            flags = []
            if t.writes:
                flags.append("writes")
            if t.confirm == "always":
                flags.append("needs confirm")
            elif t.confirm == "on_overwrite":
                flags.append("confirm to replace")
            if t.origin != "mace":
                flags.append(t.origin)
            first = t.description.split(". ")[0].rstrip(".")
            print(f"  {t.name:<22}{first}" + (f"  [{', '.join(flags)}]" if flags else ""))
    for name, why in disp.registry.skipped:
        print(f"not loaded: {name} - {why}")
    print(f"\n{len(rows)} tools; resources: mace://templates/d12/*, mace://templates/d3/*, mace://docs/mcp")


def _check(disp, args) -> int:
    init = disp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "mace-mcp-check", "version": "1"}}})
    disp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = disp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    res = disp.handle({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    ok = "result" in init and "result" in tools and "result" in res
    info = init.get("result", {}).get("serverInfo", {})
    print(f"server: {info.get('name')} {info.get('version')}, protocol {init.get('result', {}).get('protocolVersion')}")
    print(f"root: {disp.config.root}{' (read-only)' if disp.config.read_only else ''}")
    print(f"tools: {len(tools.get('result', {}).get('tools', []))}, "
          f"resources: {len(res.get('result', {}).get('resources', []))}")
    from . import sdk
    try:
        sdk.load()
        print(f"transport: mcp package {sdk.version() or '?'} (auto), built-in available")
    except sdk.SdkUnavailable:
        print("transport: built-in (the mcp package is not installed; it is optional)")
    for name, why in disp.registry.skipped:
        print(f"not loaded: {name} - {why}")
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
