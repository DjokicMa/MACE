"""``mace gui login-url [--port N] [--open]``: a fresh one-time sign-in link
for a running MACE GUI on this machine, without restarting it.

It reads the server's private state file (written by ``mace gui``, 0600,
removed when the server stops), asks the server for a new code with the
token in the ``X-Mace-Token`` header, and prints the link. With several
servers and no ``--port`` it lists them. Standard library only.

A server started by ``mace gui connect`` is reached through that command's
tunnel: on the computer running ``mace gui connect`` its state file names the
forwarded port, so the link printed there works in that computer's browser.
On the node itself the server leaves only a marker without the token, and
login-url says to run it on the connect computer instead.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def state_files() -> List[Path]:
    out: List[Path] = []
    run = os.environ.get("XDG_RUNTIME_DIR")
    if run and os.path.isdir(run):
        out += sorted(Path(run).glob("mace-gui-*.json"))
    from .managed import state_dir
    host = socket.gethostname().split(".")[0] or "host"
    try:
        out += sorted(state_dir().glob(f"url-{host}-*.json"))
    except OSError:
        pass
    return out


def _alive(pid: int) -> bool:
    from .registry import pid_alive          # os.kill(pid, 0) would end the process on Windows
    return pid_alive(pid)


def servers() -> List[Dict[str, Any]]:
    found = []
    for f in state_files():
        try:
            d = json.loads(f.read_text())
            if isinstance(d, dict) and _alive(int(d.get("pid", 0))):
                found.append(d)
        except (OSError, ValueError):
            continue
    return found


def new_link(d: Dict[str, Any]) -> Optional[str]:
    req = urllib.request.Request(d["base"] + "api/login-code", data=b"{}", method="POST",
                                 headers={"X-Mace-Token": d["token"], "X-Requested-With": "mace-gui",
                                          "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=10) as r:
            code = json.loads(r.read()).get("code")
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return None
    return f"{d['base']}?login={code}" if code else None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mace gui login-url",
                                description="Print a fresh one-time sign-in link for a MACE GUI running on this "
                                            "machine (the link signs in once).")
    p.add_argument("--port", type=int, help="the server's port (needed when several run)")
    p.add_argument("--open", action="store_true", help="also open it in a browser on this machine")
    args = p.parse_args(argv)
    found = servers()
    if args.port is not None:
        found = [d for d in found if int(d.get("port", 0)) == args.port or f":{args.port}/" in d.get("base", "")]
    managed = [d for d in found if d.get("managed") and not d.get("token")]
    found = [d for d in found if d.get("token")]
    if not found and managed:
        ports = ", ".join(str(d.get("port")) for d in managed)
        print(f"mace gui login-url: the MACE GUI here (port {ports}) was started by mace gui connect, and your "
              "browser reaches it through that connection's tunnel. Run mace gui login-url on the computer where "
              "mace gui connect runs (or press Enter in its window) for a link that works there.", file=sys.stderr)
        return 1
    if not found:
        print("mace gui login-url: no MACE GUI of yours is running here"
              + (f" on port {args.port}" if args.port else "") + ".", file=sys.stderr)
        return 1
    if len(found) > 1:
        print("Several MACE GUI servers run here; pick one with --port:")
        for d in found:
            print(f"  --port {d.get('port')}  {d.get('root')}")
        return 2
    link = new_link(found[0])
    if not link:
        print("mace gui login-url: the server did not answer.", file=sys.stderr)
        return 1
    print(link)
    if args.open:
        from .appwindow import open_url
        open_url(link)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
