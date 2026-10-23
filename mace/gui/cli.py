"""``mace gui``: start the MACE GUI for a directory.

    mace gui [--dir DIR] [--port P] [--open [--gpu N]]
    mace gui connect ...      (connect.py: start it on a cluster node over ssh)
    mace gui --slurm ...      (slurmsession.py: run it as a batch job on a compute node)

Prints the URL to open (with the session token, valid while this server
runs) and, when the shell is an ssh session (a cluster node), the exact node
and the commands that forward the port to your computer's browser. When
stdout is not a terminal (sbatch, nohup, a log file), the token is not
printed: the full URL goes to a file only you can read, removed when the
server stops. ``--managed`` is the machine-readable mode ``mace gui
connect`` drives over ssh (managed.py).

This module imports the server only inside main(), so ``mace gui connect``
(and ``python -m mace.gui.cli connect``) run with the standard library alone.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import signal
import stat
import socket
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PORT = 8765
# what the hint prints when no gateway is known: a word bash and PowerShell
# take as is (a <gateway> is a redirect there), named in the note under it
GATEWAY_PLACEHOLDER = "LOGIN_HOST"
GATEWAY_RE = re.compile(r"^(?:[A-Za-z0-9._-]{1,64}@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
# a chosen token goes into a URL and a cookie as it is, and the session path
# is derived from it: URL-safe characters only, and long enough not to guess
TOKEN_RE = re.compile(r"[A-Za-z0-9._~-]{16,}")
TOKEN_RULE = ("at least 16 characters, each a letter, a digit or one of . _ ~ - "
              "(python -c \"import secrets; print(secrets.token_urlsafe(24))\" makes one)")
DOCS = "docs/gui.md"
TOOLS = ("connect", "login-url", "diagnostics")


def mace_home() -> str:
    """The MACE install folder (where mace_cli is): what `mace gui connect`
    names when it suggests updating the cluster's MACE."""
    return str(Path(__file__).resolve().parents[2])


STOP_LINES = {
    "browser": "mace gui: stopped from the browser (Sign Out and Stop Server).",
    "idle": "mace gui: stopped after {hours} without use and with no task running (Settings > Server).",
    "cpu-limit": ("mace gui: stopped before this node's CPU-time limit ({minutes} min per process) ended it. "
                  "Start it again for a fresh limit; your drafts are kept in the browser."),
    "session-closed": ("mace gui: stopped because the session that started it closed (ssh without a terminal sends no "
                       "hangup). To keep it running after you log out: nohup mace gui ... & or mace gui --keep."),
    "managed-disconnect": "mace gui: the connection closed and no task was running; stopped.",
    "managed-stop": "mace gui: stopped by mace gui connect.",
    "managed-done": "mace gui: the last task ended and nobody reattached; stopped.",
    "managed-cap": "mace gui: stopped after lingering 24 hours.",
    "walltime": "mace gui: stopped because the job's time limit is near.",
}


def dispatch_tool(name: str, argv: List[str]) -> int:
    """The standard-library tools under ``mace gui``: never import the server."""
    if name == "connect":
        from .connect import main as connect_main
        return connect_main(argv)
    if name == "login-url":
        from .loginurl import main as login_main
        return login_main(argv)
    if name == "diagnostics":
        from .diagnostics import cli_main
        return cli_main(argv)
    raise SystemExit(f"mace gui: unknown tool {name}")


def _env_port() -> Optional[int]:
    raw = os.environ.get("MACE_GUI_PORT", "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if 0 <= port <= 65535 else None


def build_parser() -> argparse.ArgumentParser:
    default_port = _env_port()
    p = argparse.ArgumentParser(
        prog="mace gui",
        description="Browse structures, optimizations, plots, materials.db and deck "
                    "settings of a directory in your web browser. The server uses only the "
                    "Python standard library; ase, spglib and plotly (from requirements.txt) "
                    "enable structures and plots.",
        epilog="It prints a one-time sign-in link and the session token for the sign-in form (valid while the "
               "server runs; mace gui login-url prints a new link). On a "
               "cluster: run it on a dev or compute node, then forward the port from your "
               "computer with the ssh command it prints, or start it from your computer in one "
               f"step with `mace gui connect` (see mace gui connect --help). See {DOCS}.")
    p.add_argument("--dir", "-d", default=".", help="directory to serve (default: .)")
    p.add_argument("--port", "-p", type=int, default=default_port,
                   help=f"port (default: MACE_GUI_PORT, else {DEFAULT_PORT} on your own computer and a port of your "
                        "own on a shared node: 20000 + your user id mod 10000, the next free one if that is taken; "
                        "0 picks a free one). A port you name is used as is")
    p.add_argument("--keep", action="store_true",
                   help="keep serving after the session that started it closes (by default a server started over "
                        "ssh without a terminal stops when that connection ends; nohup does the same as --keep)")
    p.add_argument("--new-session", action="store_true",
                   help="start another server even when one of yours already serves this folder")
    p.add_argument("--sessions", action="store_true",
                   help="list your running MACE GUI sessions (any node of this cluster that shares your home "
                        "folder) with the line that forwards each one, and exit")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1). Any other address also needs "
                        "--insecure-network")
    p.add_argument("--insecure-network", action="store_true",
                   help="allow a --host other than loopback: the token and your data then cross "
                        "the network unencrypted (the ssh tunnel is the supported way)")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="also accept this Host header")
    p.add_argument("--token",
                   help="use this session token instead of a random one: 16 or more letters, digits "
                        "or . _ ~ - (other users of the node can see it in ps; MACE_GUI_TOKEN does "
                        "the same privately)")
    p.add_argument("--gateway", default=os.environ.get("MACE_GUI_GATEWAY", "").strip() or None,
                   help="your cluster's login host, shown in the port-forwarding hint "
                        f"(default MACE_GUI_GATEWAY, else [gui] gateway in site.toml, else {GATEWAY_PLACEHOLDER})")
    p.add_argument("--open", action="store_true",
                   help="open MACE GUI on this machine: in its own window of a Chromium-family browser (Brave, "
                        "Chrome, Chromium, Edge, Vivaldi), else in a browser tab")
    p.add_argument("--browser", choices=("app", "tab"), default=None,
                   help="with --open: app (its own window, no tabs or address bar; the default when one of those "
                        "browsers is installed) or tab (the default browser; also the default when $BROWSER is "
                        "set). [gui] open in site.toml sets your default")
    p.add_argument("--gpu", type=int, metavar="N",
                   help="with --open: the NVIDIA GPU the browser that this starts renders on "
                        "(default: GPU 0, the one that usually drives the display; N >= 1 uses PRIME "
                        "render offload). A browser that is already running keeps its GPU")
    p.add_argument("--verbose", "-v", action="store_true", help="log requests")
    p.add_argument("--root", action="append", default=None, metavar="DIR",
                   help="a folder File > Open Folder may switch within (repeatable; replaces the defaults, "
                        "your home and scratch folders; MACE_GUI_ROOTS does the same). --root none turns "
                        "Open Folder off")
    p.add_argument("--managed", action="store_true",
                   help="used by mace gui connect: machine-readable READY line, control lines on stdin, "
                        "stops when the connection closes and no task runs")
    p.add_argument("--http-control", action="store_true",
                   help="with --managed, used by mace gui --slurm (a batch job has no stdin): control through "
                        "the managed/* routes only; stops when no watcher attached within 5 minutes, or the "
                        "last one left and no task runs")
    p.add_argument("--ready-file", metavar="FILE",
                   help="with --http-control: write the READY data (with the token) to this 0600 file instead "
                        "of stdout; a stopped.json beside it says why the server stopped")
    p.add_argument("--slurm", action="store_true",
                   help="run MACE GUI as a batch job on a compute node and forward a port of this node to it "
                        "(see mace gui --slurm --help)")
    return p


def browser_gpu_env(gpu: Optional[int]) -> dict:
    """The environment that makes a newly started browser render on NVIDIA GPU
    `gpu` (spec section 9.3): PRIME render offload to the X server's offload
    provider for that GPU. With GPU0 driving the display, GPU n >= 1 is the
    provider NVIDIA-G{n-1} (`xrandr --listproviders`)."""
    if not gpu or gpu < 1:
        return {}
    return {"__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__NV_PRIME_RENDER_OFFLOAD_PROVIDER": f"NVIDIA-G{gpu - 1}"}


def open_browser(url: str, gpu: Optional[int] = None, mode: Optional[str] = None) -> bool:
    """The URL in this computer's browser (appwindow.open_url: its own app
    window by default, mode "tab" for a browser tab; the browser's own output
    goes to a log file, not this terminal), with the GPU offload environment
    for the browser it starts."""
    from . import appwindow
    extra = browser_gpu_env(gpu)
    saved = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        return appwindow.open_url(url, mode, say=print)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def node_name() -> Tuple[str, Optional[str]]:
    """This node's name for a tunnel: (name, fqdn comment or None). The short
    form only when the full name has dots and the short form resolves here."""
    full = socket.gethostname() or "localhost"
    if "." in full:
        short = full.split(".")[0]
        try:
            socket.gethostbyname(short)
            return short, full
        except OSError:
            return full, None
    return full, None


def in_vscode(env=None) -> bool:
    e = os.environ if env is None else env
    return e.get("TERM_PROGRAM") == "vscode" or bool(e.get("VSCODE_IPC_HOOK_CLI"))


VSCODE_HINT = ["VS Code: this terminal forwards the port for you. Ctrl+click the sign-in link above (VS Code rewrites",
               "it to the forwarded port; it works once, mace gui login-url prints another). The Ports tab opens",
               "the bare address, which needs the token."]


def default_gateway() -> Optional[str]:
    """The cluster's login host for the hints: MACE_GUI_GATEWAY, else
    ``[gui] gateway`` in site.toml (a cluster's group file can set it for
    everyone). Only a plain host or user@host is used; anything else (or an
    unreadable settings file) gives None."""
    gw = (os.environ.get("MACE_GUI_GATEWAY") or "").strip()
    if not gw:
        try:
            from mace.submission import site
            tables, _, _ = site.load_tables()
            val = (tables.get("gui") or {}).get("gateway")
            gw = val.strip() if isinstance(val, str) else ""
        except Exception:
            gw = ""
    if not gw:
        gw = config_alias_for_domain() or ""
    return gw if gw and GATEWAY_RE.match(gw) else None


def node_domain() -> str:
    """This node's domain (dev-node-01.cluster.example.edu -> cluster.example.edu), or ''."""
    try:
        full = socket.getfqdn() or socket.gethostname()
    except OSError:
        return ""
    labels = full.split(".")
    return ".".join(labels[1:]).lower() if len(labels) > 2 else ""


def config_alias_for_domain(domain: Optional[str] = None, path: Optional[Path] = None) -> Optional[str]:
    """The one Host alias of ~/.ssh/config (shared home folders carry it to the node) whose HostName is in this
    node's domain: the cluster's login host under the name you chose for it. None when there is none or several."""
    domain = node_domain() if domain is None else domain
    if not domain:
        return None
    p = path or Path(os.path.expanduser("~")) / ".ssh" / "config"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits, aliases = [], []
    for line in text.splitlines() + ["Host __end__"]:
        m = re.match(r"\s*Host(?:\s+|\s*=\s*)(.+)", line, re.I)
        if m:
            aliases = [w.strip('"') for w in m.group(1).split() if re.fullmatch(r"[A-Za-z0-9._-]+", w.strip('"'))]
            continue
        m = re.match(r"\s*HostName(?:\s+|\s*=\s*)(\S+)", line, re.I)
        if m and aliases:
            hn = m.group(1).strip('"').lower()
            if hn == domain or hn.endswith("." + domain):
                hits.append(aliases[0])
            aliases = []
    return hits[0] if len(set(hits)) == 1 else None


def connect_line(gateway: Optional[str], root: Optional[str] = None) -> str:
    user = getpass.getuser()
    node, _ = node_name()
    gw = gateway or default_gateway() or GATEWAY_PLACEHOLDER
    gw = gw if "@" in gw else f"{user}@{gw}"
    return f"mace gui connect --gateway {gw} --node {node}" + (f" --dir {_shq(root)}" if root else "")


def _shq(text: str) -> str:
    import shlex
    return shlex.quote(text)


def ssh_hint(port: int, gateway: Optional[str], root: Optional[str] = None) -> List[str]:
    """What to run on your computer to reach this server: the exact node, the
    connect one-liner, the plain ssh line, the hop note."""
    user = getpass.getuser()
    node, fqdn = node_name()
    gateway = gateway or default_gateway()
    gw = gateway or GATEWAY_PLACEHOLDER
    gw = gw if "@" in gw else f"{user}@{gw}"
    lines = [
        f"This server runs on {node}. The tunnel must go to this exact node." + (f" ({fqdn})" if fqdn else ""),
        "From your computer, with MACE installed there:",
        f"  {connect_line(gateway, root)}",
        "Without MACE there (MobaXterm's local terminal, Git Bash, PowerShell): make the standalone script here, copy",
        "it to your computer and run it there (bash mace-gui-connect.sh; for PowerShell use --export-script powershell",
        "-o mace-gui-connect.ps1 and run .\\mace-gui-connect.ps1):",
        f"  {connect_line(gateway, root)} --export-script bash -o mace-gui-connect.sh",
        "Or with ssh alone (leave it running), then open the URL above:",
        f"  ssh -N -J {gw} -L {port}:127.0.0.1:{port} {user}@{node}",
        "(if the port is taken on your computer, change the first number and the URL's port)",
        "Windows: if PowerShell says ssh is not recognized, mace gui connect (above) finds Windows' OpenSSH itself,",
        "or run ssh by its full path: & \"$env:SystemRoot\\System32\\OpenSSH\\ssh.exe\" with the same arguments.",
        f"(it listens only on {node}'s own loopback, 127.0.0.1 and ::1: a tunnel must end at 127.0.0.1:{port} on {node}, as this line does;",
        f"one that ends at {node}:{port} never reaches it and the browser keeps loading)",
        "If your cluster's nodes accept logins only from the login node, add --hop nested (see docs/gui.md).",
    ]
    if not gateway:
        dom = node_domain()
        lines.append(f"(replace {GATEWAY_PLACEHOLDER} with your cluster's login host, the name you ssh to from your "
                     "computer"
                     + (f"; this node is in {dom}, where it often is" if dom else "")
                     + "; export MACE_GUI_GATEWAY=host, pass --gateway, or set [gui] gateway in site.toml to fill it in)")
    return lines


def headless_ssh_node(env=None) -> bool:
    """An ssh session on Linux with no display and no $BROWSER: a browser
    started here would be a text browser taking over the terminal."""
    e = os.environ if env is None else env
    return (sys.platform.startswith("linux") and bool(e.get("SSH_CONNECTION"))
            and not any(e.get(k) for k in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER")))


def x11_forwarded(env=None) -> bool:
    """DISPLAY is ssh's X11 forwarding (MobaXterm turns it on by default):
    an ssh session, a DISPLAY with a host part (localhost:10.0), no $BROWSER
    (VS Code's opens on your computer)."""
    e = os.environ if env is None else env
    display = e.get("DISPLAY") or ""
    return bool(e.get("SSH_CONNECTION")) and not e.get("BROWSER") and bool(re.match(r"^[^:/]+:\d", display))


def browser_blocked_reason(env=None) -> Optional[str]:
    """Why --open starts no browser here, or None."""
    if headless_ssh_node(env):
        return NO_BROWSER
    if x11_forwarded(env):
        return ("Not opening a browser on this node: DISPLAY is ssh's X11 forwarding, so it would draw on your "
                "computer through X11, slowly. Open the URL on your computer through a tunnel (mace gui connect, "
                "or the ssh line below).")
    return None


def server_environment() -> None:
    """The server never draws through X: matplotlib (plots made in the
    server and in its workers, which inherit this) uses Agg even when DISPLAY
    is set by X11 forwarding or MPLBACKEND names an interactive backend. Set
    after capture_login_env, so jobs submitted from the GUI keep the login
    environment."""
    os.environ["MPLBACKEND"] = "Agg"


NO_BROWSER = ("No browser on this node: open the URL on your computer through a tunnel (mace gui connect, "
              "or the ssh line below).")


def choose_token(flag: Optional[str]) -> str:
    """--token wins, then MACE_GUI_TOKEN, else a fresh random token."""
    if flag:
        return flag
    env = os.environ.get("MACE_GUI_TOKEN", "")
    return env if env else secrets.token_urlsafe(24)


def token_problem(flag: Optional[str]) -> Optional[str]:
    """Why a chosen token (--token or MACE_GUI_TOKEN) is refused, or None."""
    if flag is not None:
        source, value = "--token", flag
    elif os.environ.get("MACE_GUI_TOKEN", ""):
        source, value = "MACE_GUI_TOKEN", os.environ["MACE_GUI_TOKEN"]
    else:
        return None
    if TOKEN_RE.fullmatch(value):
        return None
    return f"{source} must be {TOKEN_RULE}"


def state_file(port: int) -> Path:
    """The server's private state file (0600, removed at exit): its base URL,
    token, pid and folder, for `mace gui login-url` and for scripts. In
    $XDG_RUNTIME_DIR, else in the GUI's cache folder (0700)."""
    run = os.environ.get("XDG_RUNTIME_DIR")
    if run and os.path.isdir(run) and os.access(run, os.W_OK):
        return Path(run) / f"mace-gui-{port}.json"
    from .managed import state_dir
    host = socket.gethostname().split(".")[0] or "host"
    return state_dir() / f"url-{host}-{port}.json"


url_file = state_file          # the name before the one-time links


def write_state_file(path: Path, data: Dict[str, Any]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == "gui":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
        fh.write("\n")


def session_watch_wanted(args, stdin_fd: int = 0) -> bool:
    """Whether this foreground server should stop when the session that started it goes away.

    Over `ssh node "mace gui"` without a terminal, sshd sends no hangup when the connection ends: the server
    would keep its port and memory on a shared node for good. Its standard input is then a pipe or socket from
    sshd, and it reaches end of file at that moment. Not watched: a terminal (the hangup handler stops it), a
    stdin that is /dev/null or a file (cron, `nohup ... < /dev/null &`, systemd: started to run alone), nohup
    (SIGHUP ignored), --keep, mace gui connect (--managed) and the batch job (--http-control).
    """
    if os.name != "posix" or args.keep or args.managed or args.http_control:
        return False
    if os.environ.get("MACE_GUI_KEEP") == "1":
        return False
    try:
        mode = os.fstat(stdin_fd).st_mode
    except OSError:
        return False
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
        return False
    try:
        if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is signal.SIG_IGN:
            return False
    except (ValueError, OSError):
        return False
    return not _closed_at_start(stdin_fd, mode)


def _closed_at_start(fd: int, mode: int) -> bool:
    """Standard input is already at end of file: it was launched with a closed pipe (a test runner, a cron-like
    parent), so there is no session to watch."""
    import select
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP)
        events = poller.poll(0)
        if not events:
            return False
        flags = events[0][1]
        if stat.S_ISFIFO(mode):
            return bool(flags & select.POLLHUP) and not flags & select.POLLIN
        import socket as _socket
        dup = _socket.socket(fileno=os.dup(fd))
        try:
            return dup.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT) == b""
        finally:
            dup.close()
    except (OSError, ValueError, AttributeError):
        return False


def start_session_watch(state, stdin_fd: int = 0) -> threading.Thread:
    """Stop the server (state.request_stop, so the registry record, URL files and ports go as usual) when
    standard input reaches end of file. Whatever arrives on it is discarded."""
    def run():
        while True:
            try:
                data = os.read(stdin_fd, 65536)
            except InterruptedError:
                continue
            except OSError:
                data = b""
            if not data:
                break
        state.request_stop("session-closed")
    t = threading.Thread(target=run, name="mace-gui-session-watch", daemon=True)
    t.start()
    return t


def _port_open(port: int) -> bool:
    """Something accepts connections on 127.0.0.1:port (a record's process id may have been reused)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def sessions_report(gateway: Optional[str], sessions: List[Dict[str, Any]], out=print) -> None:
    """Your running sessions, each with the line that forwards its port to your computer."""
    from . import registry
    user = getpass.getuser()
    if not sessions:
        out("No MACE GUI of yours is running (on this computer, or on another node that shares your home folder).")
        return
    out(f"Your running MACE GUI sessions ({len(sessions)}):")
    for d in sessions:
        out("  " + registry.describe(d))
        node = d.get("host") or node_name()[0]
        gw = gateway or default_gateway()
        gw = gw if gw and "@" in gw else f"{user}@{gw or GATEWAY_PLACEHOLDER}"
        out(f"    forward: ssh -N -J {gw} -L {d['port']}:127.0.0.1:{d['port']} {user}@{node}")
        if d.get("here"):
            out(f"    new sign-in link: mace gui login-url --port {d['port']}")
        else:
            out(f"    (on {node}: run mace gui login-url --port {d['port']} there for a new sign-in link)")


def already_running(d: Dict[str, Any], args) -> None:
    """A session of yours already serves this folder: say so, with a new sign-in link and the forwarding line."""
    from . import registry
    user = getpass.getuser()
    port = int(d["port"])
    print("MACE GUI is already running for you on this folder:")
    print("  " + registry.describe(d))
    link = None
    try:
        from . import loginurl
        for st in loginurl.servers():
            if int(st.get("port", 0)) == port and st.get("token"):
                link = loginurl.new_link(st)
    except Exception:
        link = None
    if link:
        print("New sign-in link (works once):")
        print(f"  {link}")
    else:
        print(f"New sign-in link: mace gui login-url --port {port}")
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SLURM_JOB_ID"):
        print("From your computer, forward its port (leave this running), then open the link:")
        for ln in _forward(port, d.get("host"), args.gateway):
            print(f"  {ln}")
    print(f"To stop it: File > Sign Out and Stop Server in MACE GUI" + (f", or kill {d['pid']}" if d.get("pid") else "")
          + ". To start a second one anyway: mace gui --new-session. All of yours: mace gui --sessions.")


def _forward(port: int, node: Optional[str], gateway: Optional[str]) -> List[str]:
    from . import ports
    return ports.forward_lines(port, node or node_name()[0], getpass.getuser(), gateway or default_gateway())[:1]


def add_ipv6_loopback(httpd, port: int):
    """A second listener for the IPv6 loopback ::1 on the same port, serving the same handler, or None.

    The server is on 127.0.0.1. Tools that reach it through the name `localhost` (VS Code's port
    forwarding runs on Node, which tries ::1 first; browsers on Windows do too) otherwise connect to
    nothing, and the page keeps loading. ::1 is as private as 127.0.0.1: no other computer reaches it.
    """
    if not socket.has_ipv6:
        return None
    base = type(httpd)

    class _Loopback6(base):
        address_family = socket.AF_INET6

        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except (AttributeError, OSError):
                pass
            super().server_bind()
    try:
        h6 = _Loopback6(("::1", port), httpd.RequestHandlerClass)
    except OSError:
        return None             # no IPv6 loopback here, or ::1:port belongs to another program
    threading.Thread(target=h6.serve_forever, kwargs={"poll_interval": 0.5}, name="mace-gui-loopback6",
                     daemon=True).start()
    return h6


def _stop_loopback6(h6) -> None:
    if h6 is None:
        return
    try:
        threading.Thread(target=h6.shutdown, daemon=True).start()
        h6.server_close()
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in TOOLS:
        return dispatch_tool(argv[0], argv[1:])
    if "--slurm" in argv:
        # a batch job on a compute node: standard library only, the server
        # starts inside the job (slurmsession.py)
        from .slurmsession import main as slurm_main
        return slurm_main([a for a in argv if a != "--slurm"])
    from .childenv import capture_login_env
    capture_login_env()          # before the server: what jobs submitted from the GUI get (submit.py)
    args = build_parser().parse_args(argv)
    if args.sessions:
        from . import registry
        sessions_report(args.gateway, registry.live(port_open=_port_open))
        return 0
    server_environment()
    from . import diaglog
    diaglog.install()           # the last server log lines, for Copy Diagnostics
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        print(f"mace gui: {root} is not a directory", file=sys.stderr)
        return 2
    from .server import is_loopback, make_server
    if not is_loopback(args.host) and not args.insecure_network:
        print(f"mace gui: refusing to listen on {args.host}: the session token and your data would "
              "cross the network unencrypted. Use the ssh tunnel the docs describe (the server "
              "stays on 127.0.0.1), or add --insecure-network if you accept that.", file=sys.stderr)
        return 2
    bad = token_problem(args.token)
    if bad:
        print(f"mace gui: {bad}", file=sys.stderr)
        return 2
    if args.token:
        print("mace gui: note: other users of this node can see --token in ps; "
              "set MACE_GUI_TOKEN instead to keep it private.", file=sys.stderr)
    if (args.http_control or args.ready_file) and not args.managed:
        print("mace gui: --http-control and --ready-file go with --managed", file=sys.stderr)
        return 2
    if args.managed and not args.http_control:
        # a server for this folder still lingering here (its tasks run on):
        # relay to it instead of starting a second one
        from . import managed
        found = managed.find_lingering(root)
        if found is not None:
            from mace import __version__
            return managed.attach_proxy(found, args.port, {"host": node_name()[0], "fqdn": socket.getfqdn(),
                                                           "version": __version__, "home": mace_home()})
    from . import update_gui
    restarted = os.environ.pop("MACE_GUI_RESTARTED", "") == "1"    # Update Now and Restart (update_gui.py)
    handoff = update_gui.take_handoff()
    from . import ports, registry
    if not args.managed and not restarted and handoff is None:
        mine = registry.live(port_open=_port_open)
        again = registry.serving(str(root), mine)
        if again is not None and not args.new_session:
            already_running(again, args)
            return 0
        others = [d for d in mine if d is not again]
        if others:
            print("mace gui: you also have " + ("these sessions" if len(others) > 1 else "this session") + " running:",
                  file=sys.stderr)
            for d in others:
                print("  " + registry.describe(d), file=sys.stderr)
    defaulted = args.port is None
    wanted = None
    if defaulted:
        wanted, _why = ports.default_port()
        picked = ports.pick(wanted, args.host, free=ports.is_free_loopback)
        if picked is None:
            print(f"mace gui: ports {wanted}-{wanted + ports.PROBE_TRIES} are all busy here. Pick one with "
                  "--port, or --port 0 for any free one.", file=sys.stderr)
            return 2
        args.port = picked
        args.port_note = None if picked == wanted else (
            f"Port {wanted} is busy here; using {picked}. Drafts saved under port {wanted} won't be offered on "
            f"{picked}.")
    token = handoff["token"] if handoff else choose_token(args.token)
    httpd = state = None
    for attempt in range(4 if defaulted else 1):
        try:
            httpd, state = make_server(root, args.host, args.port, token,
                                       args.allow_host, args.verbose, roots=args.root)
            state.restarted = restarted
            update_gui.resume(state, handoff)
            break
        except OSError as exc:
            if defaulted and attempt < 3:       # taken between the check and the bind
                nxt = ports.pick(args.port + 1, args.host, free=ports.is_free_loopback)
                if nxt is not None:
                    args.port = nxt
                    args.port_note = (f"Port {wanted} is busy here; using {nxt}. Drafts saved under port {wanted} "
                                      f"won't be offered on {nxt}.")
                    continue
            if args.managed:
                print("MACE_GUI_ERROR port-busy", flush=True)
                print(f"mace gui: cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
                return 98
            print(f"mace gui: cannot listen on {args.host}:{args.port}: {exc}. Another program (or another user "
                  "of this node) has that port. Leave --port off for a port of your own, or pick one "
                  "(--port N, or export MACE_GUI_PORT=N in ~/.bashrc so your browser drafts stay on one "
                  "port); --port 0 picks a free one for a one-off run. mace gui --sessions lists your own servers.",
                  file=sys.stderr)
            return 2
    from mace import __version__ as _mace_version
    h6 = add_ipv6_loopback(httpd, httpd.server_address[1]) if args.host in ("127.0.0.1", "localhost") else None
    reg = registry.Registration(httpd.server_address[1], str(state.root), managed=args.managed,
                                version=_mace_version, root_fn=lambda: state.root).start()
    # from here on SIGTERM, a hangup and Ctrl+C all stop the server through
    # the finally below, so the URL file and the temporary folders never stay
    stops = [signal.SIGTERM]
    # a hangup (the ssh session or terminal closed) cleans up the same way;
    # under nohup SIGHUP is ignored, and stays so
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        stops.append(signal.SIGHUP)

    def _stop(*_):
        # once: a closing terminal can send SIGHUP more than once, and a
        # second one must not interrupt the cleanup below
        for sig in stops:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt
    for sig in stops:
        signal.signal(sig, _stop)
    install_request_stop(httpd, state)
    if session_watch_wanted(args):
        start_session_watch(state)
    from .server import start_watchdog
    start_watchdog(state)
    url_files: List[Path] = []
    try:
        code = _serve(args, httpd, state, url_files.append)
        _report_stop(args, state)
        return code
    except KeyboardInterrupt:
        try:
            print("\nStopping.", flush=True)
        except OSError:     # the terminal is gone after a hangup
            _quiet_stdout()
        return 0
    finally:
        for sig in stops:   # nothing interrupts the cleanup
            signal.signal(sig, signal.SIG_IGN)
        # a preference saved as the page closed (a keepalive request) lands before the stop
        try:
            from .server import _profile_writes
            _profile_writes.drain(2.0)
        except Exception:
            pass
        reg.close()
        _stop_loopback6(h6)
        httpd.server_close()
        state.close()
        for path in url_files:
            try:
                path.unlink()
            except OSError:
                pass
        if getattr(state, "managed", None) is not None:
            state.managed.remove_state_file()
        if args.ready_file and getattr(state, "stop_reason", None) != "update-restart":
            _ready_file_stopped(Path(args.ready_file), getattr(state, "stop_reason", None) or "stopped")
        # last: exec the updated server in place, or the update deferred to the sign-out
        update_gui.after_stop(state, argv, httpd.server_address[1], stops)


def install_request_stop(httpd, state) -> None:
    """state.request_stop(reason): the one way code stops the server (D14).
    It records the reason and runs httpd.shutdown() in a new thread (never
    the serving thread, which would deadlock), so serve_forever returns and
    main()'s finally does the usual cleanup. Works the same on Windows."""
    import threading
    lock = threading.Lock()

    def request_stop(reason: str) -> None:
        with lock:
            if getattr(state, "stop_reason", None):
                return
            state.stop_reason = reason
        threading.Thread(target=httpd.shutdown, name="mace-request-stop", daemon=True).start()
    state.stop_reason = None
    state.request_stop = request_stop


def _report_stop(args, state) -> None:
    reason = getattr(state, "stop_reason", None)
    if not reason:
        return
    if reason == "update-restart":
        # not MACE_GUI_STOPPED: the connect clients keep the session; the same process serves again
        try:
            print("mace gui: MACE was updated; restarting.", file=sys.stderr if args.managed else sys.stdout,
                  flush=True)
        except OSError:
            pass
        return
    extra = getattr(state, "stop_detail", {}) or {}
    line = STOP_LINES.get(reason, f"mace gui: stopped ({reason}).")
    try:
        line = line.format(**{"hours": "", "minutes": "", **extra})
    except (KeyError, IndexError):
        pass
    if args.managed:
        from . import managed
        managed.emit("MACE_GUI_STOPPED", {"reason": reason})
        managed.log(line[len("mace gui: "):] if line.startswith("mace gui: ") else line)
        return
    try:
        print(line, flush=True)
    except OSError:
        _quiet_stdout()


class _GuardedStream:
    """sys.stdout / sys.stderr while serving: the terminal or pipe that
    started the server may go away (a plain `ssh node mace gui ...` that is
    closed, a log reader that quits) while the server keeps running; then a
    print (Open Folder's note, a task's line) must not fail the request that
    made it. The first failed write points fds 1 and 2 at /dev/null."""

    def __init__(self, stream):
        self._stream = stream

    @staticmethod
    def _gone() -> None:
        # dup2 only: flushing here would write to the dead pipe again
        try:
            null = os.open(os.devnull, os.O_WRONLY)
        except OSError:
            return
        for fd in (1, 2):
            try:
                os.dup2(null, fd)
            except OSError:
                pass
        os.close(null)

    def write(self, text):
        try:
            return self._stream.write(text)
        except (OSError, ValueError):
            self._gone()
            try:
                return self._stream.write(text)
            except (OSError, ValueError):
                return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            self._gone()
            try:
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _guard_std_streams() -> None:
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if cur is not None and not isinstance(cur, _GuardedStream):
            setattr(sys, name, _GuardedStream(cur))


def _quiet_stdout() -> None:
    """stdout's terminal is gone: nothing left to flush at exit (status 0, not 120)."""
    try:
        sys.stdout = open(os.devnull, "w")
    except OSError:
        pass


def _serve(args, httpd, state, remember) -> int:
    """Print where the server is (or write the URL file), then serve until stopped."""
    from .server import is_loopback
    if not is_loopback(args.host):
        print(f"WARNING: listening on {args.host} without encryption: anyone on the network path "
              "can read the token and your data.", file=sys.stderr)
        names = sorted(state.allowed_hosts - {"127.0.0.1", "localhost", "::1", "[::1]"})
        if names:
            print("mace gui: accepting the Host names " + ", ".join(names), file=sys.stderr)
    port = httpd.server_address[1]
    shown = "127.0.0.1" if args.host in ("127.0.0.1", "0.0.0.0", "localhost", "::") else args.host
    if ":" in shown:
        shown = f"[{shown}]"
    base = f"http://{shown}:{port}{state.prefix}"
    state.tunnel = next((ln.strip() for ln in ssh_hint(port, args.gateway)
                         if ln.strip().startswith("ssh ")), "")
    state.connect = connect_line(args.gateway, str(state.root))
    state.connect_base = connect_line(args.gateway)     # /api/info adds the folder served now (Open Folder)
    state.vscode = in_vscode()
    state.node = node_name()[0]
    if args.managed:
        # a marker without the token: `mace gui login-url` here says the links
        # come from the computer running `mace gui connect` (its tunnel port)
        marker = state_file(port)
        try:
            write_state_file(marker, {"base": base, "pid": os.getpid(), "root": str(state.root), "port": port,
                                      "managed": True})
            remember(marker)
        except OSError:
            pass
        return _serve_managed(args, httpd, state, base, base)
    # the private state file: always (also on a terminal), removed at exit
    private = state_file(port)
    remember(private)
    try:
        write_state_file(private, {"base": base, "token": state.token, "pid": os.getpid(),
                                   "root": str(state.root), "port": port})
    except OSError as exc:
        print(f"mace gui: cannot write the state file {private}: {exc}", file=sys.stderr)
        return 2
    if sys.stdout.isatty() or os.environ.get("MACE_GUI_PRINT_TOKEN") == "1":
        # a one-time sign-in link: reopening it from the browser's history signs nobody in
        lines = [f"MACE GUI serving {state.root}", f"  {state.mint_login_url(base)}",
                 "  (the link signs in once; mace gui login-url prints another)",
                 f"Token (for the sign-in form): {state.token}"]
    else:
        # stdout is a file or a pipe (sbatch, nohup, > log): often readable by
        # others, so no sign-in link and no token here
        lines = [f"MACE GUI serving {state.root}", f"  {base}",
                 f"Sign in with a link from mace gui login-url --port {port} (the token is in {private},",
                 "readable only by you; removed when the server stops). MACE_GUI_PRINT_TOKEN=1 prints a "
                 "link here instead."]
    from . import cpulimit
    lim = cpulimit.limit()
    if lim and not os.environ.get("SLURM_JOB_ID"):
        lines += [f"This node limits each process to {cpulimit.minutes(lim)} CPU minutes. MACE GUI restarts its helper "
                  "processes before they reach it and stops cleanly if it gets close itself."]
    if state.vscode:
        lines += [""] + VSCODE_HINT
    elif os.environ.get("SSH_CONNECTION") or os.environ.get("SLURM_JOB_ID"):
        lines += [""] + ssh_hint(port, args.gateway, str(state.root))
    from . import ports
    normal = ports.default_port()[0]
    if getattr(args, "port_note", None):
        lines += [args.port_note]
    elif port != normal:
        # drafts live in the browser per origin (http://127.0.0.1:<port>)
        lines += ["Drafts saved under another port won't be offered on this one."]
    elif normal != DEFAULT_PORT:
        lines += [f"Port {port} is yours on shared nodes (20000 + your user id mod 10000), so you and the others "
                  "on this node never share one."]
    lines += ["Press Ctrl+C to stop."]
    try:
        print("\n".join(lines), flush=True)
    except OSError:         # the terminal closed already: its hangup is on the way
        _quiet_stdout()
        return 0
    if args.gpu is not None and not args.open:
        print("mace gui: --gpu applies with --open (the browser it starts)", file=sys.stderr)
    if args.open:
        blocked = browser_blocked_reason()
        if blocked:
            print(blocked, file=sys.stderr)
        else:
            open_browser(state.mint_login_url(base), args.gpu, args.browser)
    _guard_std_streams()
    httpd.serve_forever(poll_interval=0.5)
    return 0


def _serve_managed(args, httpd, state, base, url) -> int:
    """--managed: the READY line on stdout, control lines on stdin, the watchdog."""
    from mace import __version__
    from . import managed
    ctl = managed.Controller(running=state.tasks.running_count, stop=state.request_stop)
    port = httpd.server_address[1]
    ctl.info = {"host": node_name()[0], "port": port, "prefix": state.prefix, "token": state.token,
                "root": str(state.root), "started": time.time()}
    state.managed = ctl
    login = state.mint_login_url(base)
    ctl.on_login = lambda: state.mint_login_url(base)
    ready = {"url": base, "port": port, "prefix": state.prefix, "token": state.token, "login": login,
             "host": node_name()[0], "fqdn": socket.getfqdn(), "root": str(state.root), "version": __version__,
             "home": mace_home(),
             "reattached": False, "tasks": 0}
    if args.http_control:
        ctl.http_control = True
    # after Update Now and Restart: the same watchers, and a first connection
    # that was gone stays gone (its stdin is a dead pipe)
    ctl.restore(getattr(state, "_managed_handoff", None))
    if args.http_control:
        # a batch job: no stdin; the READY data (token included) only in a 0600 file
        if args.ready_file:
            try:
                managed.write_private_json(Path(args.ready_file), {**ready, "pid": os.getpid()})
            except OSError as exc:
                print(f"mace gui: cannot write {args.ready_file}: {exc}", file=sys.stderr)
                return 2
            print(f"mace gui: ready on {node_name()[0]} port {port} (details in the ready file)", flush=True)
        elif not getattr(state, "restarted", False):
            managed.emit("MACE_GUI_READY", ready)
        ctl.run(stdin=False)
    else:
        # after Update Now and Restart the client has its READY line already (a
        # second one would open another browser tab from the connect scripts)
        if not getattr(state, "restarted", False) and not managed.emit("MACE_GUI_READY", ready):
            return 0
        ctl.run(stdin=None if ctl.connected else False)
    httpd.serve_forever(poll_interval=0.5)
    return 0


def _ready_file_stopped(path: Path, reason: str) -> None:
    """The server stopped: the token-bearing ready file goes, a stopped.json
    beside it tells mace gui --slurm why."""
    from . import managed
    try:
        path.unlink()
    except OSError:
        pass
    try:
        managed.write_private_json(path.with_name("stopped.json"), {"reason": reason, "time": time.time()})
    except OSError:
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
