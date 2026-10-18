"""``mace gui connect``: start MACE GUI on a cluster node and open it in the
browser on your computer, with one ssh session.

    mace gui connect [PROFILE] [--gateway G] [--node N] [--dir D] ...

One ssh process starts ``mace gui --managed`` on the node (through the
login node, the "gateway"), forwards a local port to it, and carries the
watchdog: when you press Ctrl+C, close the window or lose the connection, the
server stops, unless GUI tasks are still running, in which case it keeps
running for them and ``mace gui connect`` again reattaches (managed.py).

Two ways to reach the node (the profile's ``hop``):

* ``jump``: ``ssh -J <gateway> <node>``: one TCP path from this computer to
  the node; the node must accept your own login.
* ``nested``: ``ssh <gateway> "ssh <node> ..."``: the gateway logs in to the
  node itself (for clusters whose nodes accept logins only from the login
  node); the inner hop is non-interactive (BatchMode).
* ``auto`` (default): ``jump``, and ``nested`` once when the node refused the
  login through the gateway.

Profiles live in ``$XDG_CONFIG_HOME/mace/connect.toml`` (Windows:
``%APPDATA%\\mace\\connect.toml``), no secrets; mode 0600 on Linux and macOS
(on Windows the file has the access rights of your own AppData folder).

Standard library only: this runs on your own computer, which needs Python 3
and ssh but not MACE's scientific packages.
"""
from __future__ import annotations

import argparse
import codecs
import ctypes
import fnmatch
import getpass
import http.client
import json
import os
import queue
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_LOCAL_PORT = 8765
PORT_TRIES = 20
RP_RANGE = (20000, 60999)
READY_POLL_S = 20.0
PING_S = 60.0
SLOW_NOTE_S = 15.0
# ssh's ConnectTimeout (seconds) unless --ssh-option or your ssh config sets
# one: an unreachable host fails in this time with "cannot reach host", not
# after the system's TCP timeout of a minute or more
CONNECT_TIMEOUT_S = 20
# where the remote login shell looks for MACE when the mace command is not on
# its PATH (the default "mace", or a path that does not exist there)
MACE_FALLBACKS = ('${MACE_HOME:+"$MACE_HOME"/mace_cli}', '"$HOME"/mace/mace_cli', '"$HOME"/MACE/mace_cli')
# --slurm: run the server as a batch job (slurmsession.py on the node or gateway)
SLURM_KEYS = ("slurm_time", "slurm_cpus", "slurm_mem", "slurm_partition", "slurm_account", "slurm_qos",
              "slurm_constraint", "slurm_gpus", "slurm_args")
PROFILE_KEYS = ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda", "local_port", "ssh_options",
                "open", "roots", "remote_shell", "slurm") + SLURM_KEYS
HOPS = ("auto", "jump", "nested")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DOCS_MANUAL = "docs/gui.md#run-it-on-a-cluster-and-view-it-on-your-computer"
DOCS_WINDOWS = "docs/gui.md#windows-powershell"
DOCS_NODE_PASSWORD = "docs/gui.md#mace-gui-connect-the-node-asks-for-a-password"
HEADER = """\
# mace gui connect profiles (mace gui connect --help). No passwords or tokens.
# Fields: gateway (login host, e.g. "me@hpcc.example.edu" or an ~/.ssh/config Host),
# jump (bastion hosts before the gateway, comma separated), node (the dev node
# to run on; empty = the gateway itself), hop ("auto", "jump" or "nested"),
# dir (folder to serve on the node, "~/..." allowed), mace (the mace command
# there), setup (shell lines run first, e.g. "module load Python"), conda
# (an environment to activate), local_port, ssh_options (extra ssh arguments),
# open (open the browser), roots (folders Open Folder may switch to).
# slurm = true runs MACE GUI as a batch job on a compute node (mace gui --slurm),
# with slurm_time, slurm_cpus, slurm_mem, slurm_partition, slurm_account,
# slurm_qos, slurm_constraint, slurm_gpus and slurm_args (extra sbatch options).
"""


def say_line(text: str) -> None:
    """print, flushed at once (stdout may be a pipe: a script, a test, a log)."""
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


# ----------------------------------------------------------------- profiles

def config_path(env: Optional[Dict[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    if os.name == "nt" and e.get("APPDATA"):
        return Path(e["APPDATA"]) / "mace" / "connect.toml"
    base = e.get("XDG_CONFIG_HOME") or os.path.join(e.get("HOME") or os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / "connect.toml"


def _unquote_basic(text: str) -> str:
    """A TOML basic string's escapes, in one pass, as tomllib reads them."""
    out, i = [], 0
    esc = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            n = text[i + 1]
            width = {"u": 4, "U": 8}.get(n, 0)
            if width and re.fullmatch(r"[0-9A-Fa-f]{%d}" % width, text[i + 2:i + 2 + width]):
                out.append(chr(int(text[i + 2:i + 2 + width], 16)))
                i += 2 + width
                continue
            out.append(esc.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


_STR = r'"(?:[^"\\]|\\.)*"'


def parse_flat(text: str) -> Tuple[Dict[str, Any], List[int]]:
    """The flat TOML subset MACE writes (basic strings, ints, bools, one-line
    string arrays, [table] headers): what the standalone scripts read too.
    Returns (data, line numbers it could not read)."""
    data: Dict[str, Any] = {}
    cur = data
    bad: List[int] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([A-Za-z0-9_.-]+)\]", line)
        if m:
            cur = data
            for part in m.group(1).split("."):
                cur = cur.setdefault(part, {})
            continue
        m = re.fullmatch(r"([A-Za-z0-9_-]+)\s*=\s*(.*?)\s*(#.*)?", line)
        if not m:
            bad.append(n)
            continue
        key, val = m.group(1), m.group(2)
        if re.fullmatch(_STR, val):
            cur[key] = _unquote_basic(val[1:-1])
        elif re.fullmatch(r"-?\d+", val):
            cur[key] = int(val)
        elif val in ("true", "false"):
            cur[key] = val == "true"
        elif re.fullmatch(r"\[\s*(?:%s\s*(?:,\s*%s\s*)*,?\s*)?\]" % (_STR, _STR), val):
            cur[key] = [_unquote_basic(x[1:-1]) for x in re.findall(_STR, val)]
        else:
            bad.append(n)
    return data, bad


def load_profiles(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or config_path()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import tomllib
        data = tomllib.loads(text)
    except ImportError:                       # Python < 3.11: the flat subset
        data, _ = parse_flat(text)
    except ValueError as exc:
        raise SystemExit(f"mace gui connect: {p} is not valid TOML: {exc}")
    return data if isinstance(data, dict) else {}


def _qstr(v: str) -> str:
    out = []
    for c in v:
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif ord(c) < 0x20 or ord(c) == 0x7f:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def _value(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return _qstr(v)
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return "[" + ", ".join(_qstr(x) for x in v) + "]"
    return None                               # outside the flat subset: not written


def dump_profiles(data: Dict[str, Any]) -> str:
    lines = [HEADER.rstrip("\n"), ""]
    tops = {k: v for k, v in data.items() if not isinstance(v, dict)}
    for k, v in tops.items():
        s = _value(v)
        if s is not None:
            lines.append(f"{k} = {s}")
    tables: List[Tuple[str, Dict[str, Any]]] = []

    def walk(prefix: str, d: Dict[str, Any]):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if prefix and (scalars or not any(isinstance(v, dict) for v in d.values())):
            tables.append((prefix, scalars))
        for k, v in d.items():
            if isinstance(v, dict):
                walk(f"{prefix}.{k}" if prefix else k, v)
    walk("", {k: v for k, v in data.items() if isinstance(v, dict)})
    for name, scalars in tables:
        lines += ["", f"[{name}]"]
        for k, v in scalars.items():
            s = _value(v)
            if s is not None:
                lines.append(f"{k} = {s}")
    return "\n".join(lines) + "\n"


def save_profiles(data: Dict[str, Any], path: Optional[Path] = None) -> Path:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # LF on every system: the standalone scripts read this file too (a CRLF
    # file from Windows' text mode broke mace-gui-connect.sh in MobaXterm)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dump_profiles(data))
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


def default_profile_name(data: Dict[str, Any]) -> Optional[str]:
    d = data.get("defaults") or {}
    name = d.get("profile") if isinstance(d, dict) else None
    return name if isinstance(name, str) and name in (data.get("profiles") or {}) else None


# ----------------------------------------------------------------- ssh

def local_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("USERNAME") or "user"


def effective_node_user(gateway: str, runner: Callable = subprocess.run) -> Optional[str]:
    """The user ssh logs in to the gateway as (``ssh -G``), when it differs
    from the local user name: the node then needs the same ``user@``."""
    if not gateway:
        return None
    if "@" in gateway:
        u = gateway.split("@", 1)[0]
        return u if u != local_user() else None
    try:
        r = runner(["ssh", "-G", gateway], capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (r.stdout or "").splitlines():
        k, _, v = line.partition(" ")
        if k.lower() == "user" and v.strip():
            u = v.strip()
            return u if u != local_user() else None
    return None


def ssh_g(host: str, runner: Callable = subprocess.run) -> Dict[str, List[str]]:
    """``ssh -G host``: the settings ssh would use for host, as {key: [values]}."""
    out: Dict[str, List[str]] = {}
    try:
        r = runner(["ssh", "-G", host], capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return out
    for line in (r.stdout or "").splitlines():
        k, _, v = line.partition(" ")
        if k and v.strip():
            out.setdefault(k.lower(), []).append(v.strip())
    return out


def node_key_options(gateway: str, target: str, runner: Callable = subprocess.run) -> List[str]:
    """ssh options that give the node hop of ``ssh -J <gateway> <node>`` the
    key setup your ssh config has for the gateway.

    With -J, ssh logs in to the node from THIS computer, with the settings
    ssh's config has for the NODE's name: an IdentityFile (or IdentityAgent,
    IdentitiesOnly) set in the gateway's Host block (``Host hpcc``) does not
    apply to it. Without an agent holding that key, the node is offered only
    the default key files, refuses them, and ssh asks for the node's password.
    The command line's ``-o`` options apply to the destination (the node)
    only; the jump to the gateway keeps its own settings."""
    if not gateway or not target:
        return []
    g, n = ssh_g(gateway, runner), ssh_g(target, runner)
    opts: List[str] = []
    for f in g.get("identityfile", []):
        if f not in n.get("identityfile", []) and os.path.isfile(os.path.expanduser(f)):
            opts += ["-o", f"IdentityFile={f}"]
    agent = (g.get("identityagent") or [""])[0]
    if agent and agent != (n.get("identityagent") or [""])[0]:
        opts += ["-o", f"IdentityAgent={agent}"]
    if (g.get("identitiesonly") or [""])[0] == "yes" and (n.get("identitiesonly") or [""])[0] != "yes":
        opts += ["-o", "IdentitiesOnly=yes"]
    return opts


def pick_local_port(start: int, tries: int = PORT_TRIES, warn: Callable[[str], None] = say_line) -> int:
    for p in range(start, start + tries + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                if os.name != "nt":
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        if p != start:
            warn(f"Port {start} is busy here; using {p}. Drafts saved under port {start} won't be offered on {p}.")
        return p
    raise SystemExit(f"mace gui connect: ports {start}-{start + tries} are all busy on this computer; "
                     "pass --local-port")


def random_port(rng=None) -> int:
    import random
    r = rng or random.SystemRandom()
    return r.randint(*RP_RANGE)


# A leading // marks a literal absolute path on the node: never rewritten to
# ~/... (one that sits under this computer's home folder too), sent with one
# slash dropped. The scripts (scripts/mace-gui-connect.sh, .ps1) do the same.
LITERAL = "//"


def _home_path(text: str) -> str:
    """A shell word for a path; "~" and "~/..." become "$HOME"-relative, the rest is literal
    ("//abs" is a literal /abs, see LITERAL)."""
    if text.startswith(LITERAL):
        return shlex.quote(text[1:])
    if text == "~":
        return '"$HOME"'
    if text.startswith("~/"):
        return '"$HOME"/' + shlex.quote(text[2:])
    return shlex.quote(text)


def _local_home_forms(home: Optional[str] = None) -> List[str]:
    """This computer's home folder as a path prefix ("" when it is "/" or unknown)."""
    h = os.path.expanduser("~") if home is None else home
    forms = []
    for f in (h, h.replace("\\", "/") if os.name == "nt" else h):
        f = f.rstrip("/\\")
        if f and f != "~" and f not in forms:
            forms.append(f)
    return forms


def _localized(value: str, forms: Sequence[str]) -> Optional[str]:
    """``~`` or ``~/rest`` for a path under this computer's home folder, else None."""
    if value.startswith(LITERAL):
        return None
    seps = "/\\" if os.name == "nt" else "/"
    for f in forms:
        if value == f:
            return "~"
        if value.startswith(f) and len(value) > len(f) and value[len(f)] in seps:
            rest = value[len(f) + 1:]
            return "~/" + (rest.replace("\\", "/") if os.name == "nt" else rest)
    return None


def localize_home_paths(profile: Dict[str, Any], home: Optional[str] = None) -> List[Tuple[str, str]]:
    """Paths on the node given as paths under THIS computer's home folder:
    an unquoted ``~/x`` that the local shell expanded before MACE saw it
    (``--mace ~/mace/mace_cli`` arrives as /home/you/mace/mace_cli, which the
    node does not have when its home is elsewhere). Rewrites dir, the words
    of mace, roots and conda to ``~/...`` in place (expanded on the node) and
    returns the (before, after) pairs. ``setup`` is shell code the user
    quotes, so the local shell never expands it: left alone. A path written
    with a leading // is literal (LITERAL): left alone too."""
    forms = _local_home_forms(home)
    if not forms:
        return []
    done: List[Tuple[str, str]] = []

    def one(v: str) -> str:
        new = _localized(v, forms)
        if new is None:
            return v
        done.append((v, new))
        return new
    for key in ("dir", "conda"):
        if isinstance(profile.get(key), str) and profile[key]:
            profile[key] = one(profile[key])
    if isinstance(profile.get("mace"), str) and profile["mace"]:
        # word by word, keeping the rest of the string as written (the scripts split it on blanks)
        profile["mace"] = re.sub(r"\S+", lambda m: one(m.group(0)), profile["mace"])
    if isinstance(profile.get("roots"), list):
        profile["roots"] = [one(r) if isinstance(r, str) else r for r in profile["roots"]]
    if done:
        profile["_localized"] = list(done)
    return done


def localized_note(pairs: Sequence[Tuple[str, str]]) -> str:
    what = ", ".join(f"{a} as {b}" for a, b in pairs)
    return (f"Note: treated {what} on the remote (your shell turned ~ into this computer's home folder; quote the "
            f"~, as '{pairs[0][1]}', to avoid this).")


def remote_command(profile: Dict[str, Any], rp: int) -> str:
    """The command the node's login shell runs: a bash login shell that sets
    the environment up, changes to the folder and execs the managed server.
    Every value is quoted; ``setup`` is shell code on purpose (yours)."""
    parts: List[str] = []
    setup = profile.get("setup")
    if isinstance(setup, (list, tuple)):
        setup = "; ".join(str(x) for x in setup if str(x).strip())
    if setup and str(setup).strip():
        parts.append(str(setup).strip().rstrip(";"))
    if profile.get("conda"):
        parts.append("conda activate " + _home_path(str(profile["conda"])))
    d = str(profile.get("dir") or "~")
    mace = shlex.split(str(profile.get("mace") or "mace")) or ["mace"]
    cmd = ["exec"]
    if not profile.get("node") and not profile.get("slurm"):
        cmd += ["env", "MACE_GUI_MAX_WORKERS=2"]          # a shared login node: two workers at most
    find = mace_fallback(mace)
    # --no-banner: the credits banner would only show as "remote:" lines
    cmd += (['"$M"'] if find else [_home_path(w) for w in mace]) + ["--no-banner", "gui"]
    if profile.get("slurm"):
        cmd += ["--slurm"] + slurm_args(profile)
    cmd += ["--managed", "--port", str(int(rp)), "--dir", "."]
    for r in profile.get("roots") or []:
        cmd += ["--root", _home_path(str(r))]
    if find:
        parts.append(f"cd -- {_home_path(d)} && {{ {find}; " + " ".join(cmd) + "; }")
    else:
        parts.append(f"cd -- {_home_path(d)} && " + " ".join(cmd))
    script = "; ".join(parts)
    if (profile.get("remote_shell") or "bash") == "sh":
        return "exec sh -c " + shlex.quote(script)
    return "exec bash -lc " + shlex.quote(script)


def mace_fallback(mace: Sequence[str]) -> str:
    """Shell code that sets M to the mace command, or, when that is not
    there, to the first mace_cli found in MACE_FALLBACKS (printing
    ``MACE_GUI_USING <path>``, so connect can say which one it used). Only for
    the default ``mace`` and a one-word path not ending in mace_cli (then its
    folder's mace_cli is tried first); "" otherwise (the command runs as given)."""
    if len(mace) != 1:
        return ""
    w = mace[0]
    if w == "mace":
        test, cands = 'command -v "$M" >/dev/null 2>&1', list(MACE_FALLBACKS)
    elif "/" in w and not w.endswith("/mace_cli") and not w.endswith("/"):
        test = '[ -x "$M" ]'
        own = _home_path(w.rsplit("/", 1)[0] + "/mace_cli")
        cands = [own] + [c for c in MACE_FALLBACKS if c != own]
    else:
        return ""
    return (f"M={_home_path(w)}; {test} || for c in {' '.join(cands)}; do if [ -x \"$c\" ]; then M=$c; "
            'echo "MACE_GUI_USING $c"; break; fi; done')


def slurm_args(profile: Dict[str, Any]) -> List[str]:
    """``mace gui --slurm`` options from the profile (each value quoted)."""
    out: List[str] = []
    for key, flag in (("slurm_time", "--time"), ("slurm_cpus", "--cpus"), ("slurm_mem", "--mem"),
                      ("slurm_partition", "--partition"), ("slurm_account", "--account"), ("slurm_qos", "--qos"),
                      ("slurm_constraint", "--constraint"), ("slurm_gpus", "--gpus")):
        v = profile.get(key)
        if v not in (None, ""):
            out += [flag, shlex.quote(str(v))]
    for a in profile.get("slurm_args") or []:
        out.append(shlex.quote(f"--sbatch-arg={a}"))
    return out


def _base_opts(extra: Sequence[str] = (), configured: Optional[str] = None) -> List[str]:
    out = ["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
    return out + _timeout_opts(extra, configured)


def _timeout_opts(extra: Sequence[str] = (), configured: Optional[str] = None) -> List[str]:
    """-o ConnectTimeout=CONNECT_TIMEOUT_S, unless the extra ssh options or
    your ssh config (``configured``: config_connect_timeout) set one: a
    command-line -o would beat the config's own value (a slow VPN's)."""
    if configured or any("connecttimeout" in str(x).lower() for x in extra):
        return []
    return ["-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}"]


def opt_connect_timeout(extra: Sequence[str]) -> Optional[str]:
    """The ConnectTimeout (seconds) the extra ssh options set, else None."""
    m = re.search(r"(?i)connecttimeout\s*=?\s*(\d+)", " ".join(str(x) for x in extra))
    return m.group(1) if m else None


def config_connect_timeout(host: str, runner: Callable = subprocess.run) -> Optional[str]:
    """The ConnectTimeout (seconds) your ssh config sets for ``host`` (``ssh
    -G``), else None: none set, or ssh -G failed (the default then applies)."""
    if not host:
        return None
    v = (ssh_g(host, runner).get("connecttimeout") or [""])[0]
    return v if v.isdigit() and int(v) > 0 else None


def ssh_destination(profile: Dict[str, Any], hop: str, user_for_node: Optional[str] = None) -> str:
    """The host the connection's command-line options apply to: the node for
    the jump hop, else the gateway (or the node without one)."""
    gateway = str(profile.get("gateway") or "")
    node = profile.get("node")
    if node and (not gateway or hop != "nested"):
        return node_target(profile, user_for_node)
    return gateway


def node_target(profile: Dict[str, Any], user_for_node: Optional[str]) -> str:
    node = str(profile["node"])
    if "@" in node or not user_for_node:
        return node
    return f"{user_for_node}@{node}"


def build_ssh_argv(profile: Dict[str, Any], lp: int, rp: int, gp: int, hop: str,
                   user_for_node: Optional[str] = None, ssh: str = "ssh",
                   node_keys: Sequence[str] = (), timeout: Optional[str] = None) -> List[str]:
    """The one ssh command for a hop ("jump" or "nested"); no -n, no -t, so
    stdin and its EOF pass through every hop to the server. node_keys: the
    gateway's key options for the node hop of "jump" (node_key_options).
    timeout: the ConnectTimeout your ssh config sets for ssh_destination
    (config_connect_timeout): then no default is added, and the nested hop's
    inner ssh gets it (or --ssh-option's) too."""
    gateway = str(profile.get("gateway") or "")
    jumps = [j.strip() for j in str(profile.get("jump") or "").split(",") if j.strip()]
    extra = [str(x) for x in (profile.get("ssh_options") or [])]
    rcmd = remote_command(profile, rp)
    node = profile.get("node")
    argv = [ssh]
    base = _base_opts(extra, timeout)
    if not node:                                 # the server runs on the gateway itself
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + base + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}", gateway, rcmd]
    if not gateway:                              # a node with its own Host block: no hop
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + base + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}",
                                      node_target(profile, user_for_node), rcmd]
    if hop == "nested":
        inner_t = opt_connect_timeout(extra) or timeout or str(CONNECT_TIMEOUT_S)
        inner = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={inner_t}",
                            "-o", "ExitOnForwardFailure=yes",
                            "-L", f"127.0.0.1:{gp}:127.0.0.1:{rp}", str(node), rcmd])
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + base + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{gp}", gateway, inner]
    argv += ["-J", ",".join(jumps + [gateway])]
    return argv + base + list(node_keys) + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}",
                                                    node_target(profile, user_for_node), rcmd]


def _host_of(target: str) -> str:
    return target.split("@", 1)[-1] if target else ""


# The node's shell could not change to --dir: bash ("cd: <dir>: No such file
# or directory"), dash and sh ("cd: can't cd to <dir>"). Checked before the
# missing-mace test, whose "No such file or directory" this line also holds.
CD_FAILED_RE = re.compile(r"\bcd: (?:can't cd to (?P<a>.+?)\s*$|(?P<b>.+?): (?:No such file or directory|Not a "
                          r"directory|Permission denied)\s*$)", re.M)


def host_key_trust_command(profile: Dict[str, Any], hop: str, user_for_node: Optional[str]) -> str:
    """The ssh line that asks once, in a terminal, to trust the host key(s)
    this session needs: ``ssh -J <gateway> <node> true`` for the jump hop
    (the gateway's key, then the node's), else ``ssh <gateway> true``."""
    gateway = str(profile.get("gateway") or "")
    jumps = [j.strip() for j in str(profile.get("jump") or "").split(",") if j.strip()]
    node = profile.get("node")
    if node and gateway and hop != "nested":
        jumps = jumps + [gateway]
        target = node_target(profile, user_for_node)
    else:
        target = gateway or (node_target(profile, user_for_node) if node else "")
    argv = ["ssh"] + (["-J", ",".join(jumps)] if jumps else []) + [target, "true"]
    return shlex.join(argv)


# The host whose key ssh could not verify: named before "Host key
# verification failed." (a prompt that got no yes, or a check with no
# terminal to ask on). With -J the gateway leg is a separate ssh process
# that checks the gateway's key against the real known_hosts (the command
# line's -o options do not reach it), and it prints to the same stderr.
HOST_KEY_HOST_RE = re.compile(r"The authenticity of host '\[?(?P<a>[^' \]]+)\]?(?::\d+)? \((?P<ip>[^)]*)\)'"
                              r"|No \S+ host key is known for \[?(?P<b>[^ \]]+)\]?(?::\d+)? and")


def host_key_leg(stderr_lines: Sequence[str], node: str, gateway: str, gateway_hostname: str = "") -> str:
    """Which host's key failed: "node", "gateway" or "" (not named, or another host)."""
    nh, gh = _host_of(node).lower(), _host_of(gateway).lower()
    for ln in stderr_lines:
        m = HOST_KEY_HOST_RE.search(ln)
        if not m:
            continue
        h = (m.group("a") or m.group("b") or "").lower()
        if node and (h == nh or (m.group("ip") or "") == "<no hostip for proxy command>"):
            return "node"
        if h and h in (gh, gateway_hostname.lower()):
            return "gateway"
    return ""


def classify_failure(code: Optional[int], stderr_lines: Sequence[str], node: str = "", gateway: str = "",
                     stdout_lines: Sequence[str] = (), hop: str = "", trust_cmd: str = "",
                     gateway_cmd: str = "", gateway_hostname: str = "") -> Tuple[str, str]:
    """(kind, message) for a session that ended before the server was ready.
    hop: the hop this session used; trust_cmd: host_key_trust_command's line
    for this hop; gateway_cmd: the same for the gateway alone (hop nested);
    gateway_hostname: the gateway's HostName (ssh -G), which ssh's messages use."""
    err = "\n".join(stderr_lines)
    out = "\n".join(stdout_lines)
    both = err + "\n" + out
    where = node or gateway or "the node"
    if "unrecognized arguments: --managed" in both or "invalid choice: 'gui'" in both:
        return ("old-remote", f"The MACE on {where} is older than this connect (it has no managed GUI). Update it, "
                              f"or use the manual tunnel ({DOCS_MANUAL}).")
    if "MACE_GUI_ERROR port-busy" in out or code == 98:
        return ("port-busy", f"The chosen port is busy on {where}; trying another.")
    if "administratively prohibited" in err or "forwarding disabled" in err.lower():
        line = next((ln for ln in stderr_lines if "prohibited" in ln or "forwarding" in ln.lower()), "")
        return ("forwarding", f"{line.strip()}\nYour cluster turns off port forwarding on this host; ask your "
                              "administrators.")
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in err or re.search(r"Host key for \S+ has changed", err):
        return ("host-key", "ssh refused a host key that has CHANGED since this computer last connected (see "
                            "ssh's message above). That can mean an attack, or that the host was reinstalled: ask "
                            "your administrators before you remove the old key (ssh-keygen -R <host>).")
    if "Host key verification failed" in err:
        # never trusted silently (no automatic acceptance of new keys): the user checks and answers once
        jump = bool(node and gateway and hop != "nested")
        leg = host_key_leg(stderr_lines, node, gateway, gateway_hostname) if jump else ""
        tail = ("in this terminal, compare the fingerprint with your cluster's, answer yes, and run mace gui "
                "connect again.")
        why = (", and nobody answered yes to ssh's question" +
               (", or the question could not be shown here" if os.name == "nt" else "") + ". ")
        gw_cmd = gateway_cmd or f"ssh {gateway} true"
        if jump and leg == "gateway":
            return ("host-key", f"ssh could not verify the gateway {gateway}'s host key (see ssh's message above): "
                                "this computer has not connected to it before" + why +
                                f"Check the key, then trust it once: run\n    {gw_cmd}\n" + tail)
        if jump and not leg:
            return ("host-key", "ssh could not verify a host key (see ssh's message above): this computer has not "
                                "connected to that host before" + why + "With the jump hop this computer checks "
                                "both keys itself, the gateway's first, then the node's. Check them, then trust "
                                f"them once: run\n    {gw_cmd}\nthen\n    {trust_cmd or 'ssh -J <gateway> <node> true'}"
                                "\n" + tail + " Or use --hop nested (the gateway then checks the node's key "
                                "with its own list of known hosts; its own key is still checked here).")
        what = f"{node}'s" if jump else "the"
        msg = (f"ssh could not verify {what} host key (see ssh's message above): this computer has not connected "
               "to it before" + why)
        if jump:
            msg += ("With the jump hop, this computer checks the node's key itself, so its known_hosts needs "
                    "an entry for the node too. ")
        cmd = trust_cmd or ("ssh <host> true")
        msg += f"Check the key, then trust it once: run\n    {cmd}\n" + tail
        if jump:
            msg += (" Or use --hop nested: the gateway then logs in to the node and checks its key with the "
                    "gateway's own list of known hosts.")
        return ("host-key", msg)
    m = CD_FAILED_RE.search(err)
    if m:
        folder = (m.group("a") or m.group("b") or "").strip()
        return ("dir", f"The folder {folder} does not exist on {where} (the remote shell could not cd to it): "
                       "check --dir, which is a path on the node (~/... is the node's home folder).")
    if code == 127 or re.search(r"(mace|mace_cli|python\S*): (command )?not found|No such file or directory", both):
        return ("mace-not-found", f"The remote login shell on {where} did not find the mace command. Pass "
                                  "--mace <full path to mace_cli or mace>, or a --setup line that puts it on PATH "
                                  "(e.g. --setup 'source <MACE>/activate_mace.sh').")
    if "is not a directory" in both or re.search(r"cd: .*(No such file|not a directory)", both) or code == 2:
        return ("dir", f"The folder does not exist on {where}, or the arguments were refused: check --dir "
                       "(and --setup / --mace).")
    gone = unreachable(stderr_lines, node)
    if gone:
        return ("unreachable", unreachable_message(*gone))
    if node_auth_refused(stderr_lines, node, gateway):
        return ("node-auth", f"The node {node} did not accept a login through the gateway: with ssh -J the node "
                             f"checks your key from this computer, and refused it ({DOCS_NODE_PASSWORD}).")
    if key_auth_failed(code, stderr_lines):
        return ("key", KEY_HINT)
    if code == 255:
        return ("ssh", f"ssh could not connect: check the gateway name and that `ssh {gateway or where}` works "
                       "in this terminal.")
    return ("ended", f"The connection ended before MACE GUI was ready (exit code {code}).")


_UNREACHABLE = (
    (re.compile(r"Could not resolve hostname ([^:\s]+)"), "DNS: the name does not resolve"),
    (re.compile(r"connect to host (\S+) port \d+: (?:Connection|Operation) timed out"), "timeout"),
    (re.compile(r"Connection timed out during banner exchange|Connection to (\S+) port \d+ timed out"), "timeout"),
    (re.compile(r"connect to host (\S+) port \d+: Connection refused"), "connection refused"),
    (re.compile(r"connect to host (\S+) port \d+: (?:No route to host|Network is unreachable)"), "no route"),
)
_CHANNEL_FAILED = re.compile(r"open failed: (?:connect failed|administratively prohibited): (.+)")


def unreachable(stderr_lines: Sequence[str], node: str = "") -> Optional[Tuple[str, str]]:
    """(host, why) when ssh's messages say a host could not be reached: a name
    that does not resolve, a timeout, a refused connection, no route. Through
    a jump host ("channel 0: open failed: connect failed: ..."), the host is
    the node, as the gateway saw it."""
    for ln in stderr_lines:
        for rx, why in _UNREACHABLE:
            m = rx.search(ln)
            if m:
                host = next((g for g in m.groups() if g), "") or _banner_host(stderr_lines)
                return host, why
        m = _CHANNEL_FAILED.search(ln)
        if m and node:
            text = m.group(1).lower()
            why = ("DNS on the gateway: the name does not resolve" if "not known" in text or "servname" in text
                   or "resolve" in text else "timeout" if "timed out" in text else
                   "connection refused" if "refused" in text else "no route" if "route" in text or "unreachable"
                   in text else text.strip())
            return _host_of(node), why
    return None


def _banner_host(stderr_lines: Sequence[str]) -> str:
    for ln in stderr_lines:
        m = re.search(r"Connection to (\S+) port", ln)
        if m:
            return m.group(1)
    return "the host"


def unreachable_message(host: str, why: str) -> str:
    return f"cannot reach host {host} ({why}): check the name, and your network or VPN."


def node_auth_refused(stderr_lines: Sequence[str], node: str, gateway: str) -> bool:
    """The gateway login worked but the node refused ours (the auto-hop
    rule): a "Permission denied (" line that names the node, none naming the
    gateway, and no host-key failure."""
    if not node:
        return False
    nh, gh = _host_of(node), _host_of(gateway)
    if any("Host key verification failed" in ln for ln in stderr_lines):
        return False
    denied = [ln for ln in stderr_lines if "Permission denied (" in ln]
    if not denied:
        return False
    if gh and any(gh in ln.split(":")[0] for ln in denied):
        return False
    return any(nh and nh in ln.split(":")[0] for ln in denied)


# ssh's own words when it could not log in with a key: the server refused
# every key ("Permission denied (publickey...)"), or a key with a passphrase
# could not be unlocked (no terminal to ask on, a wrong passphrase, the agent
# declined). Only with ssh's exit code 255 (ssh itself failed).
KEY_FAILURE_RE = re.compile(r"Permission denied \([^)]*publickey|Enter passphrase for key|read_passphrase: can't open"
                            r"|incorrect passphrase supplied|agent refused operation")
KEY_HINT = ("ssh could not log in with your key (see ssh's message above). If the key has a passphrase, ssh "
            "cannot ask for it here: load it into an agent first, with eval \"$(ssh-agent)\" and ssh-add (on "
            "Windows: start the OpenSSH Authentication Agent service, then run ssh-add), then run mace gui connect "
            "again. A Host alias from ~/.ssh/config also works as the gateway.")


def key_auth_failed(code: Optional[int], stderr_lines: Sequence[str]) -> bool:
    """ssh failed (exit 255) because no key got it in (KEY_FAILURE_RE)."""
    return code == 255 and any(KEY_FAILURE_RE.search(ln) for ln in stderr_lines)


def ssh_config_hosts(path: Optional[Path] = None) -> List[str]:
    """The Host patterns of ~/.ssh/config (negated ones and Include files left out)."""
    p = path or Path(os.path.expanduser("~")) / ".ssh" / "config"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: List[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*Host(?:\s+|\s*=\s*)(.+)", line, re.I)
        if m:
            out += [w.strip('"') for w in m.group(1).split() if not w.startswith("!")]
    return out


def gateway_looks_like_email(gateway: str, hosts: Optional[Sequence[str]] = None) -> bool:
    """A gateway such as user@university.edu: an email address, not a login host.

    The heuristic: an email's domain is usually the bare registered name, two
    labels (university.edu), while a login host has a host label in front
    (login.university.edu, hpcc.example.edu). So user@<label>.<letters> is
    flagged unless an ~/.ssh/config Host pattern matches it (an alias the user
    chose). It misses emails at longer domains (me@cs.university.edu,
    me@university.ac.uk) and flags a real login host named by two labels
    (me@cluster.org); that is only a warning, which a Host alias silences."""
    if "@" not in gateway:
        return False
    host = gateway.rsplit("@", 1)[1].lower()
    if not re.fullmatch(r"[a-z0-9-]+\.[a-z]{2,}", host):
        return False
    pats = ssh_config_hosts() if hosts is None else hosts
    return not any(fnmatch.fnmatchcase(t, pat.lower()) for pat in pats for t in (host, gateway.lower()))


def email_gateway_warning(gateway: str, path: Path, name: Optional[str]) -> str:
    return (f"Warning: the gateway {gateway} looks like an email address. It should be user@login-host (your "
            "cluster's login node, e.g. netid@login.example.edu) or a Host alias from ~/.ssh/config. Connecting "
            f"anyway; to fix the profile in {path}, run: mace gui connect {name + ' ' if name else ''}"
            "--gateway <user@login-host> --save")


def gateway_unresolvable(gateway: str, jump: str = "", runner: Callable = subprocess.run,
                         resolver: Callable = socket.getaddrinfo,
                         hosts: Optional[Sequence[str]] = None) -> Optional[str]:
    """The gateway's host name when no ~/.ssh/config Host pattern matches it and
    it does not resolve (DNS) here; None otherwise, or when a jump host,
    ProxyJump or ProxyCommand reaches it (the name then resolves over there)."""
    if not gateway or jump:
        return None
    host = gateway.rsplit("@", 1)[-1]
    pats = ssh_config_hosts() if hosts is None else hosts
    if any(fnmatch.fnmatchcase(host.lower(), p.lower()) for p in pats):
        return None
    g = ssh_g(gateway, runner)
    if any((g.get(k) or ["none"])[0].lower() != "none" for k in ("proxyjump", "proxycommand")):
        return None
    name = (g.get("hostname") or [host])[0]
    try:
        resolver(name, 22)
    except socket.gaierror:
        return name
    except (OSError, UnicodeError):
        return None
    return None


def unresolvable_warning(host: str) -> str:
    return (f"Warning: {host} matches no Host in ~/.ssh/config and does not resolve (DNS): check the gateway's "
            "name (your cluster's login host, e.g. login.example.edu). Connecting anyway.")


def version_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(v).split("+")[0])[:3])


def version_note(lv: str, rv: str, where: str, home: str = "") -> Optional[str]:
    """A warning when MACE here and on the cluster differ (None when they match)."""
    if not lv or not rv or lv == rv:
        return None
    if version_tuple(rv) < version_tuple(lv):
        how = f"git -C {shlex.quote(home)} pull" if home else "git pull in its MACE_HOME folder"
        return (f"Warning: MACE on {where} is {rv}, older than {lv} here. The page comes from {where}, so its "
                f"features are {rv}'s; update the cluster install ({how}), then connect again.")
    return (f"Note: MACE here is {lv}, on {where} {rv}. That is fine: the page comes from {where} (update this "
            "computer's MACE for the newest connect fixes).")


def windows_ssh_too_old(version_text: str) -> bool:
    m = re.search(r"OpenSSH_for_Windows_(\d+)\.(\d+)", version_text or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) < (8, 1)


# ----------------------------------------------------------------- prompts

def first_run_prompts(ask: Callable[[str], str], say: Callable[[str], None] = say_line) -> Dict[str, Any]:
    say("No connection profile yet: a few questions (saved to "
        f"{config_path()}, no passwords).")
    prof: Dict[str, Any] = {}
    gw = ""
    while not gw:
        gw = ask("Gateway (your cluster's login host, e.g. me@hpcc.example.edu): ").strip()
    prof["gateway"] = gw
    j = ask("Jump host before the gateway (blank = none): ").strip()
    if j:
        prof["jump"] = j
    say("Login nodes are shared; prefer a dev node for MACE GUI.")
    n = ask("Node to run on (blank = the gateway itself): ").strip()
    if n:
        prof["node"] = n
    d = ask("Folder to open there (blank = your home folder): ").strip()
    if d:
        prof["dir"] = d
    s = ask("Setup line run first (blank = none; e.g. module load Python): ").strip()
    if s:
        prof["setup"] = s
    m = ask("mace command there (blank = mace): ").strip()
    if m:
        prof["mace"] = m
    name = ask("Save as profile name (blank = default): ").strip() or "default"
    if not NAME_RE.match(name):
        say("Profile names use letters, digits, - and _; saving as default.")
        name = "default"
    prof["_name"] = name
    return prof


# ----------------------------------------------------------------- Windows

def on_windows() -> bool:
    return os.name == "nt"


def powershell_line(argv: Sequence[str]) -> str:
    """argv as one line to paste into PowerShell (5.1 or 7): ``--%`` (the
    stop-parsing token) hands the rest to ssh.exe as is, quoted by the
    Microsoft C runtime rules ssh.exe parses (what subprocess passes too).
    Not for cmd.exe, whose own quoting differs."""
    return f"{argv[0]} --% {subprocess.list2cmdline(list(argv[1:]))}"


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)      # type: ignore[attr-defined]
    k.CreateJobObjectW.restype = ctypes.c_void_p
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k.SetInformationJobObject.restype = ctypes.c_int
    k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k.AssignProcessToJobObject.restype = ctypes.c_int
    k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def kill_with_this_process(handle: int, os_name: Optional[str] = None, kernel32=None) -> Optional[int]:
    """Windows: put a child process (ssh) in a new Job Object that kills it
    when the job's last handle closes, which happens when this process ends,
    also when it is killed (Task Manager, a closed window): ssh then does not
    stay behind holding the tunnel. The handle is kept open on purpose. A
    no-op elsewhere, and harmless when Windows refuses (None)."""
    if (os_name or os.name) != "nt" or not handle:
        return None
    try:
        k = kernel32 if kernel32 is not None else _kernel32()
        job = k.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not (k.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.pointer(info),
                                          ctypes.sizeof(info)) and k.AssignProcessToJobObject(job, handle)):
            k.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


# ----------------------------------------------------------------- the session

# A node's login profile that runs stty without a terminal (bash -l with no
# tty, as every hop runs it) prints this; before READY it is noise, dropped
# from the console but kept in stderr_lines. The scripts drop it too.
STTY_NOISE_RE = re.compile(r"^stty: .*standard input.*: Inappropriate ioctl for device\s*$")
# The local forward listens from the moment ssh connects (the jump hop: on
# this computer, to the node's 127.0.0.1): a connection that comes before the
# server listens, such as an old GUI tab retrying, makes ssh print this. Noise
# until the tunnel answers (Session.tunnel_confirmed); a real problem after.
REFUSED_NOISE_RE = re.compile(r"^channel \d+: open failed: connect failed: Connection refused\s*$")

class Session:
    """One ssh process and the protocol it carries."""

    def __init__(self, argv: List[str], say: Callable[[str], None] = say_line, env=None):
        self.argv = argv
        self.say = say
        self.stderr_lines: List[str] = []
        self.stdout_lines: List[str] = []
        self.events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        kw: Dict[str, Any] = {}
        if os.name == "posix":
            import signal

            def _child():
                # OpenSSH keeps an ignored SIGINT ignored: Ctrl+C reaches this
                # program, which then stops the server in an orderly way
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            kw["preexec_fn"] = _child
        elif os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, **kw)
        # Windows: ssh (and its -J helper) die with this process, even a killed one
        self.job = kill_with_this_process(int(getattr(self.proc, "_handle", 0) or 0)) if os.name == "nt" else None
        self.lock = threading.Lock()
        self._quiet = threading.Event()
        self._ready = threading.Event()         # READY seen: stderr is shown as it is from then on
        self._confirmed = threading.Event()     # the tunnel answered: refused channels are shown from then on
        self._held = ""                          # a partial stderr line that may turn out to be noise
        self._readers = [threading.Thread(target=self._read_err, name="mace-connect-stderr", daemon=True),
                         threading.Thread(target=self._read_out, name="mace-connect-stdout", daemon=True)]
        for t in self._readers:
            t.start()

    def finish_readers(self, timeout: float = 2.0) -> None:
        """Wait (up to timeout) for both pipes to end, then silence the
        readers: ssh's -J helper can keep stderr open after ssh exits, and a
        daemon thread that writes to the console while Python shuts down
        can abort it ("could not acquire lock for <stderr>")."""
        end = time.monotonic() + timeout
        for t in self._readers:
            if t is not threading.current_thread():
                t.join(max(0.0, end - time.monotonic()))
        self._quiet.set()

    def _echo_err(self, text: str) -> None:
        if self._quiet.is_set() or not text:
            return
        try:
            sys.stderr.write(text)
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    def tunnel_confirmed(self) -> None:
        """The server answered through the tunnel (wait_ready)."""
        self._confirmed.set()

    def _is_noise(self, line: str) -> bool:
        if STTY_NOISE_RE.match(line):
            return not self._ready.is_set()
        if REFUSED_NOISE_RE.match(line):
            return not self._confirmed.is_set()
        return False

    def _without_noise(self, text: str) -> str:
        """text without whole noise lines (the stty line before READY, refused
        forward channels before the tunnel answers); a partial line that
        starts like one waits for its newline, any other text shows at once
        (a prompt has no newline)."""
        text = self._held + text
        self._held = ""
        if self._confirmed.is_set():
            return text
        *lines, tail = text.split("\n")
        out = [ln + "\n" for ln in lines if not self._is_noise(ln)]
        if tail and any(w.startswith(tail[:len(w)]) for w in ("stty", "channel ")):
            self._held = tail
        else:
            out.append(tail)
        return "".join(out)

    def _read_err(self):
        # chunks, not lines: a prompt ssh writes here without a newline shows at once
        stream = self.proc.stderr
        read = getattr(stream, "read1", None) or stream.read
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        while True:
            try:
                chunk = read(4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            text = dec.decode(chunk)
            self._echo_err(self._without_noise(text.replace("\r\n", "\n")))
            buf += text
            *done, buf = buf.split("\n")
            self.stderr_lines.extend(ln.rstrip("\r") for ln in done)
        buf += dec.decode(b"", final=True)
        if self._held:
            self._echo_err("" if self._is_noise(self._held) else self._held)
            self._held = ""
        if buf.rstrip("\r"):
            self.stderr_lines.append(buf.rstrip("\r"))

    def _read_out(self):
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip("\r\n")
            self.stdout_lines.append(line)
            if line.startswith("MACE_GUI_"):
                tag, _, rest = line.partition(" ")
                if tag == "MACE_GUI_READY":
                    self._ready.set()
                try:
                    data = json.loads(rest) if rest.strip().startswith("{") else rest.strip()
                except ValueError:
                    data = rest.strip()
                self.events.put((tag, data))
            elif line.strip() and not self._quiet.is_set():
                self.say(f"remote: {line}")
        self.events.put(("EOF", None))

    def send(self, line: str) -> bool:
        with self.lock:
            try:
                self.proc.stdin.write((line + "\n").encode())
                self.proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

    def close_stdin(self) -> None:
        with self.lock:
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass

    def end(self, wait_s: float = 5.0) -> Optional[int]:
        self.close_stdin()
        try:
            return self.proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            pass
        self.proc.terminate()
        try:
            return self.proc.wait(3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(3)


def _http(url: str, token: str, timeout: float = 5.0, data: Optional[bytes] = None) -> Optional[Tuple[int, bytes]]:
    req = urllib.request.Request(url, data=data, headers={"X-Mace-Token": token, "X-Requested-With": "mace-gui"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (OSError, urllib.error.URLError, http.client.HTTPException):
        return None           # HTTPException: a reply cut off (a stopping server, a tunnel closing)


def wait_ready(base: str, token: str, limit: float = READY_POLL_S) -> bool:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        r = _http(base + "api/info", token, timeout=3)
        if r and r[0] == 200:
            return True
        time.sleep(0.3)
    return False


def running_tasks(base: str, token: str) -> List[Dict[str, Any]]:
    r = _http(base + "api/tasks", token, timeout=5)
    if not r or r[0] != 200:
        return []
    try:
        return [t for t in json.loads(r[1]).get("tasks", []) if t.get("state") in ("queued", "running")]
    except ValueError:
        return []


def open_url(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except Exception:
        return False


def local_version() -> str:
    try:
        from mace import __version__
        return __version__
    except Exception:
        return ""


STOP_MESSAGES = {
    "browser": "The server on {node} was stopped from the browser.",
    "walltime": "The server's SLURM job reached its time limit; MACE GUI stopped cleanly.",
    "tunnel-lost": ("Lost the forward to the job's node; the server keeps running in its job. Run mace gui "
                    "connect again to reattach."),
    "managed-disconnect": "The server stopped (no task was running).",
    "cpu-limit": ("The server reached the node's CPU-time limit and stopped cleanly. Run mace gui connect again "
                  "to start a fresh one."),
    "idle": "The server on {node} stopped after a long time without use.",
    "managed-stop": "Stopped the server on {node}.",
    "managed-done": "The server on {node} stopped.",
}


def run(profile: Dict[str, Any], args, say: Callable[[str], None] = say_line,
        ask: Optional[Callable[[str], str]] = None) -> int:
    """Connect (retrying a busy remote port and, with hop auto, the nested
    hop once), then serve until Ctrl+C or the server stops."""
    gateway = str(profile.get("gateway") or "")
    node = str(profile.get("node") or "")
    where = node or gateway
    hop = str(profile.get("hop") or "auto")
    hops = ["jump", "nested"] if hop == "auto" else [hop]
    lp = pick_local_port(int(profile.get("local_port") or DEFAULT_LOCAL_PORT), warn=say)
    user_for_node = effective_node_user(gateway) if (node and gateway and "@" not in node) else None
    keys = node_key_options(gateway, node_target(profile, user_for_node)) if (node and gateway and "jump" in hops) \
        else []
    # the gateway's HostName (an alias's real name), which ssh's host-key messages use
    gw_hostname = ((ssh_g(gateway).get("hostname") or [""])[0]) if (node and gateway and "jump" in hops) else ""
    if os.name == "nt" and (profile.get("jump") or (node and gateway and hop != "nested")):
        try:
            v = subprocess.run(["ssh", "-V"], capture_output=True, text=True, timeout=10)
            if windows_ssh_too_old(v.stderr + v.stdout):
                say("Your ssh is older than 8.1, whose -J is unreliable on Windows; update Windows OpenSSH or use "
                    "--hop nested.")
        except (OSError, subprocess.SubprocessError):
            pass
    bad = gateway_unresolvable(gateway, str(profile.get("jump") or "")) if gateway else None
    if bad:
        say(unresolvable_warning(bad))
    tries = 0
    hop_i = 0
    while True:
        cur = hops[hop_i] if node and gateway else "jump"
        rp, gp = random_port(), random_port()
        argv = build_ssh_argv(profile, lp, rp, gp, cur, user_for_node, node_keys=keys,
                              timeout=config_connect_timeout(ssh_destination(profile, cur, user_for_node)))
        via = f" through {gateway}" if node and gateway else ""
        say(f"Connecting to {where}{via}...")
        sess = Session(argv, say)
        try:
            outcome = _drive(sess, profile, args, lp, where, say, ask,
                             jump_node=node if (gateway and cur == "jump") else "", hop=cur,
                             trust_cmd=host_key_trust_command(profile, cur, user_for_node),
                             gateway_cmd=host_key_trust_command(profile, "nested", user_for_node),
                             gateway_hostname=gw_hostname)
        finally:
            # the readers are daemon threads: none may write to the console while Python shuts down
            sess.finish_readers()
        if isinstance(outcome, int):
            return outcome
        kind, msg = outcome
        if kind == "port-busy" and tries < 3:
            tries += 1
            continue
        # auto falls back on the node's refusal of the login only: a host-key
        # failure is a trust decision (or a changed key), which another hop
        # must not paper over; classify_failure says how to trust it once
        if kind == "node-auth" and hop == "auto" and hop_i == 0 and len(hops) > 1:
            say("The node did not accept a login through the gateway; connecting through the gateway instead "
                "(hop = nested). Save it with --save so this is not tried twice.")
            profile["hop"] = "nested"
            hop_i = 1
            continue
        say(f"mace gui connect: {msg}")
        if kind == "mace-not-found" and profile.get("_localized"):
            a, b = profile["_localized"][0]
            say(f"{a} was a path on this computer (your shell expands an unquoted ~ here), so it was tried as {b} on "
                f"{where}: check that it exists there. Quote the ~ ('~/...') so that your shell passes it as is.")
        if kind == "node-auth":
            say("If your cluster's nodes accept logins only from the login node, use --hop nested.")
        if kind == "mace-not-found" and mace_fallback(shlex.split(str(profile.get("mace") or "mace")) or ["mace"]):
            say("It also looked for mace_cli in $MACE_HOME, ~/mace and ~/MACE there.")
        if kind in ("unreachable", "ssh"):
            say("mace gui connect --check (with this profile or these flags) tests each step of the connection.")
        if os.name == "nt" and kind in ("ssh", "ended") and not any(
                "password" in ln.lower() or "verification" in ln.lower() for ln in sess.stderr_lines):
            say("Windows ssh cannot always show a password or 2FA prompt here. Set up an ssh key or agent "
                f"({DOCS_WINDOWS}), or use the two-terminal manual tunnel.")
        return {"old-remote": 3, "mace-not-found": 127, "dir": 2}.get(kind, 1)


def waiting_note(jump_node: str = "") -> str:
    """Shown once when nothing came back for SLOW_NOTE_S: ssh may be waiting
    for an answer in this terminal (2FA, or the host-key question that a
    node reached with ssh -J for the first time asks)."""
    note = ("(waiting: approve the 2FA prompt if there is one. If ssh asks whether to continue connecting "
            "(the host-key yes/no question), check the fingerprint and answer yes")
    if jump_node:
        note += (f". If ssh asks for {jump_node}'s password, your key is not accepted on {jump_node} from this "
                 f"computer: see {DOCS_NODE_PASSWORD}")
    return note + ")"


def _drive(sess: Session, profile, args, lp: int, where: str, say, ask, jump_node: str = "",
           hop: str = "", trust_cmd: str = "", gateway_cmd: str = "", gateway_hostname: str = "") -> Any:
    """Wait for READY, then serve; an int exit code, or (kind, message) on failure before READY."""
    t0 = time.monotonic()
    noted = False
    ready = None
    job = None
    pending = None
    while ready is None:
        try:
            tag, data = sess.events.get(timeout=1.0)
        except queue.Empty:
            if not noted and job is None and time.monotonic() - t0 > SLOW_NOTE_S:
                say(waiting_note(jump_node))
                noted = True
            continue
        except KeyboardInterrupt:
            sess.end(15 if job else 2)      # closing stdin cancels a pending job (slurmsession)
            say(f"Canceled{'; the pending job ' + str(job) + ' was canceled' if job else ''}.")
            return 130
        if tag == "MACE_GUI_JOB" and isinstance(data, dict):
            job = data.get("job")
            say(f"Submitted job {job} for MACE GUI.")
        elif tag == "MACE_GUI_PENDING" and isinstance(data, dict):
            cur = (data.get("reason"), data.get("start"))
            if cur != pending:
                pending = cur
                say(f"Waiting for a compute node (job {data.get('job')}, pending: {data.get('reason')}, expected "
                    f"start {data.get('start')}). Press Ctrl+C to give up.")
        elif tag == "MACE_GUI_ERROR" and isinstance(data, dict) and data.get("kind") == "port-busy":
            sess.end(5)          # --slurm: the port is taken on the node; another one (a running job is reattached)
            return ("port-busy", f"The chosen port is busy on {where}; trying another.")
        elif tag == "MACE_GUI_ERROR" and isinstance(data, dict) and data.get("message"):
            sess.end(5)
            say(f"mace gui connect: {data['message']}")
            return 2 if data.get("kind") in ("args", "dir") else 130 if data.get("kind") == "canceled" else 1
        elif tag == "MACE_GUI_READY" and isinstance(data, dict):
            ready = data
        elif tag == "MACE_GUI_USING" and data:
            say(f"mace is not on {where}'s PATH there; using {data}. To skip the search, save it: mace gui connect "
                f"--mace {shlex.quote(str(data))} --save")
        elif tag == "EOF":
            code = sess.proc.wait()
            sess.finish_readers()        # every stderr line in, before it is classified
            return classify_failure(code, sess.stderr_lines, str(profile.get("node") or ""),
                                    str(profile.get("gateway") or ""), sess.stdout_lines, hop=hop,
                                    trust_cmd=trust_cmd, gateway_cmd=gateway_cmd,
                                    gateway_hostname=gateway_hostname)
        elif tag == "MACE_GUI_ERROR":
            sess.end(3)
            return classify_failure(98 if "port-busy" in str(data) else 1, sess.stderr_lines,
                                    str(profile.get("node") or ""), str(profile.get("gateway") or ""),
                                    sess.stdout_lines + [f"MACE_GUI_ERROR {data}"])
    prefix = str(ready.get("prefix") or "/")
    token = str(ready.get("token") or "")
    base = f"http://127.0.0.1:{lp}{prefix}"
    ok = wait_ready(base, token)
    if ok:
        sess.tunnel_confirmed()
    if not ok:
        refused = [ln for ln in sess.stderr_lines if REFUSED_NOISE_RE.match(ln)]
        for ln in refused[-3:]:              # quiet while waiting; part of the diagnosis now
            say(ln)
        say(f"mace gui connect: the server on {where} started, but the tunnel to port {lp} does not answer.")
        sess.send("stop")
        sess.end()
        return 1
    login = ready.get("login")
    # a one-time sign-in link when the server has them; an older server takes the token in the URL
    url = _local_url(login, lp) if login else f"{base}?token={token}"
    # `mace gui login-url` on this computer mints links through the tunnel
    private = _write_login_state(base, token, lp, str(ready.get("root") or ""), where) if login else None
    try:
        return _serve_session(sess, profile, args, lp, where, say, ask, ready, base, token, url, login)
    finally:
        if private is not None:
            try:
                private.unlink()
            except OSError:
                pass


def _write_login_state(base: str, token: str, lp: int, root: str, where: str) -> Optional[Path]:
    """The private state file `mace gui login-url` reads (0600, removed when
    connect ends), naming the forwarded port: the remote server's sign-in
    links work only through this tunnel."""
    from .cli import state_file, write_state_file
    path = state_file(lp)
    try:
        write_state_file(path, {"base": base, "token": token, "pid": os.getpid(), "port": lp,
                                "root": f"{root} on {where}" if root else where, "via": "connect"})
    except OSError:
        return None
    return path


def _serve_session(sess: Session, profile, args, lp: int, where: str, say, ask, ready: Dict[str, Any],
                   base: str, token: str, url: str, login: Any) -> Any:
    if ready.get("reattached"):
        n = int(ready.get("tasks") or 0)
        say(f"Reattached to the server still running on {where} ({n} task{'s' if n != 1 else ''} running).")
    note = version_note(local_version(), str(ready.get("version") or ""), where, str(ready.get("home") or ""))
    if note:
        say(note)
    if ready.get("job"):
        say(f"MACE GUI runs as job {ready['job']} on {ready.get('node')}"
            + (f" until {ready['ends']}" if ready.get("ends") else "") + ".")
        # --slurm: the server is on the job's node, not the node ssh reached
        where = str(ready.get("node") or where)
    elif not profile.get("node"):
        say("Note: this runs on the login node, which is shared; prefer a dev node (--node).")
    if not getattr(args, "no_open", False) and profile.get("open", True) is not False:
        open_url(url)
    say(f"MACE GUI on {where}: {url}")
    if login:
        say("(the link signs in once; press Enter here to open MACE GUI in the browser again)")
    if ready.get("job"):
        say(f"Press Ctrl+C to stop MACE GUI; its job {ready['job']} ends and the allocation is released.")
    else:
        say(f"Press Ctrl+C to stop the server on {where}.")
    stop_ev = threading.Event()

    def pinger():
        while not stop_ev.wait(PING_S):
            if not sess.send("ping"):
                return
    threading.Thread(target=pinger, daemon=True).start()
    keys = _KeyLines() if (sys.stdin is not None and sys.stdin.isatty()) else None
    if keys is not None:
        keys.on_line = lambda _line: sess.send("login")
        keys.start()
    try:
        while True:
            tag, data = sess.events.get()
            if tag == "MACE_GUI_LOGIN" and isinstance(data, dict) and data.get("url"):
                u = _local_url(data["url"], lp)
                say(f"New sign-in link: {u}")
                if not getattr(args, "no_open", False):
                    open_url(u)
            elif tag == "MACE_GUI_STOPPED":
                reason = data.get("reason") if isinstance(data, dict) else str(data)
                stop_ev.set()
                sess.end(5)
                msg = STOP_MESSAGES.get(reason, "The server on {node} stopped.").format(node=where)
                say(msg)
                return 0
            elif tag == "EOF":
                stop_ev.set()
                sess.proc.wait()
                say(f"The connection to {where} closed. If GUI tasks were running, the server keeps them "
                    "running: run mace gui connect again to reattach.")
                return 1
    except KeyboardInterrupt:
        stop_ev.set()
        if keys is not None:
            keys.on_line = None
            ask = keys.ask
        return _interrupt(sess, base, token, where, say, ask)


class _KeyLines:
    """One reader of this terminal's lines (Enter while serving opens a new
    sign-in link; after Ctrl+C the same reader answers the question), so two
    readers never compete for a line."""

    def __init__(self):
        self.on_line: Optional[Callable[[str], None]] = None
        self.answers: "queue.Queue[str]" = queue.Queue()

    def start(self):
        def read():
            while True:
                try:
                    line = sys.stdin.readline()
                except (OSError, ValueError):
                    return
                if not line:
                    return
                cb = self.on_line
                if cb is not None:
                    cb(line)
                else:
                    self.answers.put(line)
        threading.Thread(target=read, name="mace-connect-keys", daemon=True).start()

    def ask(self, prompt: str) -> str:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            return self.answers.get(timeout=120)
        except queue.Empty:
            return ""


def _local_url(url: str, lp: int) -> str:
    """The server's URL with this computer's forwarded port."""
    return re.sub(r"^http://(?:127\.0\.0\.1|localhost|\[::1\]):\d+", f"http://127.0.0.1:{lp}", url)


def _interrupt(sess: Session, base: str, token: str, where: str, say, ask) -> int:
    tasks = running_tasks(base, token)
    if tasks:
        kinds = ", ".join(sorted({str(t.get("kind") or "task") for t in tasks}))
        answer = "n"
        if ask is not None and sys.stdin is not None and sys.stdin.isatty():
            try:
                answer = ask(f"{len(tasks)} task{'s are' if len(tasks) != 1 else ' is'} running on {where} "
                             f"({kinds}). Stop them too? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
        if answer.startswith("y"):
            sess.send("stop")
            sess.end(10)
            say(f"Stopped the server on {where} and its tasks.")
        else:
            sess.send("detach")
            sess.end(5)
            say(f"Left the server running on {where} for its tasks; it stops when they finish "
                "(run mace gui connect again to reattach).")
        return 0
    sess.send("stop")
    sess.end(5)
    say(f"Stopped the server on {where}.")
    return 0


# ----------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mace gui connect",
        description="Start MACE GUI on a cluster node and open it in this computer's browser, with one ssh "
                    "session (start, tunnel and stop in one step). Needs ssh and Python 3 here; MACE on the "
                    "node.",
        epilog="The first run asks for the gateway, node and folder and saves them as a profile in "
               f"{config_path()} (no passwords). Ctrl+C stops the server on the node; when GUI tasks are "
               "still running it asks first, and a dropped connection leaves the server running only for "
               "them (mace gui connect again reattaches). See docs/gui.md#first-run-from-your-computer.")
    p.add_argument("profile", nargs="?", help="a saved profile (default: the [defaults] profile)")
    p.add_argument("--gateway", help="your cluster's login host (user@host or an ~/.ssh/config Host)")
    p.add_argument("--jump", help="bastion host(s) before the gateway, comma separated")
    p.add_argument("--node", help="the node to run on, e.g. a dev node (default: the gateway itself)")
    p.add_argument("--hop", choices=HOPS, help="how to reach the node: jump (ssh -J), nested (the gateway "
                                                "logs in to the node), auto (jump, then nested; default)")
    p.add_argument("--dir", help="folder to serve on the node (default: your home folder; ~/... allowed)")
    p.add_argument("--mace", help="the mace command on the node (default: mace)")
    p.add_argument("--setup", help="shell line(s) run first on the node, e.g. 'module load Python'")
    p.add_argument("--conda", help="a conda environment to activate on the node")
    p.add_argument("--local-port", type=int, help=f"port on this computer (default {DEFAULT_LOCAL_PORT}; "
                                                  "the next free one when busy)")
    p.add_argument("--ssh-option", action="append", default=None, metavar="OPT",
                   help="an extra ssh argument (repeatable), e.g. --ssh-option=-oIdentityFile=~/.ssh/k")
    p.add_argument("--root", action="append", default=None, metavar="DIR",
                   help="a folder File > Open Folder may switch to (repeatable; none turns it off)")
    p.add_argument("--no-open", action="store_true", help="print the URL; do not open a browser")
    g = p.add_argument_group("run MACE GUI as a SLURM batch job (mace gui --slurm on the node or gateway)")
    g.add_argument("--slurm", action="store_true", default=None,
                   help="submit MACE GUI as a small batch job, wait for a compute node, and tunnel to it; the "
                        "job ends (its allocation is released) when the server stops (profile: slurm = true)")
    g.add_argument("--time", help="the job's time limit (default 4:00:00)")
    g.add_argument("--cpus", type=int, help="CPUs (default 2)")
    g.add_argument("--mem", help="memory (default 4G)")
    g.add_argument("--partition", help="partition (default: the node's site settings, else SLURM's default)")
    g.add_argument("--account", help="account (default: the node's site settings, else your default account)")
    g.add_argument("--qos", help="QOS")
    g.add_argument("--constraint", help="node features")
    g.add_argument("--gpus", help="GPUs, e.g. 1 (default: none)")
    g.add_argument("--sbatch-arg", action="append", default=None, metavar="ARG",
                   help="one more sbatch option as --name=value (repeatable)")
    p.add_argument("--save", nargs="?", const="", metavar="NAME",
                   help="save these settings as a profile (default: the profile used, else 'default')")
    p.add_argument("--list", action="store_true", help="list the saved profiles")
    p.add_argument("--forget", metavar="NAME", help="delete a saved profile")
    p.add_argument("--dry-run", action="store_true", help="print the ssh command(s) and exit")
    c = p.add_argument_group("check the ssh setup (read-only)")
    c.add_argument("--check", action="store_true",
                   help="check ssh, its config, keys, agent and permissions for this profile, try each hop with "
                        "only `true`, and print OK/WARN/FAIL lines, each problem with one fix line, and the "
                        "connect line that works")
    c.add_argument("--no-net", action="store_true", help="with --check: no ssh logins or DNS lookups")
    c.add_argument("--fix-config", action="store_true",
                   help="print a suggested ~/.ssh/config block for this profile, from ssh -G (never written)")
    p.add_argument("--export-script", choices=("bash", "powershell"),
                   help="print a standalone connect script with this profile filled in (for a computer "
                        "without MACE)")
    p.add_argument("-o", "--output", help="with --export-script: write it to this file")
    return p


def _flags_profile(args) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda"):
        v = getattr(args, k)
        if v is not None:
            out[k] = v
    if args.local_port is not None:
        out["local_port"] = int(args.local_port)
    if args.ssh_option:
        out["ssh_options"] = list(args.ssh_option)
    if args.root:
        out["roots"] = list(args.root)
    if args.no_open:
        out["open"] = False
    if args.slurm:
        out["slurm"] = True
    for k in ("time", "cpus", "mem", "partition", "account", "qos", "constraint", "gpus"):
        v = getattr(args, k, None)
        if v is not None:
            out[f"slurm_{k}"] = v
    if args.sbatch_arg:
        out["slurm_args"] = list(args.sbatch_arg)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = config_path()
    data = load_profiles(path)
    profiles = data.setdefault("profiles", {}) if isinstance(data.get("profiles", {}), dict) else {}
    if args.list:
        default = default_profile_name(data)
        if not profiles:
            print(f"No profiles yet ({path}).")
        for name, prof in profiles.items():
            mark = " (default)" if name == default else ""
            print(f"{name}{mark}: {prof.get('gateway', '')} -> {prof.get('node') or '(gateway)'} "
                  f"{prof.get('dir') or '~'}")
        return 0
    if args.forget:
        if args.forget not in profiles:
            print(f"mace gui connect: no profile {args.forget}", file=sys.stderr)
            return 2
        del profiles[args.forget]
        if (data.get("defaults") or {}).get("profile") == args.forget:
            data["defaults"].pop("profile", None)
        save_profiles(data, path)
        print(f"Removed the profile {args.forget}.")
        return 0
    name = args.profile or default_profile_name(data)
    if args.profile and args.profile not in profiles and not args.gateway:
        print(f"mace gui connect: no profile {args.profile} (mace gui connect --list)", file=sys.stderr)
        return 2
    prof: Dict[str, Any] = dict(profiles.get(name, {})) if name else {}
    prof.update(_flags_profile(args))
    new_name = None
    if not prof.get("gateway") and not prof.get("node"):
        tty = bool(sys.stdin and sys.stdin.isatty())
        if args.export_script or args.dry_run or args.check or args.fix_config or not tty:
            print("mace gui connect: no profile yet. Pass --gateway <login host> (and usually --node <dev node> "
                  "and --dir <folder>), or run it in a terminal to answer a few questions.", file=sys.stderr)
            return 2
        got = first_run_prompts(input)
        new_name = got.pop("_name")
        prof.update(got)
        if args.save is None:
            args.save = new_name
    moved = localize_home_paths(prof)
    if moved:
        print(localized_note(moved), file=sys.stderr)
    if not prof.get("slurm") and any(prof.get(k) not in (None, "", []) for k in SLURM_KEYS):
        print("mace gui connect: --time, --cpus, --mem, --partition, --account, --qos, --constraint, --gpus and "
              "--sbatch-arg go with --slurm", file=sys.stderr)
        return 2
    if prof.get("hop") and prof["hop"] not in HOPS:
        print(f"mace gui connect: hop must be one of {', '.join(HOPS)}", file=sys.stderr)
        return 2
    if args.save is not None:
        save_as = args.save or name or "default"
        if not NAME_RE.match(save_as):
            print("mace gui connect: profile names use letters, digits, - and _", file=sys.stderr)
            return 2
        keep = {k: v for k, v in profiles.get(save_as, {}).items() if k not in PROFILE_KEYS}
        profiles[save_as] = {**keep, **{k: v for k, v in prof.items() if k in PROFILE_KEYS}}
        data.setdefault("defaults", {})
        if not data["defaults"].get("profile") or new_name:
            data["defaults"]["profile"] = save_as
        save_profiles(data, path)
        print(f"Saved the profile {save_as} in {path}.")
    if args.export_script:
        from .connect_scripts import export_script
        text = export_script(args.export_script, prof)
        if args.output:
            out = Path(args.output).expanduser()
            out.write_text(text, encoding="utf-8", newline="\n" if args.export_script == "bash" else "\r\n")
            if args.export_script == "bash":
                os.chmod(out, 0o755)
            print(f"Wrote {out}.")
        else:
            sys.stdout.write(text)
        return 0
    checking = args.check or args.fix_config
    if prof.get("gateway") and not checking and gateway_looks_like_email(str(prof["gateway"])):
        print(email_gateway_warning(str(prof["gateway"]), path, args.save or name), file=sys.stderr)
    if args.check or args.fix_config:
        from .connect_check import main as check_main
        return check_main(prof, args.save or name, check=args.check, no_net=args.no_net,
                          fix_config=args.fix_config)
    if args.dry_run:
        node, gateway = prof.get("node"), prof.get("gateway")
        hop = str(prof.get("hop") or "auto")
        lp = int(prof.get("local_port") or DEFAULT_LOCAL_PORT)
        unode = effective_node_user(str(gateway)) if (node and gateway and "@" not in str(node)) else None
        keys = node_key_options(str(gateway), node_target(prof, unode)) if (node and gateway and hop != "nested") \
            else []
        for h in (["jump", "nested"] if hop == "auto" else [hop]) if (node and gateway) else ["jump"]:
            argv = build_ssh_argv(prof, lp, 40000, 40001, h, unode, node_keys=keys,
                                  timeout=config_connect_timeout(ssh_destination(prof, h, unode)))
            label = f"hop {h}" if node and gateway else "no hop"
            print(f"# {label} (the remote port is random on a real run)")
            if on_windows():
                print("# for bash, zsh, Git Bash, MobaXterm or WSL (POSIX quoting):")
            print(shlex.join(argv))
            if on_windows():
                print("# for PowerShell 5.1 or 7 (--% passes the rest to ssh.exe as is; not for cmd.exe):")
                print(powershell_line(argv))
        print("# remote command")
        print(remote_command(prof, 40000))
        return 0
    return run(prof, args, ask=input)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
