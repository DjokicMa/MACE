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
``%APPDATA%\\mace\\connect.toml``), mode 0600, no secrets.

Standard library only: this runs on your own computer, which needs Python 3
and ssh but not MACE's scientific packages.
"""
from __future__ import annotations

import argparse
import fnmatch
import getpass
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
# --slurm: run the server as a batch job (slurmsession.py on the node or gateway)
SLURM_KEYS = ("slurm_time", "slurm_cpus", "slurm_mem", "slurm_partition", "slurm_account", "slurm_qos",
              "slurm_constraint", "slurm_gpus", "slurm_args")
PROFILE_KEYS = ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda", "local_port", "ssh_options",
                "open", "roots", "remote_shell", "slurm") + SLURM_KEYS
HOPS = ("auto", "jump", "nested")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DOCS_MANUAL = "docs/gui.md#run-it-on-a-cluster-and-view-it-on-your-computer"
DOCS_WINDOWS = "docs/gui.md#windows-powershell"
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
    out, i = [], 0
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            n = text[i + 1]
            if n == "u" and i + 5 < len(text):
                out.append(chr(int(text[i + 2:i + 6], 16)))
                i += 6
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
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
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


def _home_path(text: str) -> str:
    """A shell word for a path; "~" and "~/..." become "$HOME"-relative, the rest is literal."""
    if text == "~":
        return '"$HOME"'
    if text.startswith("~/"):
        return '"$HOME"/' + shlex.quote(text[2:])
    return shlex.quote(text)


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
        parts.append("conda activate " + shlex.quote(str(profile["conda"])))
    d = str(profile.get("dir") or "~")
    mace = shlex.split(str(profile.get("mace") or "mace")) or ["mace"]
    cmd = ["exec"]
    if not profile.get("node") and not profile.get("slurm"):
        cmd += ["env", "MACE_GUI_MAX_WORKERS=2"]          # a shared login node: two workers at most
    # --no-banner: the credits banner would only show as "remote:" lines
    cmd += [_home_path(w) for w in mace] + ["--no-banner", "gui"]
    if profile.get("slurm"):
        cmd += ["--slurm"] + slurm_args(profile)
    cmd += ["--managed", "--port", str(int(rp)), "--dir", "."]
    for r in profile.get("roots") or []:
        cmd += ["--root", _home_path(str(r))]
    parts.append(f"cd -- {_home_path(d)} && " + " ".join(cmd))
    script = "; ".join(parts)
    if (profile.get("remote_shell") or "bash") == "sh":
        return "exec sh -c " + shlex.quote(script)
    return "exec bash -lc " + shlex.quote(script)


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


def _base_opts() -> List[str]:
    return ["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]


def node_target(profile: Dict[str, Any], user_for_node: Optional[str]) -> str:
    node = str(profile["node"])
    if "@" in node or not user_for_node:
        return node
    return f"{user_for_node}@{node}"


def build_ssh_argv(profile: Dict[str, Any], lp: int, rp: int, gp: int, hop: str,
                   user_for_node: Optional[str] = None, ssh: str = "ssh") -> List[str]:
    """The one ssh command for a hop ("jump" or "nested"); no -n, no -t, so
    stdin and its EOF pass through every hop to the server."""
    gateway = str(profile.get("gateway") or "")
    jumps = [j.strip() for j in str(profile.get("jump") or "").split(",") if j.strip()]
    extra = [str(x) for x in (profile.get("ssh_options") or [])]
    rcmd = remote_command(profile, rp)
    node = profile.get("node")
    argv = [ssh]
    if not node:                                 # the server runs on the gateway itself
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + _base_opts() + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}", gateway, rcmd]
    if not gateway:                              # a node with its own Host block: no hop
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + _base_opts() + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}",
                                              node_target(profile, user_for_node), rcmd]
    if hop == "nested":
        inner = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                            "-L", f"127.0.0.1:{gp}:127.0.0.1:{rp}", str(node), rcmd])
        if jumps:
            argv += ["-J", ",".join(jumps)]
        return argv + _base_opts() + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{gp}", gateway, inner]
    argv += ["-J", ",".join(jumps + [gateway])]
    return argv + _base_opts() + extra + ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}",
                                          node_target(profile, user_for_node), rcmd]


def _host_of(target: str) -> str:
    return target.split("@", 1)[-1] if target else ""


def classify_failure(code: Optional[int], stderr_lines: Sequence[str], node: str = "", gateway: str = "",
                     stdout_lines: Sequence[str] = ()) -> Tuple[str, str]:
    """(kind, message) for a session that ended before the server was ready."""
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
    if "Host key verification failed" in err or "REMOTE HOST IDENTIFICATION HAS CHANGED" in err:
        return ("host-key", "ssh refused the host key (see ssh's message above). Check it, then run "
                            "`ssh <host>` once in this terminal to accept the right one.")
    if code == 127 or re.search(r"(mace|mace_cli|python\S*): (command )?not found|No such file or directory", both):
        return ("mace-not-found", f"The remote login shell on {where} did not find the mace command. Pass "
                                  "--mace <full path to mace_cli or mace>, or a --setup line that puts it on PATH "
                                  "(e.g. --setup 'source <MACE>/activate_mace.sh').")
    if "is not a directory" in both or re.search(r"cd: .*(No such file|not a directory)", both) or code == 2:
        return ("dir", f"The folder does not exist on {where}, or the arguments were refused: check --dir "
                       "(and --setup / --mace).")
    if node_auth_refused(stderr_lines, node, gateway):
        return ("node-auth", f"The node {node} did not accept a login through the gateway.")
    if key_auth_failed(code, stderr_lines):
        return ("key", KEY_HINT)
    if code == 255:
        return ("ssh", f"ssh could not connect: check the gateway name and that `ssh {gateway or where}` works "
                       "in this terminal.")
    return ("ended", f"The connection ended before MACE GUI was ready (exit code {code}).")


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


# ----------------------------------------------------------------- the session

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
        threading.Thread(target=self._read_err, daemon=True).start()
        threading.Thread(target=self._read_out, daemon=True).start()
        self.lock = threading.Lock()

    def _read_err(self):
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode(errors="replace").rstrip("\r\n")
            self.stderr_lines.append(line)
            try:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except (OSError, ValueError):
                pass

    def _read_out(self):
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip("\r\n")
            self.stdout_lines.append(line)
            if line.startswith("MACE_GUI_"):
                tag, _, rest = line.partition(" ")
                try:
                    data = json.loads(rest) if rest.strip().startswith("{") else rest.strip()
                except ValueError:
                    data = rest.strip()
                self.events.put((tag, data))
            elif line.strip():
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
    except (OSError, urllib.error.URLError):
        return None


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
    if os.name == "nt" and (profile.get("jump") or (node and gateway and hop != "nested")):
        try:
            v = subprocess.run(["ssh", "-V"], capture_output=True, text=True, timeout=10)
            if windows_ssh_too_old(v.stderr + v.stdout):
                say("Your ssh is older than 8.1, whose -J is unreliable on Windows; update Windows OpenSSH or use "
                    "--hop nested.")
        except (OSError, subprocess.SubprocessError):
            pass
    tries = 0
    hop_i = 0
    while True:
        cur = hops[hop_i] if node and gateway else "jump"
        rp, gp = random_port(), random_port()
        argv = build_ssh_argv(profile, lp, rp, gp, cur, user_for_node)
        via = f" through {gateway}" if node and gateway else ""
        say(f"Connecting to {where}{via}...")
        sess = Session(argv, say)
        outcome = _drive(sess, profile, args, lp, where, say, ask)
        if isinstance(outcome, int):
            return outcome
        kind, msg = outcome
        if kind == "port-busy" and tries < 3:
            tries += 1
            continue
        if kind == "node-auth" and hop == "auto" and hop_i == 0 and len(hops) > 1:
            say("The node did not accept a login through the gateway; connecting through the gateway instead "
                "(hop = nested). Save it with --save so this is not tried twice.")
            profile["hop"] = "nested"
            hop_i = 1
            continue
        say(f"mace gui connect: {msg}")
        if kind == "node-auth":
            say("If your cluster's nodes accept logins only from the login node, use --hop nested.")
        if os.name == "nt" and kind in ("ssh", "ended") and not any(
                "password" in ln.lower() or "verification" in ln.lower() for ln in sess.stderr_lines):
            say("Windows ssh cannot always show a password or 2FA prompt here. Set up an ssh key or agent "
                f"({DOCS_WINDOWS}), or use the two-terminal manual tunnel.")
        return {"old-remote": 3, "mace-not-found": 127, "dir": 2}.get(kind, 1)


def _drive(sess: Session, profile, args, lp: int, where: str, say, ask) -> Any:
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
                say("(waiting: approve the 2FA prompt if there is one)")
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
        elif tag == "EOF":
            code = sess.proc.wait()
            return classify_failure(code, sess.stderr_lines, str(profile.get("node") or ""),
                                    str(profile.get("gateway") or ""), sess.stdout_lines)
        elif tag == "MACE_GUI_ERROR":
            sess.end(3)
            return classify_failure(98 if "port-busy" in str(data) else 1, sess.stderr_lines,
                                    str(profile.get("node") or ""), str(profile.get("gateway") or ""),
                                    sess.stdout_lines + [f"MACE_GUI_ERROR {data}"])
    prefix = str(ready.get("prefix") or "/")
    token = str(ready.get("token") or "")
    base = f"http://127.0.0.1:{lp}{prefix}"
    if not wait_ready(base, token):
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
    lv, rv = local_version(), str(ready.get("version") or "")
    if lv and rv and lv != rv:
        say(f"Note: MACE here is {lv}, on {where} {rv}. That is fine: the page comes from {where}.")
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
        if args.export_script or args.dry_run or not (sys.stdin and sys.stdin.isatty()):
            print("mace gui connect: no profile yet. Pass --gateway <login host> (and usually --node <dev node> "
                  "and --dir <folder>), or run it in a terminal to answer a few questions.", file=sys.stderr)
            return 2
        got = first_run_prompts(input)
        new_name = got.pop("_name")
        prof.update(got)
        if args.save is None:
            args.save = new_name
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
    if prof.get("gateway") and gateway_looks_like_email(str(prof["gateway"])):
        print(email_gateway_warning(str(prof["gateway"]), path, args.save or name), file=sys.stderr)
    if args.dry_run:
        node, gateway = prof.get("node"), prof.get("gateway")
        hop = str(prof.get("hop") or "auto")
        lp = int(prof.get("local_port") or DEFAULT_LOCAL_PORT)
        unode = effective_node_user(str(gateway)) if (node and gateway and "@" not in str(node)) else None
        for h in (["jump", "nested"] if hop == "auto" else [hop]) if (node and gateway) else ["jump"]:
            argv = build_ssh_argv(prof, lp, 40000, 40001, h, unode)
            label = f"hop {h}" if node and gateway else "no hop"
            print(f"# {label} (the remote port is random on a real run)")
            print(shlex.join(argv))
        print("# remote command")
        print(remote_command(prof, 40000))
        return 0
    return run(prof, args, ask=input)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
