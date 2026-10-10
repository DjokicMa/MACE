"""``mace gui --slurm``: MACE GUI as a batch job on a compute node.

    mace gui --slurm [--dir D] [--time 4:00:00] [--cpus 2] [--mem 4G]
                     [--partition P] [--account A] [--qos Q] [--constraint C]
                     [--gpus G] [--sbatch-arg ARG ...] [--dry-run]

Run on a node that can submit jobs (a dev or login node). It submits one
small batch job (``sbatch``, named ``mace-gui``) that runs
``mace gui --managed --http-control`` in the folder, waits while the job is
pending, then forwards a port of this node to the server on the job's node
(``ssh -N -L`` to the node, which needs ssh to the node of your own running
job; most clusters allow it through pam_slurm_adopt). When the server stops
(Ctrl+C here, Sign Out and Stop Server, the idle stop, the job's time limit)
the job ends and its allocation is released.

``mace gui connect --slurm`` runs this through ssh from your computer
(``--managed``: machine-readable lines on stdout, control lines on stdin,
as ``mace gui --managed``):

* ``MACE_GUI_JOB {"job"}``                 submitted
* ``MACE_GUI_PENDING {"job", "reason", "start"}``  while the job waits
* ``MACE_GUI_READY {... "job", "node", "ends"}``   the server answers
* ``MACE_GUI_ERROR {"kind", "message"}``   refused, never started, no ssh;
  kind ``port-busy`` (exit 98): the forwarding port is taken here, connect
  tries another (a job already running is kept and reattached to)
* ``MACE_GUI_STOPPED {"reason"}``          the server stopped

A job this run submitted is canceled when you give up (Ctrl+C, the
connection closes) before its server answers, pending or already started;
once the server answers it is never canceled from here: its server stops,
and the job with it. A job reattached to is never canceled. Every SLURM read goes through the shared, rate-limited query layer
(mace.cluster.reader: ``ownjobs``, ``job:<id>``).

Standard library only (dispatched before the server is imported).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

JOB_NAME = "mace-gui"
DEFAULTS = {"time": "4:00:00", "cpus": 2, "mem": "4G"}
PENDING_POLL_S = 30.0
READY_WAIT_S = 120.0
TUNNEL_WAIT_S = 20.0
TUNNEL_GRACE_S = 3.0             # SIGTERM to the tunnel's group, then SIGKILL after this
PING_S = 60.0
TIME_RE = re.compile(r"^(\d{1,3}-)?\d{1,4}(:\d{2}){0,2}$")
MEM_RE = re.compile(r"^[0-9]{1,7}[KMGT]?$")
NAME_RE = re.compile(r"^[A-Za-z0-9._,-]{1,64}$")
CONSTRAINT_RE = re.compile(r"^[A-Za-z0-9._&|\[\]*:,!()-]{1,200}$")
GPUS_RE = re.compile(r"^[A-Za-z0-9._:,-]{1,64}$")
ARG_RE = re.compile(r"^--?[A-Za-z][A-Za-z0-9-]{0,40}(=[^\s]{0,200})?$")
# options MACE sets itself, or that would take the job away from the server
REFUSED_ARGS = {"-J", "--job-name", "-o", "--output", "-e", "--error", "--wrap", "-D", "--chdir", "--parsable",
                "-a", "--array", "-W", "--wait", "-H", "--hold", "--test-only", "-i", "--input"}
SSH_REFUSED = ("This cluster does not allow ssh to the node of your job, which --slurm needs. Run MACE GUI on a "
               "dev node instead, or ask your administrators whether ssh to your job's node is allowed "
               "(pam_slurm_adopt).")


class SlurmError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ------------------------------------------------------------------ options

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mace gui --slurm",
        description="Run MACE GUI as a small batch job on a compute node and forward a port of this node to it. "
                    "The job ends, and its allocation is released, when the server stops.",
        epilog="Defaults: --time 4:00:00, --cpus 2, --mem 4G (the server is mostly idle; raise them for ML "
               "Clean or large files). Partition, account, QOS and constraint come from these flags, else from "
               "your site settings ([slurm] in ~/.config/mace/site.toml, mace site), else SLURM's defaults. "
               "From your computer: mace gui connect --slurm. See docs/gui.md#run-it-as-a-slurm-job.")
    p.add_argument("--slurm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dir", "-d", default=".", help="folder to serve (default: .)")
    p.add_argument("--port", "-p", type=int, default=None,
                   help="port on this node that forwards to the job (default: a free one)")
    p.add_argument("--time", help=f"the job's time limit (default {DEFAULTS['time']}; D-HH:MM:SS, H:MM:SS, MM)")
    p.add_argument("--cpus", type=int, help=f"CPUs for the server (default {DEFAULTS['cpus']})")
    p.add_argument("--mem", help=f"memory (default {DEFAULTS['mem']}; e.g. 8G)")
    p.add_argument("--partition", help="partition (default: site settings, else SLURM's default)")
    p.add_argument("--account", help="account (default: site settings, else your SLURM default account)")
    p.add_argument("--qos", help="QOS (default: site settings, else none)")
    p.add_argument("--constraint", help="node features (default: site settings, else none)")
    p.add_argument("--gpus", help="GPUs for ML Clean, e.g. 1 or a100:1 (default: none)")
    p.add_argument("--sbatch-arg", action="append", default=None, metavar="ARG",
                   help="one more sbatch option, as --name=value (repeatable), e.g. --sbatch-arg=--reservation=r1")
    p.add_argument("--setup", action="append", default=None, metavar="LINE",
                   help="a shell line the job runs first (repeatable), e.g. 'module load Python'")
    p.add_argument("--root", action="append", default=None, metavar="DIR",
                   help="a folder File > Open Folder may switch within (passed to the server)")
    p.add_argument("--gateway", default=os.environ.get("MACE_GUI_GATEWAY", "").strip() or None,
                   help="your cluster's login host, for the tunnel hint (default MACE_GUI_GATEWAY)")
    p.add_argument("--managed", action="store_true", help="used by mace gui connect --slurm")
    p.add_argument("--dry-run", action="store_true", help="print the job script and the sbatch command; submit "
                                                          "nothing")
    p.add_argument("--no-banner", action="store_true", help=argparse.SUPPRESS)
    return p


def _site_values(env: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """{key: (value, source)} for the [slurm] values a user or group set
    (never a built-in value: those are one cluster's)."""
    out: Dict[str, Tuple[str, str]] = {}
    try:
        from mace.submission import site
        s = site.load(env)
    except Exception:
        return out
    for key, attr in (("partition", "partition"), ("account", "account"), ("qos", "qos"),
                      ("constraint", "constraint")):
        src = (getattr(s, "sources", {}) or {}).get(attr, "built-in")
        val = getattr(s, attr, None)
        if val and src and src != "built-in":
            out[key] = (str(val), str(src))
    return out


def resolve_options(args, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The job's request: flags, then site settings, then the defaults; every
    value checked (each becomes one sbatch argv element)."""
    env = dict(os.environ if env is None else env)
    site_vals = _site_values(env)
    o: Dict[str, Any] = {"sources": {}}
    for k in ("time", "cpus", "mem"):
        v = getattr(args, k, None)
        o[k] = DEFAULTS[k] if v in (None, "") else v
        o["sources"][k] = "flag" if v not in (None, "") else "default"
    for k in ("partition", "account", "qos", "constraint"):
        v = getattr(args, k, None)
        if v not in (None, ""):
            o[k], o["sources"][k] = str(v), "flag"
        elif k in site_vals:
            o[k], o["sources"][k] = site_vals[k][0], f"site settings ({site_vals[k][1]})"
        else:
            o[k] = None
    o["gpus"] = getattr(args, "gpus", None) or None
    o["extra"] = list(getattr(args, "sbatch_arg", None) or [])
    o["setup"] = [s for s in (getattr(args, "setup", None) or []) if str(s).strip()]
    if not TIME_RE.match(str(o["time"])):
        raise SlurmError("args", f"--time {o['time']!r}: use D-HH:MM:SS, H:MM:SS or minutes")
    try:
        o["cpus"] = int(o["cpus"])
    except (TypeError, ValueError):
        raise SlurmError("args", f"--cpus {o['cpus']!r}: a number")
    if not 1 <= o["cpus"] <= 256:
        raise SlurmError("args", "--cpus is 1 to 256")
    if not MEM_RE.match(str(o["mem"])):
        raise SlurmError("args", f"--mem {o['mem']!r}: a size such as 4G or 8000M")
    for k in ("partition", "account", "qos"):
        if o[k] is not None and not NAME_RE.match(o[k]):
            raise SlurmError("args", f"--{k} {o[k]!r}: letters, digits and . _ , - only")
    if o["constraint"] is not None and not CONSTRAINT_RE.match(o["constraint"]):
        raise SlurmError("args", f"--constraint {o['constraint']!r} has characters SLURM features do not use")
    if o["gpus"] is not None and not GPUS_RE.match(str(o["gpus"])):
        raise SlurmError("args", f"--gpus {o['gpus']!r}: e.g. 1 or a100:1")
    for a in o["extra"]:
        name = a.split("=", 1)[0]
        if not ARG_RE.match(a) or "\n" in a:
            raise SlurmError("args", f"--sbatch-arg {a!r}: one option as --name=value")
        if name in REFUSED_ARGS:
            raise SlurmError("args", f"--sbatch-arg {name}: MACE sets it itself (or it would take the job away "
                                     "from the server)")
    return o


# ------------------------------------------------------------------ the job

def base_dir() -> Path:
    from .managed import state_dir
    return state_dir() / "slurm"


def _private_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    return p


def mace_command() -> List[str]:
    """This MACE: the Python that runs it and its mace_cli (the job gets the
    environment sbatch exports, so a module or conda Python keeps working)."""
    cli = Path(__file__).resolve().parents[2] / "mace_cli"
    return [sys.executable or "python3", str(cli)]


def job_script(opts: Dict[str, Any], root: Path, base: Path, roots: Optional[Sequence[str]] = None) -> str:
    """The batch script: only the run part; the request is sbatch's argv.
    Paths and values are quoted; no token is in it."""
    lines = ["#!/bin/bash",
             "# MACE GUI as a batch job (mace gui --slurm). The job ends when the server stops:",
             "# Ctrl+C in mace gui --slurm or mace gui connect, Sign Out and Stop Server, the",
             "# idle stop, or 60 s before this job's time limit."]
    for s in opts.get("setup") or []:
        lines.append(str(s).strip())
    lines.append(f"cd -- {shlex.quote(str(root))} || exit 2")
    lines.append('export MACE_GUI_SLURM="$SLURM_JOB_ID"')
    cmd = [shlex.quote(w) for w in mace_command()] + ["--no-banner", "gui", "--managed", "--http-control",
                                                       "--port", "0", "--dir", "."]
    for r in roots or []:
        cmd += ["--root", shlex.quote(str(r))]
    cmd += ["--ready-file", shlex.quote(str(base)) + '/"$SLURM_JOB_ID"/ready.json']
    lines.append("exec " + " ".join(cmd))
    return "\n".join(lines) + "\n"


def sbatch_argv(opts: Dict[str, Any], base: Path, script: Path) -> List[str]:
    argv = ["sbatch", "--parsable", f"--job-name={JOB_NAME}", f"--time={opts['time']}", "--ntasks=1",
            f"--cpus-per-task={opts['cpus']}", f"--mem={opts['mem']}",
            f"--output={base}/{JOB_NAME}-%j.log",
            # a dev node's per-process CPU-time limit is never carried into the job
            "--propagate=NONE"]
    for k, flag in (("partition", "--partition"), ("account", "--account"), ("qos", "--qos"),
                    ("constraint", "--constraint"), ("gpus", "--gpus")):
        if opts.get(k):
            argv.append(f"{flag}={opts[k]}")
    argv += list(opts.get("extra") or [])
    argv.append(str(script))
    return argv


def parse_parsable(text: str) -> Optional[str]:
    """``sbatch --parsable``: ``123`` or ``123;cluster``."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    jid = first.split(";", 1)[0].strip()
    return jid if re.fullmatch(r"\d+", jid) else None


def first_node(nodelist: str) -> Optional[str]:
    """The first host of a SLURM node list (``a[01-03,07]`` -> ``a01``)."""
    s = (nodelist or "").strip()
    if not s or s in ("(null)", "None", "N/A"):
        return None
    m = re.match(r"^([^\[,]+)\[([^\]]+)\]", s)
    if m:
        first = m.group(2).split(",")[0].split("-")[0]
        return m.group(1) + first
    return s.split(",")[0]


def scancel(job: str, env: Optional[Dict[str, str]] = None) -> bool:
    """Cancel this pending job (only the id sbatch gave us)."""
    if not re.fullmatch(r"\d+", str(job)):
        return False
    try:
        from mace.queue.slurm_ids import outside_allocation_env, slurm_env
        e = slurm_env(outside_allocation_env(dict(os.environ if env is None else env)))
    except Exception:
        e = dict(os.environ if env is None else env)
    try:
        return subprocess.run(["scancel", str(job)], env=e, stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def job_state(job: str, refresh: bool = True) -> Optional[Dict[str, str]]:
    """``scontrol show job -o <id>`` through the shared cache (key job:<id>)."""
    from mace.cluster import reader
    try:
        r = reader.job_detail(str(job), refresh=refresh, tag="gui-slurm")
    except ValueError:
        return None
    return r.value if isinstance(r.value, dict) else None


def own_gui_jobs() -> List[Dict[str, Any]]:
    """Your running ``mace-gui`` jobs (key ownjobs, shared with the Jobs panel)."""
    from mace.cluster import reader
    r = reader.squeue_own(max_age=15, refresh=True, tag="gui-slurm")
    rows = r.value if isinstance(r.value, list) else []
    return [x for x in rows if x.get("name") == JOB_NAME and x.get("state") == "RUNNING"]


# ------------------------------------------------------------------ http

def _api(port: int, prefix: str, token: str, route: str, body: Optional[Dict[str, Any]] = None,
         timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    from .managed import _api as api
    return api(port, prefix, token, route, body, timeout)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_free(port: int) -> bool:
    """127.0.0.1:port can be listened on here (the forward's end)."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


# ssh could not listen on the forward's port here: a busy port, not a refusal by the node
BIND_FAILED = ("Could not request local forwarding", "Address already in use", "cannot listen to port",
               "bind [127.0.0.1]")


def ssh_command() -> List[str]:
    v = os.environ.get("MACE_GUI_SLURM_SSH", "").strip()
    return shlex.split(v) if v else ["ssh"]


def _death_signal() -> Optional[Callable[[], None]]:
    """preexec_fn for the tunnel on Linux: SIGTERM when the thread that
    started it ends (PR_SET_PDEATHSIG), so even a session killed with
    SIGKILL leaves no ssh behind. libc's prctl is looked up here, before the
    fork; the child only calls it. Only from the main thread (the signal
    follows the starting thread, not the process); None elsewhere."""
    if not sys.platform.startswith("linux") or threading.current_thread() is not threading.main_thread():
        return None
    try:
        import ctypes
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError, ImportError):
        return None
    parent = os.getpid()

    def preexec():
        prctl(1, int(signal.SIGTERM), 0, 0, 0)          # PR_SET_PDEATHSIG
        if os.getppid() != parent:                      # the session died before that call
            os._exit(1)
    return preexec


_TUNNELS: "set[Tunnel]" = set()      # open tunnels: closed at exit whatever way it comes


def close_tunnels() -> None:
    for t in list(_TUNNELS):
        t.close()


atexit.register(close_tunnels)


class Tunnel:
    """``ssh -N -L 127.0.0.1:<local>:127.0.0.1:<remote> <node>`` (non-interactive),
    in a process group of its own: close() ends the whole group (an ssh
    wrapper's children too, never anything this run did not start)."""

    def __init__(self, node: str, local: int, remote: int):
        self.node, self.local, self.remote = node, local, remote
        self.argv = ssh_command() + ["-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                                     "-o", "ServerAliveInterval=30", "-L",
                                     f"127.0.0.1:{local}:127.0.0.1:{remote}", node]
        self._group = False            # its process group may still hold processes to end
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_death_signal())
        self._group = hasattr(os, "killpg")
        _TUNNELS.add(self)

    def _signal_group(self, sig: int) -> None:
        if not self._group:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            self._group = False
        except OSError:
            pass

    def alive(self) -> bool:
        if self.proc.poll() is None:
            return True
        # ssh ended: what is left of its group goes now, while the group id
        # cannot belong to anything else yet
        self._signal_group(signal.SIGKILL)
        self._group = False
        return False

    def stderr(self) -> str:
        if self.alive():
            return ""
        try:
            return (self.proc.stderr.read() or b"").decode(errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        """SIGTERM to the tunnel's group, SIGKILL after TUNNEL_GRACE_S."""
        if self.proc.poll() is None:
            if self._group:
                self._signal_group(signal.SIGTERM)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(TUNNEL_GRACE_S)
            except subprocess.TimeoutExpired:
                if not self._group:
                    self.proc.kill()
            except KeyboardInterrupt:          # a second Ctrl+C: no more grace
                pass
        if self._group:
            self._signal_group(signal.SIGKILL)
            self._group = False
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        if self.proc.stderr is not None:
            try:
                self.proc.stderr.close()
            except OSError:
                pass
        _TUNNELS.discard(self)             # last: one cut short is closed again at exit


def open_tunnel(node: str, local: int, ready: Dict[str, Any], wait_s: float = TUNNEL_WAIT_S) -> Tunnel:
    """The inner hop; it answers api/info with the server's token, or SlurmError."""
    t = Tunnel(node, local, int(ready["port"]))
    try:
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if not t.alive():
                err = t.stderr()
                if any(b in err for b in BIND_FAILED):
                    raise SlurmError("port-busy", f"Port {local} is busy on this node (ssh could not listen on it).")
                raise SlurmError("ssh-node", SSH_REFUSED + (f"\n(ssh: {err.splitlines()[-1]})" if err else ""))
            if _api(local, ready["prefix"], ready["token"], "info", timeout=3) is not None:
                return t
            time.sleep(0.3)
        raise SlurmError("ssh-node", SSH_REFUSED)
    except BaseException:                  # Ctrl+C or an error while it came up: no ssh left behind
        t.close()
        raise


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def log_tail(base: Path, job: str, n: int = 12) -> str:
    try:
        lines = (base / f"{JOB_NAME}-{job}.log").read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def cleanup_old(base: Path, keep: Sequence[str] = (), age_s: float = 7 * 86400) -> None:
    """Remove the folders and logs of GUI jobs that ended more than a week ago."""
    now = time.time()
    try:
        entries = list(base.iterdir())
    except OSError:
        return
    for p in entries:
        jid = p.name.split("-")[-1].split(".")[0] if p.is_file() else p.name
        if jid in keep:
            continue
        try:
            if now - p.stat().st_mtime < age_s:
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------ the session

class Session:
    """One ``mace gui --slurm`` run: submit or reattach, wait, tunnel, relay."""

    def __init__(self, args, out: Callable[[str, Any], None], say: Callable[[str], None],
                 lines: "queue.Queue[Optional[str]]", ask: Optional[Callable[[str], str]] = None):
        self.args = args
        self.out = out                 # a protocol event (managed) or nothing
        self.say = say                 # a human line
        self.lines = lines             # control lines (None = EOF / Ctrl+C)
        self.ask = ask
        self.root = Path(args.dir).expanduser().resolve()
        self.base = _private_dir(base_dir())
        self.wid = f"s{os.getpid()}-{secrets.token_hex(3)}"
        self.tunnel: Optional[Tunnel] = None
        self.job: Optional[str] = None
        self.submitted = False         # this run submitted self.job (not a reattach)
        self.ready: Optional[Dict[str, Any]] = None
        self.node: Optional[str] = None
        self.port = args.port if args.port else free_port()

    # -- steps ---------------------------------------------------------------
    def reattach_candidate(self) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """(job, ready, node) of a running GUI job of yours for this folder; a job
        for another folder asks (a terminal) or is left alone."""
        try:
            jobs = own_gui_jobs()
        except Exception:
            return None
        others = []
        for j in jobs:
            ready = read_json(self.base / str(j.get("id")) / "ready.json")
            node = first_node(j.get("nodelist") or "")
            if not ready or not node:
                continue
            if os.path.realpath(str(ready.get("root", ""))) == str(self.root):
                return str(j["id"]), ready, node
            others.append((str(j["id"]), ready, node))
        if others and self.ask is not None and not self.args.managed and sys.stdin and sys.stdin.isatty():
            jid, ready, node = others[0]
            a = (self.ask(f"MACE GUI already runs as job {jid} on {node} for {ready.get('root')}. "
                          "[r]eattach to it, start [a]nother, or [c]ancel? ") or "a").strip().lower()
            if a.startswith("r"):
                return jid, ready, node
            if a.startswith("c"):
                raise SlurmError("canceled", "Canceled.")
        return None

    def submit(self, opts: Dict[str, Any]) -> str:
        script = self.base / f"job-{secrets.token_hex(6)}.sh"
        fd = os.open(str(script), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
        with os.fdopen(fd, "w") as fh:
            fh.write(job_script(opts, self.root, self.base, self.args.root))
        argv = sbatch_argv(opts, self.base, script)
        try:
            from mace.queue.slurm_ids import outside_allocation_env, slurm_env
            env = slurm_env(outside_allocation_env(dict(os.environ)))
        except Exception:
            env = dict(os.environ)
        try:
            r = subprocess.run(argv, env=env, cwd=str(self.root), stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=120)
        except FileNotFoundError:
            raise SlurmError("no-slurm", "sbatch is not on PATH here: run mace gui --slurm on a node that can "
                                         "submit jobs (a login or dev node of your cluster).")
        except subprocess.SubprocessError as exc:
            raise SlurmError("sbatch", f"sbatch did not answer: {exc}")
        finally:
            try:
                script.unlink()          # SLURM keeps its own copy of a submitted script
            except OSError:
                pass
        jid = parse_parsable(r.stdout) if r.returncode == 0 else None
        if not jid:
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            raise SlurmError("refused", "SLURM refused the MACE GUI job: " + (msg[-1] if msg else f"exit {r.returncode}")
                             + "\nCheck --partition/--account/--time (or [slurm] in mace site).")
        return jid

    def wait_running(self, job: str) -> Tuple[str, Dict[str, str]]:
        """Poll the job (every 30 s while pending) until it runs; Ctrl+C or EOF cancels it."""
        last = None
        while True:
            info = job_state(job) or {}
            st = (info.get("JobState") or "").upper()
            if st == "RUNNING":
                node = first_node(info.get("NodeList") or "")
                if node:
                    return node, info
            elif st in ("", "PENDING", "CONFIGURING", "REQUEUED", "RESIZING"):
                if not info:
                    raise SlurmError("gone", f"The job {job} is no longer in the queue (refused or canceled).")
                reason, start = info.get("Reason") or "None", info.get("StartTime") or "Unknown"
                if (reason, start) != last:
                    last = (reason, start)
                    self.out("MACE_GUI_PENDING", {"job": job, "reason": reason, "start": start})
                    self.say(f"Waiting for a compute node (job {job}, pending: {reason}, expected start {start}). "
                             "Press Ctrl+C to give up.")
            else:
                raise SlurmError("ended", f"The job {job} ended before MACE GUI started ({st.lower()}"
                                          f"{', ' + info['Reason'] if info.get('Reason') not in (None, 'None') else ''})."
                                          + _tail_note(log_tail(self.base, job)))
            # SLURM is asked every 30 s; the job's ready file (on the shared home)
            # is looked at every second, so a job that started is seen at once
            ready_file = self.base / job / "ready.json"
            if self._wait_line(PENDING_POLL_S if st != "RUNNING" else 2.0, until=ready_file.exists):
                scancel(job)
                raise SlurmError("canceled", f"Canceled the pending job {job}.")

    def _wait_line(self, seconds: float, until: Optional[Callable[[], bool]] = None) -> bool:
        """True when the user gave up (EOF, Ctrl+C, a stop line) within
        ``seconds``; False at the end, or as soon as ``until()`` is true."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0 or (until is not None and until()):
                return False
            try:
                line = self.lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None or line.strip().lower() in ("stop", "detach"):
                return True

    def wait_ready(self, job: str) -> Dict[str, Any]:
        path = self.base / job / "ready.json"
        end = time.monotonic() + READY_WAIT_S
        while time.monotonic() < end:
            ready = read_json(path)
            if ready and ready.get("token") and ready.get("port"):
                return ready
            info = job_state(job, refresh=False) or {}
            if info and (info.get("JobState") or "").upper() not in ("RUNNING", "COMPLETING"):
                raise SlurmError("ended", f"The job {job} ended before MACE GUI was ready."
                                          + _tail_note(log_tail(self.base, job)))
            if self._wait_line(1.0):
                break
        scancel(job)
        raise SlurmError("not-ready", f"MACE GUI did not start in job {job} within {int(READY_WAIT_S)} s; the job "
                                      "was canceled." + _tail_note(log_tail(self.base, job)))

    def start(self) -> Dict[str, Any]:
        """Reattach or submit; then the tunnel. Returns the READY payload."""
        opts = resolve_options(self.args)
        found = self.reattach_candidate()
        reattached = False
        if found:
            job, ready, node = found
            reattached = True
        else:
            try:
                keep = [str(j.get("id")) for j in own_gui_jobs()]
            except Exception:
                keep = []
            cleanup_old(self.base, keep=keep)         # never a running job's folder (a long --time)
            job = self.submit(opts)
            self.job, self.submitted = job, True
            self.out("MACE_GUI_JOB", {"job": job})
            self.say(f"Submitted job {job} ({opts['cpus']} CPUs, {opts['mem']}, {opts['time']}"
                     f"{', partition ' + opts['partition'] if opts.get('partition') else ''}).")
            node, _ = self.wait_running(job)
            ready = self.wait_ready(job)
        self.job, self.node, self.ready = job, node, ready
        for attempt in range(3):
            try:
                self.tunnel = open_tunnel(node, self.port, ready)
                break
            except SlurmError as exc:
                if exc.kind == "port-busy" and not self.args.port and attempt < 2:
                    self.port = free_port()           # a port of our own choosing: take another one
                    continue
                if exc.kind == "port-busy":
                    # a port given to us (mace gui connect picks another and reattaches): the job is kept
                    raise SlurmError("port-busy", f"{exc} The job {job} keeps running; MACE GUI reattaches to it "
                                                  f"on another port.")
                if not reattached:
                    scancel(job)        # the server never became usable: give the allocation back
                raise
        got = _api(self.port, ready["prefix"], ready["token"], "managed/attach", {"watcher": self.wid}) or {}
        login = None
        lc = _api(self.port, ready["prefix"], ready["token"], "login-code", {})
        if lc and lc.get("code"):
            login = f"http://127.0.0.1:{self.port}{ready['prefix']}?login={lc['code']}"
        info = job_state(job, refresh=False) or {}
        try:
            from mace import __version__ as version
        except Exception:
            version = ""
        return {"url": f"http://127.0.0.1:{self.port}{ready['prefix']}", "port": self.port,
                "prefix": ready["prefix"], "token": ready["token"], "login": login, "host": node, "fqdn": node,
                "root": ready.get("root"), "version": ready.get("version") or version, "reattached": reattached,
                "tasks": int(got.get("tasks") or 0), "job": job, "node": node, "ends": info.get("EndTime")}

    def stopped_reason(self) -> Optional[str]:
        if not self.job:
            return None
        d = read_json(self.base / self.job / "stopped.json")
        return str(d.get("reason")) if d and d.get("reason") else None

    def relay(self) -> int:
        """Control lines -> the server's managed/* routes; ping; detach on EOF."""
        ready = self.ready or {}
        port, prefix, token = self.port, ready.get("prefix", "/"), ready.get("token", "")
        last_ping = last_look = time.monotonic()
        while True:
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                line = ""
            now = time.monotonic()
            if now - last_look >= 2.0:
                last_look = now
                if self.stopped_reason() is not None:      # it stopped on its own (idle, time limit, browser)
                    return self._finish(wait_s=0)
            if line is None:                                   # EOF: the user left
                _api(port, prefix, token, "managed/detach", {"watcher": self.wid})
                return self._finish(wait_s=0)
            cmd = line.strip().lower()
            if cmd == "stop":
                _api(port, prefix, token, "managed/stop", {"watcher": self.wid})
                return self._finish(wait_s=30)
            if cmd == "detach":
                _api(port, prefix, token, "managed/detach", {"watcher": self.wid})
                return self._finish(wait_s=0)
            if cmd == "login":
                lc = _api(port, prefix, token, "login-code", {})
                if lc and lc.get("code"):
                    self.out("MACE_GUI_LOGIN", {"url": f"http://127.0.0.1:{port}{prefix}?login={lc['code']}"})
            if cmd == "ping" or now - last_ping >= PING_S:
                last_ping = now
                if _api(port, prefix, token, "managed/ping", {"watcher": self.wid}) is None:
                    if self._server_gone():
                        return self._finish(wait_s=0)
            elif self.tunnel is not None and not self.tunnel.alive():
                if self._server_gone():
                    return self._finish(wait_s=0)
                self.tunnel.close()
                try:                                           # one new hop (a dropped ssh)
                    self.tunnel = open_tunnel(self.node or "", self.port, ready, wait_s=10)
                except SlurmError:
                    self.out("MACE_GUI_STOPPED", {"reason": "tunnel-lost"})
                    self.say(f"Lost the forward to {self.node}; the server keeps running in job {self.job}. Run "
                             "the same command again to reattach.")
                    return 1

    def _server_gone(self) -> bool:
        info = job_state(self.job or "", refresh=True) if self.job else None
        st = ((info or {}).get("JobState") or "").upper()
        return not info or st not in ("RUNNING",) or self.stopped_reason() is not None

    def _finish(self, wait_s: float) -> int:
        """Wait (up to wait_s) for the job to end, report why it stopped."""
        end = time.monotonic() + wait_s
        reason = self.stopped_reason()
        while reason is None and time.monotonic() < end:
            time.sleep(0.5)
            reason = self.stopped_reason()
        if self.tunnel is not None:
            self.tunnel.close()
        if reason:
            self.out("MACE_GUI_STOPPED", {"reason": reason, "job": self.job})
            self.say(f"MACE GUI stopped ({reason}); its job {self.job} ends and the allocation is released.")
        return 0

    def close(self) -> None:
        if self.tunnel is not None:
            self.tunnel.close()


def _tail_note(tail: str) -> str:
    return f"\nThe job's log ends with:\n{tail}" if tail.strip() else ""


# ------------------------------------------------------------------ entry

def _emit(tag: str, data: Any) -> None:
    from .managed import emit
    emit(tag, data)


class _Terminated(KeyboardInterrupt):
    """SIGTERM or SIGHUP (the connection that started this run closed): as
    Ctrl+C before the server answers; after it, this run leaves (the job
    keeps running, as on EOF), always closing its tunnel."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _on_termination() -> Dict[int, Any]:
    """Termination signals raise _Terminated (once) in the main thread; returns
    the handlers to put back."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    fired = []

    def handler(signum, frame):
        if not fired:                      # a second one must not cut the cleanup short
            fired.append(signum)
            raise _Terminated(signum)
    old = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return old


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        msg = f"{root} is not a directory"
        if args.managed:
            _emit("MACE_GUI_ERROR", {"kind": "dir", "message": msg})
        print(f"mace gui --slurm: {msg}", file=sys.stderr)
        return 2
    try:
        opts = resolve_options(args)
    except SlurmError as exc:
        if args.managed:
            _emit("MACE_GUI_ERROR", {"kind": exc.kind, "message": str(exc)})
        print(f"mace gui --slurm: {exc}", file=sys.stderr)
        return 2
    if args.port and not args.dry_run and not port_free(args.port):
        # before anything is submitted: connect tries another port
        msg = f"Port {args.port} is busy on this node."
        if args.managed:
            _emit("MACE_GUI_ERROR", {"kind": "port-busy", "message": msg})
        print(f"mace gui --slurm: {msg}", file=sys.stderr)
        return 98
    if args.dry_run:
        base = base_dir()
        script = base / "job-XXXXXXXXXXXX.sh"
        print("# sbatch command")
        print(shlex.join(sbatch_argv(opts, base, script)))
        print(f"# job script ({script}, removed after sbatch)")
        sys.stdout.write(job_script(opts, root.resolve(), base, args.root))
        for k, src in sorted(opts["sources"].items()):
            if src != "default":
                print(f"# {k}: {src}")
        return 0
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    if args.managed:
        out = _emit
        say = lambda text: None                                    # noqa: E731 - connect prints the events

        def reader():
            try:
                for raw in sys.stdin:
                    lines.put(raw)
            except (OSError, ValueError):
                pass
            lines.put(None)
        threading.Thread(target=reader, name="mace-slurm-stdin", daemon=True).start()
    else:
        out = lambda tag, data: None                               # noqa: E731
        say = lambda text: print(text, flush=True)                 # noqa: E731
    sess = Session(args, out, say, lines, ask=input)
    handlers = _on_termination()
    try:
        try:
            ready = sess.start()
        except KeyboardInterrupt as exc:
            # before the server could be reached: the job this run submitted
            # (pending, or started with no one able to use it) is canceled; a
            # job reattached to is left running
            if sess.job and sess.submitted:
                info = job_state(sess.job) or {}
                st = (info.get("JobState") or "PENDING").upper()
                if st in ("PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "RUNNING"):
                    scancel(sess.job)
                    say(f"Canceled job {sess.job}" + (" (pending)." if st == "PENDING" else
                                                     " before MACE GUI was ready."))
            return 128 + exc.signum if isinstance(exc, _Terminated) else 130
        except SlurmError as exc:
            out("MACE_GUI_ERROR", {"kind": exc.kind, "message": str(exc), "job": sess.job})
            print(f"mace gui --slurm: {exc}", file=sys.stderr)
            return {"canceled": 130, "port-busy": 98}.get(exc.kind, 1)
        out("MACE_GUI_READY", ready)
        if not args.managed:
            _banner(ready, args)
        try:
            return sess.relay()
        except _Terminated as exc:
            r = ready
            _api(r["port"], r["prefix"], r["token"], "managed/detach", {"watcher": sess.wid}, timeout=2)
            return 128 + exc.signum
        except KeyboardInterrupt:
            r = ready
            _api(r["port"], r["prefix"], r["token"], "managed/stop", {"watcher": sess.wid})
            print("\nStopping MACE GUI; its job ends and the allocation is released.", flush=True)
            return sess._finish(wait_s=30)
    finally:
        sess.close()
        for sig, h in handlers.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError, TypeError):
                pass


def _banner(ready: Dict[str, Any], args) -> None:
    from .cli import ssh_hint
    lines = [f"MACE GUI runs as job {ready['job']} on {ready['node']}"
             + (f" until {ready['ends']}" if ready.get("ends") else "") + f", serving {ready.get('root')}",
             f"  {ready.get('login') or ready['url']}",
             "  (the link signs in once)",
             f"Token (for the sign-in form): {ready['token']}" if sys.stdout.isatty() else
             "(not a terminal: the token is not printed)"]
    lines += [""] + ssh_hint(int(ready["port"]), args.gateway, str(ready.get("root") or ""))
    lines += ["Press Ctrl+C to stop MACE GUI; its job ends and the allocation is released."]
    print("\n".join(lines), flush=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
