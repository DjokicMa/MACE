"""The MACE side of the ML worker protocol (see ``mlip_worker``).

``WorkerProcess`` starts the worker in the ML interpreter and owns it:
``[python, -I, mlip_worker.py, ...]`` in a fresh private directory, its own
session, the child environment of ``proc`` (no PYTHON*, no inherited MACE_*,
thread counts and CUDA_VISIBLE_DEVICES forced, offline). It waits for the
worker's ``hello`` (up to 180 s, polling the pure ``canceled()`` every
50 ms) and checks the protocol, the model id and the elements.

Two carriers with the same messages: the worker's stdin/stdout (``pipe``,
the CLI: the worker is a child of ``mace preopt`` and dies with it) and a
Unix socket in the worker's private 0700 directory (``socket``, the GUI: one
worker owned by the server, reached from the pool job). A call has 30 s +
0.05 s per atom; past that the worker's process group is killed. On a cancel
an in-flight call gets 5 s to finish (its reply is dropped and the loaded
model survives); only past that grace is the worker killed.

``MlipCalculator`` is an ase calculator that sends one ``calc`` request per
call and checks the answer: NaN/Inf energy, any force above 50 eV/Å or a
wrong shape raise ``MlipError``. Nothing ever falls back silently.
"""
from __future__ import annotations

import json
import math
import os
import select
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import proc

PROTOCOL = 1
HELLO_TIMEOUT = 180.0
CANCEL_GRACE = 5.0
MAX_FORCE = 50.0
MAX_LINE = 64 * 1024 * 1024
WORKER = Path(__file__).with_name("mlip_worker.py")


class MlipError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class MlipTimeout(MlipError):
    def __init__(self, message: str):
        super().__init__("timeout", message)


def call_timeout(n_atoms: int) -> float:
    return 30.0 + 0.05 * n_atoms


def device_mapping(device: str, environ: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
    """(CUDA_VISIBLE_DEVICES, the worker's --device, the provenance text).
    cpu hides every GPU; cuda:N shows only N, which torch then calls cuda:0.

    Inside a job that already restricts the GPUs (CUDA_VISIBLE_DEVICES set by
    SLURM, a site wrapper or a shared node's policy), cuda:N counts within
    that list, so the worker never sees a GPU the job was not given:
    "3,5" and cuda:1 -> CUDA_VISIBLE_DEVICES=5. An index outside the list, or
    a cuda device when the list is empty ("" or -1, no GPU), is refused."""
    device = str(device or "cpu")
    if device == "cpu":
        return "", "cpu", "cpu"
    if not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError(f"device {device!r}: use cpu or cuda:N")
    n = device[5:]
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" not in env:
        return n, "cuda:0", f"cuda:{n} (visible to the worker as cuda:0)"
    inherited = env["CUDA_VISIBLE_DEVICES"].strip()
    gpus = [g.strip() for g in inherited.split(",") if g.strip()]
    if not gpus or gpus[0] in ("-1", "none", "NoDevFiles"):
        raise ValueError(f"{device}: no GPU is visible to this job (CUDA_VISIBLE_DEVICES={inherited!r}); use cpu")
    k = int(n)
    if k >= len(gpus):
        raise ValueError(f"{device}: this job can see {len(gpus)} GPU(s) (CUDA_VISIBLE_DEVICES={inherited}); "
                         f"use cuda:0..cuda:{len(gpus) - 1}")
    phys = gpus[k]
    return phys, "cuda:0", f"cuda:{n} = physical {phys} of this job's CUDA_VISIBLE_DEVICES={inherited} (visible to the worker as cuda:0)"


def launch(python: str, model: Dict[str, Any], *, weights: Optional[str], threads: int, device: str = "cpu",
           dtype: Optional[str] = None, socket_path: Optional[str] = None, workdir: Path,
           nice: int = 0, environ: Optional[Dict[str, str]] = None) -> Tuple[List[str], Dict[str, str], str]:
    """The worker's argv and environment (one place: the CLI, the GUI and the tests).
    `environ`: the environment the device maps within (default os.environ)."""
    visible, wdev, text = device_mapping(device, environ)
    k = str(max(1, int(threads)))
    argv = [python, "-I", str(WORKER), "--model-id", model["id"], "--backend", model["backend"],
            "--weights", weights or "", "--device", wdev,
            "--dtype", dtype or (model.get("defaults") or {}).get("dtype", "float32"), "--threads", k]
    if socket_path:
        argv += ["--socket", socket_path]
    if nice > 0:
        import shutil
        n = shutil.which("nice")
        if n:
            argv = [n, "-n", str(int(nice))] + argv
    env = proc.child_env(OMP_NUM_THREADS=k, MKL_NUM_THREADS=k, OPENBLAS_NUM_THREADS=k,
                         CUDA_VISIBLE_DEVICES=visible, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                         TMPDIR=str(workdir))
    env["MACE_MLIP_NO_DOWNLOAD"] = "1"
    return argv, env, text


class _Lines:
    """Buffered newline-delimited reads from a file descriptor, with select()."""

    def __init__(self, fd: int):
        self.fd, self.buf = fd, b""

    def readline(self, deadline: float, canceled=None, poll: float = 0.05) -> Optional[bytes]:
        while b"\n" not in self.buf:
            if canceled is not None and canceled():
                raise proc.Canceled("canceled")
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            r, _, _ = select.select([self.fd], [], [], min(poll, left))
            if not r:
                continue
            chunk = os.read(self.fd, 1 << 20)
            if not chunk:
                return None                           # EOF
            self.buf += chunk
            if len(self.buf) > MAX_LINE:
                raise MlipError("protocol", "a reply line over 64 MB")
        line, _, self.buf = self.buf.partition(b"\n")
        return line


def _check_hello(hello: Dict[str, Any], model_id: str) -> None:
    if hello.get("op") != "hello":
        raise MlipError("protocol", f"the worker did not say hello: {str(hello)[:200]}")
    if hello.get("protocol") != PROTOCOL:
        raise MlipError("protocol", f"worker protocol {hello.get('protocol')!r}, this MACE speaks {PROTOCOL}")
    if hello.get("model") != model_id:
        raise MlipError("protocol", f"the worker loaded {hello.get('model')!r}, not {model_id!r}")


class WorkerProcess:
    """A worker owned by this process (pipe transport, or socket for the GUI host)."""

    def __init__(self, python: str, model: Dict[str, Any], *, weights: Optional[str] = None, threads: int = 1,
                 device: str = "cpu", dtype: Optional[str] = None, use_socket: bool = False, nice: int = 0,
                 canceled: Optional[Callable[[], bool]] = None, hello_timeout: float = HELLO_TIMEOUT):
        self.model = model
        self.workdir = proc.private_dir("mlip")
        self.socket_path = str(self.workdir / "w.sock") if use_socket else None
        self.argv, self.env, self.device_text = launch(python, model, weights=weights, threads=threads,
                                                       device=device, dtype=dtype, socket_path=self.socket_path,
                                                       workdir=self.workdir, nice=nice)
        self.log = self.workdir / "worker.log"
        self.p: Optional[subprocess.Popen] = None
        self.hello: Dict[str, Any] = {}
        self._lines: Optional[_Lines] = None
        self._id = 0
        self.dead = False
        self.started = 0
        self._start(canceled, hello_timeout)

    def _start(self, canceled, hello_timeout):
        errf = open(self.log, "wb")
        try:
            self.p = subprocess.Popen(self.argv, cwd=str(self.workdir), env=self.env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=errf, start_new_session=True)
        except OSError as exc:
            errf.close()
            self.close()
            raise MlipError("exception", f"the ML worker could not start: {exc}")
        errf.close()
        self.started = time.monotonic()
        self._lines = _Lines(self.p.stdout.fileno())
        deadline = time.monotonic() + hello_timeout
        try:
            line = self._lines.readline(deadline, canceled)
            for _ in range(200):          # anything a library printed before the hello is not the protocol
                if line is None or line.lstrip().startswith(b"{"):
                    break
                line = self._lines.readline(deadline, canceled)
        except proc.Canceled:
            self._abort()
            raise
        except TimeoutError:
            self._abort()
            raise MlipTimeout(f"the ML worker did not load its model within {hello_timeout:g} s"
                              + self._tail())
        except MlipError:
            self._abort()
            raise
        if line is None:
            self.p.wait(timeout=10)
            rc = self.p.returncode
            msg = (f"the ML worker exited (code {rc}) before it was ready" + self._tail())
            self.close()
            raise MlipError("crash", msg)
        try:
            hello = json.loads(line)
            _check_hello(hello, self.model["id"])
        except (ValueError, MlipError) as exc:
            self._abort()
            raise exc if isinstance(exc, MlipError) else MlipError("protocol", "the worker's hello is not JSON")
        self.hello = hello

    @property
    def pid(self) -> Optional[int]:
        return self.p.pid if self.p else None

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None and not self.dead

    def _tail(self, n: int = 20) -> str:
        try:
            lines = self.log.read_text(errors="replace").strip().splitlines()[-n:]
        except OSError:
            return ""
        return ("\n" + "\n".join(lines)) if lines else ""

    def stderr_tail(self, n: int = 20) -> str:
        return self._tail(n).strip()

    def _send(self, msg: Dict[str, Any]) -> None:
        try:
            self.p.stdin.write((json.dumps(msg) + "\n").encode())
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.dead = True
            raise MlipError("crash", "the ML worker is gone" + self._tail())

    def request(self, msg: Dict[str, Any], timeout: float, canceled=None) -> Dict[str, Any]:
        """One request on the pipe. Cancel: 5 s grace, then kill."""
        if not self.alive():
            raise MlipError("crash", "the ML worker is not running" + self._tail())
        self._id += 1
        msg = dict(msg, id=self._id) if msg.get("op") == "calc" else msg
        self._send(msg)
        deadline = time.monotonic() + timeout
        try:
            line = self._lines.readline(deadline, canceled)
        except proc.Canceled:
            try:                                              # in flight: let it finish, drop the answer
                line = self._lines.readline(time.monotonic() + CANCEL_GRACE)
                if line is None:
                    self.dead = True
            except (TimeoutError, MlipError):
                self.kill()
            raise
        except TimeoutError:
            self.kill()
            raise MlipTimeout(f"the ML worker gave no answer within {timeout:g} s")
        if line is None:
            self.p.wait(timeout=10)
            self.dead = True
            raise MlipError("crash", f"the ML worker exited (code {self.p.returncode}) during a call" + self._tail())
        try:
            return json.loads(line)
        except ValueError:
            self.kill()
            raise MlipError("protocol", "the worker's reply is not JSON")

    def ping(self, timeout: float = 5.0) -> bool:
        try:
            return self.request({"op": "ping"}, timeout).get("op") == "pong"
        except (MlipError, OSError):
            return False

    def _abort(self) -> None:
        """A start that failed: stop the group, drop the work directory."""
        self.kill()
        self.close()

    def kill(self) -> None:
        self.dead = True
        if self.p is not None:
            proc.stop(self.p)

    def close(self) -> None:
        """quit, then stop the group if it lingers; the work directory goes."""
        if self.p is not None and self.p.poll() is None:
            try:
                self.p.stdin.write(b'{"op": "quit"}\n')
                self.p.stdin.flush()
                self.p.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                self.p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.stop(self.p)
        if self.p is not None:
            for f in (self.p.stdin, self.p.stdout):
                try:
                    if f:
                        f.close()
                except (OSError, ValueError):
                    pass
        self.dead = True
        proc.remove(self.workdir)


class SocketClient:
    """One connection to a host-owned worker (the GUI's pool job)."""

    def __init__(self, path: str, connect_timeout: float = 10.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        target, dfd = path, None
        if len(path) > 100:                    # sun_path is 108 bytes: reach it through the directory's fd
            dfd = os.open(os.path.dirname(path), os.O_RDONLY)
            target = f"/proc/self/fd/{dfd}/{os.path.basename(path)}"
        self.s.settimeout(connect_timeout)
        try:
            self.s.connect(target)
        except OSError as exc:
            self.s.close()
            raise MlipError("crash", f"the ML worker's socket does not answer: {exc}")
        finally:
            if dfd is not None:
                os.close(dfd)
        self.s.setblocking(False)
        self._lines = _Lines(self.s.fileno())
        self._id = 0

    def request(self, msg: Dict[str, Any], timeout: float, canceled=None) -> Dict[str, Any]:
        self._id += 1
        msg = dict(msg, id=self._id) if msg.get("op") == "calc" else msg
        data = (json.dumps(msg) + "\n").encode()
        self.s.setblocking(True)
        try:
            self.s.sendall(data)
        except OSError as exc:
            raise MlipError("crash", f"the ML worker went away: {exc}")
        finally:
            self.s.setblocking(False)
        try:
            line = self._lines.readline(time.monotonic() + timeout, canceled)
        except proc.Canceled:
            self.close()                    # the host gives the in-flight call its grace
            raise
        except TimeoutError:
            self.close()
            raise MlipTimeout(f"the ML worker gave no answer within {timeout:g} s")
        if line is None:
            raise MlipError("crash", "the ML worker closed the connection (it may have crashed)")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.s.close()
        except OSError:
            pass


def _calc_base():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError:
        return object, ()
    return Calculator, all_changes


_Calculator, _ALL = _calc_base()


class MlipCalculator(_Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, transport, *, owner: Optional[WorkerProcess] = None, canceled=None, info=None, **kw):
        super().__init__(**kw)
        self.transport = transport
        self.owner = owner
        self.canceled = canceled
        self.info = info or {}
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL):
        import numpy as np
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        props = ["energy", "forces"] + (["stress"] if "stress" in properties else [])
        msg = {"op": "calc", "numbers": [int(z) for z in a.numbers], "cell": a.cell.array.tolist(),
               "positions": a.positions.tolist(), "pbc": [bool(x) for x in a.pbc], "properties": props}
        rep = self.transport.request(msg, call_timeout(len(a)), self.canceled)
        self.calls += 1
        if not rep.get("ok"):
            raise MlipError(rep.get("kind", "exception"), rep.get("error", "the ML worker failed"))
        e = rep.get("energy")
        if not isinstance(e, (int, float)) or not math.isfinite(e):
            raise MlipError("nan", "the ML potential returned a non-finite energy")
        f = np.asarray(rep.get("forces"), float)
        if f.shape != (len(a), 3):
            raise MlipError("protocol", f"forces shaped {f.shape}, expected ({len(a)}, 3)")
        if not np.all(np.isfinite(f)):
            raise MlipError("nan", "the ML potential returned non-finite forces")
        if np.abs(f).max(initial=0.0) > MAX_FORCE:
            raise MlipError("bigforce", f"a force of {np.abs(f).max():.1f} eV/Å (more than {MAX_FORCE:g}): "
                                        "the ML potential is outside what it was trained on")
        self.results["energy"] = self.results["free_energy"] = float(e)
        self.results["forces"] = f
        if "stress" in props:
            s = np.asarray(rep.get("stress"), float)
            if s.shape != (6,):
                raise MlipError("protocol", "the stress is not 6 Voigt components")
            self.results["stress"] = s

    def close(self) -> None:
        if self.owner is not None:
            self.owner.close()
            self.owner = None
        elif self.transport is not None and hasattr(self.transport, "close"):
            self.transport.close()
        self.transport = None


def make_calculator(model: Dict[str, Any], *, threads: int = 1, device: str = "cpu", lease=None,
                    canceled=None, elements=None, nice: int = 0, python: Optional[str] = None) -> MlipCalculator:
    """The CLI: start a worker of our own (pipe). The GUI: connect to the
    host's worker named by ``lease`` ({"socket", "hello"})."""
    from . import detect
    if lease is not None:
        hello = lease.get("hello") or {}
        _check_hello(hello, model["id"])
        _check_elements(hello, elements)
        return MlipCalculator(SocketClient(lease["socket"]), canceled=canceled, info={"hello": hello})
    py = python or detect.configured_python()
    if not py:
        raise MlipError("setup", "no ML environment is set up (mace mlip setup)")
    ws = detect.weights_status(model)
    if ws["status"] not in ("ready", "bundled"):
        raise MlipError("weights", ws.get("reason") or ws["status"])
    w = WorkerProcess(py, model, weights=ws.get("path"), threads=threads, device=device, canceled=canceled, nice=nice)
    try:
        _check_elements(w.hello, elements)
    except MlipError:
        w.close()
        raise
    return MlipCalculator(w, owner=w, canceled=canceled, info={"hello": w.hello, "device": w.device_text})


def _check_elements(hello, elements) -> None:
    if elements is None:
        return
    z = set(int(x) for x in hello.get("z") or [])
    missing = sorted(set(elements) - z)
    if missing:
        from ase.data import chemical_symbols
        raise MlipError("elements", "the model has no parameters for " +
                        ", ".join(chemical_symbols[m] for m in missing))
