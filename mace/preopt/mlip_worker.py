"""The ML-potential worker: a standalone script run in the ML interpreter.

    python -I mlip_worker.py --model-id ID --backend mace_mp|chgnet|orb|mattersim|fairchem
                             --weights PATH --device cpu|cuda:0|auto --dtype float32|float64
                             --threads K [--socket PATH] [--variant NAME] [--task HEAD]
                             [--gpu-mem-fraction F]

``--device auto`` takes cuda:0 when torch sees a GPU (the client has already
narrowed CUDA_VISIBLE_DEVICES to one GPU, GPU 0 unless the user chose
another), else the CPU; the hello reports which one it took.

It is never imported as ``mace.*``: inside the ML environment ``import mace``
must be ACEsuit's mace-torch (MACE-MP), which is unrelated to this
MACE tool. ``-I``
(isolated mode) keeps PYTHONPATH and the script directory off sys.path, and
the worker checks before loading anything that ``mace`` really belongs to
the installed ``mace-torch`` distribution (exit 3 otherwise).

Protocol 1: newline-delimited JSON. Once the model is loaded the worker
prints ``{"op": "hello", ...}`` on stdout. Then requests come on stdin (pipe
mode) or over the Unix socket ``--socket`` (one connection at a time):

    {"id": n, "op": "calc", "numbers": [...], "cell": [[3x3]], "positions": [[Nx3]],
     "pbc": [b, b, b], "properties": ["energy", "forces", "stress"],
     "charge": q, "spin": m, "task": "omol"}     (charge/spin/task only for the models that take them)
    -> {"id": n, "ok": true, "energy": E, "forces": [[...]], "stress": [6]}   (eV, Å; ase Voigt)
    -> {"id": n, "ok": false, "kind": "nan"|"oom"|"elements"|"exception", "error": "..."}
    {"op": "ping"} -> {"op": "pong"};  {"op": "quit"} or stdin EOF -> exit 0

The worker never downloads anything: the model is given as a local file
(mace_mp, orb, mattersim), ships inside the chgnet wheel, or (fairchem: UMA)
is read from the Hugging Face cache, offline, after ``mace mlip fetch``
downloaded it with the user's own access.
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time

PROTOCOL = 1
MAX_LINE = 64 * 1024 * 1024
OUT = None          # the protocol stream: the original stdout (see claim_stdout)


def claim_stdout():
    """Keep the real stdout for the protocol only. The model libraries print
    to stdout (mace_mp announces the model it loads, torch warns), so file
    descriptor 1 and sys.stdout are pointed at stderr (the worker's log)."""
    global OUT
    OUT = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def send(obj):
    OUT.write(json.dumps(obj) + "\n")
    OUT.flush()


def _fail(code, msg):
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()
    os._exit(code)


MACE_TORCH_BACKENDS = ("mace_mp",)


def self_check(backend):
    """mace_mp: 'mace' must be the module the mace-torch distribution installed."""
    if backend not in MACE_TORCH_BACKENDS:
        return
    import importlib
    import importlib.metadata as md
    try:
        dist = md.distribution("mace-torch")
    except md.PackageNotFoundError:
        _fail(3, "package name clash: the mace-torch distribution is not installed in this interpreter")
    try:
        mod = importlib.import_module("mace")
    except Exception as exc:
        _fail(3, f"package name clash: import mace failed: {exc}")
    owned = {os.path.realpath(str(dist.locate_file(f))) for f in (dist.files or [])}
    here = os.path.realpath(getattr(mod, "__file__", "") or "")
    if here not in owned:
        _fail(3, f"package name clash: 'mace' resolves to {here}, which the mace-torch distribution "
                 "does not own (this MACE tool on the path?)")
    try:
        importlib.import_module("mace.calculators")
        from mace.calculators import mace_mp  # noqa: F401
    except Exception as exc:
        _fail(3, f"package name clash: mace.calculators.mace_mp is not importable: {exc}")


def pick_device(requested, torch):
    """auto: cuda:0 when torch sees a GPU, else cpu."""
    if requested == "auto":
        try:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return requested


def _weights(args):
    if not args.weights or not os.path.isfile(args.weights):
        _fail(4, f"weights file not found: {args.weights!r}")
    return args.weights


def load(args):
    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    dev = pick_device(args.device, torch)
    args.device = dev
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.01, float(args.gpu_mem_fraction))), 0)
    versions = {"python": "%d.%d.%d" % sys.version_info[:3], "torch": torch.__version__}
    import ase
    versions["ase"] = ase.__version__
    import importlib.metadata as md
    if args.backend in MACE_TORCH_BACKENDS:
        from mace.calculators import mace_mp
        calc = mace_mp(model=_weights(args), device=dev, default_dtype=args.dtype)
        versions["mace-torch"] = md.version("mace-torch")
        try:
            z = sorted(int(x) for x in calc.z_table.zs)
        except Exception:
            z = list(range(1, 90))
    elif args.backend == "orb":
        from orb_models.forcefield import pretrained
        try:
            from orb_models.forcefield.inference.calculator import ORBCalculator
        except ImportError:                       # orb-models before 0.6
            from orb_models.forcefield.calculator import ORBCalculator
        loader = pretrained.ORB_PRETRAINED_MODELS.get(args.variant)
        if loader is None:
            _fail(4, f"orb-models has no model {args.variant!r} (installed version {md.version('orb-models')})")
        precision = args.dtype if args.dtype in ("float32-high", "float32-highest", "float64") else "float32-high"
        got = loader(weights_path=_weights(args), device=dev, precision=precision, compile=False)
        if isinstance(got, tuple):
            calc = ORBCalculator(got[0], atoms_adapter=got[1], device=dev)
        else:
            calc = ORBCalculator(got, device=dev)
        versions["orb-models"] = md.version("orb-models")
        z = list(range(1, 119))
    elif args.backend == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(load_path=_weights(args), device=dev)
        versions["mattersim"] = md.version("mattersim")
        z = list(range(1, 95))
    elif args.backend == "fairchem":
        os.environ["HF_HUB_OFFLINE"] = "1"            # the weights were fetched with the user's own access
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        unit = pretrained_mlip.get_predict_unit(args.variant, device="cuda" if dev.startswith("cuda") else "cpu")
        calc = TaskCalcs(lambda task: FAIRChemCalculator(unit, task_name=task), args.task or "omat")
        versions["fairchem-core"] = md.version("fairchem-core")
        z = list(range(1, 101))
    elif args.backend == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        model = CHGNet.load(model_name="0.3.0", use_device=dev, verbose=False)
        calc = CHGNetCalculator(model=model, use_device=dev)
        versions["chgnet"] = md.version("chgnet")
        z = list(range(1, 95))
    else:
        _fail(2, f"unknown backend {args.backend!r}")
    return calc, z, versions


class TaskCalcs:
    """One calculator per task head over one loaded model (UMA): the head is
    chosen per call."""

    def __init__(self, make, default):
        self.make, self.default, self.calcs = make, default, {}

    def get(self, task):
        task = task or self.default
        if task not in self.calcs:
            self.calcs[task] = self.make(task)
        return self.calcs[task]


def compute(calc, msg, zset):
    from ase import Atoms
    import numpy as np
    nums = [int(n) for n in msg["numbers"]]
    if not set(nums) <= zset:
        missing = sorted(set(nums) - zset)
        return {"id": msg.get("id"), "ok": False, "kind": "elements",
                "error": f"the model has no parameters for Z = {missing}"}
    atoms = Atoms(numbers=nums, cell=msg["cell"], positions=msg["positions"], pbc=msg.get("pbc", [True] * 3))
    if "charge" in msg:                   # OrbMol-v2, UMA omol: total charge and spin multiplicity (2S+1)
        atoms.info["charge"] = int(msg["charge"])
        atoms.info["spin"] = int(msg["spin"])
    atoms.calc = calc.get(msg.get("task")) if isinstance(calc, TaskCalcs) else calc
    props = msg.get("properties") or ["energy", "forces"]
    e = float(atoms.get_potential_energy())
    out = {"id": msg.get("id"), "ok": True, "energy": e}
    if "forces" in props:
        out["forces"] = np.asarray(atoms.get_forces(), float).tolist()
    if "stress" in props:
        out["stress"] = np.asarray(atoms.get_stress(voigt=True), float).tolist()
    if not math.isfinite(e) or ("forces" in out and not np.all(np.isfinite(out["forces"]))):
        return {"id": msg.get("id"), "ok": False, "kind": "nan", "error": "the model returned NaN or Inf"}
    return out


def answer(calc, line, zset):
    """(reply dict or None, quit?, fatal?)"""
    try:
        msg = json.loads(line)
    except ValueError:
        return {"ok": False, "kind": "exception", "error": "not JSON"}, False, False
    op = msg.get("op")
    if op == "ping":
        return {"op": "pong"}, False, False
    if op == "quit":
        return None, True, False
    if op != "calc":
        return {"id": msg.get("id"), "ok": False, "kind": "exception", "error": f"unknown op {op!r}"}, False, False
    try:
        return compute(calc, msg, zset), False, False
    except MemoryError:
        return {"id": msg.get("id"), "ok": False, "kind": "oom", "error": "out of memory"}, False, True
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            return {"id": msg.get("id"), "ok": False, "kind": "oom", "error": str(exc).splitlines()[0][:300]}, False, True
        return {"id": msg.get("id"), "ok": False, "kind": "exception",
                "error": (type(exc).__name__ + ": " + str(exc)).splitlines()[0][:300]}, False, False


def serve_pipe(calc, zset):
    for line in sys.stdin:
        if len(line) > MAX_LINE:
            _fail(2, "protocol error: line too long")
        if not line.strip():
            continue
        rep, quit_, fatal = answer(calc, line, zset)
        if quit_:
            break
        send(rep)
        if fatal:
            os._exit(5)
    os._exit(0)


def _watch_stdin():
    """Socket mode: stdin stays open as a watchdog; EOF means the owner died."""
    try:
        while sys.stdin.readline():
            pass
    except Exception:
        pass
    os._exit(0)


def bind_socket(path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        srv.bind(os.path.basename(path) if len(path) > 100 else path)   # cwd is the socket's directory
    finally:
        os.umask(old)
    srv.listen(4)
    return srv


def serve_socket(calc, zset, srv):
    threading.Thread(target=_watch_stdin, daemon=True).start()
    while True:
        conn, _ = srv.accept()
        with conn:
            rf = conn.makefile("r", encoding="utf-8", newline="\n")
            for line in rf:
                if len(line) > MAX_LINE:
                    break
                if not line.strip():
                    continue
                rep, quit_, fatal = answer(calc, line, zset)
                if quit_:
                    os._exit(0)
                try:
                    conn.sendall((json.dumps(rep) + "\n").encode())
                except OSError:          # the client went away mid-call: the reply is dropped
                    break
                if fatal:
                    os._exit(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--socket", default=None)
    ap.add_argument("--variant", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    args = ap.parse_args()
    claim_stdout()
    import warnings
    warnings.filterwarnings("ignore")
    self_check(args.backend)
    t0 = time.perf_counter()
    calc, z, versions = load(args)
    srv = bind_socket(args.socket) if args.socket else None
    mod = None
    if args.backend in MACE_TORCH_BACKENDS:
        import mace
        mod = os.path.realpath(mace.__file__)
    hello = {"op": "hello", "protocol": PROTOCOL, "model": args.model_id, "backend": args.backend,
             "device": args.device, "dtype": args.dtype, "threads": args.threads, "pid": os.getpid(),
             "versions": versions, "z": z, "load_s": round(time.perf_counter() - t0, 3), "module": mod}
    send(hello)
    if srv is not None:
        serve_socket(calc, set(z), srv)
    else:
        serve_pipe(calc, set(z))


if __name__ == "__main__":
    main()
