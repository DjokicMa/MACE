"""A stand-in ML worker for the tests: the real ``mlip_worker.py`` serving
code (pipe and socket, the same protocol) with a harmonic pair potential in
place of a model, so no torch is needed.

FAKE_MODE selects misbehavior: crash (exit mid-call), nan, hang (a call that
never returns), bigforce, badhello (protocol 2), slowload (10 s before the
hello), oom, chatty (prints to stdout while loading), slowcall (FAKE_SLOWCALL seconds per call). FAKE_HELLO_LOG gets
one line per hello (one line = one model load); FAKE_ENV_LOG gets the
worker's environment, cwd and argv as JSON. FAKE_MODE_FILE, when it exists,
sets the mode again before every call (tests that change it mid-session).
"""
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "mace", "preopt", "mlip_worker.py")
spec = importlib.util.spec_from_file_location("mlip_worker_under_test", WORKER)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
MODE = os.environ.get("FAKE_MODE", "")


def _calc_class():
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    class Harmonic(Calculator):
        """Pairs within 1.3 x the sum of covalent radii pull to that sum."""
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            a = self.atoms
            rad = np.array([covalent_radii[z] for z in a.numbers])
            ii, jj, D = neighbor_list("ijD", a, list(1.3 * rad))
            e, f, vir = 0.0, np.zeros((len(a), 3)), np.zeros((3, 3))
            for i, j, d in zip(ii, jj, D):
                r = np.linalg.norm(d)
                r0 = rad[i] + rad[j]
                e += 0.25 * 2.0 * (r - r0) ** 2          # each pair listed twice
                g = 0.5 * 2.0 * (r - r0) * d / r
                f[i] += g
                f[j] -= g
                vir += np.outer(g, d)
            v = a.get_volume()
            s = 0.5 * (vir + vir.T) / v
            self.results = {"energy": e, "forces": f,
                            "stress": np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])}
    return Harmonic


def fake_self_check(backend):
    return None


def fake_load(args):
    if MODE == "slowload":
        time.sleep(10)
    if MODE == "chatty":            # what mace_mp and torch do: print to stdout while loading
        print("Using Materials Project MACE for MACECalculator with /some/model")
        os.write(1, b"a C-level line on fd 1\n")
    log = os.environ.get("FAKE_HELLO_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(f"{os.getpid()}\n")
    envlog = os.environ.get("FAKE_ENV_LOG")
    if envlog:
        with open(envlog, "w") as fh:
            json.dump({"env": dict(os.environ), "cwd": os.getcwd(), "argv": sys.argv,
                       "nice": os.nice(0), "flags_isolated": sys.flags.isolated}, fh)
    z = list(range(1, 84)) if os.environ.get("FAKE_NO_HEAVY") else list(range(1, 95))
    return _calc_class()(), z, {"python": "%d.%d.%d" % sys.version_info[:3], "torch": "0.0-fake",
                                "ase": "fake", "mace-torch": "0.0-fake"}


_real_compute = W.compute


def fake_compute(calc, msg, zset):
    global MODE
    mf = os.environ.get("FAKE_MODE_FILE")
    if mf and os.path.exists(mf):
        MODE = open(mf).read().strip()
    if MODE == "crash":
        os._exit(9)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "slowcall":
        time.sleep(float(os.environ.get("FAKE_SLOWCALL", "1")))
    if MODE == "oom":
        raise MemoryError()
    out = _real_compute(calc, msg, zset)
    if MODE == "nan" and out.get("ok"):
        return {"id": msg.get("id"), "ok": False, "kind": "nan", "error": "the model returned NaN or Inf"}
    if MODE == "nanenergy" and out.get("ok"):
        out["energy"] = math.nan
    if MODE == "bigforce" and out.get("ok"):
        out["forces"] = [[x * 0 + 99.0 for x in row] for row in out["forces"]]
    return out


W.self_check = fake_self_check
W.load = fake_load
W.compute = fake_compute
if MODE == "badhello":
    W.PROTOCOL = 2

if __name__ == "__main__":
    W.main()
