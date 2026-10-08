"""Two pre-optimizations with the same method in one process (the GUI's pool
threads) each run their own tier's calculator.

core.preoptimize used to put each tier's factory into the shared
mace.gui.clean.ADAPTERS entry around the run and pop it afterwards, so a
thread could build its calculator with the other thread's factory (another
charge, another backend), or find the entry gone and fall back to the
springs silently. The factory now goes to clean() as an argument.

The test forces the interleaving that crossed them: thread A pauses between
starting its tier and building the calculator until thread B has started
its own tier with the same method.
"""
import threading

import pytest

pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)


def test_same_method_in_two_threads_never_shares_a_factory(monkeypatch):
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from mace.gui import clean as C
    from mace.preopt import core

    calls = []                       # (thread, factory tag, charge)
    b_started = threading.Event()
    a_built = threading.Event()
    real_hook = core._hook

    def hook(*args, **kwargs):
        me = threading.current_thread().name
        if me == "B":
            b_started.set()
        elif me == "A":
            assert b_started.wait(30)
        return real_hook(*args, **kwargs)

    def make_resolve(tag):
        def factory(atoms, **opts):
            me = threading.current_thread().name
            calls.append((me, tag, opts.get("charge")))
            if me == "A":
                a_built.set()
            else:
                a_built.wait(5)      # keep B's run (and, before the fix, its entry) alive
            return LennardJones(sigma=2.0, epsilon=0.01, rc=5.0)

        def resolve(m, **kw):
            return core.Tier(m, core.method_kind(m), f"stub-{tag}", True, "", factory, {})
        return resolve

    monkeypatch.setattr(core, "_hook", hook)
    monkeypatch.setattr(core, "refusal", lambda *a, **k: None)
    results, errors = {}, []

    def run(tag, charge):
        try:
            a = bulk("NaCl", "rocksalt", a=5.64)
            a.rattle(0.05, seed=1)
            results[tag] = core.preoptimize(a, "gfnff", charge=charge, resolve=make_resolve(tag), steps=5,
                                            time_limit=5, route="gui")
        except Exception as exc:     # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, name="A", args=("A", 0)),
               threading.Thread(target=run, name="B", args=("B", 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    assert sorted(calls) == [("A", "A", 0), ("B", "B", 2)]
    for tag in ("A", "B"):
        tiers = [(x["method"], x["status"]) for x in results[tag].attempts]
        assert ("gfnff", "ok") in tiers, results[tag].attempts
    assert "gfnff" not in C.ADAPTERS
