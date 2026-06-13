#!/usr/bin/env python3
"""Self-tests for menu_nav.py (the 'press b to go back' controller).

Run: python test_menu_nav.py    (exit 0 = all pass)

Tests drive the controller with a scripted feeder of the LIVE inputs the user would type
(replayed answers do not consume the feeder, matching real usage). _FORCE_ENABLE is set so
the controller runs without a real TTY.
"""
import builtins
import io
import sys

import menu_nav
from menu_nav import run_with_back, nav_read, MenuBack, back_available_live

menu_nav._FORCE_ENABLE = True

_failures = []


def feeder(seq):
    """Return a callable that yields scripted live inputs, raising if over-consumed."""
    it = iter(list(seq))

    def _read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"feeder exhausted at prompt {prompt!r}")
    return _read


def use(seq):
    menu_nav._REAL_INPUT = feeder(seq)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        _failures.append(name)
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def ask(prompt, valid):
    """A helper-style validation loop (mirrors get_user_input/get_user_choice)."""
    while True:
        x = nav_read(prompt, valid_set=valid)
        if x in valid:
            return x
        print("invalid, try again")


# T1: inert when no session active -> nav_read == input
def t1():
    menu_nav._REAL_INPUT = lambda p="": "raw"
    check("T1 inert (no session): nav_read returns plain input", nav_read("q", {"1"}) == "raw")
    check("T1 inert: back_available_live() False with no session", back_available_live() is False)


# T2: straight flow records and returns
def t2():
    use(["1", "x", "p"])
    r = run_with_back(lambda: (nav_read("q1", {"1", "2"}), nav_read("q2", {"x", "y"}), nav_read("q3", {"p", "q"})))
    check("T2 straight flow", r == ("1", "x", "p"), str(r))


# T3: 'b' at q3 -> re-ask q2, q1 preserved
def t3():
    use(["1", "x", "b", "y", "p"])
    r = run_with_back(lambda: (nav_read("q1", {"1", "2"}), nav_read("q2", {"x", "y"}), nav_read("q3", {"p", "q"})))
    check("T3 back one step (q1 kept, q2 changed)", r == ("1", "y", "p"), str(r))


# T4: 'b' twice from q3 -> land back at q1
def t4():
    use(["1", "x", "b", "b", "2", "y", "p"])
    r = run_with_back(lambda: (nav_read("q1", {"1", "2"}), nav_read("q2", {"x", "y"}), nav_read("q3", {"p", "q"})))
    check("T4 back twice lands at q1", r == ("2", "y", "p"), str(r))


# T5: 'b' at the FIRST question is not a back signal (nothing to undo) -> treated as input
def t5():
    use(["b", "1", "x"])  # 'b' at q1 -> not intercepted -> invalid -> reprompt
    r = run_with_back(lambda: (ask("q1", {"1", "2"}), ask("q2", {"x", "y"})))
    check("T5 'b' at first question is a literal (no back)", r == ("1", "x"), str(r))


# T6: invalid-then-valid retry, then back -> goes back one input()-call (per design)
def t6():
    # q1: invalid 'zz' then '1'; q2: 'x'; q3: 'b' -> back lands in q2's loop live re-prompt
    use(["zz", "1", "x", "b", "y", "p"])
    r = run_with_back(lambda: (ask("q1", {"1", "2"}), ask("q2", {"x", "y"}), ask("q3", {"p", "q"})))
    check("T6 retry then back completes correctly", r == ("1", "y", "p"), str(r))


# T7: MenuBack propagates out of a helper-style while-loop
def t7():
    use(["1", "b", "2", "x"])  # q2 'b' -> back to q1 (inside ask's loop)
    r = run_with_back(lambda: (ask("q1", {"1", "2"}), ask("q2", {"x", "y"})))
    check("T7 back through helper loop", r == ("2", "x"), str(r))


# T8: builtins.input and sys.stdout restored after a non-MenuBack exception
def t8():
    orig_input, orig_stdout = builtins.input, sys.stdout

    def boom():
        nav_read("q1", {"1"})
        raise ValueError("kaboom")
    use(["1"])
    raised = False
    try:
        run_with_back(boom)
    except ValueError:
        raised = True
    check("T8 propagates non-MenuBack exception", raised)
    check("T8 builtins.input restored after exception", builtins.input is orig_input)
    check("T8 sys.stdout restored after exception", sys.stdout is orig_stdout)


# T9: branching flow (q2 options depend on q1) replays deterministically across a back
def t9():
    def flow():
        a = nav_read("kind", {"num", "let"})
        if a == "num":
            b = nav_read("pick-num", {"1", "2"})
        else:
            b = nav_read("pick-let", {"x", "y"})
        c = nav_read("confirm", {"ok", "no"})
        return (a, b, c)
    # choose num/1, then 'b' at confirm -> re-ask pick-num as '2', then confirm ok
    use(["num", "1", "b", "2", "ok"])
    r = run_with_back(flow)
    check("T9 branching replay deterministic", r == ("num", "2", "ok"), str(r))


# T10: free-text (valid_set=None) never intercepts 'b'/'B'
def t10():
    use(["1", "B", "ok"])  # 'B' is a legitimate Brillouin label entered free-text
    r = run_with_back(lambda: (nav_read("q1", {"1", "2"}), nav_read("label", None), nav_read("c", {"ok"})))
    check("T10 free-text 'B' not stolen as back", r == ("1", "B", "ok"), str(r))
    # also via raw builtins.input inside a wrapped flow
    use(["1", "B", "ok"])
    r2 = run_with_back(lambda: (nav_read("q1", {"1", "2"}), input("label: "), nav_read("c", {"ok"})))
    check("T10 raw input() 'B' not stolen", r2 == ("1", "B", "ok"), str(r2))


# T11: buffering silencer -> landed question's header shown, earlier ones suppressed
def t11():
    def flow():
        print("HEADER-Q1")
        a = nav_read("q1> ", {"1", "2"})
        print("HEADER-Q2")
        b = nav_read("q2> ", {"x", "y"})
        print("HEADER-Q3")
        c = nav_read("q3> ", {"p", "q"})
        return (a, b, c)
    use(["1", "x", "b", "y", "p"])
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        r = run_with_back(flow)
    finally:
        sys.stdout = real
    out = buf.getvalue()
    # On the back+replay pass, HEADER-Q1 (replayed/answered) is suppressed but HEADER-Q2
    # (the landed live question) is shown. We assert Q2's header appears for the re-ask.
    check("T11 result correct with prints", r == ("1", "y", "p"), str(r))
    check("T11 landed question header shown on back", out.count("HEADER-Q2") >= 2, repr(out[-200:]))


# T12: MenuBack (BaseException) is NOT swallowed by `except Exception:`
def t12():
    def flow():
        a = nav_read("q1", {"1", "2"})
        try:
            b = nav_read("q2", {"x", "y"})
        except Exception:
            b = "SWALLOWED"
        return (a, b)
    use(["1", "b", "2", "x"])  # 'b' at q2 -> MenuBack must skip the except Exception guard
    r = run_with_back(flow)
    check("T12 except Exception does not catch MenuBack", r == ("2", "x"), str(r))


# T13: back-equals-no-back -> identical final result with/without an intervening back
def t13():
    def flow():
        return (nav_read("q1", {"1", "2"}), nav_read("q2", {"x", "y"}), nav_read("q3", {"p", "q"}))
    use(["1", "y", "p"])
    a = run_with_back(flow)
    use(["1", "x", "b", "y", "p"])  # detour through q2='x' then back, end same
    b = run_with_back(flow)
    check("T13 back-equals-no-back byte-identical result", a == b == ("1", "y", "p"), f"{a} vs {b}")


if __name__ == "__main__":
    for t in [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]:
        t()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} -> {_failures}")
        sys.exit(1)
    print("ALL MENU_NAV TESTS PASSED")
