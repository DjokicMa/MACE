#!/usr/bin/env python3
"""Integration tests: back-navigation wired through the REAL shared helpers + wrap points.

Run from Crystal_d12/:  python test_back_integration.py
"""
import sys

import menu_nav
from menu_nav import run_with_back
import d12_constants
from d12_constants import get_user_input, yes_no_prompt, get_valid_input

menu_nav._FORCE_ENABLE = True
_fail = []


def feeder(seq):
    it = iter(list(seq))

    def _read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"feeder exhausted at {prompt!r}")
    return _read


def use(seq):
    menu_nav._REAL_INPUT = feeder(seq)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        _fail.append(name)


def flow():
    a = get_user_input("Pick", {"1": "one", "2": "two"}, "1")
    b = yes_no_prompt("Sure?", "yes")
    c = get_valid_input("Code: ", ["aa", "bb"], None)
    return (a, b, c)


# I1: real helpers, straight flow
use(["2", "yes", "bb"])
r1 = run_with_back(flow)
check("I1 straight through real helpers", r1 == ("2", True, "bb"), str(r1))

# I2: 'b' at get_valid_input -> re-ask yes_no_prompt (answer changes)
use(["2", "yes", "b", "no", "bb"])
r2 = run_with_back(flow)
check("I2 back through real helpers re-asks prior", r2 == ("2", False, "bb"), str(r2))

# I3: back-equals-no-back -> identical config when re-answered the same
use(["2", "yes", "b", "yes", "bb"])
r3 = run_with_back(flow)
check("I3 back-equals-no-back byte-identical", r3 == r1, f"{r3} vs {r1}")

# I4: regression -- helpers behave normally when NOT wrapped (no session)
menu_nav._REAL_INPUT = feeder(["1", "n", "aa"])
a = get_user_input("Pick", {"1": "one", "2": "two"}, "1")
b = yes_no_prompt("Sure?", "yes")
c = get_valid_input("Code: ", ["aa", "bb"], None)
check("I4 unwrapped helpers unchanged", (a, b, c) == ("1", False, "aa"), str((a, b, c)))

# I5: 'b' as a legitimate value is NOT stolen when it's in the valid set
use(["b"])  # valid set literally includes 'b'
got = run_with_back(lambda: get_valid_input("x: ", ["a", "b"], None))
check("I5 'b' kept when it is a valid option", got == "b", str(got))

# I6: imports of the wrapped modules succeed and wrappers are installed
try:
    import d12_calc_basic
    import d12_calc_freq
    import d12_interactive  # noqa: F401
    imported = True
    has_impl = (hasattr(d12_calc_basic, "_configure_optimization_impl")
                and hasattr(d12_calc_basic, "_configure_single_point_impl")
                and hasattr(d12_calc_freq, "_get_frequency_configuration_impl")
                and callable(d12_calc_basic.configure_optimization)
                and callable(d12_calc_freq.get_frequency_configuration))
except Exception as e:
    imported = False
    has_impl = False
    print("   import error:", e)
check("I6 wrapped modules import cleanly", imported)
check("I6 public wrappers + _impl bodies present", has_impl)

if __name__ == "__main__":
    print()
    if _fail:
        print(f"FAILED: {_fail}")
        sys.exit(1)
    print("ALL BACK-INTEGRATION TESTS PASSED")
