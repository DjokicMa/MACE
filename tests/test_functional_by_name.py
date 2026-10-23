"""A functional's name answers the method prompt (1.1.6).

opt2d12 asks the method (Hartree-Fock or DFT), then the functional's family, then the functional. The
functional says all three, so naming it at the first prompt (or at the family prompt) skips the rest:
the same options as the numbers give. Numbers answer the menus as before.
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_interactive  # noqa: E402
import menu_nav  # noqa: E402
from d12_constants import FUNCTIONAL_CATEGORIES  # noqa: E402


def _answers(monkeypatch, typed):
    it = iter(typed)
    asked = []

    def fake(prompt=""):
        asked.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"asked again: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake)
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", fake)
    return asked


def _method(monkeypatch, typed, parent=None):
    asked = _answers(monkeypatch, typed)
    opts = d12_interactive.configure_method(dict(parent or {"functional": "PBE0", "method": "DFT", "dispersion": False}))
    return opts, asked


def _number(name):
    for fam in ("LDA", "GGA", "HYBRID", "MGGA", "3C"):
        if name in FUNCTIONAL_CATEGORIES[fam]["functionals"]:
            return str(["LDA", "GGA", "HYBRID", "MGGA", "3C"].index(fam) + 1), str(FUNCTIONAL_CATEGORIES[fam]["functionals"].index(name) + 1)


@pytest.mark.parametrize("name", ["HSE06", "PBE", "SCAN", "HSESOL3C", "svwn"])
def test_a_name_at_the_method_prompt_is_the_numbers_answer(monkeypatch, name):
    real = next(f for fam in ("LDA", "GGA", "HYBRID", "MGGA", "3C") for f in FUNCTIONAL_CATEGORIES[fam]["functionals"]
                if f.lower() == name.lower())
    fam, num = _number(real)
    d3 = "y" if real in d12_interactive.D3_FUNCTIONALS else None
    by_name, asked_name = _method(monkeypatch, [name] + ([d3] if d3 else []))
    by_number, asked_number = _method(monkeypatch, ["2", fam, num] + ([d3] if d3 else []))
    assert by_name == by_number and by_name["functional"].startswith(real)
    assert len(asked_name) == 1 + bool(d3) and len(asked_number) == 3 + bool(d3)
    assert not any("category" in a or "functional" in a.lower() and "Add" not in a for a in asked_name[1:])


def test_a_name_at_the_family_prompt_skips_the_functional_menu(monkeypatch):
    by_name, asked = _method(monkeypatch, ["2", "r2SCAN"])
    assert by_name["functional"] == "r2SCAN" and len(asked) == 2


def test_a_hartree_fock_name_skips_the_variant_menu(monkeypatch):
    opts, asked = _method(monkeypatch, ["UHF"])
    assert opts["functional"] == "UHF" and opts["method_type"] == "HF" and len(asked) == 1


def test_numbers_and_unknown_names_behave_as_before(monkeypatch):
    opts, asked = _method(monkeypatch, ["2", "3", "4", "n"])
    assert opts["method_type"] == "DFT" and len(asked) == 4
    # an unknown name is not a key: the menu says so and asks again
    opts, asked = _method(monkeypatch, ["NOT-A-FUNCTIONAL", "2", "3", "4", "n"])
    assert opts["method_type"] == "DFT" and len(asked) == 5
