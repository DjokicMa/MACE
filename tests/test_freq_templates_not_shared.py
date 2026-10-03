"""opt2d12's FREQ templates are copied, not shared, between runs.

get_advanced_frequency_settings started from a shallow copy of
FREQ_TEMPLATES[...], so the phonon templates' nested "bands" dict was the
template's own: answers to one run (a k-point density, a template path) were
written into FREQ_TEMPLATES and became the next run's defaults in the same
process. Each run now starts from a fresh deep copy.
"""
import copy
import io

import pytest

import d12_calc_freq
import menu_nav
from conftest import REPO_ROOT

PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr.out"   # R -3 C
SPACE_GROUP = 167


def _answer(monkeypatch, replies):
    def reply(prompt=""):
        for key, value in replies.items():
            if key in prompt:
                return value
        return ""
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", reply)
    monkeypatch.setattr("builtins.input", reply)


def _template_deck(monkeypatch, replies):
    _answer(monkeypatch, {"Select template": "6", **replies})
    settings = d12_calc_freq.get_advanced_frequency_settings()
    f = io.StringIO()
    d12_calc_freq.write_frequency_section(f, settings, space_group=SPACE_GROUP,
                                          optimization_section=PARENT.read_text())
    return f.getvalue()


VECTORS_DENSE = {"Select method (1-4)": "1", "Select format": "2",
                 "Customize k-point density": "y",
                 "Shrink factor for k-points": "8",
                 "Points per band segment": "50"}


@pytest.fixture(autouse=True)
def _restore_templates():
    saved = copy.deepcopy(d12_calc_freq.FREQ_TEMPLATES)
    yield
    d12_calc_freq.FREQ_TEMPLATES.clear()
    d12_calc_freq.FREQ_TEMPLATES.update(saved)


def test_template_6_twice_in_one_process_gives_the_same_deck(monkeypatch):
    first = _template_deck(monkeypatch, {})
    assert "BANDS" in first
    assert _template_deck(monkeypatch, {}) == first


def test_one_runs_answers_are_not_the_next_runs_defaults(monkeypatch):
    """The default label route set the shared template's shrink to 0, so a
    later run in the process never got the k-point density question."""
    pristine = copy.deepcopy(d12_calc_freq.FREQ_TEMPLATES)
    dense = _template_deck(monkeypatch, VECTORS_DENSE)
    assert "BANDS\n8 50\n" in dense
    default = _template_deck(monkeypatch, {})
    assert d12_calc_freq.FREQ_TEMPLATES == pristine
    assert _template_deck(monkeypatch, VECTORS_DENSE) == dense
    assert _template_deck(monkeypatch, {}) == default


def test_template_7_leaves_the_templates_alone(monkeypatch):
    pristine = copy.deepcopy(d12_calc_freq.FREQ_TEMPLATES)
    _answer(monkeypatch, {"Select template": "7"})
    settings = d12_calc_freq.get_advanced_frequency_settings()
    settings["pdos"]["nbins"] = 1
    assert d12_calc_freq.FREQ_TEMPLATES == pristine
