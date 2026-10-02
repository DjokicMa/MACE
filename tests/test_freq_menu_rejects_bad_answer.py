"""opt2d12's FREQ menus ask again on an answer that is not on the menu.

The FREQ template menu ("Select template (1-8)") used to take any other
answer as "8: Custom settings", the FREQCALC/ANHARM question took anything
but "2" as FREQCALC, and the crystal-system template menus of the phonon
band path fell back to their first entry without saying so.
"""
import pytest

import menu_nav
import d12_calc_freq


class Answers:
    """Answer prompts by their text; anything not listed gets a blank answer."""

    def __init__(self, scripted):
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if len(self.prompts) > 200:
            raise AssertionError("menu keeps asking")
        for key, answers in self.scripted.items():
            if prompt.startswith(key) and answers:
                return answers.pop(0)
        return ""

    def count(self, prompt):
        """How often ``prompt`` was shown (a " [b=back]" suffix is ignored)."""
        return sum(p.replace(" [b=back]", "").rstrip() == prompt.rstrip()
                   for p in self.prompts)


@pytest.fixture
def answers(monkeypatch):
    def use(scripted):
        a = Answers(scripted)
        monkeypatch.setattr(menu_nav, "_REAL_INPUT", a)
        monkeypatch.setattr("builtins.input", a)
        return a
    return use


def test_bad_template_answer_is_asked_again_not_taken_as_custom(answers, capsys):
    a = answers({"Select template (1-8) [1]: ": ["9", "1"]})
    settings = d12_calc_freq.get_advanced_frequency_settings()
    assert a.count("Select template (1-8) [1]: ") == 2
    assert "CUSTOM FREQUENCY SETTINGS" not in capsys.readouterr().out
    assert settings["mode"] == "GAMMA"


def test_bad_frequency_type_is_asked_again(answers):
    a = answers({"Select type (1-2) [1]: ": ["3", "2"],
                 "\nEnter label (atom number)": ["4"]})
    got = d12_calc_freq.get_frequency_configuration()
    assert a.count("Select type (1-2) [1]: ") == 2
    assert got["freq_mode"] == "ANHARM"


def test_bad_cubic_template_answer_is_asked_again(answers):
    """Phonon bands -> template path -> cubic: "5" is not one of 1-3."""
    a = answers({"Select template (1-8) [1]: ": ["6"],
                 "Select method (1-4) [1]: ": ["2"],
                 "Select system (1-6) [1]: ": ["1"],
                 "Select template (1-3) [1]: ": ["5", "2"]})
    settings = d12_calc_freq.get_advanced_frequency_settings()
    assert a.count("Select template (1-3) [1]: ") == 2
    assert settings["bands"]["template"] == "cubic_fc"
