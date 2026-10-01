"""'b' to go back through the real interactive questionnaires.

test_menu_nav.py covers the controller and test_back_integration.py the shared
prompt helpers. This drives the questionnaires that are wrapped for back
navigation - opt2d12's optimization, single-point and frequency menus
(Crystal_d12/d12_calc_basic.py, d12_calc_freq.py) and opt2d3's property menus
(Crystal_d3/d3_interactive.py, wrapped the way CRYSTALOptToD3 wraps them) -
with scripted answers, and checks the configuration each returns and how many
times the terminal was actually read (answers replayed after a 'b' are not
read again).

The workflow planner's own prompts (mace/workflow/planner.py) have no back
navigation; nothing here applies to them.
"""
import io

import pytest

import menu_nav
import d12_calc_basic
import d12_calc_freq
import d3_interactive

STANDARD = {"toldeg": 0.0003, "toldex": 0.0012, "toldee": 7, "maxcycle": 800}


class Terminal:
    """Scripted answers for menu_nav's real reads; records every prompt shown."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"no scripted answer left for {prompt!r}")
        return self.answers.pop(0)

    @property
    def reads(self):
        return len(self.prompts)


@pytest.fixture
def terminal(monkeypatch):
    """Back navigation on (as at a TTY) with the terminal scripted."""
    monkeypatch.setattr(menu_nav, "_FORCE_ENABLE", True)

    def use(answers):
        t = Terminal(answers)
        monkeypatch.setattr(menu_nav, "_REAL_INPUT", t)
        return t
    return use


def opt(**parent):
    return d12_calc_basic.configure_optimization(parent or None)


# ------------------------------------------------------------ optimization

def test_optimization_straight_through(terminal):
    t = terminal(["2", "3", "n"])
    assert opt() == {"type": "CELLONLY", "convergence": "Very Tight", "toldeg": 3e-05,
                     "toldex": 0.00012, "toldee": 9, "maxcycle": 800, "maxtradius": None}
    assert t.reads == 3
    assert "[b=back]" not in t.prompts[0]              # nothing to go back to yet
    assert all(p.rstrip().endswith("[b=back]") for p in t.prompts[1:])


def test_back_from_maxtradius_changes_the_convergence_level(terminal):
    t = terminal(["1", "1", "b", "3", "n"])
    got = opt()
    assert got["convergence"] == "Very Tight" and got["toldee"] == 9
    assert got["maxtradius"] is None
    assert t.reads == 5
    # the question landed on shows what was answered there before
    assert "(previously: 1) [b=back]" in t.prompts[3]


def test_back_twice_reaches_the_first_question(terminal):
    t = terminal(["1", "2", "b", "b", "4", "", "n"])
    got = opt()
    assert got["type"] == "ITATOCEL" and got["convergence"] == "Standard"
    assert t.reads == 7
    assert t.prompts[4].startswith("Enter your choice:")


def test_back_inside_custom_tolerances(terminal):
    """'b' at a number prompt goes back one number, never into float()."""
    t = terminal(["1", "4", "0.0001", "b", "0.0002", "0.0008", "8", "back", "9", "400",
                  "y", "0.1"])
    got = opt()
    assert got == {"type": "FULLOPTG", "convergence": "Custom", "toldeg": 0.0002,
                   "toldex": 0.0008, "toldee": 9, "maxcycle": 400, "maxtradius": 0.1}
    assert t.reads == 12


def test_going_back_and_giving_the_same_answers_changes_nothing(terminal):
    terminal(["6", "2", "y", "0.3"])
    straight = opt()
    t = terminal(["6", "2", "y", "b", "y", "0.3"])
    assert opt() == straight
    assert t.reads == 6


def test_parent_settings_stay_the_defaults_after_going_back(terminal):
    """A blank answer keeps the parent's OPTGEOM, also on a question reached
    again with 'b'."""
    parent = {"type": "CELLONLY", "toldeg": 1e-4, "toldex": 4e-4, "toldee": 8,
              "maxcycle": 300, "maxtradius": 0.25}
    t = terminal(["", "", "b", "", "", ""])
    got = opt(**parent)
    assert got == {"type": "CELLONLY", "maxcycle": 300, "toldeg": 1e-4, "toldex": 4e-4,
                   "toldee": 8, "maxtradius": 0.25}
    assert t.reads == 6


def test_b_at_the_first_question_is_just_an_invalid_answer(terminal):
    t = terminal(["b", "1", "1", "n"])
    assert opt()["type"] == "FULLOPTG"
    assert t.reads == 4


def test_unknown_convergence_level_is_asked_again(terminal):
    t = terminal(["1", "5", "1", "n"])
    assert opt()["convergence"] == "Standard"
    assert t.reads == 4


# ------------------------------------------------------------ single point

def test_single_point(terminal):
    t = terminal(["y"])
    got = d12_calc_basic.configure_single_point()
    assert got["use_tight_sp"] is True and got["tolerances"]["TOLDEE"] == 9
    assert t.reads == 1


# ------------------------------------------------------------ frequencies

def test_frequency_back_to_the_type_question(terminal):
    """Raman template, then back twice to the FREQCALC/ANHARM question."""
    t = terminal(["1", "3", "b", "b", "2", "7", "n", "n", "n"])
    got = d12_calc_freq.get_frequency_configuration()
    assert got["freq_mode"] == "ANHARM"
    assert got["anharm_settings"] == {"h_atom": 7}
    assert t.reads == 9
    assert "(previously: 3)" in t.prompts[3] and "(previously: 1)" in t.prompts[4]


def test_frequency_ir_method_after_going_back(terminal):
    t = terminal(["1", "1", "2", "y", "1", "b", "2", "n"])
    got = d12_calc_freq.get_frequency_configuration()
    assert got["freq_mode"] == "FREQCALC"
    assert got["numderiv"] == 2 and got["intensities"] is True
    assert got["ir_method"] == "WANNIER"
    assert t.reads == 8


def test_a_b_label_in_a_free_text_answer_is_not_taken_as_back(terminal):
    """Free-text prompts (the H atom label here) are read with plain input():
    an answer of 'b' there is data, not navigation."""
    t = terminal(["2", "b", "n", "n", "n"])
    got = d12_calc_freq.get_frequency_configuration()
    assert got["freq_mode"] == "ANHARM"
    assert got["anharm_settings"]["h_atom"] == 1        # 'b' is not a number
    assert t.reads == 5


@pytest.mark.xfail(strict=True, raises=KeyError, reason="an answer outside 1-3 at the FREQ "
                   "IR-method menu raises KeyError instead of being asked again")
def test_unknown_ir_method_is_asked_again(terminal):
    terminal(["1", "1", "", "y", "4", "3", "n"])
    assert d12_calc_freq.get_frequency_configuration()["ir_method"] == "CPHF"


@pytest.mark.xfail(strict=True, raises=ValueError, reason="a non-number at the FREQ NUMDERIV "
                   "menu raises ValueError from int() instead of being asked again")
def test_unknown_numderiv_is_asked_again(terminal):
    terminal(["1", "1", "x", "1", "n", "n"])
    assert d12_calc_freq.get_frequency_configuration()["numderiv"] == 1


# ------------------------------------------------------------ opt2d3

def d3(calc_type):
    # CRYSTALOptToD3 runs the questionnaire inside run_with_back.
    return menu_nav.run_with_back(lambda: d3_interactive.configure_d3_calculation(calc_type))


def test_transport_back_one_number(terminal):
    t = terminal(["", "", "", "2", "", "", "b", "1.5", "0.02", "", "", "", "20", "n"])
    got = d3("TRANSPORT")
    assert got == {"temperature_range": (100.0, 800.0, 50.0),
                   "mu_range": (-2.0, 1.5, 0.02), "mu_reference": "vbm",
                   "tdf_range": (-5.0, 5.0, 0.01), "relaxation_time": 20.0,
                   "calculation_type": "TRANSPORT"}
    assert t.reads == 14


def test_transport_back_across_the_smearing_question(terminal):
    t = terminal(["", "", "", "3", "", "", "", "", "", "", "", "y", "b", "y", "0.05", "1"])
    got = d3("TRANSPORT")
    assert got["mu_reference"] == "absolute"
    assert (got["smearing"], got["smearing_type"]) == (0.05, 1)
    assert t.reads == 16


# ------------------------------------------------------------ not at a TTY

def test_without_a_terminal_there_is_no_back(monkeypatch):
    """Piped/batch input: the controller stays out of the way and a 'b' at a
    number prompt is just an invalid number, asked again."""
    monkeypatch.setattr(menu_nav, "_FORCE_ENABLE", False)
    monkeypatch.setattr("sys.stdin", io.StringIO())        # not a TTY
    answers = ["1", "4", "b", "0.0001", "", "", "", "n"]
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda p="": answers.pop(0))
    got = opt()
    assert got["toldeg"] == 0.0001 and got["convergence"] == "Custom"
    assert answers == []
