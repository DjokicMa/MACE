"""FREQ "Custom settings": the IR/Raman spectrum questions reach the deck.

After "Generate IR/Raman spectrum plot?" the custom route read
``minimal_ir``/``minimal_raman`` with True defaults, so its spectrum questions
(range, step, line shape, width, RAMANEXP) were never asked, and a Yes never
set ``irspec``/``ramspec``: the deck had no IRSPEC/RAMSPEC block although the
user asked for a plot. Each answer must now become the CRYSTAL23 keyword the
manual names (sec. 8.6 IRSPEC p. 228, sec. 8.7 RAMSPEC p. 230, RAMANEXP
p. 218), and the template routes must write the same FREQCALC blocks as before.
"""
import io

import pytest

import menu_nav
import d12_calc_freq
from d12_calc_freq import crystal_system_with_lattice, write_frequency_section

# MgO (Fm-3m) as cif2d12 passes it
CUBIC = (crystal_system_with_lattice(225), 225)


class Answers:
    """Answer a prompt by a piece of its text; anything not listed gets Enter."""

    def __init__(self, scripted):
        self.scripted = dict(scripted)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if len(self.prompts) > 300:
            raise AssertionError("menu keeps asking")
        for key, answer in self.scripted.items():
            if key in prompt:
                return answer
        return ""

    def asked(self, piece):
        return any(piece in p for p in self.prompts)


@pytest.fixture
def answers(monkeypatch):
    def use(scripted):
        a = Answers(scripted)
        monkeypatch.setattr(menu_nav, "_REAL_INPUT", a)
        monkeypatch.setattr("builtins.input", a)
        return a
    return use


def _freqcalc(settings):
    """The FREQCALC block ``settings`` write, FREQCALC through ENDFREQ."""
    buf = io.StringIO()
    write_frequency_section(buf, dict(settings), *CUBIC)
    lines = [ln.strip() for ln in buf.getvalue().splitlines()]
    return lines[lines.index("FREQCALC"):lines.index("ENDFREQ") + 1]


def _block(lines, name):
    """The lines of the IRSPEC/RAMSPEC block, its keyword through its END."""
    i = lines.index(name)
    return lines[i:lines.index("END", i) + 1]


CUSTOM = {"Select template (1-8)": "8"}
IR = {**CUSTOM, "Calculate IR intensities?": "y"}
RAMAN = {**CUSTOM, "Calculate Raman intensities?": "y"}


# ------------------------------------------------------------ custom IR: the plot is written
def test_custom_ir_plot_yes_writes_irspec(answers):
    answers(IR)
    lines = _freqcalc(d12_calc_freq.get_advanced_frequency_settings())
    # every answer left at its default: CRYSTAL's defaults, so a bare block
    assert _block(lines, "IRSPEC") == ["IRSPEC", "END"]
    assert "RAMSPEC" not in lines


def test_custom_ir_spectrum_answers_become_irspec_keywords(answers):
    a = answers({**IR, "Customize spectral range?": "y", "Minimum frequency": "100",
                 "Maximum frequency": "3000", "Spectrum step size": "2",
                 "Select broadening (1-2)": "2", "FWHM": "12"})
    block = _block(_freqcalc(d12_calc_freq.get_advanced_frequency_settings()), "IRSPEC")
    assert block == ["IRSPEC", "RANGE", "100.0 3000.0", "LENSTEP", "2",
                     "DAMPFAC", "12", "GAUSS", "END"]
    # no Raman: no Raman question
    assert not a.asked("RAMANEXP")


def test_custom_ir_plot_no_writes_no_irspec(answers):
    a = answers({**IR, "Generate IR spectrum plot?": "n"})
    lines = _freqcalc(d12_calc_freq.get_advanced_frequency_settings())
    assert "IRSPEC" not in lines
    assert not a.asked("Customize spectral range?")


# ------------------------------------------------------------ custom Raman
def test_custom_raman_answers_become_ramspec_and_ramanexp(answers):
    answers({**RAMAN, "Customize spectral range?": "y", "Minimum frequency": "100",
             "Maximum frequency": "3000", "Select broadening (1-2)": "2", "FWHM": "12",
             "Use RAMANEXP": "y", "Temperature (K)": "300", "Laser wavelength (nm)": "785"})
    settings = d12_calc_freq.get_advanced_frequency_settings()
    lines = _freqcalc(settings)
    # RAMANEXP: T, then the laser wavelength in nm (manual p. 218)
    i = lines.index("RAMANEXP")
    assert lines[i + 1].split() == ["300.0", "785.0"]
    # Gaussian is VOIGT 0 for the pseudo-Voigt Raman line shape; DAMPFAC is its FWHM
    assert _block(lines, "RAMSPEC") == ["RAMSPEC", "RANGE", "100.0 3000.0", "VOIGT", "0.0",
                                        "DAMPFAC", "12", "END"]
    # the Raman temperature is RAMANEXP's: no THERMO TEMP from it
    assert "temperature" not in settings
    assert "THERMO" not in lines


def test_raman_temperature_does_not_reach_thermo(answers):
    answers({**RAMAN, "Use RAMANEXP": "y", "Temperature (K)": "300",
             "Calculate thermodynamic properties?": "y"})
    lines = _freqcalc(d12_calc_freq.get_advanced_frequency_settings())
    thermo = _block(lines, "THERMO")
    assert "TEMP" not in thermo, thermo


def test_custom_ir_and_raman_write_both_spectra(answers):
    a = answers({**IR, "Calculate Raman intensities?": "y"})
    lines = _freqcalc(d12_calc_freq.get_advanced_frequency_settings())
    assert a.asked("Generate IR spectrum plot?") and a.asked("Generate Raman spectrum plot?")
    assert _block(lines, "IRSPEC") == ["IRSPEC", "END"]
    assert _block(lines, "RAMSPEC") == ["RAMSPEC", "END"]


def test_questions_with_no_crystal_input_are_gone(answers):
    a = answers({**RAMAN, "Use RAMANEXP": "y"})
    d12_calc_freq.get_advanced_frequency_settings()
    for gone in ("Temperature for Raman", "Wavelength [", "Select units (1-2)",
                 "Select output format (1-3)", "Resolution for spectrum"):
        assert not a.asked(gone), gone


# ------------------------------------------------------------ golden: the template routes are unchanged
GOLDEN = {
    # template menu answer: the FREQCALC block every other question at its default writes
    "1": ["FREQCALC", "NOINTENS", "ENDFREQ"],
    "2": ["FREQCALC", "INTENS", "INTCPHF", "ENDCPHF", "IRSPEC", "RANGE", "0 4000", "END", "ENDFREQ"],
    "3": ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "ENDCPHF", "RAMSPEC", "RANGE", "0 4000", "END",
          "ENDFREQ"],
    "4": ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "ENDCPHF", "IRSPEC", "RANGE", "0 4000", "END",
          "RAMSPEC", "RANGE", "0 4000", "END", "ENDFREQ"],
    "5": ["FREQCALC", "TEMPERAT", "20 0 400", "NOINTENS", "THERMO", "END", "ENDFREQ"],
    "6": ["FREQCALC", "NOINTENS", "DISPERSION", "BANDS", "0 100", "4", "X GAMMA", "GAMMA L", "L W",
          "W GAMMA", "ENDFREQ"],
    "7": ["FREQCALC", "NOINTENS", "DISPERSION", "PDOS", "2000.0 200", "1", "ENDFREQ"],
    # custom settings, no intensities
    "8": ["FREQCALC", "NOINTENS", "ENDFREQ"],
}


@pytest.mark.parametrize("choice", sorted(GOLDEN))
def test_template_freqcalc_blocks_are_unchanged(answers, choice):
    answers({"Select template (1-8)": choice})
    assert _freqcalc(d12_calc_freq.get_advanced_frequency_settings()) == GOLDEN[choice]


def test_custom_without_intensities_asks_no_spectrum_question(answers):
    a = answers(CUSTOM)
    d12_calc_freq.get_advanced_frequency_settings()
    assert not a.asked("spectrum plot?")
