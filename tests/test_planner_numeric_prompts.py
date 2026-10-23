"""The workflow planner's number prompts ask again on an answer that is not
a number (or is out of range) instead of raising ValueError and ending
`mace workflow`. A blank answer still gives the default, with its type.
"""
import builtins

import pytest

from mace.workflow.planner import WorkflowPlanner


@pytest.fixture
def planner(tmp_path, monkeypatch):
    p = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr(p, "_copy_required_scripts_for_expert_mode", lambda: None)
    monkeypatch.setattr(p, "_run_interactive_crystal_opt_config", lambda *a, **k: None)
    return p


def scripted(monkeypatch, answers):
    """Answer prompts in order; running out of answers fails the test."""
    it = iter(answers)
    prompts = []

    def fake(prompt=""):
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unexpected extra prompt {prompt!r}") from None

    monkeypatch.setattr(builtins, "input", fake)
    return prompts


def keyed(monkeypatch, answers):
    """Answer prompts containing a key with its answers in turn, anything
    else blank; a prompt shown more than 200 times fails the test."""
    queues = {k: list(v) for k, v in answers.items()}
    prompts = []

    def fake(prompt=""):
        prompts.append(prompt)
        if len(prompts) > 200:
            raise AssertionError("prompt loop")
        for key, queue in queues.items():
            if key in prompt and queue:
                return queue.pop(0)
        return ""

    monkeypatch.setattr(builtins, "input", fake)
    return prompts


# ------------------------------------------------------------ queue settings

def test_queue_settings_ask_again_on_bad_answers(planner, monkeypatch):
    prompts = scripted(monkeypatch, ["y",
                                     "lots", "900",      # max_jobs
                                     "-1", "0",          # reserve_slots (0 is allowed)
                                     "",                 # max_submit_batch
                                     ""])                # max_recovery_attempts
    config = planner.configure_queue_management()
    assert config["max_jobs"] == 900
    assert config["reserve_slots"] == 0
    assert isinstance(config["max_submit_batch"], int)
    assert isinstance(config["max_recovery_attempts"], int)
    assert len(prompts) == 7


def test_queue_settings_accept_zero_recovery_attempts(planner, monkeypatch):
    scripted(monkeypatch, ["y", "", "", "", "0"])
    assert planner.configure_queue_management()["max_recovery_attempts"] == 0


# ------------------------------------------------------------ properties

def test_band_nsub_not_a_number_is_asked_again(planner, monkeypatch):
    prompts = scripted(monkeypatch, ["", "", "", "ten", "500"])
    assert planner._get_advanced_d3_config("BAND")["n_points"] == 500
    assert prompts[-2] == prompts[-1] == "  Total k-points along the whole path (NSUB) [1000]: "


def test_doss_projection_out_of_range_is_asked_again_and_zero_is_kept(planner, monkeypatch):
    """The menu lists 0 (total DOS only) to 4."""
    prompts = scripted(monkeypatch, ["7", "0", "", "", "", "", "", "", ""])   # ... the resolution, the window, its
    #                                                                           bounds, a 1D material's conduction bands
    config = planner._get_advanced_d3_config("DOSS")
    assert config["projection_type"] == 0
    assert prompts[0] == prompts[1] == "  Select projection type [4]: "


def test_doss_energy_range_not_a_number_is_asked_again(planner, monkeypatch):
    scripted(monkeypatch, ["", "", "low", "-5", "5.5", "", "", "", "", ""])
    assert planner._get_advanced_d3_config("DOSS")["e_range"] == [-5.0, 5.5]


def test_transport_absolute_mu_has_no_default_and_is_asked_again(planner, monkeypatch):
    prompts = scripted(monkeypatch, ["3", "", "a", "-1", "1", "", "", "", "", ""])   # ..., the k mesh
    config = planner._get_advanced_d3_config("TRANSPORT")
    assert config["mu_range"] == [-1.0, 1.0, 0.01]
    assert prompts[1:4] == ["  Min μ absolute (eV): "] * 3


def test_transport_temperature_and_relaxation_ask_again(planner, monkeypatch):
    scripted(monkeypatch, ["1", "", "", "hot", "200", "", "0", "25", "short", "5.5", ""])   # ..., the k mesh
    config = planner._get_advanced_d3_config("TRANSPORT")
    assert config["temperature_range"] == [200, 800, 25]
    assert config["relaxation_time"] == 5.5


def test_charge_potential_grid_not_a_number_is_asked_again(planner, monkeypatch):
    keyed(monkeypatch, {"Grid points in x": ["many", "80"]})
    assert planner._get_advanced_d3_config("CHARGE+POTENTIAL")["mapnet"] == [80, 100, 100]


def test_all_blank_answers_keep_the_defaults_and_their_types(planner, monkeypatch):
    keyed(monkeypatch, {})
    band = planner._get_advanced_d3_config("BAND")
    doss = planner._get_advanced_d3_config("DOSS")
    transport = planner._get_advanced_d3_config("TRANSPORT")
    charge = planner._get_advanced_d3_config("CHARGE+POTENTIAL")
    assert band["n_points"] == 1000
    assert doss["projection_type"] == 4
    assert doss["e_range"] == [-20, 20] and all(type(v) is int for v in doss["e_range"])
    assert doss["energy_resolution"] == "standard" and doss["n_points"] == "auto"
    assert transport["mu_range"] == [-2.0, 2.0, 0.01]
    assert transport["temperature_range"] == [100, 800, 50]
    assert type(transport["relaxation_time"]) is int and transport["relaxation_time"] == 10
    assert charge["mapnet"] == [100, 100, 100]


# ------------------------------------------------------------ FREQ step

def freq(planner):
    return planner.configure_frequency_step("FREQ", 3)["frequency_settings"]


def test_scelphono_not_a_number_is_asked_again(planner, monkeypatch):
    keyed(monkeypatch, {"customization level": ["2"], "Select mode [1]": ["2"],
                        "Select supercell [1]": ["2"],
                        "Expansion in x": ["two", "0", "3"]})
    assert freq(planner)["scelphono"] == [3, 2, 2]


def test_custom_kpoint_count_is_asked_again(planner, monkeypatch):
    keyed(monkeypatch, {"customization level": ["2"], "Select mode [1]": ["2"],
                        "Select phonon type [1]": ["2"],
                        "Number of k-points": ["four", "6"]})
    assert freq(planner)["num_kpoints"] == 6


def test_cphf_settings_ask_again(planner, monkeypatch):
    keyed(monkeypatch, {"customization level": ["2"], "Select calculation type": ["3"],
                        "CPHF max iterations": ["0", "x", "40"],
                        "CPHF convergence": ["tight", "8"]})
    settings = freq(planner)
    assert settings["cphf_max_iter"] == 40 and settings["cphf_tolerance"] == 8


def test_ir_spectrum_width_and_dielectric_ask_again(planner, monkeypatch):
    keyed(monkeypatch, {"customization level": ["2"], "Select calculation type": ["2"],
                        "Skip IR spectrum plot": ["n"], "Generate IR spectrum plot": ["y"],
                        "IR peak width": ["wide", "12.5"],
                        "Dielectric constant": ["high", "4"]})
    settings = freq(planner)
    assert settings["spec_dampfac"] == 12.5
    assert settings["dielectric_constant"] == 4.0


def test_temperature_list_with_a_non_number_is_asked_again(planner, monkeypatch):
    prompts = keyed(monkeypatch, {"customization level": ["2"],
                                  "Temperatures (K)": ["300 warm", "300 400"]})
    assert freq(planner)["temperatures"] == [300.0, 400.0]
    assert sum("Temperatures (K)" in p for p in prompts) == 2


def test_freq_blank_answers_keep_the_defaults(planner, monkeypatch):
    keyed(monkeypatch, {"customization level": ["2"], "Select calculation type": ["2"],
                        "Select method [1]": ["3"],
                        "Skip IR spectrum plot": ["n"], "Generate IR spectrum plot": ["y"]})
    settings = freq(planner)
    assert settings["cphf_max_iter"] == 30 and settings["cphf_tolerance"] == 6
    assert type(settings["spec_dampfac"]) is int and settings["spec_dampfac"] == 10
    assert settings["dielectric_constant"] == 2.5
    assert settings["temperatures"] == [298.15]


def test_float_input_returns_the_default_exactly_and_needs_an_answer_without_one(
        planner, monkeypatch):
    scripted(monkeypatch, [""])
    value = planner.get_safe_float_input("x: ", 10)
    assert value == 10 and type(value) is int
    scripted(monkeypatch, ["", "nan?", "2.5"])
    assert planner.get_safe_float_input("x: ") == 2.5
    scripted(monkeypatch, ["-3", "3"])
    assert planner.get_safe_float_input("x: ", 1.0, min_val=0) == 3.0
