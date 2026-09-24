"""The plan's explicit overrides for a numbered step reach the generated deck.

- The planner nests FREQ tolerances in ``frequency_settings.custom_tolerances``,
  and nothing read that key, so a FREQ deck kept the parent's TOLINTEG/TOLDEE
  although the plan promised tighter ones. This applied to the engine and to
  ``mace opt2d12 --config-file`` alike.
- The engine wrote an OPT2 plan's ``custom_tolerances`` as ``tolinteg`` /
  ``scf_toldee`` config keys, which CRYSTALOptToD12 never reads.
- An SP plan's ``tolerance_modifications``, ``basis_modifications`` and
  ``method_modifications.custom_functional`` were never forwarded.

A partial override (only TOLDEE) keeps the parent's TOLINTEG.
"""
import pytest

import test_numbered_calc_keeps_parent_method as base

FREQ_TOL = {"TOLINTEG": "12 12 12 12 24", "TOLDEE": 12}
FREQ_STEP = {
    "calculation_type": "FREQ",
    "inherit_base_settings": True,
    "frequency_settings": {"mode": "GAMMA", "numderiv": 2, "intensities": True,
                           "ir_method": "CPHF", "raman": True,
                           "custom_tolerances": FREQ_TOL},
}
PBE0 = base.DECK.format(ham="DFT\nSPIN\nPBE0\nENDDFT")


def _cfg(tmp_path, steps, target, deck=PBE0):
    return base._numbered(base._engine(tmp_path, deck, "dummy\n", steps), target)["config"]


# ---------------------------------------------------------------------------
# What the engine passes (no corpus needed; the script is not run)
# ---------------------------------------------------------------------------
def test_opt2_plan_tolerances_use_the_key_crystaloptto_d12_reads(tmp_path):
    step = {"calculation_type": "OPT", "optimization_type": "FULLOPTG",
            "optimization_settings": {"TOLDEG": 3e-05}, "inherit_settings": True,
            "custom_tolerances": {"TOLDEE": 9}}
    cfg = _cfg(tmp_path, {"OPT2_4": step}, "OPT2")
    assert cfg["tolerance_modifications"] == {"custom_tolerances": {"TOLDEE": 9}}
    assert "tolinteg" not in cfg and "scf_toldee" not in cfg


def test_sp_plan_modifications_are_forwarded(tmp_path):
    step = {"calculation_type": "SP", "inherit_settings": False,
            "method_modifications": {"custom_functional": "HSESOL3C"},
            "basis_modifications": {"new_basis": "POB-TZVP-REV2"},
            "tolerance_modifications": {"custom_tolerances":
                                        {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}}}
    cfg = _cfg(tmp_path, {"SP_2": step}, "SP")
    assert cfg["method_modifications"] == {"functional": "HSESOL3C"}
    assert cfg["tolerance_modifications"]["custom_tolerances"]["TOLDEE"] == 11
    assert cfg["basis_set"] == "POB-TZVP-REV2"


def test_sp_plan_new_functional_keeps_its_d3_suffix(tmp_path):
    """new_functional goes through the plan-override builder (with -D3); it must
    not also be forwarded as a method_modification that drops the suffix."""
    step = {"calculation_type": "SP",
            "method_modifications": {"new_functional": "PBE0", "use_dispersion": True}}
    cfg = _cfg(tmp_path, {"SP_2": step}, "SP")
    assert cfg["functional"] == "PBE0-D3"
    assert "method_modifications" not in cfg


def test_plan_without_tolerances_passes_none(tmp_path):
    cfg = _cfg(tmp_path, {"SP_2": {"calculation_type": "SP", "inherit_settings": True}}, "SP")
    assert "tolerance_modifications" not in cfg


# ---------------------------------------------------------------------------
# Real decks, real script (corpus-gated)
# ---------------------------------------------------------------------------
def _real(tmp_path, steps, target, ham=None):
    base._need(base.DIA)
    if ham is None:
        d12 = base.DIA.with_suffix(".d12").read_text()
        out = base.DIA.with_suffix(".out").read_text()
    else:
        d12, out = base._dia_with(ham)
    eng = base._engine(tmp_path, d12, out, steps)
    name, deck = base._only_deck(base._numbered(eng, target, True))
    return name, deck.splitlines()


def _after(body, key):
    return body[body.index(key) + 1].strip()


def test_real_freq_deck_gets_the_plan_tolerances_and_keeps_the_parent_method(tmp_path):
    name, body = _real(tmp_path, {"FREQ_3": FREQ_STEP}, "FREQ")
    assert _after(body, "TOLINTEG") == "12 12 12 12 24"
    assert _after(body, "TOLDEE") == "12"
    assert "B3LYP-D3" in body and "NUMDERIV" in body and "INTRAMAN" in body


def test_real_freq_plan_functional_switch_keeps_freq_settings_and_tolerances(tmp_path):
    step = dict(FREQ_STEP, method_modifications={"new_functional": "PBE0",
                                                 "use_dispersion": True})
    name, body = _real(tmp_path, {"FREQ_3": step}, "FREQ",
                       ham=["DFT", "SPIN", "PBESOL0", "XLGRID", "ENDDFT"])
    assert "PBE0-D3" in body and "PBESOL0" not in body
    assert "NUMDERIV" in body and "INTRAMAN" in body and "NOINTENS" not in body
    assert _after(body, "TOLINTEG") == "12 12 12 12 24"
    assert _after(body, "TOLDEE") == "12"


def test_real_freq_without_plan_gets_the_freq_default_tolerances(tmp_path):
    # FREQ's SCF default is Very tight, not the parent optimization's 7 7 7 7 14
    name, body = _real(tmp_path, None, "FREQ")
    assert _after(body, "TOLINTEG") == "9 9 9 11 38"
    assert _after(body, "TOLDEE") == "11"


def test_real_opt2_partial_tolerance_override_keeps_parent_tolinteg(tmp_path):
    step = {"calculation_type": "OPT", "optimization_type": "FULLOPTG",
            "optimization_settings": {"TOLDEG": 3e-05, "TOLDEX": 0.00012,
                                      "TOLDEE": 7, "MAXCYCLE": 800},
            "inherit_settings": True, "custom_tolerances": {"TOLDEE": 9}}
    name, body = _real(tmp_path, {"OPT2_4": step}, "OPT2")
    parent = base.DIA.with_suffix(".d12").read_text().splitlines()
    scf = body[body.index("END", body.index("ENDOPT")):]  # past OPTGEOM
    assert _after(scf, "TOLDEE") == "9"
    assert _after(scf, "TOLINTEG") == _after(parent, "TOLINTEG")


def test_real_sp_plan_tight_tolerances(tmp_path):
    step = {"calculation_type": "SP", "inherit_settings": False,
            "method_modifications": {"inherit_functional": True},
            "basis_modifications": {"inherit_basis": True},
            "tolerance_modifications": {"custom_tolerances":
                                        {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}}}
    name, body = _real(tmp_path, {"SP_2": step}, "SP")
    assert _after(body, "TOLINTEG") == "9 9 9 11 38"
    assert _after(body, "TOLDEE") == "11"
    assert "B3LYP-D3" in body and "OPTGEOM" not in body


def test_real_sp_plan_custom_3c_functional_gets_its_basis(tmp_path):
    step = {"calculation_type": "SP", "inherit_settings": False,
            "method_modifications": {"custom_functional": "HSESOL3C"},
            "basis_modifications": {"inherit_basis": True}}
    name, body = _real(tmp_path, {"SP_2": step}, "SP")
    assert "HSESOL3C" in body and "B3LYP-D3" not in body
    assert _after(body, "BASISSET") == "SOLDEF2MSVP"
    assert "_sp_HSESOL3C_" in name


@pytest.mark.parametrize("mods,written", [
    ({"custom_functional": "PBE0"}, "PBE0"),
    ({"custom_functional": "PBE0-D3"}, "PBE0-D3"),
    ({"new_functional": "PBE0"}, "PBE0"),
    ({"new_functional": "PBE0", "use_dispersion": True}, "PBE0-D3"),
])
def test_real_sp_plan_functional_is_written_exactly_as_planned(tmp_path, mods, written):
    """The planner appends -D3 only when the user asked for it. The B3LYP-D3
    parent's dispersion flag used to survive the switch, so a planned PBE0
    came out as PBE0-D3."""
    step = {"calculation_type": "SP", "inherit_settings": False,
            "method_modifications": mods,
            "basis_modifications": {"inherit_basis": True}}
    name, body = _real(tmp_path, {"SP_2": step}, "SP")
    dft = body[body.index("DFT") + 1:body.index("ENDDFT")]
    assert dft == ["SPIN", written, "XLGRID"], dft
    assert f"_sp_{written}_optimized" in name, name
