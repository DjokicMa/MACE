"""The charged-system config keys reach the options cif2d12 and opt2d12 read, and bad values stop early."""
import pytest

from d12_config import apply_config_to_options, config_to_cif_options, validate_d12_config

BASE = {"calculation_type": "SP", "method": "DFT", "functional": "HSESOL3C", "basis_set": "SOLDEF2MSVP",
        "basis_set_type": "INTERNAL", "dimensionality": "MOLECULE",
        "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 9}}
KEYS = {"ion_charges": "auto", "net_charge": -1, "fragment_bond_scale": 1.2, "chemod": "auto",
        "site_states": [{"atom": 1, "shell_charges": [2.0, 0.0, 0.0, 0.0, 0.0]}],
        "spinlock": 1, "spinlock_cycles": 30}


def test_keys_pass_through_to_cif2d12_options():
    opts = config_to_cif_options({**BASE, **KEYS})
    for k, v in KEYS.items():
        assert opts[k] == v


def test_keys_pass_through_apply_config_to_options():
    opts = apply_config_to_options({**BASE, **KEYS}, {})
    for k, v in KEYS.items():
        assert opts[k] == v


def test_valid_keys_validate():
    ok, issues = validate_d12_config({**BASE, **KEYS, "is_3c_method": True})
    assert ok, issues


@pytest.mark.parametrize("bad,match", [
    ({"ion_charges": "yes"}, "ion_charges must be 'auto'"),
    ({"net_charge": "minus one"}, "net_charge must be a number"),
    ({"site_states": [{"atom": "1", "shell_charges": [2]}], "net_charge": 0}, "site_states must be"),
    ({"site_states": [{"atom": 1, "shell_charges": [2]}]}, "site_states needs net_charge"),
    ({"fragment_bond_scale": 0}, "fragment_bond_scale must be a positive number"),
    ({"chemod": "maybe"}, "chemod must be auto, yes or no"),
])
def test_bad_values_are_named(bad, match):
    ok, issues = validate_d12_config({**BASE, "is_3c_method": True, **bad})
    assert not ok and any(match in i for i in issues)
    with pytest.raises(ValueError, match=match):
        config_to_cif_options({**BASE, **bad})


def test_configs_without_the_keys_are_unchanged():
    assert "ion_charges" not in config_to_cif_options(dict(BASE))
    assert "net_charge" not in apply_config_to_options(dict(BASE), {})
