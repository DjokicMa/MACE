"""opt2d12 interactive: "Use Fermi smearing? y" on a parent without smearing.

The parent's settings carry smearing_width None, and the width menu computed
abs(None - 0.01): a TypeError traceback (in 1.1.5 too)."""
import builtins

import pytest


@pytest.mark.parametrize("width", [None, 0, 0.0, ""])
def test_smearing_yes_on_a_parent_without_smearing(monkeypatch, width):
    import d12_interactive as D

    monkeypatch.setattr(D, "yes_no_prompt", lambda *a, **k: True)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    cfg = D.configure_smearing_with_defaults({"smearing": False, "smearing_width": width})
    assert cfg["use_smearing"] is True
    assert cfg["smearing_width"] == pytest.approx(0.01)


def test_a_parent_width_is_kept_as_the_custom_default(monkeypatch):
    import d12_interactive as D

    monkeypatch.setattr(D, "yes_no_prompt", lambda *a, **k: True)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
    cfg = D.configure_smearing_with_defaults({"smearing": True, "smearing_width": 0.005})
    assert cfg["smearing_width"] == pytest.approx(0.005)
