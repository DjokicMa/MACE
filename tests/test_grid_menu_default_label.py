"""The opt2d12 grid menu marks the parent's grid as the default, not XLGRID.

The menu always labelled "4: XLGRID - Extra large grid (default)", while a
blank answer keeps the parent's grid (DEFAULT, LGRID, ...). The label now
sits on the grid a blank answer gives; cif2d12's default stays XLGRID.
"""
import builtins
import re

import pytest

import d12_constants
import d12_interactive


def _menu(monkeypatch, capsys, ask):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    import menu_nav
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda prompt="": "")
    grid = ask()
    out = capsys.readouterr().out
    labelled = [l for l in out.splitlines() if re.match(r"\d: \w+GRID|\d: DEFAULT", l)
                and "(default)" in l]
    return grid, labelled


@pytest.mark.parametrize("parent, label", [("DEFAULT", "2: DEFAULT"), ("LGRID", "3: LGRID"),
                                           ("XXLGRID", "5: XXLGRID")])
def test_opt2d12_labels_the_parents_grid(monkeypatch, capsys, parent, label):
    grid, labelled = _menu(monkeypatch, capsys,
                           lambda: d12_interactive.configure_dft_grid_with_defaults("PBE0", parent))
    assert grid == parent
    assert len(labelled) == 1 and labelled[0].startswith(label), labelled


def test_cif2d12_default_is_still_xlgrid(monkeypatch, capsys):
    grid, labelled = _menu(monkeypatch, capsys, lambda: d12_constants.configure_dft_grid("PBE0"))
    assert grid == "XLGRID"
    assert len(labelled) == 1 and labelled[0].startswith("4: XLGRID"), labelled
