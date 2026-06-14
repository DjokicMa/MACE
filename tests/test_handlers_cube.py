"""Phase 3 (TODO #3): the cube plotting handler.

The handler drives the relocated ``crystal_cubeviz_plotly`` engine via a
constructed argv + an explicit ``--save`` path (the same headless pattern the
legacy band/DOS handlers use), so engines *write* files rather than ``.show()``.

Fast unit tests cover registration, config mapping and the L4 engine-args screen;
one real-render integration test (against the gitignored cube corpus, skipped if
absent) proves a cube actually renders to HTML.
"""
import argparse
from pathlib import Path

import pytest

from conftest import find_data

from mace.plotting import handlers  # noqa: F401  (import side-effect: registration)
from mace.plotting.handlers import cube as cube_h
from mace.plotting.registry import PlotKind, get


def test_cube_entry_registered():
    e = get(PlotKind.CUBE)
    assert e is not None
    assert e.flag == "cube"
    assert e.patterns and any("cube" in p.lower() for p in e.patterns)
    assert callable(e.handler) and callable(e.configure)


def test_config_from_args_maps_view_iso_colorscale():
    ns = argparse.Namespace(
        view="slice", iso="0.001,0.01", colorscale="Viridis", alpha=0.5,
        no_atoms=True, bonds=False, publication=False, log=False, linear=False,
        clip=99.5, slice=["z", "50"], slice_all=None, format="html",
        engine_args=None, density=False, esp=False, spin=False, diff=None,
    )
    cfg = cube_h._cube_config_from_args(ns)
    assert cfg["view"] == "slice"
    assert cfg["iso"] == [0.001, 0.01]
    assert cfg["colorscale"] == "Viridis"
    assert cfg["show_atoms"] is False
    assert cfg["slice_axis"] == "z"
    assert cfg["slice_pos"] == 50


def test_engine_args_screen_rejects_output_flags():
    kept, rejected = cube_h._screen_engine_args(
        ["--camera-preset", "top", "--save", "x.html", "--save-png", "y.png"]
    )
    assert "--camera-preset" in kept and "top" in kept
    assert "--save" in rejected and "--save-png" in rejected
    assert "--save" not in kept and "--save-png" not in kept


def test_plot_cube_writes_html_for_real_cube(tmp_path):
    cube = find_data("ECH3POT3/*.CUBE")
    cfg = cube_h.configure_cube_plot(interactive=False)
    cfg["view"] = "iso"
    cfg["iso"] = [0.01]
    cfg["show_atoms"] = False
    out = cube_h.plot_cube([str(cube)], cfg, str(tmp_path))
    assert out, "handler returned no generated files"
    assert all(Path(p).exists() for p in out)
    assert any(p.endswith(".html") for p in out)


# ---- interactive configuration flows ----

def _feed(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, ""))


def test_interactive_cube_all_defaults_is_iso_html(monkeypatch):
    # view, auto-iso, atoms, bonds, format, advanced -> all defaults
    _feed(monkeypatch, [])
    cfg = cube_h.configure_cube_plot(interactive=True)
    assert cfg["view"] == "iso"
    assert cfg["iso"] is None          # auto-select
    assert cfg["show_atoms"] is True
    assert cfg["formats"] == ["html"]
    assert cfg["operation"] is None


def test_interactive_cube_single_slice_uses_auto_middle(monkeypatch):
    # view=2 (slice), axis default z, "use best/middle slice?" yes -> slice_pos None
    _feed(monkeypatch, ["2"])
    cfg = cube_h.configure_cube_plot(interactive=True)
    assert cfg["view"] == "slice"
    assert cfg["slice_axis"] == "z"
    assert cfg["slice_pos"] is None


def test_interactive_cube_slice_auto_renders_a_real_cube(tmp_path):
    # The auto/middle slice path must produce a file (engine resolves the index).
    cube = find_data("ECH3POT3/*.CUBE")
    cfg = cube_h.configure_cube_plot(interactive=False)
    cfg["view"] = "slice"
    cfg["slice_axis"] = "z"
    cfg["slice_pos"] = None
    cfg["show_atoms"] = False
    out = cube_h.plot_cube([str(cube)], cfg, str(tmp_path))
    assert out and all(Path(p).exists() for p in out)
