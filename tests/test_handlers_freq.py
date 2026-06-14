"""Phase 3 (TODO #3): the FREQ vibrational-mode plotting handler.

Drives the relocated ``engines.vibmode_viewer`` engine headlessly (always passes
``--mode``/``--all``/``--list`` + ``--save``/``--gif`` so it never drops into the
engine's interactive ``input()`` loop or ``fig.show()``). Each file renders into
a per-file subdirectory so multi-file batches don't collide on the engine's fixed
output names.

Real-corpus integration tests use a molecular FREQ ``.out`` (small, fast); they
skip if the gitignored corpus is absent.
"""
import argparse
from pathlib import Path

import pytest

from conftest import find_data

from mace.plotting import detect, handlers  # noqa: F401  (registration side-effect)
from mace.plotting.handlers import freq as freq_h
from mace.plotting.registry import PlotKind, get


def test_freq_entry_registered():
    e = get(PlotKind.FREQ)
    assert e is not None
    assert e.flag == "freq"
    # FREQ must content-sniff: .out is shared with hundreds of non-FREQ runs.
    assert e.sniff is detect.is_freq_output
    assert e.patterns and any(p.lower().endswith(".out") for p in e.patterns)


def test_freq_config_from_args_maps_fields():
    ns = argparse.Namespace(
        mode=7, all_modes=False, list_modes=False, compare=False,
        amplitude=2.0, gif=True, gif_fps=10, static=False, normalize=True,
        frames=None, format="html",
    )
    cfg = freq_h._freq_config_from_args(ns)
    assert cfg["mode"] == 7
    assert cfg["gif"] is True
    assert cfg["amplitude"] == 2.0
    assert cfg["normalize"] is True


def test_representative_mode_is_a_real_mode():
    f = find_data("FREQ/*.out", "NORMAL MODES NORMALIZED")
    n = freq_h._representative_mode(str(f))
    assert isinstance(n, int) and n > 0


def test_plot_freq_list_modes_returns_no_files(tmp_path, capsys):
    f = find_data("FREQ/*.out", "NORMAL MODES NORMALIZED")
    cfg = freq_h.configure_freq_plot(interactive=False)
    cfg["list_modes"] = True
    out = freq_h.plot_freq([str(f)], cfg, str(tmp_path))
    assert out == []
    # the engine prints a mode table
    assert "Freq" in capsys.readouterr().out


def test_plot_freq_writes_html_for_representative_mode(tmp_path):
    # A molecular FREQ run is small -> fast animation render.
    f = find_data("FREQ/*MOLECULE*.out", "NORMAL MODES NORMALIZED")
    cfg = freq_h.configure_freq_plot(interactive=False)
    cfg["frames"] = 4  # keep the render quick
    out = freq_h.plot_freq([str(f)], cfg, str(tmp_path))
    assert out, "handler returned no generated files"
    assert all(Path(p).exists() for p in out)
    assert any(p.endswith(".html") for p in out)


# ---- interactive configuration flows ----

def _feed(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, ""))


def test_interactive_freq_defaults_single_representative_html(monkeypatch):
    # action default (render one), representative yes, output style default (html), advanced no
    _feed(monkeypatch, [])
    cfg = freq_h.configure_freq_plot(interactive=True)
    assert cfg["list_modes"] is False
    assert cfg["all_modes"] is False
    assert cfg["mode"] is None      # representative
    assert cfg["gif"] is False and cfg["static"] is False


def test_interactive_freq_list_only_short_circuits(monkeypatch):
    _feed(monkeypatch, ["2"])       # "List the mode table only"
    cfg = freq_h.configure_freq_plot(interactive=True)
    assert cfg["list_modes"] is True


def test_interactive_freq_gif_output_style(monkeypatch):
    # render one (default), representative yes, output style 2 (GIF), advanced no
    _feed(monkeypatch, ["", "", "2"])
    cfg = freq_h.configure_freq_plot(interactive=True)
    assert cfg["gif"] is True and cfg["static"] is False
