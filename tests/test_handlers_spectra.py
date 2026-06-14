"""Phase 4: IR/Raman spectra handlers (registry entries + config + render)."""
import argparse
from pathlib import Path

import numpy as np
import pytest

from conftest import find_data

from mace.plotting import handlers  # noqa: F401  (registration side-effect)
from mace.plotting.handlers import spectra as sp
from mace.plotting.registry import PlotKind, get


def test_ir_and_raman_entries_registered():
    ir = get(PlotKind.SPECTRA_IR)
    ram = get(PlotKind.SPECTRA_RAMAN)
    assert ir is not None and ir.flag == "ir" and ir.patterns
    assert ram is not None and ram.flag == "raman" and ram.patterns
    assert any("IRSPEC" in p for p in ir.patterns)
    assert any("RAMSPEC" in p for p in ram.patterns)


def test_ir_config_from_args():
    ns = argparse.Namespace(average=True, ir_column=2, format="png", raman_mode=None)
    cfg = sp._ir_config_from_args(ns)
    assert cfg["average"] is True
    assert cfg["formats"] == ["png"]


def test_raman_config_from_args_mode_and_format():
    ns = argparse.Namespace(average=False, raman_mode="total", format="svg", ir_column=2)
    cfg = sp._raman_config_from_args(ns)
    assert cfg["raman_mode"] == "total"
    assert cfg["formats"] == ["svg"]


def test_format_html_falls_back_to_png_for_spectra():
    # matplotlib spectra can't do html -> fall back to png
    ns = argparse.Namespace(average=False, ir_column=2, format="html", raman_mode=None)
    assert sp._ir_config_from_args(ns)["formats"] == ["png"]


def test_plot_ir_individual_writes_png(tmp_path):
    f = find_data("FREQ/*IRSPEC.DAT")
    cfg = sp.configure_ir_spectra(interactive=False)
    out = sp.plot_ir_spectra([str(f)], cfg, str(tmp_path))
    assert out and all(Path(p).exists() for p in out)
    assert any(p.endswith("_absorbance.png") for p in out)


def test_plot_raman_individual_writes_png(tmp_path):
    f = find_data("FREQ/*RAMSPEC.DAT")
    cfg = sp.configure_raman_spectra(interactive=False)  # default mode
    out = sp.plot_raman_spectra([str(f)], cfg, str(tmp_path))
    assert out and Path(out[0]).exists()


def test_plot_ir_averaged_groups_confs(tmp_path):
    grid = np.linspace(0, 100, 5)
    files = []
    for i, v in enumerate([1.0, 3.0]):
        p = tmp_path / f"Mat-conf{i + 1}_x_IRSPEC.DAT"
        np.savetxt(p, np.column_stack([grid, grid, np.full_like(grid, v)]))
        files.append(str(p))
    cfg = sp.configure_ir_spectra(interactive=False)
    cfg["average"] = True
    out = sp.plot_ir_spectra(files, cfg, str(tmp_path))
    pngs = [p for p in out if p.endswith(".png")]
    assert len(pngs) == 1                          # one averaged plot, not two
    assert any("Mat_IRSPEC_avg" in p for p in pngs)


# ---- interactive ----

def _feed(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, ""))


def test_interactive_raman_defaults(monkeypatch):
    _feed(monkeypatch, [])
    cfg = sp.configure_raman_spectra(interactive=True)
    assert cfg["raman_mode"] in ("total", "par_perp", "all")


def test_interactive_ir_average_choice(monkeypatch):
    # "Average conf-style configs?" -> yes
    _feed(monkeypatch, ["y"])
    cfg = sp.configure_ir_spectra(interactive=True)
    assert cfg["average"] is True
