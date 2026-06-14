"""Phase 4: CLI surface for IR/Raman spectra.

--ir / --raman pin a single spectra kind; --spectra is an umbrella that plots
both. Spectra options: --raman-mode, --average, --ir-column. Routing asserted
through REGISTRY[kind].handler.
"""
import importlib

import pytest

from mace.plotting import handlers  # noqa: F401  (registration)

plotting_main = importlib.import_module("mace.plotting.main")
from mace.plotting.registry import REGISTRY, PlotKind


def _capture(monkeypatch, kind):
    calls = {}

    def stub(files, config, output_dir):
        calls["files"] = list(files)
        calls["config"] = config
        return []

    monkeypatch.setattr(REGISTRY[kind], "handler", stub)
    return calls


def test_parser_accepts_spectra_flags():
    p = plotting_main.create_parser()
    assert p.parse_args(["--ir"]).ir is True
    assert p.parse_args(["--raman"]).raman is True
    assert p.parse_args(["--spectra"]).spectra is True


def test_spectra_flags_mutually_exclusive_with_cube():
    p = plotting_main.create_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--ir", "--cube"])


def test_raman_mode_and_average_parse():
    p = plotting_main.create_parser()
    a = p.parse_args(["--raman", "--raman-mode", "total", "--average"])
    assert a.raman_mode == "total"
    assert a.average is True


def test_ir_flag_routes_to_ir_handler(tmp_path, monkeypatch):
    (tmp_path / "x_IRSPEC.DAT").write_text("1 2 3\n")
    calls = _capture(monkeypatch, PlotKind.SPECTRA_IR)
    rc = plotting_main.main(["--ir", "-d", str(tmp_path), "-o", str(tmp_path)])
    assert rc == 0
    assert [p.rsplit("/", 1)[-1] for p in calls["files"]] == ["x_IRSPEC.DAT"]


def test_raman_flag_routes_with_mode(tmp_path, monkeypatch):
    (tmp_path / "x_RAMSPEC.DAT").write_text("1 2 3\n")
    calls = _capture(monkeypatch, PlotKind.SPECTRA_RAMAN)
    rc = plotting_main.main(["--raman", "--raman-mode", "total",
                             "-d", str(tmp_path), "-o", str(tmp_path)])
    assert rc == 0
    assert calls["config"]["raman_mode"] == "total"


def test_spectra_umbrella_routes_both(tmp_path, monkeypatch):
    (tmp_path / "a_IRSPEC.DAT").write_text("1 2 3\n")
    (tmp_path / "b_RAMSPEC.DAT").write_text("1 2 3\n")
    ir = _capture(monkeypatch, PlotKind.SPECTRA_IR)
    ram = _capture(monkeypatch, PlotKind.SPECTRA_RAMAN)
    rc = plotting_main.main(["--spectra", "-d", str(tmp_path), "-o", str(tmp_path)])
    assert rc == 0
    assert ir["files"] and ram["files"]


def test_spectra_umbrella_returns_1_when_none(tmp_path):
    rc = plotting_main.main(["--spectra", "-d", str(tmp_path)])
    assert rc == 1


def test_ir_positional_classifies(tmp_path, monkeypatch):
    f = tmp_path / "foo_IRSPEC.DAT"
    f.write_text("1 2 3\n")
    calls = _capture(monkeypatch, PlotKind.SPECTRA_IR)
    rc = plotting_main.main([str(f), "-o", str(tmp_path)])
    assert rc == 0
    assert calls["files"] == [str(f)]
