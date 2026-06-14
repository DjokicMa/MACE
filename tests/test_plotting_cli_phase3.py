"""Phase 3 (TODO #4): CLI surface for cube + FREQ plotting.

Adds the type-pin flags (--cube / --freq[/--vibmodes]) and their option groups to
the existing argparse surface, plus positional FILE(s) handling so
``mace plotting --cube foo.CUBE`` and ``mace plotting --diff A B`` work. Routing
is asserted through ``REGISTRY[kind].handler`` (the registry dispatch contract).

H4: --diff consumes exactly its two operands and rejects extra positionals.
H5: --all-modes is NOT in the mode group (must not clash with top-level --all);
    --list-modes pins the list-only config.
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
        calls["output_dir"] = output_dir
        return []

    monkeypatch.setattr(REGISTRY[kind], "handler", stub)
    return calls


# ---- parser characterization ----

def test_parser_accepts_cube_and_freq_flags():
    p = plotting_main.create_parser()
    assert p.parse_args(["--cube"]).cube is True
    assert p.parse_args(["--freq"]).freq is True
    assert p.parse_args(["--vibmodes"]).freq is True  # alias -> same dest


def test_cube_and_freq_are_mutually_exclusive():
    p = plotting_main.create_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--cube", "--freq"])


def test_all_modes_is_not_mutually_exclusive_with_all():
    p = plotting_main.create_parser()
    a = p.parse_args(["--freq", "--all-modes"])
    assert a.all_modes is True and a.all is False


def test_format_option_parses():
    p = plotting_main.create_parser()
    assert p.parse_args(["--cube", "--format", "png"]).format == "png"


# ---- dispatch routing ----

def test_cube_flag_with_positional_file_routes(tmp_path, monkeypatch):
    f = tmp_path / "foo_DENS.CUBE"
    f.write_text("x")
    calls = _capture(monkeypatch, PlotKind.CUBE)
    rc = plotting_main.main(["--cube", str(f), "-o", str(tmp_path)])
    assert rc == 0
    assert calls["files"] == [str(f)]


def test_cube_flag_discovers_when_no_positional(tmp_path, monkeypatch):
    (tmp_path / "a.CUBE").write_text("x")
    (tmp_path / "b.CUBE").write_text("x")
    calls = _capture(monkeypatch, PlotKind.CUBE)
    rc = plotting_main.main(["--cube", "-d", str(tmp_path), "-o", str(tmp_path)])
    assert rc == 0
    assert len(calls["files"]) == 2


def test_diff_routes_with_two_operands(tmp_path, monkeypatch):
    a, b = tmp_path / "a.CUBE", tmp_path / "b.CUBE"
    a.write_text("x")
    b.write_text("x")
    calls = _capture(monkeypatch, PlotKind.CUBE)
    rc = plotting_main.main(["--diff", str(a), str(b), "-o", str(tmp_path)])
    assert rc == 0
    assert calls["files"] == [str(a), str(b)]
    assert calls["config"]["operation"] == "diff"


def test_diff_rejects_extra_positional(tmp_path):
    a, b, c = tmp_path / "a.CUBE", tmp_path / "b.CUBE", tmp_path / "c.CUBE"
    for x in (a, b, c):
        x.write_text("x")
    with pytest.raises(SystemExit):
        plotting_main.main(["--diff", str(a), str(b), str(c)])


def test_freq_flag_routes_with_mode(tmp_path, monkeypatch):
    f = tmp_path / "m.out"
    f.write_text("x")
    calls = _capture(monkeypatch, PlotKind.FREQ)
    rc = plotting_main.main(["--freq", str(f), "--mode", "7", "-o", str(tmp_path)])
    assert rc == 0
    assert calls["files"] == [str(f)]
    assert calls["config"]["mode"] == 7


def test_freq_list_modes_pins_config(tmp_path, monkeypatch):
    f = tmp_path / "m.out"
    f.write_text("x")
    calls = _capture(monkeypatch, PlotKind.FREQ)
    rc = plotting_main.main(["--freq", str(f), "--list-modes"])
    assert rc == 0
    assert calls["config"]["list_modes"] is True


def test_density_subflag_implies_cube_and_filters(tmp_path, monkeypatch):
    (tmp_path / "x_DENS.CUBE").write_text("x")
    (tmp_path / "x_POT.CUBE").write_text("x")
    calls = _capture(monkeypatch, PlotKind.CUBE)
    rc = plotting_main.main(["--density", "-d", str(tmp_path), "-o", str(tmp_path)])
    assert rc == 0
    assert [p.rsplit("/", 1)[-1] for p in calls["files"]] == ["x_DENS.CUBE"]


def test_positional_file_no_flag_classifies(tmp_path, monkeypatch):
    f = tmp_path / "foo.CUBE"
    f.write_text("x")
    calls = _capture(monkeypatch, PlotKind.CUBE)
    rc = plotting_main.main([str(f), "-o", str(tmp_path)])
    assert rc == 0
    assert calls["files"] == [str(f)]
