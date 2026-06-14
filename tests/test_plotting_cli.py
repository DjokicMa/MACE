"""Phase 0: mace plotting CLI must dispatch through the registry.

These pin the observable behavior of ``main()`` so the registry refactor is
provably behavior-preserving: the same flags route to the same plotter, the
same "no files" paths return 1, and bare invocation still drops to interactive.
The routing assertions go through ``REGISTRY[kind].handler`` (the new
contract), so they fail against the pre-refactor direct-call ``main()``.
"""
import pytest

import importlib

from mace.plotting import handlers  # noqa: F401  (registration)

# The package re-exports the ``main`` *function*, which shadows the ``main``
# submodule as an attribute, so import the module object explicitly.
plotting_main = importlib.import_module('mace.plotting.main')
from mace.plotting.registry import REGISTRY, PlotKind


# ---- parser characterization (stable across the refactor) ----

def test_parser_accepts_legacy_mode_flags():
    p = plotting_main.create_parser()
    assert p.parse_args(['--band']).band is True
    assert p.parse_args(['--dos']).dos is True
    assert p.parse_args(['--structure']).structure is True
    assert p.parse_args(['--all']).all is True
    assert p.parse_args(['-i']).interactive is True


def test_parser_mode_flags_are_mutually_exclusive():
    p = plotting_main.create_parser()
    with pytest.raises(SystemExit):
        p.parse_args(['--band', '--dos'])


def test_parser_keeps_option_defaults():
    p = plotting_main.create_parser()
    a = p.parse_args([])
    assert a.directory == '.'
    assert a.output == '.'
    assert a.projection == 'both'
    assert a.supercell == [2, 2, 2]


# ---- dispatch routing (new registry contract) ----

def _capture(monkeypatch, kind):
    calls = {}

    def stub(files, config, output_dir):
        calls['files'] = list(files)
        calls['config'] = config
        calls['output_dir'] = output_dir
        return []

    monkeypatch.setattr(REGISTRY[kind], 'handler', stub)
    return calls


def test_band_flag_routes_to_band_handler(tmp_path, monkeypatch):
    (tmp_path / "foo.BAND.DAT").write_text("x")
    calls = _capture(monkeypatch, PlotKind.BAND)
    rc = plotting_main.main(['--band', '-d', str(tmp_path), '-o', str(tmp_path)])
    assert rc == 0
    assert [p.rsplit('/', 1)[-1] for p in calls['files']] == ["foo.BAND.DAT"]


def test_band_flag_returns_1_when_no_band_files(tmp_path):
    rc = plotting_main.main(['--band', '-d', str(tmp_path)])
    assert rc == 1


def test_all_flag_routes_to_every_present_handler(tmp_path, monkeypatch):
    (tmp_path / "a.BAND.DAT").write_text("x")
    (tmp_path / "b.DOSS.DAT").write_text("x")
    (tmp_path / "c.cif").write_text("x")
    band = _capture(monkeypatch, PlotKind.BAND)
    dos = _capture(monkeypatch, PlotKind.DOS)
    struct = _capture(monkeypatch, PlotKind.STRUCTURE)
    rc = plotting_main.main(['--all', '-d', str(tmp_path), '-o', str(tmp_path)])
    assert rc == 0
    assert band['files'] and dos['files'] and struct['files']


def test_no_args_runs_interactive(monkeypatch):
    called = {}
    monkeypatch.setattr(plotting_main, 'run_interactive',
                        lambda directory='.': called.setdefault('dir', directory))
    rc = plotting_main.main([])
    assert rc == 0
    assert 'dir' in called


def test_band_single_mode_passes_cli_energy_range(tmp_path, monkeypatch):
    (tmp_path / "foo.BAND.DAT").write_text("x")
    calls = _capture(monkeypatch, PlotKind.BAND)
    plotting_main.main(['--band', '-d', str(tmp_path), '-o', str(tmp_path),
                        '--e-lower', '-8', '--e-upper', '8'])
    assert calls['config']['e_lower'] == -8.0
    assert calls['config']['e_upper'] == 8.0
