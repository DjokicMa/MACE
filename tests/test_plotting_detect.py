"""Phase 0: registry-driven file discovery for mace plotting.

``detect.discover()`` must reproduce the legacy ``discover_plottable_files``
categorization exactly (same glob patterns, dedup, sort) while being driven by
the registry rather than hard-coded keys.
"""
from pathlib import Path

import pytest

from mace.plotting import detect, handlers  # noqa: F401  (handlers import = registration)
from mace.plotting.registry import PlotKind


def _make(tmp_path, names):
    for n in names:
        (tmp_path / n).write_text("x")


def test_discover_categorizes_band_dos_structure(tmp_path):
    _make(tmp_path, ["foo.BAND.DAT", "bar_doss.DOSS.DAT", "baz.cif", "ignore.txt"])
    got = detect.discover(str(tmp_path))
    assert [Path(f).name for f in got[PlotKind.BAND]] == ["foo.BAND.DAT"]
    assert [Path(f).name for f in got[PlotKind.DOS]] == ["bar_doss.DOSS.DAT"]
    assert [Path(f).name for f in got[PlotKind.STRUCTURE]] == ["baz.cif"]


def test_discover_dedups_and_sorts(tmp_path):
    _make(tmp_path, ["b.BAND.DAT", "a.BAND.DAT"])
    names = [Path(f).name for f in detect.discover(str(tmp_path))[PlotKind.BAND]]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_discover_matches_legacy_glob_patterns(tmp_path):
    _make(tmp_path, ["m.band.band.dat", "n.DOSS.DAT", "o.cif", "p_band.dat"])
    got = detect.discover(str(tmp_path))
    assert {Path(f).name for f in got[PlotKind.BAND]} == {"m.band.band.dat", "p_band.dat"}
    assert {Path(f).name for f in got[PlotKind.DOS]} == {"n.DOSS.DAT"}
    assert {Path(f).name for f in got[PlotKind.STRUCTURE]} == {"o.cif"}


def test_classify_file_maps_name_to_kind():
    assert detect.classify_file("x.BAND.DAT") == PlotKind.BAND
    assert detect.classify_file("y_doss.DOSS.DAT") == PlotKind.DOS
    assert detect.classify_file("z.cif") == PlotKind.STRUCTURE
    assert detect.classify_file("readme.txt") is None


def test_legacy_kinds_are_registered():
    # importing handlers must populate the registry with the three legacy kinds
    from mace.plotting.registry import REGISTRY
    assert {PlotKind.BAND, PlotKind.DOS, PlotKind.STRUCTURE} <= set(REGISTRY)
