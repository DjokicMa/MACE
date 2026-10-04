"""A slab's band path does not tell the user to install seekpath.

The seekpath library is skipped for a slab on purpose (it would standardize
the vacuum axis into the cell; BAND takes the slab's third coordinates as
zero, manual p.310 note 3), so "Install seekpath" was advice that changed
nothing. A 3D cell without the library stops (seekpath is required) unless
the static path was asked for, and then keeps the hint.
"""
import pytest

import d3_kpoints


def _out(tmp_path, slab):
    path = tmp_path / ("slab.out" if slab else "bulk.out")
    path.write_text(" SLAB CALCULATION\n" if slab else " CRYSTAL CALCULATION\n")
    return str(path)


def _path(tmp_path, capsys, slab):
    segments, info = d3_kpoints.get_seekpath_full_kpath(25, "P", _out(tmp_path, slab))
    return segments, info, capsys.readouterr().out


def test_slab_with_seekpath_installed_gets_no_install_hint(tmp_path, capsys):
    pytest.importorskip("seekpath")
    assert d3_kpoints.SEEKPATH_LIBRARY_AVAILABLE
    segments, info, out = _path(tmp_path, capsys, slab=True)
    assert "Install" not in out
    assert "slab" in out
    assert info["in_plane"] is True and segments


def test_slab_without_seekpath_gets_no_install_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    segments, info, out = _path(tmp_path, capsys, slab=True)
    assert "Install" not in out
    assert "slab" in out


def test_bulk_without_seekpath_stops(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    monkeypatch.delenv("MACE_STATIC_KPATH", raising=False)
    with pytest.raises(d3_kpoints.SeekpathRequired, match="pip install seekpath"):
        _path(tmp_path, capsys, slab=False)


def test_bulk_static_path_asked_for_still_gets_the_install_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    monkeypatch.setenv("MACE_STATIC_KPATH", "1")  # the static route, asked for
    _, _, out = _path(tmp_path, capsys, slab=False)
    assert "Install 'seekpath' library for accurate parametric k-points." in out
    assert "Install the 'seekpath' library for a path in CRYSTAL's cell." in out


def test_messages_do_not_change_the_slab_path(tmp_path, capsys, monkeypatch):
    with_lib = _path(tmp_path, capsys, slab=True)[0]
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    assert _path(tmp_path, capsys, slab=True)[0] == with_lib
