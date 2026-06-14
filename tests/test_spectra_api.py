"""Phase 4: pure spectra read/render/average API.

Lifted from plotIRRAM.py (the verified superset of the 5 standalone IR/Raman
scripts — their read_* bodies are byte-identical, so they collapse to one
read_spectrum). Real corpus: 132 IRSPEC.DAT (3-col) + 132 RAMSPEC.DAT (10-col)
in test/FREQ. Tests skip if the gitignored corpus is absent.
"""
from pathlib import Path

import numpy as np
import pytest

from conftest import find_data

from mace.plotting.handlers import spectra_api as api


def test_read_spectrum_ir_is_3col():
    d = api.read_spectrum(str(find_data("FREQ/*IRSPEC.DAT")))
    assert d.ndim == 2 and d.shape[1] == 3 and d.shape[0] > 0


def test_read_spectrum_raman_is_10col():
    d = api.read_spectrum(str(find_data("FREQ/*RAMSPEC.DAT")))
    assert d.shape[1] == 10


def test_read_spectrum_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "x.DAT"
    f.write_text("# header line\n\n1.0 2.0 3.0\n4.0 5.0 6.0\n")
    d = api.read_spectrum(str(f))
    assert d.shape == (2, 3)


def test_get_material_name():
    assert api.get_material_name("Foo-Bar-conf3_blah_IRSPEC.DAT") == "Foo-Bar"
    assert api.get_material_name("noconf_IRSPEC.DAT") is None


def test_group_files_by_material():
    g = api.group_files_by_material(
        ["A-conf1_x.DAT", "A-conf2_x.DAT", "B-conf1_y.DAT", "noconf.DAT"]
    )
    assert set(g) == {"A", "B"}
    assert len(g["A"]) == 2 and len(g["B"]) == 1


def test_render_ir_writes_png(tmp_path):
    d = api.read_spectrum(str(find_data("FREQ/*IRSPEC.DAT")))
    out = api.render_ir(d, str(tmp_path / "ir"), formats=("png",))
    assert out and all(Path(p).exists() for p in out)
    assert out[0].endswith("_absorbance.png")


def test_render_ir_multiple_formats(tmp_path):
    d = api.read_spectrum(str(find_data("FREQ/*IRSPEC.DAT")))
    out = api.render_ir(d, str(tmp_path / "ir"), formats=("png", "svg"))
    exts = {Path(p).suffix for p in out}
    assert {".png", ".svg"} <= exts


@pytest.mark.parametrize("mode,suffix", [("total", "_total"),
                                         ("par_perp", "_par_perp"),
                                         ("all", "_all")])
def test_render_raman_modes_write_files(tmp_path, mode, suffix):
    d = api.read_spectrum(str(find_data("FREQ/*RAMSPEC.DAT")))
    out = api.render_raman(d, str(tmp_path / f"r_{mode}"), mode=mode, formats=("png",))
    assert out and Path(out[0]).exists()
    assert out[0].endswith(suffix + ".png")


def test_render_raman_skips_when_too_few_columns(tmp_path):
    # a 2-col array can do 'total' but not 'all'
    arr = np.column_stack([np.linspace(0, 100, 5), np.ones(5)])
    assert api.render_raman(arr, str(tmp_path / "r"), mode="all") == []
    assert api.render_raman(arr, str(tmp_path / "r"), mode="total")


def test_average_spectrum_ir(tmp_path):
    grid = np.linspace(0, 100, 5)
    files = []
    for i, val in enumerate([2.0, 4.0]):
        p = tmp_path / f"m-conf{i + 1}_IRSPEC.DAT"
        np.savetxt(p, np.column_stack([grid, grid, np.full_like(grid, val)]))
        files.append(str(p))
    avg, n = api.average_spectrum(files, min_cols=3)
    assert n == 2
    assert np.allclose(avg[:, 2], 3.0)        # (2 + 4) / 2
    assert np.allclose(avg[:, 0], grid)       # wavenumber preserved


def test_average_spectrum_skips_mismatched_grid(tmp_path):
    np.savetxt(tmp_path / "m-conf1_IRSPEC.DAT",
               np.column_stack([np.linspace(0, 100, 5)] * 3))
    np.savetxt(tmp_path / "m-conf2_IRSPEC.DAT",
               np.column_stack([np.linspace(0, 100, 7)] * 3))  # different length
    files = [str(tmp_path / "m-conf1_IRSPEC.DAT"), str(tmp_path / "m-conf2_IRSPEC.DAT")]
    avg, n = api.average_spectrum(files, min_cols=3)
    assert n == 1   # second skipped, no ragged crash
