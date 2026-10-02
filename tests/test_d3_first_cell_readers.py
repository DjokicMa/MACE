"""The d3 helpers read the cell and group of the run, not the first cell printed.

* The custom-labels menu (BAND option 3) listed the CRYSTAL-supported labels
  for crystal_system / lattice_type read from extract_lattice_parameters_
  from_output, which returns only the cell: every output was shown the cubic
  P labels. It now reads the output's space group.
* extract_and_process_shrink's last fallback (no SHRINK in the parent deck
  or output) took the first LATTICE PARAMETERS printed, the input's cell; it
  now takes the computed block fort.9 holds (a SUPERCEL supercell, say).
Real outputs: the AgBr R -3 C OPT and the phonopy Si F D 3 M parent in the
repo; the supercell case uses the synthetic layout of crystal_out_text.
"""
import contextlib
import io
from pathlib import Path

import pytest

from conftest import REPO_ROOT

import crystal_out_text as cot

d3i = pytest.importorskip("d3_interactive")
d3k = pytest.importorskip("d3_kpoints")


@pytest.mark.parametrize("out,expected", [
    (REPO_ROOT / "tests/data/opt_restart/tqc_agbr.out", ("TRIGONAL", "R")),
    (REPO_ROOT / "tests/fixtures/phonopy_crystal/Si/parent.out", ("CUBIC", "F")),
])
def test_family_and_centring_from_the_output(out, expected):
    assert d3i.crystal_family_and_centring(str(out)) == expected
    labels = d3k.get_crystal_supported_labels(*expected)
    assert labels == d3k.CRYSTAL_KPOINT_LABELS[
        {"TRIGONAL": "Rhombohedral (R Trigonal)", "CUBIC": "FC Cubic"}[expected[0]]]


def test_custom_label_menu_lists_the_outputs_labels(monkeypatch):
    out = REPO_ROOT / "tests/data/opt_restart/tqc_agbr.out"
    answers = iter(["3", "GAMMA T"])
    monkeypatch.setattr(d3i, "_nav_read", lambda *a, **k: next(answers, ""))
    import menu_nav
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda *a, **k: next(answers, ""))
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers, ""))
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        config = d3i.configure_band_calculation(str(out))
    assert "labels for TRIGONAL (R centering)" in printed.getvalue()
    assert config["path"] == ["GAMMA", "T"]


def test_fallback_shrink_uses_the_computed_cell(tmp_path):
    out = tmp_path / "super.out"
    big = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    out.write_text(cot.echo("P M 3 M", (2.0, 2.0, 2.0, 90, 90, 90))
                   + cot.supercel([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
                   + cot.block(big, [(20, "CA", 0, 0, 0)], centring="1/1"))
    with contextlib.redirect_stdout(io.StringIO()):
        shrink = d3k.extract_and_process_shrink(str(out), "super", tmp_path, {})
    assert shrink == 4          # 60 / 20 -> 3, made even; the 2 A input cell gave 30
