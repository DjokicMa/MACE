"""Bug (1.1.6 preview): MACE's DOSS decks carry a "105 -1" print record (NPR > 0) before END
(Crystal_d3/CRYSTALOptToD3.py writes it; decks on disk have it). The label reader took that
record as the total's label: no Total, no fill, no spin pair, and --dos-colors total drew an
empty plot, in the GUI and in mace plotting. tests/data/spin_fe keeps the record."""
import sys
from pathlib import Path

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Plotting"))
import ipDOS_V2 as ip  # noqa: E402

FE = REPO_ROOT / "tests" / "data" / "spin_fe"


def test_fixture_keeps_the_print_record():
    assert "105 -1" in (FE / "fe_doss.d3").read_text().splitlines()


def test_reader_skips_the_print_record():
    labels = [ip.doss_label(r).strip() for r in ip.read_doss_projection_labels(str(FE / "fe_doss.d3"))]
    assert labels == ["Fe (s)", "Fe (p)", "Fe (d)", "Fe (f)", "Fe", "Total"]


def _deck(tmp_path, body):
    p = tmp_path / "x_doss.d3"
    p.write_text("NEWK\n8 8\n1 0\n" + body)
    return str(p)


def test_reader_without_an_energy_window_or_a_print_record(tmp_path):
    # INZB, IFNB >= 0: no BMI BMA record; the first projection follows the header
    f = _deck(tmp_path, "DOSS\n2 400 1 20 2 14 0\n2 1 2 #MG S\n3 3 4 5 #O P\nEND\n")
    assert ip.read_doss_projection_labels(f) == ["#MG S", "#O P", "END"]


def test_reader_with_a_window_and_several_print_records(tmp_path):
    f = _deck(tmp_path, "DOSS\n1 400 -1 -1 2 14 2\n-0.5 0.6\n2 1 2 #MG S\n105 -1\n106 0\nEND\n")
    assert ip.read_doss_projection_labels(f) == ["#MG S", "END"]


def test_gui_reader_names_the_total():
    from mace.gui import results
    d = results.dos_data(FE / "fe_doss.DOSS.DAT")
    assert [p["label"] for p in d["projections"]] == ["Fe (s)", "Fe (p)", "Fe (d)", "Fe (f)", "Fe (all)", "Total"]
