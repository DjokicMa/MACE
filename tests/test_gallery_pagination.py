"""The Plot Gallery pages a set larger than one page (it stopped at the first 200 panes): the same files
page by page, a page size, the global numbering, the CLI's --page and --max."""
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.gui import gallery as GG
from mace.plotting import gallery as G


@pytest.fixture()
def folder(tmp_path):
    for i in range(25):
        (tmp_path / f"m{i:02d}.cif").write_text("data_x\n")
    return tmp_path


def names(got):
    return [p.name for p in got["items"]]


def test_pages_cover_the_set_once(folder):
    seen = []
    for page in (1, 2, 3):
        got = G.collect([folder], "structure", cap=10, page=page)
        assert got["total"] == 25 and got["pages"] == 3 and got["page"] == page and got["page_size"] == 10
        assert got["offset"] == (page - 1) * 10
        seen += names(got)
    assert seen == [f"m{i:02d}.cif" for i in range(25)]
    last = G.collect([folder], "structure", cap=10, page=3)
    assert len(last["items"]) == 5 and last["message"] == "Showing files 21-25 of 25 (page 3 of 3)."
    first = G.collect([folder], "structure", cap=10, page=1)
    assert first["message"].startswith("Showing the first 10 of 25 files") and first["capped"]


def test_a_page_past_the_end_is_the_last_and_a_small_set_is_one_page(folder):
    assert G.collect([folder], "structure", cap=10, page=99)["page"] == 3
    one = G.collect([folder], "structure", cap=50)
    assert one["pages"] == 1 and len(one["items"]) == 25 and one["message"] is None and not one["capped"]


def test_captions_number_from_the_page_offset(folder):
    got = G.collect([folder], "structure", cap=10, page=2)
    caps = G.captions(got["items"], start=got["offset"] + 1)
    assert [r["n"] for r in caps["rows"]] == list(range(11, 21))


def test_the_gui_request_checks_page_and_size():
    assert GG._page_args({}) == (G.MAX_ITEMS, 1)
    assert GG._page_args({"page_size": 100, "page": 3}) == (100, 3)
    for bad in ({"page_size": -1}, {"page_size": 501}, {"page": -2}, {"page": "x"}):
        with pytest.raises(ValueError):
            GG._page_args(bad)
    assert GG.PAGE_MAX == 500 and 200 in GG.PAGE_SIZES


def test_the_copied_command_names_the_page():
    cmd = G.command("band", ["d"], 3, "o.svg", page=3, page_size=50)
    assert "--max" in cmd and cmd[cmd.index("--max") + 1] == "50" and cmd[cmd.index("--page") + 1] == "3"
    plain = G.command("band", ["d"], 3, "o.svg")
    assert "--max" not in plain and "--page" not in plain


def test_cli_draws_the_second_page(folder, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "g.svg"
    # structures with no atoms draw as an unreadable pane, which is fine: the page and its numbering are the point
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--gallery", "structure", str(folder),
                        "--max", "10", "--page", "3", "-o", str(out)], capture_output=True, text=True, timeout=300)
    assert "Showing files 21-25 of 25 (page 3 of 3)" in r.stdout, r.stdout + r.stderr
    assert "21." in r.stdout or "m20.cif" in r.stdout


def test_a_later_page_numbers_its_panes_from_the_page_offset(tmp_path):
    """The figure's pane titles go on from the page before (21., 22., ...), as its captions: page 3 said 1., 2."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "g.svg"
    panes = [{"title": f"m{i}", "panels": [], "error": "x"} for i in range(2)]
    G.render_figure(panes, out, cols=2, start=21)
    text = out.read_text()
    assert "21. m0" in text and "22. m1" in text and "1. m0" not in text.replace("21. m0", "")
