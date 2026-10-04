"""opt2d3 BAND path method 4 with all segments of one kind writes that path.

Method 4 ("Fractional coordinates"), answered "no" to mixed segments, stopped
with UnboundLocalError ('Path') before it asked for the segments when all
were coordinates. Past that, the segments of both all-coordinates and
all-labels were stored where the BAND writer of a manual path does not look,
so the deck would have been an empty BAND (0 segments). Both now write the
segments given, as the mixed route does.
"""
import os
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = REPO_ROOT / "tests" / "data" / "label_band" / "lih_p1bar.out"


def _opt2d3_band(tmp_path, answers):
    shutil.copy(PARENT, tmp_path / "p.out")
    (tmp_path / "p.f9").write_bytes(b"x")
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d3", "--input", "p.out",
                        "--calc-type", "BAND"], cwd=str(tmp_path), env=env,
                       input="\n".join(answers) + "\n" * 10, capture_output=True,
                       text=True, timeout=300)
    log = r.stdout + r.stderr
    assert "Traceback" not in log, log[-3000:]
    assert r.returncode == 0, log[-3000:]
    deck = tmp_path / "p_band.d3"
    assert deck.exists(), log[-3000:]
    return deck.read_text().splitlines()


def _band_block(lines):
    i = lines.index("BAND")
    return lines[i + 2:lines.index("END", i)]


def test_all_fractional_coordinates_write_the_scaled_path(tmp_path):
    # method 4, not mixed, all coordinates, 2 segments, shrink 8
    lines = _opt2d3_band(tmp_path, ["4", "n", "2", "2", "8",
                                    "0 0 0 0.5 0 0", "0.5 0 0 0.5 0.5 0"])
    header, *segments = _band_block(lines)
    nline, shrink = header.split()[:2]
    assert (nline, shrink) == ("2", "8"), lines
    assert [s.split() for s in segments] == [["0", "0", "0", "4", "0", "0"],
                                             ["4", "0", "0", "4", "4", "0"]], lines


def test_shrink_raised_for_a_point_it_cannot_reach_applies_to_every_segment(tmp_path):
    # shrink 2 cannot hold 1/4: the scaling raises it, and the header says so
    lines = _opt2d3_band(tmp_path, ["4", "n", "2", "2", "2",
                                    "0 0 0 0.25 0 0", "0.25 0 0 0.5 0 0"])
    header, *segments = _band_block(lines)
    shrink = int(header.split()[1])
    assert shrink % 4 == 0, lines
    q = shrink // 4
    assert [s.split() for s in segments] == [
        ["0", "0", "0", str(q), "0", "0"], [str(q), "0", "0", str(2 * q), "0", "0"]], lines


def test_all_labels_write_the_labels(tmp_path):
    lines = _opt2d3_band(tmp_path, ["4", "n", "1", "2", "G X", "X Y"])
    header, *segments = _band_block(lines)
    assert header.split()[:2] == ["2", "0"], lines
    assert segments == ["G X", "X Y"], lines
