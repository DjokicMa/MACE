"""opt2d3 BAND asks "Edit the band k-path?" only at a terminal (or with --edit-kpath).

1.1.6 added the question after the BAND settings. Asked on piped answers it
took one line meant for the next question, so every later answer of a 1.1.5
script moved by one (and a "y" meant for a later question opened the k-path
editor). Piped answers now get 1.1.5's questions, in 1.1.5's order;
--edit-kpath asks it on a pipe too (the GUI's Calculation Setup has its own
k-path editor and never needs it).
"""
import json
import re
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "examples" / "gui" / "outputs" / "Si_sp.out"
QUESTION = "Edit the band k-path"


def _run(cwd, args, stdin):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "opt2d3", *args], cwd=cwd, input=stdin,
                          capture_output=True, text=True, timeout=300)


def _outputs(d, names):
    d.mkdir()
    for n in names:
        shutil.copy(SI, d / f"{n}_sp.out")
        (d / f"{n}_sp.f9").touch()


def _npoints(deck):
    return deck.read_text().splitlines()[2].split()[2]


def test_piped_batch_answers_keep_their_1_1_5_order(tmp_path):
    d = tmp_path / "in"
    _outputs(d, ["a", "b"])
    # 1.1.5's answers: no shared settings; the first file: every default; the second: 500 k-points
    stdin = "n\n" + "\n" * 5 + "\n\n500\n\n\n" + "\n" * 10
    r = _run(d, ["--batch", "-t", "BAND"], stdin)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert QUESTION not in log, log[-3000:]
    first, second = re.findall(r"Input file: .*[/\\]([ab])_sp\.out", log)
    assert _npoints(d / f"{first}_band.d3") == "1000" and _npoints(d / f"{second}_band.d3") == "500", log[-3000:]


def test_piped_single_file_is_not_asked(tmp_path):
    d = tmp_path / "in"
    _outputs(d, ["s"])
    r = _run(d, ["-i", "s_sp.out", "-t", "BAND"], "\n" * 20)
    log = r.stdout + r.stderr
    assert r.returncode == 0 and QUESTION not in log, log[-3000:]
    assert (d / "s_band.d3").exists()


def test_edit_kpath_asks_on_a_pipe(tmp_path):
    d = tmp_path / "in"
    _outputs(d, ["s"])
    stdin = "\n" * 5 + "y\nr 1\n\n" + "\n" * 10
    r = _run(d, ["-i", "s_sp.out", "-t", "BAND", "--edit-kpath", "--save-config", "--options-file", "cfg.json"], stdin)
    log = r.stdout + r.stderr
    assert r.returncode == 0 and QUESTION in log, log[-3000:]
    cfg = json.loads((d / "cfg.json").read_text())
    assert cfg.get("configuration", cfg).get("band_path"), log[-3000:]
    assert (d / "s_band.d3").read_text().splitlines()[1].endswith("GAMMA-X-|-GAMMA-L-W-GAMMA"), log[-3000:]
