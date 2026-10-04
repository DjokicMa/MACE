"""`mace submit --interactive` takes the walltimes its own prompt gives as examples.

The validator's regex wanted a digit run followed by a full HH:MM:SS, so
'12:00:00', '12:00' and '2:00:00' were all rejected (only '7-00:00:00'
passed); the next answer then landed on the following question. Applies to
the D12 (crystal.py) and D3 (properties.py) job-script writers.
"""
import builtins
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


@pytest.mark.parametrize("deck, text", [("mat_sp.d12", "mat\nCRYSTAL\nEND\n"),
                                        ("mat_band.d3", "BAND\nEND\n")])
@pytest.mark.parametrize("walltime", ["12:00:00", "2:00:00", "12:00", "1-12:00:00"])
def test_submit_interactive_takes_the_walltime(tmp_path, deck, text, walltime):
    (tmp_path / deck).write_text(text)
    # customize, cores, memory, walltime, account, node exclusion
    r = subprocess.run([sys.executable, str(MACE_CLI), "submit", deck, "--interactive",
                        "--nosubmit"], cwd=tmp_path,
                       input=f"y\n\n\n{walltime}\n\nn\n" + "\n" * 5,
                       capture_output=True, text=True, timeout=120, env=_env())
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "Please enter walltime" not in log, log[-3000:]
    script = (tmp_path / deck.replace(deck[deck.rindex("."):], ".sh")).read_text()
    assert f"#SBATCH -t {walltime}\n" in script, script[:400]


@pytest.mark.parametrize("module", ["mace.submission.crystal", "mace.submission.properties"])
@pytest.mark.parametrize("bad", ["12", "abc", "12:0", "1-2", "12:00:00:00"])
def test_malformed_walltimes_are_still_refused(monkeypatch, module, bad):
    import importlib
    mod = importlib.import_module(module)
    replies = iter([bad, ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))
    assert mod.get_safe_walltime_input("Walltime: ", "7-00:00:00") == "7-00:00:00"
