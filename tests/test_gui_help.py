"""`mace gui` help: complete, routed the same in every position, and true.

The curated text in mace_cli lists every option build_parser() accepts; `-h`
prints it before or after other arguments; neither the curated text nor
argparse's own help claims a one-time token or a standard-library-only GUI
(structures need ase/spglib, plots plotly, from requirements.txt)."""
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM")}
    env.update(PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    return subprocess.run([sys.executable, str(MACE_CLI), *args], capture_output=True, text=True, env=env,
                          timeout=120)


def _options():
    from mace.gui.cli import build_parser
    out = set()
    for action in build_parser()._actions:
        out.update(o for o in action.option_strings if o.startswith("--") and o != "--help")
    return out


def test_every_option_is_in_the_curated_help():
    text = _run("gui", "--help").stdout
    missing = sorted(o for o in _options() if not re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])", text))
    assert not missing, missing
    assert "docs/gui.md" in text


@pytest.mark.parametrize("args", [("gui", "-h"), ("--no-banner", "gui", "-h"), ("gui", "x", "-h"),
                                  ("gui", "--dir", ".", "--help")])
def test_help_routes_the_same_in_every_position(args):
    r = _run(*args)
    assert r.returncode == 0 and "Usage: mace gui" in r.stdout, (args, r.stdout[:300], r.stderr[-500:])


def test_help_makes_no_false_claims():
    from mace.gui.cli import build_parser
    for text in (_run("gui", "--help").stdout, build_parser().format_help()):
        assert "standard library only" not in text
        assert "session token" in text
        assert "(demo)" not in text and "gui_demo" not in text


def test_parser_documents_the_same_things_as_the_curated_text():
    from mace.gui.cli import build_parser
    help_text = " ".join(build_parser().format_help().split())
    assert "ase, spglib and plotly" in help_text and "docs/gui.md" in help_text
    assert "MACE_GUI_TOKEN" in help_text and "MACE_GUI_PORT" in help_text and "--insecure-network" in help_text


def test_gpu_help_names_gpu_0_as_the_default():
    """Polish A7: GPU 0 is the default; the old "GPU0 drives the display, N >= 1" rule is gone."""
    from mace.gui.cli import build_parser
    curated = " ".join(_run("gui", "--help").stdout.split())
    parser = " ".join(build_parser().format_help().split())
    for text in (curated, parser):
        assert "default: GPU 0" in text, text
        assert "N >= 1; GPU0 drives the display" not in text and "(N >= 1)" not in text


def _serve_briefly(tmp_path, *extra):
    """Start `mace gui` on an empty folder, read until the URL line, stop it: (stdout+stderr text)."""
    import signal
    import time
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM",
                                                              "MACE_GUI_TOKEN", "MACE_GUI_PORT")}
    env.update(PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1", XDG_CONFIG_HOME=str(tmp_path / "xdg"),
               XDG_CACHE_HOME=str(tmp_path / "xdg" / "cache"))
    (tmp_path / "d").mkdir()
    p = subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path / "d"),
                          "--port", "0", *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, stdin=subprocess.DEVNULL)
    try:
        end = time.time() + 60
        out = []
        while time.time() < end:
            line = p.stdout.readline()
            if not line:
                break
            out.append(line)
            if "/s/" in line and "http" in line:
                break
        time.sleep(0.3)
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            so, se = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            so, se = p.communicate()
    return "".join(out) + so, se


def test_plain_run_says_nothing_about_gpu(tmp_path):
    out, err = _serve_briefly(tmp_path)
    assert "--gpu" not in err and "--gpu" not in out, err


def test_gpu_without_open_still_warns(tmp_path):
    out, err = _serve_briefly(tmp_path, "--gpu", "1")
    assert "--gpu applies with --open" in err, err


def test_help_lists_every_gui_tool():
    """`mace gui connect`, `login-url` and `diagnostics` are all named in `mace gui --help`."""
    from mace.gui.cli import TOOLS
    out = _run("gui", "--help").stdout
    for tool in TOOLS:
        assert f"mace gui {tool} " in out, tool
