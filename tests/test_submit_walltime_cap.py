"""The partition's MaxTime caps the default walltime on every `mace submit`
path, and an explicit walltime is written as given with a warning.

Where the default partition allows less than the template's 7 days, MACE
said so ("partition short allows at most 2-00:00:00; job scripts whose
default walltime is longer ask for 2-00:00:00") but only the plain path
applied it: `-i` (and --exclude-nodes, which use the same customized
generator) wrote the -t line as a literal 7-00:00:00, which the cap never
looked at. Now:

* the default, and the default accepted at the `-i` prompt, is capped;
* a walltime the user gives (--walltime, or typed at the prompt) is written
  as given, with a warning when it is longer than the partition allows.

Runs the real `mace_cli` with a fake SLURM (tests/test_site_generic_cluster.py).
"""
import re

import pytest

from test_site_generic_cluster import _cluster, _fresh  # noqa: F401 (autouse fixture)
from test_site_jobscript import _deck, _mace

CAP_WARNING = "is longer than partition short allows"


def _t(script_text):
    return re.search(r"(?m)^#SBATCH -t (\S+)", script_text).group(1)


def _submit(tmp_path, *args, stdin=""):
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", *args, "m.d12", stdin=stdin)
    assert p.returncode == 0, p.stdout + p.stderr
    return _t((d / "m.sh").read_text()), p.stdout + p.stderr


@pytest.mark.parametrize("args,stdin", [
    ([], ""),                                   # the plain path (capped before too)
    (["-i"], "n\n"),                            # -i, defaults not customized
    (["-i"], "y\n\n\n\n\nn\n"),                 # -i, every prompt left blank
    (["--exclude-nodes", "node9"], ""),         # the customized generator without -i
])
def test_default_walltime_is_capped(tmp_path, args, stdin):
    t, out = _submit(tmp_path, *args, stdin=stdin)
    assert t == "2-00:00:00", out
    assert CAP_WARNING not in out


def test_interactive_prompt_offers_the_capped_default(tmp_path):
    t, out = _submit(tmp_path, "-i", stdin="y\n\n\n\n\nn\n")
    assert "Walltime [2-00:00:00]" in out


@pytest.mark.parametrize("args,stdin", [
    (["--walltime", "5-00:00:00"], ""),
    (["-i"], "y\n\n\n5-00:00:00\n\nn\n"),
])
def test_explicit_walltime_above_the_limit_is_kept_with_a_warning(tmp_path, args, stdin):
    t, out = _submit(tmp_path, *args, stdin=stdin)
    assert t == "5-00:00:00", out
    assert CAP_WARNING in out and "2-00:00:00" in out


def test_explicit_walltime_within_the_limit_has_no_warning(tmp_path):
    t, out = _submit(tmp_path, "--walltime", "1-00:00:00")
    assert t == "1-00:00:00"
    assert CAP_WARNING not in out


def test_tracked_explicit_walltime_above_the_limit_warns(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--track", "--walltime", "5-00:00:00", "m.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    (call,) = fake.calls("sbatch")
    assert _t(call["script_text"]) == "5-00:00:00"
    assert CAP_WARNING in out
