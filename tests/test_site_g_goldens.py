"""Job scripts with no new (PLAN_G) site key are byte-identical to the ones
MACE wrote before those keys existed (A1, A2, A8).

The goldens in tests/data/site_g_golden.json were captured by running these
very scenarios against the code before PLAN_G (the
polish/generic head), through the real mace_cli: the -i editor with blank
answers and --walltime (whose lines the editor writes itself), a site.toml
with account and partition, another university's cluster (fake SLURM and
modules) and the owner's cluster. Paths are written as @TMP@ (the test
folder), @REPO@ (this install) and @PY@ (the Python running the tests).

Regenerate (only for an intended change, reviewed by hand):
    MACE_UPDATE_SITE_G_GOLDEN=1 pytest tests/test_site_g_goldens.py
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import SITE_TOML, _deck, _env, _mace, _toml

GOLDEN = Path(__file__).resolve().parent / "data" / "site_g_golden.json"
UPDATE = os.environ.get("MACE_UPDATE_SITE_G_GOLDEN") == "1"
BLANK_I = "y\n\n\n\n\nn\n"


def _norm(text, tmp):
    text = (text.replace(str(tmp), "@TMP@").replace(str(REPO_ROOT), "@REPO@")
            .replace(os.path.abspath(sys.executable), "@PY@"))
    # the module-discovery comment names the machine the test runs on
    return re.sub(r"does not load on \S+; using", "does not load on @HOST@; using", text)


def _cluster_env(tmp, owner):
    from test_site_generic_cluster import _cluster
    return _cluster(tmp, owner=owner)[0]


def _run(tmp, env, name, args, stdin=""):
    d = tmp / "decks"
    deck = "m.d12" if name.endswith(".d12") else "m_band.d3"
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / "m_band.f9").write_text("fort9\n")
    before = (d / deck).read_bytes()
    p = _mace(d, env, "submit", "--nosubmit", *args, deck, stdin=stdin)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert (d / deck).read_bytes() == before                     # the deck is never touched
    return _norm((d / (Path(deck).stem + ".sh")).read_text(), tmp)


SCENARIOS = {
    "workstation_walltime_site_scf": ("ws", SITE_TOML, "m.d12", ["--walltime", "2:00:00"], ""),
    "workstation_walltime_site_prop": ("ws", SITE_TOML, "m.d3", ["--walltime", "1:00:00"], ""),
    "workstation_interactive_blank_scf": ("ws", None, "m.d12", ["-i"], BLANK_I),
    "workstation_interactive_blank_prop": ("ws", None, "m.d3", ["-i"], BLANK_I),
    "workstation_interactive_site_scf": ("ws", SITE_TOML, "m.d12", ["-i"], BLANK_I),
    "other_cluster_scf": ("other", None, "m.d12", [], ""),
    "other_cluster_prop": ("other", None, "m.d3", [], ""),
    "other_cluster_walltime_scf": ("other", None, "m.d12", ["--walltime", "1:00:00"], ""),
    "owner_cluster_walltime_scf": ("owner", None, "m.d12", ["--walltime", "1:00:00"], ""),
    "owner_cluster_interactive_prop": ("owner", None, "m.d3", ["-i"], BLANK_I),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scripts_match_the_pre_g_goldens(name, tmp_path):
    where, toml, deck, args, stdin = SCENARIOS[name]
    env = _env(tmp_path) if where == "ws" else _cluster_env(tmp_path, owner=(where == "owner"))
    if toml:
        _toml(tmp_path, toml)
    got = _run(tmp_path, env, deck, args, stdin)
    if UPDATE:
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[name] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(GOLDEN.read_text())[name]
