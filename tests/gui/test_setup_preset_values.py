"""Convergence and SCF tolerance levels show what they set (1.1.6).

Marcus: "Under Convergence and SCF tolerances, in addition to the list, it should be clear what the
option values are between say standard and tight."

Each level of the SCF tolerance menu (Setup's and the workflow builder's) carries its TOLINTEG and TOLDEE,
each optimization convergence level its TOLDEG, TOLDEX, TOLDEE and MAXCYCLE, inline in the option's label
and as the row's hint for the level chosen, from d12_constants' presets (the CLI's own table). The CLI's
menus print the same values from the same table.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

import guiharness as H

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
sys.path.insert(0, str(H.REPO_ROOT / "Crystal_d12"))
from mace.gui import setup as S  # noqa: E402
import d12_constants as C  # noqa: E402


def _run(work, path, rel, mode, calc, answers=None):
    store = {}
    get, put = (lambda k: store.get(json.dumps(k, default=str)),
                lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v))
    body = {"answers": answers or {}, "patches": {}, "preview": False, "json_text": None}
    return S.run_setup(S.Request(body), mode, calc, work, path, rel, S.source_key(mode, path), get, put, None,
                       schema_info=S.schema())


@pytest.fixture(scope="module")
def outs(tmp_path_factory):
    d = tmp_path_factory.mktemp("presets")
    shutil.copytree(H.EXAMPLES / "outputs", d / "outputs")
    return d


def test_the_presets_are_described_from_d12_constants():
    for level, p in C.SCF_TOLERANCE_PRESETS.items():
        assert S.preset_values("tolerance", p["name"]) == C.describe_scf_preset(level)
    for level, p in C.OPT_CONVERGENCE_PRESETS.items():
        assert S.preset_values("opt_convergence", p["name"]) == C.describe_opt_preset(level)
    assert S.preset_values("tolerance", "Custom") is None


@pytest.mark.parametrize("calc", ["SP", "OPT", "FREQ"])
def test_each_level_shows_its_values(tmp_path, outs, calc):
    r = _run(tmp_path / "w", outs / "outputs" / "Si_sp.out", "outputs/Si_sp.out", "opt2d12", calc,
             answers={"tolerance": "Tight"})
    form = {f["id"]: f for f in r["form"]}
    opts = {o["head"]: o for o in form["tolerance"]["options"]}
    assert opts["Tight"]["values"] == "TOLINTEG: 8 8 8 9 24, TOLDEE: 9"
    assert opts["Standard"]["values"] == "TOLINTEG: 7 7 7 7 14, TOLDEE: 7"
    assert "values" not in opts["Custom"]
    assert form["tolerance"]["hint"] == "Tight: TOLINTEG: 8 8 8 9 24, TOLDEE: 9"
    if calc == "OPT":
        conv = {o["head"]: o for o in form["opt_convergence"]["options"]}
        assert conv["Tight"]["values"] == C.describe_opt_preset("2")


def test_the_cli_menus_print_the_same_values(tmp_path, outs):
    """mace opt2d12, asked for a custom SP: the SCF level menu names each level's values."""
    shutil.copy(outs / "outputs" / "Si_sp.out", tmp_path / "p.out")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "p.out",
                        "--calc-type", "SP", "--output-dir", "o"], cwd=tmp_path, input="n\n" + "\n" * 60,
                       capture_output=True, text=True, timeout=300, env=env)
    log = r.stdout + r.stderr
    for level, p in C.SCF_TOLERANCE_PRESETS.items():
        assert f"{level}. {p['name']} - {C.describe_scf_preset(level)}" in log, log[-3000:]
