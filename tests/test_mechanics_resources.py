"""Job sizes of ELASTIC, EOS, QHA and CPHF decks.

A deck's type came from OPTGEOM (OPT) and FREQCALC (FREQ) only, so an
ELASTCON, ELAPIEZO, PHOTOELA, EOS, QHA or CPHF deck was an SP: the SP's
[resources.SP] table and size-aware suggestion, while the HPCC runs of
tests/data/hpcc_116/mechanics took 9 to 93 times the SP's walltime. They are
now their own types (ELASTIC for the elastic modules), with their own
[resources.<TYPE>] tables and work factors scaled from those runs.

Real decks and outputs: tests/data/hpcc_116/mechanics.
"""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT
from test_site_jobscript import _env, _mace, _toml

from mace.queue import manager as M
from mace.submission import resources_suggest as rs
from mace.submission import site
from mace.utils.calc_detection import d12_calc_type, deck_records

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
TYPES = {"dia_elastcon": "ELASTIC", "dia_elapiezo": "ELASTIC", "bn_elapiezo": "ELASTIC",
         "dia_photoela": "ELASTIC", "dia_eos": "EOS", "dia_qha": "QHA", "dia_cphf": "CPHF"}


@pytest.mark.parametrize("name", sorted(TYPES))
def test_the_deck_has_its_type(name):
    deck = MECH / f"{name}.d12"
    assert d12_calc_type(deck_records(deck.read_text())) == TYPES[name]
    assert site.deck_calc_type(deck) == TYPES[name]


def test_opt_freq_and_sp_decks_are_unchanged():
    sp = REPO_ROOT / "tests" / "data" / "hpcc_116" / "soc" / "HgTe_soc.d12"
    if not sp.exists():
        sp = next((REPO_ROOT / "tests" / "data" / "hpcc_116").rglob("*soc*.d12"))
    assert site.deck_calc_type(sp) == "SP"
    assert d12_calc_type({"OPTGEOM", "CPHF"}) == "OPT"
    assert d12_calc_type({"FREQCALC", "EOS"}) == "FREQ"


def _sized(tmp_path, name, ct=None):
    """The deck beside its own (real) output: the size CRYSTAL printed."""
    shutil.copy(MECH / f"{name}.d12", tmp_path / f"{name}.d12")
    shutil.copy(MECH / f"{name}.out", tmp_path / f"{name}.out")
    return rs.suggest_for_deck(tmp_path / f"{name}.d12", calc_type=ct, limits={})


def test_the_suggestions_follow_the_measured_ratios(tmp_path):
    sp = _sized(tmp_path, "dia_eos", "SP")
    got = {n: _sized(tmp_path, n) for n in ("dia_elastcon", "dia_eos", "dia_qha", "dia_cphf")}
    assert {n: s["calc_type"] for n, s in got.items()} == {
        "dia_elastcon": "ELASTIC", "dia_eos": "EOS", "dia_qha": "QHA", "dia_cphf": "CPHF"}
    ratio = {n: s["estimate"]["core_hours"] / sp["estimate"]["core_hours"] for n, s in got.items()}
    # HPCC core-seconds over the SP's: ELASTCON 46x / ELAPIEZO 58x, EOS 22x,
    # QHA 50x (4 displacements of diamond's one asymmetric atom), CPHF 19x
    assert ratio["dia_elastcon"] == pytest.approx(52, rel=0.02)
    assert ratio["dia_eos"] == pytest.approx(22, rel=0.02)
    assert ratio["dia_qha"] == pytest.approx(50, rel=0.02)
    assert ratio["dia_cphf"] == pytest.approx(19, rel=0.02)
    assert "3 displacements" in got["dia_qha"]["reason"]          # as a FREQ's
    for s in got.values():
        assert rs.walltime_text(0) < s["walltime"]
        assert s["walltime"] >= sp["walltime"]


def test_a_type_table_applies_to_its_decks(tmp_path):
    """[resources.ELASTIC] / [resources.QHA] are read, checked and given to
    their decks; the other decks keep [resources]."""
    s = site.load({"MACE_SITE_CONFIG": str(_write(tmp_path, "[resources.ELASTIC]\nntasks = 64\n"
                                                    'walltime = "3-00:00:00"\n[resources.QHA]\nntasks = 48\n')),
                   "MACE_SITE_GROUP_CONFIG": "none"})
    assert s.typed("scf") == ["ELASTIC", "QHA"]
    d = tmp_path / "decks"
    d.mkdir()
    for name in ("dia_elastcon", "dia_photoela", "dia_qha", "dia_eos"):
        shutil.copy(MECH / f"{name}.d12", d)
    env = {"MACE_SITE_CONFIG": str(tmp_path / "site.toml"), "MACE_SITE_GROUP_CONFIG": "none"}
    groups = site.group_decks(d, sorted(p.name for p in d.iterdir()), "scf", env)
    assert groups == {None: ["dia_eos.d12"], "ELASTIC": ["dia_elastcon.d12", "dia_photoela.d12"],
                      "QHA": ["dia_qha.d12"]}


def _write(tmp_path, text):
    p = tmp_path / "site.toml"
    p.write_text(text)
    return p


def test_mace_site_set_and_the_gui_keys_take_the_types(tmp_path):
    env = _env(tmp_path)
    _toml(tmp_path, "")
    r = _mace(tmp_path, env, "site", "set", "resources.EOS.walltime", "2-00:00:00")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[resources.EOS]" in (tmp_path / "xdg" / "mace" / "site.toml").read_text()
    from mace.gui import site_routes
    keys = site_routes.keys()
    for t in ("ELASTIC", "EOS", "QHA", "CPHF"):
        assert f"resources.{t}.walltime" in keys and f"resources.{t}.ntasks" in keys


def test_mace_site_suggest_names_the_type(tmp_path):
    env = _env(tmp_path)
    for ext in ("d12", "out"):
        shutil.copy(MECH / f"dia_photoela.{ext}", tmp_path)
    r = _mace(tmp_path, env, "site", "suggest", "dia_photoela.d12")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dia_photoela.d12  (ELASTIC)" in r.stdout


@pytest.mark.parametrize("ct", ["ELASTIC", "EOS", "QHA", "CPHF"])
def test_the_queue_manager_submits_them_with_the_scf_template(tmp_path, ct):
    shutil.copy(MECH / "dia_eos.d12", tmp_path)
    # as mace/recovery/resubmit.py drives it, without a database
    cls = M.EnhancedCrystalQueueManager
    ns = SimpleNamespace(organize_outputs=False, d12_dir=tmp_path, is_workflow_context=False)
    ns.script_paths = cls._setup_script_paths(ns)
    script = cls._get_submit_script_for_calc_type(ns, ct)
    assert script and Path(script).name == "submitcrystal23.sh"
    assert cls.determine_calc_type_from_file(ns, tmp_path / "dia_eos.d12") == "EOS"
