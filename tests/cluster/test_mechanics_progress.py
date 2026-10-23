"""`mace cluster` progress of elastic, EOS and QHA runs.

mode_of read these outputs as "freq" (the elastic modules, QHA: they print
FREQCALC lines) or "scf" (EOS: is_optimization_output excludes them), so a
running ELAPIEZO showed SCF or frequency progress, and without its deck
beside it an EOS never got the volume count. mechanics_output_type now comes
first, as mace.cluster.history.deck_calc_type does: the elastic modules share
ELASTCON's strain loop, EOS its volumes; QHA has no progress model and is
shown as itself, not as a FREQ.

Real HPCC outputs and decks: tests/data/hpcc_116/mechanics.
"""
import shutil
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from mace.cluster import progress as P

REPO = Path(__file__).resolve().parents[2]
MECH = REPO / "tests" / "data" / "hpcc_116" / "mechanics"
MODES = {"dia_elastcon": "elastcon", "dia_elastcon_nuchess": "elastcon", "dia_elapiezo": "elastcon",
         "dia_elapiezo_strained": "elastcon", "bn_elapiezo": "elastcon", "dia_photoela": "elastcon",
         "dia_eos": "eos", "dia_eos_strained": "eos", "dia_qha": "qha", "dia_cphf": "cphf", "dia_cphf_running": "cphf"}


@pytest.mark.parametrize("name", sorted(MODES))
def test_the_mode_is_the_runs_own(name):
    assert P.mode_of((MECH / f"{name}.out").read_text()) == MODES[name]


def test_an_eos_output_without_its_deck_counts_volumes(tmp_path):
    shutil.copy(MECH / "dia_eos_strained.out", tmp_path / "e.out")
    r = P.progress(tmp_path / "e.out")
    assert r["mode"] == "eos"
    lv = [x for x in r["levels"] if x["id"] == "eos.units"][0]
    assert lv["info"]["done"] == 1 and lv["count"]["exact"] == P.EOS_DEFAULT_POINTS


def test_an_elapiezo_run_uses_its_own_block(tmp_path):
    shutil.copy(MECH / "dia_elapiezo_strained.out", tmp_path / "p.out")
    lines = (MECH / "dia_elapiezo.d12").read_text().splitlines()
    lines.insert(lines.index("ELAPIEZO") + 1, "DEFORM")
    lines.insert(lines.index("DEFORM") + 1, "1 0 0 0 0 0")
    (tmp_path / "p.d12").write_text("\n".join(lines) + "\n")
    r = P.progress(tmp_path / "p.out")
    assert r["mode"] == "elastcon"
    lv = [x for x in r["levels"] if x["id"] == "elastcon.units"][0]
    assert lv["count"]["exact"] == 1 * 2 + 1 and lv["info"]["done"] == 1


def test_a_qha_run_is_not_a_frequency_run(tmp_path):
    shutil.copy(MECH / "dia_qha.out", tmp_path / "q.out")
    shutil.copy(MECH / "dia_qha.d12", tmp_path / "q.d12")
    r = P.progress(tmp_path / "q.out")
    assert r["mode"] == "qha"
    assert not any(str(x["id"]).startswith("freq") for x in r["levels"])
