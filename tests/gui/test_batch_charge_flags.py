"""Batch Setup takes Calculation Setup's charged-system switches (1.1.6
parity A1): --ion-charges, --net-charge, --chemod, --fragment-bond-scale,
--spinlock and --spinlock-cycles go on every command of the batch as they
are, and each deck is byte for byte the CLI's typed by hand with the same
switches on that parent. --site-states names atoms of one structure, so a
batch refuses it. Real molecule outputs from the corpus (test/); skipped
without it."""
import json
import shlex
import shutil
from pathlib import Path

import pytest

import setup_routes as R
import submit_util as SU
from conftest import TEST_DATA

pytest.importorskip("ase")

STEMS = [f"1LiFSI-1DEC-conf{i}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized" for i in (1, 3)]
FLAGS = {"ion_charges": "auto", "chemod": "yes", "fragment_bond_scale": 1.2}
SWITCHES = ["--ion-charges", "auto", "--chemod", "yes", "--fragment-bond-scale", "1.2"]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    if not (TEST_DATA / "OPT" / f"{STEMS[0]}.out").is_file():
        pytest.skip("test/ data corpus not present")
    s = SU.Served(tmp_path_factory.mktemp("batch_charge"))
    runs = s.root / "ions"
    runs.mkdir()
    for i, stem in enumerate(STEMS):
        for ext in (".out", ".d12"):
            shutil.copy2(TEST_DATA / "OPT" / f"{stem}{ext}", runs / f"ion{i}_opt{ext}")
    yield s
    s.stop()


def _plan(s, files, calc, **kw):
    return s.task_result(s.req("api/batch/plan", {"folder": "ions", "files": files, "calc": calc, "template_answers": {},
                                                   "overrides": {}, "wait": 1, **kw}), timeout=900)


def test_batch_commands_carry_the_switches_and_match_the_cli(srv, tmp_path):
    files = ["ions/ion0_opt.out", "ions/ion1_opt.out"]
    st, r = _plan(srv, files, "SP", flags=FLAGS)
    assert st == 200 and all(f["status"] == "ok" for f in r["files"]), r
    assert all(f.get("changed_lines") for f in r["files"]), "the switches change the deck from the plain one"
    st, out = srv.task_result(srv.req("api/batch/write", {"batch_id": r["batch_id"], "name": "batch_SP_ions", "wait": 1}),
                              timeout=900)
    assert st == 200 and out["code"] == 0, out
    B = srv.root / out["folder_rel"]
    cmds = [shlex.split(c) for c in json.loads((B / "batch.json").read_text())["commands"]]
    assert len(cmds) == 2 and all(c[-len(SWITCHES):] == SWITCHES for c in cmds), cmds
    assert json.loads((B / "batch.json").read_text())["flags"] == FLAGS
    assert all(l.endswith(" ".join(SWITCHES)) for l in (B / "commands.sh").read_text().splitlines() if l.startswith("mace "))
    for f in files:
        m = tmp_path / Path(f).stem
        (m / "in").mkdir(parents=True)
        for ext in (".out", ".d12"):
            shutil.copy2(srv.root / f.replace(".out", ext), m / "in")
        res = R.run_cli_manual(["opt2d12", "--out-file", f"in/{Path(f).name}", "--calc-type", "SP", "--output-dir", "m_out",
                                *SWITCHES], R.YES, m)
        made = sorted((m / "m_out").glob("*.d12"))
        assert res.returncode == 0 and made, (res.stdout + res.stderr)[-3000:]
        mine = B / "decks" / made[0].name
        assert mine.read_text() == made[0].read_text()
        assert "CHEMOD" in mine.read_text().split()


def test_batch_refuses_site_states(srv):
    st, r = _plan(srv, ["ions/ion0_opt.out"], "SP",
                  flags={"net_charge": 0, "site_states": '[{"atom": 1, "shell_charges": [2, 2, 0, 0, 4, 0, 0]}]'})
    assert st == 400 and "names atoms of one structure" in r["error"], r


def test_batch_shows_the_clis_refusal_per_file(srv):
    st, r = _plan(srv, ["ions/ion0_opt.out"], "SP", flags={"net_charge": 1, "spinlock": 2})
    assert st == 200 and r["files"][0]["status"] == "refused", r
    assert "SPINLOCK" in r["files"][0]["message"], r
