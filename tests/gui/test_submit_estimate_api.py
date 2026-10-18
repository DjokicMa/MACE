"""The Submit dialog's runtime estimate (1.1.6 parity A3): `mace cluster
estimate DECK --cores N` for a deck about to be submitted, its own words (the
CLI's estimate_text of the same history.estimate) and the command that
prints them. Advisory: a deck with no history says it cannot estimate, and
nothing about submitting changes. Real finished runs from the corpus
(test/) for the history; that test skips without it."""
import os
import shutil
import subprocess
import sys

import pytest

import submit_util as SU
from conftest import TEST_DATA

RUNS = ["OPT/1_dia_opt_rev1", "OPT/2_dia2_opt_rev1", "OPT/3_dia3_opt_rev1"]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("estimate"))
    s.decks("lone", ["x"])
    yield s
    s.stop()


def _est(s, rel, cores=None):
    body = {"deck_rel": rel}
    if cores:
        body["cores"] = cores
    st, _, r = s.req("api/cluster/estimate", body)
    return st, r


def _twin(s, rel, cores):
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH", "COLORTERM", "TERM", "NO_COLOR")}
    env.update(s.env)
    return subprocess.run([sys.executable, str(SU.MACE_CLI), "--no-banner", "cluster", "estimate", rel, "--cores", str(cores)],
                          cwd=str(s.root), env=env, capture_output=True, text=True, timeout=300)


def test_no_history_cannot_estimate_in_the_clis_words(srv):
    st, r = _est(srv, "lone/x.d12", 16)
    assert st == 200 and not r.get("wall"), r
    assert r["command"] == "mace cluster estimate lone/x.d12 --cores 16"
    assert r["text"] and r["text"] == _twin(srv, "lone/x.d12", 16).stdout.rstrip("\n")
    # without a core count (no job script yet): the CLI's reason, not an error
    st, r = _est(srv, "lone/x.d12")
    assert st == 200 and "core count is unknown" in r["text"], r
    assert r["command"] == "mace cluster estimate lone/x.d12"


def test_not_a_deck_is_refused(srv):
    (srv.root / "lone" / "y.d3").write_text("NEWK\nEND\n")
    st, r = _est(srv, "lone/y.d3", 4)
    assert st == 400 and ".d12 deck" in r["error"], r


def test_history_estimate_is_the_clis(srv):
    if not (TEST_DATA / "OPT").is_dir():
        pytest.skip("test/ data corpus not present")
    d = srv.root / "hist"
    d.mkdir()
    for p in RUNS:
        for ext in (".out", ".d12"):
            shutil.copy2(TEST_DATA / f"{p}{ext}", d)
    shutil.copy2(TEST_DATA / f"{RUNS[0]}.d12", d / "next_opt.d12")
    st, r = _est(srv, "hist/next_opt.d12", 8)
    assert st == 200, r
    assert r["text"] == _twin(srv, "hist/next_opt.d12", 8).stdout.rstrip("\n")
    assert r["command"] == "mace cluster estimate hist/next_opt.d12 --cores 8"
