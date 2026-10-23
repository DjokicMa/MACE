"""A workflow started from a folder of existing calculations (1.1.6).

The builder used to stop at decks whose job scripts had been edited ("would not reach N job script(s)",
then a "write them again" choice) and at decks with finished outputs ("Overwrite finished results"). A
workflow now runs every deck in its own step folder (workflow_outputs/<id>/step_NNN_<TYPE>/<material>/) and
leaves the folder as it is, so neither choice exists: the review shows no refusal, Submit sends the jobs, a
deck whose output finished already is not run again (its next step goes out instead), and every file of the
folder is byte for byte what it was.
"""
import hashlib
import shutil

import pytest

import guiharness as H
import submit_util as SU
from test_workflow_builder_api import _preview, _run, _state_changes

DATA = H.REPO_ROOT / "tests" / "data" / "existing_calcs"
NAMES = ["dia_opt", "fresh_opt", "stopped_opt"]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfscripts"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _folder(srv, name):
    d = srv.root / name
    shutil.copytree(DATA, d)
    (d / "dia_opt.f9").write_bytes(b"a few bytes standing in for a density matrix\n")
    return d


def _sums(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()}


def _draft(folder, names=NAMES):
    return {"inputs": {"type": "d12", "files": [f"{folder}/{n}.d12" for n in names]}, "label": "sh",
            "template": "opt_sp", "sequence": ["OPT", "SP"], "steps": {}, "execution": {}}


def test_a_folder_of_existing_calculations_submits_as_it_is(srv):
    d = _folder(srv, "old")
    before = _sums(d)
    srv.clear_calls()
    st, prev = _preview(srv, _draft("old"))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    assert not prev.get("refused") and not prev["needs"] and "overwrite_sh" not in prev
    assert [f["deck"] for f in prev["finished"]] == ["dia_opt.d12"]
    first = {f["material_id"]: f for f in prev["first_step"]}
    assert sorted(first) == ["dia_sp", "fresh_opt", "stopped_opt"], first
    assert first["dia_sp"]["calc_type"] == "SP" and first["fresh_opt"]["calc_type"] == "OPT"
    assert all("/workflow_outputs/" in f"/{f['script_rel']}" for f in first.values())
    assert "--ntasks=8" not in first["fresh_opt"]["script_text"]
    assert not _state_changes(srv.calls())
    assert _sums(d) == before
    st, res = _run(srv, prev["preview_id"])
    assert st == 200 and res["code"] == 0 and res["script_drift"] == [], res
    assert len(res["job_ids"]) == 3 and len(_state_changes(srv.calls())) == 3
    after = _sums(d)
    assert {k: after[k] for k in before} == before
    assert sorted(set(after) - set(before)) == [], sorted(set(after) - set(before))
    wf = d / "workflow_outputs" / res["workflow_id"]
    assert (wf / "step_001_OPT" / "fresh" / "fresh_opt.sh").is_file()
    assert (wf / "step_002_SP" / "dia_sp" / "dia_sp.sh").is_file()


def test_decks_never_run_need_no_choice_either(srv):
    d = _folder(srv, "fresh")
    for n in NAMES:
        for ext in (".out", ".sh"):
            (d / (n + ext)).unlink(missing_ok=True)
    st, prev = _preview(srv, _draft("fresh"))
    assert st == 200 and prev["code"] == 0 and not prev["needs"] and prev["finished"] == []
    assert sorted(f["material_id"] for f in prev["first_step"]) == ["dia_opt", "fresh_opt", "stopped_opt"]
