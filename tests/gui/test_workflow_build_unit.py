"""The workflow builder's pure pieces (mace/gui/workflow_build.py): the
draft -> spec translation, the partition time-limit check, the one-launch-
at-a-time lock, the job-script normalization the preview == run check uses
and the double-submission guard's database read."""
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from mace.gui import submit as sub
from mace.gui import workflow_build as wb


def _inputs(tmp_path, names=("A", "B"), kind="cif"):
    d = tmp_path / "cifs"
    d.mkdir(exist_ok=True)
    files = []
    for n in names:
        f = d / f"{n}.{kind}"
        f.write_text("data_x\n")
        files.append(f)
    return wb.Inputs(kind, d, files, list(names))


def _spec(files):
    return json.loads(files["spec.json"])


def test_spec_from_a_template_draft(tmp_path):
    inp = _inputs(tmp_path)
    spec_dir = inp.folder / wb.SPEC_DIR / "x"
    files = wb.spec_files(inp, {"template": "full_electronic", "sequence": ["OPT", "SP", "BAND", "DOSS"],
                                "label": "perov", "steps": {}}, spec_dir)
    spec = _spec(files)
    assert spec == {"version": 1, "inputs": {"type": "cif", "files": ["../../../A.cif", "../../../B.cif"]},
                    "label": "perov", "template": "full_electronic"}
    assert list(files) == ["spec.json"]


def test_spec_steps_method_resources_and_settings(tmp_path):
    inp = _inputs(tmp_path)
    band = json.dumps({"calculation_type": "BAND", "path_method": "auto", "n_points": 1000, "bands": "all"})
    files = wb.spec_files(inp, {
        "template": "full_electronic", "sequence": ["OPT", "SP", "BAND", "DOSS"],
        "steps": {"OPT_1": {"answers": {"functional": "PBE0"}},
                  "SP_2": {"method": {"functional": "HSE06", "dispersion": True}},
                  "BAND_3": {"settings_text": band, "resources": {"memory": "48G", "ntasks": "16", "walltime": ""}}},
        "execution": {"job_arrays": {"max_running": "5"}, "max_jobs": "50"}},
        None, step1_overrides={"dft_functional": "PBE0"})
    spec = _spec(files)
    assert spec["steps"]["OPT_1"] == {"overrides": {"dft_functional": "PBE0"}}
    assert spec["steps"]["SP_2"] == {"overrides": {"method_modifications": {"new_functional": "HSE06",
                                                                            "use_dispersion": True}}}
    assert spec["steps"]["BAND_3"] == {"settings_file": "BAND_3_settings.json",
                                       "resources": {"memory": "48G", "ntasks": 16}}
    assert json.loads(files["BAND_3_settings.json"]) == json.loads(band)
    assert spec["execution"] == {"max_jobs": 50, "job_arrays": {"first_step": True, "max_running": 5}}
    # the input paths are absolute for a check in the GUI's scratch
    assert spec["inputs"]["files"][0] == str(inp.files[0])


def test_a_custom_chain_is_a_sequence(tmp_path):
    inp = _inputs(tmp_path)
    spec = _spec(wb.spec_files(inp, {"template": "full_electronic", "sequence": ["OPT", "SP", "FREQ"]}, None))
    assert spec["sequence"] == ["OPT", "SP", "FREQ"] and "template" not in spec


@pytest.mark.parametrize("draft,words", [
    ({"sequence": ["OPT", "SP"], "steps": {"SP_2": {"settings_text": "{}"}}}, "properties step"),
    ({"sequence": ["OPT", "SP", "TRANSPORT"],
      "steps": {"TRANSPORT_3": {"settings_text": '{"calculation_type": "BAND"}'}}}, "for a BAND step"),
    ({"sequence": ["OPT", "SP"], "steps": {"OPT_1": {"method": {"functional": "PBE0"}}}}, "later SCF step"),
    ({"sequence": ["OPT", "SP"], "steps": {"SP_2": {"method": {"functional": "NOPE"}}}}, "not a functional"),
    ({"sequence": ["OPT", "SP"], "steps": {"BAND_3": {}}}, "not in the chain"),
    ({"sequence": ["OPT", "SP"], "label": "has space"}, "no spaces"),
    ({"sequence": ["OPT", "SP"], "steps": {"SP_2": {"resources": {"gpus": 1}}}}, "unknown resources"),
    ({"sequence": []}, "at least one step"),
])
def test_drafts_the_builder_must_correct(tmp_path, draft, words):
    with pytest.raises(wb.BuildError) as e:
        wb.spec_files(_inputs(tmp_path), draft, None)
    assert words in str(e.value)


@pytest.mark.parametrize("calc", ["TRANSPORT", "CHARGE+POTENTIAL"])
def test_transport_and_charge_potential_take_their_settings_file(tmp_path, calc):
    """Like BAND and DOSS, these steps take the opt2d3 settings their form made."""
    conf = {"calculation_type": calc, "temperature_range": [200.0, 600.0, 50.0]} if calc == "TRANSPORT" else \
        {"calculation_type": calc, "charge_config": {"type": "ECH3", "n_points": 150},
         "potential_config": {"type": "POT3", "n_points": 100}}
    text = json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": calc,
                       "configuration": conf})
    key = f"{calc}_3"
    files = wb.spec_files(_inputs(tmp_path), {"sequence": ["OPT", "SP", calc], "steps": {key: {"settings_text": text}}},
                          None)
    name = f"{key.replace('+', '_')}_settings.json"
    assert _spec(files)["steps"][key] == {"settings_file": name}
    assert json.loads(files[name]) == json.loads(text)
    assert not hasattr(wb, "D3_FIXED")


def test_a_deck_start_runs_its_decks_as_they_are(tmp_path):
    inp = _inputs(tmp_path, kind="d12")
    with pytest.raises(wb.BuildError) as e:
        wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"OPT_1": {"settings_text": "{}"}}}, None)
    assert "as they are" in str(e.value)
    # step-1 overrides never reach a deck start
    spec = _spec(wb.spec_files(inp, {"sequence": ["OPT", "SP"]}, None, step1_overrides={"dft_functional": "X"}))
    assert "steps" not in spec


def test_walltime_over_the_partition_limit(monkeypatch):
    from mace.submission import site
    monkeypatch.setattr(site, "partition_limit", lambda part, env=None, refresh=False: (4 * 3600, part or "short"))
    out = wb.resource_checks({"walltime": "7-00:00:00", "partition": "short"}, {})
    assert out and out[0]["field"] == "walltime" and "short partition allows (04:00:00)" in out[0]["message"]
    assert wb.resource_checks({"walltime": "2:00:00"}, {}) == []
    monkeypatch.setattr(site, "partition_limit", lambda part, env=None, refresh=False: (None, None))
    assert wb.resource_checks({"walltime": "30-00:00:00"}, {}) == []


def test_one_launch_at_a_time(tmp_path):
    with wb.LaunchLock(tmp_path):
        with pytest.raises(sub.SubmitConflict) as e:
            with wb.LaunchLock(tmp_path):
                pass
        assert "Another workflow" in str(e.value)
    assert not (tmp_path / "workflow_configs" / wb.LOCK_NAME).exists()
    # a crashed launch's lock (half an hour old) does not block forever
    lock = tmp_path / "workflow_configs" / wb.LOCK_NAME
    lock.write_text("1 0\n")
    old = time.time() - wb.LOCK_STALE_S - 60
    os.utime(lock, (old, old))
    with wb.LaunchLock(tmp_path):
        pass


def test_launch_lock_follows_its_process(tmp_path):
    """A lock of this host: a running launch holds it past half an hour, a
    dead process's lock is free at once."""
    import socket
    import subprocess
    lock = tmp_path / "workflow_configs" / wb.LOCK_NAME
    lock.parent.mkdir(parents=True)
    old = time.time() - wb.LOCK_STALE_S - 60
    lock.write_text(f"{os.getpid()} 0 {socket.gethostname()}\n")
    os.utime(lock, (old, old))
    with pytest.raises(sub.SubmitConflict):
        with wb.LaunchLock(tmp_path):
            pass
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock.write_text(f"{dead.pid} {time.time():.0f} {socket.gethostname()}\n")
    with wb.LaunchLock(tmp_path):
        assert socket.gethostname() in lock.read_text()


def test_script_normalization_and_cli_json():
    a = "#SBATCH -J A\ncd workflow_outputs/workflow_20261015_100000_ab12/x\n# MACE-generated; cksum of the lines above: 1712345678 25939\n"
    b = "#SBATCH -J A\ncd workflow_outputs/workflow_20261015_100500_cd34/x\n# MACE-generated; cksum of the lines above: 2601851173 25939\n"
    assert wb.normalize_script(a) == wb.normalize_script(b)
    assert wb.parse_json("MACE v1.1.5 banner\n====\n{\n  \"ok\": true\n}\n") == {"ok": True}
    assert wb.parse_json("no json here") is None


def test_unfinished_steps_of_the_same_structures(tmp_path):
    from mace.database.materials import MaterialDatabase
    inp = _inputs(tmp_path)
    ctx = inp.folder / ".mace_context_workflow_20261015_100000_ab12"
    ctx.mkdir()
    MaterialDatabase(str(ctx / "materials.db"))
    con = sqlite3.connect(ctx / "materials.db")
    with con:
        for cid, mid, status, job in (("a1", "A", "submitted", "5001"), ("b1", "B", "completed", "5002"),
                                      ("c1", "C", "running", "5003"), ("a0", "A", "pending", None)):
            con.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, slurm_job_id, created_at) "
                        "VALUES (?,?,?,?,?,?)", (cid, mid, "OPT", status, job, "2026-10-15T10:00:00"))
    con.close()
    rows = wb.active_rows(inp.folder, inp)
    assert [(r["material_id"], r["job_id"]) for r in rows] == [("A", "5001")]


def test_a_method_without_dispersion_keeps_the_parents_d3_in_the_spec(tmp_path):
    """E-M5: the CLI spec path and the GUI builder agree. A spec step whose
    new_functional names no use_dispersion keeps its parent's D3 (engine and
    opt2d12), so a clone of it carries no dispersion answer either; an
    explicit use_dispersion is kept."""
    from mace.gui import workflow_actions as wa
    seq = ["OPT", "SP", "SP2"]
    spec = {"steps": {"SP_2": {"overrides": {"method_modifications": {"new_functional": "PBE0"}}},
                      "SP2_3": {"overrides": {"method_modifications": {"new_functional": "PBE0",
                                                                       "use_dispersion": False}}}}}
    draft, _ = wa._draft_of_spec(tmp_path, spec, seq, tmp_path, "wf", "x")
    assert draft["steps"]["SP_2"]["method"] == {"functional": "PBE0"}
    assert "dispersion" not in draft["steps"]["SP_2"]["answers"]
    assert draft["steps"]["SP2_3"]["method"] == {"functional": "PBE0", "dispersion": False}


def test_a_refused_parent_deck_is_named_with_the_clis_reason():
    """A step whose parent deck the CLI refused: the form's error gives the CLI's reason, never the bare
    "[ERROR]" its log ends with."""
    log = ("SOC deck: TWOCOMPON\nNot writing PbTe_sp.d12: the internal basis-set library (BASISSET) has no SOC "
           "terms; a SOC deck needs explicit per-element basis sets\n[ERROR]\n")
    assert wb.refusal_line({"refused": "[ERROR]", "log": log}).startswith("the internal basis-set library")
    assert wb.refusal_line({"refused": "[ERROR] mace convert: bad answer\n[ERROR]", "log": ""}) == "[ERROR] mace convert: bad answer"
    assert wb.refusal_line({"refused": "", "log": ""}) == "the CLI wrote none"


def test_draft_sequence_refuses_types_that_are_not_workflow_types():
    assert wb.draft_sequence({"sequence": ["OPT", "SP", "OPT2", "BAND"]})[0] == ["OPT", "SP", "OPT2", "BAND"]
    with pytest.raises(wb.BuildError, match="not a workflow step type: CHARGE"):
        wb.draft_sequence({"sequence": ["OPT", "SP", "CHARGE"]})
