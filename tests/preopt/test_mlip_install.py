"""The ML-environment installer (mace/preopt/mlip_install.py) behind `mace mlip
install` and the GUI's Set up ML environment, with a fake base Python whose
venv's pip only logs what it was asked (no network, nothing installed)."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)

@pytest.fixture
def fake(tmp_path, monkeypatch):
    base, fenv = F.make_fake_base_python(tmp_path / "fakepy", tmp_path / "pip.log")
    log = tmp_path / "pip.log"
    for k, v in fenv.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PYTHONPATH", "/opt/module/site-packages")   # the module's: must never reach pip
    from mace.preopt import mlip_install as MI
    monkeypatch.setattr(MI, "gpu_present", lambda: False)
    return {"python": str(base), "log": log, "MI": MI}


def pip_calls(log):
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def test_plan_names_the_default_place_pins_licenses_and_size(fake, tmp_path):
    from mace.preopt import detect
    MI = fake["MI"]
    plan = MI.make_plan(backends=["mace-torch", "chgnet"], python=fake["python"])
    assert plan["dir"] == str(tmp_path / "data" / "mace-tool" / "mlip" / "venv") == str(detect.default_env_dir())
    assert plan["state"] == "new" and plan["problems"] == [] and plan["torch"] == "cpu"
    assert plan["pins"] == ["mace-torch==0.3.16", "chgnet==0.4.2"]
    assert [x["license"] for x in plan["licenses"]] == ["MIT", "BSD-3-Clause-LBNL"]
    assert plan["estimate_files"] > 40000 and plan["python_version"] == "3.12.2" and plan["python_why"] == "chosen"
    cmds = plan["commands"]
    assert cmds[0] == "unset PYTHONPATH" and "--upgrade pip" in cmds[2] and MI.CPU_INDEX in cmds[3]
    assert cmds[-1] == f"mace mlip setup --python {plan['dir']}/bin/python"
    with pytest.raises(MI.InstallError):
        MI.make_plan(backends=["evil-package"], python=fake["python"])


def test_install_runs_the_steps_without_pythonpath_and_records(fake):
    from mace.preopt import detect
    MI = fake["MI"]
    plan = MI.make_plan(backends=["chgnet"], python=fake["python"])
    lines = []
    res = MI.install(plan, log=lines.append)
    calls = pip_calls(fake["log"])
    assert [c["argv"][:2] for c in calls] == [["install", "--upgrade"], ["install", "torch"], ["install", "chgnet==0.4.2"],
                                              ["check"]]
    assert calls[1]["argv"] == ["install", "torch", "--index-url", MI.CPU_INDEX]
    assert not any(c["pythonpath"] for c in calls)
    py = str(Path(plan["dir"]) / "bin" / "python")
    assert res["ok"] and res["python"] == py and detect.python_source() == (py, f"recorded in {detect.config_path()}")
    assert any("Collecting things (fake pip)" in x for x in lines) and any(x.startswith("== ") for x in lines)
    # a second run repairs in place: no new venv, the same pip steps
    plan2 = MI.make_plan(backends=["chgnet"], python=fake["python"])
    assert plan2["state"] == "existing"
    MI.install(plan2, log=lambda s: None)
    assert len(pip_calls(fake["log"])) == 8


def test_a_failed_install_removes_only_the_folder_it_created(fake, monkeypatch):
    from mace.preopt import detect
    MI = fake["MI"]
    monkeypatch.setenv("FAKE_PIP_FAIL", "torch")
    plan = MI.make_plan(backends=["chgnet"], python=fake["python"])
    lines = []
    with pytest.raises(MI.InstallError) as exc:
        MI.install(plan, log=lines.append)
    assert "the torch install failed" in str(exc.value) and "removed" in str(exc.value)
    assert not Path(plan["dir"]).exists() and detect.python_source() == (None, "")
    assert any(x.startswith("Removed ") for x in lines)
    # an environment that was there before stays, whatever fails
    monkeypatch.delenv("FAKE_PIP_FAIL")
    MI.install(MI.make_plan(backends=["chgnet"], python=fake["python"]), log=lambda s: None)
    monkeypatch.setenv("FAKE_PIP_FAIL", "chgnet")
    with pytest.raises(MI.InstallError):
        MI.install(MI.make_plan(backends=["chgnet"], python=fake["python"]), log=lambda s: None)
    assert (Path(plan["dir"]) / "bin" / "python").is_file()


def test_an_empty_folder_is_used_and_emptied_again_on_failure(fake, tmp_path, monkeypatch):
    MI = fake["MI"]
    d = tmp_path / "mine"
    d.mkdir()
    monkeypatch.setenv("FAKE_PIP_FAIL", "check")       # pip check failing is only a warning
    plan = MI.make_plan(str(d), backends=["chgnet"], python=fake["python"])
    assert plan["state"] == "empty"
    MI.install(plan, log=lambda s: None)
    monkeypatch.setenv("FAKE_PIP_FAIL", "upgrade")
    (d2 := tmp_path / "mine2").mkdir()
    with pytest.raises(MI.InstallError):
        MI.install(MI.make_plan(str(d2), backends=["chgnet"], python=fake["python"]), log=lambda s: None)
    assert d2.is_dir() and not any(d2.iterdir())


def test_a_folder_that_is_not_an_environment_is_refused(fake, tmp_path):
    MI = fake["MI"]
    d = tmp_path / "busy"
    d.mkdir()
    (d / "thesis.tex").write_text("x")
    plan = MI.make_plan(str(d), backends=["chgnet"], python=fake["python"])
    assert plan["problems"] and "not a Python environment" in plan["problems"][0]
    with pytest.raises(MI.InstallError):
        MI.install(plan, log=lambda s: None)
    assert (d / "thesis.tex").read_text() == "x"


def test_cancel_stops_the_child_and_rolls_back(fake, monkeypatch):
    MI = fake["MI"]
    monkeypatch.setenv("FAKE_PIP_SLEEP", "chgnet:60")
    plan = MI.make_plan(backends=["chgnet"], python=fake["python"])
    flag = threading.Event()
    threading.Timer(1.5, flag.set).start()
    t0 = time.monotonic()
    with pytest.raises(MI.InstallCanceled):
        MI.install(plan, log=lambda s: None, canceled=flag.is_set)
    assert time.monotonic() - t0 < 20 and not Path(plan["dir"]).exists()


def test_gpu_asked_without_a_gpu_installs_the_cpu_build(fake):
    MI = fake["MI"]
    plan = MI.make_plan(backends=["mace-torch"], gpu=True, python=fake["python"])
    assert plan["torch"] == "cpu" and any("no NVIDIA GPU" in w for w in plan["warnings"])


QUOTA = """Disk quotas for user someone (uid 1000):
     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace
hpcc-ib.vast.example.edu:/home 58022588  104857600 131072000          971543  1000000 1000000
/dev/sdb1  10*  20  30  7days  1  2  3
"""


def test_quota_rows_and_the_ninety_percent_warning(fake, monkeypatch, tmp_path):
    MI = fake["MI"]
    rows = MI.parse_quota(QUOTA)
    assert rows[0]["files"] == 971543 and rows[0]["files_quota"] == 1000000 and rows[1]["used_kb"] == 10
    monkeypatch.setattr(MI, "_mount_source", lambda p: "hpcc-dmz.vast.example.edu:/home")   # same export, other name
    q = MI.quota_check(tmp_path, 1500, 46000, runner=lambda: QUOTA)
    assert q["available"] and q["files_quota"] == 1000000
    assert len(q["warnings"]) == 1 and "file quota" in q["warnings"][0] and "102 %" in q["warnings"][0]
    plan = MI.make_plan(backends=["chgnet"], python=fake["python"], quota_runner=lambda: QUOTA)
    assert any("file quota" in w for w in plan["warnings"])
    assert MI.quota_check(tmp_path, 1, 1, runner=lambda: None) == {"available": False, "warnings": []}


def test_no_python_310_is_a_problem_not_a_crash(fake, monkeypatch, tmp_path):
    MI = fake["MI"]
    old = tmp_path / "py39"
    old.write_text("#!/bin/sh\necho 3 9 18\n")
    old.chmod(0o755)
    plan = MI.make_plan(backends=["chgnet"], python=str(old))
    assert plan["problems"] and "3.10 or newer" in plan["problems"][0] and plan["commands"] == []


def mace(*args, env=None, stdin=subprocess.DEVNULL):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(env or {})
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "mlip", *args], env=e,
                       cwd=os.environ["TMPDIR"], stdin=stdin, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_cli_install_dry_run_refusal_and_a_real_run(fake, tmp_path):
    d = tmp_path / "env"
    rc, out = mace("install", "--dir", str(d), "--python", fake["python"], "--backend", "chgnet", "--dry-run")
    assert rc == 0 and "chgnet==0.4.2" in out and "code BSD-3-Clause-LBNL" in out and "The same steps by hand" in out
    assert not d.exists()
    rc, out = mace("install", "--dir", str(d), "--python", fake["python"], "--backend", "chgnet", "--no-fetch")
    assert rc == 2 and "pass --yes" in out and not d.exists()
    rc, out = mace("install", "--dir", str(d), "--python", fake["python"], "--backend", "chgnet", "--no-fetch", "--yes")
    assert rc == 0, out
    assert "The ML environment is ready" in out and "Upgrading pip" in out
    rec = json.loads((tmp_path / "config" / "mace" / "mlip.json").read_text())
    assert rec["python"] == str(d / "bin" / "python")
    rc, out = mace("install", "--backend", "nonsense", "--python", fake["python"], "--yes")
    assert rc == 2 and "unknown package" in out
    rc, out = mace("setup")
    assert "`mace mlip install` makes the ML environment for you" in out


def test_named_environments_are_found_recorded_and_kept_apart(tmp_path, monkeypatch):
    """A model whose pins conflict with mace-torch's runs in its own venv
    (registry ``env``): venv-<env> in the default place, envs.<env> in mlip.json,
    MACE_MLIP_PYTHON_<ENV>; the default environment is unaffected."""
    from mace.preopt import detect
    assert detect.default_env_dir("fairchem") == tmp_path / "data" / "mace-tool" / "mlip" / "venv-fairchem"
    fake = F.make_fake_python(detect.default_env_dir("fairchem"))
    assert detect.python_source("fairchem") == (str(fake), "the default location")
    assert detect.python_source() == (None, "")
    assert detect.configured_python({"id": "x", "env": "fairchem"}) == str(fake)
    other = F.make_fake_python(tmp_path / "elsewhere")
    detect.save_python(str(other), "fairchem")
    rec = json.loads(detect.config_path().read_text())
    assert rec == {"envs": {"fairchem": str(other)}}
    assert detect.python_source("fairchem")[0] == str(other) and detect.python_source() == (None, "")
    monkeypatch.setenv("MACE_MLIP_PYTHON_FAIRCHEM", str(fake))
    assert detect.python_source("fairchem") == (str(fake), "MACE_MLIP_PYTHON_FAIRCHEM")
