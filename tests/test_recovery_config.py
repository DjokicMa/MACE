"""Which recovery_config.yaml the recovery engine reads, and how.

Before: ErrorRecoveryEngine looked only for a RELATIVE recovery_config.yaml
(the job directory), so the shipped mace/config/recovery_config.yaml was never
read; and a file that was read REPLACED the whole error_recovery section
(shallow dict.update), so the shipped one - which named handlers that do not
exist - would have switched disk-space recovery off.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mace.recovery import recovery as rec
from mace.recovery.recovery import ErrorRecoveryEngine

REPO = Path(__file__).resolve().parent.parent
MACE_CLI = REPO / "mace_cli"

#: Keys some handler or the engine actually reads, per error type.
READ_KEYS = {"handler", "max_retries", "resubmit_delay", "memory_factor", "max_memory",
             "max_cycles_increase", "fmixing_adjustment", "walltime_factor",
             "max_walltime", "enable_restart", "cleanup_scratch", "escalate_on_failure"}

#: The effective configuration before this change (the built-in defaults - the
#: only thing that was ever in force), for every error type recovered today.
BEFORE = {
    "shrink_error": {"handler": "fixk_handler", "max_retries": 3, "resubmit_delay": 300,
                     "escalate_on_failure": True},
    "memory_error": {"handler": "memory_handler", "memory_factor": 1.5, "max_memory": "200GB",
                     "max_retries": 2, "resubmit_delay": 600},
    "convergence_error": {"handler": "convergence_handler", "max_cycles_increase": 1000,
                          "fmixing_adjustment": 10, "max_retries": 2, "resubmit_delay": 300},
    "timeout_error": {"handler": "timeout_handler", "walltime_factor": 2.0,
                      "max_walltime": "7-00:00:00", "max_retries": 2, "resubmit_delay": 900,
                      "enable_restart": True},   # was the timeout_handler's own default
    "opt_trust_radius_error": {"handler": "opt_fresh_start_handler", "max_retries": 2,
                               "resubmit_delay": 300},
    "disk_space_error": {"handler": "cleanup_handler", "cleanup_scratch": True,
                         "max_retries": 1, "resubmit_delay": 1800},
}

#: The shipped file as it was: five handlers that do not exist, and
#: disk_space_error sent to one of them with max_retries 0.
OLD_SHIPPED = """
error_recovery:
  disk_space_error:
    handler: "manual_escalation"
    max_retries: 0
  basis_set_error:
    handler: "manual_escalation"
    max_retries: 0
  geometry_error:
    handler: "geometry_handler"
    max_retries: 2
  basis_linear_dependence:
    handler: "linear_dependence_handler"
  symmetry_error:
    handler: "symmetry_handler"
  timeout_error:
    handler: "timeout_handler"
    max_retries: 2
"""


def _engine(tmp_path, config_path=None):
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"), config_path=config_path)


def _read(cfg):
    return {t: {k: v for k, v in e.items() if k in READ_KEYS}
            for t, e in cfg["error_recovery"].items()}


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    work = tmp_path / "job"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def test_the_shipped_file_is_found_from_any_directory(tmp_path, in_tmp):
    eng = _engine(tmp_path)
    assert eng.config_path == rec.PACKAGED_RECOVERY_CONFIG
    assert rec.PACKAGED_RECOVERY_CONFIG == REPO / "mace" / "config" / "recovery_config.yaml"


def test_behaviour_is_unchanged_for_every_error_type_recovered_today(tmp_path, in_tmp, capsys):
    eng = _engine(tmp_path)
    assert _read(eng.config) == BEFORE
    # ... and the shipped file itself is clean: no ignored entries.
    assert "does not exist" not in capsys.readouterr().out


def test_shipped_file_agrees_with_the_built_in_defaults(tmp_path, in_tmp, monkeypatch):
    shipped = _read(_engine(tmp_path).config)
    monkeypatch.setattr(rec, "PACKAGED_RECOVERY_CONFIG", tmp_path / "missing.yaml")
    builtin = _engine(tmp_path)
    assert builtin.config_path is None
    assert _read(builtin.config) == shipped


def test_shipped_file_names_only_real_handlers():
    data = yaml.safe_load(rec.PACKAGED_RECOVERY_CONFIG.read_text())
    known = ErrorRecoveryEngine.known_handlers()
    assert {e["handler"] for e in data["error_recovery"].values()} <= known
    for name in known:
        assert callable(getattr(ErrorRecoveryEngine, name))


def test_job_directory_file_wins_over_the_shipped_one(tmp_path, in_tmp):
    (in_tmp / "recovery_config.yaml").write_text("error_recovery:\n  timeout_error:\n    max_retries: 1\n")
    eng = _engine(tmp_path)
    assert eng.config_path == in_tmp / "recovery_config.yaml"
    t = eng.config["error_recovery"]["timeout_error"]
    assert t["max_retries"] == 1
    # Deep merge: the rest of timeout_error and every other type are untouched.
    assert {k: v for k, v in t.items() if k in READ_KEYS} == dict(BEFORE["timeout_error"], max_retries=1)
    assert _read(eng.config)["disk_space_error"] == BEFORE["disk_space_error"]
    assert eng.config["global_settings"]["max_concurrent_recoveries"] == 10


def test_explicit_path_wins_over_both(tmp_path, in_tmp):
    (in_tmp / "recovery_config.yaml").write_text("error_recovery:\n  memory_error:\n    max_retries: 5\n")
    mine = tmp_path / "mine.yaml"
    mine.write_text("error_recovery:\n  memory_error:\n    memory_factor: 2.0\n")
    eng = _engine(tmp_path, config_path=str(mine))
    assert eng.config_path == mine
    m = eng.config["error_recovery"]["memory_error"]
    assert (m["memory_factor"], m["max_retries"]) == (2.0, 2)


def test_missing_explicit_path_falls_back(tmp_path, in_tmp, capsys):
    eng = _engine(tmp_path, config_path=str(tmp_path / "nope.yaml"))
    assert eng.config_path == rec.PACKAGED_RECOVERY_CONFIG
    assert "not found" in capsys.readouterr().out


def test_unknown_handlers_are_ignored_and_disk_space_recovery_survives(tmp_path, in_tmp, capsys):
    (in_tmp / "recovery_config.yaml").write_text(OLD_SHIPPED)
    eng = _engine(tmp_path)
    out = capsys.readouterr().out
    for bad in ("manual_escalation", "geometry_handler", "linear_dependence_handler",
                "symmetry_handler"):
        assert bad in out
    assert _read(eng.config) == BEFORE           # the good timeout entry merges in unchanged
    for dropped in ("basis_set_error", "geometry_error", "basis_linear_dependence",
                    "symmetry_error"):
        assert dropped not in eng.config["error_recovery"]


def test_malformed_entries_do_not_crash(tmp_path, in_tmp, capsys):
    (in_tmp / "recovery_config.yaml").write_text(
        "error_recovery:\n  timeout_error: 3\n  memory_error:\n    handler: null\n")
    eng = _engine(tmp_path)
    assert _read(eng.config) == BEFORE
    (in_tmp / "recovery_config.yaml").write_text("- just\n- a list\n")
    assert _read(_engine(tmp_path).config) == BEFORE
    (in_tmp / "recovery_config.yaml").write_text("")
    assert _read(_engine(tmp_path).config) == BEFORE


def test_max_retries_zero_switches_a_recovery_off(tmp_path, in_tmp):
    (in_tmp / "recovery_config.yaml").write_text("error_recovery:\n  convergence_error:\n    max_retries: 0\n")
    eng = _engine(tmp_path)
    assert eng.config["error_recovery"]["convergence_error"]["max_retries"] == 0
    assert eng.config["error_recovery"]["convergence_error"]["handler"] == "convergence_handler"


def test_queue_manager_uses_the_job_directory_config(tmp_path, in_tmp):
    """The queue manager builds its engine with only a db path, from the
    directory it runs in (the job directory, as a completion callback)."""
    (in_tmp / "recovery_config.yaml").write_text("error_recovery:\n  timeout_error:\n    max_retries: 1\n")
    from mace.queue.manager import EnhancedCrystalQueueManager
    qm = EnhancedCrystalQueueManager(d12_dir=str(in_tmp), db_path=str(tmp_path / "q.db"))
    assert qm.error_recovery_engine is not None
    assert qm.error_recovery_engine.config["error_recovery"]["timeout_error"]["max_retries"] == 1
    assert qm.error_recovery_engine.config["error_recovery"]["timeout_error"]["walltime_factor"] == 2.0


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_mace_recover_config_writes_the_effective_configuration(tmp_path, in_tmp):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(MACE_CLI), "recover", "--action", "config",
                        "--db", str(tmp_path / "cli.db")],
                       cwd=in_tmp, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    written = yaml.safe_load((in_tmp / "recovery_config.yaml").read_text())
    assert _read(written) == BEFORE
