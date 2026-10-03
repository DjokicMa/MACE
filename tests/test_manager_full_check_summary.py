"""`mace manager` (a full_check cycle) prints its database summary once.

The monitoring cycle prints "Database Stats: ..." and the callback wrapper
around it printed the same line again.
"""
import subprocess
import sys

from test_job_left_queue_without_output import MACE_CLI, _env


def test_full_check_prints_the_database_stats_once(tmp_path):
    env = _env(tmp_path, "COMPLETED")
    env.pop("PYTHONPATH", None)
    for var in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(var, None)
    r = subprocess.run([sys.executable, str(MACE_CLI), "manager"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "=== Callback Complete ===" in r.stdout          # the full_check path ran
    assert r.stdout.count("Database Stats:") == 1, r.stdout
