"""Plotting/ipDOS_V2.py run on its own exits non-zero when a file fails (it exited 0 and drew nothing)."""
import os
import subprocess
import sys

from conftest import REPO_ROOT


def test_a_failed_doss_file_is_a_failed_run(tmp_path):
    (tmp_path / "bad_doss.DOSS.DAT").write_text("# NEPTS 5 NPROJ 1 NSPIN 1\nnot numbers\n")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "Plotting" / "ipDOS_V2.py"), "-6", "6", "total", "auto", "all",
                        "--file", "bad_doss.DOSS.DAT"], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "MPLBACKEND": "Agg"}, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DOS file(s) failed" in r.stdout
