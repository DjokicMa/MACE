"""`setup_mace.py --update-templates` must look at the job-script templates
MACE ships (mace/submission/), not the pre-reorganization code/Job_Scripts/
paths it named, where it silently found nothing."""
import os
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

TEMPLATES = ("submitcrystal23.sh", "submit_prop.sh")


def _checkout_copy(tmp_path):
    """A throwaway copy with the files setup_mace.py checks for, so its
    activation and example scripts are not written into the checkout."""
    root = tmp_path / "mace_copy"
    for rel in ("mace/enhanced_queue_manager.py", "Crystal_d12/NewCifToD12.py",
                "Crystal_d3/CRYSTALOptToD3.py", "mace_cli"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    shutil.copy(REPO_ROOT / "setup_mace.py", root / "setup_mace.py")
    (root / "mace" / "submission").mkdir(parents=True)
    for name in TEMPLATES:
        shutil.copy(REPO_ROOT / "mace" / "submission" / name,
                    root / "mace" / "submission" / name)
    return root


def _run(root, tmp_path, *args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "MACE_HOME")}
    env["HOME"] = str(tmp_path)
    return subprocess.run([sys.executable, str(root / "setup_mace.py"), *args],
                          cwd=root, env=env, capture_output=True, text=True,
                          timeout=120)


def test_update_templates_checks_the_shipped_templates(tmp_path):
    root = _checkout_copy(tmp_path)
    before = {n: (root / "mace/submission" / n).read_text() for n in TEMPLATES}
    proc = _run(root, tmp_path, "--update-templates", "--shell", "bash",
                "--no-shell", "--skip-deps", "--force")
    assert proc.returncode == 0, proc.stderr
    for name in TEMPLATES:
        assert f"mace/submission/{name}" in proc.stdout
        # The shipped templates find MACE_HOME at run time: nothing to rewrite.
        assert (root / "mace/submission" / name).read_text() == before[name]
    assert "not found" not in proc.stdout


def test_update_templates_reports_a_missing_template(tmp_path):
    root = _checkout_copy(tmp_path)
    (root / "mace/submission/submit_prop.sh").unlink()
    proc = _run(root, tmp_path, "--update-templates", "--shell", "bash",
                "--no-shell", "--skip-deps", "--force")
    assert proc.returncode == 0, proc.stderr
    assert "mace/submission/submit_prop.sh not found" in proc.stdout


def test_help_names_this_script(tmp_path):
    proc = _run(REPO_ROOT, tmp_path, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "setup_crystal_tools.py" not in proc.stdout
    assert "python setup_mace.py --update-templates" in proc.stdout
