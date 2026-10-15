"""The repository holds no symbolic links: a Windows zip extraction drops
them, and a git checkout without core.symlinks turns each into a small text
file holding the target path, which a reader then takes as the real file."""
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT


@pytest.mark.skipif(shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_symlinks_in_the_repository():
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-s"], capture_output=True, text=True, check=True)
    links = [ln.split("\t", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("120000 ")]
    assert links == []


def test_workflow_templates_come_from_mace_submission():
    """The planner and the executor read the job-script generators from
    mace/submission, where the one copy lives."""
    from mace.workflow import executor, planner
    sub = REPO_ROOT / "mace" / "submission"
    assert planner.TEMPLATE_DIR.resolve() == sub.resolve()
    assert executor.TEMPLATE_DIR.resolve() == sub.resolve()
    for name in ("submitcrystal23.sh", "submit_prop.sh"):
        assert (planner.TEMPLATE_DIR / name).read_text().startswith("#!/bin/bash")
