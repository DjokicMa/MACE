"""pytest.ini's comment must describe where the tests are.

It said the interactive menu tests under Crystal_d12/ run separately; there
are none there - every test is under tests/.
"""
import subprocess

from conftest import REPO_ROOT


def test_every_test_module_is_under_tests():
    tracked = subprocess.run(["git", "ls-files", "*test_*.py"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.split()
    if not tracked:  # not a git checkout (e.g. an sdist); nothing to compare
        return
    assert all(p.startswith("tests/") for p in tracked), [
        p for p in tracked if not p.startswith("tests/")]


def test_pytest_ini_names_no_other_test_location():
    comments = "\n".join(l for l in (REPO_ROOT / "pytest.ini").read_text().splitlines()
                         if l.lstrip().startswith("#"))
    for tree in ("Crystal_d12", "Crystal_d3", "Plotting", "mace/"):
        assert tree not in comments, f"pytest.ini comment mentions {tree}"
