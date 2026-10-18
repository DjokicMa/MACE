"""The GitHub Actions workflow keeps its shape (polish CI2): the two pytest
jobs unchanged, plus `gui-smoke`, which runs the browser smoke subset with
headless Firefox and Chromium and fails (never skips) without a browser."""
import yaml

from conftest import REPO_ROOT

WF = REPO_ROOT / ".github" / "workflows" / "tests.yml"


def jobs():
    return yaml.safe_load(WF.read_text())["jobs"]


def test_gui_smoke_job():
    j = jobs()["gui-smoke"]
    assert j["runs-on"] == "ubuntu-24.04" and 0 < j["timeout-minutes"] <= 35
    steps = j["steps"]
    uses = [s["uses"] for s in steps if "uses" in s]
    assert uses and all(u.startswith("actions/") for u in uses), uses
    run = "\n".join(s.get("run", "") for s in steps)
    assert "requirements-gui-test.txt" in run and "requirements.txt" in run
    smoke = next(s for s in steps if "-m guismoke" in s.get("run", ""))
    assert smoke["env"]["MACE_GUI_REQUIRE_BROWSER"] == "1"
    assert "pytest -m guismoke" in smoke["run"]
    # headless Firefox has no WebGL without an X display: the smoke run needs Xvfb
    assert smoke["run"].startswith("xvfb-run ") and "xvfb" in run.split("pip install")[0]
    assert any("upload-artifact" in u for u in uses)


def test_existing_jobs_unchanged():
    j = jobs()
    assert j["pytest"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    for name in ("pytest", "pytest-optional"):
        runs = [s.get("run", "") for s in j[name]["steps"]]
        assert any("python -m pytest -ra" in r for r in runs), name
        # the whole suite takes over 30 minutes on a runner (1.1.5 hit that cap)
        assert 60 <= j[name]["timeout-minutes"] <= 120, name
