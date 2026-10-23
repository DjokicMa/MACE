"""The GUI's Help links must name a Git ref that exists on GitHub."""
import subprocess
from pathlib import Path

from mace.gui import helplinks


def _repo_with_tag(tmp_path: Path, tag: str = "") -> Path:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "--allow-empty", "-m", "x"], check=True, env=env)
    if tag:
        subprocess.run(["git", "-C", str(tmp_path), "tag", tag], check=True, env=env)
    return tmp_path


def test_untagged_final_version_points_at_main(tmp_path):
    assert helplinks.docs_ref("9.8.7", _repo_with_tag(tmp_path)) == "main"


def test_tagged_final_version_points_at_its_tag(tmp_path):
    assert helplinks.docs_ref("9.8.6", _repo_with_tag(tmp_path, "v9.8.6")) == "v9.8.6"


def test_preview_and_dev_versions_always_point_at_main(tmp_path):
    repo = _repo_with_tag(tmp_path, "v9.8.5-preview")
    assert helplinks.docs_ref("9.8.5-preview", repo) == "main"
    assert helplinks.docs_ref("9.8.5.dev1", repo) == "main"
    assert helplinks.docs_ref("", repo) == "main"


def test_not_a_git_checkout_points_at_main(tmp_path):
    assert helplinks.docs_ref("1.2.3", tmp_path) == "main"


def test_shell_builds_the_url_from_the_server_ref():
    js = (Path(__file__).resolve().parents[1] / "mace" / "gui" / "static" / "js" / "shell.js").read_text()
    assert "/blob/${encodeURIComponent(info && info.docs_ref" in js
    assert "blob/v${" not in js
