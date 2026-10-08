"""docs/gui.md is a guide a new user can follow, and it matches the code.

It has the laptop walkthrough (with a ~/.ssh/config ProxyJump example) and a
Troubleshooting section whose messages are the ones the server, the CLI and
the page really show; no dated test notes, no "demo", no Open OnDemand proxy
advice (the session cookie is scoped to its own path instead); the README's
images exist."""
import re

from conftest import REPO_ROOT

DOC = (REPO_ROOT / "docs" / "gui.md").read_text()


def test_the_new_user_sections_are_there():
    heads = re.findall(r"^#+ (.+)$", DOC, re.M)
    for h in ("First run from your computer", "Troubleshooting", "Security", "Dependencies", "Limitations",
              "Display range, boundary and rendering", "Settings and your profile"):
        assert h in heads, h
    assert heads[0] == "MACE GUI (`mace gui`)"
    assert "ProxyJump hpcc" in DOC and "Host <dev-node>" in DOC
    assert "PowerShell" in DOC and "PuTTY" in DOC and "macOS" in DOC


def test_no_internal_notes():
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", DOC), re.findall(r".{40}\b20\d\d-\d\d-\d\d\b.{20}", DOC)
    assert not re.search(r"\bdemo\b", DOC, re.I)
    assert "/rnode/" not in DOC and "OnDemand" not in DOC
    assert "nvidia-smi`, Firefox" not in DOC and "screen was locked" not in DOC


def test_troubleshooting_quotes_real_messages():
    shell = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "shell.js").read_text()
    server = (REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    cli = (REPO_ROOT / "mace" / "gui" / "cli.py").read_text()
    assert "Lost the connection to the MACE server" in DOC and "Lost the connection to the MACE server" in shell
    assert "Misdirected request" in DOC and "Misdirected request" in server
    assert "cannot listen on" in DOC and "cannot listen on" in cli
    assert "MACE_GUI_PORT" in DOC and "MACE_GUI_PORT" in cli
    assert "--insecure-network" in DOC and "MACE_GUI_TOKEN" in DOC and "MACE_GUI_PRINT_TOKEN" in DOC


def test_every_cli_option_is_documented():
    from mace.gui.cli import build_parser
    opts = {o for a in build_parser()._actions for o in a.option_strings if o.startswith("--") and o != "--help"}
    missing = sorted(o for o in opts if o not in DOC)
    assert not missing, missing


def test_readme_images_exist():
    readme = (REPO_ROOT / "README.md").read_text()
    for src in re.findall(r'(?:src|srcset)="([^"]+)"', readme):
        if not src.startswith("http"):
            assert (REPO_ROOT / src).is_file(), src


def _slug(heading):
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to -."""
    h = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


def test_every_anchor_link_has_its_heading():
    for name in ("docs/gui.md", "docs/preopt.md"):
        text = (REPO_ROOT / name).read_text()
        anchors = {_slug(h) for h in re.findall(r"^#+ (.+)$", text, re.M)}
        missing = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in anchors]
        assert not missing, (name, missing)


def test_the_tunnel_hint_names_no_site():
    """The hint's gateway is <gateway> unless MACE_GUI_GATEWAY names one."""
    assert "msu.edu" not in DOC and "you@<gateway>" in DOC and "MACE_GUI_GATEWAY" in DOC
