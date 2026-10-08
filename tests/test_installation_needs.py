"""INSTALLATION.md's "What MACE needs" matches the code and requirements.txt.

Every package of requirements.txt is named in the first table; every
environment variable the section names is read somewhere in MACE (a renamed
or removed variable fails here); and the section keeps saying that no shell
startup file is needed, which tests/test_cli_through_symlink.py checks in
practice.
"""
import re

from conftest import REPO_ROOT

TEXT = (REPO_ROOT / "INSTALLATION.md").read_text()
SECTION = TEXT[TEXT.index("\n## What MACE needs\n"):TEXT.index("\n## Detailed Installation")]


def _sources():
    files = [REPO_ROOT / "mace_cli", *REPO_ROOT.joinpath("mace").rglob("*.py"),
             *REPO_ROOT.joinpath("mace").rglob("*.sh"), *REPO_ROOT.joinpath("Crystal_d12").glob("*.py"),
             *REPO_ROOT.joinpath("Crystal_d3").glob("*.py")]
    return "\n".join(p.read_text(errors="ignore") for p in files if "__pycache__" not in p.parts)


def test_every_required_package_is_named():
    first_table = SECTION[:SECTION.index("| Optional feature |")].lower()
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~;\[ ]", line)[0].lower()
        assert {"pyyaml": "pyyaml"}.get(name, name) in first_table, name


def test_every_named_variable_exists_in_the_code():
    src = _sources()
    names = set(re.findall(r"\bMACE_[A-Z0-9_]*[A-Z0-9]\b", SECTION))
    assert len(names) > 30
    missing = sorted(n for n in names if not re.search(rf"\b{n}\b", src))
    assert not missing, missing


def test_no_shell_startup_file_needed():
    assert "**No shell startup file is needed.**" in SECTION
    assert "ln -s <checkout>/mace_cli ~/bin/mace" in SECTION
