"""The Python version the docs require is the oldest one CI tests.

README.md, INSTALLATION.md and the Crystal_d12/Crystal_d3 READMEs said 3.7+
(or 3.6+); CI tests only 3.11 and 3.12 and the HPCC module Python is 3.11,
so nothing older is known to work.
"""
import re

from conftest import REPO_ROOT

DOCS = ("README.md", "INSTALLATION.md", "Crystal_d12/README.md",
        "Crystal_d3/README.md")


def _ci_minimum():
    text = (REPO_ROOT / ".github/workflows/tests.yml").read_text()
    versions = re.search(r"python-version:\s*\[([^\]]*)\]", text).group(1)
    return min((tuple(int(p) for p in v.strip().strip("\"'").split("."))
                for v in versions.split(",")))


def test_docs_state_the_ci_tested_minimum():
    major, minor = _ci_minimum()
    claimed = []
    for rel in DOCS:
        text = (REPO_ROOT / rel).read_text()
        for m in re.finditer(r"(?:Python|needs)\s+(\d+)\.(\d+)(\+| or (?:higher|newer))",
                             text):
            claimed.append((rel, int(m.group(1)), int(m.group(2))))
    assert claimed, "no Python requirement stated in the docs"
    wrong = [c for c in claimed if (c[1], c[2]) != (major, minor)]
    assert not wrong, f"docs must require Python {major}.{minor}+: {wrong}"
    # The top-level install docs both state it.
    assert {c[0] for c in claimed} >= {"README.md", "INSTALLATION.md"}
