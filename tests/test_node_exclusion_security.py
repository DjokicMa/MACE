"""Regression tests for the node_exclusion SLURM query.

query_nodes_by_type() built ``scontrol show nodes | grep 'NodeName={node_type}'``
and ran it with shell=True. node_type is user-controlled (CLI --query /
--exclude-type and the interactive planner), so this was a shell-injection
vector. It now runs ``scontrol show nodes`` as an argv list (no shell) and
filters in Python with an re.escape'd pattern.
"""
import os
import stat

import pytest

from conftest import REPO_ROOT
from mace.utils.node_exclusion import NodeExclusionManager


def _fake_scontrol(bin_dir):
    """A fake `scontrol` that emits a few NodeName= records like real SLURM."""
    fake = bin_dir / "scontrol"
    fake.write_text(
        "#!/bin/bash\n"
        "cat <<'EOF'\n"
        "NodeName=amr-042 Arch=x86_64 State=IDLE\n"
        "NodeName=amr-007 Arch=x86_64 State=IDLE\n"
        "NodeName=nvf-001 Arch=x86_64 State=ALLOCATED\n"
        "EOF\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return fake


def test_no_shell_true_in_source():
    """No active (non-comment) line may use shell=True."""
    for line in (REPO_ROOT / "mace/utils/node_exclusion.py").read_text().splitlines():
        assert "shell=True" not in line.split("#", 1)[0], (
            f"active shell=True reintroduced: {line.strip()!r}")


def test_query_nodes_by_type_filters_correctly(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_scontrol(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    mgr = NodeExclusionManager()
    assert mgr.query_nodes_by_type("amr") == ["amr-007", "amr-042"]
    assert mgr.query_nodes_by_type("nvf") == ["nvf-001"]


def test_query_nodes_by_type_does_not_execute_injection(tmp_path, monkeypatch):
    """A node_type carrying shell metacharacters must not run anything."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_scontrol(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    sentinel = tmp_path / "pwned"
    malicious = f"amr'; touch {sentinel}; echo '"

    mgr = NodeExclusionManager()
    # No shell -> the regex simply matches nothing; crucially, no side effect.
    result = mgr.query_nodes_by_type(malicious)
    assert result == []
    assert not sentinel.exists(), "injection executed — shell=True regression"
