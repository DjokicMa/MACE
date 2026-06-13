"""Tests for WorkflowExecutor._extract_basis_path_from_d12 basis resolution.

Two fixes are covered:
1. The hardcoded MSU/Mendoza institutional basis path is no longer the only
   non-relative option — MACE_TZ_BASIS_PATH / MACE_DZ_BASIS_PATH / MACE_BASIS_DIR
   let any site point at its own basis directory (and an explicit override wins).
2. The bundled-config lookup used Path(__file__).parent.parent (-> mace/) so it
   never found the repo-root mace_config.py and the shipped basis sets were
   silently unused. It is now parent.parent.parent (repo root).

The method does not touch ``self`` on these paths, so a SimpleNamespace stands
in for the executor instance.
"""
import os
import types
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.workflow.executor import WorkflowExecutor

resolve = WorkflowExecutor._extract_basis_path_from_d12
DUMMY = types.SimpleNamespace()


def test_tz_env_override_wins(tmp_path, monkeypatch):
    tz = tmp_path / "mytz"
    tz.mkdir()
    monkeypatch.setenv("MACE_TZ_BASIS_PATH", str(tz))
    assert resolve(DUMMY, "mat_full.basis.triplezeta.d12") == str(tz)


def test_dz_env_override_wins(tmp_path, monkeypatch):
    dz = tmp_path / "mydz"
    dz.mkdir()
    monkeypatch.setenv("MACE_DZ_BASIS_PATH", str(dz))
    assert resolve(DUMMY, "mat_full.basis.doublezeta.d12") == str(dz)


def test_mace_basis_dir_parent(tmp_path, monkeypatch):
    (tmp_path / "full.basis.triplezeta").mkdir()
    monkeypatch.setenv("MACE_BASIS_DIR", str(tmp_path))
    monkeypatch.delenv("MACE_TZ_BASIS_PATH", raising=False)
    assert resolve(DUMMY, "mat_full.basis.triplezeta.d12") == str(
        tmp_path / "full.basis.triplezeta")


def test_bundled_config_path_now_resolves(monkeypatch):
    """With no env override, a triplezeta filename resolves to the repo's bundled
    basis dir — proving the parent.parent.parent fix found mace_config.py."""
    for var in ("MACE_TZ_BASIS_PATH", "MACE_BASIS_DIR"):
        monkeypatch.delenv(var, raising=False)
    bundled = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"
    if not bundled.exists():
        pytest.skip("repo does not ship the bundled triplezeta basis set")
    result = resolve(DUMMY, "mat_full.basis.triplezeta.d12")
    assert result is not None
    assert "basis_sets/full.basis.triplezeta" in result
