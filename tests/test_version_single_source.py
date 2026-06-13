"""Regression tests: the package version has a single source of truth.

mace_cli and mace/database/export/formats.py used to hardcode their own "1.0.5"
literals that drifted from mace/__init__.py on every bump. They now derive the
version from the package. Self-contained (no test/ corpus).
"""
import re
from pathlib import Path

import mace
from conftest import REPO_ROOT


def test_package_version_is_nonempty_string():
    assert isinstance(mace.__version__, str) and mace.__version__


def test_formats_export_version_matches_package():
    from mace.database.export.formats import MACE_VERSION
    assert MACE_VERSION == mace.__version__


def test_mace_cli_imports_version_and_has_no_literal():
    src = (REPO_ROOT / "mace_cli").read_text()
    assert "from mace import __version__" in src
    # No `__version__ = "x.y.z"` literal assignment remains in the CLI.
    assert not re.search(r'__version__\s*=\s*["\']', src)


def test_animation_default_resolves_to_package():
    src = (REPO_ROOT / "mace" / "utils" / "animation.py").read_text()
    assert "def animate_mace_assembly(version=None)" in src
    assert 'version="1.0.5"' not in src


def test_only_init_holds_a_version_literal():
    """No non-test .py outside mace/__init__.py pins a bare 'x.y.z' version."""
    offenders = []
    for py in REPO_ROOT.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT)
        parts = rel.parts
        if parts and parts[0] in ("test", "tests"):
            continue
        if rel.as_posix() == "mace/__init__.py":
            continue
        try:
            text = py.read_text(errors="ignore")
        except OSError:
            continue
        # A version-literal assignment like __version__ = "1.2.3" or mace_version = '1.2.3'
        if re.search(r'(?:__version__|mace_version)\s*=\s*["\']\d+\.\d+', text):
            offenders.append(rel.as_posix())
    assert offenders == [], f"unexpected hardcoded version literal(s): {offenders}"
