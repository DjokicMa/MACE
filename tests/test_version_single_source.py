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
        # Skip hidden/tooling dirs (e.g. .venv, editor worktrees) that may hold
        # checked-out copies of the source tree and would otherwise be false positives.
        if any(p.startswith(".") for p in parts):
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


def test_readme_banner_matches_package_version():
    """The README banner drifted to 1.1.0 while the package said 1.1.1.

    The checks above only ever looked at .py files and mace_cli, so a version
    written in prose had nothing holding it to the package. Only the banner is
    pinned here: "NEW in v1.1.0" is a historical statement that must NOT track
    the current version, and the citation block deliberately keeps the v1.0.0
    Zenodo DOI until a later deposit exists.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    banners = re.findall(r"<strong>Version (\d+\.\d+\.\d+)</strong>", readme)
    assert banners, "README no longer has a <strong>Version x.y.z</strong> banner"
    assert banners == [mace.__version__], (
        f"README banner says {banners}, package says {mace.__version__}")
