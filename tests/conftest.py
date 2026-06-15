"""Shared pytest fixtures and helpers for the MACE regression suite.

These tests verify the parsers/handlers against the REAL CRYSTAL outputs under
``test/`` (per project policy: no synthetic CRYSTAL output). That corpus is
~12 GB and gitignored, so it is present on developer machines but not in CI —
the data-dependent tests therefore ``skip`` cleanly when it is absent, while the
self-contained logic tests (recovery input editing, etc.) always run.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "test"

# Make the in-repo `mace` package importable without installation.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Crystal_d12/ and Crystal_d3/ are not packages and their modules use bare
# sibling imports (e.g. `from d12_constants import ...`, `from d3_kpoints import ...`),
# so they must be on sys.path directly.
_CRYSTAL_D12 = REPO_ROOT / "Crystal_d12"
if _CRYSTAL_D12.is_dir() and str(_CRYSTAL_D12) not in sys.path:
    sys.path.insert(0, str(_CRYSTAL_D12))

_CRYSTAL_D3 = REPO_ROOT / "Crystal_d3"
if _CRYSTAL_D3.is_dir() and str(_CRYSTAL_D3) not in sys.path:
    sys.path.insert(0, str(_CRYSTAL_D3))


def find_data(pattern: str, must_contain: str = None) -> Path:
    """First ``test/`` file matching glob ``pattern`` (optionally whose text
    contains ``must_contain``), else ``pytest.skip``.

    Skips — rather than fails — when the gitignored ``test/`` corpus is missing,
    so the suite is safe to run in a data-less CI.
    """
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")
    matches = sorted(TEST_DATA.glob(pattern))
    for m in matches:
        if must_contain is None:
            return m
        try:
            if must_contain in m.read_text(errors="ignore"):
                return m
        except OSError:
            continue
    extra = f" containing {must_contain!r}" if must_contain else ""
    pytest.skip(f"no test/ file matching {pattern!r}{extra}")


@pytest.fixture(scope="session")
def extractor(tmp_path_factory):
    """A CrystalPropertyExtractor backed by a throwaway DB in tmp (so no
    materials.db is created in the repo)."""
    from mace.utils.property_extractor import CrystalPropertyExtractor
    db = tmp_path_factory.mktemp("db") / "throwaway.db"
    return CrystalPropertyExtractor(db_path=str(db))


def energy_props(extractor, path: Path) -> dict:
    """Combined energy + frequency properties for a CRYSTAL .out file."""
    content = Path(path).read_text(errors="ignore")
    return {
        **extractor._extract_energy_properties(content),
        **extractor._extract_frequency_properties(content),
    }
