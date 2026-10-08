"""The MACE logo set (docs/logo/) and the copies the GUI serves (mace/gui/static/brand/)."""
import re

import pytest

from conftest import REPO_ROOT

LOGO = REPO_ROOT / "docs" / "logo"
BRAND = REPO_ROOT / "mace" / "gui" / "static" / "brand"


@pytest.mark.parametrize("name", sorted(p.name for p in BRAND.iterdir()))
def test_gui_copy_is_byte_identical_to_docs_logo(name):
    assert (BRAND / name).read_bytes() == (LOGO / name).read_bytes(), name


@pytest.mark.parametrize("name", sorted(p.name for p in LOGO.glob("*.svg")))
def test_svgs_are_self_contained_and_labeled(name):
    text = (LOGO / name).read_text()
    assert 'role="img"' in text and re.search(r'aria-label="MACE[^"]*"', text) and "<title>MACE" in text
    assert "<text" not in text and "@import" not in text and "<image" not in text
    assert not re.search(r'(?:href|src)="(?:https?:)?//', text)
    assert re.search(r'\bwidth="[\d.]+"', text) and re.search(r'\bheight="[\d.]+"', text)


def test_gui_and_readme_use_the_logo():
    index = (REPO_ROOT / "mace" / "gui" / "static" / "index.html").read_text()
    for ref in ("brand/icon.svg", "brand/favicon-32.png", "brand/favicon-16.png"):
        assert f"static/@V@/{ref}" in index, ref
    assert 'aria-label="MACE"' in index
    assert not (REPO_ROOT / "mace" / "gui" / "static" / "favicon.svg").exists()
    readme = (REPO_ROOT / "README.md").read_text()
    assert 'srcset="docs/logo/hero-on-dark.svg"' in readme and 'src="docs/logo/hero.svg" alt="MACE"' in readme
    shell = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "shell.js").read_text()
    for name in ("hero.svg", "hero-on-dark.svg", "lockup.svg", "lockup-on-dark.svg"):
        assert name in shell and (BRAND / name).is_file(), name


def test_logo_guide_names_only_files_that_exist():
    guide = (LOGO / "README.md").read_text()
    for name in set(re.findall(r"`([\w-]+\.(?:svg|png))`", guide)):
        assert (LOGO / name).is_file(), name
