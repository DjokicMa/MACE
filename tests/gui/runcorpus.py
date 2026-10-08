"""Real CRYSTAL outputs for the run-analysis tests (Area C), copied read-only
from the gitignored test/ corpus into a temp dir. Tests that need them skip
when the corpus is absent; nothing from it is committed (files are found at
test time by name pattern or by property)."""
import shutil
from pathlib import Path

import pytest

import guiharness as H

CORPUS = H.REPO_ROOT / "test"
DIA_OPT = CORPUS / "OPT" / "1_dia_opt_rev1.out"


def need(path: Path) -> Path:
    if not Path(path).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    return Path(path)


def copy_into(dst_dir: Path, *paths: Path) -> list:
    """Copy each corpus file into dst_dir (never opened in place), with its mtime."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        need(p)
        q = dst_dir / Path(p).name
        shutil.copy2(p, q)          # the mtime too: a finished run is not "recently written"
        out.append(q)
    return out


def corpus_outs(*subdirs: str) -> list:
    """Every .out under the given corpus subfolders (sorted); skips when absent."""
    if not CORPUS.is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    out = []
    for s in subdirs:
        out += sorted((CORPUS / s).glob("*.out"))
    return out
