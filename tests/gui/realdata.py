"""Real CRYSTAL data for the viewer tests (polish area A): copied read-only
from the gitignored test/ corpus into a pytest temp dir under neutral names,
so no research name or user path reaches a fixture, a screenshot or a log.
Every helper skips when the corpus (or one file) is absent, as find_data does."""
import shutil
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[2] / "test"

# neutral name -> (glob under test/, what it is for)
FILES = {
    # long a = 19.5 A: the clipping case at range 3
    "agte.cif": ("CIFs/AgTe_mp-568761_sg62_*_optimized.cif", "large orthorhombic cell"),
    # polyhedra, boundary atoms, a 3D bonded network
    "tipbo3.cif": ("CIFs/TiPbO3_mp-1106215_sg87_*_optimized.cif", "3D network"),
    # a non-periodic structure
    "molecule.cif": ("CIFs/1LiFSI-8Sulfolane-conf1_MOLECULE_*_optimized.cif", "molecule"),
    # a molecular crystal (S4Br4-type cages, 16 atoms) whose molecules cross the cell faces:
    # every component is 0-dimensional (ase rank determination), 14 bonds cross a face
    "molcrystal.cif": ("CIFs/Br4S4_sym_*_optimized.cif", "molecular crystal"),
    # a large slab optimization (2D periodic), the classic OPT viewer
    "slab_opt.out": ("OPT/4LG_FSI_2x2_ABAB_opt_HSESOL3C_Attempt2_SLAB_OPT_*_optimized.out", "slab trajectory"),
}


def _guard(pattern):
    from conftest import corpus_guard
    corpus_guard(pattern)


def source(name):
    _guard(FILES[name][0])
    if not CORPUS.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")
    pattern = FILES[name][0]
    hits = sorted(CORPUS.glob(pattern))
    if not hits:
        pytest.skip(f"no test/ file matching {pattern!r}")
    return hits[0]


def copy(dest: Path, *names, extra=None):
    """Copy the named real files into dest (created); extra = {name: Path}."""
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copyfile(source(n), dest / n)
    for n, p in (extra or {}).items():
        shutil.copyfile(p, dest / n)
    return dest


def first(pattern, must_contain=None):
    """The first corpus file matching pattern (optionally holding a string), else skip."""
    _guard(pattern)
    if not CORPUS.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")
    for p in sorted(CORPUS.glob(pattern)):
        if must_contain is None:
            return p
        try:
            if must_contain in p.read_text(errors="ignore"):
                return p
        except OSError:
            continue
    pytest.skip(f"no test/ file matching {pattern!r}")
