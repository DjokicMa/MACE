# Vendored: lcao2wannier

This directory is a copy of **lcao2wannier** by **William Comaskey**, bundled
with MACE under its own MIT license (`LICENSE`, this directory). The
LCAO->Wannier90 method and this package are his work. MACE drives it through
`mace wannier` (`mace/wannier/driver.py`), which runs it as
`python -m mace.wannier.lcao2wannier`.

## Source

| | |
|---|---|
| Archive | `LCAO-to-Wannier-release-v1.0.zip` (sha256 `05764988bc06acb25d1290fa7da7e0b738afc70183998e7871c9fd92eca045d0`) |
| Package version | `1.0.0` (`pyproject.toml` `version`, `lcao2wannier/__init__.py` `__version__`) |
| Vendored on | 2026-09-30 |
| Upstream URL | none confirmed public when vendored |

## What was copied

From `lcao2wannier/` in the archive: all 30 `.py` modules, the two optional
Fortran kernel sources (`_spread_fortran.f90`, `_disentangle_fortran.f90`) and
their f2py type map (`.f2py_f2cmap`, needed to build them). From the archive
root: `LICENSE`.

Not copied: the `lcao_wannier` compatibility shim and the top-level
`lcao_to_wannier90.py` legacy script (both only re-export this package under
older names), `scripts/`, `tests/` (including the 9.5 MB `Bismuth_basis_40.out`
reference dump and its 9 MB `.parsecache.pkl`), `examples/`, `docs/`, the
Markdown guides, `build/`, `*.egg-info`, `__pycache__/`, generated `.win`/
`.amn`/`.eig`/`.mmn` outputs, and the `__MACOSX/` archive metadata. No compiled
binary is included; the Fortran kernels are optional and the package falls back
to pure Python without them.

The `LICENSE` names "Computational Materials Science Team" as the copyright
holder; `pyproject.toml` and `__init__.py` name William Comaskey as the author.
Both are kept as shipped.

## Local modifications

Every change to his files is listed here. Nothing was reformatted or refactored.

1. **Copyright header** (all 30 `.py` files and both `.f90` files). None of the
   files carried one. Three comment lines were added at the top (after the
   `#!` line in `workflow.py`):

   ```
   # Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
   # MIT License, see LICENSE in this directory.
   # Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
   ```

   (`!` instead of `#` in the Fortran files.) All line numbers below are in
   the headed files, i.e. upstream line + 2.

2. **Absolute self-imports made relative.** As `mace.wannier.lcao2wannier`, a
   bare `from lcao2wannier...` would fail, or silently bind to a separately
   installed (unfixed) copy. Each `from lcao2wannier import X` became
   `from . import X` and each `from lcao2wannier.M import X` became
   `from .M import X`; nothing else on those lines changed.
   - `engine.py`: 544
   - `lcao_pdwf.py`: 247
   - `workflow.py`: 42, 55-61 and 64 (the top-level imports), 129, 440, 484-485,
     797, 849, 956, 990, 1038, 1074, 1235, 1239, 1462, 1528-1529, 1922, 2026,
     2177, 2233, 2363, 2370, 2388, 2872, 2960, 2964-2965, 2968, 2972
     (37 lines in all)

   Consequence: `workflow.py` can no longer be run as a plain script
   (`python workflow.py`); use `python -m mace.wannier.lcao2wannier.workflow`,
   or the canonical CLI via `mace wannier`.

3. **Cell index >= 1000 (bug fix), `parser.py` lines 21, 24, 28.** CRYSTAL
   writes the direct-lattice cell index as I4, so from 1000 on the header has
   no space after `N.` (`OVERLAP MATRIX - CELL N.1000( -4  1 -3)`). The three
   header patterns (`overlap_header_pattern`, `fock_header_pattern`,
   `fock_simple_header_pattern`) required `N\.\s+\d+\(`, so every cell from
   1000 on was skipped - and because an unmatched header does not end the
   previous block, that cell's rows were written into cell 999's matrix.
   MEASURED on a diamond dump at N = 1247: 999 of 1247 cells read, no warning.
   Changed `N\.\s+\d+\(` to `N\.\s*\d+\(`.

   The same patterns separate the three lattice-vector components with `\s+`,
   but CRYSTAL prints them as I3, so a component of -10 or below runs into its
   neighbour (`( 12-11  0)`). That form has not been observed in a real dump;
   it follows from the field width and appears only at large cell radii (big
   2-D supercells). Those separators are `\s*` now too. Every header the old
   patterns matched is matched identically by the new ones (the `-?\d+` groups
   are greedy), so this only adds headers that were being dropped.

4. **Parse-cache key, `parser.py` lines 769-771.** `parse_overlap_and_fock_matrices_cached`
   keyed `<input>.parsecache.pkl` on `(mtime_ns, size)` only, so a cache
   written by the unfixed patterns would still match after fix 3 and return
   the truncated cell list. The key now starts with the tag `'cell-index-i4'`,
   which no stock-1.0.0 cache has, so those are re-parsed once.

Tests for 2-4: `tests/test_wannier_vendored.py` (synthetic headers at cells
999, 1000 and 1247; stale-cache rejection; no `lcao2wannier` import leak).

## Known, deliberately left alone

- `cli.py` `_source_revision()` runs `git rev-parse` two directories up. Inside
  MACE that reports MACE's commit, not an lcao2wannier one; `--version` still
  shows `lcao2wannier 1.0.0`.
- Messages in `spread.py`/`hybrid.py` suggest `scripts/build_spread_fortran.sh`
  and `scripts/build_disentangle_fortran.sh`, which are not bundled. The
  equivalent, run in this directory (needs gfortran, meson, ninja, BLAS):

  ```
  python -m numpy.f2py -c --backend meson -m _spread_fortran _spread_fortran.f90 -lopenblas
  python -m numpy.f2py -c --backend meson -m _disentangle_fortran _disentangle_fortran.f90 -lopenblas
  ```

  Compiled `*.so` files must not be committed.

## Updating

Copy the new release over these files, re-apply the headers and change 2,
drop change 3/4 if upstream has fixed the patterns (keep the tests), and
update the version, date and this list.
