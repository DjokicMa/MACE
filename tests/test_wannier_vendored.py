"""The vendored lcao2wannier copy (William Comaskey, MIT) inside MACE.

Two things are checked here: that the copy under ``mace/wannier/lcao2wannier``
is self-contained (it never reaches for a separately installed lcao2wannier),
and the one functional change made to it - the matrix-header regexes now read
CRYSTAL's I4 cell index past 999.

CRYSTAL writes the direct-lattice cell index as I4, so from 1000 on the header
has no space after ``N.``::

     OVERLAP MATRIX - CELL N. 999(  4  3 -6)
     OVERLAP MATRIX - CELL N.1000( -4  1 -3)

The stock 1.0.0 patterns required ``\\s+`` there, so every cell from 1000 on
was skipped - and because an unmatched header does not end the previous
block, its rows were written into cell 999's matrix. MEASURED on the corpus
diamond at N = 1247: 999 of 1247 cells read, no warning.

The snippets below are synthetic by necessity (the smallest real dump past
cell 999 is hundreds of MB), but reproduce CRYSTAL's field widths exactly.
"""
import os
import pickle
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

np = pytest.importorskip("numpy")

from mace.wannier.lcao2wannier import parser as l2w_parser  # noqa: E402

# (cell index, lattice vector, overlap value). The last one also exercises
# the lattice-vector I3 fields running together (" 12-11  0"), which the
# same whitespace assumption would have dropped at a large enough radius.
CELLS = [
    (999, (4, 3, -6), 0.25),
    (1000, (-4, 1, -3), 0.50),
    (1247, (12, -11, 0), 0.75),
]


def _header(kind, index, vec):
    return f" {kind} - CELL N.{index:>4}({vec[0]:>3}{vec[1]:>3}{vec[2]:>3})"


def _block(value):
    return [
        "",
        "                 1              2",
        "",
        f"   1     {value:.7E}",
        f"   2     {value / 10:.7E}  {value:.7E}",
        "",
    ]


def _dump_lines():
    lines = [" SYNTHETIC MATRIX DUMP (CRYSTAL field widths)", ""]
    for index, vec, value in CELLS:
        lines.append(_header("OVERLAP MATRIX", index, vec))
        lines += _block(value)
    lines.append("   ALPHA_ALPHA ELECTRONS")
    for index, vec, value in CELLS:
        lines.append(_header("FOCK MATRIX (REAL PART)", index, vec))
        lines += _block(-value)
        lines.append(_header("FOCK MATRIX (IMAG PART)", index, vec))
        lines += _block(value / 100)
    return [line + "\n" for line in lines]


def test_the_headers_are_exactly_crystals_i4_form():
    assert _header("OVERLAP MATRIX", 999, (4, 3, -6)) == \
        " OVERLAP MATRIX - CELL N. 999(  4  3 -6)"
    assert _header("OVERLAP MATRIX", 1000, (-4, 1, -3)) == \
        " OVERLAP MATRIX - CELL N.1000( -4  1 -3)"
    assert _header("OVERLAP MATRIX", 1247, (12, -11, 0)) == \
        " OVERLAP MATRIX - CELL N.1247( 12-11  0)"


@pytest.mark.parametrize("pattern, header, groups", [
    ("overlap_header_pattern",
     " OVERLAP MATRIX - CELL N.1000( -4  1 -3)", ("-4", "1", "-3")),
    ("fock_header_pattern",
     " FOCK MATRIX (REAL PART) - CELL N.1000( -4  1 -3)", ("REAL", "-4", "1", "-3")),
    ("fock_simple_header_pattern",
     " FOCK MATRIX - CELL N.1247( 12-11  0)", ("12", "-11", "0")),
    ("overlap_header_pattern",
     " OVERLAP MATRIX - CELL N.   1(  0  0  0)", ("0", "0", "0")),
])
def test_every_header_pattern_reads_a_four_digit_cell_index(pattern, header, groups):
    match = getattr(l2w_parser, pattern).match(header)
    assert match is not None, f"{pattern} dropped {header!r}"
    assert match.groups() == groups


def test_the_list_parser_keeps_cells_999_1000_and_1247():
    """Reintroducing ``\\s+`` drops 1000 and 1247 and corrupts 999."""
    matrices, _ = l2w_parser.parse_overlap_and_fock_matrices(_dump_lines())
    overlaps = {tuple(m["lattice_vector"]): m["data"]
                for m in matrices if m["type"] == "overlap"}
    focks = {tuple(m["lattice_vector"]): m["data"]
             for m in matrices if m["type"] == "fock"}
    expected = {vec: value for _, vec, value in CELLS}
    assert set(overlaps) == set(expected)
    assert set(focks) == set(expected)
    for vec, value in expected.items():
        # each cell holds its OWN rows, not a later cell's
        assert overlaps[vec][0, 0] == pytest.approx(value)
        assert overlaps[vec][1, 0] == pytest.approx(value / 10)
        assert focks[vec][0, 0] == pytest.approx(complex(-value, value / 100))


def test_the_streaming_parser_keeps_them_too(tmp_path):
    dump = tmp_path / "dump.out"
    dump.write_text("".join(_dump_lines()))
    H, S, _, _ = l2w_parser.parse_overlap_and_fock_matrices_streaming(str(dump))
    expected = {vec for _, vec, _ in CELLS}
    assert set(S) == expected
    assert set(H) == expected
    assert S[(-4, 1, -3)][0, 0] == pytest.approx(0.50)


def test_a_parse_cache_from_the_stock_regexes_is_not_reused(tmp_path):
    """A ``.parsecache.pkl`` written by stock 1.0.0 is keyed on (mtime, size)
    only, so after the regex fix it would still match and hand back the
    truncated cell list. The vendored key carries a tag that such a cache
    cannot have."""
    dump = tmp_path / "dump.out"
    lines = _dump_lines()
    dump.write_text("".join(lines))
    st = os.stat(dump)
    stale = tmp_path / "dump.out.parsecache.pkl"
    with open(stale, "wb") as fh:
        pickle.dump(((st.st_mtime_ns, st.st_size), ["TRUNCATED"], None), fh)

    raw, _ = l2w_parser.parse_overlap_and_fock_matrices_cached(str(dump), lines)
    assert raw != ["TRUNCATED"]
    assert len([m for m in raw if m["type"] == "overlap"]) == len(CELLS)


def test_the_vendored_copy_never_imports_an_installed_lcao2wannier():
    """Every module imports from its own package. A leftover absolute
    ``from lcao2wannier...`` would silently bind to a pip-installed copy (with
    the unfixed parser) or fail where none is installed."""
    code = (
        "import importlib, pkgutil, sys\n"
        "import mace.wannier.lcao2wannier as pkg\n"
        "for m in pkgutil.iter_modules(pkg.__path__):\n"
        "    if m.name != '__main__':\n"
        "        importlib.import_module(pkg.__name__ + '.' + m.name)\n"
        "print(sorted(k for k in sys.modules if k.split('.')[0] == 'lcao2wannier'))\n"
    )
    pytest.importorskip("scipy")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_vendored_copy_keeps_its_license_and_provenance():
    here = REPO_ROOT / "mace" / "wannier" / "lcao2wannier"
    assert "MIT License" in (here / "LICENSE").read_text()
    assert "1.0.0" in (here / "VENDORED.md").read_text()
    for source in sorted(here.glob("*.py")) + sorted(here.glob("*.f90")):
        head = "".join(source.read_text().splitlines(keepends=True)[:3])
        assert "William Comaskey" in head and "MIT" in head, source.name

