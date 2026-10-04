"""The k-path structure is the computed geometry BAND's fort.9 holds.

parse_crystal_structure took the cell from the first "PRIMITIVE CELL"
lattice parameters and the atoms from the first atom list. In an
optimisation output those are the initial geometry, not the optimised one;
after SUPERCEL the cell was the input primitive cell (FREQ
1_dia..._supercel222.out: a = 2.50656) while the atoms were the 16 of the
supercell (a = 5.01312). Cell and atoms now come from one computed block:
the one after the last FINAL OPTIMIZED GEOMETRY, otherwise the first
(GEOMETRY FOR WAVE FUNCTION).
"""
import contextlib
import io
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
si = pytest.importorskip("seekpath_interface")

import crystal_out_text as cot  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "test"
DIA_SUPER = CORPUS / "FREQ" / "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out"

# Diamond, the supercel222 numbers: primitive 2.50656, supercell 5.01312
DIA_PRIM = (2.50656, 2.50656, 2.50656, 60.0, 60.0, 60.0)
DIA_SUPER_CELL = (5.01312008, 5.01312008, 5.01312008, 60.0, 60.0, 60.0)
F_W = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))


def _dia_supercell_atoms():
    base = [(0.125, 0.125, 0.125), (-0.125, -0.125, -0.125)]
    atoms = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                for x, y, z in base:
                    atoms.append((6, "C", (x + i) / 2, (y + j) / 2, (z + k) / 2))
    return atoms


def _supercell_output(scelphono=False):
    return (cot.echo("F D 3 M", (3.54481, 3.54481, 3.54481, 90, 90, 90), DIA_PRIM)
            + cot.supercel(((2, 0, 0), (0, 2, 0), (0, 0, 2)), scelphono)
            + cot.block(DIA_SUPER_CELL, _dia_supercell_atoms(), centering="1/1"))


def _parse(path):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = si.parse_crystal_structure(str(path))
    return result, out.getvalue()


def test_supercel_output_reads_the_supercell(tmp_path):
    out = tmp_path / "super.out"
    out.write_text(_supercell_output())
    st, _ = _parse(out)
    assert len(st["numbers"]) == 16
    np.testing.assert_allclose(np.linalg.norm(st["cell"], axis=1), 5.01312008, atol=1e-6)
    assert st["supercell"] == "supercel"
    assert st["geometry"] == "first"
    np.testing.assert_array_equal(st["expansion"], 2 * np.eye(3))


def test_scelphono_is_told_from_supercel(tmp_path):
    """Both print the expansion matrix; SCELPHONO then reorders the atoms."""
    out = tmp_path / "scelphono.out"
    out.write_text(_supercell_output(scelphono=True))
    assert _parse(out)[0]["supercell"] == "scelphono"


def _ti_atoms(z_c):
    # one atom at the origin and a pair along c: I4/mmm in the primitive cell
    return [(20, "CA", 0.0, 0.0, 0.0), (6, "C", z_c, z_c, 0.0), (6, "C", -z_c, -z_c, 0.0)]


TI_W = ((0, 1, 1), (1, 0, 1), (1, 1, 0))


def _ti_prim(a, c):
    edge = np.sqrt(2 * a * a + c * c) / 2
    cos_ab = (c * c - 2 * a * a) / (4 * edge * edge)   # between (-a+b+c)/2 and (a-b+c)/2
    cos_c = -c * c / (4 * edge * edge)                  # with (a+b-c)/2
    return (edge, edge, edge, np.degrees(np.arccos(cos_c)), np.degrees(np.arccos(cos_c)),
            np.degrees(np.arccos(cos_ab)))


def _opt_output(final=True):
    # an optimisation from c < a (tI1) to c > a (tI2)
    first, last = (4.0, 3.0), (3.89, 6.38)
    text = cot.echo("I 4/M M M", (first[0], first[0], first[1], 90, 90, 90), _ti_prim(*first))
    text += cot.block(_ti_prim(*first), _ti_atoms(0.4), w=TI_W,
                      conventional=(first[0], first[0], first[1], 90, 90, 90), centering="6/0")
    text += cot.opt_step(_ti_prim(3.97, 3.6), _ti_atoms(0.4), w=TI_W,
                         conventional=(3.97, 3.97, 3.6, 90, 90, 90), centering="6/0")
    if final:
        text += cot.final_block(_ti_prim(*last), _ti_atoms(0.4), w=TI_W,
                                conventional=(last[0], last[0], last[1], 90, 90, 90),
                                centering="6/0")
    return text


def test_optimization_reads_the_final_geometry(tmp_path):
    out = tmp_path / "opt.out"
    out.write_text(_opt_output())
    st, _ = _parse(out)
    assert st["geometry"] == "final"
    np.testing.assert_allclose(np.linalg.norm(st["cell"], axis=1), _ti_prim(3.89, 6.38)[:3],
                               atol=1e-6)
    np.testing.assert_array_equal(st["W"], np.array(TI_W, dtype=float))
    assert st["supercell"] == "none" and st["expansion"] is None


def test_unconverged_optimization_keeps_the_first_geometry(tmp_path):
    """No FINAL OPTIMIZED GEOMETRY: the first block, as before (fort.9 would
    hold the last step - a known limitation)."""
    out = tmp_path / "opt_unconverged.out"
    out.write_text(_opt_output(final=False))
    st, _ = _parse(out)
    assert st["geometry"] == "first"
    np.testing.assert_allclose(np.linalg.norm(st["cell"], axis=1), _ti_prim(4.0, 3.0)[:3], atol=1e-6)


def test_atom_count_mismatch_falls_back_to_the_old_parser(tmp_path):
    out = tmp_path / "short.out"
    text = (cot.echo("F D 3 M", (3.54481, 3.54481, 3.54481, 90, 90, 90), DIA_PRIM)
            + cot.block(DIA_PRIM, [(6, "C", 0.125, 0.125, 0.125)], n_atoms=2, w=F_W,
                        conventional=(3.54481, 3.54481, 3.54481, 90, 90, 90)))
    out.write_text(text)
    st, printed = _parse(out)
    assert "WARNING" in printed and "lists 1 atoms where CRYSTAL prints 2" in printed
    assert st["geometry"] == "legacy"
    assert len(st["numbers"]) == 1


def test_no_block_falls_back_with_a_warning(tmp_path):
    out = tmp_path / "old.out"
    out.write_text(" LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL\n"
                   "       A          B          C        ALPHA      BETA     GAMMA        VOLUME\n"
                   "    2.50656    2.50656    2.50656    60.00000  60.00000  60.00000     11.135747\n\n"
                   " ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:    2\n"
                   "     ATOM                 X/A                 Y/B                 Z/C    \n"
                   + cot.STARS + "\n"
                   "      1 T   6 C     1.250000000000E-01  1.250000000000E-01  1.250000000000E-01\n"
                   "      2 F   6 C    -1.250000000000E-01 -1.250000000000E-01 -1.250000000000E-01\n\n")
    st, printed = _parse(out)
    assert "WARNING: no computed geometry block" in printed
    assert st["geometry"] == "legacy" and len(st["numbers"]) == 2


@pytest.mark.parametrize("w,expected", [
    # dia (F): the printed matrix is symmetric, the inverse of F's primitive one
    (F_W, None),
    # C-centred: printed "1 1 0 -1 1 0 0 0 1"; CRYSTAL's primitive vectors
    # are (a/2, b/2, 0), (-a/2, b/2, 0), c (DIRECT LATTICE VECTORS of
    # Ag1Cl5 C m), so a = p1 - p2: the conventional vectors are W's columns
    (((1, 1, 0), (-1, 1, 0), (0, 0, 1)), ((0.5, 0.5, 0), (-0.5, 0.5, 0), (0, 0, 1))),
    # R (hexagonal axes): printed "1 0 1 -1 1 1 0 -1 1"; CRYSTAL's primitive
    # vectors are (2a+b+c)/3, (-a+b+c)/3, (-a-2b+c)/3 (obverse)
    (((1, 0, 1), (-1, 1, 1), (0, -1, 1)),
     ((2 / 3, 1 / 3, 1 / 3), (-1 / 3, 1 / 3, 1 / 3), (-1 / 3, -2 / 3, 1 / 3))),
])
def test_w_direction(w, expected):
    """Rows: A_conv = W^T A_prim, i.e. A_prim = (W^T)^-1 A_conv."""
    w = np.array(w, dtype=float)
    prim_in_conv = np.linalg.inv(w.T)
    if expected is None:
        expected = ((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0))
    np.testing.assert_allclose(prim_in_conv, np.array(expected), atol=1e-12)


def _direct_vectors(lines, start):
    for i in range(start, len(lines)):
        if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS" in lines[i]:
            return np.array([[float(x) for x in lines[i + k].split()] for k in (2, 3, 4)])
    return None


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
def test_w_direction_against_printed_vectors():
    """Every centred corpus output: W^T times the printed primitive vectors
    is a cell with the printed crystallographic cell's parameters."""
    checked = 0
    for out in sorted(CORPUS.glob("*/*.out")):
        text = out.read_text(errors="ignore")
        if "TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC" not in text:
            continue
        lines = text.split("\n")
        blocks = [b for b in si._geometry_blocks(lines) if b["W"] is not None and b["conventional"]]
        if not blocks:
            continue
        block = blocks[0]
        prim = _direct_vectors(lines, block["line"])
        if prim is None:
            continue
        conv = block["W"].T @ prim
        lengths = np.linalg.norm(conv, axis=1)
        c = block["conventional"]
        np.testing.assert_allclose(lengths, [c["a"], c["b"], c["c"]], rtol=1e-5, err_msg=out.name)
        checked += 1
    assert checked >= 10


@pytest.mark.skipif(not DIA_SUPER.is_file(), reason="test/ corpus not present")
def test_corpus_supercel222():
    st, _ = _parse(DIA_SUPER)
    assert len(st["numbers"]) == 16 and st["supercell"] == "supercel"
    np.testing.assert_allclose(np.linalg.norm(st["cell"], axis=1), 5.01312008, atol=1e-6)


@pytest.mark.skipif(not (CORPUS / "OPT" / "1_dia_opt_rev1.out").is_file(),
                    reason="test/ corpus not present")
def test_corpus_optimization_final_cell():
    """1_dia_opt_rev1: the first block has V = 11.248833, the final 11.135747."""
    st, _ = _parse(CORPUS / "OPT" / "1_dia_opt_rev1.out")
    assert st["geometry"] == "final"
    assert abs(abs(np.linalg.det(st["cell"])) - 11.135747) < 1e-4


def test_library_route_takes_the_final_cell(tmp_path):
    """tI1 initial cell, tI2 optimised: SeeK-path sees the optimised one."""
    pytest.importorskip("seekpath")
    out = tmp_path / "opt.out"
    out.write_text(_opt_output())
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, info = si.get_accurate_bandpath(str(out))
    assert info["bravais_lattice_extended"] == "tI2"
