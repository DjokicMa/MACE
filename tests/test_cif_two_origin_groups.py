"""A CIF in one of the 24 space groups with two origin choices, given by its
H-M symbol and number only (no operator loop), describes one structure.

ase expands such a CIF in origin choice 1, and CRYSTAL reads IFSO 0
("0 0 0") as origin choice 2 and IFSO 1 ("0 0 1") as origin choice 1
(manual p. 19). Before this was handled for every group (only Fd-3m was),
FeSe P4/nmm in origin choice 2 became 12 atoms in the P1 deck (Fe4Se8,
1.48 A contacts), anatase 24 atoms with 0.41 A contacts, and an origin
choice 1 FeSe CIF was written under "0 0 0", which CRYSTAL reads in origin
choice 2 - a different structure, with no warning.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase", reason="CIF parsing needs ase")
pytest.importorskip("spglib", reason="origin detection checks the space group with spglib")
import NewCifToD12  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES, MULTI_ORIGIN_SPACEGROUPS  # noqa: E402

from mace.gui import structures  # noqa: E402

FESE = """data_FeSe
_symmetry_space_group_name_H-M   '{hm}'
_symmetry_Int_Tables_number   129
{extra}_cell_length_a   3.7738
_cell_length_b   3.7738
_cell_length_c   5.5248
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}"""

ORIGIN2_ROWS = "Fe1 Fe 0.75 0.25 0.0\nSe1 Se 0.25 0.25 0.2672\n"
ORIGIN1_ROWS = "Fe1 Fe 0.0 0.0 0.0\nSe1 Se 0.0 0.5 0.2328\n"

ANATASE = """data_TiO2
_symmetry_space_group_name_H-M   'I 41/a m d'
_symmetry_Int_Tables_number   141
_cell_length_a   3.7845
_cell_length_b   3.7845
_cell_length_c   9.5143
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 Ti 0.0 0.75 0.125
O1 O 0.0 0.25 0.0817
"""


def _options(mode, **extra):
    return dict(dimensionality="CRYSTAL", calculation_type="SP",
                basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2", method="DFT",
                dft_functional="PBE", is_spin_polarized=False,
                tolerances=DEFAULT_TOLERANCES, scf_method="DIIS",
                symmetry_handling=mode, **extra)


def _deck(tmp_path, text, mode, **extra):
    src = tmp_path / f"src_{mode}"
    src.mkdir()
    (src / "x.cif").write_text(text)
    out = tmp_path / f"out_{mode}"
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        written, found = NewCifToD12.process_cifs(str(src), _options(mode, **extra), str(out),
                                                  interactive=False)
    decks = list(out.glob("*.d12")) if out.exists() else []
    return (decks[0].read_text().splitlines() if decks else None), buf.getvalue()


def _atoms(lines):
    n = int(lines[5])
    return [ln.split() for ln in lines[6:6 + n]]


def _min_contact(lines):
    a, c = float(lines[4].split()[0]), float(lines[4].split()[-1])
    frac = np.array([[float(v) for v in r[1:4]] for r in _atoms(lines)])
    best = np.inf
    for i in range(len(frac)):
        d = frac - frac[i]
        d -= np.round(d)
        r = np.linalg.norm(d * [a, a, c], axis=1)
        r[i] = np.inf
        best = min(best, r.min())
    return best


@pytest.mark.parametrize("rows", [ORIGIN2_ROWS, ORIGIN1_ROWS], ids=["origin2", "origin1"])
def test_fese_p1_deck_is_four_atoms(tmp_path, rows):
    lines, _ = _deck(tmp_path, FESE.format(hm="P 4/n m m", extra="", rows=rows), "P1")
    atoms = _atoms(lines)
    assert sorted(r[0] for r in atoms) == ["26", "26", "34", "34"]
    assert _min_contact(lines) > 2.3   # Fe-Se ~2.4 A; the wrong origin gave 1.48 A


@pytest.mark.parametrize("rows,header", [(ORIGIN2_ROWS, "0 0 0"), (ORIGIN1_ROWS, "0 0 1")],
                         ids=["origin2", "origin1"])
def test_fese_symmetric_deck_declares_the_cifs_origin(tmp_path, rows, header):
    for mode in ("CIF", "SPGLIB"):
        lines, log = _deck(tmp_path, FESE.format(hm="P 4/n m m", extra="", rows=rows), mode)
        assert lines[2] == header, mode
        assert lines[3] == "129"
        assert len(_atoms(lines)) == 2
        assert "mismatch" not in log.lower()


def test_anatase_p1_deck_is_twelve_atoms(tmp_path):
    lines, _ = _deck(tmp_path, ANATASE, "P1")
    assert len(_atoms(lines)) == 12
    assert _min_contact(lines) > 1.8   # Ti-O ~1.93 A; the wrong origin gave 0.41 A


@pytest.mark.parametrize("hm,extra,expected", [
    ("P 4/n m m :1", "", 1), ("P 4/n m m :2", "", 2),
    ("P 4/n m m S", "", 1), ("P 4/n m m Z", "", 2),
    ("P 4/n m m", "_space_group.IT_coordinate_system_code '1'\n", 1),
    ("P 4/n m m", "_space_group.IT_coordinate_system_code 2\n", 2),
])
def test_origin_tags_are_read(hm, extra, expected):
    text = FESE.format(hm=hm, extra=extra, rows=ORIGIN2_ROWS)
    assert NewCifToD12.cif_origin_tag(text)[0] == expected


def test_origin_tag_wins_over_the_atoms(tmp_path):
    """The tag says origin choice 1: the deck is written under "0 0 1"
    with the records as given, even though the atoms alone read as 2."""
    text = FESE.format(hm="P 4/n m m", extra="_space_group.IT_coordinate_system_code '1'\n",
                       rows=ORIGIN1_ROWS)
    lines, _ = _deck(tmp_path, text, "CIF")
    assert lines[2] == "0 0 1"


def test_explicit_origin_setting_overrides(tmp_path):
    lines, _ = _deck(tmp_path, FESE.format(hm="P 4/n m m", extra="", rows=ORIGIN1_ROWS),
                     "P1", origin_setting="ALTERNATE")
    assert len(_atoms(lines)) == 4


def test_listed_operators_decide_the_origin(tmp_path):
    """A CIF with an origin choice 1 operator loop is written under "0 0 1"."""
    from ase.spacegroup import Spacegroup
    ops = Spacegroup(129, setting=1).get_symop()

    def xyz(rot, trans):
        out = []
        for row, t in zip(rot, trans):
            term = "".join(("+" if v > 0 else "-") + ax for v, ax in zip(row, "xyz") if v)
            if abs(t) > 1e-6:
                term += f"+{t:.4f}"
            out.append(term.lstrip("+"))
        return ",".join(out)

    loop = "loop_\n_symmetry_equiv_pos_as_xyz\n" + "".join(
        f"'{xyz(r, t)}'\n" for r, t in ops)
    text = FESE.format(hm="P 4/n m m", extra=loop, rows=ORIGIN1_ROWS)
    lines, _ = _deck(tmp_path, text, "CIF")
    assert lines[2] == "0 0 1"


def test_every_two_origin_group_is_handled():
    """The GUI viewer and cif2d12 cover the same 24 groups."""
    assert set(MULTI_ORIGIN_SPACEGROUPS) == set(structures._TWO_ORIGINS)
    assert len(MULTI_ORIGIN_SPACEGROUPS) == 24


def test_ambiguous_origin_is_refused():
    """Two origin readings that build different structures of the same size
    (and both in the CIF's group) are refused, not guessed."""
    from unittest import mock

    with mock.patch.object(NewCifToD12, "_expand_in_origin") as exp, \
            mock.patch.object(NewCifToD12, "_structure_fingerprint", side_effect=[1, 2]):
        from ase import Atoms
        exp.side_effect = lambda *a, **k: Atoms("Fe4", scaled_positions=[
            [0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], cell=[4, 4, 4], pbc=True)
        with mock.patch.object(NewCifToD12.spglib, "get_symmetry_dataset") as ds:
            ds.return_value = type("D", (), {"number": 129})()
            with pytest.raises(NewCifToD12.AmbiguousOriginError, match="IT_coordinate_system_code"):
                NewCifToD12._origin_from_expansion(["Fe"], [[0, 0, 0]], 129,
                                                   (4, 4, 4, 90, 90, 90))
