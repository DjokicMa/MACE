"""The CLI and the GUI must never read a structure differently.

For every CIF in the test/ corpus (each distinct file once) and for synthetic
all-caps, label-only, origin-choice and POSCAR cases:

* CLI: NewCifToD12.parse_cif, then the deck's own symmetry handling
  (P1: expand_to_p1; CIF: verify_and_reduce_to_asymmetric_unit), then the
  deck written by create_d12_file and expanded again by the GUI's d12
  reader (asymmetric unit + space group, the cell CRYSTAL builds).
* GUI: mace.gui.structures.read_cif (its P1 fast path or ase's path).
* mace preopt: mace.preopt.cli.read_input, against the GUI (corpus) and the
  CLI deck (synthetic cases).

Both must give the same cell, the same number of atoms and the same element
on every site (fractional positions within 1e-3, modulo lattice vectors).
No corpus CIF is exempt today; a file that must differ for a documented
reason goes in KNOWN_DIFFERENT with that reason, never a silent skip.

The corpus is read from tests/conftest.TEST_DATA (``<repo>/test``), or from
$MACE_TEST_DATA when set (a worktree has no test/); without it those cases
skip. All of it (about 190 distinct CIFs, both modes) takes about 15 s.
"""

import contextlib
import hashlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase", reason="CIF parsing needs ase")
pytest.importorskip("spglib", reason="the CIF-symmetry deck path needs spglib")

import NewCifToD12  # noqa: E402
from conftest import TEST_DATA  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES  # noqa: E402

from mace.gui import structures  # noqa: E402

# name -> reason the CLI and GUI legitimately differ for that corpus CIF.
KNOWN_DIFFERENT = {}

MODES = ("P1", "CIF")


def _corpus_root():
    env = os.environ.get("MACE_TEST_DATA")
    root = Path(env) if env else TEST_DATA
    return root if root.is_dir() else None


def _corpus_cifs():
    root = _corpus_root()
    if root is None:
        return [pytest.param(None, id="no-corpus")]
    seen = {}
    for p in sorted(root.rglob("*.cif")):
        try:
            seen.setdefault(hashlib.sha1(p.read_bytes()).hexdigest(), p)
        except OSError:
            continue
    return [pytest.param(p, id=str(p.relative_to(root))) for p in seen.values()]


def _options(mode):
    return dict(dimensionality="CRYSTAL", calculation_type="SP",
                basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2", method="DFT",
                dft_functional="PBE", is_spin_polarized=False,
                tolerances=DEFAULT_TOLERANCES, scf_method="DIIS",
                symmetry_handling=mode)


def _cli_structure(cif: Path, mode: str, tmp_path: Path):
    """The cell the cif2d12 deck describes, read back by the GUI's d12 reader."""
    deck = tmp_path / f"{mode}.d12"
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        data = NewCifToD12.parse_cif(str(cif), interactive=False)
        options = _options(mode)
        if mode == "P1":
            data = NewCifToD12.expand_to_p1(data, options)
            data["spacegroup"] = 1
        else:
            data = NewCifToD12.verify_and_reduce_to_asymmetric_unit(
                data, 1e-5, False, interactive=False)
        assert NewCifToD12.create_d12_file(data, str(deck), options, interactive=False)
        return structures.read_d12(deck)


def _gui_structure(path: Path, fmt: str = "cif"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return structures.read_cif(path) if fmt == "cif" else structures.read_any(path, fmt)


def _assert_same(cli, gui, tol=1e-3):
    from ase.geometry import cell_to_cellpar
    np.testing.assert_allclose(cell_to_cellpar(np.array(cli["cell"])),
                               cell_to_cellpar(np.array(gui["cell"])), atol=1e-3,
                               err_msg="cell differs")
    zc, zg = np.array(cli["numbers"]), np.array(gui["numbers"])
    assert len(zc) == len(zg), f"CLI deck has {len(zc)} atoms, GUI {len(zg)}"
    assert sorted(zc.tolist()) == sorted(zg.tolist()), "element counts differ"
    fc, fg = np.array(cli["frac"], float), np.array(gui["frac"], float)
    used = np.zeros(len(fg), bool)
    for z, f in zip(zc, fc):
        d = f - fg
        d -= np.round(d)
        dist = np.linalg.norm(d, axis=1)
        dist[(zg != z) | used] = np.inf
        k = int(np.argmin(dist))
        assert dist[k] <= tol, f"no GUI site with Z={z} at {np.round(f, 4)} (nearest {dist[k]:.3g})"
        used[k] = True


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cif", _corpus_cifs())
def test_corpus_cif_reads_the_same(cif, mode, tmp_path):
    if cif is None:
        pytest.skip("test/ data corpus not present (gitignored, ~12GB); set MACE_TEST_DATA")
    if cif.name in KNOWN_DIFFERENT:
        pytest.xfail(KNOWN_DIFFERENT[cif.name])
    _assert_same(_cli_structure(cif, mode, tmp_path), _gui_structure(cif))


# ------------------------------------------------------------ synthetic cases

HEAD = """data_t
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma {gamma}
_space_group_name_H-M_alt '{hm}'
_space_group_IT_number {no}
loop_
_atom_site_label
{type_col}_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"""


def _write(tmp_path, name, rows, a=5.0, b=5.0, c=5.0, gamma=90, hm="P 1", no=1, typed=True):
    text = HEAD.format(a=a, b=b, c=c, gamma=gamma, hm=hm, no=no,
                       type_col="_atom_site_type_symbol\n" if typed else "")
    text += "\n".join(" ".join(map(str, r)) for r in rows) + "\n"
    p = tmp_path / name
    p.write_text(text)
    return p


SYNTH = {
    "all_caps_type": dict(rows=[("BI1", "BI", 0, 0, 0), ("B2", "B", .5, .5, .5),
                                ("CL1", "CL", .25, .25, .25), ("SN1", "SN4+", .75, .75, .75)]),
    "label_only": dict(rows=[("BI1", 0, 0, 0), ("B2", .5, .5, .5),
                             ("CL1", .25, .25, .25), ("SN1", .75, .75, .75)], typed=False),
    "lower_case_type": dict(rows=[("x1", "bi", 0, 0, 0), ("x2", "b", .5, .5, .5)]),
    # Fd-3m with no operators listed: diamond at 1/8 is origin choice 2 (8 atoms);
    # ase alone would expand it in origin choice 1 (16 overlapping atoms).
    "diamond_origin2": dict(rows=[("C1", "C", .125, .125, .125)], a=3.567, b=3.567, c=3.567,
                            hm="F d -3 m", no=227),
    # The other two-origin groups, H-M symbol only (no operators, no origin
    # tag): FeSe P4/nmm in origin choice 2 (4 atoms; ase alone builds 12) and
    # in origin choice 1 (the deck must declare "0 0 1"), and anatase I41/amd
    # in origin choice 2 (12 atoms; ase alone builds 24 with 0.41 A contacts).
    "fese_origin2": dict(rows=[("Fe1", "Fe", .75, .25, 0), ("Se1", "Se", .25, .25, .2672)],
                         a=3.7738, b=3.7738, c=5.5248, hm="P 4/n m m", no=129),
    "fese_origin1": dict(rows=[("Fe1", "Fe", 0, 0, 0), ("Se1", "Se", 0, .5, .2328)],
                         a=3.7738, b=3.7738, c=5.5248, hm="P 4/n m m", no=129),
    "anatase_origin2": dict(rows=[("Ti1", "Ti", 0, .75, .125), ("O1", "O", 0, .25, .0817)],
                            a=3.7845, b=3.7845, c=9.5143, hm="I 41/a m d", no=141),
    # P63/mmc written to 6 decimals, which spglib at 1e-5 reads as Cmcm: the
    # deck keeps the CIF group and must then hold the CIF's 3 records, not
    # the 12 expanded atoms CRYSTAL would generate again on top of themselves.
    "p63mmc_6_decimals": dict(rows=[("Ag1", "Ag", 0.170267, -0.170267, -0.25),
                                    ("Ag2", "Ag", 0, 0, 0), ("Cl1", "Cl", 0.333333, -0.333333, 0.009755)],
                              a=5.899731, b=5.899731, c=7.765412, gamma=120, hm="P63/mmc", no=194),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(SYNTH))
def test_synthetic_cif_reads_the_same(case, mode, tmp_path):
    cif = _write(tmp_path, f"{case}.cif", **SYNTH[case])
    _assert_same(_cli_structure(cif, mode, tmp_path), _gui_structure(cif))


def test_diamond_origin2_is_eight_atoms(tmp_path):
    cif = _write(tmp_path, "dia.cif", **SYNTH["diamond_origin2"])
    gui = _gui_structure(cif)
    assert len(gui["numbers"]) == 8


@pytest.mark.parametrize("case,count", [("fese_origin2", 4), ("fese_origin1", 4),
                                        ("anatase_origin2", 12)])
def test_two_origin_cif_gui_atom_count(tmp_path, case, count):
    cif = _write(tmp_path, f"{case}.cif", **SYNTH[case])
    assert len(_gui_structure(cif)["numbers"]) == count


def test_p63mmc_mismatch_deck_holds_the_cif_records(tmp_path):
    cif = _write(tmp_path, "ag.cif", **SYNTH["p63mmc_6_decimals"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = NewCifToD12.parse_cif(str(cif), interactive=False)
        kept = NewCifToD12.verify_and_reduce_to_asymmetric_unit(data, 1e-5, False, interactive=False)
    assert kept["symbols"] == ["Ag", "Ag", "Cl"]
    assert kept["spacegroup"] == 194


def test_poscar_all_caps_reads_like_the_cif(tmp_path):
    """The GUI's POSCAR reader against the CLI deck of the same structure."""
    cif = _write(tmp_path, "t.cif", **SYNTH["all_caps_type"])
    poscar = tmp_path / "POSCAR"
    poscar.write_text("t\n1.0\n5 0 0\n0 5 0\n0 0 5\nBI B CL SN\n1 1 1 1\nDirect\n"
                      "0 0 0\n.5 .5 .5\n.25 .25 .25\n.75 .75 .75\n")
    _assert_same(_cli_structure(cif, "P1", tmp_path), _gui_structure(poscar, "poscar"))


# ------------------------------------------------------------ mace preopt


def _preopt_structure(path: Path):
    from mace.preopt.cli import read_input
    atoms, _, _, _ = read_input(path)
    return {"cell": atoms.cell.array, "numbers": atoms.numbers,
            "frac": atoms.get_scaled_positions(wrap=False)}


@pytest.mark.parametrize("cif", _corpus_cifs())
def test_corpus_preopt_reads_like_the_gui(cif):
    if cif is None:
        pytest.skip("test/ data corpus not present (gitignored, ~12GB); set MACE_TEST_DATA")
    if cif.name in KNOWN_DIFFERENT:
        pytest.xfail(KNOWN_DIFFERENT[cif.name])
    _assert_same(_preopt_structure(cif), _gui_structure(cif))


@pytest.mark.parametrize("case", sorted(SYNTH))
def test_synthetic_preopt_reads_like_the_cli(case, tmp_path):
    cif = _write(tmp_path, f"{case}.cif", **SYNTH[case])
    _assert_same(_preopt_structure(cif), _cli_structure(cif, "P1", tmp_path))


def test_preopt_diamond_origin2_keeps_its_group(tmp_path):
    """preopt writes its CIF from the group and kinds read_input returns: the
    8-atom diamond in Fd-3m, which reads back as the same 8 atoms."""
    from mace.preopt import cifout
    from mace.preopt.cli import read_input
    cif = _write(tmp_path, "dia.cif", **SYNTH["diamond_origin2"])
    atoms, sg, kinds, _ = read_input(cif)
    assert len(atoms) == 8 and sg.no == 227 and len(kinds) == 8
    out = tmp_path / "dia_preopt.cif"
    mode = cifout.write(out, atoms, name="dia", header=[], spacegroup=sg, kinds=kinds)
    assert mode["mode"] == "input_group"
    _assert_same(_gui_structure(out), _gui_structure(cif))
