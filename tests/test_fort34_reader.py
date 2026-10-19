"""fort.34 (CRYSTAL's EXTERNAL geometry, .f34 / .gui / .ext) read with its symmetry.

Reported in 1.1.6-preview: an EXTERNAL deck whose job script stages an
optimized geometry (graphite, AB slip, 1x1x2 conventional cell) could not be
drawn: "High symmetry geometry is not allowed." That was ase's fort.34 reader
(ase/io/crystal.py), which refuses any file with more than the identity
operator, and CRYSTAL writes the full point group with the asymmetric unit.
mace.utils.fort34 expands the asymmetric unit with the operators instead.

Fixtures (tests/data/fort34/): the asymmetric unit and Cartesian operators of
the final geometry of four corpus OPT outputs (diamond Fd-3m, Ag2Br2 P6_3mc,
Ag2Br3 R-3c, and the P1 4-layer graphene slab), plus graphite AB 1x1x2
(P6_3/mmc) and a graphene slab (p6/mmm, c = 500 Å).
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, TEST_DATA
from test_gui_examples import _json, _start, needs_ase

F34 = REPO_ROOT / "tests" / "data" / "fort34"
OPT = TEST_DATA / "OPT"

# fixture -> (its OPT output, atoms in the cell, space group)
FROM_OUT = {
    "diamond_Fd-3m.f34": ("1_dia_opt_rev1.out", 2, "Fd-3m"),
    "Ag2Br2_P63mc.f34": ("Ag2Br2_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out",
                         4, "P6_3mc"),
    "Ag2Br3_R-3c.f34": ("Ag2Br3_mp-862982_sg167_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out",
                        10, "R-3c"),
    "4LG_ABAB_slab_P1.f34": ("4LG_2x2_ABAB_opt_HSESOL3C_optimized.out", 96, None),
}


def _column_convention(text: str) -> str:
    """The same file with every rotation transposed (x' = R x + t)."""
    lines = text.splitlines()
    n = int(lines[4])
    for k in range(n):
        i = 5 + 4 * k
        rows = np.array([[float(x) for x in lines[i + j].split()] for j in range(3)]).T
        for j in range(3):
            lines[i + j] = "".join(f"{x:20.12E}" for x in rows[j])
    return "\n".join(lines) + "\n"


# --- the reader ------------------------------------------------------------------

def test_ase_refuses_what_mace_reads():
    """The reported words came from ase, not MACE."""
    ase_crystal = pytest.importorskip("ase.io.crystal")
    with pytest.raises(ValueError, match="High symmetry geometry is not allowed"):
        ase_crystal.read_crystal(str(F34 / "graphite_AB_conv112.f34"))


def test_graphite_ab_conv112_expands_to_eight_atoms():
    from mace.utils import fort34
    f = fort34.read(F34 / "graphite_AB_conv112.f34")
    assert (f["nsym"], f["nirr"], len(f["numbers"])) == (48, 2, 8)
    assert f["dim"] == 3 and f["pbc"] == [True] * 3
    z = sorted(round(float(x), 4) for x in np.mod(f["cart"][:, 2] / 13.422, 1.0))
    assert z == [0.125] * 2 + [0.375] * 2 + [0.625] * 2 + [0.875] * 2   # four layers, two atoms each


def test_graphene_slab_keeps_the_500_angstrom_convention():
    from mace.utils import fort34
    f = fort34.read(F34 / "graphene_slab_p6mmm.f34")
    assert (f["dim"], f["nsym"], f["nirr"], len(f["numbers"])) == (2, 24, 1, 2)
    assert f["pbc"] == [True, True, False]
    assert np.allclose(f["cart"][:, 2], 0.0)       # never folded along the 500 Å vector


@needs_ase
@pytest.mark.parametrize("name", sorted(FROM_OUT))
def test_full_cell_matches_the_outputs_final_geometry(name):
    """Atom count, cell and every position against the .out CRYSTAL wrote."""
    from mace.gui import structures
    out_name, n, sg = FROM_OUT[name]
    out = OPT / out_name
    if not out.is_file():
        pytest.skip(f"corpus file {out_name} is absent")
    a = structures.read_structure(F34 / name)
    b = structures.read_out(out, -1)
    assert len(a["numbers"]) == len(b["numbers"]) == n
    assert a["dimensionality"] == b["dimensionality"]
    per = sum(a["pbc"])
    for k in "abc"[:per]:
        assert a["lattice"][k] == pytest.approx(b["lattice"][k], abs=1e-6)
    cell_b = np.array(b["cell"])
    fa, fb = np.array(a["frac"]), np.array(b["frac"])
    ca, cb = np.array(a["cart"]), np.array(b["cart"])
    for i in range(n):
        best = np.inf
        for j in range(n):
            if a["numbers"][i] != b["numbers"][j]:
                continue
            df = fa[i] - fb[j]
            df[:per] -= np.round(df[:per])
            d = np.r_[df[:per] @ cell_b[:per, :per] if per < 3 else df @ cell_b, (ca[i] - cb[j])[per:]]
            best = min(best, float(np.linalg.norm(d)))
        assert best < 1e-5, (name, i, best)
    if sg:
        assert a["symmetry"]["international"] == sg


@pytest.mark.parametrize("name", ["graphite_AB_conv112.f34", "diamond_Fd-3m.f34", "Ag2Br3_R-3c.f34",
                                  "graphene_slab_p6mmm.f34"])
def test_operators_written_either_way_round(tmp_path, name):
    from mace.utils import fort34
    row = fort34.read(F34 / name)
    p = tmp_path / name
    p.write_text(_column_convention((F34 / name).read_text()))
    col = fort34.read(p)
    assert len(col["numbers"]) == len(row["numbers"])


def test_fortran_d_exponents(tmp_path):
    from mace.utils import fort34
    p = tmp_path / "d.f34"
    p.write_text((F34 / "diamond_Fd-3m.f34").read_text().replace("E", "D"))
    assert len(fort34.read(p)["numbers"]) == 2


def _damaged(text, how):
    lines = text.splitlines()
    if how == "truncated":
        return "\n".join(lines[:20]) + "\n"
    if how == "bad_number":
        lines[2] = "  abc  0.0  0.0"
        return "\n".join(lines) + "\n"
    if how == "not_a_rotation":
        lines[9] = "".join(f"{x:20.12E}" for x in (2.0, 0.0, 0.0))
        return "\n".join(lines) + "\n"
    if how == "open_orbit":          # a screw axis' translation lost: the operators no longer form a group
        n = int(lines[4])
        k = next(k for k in range(n) if any(abs(float(x)) > 1e-6 for x in lines[8 + 4 * k].split())
                 and abs(float(lines[7 + 4 * k].split()[2])) > 0.5)
        lines[8 + 4 * k] = "".join(f"{x:20.12E}" for x in (0.0, 0.0, 0.0))
        return "\n".join(lines) + "\n"
    raise AssertionError(how)


@pytest.mark.parametrize("how,words", [
    ("truncated", "symmetry operators (4 records each) announced"),
    ("bad_number", "line 3 (lattice vector)"),
    ("not_a_rotation", "is not a rotation"),
    ("open_orbit", "do not map"),
])
def test_a_damaged_file_says_what_is_wrong(tmp_path, how, words):
    from mace.gui import structures
    p = tmp_path / "graphite.f34"
    p.write_text(_damaged((F34 / "graphite_AB_conv112.f34").read_text(), how))
    with pytest.raises(ValueError) as exc:
        structures.read_structure(p)
    msg = str(exc.value)
    assert msg.startswith("graphite.f34 is not a readable fort.34: ") and words in msg
    assert "High symmetry" not in msg


# --- the deck that reported it, in the reader and through the GUI API -------------------

DECK = "graphiteABslip_tier9_SP8_freeC\nEXTERNAL\nEND\nBASISSET\nPOB-TZVP-REV2\nDFT\nPBE\nEND\nSHRINK\n4 4\nEND\n"
SH = """\
#!/bin/bash --login
export JOB=graphiteABslip_tier9_SP8_freeC; export DIR=$SLURM_SUBMIT_DIR
cp $DIR/$JOB.d12 INPUT; cp $DIR/graphiteABslip_tier9_E_freeC_conv112.opt.f34 fort.34
mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal >& $DIR/$JOB.out
"""
STEM = "graphiteABslip_tier9_SP8_freeC"
SRC = "graphiteABslip_tier9_E_freeC_conv112.opt.f34"


def _reported(root: Path, f34_text: str):
    (root / f"{STEM}.d12").write_text(DECK)
    (root / f"{STEM}.sh").write_text(SH)
    (root / SRC).write_text(f34_text)


@needs_ase
def test_reported_external_deck_draws_its_staged_f34(tmp_path):
    from mace.gui import structures
    _reported(tmp_path, (F34 / "graphite_AB_conv112.f34").read_text())
    s = structures.read_structure(tmp_path / f"{STEM}.d12")
    assert len(s["numbers"]) == 8 and s["source"]["geometry_from"].endswith(SRC)
    assert s["symmetry"]["international"] == "P6_3/mmc"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("gui_f34") / "decks"
    root.mkdir()
    _reported(root, (F34 / "graphite_AB_conv112.f34").read_text())
    (root / "broken").mkdir()
    _reported(root / "broken", _damaged((F34 / "graphite_AB_conv112.f34").read_text(), "open_orbit"))
    shutil.copy(F34 / "graphene_slab_p6mmm.f34", root / "graphene.f34")
    proc, base, _ = _start(root)
    yield root, base
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


@needs_ase
def test_api_draws_the_reported_deck_and_its_f34(served):
    _, base = served
    status, body = _json(base, f"api/structure?path={STEM}.d12")
    assert status == 200, body
    assert len(body["numbers"]) == 8
    status, body = _json(base, f"api/structure?path={SRC}")
    assert status == 200, body
    assert len(body["numbers"]) == 8 and body["source"]["symmetry_operators"] == 48
    status, body = _json(base, "api/structure?path=graphene.f34")
    assert status == 200, body
    assert len(body["numbers"]) == 2 and body["pbc"] == [True, True, False]


@needs_ase
def test_api_damaged_f34_names_the_file_once_and_says_why(served):
    _, base = served
    status, body = _json(base, f"api/structure?path=broken/{STEM}.d12")
    assert status == 400, body
    err = body["error"]
    assert err.startswith(f"{STEM}.d12 reads its geometry from {SRC} (fort.34), which cannot be read: ")
    assert "do not map" in err and "High symmetry" not in err
    assert err.count(SRC) == 1                  # not "X.f34 ...: X.f34 is not a readable ..."
