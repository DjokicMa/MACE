"""Slabs, polymers and molecules in `mace preopt` and `mace calc` (1.1.6).

A group-1 SLAB / POLYMER / MOLECULE deck, an extended XYZ with its pbc and a
plain XYZ (a molecule) are read with their periodicity: the dimensionality
keyword is not a geometry edit, nothing wraps along a non-periodic axis, a
cell relaxation moves only the periodic directions, and a molecule keeps its
box. The results are written as an extended XYZ (exact, pbc kept) and a
boxed P1 CIF. Codes that want a 3D cell (xtb, DFTB+, ML potentials) get a
copy boxed with lowdim.VACUUM Å, the stress rescaled to the real cell."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

import preopt_fixtures as F  # noqa: E402
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401

REPO = F.REPO
CLI = REPO / "mace_cli"
SLAB = REPO / "tests" / "data" / "label_band" / "hbn_p1_slab.d12"
POLYMER = """polyyne_p1
POLYMER
1
2.60000000
2
6 0.000000000000E+00 0.000000 0.000000
6 4.800000000000E-01 0.000000 0.000000
BASISSET
POB-TZVP-REV2
DFT
PBE
ENDDFT
END
"""
MOLECULE = """water_p1
MOLECULE
1
3
8 10.000000 10.000000 10.120000
1 10.000000 10.780000 9.520000
1 10.000000 9.220000 9.520000
BASISSET
POB-TZVP-REV2
DFT
PBE
ENDDFT
END
"""
CORPUS = Path("/mnt/iscsi/UsefulScripts/Codebase/reorganization/test")


def mace(*args, cwd=None, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_XTB", "MACE_MLIP_PYTHON", "MACE_DFTB")}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(CLI), "--no-banner", *args], cwd=cwd, env=e, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


@pytest.fixture
def decks(tmp_path):
    (tmp_path / "slab.d12").write_text(SLAB.read_text())
    (tmp_path / "polymer.d12").write_text(POLYMER)
    (tmp_path / "water.d12").write_text(MOLECULE)
    return tmp_path


def test_dimensionality_is_not_an_edit():
    from mace.preopt import deckscan
    for text, kind in ((SLAB.read_text(), "SLAB"), (POLYMER, "POLYMER"), (MOLECULE, "MOLECULE")):
        s = deckscan.scan(text)
        assert s["kind"] == kind and s["geometry_edits"] == []
    s = deckscan.scan(POLYMER.replace("POLYMER", "HELIX", 1))
    assert s["geometry_edits"] == ["HELIX"]


def test_read_input_keeps_pbc(decks):
    from mace.preopt.cli import read_input
    pbcs = {}
    for name in ("slab.d12", "polymer.d12", "water.d12"):
        atoms, sg, kinds, scan = read_input(decks / name)
        pbcs[name] = [bool(x) for x in atoms.pbc]
        assert sg is None and not scan["geometry_edits"]
    assert pbcs == {"slab.d12": [True, True, False], "polymer.d12": [True, False, False],
                    "water.d12": [False, False, False]}
    from ase.build import fcc111, molecule
    from ase.io import write
    s = fcc111("Cu", (2, 2, 2), vacuum=5.0)
    s.pbc = [True, True, False]
    write(str(decks / "cu.extxyz"), s, format="extxyz")
    (decks / "w.xyz").write_text("3\nwater\nO 0 0 0.12\nH 0 0.76 -0.48\nH 0 -0.76 -0.48\n")
    write(str(decks / "POSCAR"), F.demo("Si.cif"), format="vasp")
    assert [bool(x) for x in read_input(decks / "cu.extxyz")[0].pbc] == [True, True, False]
    assert not any(read_input(decks / "w.xyz")[0].pbc)
    assert all(read_input(decks / "POSCAR")[0].pbc)


def test_preopt_cli_slab_polymer_molecule(decks):
    from ase.io import read
    for name, pbc in (("slab", [True, True, False]), ("polymer", [True, False, False]), ("water", [False] * 3)):
        rc, out, err = mace("preopt", f"{name}.d12", "--relax-cell", cwd=decks)
        assert rc == 0, err
        assert (decks / f"{name}_preopt.cif").is_file() and "box with 15" in out
        a0 = read(str(decks / f"{name}_preopt.extxyz"))
        assert [bool(x) for x in a0.pbc] == pbc
        prov = json.loads((decks / f"{name}_preopt.json").read_text())
        assert prov["pbc"] == pbc and prov["output_extxyz"] == f"{name}_preopt.extxyz"
        assert any("non-periodic axes" in n for n in prov["notes"])
        from mace.preopt.cli import read_input
        before = read_input(decks / f"{name}.d12")[0]
        # the non-periodic cell vectors (the drawing box) never change; a molecule keeps its box
        for i in range(3):
            if not pbc[i]:
                assert np.allclose(a0.cell.array[i], before.cell.array[i], atol=1e-8)
        if name == "water":
            assert np.allclose(a0.cell.array, before.cell.array)


def test_no_wrapping_along_open_axes(tmp_path):
    """An atom far outside the drawing box along a non-periodic axis stays there."""
    from mace.preopt import core
    from ase import Atoms
    a = Atoms("BN", positions=[[0, 0.72, 4.0], [1.25, 1.44, 4.05]],
              cell=[[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 0.5]], pbc=[True, True, False])
    r = core.preoptimize(a, "springs")
    assert r.atoms.positions[:, 2].min() > 3.5
    assert core.spacegroup(a) is None


def test_boxed_wrapper_scales_stress():
    from ase import Atoms
    from mace.preopt import lowdim
    inner = F.load_fake_calc()
    a = Atoms("C2", positions=[[0, 0, 0], [1.3, 0.1, 0]], cell=[[2.6, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
              pbc=[True, False, False])
    a.calc = lowdim.Boxed(inner)
    e = a.get_potential_energy()
    s = a.get_stress(voigt=False)
    b = lowdim.boxed(a, lowdim.VACUUM)
    b.calc = F.load_fake_calc()
    assert e == pytest.approx(b.get_potential_energy())
    vir_box = b.get_stress(voigt=False) * b.get_volume()
    assert np.allclose(s * a.get_volume(), vir_box)
    assert b.cell.lengths()[1] > lowdim.VACUUM and all(b.pbc)


def test_mlip_tier_on_slab_is_boxed(tmp_path, monkeypatch, decks):
    """The fake ML worker answers for a slab: the tier ran (boxed) and the cell kept its open axis."""
    py = F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("MACE_MLIP_PYTHON", str(py))
    from mace.preopt import core
    from mace.preopt.cli import read_input
    a = read_input(decks / "slab.d12")[0]
    a = a.repeat((2, 2, 1))
    a.rattle(0.02, seed=1)
    r = core.preoptimize(a, "mlip:chgnet-0.3.0", relax_cell=True)
    assert r.method_ran in ("mlip:chgnet-0.3.0", "springs")
    tried = [x for x in r.attempts if x["method"] == "mlip:chgnet-0.3.0"][0]
    assert tried["status"] in ("ok", "rejected"), tried
    assert np.allclose(r.atoms.cell.array[2], a.cell.array[2])


def test_calc_on_low_dim_decks(decks):
    from ase.io import read
    rc, out, err = mace("calc", "opt", "polymer.d12", "-b", "uff", "--relax-cell", cwd=decks)
    assert rc == 0, err
    assert "only the periodic directions" in out
    p = read(str(decks / "polymer_uff-uff4mof_opt.extxyz"))
    assert [bool(x) for x in p.pbc] == [True, False, False] and (decks / "polymer_uff-uff4mof_opt.cif").is_file()
    from mace.preopt.cli import read_input
    before = read_input(decks / "polymer.d12")[0]
    assert np.allclose(p.cell.array[1:], before.cell.array[1:], atol=1e-8)
    rc, out, err = mace("calc", "opt", "water.d12", "-b", "uff", "--relax-cell", cwd=decks)
    assert rc == 0, err
    assert "keeps its box" in out
    rc, out, err = mace("calc", "charges", "slab.d12", cwd=decks)
    assert rc == 0, err
    assert (decks / "slab_eqeq-eqeq_charges.extxyz").is_file()
    rep = json.loads((decks / "slab_eqeq-eqeq_charges.json").read_text())
    assert rep["report"]["method"] == "direct" and abs(sum(rep["charges"])) < 1e-8


@pytest.mark.skipif(not CORPUS.is_dir(), reason="the real CRYSTAL corpus (test/) is not here")
def test_corpus_slab_and_molecule_decks(tmp_path):
    slab = CORPUS / "SP" / "4LG_FSI_2x2_AA_opt_sp.d12"
    mol = next(iter(sorted((CORPUS / "BAND").glob("Suflolane_MOLECULE*optimized.d12"))), None)
    for src in (slab, mol):
        if src is None or not src.is_file():
            continue
        dst = tmp_path / src.name
        dst.write_text(src.read_text())
        rc, out, err = mace("preopt", dst.name, "--relax-cell", cwd=tmp_path)
        assert rc == 0, (src.name, err)
        from ase.io import read
        a = read(str(tmp_path / f"{dst.stem}_preopt.extxyz"))
        assert sum(bool(x) for x in a.pbc) == (2 if "SLAB" in src.read_text().splitlines()[1] else 0)


# --- a deck in a low-dimensional group other than 1, and CRYSTAL outputs ----------------
# graphene_opt.d12 (layer group 80) was refused with the GUI's words ("MACE GUI reads 3D ...")
# and told to open the .out, which mace preopt did not read. The words now fit every reader,
# and mace preopt reads a CRYSTAL .out (its last geometry) through the GUI's reader.

GROUPS = REPO / "tests" / "data" / "low_dim_groups"
SUITE = REPO / "examples" / "suite"


def _unxz(src, dst):
    import lzma
    dst.write_bytes(lzma.decompress(src.read_bytes()))
    return dst


def test_slab_deck_in_layer_group_80_is_refused_in_words_every_reader_can_follow(tmp_path):
    import shutil
    run = SUITE / "3-slab-2d" / "graphene" / "OPT"
    shutil.copy(run / "graphene_opt.d12", tmp_path)
    for cmd in (("preopt", "graphene_opt.d12"), ("calc", "sp", "graphene_opt.d12", "-b", "uff")):
        rc, out, err = mace(*cmd, cwd=tmp_path)
        assert rc != 0 and "graphene_opt.d12 is a SLAB deck in layer group 80" in err, (cmd, err)
        assert "decks only in group 1 (every atom listed)" in err and "Use the .out of its run instead" in err
        assert "MACE GUI" not in err
    shutil.copy(run / "graphene_opt.out.xz", tmp_path)
    rc, out, err = mace("preopt", "graphene_opt.d12", cwd=tmp_path)
    assert rc == 1 and "unpack it (xz -dk graphene_opt.out.xz) and use graphene_opt.out instead" in err
    _unxz(tmp_path / "graphene_opt.out.xz", tmp_path / "graphene_opt.out")
    rc, out, err = mace("preopt", "graphene_opt.d12", cwd=tmp_path)
    assert rc == 1 and "graphene_opt.out, the output of its run, has the structure: use that file instead" in err
    assert not list(tmp_path.glob("*_preopt.*"))


def test_preopt_reads_the_out_of_a_slab_run_and_keeps_its_vacuum(tmp_path):
    from ase.io import read
    from mace.calculators import structio
    out = _unxz(SUITE / "3-slab-2d" / "graphene" / "OPT" / "graphene_opt.out.xz", tmp_path / "graphene_opt.out")
    before, _, _ = structio.read(out)
    assert list(before.pbc) == [True, True, False]
    rc, so, err = mace("preopt", "graphene_opt.out", "--relax-cell", cwd=tmp_path)
    assert rc == 0, err
    a = read(str(tmp_path / "graphene_opt_preopt.extxyz"))
    assert list(a.pbc) == [True, True, False] and len(a) == len(before)
    assert np.allclose(a.cell[2], before.cell[2])                     # no cell relaxation along z
    assert np.allclose(a.positions[:, 2], before.positions[:, 2], atol=1e-6)
    prov = json.loads((tmp_path / "graphene_opt_preopt.json").read_text())
    assert prov["input"] == "graphene_opt.out" and prov["output_extxyz"] == "graphene_opt_preopt.extxyz"
    assert "box with" in so and (tmp_path / "graphene_opt_preopt.cif").is_file()


@pytest.mark.parametrize("name,dim", [("polyyne_rg51", 1), ("graphene_lg37", 2)])
def test_preopt_reads_the_out_where_the_deck_is_refused(tmp_path, name, dim):
    import shutil
    shutil.copy(GROUPS / f"{name}.d12", tmp_path)
    rc, out, err = mace("preopt", f"{name}.d12", cwd=tmp_path)
    assert rc == 1 and "only in group 1" in err and "MACE GUI" not in err
    shutil.copy(GROUPS / f"{name}.out", tmp_path)
    rc, out, err = mace("preopt", f"{name}.out", cwd=tmp_path)
    assert rc == 0, err
    from ase.io import read
    assert sum(bool(x) for x in read(str(tmp_path / f"{name}_preopt.extxyz")).pbc) == dim


def test_group_1_polymer_and_molecule_decks_and_their_outs(tmp_path):
    import shutil
    from ase.io import read
    cases = ((SUITE / "4-polymer-1d" / "polyacetylene" / "OPT", "polyacetylene_opt", 1),
             (SUITE / "5-molecule-0d" / "EC" / "OPT", "EC_opt", 0))
    for run, stem, dim in cases:
        shutil.copy(run / f"{stem}.d12", tmp_path)
        rc, out, err = mace("preopt", f"{stem}.d12", "--relax-cell", cwd=tmp_path)
        assert rc == 0, (stem, err)
        assert sum(bool(x) for x in read(str(tmp_path / f"{stem}_preopt.extxyz")).pbc) == dim
        sub = tmp_path / "out"
        sub.mkdir(exist_ok=True)
        _unxz(run / f"{stem}.out.xz", sub / f"{stem}.out")
        rc, out, err = mace("preopt", f"{stem}.out", "--relax-cell", cwd=sub)
        assert rc == 0, (stem, err)
        assert sum(bool(x) for x in read(str(sub / f"{stem}_preopt.extxyz")).pbc) == dim


@pytest.mark.skipif(not CORPUS.is_dir(), reason="the real CRYSTAL corpus (test/) is not here")
def test_corpus_slab_and_molecule_outs(tmp_path):
    from ase.io import read
    slab = CORPUS / "SP" / "4LG_FSI_2x2_AA_opt_sp.out"
    mol = next(iter(sorted((CORPUS / "BAND").glob("Suflolane_MOLECULE*optimized.out"))), None)
    for src, dim in ((slab, 2), (mol, 0)):
        if src is None or not src.is_file():
            continue
        dst = tmp_path / src.name
        dst.write_bytes(src.read_bytes())
        rc, out, err = mace("preopt", dst.name, "--steps", "5", cwd=tmp_path)
        assert rc == 0, (src.name, err)
        assert sum(bool(x) for x in read(str(tmp_path / f"{dst.stem}_preopt.extxyz")).pbc) == dim


def test_preopt_names_the_formats_it_reads(tmp_path):
    (tmp_path / "x.txt").write_text("nothing\n")
    rc, out, err = mace("preopt", "x.txt", cwd=tmp_path)
    assert rc == 1 and ".cif, .d12, .xyz/.extxyz, POSCAR and CRYSTAL .out files" in err
