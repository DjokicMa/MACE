"""A database build records each material's formula and dimensionality from
its output (bug hunt 1.1.5, #2).

The populate scan created every material as dimensionality CRYSTAL, and the
formula came from the .d12's atom lines, i.e. the asymmetric unit: TiPbO3
(sg 99) was stored as TiPbO2, TiO (sg 189, 3 Ti + 3 O in the cell) as Ti2O, a
molecule as a CRYSTAL, and an output without a .d12 beside it as "Unknown".
Now the formula is the reduced formula of the cell the output prints (the
cell formula goes into metadata_json), the dimensionality is the one CRYSTAL
prints, and a deck is only the fallback (expanded by its space group).

Real outputs from the test/ corpus, through the real CLI (`mace_cli analyze
--extract-properties DIR --register`)."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

# the formulas and dimensionalities come from the GUI's structure reader, which needs ase
pytest.importorskip("ase", reason="the structure reader needs ase")

EC = "OPT/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
TIPBO3 = "FREQ/TiPbO3_mp-20459_sg99_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_freq_B3LYP-D3-D3_optimized"
TIO = "OPT/TiO_mp-1071163_sg189_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
DIA2 = "FREQ/2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized"
SLAB = "SP/4LG_FSI_2x2_AA_graphene_opt_sp"


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _folder(tmp_path, stems_with_deck, stems_without_deck=()):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp_path / "runs"
    for stem in stems_with_deck:
        sub = d / Path(stem).parent
        sub.mkdir(parents=True, exist_ok=True)
        for ext in (".out", ".d12"):
            shutil.copyfile(TEST_DATA / (stem + ext), sub / (Path(stem).name + ext))
    for stem in stems_without_deck:
        sub = d / Path(stem).parent
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEST_DATA / (stem + ".out"), sub / (Path(stem).name + ".out"))
    return d


def _build(d, db):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze",
                        "--extract-properties", str(d), "--db-path", str(db), "--register"],
                       capture_output=True, text=True, env=_env(), cwd=str(d.parent), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _materials(db):
    c = sqlite3.connect(str(db))
    try:
        return {r[0]: {"formula": r[1], "dimensionality": r[2], "space_group": r[3],
                       "metadata": json.loads(r[4]) if r[4] else {}}
                for r in c.execute("SELECT material_id, formula, dimensionality, space_group, metadata_json "
                                   "FROM materials")}
    finally:
        c.close()


def test_formula_and_dimensionality_come_from_the_output(tmp_path):
    d = _folder(tmp_path, [EC, TIPBO3, TIO, SLAB], [DIA2])
    db = tmp_path / "materials.db"
    _build(d, db)
    m = _materials(db)
    assert m["EC"]["dimensionality"] == "MOLECULE"
    assert m["EC"]["formula"] == "C3H4O3"
    assert m["TiPbO3_mp-20459_sg99_sym"]["formula"] == "TiPbO3"
    assert m["TiPbO3_mp-20459_sg99_sym"]["dimensionality"] == "CRYSTAL"
    tio = m["TiO_mp-1071163_sg189_sym"]
    assert tio["formula"] == "TiO"
    assert tio["metadata"].get("cell_formula") == "Ti3O3"
    assert tio["space_group"] == 189
    # no deck beside the output: the output still names the material's formula
    assert m["2_dia2"]["formula"] == "C"
    assert m["2_dia2"]["metadata"].get("cell_formula") == "C4"
    assert m["4LG_FSI_2x2_AA_graphene"]["dimensionality"] == "SLAB"


def test_an_existing_database_is_corrected_but_a_hand_set_formula_stays(tmp_path):
    """A database an earlier build wrote (deck formulas, every material
    CRYSTAL) is corrected by the next build; a formula the user typed is not a
    deck's asymmetric unit and stays."""
    d = _folder(tmp_path, [EC, TIPBO3, TIO])
    db = tmp_path / "materials.db"
    _build(d, db)
    c = sqlite3.connect(str(db))
    with c:
        # what 1.1.5 before this fix stored
        c.execute("UPDATE materials SET formula = 'TiPbO2', metadata_json = NULL WHERE material_id = 'TiPbO3_mp-20459_sg99_sym'")
        c.execute("UPDATE materials SET dimensionality = 'CRYSTAL' WHERE material_id = 'EC'")
        # a formula typed by hand
        c.execute("UPDATE materials SET formula = 'Ti3O3 (hand)' WHERE material_id = 'TiO_mp-1071163_sg189_sym'")
    c.close()
    _build(d, db)
    m = _materials(db)
    assert m["TiPbO3_mp-20459_sg99_sym"]["formula"] == "TiPbO3"
    assert m["TiPbO3_mp-20459_sg99_sym"]["metadata"].get("cell_formula") == "TiPbO3"
    assert m["EC"]["dimensionality"] == "MOLECULE"
    assert m["TiO_mp-1071163_sg189_sym"]["formula"] == "Ti3O3 (hand)"
    # and a third build changes nothing more (idempotent)
    before = _materials(db)
    _build(d, db)
    after = _materials(db)
    assert {k: (v["formula"], v["dimensionality"]) for k, v in before.items()} == \
           {k: (v["formula"], v["dimensionality"]) for k, v in after.items()}


def test_a_deck_is_expanded_by_its_space_group_when_the_output_has_no_geometry(tmp_path):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.utils.formula_extractor import material_info
    empty = tmp_path / "x.out"
    empty.write_text(" CRYSTAL CALCULATION\n")
    info = material_info(output_file=empty, d12_file=TEST_DATA / (TIO + ".d12"))
    assert info["formula"] == "TiO" and info["cell_formula"] == "Ti3O3"
    info = material_info(output_file=None, d12_file=TEST_DATA / "OPT/1_dia_opt_rev1.d12")
    assert info["formula"] == "C"
    info = material_info(output_file=TEST_DATA / (TIPBO3 + ".out"), d12_file=None)
    assert (info["formula"], info["dimensionality"], info["source"]) == ("TiPbO3", "CRYSTAL", "output")


@pytest.mark.parametrize("stem, formula, cell", [
    ("OPT/TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized",
     "TiPbO3", "TiPbO3"),                                      # the deck's atom lines: TiPbO
    ("OPT/Ag1Cl2_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized",
     "AgCl2", "AgCl2"),                                        # AgCl
    ("OPT/Ti9Se2_mp-619905_sg55_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized",
     "Ti9Se2", "Ti18Se4"),                                     # Ti5Se
])
def test_corpus_formulas_the_cli_hunt_found(stem, formula, cell):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.utils.formula_extractor import material_info
    info = material_info(output_file=TEST_DATA / (stem + ".out"), d12_file=TEST_DATA / (stem + ".d12"))
    assert (info["formula"], info["cell_formula"], info["source"]) == (formula, cell, "output")


MOLECULE_DECK = "OPT/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized.d12"
SLAB_DECK = "SP/4LG_FSI_2x2_AA_opt_sp.d12"
ETHYLENE = """ethylene
MOLECULE
1
6
6 0.0 0.0 0.667
6 0.0 0.0 -0.667
1 0.0 0.923 1.238
1 0.0 -0.923 1.238
1 0.0 0.923 -1.238
1 0.0 -0.923 -1.238
END
"""


def test_a_low_dimensional_deck_keeps_its_dimensionality(tmp_path):
    """A group-1 MOLECULE/SLAB deck (read when the output has no geometry) was recorded as CRYSTAL, and a
    molecule's formula reduced (C2H4 as CH2); 3D decks are recorded as before."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.gui.structures import d12_atoms
    from mace.utils import formula_extractor as FE
    empty = tmp_path / "x.out"
    empty.write_text(" CRYSTAL CALCULATION\n")
    info = FE.material_info(output_file=empty, d12_file=TEST_DATA / MOLECULE_DECK)
    assert (info["formula"], info["dimensionality"], info["source"]) == ("C3H4O3", "MOLECULE", "d12")
    info = FE.material_info(output_file=None, d12_file=TEST_DATA / SLAB_DECK)
    assert (info["dimensionality"], info["source"]) == ("SLAB", "d12"), info
    (tmp_path / "eth.d12").write_text(ETHYLENE)
    info = FE.material_info(output_file=None, d12_file=tmp_path / "eth.d12")
    assert (info["formula"], info["cell_formula"], info["dimensionality"]) == ("C2H4", "C2H4", "MOLECULE")
    # 3D decks: the reduced formula and CRYSTAL, as before
    decks = [p for p in sorted((TEST_DATA / "OPT").glob("*_CRYSTAL_*.d12"))][:25]
    assert decks
    for p in decks:
        try:
            atoms, _ = d12_atoms(p)
        except Exception:
            continue
        f = FE._formulas(atoms.numbers, molecule=False)
        assert FE._info_from_d12(p) == {"formula": f[0], "cell_formula": f[1], "dimensionality": "CRYSTAL",
                                        "source": "d12"}, p.name
