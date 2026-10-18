"""The database stores an index of an ORBITALS run (files, k points, levels), never grids."""
import json
import shutil
from pathlib import Path

from mace.utils.property_extractor import CrystalPropertyExtractor

DATA = Path(__file__).parent / "data" / "orbitals"


def _extract(folder, out_name, tmp_path):
    for f in (DATA / folder).iterdir():
        shutil.copy(f, tmp_path / f.name)
    x = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    out = tmp_path / out_name
    return x, x._extract_orbitals_properties(out.read_text(), out)


def test_periodic_index(tmp_path):
    x, p = _extract("si_f", "si_f.out", tmp_path)
    assert p["orbitals_num_files"] == 3 and p["orbitals_num_kpoints"] == 8
    assert p["orbitals_nbf"] == 36 and p["orbitals_dimensionality"] == 3
    assert p["orbitals_usable_kpoints"] == 1                 # the complex K110 pair is not drawable
    assert p["orbitals_kpoints"] == "3:0.5,0,0:R;4:0.25,0.25,0:C"
    assert p["orbitals_homo_band"] == 14 and p["orbitals_lumo_band"] == 15
    assert p["orbitals_self_check"] == "ok"
    blob = p["orbitals"]
    assert len(blob["missing_files"]) == 10                   # listed in the .out, not copied here
    k3 = next(k for k in blob["kpoints"] if k["index"] == 3)
    assert k3["spins"][0]["homo"] == 14 and k3["spins"][0]["levels_from"] == 6 and len(k3["spins"][0]["energies_au"]) == 18
    # no grids and no coefficients: the index stays small
    assert len(json.dumps(blob)) < 20_000
    assert x._get_property_unit("orbitals_homo_energy_au") == "Hartree"
    assert x._get_property_unit("orbitals_gap_ev") == "eV"
    assert x._categorize_property("orbitals_homo_energy_au") == "orbitals"


def test_open_shell_molecule_index(tmp_path):
    _, p = _extract("o2", "o2_c.out", tmp_path)
    assert p["orbitals_num_files"] == 1 and p["orbitals_dimensionality"] == 0
    assert p["orbitals_spin_polarized"] is True
    spins = {s["spin"] for s in p["orbitals"]["kpoints"][0]["spins"]}
    assert spins == {"ALPHA", "BETA"}


def test_not_an_orbitals_run():
    x = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    assert x._extract_orbitals_properties("TOTAL ENERGY -1.0", Path("x.out")) == {}
