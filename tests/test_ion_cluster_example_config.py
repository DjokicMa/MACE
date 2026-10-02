"""The shipped ion_cluster_molecule.json config works on a cluster CIF that declares symmetry.

Ion charges need every atom of the cluster: CHEMOD labels are positions in
the full atom list and CHEMOD needs a deck without symmetry (manual p.80).
The example config left symmetry_handling at its default (CIF), so a cluster
CIF that declares a space group was reduced to its asymmetric unit and
cif2d12 refused the ion charges. It now sets "symmetry_handling": "P1" and
"write_only_unique": false, as the tests' own MOLECULE options do.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

CONFIG = REPO_ROOT / "Crystal_d12" / "example_configs" / "ion_cluster_molecule.json"

# Li+ . H2O in a 30 A box (Li 1.2 A out of the water plane, so the only symmetry
# is the mirror x, -y, z at y = 1/2), written in space group Pm: the two H are
# one orbit, so the CIF lists three atoms
PM_CIF = """data_li_water
_cell_length_a 30.000000
_cell_length_b 30.000000
_cell_length_c 30.000000
_cell_angle_alpha 90.000000
_cell_angle_beta 90.000000
_cell_angle_gamma 90.000000
_symmetry_space_group_name_H-M 'P 1 m 1'
_symmetry_Int_Tables_number 6
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
  'x, -y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0.500000 0.500000 0.540000
O1 O 0.565000 0.500000 0.500000
H1 H 0.584667 0.525333 0.500000
"""


def test_the_config_writes_every_atom():
    cfg = json.loads(CONFIG.read_text())["configuration"]
    assert cfg["symmetry_handling"] == "P1"
    assert cfg["write_only_unique"] is False


def test_the_config_charges_a_cluster_cif_with_declared_symmetry(tmp_path):
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / "li_water.cif").write_text(PM_CIF)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    run = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "convert", "--no-banner", "--batch",
                          "--options_file", str(CONFIG), "--cif_dir", str(cifs), "--output_dir",
                          str(tmp_path / "out")], cwd=tmp_path, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, timeout=300, env=env)
    decks = list((tmp_path / "out").glob("*.d12")) if (tmp_path / "out").exists() else []
    assert run.returncode == 0 and decks, run.stdout + run.stderr
    lines = decks[0].read_text().splitlines()
    assert lines[1:4] == ["MOLECULE", "1", "4"]
    assert [l.split()[-1] for l in lines[4:8]].count("H") == 2
    i = lines.index("SOLDEF2MSVP")
    assert lines[i + 1:i + 3] == ["CHEMOD", "1"]
    assert lines[-3:] == ["DOPING", "-1", "END"]
