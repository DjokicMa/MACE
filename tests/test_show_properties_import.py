"""Importing mace.utils.show_properties must not open (and so create) a
materials.db in the current directory; running it still prints a material's
properties."""
import os
import subprocess
import sys

from conftest import REPO_ROOT


def _env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "MACE_HOME")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_import_creates_no_database(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", "import mace.utils.show_properties"],
        cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "materials.db").exists()


def test_running_it_lists_the_material_properties(tmp_path):
    seed = (
        "from mace.database.materials import MaterialDatabase\n"
        "db = MaterialDatabase('materials.db')\n"
        "with db._get_connection() as c:\n"
        "    c.execute(\"INSERT INTO properties (material_id, property_category,"
        " property_name, property_value, property_unit, extracted_at)"
        " VALUES ('dia', 'electronic', 'band_gap', 5.4, 'eV', '2026-01-01')\")\n")
    proc = subprocess.run([sys.executable, "-c", seed], cwd=tmp_path,
                          env=_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace" / "utils" / "show_properties.py"), "dia"],
        cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "ELECTRONIC" in proc.stdout
    assert "band_gap: 5.4 eV" in proc.stdout
