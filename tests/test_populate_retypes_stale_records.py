"""A record 1.1.1 typed from the file name gets the type the output shows.

1.1.1's populate scan took the calc type from file-name substrings, so an SP
named "..._SLAB_OPT_FSI.out" (test/SP/4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT_FSI.out)
was recorded as OPT. Since 1.1.2 the scan uses the completion checker's
verdict (SP), but populate_database matched the existing record by output
file and updated only status/output_file, so the stale OPT stayed.
"""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from conftest import find_data

REPO = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data" / "charged_runs"


def _seed_111_record(db_path, out, calc_type, status="completed"):
    from mace.database.materials import MaterialDatabase, create_material_id_from_file
    db = MaterialDatabase(str(db_path))
    mid = create_material_id_from_file(out.name)
    db.create_material(material_id=mid, formula="Unknown", space_group=1,
                       dimensionality="CRYSTAL", source_type="d12", source_file="unknown")
    cid = db.create_calculation(material_id=mid, calc_type=calc_type, input_file=None,
                                work_dir=str(out.parent))
    db.update_calculation_status(cid, status, output_file=str(out))
    return cid


def _populate(scan_dir, db_path):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "mace.database.populate_completed_jobs", str(scan_dir),
                        "--db", str(db_path)], cwd=REPO, env=env, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def _calc_types(db_path):
    with sqlite3.connect(db_path) as c:
        return {r[0]: r[1] for r in c.execute("SELECT calc_id, calc_type FROM calculations")}


def test_sp_named_opt_is_retyped(tmp_path):
    d = tmp_path / "sorted"
    d.mkdir()
    out = d / "li_FSI_SLAB_OPT_FSI.out"
    shutil.copy(DATA / "li_fsi_ec_neutral.out", out)
    db = tmp_path / "m.db"
    cid = _seed_111_record(db, out, "OPT")
    _populate(d, db)
    types = _calc_types(db)
    assert types == {cid: "SP"}, types


def test_the_audits_corpus_file(tmp_path):
    src = find_data("SP/4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT_FSI.out")
    d = tmp_path / "completesp"
    d.mkdir()
    out = d / src.name
    shutil.copy(src, out)
    db = tmp_path / "m.db"
    cid = _seed_111_record(db, out, "OPT")
    _populate(d, db)
    assert _calc_types(db) == {cid: "SP"}


def test_workflow_numbered_types_are_left_alone(tmp_path):
    """OPT2/SP2/... are written by the workflow engine, which knows what it
    ran; only a base type the scan itself could have written is corrected."""
    d = tmp_path / "w"
    d.mkdir()
    out = d / "li_FSI_SLAB_OPT_FSI.out"
    shutil.copy(DATA / "li_fsi_ec_neutral.out", out)
    db = tmp_path / "m.db"
    cid = _seed_111_record(db, out, "OPT2")
    _populate(d, db)
    assert _calc_types(db) == {cid: "OPT2"}
