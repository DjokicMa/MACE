"""A big materials.db for the database view's tests and benchmarks (plan T7, T14).

The schema is made by MACE's own MaterialDatabase (as the spec says); the rows
are generated (made-up material ids, formulas from a few binary compounds,
no research data) and bulk-inserted with executemany on one transaction:
MaterialDatabase's per-row API would take far too long for 100k materials.
The file goes into a temp dir and the caller deletes it.
"""
import random
import sqlite3
import sys
from pathlib import Path

import guiharness as H

FORMULAS = ["Mg4O4", "F4Li4", "Si2", "B2N2", "Ag4Br6", "Na4Cl4", "C2", "Ti1O2", "Zn2O2", "Ga2N2", "Ca4F8", "Li2O1"]
TYPES = ["OPT", "SP", "BAND", "DOSS", "FREQ"]
STATUS = ["completed", "completed", "completed", "failed", "pending", "running"]
DIMS = ["CRYSTAL"] * 6 + ["SLAB", "POLYMER", "MOLECULE"]
PROPS = [("structural", "lattice_a", "Angstrom"), ("structural", "volume", "Angstrom^3"), ("electronic", "band_gap", "eV"),
         ("energy", "total_energy", "Hartree"), ("electronic", "fermi_energy", "eV")]


def make(path: Path, n_materials: int, calcs_per=3, props_per=10, seed=0) -> Path:
    sys.path.insert(0, str(H.REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    path = Path(path)
    db = MaterialDatabase(str(path), ase_db_path=str(path.parent / "structures_bench.db"))
    del db
    rng = random.Random(seed)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    ts = "2026-10-05T10:00:00"
    with conn:
        mats = [(f"mat{k:06d}", rng.choice(FORMULAS), rng.choice([225, 227, 194, 221, 166, 62, 14, 2, 1, None]),
                 rng.choice(DIMS), ts, f"2026-10-0{1 + k % 5}T{k % 24:02d}:00:00", "cif", f"cifs/mat{k:06d}.cif",
                 rng.choice(["active", "active", "archived"])) for k in range(n_materials)]
        conn.executemany("INSERT INTO materials (material_id, formula, space_group, dimensionality, created_at, updated_at, "
                         "source_type, source_file, status) VALUES (?,?,?,?,?,?,?,?,?)", mats)
        calcs = []
        for k in range(n_materials):
            for j in range(calcs_per):
                calcs.append((f"mat{k:06d}_{j}", f"mat{k:06d}", TYPES[j % len(TYPES)], rng.choice(STATUS), ts,
                              f"mat{k:06d}_{j}.d12", f"mat{k:06d}_{j}.out"))
        conn.executemany("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, input_file, output_file) "
                         "VALUES (?,?,?,?,?,?,?)", calcs)
        props = []
        for k in range(n_materials):
            for j in range(props_per):
                cat, name, unit = PROPS[j % len(PROPS)]
                props.append((f"mat{k:06d}", f"mat{k:06d}_{j % calcs_per}", cat, f"{name}" if j < len(PROPS) else f"{name}_{j}",
                              round(rng.uniform(-10, 10), 6), unit, ts))
        conn.executemany("INSERT INTO properties (material_id, calc_id, property_category, property_name, property_value, "
                         "property_unit, extracted_at) VALUES (?,?,?,?,?,?,?)", props)
    conn.close()
    (path.parent / "structures_bench.db").unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    return path
