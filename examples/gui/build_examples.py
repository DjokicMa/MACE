#!/usr/bin/env python3
"""Rebuild the `mace gui` example folder in this directory.

    python examples/gui/build_examples.py
    python examples/gui/build_examples.py --suite --corpus /path/to/test   # examples/suite (build_suite.py)

Textbook structures are built here from their experimental lattice
constants. The CRYSTAL outputs are copies of files MACE's own tests already
use (tests/data and tests/fixtures), and materials.db is written with
MaterialDatabase from what those files hold; no number in it is invented.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from ase.spacegroup import crystal  # noqa: E402

from mace.gui import structures as S  # noqa: E402


# name: (H-M symbol, IT number, a, b, c, alpha, beta, gamma, asymmetric unit)
# Experimental room-temperature lattice constants (Angstrom); the ITA
# default setting (origin choice 1 for Fd-3m, as ase reads it).
TEXTBOOK = {
    "MgO": ("F m -3 m", 225, 4.212, 4.212, 4.212, 90, 90, 90,
            [("Mg", 0, 0, 0), ("O", 0.5, 0.5, 0.5)]),
    "LiF": ("F m -3 m", 225, 4.027, 4.027, 4.027, 90, 90, 90,
            [("Li", 0, 0, 0), ("F", 0.5, 0.5, 0.5)]),
    "NaCl": ("F m -3 m", 225, 5.640, 5.640, 5.640, 90, 90, 90,
             [("Na", 0, 0, 0), ("Cl", 0.5, 0.5, 0.5)]),
    "Si": ("F d -3 m", 227, 5.431, 5.431, 5.431, 90, 90, 90, [("Si", 0, 0, 0)]),
    "diamond": ("F d -3 m", 227, 3.567, 3.567, 3.567, 90, 90, 90, [("C", 0, 0, 0)]),
    "hBN": ("P 63/m m c", 194, 2.504, 2.504, 6.661, 90, 90, 120,
            [("B", 1 / 3, 2 / 3, 0.25), ("N", 1 / 3, 2 / 3, 0.75)]),
}


def cifs():
    """Symmetrized CIFs: space group + asymmetric unit, as a database would give."""
    for name, (hm, number, a, b, c, al, be, ga, sites) in TEXTBOOK.items():
        # check the asymmetric unit expands cleanly (no overlapping atoms)
        crystal([s[0] for s in sites], [s[1:] for s in sites], spacegroup=number,
                cellpar=[a, b, c, al, be, ga], onduplicates="error")
        lines = [f"data_{name}", f"_chemical_name_common '{name}'",
                 f"_cell_length_a {a:.4f}", f"_cell_length_b {b:.4f}",
                 f"_cell_length_c {c:.4f}", f"_cell_angle_alpha {al:.2f}",
                 f"_cell_angle_beta {be:.2f}", f"_cell_angle_gamma {ga:.2f}",
                 f"_symmetry_space_group_name_H-M '{hm}'",
                 f"_symmetry_Int_Tables_number {number}", "", "loop_",
                 "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
                 "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
        for k, (el, x, y, z) in enumerate(sites, 1):
            lines.append(f"{el}{k} {el} {x:.10f} {y:.10f} {z:.10f} 1.0")
        (HERE / "structures" / f"{name}.cif").write_text("\n".join(lines) + "\n")


COPIES = {
    # MACE's own test files (see tests/fixtures/phonopy_crystal/README.md for Si)
    "outputs/Si_sp.out": "tests/fixtures/phonopy_crystal/Si/parent.out",
    "outputs/Si_sp.d12": "tests/fixtures/phonopy_crystal/Si/parent.d12",
    # a complete CRYSTAL23 SP of textbook graphene in layer group 80 (P6/mmm)
    "outputs/graphene_slab_sp.out": "tests/data/low_dim_groups/graphene_lg80.out",
    "outputs/graphene_slab_sp.d12": "tests/data/low_dim_groups/graphene_lg80.d12",
    "outputs/Ag2Br3_opt_unfinished.out": "tests/data/opt_restart/tqc_agbr.out",
    "outputs/Ag2Br3_opt_unfinished.d12": "tests/data/opt_restart/tqc_agbr.d12",
}


def copies():
    for dst, src in COPIES.items():
        (HERE / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / src, HERE / dst)


def database():
    from mace.database.materials import MaterialDatabase

    target = HERE / "materials.db"
    for p in (target, target.with_name("materials.db-wal"), target.with_name("materials.db-shm")):
        p.unlink(missing_ok=True)
    scratch = Path(tempfile.mkdtemp())
    db = MaterialDatabase(str(target), ase_db_path=str(scratch / "structures.db"))

    def add_structure(mid, path, source_type):
        s = S.read_structure(HERE / path)
        sg = s["symmetry"].get("number")
        db.create_material(mid, s["formula"], sg, s["dimensionality"], source_type, path)
        # only the periodic lengths: a SLAB has no c (CRYSTAL sets it to 500
        # formally) and a POLYMER no b or c; a volume only for a 3D cell
        for key, periodic in zip(("a", "b", "c"), s["pbc"]):
            if periodic:
                db.store_material_property(mid, f"lattice_{key}", s["lattice"][key], "structural",
                                           property_unit="Angstrom", extractor_script="mace gui example")
        if all(s["pbc"]):
            db.store_material_property(mid, "volume", s["lattice"]["volume"], "structural",
                                       property_unit="Angstrom^3", extractor_script="mace gui example")
        return s

    for name in ("MgO", "LiF"):
        add_structure(name, f"structures/{name}.cif", "cif")
        db.create_calculation(name, "OPT", input_file=f"{name}_opt.d12",
                              settings={"functional": "B3LYP-D3", "basis_set": "POB-TZVP-REV2"})

    s = add_structure("Si", "outputs/Si_sp.out", "d12")
    calc = db.create_calculation("Si", "SP", input_file="outputs/Si_sp.d12",
                                 settings={"functional": "PBE0", "basis_set": "SVP (external)"})
    db.update_calculation_status(calc, "completed", output_file="outputs/Si_sp.out")
    db.store_material_property("Si", "total_energy", s["energy_au"], "energy", calc_id=calc,
                               property_unit="Ha", extractor_script="mace gui example")

    s = add_structure("graphene_slab", "outputs/graphene_slab_sp.out", "d12")
    calc = db.create_calculation("graphene_slab", "SP", input_file="outputs/graphene_slab_sp.d12",
                                 settings={"functional": "HSEsol-3c", "basis_set": "SOLDEF2MSVP"})
    db.update_calculation_status(calc, "completed", output_file="outputs/graphene_slab_sp.out")
    db.store_material_property("graphene_slab", "total_energy", s["energy_au"], "energy", calc_id=calc,
                               property_unit="Ha", extractor_script="mace gui example")

    s = add_structure("Ag2Br3", "outputs/Ag2Br3_opt_unfinished.out", "d12")
    calc = db.create_calculation("Ag2Br3", "OPT", input_file="outputs/Ag2Br3_opt_unfinished.d12",
                                 settings={"functional": "B3LYP-D3", "optimization": "FULLOPTG"})
    db.update_calculation_status(calc, "failed", output_file="outputs/Ag2Br3_opt_unfinished.out",
                                 error_type="timeout",
                                 error_message="walltime reached during the optimization")

    # one self-contained file: fold the WAL back in and leave journal mode DELETE
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("VACUUM")
    conn.close()
    shutil.rmtree(scratch, ignore_errors=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="Without --suite: the `mace gui` example folder (examples/gui).")
    ap.add_argument("--suite", action="store_true",
                    help="build the five-archetype suite (examples/suite, build_suite.py) instead")
    ap.add_argument("--corpus", default=None,
                    help="--suite: the CRYSTAL output corpus (default: $MACE_TEST_CORPUS, else the repository's test/)")
    ap.add_argument("--results", default=None, help="finished runs of deck-only cells: <archetype folder>/<entry>/<calc>/")
    a = ap.parse_args()
    if a.suite:
        # the five-archetype suite (examples/suite); needs the output corpus: --corpus PATH
        import build_suite
        build_suite.build(Path(a.corpus or __import__("os").environ.get("MACE_TEST_CORPUS") or REPO / "test"),
                          Path(a.results) if a.results else None)
        return
    (HERE / "structures").mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cifs()
        copies()
        database()
    print(f"example folder written to {HERE}")


if __name__ == "__main__":
    main()
