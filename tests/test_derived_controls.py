"""Derived properties: turning them off, failures that keep what is stored,
the registry under threads, CPHF runs, the functional a deck names, and
the reference energies' normalization.

* ``--no-derived``, ``--skip-property NAME`` and ``MACE_DERIVED=0`` compute
  nothing (or not that property) and keep the stored rows; the queue's
  completion callback and the MCP results tool skip the slow Bader and
  packing-fraction properties; derived properties run under a BLAS/OpenMP
  thread cap.
* A reference lookup that raises is an ``error`` with its reason (not
  "needs the reference run").
* ``registry.load()`` from several threads at once gives every thread the
  whole registry.
* ``store_material_property`` inside ``transaction()`` does not commit it.
* A standalone CPHF run is an SCF run (real HPCC CRYSTAL23 output,
  tests/data/hpcc_116/mechanics/dia_cphf_scf.out: the full run's SCF, its
  CPHF results table and its end).
* A deck's ``EXCHANGE``/``CORRELAT`` pair is its functional ("PBE/PBE"), so
  PBE and LDA runs are never combined.
* Every reference energy is normalized: per atom (formation, cohesive), per
  area (surface), per adsorbate (adsorption), per reaction as written
  (reaction). The same material in another cell size gives the same values,
  and a run without FREQ thermodynamics borrows only the thermal correction
  of its material's FREQ run, scaled by their atoms."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.properties import reference as RF
from mace.properties import registry as R
from mace.properties.defs import run_info
from test_props_reference import AREA, Fake, run

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
CPHF = MECH / "dia_cphf_scf.out"


def _env(**extra):
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DERIVED",
                      "MACE_SKIP_PROPERTIES", "MACE_DERIVED_EXPENSIVE")}
    e["OMP_NUM_THREADS"] = "1"
    e.update(extra)
    return e


def _cli(cwd, *args, ok=True, **env):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=_env(**env), cwd=str(cwd), timeout=600)
    if ok:
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _rows(db):
    conn = sqlite3.connect(db)
    try:
        return {n: (v, t) for n, v, t in conn.execute(
            "SELECT property_name, property_value, property_value_text FROM properties WHERE calc_id IS NOT NULL")}
    finally:
        conn.close()


def _prov(rows):
    return json.loads(rows["extraction_provenance"][1])["derived"]


# ------------------------------------------------------------------ turning derived properties off
def test_no_derived_and_skip_property_keep_the_stored_rows(tmp_path):
    pytest.importorskip("ase")                              # packing_fraction reads the structure with ase
    shutil.copy(CPHF, tmp_path / CPHF.name)
    shutil.copy(MECH / "dia_cphf.d12", tmp_path / "dia_cphf_scf.d12")
    out, db = str(tmp_path / CPHF.name), str(tmp_path / "m.db")
    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", db, "--register")
    full = _rows(db)
    assert full["packing_fraction_covalent"][0] > 0 and "energy_per_atom_ev" in full

    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", db,
         "--skip-property", "packing_fraction")
    rows = _rows(db)
    assert rows["packing_fraction_covalent"] == full["packing_fraction_covalent"]
    e = _prov(rows)["packing_fraction"]
    assert e["status"] == "skipped" and "turned off" in e["reason"]
    assert _prov(rows)["energy_per_atom"]["status"] == "stored"

    for args, env in (((("--no-derived",)), {}), ((), {"MACE_DERIVED": "0"})):
        _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", db, *args, **env)
        rows = _rows(db)
        assert set(full) <= set(rows), sorted(set(full) - set(rows))
        assert rows["energy_per_atom_ev"] == full["energy_per_atom_ev"]

    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", db,
         MACE_SKIP_PROPERTIES="packing_fraction,run_info")
    prov = _prov(_rows(db))
    assert prov["run_info"]["status"] == prov["packing_fraction"]["status"] == "skipped"

    p = _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", db,
             "--skip-property", "bader", ok=False)
    assert p.returncode == 2 and "'bader': not a derived property" in p.stderr


def test_the_mcp_tool_and_the_queue_skip_the_slow_properties(tmp_path, monkeypatch):
    pytest.importorskip("ase")                              # packing_fraction reads the structure with ase
    from mace.mcp_server.tools import results
    assert set(R.EXPENSIVE) == {"bader_charges", "packing_fraction"}
    monkeypatch.delenv("MACE_DERIVED_EXPENSIVE", raising=False)
    e = results._extract(CPHF)["extraction_provenance"]["derived"]["packing_fraction"]
    assert e["status"] == "skipped" and "turned off" in e["reason"]
    monkeypatch.setenv("MACE_DERIVED_EXPENSIVE", "1")
    assert results._extract(CPHF)["extraction_provenance"]["derived"]["packing_fraction"]["status"] == "stored"
    monkeypatch.delenv("MACE_DERIVED_EXPENSIVE")

    shutil.copy(CPHF, tmp_path / CPHF.name)
    db = tmp_path / "m.db"
    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", str(tmp_path / CPHF.name),
         "--db-path", str(db), "--register")
    conn = sqlite3.connect(db)
    cid, mid = conn.execute("SELECT calc_id, material_id FROM calculations").fetchone()
    conn.close()
    from mace.queue.manager import EnhancedCrystalQueueManager
    qm = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    qm.db_path = str(db)
    qm.extract_and_store_properties({"calc_id": cid, "material_id": mid, "output_file": str(tmp_path / CPHF.name)})
    rows = _rows(db)
    e = _prov(rows)["packing_fraction"]
    assert e["status"] == "skipped" and "turned off" in e["reason"]
    assert "packing_fraction_covalent" in rows             # the value the build stored stays


def test_derived_properties_run_under_a_thread_cap(monkeypatch):
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy  # noqa: F401  (its BLAS pool)
    monkeypatch.setenv("MACE_DERIVED_THREADS", "1")
    with R.thread_limit():
        info = threadpoolctl.threadpool_info()
        assert all(p["num_threads"] == 1 for p in info), info


# ------------------------------------------------------------------ failures and threads
def test_a_lookup_that_raises_is_an_error_with_its_reason(tmp_path):
    def boom(ctx, conn):
        raise RuntimeError("disk I/O error")
    R.load()
    R.register(R.DerivedProperty(
        name="zz_lookup_probe", calc_types=("SP",), category="energy", compute=lambda ctx: {"zz_probe_ev": 1.0},
        outputs=(R.Output("zz_probe_ev", "eV"),), references=(R.Reference("bulk_probe", find=boom),),
        module="tests"))
    try:
        out = tmp_path / "r.out"
        out.write_text("")
        _, rep = R.run_derived(out, "", {"calculation_type": "SP"}, only=["zz_lookup_probe"], conn=lambda: object())
        e = rep["zz_lookup_probe"]
        assert e["status"] == "error" and "disk I/O error" in e["reason"] and "bulk_probe" in e["reason"]
    finally:
        R.unregister("zz_lookup_probe")


def test_the_registry_loads_whole_in_every_thread():
    code = textwrap.dedent("""
        import threading
        from mace.properties import registry
        seen = []
        go = threading.Barrier(8)
        def worker():
            go.wait()
            seen.append(len(registry.properties()))
        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        print(sorted(set(seen)), len(registry.properties()))
    """)
    for _ in range(3):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                           env={**_env(), "PYTHONPATH": str(REPO_ROOT)}, cwd=str(REPO_ROOT))
        assert p.returncode == 0, p.stderr[-2000:]
        seen, total = p.stdout.strip().rsplit(" ", 1)
        assert seen == f"[{total}]", p.stdout


def test_store_material_property_inside_a_transaction_does_not_commit_it(tmp_path):
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "m.db"), ase_db_path=str(tmp_path / "s.db"))
    db.create_material("X", "C")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.store_material_property("X", "reference_runs", {"a": 1}, "computational", calc_id="c1")
            db.store_material_property("X", "big", {"b": 2}, "computational", calc_id="c1", blob=True)
            raise RuntimeError("the rest of the output's rows failed")
    conn = sqlite3.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT count(*) FROM properties WHERE material_id = 'X'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM property_blobs WHERE material_id = 'X'").fetchone()[0] == 0
    finally:
        conn.close()
    db.store_material_property("X", "reference_runs", {"a": 1}, "computational", calc_id="c1")   # outside: committed
    conn = sqlite3.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT count(*) FROM properties WHERE material_id = 'X'").fetchone()[0] == 1
    finally:
        conn.close()


# ------------------------------------------------------------------ CPHF runs
def test_a_cphf_run_is_an_scf_run():
    pytest.importorskip("ase")                              # packing_fraction reads the structure with ase
    from mace.properties.defs import scf_summary
    from mace.utils.property_extractor import CrystalPropertyExtractor
    import contextlib
    import io
    assert "CPHF" in scf_summary.SCF_RUNS and "CPHF" in RF.REF_TYPES
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        p = x.extract_all_properties(CPHF)
    assert p["calculation_type"] == "CPHF"
    assert p["total_energy_corrected_au"] == pytest.approx(-76.21718774702447)
    rep = p["extraction_provenance"]["derived"]
    assert rep["packing_fraction"]["status"] == "stored"
    assert {"mulliken_summary", "dipole_moment", "formation_energy", "cohesive_energy", "reaction_energy"} <= set(rep)


# ------------------------------------------------------------------ the functional of a deck
def _deck(tmp_path, name, dft):
    d = tmp_path / f"{name}.d12"
    d.write_text("test\nCRYSTAL\n0 0 0\n225\n3.567\n1\n6 0 0 0\nEND\nBASISSET\nPOB-TZVP-REV2\n" + dft
                 + "\nEND\nEND\n")
    d.with_suffix(".out").write_text("")
    return d.with_suffix(".out")


@pytest.mark.parametrize("dft, want", [
    ("DFT\nEXCHANGE\nPBE\nCORRELAT\nPBE", "PBE/PBE"),
    ("DFT\nSPIN\nEXCHANGE\nLDA\nCORRELAT\nVWN\nXLGRID", "LDA/VWN"),
    ("DFT\nEXCHANGE\nPBE\nCORRELAT\nPBE\nHYBRID\n25", "PBE/PBE HYBRID 25"),
    ("DFT\nEXCHANGE\nBECKE\nCORRELAT\nLYP\nNONLOCAL\n0.9 0.81\nHYBRID\n20.0", "BECKE/LYP HYBRID 20"),
    ("DFT\nEXCHANGE\nPBE", "PBE/NONE"),
    ("DFT\nSPIN\nB3LYP-D3\nXLGRID", "B3LYP-D3"),
    ("DFT\nSPIN\nHSESOL3C\nXLGRID", "HSESOL3C"),
    ("DFT\nRADIAL\n1\n4.0\nANGULAR\n1\n9999.0\nPBE0", "PBE0"),
])
def test_the_functional_of_a_deck(tmp_path, dft, want):
    assert run_info.functional("", _deck(tmp_path, "a", dft)) == want


def test_pbe_and_lda_are_never_combined(tmp_path):
    common = {"basis_set": "POB-TZVP-REV2", "crystal_version": "CRYSTAL23 public : 1.0.1",
              "total_energy_au": -76.0, "atoms_in_unit_cell": 2}

    def part(label, fn):
        return RF.Part(label, {**common, "dft_functional": fn})
    pbe, lda, pbexc = part("a", "PBE/PBE"), part("b", "LDA/VWN"), part("c", "PBEXC")
    refuse, _ = RF.check(pbe, lda)
    assert refuse and "the functional differs" in refuse[0]
    assert not RF.check(pbe, pbexc)[0]                       # CRYSTAL's shorthand for the same pair
    assert RF.check(pbe, part("d", "PBE/PBE HYBRID 25"))[0]  # PBE0-like: another functional
    expr = part("e", "HARTREE+FOCK EXCH*0.00000+(PBE        EXCH)*1.00000+PBE        CORR")
    _, warn = RF.check(pbe, expr)
    assert any("not compared" in w for w in warn)
    # the real decks the reviewer wrote: a PBE and an LDA diamond
    a, b = (_deck(tmp_path, n, f"DFT\nEXCHANGE\n{x}\nCORRELAT\n{c}") for n, x, c in
            (("pbe", "PBE", "PBE"), ("lda", "LDA", "VWN")))
    with pytest.raises(R.Skip, match="the functional differs: pbe.out PBE/PBE, lda.out LDA/VWN"):
        RF.guard([RF.Part("pbe.out", common, output_file=a), RF.Part("lda.out", common, output_file=b)])


# ------------------------------------------------------------------ normalization: the cell does not matter
G, H = -0.004, 0.006          # thermal corrections per atom (G - EL, H - EL), made up


def _add(fake, mid, formula, dim, e_per_atom, comp, n=1, cell=None, **kw):
    """A run of ``n`` cells of ``comp`` (its energy, thermodynamics and cell n times one cell's)."""
    comp = Counter({k: v * n for k, v in comp.items()})
    atoms = sum(comp.values())
    e = e_per_atom * atoms
    extra = {}
    if dim == 3:
        extra["primitive_cell_volume"] = 100.0 * n
    elif dim == 2:
        extra["primitive_cell_area"] = (cell or AREA) * n
    extra.update(kw)
    return fake.add(mid, formula, dim, e, dict(comp), thermo=(e + G * atoms, e + H * atoms),
                    electronic_energy_au=e, **extra)


def _run(tmp_path, fake, dim, e_per_atom, comp, only, refs, n=1, cell=None):
    comp = {k: v * n for k, v in comp.items()}
    atoms = sum(comp.values())
    e = e_per_atom * atoms
    props = {"total_energy_corrected_au": e, "gibbs_free_energy_au": e + G * atoms, "enthalpy_au": e + H * atoms,
             "thermodynamic_temperature_k": 298.15, "thermodynamic_pressure_mpa": 0.101325}
    if dim == 2:
        props["primitive_cell_area"] = (cell or AREA) * n
    conn = fake.conn()
    try:
        vals, e = run(tmp_path, props, comp, dim, only, refs=refs, conn=conn)
    finally:
        conn.close()
    assert e["status"] == "stored", e
    return vals


def _forms(vals, prefix, unit):
    return [vals[f"{prefix}_{f}_{unit}"] for f in ("energy", "enthalpy", "gibbs_energy")]


@pytest.fixture
def cells(tmp_path):
    f = Fake(tmp_path)
    f.ids = {
        "mg": _add(f, "Mg", "Mg", 3, -200.05, {"Mg": 2}),
        "mg8": _add(f, "Mg", "Mg", 3, -200.05, {"Mg": 2}, n=8),
        "o2": _add(f, "O2", "O", 0, -75.15, {"O": 2}),
        "mg_at": _add(f, "Mg-atom", "Mg", 0, -199.8, {"Mg": 1}),
        "o_at": _add(f, "O-atom", "O", 0, -74.9, {"O": 1}),
        "bulk": _add(f, "MgO", "MgO", 3, -137.75, {"Mg": 4, "O": 4}),
        "bulk3": _add(f, "MgO", "MgO", 3, -137.75, {"Mg": 4, "O": 4}, n=3),
        "slab": _add(f, "MgO-slab", "MgO", 2, -137.74, {"Mg": 8, "O": 8}),
        "slab2": _add(f, "MgO-slab", "MgO", 2, -137.74, {"Mg": 8, "O": 8}, n=2),
        "co2": _add(f, "CO2", "CO2", 0, -62.9, {"C": 1, "O": 2}),
        "c": _add(f, "C", "C", 3, -38.0, {"C": 2}),
    }
    return f


def test_formation_and_cohesive_energies_are_per_atom(tmp_path, cells):
    i = cells.ids
    for prop, prefix, refs, refs_big in (
            ("formation_energy", "formation", {"elements": [i["mg"], i["o2"]]},
             {"elements": [i["mg8"], i["o2"]]}),
            ("cohesive_energy", "cohesive", {"atoms": [i["mg_at"], i["o_at"]]}, None)):
        one = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 4, "O": 4}, prop, refs), prefix, "ev_per_atom")
        sup = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 4, "O": 4}, prop, refs, n=2), prefix, "ev_per_atom")
        assert one[0] != 0 and sup == pytest.approx(one, rel=1e-12, abs=1e-12)
        if refs_big:     # a reference in a supercell
            big = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 4, "O": 4}, prop, refs_big), prefix, "ev_per_atom")
            assert big == pytest.approx(one, rel=1e-12, abs=1e-12)


def test_the_surface_energy_is_per_area(tmp_path, cells):
    i = cells.ids
    vals = [_forms(_run(tmp_path, cells, 2, -137.74, {"Mg": 8, "O": 8}, "surface_energy", {"bulk": [b]}, n=n),
                   "surface", "j_m2")
            for b, n in ((i["bulk"], 1), (i["bulk3"], 1), (i["bulk"], 2), (i["bulk3"], 2))]
    assert vals[0][0] > 0
    for v in vals[1:]:
        assert v == pytest.approx(vals[0], rel=1e-9)


def test_the_adsorption_energy_is_per_adsorbate(tmp_path, cells):
    i = cells.ids
    one = _run(tmp_path, cells, 2, -110.0, {"Mg": 8, "O": 10, "C": 1}, "adsorption_energy",
               {"surface": [i["slab"]], "adsorbate": [i["co2"]]})
    two = _run(tmp_path, cells, 2, -110.0, {"Mg": 8, "O": 10, "C": 1}, "adsorption_energy",
               {"surface": [i["slab2"]], "adsorbate": [i["co2"]]}, n=2)
    assert one["adsorption_energy"]["k_adsorbates"] == 1 and two["adsorption_energy"]["k_adsorbates"] == 2
    assert _forms(two, "adsorption", "ev") == pytest.approx(_forms(one, "adsorption", "ev"), rel=1e-9)


def test_the_reaction_energy_is_per_reaction_as_written(tmp_path, cells):
    i = cells.ids
    # Mg + 1/2 O2 -> MgO; the Mg reference as a supercell: 1/8 of it, the same reaction
    one = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 1, "O": 1}, "reaction_energy",
                      {"reactants": [i["mg"], i["o2"]]}), "reaction", "ev")
    big = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 1, "O": 1}, "reaction_energy",
                      {"reactants": [i["mg8"], i["o2"]]}), "reaction", "ev")
    assert big == pytest.approx(one, rel=1e-12)
    # the run as a 4-cell supercell: 4 reactions as written
    four = _forms(_run(tmp_path, cells, 3, -137.75, {"Mg": 1, "O": 1}, "reaction_energy",
                       {"reactants": [i["mg"], i["o2"]]}, n=4), "reaction", "ev")
    assert four == pytest.approx([4 * v for v in one], rel=1e-12)


def test_one_material_in_two_cell_sizes_reacts_to_nothing(tmp_path, cells):
    vals = _run(tmp_path, cells, 3, -200.05, {"Mg": 2}, "reaction_energy", {"reactants": [cells.ids["mg8"]]})
    assert vals["reaction_energy"]["equation"] == "1/8 Mg16 → 1 Mg2"
    assert _forms(vals, "reaction", "ev") == pytest.approx([0, 0, 0], abs=1e-9)


def test_borrowed_thermodynamics_are_the_scaled_thermal_correction(tmp_path):
    f = Fake(tmp_path)
    e2, e16 = -76.4302, -611.4401                          # the supercell's energy is not 8 x the cell's
    run_id = f.add("C", "C", 3, e2, {"C": 2})              # an OPT run: no thermodynamics of its own
    f.add("C", "C", 3, e16, {"C": 16}, calc_type="FREQ", thermo=(e16 - 0.012, e16 + 0.035),
          electronic_energy_au=e16)
    conn = f.conn()
    try:
        part = RF.Part("run", RF.light_values(conn, [run_id])[run_id], calc_id=run_id, material_id="C", via="database")
        th, why = RF.thermo(part, conn)
        assert not why
        assert th["G"] == pytest.approx(e2 - 0.012 / 8, rel=1e-14)
        assert th["H"] == pytest.approx(e2 + 0.035 / 8, rel=1e-14)
        assert th["borrowed"]["scale"] == pytest.approx(1 / 8)
        # a FREQ run whose composition is not a multiple of the run's lends nothing
        f.add("CO", "CO", 3, -113.0, {"C": 1, "O": 1})
        f.add("CO", "CO", 3, -340.0, {"C": 2, "O": 3}, calc_type="FREQ", thermo=(-340.01, -339.98))
        co_id = conn.execute("SELECT calc_id FROM calculations WHERE material_id = 'CO' AND calc_type = 'OPT'"
                             ).fetchone()[0]
        co = RF.Part("co", RF.light_values(conn, [co_id])[co_id], calc_id=co_id, material_id="CO", via="database")
        th, why = RF.thermo(co, conn)
        assert th is None and "is not a multiple of CO" in why
    finally:
        conn.close()
