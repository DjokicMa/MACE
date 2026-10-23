"""The derived-property registry (mace/properties): what a registration must
declare, how run_derived decides what applies (required keys, sibling files,
reference runs), that an extraction stores registered properties with the
run's rows (a JSON curve in property_blobs) and replaces them on the next
extraction, that an interactive Save to Database survives it, and the
built-in definitions on real outputs: the run card's values, the band-edge
effective masses of a BAND run (mace tau's automatic masses) and a FREQ
run's thermodynamics at every temperature it printed. With the gitignored
test/ corpus present: a sample of its outputs extracts the same keys and
values as without the registry, plus only registered new keys."""
import contextlib
import io
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.properties import registry as R
from mace.properties import Output, Reference, RunRef, Sibling, Skip
from mace.utils import property_keys as PK
from mace.utils.property_extractor import CrystalPropertyExtractor

MACE_CLI = REPO_ROOT / "mace_cli"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    return ES.copy_suite(tmp_path_factory.mktemp("suite") / "s")


@pytest.fixture
def temp_prop():
    """Register test properties; unregistered afterwards."""
    names = []

    def reg(name, **kw):
        fn = kw.pop("fn")
        R.derived(name, **kw)(fn)
        names.append(name)
        return R.get(name)
    yield reg
    for n in names:
        R.unregister(n)


def _extract(path, **kw):
    """Offline extraction (no database); ``kw`` set extractor attributes."""
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    for k, v in kw.items():
        setattr(x, k, v)
    with contextlib.redirect_stdout(io.StringIO()):
        return x, x.extract_all_properties(Path(path))


# ------------------------------------------------------------------ declarations
def test_a_registration_declares_what_it_writes(temp_prop):
    ok = dict(calc_types=("SP",), category="energy", fn=lambda ctx: {})
    with pytest.raises(R.RegistryError, match="extractor itself writes"):
        temp_prop("t_clash", outputs=(Output("total_energy_au", "Hartree"),), **ok)
    with pytest.raises(R.RegistryError, match="category"):
        temp_prop("t_cat", outputs=(Output("t_cat_x", "eV"),), calc_types=("SP",), category="nope", fn=lambda c: {})
    with pytest.raises(R.RegistryError, match="bool"):
        temp_prop("t_bool", outputs=(Output("t_bool_x", "eV", "bool"),), **ok)
    with pytest.raises(R.RegistryError, match="property_blobs"):
        temp_prop("t_blob", outputs=(Output("t_blob_x", "eV", blob=True),), **ok)
    with pytest.raises(R.RegistryError, match="calc_types"):
        temp_prop("t_none", outputs=(Output("t_none_x", "eV"),), calc_types=(), category="energy", fn=lambda c: {})
    temp_prop("t_one", outputs=(Output("t_one_x", "eV"),), **ok)
    with pytest.raises(R.RegistryError, match="already written"):
        temp_prop("t_two", outputs=(Output("t_one_x", "eV"),), **ok)
    # the key is a database key now: category, unit and type from the declaration
    assert PK.key("t_one_x") == PK.Key("energy", "eV", "float", PK.key("t_one_x").description)
    assert PK.known("t_one_x") and PK.category("t_one_x") == "energy" and PK.unit("t_one_x") == "eV"
    R.unregister("t_one")
    assert not PK.known("t_one_x")


def test_builtin_definitions_load_and_are_well_formed():
    reg = R.load()
    assert {"energy_per_atom", "run_info", "effective_masses", "thermodynamics_vs_temperature",
            "relaxation_time_dpt"} <= set(reg)
    for p in reg.values():
        assert p.description and p.source and p.group, p.name
        for o in p.outputs:
            assert PK.key(o.name).unit == o.unit and PK.is_blob(o.name) == o.blob, o.name
            assert o.name not in PK.KEYS                     # never one of the extractor's own names
    assert R.get("relaxation_time_dpt").interactive
    assert sorted(p.name for p in R.properties("FREQ", interactive=False)) == [
        "adsorption_energy", "cohesive_energy", "dipole_moment", "energy_per_atom", "formation_energy",
        "mean_square_displacement", "mechanics_derived", "mulliken_summary", "packing_fraction",
        "phonon_debye_temperature", "phonon_sound_velocities", "reaction_energy", "run_info", "surface_energy",
        "thermodynamics_vs_temperature"]


# ------------------------------------------------------------------ running
def test_what_applies_and_why_not(tmp_path, temp_prop):
    out = tmp_path / "x_band.out"
    out.write_text("a run\n")
    dat = tmp_path / "x_band.EXTRA.DAT"
    calls = []

    def need_key(ctx):
        calls.append("need_key")
        return {"t_a": ctx.value("total_energy_au") * 2}
    temp_prop("t_need_key", calc_types=("SP",), category="energy", requires=("total_energy_au",),
              outputs=(Output("t_a", "Hartree"),), fn=need_key)
    temp_prop("t_sibling", calc_types=("*",), category="energy",
              siblings=(Sibling("EXTRA", find=lambda p: dat if dat.is_file() else None),),
              outputs=(Output("t_b", "dimensionless", "int"),), fn=lambda ctx: {"t_b": len(ctx.sibling("EXTRA").read_text())})
    temp_prop("t_skip", calc_types=("SP",), category="energy", outputs=(Output("t_c", "eV"),),
              fn=lambda ctx: (_ for _ in ()).throw(Skip("not this one")))
    temp_prop("t_boom", calc_types=("SP",), category="energy", outputs=(Output("t_d", "eV"),),
              fn=lambda ctx: 1 / 0)
    temp_prop("t_extra", calc_types=("SP",), category="energy", outputs=(Output("t_e", "eV"),),
              fn=lambda ctx: {"t_e": 1.0, "t_undeclared": 2})
    temp_prop("t_nan", calc_types=("SP",), category="energy", outputs=(Output("t_f", "eV"), Output("t_g", "eV")),
              fn=lambda ctx: {"t_f": float("nan"), "t_g": None})
    only = ["t_need_key", "t_sibling", "t_skip", "t_boom", "t_extra", "t_nan"]

    vals, rep = R.run_derived(out, "", {"calculation_type": "BAND"}, only=only)
    assert set(rep) == {"t_sibling"} and rep["t_sibling"]["status"] == "skipped"
    assert rep["t_sibling"]["reason"] == "no EXTRA file beside the run"

    dat.write_text("12345")
    vals, rep = R.run_derived(out, "", {"calculation_type": "SP"}, only=only)
    assert vals == {"t_b": 5}
    assert rep["t_sibling"]["inputs"] == ["x_band.out", "x_band.EXTRA.DAT"]
    assert rep["t_need_key"] == {"version": "1", "status": "skipped", "reason": "the run has no total_energy_au"}
    assert calls == []
    assert rep["t_skip"]["reason"] == "not this one"
    assert rep["t_boom"]["status"] == "error" and "ZeroDivisionError" in rep["t_boom"]["reason"]
    assert rep["t_extra"]["status"] == "error" and "t_undeclared" in rep["t_extra"]["reason"]
    assert rep["t_nan"]["status"] == "skipped"            # NaN and None are not stored

    vals, rep = R.run_derived(out, "", {"calculation_type": "SP", "total_energy_au": -1.5}, only=only)
    assert vals["t_a"] == -3.0 and rep["t_need_key"]["status"] == "stored"


def test_reference_runs_given_or_found_in_the_database(tmp_path, temp_prop):
    """How a property declares other runs (the reference-energy properties
    build on it): given by the user (a file or a calc_id), else found by
    the property's own lookup in the database, else skipped."""
    def surface(ctx):
        bulk = ctx.reference("bulk")
        return {"t_surface": ctx.value("total_energy_au") - bulk.value("total_energy_au")}

    def find_bulk(ctx, conn):
        # the same material's newest OPT (registry.db_runs: none without a database)
        return R.db_runs(conn, calc_types=("OPT",), formula="C2", status=None)
    temp_prop("t_surface", calc_types=("SP",), category="energy", requires=("total_energy_au",),
              references=(Reference("bulk", calc_types=("OPT",), find=find_bulk),),
              outputs=(Output("t_surface", "Hartree"),), fn=surface)
    out = tmp_path / "slab.out"
    out.write_text("")
    props = {"calculation_type": "SP", "total_energy_au": -10.0}

    vals, rep = R.run_derived(out, "", props, only=["t_surface"])
    assert rep["t_surface"]["status"] == "skipped" and "reference run bulk" in rep["t_surface"]["reason"]

    # given: a RunRef with its values
    vals, rep = R.run_derived(out, "", props, only=["t_surface"],
                              references={"bulk": RunRef(output_file=Path("bulk.out"), properties={"total_energy_au": -4.0})})
    assert vals == {"t_surface": -6.0} and rep["t_surface"]["inputs"] == ["slab.out", "bulk.out"]

    # found in the database: the lookup gets a read-only connection, the values are the stored rows
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "m.db"), ase_db_path=str(tmp_path / "s.db"))
    db.create_material("bulk", "C2")
    cid = db.create_calculation("bulk", "OPT")
    db.store_material_property("bulk", "total_energy_au", -3.0, "energy", calc_id=cid)
    from mace.utils import db_build
    conn = db_build.open_ro(tmp_path / "m.db")
    vals, rep = R.run_derived(out, "", props, only=["t_surface"], conn=lambda: conn)
    conn.close()
    assert vals == {"t_surface": -7.0} and rep["t_surface"]["inputs"] == ["slab.out", cid]


# ------------------------------------------------------------------ extraction and the database
def _rows(db, calc_id):
    conn = sqlite3.connect(db)
    rows = {r[0]: r[1:] for r in conn.execute(
        "SELECT property_name, property_value, property_value_text, property_unit, property_category, extractor_script "
        "FROM properties WHERE calc_id = ?", (calc_id,))}
    blobs = {r[0]: r[1:] for r in conn.execute(
        "SELECT property_name, property_value_text, property_unit, property_category, extractor_script "
        "FROM property_blobs WHERE calc_id = ?", (calc_id,))}
    conn.close()
    return rows, blobs


def test_extraction_stores_a_registered_property_and_replaces_it(tmp_path, suite, temp_prop):
    sp = suite / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out"
    temp_prop("t_twice", calc_types=("SP",), category="energy", requires=("total_energy_au",),
              outputs=(Output("t_twice_ev", "eV", label="2E"),
                       Output("t_curve", "JSON", "json", blob=True, label="a curve")),
              fn=lambda ctx: {"t_twice_ev": 2 * ctx.value("total_energy_ev"),
                              "t_curve": {"x": [1, 2, 3], "y": [ctx.value("total_energy_au")] * 3}})
    from mace.database.materials import MaterialDatabase
    dbp = tmp_path / "materials.db"
    db = MaterialDatabase(str(dbp), ase_db_path=str(tmp_path / "structures.db"))
    db.create_material("diamond", "C2")
    cid = db.create_calculation("diamond", "SP")
    x = CrystalPropertyExtractor(str(dbp), ase_db_path=str(tmp_path / "structures.db"))
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(sp, material_id="diamond", calc_id=cid)
        assert x.save_properties_to_database(props) > 0
    rows, blobs = _rows(dbp, cid)
    assert rows["t_twice_ev"][0] == pytest.approx(2 * props["total_energy_ev"])
    assert rows["t_twice_ev"][2:] == ("eV", "energy", "crystal_property_extractor.py")
    assert "t_curve" not in rows and json.loads(blobs["t_curve"][0])["x"] == [1, 2, 3]
    assert blobs["t_curve"][1:] == ("JSON", "energy", "crystal_property_extractor.py")
    prov = json.loads(rows["extraction_provenance"][1])
    assert prov["derived"]["t_twice"]["status"] == "stored" and prov["output_file"] == "diamond_sp.out"
    assert prov["mace_version"] and prov["registry_version"] == R.REGISTRY_VERSION
    # the built-in ones too
    assert rows["energy_per_atom_ev"][0] == pytest.approx(props["total_energy_au"] * 27.211386245988 / props["atoms_in_unit_cell"])
    assert rows["crystal_version"][1].startswith("CRYSTAL23")

    # a property that no longer applies loses its rows at the next extraction (the run's own rows are replaced)
    R.unregister("t_twice")
    with contextlib.redirect_stdout(io.StringIO()):
        x.save_properties_to_database(x.extract_all_properties(sp, material_id="diamond", calc_id=cid))
    rows, blobs = _rows(dbp, cid)
    assert "t_twice_ev" not in rows and "t_curve" not in blobs and "energy_per_atom_ev" in rows


def test_an_interactive_save_survives_reextraction(tmp_path, suite):
    sp = suite / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out"
    from mace.database.materials import MaterialDatabase
    dbp = tmp_path / "materials.db"
    db = MaterialDatabase(str(dbp), ase_db_path=str(tmp_path / "structures.db"))
    db.create_material("diamond", "C2")
    cid = db.create_calculation("diamond", "SP")
    res = {"carrier": "electron", "axis": "x", "tau_fs": 12.5, "mobility_cm2_Vs": 1800.0, "method": "dpt",
           "inputs": {"E1": {"value": 9.1, "unit": "eV", "source": "fit"}}, "relaxtim": "RELAXTIM\n12.5"}
    vals = R.compute_interactive("relaxation_time_dpt", sp, {"result": res})
    assert vals["relaxation_time_dpt_electron_x_fs"] == 12.5 and vals["mobility_dpt_electron_x_cm2_vs"] == 1800.0
    stored = R.store_interactive(dbp, "relaxation_time_dpt", vals, material_id="diamond", calc_id=cid)
    assert sorted(stored) == sorted(vals)
    with pytest.raises(R.RegistryError, match="computed on extraction"):
        R.store_interactive(dbp, "energy_per_atom", {"energy_per_atom_ev": 1.0}, material_id="diamond", calc_id=cid)
    x = CrystalPropertyExtractor(str(dbp), ase_db_path=str(tmp_path / "structures.db"))
    with contextlib.redirect_stdout(io.StringIO()):
        x.save_properties_to_database(x.extract_all_properties(sp, material_id="diamond", calc_id=cid))
    rows, blobs = _rows(dbp, cid)
    assert rows["relaxation_time_dpt_electron_x_fs"][0] == 12.5
    assert rows["relaxation_time_dpt_electron_x_fs"][2:] == ("fs", "transport", "mace-gui")
    assert json.loads(blobs["relaxation_time_dpt_electron_x"][0])["inputs"]["E1"]["value"] == 9.1
    assert "relaxtim" not in json.loads(blobs["relaxation_time_dpt_electron_x"][0])


def test_store_material_property_takes_json_and_blobs(tmp_path):
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "m.db"), ase_db_path=str(tmp_path / "s.db"))
    db.store_material_property("m", "a_list", [1, 2], "other")
    db.store_material_property("m", "a_blob", {"k": [1, 2]}, "other", blob=True)
    db.store_material_property("m", "a_blob", {"k": [3]}, "other", blob=True)        # replaced, not duplicated
    conn = sqlite3.connect(tmp_path / "m.db")
    assert conn.execute("SELECT property_value_text FROM properties WHERE property_name = 'a_list'").fetchone()[0] == "[1, 2]"
    assert conn.execute("SELECT property_value_text FROM property_blobs WHERE property_name = 'a_blob'").fetchall() == [('{"k": [3]}',)]


# ------------------------------------------------------------------ the built-in definitions on real outputs
def test_the_run_card_and_the_database_say_the_same(suite):
    from mace.gui import runinfo
    for rel in ("2-dense-3d/diamond/SP/diamond_sp.out", "5-molecule-0d/EC/OPT/EC_opt.out",
                "3-slab-2d/graphene/BAND/graphene_band.out"):
        out = suite / rel
        _, props = _extract(out)
        card = runinfo.meta_job(path=str(out))
        assert card["energy_ev_per_atom"] == props.get("energy_per_atom_ev"), rel
        assert card["functional"] == props.get("dft_functional"), rel
        assert card["basis"] == props.get("basis_set"), rel
        assert card["crystal_version"] == props.get("crystal_version"), rel
        assert card["mpi_processes"] == props.get("mpi_processes"), rel


def test_effective_masses_are_mace_taus_automatic_masses(suite):
    from mace.transport import mass
    band = suite / "2-dense-3d" / "diamond" / "BAND" / "diamond_band.out"
    _, props = _extract(band)
    dat = next(band.parent.glob("*.BAND.DAT"))
    for carrier in ("electron", "hole"):
        with contextlib.redirect_stdout(io.StringIO()):
            am = mass.auto_masses(carrier, None, 3, band=dat)
        assert props[f"effective_mass_{carrier}_mc"] == pytest.approx(am["m_star"])
        assert props[f"effective_mass_{carrier}_md"] == pytest.approx(am["m_dos"])
        assert props["effective_masses"][carrier]["sources"] == am["sources"]
    assert props["effective_masses"]["dimensionality"] == 3
    assert props["extraction_provenance"]["derived"]["effective_masses"]["inputs"] == [band.name, dat.name]
    # a molecule's "band structure" has no edge to fit: nothing stored, and why
    ec = suite / "5-molecule-0d" / "EC" / "SP" / "EC_sp.out"
    _, props = _extract(ec)
    assert "effective_masses" not in props


def _with_temperatures(text, temps):
    """The FREQ output with its one thermodynamics block printed at each of
    ``temps`` (CRYSTAL's TEMPERAT layout: the block repeats per T; values
    scaled so each point differs)."""
    i = text.index(" THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS")
    j = text.index(" *****", text.index("HEAT CAPACITY :", i))
    block = text[i:j]
    t0 = float(re.search(r"AT \(T =\s*([\d.]+)", block).group(1))
    out = []
    for t in temps:
        b = re.sub(r"AT \(T =\s*[\d.]+ K", f"AT (T = {t:7.2f} K", block)
        b = re.sub(r"(HEAT CAPACITY :)(.*)", lambda m: m.group(1) + "".join(
            f"{float(v) * t / t0:22.12f}" for v in m.group(2).split()), b)
        out.append(b)
    sep = "\n *******************************************************************************\n\n"
    return text[:i] + sep.join(out) + text[j:]


def test_freq_thermodynamics_at_every_temperature(tmp_path, suite):
    src = suite / "2-dense-3d" / "diamond" / "FREQ" / "diamond_freq.out"
    _, one = _extract(src)
    # one block: the scalar keys hold it, no curve
    assert "thermodynamics_vs_temperature" not in one
    assert one["extraction_provenance"]["derived"]["thermodynamics_vs_temperature"]["status"] == "skipped"
    dst = tmp_path / "diamond_freq.out"
    dst.write_text(_with_temperatures(src.read_text(), (100.0, 298.15, 600.0)))
    for d12 in src.parent.glob("*.d12"):
        (tmp_path / d12.name).write_text(d12.read_text())
    _, many = _extract(dst)
    curve = many["thermodynamics_vs_temperature"]
    assert many["thermodynamic_n_points"] == 3
    assert [p["temperature_k"] for p in curve["points"]] == [100.0, 298.15, 600.0]
    # the scalar keys are the first block's, unchanged; the curve's first point is that block
    assert many["thermodynamic_temperature_k"] == 100.0
    first = curve["points"][0]
    for k in ("gibbs_free_energy_au", "enthalpy_ev", "entropy_term_kj_mol", "entropy_j_mol_k", "heat_capacity_j_mol_k"):
        assert first[k] == pytest.approx(many[k]), k
    assert curve["points"][2]["heat_capacity_j_mol_k"] == pytest.approx(6 * first["heat_capacity_j_mol_k"])
    assert curve["units"]["gibbs_free_energy_kj_mol"] == "kJ/mol" and curve["units"]["entropy_j_mol_k"] == "J/(mol·K)"
    assert curve["zero_point_energy_au"] == many["zero_point_energy_au"]


# ------------------------------------------------------------------ the 707-output corpus
def _corpus_sample(per_dir=8):
    if not TEST_DATA.is_dir():
        return []
    outs = []
    for d in ("SP", "OPT", "FREQ", "BAND", "DOSS", "TRANSPORT", "ECH3POT3"):
        outs += sorted((TEST_DATA / d).glob("*.out"))[:per_dir]
    return outs


def test_corpus_extraction_is_unchanged_but_for_registered_keys():
    outs = _corpus_sample()
    if not outs:
        pytest.skip("test/ data corpus not present (gitignored)")
    allowed = {o.name for p in R.load().values() for o in p.outputs} | {"extraction_provenance"}
    for out in outs:
        _, base = _extract(out, run_derived=False)
        _, new = _extract(out)
        base.pop("_metadata"), new.pop("_metadata")
        assert {k: new[k] for k in base} == base, out.name
        assert set(new) - set(base) <= allowed, (out.name, set(new) - set(base) - allowed)


# ------------------------------------------------------------------ the CLI
def test_list_properties_through_mace_cli():
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "--list-properties",
                        "--calc-type", "FREQ", "--json"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = {x["name"]: x for x in json.loads(r.stdout)}
    assert rows["gibbs_free_energy_kj_mol"]["unit"] == "kJ/mol" and rows["gibbs_free_energy_kj_mol"]["derived"] is None
    assert rows["thermodynamics_vs_temperature"]["derived"] == "thermodynamics_vs_temperature"
    assert "band_fundamental_gap_ev" not in rows                 # a BAND key
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "--list-properties", "--gui"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "[gap]" in r.stdout and "[derived]" in r.stdout


def test_properties_doc_is_current():
    from mace.properties import inventory
    text = (REPO_ROOT / "docs" / "properties.md").read_text(encoding="utf-8")
    assert inventory.generated_section(text) == inventory.markdown().strip() + "\n", \
        "docs/properties.md is stale: mace analyze --list-properties --update-doc docs/properties.md"


def test_the_inventory_names_every_calc_type_a_key_comes_from():
    outs = _corpus_sample(6)
    if not outs:
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.properties import inventory
    types = {r["name"]: r["calc_types"] for r in inventory.everything()}
    for out in outs:
        _, props = _extract(out)
        ct = props.get("calculation_type")
        for k in props:
            if k.startswith("_") or not ct:
                continue
            assert "*" in types[k] or ct in types[k], (out.name, k, ct)
