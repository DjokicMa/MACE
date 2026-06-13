"""Regression tests for the ContextualMaterialDatabase / property-store path.

These exercise code that was latent (no live callers yet) but contained four
bugs that would crash the moment the workflow-isolation features were wired up:

1. ``MaterialDatabase.store_material_property`` inserted a string UUID into an
   ``INTEGER PRIMARY KEY AUTOINCREMENT`` column -> SQLite datatype mismatch.
2. ``ContextualMaterialDatabase.copy_to_context`` called a non-existent
   ``create_or_update_material`` (the method is ``create_material``).
3. ``copy_to_context`` passed ``unit=``/``conditions=`` to
   ``store_material_property`` whose real params are ``property_unit`` (and there
   is no ``conditions`` column) -> TypeError.
4. ``get_workflow_materials``/``get_workflow_calculations`` called
   ``self.get_connection()`` (the real method is ``_get_connection``).

Everything here is pure-SQLite, so it always runs (no test/ corpus needed).
"""
import pytest

from mace.database.materials import MaterialDatabase
from mace.database.materials_contextual import ContextualMaterialDatabase
from mace.workflow.context import WorkflowContext


def test_store_material_property_autoincrements_integer_pk(tmp_path):
    """property_id must autoincrement; storing must not raise a datatype mismatch."""
    db = MaterialDatabase(db_path=str(tmp_path / "m.db"),
                          ase_db_path=str(tmp_path / "s.db"))
    db.create_material(material_id="mat1", formula="Si2")

    pid1 = db.store_material_property("mat1", "band_gap", 1.12, property_unit="eV")
    pid2 = db.store_material_property("mat1", "total_energy", -42.0,
                                     property_unit="eV")

    # Autoincremented integer PKs, not string UUIDs.
    assert isinstance(pid1, int)
    assert isinstance(pid2, int)
    assert pid2 != pid1

    props = {p["property_name"]: p for p in db.get_material_properties("mat1")}
    assert props["band_gap"]["property_value"] == pytest.approx(1.12)
    assert props["band_gap"]["property_unit"] == "eV"
    # Text-valued properties go into the text column without crashing.
    pid3 = db.store_material_property("mat1", "spacegroup_symbol", "Fd-3m")
    assert isinstance(pid3, int)
    props = {p["property_name"]: p for p in db.get_material_properties("mat1")}
    assert props["spacegroup_symbol"]["property_value"] == "Fd-3m"


def test_get_workflow_materials_and_calculations(tmp_path):
    """The workflow filters must use _get_connection (no AttributeError) and
    actually filter by the workflow_id embedded in settings_json."""
    db = ContextualMaterialDatabase(db_path=str(tmp_path / "m.db"),
                                    ase_db_path=str(tmp_path / "s.db"),
                                    auto_detect_context=False)
    db.create_material(material_id="matA", formula="C")
    db.create_material(material_id="matB", formula="O2")

    db.create_calculation("matA", "OPT", settings={"workflow_id": "wf_keep"})
    db.create_calculation("matB", "SP", settings={"workflow_id": "wf_other"})

    mats = db.get_workflow_materials("wf_keep")
    assert [m["material_id"] for m in mats] == ["matA"]

    calcs = db.get_workflow_calculations("wf_keep")
    assert len(calcs) == 1
    assert calcs[0]["calc_type"] == "OPT"


def test_copy_to_context_round_trips_materials_calcs_and_properties(tmp_path):
    """End-to-end copy across two isolated contexts must move materials,
    calculations, and properties without raising — and JSON columns must not be
    double-encoded."""
    src_ctx = WorkflowContext("src", base_dir=tmp_path, isolation_mode="isolated")
    src_ctx.initialize()
    tgt_ctx = WorkflowContext("tgt", base_dir=tmp_path, isolation_mode="isolated")
    tgt_ctx.initialize()

    src = ContextualMaterialDatabase(workflow_context=src_ctx,
                                     auto_detect_context=False)
    src.create_material(material_id="mat1", formula="Si2",
                        space_group=227, metadata={"note": "seed"})
    src.create_calculation("mat1", "OPT",
                           settings={"workflow_id": "src", "functional": "PBE"})
    src.store_material_property("mat1", "band_gap", 0.66, property_unit="eV",
                               property_category="Electronic")

    src.copy_to_context(tgt_ctx)

    tgt = ContextualMaterialDatabase(workflow_context=tgt_ctx,
                                     auto_detect_context=False)

    mat = tgt.get_material("mat1")
    assert mat is not None
    assert mat["formula"] == "Si2"
    assert mat["space_group"] == 227
    # metadata round-trips as a single-encoded JSON object, not a quoted string.
    assert json_loads(mat["metadata_json"]) == {"note": "seed"}

    calcs = tgt.get_calculations_by_material("mat1")
    assert len(calcs) == 1
    assert json_loads(calcs[0]["settings_json"]) == {"workflow_id": "src",
                                                     "functional": "PBE"}

    props = {p["property_name"]: p for p in tgt.get_material_properties("mat1")}
    assert props["band_gap"]["property_value"] == pytest.approx(0.66)
    assert props["band_gap"]["property_unit"] == "eV"
    assert props["band_gap"]["property_category"] == "Electronic"


def json_loads(value):
    import json
    return json.loads(value) if value is not None else None
