"""The property key registry (mace/utils/property_keys.py) is the one list of
what the database stores: every name the extractor can write is in it, with
a category and a unit, and every name in it is one the extractor writes.

Two ways in: the names in the extractor's source (string keys put into the
property dicts, and the names its f-strings build), and the names real
outputs give (the repo's own data, and corpus outputs of each kind when the
gitignored test/ corpus is there).
"""
import ast
import contextlib
import io
import shutil
from pathlib import Path

import pytest

from conftest import TEST_DATA
from mace.utils import property_keys as PK
from mace.utils.property_extractor import CrystalPropertyExtractor

SRC = Path(PK.__file__).with_name("property_extractor.py")
DATA = Path(__file__).parent / "data"

# dicts whose keys become property names (the JSON summaries' own fields are
# not names: summary, detail, entry, rec, ...)
_TARGETS = {"props", "properties", "advanced_props", "classification", "components", "extra_scalars", "total"}

# names built by f-strings, expanded by hand (the source line in a comment)
_AXES = ("xx", "yy", "zz")
EXPANDED = {
    *(f"{k}_band_gap" for k in ("direct", "indirect")),                        # f'{fundamental[0]}_band_gap'
    *(f"{pre}primitive_cell_{m}" for pre in ("", "initial_") for m in ("area", "length")),  # measure
    *(f"dos_at_fermi_{s}" for s in ("alpha", "beta")),                          # f'dos_at_fermi_{label}'
    *(f"processed_{k}" for k in ("atomic_charges", "bonding_analysis", "coordination_environments",
                                 "magnetic_info", "chemical_summary")),        # f'processed_{key}'
    *(f"transport_{q}_max_{a}" for q in ("power_factor", "zt_electronic", "zt_total") for a in _AXES),
    *(f"transport_{q}_best_axis" for q in ("power_factor", "zt_electronic", "zt_total")),
    *(f"final_{n}" for n in ("neighbor_analysis", "min_bond_distance_ang", "max_bond_distance_ang",
                             "max_coordination_number", "total_coordination_shells")),  # f'final_{name}'
    "electron_count", "fragment_charges",                                     # ion_fragments.fragment_charges
    *(f"{c}_energy_ev" for c in ("kinetic", "electron_electron", "electron_nuclear", "nuclear_nuclear",
                                 "exchange", "correlation")),                 # prop_name.replace('_au', '_ev')
}


def _source_names():
    """String keys the extractor puts into its property dicts."""
    tree = ast.parse(SRC.read_text())
    names = set()

    def target_name(node):
        return node.id if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        # props['name'] = ...
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if (isinstance(t, ast.Subscript) and target_name(t.value) in _TARGETS
                        and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                    names.add(t.slice.value)
        # props: Dict[str, Any] = {'name': summary}
        if isinstance(node, ast.AnnAssign) and target_name(node.target) in _TARGETS \
                and isinstance(node.value, ast.Dict):
            names.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))
        # props.update({'name': ...}) / props.setdefault('name', ...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and target_name(node.func.value) in _TARGETS:
            if node.func.attr == "update" and node.args and isinstance(node.args[0], ast.Dict):
                names.update(k.value for k in node.args[0].keys if isinstance(k, ast.Constant))
            if node.func.attr == "setdefault" and node.args and isinstance(node.args[0], ast.Constant):
                names.add(node.args[0].value)
    # dicts of names: returned ones ({'raman_intensities': ...}), and the
    # regex tables (component_patterns, info_patterns, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, (ast.Dict, ast.IfExp)):
            d = node.value.body if isinstance(node.value, ast.IfExp) else node.value
            if isinstance(d, ast.Dict):
                names.update(k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(
                isinstance(t, ast.Name) and t.id.endswith("_patterns") for t in node.targets):
            names.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    return {n for n in names if not n.startswith("_")}


SOURCE = _source_names()
# fields of JSON values in those dicts, not names (a Mulliken section's
# {'total_electrons', 'atoms'}, a peak's {'temperature_k', 'mu_ev'})
_NOT_NAMES = {"atoms", "total_electrons", "temperature_k", "mu_ev"}


def _initial_copies(names):
    """_extract_structural_properties files each input-geometry name also as initial_<name>."""
    return {f"initial_{n}" for n in names if not n.startswith("initial_")}


def test_every_source_name_is_registered():
    names = SOURCE - _NOT_NAMES
    missing = sorted(n for n in names | EXPANDED if n not in PK.KEYS)
    assert not missing, f"names the extractor writes that property_keys.KEYS lacks: {missing}"


def test_no_orphan_in_the_registry():
    reachable = (SOURCE - _NOT_NAMES) | EXPANDED | _initial_copies(SOURCE)
    orphans = sorted(n for n in PK.KEYS if n not in reachable)
    assert not orphans, f"registry names the extractor never writes: {orphans}"


def test_entries_are_well_formed():
    for name, k in PK.KEYS.items():
        assert k.category in PK.CATEGORIES, name
        assert k.type in ("float", "int", "bool", "text", "json"), name
        assert k.unit and k.description, name
        if k.type == "bool":
            assert k.unit == "boolean", name
        if k.type == "text":
            assert k.unit in ("category", "code"), name


def test_mulliken_rows_stay_population_analysis():
    # population_analysis_processor reads its input by this category
    for name in ("mulliken_alpha_plus_beta", "mulliken_alpha_minus_beta", "mulliken_population",
                 "overlap_population_alpha_plus_beta", "overlap_population", "mulliken_valence_charges"):
        assert PK.KEYS[name].category == "population_analysis"


def test_aliases_point_at_current_names():
    for old, new in PK.ALIASES.items():
        assert new in PK.KEYS, (old, new)
        assert old not in PK.KEYS, old
    assert not (PK.REMOVED & set(PK.KEYS))
    assert not (PK.REMOVED & set(PK.ALIASES))
    assert PK.BLOB_KEYS <= set(PK.KEYS)


def test_query_names_and_normalize_rows():
    assert PK.query_names("band_gap") == ["band_gap", "classification_band_gap"]
    assert PK.query_names("transport_zt_max", warn=False)[0] == "transport_zt_electronic_max_avg"
    rows = [{"material_id": "m", "calc_id": "c", "property_name": "transport_zt_max", "v": 1},
            {"material_id": "m", "calc_id": "c", "property_name": "transport_zt_electronic_max", "v": 2},
            {"material_id": "m", "calc_id": "d", "property_name": "fermi_energy_dos", "v": 3},
            {"material_id": "m", "calc_id": "e", "property_name": "fermi_energy_au", "v": 4},
            {"material_id": "m", "calc_id": "e", "property_name": "fermi_energy", "v": 5}]
    out = PK.normalize_rows(rows)
    assert [(r["calc_id"], r["property_name"], r["v"]) for r in out] == [
        ("c", "transport_zt_electronic_max_avg", 2),   # the first-listed old name wins
        ("d", "fermi_energy_au", 3),
        ("e", "fermi_energy_au", 4)]                    # a current name beats an old one


def test_canonical_notes_an_old_name_once():
    said = []
    PK._warned.discard("total_cpu_time")
    assert PK.canonical("total_cpu_time", out=said.append) == "cpu_time_total_s"
    assert PK.canonical("total_cpu_time", out=said.append) == "cpu_time_total_s"
    assert len(said) == 1 and "1.1.7" in said[0]


def test_classify_gap_is_one_scheme():
    assert [PK.classify_gap(g) for g in (0.0, 0.005, 0.01, 2.5, 4.0, 4.01, 5.5)] == [
        "metal", "metal", "semiconductor", "semiconductor", "semiconductor", "insulator", "insulator"]


def _extract(path):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    x.extract_fragments = True
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(path))


def _repo_outputs(tmp_path):
    outs = []
    for folder in ("orbitals/si_f", "orbitals/o2", "orbitals/slab"):
        src = DATA / folder
        if not src.is_dir():
            continue
        dst = tmp_path / folder.replace("/", "_")
        shutil.copytree(src, dst)
        outs += sorted(dst.glob("*.out"))
    outs += sorted((DATA / "charged_runs").glob("*.out"))
    return outs


def _corpus_outputs(per_dir=4):
    if not TEST_DATA.is_dir():
        return []
    outs = []
    for d in ("SP", "OPT", "FREQ", "BAND", "DOSS", "TRANSPORT", "ECH3POT3"):
        outs += sorted((TEST_DATA / d).glob("*.out"))[:per_dir]
    return outs


def test_real_outputs_write_registered_names_only(tmp_path):
    outs = _repo_outputs(tmp_path) + _corpus_outputs()
    assert outs, "no output to extract"
    seen = set()
    for out in outs:
        props = _extract(out)
        seen |= {n for n in props if not n.startswith("_")}
    # derived properties (mace/properties) register their keys at load
    missing = sorted(n for n in seen if not PK.known(n))
    assert not missing, f"names real outputs give that property_keys.KEYS lacks: {missing}"
    # nothing an earlier version wrote and 1.1.6 renamed or removed
    stale = sorted(seen & (set(PK.ALIASES) | PK.REMOVED))
    assert not stale, stale


def test_the_save_path_files_rows_by_the_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = next(iter(_corpus_outputs(1)), None)
    if out is None:
        pytest.skip("test/ data corpus not present (gitignored)")
    x = CrystalPropertyExtractor(db_path=str(tmp_path / "m.db"))
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(out, material_id="m", calc_id=None)
        x.save_properties_to_database(props)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "m.db")
    rows = conn.execute("SELECT property_name, property_category, property_unit FROM properties "
                        "UNION ALL SELECT property_name, property_category, property_unit FROM property_blobs").fetchall()
    assert rows
    for name, cat, unit in rows:
        assert (cat, unit) == (PK.key(name).category, PK.key(name).unit), name
    blob_names = {r[0] for r in conn.execute("SELECT property_name FROM property_blobs")}
    assert all(PK.is_blob(n) for n in blob_names)
    assert not any(PK.is_blob(r[0]) for r in conn.execute("SELECT property_name FROM properties"))
