"""The Properties tab's display units (mace/properties/units.py): exact
CODATA 2018 factors from mace/constants.py (no second, slightly different
value), round trips in every dimension, a decision for every unit the
registry uses, the keys that hold one quantity in several units (found from
the names and units, and all declared), the explicit extensive/intensive
classification, the rows the server annotates (one row per quantity, the
database's rows all kept) and ``mace analyze --list-properties --units``."""
import itertools
import json
import math
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace import constants as C
from mace.properties import registry, units as U
from mace.utils import property_keys as PK

registry.load()


def _all_keys():
    return {**PK.KEYS, **PK.EXTRA}


# ------------------------------------------------------------------ factors
def test_factors_are_the_shared_constants_exactly():
    assert U.convert(1.0, "energy", "Hartree", "eV") == C.HARTREE_TO_EV
    assert U.convert(1.0, "length", "Bohr", "Å") == C.BOHR_TO_ANGSTROM
    assert U.convert(2.0, "energy", "Ry", "Hartree") == 1.0
    # the reference energies' constants (mace/properties/reference.py) are the same numbers
    from mace.properties import reference as R
    assert U.EV_KJ_MOL == R.EV_KJ_MOL
    assert U.convert(1.0, "energy", "eV", "kJ/mol") == pytest.approx(R.EV_KJ_MOL, rel=1e-15)
    assert U.HARTREE_J == pytest.approx(R.HARTREE_J, rel=1e-15)


@pytest.mark.parametrize("dim, a, b, factor", [
    # CODATA 2018 (exact SI 2019 constants, Hartree from mace/constants.py)
    ("energy", "Hartree", "kJ/mol", 2625.4996394799),
    ("energy", "Hartree", "kcal/mol", 627.5094740631),
    ("energy", "Hartree", "cm⁻¹", 219474.6313632),
    ("energy", "eV", "cm⁻¹", 8065.543937349212),
    ("energy", "kcal/mol", "kJ/mol", 4.184),
    ("energy", "eV", "meV", 1000.0),
    ("frequency", "THz", "cm⁻¹", 33.35640951981521),
    ("frequency", "meV", "cm⁻¹", 8.065543937349212),
    ("characteristic_temperature", "meV", "K", 11.604518121550082),
    ("dipole", "e·Å", "D", 4.803204712570263),
    ("surface_energy", "eV/Å²", "J/m²", 16.02176634),
    ("pressure", "GPa", "kbar", 10.0),
    ("pressure", "Mbar", "GPa", 100.0),
    ("pressure", "atm", "bar", 1.01325),
    ("power_factor", "μW/(cm·K²)", "W/(m·K²)", 1e-4),
    ("entropy", "J/(mol·K)", "meV/K", 1.0 / 96.48533212331002),
    ("dos", "states/eV", "states/Hartree", 27.211386245988),
    ("inverse_pressure", "GPa⁻¹", "TPa⁻¹", 1000.0),
    ("velocity", "Å/fs", "km/s", 100.0),
    ("duration", "h", "seconds", 3600.0),
])
def test_codata_factors(dim, a, b, factor):
    assert U.convert(1.0, dim, a, b) == pytest.approx(factor, rel=1e-12)


def test_energy_decimals_follow_crystals_hartree_precision():
    """Energies are shown to 1e-10 Hartree in every unit; other values keep their digits."""
    assert {u: U.energy_decimals(u) for u in U.units_of("energy")} == {
        "eV": 9, "meV": 6, "Hartree": 10, "mHartree": 7, "Ry": 10, "kJ/mol": 7, "kcal/mol": 8, "cm⁻¹": 5}
    for n in ("total_energy_au", "d3_dispersion_energy_au", "gibbs_free_energy_au", "fermi_energy_au"):
        assert U.display_decimals(n) == 10, n
    assert U.display_decimals("energy_per_atom_ev") == 9
    assert U.display_decimals("formation_energy_ev_per_atom") == 9
    assert U.display_decimals("adsorption_energy_kj_mol") == 7
    for n in ("band_gap", "vbm_spin_splitting_mev", "primitive_a", "bulk_modulus_hill_gpa", "entropy_mhartree_cell_k"):
        assert U.display_decimals(n) is None, n
    from mace.properties import display
    assert display.format_value(-76.19421501234567, 6, U.display_decimals("total_energy_au")) == "-76.1942150123"
    assert display.format_value(6.0112, 6, U.display_decimals("band_gap")) == "6.0112"
    cat = U.catalog(["energy", "length"])
    assert {u["id"]: u["decimals"] for u in cat["dimensions"]["energy"]["units"]}["kJ/mol"] == 7
    assert all(u["decimals"] is None for u in cat["dimensions"]["length"]["units"])


def test_energy_per_atom_says_it_is_without_d3():
    from mace.properties import display
    d = display.describe("energy_per_atom_ev")
    assert "without D3" in d["label"] and "per atom" in d["description"] and "without the D3" in d["description"]


def test_hartree_per_bohr3_follows_the_shared_bohr():
    # MACE's Bohr (mace/constants.py) is the one every converter uses
    expect = C.HARTREE_TO_EV * C.ELEMENTARY_CHARGE_C / (C.BOHR_TO_ANGSTROM * 1e-10) ** 3 / 1e9
    assert U.convert(1.0, "pressure", "Hartree/Bohr³", "GPa") == pytest.approx(expect, rel=1e-15)
    assert U.convert(1.0, "pressure", "Hartree/Bohr³", "GPa") == pytest.approx(29421.015697, rel=1e-10)   # CODATA 2018


@pytest.mark.parametrize("dim", sorted(U.dimensions()))
def test_round_trips(dim):
    for a, b in itertools.permutations(U.units_of(dim), 2):
        for v in (1.0, -1234.56789, 3.2e-7):
            assert U.convert(U.convert(v, dim, a, b), dim, b, a) == pytest.approx(v, rel=1e-14), (a, b)


def test_presets_name_units_that_exist():
    ids = [p["id"] for p in U.PRESETS]
    assert ids[0] == "stored" and len(set(ids)) == len(ids)
    for p in U.PRESETS:
        for dim, unit in p["units"].items():
            assert unit in U.units_of("energy" if dim == "energy_level" else dim), (p["id"], dim, unit)
    json.dumps(U.catalog())


def test_labels_use_real_super_and_subscripts():
    text = json.dumps(U.catalog(), ensure_ascii=False)
    assert "^" not in text and "**" not in text
    assert U.label("volume", "Å³", "atom") == "Å³/atom"
    assert U.label("energy", "kJ/mol", "f.u.") == "kJ/mol per f.u."
    assert U.label("entropy", "mHartree/K", "cell") == "mHartree/(cell·K)"
    assert U.label("energy", "eV") == "eV"


# ------------------------------------------------------------------ every key has a decision
def test_every_unit_the_registry_uses_has_a_dimension_or_is_unitless():
    missing = sorted({(k.unit, n) for n, k in _all_keys().items() if not U.known(k.unit)})
    assert not missing, missing


def test_stored_units_map_to_their_own_spelling():
    """A key's unit as stored is a unit of its dimension (As stored shows the stored unit)."""
    for n, k in _all_keys().items():
        i = U.info(n, k.unit)
        if i["dim"] in U.dimensions():
            assert i["unit_id"] in U.units_of(i["dim"]), (n, k.unit)


def test_ambiguous_units_are_decided_by_key():
    # cm⁻¹: a wavenumber for a frequency, a count per cm for a 1D carrier concentration
    assert U.info("lowest_frequency_cm")["dim"] == "frequency"
    assert U.info("lo_to_splitting_max")["dim"] == "frequency"
    i = U.info("intrinsic_carrier_concentration_300k_cm1")
    assert i["dim"] == "carrier_concentration_1d" and not i["convertible"]
    assert U.info("debye_temperature_phonon_k")["dim"] == "characteristic_temperature"
    assert U.info("thermodynamic_temperature_k")["dim"] == "temperature"
    assert not U.info("thermodynamic_temperature_k")["convertible"]
    # categorical, boolean and JSON keys get no picker
    for n in ("band_gap_type", "spin_polarized", "band_structure", "atoms_in_unit_cell", "vibrational_frequencies"):
        assert not U.info(n)["convertible"], n


# ------------------------------------------------------------------ one quantity, several units
def test_every_unit_duplicate_is_declared():
    """Keys that look like one quantity in several units (the same name without
    its unit suffix, dimension and basis) are exactly the declared groups."""
    found = {tuple(g) for g in U.find_unit_duplicates()}
    declared = {tuple(sorted(g)) for g in U.same_quantity_groups().values()}
    assert found == declared
    assert len(declared) == 30


def test_the_shown_key_is_the_most_precise_and_alike():
    groups = U.same_quantity_groups()
    assert groups["d3_dispersion_energy_au"] == ["d3_dispersion_energy_au", "d3_dispersion_energy_ev"]
    assert set(groups["gibbs_free_energy_au"]) == {"gibbs_free_energy_au", "gibbs_free_energy_ev",
                                                   "gibbs_free_energy_kj_mol"}
    assert set(groups["entropy_mhartree_cell_k"]) == {"entropy_mhartree_cell_k", "entropy_mev_cell_k", "entropy_j_mol_k"}
    assert groups["surface_energy_j_m2"] == ["surface_energy_j_m2", "surface_energy_ev_ang2"]
    assert groups["adsorption_energy_ev"] == ["adsorption_energy_ev", "adsorption_energy_kj_mol"]
    for head, members in groups.items():
        h = U.info(head)
        assert U.same_as(head) is None
        for m in members[1:]:
            i = U.info(m)
            assert (i["dim"], i["basis"], i["extensive"]) == (h["dim"], h["basis"], h["extensive"]), m
            assert i["unit_id"] != h["unit_id"], m


def test_a_derived_same_as_must_be_another_output():
    bad = registry.DerivedProperty("t_units_bad", ("SP",), (registry.Output("t_units_a", "eV", same_as="t_units_x"),),
                                   lambda ctx: {}, "energy")
    with pytest.raises(registry.RegistryError):
        registry.register(bad)


# ------------------------------------------------------------------ extensive or intensive
def test_extensive_is_explicit():
    for n in ("total_energy_au", "total_energy_ev", "d3_dispersion_energy_au", "gibbs_free_energy_kj_mol",
              "entropy_j_mol_k", "primitive_cell_volume", "dos_at_fermi_total", "mulliken_spin_total"):
        assert U.is_extensive(n), n
    for n in ("band_gap", "fermi_energy_au", "energy_per_atom_ev", "formation_energy_ev_per_atom",
              "cohesive_energy_ev_per_atom", "adsorption_energy_ev", "reaction_energy_kj_mol", "surface_energy_j_m2",
              "doss_dat_energy_min", "bulk_modulus_hill_gpa", "density", "crystallographic_cell_volume"):
        assert not U.is_extensive(n), n
    assert U.info("energy_per_atom_ev")["basis"] == "atom"
    assert U.info("formation_energy_ev_per_atom")["basis"] == "atom"
    assert PK.EXTENSIVE <= set(PK.KEYS)
    for n in PK.EXTENSIVE:
        assert U.info(n)["convertible"], n


def test_presets_treat_energy_levels_apart():
    """The chemistry preset shows energies in kJ/mol and keeps levels (gaps, band edges, E_F) in eV."""
    for n in ("band_gap", "fermi_energy_au", "work_function_ev", "doss_dat_energy_min", "orbitals_homo_energy_au",
              "transport_zt_total_max_mu_ev"):
        assert U.info(n)["preset_dim"] == "energy_level", n
    for n in ("total_energy_au", "gibbs_free_energy_kj_mol", "formation_energy_ev_per_atom", "adsorption_energy_ev"):
        assert U.info(n)["preset_dim"] == "energy", n
    chem = next(p for p in U.PRESETS if p["id"] == "chemistry")["units"]
    assert (chem["energy"], chem["energy_level"]) == ("kJ/mol", "eV")


def test_counts_of_a_run():
    pos = [{"element": e} for e in ("Si", "Si", "O", "O", "O", "O")]
    assert U.counts({"atoms_in_unit_cell": 6, "atomic_positions": pos}) == {"atoms": 6, "formula_units": 2,
                                                                           "formula": "SiO2"}
    # no atoms in the run: the materials table's cell formula, when its count agrees
    assert U.counts({"atoms_in_unit_cell": 8}, "F4Li4")["formula_units"] == 4
    assert U.counts({"atoms_in_unit_cell": 9}, "F4Li4")["formula_units"] is None
    assert U.counts({}) == {"atoms": None, "formula_units": None, "formula": None}


# ------------------------------------------------------------------ the rows the tab gets
def _r(name, value, calc="c1", unit=None):
    return {"name": name, "value": value, "unit": unit if unit is not None else PK.key(name).unit, "calc_id": calc}


def test_annotate_merges_one_quantity_per_run_and_keeps_every_row():
    rows = [_r("total_energy_au", -10.0), _r("total_energy_ev", -272.11386245988), _r("band_gap", 1.2),
            _r("total_energy_au", -20.0, "c2"), _r("total_energy_ev", -544.2, "c2"),
            _r("gibbs_free_energy_ev", -5.0), _r("gibbs_free_energy_kj_mol", -482.4)]
    props = {"c1": {"atoms_in_unit_cell": 2, "atomic_positions": [{"element": "C"}] * 2}}
    cat = U.annotate(rows, props)
    by = {(r["calc_id"], r["name"]): r for r in rows}
    assert len(rows) == 7
    assert by[("c1", "total_energy_au")]["also"] == ["total_energy_ev"]
    assert by[("c1", "total_energy_ev")]["merged_into"] == "total_energy_au"
    assert by[("c2", "total_energy_ev")]["merged_into"] == "total_energy_au"
    # a database without the Hartree key: the eV one is shown
    assert by[("c1", "gibbs_free_energy_ev")]["also"] == ["gibbs_free_energy_kj_mol"]
    assert "merged_into" not in by[("c1", "band_gap")] and "also" not in by[("c1", "band_gap")]
    t = by[("c1", "total_energy_au")]
    assert (t["dim"], t["unit_id"], t["extensive"], t["atoms"], t["formula_units"], t["formula"]) == \
        ("energy", "Hartree", True, 2, 2, "C")
    assert by[("c1", "band_gap")]["convertible"] and not by[("c1", "band_gap")]["extensive"]
    assert set(cat["dimensions"]) == {"energy"}


@pytest.mark.skipif(not (TEST_DATA / "SP").is_dir(), reason="needs the real corpus (test/)")
def test_a_real_run_converts_to_its_stored_ev_values():
    from mace.gui import properties_routes as PR
    out = sorted((TEST_DATA / "SP").rglob("*.out"))[0]
    res = PR.compute_job(path=str(out))
    rows = {r["name"]: r for r in res["rows"]}
    assert "units" in res and "energy" in res["units"]["dimensions"]
    # the server's value text: energies to 10 decimals in Hartree, the rest as before
    t = rows["total_energy_au"]
    assert t["display"] == f"{t['value']:.10f}" and t["full_precision"]
    if "band_gap" in rows:
        assert rows["band_gap"]["display"] == f"{rows['band_gap']['value']:.6g}"
    checked = 0
    for head, members in U.same_quantity_groups().items():
        if head in rows and rows[head].get("convertible"):
            for m in members[1:]:
                if m in rows and rows[m]["unit"] == "eV":
                    got = U.convert(rows[head]["value"], "energy", rows[head]["unit_id"], "eV")
                    assert got == pytest.approx(rows[m]["value"], rel=1e-9), m
                    assert rows[m]["merged_into"] == head
                    checked += 1
    assert checked >= 2


# ------------------------------------------------------------------ the CLI
def test_list_properties_units_cli():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "COLORTERM", "TERM", "NO_COLOR")}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--list-properties",
                        "--units", "--json"], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    d = json.loads(p.stdout)
    assert d["same_quantity"]["d3_dispersion_energy_au"] == ["d3_dispersion_energy_au", "d3_dispersion_energy_ev"]
    e = {u["id"]: u["scale"] for u in d["dimensions"]["energy"]["units"]}
    assert e["Hartree"] == C.HARTREE_TO_EV and e["eV"] == 1.0
    q = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--list-properties",
                        "--json"], capture_output=True, text=True, env=env, timeout=300)
    rows = {r["name"]: r for r in json.loads(q.stdout)}
    assert rows["total_energy_ev"]["shown_as"] == "total_energy_au" and rows["total_energy_au"]["extensive"]
    assert rows["band_gap"]["dimension"] == "energy" and not rows["band_gap"]["extensive"]
    assert math.isfinite(e["kcal/mol"])
