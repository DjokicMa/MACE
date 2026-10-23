"""Reference energies (mace/properties/defs/reference_energies.py): formation,
cohesive, surface, adsorption and reaction energies, as ΔE and, when every
run has FREQ thermodynamics, ΔH and ΔG; the guard that refuses runs of
another functional, basis set or CRYSTAL version (and warns on spin); the
database lookups; saved choices.

No real set of elemental, isolated-atom or slab-plus-bulk runs with one
functional exists in the test/ corpus, so these cases are a small synthetic
database (made-up energies, checked against the formulas by hand); the real
sets (LiFSI + EC -> LiFSI-EC with FREQ thermodynamics, FSI on four-layer
graphene, a carbon allotrope against diamond, a slab against a bulk of
another functional) are in tests/test_props_reference_real.py."""
import json

import pytest

from mace.properties import reference as RF
from mace.properties import registry as R
from mace.utils import db_build

EV = 27.211386245988
J = 4.3597447222071e-18
KJ = 96.48533212331002
SET = {"dft_functional": "PBE0", "basis_set": "POB-TZVP-REV2", "crystal_version": "CRYSTAL23 public : 1.0.1",
       "spin_polarized": False}
AREA = 20.0


def atoms(**n):
    return [{"element": el} for el, k in n.items() for _ in range(k)]


class Fake:
    """A materials.db of made-up finished runs: each run's stored rows as an extraction writes them."""

    def __init__(self, tmp):
        from mace.database.materials import MaterialDatabase
        self.path = tmp / "materials.db"
        self.db = MaterialDatabase(str(self.path), ase_db_path=str(tmp / "structures.db"))
        self.mats = set()

    def add(self, mid, formula, dim, energy, comp, calc_type="OPT", thermo=None, curve=None, **extra):
        kind = {3: "CRYSTAL", 2: "SLAB", 1: "POLYMER", 0: "MOLECULE"}[dim]
        if mid not in self.mats:
            self.db.create_material(mid, formula, dimensionality=kind)
            self.mats.add(mid)
        cid = self.db.create_calculation(mid, calc_type)
        self.db.update_calculation_status(cid, "completed")
        vals = {**SET, "calculation_type": calc_type, "total_energy_corrected_au": energy,
                "atoms_in_unit_cell": sum(comp.values()), "atomic_positions": atoms(**comp)}
        if dim == 3:
            vals["primitive_cell_volume"] = 100.0
        elif dim == 2:
            vals["primitive_cell_area"] = AREA
        if thermo:                                       # (G, H) at 298.15 K, 0.101325 MPa
            vals.update(gibbs_free_energy_au=thermo[0], enthalpy_au=thermo[1], thermodynamic_temperature_k=298.15,
                        thermodynamic_pressure_mpa=0.101325)
        vals.update(extra)
        for k, v in vals.items():
            if v is not None:
                self.db.store_material_property(mid, k, v, "energy", calc_id=cid)
        if curve:
            pts = [{"temperature_k": t, "pressure_mpa": 0.101325, "gibbs_free_energy_au": g, "enthalpy_au": h}
                   for t, g, h in curve]
            self.db.store_material_property(mid, "thermodynamics_vs_temperature", {"points": pts}, "thermodynamics",
                                            calc_id=cid, blob=True)
        return cid

    def conn(self):
        return db_build.open_ro(self.path)


def run(tmp, props, comp, dim, only, refs=None, conn=None, calc_id=None, material_id=None, ct="OPT"):
    """run_derived on a run that is not a file (its values given)."""
    out = tmp / "run.out"
    out.write_text("")
    p = {**SET, "calculation_type": ct, "atoms_in_unit_cell": sum(comp.values()), "atomic_positions": atoms(**comp),
         **props}
    if dim == 2:
        p.setdefault("primitive_cell_area", AREA)
    vals, rep = R.run_derived(out, "", p, only=[only], references=refs or {}, conn=(lambda: conn) if conn else None,
                              dimensionality=dim, calc_id=calc_id, material_id=material_id)
    return vals, rep[only]


@pytest.fixture
def fake(tmp_path):
    f = Fake(tmp_path)
    f.mg = f.add("Mg", "Mg", 3, -400.1, {"Mg": 2})
    f.mg_freq = f.add("Mg", "Mg", 3, -400.1, {"Mg": 2}, calc_type="FREQ", thermo=(-400.12, -400.09))
    f.mg_hi = f.add("Mg-hcp-strained", "Mg", 3, -200.0, {"Mg": 1})          # higher per atom: not the reference
    f.mg_b3 = f.add("Mg-b3lyp", "Mg", 3, -401.0, {"Mg": 2}, dft_functional="B3LYP")   # lower, other functional
    f.o2 = f.add("O2", "O", 0, -150.3, {"O": 2})
    f.mg_at = f.add("Mg-atom", "Mg", 0, -199.8, {"Mg": 1})
    f.o_at = f.add("O-atom", "O", 0, -74.9, {"O": 1}, spin_polarized=True)
    f.o_at_hi = f.add("O-atom-singlet", "O", 0, -74.85, {"O": 1}, spin_polarized=True)
    f.bulk = f.add("MgO", "MgO", 3, -1102.0, {"Mg": 4, "O": 4})
    f.bulk_freq = f.add("MgO", "MgO", 3, -1102.0, {"Mg": 4, "O": 4}, calc_type="FREQ", thermo=(-1102.03, -1101.99))
    f.slab = f.add("MgO-slab", "MgO", 2, -2203.9, {"Mg": 8, "O": 8})
    f.slab_big = f.add("MgO-slab-2x1", "MgO", 2, -4407.8, {"Mg": 16, "O": 16}, primitive_cell_area=2 * AREA)
    f.slab_wide = f.add("MgO-slab-wide", "MgO", 2, -2203.95, {"Mg": 8, "O": 8}, primitive_cell_area=25.0)
    f.co2 = f.add("CO2", "CO2", 0, -188.6, {"C": 1, "O": 2})
    f.h2 = f.add("H2", "H", 0, -1.17, {"H": 2}, calc_type="FREQ", thermo=(-1.16, -1.15),
                 curve=[(100.0, -1.165, -1.152), (298.15, -1.16, -1.15)])
    f.o2f = f.add("O2", "O", 0, -150.3, {"O": 2}, calc_type="FREQ", thermo=(-150.32, -150.29),
                  curve=[(100.0, -150.31, -150.292), (298.15, -150.32, -150.29)])
    return f


# ------------------------------------------------------------------ the five energies
def test_formation_energy_against_the_lowest_elemental_phases(tmp_path, fake):
    conn = fake.conn()
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "formation_energy",
                  conn=conn)
    assert e["status"] == "stored", e
    # Mg: -400.1/2 (not -200.0, and not the B3LYP run's -401.0/2); O: -150.3/2
    want = (-1102.0 - 4 * (-400.1 / 2) - 4 * (-150.3 / 2)) / 8 * EV
    assert vals["formation_energy_ev_per_atom"] == pytest.approx(want, rel=1e-12)
    used = {r["calc_id"] for r in e["references"]["elements"]}
    assert used == {fake.mg, fake.o2} and all(r["via"] == "database" for r in e["references"]["elements"])
    # the run is no FREQ run and has no material: ΔE only, and why
    d = vals["formation_energy"]
    assert "formation_gibbs_energy_ev_per_atom" not in vals and "run.out has no FREQ thermodynamics" in d["thermo_note"]
    assert d["reference_energy_per_atom_au"] == {"Mg": pytest.approx(-200.05), "O": pytest.approx(-75.15)}
    # with the FREQ runs of each material (the run is MgO's): ΔH and ΔG from every run's H and G
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "formation_energy",
                  conn=conn, material_id="MgO", calc_id="the-run")
    assert vals["formation_gibbs_energy_ev_per_atom"] == pytest.approx(
        (-1102.03 - 4 * (-400.12 / 2) - 4 * (-150.32 / 2)) / 8 * EV, rel=1e-12)
    assert vals["formation_enthalpy_ev_per_atom"] == pytest.approx(
        (-1101.99 - 4 * (-400.09 / 2) - 4 * (-150.29 / 2)) / 8 * EV, rel=1e-12)
    th = vals["formation_energy"]["thermo"]
    assert th["temperature_k"] == 298.15 and len(th["from"]) == 3
    conn.close()


def test_an_element_and_a_missing_element(tmp_path, fake):
    conn = fake.conn()
    # a strained Mg phase against the most stable Mg
    vals, e = run(tmp_path, {"total_energy_corrected_au": -200.0}, {"Mg": 1}, 3, "formation_energy", conn=conn)
    assert vals["formation_energy_ev_per_atom"] == pytest.approx((-200.0 + 200.05) * EV)
    # the most stable phase itself: zero by definition, nothing stored
    vals, e = run(tmp_path, {"total_energy_corrected_au": -400.2}, {"Mg": 2}, 3, "formation_energy", conn=conn)
    assert e["status"] == "skipped" and "zero by definition" in e["reason"]
    vals, e = run(tmp_path, {"total_energy_corrected_au": -500.0}, {"Mg": 1, "Cl": 2}, 3, "formation_energy", conn=conn)
    assert e["status"] == "skipped" and "no elemental reference run for Cl" in e["reason"]
    # no database: the reference must be given
    vals, e = run(tmp_path, {"total_energy_corrected_au": -500.0}, {"Mg": 1, "Cl": 2}, 3, "formation_energy")
    assert e["status"] == "skipped" and "reference run elements" in e["reason"]
    conn.close()


def test_cohesive_energy_against_isolated_atoms(tmp_path, fake):
    conn = fake.conn()
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "cohesive_energy", conn=conn)
    assert e["status"] == "stored", e
    # the lowest O atom (its ground spin state), positive when bound
    assert vals["cohesive_energy_ev_per_atom"] == pytest.approx((4 * -199.8 + 4 * -74.9 + 1102.0) / 8 * EV, rel=1e-12)
    assert {r["calc_id"] for r in e["references"]["atoms"]} == {fake.mg_at, fake.o_at}
    # the O atoms are spin-polarized and the solid is not: stored, with the warning
    assert any("spin-polarized" in w for w in vals["cohesive_energy"]["warnings"])
    # a given run that is no atom is refused
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "cohesive_energy", conn=conn,
                  refs={"atoms": [fake.mg_at, fake.o2]})
    assert e["status"] == "skipped" and "is not an isolated atom" in e["reason"]
    conn.close()


def test_surface_energy_of_a_slab(tmp_path, fake):
    conn = fake.conn()
    vals, e = run(tmp_path, {"total_energy_corrected_au": -2203.9}, {"Mg": 8, "O": 8}, 2, "surface_energy", conn=conn)
    assert e["status"] == "stored", e
    d = -2203.9 - 2 * -1102.0
    assert vals["surface_energy_ev_ang2"] == pytest.approx(d * EV / (2 * AREA), rel=1e-12)
    assert vals["surface_energy_j_m2"] == pytest.approx(d * J / (2 * AREA * 1e-20), rel=1e-12)
    assert vals["surface_energy_j_m2"] == pytest.approx(1.08994, rel=1e-5)          # by hand: 0.1 Ha / 40 Å²
    assert vals["surface_energy"]["n_bulk_cells"] == 2 and vals["surface_energy"]["area_ang2"] == AREA
    assert e["references"]["bulk"][0]["calc_id"] == fake.bulk
    # not commensurate (a non-stoichiometric slab): refused
    vals, e = run(tmp_path, {"total_energy_corrected_au": -2100.0}, {"Mg": 8, "O": 7}, 2, "surface_energy", conn=conn,
                  refs={"bulk": fake.bulk})
    assert e["status"] == "skipped" and "not commensurate" in e["reason"]
    # a bulk run is no slab
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "surface_energy", conn=conn)
    assert e["status"] == "skipped" and e["reason"] == "not a slab (2D) run"
    conn.close()


def test_adsorption_energy_found_in_the_database(tmp_path, fake):
    conn = fake.conn()
    e_cx = -2203.9 + 2 * -188.6 - 2 * 0.01
    vals, e = run(tmp_path, {"total_energy_corrected_au": e_cx}, {"Mg": 8, "O": 12, "C": 2}, 2, "adsorption_energy",
                  conn=conn)
    assert e["status"] == "stored", e
    # the slab in the same cell (not the 2x1 one), and k = 2 CO2
    assert e["references"]["surface"][0]["calc_id"] == fake.slab
    assert e["references"]["adsorbate"][0]["calc_id"] == fake.co2
    assert vals["adsorption_energy"]["k_adsorbates"] == 2
    assert vals["adsorption_energy_ev"] == pytest.approx(-0.01 * EV, rel=1e-9)
    assert vals["adsorption_energy_kj_mol"] == pytest.approx(-0.01 * EV * KJ, rel=1e-9)
    # a surface in another cell (the lookup passed it over) is refused when given
    vals, e = run(tmp_path, {"total_energy_corrected_au": e_cx}, {"Mg": 8, "O": 12, "C": 2}, 2, "adsorption_energy",
                  conn=conn, refs={"surface": fake.slab_wide, "adsorbate": fake.co2})
    assert e["status"] == "skipped" and e["reason"].startswith("the cells differ")
    vals, e = run(tmp_path, {"total_energy_corrected_au": e_cx}, {"Mg": 8, "O": 12, "C": 2}, 2, "adsorption_energy",
                  conn=conn, refs={"surface": fake.slab_big, "adsorbate": fake.co2})
    assert e["status"] == "skipped" and "plus a whole number of" in e["reason"]
    # the adsorbate looked up for the surface the user gave
    vals, e = run(tmp_path, {"total_energy_corrected_au": e_cx}, {"Mg": 8, "O": 12, "C": 2}, 2, "adsorption_energy",
                  conn=conn, refs={"surface": fake.slab})
    assert e["references"]["surface"][0]["via"] == "given" and e["references"]["adsorbate"][0]["calc_id"] == fake.co2
    conn.close()


def test_reaction_energy_balanced_and_with_thermodynamics(tmp_path, fake):
    conn = fake.conn()
    # H2 + 1/2 O2 -> H2O (the run, a FREQ run printing two temperatures)
    curve = {"points": [{"temperature_k": 100.0, "pressure_mpa": 0.101325, "gibbs_free_energy_au": -76.39,
                         "enthalpy_au": -76.37},
                        {"temperature_k": 298.15, "pressure_mpa": 0.101325, "gibbs_free_energy_au": -76.40,
                         "enthalpy_au": -76.36}]}
    props = {"total_energy_corrected_au": -76.4, "gibbs_free_energy_au": -76.40, "enthalpy_au": -76.36,
             "thermodynamic_temperature_k": 298.15, "thermodynamic_pressure_mpa": 0.101325,
             "thermodynamics_vs_temperature": curve}
    vals, e = run(tmp_path, props, {"H": 2, "O": 1}, 0, "reaction_energy", conn=conn, ct="FREQ",
                  refs={"reactants": [fake.h2, fake.o2f]})
    assert e["status"] == "stored", e
    d = vals["reaction_energy"]
    assert d["equation"] == "1 H2 + 1/2 O2 → 1 H2O"
    assert vals["reaction_energy_ev"] == pytest.approx((-76.4 + 1.17 + 0.5 * 150.3) * EV, rel=1e-12)
    assert vals["reaction_gibbs_energy_ev"] == pytest.approx((-76.40 + 1.16 + 0.5 * 150.32) * EV, rel=1e-12)
    assert vals["reaction_enthalpy_kj_mol"] == pytest.approx((-76.36 + 1.15 + 0.5 * 150.29) * EV * KJ, rel=1e-12)
    low = d["vs_temperature"][0]
    assert low["temperature_k"] == 100.0 and low["gibbs_energy"]["ev"] == pytest.approx(
        (-76.39 + 1.165 + 0.5 * 150.31) * EV, rel=1e-12)
    # given coefficients must balance
    vals, e = run(tmp_path, props, {"H": 2, "O": 1}, 0, "reaction_energy", conn=conn, ct="FREQ",
                  refs={"reactants": [f"2*{fake.h2}", fake.o2f]})
    assert e["status"] == "skipped" and "not balanced" in e["reason"] and "H -2" in e["reason"]
    # a reaction is never looked up
    vals, e = run(tmp_path, props, {"H": 2, "O": 1}, 0, "reaction_energy", conn=conn, ct="FREQ")
    assert e["status"] == "skipped" and "reactants" in e["reason"]
    conn.close()


# ------------------------------------------------------------------ the guard
def test_runs_of_another_method_are_refused_or_warned(tmp_path, fake):
    conn = fake.conn()
    comp = {"Mg": 4, "O": 4}
    base = {"total_energy_corrected_au": -1102.0}
    cases = [
        ({"dft_functional": "B3LYP"}, "refuse", "the functional differs"),
        ({"basis_set": "def2-TZVP"}, "refuse", "the basis set differs"),
        ({"crystal_version": "CRYSTAL17 public : 1.0.2"}, "refuse", "the CRYSTAL version differs"),
        ({"crystal_version": "CRYSTAL23 public : 1.0.2"}, "warn", "another CRYSTAL release"),
        ({"basis_set": RF.CUSTOM_BASIS}, "warn", "the basis set is not compared"),
        ({"dft_functional": "HARTREE+FOCK EXCH*0.25+PBE"}, "warn", "the functional is not compared"),
        ({"spin_polarized": True}, "warn", "is spin-polarized"),
    ]
    for extra, kind, text in cases:
        vals, e = run(tmp_path, {**base, **extra}, comp, 3, "formation_energy", conn=conn,
                      refs={"elements": [fake.mg, fake.o2]})
        if kind == "refuse":
            assert e["status"] == "skipped" and e["reason"].startswith("not combined: ") and text in e["reason"], (extra, e)
        else:
            assert e["status"] == "stored" and any(text in w for w in vals["formation_energy"]["warnings"]), (extra, e)
    # D3 in one run only (both print their terms)
    a = RF.Part("a", {**SET, "total_energy_au": -1.0, "d3_dispersion_energy_au": -0.01})
    b = RF.Part("b", {**SET, "total_energy_au": -1.0})
    assert RF.check(a, b)[0] == ["D3 dispersion is in a and not in b"]
    conn.close()


# ------------------------------------------------------------------ saved choices
def test_a_saved_choice_is_used_until_auto(tmp_path, fake):
    other_o2 = fake.add("O2-triplet", "O", 0, -150.4, {"O": 2})          # the lowest O2: what the lookup takes
    run_id = fake.add("MgO-run", "MgO", 3, -1102.0, {"Mg": 4, "O": 4})
    fake.db.store_material_property("MgO-run", R.CHOICES_KEY,
                                    {"formation_energy": {"elements": [{"calc_id": fake.mg}, {"calc_id": fake.o2}]}},
                                    "computational", calc_id=run_id, extractor_script=R.REFERENCE_SCRIPT)
    conn = fake.conn()
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "formation_energy",
                  conn=conn, calc_id=run_id, material_id="MgO-run")
    assert [r["via"] for r in e["references"]["elements"]] == ["saved", "saved"]
    assert {r["calc_id"] for r in e["references"]["elements"]} == {fake.mg, fake.o2}
    vals, e = run(tmp_path, {"total_energy_corrected_au": -1102.0}, {"Mg": 4, "O": 4}, 3, "formation_energy",
                  conn=conn, calc_id=run_id, material_id="MgO-run", refs={"elements": ["auto"]})
    assert {r["calc_id"] for r in e["references"]["elements"]} == {fake.mg, other_o2}
    conn.close()
    merged = R.merge_choices({"formation_energy": {"elements": [{"calc_id": "x"}]}},
                             {"reaction_energy": {"reactants": [{"calc_id": "y", "weight": 2.0, "via": "given"}]}},
                             {"formation_energy": ["elements"]})
    assert merged == {"reaction_energy": {"reactants": [{"calc_id": "y", "weight": 2.0}]}}


def test_given_values_and_balance():
    assert R.given_ref("auto") == R.AUTO
    r = R.given_ref("2*EC_OPT_1", exists=lambda s: False)
    assert (r.calc_id, r.weight, r.via) == ("EC_OPT_1", 2.0, "given")
    r = R.given_ref("0.5 * o2.out", exists=lambda s: s == "o2.out")
    assert (str(r.output_file), r.weight) == ("o2.out", 0.5)
    from collections import Counter
    # 2 H2 + O2 -> 2 H2O with the H2O coefficient fixed at 1
    co = RF.balance([(1, Counter(H=2, O=1)), (-1, Counter(H=2)), (-1, Counter(O=2))])
    assert [str(c) for c in co] == ["1", "1", "1/2"]
    assert RF.balance([(1, Counter(C=1)), (-1, Counter(H=2))]) is None                 # cannot balance
    assert RF.balance([(1, Counter(C=2)), (-1, Counter(C=1)), (-1, Counter(C=2))]) is None   # not in one way
    assert RF.parse_formula("LiC3NH4S2F2O7") == Counter(Li=1, C=3, N=1, H=4, S=2, F=2, O=7)
    assert RF.parse_formula("Unknown") is None and RF.formula(Counter(O=2, C=1, H=4)) == "CH4O2"
    assert RF.symbol("AG") == "Ag" and RF.symbol("XX") is None
