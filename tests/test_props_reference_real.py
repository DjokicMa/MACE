"""Reference energies on real CRYSTAL23 outputs, through mace_cli as a
subprocess where they touch a database.

The example suite (always present): 4LG_AA (four-layer graphene, C96)
from 48 graphene cells, a reaction given by calc_id on the command line; the
choice is saved with the run, a re-extraction of the folder keeps it, and
``reactants=auto`` forgets it. The Add Reference… picker's data and its
command for the same pair.

The test/ corpus (gitignored; skipped without it): LiFSI + EC -> LiFSI-EC
(HSEsol-3c, FREQ thermodynamics for all three: ΔE, ΔH, ΔG), FSI on
four-layer graphene (complex, clean surface and FSI layer in one cell, found
by the database's lookup), a carbon allotrope against diamond (B3LYP-D3), and
an HSEsol-3c slab refused against a B3LYP-D3 bulk. Energies checked by hand:
E in Hartree x 27.211386245988 eV."""
import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.properties import registry as R

EV = 27.211386245988
KJ = 96.48533212331002
MOL = "_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=_env(), cwd=str(cwd), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p


def _rows(db, calc_id):
    conn = sqlite3.connect(db)
    try:
        rows = {n: (v, t, s) for n, v, t, s in conn.execute(
            "SELECT property_name, property_value, property_value_text, extractor_script FROM properties "
            "WHERE calc_id = ?", (calc_id,))}
    finally:
        conn.close()
    prov = json.loads(rows["extraction_provenance"][1])["derived"]
    return rows, prov


def _calc(db, material, calc_type):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT calc_id FROM calculations WHERE material_id = ? AND calc_type = ?",
                            (material, calc_type)).fetchone()[0]
    finally:
        conn.close()


def _extract(path, refs=None):
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    x.references = refs or {}
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(path))


class St:
    """The server state the routes use: the served folder and its path rules."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, rel, must_exist=True):
        return (self.root / rel).resolve()

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()


# ------------------------------------------------------------------ the example suite
@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    root = tmp_path_factory.mktemp("refsuite") / "root"
    ES.copy_suite(root / "suite")
    _cli(root, "analyze", "--extract-properties", "suite", "--db-path", "suite/materials.db")
    return root


def test_a_reaction_given_on_the_command_line_is_saved_with_the_run(suite):
    db = suite / "suite" / "materials.db"
    graphene, four = _calc(db, "graphene", "OPT"), _calc(db, "4LG_AA", "OPT")
    out = next((suite / "suite").rglob("4LG_AA_opt.out")).relative_to(suite).as_posix()
    rows, prov = _rows(db, four)
    assert prov["reaction_energy"]["status"] == "skipped"            # never looked up
    _cli(suite, "analyze", "--extract-properties", "--output-file", out, "--db-path", "suite/materials.db",
         "--reference", f"reactants={graphene}")
    rows, prov = _rows(db, four)
    e_four = rows["total_energy_corrected_au"][0]
    e_gra = _rows(db, graphene)[0]["total_energy_corrected_au"][0]
    # 48 C2 -> C96, balanced from the compositions; by hand -0.11073 Ha = -3.0131 eV per 4LG_AA cell
    want = (e_four - 48 * e_gra) * EV
    assert rows["reaction_energy_ev"][0] == pytest.approx(want, rel=1e-12)
    assert rows["reaction_energy_ev"][0] == pytest.approx(-3.0131, abs=5e-4)
    assert rows["reaction_energy_kj_mol"][0] == pytest.approx(want * KJ, rel=1e-12)
    d = json.loads(rows["reaction_energy"][1])
    assert d["equation"] == "48 C2 → 1 C96"
    assert not any("spin-polarized" in w for w in d["warnings"])       # both are (the suite's runs keep SPIN)
    assert "ΔE only" in d["thermo_note"] and "reaction_gibbs_energy_ev" not in rows
    saved = json.loads(rows[R.CHOICES_KEY][1])
    assert saved == {"reaction_energy": {"reactants": [{"calc_id": graphene}]}}
    assert rows[R.CHOICES_KEY][2] == R.REFERENCE_SCRIPT
    # the whole folder again, without --reference: the saved choice is used
    _cli(suite, "analyze", "--extract-properties", "suite", "--db-path", "suite/materials.db")
    rows, prov = _rows(db, four)
    assert rows["reaction_energy_ev"][0] == pytest.approx(want, rel=1e-12)
    assert prov["reaction_energy"]["references"]["reactants"][0]["via"] == "saved"
    # auto forgets it: nothing to look up for a reaction
    _cli(suite, "analyze", "--extract-properties", "--output-file", out, "--db-path", "suite/materials.db",
         "--reference", "reactants=auto")
    rows, prov = _rows(db, four)
    assert R.CHOICES_KEY not in rows and "reaction_energy_ev" not in rows
    assert prov["reaction_energy"]["status"] == "skipped"


def test_the_pickers_data_and_command(suite):
    from mace.gui import properties_routes as PR
    st = St(suite)
    out = next((suite / "suite").rglob("4LG_AA_opt.out"))
    v = PR.references_view(st, out)
    assert v["state"] == "db" and v["formula"] == "C96" and v["dim"] == 2 and v["functional"] == "HSESOL3C"
    props = {p["name"]: p for p in v["properties"]}
    assert set(props) == {"formation_energy", "cohesive_energy", "surface_energy", "adsorption_energy",
                          "reaction_energy"}
    assert [r["name"] for r in props["reaction_energy"]["roles"]] == ["reactants", "products"]
    assert props["reaction_energy"]["roles"][0]["weighted"] and not props["surface_energy"]["roles"][0]["weighted"]
    cands = {c["label"]: c for c in v["candidates"]}
    assert cands["graphene_opt.out"]["check"] == "ok"                     # spin-polarized as 4LG_AA
    assert cands["diamond_opt.out"]["check"] == "refuse"
    assert any("the functional differs" in r for r in cands["diamond_opt.out"]["reasons"])
    assert "surface" in cands["graphene_opt.out"]["fits"] and "bulk" not in cands["graphene_opt.out"]["fits"]
    assert "4LG_AA_opt.out" not in cands                                  # not the run itself
    known = {c["calc_id"] for c in v["candidates"]}
    g = cands["graphene_opt.out"]["calc_id"]
    assert PR.reference_argv("reaction_energy", {"reactants": [f"48*{g}"]}, False, known) == \
        ["--reference", f"reactants=48*{g}"]
    assert PR.reference_argv("surface_energy", {}, True, known) == ["--reference", "bulk=auto"]
    for bad, why in (({"reactants": ["../etc/passwd"]}, "not a calculation"),
                     ({"reactants": [f"{g} --db-path x"]}, "not a calculation"),
                     ({"bulk": [g]}, "has no reference")):
        with pytest.raises(ValueError, match=why):
            PR.reference_argv("reaction_energy", bad, False, known)
    with pytest.raises(ValueError, match="coefficient"):
        PR.reference_argv("surface_energy", {"bulk": [f"2*{g}"]}, False, known)
    with pytest.raises(ValueError, match="not a reference energy"):
        PR.reference_argv("relaxation_time_dpt", {"strained_runs": [g]}, False, known)


def test_a_phonon_supercell_is_the_structure_of_its_cell(suite):
    """The suite's phonon runs (SCELPHONO 2x2x2, 2x2, 2): their atoms and energy are the supercell's,
    the cell the extractor reads the input cell's. h-BN's 2x2 supercell (B4N4) is no BN sheet with three
    BN adsorbates; every diamond run, its supercells, CPHF and FREQ included, is the lowest-energy C phase
    (no formation energy against its own supercell); a SCELPHONO run's thermodynamics (per input cell)
    are scaled to its supercell."""
    from mace.properties import reference as Ref
    db = suite / "suite" / "materials.db"
    for name in ("hBN_phonon", "hBN_phonon_dos"):
        rows, prov = _rows(db, _calc(db, name, "FREQ"))
        assert "adsorption_energy_ev" not in rows and "in the same cell" in prov["adsorption_energy"]["reason"]
    for material, ct in (("diamond", "SP"), ("diamond", "OPT"), ("diamond", "FREQ"), ("diamond_cphf", "CPHF"),
                         ("diamond_phonon", "FREQ"), ("diamond_phonon_dos", "FREQ")):
        rows, prov = _rows(db, _calc(db, material, ct))
        assert "formation_energy_ev_per_atom" not in rows, material
        assert "lowest-energy C run" in prov["formation_energy"]["reason"], (material, prov["formation_energy"])
    parts = {}
    for stem in ("diamond_phonon_dos", "diamond_freq", "hBN_phonon_band"):
        out = next((suite / "suite").rglob(f"{stem}.out"))
        props = _extract(out)
        parts[stem] = Ref.Part(out.name, props, output_file=out, text=out.read_text(errors="replace"))
    sc, prim = parts["diamond_phonon_dos"], parts["diamond_freq"]
    assert (sc.multiple, sc.atoms, prim.multiple, prim.atoms) == (8, 16, 1, 2)
    assert sc.cell == pytest.approx(8 * prim.cell) and Ref.same_structure(sc, prim)
    assert parts["hBN_phonon_band"].multiple == 4 and Ref.thermo_cells(sc) == 8
    # the thermal correction H - E per atom: a few mHa for both (the supercell's was 33 Ha/atom, its H per C2
    # against its C16 energy)
    for p in (sc, prim):
        assert 0 < (Ref.thermo(p, None)[0]["H"] - p.energy) / p.atoms < 0.01, p.label


# ------------------------------------------------------------------ the test/ corpus
corpus = pytest.mark.skipif(not TEST_DATA.is_dir(), reason="test/ data corpus not present (gitignored)")


@corpus
def test_lifsi_and_ec_bind_with_freq_thermodynamics():
    freq = TEST_DATA / "FREQ"
    props = _extract(freq / f"1LiFSI-1EC-conf1{MOL}_freq_HSESOL3C_optimized_temp.out",
                     {"reactants": [str(freq / f"LiFSI{MOL}_freq_HSESOL3C_optimized_temp.out"),
                                    str(freq / f"EC{MOL}_freq_HSESOL3C_optimized_temp.out")]})
    prov = props["extraction_provenance"]["derived"]["reaction_energy"]
    assert prov["status"] == "stored", prov
    # by hand from the three outputs (E with D3 and gCP; G; H = G + TS), 298.15 K
    assert props["reaction_energy_ev"] == pytest.approx((-1696.98034089 + 1355.81867661 + 341.1108664474) * EV, rel=1e-9)
    assert props["reaction_energy_ev"] == pytest.approx(-1.38228, abs=1e-5)
    assert props["reaction_gibbs_energy_ev"] == pytest.approx(-0.83867, abs=1e-5)
    assert props["reaction_enthalpy_ev"] == pytest.approx(-1.31052, abs=1e-5)
    assert props["reaction_gibbs_energy_kj_mol"] == pytest.approx(-80.9195, abs=1e-3)
    d = props["reaction_energy"]
    assert d["equation"] == "1 F2LiNO4S2 + 1 C3H4O3 → 1 C3H4F2LiNO7S2" and d["warnings"] == []
    assert d["thermo"]["temperature_k"] == 298.15


@corpus
def test_fsi_on_four_layer_graphene():
    sp = TEST_DATA / "SP"
    props = _extract(sp / "4LG_FSI_2x2_ABAB_opt_sp.out",
                     {"surface": [str(sp / "4LG_FSI_2x2_ABAB_graphene_opt_sp.out")],
                      "adsorbate": [str(sp / "4LG_FSI_2x2_ABAB_FSI_opt_sp.out")]})
    # -4992.927069008659 - (-3644.644016049629) - (-1348.1617126054202) = -0.12134035 Ha
    assert props["adsorption_energy_ev"] == pytest.approx(-0.12134035361 * EV, rel=1e-8)
    assert props["adsorption_energy_ev"] == pytest.approx(-3.30184, abs=1e-5)
    d = props["adsorption_energy"]
    assert d["k_adsorbates"] == 1 and d["complex"] == "C96F2NO4S2" and d["adsorbate"] == "F2NO4S2"
    assert "no FREQ thermodynamics" in d["thermo_note"] and "BSSE" in d["not_included"]


@corpus
def test_another_functional_is_refused_and_an_allotrope_is_measured_against_diamond():
    opt = TEST_DATA / "OPT"
    props = _extract(opt / "4LG_2x2_ABAB_opt_HSESOL3C_optimized.out", {"bulk": [str(opt / "1_dia_opt_rev1.out")]})
    e = props["extraction_provenance"]["derived"]["surface_energy"]
    assert e["status"] == "skipped" and "the functional differs" in e["reason"] and "HSESOL3C" in e["reason"]
    assert not any(k.startswith("surface_energy_") for k in props)
    t2 = opt / "3.4^9T2_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1.out"
    props = _extract(t2, {"elements": [str(opt / "1_dia_opt_rev1.out")]})
    dia = _extract(opt / "1_dia_opt_rev1.out")
    t = _extract(t2)
    want = (t["total_energy_corrected_au"] / 20 - dia["total_energy_corrected_au"] / 2) * EV
    assert props["formation_energy_ev_per_atom"] == pytest.approx(want, rel=1e-9)
    assert props["formation_energy_ev_per_atom"] == pytest.approx(0.99714, abs=1e-5)
    assert any("write their basis sets in the deck" in w for w in props["formation_energy"]["warnings"])


@corpus
def test_the_database_finds_the_surface_and_adsorbate(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("4LG_FSI_2x2_ABAB_opt_sp", "4LG_FSI_2x2_ABAB_graphene_opt_sp", "4LG_FSI_2x2_ABAB_FSI_opt_sp",
                 "4LG_FSI_2x2_AA_graphene_opt_sp"):
        for ext in (".out", ".d12"):
            shutil.copy(TEST_DATA / "SP" / (name + ext), runs / (name + ext))
    _cli(tmp_path, "analyze", "--extract-properties", "runs", "--db-path", "runs/materials.db")
    db = runs / "materials.db"
    cx = _calc(db, "4LG_FSI_2x2_ABAB", "SP")
    rows, prov = _rows(db, cx)
    e = prov["adsorption_energy"]
    assert e["status"] == "stored", e
    # the clean surface of the complex's own cell (62.46609 Å²), not the AA stacking's (62.468723 Å²)
    assert e["references"]["surface"][0]["calc_id"] == _calc(db, "4LG_FSI_2x2_ABAB_graphene", "SP")
    assert e["references"]["adsorbate"][0]["calc_id"] == _calc(db, "4LG_FSI_2x2_ABAB_FSI", "SP")
    assert rows["adsorption_energy_ev"][0] == pytest.approx(-3.30184, abs=1e-5)
    assert R.CHOICES_KEY not in rows                                     # a lookup is not saved
