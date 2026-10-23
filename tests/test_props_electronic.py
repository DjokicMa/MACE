"""Electronic derived properties (mace/properties/defs: band_edges,
dos_analysis, orbital_composition) on real CRYSTAL23 outputs: the example
suite (always), tests/data/orbitals, and the gitignored test/ corpus when it
is there. Each value is checked against a hand calculation from the raw
BAND.DAT / DOSS.DAT numbers (read here by a parser of their own, not MACE's),
against CRYSTAL's printed numbers, or against the extractor's own keys; and
the extraction through mace_cli stores them in materials.db."""
import contextlib
import io
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.properties import registry as R
from mace.utils.property_extractor import CrystalPropertyExtractor

AU_V = 2.18769126364e6                     # m/s per atomic unit of velocity
HA = 27.211386245988
ORB = REPO_ROOT / "tests" / "data" / "orbitals"
trapz = getattr(np, "trapezoid", None) or np.trapz


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    return ES.copy_suite(tmp_path_factory.mktemp("suite") / "s")


def _extract(path):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(path))


def _band_dat(path):
    """(k, [spin blocks (nk, nbands) in hartree from E_F]) of a BAND.DAT, read directly."""
    lines = Path(path).read_text().splitlines()
    head = lines[0].split()
    nk, nb, nspin = (int(head[head.index(w) + 1]) for w in ("NKPT", "NBND", "NSPIN"))
    rows = _records(lines, 1 + nb)
    return rows[:nk, 0], [rows[s * nk:(s + 1) * nk, 1:] for s in range(nspin)]


def _records(lines, width):
    """The data records of a CRYSTAL .DAT (a record wraps over several lines when it has many columns)."""
    vals = [float(v) for ln in lines if ln.strip() and ln.lstrip()[0] not in "#@&" for v in ln.split()]
    return np.array(vals).reshape(-1, width)


def _doss_dat(path):
    """(E in eV, columns per spin block in states/eV) of a DOSS.DAT, read directly (the plotters' 27.2114)."""
    lines = Path(path).read_text().splitlines()
    head = next(ln for ln in lines if "NEPTS" in ln).split()
    ne, npr, nspin = (int(head[head.index(w) + 1]) for w in ("NEPTS", "NPROJ", "NSPIN"))
    rows = _records(lines, 1 + npr)
    return rows[:ne, 0] * 27.2114, [rows[s * ne:(s + 1) * ne, 1:] / 27.2114 for s in range(nspin)]


def _corpus(pattern):
    hits = sorted(TEST_DATA.glob(pattern)) if TEST_DATA.is_dir() else []
    if not hits:
        pytest.skip(f"test/ corpus file {pattern} not present (gitignored)")
    return hits[0]


# ------------------------------------------------------------------ BAND
def test_diamond_gap_is_indirect_gamma_to_delta(suite):
    out = suite / "2-dense-3d" / "diamond" / "BAND" / "diamond_band.out"
    p = _extract(out)
    assert p["band_path_gap_type"] == "indirect"
    assert p["band_path_vbm_k"] == "Γ" and p["band_path_direct_gap_k"] == "Γ"
    assert p["band_path_cbm_k"].endswith("of Γ–X")          # the Δ valley, 0.7 of the way to X
    k, blocks = _band_dat(next(out.parent.glob("*.BAND.DAT")))
    a = blocks[0]
    # its SCF ended spin-locked (SPINLOCK): the file is E - (-1.0), CRYSTAL's "no Fermi level"; 12 electrons,
    # 6 occupied bands per spin (bands 1-6)
    occ = np.arange(a.shape[1]) < 6
    vbm, cbm = a[:, occ].max(), a[:, ~occ].min()
    # the gap is the extractor's band_fundamental_gap_ev; the direct gap at Γ its band_direct_gap_ev
    assert (cbm - vbm) * HA == pytest.approx(p["band_fundamental_gap_ev"], abs=1e-6)
    assert p["band_edges"]["direct_gap_ev"] == pytest.approx(p["band_direct_gap_ev"], abs=1e-6)
    assert p["band_edges"]["cbm"]["frac"][0] == pytest.approx(0.71 * 0.5, abs=0.01)
    # valence band width: Γ1 (the bottom of the 2s band, band 3) to the VBM: 23.16 eV (B3LYP; exp. ~23 eV)
    assert p["valence_band_width_ev"] == pytest.approx((vbm - a[:, 2].min()) * HA, abs=1e-6)
    assert p["valence_band_width_ev"] == pytest.approx(23.16, abs=0.01)
    # the top valence band (band 6, the highest of Γ25'): its dispersion runs down to W (-8.95 eV)
    assert p["vbm_band_dispersion_ev"] == pytest.approx((a[:, 5].max() - a[:, 5].min()) * HA, abs=1e-6)
    assert p["band_edges"]["widths"]["all"]["valence"]["bands"] == [3, 6]
    # spin-polarized (MACE's SPIN default) but not magnetic: no splitting
    assert p["vbm_spin_splitting_mev"] == 0.0 and p["band_max_spin_splitting_mev"] == 0.0
    assert p["band_path_gap_alpha_ev"] == p["band_path_gap_beta_ev"]
    assert "fermi_velocity_mean_m_s" not in p
    rep = p["extraction_provenance"]["derived"]["fermi_velocity"]
    assert rep["status"] == "skipped" and "not a metal" in rep["reason"]


def test_polyacetylene_gap_is_direct_at_the_zone_edge(suite):
    p = _extract(suite / "4-polymer-1d" / "polyacetylene" / "BAND" / "polyacetylene_band.out")
    assert p["band_path_gap_type"] == "direct"           # the Peierls gap at X
    assert p["band_path_vbm_k"] == p["band_path_cbm_k"] == "X"
    assert p["band_edges"]["gap_ev"] == pytest.approx(p["band_fundamental_gap_ev"], abs=1e-6)


def test_graphene_fermi_velocity_at_the_dirac_point(suite):
    out = suite / "3-slab-2d" / "graphene" / "BAND" / "graphene_band.out"
    p = _extract(out)
    assert p["band_path_gap_type"] == "semimetal"          # its DOSS run (../DOSS) has no states at E_F
    cr = p["fermi_velocity_crossings"]["crossings"]
    assert p["fermi_level_crossings"] == 2 and {c["label"] for c in cr} == {"K"}
    assert {c["band"] for c in cr} == {6, 7}               # π and π* touch at K
    # by hand: the slope of the last two path points before K (the end of Γ–K, rows 465-1000 of the
    # file); the k step from the segment's ends, as the 5-figure k column's single differences are
    # quantized (0.00168 / 0.00170 for 0.0016834: a median of them put v_F 1.6 % low, 1.006e6)
    k, blocks = _band_dat(next(out.parent.glob("*.BAND.DAT")))
    a = blocks[0]                                           # spin-polarized (SPIN), alpha = beta
    dk = (k[999] - k[464]) / (999 - 464)
    for c in cr:
        b = c["band"] - 1
        v = abs(a[-1, b] - a[-2, b]) / dk * AU_V
        assert c["v_m_s"] == pytest.approx(v, rel=1e-3)
    # HSEsol-3c: 1.016e6 m/s (experiment ~1.0-1.1e6 m/s)
    assert p["fermi_velocity_mean_m_s"] == pytest.approx(1.0156e6, rel=2e-3)
    # spin-polarized (MACE's SPIN default) and not magnetic: alpha = beta, no splitting
    assert p["spin_polarized"] and p["band_max_spin_splitting_mev"] == 0.0


def test_sodalite_cbm_on_a_flat_band_is_named_at_gamma(suite):
    """SOD's lowest empty band is flat to the file's 5 decimals around Γ (0.31712 Ha on rows 322-328):
    the edge is named at the high-symmetry point it ties with, not at the first tied row (0.99 of H–Γ);
    its energy stays the band's minimum."""
    out = suite / "1-porous" / "SOD" / "BAND" / "SOD_band.out"
    p = _extract(out)
    assert p["band_path_cbm_k"] == "Γ" and p["band_edges"]["cbm"]["label"] == "Γ"
    assert p["band_path_vbm_k"] == "H"
    _, blocks = _band_dat(next(out.parent.glob("*.BAND.DAT")))
    bottom = min(x[:, x.max(axis=0) > 1e-3].min() for x in blocks)
    assert p["band_edges"]["cbm"]["energy_ev"] == pytest.approx(bottom * HA, abs=1e-6)


def test_spin_splitting_of_a_magnetic_slab():
    band = _corpus("BAND/4LG_2x2_AA_opt_HSESOL3C_optimized_band.BAND.DAT")
    p = _extract(band.with_name(band.name.replace(".BAND.DAT", ".out")))
    _, (a, b) = _band_dat(band)
    near = (np.abs(a) <= 1 / HA) | (np.abs(b) <= 1 / HA)
    assert p["band_max_spin_splitting_mev"] == pytest.approx(np.abs(a - b)[near].max() * HA * 1000, abs=1e-6)
    assert p["band_max_spin_splitting_mev"] > 50
    assert p["band_path_gap_type"] == "semimetal"
    # a magnetic metal: no edges to split, each channel's crossings counted
    assert "vbm_spin_splitting_mev" not in p
    assert {c["spin"] for c in p["fermi_velocity_crossings"]["crossings"]} == {"alpha", "beta"}


def test_corpus_gaps_agree_with_the_extractor():
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    n = 0
    for out in sorted((TEST_DATA / "BAND").glob("*_band.out")):
        if "MOLECULE" in out.name or not out.with_name(out.stem + ".BAND.DAT").is_file():
            continue
        p = _extract(out)
        if p.get("band_path_gap_type") in ("direct", "indirect"):
            assert p["band_edges"]["gap_ev"] == pytest.approx(p["band_fundamental_gap_ev"], abs=1e-6), out.name
            n += 1
        if p.get("band_path_gap_type") == "direct":
            assert p["band_gap_is_direct"], out.name
    assert n >= 10


# ------------------------------------------------------------------ DOSS
def test_diamond_dos_carriers_by_hand(suite):
    out = suite / "2-dense-3d" / "diamond" / "DOSS" / "diamond_doss.out"
    p = _extract(out)
    curve = p["carrier_concentration_300k"]
    vbm, cbm, size = curve["vbm_ev"], curve["cbm_ev"], curve["cell_size"]
    E, (a, b) = _doss_dat(next(out.parent.glob("*.DOSS.DAT")))
    g = np.clip(a[:, -1] - b[:, -1], 0, None)               # both spins (the beta block is written negative)
    g[1] = g[-2] = 0.0                                      # CRYSTAL's window-edge pile-up bins
    g[(E > vbm) & (E < cbm)] = 0.0
    kT = 8.617333262e-5 * 300.0
    mu = p["intrinsic_fermi_level_300k_ev"]
    cb, vb = E >= cbm, E <= vbm
    with np.errstate(over="ignore"):              # exp overflows to inf far from mu: f = 0 there
        n = trapz(g[cb] / (1 + np.exp((E[cb] - mu) / kT)), E[cb]) / size
        pp = trapz(g[vb] / (1 + np.exp((mu - E[vb]) / kT)), E[vb]) / size
    assert n == pytest.approx(pp, rel=1e-4)                # intrinsic: n = p
    assert p["intrinsic_carrier_concentration_300k_cm3"] == pytest.approx(math.sqrt(n * pp), rel=1e-3)
    assert vbm < mu < cbm and abs(mu - (vbm + cbm) / 2) < 0.2
    # diamond's primitive cell, a = 3.57 A fcc: a^3 / 4
    assert size == pytest.approx(3.57 ** 3 / 4 * 1e-24, rel=0.02)
    nn, ps = np.array(curve["n"]), np.array(curve["p"])
    assert (np.diff(nn) > 0).all() and (np.diff(ps) < 0).all()
    assert curve["units"]["n"] == "cm⁻³"


def test_diamond_dos_masses_and_edge_character(suite):
    from mace.transport import mass
    # the suite's DOSS covers every band (1-36) at the Standard step (20000 points: a 0.045 eV grid over
    # 890 eV; the preview's 1000 points gave 0.9 eV, too coarse for these), the run of the corpus a
    # -10..+20 eV window (0.03 eV): the masses agree with mace tau, and the physics holds, on both
    sout = suite / "2-dense-3d" / "diamond" / "DOSS" / "diamond_doss.out"
    sp = _extract(sout)
    for c in ("electron", "hole"):
        with contextlib.redirect_stdout(io.StringIO()):
            assert sp[f"dos_effective_mass_{c}"] == pytest.approx(mass.dos_mass(sout.with_suffix(".DOSS.DAT"), c, 3)["m_d"])
    runs = [sp]
    if (TEST_DATA / "DOSS").is_dir():
        runs.append(_extract(_corpus("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out")))
    for p in runs:
        # six Δ valleys: per valley m_d = 1.85 / 6^(2/3) = 0.56 m_e, (m_l m_t^2)^(1/3) of diamond (~0.5-0.6)
        assert p["dos_effective_mass_electron"] / 6 ** (2 / 3) == pytest.approx(0.56, abs=0.03)
        assert p["vbm_character"] == "C p" and p["cbm_character"] == "C p"
        comp = p["band_edge_composition"]["vbm"]["channels"]["total"]
        assert sum(comp["shares"].values()) + comp["unassigned"] == pytest.approx(1.0)
        assert "d_band_center_ev" not in p                     # no transition metal
        assert "sommerfeld_gamma_mj_mol_k2" not in p           # an insulator


def test_sommerfeld_coefficient_of_a_metal():
    out = _corpus("DOSS/3,4^2T1-CA_rev1_sp_B3LYP-D3-D3_optimized_doss.out")
    p = _extract(out)
    # gamma = (pi^2/3) k_B^2 N_A N(E_F) = 2.3571 mJ/(mol K^2) per state/eV
    assert p["sommerfeld_gamma_mj_mol_k2"] == pytest.approx(2.3571 * p["dos_at_fermi_total"], rel=1e-4)
    assert p["sommerfeld_gamma_per_atom_mj_mol_k2"] == pytest.approx(p["sommerfeld_gamma_mj_mol_k2"] / 5)
    assert p["fermi_level_character"] == "C p"
    assert "carrier_concentration_300k" not in p


def test_d_band_of_copper_and_titanium_by_hand():
    out = _corpus("DOSS/PbCuSeCl_*_doss.out")
    p = _extract(out)
    E, (a, b) = _doss_dat(out.with_name(out.stem + ".DOSS.DAT"))
    labels = [ln.split("#")[1].strip() for ln in out.with_suffix(".d3").read_text().splitlines() if "#" in ln]
    rho = a[:, labels.index("CU D")] - b[:, labels.index("CU D")]
    m = E <= 10.0
    E, rho = E[m], rho[m]
    w = trapz(rho, E)
    c = trapz(E * rho, E) / w
    assert p["d_band_element"] == "Cu"
    assert p["d_band_center_ev"] == pytest.approx(c, abs=2e-3)
    assert p["d_band_filling"] == pytest.approx(trapz(rho[E <= 0], E[E <= 0]) / w, abs=1e-3)
    assert -3.0 < p["d_band_center_ev"] < -2.0 and p["d_band_filling"] > 0.95      # Cu(I), d10
    occ = E <= 0
    assert p["d_band_center_occupied_ev"] == pytest.approx(trapz(E[occ] * rho[occ], E[occ]) / trapz(rho[occ], E[occ]),
                                                           abs=2e-3)
    # the values the d-band property was validated on: Cu(I) -2.46 eV, filling 0.98
    assert p["d_band_center_ev"] == pytest.approx(-2.46, abs=0.01)
    assert p["d_band_filling"] == pytest.approx(0.98, abs=0.01)
    assert p["d_band_center_occupied_ev"] < p["d_band_center_ev"] < 0
    t = _extract(_corpus("DOSS/Ti2O4Pb3Se3_*_doss.out"))
    assert t["d_band_element"] == "Ti" and t["d_band_center_ev"] > 1.5 and t["d_band_filling"] < 0.25   # Ti(IV), d0
    assert t["d_band_center_ev"] == pytest.approx(2.59, abs=0.01)
    assert t["d_band_filling"] == pytest.approx(0.16, abs=0.01)
    # d0: the few occupied d states are the Ti d share of the O/Se p valence band, far below E_F
    assert t["d_band_center_occupied_ev"] < -2.0
    assert t["cbm_character"] == "Ti d" and t["vbm_character"] == "Se p"


def test_d_band_labels_name_their_window():
    assert "E<sub>F</sub> + 10 eV" in R.output_of("d_band_center_ev").label
    assert "occupied" in R.output_of("d_band_center_occupied_ev").label
    assert "d_band_center_occupied_ev" in {o.name for o in R.get("d_band_center").outputs}


# ------------------------------------------------------------------ metal or semimetal (gapless BAND runs)
def _fermi_dos_by_hand(dat, n_atoms):
    """N(E_F) per atom: the total DOS (both spins), mean over E_F +- 0.05 eV, from the raw DOSS.DAT."""
    E, blocks = _doss_dat(dat)
    g = sum(np.abs(b[:, -1]) for b in blocks)
    xs = np.linspace(-0.05, 0.05, 41)
    return float(np.interp(xs, E, g).mean()) / n_atoms


@pytest.mark.parametrize("band,n_atoms,kind", [
    ("BAND/4LG_2x2_AA_opt_HSESOL3C_optimized_band.out", 96, "semimetal"),         # four-layer graphene
    ("BAND/4LG_2x2_ABAB_opt_HSESOL3C_optimized_band.out", 96, "semimetal"),
    ("BAND/3,4^2T1-CA_rev1_sp_B3LYP-D3-D3_optimized_band.out", 5, "metal"),       # a metallic carbon net
])
def test_corpus_gapless_runs_are_metal_or_semimetal_by_the_dos(band, n_atoms, kind):
    out = _corpus(band)
    dos = TEST_DATA / "DOSS" / out.name.replace("_band.out", "_doss.DOSS.DAT")
    p = _extract(out)
    assert p["band_path_gap_type"] == kind
    assert p["band_path_semimetal_threshold"] == 0.05
    assert p["band_path_fermi_dos_per_atom"] == pytest.approx(_fermi_dos_by_hand(dos, n_atoms), rel=1e-3)
    fd = p["band_edges"]["fermi_dos"]
    assert fd["file"] == dos.name and fd["atoms"] == n_atoms
    # one classification: the extractor's electronic_classification agrees
    assert p["electronic_classification"] == kind
    # the Fermi velocity of a semimetal is still there (the Dirac cones of graphene)
    assert p["fermi_level_crossings"] > 0


def test_graphene_semimetal_and_iron_metal(suite):
    from mace.properties.defs import band_edges as BE
    out = suite / "3-slab-2d" / "graphene" / "BAND" / "graphene_band.out"
    p = _extract(out)
    dos = suite / "3-slab-2d" / "graphene" / "DOSS" / "graphene_doss.DOSS.DAT"
    assert p["band_path_fermi_dos_per_atom"] == pytest.approx(_fermi_dos_by_hand(dos, 2), rel=1e-3)
    assert p["band_path_fermi_dos_per_atom"] < 0.05 and p["electronic_classification"] == "semimetal"
    # bcc Fe (one atom; 8 valence electrons below E_F in its DOSS.DAT): 0.89 states/eV/atom
    fe = REPO_ROOT / "tests" / "data" / "spin_fe"
    r = BE.classify(BE.analyze(fe / "fe_band.BAND.DAT"), fe / "fe_doss.DOSS.DAT", 1)
    assert r["character"] == "metal"
    assert r["fermi_dos"]["per_atom"] == pytest.approx(_fermi_dos_by_hand(fe / "fe_doss.DOSS.DAT", 1), rel=1e-3)
    assert r["fermi_dos"]["per_atom"] == pytest.approx(0.885, abs=0.01)
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    a = AdvancedElectronicAnalyzer().analyze_material(fe / "fe_band.BAND.DAT", fe / "fe_doss.DOSS.DAT", 1)
    assert a["electronic_classification"] == "metal" and a["fermi_dos_per_atom"] == r["fermi_dos"]["per_atom"]


def test_a_gapless_run_without_a_dos_is_not_guessed(suite, tmp_path):
    import shutil
    src = suite / "3-slab-2d" / "graphene" / "BAND"
    shutil.copytree(src, tmp_path / "BAND")
    p = _extract(tmp_path / "BAND" / "graphene_band.out")
    assert p["band_path_gap_type"] == "metal or semimetal (no DOS to tell)"
    assert "band_path_fermi_dos_per_atom" not in p
    assert p["band_edges"]["fermi_dos"]["reason"].startswith("no DOSS run")
    assert p["fermi_level_crossings"] == 2


def test_semimetal_stored_through_mace_cli(suite, tmp_path):
    import shutil
    for kind in ("BAND", "DOSS"):
        shutil.copytree(suite / "3-slab-2d" / "graphene" / kind, tmp_path / "g" / "graphene" / kind)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        "g", "--db-path", "g/materials.db"], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                       timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    conn = sqlite3.connect(str(tmp_path / "g" / "materials.db"))
    rows = dict(conn.execute("SELECT property_name, COALESCE(property_value_text, property_value) FROM properties "
                             "WHERE property_name IN ('band_path_gap_type', 'electronic_classification', "
                             "'band_path_semimetal_threshold', 'band_path_fermi_dos_per_atom') "
                             "AND calc_id IN (SELECT calc_id FROM properties WHERE property_name = 'band_path_gap_type')"))
    assert rows["band_path_gap_type"] == "semimetal" and rows["electronic_classification"] == "semimetal"
    assert float(rows["band_path_semimetal_threshold"]) == 0.05 and float(rows["band_path_fermi_dos_per_atom"]) < 0.05


def test_effective_mass_labels_say_which_to_use():
    from mace.properties import display
    for c in ("electron", "hole"):
        for m in ("mc", "md"):
            o = R.output_of(f"effective_mass_{c}_{m}")
            assert "(accurate)" in o.label and "Use this one" in o.description
        quick = display.describe(f"real_{c}_effective_mass")
        assert "quick estimate" in quick["label"] and "one band along the path" in quick["label"]
        assert quick["group"] == "Effective masses" and "accurate" in quick["description"]
        rough = R.output_of(f"dos_effective_mass_{c}")
        assert "rough, from the smoothed DOS" in rough.label and f"effective_mass_{c}_md" in rough.description
    assert "quick estimate" in display.describe("real_average_effective_mass")["label"]


# ------------------------------------------------------------------ ORBITALS
@pytest.mark.parametrize("out,homo,lumo", [
    ("h2o/h2o_c.out", "O p", "H s"),                    # 1b1 lone pair; 4a1 O-H antibonding
    ("slab/slab_f.out", "O p", "Mg s"),                # MgO: O 2p valence, Mg 3s conduction
    ("fe/fe_f.out", "Fe d", "Fe d"),
])
def test_orbital_composition(out, homo, lumo):
    p = _extract(ORB / out)
    assert p["homo_character"] == homo and p["lumo_character"] == lumo
    comp = p["orbital_composition"]
    for k in comp["kpoints"]:
        for s in k["spins"]:
            assert s["homo"]["total"] == pytest.approx(1.0, abs=2e-3)      # Mulliken shares add up to 1
            assert sum(s["homo"]["by_element"].values()) == pytest.approx(s["homo"]["total"], abs=1e-4)
    fr = comp["frontier"]
    k = next(k for k in comp["kpoints"] if k["index"] == fr["k_index"])
    s = next(s for s in k["spins"] if s["spin"] == fr["spin"])
    assert s["homo"]["energy_au"] == pytest.approx(p["orbitals_homo_energy_au"])


def test_orbital_composition_keeps_both_spins_of_triplet_o2():
    p = _extract(ORB / "o2" / "o2_c.out")
    spins = p["orbital_composition"]["kpoints"][0]["spins"]
    assert [s["spin"] for s in spins] == ["ALPHA", "BETA"]
    # the alpha HOMO is the doubly degenerate π*, averaged over its pair
    assert len(spins[0]["homo"]["levels"]) == 2 and p["homo_character"] == "O p"


# ------------------------------------------------------------------ the database, through mace_cli
def test_extraction_through_mace_cli_stores_them(suite, tmp_path):
    import shutil
    root = tmp_path / "d"
    for kind in ("BAND", "DOSS"):
        shutil.copytree(suite / "2-dense-3d" / "diamond" / kind, root / "diamond" / kind)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        "d", "--db-path", "d/materials.db"], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                       timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    conn = sqlite3.connect(str(root / "materials.db"))
    names = {n for (n,) in conn.execute("SELECT property_name FROM properties UNION SELECT property_name FROM "
                                        "property_blobs")}
    blobs = {n for (n,) in conn.execute("SELECT property_name FROM property_blobs")}
    for k in ("band_path_gap_type", "band_path_vbm_k", "valence_band_width_ev", "vbm_spin_splitting_mev",
              "intrinsic_carrier_concentration_300k_cm3", "dos_effective_mass_electron", "vbm_character",
              "cbm_character"):
        assert k in names, k
    assert {"band_edges", "carrier_concentration_300k"} <= blobs
    units = dict(conn.execute("SELECT property_name, property_unit FROM properties"))
    assert units["valence_band_width_ev"] == "eV" and units["intrinsic_carrier_concentration_300k_cm3"] == "cm⁻³"


def test_every_electronic_property_is_registered():
    names = {p.name for p in R.properties()}
    assert {"band_gap_character", "band_widths", "band_spin_splitting", "fermi_velocity", "sommerfeld_coefficient",
            "d_band_center", "band_edge_character", "carrier_concentration", "dos_effective_mass",
            "orbital_composition"} <= names


def test_a_path_gap_narrower_than_0_1_ev_does_not_end_the_valence_band(suite):
    # SOD's O 2p band shows a 30 meV gap on H-Γ-P-N; the manifold ends at the real 0.48 eV gap below it
    out = suite / "1-porous" / "SOD" / "BAND" / "SOD_band.out"
    p = _extract(out)
    assert p["valence_band_width_ev"] == pytest.approx(3.87, abs=0.01)
    assert p["band_path_gap_type"] == "indirect" and p["band_path_vbm_k"] == "H"
