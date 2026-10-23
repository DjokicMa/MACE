"""Automatic effective masses for ``mace tau`` (mace/transport/mass.py).

* the parabola fit on synthetic bands of known mass, built in SI units and
  converted to the files' hartree and bohr^-1;
* a synthetic BAND.DAT with its .d3 and .out (simple cubic, Gamma-X along
  x): the k axis, the direction and the transport axis;
* a synthetic DOSS.DAT of a parabolic band: the DOS normalization;
* tau depends on m_d only, the mobility on m_c too (dpt);
* the corpus diamond (B3LYP-D3; test/BAND, test/DOSS) against the
  literature masses, and ``mace tau fit`` through mace_cli: masses found
  beside the strain series (tests/data/hpcc_116/strain, the same
  structure), an explicit --m-star winning, and the refusal without files.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, find_data
from mace.transport import dpt, mass

MACE_CLI = REPO_ROOT / "mace_cli"
STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"
HBAR, M_E = 1.054571817e-34, 9.1093837015e-31
HARTREE_J, BOHR_M = 4.3597447222071e-18, 0.529177210903e-10
DIA_BAND = "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
DIA_DOSS = "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss"


def _mace(*args, cwd=None):
    env = dict(os.environ, MACE_NO_BANNER="1")
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _parabola(m, k_inv_m):
    """E (J) = hbar^2 k^2 / (2 m m_e)."""
    return HBAR ** 2 * k_inv_m ** 2 / (2 * m * M_E)


# ------------------------------------------------------------------ the fit
@pytest.mark.parametrize("two_sided", [True, False])
def test_parabola_of_known_mass_in_file_units(two_sided):
    k = np.linspace(-2e8 if two_sided else 0.0, 2e8, 21)            # 1/m
    E = _parabola(0.26, k)
    m, r2 = mass.parabola_mass(k * BOHR_M, E / HARTREE_J, two_sided)    # bohr^-1, hartree
    assert m == pytest.approx(0.26, rel=1e-6) and r2 == pytest.approx(1.0)
    m, _ = mass.parabola_mass(k * BOHR_M, -E / HARTREE_J, two_sided)   # a valence-band top
    assert m == pytest.approx(-0.26, rel=1e-6)


def _write_cubic_band(d, a=5.0, n=101, mh=0.4, me=0.25, gap=0.05):
    """Simple cubic, Gamma-X along x (0.5 b1 = pi/a), a VB top and a CB bottom at Gamma."""
    kx = np.linspace(0, math.pi / (a / 0.529177210903), n)
    vb, cb = -kx ** 2 / (2 * mh), gap + kx ** 2 / (2 * me)
    rows = [f"# NKPT {n:5d} NBND     2 NSPIN     1", "# NPANEL     1", "#      1   A", f"#    {n}   A"]
    rows += [f"{x:14.7E}{v:15.7E}{c:15.7E}" for x, v, c in zip(kx, vb, cb)]
    (d / "sc_band.BAND.DAT").write_text("\n".join(rows) + "\n")
    (d / "sc_band.d3").write_text(f"NEWK\n4 8\n1 0\nBAND\nsc - GAMMA-X\n1 8 {n - 1} 1 2 1 0\n0 0 0 4 0 0\nEND\n")
    (d / "sc_band.out").write_text(" DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)\n"
                                   f"     {a:.5f}    0.00000    0.00000\n     0.00000    {a:.5f}    0.00000\n"
                                   f"     0.00000    0.00000    {a:.5f}\n")
    return d / "sc_band.BAND.DAT"


def test_synthetic_band_masses_units_direction_and_axis(tmp_path):
    f = _write_cubic_band(tmp_path)
    h = mass.band_masses(f, "hole", "x", 3, window_mev=200)
    e = mass.band_masses(f, "electron", "y", 3, window_mev=200)
    assert not [w for w in h["warnings"] if "does not match the lattice" in w]
    (dh,), (de,) = h["directions"], e["directions"]
    assert dh["mass"] == pytest.approx(0.4, rel=1e-4) and de["mass"] == pytest.approx(0.25, rel=1e-4)
    assert dh["direction"] == "Γ→X" and dh["along_axis"] and dh["cartesian"] == [1.0, 0.0, 0.0]
    # y is not on the path, but the cubic lattice makes it equivalent to x
    assert not de["along_axis"] and de["equivalent_to_axis"]
    assert h["valleys"] == 1 and h["m_c"] == pytest.approx(0.4, rel=1e-4)
    # one direction cannot give a 3D density-of-states mass
    assert h["m_d"] is None


def test_a_coarse_path_is_refused_not_fitted(tmp_path):
    f = _write_cubic_band(tmp_path, n=6)
    r = mass.band_masses(f, "electron", "x", 3, window_mev=40)
    assert not r["directions"] and "too coarse" in r["refused"][0]["reason"]


def test_dos_of_a_parabolic_band_gives_its_mass(tmp_path):
    a, m = 5.0, 0.5
    vol = (a / 0.529177210903) ** 3
    E = np.arange(-0.3, 0.4, 0.0005)
    gap = 0.05
    g = np.where(E <= 0, vol / (2 * math.pi ** 2) * (2 * m) ** 1.5 * np.sqrt(np.clip(-E, 0, None)), 0.0)
    g += np.where(E >= gap, vol / (2 * math.pi ** 2) * (2 * m) ** 1.5 * np.sqrt(np.clip(E - gap, 0, None)), 0.0)
    rows = [f"# NEPTS {len(E)} NPROJ     1 NSPIN     1", "#"] + [f"{x:14.7E}{y:12.4E}" for x, y in zip(E, g)]
    (tmp_path / "sc_doss.DOSS.DAT").write_text("\n".join(rows) + "\n")
    shutil.copy(_write_cubic_band(tmp_path).with_suffix("").with_suffix(".out"), tmp_path / "sc_doss.out")
    for carrier in ("hole", "electron"):
        r = mass.dos_mass(tmp_path / "sc_doss.DOSS.DAT", carrier, 3)
        assert r["m_d"] == pytest.approx(m, rel=0.03), carrier


def test_valley_counts_of_an_fcc_lattice():
    lat = np.array([[0, 1.77, 1.77], [1.77, 0, 1.77], [1.77, 1.77, 0]])
    ops = mass.holohedry(lat)
    assert len(ops) == 48
    assert mass.valley_count([0, 0, 0], lat, ops) == 1
    assert mass.valley_count([0.5, 0, 0.5], lat, ops) == 3          # X
    assert mass.valley_count([0.5, 0.5, 0.5], lat, ops) == 4        # L
    assert mass.valley_count([0.35, 0, 0.35], lat, ops) == 6        # Delta (diamond, Si CBM)


# ------------------------------------------------------------------ which mass the formulas take
def test_tau_takes_the_dos_mass_and_the_mobility_the_band_mass():
    a = dpt.dp_mobility(3, 0.3, 9.0, 500, 300, m_dos=0.9)
    b = dpt.dp_mobility(3, 0.6, 9.0, 500, 300, m_dos=0.9)
    assert a["tau_fs"] == pytest.approx(b["tau_fs"])
    assert a["mobility_cm2_Vs"] == pytest.approx(2 * b["mobility_cm2_Vs"])
    # isotropic band: unchanged from the m*^(5/2) form
    assert dpt.dp_mobility(3, 0.5, 9.0, 500)["tau_fs"] == pytest.approx(
        dpt.dp_mobility(3, 0.5, 9.0, 500, m_dos=0.5)["tau_fs"])
    # 2D: m_d given equals sqrt(m* m_other)
    assert dpt.dp_mobility(2, 0.17, 2.72, 28.94, 300, 1.12)["tau_fs"] == pytest.approx(
        dpt.dp_mobility(2, 0.17, 2.72, 28.94, 300, m_dos=math.sqrt(0.17 * 1.12))["tau_fs"])
    # tau ~ m_d^(-3/2) in 3D
    assert dpt.dp_mobility(3, 0.5, 9, 500, m_dos=0.25)["tau_fs"] == pytest.approx(
        8 ** 0.5 * dpt.dp_mobility(3, 0.5, 9, 500, m_dos=0.5)["tau_fs"])


# ------------------------------------------------------------------ the corpus diamond
def test_diamond_band_masses_against_the_literature():
    """B3LYP-D3 diamond, path Gamma-X-U|K-Gamma-L-W-X (1000 points). Literature: electrons m_l 1.4-1.56
    (Nava 1980; Naka et al., PRB 88, 035205 (2013)); holes from Naka's Luttinger parameters
    (gamma1 2.67, gamma2 -0.403, gamma3 0.680): [100] 0.54 / 0.29, [111] 0.76 / 0.25. The calculation has
    no spin-orbit coupling (a threefold top), so the hole masses compare within ~25%."""
    dat = find_data(DIA_BAND + ".BAND.DAT")
    e = mass.band_masses(dat, "electron", "x", 3)
    (d,) = e["directions"]
    assert 1.4 <= d["mass"] <= 1.75 and d["equivalent_to_axis"] and e["valleys"] == 6
    h = mass.band_masses(dat, "hole", "x", 3)
    assert h["bands"] == [4, 5, 6] and h["at"] == ["Γ"]
    gx = sorted(x["mass"] for x in h["directions"] if x["direction"] == "Γ→X")
    gl = sorted(x["mass"] for x in h["directions"] if x["direction"] == "Γ→L")
    assert gx[0] == pytest.approx(0.29, rel=0.15) and gx[-1] == pytest.approx(0.54, rel=0.25)
    assert gl[0] == pytest.approx(0.25, rel=0.4) and gl[-1] == pytest.approx(0.76, rel=0.25)
    # the DOS mass of the three bands vs Naka's (0.667^1.5 + 0.26^1.5 + 0.375^1.5)^(2/3) = 0.94
    assert h["m_d"] == pytest.approx(0.94, rel=0.2)
    # the f25 of the same run gives the same masses
    h25 = mass.band_masses(find_data(DIA_BAND + ".f25"), "hole", "x", 3)
    assert h25["m_d"] == pytest.approx(h["m_d"], rel=0.01) and h25["m_c"] == pytest.approx(h["m_c"], rel=0.01)


def test_band_dat_k_step_from_the_segment_ends(tmp_path):
    """The BAND.DAT's k column has 5 significant figures, so its single differences are quantized
    (0.00420 / 0.00421 for 0.0042061): the k step is (k_last - k_first) / (points - 1) of each
    segment, which matches the f25 header's step of the same run to its rounding (a median of the
    differences was 0.2-0.6 % off, the masses with it), and the masses of both files agree. The
    example suite's diamond BAND run (the corpus's 1_dia_..._band, the same files)."""
    import lzma
    from mace.examples_suite import default_suite_dir
    src = default_suite_dir() / "2-dense-3d" / "diamond" / "BAND"
    for f in src.iterdir():
        dest = tmp_path / f.name.removesuffix(".xz")
        dest.write_bytes(lzma.decompress(f.read_bytes()) if f.suffix == ".xz" else f.read_bytes())
    dat, f25 = tmp_path / "diamond_band.BAND.DAT", tmp_path / "diamond_band.f25"
    a, b = mass.read_band(dat), mass.read_band(f25)
    assert len(a.segments) == len(b.segments)
    for sa, sb in zip(a.segments, b.segments):
        assert sa.dk == pytest.approx(sb.dk, rel=2e-4), (sa.name, sa.dk, sb.dk)
    ha, hb = mass.band_masses(dat, "hole", "x", 3), mass.band_masses(f25, "hole", "x", 3)
    # (its SCF ended spin-locked: both files' E_F is the derived one, mace.utils.fermi_level; the heavy band
    # along K-Γ, ~2 m_e, carries the DAT's 5 decimals into the third figure)
    assert ha["m_c"] == pytest.approx(hb["m_c"], rel=2e-3) and ha["m_d"] == pytest.approx(hb["m_d"], rel=2e-3)
    assert "1.00 of K–Γ" not in ha["at"]                  # the flat top ties on several points: the vertex is Γ


def test_diamond_dos_mass_per_valley():
    """Six Delta valleys: the DOS mass / 6^(2/3) is one valley's (m_l m_t^2)^(1/3), 0.50-0.57 from the
    literature masses (1.56, 0.28 and 1.4, 0.36)."""
    r = mass.dos_mass(find_data(DIA_DOSS + ".DOSS.DAT"), "electron", 3, edge_ev=5.996)
    assert r["m_d"] / 6 ** (2 / 3) == pytest.approx(0.535, rel=0.12)
    assert r["npol"] == 14 and any("Fourier-Legendre" in w for w in r["warnings"])


@pytest.fixture
def diamond_run(tmp_path):
    """The strain series with the diamond's BAND and DOSS runs beside it (nested: the folder above holds
    nothing else)."""
    d = tmp_path / "dia" / "sp"
    d.mkdir(parents=True)
    for f in STRAIN.glob("dia_sp_dp_x_*.out"):
        shutil.copy(f, d / f.name)
    band, doss = find_data(DIA_BAND + ".BAND.DAT"), find_data(DIA_DOSS + ".DOSS.DAT")
    for src, stem in ((band, "dia_sp_band"), (doss, "dia_sp_doss")):
        base = mass.run_stem(src)
        for suf in (".d3", ".out"):
            shutil.copy(src.parent / (base + suf), d / (stem + suf))
        shutil.copy(src, d / (stem + (".BAND.DAT" if stem.endswith("band") else ".DOSS.DAT")))
    return d


def test_mace_tau_fit_finds_the_masses_beside_the_outputs(diamond_run):
    outs = sorted(p.name for p in diamond_run.glob("dia_sp_dp_x_*.out"))
    r = _mace("tau", "fit", *outs, "--carrier", "electron", "--json", cwd=diamond_run)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    ms, md = res["inputs"]["m_star"], res["inputs"]["m_d"]
    assert 1.4 <= ms["value"] <= 1.75 and "dia_sp_band.BAND.DAT" in ms["source"]
    assert md["value"] == pytest.approx(0.535, rel=0.12) and "6 equivalent valleys" in md["source"]
    assert res["masses"]["band"]["directions"][0]["direction"].startswith("Γ")
    # the text form says where each mass came from
    t = _mace("tau", "fit", *outs, "--carrier", "hole", cwd=diamond_run)
    assert t.returncode == 0, t.stderr
    assert "Effective masses (automatic)" in t.stdout and "3 bands meet at the edge" in t.stdout
    assert "m_d " in t.stdout and "RELAXTIM" in t.stdout


def test_an_explicit_m_star_wins(diamond_run):
    outs = sorted(p.name for p in diamond_run.glob("dia_sp_dp_x_*.out"))
    r = _mace("tau", "fit", *outs, "--carrier", "hole", "--m-star", "0.5", "--json", cwd=diamond_run)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["inputs"]["m_star"] == {"value": 0.5, "unit": "m_e", "source": "given (--m-star)"}
    assert "masses" not in res and "m_d" not in res["inputs"]


def test_without_band_or_dos_m_star_is_still_required(tmp_path):
    d = tmp_path / "alone" / "sp"
    d.mkdir(parents=True)
    for f in STRAIN.glob("dia_sp_dp_x_*.out"):
        shutil.copy(f, d / f.name)
    r = _mace("tau", "fit", *sorted(p.name for p in d.glob("*.out")), "--carrier", "hole", cwd=d)
    assert r.returncode == 2
    assert "m* is required" in r.stderr and "BAND.DAT" in r.stderr and "--band FILE" in r.stderr


def test_the_gui_fit_route_takes_auto_and_returns_the_masses(diamond_run):
    from mace.gui import mechanics_routes as MR

    class St:
        root = diamond_run.parent.resolve()

        def resolve(self, rel, must_exist=True):
            return (self.root / rel).resolve()

        def rel(self, p):
            return p.resolve().relative_to(self.root).as_posix()

    outs = [f"sp/{p.name}" for p in sorted(diamond_run.glob("dia_sp_dp_x_*.out"))]
    r = MR.fit(St(), {"outputs": outs, "carrier": "hole", "m_star": "auto"})
    m = r["masses"]
    assert m["m_star"] == pytest.approx(r["result"]["inputs"]["m_star"]["value"])
    assert "dia_sp_band.BAND.DAT" in m["sources"]["m_star"] and "Γ→X" in m["report"]
    assert r["result"]["inputs"]["m_d"]["value"] == pytest.approx(m["m_dos"])
    r = MR.fit(St(), {"outputs": outs, "carrier": "hole", "m_star": 0.5})
    assert "masses" not in r and r["result"]["inputs"]["m_star"]["source"] == "given"
