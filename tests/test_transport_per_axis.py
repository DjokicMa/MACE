"""Thermoelectric values per axis (B8, 1.1.6). PF = S²σ and ZTₑ of a tensor
averaged over its diagonal are no physical quantity of an anisotropic cell,
so each axis gets its own peak (S_aa, σ_aa, κ₀,aa alone), with the best axis;
the averaged value stays, named *_avg, with the average it is
(transport_tensor_average). A slab has xx and yy only.

The 3D numbers are checked against an independent numpy computation on the
real afi run (CRYSTAL23 manual pp. 314-315: S, σ and κ₀ per Cartesian
component). The slab case is the same run's data written in the manual's
2D layout (sigma_xx sigma_xy sigma_yy, p. 315) with a 2D .out: the corpus
has no 2D TRANSPORT run."""
import contextlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import TEST_DATA, REPO_ROOT
from mace.utils import transport_zt as Z
from mace.utils.property_extractor import CrystalPropertyExtractor

AFI = "C1-RCSR-afi_optimized_TRANSPORT"


def _afi_dir():
    d = TEST_DATA / "TRANSPORT"
    if not (d / f"{AFI}.out").is_file():
        pytest.skip("test/ data corpus not present (gitignored)")
    return d


def _extract(out, kappa=None):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    if kappa is not None:
        x.kappa_lattice = Z.parse_kappa_lattice(kappa)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(out))


def _numpy_peaks(d, stem, axes_idx):
    """ZTₑ and PF peaks per axis and of the averaged tensors, each with the
    extractor's rule (|σ| at least 1e-3 of that quantity's peak |σ|,
    κₑ = κ₀ − TσS² > 0)."""
    np = pytest.importorskip("numpy")

    def load(q, idx):
        a = np.loadtxt(d / f"{stem}.{q}.dat", comments="#")
        return a[:, 1], a[:, 3 + np.array(idx)]
    T, S = load("SEEBECK", axes_idx["S"])
    _, G = load("SIGMA", axes_idx["G"])
    _, K = load("KAPPA", axes_idx["G"])

    def peaks(s, g, k):
        floor = np.abs(g).max() * 1e-3
        ok = np.abs(g) >= floor
        pf = np.where(ok, s * s * np.abs(g), 0)
        ke = np.abs(k) - pf * T
        zt = np.where(ok & (ke > 0), pf * T / np.where(ke > 0, ke, 1), 0)
        return pf.max(), zt.max()
    per = [peaks(S[:, i], G[:, i], K[:, i]) for i in range(S.shape[1])]
    avg = peaks(S.mean(1), G.mean(1), K.mean(1))
    return per, avg


def test_afi_per_axis_against_numpy():
    d = _afi_dir()
    p = _extract(d / f"{AFI}.out")
    per, avg = _numpy_peaks(d, AFI, {"S": [0, 4, 8], "G": [0, 3, 5]})
    for (pf, zt), a in zip(per, ("xx", "yy", "zz")):
        assert p[f"transport_power_factor_max_{a}"] == pytest.approx(pf, rel=1e-9)
        assert p[f"transport_zt_electronic_max_{a}"] == pytest.approx(zt, rel=1e-9)
    # the review's case: one averaged ZTₑ 19.55 for an anisotropic cell whose
    # axes give 15.4 / 15.4 / 20.5 (by each axis's own σ cut)
    assert p["transport_zt_electronic_max_avg"] == pytest.approx(19.554, abs=1e-3)
    assert p["transport_zt_electronic_max_avg"] == pytest.approx(avg[1], rel=1e-9)
    assert p["transport_power_factor_max_avg"] == pytest.approx(avg[0], rel=1e-9)
    assert [round(p[f"transport_zt_electronic_max_{a}"], 2) for a in ("xx", "yy", "zz")] == [15.44, 15.44, 20.46]
    assert p["transport_zt_electronic_best_axis"] == "zz" and p["transport_power_factor_best_axis"] == "zz"
    assert p["transport_tensor_average"].startswith("(xx+yy+zz)/3")
    assert set(p["transport"]["per_axis"]) == {"xx", "yy", "zz"}
    for old in ("transport_zt_max", "transport_zt_electronic_max", "transport_power_factor_max"):
        assert old not in p


def test_afi_total_zt_per_axis_with_kappa_lattice():
    d = _afi_dir()
    p = _extract(d / f"{AFI}.out", kappa="1.0")
    rows = {q: CrystalPropertyExtractor._transport_tensor_rows(d / f"{AFI}.{q}.dat") for q in ("SEEBECK", "SIGMA", "KAPPA")}
    tau, scale, _ = Z.resolve_tau(Z.read_relaxation_time_fs((d / f"{AFI}.out").read_text()), None)
    for a in ("xx", "yy", "zz"):
        s, g, k = (CrystalPropertyExtractor._axis_rows(rows[q][1], rows[q][2], a) for q in ("SEEBECK", "SIGMA", "KAPPA"))
        want = Z.zt_points(s, g, k, Z.parse_kappa_lattice("1.0"), scale)["zt_total_max"]["value"]
        assert p[f"transport_zt_total_max_{a}"] == pytest.approx(want, rel=1e-12)
    assert p["transport_zt_total_best_axis"] == "zz"
    assert "transport_zt_total_max_avg" in p and "transport_zt_total_max" not in p
    assert set(p["transport_zt_total"]["per_axis"]) == {"xx", "yy", "zz"}


def _slab(tmp_path, layout):
    """The afi run as a slab: a slab's .out, and .dat files in the manual's 2D
    layout ('2d': S_xx S_xy S_yx S_yy, sigma_xx sigma_xy sigma_yy) or the 3D
    one ('3d': zz columns still there)."""
    d = _afi_dir()
    # a properties run prints no DIMENSIONALITY line: a slab's primitive cell
    # has the formal C = 500 A (as the corpus's slab BAND outputs print it)
    out = (d / f"{AFI}.out").read_text()
    cell = "    6.79092    6.79092    4.16563     90.0000   90.0000  120.0000     166.36792"
    assert cell in out
    out = out.replace(cell, "    6.79092    6.79092  500.00000     90.0000   90.0000  120.0000      39.93820")
    stem = "slab_TRANSPORT"
    (tmp_path / f"{stem}.out").write_text(out)
    keep = {"SEEBECK": [0, 1, 3, 4], "SIGMA": [0, 1, 3], "KAPPA": [0, 1, 3]}
    for q in ("SEEBECK", "SIGMA", "KAPPA"):
        src = (d / f"{AFI}.{q}.dat").read_text().splitlines()
        if layout == "3d":
            lines = src
        else:
            lines = []
            for line in src:
                if line.lstrip().startswith("#"):
                    if "Mu(eV)" in line:
                        parts = line.split()
                        cols = parts[4:]
                        line = " ".join(parts[:4] + [cols[i] for i in keep[q]])
                    lines.append(line)
                    continue
                v = line.split()
                lines.append("  ".join(v[:3] + [v[3 + i] for i in keep[q]]))
        (tmp_path / f"{stem}.{q}.dat").write_text("\n".join(lines) + "\n")
    return tmp_path / f"{stem}.out"


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_a_slab_is_in_plane(tmp_path, layout):
    out = _slab(tmp_path, layout)
    p = _extract(out)
    afi = _extract(_afi_dir() / f"{AFI}.out")
    for a in ("xx", "yy"):
        assert p[f"transport_zt_electronic_max_{a}"] == pytest.approx(afi[f"transport_zt_electronic_max_{a}"], rel=1e-9)
        assert p[f"transport_power_factor_max_{a}"] == pytest.approx(afi[f"transport_power_factor_max_{a}"], rel=1e-9)
    assert not [k for k in p if k.endswith("_zz")]
    assert p["transport_tensor_average"].startswith("(xx+yy)/2, in plane")
    # the average is the in-plane one: the vacuum direction no longer dilutes it
    per, avg = _numpy_peaks(_afi_dir(), AFI, {"S": [0, 4], "G": [0, 3]})
    assert p["transport_zt_electronic_max_avg"] == pytest.approx(avg[1], rel=1e-9)
    assert p["transport_power_factor_max_avg"] == pytest.approx(avg[0], rel=1e-9)
    assert p["transport_zt_electronic_best_axis"] in ("xx", "yy")


def test_the_2d_column_layout_is_read_by_name():
    lay = CrystalPropertyExtractor._diagonal_layout
    assert lay(["sigma_xx", "sigma_xy", "sigma_yy"], 3) == ((0, 2), ("xx", "yy"))
    assert lay(["S_xx", "S_xy", "S_yx", "S_yy"], 4) == ((0, 3), ("xx", "yy"))
    assert lay(["S_xx", "S_xy", "S_xz", "S_yx", "S_yy", "S_yz", "S_zx", "S_zy", "S_zz"], 9) == ((0, 4, 8), ("xx", "yy", "zz"))
    assert lay(None, 6) == ((0, 3, 5), ("xx", "yy", "zz"))
    assert lay(None, 3) == ((0, 2), ("xx", "yy"))   # the manual's 2D example, p. 315


def test_mace_analyze_prints_the_axes(tmp_path):
    d = _afi_dir()
    run = tmp_path / "run"
    run.mkdir()
    for f in d.glob(f"{AFI}.*"):
        if f.suffix in (".out", ".dat"):
            shutil.copyfile(f, run / f.name)
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        str(run), "--db-path", str(tmp_path / "m.db"), "--kappa-lattice", "1.0"],
                       capture_output=True, text=True, cwd=str(tmp_path), timeout=600, stdin=subprocess.DEVNULL,
                       env={**__import__("os").environ, "NO_COLOR": "1", "MACE_NO_BANNER": "1"})
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "ZTₑ (no lattice κ) peak: xx 15.44, yy 15.44, zz 20.46 (best zz); of the averaged tensors 19.55" in p.stdout
    assert "total ZT peak: xx" in p.stdout and "(best zz); of the averaged tensors" in p.stdout
