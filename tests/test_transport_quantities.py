"""Transport plots of every quantity (1.1.6): mace.utils.transport_plot, shared by
the GUI's transport plot and `mace plotting --transport`.

Real BOLTZTRA runs: the example suite's diamond (3D, spin-polarized, written by
MACE as .DAT), graphene (a slab) and polyacetylene (a polymer), and the corpus's
closed-shell afi run (test/, read only; those checks skip without it).

* SIGMAS and TDF draw: an open-shell SIGMAS.dat's header ("conductivity *
  Seebeck coefficient") had it read as a per-spin SEEBECK.dat, which gave no
  rows; an open-shell TDF.dat's beta block was drawn after its alpha block on
  one line, and a slab's or polymer's TDF.dat (4 or 2 columns) gave no rows;
* the run's files are found beside any one of them, in any case (MACE writes
  .DAT, and the GUI found only .dat siblings);
* PF, κₑ, ZT, L, σ/τ and PF/τ at one grid point equal a hand calculation from
  the files' own text; ZT's peak is the database's; files on different grids
  are refused; τ comes from the .out, else the .d3 (CRYSTAL's 10 fs without
  RELAXTIM);
* mace plotting --transport draws them (a mace_cli subprocess), and the GUI's
  Copy CLI Command is a command its parser takes.
"""
import json
import lzma
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.utils import transport_plot as T

SUITE = REPO_ROOT / "examples" / "suite"
RUNS = {"diamond": "2-dense-3d/diamond/TRANSPORT", "graphene": "3-slab-2d/graphene/TRANSPORT",
        "polyacetylene": "4-polymer-1d/polyacetylene/TRANSPORT"}
AFI = TEST_DATA / "TRANSPORT" / "C1-RCSR-afi_optimized_TRANSPORT"
corpus = pytest.mark.skipif(not (TEST_DATA / "TRANSPORT").is_dir(), reason="test/ corpus not present (gitignored)")


def unpack(name, dst: Path, skip=()) -> Path:
    """The example run's files (decompressed) in dst; the path of its SEEBECK file."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((SUITE / RUNS[name]).iterdir()):
        out = f.name[:-3] if f.name.endswith(".xz") else f.name
        if any(out.endswith(s) for s in skip):
            continue
        (dst / out).write_bytes(lzma.decompress(f.read_bytes()) if f.name.endswith(".xz") else f.read_bytes())
    return dst / f"{name}_transport.SEEBECK.DAT"


@pytest.fixture(scope="module")
def dia(tmp_path_factory):
    return unpack("diamond", tmp_path_factory.mktemp("dia"))


def raw_rows(path: Path, T_k: float, mu: float):
    """The data lines of (T, μ) in a file, read with nothing but split (the alpha and beta lines of an open-shell file)."""
    out = []
    for line in path.read_text().splitlines():
        v = line.split()
        if line.startswith("#") or len(v) < 4:
            continue
        if float(v[0]) == mu and float(v[1]) == T_k:
            out.append([float(x) for x in v])
    return out


# ------------------------------------------------------------------ the two files that did not plot
def test_open_shell_sigmas_has_rows_and_is_minus_sigma_times_s(dia):
    from mace.utils.property_extractor import CrystalPropertyExtractor as C
    sigmas = dia.with_name("diamond_transport.SIGMAS.DAT")
    assert sigmas.read_text().startswith("# 1 [Alpha electrical conductivity * Seebeck")
    unit, rows, axes = C._parse_transport_dat_tensor(sigmas, with_axes=True)
    assert unit == "A/m/K" and axes == ("xx", "yy", "zz") and len(rows) == 15 * 401
    # CRYSTAL writes -sigma*S (the sign of Eq. 14.2): alpha + beta sigma times S, at every point of the run
    s = {(r[0], r[1]): r[3] for r in C._parse_transport_dat_tensor(dia)[1]}
    g = {(r[0], r[1]): r[3] for r in C._parse_transport_dat_tensor(dia.with_name("diamond_transport.SIGMA.DAT"))[1]}
    checked = 0
    for T_k, mu, _n, x in rows:
        for i in range(3):
            prod = g[(T_k, mu)][i] * s[(T_k, mu)][i]
            if abs(prod) > 1e-3:
                assert x[i] == pytest.approx(-prod, rel=2e-6)
                checked += 1
    assert checked > 10000
    d = T.quantity_data(sigmas)
    assert d["quantity"] == "SIGMAS" and len(d["blocks"]) == 15 and "−σ·S" in d["label"]


@corpus
def test_closed_shell_sigmas_is_read_as_before():
    from mace.utils.property_extractor import CrystalPropertyExtractor as C
    p = Path(str(AFI) + ".SIGMAS.dat")
    rows = C._parse_transport_dat_tensor(p)[1]
    first = raw_rows(p, 300.0, -3.4)[0]
    assert len(rows) == 3 * 711 and rows[0] == (300.0, -3.4, first[2], (first[3], first[6], first[8]))


def test_open_shell_tdf_sums_alpha_and_beta_on_one_energy_axis(dia):
    p = dia.with_name("diamond_transport.TDF.DAT")
    d = T.quantity_data(p)
    b = d["blocks"][0]
    assert d["x"] == "energy" and b["T"] is None and d["components"] == ["xx", "yy", "zz"]
    assert len(b["mu"]) == 1040 and all(x < y for x, y in zip(b["mu"], b["mu"][1:]))
    # an energy of the valence band where the TDF is not zero (the run's TDFRANGE is E_F -5..+5 eV, E_F -2.12 eV)
    e = next(x for x, v in zip(b["mu"], b["xx"]) if -4.0 < x < -2.2 and v)
    i = b["mu"].index(e)
    lines = [ln.split() for ln in p.read_text().splitlines() if not ln.startswith("#") and ln.strip()
             and abs(float(ln.split()[0]) - e) < 1e-6]
    assert len(lines) == 2                                 # an alpha and a beta line
    assert b["xx"][i] == pytest.approx(float(lines[0][1]) + float(lines[1][1]), rel=1e-12)
    assert b["zz"][i] == pytest.approx(float(lines[0][6]) + float(lines[1][6]), rel=1e-12)
    assert "alpha and beta" in d["notes"][0]


@pytest.mark.parametrize("name, axes", [("graphene", ["xx", "yy"]), ("polyacetylene", ["xx"])])
def test_slab_and_polymer_tdf_and_derived_quantities(tmp_path, name, axes):
    seebeck = unpack(name, tmp_path / name)
    d = T.quantity_data(seebeck.with_name(f"{name}_transport.TDF.DAT"))
    assert d["components"] == axes and len(d["blocks"][0]["mu"]) == 1040
    for q in T.QUANTITIES:
        d = T.quantity_data(seebeck, q, "1.5" if q == "zt" else None)
        assert d["components"] == axes and d["dim"] == len(axes), q
        assert T.traces(d, "all" if len(axes) > 1 else "avg"), q


@corpus
def test_a_tdf_with_a_header_alone_says_so():
    p = Path(str(AFI) + ".TDF.dat")
    with pytest.raises(T.TransportError, match="header and no data rows"):
        T.quantity_data(p)
    avail = T.quantity_data(Path(str(AFI) + ".SEEBECK.dat"))["available"]
    assert "no data rows" in avail["TDF"] and avail["zt"] is None


# ------------------------------------------------------------------ the run's files beside any one
def test_siblings_in_any_case_and_only_the_runs(tmp_path):
    seebeck = unpack("diamond", tmp_path / "run")
    (seebeck.parent / "other_transport.SIGMA.dat").write_text("# 0 [x]\n")
    shutil.copy(seebeck.with_name("diamond_transport.KAPPA.DAT"), seebeck.with_name("diamond_transport.KAPPA.dat.bak"))
    for kind in T.FILE_KINDS:
        f = seebeck.with_name(f"diamond_transport.{kind}.DAT")
        assert {k: p.name for k, p in T.run_files(f).items()} == {
            k: f"diamond_transport.{k}.DAT" for k in T.FILE_KINDS}
    # mixed case, and a filter on what may be read (the GUI's served folder)
    seebeck.with_name("diamond_transport.KAPPA.DAT").rename(seebeck.with_name("diamond_transport.kappa.dat"))
    files = T.run_files(seebeck, inside=lambda p: "SIGMAS" not in p.name)
    assert files["KAPPA"].name == "diamond_transport.kappa.dat" and "SIGMAS" not in files
    from mace.gui import results
    assert results._transport_siblings(seebeck, "diamond_transport")["KAPPA"] == "diamond_transport.kappa.dat"


def test_availability_greys_what_the_files_do_not_allow(tmp_path):
    seebeck = unpack("diamond", tmp_path / "nok", skip=(".KAPPA.DAT", ".out"))
    d = T.quantity_data(seebeck)
    av = d["available"]
    for q in ("KAPPA", "zt", "lorenz", "kappa_e"):
        assert "KAPPA.dat" in av[q], q
    for q in ("SEEBECK", "SIGMA", "SIGMAS", "TDF", "pf", "sigma_tau", "pf_tau"):
        assert av[q] is None, q                            # τ from the .d3: CRYSTAL's 10 fs (no RELAXTIM)
    assert d["tau_run_fs"] == 10.0 and "default" in d["tau_source"]
    with pytest.raises(T.TransportError, match="KAPPA.dat"):
        T.quantity_data(seebeck, "zt")
    seebeck.with_name("diamond_transport.d3").unlink()
    av = T.quantity_data(seebeck)["available"]
    assert "relaxation time" in av["sigma_tau"] and "relaxation time" in av["pf_tau"] and av["pf"] is None


# ------------------------------------------------------------------ the numbers
def test_derived_values_at_one_point_are_the_hand_calculation(tmp_path):
    """Diamond, 300 K, μ = -3.13 eV (1 eV into the valence band: E_F -2.12 eV), xx; τ written into the run's .d3
    as RELAXTIM 3387 (no .out)."""
    seebeck = unpack("diamond", tmp_path / "d", skip=(".out",))
    d3 = seebeck.with_name("diamond_transport.d3")
    d3.write_text(d3.read_text().replace("TDFRANGE", "RELAXTIM\n3387.0\nTDFRANGE"))
    T_k, mu, tau = 300.0, -3.13, 3387e-15
    S = raw_rows(seebeck, T_k, mu)[0][3]                                    # one block: not spin dependent
    sig = sum(r[3] for r in raw_rows(seebeck.with_name("diamond_transport.SIGMA.DAT"), T_k, mu))   # alpha + beta
    k0 = sum(r[3] for r in raw_rows(seebeck.with_name("diamond_transport.KAPPA.DAT"), T_k, mu))
    pf = S * S * sig
    ke = k0 - T_k * pf
    hand = {"SEEBECK": S * 1e6, "SIGMA": sig, "KAPPA": k0, "pf": pf * 1e4, "kappa_e": ke, "zt": pf * T_k / ke,
            "lorenz": ke / (sig * T_k) * 1e8, "sigma_tau": sig / tau * 1e-18, "pf_tau": pf / tau * 1e-11}
    for q, want in hand.items():
        d = T.quantity_data(seebeck, q)
        b = next(b for b in d["blocks"] if b["T"] == T_k)
        assert b["xx"][b["mu"].index(mu)] == pytest.approx(want, rel=1e-12), q
    d = T.quantity_data(seebeck, "zt", "1.5")
    b = next(b for b in d["blocks"] if b["T"] == T_k)
    i = b["mu"].index(mu)
    assert b["xx"][i] == pytest.approx(pf * T_k / (ke + 1.5), rel=1e-12)
    assert b["bound"]["xx"][i] == pytest.approx(pf * T_k / ke, rel=1e-12)
    assert d["tau_fs"] == 3387.0 and d["tau_source"] == "RELAXTIM, diamond_transport.d3"
    # no .out: the cell volume from the header's V(cm^3), converted from bohr³ (4 digits)
    assert b["volume_cm3"] == pytest.approx(11.13575e-24, rel=2e-4)
    # in a band (a degenerate gas) L is Sommerfeld's: at 300 K, at the Standard k mesh (NEWK IS 48; with the
    # SCF's IS 12 L swung from 0.02 to 24 x 1e-8 across the valence band), the median over the band is
    # Sommerfeld's within 5%; single points still scatter by up to 30% (0.01 eV TDF bins)
    assert T.L0 == pytest.approx(2.443e-8, rel=1e-3)
    lz = next(b for b in T.quantity_data(seebeck, "lorenz")["blocks"] if b["T"] == T_k)
    band = sorted(v for x, v in zip(lz["mu"], lz["xx"]) if -4.0 <= x <= -2.6 and v is not None)
    assert len(band) > 100 and band[len(band) // 2] == pytest.approx(T.L0 * 1e8, rel=0.05)
    # a τ given rescales σ and κ₀ (PF, κₑ, the total ZT); L, S and ZTₑ do not move
    d2 = T.quantity_data(seebeck, "pf", tau_fs="100")
    b2 = next(b for b in d2["blocks"] if b["T"] == T_k)
    assert b2["xx"][i] == pytest.approx(hand["pf"] * 100e-15 / tau, rel=1e-12) and d2["tau_fs"] == 100.0


def test_zt_peak_is_the_databases(dia):
    from mace.gui import results
    d = T.quantity_data(dia, "zt", "1.5")
    z = results.transport_zt(dia, "1.5", None)
    assert d["zt_total_max"] == z["zt_total_max"] and d["zt_e_max"] == z["zt_e_max"]
    assert not d["upper_bound"] and d["kappa_l_label"] == "κ<sub>L</sub> = 1.5 W m<sup>−1</sup> K<sup>−1</sup>"
    e = T.quantity_data(dia, "zt")
    assert e["upper_bound"] and e["zt_total_max"] is None and e["zt_e_max"] == z["zt_e_max"]
    # the axis spans the total, not the ZTₑ bound
    curves = T.traces(d, "avg", 100)
    top = T.y_range(d, curves)[1]
    assert top == pytest.approx(1.08 * max(v for v in curves[0]["y"] if v is not None))
    assert max(v for v in curves[1]["y"] if v is not None) > top


def test_files_on_different_grids_are_refused(tmp_path):
    seebeck = unpack("diamond", tmp_path / "g")
    sigma = seebeck.with_name("diamond_transport.SIGMA.DAT")
    keep = [ln for ln in sigma.read_text().splitlines() if ln.startswith("#") or float(ln.split()[1]) != 800.0]
    sigma.write_text("\n".join(keep) + "\n")
    for q in ("pf", "zt", "lorenz"):
        with pytest.raises(T.TransportError, match="not on the same chemical potential and temperature grid"):
            T.quantity_data(seebeck, q)
    assert T.quantity_data(sigma)["blocks"]                  # each file still draws by itself


def test_traces_temperatures_components_and_carriers(dia):
    d = T.quantity_data(dia, "SEEBECK")
    assert [c["name"] for c in T.traces(d)] == ["(xx + yy + zz) / 3"]
    assert [c["name"] for c in T.traces(d, "all", 300)] == ["xx", "yy", "zz"]
    assert [c["name"] for c in T.traces(d, "zz", "all")] == [f"{b['T']:g} K" for b in d["blocks"]]
    with pytest.raises(T.TransportError, match="one component"):
        T.traces(d, "all", "all")
    with pytest.raises(T.TransportError, match="no 333 K block"):
        T.traces(d, "avg", 333)
    c = T.traces(d, "avg", 300, "carriers")
    assert [x["name"] for x in c] == ["(xx + yy + zz) / 3, p-type"]    # diamond's μ range is in its valence band
    b = next(b for b in d["blocks"] if b["T"] == 300.0)
    i = next(i for i, n in enumerate(b["carriers"]) if n > 0)
    # per cm³ of the cell: the .out's CELL VOLUME (A.U.) 75.147743 bohr³ = 11.136 Å³ (its PRIMITIVE CELL VOLUME);
    # CRYSTAL's V(cm^3) headers print 7.515E-23 (bohr³ × 1e-24; the beta headers 7.515E-47)
    assert b["volume_cm3"] == pytest.approx(11.13575e-24, rel=1e-6)
    assert c[0]["x"][0] == pytest.approx(b["carriers"][i] / b["volume_cm3"])
    assert T._volumes(dia.with_name("diamond_transport.SIGMA.DAT"))[300.0] == 7.515e-23
    # the carriers are SIGMA.dat's, alpha + beta (SEEBECK.dat's N is one spin's)
    sig = sum(r[2] for r in raw_rows(dia.with_name("diamond_transport.SIGMA.DAT"), 300.0, b["mu"][0]))
    assert b["carriers"][0] == pytest.approx(sig)
    lz = T.traces(T.quantity_data(dia, "lorenz"), "avg", 300)
    assert lz[-1]["role"] == "reference" and lz[-1]["y"][0] == pytest.approx(2.443, rel=1e-3)


# ------------------------------------------------------------------ mace plotting --transport and the GUI's command
def _cli(argv, cwd):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *argv], cwd=str(cwd),
                          capture_output=True, text=True, timeout=600)


def test_mace_plotting_transport_subprocess(tmp_path):
    pytest.importorskip("matplotlib")
    seebeck = unpack("diamond", tmp_path / "cli")
    run = seebeck.parent
    p = _cli(["plotting", "--transport", "diamond_transport.SIGMAS.DAT", "--quantity", "zt", "--kappa-lattice", "1.5",
              "--temperature", "300", "--format", "png"], run)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert (run / "diamond_transport.ZT_300K.png").stat().st_size > 10000
    assert "κ_L = 1.5 W m⁻¹ K⁻¹" in p.stdout and "τ = 10 fs (RELAXTIM, diamond_transport.out)" in p.stdout
    # the file's own quantity (the TDF, against energy) and the carrier axis, into -o
    p = _cli(["plotting", "--transport", "diamond_transport.TDF.DAT", "diamond_transport.SIGMAS.DAT", "-o", "figs",
              "--format", "svg"], run)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert {f.name for f in (run / "figs").iterdir()} == {"diamond_transport.TDF.svg", "diamond_transport.SIGMAS_100K.svg"}
    p = _cli(["plotting", "--transport", "diamond_transport.SEEBECK.DAT", "--quantity", "pf", "--x-axis", "carriers",
              "--temperature", "all", "--component", "xx", "--format", "png"], run)
    assert p.returncode == 0 and (run / "diamond_transport.PF_allT_xx_carriers.png").is_file(), p.stdout[-2000:]
    # a quantity the files do not allow: the reason, exit 1
    seebeck.with_name("diamond_transport.KAPPA.DAT").unlink()
    p = _cli(["plotting", "--transport", "diamond_transport.SEEBECK.DAT", "--quantity", "lorenz"], run)
    assert p.returncode == 1 and "KAPPA.dat" in p.stdout + p.stderr


def test_gui_publication_flags_are_what_the_parser_takes():
    from mace.gui import plots
    from mace.plotting.main import create_parser
    get = {"quantity": "kappa_e", "component": "yy", "temperature": "300.0", "x": "carriers", "kappa_l": "", "tau_fs": "44"}.get
    flags, files = plots.transport_args(get)
    assert flags == ["--quantity=kappa-e", "--component=yy", "--temperature=300", "--x-axis=carriers", "--tau-fs=44"] and not files
    args = create_parser().parse_args(["--transport", "x.SEEBECK.DAT", *flags])
    assert (args.quantity, args.component, args.temperature, args.x_axis, args.tau_fs) == ("kappa-e", "yy", "300", "carriers", "44")
    for bad in ({"component": "xy"}, {"temperature": "-3"}, {"x": "n"}, {"quantity": "zz"}, {"kappa_l": "nope.csv"}):
        with pytest.raises(ValueError):
            plots.transport_args(bad.get)


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_copied_transport_command_is_what_mace_plotting_reads(tmp_path):
    from mace.plotting.main import create_parser
    js = REPO_ROOT / "mace" / "gui" / "static" / "js" / "docs" / "clicmd.js"
    cases = [{"kind": "transport", "file": "my run.SIGMAS.DAT", "look": {"style": "publication"},
              "transport": {"quantity": "zt", "component": "avg", "temperature": 300, "x": "mu", "kappa_l": "kl.csv", "tau_fs": 44}},
             {"kind": "transport", "file": "r.TDF.DAT", "look": {},
              "transport": {"quantity": "TDF", "component": "all", "temperature": None, "x": None, "kappa_l": "", "tau_fs": None}},
             {"kind": "transport", "file": "r.SIGMA.DAT", "look": {},
              "transport": {"quantity": "sigma_tau", "component": "zz", "temperature": "all", "x": "carriers", "kappa_l": "", "tau_fs": None}}]
    script = tmp_path / "run.mjs"
    script.write_text(f"import {{ cliCommand, TRANSPORT_CLI }} from {json.dumps(js.as_uri())};\n"
                      f"console.log(JSON.stringify({{cmds: {json.dumps(cases)}.map((c) => cliCommand(c)), names: TRANSPORT_CLI}}));\n")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["names"] == T.CLI_OF                     # the JS names are mace plotting's --quantity choices
    want = [("zt", None, "300", "mu", "kl.csv", "44", "publication"), ("tdf", "all", None, None, None, None, None),
            ("sigma-tau", "zz", "all", "carriers", None, None, None)]
    for case, text, w in zip(cases, out["cmds"], want):
        words = shlex.split(text)
        assert words[:4] == ["mace", "plotting", "--transport", case["file"]]
        a = create_parser().parse_args(words[2:])
        assert (a.quantity, a.component, a.temperature, a.x_axis, a.kappa_lattice, a.tau_fs, a.style) == w, text


# ------------------------------------------------------------------ κₑ below the files' precision
def _old_lorenz(seebeck, T_k):
    """L/1e-8 at every point the rule before the guard kept (κₑ > 0), from the module's own tensors."""
    files = T.run_files(seebeck)
    ts = {k: T.read_tensor(files[k]) for k in ("SEEBECK", "SIGMA", "KAPPA")}
    b = [next(x for x in ts[k]["blocks"] if x["T"] == T_k) for k in ("SEEBECK", "SIGMA", "KAPPA")]
    out = {}
    for i, mu in enumerate(b[0]["mu"]):
        s = sum(b[0]["diag"][a][i] for a in ("xx", "yy", "zz")) / 3
        g = abs(sum(b[1]["diag"][a][i] for a in ("xx", "yy", "zz")) / 3)
        k = abs(sum(b[2]["diag"][a][i] for a in ("xx", "yy", "zz")) / 3)
        ke = k - T_k * s * s * g
        out[mu] = (ke, k, ke / (g * T_k) * 1e8 if ke > 0 else None)
    return out


def test_kappa_e_below_the_files_precision_is_masked(dia):
    from mace.utils import transport_zt as Z
    # the rule: 8 printed digits put |κ₀ − TσS²| within 2e-7 κ₀ of 0; every negative κₑ of the run (noise by
    # definition) is inside 1e-7 κ₀, and the guard is 2e-6 κ₀
    assert Z.KAPPA_E_EPS == 2e-6
    d = T.quantity_data(dia, "lorenz")
    # at the Standard k mesh no κₑ is below the files' precision (the SCF mesh's spiky TDF gave 16 such points;
    # the masking itself: test_total_zt_at_a_masked_point_only_where_kappa_l_dominates)
    assert d.get("n_masked", 0) == 0 and not any("below the files' precision" in n for n in d["notes"])
    zt = T.quantity_data(dia, "zt")
    pf = T.quantity_data(dia, "pf")
    masked = 0
    for b, bz, bp in zip(d["blocks"], zt["blocks"], pf["blocks"]):
        old = _old_lorenz(dia, b["T"])
        for i, mu in enumerate(b["mu"]):
            ke, k, l_old = old[mu]
            if bp["avg"][i] is not None and k and abs(ke) < 2e-6 * k:
                masked += 1                                   # a gap in κₑ, L and ZTₑ; PF is still drawn
                assert b["avg"][i] is None and bz["avg"][i] is None
            elif b["avg"][i] is not None:
                # L unchanged at every point the guard keeps
                assert b["avg"][i] == pytest.approx(l_old, rel=1e-12)
            else:
                assert l_old is None or bp["avg"][i] is None
    assert masked == 0
    # the band point of the hand calculation is kept
    b300 = next(b for b in d["blocks"] if b["T"] == 300.0)
    assert b300["xx"][b300["mu"].index(-3.13)] == pytest.approx(2.65981, rel=1e-4)


def test_the_databases_zt_peak_is_not_a_noise_point(dia):
    from mace.utils import transport_zt as Z
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(enable_tracking=False)
    x.kappa_lattice, x.tau_fs = Z.parse_kappa_lattice("1.5"), None
    props = x.extract_all_properties(dia.with_name("diamond_transport.out"))
    assert props["transport_data_available"] is True        # MACE's .DAT files (the extractor looked for .dat)
    tr = props["transport"]
    assert tr["zt_points_kappa_e_below_precision"] == 0
    assert "transport_mesh_note" not in props and tr["mesh_note"] is None      # the Standard mesh
    # before the guard the peak was 1.6e9 at 100 K, μ -2.82 eV, where κₑ was 6e-10 of κ₀
    peak = props["transport_zt_electronic_max_avg"]
    assert peak < 1e6 and tr["zt_electronic_max_at"] != {"temperature_k": 100.0, "mu_ev": -2.82}
    assert tr["kappa_electronic_at_zt_max_w_per_m_k"] >= Z.KAPPA_E_EPS * tr["kappa0_at_zt_max_w_per_m_k"]
    d = T.quantity_data(dia, "zt", "1.5")
    assert d["zt_e_max"]["value"] == peak and d["zt_total_max"]["value"] == props["transport_zt_total_max_avg"]
    for a, e in tr["per_axis"].items():
        assert e["kappa_electronic_at_zt_max_w_per_m_k"] >= Z.KAPPA_E_EPS * e["kappa0_at_zt_max_w_per_m_k"], a


def test_total_zt_at_a_masked_point_only_where_kappa_l_dominates():
    from mace.utils import transport_zt as Z
    # one point: κ₀ = 1000, TσS² = 1000 − 1e-4 (κₑ = 1e-7 κ₀: noise); S = 1e-4 V/K, T = 300 K
    T_k, S = 300.0, 1e-4
    sig = (1000 - 1e-4) / (T_k * S * S)
    rows = lambda v: [(T_k, 0.0, 0.0, v)]  # noqa: E731
    big = Z.zt_points(rows(S), rows(sig), rows(1000.0), Z.parse_kappa_lattice("1"))     # uncertainty 2e-3 ≤ 1% of 1
    assert big["blocks"][T_k]["zt_e"] == [None] and big["n_masked"] == 1
    assert big["blocks"][T_k]["zt_total"][0] == pytest.approx((1000 - 1e-4) / (1e-4 + 1), rel=1e-6)
    small = Z.zt_points(rows(S), rows(sig), rows(1000.0), Z.parse_kappa_lattice("0.1"))  # 2e-3 > 1% of 0.1
    assert small["blocks"][T_k]["zt_total"] == [None] and small["zt_total_max"] is None


def test_a_low_dimensional_run_at_the_standard_mesh_has_no_unreliable_note(tmp_path):
    """The slab's and polymer's "κ₀ nearly equals TσS² ... not reliable" note fired on their runs at the SCF's
    k mesh (polyacetylene IS 18, graphene IS 12: L at 0.01-0.5 L₀); at the Standard mesh (IS 284 and 197) L
    is Sommerfeld's within 5% and the note is gone."""
    for name in ("polyacetylene", "graphene"):
        seebeck = unpack(name, tmp_path / name)
        d = T.quantity_data(seebeck, "lorenz")
        assert not any(n.startswith("in this") for n in d["notes"]), name


def test_a_3d_run_has_no_low_dimensional_note(dia):
    assert not any(n.startswith("in this") for n in T.quantity_data(dia, "lorenz")["notes"])
