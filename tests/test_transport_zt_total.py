"""Total ZT of a TRANSPORT run with a user-supplied lattice kappa (1.1.6).

ZT_total = S^2 sigma T / (kappa_e + kappa_L), kappa_e = kappa0 - T sigma S^2.
SIGMA.dat and KAPPA.dat are sigma and kappa0 at the run's RELAXTIM tau
(CRYSTAL23 manual Eq. 14.4: the transport distribution function carries tau;
the files are "in SI units"), so ZT_total is evaluated at the .out's tau, or
at a tau the user gives with sigma and kappa0 scaled by tau / tau_run.

Hand-computed values on small synthetic rows; the real corpus runs
(test/TRANSPORT, read only, skipped without it) for the parser, the
extractor, the CLI as a subprocess and the GUI's numbers.
"""
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA, find_data
from mace.utils import transport_zt as Z
from mace.utils.property_extractor import CrystalPropertyExtractor

BCU = "TRANSPORT/C1-RCSR-bcu-f_optimized_TRANSPORT"
RUNS = ("afi", "ana", "ato", "bcu-f")


# ------------------------------------------------------------------ kappa_L parsing
def test_constant_value():
    k = Z.parse_kappa_lattice("1.5")
    assert k.kind == "constant" and k.at(300.0) == 1.5 and k.at(1e4) == 1.5
    assert k.source() == "constant 1.5 W/(m·K)"
    assert Z.parse_kappa_lattice(0.25).at(10.0) == 0.25


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "", "  "])
def test_constant_refusals(bad):
    with pytest.raises(Z.KappaLatticeError):
        Z.parse_kappa_lattice(bad)


def test_neither_number_nor_file(tmp_path):
    with pytest.raises(Z.KappaLatticeError, match="neither a number nor a file"):
        Z.parse_kappa_lattice(str(tmp_path / "missing.csv"))


def _csv(tmp_path, text, name="kl.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_table_interpolates_linearly(tmp_path):
    # unsorted, with a comment and a header: sorted by T
    p = _csv(tmp_path, "# Si, measured\nT,kappa_L\n500,1.0\n300,2.0\n700,0.5\n")
    k = Z.parse_kappa_lattice(str(p))
    assert k.kind == "table" and k.temps == [300.0, 500.0, 700.0]
    assert k.at(300.0) == 2.0 and k.at(700.0) == 0.5 and k.at(500.0) == 1.0
    assert k.at(400.0) == pytest.approx(1.5, abs=1e-15)        # halfway 2.0 -> 1.0
    assert k.at(600.0) == pytest.approx(0.75, abs=1e-15)       # halfway 1.0 -> 0.5
    assert k.at(350.0) == pytest.approx(1.75, abs=1e-15)
    assert k.at(299.9999996) == 2.0 and k.at(700.0000004) == 0.5   # an end within 1e-6 K covers it
    src = k.source()
    assert str(p.resolve()) in src and "3 points" in src and "300-700 K" in src and "linear" in src
    assert "sha256" in src


def test_table_relative_to_base(tmp_path):
    _csv(tmp_path, "300,1\n400,2\n")
    assert Z.parse_kappa_lattice("kl.csv", base=tmp_path).at(350.0) == pytest.approx(1.5)


@pytest.mark.parametrize("T", [299.0, 250.0, 700.5, 1000.0])
def test_table_refuses_out_of_range(tmp_path, T):
    k = Z.parse_kappa_lattice(str(_csv(tmp_path, "300,2.0\n500,1.0\n700,0.5\n")))
    assert not k.covers(T)
    with pytest.raises(Z.KappaLatticeError) as e:
        k.at(T)
    msg = str(e.value)
    assert "300-700 K" in msg and "outside" in msg and "does not extrapolate" in msg and "kl.csv" in msg


def test_single_row_table_covers_its_temperature_only(tmp_path):
    k = Z.parse_kappa_lattice(str(_csv(tmp_path, "T,kappa_L\n300,1.2\n")))
    assert k.at(300.0) == 1.2
    with pytest.raises(Z.KappaLatticeError, match="300 K only"):
        k.at(500.0)


@pytest.mark.parametrize("text,match", [
    ("300,1\n300,2\n", "appears twice"),
    ("300,1\n400\n", "two columns"),
    ("300,1,2\n", "two columns"),
    ("300,-1\n", "kappa_L must be positive"),
    ("300,0\n", "kappa_L must be positive"),
    ("-5,1\n", "temperature must be positive"),
    ("T,kappa_L\n300,abc\n", "not two numbers"),
    ("300,1\nfoo,bar\n", "not two numbers"),
    ("# nothing\nT,kappa_L\n", "no T,kappa_L rows"),
])
def test_table_refusals(tmp_path, text, match):
    with pytest.raises(Z.KappaLatticeError, match=match):
        Z.parse_kappa_lattice(str(_csv(tmp_path, text)))


# ------------------------------------------------------------------ tau
def test_relaxation_time_from_the_corpus_outs():
    for run in RUNS:
        out = find_data(f"TRANSPORT/C1-RCSR-{run}_optimized_TRANSPORT.out")
        assert Z.read_relaxation_time_fs(out.read_text(errors="ignore")) == 22.0
    assert Z.read_relaxation_time_fs("  RELAXATION TIME (fs):                 10.0000\n") == 10.0
    assert Z.read_relaxation_time_fs("no such line") is None


def test_tau_resolution():
    assert Z.resolve_tau(22.0, None)[:2] == (22.0, 1.0)
    assert Z.resolve_tau(22.0, 44.0)[:2] == (44.0, 2.0)
    with pytest.raises(Z.KappaLatticeError, match="relaxation time"):
        Z.resolve_tau(None, 10.0)
    for bad in ("0", "-3", "x", "nan"):
        with pytest.raises(Z.KappaLatticeError):
            Z.check_tau_fs(bad)
    assert Z.check_tau_fs("") is None and Z.check_tau_fs(None) is None and Z.check_tau_fs("15") == 15.0


# ------------------------------------------------------------------ the formula, by hand
def _rows(T, pts):
    """pts: [(mu, S, sigma, kappa0)] -> SEEBECK, SIGMA, KAPPA rows (T, mu, n, value)."""
    return ([(T, mu, 0.0, s) for mu, s, _, _ in pts], [(T, mu, 0.0, g) for mu, _, g, _ in pts],
            [(T, mu, 0.0, k) for mu, _, _, k in pts])


def test_formula_by_hand():
    # S = 2e-4 V/K, sigma = 1e5 S/m, T = 300 K: T sigma S^2 = 1.2 W/(m K);
    # kappa0 = 2.0 -> kappa_e = 0.8; ZT_e = 1.2 / 0.8 = 1.5;
    # kappa_L = 0.4 -> ZT_total = 1.2 / (0.8 + 0.4) = 1.0
    s, g, k = _rows(300.0, [(-0.1, 2e-4, 1e5, 2.0),
                            (0.0, 1e-4, 1e1, 2.0),       # sigma below 1e-3 of the peak: left out
                            (0.1, 2e-4, 1e5, 1.0)])      # kappa_e = 1.0 - 1.2 < 0: left out
    r = Z.zt_points(s, g, k, Z.parse_kappa_lattice("0.4"))
    b = r["blocks"][300.0]
    assert b["zt_e"][0] == pytest.approx(1.5, rel=1e-12) and b["zt_total"][0] == pytest.approx(1.0, rel=1e-12)
    assert b["zt_e"][1:] == [None, None] and b["zt_total"][1:] == [None, None]
    assert r["zt_total_max"]["value"] == pytest.approx(1.0) and r["zt_total_max"]["kappa_l"] == 0.4
    assert r["zt_total_max"]["kappa_e"] == pytest.approx(0.8)
    # tau doubled: sigma and kappa0 double, S does not:
    # 2.4 / (1.6 + 0.4) = 1.2, and ZT_e stays 1.5
    r2 = Z.zt_points(s, g, k, Z.parse_kappa_lattice("0.4"), scale=2.0)
    assert r2["blocks"][300.0]["zt_total"][0] == pytest.approx(1.2, rel=1e-12)
    assert r2["blocks"][300.0]["zt_e"][0] == pytest.approx(1.5, rel=1e-12)
    # no kappa_L: no total
    assert Z.zt_points(s, g, k)["blocks"][300.0]["zt_total"] is None


def test_formula_by_hand_with_a_table(tmp_path):
    # two temperatures; kappa_L(T) = 2.0 at 300 K, 1.0 at 500 K -> 1.5 at 400 K.
    # 400 K: S = 2e-4, sigma = 1e5: T sigma S^2 = 1.6; kappa0 = 2.4 -> kappa_e = 0.8;
    # ZT_total = 1.6 / (0.8 + 1.5) = 0.695652...
    kl = Z.parse_kappa_lattice(str(_csv(tmp_path, "300,2.0\n500,1.0\n")))
    s1, g1, k1 = _rows(300.0, [(0.0, 2e-4, 1e5, 2.0)])
    s2, g2, k2 = _rows(400.0, [(0.0, 2e-4, 1e5, 2.4)])
    s3, g3, k3 = _rows(600.0, [(0.0, 2e-4, 1e5, 9.0)])
    r = Z.zt_points(s1 + s2 + s3, g1 + g2 + g3, k1 + k2 + k3, kl)
    assert r["blocks"][300.0]["zt_total"][0] == pytest.approx(1.2 / (0.8 + 2.0), rel=1e-12)
    assert r["blocks"][400.0]["kappa_l"] == pytest.approx(1.5)
    assert r["blocks"][400.0]["zt_total"][0] == pytest.approx(1.6 / 2.3, rel=1e-12)
    # 600 K is outside the table: refused there, with the reason; the other temperatures stand
    assert r["blocks"][600.0]["zt_total"] is None and "600 K is outside" in r["blocks"][600.0]["refused"]
    assert r["blocks"][600.0]["zt_e"][0] is not None
    assert Z.uncovered(kl, r["temperatures"]) == [600.0]


# ------------------------------------------------------------------ the extractor on real runs
def _props(out, kappa_l=None, tau_fs=None):
    x = CrystalPropertyExtractor(enable_tracking=False)
    x.kappa_lattice = Z.parse_kappa_lattice(kappa_l) if kappa_l is not None else None
    x.tau_fs = tau_fs
    return x.extract_all_properties(out)


def _dat(path):
    """An independent reader: (T, mu) -> average of the diagonal."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith("#") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        t = v[3:]
        diag = (t[0], t[4], t[8]) if len(t) == 9 else (t[0], t[3], t[5])
        out[(v[1], v[0])] = sum(diag) / 3
    return out


def _brute(stem, kappa_l, scale=1.0):
    S, G, K = (_dat(f"{stem}.{q}.dat") for q in ("SEEBECK", "SIGMA", "KAPPA"))
    floor = max(abs(v) for v in G.values() if v) * 1e-3
    best = (0.0, None)
    for key, s in S.items():
        g, k = G.get(key), K.get(key)
        if g is None or abs(g) < floor or not k:
            continue
        T = key[0]
        pf_t = s * s * abs(g) * T
        ke = abs(k) - pf_t
        if ke <= 0:
            continue
        zt = scale * pf_t / (scale * ke + kappa_l(T))
        if zt > best[0]:
            best = (zt, key)
    return best


def test_bcu_peak_point_by_hand():
    # the ZT_e peak point (300 K, mu -2.610 eV) of the existing kappa0 test,
    # now with kappa_L = 1.5 W/(m K) at the run's tau (22 fs)
    S = (2 * -3.2416492E-04 - 3.2727810E-04) / 3
    sigma = (2 * 5.6830186E+03 + 1.4938241E+03) / 3
    kappa0 = (2 * 1.9225154E-01 + 5.1793779E-02) / 3
    pf_t = S * S * sigma * 300.0
    want = pf_t / (kappa0 - pf_t + 1.5)
    from mace.gui import results
    z = results.transport_zt(find_data(BCU + ".SEEBECK.dat"), "1.5", None)
    b = next(b for b in z["blocks"] if b["T"] == 300.0)
    i = b["mu"].index(-2.61)
    assert b["zt_total"][i] == pytest.approx(want, rel=1e-9)
    assert b["zt_e"][i] == pytest.approx(pf_t / (kappa0 - pf_t), rel=1e-9)


@pytest.mark.parametrize("run", ["afi", "ana", "bcu-f"])
def test_extractor_total_matches_an_independent_brute_force(run):
    out = find_data(f"TRANSPORT/C1-RCSR-{run}_optimized_TRANSPORT.out")
    stem = str(out)[:-4]
    base = _props(out)
    p = _props(out, "1.5")
    zt, (T, mu) = _brute(stem, lambda T: 1.5)
    assert p["transport_zt_total_max_avg"] == pytest.approx(zt, rel=1e-12)
    assert p["transport_zt_total_max_temperature_k"] == T and p["transport_zt_total_max_mu_ev"] == mu
    assert p["transport_zt_total_kappa_lattice_at_max"] == 1.5
    assert p["transport_zt_total_tau_fs"] == 22.0
    assert p["transport_zt_total_lattice_source"] == "constant 1.5 W/(m·K)"
    assert "RELAXTIM" in p["transport_zt_total_tau_source"]
    # below the electronic ZT, which it never touches
    assert p["transport_zt_total_max_avg"] < p["transport_zt_electronic_max_avg"]
    for k in [k for k in base if k.startswith("transport")]:
        assert p[k] == base[k], k
    assert not any(k.startswith("transport_zt_total") for k in base)
    # the detail record
    d = p["transport_zt_total"]
    assert d["zt_total_max"] == p["transport_zt_total_max_avg"]
    assert d["formula"].startswith("S^2*sigma*T/(kappa_e + kappa_L)")


def test_extractor_tau_scaling():
    out = find_data(BCU + ".out")
    p = _props(out, "1.5", tau_fs=44.0)
    zt, _ = _brute(str(out)[:-4], lambda T: 1.5, scale=2.0)
    assert p["transport_zt_total_max_avg"] == pytest.approx(zt, rel=1e-12)
    assert p["transport_zt_total_tau_fs"] == 44.0
    assert "the run used 22 fs" in p["transport_zt_total_tau_source"]
    # a longer tau raises sigma and kappa_e but not kappa_L: the total rises
    assert p["transport_zt_total_max_avg"] > _props(out, "1.5")["transport_zt_total_max_avg"]
    # and ZT_e does not move
    assert p["transport_zt_electronic_max_avg"] == _props(out)["transport_zt_electronic_max_avg"]


def test_extractor_table(tmp_path):
    out = find_data(BCU + ".out")
    kl = _csv(tmp_path, "T,kappa_L\n300,3.0\n500,2.0\n700,1.0\n")
    p = _props(out, str(kl))
    f = lambda T: 3.0 - (T - 300.0) / 200.0
    zt, (T, mu) = _brute(str(out)[:-4], f)
    assert p["transport_zt_total_max_avg"] == pytest.approx(zt, rel=1e-12)
    assert p["transport_zt_total_kappa_lattice_at_max"] == pytest.approx(f(T))
    assert p["transport_zt_total_lattice_source"].startswith(f"table {kl.resolve()}")


def test_extractor_refuses_a_table_that_misses_a_run_temperature(tmp_path, capsys):
    out = find_data(BCU + ".out")                     # TRANGE 300 700 200
    kl = _csv(tmp_path, "300,3.0\n500,2.0\n")
    p = _props(out, str(kl))
    assert not any(k.startswith("transport_zt_total") for k in p)
    assert p["transport_zt_electronic_max_avg"] > 0
    msg = capsys.readouterr().out
    assert "700 K" in msg and "no total ZT" in msg and "does not extrapolate" in msg


def test_extractor_refuses_without_the_runs_tau(tmp_path, capsys):
    out = find_data(BCU + ".out")
    for f in out.parent.glob(out.stem + ".*"):
        if f.suffix in (".out", ".dat"):
            shutil.copy(f, tmp_path / f.name)
    o = tmp_path / out.name
    o.write_text("\n".join(l for l in o.read_text(errors="ignore").splitlines() if "RELAXATION TIME" not in l))
    p = _props(o, "1.5", tau_fs=10.0)
    assert not any(k.startswith("transport_zt_total") for k in p)
    assert "relaxation time" in capsys.readouterr().out


def test_the_degenerate_run_stores_no_total():
    p = _props(find_data("TRANSPORT/C1-RCSR-ato_optimized_TRANSPORT.out"), "1.5")
    assert "transport_zt_electronic_max_avg" not in p and "transport_zt_total_max_avg" not in p


def test_units():
    u = lambda n: CrystalPropertyExtractor._get_property_unit(None, n)
    assert u("transport_zt_total_max_avg") == "dimensionless"
    assert u("transport_zt_total_kappa_lattice_at_max") == "W/(m·K)"
    assert u("transport_zt_total_kappa_electronic_at_max") == "W/(m·K)"
    assert u("transport_zt_total_tau_fs") == "fs"
    assert u("transport_zt_total_max_temperature_k") == "K"
    assert u("transport_zt_total_max_mu_ev") == "eV"
    assert u("transport_zt_total_lattice_source") == "category"   # a text key (the registry, 1.1.6)


# ------------------------------------------------------------------ the CLI, as a subprocess
def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", *args],
                          capture_output=True, text=True, env=_env(), cwd=str(cwd), timeout=900)


def _run_dir(tmp_path):
    if not (TEST_DATA / "TRANSPORT").is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    d = tmp_path / "runs"
    d.mkdir()
    src = TEST_DATA / BCU
    for q in (".out", ".SEEBECK.dat", ".SIGMA.dat", ".KAPPA.dat"):
        shutil.copyfile(f"{src}{q}", d / (src.name + q))
    return d / (src.name + ".out")


def _rows_db(db):
    c = sqlite3.connect(str(db))
    try:
        return {n: (v, t, u) for n, v, t, u in c.execute(
            "SELECT property_name, property_value, property_value_text, property_unit FROM properties "
            "WHERE property_name LIKE 'transport%'")}
    finally:
        c.close()


def test_cli_writes_the_total_keys(tmp_path):
    out = _run_dir(tmp_path)
    db = tmp_path / "materials.db"
    p = _cli(tmp_path, "--extract-properties", "--output-file", str(out), "--db-path", str(db),
             "--kappa-lattice", "1.5")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    rows = _rows_db(db)
    want = _props(find_data(BCU + ".out"), "1.5")
    assert rows["transport_zt_total_max_avg"][0] == want["transport_zt_total_max_avg"]
    assert rows["transport_zt_total_max_avg"][2] == "dimensionless"
    assert rows["transport_zt_total_lattice_source"][1] == "constant 1.5 W/(m·K)"
    assert rows["transport_zt_total_tau_fs"][0] == 22.0
    assert rows["transport_zt_electronic_max_avg"][0] == want["transport_zt_electronic_max_avg"]
    # extracted again without --kappa-lattice: the total keys stay (the user's
    # kappa_L is not something a later extraction can know); the electronic ones are rewritten
    p = _cli(tmp_path, "--extract-properties", "--output-file", str(out), "--db-path", str(db))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    again = _rows_db(db)
    assert again["transport_zt_total_max_avg"] == rows["transport_zt_total_max_avg"]
    assert again["transport_zt_electronic_max_avg"] == rows["transport_zt_electronic_max_avg"]
    # a new kappa_L replaces them
    kl = _csv(tmp_path, "T,kappa_L\n300,3.0\n700,1.0\n")
    p = _cli(tmp_path, "--extract-properties", "--output-file", str(out), "--db-path", str(db),
             "--kappa-lattice", str(kl), "--tau-fs", "44")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    third = _rows_db(db)
    assert third["transport_zt_total_tau_fs"][0] == 44.0
    assert third["transport_zt_total_lattice_source"][1].startswith(f"table {kl.resolve()}")
    assert third["transport_zt_total_max_avg"][0] == _props(find_data(BCU + ".out"), str(kl), 44.0)["transport_zt_total_max_avg"]


@pytest.mark.parametrize("args,match", [
    (["--kappa-lattice", "-1"], "kappa_L must be a positive number"),
    (["--kappa-lattice", "nosuch.csv"], "neither a number nor a file"),
    (["--tau-fs", "10"], "--tau-fs goes with --kappa-lattice"),
    (["--kappa-lattice", "1.5", "--tau-fs", "0"], "relaxation time"),
])
def test_cli_refusals(tmp_path, args, match):
    out = _run_dir(tmp_path)
    db = tmp_path / "materials.db"
    p = _cli(tmp_path, "--extract-properties", "--output-file", str(out), "--db-path", str(db), *args)
    assert p.returncode == 2 and match in p.stderr, p.stdout[-2000:] + p.stderr[-2000:]
    assert not db.exists()


def test_cli_refuses_an_uncovered_temperature(tmp_path):
    out = _run_dir(tmp_path)
    db = tmp_path / "materials.db"
    kl = _csv(tmp_path, "300,3.0\n500,2.0\n")
    p = _cli(tmp_path, "--extract-properties", "--output-file", str(out), "--db-path", str(db),
             "--kappa-lattice", str(kl))
    assert p.returncode == 0
    assert "700 K" in p.stdout + p.stderr
    rows = _rows_db(db)
    assert "transport_zt_total_max_avg" not in rows and "transport_zt_electronic_max_avg" in rows


def test_cli_help_lists_the_flags():
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "help", "analyze"],
                       capture_output=True, text=True, env=_env(), timeout=300)
    assert "--kappa-lattice" in p.stdout and "--tau-fs" in p.stdout


# ------------------------------------------------------------------ CLI and GUI: the same numbers
@pytest.mark.parametrize("kl,tau", [("1.5", None), ("0.3", 44.0), ("table", None), ("table", 11.0)])
@pytest.mark.parametrize("run", ["afi", "ana", "bcu-f"])
def test_gui_numbers_are_the_clis(tmp_path, run, kl, tau):
    from mace.gui import results
    out = find_data(f"TRANSPORT/C1-RCSR-{run}_optimized_TRANSPORT.out")
    if kl == "table":
        kl = str(_csv(tmp_path, "T,kappa_L\n300,3.0\n500,2.0\n700,1.0\n"))
    p = _props(out, kl, tau)
    for q in ("SEEBECK", "SIGMA", "KAPPA"):       # opened from any of the three files
        z = results.transport_zt(Path(str(out)[:-4] + f".{q}.dat"), kl, tau)
        m = z["zt_total_max"]
        assert m["value"] == p["transport_zt_total_max_avg"]                      # identical, not approx
        assert (m["temperature_k"], m["mu_ev"]) == (p["transport_zt_total_max_temperature_k"],
                                                    p["transport_zt_total_max_mu_ev"])
        assert z["zt_e_max"]["value"] == p["transport_zt_electronic_max_avg"]
        assert z["tau_fs"] == p["transport_zt_total_tau_fs"] and z["tau_run_fs"] == 22.0
        assert z["kappa_l_source"] == p["transport_zt_total_lattice_source"]
        # the curve's highest point is that peak
        best = max(v for b in z["blocks"] for v in (b["zt_total"] or []) if v is not None)
        assert best == m["value"]


def test_gui_refuses_like_the_cli(tmp_path):
    from mace.gui import results
    seebeck = find_data(BCU + ".SEEBECK.dat")
    kl = _csv(tmp_path, "300,3.0\n500,2.0\n")
    z = results.transport_zt(seebeck, str(kl), None)
    by_t = {b["T"]: b for b in z["blocks"]}
    assert by_t[700.0]["zt_total"] is None and "700 K is outside" in by_t[700.0]["refused"]
    assert by_t[300.0]["zt_total"] and by_t[300.0]["refused"] is None
    assert z["zt_total_max"] is None and "700 K" in z["refused"]     # no run-wide peak, as the CLI
    with pytest.raises(Z.KappaLatticeError):
        results.transport_zt(seebeck, "-2", None)
    # no .out beside the data: no tau, no total
    d = tmp_path / "bare"
    d.mkdir()
    for q in ("SEEBECK", "SIGMA", "KAPPA"):
        shutil.copy(find_data(BCU + f".{q}.dat"), d)
    z = results.transport_zt(d / Path(BCU + ".SEEBECK.dat").name, "1.5", None)
    assert z["tau_run_fs"] is None and z["zt_total_max"] is None and "relaxation time" in z["refused"]
    assert all(b["zt_total"] is None for b in z["blocks"]) and z["zt_e_max"]["value"] > 0


def test_tracked_rows_are_kept_too(tmp_path):
    # registered (calc_id set): re-extraction replaces the calculation's rows,
    # except the total ZT's when this extraction had no --kappa-lattice
    out = _run_dir(tmp_path)
    db = tmp_path / "materials.db"
    p = _cli(tmp_path, "--extract-properties", str(out.parent), "--db-path", str(db), "--register",
             "--kappa-lattice", "1.5")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    c = sqlite3.connect(str(db))
    try:
        (calc,), = c.execute("SELECT DISTINCT calc_id FROM properties WHERE property_name = 'transport_zt_total_max_avg'")
    finally:
        c.close()
    assert calc is not None
    first = _rows_db(db)
    p = _cli(tmp_path, "--extract-properties", str(out.parent), "--db-path", str(db))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert _rows_db(db)["transport_zt_total_max_avg"] == first["transport_zt_total_max_avg"]
