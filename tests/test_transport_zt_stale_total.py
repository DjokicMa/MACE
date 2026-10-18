"""A total ZT the database keeps is always the one of the run as it is now.

The reviewer's recipe on the real afi run: `--kappa-lattice 1.5 --tau-fs 10`
stored 0.8799; a re-run with a kappa_L table that misses 700 K printed "no
total ZT stored" while the database still held 0.8799 (lattice source
"constant 1.5", tau 10). Now:

* a kappa_L given and refused (a table that misses a temperature, no
  RELAXTIM in the .out) removes the run's earlier total rows, and says so;
* an extraction without --kappa-lattice keeps them only while the run's
  .out and .dat files are the ones they were computed from (the size and
  first/last 64 KiB fingerprint of the database build's change detection);
  otherwise it removes them and says so.

Both for an untracked extraction (calc_id NULL) and a registered one.
"""
import shutil
import sqlite3

import pytest

from conftest import TEST_DATA
from test_transport_zt_total import _cli, _csv, _rows_db

AFI = "TRANSPORT/C1-RCSR-afi_optimized_TRANSPORT"


def _afi_dir(tmp_path):
    if not (TEST_DATA / "TRANSPORT").is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    d = tmp_path / "runs"
    d.mkdir()
    src = TEST_DATA / AFI
    for q in (".out", ".SEEBECK.dat", ".SIGMA.dat", ".KAPPA.dat"):
        shutil.copyfile(f"{src}{q}", d / (src.name + q))
    return d / (src.name + ".out")


def _totals(db):
    return {k: v for k, v in _rows_db(db).items() if k.startswith("transport_zt_total")}


def _extract(tmp_path, out, db, *args, register=False):
    target = ["--extract-properties", str(out.parent), "--register"] if register else \
        ["--extract-properties", "--output-file", str(out)]
    p = _cli(tmp_path, *target, "--db-path", str(db), *args)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout + p.stderr


@pytest.mark.parametrize("register", [False, True], ids=["untracked", "registered"])
def test_a_refused_table_removes_the_earlier_total(tmp_path, register):
    out = _afi_dir(tmp_path)
    db = tmp_path / "materials.db"
    _extract(tmp_path, out, db, "--kappa-lattice", "1.5", "--tau-fs", "10", register=register)
    first = _totals(db)
    assert first["transport_zt_total_max_avg"][0] == pytest.approx(0.879946578, rel=1e-8)
    kl = _csv(tmp_path, "T,kappa_L\n300,3.0\n600,1.0\n")
    log = _extract(tmp_path, out, db, "--kappa-lattice", str(kl), register=register)
    assert "700 K" in log and "no total ZT stored" in log
    assert "earlier total ZT" in log and "removed" in log, log[-2000:]
    assert _totals(db) == {}
    assert "transport_zt_electronic_max_avg" in _rows_db(db)


def test_a_refused_tau_removes_the_earlier_total(tmp_path):
    out = _afi_dir(tmp_path)
    db = tmp_path / "materials.db"
    _extract(tmp_path, out, db, "--kappa-lattice", "1.5")
    assert _totals(db)
    out.write_text("\n".join(l for l in out.read_text(errors="ignore").splitlines()
                             if "RELAXATION TIME" not in l) + "\n")
    log = _extract(tmp_path, out, db, "--kappa-lattice", "1.5")
    assert "relaxation time" in log and "earlier total ZT" in log
    assert _totals(db) == {}


@pytest.mark.parametrize("register", [False, True], ids=["untracked", "registered"])
def test_without_the_flag_the_total_stays_only_while_the_run_is_unchanged(tmp_path, register):
    out = _afi_dir(tmp_path)
    db = tmp_path / "materials.db"
    _extract(tmp_path, out, db, "--kappa-lattice", "1.5", register=register)
    first = _totals(db)
    assert first
    # unchanged run (a touch changes nothing that matters): kept, said nothing about it
    out.touch()
    log = _extract(tmp_path, out, db, register=register)
    assert _totals(db) == first
    assert "earlier total ZT" not in log
    # the SIGMA.dat the total was computed from has changed: removed, and said so
    sig = out.with_name(out.stem + ".SIGMA.dat")
    lines = sig.read_text().splitlines()
    sig.write_text("\n".join(lines[:3] + [lines[3].replace("E+06", "E+07", 1)] + lines[4:]) + "\n")
    log = _extract(tmp_path, out, db, register=register)
    assert _totals(db) == {}
    assert "earlier total ZT" in log and "changed" in log, log[-2000:]
    assert "transport_zt_electronic_max_avg" in _rows_db(db)


def test_a_new_flag_value_replaces_the_total(tmp_path):
    out = _afi_dir(tmp_path)
    db = tmp_path / "materials.db"
    _extract(tmp_path, out, db, "--kappa-lattice", "1.5")
    _extract(tmp_path, out, db, "--kappa-lattice", "1.5", "--tau-fs", "10")
    t = _totals(db)
    assert t["transport_zt_total_max_avg"][0] == pytest.approx(0.879946578, rel=1e-8)
    assert t["transport_zt_total_tau_fs"][0] == 10.0
    c = sqlite3.connect(str(db))
    try:
        n, = c.execute("SELECT COUNT(*) FROM properties WHERE property_name = 'transport_zt_total_max_avg'").fetchone()
    finally:
        c.close()
    assert n == 1


def test_one_kappa_lattice_for_several_materials_is_warned_about(tmp_path):
    if not (TEST_DATA / "TRANSPORT").is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    d = tmp_path / "runs"
    d.mkdir()
    for run in ("afi", "bcu-f"):
        src = TEST_DATA / f"TRANSPORT/C1-RCSR-{run}_optimized_TRANSPORT"
        for q in (".out", ".SEEBECK.dat", ".SIGMA.dat", ".KAPPA.dat"):
            shutil.copyfile(f"{src}{q}", d / (src.name + q))
    db = tmp_path / "materials.db"
    p = _cli(tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--kappa-lattice", "1.5")
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-3000:]
    assert "One κL (constant 1.5 W/(m·K)) was used for the total ZT of 2 materials" in log, log[-3000:]
    # one material: no warning
    one = tmp_path / "one"
    one.mkdir()
    for f in d.glob("C1-RCSR-afi*"):
        shutil.copyfile(f, one / f.name)
    p = _cli(tmp_path, "--extract-properties", str(one), "--db-path", str(tmp_path / "b.db"), "--kappa-lattice", "1.5")
    assert p.returncode == 0 and "One κL" not in p.stdout + p.stderr
