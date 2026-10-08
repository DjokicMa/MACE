"""F9: runtime estimates from history: a perturbation test, the neighbor
rules, leave-one-out coverage on the real corpus, and a read-only DB."""
import math
import os
import random
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import TEST_DATA
from mace.cluster import history as H


def _law(nao, nk, cores, alpha=0.8, group_factor=1.0, cycles=20):
    per = 1e-6 * nao ** 3 * nk * (32.0 / cores) ** alpha * group_factor
    return per * cycles


def _synthetic(seed, one_core_count=False):
    rng = random.Random(seed)
    recs = []
    for i in range(80):
        nao, nk = rng.randint(80, 400), rng.randint(10, 80)
        group = "genA" if i % 2 == 0 else "genB"
        gf = 1.0 if group == "genA" else 2.0
        core_list = [32] if one_core_count else [16, 64]
        for cores in core_list:
            wall = _law(nao, nk, cores, group_factor=gf) * math.exp(rng.gauss(0, 0.2))
            recs.append({"file": f"r{i}_{cores}", "stem": f"r{i}", "calc_type": "SP", "family": "hybrid",
                         "nao": nao, "nk": nk, "atoms": nao // 10, "cores": cores, "cycles": 20, "wall_s": wall,
                         "group": group})
    recs[3]["wall_s"] *= 3.0                                  # one outlier
    return recs


def test_perturbation():
    """F9-a: truth inside p10-p90 for >= 70 % of 200 queries; median within
    30 % for >= 70 %; dropping the node group widens the range."""
    recs = _synthetic(1)
    rng = random.Random(7)
    inside = close = 0
    widths_g, widths_nog = [], []
    for q in range(200):
        nao, nk, cores = rng.randint(100, 380), rng.randint(12, 70), rng.choice([16, 32, 64])
        truth = _law(nao, nk, cores)                        # group genA
        feats = {"calc_type": "SP", "family": "hybrid", "nao": nao, "nk": nk, "atoms": nao // 10}
        e = H.estimate_from(feats, recs, cores=cores, group="genA")
        w = e["wall"]
        inside += w["p10"] <= truth <= w["p90"]
        close += abs(w["median"] / truth - 1) <= 0.3
        widths_g.append(w["p90"] / w["p10"])
        e2 = H.estimate_from(feats, recs, cores=cores, group=None)
        widths_nog.append(e2["wall"]["p90"] / e2["wall"]["p10"])
    print(f"\nF9-a: inside {inside / 200:.0%}, median within 30 % {close / 200:.0%}, "
          f"width with group {sorted(widths_g)[100]:.2f} vs without {sorted(widths_nog)[100]:.2f}")
    assert inside / 200 >= 0.7
    assert close / 200 >= 0.7
    assert sorted(widths_nog)[100] > 1.2 * sorted(widths_g)[100]


def test_single_core_count_uses_alpha_range():
    recs = _synthetic(2, one_core_count=True)
    e = H.estimate_from({"calc_type": "SP", "family": "hybrid", "nao": 200, "nk": 30, "atoms": 20}, recs,
                        cores=64, group="genA")
    assert any("alpha 0.6..1.0" in r for r in e["reasons"])
    assert e["wall"]["p90"] > e["wall"]["p10"]


def test_too_few_and_other_types():
    """F9-b."""
    recs = _synthetic(3)
    feats = {"calc_type": "SP", "family": "gga", "nao": 200, "nk": 30, "atoms": 20}
    e = H.estimate_from(feats, recs, cores=32)
    assert e["message"] == "Not enough similar runs (0 found; 3 needed)" and e["wall"] is None
    two = [dict(r, family="gga") for r in recs[:2]]
    e = H.estimate_from(feats, recs + two, cores=32)
    assert e["message"] == "Not enough similar runs (2 found; 3 needed)"
    opt = [dict(r, calc_type="OPT") for r in recs]
    e = H.estimate_from({**feats, "family": "hybrid"}, opt, cores=32)
    assert e["n"] == 0


@pytest.mark.parametrize("sub", ["OPT", "SP"])
def test_leave_one_out_on_real_corpus(sub, tmp_path, monkeypatch):
    """F9-c: per calc type with >= 5 runs, p10-p90 holds the true wall time
    in >= 50 % (else estimate() hides it). Cores from each run's .sh."""
    folder = TEST_DATA / sub
    if not folder.is_dir():
        pytest.skip("test/ data corpus not present")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # a copy of the DB: even a read-only reader updates a WAL database's
    # -shm file, and nothing may write into the read-only test/ corpus
    before = {f.name: f.stat().st_mtime_ns for f in folder.iterdir() if f.is_file()}
    db = None
    if (folder / "materials.db").is_file():
        db = tmp_path / "materials.db"
        shutil.copyfile(folder / "materials.db", db)
    recs = H.index(folder, use_sacct=False, db_path=db)
    assert {f.name: f.stat().st_mtime_ns for f in folder.iterdir() if f.is_file()} == before
    assert any(r["cores"] for r in recs)
    cov = H.coverage(recs)
    print(f"\nF9-c {sub}: " + ", ".join(f"{k} {v['hit']}/{v['n']}" for k, v in cov.items() if v["n"]))
    for k, v in cov.items():
        if v["n"] >= 5:
            assert v["coverage"] >= 0.5, (k, v)


def test_reader_does_not_touch_a_db_being_written(tmp_path):
    """F9-d: index() reads a WAL DB while a writer writes; integrity stays
    ok; the reader never changes the file's mtime."""
    from mace.database.materials import MaterialDatabase
    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("m1", "C2")
    stop = threading.Event()

    def writer():
        k = 0
        while not stop.is_set():
            c = db.create_calculation("m1", "SP", input_file=str(tmp_path / f"x{k}.d12"), work_dir=str(tmp_path))
            db.update_calculation_status(c, "completed", output_file=str(tmp_path / f"x{k}.out"))
            k += 1
            time.sleep(0.01)
    t = threading.Thread(target=writer)
    t.start()
    try:
        time.sleep(0.2)
        for _ in range(5):
            m0 = os.stat(db_path).st_mtime_ns
            stop_writes = len(H._db_records(db_path))
            assert stop_writes >= 0
            H.index(tmp_path, use_sacct=False)
    finally:
        stop.set()
        t.join()
    m1 = os.stat(db_path).st_mtime_ns
    H.index(tmp_path, use_sacct=False)
    assert os.stat(db_path).st_mtime_ns == m1
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()
