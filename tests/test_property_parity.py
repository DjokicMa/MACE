"""GUI/database property parity on every run of the example suite
(mace/properties/parity.py): after `mace analyze --extract-properties` (run
as the CLI), the Properties tab of each run lists exactly the run's rows in
the database; what the tab computes for a run not in a database (the
extractor and the derived-property registry) is the same keys with the same
values; and every value the GUI computes elsewhere (the run card, the
automatic masses of the Relaxation Time dialog, the transport plot's ZT) is
in the database with the same value. The next property agents extend
parity.PRODUCERS. Also: the tab's routes (the run of a plot file, a run not
in the database, no database, a JSON value in full, the material scope) and
the Relaxation Time dialog's Save to Database, in process."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.gui import properties_routes as PR
from mace.properties import parity, registry
from mace.utils import db_build

STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=_env(), cwd=str(cwd), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p


class St:
    """The server state the routes use: the served folder, its path rules, its writes."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root_gen = 0
        self.test_hooks = False

    def resolve(self, rel, must_exist=True):
        t = (self.root / rel).resolve()
        if t != self.root and self.root not in t.parents:
            raise PermissionError("outside the served directory")
        if must_exist and not t.exists():
            raise FileNotFoundError(rel)
        return t

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()

    def check_write(self, target, allow=()):
        self.resolve(Path(target).resolve().relative_to(self.root).as_posix(), must_exist=False)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    root = tmp_path_factory.mktemp("parity") / "root"
    ES.copy_suite(root / "suite")
    _cli(root, "analyze", "--extract-properties", "suite", "--db-path", "suite/materials.db")
    return root


def _runs(root):
    return sorted((root / "suite").rglob("*.out"))


def test_the_gui_and_the_database_agree_on_every_run(suite):
    conn = db_build.open_ro(suite / "suite" / "materials.db")
    used = {p.name: 0 for p in parity.PRODUCERS}
    problems = {}
    runs = _runs(suite)
    assert len(runs) >= 30
    try:
        for out in runs:
            cid = db_build.calc_for_output(conn, out.resolve())
            assert cid, f"{out.name} is not in the database"
            ct = conn.execute("SELECT calc_type FROM calculations WHERE calc_id = ?", (cid,)).fetchone()[0]
            db = registry.stored_values(conn, cid)
            # the tab of a run in the database: every row, nothing else
            view = PR.run_view(St(suite), out)
            assert view["state"] == "db" and view["calc"]["calc_id"] == cid, out.name
            assert sorted(r["name"] for r in view["rows"]) == sorted(db), out.name
            for r in view["rows"]:
                assert r["label"] and r["group"] and r["script"], (out.name, r["name"])
            # what the tab computes from the run (its properties/compute route: the reference energies
            # look their runs up in the run's database, as the run), and the GUI's other computations
            computed = PR.computed_values(out, suite / "suite" / "materials.db")
            p = parity.compare(computed, db) + parity.producer_problems(out, ct, db)
            for prod in parity.PRODUCERS:
                if "*" in prod.calc_types or ct in prod.calc_types:
                    if any(v is not None for v in prod.values(out, ct).values()):
                        used[prod.name] += 1
            if p:
                problems[out.name] = p
    finally:
        conn.close()
    assert not problems, json.dumps(problems, indent=1)[:4000]
    # every producer was compared on at least one run (the suite has ELASTIC, EOS, QHA and CPHF runs)
    assert all(used.values()), used


def test_derived_rows_say_where_they_came_from(suite):
    band = next(p for p in _runs(suite) if p.name == "diamond_band.out")
    view = PR.run_view(St(suite), band)
    rows = {r["name"]: r for r in view["rows"]}
    m = rows["effective_mass_hole_mc"]
    assert m["derived"] == "effective_masses" and m["version"] == "1" and m["unit_text"] == "m_e"
    assert m["label"] == "Effective mass m<sub>c</sub>, holes (accurate)" and m["inputs"][0] == "diamond_band.out"
    assert m["script"] == registry.EXTRACTOR_SCRIPT and m["file"] == "diamond_band.out"
    assert rows["effective_masses"]["blob"] and rows["effective_masses"]["json"]
    assert {d["name"]: d["status"] for d in view["derived"]}["effective_masses"] == "stored"
    assert view["missing"] == []
    # the full value of a large JSON row
    st = St(suite)
    big = next(r for r in view["rows"] if r["json"] and r["value"] is None)
    got = PR._value(st, {"db": [view["db"]], "calc": [big["calc_id"]], "name": [big["name"]]})
    assert got["value"] and got["name"] == big["name"]


def test_a_plot_file_shows_its_runs_properties(suite):
    dat = next((suite / "suite").rglob("diamond_band.BAND.DAT"))
    assert PR.run_of(dat).name == "diamond_band.out"
    view = PR.run_view(St(suite), dat)
    assert view["state"] == "db" and view["run"].endswith("diamond_band.out")
    cif = next((suite / "suite").rglob("*.d12"))
    assert PR.run_of(cif) is None or PR.run_of(cif).suffix == ".out"
    # all runs of the material, grouped by calculation type
    view = PR.run_view(St(suite), next(p for p in _runs(suite) if p.name == "diamond_sp.out"), scope="material")
    assert {"SP", "BAND", "DOSS", "FREQ", "OPT"} <= {r["calc_type"] for r in view["rows"]}


def test_a_run_not_in_the_database_and_no_database(suite, tmp_path):
    st = St(suite)
    extra = suite / "suite" / "extra"
    extra.mkdir(exist_ok=True)
    src = next(p for p in _runs(suite) if p.name == "EC_sp.out")
    shutil.copy(src, extra / "ECX_sp.out")
    view = PR.run_view(st, extra / "ECX_sp.out")
    assert view["state"] == "not-in-db" and view["db"] == "suite/materials.db"
    assert view["extract"]["command"] == ("mace analyze --extract-properties --output-file suite/extra/ECX_sp.out "
                                          "--db-path suite/materials.db --register --register-output-only --backup --keep-backups 3")
    # the command Extract runs, as the CLI: the run is then in the database
    _cli(suite, *PR.extract_argv("suite/extra/ECX_sp.out", "suite/materials.db"))
    view = PR.run_view(st, extra / "ECX_sp.out")
    assert view["state"] == "db" and "energy_per_atom_ev" in {r["name"] for r in view["rows"]}
    # no database anywhere up to the served folder
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(src, alone / "EC_sp.out")
    view = PR.run_view(St(tmp_path), alone / "EC_sp.out")
    assert view["state"] == "no-db" and view["folder"] == "alone"
    res = PR.compute_job(path=str(alone / "EC_sp.out"))
    names = {r["name"] for r in res["rows"]}
    assert res["state"] == "computed" and {"energy_per_atom_ev", "crystal_version", "extraction_provenance"} <= names
    assert all(r["script"] == "computed now (not stored)" for r in res["rows"])


def test_save_to_database_from_the_relaxation_time_dialog(tmp_path):
    from mace.gui import mechanics_routes as MR
    root = tmp_path / "root"
    (root / "strain").mkdir(parents=True)
    for f in STRAIN.glob("*.out"):
        shutil.copy(f, root / "strain" / f.name)
    _cli(root, "analyze", "--extract-properties", "strain", "--db-path", "strain/materials.db")
    st = St(root)
    outs = sorted(f"strain/{p.name}" for p in (root / "strain").glob("*.out"))
    body = {"outputs": outs, "carrier": "electron", "m_star": 0.3, "save_db": True}
    out = MR.fit(st, body)
    sd = out["saved_db"]
    assert "error" not in sd, sd
    assert sd["run"] == "strain/dia_sp_dp_x_p0pct.out" and sd["calc_id"]
    assert sorted(sd["keys"]) == sorted(["relaxation_time_dpt_electron_x_fs", "mobility_dpt_electron_x_cm2_vs",
                                         "relaxation_time_dpt_electron_x"])
    view = PR.run_view(st, root / "strain" / "dia_sp_dp_x_p0pct.out")
    rows = {r["name"]: r for r in view["rows"]}
    assert rows["relaxation_time_dpt_electron_x_fs"]["value"] == pytest.approx(out["result"]["tau_fs"])
    assert rows["relaxation_time_dpt_electron_x_fs"]["script"] == registry.GUI_SCRIPT
    assert rows["relaxation_time_dpt_electron_x_fs"]["interactive"]
    # a re-extraction of the run keeps what the GUI saved
    _cli(root, "analyze", "--extract-properties", "strain", "--db-path", "strain/materials.db")
    conn = sqlite3.connect(root / "strain" / "materials.db")
    assert conn.execute("SELECT extractor_script FROM properties WHERE property_name = "
                        "'relaxation_time_dpt_electron_x_fs'").fetchall() == [(registry.GUI_SCRIPT,)]
    conn.close()
    # without a database the fit stands and the save says why
    (root / "strain" / "materials.db").unlink()
    out = MR.fit(st, body)
    assert out["result"]["tau_fs"] > 0 and "Create or Update Database" in out["saved_db"]["error"]
