"""Building a materials database from a folder through the CLI
(PLAN_analysis C8 extractor fixes and C9 opt-in flags).

`mace_cli analyze --extract-properties DIR --db-path DB [--register]
[--dry-run --json] [--skip-unchanged] [--backup --keep-backups N]
[--progress-json]` as subprocesses on real outputs copied from the corpus.
The pre-change tree (for the "nothing moved by DELETE" and "unchanged
without the flags" checks) is `git archive` of the base commit, unpacked in
tmp at test time; nothing of it is committed."""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.properties import registry as R
from mace.utils import property_keys as PK

BASE_COMMIT = "daa5468e"          # the 1.1.5 base before the GUI (reachable in every clone)
# properties the extraction-depth audit added (C11): not in the base tree's rows
NEW_PROPS = ("space_group_number", "raman_intensities", "num_raman_modes",
             # and those whose patterns matched no CRYSTAL23 line before 1.1.6
             "scf_cycles", "scf_fmixing_percentage",
             # 1.1.6's key clean-up: new keys
             "wall_time_s", "layer_group_number", "layer_group_symbol", "transport_tensor_average",
             # derived properties (mace/properties) and their provenance row
             "extraction_provenance", *sorted({o.name for p in R.load().values() for o in p.outputs}))
NEW_MARKS = ",".join("?" * len(NEW_PROPS))
# properties the base tree stored wrongly and this tree no longer stores at all
# (so a re-extraction removes their rows): a multi-point OPT's point-1 energy
# components (155d83b1) and the DOS-only advanced gap/classification (a3308d9d)
DROPPED_PROPS = tuple(f"{c}_energy_{u}" for c in ("kinetic", "electron_electron", "electron_nuclear",
                                                   "nuclear_nuclear", "exchange", "correlation")
                      for u in ("au", "ev")) + (
    "band_gap_advanced_ev", "band_gap_advanced_ha", "electronic_classification_advanced") + tuple(
    # 1.1.6's key clean-up: keys removed (property_keys.REMOVED), and the
    # constant "No BAND.DAT or DOSS.DAT files found" note
    sorted(PK.REMOVED - {"band_gap_advanced_ha"})) + ("advanced_analysis_note", "band_start")
DROPPED = f"property_name IN ({','.join('?' * len(DROPPED_PROPS))})"
# doubled geometry keys the base tree stored and this tree stores under their
# single prefix (d6408ac5): the same values, so where the single name was stored
# as well the row is gone, else it comes back under the single name
RENAMED_PROPS = {"final_final_atomic_positions": "final_atomic_positions",
                 "final_final_atoms_count": "final_atoms_count",
                 "final_final_atomic_position_axes": "final_atomic_position_axes",
                 "initial_initial_atomic_positions": "initial_atomic_positions",
                 "initial_initial_atoms_count": "initial_atoms_count"}
# and 1.1.6's renamed and merged keys (property_keys.ALIASES), chained
RENAMED_PROPS = {old: PK.ALIASES.get(new, new) for old, new in RENAMED_PROPS.items()}
RENAMED_PROPS.update({old: new for old, new in PK.ALIASES.items()
                      if old not in DROPPED_PROPS and old not in RENAMED_PROPS})
RENAMED = f"property_name IN ({','.join('?' * len(RENAMED_PROPS))})"
# the rows both trees store under the same name (bind DROPPED_PROPS + tuple(RENAMED_PROPS))
KEPT = f"NOT ({DROPPED}) AND NOT ({RENAMED})"
GONE = DROPPED_PROPS + tuple(RENAMED_PROPS)
DIA = {
    "OPT": ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12"],
    "SP": ["SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12"],
    "BAND": ["BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out", "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.d3",
             "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT"],
    "DOSS": ["DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.d3",
             "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.DOSS.DAT"],
}
MIXED_EXTRA = ["FREQ/2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out",
               "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.out", "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.d3",
               "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.SEEBECK.dat",
               "ECH3POT3/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential.out",
               "ECH3POT3/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential.d3",
               "FAILED_QA/3_dia3_opt_killed_signal9_oom.out"]


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _folder(tmp, rels):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for r in rels:
        shutil.copyfile(TEST_DATA / r, d / Path(r).name)
    return d


def _cli(tree, cwd, *args, ok=True):
    p = subprocess.run([sys.executable, str(Path(tree) / "mace_cli"), "--no-banner", "analyze", *args],
                       capture_output=True, text=True, env=_env(), cwd=str(cwd), timeout=900)
    if ok:
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _populate(tree, cwd, d, db):
    p = subprocess.run([sys.executable, "-m", "mace.database.populate_completed_jobs", str(d), "--db", str(db)],
                       capture_output=True, text=True, env=dict(_env(), PYTHONPATH=str(tree)), cwd=str(cwd), timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


def _q(db, sql, *a):
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    """The base commit's tree (git archive), or skip."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", BASE_COMMIT + "^{commit}"], capture_output=True)
    if r.returncode:
        pytest.skip(f"base commit {BASE_COMMIT} not in this clone")
    d = tmp_path_factory.mktemp("basetree")
    arch = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", BASE_COMMIT, "mace", "mace_cli", "Crystal_d12",
                           "Crystal_d3", "mace_config.py"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(d)], input=arch, check=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ C8
def test_tracked_rows_go_under_the_calculations_material(tmp_path):
    d = _folder(tmp_path, sum(DIA.values(), []))
    db = tmp_path / "materials.db"
    _populate(REPO_ROOT, tmp_path, d, db)
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db))
    mats = {r[0] for r in _q(db, "SELECT material_id FROM materials")}
    assert {r[0] for r in _q(db, "SELECT DISTINCT material_id FROM properties")} <= mats
    (formula, sg), = _q(db, "SELECT formula, space_group FROM materials")
    assert formula != "Unknown" and sg == 227
    assert "Materials processed: 1" in p.stdout


def test_rows_from_the_old_tree_are_moved_not_deleted(tmp_path, base_tree):
    d = _folder(tmp_path, sum(DIA.values(), []))
    db = tmp_path / "materials.db"
    _populate(base_tree, tmp_path, d, db)
    _cli(base_tree, tmp_path, "--extract-properties", str(d), "--db-path", str(db))
    before = _q(db, f"SELECT count(*) FROM properties WHERE {KEPT}", *GONE)[0][0]
    mats = {r[0] for r in _q(db, "SELECT material_id FROM materials")}
    old_ids = {r[0] for r in _q(db, "SELECT DISTINCT material_id FROM properties WHERE calc_id IS NOT NULL")} - mats
    assert old_ids, "the base tree filed tracked rows under file-name ids (the bug this fixes)"
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db))
    assert _q(db, f"SELECT count(*) FROM properties WHERE {KEPT}", *GONE)[0][0] >= before
    # the re-extraction replaced the doubled geometry keys by their single-prefix names
    assert not _q(db, f"SELECT 1 FROM properties WHERE calc_id IS NOT NULL AND {RENAMED}", *RENAMED_PROPS)
    left = _q(db, f"SELECT count(*) FROM properties WHERE calc_id IS NOT NULL AND material_id IN "
                  f"({','.join('?' * len(old_ids))})", *old_ids)[0][0]
    assert left == 0
    assert not _q(db, "SELECT calc_id, property_name, count(*) FROM properties WHERE calc_id IS NOT NULL "
                      "GROUP BY calc_id, property_name HAVING count(*) > 1")
    assert re.search(r"moved \d+ rows of \S+ from \S+ to 1_dia", p.stdout), p.stdout[-3000:]


def test_untracked_folder_behaves_as_before(tmp_path, base_tree):
    d = _folder(tmp_path, sum(DIA.values(), []))
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _cli(base_tree, tmp_path, "--extract-properties", str(d), "--db-path", str(a))
    _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(b))
    # the same properties; filed under the id a registration gives the outputs
    # ("1_dia"), where the base tree split them by a one-suffix strip of each name
    # (the names this tree no longer stores left out; its large JSON values are
    # in property_blobs)
    q = f"SELECT DISTINCT property_name FROM properties WHERE property_name NOT IN ({NEW_MARKS}) AND NOT ({DROPPED})"
    qb = (q + f" UNION SELECT DISTINCT property_name FROM property_blobs WHERE property_name NOT IN ({NEW_MARKS})"
          f" AND NOT ({DROPPED})")
    assert {(RENAMED_PROPS.get(n, n),) for (n,) in _q(a, q, *NEW_PROPS, *DROPPED_PROPS)} == \
        set(_q(b, qb, *NEW_PROPS, *DROPPED_PROPS, *NEW_PROPS, *DROPPED_PROPS))
    assert not _q(b, f"SELECT 1 FROM properties WHERE {RENAMED}", *RENAMED_PROPS)
    assert len(_q(a, "SELECT DISTINCT material_id FROM properties")) > 1
    assert _q(b, "SELECT DISTINCT material_id FROM properties") == [("1_dia",)]


# ------------------------------------------------------------------ C9
def _mixed(tmp_path):
    return _folder(tmp_path, sum(DIA.values(), []) + MIXED_EXTRA)


def _dry(cwd, d, db, *extra):
    p = _cli(REPO_ROOT, cwd, "--extract-properties", str(d), "--db-path", str(db), "--dry-run", "--json", *extra)
    return json.loads(p.stdout)


def _by_name(doc):
    return {Path(f["file"]).name: f["class"] for f in doc["files"]}


def test_dry_run_classes(tmp_path):
    d = _mixed(tmp_path)
    db = tmp_path / "materials.db"
    doc = _dry(tmp_path, d, db, "--register")
    assert not db.exists() and not (tmp_path / "structures.db").exists() and not (d / "materials.db").exists()
    cls = _by_name(doc)
    failed = "3_dia3_opt_killed_signal9_oom.out"
    # a failed run no calculation records: listed, not extracted (its rows had no material)
    assert cls.pop(failed) == "skipped"
    assert set(cls.values()) == {"new"}, cls
    # without --register (--no-register; registering is the default for a
    # folder) on a fresh database: not tracked
    # (the failed run is extracted then, as by a plain run: incomplete)
    assert set(_by_name(_dry(tmp_path, d, db, "--no-register")).values()) <= {"not tracked (always re-extracted)", "skipped",
                                                              "incomplete"}
    _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register")
    cls = _by_name(_dry(tmp_path, d, db, "--no-register"))
    # without --register a failed run is extracted, as a plain run does
    assert cls.pop(failed) == "incomplete"
    assert set(cls.values()) == {"unchanged"}, cls
    opt = d / "1_dia_opt_rev1.out"
    with open(opt, "a") as fh:                     # one byte more
        fh.write("\n")
    assert _by_name(_dry(tmp_path, d, db))[opt.name] == "changed"
    sp = d / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
    st = sp.stat()
    other = TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
    data = bytearray(other.read_bytes())
    k = data.rindex(b"TELAPSE")                       # different content near the end, same size
    data[k:k + 7] = b"TELAPSe"
    data = bytes(data)
    sp.write_bytes(data)
    os.utime(sp, ns=(st.st_atime_ns, st.st_mtime_ns))                             # as cp -p would
    assert sp.stat().st_size == st.st_size
    assert _by_name(_dry(tmp_path, d, db))[sp.name] == "changed"
    sp.write_bytes(other.read_bytes())
    _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register")
    band = d / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out"
    os.utime(band)                                                                # touch only
    assert _by_name(_dry(tmp_path, d, db))[band.name] == "unchanged"
    if os.geteuid() != 0:
        band.chmod(0)
        try:
            assert _by_name(_dry(tmp_path, d, db))[band.name] == "unreadable"
        finally:
            band.chmod(0o644)


def test_backup_and_pruning(tmp_path):
    d = _folder(tmp_path, DIA["OPT"] + DIA["SP"])
    db = tmp_path / "materials.db"
    _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register")
    counts = {t: _q(db, f"SELECT count(*) FROM {t}")[0][0] for t in ("materials", "calculations", "properties")}
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups" / "notes.txt").write_text("mine")
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--backup")
    (b,) = (tmp_path / "backups").glob("materials_backup_*.db")
    assert _q(b, "PRAGMA integrity_check")[0][0] == "ok"
    assert {t: _q(b, f"SELECT count(*) FROM {t}")[0][0] for t in counts} == counts
    pruned = []
    for _ in range(3):
        p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--backup")
        pruned += re.findall(r"Pruned old backup (\S+)", p.stdout + p.stderr)
    assert len(list((tmp_path / "backups").glob("materials_backup_*.db"))) == 3
    assert pruned and (tmp_path / "backups" / "notes.txt").read_text() == "mine"
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--keep-backups", "2", ok=False)
    assert p.returncode == 2


def test_skip_unchanged_extracts_only_the_changed_file(tmp_path):
    d = _folder(tmp_path, sum(DIA.values(), []))
    db = tmp_path / "materials.db"
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--skip-unchanged",
             "--no-register", ok=False)
    assert p.returncode == 2 and "add --register" in p.stderr
    _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register")
    counts = [_q(db, f"SELECT count(*) FROM {t}")[0][0] for t in ("materials", "calculations", "properties")]
    with open(d / "1_dia_opt_rev1.out", "a") as fh:
        fh.write("\n")
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register",
             "--skip-unchanged", "--progress-json")
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    done = [x for x in lines if x.get("done")]
    per = [x for x in lines if "i" in x]
    assert len(done) == 1 and [x["file"] for x in per if x["status"] == "extracted"] == [str(d / "1_dia_opt_rev1.out")]
    after = [_q(db, f"SELECT count(*) FROM {t}")[0][0] for t in ("materials", "calculations", "properties")]
    assert all(a >= b for a, b in zip(after, counts))


def test_progress_json_stdout_is_only_json(tmp_path):
    d = _mixed(tmp_path)
    db = tmp_path / "materials.db"
    p = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(db), "--register", "--backup",
             "--progress-json")
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    docs = [json.loads(x) for x in lines]                        # every stdout line parses
    assert docs[-1]["done"] is True and len(docs) == len(list(d.glob("*.out"))) + 1
    assert "Processing complete" in p.stderr


def test_without_the_new_flags_stdout_matches_the_base_tree(tmp_path, base_tree):
    d = _folder(tmp_path, sum(DIA.values(), []))
    a = _cli(base_tree, tmp_path, "--extract-properties", str(d), "--db-path", str(tmp_path / "a.db")).stdout
    # (the base tree's plain run is untracked: --no-register here, the default being to register)
    b = _cli(REPO_ROOT, tmp_path, "--extract-properties", str(d), "--db-path", str(tmp_path / "b.db"),
             "--no-register").stdout

    def norm(t):
        t = re.sub(r"\d{4}-\d\d-\d\dT[\d:.]+", "<T>", t)
        # property counts grow by the properties C11 added (checked below)
        t = re.sub(r"(Extracted and saved|Total properties extracted:) \d+", r"\1 <N>", t)
        # the advanced analysis' classification of diamond: the base tree's
        # BAND.DAT edges (by eigenvalue sign) and DOSS.DAT rows were misread
        # ("semimetal"); a3308d9d corrected it (checked below)
        t = re.sub(r"(Advanced analysis:) \S+ (classification)", r"\1 <C> \2", t)
        # the quick effective-mass estimate is named as one now (beside the accurate m_c / m_d)
        t = t.replace("📊 Real effective masses calculated from band structure",
                      "📊 Effective mass, quick estimate (one band along the path): real_*_effective_mass; "
                      "the accurate m_c / m_d are effective_mass_*")
        # without ase the base tree printed its warning on stdout; it is on stderr now
        # this tree's note on outputs without a calculation record is new
        return [ln for ln in t.splitlines() if "Materials processed" not in ln and "ASE not available" not in ln
                and "no calculation record" not in ln]
    assert norm(a) == norm(b)
    assert "Advanced analysis: insulator classification" in b
    assert "Advanced analysis: semimetal" not in b
    total = lambda t: int(re.search(r"Total properties extracted: (\d+)", t).group(1))   # noqa: E731
    added = _q(tmp_path / "b.db", f"SELECT count(*) FROM properties WHERE property_name IN ({NEW_MARKS})", *NEW_PROPS)[0][0]
    # rows of the properties this tree no longer stores (DROPPED_PROPS): in the
    # base tree's database but not in this one
    dropped = lambda db: _q(db, f"SELECT count(*) FROM properties WHERE {DROPPED}", *DROPPED_PROPS)[0][0]  # noqa: E731
    lost = dropped(tmp_path / "a.db") - dropped(tmp_path / "b.db")
    # rows of the doubled geometry keys (RENAMED_PROPS): each one is gone where the
    # output also printed its single-prefix name, else it is printed under that name
    doubled = lambda db: _q(db, f"SELECT count(*) FROM properties WHERE {RENAMED}", *RENAMED_PROPS)[0][0]  # noqa: E731
    renamed = doubled(tmp_path / "a.db") - doubled(tmp_path / "b.db")
    # the outputs of one material share its untracked rows now (one id), so the
    # database holds each new property once while every output printed it;
    # and the base tree's database holds a dropped or renamed key once per id
    # while up to every output printed it (1.1.6 renamed and merged many)
    assert total(a) - (lost + renamed) * len(DIA) + added <= total(b) <= total(a) - lost + added * len(DIA)


def test_register_from_a_relative_folder_records_absolute_paths(tmp_path):
    """The GUI runs `--extract-properties runs --db-path runs/materials.db`
    from the served folder: the registered calculations must name their
    files so that the Database view (relative to the database) and a later
    run from another folder both find them."""
    d = _folder(tmp_path, DIA["OPT"] + DIA["SP"])
    _cli(REPO_ROOT, tmp_path, "--extract-properties", "runs", "--db-path", "runs/materials.db", "--register")
    rows = _q(d / "materials.db", "SELECT input_file, output_file, work_dir FROM calculations")
    assert len(rows) == 2
    for row in rows:
        for p in row:
            assert os.path.isabs(p) and Path(p).exists(), row
    # from another folder, by absolute paths: the same calculations, nothing new
    doc = _dry(tmp_path / "runs", d, d / "materials.db")
    assert set(_by_name(doc).values()) == {"unchanged"}, doc
