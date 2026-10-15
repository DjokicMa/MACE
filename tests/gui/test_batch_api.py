"""Batch Setup through the real `mace gui` (PLAN_submit D9-D14).

D9 filters and eligibility on real output copies; D10 follow-ups keep each
parent's settings (the template changes only what it names); D11 an
override touches its own file only; D12 `bash commands.sh` re-creates every
deck and settings file byte for byte, and template.json loads in `mace
convert`; D12b hostile names are refused and never executed; D13 nothing is
written outside the batch folder; D14 per-file diffs and refused rows.

Real outputs are copied from the read-only corpus (test/) at run time; the
test skips without it."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import submit_util as SU
from conftest import TEST_DATA

pytest.importorskip("ase")          # mace convert and the follow-ups read structures with ase

PARENTS = ["OPT/1_dia_opt_rev1", "OPT/2_dia2_opt_rev1", "OPT/3_dia3_opt_rev1"]


def _corpus(*names):
    if not (TEST_DATA / "OPT").is_dir():
        pytest.skip("test/ data corpus not present")
    return names


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    _corpus()
    s = SU.Served(tmp_path_factory.mktemp("batch"))
    runs = s.root / "runs"
    runs.mkdir()
    for p in PARENTS:
        for ext in (".out", ".d12", ".f9"):
            shutil.copy2(TEST_DATA / f"{p}{ext}", runs)
    for pat in ("OPT/Ag1Br1_sym_*_optimized", "OPT/EC_MOLECULE_*HSESOL3C*_optimized"):
        base = str(sorted(TEST_DATA.glob(pat + ".out"))[0])[:-len(".out")]
        for ext in (".out", ".d12", ".f9"):
            shutil.copy2(base + ext, runs)
    failed = s.root / "failed"
    failed.mkdir()
    for f in sorted((TEST_DATA / "FAILED_QA").glob("*.out"))[:4]:
        shutil.copy2(f, failed)
    cifs = s.root / "cifs"
    cifs.mkdir()
    for n in ("MgO", "NaCl", "LiF"):
        shutil.copy2(SU.H.EXAMPLES / "structures" / f"{n}.cif", cifs)
    yield s
    s.stop()


def _scan(s, folder, filt, fu=None, rec=False):
    st, r = s.task_result(s.req("api/batch/scan", {"folder": folder, "filter": filt, "followup_type": fu,
                                                    "recursive": rec, "wait": 60}))
    assert st == 200, r
    return {x["rel"]: x for x in r["rows"]}


def _plan(s, folder, files, calc, template=None, overrides=None, **kw):
    st, r = s.task_result(s.req("api/batch/plan", {"folder": folder, "files": files, "calc": calc,
                                                    "template_answers": template or {}, "overrides": overrides or {},
                                                    "wait": 1, **kw}), timeout=900)
    assert st == 200, r
    return r


def _write(s, bid, name=None, overwrite=False):
    body = {"batch_id": bid, "wait": 1, "overwrite": overwrite}
    if name:
        body["name"] = name
    r = s.req("api/batch/write", body)
    if r[0] in (400, 403, 409):
        return r[0], r[2]
    return s.task_result(r, timeout=900)


def _tree(d):
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- D9
def test_scan_cif_and_failed(srv):
    rows = _scan(srv, "cifs", "cif")
    assert sorted(rows) == ["cifs/LiF.cif", "cifs/MgO.cif", "cifs/NaCl.cif"] and all(r["eligible"] for r in rows.values())
    rows = _scan(srv, "failed", "failed")
    assert rows and all(r["bucket"] and not r["eligible"] for r in rows.values())
    assert all(r["recover"] is None for r in rows.values())            # untracked: no per-row Recover


def test_scan_followups_eligibility(srv):
    rows = _scan(srv, "runs", "followup", "SP")
    dia = "runs/1_dia_opt_rev1.out"
    assert rows[dia]["eligible"] and rows[dia]["calc_type"] == "OPT"
    # (a) a follow-up already beside its parent
    beside = srv.root / "runs" / "2_dia2_opt_rev1_sp_B3LYP-D3_optimized.d12"
    beside.write_text("x")
    # (b) an earlier batch maps the parent to an SP deck
    eb = srv.root / "runs" / "batch_SP_2026-01-01"
    (eb / "decks").mkdir(parents=True)
    (eb / "decks" / "3_dia3_sp.d12").write_text("x")
    (eb / "batch.json").write_text(json.dumps({"calc": "SP", "parents": {
        "runs/3_dia3_opt_rev1.out": "runs/batch_SP_2026-01-01/decks/3_dia3_sp.d12"}}))
    try:
        rows = _scan(srv, "runs", "followup", "SP")
        assert not rows["runs/2_dia2_opt_rev1.out"]["eligible"]
        assert "already beside" in rows["runs/2_dia2_opt_rev1.out"]["why_not"]
        assert rows["runs/3_dia3_opt_rev1.out"]["why_not"] == "In batch_SP_2026-01-01 (written)"
        # FREQ only from a converged OPT; BAND needs the wavefunction
        rows = _scan(srv, "runs", "followup", "FREQ")
        assert rows[dia]["eligible"]
        rows = _scan(srv, "runs", "followup", "BAND")
        assert rows[dia]["eligible"]
    finally:
        beside.unlink()
        shutil.rmtree(eb)


def test_scan_db_child_and_workflow_plan(srv, tmp_path):
    sys.path.insert(0, str(SU.REPO))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(srv.root / "runs" / "materials.db"))
    parent = str(srv.root / "runs" / "1_dia_opt_rev1.out")
    db.create_material(material_id="dia1", formula="C2", source_type="d12", source_file=parent)
    c1 = db.create_calculation(material_id="dia1", calc_type="OPT", input_file=parent.replace(".out", ".d12"),
                               work_dir=str(srv.root / "runs"))
    db.update_calculation_status(c1, "completed", output_file=parent)
    c2 = db.create_calculation(material_id="dia1", calc_type="SP", input_file=str(srv.root / "runs" / "x_sp.d12"),
                               work_dir=str(srv.root / "runs"))
    db.update_calculation_status(c2, "submitted", slurm_job_id="123")
    try:
        rows = _scan(srv, "runs", "followup", "SP")
        assert "The database has a SP of this material" in rows["runs/1_dia_opt_rev1.out"]["why_not"]
        # a plan written by `mace submit --progress opt_sp` whose next step is SP
        r = SU.twin_submit(tmp_path, srv.fake, srv.root / "runs", ["submit", "--progress", "opt_sp", "2_dia2_opt_rev1.d12"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rows = _scan(srv, "runs", "followup", "SP")
        assert rows["runs/2_dia2_opt_rev1.out"]["why_not"].startswith("Workflow ")
        assert rows["runs/2_dia2_opt_rev1.out"]["why_not"].endswith("will run SP next")
    finally:
        for f in ["materials.db", "structures.db"]:
            p = srv.root / "runs" / f
            if p.exists():
                p.unlink()
        for d in ("workflow_configs", "workflow_scripts", "workflow_outputs", "workflow_inputs", "workflow_temp"):
            shutil.rmtree(srv.root / "runs" / d, ignore_errors=True)
        for p in (srv.root / "runs").glob(".mace_context_*"):
            shutil.rmtree(p, ignore_errors=True)
        for p in (srv.root / "runs").glob("2_dia2_opt_rev1.sh"):
            p.unlink()
        for p in (srv.root / "runs").glob("crystal_job_status.json"):
            p.unlink()


# ---------------------------------------------------------------- D10, D11, D14
def test_followups_preserve_parent_and_override_isolated(srv):
    files = sorted(x.relative_to(srv.root).as_posix() for x in (srv.root / "runs").glob("*.out"))
    files = [f for f in files if "dia2" not in f and "dia3" not in f]          # 1_dia, Ag1Br1, EC
    assert len(files) == 3
    r = _plan(srv, "runs", files, "SP", {"maxcycle": 321})
    by = {f["rel"]: f for f in r["files"]}
    assert all(f["status"] == "ok" for f in by.values()), r
    bid = r["batch_id"]
    for rel in files:
        st, _, d = srv.req(f"api/batch/diff?batch_id={bid}&rel={rel}&against=default")
        changed = [l for l in d["diff"].splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        # only MAXCYCLE changes: from the parent's own value (kept by default) to the template's
        assert len(changed) == 2 and "+321" in changed and [c for c in changed if c != "+321"][0][1:].isdigit(), \
            (rel, changed)
        st, _, deck = srv.req(f"api/batch/deck?batch_id={bid}&rel={rel}")
        parent_d12 = (srv.root / rel).with_suffix(".d12").read_text()
        # each follow-up keeps its own parent's functional
        func = [l for l in parent_d12.splitlines() if l.strip() in ("B3LYP", "HSESOL3C", "PBE0", "HSE06")]
        if func:
            assert func[0] in deck["deck"].splitlines(), rel
    # an override changes only its file
    ov = {files[0]: {"maxcycle": 999}}
    r2 = _plan(srv, "runs", files, "SP", {"maxcycle": 321}, ov, batch_id=bid)
    by2 = {f["rel"]: f for f in r2["files"]}
    assert by2[files[0]]["deck_sha"] != by[files[0]]["deck_sha"] and by2[files[0]]["override"]
    for rel in files[1:]:
        assert by2[rel]["deck_sha"] == by[rel]["deck_sha"]
    st, _, d = srv.req(f"api/batch/diff?batch_id={bid}&rel={files[1]}&against=template")
    assert st == 200 and d["against"] == "template"


# ---------------------------------------------------------------- D12, D13
def _reproduce(srv, out, tmp_path, keep_inputs=True):
    """bash commands.sh in a fresh twin of the folder holding the batch."""
    B = srv.root / out["folder_rel"]
    cwd = B.parent
    twin = tmp_path / "twin"
    shutil.copytree(cwd, twin, ignore=shutil.ignore_patterns("decks"))
    (twin / B.name / "decks").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH")}
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "mace").write_text(f"#!/bin/bash\nexec {sys.executable} {SU.MACE_CLI} --no-banner \"$@\"\n")
    (bindir / "mace").chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    r = subprocess.run(["bash", str(twin / B.name / "commands.sh")], cwd=twin, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    return B, twin / B.name


def test_commands_sh_reproduces_cif_groups(srv, tmp_path):
    rows = _scan(srv, "cifs", "cif")
    r = _plan(srv, "cifs", sorted(rows), "SP", {"maxcycle": 400}, {"cifs/NaCl.cif": {"maxcycle": 410}})
    assert r["groups"] == 2
    st, out = _write(srv, r["batch_id"], "batch_SP_groups")
    assert st == 200 and out["code"] == 0, out
    assert len(out["commands"]) == 2 and all(c.startswith("mace convert --batch --options_file ") for c in out["commands"])
    B, T = _reproduce(srv, out, tmp_path)
    assert _tree(B / "decks") == _tree(T / "decks")
    assert _tree(B / "options") == _tree(T / "options")
    assert (B / "commands.sh").stat().st_mode & 0o777 == 0o644


def test_template_json_loads(srv, tmp_path):
    rows = _scan(srv, "cifs", "cif")
    r = _plan(srv, "cifs", sorted(rows), "SP", {"maxcycle": 420})
    st, out = _write(srv, r["batch_id"], "batch_tmpl")
    B = srv.root / out["folder_rel"]
    d = tmp_path / "manual"
    shutil.copytree(srv.root / "cifs", d / "cifs", ignore=shutil.ignore_patterns("batch_*"))
    p = subprocess.run([sys.executable, str(SU.MACE_CLI), "--no-banner", "convert", "--batch", "--options_file",
                        str(B / "options" / "template.json"), "--cif_dir", "cifs", "--output_dir", "o"], cwd=d,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:]
    assert _tree(d / "o") == _tree(B / "decks")


def test_commands_sh_reproduces_opt2d12_and_opt2d3(srv, tmp_path):
    files = ["runs/1_dia_opt_rev1.out", "runs/2_dia2_opt_rev1.out"]
    r = _plan(srv, "runs", files, "SP", {"maxcycle": 333})
    st, out = _write(srv, r["batch_id"], "batch_SP_follow")
    assert st == 200 and out["code"] == 0 and len(out["commands"]) == 2, out
    assert out["one_file_per_parent"]
    B, T = _reproduce(srv, out, tmp_path / "a")
    assert _tree(B / "decks") == _tree(T / "decks")
    r = _plan(srv, "runs", ["runs/1_dia_opt_rev1.out"], "BAND", {})
    assert r["files"][0]["status"] == "ok", r
    st, out = _write(srv, r["batch_id"], "batch_BAND_follow")
    assert st == 200 and out["code"] == 0, out
    B, T = _reproduce(srv, out, tmp_path / "b")
    assert _tree(B / "decks") == _tree(T / "decks")
    f9s = list((B / "decks").glob("*.f9"))
    assert f9s and f9s[0].stat().st_size == (srv.root / "runs" / "1_dia_opt_rev1.f9").stat().st_size


def test_write_stays_inside_and_refuses_existing(srv):
    before = {p for p in srv.root.rglob("*") if "batch_" not in p.as_posix()}
    rows = _scan(srv, "cifs", "cif")
    r = _plan(srv, "cifs", sorted(rows)[:1], "OPT")
    st, out = _write(srv, r["batch_id"], "batch_inside")
    assert st == 200
    after = {p for p in srv.root.rglob("*") if "batch_" not in p.as_posix()}
    assert after == before
    st, res = _write(srv, r["batch_id"], "batch_inside")
    assert st == 409 and "already exists" in res["error"]
    st, out = _write(srv, r["batch_id"], "batch_inside", overwrite=True)
    assert st == 200 and out["backups"]            # the replaced deck keeps a .bak


@pytest.mark.parametrize("name", ["a/b", "..", ".hidden", "x$y", "a b"])
def test_batch_name_refused(srv, name):
    rows = _scan(srv, "cifs", "cif")
    r = _plan(srv, "cifs", sorted(rows)[:1], "OPT")
    st, res = _write(srv, r["batch_id"], name)
    assert st == 400 and "batch folder name" in res["error"]


def test_hostile_names(srv, tmp_path):
    h = srv.root / "hostile"
    h.mkdir()
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    for n in ("$(touch PWNED).out", "a b.out"):
        shutil.copy2(src, h / n)
    rows = _scan(srv, "hostile", "followup", "SP")
    assert all(not r["eligible"] and "rename" in r["why_not"] for r in rows.values())
    st, _, res = srv.req("api/batch/plan", {"folder": "hostile", "files": ["hostile/$(touch PWNED).out"], "calc": "SP"})
    assert st == 400 and "rename" in res["error"]
    assert not list(srv.root.rglob("PWNED")) and not list(SU.REPO.glob("PWNED"))


def test_refused_rows_are_excluded(srv):
    bad = srv.root / "cifs" / "broken.cif"
    bad.write_text("data_x\n_cell_length_a 1\n")
    try:
        r = _plan(srv, "cifs", ["cifs/broken.cif", "cifs/MgO.cif"], "OPT")
        st = {f["rel"]: f["status"] for f in r["files"]}
        assert st["cifs/broken.cif"] == "refused" and st["cifs/MgO.cif"] == "ok" and r["refused"] == 1
        st2, out = _write(srv, r["batch_id"], "batch_refused")
        assert st2 == 200 and all("broken" not in w for w in out["written"])
    finally:
        bad.unlink()


def test_opt2d3_free_space_refused(monkeypatch, tmp_path):
    from mace.gui import batch
    class St:
        root = tmp_path
        def check_write(self, t): pass
        def rel_new(self, p): return str(p)
    b = batch.Batch("b1", 0, tmp_path, [], "opt2d3", "BAND")
    b.items = {"x.out": {"status": "ok", "f9": {"name": "x.f9", "bytes": 10 << 30}, "json_text": "{}", "rel": "x.out"}}
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 30})())
    with pytest.raises(batch.BatchError) as e:
        batch.write(St(), b, "batch_x", False)
    assert "wavefunction copies need" in str(e.value)
    assert not (tmp_path / "batch_x").exists()


@pytest.mark.parametrize("link", ["options", "commands.sh", "batch.json"])
def test_overwrite_never_writes_through_a_symlink(srv, tmp_path, link):
    """D13: Overwrite into an existing batch folder writes nothing through a
    symbolic link it holds (a folder or a file leading out of the root)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine\n")
    rows = _scan(srv, "cifs", "cif")
    r = _plan(srv, "cifs", sorted(rows)[:1], "OPT")
    B = srv.root / "cifs" / f"batch_link_{link.replace('.', '_')}"
    B.mkdir()
    if link == "options":
        (B / "options").symlink_to(outside, target_is_directory=True)
    else:
        (B / link).symlink_to(outside / "keep.txt")
    before = _tree(outside)
    st, res = _write(srv, r["batch_id"], B.name, overwrite=True)
    assert st != 200 and "symbolic link" in json.dumps(res), res
    assert _tree(outside) == before
    assert not (B / "decks").exists()               # refused before anything was written


def test_same_name_parents_in_subfolders_keep_their_own_files(srv, tmp_path):
    """Two parents named alike in different subfolders: each keeps its own
    settings file and deck (named after its subfolder), and commands.sh
    still re-creates both."""
    dup = srv.root / "dup"
    for sub, p in (("A", PARENTS[0]), ("B", PARENTS[1])):
        (dup / sub).mkdir(parents=True)
        for ext in (".out", ".d12", ".f9"):
            shutil.copy2(TEST_DATA / f"{p}{ext}", dup / sub / f"mat_opt{ext}")
    r = _plan(srv, "dup", ["dup/A/mat_opt.out", "dup/B/mat_opt.out"], "SP", {"maxcycle": 321})
    assert all(f["status"] == "ok" for f in r["files"]), r
    st, out = _write(srv, r["batch_id"], "batch_dup")
    assert st == 200 and out["code"] == 0, out
    B = srv.root / out["folder_rel"]
    opts = sorted(p.name for p in (B / "options").iterdir() if p.name != "template.json")
    assert len(opts) == 2 and opts[0].startswith("A_") and opts[1].startswith("B_"), opts
    decks = sorted(p.name for p in (B / "decks").glob("*.d12"))
    assert len(decks) == 2 and decks[0].startswith("A_") and decks[1].startswith("B_"), decks
    assert not list((B / "decks").glob("*.bak"))
    assert len(set(out["parents"].values())) == 2
    for rel, deck in out["parents"].items():          # each deck holds its own parent's geometry
        assert (srv.root / deck).name.startswith(Path(rel).parent.name + "_")
    a, b = ((B / "decks" / d).read_text() for d in decks)
    assert a != b
    T = _reproduce(srv, out, tmp_path)[1]
    assert _tree(B / "decks") == _tree(T / "decks")
    assert _tree(B / "options") == _tree(T / "options")


def test_same_name_cifs_in_subfolders(srv, tmp_path):
    dup = srv.root / "cdup"
    for sub, n in (("A", "MgO"), ("B", "NaCl")):
        (dup / sub).mkdir(parents=True)
        shutil.copy2(SU.H.EXAMPLES / "structures" / f"{n}.cif", dup / sub / "mat.cif")
    r = _plan(srv, "cdup", ["cdup/A/mat.cif", "cdup/B/mat.cif"], "OPT")
    st, out = _write(srv, r["batch_id"], "batch_cdup")
    assert st == 200 and out["code"] == 0, out
    B = srv.root / out["folder_rel"]
    decks = sorted(p.name for p in (B / "decks").glob("*.d12"))
    assert len(decks) == 2 and decks[0].startswith("A_") and decks[1].startswith("B_"), decks
    assert len(set(out["parents"].values())) == 2, out["parents"]
    T = _reproduce(srv, out, tmp_path)[1]
    assert _tree(B / "decks") == _tree(T / "decks")


def test_same_name_collision_refused_before_writing(tmp_path):
    """Names that still clash after the subfolder prefix (x.out and x.log in
    one folder) are refused before anything is written."""
    from mace.gui import batch

    class St:
        root = tmp_path
        def check_write(self, t): pass
        def rel_new(self, p): return str(p)
    b = batch.Batch("b1", 0, tmp_path, [], "opt2d12", "SP")
    b.items = {r: {"status": "ok", "json_text": "{}", "rel": r, "deck_name": "x_sp.d12", "group": None}
               for r in ("x.out", "x.log")}
    with pytest.raises(batch.BatchError) as e:
        batch.write(St(), b, "batch_x", False)
    assert "same name" in str(e.value)
    assert not (tmp_path / "batch_x").exists()


@pytest.mark.parametrize("calc", ["BAND", "DOSS", "TRANSPORT"])
def test_molecule_has_no_band_doss_or_transport(srv, calc):
    """CRYSTAL runs BAND and DOSS for periodic systems only (manual p.309, p.320):
    a MOLECULE parent is not eligible, and a plan with it is refused (opt2d3's words)."""
    rows = _scan(srv, "runs", "followup", calc)
    ec = next(r for r in rows if "EC_MOLECULE" in r)
    assert not rows[ec]["eligible"] and "molecule" in rows[ec]["why_not"], rows[ec]
    assert "molecule" not in rows["runs/1_dia_opt_rev1.out"]["why_not"]          # a crystal is not refused for it
    if calc == "BAND":
        r = _plan(srv, "runs", [ec], "BAND")
        f = r["files"][0]
        assert f["status"] == "refused" and "periodic structure" in f["message"], f
    rows = _scan(srv, "runs", "followup", "CHARGE")
    assert rows[ec]["eligible"], rows[ec]
