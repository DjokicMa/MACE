"""F7/F8: job detail, efficiency, progress and ETA on REAL CRYSTAL outputs
(read-only copies from test/), walltime risk sentences and the recovery
config drift check."""
import glob
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import fake, use_env

from cluster_fixtures import TEST_DATA
from mace.cluster import jobinfo, progress as P, recoveryview

OPT_DIR = TEST_DATA / "OPT"
FREQ_DIR = TEST_DATA / "FREQ"
SUPERCELL = FREQ_DIR / "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out"


def need(path):
    if not Path(path).exists():
        pytest.skip("test/ data corpus not present")


def copy_lines(src, dst, n=None):
    lines = Path(src).read_text(errors="replace").splitlines(True)
    Path(dst).write_text("".join(lines if n is None else lines[:n]))
    return dst


def heads(lines):
    return [i for i, l in enumerate(lines) if P._POINT.search(l)]


# ------------------------------------------------------------------ F8-a
def _gate_cases():
    out = {"dev": [], "held": []}
    for f in sorted(glob.glob(str(OPT_DIR / "*.out"))):
        text = Path(f).read_text(errors="replace")
        if "OPT END" not in text:
            continue
        lines = text.splitlines()
        full = P._mini_parse(lines)["points"]
        n = len(full)
        if n < 8:
            continue
        reason = "all_criteria" if all(full[-1]["converged"].get(k) for k in P.CRITERIA) else "gradient_and_energy"
        t_end = P.timings(lines)[-1][2]
        half = "held" if int(hashlib.sha1(os.path.basename(f).encode()).hexdigest(), 16) % 2 else "dev"
        hs = heads(lines)
        for p in sorted({3, 5, 8, n - 2}):
            if p >= len(hs):
                continue
            sub = lines[:hs[p]]
            out[half].append((sub, n, t_end, reason))
    return out


def test_opt_eta_gate_points_quick():
    """F8-a, kept as a fast regression check (basename split, 4 cuts per
    run): the true remaining points lie inside the range in >= 60 % of the
    truncations that got one; "unknown" <= 50 %; time coverage >= 60 %.
    The family-split gate is test_cluster_eta.test_opt_eta_held_out_gate."""
    need(OPT_DIR)
    cases = _gate_cases()["held"]
    assert len(cases) >= 40
    ranged = cov = tr = tcov = 0
    by_reason = {}
    for sub, n, t_end, reason in cases:
        parsed = P._mini_parse(sub)                       # never raises
        eta = P.opt_eta(parsed, sub)
        assert eta["low_points"] is None or eta["high_points"] is None or eta["low_points"] <= eta["high_points"]
        if eta["low_points"] is None:
            assert eta["low_s"] is None and eta["basis"]
            continue
        ranged += 1
        true = n - len(parsed["points"])
        hit = eta["low_points"] <= true <= (eta["high_points"] if eta["high_points"] is not None else 10 ** 9)
        cov += hit
        r = by_reason.setdefault(reason, [0, 0])
        r[0] += 1
        r[1] += hit
        t_now = [t for (_, _, t) in P.timings(sub)][-1]
        if eta["low_s"] is not None:
            tr += 1
            tcov += eta["low_s"] <= t_end - t_now <= eta["high_s"]
    unknown = 1 - ranged / len(cases)
    print(f"\nF8-a held-out: {len(cases)} truncations, unknown {unknown:.0%}, points coverage {cov / ranged:.0%}, "
          f"time coverage {tcov / max(1, tr):.0%}; by end reason "
          + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_reason.items())))
    assert unknown <= 0.5
    assert cov / ranged >= 0.6
    assert tcov / tr >= 0.6


def test_truncated_opt_progress(tmp_path):
    """F13-d's file: a real OPT truncated at POINT 5 reads "Point 5 ..."."""
    src = sorted(OPT_DIR.glob("Ag1Cl5_sym*.out")) if OPT_DIR.exists() else []
    if not src:
        pytest.skip("test/ data corpus not present")
    lines = src[0].read_text(errors="replace").splitlines(True)
    hs = heads([l.rstrip("\n") for l in lines])
    out = tmp_path / "job.out"
    out.write_text("".join(lines[:hs[5]]))
    r = P.progress(out)
    assert r["mode"] == "opt" and r["done"] == 5 and r["summary"].startswith("Point 5")
    assert r["ended"] is None


# ------------------------------------------------------------------ F8-b
def test_half_line_and_replaced(tmp_path):
    need(OPT_DIR / "1_dia_opt_rev1.out")
    out = copy_lines(OPT_DIR / "1_dia_opt_rev1.out", tmp_path / "a.out", 760)
    whole = P.progress(out)
    with open(out, "a") as fh:
        fh.write(" MAX GRADIENT      0.0001")              # a half-written line
    half = P.progress(out)
    for k in ("mode", "done", "stage", "summary"):
        assert whole[k] == half[k]
    copy_lines(OPT_DIR / "1_dia_opt_rev1.out", out, 450)      # replaced by a new run
    assert P.progress(out)["done"] == 0
    src = Path(P.__file__).read_text()
    assert not re.search(r"sha1|md5|hashlib", src)


# ------------------------------------------------------------------ F8-c
def test_freq_after_central_point(tmp_path):
    need(SUPERCELL)
    lines = SUPERCELL.read_text().splitlines()
    cut = next(i for i, l in enumerate(lines) if "TOTGRA" in l and "TELAPSE" in l) + 1
    assert cut == 1021
    out = copy_lines(SUPERCELL, tmp_path / "f.out", cut)
    r = P.progress(out)
    assert r["mode"] == "freq" and r["stage"] == "displacements"
    assert (r["done"], r["total"]) == (0, 5)
    assert r["eta"]["low_s"] is None and r["eta"]["high_s"] is None
    # the central point's SCF + gradient (8 min), the same time its level shows; not the setup before it
    assert ("Central point done (8 min); 5 displacements remaining; time per displacement unknown until the "
            "first finishes (displaced cells have lower symmetry and usually cost several times more than the "
            "central point)") == r["freq"]["basis"]
    cen = next(lv for lv in r["levels"] if lv["id"] == "freq.central")
    assert cen["text"] == "Central point (SCF + gradient): done in 8 min"


def test_freq_stages_and_end(tmp_path):
    need(SUPERCELL)
    full = copy_lines(SUPERCELL, tmp_path / "full.out")
    r = P.progress(full)
    assert r["stage"] == "end" and r["ended"]
    lines = SUPERCELL.read_text().splitlines()
    table = next(i for i, l in enumerate(lines) if "MAX ABS(DGRAD)" in l and "TOTAL ENERGY" in l)
    after_table = next(i for i, l in enumerate(lines) if "GCALCO" in l and i > table)
    part = copy_lines(SUPERCELL, tmp_path / "part.out", after_table)
    r = P.progress(part)
    assert r["stage"] != "end" and r["ended"] is None and r["percent"] != 100
    assert r["done"] == 5


def test_freq_intensity_stage(tmp_path):
    need(FREQ_DIR)
    for out in sorted(FREQ_DIR.glob("*.out")):
        deck = out.with_suffix(".d12")
        if not deck.exists() or "INTRAMAN" not in deck.read_text(errors="replace"):
            continue
        lines = out.read_text(errors="replace").splitlines()
        table = [i for i, l in enumerate(lines) if "MAX ABS(DGRAD)" in l and "TOTAL ENERGY" in l]
        if not table:
            continue
        end = next((i for i in range(table[0], len(lines)) if not lines[i].strip() and i > table[0] + 3), None)
        if end is None:
            continue
        d = tmp_path / "x"
        d.mkdir(exist_ok=True)
        cp = copy_lines(out, d / out.name, end + 1)
        shutil.copy(deck, d / deck.name)
        r = P.progress(cp)
        assert r["stage"] == "intensities", (out.name, r["stage"])
        assert "intensities and frequencies being written" in r["summary"]
        return
    pytest.skip("no INTRAMAN FREQ with a displacement table in the corpus")


def test_freq_restart_counts_restored(tmp_path):
    need(FREQ_DIR)
    for out in sorted(FREQ_DIR.glob("*.out")):
        text = out.read_text(errors="replace")
        if "RESTORED FROM A PREV. HESS. MATRIX" not in text:
            continue
        r = P.progress(copy_lines(out, tmp_path / out.name))
        n = text.count("RESTORED FROM A PREV. HESS. MATRIX")
        assert r["freq"]["restored"] == n and r["done"] >= min(n, r["total"])
        return
    pytest.skip("no restarted FREQ in the corpus")


def test_freq_corpus_ratio_and_no_central_extrapolation(tmp_path):
    """Per-displacement / central time ratios of the complete corpus FREQs
    are recorded; no truncated FREQ shows a time derived from the central
    point."""
    need(FREQ_DIR)
    ratios = []
    for out in sorted(FREQ_DIR.glob("*.out")):
        text = out.read_text(errors="replace")
        m = P._NCALC.search(text)
        if not m or int(m.group(1)) < 2:
            continue
        lines = text.splitlines()
        tl = P.timings(lines)
        cen = next(((i, t) for i, n, t in tl if n == "TOTGRA"), None)
        table = next((i for i, l in enumerate(lines) if "MAX ABS(DGRAD)" in l and "TOTAL ENERGY" in l), None)
        if not cen or table is None or cen[1] <= 0:
            continue
        after = next((t for i, n, t in tl if i > table), None)
        if after is None or "RESTORED" in text:
            continue
        ratios.append((out.name[:40], round((after - cen[1]) / (int(m.group(1)) - 1) / cen[1], 2)))
        r = P.progress(copy_lines(out, tmp_path / "t.out", cen[0] + 1))
        assert r["eta"]["low_s"] is None and r["eta"]["high_s"] is None, out.name
    print("\nFREQ per-displacement / central-point time ratios:", ratios)
    sup = [r for n, r in ratios if n.startswith("1_dia_opt_rev1_freq_B3LYP")]
    if sup:
        assert sup[0] == pytest.approx(6.1, abs=0.1)


# ------------------------------------------------------------------ F8-d / F7
def _job_dir(tmp_path, retries=2):
    need(OPT_DIR)
    src = sorted(OPT_DIR.glob("Ag1Cl5_sym*.out"))[0]
    d = tmp_path / "proj"
    d.mkdir()
    lines = src.read_text(errors="replace").splitlines(True)
    hs = heads([l.rstrip("\n") for l in lines])
    (d / "mat1_opt.out").write_text("".join(lines[:hs[5]]))
    shutil.copy(src.with_suffix(".d12"), d / "mat1_opt.d12")
    (d / "recovery_config.yaml").write_text(f"error_recovery:\n  timeout_error:\n    max_retries: {retries}\n")
    return d


def _db(d, job_ids, spent=0):
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(d / "materials.db"))
    db.create_material("mat1", "AgCl5")
    for k in range(spent):
        c = db.create_calculation("mat1", "OPT", input_file=str(d / "mat1_opt.d12"), work_dir=str(d))
        db.update_calculation_status(c, "resubmitted", slurm_job_id=f"39{k}", error_type="timeout_error")
    for jid in job_ids:
        c = db.create_calculation("mat1" if jid == job_ids[0] else "mat1", "OPT", input_file=str(d / "mat1_opt.d12"),
                                  work_dir=str(d))
        db.update_calculation_status(c, "running", slurm_job_id=jid)


def _fake_job(tmp_path, monkeypatch, d, other=True, limit="0:30:00", elapsed="0:28:00"):
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    monkeypatch.delenv("MACE_GUI_TEST_HOOKS", raising=False)
    fs.add_record(id="4001", state="RUNNING", name="mat1_opt", workdir=str(d), time_limit=limit, elapsed=elapsed,
                  cpus=32, partition="general-long")
    if other:
        fs.add_record(id="4002", state="RUNNING", name="mat2_opt", workdir=str(d))
    fs.update(job_text={"4001": f"JobId=4001 JobName=mat1_opt UserId=user1(1001) GroupId=g(1) JobState=RUNNING "
                                 f"Reason=None RunTime=00:28:00 TimeLimit=00:30:00 StartTime=2030-01-01T00:00:00 "
                                 f"Partition=general-long NodeList=cpu-001 NumNodes=1 NumCPUs=32 "
                                 f"WorkDir={d} StdOut={d}/mat1_opt-4001.o StdErr={d}/mat1_opt-4001.o "
                                 f"Command={d}/mat1_opt.sh Account=acct1 QOS=normal"})
    return fs


@pytest.mark.parametrize("other, spent, retries, expect", [
    (True, 0, 2, "when the queue manager next runs"),
    (False, 0, 2, "nothing will notice it automatically"),
    (True, 2, 2, "MACE already restarted it twice; a further timeout is not retried."),
    (True, 0, 1, "attempt 1 of 1"),
])
def test_risk_sentences(tmp_path, monkeypatch, other, spent, retries, expect):
    """F8-d: the sentence follows the job folder's recovery_config.yaml (the
    test's cwd is elsewhere) and whether another tracked job is queued."""
    d = _job_dir(tmp_path, retries)
    _db(d, ["4001"] + (["4002"] if other else []), spent)
    _fake_job(tmp_path, monkeypatch, d, other=other)
    monkeypatch.chdir(tmp_path)
    r = jobinfo.job("4001")
    risk = r["progress"]["risk"]
    assert risk["level"] in ("tight", "likely to exceed"), r["progress"]["eta"]
    assert expect in risk["sentence"], risk["sentence"]
    assert risk["config"]["source"] == "job folder"


def test_untracked_sentence(tmp_path, monkeypatch):
    d = _job_dir(tmp_path)
    _fake_job(tmp_path, monkeypatch, d)
    r = jobinfo.job("4001")
    assert "This job is not tracked" in r["progress"]["risk"]["sentence"]


def test_ended_job_from_accounting(tmp_path, monkeypatch):
    """A job that ended (scontrol forgot it, squeue no longer lists it): the
    Job document still names it and finds its output through sacct (seen on
    HPCC: name, CPUs, node and files were empty for a finished job)."""
    d = tmp_path / "jobs"
    d.mkdir()
    (d / "tiny.out").write_text("x\n")
    line = "|".join(["5001", "tiny", "COMPLETED", "00:00:09", "00:15:00", "4", "8G", "", "00:11.317",
                     "billing=5,cpu=4,mem=8G,node=1", "n01", "0:0", "2030-01-01T00:00:00", "2030-01-01T00:00:09", str(d)])
    fs = fake(tmp_path, sacct_rows={"5001": [line]})
    use_env(monkeypatch, fs)
    r = jobinfo.job("5001")
    assert r["state"] == "COMPLETED" and r["name"] == "tiny" and r["ncpus"] == "4" and r["nodelist"] == "n01"
    assert r["start"] == "2030-01-01T00:00:00"
    assert r["files"]["workdir"] == str(d) and r["files"]["out"] == str(d / "tiny.out")


def test_not_yours_and_bad_ids(tmp_path, monkeypatch):
    """F7-b (unit side): another user's id and malformed ids."""
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    with pytest.raises(jobinfo.NotYours):
        jobinfo.job("777")
    with pytest.raises(ValueError):
        jobinfo.job("12_3;rm")


def test_efficiency_plausibility():
    """F7-a: TotalCPU 00:00:00 on the allocation line and a value on .batch;
    an implausible value is hidden."""
    rows = {"4100": {"JobID": "4100", "Elapsed": "01:00:00", "NCPUS": "4", "TotalCPU": "00:00:00",
                     "MaxRSS": "", "ReqMem": "16G", "State": "TIMEOUT"},
            "4100.batch": {"JobID": "4100.batch", "Elapsed": "01:00:00", "NCPUS": "4", "TotalCPU": "03:00:00",
                           "MaxRSS": "476304K", "ReqMem": "16G"}}
    e = jobinfo.efficiency(rows, "4100")
    assert e["cpu"] == 75.0 and e["memory"] == pytest.approx(2.8, abs=0.1)
    rows["4100.batch"]["TotalCPU"] = "10:00:00"               # more than 4 cores x 1 h: implausible
    rows["4100.batch"]["MaxRSS"] = ""
    e = jobinfo.efficiency(rows, "4100")
    assert e["cpu"] is None and e["cpu_note"] == jobinfo.NOT_MEASURED and e["memory"] is None


def test_array_task_deck_from_manifest(tmp_path):
    """F7-c: an array task's deck comes from a manifest written by
    job_array.write_manifest."""
    from mace.submission import job_array
    adir = tmp_path / ".mace_arrays" / "arr_0"
    adir.mkdir(parents=True)
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    members = [job_array.ArrayMember(dir=str(m1), job="m1_sp", script=str(m1 / "m1_sp.sh"), calc_id=None),
               job_array.ArrayMember(dir=str(m2), job="m2_sp", script=str(m2 / "m2_sp.sh"), calc_id=None)]
    job_array.write_manifest(adir / "manifest.tsv", members)
    files = jobinfo._files({"WorkDir": str(tmp_path), "JobName": "mace_arr_x_0", "Command": str(adir / "arr_0.sh")},
                           None, "4242_1")
    assert files["out"] == str(m2 / "m2_sp.out") and files["deck_stem"] == "m2_sp"


# ------------------------------------------------------------------ F8-f
def test_recovery_config_drift(tmp_path, monkeypatch):
    """F8-f: the built-in copy equals the engine's defaults and the packaged
    YAML; a job-folder file is merged like the engine; without PyYAML the
    built-in values come back with the note."""
    import yaml
    from mace.recovery.recovery import ErrorRecoveryEngine
    import types
    eng = ErrorRecoveryEngine.load_recovery_config(types.SimpleNamespace(
        config_path=None, known_handlers=ErrorRecoveryEngine.known_handlers))
    assert eng["error_recovery"]["timeout_error"] == recoveryview.BUILTIN_TIMEOUT
    packaged = yaml.safe_load(recoveryview.PACKAGED.read_text())["error_recovery"]["timeout_error"]
    assert {k: v for k, v in packaged.items() if k != "description"} == recoveryview.BUILTIN_TIMEOUT
    d = tmp_path / "job"
    d.mkdir()
    (d / "recovery_config.yaml").write_text("error_recovery:\n  timeout_error:\n    max_retries: 1\n")
    a = recoveryview.timeout_config(d, use_engine=True)
    b = recoveryview.timeout_config(d, use_engine=False)
    assert a["via"] == "engine" and b["via"] == "yaml"
    assert a["values"]["max_retries"] == b["values"]["max_retries"] == 1
    assert {k: a["values"][k] for k in recoveryview.BUILTIN_TIMEOUT} == b["values"]
    monkeypatch.setitem(sys.modules, "yaml", None)
    c = recoveryview.timeout_config(d, use_engine=False)
    assert c["values"] == recoveryview.BUILTIN_TIMEOUT and "PyYAML is not installed" in c["note"]


def test_freq_rows_are_the_progress(tmp_path):
    """The displacement table's header and central row come right after the
    central point (seen on a live cluster run); each finished displacement
    adds a row. 0 of 5 after the header, 2 of 5 after two rows; a time range
    only from the displacements finished in this run."""
    need(SUPERCELL)
    lines = SUPERCELL.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if "MAX ABS(DGRAD)" in l and "TOTAL ENERGY" in l)
    central = head + 1
    r = P.progress(copy_lines(SUPERCELL, tmp_path / "a.out", central + 1))
    assert (r["stage"], r["done"], r["total"]) == ("displacements", 0, 5)
    assert r["eta"]["low_s"] is None
    rows = [i for i in range(central + 1, len(lines)) if P._DISP_ROW.match(lines[i])]
    second = [i for i in rows if P._f(P._DISP_ROW.match(lines[i]).group(1)) is not None][1]
    out = copy_lines(SUPERCELL, tmp_path / "b.out", second + 1)
    r = P.progress(out)
    assert (r["stage"], r["done"], r["percent"]) == ("displacements", 2, 40)
    assert r["eta"]["low_s"] is None                         # no job start known: no time
    st = os.stat(out)
    start = st.st_mtime - (555.03 + 2 * 3400)              # as if two displacements took 3400 s each
    fp = P.freq_progress(out.read_text(), out.read_text().splitlines(), False, mtime=st.st_mtime, job_start=start)
    assert fp["per_displacement_s"] == pytest.approx(3400, rel=0.01)
    assert fp["low_s"] == pytest.approx(3 * 3400 * 0.75, rel=0.01) and fp["high_s"] == pytest.approx(3 * 3400 * 1.5, rel=0.01)


def test_db_in_a_folder_with_uri_characters(tmp_path):
    """materials.db in a folder whose name holds '#', '?' or '%' is still
    read (read-only), not a missing or wrong file."""
    from mace.cluster import progress
    d = tmp_path / "run #2?x%41"
    d.mkdir()
    _db(d, ["4001"], spent=1)
    tracked, spent, ctype = progress.retries_spent(d / "materials.db", "4001", "mat1_opt")
    assert tracked is True and spent == 1 and ctype == "OPT"
