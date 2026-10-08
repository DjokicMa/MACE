"""Plan E1/E2/E14 through the real server (`mace_cli gui` as a subprocess):
file kinds and calculation types in /api/files?details=1, molecule and
properties outputs, job logs (never parsed for geometry; linked to their
calculation), the info document's payload, and the job-log database lookup
under a concurrent writer.

Real corpus files are copied under neutral names into a temp folder
(skipped without the gitignored test/ corpus); the database test uses the
shipped example database."""
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytest.importorskip("numpy")
pytest.importorskip("ase")
CORPUS = H.REPO_ROOT / "test"
MOL = "1LiFSI-1DEC-conf{}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
FREQ_SRC = CORPUS / "FREQ" / (MOL.format(1) + "_freq_HSESOL3C_optimized.out")
OPT_SRC = CORPUS / "OPT" / (MOL.format(3) + ".out")
DOSS_STEM = CORPUS / "DOSS" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss"
CUBE_SRC = CORPUS / "ECH3POT3" / "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential_DENS.CUBE"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not FREQ_SRC.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("parity")
    root = H.examples_copy(tmp)
    m = root / "mixed"
    m.mkdir()
    shutil.copyfile(FREQ_SRC, m / "mol_freq.out")
    for suffix in ("_IRSPEC.DAT", "_RAMSPEC.DAT"):
        src = FREQ_SRC.with_name(FREQ_SRC.stem + suffix)
        if src.is_file():
            shutil.copyfile(src, m / ("mol_freq" + suffix))
    log = next(FREQ_SRC.parent.glob(MOL.format(1) + "_freq_HSESOL3C_optimized_temp-*.o"))
    shutil.copyfile(log, m / "mol_freq_temp-1035400.o")
    shutil.copyfile(OPT_SRC, m / "mol_opt.out")
    for ext in (".out", ".d3", ".DOSS.DAT", ".f25", ".f9", ".sh"):
        shutil.copyfile(Path(str(DOSS_STEM) + ext), m / f"dia_doss{ext}")
    shutil.copyfile(CORPUS / "BAND" / "ipBANDS_2151873.out", m / "bands_2151873.out")
    shutil.copyfile(CORPUS / "BAND" / "ipBANDS_2151873.err", m / "bands_2151873.err")
    shutil.copyfile(CUBE_SRC, m / "DENS_CUBE.DAT")
    (m / "x_CUBE.DAT").write_text("not a cube\n")
    # a CRYSTAL output that died before printing a geometry: the first 120 lines of a real one
    (m / "died.out").write_text("\n".join(OPT_SRC.read_text().splitlines()[:120]) + "\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_MAX_WORKERS="2")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)


def result(base, path, timeout=120):
    st, _, r = U.req(base, path)
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                return r
            time.sleep(0.1)
        raise AssertionError("the task never finished")
    assert st == 200, r
    return r


def test_listing_and_details_by_content(served):
    root, base, _ = served
    st, _, listing = U.req(base, "api/files")
    kinds = {f["path"]: f["kind"] for f in listing["files"] if f["path"].startswith("mixed/")}
    assert kinds["mixed/mol_freq.out"] == "out" and kinds["mixed/mol_freq_temp-1035400.o"] == "out"
    assert kinds["mixed/DENS_CUBE.DAT"] == "cube" and "mixed/x_CUBE.DAT" not in kinds
    assert kinds["mixed/dia_doss.f9"] == "wavefunction" and kinds["mixed/dia_doss.sh"] == "script"
    assert kinds["mixed/dia_doss.f25"] == "f25" and kinds["mixed/bands_2151873.err"] == "log"
    t0 = time.perf_counter()
    st, _, d = U.req(base, "api/files?details=1")
    first = time.perf_counter() - t0
    det = d["details"]
    assert (det["mixed/mol_freq.out"]["role"], det["mixed/mol_freq.out"]["calc"], det["mixed/mol_freq.out"]["dim"]) == \
        ("crystal_out", "FREQ", 0)
    assert det["mixed/mol_freq.out"]["kind"] == "freq" and det["mixed/mol_freq.out"]["n"] == 28
    assert (det["mixed/mol_opt.out"]["calc"], det["mixed/mol_opt.out"]["dim"]) == ("OPT", 0)
    assert (det["mixed/dia_doss.out"]["role"], det["mixed/dia_doss.out"]["calc"]) == ("properties_out", "DOSS")
    for p in ("mixed/mol_freq_temp-1035400.o", "mixed/bands_2151873.out", "mixed/bands_2151873.err"):
        assert det[p]["role"] == "joblog", (p, det[p])
        assert "formula" not in det[p]                       # never parsed for geometry
    assert det["mixed/died.out"]["nogeom"] is True
    assert det["outputs/Si_sp.out"]["calc"] == "SP" and det["outputs/Si_sp.out"]["formula"]
    # the second listing is served from the parse cache
    t0 = time.perf_counter()
    st, _, d2 = U.req(base, "api/files?details=1")
    assert d2["details"] == det
    assert time.perf_counter() - t0 < max(0.5, first)


def test_filekind_route(served):
    _, base, _ = served
    st, _, r = U.req(base, "api/filekind?path=mixed/mol_freq.out")
    assert st == 200 and (r["role"], r["calc"], r["freq"], r["nogeom"], r["dim"]) == ("crystal_out", "FREQ", True, False, 0)
    st, _, r = U.req(base, "api/filekind?path=mixed/DENS_CUBE.DAT")
    assert (r["kind"], r["cube_type"], r["cube_ok"]) == ("cube", "density", True)
    st, _, r = U.req(base, "api/filekind?path=mixed/died.out")
    assert r["nogeom"] is True


def test_molecule_freq_modes_in_the_drawn_frame(served):
    pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
    _, base, _ = served
    r = result(base, "api/freq?path=mixed/mol_freq.out")
    assert r["n_atoms"] == r["mode_atoms"] == 28 and len(r["modes"]) == 84 and r["dim"] == 0
    st, _, m = U.req(base, "api/freq/mode?path=mixed/mol_freq.out&k=10")
    assert st == 200 and m["frame"] == "gui"


def test_job_logs_and_dead_outputs_answer_409(served):
    _, base, _ = served
    for route in ("out/summary", "structure", "freq"):
        st, _, r = U.req(base, f"api/{route}?path=mixed/mol_freq_temp-1035400.o")
        assert st == 409 and r["error"] == "This is a job log, not a CRYSTAL output", (route, st, r)
    for route in ("out/summary", "structure"):
        st, _, r = U.req(base, f"api/{route}?path=mixed/died.out")
        assert st == 409 and r["error"].startswith("No geometry in this output"), (route, st, r)


def test_job_log_link(served):
    _, base, _ = served
    st, _, r = U.req(base, "api/joblog/link?path=mixed/mol_freq_temp-1035400.o")
    assert st == 200 and r["calc"] == "mixed/mol_freq.out" and r["job_id"] == "1035400" and r["how"] == "name"
    st, _, r = U.req(base, "api/joblog/link?path=mixed/bands_2151873.out")
    assert st == 200 and r["calc"] is None and r["job_id"] == "2151873"
    st, _, r = U.req(base, "api/joblog/link?path=mixed/bands_2151873.err")
    assert st == 200 and r["errors_first_line"].startswith("Traceback")
    st, _, r = U.req(base, "api/joblog/link?path=mixed/mol_freq.out")
    assert st == 409


def test_properties_and_info_documents(served):
    _, base, _ = served
    st, _, r = U.req(base, "api/props?path=mixed/dia_doss.out")
    assert st == 200 and r["calc"] == "DOSS" and r["status"] == "Finished"
    kinds = {f["name"]: f["kind"] for f in r["files"]}
    assert kinds == {"dia_doss.DOSS.DAT": "dos", "dia_doss.f25": "f25"}
    st, _, r = U.req(base, "api/fileinfo?path=mixed/dia_doss.f9")
    assert st == 200 and r["kind"] == "wavefunction" and r["sibling"] == "mixed/dia_doss.out" and r["size"] > 0


def test_reads_head_and_tail_only(served, tmp_path, monkeypatch):
    """A role on a 2 GB sparse file reads at most 320 KB (head + tail)."""
    from mace.gui import filekind
    big = tmp_path / "huge.out"
    with open(big, "wb") as fh:
        fh.write(b"submit directory: \n/scratch/x\n")
        fh.seek(2 * 1024 ** 3)
        fh.write(b"\n=== Callback Complete ===\n")
    read = {"n": 0}
    real_open = open

    class Counting:
        def __init__(self, fh):
            self.fh = fh

        def read(self, n=-1):
            b = self.fh.read(n)
            read["n"] += len(b)
            return b

        def __getattr__(self, k):
            return getattr(self.fh, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

    monkeypatch.setattr("builtins.open", lambda p, mode="r", *a, **k: Counting(real_open(p, mode, *a, **k))
                        if str(p) == str(big) else real_open(p, mode, *a, **k))
    assert filekind.refine(big)["role"] == "joblog"
    assert read["n"] <= 320 * 1024
    big.unlink()


def test_job_log_database_lookup_under_a_writer(tmp_path):
    """A writer commits in a loop while slurm-<id>.out is linked 200 times: every
    answer is the right calculation or "busy", never a wrong one or an exception,
    and the database is never opened immutable."""
    from mace.gui import joblink
    root = tmp_path / "root"
    root.mkdir()
    db = root / "materials.db"
    shutil.copyfile(H.EXAMPLES / "materials.db", db)
    (root / "calc.out").write_text("EEEEEEEEEE STARTING\n")
    with sqlite3.connect(db) as c:
        c.execute("PRAGMA journal_mode=WAL")
        cols = [r[1] for r in c.execute("PRAGMA table_info(calculations)")]
        mid = c.execute("SELECT material_id FROM materials LIMIT 1").fetchone()[0]
        c.execute("INSERT INTO calculations (calc_id, material_id, calc_type, created_at, slurm_job_id, output_file, work_dir)"
                  " VALUES ('c-link', ?, 'SP', '2026-01-01', '424242', 'calc.out', ?)", (mid, str(root)))
    assert "slurm_job_id" in cols
    log = root / "slurm-424242.out"
    log.write_text("srun: job 424242 queued\n")
    stop = threading.Event()

    def writer():
        k = 0
        while not stop.is_set():
            w = sqlite3.connect(db, timeout=10)
            w.execute("BEGIN IMMEDIATE")
            w.execute("INSERT INTO calculations (calc_id, material_id, calc_type, created_at) VALUES (?, ?, 'SP', 'x')",
                      (f"w-{k}", mid))
            time.sleep(0.002)
            w.commit()
            w.close()
            k += 1

    t = threading.Thread(target=writer)
    joblink.OPEN_LOG.clear()
    t.start()
    try:
        answers = [joblink.link(root, log) for _ in range(200)]
    finally:
        stop.set()
        t.join()
    for a in answers:
        assert a["busy"] or a["calc"] == "calc.out", a
    assert all(a["calc"] == "calc.out" for a in answers if not a["busy"])
    assert joblink.OPEN_LOG and set(joblink.OPEN_LOG) == {"ro"}
