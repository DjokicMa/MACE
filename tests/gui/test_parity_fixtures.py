"""CI coverage of plan E without the research corpus: the generic CRYSTAL23 runs in
tests/fixtures/gui_parity (see its README) classify, open and link as the corpus
tests check on real research outputs. A gzipped fixture is decompressed into a
temporary folder; a browser smoke test (marked gui) opens the folder."""
import gzip
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
FIX = H.REPO_ROOT / "tests" / "fixtures" / "gui_parity"
LOG = ("submit directory: \n/scratch/someone/run\nFound queue manager at: /opt/mace/queue/manager.py\n\n"
       "=== Queue Manager Callback (completion) - 2026-01-01 00:00:00 ===\n=== Callback Complete ===\n")


def unpack(dst: Path) -> Path:
    """The fixtures as a folder of plain files (as the CLI leaves them), plus a job log and a broken cube."""
    for src in FIX.rglob("*"):
        if not src.is_file() or src.name == "README.md":
            continue
        rel = src.relative_to(FIX)
        out = dst / (str(rel)[:-3] if src.suffix == ".gz" else str(rel))
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".gz":
            out.write_bytes(gzip.decompress(src.read_bytes()))
        else:
            shutil.copyfile(src, out)
    (dst / "water_opt-123456.o").write_text(LOG)
    (dst / "x_CUBE.DAT").write_text("not a cube\n")
    return dst


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return unpack(tmp_path_factory.mktemp("gui_parity"))


def test_kinds_and_roles(folder):
    from mace.gui import filekind
    K = filekind.kind_by_name
    want = {"water_opt.out": ("crystal_out", "OPT", 0), "water_freq.out": ("crystal_out", "FREQ", 0),
            "mgo_doss.out": ("properties_out", "DOSS", None), "water_opt-123456.o": ("joblog", None, None)}
    for name, (role, calc, dim) in want.items():
        r = filekind.refine(folder / name)
        assert (r["role"], r["calc"], r["dim"]) == (role, calc, dim), (name, r)
    assert [K(n) for n in ("DENS_CUBE.DAT", "1_WAN.CUBE", "mgo_doss.f25", "mgo_doss.DOSS.DAT", "water_freq_IRSPEC.DAT")] == \
        ["cube", "cube", "f25", "dos", "ir"]
    assert filekind.cube_header_ok(folder / "mgo_dens" / "DENS_CUBE.DAT")
    assert not filekind.cube_header_ok(folder / "x_CUBE.DAT")
    assert filekind.cube_type("1_WAN.CUBE") == "orbital" and filekind.cube_type("DENS_CUBE.DAT") == "density"


def test_molecule_frames(folder):
    pytest.importorskip("ase")          # the frames are built with ase's cell geometry
    from mace.gui import outindex
    opt = outindex.build(folder / "water_opt.out")
    # six optimization points, two printed tables (plan F15); a C2v table lists all three atoms
    assert opt.dim == 0 and opt.labels == ["initial", "final"]
    assert (folder / "water_opt.out").read_text().count("OPTIMIZATION - POINT") >= 3
    s = opt.frame(-1)["structure"]
    assert len(s["numbers"]) == 3 and s["pbc"] == [False, False, False]
    oh = np.linalg.norm(np.asarray(s["cart"][1]) - np.asarray(s["cart"][0]))
    assert 0.9 < oh < 1.0
    freq = outindex.build(folder / "water_freq.out")
    assert freq.dim == 0 and freq.freq and freq.labels == ["initial", "final"]      # PREOPTGEOM: before and after


def test_spectrum_from_modes_matches_crystal(folder):
    pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
    from mace.gui import spectra
    from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
    from mace.plotting.handlers.spectra_api import read_spectrum
    parsed = {"modes": Crystal23FreqParser(str(folder / "water_freq.out")).modes}
    for kind, name, col in (("ir", "water_freq_IRSPEC.DAT", 2), ("raman", "water_freq_RAMSPEC.DAT", 1)):
        r = spectra.from_modes(parsed, kind)
        dat = read_spectrum(str(folder / name))
        x, y = np.array(r["x"]), np.array(next(iter(r["columns"].values())))
        for p in r["peaks"]:
            m = (dat[:, 0] > p["freq"] - 6) & (dat[:, 0] < p["freq"] + 6)
            assert abs(dat[m, 0][np.argmax(dat[m, col])] - p["freq"]) <= 2.0
            k = (x > p["freq"] - 6) & (x < p["freq"] + 6)
            assert abs(x[k][np.argmax(y[k])] - p["freq"]) <= 0.6


def test_volumes(folder):
    pytest.importorskip("plotly", reason="the cube engine imports plotly")
    from mace.gui import volume
    wan = volume.read(str(folder / "mgo_wan" / "1_WAN.CUBE"))
    meta, _ = volume.payload(wan, folder / "mgo_wan" / "1_WAN.CUBE")
    assert meta["type"] == "orbital" and meta["signed"]
    dens = volume.read(str(folder / "mgo_dens" / "DENS_CUBE.DAT"))
    meta, _ = volume.payload(dens, folder / "mgo_dens" / "DENS_CUBE.DAT")
    assert meta["type"] == "density" and meta["stats"]["duplicate_planes"] == ["a", "b", "c"] and meta["stats"]["integral"] > 0
    d, t = volume.combination([str(folder / "mgo_dens" / "DENS_CUBE.DAT"), str(folder / "mgo_pato" / "DENS_CUBE.DAT")], [1, -1])
    a = np.asarray(volume.read(str(folder / "mgo_dens" / "DENS_CUBE.DAT"), skip_vacuum=True).data)
    b = np.asarray(volume.read(str(folder / "mgo_pato" / "DENS_CUBE.DAT"), skip_vacuum=True).data)
    assert np.array_equal(d, a - b)                                  # same grid: the CLI's arithmetic


def test_echg_map(folder):
    """The 2D map reads to NPX x NPY values whose extremes are the ones CRYSTAL prints (VALMIN, VALMAX)."""
    import re
    from mace.gui import maps, filekind
    p = maps.map_payload(folder / "mgo_echg.f25")
    m = p["maps"][0]
    out = (folder / "mgo_echg.out").read_text()
    lo, hi = (float(x) for x in re.search(r"VALMIN\s+(\S+)\s+VALMAX\s+(\S+)", out).groups())
    assert (m["nrow"], m["ncol"]) == (50, 50) and len(m["values"]) == 50 and len(m["values"][0]) == 50
    assert abs(m["min"] - lo) <= 1e-4 * abs(lo) and abs(m["max"] - hi) <= 1e-4 * abs(hi)
    assert m["A"] == [0.0, 0.0, 0.0] and m["B"] == [4.21, 0.0, 0.0] and m["C"] == [4.21, 4.21, 0.0]
    assert filekind.info_payload(folder / "mgo_echg.f25", "f25")["f25_kind"] == "MAPN"


def test_joblog_link(folder):
    from mace.gui import joblink
    r = joblink.link(folder, folder / "water_opt-123456.o")
    assert (r["calc"], r["job_id"], r["how"]) == ("water_opt.out", "123456", "name")


@pytest.mark.gui
def test_browser_smoke(tmp_path):
    """The fixtures in a folder: chips by content, a FREQ output opens as Modes, the job log
    as a linked text, a WAN cube as a Volume; no error notice."""
    pytest.importorskip("selenium.webdriver")
    pytest.importorskip("plotly")
    pytest.importorskip("ase")
    from selenium.webdriver.common.by import By
    root = unpack(tmp_path / "root")
    app = U.start_app(tmp_path, root=root, MACE_GUI_MAX_WORKERS="2")
    drv = app[0]
    try:
        H.wait(drv, "const d = window.MACE.state.details; return d && d['water_freq.out'] && d['water_freq.out'].refined", 60)
        chips = drv.execute_script("return Object.fromEntries([...document.querySelectorAll('#file-list .file')].map((b) => [b.dataset.path, b.querySelector('.kb').textContent]))")
        assert chips["water_freq.out"] == "FREQ" and chips["water_opt.out"] == "OPT" and chips["mgo_doss.out"] == "out"
        assert chips["water_opt-123456.o"] == "log"
        drv.execute_script("window.__mace.open('water_freq.out')")
        H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode && s.n === 9", 60)
        assert not drv.execute_script("return document.querySelector('#freq-host .fv-steps').hidden")    # PREOPTGEOM
        drv.execute_script("window.__mace.freqView.play(false); window.__mace.open('water_opt-123456.o')")
        H.wait(drv, "return document.querySelector('#text-job') && document.querySelector('#text-job').textContent.includes('water_opt.out')", 30)
        drv.execute_script("window.__mace.open('mgo_echg.f25')")
        H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'map' && s.drawn", 60)
        ann = drv.execute_script("return window.__mace.plotView.el().layout.annotations.map((a) => a.text)")
        assert ann == ["A", "B", "C"]
        drv.execute_script("window.__mace.open('mgo_wan/1_WAN.CUBE')")
        # the orbital is signed (both phases offered); at the default level (0.156) only the + phase has a
        # surface (min -0.037): an empty lobe is not drawn, and a level below 0.037 shows both
        H.wait(drv, "const v = window.__mace.volumeView, s = v && v.state(); return s && s.type === 'orbital' && s.surfaces === 1", 60)
        assert drv.find_elements(By.ID, "vo-minus")
        drv.execute_script("const el = document.querySelector('#vo-level'); el.value = 0.02; el.dispatchEvent(new Event('input', {bubbles: true}))")
        H.wait(drv, "return window.__mace.volumeView.state().surfaces === 2", 10)
        # switching between open plot tabs (real clicks) draws each tab's own plot, with a file or not
        drv.execute_script("window.__mace.open('mgo_doss.DOSS.DAT')")
        H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'dos' && s.drawn", 60)
        for title, kind in (("mgo_echg.f25 · 2D map", "map"), ("mgo_doss.DOSS.DAT", "dos"), ("mgo_echg.f25 · 2D map", "map"),
                            ("mgo_doss.DOSS.DAT", "dos")):
            tab = [t for t in drv.find_elements(By.CSS_SELECTOR, ".doc-tab") if title in t.get_attribute("textContent")]
            assert tab, (title, [t.get_attribute("textContent") for t in drv.find_elements(By.CSS_SELECTOR, ".doc-tab")])
            drv.execute_script("arguments[0].scrollIntoView({block: 'nearest', inline: 'nearest'})", tab[0])
            tab[0].click()
            H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === arguments[0] && s.drawn", 30, kind)
        assert drv.execute_script("return document.querySelector('#st-doc').textContent").startswith("Saved")
        drv.execute_script("window.__mace.open('mgo_echg.f25')")
        H.wait(drv, "return document.querySelector('#st-doc').textContent.startsWith('Read only')", 10)
        errs = drv.execute_script("return [...document.querySelectorAll('#toasts .toast2.error .msg')].map((m) => m.textContent)")
        assert not errs, errs
        assert not [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    finally:
        U.stop_app(app)


def test_map_names_come_from_the_properties_output(folder, tmp_path):
    """fort.25 does not say what a MAPN block holds (ECHG, POTM and CLAS share the layout): the
    properties output beside it does; without one the map is not called a density."""
    from mace.gui import maps
    p = maps.map_payload(folder / "mgo_echg.f25")
    assert [m["label"] for m in p["maps"]] == ["Charge density"] and p["maps"][0]["unit"] == "e bohr<sup>−3</sup>"
    alone = tmp_path / "mgo_echg.f25"
    shutil.copyfile(folder / "mgo_echg.f25", alone)
    p = maps.map_payload(alone)
    assert [m["label"] for m in p["maps"]] == ["2D map"] and p["named_by"] is None
    # the same block written by POTM: its output closes the keyword with "TTTT... POTM"
    shutil.copyfile(folder / "mgo_echg.f25", tmp_path / "pot.f25")
    text = (folder / "mgo_echg.out").read_text()
    (tmp_path / "pot.out").write_text(text.replace("ALPHA+BETA ELECTRONS DENSITY", "ELECTROSTATIC POTENTIAL")
                                      .replace("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ECHG", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT POTM"))
    p = maps.map_payload(tmp_path / "pot.f25")
    assert [m["label"] for m in p["maps"]] == ["Electrostatic potential"]


def test_difference_slice_is_the_difference_and_save_is_exclusive(folder, tmp_path):
    """A slice of a difference made in the GUI is a plane of the difference (not of its first
    operand); Save Result As refuses a name that appeared after the server's check, and leaves
    no temporary file behind."""
    from mace.gui import volume
    pytest.importorskip("plotly", reason="the cube engine imports plotly")
    paths = [str(folder / "mgo_dens" / "DENS_CUBE.DAT"), str(folder / "mgo_pato" / "DENS_CUBE.DAT")]
    a = np.asarray(volume.read(paths[0], skip_vacuum=True).data)
    b = np.asarray(volume.read(paths[1], skip_vacuum=True).data)
    k = a.shape[2] // 2
    r = volume.slice_job(paths=paths, coefficients=[1.0, -1.0], axis="c", index=k)
    assert np.allclose(np.asarray(r["values"]), np.round((a - b)[:, :, k], 10), atol=1e-12)
    assert not np.allclose(np.asarray(r["values"]), np.round(a[:, :, k], 10))
    target = tmp_path / "taken.CUBE"
    target.write_text("someone else's file\n")
    with pytest.raises(ValueError, match="never writes over"):
        volume.save_job(paths=paths, coefficients=[1.0, -1.0], target=str(target))
    assert target.read_text() == "someone else's file\n"
    assert [x.name for x in tmp_path.iterdir()] == ["taken.CUBE"]
    r = volume.save_job(paths=paths, coefficients=[1.0, -1.0], target=str(tmp_path / "diff.CUBE"))
    assert r["bytes"] > 0 and sorted(x.name for x in tmp_path.iterdir()) == ["diff.CUBE", "taken.CUBE"]


def test_joblink_never_reads_outside_the_served_folder(tmp_path):
    """A log at the served folder's root: the materials.db one folder up (outside) is not read."""
    import sqlite3
    from mace.gui import joblink
    outer = tmp_path / "outer"
    root = outer / "served"
    root.mkdir(parents=True)
    con = sqlite3.connect(outer / "materials.db")
    con.execute("CREATE TABLE calculations (calc_id, calc_type, output_file, work_dir, input_file, slurm_job_id)")
    con.execute("INSERT INTO calculations VALUES (1, 'SP', ?, ?, NULL, '777777')", (str(root / "x.out"), str(root)))
    con.commit()
    con.close()
    (root / "x.out").write_text("not linked by name\n")
    (root / "slurm-777777.out").write_text("srun: started\n")
    assert joblink._db_files(root.resolve(), (root / "slurm-777777.out").resolve()) == []
    r = joblink.link(root, root / "slurm-777777.out")
    assert r["how"] != "database"


def test_files_beside_a_run_never_lead_out_of_the_folder(folder, tmp_path):
    """A data file beside a properties output that is a symlink out of the served folder is left
    out of the properties document (it was an error for the whole document)."""
    import os
    from types import SimpleNamespace
    from mace.gui import filekind
    root = tmp_path / "served"
    root.mkdir()
    for name in ("mgo_doss.out", "mgo_doss.DOSS.DAT"):
        shutil.copyfile(folder / name, root / name)
    outside = tmp_path / "elsewhere.f25"
    shutil.copyfile(folder / "mgo_doss.f25", outside)
    os.symlink(outside, root / "mgo_doss.f25")

    class Cache(dict):
        def put(self, k, v, n=0):
            self[k] = v
    st = SimpleNamespace(root=root.resolve(), parse_cache=Cache(), rel=lambda p: Path(p).resolve().relative_to(root.resolve()).as_posix())
    got = filekind.props_payload(st, root / "mgo_doss.out")
    assert [f["name"] for f in got["files"]] == ["mgo_doss.DOSS.DAT"]
    assert filekind.rel_inside(st, root / "mgo_doss.f25") is None and filekind.rel_inside(st, root / "missing.out") is None
