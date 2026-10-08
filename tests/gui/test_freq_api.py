"""IR/Raman spectra and FREQ normal modes through the server (plan T9, B5).

Corpus outputs are copied into a temp dir at run time (skipped without the
test/ corpus); nothing is committed. The modes equal mace plotting's own
Crystal23FreqParser on the whole file; the server finds them through the
output index (version 2) and parses only the FREQ section in a pool worker;
listing a folder of FREQ outputs opens none of them; a 60 MB output's FREQ
request never costs its size in memory."""
import builtins
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytest.importorskip("numpy")
sys.path.insert(0, str(H.REPO_ROOT))
CORPUS = H.REPO_ROOT / "test" / "FREQ"
DIA = "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"
CONF = "1LiFSI-1DEC-conf{}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized{}"

needs_corpus = pytest.mark.skipif(not (CORPUS / DIA).is_file(), reason="test/ corpus not present (gitignored)")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not (CORPUS / DIA).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("freq")
    root = H.examples_copy(tmp)
    f = root / "freq"
    f.mkdir()
    shutil.copy2(CORPUS / DIA, f / DIA)
    for k in (1, 2, 3):
        for suffix in ("_IRSPEC.DAT", "_RAMSPEC.DAT"):
            src = CORPUS / CONF.format(k, suffix)
            if src.is_file():
                shutil.copy2(src, f / src.name)
    out1 = CORPUS / CONF.format(1, ".out")
    if out1.is_file():
        shutil.copy2(out1, f / out1.name)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    yield root, base, tmp, proc
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


def test_modes_equal_mace_plotting_parser(served):
    pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
    root, base, _, _ = served
    from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
    whole = Crystal23FreqParser(str(root / "freq" / DIA))
    r = result(base, f"api/freq?path=freq/{DIA}")
    assert [(m["mode"], m["freq"], m["irrep"], m["ir_active"], m["raman_active"]) for m in r["modes"]] == \
        [(m["mode"], m["freq"], m["irrep"], m["ir_active"], m["raman_active"]) for m in whole.modes]
    assert r["mode_atoms"] == r["n_atoms"] == 4
    import numpy as np
    for k in (1, 4, 6):
        st, _, m = U.req(base, f"api/freq/mode?path=freq/{DIA}&k={k}")
        d = np.asarray(m["displacement"])
        ref = whole.get_displacement(k) * 0.529177210903
        # the same vectors in the GUI's Cartesian frame: a rotation keeps every length
        assert np.allclose(np.linalg.norm(d, axis=1), np.linalg.norm(ref, axis=1), atol=1e-5)
        assert m["frame"] == "gui"


def test_section_parse_equals_whole_file_parse(served):
    pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
    root, base, tmp, _ = served
    from mace.gui import outindex, freq
    from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
    p = root / "freq" / DIA
    idx = outindex.build(p)
    assert outindex.VERSION >= 2 and idx.freq and idx.freq[0] > 0
    part = freq.parse_job(path=str(p), start=idx.freq[0], end=idx.freq[1], tmpdir=str(tmp))
    whole = Crystal23FreqParser(str(p))
    assert part["modes"] == [{k: m[k] for k in part["modes"][0]} for m in whole.modes]
    assert sorted(part["displacements"]) == sorted(whole.displacements)
    for k, v in whole.displacements.items():
        assert (abs(part["displacements"][k] - v) < 1e-6).all()
    assert not outindex.build(H.EXAMPLES / "outputs" / "Si_sp.out").freq


def test_spectrum_arrays_are_read_spectrums(served):
    root, base, _, _ = served
    from mace.plotting.handlers.spectra_api import read_spectrum
    ir = next((root / "freq").glob("*conf1*_IRSPEC.DAT"), None)
    if ir is None:
        pytest.skip("no IRSPEC.DAT in the corpus copy")
    st, _, r = U.req(base, f"api/spectrum?path=freq/{ir.name}")
    data = read_spectrum(str(ir))
    assert r["kind"] == "ir" and r["n_points"] == len(data)
    assert r["x"][:5] == [round(float(v), 5) for v in data[:5, 0]]
    assert r["columns"]["absorbance"][:5] == [round(float(v), 7) for v in data[:5, 2]]
    assert len(r["group"]) >= 2                       # the conformers beside it
    st, _, a = U.req(base, f"api/spectrum?path=freq/{ir.name}&average=1")
    assert a["n_used"] == len(r["group"])
    ram = next((root / "freq").glob("*conf1*_RAMSPEC.DAT"))
    st, _, rr = U.req(base, f"api/spectrum?path=freq/{ram.name}")
    assert list(rr["columns"])[:3] == ["total", "parallel", "perpendicular"] and len(rr["columns"]) == 9


def test_listing_a_folder_of_freq_outputs_opens_none(served, monkeypatch):
    root, base, _, _ = served
    from mace.gui.server import GuiState
    opened = []
    real = builtins.open

    def counting(p, *a, **k):
        if str(p).endswith(".out"):
            opened.append(str(p))
        return real(p, *a, **k)
    monkeypatch.setattr(builtins, "open", counting)
    st = GuiState(root, "t")
    try:
        files = st.list_files()["files"]
    finally:
        st.close()
    assert any(f["kind"] == "out" for f in files) and any(f["kind"] in ("ir", "raman") for f in files)
    assert not opened, opened


def _rss_mb(pid):
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0


@needs_corpus
def test_a_60_mb_freq_output_is_never_read_whole(tmp_path):
    pytest.importorskip("ase", reason="the FREQ route reads the geometry with ase")
    pytest.importorskip("plotly", reason="mace plotting's vibmode_viewer imports plotly")
    if not U.tmp_free_ok(1) or not U.mem_ok(2):
        pytest.skip("needs 1 GB free in /tmp and 2 GB of memory")
    src = (CORPUS / DIA).read_bytes()
    k = src.index(b"CALCULATION OF PHONON FREQUENCIES AT THE GAMMA POINT")
    k = src.rindex(b"\n", 0, k) + 1
    head, tail = src[:k], src[k:]
    root = tmp_path / "big"
    root.mkdir()
    filler = b" CYC   1 ETOT(AU) -1.000000000000E+01 DETOT -1.00E+00 tst  0.00E+00 PX  1.00E+00\n" * 1000
    with open(root / "big_freq.out", "wb") as fh:
        fh.write(head)
        while fh.tell() < 60 * 1024 * 1024:
            fh.write(filler)
        fh.write(tail)
    (root / "small_freq.out").write_bytes(src)
    try:
        proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="2")
        try:
            U.req(base, "api/info")
            # the same request on the small original first: what the server imports for any FREQ
            # output (numpy, the index, the geometry parser) is not what this test is about
            result(base, "api/freq?path=small_freq.out")
            before = _rss_mb(proc.pid)
            r = result(base, "api/freq?path=big_freq.out")
            assert len(r["modes"]) == 12
            grew = _rss_mb(proc.pid) - before
            kids = [_rss_mb(p) for p in H.descendants(proc.pid)]
            assert grew < 30, grew                      # the server: an index, never the 60 MB file
            assert all(k < 200 for k in kids), kids     # the worker: the FREQ section only
        finally:
            proc.terminate()
            proc.wait(timeout=15)
    finally:
        (root / "big_freq.out").unlink(missing_ok=True)
        (root / "small_freq.out").unlink(missing_ok=True)


def test_frame_rotation_matches_a_slab_with_a_different_padding_vector():
    """CRYSTAL prints a slab's c as its own padding (500 Å); the GUI's cell has its own vacuum.
    The rotation still comes out, from a, b and their normal; an unrelated cell gives none."""
    import numpy as np
    from mace.gui import freq
    th = np.radians(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    gui = np.array([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 20.0]])
    crystal = gui @ Rz.T
    crystal[2] = [0, 0, 500.0]
    lines = ["DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)", "  X  Y  Z"] + [" ".join(f"{v:.10f}" for v in r) for r in crystal]
    R = freq.frame_rotation(lines, gui)
    assert R is not None and np.allclose(crystal[:2] @ R, gui[:2], atol=1e-6)
    bad = ["DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)", "  X  Y  Z", "3 0 0", "0 3 0", "0 0 500"]
    assert freq.frame_rotation(bad, gui) is None
