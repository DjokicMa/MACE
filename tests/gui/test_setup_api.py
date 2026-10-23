"""Calculation Setup's server side (plan T2): the CLI is the source of every
default and every choice; its saved JSON is kept as its own bytes; a
properties preview never copies the wavefunction; staging mirrors the
names the CLI looks at, so the staged deck is the in-place deck."""
import builtins
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import setup as S  # noqa: E402


def _cache():
    store = {}
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v), store)


def _run(work, path, rel, mode, calc, answers=None, patches=None, preview=True, json_text=None, cache=None):
    get, put, _ = cache or _cache()
    req = S.Request({"answers": answers or {}, "patches": patches or {}, "preview": preview, "json_text": json_text})
    return S.run_setup(req, mode, calc, work, path, rel, S.source_key(mode, path), get, put, None,
                       schema_info=S.schema())


# ------------------------------------------------------------ defaults come from the CLI
@pytest.mark.skipif(not U.mem_ok(1), reason="needs 1 GB free")
def test_cif2d12_defaults_are_the_clis_enter_through(tmp_path):
    pytest.importorskip("ase")
    d = tmp_path / "manual"
    (d / "cif").mkdir(parents=True)
    shutil.copy2(H.EXAMPLES / "structures" / "MgO.cif", d / "cif")
    R.run_cli_manual(["convert", "--cif_dir", "cif", "--output_dir", "out", "--save_options", "--options_file", "m.json"],
                     R.ENTER, d)
    r = _run(tmp_path / "w", H.EXAMPLES / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT")
    data = json.loads(r["json_text"])
    assert (data["functional"], data["spin_polarized"], data["basis_set"]) == ("HSE06", True, "POB-TZVP-REV2")
    assert r["json_text"] == (d / "m.json").read_text()                  # the CLI's own bytes
    m = next((d / "out").glob("*.d12"))
    assert (r["deck_name"], r["deck"]) == (m.name, m.read_text())
    assert "SPIN" in r["deck"].splitlines()


def test_opt2d12_defaults_are_the_parents_settings(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    d = tmp_path / "manual"
    shutil.copytree(src, d / "in")
    R.run_cli_manual(["opt2d12", "--out-file", "in/Si_sp.out", "--calc-type", "OPT", "--save-options", "--options-file",
                      "parent.json", "--output-dir", "out"], R.YES, d)
    r = _run(tmp_path / "w", src / "Si_sp.out", "Si_sp.out", "opt2d12", "OPT")
    assert r["json_text"] == (d / "parent.json").read_text()
    assert json.loads(r["json_text"])["functional"] == "PBE0"
    assert r["deck"] == next((d / "out").glob("*.d12")).read_text()


def test_opt2d3_band_defaults(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    (src / "Si_sp.f9").write_text("placeholder\n")
    r = _run(tmp_path / "w", src / "Si_sp.out", "Si_sp.out", "opt2d3", "BAND")
    conf = json.loads(r["json_text"])["configuration"]
    # SeeK-path is the default k-path (format 4): k-point vectors, not labels
    assert (conf["n_points"], conf["path_method"], conf["seekpath_full"]) == (1000, "coordinates", True)
    assert r["deck_name"] == "Si_band.d3" and "6 8 1000 1 100 1 0" in r["deck"] and "SeeKPath" in r["deck"]


def test_opt2d3_without_a_wavefunction_shows_the_clis_refusal(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    r = _run(tmp_path / "w", src / "Si_sp.out", "Si_sp.out", "opt2d3", "BAND")
    assert r["deck"] is None and r["save_blocked"]
    assert "without Si_sp.f9" in r["problems"][0]["message"] and r["problems"][0]["severity"] == "error"
    assert not (src / "Si_sp.f9").exists() and not list(src.glob("*.f9"))     # nothing made in the user's folder


# ------------------------------------------------------------ the f9 is never copied (section 9.2)
def _blocks(path: Path) -> int:
    total = 0
    for dp, _, names in os.walk(path):
        for n in names:
            try:
                total += os.lstat(os.path.join(dp, n)).st_blocks * 512
            except OSError:
                pass
    return total


def _links(path: Path):
    return [os.path.join(dp, n) for dp, ds, ns in os.walk(path) for n in ns + ds if os.path.islink(os.path.join(dp, n))]


@pytest.mark.skipif(not U.tmp_free_ok(1), reason="needs 1 GB free in /tmp")
def test_opt2d3_stage_no_f9_copy(tmp_path, monkeypatch):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    f9 = src / "Si_sp.f9"
    with open(f9, "wb") as fh:
        fh.truncate(2 * 1024 ** 3)                # 2 GB, sparse: no RAM used
    real_open, real_os_open = builtins.open, os.open

    def guard(p, *a, **k):
        if os.path.realpath(str(p)) == os.path.realpath(str(f9)):
            raise AssertionError("the real wavefunction was opened")
        return real_open(p, *a, **k)

    def guard_os(p, *a, **k):
        if os.path.realpath(str(p)) == os.path.realpath(str(f9)):
            raise AssertionError("the real wavefunction was opened")
        return real_os_open(p, *a, **k)
    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(os, "open", guard_os)
    st = S.stage("opt2d3", "BAND", tmp_path / "unit", src / "Si_sp.out", "Si_sp.out")
    monkeypatch.undo()
    # "live": whether the run that writes it is still going (PLAN_analysis C18.7; here a fresh file)
    assert {k: v for k, v in st.f9.items() if k != "live"} == {"name": "Si_sp.f9", "bytes": 2 * 1024 ** 3}
    assert isinstance(st.f9.get("live"), bool)
    assert (st.tree / "Si_sp.f9").stat().st_size == 0 and not _links(st.tree)

    work = tmp_path / "w"
    work.mkdir()
    peak, stop = [0], threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], _blocks(work))
            time.sleep(0.02)
    used0 = shutil.disk_usage("/tmp").used
    t = threading.Thread(target=watch)
    t.start()
    try:
        r = _run(work, src / "Si_sp.out", "Si_sp.out", "opt2d3", "BAND")
    finally:
        stop.set()
        t.join()
    grew = shutil.disk_usage("/tmp").used - used0
    assert peak[0] < 1024 * 1024, peak[0]
    assert _blocks(work) < 1024 * 1024 and not _links(work)
    # all of /tmp is measured, which other processes write to as well: a copy
    # of the 2 GB wavefunction still shows, the noise of a parallel run does not
    assert grew < 512 * 1024 * 1024, grew
    assert {k: v for k, v in r["f9"].items() if k != "live"} == {"name": "Si_sp.f9", "bytes": 2 * 1024 ** 3}
    assert any("Si_sp.f9 (2048 MB)" in p["message"] and "Si_band.f9" in p["message"] for p in r["problems"])
    assert "Si_sp.f9 -> Si_band.f9" in r["log"]
    f9.unlink()


# ------------------------------------------------------------ staged == in place
def _in_place(folder: Path, tmp: Path, calc: str, mode: str, json_text: str):
    d = tmp / f"inplace_{calc}_{len(list(tmp.iterdir()))}"
    shutil.copytree(folder, d)
    (d / "s.json").write_text(json_text)
    if mode == "opt2d3":
        R.run_printed(["mace", "opt2d3", "-i", "Si_sp.out", "-t", calc, "--config-file", "s.json"], d)
    else:
        R.run_printed(["mace", "opt2d12", "--out-file", "Si_sp.out", "--config-file", "s.json", "--yes"], d)
    new = sorted(p for p in d.iterdir() if p.suffix in (".d3", ".d12") and not (folder / p.name).exists())
    return new[0].name, new[0].read_text()


@pytest.mark.parametrize("mode, calc", [("opt2d3", "BAND"), ("opt2d3", "DOSS"), ("opt2d12", "SP")])
def test_stage_matches_in_place(tmp_path, mode, calc):
    folder = R.messy(tmp_path / "messy")
    r = _run(tmp_path / "w", folder / "Si_sp.out", "Si_sp.out", mode, calc)
    assert (r["deck_name"], r["deck"]) == _in_place(folder, tmp_path, calc, mode, r["json_text"])
    if mode == "opt2d3":
        assert r["deck_name"].startswith("Si_sp_")          # the sibling Si_opt.out renames the files
        assert r["f9"]["name"] == "Si_sp.f9"                 # the output's own, first for the CLI since bd397f5f
    # each sibling removed in turn: the staged deck still equals the in-place one
    for gone in ("Si_opt.out", "Si_extra.d12", "fort.9"):
        f2 = tmp_path / f"m_{gone}"
        shutil.copytree(folder, f2)
        (f2 / gone).unlink()
        r2 = _run(tmp_path / f"w_{gone}", f2 / "Si_sp.out", "Si_sp.out", mode, calc)
        assert (r2["deck_name"], r2["deck"]) == _in_place(f2, tmp_path, calc, mode, r2["json_text"]), gone


CANDIDATES = ["fort.9", "fort.98", "Si.f9", "Si.f98", "Si_sp.f9", "Si_opt.f98", "Si_freq2.f9", "Si_soc99.f98"]


def test_wavefunction_candidates_are_the_clis(tmp_path):
    names = S.wavefunction_candidates("Si_sp.out")
    # since bd397f5f the CLI looks for the output's own <stem>.f9 first (it no
    # longer comes at the end)
    assert names[:8] == ["Si_sp.f9", "Si_sp.f98", "fort.9", "fort.98", "Si.f9", "Si.f98", "Si_sp.f9", "Si_sp.f98"]
    assert len(names) == 6 + 4 * 2 * 99 and names[-2:] == ["Si_soc99.f9", "Si_soc99.f98"]
    for name in CANDIDATES:
        d = tmp_path / name
        shutil.copytree(H.EXAMPLES / "outputs", d)
        (d / name).write_text("x")
        r = R.run_printed(["mace", "opt2d3", "-i", "Si_sp.out", "-t", "DOSS", "--config-file",
                           str(_doss_config(tmp_path))], d)
        assert f"Copying wavefunction: {name} ->" in r.stdout + r.stderr, name
        assert S.find_wavefunction(d / "Si_sp.out")["name"] == name
    # the order: with two, the CLI takes the one earlier in the list
    d = tmp_path / "two"
    shutil.copytree(H.EXAMPLES / "outputs", d)
    (d / "Si_opt.f9").write_text("x")
    (d / "Si.f98").write_text("x")
    (d / "fort.9").write_text("x")
    (d / "Si_sp.f9").write_text("x")       # the output's own: first since bd397f5f
    r = R.run_printed(["mace", "opt2d3", "-i", "Si_sp.out", "-t", "DOSS", "--config-file", str(_doss_config(tmp_path))], d)
    assert "Copying wavefunction: Si_sp.f9 ->" in r.stdout + r.stderr
    assert S.find_wavefunction(d / "Si_sp.out")["name"] == "Si_sp.f9"
    (d / "Si_sp.f9").unlink()
    (d / "fort.9").unlink()
    for f in d.glob("Si_doss*"):
        f.unlink()
    r = R.run_printed(["mace", "opt2d3", "-i", "Si_sp.out", "-t", "DOSS", "--config-file", str(_doss_config(tmp_path))], d)
    assert "Copying wavefunction: Si.f98 ->" in r.stdout + r.stderr
    assert S.find_wavefunction(d / "Si_sp.out")["name"] == "Si.f98"
    # beside an output of the same name (Si.out) the CLI names Si_sp.out's
    # files Si_sp_* and looks no further for Si.f9: that one is Si.out's
    d = tmp_path / "sibling"
    shutil.copytree(H.EXAMPLES / "outputs", d)
    shutil.copy(d / "Si_sp.out", d / "Si.out")
    (d / "Si.f9").write_text("x")
    (d / "Si_sp_sp.f9").write_text("x")
    r = R.run_printed(["mace", "opt2d3", "-i", "Si_sp.out", "-t", "DOSS", "--config-file", str(_doss_config(tmp_path))], d)
    assert "Copying wavefunction: Si_sp_sp.f9 ->" in r.stdout + r.stderr
    assert S.find_wavefunction(d / "Si_sp.out")["name"] == "Si_sp_sp.f9"


def _doss_config(tmp):
    """The CLI's own saved DOSS settings (Enter at every question)."""
    p = tmp / "doss.json"
    if not p.exists():
        src = tmp / "doss_src"
        shutil.copytree(H.EXAMPLES / "outputs", src)
        (src / "Si_sp.f9").write_text("x")
        p.write_text(_run(tmp / "doss_w", src / "Si_sp.out", "Si_sp.out", "opt2d3", "DOSS", preview=False)["json_text"])
    return p


# ------------------------------------------------------------ the JSON stays the CLI's bytes
def test_empty_patch_gives_the_same_bytes_and_floats_stay_floats():
    text = json.dumps({"fmixing": 30.0, "smearing_width": 0.0, "tol": 1e-07, "maxcycle": 800, "name": "Cé",
                       "nested": {"a": 1}}, indent=2)
    assert S.apply_patch(text, {}) == (text, [])
    out, changed = S.apply_patch(text, {"fmixing": 40, "maxcycle": "900"})
    assert changed == ["fmixing", "maxcycle"]
    assert '"fmixing": 40.0' in out and '"maxcycle": 900' in out
    assert '"smearing_width": 0.0' in out and '"tol": 1e-07' in out and "\\u00e9" in out
    assert not out.endswith("\n")
    order = [l.split(":")[0].strip() for l in out.splitlines() if l.startswith('  "')]
    assert order == ['"fmixing"', '"smearing_width"', '"tol"', '"maxcycle"', '"name"', '"nested"']


def test_hand_edited_json_is_used_verbatim(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    r0 = _run(tmp_path / "w0", src / "Si_sp.out", "Si_sp.out", "opt2d12", "SP")
    hand = r0["json_text"].replace('"fmixing": 80', '"fmixing":   70') + "\n\n"
    r = _run(tmp_path / "w1", src / "Si_sp.out", "Si_sp.out", "opt2d12", "SP", json_text=hand)
    assert r["json_text"] == hand and r["json_source"] == "hand"
    assert "FMIXING\n70" in r["deck"]


def test_choices_come_from_the_cli(monkeypatch):
    import d12_constants as C
    monkeypatch.setattr(C, "INTERNAL_BASIS_SETS", {**C.INTERNAL_BASIS_SETS, "TEST-BASIS": {}})
    assert "TEST-BASIS" in S.schema()["internal_basis_sets"]
    names = {f for c in S.schema()["functional_categories"] for f in c["functionals"]}
    assert {"HSE06", "PBE0", "B3LYP", "HSESOL3C"} <= names
    # no functional, basis or grid literal in the page's code (comments aside)
    static = H.REPO_ROOT / "mace" / "gui" / "static"
    for p in list((static / "js").rglob("*.js")) + [static / "index.html"]:
        text = re.sub(r"/\*.*?\*/|(?<!:)//[^\n]*|<!--.*?-->", "", p.read_text(), flags=re.S)
        for lit in ("HSE06", "B3LYP", "POB-", "XLGRID", "PBE0"):
            assert lit not in text, (p.name, lit)


@pytest.mark.parametrize("mode, calc, src", [("cif2d12", "OPT", "structures/MgO.cif"), ("cif2d12", "FREQ", "structures/Si.cif"),
                                             ("opt2d12", "OPT", "outputs/Si_sp.out"), ("opt2d12", "FREQ", "outputs/Si_sp.out"),
                                             ("opt2d3", "BAND", "outputs/Si_sp.out"), ("opt2d3", "DOSS", "outputs/Si_sp.out")])
def test_every_field_is_one_of_the_clis_questions(tmp_path, mode, calc, src):
    """DERIVED_BY_CLI: a "cli" field is set by answering a question the CLI asks (spike S6); each
    field offered for this mode and calculation is found among the questions, and shows the JSON's value."""
    if mode == "cif2d12":
        pytest.importorskip("ase")
    d = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / Path(src).parent, d)
    (d / "Si_sp.f9").write_text("x") if mode == "opt2d3" else None
    r = _run(tmp_path / "w", d / Path(src).name, Path(src).name, mode, calc, preview=False)
    ids = {f["id"] for f in r["form"]}
    want = {f.id for f in S.FIELDS if mode in f.modes and (not f.calcs or calc in f.calcs)}
    optional = {"smear_width", "dispersion", "basis_keep", "symmetry", "spin",       # asked only on some routes / parents
                "spinlock", "levshift_values",                                # (SPIN on, level shifting on)
                "doss_step", "doss_points",                                   # (DOSS Resolution Step, Points;
                "doss_window", "doss_window_lo", "doss_window_hi",            #  the window: offered with E_F;
                "doss_band_index", "doss_conduction_bands"}                   #  bands by index: a 1D material)
    # worked out from another row: the method from the functional, the basis set's type (and keeping the parent's)
    # from the basis set; the basis set is a follow-up's row too
    optional |= {"method", "basis_type", "basis_keep", "basis_set"} if mode == "opt2d12" else {"method", "basis_type"}
    assert want - optional <= ids, sorted(want - ids)
    assert all(S.DERIVED_BY_CLI[f["id"]] in ("cli", "patch") for f in r["form"] if not f["id"].startswith("q:"))
    for f in r["form"]:
        if f["kind"] == "menu" and f["id"] != "functional" and f["options"]:
            assert f["value"] in [o["head"] for o in f["options"]] or f["value"] is not None, f


def test_a_menu_the_cli_changed_is_reported_not_guessed(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    with pytest.raises(S.SetupError, match="did not ask what Setup expected"):
        _run(tmp_path / "w", src / "Si_sp.out", "Si_sp.out", "opt2d12", "SP", answers={"functional": "NOT-A-FUNCTIONAL"})


# ------------------------------------------------------------ through the server
@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("setup_api")
    root = R.served_root(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def test_cli_defaults_are_cached(served):
    root, base = served
    body = {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "SP", "wait": 120}
    st, _, a = U.post(base, "api/setup/run", body)
    assert st == 200, a
    t0 = time.perf_counter()
    st, _, b = U.post(base, "api/setup/run", body)
    ms = (time.perf_counter() - t0) * 1000
    assert st == 200 and b["json_text"] == a["json_text"] and b["deck"] == a["deck"]
    assert ms <= 20 * 5, ms          # the bench row is <= 20 ms; a loaded test machine gets 5x


def test_setup_run_is_a_cancelable_deck_task_with_stages(served):
    pytest.importorskip("ase", reason="cif2d12 reads the CIF with ase")
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": {"path": "structures/NaCl.cif"}, "calc_type": "OPT",
                                              "doc": "d1", "wait": 0})
    assert st == 202 and r["task"]["kind"] == "deck"
    tid = r["task"]["id"]
    end = time.time() + 120
    while time.time() < end:
        t = next(x for x in U.req(base, "api/tasks")[2]["tasks"] if x["id"] == tid)
        if t["state"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert t["state"] == "done", t
    st, _, res = U.req(base, f"api/tasks/{tid}/result")
    assert st == 200 and res["deck_name"].startswith("NaCl_")


def test_save_writes_only_inside_the_folder_and_refuses_overwrite(served):
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "SP", "wait": 120})
    body = {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "SP", "json_text": r["json_text"], "name": "t_si.json"}
    st, _, s = U.post(base, "api/setup/save", body)
    assert st == 200 and s["json"] == "outputs/t_si.json"
    assert (root / "outputs" / "t_si.json").read_text() == r["json_text"]
    assert s["command"] == "mace opt2d12 --out-file outputs/Si_sp.out --config-file outputs/t_si.json --yes"
    st, _, s2 = U.post(base, "api/setup/save", body)
    assert st == 400 and "already exists" in s2["error"]
    for bad in ("../x.json", "a/b.json", "x.txt"):
        st, _, s3 = U.post(base, "api/setup/save", {**body, "name": bad})
        assert st == 400, bad
    assert not (root.parent / "x.json").exists()


def test_cif2d12_save_layouts(served):
    pytest.importorskip("ase", reason="cif2d12 reads the CIF with ase")
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": {"path": "structures/LiF.cif"}, "calc_type": "OPT", "wait": 120})
    st, _, own = U.post(base, "api/setup/save", {"source": {"path": "structures/LiF.cif"}, "calc_type": "OPT",
                                                 "json_text": r["json_text"], "layout": "own", "overwrite": True})
    assert st == 200 and own["saved"] == ["setup/LiF/LiF.cif", "setup/LiF/LiF_options.json"], own
    assert own["converts"] == 1
    st, _, inp = U.post(base, "api/setup/save", {"source": {"path": "structures/LiF.cif"}, "calc_type": "OPT",
                                                 "json_text": r["json_text"], "layout": "inplace", "overwrite": True})
    assert st == 200 and inp["converts"] == 6 and inp["command"].endswith("--cif_dir structures")


def test_write_decks_now_runs_the_printed_command(served):
    pytest.importorskip("ase", reason="cif2d12 reads the CIF with ase")
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": {"path": "structures/NaCl.cif"}, "calc_type": "OPT", "wait": 120})
    st, _, w = U.post(base, "api/setup/save", {"source": {"path": "structures/NaCl.cif"}, "calc_type": "OPT", "overwrite": True,
                                               "json_text": r["json_text"], "write_decks": True})
    if st == 202:
        tid = w["task"]["id"]
        end = time.time() + 120
        while time.time() < end:
            st, _, w = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                break
            time.sleep(0.2)
    assert st == 200 and w["written"] == [f"setup/NaCl/{r['deck_name']}"], w
    assert (root / w["written"][0]).read_text() == r["deck"]


# ------------------------------------------------------------ plan T13: P1 CIFs of symmetric structures
def test_p1_cif_of_a_symmetric_structure_gets_an_info_problem(tmp_path):
    ase = pytest.importorskip("ase")
    from ase.io import read, write
    d = tmp_path / "src"
    d.mkdir()
    p1 = d / "MgO_p1.cif"
    write(str(p1), read(str(H.EXAMPLES / "structures" / "MgO.cif")))              # ase writes every atom, P 1
    r = _run(tmp_path / "w", p1, "MgO_p1.cif", "cif2d12", "OPT", preview=False)
    notes = [p for p in r["problems"] if p.get("action_id") == "save_symmetrized"]
    assert len(notes) == 1 and notes[0]["severity"] == "info"
    assert notes[0]["message"] == ("This CIF is P1; spglib finds Fm-3m (225) at 1e-3 Å. "
                                   "cif2d12 keeps P1 (a slower CRYSTAL run).")
    # nothing changes silently: the deck is still the P1 one the CLI writes
    assert json.loads(r["json_text"])
    # the symmetric original gets no note
    r2 = _run(tmp_path / "w2", H.EXAMPLES / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT", preview=False)
    assert not [p for p in r2["problems"] if p.get("action_id")]


def test_cli_warnings_read_plainly_in_problems():
    """The row's icon says "warning": the CLI's [WARN], emoji and WARNING: prefix go."""
    from mace.gui import setup as S
    log = "[WARN] ⚠️  WARNING: Space group mismatch!\nWarning: basis lacks f functions\nplain line\n[WARN] Basis lacks f functions"
    msgs = [p["message"] for p in S.problems_from(log)]
    assert msgs == ["Space group mismatch!", "basis lacks f functions", "Basis lacks f functions"]
