"""Calculation Setup from a .d12 deck or a finished .out, not only a CIF (1.1.6).

The GUI calls the CLI's own routes and the deck is the CLI's deck byte for
byte (P == G == M, as test_setup_routes.py holds every other route to):

* a .out: ``mace opt2d12 --out-file`` (opt2d3 for BAND, DOSS, ...): the
  geometry is the run's, the settings its .d12's ("defaults follow the
  parent"). An unfinished optimization gives the deck the CLI gives it (its
  starting geometry) and Setup shows the CLI's warning with the source;
* a .d12: ``mace opt2d12 --d12-file`` (no .out): the deck is its own
  parent. Properties (BAND, DOSS, ...) need a run's wavefunction and are
  refused with that reason; EXTERNAL decks and decks that edit their
  geometry (SUPERCEL, ...) are refused with the CLI's own reason.

Each answer names where its values come from (``parent``: the .out or .d12).
Batch Setup takes a folder of decks the same way (``--d12-file`` per deck).
"""
import shlex
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R
import submit_util as SU
import test_setup_routes as T
from conftest import TEST_DATA

pytest.importorskip("numpy")
pytest.importorskip("ase")

DATA = H.REPO_ROOT / "tests" / "data"
MOLECULE = "OPT/1LiFSI-1DEC-conf3_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"

EXTERNAL_DECK = """ext_test
EXTERNAL
END
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
SHRINK
0 8
8 8 8
TOLDEE
7
END
"""


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("setup_sources")
    root = R.served_root(tmp)
    decks = root / "decks"
    decks.mkdir()
    for p in (DATA / "low_dim_groups" / "polyyne_rg28.d12", DATA / "low_dim_groups" / "graphene_lg80.d12",
              *sorted((DATA / "ecp_decks").glob("Ag1Br1*.d12"))):
        shutil.copy2(p, decks / ("agbr_ecp.d12" if p.name.startswith("Ag1Br1") else p.name))
    (decks / "ext_geom.d12").write_text(EXTERNAL_DECK)
    sc = (H.EXAMPLES / "outputs" / "Si_sp.d12").read_text().splitlines()
    i = next(k for k, x in enumerate(sc) if x.strip() == "END")
    (decks / "si_super.d12").write_text("\n".join(sc[:i] + ["SUPERCEL", "2 0 0", "0 2 0", "0 0 2"] + sc[i:]) + "\n")
    if (TEST_DATA / f"{MOLECULE}.d12").is_file():
        shutil.copy2(TEST_DATA / f"{MOLECULE}.d12", decks / "mol.d12")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def manual_route(root, row, tmp):
    """The manual CLI route for a deck: `mace opt2d12 --d12-file` asked interactively."""
    if not row.src.endswith(".d12"):
        return T.manual_route(root, row, tmp)
    d, rel = T._manual_input(root, row, tmp, "m")
    args = ["opt2d12", "--d12-file", rel, "--calc-type", row.calc, "--save-options", "--options-file", "Jm.json",
            "--output-dir", "m_out"]
    r = R.run_cli_manual(args, row.manual, d)
    decks = sorted(T._decks(d / "m_out"))
    assert r.returncode == 0 and decks, (r.stdout + r.stderr)[-3000:]
    return (d / "Jm.json").read_text(), decks[0], (d / "m_out" / decks[0]).read_text()


ROWS = [
    # a .out: opt2d12 --out-file, the parent's settings; an unfinished optimization
    R.Row(101, "outputs/Si_sp.out", "opt2d12", "SP", R.YES, keep_parent=True),
    R.Row(102, "outputs/Ag2Br3_opt_unfinished.out", "opt2d12", "SP", R.YES, keep_parent=True),
    # a .d12: opt2d12 --d12-file
    R.Row(111, "outputs/Si_sp.d12", "opt2d12", "SP", R.YES, keep_parent=True),
    R.Row(112, "outputs/Si_sp.d12", "opt2d12", "OPT", R.YES, keep_parent=True),
    R.Row(113, "outputs/Si_sp.d12", "opt2d12", "FREQ", R.YES, keep_parent=True),
    R.Row(114, "outputs/Si_sp.d12", "opt2d12", "OPT", R.MANUAL["hse06_follow"], {"functional": "HSE06"}),
    R.Row(115, "outputs/graphene_slab_sp.d12", "opt2d12", "SP", R.YES, keep_parent=True),       # SLAB
    R.Row(116, "decks/polyyne_rg28.d12", "opt2d12", "OPT", R.YES, keep_parent=True),             # POLYMER
    R.Row(117, "decks/agbr_ecp.d12", "opt2d12", "SP", R.YES, keep_parent=True),                  # EXTERNAL basis, ECP
    R.Row(118, "decks/mol.d12", "opt2d12", "SP", R.YES, keep_parent=True),                       # MOLECULE (corpus)
]


@pytest.mark.parametrize("row", ROWS, ids=repr)
def test_gui_deck_is_the_cli_deck(served, row, tmp_path):
    root, base = served
    if not (root / row.src).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    r, s, jg, gname, g = T.gui_route(root, base, row, tmp_path)
    jm, mname, m = manual_route(root, row, tmp_path)
    assert (r["deck_name"], r["deck"]) == (gname, g), "the preview is the printed command's deck"
    assert gname == mname and g == m, "the GUI's deck is the manual CLI's"
    if not row.answers:
        assert jg == jm, "the GUI changed nothing: the same settings file"
    argv = shlex.split(s["command"])
    flag = "--d12-file" if row.src.endswith(".d12") else "--out-file"
    assert argv[1] == "opt2d12" and argv[argv.index(flag) + 1] == row.src
    if row.src.endswith(".d12"):
        assert "--out-file" not in argv and "_optimized" not in gname
    lines = [x.strip() for x in g.splitlines()]
    dim = (root / row.src).read_text().splitlines()[1].strip() if row.src.endswith(".d12") else None
    if dim:
        assert lines[1] == dim, "a follow-up keeps its parent's dimensionality"
    # where the values came from
    assert r["parent"]["kind"] == Path(row.src).suffix[1:] and r["parent"]["name"] == Path(row.src).name
    if row.answers:
        changed = [f for f in r["form"] if f["changed"]]
        assert [f["id"] for f in changed] == list(row.answers)


def test_an_unfinished_out_gives_the_clis_deck_with_its_warning(served):
    root, base = served
    r = T.gui_run(base, {"source": {"path": "outputs/Ag2Br3_opt_unfinished.out"}, "calc_type": "SP"})
    assert r["deck"]
    w = r["parent"]["warning"]
    assert "unfinished optimization" in w and "starting geometry" in w, r["parent"]
    assert any("unfinished optimization" in p["message"] for p in r["problems"])
    done = T.gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "SP"})
    assert done["parent"]["warning"] is None


@pytest.mark.parametrize("calc", ["BAND", "DOSS"])
def test_properties_from_a_deck_are_refused_with_the_reason(served, calc):
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": {"path": "outputs/Si_sp.d12"}, "calc_type": calc,
                                             "preview": True, "wait": 60})
    assert st == 400, r
    msg = r.get("error") or r.get("message") or str(r)
    assert "wavefunction" in msg and "Si_sp.out" in msg


@pytest.mark.parametrize("name,words", [("ext_geom.d12", ["EXTERNAL", "fort.34", "--out-file"]),
                                        ("si_super.d12", ["SUPERCEL", ".out"])])
def test_decks_without_their_own_geometry_are_refused_with_the_clis_reason(served, name, words):
    root, base = served
    r = T.gui_run(base, {"source": {"path": f"decks/{name}"}, "calc_type": "SP"})
    assert r["deck"] is None and r.get("save_blocked"), r
    for w in words:
        assert w in r["save_blocked"], r["save_blocked"]


def test_save_writes_the_decks_beside_the_parent_deck(served, tmp_path):
    root, base = served
    body = {"source": {"path": "outputs/Si_sp.d12"}, "calc_type": "OPT"}
    r = T.gui_run(base, body)
    st, _, s = U.post(base, "api/setup/save", {**body, "json_text": r["json_text"], "overwrite": True,
                                              "write_decks": True})
    if st == 202:
        tid = s["task"]["id"]
        end = time.time() + 120
        while time.time() < end:
            st, _, s = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                break
            time.sleep(0.2)
    assert st == 200, s
    assert s["written"] == [f"outputs/{r['deck_name']}"], s
    assert (root / "outputs" / r["deck_name"]).read_text() == r["deck"]


# ---------------------------------------------------------------- Batch Setup: a folder of decks
@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("batch_decks"))
    d = s.root / "decks"
    d.mkdir()
    shutil.copy2(H.EXAMPLES / "outputs" / "Si_sp.d12", d / "si.d12")
    shutil.copy2(H.EXAMPLES / "outputs" / "graphene_slab_sp.d12", d / "gr.d12")
    shutil.copy2(DATA / "low_dim_groups" / "polyyne_rg28.d12", d / "py.d12")
    (d / "ext.d12").write_text(EXTERNAL_DECK)
    yield s
    s.stop()


def test_batch_scans_a_folder_of_decks(srv):
    st, r = srv.task_result(srv.req("api/batch/scan", {"folder": "decks", "filter": "d12", "wait": 60}))
    assert st == 200, r
    rows = {x["rel"]: x for x in r["rows"]}
    assert sorted(rows) == ["decks/ext.d12", "decks/gr.d12", "decks/py.d12", "decks/si.d12"]
    assert [k for k, v in rows.items() if v["eligible"]] == ["decks/gr.d12", "decks/py.d12", "decks/si.d12"]
    assert "fort.34" in rows["decks/ext.d12"]["why_not"]


def test_batch_of_decks_writes_the_clis_decks(srv, tmp_path):
    files = ["decks/gr.d12", "decks/py.d12", "decks/si.d12"]
    st, p = srv.task_result(srv.req("api/batch/plan", {"folder": "decks", "files": files, "calc": "OPT",
                                                       "template_answers": {}, "overrides": {}, "wait": 1}), timeout=900)
    assert st == 200, p
    assert p["mode"] == "opt2d12" and p["refused"] == 0, p
    st, w = srv.task_result(srv.req("api/batch/write", {"batch_id": p["batch_id"], "name": "b1", "wait": 1}), timeout=900)
    assert st == 200, w
    assert w["code"] == 0, w["log"]
    assert all("--d12-file" in c and "--out-file" not in c for c in w["commands"] if c.startswith("mace "))
    decks = sorted(x for x in w["written"] if x.endswith(".d12"))
    assert len(decks) == 3, w
    # each the deck the CLI writes from that parent alone
    for f in files:
        d = tmp_path / Path(f).stem
        d.mkdir()
        shutil.copy2(srv.root / f, d / Path(f).name)
        R.run_cli_manual(["opt2d12", "--d12-file", Path(f).name, "--calc-type", "OPT", "--output-dir", "o"], R.YES, d)
        (kid,) = sorted((d / "o").glob("*.d12"))
        assert (srv.root / "decks" / "b1" / "decks" / kid.name).read_text() == kid.read_text()
