"""GUI -> JSON -> CLI gives the manual CLI route's decks (plan T3, the central proof).

For every row of setup_routes.MATRIX, through the real `mace gui`:

1. manual route: the CLI asked interactively, the pinned answers typed on
   stdin, with its "save the answers" switch: deck M, saved JSON Jm;
2. GUI route: /api/setup/run with the row's field answers (what a click in
   the form sends), /api/setup/save, then the printed command run
   unchanged in a copy of the served folder: deck G; the preview's deck P;
3. control route: the CLI on its own saved file Jm: deck C.

G == M and P == G byte for byte (names too); the saved file is the bytes
the preview ran; Jg and Jm agree type-strictly (0 is not 0.0), and are the
same bytes when the GUI changed nothing. If C != M the CLI's own two routes
disagree: that is a CLI finding (xfail with the repro), never blamed on
the GUI. The house rules are checked on G: SPIN by default, follow-ups
keep the parent's settings, BAND has NSUB 1000.
"""
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

pytest.importorskip("numpy")
pytest.importorskip("ase")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("setup_routes")
    root = R.served_root(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def gui_run(base, body):
    st, _, r = U.post(base, "api/setup/run", {"preview": True, "wait": 300, **body}, timeout=400)
    assert st == 200, r
    return r


def _decks(folder: Path):
    return {p.name for p in folder.iterdir() if p.suffix.lower() in (".d12", ".d3")} if folder.is_dir() else set()


def gui_route(root, base, row, tmp):
    body = {"source": {"path": row.src}, "calc_type": row.calc, "answers": row.answers, "patches": row.patches}
    r = gui_run(base, body)
    assert r["deck"], r.get("refused") or r.get("log")
    st, _, s = U.post(base, "api/setup/save", {**body, "json_text": r["json_text"], "overwrite": True})
    assert st == 200, s
    saved = root / s["json"]
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == r["json_sha256"] == s["json_sha256"]
    g = tmp / "g"
    shutil.copytree(root, g)
    argv = shlex.split(s["command"])
    out = g / (argv[argv.index("--output_dir") + 1] if "--output_dir" in argv else str(Path(row.src).parent))
    before = _decks(out)
    R.run_printed(argv, g)
    new = sorted(_decks(out) - before)
    assert new, f"the printed command wrote no deck: {s['command']}"
    return r, s, saved.read_text(), new[0], (out / new[0]).read_text()


def _manual_input(root, row, tmp, name):
    d = tmp / name
    src = root / row.src
    if row.mode == "cif2d12":
        (d / "cif").mkdir(parents=True)
        shutil.copy2(src, d / "cif" / src.name)
        return d, None
    shutil.copytree(src.parent, d / "in")
    return d, f"in/{src.name}"


def manual_route(root, row, tmp):
    d, rel = _manual_input(root, row, tmp, "m")
    if row.mode == "cif2d12":
        args = ["convert", "--cif_dir", "cif", "--output_dir", "m_out", "--save_options", "--options_file", "Jm.json"]
    elif row.mode == "opt2d12":
        args = ["opt2d12", "--out-file", rel, "--calc-type", row.calc, "--save-options", "--options-file", "Jm.json",
                "--output-dir", "m_out"]
    else:
        args = ["opt2d3", "-i", rel, "-t", row.calc, "--save-config", "--options-file", "Jm.json", "--output-dir", "m_out"]
    r = R.run_cli_manual(args, row.manual, d)
    decks = sorted(_decks(d / "m_out"))
    assert r.returncode == 0 and decks, (r.stdout + r.stderr)[-3000:]
    return (d / "Jm.json").read_text(), decks[0], (d / "m_out" / decks[0]).read_text()


def control_route(root, row, tmp, jm_text):
    d, rel = _manual_input(root, row, tmp, "c")
    (d / "Jm.json").write_text(jm_text)
    if row.mode == "cif2d12":
        args = ["mace", "convert", "--batch", "--options_file", "Jm.json", "--cif_dir", "cif", "--output_dir", "c_out"]
    elif row.mode == "opt2d12":
        args = ["mace", "opt2d12", "--out-file", rel, "--config-file", "Jm.json", "--yes", "--output-dir", "c_out"]
    else:
        args = ["mace", "opt2d3", "-i", rel, "-t", row.calc, "--config-file", "Jm.json", "--output-dir", "c_out"]
    R.run_printed(args, d)
    decks = sorted(_decks(d / "c_out"))
    return decks[0], (d / "c_out" / decks[0]).read_text()


def _block(deck, key):
    lines = [x.strip() for x in deck.splitlines()]
    return [lines[i + 1] for i, x in enumerate(lines) if x == key and i + 1 < len(lines)]


def _mesh(deck):
    """The Monkhorst-Pack mesh a SHRINK block asks for, as (k1, k2, k3)."""
    lines = [x.strip() for x in deck.splitlines()]
    i = lines.index("SHRINK")
    first = lines[i + 1].split()
    if first[0] == "0":
        k = [int(x) for x in lines[i + 2].split()]
        return tuple(k + [1] * (3 - len(k)))
    n = int(first[0])
    return (n, n, 1) if "SLAB" in lines[:12] else (n, n, n)


def _functional(deck):
    lines = [x.strip() for x in deck.splitlines()]
    out = []
    for i, x in enumerate(lines):
        if x == "DFT":
            j = i + 1
            while j < len(lines) and lines[j] in ("SPIN",):
                j += 1
            if j < len(lines):
                out.append(lines[j])
    return out


RESULTS = {}


@pytest.mark.parametrize("row", R.MATRIX, ids=repr)
def test_gui_route_gives_the_manual_decks(served, row, tmp_path):
    root, base = served
    r, s, jg, gname, g = gui_route(root, base, row, tmp_path)
    jm, mname, m = manual_route(root, row, tmp_path)
    cname, c = control_route(root, row, tmp_path, jm)
    RESULTS[row.id] = {"G": g, "name": gname, "json": jg}
    # P == G: the preview is the printed command's deck
    assert (r["deck_name"], r["deck"]) == (gname, g)
    if (cname, c) != (mname, m):
        pytest.xfail(f"CLI finding: {row.mode} writes {cname} from its own saved answers but {mname} when asked "
                     f"(row {row.id}); not a GUI difference")
    # G == M, names and bytes
    assert gname == mname
    assert g == m, "\n".join(l for l in __import__("difflib").unified_diff(m.splitlines(), g.splitlines(), "manual", "gui", lineterm=""))
    # the JSON: type-strict, and the same bytes when the GUI changed nothing
    diff = R.strict_diff(R.strict_load(jg), R.strict_load(jm))
    assert not diff, diff
    if not row.answers and not row.patches:
        assert jg == jm
    # house rules on G
    if row.spin is True:
        assert "SPIN" in g.splitlines(), "SPIN is the CLI's default"
    if row.spin is False:
        assert "SPIN" not in g.splitlines()
    if row.band:
        assert re.search(r"^\s*-?\d+\s+\d+\s+1000\s", g, re.M), "BAND NSUB 1000"
    if row.keep_parent and row.mode == "opt2d12":
        parent = (root / row.src).with_suffix(".d12")
        if parent.is_file():
            p = parent.read_text()
            assert _functional(g) == _functional(p) or not _functional(p), (_functional(g), _functional(p))
            if _block(p, "TOLINTEG"):
                gt, pt = _block(g, "TOLINTEG")[0].split(), _block(p, "TOLINTEG")[0].split()
                if row.calc == "FREQ":
                    # the CLI tightens the integrals for frequencies (its FREQ preset): never looser than the parent
                    assert all(int(a) >= int(b) for a, b in zip(gt, pt)), (gt, pt)
                else:
                    assert gt == pt, (gt, pt)
            if _block(p, "SHRINK"):
                # the parent's Monkhorst-Pack mesh is kept; the CLI writes a SLAB's mesh in the
                # directional form ("0 16" / "8 8 1") whatever form the parent used ("8 8")
                assert _mesh(g) == _mesh(p), (_block(g, "SHRINK"), _block(p, "SHRINK"))


def test_follow_up_changes_only_the_functional(served, tmp_path):
    """Row 11 against row 8: the functional lines change, every other keyword line stays the parent's."""
    root, base = served
    a = gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "OPT"})["deck"]
    b = gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "OPT", "answers": {"functional": "HSE06"}})["deck"]
    import difflib
    changed = [l for l in difflib.ndiff(a.splitlines(), b.splitlines()) if l[:1] in "+-"]
    other = [l for l in changed if not re.search(r"PBE0|HSE06", l)]
    # the n-route writes the OPT block's criteria explicitly (the CLI's own defaults): allowed, reported
    assert all(re.match(r"^[+-] (TOLDEG|TOLDEX|TOLDEE|0\.0003|0\.0012|7)$", l) for l in other), other
    assert _block(b, "SHRINK") == _block(a, "SHRINK") and _block(b, "TOLINTEG") == _block(a, "TOLINTEG")


def test_k_points_are_read_only_for_cif2d12(served, tmp_path):
    """Row 24: convert works the mesh out from the cell; a JSON k_points is ignored, so the form offers no edit."""
    root, base = served
    r = gui_run(base, {"source": {"path": "structures/diamond.cif"}, "calc_type": "OPT"})
    assert not any("k_point" in f["id"] or "shrink" in f["id"] for f in r["form"])
    assert r["shrink"], r.get("shrink")
    data = json.loads(r["json_text"])
    data["k_points"] = 3 if data.get("k_points") != 3 else 4
    d = tmp_path / "k"
    (d / "cif").mkdir(parents=True)
    shutil.copy2(root / "structures" / "diamond.cif", d / "cif")
    (d / "k.json").write_text(json.dumps(data, indent=2))
    R.run_printed(["mace", "convert", "--batch", "--options_file", "k.json", "--cif_dir", "cif", "--output_dir", "out"], d)
    deck = next((d / "out").glob("*.d12")).read_text()
    assert _block(deck, "SHRINK") == _block(r["deck"], "SHRINK"), "convert ignores k_points (the CLI's behavior)"


def test_mesh_override_writes_k_points_for_all_files(served, tmp_path):
    """Row 25: the one key opt2d12 reads for one mesh; only SHRINK changes, and the hand-written config route agrees."""
    root, base = served
    a = gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "OPT"})
    b = gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "OPT", "patches": {"k_points_for_all_files": "6 6 6"}})
    assert json.loads(b["json_text"])["k_points_for_all_files"] in ("6 6 6", [6, 6, 6])
    import difflib
    changed = [l for l in difflib.ndiff(a["deck"].splitlines(), b["deck"].splitlines()) if l[:1] in "+-"]
    assert changed and all(re.match(r"^[+-] [\d ]+$", l) for l in changed), changed
    # the hand-written route: the saved parent settings plus the key, by hand
    data = json.loads(a["json_text"])
    data["k_points_for_all_files"] = "6 6 6"
    d = tmp_path / "k"
    shutil.copytree(root / "outputs", d / "in")
    (d / "c.json").write_text(json.dumps(data, indent=2))
    R.run_printed(["mace", "opt2d12", "--out-file", "in/Si_sp.out", "--config-file", "c.json", "--yes", "--output-dir", "out"], d)
    assert next((d / "out").glob("*.d12")).read_text() == b["deck"]


def test_unsaved_session_previews_and_save_asks_for_a_cif_first(served, tmp_path):
    """Row 7: a built structure (MgO 2x2x2 with an O vacancy, never saved): its preview is the deck the
    CLI writes from the same CIF bytes (Save As would write them); Save asks for the CIF first."""
    root, base = served
    st, _, b = U.post(base, "api/build/defect", {"source": {"path": "structures/MgO.cif"},
                                                 "params": {"kind": "vacancy", "site": 4, "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
                                                 "open_as": {"name": "MgO_VO"}})
    assert st == 200, b
    sess = b["session"]["session"]
    src = {"session": sess, "rev": b["session"]["rev"]}
    r = gui_run(base, {"source": src, "calc_type": "SP", "name": "MgO_VO"})
    assert r["deck_name"].startswith("MgO_VO_CRYSTAL_SP"), r["deck_name"]
    st, _, cif = U.post(base, "api/structure/export", {"source": src, "format": "cif", "cif_mode": "p1"})
    assert st == 200
    d = tmp_path / "m"
    (d / "cif").mkdir(parents=True)
    (d / "cif" / "MgO_VO.cif").write_bytes(cif if isinstance(cif, bytes) else str(cif).encode())
    R.run_cli_manual(["convert", "--cif_dir", "cif", "--output_dir", "out", "--save_options", "--options_file", "Jm.json"],
                     R.MANUAL["sp"], d)
    m = next((d / "out").glob("*.d12"))
    assert (m.name, m.read_text()) == (r["deck_name"], r["deck"])
    st, _, s = U.post(base, "api/setup/save", {"source": src, "calc_type": "SP", "json_text": r["json_text"]})
    assert st == 400 and "Save the structure as a CIF first" in s["error"]


def test_seekpath_is_required_for_band(tmp_path):
    """Step 4: with seekpath hidden, a 3D BAND setup shows the CLI's own refusal; no static path is made silently."""
    hide = tmp_path / "hide"
    hide.mkdir()
    (hide / "seekpath.py").write_text("raise ImportError('No module named seekpath (hidden by the test)')\n")
    root = R.served_root(tmp_path)
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_TEST_CLI_PYTHONPATH=str(hide))
    try:
        # the labels path is the CLI's own table; SeeK-path's full paths need seekpath
        r = gui_run(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "BAND",
                           "answers": {"band_format": "SeeK-path full paths"}})
        text = json.dumps(r.get("problems")) + (r.get("refused") or "") + (r.get("log") or "")
        assert r["deck"] is None or "seekpath" in text.lower(), r["deck"]
        assert "seekpath" in text.lower() and "static-kpath" in text, text[-2000:]
        assert any(p["severity"] == "error" for p in r["problems"])
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_follow_up_fmixing_changes_only_fmixing(served):
    """Row 23 (rg_q0, a smeared, spin-polarized parent): FMIXING 30 -> 40 changes that line only."""
    root, base = served
    a = gui_run(base, {"source": {"path": "data/rg/rg_q0.out"}, "calc_type": "SP"})
    b = gui_run(base, {"source": {"path": "data/rg/rg_q0.out"}, "calc_type": "SP", "answers": {"fmixing": "40"}})
    import difflib
    changed = [l for l in difflib.ndiff(a["deck"].splitlines(), b["deck"].splitlines()) if l[:1] in "+-"]
    assert changed == ["- 30", "+ 40"], changed
    assert json.loads(b["json_text"])["fmixing"] == 40
    assert _block(b["deck"], "SMEAR") == _block(a["deck"], "SMEAR")


def test_band_path_edit_keeps_the_kpath_editor(served):
    """A BAND preview with a band_path edit still returns the k-path editor's data (the unedited
    segments), before and after the unedited preview is cached: the editor stays after an edit."""
    root, base = served
    edit = [{"from": "GAMMA", "to": "X"}, {"from": "GAMMA", "to": "L"}, {"from": "L", "to": "W"},
            {"from": "W", "to": "GAMMA"}]
    body = {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "BAND"}
    first = gui_run(base, {**body, "patches": {"band_path": edit}})
    plain = gui_run(base, {**body, "patches": {}})
    again = gui_run(base, {**body, "patches": {"band_path": edit}})
    assert plain["kpath"] and plain["kpath"]["computed"], plain.get("refused") or plain.get("log")
    for r in (first, again):
        assert r["deck"], r.get("refused") or r.get("log")
        assert r.get("kpath"), "the k-path editor's data is missing after an edit"
        assert r["kpath"]["computed"] == plain["kpath"]["computed"]
        assert r["kpath"]["edited"] == edit
