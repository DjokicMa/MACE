"""What `mace gui` writes, through the real server (`mace_cli gui` subprocess).

Save CIF (the classic panel) and Save JSON (deck settings) follow saving.py's
rules: the target is a plain name inside --dir, a symbolic link is never
written through (it may lead out of the folder), and the write is atomic. A
refused save changes nothing, inside the folder or outside it.
"""
import importlib.util
import json
import shutil

import pytest

from test_gui_examples import EXAMPLE_CONFIGS, EXAMPLES, _json, _start

needs_ase = pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="ase is not installed")


@pytest.fixture()
def served(tmp_path):
    root = tmp_path / "served"
    shutil.copytree(EXAMPLES, root)
    outside = tmp_path / "outside"
    outside.mkdir()
    proc, base, _ = _start(root)
    try:
        yield root, outside, base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def _tree(path):
    return sorted((str(p.relative_to(path)), p.read_bytes() if p.is_file() and not p.is_symlink() else None)
                  for p in path.rglob("*"))


def test_save_json_refuses_a_symlinked_target(served):
    root, outside, base = served
    victim = outside / "victim.json"
    victim.write_text("keep me\n")
    (root / "x.json").symlink_to(victim)
    before_out, before_in = _tree(outside), _tree(root)
    config = json.loads((EXAMPLE_CONFIGS / "high_accuracy_sp.json").read_text())
    st, r = _json(base, "api/deck/save", data={"mode": "opt2d12", "name": "x.json", "json": config,
                                                "overwrite": True})
    assert st == 403 and "symbolic link" in r["error"], r
    assert victim.read_text() == "keep me\n"
    assert _tree(outside) == before_out and _tree(root) == before_in


def test_save_json_writes_atomically_inside_the_folder(served):
    root, outside, base = served
    config = json.loads((EXAMPLE_CONFIGS / "high_accuracy_sp.json").read_text())
    st, r = _json(base, "api/deck/save", data={"mode": "opt2d12", "name": "ok.json", "json": config})
    assert st == 200 and r["saved"] == "ok.json", r
    assert json.loads((root / "ok.json").read_text()) == config
    assert not [p for p in root.iterdir() if p.name.startswith(".ok.json.")]      # no temp file left
    assert _tree(outside) == []


@needs_ase
def test_save_cif_refuses_a_symlinked_target(served):
    root, outside, base = served
    victim = outside / "victim.cif"
    victim.write_text("keep me\n")
    (root / "x.cif").symlink_to(victim)
    st, r = _json(base, "api/edit/open", data={"path": "structures/MgO.cif"})
    assert st == 200, r
    before_out = _tree(outside)
    st, r = _json(base, "api/edit/save_cif", data={"session": r["session"], "name": "x.cif", "overwrite": True})
    assert st == 403 and "symbolic link" in r["error"], r
    assert victim.read_text() == "keep me\n"
    assert _tree(outside) == before_out
    assert not (root / "x.cif.bak").exists()


@needs_ase
def test_save_cif_overwrite_keeps_one_backup(served):
    root, outside, base = served
    st, r = _json(base, "api/edit/open", data={"path": "structures/MgO.cif"})
    sid = r["session"]
    assert _json(base, "api/edit/save_cif", data={"session": sid, "name": "new.cif"})[0] == 200
    first = (root / "new.cif").read_bytes()
    st, r = _json(base, "api/edit/save_cif", data={"session": sid, "name": "new.cif", "overwrite": True,
                                                   "mode": "symmetrized"})
    assert st == 200, r
    assert (root / "new.cif.bak").read_bytes() == first
    assert _tree(outside) == []


def _setup_result(base, w, st):
    """A 202 task answer: poll its result."""
    import time
    if st == 202:
        tid = w["task"]["id"]
        end = time.time() + 180
        while time.time() < end:
            st, w = _json(base, f"api/tasks/{tid}/result")
            if st not in (202, 409) or w.get("task", {}).get("state") not in ("queued", "running"):
                break
            time.sleep(0.2)
    return st, w


@needs_ase
def test_write_decks_now_never_writes_through_a_symlink(served):
    """"Write decks now" runs the printed CLI command. A deck name that is
    already a symbolic link in the output folder (planted by someone else in
    a shared folder) is refused: the file it points to is unchanged, and no
    deck is moved in."""
    root, outside, base = served
    src = {"path": "structures/MgO.cif"}
    st, r = _json(base, "api/setup/run", data={"source": src, "calc_type": "SP", "wait": 120})
    assert st == 200 and r.get("deck_name"), r
    folder = root / "setup" / "MgO"
    folder.mkdir(parents=True)
    victim = outside / "victim.d12"
    victim.write_text("keep me\n")
    (folder / r["deck_name"]).symlink_to(victim)
    st, w = _json(base, "api/setup/save", data={"source": src, "calc_type": "SP", "json_text": r["json_text"],
                                                  "layout": "own", "overwrite": True, "write_decks": True})
    st, w = _setup_result(base, w, st)
    assert st != 200 and "symbolic link" in (w["task"]["error"] or ""), w
    assert victim.read_text() == "keep me\n"
    assert _tree(outside) == [("victim.d12", b"keep me\n")]
    assert (folder / r["deck_name"]).is_symlink()
    assert not [p for p in folder.iterdir() if p.name.startswith(".mace-decks-")]     # the private folder is gone


@needs_ase
def test_write_decks_now_keeps_a_bak_of_a_replaced_deck(served):
    """A deck already there is replaced as every other save replaces a
    file: through a rename, keeping <deck>.bak, and listed in "written"."""
    root, outside, base = served
    src = {"path": "structures/MgO.cif"}
    st, r = _json(base, "api/setup/run", data={"source": src, "calc_type": "SP", "wait": 120})
    assert st == 200, r
    body = {"source": src, "calc_type": "SP", "json_text": r["json_text"], "layout": "own", "overwrite": True,
            "write_decks": True}
    st, w = _json(base, "api/setup/save", data=body)
    st, w = _setup_result(base, w, st)
    deck = f"setup/MgO/{r['deck_name']}"
    assert st == 200 and w["written"] == [deck] and w["backups"] == [], w
    assert (root / deck).read_text() == r["deck"]
    (root / deck).write_text("an older deck\n")
    st, w = _json(base, "api/setup/save", data=body)
    st, w = _setup_result(base, w, st)
    assert st == 200 and w["written"] == [deck] and w["backups"] == [deck + ".bak"], w
    assert (root / deck).read_text() == r["deck"]
    assert (root / (deck + ".bak")).read_text() == "an older deck\n"
    assert not [p for p in (root / "setup" / "MgO").iterdir() if p.name.startswith(".mace-decks-")]
    assert _tree(outside) == []
