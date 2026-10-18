"""The Plot Gallery's routes on the real `mace gui` subprocess (MACE 1.1.6),
against mace.plotting.gallery, the module `mace plotting --gallery` runs.

* gallery/collect for each kind of set: a folder, the Explorer's selection, a
  Batch Setup folder (its decks and the files it was made from) and a
  database query (the filtered materials' files below the database's
  folder); the cap; paths outside the served folder refused;
* gallery/pane is gallery.pane() (the same description, paths relative);
* gallery/export runs `mace plotting --gallery` and gives the same figure the
  CLI writes for the same command; gallery/csv is gallery.captions_csv().
The shipped demo files only."""
import json
import shutil
import subprocess
import sys
import time
import urllib.parse

import pytest

import guiharness as H
import p4util as U

sys.path[:0] = [p for p in (str(H.REPO_ROOT),) if p not in sys.path]
from mace.plotting import gallery as G  # noqa: E402


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gallery_api")
    root = H.examples_copy(tmp)
    many = root / "many"
    many.mkdir()
    for k in range(G.MAX_ITEMS + 3):
        shutil.copyfile(root / "structures" / "Si.cif", many / f"Si{k:03d}.cif")
    b = root / "batch_SP_20261014"
    (b / "decks").mkdir(parents=True)
    shutil.copyfile(root / "outputs" / "Si_sp.d12", b / "decks" / "Si_sp_new.d12")
    (b / "batch.json").write_text(json.dumps({"version": 1, "calc": "SP", "selection": ["structures/MgO.cif", "outputs/Si_sp.out"],
                                              "parents": {}}))
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        yield base, root
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def post(base, route, body):
    return U.req(base, "api/" + route, body)


def get(base, route, **q):
    return U.req(base, f"api/{route}?" + urllib.parse.urlencode(q))


def collect(base, kind, source, **extra):
    st, _, b = post(base, "gallery/collect", {"kind": kind, "source": source, **extra})
    assert st == 200, b
    return b


def test_collect_folder_and_captions(srv):
    base, root = srv
    b = collect(base, "structure", {"type": "folder", "folder": "structures"})
    assert [i["path"] for i in b["items"]] == [f"structures/{n}.cif" for n in ("diamond", "hBN", "LiF", "MgO", "NaCl", "Si")]
    assert b["total"] == 6 and not b["capped"] and b["message"] is None
    by = {r["material_id"]: r for r in b["captions"]["rows"]}
    assert by["MgO"]["formula"] == "Mg4O4" and by["MgO"]["file"] == "structures/MgO.cif"
    assert b["captions"]["databases"] == ["materials.db"]
    assert b["cli_inputs"] == ["structures"]
    assert b["command"] == "mace plotting --gallery structure structures --cols 3 -o gallery_structure.svg"
    # the module, on the same folder, finds the same files with the same captions
    mod = G.collect([root / "structures"], "structure")
    assert [str(p.relative_to(root)) for p in mod["items"]] == [i["path"] for i in b["items"]]


def test_collect_selection_batch_and_db(srv):
    base, root = srv
    b = collect(base, "scf", {"type": "paths", "paths": ["outputs/Si_sp.out", "outputs/graphene_slab_sp.out"]})
    assert [i["path"] for i in b["items"]] == ["outputs/Si_sp.out", "outputs/graphene_slab_sp.out"]
    assert [r["energy_eV"] is not None for r in b["captions"]["rows"]] == [True, True]   # total_energy (Ha) from the db
    # a Batch Setup folder: its decks and the files it was made from
    b = collect(base, "structure", {"type": "batch", "batch": "batch_SP_20261014"})
    assert [i["path"] for i in b["items"]] == ["batch_SP_20261014/decks/Si_sp_new.d12", "structures/MgO.cif", "outputs/Si_sp.out"]
    assert b["source_label"] == "Batch batch_SP_20261014"
    st, _, err = post(base, "gallery/collect", {"kind": "structure", "source": {"type": "batch", "batch": "structures"}})
    assert st == 400 and "not a Batch Setup folder" in err["error"]
    # a database query: the filtered materials' files below the database's folder
    b = collect(base, "structure", {"type": "db", "db": "materials.db", "filters": {"q": "Mg"}})
    assert [i["path"] for i in b["items"]] == ["structures/MgO.cif"]
    assert b["source_label"] == "1 materials of materials.db"
    b = collect(base, "scf", {"type": "db", "db": "materials.db", "filters": {}})
    assert {i["path"] for i in b["items"]} == {"outputs/Si_sp.out", "outputs/graphene_slab_sp.out", "outputs/Ag2Br3_opt_unfinished.out"}


def test_collect_cap_and_refusals(srv):
    base, root = srv
    b = collect(base, "structure", {"type": "folder", "folder": "many"})
    assert len(b["items"]) == G.MAX_ITEMS and b["total"] == G.MAX_ITEMS + 3 and b["capped"]
    assert b["message"].startswith(f"Showing the first {G.MAX_ITEMS} of {G.MAX_ITEMS + 3} files")
    st, _, err = post(base, "gallery/collect", {"kind": "structure", "source": {"type": "folder", "folder": "../.."}})
    assert st == 403, err
    st, _, err = post(base, "gallery/collect", {"kind": "phonons", "source": {"type": "folder"}})
    assert st == 400 and "gallery kind" in err["error"]
    st, _, err = get(base, "gallery/pane", kind="band", path="structures/Si.cif", emin="3", emax="1")
    assert st == 400 and "energy window" in err["error"]


def test_pane_is_the_modules(srv):
    base, root = srv
    for kind, rel in (("structure", "structures/MgO.cif"), ("scf", "outputs/graphene_slab_sp.out"),
                      ("opt", "outputs/Ag2Br3_opt_unfinished.out")):
        st, _, b = get(base, "gallery/pane", kind=kind, path=rel)
        assert st == 200, b
        mod = json.loads(json.dumps(G.pane(root / rel, kind)))
        assert b["path"] == rel
        b.pop("path"), mod.pop("path")
        assert b == mod, kind
    # an unreadable file: the pane says why
    st, _, b = get(base, "gallery/pane", kind="band", path="structures/Si.cif")
    assert st == 200 and b["error"] and b["panels"] == []


def _wait(base, r, timeout=300):
    st, _, body = r
    if st == 200:
        return body
    assert st == 202, body
    tid = body["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        s, _, res = U.req(base, f"api/tasks/{tid}/result")
        if s == 200 and isinstance(res, dict) and res.get("key"):
            return res
        time.sleep(0.3)
    raise AssertionError("the export task did not finish")


def test_export_is_the_clis_figure_and_csv(srv, tmp_path):
    base, root = srv
    paths = ["structures/MgO.cif", "structures/NaCl.cif", "outputs/Si_sp.d12"]
    res = _wait(base, post(base, "gallery/export", {"kind": "structure", "paths": paths, "cols": 2, "format": "png", "detach": True}))
    assert res["name"] == "gallery_structure.png"
    assert res["command"] == "mace plotting --gallery structure " + " ".join(paths) + " --cols 2 -o gallery_structure.png"
    st, _, png = U.req(base, "api/plot-image?" + urllib.parse.urlencode({"key": res["key"], "name": res["name"]}))
    assert st == 200 and png[:4] == b"\x89PNG"
    # the CLI, run by hand with the same command on the same files, draws the same picture
    out = tmp_path / "gallery_structure.png"
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *G.command("structure", paths, 2, str(out)),
                        "--max", "3"], cwd=str(root), capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    from PIL import Image
    import io
    a, b = Image.open(io.BytesIO(png)).convert("RGB"), Image.open(out).convert("RGB")
    assert a.size == b.size
    assert a.tobytes() == b.tobytes()
    # the caption table
    st, hdr, text = post(base, "gallery/csv", {"paths": paths, "db": "materials.db"})
    assert st == 200
    caps = G.captions([root / x for x in paths], db=root / "materials.db", rel=lambda q: str(q.relative_to(root)))
    assert text.decode() == G.captions_csv(caps["rows"])
    st, _, err = post(base, "gallery/export", {"kind": "structure", "paths": ["../x.cif"], "format": "svg"})
    assert st == 403
    st, _, err = post(base, "gallery/export", {"kind": "structure", "paths": paths, "format": "jpg"})
    assert st == 400 and "format" in err["error"]
