"""The served folder's mace-folder.toml through the real `mace gui` (MACE
1.1.6): /api/boot's folder layer, /api/folder-settings (Save to this folder,
Reset to my default), the checks (keys a folder cannot hold, bad values), a
file above the served folder or one MACE refuses (read, never written), and
nothing written into the folder without a save."""
import json

import pytest

import guiharness as H
from test_shell_api import req

from mace.utils import folder_settings as F


def _start(root, xdg):
    # Open Folder may switch within the served copy (the default roots are HOME and
    # scratch: a pytest tmp_path under /tmp is outside them)
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_ROOTS=str(root))
    return proc, base


def _stop(proc):
    proc.terminate()
    proc.wait(timeout=15)


@pytest.fixture
def served(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base = _start(root, xdg)
    yield base, root, xdg
    _stop(proc)


def boot(base):
    st, _, body = req(base, "api/boot")
    assert st == 200, body
    return json.loads(body)


def save(base, body):
    st, _, out = req(base, "api/folder-settings", data=body)
    return st, json.loads(out)


def test_no_folder_file_changes_nothing(served):
    base, root, xdg = served
    H.local_only()
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    b = boot(base)
    f = b["folder"]
    assert f["file"] is None and f["gui"] == {} and f["cli"] == {} and f["warnings"] == []
    assert f["can_save"] and f["target"] == str(root / F.FILENAME)
    assert f["keys"]["atomScale"] == "viewer.atom_scale" and "theme" not in f["keys"]
    # the user's settings route is as before, and nothing was written into the folder
    st, _, body = req(base, "api/settings")
    assert st == 200 and json.loads(body)["settings"] == {}
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before


def test_save_to_folder_and_reset(served):
    base, root, xdg = served
    H.local_only()
    st, v = save(base, {"set": {"atomScale": 0.5, "plots.energyWindow": [-4, 4], "energyUnit": "meV",
                                "database.columns": ["formula", "status"]}})
    assert st == 200, v
    f = root / F.FILENAME
    text = f.read_text()
    assert text.startswith("# MACE folder settings") and "atom_scale = 0.5" in text and 'energy = "meV"' in text
    assert v["file"] == str(f) and v["gui"] == {"atomScale": 0.5, "plots.energyWindow": [-4, 4], "energyUnit": "meV",
                                                 "database.columns": ["formula", "status"]}
    assert boot(base)["folder"]["gui"]["atomScale"] == 0.5
    # the user's own file is untouched by a folder save
    st, _, body = req(base, "api/settings")
    assert json.loads(body)["settings"] == {}
    # Reset to my default: out of the file, the rest kept (and the file's comments)
    st, v = save(base, {"unset": ["atomScale"]})
    assert st == 200 and "atomScale" not in v["gui"] and v["gui"]["energyUnit"] == "meV"
    assert "atom_scale = " not in f.read_text().replace("# atom_scale", "") and "# color_scheme" in f.read_text()
    # a blank database path is "the built-in choice": nothing is kept for it
    st, v = save(base, {"set": {"defaultDb": "materials.db"}})
    assert st == 200 and v["gui"]["defaultDb"] == "materials.db"
    st, v = save(base, {"set": {"defaultDb": ""}})
    assert st == 200 and "defaultDb" not in v["gui"]


def test_bad_keys_and_values_are_refused(served):
    base, root, xdg = served
    H.local_only()
    for body, msg in (({"set": {"theme": "dark"}}, "cannot be saved to a folder"),
                      ({"set": {"atomScale": 7}}, "atom_scale must be a number"),
                      ({"set": {"defaultDb": "../x.db"}}, "must not leave this folder"),
                      ({"set": {"defaultDb": "/etc/passwd"}}, "relative path"),
                      ({"set": []}, "set must be")):
        st, out = save(base, body)
        assert st == 400 and msg in out["error"], (body, out)
    assert not (root / F.FILENAME).exists()


def test_a_file_above_the_folder_is_read_not_written(tmp_path):
    H.local_only()
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    above = tmp_path / F.FILENAME
    above.write_text('[viewer]\nprojection = "perspective"\n[database]\nfile = "demo/materials.db"\n'
                     '[submit]\nntasks = 16\n[viewer.extra]\nx = 1\n')
    proc, base = _start(root, xdg)
    try:
        f = boot(base)["folder"]
        assert f["file"] == str(above) and not f["can_save"] and "above this folder" in f["why_not"]
        assert f["gui"] == {"projection": "perspective", "defaultDb": "materials.db"}
        assert f["cli"] == {"submit.ntasks": 16} and "unknown keys ignored: viewer.extra" in f["warnings"][0]
        before = above.read_text()
        st, out = save(base, {"set": {"projection": "orthographic"}})
        assert st == 400 and "above this folder" in out["error"] and above.read_text() == before
        assert not (root / F.FILENAME).exists()
    finally:
        _stop(proc)


def test_a_refused_file_is_never_overwritten(tmp_path):
    H.local_only()
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (root / F.FILENAME).write_text("[viewer\n")
    proc, base = _start(root, xdg)
    try:
        f = boot(base)["folder"]
        assert not f["can_save"] and "not valid TOML" in f["why_not"] and f["gui"] == {}
        st, out = save(base, {"set": {"atomScale": 0.5}})
        assert st == 400 and (root / F.FILENAME).read_text() == "[viewer\n"
    finally:
        _stop(proc)


def test_open_folder_reads_the_new_folders_file(served, tmp_path):
    base, root, xdg = served
    H.local_only()
    other = root / "structures"
    (other / F.FILENAME).write_text('[viewer]\nshow_cell = false\n')
    assert boot(base)["folder"]["gui"] == {}
    st, _, body = req(base, "api/open-folder", data={"path": str(other)})
    assert st == 200, body
    assert boot(base)["folder"]["gui"] == {"showCell": False}


def test_submit_preview_and_run_read_the_folder_file(tmp_path):
    """GUI Submit runs `mace submit`: the staged preview reads the real
    folder's file (MACE_FOLDER_SETTINGS), equals the real run, and a change
    to the file after the preview needs a new one."""
    H.local_only()
    import submit_util as SU
    srv = SU.Served(tmp_path / "srv")
    try:
        d = srv.decks("fp/decks", ["m0"], SU.SP_D12)
        folder_file = d.parent / F.FILENAME             # the folder `cd fp && mace submit decks` runs in
        folder_file.write_text('[submit]\nntasks = 12\nwalltime = "6:00:00"\n')
        st, prev = srv.preview(["fp/decks"])
        assert st == 200 and prev["code"] == 0, prev.get("cli_log")
        (text,) = prev["scripts"].values()
        assert "#SBATCH --ntasks=12" in text and "#SBATCH -t 0-06:00:00" in text
        st, res = srv.run(prev["preview_id"])
        assert st == 200 and res["code"] == 0 and res["script_drift"] == [], res
        assert "#SBATCH --ntasks=12" in (d / "m0.sh").read_text()
        # the folder file changed after the preview: the preview no longer holds
        srv.decks("fp/more", ["m1"], SU.SP_D12)
        st, prev = srv.preview(["fp/more"])
        assert st == 200 and prev["code"] == 0
        folder_file.write_text('[submit]\nntasks = 24\n')
        st, res = srv.run(prev["preview_id"])
        assert st == 409 and "changed since the preview" in res["error"]
    finally:
        srv.stop()
