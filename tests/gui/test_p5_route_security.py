"""Every route Phase 5 adds keeps the server's rules (plan T1, spec section 7 test strategy):

* no token, or a wrong one: 403;
* a Host that is not the loopback: refused (the server answers 421
  Misdirected Request, as every route has since Phase 1);
* a POST without X-Requested-With, or sent as text/plain (a "simple" CORS
  request): 403;
* a path parameter that leaves --dir (../, an absolute path, a symlink
  inside --dir that points out, a NUL byte, %2e%2e/) is refused and the
  file outside is never read; a write target likewise.

A meta-test lists the server's API routes and fails when a route added
after Phase 4 is missing from P5_ROUTES.
"""
import json
import os
import re
import time
import urllib.parse

import pytest

import guiharness as H
import p4util as U

# (method, route, body or query, the path parameters it takes, writes?)
P5_ROUTES = [
    ("GET", "setup/schema", {}, [], False),
    ("POST", "setup/run", {"source": {"path": "structures/MgO.cif"}, "calc_type": "OPT", "wait": 0}, ["source.path"], False),
    ("POST", "setup/defaults", {"source": {"path": "structures/MgO.cif"}, "wait": 0}, ["source.path"], False),
    ("POST", "setup/apply", {"source": {"path": "structures/MgO.cif"}, "wait": 0}, ["source.path"], False),
    ("POST", "setup/preview", {"source": {"path": "structures/MgO.cif"}, "wait": 0}, ["source.path"], False),
    ("POST", "setup/save", {"source": {"path": "structures/MgO.cif"}, "json_text": "{}", "name": "x.json"}, ["source.path"], True),
    ("POST", "setup/preflight", {"deck_name": "x.d12", "deck": "TEST\nEND\n", "wait": 0}, [], False),
    ("POST", "deck/preview", {"structure": "structures/MgO.cif", "mode": "cif2d12", "json": {}}, ["structure"], False),
    ("GET", "plot", {"path": "structures/MgO.cif", "detach": "1"}, ["path"], False),
    # the k-path editor's Brillouin zone view (1.1.6)
    ("GET", "bz", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # the Inspector's Brillouin zone section of a band plot (1.1.6): the run's k-path
    ("GET", "kpath", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # a band + DOS pair from two folders (1.1.6): the check, and Publication Figure's other half
    ("GET", "bandos-pair", {"band": "outputs/Si_sp.out", "dos": "outputs/Si_sp.out"}, ["band", "dos"], False),
    ("GET", "plot", {"path": "outputs/Si_sp.out", "kind": "bandos", "partner": "outputs/Si_sp.out", "detach": "1"}, ["partner"], False),
    # the panel context (1.1.6): the run a plot, volume or log came from
    ("GET", "related", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # the Plot Gallery (1.1.6): its gallery/* routes
    ("GET", "gallery/pane", {"kind": "structure", "path": "structures/MgO.cif"}, ["path"], False),
    ("POST", "gallery/collect", {"kind": "structure", "paths": ["structures/MgO.cif"]}, [], False),
    ("POST", "gallery/collect", {"kind": "structure", "source": {"type": "folder", "folder": "structures"}}, ["source.folder"], False),
    ("POST", "gallery/csv", {"kind": "structure", "paths": ["structures/MgO.cif"]}, [], False),
    ("POST", "gallery/export", {"kind": "structure", "paths": ["structures/MgO.cif"], "detach": True}, [], False),
    # Batch Setup's STUTSC check (1.1.6): the selected files' elements
    ("POST", "batch/stutsc", {"files": ["structures/MgO.cif"]}, [], False),
    # the Workflows view and builder (1.1.6): its workflow/* routes
    ("GET", "workflow/list", {}, [], False),
    ("GET", "workflow/status", {"id": "workflow_20261015_000000_abcd", "folder": "structures"}, ["folder"], False),
    ("POST", "workflow/control", {"folder": "", "id": "workflow_20261015_000000_abcd", "action": "pause"}, ["folder"], True),
    # the served folder's mace-folder.toml (1.1.6)
    ("GET", "folder-settings", {}, [], False),
    ("POST", "folder-settings", {"set": {}, "unset": []}, [], True),
    ("GET", "text/lines", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "spectrum", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "freq", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "freq/mode", {"path": "outputs/Si_sp.out", "k": "1"}, ["path"], False),
    ("GET", "db/page", {"path": "materials.db"}, ["path"], False),
    ("GET", "db/count", {"path": "materials.db"}, ["path"], False),
    ("GET", "db/stats", {"path": "materials.db"}, ["path"], False),
    ("GET", "db/props", {"path": "materials.db"}, ["path"], False),
    ("GET", "db/export", {"path": "materials.db", "format": "csv"}, ["path"], False),
    # Area C (PLAN_analysis): the Database view's live refresh (stat only)
    ("GET", "db/stamp", {"path": "materials.db"}, ["path"], False),
    # Create or Update Database from a Folder (C10): the dialog facts, the preview, the build (a write)
    ("GET", "db/build/info", {"folder": "outputs"}, ["folder"], False),
    ("POST", "db/build/preview", {"folder": "outputs", "wait": 0}, ["folder"], False),
    ("POST", "db/build", {"folder": "outputs", "db": "outputs/x.db", "confirm": True, "wait": 0}, ["folder", "db"], True),
    # run analysis (C2, C6, C18)
    ("GET", "out/convergence", {"path": "outputs/Ag2Br3_opt_unfinished.out"}, ["path"], False),
    ("GET", "out/problems", {"path": "outputs/Ag2Br3_opt_unfinished.out"}, ["path"], False),
    ("GET", "out/meta", {"path": "outputs/Ag2Br3_opt_unfinished.out", "wait": "0"}, ["path"], False),
    ("GET", "out/live", {"path": "outputs/Ag2Br3_opt_unfinished.out"}, ["path"], False),
    # 1.1.6: the time-left levels for the run card
    ("GET", "out/progress", {"path": "outputs/Ag2Br3_opt_unfinished.out"}, ["path"], False),
    ("POST", "out/track", {"path": "outputs/Ag2Br3_opt_unfinished.out",
                           "measurement": {"type": "distance", "atoms": [{"base": 0}, {"base": 1}]}}, ["path"], False),
    # File History and the protected-kind check (C19, C20)
    ("GET", "write/check", {"path": "structures/MgO.cif"}, ["path"], False),
    ("GET", "history", {"path": "structures/MgO.cif"}, ["path"], False),
    ("GET", "history/file", {"id": "20260101T000000Z_1_MgO.cif"}, [], False),
    ("POST", "history/restore", {"id": "20260101T000000Z_1_MgO.cif", "path": "structures/MgO.cif"}, ["path"], True),
    ("POST", "text/find", {"path": "outputs/Si_sp.out", "query": "ENERGY"}, ["path"], False),
    ("GET", "compare/input", {"path": "outputs/Ag2Br3_opt_unfinished.out"}, ["path"], False),
    # Clean's tiers (the method list and its recheck): the same token, Host and header rules
    ("GET", "clean/methods", {}, [], False),
    ("POST", "clean/methods/recheck", {}, [], False),
    # the GUI profile (gui.json) and per-document view options (gui-docprefs.json): the XDG config dir, never the folder
    ("GET", "profile", {}, [], False),
    ("POST", "profile", {"memory": {"set": {"x": 1}}}, [], False),
    ("POST", "profile/reset", {"sections": []}, [], False),
    ("POST", "profile/docprefs", {"kinds": {}}, [], False),
    ("GET", "docprefs", {"kind": "display", "path": "structures/MgO.cif"}, ["path"], False),
    ("POST", "docprefs", {"kind": "display", "path": "structures/MgO.cif", "values": {}}, ["path"], False),
    # element color palettes (1.1.6): $XDG_CONFIG_HOME/mace/element_colors.json, never the folder
    ("GET", "palettes", {}, [], False),
    ("POST", "palettes", {"op": "default", "id": "vesta"}, [], False),
    # element legends (1.1.6): reads structure files in the folder, writes nothing (paths are checked in test_themes_palettes_e2e)
    ("POST", "legend", {"paths": ["structures/MgO.cif"]}, [], False),
    # connectivity (PLAN_connect): Open Folder and the managed-mode control routes
    ("GET", "folders", {}, [], False),
    ("POST", "open-folder", {"path": "/nonexistent-folder-zq"}, [], False),
    ("POST", "managed/", {"watcher": "w"}, [], False),
    ("POST", "shutdown", {"mode": "cancel"}, [], False),
    # the update notice (1.1.6; only refused calls here: an accepted check would fetch from origin)
    ("GET", "update", {}, [], False),
    ("POST", "update/check", {}, [], False),
    ("POST", "update/choose", {"choice": "later"}, [], False),
    # sessions (one-time links, activity, expiry)
    ("POST", "login-code", {}, [], False),
    ("GET", "session", {}, [], False),
    ("POST", "session/touch", {}, [], False),
    ("POST", "diagnostics", {"client": {}}, [], False),
    # the Cluster view (area F): read-only; the scan's root must stay in an allowed folder
    ("GET", "cluster/", {}, [], False),
    ("GET", "cluster/capabilities", {}, [], False),
    ("GET", "cluster/quotas", {}, [], False),
    ("GET", "cluster/job", {"id": "1"}, [], False),
    ("POST", "cluster/scan", {"root": "outputs"}, ["root"], False),
    ("POST", "cluster/estimate", {"deck_rel": "outputs/x.d12"}, ["deck_rel"], False),
    ("POST", "cluster/predict", {"script_rel": "outputs/x.sh"}, ["script_rel"], False),
    # plan E (CLI parity): file kinds by content, the info and properties documents, job-log links
    ("GET", "filekind", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "fileinfo", {"path": "outputs/Si_sp.d12"}, ["path"], False),
    ("GET", "props", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "joblog/link", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "transport", {"path": "outputs/Si_sp.out", "quantity": "zt", "kappa_l": "1.5"}, ["path", "kappa_l"], False),
    ("GET", "transport-zt", {"path": "outputs/Si_sp.out", "kappa_l": "1.5"}, ["path", "kappa_l"], False),
    ("GET", "convergence", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "map", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # plan E5: the Volume document (a cube's grid, slices, MEP range, differences)
    ("GET", "volume", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "volume/grid", {"key": "x"}, [], False),
    ("GET", "volume/slice", {"path": "outputs/Si_sp.out", "axis": "c", "index": "0"}, ["path"], False),
    ("POST", "volume/check", {"a": "outputs/Si_sp.out", "b": "outputs/Si_sp.out"}, ["a", "b"], False),
    ("POST", "volume/mep_range", {"path": "outputs/Si_sp.out", "vertices": [0, 0, 0]}, ["path"], False),
    ("POST", "volume/combine", {"paths": ["outputs/Si_sp.out", "outputs/Si_sp.out"], "coefficients": [1, -1]}, [], False),
    ("POST", "volume/save", {"paths": ["outputs/Si_sp.out", "outputs/Si_sp.out"], "coefficients": [1, -1], "name": "x.CUBE"}, [], True),
    # CRYSTAL23 ORBITALS: the orbital picker's index, grids and Save Cube (.molden files only)
    ("GET", "orbitals", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "orbitals/grid", {"path": "outputs/Si_sp.out", "k": "1", "mo": "1"}, ["path"], False),
    ("POST", "orbitals/save", {"path": "outputs/Si_sp.out", "k": 1, "mos": [1]}, ["path"], True),
    # Analysis › Structure Analysis (1.1.6): mace analyze in the GUI
    ("GET", "analysis/info", {}, [], False),
    ("POST", "analysis/run", {"tool": "pxrd", "source": {"path": "structures/MgO.cif"}, "params": {}, "wait": 0,
                              "pattern_path": "outputs/Si_sp.out"}, ["source.path", "pattern_path"], False),
    ("POST", "analysis/rcsr/import", {"path": "structures/MgO.cif"}, ["path"], False),
    # 1.1.6: a porosity result's surfaces at another mesh quality (reads the structure, writes nothing)
    ("POST", "analysis/remesh", {"source": {"path": "structures/MgO.cif"}, "params": {}, "mesh": {"quality": "draft"}, "wait": 0},
     ["source.path"], False),
    # the Magnetic style's Mulliken spins (1.1.6): reads the output (or the one beside a structure), writes nothing
    ("GET", "spins", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # a cube's plane (1.1.6, mace slice in the GUI): a read-only pool task, as volume/slice
    ("GET", "volume/plane", {"path": "outputs/Si_sp.out", "mode": "hkl"}, ["path"], False),
    # File › Export Animation (1.1.6): frames and the encoded file live in the server's cache, never the folder
    ("GET", "anim/caps", {}, [], False),
    ("GET", "anim/result", {"key": "x"}, [], False),
    ("POST", "anim/start", {}, [], False),
    ("POST", "anim/frame", {"key": "x", "index": 0, "data": "data:image/png;base64,"}, [], False),
    ("POST", "anim/encode", {"key": "x", "format": "gif"}, [], False),
    ("POST", "anim/render", {"path": "outputs/Si_sp.out", "format": "gif"}, ["path"], False),
    ("POST", "anim/cancel", {"key": "x"}, [], False),
    # Settings › ML potentials (1.1.6, mace mlip install in the GUI): status only here; plan/install/record/fetch
    # start real installs, so their POST-only, token and loopback checks are in test_mlip_routes_api.py
    ("GET", "mlip/status", {}, [], False),
    # the Calculators panel (1.1.6, mace calc in the GUI): Save writes beside the document or into save_dir
    ("GET", "calc/backends", {}, [], False),
    ("POST", "calc/backends/recheck", {}, [], False),
    ("POST", "calc/run", {"backend": "uff", "task": "sp", "source": {"path": "structures/MgO.cif"}, "save": True,
                          "save_dir": "outputs"}, ["source.path", "save_dir"], True),
    # 1.1.6: the Phonopy, Mechanical Properties and Relaxation Time dialogs
    ("GET", "phonopy/status", {"dir": "outputs"}, ["dir"], False),
    ("GET", "phonopy/data", {"dir": "outputs"}, ["dir"], False),
    ("POST", "phonopy/generate", {"parent": "outputs/Si_sp.out", "dim": "2 2 2", "wait": 0}, ["parent"], True),
    ("POST", "phonopy/collect", {"dir": "outputs", "check": True, "wait": 0}, ["dir"], True),
    ("GET", "elastic/analyze", {"path": "outputs/Si_sp.out"}, ["path"], False),
    # the results document of an ELASTIC, EOS, QHA or CPHF output (1.1.6, js/docs/mech.js)
    ("GET", "elastic/document", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "elastic/figure", {"path": "outputs/Si_sp.out", "view": "youngs"}, ["path"], False),
    # the enthalpy plot's other phases (1.1.6 mechmore: mace analyze elastic --compare)
    ("GET", "elastic/figure", {"path": "outputs/Si_sp.out", "view": "enthalpy", "compare": "outputs/Si_sp.out"}, ["path", "compare"], False),
    ("POST", "elastic/generate", {"parent": "outputs/Si_sp.out", "settings": {"mode": "ELASTCON"}, "wait": 0},
     ["parent"], True),
    ("GET", "tau/series", {"dir": "outputs"}, ["dir"], False),
    ("POST", "tau/strain", {"parent": "outputs/Si_sp.d12", "axis": "x"}, ["parent"], True),
    ("POST", "tau/fit", {"outputs": ["outputs/Si_sp.out"] * 3, "carrier": "electron", "m_star": 0.3}, [], False),
    # the Properties tab (1.1.6)
    ("GET", "properties/run", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("GET", "properties/compute", {"path": "outputs/Si_sp.out", "wait": 0}, ["path"], False),
    ("GET", "properties/value", {"db": "materials.db", "calc": "none", "name": "total_energy_au"}, ["db"], False),
    ("POST", "properties/extract", {"path": "outputs/Si_sp.out", "db": "materials.db", "wait": 0}, ["path", "db"], True),
    # the Properties tab's Add Reference… (reference energies): the picker's data, and the extraction with --reference
    ("GET", "properties/references", {"path": "outputs/Si_sp.out"}, ["path"], False),
    ("POST", "properties/references", {"path": "outputs/Si_sp.out", "db": "materials.db", "property": "reaction_energy",
                                       "references": {"reactants": ["none"]}, "wait": 0}, ["path", "db"], True),
]
# routes that only exist with MACE_GUI_TEST_HOOKS=1
TEST_ONLY = {"test/cli"}
# Phase 4 routes Phase 5 changed: checked here too
CHANGED = [("GET", "text", {"path": "outputs/Si_sp.out", "offset": "100"}, ["path"], False),
           ("GET", "compare", {"a": "structures/MgO.cif", "b": "structures/MgO.cif"}, ["a", "b"], False)]
PHASE4 = {"bands", "boot", "build/", "clean", "clean_accept", "clean_discard", "close", "compare", "db", "db/material",
          "deck/preview", "deck/save", "deck_preview", "defect/info", "dos", "edit/", "edit/state", "events", "files",
          "from_structure", "held", "held/claim", "held/release", "info", "jump", "new", "op", "open", "out/step",
          "out/summary", "plot", "plot-image", "raw", "resources", "save_cif", "settings", "structure", "structure/export",
          "structure/save", "symmetrize/sweep", "symmetry", "symmetry/compute", "tasks", "tasks/", "templates", "text",
          "trajectory", "trajectory/compare"}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p5sec")
    outside = tmp / "outside"
    outside.mkdir()
    (outside / "secret.cif").write_text((H.EXAMPLES / "structures" / "MgO.cif").read_text())
    (outside / "secret.out").write_text((H.EXAMPLES / "outputs" / "Si_sp.out").read_text())
    (outside / "secret.db").write_bytes((H.EXAMPLES / "materials.db").read_bytes())
    root = H.examples_copy(tmp)
    os.symlink(outside / "secret.cif", root / "structures" / "link.cif")
    os.symlink(outside, root / "linkdir")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1")
    yield base, root, outside
    proc.terminate()
    proc.wait(timeout=15)


def _call(base, method, route, data, **kw):
    if method == "GET":
        q = "&".join(f"{k}={urllib.parse.quote(str(v), safe='%')}" for k, v in data.items())
        return U.req(base, f"api/{route}" + (f"?{q}" if q else ""), **kw)
    return U.req(base, f"api/{route}", data=data, **kw)


def _set(data, dotted, value):
    data = json.loads(json.dumps(data))
    d = data
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
    return data


@pytest.mark.parametrize("method, route, data, params, writes", P5_ROUTES + CHANGED, ids=[r[1] for r in P5_ROUTES + CHANGED])
def test_token_host_and_header_rules(served, method, route, data, params, writes):
    base = served[0]
    assert _call(base, method, route, data, token=False)[0] == 403
    assert _call(base, method, route, data, token=False, headers={"X-Mace-Token": "wrong"})[0] == 403
    st = _call(base, method, route, data, headers={"Host": "evil.example"})[0]
    assert st in (403, 421), st
    if method == "POST":
        assert U.req(base, f"api/{route}", data=data, headers={"X-Requested-With": ""})[0] == 403
        raw = json.dumps(data).encode()
        st = U.req(base, f"api/{route}", raw=raw, headers={"X-Requested-With": "", "Content-Type": "text/plain"})[0]
        assert st == 403


ESCAPES = ["../outside/secret.cif", "{outside}/secret.cif", "structures/link.cif", "linkdir/secret.cif", "linkdir/secret.db",
           "structures/MgO.cif\x00.txt", "%2e%2e/outside/secret.cif"]


@pytest.mark.parametrize("method, route, data, params, writes", [r for r in P5_ROUTES + CHANGED if r[3]],
                         ids=[r[1] for r in P5_ROUTES + CHANGED if r[3]])
def test_paths_stay_inside_the_folder(served, method, route, data, params, writes):
    base, root, outside = served
    for param in params:
        for esc in ESCAPES:
            val = esc.replace("{outside}", str(outside))
            if method == "GET":
                d = {**data, param: val}
            else:
                d = _set(data, param, val)
            st, _, r = _call(base, method, route, d)
            assert st in (400, 403, 404), (route, param, esc, st, r)


def test_setup_save_never_writes_through_a_link(served):
    base, root, outside = served
    st, _, r = U.post(base, "api/setup/save", {"source": {"path": "linkdir/secret.cif"}, "json_text": "{}"})
    assert st in (400, 403, 404)
    assert sorted(p.name for p in outside.iterdir()) == ["secret.cif", "secret.db", "secret.out"]


def _routes():
    src = (H.REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    names = set(re.findall(r'route ==\s*"([^"]+)"', src))
    for grp in re.findall(r'route in \(([^)]*)\)', src):
        names |= set(re.findall(r'"([^"]+)"', grp))
    names |= set(re.findall(r'route\.startswith\("([^"]+)"\)', src))
    return {n for n in names if not n.startswith("test/")}


def test_every_new_route_is_in_the_table():
    listed = {r[1] for r in P5_ROUTES}
    listed |= {r[1].split("/")[0] + "/" for r in P5_ROUTES if "/" in r[1]}     # a prefix route (gallery/) by its rows
    missing = sorted(_routes() - PHASE4 - listed - TEST_ONLY)
    assert not missing, f"Phase 5 routes missing from P5_ROUTES: {missing}"


XE_CIF = """data_xe
_cell_length_a 6.2
_cell_length_b 6.2
_cell_length_c 6.2
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Xe1 Xe 0 0 0
"""


@pytest.fixture(scope="module")
def evil(served):
    """A folder in the served root with symbolic links out of it: a structure file and a folder of
    structures (xenon, which no file inside the root holds), beside one real file (Si)."""
    base, root, outside = served
    far = outside / "far"
    far.mkdir(exist_ok=True)
    (far / "xenon.cif").write_text(XE_CIF)
    d = root / "evil"
    d.mkdir(exist_ok=True)
    (d / "Si.cif").write_text((H.EXAMPLES / "structures" / "Si.cif").read_text())
    if not (d / "outside_struct.cif").is_symlink():
        os.symlink(far / "xenon.cif", d / "outside_struct.cif")
        os.symlink(far, d / "linkdir")
    return base, root, outside


def test_legend_walk_stays_inside_the_folder(evil):
    """Export Element Legend on a folder (recursive) reads only files inside the served folder: a link
    out of it (a file or a folder) is left out, and nothing about the outside files comes back."""
    pytest.importorskip("ase")      # the legend reads CIFs, which needs ase (without it: a 400 that says so)
    base, root, outside = evil
    for paths in (["evil"], ["."]):
        st, _, r = U.post(base, "api/legend", {"paths": paths, "recursive": True})
        assert st == 200, r
        assert "Xe" not in r["symbols"], (paths, r["symbols"])
        assert "xenon" not in json.dumps(r["skipped"]) and str(outside) not in json.dumps(r), r["skipped"]
    st, _, r = U.post(base, "api/legend", {"paths": ["evil"], "recursive": True})
    assert r["symbols"] == ["Si"] and r["files"] == 1


def test_gallery_leaves_out_links_out_of_the_folder(evil):
    """A folder holding a link out of the served folder still opens in the Gallery: the link is left
    out (as the Explorer leaves it out), with no error and no absolute path in the answer."""
    base, root, outside = evil
    for src in ({"type": "folder", "folder": "evil"}, {"type": "folder", "folder": "."}):
        st, _, r = U.post(base, "api/gallery/collect", {"kind": "structure", "source": src})
        assert st == 200, r
        names = [x["path"] for x in r["items"]]
        assert "evil/Si.cif" in names and not any("outside_struct" in n or n.startswith(("evil/linkdir", "linkdir"))
                                                   for n in names), names
        assert str(outside) not in json.dumps(r) and str(root) not in json.dumps(r)


def test_an_infinite_or_huge_wait_is_capped(served):
    """JSON's Infinity (or 1e300) as a task route's wait: capped, never a 500 OverflowError."""
    base = served[0]
    for w in (b"Infinity", b"1e300", b"NaN", b"-5"):
        st, _, r = U.req(base, "api/batch/scan", raw=b'{"folder": "structures", "filter": "cif", "wait": ' + w + b"}",
                         headers={"Content-Type": "application/json"})
        assert st in (200, 202), (w, st, r)


def test_an_orbital_run_whose_out_leads_outside_is_refused_with_403(evil):
    """A .molden inside the folder whose run output (.out) is a link out of it: 403 with a short
    reason, as for any other outside path (it was a 500 with the task's traceback)."""
    base, root, outside = evil
    src = H.REPO_ROOT / "tests" / "data" / "orbitals" / "si_g"
    far = outside / "orb"
    far.mkdir(exist_ok=True)
    (far / "si_g.out").write_text((src / "si_g.out").read_text())
    d = root / "evil" / "mixed"
    d.mkdir(exist_ok=True)
    (d / "si_g_00001_K000_real.molden").write_text((src / "si_g_00001_K000_real.molden").read_text())
    if not (d / "si_g.out").is_symlink():
        os.symlink(far / "si_g.out", d / "si_g.out")
    st, _, r = U.req(base, "api/orbitals?path=evil/mixed/si_g_00001_K000_real.molden&wait=60")
    assert st == 403, (st, r)
    assert "traceback" not in r and str(outside) not in json.dumps(r), r


def test_properties_and_kpath_answer_without_absolute_paths_or_500s(served):
    """1.1.6 review: the Properties tab's computed values name the run relative to the folder (never its
    absolute path); a file that is not a database is a 400; a BAND.DAT header point over 0 ("/0") is no 500;
    a job log named like an output is no run."""
    base, root, outside = served
    st, _, r = U.req(base, "api/properties/compute?path=outputs/Si_sp.out&wait=30")
    if st == 202:
        tid = r["task"]["id"]
        for _ in range(600):
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st != 409:
                break
            time.sleep(0.2)
    assert st == 200, (st, r)
    text = json.dumps(r)
    assert str(root) not in text and r["calc"]["output_file"] == "outputs/Si_sp.out", r["calc"]
    (root / "notes.db").write_text("not a database")
    st, _, r = U.req(base, "api/properties/value?db=notes.db&calc=x&name=total_energy_au")
    assert st == 400 and "not a MACE materials database" in r["error"], (st, r)
    (root / "outputs" / "Bad.BAND.DAT").write_text("# NKPT 2 NBND 1 NSPIN 1\n# 1 (0,0,0)/0\n# 2 (1,0,0)/0\n0.0 1.0\n")
    st, _, r = U.req(base, "api/kpath?path=outputs/Bad.BAND.DAT")
    assert st == 200 and r["found"] and r["iss"] is None and r["segments"] == [], (st, r)
    (root / "outputs" / "slurm-77.out").write_text("SLURM_JOB_ID=77\n")
    st, _, r = U.req(base, "api/properties/run?path=outputs/slurm-77.out")
    assert st == 200 and r["state"] == "no-run", (st, r)


def test_a_rejected_post_never_leaves_its_body_as_the_next_request(served):
    """A POST refused before its body is read (no token: 403) ends the keep-alive connection, so the
    body is never parsed as a second request on it."""
    import socket
    base = served[0]
    u = urllib.parse.urlparse(base)
    smuggled = b"GET /evil HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    req = (f"POST {u.path.rstrip('/')}/api/folder-settings HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
           f"Content-Type: application/json\r\nX-Requested-With: x\r\nContent-Length: {len(smuggled)}\r\n\r\n").encode() + smuggled
    with socket.create_connection((u.hostname, u.port), timeout=10) as s:
        s.sendall(req)
        data = b""
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pass
    assert data.count(b"HTTP/1.") == 1, data[:600]
    assert data.startswith(b"HTTP/1.0 403") or data.startswith(b"HTTP/1.1 403"), data[:200]
    assert b"connection: close" in data.lower()


def test_saving_a_held_structure_outside_the_folder_is_refused_before_it_is_built(tmp_path):
    """structure/save with a {held} source checks the path before it builds the structure, which
    needs ase: without ase an escaping path got a 503 for the missing package, not a refusal."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = H.examples_copy(tmp_path)
    os.symlink(outside, root / "linkdir")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    try:
        # building a held structure needs ase; an unknown id is refused for its path before the
        # held store is read (404 for the id), so the order shows without ase too
        st, _, r = U.post(base, "api/test/held/put", {"atoms": 8})
        held = ["no-such-id"] + ([r["held"]] if st == 200 else [])
        for hid in held:
            for esc in ("../outside/x.cif", f"{outside}/x.cif", "linkdir/x.cif", "x.cif\x00.txt", "%2e%2e/outside/x.cif"):
                st, _, r = U.post(base, "api/structure/save", {"source": {"held": hid}, "path": esc, "format": "cif"})
                assert st in (400, 403), (hid, esc, st, r)
        assert not list(outside.iterdir())
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_calc_skf_folder_outside_is_refused_on_the_network(tmp_path, monkeypatch):
    """calc/run's DFTB+ Slater-Koster folder outside the served folder: a loopback GUI accepts an
    absolute folder (the user's own set), a network-exposed one (--insecure-network) refuses it
    before anything runs."""
    import subprocess
    import sys
    import time
    pytest.importorskip("ase")
    for k in ("MACE_DFTB_SKF", "DFTB_PREFIX"):
        monkeypatch.delenv(k, raising=False)
    outside = tmp_path / "skf"
    outside.mkdir()
    exe = tmp_path / "bin" / "dftb+"                   # a stand-in dftb+: installed, so skf needs only the folder
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\necho '|  DFTB+ release 24.1'\n")
    exe.chmod(0o755)
    root = H.examples_copy(tmp_path)
    extra = dict(MACE_GUI_NO_WARMUP="1", MACE_DFTB=str(exe))
    body = {"source": {"path": "structures/Si.cif"}, "backend": "dftb", "method": "skf", "task": "sp",
            "options": {"skf_dir": str(outside)}}

    def ready(base):
        end = time.time() + 60
        while time.time() < end and U.req(base, "api/calc/backends")[2].get("checking"):
            time.sleep(0.2)

    proc, base, _ = H.start_server(root, tmp_path / "xdg", **extra)
    try:
        ready(base)
        st, _, r = U.post(base, "api/calc/run", body)
        assert st in (200, 202) and "on the network" not in json.dumps(r), (st, r)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    env = H.env(tmp_path / "xdg2", **extra)
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--host", "127.0.0.2", "--insecure-network", "--token", H.TOKEN],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        env=env, start_new_session=True, preexec_fn=H.die_with_pytest)
    try:
        m = None
        for _ in range(40):
            m = re.search(r"(http://127\.0\.0\.2:\d+/s/[^/\s]+/)", p.stdout.readline())
            if m:
                break
        assert m, "the network server printed no URL"
        nbase = m.group(1)
        ready(nbase)
        st, _, r = U.post(nbase, "api/calc/run", body)
        assert st == 400 and "inside the served folder" in r.get("error", ""), (st, r)
    finally:
        p.kill()
        p.wait(5)


def test_write_routes_never_write_through_a_link_out_of_the_folder(tmp_path):
    """The output folders of tau/strain, phonopy/generate and elastic/generate and the file tau/fit
    saves are resolved before anything is written: a symlinked folder (or file) inside --dir that
    leads out is refused, and nothing appears outside."""
    import shutil
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.json").write_text("ORIGINAL\n")
    (outside / "victim.d12").write_text("ORIGINAL\n")
    root = H.examples_copy(tmp_path)
    os.symlink(outside, root / "linkdir")
    strain = root / "strain"
    shutil.copytree(H.REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain", strain)
    os.symlink(outside / "victim.json", strain / "tau_hole_x.json")
    os.symlink(outside / "victim.d12", root / "outputs" / "Si_sp_dp_y_p1pct.d12")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1")
    try:
        gen = U.req(base, "api/info")[2].get("root_gen", 0)
        cases = []
        for od in ("../linkdir/x", "../linkdir", f"{outside}/x", "../../outside/x"):
            cases += [("tau/strain", {"parent": "outputs/Si_sp.d12", "axis": "x", "output_dir": od}),
                      ("phonopy/generate", {"parent": "outputs/Si_sp.out", "dim": "2 2 2", "output_dir": od, "wait": 0}),
                      ("elastic/generate", {"parent": "outputs/Si_sp.out", "settings": {"mode": "ELASTCON"},
                                            "output_dir": od, "wait": 0})]
        # a deck of the set that is a symlink out (with force), and the saved tau through a symlink
        cases.append(("tau/strain", {"parent": "outputs/Si_sp.d12", "axis": "y", "force": True}))
        outs = U.req(base, "api/tau/series?dir=strain")[2]["series"][0]["outputs"]
        cases.append(("tau/fit", {"outputs": outs, "carrier": "hole", "m_star": 0.3, "save": True}))
        for route, body in cases:
            st, _, r = U.post(base, f"api/{route}", {**body, "root_gen": gen})
            assert st in (400, 403), (route, body, st, r)
        assert sorted(p.name for p in outside.iterdir()) == ["victim.d12", "victim.json"]
        assert (outside / "victim.json").read_text() == "ORIGINAL\n"
        assert (outside / "victim.d12").read_text() == "ORIGINAL\n"
        # the refused set wrote nothing
        assert [p.name for p in (root / "outputs").glob("Si_sp_dp_y_*.d12")] == ["Si_sp_dp_y_p1pct.d12"]
        # inside the folder the same routes still write
        st, _, r = U.post(base, "api/tau/strain", {"parent": "outputs/Si_sp.d12", "axis": "x", "output_dir": "strained",
                                                    "root_gen": gen})
        assert st == 200 and len(r["written"]) == 5 and all(w.startswith("outputs/strained/") for w in r["written"]), r
    finally:
        proc.terminate()
        proc.wait(timeout=15)
