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
    # connectivity (PLAN_connect): Open Folder and the managed-mode control routes
    ("GET", "folders", {}, [], False),
    ("POST", "open-folder", {"path": "/nonexistent-folder-zq"}, [], False),
    ("POST", "managed/", {"watcher": "w"}, [], False),
    ("POST", "shutdown", {"mode": "cancel"}, [], False),
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
    ("GET", "transport", {"path": "outputs/Si_sp.out"}, ["path"], False),
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
    missing = sorted(_routes() - PHASE4 - listed - TEST_ONLY)
    assert not missing, f"Phase 5 routes missing from P5_ROUTES: {missing}"
