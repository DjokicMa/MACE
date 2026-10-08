"""`mace gui` on the example folder: the server, its token, its API, and the GUI -> JSON -> CLI loop.

The server runs as `mace_cli gui` in a subprocess on an ephemeral port and is
spoken to with urllib, so the tests go through the real entry point. It
serves a copy of examples/gui, so a saved JSON lands in tmp_path.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
EXAMPLES = REPO_ROOT / "examples" / "gui"
EXAMPLE_CONFIGS = REPO_ROOT / "Crystal_d12" / "example_configs"
TOKEN = "test-token-1234567"

# ase and plotly are optional for MACE itself (the minimal CI venv has
# neither): the server answers 503 without them, and these tests skip
_has = lambda mod: importlib.util.find_spec(mod) is not None  # noqa: E731
needs_ase = pytest.mark.skipif(not _has("ase"), reason="ase is not installed")


def _env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MACE_GUI_PRINT_TOKEN"] = "1"       # stdout is a pipe: keep the token in the printed URL
    for k in ("MACE_GUI_TOKEN", "MACE_GUI_PORT", "MACE_GUI_TEST_HOOKS"):
        env.pop(k, None)
    return env


def _die_with_pytest():
    """SIGTERM when the pytest process that started the server dies (Linux)."""
    try:
        import ctypes
        import signal as _signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(_signal.SIGTERM), 0, 0, 0)   # PR_SET_PDEATHSIG
    except Exception:
        pass


class ServerProcess(subprocess.Popen):
    """Teardown never hangs a CI runner: after terminate(), a wait() that
    times out kills the server's session (it and its workers), then waits."""

    _terminated = False

    def terminate(self):
        self._terminated = True
        super().terminate()

    def wait(self, timeout=None):
        try:
            return super().wait(timeout)
        except subprocess.TimeoutExpired:
            if not self._terminated:
                raise
            try:
                os.killpg(self.pid, 9)
            except (ProcessLookupError, PermissionError):
                self.kill()
            return super().wait(5)


def _start(root: Path, *extra, env=None):
    proc = ServerProcess(
        [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(root),
         "--port", "0", "--token", TOKEN, *extra],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env={**_env(), **(env or {})}, start_new_session=True, preexec_fn=_die_with_pytest)
    lines = []
    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        m = re.search(r"(http://127\.0\.0\.1:\d+/s/[A-Za-z0-9_-]+/)\?login=(\S+)", line)
        if m:
            return proc, m.group(1), lines
    proc.kill()
    raise AssertionError("server did not print its URL:\n" + "".join(lines))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("gui") / "demo"
    shutil.copytree(EXAMPLES, root)
    proc, base, lines = _start(root)
    yield root, base, lines
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


def _cookie(base):
    """The session cookie is named per port (browsers share cookies across ports)."""
    return "mace_gui_" + re.search(r":(\d+)/", base).group(1)


def _request(base, path, *, token=True, data=None, headers=None, method=None):
    h = dict(headers or {})
    if token:
        # the token works only in this header (and the sign-in form), never in a URL or a cookie
        h.setdefault("X-Mace-Token", TOKEN)
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h.setdefault("Content-Type", "application/json")
        h.setdefault("X-Requested-With", "mace-gui")
    req = urllib.request.Request(base + path, data=body, headers=h, method=method)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=120) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _json(base, path, **kw):
    status, _, body = _request(base, path, **kw)
    return status, json.loads(body)


# --- start-up and access control ------------------------------------------

def test_prints_a_one_time_link_and_binds_loopback(served):
    _, base, lines = served
    assert base.startswith("http://127.0.0.1:")
    assert any(re.search(r"\?login=[A-Za-z0-9_-]{20,}", line) for line in lines)
    assert not any(f"?token={TOKEN}" in line for line in lines)


def test_ssh_hint_when_run_over_ssh(tmp_path):
    env = _env()
    env["SSH_CONNECTION"] = "10.0.0.1 5555 10.0.0.2 22"
    proc = subprocess.Popen(
        [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path),
         "--port", "0", "--gateway", "gw.example.edu"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env)
    lines = []
    try:
        for _ in range(40):
            line = proc.stdout.readline()
            lines.append(line)
            if not line or line.startswith("Press Ctrl+C"):
                break
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    out = "".join(lines)
    port = re.search(r"127\.0\.0\.1:(\d+)/", out).group(1)
    assert re.search(rf"ssh -N -J \S+@gw\.example\.edu -L {port}:127\.0\.0\.1:{port} \S+@\S+", out), out


def test_drafts_note_when_port_is_not_the_default(tmp_path):
    """Drafts live in the browser per origin: another port won't see them (plan T11)."""
    proc = subprocess.Popen(
        [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port", "0"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env())
    # read the whole banner: inside an ssh session (SSH_CONNECTION) the tunnel
    # hint comes first and pushes the note below the fourth line
    lines = []
    try:
        for _ in range(40):
            line = proc.stdout.readline()
            lines.append(line)
            if not line or line.startswith("Press Ctrl+C"):
                break
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    out = "".join(lines)
    assert "Drafts saved under another port won't be offered on this one." in out, out


def test_api_needs_the_token(served):
    _, base, _ = served
    assert _request(base, "api/info", token=False)[0] == 403
    assert _request(base, "api/info", token=False,
                    headers={"Cookie": f"{_cookie(base)}=wrong"})[0] == 403
    assert _request(base, "api/files", token=False)[0] == 403
    assert _request(base, "static/app.js", token=False)[0] == 403
    assert _request(base, "api/info")[0] == 200


def test_page_without_token_is_refused(served):
    _, base, _ = served
    status, _, body = _request(base, "", token=False)
    assert status == 403
    assert b"sign-in link" in body and b"login.js" in body


def test_login_link_becomes_an_httponly_session_cookie(served):
    _, base, _ = served
    status, _, body = _request(base, "api/login-code", data={})
    code = json.loads(body)["code"]
    status, headers, _ = _request(base, f"?login={code}", token=False)
    assert status == 303
    assert headers["Location"] == "./"
    cookie = headers["Set-Cookie"]
    name, value = cookie.split(";")[0].split("=", 1)
    assert name == _cookie(base) and value and value != TOKEN
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert f"Path={re.search(r'(/s/[^/]+/)', base).group(1)}" in cookie
    # the page itself, with the session cookie
    status, headers, body = _request(base, "", token=False, headers={"Cookie": f"{name}={value}"})
    assert status == 200 and b"<title>MACE</title>" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    # the same link again (from the browser's history): no session
    status, headers, body = _request(base, f"?login={code}", token=False)
    assert status == 403 and b"already used" in body and not headers.get("Set-Cookie")
    # the token in a URL or as the cookie value never signs in
    assert _request(base, f"?token={TOKEN}", token=False)[0] == 403
    assert _request(base, "api/info", token=False, headers={"Cookie": f"{name}={TOKEN}"})[0] == 403


def test_foreign_host_header_is_refused(served):
    _, base, _ = served
    status, _, _ = _request(base, "api/info", headers={"Host": "attacker.example:80"})
    assert status == 421


def test_post_needs_the_gui_header(served):
    _, base, _ = served
    status, _, _ = _request(base, "api/deck/save", data={"name": "x.json"},
                            headers={"X-Requested-With": "other"})
    assert status == 403


@pytest.mark.parametrize("path", ["../README.md", "/etc/passwd", "outputs/../../x"])
def test_paths_stay_inside_the_served_directory(served, path):
    _, base, _ = served
    status, data = _json(base, "api/text?path=" + urllib.request.quote(path))
    assert status in (403, 404), data
    if status == 403:
        assert "outside" in data["error"]


# --- static assets: vendored, relative, no CDN ------------------------------

def test_page_loads_only_relative_local_assets(served):
    _, base, _ = served
    html = (REPO_ROOT / "mace" / "gui" / "static" / "index.html").read_text()
    for url in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not url.startswith(("http:", "https:", "//", "/")), url
    js = (REPO_ROOT / "mace" / "gui" / "static" / "app.js").read_text()
    assert "http://" not in js and "https://" not in js
    status, _, body = _request(base, "static/vendor/3dmol/3Dmol-min.js")
    assert status == 200 and len(body) > 400_000
    status, _, body = _request(base, "static/vendor/plotly.min.js")
    if _has("plotly"):
        assert status == 200 and len(body) > 1_000_000
    else:
        assert status == 404


def test_vendored_3dmol_carries_its_license():
    vendor = REPO_ROOT / "mace" / "gui" / "static" / "vendor" / "3dmol"
    assert "BSD" in (vendor / "LICENSE").read_text()
    assert "3dmol v2.5.5" in (vendor / "3Dmol-min.js.LICENSE.txt").read_text()
    bundled = (vendor / "THIRD_PARTY_LICENSES.txt").read_text()
    for lib in ("iobuffer", "netcdfjs", "pako", "upng-js"):
        assert lib in bundled


@needs_ase
def test_ionic_solids_draw_no_metal_metal_bonds():
    """Covalent radii alone join Mg-Mg in MgO (2.98 A < 1.2 x 2 x 1.41 A)."""
    from mace.gui import structures as S
    s = S.read_structure(EXAMPLES / "structures" / "MgO.cif")
    assert s["metal"] == [n == 12 for n in s["numbers"]]
    pairs = S._neighbor_pairs(np.array(s["cell"]), np.array(s["cart"]),
                               s["numbers"], s["pbc"])
    kinds = {tuple(sorted((s["symbols"][i], s["symbols"][j]))) for i, j, _ in pairs}
    assert kinds == {("Mg", "O")}


# --- structures ---------------------------------------------------------------

def test_file_listing_classifies_the_demo(served):
    _, base, _ = served
    status, data = _json(base, "api/files")
    assert status == 200
    kinds = {f["path"]: f["kind"] for f in data["files"]}
    assert kinds["structures/MgO.cif"] == "cif"
    assert kinds["outputs/Ag2Br3_opt_unfinished.out"] == "out"
    assert kinds["outputs/Si_sp.d12"] == "d12"
    assert kinds["materials.db"] == "db"


@pytest.mark.parametrize("path, formula, n, sg, dim", [
    ("structures/MgO.cif", "Mg4O4", 8, 225, "CRYSTAL"),
    ("structures/Si.cif", "Si8", 8, 227, "CRYSTAL"),
    ("structures/hBN.cif", "B2N2", 4, 194, "CRYSTAL"),
    ("outputs/Si_sp.d12", "Si8", 8, 227, "CRYSTAL"),
    ("outputs/Si_sp.out", "Si2", 2, 227, "CRYSTAL"),
    ("outputs/Ag2Br3_opt_unfinished.out", "Ag4Br6", 10, 167, "CRYSTAL"),
    ("outputs/graphene_slab_sp.out", "C2", 2, None, "SLAB"),
])
@needs_ase
def test_structure_and_symmetry(served, path, formula, n, sg, dim):
    _, base, _ = served
    status, s = _json(base, "api/structure?path=" + path)
    assert status == 200, s
    assert s["formula"] == formula
    assert len(s["numbers"]) == n == len(s["cart"]) == len(s["frac"])
    assert s["dimensionality"] == dim
    assert s["symmetry"].get("number") == sg
    # fractional and Cartesian agree
    assert np.allclose(np.asarray(s["frac"]) @ np.asarray(s["cell"]), s["cart"], atol=1e-6)


@needs_ase
def test_si_deck_expands_like_the_output_says():
    """The .d12 (asymmetric unit, CRYSTAL origin) and the .out (CRYSTAL's own
    primitive cell) of the same run give the same crystal."""
    from mace.gui import structures as S
    d12 = S.read_structure(EXAMPLES / "outputs" / "Si_sp.d12")
    out = S.read_structure(EXAMPLES / "outputs" / "Si_sp.out")
    assert d12["lattice"]["volume"] == pytest.approx(4 * out["lattice"]["volume"], rel=1e-6)
    assert d12["symmetry"]["number"] == out["symmetry"]["number"] == 227


# --- before / after ------------------------------------------------------------

@needs_ase
def test_opt_trajectory_and_comparison(served):
    _, base, _ = served
    status, t = _json(base, "api/trajectory?path=outputs/Ag2Br3_opt_unfinished.out")
    assert status == 200, t
    assert len(t["frames"]) == 6
    assert t["labels"][2:] == ["OPT point"] * 4
    # SCF energies CRYSTAL printed for the OPT points
    assert t["energies_au"][1] == pytest.approx(-1.6032187293106e4)
    c = t["comparison"]
    vol = next(r for r in c["cell"] if r["param"] == "volume")
    assert vol["percent"] == pytest.approx(1.2297, abs=1e-3)
    a = next(r for r in c["cell"] if r["param"] == "a")
    assert a["before"] == pytest.approx(7.31165435) and a["after"] == pytest.approx(7.35274787)
    assert 0 < c["max_displacement"] < 0.2
    assert {r["pair"] for r in c["bonds"]} >= {"Ag-Br"}
    status, c2 = _json(base, "api/trajectory/compare?path=outputs/Ag2Br3_opt_unfinished.out&i=0&j=0")
    assert status == 200 and c2["max_displacement"] == pytest.approx(0, abs=1e-12)


def _frame(cell, frac, numbers):
    cell = np.asarray(cell, float)
    frac = np.asarray(frac, float)
    return {"cell": cell, "frac": frac, "cart": frac @ cell, "numbers": numbers,
            "pbc": [True, True, True]}


@needs_ase
def test_cell_strain_alone_displaces_nothing():
    from mace.gui.structures import compare
    frac = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.9, 0.1, 0.25]]
    before = _frame(np.eye(3) * 4.0, frac, [12, 8, 8])
    after = _frame(np.eye(3) * 4.2, frac, [12, 8, 8])
    c = compare(before, after)
    assert c["max_displacement"] == pytest.approx(0, abs=1e-12)
    vol = next(r for r in c["cell"] if r["param"] == "volume")
    assert vol["percent"] == pytest.approx((1.05 ** 3 - 1) * 100)


@needs_ase
def test_wrap_across_the_cell_is_not_a_displacement():
    from mace.gui.structures import compare
    before = _frame(np.eye(3) * 5.0, [[0.99, 0.5, 0.5], [0.0, 0.0, 0.0]], [3, 9])
    after = _frame(np.eye(3) * 5.0, [[0.01, 0.5, 0.5], [0.0, 0.0, 0.0]], [3, 9])
    c = compare(before, after)
    assert c["displacements"][0]["length"] == pytest.approx(0.1)
    assert c["displacements"][0]["vector"][0] == pytest.approx(0.1)


@needs_ase
def test_compare_two_structure_files(served):
    _, base, _ = served
    status, r = _json(base, "api/compare?a=structures/MgO.cif&b=structures/MgO.cif")
    assert status == 200 and r["comparison"]["max_displacement"] == pytest.approx(0)
    status, r = _json(base, "api/compare?a=structures/MgO.cif&b=structures/Si.cif")
    assert status == 400 and "same atoms" in r["error"]


# --- materials.db ----------------------------------------------------------------

def test_database_browser_is_read_only(served):
    root, base, _ = served
    db = root / "materials.db"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    status, d = _json(base, "api/db?path=materials.db")
    assert status == 200, d
    ids = {m["material_id"] for m in d["materials"]}
    assert {"MgO", "Si", "Ag2Br3", "graphene_slab"} <= ids
    assert d["calc_status"]["completed"] == 2 and d["calc_status"]["failed"] == 1
    shown = {m["material_id"]: (m.get("reduced_formula"), m.get("z")) for m in d["materials"]}
    assert shown["LiF"] == ("LiF", 4) and shown["Ag2Br3"] == ("Ag2Br3", 2)
    status, m = _json(base, "api/db/material?path=materials.db&id=Si")
    assert status == 200
    assert (m["material"]["reduced_formula"], m["material"]["z"]) == ("Si", 2)
    energy = next(p for p in m["properties"] if p["property_name"] == "total_energy")
    assert energy["property_value"] == pytest.approx(-578.64691730492)
    assert m["calculations"][0]["status"] == "completed"
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert not (root / "materials.db-wal").exists()
    assert not (root / "structures.db").exists()


# --- the GUI -> JSON -> CLI loop -----------------------------------------------------

def _cli(*args, cwd):
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=str(cwd),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=300, env=_env())
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return r


@needs_ase
def test_cif2d12_preview_is_the_deck_the_cli_writes_from_the_saved_json(served, tmp_path):
    root, base, _ = served
    config = json.loads((EXAMPLE_CONFIGS / "standard_dft_opt.json").read_text())
    config["configuration"]["functional"] = "PBE0"
    status, prev = _json(base, "api/deck/preview",
                         data={"mode": "cif2d12", "structure": "structures/MgO.cif",
                               "json": config})
    assert status == 200, prev
    assert prev["deck_name"] == "MgO_CRYSTAL_OPT_symm_PBE0-D3_POB-TZVP-REV2.d12"
    assert "\nPBE0-D3\n" in prev["deck"] and "\n225\n" in prev["deck"]

    status, saved = _json(base, "api/deck/save",
                          data={"mode": "cif2d12", "name": "gui_mgo.json", "json": config,
                                "structure": "structures/MgO.cif"})
    assert status == 200, saved
    assert saved["saved"] == "gui_mgo.json"
    assert saved["command"].startswith("mace convert --batch --options_file gui_mgo.json")
    assert json.loads((root / "gui_mgo.json").read_text()) == config

    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    shutil.copy2(root / "structures" / "MgO.cif", cif_dir)
    _cli("convert", "--no-banner", "--batch", "--options_file", str(root / "gui_mgo.json"),
         "--cif_dir", str(cif_dir), "--output_dir", str(tmp_path / "decks"), cwd=tmp_path)
    deck = tmp_path / "decks" / prev["deck_name"]
    assert deck.read_text() == prev["deck"]


def test_opt2d12_preview_is_the_deck_the_cli_writes_from_the_saved_json(served, tmp_path):
    root, base, _ = served
    config = json.loads((EXAMPLE_CONFIGS / "high_accuracy_sp.json").read_text())
    status, prev = _json(base, "api/deck/preview",
                         data={"mode": "opt2d12", "structure": "outputs/Si_sp.out",
                               "json": config})
    assert status == 200, prev
    status, saved = _json(base, "api/deck/save",
                          data={"mode": "opt2d12", "name": "gui_si_sp.json", "json": config,
                                "structure": "outputs/Si_sp.out"})
    assert status == 200, saved
    assert saved["command"].startswith("mace opt2d12 --out-file outputs/Si_sp.out "
                                       "--config-file gui_si_sp.json")
    work = tmp_path / "run"
    work.mkdir()
    for name in ("Si_sp.out", "Si_sp.d12"):
        shutil.copy2(root / "outputs" / name, work)
    _cli("opt2d12", "--no-banner", "--out-file", str(work / "Si_sp.out"), "--config-file",
         str(root / "gui_si_sp.json"), "--yes", "--output-dir", str(tmp_path / "decks"),
         cwd=tmp_path)
    assert (tmp_path / "decks" / prev["deck_name"]).read_text() == prev["deck"]


@pytest.mark.parametrize("payload, message", [
    ({"mode": "cif2d12", "name": "../evil.json", "json": {"method": "DFT"}}, "plain name"),
    ({"mode": "cif2d12", "name": "bad.json", "json": {"method": "NOPE", "calculation_type": "OPT"}}, ""),
    ({"mode": "cif2d12", "name": "bad.txt", "json": {}}, "plain name"),
])
def test_save_refuses_bad_names_and_settings(served, payload, message):
    root, base, _ = served
    status, r = _json(base, "api/deck/save", data=payload)
    assert status == 400, r
    assert message in r["error"]
    assert not (root.parent / "evil.json").exists()


def test_save_does_not_overwrite_unless_asked(served):
    root, base, _ = served
    config = json.loads((EXAMPLE_CONFIGS / "quick_screen.json").read_text())
    body = {"mode": "cif2d12", "name": "keep.json", "json": config}
    assert _json(base, "api/deck/save", data=body)[0] == 200
    status, r = _json(base, "api/deck/save", data=body)
    assert status == 400 and "already exists" in r["error"]
    assert _json(base, "api/deck/save", data={**body, "overwrite": True})[0] == 200


def test_preview_reports_a_deck_the_cli_refuses(served):
    _, base, _ = served
    status, r = _json(base, "api/deck/preview",
                      data={"mode": "cif2d12", "structure": "structures/MgO.cif",
                            "json": {"method": "DFT"}})
    assert status == 400


# --- plots: made by mace plotting, never written into the served directory ----------

def test_band_plot_is_made_by_mace_plotting_outside_the_served_dir(tmp_path):
    bands = sorted(TEST_DATA.glob("BAND/1_dia_opt*_band.BAND.DAT")) if TEST_DATA.is_dir() else []
    if not bands:
        pytest.skip("test/ corpus not present (gitignored)")
    root = tmp_path / "served"
    root.mkdir()
    shutil.copy2(bands[0], root)
    proc, base, _ = _start(root)
    try:
        listing = sorted(p.name for p in root.iterdir())
        status, r = _json(base, "api/plot?path=" + bands[0].name)
        assert status == 200, r
        svg = [n for n in r["images"] if n.endswith(".svg")]
        assert svg, r["log"]
        status, headers, body = _request(base, f"api/plot-image?key={r['key']}&name={svg[0]}")
        assert status == 200 and b"<svg" in body[:2000]
        assert headers["Content-Type"] == "image/svg+xml"
        assert sorted(p.name for p in root.iterdir()) == listing   # nothing written there
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_help_names_the_command():
    r = subprocess.run([sys.executable, str(MACE_CLI), "gui", "--help"], cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
                       env=_env())
    assert r.returncode == 0
    assert "mace gui" in r.stdout and "--port" in r.stdout



def test_gui_server_imports_without_ase():
    """ase is optional for the GUI (only the editor needs it): importing the
    server must not pull it in, or `mace gui` cannot start without ase."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'ase' or name.startswith('ase.'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import mace.gui.server, mace.gui.clean, mace.gui.editor\n"
        "assert not any(m == 'ase' or m.startswith('ase.') for m in sys.modules)\n")
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=_env(),
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_missing_optional_package_is_a_503_that_names_it(tmp_path):
    """Without ase the server keeps running and says what to install,
    instead of a bare 500 (the minimal Python 3.11 CI venv has no ase)."""
    root = tmp_path / "demo"
    shutil.copytree(EXAMPLES, root)
    block = tmp_path / "block"
    block.mkdir()
    (block / "sitecustomize.py").write_text(
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'ase' or name.startswith('ase.'):\n"
        "            raise ImportError('blocked: ' + name, name=name)\n"
        "sys.meta_path.insert(0, Block())\n")
    proc, base, _ = _start(root, env={"PYTHONPATH": str(block)})
    try:
        status, _, body = _request(base, "api/structure?path=structures/MgO.cif")
        assert status == 503, body
        assert "pip install ase" in json.loads(body)["error"]
        assert _request(base, "api/files")[0] == 200
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_example_database_holds_only_numbers_from_its_files():
    """materials.db: a slab has a and b only (its c is formally 500 Å in the
    .out, no length at all), no volume; every output it records as
    completed is a real CRYSTAL run that terminated, not an excerpt."""
    import sqlite3
    conn = sqlite3.connect(f"file:{EXAMPLES / 'materials.db'}?mode=ro", uri=True)
    try:
        props = {(m, n): v for m, n, v in conn.execute(
            "select material_id, property_name, property_value from properties")}
        done = [r[0] for r in conn.execute("select output_file from calculations where status = 'completed'")]
    finally:
        conn.close()
    assert ("graphene_slab", "lattice_c") not in props and ("graphene_slab", "volume") not in props
    assert props[("graphene_slab", "lattice_a")] == pytest.approx(2.4612)
    assert props[("graphene_slab", "total_energy")] == pytest.approx(-75.927962190387)
    for m in ("MgO", "Si", "Ag2Br3"):
        assert (m, "lattice_c") in props and (m, "volume") in props
    assert done
    for out in done:
        text = (EXAMPLES / out).read_text()
        assert "EEEEEEEEEE TERMINATION" in text and "synthetic" not in text.lower(), out
