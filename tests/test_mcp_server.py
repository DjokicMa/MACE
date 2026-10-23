"""`mace mcp`: the MCP server over real stdin/stdout.

Every protocol check starts the real ``mace_cli mcp`` in a subprocess and
talks JSON-RPC to it, so the stdout discipline (nothing but protocol lines),
the early dispatch before any banner and the imports are tested as a client
sees them. The tools that read CRYSTAL outputs use the real corpus under
test/ and skip without it; the structure tools use a small CIF written here.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
# the transport the protocol tests use (builtin; "sdk" runs them through the optional mcp package)
TRANSPORT = os.environ.get("MACE_MCP_TEST_TRANSPORT", "builtin")

NACL_CIF = textwrap.dedent("""\
    data_NaCl
    _symmetry_space_group_name_H-M   'F m -3 m'
    _symmetry_Int_Tables_number      225
    _cell_length_a   5.6402
    _cell_length_b   5.6402
    _cell_length_c   5.6402
    _cell_angle_alpha   90
    _cell_angle_beta    90
    _cell_angle_gamma   90
    loop_
    _symmetry_equiv_pos_as_xyz
    x,y,z
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Na1 Na 0.0 0.0 0.0
    Na2 Na 0.0 0.5 0.5
    Na3 Na 0.5 0.0 0.5
    Na4 Na 0.5 0.5 0.0
    Cl1 Cl 0.5 0.5 0.5
    Cl2 Cl 0.5 0.0 0.0
    Cl3 Cl 0.0 0.5 0.0
    Cl4 Cl 0.0 0.0 0.5
    """)

PLUGIN_MODULE = textwrap.dedent('''\
    """Test plugins for mace mcp (MACE_MCP_PLUGINS)."""
    import time
    from pathlib import Path
    from typing import Literal, Optional


    def slow_count(ctx, seconds: float = 30.0, marker: Optional[Path] = None):
        """Count slowly; stops when the client cancels.

        Args:
            seconds: how long to run
            marker: a file written when the cancel arrives
        """
        t0 = time.time()
        while time.time() - t0 < seconds:
            if ctx.canceled:
                if marker is not None:
                    marker.write_text("canceled")
                return {"canceled": True}
            ctx.progress((time.time() - t0) / seconds, "counting")
            time.sleep(0.05)
        return {"canceled": False}


    def shout(word: str, times: int = 2, style: Literal["upper", "lower"] = "upper"):
        """Repeat a word."""
        w = word.upper() if style == "upper" else word.lower()
        return {"text": " ".join([w] * times)}


    def wipe(ctx, target: Path):
        """Remove a file (needs confirm)."""
        target.unlink()
        return {"removed": ctx.show(target)}


    wipe.__mcp__ = {"writes": True, "confirm": "always"}


    def stretch(ctx=None, cell=None, frac=None, numbers=None, pbc=(True, True, True), factor: float = 1.1, **_):
        """Scale the cell (a builder: arrays in, arrays out)."""
        return {"cell": [[x * factor for x in row] for row in cell], "frac": frac, "numbers": numbers, "pbc": pbc,
                "summary": f"x{factor}"}


    stretch.__mcp__ = {"kind": "builder"}
    ''')


def _env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    env.update(MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra or {})
    return env


class Client:
    """A minimal MCP client: the server's stdout lines go through a reader thread."""

    def __init__(self, root, *args, env=None):
        self.proc = subprocess.Popen([sys.executable, str(MACE_CLI), "mcp", "--root", str(root), *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=_env(env), cwd=str(root))
        self.lines = queue.Queue()
        self.raw = []
        self.err = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self.next_id = 0

    def _read(self):
        for line in self.proc.stdout:
            self.raw.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def _read_err(self):
        for line in self.proc.stderr:
            self.err.append(line.decode("utf-8", "replace"))

    def send(self, obj):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()

    def recv(self, timeout=120.0):
        line = self.lines.get(timeout=timeout)
        assert line is not None, "server closed stdout:\n" + "".join(self.err)
        return json.loads(line)

    def request(self, method, params=None, timeout=180.0):
        self.next_id += 1
        rid = self.next_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        while True:
            got = self.recv(timeout)
            if got.get("id") == rid:
                return got

    def initialize(self, version="2025-06-18"):
        r = self.request("initialize", {"protocolVersion": version, "capabilities": {},
                                        "clientInfo": {"name": "pytest", "version": "1"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r

    def call(self, name, arguments=None, timeout=300.0):
        r = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        assert "result" in r, r
        return r["result"]

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for line in self.raw:          # stdout carried protocol lines and nothing else
            json.loads(line)
        return self.proc.returncode


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def client(root):
    c = Client(root, "--transport", TRANSPORT)
    c.initialize()
    yield c
    assert c.close() == 0


@pytest.fixture
def builtin(root):
    """The built-in transport: the JSON-RPC details below are its own (the SDK has its own rules)."""
    c = Client(root, "--transport", "builtin")
    c.initialize()
    yield c
    assert c.close() == 0


def _needs_ase():
    pytest.importorskip("ase")
    pytest.importorskip("spglib")


def _corpus(*parts):
    p = TEST_DATA.joinpath(*parts)
    if not p.exists():
        pytest.skip(f"test/ corpus file {'/'.join(parts)} not present")
    return p


# ---------------------------------------------------------------- protocol
def test_handshake_lists_tools_and_negotiates_version(root):
    c = Client(root, "--transport", TRANSPORT)
    r = c.initialize("2025-03-26")
    res = r["result"]
    assert res["protocolVersion"] == "2025-03-26"
    assert res["serverInfo"]["name"] == "mace"
    assert set(res["capabilities"]) >= {"tools", "resources"}
    assert str(root.resolve()) in res["instructions"]
    assert c.request("ping")["result"] == {}
    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    for name in ("structure_inspect", "structure_supercell", "structure_clean", "setup_d12_preview",
                 "setup_d12_write", "setup_d3_preview", "deck_preflight", "job_submit", "job_status",
                 "job_cancel", "outputs_scan", "output_summary", "output_properties", "db_search", "files_list"):
        assert name in tools, name
    for t in tools.values():
        s = t["inputSchema"]
        assert s["type"] == "object" and s["additionalProperties"] is False, t["name"]
        assert set(t["annotations"]) >= {"readOnlyHint", "destructiveHint"}
    assert tools["job_submit"]["annotations"]["destructiveHint"] is True
    assert "confirm" in tools["job_submit"]["inputSchema"]["properties"]
    assert tools["structure_inspect"]["annotations"]["readOnlyHint"] is True
    # an unknown version gets the newest one the server speaks
    c2 = Client(root, "--transport", "builtin")
    assert c2.initialize("1999-01-01")["result"]["protocolVersion"] == "2025-11-25"
    assert c2.close() == 0
    assert c.close() == 0


def test_json_rpc_errors(builtin):
    client = builtin
    client.send(b"{not json")
    assert client.recv()["error"]["code"] == -32700
    assert client.request("no/such/method")["error"]["code"] == -32601
    assert client.request("tools/call", {"name": "no_such_tool", "arguments": {}})["error"]["code"] == -32602
    bad = client.call("files_list", {"directory": ".", "nonsense": 1})
    assert bad["isError"] is True and "unknown argument" in bad["content"][0]["text"]
    bad = client.call("files_list", {"limit": "many"})
    assert bad["isError"] is True and "limit" in bad["content"][0]["text"]
    # a batch is answered as one array
    client.send([{"jsonrpc": "2.0", "id": "a", "method": "ping"}, {"jsonrpc": "2.0", "id": "b", "method": "ping"}])
    got = client.recv()
    assert sorted(x["id"] for x in got) == ["a", "b"]


def test_resources_templates(builtin):
    client = builtin
    res = client.request("resources/list")["result"]["resources"]
    uris = [r["uri"] for r in res]
    assert any(u.startswith("mace://templates/d12/") for u in uris)
    assert any(u.startswith("mace://templates/d3/") for u in uris)
    d3 = next(u for u in uris if u.startswith("mace://templates/d3/band"))
    text = client.request("resources/read", {"uri": d3})["result"]["contents"][0]["text"]
    assert json.loads(text)["calculation_type"] == "BAND"
    assert client.request("resources/read", {"uri": "mace://nothing"})["error"]["code"] == -32002


def test_paths_stay_inside_the_root(client, root, tmp_path):
    (tmp_path / "outside.cif").write_text(NACL_CIF)
    r = client.call("structure_inspect", {"path": "../outside.cif"})
    assert r["isError"] and "outside the served folder" in r["content"][0]["text"]
    r = client.call("structure_inspect", {"path": str(tmp_path / "outside.cif")})
    assert r["isError"] and "outside the served folder" in r["content"][0]["text"]
    (root / "link.cif").symlink_to(tmp_path / "outside.cif")
    r = client.call("structure_inspect", {"path": "link.cif"})
    assert r["isError"] and "outside the served folder" in r["content"][0]["text"]


def test_read_only_hides_writers(root):
    c = Client(root, "--read-only", "--transport", "builtin")
    c.initialize()
    names = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
    assert "structure_inspect" in names and "setup_d12_preview" in names
    assert not names & {"job_submit", "job_cancel", "structure_supercell", "setup_d12_write", "structure_clean"}
    r = c.request("tools/call", {"name": "job_submit", "arguments": {"paths": ["x.d12"]}})
    assert r["error"]["code"] == -32602 and "read-only" in r["error"]["message"]
    assert c.close() == 0


# ---------------------------------------------------------------- confirmations
def test_submit_and_cancel_need_confirm(client, root):
    deck = root / "mat.d12"
    deck.write_text("mat\nCRYSTAL\n0 0 0\n225\n5.64\n2\n11 0 0 0\n17 .5 .5 .5\nEND\n")
    r = client.call("job_submit", {"paths": ["mat.d12"], "track": True})
    assert r["isError"] is True
    sc = r["structuredContent"]
    assert sc["confirmation_required"] is True
    assert sc["preview"]["decks"] == ["mat.d12"]
    assert sc["preview"]["commands"] == ["mace submit mat.d12 --track"]
    assert "confirm" in r["content"][0]["text"]
    assert sorted(p.name for p in root.iterdir()) == ["mat.d12"]          # nothing written, nothing run
    r = client.call("job_cancel", {"job_ids": ["123456"]})
    assert r["isError"] and r["structuredContent"]["confirmation_required"] is True
    # confirm must be the JSON true
    r = client.call("job_cancel", {"job_ids": ["123456"], "confirm": "yes"})
    assert r["isError"] and "confirm" in r["content"][0]["text"]


def test_supercell_writes_new_file_and_replacing_needs_confirm(client, root):
    _needs_ase()
    (root / "NaCl.cif").write_text(NACL_CIF)
    info = client.call("structure_inspect", {"path": "NaCl.cif"})
    assert not info["isError"], info
    s = info["structuredContent"]
    assert s["reduced_formula"] == "NaCl" and s["n_atoms"] == 8
    assert s["symmetry"]["number"] == 225 and s["dimensionality"] == "CRYSTAL"
    assert len(s["sites"]) == 8
    r = client.call("structure_supercell", {"path": "NaCl.cif", "matrix": [2, 2, 2]})
    assert not r["isError"], r
    out = r["structuredContent"]
    assert out["written"] == {"path": "NaCl_2x2x2.cif", "bytes": out["written"]["bytes"], "replaced": False}
    assert out["n_atoms"] == 64 and out["space_group"]["number"] == 225
    again = client.call("structure_supercell", {"path": "NaCl.cif", "matrix": [2, 2, 2]})
    assert again["isError"] and again["structuredContent"]["confirmation_required"] is True
    ok = client.call("structure_supercell", {"path": "NaCl.cif", "matrix": [2, 2, 2], "confirm": True})
    assert ok["structuredContent"]["written"]["replaced"] is True
    back = client.call("structure_inspect", {"path": "NaCl_2x2x2.cif", "sites": False})
    assert back["structuredContent"]["n_atoms"] == 64 and "sites" not in back["structuredContent"]
    prim = client.call("structure_standardize", {"path": "NaCl.cif", "kind": "primitive", "output": "prim.xyz"})
    assert prim["structuredContent"]["n_atoms"] == 2
    listing = client.call("files_list", {"pattern": "*.cif"})["structuredContent"]
    assert [f["path"] for f in listing["files"]] == ["NaCl.cif", "NaCl_2x2x2.cif"]
    assert {f["kind"] for f in listing["files"]} == {"cif"}


# ---------------------------------------------------------------- plugins, progress, cancel
def _plugin_client(root, tmp_path, entries):
    mod_dir = tmp_path / "plugmods"
    mod_dir.mkdir(exist_ok=True)
    (mod_dir / "mcp_test_plugins.py").write_text(PLUGIN_MODULE)
    pp = os.pathsep.join(x for x in (str(mod_dir), os.environ.get("PYTHONPATH", "")) if x)
    c = Client(root, "--transport", TRANSPORT, env={"MACE_MCP_PLUGINS": entries, "PYTHONPATH": pp})
    c.initialize()
    return c


def test_plugin_schema_from_signature(root, tmp_path):
    c = _plugin_client(root, tmp_path, "mcp_test_plugins:shout,wipe=mcp_test_plugins:wipe,"
                                       "gone=no_such_module_xyz:nothing")
    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    s = tools["shout"]["inputSchema"]
    assert s["required"] == ["word"]
    assert s["properties"]["times"] == {"type": "integer", "default": 2}
    assert s["properties"]["style"]["enum"] == ["upper", "lower"]
    assert tools["shout"]["description"] == "Repeat a word."
    assert "gone" not in tools
    assert any("gone" in line and "not loaded" in line for line in c.err)
    assert c.call("shout", {"word": "mace", "times": 3})["structuredContent"] == {"text": "MACE MACE MACE"}
    assert c.call("shout", {"word": "x", "style": "loud"})["isError"]
    # __mcp__ flags: a writer that needs confirm, with its Path resolved inside the root
    assert tools["wipe"]["annotations"]["destructiveHint"] is True
    (root / "victim.txt").write_text("x")
    r = c.call("wipe", {"target": "victim.txt"})
    assert r["isError"] and r["structuredContent"]["confirmation_required"] is True
    assert (root / "victim.txt").exists()
    assert c.call("wipe", {"target": "../../etc/passwd", "confirm": True})["isError"]
    assert not c.call("wipe", {"target": "victim.txt", "confirm": True})["isError"]
    assert not (root / "victim.txt").exists()
    assert c.close() == 0


def test_builder_plugin_reads_and_writes_structures(root, tmp_path):
    _needs_ase()
    (root / "NaCl.cif").write_text(NACL_CIF)
    c = _plugin_client(root, tmp_path, "mcp_test_plugins:stretch")
    tool = next(t for t in c.request("tools/list")["result"]["tools"] if t["name"] == "stretch")
    assert set(tool["inputSchema"]["properties"]) == {"path", "frame", "output", "factor", "confirm"}
    assert tool["inputSchema"]["required"] == ["path"] and tool["annotations"]["readOnlyHint"] is False
    r = c.call("stretch", {"path": "NaCl.cif", "factor": 2.0})
    assert not r["isError"], r
    out = r["structuredContent"]
    assert out["written"]["path"] == "NaCl_stretch.cif" and abs(out["lattice"]["a"] - 11.2804) < 1e-6
    assert c.call("stretch", {"path": "NaCl.cif"})["structuredContent"]["confirmation_required"] is True
    assert c.close() == 0


def test_progress_and_cancel(root, tmp_path):
    c = _plugin_client(root, tmp_path, "mcp_test_plugins:slow_count")
    marker = root / "canceled.txt"
    c.send({"jsonrpc": "2.0", "id": 77, "method": "tools/call",
            "params": {"name": "slow_count", "arguments": {"seconds": 60, "marker": "canceled.txt"},
                       "_meta": {"progressToken": "tok"}}})
    prog = c.recv(timeout=30)
    assert prog["method"] == "notifications/progress" and prog["params"]["progressToken"] == "tok"
    assert 0 <= prog["params"]["progress"] <= 1
    # the server answers while the call runs, then drops the canceled call's response
    assert c.request("ping")["result"] == {}
    c.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 77, "reason": "test"}})   # british-ok: the MCP method name
    deadline = time.time() + 20
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == "canceled"
    assert c.request("ping")["result"] == {}
    seen = []
    while True:
        try:
            m = c.lines.get(timeout=1.0)
        except queue.Empty:
            break
        if m is None:
            break
        seen.append(json.loads(m))
    assert not [m for m in seen if m.get("id") == 77]
    assert c.close() == 0


# ---------------------------------------------------------------- real CRYSTAL outputs (test/ corpus)
def _copy(src: Path, dst_dir: Path, *suffixes):
    for suf in suffixes:
        f = src.with_suffix(suf)
        if f.exists():
            shutil.copy2(f, dst_dir / f.name)
    return dst_dir / src.name


def test_output_tools_on_a_real_opt(client, root):
    out = _copy(_corpus("OPT", "1_dia_opt_BULK_OPTGEOM.out"), root, ".out", ".d12")
    s = client.call("output_summary", {"path": out.name, "energies": True})
    assert not s["isError"], s
    s = s["structuredContent"]
    assert s["status"] != "ongoing" and s["is_opt"] is True and s["complete"] is True
    assert s["optimization"]["steps"] == 3 and len(s["optimization"]["energies_au"]) == 3
    assert s["properties"]["calculation_type"] == "OPT"
    assert abs(s["properties"]["band_gap"] - 6.001) < 1e-6
    p = client.call("output_properties", {"path": out.name, "names": ["space_group_number", "atomic_positions",
                                                                     "no_such_property"]})["structuredContent"]
    assert p["properties"]["space_group_number"] == 227 and len(p["properties"]["atomic_positions"]) == 2
    assert p["missing"] == ["no_such_property"]
    scan = client.call("outputs_scan", {})["structuredContent"]
    assert scan["n_outputs"] == 1
    frame = client.call("structure_inspect", {"path": out.name})["structuredContent"]
    assert frame["symmetry"]["number"] == 227 and frame["n_frames"] >= 2


def test_deck_previews_and_write_on_a_real_run(client, root):
    _needs_ase()
    src = _corpus("OPT", "1_dia_opt_rev1.out")
    out = _copy(src, root, ".out", ".d12", ".f9")
    d12 = client.call("setup_d12_preview", {"structure": out.name, "template": "high_accuracy_sp"})
    assert not d12["isError"], d12["content"][0]["text"]
    deck = d12["structuredContent"]["decks"][0]
    assert deck["name"].endswith(".d12") and "END" in deck["text"]
    assert sorted(p.name for p in root.iterdir()) == ["1_dia_opt_rev1.d12", "1_dia_opt_rev1.f9", "1_dia_opt_rev1.out"]
    d3 = client.call("setup_d3_preview", {"output_file": out.name, "template": "band_high_symmetry"})
    assert not d3["isError"], d3["content"][0]["text"]
    d3 = d3["structuredContent"]
    assert d3["calc_type"] == "BAND" and d3["decks"][0]["name"].endswith(".d3") and "BAND" in d3["decks"][0]["text"]
    w = client.call("setup_d12_write", {"structure": out.name, "template": "high_accuracy_sp", "output_dir": "sp"})
    assert not w["isError"], w["content"][0]["text"]
    w = w["structuredContent"]
    assert w["decks"][0]["text"] == deck["text"]                # the preview is what gets written
    again = client.call("setup_d12_write", {"structure": out.name, "template": "high_accuracy_sp",
                                            "output_dir": "sp"})
    assert again["isError"] and again["structuredContent"]["confirmation_required"] is True


def test_db_tools_on_a_real_database(client, root):
    db = _corpus("SP", "materials.db")
    shutil.copy2(db, root / "materials.db")
    s = client.call("db_summary", {})
    assert not s["isError"], s
    s = s["structuredContent"]
    assert s["n_materials"] >= 1
    page = client.call("db_search", {"limit": 5})["structuredContent"]
    assert page["rows"]
    mid = page["rows"][0]["material_id"]
    m = client.call("db_material", {"material_id": mid})["structuredContent"]
    assert m["material"]["material_id"] == mid


# ---------------------------------------------------------------- the official SDK transport
def test_sdk_transport_when_installed(root):
    try:
        import mcp.server.lowlevel  # noqa: F401
    except Exception:
        pytest.skip("the optional mcp package is not installed")
    c = Client(root, "--transport", "sdk")
    r = c.initialize("2025-06-18")
    assert r["result"]["serverInfo"]["name"] == "mace", r
    names = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
    assert "structure_inspect" in names
    res = c.call("files_list", {})
    assert res["isError"] is False
    r = c.call("job_cancel", {"job_ids": ["1"]})
    assert r["isError"] is True
    assert c.close() == 0


# ---------------------------------------------------------------- 1.1.6 plugins (plugins.py)
WATER_XYZ = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"


def test_shipped_plugins_load(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    for name in ("structure_vacuum", "analysis_porosity", "analysis_pxrd", "analysis_topology",
                 "calc_backends", "calc_run"):
        assert name in tools, name
    assert tools["calc_run"]["inputSchema"]["properties"]["task"]["enum"] == ["sp", "opt", "freq", "md", "charges"]
    assert tools["analysis_pxrd"]["annotations"]["readOnlyHint"] is True


def test_structure_vacuum_repeats_a_molecule(client, root):
    _needs_ase()
    (root / "water.xyz").write_text(WATER_XYZ)
    r = client.call("structure_vacuum", {"path": "water.xyz", "n1": 2, "vacuum": "crystal"})
    assert not r["isError"], r
    out = r["structuredContent"]
    assert out["n_atoms"] == 6 and out["pbc"] == [False, False, False] and out["vacuum_mode"] == "crystal"
    assert out["written"]["path"] == "water_2x1x1.xyz"
    assert abs(sum(x * x for x in out["vacuum_vectors"][0]) ** 0.5 - 500.0) < 1e-6
    assert client.call("structure_vacuum", {"path": "water.xyz", "n1": 2, "vacuum": "open"})["isError"]



def test_structure_vacuum_xyz_output_keeps_its_periodicity(client, root):
    """structure_vacuum writes extended XYZ as .xyz; reading it back keeps its cell and pbc (a slab
    stays a slab, not a molecule), and a plain XYZ is still a molecule."""
    _needs_ase()
    (root / "g.extxyz").write_text('2\nLattice="2.46 0.0 0.0 -1.23 2.1304225867 0.0 0.0 0.0 16.0" '
                                   'Properties=species:S:1:pos:R:3 pbc="T T F"\n'
                                   "C 0.0 0.0 8.0\nC 1.23 0.7101408622 8.0\n")
    r = client.call("structure_vacuum", {"path": "g.extxyz", "n1": 2, "n2": 2})
    assert not r["isError"], r
    assert r["structuredContent"]["written"]["path"] == "g_2x2x1.xyz"
    back = client.call("structure_inspect", {"path": "g_2x2x1.xyz", "sites": False})["structuredContent"]
    assert back["pbc"] == [True, True, False] and back["n_atoms"] == 8, back
    (root / "water.xyz").write_text(WATER_XYZ)
    w = client.call("structure_inspect", {"path": "water.xyz", "sites": False})["structuredContent"]
    assert w["pbc"] == [False, False, False], w

def test_analysis_and_calc_plugins(client, root):
    _needs_ase()
    (root / "NaCl.cif").write_text(NACL_CIF)
    r = client.call("analysis_pxrd", {"path": "NaCl.cif", "two_theta_max": 60.0})
    assert not r["isError"], r
    assert r["structuredContent"]["two_theta"]
    b = client.call("calc_backends", {})
    assert not b["isError"], b
    assert b["structuredContent"]["exit_code"] == 0
    (root / "water.xyz").write_text(WATER_XYZ)
    c = client.call("calc_run", {"path": "water.xyz", "task": "sp", "backend": "uff", "method": "uff"})
    assert not c["isError"], c
    assert c["structuredContent"]["exit_code"] == 0, c
    assert (root / "water_uff-uff_sp.json").is_file()


def test_calc_run_backend_failure_is_a_message(root, tmp_path):
    """A backend that fails at run time (xtb's SCF) gives calc_run mace calc's message, exit code 1 and the
    backend's own words, not a Python traceback."""
    _needs_ase()
    xtb = tmp_path / "bin" / "xtb"
    xtb.parent.mkdir()
    xtb.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n"
                   "    print('      * xtb version 6.7.1 (fake) compiled by x on 2024')\n    sys.exit(0)\n"
                   "print('  -2- scc: Self consistent charge iterator did not converge')\nsys.exit(1)\n")
    xtb.chmod(0o755)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    c = Client(root, "--transport", TRANSPORT, env={"MACE_XTB": str(xtb), "TMPDIR": str(tmp)})
    c.initialize()
    try:
        (root / "water.xyz").write_text(WATER_XYZ)
        r = c.call("calc_run", {"path": "water.xyz", "task": "sp", "backend": "xtb", "method": "gfn2"})
        out = r["structuredContent"]
        assert out["exit_code"] == 1, r
        assert "Traceback" not in out["error"], out["error"]
        assert "GFN2-xTB failed during the single point" in out["error"]
        assert "Self consistent charge iterator did not converge" in out["error"]
        assert sorted(p.name for p in root.iterdir()) == ["water.xyz"]
    finally:
        assert c.close() == 0
        shutil.rmtree(tmp)


def test_calc_run_passes_on_only_allowed_options(root, tmp_path):
    """calc_run's options are an allow-list of full names: no abbreviation (--forc, --output=), no path
    option leading out of the served folder, no model file or license statement from the client."""
    _needs_ase()
    hist = tmp_path / "history"
    c = Client(root, "--transport", TRANSPORT, env={"MACE_GUI_HISTORY_DIR": str(hist)})
    c.initialize()
    try:
        out = tmp_path / "outside"
        out.mkdir()
        (out / "evil.model").write_text("x")
        (root / "water.xyz").write_text(WATER_XYZ)
        base = {"path": "water.xyz", "task": "sp", "backend": "uff", "method": "uff"}
        for opts, why in ((["--output=" + str(out)], "not an option this tool passes on"),
                          (["--output-dir=" + str(out)], "not an option this tool passes on"),
                          (["--forc"], "not an option this tool passes on"),
                          (["--force"], "not an option this tool passes on"),
                          (["--model-file=" + str(out / "evil.model"), "--model-backend=mace-torch",
                            "--model-license-ok"], "--model-file is not passed on"),
                          (["--model-license-ok"], "--model-license-ok is not passed on"),
                          (["--skf-dir=" + str(out)], "--skf-dir is not passed on"),
                          (["--option=skf_dir=" + str(out)], "the keys this tool passes on"),
                          (["--option=_lease=1"], "the keys this tool passes on"),
                          (["--fmax", "0.02"], "not an option this tool passes on"),
                          (["--relax-cell=1"], "not an option this tool passes on"),
                          (["-o", str(out)], "not an option")):
            r = c.call("calc_run", dict(base, options=opts))
            assert r["isError"], (opts, r)
            assert why in r["content"][0]["text"], (opts, r)
        assert sorted(p.name for p in root.iterdir()) == ["water.xyz"]
        assert sorted(p.name for p in out.iterdir()) == ["evil.model"]
        r = c.call("calc_run", dict(base, output_dir=str(out)))
        assert r["isError"] and "outside the served folder" in r["content"][0]["text"], r
        ok = c.call("calc_run", dict(base, options=["--cutoff=10", "--option=cutoff=10"]))
        assert not ok["isError"] and ok["structuredContent"]["exit_code"] == 0, ok
        report = root / "water_uff-uff_sp.json"
        first = report.read_bytes()
        # replacing the report needs confirm, and keeps the old one in File History
        again = c.call("calc_run", base)
        assert again["isError"] and again["structuredContent"]["confirmation_required"] is True, again
        assert "water_uff-uff_sp.json" in again["structuredContent"]["preview"]["existing"]
        assert not hist.exists() or not any(p.is_file() and p.read_bytes() == first for p in hist.rglob("*"))
        done = c.call("calc_run", dict(base, confirm=True))
        assert not done["isError"] and done["structuredContent"]["exit_code"] == 0, done
        assert any(p.is_file() and p.read_bytes() == first for p in hist.rglob("*")), list(hist.rglob("*"))
    finally:
        assert c.close() == 0
