"""The shell's server side and static rules, through the real `mace gui`:
settings, the event stream, versioned caching, timing headers, the security
checks on the new routes, and the design-system rules (tokens, spelling)."""
import json
import re
import socket
import time
import urllib.error
import urllib.request

import pytest

import guiharness as H

STATIC = H.REPO_ROOT / "mace" / "gui" / "static"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guiapi")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    yield base, root, xdg
    proc.terminate()
    proc.wait(timeout=15)


def req(base, path, *, data=None, headers=None, token=True, method=None):
    h = {"X-Requested-With": "mace-gui"}
    if token:
        h["X-Mace-Token"] = H.TOKEN
    h.update(headers or {})
    body = json.dumps(data).encode() if data is not None else None
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_settings_round_trip_writes_only_the_xdg_file(served):
    base, root, xdg = served
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    status, _, body = req(base, "api/settings")
    assert status == 200 and json.loads(body)["settings"] == {}
    assert json.loads(body)["path"] == str(xdg / "mace" / "gui.json")
    status, _, _ = req(base, "api/settings", data={"settings": {"theme": "dark", "uiScale": 110}})
    assert status == 200
    stored = json.loads((xdg / "mace" / "gui.json").read_text())
    # the profile file, version 2 (polish P1): settings keep their meaning, the other sections start empty
    assert stored["version"] == 2 and stored["settings"] == {"theme": "dark", "uiScale": 110}
    assert stored["ui"] == {} and stored["memory"] == {} and stored["recent"] == {"files": {}, "folders": []}
    assert json.loads(req(base, "api/settings")[2])["settings"] == {"theme": "dark", "uiScale": 110}
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before
    assert req(base, "api/settings", data={"settings": ["no"]})[0] == 400
    status, _, body = req(base, "api/settings", data={"settings": {"x" * 200: 1}})
    assert status == 200 and list(json.loads(body)["rejected"]) == ["x" * 80]     # named, not stored
    req(base, "api/settings", data={"settings": {}})


@pytest.mark.parametrize("path, method", [("api/settings", "GET"), ("api/settings", "POST"),
                                          ("api/events", "GET"), ("api/boot", "GET")])
def test_new_routes_keep_the_token_host_and_header_rules(served, path, method):
    base = served[0]
    data = {"settings": {}} if method == "POST" else None
    assert req(base, path, data=data, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert req(base, path, data=data, headers={"Host": f"evil.example:{port}"})[0] == 421
    if method == "POST":
        assert req(base, path, data=data, headers={"X-Requested-With": "x"})[0] == 403


def test_event_stream_says_hello_then_beats(served):
    base = served[0]
    from urllib.parse import urlsplit
    u = urlsplit(base)
    host, port, prefix = u.hostname, u.port, u.path
    s = socket.create_connection((host, int(port)), timeout=10)
    s.sendall(f"GET {prefix}api/events?interval=0.3 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
              f"X-Mace-Token: {H.TOKEN}\r\n\r\n".encode())
    data, t0 = b"", time.time()
    while data.count(b"event: heartbeat") < 2 and time.time() - t0 < 5:
        data += s.recv(4096)
    s.close()
    head, _, stream = data.partition(b"\r\n\r\n")
    assert b"text/event-stream" in head
    assert stream.startswith(b"retry: 2000\nevent: hello\n")
    assert stream.count(b"event: heartbeat") >= 2
    assert time.time() - t0 < 3


def test_versioned_static_is_immutable_and_index_names_the_version(served):
    base = served[0]
    status, headers, body = req(base, "", headers={"Cookie": ""})
    # the page carries the profile's first-paint values (polish P3): never stored, never a 304
    assert status == 200 and headers["Cache-Control"] == "no-store" and "ETag" not in headers
    assert req(base, "", headers={"If-None-Match": '"anything"'})[0] == 200
    version = re.search(rb'src="static/(v-[0-9a-f]{10})/js/shell\.js"', body).group(1).decode()
    assert b"@V@" not in body
    for name in ("js/shell.js", "css/tokens.css", "icons/lucide.svg", "vendor/3dmol/3Dmol-min.js"):
        status, headers, _ = req(base, f"static/{version}/{name}", headers={"Accept-Encoding": "gzip"})
        assert status == 200, name
        assert headers["Cache-Control"] == "public, max-age=31536000, immutable", name
        again = req(base, f"static/{version}/{name}", headers={"Accept-Encoding": "gzip", "If-None-Match": headers["ETag"]})
        assert again[0] == 304 and again[2] == b"", name
    # a stale version is not served from the immutable path
    assert req(base, "static/v-0000000000/js/shell.js")[0] == 404


def test_json_has_an_etag_and_server_timing(served):
    base = served[0]
    status, headers, _ = req(base, "api/files")
    assert status == 200 and headers["ETag"] and re.match(r"app;dur=\d", headers["Server-Timing"])
    status, _, body = req(base, "api/files", headers={"If-None-Match": headers["ETag"]})
    assert status == 304 and body == b""


def test_info_reports_static_version_heartbeat_and_tunnel(served):
    info = json.loads(req(served[0], "api/info")[2])
    assert info["static"].startswith("v-") and info["heartbeat"] == 10
    assert "tunnel" in info


# --- design-system rules -------------------------------------------------------------------

def _own_text_files():
    return [p for p in STATIC.rglob("*") if p.is_file() and "vendor" not in p.parts
            and p.suffix in (".js", ".css", ".html", ".svg")]


def test_colors_only_in_tokens():
    """No literal hex colors outside tokens.css (and the element color
    tables, which are data). HTML entities like &#772; do not count."""
    hexcolor = re.compile(r"(?<![&\w])#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b(?![\w-])")
    hits = []
    for p in _own_text_files():
        # the logo files are images: the browser draws them without the page's CSS
        if p.name == "tokens.css" or p.parent.name in ("icons", "brand"):
            continue
        text = re.sub(r"&#\d+;", "", p.read_text())
        text = re.sub(r'href="[^"]*"|#[a-z][\w-]*', "", text)   # ids and fragment links
        hits += [(p.name, m.group(0)) for m in hexcolor.finditer(text)]
    assert not hits, hits
    assert "scrollbar-color" in (STATIC / "css" / "tokens.css").read_text()


def test_icons_are_vendored_with_their_license():
    icons = STATIC / "icons"
    assert "ISC License" in (icons / "LICENSE-lucide.txt").read_text()
    sprite = (icons / "lucide.svg").read_text()
    assert "Lucide icons (ISC" in sprite
    used = set()
    for p in (STATIC / "js").glob("*.js"):
        used |= set(re.findall(r'icon: "([a-z0-9-]+)"', p.read_text()))
        used |= set(re.findall(r'icon\("([a-z0-9-]+)"', p.read_text()))
    defined = set(re.findall(r'<symbol id="([a-z0-9-]+)"', sprite + (icons / "mace-icons.svg").read_text()))
    assert used - {""} <= defined, sorted(used - defined)


def test_boot_is_one_request_with_everything_the_page_starts_with(served):
    status, headers, body = req(served[0], "api/boot", headers={"Accept-Encoding": "gzip"})
    assert status == 200 and headers.get("Content-Encoding") == "gzip"
    import gzip
    boot = json.loads(gzip.decompress(body))
    assert set(boot) == {"info", "files", "templates", "settings", "mace_defaults", "profile"}
    assert any(f["path"] == "structures/MgO.cif" for f in boot["files"]["files"])
    assert len(body) < 3000


def test_static_cache_is_bounded(tmp_path, monkeypatch):
    """Spec section 9: caches are capped by bytes (MACE_GUI_CACHE_MB) and
    drop the least recently used entry first."""
    from mace.gui import server
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    st = server.GuiState(tmp_path, "t")
    try:
        assert st.cache_limit == 64 * 1024 * 1024          # the floor: 64 MB (spec section 9)
        st.cache_limit = 40_000           # smaller than the shell's files together
        names = ["js/shell.js", "css/shell.css", "js/actions.js", "app.js", "index.html"]
        for name in names:
            body, _ = st.static_bytes(name, False)
            assert body == (STATIC / name).read_bytes().replace(b"@V@", st.version.encode())
        assert st._static_bytes <= max(st.cache_limit, max(len(b) for b in st._static_cache.values()))
        assert list(st._static_cache)[-1] == ("index.html", False)
        assert ("js/shell.js", False) not in st._static_cache
        assert st._static_bytes == sum(len(b) for b in st._static_cache.values())
    finally:
        st.close()
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "lots")
    assert server.cache_limit_bytes() == 512 * 1024 * 1024


def test_no_hard_coded_shortcuts():
    """Spec §4.1: menus, toolbar, tooltips and the palette read shortcuts from
    the registry; no data-keys="..." literal may creep back into the shell."""
    hits = []
    for p in list((STATIC / "js").rglob("*.js")) + [STATIC / "index.html"]:
        for m in re.finditer(r'data-keys="([^"$]*)"', p.read_text()):
            hits.append((p.name, m.group(0)))
    assert not hits, hits



# --- Phase 2: validated settings, the section 9 resources, the CLI's defaults ------------------------

import hashlib
import os


def dir_md5(root):
    h = hashlib.md5()
    for p in sorted(root.rglob("*")):
        h.update(p.relative_to(root).as_posix().encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def post_settings(base, values):
    status, _, body = req(base, "api/settings", data={"settings": values})
    return status, json.loads(body)


def test_one_bad_key_does_not_block_the_others(served):
    base, root, xdg = served
    md5 = dir_md5(root)
    assert post_settings(base, {"cache_mb": 256})[0] == 200
    status, r = post_settings(base, {"theme": "dark", "cache_mb": "big", "max_workers": -1, "gpu_device": "GPU0",
                                     "labelSize": float("1e400") if False else 14})
    assert status == 200
    assert r["saved"] == ["labelSize", "theme"], r
    assert set(r["rejected"]) == {"cache_mb", "max_workers", "gpu_device"}, r
    assert "64 to 65536" in r["rejected"]["cache_mb"]
    stored = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert stored == {"theme": "dark", "labelSize": 14, "cache_mb": 256}    # the old cache_mb is kept
    # a request that is not a settings object at all is refused whole
    assert req(base, "api/settings", data=["nope"])[0] == 400
    r400 = urllib.request.Request(base + "api/settings", data=b"{not json", method="POST",
                                  headers={"X-Requested-With": "mace-gui", "X-Mace-Token": H.TOKEN, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(r400, timeout=10)
        assert False, "bad JSON was accepted"
    except urllib.error.HTTPError as e:
        assert e.code == 400
    assert dir_md5(root) == md5
    post_settings(base, {})


def test_keymap_entries_are_checked_one_by_one(served):
    base, _, xdg = served
    km = [{"command": "view.alongA", "keys": ["7"]}, {"command": 5}, {"command": "edit.duplicate", "remove": True},
          {"command": "x", "keys": "notalist"}, "junk"]
    status, r = post_settings(base, {"keymap": km})
    assert status == 200 and r["saved"] == ["keymap"] and "3 entries" in r["notes"]["keymap"], r
    stored = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert stored["keymap"] == [km[0], km[2]]
    post_settings(base, {})


def test_max_workers_stored_then_clamped_at_use(served):
    """A value written on a bigger host stays stored; this host clamps it when used."""
    base, _, _ = served
    from mace.gui import resources
    status, r = post_settings(base, {"max_workers": 999})
    assert status == 200 and r["saved"] == ["max_workers"]
    assert json.loads(req(base, "api/settings")[2])["settings"]["max_workers"] == 999
    res = json.loads(req(base, "api/info")[2])["resources"]
    host = min(len(os.sched_getaffinity(0)), resources.slurm_cpus() or 10 ** 9)
    assert res["workers_stored"] == 999 and res["workers_effective"] == host, res
    post_settings(base, {})
    res = json.loads(req(base, "api/info")[2])["resources"]
    assert res["workers_effective"] == res["workers_cap"] == resources.worker_cap() <= 8
    assert res["workers_cap"] == max(1, min(len(os.sched_getaffinity(0)) // 4, 8))
    assert res["pool_started"] is False and res["worker_niceness"] == 10
    assert res["gpu_device"] == "0" and res["cache_mb"] == 512


def test_cache_mb_setting_shrinks_the_cache(served, tmp_path):
    base = served[0]
    post_settings(base, {"cache_mb": 64})
    res = json.loads(req(base, "api/info")[2])["resources"]
    assert res["cache_mb"] == 64 and res["cache_used_mb"] <= 64
    post_settings(base, {})
    assert json.loads(req(base, "api/info")[2])["resources"]["cache_mb"] == 512
    # and the live cache evicts at once when its cap shrinks
    from mace.gui import server
    os.environ.pop("MACE_GUI_CACHE_MB", None)
    st = server.GuiState(tmp_path, "t")
    try:
        for name in ["js/shell.js", "css/shell.css", "js/actions.js", "app.js"]:
            st.static_bytes(name, False)
        full = st.cache_used()
        st.set_limit(full // 3)
        assert st.cache_used() <= max(full // 3, max(len(b) for b in st._static_cache.values()))
        assert len(st._static_cache) < 4
    finally:
        st.close()


def test_resources_env_beats_file_beats_default(monkeypatch):
    from mace.gui import resources
    for k in ("MACE_GUI_MAX_WORKERS", "MACE_GUI_CACHE_MB", "MACE_GUI_GPU", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(k, raising=False)
    assert resources.effective_workers(None) == resources.worker_cap()
    assert resources.effective_workers(2) == 2
    assert resources.cache_mb(None) == 512 and resources.cache_mb(1024) == min(1024, resources.physical_ram_mb() // 4)
    assert resources.gpu_device(None) == "0" and resources.gpu_device("none") == "none"
    monkeypatch.setenv("MACE_GUI_MAX_WORKERS", "3")
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "128")
    monkeypatch.setenv("MACE_GUI_GPU", "1")
    # a workstation that keeps GPU 0 for the display sets MACE_GUI_GPU=1: it wins over the file
    assert resources.effective_workers(2) == 3 and resources.cache_mb(1024) == 128 and resources.gpu_device("0") == "1"
    assert set(resources.pinned_by_env()) == {"max_workers", "cache_mb", "gpu_device"}
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "2")
    assert resources.worker_cap() == max(1, min(len(os.sched_getaffinity(0)) // 4, 8, 2))
    assert resources.effective_workers(None) == 2                # the env's 3, within SLURM's 2
    monkeypatch.delenv("MACE_GUI_MAX_WORKERS")
    assert resources.effective_workers(64) == 2                 # never above what SLURM gave


def test_unparsable_worker_env_does_not_break_info(tmp_path):
    """MACE_GUI_MAX_WORKERS=inf (or 1e400) is ignored like any non-number, not a 500."""
    from mace.gui import resources
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="inf")
    try:
        for route in ("api/info", "api/boot"):
            status, _, body = req(base, route)
            assert status == 200, (route, body[:200])
        res = json.loads(req(base, "api/info")[2])["resources"]
        assert res["workers_effective"] == res["workers_cap"] == resources.worker_cap()
        assert "max_workers" not in res["pinned_by_env"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_bad_file_values_do_not_break_boot(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "gui.json").write_text(json.dumps(
        {"version": 1, "settings": {"theme": "dark", "max_workers": 4096, "cache_mb": "big", "keymap": [{"command": 1}]}}))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        status, _, body = req(base, "api/boot")
        assert status == 200
        boot = json.loads(body)
        assert boot["settings"]["settings"] == {"theme": "dark", "keymap": []}
        assert boot["info"]["resources"]["cache_mb"] == 512
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_boot_names_the_cli_defaults(served):
    """Settings › MACE defaults shows what the CLI does, read from its own modules."""
    import sys
    sys.path.insert(0, str(H.REPO_ROOT / "Crystal_d12"))
    import d12_config
    import d12_constants
    d = json.loads(req(served[0], "api/boot")[2])["mace_defaults"]
    assert d["functional"] == d12_constants.UNRECOGNISED_FUNCTIONAL_FALLBACK == "HSE06"
    std = d12_config.get_default_d12_configs()["standard_dft_opt"]
    assert d["basis_set"] == std["basis_set"] and d["example_config"] == "standard_dft_opt"
    assert "UNRECOGNISED_FUNCTIONAL_FALLBACK" in d["source"]["functional"]
    assert "POB-TZVP-REV2" in d["basis_sets"] and "PBE" in d["functionals"]


def test_open_gpu_starts_the_browser_with_prime_offload(monkeypatch):
    """Spec 9.3: `mace gui --open --gpu 1` starts the browser on GPU1 (PRIME
    render offload, proven with nvidia-smi on the workstation); the default
    leaves the environment alone, and the server's own environment is restored."""
    from mace.gui import cli
    seen = {}
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: seen.update(
        {k: os.environ.get(k) for k in ("__NV_PRIME_RENDER_OFFLOAD", "__NV_PRIME_RENDER_OFFLOAD_PROVIDER")}) or True)
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    cli.open_browser("http://x", 1)
    assert seen == {"__NV_PRIME_RENDER_OFFLOAD": "1", "__NV_PRIME_RENDER_OFFLOAD_PROVIDER": "NVIDIA-G0"}
    assert "__NV_PRIME_RENDER_OFFLOAD" not in os.environ
    seen.clear()
    cli.open_browser("http://x", None)
    assert seen == {"__NV_PRIME_RENDER_OFFLOAD": None, "__NV_PRIME_RENDER_OFFLOAD_PROVIDER": None}
    assert cli.browser_gpu_env(0) == {} and cli.build_parser().parse_args(["--open", "--gpu", "1"]).gpu == 1


def test_display_range_defaults_are_storable(served):
    """Polish A3: the range is two plain lists (a dict of lists could never be
    saved), checked on the server: 3 numbers in [-5, 10], hi above lo by 0.05."""
    base, root, xdg = served
    status, _, body = req(base, "api/settings", data={"settings": {"displayLo": [-0.1, -0.1, -0.1],
                                                                   "displayHi": [1.1, 1.1, 1.1], "depthCue": 0.3}})
    r = json.loads(body)
    assert status == 200 and not r["rejected"], r
    stored = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert stored["displayLo"] == [-0.1, -0.1, -0.1] and stored["displayHi"] == [1.1, 1.1, 1.1]
    r = json.loads(req(base, "api/settings", data={"settings": {"displayLo": [0, 0, 0], "displayHi": [0, 0, 0]}})[2])
    assert set(r["rejected"]) == {"displayLo", "displayHi"} and "at least 0.05" in r["rejected"]["displayHi"]
    r = json.loads(req(base, "api/settings", data={"settings": {"displayLo": [0, 0]}})[2])
    assert "displayLo" in r["rejected"]
    r = json.loads(req(base, "api/settings", data={"settings": {"depthCue": 2, "lightDirection": "below",
                                                                "bondedOutside": "all", "outlineColor": "red",
                                                                "lightStrength": 2}})[2])
    assert set(r["rejected"]) == {"depthCue", "lightDirection", "bondedOutside", "outlineColor", "lightStrength"}
    req(base, "api/settings", data={"settings": {}})


def test_index_carries_server_prefs(served):
    """Polish P3: theme and UI scale from gui.json on the page's <html> tag, so
    a new port or a fresh browser paints them at once; per request, gzip equal
    to plain, nothing hostile reflected, no prefs in the static cache."""
    import gzip
    base, root, xdg = served
    req(base, "api/settings", data={"settings": {"theme": "dark", "uiScale": 120}})
    status, headers, body = req(base, "")
    assert status == 200 and headers["Cache-Control"] == "no-store" and "ETag" not in headers
    html = re.search(rb"<html[^>]*>", body).group(0).decode()
    assert 'data-theme="dark"' in html and "--ui-scale:1.2" in html and 'data-prefs="server"' in html, html
    st, h2, gz = req(base, "", headers={"Accept-Encoding": "gzip"})
    assert h2.get("Content-Encoding") == "gzip" and gzip.decompress(gz) == body
    assert req(base, "", headers={"If-None-Match": '"v-x-0"'})[0] == 200
    # a hand-edited, hostile value is never reflected
    f = xdg / "mace" / "gui.json"
    data = json.loads(f.read_text())
    data["settings"]["theme"] = 'x" onload="alert(1)'
    data["settings"]["uiScale"] = "120; background:url(x)"
    f.write_text(json.dumps(data))
    body = req(base, "")[2]
    html = re.search(rb"<html[^>]*>", body).group(0).decode()
    assert "onload" not in html and "url(" not in html and "data-theme" not in html and "--ui-scale:1" in html, html
    req(base, "api/settings", data={"settings": {}})
