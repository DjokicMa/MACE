"""The ML registry (A16), `mace mlip fetch` against a local HTTP server
(A15), `mace mlip setup`, and ML-environment detection and its cache (A2,
A3) with fake interpreters. No torch, no network."""
import copy
import hashlib
import http.server
import json
import os
import re
import stat
import sys
import threading

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)


# --- registry -------------------------------------------------------------------------

def test_registry_ships_two_allowed_models():
    from mace.preopt import registry
    models = registry.load()
    assert [m["id"] for m in models] == ["acesuit-mace-mpa-0-medium", "chgnet-0.3.0"]
    assert registry.default_id() == "acesuit-mace-mpa-0-medium" and registry.fallback_id() == "chgnet-0.3.0"
    assert registry.ALLOWED_LICENSES == frozenset({"MIT", "BSD-3-Clause", "BSD-3-Clause-LBNL", "Apache-2.0"})
    mpa = registry.get("acesuit-mace-mpa-0-medium")
    assert mpa["label"] == "MACE-MPA-0 (ACEsuit mace-torch)" and "unrelated to, this MACE tool" in mpa["unrelated_note"]
    assert mpa["weights"]["sha256"] == "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
    assert mpa["weights"]["size_bytes"] == 79462305
    for m in models:
        for url in (m["code"]["license_text_url"], m["weights"].get("license_text_url", m["code"]["license_text_url"])):
            assert re.match(r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/LICENSE", url), url
        assert m["training_data"] and m["citation"]


def _raw():
    from mace.preopt import registry
    return json.loads(registry.PATH.read_text())


@pytest.mark.parametrize("bad", ["asl", "omat", "nolicense", "json_allowlist"])
def test_registry_rejects_disallowed_models(bad, tmp_path):
    from mace.preopt import registry
    data = _raw()
    extra = copy.deepcopy(data["models"][1])
    extra.update(id="evil-model", default=False, fallback=False)
    if bad == "asl":
        extra["weights"]["license"] = "ASL-1.0"
    elif bad == "omat":
        extra["weights"]["license"] = "OMat24-License"
    elif bad == "nolicense":
        del extra["code"]["license"]
    else:
        data["allowed_licenses"] = sorted(registry.ALLOWED_LICENSES) + ["ASL-1.0"]
        extra["weights"]["license"] = "ASL-1.0"
    data["models"].append(extra)
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(data))
    with pytest.raises(registry.RegistryError):
        registry.load(p)


def test_no_asl_or_meta_model_ids_anywhere():
    from mace.preopt import registry
    words = re.compile(r"\b(omat|fairchem|uma|grace|asl)\b", re.I)
    for m in registry.load():
        for field in (m["id"], m["code"]["package"]):
            assert not words.search(field.replace("-", " ")), field


# --- fetch ----------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}

    def do_GET(self):
        body = self.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _Handler.files
    srv.shutdown()
    _Handler.files.clear()


def _model(base, body, license_path="/LICENSE", sha=None, size=None):
    from mace.preopt import registry
    m = copy.deepcopy(registry.get("acesuit-mace-mpa-0-medium"))
    m["weights"].update(url=f"{base}/w.model", sha256=sha or hashlib.sha256(body).hexdigest(),
                        size_bytes=len(body) if size is None else size, license_text_url=f"{base}{license_path}")
    m["weights"].pop("mirror_url", None)
    m["code"]["license_text_url"] = f"{base}/CODE_LICENSE"
    return m


def test_fetch_shows_everything_before_downloading_and_verifies(server, tmp_path):
    from mace.preopt import detect, fetch
    base, files = server
    body = os.urandom(300_000)
    files.update({"/w.model": body, "/LICENSE": b"MIT License\n\nCopyright (c) 2024 MACE-MP\n",
                  "/CODE_LICENSE": b"MIT License\n\nCopyright (c) 2022 ACEsuit/mace\n"})
    m = _model(base, body)
    out = []
    order = []
    rc = fetch.fetch(m, yes=False, out=out.append, ask=lambda q: (order.append(("ask", len("".join(out)))), "y")[1],
                     isatty=True)
    assert rc == 0
    text = "".join(out)
    asked_at = order[0][1]
    before = text[:asked_at]
    for s in ("unrelated to this MACE tool", "license MIT", m["weights"]["url"], m["weights"]["sha256"],
              "Training data: MPtrj", "Cite:", "Size:"):
        assert s in before, s
    target = detect.weights_root() / m["id"] / m["weights"]["filename"]
    assert target.read_bytes() == body and not target.with_name(target.name + ".part").exists()
    lic = (target.parent / "LICENSE.txt").read_text()
    assert "Copyright (c) 2024 MACE-MP" in lic and "Copyright (c) 2022 ACEsuit/mace" in lic
    acc = json.loads((target.parent / "ACCEPTED.json").read_text())
    assert acc["url"] == m["weights"]["url"]
    assert set(acc) == {"id", "license", "url", "sha256", "size", "mtime_ns", "accepted_utc"}
    assert os.environ.get("USER", "nobody") not in json.dumps(acc)
    assert detect.weights_status(m)["status"] == "ready"
    target.write_bytes(body[:-1] + b"x")
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))        # a later edit of the file
    assert detect.weights_status(m)["status"] == "bad_hash"


def test_fetch_refusals_and_failures(server, monkeypatch):
    from mace.preopt import detect, fetch
    base, files = server
    body = b"weights" * 1000
    files.update({"/w.model": body})
    m = _model(base, body, license_path="/missing")
    out = []
    assert fetch.fetch(m, out=out.append, isatty=False) == 2 and "pass --yes" in "".join(out)
    assert not (detect.weights_root() / m["id"]).exists()
    assert fetch.fetch(m, out=out.append, isatty=True, ask=lambda q: "n") == 2
    bad = _model(base, body, sha="0" * 64)
    out = []
    assert fetch.fetch(bad, yes=True, out=out.append) == 1 and "sha256 mismatch" in "".join(out)
    d = detect.weights_root() / m["id"]
    assert not list(d.glob("*.part")) and not (d / m["weights"]["filename"]).exists()
    short = _model(base, body, size=len(body) + 10)
    assert fetch.fetch(short, yes=True, out=out.append) == 1 and not list(d.glob("*.part"))
    out = []
    long_ = _model(base, body, size=len(body) - 10)              # a server that sends more than pinned
    assert fetch.fetch(long_, yes=True, out=out.append) == 1 and not list(d.glob("*.part"))
    assert "more than the" in "".join(out)
    assert fetch.fetch(m, yes=True, out=out.append) == 0
    lic = (d / "LICENSE.txt").read_text()
    assert "could not be fetched" in lic and f"{base}/missing" in lic
    monkeypatch.setenv("MACE_MLIP_NO_DOWNLOAD", "1")
    out = []
    assert fetch.fetch(m, yes=True, out=out.append) == 1 and "MACE_MLIP_NO_DOWNLOAD=1" in "".join(out)
    from mace.preopt import registry
    out = []
    assert fetch.fetch(registry.get("chgnet-0.3.0"), out=out.append) == 0 and "nothing to fetch" in "".join(out)


# --- setup and detection --------------------------------------------------------------

def mlip(*args, env=None):
    import subprocess
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(env or {})
    r = subprocess.run([sys.executable, str(F.REPO / "mace_cli"), "--no-banner", "mlip", *args],
                       env=e, cwd=os.environ["TMPDIR"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


def test_setup_without_python_only_prints(tmp_path):
    before = set(os.listdir(os.environ["TMPDIR"]))
    rc, out = mlip("setup")
    assert rc == 0
    for s in ("--system-site-packages", "mace-torch==0.3.16 chgnet==0.4.2", "about 80 MB",
              "https://download.pytorch.org/whl/cpu", "CUDA build of several GB", "df -h DIR",
              "mace mlip setup --python DIR/bin/python", "mace mlip fetch acesuit-mace-mpa-0-medium"):
        assert s in out, s
    assert not (tmp_path / "config" / "mace" / "mlip.json").exists()
    assert set(os.listdir(os.environ["TMPDIR"])) <= before | {f"mace-preopt-{os.getuid()}"}
    rc, out = mlip("setup", "--create")
    assert rc == 2


def test_setup_records_only_a_passing_interpreter(tmp_path):
    fake = F.make_fake_python(tmp_path / "venv")
    rc, out = mlip("setup", "--python", str(fake))
    assert rc == 0 and "Recorded in" in out, out
    rec = json.loads((tmp_path / "config" / "mace" / "mlip.json").read_text())
    assert rec == {"python": str(fake)}
    (tmp_path / "config" / "mace" / "mlip.json").unlink()
    rc, out = mlip("setup", "--python", str(fake), env={"FAKE_PROBE": "fail"})
    assert rc == 1 and "Not recorded" in out and "libtorch.so" in out
    assert not (tmp_path / "config" / "mace" / "mlip.json").exists()
    rc, out = mlip("setup", "--python", str(tmp_path / "nope"))
    assert rc == 1 and "not an executable file" in out


def test_mlip_list_statuses(tmp_path, monkeypatch):
    from mace.preopt import detect, registry
    mpa, chg = registry.get("acesuit-mace-mpa-0-medium"), registry.get("chgnet-0.3.0")
    assert detect.mlip_status(mpa)["status"] == "needs_setup"
    fake = F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("MACE_MLIP_PYTHON", str(fake))
    st = detect.mlip_status(mpa)
    assert st["status"] == "needs_fetch" and st["hint"] == "mace mlip fetch acesuit-mace-mpa-0-medium"
    assert detect.mlip_status(chg)["status"] == "ready"
    monkeypatch.setenv("FAKE_PROBE", "fail")
    assert detect.mlip_status(chg)["status"] == "import_failed"
    monkeypatch.delenv("FAKE_PROBE")
    rc, out = mlip("list", env={"MACE_MLIP_PYTHON": str(fake)})
    assert rc == 0 and out.startswith("ML potentials are third-party models, unrelated to this MACE tool.")
    assert "mlip:acesuit-mace-mpa-0-medium (default)" in out and "status: needs_fetch" in out
    assert "mlip:chgnet-0.3.0 (fallback)" in out and "status: ready" in out
    rc, out = mlip("list", "--json", env={"MACE_MLIP_PYTHON": str(fake)})
    data = json.loads(out)
    assert [m["status"] for m in data["models"]] == ["needs_fetch", "ready"]


def test_probe_timeout_and_garbage_are_not_cached(tmp_path, monkeypatch):
    from mace.preopt import detect
    fake = F.make_fake_python(tmp_path / "venv")
    monkeypatch.setenv("FAKE_PROBE", "hang")
    r = detect.probe(str(fake), timeout=1.0)
    assert r["status"] == "timeout" and "Recheck" in r["reason"]
    junk = tmp_path / "junk" / "bin"
    junk.mkdir(parents=True)
    (junk / "python").write_text("#!/bin/sh\necho hello\n")
    (junk / "python").chmod(0o755)
    assert detect.probe(str(junk / "python"))["status"] == "failed"
    assert not detect.cache_path().exists() or json.loads(detect.cache_path().read_text()) == {}


def _fake_venv(root, base_site, name):
    """A venv-shaped directory: bin/python (a shell script answering the probe),
    pyvenv.cfg with system site packages from base_site's interpreter."""
    v = root / name
    (v / "bin").mkdir(parents=True)
    site = v / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "mace_torch-0.3.16.dist-info").mkdir()
    (v / "pyvenv.cfg").write_text(f"home = {base_site}/bin\ninclude-system-site-packages = true\nversion = 3.12.2\n")
    py = v / "bin" / "python"
    py.write_text("#!/bin/sh\n"
                  "if [ -n \"$NEEDS_LD\" ] && [ \"$LD_LIBRARY_PATH\" != \"$NEEDS_LD\" ]; then exit 1; fi\n"
                  "echo 'MACE_PROBE_JSON {\"python\": \"3.12.2\", \"packages\": {\"torch\": {\"ok\": true, \"version\": \"2\"}, "
                  "\"ase\": {\"ok\": true, \"version\": \"3\"}, \"mace-torch\": {\"ok\": true, \"version\": \"0.3.16\"}, "
                  "\"chgnet\": {\"ok\": true, \"version\": \"0.4.2\"}}}'\n")
    py.chmod(0o755)
    return v


def test_cache_keys_venvs_on_one_base_and_invalidates(tmp_path, monkeypatch):
    from mace.preopt import detect
    base = tmp_path / "base"
    bsite = base / "lib" / "python3.12" / "site-packages"
    bsite.mkdir(parents=True)
    (bsite / "torch-2.6.0.dist-info").mkdir()
    a = _fake_venv(tmp_path, base, "venv_a")
    b = _fake_venv(tmp_path, base, "venv_b")
    ka, kb = detect.cache_key(str(a / "bin/python")), detect.cache_key(str(b / "bin/python"))
    assert ka and kb and ka != kb
    assert detect.probe(str(a / "bin/python"))["status"] == "ok"
    cache = json.loads(detect.cache_path().read_text())
    assert ka in cache and kb not in cache
    (bsite / "torch-2.6.0.dist-info").rmdir()                   # an uninstall in the base
    assert detect.cache_key(str(a / "bin/python")) != ka
    (a / "lib/python3.12/site-packages/mace_torch-0.3.16.dist-info").rmdir()
    (b / "lib/python3.12/site-packages/mace_torch-0.3.16.dist-info").rename(
        b / "lib/python3.12/site-packages/mace_torch-0.3.17.dist-info")
    assert detect.cache_key(str(b / "bin/python")) != kb
    detect.cache_path().write_text("{not json")
    detect._MEM.clear()
    assert detect.probe(str(b / "bin/python"))["status"] == "ok"           # a corrupt cache is ignored
    assert json.loads(detect.cache_path().read_text())                      # ... and rewritten


def test_unwritable_cache_keeps_working(tmp_path, monkeypatch):
    from mace.preopt import detect
    ro = tmp_path / "rocache"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("XDG_CACHE_HOME", str(ro))
    try:
        v = _fake_venv(tmp_path, tmp_path / "base2", "venv_c")
        assert detect.probe(str(v / "bin/python"))["status"] == "ok"
    finally:
        ro.chmod(0o755)


def test_probe_env_keeps_ld_library_path(tmp_path, monkeypatch):
    from mace.preopt import detect
    v = _fake_venv(tmp_path, tmp_path / "base3", "venv_ld")
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setenv("NEEDS_LD", str(lib))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(lib))
    assert detect.probe(str(v / "bin/python"), recheck=True)["status"] == "ok"
    monkeypatch.setenv("LD_LIBRARY_PATH", "/elsewhere")
    assert detect.probe(str(v / "bin/python"), recheck=True)["status"] == "failed"


def test_fetch_falls_back_to_the_mirror_and_records_it(server):
    from mace.preopt import detect, fetch
    base, files = server
    body = os.urandom(50_000)
    files.update({"/mirror.model": body})                       # the primary URL answers 404
    m = _model(base, body)
    m["weights"]["mirror_url"] = f"{base}/mirror.model"
    out = []
    assert fetch.fetch(m, yes=True, out=out.append) == 0
    target = detect.weights_root() / m["id"] / m["weights"]["filename"]
    assert target.read_bytes() == body
    acc = json.loads((target.parent / "ACCEPTED.json").read_text())
    assert acc["url"] == f"{base}/mirror.model"
