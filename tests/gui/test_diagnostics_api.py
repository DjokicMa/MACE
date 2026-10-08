"""Copy Diagnostics through the real server and `mace_cli gui diagnostics`:
the bundle carries versions, the OS and `<file>.out`, and none of the
needles: the token, the session path, session ids, sign-in codes, the fake
HOME and served path, a made-up system name in every form (plain, encoded,
as a stem), free-text settings, the user and host names."""
import getpass
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

import guiharness as H

REPO = Path(__file__).resolve().parents[2]
CORPUS_OUT = REPO / "test" / "OPT" / "1_dia_opt_rev1.out"
SYSTEM = "Zq9Fake_system"


@pytest.fixture
def world(tmp_path):
    home = tmp_path / "home of Zq9Fakehome"
    served = home / "zq9proj"
    served.mkdir(parents=True)
    src = CORPUS_OUT if CORPUS_OUT.is_file() else H.EXAMPLES / "outputs" / "Si_sp.out"
    shutil.copy(src, served / f"{SYSTEM}.out")
    shutil.copy(H.EXAMPLES / "structures" / "MgO.cif", served / f"{SYSTEM}.cif")
    return tmp_path, home, served


def test_bundle_has_versions_and_no_needles(world):
    tmp, home, served = world
    import re as _re
    e = H.env(tmp / "xdg", HOME=str(home), USER="zqxuser", LOGNAME="zqxuser", MACE_GUI_NO_WARMUP="1",
              MACE_GUI_TEST_HOOKS="1")
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(served), "--port", "0",
                         "--token", H.TOKEN, "--verbose"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True,
                        preexec_fn=H.die_with_pytest)
    try:
        for _ in range(40):
            m = _re.search(r"(http://127\.0\.0\.1:(\d+)(/s/[^/]+/))\?login=(\S+)", p.stdout.readline())
            if m:
                break
        base, port, prefix, code = m.group(1), int(m.group(2)), m.group(3), m.group(4)

        def req(path, body=None, headers=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            h = {"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui", "Content-Type": "application/json",
                 **(headers or {})}
            c.request("POST" if body is not None else "GET", prefix + path,
                      body=json.dumps(body) if body is not None else None, headers=h)
            r = c.getresponse()
            out = r.read()
            c.close()
            return r.status, out, {k.lower(): v for k, v in r.getheaders()}
        # a session cookie and a fresh code exist (both must be redacted)
        st, _, hd = req(f"?login={code}", headers={"X-Mace-Token": ""})
        sid = (hd.get("set-cookie") or "=;").split(";")[0].split("=", 1)[1]
        code2 = json.loads(req("api/login-code", {})[1])["code"]
        # the Explorer lists the folder; the file opens by relative name, with an encoded request in the log
        assert req("api/files")[0] == 200
        assert req("api/text?path=" + quote(f"{SYSTEM}.out", safe=""))[0] == 200
        # a failing task that names the file in its error and traceback
        req("api/structure?path=" + quote(f"missing/{SYSTEM}_opt.out", safe=""))
        # free-text settings
        assert req("api/settings", {"settings": {"slurm.account": "zqxacct", "partition": "zqxpart", "theme": "dark"}})[0] == 200
        client = {"errors": [f"{base}?login={code} failed", f"TypeError at {prefix}static/x.js:3",
                             f"open {served}/{SYSTEM}.out"],
                  "failed requests": ["GET /api/text 404 12ms"], "user agent": "Mozilla/5.0 (X11) Firefox/140.0"}
        st, body, _ = req("api/diagnostics", {"client": client})
        assert st == 200, body
        text = json.loads(body)["text"]
        dump = text + json.dumps(json.loads(body)["json"])
        needles = [H.TOKEN, prefix.strip("/").split("/")[-1], sid, code, code2, str(home), str(served), SYSTEM,
                   quote(SYSTEM), "Zq9Fakehome", "zqxacct", "zqxpart", "zqxuser", getpass.getuser(),
                   socket.gethostname(), socket.gethostname().split(".")[0]]
        for n in needles:
            if n and len(n) >= 3:
                assert n not in dump, (n, [ln for ln in dump.splitlines() if n in ln][:3])
        assert text.startswith("MACE GUI diagnostics. Paths, file and folder names, tokens, user and host names "
                               "were removed; read it before you share it.")
        from mace import __version__
        assert f"MACE: {__version__}" in text and "Python: " in text and "OS: " in text
        assert "<file>.out" in text and "theme: dark" in text and "slurm.account: <text>" in text
        assert "GET /api/text 404 12ms" in text
    finally:
        p.terminate()
        p.wait(15)


def test_cli_twin_runs_and_hides_names(world, tmp_path):
    tmp, home, served = world
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    e.update(HOME=str(home), XDG_CONFIG_HOME=str(tmp / "xdg2"), XDG_CACHE_HOME=str(tmp / "cache2"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "diagnostics", "--dir",
                        str(served)], env=e, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("MACE GUI diagnostics.")
    assert "mode: no server (mace gui diagnostics)" in r.stdout
    for n in (SYSTEM, str(served), str(home), getpass.getuser()):
        assert n not in r.stdout, n


def test_cli_twin_with_the_standard_library_only(world, tmp_path):
    import test_connect_stdlib as B
    tmp, home, served = world
    d = tmp / "blocker"
    d.mkdir()
    (d / "sitecustomize.py").write_text(B.BLOCKER)
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    e.update(PYTHONPATH=str(d), HOME=str(home), XDG_CONFIG_HOME=str(tmp / "x3"), XDG_CACHE_HOME=str(tmp / "c3"))
    for argv in (["gui", "diagnostics", "--dir", str(served)], ["gui", "login-url", "--port", "1"]):
        r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", *argv], env=e,
                           capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        assert "blocked" not in r.stderr, r.stderr
        assert r.returncode in (0, 1), r.stderr
