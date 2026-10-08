"""Browser sessions (plan T11) through the real `mace_cli gui` subprocess and
HTTP: one-time sign-in links (the history URL signs nobody in), the token only
in the header and the sign-in form, the pre-auth allow-list and its CSP, paced
wrong credentials that never block a correct one, expiry that only real input
prevents (polls, the event stream and background POSTs do not), `mace gui
login-url`, and the idle stop."""
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import guiharness as H


class Srv:
    def __init__(self, tmp, **env):
        root = tmp / "served"
        root.mkdir(exist_ok=True)
        (tmp / "run").mkdir(exist_ok=True)
        e = H.env(tmp / "xdg", MACE_GUI_NO_WARMUP="1", XDG_RUNTIME_DIR=str(tmp / "run"), **env)
        self.env = e
        self.p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root),
                                  "--port", "0", "--token", H.TOKEN], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                                 start_new_session=True, preexec_fn=H.die_with_pytest)
        self.lines = []
        for _ in range(40):
            line = self.p.stdout.readline()
            self.lines.append(line)
            m = re.search(r"(http://127\.0\.0\.1:(\d+)(/s/[^/]+/))\?login=(\S+)", line)
            if m:
                self.base, self.port, self.prefix, self.code = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                break
        else:
            raise AssertionError("".join(self.lines))
        threading.Thread(target=lambda: [self.lines.append(x) for x in self.p.stdout], daemon=True).start()
        self.cookie_name = f"mace_gui_{self.port}"

    def req(self, path, method="GET", body=None, headers=None, raw=False):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
            h.setdefault("X-Requested-With", "mace-gui")
        c.request(method, path if raw else self.prefix + path, body=data, headers=h)
        r = c.getresponse()
        out = r.read()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        c.close()
        return r.status, hdrs, out

    def sign_in(self, code=None):
        st, h, _ = self.req(f"?login={code or self.code}")
        assert st == 303, st
        return h["set-cookie"].split(";")[0]

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            self.p.wait(15)


def test_printed_link_signs_in_once_and_the_token_never_in_a_url_or_cookie(tmp_path):
    s = Srv(tmp_path)
    try:
        assert any(f"Token (for the sign-in form): {H.TOKEN}" in ln for ln in s.lines)
        cookie = s.sign_in()
        name, sid = cookie.split("=", 1)
        assert name == s.cookie_name and sid != H.TOKEN
        assert s.req("api/info", headers={"Cookie": cookie})[0] == 200
        # the history case: the same link again
        st, h, body = s.req(f"?login={s.code}")
        assert st == 403 and b"already used" in body and "set-cookie" not in h
        assert b"name=color-scheme content='light dark'" in body
        # the same link in the browser that is signed in already: the app, no new session
        st, h, _ = s.req(f"?login={s.code}", headers={"Cookie": cookie})
        assert st == 303 and h["location"] == "./" and "set-cookie" not in h
        # ... but an expired or unknown session cookie still gets the deny page
        st, h, body = s.req(f"?login={s.code}", headers={"Cookie": f"{s.cookie_name}=NotASession0123456789"})
        assert st == 403 and b"already used" in body
        # the token in a URL: the deny page, no cookie
        st, h, body = s.req(f"?token={H.TOKEN}")
        assert st == 403 and "set-cookie" not in h and b"sign-in link" in body
        # a cookie whose value is the token
        assert s.req("api/info", headers={"Cookie": f"{s.cookie_name}={H.TOKEN}"})[0] == 403
        # the header works
        assert s.req("api/info", headers={"X-Mace-Token": H.TOKEN})[0] == 200
    finally:
        s.stop()


def test_sign_in_form_and_the_pre_auth_allow_list(tmp_path):
    s = Srv(tmp_path)
    try:
        st, h, body = s.req("login", "POST", {"token": H.TOKEN})
        assert st == 200 and json.loads(body) == {"signed_in": True}
        cookie = h["set-cookie"].split(";")[0]
        assert s.req("api/info", headers={"Cookie": cookie})[0] == 200
        st, info = s.req("api/info", headers={"X-Mace-Token": H.TOKEN})[:2]
        version = json.loads(s.req("api/info", headers={"X-Mace-Token": H.TOKEN})[2])["static"]
        for path in ("static/js/app.js", f"static/{version}/js/shell.js", "api/info", "static/login.js"):
            assert s.req(path)[0] == 403, path
        st, h, body = s.req("login.js")
        assert st == 200 and b"signin" in body
        assert "form-action 'none'" in h["content-security-policy"] and "default-src 'none'" in h["content-security-policy"]
        st, h, body = s.req("")
        assert st == 403 and "script-src 'self'" in h["content-security-policy"] and b"login.js" in body
        # a cross-site page cannot sign in
        st, _, _ = s.req("login", "POST", {"token": H.TOKEN}, headers={"Origin": "http://evil.example"})
        assert st == 403
        st, _, _ = s.req("login", "POST", {"token": H.TOKEN}, headers={"Sec-Fetch-Site": "cross-site"})
        assert st == 403
        assert s.req("login", "POST", {"token": H.TOKEN}, headers={"X-Requested-With": ""})[0] == 403
    finally:
        s.stop()


def test_wrong_attempts_are_paced_and_never_block_a_correct_one(tmp_path):
    s = Srv(tmp_path)
    try:
        results, lock = [], threading.Lock()

        def wrong(i):
            t0 = time.time()
            st = s.req("login", "POST", {"token": f"wrong-token-{i:04d}-xxxxxxxx"})[0]
            with lock:
                results.append((st, time.time() - t0))
        threads = [threading.Thread(target=wrong, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        t0 = time.time()
        st, h, _ = s.req("login", "POST", {"token": H.TOKEN})
        took = time.time() - t0
        assert st == 200 and took < 0.5, took
        for t in threads:
            t.join(60)
        codes = [r[0] for r in results]
        # at most 16 wait their second each; how many get the immediate 429 depends on thread timing
        assert set(codes) <= {403, 429} and 1 <= codes.count(403) <= 40
        paced = sorted(r[1] for r in results if r[0] == 403)
        # one at a time, one second each
        assert paced and paced[-1] >= len(paced) - 1.5, paced
    finally:
        s.stop()


def _sse(s, cookie):
    c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=60)
    c.request("GET", s.prefix + "api/events?interval=0.5", headers={"Cookie": cookie})
    r = c.getresponse()
    events = []

    def read():
        try:
            for line in r:
                events.append(line.decode().strip())
        except Exception:
            pass
    threading.Thread(target=read, daemon=True).start()
    return events, c


def test_only_input_keeps_a_session(tmp_path):
    s = Srv(tmp_path, MACE_GUI_SESSION_IDLE_HOURS="0.001", MACE_GUI_TEST_HOOKS="1")   # 3.6 s
    try:
        idle = s.sign_in()
        code2 = json.loads(s.req("api/login-code", "POST", {}, headers={"X-Mace-Token": H.TOKEN})[2])["code"]
        busy = s.sign_in(code2)
        events, conn = _sse(s, idle)
        stop = threading.Event()

        def background():                      # polls, a live tail, background POSTs: no input
            while not stop.is_set():
                s.req("api/tasks", headers={"Cookie": idle})
                s.req("api/files", headers={"Cookie": idle})
                s.req("api/settings", "POST", {"settings": {}}, headers={"Cookie": idle})
                time.sleep(0.3)

        def toucher():
            while not stop.is_set():
                s.req("api/session/touch", "POST", {}, headers={"Cookie": busy})
                time.sleep(2)
        threads = [threading.Thread(target=background, daemon=True), threading.Thread(target=toucher, daemon=True)]
        for t in threads:
            t.start()
        time.sleep(5.5)
        st, _, body = s.req("api/info", headers={"Cookie": idle})
        assert st == 401 and json.loads(body) == {"error": "session expired", "expired": True}
        assert "event: expired" in events, events[-10:]
        assert s.req("api/info", headers={"Cookie": busy})[0] == 200
        # an expired session still loads the page's files (the Signed Out dialog), and api/session says why
        assert s.req("static/js/shell.js", headers={"Cookie": idle})[0] == 200
        st, _, body = s.req("api/session", headers={"Cookie": idle})
        assert st == 200 and json.loads(body)["expired"] is True
        stop.set()
        conn.close()
    finally:
        s.stop()


def test_login_url_prints_a_fresh_working_link(tmp_path):
    s = Srv(tmp_path)
    try:
        e = {k: v for k, v in s.env.items()}
        r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "login-url", "--port", str(s.port)],
                           env=e, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        link = r.stdout.strip()
        assert link.startswith(s.base + "?login=") and link != s.base + "?login=" + s.code
        code = link.split("?login=")[1]
        assert s.sign_in(code)
        st = json.loads((tmp_path / "run" / f"mace-gui-{s.port}.json").read_text())
        assert st["token"] == H.TOKEN and st["base"] == s.base
        # no running server on another port
        r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "login-url", "--port", "1"],
                           env=e, capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "no MACE GUI" in r.stderr
    finally:
        s.stop()


def test_idle_stop_without_input(tmp_path):
    s = Srv(tmp_path, MACE_GUI_IDLE_STOP_HOURS="0.002")          # 7.2 s
    try:
        cookie = s.sign_in()
        events, conn = _sse(s, cookie)
        stop = threading.Event()

        def polls():
            while not stop.is_set():
                try:
                    s.req("api/tasks", headers={"Cookie": cookie})
                except OSError:          # the server stopped
                    return
                time.sleep(0.5)
        threading.Thread(target=polls, daemon=True).start()
        s.p.wait(7.2 + 2 + 6)
        stop.set()
        assert s.p.returncode == 0
        time.sleep(0.3)
        assert any("stopped after" in ln and "without use" in ln for ln in s.lines), s.lines[-5:]
    finally:
        s.stop()


def test_idle_stop_kept_off_by_touches_and_by_a_running_task(tmp_path):
    s = Srv(tmp_path, MACE_GUI_IDLE_STOP_HOURS="0.002", MACE_GUI_TEST_HOOKS="1")
    try:
        cookie = s.sign_in()
        t_end = time.time() + 11
        while time.time() < t_end:
            s.req("api/session/touch", "POST", {}, headers={"Cookie": cookie})
            time.sleep(2)
        assert s.p.poll() is None, "touches every 2 s keep it running"
        # a task: no stop until it ends (no more touches from here)
        assert s.req("api/test/task", "POST", {"seconds": 12, "job": "sleep", "wait": 0},
                     headers={"Cookie": cookie})[0] == 202
        time.sleep(10)
        assert s.p.poll() is None, "a running task keeps it"
        s.p.wait(12)
        assert s.p.returncode == 0
    finally:
        s.stop()
