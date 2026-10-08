"""POST /api/shutdown (Sign Out and Stop Server, plan T10) through the real
`mace_cli gui` subprocess and HTTP: the header rule, a clean exit 0 with the
reason line, the cache folder and URL file removed, the port freed; `now`
cancels a running task; `after-tasks` keeps the session working and stops
when the task ends; `cancel` undoes it; managed mode reports the reason."""
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H


class Srv:
    def __init__(self, tmp, *extra, managed=False, **env):
        root = tmp / "served"
        root.mkdir(exist_ok=True)
        self.tmpdir = tmp / "tmp"
        self.tmpdir.mkdir(exist_ok=True)
        e = H.env(tmp / "xdg", MACE_GUI_NO_WARMUP="1", TMPDIR=str(self.tmpdir), **env)
        if not managed:
            e.pop("MACE_GUI_PRINT_TOKEN")              # stdout is a pipe: the URL file is written
            e["XDG_RUNTIME_DIR"] = str(tmp / "run")
            (tmp / "run").mkdir(exist_ok=True)
        argv = [sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                "--token", H.TOKEN, *extra] + (["--managed"] if managed else [])
        self.p = H.ServerProcess(argv, stdin=subprocess.PIPE if managed else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                                 start_new_session=True, preexec_fn=H.die_with_pytest)
        self.lines = []
        for _ in range(40):
            line = self.p.stdout.readline()
            self.lines.append(line)
            if managed and line.startswith("MACE_GUI_READY "):
                r = json.loads(line.split(" ", 1)[1])
                self.port, self.prefix = r["port"], r["prefix"]
                break
            m = re.search(r"http://127\.0\.0\.1:(\d+)(/s/[^/]+/)", line)
            if m and not managed:
                self.port, self.prefix = int(m.group(1)), m.group(2)
                break
        threading.Thread(target=lambda: [self.lines.append(x) for x in self.p.stdout], daemon=True).start()
        self.urlfile = next(iter((tmp / "run").glob("mace-gui-*.json")), None) if not managed else None

    def req(self, route, body=None, headers=None, xrw=True):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"X-Mace-Token": H.TOKEN, "Content-Type": "application/json", **(headers or {})}
        if xrw:
            h["X-Requested-With"] = "mace-gui"
        c.request("POST" if body is not None else "GET", self.prefix + route,
                  body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data), r.getheader("Set-Cookie")
        except ValueError:
            return r.status, data, r.getheader("Set-Cookie")

    def gone(self, s):
        try:
            self.p.wait(s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(5)


def port_free(port):
    with socket.socket() as s:
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            return False
        except OSError:
            return True


def test_needs_the_header_then_stops_cleanly(tmp_path):
    s = Srv(tmp_path)
    try:
        assert s.urlfile is not None and s.urlfile.exists()
        assert list(s.tmpdir.glob("mace_gui_cache_*"))
        assert s.req("api/shutdown", {"mode": "now"}, xrw=False)[0] == 403
        st, body, cookie = s.req("api/shutdown", {"mode": "now"})
        assert st == 202 and body == {"stopping": True}
        assert "Max-Age=0" in cookie and f"Path={s.prefix}" in cookie
        t0 = time.time()
        assert s.gone(3) and s.p.returncode == 0 and time.time() - t0 < 3
        time.sleep(0.2)
        assert any("stopped from the browser (Sign Out and Stop Server)" in ln for ln in s.lines), s.lines
        assert not list(s.tmpdir.glob("mace_gui_cache_*"))
        assert not s.urlfile.exists()
        assert port_free(s.port)
    finally:
        s.kill()


def test_now_cancels_a_running_task(tmp_path):
    s = Srv(tmp_path, MACE_GUI_TEST_HOOKS="1")
    try:
        st, t, _ = s.req("api/test/task", {"seconds": 30, "job": "sleep", "wait": 0})
        assert st == 202
        ev = H.SSE(f"http://127.0.0.1:{s.port}{s.prefix}")
        st, _, _ = s.req("api/shutdown", {"mode": "now"})
        assert st == 202
        ev.wait(lambda e: e[0] == "server-stopping" and e[1] == {"reason": "browser"}, 5)
        ev.wait(lambda e: e[0] == "task" and e[1]["state"] == "canceled", 5)
        assert s.gone(3) and s.p.returncode == 0
    finally:
        s.kill()


def test_after_tasks_keeps_the_session_then_stops(tmp_path):
    s = Srv(tmp_path, MACE_GUI_TEST_HOOKS="1")
    try:
        s.req("api/test/task", {"seconds": 4, "job": "sleep", "wait": 0})
        st, body, cookie = s.req("api/shutdown", {"mode": "after-tasks"})
        assert st == 202 and body["pending_stop"] is True and len(body["waiting_for"]) == 1 and cookie is None
        st, info, _ = s.req("api/info")
        assert st == 200 and info["stop_pending"]["waiting_for"] == body["waiting_for"]
        assert s.p.poll() is None
        assert s.gone(4 + 2 + 5), "no stop after the task ended"
        assert s.p.returncode == 0
    finally:
        s.kill()


def test_cancel_stop_keeps_it_running(tmp_path):
    s = Srv(tmp_path, MACE_GUI_TEST_HOOKS="1")
    try:
        assert s.req("api/shutdown", {"mode": "cancel"})[0] == 409
        s.req("api/test/task", {"seconds": 2, "job": "sleep", "wait": 0})
        assert s.req("api/shutdown", {"mode": "after-tasks"})[0] == 202
        st, body, _ = s.req("api/shutdown", {"mode": "cancel"})
        assert st == 200 and body == {"canceled": True}
        time.sleep(2 + 3)
        assert s.p.poll() is None and s.req("api/info")[0] == 200
    finally:
        s.kill()


def test_managed_prints_the_reason(tmp_path):
    s = Srv(tmp_path, managed=True)
    try:
        assert s.req("api/shutdown", {"mode": "now"})[0] == 202
        assert s.gone(5)
        time.sleep(0.2)
        assert 'MACE_GUI_STOPPED {"reason": "browser"}' in "".join(s.lines)
    finally:
        s.kill()
