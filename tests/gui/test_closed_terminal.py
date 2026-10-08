"""A server whose terminal went away keeps answering.

`ssh node 'mace gui --dir ...'` without a tty: closing that ssh leaves the
server running (no hangup) with its stdout and stderr on a dead pipe. Found
on the HPCC dev node: File > Open Folder switched the folder, then its
"now serving" note hit the dead pipe and the request died without a
response. Real `mace_cli gui` subprocess; the test closes the pipe's reader."""
import http.client
import json
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

import guiharness as H


def _post(port, prefix, route, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", prefix + "api/" + route, body=json.dumps(body), headers={
        "X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui", "Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data)


def test_open_folder_answers_after_the_terminal_closed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    e = H.env(tmp_path / "xdg", HOME=str(tmp_path), MACE_GUI_NO_WARMUP="1")   # a and b: in the home folder
    e.pop("MACE_GUI_ROOTS", None)
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(a),
                         "--port", "0", "--token", H.TOKEN],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        env=e, start_new_session=True, preexec_fn=H.die_with_pytest)
    try:
        base = None
        for _ in range(40):
            m = re.search(H.URL_RE, p.stdout.readline().decode(errors="replace"))
            if m:
                base = m.group(1)
                break
        assert base, "the server printed no URL"
        p.stdout.close()                      # the terminal is gone; the server is not
        time.sleep(0.5)
        u = urlsplit(base)
        st, out = _post(u.port, u.path, "open-folder", {"path": str(b)})
        assert st == 200 and out["root"] == str(b.resolve()), out
        st, out = _post(u.port, u.path, "open-folder", {"path": str(a)})
        assert st == 200 and out["root"] == str(a.resolve()), out
        assert p.poll() is None
    finally:
        p.terminate()
        p.wait(10)
