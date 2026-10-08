"""File > Open Folder through the real `mace_cli gui` subprocess and HTTP, with
a fake HOME: allowed roots, symlink escape, home and credential folders,
hidden folders (targets, pasted paths, writes), credential paths refused for
reads even when serving the home folder, --root none, 409 while a task runs,
the switch itself (files, old paths, dir_hash, the stale-tab 409), and a
hammer test: no request mixes two roots."""
import http.client
import json
import os
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pytest

import guiharness as H

REPO = Path(__file__).resolve().parents[2]


class Srv:
    def __init__(self, tmp, served, *extra, **env):
        e = H.env(tmp / "xdg", HOME=str(tmp / "home"), MACE_GUI_NO_WARMUP="1", **env)
        e.pop("MACE_GUI_ROOTS", None) if "MACE_GUI_ROOTS" not in env else None
        import re
        import subprocess
        import sys
        self.p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(served),
                                  "--port", "0", "--token", H.TOKEN, *extra],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=e, start_new_session=True, preexec_fn=H.die_with_pytest)
        self.lines = []
        for _ in range(40):
            line = self.p.stdout.readline()
            self.lines.append(line)
            m = re.search(H.URL_RE, line)
            if m:
                self.base = m.group(1)
                break
        else:
            raise AssertionError("".join(self.lines))
        threading.Thread(target=lambda: [self.lines.append(x) for x in self.p.stdout], daemon=True).start()
        from urllib.parse import urlsplit
        u = urlsplit(self.base)
        self.port, self.prefix = u.port, u.path

    def req(self, route, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui", "Content-Type": "application/json",
             **(headers or {})}
        c.request("POST" if body is not None else "GET", self.prefix + route,
                  body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, data

    def stop(self):
        self.p.terminate()
        self.p.wait(15)


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".ssh").mkdir(parents=True)
    (h / ".ssh" / "id_x").write_text("PRIVATE KEY")
    (h / ".config" / "gh").mkdir(parents=True)
    (h / ".config" / "gh" / "hosts.yml").write_text("token: zz")
    (h / ".kube").mkdir()
    (h / ".local" / "share" / "x").mkdir(parents=True)
    (h / ".netrc").write_text("machine x")
    for name in ("projA", "projB", "projA/.git"):
        (h / name).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "examples" / "gui" / "structures" / "MgO.cif", h / "projA" / "MgO.cif")
    (h / "projB" / "sub").mkdir()
    shutil.copy(REPO / "examples" / "gui" / "structures" / "diamond.cif", h / "projB" / "diamond.cif")
    return h


def test_switch_lists_the_new_folder_and_refuses_the_stale_tab(tmp_path, home):
    s = Srv(tmp_path, home / "projA")
    try:
        st, info = s.req("api/info")
        assert info["open_folder"] is True and info["root_gen"] == 0
        assert [r["label"] for r in info["roots"]] == ["Home"]
        old_hash = info["dir_hash"]
        st, f = s.req("api/folders?path=" + quote(str(home)))
        assert st == 200 and f["dirs"] == ["projA", "projB"] and f["parent"] is None
        st, r = s.req("api/open-folder", {"path": str(home / "projB")})
        assert st == 200 and r["root"] == str((home / "projB").resolve()) and r["gen"] == 1
        assert r["dir_hash"] != old_hash
        st, files = s.req("api/files")
        assert [x["path"] for x in files["files"]] == ["diamond.cif"]
        assert s.req("api/text?path=MgO.cif")[0] == 404
        st, info = s.req("api/info")
        assert info["root_gen"] == 1 and info["dir_hash"] == r["dir_hash"]
        # the tunnel dialog's connect line follows the folder served now
        assert info["connect"].startswith("mace gui connect --gateway ")
        assert info["connect"].endswith(f" --dir {(home / 'projB').resolve()}"), info["connect"]
        # a tab still on generation 0 cannot write into the new folder
        st, body = s.req("api/deck/save", {"name": "x.json", "mode": "d12", "data": {}},
                         headers={"X-Mace-Root-Gen": "0"})
        assert st == 409 and "folder changed in another tab" in body["error"]
        assert any("now serving" in ln and "(Open Folder)" in ln for ln in s.lines)
    finally:
        s.stop()


@pytest.mark.parametrize("target,why", [
    ("{home}/.config", "Hidden folders cannot be opened here"),
    ("{home}/.config/gh", "Hidden folders cannot be opened here"),
    ("{home}/.kube", "Hidden folders cannot be opened here"),
    ("{home}/projA/.git", "Hidden folders cannot be opened here"),
    ("{home}/.local/share/x", "Hidden folders cannot be opened here"),
    ("{home}", "Open a project folder inside your home, not your home folder itself."),
    ("{tmp}/outside", "outside the folders this server may open (Home)"),
    ("{home}/projA/escape", "outside the folders this server may open"),
])
def test_refused_targets(tmp_path, home, target, why):
    (tmp_path / "outside").mkdir()
    (home / "projA" / "escape").symlink_to(tmp_path / "outside")
    s = Srv(tmp_path, home / "projB")
    try:
        st, body = s.req("api/open-folder", {"path": target.format(home=home, tmp=tmp_path)})
        assert st == 403 and why in body["error"], body
        st, info = s.req("api/info")
        assert info["root_gen"] == 0
    finally:
        s.stop()


def test_pasted_dot_path_in_the_folder_list_is_refused(tmp_path, home):
    s = Srv(tmp_path, home / "projB")
    try:
        st, body = s.req("api/folders?path=" + quote(str(home / ".config")))
        assert st == 403 and "Hidden folders" in body["error"]
        st, f = s.req("api/folders?path=" + quote(str(home)))
        assert ".ssh" not in f["dirs"] and ".config" not in f["dirs"]
    finally:
        s.stop()


def test_credential_paths_refused_even_when_serving_home(tmp_path, home):
    s = Srv(tmp_path, home)
    try:
        for rel in (".ssh/id_x", ".config/gh/hosts.yml", ".netrc", "projA/../.ssh/id_x"):
            st, body = s.req("api/text?path=" + quote(rel))
            assert st == 403 and "credentials" in body["error"], (rel, body)
        st, files = s.req("api/files")
        assert not [f for f in files["files"] if f["path"].startswith(".")]
        # writes into hidden folders: refused (Save As of a structure, which needs ase)
        pytest.importorskip("ase")
        st, opened = s.req("api/structure/save", {"open": {"path": "projA/MgO.cif"}, "path": ".hidden/x.cif",
                                                   "format": "cif"})
        (home / ".hidden").mkdir()
        st, opened = s.req("api/structure/save", {"open": {"path": "projA/MgO.cif"}, "path": ".hidden/x.cif",
                                                   "format": "cif"})
        assert st == 403 and "hidden" in str(opened), opened
        assert not (home / ".hidden" / "x.cif").exists()
    finally:
        s.stop()


SECRETS = {".pypirc": "[pypi]\npassword = s3cret", ".npmrc": "//registry/:_authToken=s3cret",
           ".git-credentials": "https://u:s3cret@host", ".env": "API_KEY=s3cret",
           ".env.local": "API_KEY=s3cret", ".bash_history": "export TOKEN=s3cret",
           ".zsh_history": "s3cret", ".python_history": "s3cret", ".vault-token": "s3cret",
           ".docker/config.json": '{"auths": "s3cret"}', ".kube/config": "token: s3cret",
           ".aws/credentials": "s3cret", ".azure/accessTokens.json": "s3cret",
           ".config/gcloud/credentials.db": "s3cret", ".gnupg/secring.gpg": "s3cret",
           "projA/.env": "API_KEY=s3cret", "projA/.npmrc": "s3cret",
           "projA/keys/id_ed25519": "-----BEGIN OPENSSH PRIVATE KEY-----", "id_rsa": "s3cret"}


def test_secret_files_hidden_for_safety_when_serving_home(tmp_path, home):
    """Every secret file is refused as "hidden for safety" by the read, the
    raw download, the file info and the docprefs routes, through a symlink
    too; none is listed; an ordinary file next to them still opens."""
    for rel, text in SECRETS.items():
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(text)
    (home / "projA" / "notes.cif").symlink_to(home / ".bash_history")
    (home / "projA" / "run_history").mkdir()
    shutil.copy(home / "projA" / "MgO.cif", home / "projA" / "run_history" / "MgO.cif")
    s = Srv(tmp_path, home)
    try:
        for rel in list(SECRETS) + ["projA/notes.cif", "projA/../.env.local"]:
            for route in ("api/text?path=", "api/raw?path=", "api/fileinfo?path=", "api/text/lines?path="):
                st, body = s.req(route + quote(rel))
                assert st == 403 and "hidden for safety" in body["error"], (route, rel, body)
                assert "s3cret" not in json.dumps(body)
        st, files = s.req("api/files?details=1")
        listed = {f["path"] for f in files["files"]}
        assert not [p for p in listed if "s3cret" in p or p.startswith(".") or "id_" in p or "/." in p]
        assert {"projA/MgO.cif", "projA/run_history/MgO.cif"} <= listed
        st, body = s.req("api/text?path=" + quote("projA/MgO.cif"))
        assert st == 200 and "s3cret" not in json.dumps(body)
        st, body = s.req("api/text?path=" + quote("projA/run_history/MgO.cif"))
        assert st == 200
    finally:
        s.stop()


def test_root_none_turns_it_off(tmp_path, home):
    s = Srv(tmp_path, home / "projA", "--root", "none")
    try:
        st, info = s.req("api/info")
        assert info["open_folder"] is False and info["roots"] == []
        st, body = s.req("api/open-folder", {"path": str(home / "projB")})
        assert st == 403 and "turned off" in body["error"]
    finally:
        s.stop()
    s = Srv(tmp_path, home / "projA", MACE_GUI_ROOTS="")
    try:
        assert s.req("api/info")[1]["open_folder"] is False
    finally:
        s.stop()


def test_root_flag_replaces_the_defaults_and_dir_is_always_allowed(tmp_path, home):
    other = tmp_path / "data"
    (other / "x").mkdir(parents=True)
    s = Srv(tmp_path, home / "projA", "--root", str(other))
    try:
        st, info = s.req("api/info")
        assert [r["path"] for r in info["roots"]] == [str(other.resolve())]
        assert s.req("api/open-folder", {"path": str(home / "projB")})[0] == 403
        assert s.req("api/open-folder", {"path": str(other / "x")})[0] == 200
        # back to the start folder: always allowed
        assert s.req("api/open-folder", {"path": str(home / "projA")})[0] == 200
    finally:
        s.stop()


def test_409_while_a_task_runs(tmp_path, home):
    s = Srv(tmp_path, home / "projA", MACE_GUI_TEST_HOOKS="1")
    try:
        assert s.req("api/test/task", {"seconds": 6, "job": "sleep", "wait": 0})[0] == 202
        st, body = s.req("api/open-folder", {"path": str(home / "projB")})
        assert st == 409 and "Cancel or wait for the running tasks first" in body["error"]
    finally:
        s.stop()


def test_no_request_mixes_two_roots(tmp_path, home):
    """50 switches while other threads list files: every listing is one folder's."""
    s = Srv(tmp_path, home / "projA")
    want = {frozenset(["MgO.cif"]), frozenset(["diamond.cif"])}
    bad, stop = [], threading.Event()

    def hammer():
        while not stop.is_set():
            st, files = s.req("api/files")
            if st == 200:
                got = frozenset(x["path"] for x in files["files"])
                root = files["root"]
                if got not in want or (("MgO.cif" in got) != root.endswith("projA")):
                    bad.append((root, sorted(got)))
    threads = [threading.Thread(target=hammer, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        for i in range(50):
            st, _ = s.req("api/open-folder", {"path": str(home / ("projB" if i % 2 == 0 else "projA"))})
            assert st == 200
        stop.set()
        for t in threads:
            t.join(10)
        assert not bad, bad[:3]
    finally:
        stop.set()
        s.stop()
