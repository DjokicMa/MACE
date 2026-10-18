"""Settings › Advanced: cluster defaults (1.1.6 parity A2): the resolved site
settings with where each comes from, as `mace site` prints them, and edits
made by running `mace site set KEY VALUE`, `mace site unset KEY` and
`mace site --init` (the CLI itself). Every edit needs the page's root_gen,
and none is made while MACE GUI is open to the network. The site.toml is a
private one (MACE_SITE_CONFIG): never the user's."""
import re
import subprocess
import sys

import pytest

import submit_util as SU


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("site_defaults")
    s = SU.Served(tmp, MACE_SITE_CONFIG=str(tmp / "cfg" / "site.toml"))
    s.site = tmp / "cfg" / "site.toml"
    yield s
    s.stop()


def _gen(s):
    return _settings(s)["root_gen"]          # what the page was given


def _settings(s):
    st, _, r = s.req("api/site/settings")
    assert st == 200, r
    return r


def _edit(s, **body):
    st, _, r = s.req("api/site/edit", {"root_gen": _gen(s), **body})
    return st, r


def _twin(s, *args):
    """`mace site ...` typed in a shell with the server's environment."""
    import os
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH", "COLORTERM", "TERM", "NO_COLOR")}
    env.update(s.env)
    return subprocess.run([sys.executable, str(SU.MACE_CLI), "--no-banner", "site", *args], cwd=str(s.root), env=env,
                          capture_output=True, text=True, timeout=300)


def test_shows_the_settings_with_their_origins_as_mace_site_does(srv):
    r = _settings(srv)
    assert r["command"] == "mace site" and r["path"] == str(srv.site) and r["exists"] is False
    assert not r["network"] and r["root_gen"] is not None
    twin = _twin(srv).stdout
    # the same lines as the CLI prints (a few are live answers and may differ between two calls)
    for line in ("built-in", str(srv.site)):
        assert line in r["text"] and line in twin, line
    same = [l for l in twin.splitlines() if "built-in" in l]
    assert same and all(l in r["text"].splitlines() for l in same[:5]), (same[:5], r["text"])
    assert "resources.ntasks" in r["keys"] and "resources.OPT.walltime" in r["keys"] and "slurm.account" in r["keys"]


def test_edits_run_the_cli_and_need_root_gen(srv):
    st, _, r = srv.req("api/site/edit", {"action": "init"})
    assert st == 409 and "reload" in r["error"], r
    assert not srv.site.exists()
    st, _, r = srv.req("api/site/edit", {"action": "init", "root_gen": -5})
    assert st == 409 and not srv.site.exists()
    # --init: the CLI's file, its words
    st, r = _edit(srv, action="init")
    assert st == 200 and r["ok"] and r["command"] == "mace site --init", r
    assert srv.site.is_file() and "Wrote" in r["output"]
    assert r["settings"]["exists"] is True
    st, r = _edit(srv, action="init")
    assert st == 200 and not r["ok"] and "already exists" in r["output"], r
    # set: the value as written, the origin shown afterwards
    st, r = _edit(srv, action="set", key="resources.ntasks", value="16")
    assert st == 200 and r["ok"] and r["command"] == "mace site set resources.ntasks 16", r
    assert re.search(r"^\s*ntasks\s*=\s*16", srv.site.read_text(), re.M)
    assert "site.toml" in r["settings"]["text"] and "16" in r["settings"]["text"]
    # a value that starts with -- is taken as written (mace site set slurm.extra --exclusive)
    st, r = _edit(srv, action="set", key="slurm.extra", value="--exclusive")
    assert st == 200 and r["ok"], r
    assert "--exclusive" in srv.site.read_text()
    # the CLI's own refusal, nothing written
    before = srv.site.read_text()
    st, r = _edit(srv, action="set", key="bogus.key", value="1")
    assert st == 200 and not r["ok"] and "'bogus.key' is not a site setting" in r["output"], r
    assert srv.site.read_text() == before
    st, r = _edit(srv, action="unset", key="resources.ntasks")
    assert st == 200 and r["ok"] and r["command"] == "mace site unset resources.ntasks", r
    assert not re.search(r"^\s*ntasks\s*=", srv.site.read_text(), re.M)
    st, r = _edit(srv, action="unset", key="slurm.extra")
    assert st == 200 and r["ok"], r
    # the Submit dialog's Site row sees the change at once (its cache is dropped)
    st, _, caps = srv.req("api/submit/capabilities")
    assert st == 200 and caps["site"]["config"]["exists"] is True


@pytest.mark.parametrize("body,match", [
    ({"action": "remove"}, "action is set, unset or init"),
    ({"action": "set", "key": "resources.ntasks"}, "a key and a value"),
    ({"action": "unset", "key": ""}, "a key"),
    ({"action": "set", "key": "x" * 300, "value": "1"}, "a key"),
])
def test_bad_requests_are_named(srv, body, match):
    st, r = _edit(srv, **body)
    assert st == 400 and match in r["error"], r


def test_refused_while_open_to_the_network(tmp_path):
    import re as _re
    s = SU.Served.__new__(SU.Served)
    s.tmp, s.root = tmp_path, tmp_path / "proj"
    s.root.mkdir()
    s.fake = SU.FakeSlurm(tmp_path / "fs")
    s.env = SU.fake_env(s.fake, tmp_path)
    s.env.pop("XDG_CACHE_HOME")
    site = tmp_path / "cfg" / "site.toml"
    port = SU.H.free_port()
    env = SU.H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG=str(site), **s.env)
    proc = SU.H.ServerProcess([sys.executable, str(SU.MACE_CLI), "--no-banner", "gui", "--dir", str(s.root), "--port",
                               str(port), "--token", SU.H.TOKEN, "--host", "0.0.0.0", "--insecure-network",
                               "--allow-host", "127.0.0.1"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    try:
        base = None
        for _ in range(60):
            m = _re.search(r"(http://[^/\s]+:\d+/s/[A-Za-z0-9_-]+/)", proc.stdout.readline())
            if m:
                base = m.group(1).replace("0.0.0.0", "127.0.0.1")
                break
        assert base
        s.base, s.proc = base, proc
        r = _settings(s)
        assert r["network"] is True and r["text"]
        for body in ({"action": "init"}, {"action": "set", "key": "resources.ntasks", "value": "4"}):
            st, _, res = s.req("api/site/edit", {"root_gen": r["root_gen"], **body})
            assert st == 409 and "turned off while MACE GUI is open to the network" in res["error"], res
        assert not site.exists()
    finally:
        proc.terminate()
        proc.wait(timeout=30)
