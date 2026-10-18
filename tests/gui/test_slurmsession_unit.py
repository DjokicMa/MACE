"""Unit tests of `mace gui --slurm` (slurmsession.py) and its server side:
the job script and sbatch argv (quoting, one argv element per value, the
2 CPU / 4G defaults, --propagate=NONE, no token), where each value comes
from (flags, the site settings, never a built-in value), refused options,
parsing (`--parsable`, node lists), scancel only with the parsed id, every
SLURM read through the shared query layer, the GPU default inside a job,
the --http-control watchdog, the idle-stop default in a job and connect's
remote command."""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from mace.gui import connect, managed, resources, slurmsession as ss
from mace.gui.sessions import Sessions


def _args(**kw):
    a = ss.build_parser().parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def no_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", str(tmp_path / "nosite.toml"))
    monkeypatch.setenv("MACE_SITE_GROUP_CONFIG", "")
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "noxdg"))
    for k in ("MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION", "MACE_SLURM_QOS", "MACE_SLURM_CONSTRAINT"):
        monkeypatch.delenv(k, raising=False)


def test_defaults_and_one_argv_element_per_value(no_site, tmp_path):
    o = ss.resolve_options(_args())
    assert (o["time"], o["cpus"], o["mem"]) == ("4:00:00", 2, "4G")
    argv = ss.sbatch_argv(o, tmp_path / "b", tmp_path / "b" / "job.sh")
    assert argv[:2] == ["sbatch", "--parsable"]
    assert "--cpus-per-task=2" in argv and "--mem=4G" in argv and "--propagate=NONE" in argv
    assert "--job-name=mace-gui" in argv and "--time=4:00:00" in argv
    assert not any(a.startswith(("--partition", "--account", "--qos", "--constraint")) for a in argv)
    assert all(" " not in a for a in argv[:-1])


def test_site_settings_fill_in_but_never_a_built_in_value(no_site, monkeypatch, tmp_path):
    o = ss.resolve_options(_args())
    assert o["account"] is None              # the built-in account belongs to one cluster's group
    site = tmp_path / "site.toml"
    site.write_text('[slurm]\naccount = "grp_x"\npartition = "short"\n')
    monkeypatch.setenv("MACE_SITE_CONFIG", str(site))
    o = ss.resolve_options(_args())
    assert o["account"] == "grp_x" and o["partition"] == "short"
    assert o["sources"]["account"].startswith("site settings")
    o = ss.resolve_options(_args(partition="other"))
    assert o["partition"] == "other" and o["sources"]["partition"] == "flag"


def test_job_script_quotes_the_folder_and_holds_no_token(no_site, tmp_path):
    root = tmp_path / "my runs's $HOME `x`"
    root.mkdir()
    o = ss.resolve_options(_args(setup=["module load Python"]))
    text = ss.job_script(o, root, tmp_path / "base dir", roots=["/a b"])
    assert f"cd -- {shlex.quote(str(root))} || exit 2" in text
    assert "module load Python" in text
    assert "--http-control" in text and "--port 0" in text and "--managed" in text
    assert shlex.quote(str(tmp_path / "base dir")) + '/"$SLURM_JOB_ID"/ready.json' in text
    assert "--root " + shlex.quote("/a b") in text
    assert "token" not in text.lower()
    assert 'export MACE_GUI_SLURM="$SLURM_JOB_ID"' in text
    # the exec line parses as one command whose words are the intended ones
    words = shlex.split(text.strip().splitlines()[-1])
    assert words[0] == "exec" and words[1] == sys.executable


@pytest.mark.parametrize("kw, word", [
    ({"time": "4 hours"}, "--time"), ({"cpus": 0}, "--cpus"), ({"mem": "4 GB"}, "--mem"),
    ({"partition": "a;b"}, "--partition"), ({"account": "x y"}, "--account"),
    ({"constraint": "a`b`"}, "--constraint"), ({"gpus": "1 2"}, "--gpus"),
    ({"sbatch_arg": ["--output=/tmp/x"]}, "--output"), ({"sbatch_arg": ["--wrap=id"]}, "--wrap"),
    ({"sbatch_arg": ["-J"]}, "-J"), ({"sbatch_arg": ["--reservation=a b"]}, "--sbatch-arg"),
])
def test_bad_values_are_refused(no_site, kw, word):
    with pytest.raises(ss.SlurmError) as e:
        ss.resolve_options(_args(**kw))
    assert word in str(e.value)


def test_extra_sbatch_args_pass_through(no_site, tmp_path):
    o = ss.resolve_options(_args(sbatch_arg=["--reservation=r1", "--exclusive"], gpus="a100:1"))
    argv = ss.sbatch_argv(o, tmp_path, tmp_path / "j.sh")
    assert "--reservation=r1" in argv and "--exclusive" in argv and "--gpus=a100:1" in argv


@pytest.mark.parametrize("text, want", [("123\n", "123"), ("123;cluster2\n", "123"), ("", None),
                                        ("Submitted batch job 5", None)])
def test_parsable(text, want):
    assert ss.parse_parsable(text) == want


@pytest.mark.parametrize("nl, want", [("node01", "node01"), ("amr-[001-003,007]", "amr-001"),
                                      ("a,b", "a"), ("(null)", None), ("", None)])
def test_first_node(nl, want):
    assert ss.first_node(nl) == want


def test_scancel_only_the_parsed_id(monkeypatch):
    seen = []
    monkeypatch.setattr(ss.subprocess, "run", lambda argv, **kw: seen.append(argv) or
                        type("R", (), {"returncode": 0})())
    assert ss.scancel("5000") is True
    assert ss.scancel("5000;cluster") is False and ss.scancel("-u me") is False and ss.scancel("") is False
    assert seen == [["scancel", "5000"]]


def test_every_slurm_read_goes_through_the_query_layer(monkeypatch):
    from mace.cluster import query
    keys = []

    def spy(key, argv, **kw):
        keys.append((key, list(argv)))
        if key == "ownjobs":
            out = "5001|RUNNING|None|N/A|1:00|4:00:00|0:10|1|2|gen|acct|n7|/w|5001|N/A|mace-gui\n"
        else:
            out = "JobId=5001 JobState=RUNNING Reason=None StartTime=2030-01-01T00:00:00 EndTime=2030-01-01T04:00:00 NodeList=n7\n"
        proc = query.Proc(0, out, "")
        return query.Result(value=kw["parse"](proc) if kw.get("parse") else out, source="cli", key=key)
    monkeypatch.setattr(query, "run", spy)

    def forbidden(argv, *a, **kw):
        if argv and argv[0] in ("squeue", "scontrol", "sacct"):
            raise AssertionError(f"direct SLURM call {argv}")
        raise AssertionError(f"unexpected subprocess {argv}")
    monkeypatch.setattr(ss.subprocess, "run", forbidden)
    jobs = ss.own_gui_jobs()
    assert jobs and jobs[0]["id"] == "5001" and jobs[0]["nodelist"] == "n7"
    info = ss.job_state("5001")
    assert info["JobState"] == "RUNNING" and info["NodeList"] == "n7"
    assert [k for k, _ in keys] == ["ownjobs", "job:5001"]
    assert keys[1][1] == ["scontrol", "show", "job", "-o", "5001"]


@pytest.mark.parametrize("cvd, want", [(None, "none"), ("", "none"), ("3,5", "0")])
def test_gpu_default_inside_a_job(cvd, want):
    env = {"SLURM_JOB_ID": "77"}
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    assert resources.gpu_device(None, env) == want
    assert resources.gpu_device(None, {}) == "0"                    # outside a job: GPU 0
    assert resources.gpu_device("2", env) == "2"                     # a stored choice wins
    assert resources.gpu_device(None, {**env, "MACE_GUI_GPU": "1"}) == "1"


def test_idle_stop_defaults_to_one_hour_in_a_job(monkeypatch):
    monkeypatch.delenv("MACE_GUI_IDLE_STOP_HOURS", raising=False)
    monkeypatch.delenv("MACE_GUI_SLURM", raising=False)
    assert Sessions("t" * 20, lambda: {}).policy()["idle_stop_hours"] is None
    monkeypatch.setenv("MACE_GUI_SLURM", "5000")
    assert Sessions("t" * 20, lambda: {}).policy()["idle_stop_hours"] == 1.0
    assert Sessions("t" * 20, lambda: {"server.idleStopHours": 3}).policy()["idle_stop_hours"] == 3


def test_http_control_watchdog():
    stops = []
    c = managed.Controller(running=lambda: 0, stop=stops.append, timeout_s=0.3, grace_s=0.3)
    c.http_control = True
    c.attach("w1")
    assert c.connected is False and c.watchers == 1
    c.tick()
    assert not stops
    time.sleep(0.4)                       # the watcher stopped pinging: a disconnect, no task -> stop
    c.tick()
    assert stops == ["managed-disconnect"]
    # nobody attaches within the timeout of READY: stop
    stops2 = []
    c2 = managed.Controller(running=lambda: 0, stop=stops2.append, timeout_s=0.2, grace_s=10)
    c2.http_control = True
    time.sleep(0.3)
    c2.tick()
    assert stops2 == ["managed-disconnect"]
    # a detach with a task running lingers instead
    stops3 = []
    c3 = managed.Controller(running=lambda: 1, stop=stops3.append, timeout_s=5, grace_s=5)
    c3.http_control = True
    c3.attach("w")
    c3.detach("w")
    assert not stops3 and c3.lingering_since is not None


def test_a_ping_from_an_unknown_watcher_attaches_it():
    """After Update Now the new process starts with no watchers; the slurmsession
    client only pings (it attached once, before the restart), so a ping of an
    unknown watcher must attach it, or the watchdog stops the job's server."""
    stops = []
    c = managed.Controller(running=lambda: 0, stop=stops.append, timeout_s=0.3, grace_s=10)
    c.http_control = True
    got = c.ping("w9")
    assert got["ok"] is True and c.watchers == 1 and c.connected is False
    for _ in range(5):
        time.sleep(0.1)
        c.ping("w9")
        c.tick()
    assert not stops
    assert c.ping("")["ok"] is True and c.watchers == 1          # an empty id never attaches


def test_controller_hand_off_round_trip():
    """The restart hand-off carries the watchers and whether the first connection
    is gone; a lingering server comes back lingering."""
    c = managed.Controller(running=lambda: 0, stop=lambda r: None, timeout_s=5, grace_s=5)
    c.attach("w1")
    c.connected = False
    c.lingering_since = time.monotonic() - 100
    saved = json.loads(json.dumps(c.handoff()))
    assert saved["connected"] is False and saved["watchers"] == ["w1"] and saved["lingering_for"] >= 100
    stops = []
    n = managed.Controller(running=lambda: 0, stop=stops.append, timeout_s=0.3, grace_s=10)
    n.restore(saved)
    assert n.connected is False and n.watchers == 1 and n.lingering_since is not None
    assert time.monotonic() - n.lingering_since >= 100
    n.ping("w1")
    n.tick()
    assert not stops
    n.restore({"watchers": "bad", "connected": "x"})                # junk is ignored
    n.restore(None)
    assert n.watchers == 1 and n.connected is False


def test_connect_remote_command_for_slurm():
    prof = {"gateway": "me@login", "node": "dev1", "dir": "~/runs", "slurm": True, "slurm_time": "2:00:00",
            "slurm_partition": "short", "slurm_args": ["--reservation=r 1"]}
    cmd = connect.remote_command(prof, 41000)
    inner = shlex.split(shlex.split(cmd)[3])
    # the default mace: looked up first (mace_fallback), then exec'd as "$M"
    words = inner[inner.index("exec"):]
    assert words[:4] == ["exec", "$M", "--no-banner", "gui"] and words[4] == "--slurm"
    assert inner[inner.index("&&") + 1:inner.index("&&") + 3] == ["{", "M=mace;"]
    assert words[words.index("--time") + 1] == "2:00:00" and words[words.index("--partition") + 1] == "short"
    assert "--sbatch-arg=--reservation=r 1" in words
    assert words[-6:] == ["--managed", "--port", "41000", "--dir", ".;", "}"]
    assert "MACE_GUI_MAX_WORKERS=2" not in cmd
    plain = connect.remote_command({"gateway": "me@login", "dir": "~"}, 41000)
    assert "--slurm" not in plain


def test_connect_refuses_slurm_options_without_slurm(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    rc = connect.main(["--gateway", "g", "--time", "1:00:00", "--dry-run"])
    assert rc == 2 and "go with --slurm" in capsys.readouterr().err
    rc = connect.main(["--gateway", "g", "--slurm", "--time", "1:00:00", "--dry-run"])
    assert rc == 0 and "--slurm --time 1:00:00" in capsys.readouterr().out


def _fake_ssh(tmp_path, monkeypatch, stderr):
    """An "ssh" that fails at once with this stderr (as OpenSSH does)."""
    f = tmp_path / "ssh_fails.sh"
    f.write_text("#!/bin/bash\ncat >&2 <<'X'\n" + stderr + "\nX\nexit 255\n")
    f.chmod(0o755)
    monkeypatch.setenv("MACE_GUI_SLURM_SSH", str(f))


@pytest.mark.parametrize("stderr, kind", [
    ("bind [127.0.0.1]:40123: Address already in use\nchannel_setup_fwd_listener_tcpip: cannot listen to port: 40123\n"
     "Could not request local forwarding.", "port-busy"),
    ("Access denied: user tester (uid=5) has no active jobs on this node.\nAuthentication failed.", "ssh-node"),
])
def test_a_busy_port_is_not_reported_as_a_node_that_refuses_ssh(tmp_path, monkeypatch, stderr, kind):
    _fake_ssh(tmp_path, monkeypatch, stderr)
    with pytest.raises(ss.SlurmError) as e:
        ss.open_tunnel("node01", 40123, {"port": 5555, "prefix": "/p/", "token": "t"}, wait_s=5)
    assert e.value.kind == kind
    assert ("pam_slurm_adopt" in str(e.value)) == (kind == "ssh-node")


def test_a_busy_port_given_by_connect_stops_before_anything_is_submitted(tmp_path, monkeypatch, capsys):
    import socket
    calls = []
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert ss.port_free(port) is False
        code = ss.main(["--managed", "--port", str(port), "--dir", str(tmp_path)])
    assert code == 98 and not calls
    out = capsys.readouterr().out
    assert 'MACE_GUI_ERROR {"kind": "port-busy"' in out


def test_connect_takes_another_port_when_the_slurm_side_says_busy():
    import queue as q

    class FakeSess:
        events = q.Queue()
        ended = None

        def end(self, wait=0):
            FakeSess.ended = wait
    s = FakeSess()
    s.events.put(("MACE_GUI_JOB", {"job": "5000"}))
    s.events.put(("MACE_GUI_ERROR", {"kind": "port-busy", "message": "Port 40123 is busy on this node."}))
    said = []
    out = connect._drive(s, {}, argparse.Namespace(no_open=True), 8899, "gw", said.append, None)
    assert out[0] == "port-busy" and FakeSess.ended is not None
    assert not any("pam_slurm_adopt" in x for x in said)
