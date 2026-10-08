"""`mace gui --slurm` end to end on a fake cluster: the real mace_cli submits
the GUI job to FakeSlurm, which keeps it pending, then runs the job script in
the background (the real `mace gui --managed --http-control` server, started
by the script); a fake ssh forwards the port to the "node". Checked: the
protocol lines, the server answering through the forward with its token, a
clean stop that ends the job (its allocation released), Ctrl+C/EOF while
pending cancels exactly that job, a refused submission, a node that refuses
ssh (the job is canceled), the idle stop and the time-limit stop, and that
`scontrol show job` is asked at most once per 15 s while pending."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fake_slurm import FakeSlurm  # noqa: E402

import guiharness as H  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAKE_SSH = Path(__file__).with_name("fake_ssh.py")
EXAMPLE = H.EXAMPLES / "structures" / "MgO.cif"


@pytest.fixture
def cluster(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    root = tmp_path / "proj"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "MgO.cif")
    fake = FakeSlurm(tmp_path / "fs", {"run_batch": {"pending_s": 3}})
    os.chmod(FAKE_SSH, 0o755)
    env = {k: v for k, v in fake.env.items()
           if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM", "NO_COLOR", "BROWSER", "SSH_CONNECTION",
                        "MACE_GUI_TOKEN", "MACE_GUI_PORT")}
    env.update(HOME=str(tmp_path / "home"), XDG_CACHE_HOME=str(tmp_path / "cache"),
               XDG_CONFIG_HOME=str(tmp_path / "cfg"), MACE_GUI_SLURM_SSH=f"{sys.executable} {FAKE_SSH}",
               MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               MACE_MODULES_NO_PROFILE="1", USER=os.environ.get("USER", "tester"))
    (tmp_path / "home").mkdir()
    procs = []
    yield {"tmp": tmp_path, "root": root, "fake": fake, "env": env, "procs": procs}
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(10)
    for r in fake.records():                     # nothing may stay "running" after a test
        if r.get("pgid"):
            try:
                os.kill(int(r["pgid"]), 9)
            except OSError:
                pass
    subprocess.run(["pkill", "-f", str(tmp_path / "fs")], capture_output=True)


class Slurm:
    """`mace gui --slurm --managed` as mace gui connect starts it (stdin = control lines)."""

    def __init__(self, c, *extra, env=None):
        self.port = H.free_port()
        self.p = subprocess.Popen([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "--slurm",
                                   "--managed", "--port", str(self.port), "--dir", str(c["root"]), *extra],
                                  env={**c["env"], **(env or {})}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        c["procs"].append(self.p)
        self.events = []
        self.err = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: [self.err.append(x) for x in self.p.stderr], daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            tag, _, rest = line.strip().partition(" ")
            try:
                data = json.loads(rest) if rest.startswith("{") else rest
            except ValueError:
                data = rest
            self.events.append((tag, data))

    def wait(self, tag, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            for t, d in self.events:
                if t == tag:
                    return d
            if self.p.poll() is not None and not any(t == tag for t, _ in self.events):
                time.sleep(0.5)
                for t, d in self.events:
                    if t == tag:
                        return d
                raise AssertionError(f"exited {self.p.returncode} before {tag}: {self.events} {''.join(self.err)}")
            time.sleep(0.2)
        raise AssertionError(f"no {tag} within {timeout} s: {self.events} {''.join(self.err)}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()


def _sigint_default():
    """In the child: Ctrl+C as in a terminal. A suite started in the background
    (`pytest ... &`) hands its children SIGINT ignored, and Python then never
    raises KeyboardInterrupt; the tests that press Ctrl+C would hang."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _info(port, prefix, token):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{prefix}api/info",
                                 headers={"X-Mace-Token": token, "X-Requested-With": "mace-gui"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as r:
        return json.loads(r.read())


def _job_state(fake, job):
    return (fake.load().get("job_text") or {}).get(job, "")


def _wait_gone(fake, job, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if not any(r["id"] == job for r in fake.records()):
            return
        time.sleep(0.3)
    raise AssertionError(f"job {job} still queued: {fake.records()}")


def test_runs_as_a_job_and_a_clean_stop_releases_it(cluster):
    s = Slurm(cluster)
    job = s.wait("MACE_GUI_JOB")["job"]
    pend = s.wait("MACE_GUI_PENDING")
    assert pend["job"] == job and pend["reason"] == "Priority"
    ready = s.wait("MACE_GUI_READY")
    assert ready["job"] == job and ready["node"] == "localhost" and ready["port"] == s.port
    assert ready["login"] and "?login=" in ready["login"] and ready["ends"]
    info = _info(s.port, ready["prefix"], ready["token"])
    assert info["slurm"]["job"] == job and Path(info["root"]).resolve() == cluster["root"].resolve()
    # the request: one argv element each, the defaults, no token anywhere in the script
    call = [c for c in cluster["fake"].calls("sbatch")][0]
    argv = call["argv"]
    for want in ("--parsable", "--job-name=mace-gui", "--time=4:00:00", "--cpus-per-task=2", "--mem=4G",
                 "--propagate=NONE", "--ntasks=1"):
        assert want in argv, argv
    assert ready["token"] not in call["script_text"]
    # Stop: the server stops, the job leaves the queue (its allocation is released)
    s.send("stop")
    stopped = s.wait("MACE_GUI_STOPPED", 60)
    assert stopped["reason"] == "managed-stop"
    _wait_gone(cluster["fake"], job)
    assert "JobState=COMPLETED" in _job_state(cluster["fake"], job)
    assert s.p.wait(30) == 0
    # never scanceled: the job ended because its server stopped
    assert not cluster["fake"].calls("scancel")
    # the ready file (it holds the token) is gone; stopped.json says why
    jobdir = cluster["tmp"] / "cache" / "mace" / "gui" / "slurm" / job
    assert not (jobdir / "ready.json").exists()
    assert json.loads((jobdir / "stopped.json").read_text())["reason"] == "managed-stop"


def test_scontrol_at_most_once_per_15_s_while_pending(cluster):
    cluster["fake"].update(run_batch={"pending_s": 40})
    s = Slurm(cluster)
    job = s.wait("MACE_GUI_JOB")["job"]
    s.wait("MACE_GUI_PENDING")
    time.sleep(35)
    calls = [c for c in cluster["fake"].calls("scontrol") if "job" in c["argv"]]
    assert 1 <= len(calls) <= 3, [c["argv"] for c in calls]          # 35 s / 15 s floor
    # stdin EOF (the connection closed) while pending: exactly this job is canceled
    s.p.stdin.close()
    assert s.p.wait(60) != 0
    cancels = [c["argv"] for c in cluster["fake"].calls("scancel")]
    assert cancels == [["scancel", job]], cancels
    assert not cluster["fake"].records()


def test_ctrl_c_after_the_job_started_but_before_the_server_answers(cluster):
    """In a terminal (no --managed): the job runs, its server is not up yet
    (a slow setup line), Ctrl+C cancels exactly that job instead of leaving
    an allocation nobody can reach."""
    import signal
    fake = cluster["fake"]
    fake.update(run_batch={"pending_s": 1})
    p = subprocess.Popen([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "--slurm", "--port",
                          str(H.free_port()), "--dir", str(cluster["root"]), "--setup", "sleep 300"],
                         env=cluster["env"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, preexec_fn=_sigint_default)
    cluster["procs"].append(p)
    end = time.time() + 60
    job = None
    while time.time() < end:
        running = [r for r in fake.records() if r.get("state") == "RUNNING"]
        if running:
            job = running[0]["id"]
            break
        time.sleep(0.3)
    assert job, fake.records()
    time.sleep(36)                  # one 30 s poll later the session has seen it running: it waits for ready.json
    p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 130, out + err
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", job]]
    assert f"Canceled job {job} before MACE GUI was ready." in out, out + err


def test_a_port_taken_while_the_job_waits_keeps_the_job_for_the_next_try(cluster):
    """The forward's port is taken on the login node while the job waits (a
    shared node): port-busy, not "no ssh to the node"; the job is kept, and
    the next try (mace gui connect picks another port) reattaches to it."""
    import socket
    cluster["fake"].update(run_batch={"pending_s": 4})
    s = Slurm(cluster)
    job = s.wait("MACE_GUI_JOB")["job"]
    held = socket.socket()
    held.bind(("127.0.0.1", s.port))
    held.listen(1)
    s2 = None
    try:
        err = s.wait("MACE_GUI_ERROR", 120)
        assert err["kind"] == "port-busy" and "pam_slurm_adopt" not in err["message"], err
        assert s.p.wait(30) == 98
        assert not cluster["fake"].calls("scancel")
        s2 = Slurm(cluster)
        r = s2.wait("MACE_GUI_READY", 120)
        assert r["reattached"] is True and r["job"] == job
        assert len(cluster["fake"].calls("sbatch")) == 1
        s2.send("stop")
        s2.wait("MACE_GUI_STOPPED", 60)
        _wait_gone(cluster["fake"], job)
    finally:
        held.close()
        if s2 is not None and s2.p.poll() is None:
            s2.p.kill()


def test_refused_submission_is_a_clear_error(cluster):
    cluster["fake"].update(reject_with="Batch job submission failed: Invalid partition name specified")
    s = Slurm(cluster, "--partition", "nosuch")
    err = s.wait("MACE_GUI_ERROR")
    assert err["kind"] == "refused" and "Invalid partition name specified" in err["message"]
    assert "--partition" in err["message"]
    assert s.p.wait(30) == 1 and not cluster["fake"].records()


def test_node_that_refuses_ssh_cancels_the_job(cluster):
    s = Slurm(cluster, env={"FAKE_SSH_REFUSE": "localhost"})
    job = s.wait("MACE_GUI_JOB")["job"]
    err = s.wait("MACE_GUI_ERROR", 120)
    assert err["kind"] == "ssh-node" and "pam_slurm_adopt" in err["message"]
    _wait_gone(cluster["fake"], job)
    assert [c["argv"] for c in cluster["fake"].calls("scancel")] == [["scancel", job]]


def test_idle_stop_ends_the_job(cluster):
    s = Slurm(cluster, env={"MACE_GUI_IDLE_STOP_HOURS": "0.005"})       # 18 s with nobody using it
    job = s.wait("MACE_GUI_JOB")["job"]
    s.wait("MACE_GUI_READY")
    stopped = s.wait("MACE_GUI_STOPPED", 90)
    assert stopped["reason"] == "idle"
    _wait_gone(cluster["fake"], job)


def test_time_limit_stops_cleanly_a_minute_before(cluster):
    cluster["fake"].update(run_batch={"pending_s": 1, "end_in_s": 100})
    s = Slurm(cluster, env={"MACE_GUI_TEST_WALLTIME_CHECK_S": "2"})
    job = s.wait("MACE_GUI_JOB")["job"]
    ready = s.wait("MACE_GUI_READY")
    t0 = time.time()
    stopped = s.wait("MACE_GUI_STOPPED", 90)
    assert stopped["reason"] == "walltime"
    assert time.time() - t0 < 60                              # 60 s before the 100 s limit, give or take a check
    _wait_gone(cluster["fake"], job)
    assert ready["ends"]


def test_reattach_to_the_running_job_of_this_folder(cluster):
    s = Slurm(cluster)
    job = s.wait("MACE_GUI_JOB")["job"]
    first = s.wait("MACE_GUI_READY")
    # a task keeps the server: the first session leaves (detach), a second one reattaches
    s2 = None
    try:
        s.send("detach")
        assert s.p.wait(30) == 0
        # with no task the server stops on detach; start a new session to check the job is gone
        _wait_gone(cluster["fake"], job)
        s2 = Slurm(cluster)
        job2 = s2.wait("MACE_GUI_JOB")["job"]
        assert job2 != job
        r2 = s2.wait("MACE_GUI_READY")
        # a third session while job2 runs: no new submission, the same server
        s3 = Slurm(cluster)
        r3 = s3.wait("MACE_GUI_READY")
        assert r3["reattached"] is True and r3["job"] == job2 and r3["token"] == r2["token"]
        assert len([c for c in cluster["fake"].calls("sbatch")]) == 2
        s3.send("stop")
        s3.wait("MACE_GUI_STOPPED", 60)
        _wait_gone(cluster["fake"], job2)
        assert first["token"] != r2["token"]
    finally:
        if s2 is not None and s2.p.poll() is None:
            s2.p.kill()


def test_dry_run_prints_the_request_and_names_site_values(cluster, tmp_path):
    site = tmp_path / "site.toml"
    site.write_text('[slurm]\npartition = "short"\nqos = "fast"\n')
    env = {**cluster["env"], "MACE_SITE_CONFIG": str(site)}
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "gui", "--slurm", "--dry-run", "--dir",
                        str(cluster["root"]), "--cpus", "4"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--partition=short" in r.stdout and "--qos=fast" in r.stdout and "--cpus-per-task=4" in r.stdout
    assert "# partition: site settings" in r.stdout
    assert not cluster["fake"].calls("sbatch")


def _connect_slurm(cluster):
    import re
    fake = cluster["fake"]
    fake.update(run_batch={"pending_s": 2})
    node_bin = fake.bindir
    if not (node_bin / "ssh").exists():
        (node_bin / "ssh").symlink_to(FAKE_SSH)
    node_env = {k: cluster["env"][k] for k in ("FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "FAKE_SLURM_HOOK_LOG",
                                               "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MACE_GUI_SLURM_SSH",
                                               "MACE_GUI_NO_WARMUP", "MACE_GUI_TEST_HOOKS", "OMP_NUM_THREADS",
                                               "MACE_MODULES_NO_PROFILE", "PYTHONDONTWRITEBYTECODE")}
    env = dict(cluster["env"], PATH=f"{node_bin}:{cluster['env']['PATH']}", FAKE_SSH_HOME=cluster["env"]["HOME"],
               FAKE_SSH_BIN=str(node_bin), FAKE_SSH_ENV=json.dumps(node_env),
               XDG_CONFIG_HOME=str(cluster["tmp"] / "localcfg"))
    lp = H.free_port()
    p = subprocess.Popen([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--gateway",
                          "fakegw", "--slurm", "--cpus", "1", "--dir", str(cluster["root"]), "--no-open",
                          "--mace", f"{sys.executable} {REPO / 'mace_cli'}", "--local-port", str(lp)],
                         env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, preexec_fn=_sigint_default)
    cluster["procs"].append(p)
    out = []
    threading.Thread(target=lambda: [out.append(x) for x in p.stdout], daemon=True).start()

    def wait_for(pattern, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            m = re.search(pattern, "".join(out))
            if m:
                return m
            if p.poll() is not None:
                time.sleep(0.5)
                m = re.search(pattern, "".join(out))
                if m:
                    return m
                raise AssertionError("".join(out))
            time.sleep(0.2)
        raise AssertionError(f"no {pattern!r} in:\n{''.join(out)}")
    job = wait_for(r"Submitted job (\d+)").group(1)
    wait_for(r"Waiting for a compute node \(job \d+, pending: Priority")
    wait_for(rf"MACE GUI runs as job {job} on localhost")
    url = wait_for(r"MACE GUI on localhost: (\S+)").group(1)   # the job's node, not the gateway
    assert url.startswith(f"http://127.0.0.1:{lp}/") and "?login=" in url
    return p, job, url, wait_for


def test_connect_slurm_from_your_computer(cluster):
    """`mace gui connect --slurm`: the outer (fake) ssh runs mace gui --slurm on the
    gateway, which submits, waits, hops to the job's node; Ctrl+C on your
    computer stops the server and the job leaves the queue."""
    import signal
    fake = cluster["fake"]
    p, job, url, wait_for = _connect_slurm(cluster)
    sb = fake.calls("sbatch")[0]["argv"]
    assert "--cpus-per-task=1" in sb and "--mem=4G" in sb
    # the page answers through both hops (your computer -> gateway -> the job's node)
    import http.cookiejar
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                         urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(url, timeout=20) as r:            # the one-time link signs in (a cookie), then the page
        assert r.status == 200 and b"MACE" in r.read()
    os.killpg(p.pid, signal.SIGINT)
    wait_for(r"Stopped the server on localhost", 60)
    assert p.wait(60) == 0
    _wait_gone(fake, job)
    assert not fake.calls("scancel")


@pytest.mark.gui
def test_browser_shows_the_job_and_sign_out_ends_it(cluster):
    """Real clicks: the status bar names the job; File > Sign Out and Stop Server
    says it ends the job, and after the click the job leaves the queue."""
    fake = cluster["fake"]
    p, job, url, wait_for = _connect_slurm(cluster)
    drv = H.firefox()
    try:
        drv.get(url)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        chip = H.wait(drv, "const e = document.querySelector('#st-slurm'); return e && e.textContent", 30)
        assert chip.startswith(f"Job {job}"), chip
        top = H.wait(drv, "return [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === 'File')", 10)
        top.click()
        H.wait(drv, "return !!document.querySelector(\".menu .menu-item[data-cmd-id='file.signOut']\")", 5)
        drv.find_element("css selector", ".menu .menu-item[data-cmd-id='file.signOut']").click()
        text = H.wait(drv, "const e = document.querySelector('#so-job'); return e && e.textContent", 15)
        assert f"This also ends the job {job}" in " ".join(text.split())
        drv.find_element("css selector", "#so-now").click()
        H.wait(drv, "return !!document.querySelector('#stopped-page')", 30)
        assert not H.errors(drv)
    finally:
        drv.quit()
    wait_for(r"stopped from the browser", 60)
    assert p.wait(60) == 0
    _wait_gone(fake, job)
    assert not fake.calls("scancel")
