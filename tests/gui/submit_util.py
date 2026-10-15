"""Shared plumbing of the Submit / Jobs / Batch tests: a real `mace gui`
server with the logging fake SLURM (tests/fake_slurm.py) first on its PATH,
a project folder with decks, and the preview -> run round trip.

Decks are the small diamond deck of the default-path goldens (public); the
corpus tests copy real outputs from test/ into the project at run time."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guiharness as H  # noqa: E402
import p4util as U  # noqa: E402
from fake_slurm import FakeSlurm  # noqa: E402
from test_array_default_path_golden import BAND_D3, OPT_D12, SP_D12  # noqa: E402

REPO = H.REPO_ROOT
MACE_CLI = REPO / "mace_cli"
USER = "tester"


def fake_env(fake, tmp):
    """The variables a test's server (and a twin shell) get: the fake SLURM
    first on PATH, a private HOME/XDG, one user name."""
    (tmp / "home").mkdir(parents=True, exist_ok=True)
    return {"PATH": fake.env["PATH"], "FAKE_SLURM_STATE": fake.env["FAKE_SLURM_STATE"],
            "FAKE_SLURM_LOG": fake.env["FAKE_SLURM_LOG"], "FAKE_SLURM_HOOK_LOG": fake.env["FAKE_SLURM_HOOK_LOG"],
            "USER": USER, "LOGNAME": USER, "HOME": str(tmp / "home"), "MACE_MODULES_NO_PROFILE": "1",
            "XDG_CACHE_HOME": str(tmp / "xdg" / "cache")}


class Served:
    def __init__(self, tmp, state=None, **extra):
        self.tmp = Path(tmp)
        self.root = self.tmp / "proj"
        self.root.mkdir(parents=True, exist_ok=True)
        self.fake = FakeSlurm(self.tmp / "fs", state)
        self.env = fake_env(self.fake, self.tmp)
        self.env.pop("XDG_CACHE_HOME")              # guiharness sets it under the server's XDG folder
        self.env.update(extra)
        self.proc, self.base, _ = H.start_server(self.root, self.tmp / "xdg", MACE_GUI_TEST_HOOKS="1",
                                                 MACE_GUI_NO_WARMUP="1", **self.env)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)

    # ---------------------------------------------------------------- files
    def decks(self, folder, names, text=SP_D12, ext=".d12"):
        d = self.root / folder
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / f"{n}{ext}").write_text(text)
        return d

    # ---------------------------------------------------------------- api
    def req(self, path, data=None, **kw):
        return U.req(self.base, path, data=data, **kw)

    def task_result(self, r, timeout=300):
        st, _, body = r
        if st == 202:
            tid = body["task"]["id"]
            end = time.time() + timeout
            while time.time() < end:
                s, _, res = self.req(f"api/tasks/{tid}/result")
                if s == 200:
                    return 200, res
                if s != 409:
                    return s, res
                if isinstance(res, dict) and res.get("task", {}).get("state") in ("failed", "canceled"):
                    return 500, res
                time.sleep(0.15)
            raise AssertionError("task did not finish")
        return st, body

    def preview(self, targets, where="slurm", options=None, mixed=None, **extra):
        body = {"targets": targets, "where": where, "options": options or {}, "wait": 60, **extra}
        if mixed is not None:
            body["mixed_choice"] = mixed
        return self.task_result(self.req("api/submit/preview", body))

    def run(self, pid, **ticks):
        r = self.req("api/submit/run", {"preview_id": pid, **ticks})
        if r[0] in (400, 403, 409):
            return r[0], r[2]
        return self.task_result(r)

    def calls(self, *tools):
        return [c for c in self.fake.calls() if not tools or c["tool"] in tools]

    def clear_calls(self):
        self.fake.log_file.write_text("")


def twin_submit(tmp, fake, folder, args, stdin=""):
    """`mace submit ARGS` typed in a shell, in a copy of ``folder``'s parent."""
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH", "COLORTERM", "TERM", "NO_COLOR")}
    env.update(fake_env(fake, tmp))
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=str(folder), env=env, input=stdin,
                          capture_output=True, text=True, timeout=600)


ID_NORM = [(re.compile(r"(?<=# MACE-generated; cksum of the lines above: )\d+ \d+"), "<CKSUM>"),
           (re.compile(r"\b(9\d{8}|4\d{3}|5\d{3})(_\d+)?\b"), "<JOB>"),
           (re.compile(r"arr_\d{8}_\d{6}_\d+"), "arr_<TAG>"),
           (re.compile(r"(wf_[0-9a-f]{6,}|workflow_\d{8}_\d{6}(?:_[0-9a-f]{4})?|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"), "<WF>"),
           (re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?"), "<TS>"),
           (re.compile(r"(?<!\d)\d{8}_\d{6}\d*(?:_[0-9a-f]{4}(?![0-9A-Za-z]))?(?!\d)"), "<TS>")]


def norm(text, *roots):
    text = str(text)
    for r in roots:
        text = text.replace(str(r), "<ROOT>")
    for rx, tag in ID_NORM:
        text = rx.sub(tag, text)
    return text
