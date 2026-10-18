"""Common SLURM variants and the last MSU-shaped limits (PLAN_G G7), through
the real mace_cli with the fake SLURM: [slurm] qos/constraint/extra,
[modules] purge, the sbatch hints, the workflow queue clamp and the
`mace run` time cap."""
import json
import subprocess

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import ASSOC_MGR, _cluster
from test_site_jobscript import _deck, _env, _mace, _toml

from mace.submission import local_run, site

DIRECTIVES = '[slurm]\nqos = "normal"\nconstraint = "epyc&ib"\nextra = ["--exclusive", "--reservation=maint"]\n'


def _sbatch(text):
    return [l for l in text.splitlines() if l.startswith("#SBATCH")]


def _submit(tmp, env, deck="m.d12", *args):
    d = tmp / "decks"
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    (d / (deck.rsplit(".", 1)[0] + ".sh")).unlink(missing_ok=True)
    p = _mace(d, env, "submit", "--nosubmit", *args, deck)
    return p, d / (deck.rsplit(".", 1)[0] + ".sh")


@pytest.mark.parametrize("deck", ["m.d12", "m_band.d3"])
def test_directives_follow_the_account(tmp_path, deck):
    _toml(tmp_path, DIRECTIVES + 'account = "grp_a"\npartition = "p1"\n')
    p, sh = _submit(tmp_path, _env(tmp_path), deck)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = _sbatch(sh.read_text())
    i = lines.index("#SBATCH -A grp_a")
    assert lines[i + 1:i + 6] == ["#SBATCH -p p1", "#SBATCH --qos=normal", "#SBATCH --constraint=epyc&ib",
                                  "#SBATCH --exclusive", "#SBATCH --reservation=maint"]


def test_directives_without_an_account_line(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)                    # the -A line is dropped there
    _toml(tmp_path, DIRECTIVES)
    p, sh = _submit(tmp_path, env)
    lines = _sbatch(sh.read_text())
    assert "#SBATCH -A" not in sh.read_text()
    i = lines.index("#SBATCH --ntasks=32")
    assert lines[i + 1:i + 5] == ["#SBATCH --qos=normal", "#SBATCH --constraint=epyc&ib", "#SBATCH --exclusive",
                                  "#SBATCH --reservation=maint"]
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0


def test_env_directives(tmp_path):
    env = _env(tmp_path, MACE_SLURM_QOS="debug", MACE_SLURM_EXTRA="--licenses=crystal:1,foo:2,--hint=nomultithread")
    p, sh = _submit(tmp_path, env)
    t = sh.read_text()
    assert "#SBATCH --qos=debug" in t and "#SBATCH --licenses=crystal:1,foo:2" in t and "#SBATCH --hint=nomultithread" in t


def test_gres_and_gpus_pass_through_as_extra(tmp_path):
    """No MACE job asks for GPUs (G8 is 1.2); a site may still add --gres or --gpus lines."""
    _toml(tmp_path, '[slurm]\nextra = ["--gres=craynetwork:1", "--gpus=1"]\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    t = sh.read_text()
    assert "#SBATCH --gres=craynetwork:1" in t and "#SBATCH --gpus=1" in t


@pytest.mark.parametrize("bad,msg", [('["--ntasks=4"]', "[resources] ntasks"), ('["--x;id"]', "#SBATCH option"),
                                     ('["--a b"]', "#SBATCH option"), ('["-A x"]', "#SBATCH option"),
                                     ('["--account=x"]', "[slurm] account"), ('["--chdir=x"]', "MACE runs the job"),
                                     ('["--x=$(id)"]', "#SBATCH option"), ('"--exclusive\\n--x"', "#SBATCH option")])
def test_extra_negatives(tmp_path, bad, msg):
    _toml(tmp_path, f"[slurm]\nextra = {bad}\n")
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and msg in p.stdout and not sh.exists(), p.stdout


@pytest.mark.parametrize("bad", ['qos = "a b"', "qos = \"x'\"", 'constraint = "a;b"', 'constraint = "$(id)"'])
def test_qos_constraint_negatives(tmp_path, bad):
    _toml(tmp_path, f"[slurm]\n{bad}\n")
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and not sh.exists()


def test_extra_in_job_arrays(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[slurm]\nextra = ["--exclusive"]\nqos = "normal"\n')
    d = tmp_path / "decks"
    for n in ("a", "b"):
        _deck(d, f"{n}.d12")
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    subs = [c for c in fake.calls("sbatch") if "--test-only" not in c["argv"]]
    assert subs and all("#SBATCH --exclusive" in c["script_text"] and "#SBATCH --qos=normal" in c["script_text"]
                        for c in subs), [c["script_text"][:400] for c in subs]
    from mace.submission import job_array
    group = job_array.ArrayGroup.__new__(job_array.ArrayGroup)
    group.directive_lines = ["#SBATCH --exclusive", "#SBATCH --reservation=maint", "#SBATCH --qos=normal"]
    assert job_array.describe_request(group) == "--exclusive --reservation=maint --qos=normal"


def test_module_purge_off(tmp_path):
    _toml(tmp_path, "[modules]\npurge = false\n")
    for deck in ("m.d12", "m_band.d3"):
        p, sh = _submit(tmp_path, _env(tmp_path), deck)
        t = sh.read_text()
        assert site.PURGE_COMMENT in t and "\nmodule purge\n" not in t
    p, sh = _submit(tmp_path, _env(tmp_path, MACE_MODULE_PURGE="1"))
    assert "\nmodule purge\n" in sh.read_text()


# ---------------------------------------------------------------- hints (G7f)
@pytest.mark.parametrize("stderr,hint", [
    ("Batch job submission failed: Invalid account or account/partition combination specified", "Set your cluster account in"),
    ("Batch job submission failed: You must specify an account", "SLURM needs an account here"),
    ("Batch job submission failed: Invalid qos specification", "[slurm] qos"),
    ("Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or "
     "time limits)", "[slurm] qos"),
    ("Batch job submission failed: Requested node configuration is not available", "[resources] ntasks"),
    ("Batch job submission failed: Memory required by task is not available", "[resources] ntasks"),
    ("Batch job submission failed: Invalid feature specification", "[slurm] constraint"),
    ("Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)", "--walltime"),
])
def test_sbatch_hints_once_per_run(tmp_path, stderr, hint):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(reject_with=stderr)
    d = tmp_path / "decks"
    for n in ("a", "b", "c"):
        _deck(d, f"{n}.d12")
    p = _mace(tmp_path, env, "submit", "decks")
    out = p.stdout + p.stderr
    assert out.count(hint) == 1, out[-3000:]
    assert site.sbatch_hint(stderr) and hint in site.sbatch_hint(stderr)


def test_submit_limit_refusal_gives_no_qos_hint():
    assert site.sbatch_hint("sbatch: error: QOSMaxSubmitJobPerUserLimit\nJob violates accounting/QOS policy") is None


# ---------------------------------------------------------------- workflow queue clamp (G7b)
def test_resolve_queue_counts_the_reserve_once():
    # the callback keeps max_jobs - reserve queued: max_jobs is the limit
    # itself and the reserve (100, a tenth of a smaller limit) is not added again
    # (a limit of 1000 gives the built-in budget, 900: the templates' 950/50 say it)
    for limit, want in ((200, (200, 20)), (1000, (950, 50)), (1040, (1040, 100)), (2, (2, 1))):
        assert site.queue_from_limit(limit) == want
    assert site.clamp_queue(250, 30, None, {"PATH": "/nonexistent"})[:2] == (250, 30)


@pytest.mark.parametrize("limit,want", [(200, "--max-jobs 200 --reserve 20 --max-submit 10"),
                                        (1000, "--max-jobs 950 --reserve 50 --max-submit 10"),
                                        (None, "--max-jobs 950 --reserve 50 --max-submit 10")])
def test_workflow_queue_clamp(tmp_path, limit, want):
    env, fake = _cluster(tmp_path, owner=False)
    if limit is None:
        fake.update(assoc_mgr="")
    else:
        fake.update(assoc_mgr=ASSOC_MGR.replace("MaxSubmitJobs=200", f"MaxSubmitJobs={limit}"))
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1))
    p = _mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    for c in sorted((d / "workflow_scripts").glob("*.sh")):
        assert want in c.read_text(), c.name
    (plan,) = sorted((d / "workflow_configs").glob("workflow_plan_*.json"))
    q = json.loads(plan.read_text())["queue_management"]
    assert f"--max-jobs {q['max_jobs']} --reserve {q['reserve_slots']}" in want
    # the plan's numbers come from the limit already: nothing to lower, nothing said
    assert "MaxSubmitJobs is 200" not in p.stdout + p.stderr


def test_planner_regex_line_follows_the_clamp(tmp_path):
    from types import SimpleNamespace
    from mace.workflow.planner import WorkflowPlanner
    stub = SimpleNamespace(work_dir=tmp_path)
    text = "python manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion\n" \
           "echo 'export JOB='$1 >> $1.sh\n"
    q = {"max_jobs": 190, "reserve_slots": 10, "max_submit_batch": 10, "max_recovery_attempts": 3}
    cfg = {"resources": {"ntasks": 32, "walltime": "1:00:00"}, "workflow_id": "w"}
    out = WorkflowPlanner.apply_script_customizations(stub, text, cfg, q)
    assert "--max-jobs 190 --reserve 10 --max-submit 10" in out


# ---------------------------------------------------------------- mace run cap (G7a)
def _cap(monkeypatch, soft, host, max_time=None):
    import resource
    monkeypatch.setattr(resource, "getrlimit", lambda what: (soft, resource.RLIM_INFINITY))
    monkeypatch.setenv("MACE_HOSTNAME", host)
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    if max_time:
        monkeypatch.setenv("MACE_RUN_MAX_TIME", max_time)
    else:
        monkeypatch.delenv("MACE_RUN_MAX_TIME", raising=False)
    return local_run.timeout_cap()


def test_timeout_cap_rules(monkeypatch):
    import resource
    inf = resource.RLIM_INFINITY
    assert _cap(monkeypatch, inf, "dev-amd20.hpcc.msu.edu")[0] == local_run.MAX_TIMEOUT
    assert _cap(monkeypatch, 3600, "dev-amd20.hpcc.msu.edu")[0] == 3600 - 300
    assert _cap(monkeypatch, inf, "login1.cluster.example.edu")[0] is None
    assert _cap(monkeypatch, 4 * 3600, "login1.cluster.example.edu")[0] == 4 * 3600 - 300
    assert _cap(monkeypatch, inf, "login1.cluster.example.edu", "4h")[:1] == (4 * 3600,)
    assert _cap(monkeypatch, 3600, "dev-amd20.hpcc.msu.edu", "4h")[0] == 3600 - 300        # min'ed with ulimit
    assert _cap(monkeypatch, inf, "dev-amd20.hpcc.msu.edu", "3h")[0] == 3 * 3600


def _run_env(tmp, host, **extra):
    stubs = tmp / "stubs"
    stubs.mkdir(exist_ok=True)
    for tool in ("mpirun", "Pcrystal", "Pproperties"):
        (stubs / tool).write_text("#!/bin/bash\nexit 0\n")
        (stubs / tool).chmod(0o755)
    eb = tmp / "eb" / "bin"
    eb.mkdir(parents=True, exist_ok=True)
    for b in ("Pcrystal", "Pproperties"):
        (eb / b).write_text("#!/bin/bash\nexit 0\n")
        (eb / b).chmod(0o755)
    (tmp / "scr").mkdir(exist_ok=True)
    return _env(tmp, PATH=f"{stubs}:/usr/bin:/bin", EBROOTCRYSTAL=str(tmp / "eb"), SCRATCH=str(tmp / "scr"),
                MACE_HOSTNAME=host, **extra)


def test_mace_run_three_hours(tmp_path):
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    other = _mace(d, _run_env(tmp_path, "login1.cluster.example.edu"), "run", "--dry-run", "--timeout", "3h", "m.d12")
    assert other.returncode == 0, other.stdout + other.stderr
    msu = _mace(d, _run_env(tmp_path, "dev-amd20.hpcc.msu.edu"), "run", "--dry-run", "--timeout", "3h", "m.d12")
    assert msu.returncode == 2 and "1h55m" in msu.stdout + msu.stderr
    _toml(tmp_path, '[local_run]\nmax_time = "4h"\n')
    ok = _mace(d, _run_env(tmp_path, "dev-amd20.hpcc.msu.edu"), "run", "--dry-run", "--timeout", "3h", "m.d12")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    no = _mace(d, _run_env(tmp_path, "login1.cluster.example.edu"), "run", "--dry-run", "--timeout", "5h", "m.d12")
    assert no.returncode == 2 and "[local_run] max_time allows at most 4h00m00s" in no.stdout + no.stderr


def test_timeout_help_text(tmp_path):
    r = _mace(tmp_path, _env(tmp_path), "run", "--help")
    flat = " ".join(r.stdout.split())
    assert "1h55m on MSU HPCC, or [local_run] max_time" in flat or "1h55m" in flat
    assert " ".join(local_run.build_parser().format_help().split()).count("[local_run] max_time") == 1
