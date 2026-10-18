"""MACE on any SLURM cluster (mace/submission/site.py, resolve()).

With no setting of the user's own, the built-in values (MSU HPCC's) are
checked against the cluster MACE runs on:

* another university's cluster (fake sacctmgr/scontrol/module/sbatch and a
  fake host name): no `-A mendoza_q`, no MSU module line, no MSU node name in
  any generated script or message; the account is left to SLURM's default,
  the completion step runs the Python that runs MACE, the queue limits follow
  MaxSubmitJobs and the walltime the partition's MaxTime;
* the owner's case (a member of mendoza_q, the modules present, an MSU host):
  every script byte-identical to the shipped template's own output;
* no SLURM and no module system (a workstation): unchanged (also pinned by
  test_site_jobscript.py::test_default_bytes).

Everything runs the real `mace_cli` as a subprocess."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_modules import FakeModules
from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site

# what no generated script may carry off MSU HPCC (the templates' comments that
# record where a scratch fallback was measured are not settings)
SCRIPT_PATTERN = re.compile(r"mendoza|dev-amd|dev-intel|amd20|intel18|Python/3\.11\.3|Python-bundle", re.I)
# and what no message may say
MSU_PATTERN = re.compile(SCRIPT_PATTERN.pattern + r"|HPCC|\bMSU\b", re.I)
OTHER_HOST = "login1.cluster.example.edu"
MSU_HOST = "dev-amd20.hpcc.msu.edu"
PARTITIONS = ("PartitionName=short AllowAccounts=ALL MaxTime=2-00:00:00 State=UP Default=YES\n"
              "PartitionName=long AllowAccounts=ALL MaxTime=14-00:00:00 State=UP\n")
ASSOC_MGR = ("ClusterName=c Account=grp_x UserName=tester(5) Partition= Priority=0 ID=2\n"
             "    ParentAccount= Lineage=/grp_x/0-tester/ DefAssoc=Yes\n"
             "    MaxJobs=100(0) MaxJobsAccrue= MaxSubmitJobs=200(3) MaxWallPJ=\n")


@pytest.fixture(autouse=True)
def _fresh():
    site._MEMO.clear()
    site._SLURM_MEMO.clear()
    yield
    site._MEMO.clear()
    site._SLURM_MEMO.clear()


def _cluster(tmp, *, owner, gateway=False):
    """(env, fake slurm): another cluster, or the owner's (MSU HPCC; on its
    gateway no module loads and the default partition is a short one)."""
    if owner:
        fake = FakeSlurm(tmp / "fs", {"assoc": ["mendoza_q", "general"], "default_account": "general",
                                      "partitions": PARTITIONS.replace("2-00:00:00", "04:00:00" if gateway
                                                                       else "7-00:00:00"),
                                      "assoc_mgr": ASSOC_MGR.replace("grp_x", "mendoza_q")
                                                            .replace("MaxSubmitJobs=200", "MaxSubmitJobs=1000")})
        fm = FakeModules(tmp / "fm", {} if gateway else {"CRYSTAL/23-intel-2023a": {"root": True}},
                         python=not gateway)
        host = "gateway-00.hpcc.msu.edu" if gateway else MSU_HOST
    else:
        fake = FakeSlurm(tmp / "fs", {"assoc": ["grp_x", "grp_y"], "default_account": "grp_x",
                                      "partitions": PARTITIONS, "assoc_mgr": ASSOC_MGR})
        fm = FakeModules(tmp / "fm", {"CRYSTAL/23-foss-2023b": {"root": True}}, python=False)
        host = OTHER_HOST
    base = _env(tmp, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
                FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME=host)
    return fm.env(base), fake


def _python_line(text):
    lines = [l for l in text.splitlines() if l.startswith("python() {")]
    return lines[0] if lines else None


# ---------------------------------------------------------------- another cluster
@pytest.mark.parametrize("kind", ["d12", "d3"])
def test_other_cluster_scripts_name_nothing_of_msu(tmp_path, kind):
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    name, text = ("m.d12", SP_D12) if kind == "d12" else ("m_band.d3", BAND_D3)
    _deck(d, name, text)
    if kind == "d3":
        (d / "m_band.f9").write_text("fort9\n")
    p = _mace(d, env, "submit", name)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    out = p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    script = call["script_text"]
    assert not SCRIPT_PATTERN.search(script), SCRIPT_PATTERN.search(script)
    assert "#SBATCH -A" not in script                          # SLURM's default account
    # the CRYSTAL module of this cluster, the Python that runs MACE, its queue limits
    assert "module load CRYSTAL/23-foss-2023b" in script
    assert _python_line(script) == f'python() {{ "{os.path.abspath(sys.executable)}" "$@"; }}'
    assert "--max-jobs 200 --reserve 20 --max-submit 10" in script and "--max-jobs 950" not in script
    if kind == "d12":
        assert "#SBATCH -t 2-00:00:00" in script               # the default partition's MaxTime
    else:
        assert "#SBATCH -t 2:00:00" in script                  # properties' 2 hours fit
    # said once each, and no MSU name in anything MACE printed
    assert out.count("SLURM charges your default account (grp_x)") == 1
    assert out.count("the job scripts run their completion step with the Python that runs MACE") == 1
    assert not MSU_PATTERN.search(out), MSU_PATTERN.search(out)
    # the job script is valid bash and its python() is the Python that runs MACE
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    r = subprocess.run(["bash", "-c", _python_line(script) + "\npython -c 'import sys; print(sys.executable)'"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == sys.executable


def test_other_cluster_user_settings_win(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[slurm]\naccount = "grp_y"\npartition = "long"\n[modules]\npython = ["Python/3.12"]\n'
                    '[queue]\nmax_jobs = 40\nreserve = 5\n')
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    sh = (d / "m.sh").read_text()
    assert "#SBATCH -A grp_y" in sh and "#SBATCH -p long" in sh
    assert "module load Python/3.12" in sh and _python_line(sh) is None
    assert "--max-jobs 40 --reserve 5 --max-submit 10" in sh
    assert "#SBATCH -t 7-00:00:00" in sh                         # long allows 14 days
    assert "default account" not in p.stdout


def test_other_cluster_partition_without_account_line(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    env["MACE_SLURM_PARTITION"] = "long"
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    sh = (d / "m.sh").read_text()
    lines = sh.splitlines()
    assert "#SBATCH -A" not in sh
    assert lines.index("#SBATCH -p long") == lines.index("#SBATCH --ntasks=32") + 1


def test_other_cluster_interactive_blank_answers(tmp_path):
    """mace submit -i: the account default is SLURM's; blank answers keep it out."""
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "-i", "--nosubmit", "m.d12", stdin="y\n\n\n\n\nn\n")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "Account: (your SLURM default account)" in p.stdout
    assert "AMD20" not in p.stdout
    sh = (d / "m.sh").read_text()
    assert "#SBATCH -A" not in sh and not SCRIPT_PATTERN.search(sh)


def test_other_cluster_workflow(tmp_path):
    """A planned workflow stores the decisions once; every step script follows them."""
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1))
    p = _mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    (plan,) = sorted((d / "workflow_configs").glob("workflow_plan_*.json"))
    assert "mendoza" not in plan.read_text()
    scripts = [c["script_text"] for c in fake.calls("sbatch")]
    assert scripts
    for s in scripts:
        assert "#SBATCH -A" not in s and not SCRIPT_PATTERN.search(s), SCRIPT_PATTERN.search(s)
        assert _python_line(s)
    for c in sorted((d / "workflow_scripts").glob("*.sh")):
        assert "#SBATCH -A" not in c.read_text(), c.name


def test_other_cluster_site_and_describe(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Account:     (your SLURM default account)" in r.stdout
    assert "SLURM charges your default account (grp_x)" in r.stdout
    assert "(the Python that runs MACE" in r.stdout
    assert "--max-jobs 200 --reserve 20  (MaxSubmitJobs 200)" in r.stdout
    assert "at most 2-00:00:00 (partition short)" in r.stdout
    d = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert d["account"]["value"] is None and d["account_check"]["status"] == "default"
    assert d["python"]["executable"] == os.path.abspath(sys.executable)
    assert d["cluster"]["msu_hpcc"] is False
    assert not MSU_PATTERN.search(d["messages"]["gateway_message"]["value"])
    # the GUI asks for no "submit with this account anyway" tick
    from mace.gui import submit as gsub
    p = gsub.Preview(id="", created=0, where="slurm", cmds=[], decks=[], fp={}, result={}, needs={},
                     scripts={}, modified=[], tracked=False)

    class _Store:
        def sign(self, fp):
            return "x"

        def put(self, p):
            pass
    out = gsub.finalize(_Store(), p, d["account_check"])
    assert not out["needs"].get("account")


def test_other_cluster_exclusion_menus(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    from mace.utils import node_exclusion as ne
    assert ne.site_menu(ne.MANAGER_MENU, env) == ne.GENERIC_MENU
    assert ne.site_menu(ne.PLANNER_MENU, env) == ne.GENERIC_MENU
    assert ne.menu_default_answer(ne.GENERIC_MENU) == "1"
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "--exclude-mendoza", "m.d12")
    assert p.returncode == 0 and "applies only to members" in p.stdout + p.stderr
    assert "--exclude" not in (d / "m.sh").read_text().replace("#echo '#SBATCH --exclude", "")


def test_msu_host_but_not_a_member_hides_only_mendoza(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    env["MACE_HOSTNAME"] = MSU_HOST
    from mace.utils import node_exclusion as ne
    menu = ne.site_menu(ne.PLANNER_MENU, env)
    assert menu == ("none", "amd20", None, "by_type", "manual")
    assert ne.menu_choice(menu, "3") is None and ne.menu_choice(menu, "4") == "by_type"
    lines = ne.NodeExclusionManager().menu_lines(menu)
    assert [l.split(")")[0] for l in lines] == ["1", "2", "4", "5"]


def test_callback_in_a_job_inherits_the_python_choice(tmp_path):
    """Inside a job (a completion callback) nothing is probed: the Python
    modules the job loaded are kept, and a job that loaded none keeps calling
    the Python that runs MACE."""
    env, fake = _cluster(tmp_path, owner=False)
    env.update(SLURM_JOB_ID="77", LOADEDMODULES="CRYSTAL/23-foss-2023b")
    s = site.load(env)
    assert site.resolve_python(s, env)[0] == os.path.abspath(sys.executable)
    env["LOADEDMODULES"] = "CRYSTAL/23-foss-2023b:" + ":".join(site.DEFAULTS["python"])
    assert site.resolve_python(site.load(env), env) == (None, "inherited", "")


def test_executor_drops_the_account_line_for_none(tmp_path):
    from types import SimpleNamespace
    from mace.workflow.executor import WorkflowExecutor
    stub = SimpleNamespace(work_dir=tmp_path, isolation_mode="isolated")
    out = WorkflowExecutor.apply_script_customizations(stub, SCF.read_text(),
                                                       {"resources": {"account": None}, "workflow_id": "w"})
    assert "#SBATCH -A" not in out and "--account" not in out
    out = site.customize_workflow_text(SCF.read_text(), {"python_executable": "/opt/py/bin/python3",
                                                          "queue_limits": [90, 10]})
    assert 'python() { "/opt/py/bin/python3" "$@"; }' in out and "Python/3.11.3" not in out
    assert "--max-jobs 90 --reserve 10 --max-submit 10" in out


def test_unsafe_python_path_keeps_the_module_lines(tmp_path, monkeypatch):
    env, fake = _cluster(tmp_path, owner=False)
    monkeypatch.setattr(sys, "executable", "/home/a b/python")
    exe, status, note = site.resolve_python(site.load(env), env)
    assert exe is None and status == "kept" and "cannot be written" in note


# ---------------------------------------------------------------- the owner's case
@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("kind", ["d12", "d3"])
def test_owner_cluster_keeps_every_byte(tmp_path, kind, gateway):
    env, fake = _cluster(tmp_path, owner=True, gateway=gateway)
    # the implicit group-file locations are looked at (and empty), and sinfo answers
    env.pop("MACE_SITE_GROUP_CONFIG", None)
    (tmp_path / "xdg_dirs").mkdir()
    env["XDG_CONFIG_DIRS"] = str(tmp_path / "xdg_dirs")
    fake.update(sinfo="".join(f"amr-{i:03d}|idle|2|64|1|128|500000|amd20|(null)|general*\n" for i in range(3)))
    d = tmp_path / "decks"
    name, text, tmpl = ("m.d12", SP_D12, SCF) if kind == "d12" else ("m_band.d3", BAND_D3, PROP)
    _deck(d, name, text)
    p = _mace(d, env, "submit", "--nosubmit", name)
    assert p.returncode == 0, p.stdout + p.stderr
    got = (d / (Path(name).stem + ".sh")).read_bytes()
    assert got == _template_output(tmp_path, tmpl, Path(name).stem)
    assert not list(d.glob(".*_site.*.sh"))
    assert "default account" not in p.stdout and "Python that runs MACE" not in p.stdout
    assert gateway or "MACE:" not in p.stdout          # (the gateway says no CRYSTAL module loads there)
    r = _mace(tmp_path, env, "site")
    assert "Account check: mendoza_q is one of your SLURM accounts" in r.stdout


def test_owner_cluster_menus_and_messages_unchanged(tmp_path):
    env, fake = _cluster(tmp_path, owner=True)
    from mace.utils import node_exclusion as ne
    assert ne.site_menu(ne.MANAGER_MENU, env) == ne.MANAGER_MENU
    assert ne.site_menu(ne.PLANNER_MENU, env) == ne.PLANNER_MENU
    msg = site.load(env).message("gateway_message", env, reason="r", host="h")
    assert msg.endswith("On MSU HPCC: ssh dev-amd20, dev-amd24 or dev-intel18.")


def test_workstation_without_slurm_keeps_msu_menus(tmp_path):
    env = _env(tmp_path, PATH="/usr/bin:/bin", MACE_HOSTNAME="laptop")
    from mace.utils import node_exclusion as ne
    assert ne.site_menu(ne.MANAGER_MENU, env) == ne.MANAGER_MENU
    s = site.resolve(site.load(env), env, out=lambda t: None)
    assert (s.account, s.python_executable, s.queue_text) == ("mendoza_q", None, site.QUEUE_TEXT)
    assert s.checks["account"] == "not checked" and s.checks["python"] == "unchecked"


# ------------------------------------------------- MSU-only submit options
MSU_FLAGS = ("--exclude-amd20", "--exclude-mendoza")


@pytest.mark.parametrize("ask", [("submit", "--help"), ("help", "submit")])
def test_other_cluster_submit_help_names_no_msu_option(tmp_path, ask):
    """Off MSU HPCC the help lists the generic exclusion options only; the
    MSU-only ones (and its node names) are not shown."""
    env, fake = _cluster(tmp_path, owner=False)
    p = _mace(tmp_path, env, *ask)
    out = p.stdout + p.stderr
    assert "--exclude-nodes" in out and "--exclude-type" in out, out[-2000:]
    for flag in MSU_FLAGS:
        assert flag not in out
    assert not MSU_PATTERN.search(out), MSU_PATTERN.search(out)


@pytest.mark.parametrize("ask", [("submit", "--help"), ("help", "submit")])
def test_msu_submit_help_keeps_the_msu_options(tmp_path, ask):
    env, fake = _cluster(tmp_path, owner=True)
    p = _mace(tmp_path, env, *ask)
    out = p.stdout + p.stderr
    for flag in MSU_FLAGS + ("--exclude-nodes",):
        assert flag in out, out[-2000:]
    if ask[0] == "help":     # the curated page, exactly as before
        assert ("  --exclude-amd20      MSU HPCC only: exclude all AMD20 nodes (amr + nvf types)\n"
                "  --exclude-mendoza    MSU HPCC only: exclude the Mendoza group's nodes (members\n"
                "                       of its account)\n\nExamples:") in out


def test_other_cluster_exclude_amd20_is_accepted_but_excludes_nothing(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "--exclude-amd20", "m.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "--exclude-amd20 applies only on MSU HPCC" in out, out[-3000:]
    assert "#SBATCH --exclude" not in (d / "m.sh").read_text()
    assert not fake.calls("sinfo")
