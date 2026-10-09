"""Site settings in the generated job scripts (PLAN_submit D27, D28, D31).

D27: with no site.toml (or the all-commented one `mace site --init` writes)
and no module system, every script MACE generates is byte-identical to the
shipped template's own output, install pin and line 12's trailing space
included. (The workflow copies and plans are pinned by the default-path
goldens in test_array_default_path_golden.py.)
D28: account, partition (one or a list) and Python modules reach every
shipped-template route, and only those lines change.
D31: only MACE's own templates are rewritten, idempotently; copies and
edited lines are left as written."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12

from mace.submission import site

MACE_CLI = REPO_ROOT / "mace_cli"
SUB = REPO_ROOT / "mace" / "submission"
SCF, PROP = SUB / "submitcrystal23.sh", SUB / "submit_prop.sh"
SITE_TOML = '[slurm]\naccount = "grp_example"\npartition = "a,b"\n[modules]\npython = ["Python/3.12-x"]\n'


def _env(tmp, **extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MACE_SLURM_", "MACE_CRYSTAL", "MACE_SITE", "MODULE", "LMOD", "SLURM_"))
           and k not in ("MACE_HOME", "PYTHONPATH", "BASH_ENV", "LOADEDMODULES")}
    (tmp / "home").mkdir(parents=True, exist_ok=True)
    # no group site settings (/etc/mace, $XDG_CONFIG_DIRS) unless a test names one
    env.update({"XDG_CONFIG_HOME": str(tmp / "xdg"), "XDG_CACHE_HOME": str(tmp / "cache"), "HOME": str(tmp / "home"),
                "MACE_MODULES_NO_PROFILE": "1", "MACE_NO_BANNER": "1", "USER": "tester",
                "MACE_SITE_GROUP_CONFIG": "none", **extra})
    return env


def _toml(tmp, text=SITE_TOML):
    d = tmp / "xdg" / "mace"
    d.mkdir(parents=True, exist_ok=True)
    (d / "site.toml").write_text(text)


def _mace(cwd, env, *args, stdin=""):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args], cwd=cwd, env=env,
                          input=stdin, capture_output=True, text=True, timeout=300)


def _template_output(tmp, template, job):
    """The shipped template run by hand in a twin folder, pinned to this
    install as mace_cli pins it (MACE_HOME)."""
    twin = tmp / "twin"
    twin.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["MACE_HOME"] = str(REPO_ROOT)
    subprocess.run(["bash", str(template), job, "100"], cwd=twin, env=env, check=True, capture_output=True)
    return (twin / f"{job}.sh").read_bytes()


def _deck(d, name, text=SP_D12):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


# ---------------------------------------------------------------- D27
@pytest.mark.parametrize("kind", ["d12", "d3"])
@pytest.mark.parametrize("init", [False, True])
def test_default_bytes(tmp_path, kind, init):
    env = _env(tmp_path)
    if init:
        r = _mace(tmp_path, env, "site", "--init")
        assert r.returncode == 0, r.stdout + r.stderr
    d = tmp_path / "decks"
    name, text, tmpl = ("m.d12", SP_D12, SCF) if kind == "d12" else ("m_band.d3", BAND_D3, PROP)
    _deck(d, name, text)
    p = _mace(d, env, "submit", "--nosubmit", name)
    assert p.returncode == 0, p.stdout + p.stderr
    got = (d / (Path(name).stem + ".sh")).read_bytes()
    assert got == _template_output(tmp_path, tmpl, Path(name).stem)
    assert not list(d.glob(".*_site.*.sh"))                       # no site copy was needed


def test_default_bytes_array(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    d = tmp_path / "decks"
    for n in ("a", "b"):
        _deck(d, f"{n}.d12")
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout + p.stderr
    for n in ("a", "b"):
        assert (d / f"{n}.sh").read_bytes() == _template_output(tmp_path, SCF, n)


def test_default_bytes_local_generate_script(tmp_path, monkeypatch):
    from mace.submission import local_run
    for k in ("MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION", "MACE_CRYSTAL_MODULE", "MODULEPATH", "LMOD_CMD",
              "MODULESHOME", "BASH_ENV", "SLURM_JOB_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_MODULES_NO_PROFILE", "1")
    monkeypatch.setenv("MACE_HOME", str(REPO_ROOT))
    site._MEMO.clear()
    _deck(tmp_path / "d", "m.d12")
    script = local_run.generate_script(local_run.Deck(tmp_path / "d" / "m.d12"))
    twin = tmp_path / "twin2" / ".mace_local"
    twin.mkdir(parents=True)
    subprocess.run(["bash", str(SCF), "m"], cwd=twin, check=True, capture_output=True)
    assert script.read_bytes() == (twin / "m.sh").read_bytes()
    _toml(tmp_path)
    site._MEMO.clear()
    script = local_run.generate_script(local_run.Deck(tmp_path / "d" / "m.d12"))
    text = script.read_text()
    assert "#SBATCH -A grp_example" in text and "#SBATCH -p a,b" in text and "module load Python/3.12-x" in text
    assert not list((tmp_path / "d" / ".mace_local").glob(".*_site.*"))


def test_unchanged_text_with_built_in_values():
    for kind, t in (("scf", SCF), ("prop", PROP)):
        text = t.read_text()
        new, changes, notes = site.customize_generator_text(
            text, kind, site.Site(), site.Resolution(list(site.DEFAULTS["crystal"]), "unchecked"))
        assert (new, changes, notes) == (text, [], [])


# SHA-256 of the templates at daa5468e (the 1.1.5 base before the GUI), frozen
# here so the check also runs in a shallow clone (CI) that lacks that commit.
BASE_TEMPLATE_SHA256 = {
    "submitcrystal23.sh": "76926e20f3a1f1b7c42a3b8ec84fb55d59d434cd20f853c309458a0eab26dace",
    "submit_prop.sh": "87019c4598f2ed8944376476dad3191853dbb2350b2d88985b233c9fb6dbd42a",
}


def test_templates_unchanged_since_base():
    """The templates themselves are never edited (site settings are layered on top).
    daa5468e is the 1.1.5 base before the GUI; the templates are unchanged since."""
    for t in (SCF, PROP):
        assert hashlib.sha256(t.read_bytes()).hexdigest() == BASE_TEMPLATE_SHA256[t.name], t.name


def test_frozen_template_digests_match_the_base_commit():
    """The frozen digests are those of daa5468e, when git history is available."""
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", "daa5468e^{commit}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("base commit not in this clone (shallow); the frozen digests are checked above")
    for t in (SCF, PROP):
        old = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"daa5468e:mace/submission/{t.name}"],
                             capture_output=True, check=True).stdout
        assert hashlib.sha256(old).hexdigest() == BASE_TEMPLATE_SHA256[t.name], t.name


# ---------------------------------------------------------------- D28
def _diff_lines(a: bytes, b: bytes):
    la, lb = a.decode().splitlines(), b.decode().splitlines()
    return sorted(set(la) - set(lb)), sorted(set(lb) - set(la))


@pytest.mark.parametrize("kind", ["d12", "d3"])
def test_site_values_reach_submit(tmp_path, kind):
    env = _env(tmp_path)
    _toml(tmp_path)
    d = tmp_path / "decks"
    name, text, tmpl = ("m.d12", SP_D12, SCF) if kind == "d12" else ("m_band.d3", BAND_D3, PROP)
    _deck(d, name, text)
    p = _mace(d, env, "submit", "--nosubmit", name)
    assert p.returncode == 0, p.stdout + p.stderr
    got = (d / (Path(name).stem + ".sh")).read_bytes()
    added, removed = _diff_lines(got, _template_output(tmp_path, tmpl, Path(name).stem))
    assert added == sorted(["#SBATCH -A grp_example", "#SBATCH -p a,b", "module load Python/3.12-x"])
    assert all(l.startswith(("#SBATCH -A ", "module load Python")) for l in removed), removed
    pin = [l for l in got.decode().splitlines() if l.startswith("MACE_SUBMIT_HOME=")]
    assert pin == [f"MACE_SUBMIT_HOME='{REPO_ROOT}'"]
    assert not list(d.glob(".*_site.*.sh"))                       # the copy was removed


def test_site_values_reach_arrays(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_SLURM_ACCOUNT="grp_env")
    d = tmp_path / "decks"
    for n in ("a", "b"):
        _deck(d, f"{n}.d12")
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 0, p.stdout + p.stderr
    for n in ("a", "b"):
        assert "#SBATCH -A grp_env\n" in (d / f"{n}.sh").read_text()
    assert fake.calls("sbatch")


def test_walltime_custom_generator_gets_partition(tmp_path):
    """--walltime makes crystal.py's temporary generator in mace/submission/;
    the site values are applied to it in place (its install pin unchanged)."""
    env = _env(tmp_path)
    _toml(tmp_path)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "--walltime", "0:30:00", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    text = (d / "m.sh").read_text()
    assert "#SBATCH -t 0:30:00" in text and "#SBATCH -p a,b" in text and "#SBATCH -A grp_example" in text
    assert f"MACE_SUBMIT_HOME='{REPO_ROOT}'" in text
    assert not list(SUB.glob("*_custom_*.sh"))


@pytest.mark.parametrize("rewrite", [False, True])
def test_interactive_custom_location_and_pin(tmp_path, monkeypatch, rewrite):
    import importlib
    sys.path.insert(0, str(SUB.parent))
    crystal = importlib.import_module("mace.submission.crystal")
    for k in ("MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION", "MACE_CRYSTAL_MODULE", "MODULEPATH", "BASH_ENV"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_MODULES_NO_PROFILE", "1")
    if rewrite:
        _toml(tmp_path)
    site._MEMO.clear()
    res = {"ntasks": 32, "memory_per_cpu": "5G", "walltime": "7-00:00:00",
           "account": crystal.get_default_d12_resources()["account"], "node_exclusion": None}
    custom = crystal.create_custom_slurm_script(SCF, res)
    try:
        assert custom.parent == SUB                              # where it always was
        site.customize_file_in_place(custom, "scf")
        assert custom.parent == SUB and custom.exists()
        text = custom.read_text()
        assert ("echo '#SBATCH -p a,b' >> $1.sh" in text) == rewrite
        assert ("echo '#SBATCH -A grp_example' >> $1.sh" in text) == rewrite
        twin = tmp_path / "t"
        twin.mkdir()
        e = {k: v for k, v in os.environ.items() if k != "MACE_HOME"}
        subprocess.run(["bash", str(custom), "m"], cwd=twin, env=e, check=True, capture_output=True)
        pin = [l for l in (twin / "m.sh").read_text().splitlines() if l.startswith("MACE_SUBMIT_HOME=")]
        assert pin == [f"MACE_SUBMIT_HOME='{REPO_ROOT}'"]
    finally:
        custom.unlink()


def test_interactive_blank_answers_use_the_site_account(tmp_path):
    env = _env(tmp_path, MACE_SLURM_ACCOUNT="grp_env")
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "-i", "--nosubmit", "m.d12", stdin="y\n\n\n\n\nn\n")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "Account [grp_env]" in p.stdout
    assert "#SBATCH -A grp_env\n" in (d / "m.sh").read_text()
    assert not list(SUB.glob("*_custom_*.sh"))


def test_real_line12_trailing_space():
    text = SCF.read_text()
    line = next(l for l in text.splitlines() if l.startswith("echo '#SBATCH -A "))
    assert line.endswith(" >> $1.sh ")                         # the template's own trailing space
    s = site.Site(account="grp_x")
    new, changes, _ = site.customize_generator_text(text, "scf", s, None)
    assert "echo '#SBATCH -A grp_x' >> $1.sh \n" in new
    stripped = text.replace(line, line.rstrip())
    new2, _, _ = site.customize_generator_text(stripped, "scf", s, None)
    assert "echo '#SBATCH -A grp_x' >> $1.sh\n" in new2


# ---------------------------------------------------------------- D31
def test_idempotent_and_edited_line_kept():
    s = site.Site(account="grp_x", partition="p1", python=["Python/3.12-x"])
    r = site.Resolution(["CRYSTAL/23-other"], "site.toml")
    text = SCF.read_text()
    once, _, _ = site.customize_generator_text(text, "scf", s, r)
    twice, changes, _ = site.customize_generator_text(once, "scf", s, r)
    assert once == twice and changes == []
    edited = text.replace("echo '#SBATCH -A mendoza_q # or general' >> $1.sh", "echo '#SBATCH -A mine' >> $1.sh")
    out, _, notes = site.customize_generator_text(edited, "scf", s, r)
    assert "echo '#SBATCH -A mine' >> $1.sh" in out and any("left as written" in n for n in notes)


def test_copies_untouched(tmp_path, monkeypatch):
    """A workflow_scripts/ copy (here: the 1.1.4 template copied by a planner)
    is used as written, with site.toml present."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _toml(tmp_path)
    copy = tmp_path / "workflow_scripts" / "submitcrystal23_sp_2.sh"
    copy.parent.mkdir()
    shutil.copy2(SCF, copy)
    path, cleanup, res = site.generator_for(copy, None, tmp_path)
    assert path == copy and res is None
    assert site.customize_workflow_text(copy.read_text(), {"account": "mendoza_q", "ntasks": 32}) == copy.read_text()


def test_saved_plan_keeps_account(monkeypatch, tmp_path):
    from mace.workflow.executor import WorkflowExecutor
    from types import SimpleNamespace
    monkeypatch.setenv("MACE_SLURM_ACCOUNT", "grp_example")
    stub = SimpleNamespace(work_dir=tmp_path, isolation_mode="isolated")
    out = WorkflowExecutor.apply_script_customizations(stub, SCF.read_text(),
                                                       {"resources": {"account": "mendoza_q"}, "workflow_id": "w"})
    assert "echo '#SBATCH -A mendoza_q' >> $1.sh" in out
    out = WorkflowExecutor.apply_script_customizations(stub, SCF.read_text(), {"resources": {}, "workflow_id": "w"})
    assert "echo '#SBATCH -A grp_example' >> $1.sh" in out


def test_workflow_account_through_executor(tmp_path):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    _toml(tmp_path, '[slurm]\naccount = "grp_example"\n')
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                                        "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1))
    p = _mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    copies = sorted((d / "workflow_scripts").glob("*.sh"))
    assert copies
    for c in copies:
        assert "echo '#SBATCH -A grp_example' >> $1.sh" in c.read_text(), c.name
    (plan,) = sorted((d / "workflow_configs").glob("workflow_plan_*.json"))
    assert "grp_example" in plan.read_text()
    scripts = [c["script_text"] for c in fake.calls("sbatch")]
    assert scripts and all("#SBATCH -A grp_example" in s for s in scripts)
