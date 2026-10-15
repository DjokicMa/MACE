"""The built-in account where SLURM accounting cannot be read (1.1.5).

`resolve_account` used to keep MSU HPCC's built-in `mendoza_q` whenever
membership could not be checked. On another cluster that has SLURM but no
readable accounting (no sacctmgr, or one that lists nothing for the user)
every job script then carried `#SBATCH -A mendoza_q # or general`, which
that cluster may reject. Now, when membership cannot be checked:

* on MSU HPCC the built-in account is kept (byte-identical scripts);
* on a host with no SLURM at all (a workstation, CI: the scripts are written
  for MSU HPCC) it is kept, as before;
* on any other host with SLURM there is no account line (SLURM's default
  account), and `mace site`, the warning and the GUI's Submit preview say so.

A user's own account (site.toml, MACE_SLURM_ACCOUNT) is always written, and
a member of mendoza_q always gets it.

Behavior table (host x membership -> the -A line):

    host        member  not member  unreadable (no sacctmgr / lists nothing)
    MSU HPCC    -A mq   none        -A mendoza_q
    other SLURM -A mq   none        none
    no SLURM    -       -           -A mendoza_q
    any, user's own account set: -A <theirs>
"""
import json
import os

import pytest

from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import MSU_HOST, OTHER_HOST, SCRIPT_PATTERN
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output

from mace.submission import site

NO_ACCOUNT = "no account line (SLURM's default account); set yours with MACE_SLURM_ACCOUNT or site.toml"
SCF_LINE = "#SBATCH -A mendoza_q # or general"


@pytest.fixture(autouse=True)
def _fresh():
    site._MEMO.clear()
    site._SLURM_MEMO.clear()
    yield
    site._MEMO.clear()
    site._SLURM_MEMO.clear()


def _host_env(tmp, host, membership):
    """env (and the fake SLURM, None on a workstation) for a host type and a
    membership state: member, not_member, no_sacctmgr, empty."""
    if host == "workstation":
        return _env(tmp, PATH="/usr/bin:/bin", MACE_HOSTNAME="laptop"), None
    state = {"member": {"assoc": ["mendoza_q", "general"], "default_account": "general"},
             "not_member": {"assoc": ["grp_x"], "default_account": "grp_x"},
             "no_sacctmgr": {}, "empty": {}}[membership]
    fake = FakeSlurm(tmp / "fs", state)
    if membership == "no_sacctmgr":
        (fake.bindir / "sacctmgr").unlink()
    env = _env(tmp, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"],
               MACE_HOSTNAME={"msu": MSU_HOST, "other": OTHER_HOST}[host])
    return env, fake


# ------------------------------------------------------------- the table
@pytest.mark.parametrize("host,membership,expected", [
    ("msu", "member", "mendoza_q"),
    ("msu", "not_member", None),
    ("msu", "no_sacctmgr", "mendoza_q"),
    ("msu", "empty", "mendoza_q"),
    ("other", "member", "mendoza_q"),
    ("other", "not_member", None),
    ("other", "no_sacctmgr", None),
    ("other", "empty", None),
    ("workstation", None, "mendoza_q"),
])
def test_resolve_account_table(tmp_path, host, membership, expected):
    env, _ = _host_env(tmp_path, host, membership)
    acct, status, note = site.resolve_account(site.load(env), env)
    assert acct == expected
    if host == "other" and membership in ("no_sacctmgr", "empty"):
        assert status == "unreadable" and NO_ACCOUNT in note
        assert "mendoza" not in note.lower() and "MSU" not in note
    elif expected == "mendoza_q" and membership != "member":
        assert status == "not checked" and note == ""


@pytest.mark.parametrize("host", ["msu", "other", "workstation"])
@pytest.mark.parametrize("how", ["env", "toml"])
def test_users_own_account_is_always_written(tmp_path, host, how):
    env, _ = _host_env(tmp_path, host, "no_sacctmgr")
    if how == "env":
        env["MACE_SLURM_ACCOUNT"] = "physics"
    else:
        p = tmp_path / "site.toml"
        p.write_text('[slurm]\naccount = "physics"\n')
        env["MACE_SITE_CONFIG"] = str(p)
    assert site.resolve_account(site.load(env), env)[:2] == ("physics", "set")


def test_builtin_fallbacks_follow_the_same_rule(tmp_path):
    """account_usable (the planner's MSU "general" fallback, the Mendoza
    menu choice) answers as resolve_account does."""
    env, _ = _host_env(tmp_path, "other", "no_sacctmgr")
    assert not site.account_usable("general", env) and not site.account_usable("mendoza_q", env)
    site._SLURM_MEMO.clear()
    env, _ = _host_env(tmp_path / "m", "msu", "no_sacctmgr")
    assert site.account_usable("general", env) and site.account_usable("mendoza_q", env)
    site._SLURM_MEMO.clear()
    env, _ = _host_env(tmp_path / "w", "workstation", None)
    assert site.account_usable("mendoza_q", env)


# ------------------------------------------------------------- the real CLI
@pytest.mark.parametrize("membership", ["no_sacctmgr", "empty"])
def test_other_cluster_unreadable_submit_has_no_account_line(tmp_path, membership):
    env, fake = _host_env(tmp_path, "other", membership)
    d = tmp_path / "decks"
    _deck(d, "m.d12", SP_D12)
    p = _mace(d, env, "submit", "m.d12")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    (call,) = fake.calls("sbatch")
    assert "#SBATCH -A" not in call["script_text"] and "--account" not in call["script_text"]
    assert not SCRIPT_PATTERN.search(call["script_text"].split("module load")[0])
    out = p.stdout + p.stderr
    assert out.count(NO_ACCOUNT) == 1, out[-3000:]


def test_other_cluster_unreadable_mace_site_says_so(tmp_path):
    env, _ = _host_env(tmp_path, "other", "no_sacctmgr")
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Account:     (your SLURM default account)" in r.stdout
    assert NO_ACCOUNT in r.stdout
    assert "mendoza_q" not in r.stdout and "account not checked" not in r.stdout
    d = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert d["account"]["value"] is None and d["account"]["check"] == "unreadable"
    assert d["account_check"]["status"] == "unreadable"
    assert "not a member" not in d["account"]["source"]


def test_other_cluster_unreadable_gui_preview_needs_no_tick(tmp_path):
    env, _ = _host_env(tmp_path, "other", "empty")
    chk = site.describe(env, check_account=True)["account_check"]
    from mace.gui import submit as gsub
    p = gsub.Preview(id="", created=0, where="slurm", cmds=[], decks=[], fp={}, result={}, needs={},
                     scripts={}, modified=[], tracked=False)

    class _Store:
        def sign(self, fp):
            return "x"

        def put(self, p):
            pass
    assert not gsub.finalize(_Store(), p, chk)["needs"].get("account")
    js = (site.SUBMISSION_DIR.parent / "gui" / "static" / "js" / "lazy" / "submit_dialog.js").read_text()
    assert '"unreadable"' in js and NO_ACCOUNT in js


def test_other_cluster_unreadable_workflow_scripts(tmp_path):
    """The planned workflow stores no account; every step script and the
    workflow_scripts copies carry no -A line."""
    env, fake = _host_env(tmp_path, "other", "no_sacctmgr")
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
        assert "#SBATCH -A" not in s
    for c in sorted((d / "workflow_scripts").glob("*.sh")):
        assert "#SBATCH -A" not in c.read_text(), c.name


# ------------------------------------------------------------- MSU: every byte
@pytest.mark.parametrize("membership", ["no_sacctmgr", "empty"])
@pytest.mark.parametrize("kind", ["d12", "d3"])
def test_msu_unreadable_keeps_every_byte(tmp_path, kind, membership):
    env, fake = _host_env(tmp_path, "msu", membership)
    d = tmp_path / "decks"
    name, text, tmpl = ("m.d12", SP_D12, SCF) if kind == "d12" else ("m_band.d3", BAND_D3, PROP)
    _deck(d, name, text)
    p = _mace(d, env, "submit", "--nosubmit", name)
    assert p.returncode == 0, p.stdout + p.stderr
    stem = os.path.splitext(name)[0]
    got = (d / f"{stem}.sh").read_bytes()
    assert got == _template_output(tmp_path, tmpl, stem)
    assert (SCF_LINE if kind == "d12" else "#SBATCH -A mendoza_q").encode() in got
    assert NO_ACCOUNT not in p.stdout + p.stderr
    r = _mace(tmp_path, env, "site")
    assert "Account:     mendoza_q  (built-in)" in r.stdout and "account not checked" in r.stdout


def test_workstation_unchanged(tmp_path):
    env, _ = _host_env(tmp_path, "workstation", None)
    d = tmp_path / "decks"
    _deck(d, "m.d12", SP_D12)
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "m.sh").read_bytes() == _template_output(tmp_path, SCF, "m")
    assert NO_ACCOUNT not in p.stdout + p.stderr
