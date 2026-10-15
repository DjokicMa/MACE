"""[resources]: the job size (PLAN_G G1, D8, A3, A14), through the real mace_cli.

Without a key every script keeps its bytes (test_site_g_goldens.py,
test_site_jobscript.py::test_default_bytes). A key changes exactly its line
of the shipped template; under -i, --walltime and the workflow planner the
site value is the editor's default and the editor's answer is never
overwritten, whichever memory form the site chose."""
import json

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site


def _sbatch(text):
    return [l for l in text.splitlines() if l.startswith("#SBATCH")]


def _submit(tmp, env, deck, *args, stdin=""):
    d = tmp / "decks"
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    (d / (deck.rsplit(".", 1)[0] + ".sh")).unlink(missing_ok=True)     # never reuse an earlier job script
    p = _mace(d, env, "submit", "--nosubmit", *args, deck, stdin=stdin)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return (d / (deck.rsplit(".", 1)[0] + ".sh")).read_text(), p


def _diff(tmp, got, template, job):
    """The #SBATCH lines that differ from the template's own output."""
    ref = _template_output(tmp, template, job).decode()
    return sorted(set(_sbatch(got)) - set(_sbatch(ref))), sorted(set(_sbatch(ref)) - set(_sbatch(got)))


def test_both_kinds_and_properties_level(tmp_path):
    _toml(tmp_path, '[resources]\nntasks = 16\nmem_per_cpu = "2G"\n')
    env = _env(tmp_path)
    scf, _ = _submit(tmp_path, env, "m.d12")
    assert _diff(tmp_path, scf, SCF, "m") == (["#SBATCH --mem-per-cpu=2G", "#SBATCH --ntasks=16"],
                                             ["#SBATCH --mem-per-cpu=5G", "#SBATCH --ntasks=32"])
    prop, _ = _submit(tmp_path, env, "m_band.d3")
    assert _diff(tmp_path, prop, PROP, "m_band") == (["#SBATCH --mem-per-cpu=2G", "#SBATCH --ntasks=16"],
                                                    ["#SBATCH --mem=80G", "#SBATCH --ntasks=8"])
    _toml(tmp_path, '[resources]\nntasks = 16\nmem_per_cpu = "2G"\nnodes = 1\ncpus_per_task = 1\n'
                    '[resources.properties]\nmem = "40G"\n')
    prop, _ = _submit(tmp_path, env, "m_band.d3")
    assert "#SBATCH --mem=40G" in prop and "--mem-per-cpu" not in prop and "#SBATCH --ntasks=16" in prop


def test_env_overrides_and_clears_the_other_form(tmp_path):
    _toml(tmp_path, '[resources]\nmem_per_cpu = "2G"\n[resources.properties]\nmem_per_cpu = "3G"\n')
    env = _env(tmp_path, MACE_SLURM_MEM="100G", MACE_SLURM_NTASKS="8")
    s = site.load(env)
    assert s.effective_resources("scf")["mem_form"] == "mem" and s.effective_resources("prop")["mem"] == "100G"
    scf, _ = _submit(tmp_path, env, "m.d12")
    assert "#SBATCH --mem=100G" in scf and "--mem-per-cpu" not in scf and "#SBATCH --ntasks=8" in scf


def test_nodes_and_cpus_per_task_lines(tmp_path):
    _toml(tmp_path, '[resources]\nnodes = 2\ncpus_per_task = 4\n')
    scf, _ = _submit(tmp_path, _env(tmp_path), "m.d12")
    assert _diff(tmp_path, scf, SCF, "m") == (["#SBATCH --cpus-per-task=4", "#SBATCH -N 2"],
                                             ["#SBATCH --cpus-per-task=1", "#SBATCH -N 1"])
    assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in scf
    r = _mace(tmp_path, _env(tmp_path), "site")
    assert "nodes > 1: the job needs a scratch directory shared by all nodes" in r.stdout


def test_both_memory_forms_at_one_level_are_refused(tmp_path):
    _toml(tmp_path, '[slurm]\naccount = "x"\n[resources]\nmem = "1G"\nmem_per_cpu = "1G"\n')
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, _env(tmp_path), "submit", "--nosubmit", "m.d12")
    assert p.returncode == 2 and "set mem or mem_per_cpu, not both" in p.stdout
    assert str(tmp_path / "xdg" / "mace" / "site.toml") in p.stdout and not (d / "m.sh").exists()


@pytest.mark.parametrize("bad", ['ntasks = 0', 'ntasks = 5000', 'nodes = "two"', 'mem = "4 G"',
                                 "mem_per_cpu = \"1G'\"", 'cpus_per_task = true'])
def test_invalid_values(tmp_path, bad):
    _toml(tmp_path, f"[resources]\n{bad}\n")
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, _env(tmp_path), "submit", "--nosubmit", "m.d12")
    assert p.returncode == 2, p.stdout
    assert "[resources]" in p.stdout and not (d / "m.sh").exists()


# ---------------------------------------------------------------- -i / --walltime (D8)
def test_interactive_cross_form_scf(tmp_path):
    _toml(tmp_path, '[resources]\nmem = "250G"\n')
    env = _env(tmp_path)
    sh, p = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n\n200G\n\n\nn\n")
    assert "Memory per node (total) [250G]" in p.stdout
    assert "#SBATCH --mem=200G" in sh and "--mem-per-cpu" not in sh
    sh, p = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n\n\n\n\nn\n")
    assert "#SBATCH --mem=250G" in sh and "--mem-per-cpu" not in sh


def test_interactive_cross_form_prop(tmp_path):
    _toml(tmp_path, '[resources]\nmem_per_cpu = "2G"\n')
    sh, p = _submit(tmp_path, _env(tmp_path), "m_band.d3", "-i", stdin="y\n\n3G\n\n\nn\n")
    assert "Memory per CPU [2G]" in p.stdout
    assert "#SBATCH --mem-per-cpu=3G" in sh and "--mem=" not in sh


def test_interactive_cores_typed_equal_to_the_template_literal(tmp_path):
    _toml(tmp_path, '[resources]\nntasks = 16\n')
    env = _env(tmp_path)
    sh, _ = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n32\n\n\n\nn\n")
    assert "#SBATCH --ntasks=32" in sh                  # the answer, not the site's 16
    sh, _ = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n\n\n\n\nn\n")
    assert "#SBATCH --ntasks=16" in sh
    _toml(tmp_path, '[resources]\nntasks = 256\n')
    sh, _ = _submit(tmp_path, env, "m.d12", "-i", stdin="y\n256\n\n\n\nn\n")
    assert "#SBATCH --ntasks=256" in sh


def test_walltime_alone_keeps_the_site_size(tmp_path):
    _toml(tmp_path, '[resources]\nntasks = 16\nmem = "250G"\n')
    sh, _ = _submit(tmp_path, _env(tmp_path), "m.d12", "--walltime", "2:00:00")
    lines = _sbatch(sh)
    assert {"#SBATCH --ntasks=16", "#SBATCH --mem=250G", "#SBATCH -t 2:00:00"} <= set(lines)
    assert not any("--mem-per-cpu" in l for l in lines)


# ---------------------------------------------------------------- workflows
def _plan_scripts(tmp, env, template):
    d = tmp / "decks"
    _deck(d, "m_opt.d12", SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1))
    p = _mace(tmp, env, "submit", "--progress", template, "--nosubmit", "decks")
    if p.returncode != 0:
        p = _mace(tmp, env, "submit", "--progress", template, "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return {c.name: c.read_text() for c in sorted((d / "workflow_scripts").glob("*.sh"))}, d


def test_workflow_memory_forms(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[resources]\nmem = "250G"\n[resources.properties]\nmem_per_cpu = "2G"\n')
    scripts, d = _plan_scripts(tmp_path, env, "full_electronic")
    band = [t for n, t in scripts.items() if "band" in n.lower()]
    opt = [t for n, t in scripts.items() if "opt" in n.lower()]
    assert band and opt, list(scripts)
    for t in band:
        assert "echo '#SBATCH --mem-per-cpu=2G' >> $1.sh" in t and "--mem=" not in t
    for t in opt:
        assert "echo '#SBATCH --mem=250G' >> $1.sh" in t and "--mem-per-cpu" not in t
    for c in fake.calls("sbatch"):
        s = c["script_text"]
        assert ("--mem-per-cpu=2G" in s) != ("--mem=250G" in s), _sbatch(s)


def test_workflow_plan_without_the_keys_is_unchanged(tmp_path):
    """A plan made before PLAN_G (no site keys) gives the same workflow text."""
    text = SCF.read_text()
    old = {"account": None, "ntasks": 32, "walltime": "7-00:00:00", "memory_per_cpu": "5G"}
    assert site.customize_workflow_text(text, old) == site.customize_workflow_text(text, dict(old))
    assert site.customize_workflow_text(text, {}) == text


# ---------------------------------------------------------------- detection (D3)
SINFO = "".join(f"n{i:03d}|idle|2|8|2|32|128000|epyc,ib|(null)|short*\n" for i in range(4)) \
    + "big01|idle|2|32|1|64|512000|epyc|gpu:a100:4|short*\n"


def test_fit_uses_physical_cores_and_writes_nothing(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(sinfo=SINFO.replace("big01|idle|2|32|1|64|512000|epyc|gpu:a100:4|short*\n", ""))
    r = _mace(tmp_path, env, "site", "--json")
    d = json.loads(r.stdout)
    f = d["resources"]["scf"]["fit"]
    assert f["node_cores"] == 16 and f["suggest"]["ntasks"] == 16 and not f["ok"]
    assert f["suggest"]["mem_per_cpu"] == f"{int(0.95 * 128000 / 16)}M"
    text = _mace(tmp_path, env, "site").stdout
    assert "does NOT fit: the largest node of partition short has 16 cores" in text
    assert "set [resources] ntasks = 16" in text
    sh, _ = _submit(tmp_path, env, "m.d12")
    assert "#SBATCH --ntasks=32" in sh                  # offered, never applied
    fake.update(sinfo=SINFO)
    site._SLURM_MEMO.clear()
    d = json.loads(_mace(tmp_path, env, "site", "--json", "--refresh").stdout)
    assert d["resources"]["scf"]["fit"]["ok"] and d["resources"]["scf"]["fit"]["node_cores"] == 64
