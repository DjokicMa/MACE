"""Whole fake clusters (PLAN_G A3, A5, A8, A11), each driven through the real
mace_cli with the fake SLURM and fake module system, submitting for real
(to the fake sbatch):

* an OpenMPI cluster that starts CRYSTAL with srun, Tcl modules, a required
  QOS and node features, its admins' group file and a user file on top;
* a cluster without any module system, CRYSTAL from a directory;
* a small MaxSubmitJobs, GPUs requested the two ways sites spell it;
* shell-injection attempts through every executed key, from a group file.
"""
import json
import re
import subprocess

import pytest

from fake_modules import FakeModules
from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import ASSOC_MGR, PARTITIONS, SCRIPT_PATTERN
from test_site_jobscript import _deck, _env, _mace, _toml

# the example of docs (INSTALLATION.md, "Using MACE on another cluster")
GENERIC_SITE = """\
[slurm]
account    = "my_lab"
partition  = "compute"
qos        = "normal"
constraint = "epyc"
extra      = ["--exclusive"]
[resources]
ntasks      = 64
nodes       = 1
mem_per_cpu = "3900M"
[resources.properties]
ntasks = 32
mem    = "120G"
[mpi]
launcher = "srun"
srun_mpi = "pmix"
[modules]
crystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]
python  = ["Python/3.11.5"]
[paths]
scratch = "/scratch/$USER"
[local_run]
max_time = "4h"
"""
SINFO = "".join(f"c{i:03d}|idle|2|32|2|128|256000|epyc,ib|(null)|compute*\n" for i in range(4))


def _openmpi_cluster(tmp, group_text=None, user_text=None):
    fake = FakeSlurm(tmp / "fs", {"assoc": ["my_lab"], "default_account": "my_lab", "qos": ["normal"],
                                  "partitions": "PartitionName=compute AllowAccounts=ALL AllowQos=normal "
                                                "MaxTime=3-00:00:00 State=UP Default=YES\n",
                                  "assoc_mgr": ASSOC_MGR.replace("grp_x", "my_lab"), "sinfo": SINFO,
                                  "srun_mpi_list": "MPI plugin types are...\n\tnone\n\tpmi2\n\tpmix\n"})
    fm = FakeModules(tmp / "fm", {"foss/2023a": {"bins": ()},
                                  "CRYSTAL/23-foss-2023a": {"root": True, "needs": "foss/2023a"},
                                  "Python/3.11.5": {"bins": ()}}, python=False)
    env = _env(tmp, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="login2.hpc.example.org")
    if group_text:
        g = tmp / "admins"
        g.mkdir()
        (g / "site.toml").write_text(group_text)
        (g / "site.toml").chmod(0o644)
        g.chmod(0o755)
        env["MACE_SITE_GROUP_CONFIG"] = str(g / "site.toml")
    if user_text:
        _toml(tmp, user_text)
    return fm.env(env), fake, fm


def _submit_both(tmp, env):
    d = tmp / "decks"
    _deck(d, "m.d12", SP_D12)
    _deck(d, "m_band.d3", BAND_D3)
    (d / "m_band.f9").write_text("fort9\n")
    out = {}
    for deck in ("m.d12", "m_band.d3"):
        p = _mace(d, env, "submit", deck)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        out[deck] = p.stdout + p.stderr
    return d, out


def test_openmpi_srun_tcl_group_and_user_files(tmp_path):
    # the admins' file holds the cluster's choices; the user's file only the account
    env, fake, fm = _openmpi_cluster(tmp_path, group_text=GENERIC_SITE.replace('"my_lab"', '"shared_lab"'),
                                     user_text='[slurm]\naccount = "my_lab"\n')
    d, out = _submit_both(tmp_path, env)
    scripts = {c["script"]: c["script_text"] for c in fake.calls("sbatch")}
    scf, prop = scripts["m.sh"], scripts["m_band.sh"]
    for s in (scf, prop):
        assert not SCRIPT_PATTERN.search(s), SCRIPT_PATTERN.search(s)
        sb = [l for l in s.splitlines() if l.startswith("#SBATCH")]
        i = sb.index("#SBATCH -A my_lab")                      # the user's account wins over the group's
        assert sb[i + 1:i + 5] == ["#SBATCH -p compute", "#SBATCH --qos=normal", "#SBATCH --constraint=epyc",
                                   "#SBATCH --exclusive"]
        assert "module load foss/2023a\nmodule load CRYSTAL/23-foss-2023a" in s
        assert "module load Python/3.11.5" in s
        assert 'if [ -d "/scratch/$MACE_USER" ]; then' in s
        assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
    assert "#SBATCH --ntasks=64" in scf and "#SBATCH --mem-per-cpu=3900M" in scf
    assert "#SBATCH --ntasks=32" in prop and "#SBATCH --mem=120G" in prop and "--mem-per-cpu" not in prop
    assert '  srun --mpi=pmix "$EBROOTCRYSTAL/bin/Pcrystal" 2>&1 >& "$DIR/${JOB}.out"' in scf
    assert '  srun --mpi=pmix "$EBROOTCRYSTAL/bin/Pproperties" 2>&1 >& "$DIR/${JOB}.out"' in prop
    # mace site shows where each value comes from, with the group file's owner
    j = json.loads(_mace(tmp_path, env, "site", "--json").stdout)
    assert j["account"]["source"] == "site.toml" and j["partition"]["source"] == "group"
    assert j["config"]["group"][0]["path"] == str(tmp_path / "admins" / "site.toml")
    assert j["resources"]["scf"]["fit"]["ok"] and j["resources"]["scf"]["fit"]["node_cores"] == 64
    rep = json.loads(_mace(tmp_path, env, "site", "check", "--json").stdout)
    st = {i["item"]: i["status"] for i in rep["items"]}
    assert st["Job script"] == "ok" and st["MPI launch"] == "ok" and st["QOS/constraint"] == "ok", rep


def test_no_module_system_crystal_from_a_directory(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"assoc": ["lab"], "default_account": "lab", "partitions": PARTITIONS,
                                       "assoc_mgr": ASSOC_MGR.replace("grp_x", "lab")})
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="node1.uni.example")
    _toml(tmp_path, '[slurm]\naccount = "lab"\n[crystal]\nbin_dir = "/opt/crystal23/bin"\n[modules]\npython = []\n'
                    'purge = false\n')
    d, out = _submit_both(tmp_path, env)
    for c in fake.calls("sbatch"):
        s = c["script_text"]
        assert "module load" not in s and "# MACE: module purge skipped" in s
        assert 'export EBROOTCRYSTAL="/opt/crystal23"' in s
        assert re.search(r'mpirun -n "\$SLURM_NTASKS" "/opt/crystal23/bin/P(crystal|properties)" 2>&1', s)
        assert not SCRIPT_PATTERN.search(s)


def test_small_max_submit_and_gpu_spellings(tmp_path):
    fake = FakeSlurm(tmp_path / "fs", {"assoc": ["lab"], "default_account": "lab", "partitions": PARTITIONS,
                                       "assoc_mgr": ASSOC_MGR.replace("grp_x", "lab")
                                                            .replace("MaxSubmitJobs=200", "MaxSubmitJobs=40")})
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="node1.uni.example")
    for extra, want in (('["--gres=gpu:1"]', "#SBATCH --gres=gpu:1"), ('["--gpus=1"]', "#SBATCH --gpus=1"),
                        ('["--gpus-per-node=2"]', "#SBATCH --gpus-per-node=2")):
        _toml(tmp_path, f'[slurm]\naccount = "lab"\nextra = {extra}\n')
        d = tmp_path / "decks"
        for f in d.glob("*.sh") if d.exists() else []:
            f.unlink()
        _deck(d, "m.d12")
        p = _mace(d, env, "submit", "--nosubmit", "m.d12")
        assert p.returncode == 0 and want in (d / "m.sh").read_text()
        assert "--max-jobs 40 --reserve 4 --max-submit 10" in (d / "m.sh").read_text()


INJECTIONS = [
    '[mpi]\nenv = ["A=$(touch /tmp/pwned)"]\n',
    '[mpi]\nenv = ["A=`id`"]\n',
    '[mpi]\nenv = ["A=1;id"]\n',
    '[mpi]\nlauncher = "srun; id"\n',
    '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix && id"\n',
    '[crystal]\nbin_dir = "/opt/c;id"\n',
    '[crystal]\nscf = "Pcrystal|id"\n',
    '[paths]\nscratch = "/scratch/$(id)"\n',
    '[paths]\nscratch = "/s\' >> x; echo \'"\n',
    '[slurm]\nextra = ["--comment=x\' ; id ; echo \'"]\n',
    '[slurm]\nqos = "n\\nid"\n',
    '[modules]\ncrystal = "CRYSTAL/23;id"\n',
    '[resources]\nmem = "1G;id"\n',
]


@pytest.mark.parametrize("text", INJECTIONS)
def test_injection_attempts_from_a_group_file_are_rejected(tmp_path, text):
    env, fake, fm = _openmpi_cluster(tmp_path, group_text=text)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "m.d12")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "nothing was generated" in p.stdout or "not a valid" in p.stdout
    assert str(tmp_path / "admins" / "site.toml") in p.stdout
    assert not fake.calls("sbatch") and not (d / "m.sh").exists()
