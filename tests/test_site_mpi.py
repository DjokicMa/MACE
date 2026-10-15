"""[mpi]: how CRYSTAL is started (PLAN_G G2, D6, A11), through the real mace_cli.

The default (mpirun) keeps every byte; srun gets its own block (mace run keeps
mpirun, it has no SLURM step to srun into); [mpi] env values are data that
bash cannot run: every shell metacharacter is refused, as are the variable
names that change how the job script, bash or MACE run."""
import json
import shutil
import subprocess

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site


def _submit(tmp, env, deck="m.d12"):
    d = tmp / "decks"
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    (d / (deck.rsplit(".", 1)[0] + ".sh")).unlink(missing_ok=True)
    return _mace(d, env, "submit", "--nosubmit", deck), d / (deck.rsplit(".", 1)[0] + ".sh")


def test_default_launcher_keeps_the_bytes(tmp_path):
    _toml(tmp_path, '[mpi]\nlauncher = "mpirun"\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    assert sh.read_bytes() == _template_output(tmp_path, SCF, "m")


@pytest.mark.parametrize("deck,exe", [("m.d12", "Pcrystal"), ("m_band.d3", "Pproperties")])
def test_srun_block(tmp_path, deck, exe):
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix"\nenv = ["OMPI_MCA_btl=^openib", '
                    '"I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so"]\n')
    p, sh = _submit(tmp_path, _env(tmp_path), deck)
    assert p.returncode == 0, p.stdout + p.stderr
    text = sh.read_text()
    block = ["export OMPI_MCA_btl=^openib", "export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so",
             'if [ -n "$MACE_LOCAL_RUN" ] && [ -z "$SLURM_JOB_ID" ]; then',
             f'  I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n "$SLURM_NTASKS" "$EBROOTCRYSTAL/bin/{exe}" 2>&1 >& "$DIR/${{JOB}}.out"',
             "else", f'  srun --mpi=pmix "$EBROOTCRYSTAL/bin/{exe}" 2>&1 >& "$DIR/${{JOB}}.out"', "fi"]
    lines = text.splitlines()
    i = lines.index(block[0])
    assert lines[i:i + len(block)] == block
    assert lines[i - 1] == site.LAUNCH_COMMENT
    assert site.LAUNCH_LINES["scf" if deck.endswith(".d12") else "prop"] not in lines
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0


def test_cpus_per_task_with_srun(tmp_path):
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\n[resources]\ncpus_per_task = 4\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    text = sh.read_text()
    assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in text
    assert '  srun -c "$SLURM_CPUS_PER_TASK" "$EBROOTCRYSTAL/bin/Pcrystal" 2>&1 >& "$DIR/${JOB}.out"' in text
    assert "#SBATCH --cpus-per-task=4" in text


def test_env_launcher(tmp_path):
    p, sh = _submit(tmp_path, _env(tmp_path, MACE_MPI_LAUNCHER="srun"))
    assert '  srun "$EBROOTCRYSTAL/bin/Pcrystal" 2>&1 >& "$DIR/${JOB}.out"' in sh.read_text()


BAD_VALUES = [";id", "$(id)", "`id`", "a|b", "a&b", "a>b", "a<b", "a b", "a'b", "a\\b", "a\nb", "*",
              "${PATH}", 'a"b', "~x", "a?b", "(a)", "a\tb", "a\x00b"]


@pytest.mark.parametrize("value", BAD_VALUES)
def test_env_value_metacharacters_are_refused(tmp_path, value):
    _toml(tmp_path, "[mpi]\nenv = [" + json.dumps(f"X_FLAG={value}") + "]\n")
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2, p.stdout + p.stderr
    assert "[mpi] env" in p.stdout and not sh.exists()


@pytest.mark.parametrize("name", ["PATH", "LD_PRELOAD", "MACE_HOME", "SLURM_NTASKS", "DIR", "BASH_ENV",
                                  "PYTHONPATH", "IFS", "1BAD", "A-B"])
def test_refused_names(tmp_path, name):
    _toml(tmp_path, f'[mpi]\nenv = ["{name}=x"]\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and not sh.exists(), p.stdout


@pytest.mark.parametrize("launcher,msg", [("mpiexec -n 4 {exe}", "custom launchers are planned for MACE 1.2"),
                                          ("mpiexec", "launcher (mpirun or srun)")])
def test_custom_launcher_is_refused(tmp_path, launcher, msg):
    _toml(tmp_path, f'[mpi]\nlauncher = "{launcher}"\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and msg in p.stdout and not sh.exists()


@pytest.mark.parametrize("bad", ["pmix;id", "openmpi", ""])
def test_srun_mpi_enum(tmp_path, bad):
    _toml(tmp_path, f'[mpi]\nlauncher = "srun"\nsrun_mpi = "{bad}"\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and not sh.exists()


def test_area_f_refused_names_are_a_subset():
    cfg = pytest.importorskip("mace.cluster.config")
    prefixes = getattr(cfg, "REFUSED_ENV_PREFIXES", ())
    names = getattr(cfg, "REFUSED_ENV_NAMES", ())
    if not prefixes and not names:
        pytest.skip("mace.cluster.config has no refused-name lists here")
    assert all(any(p.startswith(q) for q in site.REFUSED_ENV_PREFIXES) for p in prefixes)
    assert set(names) <= site.REFUSED_ENV_NAMES | {n for n in names if n.startswith(site.REFUSED_ENV_PREFIXES)}


def test_mace_run_with_srun_still_runs_mpirun(tmp_path):
    """mace run's job script takes the local branch: mpirun on this node."""
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix"\n')
    env = _env(tmp_path)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    s = site.load(env)
    text = site.customize_generator_text(SCF.read_text(), "scf", s, None)[0]
    script = tmp_path / "gen.sh"
    script.write_text(text)
    subprocess.run(["bash", str(script), "m"], cwd=d, check=True, capture_output=True)
    job = (d / "m.sh").read_text()
    # run only the launch block with MACE_LOCAL_RUN set and a stub mpirun/srun
    lines = job.splitlines()
    i = lines.index('if [ -n "$MACE_LOCAL_RUN" ] && [ -z "$SLURM_JOB_ID" ]; then')
    stub = tmp_path / "bin"
    stub.mkdir()
    for tool in ("mpirun", "srun"):
        (stub / tool).write_text(f"#!/bin/bash\necho {tool} \"$@\" > {tmp_path}/ran\n")
        (stub / tool).chmod(0o755)
    body = "\n".join(lines[i:i + 5])
    r = subprocess.run(["bash", "-c", body], env={"PATH": f"{stub}:/usr/bin:/bin", "MACE_LOCAL_RUN": "1",
                                                   "DIR": str(tmp_path), "JOB": "m", "SLURM_NTASKS": "2",
                                                   "EBROOTCRYSTAL": "/x"})
    assert r.returncode == 0
    assert (tmp_path / "ran").read_text().startswith("mpirun -n 2 /x/bin/Pcrystal")
    r = subprocess.run(["bash", "-c", body], env={"PATH": f"{stub}:/usr/bin:/bin", "SLURM_JOB_ID": "5",
                                                   "DIR": str(tmp_path), "JOB": "m", "EBROOTCRYSTAL": "/x"})
    assert (tmp_path / "ran").read_text().startswith("srun --mpi=pmix /x/bin/Pcrystal")


def test_srun_error_output_is_an_error(tmp_path):
    """srun's own lines after a CRYSTAL error leave the .out categorized as failed."""
    import os
    from pathlib import Path
    from conftest import TEST_DATA
    from mace.completion_checker import categorize_output_file
    corpus = TEST_DATA if TEST_DATA.is_dir() else Path(os.environ.get("MACE_TEST_DATA", "/nonexistent"))
    src = corpus / "FAILED_QA" / "3_dia3_neighb_atoms_too_close.out"
    if not src.is_file():
        pytest.skip("test/ data corpus not present (or MACE_TEST_DATA not set)")
    cat, _ = categorize_output_file(src)
    if cat.startswith("complete"):
        pytest.skip("the first .out with ERROR is categorized complete")
    out = tmp_path / "job.out"
    shutil.copy(src, out)
    with open(out, "a") as fh:
        fh.write("srun: error: node-1: task 3: Exited with exit code 1\n"
                 "srun: Terminating StepId=5000.0\n")
    cat2, _ = categorize_output_file(out)
    assert not cat2.startswith("complete"), cat2
