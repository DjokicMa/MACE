"""`mace run`: CRYSTAL decks run on this node through their normal job script.

Everything goes through mace_cli as a subprocess. CRYSTAL is a stub tree
($EBROOTCRYSTAL/bin/{Pcrystal,Pproperties,crystal,properties}) that records
what it read and writes what CRYSTAL would; mpirun is a stub that starts the
binary in a process group of its own, as Intel MPI's hydra does; `python` is a
stub that records a completion callback. The job scripts themselves are the
ones the shipped generators write, run under `bash --login`.
"""
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _module_in_login_shell() -> bool:
    r = subprocess.run(["bash", "--login", "-c", "type module"], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin",
                                                            "HOME": "/nonexistent"})
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    _module_in_login_shell(),
    reason="a login shell here has `module`: the job script would load the real CRYSTAL")

DECK = ("diamond\nCRYSTAL\n0 0 0\n227\n3.567\n1\n6 0.125 0.125 0.125\nEND\n"
        "6 0\nEND\nDFT\nPBE\nEND\nSHRINK\n4 4\nEND\n")
BAND = "BAND\ndia\n1 4 20 1 8 1 0\n0 0 0 2 0 2\nEND\n"
DOSS = "NEWK\n12 24\n1 0\nDOSS\n0 100 1 8 1 12 0\nEND\n"
ECH3 = "ECH3\n100\nEND\n"

CRYSTAL_STUB = r"""#!/bin/bash
# Stand-in for a CRYSTAL binary: run in $scratch/$JOB with INPUT there.
name=$(basename "$0")
job=$(basename "$PWD")
echo $$ > "$STUB_DIR/$job.pid"
n=$(wc -c)                                     # what arrived on stdin
echo "$name job=$job cwd=$PWD stdin=$n" >> "$STUB_LOG"
cp INPUT "$STUB_DIR/$job.INPUT"
for f in fort.9 fort.20 fort.34; do [ -f "$f" ] && cp "$f" "$STUB_DIR/$job.$f"; done
if [ -n "$STUB_SLEEP" ] && { [ -z "$STUB_SLEEP_JOB" ] || [ "$STUB_SLEEP_JOB" = "$job" ]; }; then
  [ -n "$STUB_IGNORE_TERM" ] && trap '' TERM
  echo "sleeping" >> "$STUB_DIR/$job.sleeping"
  sleep "$STUB_SLEEP"
fi
echo " EEEEEEEEEE STARTING  stub $name for $job"
if [ "$STUB_FAIL_JOB" = "$job" ]; then
  echo " ${STUB_FAIL_TEXT:-ERROR **** STUB **** deliberate failure}"
  exit 1
fi
case "$name" in
  *crystal) echo "fort9 of $job" > fort.9 ;;
  *properties)
    grep -q "^BAND" INPUT && { echo "band of $job" > BAND.DAT; echo "f25 of $job" > fort.25; }
    grep -q "^DOSS" INPUT && echo "doss of $job" > DOSS.DAT
    grep -q "^ECH3" INPUT && echo "dens of $job" > DENS_CUBE.DAT ;;
esac
echo " EEEEEEEEEE TERMINATION  DATE 02 10 2026"
echo "    TOTAL CPU TIME =        1.000"
"""

MPIRUN_STUB = r"""#!/bin/bash
# Stand-in for mpirun: the binary in a process group of its own (same session),
# TERM handed on to that group - as Intel MPI's hydra does.
echo "mpirun $*" >> "$STUB_LOG"
shift 2
set -m
"$@" &
child=$!
trap 'kill -TERM -- -$child 2>/dev/null' TERM INT HUP
wait $child; rc=$?
while kill -0 $child 2>/dev/null; do wait $child; rc=$?; done
exit $rc
"""

LOGGER = """#!/bin/bash
echo "$(basename "$0") $*" >> "$STUB_LOG"
"""

# squeue -h -u USER -o "%j %i %T": the jobs listed in $STUB_SQUEUE.
SQUEUE = """#!/bin/bash
echo "squeue $*" >> "$STUB_LOG"
[ -n "$STUB_SQUEUE" ] && printf "%s\\n" "$STUB_SQUEUE"
exit 0
"""


def _exe(path: Path, text: str):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def box(tmp_path):
    """Stub tree + environment; decks go into box.work."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _exe(stubs / "mpirun", MPIRUN_STUB)
    for name in ("python", "sbatch"):
        _exe(stubs / name, LOGGER)
    _exe(stubs / "squeue", SQUEUE)
    ebroot = tmp_path / "ebroot"
    (ebroot / "bin").mkdir(parents=True)
    for name in ("Pcrystal", "Pproperties", "crystal", "properties"):
        _exe(ebroot / "bin" / name, CRYSTAL_STUB)
    for d in ("home", "work", "stubdir", "xdg"):
        (tmp_path / d).mkdir()

    class Box:
        root = tmp_path
        work = tmp_path / "work"
        stubdir = tmp_path / "stubdir"
        scratch = tmp_path / "scratch"
        log = tmp_path / "stub.log"
        env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
               "SCRATCH": str(tmp_path / "scratch"), "EBROOTCRYSTAL": str(ebroot),
               "STUB_LOG": str(tmp_path / "stub.log"), "STUB_DIR": str(tmp_path / "stubdir"),
               "USER": "mace_no_such_user_zz", "XDG_RUNTIME_DIR": str(tmp_path / "xdg"),
               "MACE_NO_BANNER": "1", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}

        def deck(self, name, text=DECK, where=None):
            p = (where or self.work) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return p

        def args(self, *args, **env):
            e = dict(self.env)
            for k, v in env.items():
                if v is None:
                    e.pop(k, None)
                else:
                    e[k] = v
            return [sys.executable, str(MACE_CLI), "run", *map(str, args)], e

        def run(self, *args, timeout=120, input=None, preexec_fn=None, **env):
            cmd, e = self.args(*args, **env)
            r = subprocess.run(cmd, cwd=str(self.work), env=e, capture_output=True, text=True,
                               timeout=timeout, input=input, preexec_fn=preexec_fn,
                               stdin=None if input is not None else subprocess.DEVNULL)
            r.text = r.stdout + r.stderr
            return r

        def start(self, *args, preexec_fn=None, **env):
            cmd, e = self.args(*args, **env)
            return subprocess.Popen(cmd, cwd=str(self.work), env=e, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    preexec_fn=preexec_fn)

        def stub_log(self):
            return self.log.read_text() if self.log.exists() else ""

        def binary_runs(self):
            return [l for l in self.stub_log().splitlines()
                    if l.split(" ", 1)[0] in ("Pcrystal", "Pproperties", "crystal", "properties")]

        def wait_for(self, path: Path, seconds=30):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if path.exists() and path.read_text().strip():
                    return
                time.sleep(0.1)
            raise AssertionError(f"{path} never appeared")

    return Box()


def _session_alive(sid: int):
    alive = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            data = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue
        fields = data[data.rfind(")") + 2:].split()
        if fields[0] != "Z" and fields[3] == str(sid):
            alive.append(int(name))
    return alive


def _session_of(log: Path) -> int:
    m = re.search(r"session (\d+)", log.read_text())
    assert m, log.read_text()
    return int(m.group(1))


# --------------------------------------------------------------------------
# help
# --------------------------------------------------------------------------

def test_help_lists_the_command_and_its_options(box):
    r = box.run("--help")
    assert r.returncode == 0, r.text
    assert "--serial" not in r.stdout
    for text in ("Usage: mace run", "--cores", "--timeout", "--dry-run",
                 "mace run dia_sp.d12", "-local.o"):
        assert text in r.stdout, text
    general = subprocess.run([sys.executable, str(MACE_CLI), "--help"], env=box.env,
                             capture_output=True, text=True, timeout=60)
    assert re.search(r"^\s+run\s+Run decks on this node", general.stdout, re.M)
    submit = subprocess.run([sys.executable, str(MACE_CLI), "submit", "--help"], env=box.env,
                            capture_output=True, text=True, timeout=60)
    assert "mace run" in submit.stdout


@pytest.mark.parametrize("flag", [["--progress", "x"], ["--walltime", "1:00:00"],
                                  ["--track", "--nosubmit"], ["-i"]])
def test_submit_only_options_are_refused(box, flag):
    box.deck("x.d12")
    r = box.run(*flag, "x.d12")
    assert r.returncode == 2, r.text
    assert "unrecognized arguments" in r.text
    assert not (box.work / ".mace_local").exists()


# --------------------------------------------------------------------------
# one SCF deck, end to end
# --------------------------------------------------------------------------

def test_single_sp_runs_through_the_job_script_and_copies_back(box):
    deck = box.deck("x.d12")
    (box.work / "x.sh").write_text("#!/bin/bash\n# my own job script\n")
    r = box.run("x.d12", "-n", "2")
    assert r.returncode == 0, r.text
    assert "completed" in r.stdout
    assert (box.work / "x.out").read_text().count("TOTAL CPU TIME") == 1
    assert (box.work / "x.f9").read_text() == "fort9 of x\n"
    job_scratch = box.scratch / "crys23" / "x"
    assert (box.work / ".x.scratch").read_text().strip() == str(job_scratch)
    assert (box.stubdir / "x.INPUT").read_bytes() == deck.read_bytes()
    assert "mpirun -n 2 " in box.stub_log()
    assert f"Pcrystal job=x cwd={job_scratch}" in box.stub_log()
    local_o = (box.work / "x-local.o").read_text()
    assert "mace run: no completion callback" in local_o
    assert (box.work / "x.sh").read_text() == "#!/bin/bash\n# my own job script\n"
    assert (box.work / ".mace_local" / "x.sh").is_file()
    assert not (box.work / "materials.db").exists()
    assert "python " not in box.stub_log()             # no completion callback
    assert "sbatch" not in box.stub_log()


def test_default_rank_count_is_four_or_what_the_machine_has(box):
    box.deck("x.d12")
    r = box.run("x.d12")
    assert r.returncode == 0, r.text
    assert f"mpirun -n {min(4, len(os.sched_getaffinity(0)))} " in box.stub_log()


def test_default_on_a_two_core_affinity_runs_two_ranks(box):
    box.deck("x.d12")
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    r = box.run("x.d12", preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    assert r.returncode == 0, r.text
    assert "mpirun -n 2 " in box.stub_log()


def test_job_stdin_is_not_the_terminal(box):
    """mpiexec hands stdin to rank 0; the job must not read the user's keys."""
    box.deck("x.d12")
    r = box.run("x.d12", "-n", "1", input="keystrokes\n")
    assert r.returncode == 0, r.text
    assert "Pcrystal job=x" in box.stub_log() and "stdin=0" in box.stub_log()


def test_two_decks_get_their_own_scratch_directories(box):
    box.deck("a.d12")
    box.deck("b.d12")
    r = box.run(".", "-n", "1")
    assert r.returncode == 0, r.text
    assert (box.work / "a.f9").read_text() == "fort9 of a\n"
    assert (box.work / "b.f9").read_text() == "fort9 of b\n"
    assert (box.scratch / "crys23" / "a" / "INPUT").is_file()
    assert (box.scratch / "crys23" / "b" / "INPUT").is_file()


def test_staging_is_the_job_scripts(box):
    """GUESSP takes <job>.f20 as fort.20 and <job>.ext is fort.34 - the job
    script's staging, not a copy of it."""
    box.deck("g.d12", DECK.replace("SHRINK", "GUESSP\nSHRINK"))
    (box.work / "g.f20").write_text("guess matrix")
    (box.work / "g.ext").write_text("external geometry")
    r = box.run("g.d12", "-n", "1")
    assert r.returncode == 0, r.text
    assert (box.stubdir / "g.fort.20").read_text() == "guess matrix"
    assert (box.stubdir / "g.fort.34").read_text() == "external geometry"


def test_properties_outputs_get_the_slurm_names(box):
    for name, text in (("p_band", BAND), ("p_doss", DOSS), ("p_charge+potential", ECH3)):
        box.deck(f"{name}.d3", text)
        (box.work / f"{name}.f9").write_text("wavefunction")
    r = box.run(".", "-n", "1")
    assert r.returncode == 0, r.text
    assert (box.work / "p_band.BAND.DAT").read_text() == "band of p_band\n"
    assert (box.work / "p_band.f25").read_text() == "f25 of p_band\n"
    assert (box.work / "p_doss.DOSS.DAT").read_text() == "doss of p_doss\n"
    assert (box.work / "p_charge+potential_DENS.CUBE").read_text() == "dens of p_charge+potential\n"
    assert (box.scratch / "crys23" / "prop" / "p_band" / "INPUT").is_file()
    assert (box.stubdir / "p_band.fort.9").read_text() == "wavefunction"


def test_failed_deck_is_reported_with_its_error(box):
    box.deck("bad.d12")
    r = box.run("bad.d12", "-n", "1", STUB_FAIL_JOB="bad")
    assert r.returncode == 1, r.text
    assert "FAILED" in r.stdout and "deliberate failure" in r.stdout
    assert "bad-local.o" in r.stdout


# --------------------------------------------------------------------------
# order: SCF first, .f9 handed to its properties decks
# --------------------------------------------------------------------------

def test_properties_deck_runs_after_its_scf_with_the_new_f9(box):
    box.deck("a_band.d3", BAND)
    box.deck("a_sp.d12")                         # opt2d3 named a_sp's deck a_band
    r = box.run(".", "-n", "1")
    assert r.returncode == 0, r.text
    runs = box.binary_runs()
    assert [l.split()[1] for l in runs] == ["job=a_sp", "job=a_band"]
    assert (box.work / "a_band.f9").read_text() == "fort9 of a_sp\n"
    assert (box.stubdir / "a_band.fort.9").read_text() == "fort9 of a_sp\n"
    assert "a_sp.f9 -> a_band.f9" in r.stdout


def test_numbered_and_full_name_properties_decks_find_their_scf(box):
    box.deck("m.d12")
    box.deck("m_band2.d3", BAND)
    box.deck("m_charge+potential.d3", ECH3)
    r = box.run(".", "-n", "1")
    assert r.returncode == 0, r.text
    assert (box.work / "m_band2.f9").read_text() == "fort9 of m\n"
    assert (box.work / "m_charge+potential.f9").read_text() == "fort9 of m\n"


def test_existing_f9_is_replaced_not_written_through(box):
    box.deck("c.d12")
    box.deck("c_band.d3", BAND)
    target = box.root / "elsewhere.f9"
    target.write_text("someone else's file")
    (box.work / "c_band.f9").symlink_to(target)
    r = box.run(".", "-n", "1")
    assert r.returncode == 0, r.text
    assert target.read_text() == "someone else's file"
    assert not (box.work / "c_band.f9").is_symlink()
    assert (box.work / "c_band.f9").read_text() == "fort9 of c\n"
    assert "replaced the existing one" in r.stdout


def test_f9_hand_off_that_cannot_be_written_skips_only_that_deck(box):
    """A copy that fails (a full quota, here a directory in the way) skips
    that properties deck; the run goes on and ends with its summary."""
    box.deck("h.d12")
    box.deck("h_band.d3", BAND)
    box.deck("h_doss.d3", DOSS)
    (box.work / "h_band.f9").mkdir()
    (box.work / "h_band.f9" / "keep").write_text("in the way")
    r = box.run(".", "-n", "1")
    assert r.returncode == 1, r.text
    assert "Traceback" not in r.text
    assert "cannot copy h.f9 to h_band.f9" in r.stdout
    assert (box.work / "h_doss.DOSS.DAT").read_text() == "doss of h_doss\n"
    assert "1 skipped" in r.stdout
    assert not list(box.work.glob("*.mace_tmp"))


def test_failed_scf_skips_its_properties_deck(box):
    box.deck("f_sp.d12")
    box.deck("f_band.d3", BAND)
    (box.work / "f_band.f9").write_text("old wavefunction")
    r = box.run(".", "-n", "1", STUB_FAIL_JOB="f_sp")
    assert r.returncode == 1, r.text
    assert "parent f_sp did not complete" in r.stdout
    assert [l.split()[1] for l in box.binary_runs()] == ["job=f_sp"]
    assert (box.work / "f_band.f9").read_text() == "old wavefunction"


def test_properties_deck_without_scf_or_f9_is_skipped(box):
    box.deck("o_band.d3", BAND)
    r = box.run("o_band.d3", "-n", "1")
    assert r.returncode == 1, r.text
    assert "no o_band.f9" in r.stdout and "opt2d3" in r.stdout
    assert box.binary_runs() == []


def test_two_possible_scf_parents_is_ambiguous(box):
    box.deck("x_opt.d12")
    box.deck("x_sp.d12")
    box.deck("x_band.d3", BAND)
    r = box.run(".", "-n", "1")
    assert r.returncode == 1, r.text
    assert "ambiguous parent (x_opt, x_sp)" in r.stdout
    assert not (box.work / "x_band.f9").exists()


def test_failed_deck_does_not_stop_independent_decks(box):
    box.deck("a.d12")
    box.deck("b.d12")
    r = box.run(".", "-n", "1", STUB_FAIL_JOB="a")
    assert r.returncode == 1, r.text
    assert (box.work / "b.f9").read_text() == "fort9 of b\n"


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_refused_on_the_gateway(box):
    box.deck("x.d12")
    r = box.run("x.d12", MODULEPATH="/opt/modules/x86_64/gateway/all")
    assert r.returncode == 2, r.text
    assert "gateway" in r.text and "ssh dev-amd20" in r.text
    assert not (box.work / ".mace_local").exists()
    assert box.stub_log() == ""


def test_refused_without_crystal(box):
    box.deck("x.d12")
    r = box.run("x.d12", EBROOTCRYSTAL=None)
    assert r.returncode == 2, r.text
    assert "no CRYSTAL here" in r.text
    assert not (box.work / ".mace_local").exists()


def test_refused_inside_a_slurm_job(box):
    box.deck("x.d12")
    r = box.run("x.d12", SLURM_JOB_ID="4242")
    assert r.returncode == 2, r.text
    assert "inside a SLURM job" in r.text
    assert box.stub_log() == ""


@pytest.mark.parametrize("args", [["-n", "0"], ["-n", "17"], ["--serial"],
                                  ["--timeout", "3h"], ["--timeout", "soon"]])
def test_bad_cores_or_timeout_are_refused(box, args):
    box.deck("x.d12")
    r = box.run(*args, "x.d12")
    assert r.returncode == 2, r.text
    assert box.stub_log() == ""


def test_more_ranks_than_this_process_may_use_are_refused(box):
    box.deck("x.d12")
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    r = box.run("x.d12", "-n", "3", preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    assert r.returncode == 2, r.text
    assert "1 to 2 ranks" in r.text


def test_many_ranks_get_a_politeness_warning(box):
    if len(os.sched_getaffinity(0)) < 9:
        pytest.skip("needs nine CPUs")
    box.deck("x.d12")
    r = box.run("x.d12", "-n", "9")
    assert r.returncode == 0, r.text
    assert "dev nodes are shared" in r.text


def test_timeout_above_the_cpu_time_limit_is_refused(box):
    import resource
    box.deck("x.d12")
    limit = lambda: resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    r = box.run("--timeout", "20m", "x.d12", preexec_fn=limit)
    assert r.returncode == 2, r.text
    assert "ulimit -t" in r.text
    ok = box.run("--timeout", "4m", "--dry-run", "x.d12", preexec_fn=limit)
    assert ok.returncode == 0, ok.text


def test_timeout_formats(box):
    box.deck("x.d12")
    r = box.run("--timeout", "1:30:00", "--dry-run", "x.d12")
    assert r.returncode == 0, r.text
    assert "timeout 1h30m00s" in r.stdout


def test_paths_the_job_script_cannot_take_are_refused(box):
    box.deck("my deck.d12")
    r = box.run("my deck.d12")
    assert r.returncode == 2, r.text
    assert "cannot use this path" in r.text


def test_same_job_name_twice_is_refused(box):
    box.deck("x.d12", where=box.work / "one")
    box.deck("x.d12", where=box.work / "two")
    r = box.run("one", "two", "-n", "1")
    assert r.returncode == 2, r.text
    assert "share the scratch directory" in r.text


def _lock_path(box, kind, job):
    return box.root / "xdg" / f"mace-run-{os.getuid()}" / f"{kind}-{job}.lock"


def test_job_name_another_mace_run_holds_is_refused(box):
    """Scratch is $SCRATCH/crys23/<job> whatever the directory: the same job
    name in another directory must wait for the run that has it."""
    box.deck("x.d12", where=box.work / "one")
    box.deck("x.d12", where=box.work / "two")
    first = box.start("-n", "1", "one/x.d12", STUB_SLEEP="60")
    try:
        box.wait_for(box.stubdir / "x.sleeping")
        r = box.run("-n", "1", "two/x.d12")
        assert r.returncode == 2, r.text
        assert "another mace run" in r.text and "x" in r.text
        assert not (box.work / "two" / ".mace_local").exists()
    finally:
        first.send_signal(signal.SIGINT)
        first.communicate(timeout=40)
    again = box.run("-n", "1", "two/x.d12")            # free once the first has stopped
    assert again.returncode == 0, again.text
    assert _lock_path(box, "scf", "x").exists()        # never unlinked


def test_held_lock_refuses_and_properties_have_their_own(box):
    import fcntl
    box.deck("x.d12")
    lock = _lock_path(box, "scf", "x")
    lock.parent.mkdir(mode=0o700)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = box.run("-n", "1", "x.d12")
        assert r.returncode == 2, r.text
        assert "another mace run" in r.text
        box.deck("x_band.d3", BAND, where=box.work / "p")
        (box.work / "p" / "x_band.f9").write_text("wavefunction")
        ok = box.run("-n", "1", "p/x_band.d3")         # crys23/prop/x_band: no clash
        assert ok.returncode == 0, ok.text


def test_lock_directory_others_can_write_is_refused(box):
    """Without $XDG_RUNTIME_DIR the locks live in /tmp/mace-run-<uid>, a name
    anyone could create first; a lock directory that is not private is not
    used."""
    box.deck("x.d12")
    locks = _lock_path(box, "scf", "x").parent
    locks.mkdir()
    locks.chmod(0o777)
    r = box.run("-n", "1", "x.d12")
    assert r.returncode == 2, r.text
    assert "not private" in r.text
    assert box.binary_runs() == []


def test_same_named_slurm_job_is_refused(box):
    box.deck("x.d12")
    r = box.run("-n", "1", "x.d12", STUB_SQUEUE="x 777 RUNNING\nother 778 PENDING")
    assert r.returncode == 2, r.text
    assert "SLURM job 777 (RUNNING)" in r.text
    assert box.binary_runs() == []
    ok = box.run("-n", "1", "x.d12", STUB_SQUEUE="other 778 PENDING")
    assert ok.returncode == 0, ok.text


def test_opt_deck_gets_a_warning(box):
    box.deck("o.d12", DECK.replace("END\n6 0", "OPTGEOM\nENDOPT\nEND\n6 0", 1))
    r = box.run("--dry-run", "o.d12")
    assert r.returncode == 0, r.text
    assert "rarely fits on a dev node" in r.text


def test_dry_run_prints_the_plan_and_writes_nothing(box):
    box.deck("a_sp.d12")
    box.deck("a_band.d3", BAND)
    r = box.run("--dry-run", ".")
    assert r.returncode == 0, r.text
    assert r.stdout.index("a_sp.d12") < r.stdout.index("a_band.d3")
    assert "$SCRATCH/crys23/a_sp (resolved by the job script)" in r.stdout
    assert "a_sp.f9 -> a_band.f9" in r.stdout
    assert sorted(p.name for p in box.work.iterdir()) == ["a_band.d3", "a_sp.d12"]
    assert not box.scratch.exists()
    assert box.binary_runs() == []
    assert not (box.root / "xdg" / f"mace-run-{os.getuid()}").exists()     # no lock taken


# --------------------------------------------------------------------------
# stopping: timeout, Ctrl-C, hang-up
# --------------------------------------------------------------------------

def test_timeout_stops_the_whole_job(box):
    box.deck("x.d12")
    t0 = time.monotonic()
    r = box.run("--timeout", "2s", "-n", "1", "x.d12", STUB_SLEEP="60")
    assert r.returncode == 124, r.text
    assert time.monotonic() - t0 < 25
    assert "TIMED OUT" in r.stdout
    assert not (box.work / "x.f9").exists()             # nothing copied back
    log = (box.work / "x-local.o").read_text()
    assert "stopped (timeout 0m02s)" in log and "scratch kept at" in log
    assert _session_alive(_session_of(box.work / "x-local.o")) == []
    assert "python " not in box.stub_log()


def _interrupt(box, sig, *, twice=False, **env):
    box.deck("x.d12")
    box.deck("y.d12")
    p = box.start("-n", "1", ".", STUB_SLEEP="60", STUB_SLEEP_JOB="x", **env)
    try:
        box.wait_for(box.stubdir / "x.sleeping")
        p.send_signal(sig)
        if twice:
            time.sleep(0.5)
            p.send_signal(sig)
        out, _ = p.communicate(timeout=40)
    finally:
        if p.poll() is None:
            p.kill()
    return p.returncode, out


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP, signal.SIGTERM])
def test_interrupt_stops_the_job_and_the_batch(box, sig):
    rc, out = _interrupt(box, sig)
    assert rc == 128 + sig, out
    assert _session_alive(_session_of(box.work / "x-local.o")) == []
    assert not (box.work / "x.f9").exists()
    assert "python " not in box.stub_log()
    assert f"interrupted ({signal.Signals(sig).name})" in (box.work / "x-local.o").read_text()
    assert "not run: y.d12" in out
    assert "job=y" not in box.stub_log()


def test_second_ctrl_c_kills_a_job_that_ignores_sigterm(box):
    t0 = time.monotonic()
    rc, out = _interrupt(box, signal.SIGINT, twice=True, STUB_IGNORE_TERM="1")
    assert rc == 130, out
    assert time.monotonic() - t0 < 12                   # no 15 s grace wait
    assert _session_alive(_session_of(box.work / "x-local.o")) == []


def test_job_that_ignores_sigterm_is_killed_after_the_grace_period(box):
    rc, out = _interrupt(box, signal.SIGINT, STUB_IGNORE_TERM="1")
    assert rc == 130, out
    assert _session_alive(_session_of(box.work / "x-local.o")) == []


def test_nohup_keeps_the_run_going_after_a_hang_up(box):
    box.deck("x.d12")
    p = box.start("-n", "1", "x.d12", STUB_SLEEP="3",
                  preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    try:
        box.wait_for(box.stubdir / "x.sleeping")
        p.send_signal(signal.SIGHUP)
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, out
    assert (box.work / "x.f9").read_text() == "fort9 of x\n"


# --------------------------------------------------------------------------
# small pieces
# --------------------------------------------------------------------------

def test_parse_duration():
    from mace.submission.local_run import parse_duration
    assert parse_duration("90s") == 90
    assert parse_duration("30m") == 1800
    assert parse_duration("1h") == 3600
    assert parse_duration("1:30:00") == 5400
    assert parse_duration("45") == 45
    for bad in ("", "1:99:00", "ten", "0", "-5m"):
        with pytest.raises(ValueError):
            parse_duration(bad)


@pytest.mark.parametrize("scf, prop, expected", [
    ("x_sp", "x_band", True), ("x_sp", "x_sp_band", True), ("x", "x_band", True),
    ("x_opt", "x_doss3", True), ("x", "x_charge+potential", True), ("x", "x_transport", True),
    ("x", "x_y_band", False), ("x_sp", "x_spband", False), ("xy", "x_band", False),
    ("x_SP", "x_BAND", True),
])
def test_parent_matching_follows_opt2d3_names(scf, prop, expected):
    from mace.submission.local_run import parent_matches
    assert parent_matches(scf, prop) is expected


def test_an_out_this_run_did_not_write_is_no_verdict(tmp_path):
    from mace.submission.local_run import Deck, file_sig, verdict
    deck_path = tmp_path / "x.d12"
    deck_path.write_text(DECK)
    out = tmp_path / "x.out"
    out.write_text("    TOTAL CPU TIME =        1.000\n")
    deck = Deck(deck_path)
    assert verdict(deck, file_sig(out)) == (False, "no .out written")
    assert verdict(deck, None)[0] is True


# --------------------------------------------------------------------------
# --track
# --------------------------------------------------------------------------

def _calcs(box):
    import sqlite3
    db = box.work / "materials.db"
    assert db.exists()
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM calculations ORDER BY created_at")]
    con.close()
    return rows


def test_track_records_a_completed_local_run(box):
    import json
    import socket
    box.deck("t.d12")
    r = box.run("--track", "-n", "1", "t.d12")
    assert r.returncode == 0, r.text
    (calc,) = _calcs(box)
    assert calc["status"] == "completed"
    assert calc["slurm_job_id"] is None and calc["job_script"] is None
    assert Path(calc["output_file"]) == box.work / "t.out"
    assert Path(calc["work_dir"]) == box.work
    settings = json.loads(calc["settings_json"])
    assert settings["run_mode"] == "local" and settings["host"] == socket.gethostname()
    assert "python " not in box.stub_log() and "sbatch" not in box.stub_log()
    assert not (box.work / "crystal_job_status.json").exists()


def test_tracked_failure_is_never_recovered_through_slurm(box):
    """A failed local run must not be picked up by the recovery sweep that any
    later completion callback or `mace manager` runs over materials.db."""
    box.deck("t.d12")
    r = box.run("--track", "-n", "1", "t.d12", STUB_FAIL_JOB="t",
                STUB_FAIL_TEXT="TOO MANY CYCLES")
    assert r.returncode == 1, r.text
    (calc,) = _calcs(box)
    assert calc["status"] == "failed"
    assert calc["error_type"] == "local_convergence_error"
    from mace.recovery.recovery import ErrorRecoveryEngine
    engine = ErrorRecoveryEngine(str(box.work / "materials.db"))
    assert "convergence_error" in engine.config["error_recovery"]      # would be recoverable
    assert engine.get_recoverable_calculations() == []
    assert "sbatch" not in box.stub_log()


def test_tracked_timeout_is_recorded_as_such(box):
    box.deck("t.d12")
    r = box.run("--track", "-n", "1", "--timeout", "2s", "t.d12", STUB_SLEEP="60")
    assert r.returncode == 124, r.text
    (calc,) = _calcs(box)
    assert calc["status"] == "failed" and calc["error_type"] == "local_timeout"


def test_track_ignores_a_workflow_in_the_environment(box):
    """MACE_WORKFLOW_ID makes the completion handler progress a workflow and
    submit its next step; a local run never does."""
    box.deck("t.d12")
    r = box.run("--track", "-n", "1", "t.d12", MACE_WORKFLOW_ID="workflow_20260101_000000",
                MACE_CONTEXT_DIR=str(box.root / "ctx"))
    assert r.returncode == 0, r.text
    assert _calcs(box)[0]["status"] == "completed"
    assert "sbatch" not in box.stub_log()
    assert not (box.root / "ctx").exists()


def test_track_is_refused_in_a_workflow_tree(box):
    box.deck("t.d12")
    (box.work / "workflow_outputs").mkdir()
    r = box.run("--track", "t.d12")
    assert r.returncode == 2, r.text
    assert "mace workflow" in r.text
    assert not (box.work / "materials.db").exists()
    assert box.binary_runs() == []


def test_track_marks_a_run_whose_runner_died_as_lost(box):
    import json
    import socket
    from mace.database.materials import MaterialDatabase
    dead = subprocess.Popen(["true"])
    dead.wait()
    db = MaterialDatabase(str(box.work / "materials.db"))
    db.create_material(material_id="old", formula="C", source_type="d12",
                       source_file="old.d12", metadata={})
    calc_id = db.create_calculation(material_id="old", calc_type="SP", input_file="old.d12",
                                    work_dir=str(box.work),
                                    settings={"run_mode": "local", "host": socket.gethostname(),
                                              "pid": dead.pid})
    db.update_calculation_status(calc_id, "running")
    box.deck("t.d12")
    r = box.run("--track", "-n", "1", "t.d12")
    assert r.returncode == 0, r.text
    old = MaterialDatabase(str(box.work / "materials.db")).get_calculation(calc_id)
    assert old["status"] == "failed" and old["error_type"] == "local_lost"


def test_dry_run_with_track_writes_no_database(box):
    box.deck("t.d12")
    r = box.run("--track", "--dry-run", "t.d12")
    assert r.returncode == 0, r.text
    assert not (box.work / "materials.db").exists()


def test_docs_describe_mace_run():
    doc = " ".join((REPO_ROOT / "DOCUMENTATION.md").read_text().split())
    assert "### Run on a dev node (`mace run`)" in doc
    assert "HPCC dev nodes stop any process after 2 hours of CPU time" in doc
    assert "the login gateway" in doc and "`<job>-local.o`" in doc
    assert "mace run" in (REPO_ROOT / "README.md").read_text()
    assert "`mace run` runs small decks there directly" in " ".join(
        (REPO_ROOT / "INSTALLATION.md").read_text().split())
