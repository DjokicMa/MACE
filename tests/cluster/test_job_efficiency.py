"""Job CPU and memory efficiency in plain numbers (mace/cluster/job_efficiency.py),
from realistic sacct / sstat output: a multi-step MPI job, a batch-only job,
a running job (sstat), missing fields and per-CPU vs per-node ReqMem; plus
the real `mace_cli cluster job` under FakeSlurm."""
import json
import subprocess
import sys

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import MACE_CLI, REPO_ROOT, fake
from mace.cluster import job_efficiency as je
from mace.cluster import reader

GIB = 1024 ** 3


def _sacct(text):
    return reader.parse_sacct(text)


# sacct -P -n -o <reader.SACCT_FIELDS> for an ended srun-launched MPI job: 2 nodes x 16 tasks
MPI_DONE = "\n".join([
    "7001|mat_opt|COMPLETED|02:00:00|04:00:00|32|128G||1-12:00:00|billing=32,cpu=32,mem=128G,node=2|cpu-[001-002]|0:0|"
    "2026-10-01T10:00:00|2026-10-01T12:00:00|/home/u/proj|||2",
    "7001.batch|batch|COMPLETED|02:00:00||16|||00:00:05.120|cpu=16,mem=64G,node=1|cpu-001|0:0|||/home/u/proj|1|00:00:05|1",
    "7001.extern|extern|COMPLETED|02:00:00||32||1040K|00:00:00.004|billing=32,cpu=32,mem=128G,node=2|cpu-[001-002]|0:0|||"
    "/home/u/proj|2|00:00:00|2",
    "7001.0|Pcrystal|COMPLETED|01:59:50||32||2457600K|1-11:59:55|cpu=32,mem=128G,node=2|cpu-[001-002]|0:0|||/home/u/proj"
    "|32|01:07:30|2",
])


def test_parse_duration_formats():
    assert je.parse_duration("39-13:37:34") == 39 * 86400 + 13 * 3600 + 37 * 60 + 34
    assert je.parse_duration("02:00:00") == 7200
    assert je.parse_duration("12:34.567") == pytest.approx(754.567)
    assert je.parse_duration("00:00:05.120") == pytest.approx(5.12)
    assert je.parse_duration("2-04") == 2 * 86400 + 4 * 3600
    assert je.parse_duration("") is None and je.parse_duration("INVALID") is None and je.parse_duration("x") is None


def test_parse_memory_units():
    assert je.parse_memory("9813656K") == (9813656 * 1024, "")
    assert je.parse_memory("64G") == (64 * GIB, "")
    assert je.parse_memory("4000Mc") == (4000 * 1024 ** 2, "c")
    assert je.parse_memory("64Gn") == (64 * GIB, "n")
    assert je.parse_memory("1.5T")[0] == 1.5 * 1024 ** 4
    assert je.parse_memory("4000", "M")[0] == 4000 * 1024 ** 2          # a request without unit is MB
    assert je.parse_memory("") == (None, "")


def test_multistep_mpi_job_sums_steps_and_counts_tasks():
    """The allocation line's TotalCPU (the steps' sum) over 2 h x 32 CPUs;
    MaxRSS of the .0 step is one rank's peak, so 32 ranks count up to 32x."""
    e = je.summarize(_sacct(MPI_DONE), "7001")
    assert e["cpu"] == pytest.approx(56.25, abs=0.1)    # 36 h / 64 h
    assert e["cpu_text"] == "CPU efficiency 56% (used about 18 of 32 cores on average)"
    assert e["peak_step"] == "7001.0" and e["peak_tasks"] == 32
    assert e["memory_text"] == ("Memory peak 2.3 GiB per task, up to 75 GiB across 32 tasks "
                                "of 128 GiB requested (59%)")
    assert e["advice"] == []                    # neither clearly over-requested
    assert "requested 128 GiB (AllocTRES)" in e["details"]


def test_allocation_line_without_totalcpu_uses_step_sum():
    rows = _sacct(MPI_DONE)
    rows["7001"]["TotalCPU"] = ""
    e = je.summarize(rows, "7001")
    assert e["cpu_used_s"] == pytest.approx(5.12 + 0.004 + (35 * 3600 + 59 * 60 + 55))


def test_step_without_totalcpu_counts_avecpu_times_ntasks():
    rows = _sacct(MPI_DONE)
    rows["7001"]["TotalCPU"] = ""
    rows["7001.0"]["TotalCPU"] = ""
    e = je.summarize(rows, "7001")             # AveCPU 01:07:30 x 32 tasks = 36 h
    assert e["cpu_used_s"] == pytest.approx(36 * 3600 + 5.124)


def test_batch_only_job_over_requested_gives_advice():
    """mpirun inside the batch script (no srun): everything is in .batch."""
    text = "\n".join([
        "7100|mat_sp|COMPLETED|10:00:00|1-00:00:00|32|64G||00:00:00|billing=32,cpu=32,mem=64G,node=1|cpu-007|0:0|"
        "2026-10-02T00:00:00|2026-10-02T10:00:00|/home/u/p|||1",
        "7100.batch|batch|COMPLETED|10:00:00||32||9813656K|3-08:00:00|cpu=32,mem=64G,node=1|cpu-007|0:0|||/home/u/p|1"
        "|3-08:00:00|1",
    ])
    e = je.summarize(_sacct(text), "7100")
    assert e["cpu_text"] == "CPU efficiency 25% (used about 8 of 32 cores on average)"
    assert e["memory_text"] == "Memory peak 9.4 GiB of 64 GiB requested (15%)"
    assert e["advice"] == ["You could request fewer cores (about 8 were busy).",
                           "You could request less memory (the peak was 9.4 GiB)."]
    # the busy-wait caveat once, right after the CPU line; the advice stays
    assert je.lines(e)[:3] == [e["cpu_text"], je.CPU_CAVEAT, e["memory_text"]]
    assert e["cpu_caveat"] == je.CPU_CAVEAT and "Intel MPI" in je.CPU_CAVEAT


def test_cpu_caveat_only_with_a_measured_cpu_efficiency():
    """No CPU figure (not started, nothing recorded): no caveat either."""
    rows = {"7900": {"JobID": "7900", "State": "PENDING", "Elapsed": "00:00:00", "NCPUS": "8"}}
    e = je.summarize(rows, "7900")
    assert e["cpu"] is None and e["cpu_caveat"] is None
    assert je.CPU_CAVEAT not in je.lines(e)
    assert je.lines({**e, "cpu_caveat": None}) == je.lines(e)


def test_running_job_without_live_says_why_then_sstat_fills_in():
    """The maintainer's sample: sacct has nothing for a running job yet;
    sstat's MaxRSS 9813656K / AveCPU 39-13:37:34 on .batch, 32 CPUs, 32 h."""
    text = ("7200|mat_freq|RUNNING|1-08:00:00|3-00:00:00|32|64G||00:00:00|billing=32,cpu=32,mem=64G,node=1|cpu-003|"
            "0:0|2026-10-03T00:00:00|Unknown|/home/u/p|||1\n"
            "7200.batch|batch|RUNNING|1-08:00:00||32|||00:00:00|cpu=32,mem=64G,node=1|cpu-003|0:0|||/home/u/p|1||1\n"
            "7200.extern|extern|RUNNING|1-08:00:00||32|||00:00:00|cpu=32,mem=64G,node=1|cpu-003|0:0|||/home/u/p|1||1")
    rows = _sacct(text)
    e = je.summarize(rows, "7200")
    assert e["cpu"] is None and e["memory"] is None
    assert "only when its steps end" in e["cpu_note"] and "Show Live Use" in e["cpu_note"]
    assert "Not measured" not in e["cpu_line"]
    live = [{"JobID": "7200.extern", "MaxRSS": "1040K", "AveCPU": "00:00:00", "NTasks": "1"},
            {"JobID": "7200.batch", "MaxRSS": "9813656K", "AveCPU": "39-13:37:34", "NTasks": "1"}]
    e = je.summarize(rows, "7200", live=live)
    assert e["cpu_text"] == "CPU efficiency 93% so far (used about 30 of 32 cores on average)"
    assert e["memory_text"] == "Memory peak so far 9.4 GiB of 64 GiB requested (15%)"
    assert e["advice"] == []                    # a running job's memory may still grow
    assert e["source"] == "sstat"


def test_missing_fields_say_not_measured_and_why():
    text = ("7300|x|COMPLETED|01:00:00|02:00:00|8|16G||00:00:00|cpu=8,mem=16G,node=1|cpu-001|0:0|||/home/u/p|||1\n"
            "7300.batch|batch|COMPLETED|01:00:00||8||0|00:00:00|cpu=8,mem=16G,node=1|cpu-001|0:0|||/home/u/p|1||1")
    e = je.summarize(_sacct(text), "7300")
    assert e["cpu_note"].startswith(je.NOT_MEASURED) and "no CPU time" in e["cpu_note"]
    assert e["memory_note"].startswith(je.NOT_MEASURED) and "MaxRSS" in e["memory_note"]
    e = je.summarize({"7400": {"JobID": "7400", "State": "PENDING", "NCPUS": "8"}}, "7400")
    assert e["cpu_note"] == e["memory_note"] == "The job has not started yet"
    # implausible: more CPU time than 4 cores x 1 h allow
    e = je.summarize({"7500": {"JobID": "7500", "State": "COMPLETED", "Elapsed": "01:00:00", "NCPUS": "4",
                               "TotalCPU": "10:00:00", "MaxRSS": "1G", "ReqMem": ""}}, "7500")
    assert e["cpu"] is None and "more than 4 cores" in e["cpu_note"]
    assert e["memory_text"] == "Memory peak 1 GiB (the requested memory is unknown)"


@pytest.mark.parametrize("reqmem,ncpus,nnodes,expect_gib", [
    ("4000Mc", 16, 1, 4000 * 16 / 1024),        # older SLURM: per CPU
    ("32Gn", 32, 2, 64),                         # older SLURM: per node x nodes
    ("64G", 32, 2, 64),                          # SLURM >= 21.08: the job's total
])
def test_reqmem_per_cpu_and_per_node(reqmem, ncpus, nnodes, expect_gib):
    main = {"JobID": "7600", "State": "COMPLETED", "Elapsed": "01:00:00", "NCPUS": str(ncpus),
            "NNodes": str(nnodes), "ReqMem": reqmem, "AllocTRES": "", "TotalCPU": "01:00:00",
            "MaxRSS": "8G"}
    e = je.summarize({"7600": main}, "7600")
    assert e["req_bytes"] == pytest.approx(expect_gib * GIB)
    assert e["memory"] == pytest.approx(round(100 * 8 / expect_gib, 1))


def test_tres_mem_wins_over_reqmem():
    main = {"JobID": "7700", "State": "COMPLETED", "ReqMem": "4000Mc", "NCPUS": "16",
            "AllocTRES": "billing=16,cpu=16,mem=80G,node=1"}
    assert je.requested_memory(main, 16, 1) == (80 * GIB, "AllocTRES")


def test_multinode_batch_only_cpu_is_flagged_not_advised():
    """mpirun's ssh launcher: ranks on the second node are outside SLURM."""
    rows = {"7800": {"JobID": "7800", "State": "COMPLETED", "Elapsed": "10:00:00", "NCPUS": "64", "NNodes": "2",
                     "TotalCPU": "", "AllocTRES": "cpu=64,mem=128G,node=2"},
            "7800.batch": {"JobID": "7800.batch", "TotalCPU": "13-08:00:00", "MaxRSS": "40G", "NTasks": "1"}}
    e = je.summarize(rows, "7800")
    assert e["cpu"] == 50.0 and "other nodes may be missing" in e["cpu_text"]
    assert not any("fewer cores" in a for a in e["advice"])


def test_old_sacct_lines_still_parse():
    """A line that ends at WorkDir (recorded before NTasks/AveCPU/NNodes)."""
    rows = _sacct("4001|m|COMPLETED|00:10:00|01:00:00|4|4G||00:00:00|cpu=4|cpu-001|0:0|||/w")
    assert rows["4001"]["WorkDir"] == "/w" and rows["4001"]["NTasks"] == ""


# ------------------------------------------------------------------ the real CLI
def _mace(fs, *args):
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], env=dict(fs.env), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def test_cli_job_running_live_efficiency(tmp_path):
    fs = fake(tmp_path)
    fs.add_record(id="7200", state="RUNNING", name="mat_freq", workdir=str(tmp_path), time_limit="3-00:00:00",
                  elapsed="1-08:00:00", cpus=32)
    fs.update(job_text={"7200": "JobId=7200 JobName=mat_freq UserId=user1(1001) JobState=RUNNING Reason=None "
                                "RunTime=1-08:00:00 TimeLimit=3-00:00:00 Partition=general NodeList=cpu-003 "
                                f"NumNodes=1 NumCPUs=32 WorkDir={tmp_path} Account=acct1"},
              sacct_rows={"7200": ["7200|mat_freq|RUNNING|1-08:00:00|3-00:00:00|32|64G||00:00:00|"
                                   "billing=32,cpu=32,mem=64G,node=1|cpu-003|0:0|||" + str(tmp_path) + "|||1"]},
              sstat_text="7200.extern|1040K|00:00:00|1\n7200.batch|9813656K|39-13:37:34|1")
    rc, out, err = _mace(fs, "cluster", "job", "7200")
    assert rc == 0, err
    assert "only when its steps end" in out and "MaxRSS" not in out
    rc, out, err = _mace(fs, "cluster", "job", "7200", "--live")
    assert rc == 0, err
    assert "CPU efficiency 93% so far (used about 30 of 32 cores on average)" in out
    assert "Memory peak so far 9.4 GiB of 64 GiB requested (15%)" in out
    assert "AveCPU" not in out and "9813656K" not in out
    sstat = [c["argv"] for c in fs.calls("sstat")]
    assert sstat and "-a" in sstat[-1] and "7200" in sstat[-1]


def test_cli_job_ended_json(tmp_path):
    fs = fake(tmp_path)
    fs.update(sacct_rows={"7100": [
        "7100|mat_sp|COMPLETED|10:00:00|1-00:00:00|32|64G||00:00:00|billing=32,cpu=32,mem=64G,node=1|cpu-007|0:0|"
        "|||1", "7100.batch|batch|COMPLETED|10:00:00||32||9813656K|3-08:00:00|cpu=32,mem=64G,node=1|cpu-007|0:0|||"
        "|1|3-08:00:00|1"]})
    rc, out, err = _mace(fs, "cluster", "job", "7100", "--json")
    assert rc == 0, err
    e = json.loads(out)["efficiency"]
    assert e["cpu"] == 25.0 and e["memory"] == 14.6
    rc, out, err = _mace(fs, "cluster", "job", "7100")
    assert "CPU efficiency 25% (used about 8 of 32 cores on average)" in out
    assert "You could request less memory (the peak was 9.4 GiB)." in out


# ------------------------------------------------------------------ shapes seen on a real cluster
RUNNING_64 = ("7900|mat_freq|RUNNING|4-04:08:38|7-00:00:00|64|256G||00:00:00|billing=115,cpu=64,mem=256G,node=1|"
              "cpu-010|0:0|2026-10-01T00:00:00|Unknown|/home/u/p|||1\n"
              "7900.batch|batch|RUNNING|4-04:08:38||64|||00:00:00|cpu=64,mem=256G,node=1|cpu-010|0:0|||/home/u/p|1||1\n"
              "7900.extern|extern|RUNNING|4-04:08:38||64|||00:00:00|billing=115,cpu=64,mem=256G,node=1|cpu-010|0:0|||"
              "/home/u/p|1||1")


def test_garbage_extern_avecpu_from_sstat_is_ignored():
    """sstat gives a running job's extern step an absurd AveCPU (a SLURM
    sentinel); it must not hide the batch step's real CPU time."""
    live = [{"JobID": "7900.extern", "MaxRSS": "", "AveCPU": "213503982334-14:25:51", "NTasks": "1"},
            {"JobID": "7900.batch", "MaxRSS": "84601596K", "AveCPU": "227-01:38:56", "NTasks": "1"}]
    e = je.summarize(_sacct(RUNNING_64), "7900", live=live)
    assert e["cpu"] == pytest.approx(85.0, abs=0.1)
    assert e["cpu_text"] == "CPU efficiency 85% so far (used about 54 of 64 cores on average)"
    assert e["memory"] == pytest.approx(31.5, abs=0.1)
    assert any(d.startswith("ignored step 7900.extern") for d in e["details"])
    # only garbage: still said to be implausible, not summed
    e = je.summarize(_sacct(RUNNING_64), "7900", live=live[:1])
    assert e["cpu"] is None and e["cpu_note"].startswith(je.NOT_MEASURED) and "more than 64 cores" in e["cpu_note"]


def test_empty_sstat_answer_does_not_ask_for_show_live_use_again():
    """sstat from a host that cannot reach the compute node prints nothing
    (rc 0); after Show Live Use the note must not tell the user to press it."""
    e = je.summarize(_sacct(RUNNING_64), "7900", live=[])
    assert "Show Live Use" not in e["cpu_note"] and "sstat returned nothing" in e["cpu_note"]
    assert "Show Live Use" not in e["memory_note"] and "sstat returned nothing" in e["memory_note"]
    e = je.summarize(_sacct(RUNNING_64), "7900")                 # not asked yet
    assert "Show Live Use" in e["cpu_note"]


def test_out_of_memory_job_advises_more_memory():
    """MaxRSS is sampled, so an OUT_OF_MEMORY job's recorded peak sits well
    below the request (75% here); the advice must be to ask for more."""
    text = ("7950_2|mat_opt|OUT_OF_MEMORY|00:46:52|04:00:00|1|8G||00:46:40|billing=2,cpu=1,mem=8G,node=1|cpu-011|0:125|"
            "||/home/u/p|||1\n"
            "7950_2.batch|batch|OUT_OF_MEMORY|00:46:52||1||6268000K|00:46:40|cpu=1,mem=8G,node=1|cpu-011|0:125|||"
            "/home/u/p|1|00:46:40|1")
    e = je.summarize(_sacct(text), "7950_2")
    assert e["memory"] == pytest.approx(74.7, abs=0.1)
    assert e["advice"] == ["The job ran out of memory: request more. SLURM samples memory use, so the real "
                           "peak was higher than the one shown."]
    assert e["cpu_text"] == "CPU efficiency 100% (used about 1 of 1 core on average)"
    assert "1 CPU" in e["details"] and "1 CPUs" not in e["details"]


def test_one_busy_core_reads_singular():
    text = ("7960|band|COMPLETED|00:10:28|01:00:00|28|80G||00:10:23|billing=28,cpu=28,mem=80G,node=1|cpu-012|0:0|||"
            "/home/u/p|||1\n"
            "7960.batch|batch|COMPLETED|00:10:28||28||7234000K|00:10:23|cpu=28,mem=80G,node=1|cpu-012|0:0|||/home/u/p"
            "|1|00:10:23|1")
    e = je.summarize(_sacct(text), "7960")
    assert e["cpu"] == pytest.approx(3.5, abs=0.2)
    assert "You could request fewer cores (about 1 was busy)." in e["advice"]
