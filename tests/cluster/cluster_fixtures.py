"""SLURM and quota texts for the cluster tests, in the tools' real formats
(captured on a SLURM 26.05 cluster, then generalized: node names cpu-NNN /
gpu-NNN, account acct1, user user1, invented job ids and paths)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fake_slurm import FakeSlurm  # noqa: E402

REPO_ROOT = HERE.parent.parent
MACE_CLI = REPO_ROOT / "mace_cli"
TEST_DATA = REPO_ROOT / "test"


def _node(name, state, alloc, total, mem, alloc_mem, feats, parts, gres="(null)", cfg_extra="", alloc_extra=""):
    return (f"NodeName={name} Arch=x86_64 CoresPerSocket=16  CPUAlloc={alloc} CPUEfctv={total} CPUTot={total} "
            f"CPULoad=1.00 AvailableFeatures={feats} ActiveFeatures={feats} Gres={gres} NodeAddr={name} "
            f"NodeHostName={name} Version=26.05.3 OS=Linux 5.15.0-187-generic #197-Ubuntu SMP Fri Jul 17 "
            f"19:17:01 UTC 2026  RealMemory={mem} AllocMem={alloc_mem} FreeMem={mem - alloc_mem} Sockets=8 "
            f"Boards=1 State={state} ThreadsPerCore=1 TmpDisk=150000 Weight=501 Owner=N/A MCS_label=N/A "
            f"Partitions={parts}  BootTime=2026-09-21T13:18:07 SlurmdStartTime=2026-09-24T10:46:14 "
            f"CfgTRES=cpu={total},mem={mem}M,billing=225{cfg_extra} "
            f"AllocTRES=cpu={alloc},mem={alloc_mem}M{alloc_extra} CurrentWatts=0 AveWatts=0")


NODES = [
    # genA: amd24 nodes in general-short and general-long
    _node("cpu-001", "MIXED", 104, 128, 500000, 393216, "cpuA,amd24", "general-short,general-long"),
    _node("cpu-002", "IDLE", 0, 128, 500000, 0, "cpuA,amd24", "general-short,general-long"),
    _node("cpu-003", "ALLOCATED", 128, 128, 500000, 480000, "cpuA,amd24", "general-short,general-long"),
    _node("cpu-004", "IDLE+DRAIN", 0, 128, 500000, 0, "cpuA,amd24", "general-short,general-long"),
    _node("cpu-005", "MIXED+PLANNED", 64, 128, 500000, 250000, "cpuA,amd24", "general-short"),
    # intel18 nodes
    _node("cpu-101", "MIXED", 20, 40, 180000, 80000, "cpuB,intel18", "general-short,general-long,group-a"),
    _node("cpu-102", "IDLE", 0, 40, 180000, 0, "cpuB,intel18", "general-short,general-long,group-a"),
    _node("cpu-103", "DOWN+NOT_RESPONDING", 0, 40, 180000, 0, "cpuB,intel18", "general-short"),
    # GPU nodes
    _node("gpu-001", "MIXED", 16, 128, 500000, 104304, "gpuC,amd21,gpgpu,a100", "general-short,general-long-gpu",
          gres="gpu:a100:4(S:1,3,5,7)", cfg_extra=",gres/gpu=4,gres/gpu:a100=4",
          alloc_extra=",gres/gpu=3,gres/gpu:a100=3"),
    _node("gpu-002", "IDLE", 0, 128, 500000, 0, "gpuC,amd21,gpgpu,a100", "general-short,general-long-gpu",
          gres="gpu:a100:4(S:1,3,5,7)", cfg_extra=",gres/gpu=4,gres/gpu:a100=4"),
    # a node only in a group partition the user may not use
    _node("cpu-201", "IDLE", 0, 64, 250000, 0, "cpuD,amd22", "group-other"),
]
NODES_TEXT = "\n".join(NODES)


def _part(name, nodes, max_time="7-00:00:00", allow_acc="ALL", allow_qos="ALL", deny_qos=None, hidden="NO",
          allow_groups="ALL"):
    deny = f" DenyQos={deny_qos}" if deny_qos else ""
    aq = f" AllowQos={allow_qos}" if allow_qos else ""
    return (f"PartitionName={name} AllowGroups={allow_groups} AllowAccounts={allow_acc}{aq}{deny} AllocNodes=ALL "
            f"Default=NO QoS=N/A DefaultTime=00:01:00 DisableRootJobs=YES Exclusive=NO GraceTime=0 Hidden={hidden} "
            f"MaxNodes=UNLIMITED MaxTime={max_time} MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED Nodes={nodes} "
            f"PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO OverTimeLimit=NONE "
            f"PreemptMode=OFF State=UP TotalCPUs=1000 TotalNodes=10 SelectTypeParameters=NONE")


PARTITIONS_TEXT = "\n".join([
    _part("general-short", "cpu-[001-005,101-103],gpu-[001-002]", max_time="04:00:00", allow_qos=None,
          deny_qos="large"),
    _part("general-long", "cpu-[001-004,101-102]", allow_qos=None, deny_qos="large"),
    _part("general-long-gpu", "gpu-[001-002]", allow_qos=None),
    _part("group-a", "cpu-[101-102]", allow_acc="acct1", allow_qos="normal", hidden="YES"),
    _part("group-other", "cpu-201", allow_acc="other_acct", allow_qos="normal", hidden="YES"),
    # a visible partition that is restricted to another account
    _part("visible-restricted", "cpu-[101-102]", allow_acc="other_acct", allow_qos="ALL"),
])

RES_NONE = "No reservations in the system"


def res_text(start, end, nodes="ALL", flags="MAINT,IGNORE_JOBS,SPEC_NODES,ALL_NODES", name="maint1"):
    return (f"ReservationName={name} StartTime={start} EndTime={end} Duration=08:00:00 Nodes={nodes} "
            f"NodeCnt=11 CoreCnt=1000 Features=(null) PartitionName=(null) Flags={flags} TRES=cpu=1000 "
            f"Users=root Groups=(null) Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) "
            f"MaxStartDelay=(null)")


ASSOC_TEXT = "\n".join([
    "acct1||normal|520|1000||||normal",
    "acct2||normal|520|1000||||normal",
    "scav||scavenger|5000|5000||||scavenger",
])
QOS_TEXT = "\n".join([
    "normal|1000||cpu=1040|",
    "scavenger|||cpu=4116|",
    "large|||cpu=8000|",
])
SSHARE_TEXT = "\n".join([
    "Account|User|RawShares|NormShares|RawUsage|NormUsage|EffectvUsage|FairShare|GrpTRESMins|TRESRunMins",
    "acct1|user1|1|0.000046|59222656|0.004844|0.004844|0.234443||cpu=56479,mem=650643456",
    "acct2|user1|1|0.000046|0|0.000000|0.000000|1.000000||cpu=0,mem=0",
])


def assoc_mgr(user="user1", used=812, limit=1000, account="acct1"):
    return (f"ClusterName=cpu-cluster Account={account} UserName={user}(1001) Partition= Priority=0 ID=7\n"
            f"    SharesRaw/Norm/Level/Factor=1/0.00/1/0.23\n"
            f"    GrpJobs=N(0) GrpJobsAccrue=N(0) GrpSubmitJobs=N(0) GrpWall=N(0.00)\n"
            f"    MaxJobs=520(12) MaxJobsAccrue= MaxSubmitJobs={limit}({used}) MaxWallPJ=\n"
            f"    DefAssoc=Yes\n")


TEST_ONLY_NOISE = [
    "sbatch: Replacing '#!/bin/bash' with '#!/bin/bash --login' in job script",
    "sbatch: For more information, please visit:",
    "sbatch:   https://docs.example.edu/faq",
]
TEST_ONLY_ACCEPT = ("sbatch: Job 4100 to start at 2030-01-01T21:38:00 a using 24 processors on nodes "
                    "cpu-[001-002] in partition general-long")
TEST_ONLY_REFUSE_TIME = "allocation failure: Requested time limit is invalid (missing or exceeds some limit)"
TEST_ONLY_REFUSE_FEATURE = "allocation failure: Invalid feature specification"

# the site quota tool's table (ANSI colors, box drawing), generalized
ESC = "\x1b"


def _c(t, color=32):
    return f"{ESC}[{color}m{t}{ESC}[0m"


SITE_QUOTA_TABLE = "\n".join([
    "+------------------------------+-------------+------------+------------+---------+-------------+------------+------------+---------+",
    "| " + " | ".join(_c(h, 37) for h in ["Directory                   ", "Space Quota", "Space Used", "Space Left",
                                          "Space %", "Files Quota", "Files Used", "Files Left", "Files %"]) + " |",
    "+------------------------------+-------------+------------+------------+---------+-------------+------------+------------+---------+",
    "| " + _c("home (NFS)                  ", 37) + " | " + " | ".join(_c(x) for x in
        ["100G", "87.1G", "12.9G", "87.1%", "1.0M", "342.0K", "658.0K", "34.2%"]) + " |",
    "| " + _c("scratch                     ", 37) + " | " + " | ".join(_c(x) for x in
        ["50T", "2.8T", "47.2T", "5.6%", "1.0M", "910.0K", "90.0K", "91.0%"]) + " |",
    "| " + _c("project_space               ", 37) + " | " + " | ".join(_c(x) for x in
        ["2T", "1.2T", "858.1G", "60.0%", "2.1M", "1.6M", "468.3K", "77.7%"]) + " |",
    "+------------------------------+-------------+------------+------------+---------+-------------+------------+------------+---------+",
])

QUOTA_SW = """Disk quotas for user user1 (uid 1001):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
nfs-server.example:/home  89716M    100G    125G            342k   1000k   1000k
"""

LFS_QUOTA = """Disk quotas for usr user1 (uid 1001):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
   /lfs/scratch 9500000* 9000000 10000000  6d23h  120000  1000000 1100000       -
"""

DF_K = """Filesystem     1024-blocks      Used Available Capacity Mounted on
server:/home   1000000000 870000000 130000000      87% /home
"""
DF_I = """Filesystem       Inodes    IUsed     IFree IUse% Mounted on
server:/home   10000000  4000000   6000000   40% /home
"""


def fake(tmp_path, **state):
    """A FakeSlurm with the cluster texts (overridable)."""
    base = {"nodes_text": NODES_TEXT, "partitions_text": PARTITIONS_TEXT, "res_text": RES_NONE,
            "assoc_text": ASSOC_TEXT, "qos_text": QOS_TEXT, "sshare_text": SSHARE_TEXT,
            "assoc_mgr": assoc_mgr(), "cluster_name": "cpu-cluster",
            "sinfo_text": "general-short|up|4:00:00|10|5/3/2/10\ngeneral-long*|up|7-00:00:00|6|3/2/1/6",
            "test_only": {"accept": TEST_ONLY_ACCEPT, "noise": TEST_ONLY_NOISE}}
    base.update(state)
    fs = FakeSlurm(tmp_path / "slurm", base)
    fs.env["USER"] = "user1"
    fs.env["LOGNAME"] = "user1"
    fs.env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    fs.env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    fs.env["MACE_SITE_CONFIG"] = "none"
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        fs.env.pop(k, None)
    return fs


def use_env(monkeypatch, fs):
    """Make this process see the fake: PATH, cache, user."""
    for k in list(os.environ):
        if k.startswith("SLURM_"):
            monkeypatch.delenv(k, raising=False)
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "FAKE_SLURM_HOOK_LOG", "USER", "LOGNAME",
              "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MACE_SITE_CONFIG"):
        monkeypatch.setenv(k, fs.env[k])
    from mace.cluster import query
    query._which_cache.clear()
    query._name_cache.clear()


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """Every cluster test gets its own cache and config folders (imported by
    each test module): nothing reads or writes the real ~/.cache/mace or
    ~/.config/mace."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    from mace.cluster import query
    query._name_cache.clear()
    query._which_cache.clear()
