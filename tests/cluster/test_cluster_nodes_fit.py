"""F2/F3: node and partition parsing, the usable-partition filter, the
"could start now" filter, sbatch --test-only answers and reservations."""
import time

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import (NODES_TEXT, PARTITIONS_TEXT, TEST_ONLY_ACCEPT, TEST_ONLY_NOISE,
                              TEST_ONLY_REFUSE_FEATURE, TEST_ONLY_REFUSE_TIME, _node, fake, res_text, use_env)

from mace.cluster import config, fit, nodes, reservations
from mace.recovery import slurm_limits


def _nodes():
    return nodes.parse_nodes(NODES_TEXT)


def _parts(accounts=("acct1", "acct2"), qos=("normal",)):
    return nodes.usable_partitions(nodes.parse_partitions(PARTITIONS_TEXT), accounts, [], qos)


def test_parse_node_fields():
    n = {x["name"]: x for x in _nodes()}
    assert n["cpu-001"]["state"] == "mixed" and n["cpu-001"]["cpus_alloc"] == 104
    assert n["cpu-004"]["flags"] == ["DRAIN"]
    assert n["cpu-005"]["flags"] == ["PLANNED"]
    assert n["gpu-001"]["gres_total"] == {"a100": 4} and n["gpu-001"]["gres_used"] == {"a100": 3}
    assert n["cpu-101"]["features"] == ["cpuB", "intel18"]
    assert n["cpu-001"]["partitions"] == ["general-short", "general-long"]


@pytest.mark.parametrize("text, base, flags", [
    ("mixed-", "mixed", ["PLANNED"]), ("idle~", "idle", ["POWERED_DOWN"]),
    ("drained*", "drained", ["NOT_RESPONDING", "DRAIN"]), ("MIXED+DRAIN", "mixed", ["DRAIN"]),
    ("IDLE+MAINT+RESERVED", "idle", ["MAINT", "RESERVED"]), ("ALLOCATED+NEWFLAG", "allocated", ["NEWFLAG"]),
])
def test_state_strings(text, base, flags):
    assert nodes.split_state(text) == (base, flags)


def test_summary_counts_by_cpu_type():
    """F2-a: counts per group equal a hand count; drained nodes add no free
    CPUs; GPU free = total - AllocTRES."""
    rows = {r["group"]: r for r in nodes.summary(_nodes(), regex=config.DEFAULT_GROUP_REGEX)}
    a = rows["amd24"]
    assert (a["nodes"], a["idle"], a["mixed"], a["allocated"], a["down_drain"]) == (5, 1, 2, 1, 1)
    assert a["cpus_free"] == (128 - 104) + 128 + 0 + (128 - 64)
    assert a["cpus_total"] == 5 * 128
    b = rows["intel18"]
    assert (b["nodes"], b["idle"], b["mixed"], b["down_drain"], b["cpus_free"]) == (3, 1, 1, 1, 20 + 40)
    g = rows["amd21"]
    assert g["gpus_total"] == {"a100": 8} and g["gpus_free"] == {"a100": 1 + 4}


def test_usable_partitions_filter():
    """F2-b: hidden or visible, a partition is usable only when Allow* admits
    one of the user's accounts/groups/QOS; Deny* wins."""
    parts = {p["name"]: p for p in _parts()}
    assert parts["general-short"]["usable"] and parts["general-long"]["usable"]
    assert parts["group-a"]["usable"]                      # hidden, but the user's account
    assert not parts["group-other"]["usable"]
    assert not parts["visible-restricted"]["usable"]
    assert parts["visible-restricted"]["why_not"] == "not usable by your accounts"
    # DenyQos wins when the user's only QOS is denied
    p = _parts(qos=("large",))
    assert not {x["name"]: x for x in p}["general-short"]["usable"]
    # groups
    part = nodes.parse_partitions(PARTITIONS_TEXT.replace("PartitionName=group-other AllowGroups=ALL",
                                                          "PartitionName=group-other AllowGroups=grp1"))
    assert nodes.partition_usable([x for x in part if x["name"] == "group-other"][0], ["acct1"], ["grp1"],
                                  ["normal"]) == "not usable by your accounts"


def test_fit_filter():
    """F3-a: a 32-core request does not fit the node with 24 free cores; a
    24-core one does; constraints evaluated; brackets not."""
    one = nodes.parse_nodes(_node("cpu-301", "MIXED", 104, 128, 500000, 384 * 1024, "cpuA,amd24", "p1"))
    parts = [{"name": "p1", "max_time": "7-00:00:00", "usable": True}]
    r32 = fit.fit(fit.request_from(cpus=32, mem_per_cpu="5G"), one, parts)
    assert r32["partitions"][0]["fits_now"] == 0
    assert r32["unmet_reasons"] == {"not enough free cores": 1}
    r24 = fit.fit(fit.request_from(cpus=24, mem_per_cpu="4G"), one, parts)
    assert r24["partitions"][0]["fits_now"] == 1
    assert fit.eval_constraint("amd24|intel18", ["cpuA", "amd24"]) is True
    assert fit.eval_constraint("amd24&intel18", ["cpuA", "amd24"]) is False
    assert fit.eval_constraint("[a*2&b]", ["a"]) is None
    rb = fit.fit(fit.request_from(cpus=4, constraint="[a*2&b]"), one, parts)
    assert "not evaluated" in rb["constraint_note"]
    # walltime beyond MaxTime: nothing fits there
    parts2 = [{"name": "p1", "max_time": "04:00:00", "usable": True}]
    r = fit.fit(fit.request_from(cpus=4, time_text="1-00:00:00"), one, parts2)
    assert r["partitions"][0]["fits_now"] == 0 and "MaxTime" in r["partitions"][0]["note"]


def test_fit_gpus_and_usable_only():
    res = fit.fit(fit.request_from(cpus=8, gpus="a100:2"), _nodes(), _parts())
    by = {p["name"]: p for p in res["partitions"]}
    assert by["general-long-gpu"]["fits_now"] == 1          # gpu-002 only (gpu-001 has 1 free)
    assert "group-other" not in by and "visible-restricted" not in by


@pytest.mark.parametrize("text, rc, accepted, start, nodes_, part, reason", [
    ("\n".join(TEST_ONLY_NOISE + [TEST_ONLY_ACCEPT]), 0, True, "2030-01-01T21:38:00", "cpu-[001-002]",
     "general-long", None),
    ("sbatch: Job 12 to start at 2030-01-02T03:04:05 using 4 processors on nodes cpu-007 in partition p", 0,
     True, "2030-01-02T03:04:05", "cpu-007", "p", None),
    ("\n".join([n.replace("sbatch:", "sbatch: error:") for n in TEST_ONLY_NOISE] + [TEST_ONLY_REFUSE_TIME]), 1,
     False, None, None, None, "Requested time limit is invalid (missing or exceeds some limit)"),
    (TEST_ONLY_REFUSE_FEATURE, 1, False, None, None, None, "Invalid feature specification"),
    ("sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
     1, False, None, None, None, "Batch job submission failed: Invalid account or account/partition "
                                 "combination specified"),
])
def test_test_only_parse(text, rc, accepted, start, nodes_, part, reason):
    """F3-b: accepted with start/nodes/partition, refused with the reason;
    site filter noise ignored; the " a " token tolerated."""
    r = slurm_limits.parse_test_only(text, rc)
    assert (r["accepted"], r["start"], r["nodes"], r["partition"], r["reason"]) == \
        (accepted, start, nodes_, part, reason)


def test_reservation_conflicts():
    """F3-c: a MAINT reservation 2 days ahead conflicts with a 3-day request
    on an overlapping partition, not with a 1-day one."""
    now = time.time()
    fmt = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    res = reservations.parse(res_text(fmt(now + 2 * 86400), fmt(now + 2 * 86400 + 8 * 3600),
                                      nodes="cpu-[001-003]"))
    assert res[0]["maintenance"] and res[0]["node_count"] == 11
    parts = [p for p in nodes.parse_partitions(PARTITIONS_TEXT) if p["name"] == "general-long"]
    assert len(reservations.conflicts(3 * 86400, now, parts, res, _nodes())) == 1
    assert reservations.conflicts(1 * 86400, now, parts, res, _nodes()) == []
    other = [p for p in nodes.parse_partitions(PARTITIONS_TEXT) if p["name"] == "general-long-gpu"]
    assert reservations.conflicts(3 * 86400, now, other, res, _nodes()) == []
    assert reservations.parse("No reservations in the system") == []
    assert reservations.expand_hostlist("cpu-[001-003,007],gpu-01") == ["cpu-001", "cpu-002", "cpu-003",
                                                                        "cpu-007", "gpu-01"]


def test_existing_test_only_unchanged():
    """F3-e: the old sbatch_test_only keeps its signature and answers."""
    import inspect
    assert list(inspect.signature(slurm_limits.sbatch_test_only).parameters) == ["script_path", "walltime", "runner"]


def test_predict_same_inside_allocation(tmp_path, monkeypatch):
    """F3-f: predict() with and without allocation variables records the same
    argv, cwd and SLURM_* set."""
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    script = tmp_path / "job" / "mat1.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n#SBATCH -n 4\n#SBATCH -t 1:00:00\ntrue\n")
    out = fit.predict(script_path=script)
    assert out["accepted"] and out["start"] == "2030-01-01T21:38:00"
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1000")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm.conf")
    script.write_text(script.read_text() + "\n")            # a new question (no cache hit)
    fit.predict(script_path=script)
    a, b = fs.calls("sbatch")
    assert a["argv"] == b["argv"] and a["cwd"] == b["cwd"] == str(script.parent)
    assert set(b["env"]) - {"SLURM_CONF"} == set(a["env"])
    assert not [c for c in fs.calls("sbatch") if "--test-only" not in c["argv"]]
