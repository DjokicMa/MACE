"""F6: association/QOS limits, own usage, fairshare and headroom warnings."""
from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import ASSOC_TEXT, QOS_TEXT, SSHARE_TEXT, assoc_mgr, fake, use_env

from mace.cluster import limits, reader
from mace.submission.job_array import parse_submit_slots


def test_limits_per_account():
    """F6-a: MaxJobs 520, MaxSubmit 1000, cores 1040 per user; fairshare."""
    assoc = limits.parse_assoc(ASSOC_TEXT)
    qos = limits.parse_qos(QOS_TEXT)
    assert assoc[0] == {"account": "acct1", "partition": None, "qos": ["normal"], "max_jobs": 520,
                        "max_submit": 1000, "max_wall": None, "grp_tres": None, "max_tres": None,
                        "max_cpus": None, "default_qos": "normal"}
    assert qos["normal"]["max_cpus_pu"] == 1040 and qos["normal"]["max_submit_pu"] == 1000
    eff = limits.effective_limits(assoc, qos, "acct1")
    assert eff == {"max_jobs": 520, "max_submit": 1000, "max_cpus": 1040}
    fs = limits.parse_sshare(SSHARE_TEXT)
    assert fs[0]["account"] == "acct1" and fs[0]["fairshare"] == 0.234443
    assert fs[0]["effective_usage"] == 0.004844
    assert parse_submit_slots(assoc_mgr(used=812), "user1", None) == (1000, 812)


def test_own_counts_with_folded_array():
    """F6-b: a folded array A_[2-300%10] counts 299 pending tasks; running
    cores from %C."""
    rows = [{"id": "4242_[2-300%10]", "state": "PENDING", "cpus": "32", "account": "acct1"},
            {"id": "4242_1", "state": "RUNNING", "cpus": "32", "account": "acct1"},
            {"id": "5000", "state": "RUNNING", "cpus": "16", "account": "acct2"}]
    u = limits.own_usage(rows)
    assert u["pending"] == 299 and u["running"] == 2 and u["submitted"] == 301
    assert u["running_cpus"] == 48
    assert u["by_account"]["acct1"]["submitted"] == 300


def test_headroom_edges():
    """F6-c: exactly full, one over, unlimited."""
    lim = {"max_jobs": 520, "max_submit": 1000, "max_cpus": 1040}
    usage = {"submitted": 812, "running": 12, "running_cpus": 384}
    w = limits.headroom(300, 32, usage, lim, submit_slots=(1000, 812))
    assert "This adds 300 jobs; 812 of 1000 submit slots are in use, so 112 more are refused by SLURM." in w
    assert ("Running cores would exceed your QOS limit (1040): extra jobs wait with reason "
            "QOSMaxCpuPerUserLimit.") in w
    assert not [x for x in limits.headroom(188, 1, usage, {"max_submit": 1000}, (1000, 812)) if "submit" in x]
    one = limits.headroom(189, 1, usage, {"max_submit": 1000}, (1000, 812))
    assert one == ["This adds 189 jobs; 812 of 1000 submit slots are in use, so 1 more is refused by SLURM."]
    assert limits.headroom(5000, 1000, usage, {"max_jobs": None, "max_submit": None, "max_cpus": None}) == []


def test_reader_assoc_limits(tmp_path, monkeypatch):
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    r = reader.assoc_limits()
    assert r["assoc"].value[0]["account"] == "acct1"
    assert r["qos"].value["normal"]["max_cpus_pu"] == 1040
    assert "MaxSubmitJobs=1000(812)" in r["usage"].value
    assert reader.fairshare().value[1]["fairshare"] == 1.0
