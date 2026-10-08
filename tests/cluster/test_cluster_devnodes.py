"""F10-a: dev-node ranking with a fake ssh; an unreachable host is never picked."""
import os

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from mace.cluster import config, devnodes, query


def _fake_ssh(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "ssh").write_text("""#!/bin/bash
host=""
for a in "$@"; do case "$a" in -o) skip=1;; *) if [ -n "$skip" ]; then skip=; elif [ -z "$host" ]; then host="$a"; fi;; esac; done
case "$host" in
  dev-node-1) printf '64.0 60.0 55.0 3/900 1\\n128\\nMemAvailable:   400000000 kB\\n';;
  dev-node-2) printf '3.5 3.1 3.0 2/800 1\\n128\\nMemAvailable:   300000000 kB\\n';;
  dev-node-3) sleep 0.2; echo "ssh: connect to host dev-node-3 port 22: Connection timed out" >&2; exit 255;;
esac
""")
    (b / "ssh").chmod(0o755)
    return b


def test_rank_and_pick(tmp_path, monkeypatch):
    b = _fake_ssh(tmp_path)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    query._which_cache.clear()
    query._name_cache.clear()
    site = tmp_path / "site.toml"
    site.write_text('[devnodes]\nhosts = ["dev-node-1", "dev-node-2", "dev-node-3"]\n')
    cfg = config.load({"MACE_SITE_CONFIG": str(site)})
    r = devnodes.probe_all(cfg=cfg)
    assert r["pick"] == "dev-node-2"
    assert [x["host"] for x in r["rows"]] == ["dev-node-2", "dev-node-1", "dev-node-3"]
    assert r["rows"][-1]["reachable"] is False and "timed out" in r["rows"][-1]["error"]
    site.write_text('[devnodes]\nhosts = ["dev-node-1", "dev-node-2"]\npick_by = "memory"\n')
    r = devnodes.probe_all(cfg=config.load({"MACE_SITE_CONFIG": str(site)}))
    assert r["pick"] == "dev-node-1"


def test_no_hosts():
    r = devnodes.probe_all(cfg=config.load({"MACE_SITE_CONFIG": "none"}))
    assert r["message"] == devnodes.NO_HOSTS and r["pick"] is None
