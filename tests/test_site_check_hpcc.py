"""`mace site check` findings from running it on MSU HPCC itself (the
gateway, a dev node, a non-MSU site.toml), through the real mace_cli with
the fake SLURM and fake modules.

* On a login node that serves no modules (MSU's gateway) the Python modules
  cannot be tried, exactly like the CRYSTAL module: a warning that says to
  check on a compute or dev node, not a failure. Elsewhere a Python module
  that does not load still fails the check.
* A Resources warning names the fix of its own cause: a partition time
  limit points at [slurm] partition, only nodes > 1 at [paths] scratch.
"""
import json

from test_site_check import SINFO, _check, _items
from test_site_generic_cluster import _cluster
from test_site_jobscript import _toml


def test_owner_gateway_python_modules_warn_not_fail(tmp_path):
    env, fake = _cluster(tmp_path, owner=True, gateway=True)
    r, tmpdir = _check(tmp_path, env, "--json")
    py = _items(r)["Completion Python"]
    assert py["status"] == "warn", py
    assert "login node" in py["detail"] and "compute or development node" in py["detail"]
    assert "Python/3.11.3-GCCcore-12.3.0" in py["detail"]
    assert json.loads(r.stdout)["summary"]["fail"] == 0
    assert list(tmpdir.iterdir()) == []


def test_python_module_that_does_not_load_still_fails_off_a_login_node(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\n[modules]\npython = ["Python/9.9"]\n')
    r, _ = _check(tmp_path, env, "--json")
    py = _items(r)["Completion Python"]
    assert py["status"] == "fail", py
    assert "[modules] python" in py["fix"]
    assert r.returncode == 2


def test_python_module_on_another_clusters_gateway_warns(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    env = dict(env, MACE_HOSTNAME="gateway-1.example.edu")
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\n[modules]\npython = ["Python/9.9"]\n')
    r, _ = _check(tmp_path, env, "--json")
    py = _items(r)["Completion Python"]
    assert py["status"] == "warn" and "the host is gateway-1.example.edu" in py["detail"], py


def test_walltime_cap_warning_points_at_the_partition(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(sinfo=SINFO)
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\n')
    r, _ = _check(tmp_path, env, "--json")
    res = _items(r)["Resources"]
    assert res["status"] == "warn" and "allows at most" in res["detail"], res
    assert "[slurm] partition" in res["fix"] and "[paths] scratch" not in res["fix"], res


def test_multi_node_warning_points_at_scratch(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(sinfo=SINFO)
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\npartition = "long"\n[resources]\nnodes = 2\n')
    r, _ = _check(tmp_path, env, "--json")
    res = _items(r)["Resources"]
    assert res["status"] == "warn" and "nodes > 1" in res["detail"], res
    assert "[paths] scratch" in res["fix"], res
