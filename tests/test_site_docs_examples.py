"""Every site.toml example the docs print (docs/cli.md, INSTALLATION.md) is
read as written, passes `mace site check` on a fake cluster that has what it
names, and gives job scripts that bash accepts and that carry each value.

The blocks are taken from the docs files themselves, so an edit that breaks
an example (a key renamed, a value MACE refuses) fails here.
"""
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from fake_modules import FakeModules
from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_generic_cluster import ASSOC_MGR, SCRIPT_PATTERN
from test_site_jobscript import _deck, _env, _mace, _toml

REPO = Path(__file__).resolve().parents[1]
TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.S)


def _examples():
    out = []
    for doc in ("docs/cli.md", "INSTALLATION.md"):
        for i, block in enumerate(TOML_BLOCK.findall((REPO / doc).read_text())):
            if "[slurm]" in block:
                out.append(pytest.param(block, id=f"{doc}#{i}"))
    return out


EXAMPLES = _examples()
PARTS = ("general", "compute", "my_partition")
SINFO = "".join(f"c{i:03d}|idle|2|32|2|128|256000|epyc,ib|(null)|{p}{'*' if p == 'general' else ''}\n"
                for p in PARTS for i in range(2))


def _cluster(tmp, data):
    accounts = sorted({data.get("slurm", {}).get("account", "lab"), "lab"})
    qos = data.get("slurm", {}).get("qos")
    parts = "".join(f"PartitionName={p} AllowAccounts=ALL AllowQos=ALL MaxTime=7-00:00:00 State=UP"
                    f"{' Default=YES' if p == 'general' else ''}\n" for p in PARTS)
    fake = FakeSlurm(tmp / "fs", {"assoc": accounts, "default_account": accounts[0],
                                  "qos": [qos] if qos else ["normal"], "partitions": parts,
                                  "assoc_mgr": ASSOC_MGR.replace("grp_x", accounts[0]), "sinfo": SINFO,
                                  "srun_mpi_list": "MPI plugin types are...\n\tnone\n\tpmi2\n\tpmix\n"})
    mods = data.get("modules", {})
    crystal = mods.get("crystal", "CRYSTAL/23-foss-2023a")
    crystal = [crystal] if isinstance(crystal, str) else list(crystal)
    spec = {m: {"bins": ()} for m in crystal[:-1]}
    spec[crystal[-1]] = {"root": True, **({"needs": crystal[-2]} if len(crystal) > 1 else {})}
    for m in mods.get("python", []):
        spec[m] = {"bins": ()}
    fm = FakeModules(tmp / "fm", spec, python=False)
    env = _env(tmp, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="login1.hpc.example.org")
    return fm.env(env), fake


@pytest.mark.parametrize("text", EXAMPLES)
def test_docs_example_runs_on_a_fake_cluster(tmp_path, text):
    data = tomllib.loads(text)
    env, fake = _cluster(tmp_path, data)
    (tmp_path / "t").mkdir()
    env["TMPDIR"] = str(tmp_path / "t")             # where mace site check makes its folder
    _toml(tmp_path, text)
    # read as written: no warning, no error
    r = _mace(tmp_path, env, "site", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    j = json.loads(r.stdout)
    assert not j["warnings"], j["warnings"]
    # mace site check: nothing fails; the generated scripts pass sbatch --test-only
    r = _mace(tmp_path, env, "site", "check", "--json")
    rep = json.loads(r.stdout)
    st = {i["item"]: i for i in rep["items"]}
    assert not [i for i in rep["items"] if i["status"] == "fail"], rep["items"]
    assert st["Job script"]["status"] == "ok", st["Job script"]
    assert not list((tmp_path / "t").glob("mace-site-check-*"))
    # real submissions
    d = tmp_path / "decks"
    _deck(d, "m.d12", SP_D12)
    _deck(d, "m_band.d3", BAND_D3)
    (d / "m_band.f9").write_text("fort9\n")
    for deck in ("m.d12", "m_band.d3"):
        p = _mace(d, env, "submit", deck)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    scripts = {c["script"]: c["script_text"] for c in fake.calls("sbatch") if "--test-only" not in c["argv"]}
    slurm = data.get("slurm", {})
    for name, s in scripts.items():
        assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
        assert not SCRIPT_PATTERN.search(s), SCRIPT_PATTERN.search(s)
        sb = [l.split(" #")[0].rstrip() for l in s.splitlines() if l.startswith("#SBATCH")]
        if slurm.get("account"):
            assert f"#SBATCH -A {slurm['account']}" in sb
        if slurm.get("partition"):
            assert f"#SBATCH -p {slurm['partition']}" in sb
        for key, flag in (("qos", "--qos="), ("constraint", "--constraint=")):
            if slurm.get(key):
                assert f"#SBATCH {flag}{slurm[key]}" in sb
        for item in slurm.get("extra", []):
            assert f"#SBATCH {item}" in sb
        for item in data.get("mpi", {}).get("env", []):
            assert f"export {item}" in s
        if data.get("paths", {}).get("scratch"):
            assert data["paths"]["scratch"].replace("$USER", "$MACE_USER") in s
        if data.get("modules", {}).get("purge") is False:
            assert "\nmodule purge\n" not in s
    res = data.get("resources", {})
    if "ntasks" in res:
        assert f"#SBATCH --ntasks={res['ntasks']}" in scripts["m.sh"]
    prop = {**{k: v for k, v in res.items() if k != "properties"}, **res.get("properties", {})}
    if "ntasks" in prop:
        assert f"#SBATCH --ntasks={prop['ntasks']}" in scripts["m_band.sh"]
    if data.get("mpi", {}).get("launcher") == "srun":
        assert re.search(r'^  srun .*P(crystal|properties)" 2>&1', scripts["m.sh"], re.M)
