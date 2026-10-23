"""The RCSR list of 3-periodic nets bundled with MACE (mace/analysis/rcsr_data):
it loads, matches its NOTICE.txt checksums, names common nets with no import
step, is credited in the topology output, gives way to a list you import, and
keeps the repository lean."""
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("spglib")
pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402

from mace.analysis import rcsr, structure, topology  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "mace_cli"
SI = REPO / "examples" / "gui" / "structures" / "Si.cif"
DATA = REPO / "mace" / "analysis" / "rcsr_data"
NOTICE = DATA / "NOTICE.txt"
PCU_CS = [6, 18, 38, 66, 102, 146, 198, 258, 326, 402]


@pytest.fixture
def no_import(tmp_path, monkeypatch):
    """An empty MACE config folder: nothing imported."""
    monkeypatch.setenv("MACE_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"


def notice_entry(name):
    m = re.search(rf"^\s*{re.escape(name)}\s+(\d+) bytes\s+sha256 ([0-9a-f]{{64}})\s*$", NOTICE.read_text(), re.M)
    assert m, name
    return int(m.group(1)), m.group(2)


def mini_3dall(nets):
    """A file in RCSR's 3dall.txt layout: one vertex and one edge per net."""
    blocks = []
    for i, (name, cs) in enumerate(nets, 1):
        blocks.append("\n".join([
            "start", str(i), name, "1a", " 0  !number of other symbols", " 0  !number of names",
            " 0  !number of other names", " 0  !number of keywords", " 0  !number of references", "Pm-3m   221",
            "  1.0 1.0 1.0 90.0 90.0 90.0", "1", f"V1  {cs[0]}", "  0 0 0", "  0, 0, 0", "  1 a", "  m-3m", "  48", "1",
            "E1  2", "  0.5 0 0", "  1/2, 0, 0", "  3 d", "  4/mmm", "1", "1", "1", "[x]", "unk",
            "  " + "  ".join(map(str, cs)) + "  " + str(1 + sum(cs)), "  x", "1", "1"]))
    return "\n".join(blocks) + "\n"


def mace(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = "Agg"
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(CLI), "--no-banner", "analyze", *map(str, args)], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------- the file and its notice
def test_bundled_file_matches_its_notice():
    raw_size, raw_sha = notice_entry("3dall.txt")
    xz_size, xz_sha = notice_entry("3dall.txt.xz")
    packed = rcsr.BUNDLED_FILE.read_bytes()
    assert rcsr.BUNDLED_FILE.parent == DATA
    assert len(packed) == xz_size and hashlib.sha256(packed).hexdigest() == xz_sha
    raw = lzma.decompress(packed)
    assert len(raw) == raw_size and hashlib.sha256(raw).hexdigest() == raw_sha
    assert rcsr.bundled_text().encode() == raw


def test_bundled_nets_load_and_the_notice_credits_rcsr():
    nets = rcsr.bundled()
    text = " ".join(NOTICE.read_text().split())
    count = int(re.search(r"Nets: (\d+)", text).group(1))
    assert len(nets) == count > 3000
    assert len({n["name"] for n in nets}) == len(nets)
    assert all(n["dims"] == 3 and n["source"] == rcsr.BUNDLED_SOURCE and len(n["cs"]) == len(n["cn"]) for n in nets)
    assert rcsr.BUNDLED_DATE in text and rcsr.DOWNLOAD_URL in text
    for need in ("doi:10.1021/ar800124u", "O'Keeffe", "Peskov", "Ramsden", "Yaghi", "Acc. Chem. Res. 41, 1782-1789 (2008)",
                 "no license or terms of use", "MIT License does not cover it"):
        assert need in text, need
    assert "doi:10.1021/ar800124u" in rcsr.CITATION


def test_the_repository_stays_lean():
    assert rcsr.BUNDLED_FILE.stat().st_size < 1_000_000
    assert sorted(p.name for p in DATA.iterdir() if not p.name.startswith(".")) == ["3dall.txt.xz", "NOTICE.txt"]


# ---------------------------------------------------------------- recognition with no import step
def quartz():
    # alpha-quartz, P3(2)21, a = 4.913, c = 5.405 A: its Si net is qtz
    return structure.from_atoms(crystal(["Si", "O"], basis=[(0.4697, 0, 2 / 3), (0.4135, 0.2669, 0.7858)], spacegroup=154,
                                        cellpar=[4.913, 4.913, 5.405, 90, 90, 120]))


def sodalite():
    return structure.from_atoms(crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229,
                                        cellpar=[8.965] * 3 + [90] * 3))


@pytest.mark.parametrize("name,make", [
    ("dia", lambda: structure.from_atoms(bulk("Si", "diamond", a=5.431, cubic=True))),
    ("pcu", lambda: structure.from_atoms(bulk("NaCl", "rocksalt", a=5.64, cubic=True))),
    ("bcu", lambda: structure.from_atoms(bulk("CsCl", "cesiumchloride", a=4.12))),
    ("qtz", quartz),
    ("sod", sodalite),
])
def test_common_nets_come_from_the_bundled_list(no_import, name, make):
    r = topology.analyze(make())
    assert r["summary"]["topologies"] == [name]
    m = r["nets"][0]["matches"][0]
    assert m["name"] == name and m["source"] == rcsr.BUNDLED_SOURCE
    assert r["citation"] == rcsr.CITATION


def test_nets_only_rcsr_knows_are_in_the_reference_set(no_import):
    names = {(n["name"], n["dims"]) for n in rcsr.references()}
    assert {("fau", 3), ("lta", 3), ("cha", 3), ("rho", 3)} <= names
    assert {("hcb", 2), ("sql", 2)} <= names                          # the built-in layers stay
    assert len(names) == len(rcsr.references())                        # one entry per name


# ---------------------------------------------------------------- the CLI credits RCSR
def test_cli_topology_prints_the_citation(tmp_path):
    env = {"MACE_CONFIG_DIR": str(tmp_path / "config")}
    r = mace("topology", SI, env_extra=env)
    assert r.returncode == 0, r.stderr
    assert "topology dia" in r.stdout and "doi:10.1021/ar800124u" in r.stdout and "please cite" in r.stdout
    assert "only the built-in nets are known" not in r.stdout
    r = mace("topology", SI, "--json", env_extra=env)
    d = json.loads(r.stdout)
    assert d["citation"] == rcsr.CITATION and d["nets"][0]["matches"][0]["source"] == rcsr.BUNDLED_SOURCE
    r = mace("rcsr", "status", env_extra=env)
    assert r.returncode == 0 and f"bundled RCSR nets: {len(rcsr.bundled())}" in r.stdout
    assert "doi:10.1021/ar800124u" in r.stdout


# ---------------------------------------------------------------- your import wins
def test_an_imported_list_is_tried_before_the_bundled_one(no_import, tmp_path):
    f = tmp_path / "3dall.txt"
    f.write_text(mini_3dall([("zzz", PCU_CS), ("bcu", [8, 26, 56, 98, 152, 218, 296, 386, 488, 602])]))
    rcsr.import_file(f)
    nacl = structure.from_atoms(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    m = topology.analyze(nacl)["nets"][0]["matches"]
    assert [x["name"] for x in m][:2] == ["zzz", "pcu"]                 # the import first, the bundled net after
    assert m[0]["source"] == "RCSR 3dall.txt" and m[1]["source"] == rcsr.BUNDLED_SOURCE
    cscl = structure.from_atoms(bulk("CsCl", "cesiumchloride", a=4.12))
    m = topology.analyze(cscl)["nets"][0]["matches"]
    assert m[0]["name"] == "bcu" and m[0]["source"] == "RCSR 3dall.txt"      # the imported bcu replaces the bundled one
    assert [x["name"] for x in m].count("bcu") == 1
    st = rcsr.status()
    assert st["imported"] == 2 and st["bundled"] == len(rcsr.bundled())


def test_cli_import_overrides_the_bundled_list(tmp_path):
    env = {"MACE_CONFIG_DIR": str(tmp_path / "config")}
    f = tmp_path / "3dall.txt"
    f.write_text(mini_3dall([("dia", [4, 12, 24, 42, 64, 92, 124, 162, 204, 252])]))
    r = mace("rcsr", "import", f, env_extra=env)
    assert r.returncode == 0 and "imported 1 nets" in r.stdout, r.stderr
    d = json.loads(mace("topology", SI, "--json", env_extra=env).stdout)
    assert d["nets"][0]["matches"][0] == {**d["nets"][0]["matches"][0], "name": "dia", "source": "RCSR 3dall.txt"}
    r = mace("rcsr", "status", env_extra=env)
    assert "imported nets: 1" in r.stdout and "used before the bundled list" in r.stdout


def test_without_the_bundled_file_the_built_in_nets_still_answer(no_import, monkeypatch, tmp_path):
    monkeypatch.setattr(rcsr, "BUNDLED_FILE", tmp_path / "missing.xz")
    monkeypatch.setattr(rcsr, "_BUNDLED_CACHE", None)
    assert rcsr.bundled() == [] and rcsr.status()["bundled"] == 0
    m = topology.analyze(structure.from_atoms(bulk("Si", "diamond", a=5.431, cubic=True)))["nets"][0]["matches"][0]
    assert m["name"] == "dia" and m["source"] == "built-in"
