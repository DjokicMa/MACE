"""mace/utils/envmodules.py against a fake module system (tests/envmodules_fake.py)."""
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envmodules_fake as fake  # noqa: E402

from mace.utils import envmodules  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

XTB_MODS = ["xtb/6.6.1-gfbf-2023a", "xtb/6.7.0-foss-2023b", "xtb/6.10.0-x", ".xtb/9.0"]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    envmodules._MEM.clear()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    yield
    envmodules._MEM.clear()


@pytest.mark.parametrize("module,key", [
    ("xtb/6.10.0-x", (6, 10, 0)), ("xtb/6.7.1-gfbf-2024a", (6, 7, 1)), ("xtb/6.6.1-gfbf-2023a", (6, 6, 1)),
    ("CRYSTAL/23-intel-2023a", (23,)), ("CRYSTAL/17-x", (17,)), ("xtb/latest", ()), ("xtb/v6.7", (6, 7)),
])
def test_version_key(module, key):
    assert envmodules.version_key(module) == key


def test_version_order_is_numeric():
    assert envmodules.sort_newest(["xtb/6.7.1", "xtb/6.10.0", "xtb/6.6.1"])[0] == "xtb/6.10.0"
    assert envmodules.version_key("CRYSTAL/23-intel-2023a") >= (23,)


def test_parse_avail_drops_headers_hidden_and_markers():
    text = ("/opt/modules/all:\nxtb/6.6.1-gfbf-2023a\nxtb/6.7.1-gfbf-2024a (default)\n.xtb/9.0\n"
            "xtb/.hidden\nother/1.0\nxtb/6.8 <D>\n")
    assert envmodules.parse_avail(text, "xtb") == ["xtb/6.6.1-gfbf-2023a", "xtb/6.7.1-gfbf-2024a", "xtb/6.8"]


def test_no_markers_means_no_subprocess(monkeypatch, tmp_path):
    for k in ("LMOD_CMD", "LMOD_VERSION", "MODULESHOME", "MODULEPATH"):
        monkeypatch.delenv(k, raising=False)
    for k in list(os.environ):
        if k.startswith("BASH_FUNC_module"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(envmodules, "PROFILE_DIRS", (str(tmp_path),))
    import subprocess
    spawned = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError))
    t0 = time.perf_counter()
    assert envmodules.module_system(profile_dirs=(str(tmp_path),)) is None
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] is None and info["system"] is None and spawned == []
    assert time.perf_counter() - t0 < 0.05
    assert envmodules.hint("xtb", info, (6, 7, 0)) == ""


def test_discover_picks_the_newest_suitable_and_loads_it(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=XTB_MODS,
                    programs={"xtb/6.10.0-x": ("xtb", fake.XTB_671), "xtb/6.7.0-foss-2023b": ("xtb", fake.XTB_671)})
    fake.apply(monkeypatch, env)
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] == "xtb/6.10.0-x", info
    assert info["binary"].endswith("xtb_6.10.0-x/bin/xtb")
    assert "LD_LIBRARY_PATH" in info["env"] and "xtb_6.10.0-x/lib" in info["env"]["LD_LIBRARY_PATH"]
    assert not any(k.startswith("BASH_FUNC_") for k in info["env"]) and "PWD" not in info["env"]
    # hidden .xtb/9.0 never offered
    assert ".xtb/9.0" not in info["available"]
    # the cache: a second discovery runs no module command at all
    n = len(fake.calls(env))
    envmodules._MEM.clear()
    again = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert again["cached"] and again["module"] == "xtb/6.10.0-x" and len(fake.calls(env)) == n
    assert (Path(env["XDG_CACHE_HOME"]) / "mace" / "modules.json").is_file()


def test_deleted_binary_forces_a_new_capture(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=["xtb/6.7.1-a"], programs={"xtb/6.7.1-a": ("xtb", fake.XTB_671)})
    fake.apply(monkeypatch, env)
    first = envmodules.discover("xtb", "xtb", (6, 7, 0))
    os.remove(first["binary"])
    envmodules._MEM.clear()
    n = len(fake.calls(env))
    second = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert not second["cached"] and len(fake.calls(env)) > n


def test_minimum_above_everything_gives_none_and_a_computed_hint(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=["xtb/6.6.1-gfbf-2023a"], programs={"xtb/6.6.1-gfbf-2023a": ("xtb", fake.XTB_661)})
    fake.apply(monkeypatch, env)
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] is None and info["newest_any"] == "xtb/6.6.1-gfbf-2023a"
    h = envmodules.hint("xtb", info, (6, 7, 0), "GFN-FF")
    assert h == "the newest xtb module here is xtb/6.6.1-gfbf-2023a; GFN-FF needs 6.7.0 or newer"
    # failures are never cached
    assert not (Path(env["XDG_CACHE_HOME"]) / "mace" / "modules.json").exists()


def test_override_none_and_pin(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=XTB_MODS, programs={"xtb/6.7.0-foss-2023b": ("xtb", fake.XTB_671),
                                                         "xtb/6.10.0-x": ("xtb", fake.XTB_671)})
    fake.apply(monkeypatch, env)
    monkeypatch.setenv("MACE_XTB_MODULE", "none")
    assert envmodules.discover("xtb", "xtb", (6, 7, 0))["module"] is None
    assert fake.calls(env) == []
    monkeypatch.setenv("MACE_XTB_MODULE", "xtb/6.7.0-foss-2023b")
    assert envmodules.discover("xtb", "xtb", (6, 7, 0))["module"] == "xtb/6.7.0-foss-2023b"


def test_toolchain_conflict_loads_after_a_purge_in_the_child_only(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=["xtb/6.7.1-gfbf-2024a"],
                    programs={"xtb/6.7.1-gfbf-2024a": ("xtb", fake.XTB_671)}, conflicts=["xtb/6.7.1-gfbf-2024a"])
    fake.apply(monkeypatch, env)
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] == "xtb/6.7.1-gfbf-2024a"
    assert "purge" in fake.calls(env)
    assert "FAKE_PURGED" not in os.environ          # MACE's own environment untouched


def test_timeout_gives_none_and_is_not_cached(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=["xtb/6.7.1-a"], programs={"xtb/6.7.1-a": ("xtb", fake.XTB_671)})
    fake.apply(monkeypatch, env)
    # a login script that hangs
    Path(env["BASH_ENV"]).write_text(fake.MODULE_FN + "\nsleep 30\n")
    monkeypatch.setattr(envmodules, "AVAIL_TIMEOUT", 0.5)
    t0 = time.perf_counter()
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] is None and time.perf_counter() - t0 < 5
    assert not (Path(env["XDG_CACHE_HOME"]) / "mace" / "modules.json").exists()


def test_resolve_prefers_the_validated_module_then_the_newest(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=["CRYSTAL/17-x", "CRYSTAL/23-intel-2023a", "CRYSTAL/24-y"],
                    programs={"CRYSTAL/23-intel-2023a": ("crystal", "true\n"), "CRYSTAL/24-y": ("crystal", "true\n"),
                              "CRYSTAL/17-x": ("crystal", "true\n")})
    fake.apply(monkeypatch, env)
    assert envmodules.resolve("CRYSTAL", preferred="CRYSTAL/23-intel-2023a", minimum=(23,)) == "CRYSTAL/23-intel-2023a"
    assert envmodules.resolve("CRYSTAL", preferred="CRYSTAL/does-not-exist", minimum=(23,)) == "CRYSTAL/24-y"
    assert envmodules.resolve("CRYSTAL", minimum=(30,)) is None


def test_spider_fallback_hint(monkeypatch, tmp_path):
    env = fake.make(tmp_path, avail=[], spider=["xtb/6.7.1-gfbf-2024a"])
    fake.apply(monkeypatch, env)
    info = envmodules.discover("xtb", "xtb", (6, 7, 0))
    assert info["module"] is None and info["spider"] == ["xtb/6.7.1-gfbf-2024a"]
    assert "module spider xtb/6.7.1-gfbf-2024a" in envmodules.hint("xtb", info, (6, 7, 0))


def test_no_module_version_is_hard_coded_in_mace_code():
    """AC-B14: MACE finds xtb modules; it never names one with a version in code."""
    import re
    repo = Path(__file__).resolve().parents[1]
    hits = []
    for p in (repo / "mace").rglob("*.py"):
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if re.search(r"\bxtb/\d", line):
                hits.append(f"{p.relative_to(repo)}:{i}")
    assert hits == []
    src = (repo / "mace" / "utils" / "envmodules.py").read_text()
    assert not re.search(r"\b(?:xtb|CRYSTAL|Python)/\d", src)
