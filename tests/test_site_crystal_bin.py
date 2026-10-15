"""[crystal]: CRYSTAL from a directory instead of a module, and other
executable names (PLAN_G G4, D6, D7, A11, A13)."""
import subprocess
import sys

import pytest

from test_site_generic_cluster import _cluster
from test_site_jobscript import SCF, _deck, _env, _mace, _toml

from mace.submission import site


def _submit(tmp, env, deck="m.d12"):
    d = tmp / "decks"
    _deck(d, deck)
    (d / "m.sh").unlink(missing_ok=True)
    return _mace(d, env, "submit", "--nosubmit", deck), d / "m.sh"


def test_bin_dir_without_a_module(tmp_path):
    _toml(tmp_path, '[crystal]\nbin_dir = "/opt/crystal23/bin"\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    text = sh.read_text()
    assert "module load CRYSTAL/" not in text
    assert "# MACE: CRYSTAL from /opt/crystal23/bin ([crystal] bin_dir)" in text
    assert 'export EBROOTCRYSTAL="/opt/crystal23"' in text
    assert 'mpirun -n "$SLURM_NTASKS" "/opt/crystal23/bin/Pcrystal" 2>&1 >& "$DIR/${JOB}.out"' in text
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0


def test_bin_dir_with_the_users_module_keeps_the_module(tmp_path):
    _toml(tmp_path, '[crystal]\nbin_dir = "/opt/c23"\n[modules]\ncrystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    text = sh.read_text()
    assert "module load foss/2023a\nmodule load CRYSTAL/23-foss-2023a" in text
    assert '/opt/c23/Pcrystal" 2>&1' in text and "EBROOTCRYSTAL=" not in text.split("module load CRYSTAL")[0][-200:]


def test_mpp_executable_name(tmp_path):
    _toml(tmp_path, '[crystal]\nscf = "MPPcrystal"\n')
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert '"$EBROOTCRYSTAL/bin/MPPcrystal" 2>&1' in sh.read_text()


def test_module_without_the_configured_binary_is_rejected(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[crystal]\nscf = "MPPcrystal"\n[modules]\ncrystal = "CRYSTAL/23-foss-2023b"\n')
    r = _mace(tmp_path, env, "site")
    assert "MPPcrystal" in r.stdout, r.stdout
    res = site.crystal_module(env, site.load(env), refresh=True)
    assert res.source == "not found here"
    assert any("MPPcrystal" in why for _, why in res.rejected)


def test_mace_run_dry_run_with_bin_dir(tmp_path):
    bindir = tmp_path / "c23" / "bin"
    bindir.mkdir(parents=True)
    for b in ("Pcrystal", "Pproperties"):
        (bindir / b).write_text("#!/bin/bash\necho stub\n")
        (bindir / b).chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "mpirun").write_text("#!/bin/bash\nexit 0\n")
    (stubs / "mpirun").chmod(0o755)
    _toml(tmp_path, f'[crystal]\nbin_dir = "{bindir}"\n')
    env = _env(tmp_path, PATH=f"{stubs}:/usr/bin:/bin", SCRATCH=str(tmp_path / "scr"))
    env.pop("EBROOTCRYSTAL", None)
    (tmp_path / "scr").mkdir()
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    r = _mace(d, env, "run", "--dry-run", "m.d12")
    assert r.returncode == 0, r.stdout + r.stderr
    (bindir / "Pcrystal").unlink()
    r = _mace(d, env, "run", "--dry-run", "m.d12")
    assert r.returncode == 2 and f"Pcrystal not found in {bindir}" in r.stdout + r.stderr


def test_missing_binary_pattern(tmp_path):
    s = site.Site()
    s.crystal_bin_dir = "/opt/c23/bin"
    s.crystal_exe["scf"] = "MPPcrystal"
    rx = site.missing_binary_re(s)
    assert rx.search("bash: /opt/c23/bin/MPPcrystal: No such file or directory")
    assert rx.search("x/bin/Pproperties: No such file or directory")
    assert not rx.search("bash: /opt/c23/bin/other: No such file or directory")
    assert site.missing_binary_re(site.Site()) is site.MISSING_BINARY_RE


@pytest.mark.parametrize("bad", ["/opt/x;id", "/opt/../x", "opt/x", "/opt/x y", "/opt/$HOME", "/opt/x'"])
def test_invalid_bin_dir(tmp_path, bad):
    import json
    _toml(tmp_path, "[crystal]\nbin_dir = " + json.dumps(bad) + "\n")
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and "[crystal] bin_dir" in p.stdout and not sh.exists()


@pytest.mark.parametrize("bad", ["Pcrystal;id", "../Pcrystal", "P crystal", ".."])
def test_invalid_executable_names(tmp_path, bad):
    import json
    _toml(tmp_path, "[crystal]\nscf = " + json.dumps(bad) + "\n")
    p, sh = _submit(tmp_path, _env(tmp_path))
    assert p.returncode == 2 and not sh.exists()


def test_manager_regeneration_fails_closed_without_tomllib(tmp_path):
    """D7: the queue manager's regeneration (a completion callback) with a
    bin_dir site.toml and no tomllib prints Error: and writes nothing - never
    the built-in `module load CRYSTAL/23-intel-2023a` script."""
    _toml(tmp_path, '[crystal]\nbin_dir = "/opt/c23/bin"\n')
    env = _env(tmp_path)
    code = (
        "import sys; sys.modules['tomllib'] = None\n"
        f"sys.path.insert(0, {str(SCF.parents[2])!r})\n"
        "from mace.queue.manager import EnhancedCrystalQueueManager as M\n"
        "m = M.__new__(M)\n"
        f"gen, cleanup = m._site_generator({str(SCF)!r}, {str(tmp_path)!r})\n"
        "print('GEN', gen)\n")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert "Error:" in r.stdout and "Python 3.11 or newer" in r.stdout, r.stdout + r.stderr
    assert "GEN None" in r.stdout
    assert not list(tmp_path.glob(".submitcrystal23_site.*.sh"))


def test_preflight_finds_serial_crystal_in_bin_dir(tmp_path, monkeypatch):
    bindir = tmp_path / "c23" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "crystal").write_text("#!/bin/bash\n")
    (bindir / "crystal").chmod(0o755)
    _toml(tmp_path, f'[crystal]\nbin_dir = "{bindir}"\n')
    for k, v in _env(tmp_path).items():
        if k.startswith("XDG_") or k in ("HOME", "MACE_SITE_GROUP_CONFIG"):
            monkeypatch.setenv(k, v)
    for k in ("MACE_CRYSTAL_BIN", "EBROOTCRYSTAL", "MACE_SITE_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    from mace.submission import preflight
    from mace.gui import preflight as gpre
    assert preflight.find_crystal_binary() == str(bindir / "crystal")
    assert gpre.fast_crystal() == str(bindir / "crystal")
