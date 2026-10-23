"""`mace calc` through mace_cli (the real invocation path): the listing, a
UFF4MOF optimization of a CIF (files, marker, report, exit codes), EQeq
charges, MD and frequencies, usage errors, a missing backend's hint, and
`mace mlip list --all` with the calc models."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("spglib")
import calc_fixtures as CF  # noqa: E402
from calc_fixtures import preopt_sandbox  # noqa: E402,F401

CLI = CF.REPO / "mace_cli"


def mace(*args, cwd=None, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_XTB", "MACE_DFTB", "MACE_MLIP_PYTHON")}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(CLI), "--no-banner", *args], cwd=cwd, env=e, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


@pytest.fixture
def cif(tmp_path):
    p = tmp_path / "NaCl.cif"
    shutil.copy(CF.F.EXAMPLES / "NaCl.cif", p)
    return p


def test_list_json():
    rc, out, err = mace("calc", "list", "--json", env={"PATH": "/usr/bin:/bin"})
    assert rc == 0, err
    d = json.loads(out)
    assert [b["name"] for b in d["backends"]] == ["uff", "eqeq", "xtb", "dftb", "mlip"]
    assert d["backends"][0]["available"] and "device_default" in d


def test_opt_writes_files(cif):
    rc, out, err = mace("calc", "opt", str(cif), "-b", "uff4mof", "--relax-cell", "--steps", "200")
    assert rc in (0, 3), err
    stem = "NaCl_uff-uff4mof_opt"
    d = cif.parent
    assert (d / f"{stem}.cif").is_file() and (d / f"{stem}.traj").is_file()
    assert "# mace-preopt: method=calc:uff:uff4mof" in (d / f"{stem}.cif").read_text()
    rep = json.loads((d / f"{stem}.json").read_text())
    assert rep["task"] == "opt" and rep["provenance"]["tool"] == "mace calc" and rep["structure"]["numbers"]
    assert "run a CRYSTAL OPT" in out and "Cite: A. K. Rapp" in out
    rc2, _, err2 = mace("calc", "opt", str(cif), "-b", "uff4mof")
    assert rc2 == 1 and "already exist" in err2


def test_charges_md_freq(cif, tmp_path):
    out_dir = tmp_path / "out"
    rc, out, err = mace("calc", "charges", str(cif), "-o", str(out_dir), "--json")
    assert rc == 0, err
    rep = json.loads(out)
    assert rep["backend"] == "eqeq" and abs(sum(rep["charges"])) < 1e-6
    assert "_atom_site_charge" in (out_dir / "NaCl_eqeq-eqeq_charges.cif").read_text()
    rc, out, err = mace("calc", "md", str(cif), "-b", "uff", "--md-steps", "30", "--interval", "10", "--seed", "1",
                        "-o", str(out_dir))
    assert rc == 0, err
    log = (out_dir / "NaCl_uff-uff4mof_md_log.csv").read_text().splitlines()
    assert log[0].startswith("step,time_fs,epot_eV") and len(log) == 4
    rc, out, err = mace("calc", "freq", str(cif), "-b", "uff", "-o", str(out_dir))
    assert rc == 0, err
    assert "Frequencies (cm^-1)" in out and (out_dir / "NaCl_uff-uff4mof_freq_modes.extxyz").is_file()


def test_usage_and_missing(cif):
    rc, _, err = mace("calc", "opt", str(cif))
    assert rc == 2 and "--backend" in err
    rc, _, err = mace("calc", "sp", str(cif), "-b", "nosuch")
    assert rc == 2 and "no calculator backend" in err
    rc, _, err = mace("calc", "md", str(cif), "-b", "eqeq")
    assert rc == 2 and "does not do" in err
    rc, _, err = mace("calc", "sp", str(cif), "-b", "gfn2", env={"PATH": "/usr/bin:/bin", "MACE_XTB": "/nonexistent/xtb"})
    assert rc == 1 and "next:" in err and "xtb" in err
    rc, _, err = mace("calc", "sp", str(cif), "-b", "mlip", "--device", "gpu")
    assert rc == 2 and "cpu or cuda:N" in err


def test_mlip_list_all(tmp_path):
    rc, out, err = mace("mlip", "list", "--all", env={"XDG_CONFIG_HOME": str(tmp_path / "c")})
    assert rc == 0, err
    assert "[mace calc only]" in out and "restricted" not in out
    assert "mlip:acesuit-mace-off23-medium [mace calc only]" not in out          # preopt and calc
    assert "mlip:acesuit-mace-off23-medium [ACADEMIC, NON-COMMERCIAL USE ONLY: opt-in]" in out
    assert "mlip:meta-uma-s-1p1 [mace calc only] [gated: your own Hugging Face access]" in out
    rc, out, _ = mace("mlip", "list", env={"XDG_CONFIG_HOME": str(tmp_path / "c")})
    assert "orb-v3" not in out and "off23" not in out and "uma" not in out
    rc, out, _ = mace("mlip", "list", "--academic", env={"XDG_CONFIG_HOME": str(tmp_path / "c")})
    assert "off23" in out and "orb-v3" not in out


def test_mace_off_is_refused_until_academic_use_is_acknowledged(cif, tmp_path):
    """MACE-OFF23 (Academic Software License) is an explicit opt-in: before the
    user acknowledges academic, non-commercial use, naming it is refused with
    how to opt in (no traceback), and nothing is downloaded or run."""
    env = {"XDG_CONFIG_HOME": str(tmp_path / "c"), "XDG_DATA_HOME": str(tmp_path / "d")}
    for args in (("calc", "sp", str(cif), "-b", "mlip:acesuit-mace-off23-medium"),
                 ("calc", "sp", str(cif), "-b", "mlip", "-m", "acesuit-mace-off23-medium"),
                 ("mlip", "fetch", "acesuit-mace-off23-medium", "--yes")):
        rc, out, err = mace(*args, env=env)
        assert rc == 2, (args, out, err)
        assert "academic, non-commercial use only" in (out + err).lower() or "ACADEMIC, NON-COMMERCIAL USE ONLY" in out
        assert "--accept-academic-license" in out + err, (args, out, err)
        assert "Traceback" not in err
    rc, out, err = mace("preopt", str(cif), "--method", "mlip:acesuit-mace-off23-medium", env=env)
    assert rc == 3 and "academic, non-commercial use only" in out + err and "Traceback" not in err
    assert not (tmp_path / "d" / "mace-tool" / "mlip" / "acesuit-mace-off23-medium").exists()
    rc, out, err = mace("mlip", "fetch", "acesuit-mace-mpa-0-medium", "--accept-license", "ASL", env=env)
    assert rc == 2 and "--accept-license" in err


# --- a model file the user already has (mace calc --model-file) ---------------------------------

def _user_file(tmp_path):
    p = tmp_path / "my-mace-off23.model"
    p.write_bytes(b"a model file the user supplies")
    return p


def test_user_model_file_is_refused_without_the_license_acknowledgment(cif, tmp_path):
    f = _user_file(tmp_path)
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", str(f), "--model-backend", "mace-torch")
    assert rc == 2 and "--model-license-ok" in err and "user-supplied model" not in err, err
    for args in (("--model-backend", "mace-torch"), ("--model-license-ok",)):         # only with --model-file
        rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", *args)
        assert rc == 2 and "go with --model-file" in err, err
    rc, out, err = mace("calc", "sp", str(cif), "-b", "uff", "--model-file", str(f), "--model-license-ok")
    assert rc == 2 and "-b mlip" in err, err
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", str(f), "--model-license-ok")
    assert rc == 2 and "--model-backend" in err, err


def test_user_model_file_missing_path(cif, tmp_path):
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", str(tmp_path / "nope.model"),
                        "--model-backend", "mace-torch", "--model-license-ok")
    assert rc == 2 and "no such file" in err and "Traceback" not in err, err


def test_user_model_file_runs_where_it_is_and_nothing_is_fetched(cif, tmp_path):
    """The worker loads the user's own file in place (mace_mp backend); MACE prints whose license it is, the
    report records the file and its sha256, and nothing is downloaded or copied into MACE's model store."""
    import hashlib
    from mace.preopt import detect
    f = _user_file(tmp_path)
    before = (f.read_bytes(), f.stat().st_mtime_ns)
    py = CF.F.make_fake_python(tmp_path / "venv")
    log = tmp_path / "env.json"
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", str(f), "--model-backend", "mace-torch",
                        "--model-license-ok", "--device", "cpu", "-o", str(tmp_path / "out"),
                        env={"MACE_MLIP_PYTHON": str(py), "FAKE_ENV_LOG": str(log)})
    assert rc == 0, err
    assert "user-supplied model" in err and "license terms are your responsibility" in err
    w = json.loads(log.read_text())
    argv = w["argv"]
    assert argv[argv.index("--weights") + 1] == str(f.resolve()) and argv[argv.index("--backend") + 1] == "mace_mp"
    assert w["env"]["MACE_MLIP_NO_DOWNLOAD"] == "1"
    rep = json.loads((tmp_path / "out" / "NaCl_mlip-user-my-mace-off23.model_sp.json").read_text())
    p = rep["provenance"]
    assert p["user_supplied"] and p["model_file"] == str(f.resolve()) and "user's responsibility" in p["weights_license"]
    assert p["weights_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert (f.read_bytes(), f.stat().st_mtime_ns) == before
    root = detect.weights_root()
    assert not root.exists() or not [x for x in root.rglob("*") if x.is_file()]
    rc, out, _ = mace("calc", "list", "--json", env={"MACE_MLIP_PYTHON": str(py)})
    assert rc == 0 and "user:" not in out and "off23" not in out


def test_user_model_file_real_load(cif, tmp_path):
    """A real mace-torch load of a small model file already on this machine (MACE_TEST_USER_MODEL); skipped
    when mace-torch is not installed in this interpreter or there is no such file."""
    import importlib.metadata as md
    try:
        md.version("mace-torch")
    except md.PackageNotFoundError:
        pytest.skip("mace-torch is not installed in this interpreter")
    model = os.environ.get("MACE_TEST_USER_MODEL")
    if not model or not os.path.isfile(model):
        pytest.skip("no local model file (set MACE_TEST_USER_MODEL)")
    rc, out, err = mace("calc", "sp", str(cif), "-b", "mlip", "--model-file", model, "--model-backend",
                        "mace-torch", "--model-license-ok", "--device", "cpu", "--json", "-o", str(tmp_path / "out"),
                        env={"MACE_MLIP_PYTHON": sys.executable}, timeout=600)
    assert rc == 0, err
    assert json.loads(out)["energy_eV"] is not None


def test_uff_eqeq_charges_take_the_total_charge(tmp_path):
    """--uff-charges eqeq with --charge N: the EQeq point charges sum to N (they summed to 0)."""
    from ase.build import molecule
    from ase.io import write
    from mace.calculators.uff import UFFCalculator
    m = molecule("CH3OH")
    m.center(vacuum=6.0)
    m.pbc = False
    calc = UFFCalculator(mode="uff", charges="eqeq", total_charge=1)
    calc.setup(m)
    assert float(np.sum(calc.q)) == pytest.approx(1.0, abs=1e-6)
    write(str(tmp_path / "meoh.xyz"), m, format="xyz")
    e = {}
    for q in ("0", "1"):
        rc, out, err = mace("calc", "sp", str(tmp_path / "meoh.xyz"), "-b", "uff", "-m", "uff", "--uff-charges",
                            "eqeq", "--charge", q, "--json", "--force")
        assert rc == 0, err
        d = json.loads(out)
        e[q] = d["report"]["energy_parts_eV"]
        assert not [w for w in d["warnings"] if "charge" in w], d["warnings"]
    assert e["0"] != e["1"]
    rc, out, err = mace("calc", "sp", str(tmp_path / "meoh.xyz"), "-b", "uff", "-m", "uff", "--charge", "1",
                        "--json", "--force")
    assert rc == 0 and any("recorded only" in w for w in json.loads(out)["warnings"]), out
