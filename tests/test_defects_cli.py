"""`mace defects` through the real entry point (mace_cli run as a subprocess)."""
import json
import os
import subprocess
import sys

from _defects_fixtures import DATA, deck, mgo_host, same_but_title
from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _mace(*args, cwd=None):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_help_lists_the_command():
    rc, text = _mace("--help")
    assert rc == 0 and "defects" in text and "Point-defect deck sets" in text
    for flag in ("--help", "-h"):
        rc, text = _mace("defects", flag)
        assert rc == 0 and "mace defects {build,potc,analyze}" in text, flag
        assert "Command to run" not in text, flag          # not the general help
    rc, text = _mace("defects", "build", "--help")
    assert rc == 0 and "--calibrate-on" in text and "--resume" in text


def test_build_resume_analyze(tmp_path):
    (tmp_path / "mgo.d12").write_text(mgo_host())
    rc, text = _mace("defects", "build", "mgo.d12", "--vacancy", "O", "--supercon", "2x2x2",
                     "--charges", "0,1,2", "--calibrate", "-o", "set", cwd=tmp_path)
    assert rc == 0, text
    d = tmp_path / "set"
    # no label yet: only the pristine deck, and how to go on
    assert sorted(p.name for p in d.glob("*.d12")) == ["mgo_pris.d12"]
    manifest = json.loads((d / "defects_manifest.json").read_text())
    assert manifest["status"] == "awaiting_pristine"
    assert f"mace submit {d}" in text and "mace defects build --resume set" in text
    assert same_but_title((d / "mgo_pris.d12").read_text(), deck("mgo_pris"))
    # the running pristine job has printed its geometry: that is enough to resume
    (d / "mgo_pris.out").write_text((DATA / "mgo_pris_geometry.out").read_text())
    rc, text = _mace("defects", "build", "--resume", "set", cwd=tmp_path)
    assert rc == 0, text
    assert "N=640 even" in text and "q=+1: N=631 odd" in text
    loop = text.split("for j in ")[1].split(";")[0].split()
    assert loop == ["mgo_q0", "mgo_qp1", "mgo_qp2", "mgo_calm05", "mgo_calm10", "mgo_calm15"]
    assert f"mace submit {d}/$j.d12" in text                 # the pristine job is not resubmitted
    for tag, ref in (("q0", "mgo_v0"), ("qp1", "mgo_v1"), ("qp2", "mgo_v2"), ("calm10", "mgo_j10")):
        assert same_but_title((d / f"mgo_{tag}.d12").read_text(), deck(ref)), tag
    manifest = json.loads((d / "defects_manifest.json").read_text())
    assert manifest["status"] == "built" and manifest["site"] == {"label": 1, "source": "element"}
    assert manifest["calibration"]["restricted"] is True
    assert {j["name"] for j in manifest["jobs"]} >= {"mgo_pris", "mgo_q0", "mgo_qp1", "mgo_calm05"}
    assert not list(d.glob("*.d3"))
    # outputs -> analysis (synthetic outputs with the measured numbers)
    from test_defects_analysis import _mgo_out
    for tag in ("pris", "q0", "qp1", "qp2", "calm05", "calm10", "calm15"):
        (d / f"mgo_{tag}.out").write_text(_mgo_out(tag) + ((DATA / "mgo_pris_geometry.out").read_text()
                                                             if tag == "pris" else ""))
    rc, text = _mace("defects", "analyze", "set", "--json", "r.json", "--csv", "r.csv", cwd=tmp_path)
    assert rc == 0, text
    assert "eps(+1/0) = 3.5" in text and "uncorrected for image charges" in text
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["image_charge_correction"] == "none"
    assert (tmp_path / "r.csv").read_text().startswith("q,job,status")
    assert "offset (spheropole)" in text and "cross-check with the DOPING calibration offset" in text
    # the image-charge term from a CPHF output
    from test_defects_image_charge import CPHF
    (tmp_path / "cphf.out").write_text(CPHF)
    rc, text = _mace("defects", "analyze", "set", "--eps-from", "cphf.out", "--csv", "r.csv", cwd=tmp_path)
    assert rc == 0, text
    assert "eps(+1/0) = 2.88" in text and "eps(+2/+1) = -0.09" in text
    assert "image charge: Lany-Zunger-scaled Makov-Payne, eps = 2.865 (eps_inf from cphf.out)" in text
    assert "uncorrected for image charges" not in text
    assert "image_charge_ev" in (tmp_path / "r.csv").read_text().splitlines()[0]
    rc, text = _mace("defects", "analyze", "set", "--eps", "2.865", "--eps-from", "cphf.out", cwd=tmp_path)
    assert rc == 2 and "not both" in text
    rc, text = _mace("defects", "analyze", "set", "--eps-static", "--eps", "9.3", cwd=tmp_path)
    assert rc == 2 and "--eps-from" in text


def test_refusals_exit_2_with_the_reason(tmp_path):
    (tmp_path / "opt.d12").write_text(mgo_host().replace("BASISSET", "OPTGEOM\nFULLOPTG\nEND\nBASISSET"))
    rc, text = _mace("defects", "build", "opt.d12", "--vacancy", "O", "--supercon", "2", cwd=tmp_path)
    assert rc == 2 and "opt2d12 --calc-type SP" in text
    assert not (tmp_path / "opt_defects").exists()


def test_potc_staging(tmp_path):
    (tmp_path / "mgo.d12").write_text(mgo_host())
    rc, text = _mace("defects", "build", "mgo.d12", "--vacancy", "O", "--site-label", "1",
                     "--supercon", "2x2x2", "--charges", "0,2", "-o", "set", cwd=tmp_path)
    assert rc == 0, text
    d = tmp_path / "set"
    (d / "mgo_pris.f9").write_bytes(b"F9DATA")
    (d / "mgo_q0.f9").write_bytes(b"F9q0")
    (d / "mgo_qp2.f9").write_bytes(b"")
    rc, text = _mace("defects", "potc", "set", cwd=tmp_path)
    assert rc == 2 and "mgo_qp2" in text and not (d / "potc").exists()
    (d / "mgo_qp2.f9").write_bytes(b"F9q2")
    rc, text = _mace("defects", "potc", "set", cwd=tmp_path)
    assert rc == 0, text
    assert f"mace submit {d}/potc" in text
    assert (d / "potc" / "mgo_pris_potc.d3").read_text() == "POTC\n0 0 0\nEND\n"
    assert (d / "potc" / "mgo_qp2_potc.f9").read_bytes() == b"F9q2"
    assert (d / "mgo_qp2.f9").read_bytes() == b"F9q2"          # originals untouched
    assert not list(d.glob("*.d3"))                            # nothing for `mace submit set` to pick up
