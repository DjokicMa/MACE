"""Batch Setup with spin-orbit coupling (1.1.6): the SOC keys of the CLI's
settings file go to every file as ``template_patches`` (Setup's own patches),
so each options/*.json carries them and ``bash commands.sh`` re-creates the
same SOC decks; a file the CLI refuses shows the CLI's reason."""
import shutil
import subprocess
import sys

import pytest

import submit_util as SU
from test_setup_soc_api import HFO_CIF, INAS_CIF, PBTE_CIF, SOC_READY

pytest.importorskip("ase")
sys.path.insert(0, str(SU.H.REPO_ROOT))
import soc_ecp  # noqa: E402

CORPUS = SU.H.REPO_ROOT / "test"
TIPBO3_SP = "SP/TiPbO3_mp-20459_sg99_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("batch_soc"))
    d = s.root / "soc"
    d.mkdir()
    for name, text in (("PbTe", PBTE_CIF), ("InAs", INAS_CIF), ("HfO", HFO_CIF)):
        (d / f"{name}.cif").write_text(text)
    yield s
    s.stop()


def _plan(s, folder, files, calc, template=None, patches=None, **kw):
    st, r = s.task_result(s.req("api/batch/plan", {"folder": folder, "files": files, "calc": calc,
                                                    "template_answers": template or {}, "template_patches": patches,
                                                    "wait": 1, **kw}), timeout=900)
    assert st == 200, r
    return r


def _write(s, bid, name):
    return s.task_result(s.req("api/batch/write", {"batch_id": bid, "wait": 1, "name": name, "overwrite": True}),
                         timeout=900)


def _rerun(s, folder, batch, tmp_path):
    """`bash commands.sh` in a twin of the folder without the decks: the decks it writes (test_batch_api's way)."""
    import os
    twin = tmp_path / "twin"
    shutil.copytree(s.root / folder, twin, ignore=shutil.ignore_patterns("decks"))
    (twin / batch / "decks").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH")}
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "mace").write_text(f"#!/bin/bash\nexec {sys.executable} {SU.MACE_CLI} --no-banner \"$@\"\n")
    (bindir / "mace").chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    r = subprocess.run(["bash", str(twin / batch / "commands.sh")], cwd=twin, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return {p.name: p.read_text() for p in (twin / batch / "decks").glob("*.d12")}


def test_cif_batch_with_soc_is_the_clis_and_a_refusal_is_its_words(srv, tmp_path):
    files = ["soc/PbTe.cif", "soc/InAs.cif", "soc/HfO.cif"]
    r = _plan(srv, "soc", files, "SP", SOC_READY, {"soc": True, "soc_ecp": "stutsc"})
    by = {f["rel"]: f for f in r["files"]}
    assert by["soc/PbTe.cif"]["status"] == "ok" and by["soc/InAs.cif"]["status"] == "ok"
    # STUTSC for Hf: refused, with the CLI's own reason
    assert by["soc/HfO.cif"]["status"] == "refused"
    assert soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP[72][:60] in by["soc/HfO.cif"]["message"]
    # the CLI's warning for the all-electron As, on InAs's row
    assert any("As (atomic number 33) is all-electron" in w for w in by["soc/InAs.cif"]["soc_warnings"])
    st, w = _write(srv, r["batch_id"], "batch_soc")
    assert st == 200, w
    B = srv.root / "soc" / "batch_soc"
    import json
    for opt in (B / "options").glob("g*_options.json"):
        o = json.loads(opt.read_text())
        assert o["soc"] is True and o["soc_ecp"] == "stutsc"
    rec = json.loads((B / "batch.json").read_text())
    assert rec["template_patches"] == {"soc": True, "soc_ecp": "stutsc"}
    decks = {p.name: p.read_text() for p in (B / "decks").glob("*.d12")}
    assert len(decks) == 2
    for text in decks.values():
        lines = text.splitlines()
        assert "TWOCOMPON" in lines and "STUTSC" in lines
    # bash commands.sh re-creates the same decks (the CLI on the written options files)
    assert _rerun(srv, "soc", "batch_soc", tmp_path) == decks


def test_soc_deck_for_opt_is_refused_in_the_clis_words(srv):
    r = _plan(srv, "soc", ["soc/PbTe.cif"], "OPT", SOC_READY, {"soc": True})
    f = r["files"][0]
    assert f["status"] == "refused" and soc_ecp.unsupported_message("OPTGEOM") in f["message"]


def test_template_patches_are_the_soc_keys_checked_by_the_cli(srv):
    st, _, r = srv.req("api/batch/plan", {"folder": "soc", "files": ["soc/PbTe.cif"], "calc": "SP",
                                          "template_patches": {"band_path": []}, "wait": 1})
    assert st == 400 and "band_path" in r["error"]
    st, _, r = srv.req("api/batch/plan", {"folder": "soc", "files": ["soc/PbTe.cif"], "calc": "SP",
                                          "template_patches": {"soc_ecp": "stuttgart"}, "wait": 1})
    assert st == 400 and "soc_ecp must be" in r["error"]


@pytest.mark.skipif(not (CORPUS / (TIPBO3_SP + ".out")).is_file(), reason="needs the test/ corpus")
def test_followup_batch_with_soc_restart_is_the_clis(srv, tmp_path):
    """An .out parent (spin-polarized, no symmetry) with GUESSPNOSO: the deck and its .f20 copy, re-created by commands.sh."""
    from pathlib import Path
    d = srv.root / "runs"
    d.mkdir(exist_ok=True)
    name = Path(TIPBO3_SP).name
    shutil.copy2(CORPUS / f"{TIPBO3_SP}.out", d / f"{name}.out")
    (d / f"{name}.d12").write_text((CORPUS / f"{TIPBO3_SP}.d12").read_text().replace("\nEND\n", "\nSYMMREMO\nEND\n", 1))
    (d / f"{name}.f9").write_bytes(b"\1" * 300)
    r = _plan(srv, "runs", [f"runs/{name}.out"], "SP", None, {"soc": True, "soc_guess": "guesspnoso"})
    assert r["files"][0]["status"] == "ok", r["files"][0]
    st, w = _write(srv, r["batch_id"], "batch_soc_sp")
    assert st == 200, w
    B = d / "batch_soc_sp"
    decks = {p.name: p.read_text() for p in (B / "decks").glob("*.d12")}
    assert len(decks) == 1 and "GUESSPNOSO" in next(iter(decks.values())).splitlines()
    f20 = list((B / "decks").glob("*.f20"))
    assert len(f20) == 1 and f20[0].read_bytes() == b"\1" * 300
    assert _rerun(srv, "runs", "batch_soc_sp", tmp_path) == decks
    # a different .f20 in the decks folder is never replaced (the CLI's words)
    f20[0].write_bytes(b"other")
    st, w2 = _write(srv, r["batch_id"], "batch_soc_sp")
    assert st != 200 and f"{f20[0].name} already exists and is not {name}.f9; not overwriting it" in str(w2)
    assert f20[0].read_bytes() == b"other"


def test_stutsc_is_named_for_the_files_whose_element_the_cli_refuses(srv):
    """batch/stutsc: the selected files' elements whose STUTSC the CLI refuses (its table and words),
    which the Batch SOC section grays out as Setup does."""
    st, r = srv.task_result(srv.req("api/batch/stutsc", {"files": ["soc/PbTe.cif", "soc/InAs.cif", "soc/HfO.cif"],
                                                         "wait": 1}), timeout=120)
    assert st == 200, r
    assert r["files"] == 3 and r["all"] is False
    assert [(x["symbol"], x["files"]) for x in r["refused"]] == [("Hf", ["soc/HfO.cif"])]
    assert r["refused"][0]["why"] == soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP[72]
    st, r = srv.task_result(srv.req("api/batch/stutsc", {"files": ["soc/HfO.cif"], "wait": 1}), timeout=120)
    assert r["all"] is True
    st, _, r = srv.req("api/batch/stutsc", {"files": ["../x.cif"], "wait": 1})
    assert st in (400, 403, 404)


def test_a_files_override_patches_take_it_back_to_inpsoc(srv):
    """The Override dialog's SOC for one file (override_patches): HfO, refused with the batch's STUTSC,
    is written with INPSOC; the other files keep STUTSC."""
    files = ["soc/PbTe.cif", "soc/HfO.cif"]
    r = _plan(srv, "soc", files, "SP", SOC_READY, {"soc": True, "soc_ecp": "stutsc"},
              override_patches={"soc/HfO.cif": {"soc_ecp": None}})
    by = {f["rel"]: f for f in r["files"]}
    assert by["soc/HfO.cif"]["status"] == "ok" and by["soc/HfO.cif"]["override"], by["soc/HfO.cif"]
    assert by["soc/PbTe.cif"]["status"] == "ok"
