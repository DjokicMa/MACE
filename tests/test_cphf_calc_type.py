"""A finished CPHF run is a CPHF, as its deck is.

`mace opt2d12 --calc-type CPHF` decks are their own type (d12_calc_type), but
their outputs were read as SP: the completion buckets, the database scan, the
GUI's batch view, the cluster history and progress all called a CPHF run an
SP once it had finished. CRYSTAL prints the CPHF/CPKS module banner after the
SCF and the table of its results; a FREQCALC run with INTCPHF and a PHOTOELA
run print them too and stay FREQ and ELASTIC.

Real HPCC CRYSTAL23 outputs (tests/data/hpcc_116/mechanics): dia_cphf.out (the
finished run, trimmed) and dia_cphf_running.out (the full run's first 657
lines: the SCF, the CPKS banner and its first INFORMATION lines, as while the
CPHF block runs). The corpus FREQ runs with INTCPHF are in test/FREQ.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace import completion_checker as cc
from mace.utils.calc_detection import is_cphf_output

MECH = Path(__file__).resolve().parent / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
CPHF_OUTS = ("dia_cphf", "dia_cphf_running")


def _text(name):
    return (MECH / f"{name}.out").read_text(errors="ignore")


@pytest.mark.parametrize("name", CPHF_OUTS)
def test_cphf_output_is_cphf(name):
    from mace.cluster.history import deck_calc_type
    from mace.cluster.progress import mode_of
    assert is_cphf_output(_text(name))
    assert deck_calc_type("", _text(name)) == "CPHF"
    assert mode_of(_text(name)) == "cphf"


def test_finished_cphf_run_is_sorted_extracted_and_scanned_as_cphf(tmp_path):
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    from mace.gui.batch import BUCKET_TYPE
    from mace.utils.property_extractor import CrystalPropertyExtractor
    assert cc.categorize_output_file(MECH / "dia_cphf.out") == ("completecphf", "dia_cphf")
    assert BUCKET_TYPE["completecphf"] == "CPHF"
    props = CrystalPropertyExtractor(db_path="", enable_tracking=False).extract_all_properties(MECH / "dia_cphf.out")
    assert props["calculation_type"] == "CPHF"
    shutil.copy(MECH / "dia_cphf.out", tmp_path / "dia_cphf.out")
    found = {Path(c["output_file"]).stem: c["calc_type"] for c in scan_for_completed_calculations(tmp_path)}
    assert found == {"dia_cphf": "CPHF"}


def test_running_cphf_progress_is_the_scf_then_cphf(tmp_path):
    from mace.cluster import progress as P
    shutil.copy(MECH / "dia_cphf_running.out", tmp_path / "c.out")
    shutil.copy(MECH / "dia_cphf.d12", tmp_path / "c.d12")
    r = P.progress(tmp_path / "c.out")
    assert r["mode"] == "cphf" and r["stage"] == "cphf" and r["ended"] is None
    assert r["summary"] == "CPHF/CPKS after the SCF (9 cycles)", r["summary"]


def test_photoela_stays_elastic():
    text = _text("dia_photoela")
    assert "CP_DRIVER" in text and not is_cphf_output(text)
    assert cc.categorize_output_file(MECH / "dia_photoela.out")[0] == "completeelastic"


def test_corpus_freq_runs_with_intcphf_stay_freq():
    """The FREQCALC runs with INTCPHF print the CPKS banner and the dielectric table: FREQ, as before."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ corpus not present")
    from mace.cluster.history import deck_calc_type
    runs = [out for out in sorted((TEST_DATA / "FREQ").glob("*.out"))
            if "COUPLED-PERTURBED" in out.read_text(errors="ignore")]
    assert len(runs) >= 8, [p.name for p in runs]
    for out in runs:
        text = out.read_text(errors="ignore")
        assert not is_cphf_output(text), out.name
        assert cc.categorize_output_file(out)[0] == "completefreq", out.name
        assert deck_calc_type("", text) == "FREQ", out.name
    hits = [out.name for out in TEST_DATA.rglob("*.out") if is_cphf_output(out.read_text(errors="ignore"))]
    assert not hits, hits[:10]


def _submit(workdir, *args):
    bindir = workdir / "fakebin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        (bindir / name).write_text("#!/bin/bash\necho 'Submitted batch job 999001'\n")
        (bindir / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_WORKFLOW_ID")}
    env.update(PATH=f"{bindir}:{env['PATH']}", MACE_NO_BANNER="1")
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", *args], cwd=str(workdir),
                          env=env, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)


def test_submit_progress_refuses_a_cphf_deck_with_the_reason(tmp_path):
    """No template has a CPHF step: --progress says so before writing a plan (a CPHF deck used to enter
    as the template's SP step; at step 1 it would start nothing after it), as MACE GUI's Submit dialog
    offers no template for it."""
    shutil.copy(MECH / "dia_cphf.d12", tmp_path / "dia_cphf.d12")
    r = _submit(tmp_path, "--progress", "full_electronic", "dia_cphf.d12")
    text = r.stdout + r.stderr
    assert r.returncode == 1, text
    assert ("the template full_electronic (OPT, SP, BAND, DOSS) has no step for dia_cphf.d12 (a CPHF deck); "
            "ELASTIC, EOS, QHA and CPHF decks are not steps of a workflow template: submit them without "
            "--progress") in " ".join(text.split()), text
    assert not (tmp_path / "workflow_configs").exists() and not (tmp_path / "workflow_outputs").exists()


def test_gui_submit_offers_no_template_for_a_cphf_deck(tmp_path):
    from mace.gui.submit import deck_calc_type, templates_for
    shutil.copy(MECH / "dia_cphf.d12", tmp_path / "dia_cphf.d12")
    t = deck_calc_type(tmp_path / "dia_cphf.d12")
    assert t == "CPHF"
    assert all(not x["ok"] and x["why"] == "does not contain CPHF" for x in templates_for([t]))
