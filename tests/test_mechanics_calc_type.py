"""Elastic, EOS and QHA outputs are not geometry optimizations.

CRYSTAL prints the "OPTGEOM OPTIMIZES ..." NEW DEFAULT line while it reads an
ELASTCON, ELAPIEZO, PHOTOELA, EOS or QHA input, and optimizes each strained
cell or volume to its own OPT END and FINAL OPTIMIZED GEOMETRY. These runs were
sorted as OPT (EOS) or FREQ (the elastic modules and QHA, which print FREQCALC
lines too), so `mace completion` put them in complete/ and the database scan
recorded them as OPT or FREQ. They are now the calculation types `mace opt2d12
--calc-type` writes them as: ELASTIC, EOS and QHA.

The outputs are the HPCC CRYSTAL23 runs of tests/data/hpcc_116/mechanics. They
are trimmed and lost their OPT END lines; the full runs' lines are put back
where noted.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace import completion_checker as cc
from mace.utils.calc_detection import (
    is_frequency_output, is_optimization_output, mechanics_output_type)

MECH = Path(__file__).resolve().parent / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
EXPECTED = {"dia_elastcon": "ELASTIC", "dia_elapiezo": "ELASTIC", "bn_elapiezo": "ELASTIC",
            "dia_photoela": "ELASTIC", "dia_eos": "EOS", "dia_qha": "QHA"}
BUCKET = {"ELASTIC": "completeelastic", "EOS": "completeeos", "QHA": "completeqha"}
# the lines of one volume's optimization in the full dia_eos.out (HPCC)
OPT_LINES = (" *                             OPTIMIZATION STARTS                             *\n"
             " CELL OPTIMIZATION - POINT    1\n"
             " * OPT END - CONVERGED * E(AU):  -7.621280823176E+01  POINTS    1 *\n"
             " FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")


def _full_run(src: Path, dest: Path) -> Path:
    """The trimmed output with an optimization's lines put back after its header."""
    lines = src.read_text(errors="ignore").splitlines(True)
    at = next(i for i, ln in enumerate(lines) if "TOTAL ENERGY" in ln or "== SCF ENDED" in ln)
    dest.write_text("".join(lines[:at]) + OPT_LINES + "".join(lines[at:]))
    return dest


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_output_has_its_own_type_and_is_not_an_optimization(name):
    text = (MECH / f"{name}.out").read_text(errors="ignore")
    assert "OPTGEOM OPTIMIZES" in text        # the line that made it an OPT
    assert mechanics_output_type(text) == EXPECTED[name]
    assert not is_optimization_output(text)
    lines = text.splitlines(True)
    assert cc.determine_completed_subtype(MECH / f"{name}.out", lines) == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_full_run_with_opt_end_is_still_its_own_type(tmp_path, name):
    out = _full_run(MECH / f"{name}.out", tmp_path / f"{name}.out")
    text = out.read_text()
    assert not is_optimization_output(text)
    assert cc.determine_completed_subtype(out, text.splitlines(True), has_opt_end=True) == EXPECTED[name]
    assert cc.categorize_output_file(out) == (BUCKET[EXPECTED[name]], name)


def test_qha_is_told_from_eos_by_its_frequency_lines():
    assert is_frequency_output((MECH / "dia_qha.out").read_text(errors="ignore"))
    assert not is_frequency_output((MECH / "dia_eos.out").read_text(errors="ignore"))


def test_cphf_and_strain_runs_are_not_mechanics_runs():
    cphf = (MECH / "dia_cphf.out").read_text(errors="ignore")
    assert mechanics_output_type(cphf) is None
    assert cc.categorize_output_file(MECH / "dia_cphf.out")[0] == "completecphf"      # its own (test_cphf_calc_type)
    strain = MECH.parent / "strain" / "dia_sp_dp_x_p1pct.out"
    assert mechanics_output_type(strain.read_text(errors="ignore")) is None


def test_database_scan_records_the_mechanics_types(tmp_path):
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    for name in EXPECTED:
        _full_run(MECH / f"{name}.out", tmp_path / f"{name}.out")
    found = {Path(c["output_file"]).stem: c["calc_type"] for c in scan_for_completed_calculations(tmp_path)}
    assert found == EXPECTED


def test_history_records_the_mechanics_types():
    from mace.cluster.history import deck_calc_type
    for name, kind in EXPECTED.items():
        assert deck_calc_type("", (MECH / f"{name}.out").read_text(errors="ignore")) == kind


def test_mace_completion_lists_them_under_their_own_type(tmp_path):
    for name in ("dia_elastcon", "dia_eos", "dia_qha"):
        _full_run(MECH / f"{name}.out", tmp_path / f"{name}.out")
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    r = subprocess.run([sys.executable, str(MACE_CLI), "completion", "--organize"], cwd=str(tmp_path),
                       env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    assert r.returncode == 0, text
    assert "Geometry optimization" not in text, text
    for desc in ("(ELASTIC)", "Equation of state (EOS)", "Quasi-harmonic approximation (QHA)"):
        assert desc in text, text
    for name, bucket in (("dia_elastcon", "completeelastic"), ("dia_eos", "completeeos"),
                         ("dia_qha", "completeqha")):
        assert (tmp_path / "sorted" / bucket / f"{name}.out").exists(), text
    assert not (tmp_path / "sorted" / "complete").exists(), text


def test_no_corpus_output_is_a_mechanics_run():
    """The banners are in no output of test/ (OPT detection there is unchanged)."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ corpus not present")
    hits = [out.name for out in TEST_DATA.rglob("*.out")
            if mechanics_output_type(out.read_text(errors="ignore"))]
    assert not hits, hits[:10]
