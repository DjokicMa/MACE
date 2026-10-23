"""CPHF, PHOTOELA and PIEZOCP decks only with a functional CRYSTAL23's CPKS takes.

These modules solve the CPHF/CPKS equations, which CRYSTAL23 implements for
Hartree-Fock and a listed subset of the LDA/GGA functionals and hybrids, and
for no meta-GGA (manual p. 140, "Availability of XC functionals"). MACE wrote
the decks for any functional. Frequency-dependent responses (CPHF DYNAMIC,
and PHOTOELA DYNAMIC, which runs the same DYNAMIC CPHF/KS) stop with
Hartree-Fock or a hybrid (pp. 272-273).

Decks are written through mace_cli from the repo's Si parent (PBE0), the
functional changed with a config file as a user would, and from the corpus
HSEsol-3c and B3LYP-D3 optimizations when test/ is present.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_calc_elastic as E  # noqa: E402

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"


def _mace(*args, cwd):
    env = dict(os.environ, MACE_NO_BANNER="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.fixture
def si(tmp_path):
    for f in ("parent.out", "parent.d12"):
        shutil.copy(SI / f, tmp_path / f)
    return tmp_path


def _config(si, calc_type, functional, settings):
    (si / "cfg.json").write_text(json.dumps({"calculation_type": calc_type, "method": "DFT",
                                             "functional": functional, "elastic_settings": settings}))
    return _mace("opt2d12", "--out-file", "parent.out", "--config-file", "cfg.json", "--non-interactive", cwd=si)


# ------------------------------------------------------------------ the manual's table
@pytest.mark.parametrize("functional", ["SVWN", "BLYP", "PBE", "PBE-D3", "PBESOL", "SOGGA", "WCGGA",
                                        "B3LYP", "B3LYP-D3", "PBE0", "PBESOL0", "B1WC", "WC1LYP", "B97H",
                                        "RSHXLDA", "wB97", "wB97X", "LC-BLYP", "SC-BLYP", "CAM-B3LYP",
                                        "HSE06", "HSEsol", "HISS", "LC-wPBE", "LC-wPBEsol", "LC-wBLYP",
                                        "LC-PBE", "RHF", "UHF", "HF3C", "HFSOL3C", "PBESOL03C", "HSESOL3C"])
def test_listed_functionals_are_taken(functional):
    for mode in E.CPKS_MODES:
        assert E.cpks_problem(mode, functional) is None


@pytest.mark.parametrize("functional", ["VBH", "PWGGA", "B97", "SOGGA11", "B3PW", "PBE0-13", "mPW1PW91",
                                        "mPW1K", "SOGGA11X", "SCAN", "r2SCAN", "M06L", "M06", "M062X",
                                        "SCAN0", "r2SCAN0", "MN15", "PW6B95", "PBEH3C", "HSE3C", "B973C", None])
def test_other_functionals_are_refused(functional):
    for mode in E.CPKS_MODES:
        why = E.cpks_problem(mode, functional)
        assert why and "p. 140" in why and "HSEsol" in why and mode in why
    assert E.cpks_problem("ELASTCON", functional) is None       # no CPKS: any functional


def test_records_of_listed_components_are_taken():
    assert E.cpks_problem("CPHF", "CUSTOM-XC", ["EXCHANGE", "PBE", "CORRELAT", "PBE", "HYBRID", "30"]) is None
    assert E.cpks_problem("CPHF", "CUSTOM-XC", ["B3LYP", "HYBRID", "25"]) is None
    why = E.cpks_problem("CPHF", "CUSTOM-XC", ["EXCHANGE", "PWGGA", "CORRELAT", "PWGGA"])
    assert "EXCHANGE PWGGA" in why


# ------------------------------------------------------------------ real deck generation
@pytest.mark.parametrize("calc_type, settings, functional", [
    ("CPHF", {"mode": "CPHF"}, "M06"),
    ("CPHF", {"mode": "CPHF", "third": True}, "PBEH3C"),
    ("CPHF", {"mode": "CPHF"}, "B973C"),
    ("ELASTIC", {"mode": "PHOTOELA", "print": True}, "SCAN"),
    ("ELASTIC", {"mode": "PHOTOELA"}, "HSE3C"),
    ("ELASTIC", {"mode": "PIEZOCP"}, "r2SCAN0"),
    ("ELASTIC", {"mode": "PIEZOCP"}, "B3PW"),
])
def test_unsupported_functional_writes_no_deck(si, calc_type, settings, functional):
    r = _config(si, calc_type, functional, settings)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert f"this deck's functional is {functional}" in out and "manual p. 140" in out
    assert sorted(p.name for p in si.glob("*.d12")) == ["parent.d12"]


@pytest.mark.parametrize("calc_type, settings, functional, block", [
    ("CPHF", {"mode": "CPHF", "third": True}, "PBE0", ["CPHF", "THIRD", "END"]),
    ("ELASTIC", {"mode": "PHOTOELA", "print": True}, "HSE06", ["PHOTOELA", "PRINT", "END"]),
    ("ELASTIC", {"mode": "PIEZOCP"}, "B3LYP", ["PIEZOCP", "END"]),
    ("ELASTIC", {"mode": "ELASTCON", "print": True}, "SCAN", ["ELASTCON", "PRINT", "END"]),
])
def test_supported_functional_writes_the_deck(si, calc_type, settings, functional, block):
    r = _config(si, calc_type, functional, settings)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(p for p in si.glob("*.d12") if p.name != "parent.d12").read_text().splitlines()
    assert functional in deck
    i = deck.index(block[0])
    assert deck[i:i + len(block)] == block


def test_photoela_dynamic_needs_a_pure_functional(si):
    """PHOTOELA DYNAMIC runs the frequency-dependent CPHF/KS, which CRYSTAL
    stops with Hartree-Fock or a hybrid (pp. 272-273), as CPHF DYNAMIC."""
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "ELASTIC", "--non-interactive",
              "--elastic-settings", json.dumps({"mode": "PHOTOELA", "dynamic": 589, "print": True}), cwd=si)
    assert r.returncode != 0 and "PHOTOELA DYNAMIC needs a pure LDA/GGA functional" in r.stdout + r.stderr
    r = _config(si, "ELASTIC", "PBE", {"mode": "PHOTOELA", "dynamic": 589})
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(p for p in si.glob("*.d12") if p.name != "parent.d12").read_text().splitlines()
    i = deck.index("PHOTOELA")
    assert deck[i:i + 4] == ["PHOTOELA", "DYNAMIC", "589", "END"]


@pytest.mark.skipif(not TEST_DATA.is_dir(), reason="test/ corpus not present")
@pytest.mark.parametrize("stem, functional", [
    ("OPT/1_dia_opt_rev1", "B3LYP-D3"),
    ("OPT/1LiFSI-1EMS-conf2_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized", "HSESOL3C"),
])
def test_corpus_parents_keep_their_cphf_decks(tmp_path, stem, functional):
    for ext in ("out", "d12"):
        shutil.copy(TEST_DATA / f"{stem}.{ext}", tmp_path)
    name = stem.split("/")[1]
    r = _mace("opt2d12", "--out-file", f"{name}.out", "--calc-type", "CPHF", "--non-interactive", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(tmp_path.glob("*_cphf_*.d12")).read_text().splitlines()
    assert functional in deck and "CPHF" in deck
