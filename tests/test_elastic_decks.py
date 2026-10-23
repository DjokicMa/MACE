"""ELASTCON / ELAPIEZO / PIEZOCON / PIEZOCP / PHOTOELA / EOS decks
(Crystal_d12/d12_calc_elastic.py, CRYSTAL23 manual ch. 12-13).

The block syntax and defaults are the manual's (pp. 284, 291, 296-302); the
decks these tests write from the repo's Si parent are the kind that ran on
HPCC (tests/data/hpcc_116/mechanics: the diamond decks of the same writer,
all terminated normally). Generation goes through ``mace_cli`` as a
subprocess, the real invocation path.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_calc_elastic as E  # noqa: E402

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
HPCC = REPO_ROOT / "tests" / "data" / "hpcc_116"


def _mace(*args, cwd):
    env = dict(os.environ, MACE_NO_BANNER="1")
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _block(deck_text, key):
    lines = deck_text.splitlines()
    i = lines.index(key)
    j = lines.index("END", i)
    return lines[i:j + 1]


# ------------------------------------------------------------------ the block
def test_every_default_writes_the_minimal_block():
    for mode in E.ELASTIC_MODES + ("EOS",):
        s = E.normalize({"mode": mode}, "EOS" if mode == "EOS" else "ELASTIC")
        assert E.block_lines(s) == [mode, "END"]


def test_the_manual_example_of_elastcon():
    """p. 292: NUMDERIV 5, STEPSIZE 0.02, PRINT."""
    s = E.normalize({"mode": "ELASTCON", "numderiv": 5, "stepsize": 0.02, "print": True})
    assert E.block_lines(s) == ["ELASTCON", "NUMDERIV", "5", "STEPSIZE", "0.02", "PRINT", "END"]


def test_values_equal_to_the_defaults_are_not_written():
    s = E.normalize({"mode": "ELAPIEZO", "numderiv": 3, "stepsize": 0.015, "toldeg": 0.0003, "toldee": 8})
    assert E.block_lines(s) == ["ELAPIEZO", "END"]
    s = E.normalize({"mode": "ELASTCON", "stepsize": 0.015})        # 0.015 is not ELASTCON's 0.01
    assert E.block_lines(s) == ["ELASTCON", "STEPSIZE", "0.015", "END"]


def test_nuclear_relaxation_and_pressure_records():
    """p. 294-295: PREOPTGEOM + PRESSURE P; NUCHESS HESSNUM2 MODEPART HESSSTEP."""
    s = E.normalize({"mode": "ELASTCON", "pressure": 5, "nuchess": True, "hessnum2": True,
                     "modepart": True, "hessstep": 0.005})
    notes = E.validate(s, "CRYSTAL")
    assert any("PREOPTGEOM" in n for n in notes) and s["preoptgeom"] is True
    assert E.block_lines(s) == ["ELASTCON", "PRESSURE", "5.0", "PREOPTGEOM", "NUCHESS", "HESSNUM2",
                                "MODEPART", "HESSSTEP", "0.005", "END"]


def test_eos_range_and_printed_ranges():
    """p. 284-285: RANGE VOL1 VOL2 NPOINTS; the manual's quartz example."""
    s = E.normalize({"range": "0.90 1.05 10", "prange": [-5, 10, 20]}, "EOS")
    assert E.block_lines(s) == ["EOS", "RANGE", "0.9 1.05 10", "PRANGE", "-5.0 10.0 20", "END"]


def test_photoela_and_piezocp_records():
    s = E.normalize({"mode": "PHOTOELA", "dynamic": 589, "tolalpha": 5, "fmixing": 40, "clampion": True})
    assert E.block_lines(s) == ["PHOTOELA", "CLAMPION", "TOLALPHA", "5", "FMIXING", "40", "DYNAMIC", "589", "END"]
    s = E.normalize({"mode": "PIEZOCP", "hessnum2": True, "tolalpha": 2})      # p. 298's example
    assert E.block_lines(s) == ["PIEZOCP", "HESSNUM2", "TOLALPHA", "2", "END"]


def test_seismdir_and_deform():
    s = E.normalize({"mode": "ELASTCON", "deform": "1 1 1 0 0 0", "seismdir": [[1, 1, 0], "1 1 1"]})
    assert E.block_lines(s) == ["ELASTCON", "DEFORM", "1 1 1 0 0 0", "SEISMDIR", "2", "1.0 1.0 0.0",
                                "1.0 1.0 1.0", "END"]


def test_parse_block_reads_back_what_was_written():
    s = E.normalize({"mode": "ELASTCON", "numderiv": 5, "deform": [1, 0, 0, 0, 0, 1], "nuchess": True,
                     "hessnum2": True, "seismdir": [[1, 1, 0]], "restart": "RESTART2", "print": True})
    lines = E.block_lines(s)
    back, end = E.parse_block(lines, 0)
    assert end == len(lines) and E.block_lines(back) == lines


# ------------------------------------------------------------- the manual's limits
@pytest.mark.parametrize("settings, dim, words", [
    ({"mode": "EOS"}, "SLAB", "3D"),
    ({"mode": "PIEZOCP"}, "POLYMER", "3D crystals"),
    ({"mode": "ELASTCON"}, "MOLECULE", "no lattice"),
    ({"mode": "ELASTCON", "deform": [0, 0, 1, 0, 0, 0]}, "SLAB", "non-periodic"),
    ({"mode": "ELASTCON", "deform": [0, 0, 0, 0, 0, 0]}, "CRYSTAL", "no active deformation"),
    ({"mode": "ELASTCON", "numderiv": 1}, "CRYSTAL", "at least 2"),
    ({"mode": "ELASTCON", "nuchess": True, "clampion": True}, "CRYSTAL", "choose one"),
    ({"mode": "ELASTCON", "hessnum2": True}, "CRYSTAL", "NUCHESS"),
    ({"mode": "ELASTCON", "pressure": 1, "presseos": 1}, "CRYSTAL", "exclude each other"),
    ({"mode": "PIEZOCON", "tolalpha": 3}, "CRYSTAL", "does not read TOLALPHA"),
    ({"mode": "EOS", "range": [1.1, 0.9, 6]}, "CRYSTAL", "not below"),
])
def test_refusals(settings, dim, words):
    with pytest.raises(E.ElasticSettingsError) as err:
        E.validate(E.normalize(settings, "EOS" if settings["mode"] == "EOS" else "ELASTIC"), dim)
    assert words in str(err.value)


def test_two_component_parents_are_refused():
    with pytest.raises(E.ElasticSettingsError) as err:
        E.validate(E.normalize({"mode": "ELASTCON"}), "CRYSTAL", two_component=True)
    assert "TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM" in str(err.value)


def test_berry_phase_modules_refuse_mppcrystal():
    """p. 296 and 298: PIEZOCON / ELAPIEZO run with Pcrystal only."""
    with pytest.raises(E.ElasticSettingsError, match="Pcrystal only"):
        E.validate(E.normalize({"mode": "ELAPIEZO"}), "CRYSTAL", scf_executable="MPPcrystal")
    assert E.validate(E.normalize({"mode": "ELASTCON"}), "CRYSTAL", scf_executable="MPPcrystal") == []


def test_scf_toldee_is_never_looser_than_the_modules():
    assert E.scf_tolerances_for({"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}) == {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 8}
    assert E.scf_tolerances_for({"TOLDEE": 10}) == {"TOLDEE": 10}            # tighter: the parent's
    assert E.scf_tolerances_for(None) == {"TOLDEE": 8}


def test_configure_elastic_with_answers():
    answers = iter(["2",        # ELAPIEZO
                    "5", "0.02",  # NUMDERIV, STEPSIZE
                    "2", "y",   # NUCHESS + HESSNUM2
                    "n", "y"])  # PREOPTGEOM no, PRINT yes
    s = E.configure_elastic("ELASTIC", "CRYSTAL", ask=lambda p: next(answers))
    assert E.block_lines(s) == ["ELAPIEZO", "NUMDERIV", "5", "STEPSIZE", "0.02", "NUCHESS", "HESSNUM2", "PRINT", "END"]


# ------------------------------------------------------------------ opt2d12
@pytest.fixture
def si(tmp_path):
    for f in ("parent.out", "parent.d12"):
        shutil.copy(SI / f, tmp_path / f)
    return tmp_path


@pytest.mark.parametrize("mode, calc, block", [
    ("ELASTCON", "ELASTIC", ["ELASTCON", "NUCHESS", "HESSNUM2", "PRINT", "END"]),
    ("ELAPIEZO", "ELASTIC", ["ELAPIEZO", "PRINT", "END"]),
    ("PHOTOELA", "ELASTIC", ["PHOTOELA", "PRINT", "END"]),
    ("EOS", "EOS", ["EOS", "RANGE", "0.94 1.06 6", "PRINT", "END"]),
])
def test_opt2d12_writes_the_module_deck(si, mode, calc, block):
    settings = {"mode": mode, "print": True}
    if mode == "ELASTCON":
        settings.update(nuchess=True, hessnum2=True)
    if mode == "EOS":
        settings["range"] = [0.94, 1.06, 6]
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", calc, "--non-interactive",
              "--elastic-settings", json.dumps(settings), cwd=si)
    assert r.returncode == 0, r.stdout + r.stderr
    decks = sorted(p for p in si.glob(f"parent_{mode.lower()}_*.d12"))
    assert len(decks) == 1, list(si.iterdir())
    text = decks[0].read_text()
    assert _block(text, mode) == block
    lines = text.splitlines()
    i = lines.index(mode)
    # the last keyword of the geometry input: right after the atoms, before the basis
    assert "OPTGEOM" not in lines and "FREQCALC" not in lines
    toldee = lines[lines.index("TOLDEE") + 1]
    assert int(toldee) >= 8
    assert lines[i + len(block)] in ("END", "BASISSET")


def test_opt2d12_names_each_module_apart(si):
    for mode in ("ELASTCON", "ELAPIEZO"):
        r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "ELASTIC", "--non-interactive",
                  "--elastic-settings", json.dumps({"mode": mode}), cwd=si)
        assert r.returncode == 0, r.stdout + r.stderr
    assert len(list(si.glob("parent_elastcon_*.d12"))) == 1 and len(list(si.glob("parent_elapiezo_*.d12"))) == 1


def test_opt2d12_refuses_a_bad_block_and_writes_nothing(si):
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "ELASTIC", "--non-interactive",
              "--elastic-settings", '{"mode": "ELASTCON", "deform": [0,0,0,0,0,0]}', cwd=si)
    assert r.returncode != 0
    assert "no active deformation" in r.stdout + r.stderr
    assert not list(si.glob("parent_elastcon_*.d12"))


def test_opt2d12_refuses_a_two_component_parent(tmp_path):
    """The SOC CsPbI3 run (HPCC, CRYSTAL23) as the parent."""
    for f in ("CsPbI3_soc.out", "CsPbI3_soc.d12"):
        shutil.copy(HPCC / "soc" / f, tmp_path / f)
    r = _mace("opt2d12", "--out-file", "CsPbI3_soc.out", "--calc-type", "ELASTIC", "--non-interactive",
              cwd=tmp_path)
    assert r.returncode != 0, r.stdout
    assert "TWO-COMPONENT" in r.stdout + r.stderr
    assert not list(tmp_path.glob("*_elastcon_*.d12"))


def test_opt2d12_writes_the_module_decks(si):
    """The decks the 1.1.6 elastic generate command wrote: its flags are the --elastic-settings keys
    (its PRINT default is "print": true; with nothing given opt2d12 writes CRYSTAL's defaults)."""
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "EOS", "--non-interactive",
              "--elastic-settings", json.dumps({"range": "0.9 1.1 8", "print": True}), cwd=si)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(si.glob("parent_eos_*.d12")).read_text()
    assert _block(deck, "EOS") == ["EOS", "RANGE", "0.9 1.1 8", "PRINT", "END"]
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "ELASTIC", "--non-interactive",
              "--elastic-settings", '{"print": true}', cwd=si)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(si.glob("parent_elastcon_*.d12")).read_text()
    assert _block(deck, "ELASTCON") == ["ELASTCON", "PRINT", "END"]


def test_mace_elastic_is_gone(si):
    r = _mace("elastic", "generate", "parent.out", "--mode", "EOS", cwd=si)
    assert r.returncode == 2
    assert r.stderr.strip() == ("mace elastic was folded into mace opt2d12 (choices 4-7 / --calc-type) "
                                "and mace analyze elastic")
    assert not list(si.glob("parent_eos_*.d12"))


def test_opt2d12_refuses_an_unfinished_parent_for_a_module(si):
    """As for phonopy's displaced supercells (phonopy_bridge.parent.run_refusals): a module deck
    starts from a finished OPT or SP run, refused before anything is asked, from --calc-type
    and from menu choices 4-7 alike."""
    text = (si / "parent.out").read_text()
    (si / "parent.out").write_text(text[:text.index("EEEEEEEEEE TERMINATION")])
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "EOS", "--non-interactive", cwd=si)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Refusing parent.out:" in r.stderr and "[ABNORMAL_TERMINATION]" in r.stderr, r.stderr
    env = dict(os.environ, MACE_NO_BANNER="1")
    p = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out"], cwd=si, env=env,
                       input="n\n5\n", capture_output=True, text=True, timeout=300)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "Refusing parent.out:" in p.stderr and "[ABNORMAL_TERMINATION]" in p.stderr, p.stderr
    assert "Select method type" not in p.stdout
    assert not list(si.glob("parent_eos_*.d12"))


@pytest.mark.parametrize("name", ["dia_elastcon", "dia_elapiezo", "dia_eos"])
def test_the_hpcc_decks_are_this_writer_s(name):
    """The diamond decks that ran (and terminated normally) on HPCC carry the
    block this writer gives for their settings."""
    deck = (HPCC / "mechanics" / f"{name}.d12").read_text()
    mode = name.split("_")[1].upper()
    lines = deck.splitlines()
    i = lines.index(mode)
    s, end = E.parse_block(lines, i)
    assert E.block_lines(s) == lines[i:end]
    out = (HPCC / "mechanics" / f"{name}.out").read_text()
    assert "EEEEEEEEEE TERMINATION" in out


# ------------------------------------------------------------------ QHA (manual ch. 9)
def test_qha_block_of_the_manual():
    """p. 253-254: STEP 3.5, POINTS 7; TEMPERAT 100 10 2000; SCELPHONO before QHA (p. 255)."""
    s = E.normalize({"qstep": 3.5, "points": 7, "temperat": "100 10 2000", "scelphono": "2 2 2"}, "QHA")
    assert E.geometry_lines(s) == ["SCELPHONO", "2 0 0", "0 2 0", "0 0 2"]
    assert E.block_lines(s) == ["QHA", "STEP", "3.5", "POINTS", "7", "TEMPERAT", "100 10.0 2000.0", "END"]
    assert E.block_lines(E.normalize({}, "QHA")) == ["QHA", "END"]
    back, _ = E.parse_block(E.block_lines(s), 0)
    assert E.block_lines(back) == E.block_lines(s)


@pytest.mark.parametrize("settings, words", [
    ({"points": 5}, "4, 7 or 13"),
    ({"temperat": [20, 10, 300]}, "at least 50"),
    ({"scelphono": [1, 0, 0, 0, 1, 0, 0, 0, -1]}, "positive determinant"),
])
def test_qha_refusals(settings, words):
    with pytest.raises(E.ElasticSettingsError, match=words):
        E.validate(E.normalize(settings, "QHA"), "CRYSTAL")
    # QHA runs for slabs and polymers (manual p. 14, measured on CRYSTAL23), not for a molecule
    for dim in ("SLAB", "POLYMER"):
        E.validate(E.normalize({}, "QHA"), dim)
        with pytest.raises(E.ElasticSettingsError, match="3D crystal only"):
            E.validate(E.normalize({"scelphono": [2, 0, 0, 0, 2, 0, 0, 0, 1]}, "QHA"), dim)
    with pytest.raises(E.ElasticSettingsError, match="MOLECULES"):
        E.validate(E.normalize({}, "QHA"), "MOLECULE")


def test_opt2d12_qha_deck_runs_at_freqcalc_toldee(si):
    r = _mace("opt2d12", "--out-file", "parent.out", "--calc-type", "QHA", "--non-interactive", "--elastic-settings",
              json.dumps({"temperat": "60 10 600", "scelphono": "2 2 2"}), cwd=si)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = next(si.glob("parent_qha_*.d12")).read_text().splitlines()
    i = lines.index("SCELPHONO")
    assert lines[i:i + 8] == ["SCELPHONO", "2 0 0", "0 2 0", "0 0 2", "QHA", "TEMPERAT", "60 10.0 600.0", "END"]
    assert int(lines[lines.index("TOLDEE") + 1]) >= 10


# ------------------------------------------------------------------ CPHF (manual 10.2)
def test_cphf_block():
    """p. 271-273: TOLALPHA default 3; DYNAMIC NSTEPS + wavelength(s); THIRD, FOURTH."""
    assert E.block_lines(E.normalize({}, "CPHF")) == ["CPHF", "END"]
    s = E.normalize({"third": True, "cphf_dynamic": "3 400 800", "damping": 0.002, "tolalpha": 4,
                     "maxcycle": 100, "histdiis": 20}, "CPHF")
    assert E.block_lines(s) == ["CPHF", "HISTDIIS", "20", "DYNAMIC", "3", "400.0 800.0", "DAMPING", "0.002",
                                "THIRD", "TOLALPHA", "4", "END"]
    back, _ = E.parse_block(E.block_lines(s), 0)
    assert E.block_lines(back) == E.block_lines(s)
    with pytest.raises(E.ElasticSettingsError, match="NSTEPS 1"):
        E.validate(E.normalize({"cphf_dynamic": [1, 400, 800]}, "CPHF"), "MOLECULE")
    assert E.validate(E.normalize({"fourth": True}, "CPHF"), "MOLECULE") == []
    with pytest.raises(E.ElasticSettingsError, match="p. 166"):
        E.validate(E.normalize({}, "CPHF"), "CRYSTAL", two_component=True)


def test_cphf_dynamic_needs_a_pure_functional(si):
    """The Si parent is PBE0 (a hybrid): CRYSTAL stops DYNAMIC CPHF with one (p. 272)."""
    cphf = ("opt2d12", "--out-file", "parent.out", "--calc-type", "CPHF", "--non-interactive", "--elastic-settings")
    r = _mace(*cphf, json.dumps({"cphf_dynamic": "1 589"}), cwd=si)
    assert r.returncode != 0 and "pure LDA/GGA" in r.stdout + r.stderr
    r = _mace(*cphf, json.dumps({"third": True}), cwd=si)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(si.glob("parent_cphf_*.d12")).read_text()
    assert _block(deck, "CPHF") == ["CPHF", "THIRD", "END"]


def test_interactive_menu_offers_the_modules_after_freq(si):
    """mace opt2d12 at its calculation-type menu: 4-7 are ELASTIC, EOS, QHA and CPHF (appended: 1-3 keep
    SP, OPT and FREQ, so older answer files keep their meaning); 4 then asks the module's questions."""
    env = dict(os.environ, MACE_NO_BANNER="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out"], cwd=si, env=env,
                       input="n\n4\n" + "\n" * 80, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    menu = r.stdout[r.stdout.index("Select calculation type:"):]
    for n, kind in (("1", "SP"), ("2", "OPT"), ("3", "FREQ"), ("4", "ELASTIC"), ("5", "EOS"), ("6", "QHA"), ("7", "CPHF")):
        assert f"{n}. " in menu and f"({kind})" in menu.split(f"{n}. ", 1)[1].splitlines()[0], (n, kind)
    deck = (si / "parent_elastcon_PBE0_optimized.d12").read_text()
    assert _block(deck, "ELASTCON") == ["ELASTCON", "PRINT", "END"]
