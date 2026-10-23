"""mace opt2d12 writes ELASTIC, EOS, QHA and CPHF decks without a prompt, from a
real finished OPT (the diamond B3LYP-D3 run of the test/ corpus): with
--calc-type and --non-interactive, and with a --config-file carrying
"elastic_settings" (the form a workflow step's config takes). Run through
``mace_cli`` as a subprocess, the real invocation path.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data

MACE_CLI = REPO_ROOT / "mace_cli"

# (calc type, block keyword written by default, the TOLDEE floor of the module)
TYPES = [("ELASTIC", "ELASTCON", 8), ("EOS", "EOS", 8), ("QHA", "QHA", 10), ("CPHF", "CPHF", 8)]


def _mace(*args, cwd):
    env = dict(os.environ, MACE_NO_BANNER="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300)


def _block(lines, key):
    i = lines.index(key)
    return lines[i:lines.index("END", i) + 1]


@pytest.fixture
def dia(tmp_path):
    out = find_data("OPT/1_dia_opt_rev1.out", must_contain="OPT END - CONVERGED")
    for f in (out, out.with_suffix(".d12")):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def _the_deck(folder, mode):
    decks = sorted(folder.glob(f"1_dia_opt_rev1_{mode.lower()}_*.d12"))
    assert len(decks) == 1, sorted(p.name for p in folder.iterdir())
    return decks[0].read_text().splitlines()


def _scf_value(lines, key):
    """The value of an SCF-section keyword (after the DFT block, not OPTGEOM's)."""
    start = lines.index("ENDDFT")
    return lines[lines.index(key, start) + 1]


@pytest.mark.parametrize("calc, mode, toldee", TYPES)
def test_non_interactive_calc_type_writes_the_module_deck(dia, calc, mode, toldee):
    r = _mace("opt2d12", "--out-file", "1_dia_opt_rev1.out", "--calc-type", calc, "--non-interactive", cwd=dia)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    lines = _the_deck(dia, mode)
    assert _block(lines, mode) == [mode, "END"]
    # a single point at the optimized geometry: no OPTGEOM, the parent's method kept
    assert "OPTGEOM" not in lines and "FREQCALC" not in lines
    assert "B3LYP-D3" in lines and "SPIN" in lines
    assert lines[4] != "3.55677029"                     # the optimized cell, not the input's
    assert int(_scf_value(lines, "TOLDEE")) >= toldee
    assert int(_scf_value(lines, "MAXCYCLE")) >= 200


@pytest.mark.parametrize("config, mode, block", [
    ({"calculation_type": "ELASTIC", "elastic_settings": {"mode": "ELAPIEZO", "numderiv": 5}},
     "ELAPIEZO", ["ELAPIEZO", "NUMDERIV", "5", "END"]),
    ({"calculation_type": "EOS", "elastic_settings": {"mode": "EOS", "range": [0.94, 1.06, 6], "print": True}},
     "EOS", ["EOS", "RANGE", "0.94 1.06 6", "PRINT", "END"]),
    ({"calculation_type": "QHA", "elastic_settings": {"temperat": "60 10 600", "scelphono": "2 2 2"}},
     "QHA", ["QHA", "TEMPERAT", "60 10.0 600.0", "END"]),
    ({"calculation_type": "CPHF", "elastic_settings": {"third": True}},
     "CPHF", ["CPHF", "THIRD", "END"]),
])
@pytest.mark.parametrize("flag", ["--non-interactive", "--yes"])
def test_config_file_carries_the_module_settings(dia, config, mode, block, flag):
    (dia / "step.json").write_text(json.dumps(config))
    r = _mace("opt2d12", "--out-file", "1_dia_opt_rev1.out", "--config-file", "step.json", flag, cwd=dia)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    lines = _the_deck(dia, mode)
    assert _block(lines, mode) == block
    if mode == "QHA":
        i = lines.index("SCELPHONO")
        assert lines[i:i + 4] == ["SCELPHONO", "2 0 0", "0 2 0", "0 0 2"] and lines[i + 4] == "QHA"


def test_a_config_block_of_another_type_is_refused(dia):
    (dia / "step.json").write_text(json.dumps({"calculation_type": "ELASTIC", "elastic_settings": {"mode": "EOS"}}))
    r = _mace("opt2d12", "--out-file", "1_dia_opt_rev1.out", "--config-file", "step.json", "--non-interactive",
              cwd=dia)
    assert r.returncode != 0
    assert "is --calc-type EOS" in r.stdout + r.stderr
    assert not list(dia.glob("1_dia_opt_rev1_*_*.d12"))


@pytest.mark.parametrize("args, words", [
    (["--config-file", "step.json"], "--calc-type EOS with an ELASTCON block (elastic_settings mode ELASTCON): use "
                                     "--calc-type ELASTIC"),
    (["--calc-type", "EOS", "--elastic-settings", '{"mode": "ELASTCON"}'], "use --calc-type ELASTIC"),
    (["--calc-type", "EOS", "--elastic-settings", '{"MODE": "qha"}'], "with a QHA block (elastic_settings mode "
                                                                      "QHA): use --calc-type QHA"),
])
def test_a_calc_type_never_overrides_another_modules_mode(dia, args, words):
    """--calc-type EOS with elastic_settings mode ELASTCON used to write an EOS
    deck without a word; the spec and the GUI refuse it, and so does opt2d12."""
    (dia / "step.json").write_text(json.dumps({"calculation_type": "EOS", "elastic_settings": {"mode": "ELASTCON"}}))
    r = _mace("opt2d12", "--out-file", "1_dia_opt_rev1.out", *args, "--non-interactive", cwd=dia)
    assert r.returncode != 0
    assert words in r.stdout + r.stderr, r.stdout[-2000:] + r.stderr[-2000:]
    assert not list(dia.glob("1_dia_opt_rev1_*_*.d12"))


@pytest.mark.parametrize("calc, mode, _toldee", TYPES)
def test_the_scf_maxcycle_floor_applies(dia, calc, mode, _toldee):
    """A parent whose SCF MAXCYCLE is 100: the module deck runs at a tighter
    TOLDEE, so it gets the 200-cycle floor (d12_constants.tight_scf_maxcycle)."""
    d12 = dia / "1_dia_opt_rev1.d12"
    text = d12.read_text()
    assert "MAXCYCLE\n1600\n" in text
    d12.write_text(text.replace("MAXCYCLE\n1600\n", "MAXCYCLE\n100\n"))
    r = _mace("opt2d12", "--out-file", "1_dia_opt_rev1.out", "--calc-type", calc, "--non-interactive", cwd=dia)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert _scf_value(_the_deck(dia, mode), "MAXCYCLE") == "200"


# ------------------------------------------------------------------ phonopy: choice 8
def test_help_offers_phonopy(tmp_path):
    """Finite-displacement phonons with phonopy are opt2d12's PHONOPY (choice 8):
    opt2d12's help names every --calc-type, PHONOPY among them, and its flags."""
    r = _mace("opt2d12", "--help", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    text = " ".join(r.stdout.split())
    assert "--calc-type {SP,OPT,FREQ,ELASTIC,EOS,QHA,CPHF,PHONOPY}" in text
    assert "mace analyze phonopy DIR" in text


def test_the_menu_offers_phonopy_as_8(dia):
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "1_dia_opt_rev1.out"], cwd=dia,
                       env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "MACE_NO_BANNER": "1"},
                       input="n\n4\n" + "\n" * 80, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    menu = r.stdout[r.stdout.index("Select calculation type:"):]
    menu = menu[:menu.index("8. ") + 200]
    assert "8. Finite-displacement phonons with phonopy (PHONOPY)" in menu
    assert "phonopy generate" not in menu
