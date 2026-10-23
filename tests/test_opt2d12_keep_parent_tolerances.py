"""opt2d12's preset menus offer a parent's own values when they are no preset.

phonopy's Si example runs TOLINTEG 8 8 8 8 16 / TOLDEE 9, which is none of the
Standard/Tight/Very tight levels. The SCF tolerance menu listed only the
presets and Custom, so the parent's values were not a choice anyone could pick
(the Setup form showed the raw "8 8 8 8 16" outside its options, and Custom
offered 7 7 7 7 14 / 7). The same held for the optimization convergence menu.
Both now list "5. Keep the parent's - <values>" after Custom, as the default;
1-4 keep their numbers. The decks run the real ``mace_cli`` with the Si parent
in tests/fixtures/phonopy_crystal/Si.
"""
import os
import shutil
import subprocess
import sys

import pytest

import d12_constants
from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
SCF_KEEP = "5. Keep the parent's - TOLINTEG: 8 8 8 8 16, TOLDEE: 9"


def _env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["MPLBACKEND"] = "Agg"
    return env


def _run(cwd, args, stdin=""):
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", *args], cwd=cwd, input=stdin,
                          capture_output=True, text=True, timeout=300, env=_env())


def _records(deck_text):
    """The SCF TOLINTEG/TOLDEE records (after the basis set) and OPTGEOM's."""
    lines = [x.strip() for x in deck_text.splitlines()]
    out = {}
    in_opt = False
    for i, line in enumerate(lines[:-1]):
        if line == "OPTGEOM":
            in_opt = True
        elif line == "ENDOPT":
            in_opt = False
        elif line in ("TOLINTEG", "TOLDEE", "TOLDEG", "TOLDEX", "MAXCYCLE"):
            out[("OPT " if in_opt else "") + line] = " ".join(lines[i + 1].split())
    return out


@pytest.fixture
def si(tmp_path):
    for name in ("parent.out", "parent.d12"):
        shutil.copy(SI / name, tmp_path / name)
    return tmp_path


def _deck(folder, pattern="*.d12", exclude=("parent.d12", "si.d12")):
    decks = [p for p in folder.glob(pattern) if p.name not in exclude]
    assert len(decks) == 1, decks
    return decks[0].read_text()


# ------------------------------------------------------------- the SCF menu

def test_enter_through_offers_and_keeps_the_parents_tolerances(si):
    r = _run(si, ["--out-file", "parent.out"], "\n" * 200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert SCF_KEEP in r.stdout
    assert "Select SCF convergence level [1-5, default=5]" in r.stdout
    # the presets keep their numbers
    assert "1. Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7" in r.stdout
    assert "4. Custom - Set your own tolerances" in r.stdout
    rec = _records(_deck(si))
    assert (rec["TOLINTEG"], rec["TOLDEE"]) == ("8 8 8 8 16", "9")


@pytest.mark.parametrize("calc", ["SP", "OPT"])
def test_non_interactive_keeps_the_parents_tolerances(si, calc):
    r = _run(si, ["--out-file", "parent.out", "--non-interactive", "--calc-type", calc, "--output-dir", "out"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rec = _records(_deck(si / "out"))
    assert (rec["TOLINTEG"], rec["TOLDEE"]) == ("8 8 8 8 16", "9")


def test_a_preset_parent_gets_no_keep_option(tmp_path):
    shutil.copy(SI / "parent.out", tmp_path / "parent.out")
    text = (SI / "parent.d12").read_text().replace("8 8 8 8 16", "8 8 8 9 24")
    (tmp_path / "parent.d12").write_text(text)
    r = _run(tmp_path, ["--d12-file", "parent.d12"], "\n" * 200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Keep the parent's - TOLINTEG" not in r.stdout
    assert "Select SCF convergence level [1-4, default=2]" in r.stdout


def test_custom_defaults_to_the_parents_values(monkeypatch):
    """Custom used to offer Standard (7 7 7 7 14 / 7) whatever the parent had."""
    seen = []
    monkeypatch.setattr("builtins.input", lambda prompt="": seen.append(prompt) or "")
    parent = {"TOLINTEG": "8  8 8 8 16", "TOLDEE": 9}
    assert d12_constants.configure_tolerances(calculation_type="SP", current=parent) == {
        "TOLINTEG": "8 8 8 8 16", "TOLDEE": 9}
    assert any("Select tolerance level (1-4) [4]" in p for p in seen), seen
    assert any("[8 8 8 8 16]" in p for p in seen) and any("TOLDEE value (integer) [9]" in p for p in seen)
    # a parent at a preset: that level is the default
    assert d12_constants.configure_tolerances(calculation_type="SP",
                                              current=d12_constants.scf_tolerances("2")) == \
        d12_constants.scf_tolerances("2")


# ------------------------------------------------- the optimization menu

def test_enter_through_offers_and_keeps_the_parents_optgeom(tmp_path):
    lines = (SI / "parent.d12").read_text().splitlines()
    at = lines.index("END")              # the end of the geometry block
    lines[at:at] = ["OPTGEOM", "FULLOPTG", "TOLDEG", "0.00006", "TOLDEX", "0.00024", "MAXCYCLE", "300", "ENDOPT"]
    (tmp_path / "si.d12").write_text("\n".join(lines) + "\n")
    r = _run(tmp_path, ["--d12-file", "si.d12"], "\n" * 200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "5. Keep the parent's - TOLDEG=0.00006, TOLDEX=0.00024, MAXCYCLE=300" in r.stdout
    assert "Select convergence level [1-5, default=5]" in r.stdout
    assert "4. Custom - Set your own criteria" in r.stdout
    rec = _records(_deck(tmp_path))
    assert (rec["OPT TOLDEG"], rec["OPT TOLDEX"], rec["OPT MAXCYCLE"]) == ("0.00006", "0.00024", "300")
    assert "OPT TOLDEE" not in rec                     # the parent wrote none
    assert (rec["TOLINTEG"], rec["TOLDEE"]) == ("8 8 8 8 16", "9")
