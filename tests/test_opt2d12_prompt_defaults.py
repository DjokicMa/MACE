"""Every opt2d12 prompt defaults to what the parent deck had.

A blank answer (a user pressing Enter, or the planless engine path piping blank
lines) must keep the parent's value; an explicit answer still wins. These are
the prompts that did not:

* OPT convergence menu: always defaulted to Standard, so a tight parent
  (TOLDEG 0.0001 / TOLDEX 0.0004 / TOLDEE 8) was loosened, and values that
  match no preset were lost; the optimization type reset to FULLOPTG;
* "Set MAXTRADIUS?": defaulted to no and dropped the parent's MAXTRADIUS;
* FREQ SCF tolerance menu: defaulted to Very tight whatever the parent had;
* "Use level shifting?" answered yes on a parent without LEVSHIFT: stored in a
  key the deck writer never reads, so no LEVSHIFT was written.

The corpus tests run the real ``mace_cli opt2d12`` and skip when ``test/`` is
absent. The other tests run everywhere.
"""
import json
import shutil
import subprocess
import sys
import textwrap

import pytest

import d12_calc_basic
import d12_interactive
import menu_nav
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"


def _blank_input(monkeypatch, answers=None):
    """Answer prompts by substring from ``answers``; blank otherwise."""
    answers = answers or {}
    seen = []

    def fake_input(prompt=""):
        seen.append(prompt)
        for sub, reply in answers.items():
            if sub in prompt:
                return reply
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    # menu_nav keeps its own reference to input(), taken at import.
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", fake_input)
    return seen


# ------------------------------------------------------ OPT convergence menu

TIGHT = {"type": "FULLOPTG", "MAXCYCLE": 800, "TOLDEG": 0.0001, "TOLDEX": 0.0004, "TOLDEE": 8}


def test_blank_convergence_answer_keeps_a_tight_parent(monkeypatch):
    seen = _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl(dict(TIGHT))
    assert (cfg["toldeg"], cfg["toldex"], cfg["toldee"]) == (0.0001, 0.0004, 8)
    assert any("default=2" in p for p in seen), seen


def test_blank_convergence_answer_keeps_values_matching_no_preset(monkeypatch):
    parent = {"type": "FULLOPTG", "MAXCYCLE": 1600, "TOLDEG": 0.0002, "TOLDEX": 0.0008, "TOLDEE": 8}
    seen = _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl(parent)
    assert (cfg["toldeg"], cfg["toldex"], cfg["toldee"], cfg["maxcycle"]) == (0.0002, 0.0008, 8, 1600)
    assert any("default=keep current" in p for p in seen), seen


def test_tolerance_the_parent_left_out_stays_out(monkeypatch):
    """CRYSTAL's default applied to the parent, so it must apply to the child."""
    _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl({"type": "FULLOPTG", "MAXCYCLE": 1600})
    assert "toldeg" not in cfg and "toldex" not in cfg and "toldee" not in cfg
    assert cfg["maxcycle"] == 1600


def test_explicit_convergence_answer_still_wins(monkeypatch):
    _blank_input(monkeypatch, {"convergence level": "1"})
    cfg = d12_calc_basic._configure_optimization_impl(dict(TIGHT))
    assert (cfg["toldeg"], cfg["toldex"], cfg["toldee"]) == (0.0003, 0.0012, 7)


def test_no_parent_optimization_still_defaults_to_standard(monkeypatch):
    _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl()
    assert (cfg["toldeg"], cfg["toldex"], cfg["toldee"]) == (0.0003, 0.0012, 7)
    assert cfg["type"] == "FULLOPTG"


def test_parent_optimization_type_is_the_default(monkeypatch):
    _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl({**TIGHT, "type": "ATOMONLY"})
    assert cfg["type"] == "ATOMONLY"


def test_blank_maxtradius_answer_keeps_the_parents(monkeypatch):
    _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl({**TIGHT, "MAXTRADIUS": 0.25})
    assert cfg["maxtradius"] == 0.25


def test_maxtradius_can_still_be_turned_off(monkeypatch):
    _blank_input(monkeypatch, {"MAXTRADIUS) for geometry": "n"})
    cfg = d12_calc_basic._configure_optimization_impl({**TIGHT, "MAXTRADIUS": 0.25})
    assert cfg["maxtradius"] is None      # recorded as off, not just left out


@pytest.mark.parametrize("opt,expected", [
    ({"TOLDEG": 0.0003, "TOLDEX": 0.0012, "TOLDEE": 7}, "1"),
    ({"toldeg": 0.0001, "toldex": 0.0004, "toldee": 8}, "2"),
    ({"TOLDEG": 3e-05, "TOLDEX": 0.00012, "TOLDEE": 9}, "3"),
    ({"MAXCYCLE": 800}, "1"),                       # CRYSTAL defaults
    ({"TOLDEG": 0.0001, "TOLDEX": 0.0004, "TOLDEE": 7}, "keep"),
])
def test_opt_preset_matching(opt, expected):
    assert d12_calc_basic._opt_preset_for(opt) == expected


# ------------------------------------------------------------ LEVSHIFT answer

def _advanced(monkeypatch, options, answers):
    _blank_input(monkeypatch, answers)
    return d12_interactive.configure_advanced_electronic_settings(options, force_configure=True)


BASE = {"spin_polarized": False, "dimensionality": "CRYSTAL",
        "scf_settings": {"method": "DIIS", "maxcycle": 800, "fmixing": 30, "histdiis": None}}


def test_yes_to_level_shifting_reaches_the_deck_record(monkeypatch):
    cfg = _advanced(monkeypatch, dict(BASE), {"level shifting": "y", "ISHIFT ILOCK": "3 0"})
    assert tuple(cfg["scf_settings"]["levshift"]) == (3, 0)


def test_yes_to_level_shifting_with_blank_values_uses_the_offered_default(monkeypatch):
    cfg = _advanced(monkeypatch, dict(BASE), {"level shifting": "y"})
    assert tuple(cfg["scf_settings"]["levshift"]) == (5, 1)


# ------------------------------------------------- real opt2d12 on the corpus

def _copy_parent(stem, tmp_path):
    src = TEST_DATA / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return src.stem


def _opt2d12(tmp_path, name, answers=None, stdin=None, extra=()):
    """Run the real mace_cli opt2d12.

    With ``answers`` ({prompt substring: reply}, blank otherwise) the prompts
    are answered by content through a small runner, so the test does not
    depend on the prompt order. With ``stdin`` the lines are piped as is.
    """
    args = ["opt2d12", "--out-file", f"{name}.out", "--d12-file", f"{name}.d12", *extra]
    if answers is not None:
        runner = tmp_path / "_answer.py"
        runner.write_text(textwrap.dedent(f"""
            import builtins, json, runpy, sys
            answers = json.loads({json.dumps(json.dumps(answers))})
            def fake_input(prompt=""):
                print(prompt)
                for sub, reply in answers.items():
                    if sub in prompt:
                        return reply
                return ""
            builtins.input = fake_input
            sys.argv = [{str(MACE_CLI)!r}] + {args!r}
            runpy.run_path({str(MACE_CLI)!r}, run_name="__main__")
        """))
        cmd, stdin = [sys.executable, str(runner)], ""
    else:
        cmd = [sys.executable, str(MACE_CLI), *args]
    result = subprocess.run(cmd, cwd=tmp_path, input=stdin, capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _after(lines, keyword):
    return lines[lines.index(keyword) + 1].strip()


def _scf_after(lines, keyword):
    """The SCF-block value: the last occurrence (OPTGEOM has a TOLDEE too)."""
    i = len(lines) - 1 - lines[::-1].index(keyword)
    return lines[i + 1].strip()


TIGHT_OPT_PARENT = "OPT/4LG_2x2_AA_opt_HSESOL3C_optimized"   # TOLDEG 1e-4, TOLDEX 4e-4, TOLDEE 8
MAXTRADIUS_PARENT = "OPT/1_dia_opt_rev1"                      # MAXTRADIUS 0.25
NO_LEVSHIFT_PARENT = "OPT/1_dia_opt_BULK_OPTGEOM"             # no LEVSHIFT


def test_opt_child_keeps_tight_optgeom_on_blank_answers(tmp_path):
    name = _copy_parent(TIGHT_OPT_PARENT, tmp_path)
    parent = (tmp_path / f"{name}.d12").read_text().splitlines()
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "2"})
    opt = child[child.index("OPTGEOM"):child.index("ENDOPT")]
    popt = parent[parent.index("OPTGEOM"):parent.index("ENDOPT")]
    for kw in ("TOLDEG", "TOLDEX", "TOLDEE"):
        assert float(_after(opt, kw)) == float(_after(popt, kw)), kw
    assert "HSESOL3C" in child                       # method untouched


def test_opt_child_keeps_maxtradius_on_blank_answers(tmp_path):
    name = _copy_parent(MAXTRADIUS_PARENT, tmp_path)
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "2"})
    assert float(_after(child, "MAXTRADIUS")) == 0.25


def test_freq_child_keeps_parent_scf_tolerances_on_blank_answers(tmp_path):
    name = _copy_parent(TIGHT_OPT_PARENT, tmp_path)
    parent = (tmp_path / f"{name}.d12").read_text().splitlines()
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "3"})
    assert "FREQCALC" in child
    assert _scf_after(child, "TOLINTEG").split() == _scf_after(parent, "TOLINTEG").split()
    assert _scf_after(child, "TOLDEE") == _scf_after(parent, "TOLDEE")


def test_level_shift_answer_is_written(tmp_path):
    name = _copy_parent(NO_LEVSHIFT_PARENT, tmp_path)
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "level shifting": "y",
                                      "ISHIFT ILOCK": "3 1"},
                     extra=["--calc-type", "SP"])
    assert _after(child, "LEVSHIFT") == "3 1"


