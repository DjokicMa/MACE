"""Generated decks keep the parent's HISTDIIS, GUESSP and 3c grid records.

* HISTDIIS: the writer added HISTDIIS 100 to every DIIS deck, whatever the
  parent had or the answer to "Use HISTDIIS?" said;
* the grid keyword of a 3c parent other than HSESOL3C (e.g. PBEH3C / XLGRID)
  was dropped, both by the grid prompt and by the writer;
* the parent's GUESSP was dropped.

These records reach every opt2d12 path (interactive, planless, config), so the
corpus tests cover more than one. They skip when ``test/`` is absent.
"""
import io
import json
import shutil
import subprocess
import sys
import textwrap

import pytest

import d12_interactive
import menu_nav
from conftest import REPO_ROOT, TEST_DATA
from d12_writer import write_dft_section, write_scf_section

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


# ----------------------------------------------------------- HISTDIIS answer

def _advanced(monkeypatch, options, answers):
    _blank_input(monkeypatch, answers)
    return d12_interactive.configure_advanced_electronic_settings(options, force_configure=True)


BASE = {"spin_polarized": False, "dimensionality": "CRYSTAL",
        "scf_settings": {"method": "DIIS", "maxcycle": 800, "fmixing": 30, "histdiis": None}}


def test_parent_without_histdiis_gets_none_by_default(monkeypatch):
    cfg = _advanced(monkeypatch, dict(BASE), {})
    assert cfg["scf_settings"]["histdiis"] is None


def test_parent_histdiis_is_the_default(monkeypatch):
    opts = dict(BASE, scf_settings={**BASE["scf_settings"], "histdiis": 40})
    cfg = _advanced(monkeypatch, opts, {})
    assert cfg["scf_settings"]["histdiis"] == 40


def test_no_to_histdiis_is_final(monkeypatch):
    opts = dict(BASE, scf_settings={**BASE["scf_settings"], "histdiis": 100})
    cfg = _advanced(monkeypatch, opts, {"HISTDIIS (keep": "n"})
    assert cfg["scf_settings"]["histdiis"] is None


def _scf_tail(**kw):
    buf = io.StringIO()
    write_scf_section(buf, {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, None, "MOLECULE",
                      False, 0.0, "DIIS", 800, 30, 2, **kw)
    return buf.getvalue().splitlines()


def test_writer_histdiis_none_writes_no_record():
    assert "HISTDIIS" not in _scf_tail(histdiis=None)


def test_writer_keeps_its_default_histdiis_for_new_decks():
    lines = _scf_tail()
    assert lines[lines.index("HISTDIIS") + 1] == "100"


# ------------------------------------------------------------------ 3c grid

def _dft(functional, grid):
    buf = io.StringIO()
    write_dft_section(buf, functional, False, grid, False)
    return buf.getvalue().splitlines()


def test_3c_grid_is_written():
    assert _dft("PBEH3C", "XLGRID") == ["DFT", "PBEH3C", "XLGRID", "ENDDFT"]


def test_hsesol3c_block_is_unchanged():
    assert _dft("HSESOL3C", "XLGRID") == ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]
    assert _dft("HSESOL3C", None) == ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]
    assert _dft("HSESOL3C", "XXLGRID") == ["DFT", "HSESOL3C", "XLGRID", "ENDDFT"]


def test_3c_grid_prompt_keeps_the_parent_grid():
    assert d12_interactive.configure_dft_grid_with_defaults("PBEH3C", "XLGRID") == "XLGRID"
    assert d12_interactive.configure_dft_grid_with_defaults("PBEH3C", None) is None


# ------------------------------------------------------------ parser records

def _parse(tmp_path, scf_lines):
    from d12_parsers import CrystalInputParser
    deck = tmp_path / "p.d12"
    deck.write_text("\n".join(["t", "MOLECULE", "1", "1", "6 0 0 0", "END",
                               "BASISSET", "POB-TZVP", "DFT", "PBE0", "ENDDFT",
                               *scf_lines, "PPAN", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()["scf_settings"]


def test_parser_reads_histdiis_and_guessp(tmp_path):
    scf = _parse(tmp_path, ["GUESSP", "DIIS", "HISTDIIS", "40"])
    assert scf["histdiis"] == 40 and scf["guessp"] is True


def test_parser_records_a_missing_histdiis(tmp_path):
    scf = _parse(tmp_path, ["DIIS"])
    assert "histdiis" in scf and scf["histdiis"] is None
    assert not scf.get("guessp")


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


NO_HISTDIIS_PARENT = "OPT/1_dia_opt_BULK_OPTGEOM"             # DIIS, no HISTDIIS
GUESSP_PARENT = "SP/4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT_graphene"


def test_planless_sp_keeps_a_missing_histdiis_missing(tmp_path):
    name = _copy_parent(NO_HISTDIIS_PARENT, tmp_path)
    child = _opt2d12(tmp_path, name, stdin="n\n" + "\n" * 19,
                     extra=["--non-interactive", "--calc-type", "SP"])
    assert "DIIS" in child and "HISTDIIS" not in child


def test_config_sp_keeps_a_missing_histdiis_missing(tmp_path):
    name = _copy_parent(NO_HISTDIIS_PARENT, tmp_path)
    cfg = tmp_path / "sp.json"
    cfg.write_text(json.dumps({"calculation_type": "SP"}))
    child = _opt2d12(tmp_path, name, stdin="",
                     extra=["--config-file", str(cfg), "--non-interactive"])
    assert "DIIS" in child and "HISTDIIS" not in child


def test_planless_sp_keeps_the_parents_guessp(tmp_path):
    name = _copy_parent(GUESSP_PARENT, tmp_path)
    child = _opt2d12(tmp_path, name, stdin="n\n" + "\n" * 19,
                     extra=["--non-interactive", "--calc-type", "SP"])
    assert "GUESSP" in child
    assert "HSESOL3C" in child
