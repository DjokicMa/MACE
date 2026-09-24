"""A parent functional MACE cannot identify is no longer swapped for HSE06 silently.

B2PLYP is not a CRYSTAL23 functional keyword (it is not in the CRYSTAL23
manual). The parsers used to leave the functional unset, and every path then
wrote HSE06 without saying so. Now:

* the parsers record the parent's own keyword ("unrecognised_functional");
* a valid CRYSTAL23 keyword MACE's menus do not list (e.g. SOGGA11), or a
  different spelling of one they do (CRYSTAL input is not case sensitive), is
  kept as written and not asked about;
* the interactive flow names the parent's functional and asks for another one
  (validated against the CRYSTAL23 keyword list, with the menus on 'm');
  Enter uses HSE06 with a warning;
* the paths that must not ask (--non-interactive, "use these exact settings",
  a config file) warn and use HSE06;
* the engine's planless stdin (blank answers) still completes.
"""
import shutil
import subprocess
import sys

import pytest

import d12_interactive
import menu_nav
from conftest import REPO_ROOT, TEST_DATA
from d12_constants import crystal23_functional_keyword, mace_functional_name
from d12_parsers import CrystalInputParser, CrystalOutputParser

MACE_CLI = REPO_ROOT / "mace_cli"
WARNING = "WARNING: The parent's functional 'B2PLYP' is not a CRYSTAL23 functional keyword."


# ------------------------------------------------------------ keyword list

@pytest.mark.parametrize("name", ["B2PLYP", "B2PLYP-D3", "B2GPPLYP", "DSD-PBEP86", "PBESOL0-D3", ""])
def test_not_crystal23_keywords(name):
    assert crystal23_functional_keyword(name) is None


@pytest.mark.parametrize("name, keyword", [
    ("HSE06", "HSE06"), ("hsesol", "HSEsol"), ("SOGGA11", "SOGGA11"),
    ("pw1pw-d3", "PW1PW-D3"), ("HSESOL3C", "HSESOL3C"), ("r2scan0", "r2SCAN0"),
    ("LSRSH-PBE", "LSRSH-PBE"), ("B3PW", "B3PW"),
])
def test_crystal23_keywords_match_without_case(name, keyword):
    assert crystal23_functional_keyword(name) == keyword


def test_hf_methods_only_when_asked_for():
    assert crystal23_functional_keyword("UHF") is None
    assert crystal23_functional_keyword("uhf", allow_hf=True) == "UHF"


def test_mace_names_ignore_case_and_keep_d3():
    assert mace_functional_name("hsesol") == "HSEsol"
    assert mace_functional_name("pbe0-d3") == "PBE0-D3"
    assert mace_functional_name("SOGGA11") is None


# ----------------------------------------------------------------- parsers

def _dft_block(tmp_path, *dft_lines):
    deck = tmp_path / "p.d12"
    deck.write_text("\n".join(["t", "MOLECULE", "1", "1", "6 0 0 0", "END",
                               "BASISSET", "POB-TZVP", "DFT", *dft_lines,
                               "END", "TOLDEE", "7", "SHRINK", "8 8", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()


def test_input_parser_records_a_functional_crystal23_does_not_know(tmp_path):
    data = _dft_block(tmp_path, "SPIN", "B2PLYP", "XLGRID")
    assert data.get("functional") is None
    assert data["unrecognised_functional"] == "B2PLYP"
    assert data["unrecognised_functional_source"] == "input"
    assert data["spin_polarized"] is True and data["dft_grid"] == "XLGRID"


def test_input_parser_records_an_unknown_d3_functional(tmp_path):
    data = _dft_block(tmp_path, "B2PLYP-D3")
    assert data.get("functional") is None
    assert data["unrecognised_functional"] == "B2PLYP-D3"
    assert data["dispersion"] is True


def test_input_parser_keeps_an_unlisted_crystal23_keyword_as_written(tmp_path):
    data = _dft_block(tmp_path, "SOGGA11", "XLGRID")
    assert data["functional"] == "SOGGA11"
    assert "unrecognised_functional" not in data


def test_input_parser_reads_another_spelling_of_a_listed_functional(tmp_path):
    data = _dft_block(tmp_path, "HSESOL")
    assert data["functional"] == "HSEsol"
    assert "unrecognised_functional" not in data


def test_input_parser_skips_grid_and_weight_keywords(tmp_path):
    data = _dft_block(tmp_path, "SAVIN", "RADIAL", "1", "4.0", "ANGULAR", "1", "9999.0",
                      "TOLLGRID", "14", "B2PLYP")
    assert data["unrecognised_functional"] == "B2PLYP"


@pytest.mark.parametrize("line, functional", [
    ("B3LYP-D3", "B3LYP-D3"), ("PBE0", "PBE0"), ("PBESOLXC", "PBESOL"),
])
def test_input_parser_known_functionals_unchanged(tmp_path, line, functional):
    data = _dft_block(tmp_path, line)
    assert data["functional"] == functional
    assert "unrecognised_functional" not in data


def test_output_parser_records_what_crystal_printed(tmp_path):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    text = src.read_text(errors="replace").replace(
        "FUNCTIONAL:(BECKE 88)[LEE-YANG-PARR]", "FUNCTIONAL:(MADE UP)[NOT REAL]")
    out = tmp_path / "p.out"
    out.write_text(text)
    data = CrystalOutputParser(str(out)).parse()
    assert data.get("functional") is None
    assert data["unrecognised_functional"] == "(MADE UP)[NOT REAL]"
    assert data["unrecognised_functional_source"] == "output"


# ------------------------------------------------ interactive functional step

def _answer(monkeypatch, answers):
    seen = []

    def fake_input(prompt=""):
        seen.append(prompt)
        for sub, reply in answers:
            if sub in prompt:
                if isinstance(reply, list):
                    return reply.pop(0) if reply else ""
                return reply
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", fake_input)
    return seen


def _unknown(**extra):
    return {"functional": None, "method": "DFT", "unrecognised_functional": "B2PLYP",
            "unrecognised_functional_source": "input", "dispersion": False, **extra}


def test_blank_answer_uses_hse06_with_a_warning(monkeypatch, capsys):
    seen = _answer(monkeypatch, [])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "HSE06"
    assert "unrecognised_functional" not in opts
    out = capsys.readouterr().out
    assert "The parent's functional 'B2PLYP' is not a CRYSTAL23 functional keyword." in out
    assert WARNING in out and "Using HSE06 instead." in out
    assert any("press Enter to use HSE06" in p for p in seen)


def test_blank_answer_keeps_the_parents_d3(monkeypatch):
    _answer(monkeypatch, [])
    opts = d12_interactive.configure_method(_unknown(unrecognised_functional="B2PLYP-D3",
                                                     dispersion=True))
    assert opts["functional"] == "HSE06-D3" and opts["dispersion"] is True


def test_typed_functional_is_used(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", "pbe0"), ("Add D3", "n")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "PBE0" and opts["dispersion"] is False


def test_typed_unlisted_crystal23_keyword_is_used(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", "SOGGA11")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "SOGGA11"


def test_typed_hf_method_switches_to_hf(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", "UHF")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "UHF" and opts["method_type"] == "HF"


def test_invalid_answer_is_asked_again(monkeypatch, capsys):
    seen = _answer(monkeypatch, [("Enter another functional", ["B2GPPLYP", "HSEsol-D3"])])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "HSEsol-D3" and opts["dispersion"] is True
    assert sum("Enter another functional" in p for p in seen) == 2
    assert "'B2GPPLYP' is not a CRYSTAL23 functional keyword" in capsys.readouterr().out


def test_m_opens_the_menus(monkeypatch):
    # Menu: category 2 (GGA), functional 2 (PBE).
    _answer(monkeypatch, [("Enter another functional", "m"),
                          ("Select functional category", "2"),
                          ("Select GGA functional", "2"), ("Add D3", "n")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "PBE"


def test_unlisted_crystal23_parent_functional_is_the_menu_default(monkeypatch):
    seen = _answer(monkeypatch, [])
    opts = d12_interactive.configure_method({"functional": "SOGGA11", "method": "DFT"})
    assert opts["functional"] == "SOGGA11"
    assert not any("Enter another functional" in p for p in seen)


def test_listed_parent_functional_menu_unchanged(monkeypatch):
    seen = _answer(monkeypatch, [])
    opts = d12_interactive.configure_method({"functional": "PBE0", "method": "DFT",
                                             "dispersion": False})
    assert opts["functional"] == "PBE0"
    assert not any("Enter another functional" in p for p in seen)


def test_ensure_known_functional_warns_and_uses_hse06(capsys):
    opts = d12_interactive.ensure_known_functional(_unknown())
    assert opts["functional"] == "HSE06" and "unrecognised_functional" not in opts
    assert WARNING in capsys.readouterr().out


def test_ensure_known_functional_leaves_a_known_functional_alone(capsys):
    opts = d12_interactive.ensure_known_functional(
        {"functional": "PBE0", "unrecognised_functional": "(X)[Y]"})
    assert opts["functional"] == "PBE0" and "unrecognised_functional" not in opts
    assert "WARNING" not in capsys.readouterr().out


# --------------------------------------------------- real opt2d12 run paths

@pytest.fixture
def b2plyp_parent(tmp_path):
    """The diamond B3LYP-D3 corpus parent, re-labelled B2PLYP.

    The deck's functional line becomes B2PLYP (no -D3), and the output's
    functional line is removed so nothing identifies the functional.
    """
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    deck = src.with_suffix(".d12").read_text().splitlines()
    deck = ["B2PLYP" if line.strip() == "B3LYP-D3" else line for line in deck]
    (tmp_path / "parent.d12").write_text("\n".join(deck) + "\n")
    out = [line for line in src.read_text(errors="replace").splitlines(keepends=True)
           if "(EXCHANGE)[CORRELATION] FUNCTIONAL" not in line
           and "GRIMME" not in line and "DFT-D3" not in line]
    (tmp_path / "parent.out").write_text("".join(out))
    return tmp_path


def _run(cwd, *extra, stdin=""):
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
           "--d12-file", "parent.d12", *extra]
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True,
                            text=True, timeout=300)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-1500:]
    decks = [p for p in cwd.glob("*.d12") if p.name != "parent.d12"]
    assert len(decks) == 1, (sorted(p.name for p in cwd.iterdir()), log[-1500:])
    lines = decks[0].read_text().splitlines()
    dft = lines[lines.index("DFT") + 1:lines.index("ENDDFT")]
    return dft, log


def test_opt2d12_blank_answers_prompt_then_hse06(b2plyp_parent):
    dft, log = _run(b2plyp_parent, stdin="n\n1\n" + "\n" * 60)
    assert "Enter another functional, 'm' to choose from the menu, or press Enter to use HSE06" in log
    assert WARNING in log
    assert "HSE06" in dft and "B2PLYP" not in dft


def test_opt2d12_typed_functional(b2plyp_parent):
    # n (modify), 1 (SP), blank (method type: DFT), PBE0, n (no D3)
    dft, log = _run(b2plyp_parent, stdin="n\n1\n\nPBE0\nn\n" + "\n" * 60)
    assert "PBE0" in dft and WARNING not in log


@pytest.mark.parametrize("extra, stdin", [
    (("--non-interactive",), ""),
    (("--non-interactive", "--calc-type", "SP"), ""),
    ((), "y\n1\n" + "\n" * 20),  # "use these exact settings"
])
def test_non_interactive_paths_warn_and_use_hse06(b2plyp_parent, extra, stdin):
    dft, log = _run(b2plyp_parent, *extra, stdin=stdin)
    assert WARNING in log and "Using HSE06 instead." in log
    assert "HSE06" in dft
    assert "Enter another functional" not in log


def test_config_file_without_functional_warns_and_uses_hse06(b2plyp_parent):
    (b2plyp_parent / "cfg.json").write_text('{"calculation_type": "SP"}')
    dft, log = _run(b2plyp_parent, "--config-file", "cfg.json", "--non-interactive")
    assert WARNING in log and "HSE06" in dft


def test_config_file_functional_wins_without_a_warning(b2plyp_parent):
    (b2plyp_parent / "cfg.json").write_text('{"calculation_type": "SP", "functional": "PBE0"}')
    dft, log = _run(b2plyp_parent, "--config-file", "cfg.json", "--non-interactive")
    assert "PBE0" in dft and "WARNING: The parent's functional" not in log


def test_engine_planless_stdin_completes(b2plyp_parent):
    # The engine's planless SP step: these args, "n" then blank answers.
    dft, log = _run(b2plyp_parent, "--non-interactive", "--calc-type", "SP",
                    stdin="n\n" + "\n" * 39)
    assert "EOFError" not in log
    assert WARNING in log and "HSE06" in dft
