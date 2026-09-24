"""Parent-deck parsing and option round trips that changed the child deck.

* Keywords were matched as substrings of any line, the title included. A
  single-point parent titled "..._BULK_OPTGEOM_TZ_..." was read as an OPT deck,
  and its SCF MAXCYCLE 1600 / TOLDEE 9 as OPTGEOM values, which then became
  the defaults of a child optimization. A title containing SHRINK or a
  geometry keyword misread the k-mesh or the dimensionality the same way.
* ITATOCEL and INTREDUN were not read as optimization types, so a parent's
  ITATOCEL became FULLOPTG, and --opt-type was ignored for OPT parents.
* "No" to MAXTRADIUS was lost when the saved options were reused with
  --config-file: the parent's MAXTRADIUS came back.

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
from d12_parsers import CrystalInputParser, CrystalOutputParser

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


def _parse(tmp_path, text):
    deck = tmp_path / "parent.d12"
    deck.write_text(textwrap.dedent(text).lstrip("\n"))
    return CrystalInputParser(str(deck)).parse()


# A single-point deck whose title (a file name) names keywords it does not use.
SP_DECK = """
    x_BULK_OPTGEOM_TZ_SHRINK_SLAB_opt_sp
    CRYSTAL
    0 0 0
    227
    3.567
    1
    6 0.0 0.0 0.0
    END
    BASISSET
    POB-TZVP-REV2
    DFT
    B3LYP-D3
    XLGRID
    ENDDFT
    TOLINTEG
    8 8 8 9 24
    TOLDEE
    9
    SHRINK
    8 16
    MAXCYCLE
    1600
    FMIXING
    50
    END
"""


def _opt_deck(opt_type):
    return f"""
    dia_{opt_type}
    CRYSTAL
    0 0 0
    227
    3.567
    1
    6 0.0 0.0 0.0
    OPTGEOM
    {opt_type}
    MAXCYCLE
    800
    TOLDEG
    0.0001
    ENDOPT
    END
    BASISSET
    POB-TZVP-REV2
    DFT
    B3LYP-D3
    ENDDFT
    SHRINK
    8 16
    END
"""


# ---------------------------------------------------- keywords in the title

def test_optgeom_in_the_title_is_not_an_optimization(tmp_path):
    data = _parse(tmp_path, SP_DECK)
    assert data.get("calculation_type") != "OPT"
    assert data["optimization_settings"] == {}


def test_a_real_optgeom_block_is_still_read(tmp_path):
    data = _parse(tmp_path, _opt_deck("FULLOPTG"))
    assert data["calculation_type"] == "OPT"
    assert data["optimization_settings"] == {"type": "FULLOPTG", "MAXCYCLE": 800, "TOLDEG": 0.0001}


def test_shrink_and_geometry_keywords_in_the_title_are_ignored(tmp_path):
    data = _parse(tmp_path, SP_DECK)
    assert data["k_points"] == "8 16"
    assert data["dimensionality"] == "CRYSTAL"
    assert data["spacegroup"] == 227


def test_output_title_with_spin_is_not_spin_polarized(tmp_path):
    parser = CrystalOutputParser(str(tmp_path / "unused.out"))
    parser.data.setdefault("scf_settings", {})
    parser._extract_settings([
        " ./x_SPIN_opt_sp",
        " TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL",
    ])
    assert parser.data["spin_polarized"] is False
    parser._extract_settings([
        " ./x_opt_sp",
        " TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL",
    ])
    assert parser.data["spin_polarized"] is True


def test_settings_extractor_ignores_the_title(tmp_path):
    from mace.utils.settings_extractor import extract_input_settings

    deck = tmp_path / "parent.d12"
    deck.write_text(textwrap.dedent(SP_DECK.replace("B3LYP-D3", "PBE0")).lstrip("\n")
                    .replace("x_BULK", "x_HSE06_BULK", 1))
    settings = extract_input_settings(deck)
    assert "OPTGEOM" not in settings["crystal_keywords"]
    assert settings["optimization_parameters"] == {}
    assert settings["geometry_info"]["dimensionality"] == "3D"
    assert settings["functional_info"]["exchange"] == "PBE0"


# ------------------------------------------------------- optimization type

@pytest.mark.parametrize("opt_type", ["ITATOCEL", "INTREDUN", "CELLONLY", "CVOLOPT"])
def test_parent_optimization_type_is_read(tmp_path, opt_type):
    data = _parse(tmp_path, _opt_deck(opt_type))
    assert data["optimization_settings"]["type"] == opt_type


@pytest.mark.parametrize("opt_type", ["ITATOCEL", "INTREDUN"])
def test_parent_optimization_type_is_the_menu_default(monkeypatch, opt_type):
    _blank_input(monkeypatch)
    cfg = d12_calc_basic._configure_optimization_impl({"type": opt_type, "MAXCYCLE": 800})
    assert cfg["type"] == opt_type


def test_every_menu_type_is_read_back():
    from d12_constants import OPTGEOM_TYPE_KEYWORDS
    assert set(d12_calc_basic.OPT_TYPES.values()) <= set(OPTGEOM_TYPE_KEYWORDS)


def test_non_interactive_keeps_the_parent_type_and_honours_opt_type():
    from CRYSTALOptToD12 import _keep_extracted_settings

    parent = {"optimization_settings": {"type": "ITATOCEL", "MAXCYCLE": 800},
              "spacegroup": 227, "functional": "B3LYP"}
    kept = _keep_extracted_settings(dict(parent), "OPT", None, "auto")
    assert kept["optimization_type"] == "ITATOCEL"
    changed = _keep_extracted_settings(dict(parent), "OPT", "ATOMONLY", "auto")
    assert changed["optimization_settings"]["type"] == "ATOMONLY"
    assert parent["optimization_settings"]["type"] == "ITATOCEL"   # not mutated


# --------------------------------------------------------------- MAXTRADIUS

def test_no_to_maxtradius_is_recorded(monkeypatch):
    _blank_input(monkeypatch, {"MAXTRADIUS) for geometry": "n"})
    cfg = d12_calc_basic._configure_optimization_impl(
        {"type": "FULLOPTG", "MAXCYCLE": 800, "MAXTRADIUS": 0.25})
    assert "maxtradius" in cfg and cfg["maxtradius"] is None


def test_recorded_no_overrides_the_parent_maxtradius():
    from CRYSTALOptToD12 import merge_optimization_settings

    merged = merge_optimization_settings({"type": "FULLOPTG", "MAXTRADIUS": 0.25},
                                         {"maxtradius": None})
    assert merged.get("maxtradius") is None and "MAXTRADIUS" not in merged


# ------------------------------------------------- real opt2d12 on the corpus

SP_TITLED_OPTGEOM = "SP/3,4^2T7_CA_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1_sp_B3LYP-D3-D3_optimized"
MAXTRADIUS_PARENT = "OPT/1_dia_opt_rev1"                      # FULLOPTG, MAXTRADIUS 0.25


def _copy_parent(stem, tmp_path):
    src = TEST_DATA / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return src.stem


def _opt2d12(tmp_path, name, answers, extra=()):
    """Run the real mace_cli opt2d12, answering prompts by substring."""
    args = ["opt2d12", "--out-file", f"{name}.out", "--d12-file", f"{name}.d12", *extra]
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
    result = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, input="",
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _optgeom(lines):
    return lines[lines.index("OPTGEOM"):lines.index("ENDOPT")]


def test_opt_child_of_an_sp_parent_titled_optgeom_gets_standard_optgeom(tmp_path):
    name = _copy_parent(SP_TITLED_OPTGEOM, tmp_path)
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "2"})
    opt = _optgeom(child)
    assert opt[opt.index("MAXCYCLE") + 1] == "800"         # not the SCF's 1600
    assert float(opt[opt.index("TOLDEG") + 1]) == 0.0003
    assert opt[opt.index("TOLDEE") + 1] == "7"             # not the SCF's 9


def test_itatocel_parent_gives_an_itatocel_child(tmp_path):
    name = _copy_parent(MAXTRADIUS_PARENT, tmp_path)
    deck = tmp_path / f"{name}.d12"
    deck.write_text(deck.read_text().replace("\nFULLOPTG\n", "\nITATOCEL\n"))
    child = _opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "2"})
    assert _optgeom(child)[1] == "ITATOCEL"


def test_no_to_maxtradius_survives_save_options_and_config_file(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    name = _copy_parent(MAXTRADIUS_PARENT, first)
    _copy_parent(MAXTRADIUS_PARENT, second)
    child = _opt2d12(first, name, {"exact settings": "n", "calculation type": "2",
                                   "MAXTRADIUS) for geometry": "n"},
                     extra=["--save-options", "--options-file", "opts.json"])
    assert "MAXTRADIUS" not in child
    shutil.copy(first / "opts.json", second / "opts.json")
    reused = _opt2d12(second, name, {}, extra=["--config-file", "opts.json"])
    assert "MAXTRADIUS" not in reused
    assert _optgeom(reused) == _optgeom(child)
