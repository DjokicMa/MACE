"""A parent's functional records and dispersion input are kept; the title is not a keyword.

Each case here was a child deck that differed from its parent:

* The DFT scans read the title line (line 1) as a keyword: a title "DFT"
  lost a user-defined hybrid (the child ran bare PBE), and a title "DFTD3"
  swallowed the geometry into a "D3 block".
* Records that modify a functional (SR-OMEGA, MR-OMEGA, LR-OMEGA, SR-HYB)
  were dropped, and LSRSH-PBE lost its required omega/cSR/cLR record.
* A DFTD3 block after a functional keyword (the manual's own "PBE0 / END /
  DFTD3 ... END") was dropped on every path, and a GRIMME record renamed the
  functional "<name>-D3".
* A menu mPW1PW91 with D3 was written "mPW1PW91-D3", which CRYSTAL23 does
  not recognise; its D3 keyword is PW1PW-D3.
"""
import io
import subprocess
import sys

import pytest

import d12_interactive
import menu_nav
from conftest import REPO_ROOT, TEST_DATA
from d12_constants import CUSTOM_FUNCTIONAL
from d12_parsers import CrystalInputParser
from d12_writer import write_dft_section

MACE_CLI = REPO_ROOT / "mace_cli"
PBE0_RECORDS = ["EXCHANGE", "PBE", "CORRELAT", "PBE", "HYBRID", "25"]
D3_BLOCK = ["DFTD3", "VERSION", "4", "S6", "1.0000", "S8", "1.9889", "END"]


def _parse(tmp_path, dft_lines, title="t", after=(), opt=(), basis=("BASISSET", "POB-TZVP")):
    deck = tmp_path / "p.d12"
    geometry = ["CRYSTAL", "0 0 0", "227", "3.567", "1", "6 0.125 0.125 0.125"]
    optgeom = ["OPTGEOM", *opt, "ENDOPT"] if opt else []
    deck.write_text("\n".join([title, *geometry, *optgeom, "END", *basis, "DFT",
                               *dft_lines, "ENDDFT", *after, "TOLDEE", "7",
                               "SHRINK", "8 8", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()


def _written(functional, use_dispersion=False, dft_grid="DEFAULT", **kw):
    buf = io.StringIO()
    write_dft_section(buf, functional, use_dispersion, dft_grid, False, **kw)
    return buf.getvalue().split("\n")[:-1]


# ------------------------------------------------------------- the title

@pytest.mark.parametrize("title", ["DFT", "DFTD3", "ENDDFT", "EXCHANGE"])
def test_title_does_not_change_a_user_defined_hybrid(tmp_path, title):
    data = _parse(tmp_path, ["SPIN", *PBE0_RECORDS, "XLGRID"], title=title)
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_functional"] == PBE0_RECORDS
    assert data.get("custom_dftd3") is None
    assert data["dft_grid"] == "XLGRID"


def test_title_dftd3_does_not_swallow_the_geometry(tmp_path):
    data = _parse(tmp_path, ["B3PW", "XLGRID"], title="DFTD3")
    assert data["functional"] == "B3PW"
    assert data.get("custom_dftd3") is None and not data.get("dispersion")


def test_title_uhf_is_not_the_method(tmp_path):
    data = _parse(tmp_path, ["PBE0"], title="UHF")
    assert data["method"] == "DFT" and data["functional"] == "PBE0"


# ------------------------------------------------ functional modifier records

@pytest.mark.parametrize("records", [
    ["HSE06", "HYBRID", "30", "SR-OMEGA", "0.2"],
    ["HSE06", "SR-OMEGA", "0.2"],
    ["WB97X", "SR-HYB", "0.2"],
    ["HISS", "MR-OMEGA", "0.84 0.20"],
    ["LC-wPBE", "LR-OMEGA", "0.3"],
    ["LSRSH-PBE", "0.11 0.25 0.00001"],
])
def test_modifier_and_value_records_are_kept(tmp_path, records):
    data = _parse(tmp_path, ["SPIN", *records, "XLGRID"])
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_functional"] == records
    assert _written(CUSTOM_FUNCTIONAL, custom_functional=records,
                    dft_grid="XLGRID")[1:-1] == [*records, "XLGRID"]


# ----------------------------------------------- a DFTD3 block with a keyword

def test_keyword_functional_keeps_its_dftd3_block(tmp_path):
    data = _parse(tmp_path, ["SPIN", "PBE0", "XLGRID"], after=D3_BLOCK)
    assert data["functional"] == "PBE0" and data["dispersion"] is True
    assert data["custom_dftd3"] == D3_BLOCK
    assert data["custom_dftd3_functional"] == "PBE0"
    assert _written("PBE0", True, "XLGRID", custom_dftd3=D3_BLOCK) == [
        "DFT", "PBE0", "XLGRID", "ENDDFT", *D3_BLOCK]
    # The D3 question's "<name>-D3" answer, for the parent's functional
    assert _written("PBE0-D3", True, "XLGRID", custom_dftd3=D3_BLOCK) == [
        "DFT", "PBE0", "XLGRID", "ENDDFT", *D3_BLOCK]


def test_grimme_records_are_kept(tmp_path):
    grimme = ["GRIMME", "1.05 20. 25.", "1", "6 1.75 1.452"]
    data = _parse(tmp_path, ["B3LYP"], after=grimme)
    assert data["functional"] == "B3LYP"
    assert data["custom_dftd3"] == grimme


def test_dftd3_block_goes_only_with_the_parents_functional():
    from CRYSTALOptToD12 import parent_dispersion_input
    settings = {"custom_dftd3": D3_BLOCK, "custom_dftd3_functional": "PBE0",
                "dispersion": True}
    assert parent_dispersion_input({**settings, "functional": "PBE0"}) == D3_BLOCK
    assert parent_dispersion_input({**settings, "functional": "PBE0-D3"}) == D3_BLOCK
    assert parent_dispersion_input({**settings, "functional": "B3LYP-D3"}) is None
    assert parent_dispersion_input({**settings, "functional": "PBE0",
                                    "dispersion": False}) is None


def test_mpw1pw91_d3_menu_name_is_written_pw1pw_d3():
    assert _written("mPW1PW91-D3", True)[1:-1] == ["PW1PW-D3"]
    assert _written("mPW1PW91-D3", False)[1:-1] == ["PW1PW-D3"]


# ------------------------------------------------------------- interactive

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


def _unknown():
    return {"functional": None, "method": "DFT", "unrecognised_functional": "B2PLYP",
            "unrecognised_functional_source": "input", "dispersion": False}


def test_replacement_mpw1pw91_d3_is_accepted(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", ["mPW1PW91-D3"])])
    opts = d12_interactive.configure_method(_unknown())
    assert _written(opts["functional"], opts["dispersion"])[1:-1] == ["PW1PW-D3"]


def test_menu_mpw1pw91_with_d3_is_written_pw1pw_d3(monkeypatch):
    # Hybrid category, mPW1PW91 (9th), D3 yes
    _answer(monkeypatch, [("Select functional category", "3"),
                          ("Select Hybrid functional", "9"), ("Add D3", "y")])
    opts = d12_interactive.configure_method({"functional": "PBE0", "method": "DFT",
                                             "dispersion": False})
    assert _written(opts["functional"], opts["dispersion"])[1:-1] == ["PW1PW-D3"]


# --------------------------------------------------- real opt2d12 run paths

def _parent(tmp_path, dft_lines, after=()):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    deck = []
    for line in src.with_suffix(".d12").read_text().splitlines():
        if line.strip() == "B3LYP-D3":
            deck += dft_lines
        elif line.strip() == "ENDDFT":
            deck += [line, *after]
        else:
            deck.append(line)
    (tmp_path / "parent.d12").write_text("\n".join(deck) + "\n")
    (tmp_path / "parent.out").write_text(src.read_text(errors="replace"))
    return tmp_path


def _run(cwd, *extra, stdin=""):
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
           "--d12-file", "parent.d12", *extra]
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True,
                            text=True, timeout=300)
    log = result.stdout + result.stderr
    decks = [p for p in cwd.glob("*.d12") if p.name != "parent.d12"]
    assert result.returncode == 0, log[-1500:]
    assert len(decks) == 1, (sorted(p.name for p in cwd.iterdir()), log[-1500:])
    return decks[0].read_text().splitlines(), log


PATHS = [
    (("--non-interactive",), ""),
    (("--non-interactive", "--calc-type", "SP"), "n\n" + "\n" * 39),  # engine planless
    ((), "n\n1\n" + "\n" * 60),  # interactive, blank answers
]


@pytest.mark.parametrize("extra, stdin", PATHS)
def test_opt2d12_keeps_a_keyword_functionals_dftd3_block(tmp_path, extra, stdin):
    lines, _ = _run(_parent(tmp_path, ["PBE0"], after=D3_BLOCK), *extra, stdin=stdin)
    dft = lines.index("DFT")
    end = lines.index("ENDDFT")
    assert [r for r in lines[dft + 1:end] if r != "SPIN"] == ["PBE0", "XLGRID"]
    assert lines[end + 1:end + 1 + len(D3_BLOCK)] == D3_BLOCK
    assert "PBE0-D3" not in lines
