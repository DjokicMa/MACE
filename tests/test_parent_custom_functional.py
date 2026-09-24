"""A parent's functional is kept exactly, or asked about; never downgraded.

* A DFT block that defines its functional with EXCHANGE/CORRELAT/HYBRID/
  NONLOCAL records (CRYSTAL23 manual sec. 4.1, "User-defined global hybrid
  functionals") was read as its exchange name alone, so EXCHANGE PBE /
  CORRELAT PBE / HYBRID 25 (PBE0) became the GGA "PBE", and a DFTD3 block
  turned it into "PBE-D3". The records, and the parent's DFTD3 block, are now
  written back verbatim, and keeping them is the menu default.
* The menu's "Keep the parent's functional" is offered only for a CRYSTAL23
  keyword or such a custom definition; any other name (LDA, BECKE, mPW91 are
  EXCHANGE records, not functionals) goes to the unknown-functional prompt.
* The replacement prompt accepts only what the writer turns into valid
  CRYSTAL23 input: bare PBE and B97 (written as typed) and -D3 forms the
  manual has no D3 parameters for (SCAN-D3, PBESOL0-D3) are asked again.
* A .d12 functional line nothing could map is asked about, not replaced by
  the output parser's guess.
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


def _dft_block(tmp_path, *dft_lines, after=()):
    deck = tmp_path / "p.d12"
    deck.write_text("\n".join(["t", "MOLECULE", "1", "1", "6 0 0 0", "END",
                               "BASISSET", "POB-TZVP", "DFT", *dft_lines,
                               "ENDDFT", *after, "TOLDEE", "7", "SHRINK", "8 8",
                               "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()


def _written(**kw):
    buf = io.StringIO()
    args = dict(functional=kw.pop("functional"), use_dispersion=kw.pop("use_dispersion", False),
                dft_grid=kw.pop("dft_grid", "XLGRID"), is_spin_polarized=False)
    write_dft_section(buf, **args, **kw)
    return buf.getvalue().split("\n")[:-1]


# ------------------------------------------------------------------ parser

def test_user_defined_hybrid_is_kept_record_by_record(tmp_path):
    data = _dft_block(tmp_path, "SPIN", *PBE0_RECORDS, "XLGRID")
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_functional"] == PBE0_RECORDS
    assert data["dft_grid"] == "XLGRID" and data["spin_polarized"] is True


def test_nonlocal_and_a_modified_keyword_are_kept(tmp_path):
    b3pw = ["EXCHANGE", "BECKE", "CORRELAT", "PWGGA", "HYBRID", "20",
            "NONLOCAL", "0.9 0.81"]
    assert _dft_block(tmp_path, *b3pw)["custom_functional"] == b3pw
    data = _dft_block(tmp_path, "B3LYP", "HYBRID", "30")
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_functional"] == ["B3LYP", "HYBRID", "30"]


def test_gga_pair_is_kept_not_turned_into_bare_pbe(tmp_path):
    data = _dft_block(tmp_path, "EXCHANGE", "PBE", "CORRELAT", "PBE")
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_functional"] == ["EXCHANGE", "PBE", "CORRELAT", "PBE"]


@pytest.mark.parametrize("pair, name", [
    (("PWGGA", "PWGGA"), "PWGGA"), (("VBH", "VBH"), "VBH"), (("WCGGA", "PWGGA"), "WCGGA"),
])
def test_writers_own_pairs_still_read_back_as_menu_names(tmp_path, pair, name):
    data = _dft_block(tmp_path, "EXCHANGE", pair[0], "CORRELAT", pair[1])
    assert data["functional"] == name
    assert "custom_functional" not in data
    assert _written(functional=name, dft_grid="DEFAULT")[1:-1] == [
        "EXCHANGE", pair[0], "CORRELAT", pair[1]]


def test_dftd3_block_inside_the_dft_block_is_kept(tmp_path):
    data = _dft_block(tmp_path, *PBE0_RECORDS, "DFTD3", "VERSION", "4", "END", "XLGRID")
    assert data["functional"] == CUSTOM_FUNCTIONAL
    assert data["custom_dftd3"] == ["DFTD3", "VERSION", "4", "END"]
    assert data["custom_dftd3_in_dft"] is True and data["dispersion"] is True


def test_dftd3_block_in_the_hamiltonian_block_is_kept(tmp_path):
    data = _dft_block(tmp_path, *PBE0_RECORDS,
                      after=("DFTD3", "VERSION", "4", "S6", "1.0", "END"))
    assert data["custom_dftd3"] == ["DFTD3", "VERSION", "4", "S6", "1.0", "END"]
    assert data["custom_dftd3_in_dft"] is False and data["dispersion"] is True


def test_standalone_keywords_unchanged(tmp_path):
    data = _dft_block(tmp_path, "PBE0", "XLGRID")
    assert data["functional"] == "PBE0" and "custom_functional" not in data


# ------------------------------------------------------------------ writer

def test_writer_repeats_the_records_and_d3_block_where_they_were():
    inside = _written(functional=CUSTOM_FUNCTIONAL, use_dispersion=True,
                      custom_functional=PBE0_RECORDS,
                      custom_dftd3=["DFTD3", "VERSION", "4", "END"], custom_dftd3_in_dft=True)
    assert inside == ["DFT", *PBE0_RECORDS, "XLGRID", "DFTD3", "VERSION", "4", "END", "ENDDFT"]
    after = _written(functional=CUSTOM_FUNCTIONAL, use_dispersion=True,
                     custom_functional=PBE0_RECORDS,
                     custom_dftd3=["DFTD3", "VERSION", "4", "END"])
    assert after == ["DFT", *PBE0_RECORDS, "XLGRID", "ENDDFT", "DFTD3", "VERSION", "4", "END"]


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


def test_custom_functional_is_the_menu_default(monkeypatch, capsys):
    seen = _answer(monkeypatch, [])
    opts = d12_interactive.configure_method(
        {"functional": CUSTOM_FUNCTIONAL, "method": "DFT", "dispersion": False,
         "custom_functional": list(PBE0_RECORDS)})
    assert opts["functional"] == CUSTOM_FUNCTIONAL
    assert opts["custom_functional"] == PBE0_RECORDS
    assert ("Keep the parent's functional (custom: EXCHANGE PBE / CORRELAT PBE / HYBRID 25)"
            in capsys.readouterr().out)
    assert not any("Enter another functional" in p for p in seen)


@pytest.mark.parametrize("name", ["LDA", "BECKE", "mPW91"])
def test_keep_is_not_offered_for_a_name_crystal23_does_not_know(monkeypatch, capsys, name):
    seen = _answer(monkeypatch, [])
    opts = d12_interactive.configure_method({"functional": name, "method": "DFT",
                                             "dispersion": False})
    out = capsys.readouterr().out
    assert "Keep the parent's functional" not in out
    assert f"The parent's functional '{name}' is not a CRYSTAL23 functional keyword." in out
    assert any("Enter another functional" in p for p in seen)
    assert opts["functional"] == "HSE06"


def test_keep_is_still_offered_for_an_unlisted_keyword(monkeypatch, capsys):
    _answer(monkeypatch, [])
    opts = d12_interactive.configure_method({"functional": "PW1PW-D3", "method": "DFT",
                                             "dispersion": True})
    assert opts["functional"] == "PW1PW-D3"
    assert "Keep the parent's functional (PW1PW-D3)" in capsys.readouterr().out


def _unknown():
    return {"functional": None, "method": "DFT", "unrecognised_functional": "B2PLYP",
            "unrecognised_functional_source": "input", "dispersion": False}


@pytest.mark.parametrize("typed", ["PBESOL0-D3", "SCAN-D3", "PBE", "B97", "PBESOL-D3",
                                   "mPW1PW91-D3"])
def test_replacement_prompt_rejects_what_would_be_written_invalidly(monkeypatch, capsys, typed):
    seen = _answer(monkeypatch, [("Enter another functional", [typed, "PBE0"]),
                                 ("Add D3", "n")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "PBE0"
    assert sum("Enter another functional" in p for p in seen) == 2
    assert f"'{typed}' is not a CRYSTAL23 functional keyword." in capsys.readouterr().out


@pytest.mark.parametrize("typed, functional, written", [
    ("PBEXC", "PBEXC", ["PBEXC"]),
    ("pbe-d3", "PBE-D3", ["PBE-D3"]),
    ("B97-D3", "B97-D3", ["B97-D3"]),
    ("PBESOL", "PBESOL", ["PBESOLXC"]),
    ("PWGGA", "PWGGA", ["EXCHANGE", "PWGGA", "CORRELAT", "PWGGA"]),
    ("PW1PW-D3", "PW1PW-D3", ["PW1PW-D3"]),
])
def test_replacement_prompt_accepts_valid_input(monkeypatch, typed, functional, written):
    _answer(monkeypatch, [("Enter another functional", typed), ("Add D3", "n")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == functional
    assert _written(functional=opts["functional"], use_dispersion=opts["dispersion"],
                    dft_grid="DEFAULT")[1:-1] == written


def test_replacement_mpw1pw91_with_d3_is_written_as_pw1pw_d3(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", "mPW1PW91"), ("Add D3", "y")])
    opts = d12_interactive.configure_method(_unknown())
    assert _written(functional=opts["functional"], use_dispersion=opts["dispersion"],
                    dft_grid="DEFAULT")[1:-1] == ["PW1PW-D3"]


# --------------------------------------------------- real opt2d12 run paths

def _parent(tmp_path, dft_lines, keep_out_functional=True, after=()):
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
    out = src.read_text(errors="replace")
    if not keep_out_functional:
        out = "".join(line for line in out.splitlines(keepends=True)
                      if "(EXCHANGE)[CORRELATION] FUNCTIONAL" not in line
                      and "GRIMME" not in line and "DFT-D3" not in line)
    (tmp_path / "parent.out").write_text(out)
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
    return decks[0].read_text().splitlines(), log


def _dft(lines):
    """The DFT block's records, without the parent's SPIN."""
    return [r for r in lines[lines.index("DFT") + 1:lines.index("ENDDFT")] if r != "SPIN"]


@pytest.mark.parametrize("extra, stdin", [
    (("--non-interactive",), ""),
    (("--non-interactive", "--calc-type", "SP"), "n\n" + "\n" * 39),  # engine planless
    ((), "n\n1\n" + "\n" * 60),  # interactive, blank answers
])
def test_opt2d12_keeps_a_user_defined_hybrid(tmp_path, extra, stdin):
    parent = _parent(tmp_path, PBE0_RECORDS)
    lines, log = _run(parent, *extra, stdin=stdin)
    assert _dft(lines)[:6] == PBE0_RECORDS
    assert "PBE" not in _dft(lines)[6:] and "PBE-D3" not in lines
    assert "EOFError" not in log


def test_opt2d12_keeps_the_parents_dftd3_block(tmp_path):
    parent = _parent(tmp_path, ["EXCHANGE", "PBE", "CORRELAT", "PBE"],
                     after=("DFTD3", "VERSION", "4", "END"))
    lines, _ = _run(parent, "--non-interactive")
    assert _dft(lines)[:4] == ["EXCHANGE", "PBE", "CORRELAT", "PBE"]
    end = lines.index("ENDDFT")
    assert lines[end + 1:end + 5] == ["DFTD3", "VERSION", "4", "END"]


def test_unmapped_deck_functional_is_not_replaced_by_the_output_guess(tmp_path):
    # The output still prints B3LYP's exchange/correlation line; the deck's
    # functional line is one nothing maps. The guess must not be taken silently.
    parent = _parent(tmp_path, ["B2PLYP"])
    lines, log = _run(parent, "--non-interactive")
    assert "WARNING: The parent's functional 'B2PLYP' is not a CRYSTAL23 functional keyword." in log
    assert _dft(lines)[0] == "HSE06-D3"
