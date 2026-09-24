"""Planless SP decks keep the parent's dispersion and its own EXTERNAL basis.

The planless engine path answers the interactive "modify settings" flow with
blank lines, so each prompt's default decides the child's method:

* "Add D3 dispersion correction to X? [Y/n]" defaulted to yes, so PBE0,
  HSE06, HSEsol, LC-wPBE, B3LYP, PBE... parents became their -D3 versions.
  A parent whose "<name>-D3" keyword is outside D3_FUNCTIONALS (PBESOL-D3,
  SCAN-D3, wB97X-D3...) was never asked and silently lost its -D3.
* The basis step went straight to the library menu, replacing the parent's own
  EXTERNAL basis records with the library file's (0.2673545 -> 0.1973545 for
  the diamond deck). A blank answer now keeps the parent's records.

The unit tests run everywhere; the planless runs need the test/ corpus.
"""
import shutil
import subprocess
import sys

import pytest

import d12_interactive
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
DIA = TEST_DATA / "OPT" / "1_dia_opt_rev1"


# --------------------------------------------------------------- D3 default

def _configure_blank(monkeypatch, parent):
    """configure_method on blank answers; returns (options, D3 prompt default)."""
    seen = {}

    def blank_yes_no(prompt, default="yes"):
        if "D3 dispersion" in prompt:
            seen["d3_default"] = default
        return default == "yes"

    monkeypatch.setattr(d12_interactive, "yes_no_prompt", blank_yes_no)
    monkeypatch.setattr(d12_interactive, "get_user_choice",
                        lambda prompt, options, default="1": default)
    opts = d12_interactive.configure_method(dict(parent, method_type="DFT"))
    return opts, seen.get("d3_default")


@pytest.mark.parametrize("fx", ["PBE0", "HSE06", "HSEsol", "LC-wPBE", "B3LYP",
                                "PBE", "BLYP", "B97", "M06", "mPW1PW91"])
def test_non_d3_parent_stays_without_d3(monkeypatch, fx):
    opts, default = _configure_blank(monkeypatch, {"functional": fx, "dispersion": False})
    assert default == "no"
    assert opts["functional"] == fx
    assert not opts["dispersion"]



def test_unrecognised_parent_functional_does_not_crash(monkeypatch):
    """The parsers leave functional=None for a functional they don't know
    (e.g. B2PLYP); the D3 default must not call .upper() on it."""
    opts, default = _configure_blank(monkeypatch, {"functional": None, "dispersion": False})
    assert default == "no"
    assert not opts["dispersion"]

@pytest.mark.parametrize("fx", ["PBE0-D3", "HSE06-D3", "B3LYP-D3", "PBE-D3"])
def test_d3_parent_keeps_d3(monkeypatch, fx):
    opts, default = _configure_blank(monkeypatch, {"functional": fx, "dispersion": True})
    assert default == "yes"
    assert opts["functional"] == fx
    assert opts["dispersion"]


@pytest.mark.parametrize("fx", ["PBESOL-D3", "PBESOL0-D3", "SCAN-D3", "r2SCAN-D3",
                                "wB97X-D3", "CAM-B3LYP-D3", "B3PW-D3"])
def test_d3_keyword_outside_the_prompt_list_is_kept(monkeypatch, fx):
    opts, default = _configure_blank(monkeypatch, {"functional": fx, "dispersion": True})
    assert default is None            # never asked
    assert opts["functional"] == fx
    assert opts["dispersion"]


def test_hsesol3c_parent_stays_hsesol3c(monkeypatch):
    opts, default = _configure_blank(monkeypatch, {"functional": "HSESOL3C",
                                                   "dispersion": False})
    assert default is None
    assert opts["functional"] == "HSESOL3C"
    assert not opts.get("dispersion")


def test_switching_to_an_unlisted_functional_drops_the_parents_d3(monkeypatch):
    """A user who picks SCAN for a PBE0-D3 parent gets plain SCAN, and the
    dispersion flag (which names the output file) follows."""
    answers = iter(["2", "4", "1"])      # DFT, meta-GGA, first entry

    def choose(prompt, options, default="1"):
        return next(answers)

    monkeypatch.setattr(d12_interactive, "get_user_choice", choose)
    monkeypatch.setattr(d12_interactive, "yes_no_prompt", lambda p, d="yes": d == "yes")
    opts = d12_interactive.configure_method(
        {"functional": "PBE0-D3", "dispersion": True, "method_type": "DFT"})
    assert not opts["functional"].endswith("-D3")
    assert opts["functional"] not in d12_interactive.D3_FUNCTIONALS
    assert opts["dispersion"] is False


# ------------------------------------------------------- real planless SP runs

def _need():
    if not DIA.with_suffix(".d12").exists() or not DIA.with_suffix(".out").exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")


def _parent(tmp_path, fx):
    """1_dia_opt_rev1 (EXTERNAL basis) with its DFT block set to ``fx``; the D3
    markers are removed from its .out for a non-D3 functional."""
    lines = DIA.with_suffix(".d12").read_text().splitlines()
    i, j = lines.index("DFT"), lines.index("ENDDFT")
    lines[0] = "dia_synth"
    d12 = lines[:i] + ["DFT", "SPIN", fx, "XLGRID", "ENDDFT"] + lines[j + 1:]
    out = DIA.with_suffix(".out").read_text()
    if not fx.endswith("-D3"):
        out = (out.replace("DFT-D3", "DFT-Dx").replace("GRIMME D3", "GRIMME Dx")
               .replace("D3 DISPERSION", "Dx DISPERSION"))
    (tmp_path / "parent.d12").write_text("\n".join(d12) + "\n")
    (tmp_path / "parent.out").write_text(out)
    return d12


def _planless_sp(tmp_path):
    """The engine's planless SP call: the interactive flow on blank answers."""
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
         "--d12-file", "parent.d12", "--non-interactive", "--calc-type", "SP"],
        cwd=tmp_path, input="n\n" + "\n" * 19, capture_output=True, text=True,
        timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != "parent.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].name, decks[0].read_text().splitlines()


def _basis(lines):
    """The EXTERNAL basis records: from the first line after the geometry's
    closing END/ENDOPT up to '99 0', whitespace-normalised."""
    end = lines.index("99 0")
    start = max(k for k in range(end) if lines[k].strip() in ("END", "ENDOPT"))
    return [ln.split() for ln in lines[start + 1:end + 1]]


@pytest.mark.parametrize("fx", ["PBE0", "HSEsol", "PBE0-D3", "PBESOL-D3"])
def test_planless_sp_keeps_the_parents_dispersion(tmp_path, fx):
    _need()
    _parent(tmp_path, fx)
    name, child = _planless_sp(tmp_path)
    dft = child[child.index("DFT") + 1:child.index("ENDDFT")]
    assert dft == ["SPIN", fx, "XLGRID"], dft
    assert f"_sp_{fx}_optimized" in name, name


def test_planless_sp_keeps_the_parents_external_basis(tmp_path):
    _need()
    parent = _parent(tmp_path, "PBE0")
    _, child = _planless_sp(tmp_path)
    assert _basis(child) == _basis(parent)
    body = "\n".join(child)
    assert "0.2673545" in body and "0.1973545" not in body
