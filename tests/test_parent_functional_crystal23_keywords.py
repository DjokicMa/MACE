"""A parent's functional is written as a keyword CRYSTAL23 accepts, or asked about.

Checked with CRYSTAL/23-intel-2023a (TESTPDIM), not only the manual:

* PBEXC and bare PBE are both accepted (the same functional); a parent's
  PBEXC was written back as PBE and is now kept as written.
* Bare B97 is rejected ("KEYWORD B97 NOT RECOGNIZED"; B97 is not an
  EXCHANGE choice either): the menu no longer writes it, B97 is offered
  with D3 only.
* SCAN-D3, PBESOL-D3, PBESOL0-D3, r2SCAN-D3, CAM-B3LYP-D3 and B3PW-D3 are
  rejected: a parent naming one goes to the replacement prompt instead of
  being kept. wB97X-D3 is accepted and kept.
* LSRSH-PBE needs its omega/cSR/cLR record: the replacement prompt asks
  for it.
"""
import io

import pytest

import d12_interactive
import menu_nav
from d12_constants import CUSTOM_FUNCTIONAL
from d12_parsers import CrystalInputParser
from d12_writer import write_dft_section


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


# ------------------------------------------------------------------ PBEXC


def test_pbexc_is_kept_as_written(tmp_path):
    data = _parse(tmp_path, ["PBEXC", "XLGRID"])
    assert data["functional"] == "PBEXC"
    assert _written(data["functional"])[1:-1] == ["PBEXC"]


REJECTED_D3 = ["SCAN-D3", "PBESOL-D3", "PBESOL0-D3", "r2SCAN-D3", "CAM-B3LYP-D3",
               "B3PW-D3"]


@pytest.mark.parametrize("name", REJECTED_D3)
def test_d3_names_crystal23_rejects_are_unrecognised(tmp_path, name):
    data = _parse(tmp_path, [name])
    assert data.get("functional") is None
    assert data["unrecognised_functional"] == name and data["dispersion"] is True


def test_wb97x_d3_which_crystal23_accepts_is_kept(tmp_path):
    data = _parse(tmp_path, ["wB97X-D3"])
    assert data["functional"] == "wB97X-D3" and data["dispersion"] is True
    assert "unrecognised_functional" not in data


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


def test_replacement_lsrsh_pbe_asks_for_its_record(monkeypatch):
    _answer(monkeypatch, [("Enter another functional", "LSRSH-PBE"),
                          ("needs one record", "0.11 0.25 0.00001")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == CUSTOM_FUNCTIONAL
    assert opts["custom_functional"] == ["LSRSH-PBE", "0.11 0.25 0.00001"]
    assert _written(opts["functional"], opts["dispersion"],
                    custom_functional=opts["custom_functional"])[1:-1] == [
        "LSRSH-PBE", "0.11 0.25 0.00001"]


def test_replacement_lsrsh_pbe_without_a_record_is_asked_again(monkeypatch, capsys):
    seen = _answer(monkeypatch, [("Enter another functional", ["LSRSH-PBE", "PBE0"]),
                                 ("needs one record", "0.11"), ("Add D3", "n")])
    opts = d12_interactive.configure_method(_unknown())
    assert opts["functional"] == "PBE0"
    assert sum("Enter another functional" in p for p in seen) == 2
    assert "LSRSH-PBE needs 3 numbers" in capsys.readouterr().out


def test_menu_b97_without_d3_is_refused(monkeypatch, capsys):
    # GGA category, B97 (7th): "no" to D3 is refused and the choice asked again
    seen = _answer(monkeypatch, [("Select functional category", "2"),
                                 ("Select GGA functional", ["7", "7"]),
                                 ("Add D3", ["n", "y"])])
    opts = d12_interactive.configure_method({"functional": "PBE0", "method": "DFT",
                                             "dispersion": False})
    assert opts["functional"] == "B97-D3"
    assert sum("Select GGA functional" in p for p in seen) == 2
    assert "B97 is available in CRYSTAL23 only with D3" in capsys.readouterr().out


def test_menu_pbe_without_d3_stays_bare_pbe(monkeypatch):
    _answer(monkeypatch, [("Select functional category", "2"),
                          ("Select GGA functional", "2"), ("Add D3", "n")])
    opts = d12_interactive.configure_method({"functional": "PBE0", "method": "DFT",
                                             "dispersion": False})
    assert _written(opts["functional"], opts["dispersion"])[1:-1] == ["PBE"]


@pytest.mark.parametrize("name", REJECTED_D3)
def test_rejected_d3_name_goes_to_the_replacement_prompt(monkeypatch, capsys, name):
    seen = _answer(monkeypatch, [])
    d12_interactive.configure_method({"functional": name, "method": "DFT",
                                      "dispersion": True})
    assert any("Enter another functional" in p for p in seen)
    assert f"The parent's functional '{name}' is not a CRYSTAL23" in capsys.readouterr().out
