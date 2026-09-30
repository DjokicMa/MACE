"""Properties decks from a two-component (SOC) parent.

CRYSTAL23 manual sec. 6.4 (p. 183) lists what properties supports from a
2c-SCF: NEWK, BAND, ECHG, ECH3, PPAN, PROPS2COMP, ANBD, BWIDTH, DOSS, TOPO
(and auxiliary keywords); p. 166 says any feature not in chapter 6 is not
supported in 2c. So from a 2c parent MACE writes BAND, DOSS and CHARGE
(ECH3/ECHG) decks, and refuses TRANSPORT (BOLTZTRA) and POTENTIAL
(POT3/POTC), writing nothing.

The parent is recognised by the lines its .out prints for a 2c-SCF (copied
here from an fcc Au 2c-SCF run on HPCC) or by the TWOCOMPON block of its
.d12. The rest of the .out text here is made up.
"""
import sys
from pathlib import Path

import pytest

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen  # noqa: E402

OUT_HEAD = """ Au_sp
 CRYSTAL CALCULATION
 N. OF ATOMS PER CELL        1
 NUMBER OF AO               36
 N. OF ELECTRONS PER UNIT CELL    19
 TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL
"""
TWO_C_OUT = OUT_HEAD + (" POSSIBLY CONDUCTING STATE - EFERMI(AU) -7.0721845E-02 "
                        "(RES. CHARGE  3.51E-11;IT. 10)\n"
                        " - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70\n"
                        " CHARGE NORMALIZATION FACTOR   1.00000000\n"
                        " TOTAL X-COMP MAGNETIZATION    0.00000001\n"
                        " TOTAL Y-COMP MAGNETIZATION   -0.00000001\n")
SCALAR_OUT = OUT_HEAD + " ALPHA-BETA ELECTRONS\n"
SOC_D12 = "Au_sp\nCRYSTAL\n0 0 0\n225\n4.0782\n1\n279 0 0 0\nEND\n99 0\nEND\n" \
          "SHRINK\n12 12\nTWOCOMPON\nSOC\nEND\nSCFDIR\nEND\n"


@pytest.fixture
def generator(tmp_path, monkeypatch):
    """A D3Generator whose wavefunction step only records that it was reached
    (the guard comes before it) and stops there."""
    reached = []

    def fake_copy(self):
        reached.append(self.calc_type)
        return False

    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", fake_copy)

    def make(out_text, calc_type, d12_text=None):
        out = tmp_path / "Au_sp.out"
        out.write_text(out_text)
        if d12_text is not None:
            (tmp_path / "Au_sp.d12").write_text(d12_text)
        gen = d3gen.D3Generator(str(out), calc_type, output_dir=str(tmp_path / "d3"))
        return gen, reached
    return make


@pytest.mark.parametrize("calc_type", ["TRANSPORT", "POTENTIAL", "CHARGE+POTENTIAL"])
def test_unsupported_properties_of_a_2c_parent_are_refused(generator, capsys, calc_type):
    gen, reached = generator(TWO_C_OUT, calc_type)
    assert gen.generate_d3() is None
    assert reached == []   # refused before anything, the wavefunction included
    err = capsys.readouterr().err
    assert "two-component (SOC) run" in err and "p. 183" in err
    assert not (Path(gen.output_dir)).exists() or not list(Path(gen.output_dir).glob("*.d3"))


@pytest.mark.parametrize("calc_type", ["BAND", "DOSS", "CHARGE"])
def test_supported_properties_of_a_2c_parent_go_ahead(generator, capsys, calc_type):
    gen, reached = generator(TWO_C_OUT, calc_type)
    gen.generate_d3()
    assert reached == [calc_type]
    out = capsys.readouterr()
    assert "sec. 6.4, p. 183" in out.out
    # band ranges from NAO / electron count are flagged for BAND and DOSS
    assert ("2 x NAO spinor bands" in out.err) == (calc_type in ("BAND", "DOSS"))


def test_a_2c_parent_is_also_recognised_by_its_deck(generator, capsys):
    gen, reached = generator(SCALAR_OUT, "TRANSPORT", d12_text=SOC_D12)
    assert gen.parent_is_two_component()
    assert gen.generate_d3() is None and reached == []


@pytest.mark.parametrize("calc_type", ["TRANSPORT", "POTENTIAL", "BAND"])
def test_a_scalar_parent_is_not_affected(generator, capsys, calc_type):
    gen, reached = generator(SCALAR_OUT, calc_type,
                             d12_text=SOC_D12.replace("TWOCOMPON\nSOC\nEND\n", ""))
    assert not gen.parent_is_two_component()
    gen.generate_d3()
    assert reached == [calc_type]
    out = capsys.readouterr()
    assert "two-component" not in out.out + out.err
