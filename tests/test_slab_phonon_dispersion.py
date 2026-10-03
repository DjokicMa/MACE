"""opt2d12's phonon band template (6) for a SLAB parent writes the 2D deck.

Measured with CRYSTAL23 on an hBN monolayer (layer group 78, corresponding
space group 187, STO-3G/PBE, SHRINK 8 8):

* SCELPHONO takes a 2x2 matrix for a slab ("2 0 / 0 2", 4 reals, manual
  p.73); the 3x3 matrix aborted after the geometry.
* FREQCALC BANDS takes no labels for a slab ("INPFREQ **** FORMAT ERROR IN
  FREQCALC INPUT DECK"), and reads only I1 I2 J1 J2 from each record: the
  6-integer records of a 3D path ran but went to the wrong k points.
* "BANDS / 12 100 / 2 / 6 0 0 0 / 0 0 4 4" ran as M-Gamma-K, and the P1 slab
  "8 100 / 3 / 4 4 0 4 / 0 4 0 0 / 0 0 4 0" as V-Y-Gamma-X.

Before, opt2d12 asked every prompt and then refused a slab, saying a slab has
no space group to take the path from; the path is the in-plane part of the
corresponding space group's path, as opt2d3's slab BAND path is. A POLYMER
(not measured) is still refused, but before the questions that follow the
FREQ template.
"""
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

SCRIPT = REPO_ROOT / "Crystal_d12" / "CRYSTALOptToD12.py"
DATA = Path(__file__).parent / "data"
# "Use these exact settings?" no; Enter for the FREQ type; template 6; Enter
TEMPLATE_6 = ["n", "", "6"]


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME"):
        env.pop(k, None)
    return env


def _opt2d12(tmp_path, src, answers):
    d = tmp_path / "parent"
    d.mkdir()
    for ext in (".out", ".d12"):
        shutil.copy(src.with_suffix(ext), d / src.with_suffix(ext).name)
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-file", src.name,
                        "--calc-type", "FREQ"],
                       input="\n".join(answers) + "\n" * 200, capture_output=True,
                       text=True, env=_env(), timeout=180, cwd=d)
    decks = [p for p in d.glob("*.d12") if p.name != src.with_suffix(".d12").name]
    return r, decks


def _bands(lines):
    i = lines.index("BANDS")
    iss, nsub = (int(x) for x in lines[i + 1].split())
    nline = int(lines[i + 2])
    records = [[int(x) for x in rec.split()] for rec in lines[i + 3:i + 3 + nline]]
    return iss, nsub, records


def _points(iss, records):
    return [((Fraction(a, iss), Fraction(b, iss)), (Fraction(c, iss), Fraction(d, iss)))
            for a, b, c, d in records]


@pytest.mark.parametrize("parent", [DATA / "label_band" / "hbn_p1_slab.out",
                                    DATA / "low_dim_groups" / "graphene_lg80.out"])
def test_slab_phonon_bands_deck_is_two_dimensional(tmp_path, parent):
    r, decks = _opt2d12(tmp_path, parent, TEMPLATE_6)
    log = r.stdout + r.stderr
    assert r.returncode == 0 and len(decks) == 1, log[-3000:]
    lines = decks[0].read_text().splitlines()
    assert lines[1] == "SLAB"
    i = lines.index("SCELPHONO")
    assert lines[i + 1:i + 4] == ["2 0", "0 2", "FREQCALC"]
    iss, nsub, records = _bands(lines)
    assert nsub == 100 and records
    assert all(len(rec) == 4 for rec in records), records
    points = _points(iss, records)
    half, third = Fraction(1, 2), Fraction(1, 3)
    if "p1" in parent.name:
        # V-Y-Gamma-X, as measured
        assert points == [((half, half), (0, half)), ((0, half), (0, 0)),
                          ((0, 0), (half, 0))]
        assert "V-Y-GAMMA-X" in lines[0], lines[0]
    else:
        ends = {p for seg in points for p in seg}
        assert (0, 0) in ends and (half, 0) in ends and (third, third) in ends, points
        assert "GAMMA" in lines[0] and "K" in lines[0], lines[0]
    assert "INTERPHESS" not in lines and "WANG" not in lines


def test_polymer_dispersion_is_refused_before_the_method_questions(tmp_path):
    r, decks = _opt2d12(tmp_path, DATA / "low_dim_groups" / "polyyne_rg28.out", TEMPLATE_6)
    log = r.stdout + r.stderr
    assert r.returncode != 0 and decks == [], log[-3000:]
    assert "POLYMER" in log and "phonon dispersion" in log
    assert "Select functional category" not in log
    assert "has no space group" not in log


def test_refusal_reasons():
    from d12_calc_freq import phonon_dispersion_refusal
    disp = {"dispersion": True, "scelphono": [2, 2, 2]}
    slab_out = (DATA / "low_dim_groups" / "graphene_lg80.out").read_text()
    assert phonon_dispersion_refusal("CRYSTAL", disp) is None
    assert phonon_dispersion_refusal("SLAB", disp, slab_out) is None
    # not known yet (the interactive check, before the output is read)
    assert phonon_dispersion_refusal("SLAB", disp, None) is None
    # cif2d12: no parent output to read the corresponding space group from
    assert "CORRESPONDING SPACE GROUP" in phonon_dispersion_refusal("SLAB", disp)
    # WANG is 3D only (manual p.234); the rest were not run for a slab
    assert "3D" in phonon_dispersion_refusal("SLAB", dict(disp, wang=[1] * 9), slab_out)
    for key in ("interphess", "pdos"):
        assert key.upper() in phonon_dispersion_refusal("SLAB", dict(disp, **{key: {}}), slab_out)
    assert "POLYMER" in phonon_dispersion_refusal("POLYMER", disp, None)
