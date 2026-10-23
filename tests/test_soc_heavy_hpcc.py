"""Heavy-element two-component (SOC) decks and outputs, verified on HPCC.

The 1.1.6 round (tests/data/hpcc_116/README.md): MACE wrote SOC single
points from CIFs with ``mace convert --batch`` for

* CsPbI3 (Pm-3m; Cs, Pb, I) with the official pob-TZVP-rev2-SOC basis (STUTSC),
* HgTe (F-43m; Hg, Te) with the library basis and INPSOC spin-orbit ECPs,
* Bi (R-3m, a semimetal) and W (Im-3m, a metal) with INPSOC and SMEAR,

all spin-polarized (MACE's default) with PBE0, and CRYSTAL/23-intel-2023a
ran every one to a normal termination (10-19 SCF cycles). These tests pin
the decks (written again from the same CIFs, byte for byte below the title)
and what MACE reads from the real outputs.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.phonopy_bridge.parent import final_mulliken_spins
from mace.utils.property_extractor import CrystalPropertyExtractor

SOC = REPO_ROOT / "tests" / "data" / "hpcc_116" / "soc"
MACE_CLI = REPO_ROOT / "mace_cli"
TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"

OPTIONS = {"dimensionality": "CRYSTAL", "calculation_type": "SP", "method": "DFT", "is_spin_polarized": True,
           "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, "scf_method": "DIIS", "symmetry_handling": "CIF",
           "basis_set_type": "EXTERNAL", "basis_set": str(TZ) + "/", "soc": True, "dft_functional": "PBE0"}
CASES = {   # material: extra options of its run
    "CsPbI3": {"soc_basis": "pob-tzvp-rev2-soc"},
    "HgTe": {},
    "Bi": {"use_smearing": True, "smearing_width": 0.01},
    "W": {"use_smearing": True, "smearing_width": 0.01},
}


def _below_title(text):
    return text.split("\n", 1)[1]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_decks_that_ran_are_what_mace_convert_writes(tmp_path, name):
    pytest.importorskip("ase", reason="CIF parsing needs ase")
    cif_dir = tmp_path / "cif"
    cif_dir.mkdir()
    (cif_dir / f"{name}.cif").write_text((SOC / "cif" / f"{name}.cif").read_text())
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({**OPTIONS, **CASES[name]}))
    env = dict(os.environ, MACE_NO_BANNER="1")
    r = subprocess.run([sys.executable, str(MACE_CLI), "convert", "--cif_dir", str(cif_dir), "--output_dir",
                        str(tmp_path), "--batch", "--options_file", str(opts)], capture_output=True, text=True,
                       env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    deck = next(tmp_path.glob(f"{name}_CRYSTAL_SP_symm_PBE0_SOC_*.d12")).read_text()
    assert _below_title(deck) == _below_title((SOC / f"{name}_soc.d12").read_text())


@pytest.mark.parametrize("name, ecp", [("CsPbI3", "STUTSC"), ("HgTe", "INPSOC"), ("Bi", "INPSOC"), ("W", "INPSOC")])
def test_deck_features(name, ecp):
    lines = (SOC / f"{name}_soc.d12").read_text().splitlines()
    assert ecp in lines
    i = lines.index("TWOCOMPON")
    assert lines[i:i + 4] == ["TWOCOMPON", "SOC", "END", "SCFDIR"]
    assert "SPIN" in lines                                  # spin-polarized stays the default
    assert "DIIS" not in lines and "HISTDIIS" not in lines  # refused by the 2c SCF (HPCC)
    assert ("SMEAR" in lines) == ("use_smearing" in CASES[name])


@pytest.mark.parametrize("name", ["CsPbI3", "HgTe", "Bi"])
def test_crystal_accepted_and_converged(name):
    text = (SOC / f"{name}_soc.out").read_text()
    assert "EEEEEEEEEE TERMINATION" in text and "ERROR" not in text
    assert "TWO-COMPONENT SCF: SYMMETRY DISABLED" in text
    assert "== SCF ENDED - CONVERGENCE ON ENERGY" in text


def _props(name):
    ex = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    ex.extract_fragments = False
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return CrystalPropertyExtractor.extract_all_properties(ex, SOC / f"{name}_soc.out", "m", "c")


@pytest.mark.parametrize("name, kind, gap", [("CsPbI3", "indirect", 1.9278), ("HgTe", "direct", 0.7896), ("Bi", None, 0.0)])
def test_what_mace_reads_from_the_outputs(name, kind, gap):
    p = _props(name)
    assert p["scf_two_component"] is True
    assert p["band_gap"] == pytest.approx(gap)
    if kind:
        assert p["band_gap_type"] == kind and p["electronic_classification"] == "semiconductor"
    else:
        assert p["electronic_classification"] == "metal" and p["fermi_energy_au"] == pytest.approx(-0.13320861)
    # the final (Mx, My, Mz) of the 2c SCF: zero for these non-magnetic systems
    assert len(p["two_component_magnetization"]) == 3 and p["two_component_magnetization_norm"] < 1e-6
    assert len(p["two_component_atomic_magnetization"]) == {"CsPbI3": 5, "HgTe": 2, "Bi": 2}[name]


def test_a_smeared_run_reports_the_zero_smearing_energy():
    """Bi (SMEAR 0.01): CRYSTAL's TOTAL ENERGY is E0 = (E+F)/2, below the SCF's E."""
    p = _props("Bi")
    text = (SOC / "Bi_soc.out").read_text()
    assert "(E(T)+F(T))/2    (E0)  -4.296943916815E+02" in text
    assert p["total_energy_au"] == pytest.approx(-429.69439168154)


@pytest.mark.parametrize("name", ["CsPbI3", "HgTe", "Bi"])
def test_mulliken_spins_of_a_2c_run(name):
    rows = final_mulliken_spins((SOC / f"{name}_soc.out").read_text())
    assert rows and all(abs(v) < 1e-3 for _, _, v in rows)
