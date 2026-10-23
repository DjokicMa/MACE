"""`mace analyze --extract-properties` on elastic, EOS and QHA outputs.

The extractor labeled EOS, ELAPIEZO and PHOTOELA runs SP and ELASTCON and QHA
runs FREQ (the NUCHESS Hessian and QHA print frequencies), and stored the last
strained cell's geometry and energy, and those frequencies' thermodynamics, as
the material's. The type is now the run's own (ELASTIC, EOS, QHA, as the
completion checker sorts it); structure, energies and populations come from the
reference cell (the part of the output before the first strain), and no FREQ
keys are written. The module's results are read as before.

Real HPCC CRYSTAL23 outputs of tests/data/hpcc_116/mechanics (README there).
"""
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.utils.calc_detection import mechanics_reference_text
from mace.utils.property_extractor import CrystalPropertyExtractor

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
TYPES = {"dia_elastcon": "ELASTIC", "dia_elastcon_nuchess": "ELASTIC", "dia_elapiezo": "ELASTIC",
         "dia_elapiezo_strained": "ELASTIC", "bn_elapiezo": "ELASTIC", "dia_photoela": "ELASTIC",
         "dia_eos": "EOS", "dia_eos_strained": "EOS", "dia_qha": "QHA"}
FREQ_KEYS = ("has_frequency_data", "vibrational_frequencies", "zero_point_energy_au",
             "gibbs_free_energy_au", "entropy_j_mol_k", "heat_capacity_j_mol_k")


@pytest.fixture(scope="module")
def extract(tmp_path_factory):
    ex = CrystalPropertyExtractor(db_path=str(tmp_path_factory.mktemp("db") / "x.db"), enable_tracking=False)
    cache = {}

    def run(name):
        if name not in cache:
            cache[name] = ex.extract_all_properties(MECH / f"{name}.out")
        return cache[name]
    return run


@pytest.mark.parametrize("name", sorted(TYPES))
def test_the_type_is_the_runs_own(extract, name):
    props = extract(name)
    assert props["calculation_type"] == TYPES[name]
    assert not any(k in props for k in FREQ_KEYS)


def test_the_cphf_run_is_its_own_type(extract):
    """CPHF has no strained cells; its type is CPHF, as completion sorts it
    (test_cphf_calc_type), not None (the trimmed output has no SCF lines)."""
    assert extract("dia_cphf").get("calculation_type") == "CPHF"


@pytest.mark.parametrize("name", ["dia_eos_strained", "dia_elapiezo_strained"])
def test_no_strained_cell_or_its_energy(extract, name):
    props = extract(name)
    text = (MECH / f"{name}.out").read_text()
    assert "FINAL OPTIMIZED GEOMETRY" in text and "OPT END - CONVERGED" in text
    assert "crystallographic_a" not in props            # was 3.44764376 / 3.49163903
    assert props["primitive_a"] == pytest.approx(2.50656)
    assert props["primitive_alpha"] == pytest.approx(60.0)
    assert props["primitive_cell_volume"] == pytest.approx(11.135747)
    assert "total_energy_au" not in props               # was the strained cell's OPT END energy


def test_the_nuchess_frequencies_are_not_a_freq_run(extract):
    props = extract("dia_elastcon_nuchess")
    assert "VIBRATIONAL" in (MECH / "dia_elastcon_nuchess.out").read_text()
    assert props["elastic_tensor"]["matrix"][0][0] == extract("dia_elastcon")["elastic_tensor"]["matrix"][0][0]


@pytest.mark.parametrize("name", ["dia_eos", "dia_elapiezo", "dia_qha", "bn_elapiezo", "dia_photoela"])
def test_the_module_results_are_still_read(extract, name):
    props = extract(name)
    assert any(k in props for k in ("elastic_tensor", "eos_fits", "qha_thermal_properties",
                                    "photoelastic_tensor", "piezoelectric_e_tensor"))


def test_the_reference_part_ends_at_the_first_strain():
    text = (MECH / "dia_eos_strained.out").read_text()
    ref = mechanics_reference_text(text)
    assert "GEOMETRY FOR WAVE FUNCTION" in ref and "FINAL OPTIMIZED GEOMETRY" not in ref
    sp = (REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain" / "dia_sp_dp_x_p1pct.out").read_text()
    assert mechanics_reference_text(sp) is sp
