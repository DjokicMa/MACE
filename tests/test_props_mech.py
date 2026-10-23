"""The mechanics results registered as derived properties
(mace/properties/defs/mechanics.py): a wrapper around
mace.elastic.properties.all_properties(analyze(text)), the numbers the
Mechanical Properties Results document shows. Every name it gives on the
real ELASTIC, EOS, QHA, CPHF and FREQ+INTCPHF outputs of tests/data has a
database key (none is dropped), the values are all_properties' own, and
the GUI's document (the parity producer) gives the same."""
import contextlib
import io
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.elastic import properties as P
from mace.properties import parity, registry
from mace.properties.defs import mechanics as M
from mace.utils.property_extractor import CrystalPropertyExtractor

DATA = REPO_ROOT / "tests" / "data"
RUNS = sorted((DATA / "hpcc_116" / "mechanics").glob("*.out")) + sorted((DATA / "intcphf").glob("*.out"))


def _props(out: Path):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(out)


@pytest.mark.parametrize("out", RUNS, ids=lambda p: p.name)
def test_every_mechanics_scalar_has_a_key_and_is_stored(out):
    text = out.read_text(errors="replace")
    allp = P.all_properties(A.analyze(text), text)
    props = _props(out)
    rep = props["extraction_provenance"]["derived"].get("mechanics_derived", {})
    if not allp:
        assert rep.get("status") != "stored"
        return
    assert rep["status"] == "stored", rep
    assert props["mechanics_derived"] == allp
    for name, v in allp.items():
        if name in M.SKIP:
            assert M.SKIP[name] in props                  # the extractor's own key holds it
            continue
        key = M.db_key(name, v["unit"])
        assert key in M.GIVEN, f"{out.name}: {name} ({v['unit']}) has no database key"
        assert M.GIVEN[key] == v["unit"], (name, v["unit"])
        assert props[key] == v["value"]
        assert key == key.lower()
    # the GUI's document gives the same values (the parity producer)
    ct = props["calculation_type"]
    prod = next(p for p in parity.PRODUCERS if p.name == "mechanics results")
    shown = prod.values(out, ct)
    assert shown and all(props[k] == v for k, v in shown.items())
    assert not [x for x in parity.producer_problems(out, ct, props) if x.startswith("mechanics results")]


def test_known_values():
    """Diamond's ELASTIC run: Pugh's ratio B_H/G_H from the stored Hill moduli; EOS: four fits."""
    props = _props(DATA / "hpcc_116" / "mechanics" / "dia_elastcon.out")
    assert props["pugh_ratio"] == pytest.approx(props["bulk_modulus_hill_gpa"] / props["shear_modulus_hill_gpa"],
                                                rel=1e-3)
    assert props["ductile_pugh"] is False
    assert props["youngs_modulus_min"] <= props["youngs_modulus_gpa"] <= props["youngs_modulus_max"]
    eos = _props(DATA / "hpcc_116" / "mechanics" / "dia_eos.out")
    assert {f"eos_{f}_residual_rms" for f in ("murnaghan", "birch_murnaghan", "poirier_tarantola", "vinet")} <= set(eos)
    assert eos["formula_units"] == 2
    born = _props(DATA / "intcphf" / "agbr_freq.out")
    assert born["dielectric_constant_static"] > born["dielectric_constant_electronic_freq"] > 1
    assert born["born_charge_min"] == pytest.approx(-born["born_charge_max"], abs=1e-3)


def test_registered_keys_are_well_formed():
    prop = registry.get("mechanics_derived")
    names = [o.name for o in prop.outputs]
    assert len(names) == len(set(names)) and "anisotropy_universal" not in names
    assert {"sound_velocity_ql_max_km_s", "cauchy_c12_minus_c44", "lo_to_splitting_max", "qha_cp_300k"} <= set(names)


def test_cli_extraction_stores_the_mechanics_keys(tmp_path):
    """`mace analyze --extract-properties` (subprocess) on an ELASTIC run."""
    run = tmp_path / "el"
    run.mkdir()
    (run / "dia_elastcon.out").write_bytes((DATA / "hpcc_116" / "mechanics" / "dia_elastcon.out").read_bytes())
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONPATH", "MACE_HOME")}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        str(run), "--db-path", str(tmp_path / "m.db")], capture_output=True, text=True, env=env,
                       timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rows = dict(sqlite3.connect(tmp_path / "m.db").execute("SELECT property_name, property_value FROM properties"))
    assert rows["vickers_hardness_tian_gpa"] > 50 and rows["sound_velocity_ql_max_km_s"] > 17
