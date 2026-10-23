"""SP/OPT derived properties (mace/properties/defs/scf_summary.py): the
Mulliken summary per element, the dipole moment CRYSTAL prints (and a
slab's potential step from it), and the packing fractions of a 3D cell."""
import contextlib
import io
import math
from pathlib import Path

import pytest

from conftest import TEST_DATA
from mace.properties import registry
from mace.properties.defs import scf_summary as S
from mace.utils.property_extractor import CrystalPropertyExtractor


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    root = tmp_path_factory.mktemp("props_scf") / "suite"
    ES.copy_suite(root)
    return root


def props(out: Path):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(out))


def test_printed_dipole_of_a_molecule(suite):
    """EC_sp.out: the Debye line (-0.62266, 4.58203, -2.67125) D, |p| = 5.3403 D; the A.U. line has the
    opposite sign (electron charge +1) and 1 e bohr = 2.5417 D."""
    out = suite / "5-molecule-0d" / "EC" / "SP" / "EC_sp.out"
    p = props(out)
    assert p["dipole_moment_x_debye"] == pytest.approx(-0.622659)
    assert p["dipole_moment_y_debye"] == pytest.approx(4.58203)
    assert p["dipole_moment_debye"] == pytest.approx(math.sqrt(0.622659 ** 2 + 4.58203 ** 2 + 2.67125 ** 2))
    assert p["dipole_moment_y_debye"] == pytest.approx(1.80271 * S.DEBYE_PER_AU, rel=1e-4)


def test_slab_dipole_step_matches_the_potential_cube():
    """The printed z dipole of a slab's SP (0.150261 D) as -4 pi p / A agrees with the vacuum step of
    its CHARGE+POTENTIAL run's POT cube (-0.0909 eV) to 1 %."""
    sp = TEST_DATA / "SP" / "4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_sp.out"
    if not sp.is_file():
        pytest.skip("test/ corpus not present")
    p = props(sp)
    assert p["dipole_moment_z_debye"] == pytest.approx(0.150261)
    assert p["dipole_potential_step_ev"] == pytest.approx(-0.0909, rel=0.01)


def test_mulliken_summary(suite):
    """SOD (SiO2 sodalite): one Si and one O environment; the summary is the processor's charges."""
    p = props(suite / "1-porous" / "SOD" / "SP" / "SOD_sp.out")
    by = p["mulliken_by_element"]
    assert by["Si"]["count"] == 6 and by["O"]["count"] == 12
    q = [r["mulliken_charge"] for r in p["processed_atomic_charges"]]
    assert p["mulliken_charge_min"] == min(q) and p["mulliken_charge_max"] == max(q)
    assert by["Si"]["charge"]["mean"] == pytest.approx(-2 * by["O"]["charge"]["mean"], abs=2e-3)
    assert p["mulliken_spin_total"] == pytest.approx(0.0, abs=1e-6)


def test_mulliken_spin_summary_of_an_open_shell_run(suite):
    """A spin-polarized slab: the total magnetization is the sum of the spins, Σ|m| at least |Σm|."""
    p = props(suite / "3-slab-2d" / "4LG_AA" / "OPT" / "4LG_AA_opt.out")
    spins = [r["spin_density"] for r in p["processed_atomic_charges"]]
    assert p["mulliken_spin_total"] == pytest.approx(sum(spins))
    assert p["mulliken_spin_abs_total"] == pytest.approx(sum(abs(s) for s in spins))
    assert p["mulliken_spin_abs_total"] >= abs(p["mulliken_spin_total"])
    assert "spin" in p["mulliken_by_element"]["C"]


def test_packing_fractions(suite):
    """Diamond: covalent spheres (0.76 Å) do not overlap (bond 1.545 Å), so the fraction is
    N (4/3) pi r^3 / V (to the 0.15 Å grid's 2 %); the van der Waals spheres fill the cell. SOD: 1 - the porosity tool's void fraction."""
    pytest.importorskip("ase")                              # packing_fraction reads the structure with ase
    p = props(suite / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out")
    v = p["primitive_cell_volume"]
    assert p["packing_fraction_covalent"] == pytest.approx(2 * 4 / 3 * math.pi * 0.76 ** 3 / v, rel=0.02)
    assert p["packing_fraction_vdw"] == pytest.approx(1.0)
    from mace.analysis import porosity, structure
    out = suite / "1-porous" / "SOD" / "SP" / "SOD_sp.out"
    st = structure.load(out)
    pf = S.packing_fractions(st, spacing=porosity.DEFAULT_SPACING, max_points=porosity.DEFAULT_MAX_POINTS)
    void = porosity.analyze(st, probe=0.0)["void_fraction_geometric"]
    assert pf["vdw"] == pytest.approx(1 - void, abs=1e-9)
    assert props(out)["packing_fraction_vdw"] == pytest.approx(1 - void, abs=0.01)


def test_not_applicable_runs(suite):
    _, rep = registry.run_derived(Path("x.out"), "no dipole here", {"calculation_type": "SP"}, only=["dipole_moment"])
    assert rep["dipole_moment"]["status"] == "skipped"
    p = props(suite / "3-slab-2d" / "graphene" / "SP" / "graphene_sp.out")
    rep = p["extraction_provenance"]["derived"]["packing_fraction"]
    assert rep["status"] == "skipped" and "vacuum" in rep["reason"]
