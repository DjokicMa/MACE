"""The odd-electron warning for HSE-type screened hybrids covers every dimensionality.

Measured on CRYSTAL23 (HPCC, 2026-10-02): HSEsol-3c, HSE06 and HSEsol crash on
any odd-electron open-shell system, charged or not, as a MOLECULE (0D) and in
a 3D box ("DIIS TEST: NaN" at SCF cycle 1-3, then a segfault or malloc
assert; NODIIS + LEVSHIFT, SPINLOCK 1 30 and dropping BIPOSIZE/EXCHSIZE did
not help). PBE0 and PBEsol0 converged. Even counts with SPIN are fine.

The warning used to fire for molecules only. It must name PBE0/PBEsol0 as what
converged, and the deck must keep the functional asked for: MACE never
changes the user's method.
"""
import pytest

from charge_records import carry_charge_records
from ion_charges import MethodInfo, apply_charge_setup, plan_charge_setup, screened_hybrid
from ion_fixtures import Z, ec, li_near, molecule_deck, shuffled
from mace.utils.ion_fragments import cell_matrix


def _li_ec2():
    m1, m2 = ec(), [(s, (x + 9.0, y + 9.0, z)) for s, (x, y, z) in ec()]
    return shuffled(m1 + m2 + [li_near(m1, 1)], seed=2)


ODD = _li_ec2()  # Li+(EC)2 written neutral: 95 electrons
BOX = cell_matrix(30.0, 30.0, 30.0, 90, 90, 90)


def _plan(atoms, functional, periodic=False, **kw):
    syms = [s for s, _ in atoms]
    xyz = [p for _, p in atoms]
    if periodic:
        xyz = [(x + 15, y + 15, z + 15) for x, y, z in xyz]
        kw.update(cell=BOX, periodic_dirs=(0, 1, 2))
    from ion_charges import BasisInfo
    return plan_charge_setup(syms, [Z[s] for s in syms], xyz, basis=BasisInfo(internal="SOLDEF2MSVP"),
                             method=MethodInfo(functional=functional), ion_charges="auto", **kw)


def _warned(setup):
    return [w for w in setup.warnings if "PBEsol0" in w]


@pytest.mark.parametrize("name", ["HSESOL3C", "HSE06", "HSESOL", "HSEsol-3c", "HSE06-D3", "hse06"])
def test_the_screened_hybrids_are_recognised(name):
    assert screened_hybrid(name)


@pytest.mark.parametrize("name", ["PBE0", "PBESOL0", "B3LYP", "PBEH3C", "HF3C", "", None])
def test_other_functionals_are_not(name):
    assert not screened_hybrid(name)


@pytest.mark.parametrize("periodic", [False, True], ids=["0D", "3D"])
@pytest.mark.parametrize("functional", ["HSESOL3C", "HSE06", "HSESOL"])
def test_odd_open_shell_with_a_screened_hybrid_warns_in_0d_and_3d(functional, periodic):
    setup = _plan(ODD, functional, periodic=periodic, net_charge=0, spinlock=1)
    assert setup.spin and setup.spinlock == (1, 50)
    (w,) = _warned(setup)
    assert w.startswith("WARNING")
    assert "PBE0" in w and "PBEsol0" in w and "converged" in w
    assert "0D" in w and "3D" in w
    assert f"keeps {functional} as asked" in w


@pytest.mark.parametrize("periodic", [False, True], ids=["0D", "3D"])
def test_odd_open_shell_with_pbe0_is_quiet(periodic):
    setup = _plan(ODD, "PBE0", periodic=periodic, net_charge=0, spinlock=1)
    assert not _warned(setup)


def test_even_count_with_a_screened_hybrid_is_quiet():
    setup = _plan(ODD, "HSESOL3C", net_charge=1)  # Li+(EC)2 at +1: 94 electrons
    assert setup.doping is not None and not setup.spin
    assert not _warned(setup)


def test_the_functional_is_never_changed():
    setup = _plan(ODD, "HSESOL3C", net_charge=0, spinlock=1)
    deck = apply_charge_setup(molecule_deck(ODD), setup)
    assert "HSESOL3C" in deck.splitlines()
    assert "PBE0" not in deck and "PBESOL0" not in deck


# ------------------------------------------------- opt2d12 carried spin state

LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]
ODD_TAIL = "DOPING\n-2\nSPINLOCK\n1 30\n"  # Li + H2O 13 electrons; -2 leaves 11


def _deck(dft, spin, tail=""):
    return molecule_deck(LI_WATER, basis="BASISSET\nSOLDEF2MSVP\n", dft=dft, spin=spin, tail=tail)


@pytest.mark.parametrize("child_spin", [False, True], ids=["spin-from-parent", "spin-in-child"])
def test_a_derived_odd_open_shell_deck_with_a_screened_hybrid_warns(child_spin):
    parent = _deck("HSE06\nXLGRID\n", spin=True, tail=ODD_TAIL)
    child = _deck("HSE06\nXLGRID\n", spin=child_spin)
    out, notes = carry_charge_records(child, parent)
    assert "HSE06" in out.splitlines()
    assert any(n.startswith("WARNING") and "PBEsol0" in n and "keeps HSE06" in n for n in notes), notes


def test_a_derived_odd_open_shell_deck_with_pbe0_is_quiet():
    parent = _deck("PBE0\nXLGRID\n", spin=True, tail=ODD_TAIL)
    out, notes = carry_charge_records(_deck("PBE0\nXLGRID\n", spin=False), parent)
    assert not any("PBEsol0" in n for n in notes)


# ------------------------------------------------------- cif2d12, real command

def test_cif2d12_warns_with_the_functional_given_as_functional(tmp_path):
    """The example configs name the functional "functional" (not "dft_functional")."""
    import json
    import os
    import subprocess
    import sys

    from conftest import REPO_ROOT
    from ion_fixtures import molecule_cif

    pytest.importorskip("ase")
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / "li_ec2.cif").write_text(molecule_cif(ODD, name="li_ec2"))
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({
        "calculation_type": "SP", "method": "DFT", "functional": "HSESOL3C", "is_3c_method": True,
        "basis_set": "SOLDEF2MSVP", "basis_set_type": "INTERNAL", "dimensionality": "MOLECULE",
        "symmetry_handling": "P1", "write_only_unique": False,
        "ion_charges": "auto", "net_charge": 0, "spinlock": 1}))
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    run = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "convert", "--no-banner", "--batch",
                          "--options_file", str(opts), "--cif_dir", str(cifs), "--output_dir",
                          str(tmp_path / "out")], cwd=tmp_path, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    deck = next((tmp_path / "out").glob("*.d12")).read_text().splitlines()
    assert "HSESOL3C" in deck and "SPIN" in deck
    said = " ".join((run.stdout + run.stderr).split())
    assert "PBEsol0 converged" in said and "keeps HSESOL3C as asked" in said
