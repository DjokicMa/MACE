"""The relaxation time tau for BOLTZTRA (mace/transport, ``mace tau``).

* the deformation-potential formulas against the literature (phosphorene,
  Qiao et al., Nat. Commun. 5, 4475 (2014), table 1) and against each other
  (the 3D tau(E) average equals mu m*/e);
* the strained decks (CRYSTAL's ELASTIC, manual p. 47) written from the
  repo's Si parent deck;
* the fits on real CRYSTAL23 outputs of such decks (diamond, B3LYP-D3,
  HPCC, tests/data/hpcc_116/strain), with C checked against the same
  structure's ELASTCON run.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.transport import dpt, strain

MACE_CLI = REPO_ROOT / "mace_cli"
SI_DECK = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si" / "parent.d12"
HPCC = REPO_ROOT / "tests" / "data" / "hpcc_116"
SERIES = sorted((HPCC / "strain").glob("dia_sp_dp_x_*.out"))


def _mace(*args, cwd=None):
    env = dict(os.environ, MACE_NO_BANNER="1")
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


# ------------------------------------------------------------------ formulas
def test_phosphorene_mobilities_of_qiao_2014():
    """Electrons: armchair m*=0.17, m_zz=1.12, E1=2.72 eV, C=28.94 N/m -> ~1.1e3;
    zigzag m*=1.12, E1=7.11 eV, C=101.60 N/m -> ~80 cm^2/(V s)."""
    a = dpt.dp_mobility(2, 0.17, 2.72, 28.94, 300, 1.12)
    z = dpt.dp_mobility(2, 1.12, 7.11, 101.60, 300, 0.17)
    assert 1.05e3 < a["mobility_cm2_Vs"] < 1.2e3
    assert 75 < z["mobility_cm2_Vs"] < 95
    assert a["tau_fs"] == pytest.approx(a["mobility_cm2_Vs"] * 1e-4 * 0.17 * dpt.M_E / dpt.E_CHARGE * 1e15)


def test_3d_energy_resolved_tau_averages_to_the_mobility_tau():
    m, e1, c, t = 0.26, 9.5, 165.0, 300.0
    tau0 = dpt.tau0_3d(m, e1, c, t)
    avg = 4 * tau0 / (3 * math.sqrt(math.pi))
    assert dpt.dp_mobility(3, m, e1, c, t)["tau_s"] == pytest.approx(avg, rel=1e-12)
    rows = dpt.tau_energy_table(m, e1, c, t, [1.0, 4.0])
    assert rows[0]["tau_fs"] == pytest.approx(tau0 * 1e15) and rows[1]["tau_fs"] == pytest.approx(tau0 * 1e15 / 2)


def test_scaling_laws():
    base = dpt.dp_mobility(3, 0.3, 5.0, 100.0, 300)["mobility_cm2_Vs"]
    assert dpt.dp_mobility(3, 0.3, 10.0, 100.0, 300)["mobility_cm2_Vs"] == pytest.approx(base / 4)
    assert dpt.dp_mobility(3, 0.3, 5.0, 200.0, 300)["mobility_cm2_Vs"] == pytest.approx(base * 2)
    assert dpt.dp_mobility(3, 0.3, 5.0, 100.0, 600)["mobility_cm2_Vs"] == pytest.approx(base / 2 ** 1.5)
    assert dpt.dp_mobility(3, 0.3, -5.0, 100.0, 300)["mobility_cm2_Vs"] == pytest.approx(base)   # sign of E1
    one = dpt.dp_mobility(1, 0.3, 5.0, 1e-7, 300)["mobility_cm2_Vs"]
    assert dpt.dp_mobility(1, 0.6, 5.0, 1e-7, 300)["mobility_cm2_Vs"] == pytest.approx(one / 2 ** 1.5)


@pytest.mark.parametrize("args", [(3, 0.0, 5.0, 100.0), (3, 0.3, 0.0, 100.0), (3, 0.3, 5.0, -1.0), (4, 0.3, 5.0, 100.0)])
def test_bad_inputs_are_refused(args):
    with pytest.raises(dpt.TauError):
        dpt.dp_mobility(*args)


def test_provenance_names_method_reference_and_sources():
    res = dpt.estimate(2, 0.17, 2.72, 28.94, 300, 1.12, {"E1": "fit", "C": "fit"}, carrier="electron", axis="x")
    assert "Takagi" in res["reference"] and "Qiao" in res["reference"]
    assert res["inputs"]["E1"]["source"] == "fit" and res["inputs"]["m_star"]["source"] == "given"
    assert res["relaxtim"].startswith("RELAXTIM\n") and "acoustic-phonon" in res["limits"]
    text = dpt.report(res)
    assert "Reference:" in text and "RELAXTIM" in text and "from fit" in text
    mob = dpt.estimate_from_mobility(1400, 0.26)
    assert mob["tau_fs"] == pytest.approx(1400e-4 * 0.26 * dpt.M_E / dpt.E_CHARGE * 1e15)


def test_mace_tau_estimate_json():
    r = _mace("tau", "estimate", "--dim", "2", "--m-star", "0.17", "--m-other", "1.12", "--e1", "2.72",
              "--c", "28.94", "--json")
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert 1.05e3 < res["mobility_cm2_Vs"] < 1.2e3 and res["dimensionality"] == 2
    r = _mace("tau", "estimate", "--m-star", "0.3")
    assert r.returncode == 2 and "--e1 and --c are required" in r.stderr


# ------------------------------------------------------------------ decks
def test_strained_deck_inserts_elastic_after_the_atoms():
    text = SI_DECK.read_text()
    deck = strain.strained_deck(text, "x", 0.01)
    lines = deck.splitlines()
    dim, end = strain.atoms_end(text.splitlines())
    assert dim == 3
    assert lines[end:end + 5] == ["ELASTIC", "-2", "0.010000 0.0 0.0", "0.0 0.0 0.0", "0.0 0.0 0.0"]
    assert lines[:end][1:] == text.splitlines()[1:end]           # geometry untouched, title extended
    assert strain.strain_of(deck) == ("x", 0.01)
    assert strain.strained_deck(text, "z", -0.005).splitlines()[end + 4] == "0.0 0.0 -0.005000"


def test_strained_decks_raise_toldee_and_shorten_spinlock():
    text = SI_DECK.read_text().replace("TOLDEE\n", "SPINLOCK\n0 50\nTOLDEE\n", 1)
    lines = text.splitlines()
    lines[lines.index("TOLDEE") + 1] = "6"
    deck = strain.strained_deck("\n".join(lines) + "\n", "y", 0.005).splitlines()
    assert deck[deck.index("TOLDEE") + 1] == "8"
    assert deck[deck.index("SPINLOCK") + 1] == f"0 {strain.SPINLOCK_CYCLES}"


@pytest.mark.parametrize("bad, words", [("OPTGEOM", "single-point"), ("TWOCOMPON", "one-component")])
def test_strain_needs_a_one_component_sp_deck(bad, words):
    text = SI_DECK.read_text().replace("END\n", f"{bad}\nEND\n", 1)
    with pytest.raises(strain.StrainError, match=words):
        strain.strained_deck(text, "x", 0.01)


def test_a_slab_is_strained_in_plane_only():
    slab = "t\nSLAB\n1\n2.5 2.5 120\n1\n6 0.0 0.0 0.0\nEND\n"
    assert strain.atoms_end(slab.splitlines()) == (2, 6)
    with pytest.raises(strain.StrainError, match="not periodic"):
        strain.strained_deck(slab, "z", 0.01)


def test_mace_tau_strain_writes_the_set(tmp_path):
    shutil.copy(SI_DECK, tmp_path / "si_sp.d12")
    r = _mace("tau", "strain", "si_sp.d12", "--axis", "x", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    names = sorted(p.name for p in tmp_path.glob("si_sp_dp_x_*.d12"))
    assert names == ["si_sp_dp_x_m0_5pct.d12", "si_sp_dp_x_m1pct.d12", "si_sp_dp_x_p0_5pct.d12",
                     "si_sp_dp_x_p0pct.d12", "si_sp_dp_x_p1pct.d12"]
    assert "ELASTIC" not in (tmp_path / "si_sp_dp_x_p0pct.d12").read_text().splitlines()
    r = _mace("tau", "strain", "si_sp.d12", cwd=tmp_path)
    assert r.returncode == 2 and "exists" in r.stderr


def test_a_refused_strain_set_writes_nothing(tmp_path):
    """|eps| > 0.05 or an existing deck refuses the whole set before the first deck is written, so the
    corrected retry is not refused for decks the failed run left behind."""
    shutil.copy(SI_DECK, tmp_path / "si_sp.d12")
    r = _mace("tau", "strain", "si_sp.d12", "--axis", "z", "--strains=-0.01,0.01,0.09", cwd=tmp_path)
    assert r.returncode == 2 and "linear regime" in r.stderr
    assert not list(tmp_path.glob("si_sp_dp_z_*.d12"))
    r = _mace("tau", "strain", "si_sp.d12", "--axis", "z", "--strains=-0.01,0.01", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(list(tmp_path.glob("si_sp_dp_z_*.d12"))) == 3
    # one existing deck: nothing new is written either
    (tmp_path / "si_sp_dp_z_m1pct.d12").unlink()
    with pytest.raises(strain.StrainError, match="exists"):
        strain.write_strain_set(tmp_path / "si_sp.d12", "z", [-0.01, 0.01])
    assert not (tmp_path / "si_sp_dp_z_m1pct.d12").exists()
    # the GUI's check_write runs on every destination before any is written
    seen = []

    def check(p):
        seen.append(p.name)
        if p.name.endswith("p1pct.d12"):
            raise PermissionError("no")
    with pytest.raises(PermissionError):
        strain.write_strain_set(tmp_path / "si_sp.d12", "y", [-0.01, 0.01], check=check)
    assert seen == ["si_sp_dp_y_m1pct.d12", "si_sp_dp_y_p0pct.d12", "si_sp_dp_y_p1pct.d12"]
    assert not list(tmp_path.glob("si_sp_dp_y_*.d12"))


# ------------------------------------------------------------------ fits on real outputs
def test_the_real_series_reads():
    assert len(SERIES) == 5
    pts = {round(p["strain"], 4): p for p in (strain.read_point(f) for f in SERIES)}
    assert sorted(pts) == [-0.01, -0.005, 0.0, 0.005, 0.01]
    assert {p["axis"] for p in pts.values()} == {"x", None}
    zero = pts[0.0]
    assert zero["size"] == pytest.approx(11.135747) and pts[0.01]["size"] == pytest.approx(11.247104)
    assert zero["vbm_ev"] is not None and zero["cbm_ev"] - zero["vbm_ev"] > 5      # diamond, B3LYP


def test_fitted_c_matches_the_elastcon_c11():
    """The clamped-ion C of the strain series against ELASTCON's C11 (1150.4 GPa) of the same diamond."""
    ft = strain.fit([strain.read_point(f) for f in SERIES], "hole")
    assert ft["C_unit"] == "GPa" and ft["C"] == pytest.approx(1150.4, rel=0.01)
    assert ft["C_r2"] > 0.999
    # diamond's triply degenerate VBM splits under uniaxial strain: flagged, not hidden
    assert ft["E1_r2"] < 0.95 and ft["warnings"]


def test_mace_tau_fit_with_c_from_elastcon(tmp_path):
    for f in SERIES:
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(HPCC / "mechanics" / "dia_elastcon.out", tmp_path)
    r = _mace("tau", "fit", *sorted(p.name for p in tmp_path.glob("dia_sp_dp_x_*.out")), "--carrier", "electron",
              "--m-star", "0.48", "--c-from", "dia_elastcon.out", "--json", "--save", "tau.json", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["inputs"]["C"]["value"] == pytest.approx(1150.388)
    assert "ELASTCON" in res["inputs"]["C"]["source"] and "linear fit of the CBM" in res["inputs"]["E1"]["source"]
    assert json.loads((tmp_path / "tau.json").read_text())["tau_fs"] == pytest.approx(res["tau_fs"])
    r = _mace("tau", "fit", SERIES[0].name, SERIES[1].name, "--carrier", "hole", "--m-star", "0.5", cwd=tmp_path)
    assert r.returncode == 2 and "unstrained" in r.stderr


def test_a_saved_tau_feeds_the_total_zt_with_its_provenance(tmp_path):
    from mace.utils import transport_zt as Z
    res = dpt.estimate(3, 0.3, 5.0, 100.0, 300, carrier="electron", axis="x")
    path = tmp_path / "tau.json"
    path.write_text(json.dumps(res))
    tau = Z.check_tau_fs(str(path))
    assert tau == pytest.approx(res["tau_fs"])
    used, scale, how = Z.resolve_tau(10.0, tau, "x.out")
    assert used == tau and scale == pytest.approx(tau / 10.0)
    assert how.startswith("tau.json: deformation-potential theory") and "(electron x)" in how
    assert Z.resolve_tau(10.0, 20.0)[2].startswith("given;")
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(Z.KappaLatticeError, match="no tau_fs"):
        Z.check_tau_fs(str(tmp_path / "bad.json"))
