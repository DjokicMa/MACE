"""`mace phonopy collect` on phonopy's own NaCl-CRYSTAL example (needs phonopy).

The fixture is a trimmed copy of phonopy v4.7.2 example/NaCl-CRYSTAL: the
parent, and the two CRYSTAL outputs phonopy's authors ran for the 4x4x4
displacements (see tests/fixtures/phonopy_crystal/README.md).

Reference values were produced from the UNTRIMMED upstream files with the
phonopy 4.7.2 command line, no NAC:
    phonopy-init --crystal --dim="4 4 4" -d
    phonopy-init --crystal -f supercell-001.o supercell-002.o
    phonopy --qpoints 0 0 0 1/2 0 1/2 1/2 1/2 1/2
    phonopy -t --mesh 100 --tmin 300 --tmax 300
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from conftest import REPO_ROOT  # noqa: E402

from mace.phonopy_bridge import cli  # noqa: E402
from mace.phonopy_bridge import collect as C  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "NaCl"
MACE_CLI = REPO_ROOT / "mace_cli"

# phonopy CLI (see module docstring), THz.
REF_FREQ = {
    (0.0, 0.0, 0.0): [0.0, 0.0, 0.0, 6.1229209434, 6.1229209434, 6.1229209434],
    (0.5, 0.0, 0.5): [2.7403636769, 2.7403636769, 4.6976646612, 6.2005583238, 6.2005583238,
                      6.3972968819],
    (0.5, 0.5, 0.5): [4.0691650560, 4.0691650560, 5.0288751961, 5.0288751961, 5.6437419408,
                      7.3878479140],
}
REF_CV_300 = 47.1469502          # J/K/mol, mesh length 100 (31x31x31)


def _setup(tmp_path, monkeypatch, outputs=True):
    for f in ("parent.out", "parent.d12", "crystal.o", "crystal.d12"):
        shutil.copy(FIX / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "parent.out", "--dim", "4 4 4"]) == 0
    d = tmp_path / "phonopy_parent_444"
    man = json.loads((d / "mace_phonopy.json").read_text())
    if outputs:
        for i, job in enumerate(man["jobs"], start=1):
            shutil.copy(FIX / f"supercell-{i:03d}.o", d / f"{job}.out")
    return d, man


def _oracle():
    """The same calculation through phonopy's API, straight from the fixtures."""
    from phonopy import Phonopy
    from phonopy.interface.crystal import parse_set_of_forces, read_crystal

    cell, _ = read_crystal(str(FIX / "parent.out"))
    ph = Phonopy(cell, supercell_matrix=np.diag([4, 4, 4]), primitive_matrix="P",
                 calculator="crystal")
    ph.generate_displacements()
    ph.forces = parse_set_of_forces(128, [str(FIX / "supercell-001.o"),
                                          str(FIX / "supercell-002.o")], verbose=False)
    ph.produce_force_constants()
    ph.symmetrize_force_constants(use_symfc_projector=True)
    return ph


def _freqs(ph, q):
    ph.run_qpoints([list(q)])
    return ph.qpoints.frequencies[0]


# ---------------------------------------------------------------------------
# T7: physics against phonopy's API and its CLI
# ---------------------------------------------------------------------------

def test_forces_fc_frequencies_and_thermal_match_phonopy(tmp_path, monkeypatch):
    d, _ = _setup(tmp_path, monkeypatch)
    assert cli.main(["collect", str(d), "--band", "none"]) == 0
    ours = phonopy.load(str(d / "phonopy_params.yaml"), is_compact_fc=False)
    ref = _oracle()
    np.testing.assert_allclose(np.array(ours.dataset["first_atoms"][0]["forces"]),
                               np.array(ref.dataset["first_atoms"][0]["forces"]), atol=1e-12)
    np.testing.assert_allclose(ours.force_constants, ref.force_constants, atol=1e-10)
    for q, want in REF_FREQ.items():
        # 1e-6 THz: the acoustic modes at Gamma are sqrt of ~1e-14 noise.
        np.testing.assert_allclose(_freqs(ours, q), _freqs(ref, q), atol=1e-6)
        np.testing.assert_allclose(_freqs(ours, q), want, atol=1e-6)
    import yaml
    tp = yaml.safe_load((d / "thermal_properties.yaml").read_text())
    row = [r for r in tp["thermal_properties"] if abs(r["temperature"] - 300) < 1e-9][0]
    ref.run_mesh(100.0)
    ref.run_thermal_properties(t_min=300, t_max=300)
    assert abs(row["heat_capacity"] - ref.get_thermal_properties_dict()["heat_capacity"][0]) < 1e-6
    assert abs(row["heat_capacity"] - REF_CV_300) < 1e-6


def test_force_sets_equal_phonopys(tmp_path, monkeypatch):
    from phonopy.file_IO import write_FORCE_SETS

    d, _ = _setup(tmp_path, monkeypatch)
    assert cli.main(["collect", str(d), "--band", "none"]) == 0
    ref = _oracle()
    write_FORCE_SETS(ref.dataset, filename=str(tmp_path / "FORCE_SETS.ref"))
    assert (d / "FORCE_SETS").read_text() == (tmp_path / "FORCE_SETS.ref").read_text()


def test_yaml_outputs_exist_and_plot(tmp_path, monkeypatch):
    d, _ = _setup(tmp_path, monkeypatch)
    assert cli.main(["collect", str(d), "--band",
                     "0 0 0  1/2 0 1/2  1/2 1/4 3/4, 1/2 1/2 1/2  0 0 0",
                     "--band-labels", "G X W L G", "--mesh", "8 8 8", "--plot"]) == 0
    for f in ("band.yaml", "mesh.yaml", "total_dos.dat", "thermal_properties.yaml",
              "phonopy_params.yaml", "FORCE_SETS", "band_dos.pdf"):
        assert (d / f).is_file(), f


# ---------------------------------------------------------------------------
# T8: collect refuses unusable outputs, naming each file
# ---------------------------------------------------------------------------

def _check(d, capsys):
    rc = cli.main(["collect", str(d), "--check"])
    out = capsys.readouterr()
    return rc, out.out + out.err


def test_check_passes_on_good_outputs(tmp_path, monkeypatch, capsys):
    d, _ = _setup(tmp_path, monkeypatch)
    rc, text = _check(d, capsys)
    assert rc == 0 and "All 2 output(s) usable" in text
    assert not (d / "FORCE_SETS").exists()


def test_missing_output(tmp_path, monkeypatch, capsys):
    d, man = _setup(tmp_path, monkeypatch)
    (d / f"{man['jobs'][1]}.out").unlink()
    rc, text = _check(d, capsys)
    assert rc == 1 and f"{man['jobs'][1]}.out: PROBLEM" in text and "no output" in text


def test_truncated_output(tmp_path, monkeypatch, capsys):
    d, man = _setup(tmp_path, monkeypatch)
    f = d / f"{man['jobs'][0]}.out"
    text = f.read_text()
    f.write_text(text[:text.index(" CARTESIAN FORCES")])
    rc, text = _check(d, capsys)
    assert rc == 1 and f"{man['jobs'][0]}.out: PROBLEM" in text
    assert "terminate normally" in text and "force block has 0 rows" in text


def test_unconverged_scf(tmp_path, monkeypatch, capsys):
    d, man = _setup(tmp_path, monkeypatch)
    f = d / f"{man['jobs'][0]}.out"
    f.write_text(f.read_text().replace("SCF ENDED - CONVERGENCE ON ENERGY",
                                       "SCF ENDED - TOO MANY CYCLES"))
    rc, text = _check(d, capsys)
    assert rc == 1 and f"{man['jobs'][0]}.out: PROBLEM" in text and "SCF did not converge" in text


def test_output_of_another_displacement_is_caught(tmp_path, monkeypatch, capsys):
    """Converged forces for the wrong geometry (a stale job script that did not
    stage the .ext) would fit silently wrong force constants."""
    d, man = _setup(tmp_path, monkeypatch)
    a, b = (d / f"{j}.out" for j in man["jobs"])
    ta, tb = a.read_text(), b.read_text()
    a.write_text(tb)
    b.write_text(ta)
    rc, text = _check(d, capsys)
    assert rc == 1
    for j in man["jobs"]:
        assert f"{j}.out: PROBLEM" in text
    assert "geometry is not this displacement" in text and "--overwrite-sh" in text


# ---------------------------------------------------------------------------
# T10: --band auto lands in CRYSTAL's primitive basis
# ---------------------------------------------------------------------------

def test_band_auto_points_in_crystal_primitive_basis(tmp_path, monkeypatch):
    pytest.importorskip("seekpath")
    d, _ = _setup(tmp_path, monkeypatch)
    ph = phonopy.load(str(d / "phonopy_disp.yaml"), produce_fc=False)
    paths, labels = C.auto_band(ph.primitive)
    flat = [tuple(p) for seg in paths for p in seg]
    assert len(flat) == len(labels)
    x = flat[labels.index("X")]
    # fcc primitive (0,a/2,a/2),(a/2,0,a/2),(a/2,a/2,0): X = (1/2, 0, 1/2) in its basis,
    # i.e. one cubic axis at 1/a in cartesian reciprocal space.
    assert np.allclose(x, (0.5, 0.0, 0.5))
    rec = np.linalg.inv(np.array(ph.primitive.cell)).T
    cart = np.array(x) @ rec
    a = 5.56981957
    assert np.allclose(sorted(np.abs(cart)), [0, 0, 1 / a], atol=1e-6)
    assert cli.main(["collect", str(d)]) == 0
    assert (d / "band.yaml").is_file()


# ---------------------------------------------------------------------------
# T5: the real invocation path
# ---------------------------------------------------------------------------

def test_generate_and_collect_through_mace_cli(tmp_path):
    for f in ("parent.out", "parent.d12"):
        shutil.copy(FIX / f, tmp_path / f)
    env = dict(os.environ, MACE_NO_BANNER="1")
    p = subprocess.run([sys.executable, str(MACE_CLI), "phonopy", "generate", "parent.out",
                        "--dim", "4 4 4"], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "mace submit phonopy_parent_444 --walltime" in p.stdout
    d = tmp_path / "phonopy_parent_444"
    man = json.loads((d / "mace_phonopy.json").read_text())
    for i, job in enumerate(man["jobs"], start=1):
        shutil.copy(FIX / f"supercell-{i:03d}.o", d / f"{job}.out")
    p = subprocess.run([sys.executable, str(MACE_CLI), "phonopy", "collect",
                        "phonopy_parent_444", "--band", "none"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "thermal_properties.yaml").is_file()
    # phonopy itself, never something in this repository, must be what was imported.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path[:0] = [{!r}, {!r}, {!r}, {!r}]; import phonopy; "
         "print(phonopy.__file__)".format(str(REPO_ROOT), str(REPO_ROOT / "mace"),
                                          str(REPO_ROOT / "Crystal_d12"),
                                          str(REPO_ROOT / "Crystal_d3"))],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert REPO_ROOT not in Path(probe.stdout.strip()).resolve().parents, probe.stdout
