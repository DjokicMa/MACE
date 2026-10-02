"""`mace phonopy generate` on phonopy's own Si-CRYSTAL example (needs phonopy).

The fixture is a trimmed copy of phonopy v4.7.2 example/Si-CRYSTAL (see
tests/fixtures/phonopy_crystal/README.md). The oracle for the geometry is
phonopy's own CRYSTAL writer, run on the same parent in a scratch directory:
MACE must hand CRYSTAL exactly the fort.34 phonopy would.
"""
import json
import shutil

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from conftest import REPO_ROOT  # noqa: E402

from mace.phonopy_bridge import cli  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal"

INTERNAL_SI = """si_internal
CRYSTAL
0 0 0
227
5.44897370
1
14  .125  .125   .125
BASISSET
POB-TZVP-REV2
DFT
PBE0
END
SHRINK
8 8
TOLINTEG
8 8 8 8 16
TOLDEE
9
END
"""


def _parent(tmp_path, kind):
    shutil.copy(FIX / "Si" / "parent.out", tmp_path / "parent.out")
    if kind == "explicit":
        shutil.copy(FIX / "Si" / "parent.d12", tmp_path / "parent.d12")
    else:
        (tmp_path / "parent.d12").write_text(INTERNAL_SI)
    return tmp_path / "parent.out"


def _generate(tmp_path, monkeypatch, kind="explicit", *extra):
    _parent(tmp_path, kind)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "parent.out", "--dim", "2 2 2", *extra]) == 0
    return tmp_path / "phonopy_parent_222"


def _oracle(tmp_path, monkeypatch):
    """phonopy's write_supercells_with_displacements, in its own directory."""
    from phonopy import Phonopy
    from phonopy.interface.crystal import read_crystal, write_supercells_with_displacements

    odir = tmp_path / "oracle"
    odir.mkdir()
    monkeypatch.chdir(odir)
    cell, conv = read_crystal(str(FIX / "Si" / "parent.out"))
    ph = Phonopy(cell, supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="P",
                 calculator="crystal")
    ph.generate_displacements()
    write_supercells_with_displacements(
        ph.supercell, ph.supercells_with_displacements,
        list(range(1, len(ph.supercells_with_displacements) + 1)), conv, 8,
        template_file="NO_TEMPLATE")
    return odir


def test_ext_is_byte_identical_to_phonopys_writer(tmp_path, monkeypatch, capsys):
    out = _generate(tmp_path, monkeypatch)
    odir = _oracle(tmp_path, monkeypatch)
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["jobs"], "no displacement written"
    for i, job in enumerate(man["jobs"], start=1):
        assert (out / f"{job}.ext").read_bytes() == (odir / f"supercell-{i:03d}.ext").read_bytes()


def test_conventional_numbers_match_the_elements(tmp_path, monkeypatch):
    out = _generate(tmp_path, monkeypatch)
    man = json.loads((out / "mace_phonopy.json").read_text())
    ph = phonopy.load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    assert [c % 100 for c in man["conv_numbers_supercell"]] == list(ph.supercell.numbers)
    assert len(man["conv_numbers_supercell"]) == man["n_atoms_supercell"] == 16


@pytest.mark.parametrize("kind", ["explicit", "internal"])
def test_deck_structure(tmp_path, monkeypatch, kind):
    out = _generate(tmp_path, monkeypatch, kind)
    man = json.loads((out / "mace_phonopy.json").read_text())
    for job in man["jobs"]:
        lines = (out / f"{job}.d12").read_text().splitlines()
        assert lines[1] == "EXTERNAL"
        if kind == "internal":
            assert lines[2:4] == ["BASISSET", "POB-TZVP-REV2"]
        else:
            assert lines[2] == "END" and lines[3] == "14 7"
        keys = [ln.strip() for ln in lines]
        assert keys.count("TOLDEE") == 1 and keys.count("GRADCAL") == 1
        assert keys[-1] == "END" and keys[-2] == "GRADCAL"
        for banned in ("OPTGEOM", "FREQCALC", "ELASTCON", "EOS"):
            assert banned not in keys
        assert keys[keys.index("TOLDEE") + 1] == "10"


def test_unique_job_names_and_reloadable_records(tmp_path, monkeypatch):
    out = _generate(tmp_path, monkeypatch, "explicit", "--pm", "yes")
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert len(man["jobs"]) == 2 and len(set(man["jobs"])) == 2
    assert all(j.startswith("parent_ph222_disp-") for j in man["jobs"])
    ph = phonopy.load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    assert len(ph.supercells_with_displacements) == len(man["jobs"])
    assert man["phonopy_version"] == phonopy.__version__
    assert len(man["parent_out_sha256"]) == 64
    titles = [(out / f"{j}.d12").read_text().splitlines()[0] for j in man["jobs"]]
    assert titles[0].endswith("phonopy displacement 001/002")


def test_existing_output_dir_is_not_overwritten(tmp_path, monkeypatch, capsys):
    _generate(tmp_path, monkeypatch)
    assert cli.main(["generate", "parent.out", "--dim", "2 2 2"]) == 1
    assert "not empty" in capsys.readouterr().err
    assert cli.main(["generate", "parent.out", "--dim", "2 2 2", "--force"]) == 0


def test_output_dir_inside_workflow_outputs_is_refused(tmp_path, monkeypatch, capsys):
    _parent(tmp_path, "explicit")
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["generate", "parent.out", "--dim", "2 2 2",
                   "--output-dir", "workflow_outputs/step_002_PH/x"])
    assert rc == 1 and "workflow_outputs" in capsys.readouterr().err


def test_refused_parent_writes_nothing(tmp_path, monkeypatch, capsys):
    _parent(tmp_path, "explicit")
    (tmp_path / "parent.d12").write_text(
        (tmp_path / "parent.d12").read_text().replace("CRYSTAL\n", "SLAB\n", 1))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "parent.out", "--dim", "2 2 2"]) == 1
    assert "[NOT_3D]" in capsys.readouterr().err
    assert not (tmp_path / "phonopy_parent_222").exists()


def test_runs_without_contextlib_chdir(tmp_path, monkeypatch):
    """contextlib.chdir is Python 3.11+; MACE documents 3.7+, so generate and
    collect must not need it."""
    import contextlib
    monkeypatch.delattr(contextlib, "chdir", raising=False)
    out = _generate(tmp_path, monkeypatch)
    assert (out / "phonopy_disp.yaml").is_file()
