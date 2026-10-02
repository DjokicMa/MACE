"""Decks MACE writes without any charge option stay byte-identical.

The golden files under tests/data/default_decks/ were written by 1.1.4 (main
f8df25a4) before the charged-systems work started, from the synthetic inputs
below. The ion-charge options are opt-in; nothing here asks for them, so these
decks must not change. Regenerate (only for a deliberate, listed deck change)
with MACE_REGEN_GOLDEN=1.
"""
import json
import os
import subprocess
import sys

import pytest

from basis_library import read_record
from conftest import REPO_ROOT
from ion_fixtures import cluster, ec, fsi, li_near, molecule_cif, molecule_deck, p1_cif, shuffled, synth_out

MACE_CLI = REPO_ROOT / "mace_cli"
GOLDEN = REPO_ROOT / "tests" / "data" / "default_decks"
REGEN = os.environ.get("MACE_REGEN_GOLDEN") == "1"


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(k, None)
    return env


def _check(name, text):
    # .golden, not .d12: the committed-sample sweeps (test_committed_samples)
    # read every tests/data/**/*.d12 as a real CRYSTAL deck
    path = GOLDEN / (name[:-len(".d12")] + ".golden")
    if REGEN:
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.exists(), f"missing golden {name}"
    assert text == path.read_text(), f"{name} changed"


def li_fsi_ec():
    anion = fsi()
    solvent = ec()
    return shuffled(cluster((anion, (0.0, 0.0, 0.0)), (solvent, (8.0, 0.0, 0.0)))
                    + [li_near(anion, 2)])


def _convert(tmp_path, cif_name, cif_text, options):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / cif_name).write_text(cif_text)
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps(options))
    run = subprocess.run(
        [sys.executable, str(MACE_CLI), "convert", "--no-banner", "--batch", "--options_file", str(opts),
         "--cif_dir", str(cifs), "--output_dir", str(tmp_path / "out")],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=_env())
    assert run.returncode == 0, run.stdout + run.stderr
    decks = sorted((tmp_path / "out").glob("*.d12"))
    assert len(decks) == 1, run.stdout
    return decks[0].read_text()


MOLECULE_OPTIONS = {"calculation_type": "SP", "method": "DFT", "dft_functional": "HSESOL3C",
                    "basis_set": "SOLDEF2MSVP", "basis_set_type": "INTERNAL", "dimensionality": "MOLECULE",
                    "symmetry_handling": "P1", "write_only_unique": False,
                    "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 9}}


def test_cif2d12_molecule_deck_is_unchanged(tmp_path):
    pytest.importorskip("ase")
    deck = _convert(tmp_path, "li_fsi_ec.cif", molecule_cif(li_fsi_ec(), name="li_fsi_ec"), MOLECULE_OPTIONS)
    _check("cif2d12_molecule.d12", deck)


def test_cif2d12_explicit_basis_crystal_deck_is_unchanged(tmp_path):
    pytest.importorskip("ase")
    options = {"calculation_type": "SP", "method": "DFT", "dft_functional": "PBE0",
               "basis_set": str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta") + "/",
               "basis_set_type": "EXTERNAL", "dimensionality": "CRYSTAL", "symmetry_handling": "P1",
               "write_only_unique": False, "dft_grid": "XLGRID",
               "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}}
    cif = p1_cif((4.0, 4.0, 4.0, 90.0, 90.0, 90.0), [("Li", (0.0, 0.0, 0.0)), ("F", (0.5, 0.5, 0.5))],
                 name="lif_p1")
    _check("cif2d12_crystal_explicit.d12", _convert(tmp_path, "lif_p1.cif", cif, options))


LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]


def _opt2d12(tmp_path, deck_text, out_text):
    (tmp_path / "parent.d12").write_text(deck_text)
    (tmp_path / "parent.out").write_text(out_text)
    run = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out", "--calc-type", "SP",
         "--non-interactive", "--output-dir", "o"],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=_env())
    assert run.returncode == 0, run.stdout + run.stderr
    return next((tmp_path / "o").glob("*.d12")).read_text()


def test_opt2d12_sp_of_a_neutral_parent_is_unchanged(tmp_path):
    deck = molecule_deck(LI_WATER, title="parent", spin=False)
    out = synth_out(LI_WATER, title="parent")
    _check("opt2d12_sp_neutral.d12", _opt2d12(tmp_path, deck, out))


def test_opt2d12_sp_of_a_charged_internal_parent_is_unchanged(tmp_path):
    deck = molecule_deck(LI_WATER, title="parent").replace(
        "SOLDEF2MSVP\n", "SOLDEF2MSVP\nCHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\n").replace(
        "ENDDFT\n", "ENDDFT\nDOPING\n-1\n")
    out = synth_out(LI_WATER, title="parent")
    _check("opt2d12_sp_charged_internal.d12", _opt2d12(tmp_path, deck, out))


TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"


def _explicit_block():
    blocks = "".join(read_record(str(TZ), z) for z in (1, 3, 8))
    return f"END\n{blocks}99 0\n"


def test_opt2d12_sp_of_a_charged_explicit_parent_is_unchanged(tmp_path):
    basis = _explicit_block()
    deck = molecule_deck(LI_WATER, title="parent", basis=basis + "CHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\nCHARGED\nEND\n",
                         dft="PBE0\nXLGRID\n")
    out = synth_out(LI_WATER, title="parent", header_electrons=12)
    _check("opt2d12_sp_charged_explicit.d12", _opt2d12(tmp_path, deck, out))
