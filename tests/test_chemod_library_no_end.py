"""No END after CHEMOD when the basis is a BASISSET library.

Measured on CRYSTAL23 (HPCC, 2026-10-02): with "BASISSET <library>" there is
no basis-set block - the lines after the library name already belong to block
3, and CRYSTAL reads CHEMOD there ("ELECTRONIC CONFIGURATION MODIFIED IN n
ATOMS"). An END closing CHEMOD ends block 3 instead: the DFT block, TOLINTEG
and everything after it are silently ignored and the run falls back to HF
with no error. (After an explicit basis's "99 0", CHEMOD ... END is right.)

Every MACE path that writes CHEMOD after a library name - cif2d12, opt2d12
(carried or recomputed) and mace ions - must leave no END there, and a deck
that already has one is refused or reported rather than passed on.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from charge_records import carry_charge_records, library_basis_end
from ion_charges import apply_charge_setup, plan_for_deck, IonChargeError
from ion_fixtures import cluster, ec, fsi, li_near, molecule_cif, molecule_deck, synth_out

MACE_CLI = REPO_ROOT / "mace_cli"
DECKS = Path(__file__).resolve().parent / "data" / "default_decks"

LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]
CHEMOD_LI = "CHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\n"
# the measured failure: CHEMOD closed with END after the library name
BAD_PARENT = molecule_deck(LI_WATER, title="parent").replace(
    "SOLDEF2MSVP\n", "SOLDEF2MSVP\n" + CHEMOD_LI + "END\n").replace("ENDDFT\n", "ENDDFT\nDOPING\n-1\n")


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(k, None)
    return env


def _run(args, cwd):
    return subprocess.run([sys.executable, str(MACE_CLI)] + args, cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300, env=_env())


def _assert_no_end_after_library(deck):
    lines = deck.splitlines()
    assert library_basis_end(lines) is None, deck
    i = [l.strip().upper() for l in lines].index("BASISSET")
    after = [l.strip().upper() for l in lines[i + 2:]]
    # the next keyword after the CHEMOD/CHARGED records is a block-3 one
    j = 0
    if after and after[0] == "CHEMOD":
        j = 2 + int(after[1])
    if j < len(after) and after[j] == "CHARGED":
        j += 1
    assert after[j] != "END", deck


def test_the_measured_failure_shape_is_found():
    assert library_basis_end(BAD_PARENT.splitlines()) == BAD_PARENT.splitlines().index("END")
    good = BAD_PARENT.replace(CHEMOD_LI + "END\n", CHEMOD_LI)
    assert library_basis_end(good.splitlines()) is None
    # an explicit basis closes its input with END after "99 0": that END is right
    explicit = molecule_deck(LI_WATER, basis="3 1\n0 0 1 2.0 1.0\n1.0 1.0\n99 0\n" + CHEMOD_LI
                             + "CHARGED\nEND\n")
    assert library_basis_end(explicit.splitlines()) is None


def test_pinned_library_decks_have_no_end_after_chemod():
    deck = (DECKS / "opt2d12_sp_charged_internal.golden").read_text()
    assert "CHEMOD" in deck
    _assert_no_end_after_library(deck)


def test_apply_charge_setup_writes_no_end_after_chemod():
    atoms = cluster((fsi(), (0.0, 0.0, 0.0)), (ec(), (8.0, 0.0, 0.0)))
    atoms.append(li_near(fsi(), 2))
    deck = molecule_deck(atoms)
    setup = plan_for_deck(deck, ion_charges="auto")
    out = apply_charge_setup(deck, setup)
    assert "CHEMOD" in out and "DOPING" not in out  # Li+ FSI- is neutral
    _assert_no_end_after_library(out)


def test_apply_charge_setup_refuses_a_deck_with_the_stray_end():
    deck = molecule_deck(LI_WATER).replace("SOLDEF2MSVP\n", "SOLDEF2MSVP\nEND\n")
    with pytest.raises(IonChargeError, match="END right after the BASISSET"):
        apply_charge_setup(deck, plan_for_deck(deck.replace("SOLDEF2MSVP\nEND\n", "SOLDEF2MSVP\n"),
                                               net_charge=1))


def test_mace_ions_replace_does_not_keep_the_stray_end(tmp_path):
    """--replace strips CHEMOD; the END that closed it must not survive."""
    deck = tmp_path / "parent.d12"
    deck.write_text(BAD_PARENT)
    run = _run(["ions", str(deck), "--net-charge", "1", "--replace", "-o", "out"], tmp_path)
    written = tmp_path / "out" / "parent.d12"
    assert run.returncode != 0, run.stdout + run.stderr
    assert not written.exists()
    assert "END right after the BASISSET" in run.stderr


def test_carried_records_from_a_bad_parent_are_reported():
    child = molecule_deck(LI_WATER, title="child")
    out, notes = carry_charge_records(child, BAD_PARENT)
    _assert_no_end_after_library(out)
    assert any(n.startswith("WARNING") and "ignored" in n and "DFT" in n for n in notes), notes


def test_opt2d12_child_of_a_charged_library_parent_has_no_end_after_chemod(tmp_path):
    parent = molecule_deck(LI_WATER, title="parent").replace(
        "SOLDEF2MSVP\n", "SOLDEF2MSVP\n" + CHEMOD_LI).replace("ENDDFT\n", "ENDDFT\nDOPING\n-1\n")
    (tmp_path / "parent.d12").write_text(parent)
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent"))
    run = _run(["opt2d12", "--out-file", "parent.out", "--calc-type", "SP", "--non-interactive",
                "--output-dir", "o"], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    child = next((tmp_path / "o").glob("*.d12")).read_text()
    assert "CHEMOD" in child
    _assert_no_end_after_library(child)


def test_cif2d12_ion_charges_write_no_end_after_chemod(tmp_path):
    pytest.importorskip("ase")
    atoms = cluster((fsi(), (0.0, 0.0, 0.0)), (ec(), (8.0, 0.0, 0.0)))
    atoms.append(li_near(fsi(), 2))
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    (cifs / "lifsi.cif").write_text(molecule_cif(atoms, name="lifsi"))
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({
        "calculation_type": "SP", "method": "DFT", "dft_functional": "HSESOL3C",
        "basis_set": "SOLDEF2MSVP", "basis_set_type": "INTERNAL", "dimensionality": "MOLECULE",
        "symmetry_handling": "P1", "write_only_unique": False, "ion_charges": "auto"}))
    run = _run(["convert", "--no-banner", "--batch", "--options_file", str(opts), "--cif_dir", str(cifs),
                "--output_dir", str(tmp_path / "out")], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    deck = next((tmp_path / "out").glob("*.d12")).read_text()
    assert "CHEMOD" in deck
    _assert_no_end_after_library(deck)


# CRYSTAL reads each CHEMOD atom record free-format, so the label may stand on
# a line of its own with the shell charges on the next one; the decks of the
# HPCC measurement above were written that way (accepted: "ELECTRONIC
# CONFIGURATION MODIFIED IN 1 ATOMS").
CHEMOD_LI_TWO_LINE = "CHEMOD\n1\n1\n2.0 0.0 0.0 0.0 0.0\n"
BAD_PARENT_TWO_LINE = BAD_PARENT.replace(CHEMOD_LI, CHEMOD_LI_TWO_LINE)


def test_the_measured_failure_shape_is_found_with_the_label_on_its_own_line():
    lines = BAD_PARENT_TWO_LINE.splitlines()
    assert library_basis_end(lines) == lines.index("END")
    good = BAD_PARENT_TWO_LINE.replace(CHEMOD_LI_TWO_LINE + "END\n", CHEMOD_LI_TWO_LINE)
    assert library_basis_end(good.splitlines()) is None


def test_a_two_line_chemod_record_is_read_and_stripped_whole():
    from charge_records import extract_charge_records
    from ion_charges import strip_charge_records
    rec = extract_charge_records(BAD_PARENT_TWO_LINE)
    assert rec["CHEMOD"] == CHEMOD_LI_TWO_LINE.splitlines()
    stripped = strip_charge_records(BAD_PARENT_TWO_LINE)
    assert "2.0 0.0 0.0 0.0 0.0" not in stripped
    assert "SOLDEF2MSVP\nEND\n" in stripped  # the stray END stays visible to the guard


def test_mace_ions_replace_refuses_the_stray_end_after_a_two_line_chemod(tmp_path):
    deck = tmp_path / "parent.d12"
    deck.write_text(BAD_PARENT_TWO_LINE)
    run = _run(["ions", str(deck), "--net-charge", "1", "--replace", "-o", "out"], tmp_path)
    assert run.returncode != 0, run.stdout + run.stderr
    assert not (tmp_path / "out" / "parent.d12").exists()
    assert "END right after the BASISSET" in run.stderr
