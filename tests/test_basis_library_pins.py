"""Every bundled external basis record, and every deck MACE writes from one,
stays byte-identical.

The pins (tests/data/basis_library_pins.json, regenerated only by
tests/make_basis_library_pins.py) were taken from the per-element basis
directories: the sha256 of every record file, what read_basis_file returns and
warns for Z 1..99 in both external families, the external coverage check for
Z 1..100, and the decks cif2d12, d12_writer and opt2d12 write from them. They
hold however the records are stored.
"""
import contextlib
import io
import json
import os
import types

import pytest

import make_basis_library_pins as pins_mod
from conftest import REPO_ROOT

PINS = json.loads(pins_mod.PINS.read_text())


@pytest.mark.parametrize("family", pins_mod.FAMILIES)
def test_record_bytes_match_pins(family):
    got = {k: pins_mod.sha(b) for k, b in pins_mod.record_bytes(family).items()}
    assert got == PINS["files"][family]


def test_stuttgart_pins_cover_the_odd_records():
    """The SDF record and the records without a final newline are pinned too."""
    stut = PINS["files"]["stuttgart"]
    assert "231.SDF" in stut and len(stut) == 119
    records = pins_mod.record_bytes("stuttgart")
    assert not records["1"].endswith(b"\n")
    assert records["231"].endswith(b"\n")


def test_read_basis_file_matches_pins():
    assert pins_mod.pin_read_basis_file() == PINS["read_basis_file"]


def test_external_compatibility_matches_pins():
    assert pins_mod.pin_compatibility() == PINS["compatibility"]


def test_writer_basis_blocks_match_pins():
    got = pins_mod.pin_writer_blocks()
    assert got == {k: v for k, v in PINS["decks"].items() if k.startswith("d12_writer/")}


def test_opt2d12_external_tz_deck_matches_pin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = pins_mod.pin_opt2d12_deck(tmp_path)
    assert got == {k: v for k, v in PINS["decks"].items() if k.startswith("opt2d12/")}


def test_cif2d12_decks_match_pins(tmp_path):
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    got = pins_mod.pin_cif_decks(tmp_path)
    want = {k: v for k, v in PINS["decks"].items() if k.startswith("cif2d12/")}
    assert sorted(got) == sorted(want), "deck filenames changed"
    assert got == want


# --- workflow executor: deck filename -> basis directory -----------------------

def _resolve(name):
    from mace.workflow.executor import WorkflowExecutor

    return WorkflowExecutor._extract_basis_path_from_d12(types.SimpleNamespace(), name)


@pytest.fixture
def no_basis_env(monkeypatch):
    for var in ("MACE_TZ_BASIS_PATH", "MACE_DZ_BASIS_PATH", "MACE_BASIS_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_executor_maps_bundled_family_names(no_basis_env):
    import mace_config

    assert _resolve("x_full.basis.triplezeta.d12") == mace_config.DEFAULT_TZ_PATH
    assert _resolve("x_full.basis.doublezeta.d12") == mace_config.DEFAULT_DZ_PATH


def test_executor_env_override_still_wins(no_basis_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_TZ_BASIS_PATH", str(tmp_path))
    assert _resolve("x_full.basis.triplezeta.d12") == str(tmp_path)
