"""Regression tests: an aborted write_d12_file must not leave a truncated deck.

The basis-set compatibility check runs mid-write (after the geometry section is
already on disk). The non-interactive abort branch used to plain `return`: the
partial .d12 (no SCF block, no END) stayed on disk and process_files still
reported "Successfully created" and returned True — so an unattended workflow
would submit a garbage deck to SLURM. write_d12_file now removes the partial
file and returns False, and process_files propagates the failure.
"""
import sys
from pathlib import Path

import pytest

import CRYSTALOptToD12 as M
from d12_parsers import CrystalOutputParser

from conftest import find_data


class _NoTTY:
    """Stand-in stdin that is definitively not a terminal."""

    def isatty(self):
        return False


@pytest.fixture()
def real_geometry():
    return CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()


def test_noninteractive_incompatible_basis_aborts_cleanly(tmp_path, monkeypatch):
    """Forced-incompatible internal basis + non-interactive stdin: the abort
    must return False and leave NO file behind (previously: truncated deck)."""
    geo = CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()
    settings = dict(geo)
    settings["basis_set_type"] = "INTERNAL"

    monkeypatch.setattr(M, "check_basis_set_compatibility",
                        lambda *a, **k: (False, [6]))
    monkeypatch.setattr(sys, "stdin", _NoTTY())

    out = tmp_path / "continuation.d12"
    result = M.write_d12_file(str(out), geo, settings)

    assert result is False
    assert not out.exists(), "aborted write left a truncated .d12 on disk"


def test_interactive_decline_aborts_cleanly(tmp_path, monkeypatch, real_geometry):
    """Interactive 'no' at the continue-anyway prompt: same contract."""
    settings = dict(real_geometry)
    settings["basis_set_type"] = "INTERNAL"

    monkeypatch.setattr(M, "check_basis_set_compatibility",
                        lambda *a, **k: (False, [6]))

    class _TTY:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(M, "yes_no_prompt", lambda *a, **k: False)

    out = tmp_path / "continuation.d12"
    result = M.write_d12_file(str(out), geo := real_geometry, settings)

    assert result is False
    assert not out.exists()


def test_compatible_write_returns_true_and_complete_deck(tmp_path, real_geometry):
    """The success path must return True and produce a complete deck (ends with
    END) — this is what process_files now keys 'Successfully created' on."""
    settings = dict(real_geometry)

    out = tmp_path / "continuation.d12"
    result = M.write_d12_file(str(out), real_geometry, settings)

    assert result is True
    assert out.exists()
    text = out.read_text().rstrip()
    assert text.endswith("END"), "complete deck must terminate with END"
