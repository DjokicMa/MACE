"""Regression tests for SPINLOCK parse + round-trip.

The d12 input parser never read the SPINLOCK keyword, so a configured non-zero
fixed-spin lock was silently lost on OPT continuation / JSON-config reuse. These
tests verify the parser captures it and that it survives a parse -> writer
round-trip. A .d12 is an INPUT file, so editing a real one in tmp to vary the
lock value is a legitimate round-trip exercise (no synthetic CRYSTAL output).
"""
from pathlib import Path

import pytest

from d12_parsers import CrystalInputParser
from d12_writer import write_scf_section

from conftest import find_data


def _real_spinlock_d12():
    return find_data("OPT/*dia*opt_rev1.d12", must_contain="SPINLOCK")


def _with_spinlock_value(src: Path, tmp_path: Path, value_line: str) -> Path:
    """Copy a real d12, replacing the line after SPINLOCK with `value_line`."""
    lines = src.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "SPINLOCK":
            lines[i + 1] = value_line
            break
    out = tmp_path / "edited.d12"
    out.write_text("\n".join(lines) + "\n")
    return out


def test_spinlock_parsed_from_real_d12():
    """The real reference deck carries 'SPINLOCK / 0 50' — both values must now
    be captured (they were missing entirely before)."""
    d = CrystalInputParser(str(_real_spinlock_d12())).parse()
    assert d.get("spinlock") == 0
    assert d.get("spinlock_cycles") == 50


def test_nonzero_spinlock_sets_spin_polarized(tmp_path):
    edited = _with_spinlock_value(_real_spinlock_d12(), tmp_path, "2 50")
    d = CrystalInputParser(str(edited)).parse()
    assert d.get("spinlock") == 2
    assert d.get("spinlock_cycles") == 50
    assert d.get("spin_polarized") is True


def test_old_single_value_spinlock_form(tmp_path):
    edited = _with_spinlock_value(_real_spinlock_d12(), tmp_path, "3")
    d = CrystalInputParser(str(edited)).parse()
    assert d.get("spinlock") == 3


def test_nonzero_spinlock_survives_round_trip(tmp_path):
    """parse(real edited to 2 50) -> write_scf_section -> SPINLOCK 2 50 reappears."""
    edited = _with_spinlock_value(_real_spinlock_d12(), tmp_path, "2 50")
    d = CrystalInputParser(str(edited)).parse()
    out = tmp_path / "roundtrip.d12"
    with open(out, "w") as f:
        write_scf_section(
            f, tolerances=d.get("tolerances", {}), k_points=d.get("k_points", 8),
            dimensionality=d.get("dimensionality", "CRYSTAL"),
            use_smearing=d.get("use_smearing", False),
            smearing_width=d.get("smearing_width", 0.005),
            scf_method=d.get("scf_method", "DIIS"),
            scf_maxcycle=d.get("scf_maxcycle", 800),
            fmixing=d.get("fmixing", 30), num_atoms=2, spacegroup=1,
            spinlock=(d["spinlock"] if d.get("spin_polarized") else 0),
            spinlock_cycles=d.get("spinlock_cycles", 50),
        )
    text = out.read_text()
    assert "SPINLOCK" in text
    # The value line following the keyword must be the preserved "2 50".
    after = text.split("SPINLOCK", 1)[1].strip().split("\n", 1)[0].split()
    assert after[:2] == ["2", "50"]
