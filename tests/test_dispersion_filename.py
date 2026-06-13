"""Regression test for the -D3-D3 continuation-filename nuisance.

The functional string embedded in continuation filenames could double to
'...-D3-D3...'. The SCF content was always correct (functional/dispersion written
once), so this is cosmetic, but dedupe_dispersion_suffix guarantees a generated
filename never carries a doubled '-D3'. Self-contained (pure helper, no corpus).
"""
import pytest

from CRYSTALOptToD12 import dedupe_dispersion_suffix as dedupe


@pytest.mark.parametrize("functional, expected", [
    ("B3LYP-D3-D3", "B3LYP-D3"),       # the classic doubling
    ("PBE-D3-D3-D3", "PBE-D3"),         # triple collapses too
    ("CAM-B3LYP-D3-D3", "CAM-B3LYP-D3"),
    ("B3LYP-D3", "B3LYP-D3"),           # single -D3 unchanged
    ("B3LYP", "B3LYP"),                 # no dispersion unchanged
    ("HSESOL3C", "HSESOL3C"),           # 3C method unaffected
    ("PBE0-D3", "PBE0-D3"),
    ("", ""),                            # empty safe
])
def test_dedupe_dispersion_suffix(functional, expected):
    assert dedupe(functional) == expected


def test_generated_filename_never_doubles():
    """Whatever the functional, the embedded name carries at most one -D3."""
    for functional in ["B3LYP-D3-D3", "PBE-D3-D3-D3", "CAM-B3LYP-D3-D3"]:
        name = f"material_sp_{dedupe(functional)}_optimized.d12"
        assert "-D3-D3" not in name
