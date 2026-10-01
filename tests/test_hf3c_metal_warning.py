"""HF-3c (HF3C in MINIX) on a structure with metal atoms is flagged.

Manual sec. 5.3.1 (p. 158): HF-3c's "original design targeted organic
complexes", it "has been carefully tested for molecular crystals", and
"applications to inorganic crystals and metals are less well tested and should
be treated carefully". The manual gives no MINIX element range; CRYSTAL23 loads
MINIX for Z 1-80 and 86 (d12_constants.VERIFIED_INTERNAL_BASIS_ELEMENTS), so an
AgCl HF3C/MINIX deck is accepted - and, measured on CRYSTAL23, diverged to
~1e7 Ha while diamond converged. The deck is still written; the user is told.
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_constants import hf3c_metal_warning  # noqa: E402
from test_opt2d12_config_template_logic import batch  # noqa: E402,F401
from test_opt2d12_hf_config import _run  # noqa: E402

DATA = REPO_ROOT / "tests" / "data"


@pytest.mark.parametrize("numbers,named", [
    ([47, 17], "Ag"),          # AgCl, the measured divergence
    ([247, 35], "Ag"),         # ECP numbering of an external-basis parent
    ([11, 17], "Na"),
    ([26, 8], "Fe"),
])
def test_metal_atoms_get_the_caution(numbers, named):
    msg = hf3c_metal_warning(numbers)
    assert msg and f"({named})" in msg and "p. 158" in msg


@pytest.mark.parametrize("numbers", [[6], [1, 6, 7, 8], [14, 8], [5, 7], [32], [52]])
def test_organic_and_non_metal_structures_get_none(numbers):
    assert hf3c_metal_warning(numbers) is None


def test_opt2d12_warns_for_hf3c_on_metal_parents(batch, tmp_path, capsys):
    status, out = _run(batch, tmp_path, capsys, {"method": "HF", "functional": "HF3C"})
    assert status == 0, out.out + out.err
    text = out.out + out.err
    assert "metal atoms (Ag)" in text and "metal atoms (Na)" in text
    # the decks are still written as before
    assert len(list((tmp_path / "sp").glob("*.d12"))) == 2


def test_opt2d12_hfsol3c_is_not_flagged(batch, tmp_path, capsys):
    """HFSOL-3c is the variant revised for inorganic solids (p. 162)."""
    status, out = _run(batch, tmp_path, capsys, {"method": "HF", "functional": "HFSOL3C"})
    assert status == 0
    assert "metal atoms" not in out.out + out.err


def _cif2d12(tmp_path, numbers=None):
    pytest.importorskip("ase", reason="CIF parsing needs ase")
    import NewCifToD12
    from d12_constants import DEFAULT_TOLERANCES

    cif_data = NewCifToD12.parse_cif(str(DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif"))
    if numbers:
        cif_data["atomic_numbers"] = [numbers] * len(cif_data["atomic_numbers"])
        cif_data["symbols"] = ["Ag"] * len(cif_data["symbols"])
    opts = dict(dimensionality="CRYSTAL", calculation_type="SP", basis_set_type="INTERNAL",
                basis_set="MINIX", method="HF", hf_method="HF3C",
                is_spin_polarized=False, tolerances=DEFAULT_TOLERANCES,
                scf_method="DIIS", symmetry_handling="CIF")
    out = tmp_path / "deck.d12"
    assert NewCifToD12.create_d12_file(cif_data, str(out), opts)
    return out.read_text().splitlines()


def test_cif2d12_diamond_hf3c_is_not_flagged(tmp_path, capsys):
    lines = _cif2d12(tmp_path)
    assert lines[lines.index("BASISSET"):lines.index("BASISSET") + 4] == [
        "BASISSET", "MINIX", "HF3C", "END"]
    out = capsys.readouterr()
    assert "metal atoms" not in out.out + out.err


def test_cif2d12_warns_for_hf3c_on_metal_atoms(tmp_path, capsys):
    _cif2d12(tmp_path, numbers=47)
    out = capsys.readouterr()
    assert "metal atoms (Ag)" in out.out + out.err
