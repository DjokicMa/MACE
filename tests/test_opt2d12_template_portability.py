"""A --config-file template holds calculation settings; each structure keeps
what is its own. Real CLI (`mace_cli opt2d12` as a subprocess) on copies of
real corpus files, one file at a time as the workflow engine runs it ('y'
piped); the corpus-free checks at the end run in CI.

Measured on the report's template, saved from Ag1Br1 (EXTERNAL basis): applied
to any internal-basis parent it failed with "External basis set path not
configured properly". A template from a molecule wrote every atom of a
symmetric crystal under its space group, and a template naming an internal
basis wrote an external-basis parent's ECP atom numbers (247) under BASISSET.
"""
import json
import shutil
import subprocess
import sys

import pytest

import CRYSTALOptToD12 as M
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"

EXT = "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
INT = "Ag1Cl3_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized"
DIA = "1_dia_opt_rev1"
MOL3C = "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
PARENTS = (EXT, INT, DIA)


def _deck(directory, stem, functional="B3LYP-D3"):
    return directory / f"{stem}_sp_{functional}_optimized.d12"


def _copy(stems, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = TEST_DATA / "OPT" / f"{stem}.out"
        if not src.exists():
            pytest.skip("test/ corpus not present (gitignored, ~12GB)")
        shutil.copy(src, dest)
        shutil.copy(src.with_suffix(".d12"), dest)
    return dest


def _cli(args, cwd, stdin=subprocess.DEVNULL, input_text=None):
    kwargs = dict(cwd=cwd, capture_output=True, text=True, timeout=300)
    if input_text is None:
        kwargs["stdin"] = stdin
    else:
        kwargs["input"] = input_text
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", *args], **kwargs)


def _save_template(tmp_path, stem):
    """A template saved as the report did: --calc-type SP --non-interactive
    --save-options, with nothing on stdin."""
    work = _copy([stem], tmp_path / f"save_{stem[:8]}")
    template = tmp_path / f"{stem[:8]}_template.json"
    result = _cli(["--out-file", f"{stem}.out", "--d12-file", f"{stem}.d12",
                   "--calc-type", "SP", "--non-interactive", "--output-dir", "saved",
                   "--save-options", "--options-file", str(template)], work)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return template


def _as_saved_before(template, stem):
    """The same template as earlier versions wrote it: with the parent's
    external basis records, its k-point mesh and its atoms."""
    in_data = M.CrystalInputParser(str(TEST_DATA / "OPT" / f"{stem}.d12")).parse()
    data = json.loads(template.read_text())
    data["external_basis_data"] = in_data.get("external_basis_data", [])
    data["k_points"] = in_data.get("k_points")
    data["primitive_coordinates"] = [{"atom_number": "247", "x": "0", "y": "0", "z": "0"}]
    data["optimization_content"] = "the parent's optimisation log"
    old = template.with_name("old_" + template.name)
    old.write_text(json.dumps(data))
    return old

def _one(stem, template, outdir, cwd):
    """The single-file config path, as the workflow engine runs it."""
    return _cli(["--out-file", f"{stem}.out", "--d12-file", f"{stem}.d12",
                 "--config-file", str(template), "--output-dir", outdir],
                cwd, input_text="y\n")


@pytest.fixture
def ext_template(tmp_path):
    return _save_template(tmp_path, EXT)


def test_each_structure_keeps_its_own_parent_basis(tmp_path, ext_template):
    """The EXTERNAL-basis template: the external parent keeps its external
    basis records, the internal one its POB-TZVP-REV2, and the template's
    settings (SP, B3LYP-D3, XLGRID, TOLINTEG) reach both."""
    parents = _copy(PARENTS, tmp_path / "opts")
    old = _as_saved_before(ext_template, EXT)
    for template, outdir in ((ext_template, "new"), (old, "old")):
        for stem in PARENTS:
            result = _one(stem, template, outdir, parents)
            assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]

        ext_deck = _deck(parents / outdir, EXT).read_text()
        parent = (parents / f"{EXT}.d12").read_text()
        assert "BASISSET" not in ext_deck
        # The parent's own basis block, record for record.
        block = parent[parent.index("\nEND\n") + 5:parent.index("99 0")]
        assert block in ext_deck
        assert "\n247 " in ext_deck

        int_deck = _deck(parents / outdir, INT).read_text()
        assert "BASISSET\nPOB-TZVP-REV2\n" in int_deck
        assert "99 0" not in int_deck
        for deck in (ext_deck, int_deck):
            assert "OPTGEOM" not in deck and "FREQCALC" not in deck
            assert "B3LYP-D3\nXLGRID\n" in deck
            assert "TOLINTEG\n8 8 8 9 24\nTOLDEE\n9\n" in deck
    # A template saved by an earlier version gives the same decks, but for the
    # k-point mesh: its parent's mesh is regenerated from each other cell.
    for stem in PARENTS:
        new = _deck(parents / "new", stem).read_text().split("SHRINK")[0]
        old_deck = _deck(parents / "old", stem).read_text().split("SHRINK")[0]
        assert new == old_deck, stem


def test_new_template_keeps_each_parents_own_k_mesh(tmp_path, ext_template):
    parents = _copy(PARENTS, tmp_path / "opts")
    for stem in PARENTS:
        assert _one(stem, ext_template, "sp", parents).returncode == 0
        mesh = M.CrystalInputParser(str(parents / f"{stem}.d12")).parse()["k_points"]
        deck = _deck(parents / "sp", stem).read_text().splitlines()
        assert deck[deck.index("SHRINK") + 1].split() == str(mesh).split(), stem


def test_saved_template_holds_settings_not_a_structure(tmp_path, ext_template):
    data = json.loads(ext_template.read_text())
    for key in ("external_basis_data", "primitive_coordinates",
                "crystallographic_coordinates", "optimization_content", "k_points"):
        assert key not in data, key
    assert data["basis_set"] == "EXTERNAL (from original D12)"
    assert data["functional"] == "B3LYP-D3" and data["calculation_type"] == "SP"
    assert ext_template.stat().st_size < 5000


def test_molecule_template_writes_a_crystal_as_its_asymmetric_unit(tmp_path):
    """A template saved from a molecule says "write all atoms"; diamond
    (Fd-3m) still gets its one unique carbon, not both of the cell's."""
    template = _save_template(tmp_path, MOL3C)
    parents = _copy([DIA], tmp_path / "opts")
    result = _one(DIA, template, "sp", parents)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    deck = _deck(parents / "sp", DIA, "HSESOL3C").read_text().splitlines()
    assert deck[1:4] == ["CRYSTAL", "0 0 0", "227"]
    assert deck[5] == "1"
    assert "BASISSET" in deck and "SOLDEF2MSVP" in deck and "HSESOL3C" in deck


def test_internal_basis_on_an_external_parent_writes_plain_atomic_numbers(tmp_path):
    """Ag1Br1's parent numbers silver 247 (its ECP basis). With a template that
    names POB-TZVP-REV2 the deck says BASISSET, so silver is atom 47."""
    template = _save_template(tmp_path, INT)
    parents = _copy([EXT], tmp_path / "opts")
    result = _one(EXT, template, "sp", parents)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    deck = _deck(parents / "sp", EXT).read_text()
    assert "BASISSET\nPOB-TZVP-REV2\n" in deck
    assert "\n47 " in deck and "\n247 " not in deck
    assert " Ag\n" in deck



# Corpus-free

@pytest.mark.parametrize("config,expected", [
    ({"basis_set": "EXTERNAL (from original D12)", "basis_set_type": "EXTERNAL"}, True),
    ({"basis_set_type": "EXTERNAL", "use_original_external_basis": True}, True),
    ({"basis_set_type": "EXTERNAL", "use_original_external_basis": True,
      "basis_set_path": "/basis/dir"}, False),
    ({"basis_set": "POB-TZVP-REV2", "basis_set_type": "INTERNAL"}, False),
    ({"functional": "PBE0"}, False),
])
def test_which_templates_mean_each_parents_own_basis(config, expected):
    assert M.template_uses_parent_basis(config) is expected


def test_saved_template_leaves_out_structure_data():
    options = {"functional": "PBE0", "basis_set": M.PARENT_BASIS_MARKER, "k_points": "5 10",
               "external_basis_data": ["35 2"], "primitive_coordinates": [],
               "crystallographic_coordinates": [], "optimization_content": "log",
               "coordinates": [], "symmetry_operations": [], "tolerances": {"TOLDEE": 9}}
    assert M.options_for_template(options) == {
        "functional": "PBE0", "basis_set": M.PARENT_BASIS_MARKER, "tolerances": {"TOLDEE": 9}}


@pytest.mark.parametrize("number,settings,expected", [
    ("247", {"basis_set_type": "EXTERNAL"}, 247),
    ("247", {"basis_set_type": "INTERNAL"}, 47),
    ("282", {}, 82),
    ("247", {"basis_set_type": "EXTERNAL", "functional": "HSESOL3C"}, 47),
    ("35", {"basis_set_type": "INTERNAL"}, 35),
])
def test_conventional_atom_number(number, settings, expected):
    assert M.conventional_atom_number(number, settings) == expected
