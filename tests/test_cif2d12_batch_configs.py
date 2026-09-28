"""cif2d12 --batch and opt2d12 --config-file with the shipped example configs.

Two defects, each reproduced on the real invocation path first:

1. None of Crystal_d12/example_configs/*.json loaded with
   ``cif2d12 --batch --options_file``: they wrap their settings in
   ``{"version", "type", "configuration"}``, which the loader did not unwrap,
   and they spell settings the opt2d12 way (``functional``, ``dispersion``,
   ``scf_settings``) where cif2d12 reads ``dft_functional``,
   ``use_dispersion``, ``scf_method``. Every run died with
   ``KeyError: 'dimensionality'``. opt2d12 --config-file read the wrapper as
   the settings too and crashed on the missing calculation type.
2. The 3c_composite template paired PBEH3C with MINIX (HF-3c's basis); the
   pairing now comes from d12_constants' basis_requirements.

The corpus-backed tests skip cleanly when test/ is absent (CI).
"""
import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA
from d12_config import (
    config_to_cif_options,
    get_default_d12_configs,
    unwrap_d12_config,
)
from d12_constants import FUNCTIONAL_CATEGORIES, required_basis_for

MACE_CLI = REPO_ROOT / "mace_cli"
EXAMPLES = sorted((REPO_ROOT / "Crystal_d12" / "example_configs").glob("*.json"))
CIFS = TEST_DATA / "CIFs"

# Real corpus CIFs: a centred cubic cell (225, 8 atoms -> 2 unique), a P1
# molecule box (every atom its own orbit - the prompt that used to EOF), and
# one whose CIF group (167) spglib does not reproduce (158)
ROCKSALT = "AgCl_mp-22922_sg225_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.cif"
P1_BOX = "1LiFSI-1EC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized.cif"
MISMATCH = "Ag2Br3_mp-862982_sg167_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.cif"


def _need_corpus(*names):
    for name in names:
        if not (CIFS / name).exists():
            pytest.skip("test/ corpus not present (gitignored)")


def _convert(tmp_path, options_file, *cifs):
    """Run `mace convert --batch` with no terminal; return (decks, log)."""
    cif_dir = tmp_path / "cifs"
    out_dir = tmp_path / "out"
    cif_dir.mkdir()
    for name in cifs:
        (cif_dir / name).write_bytes((CIFS / name).read_bytes())
    cmd = [sys.executable, str(MACE_CLI), "convert", "--no-banner", "--batch",
           "--options_file", str(options_file), "--cif_dir", str(cif_dir),
           "--output_dir", str(out_dir)]
    result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-2000:]
    decks = {p.name: p.read_text() for p in out_dir.glob("*.d12")} if out_dir.exists() else {}
    return decks, log


def _atom_lines(deck):
    """The atom records of a 3D deck (the line after the lattice holds N)."""
    lines = deck.splitlines()
    n = int(lines[5].split()[0])
    return lines[6:6 + n]


# --- loading ---------------------------------------------------------------


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_example_config_maps_to_complete_cif2d12_options(path):
    opts = config_to_cif_options(json.loads(path.read_text()))
    conf = unwrap_d12_config(json.loads(path.read_text()))
    for key in ("dimensionality", "calculation_type", "method", "basis_set",
                "basis_set_type", "is_spin_polarized", "tolerances",
                "scf_method", "symmetry_handling"):
        assert key in opts, key
    if opts["method"] == "DFT":
        assert opts["dft_functional"] == conf["functional"]
    else:
        assert opts["hf_method"] == conf["functional"]
    assert opts["use_dispersion"] == conf["dispersion"]
    assert opts["is_spin_polarized"] == conf["spin_polarized"]
    assert opts["scf_maxcycle"] == conf["scf_settings"]["maxcycle"]
    assert opts["tolerances"] == conf["tolerances"]
    # interactive defaults for what a configuration file does not say
    assert opts["dimensionality"] == conf.get("dimensionality", "CRYSTAL")
    assert opts["symmetry_handling"] == "CIF" and opts["write_only_unique"] is True


def test_a_flat_cif2d12_options_file_is_kept_as_written():
    from d12_constants import scf_tolerances
    flat = {
        "symmetry_handling": "SPGLIB", "write_only_unique": True,
        "dimensionality": "CRYSTAL", "calculation_type": "OPT",
        "optimization_type": "FULLOPTG", "method": "DFT",
        "dft_functional": "HSESOL3C", "use_dispersion": False,
        "basis_set_type": "INTERNAL", "basis_set": "SOLDEF2MSVP",
        "dft_grid": "XLGRID", "is_spin_polarized": True,
        "tolerances": scf_tolerances("2"), "scf_method": "DIIS",
        "scf_maxcycle": 900, "fmixing": 40,
        # opt2d12-style keys present alongside must not override
        "functional": "PBE", "dispersion": True,
    }
    opts = config_to_cif_options(dict(flat))
    for key, value in flat.items():
        assert opts[key] == value, key


def test_a_3c_method_with_another_basis_is_refused():
    with pytest.raises(ValueError, match="def2-mSVP"):
        config_to_cif_options({"calculation_type": "SP", "method": "DFT",
                               "functional": "PBEH3C", "basis_set": "MINIX"})


def test_a_3c_method_without_a_basis_gets_the_one_it_is_defined_on():
    opts = config_to_cif_options({"calculation_type": "SP", "method": "DFT",
                                  "dft_functional": "HSESOL3C"})
    assert opts["basis_set"] == "SOLDEF2MSVP"
    assert opts["dft_functional"] == "HSESOL3C"


def test_an_unusable_file_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="functional"):
        config_to_cif_options({"calculation_type": "SP", "method": "DFT",
                               "basis_set": "POB-TZVP-REV2"})
    with pytest.raises(ValueError, match="symmetry_handling"):
        config_to_cif_options({"calculation_type": "SP", "method": "HF",
                               "basis_set": "STO-3G",
                               "symmetry_handling": "ask"})


# --- 3c basis: one source of truth -------------------------------------------


def test_3c_template_takes_its_basis_from_basis_requirements():
    tpl = get_default_d12_configs()["3c_composite"]
    assert tpl["basis_set"] == FUNCTIONAL_CATEGORIES["3C"]["basis_requirements"]["PBEH3C"]
    assert tpl["basis_set"] == "def2-mSVP"


def test_no_template_or_example_contradicts_basis_requirements():
    configs = list(get_default_d12_configs().values())
    configs += [unwrap_d12_config(json.loads(p.read_text())) for p in EXAMPLES]
    for conf in configs:
        required = required_basis_for(conf.get("functional"))
        if required:
            assert conf["basis_set"] == required, conf["name"]


# --- the real invocation path ------------------------------------------------


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_example_config_converts_a_corpus_cif_in_batch_mode(tmp_path, path):
    pytest.importorskip("ase")
    conf = unwrap_d12_config(json.loads(path.read_text()))
    # a SLAB config needs a layered structure; the P1 box is one CRYSTAL
    # accepts as a slab (layer group 1)
    cif = P1_BOX if conf.get("dimensionality") == "SLAB" else ROCKSALT
    _need_corpus(cif)
    decks, log = _convert(tmp_path, path, cif)
    assert len(decks) == 1, log[-2000:]
    deck = next(iter(decks.values()))
    assert f"\n{conf['basis_set']}\n" in deck
    assert "KeyError" not in log
    if cif == ROCKSALT:
        # Fm-3m written as its asymmetric unit: Ag and Cl, one each
        assert [line.split()[0] for line in _atom_lines(deck)] == ["47", "17"]


def test_opt2d12_reads_a_wrapped_example_config(tmp_path):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    (tmp_path / "parent.out").write_text(src.read_text(errors="replace"))
    (tmp_path / "parent.d12").write_text(src.with_suffix(".d12").read_text())
    for name, basis, method_line in (("3c_composite", "def2-mSVP", "PBEH3C"),
                                     ("quick_screen", "STO-3G", None)):
        out = tmp_path / name
        cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--no-banner",
               "--out-file", "parent.out", "--d12-file", "parent.d12",
               "--config-file", str(REPO_ROOT / "Crystal_d12" / "example_configs" / f"{name}.json"),
               "--non-interactive", "--output-dir", str(out)]
        result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=300)
        log = result.stdout + result.stderr
        assert result.returncode == 0, log[-2000:]
        (deck,) = out.glob("*.d12")
        lines = deck.read_text().splitlines()
        assert lines[lines.index("BASISSET") + 1] == basis
        if method_line:
            assert method_line in lines
        else:
            assert "DFT" not in lines  # RHF: no DFT block


def test_d12_from_config_runs_both_tools_with_an_example_config(tmp_path):
    """The unified front end passed the input positionally, which neither
    converter accepts, and did not find a real .out behind MPI start-up
    lines."""
    pytest.importorskip("ase")
    out = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    _need_corpus(ROCKSALT)
    if not out.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    (tmp_path / "agcl.cif").write_bytes((CIFS / ROCKSALT).read_bytes())
    (tmp_path / "dia.out").write_text(out.read_text(errors="replace"))
    (tmp_path / "dia.d12").write_text(out.with_suffix(".d12").read_text())
    cmd = [sys.executable, str(REPO_ROOT / "Crystal_d12" / "d12_from_config.py"),
           "--config", "3c_composite.json", "agcl.cif", "dia.out"]
    result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-2000:]
    decks = sorted(p.name for p in tmp_path.glob("*.d12") if p.name != "dia.d12")
    assert decks == ["agcl_CRYSTAL_OPT_symm_PBEH3C_def2-mSVP.d12",
                     "dia_opt_PBEH3C_optimized.d12"], log[-2000:]
    for name in decks:
        lines = (tmp_path / name).read_text().splitlines()
        assert lines[lines.index("BASISSET") + 1] == "def2-mSVP"


def test_phonon_dispersion_examples_name_a_supercell():
    """FREQCALC DISPERSION without SCELPHONO is rejected by CRYSTAL23
    ("ERROR **** INPFREQ **** MAKE SUPERCELL WITH SCELPHONO", seen in
    mace preflight on phonon_bands.json)."""
    for path in EXAMPLES:
        freq = unwrap_d12_config(json.loads(path.read_text())).get("freq_settings") or {}
        if freq.get("dispersion"):
            assert freq.get("scelphono"), path.name
            # a label path ("M G", ...) under FREQCALC BANDS is a FORMAT
            # ERROR IN FREQCALC INPUT DECK; the coordinate path is accepted
            if "bands" in freq:
                assert freq["bands"].get("path_method") == "coordinates", path.name
