"""cif2d12 --batch with the shipped example configs, and without a terminal.

Three defects, each reproduced on the real invocation path first:

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
3. In batch mode the "Use the 'reduced' structure anyway?" prompt read stdin,
   hit EOF, and the run reported "Error during symmetry analysis" before
   falling back. Batch mode now never asks; the answer comes from the options
   file or is the prompt's own default.

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



def test_a_3c_method_from_method_modifications_is_warned_about_too(capsys):
    """A workflow step can swap the functional through method_modifications;
    a 3c method on another basis gets the same warning, and the options
    (so the deck) are left as asked."""
    opts = config_to_cif_options({"calculation_type": "SP", "method": "DFT",
                                  "dft_functional": "HSE06",
                                  "method_modifications": {"functional": "PBEH3C"},
                                  "basis_set": "POB-TZVP-REV2"})
    assert opts["basis_set"] == "POB-TZVP-REV2"
    assert "PBEH3C is defined on the" in capsys.readouterr().err

def test_a_3c_method_with_another_basis_is_written_as_asked(capsys):
    """CRYSTAL23 runs a 3c method on another basis (the lead perovskites need
    HSE-3c on POB-TZVP-REV2: its def2-mSVP has no Pb), and main wrote those
    decks from a flat options file, so the pairing is warned about, not
    refused."""
    for shape in ({"dft_functional": "HSE3C"}, {"functional": "HSE3C"}):
        opts = config_to_cif_options(dict(shape, calculation_type="SP",
                                          method="DFT",
                                          basis_set="POB-TZVP-REV2"))
        assert opts["basis_set"] == "POB-TZVP-REV2"
        assert opts["dft_functional"] == "HSE3C"
        assert "defined on the def2-mSVP basis" in capsys.readouterr().err


@pytest.mark.parametrize("functional,basis", [("PBEH3C", "DEF2-MSVP"),
                                              ("hsesol3c", "soldef2msvp")])
def test_a_flat_file_basis_is_written_as_spelled(capsys, functional, basis):
    """The spelling is the deck's BASISSET line and part of the file name;
    it is only normalised to compare it with the table."""
    opts = config_to_cif_options({"calculation_type": "OPT", "method": "DFT",
                                  "dimensionality": "CRYSTAL",
                                  "dft_functional": functional,
                                  "basis_set": basis})
    assert opts["basis_set"] == basis
    assert opts["dft_functional"] == functional
    assert capsys.readouterr().err == ""  # the same basis, no warning


def test_calculation_type_is_read_case_insensitively():
    """main compared "opt" with "OPT" and silently wrote a single point."""
    opts = config_to_cif_options({"calculation_type": "opt", "method": "DFT",
                                  "dft_functional": "PBE",
                                  "basis_set": "POB-TZVP-REV2",
                                  "optimization_type": "ATOMONLY"})
    assert opts["calculation_type"] == "OPT"
    assert opts["optimization_type"] == "ATOMONLY"


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
    assert "KeyError" not in log and "EOF when reading" not in log
    if cif == ROCKSALT:
        # Fm-3m written as its asymmetric unit: Ag and Cl, one each
        assert [line.split()[0] for line in _atom_lines(deck)] == ["47", "17"]


def test_batch_never_reads_stdin_for_a_structure_with_no_reduction(tmp_path):
    """The P1 box used to hit the 'reduced structure anyway?' prompt."""
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    _need_corpus(P1_BOX)
    cfg = REPO_ROOT / "Crystal_d12" / "example_configs" / "standard_dft_opt.json"
    decks, log = _convert(tmp_path, cfg, P1_BOX)
    assert "EOF when reading" not in log
    assert "Error during symmetry analysis" not in log
    assert "Asymmetric unit contains all" in log  # the case really occurred
    deck = next(iter(decks.values()))
    from ase.io import read
    cif_atoms = len(read(str(CIFS / P1_BOX), format="cif"))
    assert len(_atom_lines(deck)) == cif_atoms


def test_batch_mismatch_keeps_the_cif_space_group(tmp_path):
    """The interactive prompt's default. CRYSTAL23 (mace preflight) folds the
    equivalent atoms and reports the P1 deck's density; taking spglib's group
    instead (the interactive option 2) is not offered in batch mode."""
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    _need_corpus(MISMATCH)
    base = unwrap_d12_config(json.loads(
        (REPO_ROOT / "Crystal_d12" / "example_configs" / "standard_dft_opt.json").read_text()))

    (tmp_path / "o.json").write_text(json.dumps(base))
    decks, log = _convert(tmp_path, tmp_path / "o.json", MISMATCH)
    assert "keeping the CIF space group" in log
    assert "EOF when reading" not in log
    deck = next(iter(decks.values())).splitlines()
    assert deck[3] == "167"
    assert int(deck[5]) == 30  # every atom of the hexagonal cell


def test_verify_does_not_prompt_when_not_interactive(monkeypatch):
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    _need_corpus(P1_BOX)
    import NewCifToD12

    def no_stdin(*a, **k):
        raise AssertionError("batch mode read stdin")

    monkeypatch.setattr("builtins.input", no_stdin)
    monkeypatch.setattr(NewCifToD12, "yes_no_prompt", no_stdin)
    monkeypatch.setattr(NewCifToD12, "get_user_input", no_stdin)
    data = NewCifToD12.parse_cif(str(CIFS / P1_BOX), interactive=False)
    out = NewCifToD12.verify_and_reduce_to_asymmetric_unit(data, interactive=False)
    assert len(out["symbols"]) == len(data["symbols"])


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


# --- failures are reported, not swallowed -----------------------------------

PB_CUBIC = "TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.cif"
DIAMOND = "1_dia_opt_rev1.cif"


def _run_convert(tmp_path, options, cifs=()):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for name in cifs:
        (cif_dir / name).write_bytes((CIFS / name).read_bytes())
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps(options))
    cmd = [sys.executable, str(MACE_CLI), "convert", "--no-banner", "--batch",
           "--options_file", str(opts), "--cif_dir", str(cif_dir),
           "--output_dir", str(tmp_path / "out")]
    result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    decks = sorted((tmp_path / "out").glob("*.d12")) if (tmp_path / "out").exists() else []
    return result.returncode, result.stdout + result.stderr, decks


def test_a_refused_options_file_exits_non_zero(tmp_path):
    """The workflow executor only fails a CIF conversion on a non-zero exit;
    a refused file used to print the reason and exit 0 with no decks."""
    pytest.importorskip("ase")  # NewCifToD12 imports ase.io at load
    rc, log, decks = _run_convert(tmp_path, {"calculation_type": "SP",
                                             "method": "DFT",
                                             "basis_set": "POB-TZVP-REV2"})
    assert rc != 0, log[-2000:]
    assert "must name its functional" in log
    assert decks == []


def test_a_batch_that_writes_no_deck_exits_non_zero(tmp_path):
    """HF-3c's MINIX has no Pb: the one deck is refused, so the run failed."""
    pytest.importorskip("ase")
    _need_corpus(PB_CUBIC)
    rc, log, decks = _run_convert(
        tmp_path, {"calculation_type": "SP", "method": "HF",
                   "functional": "HF3C"}, [PB_CUBIC])
    assert rc != 0, log[-2000:]
    assert decks == []
    assert "No D12 file was written for the 1 CIF file(s)" in log


def test_d12_from_config_reports_a_run_that_wrote_no_deck(tmp_path):
    pytest.importorskip("ase")
    _need_corpus(PB_CUBIC)
    (tmp_path / "pb.cif").write_bytes((CIFS / PB_CUBIC).read_bytes())
    cmd = [sys.executable, str(REPO_ROOT / "Crystal_d12" / "d12_from_config.py"),
           "--config", "3c_composite.json", "pb.cif"]
    result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    assert result.returncode != 0, log[-2000:]
    assert "Successfully" not in log
    assert "Processed 0/1 files successfully" in log
    assert list(tmp_path.glob("*.d12")) == []


# --- phonon dispersion --------------------------------------------------------


def test_phonon_dispersion_is_refused_where_it_cannot_be_written():
    from d12_calc_freq import phonon_dispersion_refusal
    disp = {"dispersion": True, "scelphono": [2, 2, 2]}
    assert phonon_dispersion_refusal("CRYSTAL", disp) is None
    assert phonon_dispersion_refusal("MOLECULE", {"dispersion": False}) is None
    assert "NOT ALLOWED FOR MOLECULES" in phonon_dispersion_refusal("MOLECULE", disp)
    for dim in ("SLAB", "POLYMER"):
        assert "only written for a 3D CRYSTAL" in phonon_dispersion_refusal(dim, disp)


def test_crystal_system_carries_the_lattice_centring():
    from d12_calc_freq import crystal_system_with_lattice
    assert crystal_system_with_lattice(227) == "cubic-F"
    assert crystal_system_with_lattice(229) == "cubic-I"
    assert crystal_system_with_lattice(221) == "cubic-P"
    assert crystal_system_with_lattice(166) == "trigonal-R"
    assert crystal_system_with_lattice(12) == "monoclinic-C"
    assert crystal_system_with_lattice(None) is None


def test_atomic_deck_never_leaves_a_partial_file(tmp_path):
    from d12_writer import atomic_deck
    deck = tmp_path / "x.d12"
    with pytest.raises(TypeError):
        with atomic_deck(str(deck)) as f:
            f.write("TITLE\n")
            raise TypeError("writer crashed half way")
    with atomic_deck(str(deck)) as f:
        f.write("TITLE\n")
        f.discard()
    assert list(tmp_path.iterdir()) == []
    with atomic_deck(str(deck)) as f:
        f.write("TITLE\n")
    assert deck.read_text() == "TITLE\n"
    assert [p.name for p in tmp_path.iterdir()] == ["x.d12"]


def _opt2d12(tmp_path, parent, config, out_dir):
    """opt2d12 on a copy of a corpus parent, given by ABSOLUTE path."""
    src = TEST_DATA / "OPT" / f"{parent}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    par = tmp_path / "parent"
    par.mkdir(exist_ok=True)
    (par / src.name).write_text(src.read_text(errors="replace"))
    (par / f"{parent}.d12").write_text(src.with_suffix(".d12").read_text())
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--no-banner",
           "--out-file", str(par / src.name), "--d12-file", str(par / f"{parent}.d12"),
           "--config-file", str(config), "--non-interactive",
           "--output-dir", str(out_dir)]
    result = subprocess.run(cmd, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr, par


PHONON_BANDS = REPO_ROOT / "Crystal_d12" / "example_configs" / "phonon_bands.json"


@pytest.mark.parametrize("parent,expect", [
    ("1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized",
     "NOT ALLOWED FOR MOLECULES"),
    ("4LG_FSI_2x2_ABAB_opt_HSESOL3C_Attempt2_SLAB_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized",
     "only written for a 3D CRYSTAL"),
], ids=["molecule", "slab"])
def test_opt2d12_refuses_phonon_dispersion_off_a_3d_crystal(tmp_path, parent, expect):
    """The SLAB crashed in the title (no space group) and left a 0-byte deck;
    the MOLECULE deck was written and CRYSTAL23 stopped at SCELPHONO."""
    out = tmp_path / "out"
    rc, log, par = _opt2d12(tmp_path, parent, PHONON_BANDS, out)
    assert rc != 0, log[-2000:]
    assert expect in log
    assert "Traceback" not in log
    assert list(out.iterdir()) == []
    assert sorted(p.suffix for p in par.iterdir()) == [".d12", ".out"]


def test_cif2d12_phonon_path_follows_the_lattice_centring(tmp_path):
    """Fd-3m from a CIF got the simple-cubic path (M-G-R-X-G); it now gets
    the face-centred one opt2d12 writes for the same structure."""
    pytest.importorskip("ase")
    _need_corpus(DIAMOND)
    decks, log = _convert(tmp_path, PHONON_BANDS, DIAMOND)
    (cif_deck,) = decks.values()
    rc, olog, _ = _opt2d12(tmp_path, "1_dia_opt_rev1", PHONON_BANDS,
                           tmp_path / "o2")
    assert rc == 0, olog[-2000:]
    (opt_deck,) = (tmp_path / "o2").glob("*.d12")

    def bands(text):
        lines = text.splitlines()
        i = lines.index("BANDS")
        return lines[i:i + 3 + int(lines[i + 2])]

    assert bands(cif_deck) == bands(opt_deck.read_text())
    assert "X-GAMMA-L-W-GAMMA" in opt_deck.read_text().splitlines()[0]


def test_opt2d12_output_dir_holds_the_deck_for_an_absolute_out_file(tmp_path):
    """An absolute --out-file path made os.path.join drop --output-dir, and
    the deck landed beside the parent (in the corpus, for one reviewer)."""
    out = tmp_path / "decks"
    cfg = REPO_ROOT / "Crystal_d12" / "example_configs" / "3c_composite.json"
    rc, log, par = _opt2d12(tmp_path, "1_dia_opt_rev1", cfg, out)
    assert rc == 0, log[-2000:]
    assert [p.name for p in out.iterdir()] == ["1_dia_opt_rev1_opt_PBEH3C_optimized.d12"]
    assert sorted(p.name for p in par.iterdir()) == ["1_dia_opt_rev1.d12",
                                                     "1_dia_opt_rev1.out"]
