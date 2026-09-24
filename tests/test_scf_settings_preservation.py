"""A generated deck keeps the parent's SCF controls.

The input parser stored MAXCYCLE, FMIXING and the SCF method as top-level
keys, but the writer reads only ``settings["scf_settings"]``, so every child
deck went back to DIIS / MAXCYCLE 800 / FMIXING 30. BROYDEN's parameter line,
LEVSHIFT and the parent's BIPOSIZE/EXCHSIZE were never carried at all, and a
parent's ``SPINLOCK / 0 50`` was dropped because 0 was read as "no lock".

The corpus tests run the real ``mace_cli opt2d12`` and skip when ``test/`` is
absent. The parser and writer tests use inline decks and run everywhere.
"""
import io
import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA
from d12_parsers import CrystalInputParser
from d12_writer import write_scf_section

MACE_CLI = REPO_ROOT / "mace_cli"

DIA_REV1 = "OPT/1_dia_opt_rev1"        # B3LYP-D3 SPIN, SPINLOCK 0 50, MAXCYCLE 1600, FMIXING 50
DIA_BULK = "OPT/1_dia_opt_BULK_OPTGEOM"  # MAXCYCLE in OPTGEOM and in the SCF block, FMIXING 60
T1CA = "OPT/3,4^2T1-CA_BULK_OPTGEOM_TZ"  # 3 atoms with BIPOSIZE/EXCHSIZE, FMIXING 80

# Geometry and basis input for the inline decks. Only the parser reads these.
_HEAD = """title with D3 in it
CRYSTAL
0 0 0
227
3.5
1
6 0.125 0.125 0.125
OPTGEOM
FULLOPTG
TOLDEE
7
FINALRUN
4
MAXTRADIUS
0.25
MAXCYCLE
333
ENDOPT
END
6 1
0 0 1 0.0 1.0
0.16 1.0
99 0
END
"""


def _deck(tmp_path, body, name="deck.d12"):
    path = tmp_path / name
    path.write_text(_HEAD + body)
    return CrystalInputParser(str(path)).parse()


# ----------------------------------------------------------------- parser

def test_parser_reads_scf_block_into_scf_settings(tmp_path):
    d = _deck(tmp_path, "DFT\nSPIN\nPBE0\nENDDFT\nSPINLOCK\n0 50\nTOLINTEG\n7 7 7 7 14\n"
              "TOLDEE\n7\nSHRINK\n8 16\nSCFDIR\nBIPOSIZE\n4000000\nEXCHSIZE\n5000000\n"
              "MAXCYCLE\n1600\nFMIXING\n50\nBROYDEN\n1.0E-4 40 3\nLEVSHIFT\n5 1\nPPAN\nEND\n")
    assert d["scf_settings"] == {
        "method": "BROYDEN", "broyden": ("1.0E-4", 40, 3), "maxcycle": 1600,
        "fmixing": 50, "levshift": (5, 1), "biposize": 4000000, "exchsize": 5000000,
    }
    assert d["spinlock"] == 0 and d["spinlock_explicit"] is True


def test_optgeom_maxcycle_is_not_the_scf_maxcycle(tmp_path):
    """OPTGEOM's MAXCYCLE sits more than 5 lines after OPTGEOM here, which the
    old look-back missed. With no SCF MAXCYCLE, none must be recorded."""
    d = _deck(tmp_path, "DFT\nPBE0\nENDDFT\nTOLINTEG\n7 7 7 7 14\nTOLDEE\n7\n"
              "SHRINK\n8 16\nFMIXING\n40\nDIIS\nPPAN\nEND\n")
    assert d["scf_settings"] == {"fmixing": 40, "method": "DIIS"}


def test_hf_deck_scf_block_starts_after_basis(tmp_path):
    d = _deck(tmp_path, "UHF\nSPINLOCK\n0 50\nTOLINTEG\n7 7 7 7 14\nTOLDEE\n7\n"
              "SHRINK\n8 16\nMAXCYCLE\n900\nFMIXING\n60\nDIIS\nPPAN\nEND\n")
    assert d["scf_settings"] == {"maxcycle": 900, "fmixing": 60, "method": "DIIS"}
    assert d["spinlock_explicit"] is True


def test_dft_block_closed_by_plain_end(tmp_path):
    d = _deck(tmp_path, "DFT\nSPIN\nPBE-D3\nXLGRID\nEND\nTOLINTEG\n7 7 7 7 14\n"
              "TOLDEE\n7\nSHRINK\n8 16\nMAXCYCLE\n1600\nFMIXING\n80\nDIIS\nPPAN\nEND\n")
    assert d["scf_settings"] == {"maxcycle": 1600, "fmixing": 80, "method": "DIIS"}


# ----------------------------------------------------------------- writer

def _scf(**kwargs):
    buf = io.StringIO()
    args = dict(tolerances={"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, k_points=(8, 8, 8),
                dimensionality="CRYSTAL", use_smearing=False, smearing_width=0.0,
                scf_method="DIIS", scf_maxcycle=800, fmixing=30, num_atoms=2,
                spacegroup=227)
    args.update(kwargs)
    write_scf_section(buf, **args)
    return buf.getvalue().splitlines()


def test_writer_defaults_unchanged():
    assert _scf() == ["TOLINTEG", "7 7 7 7 14", "TOLDEE", "7", "SHRINK", "8 16", "SCFDIR",
                      "MAXCYCLE", "800", "FMIXING", "30", "DIIS", "HISTDIIS", "100",
                      "PPAN", "END"]
    assert "BIPOSIZE" in _scf(num_atoms=6)


def test_writer_levshift_before_ppan_accepts_a_json_list():
    lines = _scf(levshift=[5, 1])
    assert lines[-4:] == ["LEVSHIFT", "5 1", "PPAN", "END"]


def test_writer_parent_buffer_sizes_replace_the_atom_count_rule():
    lines = _scf(num_atoms=2, biposize=4000000)
    assert lines[lines.index("BIPOSIZE") + 1] == "4000000"
    assert "EXCHSIZE" not in lines


def test_writer_zero_spinlock_only_when_asked():
    assert "SPINLOCK" not in _scf(spinlock=0)
    lines = _scf(spinlock=0, spinlock_cycles=50, write_zero_spinlock=True)
    assert lines[:3] == ["SPINLOCK", "0 50", "TOLINTEG"]


def test_writer_broyden_keeps_the_raw_w0_token():
    lines = _scf(scf_method="BROYDEN", broyden_w0="1.0E-4", broyden_imix=40,
                 broyden_istart=3)
    assert lines[lines.index("BROYDEN") + 1] == "1.0E-4 40 3"


# ----------------------------------------------------------------- D3 in title

def test_d3_in_title_is_not_dispersion():
    from mace.utils.settings_extractor import _extract_functional_info
    deck = "dia_PBE0-D3_synth\nCRYSTAL\nEND\nDFT\nSPIN\nPBE0\nENDDFT\nEND\n"
    assert "dispersion" not in _extract_functional_info(deck)
    assert _extract_functional_info(deck.replace("\nPBE0\n", "\nPBE0-D3\n"))["dispersion"] == "D3"


# ----------------------------------------------------------------- real opt2d12

def _copy_parent(stem, tmp_path):
    src = TEST_DATA / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return src.stem


def _run(tmp_path, name, args, stdin):
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name}.out",
         "--d12-file", f"{name}.d12", *args],
        cwd=tmp_path, input=stdin, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _config_run(tmp_path, name, calc_type):
    (tmp_path / "cfg.json").write_text(json.dumps({"calculation_type": calc_type}))
    return _run(tmp_path, name, ["--config-file", "cfg.json"], "y\n" + "\n" * 17)


def _planless_run(tmp_path, name):
    """The engine's planless SP call: the interactive flow on blank answers."""
    return _run(tmp_path, name, ["--non-interactive", "--calc-type", "SP"], "n\n" + "\n" * 19)


def _after(lines, key):
    return lines[lines.index(key) + 1].strip()


def _scf_tail(lines):
    """The deck from TOLINTEG to the end, whitespace-normalised."""
    return [" ".join(l.split()) for l in lines[lines.index("TOLINTEG"):]]


@pytest.mark.parametrize("calc_type", ["SP", "FREQ", "OPT"])
@pytest.mark.parametrize("stem,maxcycle,fmixing", [
    (DIA_REV1, "1600", "50"), (DIA_BULK, "1600", "60"), (T1CA, "1600", "80"),
])
def test_child_keeps_maxcycle_and_fmixing(tmp_path, stem, maxcycle, fmixing, calc_type):
    name = _copy_parent(stem, tmp_path)
    child = _config_run(tmp_path, name, calc_type)
    scf = child[child.index("TOLINTEG"):]
    assert _after(scf, "MAXCYCLE") == maxcycle
    assert _after(scf, "FMIXING") == fmixing


def test_child_keeps_spinlock_zero_and_buffer_sizes(tmp_path):
    name = _copy_parent(DIA_REV1, tmp_path)
    child = _config_run(tmp_path, name, "SP")
    i = child.index("SPINLOCK")
    assert child[i + 1] == "0 50" and child[i + 2] == "TOLINTEG"


def test_three_atom_parent_keeps_its_biposize(tmp_path):
    name = _copy_parent(T1CA, tmp_path)
    child = _config_run(tmp_path, name, "SP")
    assert _after(child, "BIPOSIZE") == "110000000"
    assert _after(child, "EXCHSIZE") == "110000000"


def _edited_rev1(tmp_path):
    """The real rev1 deck with BROYDEN (non-default record) and LEVSHIFT."""
    name = _copy_parent(DIA_REV1, tmp_path)
    d12 = tmp_path / f"{name}.d12"
    text = d12.read_text()
    assert "DIIS\nHISTDIIS\n100\nPPAN\n" in text
    d12.write_text(text.replace("DIIS\nHISTDIIS\n100\nPPAN\n",
                                "BROYDEN\n1.0E-4 40 3\nLEVSHIFT\n5 1\nPPAN\n"))
    return name, _scf_tail(d12.read_text().splitlines())


def test_config_child_keeps_broyden_and_levshift(tmp_path):
    name, parent_tail = _edited_rev1(tmp_path)
    child = _config_run(tmp_path, name, "SP")
    assert _scf_tail(child) == parent_tail


def test_planless_child_keeps_broyden_and_levshift(tmp_path):
    """The interactive flow rebuilds scf_settings from its prompts, which only
    cover MAXCYCLE, FMIXING and the method; the rest comes from the parent."""
    name, parent_tail = _edited_rev1(tmp_path)
    child = _planless_run(tmp_path, name)
    assert _scf_tail(child) == parent_tail


def test_saved_options_round_trip_through_json(tmp_path):
    """--save-options turns the tuples into lists; the writer must take them."""
    import CRYSTALOptToD12 as M
    from d12_parsers import CrystalOutputParser

    name, parent_tail = _edited_rev1(tmp_path)
    geo = CrystalOutputParser(str(tmp_path / f"{name}.out")).parse()
    parsed = CrystalInputParser(str(tmp_path / f"{name}.d12")).parse()
    scf = json.loads(json.dumps(parsed["scf_settings"]))
    assert scf["levshift"] == [5, 1] and scf["broyden"] == ["1.0E-4", 40, 3]
    settings = dict(geo, **{k: v for k, v in parsed.items()
                            if k in ("tolerances", "k_points", "spinlock", "spinlock_cycles",
                                     "spinlock_explicit", "spin_polarized", "functional",
                                     "dispersion", "dft_grid", "method")})
    settings.update(scf_settings=scf, calculation_type="SP", basis_set_type="INTERNAL",
                    basis_set="POB-TZVP-REV2")
    out = tmp_path / "roundtrip.d12"
    assert M.write_d12_file(str(out), geo, settings) is not False
    assert _scf_tail(out.read_text().splitlines()) == parent_tail
