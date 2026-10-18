"""`mace opt2d12 --d12-file DECK` with no --out-file: a deck is its own parent.

The geometry is the deck's own records (CrystalInputParser keeps every atom
record as written, and the deck writer prints them back byte for byte, see
test_deck_round_trip.py); the settings are the deck's, as an .out follow-up
takes them from the .d12 beside it ("defaults follow the parent"). Asked for
the deck's own calculation type with the parent's settings, the child is
the parent deck again from its second record on (the title is the file's
name).

A deck whose geometry is not in its own records is refused, with the reason:
EXTERNAL / DLVINPUT read fort.34, and the block-1 records that edit the
geometry (SUPERCEL, ATOMSUBS, SLABCUT, ...) build a structure the child would
not have. Their run's .out holds the structure CRYSTAL built.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
DATA = REPO_ROOT / "tests" / "data"
EXAMPLES = REPO_ROOT / "examples" / "gui" / "outputs"


def run(cwd, *args, stdin="y\n"):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "opt2d12", *args], cwd=cwd,
                       input=stdin + "\n" * 40, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def calc_of(text):
    lines = [x.strip() for x in text.splitlines()]
    return "OPT" if "OPTGEOM" in lines else "FREQ" if "FREQCALC" in lines else "SP"


def geometry(path):
    import contextlib
    import io
    from d12_parsers import CrystalInputParser
    with contextlib.redirect_stdout(io.StringIO()):
        p = CrystalInputParser(str(path)).parse()
    cell = {k: round(v, 6) if v is not None else None for k, v in (p.get("cell_parameters") or {}).items()}
    atoms = [(a["atomic_number"], round(a["x"], 6), round(a["y"], 6), round(a["z"], 6)) for a in p["atoms"]]
    return p["dimensionality"], p.get("spacegroup"), cell, atoms


def records(text):
    """A deck from its second record on (the title is the file's name), as opt2d12 writes
    any child, from an .out too: an EXTERNAL basis in its normalized spacing (its
    external_basis_data, never the parent's text, d12_parsers.DECK_TEXT_KEYS: its numeric
    records without their leading spaces) and a
    POLYMER's SHRINK IS as "0 IS / IS 1 1" (the same mesh). Everything else is compared as written."""
    lines = text.split("\n")[1:]
    out = [x.strip() if x.split() and all(_num(v) for v in x.split()) else x for x in lines]
    if out and out[0].strip() == "POLYMER":
        for i, x in enumerate(out[:-1]):
            if x.strip() == "SHRINK" and len(out[i + 1].split()) == 2 and out[i + 1].split()[0] != "0":
                isk, isp = out[i + 1].split()
                out[i + 1:i + 2] = [f"0 {isp}", f"{isk} 1 1"]
                break
    return out


def _num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def child(tmp_path, deck, calc=None, *extra, stdin="y\n"):
    shutil.copy2(deck, tmp_path / deck.name)
    calc = calc or calc_of(deck.read_text())
    code, log = run(tmp_path, "--d12-file", deck.name, "--calc-type", calc, *extra, stdin=stdin)
    kids = sorted(p for p in tmp_path.glob("*.d12") if p.name != deck.name)
    return code, log, kids


SAME_TYPE = [
    DATA / "low_dim_groups" / "graphene_lg80.d12",            # SLAB, layer group 80
    DATA / "low_dim_groups" / "polyyne_rg28.d12",             # POLYMER, rod group 28
    *sorted((DATA / "ecp_decks").glob("*.d12")),              # OPT, EXTERNAL basis, ECP atoms (282/247 as X)
]


def ecp_labels_as_elements(lines):
    """The deck's records with an ECP atom's trailing label written as opt2d12 writes it:
    the element of its number modulo 100 (`282 ... Biso 1.000000 Pb`), where these older
    decks have `X`. CRYSTAL reads only the number and the coordinates on these lines."""
    import re
    from mace.utils.formula import ELEMENTS
    chemical_symbols = ("X",) + ELEMENTS   # indexed by Z, as ase.data's (no ase in the CI jobs)
    out = []
    for x in lines:
        m = re.match(r"^(\d{3})(\s.*\sBiso\s+\S+\s+)X$", x)
        out.append(f"{m.group(1)}{m.group(2)}{chemical_symbols[int(m.group(1)) % 100]}" if m and int(m.group(1)) > 200 else x)
    return out


@pytest.mark.parametrize("deck", SAME_TYPE, ids=lambda p: p.name[:24])
def test_the_same_calculation_from_a_deck_is_the_deck_again(tmp_path, deck):
    code, log, kids = child(tmp_path, deck)
    assert code == 0, log[-2000:]
    assert len(kids) == 1, (kids, log[-1500:])
    assert records(kids[0].read_text()) == ecp_labels_as_elements(records(deck.read_text())), log[-1500:]
    assert "Traceback" not in log


def test_the_child_is_named_after_the_deck_and_the_log_says_where_the_geometry_came_from(tmp_path):
    code, log, kids = child(tmp_path, EXAMPLES / "Si_sp.d12", "OPT")
    assert code == 0, log[-2000:]
    assert len(kids) == 1 and kids[0].name.startswith("Si_sp_opt_") and "_optimized" not in kids[0].name, kids
    assert "geometry from the deck itself" in log.lower()
    text = kids[0].read_text()
    assert "OPTGEOM" in text
    # the geometry is the deck's: its group, cell and atoms (this hand-written deck's
    # records are spelled differently from the writer's, so they are compared as numbers)
    assert geometry(kids[0]) == geometry(EXAMPLES / "Si_sp.d12")


def test_save_options_then_config_file_gives_the_same_deck(tmp_path):
    """The two routes Setup uses: the interactive run's saved answers, and the printed command on them."""
    deck = EXAMPLES / "Si_sp.d12"
    a = tmp_path / "a"
    a.mkdir()
    code, log, kids = child(a, deck, "FREQ", "--save-options", "--options-file", "j.json", "--output-dir", "out")
    assert code == 0, log[-2000:]
    m = sorted((a / "out").glob("*.d12"))
    assert len(m) == 1
    b = tmp_path / "b"
    b.mkdir()
    shutil.copy2(deck, b / deck.name)
    shutil.copy2(a / "j.json", b / "j.json")
    code, log = run(b, "--d12-file", deck.name, "--config-file", "j.json", "--yes", "--output-dir", "out", stdin="")
    assert code == 0, log[-2000:]
    g = sorted((b / "out").glob("*.d12"))
    assert [p.name for p in g] == [p.name for p in m]
    assert g[0].read_text() == m[0].read_text()


EXTERNAL_DECK = """ext_test
EXTERNAL
OPTGEOM
FULLOPTG
ENDOPT
END
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
SHRINK
0 8
8 8 8
TOLDEE
7
END
"""


@pytest.mark.parametrize("keyword", ["EXTERNAL", "DLVINPUT"])
def test_a_deck_with_no_geometry_of_its_own_is_refused(tmp_path, keyword):
    (tmp_path / "ext.d12").write_text(EXTERNAL_DECK.replace("EXTERNAL", keyword))
    code, log = run(tmp_path, "--d12-file", "ext.d12", "--calc-type", "SP")
    assert code != 0
    assert sorted(p.name for p in tmp_path.glob("*.d12")) == ["ext.d12"]
    assert keyword in log and "fort.34" in log and "--out-file" in log, log[-1500:]
    assert "Traceback" not in log


def test_a_deck_that_edits_its_geometry_is_refused(tmp_path):
    text = (EXAMPLES / "Si_sp.d12").read_text().splitlines()
    i = next(k for k, x in enumerate(text) if x.strip() == "END")
    text[i:i] = ["SUPERCEL", "2 0 0", "0 2 0", "0 0 2"]
    (tmp_path / "sc.d12").write_text("\n".join(text) + "\n")
    code, log = run(tmp_path, "--d12-file", "sc.d12", "--calc-type", "SP")
    assert code != 0
    assert sorted(p.name for p in tmp_path.glob("*.d12")) == ["sc.d12"]
    assert "SUPERCEL" in log and ".out" in log, log[-1500:]


@pytest.mark.parametrize("edit", [["MOLECULE", "1", "1 0 0 0"], ["SLABCUT", "0 0 1", "1 3"]],
                         ids=["MOLECULE", "SLABCUT"])
def test_a_line2_kind_used_as_an_edit_is_refused(tmp_path, edit):
    """MOLECULE after line 2 is the 3D-to-0D edit, not the line-2 molecule kind:
    it was filtered out with the kind, and the 3D crystal was written."""
    text = (EXAMPLES / "Si_sp.d12").read_text().splitlines()
    i = next(k for k, x in enumerate(text) if x.strip() == "END")
    text[i:i] = edit
    (tmp_path / "mx.d12").write_text("\n".join(text) + "\n")
    code, log = run(tmp_path, "--d12-file", "mx.d12", "--calc-type", "SP")
    assert code != 0, log[-1500:]
    assert sorted(p.name for p in tmp_path.glob("*.d12")) == ["mx.d12"]
    assert f"with {edit[0]} after the atoms" in log, log[-1500:]


def test_deck_parent_refusal_keeps_the_line2_kind_and_refuses_its_edit_reuse(tmp_path):
    import CRYSTALOptToD12 as M
    mol = tmp_path / "m.d12"
    mol.write_text("water\nMOLECULE\n1\n2\n8 0 0 0\n1 0.9 0 0\nEND\nBASISSET\nSTO-3G\nEND\n")
    assert M.deck_parent_refusal(str(mol)) is None
    mol.write_text("water\nMOLECULE\n1\n2\n8 0 0 0\n1 0.9 0 0\nGHOSTS\n1\n2\nEND\nBASISSET\nSTO-3G\nEND\n")
    assert M.deck_parent_refusal(str(mol)) is None          # GHOSTS is basis input
    sl = tmp_path / "s.d12"
    sl.write_text("slab\nSLAB\n80\n2.46\n1\n6 0.333 0.667 0\nMOLECULE\n1\n1 0 0 0\nEND\n")
    assert "MOLECULE" in (M.deck_parent_refusal(str(sl)) or "")


def test_a_missing_deck_says_so(tmp_path):
    code, log = run(tmp_path, "--d12-file", "nope.d12", "--calc-type", "SP")
    assert code != 0
    assert "nope.d12" in log and "Traceback" not in log


def test_an_out_file_still_takes_the_d12_as_its_parent_deck(tmp_path):
    """--out-file with --d12-file is unchanged: the geometry is the .out's."""
    for ext in (".out", ".d12"):
        shutil.copy2(EXAMPLES / f"Si_sp{ext}", tmp_path / f"Si_sp{ext}")
    code, log = run(tmp_path, "--out-file", "Si_sp.out", "--d12-file", "Si_sp.d12", "--calc-type", "SP")
    assert code == 0, log[-2000:]
    kids = [p.name for p in tmp_path.glob("*.d12") if p.name != "Si_sp.d12"]
    assert len(kids) == 1 and kids[0].endswith("_optimized.d12"), kids
    assert "geometry from the deck itself" not in log.lower()


def _corpus_decks():
    if not TEST_DATA.is_dir():
        return []
    sys.path.insert(0, str(Path(__file__).parent))
    from test_deck_round_trip import _hand_edited
    from mace.preopt import deckscan
    out = []
    for d in sorted(TEST_DATA.glob("*/*.d12")):
        text = d.read_text(errors="replace")
        edits = [e for e in deckscan.scan(text)["geometry_edits"] if e not in ("SLAB", "POLYMER", "MOLECULE", "GHOSTS")]
        if _hand_edited(text) or edits:
            continue
        out.append(d)
    return out


def test_corpus_decks_of_every_dimensionality_come_back_unchanged(tmp_path):
    """Real decks (CRYSTAL, SLAB, MOLECULE; SP, OPT, FREQ): the same calculation from
    the deck with its own settings is the deck again, from the second record on."""
    decks = _corpus_decks()
    if not decks:
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    # one of each (dimensionality, calculation type, basis kind): the whole corpus is the slow test below
    picked, seen = [], set()
    for d in decks:
        t = d.read_text(errors="replace")
        key = (t.splitlines()[1].strip(), calc_of(t), "BASISSET" in t)
        if key not in seen:
            seen.add(key)
            picked.append(d)
    bad = []
    for k, d in enumerate(picked):
        w = tmp_path / f"w{k}"
        w.mkdir()
        code, log, kids = child(w, d)
        if code != 0 or len(kids) != 1 or kids[0].read_text().split("\n")[1:] != d.read_text().split("\n")[1:]:
            # (exactly as written: the corpus decks are MACE's, internal basis or EXTERNAL from its library)
            bad.append((d.name, code, log[-600:]))
    assert len(picked) >= 5
    assert not bad, bad
