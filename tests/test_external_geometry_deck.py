"""EXTERNAL decks (geometry from fort.34) in the GUI, `mace preflight` and `mace submit`.

A deck whose geometry record is EXTERNAL (or DLVINPUT) holds no geometry: CRYSTAL
reads it from fort.34 at run time (manual p.21). MACE's job script stages
``<stem>.ext`` as fort.34; a hand-written job script may copy another file.
Opening such a deck in `mace gui` showed "not found: 'cell_parameters'" (the
parser's KeyError leaking as a 404). Now the GUI draws the geometry from the
file the job script would stage (or the run's .out, labeled), and says in plain
words what is missing when there is none; `mace submit` and `mace preflight`
warn before such a deck goes to the queue without its fort.34.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA, corpus_guard
from test_gui_examples import _json, _start, needs_ase

MACE_CLI = REPO_ROOT / "mace_cli"
SI_OUT = REPO_ROOT / "examples" / "gui" / "outputs" / "Si_sp.out"

BASIS = "BASISSET\nPOB-TZVP-REV2\nDFT\nPBE\nEND\nSHRINK\n4 4\nEND\n"
EXTERNAL_DECK = "graphite_sp\nEXTERNAL\nEND\n" + BASIS

# fort.34 (manual pp.450-451): dimension/centering/type, the three lattice
# vectors, the symmetry operators, then the atoms in Cartesian Angstrom.
GRAPHITE_F34 = """\
   3   1   4
  0.246000000000E+01  0.000000000000E+00  0.000000000000E+00
 -0.123000000000E+01  0.213042249330E+01  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00  0.100500000000E+02
   1
  0.100000000000E+01  0.000000000000E+00  0.000000000000E+00
  0.000000000000E+00  0.100000000000E+01  0.000000000000E+00
  0.000000000000E+00  0.000000000000E+00  0.100000000000E+01
  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
   6
     6  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
     6  0.000000000000E+00  0.142028166220E+01  0.000000000000E+00
     6  0.000000000000E+00  0.142028166220E+01  0.335000000000E+01
     6  0.123000000000E+01  0.710140831100E+00  0.335000000000E+01
     6  0.123000000000E+01  0.710140831100E+00  0.670000000000E+01
     6  0.000000000000E+00  0.000000000000E+00  0.670000000000E+01
"""

# The job script that came with the reported deck: it stages ANOTHER job's
# optimized geometry, not $JOB.ext.
HANDWRITTEN_SH = """\
#!/bin/bash --login
export JOB=graphite_sp; export DIR=$SLURM_SUBMIT_DIR; export scratch=$SCRATCH/crys23
mkdir -p $scratch/$JOB && cd $scratch/$JOB
cp $DIR/$JOB.d12 INPUT; cp $DIR/graphite_opt.f34 fort.34
mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal >& $DIR/$JOB.out
"""


def _deck(tmp_path, text=EXTERNAL_DECK, name="graphite_sp.d12"):
    deck = tmp_path / name
    deck.write_text(text)
    return deck


# --- the shared helper -------------------------------------------------------

def test_geometry_keyword_is_the_record_after_the_title():
    from mace.submission.external_geometry import geometry_keyword
    assert geometry_keyword(EXTERNAL_DECK) == "EXTERNAL"
    assert geometry_keyword("t\nDLVINPUT\nEND\n") == "DLVINPUT"
    assert geometry_keyword("t\r\nexternal\r\nEND\r\n") == "EXTERNAL"
    assert geometry_keyword("t\nCRYSTAL\n0 0 0\n225\n4.2\n1\n12 0 0 0\nEND\n") is None
    # the title is free text, never the keyword
    assert geometry_keyword("EXTERNAL\nCRYSTAL\n0 0 0\n225\n4.2\n1\n12 0 0 0\nEND\n") is None


def test_sources_follow_the_mace_job_script_and_a_handwritten_one(tmp_path):
    from mace.submission.external_geometry import fort34_source, fort34_sources
    deck = _deck(tmp_path)
    assert fort34_sources(deck) == [tmp_path / "graphite_sp.ext"]
    assert fort34_source(deck) is None
    (tmp_path / "graphite_sp.sh").write_text(HANDWRITTEN_SH)
    assert fort34_sources(deck)[0] == tmp_path / "graphite_opt.f34"
    assert fort34_source(deck) is None
    (tmp_path / "graphite_opt.f34").write_text(GRAPHITE_F34)
    assert fort34_source(deck) == tmp_path / "graphite_opt.f34"


def test_a_generated_mace_job_script_names_the_ext(tmp_path):
    """The block submitcrystal23.sh writes resolves to <stem>.ext."""
    from mace.submission.external_geometry import fort34_sources
    deck = _deck(tmp_path)
    block = (REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh").read_text()
    start = block.index("# EXTERNAL geometry")
    (tmp_path / "graphite_sp.sh").write_text(
        "export JOB=graphite_sp\n" + block[start:block.index("\nfi\n", start) + 4])
    assert fort34_sources(deck) == [tmp_path / "graphite_sp.ext"]


def test_missing_message_is_plain_and_names_the_file(tmp_path):
    from mace.submission.external_geometry import missing_fort34_message
    deck = _deck(tmp_path)
    msg = missing_fort34_message(deck)
    assert "EXTERNAL" in msg and "fort.34" in msg and "graphite_sp.ext" in msg
    (tmp_path / "graphite_sp.sh").write_text(HANDWRITTEN_SH)
    msg = missing_fort34_message(deck)
    assert "graphite_sp.sh" in msg and "graphite_opt.f34" in msg
    # a same-format file the job script does not stage is pointed out
    (tmp_path / "graphite_sp.gui").write_text(GRAPHITE_F34)
    assert "graphite_sp.gui" in missing_fort34_message(deck)
    (tmp_path / "graphite_opt.f34").write_text(GRAPHITE_F34)
    assert missing_fort34_message(deck) is None
    assert missing_fort34_message(_deck(tmp_path, "t\nCRYSTAL\n0 0 0\n225\n4.2\n1\n12 0 0 0\nEND\n",
                                        "mgo.d12")) is None


# --- the GUI reader ------------------------------------------------------------

@needs_ase
def test_reader_draws_the_staged_fort34(tmp_path):
    from mace.gui import structures
    deck = _deck(tmp_path)
    (tmp_path / "graphite_sp.ext").write_text(GRAPHITE_F34)
    s = structures.read_structure(deck)
    assert len(s["numbers"]) == 6 and set(s["numbers"]) == {6}
    assert s["source"]["kind"] == "d12"
    assert s["source"]["geometry_from"] == str(tmp_path / "graphite_sp.ext")
    assert s["source"]["geometry_staged"] is True


@needs_ase
def test_reader_follows_a_handwritten_job_script(tmp_path):
    from mace.gui import structures
    deck = _deck(tmp_path)
    (tmp_path / "graphite_sp.sh").write_text(HANDWRITTEN_SH)
    (tmp_path / "graphite_opt.f34").write_text(GRAPHITE_F34)
    s = structures.read_structure(deck)
    assert s["source"]["geometry_from"].endswith("graphite_opt.f34")
    assert len(s["numbers"]) == 6


@needs_ase
def test_reader_falls_back_to_the_runs_output_labeled(tmp_path):
    from mace.gui import structures
    deck = _deck(tmp_path)
    shutil.copy(SI_OUT, tmp_path / "graphite_sp.out")
    s = structures.read_structure(deck)
    assert s["source"]["geometry_from"] == str(tmp_path / "graphite_sp.out")
    assert s["source"]["geometry_staged"] is False
    assert set(s["numbers"]) == {14}


def test_reader_without_a_companion_says_what_to_do(tmp_path):
    from mace.gui import structures
    deck = _deck(tmp_path)
    with pytest.raises(ValueError) as exc:
        structures.read_structure(deck)
    msg = str(exc.value)
    assert "cell_parameters" not in msg
    assert "EXTERNAL" in msg and "graphite_sp.ext" in msg and ".out" in msg


# Other geometry-block forms (decks written by hand: IFLAG=1 space group
# symbol, IFHR=1 rhombohedral axes, a broken record).
NACL_ATOMS = "2\n11 0 0 0\n17 0.5 0.5 0.5\nEND\n"


@needs_ase
def test_space_group_given_as_a_symbol(tmp_path):
    from mace.gui import structures
    s = structures.read_structure(_deck(tmp_path, "t\nCRYSTAL\n1 0 0\nF M 3 M\n5.64\n" + NACL_ATOMS + BASIS,
                                        "nacl.d12"))
    assert len(s["numbers"]) == 8
    assert s["source"]["deck_spacegroup"] == 225


@needs_ase
def test_rhombohedral_axes_give_the_rhombohedral_cell(tmp_path):
    """IFHR=1: the cell record is a, alpha of the rhombohedral cell, which
    holds ONE Bi at the origin in R-3m (166), not three."""
    from mace.gui import structures
    s = structures.read_structure(_deck(tmp_path, "t\nCRYSTAL\n0 1 0\n166\n4.75 57.2\n1\n83 0 0 0\nEND\n" + BASIS,
                                        "bi.d12"))
    assert len(s["numbers"]) == 1
    lat = s["lattice"]
    assert abs(lat["a"] - 4.75) < 1e-6 and abs(lat["alpha"] - 57.2) < 1e-6


@needs_ase
def test_geometry_edits_after_the_atoms_are_labeled(tmp_path):
    from mace.gui import structures
    s = structures.read_structure(_deck(
        tmp_path, "t\nCRYSTAL\n0 0 0\n225\n5.64\n2\n11 0 0 0\n17 0.5 0.5 0.5\n"
                  "SUPERCEL\n2 0 0\n0 2 0\n0 0 2\nATOMSUBS\n1\n1 19\nEND\n" + BASIS, "nacl.d12"))
    assert s["source"]["geometry_edits"] == ["SUPERCEL", "ATOMSUBS"]


@needs_ase
@pytest.mark.parametrize("text", [
    "t\nCRYSTAL\n0 0 0\n225\nabc\n" + NACL_ATOMS,            # unreadable cell record
    "t\nCRYSTAL\n0 0 0\n225\n5.64\n3\n11 0 0 0\n17 0.5 0.5 0.5\nEND\n",   # fewer atoms than it says
    "t\nCRYSTAL\n1 0 0\nQ Q Q\n5.64\n" + NACL_ATOMS,          # unknown symbol
])
def test_an_unreadable_geometry_is_a_plain_value_error(tmp_path, text):
    from mace.gui import structures
    with pytest.raises(ValueError) as exc:
        structures.read_structure(_deck(tmp_path, text + BASIS, "bad.d12"))
    msg = str(exc.value)
    assert "bad.d12" in msg and "geometry" in msg
    assert "'cell_parameters'" not in msg and "'atoms'" not in msg and "NoneType" not in msg


# --- the GUI API -----------------------------------------------------------------

@pytest.fixture(scope="module")
def served_external(tmp_path_factory):
    root = tmp_path_factory.mktemp("gui_ext") / "decks"
    root.mkdir()
    (root / "no_companion.d12").write_text(EXTERNAL_DECK.replace("graphite_sp", "no_companion"))
    (root / "with_ext.d12").write_text(EXTERNAL_DECK.replace("graphite_sp", "with_ext"))
    (root / "with_ext.ext").write_text(GRAPHITE_F34)
    (root / "broken.d12").write_text("t\nCRYSTAL\n0 0 0\n225\nabc\n" + NACL_ATOMS + BASIS)
    proc, base, _ = _start(root)
    yield root, base
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


def test_api_external_deck_without_companion_is_a_readable_400(served_external):
    _, base = served_external
    status, body = _json(base, "api/structure?path=no_companion.d12")
    assert status == 400, body
    assert "not found" not in body["error"] and "cell_parameters" not in body["error"]
    assert "EXTERNAL" in body["error"] and "no_companion.ext" in body["error"]


@needs_ase
def test_api_external_deck_with_its_ext_draws(served_external):
    _, base = served_external
    status, body = _json(base, "api/structure?path=with_ext.d12")
    assert status == 200, body
    assert len(body["numbers"]) == 6
    assert body["source"]["geometry_from"].endswith("with_ext.ext")


@needs_ase
def test_api_redraws_when_only_the_companion_changes(served_external):
    """The parse cache keyed only on the deck's own stat: a new .ext (or a
    .out appearing) kept the old drawing, or the old error."""
    root, base = served_external
    deck = root / "changing.d12"
    deck.write_text(EXTERNAL_DECK.replace("graphite_sp", "changing"))
    status, body = _json(base, "api/structure?path=changing.d12")
    assert status == 400, body
    ext = root / "changing.ext"
    ext.write_text(GRAPHITE_F34)
    status, body = _json(base, "api/structure?path=changing.d12")
    assert status == 200 and len(body["numbers"]) == 6, body
    # the same file with two atoms fewer (the last two records dropped)
    lines = GRAPHITE_F34.splitlines()
    lines[9] = "   4"
    ext.write_text("\n".join(lines[:-2]) + "\n")
    os.utime(ext, ns=(ext.stat().st_atime_ns, ext.stat().st_mtime_ns + 10**9))
    status, body = _json(base, "api/structure?path=changing.d12")
    assert status == 200 and len(body["numbers"]) == 4, len(body.get("numbers", []))


@needs_ase
def test_api_unreadable_geometry_is_a_readable_400(served_external):
    _, base = served_external
    status, body = _json(base, "api/structure?path=broken.d12")
    assert status == 400, body
    assert "not found" not in body["error"] and "geometry" in body["error"]


# --- mace preflight / mace submit -------------------------------------------------

def test_preflight_names_the_missing_fort34(tmp_path):
    from mace.submission import preflight as pf
    deck = _deck(tmp_path)
    result = pf.preflight_deck(deck, runner=None)
    assert result.status == pf.ERROR
    assert "graphite_sp.ext" in result.reason


def test_preflight_stages_what_a_handwritten_script_copies(tmp_path):
    from mace.submission import preflight as pf
    deck = _deck(tmp_path)
    (tmp_path / "graphite_sp.sh").write_text(HANDWRITTEN_SH)
    (tmp_path / "graphite_opt.f34").write_text(GRAPHITE_F34)
    staged = []

    def runner(workdir):
        f34 = Path(workdir) / "fort.34"
        staged.append(f34.read_text() if f34.exists() else None)
        raise pf.DeckStructureError("stop here")

    pf.preflight_deck(deck, runner=runner)
    assert staged == [GRAPHITE_F34]


@pytest.fixture
def fake_slurm(tmp_path):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "sbatch").write_text("#!/bin/bash\necho 'Submitted batch job 999003'\n")
    for name in ("squeue", "sacct", "scancel"):
        (bindir / name).write_text("#!/bin/bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    e = dict(os.environ)
    e["PATH"] = f"{bindir}:{e['PATH']}"
    e["MACE_NO_BANNER"] = "1"
    return e


def test_submit_warns_when_the_fort34_source_is_missing(tmp_path, fake_slurm):
    sub = tmp_path / "sub"
    sub.mkdir()
    _deck(sub)
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "sub", "--nosubmit"],
                          cwd=tmp_path, env=fake_slurm, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert "Warning:" in out and "graphite_sp.ext" in out, out
    # with the geometry next to it there is nothing to say
    (sub / "graphite_sp.ext").write_text(GRAPHITE_F34)
    (sub / "graphite_sp.sh").unlink()
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "sub", "--nosubmit"],
                          cwd=tmp_path, env=fake_slurm, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert not any("EXTERNAL" in line and "Warning" in line for line in out.splitlines()), out


def test_gui_submit_preview_mirror_keeps_the_fort34_source(tmp_path):
    """The preview runs the CLI in a mirror: the staged geometry file must be
    in it, or the CLI would warn about a file that is there."""
    from mace.gui import submit as gs
    deck = _deck(tmp_path)
    (tmp_path / "graphite_sp.ext").write_text(GRAPHITE_F34)
    sel = gs.selection(tmp_path, [deck])
    items, _ = gs.stage_plan(sel, tmp_path)
    assert (tmp_path / "graphite_sp.ext") in [p for p, _ in items]


# --- the corpus ---------------------------------------------------------------------

def test_every_corpus_deck_opens_or_says_why():
    """Each test/ .d12 either opens or is refused with a plain message (the GUI
    reads 3D decks; a SLAB/POLYMER/MOLECULE deck opens the .out of its run,
    or points at it when there is none). No raw parser error (KeyError,
    TypeError) reaches the user."""
    corpus_guard("**/*.d12")
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present")
    pytest.importorskip("ase")
    from mace.gui import structures
    opened = refused = 0
    for deck in sorted(TEST_DATA.rglob("*.d12")):
        try:
            structures.read_structure(deck)
            opened += 1
        except structures.OpenInstead as exc:
            # a SLAB/POLYMER/MOLECULE deck whose run's .out is next to it (e824dca3)
            assert exc.path.is_file() and exc.path.name in str(exc), f"{deck}: {exc}"
            assert re.search(r"is a (SLAB|POLYMER|MOLECULE) deck", str(exc)), f"{deck}: {exc}"
            refused += 1
        except ValueError as exc:
            assert "SLAB/POLYMER/MOLECULE" in str(exc), f"{deck}: {exc}"
            refused += 1
    assert opened >= 110, (opened, refused)

