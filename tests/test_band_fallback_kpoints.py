r"""BAND fallback k-paths: exact ISS-unit values, without the test/ corpus.

WHY THIS FILE EXISTS BESIDE tests/test_manual_conformance_crystal_d3.py
----------------------------------------------------------------------
The four BAND fallback branches are already covered there, but that coverage has
two limits this file removes:

* It is corpus-gated. Every BAND test in that file opens with
  ``find_data("BAND/1_dia*_band.out")``, and ``conftest.find_data`` skips when
  the gitignored ~12 GB ``test/`` tree is absent - so those assertions never run
  in CI. The tests here drive the same code with a path that does not exist and
  an overridden ``structure_info``, which is a documented guarded degradation
  (CRYSTALOptToD3.py:135-137 warns and returns the default info dict), so they
  run everywhere. No CRYSTAL text is fabricated.
* It is weakly discriminating. Those tests assert "all tokens are integers" and
  ``int(header[1]) == used["shrink"]``. The second is close to a tautology:
  _write_band_d3 reads ``config["shrink"]`` (CRYSTALOptToD3.py:604) and writes
  it into the header verbatim (:615), so it holds on an unpatched run too. And
  a fallback that scaled by the WRONG factor still produces integers. The tests
  here pin the exact integer segment lists, and additionally exercise a shrink
  that is too small (2), which forces ``scale_kpoint_segments`` to adjust it
  upward (2 -> 4) and breaks the tautology: header ISS then differs from the
  shrink that went in.

MANUAL BASIS FOR THE EXPECTED VALUES
------------------------------------
BAND's coordinate mode is defined at manual 24236-24264: NLINE is the "number
of lines in reciprocal space to be explored (max 20))" (24240), ISS is the
"shrinking factor in terms of which the coordinates of the extremes of the
segments are expressed" (24242-24243), and each of the NLINE records carries
"integers that define the starting point of the line (I1/ISS b1 + I2/ISS b2 +
I3/ISS b3)" and the matching J1,J2,J3 for the final point (24257-24264). The
same definition is repeated for the BAND sub-keyword of ANHARM (4346-4358) and
for phonon BANDS (17144-17153). Manual 24306 is explicit that "The only purpose
of ISS is to express the extremes of the segments in integer units" - it does
not change the k-point density - so scaling the fractional vectors by ISS is the
only legal way to write them.

The oracle for the fcc values is the manual's own MgO pair of equivalent decks
(24284-24295): "2 0 30 1 18 1 0 / G X / X W" and "2 12 30 1 18 1 0 /
0 0 0 6 0 6 / 6 0 6 6 3 9". Pairing them line for line at ISS=12 gives
G=(0,0,0), X=(6,0,6) and W=(6,3,9) - i.e. X=(0.5,0,0.5), W=(0.5,0.25,0.75).
test_seekpath_fallback_reproduces_the_manuals_mgo_fcc_points asserts exactly
those two triples come out of the fallback.

The oracle for the simple-cubic values is Table 14.1 (24366-24455), "Labels and
fractional coordinates ... of the special points recognized in input for each
Bravais lattice". Its P Cubic row (24372-24380) gives M=(1/2,1/2,0),
R=(1/2,1/2,1/2), X=(0,1/2,0) - which at ISS=16 is exactly the 8s in
_SIMPLE_CUBIC_DEFAULT[16] for MACE's M-GAMMA-R-X-GAMMA path.
test_literature_fallback_reproduces_table_14_1_simple_cubic_points pins it, so
those literals are anchored to the manual rather than to today's output. (The
same table's FC Cubic row, 24381-24391, independently repeats X=(1/2,0,1/2),
L=(1/2,1/2,1/2), W=(1/2,1/4,3/4) - a second confirmation of the MgO triples.)

_TRICLINIC_DEFAULT HAS NO MANUAL ORACLE - it is MACE-internal, pinned as
today's output only. This is a real gap, not an oversight: Tables 14.1 and 14.2
were read in full (24366-24455, 24459-24539) and contain NO triclinic lattice
row at all. The recognised labels are M,R,X,L,W,H,P,N,K,A,T,F,B,C,D,E,Y,Z,S,U
across P/FC/BC cubic, hexagonal, P trigonal, rhombohedral, P/AC monoclinic,
P/FC/AC/BC orthorhombic and P/BC tetragonal. MACE's triclinic path even opens
on label 'V', which appears nowhere in either table - and manual 24315-24316
says the ISS=0 label form "does not recognize the labels of every special
points; the ones recognized are only those reported in tables 14.1 and 14.2".
That constrains only the label form, and these decks are written in coordinate
form, so nothing here is provably wrong; it is simply unverifiable against the
manual. No code is changed to "fix" the triclinic table on that basis. Note the
interactive harness below can ONLY ever reach triclinic: the space-group scan
is inline in configure_band_calculation (d3_interactive.py:570-622, defaulting
space_group=1 / lattice_type='P' at :577-578) and cannot be redirected without
a real .out, so it always lands on the P1 default.

REACHABILITY - READ BEFORE TRUSTING "FALLBACK COVERED"
-----------------------------------------------------
All four fallbacks are currently UNREACHABLE from real input; the tests below
enter them only because the tests patch the upstream lookup to return a falsy
value. Two independent checks, both re-runnable:

1. ``get_seekpath_full_kpath`` (Crystal_d3/d3_kpoints.py:2731-2852) has four
   Return nodes - lines 2777, 2834, 2844, 2849 - and every one returns a
   2-element tuple, so ``if result:`` (CRYSTALOptToD3.py:1098,
   d3_interactive.py:721) can never be False. Re-verify structurally rather
   than by grepping for ``return None`` (a bare ``return``, ``return ()`` or
   ``return 0`` would all revive the branch and slip past a grep)::

       import ast, pathlib
       t = ast.parse(pathlib.Path("Crystal_d3/d3_kpoints.py").read_text())
       fn = next(n for n in ast.walk(t)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "get_seekpath_full_kpath")
       assert all(isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2
                  for r in ast.walk(fn) if isinstance(r, ast.Return))

2. Enumerating all 230 space groups against every uppercase centering letter,
   ``get_literature_kpath_vectors`` and ``get_kpoint_coordinates_from_labels``
   returned a falsy result 0 times, and all 14 crystal systems
   ``get_crystal_system_from_space_group`` can return have entries in both
   tables.

So "the fallback is covered" must NOT be read as "a seekpath-less compute node
exercises this code". It means the branch is pinned should it ever become live.

OPEN QUESTIONS FOR A HUMAN (deliberately not resolved in code here)
-------------------------------------------------------------------
1. Either ``get_seekpath_full_kpath`` should be able to return None, or the two
   ``if result:`` fallbacks are dead code. Which one is the intent is a
   MACE-internal contract the CRYSTAL23 manual does not settle; deleting or
   rewiring either side would change which k-path real jobs get, so nothing is
   changed here.
2. Same for the ``if frac_segments:`` guards on
   ``get_literature_kpath_vectors``.
3. These tests pass a non-existent .out and override ``structure_info`` rather
   than adding a synthetic CRYSTAL output. The corpus-backed tests in
   tests/test_manual_conformance_crystal_d3.py are kept, not replaced.
4. The generator's literature fallback (CRYSTALOptToD3.py:1171-1183) never
   resets ``config["kpath_source"]``, which :1157 already set to "literature",
   so a deck built from the *standard* label path is still labelled
   "Literature". Cosmetic today (the field only reaches the title), recorded so
   it is not mistaken for verified-correct.
5. MEASURED, NOT ASSERTED: the SeeK-path fallback deck's title contradicts its
   own records, and a plotter consumes that title. At site A
   (CRYSTALOptToD3.py:1140-1153) with sg 225 / 'F' / shrink 12 the deck is::

       BAND
       fb - Band Structure - default - GAMMA-X-W-K-GAMMA-L-U-W-|-L-K-|-U-X
       4 12 1000 1 100 1 0
       6 0 6  0 0 0
       0 0 0  6 6 6
       6 6 6  6 3 9
       6 3 9  0 0 0
       END

   The title advertises a 12-label SeeK-path (which the plotter merges to 10
   ticks across the two '|' discontinuities) while the four records are the
   5-node path X-GAMMA-L-W-GAMMA the branch actually emitted. Root cause: :1151
   stores ``config["path_labels"]``, but _write_band_d3 takes the
   ``elif config.get("seekpath_full")`` branch (:387-403) and re-derives labels
   from ``get_seekpath_labels``, ignoring ``path_labels``. Running
   ``Plotting/ipBANDS_V2.py`` ``parse_kpoint_path_and_segments`` (:169, which
   reads line 2, splits on " - " and takes the last field) on that deck returns
   10 labels - ['G','X','W','K','G','L','U','W|L','K|U','X'] - for a 5-node band
   structure, i.e. wrong high-symmetry tick labels on the plot. The same
   mismatch applies to the interactive format-4 fallback, whose config also
   carries ``seekpath_full=True``. (mace/utils/property_extractor.py:1345 is
   accidentally safe: its ``re.search(r'SeeKPath[^-]*-\s*(.+)$', title)``
   requires "SeeKPath" in the title and the fallback title says "default", so
   it returns [] rather than the wrong labels.) It is recorded rather than asserted because
   asserting today's title would enshrine the mismatch and asserting the correct
   invariant would fail; fixing :387-403 or :1151 changes generated decks and is
   a human decision outside a test-only change.
"""
import sys
from pathlib import Path

import pytest

from conftest import find_data

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen
import d3_interactive as d3int
import d3_kpoints  # noqa: F401  (ensures the bare-name copy is loaded)


# ---------------------------------------------------------------- helpers ---

def _kpoint_module_copies():
    """Every loaded copy of d3_kpoints.

    Crystal_d3/ has no __init__.py but still imports as a PEP-420 namespace
    package, so CRYSTALOptToD3's top-level ``from Crystal_d3.d3_kpoints import
    ...`` (CRYSTALOptToD3.py:43-46) and its function-local ``from d3_kpoints
    import ...`` (:1092-1093) bind two DISTINCT module objects.

    The copy that matters here is the BARE one. The three fallback guards -
    ``if result:`` (:1098) and ``if frac_segments:`` (:1130, :1159) - resolve
    ``get_seekpath_full_kpath`` and ``get_literature_kpath_vectors`` from the
    function-local import at :1092-1093, which is unconditionally spelled
    ``from d3_kpoints import ...``. That is why the pre-existing tests in
    tests/test_manual_conformance_crystal_d3.py:148-149 patch only
    ``d3_kpoints`` and work today.

    Patching only the DOTTED copy does not fail silently: it leaves the real
    lookup live, and the exact-segment assertions below then fail loudly
    (measured at sg 225/'F', shrink 12: 9 segments with kpath_source
    "seekpath_inv", against _FCC_DEFAULT[12]'s 4). Patching both copies is
    defensive future-proofing against the import spelling at :1092-1093
    changing, not a fix for a currently-silent failure.
    """
    copies = [mod for name, mod in list(sys.modules.items())
              if mod is not None
              and (name == "d3_kpoints" or name.endswith(".d3_kpoints"))]
    assert copies, "d3_kpoints is not importable"
    return copies


def _patch_kpoints(monkeypatch, name, value):
    for mod in _kpoint_module_copies():
        monkeypatch.setattr(mod, name, value)


def _run_generator(tmp_path, monkeypatch, *, space_group, lattice_type, shrink,
                   config):
    """Drive D3Generator.generate_d3 for BAND and return (config, deck text).

    The .out does not exist on purpose: CRYSTALOptToD3.py:135-137 warns and
    returns the default structure_info, which is then overridden here. That
    keeps the test corpus-free without inventing CRYSTAL output text.

    ``extract_and_process_shrink`` is patched on the CRYSTALOptToD3 module
    because :1092-1093 re-imports only four names and that is not one of them,
    so the fallback bodies resolve it from the module globals.
    """
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    monkeypatch.setattr(d3gen, "extract_and_process_shrink",
                        lambda *a, **k: shrink)

    gen = d3gen.D3Generator(str(tmp_path / "fb.out"), "BAND",
                            output_dir=str(tmp_path))
    gen.structure_info["space_group"] = space_group
    gen.structure_info["lattice_type"] = lattice_type

    used = gen.generate_d3(shared_config=dict(config, path_method="coordinates",
                                              auto_path=True, n_points=1000))

    decks = sorted(tmp_path.glob("*.d3"))
    assert [d.name for d in decks] == ["fb_band.d3"], decks
    return used, decks[0].read_text()


class _ScriptedPrompts:
    """Stand-in for the menu_nav readers d3_interactive binds at import.

    d3_interactive.py:44-46 binds _nav_read/_nav_int/_nav_float as module
    globals, so they must be patched on the module, not on menu_nav. Answers are
    matched on a prompt substring; unmatched prompts take their default.
    """

    def __init__(self, answers):
        self.answers = answers

    def _lookup(self, prompt):
        for fragment, value in self.answers.items():
            if fragment in prompt:
                return value
        return None

    def read(self, prompt="", valid_set=None):
        value = self._lookup(prompt)
        return "" if value is None else str(value)

    def nav_int(self, prompt="", default=None, choices=None):
        value = self._lookup(prompt)
        if value is None:
            return int(default if default is not None else 0)
        return int(value)

    def nav_float(self, prompt="", default=None):
        value = self._lookup(prompt)
        if value is None:
            return float(default if default is not None else 0.0)
        return float(value)

    def install(self, monkeypatch):
        monkeypatch.setattr(d3int, "_nav_read", self.read)
        monkeypatch.setattr(d3int, "_nav_int", self.nav_int)
        monkeypatch.setattr(d3int, "_nav_float", self.nav_float)
        return self


def _run_interactive(tmp_path, monkeypatch, *, format_choice, shrink):
    """Drive configure_band_calculation down the automatic-path route.

    The .out does not exist, so the space-group scan (d3_interactive.py:583-621)
    falls to its documented "Using default P1" branch: space group 1, lattice
    'P', triclinic path. Both format 3 and 4 pre-set band_config["shrink"] from
    extract_and_process_shrink at :688-689 before their branch runs.
    """
    monkeypatch.setattr(d3int, "extract_and_process_shrink", lambda *a, **k: shrink)
    _ScriptedPrompts({"Select method": "1",
                      "Select format": format_choice}).install(monkeypatch)
    return d3int.configure_band_calculation(str(tmp_path / "fb.out"))


def _deck_records(deck):
    """Coordinate records of a BAND deck (everything after the NLINE header)."""
    lines = deck.strip().splitlines()
    return [ln for ln in lines[3:] if ln.strip() and ln.strip() != "END"]


def _fractional_endpoints(deck):
    """Segment endpoints of a coordinate-mode BAND deck, divided by its ISS."""
    lines = deck.strip().splitlines()
    iss = int(lines[2].split()[1])
    assert iss > 0, "coordinate mode requires ISS > 0 (manual 24257)"
    points = set()
    for record in _deck_records(deck):
        tokens = record.split()
        if len(tokens) != 6:
            continue
        v = [int(t) for t in tokens]
        points.add((v[0] / iss, v[1] / iss, v[2] / iss))
        points.add((v[3] / iss, v[4] / iss, v[5] / iss))
    return points


# --------------------------------------------------------- expected values ---

# Space group 225 (Fm-3m) with centering 'F' -> get_crystal_system_from_space_group
# resolves "cubic_fc"; 221 (Pm-3m) with 'P' -> "cubic_simple". 221 is used for the
# literature-fallback cases rather than 225 because 225 IS F-centred: MACE takes
# the centering letter from the first character of the space-group symbol, so
# (225, 'P') can never come out of its own parser. The tests below drive the
# GENERATOR, whose parser does this at CRYSTALOptToD3.py:178
# (``info['lattice_type'] = symbol[0]``); d3_interactive.py:590 uses the identical
# convention for the interactive path. Both resolve to the same table and give
# byte-identical expected values.
_FCC_DEFAULT = {
    12: [[6, 0, 6, 0, 0, 0], [0, 0, 0, 6, 6, 6],
         [6, 6, 6, 6, 3, 9], [6, 3, 9, 0, 0, 0]],
    4: [[2, 0, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2],
        [2, 2, 2, 2, 1, 3], [2, 1, 3, 0, 0, 0]],
}

# Path M-GAMMA-R-X-GAMMA. Manual Table 14.1, P Cubic row (24372-24380):
# M=(1/2,1/2,0), R=(1/2,1/2,1/2), X=(0,1/2,0) - so at ISS=16 the non-zero
# entries are all 8. Pinned by
# test_literature_fallback_reproduces_table_14_1_simple_cubic_points.
_SIMPLE_CUBIC_DEFAULT = {
    16: [[8, 8, 0, 0, 0, 0], [0, 0, 0, 8, 8, 8],
         [8, 8, 8, 0, 8, 0], [0, 8, 0, 0, 0, 0]],
    4: [[2, 2, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2],
        [2, 2, 2, 0, 2, 0], [0, 2, 0, 0, 0, 0]],
}

# NO MANUAL ORACLE - see "MANUAL BASIS" above. Tables 14.1/14.2 have no
# triclinic row, and MACE's path opens on 'V', a label absent from both. These
# literals are today's MACE output, pinned to detect drift, nothing more.
_TRICLINIC_DEFAULT = {
    16: [[8, 8, 0, 0, 8, 0], [0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8],
         [0, 0, 8, 0, 8, 8], [0, 8, 8, 8, 8, 8], [8, 8, 8, 0, 0, 0],
         [0, 0, 0, 8, 0, 0], [8, 0, 0, 8, 0, 8], [8, 0, 8, 0, 0, 0]],
    4: [[2, 2, 0, 0, 2, 0], [0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2],
        [0, 0, 2, 0, 2, 2], [0, 2, 2, 2, 2, 2], [2, 2, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0], [2, 0, 0, 2, 0, 2], [2, 0, 2, 0, 0, 0]],
}


def _assert_integer_segments(segments):
    assert segments, "a fallback must still produce a path"
    for segment in segments:
        for value in segment:
            # bool is an int subclass; the manual wants plain integers.
            assert type(value) is int, segments


# ------------------------------------- site A: generator, SeeK-path branch ---

@pytest.mark.parametrize("shrink_in,shrink_out", [(12, 12), (2, 4)])
def test_generator_seekpath_fallback_writes_iss_scaled_integers(
        tmp_path, monkeypatch, shrink_in, shrink_out):
    """CRYSTALOptToD3.py:1140-1153 - SeeK-path AND literature both unavailable.

    Fails against the old behaviour, passes against the new: the pre-fix body
    was ``coord_segments = get_kpoint_coordinates_from_labels(...)`` with no
    scaling and no ``config["shrink"]``, so the deck carried "0.5 0.0 0.5 ..."
    against the writer's default ISS=16. CRYSTAL reads those fields as integers
    (manual 24257-24264), so the path silently collapsed.

    The shrink_in=2 case is the discriminating one: scale_kpoint_segments must
    raise ISS to 4 for the quarter-coordinates of W=(0.5,0.25,0.75) to be
    expressible in integer units, so header ISS != the shrink that went in and
    the ``int(header[1]) == used["shrink"]`` check cannot pass by construction.

    Note the ``max(4, ...)`` floor in get_minimum_shrink_for_segments
    (d3_kpoints.py:573, commented "Minimum of 4 for robustness") is a MACE
    choice, NOT a CRYSTAL requirement: manual 24306 makes ISS a pure unit of
    expression that "does not determine the density of k points". Here the
    floor is not what binds - W's 1/4 genuinely needs denominator 4 - but at
    site B, whose coordinates are all halves, ISS 2 would have sufficed and
    only the floor raises it. Do not read ISS>=4 as something CRYSTAL demands.
    """
    _patch_kpoints(monkeypatch, "get_seekpath_full_kpath", lambda *a, **k: None)
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])

    used, deck = _run_generator(tmp_path, monkeypatch, space_group=225,
                                lattice_type="F", shrink=shrink_in,
                                config={"seekpath_full": True})

    _assert_integer_segments(used["segments"])
    assert used["segments"] == _FCC_DEFAULT[shrink_out]
    assert used["shrink"] == shrink_out
    assert used["kpath_source"] == "default"
    header = deck.strip().splitlines()[2].split()
    assert int(header[0]) == len(_FCC_DEFAULT[shrink_out])  # NLINE, manual 24240
    assert int(header[1]) == shrink_out                     # ISS, manual 24242


def test_generator_seekpath_fallback_deck_records_are_exact(tmp_path, monkeypatch):
    """The written deck, not just the config, carries the ISS-unit integers.

    Fails against the old behaviour (records were fractional), passes now.
    """
    _patch_kpoints(monkeypatch, "get_seekpath_full_kpath", lambda *a, **k: None)
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])

    _, deck = _run_generator(tmp_path, monkeypatch, space_group=225,
                             lattice_type="F", shrink=12,
                             config={"seekpath_full": True})

    assert _deck_records(deck) == ["6 0 6  0 0 0",
                                   "0 0 0  6 6 6",
                                   "6 6 6  6 3 9",
                                   "6 3 9  0 0 0"], deck
    assert deck.strip().splitlines()[0] == "BAND"
    assert deck.strip().endswith("END")


# ------------------------------------ site B: generator, literature branch ---

@pytest.mark.parametrize("shrink_in,shrink_out", [(16, 16), (2, 4)])
def test_generator_literature_fallback_writes_iss_scaled_integers(
        tmp_path, monkeypatch, shrink_in, shrink_out):
    """CRYSTALOptToD3.py:1171-1183 - literature table has no entry.

    Fails against the old behaviour, passes against the new: the pre-fix body
    stored the fractional vectors and never touched ``config["shrink"]``.
    """
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])

    used, deck = _run_generator(tmp_path, monkeypatch, space_group=221,
                                lattice_type="P", shrink=shrink_in,
                                config={"literature_path": True})

    _assert_integer_segments(used["segments"])
    assert used["segments"] == _SIMPLE_CUBIC_DEFAULT[shrink_out]
    assert used["shrink"] == shrink_out
    header = deck.strip().splitlines()[2].split()
    assert int(header[0]) == len(_SIMPLE_CUBIC_DEFAULT[shrink_out])
    assert int(header[1]) == shrink_out


# ------------------------------ sites C and D: d3_interactive, formats 3/4 ---

@pytest.mark.parametrize("shrink_in,shrink_out", [(16, 16), (2, 4)])
def test_interactive_literature_fallback_writes_iss_scaled_integers(
        tmp_path, monkeypatch, shrink_in, shrink_out):
    """d3_interactive.py:705-713 - menu format 3, literature table empty.

    Fails against the old behaviour, passes against the new: the pre-fix body
    handed CRYSTALOptToD3 a config whose "segments" were fractional while
    "shrink" kept the value set at :688-689.
    """
    monkeypatch.setattr(d3int, "get_literature_kpath_vectors", lambda *a, **k: [])

    config = _run_interactive(tmp_path, monkeypatch, format_choice="3",
                              shrink=shrink_in)

    _assert_integer_segments(config["segments"])
    assert config["segments"] == _TRICLINIC_DEFAULT[shrink_out]
    assert config["shrink"] == shrink_out


@pytest.mark.parametrize("shrink_in,shrink_out", [(16, 16), (2, 4)])
def test_interactive_seekpath_fallback_writes_iss_scaled_integers(
        tmp_path, monkeypatch, shrink_in, shrink_out):
    """d3_interactive.py:744-752 - menu format 4, SeeK-path unavailable.

    Fails against the old behaviour, passes against the new: same unscaled
    fractional storage as format 3.
    """
    monkeypatch.setattr(d3int, "get_seekpath_full_kpath", lambda *a, **k: None)

    config = _run_interactive(tmp_path, monkeypatch, format_choice="4",
                              shrink=shrink_in)

    _assert_integer_segments(config["segments"])
    assert config["segments"] == _TRICLINIC_DEFAULT[shrink_out]
    assert config["shrink"] == shrink_out


# ------------------------------------------------- manual / corpus oracles ---

def test_seekpath_fallback_reproduces_the_manuals_mgo_fcc_points(tmp_path, monkeypatch):
    """The fcc literals come from the manual, not from today's output.

    Manual 24284-24295 gives two equivalent MgO (fcc) decks - one by label,
    "G X / X W", one by coordinate at ISS=12, "0 0 0 6 0 6 / 6 0 6 6 3 9".
    Pairing them fixes X=(6,0,6) and W=(6,3,9) in units of 1/12. Driving the
    fallback at the same ISS must reproduce both triples.

    Fails against the old behaviour (which emitted 0.5/0.25/0.75 floats),
    passes against the new. This validates the ORACLE the tests above use; it
    does not by itself prove which branch produced the values.
    """
    _patch_kpoints(monkeypatch, "get_seekpath_full_kpath", lambda *a, **k: None)
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])

    used, deck = _run_generator(tmp_path, monkeypatch, space_group=225,
                                lattice_type="F", shrink=12,
                                config={"seekpath_full": True})

    assert used["shrink"] == 12, "the manual's MgO coordinate deck uses ISS=12"
    endpoints = {tuple(seg[:3]) for seg in used["segments"]}
    endpoints |= {tuple(seg[3:]) for seg in used["segments"]}
    assert (6, 0, 6) in endpoints, deck   # X, manual 24294
    assert (6, 3, 9) in endpoints, deck   # W, manual 24295


def test_literature_fallback_reproduces_table_14_1_simple_cubic_points(
        tmp_path, monkeypatch):
    """The simple-cubic literals come from the manual, not from today's output.

    Manual Table 14.1 (24366-24455) tabulates the special points "recognized in
    input for each Bravais lattice". Its P Cubic row (24372-24380) gives
    M=(1/2,1/2,0), R=(1/2,1/2,1/2), X=(0,1/2,0). Driving the literature
    fallback at ISS=16 must reproduce all three in 1/16 units, which is what
    makes _SIMPLE_CUBIC_DEFAULT[16] a manual-anchored expectation rather than a
    self-referential snapshot.

    Fails against the old behaviour (which stored 0.5 floats, so no integer
    triple matched), passes against the new.
    """
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])

    used, deck = _run_generator(tmp_path, monkeypatch, space_group=221,
                                lattice_type="P", shrink=16,
                                config={"literature_path": True})

    assert used["shrink"] == 16
    endpoints = {tuple(seg[:3]) for seg in used["segments"]}
    endpoints |= {tuple(seg[3:]) for seg in used["segments"]}
    assert (8, 8, 0) in endpoints, deck   # M, manual 24372-24374
    assert (8, 8, 8) in endpoints, deck   # R, manual 24375-24378
    assert (0, 8, 0) in endpoints, deck   # X, manual 24379-24380
    assert (0, 0, 0) in endpoints, deck   # GAMMA


def test_fallback_x_point_matches_the_shipped_reference_band_deck(tmp_path, monkeypatch):
    """Second oracle: a real CRYSTAL-accepted deck under test/, not a fixture.

    test/BAND/*_band.d3 is a shipped fcc diamond BAND deck with ISS=16 whose
    records include "0 0 0  8 0 8", i.e. X=(0.5,0,0.5). The fallback writes
    (6,0,6) at ISS=12, which is the same fractional point. Skips (does not
    fail) when the gitignored corpus is absent.

    Fails against the old behaviour: the fallback's endpoints were already
    fractional, so dividing them by ISS again gave (0.03125,0,0.03125).
    """
    reference = find_data("BAND/1_dia*_band.d3", must_contain="BAND")
    reference_points = _fractional_endpoints(reference.read_text())
    assert (0.5, 0.0, 0.5) in reference_points, reference

    _patch_kpoints(monkeypatch, "get_seekpath_full_kpath", lambda *a, **k: None)
    _patch_kpoints(monkeypatch, "get_literature_kpath_vectors", lambda *a, **k: [])
    _, deck = _run_generator(tmp_path, monkeypatch, space_group=225,
                             lattice_type="F", shrink=12,
                             config={"seekpath_full": True})

    assert (0.5, 0.0, 0.5) in _fractional_endpoints(deck), deck


# ------------------------------------------------------ negative controls ---
#
# Anti-vacuous-coverage guards. Each re-runs a positive test WITHOUT the patch
# that disables the upstream lookup, and asserts the result differs from the
# fallback expectation. Without these, a fallback that was never entered (or an
# upstream table that happened to agree) would look covered. Note these do NOT
# discriminate old from new behaviour - the fallback is not taken either way -
# so they are guards for the tests above, not regression tests in their own
# right. Only inequality is asserted: the unpatched values depend on whether the
# optional seekpath library is installed, and pinning them here would make the
# file depend on that.

def test_generator_seekpath_branch_unpatched_does_not_hit_the_fallback(
        tmp_path, monkeypatch):
    """Guard: without the patch, sg 225/'F' does not produce the fallback path."""
    used, _ = _run_generator(tmp_path, monkeypatch, space_group=225,
                             lattice_type="F", shrink=12,
                             config={"seekpath_full": True})

    assert used["segments"] != _FCC_DEFAULT[12], (
        "the SeeK-path lookup now returns the fallback path, so "
        "test_generator_seekpath_fallback_writes_iss_scaled_integers no longer "
        "proves the fallback branch was entered")


def test_generator_literature_branch_unpatched_does_not_hit_the_fallback(
        tmp_path, monkeypatch):
    """Guard: without the patch, sg 221/'P' has a literature entry."""
    used, _ = _run_generator(tmp_path, monkeypatch, space_group=221,
                             lattice_type="P", shrink=16,
                             config={"literature_path": True})

    assert used["segments"] != _SIMPLE_CUBIC_DEFAULT[16], (
        "the literature table now returns the standard path, so the literature "
        "fallback test no longer proves the fallback branch was entered")


@pytest.mark.parametrize("format_choice", ["3", "4"])
def test_interactive_branch_unpatched_does_not_hit_the_fallback(
        tmp_path, monkeypatch, format_choice):
    """Guard: without the patch, formats 3 and 4 take their normal branch."""
    config = _run_interactive(tmp_path, monkeypatch,
                              format_choice=format_choice, shrink=16)

    assert config["segments"] != _TRICLINIC_DEFAULT[16], (
        f"menu format {format_choice} now returns the standard path, so its "
        "fallback test no longer proves the fallback branch was entered")
