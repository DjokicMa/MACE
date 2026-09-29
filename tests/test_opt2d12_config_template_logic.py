"""Corpus-free checks of applying one --config-file template to many files.

The real-file checks are in test_opt2d12_config_batch.py and skip without the
test/ corpus; these run in CI. The parsers are replaced by the dictionaries
they return, so main() runs end to end on empty placeholder files.
"""
import copy
import json
import sys

import pytest

import CRYSTALOptToD12 as M

AG_BASIS = ["247 3", "INPUT", "19. 0 1 2 1 0 0", "0 0 1 2.0 1.0", "1.0 1.0"]
BR_BASIS = ["35 2", "0 0 1 2.0 1.0", "1.0 1.0"]


def _out(atoms, spacegroup=225):
    return {
        "dimensionality": "CRYSTAL", "spacegroup": spacegroup, "origin_setting": "0 0 0",
        "conventional_cell": [6.0, 6.0, 6.0, 90.0, 90.0, 90.0],
        "coordinates": [{"atom_number": str(z), "x": f"{i / 4:.6f}", "y": "0.0",
                         "z": "0.0", "is_unique": True} for i, z in enumerate(atoms)],
        "functional": "B3LYP", "calculation_type": "OPT",
    }


def _deck_data(basis_records=None, basis="POB-TZVP-REV2", k_points="8 16"):
    data = {"functional": "B3LYP-D3", "dispersion": True, "dft_grid": "XLGRID",
            "calculation_type": "OPT", "k_points": k_points,
            "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}}
    if basis_records:
        data.update(basis_set_type="EXTERNAL", external_basis_data=basis_records)
    else:
        data.update(basis_set=basis, basis_set_type="INTERNAL")
    return data


PARENTS = {
    "agbr": (_out([247, 35]), _deck_data(AG_BASIS + BR_BASIS, k_points="5 10")),
    "nacl": (_out([11, 17]), _deck_data()),
}


class _Parser:
    def __init__(self, path, kind):
        self.stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self.kind = kind

    def parse(self):
        if self.stem == "broken":
            raise ValueError("no optimized geometry in this output")
        out, deck = PARENTS[self.stem]
        return copy.deepcopy(out if self.kind == "out" else deck)


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty

    def readline(self, *a):
        raise AssertionError("stdin was read")

    read = readline


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CrystalOutputParser", lambda p: _Parser(str(p), "out"))
    monkeypatch.setattr(M, "CrystalInputParser", lambda p: _Parser(str(p), "d12"))
    monkeypatch.chdir(tmp_path)
    for stem in PARENTS:
        (tmp_path / f"{stem}.out").write_text("")
        (tmp_path / f"{stem}.d12").write_text("")
    template = {"calculation_type": "SP", "functional": "PBE0", "dispersion": False,
                "dft_grid": "XXLGRID", "tolerances": {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9},
                "basis_set": M.PARENT_BASIS_MARKER, "basis_set_type": "EXTERNAL",
                "use_original_external_basis": True, "has_original_external_basis": True,
                # What earlier versions also saved: the source parent's records.
                "external_basis_data": ["47 1", "0 0 1 2.0 1.0", "9.9 9.9"],
                "spacegroup": 216, "write_only_unique": True}
    (tmp_path / "t.json").write_text(json.dumps(template))

    def run(*args, tty=False, answers=()):
        # tty=None: stdin closed (`<&-`), when sys.stdin is None.
        monkeypatch.setattr(sys, "stdin", None if tty is None else _Stdin(tty))
        asked = []
        replies = list(answers)

        def prompt(text, default="yes"):
            # A question nobody scripted gets its default, as Enter would.
            asked.append(text.strip())
            return replies.pop(0) if replies else default == "yes"
        monkeypatch.setattr(M, "yes_no_prompt", prompt)
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", *args])
        try:
            M.main()
            status = 0
        except SystemExit as e:
            status = e.code
        return status, asked
    return run


def _deck(tmp_path, stem, functional="PBE0"):
    return tmp_path / "sp" / f"{stem}_sp_{functional}_optimized.d12"


def test_batch_asks_nobody_and_keeps_each_parents_basis(batch, tmp_path, capsys):
    status, asked = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    out = capsys.readouterr()
    assert status == 0, out.err
    assert asked == []
    assert "2 written, 0 failed" in out.out
    agbr = _deck(tmp_path, "agbr").read_text()
    assert "\n".join(AG_BASIS + BR_BASIS) + "\n99 0\nEND\n" in agbr
    assert "9.9 9.9" not in agbr               # not the template parent's records
    assert "\n247 " in agbr and "BASISSET" not in agbr
    nacl = _deck(tmp_path, "nacl").read_text()
    assert "BASISSET\nPOB-TZVP-REV2\n" in nacl and "99 0" not in nacl
    for deck in (agbr, nacl):
        assert "PBE0\nXXLGRID\n" in deck and "TOLINTEG\n8 8 8 9 24\n" in deck
        assert deck.splitlines()[3] == "225"   # each structure's own group


def test_batch_at_a_terminal_asks_once(batch, tmp_path):
    status, asked = batch("--directory", ".", "--config-file", "t.json",
                          "--output-dir", "sp", tty=True, answers=[True])
    assert status == 0
    assert asked == ["Apply to all 2 files?"]
    assert _deck(tmp_path, "agbr").exists() and _deck(tmp_path, "nacl").exists()


def test_batch_at_a_terminal_declined_writes_nothing(batch, tmp_path):
    status, asked = batch("--directory", ".", "--config-file", "t.json",
                          "--output-dir", "sp", tty=True, answers=[False])
    assert status == 1
    assert asked == ["Apply to all 2 files?"]
    assert not (tmp_path / "sp").exists() or not list((tmp_path / "sp").iterdir())


def test_yes_skips_the_question(batch):
    status, asked = batch("--directory", ".", "--config-file", "t.json",
                          "--output-dir", "sp", "--yes", tty=True)
    assert status == 0 and asked == []


def test_several_out_files_pair_with_their_d12(batch, tmp_path):
    status, asked = batch("--out-file", "nacl.out", "agbr.out", "--config-file", "t.json",
                          "--output-dir", "sp", tty=True, answers=[True])
    assert status == 0 and asked == ["Apply to all 2 files?"]
    assert "99 0" in _deck(tmp_path, "agbr").read_text()


def test_one_file_asks_only_at_a_terminal(batch, tmp_path):
    status, asked = batch("--out-file", "nacl.out", "--d12-file", "nacl.d12",
                          "--config-file", "t.json", "--output-dir", "sp")
    assert status == 0 and asked == []
    status, asked = batch("--out-file", "nacl.out", "--d12-file", "nacl.d12",
                          "--config-file", "t.json", "--output-dir", "sp2",
                          tty=True, answers=[True])
    assert status == 0
    assert asked == ["Apply these settings from config file?"]


def test_a_failure_is_reported_and_fails_the_run(batch, tmp_path, capsys):
    (tmp_path / "broken.out").write_text("")
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    err = capsys.readouterr().err
    assert status == 1
    assert "2 written, 1 failed" in err
    assert "broken.out: could not parse the output file: no optimized geometry" in err


def test_a_missing_out_file_is_reported(batch, capsys):
    status, _ = batch("--out-file", "nacl.out", "gone.out", "--config-file", "t.json",
                      "--output-dir", "sp")
    assert status == 1
    assert "gone.out: file not found" in capsys.readouterr().err


def test_an_unreadable_config_stops_before_any_file(batch, tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{not json")
    status, _ = batch("--directory", ".", "--config-file", "bad.json", "--output-dir", "sp")
    assert status == 1
    assert "Cannot use config file bad.json" in capsys.readouterr().err
    assert not (tmp_path / "sp").exists() or not list((tmp_path / "sp").iterdir())


def test_d12_file_needs_a_single_out_file(batch):
    status, _ = batch("--out-file", "nacl.out", "agbr.out", "--d12-file", "nacl.d12")
    assert status == 2


def test_a_named_internal_basis_applies_to_every_file(batch, tmp_path):
    (tmp_path / "t.json").write_text(json.dumps(
        {"calculation_type": "SP", "functional": "PBE0", "basis_set": "POB-TZVP",
         "basis_set_type": "INTERNAL"}))
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    assert status == 0
    agbr = _deck(tmp_path, "agbr").read_text()
    assert "BASISSET\nPOB-TZVP\n" in agbr and "99 0" not in agbr
    # BASISSET takes the plain atomic number: silver is 47, not the ECP's 247.
    assert "\n47 " in agbr and "\n247 " not in agbr and " Ag\n" in agbr
    assert "BASISSET\nPOB-TZVP\n" in _deck(tmp_path, "nacl").read_text()


def test_a_p1_template_still_writes_the_asymmetric_unit(batch, tmp_path):
    PARENTS["cc"] = (_out([6, 6], spacegroup=227), _deck_data())
    PARENTS["cc"][0]["coordinates"][1]["is_unique"] = False
    try:
        for ext in ("out", "d12"):
            (tmp_path / f"cc.{ext}").write_text("")
        (tmp_path / "t.json").write_text(json.dumps(
            {"calculation_type": "SP", "functional": "PBE0", "write_only_unique": False}))
        status, _ = batch("--out-file", "cc.out", "--d12-file", "cc.d12",
                          "--config-file", "t.json", "--output-dir", "sp")
        assert status == 0
        lines = _deck(tmp_path, "cc").read_text().splitlines()
        assert lines[3] == "227" and lines[5] == "1"
    finally:
        del PARENTS["cc"]




# --- review fixes -----------------------------------------------------------

LEAD = (_out([82, 8], spacegroup=221), _deck_data())
THREE_C = {"calculation_type": "SP", "functional": "HSESOL3C", "is_3c_method": True,
           "basis_set": "SOLDEF2MSVP", "basis_set_type": "INTERNAL", "dispersion": False}


@pytest.fixture
def lead(tmp_path):
    PARENTS["pbo"] = copy.deepcopy(LEAD)
    for ext in ("out", "d12"):
        (tmp_path / f"pbo.{ext}").write_text("")
    (tmp_path / "t3c.json").write_text(json.dumps(THREE_C))
    yield
    del PARENTS["pbo"]


@pytest.mark.parametrize("tty", [True, False])
@pytest.mark.parametrize("how", [["--yes"], []])
def test_a_basis_without_an_element_fails_the_file_without_asking(
        batch, lead, tmp_path, capsys, tty, how):
    """SOLDEF2MSVP has no Pb. At a terminal the batch asked "continue anyway?"
    per file, Enter took the default yes, and a deck CRYSTAL rejects was
    counted as written. Now the file fails with the reason, terminal or not."""
    answers = [] if how else [True]
    status, asked = batch("--out-file", "nacl.out", "pbo.out", "--config-file", "t3c.json",
                          "--output-dir", "sp", *how, tty=tty, answers=answers)
    err = capsys.readouterr().err
    assert [q for q in asked if "Apply to all" not in q] == []
    assert status == 1
    assert "1 written, 1 failed" in err
    assert "pbo.out: basis set 'SOLDEF2MSVP' has no basis for Pb" in err
    assert not _deck(tmp_path, "pbo", "HSESOL3C").exists()
    assert _deck(tmp_path, "nacl", "HSESOL3C").exists()


@pytest.mark.parametrize("tty", [True, False, None])
def test_one_file_with_yes_fails_on_the_basis_without_asking(batch, lead, tmp_path, tty):
    status, asked = batch("--out-file", "pbo.out", "--config-file", "t3c.json",
                          "--output-dir", "sp", "--yes", tty=tty)
    assert asked == []
    assert status == 1
    assert not _deck(tmp_path, "pbo", "HSESOL3C").exists()


def test_closed_stdin_is_nobody_to_ask(batch, lead, tmp_path, capsys):
    """`<&-` leaves sys.stdin None: sys.stdin.isatty() crashed with
    AttributeError in main(), write_d12_file and the empty-directory check."""
    status, asked = batch("--directory", ".", "--config-file", "t.json", "--output-dir",
                          "sp", tty=None)
    assert status == 0 and asked == []
    status, asked = batch("--out-file", "pbo.out", "--config-file", "t3c.json",
                          "--output-dir", "sp3c", tty=None)
    assert status == 1 and asked == []
    assert "basis set 'SOLDEF2MSVP' has no basis for Pb" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    status, asked = batch("--directory", "empty", "--output-dir", "sp4", tty=None)
    assert status == 1 and asked == []


def test_a_lone_out_file_uses_the_d12_beside_it(batch, tmp_path, capsys):
    """A glob that matched one file ran from the .out alone: the external
    basis was replaced, while the log said the parent's basis was kept."""
    status, _ = batch("--out-file", "agbr.out", "--config-file", "t.json",
                      "--output-dir", "sp", "--yes")
    assert status == 0
    agbr = _deck(tmp_path, "agbr").read_text()
    assert "\n".join(AG_BASIS + BR_BASIS) + "\n99 0\nEND\n" in agbr
    assert "BASISSET" not in agbr
    assert "keeping this structure's own parent basis" in capsys.readouterr().out


def test_an_out_file_without_a_d12_says_so(batch, tmp_path, capsys):
    PARENTS["solo"] = (_out([11, 17]), _deck_data())
    try:
        (tmp_path / "solo.out").write_text("")
        status, _ = batch("--out-file", "solo.out", "--config-file", "t.json",
                          "--output-dir", "sp", "--yes")
        out = capsys.readouterr()
        assert status == 0
        assert "keeping this structure's own parent basis" not in out.out
        assert "converted from the .out alone" in out.err
        status, _ = batch("--out-file", "solo.out", "nacl.out", "--config-file", "t.json",
                          "--output-dir", "sp2", "--yes")
        out = capsys.readouterr()
        assert status == 0
        assert "solo.out: wrote sp2/solo_sp_PBE0_optimized.d12 (from the .out alone)" in out.out
    finally:
        del PARENTS["solo"]


def test_a_file_listed_twice_is_converted_once(batch, tmp_path, capsys):
    status, _ = batch("--out-file", "nacl.out", "./nacl.out", "agbr.out", "--config-file",
                      "t.json", "--output-dir", "sp", "--yes")
    out = capsys.readouterr()
    assert status == 0
    assert "2 written, 0 failed (of 2 files)" in out.out
    assert "./nacl.out is listed more than once" in out.err


def test_inputs_whose_decks_share_a_name_are_not_written(batch, tmp_path, capsys):
    """dupA/nacl.out and dupB/nacl.out with one --output-dir: the second deck
    overwrote the first and both were counted as written."""
    for d in ("dupA", "dupB"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "nacl.out").write_text("")
        (tmp_path / d / "nacl.d12").write_text("")
    status, _ = batch("--out-file", "dupA/nacl.out", "dupB/nacl.out", "agbr.out",
                      "--config-file", "t.json", "--output-dir", "sp", "--yes")
    err = capsys.readouterr().err
    assert status == 1
    assert "1 written, 2 failed" in err
    assert "dupA/nacl.out: FAILED: its deck would have the same name as the one from dupB/nacl.out" in err
    assert "dupB/nacl.out: FAILED: its deck would have the same name as the one from dupA/nacl.out" in err
    assert not _deck(tmp_path, "nacl").exists()
    # Beside their own .out files they do not collide.
    status, _ = batch("--out-file", "dupA/nacl.out", "dupB/nacl.out",
                      "--config-file", "t.json", "--yes")
    assert status == 0
    assert (tmp_path / "dupA" / "nacl_sp_PBE0_optimized.d12").exists()
    assert (tmp_path / "dupB" / "nacl_sp_PBE0_optimized.d12").exists()


def test_a_deck_is_never_overwritten_within_a_run(batch, tmp_path):
    """The last guard: a deck name already written in this run fails the file."""
    import CRYSTALOptToD12 as M
    reserved = {}
    (tmp_path / "sp").mkdir()
    assert M.process_files("nacl.out", "nacl.d12", config_file="t.json", output_dir="sp",
                           unattended=True, reserved_decks=reserved)[0]
    assert not M.process_files("nacl.out", "nacl.d12", config_file="t.json",
                               output_dir="sp", unattended=True, reserved_decks=reserved)[0]
    assert "already written from nacl.out in this run" in M.LAST_RESULT["reason"]


def _shrink(deck):
    lines = deck.splitlines()
    return lines[lines.index("SHRINK") + 1]


def test_a_template_mesh_is_not_used_but_each_parents_is(batch, tmp_path, capsys):
    """Templates saved by earlier versions carry their structure's mesh; applied
    to others it made every mesh be regenerated from the cell (8 16 -> 9 18)."""
    template = json.loads((tmp_path / "t.json").read_text())
    (tmp_path / "t.json").write_text(json.dumps({**template, "k_points": "3 6"}))
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    assert status == 0
    assert _shrink(_deck(tmp_path, "nacl").read_text()) == "8 16"
    assert _shrink(_deck(tmp_path, "agbr").read_text()) == "5 10"
    assert "each parent's own mesh (the config's k_points 3 6" in capsys.readouterr().out


@pytest.mark.parametrize("value,shrink", [("4 8", "4 8"), (6, "6 12"), ([4, 4, 4], "4 8")])
def test_k_points_for_all_files_sets_one_mesh(batch, tmp_path, value, shrink):
    """The explicit way to give every file one mesh, written as given."""
    template = json.loads((tmp_path / "t.json").read_text())
    (tmp_path / "t.json").write_text(json.dumps({**template, "k_points_for_all_files": value}))
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    assert status == 0
    for stem in ("nacl", "agbr"):
        assert _shrink(_deck(tmp_path, stem).read_text()) == shrink, stem


def test_a_bad_k_points_for_all_files_fails_the_files(batch, tmp_path, capsys):
    template = json.loads((tmp_path / "t.json").read_text())
    (tmp_path / "t.json").write_text(json.dumps({**template, "k_points_for_all_files": "8 x"}))
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    assert status == 1
    assert "0 written, 2 failed" in capsys.readouterr().err


def test_an_unfinished_optimisation_is_written_and_flagged(batch, tmp_path, capsys):
    """A killed OPT (no OPT END) still converts, from its starting geometry;
    the file line and the summary say so."""
    (tmp_path / "nacl.out").write_text(
        " COORDINATE AND CELL OPTIMIZATION - POINT    1\n")
    (tmp_path / "agbr.out").write_text(
        " COORDINATE AND CELL OPTIMIZATION - POINT    1\n"
        " * OPT END - CONVERGED * E(AU):  -1.0  POINTS    9 *\n")
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    out = capsys.readouterr()
    assert status == 0
    assert "2 written (1 from unfinished optimisations), 0 failed (of 2 files)" in out.err
    assert "nacl.out: wrote sp/nacl_sp_PBE0_optimized.d12 - WARNING: unfinished optimisation" in out.err
    assert "agbr.out: wrote sp/agbr_sp_PBE0_optimized.d12\n" in out.out
