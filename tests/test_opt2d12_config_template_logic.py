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
        monkeypatch.setattr(sys, "stdin", _Stdin(tty))
        asked = []
        replies = list(answers)

        def prompt(text, default="yes"):
            asked.append(text.strip())
            return replies.pop(0)
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


