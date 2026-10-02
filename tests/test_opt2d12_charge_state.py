"""opt2d12 children keep a charged parent's electron count on every path.

E4a: when the parent deck failed to parse, process_files set has_parent_deck
False and skipped carrying CHEMOD/CHARGED/DOPING although the file was there;
the child silently ran neutral. Records are now carried from the raw text.
E4b: with no parent deck at all (only the .out), a charged parent's state was
lost. The .out's Mulliken count (never the header, which DOPING does not
change) minus the valence charges gives DOPING; with no Mulliken print a
warning says the state cannot be recovered.
E4c (a pin, it passed before): a template that sets other settings keeps the
records.
"""
import builtins
import json
import os
import subprocess
import sys


import menu_nav
from conftest import REPO_ROOT
from ion_fixtures import molecule_deck, mulliken_block, synth_out

LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]
ROWS = [("LI", 3), ("O", 8), ("H", 1), ("H", 1)]
CHARGED_PARENT = molecule_deck(LI_WATER, title="parent").replace(
    "SOLDEF2MSVP\n", "SOLDEF2MSVP\nCHEMOD\n1\n1 2.0 0.0 0.0 0.0 0.0\n").replace(
    "ENDDFT\n", "ENDDFT\nDOPING\n-1\n")


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(k, None)
    return env


def _run(tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "parent.out",
         "--calc-type", "SP", "--non-interactive", "--output-dir", "o", *extra],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=_env())


def _child(tmp_path):
    decks = list((tmp_path / "o").glob("*.d12"))
    assert len(decks) == 1
    return decks[0].read_text().splitlines()


# ---------------------------------------------------------------------- E4a

def test_records_are_carried_when_the_parent_deck_fails_to_parse(tmp_path, monkeypatch):
    import CRYSTALOptToD12 as M

    (tmp_path / "parent.d12").write_text(CHARGED_PARENT)
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent"))

    def boom(self):
        raise ValueError("unreadable record")

    monkeypatch.setattr(M.CrystalInputParser, "parse", boom)
    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "o").mkdir()
    ok, _ = M.process_files("parent.out", "parent.d12", non_interactive=True, calc_type="SP",
                            output_dir="o", unattended=True)
    assert ok
    child = _child(tmp_path)
    # the deck was unreadable, so the child's basis comes from the .out alone;
    # CHEMOD belongs to the parent's basis and stays out, the charge is kept
    assert child[-3:] == ["DOPING", "-1", "END"]
    if "SOLDEF2MSVP" not in child:
        assert "CHEMOD" not in child


# ---------------------------------------------------------------------- E4b

def test_a_parent_out_alone_gives_doping_from_its_mulliken_count(tmp_path):
    out = synth_out(LI_WATER, title="parent", mulliken=mulliken_block(ROWS, [2.0, 8.8, 0.6, 0.6], 12.0))
    (tmp_path / "parent.out").write_text(out)
    run = _run(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert _child(tmp_path)[-3:] == ["DOPING", "-1", "END"]


def test_a_neutral_parent_out_alone_adds_nothing(tmp_path):
    out = synth_out(LI_WATER, title="parent", mulliken=mulliken_block(ROWS, [2.9, 8.8, 0.65, 0.65], 13.0))
    (tmp_path / "parent.out").write_text(out)
    run = _run(tmp_path)
    assert run.returncode == 0
    assert "DOPING" not in _child(tmp_path)


def test_without_deck_or_mulliken_the_state_is_reported_unknown(tmp_path):
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent"))
    run = _run(tmp_path)
    assert run.returncode == 0
    assert "cannot be recovered" in run.stdout + run.stderr
    assert "DOPING" not in _child(tmp_path)


# ---------------------------------------------------------------------- E4c

def test_a_template_with_other_settings_keeps_the_records(tmp_path):
    (tmp_path / "parent.d12").write_text(CHARGED_PARENT)
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent"))
    (tmp_path / "t.json").write_text(json.dumps({
        "calculation_type": "SP", "method": "DFT", "functional": "HSESOL3C",
        "tolerances": {"TOLINTEG": "8 8 8 8 16", "TOLDEE": 10},
        "scf_settings": {"method": "DIIS", "maxcycle": 500, "fmixing": 50}}))
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "parent.out",
         "--config-file", "t.json", "--yes", "--output-dir", "o"],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=_env())
    assert run.returncode == 0, run.stdout + run.stderr
    child = _child(tmp_path)
    assert child[child.index("TOLDEE") + 1] == "10"
    assert child[child.index("FMIXING") + 1] == "50"
    assert child[child.index("SOLDEF2MSVP") + 1] == "CHEMOD"
    assert child[-3:] == ["DOPING", "-1", "END"]
