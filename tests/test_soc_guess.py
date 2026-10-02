"""2c restarts of a SOC deck from a parent's density matrix ("soc_guess").

GUESSPSO (from a previous 2c-SCF) and GUESSPNOSO (from an unrestricted 1c-SCF
without symmetry) are records of the TWOCOMPON block (manual p. 173); both
read fort.20. Measured on HPCC (Bi2): GUESSPSO with nothing staged aborts;
from a staged 2c matrix, FMIXING 50 diverged (charges 19.7/26.3) and FMIXING
80 converged in 5 cycles. opt2d12 writes the record only when the parent's
.f9 is there, the parent deck has the same symmetry frame and shells, and
stages that .f9 as <deck stem>.f20 (what submitcrystal23.sh copies to
fort.20).
"""
import json
import os
import sys

import pytest

import CRYSTALOptToD12 as M
import soc_ecp as S
from test_soc_decks import PBTE, SOC_PARENT, _basis, _NoTTY, _Parser, cif2d12  # noqa: F401

QUIET = dict(log=lambda m: None)
SOC_PARENT_NOSOC = SOC_PARENT.replace("TWOCOMPON\nSOC\nEND\n", "TWOCOMPON\nEND\n")


def _scf(deck):
    lines = deck.split("\n")
    return {k: lines[lines.index(k) + 1] for k in ("FMIXING", "MAXCYCLE") if k in lines}


# --- soc_deck -------------------------------------------------------------------

def test_guesspso_goes_in_the_twocompon_block_with_fmixing_80():
    notes = []
    deck = S.soc_deck(PBTE, log=notes.append, guess="guesspso")
    assert "TWOCOMPON\nSOC\nGUESSPSO\nEND\nSCFDIR\n" in deck
    assert _scf(deck)["FMIXING"] == "80"
    assert any("measured on Bi2 only" in n for n in notes)
    # the rest is the default SOC deck
    assert deck.replace("GUESSPSO\n", "").replace("FMIXING\n80\n", "FMIXING\n50\n") == \
        S.soc_deck(PBTE, **QUIET)


def test_soc_fmixing_wins_over_the_restart_default():
    deck = S.soc_deck(PBTE, guess="guesspso", fmixing=60, **QUIET)
    assert _scf(deck)["FMIXING"] == "60"


def test_guesspnoso_gets_fmixing_80_too():
    notes = []
    deck = S.soc_deck(PBTE, log=notes.append, guess="guesspnoso")
    assert "TWOCOMPON\nSOC\nGUESSPNOSO\nEND\nSCFDIR\n" in deck
    assert _scf(deck)["FMIXING"] == "80"
    assert any("GUESSPNOSO restart default, measured on Bi2 only" in n for n in notes)
    assert _scf(S.soc_deck(PBTE, guess="guesspnoso", fmixing=60, **QUIET))["FMIXING"] == "60"


@pytest.mark.parametrize("bad", ["guessp", "", 1])
def test_an_unknown_soc_guess_is_refused(bad):
    with pytest.raises(S.SocError, match='soc_guess must be "guesspso"'):
        S.soc_deck(PBTE, guess=bad, **QUIET)


def test_the_guessp_refusal_points_at_soc_guess():
    with pytest.raises(S.SocError) as exc:
        S.soc_deck(PBTE.replace("SCFDIR\n", "GUESSP\nSCFDIR\n"), **QUIET)
    msg = str(exc.value)
    assert '"soc_guess"' in msg and "GUESSPSO" in msg and "GUESSPNOSO" in msg and "p. 173" in msg


# --- the frame and shell comparison ----------------------------------------------

def test_a_deck_matches_its_own_parent_and_the_ecp_records_do_not_count():
    child = S.soc_deck(PBTE, guess="guesspso", **QUIET)
    assert S.restart_mismatch(SOC_PARENT, child) is None
    # INPSOC parent, STUTSC child: the same shells
    assert S.restart_mismatch(SOC_PARENT, S.soc_deck(PBTE, ecp_form="stutsc", **QUIET)) is None
    # a 1c INPUT parent and its SOC child
    assert S.restart_mismatch(PBTE, child) is None


@pytest.mark.parametrize("change,why", [
    (lambda d: d.replace("252 0.5 0.5 0.5\n", "252 0.25 0.25 0.25\n252 0.5 0.5 0.5\n")
     .replace("\n2\n282 0.0", "\n3\n282 0.0"), "atom list differs"),
    (lambda d: d.replace("\n225\n", "\n221\n"), "symmetry or the atom list"),
    (lambda d: S.soc_deck(PBTE, basis="pob-tzvp-rev2-soc", log=lambda m: None), "basis shells differ"),
], ids=["atoms", "group", "soc_basis"])
def test_a_different_frame_or_shells_is_named(change, why):
    child = change(S.soc_deck(PBTE, **QUIET))
    assert why in S.restart_mismatch(SOC_PARENT, child)


# --- through opt2d12 ---------------------------------------------------------------

@pytest.fixture
def opt2d12_restart(tmp_path, monkeypatch):
    """opt2d12 on parent socpbte.d12 ``text`` with socpbte.f9 ``f9`` (bytes or
    None) and a template; returns (status, deck path or None, sp dir)."""
    monkeypatch.setattr(M, "CrystalOutputParser",
                        lambda p: _Parser(str(p).replace("socpbte", "pbte"), "out"))
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.chdir(tmp_path)

    def run(text=SOC_PARENT, f9=b"2c density matrix", **template):
        (tmp_path / "socpbte.out").write_text("")
        (tmp_path / "socpbte.d12").write_text(text)
        if f9 is not None:
            (tmp_path / "socpbte.f9").write_bytes(f9)
        base = {"calculation_type": "SP", "basis_set": M.PARENT_BASIS_MARKER,
                "basis_set_type": "EXTERNAL", "use_original_external_basis": True,
                "has_original_external_basis": True}
        base.update(template)
        (tmp_path / "t.json").write_text(json.dumps(base))
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", "--out-file", "socpbte.out",
                                          "--config-file", "t.json", "--output-dir", "sp",
                                          "--yes"])
        try:
            M.main()
            status = 0
        except SystemExit as e:
            status = e.code
        sp = tmp_path / "sp"
        decks = sorted(sp.glob("*.d12")) if sp.exists() else []
        return status, (decks[0] if decks else None), sp
    return run


def test_a_guesspso_child_is_written_and_the_parent_f9_staged(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(soc_guess="guesspso")
    assert status == 0 and deck is not None
    text = deck.read_text()
    assert "TWOCOMPON\nSOC\nGUESSPSO\nEND\nSCFDIR\n" in text and "FMIXING\n80\n" in text
    f20 = deck.with_suffix(".f20")
    assert f20.read_bytes() == b"2c density matrix"
    assert S.staged_2c_guess(deck) == ("GUESSPSO", f20)
    out = capsys.readouterr().out
    assert "the parent is a two-component run with SOC" in out
    assert f"staged socpbte.f9 as {f20.name}" in out


def test_a_2c_parent_without_soc_is_a_guesspso_source(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(text=SOC_PARENT_NOSOC, soc=True, soc_guess="guesspso")
    assert status == 0 and deck is not None and deck.with_suffix(".f20").exists()
    assert "without SOC (a 2c reference run" in capsys.readouterr().out


@pytest.mark.parametrize("f9,why", [(None, "missing"), (b"", "empty")])
def test_no_parent_matrix_no_deck(opt2d12_restart, capsys, f9, why):
    status, deck, sp = opt2d12_restart(f9=f9, soc_guess="guesspso")
    err = capsys.readouterr().err
    assert status == 1 and deck is None
    assert not sp.exists() or not list(sp.glob("*.f20"))
    assert f"socpbte.f9 next to socpbte.out is {why}" in err
    assert "CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured)" in err


def test_a_1c_parent_is_not_a_guesspso_source(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(text=PBTE, soc=True, soc_guess="guesspso")
    assert status == 1 and deck is None
    assert "the parent is a one-component run" in capsys.readouterr().err


def test_a_shell_mismatch_writes_no_deck_and_stages_nothing(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(soc_guess="guesspso", soc_basis="pob-tzvp-rev2-soc")
    err = capsys.readouterr().err
    assert status == 1 and deck is None
    assert "GUESSPSO cannot restart from the parent: the basis shells differ" in err
    assert not sp.exists() or not list(sp.iterdir())


def test_a_frame_mismatch_writes_no_deck(opt2d12_restart, capsys):
    parent = SOC_PARENT.replace("\n2\n282 0.0 0.0 0.0\n",
                                "\n3\n282 0.0 0.0 0.0\n252 0.25 0.25 0.25\n")
    assert parent != SOC_PARENT
    status, deck, sp = opt2d12_restart(text=parent, soc_guess="guesspso")
    assert status == 1 and deck is None
    assert "symmetry or the atom list differs" in capsys.readouterr().err


def test_a_different_existing_f20_is_not_overwritten(opt2d12_restart, tmp_path, capsys):
    sp = tmp_path / "sp"
    sp.mkdir()
    (sp / "socpbte_sp_PBE0_optimized.f20").write_bytes(b"something else")
    status, deck, _ = opt2d12_restart(soc_guess="guesspso")
    assert status == 1 and deck is None
    assert (sp / "socpbte_sp_PBE0_optimized.f20").read_bytes() == b"something else"
    assert "already exists and is not socpbte.f9" in capsys.readouterr().err


def test_a_scalar_deck_with_soc_guess_is_refused(opt2d12_restart, capsys):
    status, deck, _ = opt2d12_restart(soc=False, soc_guess="guesspso")
    assert status == 1 and deck is None
    assert 'set "soc": true' in capsys.readouterr().err


def test_without_soc_guess_nothing_is_staged(opt2d12_restart):
    status, deck, sp = opt2d12_restart()
    assert status == 0 and "GUESSP" not in deck.read_text()
    assert not list(sp.glob("*.f20"))


def test_cif2d12_refuses_soc_guess(cif2d12, capsys):
    ok, out = cif2d12("external", soc=True, soc_guess="guesspso")
    assert ok is False and not out.exists()
    assert "a deck written from a CIF has no parent" in capsys.readouterr().err


# --- GUESSPNOSO from a 1c parent (manual p. 173) -------------------------------------

P1 = ("\n225\n6.462\n", "\n1\n6.462 6.462 6.462 90.0 90.0 90.0\n")
SPIN = ("DFT\nPBE0\n", "DFT\nSPIN\nPBE0\n")
P1_SPIN_PARENT = PBTE.replace(*P1).replace(*SPIN)
assert P1_SPIN_PARENT.count("\n1\n6.462") == 1 and "SPIN" in P1_SPIN_PARENT


def test_a_p1_spin_parent_starts_a_guesspnoso_deck(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(text=P1_SPIN_PARENT, f9=b"1c matrix", soc=True,
                                       soc_guess="guesspnoso")
    assert status == 0 and deck is not None
    text = deck.read_text()
    assert "TWOCOMPON\nSOC\nGUESSPNOSO\nEND\nSCFDIR\n" in text
    assert "FMIXING\n80\n" in text and "DFT\nSPIN\nPBE0\n" in text
    assert "SYMMREMO" not in text        # TWOCOMPON deactivates symmetry (p. 169)
    assert deck.with_suffix(".f20").read_bytes() == b"1c matrix"
    out = capsys.readouterr().out
    assert "unrestricted one-component run without symmetry" in out
    assert "GUESSPNOSO restart default" in out


def test_a_symmremo_parent_with_a_group_is_accepted_with_a_warning(opt2d12_restart, capsys):
    parent = PBTE.replace("252 0.5 0.5 0.5\nEND\n", "252 0.5 0.5 0.5\nSYMMREMO\nEND\n").replace(*SPIN)
    assert "SYMMREMO" in parent
    status, deck, sp = opt2d12_restart(text=parent, soc=True, soc_guess="guesspnoso")
    err = capsys.readouterr().err
    assert "order of the parent's matrix was measured only for a Bi2 bilayer" in err
    # the child has no SYMMREMO (opt2d12 does not write it, and TWOCOMPON
    # deactivates symmetry, p. 169); its frame is the parent's otherwise
    assert status == 0 and deck is not None
    assert "SYMMREMO" not in deck.read_text() and "GUESSPNOSO" in deck.read_text()


@pytest.mark.parametrize("text,why", [
    (PBTE.replace(*P1), "the parent is restricted (no UHF, p. 131, or SPIN, p. 140)"),
    (PBTE.replace(*SPIN), "the parent ran with symmetry"),
    (PBTE, "restricted (no UHF, p. 131, or SPIN, p. 140); the parent ran with symmetry"),
], ids=["restricted", "symmetric", "both"])
def test_a_parent_guesspnoso_cannot_start_from_is_refused(opt2d12_restart, capsys, text, why):
    status, deck, sp = opt2d12_restart(text=text, soc=True, soc_guess="guesspnoso")
    err = capsys.readouterr().err
    assert status == 1 and deck is None
    assert why in err and "SYMMREMO" in err and "2NDVARIAT (p. 174)" in err


def test_guesspnoso_needs_the_parent_matrix(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(text=P1_SPIN_PARENT, f9=None, soc=True,
                                       soc_guess="guesspnoso")
    assert status == 1 and deck is None
    err = capsys.readouterr().err
    assert "socpbte.f9 next to socpbte.out is missing" in err
    # only GUESSPSO was measured to stop CRYSTAL with nothing in fort.20
    assert "GUESSPSO" not in err and "not measured" in err


def test_guesspnoso_from_a_2c_parent_is_refused(opt2d12_restart, capsys):
    status, deck, sp = opt2d12_restart(soc_guess="guesspnoso")
    assert status == 1 and deck is None
    assert '"guesspso" restarts from it' in capsys.readouterr().err
