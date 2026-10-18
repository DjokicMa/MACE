"""Follow-ups from the real CRYSTAL23 SOC runs on HPCC (CRYSTAL/23-intel-2023a).

1  ANDERSON in a 2c deck is accepted silently by CRYSTAL and makes the SCF
   diverge into "FERMI ENERGY NOT IN INTERVAL"; the same deck without it
   converges in 9 cycles. MACE refuses it, and the refusal says so.
2  OPTGEOM with TWOCOMPON stops at input, with the message only in fort.87:
   "TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM".
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import soc_ecp as S  # noqa: E402
from test_soc_decks import PB, TE, _deck, cif2d12, opt2d12  # noqa: E402,F401

PBTE = _deck(PB + TE)
QUIET = {"log": lambda m: None}
TEMPLATE = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


# --- 1: ANDERSON ---------------------------------------------------------------

def _refusal(deck, **kw):
    with pytest.raises(S.SocError) as err:
        S.soc_deck(deck, **QUIET, **kw)
    return str(err.value)


def test_the_anderson_refusal_cites_the_measured_divergence():
    why = _refusal(PBTE.replace("DIIS\n", "ANDERSON\n"))
    assert "ANDERSON" in why
    assert "FERMI ENERGY NOT IN INTERVAL" in why and "9 cycles" in why
    assert "measured" in why


def test_broyden_is_still_refused_without_the_anderson_measurement():
    why = _refusal(PBTE.replace("DIIS\n", "BROYDEN\n0.0001 50 2\n"))
    assert "BROYDEN" in why and "FERMI ENERGY" not in why


def test_cif2d12_prints_the_anderson_measurement(cif2d12, capsys):
    ok, out = cif2d12("external", soc=True, scf_method="ANDERSON")
    printed = capsys.readouterr()
    assert ok is not True and not out.exists()
    assert "FERMI ENERGY NOT IN INTERVAL" in printed.out + printed.err


def test_opt2d12_prints_the_anderson_measurement(opt2d12, capsys):
    status, decks = opt2d12(["pbte"], soc=True, scf_settings={"method": "ANDERSON"})
    printed = capsys.readouterr()
    assert status not in (0, None) and not any(p.exists() for p in decks.values())
    assert "FERMI ENERGY NOT IN INTERVAL" in printed.out + printed.err


# --- 2: OPTGEOM ----------------------------------------------------------------

def test_the_optgeom_refusal_cites_the_fort87_stop():
    why = _refusal(PBTE.replace("252 0.5 0.5 0.5\nEND\n", "252 0.5 0.5 0.5\nOPTGEOM\nENDOPT\nEND\n"))
    assert "TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM" in why and "fort.87" in why


def test_the_job_scripts_2c_restart_branch_cites_the_hpcc_result():
    text = TEMPLATE.read_text()
    block = text[text.index("# OPTGEOM RESTART"):text.index("# GUESSP restart")]
    assert "TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM" in block and "fort.87" in block


# --- 3: Hf, STUTSC and the official SOC basis -----------------------------------
#
# MACE's INPUT/INPSOC Hf potential is the published ECP60MDF (Figgen 2009).
# CRYSTAL's internal STUTSC for Hf gives its r^2 local term the exponent of
# the first term (10.2488560 where ECP60MDF has 10.7291270), +9.67e-5 Ha
# (measured). The official Hf_pob_TZVP_rev2_SOC record uses STUTSC, so
# soc_basis writes Hf with MACE's INPSOC and the official valence shells.

from test_soc_audit_fixes import O_AE, _two_atom_deck  # noqa: E402
from test_soc_decks import _library_block  # noqa: E402

HF = _library_block(272)
HFO = _two_atom_deck(HF, O_AE, labels=("272", "8"))
FAMILY = "pob-tzvp-rev2-soc"


def _hf_block(deck):
    lines = deck.split("\n")
    i = next(j for j, ln in enumerate(lines) if ln.split()[:1] == ["272"] and len(ln.split()) == 2)
    _s, _e, nxt = S._basis_block_span(lines, i)
    return lines[i:nxt]


def test_hf_stays_out_of_stutsc_cores():
    assert 72 not in S.STUTSC_CORES


def test_the_stutsc_refusal_for_hf_names_crystals_copied_exponent():
    why = _refusal(HFO, ecp_form="stutsc")
    assert "10.2488560" in why and "10.7291270" in why and "9.67e-5" in why


def test_soc_basis_writes_hf_with_inpsoc_and_the_official_shells():
    notes = []
    deck = S.soc_deck(HFO, log=notes.append, basis=FAMILY)
    block = _hf_block(deck)
    official = S.official_soc_block(72)
    assert official[1] == "STUTSC"
    assert "STUTSC" not in deck
    assert block[1:1 + len(S.inpsoc_records(S.load_so_ecp(72), 72))] == \
        S.inpsoc_records(S.load_so_ecp(72), 72)
    assert S._tokens(block[-(len(official) - 2):]) == S._tokens(official[2:])
    assert block[0].split() == official[0].split()
    said = " ".join(notes)
    assert "Hf" in said and "10.2488560" in said and "INPSOC" in said


def test_soc_basis_hf_deck_converts_again_unchanged():
    deck = S.soc_deck(HFO, basis=FAMILY, **QUIET)
    again = S.soc_deck(deck.replace("TWOCOMPON\nSOC\nEND\n", ""), **QUIET)
    assert _hf_block(again) == _hf_block(deck)


def test_an_official_hf_stutsc_block_in_a_derived_deck_becomes_inpsoc():
    official = S.official_soc_block(72)
    notes = []
    deck = S.soc_deck(_two_atom_deck(official, O_AE, labels=("272", "8")), log=notes.append)
    block = _hf_block(deck)
    assert block[1] == "INPSOC" and "STUTSC" not in deck
    assert S._tokens(block[-(len(official) - 2):]) == S._tokens(official[2:])
    assert any("Hf" in n and "10.2488560" in n for n in notes), notes


def test_other_soc_basis_elements_keep_stutsc():
    deck = S.soc_deck(PBTE, basis=FAMILY, **QUIET)
    assert deck.count("STUTSC") == 2


def test_an_official_hf_stutsc_block_with_soc_ecp_stutsc_is_refused():
    deck = _two_atom_deck(S.official_soc_block(72), O_AE, labels=("272", "8"))
    assert "10.2488560" in _refusal(deck, ecp_form="stutsc")


# --- 5: all-electron Ga-Se lose most of their spin-orbit splitting ---------------
#
# InAs on HPCC: Delta0 about 0.07 eV with the all-electron As of
# full.basis.triplezeta, about 0.35 eV with the stuttgart ECP10 As (233) and
# INPSOC (experiment 0.39 eV), while the total energy moves only ~0.6 mHa.

@pytest.mark.parametrize("label", ["31", "33", "34"])
def test_the_all_electron_ga_se_warning_gives_the_inas_numbers(label):
    notes = []
    S.soc_deck(_two_atom_deck(PB, _library_block(int(label)), labels=("282", label)),
               log=notes.append)
    warned = " ".join(n for n in notes if n.startswith("Warning"))
    for number in ("0.07 eV", "0.35 eV", "0.39 eV", "0.6 mHa", "80%"):
        assert number in warned, (number, warned)
    assert "ECP10" in warned and "band" in warned and "strongly" in warned


def test_br_gets_no_inas_numbers():
    notes = []
    S.soc_deck(_two_atom_deck(PB, _library_block(35), labels=("282", "35")),
               log=notes.append)
    warned = " ".join(n for n in notes if n.startswith("Warning"))
    assert "Br" in warned and "0.35 eV" not in warned


# --- 6: PRINTOUT EIGENVAL ----------------------------------------------------------
#
# In a 2c run it printed garbage symmetry labels, split into alpha/beta and
# segfaulted after the SCF (HPCC). cif2d12 and opt2d12 write every SOC deck
# through SocDeckBuffer, that is through soc_deck, so all three refuse it.

EIGENVAL = PBTE.replace("PPAN\n", "PRINTOUT\nEIGENVAL\n1\nEND\nPPAN\n")


def test_printout_eigenval_is_refused_with_what_crystal_did():
    why = _refusal(EIGENVAL)
    assert "EIGENVAL" in why and "segfault" in why and "symmetry labels" in why


def test_the_writers_buffer_refuses_printout_eigenval():
    class _File:
        def write(self, text):
            raise AssertionError("written")

        def discard(self):
            self.discarded = True

    f = _File()
    buf = S.SocDeckBuffer(f, log=lambda m: None)
    buf.write(EIGENVAL)
    assert "EIGENVAL" in buf.refusal()
    assert "EIGENVAL" in buf.finish() and f.discarded


# --- 7: open-shell f elements --------------------------------------------------
#
# EuO 2c from a cold start aborted on FERMI ENERGY NOT IN INTERVAL; with SMEAR
# it oscillated by about +-8 Ha until it ran out of cycles (HPCC).

EUO = _two_atom_deck(_library_block(263), O_AE, labels=("263", "8"))


def _warnings(deck, **kw):
    notes = []
    S.soc_deck(deck, log=notes.append, **kw)
    return [n for n in notes if n.startswith("Warning") and "f element" in n]


def test_an_f_element_soc_deck_is_warned_about_with_the_euo_runs():
    warned = _warnings(EUO)
    assert len(warned) == 1, warned
    text = warned[0]
    assert "Eu" in text and "FERMI ENERGY NOT IN INTERVAL" in text and "8 Ha" in text
    assert "SPIN" in text and "GUESSPNOSO" in text and "untested" in text


def test_the_f_element_warning_also_comes_with_smear():
    assert _warnings(EUO, smear=0.005)


def test_no_f_element_warning_without_one():
    assert not _warnings(PBTE)


# --- 8: GUESSPSO / GUESSPNOSO outside the TWOCOMPON block ---------------------------
#
# CRYSTAL stops: "ERROR **** READM2 **** KEYWORD GUESSPSO   NOT ALLOWED" (and
# GUESSPNOSO), measured on HPCC. A GUESSP at the top level of a 2c deck is
# read like GUESSPNOSO; GUESSPNOSO itself there is not.

import subprocess  # noqa: E402

from mace.submission import preflight as pf  # noqa: E402
from test_soc_guess_preflight import D12, _deck as _pf_deck, _submit  # noqa: E402
from test_submit_progress_cli_e2e import fake_slurm  # noqa: E402,F401
from test_guessp_restart import _stage_records, _staging_block  # noqa: E402

TOP_2C = {rec: D12.replace("SHRINK\n8 8\n", f"SHRINK\n8 8\n{rec}\nTWOCOMPON\nSOC\nEND\n")
          for rec in ("GUESSPSO", "GUESSPNOSO")}
TOP_1C = {rec: D12.replace("SHRINK\n8 8\n", f"SHRINK\n8 8\n{rec}\n")
          for rec in ("GUESSPSO", "GUESSPNOSO")}


@pytest.mark.parametrize("decks", [TOP_2C, TOP_1C], ids=["2c deck", "1c deck"])
@pytest.mark.parametrize("record", ["GUESSPSO", "GUESSPNOSO"])
def test_preflight_refuses_a_top_level_2c_guess_even_with_a_matrix(tmp_path, decks, record):
    deck = _pf_deck(tmp_path, decks[record], f20=b"m", f9=b"m")
    problem = pf.restart_file_problem(deck)
    assert problem and record in problem and "NOT ALLOWED" in problem
    assert "TWOCOMPON" in problem
    assert pf.misplaced_2c_guess_problem(deck) == problem
    result = pf.preflight_deck(deck, runner=lambda workdir: pytest.fail("ran"))
    assert result.status == pf.FAIL and "NOT ALLOWED" in result.reason


def test_a_2c_guess_inside_twocompon_is_not_misplaced(tmp_path):
    inside = D12.replace("SHRINK\n8 8\n", "SHRINK\n8 8\nTWOCOMPON\nSOC\nGUESSPSO\nEND\n")
    deck = _pf_deck(tmp_path, inside, f20=b"m")
    assert pf.misplaced_2c_guess_problem(deck) is None
    assert pf.restart_file_problem(deck) is None


def test_mace_submit_refuses_a_top_level_2c_guess(tmp_path, fake_slurm):
    deck = _pf_deck(tmp_path, TOP_2C["GUESSPNOSO"], f20=b"m")
    proc = _submit(tmp_path, fake_slurm, str(deck))
    out = proc.stdout + proc.stderr
    assert "Not submitting mat_sp.d12" in out and "NOT ALLOWED" in out
    assert "999001" not in out


@pytest.mark.parametrize("array", [False, True], ids=["loop", "array"])
def test_nosubmit_runs_the_misplaced_guess_check_too(tmp_path, fake_slurm, array):
    deck = _pf_deck(tmp_path, TOP_2C["GUESSPSO"], f20=b"m")
    proc = _submit(tmp_path, fake_slurm, "--nosubmit", *(["--array"] if array else []), str(deck))
    out = proc.stdout + proc.stderr
    assert "NOT ALLOWED" in out, out
    assert not (tmp_path / "mat_sp.sh").exists()
    assert proc.returncode != 0


def test_nosubmit_still_writes_a_script_for_a_restart_whose_matrix_comes_later(tmp_path, fake_slurm):
    inside = D12.replace("SHRINK\n8 8\n", "SHRINK\n8 8\nTWOCOMPON\nSOC\nGUESSPSO\nEND\n")
    deck = _pf_deck(tmp_path, inside)
    proc = _submit(tmp_path, fake_slurm, "--nosubmit", str(deck))
    assert (tmp_path / "mat_sp.sh").exists(), proc.stdout + proc.stderr


@pytest.mark.parametrize("record", ["GUESSPSO", "GUESSPNOSO"])
def test_soc_deck_refuses_a_top_level_2c_guess(record):
    deck = PBTE.replace("SCFDIR\n", f"{record}\nSCFDIR\n")
    why = _refusal(deck)
    assert record in why and "NOT ALLOWED" in why and "soc_guess" in why


@pytest.mark.parametrize("record", ["GUESSPSO", "GUESSPNOSO"])
def test_the_job_script_says_crystal_stops_on_a_top_level_2c_guess(tmp_path, record):
    block = _staging_block(tmp_path)
    _, out, _, _ = _stage_records(tmp_path, block, [record, "TWOCOMPON", "SOC", "END"],
                                  ["testmat.f20"], record.lower())
    assert "reads it like GUESSPNOSO" not in out
    assert f"KEYWORD {record}" in out and "NOT ALLOWED" in out


def test_a_top_level_guessp_is_still_read_like_guesspnoso(tmp_path):
    block = _staging_block(tmp_path)
    _, out, _, _ = _stage_records(tmp_path, block, ["GUESSP", "TWOCOMPON", "SOC", "END"],
                                  ["testmat.f20"], "guessp_top")
    assert "GUESSP is outside the TWOCOMPON block; CRYSTAL reads it like GUESSPNOSO" in out
