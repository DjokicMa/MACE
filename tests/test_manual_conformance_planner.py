"""Manual-conformance tests for the functional menus in
mace/workflow/planner.py (WorkflowPlanner).

Everything the planner stores as ``custom_functional`` / ``new_functional`` /
``dft_functional`` is handed to the deck writers verbatim (planner ->
engine.py temp_config['functional'] -> CRYSTALOptToD12.py -> d12_writer.py),
so a menu entry that is not a literal CRYSTAL input keyword produces a deck
CRYSTAL rejects.  The keyword spellings pinned here were read out of the
CRYSTAL23 manual:

  RHF / UHF / ROHF / DFT          single-particle Hamiltonians, 6870-6873
  M06L 10397, SCAN 10403, M06 10428, M062X 10431, M06HF 10433, MN15 10435
  HF3C 11736 (deck: BASISSET / MINIX / HF3C / END), prose 11677
  PBEH3C 11810 and HSE3C 11866 (both: BASISSET / def2-mSVP / DFT / ...)
  B973C 11916 (BASISSET / mTZVP), HFSOL3C 11960 (BASISSET / SOLMINIX)
  PBESOL03C 11996 and HSESOL3C 12041 (both: BASISSET / SOLDEF2MSVP)
  D3-parametrized functionals 11101-11104

The hyphenated forms the menus used to offer ("HF-3C", "M06-2X", ...) occur
only in the manual's prose, never as input keywords.

The 3C assertions are made against the 291 real HSESOL3C decks under test/,
which CRYSTAL actually ran, rather than against a hand-written fixture.
"""
import builtins
from pathlib import Path

import pytest

from mace.workflow import planner as planner_mod
from mace.workflow.planner import WorkflowPlanner

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DECK_DIR = REPO_ROOT / "test" / "BAND"


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    return WorkflowPlanner(work_dir=tmp_path_factory.mktemp("planner_manual"))


@pytest.fixture(scope="module")
def real_3c_method_block():
    """(basis, functional) lifted from a real CRYSTAL-accepted 3C deck.

    The production decks under test/BAND read

        BASISSET
        SOLDEF2MSVP
        DFT
        SPIN
        HSESOL3C
        XLGRID
        ENDDFT

    which is ground truth for both the keyword spelling and the basis set the
    method is parametrized for.
    """
    decks = sorted(REAL_DECK_DIR.glob("*HSESOL3C*.d12"))
    if not decks:
        pytest.skip("real HSESOL3C reference decks are not present under test/")
    lines = [line.strip() for line in decks[0].read_text().splitlines()]
    basisset_at = lines.index("BASISSET")
    basis = lines[basisset_at + 1]
    dft_at = lines.index("DFT", basisset_at)
    enddft_at = lines.index("ENDDFT", dft_at)
    functional = next(
        line for line in lines[dft_at + 1:enddft_at] if line.endswith("3C")
    )
    return basis, functional


def _feed_input(monkeypatch, answers):
    """Answer successive input() prompts from ``answers``."""
    remaining = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(remaining))


def _choose_custom_functional(planner, monkeypatch, category, index):
    """Drive _select_custom_functional to category/entry and return the offer."""
    _feed_input(monkeypatch, [category, str(index)])
    return planner._select_custom_functional()


# --- 3C composite methods -------------------------------------------------

def test_3c_menu_offers_the_keyword_crystal_accepted(
    planner, monkeypatch, real_3c_method_block
):
    """The 3C menu must offer HSESOL3C, the token in the real decks.

    Prevents: the menu used to offer the prose name "HSEsol-3C", which reached
    the deck as `DFT / HSEsol-3C / ENDDFT` and was rejected by CRYSTAL.
    """
    _, real_functional = real_3c_method_block
    assert real_functional == "HSESOL3C"          # guards the deck parse itself
    assert _choose_custom_functional(planner, monkeypatch, "6", 6) == real_functional


@pytest.mark.parametrize("index,expected", [
    (1, "HF3C"),          # manual 11736
    (2, "PBEH3C"),        # manual 11810
    (3, "HSE3C"),         # manual 11866
    (4, "B973C"),         # manual 11916
    (5, "PBESOL03C"),     # manual 11996
    (6, "HSESOL3C"),      # manual 12041
])
def test_every_3c_offer_is_an_input_keyword(planner, monkeypatch, index, expected):
    """Every 3C entry must be the unhyphenated CRYSTAL keyword.

    Prevents: the whole 3C category shipping prose names ("HF-3C", "PBEh-3C",
    "B97-3C", ...) that no CRYSTAL input parser accepts.
    """
    offered = _choose_custom_functional(planner, monkeypatch, "6", index)
    assert offered == expected
    assert "-" not in offered


def test_3c_selection_pins_its_parametrized_basis(
    planner, monkeypatch, real_3c_method_block
):
    """Selecting a 3C method must override an inherited basis with its own.

    A 3C method's D3/gCP/SRB terms are fitted to one specific basis (manual
    11685-11687 for HF-3c/MINIX).  Prevents the silent-wrong-number case: with
    the keyword fixed but the basis left inherited, CRYSTAL happily converges
    HSESOL3C on POB-TZVP-REV2 and reports plausible, meaningless energies.
    """
    real_basis, real_functional = real_3c_method_block

    monkeypatch.setattr(planner, "_get_basic_opt_config", lambda: {})
    monkeypatch.setattr(
        planner, "_get_method_modifications",
        lambda: {"custom_functional": real_functional},
    )
    monkeypatch.setattr(
        planner, "_get_basis_modifications", lambda: {"inherit_basis": True}
    )
    monkeypatch.setattr(planner, "_get_custom_tolerances", lambda: {})

    config = planner._get_advanced_opt_config()

    assert config["basis_settings"]["new_basis"] == real_basis    # SOLDEF2MSVP
    assert "inherit_basis" not in config["basis_settings"]


def test_non_3c_selection_leaves_the_inherited_basis_alone(planner, monkeypatch):
    """A normal functional must not have a basis forced on it.

    Prevents the 3C basis override from leaking onto every advanced OPT step
    and silently discarding the user's "keep current basis set" choice.
    """
    monkeypatch.setattr(planner, "_get_basic_opt_config", lambda: {})
    monkeypatch.setattr(
        planner, "_get_method_modifications", lambda: {"custom_functional": "B3LYP"}
    )
    monkeypatch.setattr(
        planner, "_get_basis_modifications", lambda: {"inherit_basis": True}
    )
    monkeypatch.setattr(planner, "_get_custom_tolerances", lambda: {})

    config = planner._get_advanced_opt_config()

    assert config["basis_settings"] == {"inherit_basis": True}


# --- Minnesota meta-GGA functionals ---------------------------------------

@pytest.mark.parametrize("index,expected", [
    (1, "M06"),      # manual 10428
    (2, "M062X"),    # manual 10431
    (3, "M06L"),     # manual 10397
    (4, "M06HF"),    # manual 10433
])
def test_minnesota_offers_are_input_keywords(planner, monkeypatch, index, expected):
    """The Minnesota entries must be the unhyphenated CRYSTAL keywords.

    Prevents: `DFT / M06-2X / ENDDFT` being written verbatim by
    d12_writer.write_dft_section and rejected by CRYSTAL.
    """
    assert _choose_custom_functional(planner, monkeypatch, "5", index) == expected


def test_quick_functional_menu_offers_m062x(planner, monkeypatch):
    """The short functional menu's entry 6 must store M062X, not M06-2X.

    Prevents the same rejected deck reaching CRYSTAL through the quick path,
    which writes method_settings['new_functional'] straight into the deck.
    """
    _feed_input(monkeypatch, ["6"])
    assert planner._get_method_modifications()["new_functional"] == "M062X"


def test_cif_advanced_metagga_offers_m06l(planner, monkeypatch):
    """The CIF advanced-customization meta-GGA path must store M06L.

    This is the second delivery path (planner -> cif_config['dft_functional']
    -> NewCifToD12.py, written verbatim); it carried the same "M06-L" bug.
    """
    # method=DFT, category=meta-GGA, functional=M06L, basis=default, opt=default
    _feed_input(monkeypatch, ["1", "4", "3", "1", "1"])
    monkeypatch.setattr(planner_mod, "yes_no_prompt", lambda *a, **k: False)

    cif_config = planner.get_advanced_customization("OPT")

    assert cif_config["dft_functional"] == "M06L"


# --- Hartree-Fock Hamiltonian ---------------------------------------------

def test_hf_category_offers_rhf_not_bare_hf(planner, monkeypatch):
    """The HF menu must offer RHF; a bare "HF" is not a Hamiltonian keyword.

    CRYSTALOptToD12 routes only RHF/UHF/HF3C/HFSOL3C down the HF branch, so
    "HF" fell through to the DFT branch and was emitted as
    `DFT / HF / ENDDFT` -- a rejected deck for a menu entry labelled
    "Hartree-Fock methods".
    """
    assert _choose_custom_functional(planner, monkeypatch, "1", 1) == "RHF"
    assert _choose_custom_functional(planner, monkeypatch, "1", 2) == "UHF"


# --- D3 dispersion gate ---------------------------------------------------

@pytest.mark.parametrize("category,index,functional", [
    ("5", 2, "M062X"),      # Minnesota: not parametrized
    ("6", 1, "HF3C"),       # 3C: correction already built in
    ("2", 1, "SVWN"),       # LDA: not parametrized
])
def test_no_d3_offer_for_unparametrized_functionals(
    planner, monkeypatch, category, index, functional
):
    """-D3 must not be offered for functionals CRYSTAL has not parametrized.

    Manual 11101-11104 lists the ten D3-parametrized methods.  Prevents the
    old unconditional suffix from producing "M062X-D3" / "SVWN-D3", which
    CRYSTAL rejects; the guard used to exclude only 3C methods.
    """
    _feed_input(monkeypatch, ["7", category, str(index)])
    monkeypatch.setattr(
        planner_mod, "yes_no_prompt",
        lambda *a, **k: pytest.fail(f"D3 was offered for {functional}"),
    )

    modifications = planner._get_method_modifications()

    assert modifications["custom_functional"] == functional


@pytest.mark.parametrize("category,index,functional", [
    ("3", 1, "PBE"),        # manual 11103
    ("4", 1, "B3LYP"),      # manual 11103, worked deck at 11091-11095
    ("5", 1, "M06"),        # manual 11104
])
def test_d3_still_offered_for_parametrized_functionals(
    planner, monkeypatch, category, index, functional
):
    """The D3 offer must survive for the ten functionals that support it.

    Prevents the new membership gate from being over-tight and silently
    dropping dispersion for PBE/B3LYP/M06 users.
    """
    _feed_input(monkeypatch, ["7", category, str(index)])
    monkeypatch.setattr(planner_mod, "yes_no_prompt", lambda *a, **k: True)

    modifications = planner._get_method_modifications()

    assert modifications["custom_functional"] == f"{functional}-D3"


def test_d3_gate_uses_the_shared_constant():
    """The gate must read d12_constants.D3_FUNCTIONALS, not a fourth copy.

    Prevents drift between the planner's dispersion offers and the list
    d12_writer.write_dft_section actually honours when appending "-D3".
    """
    assert set(planner_mod.D3_FUNCTIONALS) == {
        "BLYP", "PBE", "B97", "B3LYP", "PBE0",
        "mPW1PW91", "M06", "HSE06", "HSEsol", "LC-wPBE",
    }
