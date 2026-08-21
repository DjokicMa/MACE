"""CRYSTAL23 manual conformance for the CIF -> D12 writer (NewCifToD12).

Three deck-level contracts, each taken from the CRYSTAL23 manual and each
reproduced as a real defect before these tests existed:

1. Input block 1 (geometry) can only be closed by an END-prefixed record
   (manual L958 "optional keywords terminated by END/ENDGEOM or STOP",
   L3195-3197 "Processing of geometry input block stops when the first three
   characters of the string are 'END'") or, exclusively for internal basis
   sets, by BASISSET (L1581-1583 "the following keyword must replace the final
   keyword, END, of the structure input (input block 1): BASISSET").  The
   HF + EXTERNAL branch wrote neither: the deck ran straight from the last
   atom coordinate into the first basis shell header, so CRYSTAL parsed a
   shell record as a geometry keyword.  The DFT + EXTERNAL branch was always
   correct - the two arms simply disagreed.  Manual L4247-4286 shows the
   sanctioned shape: coordinates -> END -> shell records -> "99 0" -> END.

2. SPINLOCK requires an open-shell Hamiltonian to act on: manual L1873-1876
   "To obtain a spin polarized solution an open shell Hamiltonian must be
   defined (block3, UHF or DFT/SPIN) ... This can be performed by the keywords
   SPINLOCK", and L2646-2647 "UHF and SPINLOCK must be used to define a
   reasonable orbital occupancy".  ROHF is not an escape hatch - L9867 "ROHF
   solution is not supported by CRYSTAL any more".  Every DFT arm emits SPIN
   when spin-polarized, but on the HF path only UHF is open shell, so RHF /
   HF3C / HFSOL3C decks carried a SPINLOCK no Hamiltonian could honour.

3. A refused deck must not be reported as created.  This is a workflow
   contract rather than a manual one, and it mirrors
   tests/test_d12_abort_no_truncated_deck.py for the sibling converter.

The block-1 shape asserted below is read out of a real known-good deck under
test/ rather than hand-written here, because this project has been bitten by
fixtures that encoded the bug they were meant to catch.
"""
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")
import NewCifToD12  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES  # noqa: E402

DATA = Path(__file__).parent / "data"
DIA_CIF = DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif"
EXTERNAL_BASIS = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"

# A real deck the user ran on the HPCC: external (full.basis.triplezeta) basis,
# so it shows the block-1 boundary this writer has to reproduce.
REFERENCE_DECK = (
    TEST_DATA
    / "SP"
    / "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_"
      "B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized.d12"
)


def base_options(**overrides):
    opts = dict(
        dimensionality="CRYSTAL",
        calculation_type="SP",
        basis_set_type="INTERNAL",
        basis_set="POB-TZVP-REV2",
        method="DFT",
        dft_functional="PBE",
        is_spin_polarized=False,
        tolerances=DEFAULT_TOLERANCES,
        scf_method="DIIS",
        symmetry_handling="CIF",
    )
    opts.update(overrides)
    return opts


def write_deck(tmp_path, name, **overrides):
    """Generate one deck from the real diamond CIF and return its lines."""
    cif_data = NewCifToD12.parse_cif(str(DIA_CIF))
    out = tmp_path / name
    assert NewCifToD12.create_d12_file(cif_data, str(out), base_options(**overrides))
    return out.read_text().splitlines()


def block1_terminator_index(lines):
    """Index of the record that closes input block 1 of an external-basis deck.

    Mirrors how this project's own reader locates it
    (Crystal_d12/d12_parsers.py:1079-1092): scan back from "99 0" for a line
    that is exactly "END".  Returns None when block 1 is never closed.
    """
    end_of_basis = next(i for i, l in enumerate(lines) if l.strip() == "99 0")
    for j in range(end_of_basis - 1, -1, -1):
        if lines[j].strip() == "END":
            return j
    return None


# --------------------------------------------------------------------------
# 1. block-1 termination on the HF + EXTERNAL path
# --------------------------------------------------------------------------


def test_reference_deck_defines_the_block1_shape():
    """Pin the expected block-1 shape to a real deck, not to a fixture.

    Guards the other external-basis assertions here: if this project's notion
    of "coordinates -> END -> shells -> 99 0 -> END" ever drifts from the decks
    that actually ran, this fails first and the drift is visible.
    """
    if not REFERENCE_DECK.exists():
        pytest.skip("real CRYSTAL corpus under test/ not present")
    lines = REFERENCE_DECK.read_text().splitlines()

    end_idx = block1_terminator_index(lines)
    assert end_idx is not None
    natoms = int(lines[5].strip())
    # atom-count record at index 5, then exactly natoms coordinate records,
    # then the END that closes block 1
    assert end_idx == 6 + natoms, lines[: end_idx + 2]
    assert lines[end_idx + 1].split()[0].isdigit(), "END must be followed by a shell header"
    assert lines[block1_terminator_index(lines) :][:1] == ["END"]


@pytest.mark.parametrize("hf_method", ["RHF", "UHF"])
def test_hf_external_basis_closes_input_block_one(tmp_path, hf_method):
    """HF + EXTERNAL used to run coordinates straight into the basis shells.

    Real failure prevented: the generated deck had no END (and no BASISSET)
    anywhere between the last atom coordinate and the first shell header, so
    CRYSTAL never left input block 1 and tried to read "6 8" as a geometry
    keyword.  Manual L958 / L3195-3197 / L1581-1583.
    """
    lines = write_deck(
        tmp_path,
        f"hf_{hf_method}.d12",
        method="HF",
        hf_method=hf_method,
        basis_set_type="EXTERNAL",
        basis_set=str(EXTERNAL_BASIS) + "/",
    )

    end_idx = block1_terminator_index(lines)
    assert end_idx is not None, "input block 1 is never terminated:\n" + "\n".join(lines[:20])

    natoms = int(lines[5].strip())
    assert end_idx == 6 + natoms, lines[: end_idx + 2]
    assert lines[end_idx + 1].split()[0].isdigit(), "END must be followed by a shell header"


def test_hf_and_dft_external_decks_share_the_geometry_basis_boundary(tmp_path):
    """The two EXTERNAL arms must agree on everything except the DFT block.

    Real failure prevented: the DFT arm wrote the block-1 END and the HF arm
    did not, so switching method silently changed the deck's structural
    validity rather than just its Hamiltonian.
    """
    common = dict(
        basis_set_type="EXTERNAL",
        basis_set=str(EXTERNAL_BASIS) + "/",
    )
    hf = write_deck(tmp_path, "boundary_hf.d12", method="HF", hf_method="RHF", **common)
    dft = write_deck(tmp_path, "boundary_dft.d12", **common)

    # drop the title (filename-derived) and the DFT block the HF deck cannot have
    dft_body = dft[1:]
    start = dft_body.index("DFT")
    del dft_body[start : dft_body.index("ENDDFT") + 1]

    assert hf[1:] == dft_body


def test_hf_external_deck_round_trips_through_the_project_parser(tmp_path):
    """The repo's own reader must be able to recover the external basis.

    Real failure prevented: CrystalInputParser locates basis data by scanning
    back from "99 0" for an exact "END" (Crystal_d12/d12_parsers.py:1079-1092).
    On an un-terminated HF deck that search found nothing, so a perfectly
    populated basis block was parsed as empty and any downstream regeneration
    silently dropped it.
    """
    from d12_parsers import CrystalInputParser

    lines = write_deck(
        tmp_path,
        "roundtrip.d12",
        method="HF",
        hf_method="RHF",
        basis_set_type="EXTERNAL",
        basis_set=str(EXTERNAL_BASIS) + "/",
    )
    deck = tmp_path / "roundtrip.d12"

    parsed = CrystalInputParser(str(deck)).parse()
    assert parsed["basis_set_type"] == "EXTERNAL"
    assert parsed["external_basis_data"], "basis block parsed as empty"
    # everything between the block-1 END and "99 0" must come back
    end_idx = block1_terminator_index(lines)
    n_records = lines.index("99 0") - end_idx - 1
    assert len(parsed["external_basis_data"]) == n_records


# --------------------------------------------------------------------------
# 2. SPINLOCK needs an open-shell Hamiltonian
# --------------------------------------------------------------------------


def test_spinlock_only_written_with_an_open_shell_hamiltonian(tmp_path):
    """Closed-shell HF decks must not carry SPINLOCK; UHF and DFT must keep it.

    Real failure prevented: RHF / HF3C / HFSOL3C + a non-zero spinlock emitted
    "SPINLOCK / <n> <cycles>" with no UHF and no DFT/SPIN anywhere in the deck.
    Manual L1873-1876 requires an open shell Hamiltonian (UHF or DFT/SPIN) for
    a spin-polarized solution, and L9867 rules out ROHF as an alternative.
    """
    spin = dict(is_spin_polarized=True, spinlock=2)

    for hf_method, basis in [("RHF", "POB-TZVP-REV2"), ("HF3C", "MINIX"),
                             ("HFSOL3C", "SOLMINIX")]:
        lines = write_deck(
            tmp_path, f"spin_{hf_method}.d12",
            method="HF", hf_method=hf_method, basis_set=basis, **spin
        )
        assert "SPINLOCK" not in lines, f"{hf_method} deck locks spin without UHF/SPIN"
        assert "UHF" not in lines and "SPIN" not in lines

    # the two Hamiltonians that CAN carry it must be untouched
    uhf = write_deck(tmp_path, "spin_UHF.d12", method="HF", hf_method="UHF", **spin)
    assert uhf.index("UHF") < uhf.index("SPINLOCK")
    assert uhf[uhf.index("SPINLOCK") + 1].split()[0] == "2"

    dft = write_deck(tmp_path, "spin_DFT.d12", **spin)
    assert "SPIN" in dft and "SPINLOCK" in dft


def test_spinlock_zero_writes_nothing_and_warns_nothing(tmp_path, capsys):
    """The common case must stay a byte-for-byte no-op.

    Guards the open-shell gate against over-reach: with no spinlock configured
    an RHF deck must be identical to before and must not emit the warning.
    """
    lines = write_deck(
        tmp_path, "nospin.d12", method="HF", hf_method="RHF", is_spin_polarized=True
    )
    assert "SPINLOCK" not in lines
    assert "SPINLOCK" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# 3. a refused deck is not a created deck
# --------------------------------------------------------------------------


def test_create_d12_file_reports_success(tmp_path):
    """create_d12_file must return True once the deck is on disk.

    Real failure prevented: the function returned None on every path, so a
    caller had no way to distinguish "written" from "refused".
    """
    cif_data = NewCifToD12.parse_cif(str(DIA_CIF))
    out = tmp_path / "ok.d12"
    assert NewCifToD12.create_d12_file(cif_data, str(out), base_options()) is True
    assert out.exists()


def test_incompatible_basis_without_a_terminal_fails_cleanly(tmp_path, monkeypatch):
    """Batch runs must refuse, not raise EOFError at an unanswerable prompt.

    Real failure prevented: with no TTY the confirmation prompt hit EOF and the
    converter died with "EOF when reading a line" - a cryptic message for what
    is really an unsupported-element refusal.  Nothing is on disk at that point
    (the deck is opened further down), so the refusal just has to be reported.
    """
    monkeypatch.setattr(NewCifToD12, "check_basis_set_compatibility",
                        lambda *a, **k: (False, [6]))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    cif_data = NewCifToD12.parse_cif(str(DIA_CIF))
    out = tmp_path / "refused.d12"

    assert NewCifToD12.create_d12_file(cif_data, str(out), base_options()) is False
    assert not out.exists()


def test_declined_deck_is_not_reported_as_created(tmp_path, monkeypatch, capsys):
    """process_cifs must not print "Created" for a deck the user declined.

    Real failure prevented: answering "no" to the unsupported-element prompt
    aborted creation but process_cifs still printed "[OK] Created <path>" for a
    file that was never written, so a batch log recorded decks that do not
    exist.
    """
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (src / DIA_CIF.name).write_bytes(DIA_CIF.read_bytes())

    monkeypatch.setattr(NewCifToD12, "check_basis_set_compatibility",
                        lambda *a, **k: (False, [6]))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(NewCifToD12, "yes_no_prompt", lambda *a, **k: False)

    NewCifToD12.process_cifs(str(src), base_options(), str(dst))

    assert "Created" not in capsys.readouterr().out
    assert list(dst.iterdir()) == []
