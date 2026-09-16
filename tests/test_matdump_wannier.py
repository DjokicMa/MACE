"""MATDUMP: the CRYSTAL matrix dump, and the Wannier90 hand-off built on it.

The scientific content of this feature is William Comaskey's: the LCAO->Wannier90
method and the ``lcao2wannier`` package. MACE generates the CRYSTAL deck and
orchestrates the run. These tests cover MACE's half - the deck, the refusals and
the driver's handling of his diagnostics - and nothing about the conversion
itself.

Everything asserted here was measured on stock ``CRYSTAL/23-intel-2023a`` at MSU
HPCC against the REAL corpus material
``test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized``, whose parent SCF reports
``MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS1247``:

* the derived ``N = 1247`` dump: 1247 overlap cells, 2494 Fock blocks,
  ``ENDPROP``, no errors, 34,275,372 B, 0.97 s of CPU.
* the same deck under ``mpirun -n 28 Pproperties``: the matrix region is
  byte-identical to serial, so the shipped ``submit_prop.sh`` needs no change.
* at ``N = 60`` (the hand-written value this feature started from)
  ``lcao2wannier`` aborts: ``cond(S) = inf`` at 80 of 100 k-points.
* ``lcao2wannier`` 1.0.0 parses only 999 of those 1247 cells, silently.

The refusal tests below use kilobyte-sized distilled output text rather than the
multi-MB artifacts, so they run in the corpus-less CI where ``test/`` is absent;
the tests that need the real corpus use ``find_data`` and skip cleanly.
"""
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))

from d3_matdump import (  # noqa: E402
    SPIN_CLOSED,
    SPIN_COLLINEAR,
    SPIN_SOC,
    MatdumpRefusal,
    capability_refusal,
    derive_n_rvectors,
    detect_spin_treatment,
    parse_deck_dimensionality,
    parse_max_gvector_index,
    parse_vector_pool_size,
    predict_dump_bytes,
    validate_n_rvectors,
    write_matdump_deck,
)

# A distilled CRYSTAL SCF output: only the lines MATDUMP reads. Real CRYSTAL
# spelling and spacing, including the I4 field that runs together with the word
# "INTEGRALS" - that detail is the point of several tests below.
SCF_HEADER = """\
 *******************************************************************************
 CRYSTAL - PROPERTIES - TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL
 TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL
 NO.OF VECTORS CREATED 6999 STARS 1021 RMAX   101.51010 BOHR
 NUMBER OF AO                36  EXCHANGE OVERLAP TOL        (T3) 10**   -8
"""


def scf_out(maxg, pool=6999, spin_line=None, n_ao=36):
    """Distilled SCF output carrying a given R-vector count."""
    lines = [
        " *******************************************************************************",
        spin_line or " TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL",
        f" NO.OF VECTORS CREATED {pool} STARS 1021 RMAX   101.51010 BOHR",
        f" NUMBER OF AO                {n_ao}  EXCHANGE OVERLAP TOL        (T3) 10**   -8",
    ]
    if maxg is not None:
        # CRYSTAL writes this in a fixed I4 field. At >= 1000 there is NO space
        # after "INTEGRALS"; below it there is. Reproduce both faithfully.
        field = f"{maxg:>4}" if isinstance(maxg, int) else f"{maxg}"
        lines.append(f" MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS{field}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The deck itself - acceptance criterion 1
# --------------------------------------------------------------------------


def test_deck_is_byte_exact_to_the_verified_reference_form():
    """Manual section 14: BASISSET, NPR=2, then prtrec 60 (overlap) and 64 (Fock).

    This exact form was run on CRYSTAL23. A SETPRINT-based variant fails with
    ``ERROR **** BASE **** FORMAT ERROR IN INPUT DECK``, so the record must not
    drift.
    """
    assert write_matdump_deck(1247) == "BASISSET\n2\n60 1247\n64 1247\nEND"


def test_deck_refuses_a_nonsense_count():
    with pytest.raises(MatdumpRefusal):
        write_matdump_deck(0)


# --------------------------------------------------------------------------
# Parsing N - the I4 field is the load-bearing detail
# --------------------------------------------------------------------------


def test_maxg_is_parsed_when_the_i4_field_touches_the_word_integrals():
    """The guard: a whitespace split silently fails on the >= 1000 form.

    Reintroducing the bug - parsing with ``line.split()[-1]`` or requiring
    ``\\s+`` before the digits - makes this assertion fail, because CRYSTAL
    writes ' INTEGRALS1247' with no separator once the value needs all four
    columns.
    """
    assert parse_max_gvector_index(scf_out(1247)) == 1247
    assert parse_max_gvector_index(scf_out(103)) == 103
    assert parse_max_gvector_index(scf_out(1111)) == 1111


def test_maxg_uses_the_real_corpus_spelling():
    """Against the real diamond SP output, not a fixture."""
    out = find_data("SP/1_dia*sp*.out", must_contain="MAX G-VECTOR INDEX")
    assert parse_max_gvector_index(out.read_text(errors="ignore")) == 1247


def test_maxg_is_constant_within_every_corpus_output():
    """The value is emitted at lattice setup, not per optimization step.

    If it varied within a file, "take the last occurrence" would be a real
    choice rather than a formality, and the derivation would need a rule for
    which one. It does not vary.
    """
    import re

    pattern = re.compile(
        rb"MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS\s*(\S+)\s*$",
        re.MULTILINE)
    corpus = find_data("SP/*.out").parent.parent
    checked = 0
    for path in sorted(corpus.rglob("*.out"))[:200]:
        # Binary read: 50 corpus outputs contain NUL bytes, and a text-mode
        # grep silently drops them.
        values = set(pattern.findall(path.read_bytes()))
        assert len(values) <= 1, f"{path} carries {values}"
        checked += 1
    assert checked > 0


def test_vector_pool_size_is_parsed():
    assert parse_vector_pool_size(scf_out(1247)) == 6999


# --------------------------------------------------------------------------
# The five refusals. Each must refuse, and say why.
# --------------------------------------------------------------------------


def test_refuses_when_the_count_line_is_absent():
    with pytest.raises(MatdumpRefusal) as exc:
        derive_n_rvectors(scf_out(None))
    assert "--n-rvectors" in str(exc.value)


def test_refuses_on_i4_overflow():
    """CRYSTAL writes **** at >= 10000; the true value is unrecoverable."""
    with pytest.raises(MatdumpRefusal) as exc:
        derive_n_rvectors(scf_out("****"))
    assert "overflow" in str(exc.value).lower()


def test_refuses_a_molecule_parent():
    """N == 1 means only R = (0,0,0) exists, i.e. a 0-D parent.

    CRYSTAL does NOT error on a MOLECULE matrix dump - measured, it exits 0 with
    ENDPROP and emits all-zero blocks at uninitialised lattice indices - so this
    refusal is the only guard there is.
    """
    with pytest.raises(MatdumpRefusal) as exc:
        derive_n_rvectors(scf_out(1))
    assert "MOLECULE" in str(exc.value)


def test_refuses_an_even_count():
    """The R set is closed under negation, so the count is always odd.

    Verified across every value in the corpus. An even value means the line was
    misparsed, not that CRYSTAL changed.
    """
    with pytest.raises(MatdumpRefusal) as exc:
        derive_n_rvectors(scf_out(1246))
    assert "odd" in str(exc.value)


def test_refuses_a_count_above_the_vector_pool():
    with pytest.raises(MatdumpRefusal) as exc:
        derive_n_rvectors(scf_out(7005, pool=6999))
    assert "pool" in str(exc.value)


def test_explicit_n_is_bounded_by_the_vector_pool():
    """Over-large N is corrupting, not merely wasteful.

    MEASURED on bcc Fe at N=7005: properties exits 0 with ENDPROP and no error,
    and prints headers past the pool with indices out of uninitialised memory -
    N.7000(  0  0  0), N.7003(***  0  0), N.7005(  0  0***). Several of those
    parse as R=(0,0,0) and overwrite the genuine on-site overlap block in an
    R-keyed dict downstream, making S(k) singular at every k.

    Reintroducing the bug - accepting any user N because "too large is only
    expensive" - makes this assertion fail.
    """
    assert validate_n_rvectors(6999, scf_out(1247), explicit=True) == 6999
    with pytest.raises(MatdumpRefusal) as exc:
        validate_n_rvectors(7005, scf_out(1247), explicit=True)
    assert "7005" in str(exc.value)


def test_explicit_n_is_refused_when_the_pool_cannot_be_read():
    text = scf_out(1247).replace("NO.OF VECTORS CREATED", "NO.OF THINGS MADE")
    with pytest.raises(MatdumpRefusal):
        validate_n_rvectors(500, text, explicit=True)


def test_an_explicit_n_below_the_derived_value_is_allowed():
    """The override exists for a user who knows the true support is tighter."""
    assert validate_n_rvectors(321, scf_out(1247), explicit=True) == 321


# --------------------------------------------------------------------------
# Spin treatment and the capability gate
# --------------------------------------------------------------------------


def test_spin_treatment_from_the_real_corpus_output():
    """The corpus diamond SP deck carries SPIN and is UNRESTRICTED OPEN SHELL.

    This matters: it is the material the stock CRYSTAL23 module dumped with
    ALPHA and BETA blocks, so it is the proof that collinear spin needs no
    development binary.
    """
    out = find_data("SP/1_dia*sp*.out", must_contain="TYPE OF CALCULATION")
    assert detect_spin_treatment(out.read_text(errors="ignore")) == SPIN_COLLINEAR


def test_spin_treatment_closed_shell():
    assert detect_spin_treatment(SCF_HEADER) == SPIN_CLOSED


def test_soc_is_detected_from_the_parent_deck():
    """SOC has zero examples in the corpus, so it is read from the deck."""
    assert detect_spin_treatment(SCF_HEADER, "CRYSTAL\n0 0 0\n227\nSOC\nEND\n") == SPIN_SOC


def test_capability_gate_does_not_refuse_collinear_spin_on_a_stock_build(tmp_path):
    """The correction that matters most.

    Stock CRYSTAL23 already emits collinear spin-resolved matrices - measured:
    the corpus diamond, a SPIN/UNRESTRICTED deck, dumped 1247 overlap and 2494
    Fock blocks under ALPHA/BETA headers on the stock HPCC module, and
    lcao2wannier parses that form natively.

    Reintroducing the bug - gating every spin-polarized dump on the development
    binary - makes this assertion fail, and would refuse a calculation that
    demonstrably works.
    """
    scalar_only = tmp_path / "properties"
    scalar_only.write_bytes(b"FOCK MATRIX - CELL\x00OVERLAP MATRIX - CELL\x00")
    assert capability_refusal(SPIN_COLLINEAR, scalar_only) is None
    assert capability_refusal(SPIN_CLOSED, scalar_only) is None


def test_capability_gate_refuses_soc_on_a_scalar_only_build(tmp_path):
    scalar_only = tmp_path / "properties"
    scalar_only.write_bytes(b"FOCK MATRIX - CELL\x00OVERLAP MATRIX - CELL\x00")
    message = capability_refusal(SPIN_SOC, scalar_only)
    assert message is not None
    # The message must say all four things the spec requires of it.
    assert "ALPHA_ALPHA ELECTRONS" in message          # what is missing
    assert "SOC only" in message                        # only needed for SOC
    assert "stock CRYSTAL23" in message                 # scalar systems are fine
    assert "CRYSTAL23 developers" in message            # where to get the build
    assert "never bundles" in message


def test_capability_gate_allows_soc_on_a_capable_build(tmp_path):
    dev = tmp_path / "properties"
    dev.write_bytes(b"FOCK MATRIX (REAL PART)\x00   ALPHA_ALPHA ELECTRONS\x00")
    assert capability_refusal(SPIN_SOC, dev) is None


def test_capability_gate_does_not_refuse_when_no_binary_can_be_checked(tmp_path):
    """Unknown must never be conflated with incapable.

    `mace opt2d3` routinely runs where no CRYSTAL module is loaded.
    """
    assert capability_refusal(SPIN_SOC, None) is None
    assert capability_refusal(SPIN_SOC, tmp_path / "does-not-exist") is None


# --------------------------------------------------------------------------
# Dimensionality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keyword,expected", [
    ("MOLECULE", 0), ("POLYMER", 1), ("SLAB", 2), ("CRYSTAL", 3),
])
def test_dimensionality_from_the_deck_keyword(keyword, expected):
    assert parse_deck_dimensionality(f"title\n{keyword}\n0 0 0\n") == expected


def test_a_real_corpus_molecule_deck_reads_as_zero_dimensional():
    deck = find_data("SP/*MOLECULE*.d12", must_contain="MOLECULE")
    assert parse_deck_dimensionality(deck.read_text(errors="ignore")) == 0


# --------------------------------------------------------------------------
# Size prediction
# --------------------------------------------------------------------------


def test_size_prediction_matches_the_measured_dumps():
    """Calibrated on the corpus diamond at two values of N on real hardware.

    N=60   -> 1,660,173 B measured
    N=1247 -> 34,275,372 B measured
    Both with n_ao=36 and collinear spin (3 blocks per cell). Require the
    prediction within 5%, which is enough to size a warning and honest about
    being an estimate.
    """
    for n, measured in ((60, 1_660_173), (1247, 34_275_372)):
        predicted = predict_dump_bytes(n, 36, SPIN_COLLINEAR)
        assert abs(predicted - measured) / measured < 0.05, (n, predicted, measured)


def test_size_prediction_is_none_without_an_ao_count():
    assert predict_dump_bytes(100, None, SPIN_CLOSED) is None
