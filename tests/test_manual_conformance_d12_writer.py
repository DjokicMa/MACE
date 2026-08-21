"""CRYSTAL23 manual-conformance guards for the d12 DFT/SCF emitters.

Two deck-breaking defects are covered:

* ``write_dft_section`` mapped VBH -> VBHLYP, PWGGA -> PW91GGA and
  WCGGA -> WCGGAPBE. None of those three keywords exists anywhere in the
  CRYSTAL23 manual, so every deck written for those functionals carried a
  keyword CRYSTAL cannot resolve. The two real mappings the manual does
  document (PBESOL -> PBESOLXC, SOGGA -> SOGGAXC, manual lines 9994-10002)
  must survive.
* ``write_scf_section`` printed a bare ``BROYDEN``. The manual (lines
  7691-7700) defines BROYDEN as taking one record of three values,
  "W0 IMIX ISTART", so the emitted block was incomplete. ANDERSON, NODIIS
  and DIIS genuinely take no record (manual 7553-7554, 7738-7748) and must
  stay bare.

Where a real reference deck exists it is used as the oracle rather than a
synthetic fixture, so the unchanged parts of the block are pinned to output
that CRYSTAL has actually accepted.
"""
import io

import pytest

from d12_writer import write_dft_section, write_scf_section

from conftest import find_data


def _dft(functional, use_dispersion=False, dft_grid="DEFAULT", spin=False):
    buf = io.StringIO()
    write_dft_section(buf, functional, use_dispersion, dft_grid, spin)
    return buf.getvalue().splitlines()


def _scf(scf_method, **kwargs):
    buf = io.StringIO()
    params = dict(
        tolerances={"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, k_points=(8, 8, 8),
        dimensionality="CRYSTAL", use_smearing=False, smearing_width=0.005,
        scf_method=scf_method, scf_maxcycle=800, fmixing=30, num_atoms=2,
        spacegroup=1,
    )
    params.update(kwargs)
    write_scf_section(buf, **params)
    return buf.getvalue().splitlines()


def _real_diis_d12():
    return find_data("OPT/*dia*opt_rev1.d12", must_contain="HISTDIIS")


def _block(lines, start, end):
    """Inclusive slice of `lines` from the first `start` line to the next `end`."""
    i = lines.index(start)
    j = lines.index(end, i)
    return lines[i:j + 1]


# --- DFT functional keyword mapping ---------------------------------------

@pytest.mark.parametrize("functional", ["VBH", "PWGGA", "WCGGA"])
def test_gga_records_are_not_rewritten_to_invented_keywords(functional):
    """VBH/PWGGA/WCGGA were mapped to VBHLYP/PW91GGA/WCGGAPBE, none of which
    appear in the CRYSTAL23 manual, so the deck named a keyword CRYSTAL cannot
    resolve. They must now pass through verbatim."""
    lines = _dft(functional)
    assert lines == ["DFT", functional, "ENDDFT"]
    for invented in ("VBHLYP", "PW91GGA", "WCGGAPBE"):
        assert invented not in lines


@pytest.mark.parametrize("functional,keyword", [
    ("PBESOL", "PBESOLXC"),
    ("SOGGA", "SOGGAXC"),
])
def test_manual_documented_xc_keywords_are_still_mapped(functional, keyword):
    """PBESOLXC and SOGGAXC are real standalone XC keywords (manual 10001-10002);
    trimming the keyword map must not drop them along with the invented ones."""
    assert _dft(functional, dft_grid="XLGRID") == ["DFT", keyword, "XLGRID", "ENDDFT"]


def test_dft_section_reproduces_real_reference_deck_block():
    """Real accepted deck writes 'DFT / SPIN / B3LYP-D3 / XLGRID / ENDDFT'. The
    keyword-map edit must leave standard functionals byte-identical."""
    real = _real_diis_d12().read_text().splitlines()
    expected = _block(real, "DFT", "ENDDFT")
    assert _dft("B3LYP", use_dispersion=True, dft_grid="XLGRID", spin=True) == expected


# --- SCF method records ----------------------------------------------------

def test_broyden_is_followed_by_its_w0_imix_istart_record():
    """A bare BROYDEN is an incomplete deck: the manual (7691-7700) gives it one
    record of three values. Defaults are the manual's own suggestion (7708-7712)."""
    lines = _scf("BROYDEN")
    i = lines.index("BROYDEN")
    assert lines[i + 1] == "0.0001 50 2"
    assert lines[i + 2] == "PPAN"
    assert "HISTDIIS" not in lines


def test_broyden_record_values_are_caller_overridable():
    """The three values are real SCF knobs (IMIX overrides FMIXING), so a caller
    must be able to set them rather than being pinned to the defaults."""
    lines = _scf("BROYDEN", broyden_w0=0.001, broyden_imix=80, broyden_istart=5)
    assert lines[lines.index("BROYDEN") + 1] == "0.001 80 5"


@pytest.mark.parametrize("scf_method", ["ANDERSON", "NODIIS"])
def test_methods_taking_no_record_stay_bare(scf_method):
    """ANDERSON and NODIIS require no input data; adding a numeric record for
    them would be exactly the mirror-image deck error."""
    lines = _scf(scf_method)
    assert lines[lines.index(scf_method) + 1] == "PPAN"


def test_diis_scf_tail_matches_real_reference_deck():
    """The DIIS path is untouched: re-emitting a real deck's own SCF settings
    must reproduce its SCFDIR..PPAN tail exactly, HISTDIIS/100 included."""
    from d12_parsers import CrystalInputParser

    src = _real_diis_d12()
    real = src.read_text().splitlines()
    d = CrystalInputParser(str(src)).parse()

    lines = _scf(
        d["scf_method"], tolerances=d["tolerances"], k_points=d["k_points"],
        dimensionality=d["dimensionality"], scf_maxcycle=d["scf_maxcycle"],
        fmixing=d["fmixing"], spacegroup=d["spacegroup"],
    )
    assert _block(lines, "SCFDIR", "PPAN") == _block(real, "SCFDIR", "PPAN")
