"""Spin-orbit ECPs in CRYSTAL23's INPSOC form (Crystal_d12/soc_ecp.py).

The format is the one the CRYSTAL23 manual defines on pp. 84-87. Its worked
example enters the Stuttgart ECP28MWB potential for Eu with INPSOC and the
INTERNAL convention - the same numbers as sopseud/263.mol - so the converter
is pinned first against that example, record for record.
"""
import os
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import soc_ecp as S  # noqa: E402

EU_MOL = """ECP,Eu,28,5,3;
1; 2,1.000000,0.000000;
1; 2,23.471384,607.659331;
1; 2,16.772479,264.385476;
1; 2,13.981343,115.381375;
1; 2,23.962888,-49.400794;
1; 2,21.232458,-26.748273;
1; 2,16.772479,19.869243;
1; 2,13.981343,1.523881;
1; 2,23.962888,0.399191;
"""

# CRYSTAL23 manual p. 87, first example (comments stripped).
MANUAL_EU = [
    "INPSOC",
    "INTERNAL 1.0",
    "35. 0 1 1 1 1 1",
    "23.471384 607.659331 0.000000 0",
    "16.772479 264.385476 19.869243 0",
    "13.981343 115.381375 1.523881 0",
    "23.962888 -49.400794 0.399191 0",
    "21.232458 -26.748273 0.000000 0",
]

# sopseud/283.mol (Bi, 60 core electrons), whose SO terms share the scalar
# exponents: the p and d blocks list the j = l -/+ 1/2 components.
BI_MOL = open(os.path.join(S.SOPSEUD_DIR, "283.mol")).read()


def test_the_manuals_eu_example_is_reproduced_exactly():
    ecp = S.parse_molpro_ecp(EU_MOL)
    assert (ecp.symbol, ecp.ncore, ecp.lmax, ecp.lso) == ("Eu", 28, 5, 3)
    assert S.inpsoc_records(ecp, 63) == MANUAL_EU


def test_the_repo_eu_file_is_the_manuals_example():
    assert S.inpsoc_records(S.load_so_ecp(63), 63) == MANUAL_EU


def test_bi_records_put_each_so_term_on_its_scalar_row():
    recs = S.inpsoc_records(S.parse_molpro_ecp(BI_MOL), 83)
    assert recs[:3] == ["INPSOC", "INTERNAL 1.0", "23. 0 2 4 4 2 2"]
    rows = recs[3:]
    assert len(rows) == 14
    # s rows carry no SO term; the first p row carries the p SO term
    assert rows[0] == "13.043090 283.264227 0.000000 0"
    assert rows[2] == "10.467777 72.001499 -144.002998 0"
    assert rows[-1] == "6.227782 -12.955710 -5.182284 0"


def test_the_scale_factor_is_written_on_the_convention_record():
    recs = S.inpsoc_records(S.parse_molpro_ecp(EU_MOL), 63, soscale=2.0)
    assert recs[1] == "INTERNAL 2.0"


def test_molpro_power_two_is_crystal_power_zero():
    ecp = S.parse_molpro_ecp("ECP,X,2,1,0;\n1; 1,3.0,-1.5;\n1; 2,1.0,2.0;\n")
    assert ecp.local[0].power == -1 and ecp.semilocal[0][0].power == 0


def test_an_so_term_without_a_matching_scalar_row_gets_its_own():
    ecp = S.parse_molpro_ecp(
        "ECP,X,10,2,1;\n1; 2,5.0,-1.0;\n1; 2,4.0,10.0;\n1; 2,3.0,5.0;\n1; 2,2.5,0.7;\n")
    assert S.inpsoc_records(ecp, 20)[2:] == [
        "10. 1 1 2 0 0 0",
        "5.0 -1.0 0.000000 0",
        "4.0 10.0 0.000000 0",
        "3.0 5.0 0.000000 0",
        "2.5 0.000000 0.7 0",
    ]


@pytest.mark.parametrize("text,why", [
    ("ECP,X,2,6,0;" + "1; 2,1.0,1.0;" * 7, "semi-local terms only up to l = 4"),
    ("ECP,X,2,2,2;" + "1; 2,1.0,1.0;" * 5, "do not fit"),
])
def test_potentials_inpsoc_cannot_hold_are_refused(text, why):
    with pytest.raises(S.SocError, match=why):
        S.inpsoc_records(S.parse_molpro_ecp(text), 30)


@pytest.mark.parametrize("text", [
    "ECP,Bi,60,5;",                          # header short of lso
    "ECP,Bi,60,1,0;\n2; 2,1.0,1.0;\n",       # block announces two terms
    "ECP,Bi,60,1,0;\n1; 2,1.0,1.0;\n1; 2,1.0;\n",
    "basis,Bi;",
])
def test_malformed_molpro_input_is_refused(text):
    with pytest.raises(S.SocError):
        S.parse_molpro_ecp(text)


# --- every file in the repo ---------------------------------------------------

ALL = S.available_so_ecps()


def test_all_81_spin_orbit_ecps_are_found():
    assert len(ALL) == 81


@pytest.mark.parametrize("z", ALL)
def test_records_read_back_to_the_same_numbers(z):
    """parse -> write -> parse: every nonzero coefficient comes back exactly,
    on its own l, with its exponent and power; only the zero l = L terms are
    left out, as in the manual's example."""
    ecp = S.load_so_ecp(z)
    znuc, local, per_l = S.parse_inpsoc(S.inpsoc_records(ecp, z))
    assert znuc == z - ecp.ncore
    assert [(a, b, n) for a, b, _c, n in local] == \
        [t.values() for t in ecp.local if float(t.coefficient) != 0]
    for l in range(5):
        scalar = [t.values() for t in ecp.semilocal[l]] if l < ecp.lmax else []
        so = [t.values() for t in ecp.spin_orbit.get(l, [])]
        assert [(a, b, n) for a, b, _c, n in per_l[l]][:len(scalar)] == scalar
        assert sorted((a, c, n) for a, _b, c, n in per_l[l] if c != 0) == \
            sorted(x for x in so if x[1] != 0)


def _library_ecp(library, z):
    path = os.path.join(REPO_ROOT, "Crystal_d12", "basis_sets", library, str(200 + z))
    if not os.path.exists(path):
        return None
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    if lines[1].strip() != "INPUT":
        return None
    counts = [int(c) for c in lines[2].split()[1:]]
    return lines[1:3 + sum(counts)]


# Which elements' external-library ECP is, number for number, the scalar part
# of the spin-orbit ECP. Recorded from the files; a change to either library
# shows up here first.
FULL_BASIS_SAME = [37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55,
                   56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73,
                   74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
STUTTGART_SAME = [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 26, 27, 29, 30,
                  31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51,
                  53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 83, 84]


@pytest.mark.parametrize("library,expected", [
    ("full.basis.doublezeta", FULL_BASIS_SAME),
    ("full.basis.triplezeta", FULL_BASIS_SAME),
    ("stuttgart", STUTTGART_SAME),
])
def test_which_library_ecps_are_the_same_potential(library, expected):
    same = []
    for z in ALL:
        lines = _library_ecp(library, z)
        if lines and S.same_scalar_potential(S.load_so_ecp(z), z, lines):
            same.append(z)
    assert same == expected


def test_a_different_potential_is_refused_not_paired():
    """stuttgart/279 is a 19-valence-electron Au ECP of another form; its
    valence basis must not be given the sopseud Au spin-orbit terms."""
    with pytest.raises(S.SocError, match="not the scalar part"):
        S.soc_ecp_records(79, _library_ecp("stuttgart", 79))


def test_an_ecp_without_spin_orbit_terms_is_refused():
    assert S.load_so_ecp(70).lso == 0      # Yb: 60-electron core, no SO blocks
    with pytest.raises(S.SocError, match="no spin-orbit terms"):
        S.soc_ecp_records(70, ["INPUT", "10. 0 0 0 0 0 0"])


def test_no_file_no_spin_orbit_ecp():
    with pytest.raises(S.SocError, match="no spin-orbit ECP for Z=86"):
        S.load_so_ecp(86)


# stuttgart/<200+Z> files whose valence basis CRYSTAL cannot read as it stands
# (manual pp. 25-27: NSHL shells follow the ECP, each with a numeric formal
# charge CHE). 219, 220 and 227 hold a placeholder sentence where NSHL belongs
# and "X" for every CHE; 204 announces 2 shells and holds one; 294 announces
# 11 and holds 9. Which shells are missing, or how the electrons are shared
# among the shells, is not in those files, so they are refused, not repaired.
STUTTGART_UNREADABLE = {4: "it announces 2 shells but holds 1",
                        19: "no shell count in its first line",
                        20: "no shell count in its first line",
                        27: "no shell count in its first line",
                        94: "it announces 11 shells but holds 9"}


@pytest.mark.parametrize("z,why", sorted(STUTTGART_UNREADABLE.items()))
def test_a_stuttgart_file_crystal_cannot_read_is_refused(z, why):
    with pytest.raises(S.SocError, match=rf"stuttgart/{200 + z} \(Z={z}\) cannot be used: {why}"):
        S.read_stuttgart(z)


def test_placeholder_shell_charges_are_named():
    with pytest.raises(S.SocError, match="shell charge 'X' is not a number"):
        S.read_stuttgart(27)


def test_every_other_stuttgart_file_still_reads():
    read = []
    for name in sorted(os.listdir(S.STUTTGART_DIR)):
        if name.isdigit() and 200 < int(name) < 300 and int(name) - 200 not in STUTTGART_UNREADABLE:
            got = S.read_stuttgart(int(name) - 200)
            if got is not None:
                read.append(int(name) - 200)
                ecp, shells = got
                assert ecp[0].strip() == "INPUT" and shells
    assert len(read) > 50
