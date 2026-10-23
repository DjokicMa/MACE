"""Tell a CRYSTAL deck's or output's calculation type from its keywords.

A plain substring test over the whole file is not enough: MACE titles decks
after the file name, so a single point named
``..._BULK_OPTGEOM_TZ_opt_..._sp_...`` carries "OPTGEOM" in its title, and
CRYSTAL echoes that title into the .out (and properties runs echo the SCF
title). A substring test then reads the SP as an OPT.

Input decks (.d12 / .d3)
    A keyword is a record: a whole line. Line 1 of a .d12 is the free-text
    title and never a keyword. A .d3 has no title of its own, but the line after
    a BAND record is the band-structure title, which is free text too.

Output files (.out)
    CRYSTAL never echoes the input records, so the type is read from lines
    CRYSTAL itself prints, each matched at the start of a line so the echoed
    title (its own line, e.g. " ./1_dia_opt_BULK_OPTGEOM_...") cannot match.

This module is standard library only: Crystal_d12/CrystalOutToCif.py also
loads it when run as a standalone script.
"""

import re
from pathlib import Path
from typing import Set, Union


def deck_records(content: str, is_d3: bool = False) -> Set[str]:
    """The keyword records of a CRYSTAL input deck, upper case, free text left out.

    Args:
        content: the deck text.
        is_d3: True for a properties (.d3) deck, which has no title line; the
            line following each BAND record (the band title) is skipped instead.
    """
    lines = content.splitlines()
    if not is_d3:
        lines = lines[1:]
    records = set()
    skip_title = False
    for line in lines:
        if skip_title:
            skip_title = False
            continue
        record = line.strip().upper()
        records.add(record)
        # the line after BAND is the band title, the one after ORBITALS the
        # root name of the Molden files: free text either way
        if is_d3 and record in ('BAND', 'ORBITALS'):
            skip_title = True
    return records


def deck_records_for(path: Union[str, Path], content: str) -> Set[str]:
    """``deck_records`` with the deck kind taken from the file suffix."""
    return deck_records(content, is_d3=Path(path).suffix.lower() == '.d3')


# Lines CRYSTAL prints for a geometry optimization. Every OPT output in the
# test corpus has all of the first four; the INFORMATION lines are printed while
# the input is read, so they are also there when the run dies in the first SCF,
# before any optimization step (and before OPTINFO.DAT exists). CRYSTAL23 prints
# the "OPTGEOM OPTIMIZES ..." line for FULLOPTG, ATOMONLY, CELLONLY, ITATOCEL
# and INTREDUN decks alike (checked on HPCC runs of each).
_OPT_OUTPUT_LINE = re.compile(
    r'^[ \t]*(?:'
    r'\*[ \t]*OPT END - '                                  # * OPT END - CONVERGED/FAILED *
    r'|\*[ \t]+OPTIMIZATION STARTS[ \t]+\*'                # banner
    r'|FINAL OPTIMIZED GEOMETRY'
    r'|[A-Z]+(?: [A-Z]+)* OPTIMIZATION - POINT[ \t]+\d'   # CELL OPTIMIZATION - POINT 1
    r'|INFORMATION \*+[^*\n]*\*+[ \t]+OPTGEOM OPTIMIZES'   # NEW DEFAULT line
    r'|INFORMATION \*+[ \t]*(?:OPTGEOM|FULLOPTG|ATOMONLY|CELLONLY|ITATOCEL|CVOLOPT)[ \t]*\*+'
    r')',
    re.MULTILINE,
)

# Lines CRYSTAL prints for a FREQCALC run: the "FREQUENCY CALCULATION" banner
# and the READM information line are printed while the input is read.
_FREQ_OUTPUT_LINE = re.compile(
    r'^[ \t]*(?:'
    r'FREQUENCY CALCULATION[ \t]*$'
    r'|INFORMATION \*+[ \t]*READM2?[ \t]*\*+[ \t]*FREQCALC'
    r'|VIBRATIONAL FREQUENCIES'
    r'|MODES[ \t]+EIGV'
    r')',
    re.MULTILINE,
)

# What the properties program prints for a BOLTZTRA run: the section banner,
# the data-file notices and the BOLTZTRA timing line. The phrases hold spaces,
# which a file-name title does not.
_TRANSPORT_OUTPUT_LINE = re.compile(
    r'THERMOELECTRIC AND ELECTRONIC TRANSPORT'
    r'|SEEBECK COEFFICIENT DATA WRITTEN'
    r'|^[ \t]*T{10,}[ \t]+BOLTZTRA\b',
    re.MULTILINE,
)
# Its timing-line alternative alone, matched at a line start.
_TRANSPORT_TIMING_LINE = re.compile(r'[ \t]*T{10,}[ \t]+BOLTZTRA\b')


def is_optimization_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a geometry optimization (OPTGEOM).

    An elastic, EOS or QHA run (``mechanics_output_type``) is not one: it
    prints the OPTGEOM lines too, and every strain or volume ends in its own
    OPT END."""
    return _OPT_OUTPUT_LINE.search(content) is not None and mechanics_output_type(content) is None


# The banner CRYSTAL prints while it reads the input of an ELASTCON, ELAPIEZO,
# PIEZOCON or PHOTOELA run ("ELASTIC CALCULATION", under a row of ELAELA...)
# and of an EOS or QHA run ("EQUATION OF STATE CALCULATION", under EOSEOS...).
# These runs also print the "OPTGEOM OPTIMIZES" NEW DEFAULT line, and each
# strained cell or volume is optimized to its own OPT END, FINAL OPTIMIZED
# GEOMETRY included (CRYSTAL23 on HPCC, tests/data/hpcc_116/mechanics). A QHA
# run computes the phonons of each volume, so it prints the FREQCALC lines too.
_ELASTIC_OUTPUT_LINE = re.compile(r'^[ \t]*ELASTIC CALCULATION[ \t]*$', re.MULTILINE)
_EOS_OUTPUT_LINE = re.compile(r'^[ \t]*EQUATION OF STATE CALCULATION[ \t]*$', re.MULTILINE)


def mechanics_output_type(content: str) -> Union[str, None]:
    """'ELASTIC', 'EOS' or 'QHA' for the .out of such a run, else None.

    The names are the calculation types of ``mace opt2d12 --calc-type``:
    ELASTCON, ELAPIEZO, PIEZOCON and PHOTOELA runs are all ELASTIC."""
    if 'ELASTIC CALCULATION' in content and _ELASTIC_OUTPUT_LINE.search(content):
        return 'ELASTIC'
    if 'EQUATION OF STATE CALCULATION' in content and _EOS_OUTPUT_LINE.search(content):
        return 'QHA' if is_frequency_output(content) else 'EOS'
    return None


# The structure of such a run is its reference (unstrained) cell, not any
# FINAL OPTIMIZED GEOMETRY: those are the strained cells or volumes. CRYSTAL
# prints the reference once, as the input geometry: the first "GEOMETRY FOR
# WAVE FUNCTION" block, right after the module banner. The central point is
# computed at it: "VOLUME OF THE CELL:" (ELASTCON, ELAPIEZO, PHOTOELA) and
# "VOLUME OF PRIMITIVE CELL (A^3)=" (EOS, QHA) repeat its primitive volume
# (tests/data/hpcc_116/mechanics and the full HPCC outputs they were cut from).
# With PREOPTGEOM (or ELASTCON PRESSURE, which adds it) the reference is the
# pre-optimized structure instead; no real output shows where CRYSTAL prints
# that one, so such a run gives no structure.
_GEOMETRY_BLOCK_LINE = re.compile(r'^[ \t]*GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY', re.MULTILINE)
_BLOCK_VOLUME = re.compile(r'PRIMITIVE CELL - CENTRING CODE[^\n]*?VOLUME=\s*(\d+\.\d+)')
_CENTRAL_VOLUME = re.compile(
    r'^[ \t]*(?:VOLUME OF THE CELL:|VOLUME OF PRIMITIVE CELL \(A\^3\)=)\s*(\d+\.\d+)', re.MULTILINE)
_FULLOPTG_LINE = re.compile(r'^[ \t]*INFORMATION \*+ FULLOPTG \*+', re.MULTILINE)


def mechanics_reference_problem(content: str) -> Union[str, None]:
    """Why the reference structure of an ELASTIC/EOS/QHA output cannot be read.

    None when it can (the input geometry block, see above) or when the output
    is not such a run. The message tells the user to take the structure from
    the optimization the run started from."""
    kind = mechanics_output_type(content)
    if kind is None:
        return None
    use = "use the output of the optimization it started from (the OPT .out) as the parent"
    block = _GEOMETRY_BLOCK_LINE.search(content)
    if block is None:
        return (f"this {kind} output holds strained cells and no input geometry block, "
                f"so its reference structure is unknown: {use}")
    if _FULLOPTG_LINE.search(content):         # the strains are CVOLOPT or atoms only
        return (f"this {kind} run pre-optimized its structure (PREOPTGEOM) and printed only the "
                f"strained cells after it: {use}")
    volume = _BLOCK_VOLUME.search(content, block.start())
    central = _CENTRAL_VOLUME.search(content, block.start())
    if volume and central:
        v, vc = float(volume.group(1)), float(central.group(1))
        if abs(v - vc) > 1e-5 * max(abs(v), 1.0):
            return (f"this {kind} run computed its central point at a volume of {vc:g} A^3, not at its "
                    f"input geometry's {v:g} A^3 (a pre-optimized structure): {use}")
    return None


# Where the strained cells start: the first strain (ELASTIC: "STRAIN MATRIX   1 :";
# all: "ELASTIC STRAIN MATRIX (ADIMENSIONAL)"), a volume's CVOLOPT, or anything
# of an optimization. Before it: the header, the reference cell and its SCF.
_STRAINED_CELL_LINE = re.compile(
    r'^[ \t]*(?:(?:ELASTIC )?STRAIN MATRIX\b|INFORMATION \*+ CVOLOPT\b|REFERENCE GEOMETRY DEFINED'
    r'|\* OPT END\b|FINAL OPTIMIZED GEOMETRY\b|[A-Z]+ OPTIMIZATION - POINT\b)', re.MULTILINE)
_MECHANICS_BANNER_LINE = re.compile(
    r'^[ \t]*(?:ELASTIC CALCULATION|EQUATION OF STATE CALCULATION)[ \t]*$', re.MULTILINE)


def mechanics_reference_text(content: str) -> str:
    """The part of an output that is about its structure.

    The whole output, except for an ELASTIC, EOS or QHA run: there, what
    comes before its first strained cell (the header, the reference cell and,
    in a full output, the central point's SCF), or only what comes before the
    module banner when the reference cell cannot be read
    (``mechanics_reference_problem``)."""
    if mechanics_output_type(content) is None:
        return content
    banner = _MECHANICS_BANNER_LINE.search(content)
    if mechanics_reference_problem(content):
        return content[:banner.start()]
    strained = _STRAINED_CELL_LINE.search(content, banner.start())
    return content[:strained.start()] if strained else content


# A standalone CPHF/CPKS run (opt2d12 --calc-type CPHF: an SCF, then the
# CPHF block): the module banner CRYSTAL prints after the SCF ("* *  COUPLED-
# PERTURBED KOHN-SHAM CALCULATION (CPKS)  * *", HARTREE-FOCK/CPHF for an HF
# reference) and the table of its results (CRYSTAL23 on HPCC,
# tests/data/hpcc_116/mechanics/dia_cphf*.out). A FREQCALC run with INTCPHF
# and a PHOTOELA run print the same banner and table: they are FREQ and
# ELASTIC, so is_cphf_output leaves them out.
_CPHF_OUTPUT_LINE = re.compile(
    r'^[ \t]*(?:\*[ \t]*\*[ \t]+COUPLED-PERTURBED [A-Z -]+CALCULATION \(CP(?:HF|KS)\)'
    r'|POLARIZABILITY \(ALPHA\), DIELECTRIC \(EPSILON\))',
    re.MULTILINE,
)


def is_cphf_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a standalone CPHF/CPKS run (the deck's
    own CPHF block), not from a FREQ (INTCPHF), elastic (PHOTOELA) or OPT run."""
    if 'COUPLED-PERTURBED' not in content and 'POLARIZABILITY (ALPHA)' not in content:
        return False
    return (_CPHF_OUTPUT_LINE.search(content) is not None and not is_frequency_output(content)
            and mechanics_output_type(content) is None and _OPT_OUTPUT_LINE.search(content) is None)


def is_frequency_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a FREQCALC run."""
    return _FREQ_OUTPUT_LINE.search(content) is not None


def is_transport_output(content: str) -> bool:
    """True when a properties .out is from a BOLTZTRA (transport) run."""
    # _TRANSPORT_OUTPUT_LINE, without a MULTILINE scan of the whole file: the
    # two phrases are plain substrings, and the timing-line alternative can
    # only match on a line holding ten T's (checked line by line from there)
    if 'THERMOELECTRIC AND ELECTRONIC TRANSPORT' in content or 'SEEBECK COEFFICIENT DATA WRITTEN' in content:
        return True
    if 'BOLTZTRA' not in content:
        return False
    return _line_with_match(content, 'TTTTTTTTTT', _TRANSPORT_TIMING_LINE)


def _line_with_match(content: str, needle: str, line_re) -> bool:
    """Whether ``line_re`` (anchored at a line start) matches a line of
    ``content`` that holds ``needle``: the lines without it are skipped."""
    pos = content.find(needle)
    while pos >= 0:
        start = content.rfind('\n', 0, pos) + 1
        if line_re.match(content, start):
            return True
        end = content.find('\n', pos)
        if end < 0:
            return False
        pos = content.find(needle, end)
    return False


# --- Calculation type of a deck ----------------------------------------------

# The properties types MACE runs from a .d3 deck.
PROPERTY_CALC_TYPES = ('BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL', 'ORBITALS')

# Charge-density and electrostatic-potential records. Crystal_d3 writes ECH3 and
# POT3 for the CHARGE+POTENTIAL step (3D cube grids) and ECHG / POTC for the
# 2D-map and point-potential variants.
CHARGE_POTENTIAL_RECORDS = frozenset({'ECH3', 'ECHG', 'POT3', 'POTC'})


def d3_calc_type(records: Set[str]) -> Union[str, None]:
    """The properties type a .d3 deck's records drive, or None if none is known."""
    if 'BOLTZTRA' in records:
        return 'TRANSPORT'
    if records & CHARGE_POTENTIAL_RECORDS:
        return 'CHARGE+POTENTIAL'
    # before DOSS: an ORBITALS deck starts with NEWK, as a DOSS deck does
    if 'ORBITALS' in records:
        return 'ORBITALS'
    if 'DOSS' in records:
        return 'DOSS'
    if 'BAND' in records:
        return 'BAND'
    return None


# The module block of a deck opt2d12 --calc-type ELASTIC, EOS, QHA or CPHF
# writes (Crystal_d12/d12_calc_elastic.py), and the type it is.
D12_MODULE_TYPES = (('ELASTCON', 'ELASTIC'), ('ELAPIEZO', 'ELASTIC'), ('PIEZOCON', 'ELASTIC'),
                    ('PIEZOCP', 'ELASTIC'), ('PHOTOELA', 'ELASTIC'), ('EOS', 'EOS'), ('QHA', 'QHA'),
                    ('CPHF', 'CPHF'))


def d12_calc_type(records: Set[str]) -> str:
    """OPT, FREQ, ELASTIC, EOS, QHA, CPHF or SP from a .d12 deck's records."""
    if 'OPTGEOM' in records:
        return 'OPT'
    if 'FREQCALC' in records:
        return 'FREQ'
    for keyword, calc_type in D12_MODULE_TYPES:
        if keyword in records:
            return calc_type
    return 'SP'


# --- Calculation type named in a file name ------------------------------------

# MACE chains the type into every follow-up name, so a single file name can hold
# several type tokens: "X_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_band".
# The last one names the file's own step. A token is "_<type>" with an optional
# step number, followed by "_" or the end of the name ("_optimized" is not
# "_opt").
_FILENAME_TYPE_TOKEN = re.compile(
    r'_(charge[_+]potential|chargepot|charge|potential|cp|transport|transp'
    r'|orbitals|orb|band|doss|dos|freq|sp|opt)\d*(?=_|$)')

_FILENAME_TOKEN_TYPE = {
    'charge_potential': 'CHARGE+POTENTIAL', 'charge+potential': 'CHARGE+POTENTIAL',
    'chargepot': 'CHARGE+POTENTIAL', 'charge': 'CHARGE+POTENTIAL',
    'potential': 'CHARGE+POTENTIAL', 'cp': 'CHARGE+POTENTIAL',
    'transport': 'TRANSPORT', 'transp': 'TRANSPORT',
    'orbitals': 'ORBITALS', 'orb': 'ORBITALS',
    'band': 'BAND', 'doss': 'DOSS', 'dos': 'DOSS',
    'freq': 'FREQ', 'sp': 'SP', 'opt': 'OPT',
}

_KNOWN_SUFFIXES = ('.d12', '.d3', '.out', '.f9', '.f25', '.sh', '.log')


def calc_type_from_filename(name: Union[str, Path]) -> Union[str, None]:
    """The calculation type the last type token of a file name names, or None."""
    stem = Path(name).name.lower()
    for suffix in _KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    last = None
    for match in _FILENAME_TYPE_TOKEN.finditer(stem):
        last = match.group(1)
    return _FILENAME_TOKEN_TYPE[last] if last else None


# What the properties program prints for each properties run. After every
# record it prints a timing line, "TTTT...TTTT <RECORD>  TELAPSE ...", and
# BAND and DOSS also print a section banner. A DOSS run prints
# "FROM BAND n TO BAND m" too (its projected band range), so that line is not a
# BAND tell. Over every .out in test/ and the HPCC trees each tell is found only
# in outputs of its own type.
_BAND_OUTPUT_LINE = re.compile(
    r'^[ \t]*\*[ \t]*BAND STRUCTURE[ \t]*\*'
    r'|^[ \t]*T{10,}[ \t]+BAND\b',
    re.MULTILINE,
)
_DOSS_OUTPUT_LINE = re.compile(
    r'^[ \t]*TOTAL AND PROJECTED DENSITY OF STATES'
    r'|^[ \t]*T{10,}[ \t]+DOSS\b',
    re.MULTILINE,
)
_CHARGE_POTENTIAL_OUTPUT_LINE = re.compile(
    r'^[ \t]*T{10,}[ \t]+(?:ECH3|ECHG|POT3|POTC)\b',
    re.MULTILINE,
)


# An ORBITALS run prints "PRINTING OF MOLECULAR ORBITALS" (0D) or "PRINTING OF
# CRYSTALLINE ORBITALS" (1D-3D), a "File opened : <name>.molden" line per file,
# "FINISHED ORBITAL PRINTING" and the ORBITALS timing line (CRYSTAL23 on HPCC:
# serial and parallel runs, 0D to 3D). None of these is in any other .out of
# test/ or the HPCC trees.
_ORBITALS_OUTPUT_LINE = re.compile(
    r'^[ \t]*PRINTING OF (?:MOLECULAR|CRYSTALLINE) ORBITALS\b'
    r'|^[ \t]*FINISHED ORBITAL PRINTING\b'
    r'|^[ \t]*T{10,}[ \t]+ORBITALS\b',
    re.MULTILINE,
)


def is_orbitals_output(content: str) -> bool:
    """True when a properties .out is from an ORBITALS (Molden export) run."""
    return _ORBITALS_OUTPUT_LINE.search(content) is not None


# --- ORBITALS: the Molden files a run writes ----------------------------------
#
# MACE writes "<deck stem>_orb" as the ORBITALS root, so the files of a deck
# X.d3 are X_orb.molden (a molecule) or X_orb_<nnnnn>_K<ijk>_real.molden and
# ..._complex.molden (one per k point), next to X.out once copied back.
ORBITALS_ROOT_SUFFIX = '_orb'


def orbitals_root(stem: str) -> str:
    """The ORBITALS root name MACE writes in the deck ``<stem>.d3``."""
    return f"{stem}{ORBITALS_ROOT_SUFFIX}"


# The part of a Molden file name after the root: nothing (a molecule) or
# _<n>_K<ijk>_real / _complex (one per k point). Matched exactly: a prefix
# glob <root>*.molden also caught another job whose name starts with the root
# (job "si", root "si_orb": si_orbitals_orb_00001_K000_real.molden).
_ORBITALS_MOLDEN_TAIL = re.compile(r'(?:_\d+_K\d+_(?:real|complex))?\.molden')


def is_orbitals_molden_name(name: str, stem: str) -> bool:
    """True when ``name`` is a Molden file of the ORBITALS run of ``<stem>.d3``."""
    root = orbitals_root(stem)
    return (name.startswith(root)
            and _ORBITALS_MOLDEN_TAIL.fullmatch(name[len(root):]) is not None)


def orbitals_molden_files(path: Union[str, Path]):
    """The Molden files of the ORBITALS run of ``path`` (its .out or .d3),
    sorted: ``<stem>_orb.molden`` or ``<stem>_orb_<n>_K<ijk>_(real|complex).molden``
    in the same folder."""
    import glob as _glob
    path = Path(path)
    pattern = _glob.escape(orbitals_root(path.stem)) + '*.molden'
    return sorted(p for p in path.parent.glob(pattern)
                  if p.is_file() and is_orbitals_molden_name(p.name, path.stem))


def orbitals_missing_reason(path: Union[str, Path]) -> Union[str, None]:
    """Why the ORBITALS run of ``path`` (its .out) is not complete, or None.

    CRYSTAL can end an ORBITALS run normally without writing a file: serial
    properties exits 0 on an input "FORMAT ERROR" (measured), so a finished
    .out is not enough; its Molden files have to be there.
    """
    path = Path(path)
    if orbitals_molden_files(path):
        return None
    return (f"ORBITALS wrote no Molden file: no {orbitals_root(path.stem)}*.molden next to "
            f"{path.name} (CRYSTAL can end an ORBITALS run normally after an input error, "
            f"e.g. FORMAT ERROR IN INPUT DECK; check the ORBITALS records in {path.stem}.d3)")


# --- ECH3/POT3: the cube files a run writes ------------------------------------
#
# submit_prop.sh copies DENS_CUBE.DAT, POT_CUBE.DAT and SPIN_CUBE.DAT back as
# <stem>_DENS.CUBE, <stem>_POT.CUBE and <stem>_SPIN.CUBE next to <stem>.out.
CUBE_SUFFIXES = ('_DENS.CUBE', '_POT.CUBE', '_SPIN.CUBE')
_CUBE_RAW_NAMES = {'_DENS.CUBE': 'DENS_CUBE.DAT', '_POT.CUBE': 'POT_CUBE.DAT',
                   '_SPIN.CUBE': 'SPIN_CUBE.DAT'}


def empty_cube_reason(path: Union[str, Path]) -> Union[str, None]:
    """Why the ECH3/POT3 run of ``path`` (its .out) is not complete, or None.

    CRYSTAL can end a properties run normally with an empty cube: from a
    two-component molecule's fort.9, ECH3 exited 0 and wrote a 0-byte
    DENS_CUBE.DAT (measured on HPCC). A finished .out is not enough when a
    cube copied back next to it is empty.
    """
    path = Path(path)
    stem = path.stem
    empty = []
    for suffix in CUBE_SUFFIXES:
        cube = path.parent / (stem + suffix)
        try:
            if cube.is_file() and cube.stat().st_size == 0:
                empty.append(suffix)
        except OSError:
            continue
    if not empty:
        return None
    names = ', '.join(stem + s for s in empty)
    raw = ', '.join(_CUBE_RAW_NAMES[s] for s in empty)
    return (f"empty cube file: {names} is 0 bytes (CRYSTAL ended normally but wrote an "
            f"empty {raw}; measured for ECH3 from a two-component molecule)")


def set_orbitals_root(d3_text: str, stem: str) -> str:
    """``d3_text`` with the ORBITALS root record set to ``orbitals_root(stem)``.

    For a deck renamed to ``<stem>.d3``: the job script copies back
    ``<stem>_orb*.molden``, so the root has to follow the deck's name. A deck
    without ORBITALS is returned unchanged.
    """
    lines = d3_text.split('\n')
    for i, line in enumerate(lines[:-1]):
        if line.strip().upper() == 'ORBITALS':
            lines[i + 1] = orbitals_root(stem)
    return '\n'.join(lines)


def is_band_output(content: str) -> bool:
    """True when a properties .out is from a BAND (band structure) run."""
    return _BAND_OUTPUT_LINE.search(content) is not None


def is_doss_output(content: str) -> bool:
    """True when a properties .out is from a DOSS (density of states) run."""
    return _DOSS_OUTPUT_LINE.search(content) is not None


def is_charge_potential_output(content: str) -> bool:
    """True when a properties .out is from an ECH3/POT3 (or ECHG/POTC) run."""
    return _CHARGE_POTENTIAL_OUTPUT_LINE.search(content) is not None


# --- Two-component (spin-orbit) runs -------------------------------------------
#
# A two-component SCF is not a calculation type of its own: CRYSTAL23 runs it
# as a single point (no optimization or frequency run in 2c, manual p. 166),
# so it is a flag beside the type.
#
# A deck asks for it with a TWOCOMPON block in its SCF input, closed by END
# (manual sec. 6.2, p. 170; the p. 172 examples close it with ENDTWO), and for
# spin-orbit coupling with SOC inside that block (p. 175).
#
# In the .out, a 2c-SCF prints its density per cycle as a particle-number
# density and three magnetization components, and, when the Fermi level is
# searched, the occupied spinors:
#     TOTAL X-COMP MAGNETIZATION    0.00000001
#     TOTAL Y-COMP MAGNETIZATION   -0.00000001
#     TOTAL Z-COMP MAGNETIZATION    0.00000000
#     - NUMBER OF FULLY OCCUPIED/TOTAL SPINORS -    12 /     70
# (fcc Au 2c-SCF with SOC, stock CRYSTAL23 1.0.1 on HPCC). A 1c run prints
# "TOTAL ATOMIC CHARGES" (and "TOTAL ATOMIC SPINS") there instead, and has
# spin-orbitals, not spinors. Only lines CRYSTAL prints are matched, each at
# the start of a line, never the echoed title. Whether a 2c run WITHOUT SOC
# prints the same lines was not checked (only SOC runs were read).
_TWO_COMPONENT_OUTPUT_LINE = re.compile(
    r'^[ \t]*(?:'
    r'TOTAL [XY]-COMP MAGNETIZATION[ \t]'
    r'|- NUMBER OF FULLY OCCUPIED/TOTAL SPINORS[ \t]'
    r')',
    re.MULTILINE,
)


def is_two_component_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a two-component (TWOCOMPON) SCF."""
    return _TWO_COMPONENT_OUTPUT_LINE.search(content) is not None


def two_component_block(content: str) -> Union[Set[str], None]:
    """The records inside a .d12 deck's TWOCOMPON block (upper case), or None
    when the deck has no such block. Line 1, the title, is never a record."""
    records = None
    for line in content.splitlines()[1:]:
        record = line.strip().upper()
        if records is None:
            if record == 'TWOCOMPON':
                records = set()
            continue
        if record in ('END', 'ENDTWO'):
            return records
        records.add(record)
    return records


def is_two_component_deck(content: str) -> bool:
    """True when a .d12 deck asks for a two-component SCF (TWOCOMPON)."""
    return two_component_block(content) is not None


# Printed while the input is read, before the SCF: "INFORMATION **** TWOCOMPON
# **** TWO-COMPONENT SCF: SYMMETRY DISABLED" (CRYSTAL23 on HPCC, HI and PbTe).
_TWO_COMPONENT_INFO_LINE = re.compile(r'^[ \t]*INFORMATION \*+[ \t]*TWOCOMPON[ \t]*\*+')


def is_two_component_run(path: Union[str, Path]) -> bool:
    """Whether the run of ``path`` (a .out, or a .d12) is a two-component SCF.

    A .out: the TWOCOMPON information line or the lines a 2c-SCF prints
    (is_two_component_output), read up to the end of its first SCF only;
    else the .d12 of the same name beside it, when it has a TWOCOMPON block.
    """
    path = Path(path)
    if path.suffix.lower() == '.d12':
        try:
            return is_two_component_deck(path.read_text(errors='replace'))
        except OSError:
            return False
    try:
        with open(path, 'r', errors='replace') as fh:
            for line in fh:
                if _TWO_COMPONENT_INFO_LINE.match(line) or _TWO_COMPONENT_OUTPUT_LINE.match(line):
                    return True
                if '== SCF ENDED' in line:
                    break
    except OSError:
        pass
    deck = path.with_suffix('.d12')
    try:
        return deck.is_file() and is_two_component_deck(deck.read_text(errors='replace'))
    except OSError:
        return False


def is_spin_orbit_deck(content: str) -> bool:
    """True when a .d12 deck asks for a 2c-SCF with spin-orbit coupling (SOC
    inside its TWOCOMPON block)."""
    return 'SOC' in (two_component_block(content) or ())


# The two-component restarts (manual p. 173), records of the TWOCOMPON block:
# GUESSPSO starts from the density matrix of a previous 2c-SCF, GUESSPNOSO
# from that of a previous 1c-SCF (unrestricted, without symmetry). Both read
# fort.20; CRYSTAL stops on GUESSPSO with nothing there (measured on HPCC).
# What CRYSTAL does on GUESSPNOSO with nothing there was not measured.
TWO_COMPONENT_GUESSES = ('GUESSPSO', 'GUESSPNOSO')


def missing_2c_matrix_note(record: str) -> str:
    """What happens to a GUESSPSO or GUESSPNOSO deck run with nothing in
    fort.20, worded for that record: only GUESSPSO was measured."""
    if record.upper() == 'GUESSPSO':
        return "CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured)"
    return (f"{record.upper()} would have no 1c parent matrix in fort.20 to start from "
            f"(what CRYSTAL then does was not measured)")


def two_component_guess(content: str) -> Union[str, None]:
    """'GUESSPSO' or 'GUESSPNOSO' when a .d12 deck's TWOCOMPON block asks for
    that restart, or None."""
    block = two_component_block(content) or ()
    for record in TWO_COMPONENT_GUESSES:
        if record in block:
            return record
    return None


def staged_2c_guess(deck_path: Union[str, Path]):
    """(restart record, path of the .f20 it needs) for a deck asking for a
    two-component restart, or None. The .f20 is ``<deck stem>.f20`` next to
    the deck: the file submitcrystal23.sh stages as fort.20 for job
    ``<deck stem>``."""
    deck_path = Path(deck_path)
    try:
        content = deck_path.read_text(errors='replace')
    except OSError:
        return None
    record = two_component_guess(content)
    if record is None:
        return None
    return record, deck_path.with_suffix('.f20')
