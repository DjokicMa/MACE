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
        if is_d3 and record == 'BAND':
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


def is_optimization_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a geometry optimization (OPTGEOM)."""
    return _OPT_OUTPUT_LINE.search(content) is not None


def is_frequency_output(content: str) -> bool:
    """True when a CRYSTAL .out is from a FREQCALC run."""
    return _FREQ_OUTPUT_LINE.search(content) is not None


def is_transport_output(content: str) -> bool:
    """True when a properties .out is from a BOLTZTRA (transport) run."""
    return _TRANSPORT_OUTPUT_LINE.search(content) is not None


# --- Calculation type of a deck ----------------------------------------------

# The four properties types MACE runs from a .d3 deck.
PROPERTY_CALC_TYPES = ('BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL')

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
    if 'DOSS' in records:
        return 'DOSS'
    if 'BAND' in records:
        return 'BAND'
    return None


def d12_calc_type(records: Set[str]) -> str:
    """OPT, FREQ or SP from a .d12 deck's records."""
    if 'OPTGEOM' in records:
        return 'OPT'
    if 'FREQCALC' in records:
        return 'FREQ'
    return 'SP'


# --- Calculation type named in a file name ------------------------------------

# MACE chains the type into every follow-up name, so a single file name can hold
# several type tokens: "X_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_band".
# The last one names the file's own step. A token is "_<type>" with an optional
# step number, followed by "_" or the end of the name ("_optimized" is not
# "_opt").
_FILENAME_TYPE_TOKEN = re.compile(
    r'_(charge[_+]potential|chargepot|charge|potential|cp|transport|transp'
    r'|band|doss|dos|freq|sp|opt)\d*(?=_|$)')

_FILENAME_TOKEN_TYPE = {
    'charge_potential': 'CHARGE+POTENTIAL', 'charge+potential': 'CHARGE+POTENTIAL',
    'chargepot': 'CHARGE+POTENTIAL', 'charge': 'CHARGE+POTENTIAL',
    'potential': 'CHARGE+POTENTIAL', 'cp': 'CHARGE+POTENTIAL',
    'transport': 'TRANSPORT', 'transp': 'TRANSPORT',
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


def is_band_output(content: str) -> bool:
    """True when a properties .out is from a BAND (band structure) run."""
    return _BAND_OUTPUT_LINE.search(content) is not None


def is_doss_output(content: str) -> bool:
    """True when a properties .out is from a DOSS (density of states) run."""
    return _DOSS_OUTPUT_LINE.search(content) is not None


def is_charge_potential_output(content: str) -> bool:
    """True when a properties .out is from an ECH3/POT3 (or ECHG/POTC) run."""
    return _CHARGE_POTENTIAL_OUTPUT_LINE.search(content) is not None
