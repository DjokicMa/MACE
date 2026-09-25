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
