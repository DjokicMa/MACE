#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whether CRYSTAL can read a basis-set file as it stands
------------------------------------------------------
A basis file in MACE's external libraries (Crystal_d12/basis_sets/<family>/
<NAT>) holds one atom's basis input as CRYSTAL reads it after the geometry
(manual sec. 2.2 and pp. 25-27):

    NAT NSHL                      conventional atomic number, number of shells
    [ECP]                         only for 200 < NAT < 1000: INPUT + its
                                  records, INPSOC + its records, or a one-line
                                  library keyword (HAYWSC, STUTSC, ...)
    ITYB LAT NG CHE SCAL          NSHL shell records; for ITYB 0 each is
    exponent coefficient [coeff]  followed by NG primitive lines

A file that breaks this - a placeholder where NSHL belongs, "X" for a shell's
formal charge CHE, fewer shells than NSHL - writes a deck CRYSTAL stops on
(or, worse, reads into the next atom). Which shells are missing, or how the
electrons are shared among them, is not in such a file, so it is refused
rather than guessed. Blank lines and trailing whitespace are skipped, as
MACE's readers skip them.

Standard library only; nothing here imports mace. A basis-library loader can
call basis_block_problem or check_basis_file on any block it reads.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
from typing import List, Optional


class BasisFileError(ValueError):
    """A basis-set file CRYSTAL could not read as it stands."""


def _counts(record: str) -> List[int]:
    return [int(c) for c in record.split()[1:]]


def ecp_problem(lines: List[str]):
    """(number of ECP lines, problem or None) for the ECP records that start
    at lines[0] (the line after 'NAT NSHL' of an atom with 200 < NAT < 1000).

    INPUT: 'ZNUC M M0 M1 M2 M3 M4' and M+M0+...+M4 rows (manual pp. 84-86);
    INPSOC: a 'PSONAME SOSCALE' line, then the same (pp. 84-87); anything
    else is a one-line keyword of an internal ECP library (p. 84)."""
    if not lines:
        return 0, "no ECP input after its first line"
    key = lines[0].strip().upper()
    if key not in ("INPUT", "INPSOC"):
        if len(key.split()) != 1 or not key.isalpha():
            return 1, f"{lines[0].strip()!r} is not an ECP keyword"
        return 1, None
    head = 2 if key == "INPSOC" else 1
    if len(lines) <= head:
        return len(lines), f"its {key} ECP has no 'ZNUC M M0 ... M4' record"
    try:
        float(lines[head].split()[0])
        rows = sum(_counts(lines[head]))
    except (IndexError, ValueError):
        return head + 1, f"{lines[head].strip()!r} is not an {key} 'ZNUC M M0 ... M4' record"
    end = head + 1 + rows
    if end > len(lines):
        return len(lines), f"its {key} ECP announces {rows} rows and has fewer"
    for row in lines[head + 1:end]:
        try:
            [float(x) for x in row.split()]
        except ValueError:
            return end, f"ECP row {row.strip()!r} is not numeric"
    return end, None


def shell_problem(header: str, shells: List[str]) -> Optional[str]:
    """What stops CRYSTAL reading these valence shells, or None. The first
    line gives NAT and NSHL, and NSHL shell records follow (after the ECP,
    if any), each 'ITYB LAT NG CHE SCAL' with a numeric formal charge CHE
    and, for ITYB 0, NG primitive lines (manual pp. 25-27)."""
    shells = [ln for ln in shells if ln.strip()]
    problems = []
    try:
        nshl = int(header.split()[1])
    except (IndexError, ValueError):
        nshl = None
        problems.append("no shell count in its first line")
    found, i, ng = 0, 0, 0
    while i < len(shells):
        record = shells[i].split()
        try:
            ityb, ng = int(record[0]), int(record[2])
        except (IndexError, ValueError):
            problems.append(f"{shells[i].strip()!r} is not a shell record")
            break
        try:
            float(record[3])
        except (IndexError, ValueError):
            charge = record[3] if len(record) > 3 else ""
            problem = f"shell charge {charge!r} is not a number"
            if problem not in problems:
                problems.append(problem)
        i += 1 + (ng if ityb == 0 else 0)
        found += 1
    if i > len(shells):
        problems.append(f"its last shell announces {ng} primitives and has fewer")
    if nshl is not None and found != nshl:
        problems.append(f"it announces {nshl} shells but holds {found}")
    return "; ".join(problems) or None


def basis_block_problem(text: str) -> Optional[str]:
    """What stops CRYSTAL reading the one-atom basis block ``text``, or None."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "it is empty"
    try:
        number = int(lines[0].split()[0])
    except (IndexError, ValueError):
        return f"{lines[0].strip()!r} does not start with an atomic number"
    body = lines[1:]
    if 200 < number < 1000:
        used, problem = ecp_problem(body)
        if problem:
            return problem
        body = body[used:]
    return shell_problem(lines[0], body)


def check_basis_file(path: str, text: str, z: int) -> None:
    """Raise BasisFileError, naming ``path``, when CRYSTAL could not read the
    basis block ``text`` of element ``z`` (a conventional number 200+Z or
    1xx is reduced to Z for the message)."""
    problem = basis_block_problem(text)
    if problem:
        z = int(z)
        if z > 100:
            z %= 100
        raise BasisFileError(
            f"{path} (Z={z}) cannot be used: {problem}; the file needs repairing "
            f"from the published basis set")


def library_files(directory: str) -> List[str]:
    """The numeric (per-element) file names in a basis library directory."""
    return sorted((n for n in os.listdir(directory) if n.isdigit()), key=int)
