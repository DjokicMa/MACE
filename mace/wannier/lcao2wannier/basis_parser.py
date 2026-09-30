"""
Crystal23 basis set shell parser for LCAO-PDWF.

Parses the LOCAL ATOMIC FUNCTIONS BASIS SET block from Crystal23 output
into structured ShellInfo objects with per-shell metadata needed for
target mask construction.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from .valence_config import ELEMENT_Z, ELEMENT_SYMBOLS


L_FROM_LETTER = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}


@dataclass
class ShellInfo:
    """Information about one basis function shell."""
    shell_index: int       # 0-based shell number in the parsed list
    atom_index: int        # 0-based atom index in the unit cell
    atom_symbol: str       # Element symbol, e.g., 'Bi'
    atom_Z: int            # Atomic number
    l: int                 # Angular momentum (0=s, 1=p, 2=d, 3=f)
    ao_indices: range      # Range of 0-based AO indices for this shell
    num_ao: int            # = 2l+1
    radial_index: int      # 0-based among shells with same (atom, l)
    is_sp_shell: bool      # True if from a Crystal23 SP-type shell


def parse_basis_shells(lines: List[str], num_atoms: Optional[int] = None) -> Tuple[List[ShellInfo], int]:
    """
    Parse the LOCAL ATOMIC FUNCTIONS BASIS SET block into ShellInfo objects.

    Parameters
    ----------
    lines : list of str
        Lines from the Crystal23 output file.
    num_atoms : int, optional
        Total number of atoms in the unit cell. If provided and more
        atoms exist than are listed in the basis block (homoatomic
        symmetry-equivalent case), shells are replicated.

    Returns
    -------
    shells : list of ShellInfo
        Parsed shell information, one entry per (atom, l, radial) group.
    num_ao : int
        Total number of AOs in the spatial basis.
    """
    # Find the LOCAL ATOMIC FUNCTIONS BASIS SET section
    start_idx = None
    for i, line in enumerate(lines):
        if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Could not find 'LOCAL ATOMIC FUNCTIONS BASIS SET' section")

    # Patterns for shell headers
    # Single AO: "  1 S  " or "  1 SP  "
    shell_single_pattern = re.compile(r'^\s+(\d+)\s+(S|SP|P|D|F|G)\s*$')
    # Range AOs: "  5-     7 P  "
    shell_range_pattern = re.compile(r'^\s+(\d+)-\s+(\d+)\s+(S|SP|P|D|F|G)\s*$')
    # Atom header: "   1 BI   2.342   0.000   1.698"
    atom_header_pattern = re.compile(
        r'^\s+(\d+)\s+([A-Z]{1,2})\s+'
        r'([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$'
    )

    raw_shells = []  # (atom_index, atom_symbol, atom_Z, ao_start, ao_end, l, is_sp)
    atoms_found = []  # (atom_index, symbol)
    current_atom_idx = -1
    current_atom_sym = ''
    current_atom_Z = 0

    i = start_idx
    while i < len(lines):
        line = lines[i]

        # End of section
        if 'OVERLAP MATRIX' in line or ('*****' in line and i > start_idx + 3):
            break

        # Try atom header
        m = atom_header_pattern.match(line)
        if m:
            current_atom_idx = int(m.group(1)) - 1  # 0-based
            current_atom_sym = m.group(2).capitalize()
            current_atom_Z = ELEMENT_Z.get(current_atom_sym, 0)
            atoms_found.append((current_atom_idx, current_atom_sym))
            i += 1
            continue

        # Try shell range header: "  5-     7 P"
        m = shell_range_pattern.match(line)
        if m and current_atom_idx >= 0:
            ao_start = int(m.group(1)) - 1  # 0-based
            ao_end = int(m.group(2)) - 1
            letter = m.group(3)

            if letter == 'SP':
                # SP shell: 1 s AO + 3 p AOs = 4 total
                # ao_start is the s, ao_start+1 through ao_start+3 are p
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_start, ao_start, 0, True))
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_start + 1, ao_end, 1, True))
            else:
                l_val = L_FROM_LETTER[letter]
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_start, ao_end, l_val, False))
            i += 1
            continue

        # Try shell single header: "  1 S"
        m = shell_single_pattern.match(line)
        if m and current_atom_idx >= 0:
            ao_idx = int(m.group(1)) - 1  # 0-based
            letter = m.group(2)

            if letter == 'SP':
                # SP single: 1 s AO at ao_idx, 3 p AOs at ao_idx+1..ao_idx+3
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_idx, ao_idx, 0, True))
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_idx + 1, ao_idx + 3, 1, True))
            else:
                l_val = L_FROM_LETTER[letter]
                raw_shells.append((current_atom_idx, current_atom_sym,
                                   current_atom_Z, ao_idx, ao_idx, l_val, False))
            i += 1
            continue

        i += 1

    if not raw_shells:
        raise ValueError("No shells found in LOCAL ATOMIC FUNCTIONS BASIS SET section")

    # Handle symmetry-equivalent atoms.
    # Crystal23 lists basis functions only for irreducible atoms.
    # Equivalent atoms appear in the header but have no shells.
    # We replicate shells from a same-element template atom.
    #
    # IMPORTANT: CRYSTAL assigns AO indices in strict atom order
    # (atom 0's AOs come before atom 1's, etc.), even though the BASIS SET
    # block only prints explicit shells for symmetry-irreducible atoms.
    # When atoms are interleaved (e.g. Cr1 explicit, Cr2 template, I3 explicit,
    # I4..I8 templates), we must place each templated atom's AOs in the
    # natural atom-index slot, not at the end. Otherwise the parser double-
    # counts the "gap" left by skipped explicit AO indices.
    atoms_with_shells = set(a_idx for a_idx, _, _, _, _, _, _ in raw_shells)

    # Build element -> template atom map (first explicit atom of each element)
    element_template = {}
    for a_idx, a_sym, _, _, _, _, _ in raw_shells:
        if a_sym not in element_template:
            element_template[a_sym] = a_idx

    # Compute per-atom AO count:
    #   - Explicit atoms: sum of (ao_end - ao_start + 1) over their shells
    #   - Template atoms: copy the count from their element's template
    atom_ao_counts: Dict[int, int] = {}
    for a_idx, _, _, ao_s, ao_e, _, _ in raw_shells:
        atom_ao_counts[a_idx] = atom_ao_counts.get(a_idx, 0) + (ao_e - ao_s + 1)

    # Walk atoms in natural (index) order and assign start positions
    sorted_atoms = sorted(atoms_found, key=lambda x: x[0])
    atom_start_ao: Dict[int, int] = {}
    current_ao = 0
    for a_idx, a_sym in sorted_atoms:
        atom_start_ao[a_idx] = current_ao
        if a_idx in atom_ao_counts:
            current_ao += atom_ao_counts[a_idx]
        elif a_sym in element_template:
            template_atom = element_template[a_sym]
            current_ao += atom_ao_counts[template_atom]
        # else: no template, atom gets 0 AOs (pathological, but keep going)
    num_ao_explicit = current_ao

    # Replicate template shells into non-explicit atoms using correct offsets
    atoms_needing_replication = [
        (idx, sym) for idx, sym in atoms_found
        if idx not in atoms_with_shells
    ]

    if atoms_needing_replication:
        replicated = []
        for atom_new, atom_sym in atoms_needing_replication:
            template_atom = element_template.get(atom_sym)
            if template_atom is None:
                continue  # No template found for this element

            # Template atom's first AO (the shell's lowest ao_start)
            template_ao_min = min(
                ao_s for a_idx, _, _, ao_s, _, _, _ in raw_shells
                if a_idx == template_atom
            )
            ao_offset = atom_start_ao[atom_new] - template_ao_min

            for (a_idx, a_sym, a_Z, ao_s, ao_e, l_v, is_sp) in raw_shells:
                if a_idx == template_atom:
                    replicated.append((atom_new, a_sym, a_Z,
                                       ao_s + ao_offset, ao_e + ao_offset,
                                       l_v, is_sp))

        raw_shells.extend(replicated)

    # Assign radial indices and build ShellInfo list
    radial_counter = {}  # (atom_index, l) -> count
    shells = []

    for shell_idx, (a_idx, a_sym, a_Z, ao_s, ao_e, l_val, is_sp) in enumerate(raw_shells):
        key = (a_idx, l_val)
        radial_idx = radial_counter.get(key, 0)
        radial_counter[key] = radial_idx + 1

        num_ao_shell = ao_e - ao_s + 1
        shells.append(ShellInfo(
            shell_index=shell_idx,
            atom_index=a_idx,
            atom_symbol=a_sym,
            atom_Z=a_Z,
            l=l_val,
            ao_indices=range(ao_s, ao_e + 1),
            num_ao=num_ao_shell,
            radial_index=radial_idx,
            is_sp_shell=is_sp,
        ))

    return shells, num_ao_explicit


def get_atom_list(shells: List[ShellInfo]) -> List[str]:
    """Extract unique atom symbols in unit-cell order from shells."""
    seen = set()
    atoms = []
    for s in shells:
        if s.atom_index not in seen:
            seen.add(s.atom_index)
            atoms.append(s.atom_symbol)
    return atoms
