# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Orbital Character Analysis Module

This module provides functions for computing band composition and orbital character
from eigenvalues and eigenvectors.

Functions:
- compute_band_projections: Compute projection weights for a band onto orbitals
- compute_band_character: Decompose band into element/orbital contributions
- analyze_all_bands_character: Compute character for multiple bands
- identify_dominant_character: Identify dominant orbital type in a band
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BandCharacter:
    """
    Band character decomposition by element and orbital type.

    Attributes
    ----------
    band_index : int
        Band index (0-indexed)
    characters : dict
        Nested dict: element → orbital_type → contribution (0.0 to 1.0)
    dominant : str
        Description of dominant character (e.g., "Bi-p" or "Mixed")
    """
    band_index: int
    characters: Dict[str, Dict[str, float]]
    dominant: str


def compute_band_projections(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    band_index: int
) -> np.ndarray:
    """
    Compute projection weights for a single band onto each orbital.

    Projects the band eigenstates onto the LCAO basis using the overlap matrix:
        A_n(k) = [S(k)† @ C(k)]_n,band

    The returned weights are averaged over all k-points: <|A_n|²>_k

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvector matrices C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarray
        Overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    band_index : int
        Band index to analyze (0-indexed)

    Returns
    -------
    ndarray
        Projection weights for each orbital, shape (num_orbitals,)
        Values sum to approximately 1.0 (normalized over k-points)

    Examples
    --------
    >>> projections = compute_band_projections(eigenvectors_list, S_k_list, band_idx=5)
    >>> most_contributing_orbital = np.argmax(projections)
    """
    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]

    # Accumulate |A_n(k)|² over all k-points
    projection_weights = np.zeros(num_orbitals)

    for k_idx in range(num_kpoints):
        C_k = eigenvectors_list[k_idx]  # (num_orbitals, num_bands)
        S_k = S_k_list[k_idx]           # (num_orbitals, num_orbitals)

        # Project eigenvector onto LCAO basis
        # For non-orthogonal basis: A_k = S(k) @ C(k)
        # This gives the projection in the dual basis
        A_k = S_k @ C_k  # shape (num_orbitals, num_bands)

        # Get projection for this specific band
        A_n_k = A_k[:, band_index]  # (num_orbitals,)

        # Accumulate squared magnitude
        projection_weights += np.abs(A_n_k) ** 2

    # Average over k-points
    projection_weights /= num_kpoints

    # Normalize to sum to 1.0 (account for non-orthogonal basis)
    total_weight = np.sum(projection_weights)
    if total_weight > 1e-10:
        projection_weights /= total_weight

    return projection_weights


def compute_band_character(
    band_projections: np.ndarray,
    atom_symbols: List[str],
    basis_atom_map: np.ndarray,
    orbital_types: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    Decompose band projection into element and orbital type contributions.

    Groups orbital projections by element (Bi, Se, etc.) and orbital angular momentum
    type (s, p, d, f, g).

    Parameters
    ----------
    band_projections : ndarray
        Projection weights for each orbital, shape (num_orbitals,)
    atom_symbols : list of str
        Element symbols for each atom (e.g., ['Bi', 'Bi'])
    basis_atom_map : ndarray
        Maps each orbital index to its parent atom, shape (num_orbitals,)
    orbital_types : dict
        Maps orbital index (1-indexed) → orbital type ('s', 'p', 'd', etc.)

    Returns
    -------
    dict
        Nested dictionary: element → orbital_type → contribution
        Example: {'Bi': {'s': 0.05, 'p': 0.75, 'd': 0.15}, 'Se': {'p': 0.05}}

    Examples
    --------
    >>> character = compute_band_character(projections, ['Bi', 'Bi'],
    ...                                     basis_map, orbital_types)
    >>> bi_p_contribution = character['Bi']['p']
    >>> print(f"Bi-p character: {bi_p_contribution*100:.1f}%")
    """
    characters = {}

    # Group by element and orbital type
    num_spatial_basis = len(basis_atom_map)

    for orbital_idx, weight in enumerate(band_projections):
        # Handle SOC doubling: basis_atom_map may be for spatial basis only
        # Map orbital index to spatial basis index
        spatial_idx = orbital_idx % num_spatial_basis

        # Get atom index (0-indexed in basis_atom_map)
        atom_idx = int(basis_atom_map[spatial_idx])
        element = atom_symbols[atom_idx]

        # Get orbital type (orbital_types uses 1-indexed keys)
        orb_type = orbital_types.get(orbital_idx + 1, 'unknown')

        # Initialize nested dict if needed
        if element not in characters:
            characters[element] = {}
        if orb_type not in characters[element]:
            characters[element][orb_type] = 0.0

        # Add contribution
        characters[element][orb_type] += weight

    return characters


def identify_dominant_character(
    band_character: Dict[str, Dict[str, float]],
    threshold: float = 0.3
) -> str:
    """
    Identify dominant orbital character for a band.

    Determines which element-orbital combination dominates the band character.
    If multiple types exceed the threshold, returns a combined description.

    Parameters
    ----------
    band_character : dict
        Nested dict from compute_band_character: element → orbital_type → contribution
    threshold : float
        Minimum contribution to be considered "dominant" (default: 0.3 = 30%)

    Returns
    -------
    str
        Description of dominant character:
        - "Bi-p (75%)" if single type dominates
        - "Bi-p + Se-p" if multiple types exceed threshold
        - "Mixed" if no type exceeds threshold

    Examples
    --------
    >>> dominant = identify_dominant_character({'Bi': {'p': 0.75, 's': 0.05}})
    >>> print(dominant)  # "Bi-p (75%)"
    """
    # Flatten to list of (element-type, contribution) pairs
    contributions = []
    for element, orb_dict in band_character.items():
        for orb_type, weight in orb_dict.items():
            contributions.append((f"{element}-{orb_type}", weight))

    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[1], reverse=True)

    if not contributions:
        return "Unknown"

    # Find all types above threshold
    dominant_types = [(name, weight) for name, weight in contributions if weight >= threshold]

    if not dominant_types:
        return "Mixed"
    elif len(dominant_types) == 1:
        name, weight = dominant_types[0]
        return f"{name} ({weight*100:.0f}%)"
    else:
        # Multiple dominant types
        names = [name for name, weight in dominant_types]
        return " + ".join(names)


def analyze_all_bands_character(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    atom_symbols: List[str],
    basis_atom_map: np.ndarray,
    orbital_types: Dict[int, str],
    band_indices: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    verbose: bool = False
) -> List[BandCharacter]:
    """
    Compute orbital character for multiple bands.

    Analyzes each band's projection onto different elements and orbital types.

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvector matrices for each k-point
    S_k_list : list of ndarray
        Overlap matrices for each k-point
    atom_symbols : list of str
        Element symbols
    basis_atom_map : ndarray
        Orbital to atom mapping
    orbital_types : dict
        Orbital type mapping
    band_indices : ndarray, optional
        Specific bands to analyze (default: all bands)
    threshold : float
        Threshold for dominant character identification
    verbose : bool
        Print progress information

    Returns
    -------
    list of BandCharacter
        Character analysis for each band

    Examples
    --------
    >>> results = analyze_all_bands_character(eigvecs, S_list, atoms,
    ...                                        basis_map, orb_types,
    ...                                        band_indices=np.arange(10, 20))
    >>> for band_char in results:
    ...     print(f"Band {band_char.band_index}: {band_char.dominant}")
    """
    num_bands = eigenvectors_list[0].shape[1]

    if band_indices is None:
        band_indices = np.arange(num_bands)

    results = []

    for band_idx in band_indices:
        if verbose and band_idx % 10 == 0:
            print(f"Analyzing band {band_idx+1}/{num_bands}...")

        # Compute projections
        projections = compute_band_projections(
            eigenvectors_list, S_k_list, band_idx
        )

        # Decompose into character
        character = compute_band_character(
            projections, atom_symbols, basis_atom_map, orbital_types
        )

        # Identify dominant type
        dominant = identify_dominant_character(character, threshold)

        results.append(BandCharacter(
            band_index=band_idx,
            characters=character,
            dominant=dominant
        ))

    return results


def format_band_character_table(
    band_characters: List[BandCharacter],
    elements: Optional[List[str]] = None,
    orbital_types: Optional[List[str]] = None
) -> str:
    """
    Format band character analysis as a readable table.

    Parameters
    ----------
    band_characters : list of BandCharacter
        Results from analyze_all_bands_character
    elements : list of str, optional
        Elements to include in table (default: auto-detect)
    orbital_types : list of str, optional
        Orbital types to show (default: ['s', 'p', 'd'])

    Returns
    -------
    str
        Formatted table string

    Examples
    --------
    >>> table = format_band_character_table(results, elements=['Bi'],
    ...                                      orbital_types=['s', 'p', 'd'])
    >>> print(table)
    """
    if not band_characters:
        return "No band character data available"

    # Auto-detect elements if not specified
    if elements is None:
        elements_set = set()
        for bc in band_characters:
            elements_set.update(bc.characters.keys())
        elements = sorted(elements_set)

    # Default orbital types
    if orbital_types is None:
        orbital_types = ['s', 'p', 'd', 'f']

    # Build header
    header = f"{'Band':<6} {'Dominant Character':<25}"
    for element in elements:
        for orb in orbital_types:
            header += f" {element}-{orb:<4}"

    lines = ["=" * len(header), header, "-" * len(header)]

    # Add data rows
    for bc in band_characters:
        row = f"{bc.band_index+1:<6} {bc.dominant:<25}"

        for element in elements:
            for orb in orbital_types:
                contrib = bc.characters.get(element, {}).get(orb, 0.0)
                if contrib > 0.01:  # Only show if > 1%
                    row += f" {contrib*100:>5.1f}%"
                else:
                    row += "      "

        lines.append(row)

    lines.append("=" * len(header))

    return "\n".join(lines)

