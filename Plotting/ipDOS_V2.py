#!/usr/bin/python3
"""
================================================================================
Density of States (DOS) Plotting Script for CRYSTAL17/23 Output Files (V7)
================================================================================

FIXES IN V7:
- upper_half/lower_half now properly scales based on energy-restricted region
- Separate --vb-range and --cb-range flags for composition analysis

================================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import matplotlib as mpl
import glob
from os.path import exists
from collections import defaultdict

# =============================================================================
# SHARED CONFIGURATION (Keep consistent with ipBANDS_V7.py)
# =============================================================================

DEFAULT_FIG_WIDTH = 4
DEFAULT_FIG_HEIGHT = 8

# Subplot parameters for exact alignment between BAND and DOS plots
SUBPLOT_LEFT = 0.18
SUBPLOT_RIGHT = 0.82
SUBPLOT_TOP = 0.92
SUBPLOT_BOTTOM = 0.08

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.usetex": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
    }
)

# =============================================================================
# ELEMENT COLORS - SHARED WITH BAND PLOT
# =============================================================================

ELEMENT_COLORS = {
    # Transition metals - specified by user
    "Fe": (1, 176 / 255, 67 / 255),  # Warm orange
    "Co": (0.92, 0.37, 0.34),  # Reddish orange
    "Mn": (94 / 255, 87 / 255, 235 / 255),  # Blue-violet
    "Ni": (228 / 255, 245 / 255, 87 / 255),  # Yellow-green
    "Pd": (161 / 255, 160 / 255, 160 / 255),  # Medium gray
    "Pt": (0, 1, 1),  # Cyan
    # Key light elements - specified by user
    "B": (1.0, 0.75, 0.85),  # Light pink
    "O": (0.7, 0.1, 0.1),  # Darker red
    "H": (0.53, 0.81, 0.98),  # Light sky blue (standard for H)
    # Other common elements
    "C": (0.36, 0.36, 0.36),  # Dark gray
    "N": (0.24, 0.35, 0.50),  # Navy blue
    "F": (0.02, 1.0, 0.65),  # Bright green
    "S": (0.96, 0.82, 0.44),  # Yellow
    "P": (1.0, 0.42, 0.21),  # Orange
    "Cl": (0.61, 0.81, 0.33),  # Green
    "Si": (0.64, 0.72, 0.77),  # Gray-blue
    "Al": (0.78, 0.71, 0.65),  # Tan
    "Na": (1.0, 0.85, 0.24),  # Gold
    "Mg": (0.42, 0.81, 0.50),  # Green
    "K": (0.72, 0.25, 0.37),  # Maroon
    "Ca": (0.97, 0.78, 0.62),  # Peach
    "Ti": (0.72, 0.72, 0.82),  # Light purple-gray
    "V": (0.78, 0.64, 0.78),  # Light purple
    "Cr": (0.70, 0.66, 0.84),  # Lavender
    "Cu": (0.80, 0.50, 0.20),  # Copper
    "Zn": (0.49, 0.71, 0.83),  # Light blue
    "Ag": (0.75, 0.75, 0.75),  # Silver
    "Au": (1.0, 0.84, 0.0),  # Gold
    "Ru": (0.49, 0.56, 0.59),  # Gray-blue
    "Rh": (0.48, 0.63, 0.59),  # Teal-gray
    "Ir": (0.61, 0.63, 0.64),  # Gray
    "Os": (0.54, 0.57, 0.57),  # Dark gray
    "Re": (0.65, 0.65, 0.65),  # Gray
    "W": (0.45, 0.45, 0.45),  # Dark gray
    "Ta": (0.29, 0.67, 0.75),  # Teal
    "Hf": (0.64, 0.77, 0.74),  # Light teal
    "Zr": (0.62, 0.71, 0.72),  # Light gray-blue
    "Y": (0.71, 0.89, 0.81),  # Light green
    "Sc": (0.93, 0.80, 0.77),  # Light pink-tan
    "La": (0.44, 0.67, 0.84),  # Blue
    "Ce": (1.0, 0.90, 0.73),  # Light yellow
    "Bi": (0.62, 0.31, 0.87),  # Purple
    "Pb": (0.35, 0.35, 0.35),  # Dark gray
    "Sn": (0.65, 0.65, 0.65),  # Gray
    "In": (0.65, 0.45, 0.39),  # Brown
    "Ga": (0.65, 0.74, 0.68),  # Gray-green
    "Ge": (0.66, 0.68, 0.74),  # Gray-blue
    "As": (0.61, 0.55, 0.66),  # Purple-gray
    "Se": (1.0, 0.70, 0.22),  # Orange
    "Br": (0.55, 0.26, 0.08),  # Brown
    "Te": (0.83, 0.69, 0.21),  # Gold
    "I": (0.58, 0.0, 0.83),  # Purple
    "Li": (0.59, 0.89, 0.83),  # Light teal
    "Be": (0.95, 0.50, 0.50),  # Pink
    "Rb": (0.91, 0.29, 0.24),  # Red
    "Sr": (1.0, 0.85, 0.0),  # Yellow
    "Cs": (0.90, 0.47, 0.30),  # Orange
    "Ba": (0.0, 0.78, 0.32),  # Green
}

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
}

LIGHT_ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
]

TRANSITION_METALS = [
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
]

ORBITAL_ORDER = {"s": 0, "p": 1, "d": 2, "f": 3, None: 4}


def get_element_from_label(label):
    if label == "Total":
        return "Total"
    parts = label.split()
    if parts:
        elem = "".join(c for c in parts[0] if c.isalpha())
        return elem
    return label


def get_orbital_from_label(label):
    if "(s)" in label:
        return "s"
    elif "(p)" in label:
        return "p"
    elif "(d)" in label:
        return "d"
    elif "(f)" in label:
        return "f"
    else:
        return None


def parse_element_mode(element_mode):
    """Parse element mode - including tm_orb option"""
    element_treatment = {}

    if element_mode == "all":
        return {}
    elif element_mode == "light_total":
        for elem in ELEMENT_COLORS.keys():
            element_treatment[elem] = "orbital"
        for elem in LIGHT_ELEMENTS:
            element_treatment[elem] = "total"
        return element_treatment
    elif element_mode == "tm_both":
        for elem in ELEMENT_COLORS.keys():
            element_treatment[elem] = "total"
        for elem in TRANSITION_METALS:
            element_treatment[elem] = "both"
        return element_treatment
    elif element_mode == "tm_orb":
        for elem in ELEMENT_COLORS.keys():
            element_treatment[elem] = "total"
        for elem in TRANSITION_METALS:
            element_treatment[elem] = "orbital"
        return element_treatment
    elif element_mode.startswith("custom:"):
        spec = element_mode[7:]
        groups = spec.split(";")
        for group in groups:
            if "=" not in group:
                continue
            elements_str, treatment = group.split("=")
            elements = [e.strip() for e in elements_str.split(",")]
            for elem in elements:
                element_treatment[elem] = treatment
        return element_treatment
    return {}


def should_include_orbital(label, element_treatment, default_proj_type):
    elem = get_element_from_label(label)
    orbital = get_orbital_from_label(label)

    if label == "Total":
        return False

    treatment = element_treatment.get(elem, default_proj_type)

    if treatment == "total":
        return orbital is None
    elif treatment == "orbital":
        return orbital is not None
    elif treatment == "both":
        return True
    return False


def sort_labels_by_element_and_orbital(labels):
    def sort_key(label):
        elem = get_element_from_label(label)
        orbital = get_orbital_from_label(label)
        atomic_num = ATOMIC_NUMBERS.get(elem, 999)
        orbital_order = ORBITAL_ORDER[orbital]
        return (atomic_num, orbital_order)

    return sorted(labels, key=sort_key)


def extract_band_gap(material):
    """Extract band gap from .out file"""
    out_file = material + "_doss.out"

    if not exists(out_file):
        return None

    band_gap_info = {}
    current_spin = None

    with open(out_file) as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "ALPHA" in line and "ELECTRONS" in line:
                current_spin = "alpha"
                continue
            elif "BETA" in line and "ELECTRONS" in line:
                current_spin = "beta"
                continue

            if "POSSIBLY CONDUCTING STATE" in line:
                if current_spin == "alpha":
                    band_gap_info["alpha"] = 0.0
                elif current_spin == "beta":
                    band_gap_info["beta"] = 0.0
                else:
                    band_gap_info["gap"] = 0.0
                continue

            if "DIRECT ENERGY BAND GAP" in line:
                gap_value = float(line.split()[-2])
                if current_spin == "alpha":
                    band_gap_info["alpha"] = gap_value
                elif current_spin == "beta":
                    band_gap_info["beta"] = gap_value
                else:
                    band_gap_info["gap"] = gap_value

            if "INDIRECT ENERGY BAND GAP" in line:
                gap_value = float(line.split()[-2])
                if current_spin == "alpha":
                    if "alpha" in band_gap_info:
                        band_gap_info["alpha"] = min(band_gap_info["alpha"], gap_value)
                    else:
                        band_gap_info["alpha"] = gap_value
                elif current_spin == "beta":
                    if "beta" in band_gap_info:
                        band_gap_info["beta"] = min(band_gap_info["beta"], gap_value)
                    else:
                        band_gap_info["beta"] = gap_value
                else:
                    if "gap" in band_gap_info:
                        band_gap_info["gap"] = min(band_gap_info["gap"], gap_value)
                    else:
                        band_gap_info["gap"] = gap_value

    if "alpha" in band_gap_info and "beta" in band_gap_info:
        alpha_gap = band_gap_info["alpha"]
        beta_gap = band_gap_info["beta"]

        if alpha_gap == beta_gap:
            band_gap_info = {"gap": alpha_gap}
        elif alpha_gap == 0.0 and beta_gap > 0:
            band_gap_info = {"semimetal": beta_gap}
        elif beta_gap == 0.0 and alpha_gap > 0:
            band_gap_info = {"semimetal": alpha_gap}
        elif alpha_gap == 0.0 and beta_gap == 0.0:
            band_gap_info = {"gap": 0.0}

    return band_gap_info


def generate_color_scheme(labels, search):
    """Generate colors and linestyles for each label"""
    from matplotlib.colors import to_rgb, to_hex
    import colorsys

    color_map = {}
    linestyle_map = {}

    orbital_linestyles = {"s": "-", "p": "--", "d": ":", "f": "-."}
    dark_orbital_factors = {"s": 1.4, "p": 1.2, "d": 1.0, "f": 0.85}
    light_orbital_factors = {"s": 0.85, "p": 0.65, "d": 0.50, "f": 0.35}

    for label in search:
        elem = get_element_from_label(label)
        orbital = get_orbital_from_label(label)

        base_color = ELEMENT_COLORS.get(elem, (0.5, 0.5, 0.5))

        if isinstance(base_color, str):
            base_color = to_rgb(base_color)

        if orbital is None:
            color_map[label] = base_color
            linestyle_map[label] = "-"
        else:
            rgb = base_color
            hsv = colorsys.rgb_to_hsv(*rgb)

            is_dark = hsv[2] < 0.5
            factors = dark_orbital_factors if is_dark else light_orbital_factors
            factor = factors[orbital]

            new_value = min(1.0, hsv[2] * factor) if is_dark else hsv[2] * factor
            new_rgb = colorsys.hsv_to_rgb(hsv[0], min(1.0, hsv[1] * 1.1), new_value)

            color_map[label] = new_rgb
            linestyle_map[label] = orbital_linestyles[orbital]

    return color_map, linestyle_map


def calculate_dos_order(data_vect, labels, search, energy_data, E_l, E_u, maxV):
    """
    Calculate z-order for DOS lines based on their maximum values.
    Larger DOS values should be plotted first (in background).
    """
    max_dos_values = {}

    energy_array = np.array(energy_data)
    energy_mask = (energy_array >= E_l + maxV) & (energy_array <= E_u + maxV)

    for proj in search:
        for i in range(1, len(data_vect)):
            if labels[i] == proj:
                dos_data = np.array(data_vect[i])
                visible_dos = np.abs(dos_data[energy_mask])
                if len(visible_dos) > 0:
                    max_dos = np.max(visible_dos)
                else:
                    max_dos = 0
                max_dos_values[proj] = max_dos
                break

    sorted_search = sorted(search, key=lambda x: max_dos_values.get(x, 0), reverse=True)

    return sorted_search


def find_band_edges(data_vect, labels, energy_data, maxV, threshold_fraction=0.01):
    """
    Find the VBM and CBM energies from the Total DOS.
    Returns (vbm_energy, cbm_energy) or (None, None) if not found.
    """
    energy_array = np.array(energy_data)

    # Find Total DOS
    total_dos = None
    for i in range(1, len(data_vect)):
        if labels[i] == "Total":
            total_dos = np.abs(np.array(data_vect[i]))
            break

    if total_dos is None or len(total_dos) == 0:
        return None, None

    # Threshold for "significant" DOS
    try:
        max_dos = np.max(total_dos)
        if max_dos == 0 or not np.isfinite(max_dos):
            return None, None
        threshold = max_dos * threshold_fraction
    except (ValueError, RuntimeWarning):
        return None, None

    # Find VBM: highest energy below Fermi with significant DOS
    below_fermi = energy_array < maxV
    significant_below = (total_dos > threshold) & below_fermi

    if np.any(significant_below):
        indices = np.where(significant_below)[0]
        if len(indices) > 0:
            vbm_idx = indices[-1]
            vbm_energy = energy_array[vbm_idx]
        else:
            vbm_energy = None
    else:
        vbm_energy = None

    # Find CBM: lowest energy above Fermi with significant DOS
    above_fermi = energy_array > maxV
    significant_above = (total_dos > threshold) & above_fermi

    if np.any(significant_above):
        indices = np.where(significant_above)[0]
        if len(indices) > 0:
            cbm_idx = indices[0]
            cbm_energy = energy_array[cbm_idx]
        else:
            cbm_energy = None
    else:
        cbm_energy = None

    return vbm_energy, cbm_energy


def analyze_fermi_composition(
    data_vect, labels, energy_array, fermi_mask, maxV, fermi_range
):
    """
    Analyze DOS composition around Fermi level for conductive materials.
    Returns a single 'fermi' composition instead of separate vb/cb.
    """
    fermi_contributions = defaultdict(float)

    for i in range(1, len(data_vect)):
        label = labels[i]
        if label == "Total":
            continue

        dos_data = np.array(data_vect[i])

        if np.any(fermi_mask) and len(dos_data[fermi_mask]) > 0:
            fermi_dos = np.abs(dos_data[fermi_mask])
            fermi_contributions[label] = np.sum(fermi_dos)

    def format_composition(contributions):
        if not contributions:
            return ""

        element_totals = defaultdict(float)
        element_orbitals = defaultdict(list)

        for label, contrib in contributions.items():
            elem = get_element_from_label(label)
            orbital = get_orbital_from_label(label)
            element_totals[elem] += contrib
            if orbital:
                element_orbitals[elem].append((orbital, contrib))
            else:
                element_orbitals[elem].append((None, contrib))

        if not element_totals:
            return ""
        max_contrib = max(element_totals.values())
        if max_contrib == 0:
            return ""
        threshold = max_contrib * 0.05

        significant_elements = [
            (elem, total)
            for elem, total in element_totals.items()
            if total >= threshold
        ]
        significant_elements.sort(key=lambda x: x[1], reverse=True)
        significant_elements = significant_elements[:4]

        parts = []
        for elem, _ in significant_elements:
            orbitals = element_orbitals[elem]
            orbitals.sort(key=lambda x: x[1], reverse=True)

            orbital_strs = []
            for orb, orb_contrib in orbitals:
                if orb is not None and orb_contrib >= threshold * 0.5:
                    orbital_strs.append(orb)

            if orbital_strs:
                parts.append(f"{elem}({','.join(orbital_strs)})")
            else:
                parts.append(elem)

        return ", ".join(parts)

    return {
        "vb": "",
        "cb": "",
        "fermi": format_composition(fermi_contributions),
        "fermi_range": fermi_range,
        "vb_energy": maxV,
        "cb_energy": maxV,
        "vbm": None,
        "cbm": None,
        "is_conductive": True,
    }


def analyze_composition(
    data_vect,
    labels,
    energy_data,
    maxV,
    vb_range=0.5,
    cb_range=0.5,
    is_conductive=False,
):
    """
    Analyze DOS composition near VB and CB edges.

    Args:
        data_vect: DOS data vectors
        labels: Labels for each DOS column
        energy_data: Energy values
        maxV: Fermi level / vacuum reference
        vb_range: Energy range below VBM (or E_F) to analyze for VB composition
        cb_range: Energy range above CBM (or E_F) to analyze for CB composition
        is_conductive: If True, analyze around Fermi level instead of VB/CB

    Returns dict with formatted composition strings and energy positions.
    """
    energy_array = np.array(energy_data)

    # For conductive materials, analyze around Fermi level
    if is_conductive:
        fermi_range = max(vb_range, cb_range)
        fermi_mask = (energy_array >= maxV - fermi_range) & (
            energy_array <= maxV + fermi_range
        )

        if not np.any(fermi_mask):
            fermi_mask = (energy_array >= maxV - 1.0) & (energy_array <= maxV + 1.0)

        return analyze_fermi_composition(
            data_vect, labels, energy_array, fermi_mask, maxV, fermi_range
        )

    # Find band edges with error handling
    try:
        vbm_energy, cbm_energy = find_band_edges(data_vect, labels, energy_data, maxV)
    except Exception as e:
        print(f"  Warning: Could not find band edges: {e}")
        vbm_energy, cbm_energy = None, None

    # Define analysis regions based on band edges
    if vbm_energy is not None:
        vb_center = vbm_energy
        vb_mask = (energy_array >= vbm_energy - vb_range) & (energy_array <= vbm_energy)
    else:
        vb_center = maxV - 0.25
        vb_mask = (energy_array >= maxV - vb_range) & (energy_array < maxV)

    if cbm_energy is not None:
        cb_center = cbm_energy
        cb_mask = (energy_array >= cbm_energy) & (energy_array <= cbm_energy + cb_range)
    else:
        cb_center = maxV + 0.25
        cb_mask = (energy_array > maxV) & (energy_array <= maxV + cb_range)

    # Check if masks have any True values
    if not np.any(vb_mask):
        vb_mask = (energy_array >= maxV - vb_range) & (energy_array < maxV)
    if not np.any(cb_mask):
        cb_mask = (energy_array > maxV) & (energy_array <= maxV + cb_range)

    # Collect contributions by element and orbital
    vb_contributions = defaultdict(float)
    cb_contributions = defaultdict(float)

    for i in range(1, len(data_vect)):
        label = labels[i]
        if label == "Total":
            continue

        dos_data = np.array(data_vect[i])

        if np.any(vb_mask) and len(dos_data[vb_mask]) > 0:
            vb_dos = np.abs(dos_data[vb_mask])
            vb_contributions[label] = np.sum(vb_dos)

        if np.any(cb_mask) and len(dos_data[cb_mask]) > 0:
            cb_dos = np.abs(dos_data[cb_mask])
            cb_contributions[label] = np.sum(cb_dos)

    def format_composition(contributions):
        """
        Format: Element (orbital1, orbital2), Element2 (orbital)
        Elements ordered by total contribution, orbitals within each element by contribution.
        """
        if not contributions:
            return ""

        # Group by element
        element_totals = defaultdict(float)
        element_orbitals = defaultdict(list)

        for label, contrib in contributions.items():
            elem = get_element_from_label(label)
            orbital = get_orbital_from_label(label)
            element_totals[elem] += contrib
            if orbital:
                element_orbitals[elem].append((orbital, contrib))
            else:
                element_orbitals[elem].append((None, contrib))

        # Filter to significant contributors (> 5% of max)
        if not element_totals:
            return ""
        max_contrib = max(element_totals.values())
        if max_contrib == 0:
            return ""
        threshold = max_contrib * 0.05

        significant_elements = [
            (elem, total)
            for elem, total in element_totals.items()
            if total >= threshold
        ]
        significant_elements.sort(key=lambda x: x[1], reverse=True)

        # Limit to top 4 elements
        significant_elements = significant_elements[:4]

        # Build formatted string
        parts = []
        for elem, _ in significant_elements:
            orbitals = element_orbitals[elem]
            orbitals.sort(key=lambda x: x[1], reverse=True)

            # Get significant orbitals for this element
            orbital_strs = []
            for orb, orb_contrib in orbitals:
                if orb is not None and orb_contrib >= threshold * 0.5:
                    orbital_strs.append(orb)

            if orbital_strs:
                parts.append(f"{elem}({','.join(orbital_strs)})")
            else:
                parts.append(elem)

        return ", ".join(parts)

    return {
        "vb": format_composition(vb_contributions),
        "cb": format_composition(cb_contributions),
        "fermi": "",
        "vb_energy": vb_center,
        "cb_energy": cb_center,
        "vbm": vbm_energy,
        "cbm": cbm_energy,
        "is_conductive": False,
    }


def find_optimal_legend_position(
    ax,
    data_vect,
    labels,
    search,
    xlim_neg,
    xlim_pos,
    E_l,
    E_u,
    maxV,
    vbm_energy=None,
    cbm_energy=None,
):
    """
    Find optimal legend position, prioritizing band gap region if available.
    """
    y_bottom = E_l + maxV
    y_top = E_u + maxV
    y_range = y_top - y_bottom
    x_range = xlim_pos - xlim_neg

    # Check if there's a band gap we can use
    if vbm_energy is not None and cbm_energy is not None:
        gap_size = cbm_energy - vbm_energy
        if gap_size > 0.3:
            gap_center = (vbm_energy + cbm_energy) / 2
            if y_bottom < gap_center < y_top:
                return "center left"

    # Otherwise, check corner densities
    corner_fraction = 0.25

    corners = {
        "upper left": (
            xlim_neg,
            xlim_neg + corner_fraction * x_range,
            y_top - corner_fraction * y_range,
            y_top,
        ),
        "upper right": (
            xlim_pos - corner_fraction * x_range,
            xlim_pos,
            y_top - corner_fraction * y_range,
            y_top,
        ),
        "lower left": (
            xlim_neg,
            xlim_neg + corner_fraction * x_range,
            y_bottom,
            y_bottom + corner_fraction * y_range,
        ),
        "lower right": (
            xlim_pos - corner_fraction * x_range,
            xlim_pos,
            y_bottom,
            y_bottom + corner_fraction * y_range,
        ),
    }

    corner_density = {}
    energy_data = np.array(data_vect[0])

    for corner_name, (x1, x2, y1, y2) in corners.items():
        count = 0
        in_y = (energy_data >= y1) & (energy_data <= y2)

        for proj in search:
            for i in range(1, len(data_vect)):
                if labels[i] == proj:
                    dos_data = np.array(data_vect[i])
                    in_x = (dos_data >= x1) & (dos_data <= x2)
                    count += np.sum(in_x & in_y)
                    break

        corner_density[corner_name] = count

    best_corner = min(corner_density, key=corner_density.get)
    return best_corner


def ipDOS(
    material,
    E_l,
    E_u,
    proj_type,
    x_scale_mode="auto",
    element_mode="all",
    show_composition=True,
    vb_range=0.5,
    cb_range=0.5,
    fixed_height=DEFAULT_FIG_HEIGHT,
    formats="png,svg",
    output_dir=".",
):
    """Main DOS plotting function"""

    file = material + "_doss.DOSS.DAT"
    file1 = material + "_doss.d3"
    file4 = material + "_POTC.POTC.DAT"
    file5 = material + "_POTC.out"

    v = []
    labels = ["Energy (eV)"]
    n = 9999
    num = 9999

    with open(file1) as F:
        for i, line in enumerate(F):
            if line.startswith("DOSS"):
                n = i
            if i == n + 1:
                l = line.split()
                num = int(l[0])
            if i >= n + 3 and i <= n + 3 + num:
                v.append(str(line[max(line.find("#"), 0) :].strip()))
            if line.startswith("END"):
                break

    element_treatment = parse_element_mode(element_mode)

    search = []
    for i in v:
        j = i.replace(" all", " ")
        j = j.replace(" S", " (s)")
        j = j.replace(" P", " (p)")
        j = j.replace(" D", " (d)")
        j = j.replace(" F", " (f)")
        j = j.replace("END", "Total")
        l = j.replace("#", "")
        if len(l) > 1:
            if l[0].isupper() and l[1].isupper():
                j = list(l)
                j[1] = j[1].lower()
                l = "".join(j)
        labels.append(l)

        if should_include_orbital(l, element_treatment, proj_type):
            search.append(l)

    search = sort_labels_by_element_and_orbital(search)

    # Handle vacuum reference
    if exists(file4) and exists(file5):
        z = []
        V = []
        EF = 0
        with open(file5) as f5:
            for i, line in enumerate(f5):
                if "FERMI ENERGY" in line:
                    EF = float(line.split()[-1])
        with open(file4) as f:
            for i, line in enumerate(f):
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    z.append(float(line.split()[0]))
                    V.append(float(line.split()[1]))
        maxV = -(V[0] - EF) * 27.2114
    else:
        maxV = 0

    data_vect = [[] for n in range(len(labels))]

    # Read DOS data
    with open(file) as f:
        for i, line in enumerate(f):
            if line.startswith("# EFERMI"):
                continue
            if line.startswith("#") or line.startswith("@") or line.startswith("&"):
                continue

            data = line.split()

            if len(labels) >= 17:
                ne = next(f, "")
                if ne:
                    ne = ne.split()
                    if len(data) >= 17 and len(ne) < 17:
                        data = np.concatenate([data, ne])

            data_vect[0].append(float(data[0]) * 27.2114 + maxV)

            for j in range(1, len(labels)):
                data_vect[j].append(float(data[j]) / 27.2114)

    # Determine x-axis scaling based on mode
    energy_array = np.array(data_vect[0])

    # Safeguard: check if we have any data at all
    if len(energy_array) == 0:
        print(f"  Error: No energy data found in file")
        return

    # Determine scaling region
    if x_scale_mode == "upper_half":
        scale_e_min = maxV  # From Fermi level
        scale_e_max = E_u + maxV  # To upper plot limit
        print(
            f"  Scaling based on UPPER half only: E_F ({maxV:.2f}) to {scale_e_max:.2f} eV"
        )
    elif x_scale_mode == "lower_half":
        scale_e_min = E_l + maxV  # From lower plot limit
        scale_e_max = maxV  # To Fermi level
        print(
            f"  Scaling based on LOWER half only: {scale_e_min:.2f} to E_F ({maxV:.2f}) eV"
        )
    elif x_scale_mode.startswith("fermi:"):
        try:
            range_val = float(x_scale_mode.split(":")[1])
            scale_e_min = maxV - range_val
            scale_e_max = maxV + range_val
            print(f"  Scaling based on Fermi ± {range_val} eV")
        except:
            scale_e_min = E_l + maxV
            scale_e_max = E_u + maxV
    elif x_scale_mode.startswith("energy:"):
        try:
            _, e1, e2 = x_scale_mode.split(":")
            scale_e_min = float(e1) + maxV
            scale_e_max = float(e2) + maxV
            print(
                f"  Scaling based on energy range: {scale_e_min:.2f} to {scale_e_max:.2f} eV"
            )
        except:
            scale_e_min = E_l + maxV
            scale_e_max = E_u + maxV
    else:
        # "auto" mode - use full energy range
        scale_e_min = E_l + maxV
        scale_e_max = E_u + maxV

    # Create energy mask for the scaling region
    energy_mask = (energy_array >= scale_e_min) & (energy_array <= scale_e_max)

    # Check if mask has any True values
    if not np.any(energy_mask):
        print(
            f"  Warning: No data points in energy range [{scale_e_min:.2f}, {scale_e_max:.2f}]"
        )
        print(f"  Falling back to full energy range for scaling")
        energy_mask = np.ones(len(energy_array), dtype=bool)

    # Collect DOS values ONLY from the scaling region for x-axis limits
    scaling_dos_values = []

    # Get Total DOS in the scaling region
    for i in range(1, len(data_vect)):
        if labels[i] == "Total":
            visible_dos = np.array(data_vect[i])[energy_mask]
            if len(visible_dos) > 0:
                scaling_dos_values.extend(visible_dos)
                print(
                    f"  Total DOS in scaling region: max={np.max(np.abs(visible_dos)):.4f}"
                )
            else:
                print(f"  Warning: No Total DOS data in scaling region")
            break

    # Include orbital projections in scaling region
    for proj in search:
        for i in range(1, len(data_vect)):
            if labels[i] == proj:
                visible_dos = np.array(data_vect[i])[energy_mask]
                if len(visible_dos) > 0:
                    scaling_dos_values.extend(visible_dos)
                break

    scaling_dos_values = [v for v in scaling_dos_values if np.isfinite(v)]

    if len(scaling_dos_values) == 0:
        max_dos = 1.0
        print(f"  Warning: No valid DOS values in scaling region, using default")
    else:
        max_dos = np.max(np.abs(scaling_dos_values))
        print(f"  Max DOS in scaling region: {max_dos:.4f}")

    # Apply x-axis limits
    if x_scale_mode.startswith("custom:"):
        try:
            _, neg_val, pos_val = x_scale_mode.split(":")
            xlim_neg = -float(neg_val)
            xlim_pos = float(pos_val)
        except:
            xlimit = max_dos * 1.1
            xlim_neg = -xlimit
            xlim_pos = xlimit
    else:
        # Add 10% padding
        xlimit = max_dos * 1.1
        xlim_neg = -xlimit
        xlim_pos = xlimit

    # Sanity check - only apply if limits are truly unreasonable (like 0 or negative)
    if xlimit <= 0 or not np.isfinite(xlimit):
        print(f"  Warning: Invalid x-limit ({xlimit}), falling back to full range")
        # Fall back to using all Total DOS data
        for i in range(1, len(data_vect)):
            if labels[i] == "Total":
                all_total_dos = np.abs(np.array(data_vect[i]))
                max_dos = np.max(all_total_dos) * 1.1
                xlim_neg = -max_dos
                xlim_pos = max_dos
                break

    print(f"  X-axis limits: {xlim_neg:.4f} to {xlim_pos:.4f}")

    band_gap_info = extract_band_gap(material)

    # Analyze composition near Fermi level
    composition_info = None
    if show_composition:
        try:
            # Determine if conductive based on band_gap_info (same source as Eg legend)
            is_conductive = False
            if band_gap_info:
                if "gap" in band_gap_info and band_gap_info["gap"] == 0.0:
                    is_conductive = True
                elif "semimetal" in band_gap_info:
                    is_conductive = (
                        True  # Half-metal, treat as conductive for composition
                    )
                # If gap > 0, it's a semiconductor - use VB/CB labels

            composition_info = analyze_composition(
                data_vect, labels, data_vect[0], maxV, vb_range, cb_range, is_conductive
            )
            if composition_info.get("is_conductive"):
                if composition_info.get("fermi"):
                    fermi_range = composition_info.get("fermi_range", 0.5)
                    print(
                        f"  Conductive - E_F composition (within ±{fermi_range} eV): {composition_info['fermi']}"
                    )
            else:
                if composition_info["vb"]:
                    print(
                        f"  VB composition (within {vb_range} eV of VBM): {composition_info['vb']}"
                    )
                if composition_info["cb"]:
                    print(
                        f"  CB composition (within {cb_range} eV of CBM): {composition_info['cb']}"
                    )
        except Exception as e:
            print(f"  Warning: Could not analyze composition: {e}")
            composition_info = None

    # Calculate z-ordering for DOS lines
    search_ordered = calculate_dos_order(
        data_vect, labels, search, data_vect[0], E_l, E_u, maxV
    )

    # Create figure with fixed subplot parameters for alignment
    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, fixed_height))
    ax = fig.add_axes(
        [
            SUBPLOT_LEFT,
            SUBPLOT_BOTTOM,
            SUBPLOT_RIGHT - SUBPLOT_LEFT,
            SUBPLOT_TOP - SUBPLOT_BOTTOM,
        ]
    )

    ax.set_title("Density of States", pad=10, fontsize=18, weight="bold")

    # Set black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Y-axis label with properly BOLD E-E_F (both E and subscript F)
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. Vacuum (eV)", fontsize=16, weight="bold")
    else:
        # Use mathbf for the entire expression to make E, minus, and subscript all bold
        ax.set_ylabel(r"$\mathbf{E-E_F}$ (eV)", fontsize=16, weight="bold")

    ax.set_xlim(xlim_neg, xlim_pos)
    ax.set_ylim(E_l + maxV, E_u + maxV)

    # Tick setup
    tick_positions = np.linspace(xlim_neg, xlim_pos, 7)
    minor_positions = []
    for i in range(len(tick_positions) - 1):
        minor_positions.append((tick_positions[i] + tick_positions[i + 1]) / 2)

    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_minor_locator(FixedLocator(minor_positions))

    # Grid
    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=0.5,
        linewidth=0.8,
        color="#666666",
        linestyle="-",
    )
    ax.grid(
        True,
        which="minor",
        axis="both",
        alpha=0.25,
        linewidth=0.5,
        color="#888888",
        linestyle="-",
    )
    ax.set_axisbelow(True)

    # X-axis labels (min/max only)
    ax.set_xticks([xlim_neg, xlim_pos])

    if max(abs(xlim_neg), abs(xlim_pos)) >= 100:
        max_val = max(abs(xlim_neg), abs(xlim_pos))
        exponent = int(np.floor(np.log10(max_val)))
        scale_factor = 10**exponent
        ax.set_xticklabels(
            [f"{xlim_neg / scale_factor:.1f}", f"{xlim_pos / scale_factor:.1f}"]
        )
        xlabel_text = f"DOS (states/eV/U.C.) ×10$^{{{exponent}}}$"
    else:
        ax.set_xticklabels([f"{xlim_neg:.1f}", f"{xlim_pos:.1f}"])
        xlabel_text = r"DOS (states/eV/U.C.)"

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel(xlabel_text, fontsize=16, weight="bold")

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.tick_params(which="major", length=8, width=1.2)
    ax.tick_params(which="minor", length=5, width=1.0)

    color_map, linestyle_map = generate_color_scheme(labels, search_ordered)

    # Plot Total DOS as background (z=1)
    for i in range(1, len(data_vect)):
        if labels[i] == "Total":
            ax.fill_betweenx(
                data_vect[0],
                data_vect[i],
                label=labels[i],
                alpha=0.5,
                color="darkgray",
                zorder=1,
            )

    # Plot DOS lines in z-order
    base_zorder = 2
    for z_idx, proj in enumerate(search_ordered):
        for i in range(1, len(data_vect)):
            if labels[i] == proj:
                color = color_map.get(proj, (0, 0, 0))
                linestyle = linestyle_map.get(proj, "-")
                z = base_zorder + z_idx
                ax.plot(
                    data_vect[i],
                    data_vect[0],
                    label=labels[i],
                    linewidth=1.8,
                    alpha=0.85,
                    color=color,
                    linestyle=linestyle,
                    zorder=z,
                )

    # Fermi level line
    fermi_line = plt.axhline(
        maxV,
        color="black",
        linestyle="-.",
        lw=1.5,
        alpha=0.8,
        label="$E_F$",
        zorder=100,
    )
    plt.axvline(0, color="black", lw=2.5, alpha=0.9, zorder=101)

    # Composition labels - positioned near band edges with white boxes
    if show_composition and composition_info:
        y_bottom = E_l + maxV
        y_top = E_u + maxV
        y_range = y_top - y_bottom

        # Check if conductive (single E_F label) or semiconducting (VB/CB labels)
        if composition_info.get("is_conductive") and composition_info.get("fermi"):
            # Conductive material - single label at Fermi level
            fermi_y_axes = (maxV - y_bottom) / y_range
            fermi_y_axes = max(0.05, min(0.95, fermi_y_axes))

            ax.text(
                0.02,
                fermi_y_axes,
                f"$E_F$: {composition_info['fermi']}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="center",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                    linewidth=0.8,
                ),
                zorder=102,
            )
        else:
            # Semiconducting - separate VB and CB labels
            # VB label - position just above VBM
            if composition_info.get("vb"):
                if composition_info["vbm"] is not None:
                    vb_y_pos = composition_info["vbm"] + 0.15
                else:
                    vb_y_pos = maxV - 0.2

                vb_y_axes = (vb_y_pos - y_bottom) / y_range
                vb_y_axes = max(0.05, min(0.45, vb_y_axes))

                ax.text(
                    0.02,
                    vb_y_axes,
                    f"VB: {composition_info['vb']}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="gray",
                        linewidth=0.8,
                    ),
                    zorder=102,
                )

            # CB label - position just below CBM
            if composition_info.get("cb"):
                if composition_info["cbm"] is not None:
                    cb_y_pos = composition_info["cbm"] - 0.15
                else:
                    cb_y_pos = maxV + 0.2

                cb_y_axes = (cb_y_pos - y_bottom) / y_range
                cb_y_axes = max(0.55, min(0.95, cb_y_axes))

                ax.text(
                    0.02,
                    cb_y_axes,
                    f"CB: {composition_info['cb']}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="gray",
                        linewidth=0.8,
                    ),
                    zorder=102,
                )

    # Smart legend placement
    vbm = composition_info["vbm"] if composition_info else None
    cbm = composition_info["cbm"] if composition_info else None

    legend_loc = find_optimal_legend_position(
        ax,
        data_vect,
        labels,
        search_ordered,
        xlim_neg,
        xlim_pos,
        E_l,
        E_u,
        maxV,
        vbm,
        cbm,
    )
    print(f"  Legend placed at: {legend_loc}")

    legend = plt.legend(
        loc=legend_loc,
        frameon=True,
        fontsize=9,
        ncol=1,
        columnspacing=0.5,
        handlelength=1.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.9)

    # Band gap text box - position at same height as legend (upper area)
    if band_gap_info:
        if "semimetal" in band_gap_info:
            gap_text = f"Semimetal\n$E_g$ = {band_gap_info['semimetal']:.2f} eV"
        elif "alpha" in band_gap_info and "beta" in band_gap_info:
            gap_text = f"$E_g^\\alpha$ = {band_gap_info['alpha']:.2f} eV\n$E_g^\\beta$ = {band_gap_info['beta']:.2f} eV"
        elif "gap" in band_gap_info:
            if band_gap_info["gap"] == 0.0:
                gap_text = "Conductive"
            else:
                gap_text = f"$E_g$ = {band_gap_info['gap']:.2f} eV"
        else:
            gap_text = None

        if gap_text:
            # Position based on legend location
            if "left" in legend_loc:
                # Legend is on left, put Eg on right
                eg_x, eg_ha = 0.98, "right"
            else:
                # Legend is on right, put Eg on left (but not too far left)
                eg_x, eg_ha = 0.55, "left"

            ax.text(
                eg_x,
                0.98,
                gap_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment=eg_ha,
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"
                ),
                zorder=103,
            )

    # Save figures
    format_list = formats.lower().split(',')
    format_list = [f.strip() for f in format_list]

    saved_files = []
    for fmt in format_list:
        if fmt in ['svg', 'png', 'pdf']:
            output_path = os.path.join(output_dir, f"{material}.DOSS.{fmt}")
            if fmt == 'png':
                fig.savefig(output_path, format=fmt, dpi=600)
            else:
                fig.savefig(output_path, format=fmt)
            saved_files.append(f"{material}.DOSS.{fmt}")

    print(f"  Saved: {', '.join(saved_files)}")
    plt.close("all")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for DOS plotting."""
    if len(sys.argv) < 4:
        print(
            "Usage: python ipDOS_V7.py [E_lower] [E_upper] [projection_type] [options]"
        )
        print("")
        print("Required:")
        print("  E_lower          : Lower energy limit (eV)")
        print("  E_upper          : Upper energy limit (eV)")
        print("  projection_type  : 'total', 'orbital', or 'both'")
        print("")
        print("Optional:")
        print(
            "  x_scale_mode     : 'auto', 'upper_half', 'lower_half', 'fermi:N', etc."
        )
        print(
            "                     - 'upper_half': Scale x-axis based on DOS above E_F only"
        )
        print(
            "                     - 'lower_half': Scale x-axis based on DOS below E_F only"
        )
        print("                     - 'auto': Scale to full energy range (default)")
        print("  element_mode     : 'all', 'light_total', 'tm_both', 'tm_orb', etc.")
        print("")
        print("Flags:")
        print("  --no-composition     : Disable VB/CB composition labels")
        print(
            "  --vb-range N         : Energy range below VBM for VB composition (default: 0.5 eV)"
        )
        print(
            "  --cb-range N         : Energy range above CBM for CB composition (default: 0.5 eV)"
        )
        print("  --composition-range N: Set both VB and CB range to N (shorthand)")
        print("  --fixed-height N     : Figure height in inches (default: 8)")
        print("  --formats FMT        : Output formats (comma-separated): svg, png, pdf. Default: png,svg")
        sys.exit(1)

    E_l = float(sys.argv[1])
    E_u = float(sys.argv[2])
    proj_type = sys.argv[3].lower()

    if proj_type not in ["total", "orbital", "both"]:
        print("Error: projection_type must be 'total', 'orbital', or 'both'")
        sys.exit(1)

    x_scale_mode = "auto"
    element_mode = "all"

    positional_args = []
    flag_args = {}

    i = 4
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--"):
            if arg == "--no-composition":
                flag_args["no_composition"] = True
            elif arg == "--vb-range" and i + 1 < len(sys.argv):
                flag_args["vb_range"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--cb-range" and i + 1 < len(sys.argv):
                flag_args["cb_range"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--composition-range" and i + 1 < len(sys.argv):
                # Shorthand to set both
                val = float(sys.argv[i + 1])
                flag_args["vb_range"] = val
                flag_args["cb_range"] = val
                i += 1
            elif arg == "--fixed-height" and i + 1 < len(sys.argv):
                flag_args["fixed_height"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--formats" and i + 1 < len(sys.argv):
                flag_args["formats"] = sys.argv[i + 1]
                i += 1
        else:
            positional_args.append(arg)
        i += 1

    if len(positional_args) > 0:
        x_scale_mode = positional_args[0].lower()
    if len(positional_args) > 1:
        element_mode = positional_args[1].lower()

    show_composition = not flag_args.get("no_composition", False)
    vb_range = flag_args.get("vb_range", 0.5)
    cb_range = flag_args.get("cb_range", 0.5)
    fixed_height = flag_args.get("fixed_height", DEFAULT_FIG_HEIGHT)
    formats = flag_args.get("formats", "png,svg")

    DIR = os.getcwd() + "/"
    FIGDIR = DIR

    pathlist = glob.glob(DIR + "*_doss.DOSS.DAT")
    nDIR = len(DIR)
    ntype = len("_doss.DOSS.DAT")

    if not pathlist:
        print(f"No *_doss.DOSS.DAT files found in {DIR}")
        sys.exit(1)

    for path in pathlist:
        path_in_str = str(path)
        material = path_in_str[nDIR:-ntype]
        if material == "":
            break
        print(f"Processing {material}...")
        try:
            ipDOS(
                material,
                E_l,
                E_u,
                proj_type,
                x_scale_mode,
                element_mode,
                show_composition,
                vb_range,
                cb_range,
                fixed_height,
                formats,
                FIGDIR,
            )
        except Exception as e:
            print(f"  Error processing {material}: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
