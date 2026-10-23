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
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory
from matplotlib import patheffects
import matplotlib as mpl
import glob
from os.path import exists
from collections import defaultdict

# =============================================================================
# SHARED CONFIGURATION (Keep consistent with ipBANDS_V7.py)
# =============================================================================

# 1.1.6 (T8): a figure for a page column, 3.5 x 5 in, no title, 11 pt labels and 10 pt ticks
DEFAULT_FIG_WIDTH = 3.5
DEFAULT_FIG_HEIGHT = 5
FIG_WIDTH = DEFAULT_FIG_WIDTH   # --fixed-width (mace plotting --width)
PNG_DPI = 600                   # --dpi (mace plotting --dpi)
# the GUI's plot colors (tokens.css at a light plot): the grid (--line), the Fermi line (--text-2),
# the k-point lines (--border-strong)
GRID_COLOR = "#e3e6eb"
FERMI_COLOR = "#545b67"
KTICK_COLOR = "#848b97"

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

def _labels():
    """mace/utils/plot_labels.py (the plot labels shared with mace gui and the plotly figures)."""
    try:
        from mace.utils import plot_labels
    except ImportError:          # run as a script from Plotting/: the repository root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mace.utils import plot_labels
    return plot_labels


def _presets():
    """mace/plotting/presets.py (the plot styles and DOS colors shared with mace gui)."""
    try:
        from mace.plotting import presets
    except ImportError:          # run as a script from Plotting/: the repository root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mace.plotting import presets
    return presets


def _plot_tokens(style):
    """The GUI's plot colors (ink, secondary text, borders) of a style on a light plot (plotly_bands.tokens)."""
    try:
        from mace.plotting.plotly_bands import tokens
    except ImportError:          # run as a script from Plotting/
        _presets()
        from mace.plotting.plotly_bands import tokens
    return tokens(style)


def _mpl_color(color):
    """A presets color (#rrggbb, or rgba(r,g,b,a) from presets.rgba) as matplotlib takes it."""
    if isinstance(color, str) and color.startswith("rgba("):
        r, g, b, a = (float(x) for x in color[5:-1].split(","))
        return (r / 255, g / 255, b / 255, a)
    return color


def _group_titles(legend, rows, color):
    """Legend rows ``rows`` as group titles (matplotlib has no legend groups): no handle space, the
    title color, flush left with the entries' handles."""
    texts = legend.get_texts()
    packers = [p for col in legend._legend_handle_box.get_children() for p in col.get_children()]
    for i in rows:
        texts[i].set_color(color)
        if i < len(packers):
            box = packers[i].get_children()[0]
            box.width = 0
            packers[i].sep = 0


def projection_styles(labels, search, dos_colors=None, style="default", element_colors=None, show_total=True,
                      spin=False, weights=None):
    """{label: {"name", "color", "dash", "show", "total"}} for the projections of a DOS (mace gui's
    DOS colors, presets.py dos_styles). ``dos_colors`` (auto, total, element, orbital,
    element-orbital) picks the lines and their colors; without it the projection type picked the
    lines (``search``) and each takes the color it has in mace gui: an element's all-orbital line
    its element color, an orbital line its orbital's hue (one element) or a shade of its element's
    color (several), s/p/d/f also by line style."""
    P = _presets()
    colors = element_colors if element_colors is not None else P.element_colors()
    if dos_colors:
        look = P.dos_styles(labels, dos_colors, colors, preset=style, show_total=show_total, spin=spin, weights=weights)["lines"]
        return {x["label"]: x for x in look}
    elements = {P.parse_projection(lab)["element"] for lab in search} - {None}
    orb = P.dos_styles(labels, "orbital" if len(elements) <= 1 else "element-orbital", colors, preset=style,
                       spin=spin, weights=weights)["lines"]
    whole = P.dos_styles(labels, "element", colors, preset=style, spin=spin, weights=weights)["lines"]
    out = {}
    for o, w in zip(orb, whole):
        x = dict(w if P.parse_projection(o["label"])["orbital"] in ("all", None, "total") else o)
        x["show"] = x["total"] or o["label"] in search
        out[o["label"]] = x
    return out


# =============================================================================
# ELEMENT COLORS (legacy table: the element groups of element_mode; the line colors are presets.py's)
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
            if not elem:            # an unnamed column: no element to name ("Fe(d), , O(p)")
                continue
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
            if not elem:            # an unnamed column: no element to name ("Fe(d), , O(p)")
                continue
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


def legend_outside(fig, ax, *args, **kwargs):
    """The shared outside-the-data legend (mace/plotting/legend_layout.py)."""
    try:
        from mace.plotting.legend_layout import legend_outside as place
    except ImportError:          # run as a script from Plotting/: the repository root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mace.plotting.legend_layout import legend_outside as place
    return place(fig, ax, *args, **kwargs)


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


def read_doss_projection_labels(d3_file):
    """The projection labels of a DOSS .d3 ("#C S", ..., "END" for the total).

    The DOSS block is the header (NPRO NPT INZB IFNB IPLO NPOL NPR), a BMI BMA record when
    INZB and IFNB are negative (an energy window), NPRO projection records, then NPR print
    records ("105 -1": MACE's decks write one) and END. Only the projections are labels; the
    total, the last DOSS.DAT column, is "END" (doss_label: "Total")."""
    with open(d3_file) as F:
        lines = F.read().splitlines()
    n = next((i for i, line in enumerate(lines) if line.startswith("DOSS")), None)
    if n is None or n + 1 >= len(lines):
        return []
    head = lines[n + 1].split()
    try:
        num = int(head[0])
    except (ValueError, IndexError):
        return []
    try:
        window = int(head[2]) < 0 and int(head[3]) < 0
    except (ValueError, IndexError):
        window = True        # the layout MACE's decks always have
    start = n + 3 if window else n + 2
    v = []
    for line in lines[start:start + num]:
        if line.startswith("END"):
            break
        v.append(str(line[max(line.find("#"), 0):].strip()))
    return v + ["END"]


def doss_label(raw):
    """A .d3 projection label as the plot shows it: "#C P" -> "C (p)",
    "#MG all" -> "Mg ", "END" -> "Total"."""
    j = raw.replace(" all", " ")
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
    return l


def vacuum_shift(potc_dat, potc_out):
    """maxV (eV): the shift that puts energies on the vacuum scale, from a
    POTC run's planar potential and Fermi energy; 0 without both files."""
    if not (exists(potc_dat) and exists(potc_out)):
        return 0
    z = []
    V = []
    EF = 0
    with open(potc_out) as f5:
        for i, line in enumerate(f5):
            if "FERMI ENERGY" in line:
                EF = float(line.split()[-1])
    with open(potc_dat) as f:
        for i, line in enumerate(f):
            if line.startswith("#") or line.startswith("@"):
                continue
            else:
                z.append(float(line.split()[0]))
                V.append(float(line.split()[1]))
    return -(V[0] - EF) * 27.2114


def energy_axis_label(vacuum, shift=0.0):
    """The energy axis title, as the plotly figures' and the GUI's (plot_labels.energy_title) in
    mathtext: E − E_F (eV), the vacuum scale, or either with a Fermi shift, E − (E_F + 0.50 eV) (eV)."""
    L = _labels()
    return L.mathtext(L.with_unit(L.energy_title("vacuum" if vacuum else "fermi", shift), "eV"))


def read_doss_header(doss_file):
    """(NPROJ or None, NSPIN) from the DOSS.DAT header line."""
    with open(doss_file) as fh:
        head = fh.readline().split()
    try:
        nproj = int(head[head.index("NPROJ") + 1])
    except (ValueError, IndexError):
        nproj = None
    try:
        nspin = int(head[head.index("NSPIN") + 1])
    except (ValueError, IndexError):
        nspin = 1
    return nproj, nspin


def suppress_edge_pileup(data_vect, n_blocks=1, ratio=3.0):
    """Zero the bin at either end of the DOS window that holds a pile-up.

    When the DOSS energy window cuts through bands, CRYSTAL writes a zero at
    each end of the window and puts a large value in the second record
    (graphite: 133 states/eV in one bin, under 2 everywhere else), which is
    not a density at that energy. Drawn, it is a long horizontal line at the
    window edge that sets the DOS axis scale. A bin next to an end point is a
    pile-up when it is ``ratio`` times its inner neighbor and above every
    value inside the window; it is set to zero in every column of every spin
    block. Returns (columns, number of bins zeroed per block).
    """
    cols = [np.asarray(c, dtype=float) for c in data_vect]
    n = len(cols[0]) if cols else 0
    if n_blocks < 1 or n % n_blocks:
        n_blocks = 1
    m = n // n_blocks if n_blocks else 0
    if len(cols) < 2 or m < 8:
        return [c.tolist() for c in cols], 0
    mag = np.max(np.abs(np.vstack(cols[1:])), axis=0).reshape(n_blocks, m)
    zero = set()
    for b in mag:
        inner = b[3:-3].max()
        if b[1] > ratio * b[2] and b[1] > inner:
            zero.add(1)
        if b[-2] > ratio * b[-3] and b[-2] > inner:
            zero.add(m - 2)
    for k in zero:
        for blk in range(n_blocks):
            for c in cols[1:]:
                c[blk * m + k] = 0.0
    return [c.tolist() for c in cols], len(zero)


def read_doss_columns(doss_file, n_columns, maxV=0):
    """DOSS.DAT columns: [energies (eV, + maxV), then each projection in
    states/eV]. Every data record of every spin block is read, in file order."""
    data_vect = [[] for n in range(n_columns)]
    nepts, nspin = read_doss_points(doss_file)
    with open(doss_file) as f:
        for i, line in enumerate(f):
            if line.startswith("# EFERMI"):
                continue
            if line.startswith("#") or line.startswith("@") or line.startswith("&"):
                continue

            data = line.split()

            # A DOS record can wrap across several physical lines when there are
            # many projection columns; a complete record has n_columns values
            # (energy + projections). Pull continuation lines until the record is
            # complete. The previous code unconditionally read ONE line ahead
            # whenever n_columns >= 17 and discarded it unless it looked like a
            # short continuation — which silently dropped every second row of any
            # 17+-column DOS file that was written on single (unwrapped) lines.
            while len(data) < n_columns:
                ne = next(f, "")
                if not ne:
                    break
                ne_parts = ne.split()
                if not ne_parts:
                    continue
                data += ne_parts
            if len(data) < n_columns:
                break                   # a record cut off at the end of a short file

            data_vect[0].append(float(data[0]) * 27.2114 + maxV)

            for j in range(1, n_columns):
                data_vect[j].append(float(data[j]) / 27.2114)
    # a short file (a copy still in progress, a full quota) is refused: half of
    # spin up would be drawn as spin down, in a wrong window
    if nepts and len(data_vect[0]) < nepts * nspin:
        raise ValueError(
            f"{os.path.basename(doss_file)} is truncated: {len(data_vect[0])} of {nepts * nspin} energy points "
            f"(NEPTS {nepts}, NSPIN {nspin} in its header); copy the whole file again")
    return data_vect


def read_doss_points(doss_file):
    """(NEPTS or None, NSPIN) from the DOSS.DAT header line: the energy points
    of each spin block."""
    with open(doss_file) as fh:
        head = fh.readline().split()
    try:
        nepts = int(head[head.index("NEPTS") + 1])
    except (ValueError, IndexError):
        nepts = None
    try:
        nspin = int(head[head.index("NSPIN") + 1])
    except (ValueError, IndexError):
        nspin = 1
    return nepts, nspin


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
    fermi_shift=0.0,
    energy_reference="auto",
    style="default",
    dos_colors=None,
    show_total=True,
    spin_dos="auto",
    spin_dash=None,
    element_colors=None,
    spin_up_color=None,
    spin_down_color=None,
    band_color=None,
    band_style=None,
    dos_smooth=None,
):
    """Main DOS plotting function. ``style``: the plot preset (presets.py); ``dos_colors``: the
    projections and colors (auto, total, element, orbital, element-orbital; None: the projection
    type's lines in mace gui's colors); ``show_total``: draw the total; ``spin_dos``: auto (a
    spin-polarized run mirrored, a closed-shell run one-sided), mirrored or overlaid;
    ``spin_dash``: spin down's line style; ``spin_up_color``/``spin_down_color``: the total's spin
    pair (default: the style's). ``dos_smooth``: the Gaussian smoothing width
    (eV, display only; None or 0: off). ``energy_reference``: auto (the vacuum scale
    when the run's POTC files are beside the data, else E - E_F), fermi (E - E_F
    always) or vacuum (needs the POTC files). ``fermi_shift`` (eV) moves the
    states by -fermi_shift: the dashed line and the window stay, so the energy
    fermi_shift above E_F becomes the reference (the GUI's Fermi shift)."""

    P = _presets()
    preset = P.get(style, "band")
    total_fill = preset["dos"]["total_fill"]
    file = material + "_doss.DOSS.DAT"
    file1 = material + "_doss.d3"
    file4 = material + "_POTC.POTC.DAT"
    file5 = material + "_POTC.out"

    v = read_doss_projection_labels(file1)
    labels = ["Energy (eV)"]

    element_treatment = parse_element_mode(element_mode)

    search = []
    for i in v:
        l = doss_label(i)
        labels.append(l)

        if should_include_orbital(l, element_treatment, proj_type):
            search.append(l)

    search = sort_labels_by_element_and_orbital(search)

    # Handle vacuum reference
    energy_reference = (energy_reference or "auto").lower()
    if energy_reference == "vacuum" and not (exists(file4) and exists(file5)):
        raise ValueError("--energy-reference vacuum needs the run's POTC files beside the data "
                         f"({file4} and {file5})")
    maxV = 0 if energy_reference == "fermi" else vacuum_shift(file4, file5)
    vacuum_axis = energy_reference != "fermi" and exists(file4)

    # Read DOS data
    data_vect = read_doss_columns(file, len(labels), maxV)
    if fermi_shift:
        data_vect[0] = [e - fermi_shift for e in data_vect[0]]
    data_vect, n_pileup = suppress_edge_pileup(data_vect, read_doss_header(file)[1])
    if n_pileup:
        print("  Left out the pile-up bin at the edge of the DOS window (states outside the window)")

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
            xlimit = max(abs(xlim_neg), xlim_pos)
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

    # the two spin blocks (CRYSTAL writes both, the second negative): a closed-shell run is drawn
    # one-sided, a spin-polarized one mirrored, unless spin_dos asks otherwise (B2, the GUI's rule)
    n_all = len(data_vect[0])
    nspin = read_doss_header(file)[1]
    half = n_all // 2 if nspin == 2 and n_all % 2 == 0 else n_all
    up_cols = [np.asarray(c[:half], float) for c in data_vect[1:]]
    down_cols = [np.asarray(c[half:], float) for c in data_vect[1:]] if half < n_all else None
    polarized = P.dos_spin_polarized(up_cols, down_cols)
    if spin_dos == "mirrored" and down_cols is not None:
        side = "mirrored"
    elif spin_dos == "overlaid" and polarized:
        side = "overlaid"
    else:
        side = "mirrored" if (spin_dos == "auto" and polarized) else "one"
    energy = np.asarray(data_vect[0][:half], float)
    col_of = {lab: k for k, lab in enumerate(labels[1:])}
    both = side != "one"
    # an unrestricted run with alpha = beta (MACE's SPIN default): each block holds one spin, so the
    # one-sided curve is their sum alpha + |beta|, the total DOS (the GUI's results.dos_data spin_summed)
    summed = down_cols is not None and not polarized and side == "one"
    if summed:
        up_cols = [u - d for u, d in zip(up_cols, down_cols)]
    if polarized:
        state = "spin-polarized"
    elif down_cols is not None:
        state = "spin-polarized, alpha = beta (non-magnetic)" + (", total alpha + beta drawn" if summed else "")
    else:
        state = "closed shell"
    print(f"  Spin: {state}, drawn {side if side != 'one' else 'one-sided'}")

    # Create figure with fixed subplot parameters for alignment (no title: the caption names it)
    fig = plt.figure(figsize=(FIG_WIDTH, fixed_height))
    ax = fig.add_axes(
        [
            SUBPLOT_LEFT,
            SUBPLOT_BOTTOM,
            SUBPLOT_RIGHT - SUBPLOT_LEFT,
            SUBPLOT_TOP - SUBPLOT_BOTTOM,
        ]
    )
    label_pt, tick_pt = preset["font"], preset["tick_font"]

    # Set black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2 * preset["frame"])

    ax.yaxis.set_major_locator(MultipleLocator(1.0 if (E_u - E_l) <= 14 else 2.0))
    ax.yaxis.set_minor_locator(FixedLocator([]))

    ax.set_ylabel(energy_axis_label(vacuum_axis, fermi_shift), fontsize=label_pt)
    y_lo, y_hi = E_l + maxV, E_u + maxV
    ax.set_ylim(y_lo, y_hi)

    # Grid: the GUI's light major grid, no minor grid (a style without one: Publication, Print grayscale)
    if preset["grid"] != "none":
        ax.grid(True, which="major", axis="both", linewidth=0.8, color=GRID_COLOR, linestyle="-")
        ax.grid(False, which="minor")
    else:
        ax.grid(False)
    ax.set_axisbelow(True)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel(r"DOS (states/eV)", fontsize=label_pt)

    plt.xticks(fontsize=tick_pt)
    plt.yticks(fontsize=tick_pt)

    tick_dir = "out" if preset["ticks"] == "outside" else "in"
    ax.tick_params(which="major", length=5, width=1.0 * preset["frame"], direction=tick_dir)
    ax.tick_params(which="minor", length=3, width=0.8 * preset["frame"], direction=tick_dir)

    # each projection's color and line style: the DOS colors of mace gui (presets.py dos_styles); the
    # projections drawn are --dos-colors' (when given) or the projection type's
    weights = P.dos_weights(list(energy), [{"label": lab, "up": list(up_cols[k]),
                                            "down": list(down_cols[k]) if (polarized and down_cols is not None) else None}
                                           for k, lab in enumerate(labels[1:])], y_lo, y_hi)
    styles = projection_styles(labels[1:], search_ordered, dos_colors, style, element_colors, show_total,
                               spin=both, weights=weights)
    spin_pair = P.spin_colors(style, "band", "light", None, spin_up_color, spin_down_color)
    down_dash = spin_dash or P.SPIN_DOWN_DASH
    tk = _plot_tokens(style)
    ink = tk["--plot-ink"] or tk["--text"]
    # the one-sided total of a run that is not spin-polarized, by the Band style as the GUI draws it
    # (plotly_bands.look_parts; a DOS alone has no bands to call a metal, so auto is single): single the
    # pair's first color (or --band-color), overlay the same, ink the style's neutral color
    bstyle = P.band_style(band_style, None)
    if bstyle == "ink":
        neutral = P.band_color(style, "band", "light", None, ink)
    elif bstyle == "single" and band_color and P._HEXCOLOR.match(band_color):
        neutral = band_color.lower()
    else:
        neutral = spin_pair["up"]
    if dos_colors:
        # the 1.1.6 layout (the GUI's): by default only the 2-4 projections carrying most of the DOS in the
        # window; DOS colors Auto takes the style's projection colors, largest first, clear of the spin pair
        pk = P.pick_projections(list(styles.values()), weights)
        order = pk["order"]
        for lab in order:
            if lab not in pk["shown"]:
                styles[lab] = {**styles[lab], "show": False}
        if dos_colors == "auto":
            ppal = P.projection_palette(style, [spin_pair["up"], spin_pair["down"]] if both else None)
            if ppal:
                for k, lab in enumerate(order):
                    styles[lab] = {**styles[lab], "color": ppal[k % len(ppal)], "dash": "solid"}
        hidden = [lab for lab in order if lab not in pk["shown"]]
        if hidden:
            print(f"  Projections drawn: {', '.join(styles[lab]['name'] for lab in pk['shown'])} "
                  f"(the largest in the window; left out: {', '.join(styles[lab]['name'] for lab in hidden)})")
        search_ordered = [lab for lab in calculate_dos_order(data_vect, labels, [x["label"] for x in styles.values()
                          if x["show"] and not x["total"]], data_vect[0], E_l, E_u, maxV)]
    else:
        order = sorted(search_ordered, key=lambda lab: -float(weights.get(lab) or 0.0))
    # opt-in Gaussian smoothing (display only, --dos-smooth / [plots] dos_smooth)
    sigma = P.smooth_sigma(dos_smooth)
    energy_list = list(energy)

    def sm(y):
        return np.asarray(P.gauss_smooth(list(y), energy_list, sigma)) if sigma else np.asarray(y)

    fills = P.spin_fills(spin_pair, style)
    nfill = P.neutral_fill(neutral, style)
    # the GUI's widths: the projections 1.3, the total 1.7 (never thinner than the projections)
    lw = 1.3 * preset["line"]
    curves = []

    def draw(k, color, dash, z, label=None, total=False):
        """projection column k: spin up on the right, spin down (dashed, its own color) on the left."""
        width = 1.7 * preset["line"] if total else lw
        up = sm(up_cols[k])
        if total:
            # the total: lightly filled in spin up's color (both spins), else a neutral fill under an ink line
            ax.fill_betweenx(energy, 0, up, color=_mpl_color(fills["up"] if both else nfill), linewidth=0, zorder=1)
        line_color = (spin_pair["up"] if both else neutral) if total else color
        ax.plot(up, energy, label=label, linewidth=width, color=line_color, linestyle=P.mpl_dash(dash), zorder=z)
        curves.append(up)
        if not both or down_cols is None:
            return
        down = sm(down_cols[k] if polarized else -up_cols[k])
        if side == "overlaid":
            down = np.abs(down)
        elif total:
            ax.fill_betweenx(energy, 0, down, color=_mpl_color(fills["down"]), linewidth=0, zorder=1)
        ax.plot(down, energy, linewidth=width, color=spin_pair["down"] if total else color,
                linestyle=P.mpl_dash(down_dash if dash == "solid" else dash),
                alpha=0.65 if (side == "overlaid" and dash != "solid") else 1.0, zorder=z)
        curves.append(down)

    total_style = styles.get("Total")
    total_drawn = "Total" in col_of and (dos_colors == "total" or show_total) and total_style is not None
    if total_drawn:
        draw(col_of["Total"], total_style["color"], "solid", 50, label="Total", total=True)

    # Plot DOS lines in z-order (the largest DOS behind)
    base_zorder = 2
    rank = {proj: z for z, proj in enumerate(search_ordered)}
    drawn = []
    for proj in sorted(search_ordered, key=lambda x: col_of.get(x, 0)):
        st = styles.get(proj)
        if proj not in col_of or st is None or st["total"] or not st.get("show", True):
            continue
        draw(col_of[proj], st["color"], st["dash"], base_zorder + rank[proj], label=st["name"])
        drawn.append(proj)

    # the DOS range: the drawn curves in the scaling window, symmetric when mirrored
    if x_scale_mode.startswith("custom:"):
        xlim = (xlim_neg if side == "mirrored" else 0.0, xlim_pos)
    else:
        emask = (energy >= scale_e_min) & (energy <= scale_e_max)
        if not np.any(emask):
            emask = np.ones(len(energy), dtype=bool)
        vals = [np.nanmax(np.abs(c[emask])) for c in curves if np.any(np.isfinite(c[emask]))]
        m = max(vals) * 1.08 if vals and max(vals) > 0 else 1.0
        xlim = (-m, m) if side == "mirrored" else (0.0, m)
    ax.set_xlim(*xlim)
    print(f"  X-axis limits: {xlim[0]:.4f} to {xlim[1]:.4f}")
    if side == "mirrored" and xlim[0] < 0 < xlim[1]:
        # magnitudes on both sides of zero
        ticks = P.mirrored_ticks(xlim[0], xlim[1])
        ax.set_xticks(ticks["vals"])
        ax.set_xticklabels(ticks["text"], fontsize=tick_pt)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_minor_locator(FixedLocator([]))

    # a clear DOS = 0 baseline (the spin halves meet there)
    ax.axvline(0, color=ink, lw=1.2, zorder=1.5)
    # "↑ up" and "down ↓" over each half, beside the zero line, in the spin colors
    if side == "mirrored" and xlim[0] < 0 < xlim[1]:
        over = blended_transform_factory(ax.transData, ax.transAxes)
        for text, color, ha, dx in (("↑ up", spin_pair["up"], "left", 3), ("down ↓", spin_pair["down"], "right", -3)):
            ax.annotate(text, xy=(0, 1), xycoords=over, xytext=(dx, 2), textcoords="offset points", ha=ha, va="bottom",
                        color=color, fontweight="bold", fontsize=max(7, tick_pt - 2))

    # Fermi level line: dotted (dashed would read as spin down)
    fermi_label = _labels().mathtext(_labels().fermi_key(fermi_shift))
    plt.axhline(maxV, color=FERMI_COLOR, linestyle=P.mpl_dash("dot"), lw=1.2, label=fermi_label, zorder=100)

    # the energies the DOS run did not compute: shaded, labeled when there is room (no data, not zero states)
    across = blended_transform_factory(ax.transAxes, ax.transData)
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]     # a note readable over the grid
    finite_e = energy[np.isfinite(energy)]
    window = [float(finite_e.min()), float(finite_e.max())] if len(finite_e) else None
    for g0, g1 in P.dos_window_gaps(window, y_lo, y_hi):
        ax.axhspan(g0, g1, color=_mpl_color(P.rgba(ink, 0.05)), linewidth=0, zorder=0.4)
        if g1 - g0 >= 0.06 * (y_hi - y_lo):
            low = g0 <= y_lo
            ax.text(0.5, g1 if low else g0, P.DOS_LAYOUT["no_data"], transform=across, ha="center",
                    va="top" if low else "bottom", fontsize=max(6, tick_pt - 4), color=tk["--text-3"], zorder=0.6,
                    path_effects=halo)
    if sigma:
        ax.text(0.98, 0.01, f"Gaussian smoothing σ = {P.js_num(sigma)} eV (display only)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=max(6, tick_pt - 4), color=tk["--text-2"], zorder=101,
                path_effects=halo)

    # Composition notes (the orbitals at the band edges): with the band-gap note they head
    # the legend, outside the axes, not boxes over the curves
    comp_lines = []
    if show_composition and composition_info:
        if composition_info.get("is_conductive") and composition_info.get("fermi"):
            comp_lines.append(f"{_labels().mathtext(_labels().SYMBOLS['E_F'])}: {composition_info['fermi']}")
        else:
            if composition_info.get("cb"):
                comp_lines.append(f"CB: {composition_info['cb']}")
            if composition_info.get("vb"):
                comp_lines.append(f"VB: {composition_info['vb']}")

    # Band gap note: it heads the legend, outside the axes with it
    gap_text = None
    if band_gap_info:
        if "semimetal" in band_gap_info:
            gap_text = f"Semimetal\nE$_{{\\mathregular{{g}}}}$ = {band_gap_info['semimetal']:.2f} eV"
        elif "alpha" in band_gap_info and "beta" in band_gap_info:
            gap_text = f"E$_{{\\mathregular{{g}}}}^{{\\alpha}}$ = {band_gap_info['alpha']:.2f} eV\nE$_{{\\mathregular{{g}}}}^{{\\beta}}$ = {band_gap_info['beta']:.2f} eV"
        elif "gap" in band_gap_info:
            if band_gap_info["gap"] == 0.0:
                gap_text = "Conductive"
            else:
                gap_text = f"E$_{{\\mathregular{{g}}}}$ = {band_gap_info['gap']:.2f} eV"

    # the grouped legend (the GUI's Spin and Projections groups; matplotlib has no group titles, so each
    # group opens with a title row): Spin in neutral ink, the projections as color swatches, then E_F
    from matplotlib.lines import Line2D
    handles, names, heads = [], [], []

    def head(text):
        heads.append(len(names))
        handles.append(Line2D([], [], linestyle="none", marker=None))
        names.append(text)

    head("Spin")
    if both:
        handles.append(Line2D([], [], color=ink, lw=2))
        names.append("spin up (α)")
        handles.append(Line2D([], [], color=ink, lw=2, linestyle=P.mpl_dash(down_dash)))
        names.append("spin down (β)")
    else:
        handles.append(Line2D([], [], color=neutral, lw=2))
        names.append("α = β (closed shell)")
    keyed = (["Total"] if total_drawn else []) + [lab for lab in order if lab in drawn]
    if keyed:
        head("Projections")
    for lab in keyed:
        st = styles[lab]
        if lab == "Total":
            face = _mpl_color(P.rgba(ink, 0.12) if both else P.rgba(neutral, 0.22))
            edge = tk["--border-strong"] if both else neutral
        else:
            face = edge = st["color"]
        handles.append(Line2D([], [], linestyle="none", marker="s", markersize=8, markerfacecolor=face,
                              markeredgecolor=edge, markeredgewidth=1.0))
        names.append(st["name"])
    handles.append(Line2D([], [], color=FERMI_COLOR, lw=1.2, linestyle=P.mpl_dash("dot")))
    names.append(fermi_label)

    # the legend outside the axes and the tick labels and axis label on their right:
    # a busy DOS has no empty corner for it (matplotlib figures never put a legend over the data)
    legend = legend_outside(
        fig,
        ax,
        handles,
        names,
        frameon=True,
        fontsize=max(8, tick_pt - 1),
        ncol=1,
        columnspacing=0.5,
        handlelength=1.8,
        title="\n".join(t for t in (gap_text, *comp_lines) if t) or None,
        title_fontsize=max(8, tick_pt - 1),
    )
    _group_titles(legend, heads, tk["--text-2"])
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.9)

    # Save figures
    format_list = formats.lower().split(',')
    format_list = [f.strip() for f in format_list]

    saved_files = []
    for fmt in format_list:
        if fmt in ['svg', 'png', 'pdf']:
            output_path = os.path.join(output_dir, f"{material}.DOSS.{fmt}")
            if fmt == 'png':
                fig.savefig(output_path, format=fmt, dpi=PNG_DPI)
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
        print("  --fixed-height N     : Figure height in inches (default: 5)")
        print("  --fixed-width N      : Figure width in inches (default: 3.5)")
        print("  --dpi N              : PNG resolution (default: 600)")
        print("  --formats FMT        : Output formats (comma-separated): svg, png, pdf. Default: png,svg")
        print("  --fermi-shift EV     : Move the states by -EV: the energy EV above E_F becomes the reference")
        print("  --energy-reference R : auto (vacuum scale with the run's POTC files, default), fermi or vacuum")
        print("  --style S            : default, publication, presentation, grayscale or colorblind")
        print("  --dos-colors M       : auto, total, element, orbital or element-orbital (lines and colors)")
        print("  --no-total           : without the total DOS")
        print("  --spin-dos M         : auto (spin-polarized mirrored, closed shell one-sided), mirrored or overlaid")
        print("  --spin-style D       : spin down's line style: dash (default), dot, dashdot or solid")
        print("  --spin-up-color HEX  : the total's spin up color (default: the style's spin colors)")
        print("  --spin-down-color HEX: the total's spin down color")
        print("  --dos-smooth EV      : Gaussian smoothing width (display only; default: off)")
        print("  --output-dir DIR     : write the figures here (default: beside the data)")
        print("  --file DOSS.DAT      : plot this file only (default: every *_doss.DOSS.DAT in the folder)")
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
            elif arg == "--fixed-width" and i + 1 < len(sys.argv):
                flag_args["fixed_width"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--dpi" and i + 1 < len(sys.argv):
                flag_args["dpi"] = int(sys.argv[i + 1])
                i += 1
            elif arg == "--formats" and i + 1 < len(sys.argv):
                flag_args["formats"] = sys.argv[i + 1]
                i += 1
            elif arg == "--fermi-shift" and i + 1 < len(sys.argv):
                flag_args["fermi_shift"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--dos-smooth" and i + 1 < len(sys.argv):
                flag_args["dos_smooth"] = float(sys.argv[i + 1])
                i += 1
            elif arg == "--no-total":
                flag_args["show_total"] = False
            elif arg in ("--style", "--dos-colors", "--spin-dos", "--spin-style", "--output-dir", "--file",
                         "--spin-up-color", "--spin-down-color", "--band-color", "--band-style") and i + 1 < len(sys.argv):
                flag_args[arg[2:].replace("-", "_")] = sys.argv[i + 1]
                i += 1
            elif arg == "--energy-reference" and i + 1 < len(sys.argv):
                flag_args["energy_reference"] = sys.argv[i + 1].lower()
                if flag_args["energy_reference"] not in ("auto", "fermi", "vacuum"):
                    print("Error: --energy-reference is auto, fermi or vacuum")
                    sys.exit(1)
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
    global FIG_WIDTH, PNG_DPI
    FIG_WIDTH = flag_args.get("fixed_width", DEFAULT_FIG_WIDTH)
    PNG_DPI = flag_args.get("dpi", 600)
    formats = flag_args.get("formats", "png,svg")

    DIR = os.getcwd() + "/"
    FIGDIR = flag_args.get("output_dir") or DIR
    if flag_args.get("output_dir"):
        os.makedirs(FIGDIR, exist_ok=True)
    choices = {"style": list(_presets().ORDER), "dos_colors": list(_presets().DOS_MODES),
               "spin_dos": ["auto", "mirrored", "overlaid"], "spin_style": ["dash", "dot", "dashdot", "solid"],
               "band_style": list(_presets().BAND_STYLES)}
    for k, ok in choices.items():
        if k in flag_args and flag_args[k] not in ok:
            print(f"Error: --{k.replace('_', '-')} is one of {', '.join(ok)}")
            sys.exit(1)

    pathlist = glob.glob(DIR + "*_doss.DOSS.DAT")
    if flag_args.get("file"):
        pathlist = [p for p in pathlist if os.path.basename(p) == os.path.basename(flag_args["file"])]
    nDIR = len(DIR)
    ntype = len("_doss.DOSS.DAT")

    if not pathlist:
        print(f"No *_doss.DOSS.DAT files found in {DIR}")
        sys.exit(1)

    failed = []
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
                fermi_shift=flag_args.get("fermi_shift", 0.0),
                style=flag_args.get("style", "default"),
                dos_colors=flag_args.get("dos_colors"),
                show_total=flag_args.get("show_total", True),
                spin_dos=flag_args.get("spin_dos", "auto"),
                spin_dash=flag_args.get("spin_style"),
                energy_reference=flag_args.get("energy_reference", "auto"),
                spin_up_color=flag_args.get("spin_up_color"),
                spin_down_color=flag_args.get("spin_down_color"),
                band_color=flag_args.get("band_color"),
                band_style=flag_args.get("band_style"),
                dos_smooth=flag_args.get("dos_smooth"),
            )
        except Exception as e:
            print(f"  Error processing {material}: {e}")
            import traceback

            traceback.print_exc()
            failed.append(material)
            continue
    if failed:      # a failed file is a failed run (it once exited 0 and the caller drew nothing)
        print(f"Error: {len(failed)} of {len(pathlist)} DOS file(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
