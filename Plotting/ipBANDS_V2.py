#!/usr/bin/python3
"""
================================================================================
Band Structure Plotting Script for CRYSTAL17/23 Output Files (Version 7)
================================================================================

FIXES IN V7:
- Uses actual matplotlib alpha for transparency stacking
- Overlapping bands become more opaque; isolated bands stay transparent

================================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import glob
from os.path import exists
import re

# =============================================================================
# SHARED CONFIGURATION (Keep consistent with ipDOS_V7.py)
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

# Default spin colors
DEFAULT_SPIN_UP_COLOR = "#fa26a0"  # Pink/magenta
DEFAULT_SPIN_DOWN_COLOR = "#2ff3e0"  # Cyan


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def adjust_color_for_alpha(color, alpha):
    """
    Adjust color for transparency - only shift cyan hue to prevent purple.
    Keep original saturation/brightness so stacking approaches ORIGINAL color intensity.

    Args:
        color: hex string or RGB tuple
        alpha: transparency value (0-1)

    Returns:
        Adjusted hex color string
    """
    import colorsys

    if alpha >= 0.95:
        return color  # No adjustment needed

    # Convert to RGB
    if isinstance(color, str):
        rgb = hex_to_rgb(color)
    else:
        rgb = color

    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    # ONLY change: shift cyan hue toward blue to prevent purple appearance
    # Keep saturation and value the SAME so stacking approaches original color
    if 0.40 < h < 0.56:  # Cyan/teal range
        h = 0.62  # Shift to blue

    # Keep original saturation and brightness!
    # This way, full stacking = original color intensity
    new_s = s
    new_v = v

    # Convert back to RGB
    new_rgb = colorsys.hsv_to_rgb(h, new_s, new_v)

    # Convert to hex
    return "#{:02x}{:02x}{:02x}".format(
        int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255)
    )


class BandSegment:
    """Represents a continuous segment of the k-point path"""

    def __init__(self, index, labels, x_start, x_end, data_start_idx, data_end_idx):
        self.index = index
        self.labels = labels
        self.x_start = x_start
        self.x_end = x_end
        self.data_start_idx = data_start_idx
        self.data_end_idx = data_end_idx

    def __repr__(self):
        label_str = " → ".join(self.labels)
        return f"Segment {self.index}: {label_str} (x: {self.x_start:.3f} - {self.x_end:.3f})"


GREEK_LABELS = {
    "GAMMA": "Γ",
    "SIGMA": "Σ",
    "LAMBDA": "Λ",
    "DELTA": "Δ",
}


def _greekify_single(label):
    """Convert a single k-point label to use Greek symbols.

    Converts full Greek names like GAMMA, SIGMA_0 -> Γ, Σ_0.

    Note: We do NOT convert standalone "G" to "Γ" because in body-centered
    tetragonal (tI2) and base-centered monoclinic (mC2) lattices, "G" is a
    distinct parametric k-point (not the zone center). The zone center is
    labeled "GAMMA" to avoid this ambiguity.
    """
    for name, symbol in GREEK_LABELS.items():
        if label.startswith(name):
            label = label.replace(name, symbol, 1)
            break
    return label


def _format_mathtext(label):
    """Format a single label with mathtext subscripts and primes.
    E.g. Σ_0 -> $Σ_{0}$, S' -> $S'$, Y'_2 -> $Y'_{2}$"""
    if "_" not in label and "'" not in label:
        return label
    prime = "'" if "'" in label else ""
    label_no_prime = label.replace("'", "")
    if "_" in label_no_prime:
        base, sub = label_no_prime.split("_", 1)
        return rf"${base}{prime}_{{{sub}}}$"
    else:
        return rf"${label_no_prime}{prime}$"


def greekify_label(label):
    """Convert k-point label(s) to Greek symbols with mathtext formatting.
    Handles pipe-separated merged labels like 'X|G' -> 'X|Γ'."""
    if "|" in label:
        parts = label.split("|")
        return "|".join(_format_mathtext(_greekify_single(p)) for p in parts)
    return _format_mathtext(_greekify_single(label))


def parse_kpoint_path_and_segments(d3_file):
    """Read k-point path labels from .d3 file and identify continuous segments."""
    try:
        with open(d3_file, "r") as f:
            next(f)
            path_line = next(f).strip()

            parts = path_line.split(" - ")
            if len(parts) > 0:
                kpoint_sequence = parts[-1].strip()

                segments_raw = kpoint_sequence.split("|")
                segments_labels = []

                for segment in segments_raw:
                    labels = [s.strip() for s in segment.split("-") if s.strip()]
                    if labels:
                        labels = [greekify_label(l) for l in labels]
                        segments_labels.append(labels)

                all_labels = []
                clean_labels = []

                for i, seg_labels in enumerate(segments_labels):
                    if i == 0:
                        all_labels.extend(seg_labels)
                        clean_labels.extend(seg_labels)
                    else:
                        if all_labels and all_labels[-1] == seg_labels[0]:
                            # Continuous: previous segment ends where this one starts
                            all_labels.extend(seg_labels[1:])
                        elif all_labels:
                            all_labels[-1] = f"{all_labels[-1]}|{seg_labels[0]}"
                            all_labels.extend(seg_labels[1:])
                        else:
                            all_labels.extend(seg_labels)
                        clean_labels.extend(seg_labels)

                return all_labels, segments_labels, clean_labels
            else:
                return None, None, None

    except Exception as e:
        print(f"Error reading k-point path from {d3_file}: {str(e)}")
        return None, None, None


def identify_segments_from_data(
    segments_labels, x_labels, E, BANDS, kpoint_indices=None
):
    """Create BandSegment objects using header k-point positions."""
    segments = []
    n_segments = len(segments_labels)
    total_kpts = len(E)

    edges_per_seg = [len(labels) - 1 for labels in segments_labels]
    cumulative = [0]
    for n in edges_per_seg:
        cumulative.append(cumulative[-1] + n)

    print(f"  Edges per segment: {edges_per_seg}")

    if kpoint_indices and len(kpoint_indices) > 0:
        kpt_indices = [idx for idx, label in kpoint_indices]
        n_header_positions = len(kpt_indices)

        for seg_idx in range(n_segments):
            seg_labels = segments_labels[seg_idx]
            start_header_pos = cumulative[seg_idx]

            if start_header_pos >= n_header_positions:
                break

            data_start_kpt = kpt_indices[start_header_pos]
            data_start_idx = data_start_kpt - 1

            data_end_kpt = total_kpts
            if seg_idx < n_segments - 1:
                next_start_header_pos = cumulative[seg_idx + 1]
                if next_start_header_pos < n_header_positions:
                    next_start_kpt = kpt_indices[next_start_header_pos]
                    data_end_kpt = next_start_kpt - 1

            data_end_idx = data_end_kpt - 1
            data_start_idx = max(0, min(data_start_idx, total_kpts - 1))
            data_end_idx = max(data_start_idx, min(data_end_idx, total_kpts - 1))

            x_start = E[data_start_idx]
            x_end = E[data_end_idx]

            segment = BandSegment(
                index=seg_idx + 1,
                labels=seg_labels,
                x_start=x_start,
                x_end=x_end,
                data_start_idx=data_start_idx,
                data_end_idx=data_end_idx,
            )
            segments.append(segment)

    else:
        for seg_idx in range(n_segments):
            seg_labels = segments_labels[seg_idx]
            start_xlabel_idx = cumulative[seg_idx]
            end_xlabel_idx = cumulative[seg_idx + 1]

            if end_xlabel_idx >= len(x_labels):
                break

            x_start = x_labels[start_xlabel_idx]
            x_end = x_labels[end_xlabel_idx]

            data_mask = (E >= x_start - 1e-6) & (E <= x_end + 1e-6)
            data_indices = np.where(data_mask)[0]

            if len(data_indices) > 0:
                data_start_idx = data_indices[0]
                data_end_idx = data_indices[-1]
            else:
                data_start_idx = 0
                data_end_idx = 0

            segment = BandSegment(
                index=seg_idx + 1,
                labels=seg_labels,
                x_start=x_start,
                x_end=x_end,
                data_start_idx=data_start_idx,
                data_end_idx=data_end_idx,
            )
            segments.append(segment)

    return segments


def match_path_to_segments(path_str, segments):
    """Match a user-specified path string to available segments."""
    path_specs = [p.strip() for p in path_str.split(",")]
    matched_indices = []

    for spec in path_specs:
        spec_labels = [greekify_label(l.strip()) for l in spec.split("-") if l.strip()]
        if not spec_labels:
            continue

        for seg in segments:
            if is_subsequence(spec_labels, seg.labels) or spec_labels == seg.labels:
                if seg.index not in matched_indices:
                    matched_indices.append(seg.index)
                break
        else:
            for seg in segments:
                if spec_labels[0] in seg.labels and spec_labels[-1] in seg.labels:
                    if seg.index not in matched_indices:
                        matched_indices.append(seg.index)
                    break

    return sorted(matched_indices)


def is_subsequence(small, large):
    """Check if small list is a contiguous subsequence of large list"""
    if len(small) > len(large):
        return False
    for i in range(len(large) - len(small) + 1):
        if large[i : i + len(small)] == small:
            return True
    return False


def find_band_files(directory):
    """Find all BAND.DAT files with case-insensitive matching"""
    band_files = []
    all_files = os.listdir(directory)
    pattern = re.compile(r".*[._]band\.band\.dat$", re.IGNORECASE)

    for filename in all_files:
        if pattern.match(filename):
            band_files.append(os.path.join(directory, filename))

    return band_files


def extract_material_name(filepath):
    """Extract material name from the file path"""
    basename = os.path.basename(filepath)
    pattern = re.compile(r"([._]band\.band\.dat)$", re.IGNORECASE)
    match = pattern.search(basename)

    if match:
        suffix = match.group(1)
        material = basename[: -len(suffix)]
        return material, suffix

    return None, None


def find_associated_files(material, suffix, directory):
    """Find all associated files for a given material"""
    files = {}

    prefix_match = re.match(r"([._]band)", suffix, re.IGNORECASE)
    if prefix_match:
        if "_band" in suffix.lower():
            d3_suffix = "_band.d3"
        else:
            d3_suffix = ".d3"
    else:
        d3_suffix = ".d3"

    d3_file = os.path.join(directory, material + d3_suffix)
    if os.path.exists(d3_file):
        files["d3"] = d3_file

    potc_dat = os.path.join(directory, material + "_POTC.POTC.DAT")
    if os.path.exists(potc_dat):
        files["potc_dat"] = potc_dat

    potc_out = os.path.join(directory, material + "_POTC.out")
    if os.path.exists(potc_out):
        files["potc_out"] = potc_out

    return files


def read_band_data(band_file, potc_dat=None, potc_out=None):
    """Read band structure data from CRYSTAL output files."""
    maxV = 0
    if potc_dat and potc_out and os.path.exists(potc_dat) and os.path.exists(potc_out):
        z = []
        V = []
        EF = 0
        with open(potc_out) as f5:
            for line in f5:
                if "FERMI ENERGY" in line:
                    EF = float(line.split()[-1])
        with open(potc_dat) as f:
            for line in f:
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    z.append(float(line.split()[0]))
                    V.append(float(line.split()[1]))
        maxV = -(V[0] - EF) * 27.2114

    x_labels = []
    kpoint_indices = []
    n_panels = 0

    with open(band_file) as fb:
        header = fb.readline()
        N = int(header.split()[2])
        M = int(header.split()[4])
        E = np.zeros(N)
        Ebeta = np.zeros(N)
        BANDS = np.zeros((N, M))
        BANDSbeta = np.zeros((N, M))
        i = 0
        ib = 0
        alpha_beta_counter = 0

        l_labels = 999
        l_label = 999

        for l, line in enumerate(fb):
            if line.startswith("# NPANEL"):
                n_panels = int(line.split()[-1])
            elif line.startswith("#") and "(" in line and ")" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        kpt_idx = int(parts[1])
                        kpt_label = parts[2] if len(parts) > 2 else ""
                        kpoint_indices.append((kpt_idx, kpt_label))
                    except ValueError:
                        pass
            if line.startswith("@ XAXIS TICK SPEC"):
                n_labels = int(line.split()[-1])
                l_label = l
                l_labels = l + 1
            if l == l_labels:
                x_labels.append(float(line.split()[-1]))
                l_labels += 2
                if l_labels > l_label + 2 * n_labels - 1:
                    l_labels = 1e30
            if line.startswith("# EFERMI (HARTREE)"):
                alpha_beta_counter += 1
            if line.startswith("#") or line.startswith("@"):
                continue
            if alpha_beta_counter == 0:
                data = line.split()
                if M > 1000:
                    while len(data) != M + 1:
                        data = np.concatenate([data, next(fb).split()])
                E[i] = float(data[0])
                for j in range(0, M):
                    BANDS[i, j] = (float(data[j + 1]) * 27.2114) + maxV
                i = i + 1
            if alpha_beta_counter == 1:
                data = line.split()
                if M > 1000:
                    while len(data) != M + 1:
                        data = np.concatenate([data, next(fb).split()])
                Ebeta[ib] = float(data[0])
                for j in range(0, M):
                    BANDSbeta[ib, j] = (float(data[j + 1]) * 27.2114) + maxV
                ib = ib + 1
            if alpha_beta_counter == 2:
                break

    return (
        E,
        BANDS,
        Ebeta,
        BANDSbeta,
        np.array(x_labels),
        maxV,
        kpoint_indices,
        n_panels,
    )


def find_band_gap_region(BANDS, BANDSbeta, maxV, E_l, E_u):
    """
    Find if there's a band gap in the visible energy range.
    Returns (gap_bottom, gap_top) if gap exists, else (None, None).
    """
    y_bottom = E_l + maxV
    y_top = E_u + maxV

    all_energies_up = BANDS.flatten()
    all_energies_up = all_energies_up[
        (all_energies_up >= y_bottom) & (all_energies_up <= y_top)
    ]

    if np.any(BANDSbeta != 0):
        all_energies_down = BANDSbeta.flatten()
        all_energies_down = all_energies_down[
            (all_energies_down >= y_bottom) & (all_energies_down <= y_top)
        ]
        all_energies = np.concatenate([all_energies_up, all_energies_down])
    else:
        all_energies = all_energies_up

    if len(all_energies) == 0:
        return None, None

    below_fermi = all_energies[all_energies < maxV]
    above_fermi = all_energies[all_energies > maxV]

    if len(below_fermi) == 0 or len(above_fermi) == 0:
        return None, None

    vbm = np.max(below_fermi)
    cbm = np.min(above_fermi)

    gap_size = cbm - vbm

    if gap_size > 0.3:
        return vbm, cbm

    return None, None


def find_optimal_legend_position(E, BANDS, Ebeta, BANDSbeta, maxV, E_l, E_u, x_max):
    """
    Find the optimal legend position, prioritizing band gap region.
    """
    y_bottom = E_l + maxV
    y_top = E_u + maxV
    y_range = y_top - y_bottom
    x_min = 0
    x_range = x_max - x_min

    # Check for band gap
    vbm, cbm = find_band_gap_region(BANDS, BANDSbeta, maxV, E_l, E_u)

    if vbm is not None and cbm is not None:
        gap_center = (vbm + cbm) / 2
        gap_size = cbm - vbm

        if y_bottom < gap_center < y_top and gap_size > 0.5:
            # Good gap exists - place legend there
            # Prefer right side of plot
            return "center right"

    # Otherwise, check corner densities
    corner_fraction = 0.25

    corners = {
        "upper right": (
            x_max - corner_fraction * x_range,
            x_max,
            y_top - corner_fraction * y_range,
            y_top,
        ),
        "upper left": (
            x_min,
            x_min + corner_fraction * x_range,
            y_top - corner_fraction * y_range,
            y_top,
        ),
        "lower right": (
            x_max - corner_fraction * x_range,
            x_max,
            y_bottom,
            y_bottom + corner_fraction * y_range,
        ),
        "lower left": (
            x_min,
            x_min + corner_fraction * x_range,
            y_bottom,
            y_bottom + corner_fraction * y_range,
        ),
    }

    corner_density = {}

    for corner_name, (x1, x2, y1, y2) in corners.items():
        count = 0

        for j in range(BANDS.shape[1]):
            in_x = (E >= x1) & (E <= x2)
            in_y = (BANDS[:, j] >= y1) & (BANDS[:, j] <= y2)
            count += np.sum(in_x & in_y)

        if np.any(BANDSbeta != 0):
            for j in range(BANDSbeta.shape[1]):
                in_x = (Ebeta >= x1) & (Ebeta <= x2)
                in_y = (BANDSbeta[:, j] >= y1) & (BANDSbeta[:, j] <= y2)
                count += np.sum(in_x & in_y)

        corner_density[corner_name] = count

    best_corner = min(corner_density, key=corner_density.get)
    return best_corner


def plot_bands_with_gaps(
    E,
    BANDS,
    Ebeta,
    BANDSbeta,
    segments,
    selected_indices,
    x_labels,
    maxV,
    E_l,
    E_u,
    gap_width,
    output_dir,
    material,
    has_potc,
    use_gaps=True,
    all_segments=None,
    fixed_width=None,
    fixed_height=None,
    kpoint_indices=None,
    alpha=1.0,
    spin_up_color=None,
    spin_down_color=None,
    auto_width=False,
    formats="png,svg",
):
    """Plot band structure with visual gaps at discontinuities and transparency support."""

    if spin_up_color is None:
        spin_up_color = DEFAULT_SPIN_UP_COLOR
    if spin_down_color is None:
        spin_down_color = DEFAULT_SPIN_DOWN_COLOR

    # For low alpha, only adjust hue (cyan->blue) but use ACTUAL alpha for stacking
    if alpha < 0.95:
        plot_up_color = adjust_color_for_alpha(spin_up_color, alpha)
        plot_down_color = adjust_color_for_alpha(spin_down_color, alpha)
        plot_alpha = alpha  # USE THE ACTUAL ALPHA for stacking effect!
        print(f"  Transparency mode: alpha={alpha}")
        print(f"    Spin up: {spin_up_color} -> {plot_up_color}")
        print(f"    Spin down: {spin_down_color} -> {plot_down_color}")
    else:
        plot_up_color = spin_up_color
        plot_down_color = spin_down_color
        plot_alpha = 1.0

    if all_segments is None:
        all_segments = segments

    if selected_indices:
        plot_segments = [s for s in segments if s.index in selected_indices]
    else:
        plot_segments = segments

    if not plot_segments:
        print("No segments to plot!")
        return

    total_data_width = sum(s.x_end - s.x_start for s in plot_segments)
    n_gaps = len(plot_segments) - 1 if use_gaps else 0
    gap_size = gap_width * total_data_width if use_gaps else 0

    fig_height = fixed_height if fixed_height else DEFAULT_FIG_HEIGHT

    if auto_width:
        # Count unique tick positions: each segment contributes its labels,
        # but continuous boundaries share a label
        n_ticks = len(plot_segments[0].labels) if plot_segments else 0
        for i in range(1, len(plot_segments)):
            prev_end = plot_segments[i - 1].labels[-1]
            curr_start = plot_segments[i].labels[0]
            if prev_end == curr_start:
                # Continuous: shared label, only add the new ones
                n_ticks += len(plot_segments[i].labels) - 1
            else:
                # Discontinuous: both labels shown (possibly as X|Y)
                n_ticks += len(plot_segments[i].labels) - 1
        min_width_per_tick = 0.7  # inches per tick label
        fig_width = max(DEFAULT_FIG_WIDTH, n_ticks * min_width_per_tick)
        print(f"  Auto-width: {n_ticks} tick labels -> {fig_width:.1f} inches")
    else:
        fig_width = fixed_width if fixed_width else DEFAULT_FIG_WIDTH

    # Create figure with fixed subplot parameters for alignment
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            SUBPLOT_LEFT,
            SUBPLOT_BOTTOM,
            SUBPLOT_RIGHT - SUBPLOT_LEFT,
            SUBPLOT_TOP - SUBPLOT_BOTTOM,
        ]
    )

    ax.set_title("Band Structure", pad=10, fontsize=18, weight="bold")

    # Set black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    all_tick_positions = []
    all_tick_labels = []
    gap_positions = []
    current_x_offset = 0

    for seg_idx, segment in enumerate(plot_segments):
        idx_start = segment.data_start_idx
        idx_end = segment.data_end_idx + 1

        print(f"  Plotting segment {segment.index}: {' → '.join(segment.labels)}")

        seg_E = E[idx_start:idx_end]
        seg_BANDS = BANDS[idx_start:idx_end, :]
        seg_Ebeta = Ebeta[idx_start:idx_end]
        seg_BANDSbeta = BANDSbeta[idx_start:idx_end, :]

        x_shift = current_x_offset - segment.x_start
        shifted_E = seg_E + x_shift
        shifted_Ebeta = seg_Ebeta + x_shift

        # Plot bands with actual alpha for stacking effect
        # Spin up at lower z-order, spin down at higher z-order
        # This prevents pink+blue blending into purple - they layer instead
        for j in range(seg_BANDS.shape[1]):
            ax.plot(
                shifted_E,
                seg_BANDS[:, j],
                linewidth=2.0,
                color=plot_up_color,
                alpha=plot_alpha,
                zorder=2,  # Spin up below
            )
            ax.plot(
                shifted_Ebeta,
                seg_BANDSbeta[:, j],
                linewidth=2.0,
                linestyle="--",
                color=plot_down_color,
                dashes=(5, 2),
                alpha=plot_alpha,
                zorder=3,  # Spin down on top - no blending with pink
            )

        # Tick positions
        n_labels = len(segment.labels)
        seg_width = segment.x_end - segment.x_start

        seg_tick_positions = []
        seg_tick_labels = []

        for i, label in enumerate(segment.labels):
            if n_labels > 1:
                frac = i / (n_labels - 1)
            else:
                frac = 0.5
            tick_pos = current_x_offset + frac * seg_width
            seg_tick_positions.append(tick_pos)
            seg_tick_labels.append(label)

        # Determine if this segment is continuous with the previous one
        # (i.e. previous segment's last label matches this segment's first label)
        is_continuous = False
        if seg_idx > 0 and len(seg_tick_labels) > 0:
            prev_seg = plot_segments[seg_idx - 1]
            if prev_seg.labels[-1] == segment.labels[0]:
                is_continuous = True

        if seg_idx == 0:
            for tick_pos, label in zip(seg_tick_positions, seg_tick_labels):
                all_tick_positions.append(tick_pos)
                all_tick_labels.append(label)
                ax.axvline(tick_pos, color="gray", lw=1.0, alpha=0.4, zorder=0)
        elif is_continuous:
            # Continuous: just merge the shared boundary point, no pipe separator
            for tick_pos, label in zip(
                seg_tick_positions[1:], seg_tick_labels[1:]
            ):
                all_tick_positions.append(tick_pos)
                all_tick_labels.append(label)
                ax.axvline(tick_pos, color="gray", lw=1.0, alpha=0.4, zorder=0)
        else:
            if use_gaps:
                for tick_pos, label in zip(seg_tick_positions, seg_tick_labels):
                    all_tick_positions.append(tick_pos)
                    all_tick_labels.append(label)
                    ax.axvline(tick_pos, color="gray", lw=1.0, alpha=0.4, zorder=0)
            else:
                if len(all_tick_labels) > 0 and len(seg_tick_labels) > 0:
                    merged_label = f"{all_tick_labels[-1]}|{seg_tick_labels[0]}"
                    all_tick_labels[-1] = merged_label
                    for tick_pos, label in zip(
                        seg_tick_positions[1:], seg_tick_labels[1:]
                    ):
                        all_tick_positions.append(tick_pos)
                        all_tick_labels.append(label)
                        ax.axvline(tick_pos, color="gray", lw=1.0, alpha=0.4, zorder=0)
                else:
                    for tick_pos, label in zip(seg_tick_positions, seg_tick_labels):
                        all_tick_positions.append(tick_pos)
                        all_tick_labels.append(label)
                        ax.axvline(tick_pos, color="gray", lw=1.0, alpha=0.4, zorder=0)

        current_x_offset += seg_width

        # Only insert a gap at discontinuous boundaries
        if seg_idx < len(plot_segments) - 1 and use_gaps:
            next_seg = plot_segments[seg_idx + 1]
            if segment.labels[-1] != next_seg.labels[0]:
                gap_positions.append((current_x_offset, current_x_offset + gap_size))
                current_x_offset += gap_size

    # Legend entries - use ADJUSTED colors to match what's actually plotted
    dummy_line_up = ax.plot([], [], linewidth=2.0, color=plot_up_color, label="Spin ↑")
    dummy_line_down = ax.plot(
        [],
        [],
        linewidth=2.0,
        linestyle="--",
        color=plot_down_color,
        label="Spin ↓",
        dashes=(5, 2),
    )

    # Fermi level (always opaque)
    fermi_line = ax.axhline(
        maxV,
        color="black",
        linestyle="-.",
        lw=1.5,
        alpha=0.8,
        label=r"$E_F$",
        zorder=100,
    )

    # Set limits
    y_bottom = E_l + maxV
    y_top = E_u + maxV
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(0, current_x_offset)

    # Draw break indicators
    if use_gaps and gap_positions:
        y_range = y_top - y_bottom
        x_range = current_x_offset if current_x_offset > 0 else 1

        slash_height = y_range * 0.02
        slash_width = x_range * 0.012
        slash_spacing = x_range * 0.008

        for gap_start, gap_end in gap_positions:
            gap_center = (gap_start + gap_end) / 2

            for offset in [-slash_spacing / 2, slash_spacing / 2]:
                x_center = gap_center + offset
                ax.plot(
                    [x_center - slash_width / 2, x_center + slash_width / 2],
                    [y_bottom - slash_height / 2, y_bottom + slash_height / 2],
                    color="gray",
                    lw=1.0,
                    zorder=5,
                    clip_on=False,
                )
                ax.plot(
                    [x_center - slash_width / 2, x_center + slash_width / 2],
                    [y_top - slash_height / 2, y_top + slash_height / 2],
                    color="gray",
                    lw=1.0,
                    zorder=5,
                    clip_on=False,
                )

    # Set ticks
    ax.set_xticks(all_tick_positions)
    ax.set_xticklabels(all_tick_labels, fontsize=12)

    # Y-axis formatting
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Grid
    ax.grid(
        True,
        which="major",
        axis="y",
        alpha=0.5,
        linewidth=0.8,
        color="#666666",
        linestyle="-",
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        alpha=0.25,
        linewidth=0.5,
        color="#888888",
        linestyle="-",
    )
    ax.set_axisbelow(True)

    # Labels - BOLD E-E_F (using mathbf for whole expression)
    if has_potc:
        ax.set_ylabel(r"Energy w.r.t. Vacuum (eV)", fontsize=16, weight="bold")
    else:
        ax.set_ylabel(r"$\mathbf{E-E_F}$ (eV)", fontsize=16, weight="bold")

    ax.set_xlabel(r"Wave Vector", fontsize=16, weight="bold")

    plt.yticks(fontsize=14)
    ax.tick_params(which="major", length=8, width=1.2)
    ax.tick_params(which="minor", length=5, width=1.0)

    # Smart legend placement
    legend_loc = find_optimal_legend_position(
        E, BANDS, Ebeta, BANDSbeta, maxV, E_l, E_u, current_x_offset
    )
    print(f"  Legend placed at: {legend_loc}")

    legend = ax.legend(
        [dummy_line_up[0], dummy_line_down[0], fermi_line],
        ["Spin ↑", "Spin ↓", r"$E_F$"],
        loc=legend_loc,
        frameon=True,
        fontsize=9,
        ncol=1,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(0.9)

    # Save
    suffix = ""
    if selected_indices:
        suffix = f"_seg{'_'.join(map(str, selected_indices))}"

    # Parse formats
    format_list = formats.lower().split(',') if isinstance(formats, str) else formats
    format_list = [f.strip() for f in format_list]

    saved_files = []
    for fmt in format_list:
        if fmt in ['svg', 'png', 'pdf']:
            output_path = os.path.join(output_dir, f"{material}{suffix}.BANDS.{fmt}")
            if fmt == 'png':
                fig.savefig(output_path, format=fmt, dpi=600)
            else:
                fig.savefig(output_path, format=fmt)
            saved_files.append(output_path)

    print(f"  Saved: {', '.join(saved_files)}")
    plt.close("all")


def list_segments(segments):
    """Print available segments"""
    print("\nAvailable segments:")
    print("-" * 70)
    for seg in segments:
        label_str = " → ".join(seg.labels)
        print(f"  {seg.index}: {label_str}")
    print("-" * 70)


def ipBANDS(band_file, material, associated_files, args, output_dir):
    """Main function to plot band structure"""

    file3 = associated_files.get("d3")
    file4 = associated_files.get("potc_dat")
    file5 = associated_files.get("potc_out")

    all_labels, segments_labels, clean_labels = None, None, None
    if file3 and os.path.exists(file3):
        all_labels, segments_labels, clean_labels = parse_kpoint_path_and_segments(
            file3
        )
        if all_labels:
            print(f"  Parsed {len(segments_labels)} segments from {file3}")

    if segments_labels is None:
        segments_labels = [["M", "Γ", "K", "A", "Γ", "L", "H", "Γ"]]

    E, BANDS, Ebeta, BANDSbeta, x_labels, maxV, kpoint_indices, n_panels = (
        read_band_data(band_file, file4, file5)
    )

    print(f"  Data shape: {E.shape[0]} k-points, {BANDS.shape[1]} bands")

    segments = identify_segments_from_data(
        segments_labels, x_labels, E, BANDS, kpoint_indices
    )

    if args.list_segments:
        list_segments(segments)
        return

    selected_indices = None

    if args.segments:
        try:
            selected_indices = [int(x.strip()) for x in args.segments.split(",")]
            print(f"  Selected segments: {selected_indices}")
        except ValueError:
            print(f"  Error: Invalid segment indices")
            return

    elif args.path:
        selected_indices = match_path_to_segments(args.path, segments)
        if selected_indices:
            print(f"  Matched path to segments: {selected_indices}")
        else:
            return

    if selected_indices:
        valid_indices = [s.index for s in segments]
        selected_indices = [i for i in selected_indices if i in valid_indices]

    use_gaps = not args.no_gaps

    # Skip the duplicate boundary k-point at the start of each non-first segment
    for seg in segments[1:]:
        seg.data_start_idx += 1
        seg.x_start = E[seg.data_start_idx]

    plot_bands_with_gaps(
        E,
        BANDS,
        Ebeta,
        BANDSbeta,
        segments,
        selected_indices,
        x_labels,
        maxV,
        args.E_lower,
        args.E_upper,
        args.gap_width,
        output_dir,
        material,
        has_potc=(file4 is not None),
        use_gaps=use_gaps,
        all_segments=segments,
        fixed_width=args.fixed_width,
        fixed_height=args.fixed_height,
        kpoint_indices=kpoint_indices,
        alpha=args.alpha,
        spin_up_color=args.spin_up_color,
        spin_down_color=args.spin_down_color,
        auto_width=args.auto_width,
        formats=args.formats,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot band structures from CRYSTAL17/23 output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ipBANDS_V7.py -5 5                     Plot all segments
  python ipBANDS_V7.py -5 5 --alpha 0.3         Plot with 30% transparency (stacks where bands overlap)
  python ipBANDS_V7.py -5 5 --segments 1,2      Plot segments 1 and 2
  python ipBANDS_V7.py -2 5 --segments 1,2 --alpha 0.2 --no-gaps
        """,
    )

    parser.add_argument("E_lower", type=float, help="Lower energy limit (eV)")
    parser.add_argument("E_upper", type=float, help="Upper energy limit (eV)")
    parser.add_argument("--segments", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--list-segments", action="store_true")
    parser.add_argument("--gap-width", type=float, default=0.05)
    parser.add_argument("--no-gaps", action="store_true")
    parser.add_argument(
        "--auto-width",
        action="store_true",
        help="Scale figure width to prevent label overlap when many segments are plotted",
    )
    parser.add_argument("--fixed-width", type=float, default=DEFAULT_FIG_WIDTH)
    parser.add_argument("--fixed-height", type=float, default=DEFAULT_FIG_HEIGHT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Transparency (0-1, default: 1.0). Overlapping bands will stack and appear more opaque.",
    )
    parser.add_argument("--spin-up-color", type=str, default=DEFAULT_SPIN_UP_COLOR)
    parser.add_argument("--spin-down-color", type=str, default=DEFAULT_SPIN_DOWN_COLOR)
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg",
        help="Output formats (comma-separated): svg, png, pdf. Default: png,svg",
    )

    args = parser.parse_args()

    if args.alpha < 0 or args.alpha > 1:
        print("Error: --alpha must be between 0 and 1")
        sys.exit(1)

    DIR = os.getcwd() + "/"
    FIGDIR = DIR

    band_files = find_band_files(DIR)

    if not band_files:
        print(f"No BAND.DAT files found in {DIR}")
        sys.exit(1)

    print(f"Found {len(band_files)} BAND.DAT file(s)")

    for band_file in band_files:
        print(f"\nProcessing {band_file}...")

        material, suffix = extract_material_name(band_file)
        if not material:
            continue

        print(f"  Material: {material}")

        associated_files = find_associated_files(material, suffix, DIR)

        try:
            ipBANDS(band_file, material, associated_files, args, FIGDIR)
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
