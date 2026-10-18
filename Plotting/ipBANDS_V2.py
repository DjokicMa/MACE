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
from matplotlib.ticker import MultipleLocator, NullLocator
import matplotlib as mpl
import glob
from os.path import exists
import re

# =============================================================================
# SHARED CONFIGURATION (Keep consistent with ipDOS_V7.py)
# =============================================================================

# 1.1.6 (T8): a figure for a page column, 3.5 x 5 in, no title, 11 pt labels and 10 pt ticks
DEFAULT_FIG_WIDTH = 3.5
DEFAULT_FIG_HEIGHT = 5
PNG_DPI = 600                   # --dpi (mace plotting --dpi)
# the GUI's plot colors (tokens.css at a light plot): the grid (--line), the Fermi line (--text-2),
# the k-point lines (--border-strong)
GRID_COLOR = "#e3e6eb"
FERMI_COLOR = "#545b67"
KTICK_COLOR = "#848b97"
LABEL_PT = 11
TICK_PT = 10

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

def _presets():
    """mace/plotting/presets.py (the plot styles shared with mace gui)."""
    try:
        from mace.plotting import presets
    except ImportError:          # run as a script from Plotting/: the repository root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mace.plotting import presets
    return presets


# Default spin colors: the one constant in mace/plotting/presets.py (SPIN_COLORS, pending Marcus's
# pick: blue solid / red dashed in the interim); spin down dashed
DEFAULT_SPIN_UP_COLOR = _presets().SPIN_COLORS["light"]["up"]
DEFAULT_SPIN_DOWN_COLOR = _presets().SPIN_COLORS["light"]["down"]
DEFAULT_SPIN_DOWN_DASH = _presets().SPIN_DOWN_DASH


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


def _bare_g_is_gamma(raw_segments, rows):
    """True when a bare "G" of the path names the zone center (B1): MACE's default k-paths write
    G for Γ ("V-Y-G-Z-T-R-G-X-U-G"), while seekpath's write GAMMA and keep G for a point that is
    not the center (tI2, mC2). So: never when the path has GAMMA; else each G's k-point in the
    deck (``rows``: the BAND lines after the title) must be 0 0 0, when the deck says."""
    flat = [lab for seg in raw_segments for lab in seg]
    if "G" not in flat or any(lab.upper() == "GAMMA" for lab in flat):
        return False
    try:
        n = int(rows[0].split()[0])
        lines = [r.split() for r in rows[1:1 + n]]
        pairs = [(seg[i], seg[i + 1]) for seg in raw_segments for i in range(len(seg) - 1)]
        if len(lines) != n or len(pairs) != n or any(len(x) != 6 for x in lines):
            return True                       # symbolic or another layout: no coordinates to check
        pts = [[float(v) for v in x] for x in lines]
    except (ValueError, IndexError):
        return True
    for (a, b), x in zip(pairs, pts):
        if (a == "G" and any(x[:3])) or (b == "G" and any(x[3:])):
            return False
    return True


def parse_kpoint_path_and_segments(d3_file):
    """Read k-point path labels from .d3 file and identify continuous segments.
    A bare G that names the zone center (MACE's default paths) reads Γ, as GAMMA does."""
    try:
        with open(d3_file, "r") as f:
            next(f)
            path_line = next(f).strip()
            rows = [ln for ln in (next(f, "") for _ in range(60)) if ln.strip()]

            parts = path_line.split(" - ")
            if len(parts) > 0:
                kpoint_sequence = parts[-1].strip()

                segments_raw = kpoint_sequence.split("|")
                raw = [[s.strip() for s in segment.split("-") if s.strip()] for segment in segments_raw]
                raw = [labels for labels in raw if labels]
                gamma = _bare_g_is_gamma(raw, rows)
                segments_labels = []

                for labels in raw:
                    labels = [greekify_label("GAMMA" if gamma and l == "G" else l) for l in labels]
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


def _next_fields(fb, band_file, N, nspin):
    """The next line's fields of a k-point record wrapped over lines; a file
    that ends inside the record is truncated."""
    line = next(fb, None)
    if line is None:
        raise ValueError(f"{os.path.basename(band_file)} is truncated: it ends inside a k point "
                         f"(NKPT {N}, NSPIN {nspin or 1} in its header); copy the whole file again")
    return line.split()


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
        try:
            nspin = int(header.split()[6])
        except (IndexError, ValueError):
            nspin = None
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
                # CRYSTAL writes at most 1000 fields per line (k + 999 bands)
                # and continues the k-point on the next line, so a row with
                # NBND >= 1000 is short until stitched.
                while len(data) < M + 1:
                    data = np.concatenate([data, _next_fields(fb, band_file, N, nspin)])
                E[i] = float(data[0])
                for j in range(0, M):
                    BANDS[i, j] = (float(data[j + 1]) * 27.2114) + maxV
                i = i + 1
            if alpha_beta_counter == 1:
                data = line.split()
                # CRYSTAL writes at most 1000 fields per line (k + 999 bands)
                # and continues the k-point on the next line, so a row with
                # NBND >= 1000 is short until stitched.
                while len(data) < M + 1:
                    data = np.concatenate([data, _next_fields(fb, band_file, N, nspin)])
                Ebeta[ib] = float(data[0])
                for j in range(0, M):
                    BANDSbeta[ib, j] = (float(data[j + 1]) * 27.2114) + maxV
                ib = ib + 1
            if alpha_beta_counter == 2:
                break

    # a short file (a copy still in progress, a full quota) is refused: the
    # arrays are sized from the header, so the missing k points would be drawn
    # at k = 0, E = 0 and a missing beta block as a closed-shell run
    want = N * (nspin or 1)
    got = i + (ib if nspin == 2 else 0)
    if i < N or (nspin == 2 and ib < N):
        raise ValueError(
            f"{os.path.basename(band_file)} is truncated: {got} of {want} k points "
            f"(NKPT {N}, NSPIN {nspin or 1} in its header); copy the whole file again")

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


# alpha and beta closer than this everywhere are one set of bands: CRYSTAL
# prints energies to 1e-4 Ha (2.7 meV), so a non-magnetic spin-polarized run
# differs by that much; anything below 10 meV is invisible on a band plot.
SPIN_IDENTICAL_EV = 0.01


def energy_reference(reference, potc_dat, potc_out):
    """The POTC files to read and whether the axis is on the vacuum scale, for
    ``--energy-reference``: "auto" (the default) is the vacuum scale when the
    run's POTC files are beside the data, else E - E_F; "fermi" is E - E_F
    even then; "vacuum" needs both POTC files."""
    reference = (reference or "auto").lower()
    if reference == "fermi":
        return None, None, False
    if reference == "vacuum" and not (potc_dat and potc_out and exists(potc_dat) and exists(potc_out)):
        raise ValueError("--energy-reference vacuum needs the run's POTC files beside the data "
                         "(<material>_POTC.POTC.DAT and <material>_POTC.out)")
    return potc_dat, potc_out, potc_dat is not None


def apply_fermi_shift(BANDS, BANDSbeta, shift):
    """Move the bands by -shift (eV): the dotted Fermi line and the energy window stay
    where they are, so the energy ``shift`` above E_F becomes the reference (the
    GUI's Fermi shift). An all-zero beta block (closed shell) stays zero."""
    if not shift:
        return BANDS, BANDSbeta
    BANDSbeta = np.asarray(BANDSbeta)
    return BANDS - shift, (BANDSbeta - shift if np.any(BANDSbeta != 0) else BANDSbeta)


def energy_axis_label(vacuum, shift=0.0):
    """The energy axis title: E - E_F, the vacuum scale, or either with a Fermi shift."""
    if not shift:
        return r"$E - E_{vac}$ (eV)" if vacuum else r"$E - E_F$ (eV)"
    ref = r"E_{vac}" if vacuum else r"E_F"
    return rf"$E - ({ref}{shift:+g})$ (eV)"


def fermi_legend_label(shift=0.0):
    """The dotted Fermi line's legend entry: E_F, or E_F with the Fermi shift."""
    return r"$E_F$" if not shift else rf"$E_F{shift:+g}$ eV"


def spin_state(BANDS, BANDSbeta, tol=SPIN_IDENTICAL_EV):
    """("closed" | "identical" | "polarized", largest |alpha - beta| in eV).

    "closed": no beta block, or one equal to alpha (CRYSTAL writes both blocks
    for a closed-shell run); "identical": a spin-polarized run whose beta bands
    match alpha within ``tol`` (non-magnetic); "polarized": they differ.
    """
    a = np.asarray(BANDS, dtype=float)
    b = np.asarray(BANDSbeta, dtype=float)
    if b.size == 0 or not np.any(b != 0) or a.shape != b.shape:
        return "closed", 0.0
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if diff == 0.0:
        return "closed", 0.0
    return ("identical" if diff <= tol else "polarized"), diff


def _label_width_in(label, fontsize):
    """Rough printed width (inches) of a tick label, mathtext included."""
    text = re.sub(r"\\[A-Za-z]+", "x", str(label))
    sub = sum(len(m) for m in re.findall(r"_\{([^}]*)\}", text))
    main = len(re.sub(r"_\{[^}]*\}|[${}_^]", "", text))
    return (main * 0.62 + sub * 0.45) * fontsize / 72.0 + 0.05


def stagger_tick_labels(positions, labels, x0, x1, axis_width_in, fontsize=12):
    """Labels too close to their neighbor to fit side by side drop one line
    ("\\n" first) instead of overlapping (seekpath's Y and F0 0.04 1/Å apart)."""
    span = (x1 - x0) or 1.0
    right = [-np.inf, -np.inf]
    out = []
    for pos, label in zip(positions, labels):
        px = (pos - x0) / span * axis_width_in
        half = _label_width_in(label, fontsize) / 2
        gap = 0.04
        if px - half >= right[0] + gap:
            row = 0
        elif px - half >= right[1] + gap:
            row = 1
        else:
            row = 0 if right[0] <= right[1] else 1
        right[row] = px + half
        out.append(("\n" + label) if row else label)
    return out


def _band_path_module():
    d3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Crystal_d3")
    if d3 not in sys.path:
        sys.path.insert(0, d3)
    import d3_band_path
    return d3_band_path


def load_band_path(path):
    """A band_path edit from a JSON file: the list itself, a {"band_path": [...]}
    object, or an opt2d3 configuration that has one."""
    import json
    kp = _band_path_module()
    with open(path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        obj = obj.get("configuration", obj)
        obj = obj.get("band_path") if isinstance(obj, dict) else obj
    if obj is None:
        raise ValueError(f"{path} has no band_path")
    return kp.normalize(obj)


def d3_path_tokens(d3_file):
    """The path of a BAND deck's title as tokens with "|" (GAMMA, M, ..., |, L, ...)."""
    with open(d3_file) as fh:
        next(fh)
        title = next(fh).strip()
    part = title.split(" - ")[-1]
    return [t.strip() for t in part.split("-") if t.strip()]


def edited_band_data(E, BANDS, BANDSbeta, ticks, tokens, band_path):
    """The band data laid out along an edited path (d3_band_path): the computed
    segments' slices in the edit's order, reversed where it says so, hidden ones
    left out. Returns (E, BANDS, BANDSbeta, segments, x_labels): one BandSegment
    per continuous part, x_labels the tick of every point (one tick for the two
    labels of a jump), ready for plot_bands_with_gaps. The GUI's band plot lays
    out the same edit the same way (mace/gui/static/js/docs/plot.js bandView)."""
    kp = _band_path_module()
    computed = kp.from_tokens(tokens)
    ticks = [float(t) for t in ticks]
    if len(computed) != len(ticks) - 1:
        raise ValueError(f"the path names {len(computed)} segments, the data has {len(ticks) - 1}")
    x = np.asarray(E, dtype=float)
    near = lambda t: int(np.argmin(np.abs(x - t)))       # noqa: E731

    def slice_of(n):
        a, b = near(ticks[n]), near(ticks[n + 1])
        if n > 0 and not kp.joined(computed[n - 1], computed[n]):
            a += 1                      # after a jump the segment starts one point past the tick
        return list(range(a, b + 1))

    used = set()
    rows, xs, pieces, xt = [], [], [], []
    prev, x0 = None, 0.0
    for s in kp.visible(band_path):
        a, b = kp.canonical_label(s["from"]), kp.canonical_label(s["to"])
        n = next((i for i, c in enumerate(computed) if i not in used and (c["from"], c["to"]) == (a, b)), None)
        rev = False
        if n is None:
            n = next((i for i, c in enumerate(computed) if i not in used and (c["from"], c["to"]) == (b, a)), None)
            rev = n is not None
        if n is None:
            raise ValueError(f"{a}→{b} is not a segment of these band data")
        used.add(n)
        idx = slice_of(n)[::-1] if rev else slice_of(n)
        xa = x[idx[0]]
        cont = prev is not None and kp.joined(prev, s)
        if not cont:
            pieces.append({"labels": [greekify_label(a)], "start": len(rows)})
            xt.append(x0)
        for k, i in enumerate(idx):
            if cont and k == 0:
                continue
            rows.append(i)
            xs.append(x0 + abs(x[i] - xa))
        x0 += abs(x[idx[-1]] - xa)
        pieces[-1]["labels"].append(greekify_label(b))
        pieces[-1]["end"] = len(rows) - 1
        xt.append(x0)
        prev = s
    E2 = np.array(xs)
    segments = [BandSegment(k + 1, p["labels"], E2[p["start"]], E2[p["end"]], p["start"], p["end"])
                for k, p in enumerate(pieces)]
    # a jump's two labels share one tick: drop the end tick of each part but the last
    ticks2, k = [], 0
    for p in pieces:
        n_lab = len(p["labels"])
        ticks2 += xt[k:k + n_lab] if p is pieces[-1] else xt[k:k + n_lab - 1]
        k += n_lab
    return E2, BANDS[rows], BANDSbeta[rows], segments, np.array(ticks2)


def path_break_indices(E, ticks, labels):
    """Where a discontinuous k-path ("A|L") breaks: indices i with no line
    from point i to point i+1.

    CRYSTAL does not repeat the k-point at a break: the tick's point (A) ends
    one panel and the next point, one step further along x, is already L. Some
    writers repeat the x instead (two points at the tick); the break then goes
    between those two.
    """
    x = np.asarray(E, dtype=float)
    if len(x) < 2:
        return []
    steps = np.abs(np.diff(x))
    steps = steps[steps > 1e-9]
    tol = max(0.5 * float(np.median(steps)) if len(steps) else 0.0, 1e-6)
    out = []
    for tick, label in zip(ticks, labels):
        if "|" not in str(label):
            continue
        i = int(np.argmin(np.abs(x - float(tick))))   # the first of a repeated pair
        if abs(x[i] - float(tick)) <= tol and i + 1 < len(x):
            out.append(i)
    return sorted(set(out))


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


def legend_outside(fig, ax, *args, **kwargs):
    """The shared outside-the-data legend (mace/plotting/legend_layout.py)."""
    try:
        from mace.plotting.legend_layout import legend_outside as place
    except ImportError:          # run as a script from Plotting/: the repository root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mace.plotting.legend_layout import legend_outside as place
    return place(fig, ax, *args, **kwargs)


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
    fermi_shift=0.0,
    style="default",
    spin_dash=None,
    band_color=None,
):
    """Plot band structure with visual gaps at discontinuities and transparency support.
    ``fermi_shift``: the bands are already moved by it (apply_fermi_shift); it
    names the reference in the axis title and the legend. ``style``: the plot
    preset (presets.py: default, publication, presentation, grayscale,
    colorblind); ``spin_dash``: spin down's line style (dash, dot, dashdot, solid)."""

    P = _presets()
    preset = P.get(style, "band")
    pair = P.spin_colors(style, "band")
    if spin_up_color is None:
        spin_up_color = pair["up"]
    if spin_down_color is None:
        spin_down_color = pair["down"]
    down_dash = P.mpl_dash(spin_dash or DEFAULT_SPIN_DOWN_DASH)
    lw = 1.5 * preset["line"]

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

    spin, spin_diff = spin_state(BANDS, BANDSbeta)
    if spin == "identical":
        print(f"  Spin up and down agree within {spin_diff * 1000:.1f} meV (non-magnetic): one set drawn")
    if spin != "polarized":
        # not spin-polarized: one set of bands in a neutral color (--band-color, the style's, else ink)
        neutral = P.band_color(style, "band", "light", band_color, "#000000")
        plot_up_color = adjust_color_for_alpha(neutral, alpha) if alpha < 0.95 else neutral

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

    # Create figure with fixed subplot parameters for alignment (no title: the caption names it)
    fig = plt.figure(figsize=(fig_width, fig_height))
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

    # the BAND.DAT tick of each segment's first label: segments share the tick
    # at a joint (and at a break, where "F0|Δ0" is one tick for two labels)
    file_ticks = [float(t) for t in np.asarray(x_labels, dtype=float).ravel()] if x_labels is not None else []
    half = len(file_ticks) // 2
    if half and len(file_ticks) % 2 == 0 and np.allclose(file_ticks[:half], file_ticks[half:]):
        file_ticks = file_ticks[:half]                     # one copy per spin block
    tick_start, n_edges = {}, 0
    for seg_ in sorted(all_segments, key=lambda seg_: seg_.index):
        tick_start[seg_.index] = n_edges
        n_edges += len(seg_.labels) - 1
    if n_edges + 1 != len(file_ticks):                           # no usable ticks: spread evenly
        tick_start = {}

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
                linewidth=lw,
                color=plot_up_color,
                alpha=plot_alpha,
                zorder=2,  # Spin up below
            )
            # one set when beta repeats alpha: dashed copies of degenerate
            # bands, out of phase, read as one solid line over spin up
            if spin != "polarized":
                continue
            ax.plot(
                shifted_Ebeta,
                seg_BANDSbeta[:, j],
                linewidth=lw,
                linestyle=down_dash,
                color=plot_down_color,
                alpha=plot_alpha,
                zorder=3,  # Spin down on top - no blending with pink
            )

        # Tick positions
        n_labels = len(segment.labels)
        seg_width = segment.x_end - segment.x_start

        seg_tick_positions = []
        seg_tick_labels = []

        # each label at its own k (the BAND.DAT tick), not spread evenly over
        # the segment: Γ-Y-F0 with |Y-F0| 14x shorter than |Γ-Y| put Y mid-way
        first_tick = tick_start.get(segment.index)
        for i, label in enumerate(segment.labels):
            if first_tick is not None:
                xl = file_ticks[first_tick + i] - segment.x_start
                tick_pos = current_x_offset + min(max(xl, 0.0), seg_width)
            elif n_labels > 1:
                tick_pos = current_x_offset + i / (n_labels - 1) * seg_width
            else:
                tick_pos = current_x_offset + 0.5 * seg_width
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
                ax.axvline(tick_pos, color=KTICK_COLOR, lw=0.8, zorder=0)
        elif is_continuous:
            # Continuous: just merge the shared boundary point, no pipe separator
            for tick_pos, label in zip(
                seg_tick_positions[1:], seg_tick_labels[1:]
            ):
                all_tick_positions.append(tick_pos)
                all_tick_labels.append(label)
                ax.axvline(tick_pos, color=KTICK_COLOR, lw=0.8, zorder=0)
        else:
            if use_gaps and all_tick_labels and seg_tick_labels:
                # one tick for both labels of the jump, "H|P" over the gap (the GUI's merged tick)
                ax.axvline(seg_tick_positions[0], color=KTICK_COLOR, lw=0.8, zorder=0)
                all_tick_positions[-1] = (all_tick_positions[-1] + seg_tick_positions[0]) / 2
                all_tick_labels[-1] = f"{all_tick_labels[-1]}|{seg_tick_labels[0]}"
                for tick_pos, label in zip(seg_tick_positions[1:], seg_tick_labels[1:]):
                    all_tick_positions.append(tick_pos)
                    all_tick_labels.append(label)
                    ax.axvline(tick_pos, color=KTICK_COLOR, lw=0.8, zorder=0)
            else:
                if len(all_tick_labels) > 0 and len(seg_tick_labels) > 0:
                    merged_label = f"{all_tick_labels[-1]}|{seg_tick_labels[0]}"
                    all_tick_labels[-1] = merged_label
                    for tick_pos, label in zip(
                        seg_tick_positions[1:], seg_tick_labels[1:]
                    ):
                        all_tick_positions.append(tick_pos)
                        all_tick_labels.append(label)
                        ax.axvline(tick_pos, color=KTICK_COLOR, lw=0.8, zorder=0)
                else:
                    for tick_pos, label in zip(seg_tick_positions, seg_tick_labels):
                        all_tick_positions.append(tick_pos)
                        all_tick_labels.append(label)
                        ax.axvline(tick_pos, color=KTICK_COLOR, lw=0.8, zorder=0)

        current_x_offset += seg_width

        # Only insert a gap at discontinuous boundaries
        if seg_idx < len(plot_segments) - 1 and use_gaps:
            next_seg = plot_segments[seg_idx + 1]
            if segment.labels[-1] != next_seg.labels[0]:
                gap_positions.append((current_x_offset, current_x_offset + gap_size))
                current_x_offset += gap_size

    # Legend entries - use ADJUSTED colors to match what's actually plotted (the GUI's wording)
    # no spin legend for a run that is not spin-polarized (the GUI's)
    if spin == "polarized":
        dummy_line_up = ax.plot([], [], linewidth=lw, color=plot_up_color, label="spin up (α)")
        dummy_line_down = ax.plot(
            [],
            [],
            linewidth=lw,
            linestyle=down_dash,
            color=plot_down_color,
            label="spin down (β)",
        )

    # Fermi level (always opaque): dotted, as in the DOS and the GUI (dashed reads as spin down)
    fermi_line = ax.axhline(
        maxV,
        color=FERMI_COLOR,
        linestyle=P.mpl_dash("dot"),
        lw=1.4,
        label=fermi_legend_label(fermi_shift),
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
    tick_text = stagger_tick_labels(all_tick_positions, all_tick_labels, 0, current_x_offset,
                                    fig_width * (SUBPLOT_RIGHT - SUBPLOT_LEFT), tick_pt)
    staggered = any(t.startswith("\n") for t in tick_text)
    # the dropped row clears a subscript (F0) in the row above
    ax.set_xticklabels(tick_text, fontsize=tick_pt, linespacing=1.4 if staggered else 1.2)

    # Y-axis formatting
    ax.yaxis.set_major_locator(MultipleLocator(1.0 if (E_u - E_l) <= 14 else 2.0))
    ax.yaxis.set_minor_locator(NullLocator())        # no minor ticks (the GUI has none)

    # Grid: the GUI's light major grid, no minor grid (a style without one: Publication, Print grayscale)
    if preset["grid"] != "none":
        ax.grid(True, which="major", axis="y", linewidth=0.8, color=GRID_COLOR, linestyle="-")
        ax.grid(False, which="minor")
    else:
        ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    ax.set_ylabel(energy_axis_label(has_potc, fermi_shift), fontsize=label_pt)

    # no x-axis title (the GUI's): the k labels name the axis, with one or two rows of them

    plt.yticks(fontsize=tick_pt)
    tick_dir = "out" if preset["ticks"] == "outside" else "in"
    ax.tick_params(which="major", length=5, width=1.0 * preset["frame"], direction=tick_dir)
    ax.tick_params(which="minor", length=3, width=0.8 * preset["frame"], direction=tick_dir)

    # the legend outside the axes: a metal or a crowded path has no empty corner for it
    handles, names = [], []
    if spin == "polarized":
        handles += [dummy_line_up[0], dummy_line_down[0]]
        names += ["spin up (α)", "spin down (β)"]
    legend = legend_outside(
        fig,
        ax,
        handles + [fermi_line],
        names + [fermi_legend_label(fermi_shift)],
        frameon=True,
        fontsize=max(8, tick_pt - 1),
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
                fig.savefig(output_path, format=fmt, dpi=PNG_DPI)
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
    # --energy-reference: auto (vacuum scale with the POTC files, else E - E_F), fermi or vacuum
    file4, file5, has_potc = energy_reference(getattr(args, "energy_reference", "auto"),
                                              associated_files.get("potc_dat"), associated_files.get("potc_out"))
    fermi_shift = float(getattr(args, "fermi_shift", 0.0) or 0.0)

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
    BANDS, BANDSbeta = apply_fermi_shift(BANDS, BANDSbeta, fermi_shift)

    print(f"  Data shape: {E.shape[0]} k-points, {BANDS.shape[1]} bands")

    segments = identify_segments_from_data(
        segments_labels, x_labels, E, BANDS, kpoint_indices
    )

    if args.list_segments:
        list_segments(segments)
        return

    if getattr(args, "band_path", None):
        # an edited path (mace gui's k-path editor, opt2d3's band_path): the data
        # laid out along it; the formatting below is the same
        if not (file3 and os.path.exists(file3)):
            raise ValueError("--band-path needs the run's .d3 beside the BAND.DAT (its path labels)")
        edit = load_band_path(args.band_path)
        half = len(x_labels) // 2
        ticks = x_labels[:half] if half and len(x_labels) % 2 == 0 and np.allclose(x_labels[:half], x_labels[half:]) else x_labels
        E, BANDS, BANDSbeta, segments, x_labels = edited_band_data(E, BANDS, BANDSbeta, ticks, d3_path_tokens(file3), edit)
        print("  k-path as edited: " + " ".join(_band_path_module().path_tokens(edit)))
        plot_bands_with_gaps(E, BANDS, E, BANDSbeta, segments, None, x_labels, maxV, args.E_lower, args.E_upper,
                             args.gap_width, output_dir, material, has_potc=has_potc,
                             use_gaps=not args.no_gaps, all_segments=segments, fixed_width=args.fixed_width,
                             fixed_height=args.fixed_height, kpoint_indices=None, alpha=args.alpha,
                             spin_up_color=args.spin_up_color, spin_down_color=args.spin_down_color,
                             band_color=getattr(args, "band_color", None),
                             auto_width=args.auto_width, formats=args.formats, fermi_shift=fermi_shift,
                             style=getattr(args, "style", "default"), spin_dash=getattr(args, "spin_style", None))
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
        has_potc=has_potc,
        use_gaps=use_gaps,
        all_segments=segments,
        fixed_width=args.fixed_width,
        fixed_height=args.fixed_height,
        kpoint_indices=kpoint_indices,
        alpha=args.alpha,
        spin_up_color=args.spin_up_color,
        spin_down_color=args.spin_down_color,
        band_color=getattr(args, "band_color", None),
        auto_width=args.auto_width,
        formats=args.formats,
        fermi_shift=fermi_shift,
        style=getattr(args, "style", "default"),
        spin_dash=getattr(args, "spin_style", None),
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
    parser.add_argument("--band-path", default=None, metavar="FILE",
                        help="an edited k-path (JSON: band_path, as mace gui and opt2d3 write it)")
    parser.add_argument("--fermi-shift", type=float, default=0.0, metavar="EV",
                        help="Move the bands by -EV: the energy EV above E_F becomes the reference (default: 0)")
    parser.add_argument("--energy-reference", choices=["auto", "fermi", "vacuum"], default="auto",
                        help="auto: the vacuum scale when the run's POTC files are beside the data, else E - E_F "
                             "(default); fermi: E - E_F always; vacuum: the vacuum scale (needs the POTC files)")
    parser.add_argument("--gap-width", type=float, default=0.05)
    parser.add_argument("--no-gaps", action="store_true")
    parser.add_argument(
        "--auto-width",
        action="store_true",
        help="Scale figure width to prevent label overlap when many segments are plotted",
    )
    parser.add_argument("--fixed-width", type=float, default=DEFAULT_FIG_WIDTH)
    parser.add_argument("--fixed-height", type=float, default=DEFAULT_FIG_HEIGHT)
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    parser.add_argument("--band-color", default=None, metavar="HEX",
                        help="The bands of a run that is not spin-polarized (default: the style's, else black)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Transparency (0-1, default: 1.0). Overlapping bands will stack and appear more opaque.",
    )
    parser.add_argument("--spin-up-color", type=str, default=None,
                        help="Spin up color (default: the style's; blue in Default)")
    parser.add_argument("--spin-down-color", type=str, default=None,
                        help="Spin down color (default: the style's; red in Default)")
    parser.add_argument("--spin-style", choices=["dash", "dot", "dashdot", "solid"], default=DEFAULT_SPIN_DOWN_DASH,
                        help="Spin down's line style (default: dash)")
    parser.add_argument("--style", choices=list(_presets().ORDER), default="default",
                        help="Plot style: default, publication, presentation, grayscale, colorblind")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Write the figures here (default: beside the data)")
    parser.add_argument("--file", default=None, metavar="BAND.DAT",
                        help="Plot this file only (default: every BAND.DAT in the folder)")
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg",
        help="Output formats (comma-separated): svg, png, pdf. Default: png,svg",
    )

    args = parser.parse_args()
    global PNG_DPI
    PNG_DPI = args.dpi

    if args.alpha < 0 or args.alpha > 1:
        print("Error: --alpha must be between 0 and 1")
        sys.exit(1)

    DIR = os.getcwd() + "/"
    FIGDIR = args.output_dir or DIR
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    band_files = find_band_files(DIR)
    if args.file:
        band_files = [f for f in band_files if os.path.basename(f) == os.path.basename(args.file)]

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
