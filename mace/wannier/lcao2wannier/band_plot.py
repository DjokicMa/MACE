# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Band structure plotting module for lcao2wannier.

Provides general-purpose band structure computation and visualization with
optional PDWF projectability coloring and projected DOS.

Supports automatic k-path detection for common lattice types and custom
k-path specification.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from .fourier import fourier_transform_to_kspace
from .solver import solve_generalized_eigenvalue_problem
from .win_file import (
    KPATH_HEXAGONAL_2D, KPATH_HEXAGONAL_3D,
    KPATH_SQUARE_2D, KPATH_FCC, KPATH_BCC, KPATH_SIMPLE_CUBIC,
)

# Hartree to eV conversion
HARTREE_TO_EV = 27.211386245988


# ---- Dataclasses ----

@dataclass
class KPathSpec:
    """Specification for a k-path through the Brillouin zone.

    Attributes:
        segments: List of (start_label, end_label) pairs defining path segments.
        high_sym_points: Dictionary mapping labels to fractional coordinates.
        npts_per_segment: Number of k-points per segment.
    """
    segments: List[Tuple[str, str]]
    high_sym_points: Dict[str, np.ndarray]
    npts_per_segment: int = 60


@dataclass
class KPathResult:
    """Result of k-path generation.

    Attributes:
        kpoints_frac: (nk_path, 3) array of k-points in fractional coordinates.
        distances: (nk_path,) array of cumulative Cartesian distances.
        tick_positions: Positions of high-symmetry points along the path.
        tick_labels: Labels for high-symmetry points.
    """
    kpoints_frac: np.ndarray
    distances: np.ndarray
    tick_positions: List[float]
    tick_labels: List[str]


@dataclass
class BandStructureData:
    """Complete band structure data for plotting.

    Attributes:
        kpath: The k-path result with distances and labels.
        eigenvalues: (nk_path, num_bands) eigenvalues in eV.
        e_fermi: Fermi energy in eV.
        num_bands: Number of bands.
        projectability: Optional (nk_path, num_bands) projectability values.
        classification: Optional BandClassification from PDWF.
        windows: Optional WindowParameters from PDWF.
    """
    kpath: KPathResult
    eigenvalues: np.ndarray
    e_fermi: float
    num_bands: int
    projectability: Optional[np.ndarray] = None
    classification: object = None  # BandClassification
    windows: object = None  # WindowParameters


@dataclass
class PlotConfig:
    """Configuration for band structure plots.

    Attributes:
        energy_range: (e_min, e_max) relative to Fermi level in eV.
        figsize: Figure size in inches.
        show_dos: Whether to show projected DOS panel.
        dos_sigma: Gaussian broadening for DOS in eV.
        cmap_name: Matplotlib colormap name for projectability.
        linewidth: Line width for bands.
        dpi: Resolution for saved figures.
    """
    energy_range: Tuple[float, float] = (-20.0, 25.0)
    figsize: Tuple[float, float] = (16, 8)
    show_dos: bool = True
    dos_sigma: float = 0.3
    cmap_name: str = 'RdYlBu'
    linewidth: float = 1.5
    dpi: int = 150


# ---- K-path functions ----

def kpath_from_win_format(kpath_pairs: list, npts: int = 60) -> KPathSpec:
    """Convert KPATH_* consecutive-pair lists to KPathSpec.

    The win_file.py format stores k-paths as consecutive pairs:
    [(label1, k1), (label2, k2), (label3, k3), (label4, k4), ...]
    where (0,1), (2,3), (4,5) define segments.

    Args:
        kpath_pairs: List of (label, k_frac) pairs from win_file KPATH constants.
        npts: Number of k-points per segment.

    Returns:
        KPathSpec with extracted segments and high-symmetry points.
    """
    segments = []
    high_sym_points = {}

    for i in range(0, len(kpath_pairs), 2):
        start_label, start_k = kpath_pairs[i]
        end_label, end_k = kpath_pairs[i + 1]
        segments.append((start_label, end_label))
        high_sym_points[start_label] = np.asarray(start_k, dtype=float)
        high_sym_points[end_label] = np.asarray(end_k, dtype=float)

    return KPathSpec(
        segments=segments,
        high_sym_points=high_sym_points,
        npts_per_segment=npts,
    )


def detect_lattice_type(lattice_vectors: np.ndarray) -> str:
    """Detect crystal lattice type from lattice vectors.

    Analyzes vector lengths and angles to classify the Bravais lattice.

    Args:
        lattice_vectors: (3, 3) array of lattice vectors (rows).

    Returns:
        One of: 'hexagonal_3d', 'hexagonal_2d', 'fcc', 'bcc', 'sc', 'unknown'.
    """
    a = lattice_vectors[0]
    b = lattice_vectors[1]
    c = lattice_vectors[2]

    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)

    # Compute angles between lattice vectors
    cos_ab = np.dot(a, b) / (la * lb)
    cos_ac = np.dot(a, c) / (la * lc)
    cos_bc = np.dot(b, c) / (lb * lc)

    angle_ab = np.degrees(np.arccos(np.clip(cos_ab, -1, 1)))
    angle_ac = np.degrees(np.arccos(np.clip(cos_ac, -1, 1)))
    angle_bc = np.degrees(np.arccos(np.clip(cos_bc, -1, 1)))

    tol_len = 0.01  # Relative tolerance for length comparison
    tol_ang = 1.0   # Degrees tolerance for angle comparison

    # Check for hexagonal: |a| = |b| != |c|, angle(a,b) = 120 deg, c perp to ab
    if (abs(la - lb) / la < tol_len
            and abs(angle_ab - 120.0) < tol_ang
            and abs(angle_ac - 90.0) < tol_ang
            and abs(angle_bc - 90.0) < tol_ang):
        # 2D vs 3D: check if c is very large (slab/2D) or normal
        # A simple heuristic: if c/a > 5, it's probably a 2D system
        if lc / la > 5.0:
            return 'hexagonal_2d'
        return 'hexagonal_3d'

    # Check for cubic: |a| = |b| = |c|
    if (abs(la - lb) / la < tol_len and abs(la - lc) / la < tol_len):
        # All angles 90 degrees -> simple cubic
        if (abs(angle_ab - 90.0) < tol_ang
                and abs(angle_ac - 90.0) < tol_ang
                and abs(angle_bc - 90.0) < tol_ang):
            return 'sc'

        # All angles 60 degrees -> FCC (primitive cell)
        if (abs(angle_ab - 60.0) < tol_ang
                and abs(angle_ac - 60.0) < tol_ang
                and abs(angle_bc - 60.0) < tol_ang):
            return 'fcc'

        # Angles ~109.47 degrees -> BCC (primitive cell)
        bcc_angle = np.degrees(np.arccos(-1/3))  # ~109.47
        if (abs(angle_ab - bcc_angle) < tol_ang
                and abs(angle_ac - bcc_angle) < tol_ang
                and abs(angle_bc - bcc_angle) < tol_ang):
            return 'bcc'

    return 'unknown'


def get_kpath_for_lattice(lattice_type: str, npts: int = 60) -> KPathSpec:
    """Get a standard k-path for a given lattice type.

    Args:
        lattice_type: One of 'hexagonal_3d', 'hexagonal_2d', 'fcc', 'bcc',
                      'sc', 'unknown'.
        npts: Number of k-points per segment.

    Returns:
        KPathSpec for the lattice type.

    Raises:
        ValueError: If lattice type is unknown and no fallback is possible.
    """
    kpath_map = {
        'hexagonal_3d': KPATH_HEXAGONAL_3D,
        'hexagonal_2d': KPATH_HEXAGONAL_2D,
        'fcc': KPATH_FCC,
        'bcc': KPATH_BCC,
        'sc': KPATH_SIMPLE_CUBIC,
    }

    if lattice_type in kpath_map:
        return kpath_from_win_format(kpath_map[lattice_type], npts)

    # Fallback for unknown: use simple cubic path
    print(f"  Warning: Unknown lattice type '{lattice_type}', "
          f"using simple cubic k-path as fallback.")
    return kpath_from_win_format(KPATH_SIMPLE_CUBIC, npts)


def parse_custom_kpath(kpath_str: str, npts: int = 60) -> KPathSpec:
    """Parse a custom k-path from a CLI string.

    Format: "G:0,0,0;M:0.5,0,0;K:0.333,0.333,0;G:0,0,0"
    Consecutive pairs of points define segments: (0→1), (1→2), (2→3), ...

    Args:
        kpath_str: String defining k-path points.
        npts: Number of k-points per segment.

    Returns:
        KPathSpec with parsed path.
    """
    parts = kpath_str.strip().split(';')
    high_sym_points = {}
    labels = []

    for part in parts:
        part = part.strip()
        if ':' not in part:
            raise ValueError(f"Invalid k-path point format: '{part}'. "
                             f"Expected 'LABEL:kx,ky,kz'.")
        label, coords_str = part.split(':', 1)
        label = label.strip()
        coords = np.array([float(x) for x in coords_str.split(',')])
        if len(coords) != 3:
            raise ValueError(f"Expected 3 coordinates for point '{label}', "
                             f"got {len(coords)}.")
        high_sym_points[label] = coords
        labels.append(label)

    if len(labels) < 2:
        raise ValueError("K-path must have at least 2 points.")

    # Create segments from consecutive points
    segments = []
    for i in range(len(labels) - 1):
        segments.append((labels[i], labels[i + 1]))

    return KPathSpec(
        segments=segments,
        high_sym_points=high_sym_points,
        npts_per_segment=npts,
    )


def generate_kpath(spec: KPathSpec, lattice_vectors: np.ndarray) -> KPathResult:
    """Generate k-points along a high-symmetry path with Cartesian distances.

    Args:
        spec: K-path specification with segments and high-symmetry points.
        lattice_vectors: (3, 3) real-space lattice vectors (rows).

    Returns:
        KPathResult with k-points, distances, and tick information.
    """
    recip_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T

    kpath_points = []
    kpath_distances = []
    tick_positions = []
    tick_labels = []
    total_dist = 0.0

    for seg_idx, (start_label, end_label) in enumerate(spec.segments):
        k_start = spec.high_sym_points[start_label]
        k_end = spec.high_sym_points[end_label]

        # Cartesian distance for proper spacing
        dk_cart = recip_lattice.T @ (k_end - k_start)
        seg_length = np.linalg.norm(dk_cart)

        if seg_idx == 0:
            tick_positions.append(total_dist)
            tick_labels.append(start_label)

        for i in range(spec.npts_per_segment):
            t = i / spec.npts_per_segment
            k_frac = k_start + t * (k_end - k_start)
            kpath_points.append(k_frac)
            kpath_distances.append(total_dist + t * seg_length)

        total_dist += seg_length
        tick_positions.append(total_dist)
        tick_labels.append(end_label)

    # Add last point
    last_seg = spec.segments[-1]
    kpath_points.append(spec.high_sym_points[last_seg[1]])
    kpath_distances.append(total_dist)

    # Convert 'G' to 'Γ' in labels for display
    tick_labels = [('Γ' if l == 'G' else l) for l in tick_labels]

    return KPathResult(
        kpoints_frac=np.array(kpath_points),
        distances=np.array(kpath_distances),
        tick_positions=tick_positions,
        tick_labels=tick_labels,
    )


# ---- Band structure computation ----

def compute_band_structure(
    real_space_matrices: list,
    lattice_vectors: np.ndarray,
    kpath: KPathResult,
) -> Tuple[np.ndarray, list, list]:
    """Compute eigenvalues and eigenvectors along a k-path.

    Args:
        real_space_matrices: List of (R_vector, H_R, S_R) tuples.
        lattice_vectors: (3, 3) real-space lattice vectors.
        kpath: KPathResult with k-points to solve at.

    Returns:
        eigenvalues: (nk_path, num_bands) array in eV.
        eigvecs_list: List of eigenvector matrices.
        S_k_list: List of overlap matrices at each k-point.
    """
    nk_path = len(kpath.kpoints_frac)
    eigvecs_list = []
    S_k_list = []

    # Solve first point to determine num_bands
    H_k0, S_k0 = fourier_transform_to_kspace(
        kpath.kpoints_frac[0], real_space_matrices, lattice_vectors
    )
    evals0, evecs0 = solve_generalized_eigenvalue_problem(H_k0, S_k0)
    num_bands = len(evals0)

    eigenvalues = np.zeros((nk_path, num_bands))
    eigenvalues[0, :] = evals0 * HARTREE_TO_EV
    eigvecs_list.append(evecs0)
    S_k_list.append(S_k0)

    # Solve remaining k-points
    for ik in range(1, nk_path):
        H_k, S_k = fourier_transform_to_kspace(
            kpath.kpoints_frac[ik], real_space_matrices, lattice_vectors
        )
        evals, evecs = solve_generalized_eigenvalue_problem(H_k, S_k)
        eigenvalues[ik, :] = evals * HARTREE_TO_EV
        eigvecs_list.append(evecs)
        S_k_list.append(S_k)

    return eigenvalues, eigvecs_list, S_k_list


def compute_path_projectability(
    eigvecs_list: list,
    S_k_list: list,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Compute PDWF projectability along a k-path.

    Thin wrapper around compute_lowdin_projectability for path k-points.

    Args:
        eigvecs_list: List of eigenvector matrices from compute_band_structure.
        S_k_list: List of overlap matrices from compute_band_structure.
        target_mask: Boolean mask for target orbitals.

    Returns:
        (nk_path, num_bands) array of projectability values in [0, 1].
    """
    from .lcao_pdwf import compute_lowdin_projectability
    return compute_lowdin_projectability(eigvecs_list, S_k_list, target_mask)


# ---- Plotting ----

def plot_band_structure(
    data: BandStructureData,
    output_path: str,
    config: Optional[PlotConfig] = None,
    title: Optional[str] = None,
) -> bool:
    """Plot band structure with optional projectability coloring and DOS.

    Creates a two-panel figure:
    - Left: Band structure with projectability-colored lines
    - Right: Projected DOS (if config.show_dos is True)

    Falls back gracefully if matplotlib is not available.

    Args:
        data: BandStructureData with eigenvalues, k-path, and optional PDWF data.
        output_path: Path to save the plot (e.g., 'material_bands.png').
        config: PlotConfig for customization. Uses defaults if None.
        title: Optional plot title. Auto-generated if None.

    Returns:
        True if plot was saved successfully, False if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        return False

    if config is None:
        config = PlotConfig()

    e_min_plot, e_max_plot = config.energy_range
    has_proj = data.projectability is not None

    # Determine number of panels
    if config.show_dos and has_proj:
        fig, axes = plt.subplots(
            1, 2, figsize=config.figsize,
            gridspec_kw={'width_ratios': [3, 1]}
        )
        ax_bands = axes[0]
        ax_dos = axes[1]
    else:
        fig, ax_bands = plt.subplots(1, 1, figsize=(config.figsize[0] * 0.75,
                                                      config.figsize[1]))
        ax_dos = None

    # ---- Band structure panel ----
    if has_proj:
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap(config.cmap_name)

    for band_idx in range(data.num_bands):
        energies = data.eigenvalues[:, band_idx] - data.e_fermi

        # Skip bands outside energy range
        if np.max(energies) < e_min_plot or np.min(energies) > e_max_plot:
            continue

        if has_proj:
            # Projectability-colored line segments
            projs = data.projectability[:, band_idx]
            points = np.column_stack([data.kpath.distances, energies])
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            colors = cmap(norm((projs[:-1] + projs[1:]) / 2))
            lc = LineCollection(segments, colors=colors,
                                linewidths=config.linewidth)
            ax_bands.add_collection(lc)
        else:
            # Uniform color
            ax_bands.plot(data.kpath.distances, energies,
                          color='#1f77b4', linewidth=config.linewidth)

    # Energy windows from PDWF
    if data.windows is not None:
        w = data.windows
        if hasattr(w, 'dis_win_min') and w.dis_win_min is not None:
            ax_bands.axhspan(
                w.dis_win_min - data.e_fermi, w.dis_win_max - data.e_fermi,
                alpha=0.08, color='blue', label='Outer window'
            )
        if hasattr(w, 'dis_froz_min') and w.dis_froz_min is not None:
            ax_bands.axhspan(
                w.dis_froz_min - data.e_fermi, w.dis_froz_max - data.e_fermi,
                alpha=0.15, color='green', label='Frozen window'
            )

    # Fermi level
    ax_bands.axhline(0, color='black', linestyle='--', linewidth=0.8,
                     alpha=0.5, label='E_F')

    # High-symmetry point markers
    for pos in data.kpath.tick_positions:
        ax_bands.axvline(pos, color='gray', linestyle='-', linewidth=0.5,
                         alpha=0.5)
    ax_bands.set_xticks(data.kpath.tick_positions)
    ax_bands.set_xticklabels(data.kpath.tick_labels, fontsize=12)

    ax_bands.set_xlim(data.kpath.distances[0], data.kpath.distances[-1])
    ax_bands.set_ylim(e_min_plot, e_max_plot)
    ax_bands.set_ylabel('E - E_F (eV)', fontsize=12)

    if title is None:
        proj_str = ', PDWF projectability' if has_proj else ''
        title = f'LCAO Band Structure ({data.num_bands} bands{proj_str})'
    ax_bands.set_title(title, fontsize=13)
    ax_bands.legend(loc='upper right', fontsize=9)

    # Colorbar for projectability
    if has_proj:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_bands, label='Projectability',
                     shrink=0.8, pad=0.02)

    # ---- DOS panel ----
    if ax_dos is not None and has_proj:
        nk_path = len(data.kpath.distances)
        e_bins = np.linspace(e_min_plot, e_max_plot, 200)
        sigma = config.dos_sigma

        dos_total = np.zeros_like(e_bins)
        dos_target = np.zeros_like(e_bins)

        for band_idx in range(data.num_bands):
            for ik in range(nk_path):
                e_rel = data.eigenvalues[ik, band_idx] - data.e_fermi
                p = data.projectability[ik, band_idx]
                gauss = (np.exp(-0.5 * ((e_bins - e_rel) / sigma)**2)
                         / (sigma * np.sqrt(2 * np.pi)))
                dos_total += gauss
                dos_target += p * gauss

        # Normalize by number of k-points
        dos_total /= nk_path
        dos_target /= nk_path

        ax_dos.fill_betweenx(e_bins, 0, dos_total, alpha=0.3, color='gray',
                             label='Total')
        ax_dos.fill_betweenx(e_bins, 0, dos_target, alpha=0.5, color='blue',
                             label='Target proj.')
        ax_dos.axhline(0, color='black', linestyle='--', linewidth=0.8,
                       alpha=0.5)

        # Mirror energy windows in DOS panel
        if data.windows is not None:
            w = data.windows
            if hasattr(w, 'dis_win_min') and w.dis_win_min is not None:
                ax_dos.axhspan(
                    w.dis_win_min - data.e_fermi,
                    w.dis_win_max - data.e_fermi,
                    alpha=0.08, color='blue'
                )
            if hasattr(w, 'dis_froz_min') and w.dis_froz_min is not None:
                ax_dos.axhspan(
                    w.dis_froz_min - data.e_fermi,
                    w.dis_froz_max - data.e_fermi,
                    alpha=0.15, color='green'
                )

        ax_dos.set_ylim(e_min_plot, e_max_plot)
        ax_dos.set_xlabel('DOS (arb.)', fontsize=11)
        ax_dos.set_title('Projected DOS', fontsize=13)
        ax_dos.legend(loc='upper right', fontsize=9)
        ax_dos.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    return True


# ---- Text fallback ----

def text_band_summary(data: BandStructureData, output_dir: Optional[str] = None,
                      seedname: Optional[str] = None) -> str:
    """Generate a text summary of band structure data.

    Also writes a gnuplot-compatible .dat file if output_dir and seedname
    are provided.

    Args:
        data: BandStructureData with eigenvalues and k-path.
        output_dir: Directory to write .dat file.
        seedname: Base name for output files.

    Returns:
        Multi-line string with band structure summary.
    """
    lines = []
    e_min_plot, e_max_plot = -20.0, 25.0

    lines.append(f"  E_Fermi: {data.e_fermi:.4f} eV")
    lines.append(f"  Bands in plot range [{e_min_plot:.0f}, {e_max_plot:.0f}] eV "
                 f"(rel. to E_F):")
    lines.append("")

    has_proj = data.projectability is not None

    # Header
    hdr = f"  {'Band':>6s}  {'E_min':>10s}  {'E_max':>10s}  {'Width':>8s}"
    if has_proj:
        hdr += f"  {'Avg p':>8s}  {'Min p':>8s}"
    if data.classification is not None:
        hdr += f"  {'Class':>12s}"
    lines.append(hdr)
    lines.append(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 8}"
                 + (f"  {'-' * 8}  {'-' * 8}" if has_proj else "")
                 + (f"  {'-' * 12}" if data.classification is not None else ""))

    for b in range(data.num_bands):
        e_min_b = data.eigenvalues[:, b].min() - data.e_fermi
        e_max_b = data.eigenvalues[:, b].max() - data.e_fermi
        width = e_max_b - e_min_b

        if e_max_b < e_min_plot or e_min_b > e_max_plot:
            continue

        row = f"  {b:6d}  {e_min_b:10.4f}  {e_max_b:10.4f}  {width:8.4f}"
        if has_proj:
            avg_p = np.mean(data.projectability[:, b])
            min_p = np.min(data.projectability[:, b])
            row += f"  {avg_p:8.4f}  {min_p:8.4f}"
        if data.classification is not None:
            cls = data.classification
            if b in cls.frozen_indices:
                row += f"  {'FROZEN':>12s}"
            elif b in cls.disent_indices:
                row += f"  {'DISENT':>12s}"
            elif b in cls.excluded_indices:
                row += f"  {'excluded':>12s}"
            else:
                row += f"  {'???':>12s}"
        lines.append(row)

    # Fermi-crossing bands
    lines.append("")
    lines.append("  Key features:")
    crossing_bands = []
    for b in range(data.num_bands):
        e_min_b = data.eigenvalues[:, b].min() - data.e_fermi
        e_max_b = data.eigenvalues[:, b].max() - data.e_fermi
        if e_min_b < 0 < e_max_b:
            crossing_bands.append(b)

    if crossing_bands:
        lines.append(f"  Fermi-crossing bands: {crossing_bands}")
        for b in crossing_bands:
            e_range = (f"[{data.eigenvalues[:, b].min() - data.e_fermi:.2f}, "
                       f"{data.eigenvalues[:, b].max() - data.e_fermi:.2f}]")
            extra = ""
            if has_proj:
                avg_p = np.mean(data.projectability[:, b])
                extra = f", avg_p={avg_p:.4f}"
            lines.append(f"    Band {b}: E={e_range} eV{extra}")

    # Band gaps
    for b in range(data.num_bands - 1):
        gap_min = data.eigenvalues[:, b + 1].min() - data.eigenvalues[:, b].max()
        if gap_min > 0.5:
            e_top = data.eigenvalues[:, b].max() - data.e_fermi
            e_bot = data.eigenvalues[:, b + 1].min() - data.e_fermi
            lines.append(f"  Gap between bands {b}-{b + 1}: {gap_min:.2f} eV "
                         f"(at {e_top:.2f} to {e_bot:.2f} eV)")

    # Write gnuplot-compatible .dat file
    if output_dir is not None and seedname is not None:
        dat_path = os.path.join(output_dir, f"{seedname}_bands.dat")
        with open(dat_path, 'w') as f:
            f.write(f"# Band structure data: {data.num_bands} bands, "
                    f"{len(data.kpath.distances)} k-points\n")
            f.write(f"# E_Fermi = {data.e_fermi:.6f} eV\n")
            f.write(f"# Columns: distance  E_1-E_F  E_2-E_F  ...  E_N-E_F\n")
            for ik in range(len(data.kpath.distances)):
                vals = [f"{data.kpath.distances[ik]:.6f}"]
                for b in range(data.num_bands):
                    vals.append(f"{data.eigenvalues[ik, b] - data.e_fermi:.6f}")
                f.write("  ".join(vals) + "\n")
        lines.append(f"\n  Wrote gnuplot data: {dat_path}")

    return "\n".join(lines)


# ---- High-level API ----

def read_w90_band_outputs(seedname: str) -> Dict[str, np.ndarray]:
    """Read Wannier90's pre-interpolated band-structure outputs.

    Returns the eigenvalues, k-points (fractional), cumulative path
    distances, and high-symmetry tick info that ``wannier90.x`` writes
    when ``bands_plot = T`` is set. The eigenvalues are already
    interpolated using ``wsvec.dat`` corrections, so they are the
    canonical Wannier-interpolated bands on the chosen path.

    Use this in preference to manually Fourier-transforming ``hr.dat``
    yourself. Naïve ``Σ_R H(R) exp(2πi k·R) / weight(R)`` is **wrong**
    for coarse k-grids because it double-counts R-vectors at the
    Wannier–Seitz cell boundary that ``wsvec.dat`` accounts for
    (see ``docs/guides/WANNIER_INTERPOLATION_PITFALL.md``).

    Parameters
    ----------
    seedname : str
        The Wannier90 seedname. The function expects to find
        ``{seedname}_band.dat``, ``{seedname}_band.kpt``, and
        ``{seedname}_band.labelinfo.dat`` in the current directory.

    Returns
    -------
    dict with keys:
        ``kpoints_frac``   : (nk, 3) ndarray of fractional k-coordinates
        ``eigenvalues``    : (nk, num_wann) ndarray of W90 energies (eV)
        ``distances``      : (nk,) ndarray of cumulative path distance
        ``tick_positions`` : list[float] of high-symmetry positions
        ``tick_labels``    : list[str] of high-symmetry labels (e.g. 'G', 'M')
        ``num_wann``       : int

    Notes
    -----
    Energies are in the frame W90 internally uses, which is whatever
    frame ``{seedname}.eig`` was written in. If our Stage 2 wrote the
    eig file shifted to ``E_F = 0`` (the default), then these
    eigenvalues are likewise centered at 0 — do *not* subtract the
    Fermi energy again when plotting.
    """
    band_data = np.loadtxt(f'{seedname}_band.dat')
    k_col = band_data[:, 0]
    e_col = band_data[:, 1]
    reset = np.where(np.diff(k_col) < -1e-6)[0]
    nk = (reset[0] + 1) if len(reset) else len(k_col)
    num_wann = len(k_col) // nk
    distances = k_col[:nk].copy()
    eigenvalues = e_col.reshape(num_wann, nk).T

    with open(f'{seedname}_band.kpt') as f:
        nk_kpt = int(f.readline().split()[0])
        raw = np.loadtxt(f, max_rows=nk_kpt)
    if nk_kpt != nk:
        raise RuntimeError(
            f"{seedname}_band.kpt has {nk_kpt} k-points but "
            f"{seedname}_band.dat has {nk}; W90 outputs are inconsistent."
        )
    kpoints_frac = raw[:, :3]

    tick_labels: List[str] = []
    tick_positions: List[float] = []
    try:
        with open(f'{seedname}_band.labelinfo.dat') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    tick_labels.append(parts[0])
                    tick_positions.append(float(parts[2]))
    except FileNotFoundError:
        pass

    return {
        'kpoints_frac': kpoints_frac,
        'eigenvalues': eigenvalues,
        'distances': distances,
        'tick_positions': tick_positions,
        'tick_labels': tick_labels,
        'num_wann': num_wann,
    }


def run_band_structure(
    real_space_matrices: list,
    lattice_vectors: np.ndarray,
    e_fermi: float,
    output_path: str,
    kpath_spec: Optional[KPathSpec] = None,
    lattice_type: Optional[str] = None,
    npts: int = 60,
    target_mask: Optional[np.ndarray] = None,
    classification: object = None,
    windows: object = None,
    config: Optional[PlotConfig] = None,
    title: Optional[str] = None,
    seedname: Optional[str] = None,
    verbose: bool = True,
) -> BandStructureData:
    """Compute and plot a complete band structure.

    This is the main API entry point. It:
    1. Auto-detects k-path if not specified
    2. Generates k-points along the path
    3. Computes eigenvalues at each k-point
    4. Optionally computes PDWF projectability
    5. Plots the result (or prints text summary)

    Args:
        real_space_matrices: List of (R, H_R, S_R) tuples.
        lattice_vectors: (3, 3) lattice vectors.
        e_fermi: Fermi energy in eV.
        output_path: Path for output plot (e.g., 'material_bands.png').
        kpath_spec: Optional explicit k-path. Auto-detected if None.
        lattice_type: Optional lattice type override for path selection.
        npts: Points per k-path segment (default: 60).
        target_mask: Boolean mask for PDWF projectability. Skipped if None.
        classification: Optional BandClassification for band labels.
        windows: Optional WindowParameters for energy window overlay.
        config: Optional PlotConfig. Defaults used if None.
        title: Optional plot title.
        seedname: Seedname for .dat file output.
        verbose: Print progress messages.

    Returns:
        BandStructureData with all computed results.
    """
    # Step 1: Determine k-path
    if kpath_spec is None:
        if lattice_type is None:
            lattice_type = detect_lattice_type(lattice_vectors)
            if verbose:
                print(f"  Auto-detected lattice type: {lattice_type}")
        kpath_spec = get_kpath_for_lattice(lattice_type, npts)

    if verbose:
        path_str = ' -> '.join(
            [s[0] for s in kpath_spec.segments] + [kpath_spec.segments[-1][1]]
        )
        # Convert G to Γ for display
        path_str = path_str.replace('G', 'Γ')
        print(f"  K-path: {path_str}")

    # Step 2: Generate k-points
    kpath = generate_kpath(kpath_spec, lattice_vectors)
    nk_path = len(kpath.kpoints_frac)
    if verbose:
        print(f"  {nk_path} k-points along path")

    # Step 3: Compute band structure
    if verbose:
        print("  Computing eigenvalues along path...")
    eigenvalues, eigvecs_list, S_k_list = compute_band_structure(
        real_space_matrices, lattice_vectors, kpath
    )
    num_bands = eigenvalues.shape[1]
    if verbose:
        print(f"  {num_bands} bands, eigenvalue range: "
              f"[{eigenvalues.min():.2f}, {eigenvalues.max():.2f}] eV")

    # Step 4: Compute projectability if target mask provided
    projectability = None
    if target_mask is not None:
        if verbose:
            print("  Computing projectability along path...")
        projectability = compute_path_projectability(
            eigvecs_list, S_k_list, target_mask
        )
        if verbose:
            print(f"  Projectability range: "
                  f"[{projectability.min():.4f}, {projectability.max():.4f}]")

    # Step 5: Package data
    band_data = BandStructureData(
        kpath=kpath,
        eigenvalues=eigenvalues,
        e_fermi=e_fermi,
        num_bands=num_bands,
        projectability=projectability,
        classification=classification,
        windows=windows,
    )

    # Step 6: Plot or text summary
    output_dir = os.path.dirname(os.path.abspath(output_path))

    if verbose:
        print("  Generating plot...")
    plot_ok = plot_band_structure(band_data, output_path, config=config,
                                  title=title)

    if plot_ok:
        if verbose:
            print(f"  Saved: {output_path}")
    else:
        if verbose:
            print("  matplotlib not available, generating text summary...")

    # Always generate text summary
    summary = text_band_summary(band_data, output_dir=output_dir,
                                seedname=seedname)
    if verbose or not plot_ok:
        print(summary)

    return band_data
