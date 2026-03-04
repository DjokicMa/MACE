#!/usr/bin/python3
"""
Crystal Structure Visualizer — generates true-vector SVG renderings of CIF files.

Performs orthographic 3D→2D projection and writes native SVG elements
(circles, lines) directly.  Transparent background, depth-sorted
rendering (painter's algorithm).

Handles CRYSTAL's origin-choice-2 convention for centrosymmetric space
groups (Fd-3m, I41/amd, Pn-3m, etc.) which ASE otherwise misreads.

Usage:
    python plottingCIFs.py [input_dir] [output_dir] [--supercell NX NY NZ]
                           [--size PX] [--jobs N] [--color-by-coord]

Defaults:
    input_dir   = NewOpt2/optcifs/       (relative to this script)
    output_dir  = NewOpt2/optcifs/svgs/  (relative to this script)
    jobs        = number of CPU cores
"""

import argparse
import multiprocessing
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── visual constants ─────────────────────────────────────────────────────────

ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#808080", "N": "#0000FF",
    "O": "#FF0000", "F": "#FFFF00", "Cl": "#00FF00",
    "Br": "#A52A2A", "I": "#940094", "Fe": "#FFA500",
    "Au": "#FFD700", "Ag": "#C0C0C0", "Cu": "#FF7F50",
    "Si": "#DAA520",
}

# Coordination-based coloring (overrides element color when --color-by-coord)
COORD_COLORS = {
    1: "#FF4444",   # red         — under-coordinated
    2: "#FF8800",   # orange      — sp  (linear)
    3: "#2288FF",   # blue        — sp2 (trigonal)
    4: "#22CC44",   # green       — sp3 (tetrahedral)
    5: "#AA44FF",   # purple      — over-coordinated
}
COORD_COLOR_DEFAULT = "#FF00FF"  # magenta — 6+ or 0

CAMERA_CONFIGS = [
    {"forward": np.array([0, 0, 1.0]),  "up": np.array([0, 1.0, 0]), "label": "XY"},
    {"forward": np.array([0, 1.0, 0]),  "up": np.array([0, 0, 1.0]), "label": "XZ"},
    {"forward": np.array([1.0, 0, 0]),  "up": np.array([0, 0, 1.0]), "label": "YZ"},
]

# Centrosymmetric space groups that have two origin choices.
# CRYSTAL always uses origin choice 2 (standard ITA); ASE defaults to 1.
ORIGIN_CHOICE_SGS = {
    48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130,
    133, 134, 137, 138, 141, 142, 201, 203, 222, 224, 227, 228,
}


# ── structure loading ────────────────────────────────────────────────────────

def _parse_cif_asym_unit(cif_path: str):
    """Parse the asymmetric-unit atom sites directly from the CIF text.

    Returns list of (symbol, x, y, z) tuples.
    """
    with open(cif_path) as f:
        content = f.read()

    lines = content.split("\n")
    in_atom_loop = False
    columns = []
    sites = []

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = False
            columns = []
            continue
        if stripped.startswith("_atom_site_"):
            columns.append(stripped)
            in_atom_loop = True
            continue
        if in_atom_loop and columns and stripped and not stripped.startswith("_"):
            if stripped.startswith(("loop_", "#", "data_")):
                in_atom_loop = False
                continue
            parts = stripped.split()
            if len(parts) >= len(columns):
                row = {col: parts[i] for i, col in enumerate(columns)}
                sites.append(row)

    results = []
    for row in sites:
        sym_raw = row.get("_atom_site_type_symbol",
                          row.get("_atom_site_label", "C"))
        m = re.match(r"[A-Z][a-z]?", sym_raw)
        sym = m.group(0) if m else "C"
        x = float(row["_atom_site_fract_x"])
        y = float(row["_atom_site_fract_y"])
        z = float(row["_atom_site_fract_z"])
        results.append((sym, x, y, z))

    return results


def load_structure(cif_path: str):
    """Load CIF with correct origin-choice handling for CRYSTAL output.

    For space groups with two origin choices, CRYSTAL uses setting 2
    (standard ITA) while ASE defaults to setting 1, which doubles atoms.
    We detect this and rebuild with the correct setting.
    """
    from ase.io import read
    from ase.spacegroup import crystal as ase_crystal

    atoms = read(cif_path, format="cif", store_tags=True)
    sg = atoms.info.get("spacegroup")

    if sg is not None and sg.no in ORIGIN_CHOICE_SGS:
        asym = _parse_cif_asym_unit(cif_path)
        if asym:
            cell = atoms.get_cell()
            cellpar = list(cell.lengths()) + list(cell.angles())
            symbols = [s for s, *_ in asym]
            positions = [(x, y, z) for _, x, y, z in asym]
            try:
                atoms = ase_crystal(
                    symbols, positions,
                    spacegroup=sg.no, setting=2,
                    cellpar=cellpar,
                )
            except Exception:
                pass  # fall back to default read

    return atoms


# ── geometry extraction ──────────────────────────────────────────────────────

def get_atoms_data(supercell, color_by_coord=False, bond_cutoff=1.9):
    """Return (positions, radii, colors, coordinations) arrays.

    Args:
        supercell: ASE Atoms object
        color_by_coord: If True, color atoms by coordination number
        bond_cutoff: Maximum bond length in Ångströms (default: 1.9)
    """
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList

    positions = supercell.positions.copy()
    symbols = supercell.get_chemical_symbols()
    radii = np.array([covalent_radii[supercell.numbers[i]] * 0.5
                      for i in range(len(supercell))])

    # Use fixed cutoff: each atom gets half the total cutoff
    cutoffs = [bond_cutoff / 2.0] * len(supercell)
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(supercell)
    coordinations = np.array([len(nl.get_neighbors(i)[0])
                              for i in range(len(supercell))])

    if color_by_coord:
        colors = [COORD_COLORS.get(c, COORD_COLOR_DEFAULT)
                  for c in coordinations]
    else:
        colors = [ELEMENT_COLORS.get(s, "#808080") for s in symbols]

    return positions, radii, colors, coordinations


def get_bonds(supercell, bond_cutoff=1.9):
    """Return list of (pos_start, pos_end) for ALL bonds, including periodic.

    Uses (i, j, offset_tuple) as the unique bond key so that distinct
    periodic images are kept.  Bonds crossing the supercell boundary
    are drawn to the periodic-image position (outside the box),
    showing connectivity on all faces equally.

    Args:
        supercell: ASE Atoms object
        bond_cutoff: Maximum bond length in Ångströms (default: 1.9)
    """
    from ase.neighborlist import NeighborList

    # Use fixed cutoff: each atom gets half the total cutoff
    cutoffs = [bond_cutoff / 2.0] * len(supercell)
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(supercell)

    cell = supercell.get_cell()
    positions = supercell.positions

    bonds = []
    processed = set()

    for i in range(len(supercell)):
        indices, offsets = nl.get_neighbors(i)
        pos_i = positions[i]
        for j, offset in zip(indices, offsets):
            if i < j:
                key = (i, j, tuple(offset))
            elif i > j:
                key = (j, i, tuple(-offset))
            else:
                canonical_offset = tuple(offset) if offset[0] > 0 or \
                    (offset[0] == 0 and offset[1] > 0) or \
                    (offset[0] == 0 and offset[1] == 0 and offset[2] > 0) \
                    else tuple(-offset)
                key = (i, j, canonical_offset)

            if key in processed:
                continue
            processed.add(key)

            pos_j = positions[j] + np.dot(offset, cell)
            bonds.append((pos_i.copy(), pos_j.copy()))

    return bonds


def get_unit_cell_edges(cell_matrix):
    """Return list of (start, end) for the 12 unit cell edges.

    Args:
        cell_matrix: The 3x3 cell matrix (use original UC, not supercell)
    """
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float,
    )
    vertices = corners @ cell_matrix
    pairs = ([(0, 1), (1, 2), (2, 3), (3, 0)]
             + [(4, 5), (5, 6), (6, 7), (7, 4)]
             + [(0, 4), (1, 5), (2, 6), (3, 7)])
    return [(vertices[a].copy(), vertices[b].copy()) for a, b in pairs]


# ── projection & SVG writing ────────────────────────────────────────────────

def orthographic_project(points_3d, forward, up):
    """Project 3D points to 2D.  Returns (points_2d, depths)."""
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    pts = np.asarray(points_3d)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    x = pts @ right
    y = pts @ up
    z = pts @ forward
    return np.column_stack([x, y]), z


def write_svg(filepath, atoms_pos, atoms_radii, atoms_colors,
              bonds, cell_edges, cam_forward, cam_up, size=800):
    """Render a single view to an SVG file with painter's algorithm."""

    atom_2d, atom_depth = orthographic_project(atoms_pos, cam_forward, cam_up)

    bond_data = []
    for s3, e3 in bonds:
        se = np.array([s3, e3])
        proj, dep = orthographic_project(se, cam_forward, cam_up)
        bond_data.append((proj[0], proj[1], dep.mean()))

    cell_data = []
    for s3, e3 in cell_edges:
        se = np.array([s3, e3])
        proj, dep = orthographic_project(se, cam_forward, cam_up)
        cell_data.append((proj[0], proj[1], dep.mean()))

    # Viewport fitted to atoms + cell edges (not periodic bond stubs)
    fit_pts = [atom_2d]
    for p1, p2, _ in cell_data:
        fit_pts.append(p1.reshape(1, 2))
        fit_pts.append(p2.reshape(1, 2))
    fit_pts = np.vstack(fit_pts)

    max_radius = atoms_radii.max() if len(atoms_radii) > 0 else 1.0
    margin = max_radius * 3
    xmin, ymin = fit_pts.min(axis=0) - margin
    xmax, ymax = fit_pts.max(axis=0) + margin
    world_w = xmax - xmin
    world_h = ymax - ymin

    scale = (size * 0.9) / max(world_w, world_h)
    cx = size / 2
    cy = size / 2
    world_cx = (xmin + xmax) / 2
    world_cy = (ymin + ymax) / 2

    def to_svg(xy):
        sx = cx + (xy[0] - world_cx) * scale
        sy = cy - (xy[1] - world_cy) * scale
        return sx, sy

    # Draw list sorted by depth (back to front)
    draw_list = []
    for i in range(len(atoms_pos)):
        draw_list.append((atom_depth[i], "atom", i))
    for idx in range(len(bond_data)):
        draw_list.append((bond_data[idx][2], "bond", idx))
    for idx in range(len(cell_data)):
        draw_list.append((cell_data[idx][2], "cell", idx))

    draw_list.sort(key=lambda x: -x[0])

    bond_width = max(1.0, 0.08 * scale)

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(size),
        "height": str(size),
        "viewBox": f"0 0 {size} {size}",
    })

    for depth, typ, idx in draw_list:
        if typ == "cell":
            p1_2d, p2_2d, _ = cell_data[idx]
            s1 = to_svg(p1_2d)
            s2 = to_svg(p2_2d)
            ET.SubElement(svg, "line", {
                "x1": f"{s1[0]:.2f}", "y1": f"{s1[1]:.2f}",
                "x2": f"{s2[0]:.2f}", "y2": f"{s2[1]:.2f}",
                "stroke": "black",
                "stroke-width": "2.0",
                "opacity": "0.7",
            })
        elif typ == "bond":
            p1_2d, p2_2d, _ = bond_data[idx]
            s1 = to_svg(p1_2d)
            s2 = to_svg(p2_2d)
            ET.SubElement(svg, "line", {
                "x1": f"{s1[0]:.2f}", "y1": f"{s1[1]:.2f}",
                "x2": f"{s2[0]:.2f}", "y2": f"{s2[1]:.2f}",
                "stroke": "#B0B0B0",
                "stroke-width": f"{bond_width:.2f}",
                "stroke-linecap": "round",
                "opacity": "0.8",
            })
        elif typ == "atom":
            sx, sy = to_svg(atom_2d[idx])
            r_svg = atoms_radii[idx] * scale
            color = atoms_colors[idx]
            ET.SubElement(svg, "circle", {
                "cx": f"{sx:.2f}",
                "cy": f"{sy:.2f}",
                "r": f"{r_svg:.2f}",
                "fill": color,
                "stroke": "#404040",
                "stroke-width": f"{max(0.3, r_svg * 0.08):.2f}",
            })

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(filepath, xml_declaration=True, encoding="UTF-8")


# ── per-file worker ──────────────────────────────────────────────────────────

def _process_one(args):
    cif_path, output_dir, supercell_factors, svg_size, color_by_coord, bond_cutoff = args
    cif_file = Path(cif_path)
    try:
        atoms = load_structure(str(cif_file))
        original_cell = atoms.get_cell().array.copy()  # Save original UC
        supercell = atoms * supercell_factors

        positions, radii, colors, coordinations = get_atoms_data(
            supercell, color_by_coord=color_by_coord, bond_cutoff=bond_cutoff,
        )
        bonds = get_bonds(supercell, bond_cutoff=bond_cutoff)
        cell_edges = get_unit_cell_edges(original_cell)  # Use original UC

        out_dir = Path(output_dir)
        for i, cam in enumerate(CAMERA_CONFIGS):
            out_path = str(out_dir / f"{cif_file.stem}_view{i}.svg")
            write_svg(out_path, positions, radii, colors,
                      bonds, cell_edges,
                      cam["forward"], cam["up"], size=svg_size)

        return (cif_file.name, None)
    except Exception as e:
        import traceback
        return (cif_file.name, traceback.format_exc())


def process_cif(cif_path: str, output_dir: str, supercell_factors=(2, 2, 2),
                svg_size: int = 800, color_by_coord: bool = False,
                bond_cutoff: float = 1.9):
    """
    Process a single CIF file and generate SVG visualizations.

    This is a public wrapper for external use (e.g., MACE plotting module).

    Args:
        cif_path: Path to the CIF file
        output_dir: Directory for output SVG files
        supercell_factors: Tuple of (nx, ny, nz) for supercell expansion
        svg_size: Canvas size in pixels
        color_by_coord: If True, color atoms by coordination number
        bond_cutoff: Bond cutoff in Angstroms

    Returns:
        List of generated SVG file paths, or None on error
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    args = (cif_path, str(output_dir), supercell_factors, svg_size, color_by_coord, bond_cutoff)
    name, err = _process_one(args)

    if err:
        print(f"  Error processing {name}: {err}")
        return None

    # Return list of generated files
    cif_file = Path(cif_path)
    return [str(out_path / f"{cif_file.stem}_view{i}.svg") for i in range(len(CAMERA_CONFIGS))]


def process_cif_files(input_dir: Path, output_dir: Path, supercell_factors,
                      svg_size: int, jobs: int, color_by_coord: bool,
                      bond_cutoff: float = 1.9):
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_files = sorted(input_dir.glob("*.cif"))
    if not cif_files:
        print(f"No CIF files found in {input_dir}")
        return

    work = [(str(f), str(output_dir), supercell_factors, svg_size, color_by_coord, bond_cutoff)
            for f in cif_files]

    print(f"Processing {len(work)} CIF files with {jobs} workers...")

    with multiprocessing.Pool(processes=jobs) as pool:
        for name, err in pool.imap_unordered(_process_one, work):
            if err:
                print(f"  FAIL {name}:\n{err}")
            else:
                print(f"  OK   {name}")

    print("Done.")


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "NewOpt2" / "optcifs"
    default_output = script_dir / "NewOpt2" / "optcifs" / "svgs"

    parser = argparse.ArgumentParser(
        description="Render CIF structures as true-vector SVG images.",
    )
    parser.add_argument("input_dir", nargs="?", default=str(default_input),
                        help="Directory containing CIF files")
    parser.add_argument("output_dir", nargs="?", default=str(default_output),
                        help="Output directory for SVG files")
    parser.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2],
                        metavar=("NX", "NY", "NZ"),
                        help="Supercell factors (default: 2 2 2)")
    parser.add_argument("--size", type=int, default=800,
                        help="SVG canvas size in px (default: 800)")
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel workers (default: ncpus-1)")
    parser.add_argument("--color-by-coord", action="store_true",
                        help="Color atoms by coordination number: "
                             "red=1, orange=2(sp), blue=3(sp2), "
                             "green=4(sp3), purple=5+")
    parser.add_argument("--bond-cutoff", type=float, default=1.9,
                        help="Bond cutoff in Angstroms (default: 1.9)")
    args = parser.parse_args()

    process_cif_files(
        Path(args.input_dir), Path(args.output_dir),
        args.supercell, args.size, args.jobs, args.color_by_coord,
        args.bond_cutoff,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
