#!/usr/bin/env python3
"""
Enhanced CRYSTAL Cube File Visualizer using Plotly for better interactivity.
Handles hexagonal, orthorhombic, and triclinic crystal systems correctly.

Key features:
- Interactive 3D visualization with Plotly
- Automatic log scale for density data
- Percentile-based color clipping for better contrast
- Interactive slice browser with slider
- Grid view of all slices
- Automatic detection of best slice (maximum density)
- HTML export capability

Usage:
python crystal_cube_viz_plotly.py filename.cube [options]

Options:
--iso VALUES        Override isosurface values (comma-separated)
--cmap COLORMAP     Plotly colormap name (default: auto-detected)
--alpha ALPHA       Transparency (0-1, default: 0.7)
--slice AXIS POS    Show 2D slice along AXIS at position POS
--slice-all AXIS    Show all slices along AXIS in a grid
--slice-browse AXIS Browse through slices interactively
--no-atoms         Don't show atoms
--save FILENAME    Save figure to HTML file
--log-scale        Force log scale for density
--linear-scale     Force linear scale
--clip PERCENTILE  Color clipping percentile (default: 99.5)
-i, --interactive  Interactive mode with menu
"""

# Suppress Qt/OpenGL warnings from display libraries
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import plotly.graph_objects as go
# Use plotly.colors, NOT plotly.express: importing plotly.express pulls in
# xarray -> pandas -> pyarrow, and under the full `mace` CLI (which already loads
# the pandas/pyarrow stack) that re-init raises
# `ArrowKeyError: A type extension with name arrow.py_extension_type already
# defined`. plotly.colors is the lightweight home of sample_colorscale (the only
# plotly.express symbol this engine used) and pulls no heavy deps.
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import sys
import os
import argparse
from typing import Tuple, List, Optional, Dict

# Try to import skimage for isosurface generation
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Note: Install scikit-image for better isosurface rendering")
    print("      pip install scikit-image")
    print()

# Try to import kaleido for PNG export (needed for GIF creation)
try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

# Try to import Pillow for GIF creation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class CubeFile:
    """Class to handle cube file reading with proper lattice vector support."""
    
    def __init__(self, filename: str, z_shift: float = 0.0, wrap_atoms: bool = True,
                 skip_vacuum_detection: bool = False):
        self.filename = filename
        self.comment1 = ""
        self.comment2 = ""
        self.natoms = 0
        self.origin = np.zeros(3)
        self.nvoxels = np.zeros(3, dtype=int)
        self.voxel_vectors = np.zeros((3, 3))
        self.lattice_vectors = None
        self.atoms = []
        self.atomic_numbers = []
        self.atomic_positions = []
        self.data = None
        self.data_type = None
        self.is_2d = False
        self.z_vacuum = 0.0
        self.cell_parameters = {}
        self.crystal_system = 'unknown'
        self.z_shift = z_shift
        self.wrap_atoms = wrap_atoms
        self.cell_dims = None
        self.skip_vacuum_detection = skip_vacuum_detection

        self.read_cube()
        self._determine_data_type()
        self._analyze_structure()

        if self.z_shift != 0.0:
            self._apply_manual_shift()

        # New: Automatic vacuum detection and cropping (skip for arithmetic operations)
        if not skip_vacuum_detection:
            self._detect_and_crop_vacuum()

        self._get_recommended_isovalues()
    
    def read_cube(self):
        """Read cube file and extract all information."""
        with open(self.filename, 'r') as f:
            # Read header
            self.comment1 = f.readline().strip()
            self.comment2 = f.readline().strip()
            
            # Extract cell parameters from comment if available
            self._parse_cell_parameters()
            
            # Read atoms and origin
            line = f.readline().split()
            self.natoms = int(line[0])
            self.origin = np.array([float(line[1]), float(line[2]), float(line[3])])
            
            # Read voxel information
            for i in range(3):
                line = f.readline().split()
                self.nvoxels[i] = int(line[0])
                self.voxel_vectors[i] = np.array([float(line[1]), float(line[2]), float(line[3])])
            
            # Calculate full lattice vectors
            self.lattice_vectors = np.zeros((3, 3))
            for i in range(3):
                self.lattice_vectors[i] = self.voxel_vectors[i] * self.nvoxels[i]
            
            # Check if non-orthogonal
            if (abs(self.voxel_vectors[1, 0]) > 1e-6 or 
                abs(self.voxel_vectors[2, 0]) > 1e-6 or 
                abs(self.voxel_vectors[2, 1]) > 1e-6):
                print(f"\nNote: Non-orthogonal lattice detected")
                print(f"Lattice vectors (Bohr):")
                for i in range(3):
                    print(f"  {'abc'[i]}: [{self.lattice_vectors[i, 0]:.3f}, "
                          f"{self.lattice_vectors[i, 1]:.3f}, {self.lattice_vectors[i, 2]:.3f}]")
            
            # Read atomic positions
            for i in range(abs(self.natoms)):
                line = f.readline().split()
                atomic_number = int(line[0])
                charge = float(line[1])
                position = np.array([float(line[2]), float(line[3]), float(line[4])])
                self.atomic_numbers.append(atomic_number)
                self.atomic_positions.append(position)

            # Multi-dataset cubes (Gaussian convention): a NEGATIVE atom count
            # means an extra "m id1..idm" line follows the atom block, and the
            # volumetric data interleaves m values per grid point. CRYSTAL emits
            # positive natoms (m=1) -- this is foreign-cube interop hardening.
            self.n_datasets = 1
            self.dataset_ids = []
            self.data_all = None
            if self.natoms < 0:
                dset_line = f.readline().split()
                if dset_line:
                    self.n_datasets = int(dset_line[0])
                    self.dataset_ids = [int(x) for x in dset_line[1:1 + self.n_datasets]]

            # Read volumetric data
            data_list = []
            for line in f:
                data_list.extend([float(x) for x in line.split()])

            nx, ny, nz = (int(self.nvoxels[0]), int(self.nvoxels[1]), int(self.nvoxels[2]))
            expected = nx * ny * nz * self.n_datasets
            if len(data_list) != expected:
                print(f"  Warning: cube data length {len(data_list)} != expected {expected} "
                      f"(nx*ny*nz*n_datasets); using the first {expected} values.")

            arr = np.array(data_list[:expected])
            if self.n_datasets > 1:
                # dataset index is fastest-varying within each grid point
                self.data_all = arr.reshape(nx, ny, nz, self.n_datasets)
                self.data = self.data_all[..., 0]
            else:
                self.data = arr.reshape(nx, ny, nz)
    
    def _parse_cell_parameters(self):
        """Extract cell parameters and determine crystal system."""
        try:
            parts = self.comment2.split()
            if len(parts) >= 6:
                for i in range(len(parts) - 5):
                    try:
                        a = float(parts[i])
                        b = float(parts[i+1])
                        c = float(parts[i+2])
                        alpha = float(parts[i+3])
                        beta = float(parts[i+4])
                        gamma = float(parts[i+5])
                        
                        if (0 < a < 1000 and 0 < b < 1000 and 0 < c < 10000 and
                            0 < alpha <= 180 and 0 < beta <= 180 and 0 < gamma <= 180):
                            self.cell_parameters = {
                                'a': a, 'b': b, 'c': c,
                                'alpha': alpha, 'beta': beta, 'gamma': gamma
                            }
                            
                            # Determine crystal system
                            if abs(gamma - 120.0) < 1.0 and abs(alpha - 90.0) < 1.0 and abs(beta - 90.0) < 1.0:
                                self.crystal_system = 'hexagonal'
                            elif abs(alpha - 90.0) < 1.0 and abs(beta - 90.0) < 1.0 and abs(gamma - 90.0) < 1.0:
                                self.crystal_system = 'orthorhombic'
                            else:
                                self.crystal_system = 'triclinic'
                            break
                    except ValueError:
                        continue
        except:
            pass
    
    def _determine_data_type(self):
        """Determine if data is electron density, spin density, potential, or difference."""
        # First check comment lines (title lines) for subtraction/difference
        if hasattr(self, 'comment1') and hasattr(self, 'comment2'):
            title_combined = (self.comment1 + ' ' + self.comment2).lower()

            # Check if this is a subtracted/difference file
            if 'subtracted' in title_combined or 'difference' in title_combined:
                # Determine the type of difference based on title content
                if 'dens' in title_combined or 'density' in title_combined or 'charge' in title_combined:
                    self.data_type = 'density_difference'
                elif 'pot' in title_combined or 'potential' in title_combined:
                    self.data_type = 'potential_difference'
                elif 'spin' in title_combined:
                    self.data_type = 'spin_difference'
                else:
                    # Generic difference - check filename or data
                    self.data_type = 'difference'
                return

        # Not a difference file - check filename
        filename_lower = self.filename.lower()
        if 'spin' in filename_lower:
            self.data_type = 'spin'
        elif 'ech3' in filename_lower or 'dens' in filename_lower or 'density' in filename_lower:
            # Check for DENS, DENSITY, or ECH3 (charge density) files
            if 'spin' not in filename_lower:
                self.data_type = 'density'
            else:
                self.data_type = 'spin'
        elif 'pot' in filename_lower or 'potential' in filename_lower:
            self.data_type = 'potential'
        else:
            # Fallback: determine from data characteristics
            if np.all(self.data >= 0):
                self.data_type = 'density'
            else:
                self.data_type = 'spin' if np.abs(np.mean(self.data)) < 1e-6 else 'potential'

    def is_diverging_data(self):
        """
        Check if data type uses diverging colormap (has meaningful positive/negative values).

        Returns True for: spin, potential, and all difference types
        """
        return self.data_type in ['spin', 'potential', 'spin_difference',
                                   'potential_difference', 'density_difference', 'difference']

    def get_data_type_label(self):
        """Get human-readable label for data type."""
        labels = {
            'density': 'Density',
            'spin': 'Spin Density',
            'potential': 'Potential',
            'density_difference': 'Density Difference (Δρ)',
            'potential_difference': 'Potential Difference (ΔV)',
            'spin_difference': 'Spin Difference (Δσ)',
            'difference': 'Difference'
        }
        return labels.get(self.data_type, self.data_type.capitalize())

    def _analyze_structure(self):
        """Analyze if structure is 2D or 3D based on cell and atomic positions."""
        # Calculate cell dimensions
        self.cell_dims = np.array([
            np.linalg.norm(self.lattice_vectors[0]),
            np.linalg.norm(self.lattice_vectors[1]),
            np.linalg.norm(self.lattice_vectors[2])
        ])
        
        # Use cell parameters if available
        if self.cell_parameters:
            c_param = self.cell_parameters.get('c', 0)
            if c_param > 400:  # Large c parameter indicates 2D material
                self.is_2d = True
                self.z_vacuum = c_param
        
        # Check z-extension of atoms
        z_positions = np.array([pos[2] for pos in self.atomic_positions])
        z_min, z_max = np.min(z_positions), np.max(z_positions)
        z_extent = z_max - z_min
        
        # If z-extent is small compared to cell_z, likely 2D
        if z_extent < 10.0 and self.cell_dims[2] > 20.0:  # in Bohr
            self.is_2d = True
            self.z_vacuum = self.cell_dims[2]
        
        # Fix atom wrapping for 2D materials
        if self.is_2d and self.wrap_atoms:
            self._fix_slab_wrapping()
    
    def _fix_slab_wrapping(self):
        """Fix wrapping issues for slab structures."""
        z_positions = np.array([pos[2] for pos in self.atomic_positions])
        cell_z = self.cell_dims[2]
        
        # Find the largest gap (vacuum region)
        sorted_indices = np.argsort(z_positions)
        sorted_z = z_positions[sorted_indices]
        
        # Find gaps
        gaps = []
        for i in range(len(sorted_z)):
            next_i = (i + 1) % len(sorted_z)
            gap = sorted_z[next_i] - sorted_z[i]
            if next_i == 0:
                gap += cell_z
            gaps.append(gap)
        
        # Find largest gap
        max_gap_idx = np.argmax(gaps)
        
        # Check if slab is split
        # if z_positions.max() - z_positions.min() > cell_z * 0.6:
        #     # Unwrap atoms
        #     for i, pos in enumerate(self.atomic_positions):
        #         if pos[2] < cell_z * 0.3:
        #             pos[2] += cell_z
    
    def _apply_manual_shift(self):
        """Apply manual z-shift to atoms."""
        shift_bohr = self.z_shift / 0.529177
        cell_z = self.cell_dims[2]

        for pos in self.atomic_positions:
            pos[2] = (pos[2] + shift_bohr) % cell_z

    def _detect_and_crop_vacuum(self, density_threshold=1e-8, padding_percent=0.05):
        """
        Vacuum detection: Find where ALL values become exactly 0.
        For each axis, find the first and last slices that contain any non-zero values.
        """
        # Calculate maximum absolute value for each slice along each axis
        x_profile = np.max(np.abs(self.data), axis=(1, 2))
        y_profile = np.max(np.abs(self.data), axis=(0, 2))
        z_profile = np.max(np.abs(self.data), axis=(0, 1))

        def find_bounds_exact(profile):
            """Find bounds where ALL values are zero beyond this point."""
            # Find indices where profile has ANY non-zero values
            # Use != 0 to handle both positive and negative values (SPIN/POT)
            nonzero = profile != 0
            if not np.any(nonzero):
                return 0, len(profile)

            nonzero_indices = np.where(nonzero)[0]
            start = nonzero_indices[0]
            end = nonzero_indices[-1] + 1
            return start, end

        # Calculate bounds for each axis
        self.data_bounds = {
            'x': find_bounds_exact(x_profile),
            'y': find_bounds_exact(y_profile),
            'z': find_bounds_exact(z_profile)
        }

        # Add minimal padding (2-3 voxels) for safety
        minimal_padding = 2
        for axis in ['x', 'y', 'z']:
            idx = ['x', 'y', 'z'].index(axis)
            n = self.nvoxels[idx]
            start, end = self.data_bounds[axis]

            self.data_bounds[axis] = (
                max(0, start - minimal_padding),
                min(n, end + minimal_padding)
            )

        # Calculate vacuum statistics
        vacuum_fractions = [
            1.0 - (end - start) / self.nvoxels[i]
            for i, (start, end) in enumerate([
                self.data_bounds['x'],
                self.data_bounds['y'],
                self.data_bounds['z']
            ])
        ]

        self.vacuum_detected = any(frac > 0.2 for frac in vacuum_fractions)

        # Report findings
        if self.vacuum_detected:
            print(f"\n{'='*60}")
            print("VACUUM DETECTED - Optimizing visualization bounds:")
            print(f"{'='*60}")
            for axis, idx in [('X', 0), ('Y', 1), ('Z', 2)]:
                start, end = self.data_bounds[axis.lower()]
                total = self.nvoxels[idx]
                saved = 100 * (1 - (end - start) / total)
                print(f"  {axis}-axis: [{start:4d} - {end:4d}] of {total:4d} "
                      f"({saved:5.1f}% vacuum removed)")
            print(f"{'='*60}\n")

        # Store cropped dimensions
        self.cropped_nvoxels = np.array([
            self.data_bounds['x'][1] - self.data_bounds['x'][0],
            self.data_bounds['y'][1] - self.data_bounds['y'][0],
            self.data_bounds['z'][1] - self.data_bounds['z'][0]
        ])

        # Calculate memory/performance benefit
        original_size = np.prod(self.nvoxels)
        cropped_size = np.prod(self.cropped_nvoxels)
        memory_saving = 100 * (1 - cropped_size / original_size)

        if memory_saving > 10:
            print(f"Memory footprint reduced by {memory_saving:.1f}% "
                  f"({original_size:,} → {cropped_size:,} voxels)\n")

    def _get_recommended_isovalues(self):
        """Calculate recommended isovalues based on data statistics."""
        nonzero_data = self.data[self.data != 0]
        
        if len(nonzero_data) == 0:
            self.recommended_iso = {
                'density': [1e-6, 1e-5, 1e-4],
                'potential': [-1e-4, -1e-5, 1e-5, 1e-4],
                'spin': [-1e-5, -1e-6, 1e-6, 1e-5]
            }.get(self.data_type, [1e-5])
            return
        
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        
        if self.data_type == 'density':
            # Log-spaced values for density
            if data_max > 0.1:
                self.recommended_iso = [0.001, 0.01, 0.05, 0.1]
            elif data_max > 0.01:
                self.recommended_iso = [0.0001, 0.001, 0.01, 0.05]
            else:
                self.recommended_iso = [1e-5, 1e-4, 0.001, 0.01]
        
        elif self.data_type == 'spin':
            # Symmetric values for spin density
            max_abs = np.max(np.abs(self.data))
            if max_abs < 1e-6:
                self.recommended_iso = [-1e-7, -1e-8, 1e-8, 1e-7]
            elif max_abs < 0.01:
                self.recommended_iso = [-0.001, -0.0001, 0.0001, 0.001]
            else:
                self.recommended_iso = [-0.01, -0.001, 0.001, 0.01]
        
        else:  # potential
            data_sorted = np.sort(self.data.flatten())
            n = len(data_sorted)
            data_trimmed = data_sorted[int(0.01*n):int(0.99*n)]
            
            p10 = np.percentile(data_trimmed, 10)
            p25 = np.percentile(data_trimmed, 25)
            p75 = np.percentile(data_trimmed, 75)
            p90 = np.percentile(data_trimmed, 90)
            
            self.recommended_iso = sorted([p10, p25, p75, p90])
    
    def get_cartesian_grid(self):
        """Generate 3D grid points in Cartesian coordinates using proper lattice vectors."""
        i = np.arange(self.nvoxels[0])
        j = np.arange(self.nvoxels[1])
        k = np.arange(self.nvoxels[2])
        
        I, J, K = np.meshgrid(i, j, k, indexing='ij')
        
        X = (self.origin[0] + 
             I * self.voxel_vectors[0, 0] + 
             J * self.voxel_vectors[1, 0] + 
             K * self.voxel_vectors[2, 0])
        
        Y = (self.origin[1] + 
             I * self.voxel_vectors[0, 1] + 
             J * self.voxel_vectors[1, 1] + 
             K * self.voxel_vectors[2, 1])
        
        Z = (self.origin[2] + 
             I * self.voxel_vectors[0, 2] + 
             J * self.voxel_vectors[1, 2] + 
             K * self.voxel_vectors[2, 2])
        
        return X, Y, Z
    
    def get_slice(self, axis: str = 'z', position: Optional[int] = None):
        """Extract a 2D slice from the 3D data."""
        if position is None:
            position = self.nvoxels[['x', 'y', 'z'].index(axis)] // 2
        
        if axis == 'x':
            return self.data[position, :, :]
        elif axis == 'y':
            return self.data[:, position, :]
        else:  # z
            return self.data[:, :, position]
    
    def get_slice_with_max_density(self, axis: str = 'z'):
        """Find the slice with maximum density/activity."""
        max_sum = -np.inf
        best_idx = 0

        n_slices = self.nvoxels[['x', 'y', 'z'].index(axis)]

        for i in range(n_slices):
            slice_data = self.get_slice(axis, i)
            slice_sum = np.sum(np.abs(slice_data))
            if slice_sum > max_sum:
                max_sum = slice_sum
                best_idx = i

        return best_idx

    def get_best_visualization_slice(self, axis: str = 'z'):
        """
        Find the best slice for visualization.

        Strategy:
        1. If vacuum detected, use middle of material region (cropped bounds)
        2. Otherwise, find slice with good density distribution (not max, not vacuum)
        3. For density data, avoid nuclear cusps by looking for bonding regions
        """
        axis_idx = ['x', 'y', 'z'].index(axis)
        n_slices = self.nvoxels[axis_idx]

        # Strategy 1: Use middle of material region if vacuum detected
        if hasattr(self, 'data_bounds') and self.vacuum_detected:
            bounds_key = axis
            start, end = self.data_bounds[bounds_key]
            middle = (start + end) // 2
            print(f"Using middle of material region: slice {middle} (material spans {start}-{end})")
            return middle

        # Strategy 2: For density, find slice with good spread (not max, not vacuum)
        if self.data_type == 'density':
            # Sample slices and find one with good density characteristics
            scores = []
            for i in range(n_slices):
                slice_data = self.get_slice(axis, i)

                # Look for slices with moderate density and good spread
                # Avoid: vacuum (low sum) and nuclear cusps (very high max)
                total = np.sum(slice_data)
                max_val = np.max(slice_data)

                # Good slice has reasonable total but not extreme maximum
                # This favors bonding regions over atomic positions
                if max_val > 1e-6 and total > 1e-3:
                    # Penalize extreme maxima (nuclear cusps)
                    score = total / (1.0 + max_val**2 / 100.0)
                    scores.append((score, i))

            if scores:
                scores.sort(reverse=True)
                best_idx = scores[0][1]
                print(f"Using slice with good bonding density: slice {best_idx}")
                return best_idx

        # Fallback: use middle slice
        middle = n_slices // 2
        print(f"Using middle slice: {middle}")
        return middle
    
    def print_info(self):
        """Print summary information about the cube file."""
        print(f"\nCube File: {self.filename}")
        print(f"Data type: {self.data_type}")

        # Warn about extremely small data ranges
        data_range = self.data.max() - self.data.min()
        abs_max = np.max(np.abs(self.data))
        if abs_max > 0 and abs_max < 1e-6:
            print(f"\n{'='*60}")
            print(f"WARNING: Extremely small data values detected!")
            print(f"{'='*60}")
            print(f"  Data range: {self.data.min():.3e} to {self.data.max():.3e}")
            print(f"  Max absolute value: {abs_max:.3e}")
            print(f"")
            if abs_max < 1e-10:
                print(f"  This is very close to numerical noise (< 1e-10).")
            else:
                print(f"  These values are very small (< 1e-6).")
            print(f"  Visualizations will show these tiny values, but they may")
            print(f"  represent numerical artifacts rather than physical signal.")
            print(f"  Isosurface values will be in the range ~1e-12 to ~1e-9.")
            print(f"{'='*60}\n")
        print(f"Structure type: {'2D material' if self.is_2d else '3D material'}")
        print(f"Crystal system: {self.crystal_system}")
        print(f"Number of atoms: {abs(self.natoms)}")
        print(f"Grid points: {self.nvoxels[0]} × {self.nvoxels[1]} × {self.nvoxels[2]}")
        
        if self.cell_parameters:
            print(f"Cell parameters:")
            print(f"  a={self.cell_parameters['a']:.3f} b={self.cell_parameters['b']:.3f} "
                  f"c={self.cell_parameters['c']:.3f} Bohr")
            print(f"  α={self.cell_parameters['alpha']:.1f}° β={self.cell_parameters['beta']:.1f}° "
                  f"γ={self.cell_parameters['gamma']:.1f}°")
        
        nonzero = self.data[self.data != 0]
        if len(nonzero) > 0:
            print(f"Data range: [{np.min(self.data):.6f}, {np.max(self.data):.6f}]")
            print(f"Recommended isovalues: {', '.join([f'{x:.2e}' for x in self.recommended_iso])}")
        else:
            print("Warning: Data contains all zeros!")


def generate_smart_isovalues(data, data_type='density', n_values=20,
                            include_half_decades=True):
    """
    Generate intelligently-spaced isovalues.

    For density: Uses logarithmic spacing with extra points at 1×10^n, 2×10^n, 5×10^n
    For potential/spin: Uses symmetric asinh spacing (like log but defined at 0)

    Parameters
    ----------
    data : np.ndarray
        3D data array
    data_type : str
        Type of data: 'density', 'spin', or 'potential'
    n_values : int
        Target number of isovalues
    include_half_decades : bool
        If True, includes values at 2× and 5× in addition to 1×10^n

    Returns
    -------
    list
        Sorted list of isovalues
    """

    if data_type == 'density':
        # Find data range - use tighter percentiles to focus on interesting regime
        # Exclude extreme outliers (vacuum noise and nuclear cusps)
        pos_data = data[data > 1e-10]
        if len(pos_data) < 10:
            return [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        # Use 5% percentile for minimum to exclude vacuum noise
        data_min = np.percentile(pos_data, 5.0)

        # Ensure minimum is not too small (at least 1e-4 for useful isosurfaces)
        # Values below 1e-4 are typically just vacuum noise
        data_min = max(data_min, 1e-4)

        # Set maximum to 2.0 for chemically interesting regime
        # This covers bonding/valence density and excludes extreme nuclear cusps
        # Range 1e-4 to 2.0 captures most important features for visualization
        data_max = 2.0

        # Calculate decade range
        log_min = np.floor(np.log10(data_min))
        log_max = np.ceil(np.log10(data_max))

        isovalues = []

        if include_half_decades:
            # Pattern: 1e-n, 2e-n, 5e-n for each decade
            for exponent in np.arange(log_min, log_max + 0.1, 1.0):
                base = 10 ** exponent
                candidates = [1.0 * base, 2.0 * base, 5.0 * base]
                for val in candidates:
                    if data_min <= val <= data_max:
                        isovalues.append(val)
        else:
            # Standard log spacing
            isovalues = np.logspace(log_min, log_max, n_values).tolist()

        # Ensure we don't exceed requested count
        if len(isovalues) > n_values:
            # Keep evenly spaced subset
            indices = np.linspace(0, len(isovalues) - 1, n_values).astype(int)
            isovalues = [isovalues[i] for i in indices]

        return sorted(isovalues)

    elif data_type in ['spin', 'potential', 'spin_difference', 'potential_difference', 'density_difference', 'difference']:
        # Special handling for density_difference - use SMALL symmetric values
        if data_type in ['density_difference', 'difference']:
            # For density difference, focus on small changes (not nuclear cusps!)
            # Use symmetric logarithmic spacing from 1e-4 to 1 e/Bohr³
            # This covers physisorption (0.001) to chemisorption (0.1) regimes

            exponents = np.linspace(-4, 0, n_values // 2)  # -4 to 0 gives 1e-4 to 1
            positive_vals = 10 ** exponents

            # Return only positive values - visualization will show both +/-
            return sorted(positive_vals.tolist())

        # Check if data is truly symmetric or mostly one-sided
        data_min = np.percentile(data, 0.5)
        data_max = np.percentile(data, 99.5)

        # Check for asymmetry: if one side is >10x larger than the other, treat as one-sided
        # This handles cases like [-0.1, 1000] which are essentially one-sided
        abs_min = abs(data_min)
        abs_max = abs(data_max)
        asymmetry_ratio = max(abs_min, abs_max) / (min(abs_min, abs_max) + 1e-10)

        # For truly symmetric data (spin), only generate POSITIVE values
        # The visualization code will show both +/- surfaces
        is_symmetric = not (data_min >= 0 or data_max <= 0 or asymmetry_ratio > 10)

        # If data is mostly positive or mostly negative (asymmetric)
        if not is_symmetric:
            # Treat as density-like (one-sided data)
            # Determine which side is dominant
            if data_min >= 0 or (asymmetry_ratio > 10 and abs_max > abs_min):
                # All positive - use log spacing like density
                pos_data = data[data > 1e-10]
                if len(pos_data) < 10:
                    return [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

                data_min = np.percentile(pos_data, 0.1)
                data_max = np.percentile(pos_data, 99.9)
                log_min = np.floor(np.log10(data_min))
                log_max = np.ceil(np.log10(data_max))

                isovalues = []
                for exponent in np.arange(log_min, log_max + 0.1, 1.0):
                    base = 10 ** exponent
                    candidates = [1.0 * base, 2.0 * base, 5.0 * base]
                    for val in candidates:
                        if data_min <= val <= data_max:
                            isovalues.append(val)

                if len(isovalues) > n_values:
                    indices = np.linspace(0, len(isovalues) - 1, n_values).astype(int)
                    isovalues = [isovalues[i] for i in indices]

                return sorted(isovalues)
            else:
                # All negative - use negative log spacing
                neg_data = data[data < -1e-10]
                if len(neg_data) < 10:
                    return [-1e-2, -5e-3, -1e-3, -5e-4, -1e-4, -5e-5, -1e-5]

                abs_min = np.percentile(np.abs(neg_data), 0.1)
                abs_max = np.percentile(np.abs(neg_data), 99.9)
                log_min = np.floor(np.log10(abs_min))
                log_max = np.ceil(np.log10(abs_max))

                isovalues = []
                for exponent in np.arange(log_min, log_max + 0.1, 1.0):
                    base = 10 ** exponent
                    candidates = [1.0 * base, 2.0 * base, 5.0 * base]
                    for val in candidates:
                        if abs_min <= val <= abs_max:
                            isovalues.append(-val)

                if len(isovalues) > n_values:
                    indices = np.linspace(0, len(isovalues) - 1, n_values).astype(int)
                    isovalues = [isovalues[i] for i in indices]

                return sorted(isovalues, reverse=True)
        else:
            # Truly symmetric data (spin/potential) - generate only POSITIVE values
            # Visualization will show both +/- surfaces (red/blue) for each value
            # CRITICAL FIX (BUG 4): Check both positive and negative maxima separately
            # For spin data, spin-up and spin-down might have different magnitudes
            # We need to ensure both sides are properly represented
            data_max_pos = np.max(data[data > 0]) if np.any(data > 0) else 0
            data_max_neg = abs(np.min(data[data < 0])) if np.any(data < 0) else 0

            # Use the larger of the two to ensure both sides are visible
            # This ensures symmetric display as confirmed by user in VESTA
            abs_max = max(data_max_pos, data_max_neg)

            # Use some reasonable isovalues spanning the range
            # Generate values that give good coverage from small to large
            if abs_max > 1e-10:
                # Use logarithmic spacing for better coverage
                # Start from 0.001 * abs_max to abs_max
                log_min = np.log10(abs_max * 0.001)
                log_max = np.log10(abs_max)
                isovalues = np.logspace(log_min, log_max, n_values).tolist()
            else:
                # For very small values, use linear spacing
                isovalues = np.linspace(abs_max * 0.1, abs_max, n_values).tolist()

            return sorted(isovalues)

    else:
        # Fallback to linear
        return np.linspace(np.min(data), np.max(data), n_values).tolist()


def get_plotly_colorscale(data_type: str, cmap: Optional[str] = None):
    """Get appropriate Plotly colorscale for the data type."""
    if cmap:
        return cmap
    
    colorscales = {
        'density': 'Viridis',
        'spin': 'RdBu',
        'potential': 'RdBu'
    }
    return colorscales.get(data_type, 'Viridis')


def get_atom_properties():
    """Get atom visualization properties (VESTA/CPK color scheme)."""
    return {
        # Period 1
        1: {'color': 'white', 'name': 'H', 'size': 8},
        2: {'color': 'cyan', 'name': 'He', 'size': 8},
        # Period 2
        3: {'color': 'violet', 'name': 'Li', 'size': 12},
        4: {'color': 'darkgreen', 'name': 'Be', 'size': 10},
        5: {'color': 'pink', 'name': 'B', 'size': 10},
        6: {'color': 'black', 'name': 'C', 'size': 11},
        7: {'color': 'blue', 'name': 'N', 'size': 10},
        8: {'color': 'red', 'name': 'O', 'size': 10},
        9: {'color': 'green', 'name': 'F', 'size': 10},
        10: {'color': 'cyan', 'name': 'Ne', 'size': 10},
        # Period 3
        11: {'color': 'purple', 'name': 'Na', 'size': 13},
        12: {'color': 'darkgreen', 'name': 'Mg', 'size': 12},
        13: {'color': 'gray', 'name': 'Al', 'size': 12},
        14: {'color': 'tan', 'name': 'Si', 'size': 14},
        15: {'color': 'orange', 'name': 'P', 'size': 11},
        16: {'color': 'yellow', 'name': 'S', 'size': 13},
        17: {'color': 'lightgreen', 'name': 'Cl', 'size': 11},
        18: {'color': 'cyan', 'name': 'Ar', 'size': 12},
        # Period 4
        19: {'color': 'purple', 'name': 'K', 'size': 15},
        20: {'color': 'gray', 'name': 'Ca', 'size': 14},
        21: {'color': 'lightgray', 'name': 'Sc', 'size': 13},
        22: {'color': 'silver', 'name': 'Ti', 'size': 13},
        23: {'color': 'gray', 'name': 'V', 'size': 12},
        24: {'color': 'silver', 'name': 'Cr', 'size': 12},
        25: {'color': 'purple', 'name': 'Mn', 'size': 12},
        26: {'color': 'darkorange', 'name': 'Fe', 'size': 13},
        27: {'color': 'pink', 'name': 'Co', 'size': 12},
        28: {'color': 'green', 'name': 'Ni', 'size': 12},
        29: {'color': 'brown', 'name': 'Cu', 'size': 13},
        30: {'color': 'brown', 'name': 'Zn', 'size': 13},
        31: {'color': 'silver', 'name': 'Ga', 'size': 13},
        32: {'color': 'gray', 'name': 'Ge', 'size': 13},
        33: {'color': 'purple', 'name': 'As', 'size': 12},
        34: {'color': 'orange', 'name': 'Se', 'size': 14},
        35: {'color': 'brown', 'name': 'Br', 'size': 12},
        36: {'color': 'cyan', 'name': 'Kr', 'size': 12},
        # Period 5
        37: {'color': 'purple', 'name': 'Rb', 'size': 15},
        38: {'color': 'yellow', 'name': 'Sr', 'size': 14},
        39: {'color': 'silver', 'name': 'Y', 'size': 13},
        40: {'color': 'silver', 'name': 'Zr', 'size': 13},
        41: {'color': 'gray', 'name': 'Nb', 'size': 13},
        42: {'color': 'green', 'name': 'Mo', 'size': 13},
        43: {'color': 'silver', 'name': 'Tc', 'size': 13},
        44: {'color': 'cyan', 'name': 'Ru', 'size': 13},
        45: {'color': 'silver', 'name': 'Rh', 'size': 13},
        46: {'color': 'silver', 'name': 'Pd', 'size': 13},
        47: {'color': 'silver', 'name': 'Ag', 'size': 14},
        48: {'color': 'gray', 'name': 'Cd', 'size': 14},
        49: {'color': 'silver', 'name': 'In', 'size': 14},
        50: {'color': 'gray', 'name': 'Sn', 'size': 15},
        51: {'color': 'purple', 'name': 'Sb', 'size': 14},
        52: {'color': 'orange', 'name': 'Te', 'size': 15},
        53: {'color': 'purple', 'name': 'I', 'size': 14},
        54: {'color': 'cyan', 'name': 'Xe', 'size': 14},
        # Period 6
        55: {'color': 'purple', 'name': 'Cs', 'size': 16},
        56: {'color': 'green', 'name': 'Ba', 'size': 15},
        57: {'color': 'lightblue', 'name': 'La', 'size': 14},
        58: {'color': 'lightblue', 'name': 'Ce', 'size': 14},
        59: {'color': 'lightblue', 'name': 'Pr', 'size': 14},
        60: {'color': 'lightblue', 'name': 'Nd', 'size': 14},
        61: {'color': 'lightblue', 'name': 'Pm', 'size': 14},
        62: {'color': 'lightblue', 'name': 'Sm', 'size': 14},
        63: {'color': 'lightblue', 'name': 'Eu', 'size': 14},
        64: {'color': 'lightblue', 'name': 'Gd', 'size': 14},
        65: {'color': 'lightblue', 'name': 'Tb', 'size': 14},
        66: {'color': 'lightblue', 'name': 'Dy', 'size': 14},
        67: {'color': 'lightblue', 'name': 'Ho', 'size': 14},
        68: {'color': 'lightblue', 'name': 'Er', 'size': 14},
        69: {'color': 'lightblue', 'name': 'Tm', 'size': 14},
        70: {'color': 'lightblue', 'name': 'Yb', 'size': 14},
        71: {'color': 'lightblue', 'name': 'Lu', 'size': 14},
        72: {'color': 'silver', 'name': 'Hf', 'size': 13},
        73: {'color': 'silver', 'name': 'Ta', 'size': 13},
        74: {'color': 'silver', 'name': 'W', 'size': 13},
        75: {'color': 'silver', 'name': 'Re', 'size': 13},
        76: {'color': 'silver', 'name': 'Os', 'size': 13},
        77: {'color': 'silver', 'name': 'Ir', 'size': 13},
        78: {'color': 'silver', 'name': 'Pt', 'size': 13},
        79: {'color': 'gold', 'name': 'Au', 'size': 14},
        80: {'color': 'silver', 'name': 'Hg', 'size': 14},
        81: {'color': 'silver', 'name': 'Tl', 'size': 14},
        82: {'color': 'gray', 'name': 'Pb', 'size': 15},
        83: {'color': 'purple', 'name': 'Bi', 'size': 15},
        84: {'color': 'orange', 'name': 'Po', 'size': 15},
        85: {'color': 'purple', 'name': 'At', 'size': 14},
        86: {'color': 'cyan', 'name': 'Rn', 'size': 14},
        # Period 7
        87: {'color': 'purple', 'name': 'Fr', 'size': 16},
        88: {'color': 'green', 'name': 'Ra', 'size': 15},
        89: {'color': 'lightblue', 'name': 'Ac', 'size': 14},
        90: {'color': 'lightblue', 'name': 'Th', 'size': 14},
        91: {'color': 'lightblue', 'name': 'Pa', 'size': 14},
        92: {'color': 'lightblue', 'name': 'U', 'size': 14},
        93: {'color': 'lightblue', 'name': 'Np', 'size': 14},
        94: {'color': 'lightblue', 'name': 'Pu', 'size': 14},
        95: {'color': 'lightblue', 'name': 'Am', 'size': 14},
        96: {'color': 'lightblue', 'name': 'Cm', 'size': 14},
        97: {'color': 'lightblue', 'name': 'Bk', 'size': 14},
        98: {'color': 'lightblue', 'name': 'Cf', 'size': 14},
        99: {'color': 'lightblue', 'name': 'Es', 'size': 14},
        100: {'color': 'lightblue', 'name': 'Fm', 'size': 14},
        101: {'color': 'lightblue', 'name': 'Md', 'size': 14},
        102: {'color': 'lightblue', 'name': 'No', 'size': 14},
        103: {'color': 'lightblue', 'name': 'Lr', 'size': 14},
        104: {'color': 'silver', 'name': 'Rf', 'size': 13},
        105: {'color': 'silver', 'name': 'Db', 'size': 13},
        106: {'color': 'silver', 'name': 'Sg', 'size': 13},
        107: {'color': 'silver', 'name': 'Bh', 'size': 13},
        108: {'color': 'silver', 'name': 'Hs', 'size': 13},
        109: {'color': 'silver', 'name': 'Mt', 'size': 13},
        110: {'color': 'silver', 'name': 'Ds', 'size': 13},
        111: {'color': 'silver', 'name': 'Rg', 'size': 13},
        112: {'color': 'silver', 'name': 'Cn', 'size': 13},
        113: {'color': 'silver', 'name': 'Nh', 'size': 13},
        114: {'color': 'silver', 'name': 'Fl', 'size': 13},
        115: {'color': 'silver', 'name': 'Mc', 'size': 13},
        116: {'color': 'silver', 'name': 'Lv', 'size': 13},
        117: {'color': 'silver', 'name': 'Ts', 'size': 13},
        118: {'color': 'cyan', 'name': 'Og', 'size': 13},
    }


def export_figure_to_gif(fig, output_path, fps=5, width=800, height=600, loop=0):
    """
    Export a Plotly figure with frames to an animated GIF.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure with frames (from slice-browse or isosurface browser)
    output_path : str
        Path to save GIF file
    fps : int
        Frames per second (default: 5)
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    loop : int
        Number of loops (0 = infinite loop)
    """
    if not HAS_KALEIDO:
        print("ERROR: kaleido is required for GIF export")
        print("Install with: pip install kaleido")
        return False

    if not HAS_PIL:
        print("ERROR: Pillow is required for GIF creation")
        print("Install with: pip install Pillow")
        return False

    import tempfile
    import shutil
    from pathlib import Path
    import copy

    print(f"\n{'='*60}")
    print(f"Exporting GIF: {output_path}")
    print(f"{'='*60}")
    print(f"  Frames: {len(fig.frames)}")
    print(f"  FPS: {fps}")
    print(f"  Size: {width}x{height}")

    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp(prefix='cube_gif_')

    try:
        frame_paths = []

        # Export each frame as PNG
        for i, frame in enumerate(fig.frames):
            if i % max(1, len(fig.frames) // 10) == 0 or i == len(fig.frames) - 1:
                print(f"  Rendering frame {i+1}/{len(fig.frames)}...")

            # Create a copy of the layout and update slider position
            temp_layout = copy.deepcopy(fig.layout)
            if hasattr(temp_layout, 'sliders') and temp_layout.sliders:
                # Update the active step for the slider to match current frame
                temp_layout.sliders[0].active = i

            # Create a temporary figure with this frame's data and updated layout
            temp_fig = go.Figure(data=frame.data, layout=temp_layout)

            # Export to PNG
            frame_path = Path(temp_dir) / f"frame_{i:04d}.png"
            temp_fig.write_image(str(frame_path), width=width, height=height, format='png')
            frame_paths.append(frame_path)

        # Create GIF from frames
        print(f"  Creating GIF...")
        images = [Image.open(str(path)) for path in frame_paths]

        # Calculate duration in milliseconds
        duration = int(1000 / fps)

        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=False  # Faster, larger file size
        )

        print(f"✓ Saved: {output_path}")
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        print(f"{'='*60}\n")

        return True

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_rotation_gif(fig, output_path, rotate_axis='z', n_frames=36, fps=10,
                       width=800, height=800):
    """
    Create a GIF of the figure rotating around a specified axis.

    Parameters
    ----------
    fig : go.Figure
        Plotly 3D figure (isosurface or 3D visualization)
    output_path : str
        Path to save GIF file
    rotate_axis : str
        Axis to rotate around: 'x', 'y', or 'z' (default: 'z')
    n_frames : int
        Number of frames for full 360° rotation (default: 36 = 10° per frame)
    fps : int
        Frames per second (default: 10)
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    """
    if not HAS_KALEIDO:
        print("ERROR: kaleido is required for GIF export")
        print("Install with: pip install kaleido")
        return False

    if not HAS_PIL:
        print("ERROR: Pillow is required for GIF creation")
        print("Install with: pip install Pillow")
        return False

    import tempfile
    import shutil
    from pathlib import Path
    import math

    print(f"\n{'='*60}")
    print(f"Creating Rotation GIF: {output_path}")
    print(f"{'='*60}")
    print(f"  Rotation axis: {rotate_axis.upper()}")
    print(f"  Frames: {n_frames} (360° /{n_frames} = {360/n_frames:.1f}° per frame)")
    print(f"  FPS: {fps}")

    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp(prefix='cube_rotation_')

    try:
        frame_paths = []

        # Camera distance from origin
        camera_distance = 2.5

        # Generate camera positions for rotation
        for i in range(n_frames):
            if i % max(1, n_frames // 10) == 0 or i == n_frames - 1:
                print(f"  Rendering frame {i+1}/{n_frames}...")

            angle = 2 * math.pi * i / n_frames

            # Calculate camera eye position based on rotation axis
            if rotate_axis.lower() == 'z':
                # Rotate in XY plane (around Z axis)
                eye_x = camera_distance * math.cos(angle)
                eye_y = camera_distance * math.sin(angle)
                eye_z = 1.2
            elif rotate_axis.lower() == 'y':
                # Rotate in XZ plane (around Y axis)
                eye_x = camera_distance * math.cos(angle)
                eye_y = 1.5
                eye_z = camera_distance * math.sin(angle)
            else:  # 'x'
                # Rotate in YZ plane (around X axis)
                eye_x = 1.5
                eye_y = camera_distance * math.cos(angle)
                eye_z = camera_distance * math.sin(angle)

            # Update camera position
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=eye_x, y=eye_y, z=eye_z),
                        up=dict(x=0, y=0, z=1)
                    )
                )
            )

            # Export frame
            frame_path = Path(temp_dir) / f"frame_{i:04d}.png"
            fig.write_image(str(frame_path), width=width, height=height, format='png')
            frame_paths.append(frame_path)

        # Create GIF from frames
        print(f"  Creating GIF...")
        images = [Image.open(str(path)) for path in frame_paths]

        duration = int(1000 / fps)

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,  # Infinite loop
            optimize=False
        )

        print(f"✓ Saved: {output_path}")
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        print(f"{'='*60}\n")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def check_grid_alignment_compatibility(cube1: CubeFile, cube2: CubeFile,
                                      tolerance=1e-6, max_spacing_diff=0.05):
    """
    Check if two CUBE files can be aligned (with or without interpolation).

    Parameters
    ----------
    cube1, cube2 : CubeFile
        CUBE file objects to compare
    tolerance : float
        Tolerance for exact voxel vector match (no interpolation needed)
    max_spacing_diff : float
        Maximum fractional difference in voxel spacing allowed for interpolation (default: 5%)

    Returns
    -------
    bool
        True if grids can be aligned
    str
        Error message if not alignable, empty string if alignable
    bool
        True if interpolation is needed
    """
    # Check if voxel vectors match exactly (no interpolation needed)
    if np.allclose(cube1.voxel_vectors, cube2.voxel_vectors, atol=tolerance):
        return True, "", False

    # Voxel vectors don't match exactly - check if interpolation can help
    # Calculate voxel spacing for each dimension
    spacing1 = np.array([np.linalg.norm(cube1.voxel_vectors[i]) for i in range(3)])
    spacing2 = np.array([np.linalg.norm(cube2.voxel_vectors[i]) for i in range(3)])

    # Check if voxel directions are the same (just spacing differs)
    directions_match = True
    for i in range(3):
        dir1 = cube1.voxel_vectors[i] / spacing1[i]
        dir2 = cube2.voxel_vectors[i] / spacing2[i]
        if not np.allclose(dir1, dir2, atol=tolerance):
            directions_match = False
            break

    if not directions_match:
        error_msg = "Voxel vectors have different orientations (cannot align with interpolation)\n"
        error_msg += f"  CUBE 0 voxel vectors:\n"
        for i in range(3):
            error_msg += f"    v{i}: [{cube1.voxel_vectors[i,0]:10.6f}, {cube1.voxel_vectors[i,1]:10.6f}, {cube1.voxel_vectors[i,2]:10.6f}] Bohr\n"
        error_msg += f"  CUBE 1 voxel vectors:\n"
        for i in range(3):
            error_msg += f"    v{i}: [{cube2.voxel_vectors[i,0]:10.6f}, {cube2.voxel_vectors[i,1]:10.6f}, {cube2.voxel_vectors[i,2]:10.6f}] Bohr\n"
        return False, error_msg.rstrip(), False

    # Directions match - check if spacing difference is within acceptable range
    spacing_diff = np.abs(spacing2 - spacing1) / spacing1
    max_diff = np.max(spacing_diff)

    if max_diff > max_spacing_diff:
        error_msg = f"Voxel spacing difference too large ({max_diff*100:.1f}% > {max_spacing_diff*100:.1f}%)\n"
        error_msg += f"  Voxel spacing (Å):\n"
        error_msg += f"    CUBE 0: [{spacing1[0]*0.529177:.4f}, {spacing1[1]*0.529177:.4f}, {spacing1[2]*0.529177:.4f}]\n"
        error_msg += f"    CUBE 1: [{spacing2[0]*0.529177:.4f}, {spacing2[1]*0.529177:.4f}, {spacing2[2]*0.529177:.4f}]\n"
        error_msg += f"  Difference: [{spacing_diff[0]*100:.2f}%, {spacing_diff[1]*100:.2f}%, {spacing_diff[2]*100:.2f}%]\n"
        return False, error_msg.rstrip(), False

    # Can align with interpolation
    return True, "", True


def align_cube_grids(cubes: list, tolerance=1e-6):
    """
    Align CUBE grids with different origins, sizes, and/or spacing onto a unified grid.

    This function handles three cases:
    1. Same spacing, same origin: Simple padding
    2. Same spacing, different origin: Coordinate transformation + padding
    3. Different spacing: Interpolation onto unified grid

    Parameters
    ----------
    cubes : list of CubeFile
        List of CUBE file objects with potentially different grids
    tolerance : float
        Tolerance for floating-point comparisons

    Returns
    -------
    list of CubeFile
        List of CUBE objects with aligned grids on unified coordinate system
    """
    if len(cubes) < 2:
        return cubes

    print(f"\n{'='*60}")
    print("GRID ALIGNMENT")
    print(f"{'='*60}")

    # Check compatibility and determine which cubes need interpolation
    print("Checking alignment compatibility...")
    needs_interpolation = []
    for i in range(1, len(cubes)):
        compatible, error_msg, interp_needed = check_grid_alignment_compatibility(
            cubes[0], cubes[i], tolerance)
        if not compatible:
            raise ValueError(f"CUBE {i} cannot be aligned with CUBE 0: {error_msg}")
        needs_interpolation.append(interp_needed)

    # All compatible
    if any(needs_interpolation):
        print("✓ Grids alignable (interpolation required for some CUBEs)")
    else:
        print("✓ All grids have same voxel spacing")

    # Calculate spatial extent of each CUBE in real space
    print(f"\nCalculating spatial extents...")
    extents = []
    for i, cube in enumerate(cubes):
        # Calculate end corner: origin + nvoxels * voxel_vectors
        # For orthogonal grids: end = origin + nvoxels * diagonal(voxel_vectors)
        # For non-orthogonal: need matrix multiplication
        end_corner = cube.origin.copy()
        for dim in range(3):
            end_corner += cube.nvoxels[dim] * cube.voxel_vectors[dim]

        # Find min and max along each axis
        min_corner = cube.origin
        max_corner = end_corner

        extents.append({
            'min': min_corner,
            'max': max_corner,
            'origin': cube.origin,
            'nvoxels': cube.nvoxels
        })

        print(f"  CUBE {i}:")
        print(f"    Origin: [{cube.origin[0]:8.3f}, {cube.origin[1]:8.3f}, {cube.origin[2]:8.3f}] Bohr")
        print(f"    Extent: [{min_corner[0]:8.3f}, {min_corner[1]:8.3f}, {min_corner[2]:8.3f}] to")
        print(f"            [{max_corner[0]:8.3f}, {max_corner[1]:8.3f}, {max_corner[2]:8.3f}] Bohr")

    # Find unified bounding box
    all_mins = np.array([e['min'] for e in extents])
    all_maxs = np.array([e['max'] for e in extents])
    unified_min = all_mins.min(axis=0)
    unified_max = all_maxs.max(axis=0)

    print(f"\n  Unified extent:")
    print(f"    [{unified_min[0]:8.3f}, {unified_min[1]:8.3f}, {unified_min[2]:8.3f}] to")
    print(f"    [{unified_max[0]:8.3f}, {unified_max[1]:8.3f}, {unified_max[2]:8.3f}] Bohr")

    # Calculate unified grid dimensions
    # Use first cube's voxel vectors as reference (all should be same)
    voxel_vectors = cubes[0].voxel_vectors

    # For non-orthogonal grids, extent-based calculation can fail
    # Use maximum of original grid dimensions instead
    all_nvoxels = np.array([cube.nvoxels for cube in cubes])
    unified_nvoxels = all_nvoxels.max(axis=0)

    print(f"\n  Unified grid: {unified_nvoxels[0]} × {unified_nvoxels[1]} × {unified_nvoxels[2]}")
    print(f"    (Using maximum of input grid dimensions)")

    # Align each CUBE to the unified grid
    print(f"\nAligning CUBEs to unified grid...")
    aligned_cubes = []

    # Import scipy for interpolation if needed
    if any(needs_interpolation):
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:
            raise ImportError("scipy is required for grid interpolation. Install with: pip install scipy")

    for idx, cube in enumerate(cubes):
        print(f"\n  CUBE {idx}:")

        # Check if this cube needs interpolation
        # 1. Different voxel spacing (from needs_interpolation list)
        # 2. OR grid dimensions changed (non-orthogonal grid compaction)
        need_interp = needs_interpolation[idx-1] if idx > 0 else False

        # Also check if grid dimensions differ (even with same spacing)
        if not np.array_equal(cube.nvoxels, unified_nvoxels):
            need_interp = True
            if idx == 0:
                print(f"    Note: Grid dimensions changed ({cube.nvoxels} → {unified_nvoxels}), using interpolation")

        if need_interp:
            print(f"    Method: Interpolation (voxel spacing differs)")

            # Create interpolator for this cube's data
            x_coords_orig = np.arange(cube.nvoxels[0])
            y_coords_orig = np.arange(cube.nvoxels[1])
            z_coords_orig = np.arange(cube.nvoxels[2])

            interpolator = RegularGridInterpolator(
                (x_coords_orig, y_coords_orig, z_coords_orig),
                cube.data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            # Create unified grid
            unified_data = np.zeros(unified_nvoxels, dtype=cube.data.dtype)

            # Calculate grid points in unified coordinate system
            # We need to map unified grid points back to this cube's grid indices
            print(f"    Interpolating onto unified grid...")

            for i in range(unified_nvoxels[0]):
                if i % max(1, unified_nvoxels[0] // 10) == 0:
                    progress = 100 * i / unified_nvoxels[0]
                    print(f"      Progress: {progress:.0f}%", end='\r')

                for j in range(unified_nvoxels[1]):
                    for k in range(unified_nvoxels[2]):
                        # Real-space position of this unified grid point
                        pos_real = unified_min.copy()
                        for dim in range(3):
                            if dim == 0:
                                pos_real += i * voxel_vectors[dim]
                            elif dim == 1:
                                pos_real += j * voxel_vectors[dim]
                            else:
                                pos_real += k * voxel_vectors[dim]

                        # Convert to this cube's grid indices
                        grid_idx = np.zeros(3)
                        for dim in range(3):
                            cube_spacing = np.linalg.norm(cube.voxel_vectors[dim])
                            offset = np.dot(pos_real - cube.origin, cube.voxel_vectors[dim] / cube_spacing)
                            grid_idx[dim] = offset / cube_spacing

                        # Interpolate
                        interp_result = interpolator(grid_idx)
                        unified_data[i, j, k] = interp_result.item() if hasattr(interp_result, 'item') else float(interp_result)

            print(f"      Progress: 100%")

        else:
            print(f"    Method: Direct mapping (no interpolation needed)")

            # Calculate offset in grid indices
            offset_indices = np.zeros(3, dtype=int)
            for dim in range(3):
                spacing = np.linalg.norm(voxel_vectors[dim])
                offset = np.dot(cube.origin - unified_min, voxel_vectors[dim] / spacing)
                offset_indices[dim] = int(np.round(offset / spacing))

            print(f"    Grid offset: [{offset_indices[0]}, {offset_indices[1]}, {offset_indices[2]}]")

            # Create unified grid filled with zeros
            unified_data = np.zeros(unified_nvoxels, dtype=cube.data.dtype)

            # Copy cube's data to correct position
            i_start = offset_indices[0]
            j_start = offset_indices[1]
            k_start = offset_indices[2]
            i_end = i_start + cube.nvoxels[0]
            j_end = j_start + cube.nvoxels[1]
            k_end = k_start + cube.nvoxels[2]

            # Ensure we don't exceed unified grid bounds
            i_end = min(i_end, unified_nvoxels[0])
            j_end = min(j_end, unified_nvoxels[1])
            k_end = min(k_end, unified_nvoxels[2])

            unified_data[i_start:i_end, j_start:j_end, k_start:k_end] = cube.data[
                :i_end-i_start, :j_end-j_start, :k_end-k_start
            ]

        # Calculate statistics
        n_original = np.prod(cube.nvoxels)
        n_unified = np.prod(unified_nvoxels)

        print(f"    Original voxels: {n_original:,}")
        print(f"    Unified voxels:  {n_unified:,}")
        if need_interp:
            print(f"    Method result:   Interpolated to unified grid")
        else:
            n_padded = n_unified - n_original
            if n_padded > 0:
                pct_padded = 100 * n_padded / n_unified
                print(f"    Added zeros:     {n_padded:,} ({pct_padded:.1f}%)")
            elif n_padded < 0:
                print(f"    Grid compacted:  {-n_padded:,} fewer voxels (non-orthogonal lattice)")
            else:
                print(f"    Grid unchanged:  Perfect match")

        # Create new aligned CUBE object
        aligned_cube = CubeFile.__new__(CubeFile)
        aligned_cube.__dict__ = cube.__dict__.copy()
        aligned_cube.data = unified_data
        aligned_cube.nvoxels = unified_nvoxels.copy()
        aligned_cube.origin = unified_min.copy()
        aligned_cube.voxel_vectors = voxel_vectors.copy()

        aligned_cubes.append(aligned_cube)

    print(f"\n✓ All grids aligned to unified coordinate system")
    print(f"{'='*60}\n")

    return aligned_cubes


def validate_cube_compatibility(cube1: CubeFile, cube2: CubeFile, tolerance=1e-6):
    """
    Check if two CUBE files are compatible for dual visualization.

    Parameters
    ----------
    cube1, cube2 : CubeFile
        CUBE file objects to compare
    tolerance : float
        Tolerance for floating-point comparisons

    Returns
    -------
    bool
        True if compatible
    str
        Error message if incompatible, empty string if compatible
    """
    # Check grid dimensions
    if not np.array_equal(cube1.nvoxels, cube2.nvoxels):
        return False, f"Grid dimensions mismatch: {cube1.nvoxels} vs {cube2.nvoxels}"

    # Check origin
    if not np.allclose(cube1.origin, cube2.origin, atol=tolerance):
        return False, f"Origin mismatch: {cube1.origin} vs {cube2.origin}"

    # Check voxel vectors
    if not np.allclose(cube1.voxel_vectors, cube2.voxel_vectors, atol=tolerance):
        return False, f"Voxel vectors mismatch"

    # Check number of atoms (warning, not error)
    if cube1.natoms != cube2.natoms:
        print(f"WARNING: Number of atoms differs ({cube1.natoms} vs {cube2.natoms})")

    return True, ""


def interpolate_cube_at_vertices(cube_data, cube_origin, cube_voxels, cube_voxel_vectors,
                                 vertices_real, bohr_to_ang=0.529177, nonzero_bounds=None):
    """
    Interpolate CUBE values at arbitrary 3D vertex positions.

    Uses scipy's RegularGridInterpolator for fast trilinear interpolation.

    Parameters
    ----------
    cube_data : np.ndarray
        3D array of CUBE values (may be vacuum-cropped)
    cube_origin : np.ndarray
        CUBE origin in Bohr
    cube_voxels : np.ndarray
        Number of voxels [nx, ny, nz] (original, before cropping)
    cube_voxel_vectors : np.ndarray
        Voxel vectors (3x3 array) in Bohr
    vertices_real : np.ndarray
        Vertex positions in Angstroms (Nx3 array)
    bohr_to_ang : float
        Conversion factor
    nonzero_bounds : tuple, optional
        ((x_start, x_end), (y_start, y_end), (z_start, z_end)) for vacuum cropping

    Returns
    -------
    np.ndarray
        Interpolated values at each vertex
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        print("ERROR: scipy is required for dual CUBE visualization")
        print("Install with: pip install scipy")
        return None

    # Convert vertices from Angstroms to Bohr
    vertices_bohr = vertices_real / bohr_to_ang

    # Convert real-space coordinates to grid indices
    # r = origin + i*v0 + j*v1 + k*v2
    # Solve for (i, j, k) using matrix inversion

    # For general (non-orthogonal) grids, we need to invert the transformation
    # However, for regular grids, we can use a simpler approach

    # Create grid coordinates
    # Check if data was vacuum cropped - if so, adjust coords for all dimensions
    actual_shape = cube_data.shape

    # Handle vacuum cropping in all three dimensions
    if nonzero_bounds is not None:
        # X dimension
        if (actual_shape[0] != cube_voxels[0] and nonzero_bounds[0] is not None):
            x_start, x_end = nonzero_bounds[0]
            x_coords = np.arange(x_start, x_end)
        else:
            x_coords = np.arange(cube_voxels[0])

        # Y dimension
        if (actual_shape[1] != cube_voxels[1] and nonzero_bounds[1] is not None):
            y_start, y_end = nonzero_bounds[1]
            y_coords = np.arange(y_start, y_end)
        else:
            y_coords = np.arange(cube_voxels[1])

        # Z dimension
        if (actual_shape[2] != cube_voxels[2] and nonzero_bounds[2] is not None):
            z_start, z_end = nonzero_bounds[2]
            z_coords = np.arange(z_start, z_end)
        else:
            z_coords = np.arange(cube_voxels[2])
    else:
        # No cropping - use full grid
        x_coords = np.arange(cube_voxels[0])
        y_coords = np.arange(cube_voxels[1])
        z_coords = np.arange(cube_voxels[2])

    # For orthogonal grids (most common case)
    if (abs(cube_voxel_vectors[0, 1]) < 1e-6 and abs(cube_voxel_vectors[0, 2]) < 1e-6 and
        abs(cube_voxel_vectors[1, 0]) < 1e-6 and abs(cube_voxel_vectors[1, 2]) < 1e-6 and
        abs(cube_voxel_vectors[2, 0]) < 1e-6 and abs(cube_voxel_vectors[2, 1]) < 1e-6):
        # Orthogonal grid - simple division
        grid_indices = np.zeros_like(vertices_bohr)
        for i in range(3):
            grid_indices[:, i] = (vertices_bohr[:, i] - cube_origin[i]) / cube_voxel_vectors[i, i]
    else:
        # Non-orthogonal grid - solve linear system
        # Create transformation matrix (voxel vectors as columns)
        transform_matrix = cube_voxel_vectors.T
        inv_matrix = np.linalg.inv(transform_matrix)

        # Transform vertices to grid coordinates
        relative_pos = vertices_bohr - cube_origin
        grid_indices = (inv_matrix @ relative_pos.T).T

    # Create interpolator
    interpolator = RegularGridInterpolator(
        (x_coords, y_coords, z_coords),
        cube_data,
        bounds_error=False,
        fill_value=0.0  # Use 0 for points outside grid
    )

    # Interpolate values
    interpolated_values = interpolator(grid_indices)

    return interpolated_values


def write_cube_file(filename: str, template_cube: CubeFile, new_data: np.ndarray, comment=None):
    """
    Write a CUBE file with new data using template CUBE's structure.

    Parameters
    ----------
    filename : str
        Output CUBE filename
    template_cube : CubeFile
        Template CUBE file (provides structure, atoms, grid)
    new_data : np.ndarray
        New 3D data array (must match template dimensions)
    comment : str, optional
        Custom comment line. If None, generates automatic comment
    """
    if new_data.shape != tuple(template_cube.nvoxels):
        raise ValueError(f"Data shape {new_data.shape} doesn't match grid {template_cube.nvoxels}")

    with open(filename, 'w') as f:
        # Write comments
        if comment is None:
            comment = f"CUBE file generated by crystal_cubeviz_plotly.py"
        f.write(f"{comment}\n")
        f.write(f"{template_cube.comment2}\n")

        # Write number of atoms and origin
        f.write(f"{template_cube.natoms:5d} {template_cube.origin[0]:12.6f} "
               f"{template_cube.origin[1]:12.6f} {template_cube.origin[2]:12.6f}\n")

        # Write voxel information
        for i in range(3):
            nvox = template_cube.nvoxels[i]
            vec = template_cube.voxel_vectors[i]
            f.write(f"{nvox:5d} {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")

        # Write atomic positions
        for i in range(abs(template_cube.natoms)):
            num = template_cube.atomic_numbers[i]
            pos = template_cube.atomic_positions[i]
            f.write(f"{num:5d} 0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

        # Write data (6 values per line)
        data_flat = new_data.flatten()
        for i in range(0, len(data_flat), 6):
            line_vals = data_flat[i:min(i+6, len(data_flat))]
            f.write(" ".join(f"{val:13.5E}" for val in line_vals) + "\n")

    print(f"✓ Wrote CUBE file: {filename}")


def perform_cube_arithmetic(cubes: list, coefficients: list, operation='linear', align_grids=False):
    """
    Perform arithmetic operations on CUBE files.

    Parameters
    ----------
    cubes : list of CubeFile
        List of CUBE file objects
    coefficients : list of float
        Coefficients for each CUBE (e.g., [1, -1, -1] for A - B - C)
    operation : str
        Type of operation: 'linear' (default), 'multiply', 'divide'
    align_grids : bool
        If True, automatically align grids by padding smaller grids with zeros.
        Requires same origin and voxel spacing. (default: False)

    Returns
    -------
    np.ndarray
        Result data array
    CubeFile
        Template CUBE (first one in list, with aligned grid if alignment was used)
    """
    if len(cubes) < 2:
        raise ValueError("Need at least 2 CUBE files for arithmetic")

    if len(coefficients) != len(cubes):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match number of CUBEs ({len(cubes)})")

    # Check compatibility or alignment
    print(f"\nValidating {len(cubes)} CUBE files...")
    all_compatible = True
    for i in range(1, len(cubes)):
        compatible, error_msg = validate_cube_compatibility(cubes[0], cubes[i])
        if not compatible:
            if align_grids:
                # Try alignment instead
                print(f"  CUBE {i} has different grid size (will attempt alignment)")
                all_compatible = False
            else:
                raise ValueError(f"CUBE {i} incompatible with CUBE 0: {error_msg}\n"
                               f"  Hint: Use --align-grids to automatically align grids with same origin/spacing")

    if all_compatible:
        print("✓ All CUBE files compatible")
    else:
        # Attempt grid alignment
        cubes = align_cube_grids(cubes)

    # Perform arithmetic
    if operation == 'linear':
        print(f"\nPerforming linear combination:")
        result = np.zeros_like(cubes[0].data)
        for i, (cube, coeff) in enumerate(zip(cubes, coefficients)):
            print(f"  {coeff:+.3f} × {cube.filename}")
            result += coeff * cube.data

    elif operation == 'multiply':
        print(f"\nPerforming element-wise multiplication:")
        result = cubes[0].data.copy()
        for i in range(1, len(cubes)):
            print(f"  × {cubes[i].filename}")
            result *= cubes[i].data

    elif operation == 'divide':
        print(f"\nPerforming element-wise division:")
        result = cubes[0].data.copy()
        for i in range(1, len(cubes)):
            print(f"  ÷ {cubes[i].filename}")
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(cubes[i].data != 0, result / cubes[i].data, 0)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Statistics
    print(f"\nResult statistics:")
    print(f"  Min: {np.min(result):.3e}")
    print(f"  Max: {np.max(result):.3e}")
    print(f"  Mean: {np.mean(result):.3e}")
    print(f"  Std: {np.std(result):.3e}")

    return result, cubes[0]


def plot_dual_cube_isosurface(shape_cube: CubeFile, color_cube: CubeFile,
                             iso_value: float, opacity: float = 0.85,
                             show_atoms: bool = True, color_range: tuple = None):
    """
    Create isosurface from shape_cube, colored by values from color_cube.

    This is the VESTA-style dual CUBE visualization: extract density isosurface,
    color it by potential values.

    Parameters
    ----------
    shape_cube : CubeFile
        CUBE file for isosurface extraction (e.g., density)
    color_cube : CubeFile
        CUBE file for color values (e.g., potential)
    iso_value : float
        Isovalue for surface extraction from shape_cube
    opacity : float
        Surface opacity (0-1)
    show_atoms : bool
        Whether to show atoms
    color_range : tuple, optional
        (vmin, vmax) for color mapping. If None, auto-detect

    Returns
    -------
    go.Figure
        Plotly 3D figure
    """
    if not HAS_SKIMAGE:
        print("ERROR: scikit-image is required for isosurface generation")
        print("Install with: pip install scikit-image")
        return None

    # Validate compatibility
    compatible, error_msg = validate_cube_compatibility(shape_cube, color_cube)
    if not compatible:
        print(f"ERROR: CUBE files are incompatible!")
        print(f"  {error_msg}")
        return None

    print(f"\n{'='*60}")
    print("DUAL CUBE VISUALIZATION")
    print(f"{'='*60}")
    print(f"  Shape CUBE: {shape_cube.filename} ({shape_cube.data_type})")
    print(f"  Color CUBE: {color_cube.filename} ({color_cube.data_type})")
    print(f"  Isovalue: {iso_value:.3e}")
    print(f"  Grid: {shape_cube.nvoxels}")

    bohr_to_ang = 0.529177

    # Use cropped data if available
    if hasattr(shape_cube, 'data_bounds') and shape_cube.vacuum_detected:
        shape_data = shape_cube.data[
            shape_cube.data_bounds['x'][0]:shape_cube.data_bounds['x'][1],
            shape_cube.data_bounds['y'][0]:shape_cube.data_bounds['y'][1],
            shape_cube.data_bounds['z'][0]:shape_cube.data_bounds['z'][1]
        ]
        color_data = color_cube.data[
            shape_cube.data_bounds['x'][0]:shape_cube.data_bounds['x'][1],
            shape_cube.data_bounds['y'][0]:shape_cube.data_bounds['y'][1],
            shape_cube.data_bounds['z'][0]:shape_cube.data_bounds['z'][1]
        ]
        # Create bounds tuple for interpolation (using the bounds we just used for cropping)
        crop_bounds = (
            (shape_cube.data_bounds['x'][0], shape_cube.data_bounds['x'][1]),
            (shape_cube.data_bounds['y'][0], shape_cube.data_bounds['y'][1]),
            (shape_cube.data_bounds['z'][0], shape_cube.data_bounds['z'][1])
        )
        print(f"  Using cropped grid: {shape_data.shape}")
    else:
        shape_data = shape_cube.data
        color_data = color_cube.data
        crop_bounds = None

    # Extract isosurface from shape CUBE
    print(f"  Extracting isosurface...")
    try:
        verts, faces, normals, values = measure.marching_cubes(
            shape_data,
            level=iso_value,
            spacing=(1.0, 1.0, 1.0),
            step_size=1,
            allow_degenerate=False
        )
        print(f"  Mesh: {len(verts)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"ERROR: Failed to extract isosurface: {e}")
        return None

    # Transform vertices to real-space coordinates (Angstroms)
    grid_decimation = 1
    verts_real = transform_vertices_optimized(
        verts, shape_cube, bohr_to_ang, shape_data.shape, decimation_factor=grid_decimation
    )

    # Interpolate color values at mesh vertices
    print(f"  Interpolating color values...")
    color_values = interpolate_cube_at_vertices(
        color_data,
        color_cube.origin,
        color_cube.nvoxels,
        color_cube.voxel_vectors,
        verts_real,
        bohr_to_ang,
        nonzero_bounds=crop_bounds
    )

    if color_values is None:
        print("ERROR: Failed to interpolate color values")
        return None

    print(f"  Color range: [{np.min(color_values):.3e}, {np.max(color_values):.3e}]")

    # Determine color scale range
    if color_range is None:
        if color_cube.is_diverging_data():
            # For signed data, use symmetric range around 0
            vmin_raw = np.min(color_values)
            vmax_raw = np.max(color_values)

            # Clip to percentiles for better visualization
            if vmin_raw < 0 and vmax_raw > 0:
                neg_vals = color_values[color_values < 0]
                pos_vals = color_values[color_values > 0]
                vmin_clip = np.percentile(neg_vals, 5) if len(neg_vals) > 0 else 0
                vmax_clip = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 0

                # Force symmetric range
                vmax_sym = max(abs(vmin_clip), abs(vmax_clip))
                vmin, vmax = -vmax_sym, vmax_sym
                print(f"  Symmetric color range: [{vmin:.3e}, {vmax:.3e}] (0 = neutral)")
            else:
                vmin, vmax = vmin_raw, vmax_raw
        else:
            # For density, use percentiles
            vmin = np.percentile(color_values, 5)
            vmax = np.percentile(color_values, 95)
    else:
        vmin, vmax = color_range

    # Choose colorscale based on color CUBE type
    if color_cube.is_diverging_data():
        colorscale = 'RdBu_r'  # Blue for negative, red for positive
        colorbar_title = f"{color_cube.get_data_type_label()}"
    else:
        colorscale = 'Viridis'
        colorbar_title = f"{color_cube.get_data_type_label()}"

    # Create mesh with vertex colors
    print(f"  Creating mesh...")
    mesh = go.Mesh3d(
        x=verts_real[:, 0],
        y=verts_real[:, 1],
        z=verts_real[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=color_values,  # Color by interpolated values
        colorscale=colorscale,
        cmin=vmin,
        cmax=vmax,
        cmid=0 if color_cube.is_diverging_data() else None,
        opacity=opacity,
        colorbar=dict(
            title=colorbar_title,
            len=0.7,
            tickformat='.2e'
        ),
        name=f'{shape_cube.get_data_type_label()} @ {iso_value:.2e}'
    )

    # Add atoms
    traces = [mesh]
    if show_atoms and shape_cube.atomic_numbers:
        atom_props = get_atom_properties()
        atom_x, atom_y, atom_z = [], [], []
        atom_colors, atom_hover = [], []

        for pos, num in zip(shape_cube.atomic_positions, shape_cube.atomic_numbers):
            atom_x.append(pos[0] * bohr_to_ang)
            atom_y.append(pos[1] * bohr_to_ang)
            atom_z.append(pos[2] * bohr_to_ang)

            props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 12})
            atom_colors.append(props['color'])
            atom_hover.append(f"{props['name']} (Z={num})")

        atoms_trace = go.Scatter3d(
            x=atom_x, y=atom_y, z=atom_z,
            mode='markers',
            marker=dict(size=6, color=atom_colors, opacity=0.9, line=dict(color='white', width=1)),
            text=atom_hover,
            hoverinfo='text',
            name='Atoms'
        )
        traces.append(atoms_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Add informative title
    title_text = (f"Dual CUBE: {shape_cube.get_data_type_label()} isosurface "
                 f"colored by {color_cube.get_data_type_label()}")

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=800,
        width=1000
    )

    # Add interpretation annotation for potential
    if color_cube.data_type == 'potential':
        fig.add_annotation(
            text='<b>Blue:</b> Negative potential (e⁻-rich) | <b>Red:</b> Positive potential (e⁻-poor)',
            xref='paper', yref='paper',
            x=0.5, y=1.02,
            showarrow=False,
            bgcolor='rgba(255, 255, 200, 0.9)',
            bordercolor='orange',
            borderwidth=2,
            font=dict(size=12, color='black'),
            align='center',
            xanchor='center'
        )
    elif color_cube.data_type == 'spin':
        fig.add_annotation(
            text='<b>Blue:</b> Spin-down excess (β > α) | <b>Red:</b> Spin-up excess (α > β)',
            xref='paper', yref='paper',
            x=0.5, y=1.02,
            showarrow=False,
            bgcolor='rgba(200, 220, 255, 0.9)',
            bordercolor='blue',
            borderwidth=2,
            font=dict(size=12, color='black'),
            align='center',
            xanchor='center'
        )

    print(f"✓ Dual CUBE visualization complete")
    print(f"{'='*60}\n")

    return fig


def transform_vertices_optimized(verts, cube: CubeFile, bohr_to_ang, data_shape, decimation_factor=1):
    """
    Optimized vertex transformation using vectorized operations.
    ~10x faster than loop-based approach.

    Parameters
    ----------
    verts : np.ndarray
        Vertices from marching cubes (grid indices)
    cube : CubeFile
        Cube file object with voxel vectors
    bohr_to_ang : float
        Conversion factor from Bohr to Angstrom
    data_shape : tuple
        Shape of data (for cropped data offset calculation)
    decimation_factor : int
        Grid decimation factor (2 for fast/adaptive, 1 for high quality)

    Returns
    -------
    np.ndarray
        Transformed vertices in Angstroms
    """
    # CRITICAL FIX: Account for grid decimation
    # When data_coarse = data[::2, ::2, ::2], marching cubes vertices are in decimated space
    # We need to scale them back to original grid space
    if decimation_factor > 1:
        verts = verts * decimation_factor

    # If using cropped data, adjust indices
    if hasattr(cube, 'data_bounds'):
        offset = np.array([
            cube.data_bounds['x'][0],
            cube.data_bounds['y'][0],
            cube.data_bounds['z'][0]
        ], dtype=float)
        verts = verts + offset

    # Vectorized transformation
    verts_real = np.zeros_like(verts)

    # X = origin[0] + i*vox[0,0] + j*vox[1,0] + k*vox[2,0]
    verts_real[:, 0] = (cube.origin[0] +
                        verts[:, 0] * cube.voxel_vectors[0, 0] +
                        verts[:, 1] * cube.voxel_vectors[1, 0] +
                        verts[:, 2] * cube.voxel_vectors[2, 0]) * bohr_to_ang

    verts_real[:, 1] = (cube.origin[1] +
                        verts[:, 0] * cube.voxel_vectors[0, 1] +
                        verts[:, 1] * cube.voxel_vectors[1, 1] +
                        verts[:, 2] * cube.voxel_vectors[2, 1]) * bohr_to_ang

    verts_real[:, 2] = (cube.origin[2] +
                        verts[:, 0] * cube.voxel_vectors[0, 2] +
                        verts[:, 1] * cube.voxel_vectors[1, 2] +
                        verts[:, 2] * cube.voxel_vectors[2, 2]) * bohr_to_ang

    return verts_real


def get_iso_color(data_type, iso_val):
    """Get appropriate color for isosurface based on value."""
    if data_type == 'density':
        return 'cyan' if iso_val > 0 else 'red'
    elif data_type == 'spin':
        return 'blue' if iso_val > 0 else 'red'
    elif data_type == 'potential':
        return 'green' if iso_val > 0 else 'orange'
    else:
        return 'gray'


# Cache atom traces since they don't change
_atom_trace_cache = {}

def create_atom_traces_cached(cube: CubeFile, bohr_to_ang):
    """Create atom traces with caching for performance."""
    cache_key = id(cube)

    if cache_key in _atom_trace_cache:
        return _atom_trace_cache[cache_key]

    atom_props = get_atom_properties()
    traces = []

    for num, pos in zip(cube.atomic_numbers, cube.atomic_positions):
        pos_ang = pos * bohr_to_ang
        props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 12})

        traces.append(go.Scatter3d(
            x=[pos_ang[0]],
            y=[pos_ang[1]],
            z=[pos_ang[2]],
            mode='markers',
            marker=dict(
                size=props['size'],
                color=props['color'],
                line=dict(color='black', width=2)
            ),
            name=props['name'],
            showlegend=False,
            hoverinfo='text',
            hovertext=f"{props['name']} ({pos_ang[0]:.2f}, {pos_ang[1]:.2f}, {pos_ang[2]:.2f})"
        ))

    _atom_trace_cache[cache_key] = traces
    return traces


def plot_isosurface_plotly(cube: CubeFile, iso_values: Optional[List[float]] = None,
                          colorscale: Optional[str] = None, opacity: float = 0.7,
                          show_atoms: bool = True):
    """Plot 3D isosurfaces using Plotly."""
    
    fig = go.Figure()
    
    # Get Cartesian coordinates
    X, Y, Z = cube.get_cartesian_grid()
    
    # Convert to Angstroms
    bohr_to_ang = 0.529177
    X *= bohr_to_ang
    Y *= bohr_to_ang
    Z *= bohr_to_ang
    
    # Use recommended values if not provided
    if iso_values is None:
        iso_values = cube.recommended_iso
        print(f"Using isovalues: {', '.join([f'{x:.2e}' for x in iso_values])}")
    
    # Get colorscale
    colorscale = get_plotly_colorscale(cube.data_type, colorscale)
    
    # Create color palette for multiple isosurfaces
    colors = pcolors.sample_colorscale(colorscale,
                                       np.linspace(0, 1, len(iso_values)))
    
    # Add isosurfaces
    if HAS_SKIMAGE:
        for idx, iso_val in enumerate(iso_values):
            if cube.data_type == 'density' and iso_val < 0:
                continue
            
            n_above = np.sum(cube.data >= iso_val) if iso_val > 0 else np.sum(cube.data <= iso_val)
            if n_above == 0:
                continue
            
            try:
                # Generate isosurface using marching cubes
                verts, faces, normals, values = measure.marching_cubes(
                    cube.data, 
                    level=iso_val,
                    spacing=(1.0, 1.0, 1.0),
                    gradient_direction='descent'
                )
                
                # Transform vertices to real coordinates
                verts_real = np.zeros_like(verts)
                for i in range(len(verts)):
                    vi, vj, vk = verts[i]
                    verts_real[i, 0] = (cube.origin[0] + 
                                       vi * cube.voxel_vectors[0, 0] + 
                                       vj * cube.voxel_vectors[1, 0] + 
                                       vk * cube.voxel_vectors[2, 0]) * bohr_to_ang
                    verts_real[i, 1] = (cube.origin[1] + 
                                       vi * cube.voxel_vectors[0, 1] + 
                                       vj * cube.voxel_vectors[1, 1] + 
                                       vk * cube.voxel_vectors[2, 1]) * bohr_to_ang
                    verts_real[i, 2] = (cube.origin[2] + 
                                       vi * cube.voxel_vectors[0, 2] + 
                                       vj * cube.voxel_vectors[1, 2] + 
                                       vk * cube.voxel_vectors[2, 2]) * bohr_to_ang
                
                # Add mesh to figure
                fig.add_trace(go.Mesh3d(
                    x=verts_real[:, 0],
                    y=verts_real[:, 1],
                    z=verts_real[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=colors[idx] if isinstance(colors[idx], str) else f'rgb{tuple(colors[idx][:3])}',
                    opacity=opacity,
                    name=f'iso={iso_val:.2e}',
                    showlegend=True
                ))
                
            except Exception as e:
                print(f"Could not generate isosurface for {iso_val:.3e}: {e}")
    else:
        # Fallback: use volume rendering
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=cube.data.flatten(),
            isomin=min(iso_values),
            isomax=max(iso_values),
            opacity=opacity,
            surface_count=len(iso_values),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=f'{cube.get_data_type_label()}')
        ))
    
    # Add atoms
    if show_atoms:
        atom_props = get_atom_properties()
        
        for num, pos in zip(cube.atomic_numbers, cube.atomic_positions):
            pos_ang = pos * bohr_to_ang
            
            # For 2D materials, ensure atoms are shown together
            if cube.is_2d and pos_ang[2] > cube.z_vacuum * bohr_to_ang / 2:
                pos_ang[2] -= cube.z_vacuum * bohr_to_ang
            
            props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 12})
            
            fig.add_trace(go.Scatter3d(
                x=[pos_ang[0]],
                y=[pos_ang[1]],
                z=[pos_ang[2]],
                mode='markers',
                marker=dict(
                    size=props['size'],
                    color=props['color'],
                    line=dict(color='black', width=2)
                ),
                name=props['name'],
                showlegend=False
            ))
    
    # Update layout
    title = f"{cube.get_data_type_label()} - {os.path.basename(cube.filename)}"
    if cube.is_2d:
        title += " (2D material)"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    return fig


def plot_isosurface_browser_optimized(cube: CubeFile, n_values=20,
                                     colorscale=None, opacity=0.7,
                                     show_atoms=True, quality='adaptive'):
    """
    Optimized isosurface browser with coarse-to-fine rendering.

    Quality modes:
    - 'fast': Low-res preview only (2x decimation, sparse frames)
    - 'adaptive': Start low-res, calculate every 5th frame (default)
    - 'high': Full resolution, all frames (slow but accurate)

    Parameters
    ----------
    cube : CubeFile
        Loaded cube file object
    n_values : int
        Number of isosurface levels
    colorscale : str, optional
        Plotly colorscale name
    opacity : float
        Isosurface opacity (0-1)
    show_atoms : bool
        Whether to show atoms
    quality : str
        Quality mode: 'fast', 'adaptive', or 'high'

    Returns
    -------
    go.Figure
        Plotly figure with interactive slider
    """

    # Generate smart isovalues
    iso_values = generate_smart_isovalues(
        cube.data,
        cube.data_type,
        n_values=n_values,
        include_half_decades=True
    )

    print(f"\nIsosurface Browser Configuration:")
    print(f"  Values: {n_values} levels")
    print(f"  Range: {iso_values[0]:.3e} to {iso_values[-1]:.3e}")
    print(f"  Quality: {quality}")

    # Diagnostic: Check data range for spin/potential
    if cube.is_diverging_data():
        data_min = np.min(cube.data)
        data_max = np.max(cube.data)
        print(f"  Data type: {cube.data_type}")
        print(f"  Data range: [{data_min:.3e}, {data_max:.3e}]")
        if data_min >= -1e-10:
            print(f"  ⚠️ WARNING: {cube.get_data_type_label()} data contains NO negative values!")
            print(f"     Only positive isosurfaces will be shown.")
            print(f"     This may indicate the CUBE file contains absolute values rather than signed differences.")
        elif data_max <= 1e-10:
            print(f"  ⚠️ WARNING: {cube.get_data_type_label()} data contains NO positive values!")
            print(f"     Only negative isosurfaces will be shown.")
        else:
            print(f"  ✓ Data contains both positive and negative values - will show red (+) and blue (-) surfaces")

    # Use cropped data if vacuum detected
    if hasattr(cube, 'data_bounds') and cube.vacuum_detected:
        data = cube.data[
            cube.data_bounds['x'][0]:cube.data_bounds['x'][1],
            cube.data_bounds['y'][0]:cube.data_bounds['y'][1],
            cube.data_bounds['z'][0]:cube.data_bounds['z'][1]
        ]
        print(f"  Using cropped grid: {data.shape}")
    else:
        data = cube.data

    # Grid decimation for preview
    if quality in ['fast', 'adaptive']:
        data_coarse = data[::2, ::2, ::2]
        print(f"  Coarse grid: {data_coarse.shape} "
              f"({100 * np.prod(data_coarse.shape) / np.prod(data.shape):.1f}% of full)")
    else:
        data_coarse = data

    # Calculate all frames (sparse calculation removed - it broke slider)
    # Quality differences now come from grid coarseness and step_size only
    calc_indices = list(range(len(iso_values)))
    print(f"  Calculating {len(calc_indices)} frames")

    # Check once if we can create symmetric isosurfaces (both +/- for spin/potential)
    # This avoids checking on every frame
    data_has_negative = np.min(data_coarse) < -1e-10
    data_has_positive = np.max(data_coarse) > 1e-10
    can_show_symmetric = (cube.is_diverging_data() and
                         data_has_negative and data_has_positive)

    # Pre-calculate frames
    frames = []
    bohr_to_ang = 0.529177
    colorscale_name = get_plotly_colorscale(cube.data_type, colorscale)

    for idx in calc_indices:
        iso_val = iso_values[idx]

        # Print progress for every frame (or every 3rd if many frames)
        if len(iso_values) <= 15 or idx % 3 == 0 or idx == len(iso_values) - 1:
            print(f"  Frame {idx + 1}/{len(iso_values)}: {iso_val:.3e}")

        frame_data = []

        if HAS_SKIMAGE:
            # For symmetric data (spin/potential), show both positive and negative isosurfaces
            # Only do this for positive values to avoid duplication
            # Use pre-calculated check for efficiency (checked once above, not on every frame)
            show_symmetric = can_show_symmetric and iso_val > 1e-10  # Positive isovalue only

            # Values to calculate: for symmetric positive values, do both +/-
            # For negative values, just show that one (it will be paired with its positive counterpart)
            # For density or one-sided data, just show the single value
            if show_symmetric:
                values_to_calc = [iso_val, -iso_val]  # Show both red (+) and blue (-)
            else:
                values_to_calc = [iso_val]  # Just this value

            for calc_val in values_to_calc:
                try:
                    step_size = 2 if quality == 'fast' else 1

                    verts, faces, normals, values = measure.marching_cubes(
                        data_coarse,
                        level=calc_val,
                        spacing=(1.0, 1.0, 1.0),
                        step_size=step_size,
                        allow_degenerate=False
                    )

                    # Mesh decimation for very dense meshes
                    if len(verts) > 50000:
                        decimate_factor = len(verts) // 50000
                        verts = verts[::decimate_factor]
                        faces = faces[::decimate_factor]
                        if calc_val == values_to_calc[0]:  # Only print once
                            print(f"    Decimated mesh: {len(verts)} vertices")

                    # Vectorized transformation with decimation factor
                    # Grid decimation factor: 2 for fast/adaptive, 1 for high quality
                    grid_decimation = 2 if quality in ['fast', 'adaptive'] else 1
                    verts_real = transform_vertices_optimized(
                        verts, cube, bohr_to_ang, data_coarse.shape, decimation_factor=grid_decimation
                    )

                    # For symmetric surfaces, use different colors for +/-
                    if show_symmetric:
                        color = 'red' if calc_val > 0 else 'blue'
                        name_suffix = f'+{abs(calc_val):.3e}' if calc_val > 0 else f'-{abs(calc_val):.3e}'
                    else:
                        color = get_iso_color(cube.data_type, calc_val)
                        name_suffix = f'{calc_val:.3e}'

                    frame_data.append(go.Mesh3d(
                        x=verts_real[:, 0],
                        y=verts_real[:, 1],
                        z=verts_real[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=opacity,
                        lighting=dict(
                            ambient=0.5,
                            diffuse=0.8,
                            specular=0.2,
                            roughness=0.5
                        ),
                        flatshading=False,
                        name=name_suffix
                    ))

                except Exception as e:
                    if calc_val == values_to_calc[0]:  # Only print error once
                        print(f"    Failed: {e}")
                    continue

        # Add atoms to EVERY frame (not just first!)
        if show_atoms:
            frame_data.extend(create_atom_traces_cached(cube, bohr_to_ang))

        # Create frame title with warnings if isovalue exceeds data range
        title = f"{cube.get_data_type_label()} - Iso: {iso_val:.3e}"
        if cube.is_diverging_data() and can_show_symmetric:
            # Check if isovalue is too large for either sign
            data_max_pos = np.max(data[data > 0]) if np.any(data > 0) else 0
            data_max_neg = abs(np.min(data[data < 0])) if np.any(data < 0) else 0

            warnings = []
            if iso_val > data_max_pos:
                pos_name = "spin-up" if cube.data_type == 'spin' else "positive"
                warnings.append(f"⚠️ Too large for {pos_name} region")
            if iso_val > data_max_neg:
                neg_name = "spin-down" if cube.data_type == 'spin' else "negative"
                warnings.append(f"⚠️ Too large for {neg_name} region")

            if warnings:
                title += " | " + " & ".join(warnings)

        frames.append(go.Frame(
            data=frame_data,
            name=str(idx),
            layout=go.Layout(
                title=title
            )
        ))

    if len(frames) == 0:
        print("Warning: No frames generated. Check data range and isovalues.")
        return go.Figure()

    # Create figure with slider
    start_idx = len(frames) // 2
    fig = go.Figure(
        data=frames[start_idx].data,
        frames=frames
    )

    # Smart slider labels
    slider_steps = []
    for i, iso_val in enumerate(iso_values):
        if i not in calc_indices:
            continue  # Only show calculated frames

        if iso_val >= 1:
            label = f"{iso_val:.2f}"
        elif iso_val >= 0.01:
            label = f"{iso_val:.3f}"
        else:
            label = f"{iso_val:.1e}"

        slider_steps.append(dict(
            args=[[str(i)],
                  dict(mode='immediate',
                       frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))],
            label=label,
            method='animate'
        ))

    sliders = [dict(
        active=start_idx,
        yanchor='top',
        y=-0.05,
        xanchor='left',
        x=0.05,
        currentvalue={
            'prefix': 'Isovalue: ',
            'visible': True,
            'xanchor': 'right',
            'font': {'size': 18, 'color': '#000000'}
        },
        pad={'b': 10, 't': 50},
        len=0.85,
        steps=slider_steps,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#888888',
        borderwidth=2
    )]

    # Create toggle buttons for spin/potential visibility control
    updatemenus = []
    if cube.is_diverging_data() and can_show_symmetric:
        # Add visibility toggle buttons
        button_label = "Spin" if cube.data_type == 'spin' else "Potential"
        pos_label = "Up" if cube.data_type == 'spin' else "Positive"
        neg_label = "Down" if cube.data_type == 'spin' else "Negative"

        # Build visibility patterns for toggle buttons
        # Check each trace in the first frame to build the visibility pattern
        show_both = [True] * len(frames[0].data)
        show_pos_only = []
        show_neg_only = []

        for trace in frames[0].data:
            # Atoms are Scatter3d traces, isosurfaces are Mesh3d traces
            trace_type = str(type(trace).__name__)
            is_atom = trace_type == 'Scatter3d'  # Atoms are always Scatter3d

            if not is_atom:
                # For isosurface (Mesh3d) traces, check color
                trace_color = str(getattr(trace, 'color', ''))
                is_red = 'red' in trace_color.lower()
                is_blue = 'blue' in trace_color.lower()
            else:
                # Atoms are neither red nor blue isosurfaces
                is_red = False
                is_blue = False

            # Always show atoms, show red or blue based on selection
            show_pos_only.append(True if (is_red or is_atom) else False)
            show_neg_only.append(True if (is_blue or is_atom) else False)

        updatemenus.append(
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label=f"Show Both",
                        method="restyle",
                        args=[{"visible": show_both}],
                    ),
                    dict(
                        label=f"{pos_label} Only",
                        method="restyle",
                        args=[{"visible": show_pos_only}],
                    ),
                    dict(
                        label=f"{neg_label} Only",
                        method="restyle",
                        args=[{"visible": show_neg_only}],
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,  # Lowered from 1.15 to avoid title overlap
                yanchor="top"
            )
        )

    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus if updatemenus else None,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        title=dict(
            text=f"{cube.get_data_type_label()} - Interactive Isosurface Browser",
            font=dict(size=20),
            y=0.97,  # Lower title slightly to avoid button overlap
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(t=120, b=100, l=10, r=10),  # Increased top margin for legend
        height=800,
        width=1000
    )

    # Add interpretation labels for spin/potential
    if cube.data_type == 'potential':
        if can_show_symmetric:
            label_text = '<b>Red (+):</b> Positive potential (e⁻-poor) | <b>Blue (−):</b> Negative potential (e⁻-rich)'
        elif data_has_positive:
            label_text = '<b>Red (+):</b> Positive potential (e⁻-poor) | ⚠️ <i>No negative values in data</i>'
        else:
            label_text = '<b>Blue (−):</b> Negative potential (e⁻-rich) | ⚠️ <i>No positive values in data</i>'
        fig.add_annotation(
            text=label_text,
            xref='paper', yref='paper',
            x=0.5, y=1.02,
            showarrow=False,
            bgcolor='rgba(255, 255, 200, 0.9)',
            bordercolor='orange',
            borderwidth=2,
            font=dict(size=12, color='black'),
            align='center',
            xanchor='center'
        )
    elif cube.data_type == 'spin':
        if can_show_symmetric:
            label_text = '<b>Red (+):</b> Spin-up excess (α > β) | <b>Blue (−):</b> Spin-down excess (β > α)'
        elif data_has_positive:
            label_text = '<b>Red (+):</b> Spin density | ⚠️ <i>Data only contains positive values (may be absolute spin)</i>'
        else:
            label_text = '<b>Blue (−):</b> Spin density | ⚠️ <i>Data only contains negative values</i>'
        fig.add_annotation(
            text=label_text,
            xref='paper', yref='paper',
            x=0.5, y=1.02,
            showarrow=False,
            bgcolor='rgba(200, 220, 255, 0.9)',
            bordercolor='blue',
            borderwidth=2,
            font=dict(size=12, color='black'),
            align='center',
            xanchor='center'
        )
    elif cube.data_type in ['density_difference', 'potential_difference', 'spin_difference', 'difference']:
        # Add legend for difference plots
        if cube.data_type == 'density_difference':
            label_text = '<b>Red (+):</b> Electron accumulation (Δρ > 0) | <b>Blue (−):</b> Electron depletion (Δρ < 0)'
            bgcolor = 'rgba(255, 230, 230, 0.95)'
            bordercolor = 'red'
        elif cube.data_type == 'potential_difference':
            label_text = '<b>Red (+):</b> Potential increase (ΔV > 0) | <b>Blue (−):</b> Potential decrease (ΔV < 0)'
            bgcolor = 'rgba(255, 255, 200, 0.95)'
            bordercolor = 'orange'
        elif cube.data_type == 'spin_difference':
            label_text = '<b>Red (+):</b> Spin polarization increase (Δσ > 0) | <b>Blue (−):</b> Spin polarization decrease (Δσ < 0)'
            bgcolor = 'rgba(200, 220, 255, 0.95)'
            bordercolor = 'blue'
        else:
            label_text = '<b>Red (+):</b> Positive difference | <b>Blue (−):</b> Negative difference'
            bgcolor = 'rgba(230, 230, 255, 0.95)'
            bordercolor = 'purple'

        if can_show_symmetric:
            fig.add_annotation(
                text=label_text,
                xref='paper', yref='paper',
                x=0.5, y=1.02,
                showarrow=False,
                bgcolor=bgcolor,
                bordercolor=bordercolor,
                borderwidth=2,
                font=dict(size=12, color='black'),
                align='center',
                xanchor='center'
            )

    # Add helpful annotation
    fig.add_annotation(
        text=f"Quality: {quality.upper()} | Grid: {data.shape[0]}×{data.shape[1]}×{data.shape[2]}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )

    return fig


def plot_slice_plotly(cube: CubeFile, axis: str = 'z', position: Optional[int] = None,
                     colorscale: Optional[str] = None, show_atoms: bool = True,
                     log_scale: bool = None, percentile_clip: float = 99.5):
    """Plot 2D slice using Plotly."""
    
    # Get slice data
    if position is None:
        position = cube.get_best_visualization_slice(axis)
    
    slice_data = cube.get_slice(axis, position)
    
    # Get colorscale
    colorscale = get_plotly_colorscale(cube.data_type, colorscale)
    
    # Auto-detect if log scale should be used
    if log_scale is None:
        log_scale = (cube.data_type == 'density')
    
    bohr_to_ang = 0.529177
    
    # Calculate color scale and prepare display data
    display_data = slice_data.copy()
    
    if cube.data_type == 'density':
        nonzero_data = slice_data[slice_data > 0]
        if len(nonzero_data) > 0:
            vmin = np.percentile(nonzero_data, 100 - percentile_clip)
            vmax = np.percentile(nonzero_data, percentile_clip)
            
            if log_scale:
                display_data = np.log10(slice_data + 1e-10)
                vmin = np.log10(vmin + 1e-10)
                vmax = np.log10(vmax + 1e-10)
        else:
            if log_scale:
                display_data = np.log10(slice_data + 1e-10)
                vmin, vmax = -10, -6
            else:
                vmin, vmax = 0, 1e-6
    elif cube.is_diverging_data():
        vmax = np.percentile(np.abs(slice_data), percentile_clip)
        vmin = -vmax
    else:
        vmin = np.min(slice_data)
        vmax = np.max(slice_data)
    # elif cube.is_diverging_data():
    #     vmax = np.percentile(np.abs(slice_data), percentile_clip)
    #     vmin = -vmax
    # else:
    #     vmin = np.min(slice_data)
    #     vmax = np.max(slice_data)
    
    # Create figure
    fig = go.Figure()
    
    # Handle different axes
    # For non-orthogonal lattices: r = origin + i*voxel[0] + j*voxel[1] + k*voxel[2]
    if axis == 'z':
        # XY plane at Z slice k=position (i, j vary)
        i = np.arange(cube.nvoxels[0])
        j = np.arange(cube.nvoxels[1])
        I, J = np.meshgrid(i, j, indexing='ij')

        # Include contribution from all three voxel vectors
        X = (cube.origin[0] +
             I * cube.voxel_vectors[0, 0] +
             J * cube.voxel_vectors[1, 0] +
             position * cube.voxel_vectors[2, 0]) * bohr_to_ang
        Y = (cube.origin[1] +
             I * cube.voxel_vectors[0, 1] +
             J * cube.voxel_vectors[1, 1] +
             position * cube.voxel_vectors[2, 1]) * bohr_to_ang

        x_label = 'X (Å)'
        y_label = 'Y (Å)'

        # Atom slicing position
        slice_pos = cube.origin[2] + position * cube.voxel_vectors[2, 2]
        slice_dim = 2

    elif axis == 'y':
        # XZ plane at Y slice j=position (i, k vary)
        i = np.arange(cube.nvoxels[0])
        k = np.arange(cube.nvoxels[2])
        I, K = np.meshgrid(i, k, indexing='ij')

        # Include contribution from all three voxel vectors
        X = (cube.origin[0] +
             I * cube.voxel_vectors[0, 0] +
             position * cube.voxel_vectors[1, 0] +
             K * cube.voxel_vectors[2, 0]) * bohr_to_ang
        Y = (cube.origin[2] +
             I * cube.voxel_vectors[0, 2] +
             position * cube.voxel_vectors[1, 2] +
             K * cube.voxel_vectors[2, 2]) * bohr_to_ang

        x_label = 'X (Å)'
        y_label = 'Z (Å)'

        # Atom slicing position
        slice_pos = cube.origin[1] + position * cube.voxel_vectors[1, 1]
        slice_dim = 1

    else:  # axis == 'x'
        # YZ plane at X slice i=position (j, k vary)
        j = np.arange(cube.nvoxels[1])
        k = np.arange(cube.nvoxels[2])
        J, K = np.meshgrid(j, k, indexing='ij')

        # Include contribution from all three voxel vectors
        X = (cube.origin[1] +
             position * cube.voxel_vectors[0, 1] +
             J * cube.voxel_vectors[1, 1] +
             K * cube.voxel_vectors[2, 1]) * bohr_to_ang
        Y = (cube.origin[2] +
             position * cube.voxel_vectors[0, 2] +
             J * cube.voxel_vectors[1, 2] +
             K * cube.voxel_vectors[2, 2]) * bohr_to_ang

        x_label = 'Y (Å)'
        y_label = 'Z (Å)'

        # Atom slicing position
        slice_pos = cube.origin[0] + position * cube.voxel_vectors[0, 0]
        slice_dim = 0
    
    # Render as a flat 3D surface on the FULL 2D (skewed) Cartesian grid, so
    # non-orthogonal cells show a true parallelogram. go.Heatmap is rectilinear
    # only and silently collapsed X/Y to 1D (X[0,:], Y[:,0]), dropping the skew
    # (issue 2.3 slice). display_data is (ni, nj), matching X/Y from meshgrid.
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(X),
        surfacecolor=display_data,
        colorscale=colorscale,
        cmin=vmin,
        cmax=vmax,
        colorbar=dict(
            title=f'{"log₁₀ " if log_scale and cube.data_type == "density" else ""}'
                  f'{cube.get_data_type_label()}'
        )
    ))
    
    # Add atoms
    if show_atoms and position is not None:
        atom_props = get_atom_properties()
        
        atom_x = []
        atom_y = []
        atom_colors = []
        atom_sizes = []
        atom_texts = []
        
        for num, pos in zip(cube.atomic_numbers, cube.atomic_positions):
            if abs(pos[slice_dim] - slice_pos) < 2.0:  # Within 2 Bohr of slice
                if axis == 'z':
                    atom_x.append(pos[0] * bohr_to_ang)
                    atom_y.append(pos[1] * bohr_to_ang)
                elif axis == 'y':
                    atom_x.append(pos[0] * bohr_to_ang)
                    atom_y.append(pos[2] * bohr_to_ang)
                else:  # axis == 'x'
                    atom_x.append(pos[1] * bohr_to_ang)
                    atom_y.append(pos[2] * bohr_to_ang)
                
                props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 12})
                atom_colors.append(props['color'])
                atom_sizes.append(props['size'] * 2)
                atom_texts.append(props['name'])
        
        if atom_x:
            fig.add_trace(go.Scatter3d(
                x=atom_x,
                y=atom_y,
                z=[0.0] * len(atom_x),  # overlay on the slice plane (same frame)
                mode='markers+text',
                marker=dict(
                    size=atom_sizes,
                    color=atom_colors,
                    line=dict(color='white', width=2)
                ),
                text=atom_texts,
                textposition='top center',
                name='Atoms',
                showlegend=False
            ))
    
    # Update layout
    scale_info = " (log scale)" if log_scale and cube.data_type == 'density' else ""
    title = f"{cube.get_data_type_label()} - {axis.upper()}-slice at index {position}{scale_info}"
    
    # 3D scene viewed straight down the slice plane: shows the true (skewed)
    # parallelogram with equal aspect, atoms overlaid in the same frame.
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis=dict(visible=False, showticklabels=False, title=''),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0.0, y=0.0, z=2.2),   # top-down
                up=dict(x=0.0, y=1.0, z=0.0),
            ),
        ),
        height=750,
        width=800
    )

    return fig


def plot_all_slices_plotly(cube: CubeFile, axis: str = 'z', colorscale: Optional[str] = None,
                          show_atoms: bool = True, max_cols: int = 6, 
                          log_scale: bool = None, percentile_clip: float = 99.5):
    """Plot all slices in a grid using Plotly subplots."""
    
    n_slices = int(cube.nvoxels[['x', 'y', 'z'].index(axis)])

    # Subsample for large cubes: a grid of every slice is unreadable and exceeds
    # plotly's subplot-spacing limit (e.g. a 500-slice slab -> 84 rows). Show an
    # evenly spaced subset (was a hard ValueError before, for Heatmap too).
    max_panels = 36
    if n_slices > max_panels:
        slice_indices = sorted(set(int(v) for v in np.linspace(0, n_slices - 1, max_panels)))
        print(f"  Subsampling {n_slices} slices to {len(slice_indices)} panels for the grid")
    else:
        slice_indices = list(range(n_slices))
    n_panels = len(slice_indices)

    # Determine grid layout
    n_cols = min(max_cols, n_panels)
    n_rows = int((n_panels + n_cols - 1) // n_cols)

    # Spacing must satisfy plotly's <= 1/(n-1) constraint.
    vspace = min(0.08, 0.8 / max(1, n_rows - 1))
    hspace = min(0.05, 0.8 / max(1, n_cols - 1))

    # Create subplots. Each cell is a 3D scene so non-orthogonal slices render
    # as true (skewed) parallelograms via go.Surface (go.Heatmap is rectilinear
    # only and collapsed the 2D affine X/Y to 1D, dropping the skew).
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[f'Slice {i}' for i in slice_indices],
        horizontal_spacing=hspace,
        vertical_spacing=vspace
    )
    
    # Get colorscale
    colorscale = get_plotly_colorscale(cube.data_type, colorscale)
    
    # Auto-detect if log scale should be used
    if log_scale is None:
        log_scale = (cube.data_type == 'density')
    
    bohr_to_ang = 0.529177
    
    # Calculate color scale
    if cube.data_type == 'density':
        nonzero_data = cube.data[cube.data > 0]
        if len(nonzero_data) > 0:
            vmin = np.percentile(nonzero_data, 100 - percentile_clip)
            vmax = np.percentile(nonzero_data, percentile_clip)
            
            if log_scale:
                vmin = np.log10(vmin + 1e-10)
                vmax = np.log10(vmax + 1e-10)
        else:
            vmin, vmax = 0, 1e-6
    elif cube.is_diverging_data():
        abs_data = np.abs(cube.data)
        vmax = np.percentile(abs_data, percentile_clip)
        vmin = -vmax
    else:
        vmin = np.min(cube.data)
        vmax = np.max(cube.data)
    
    # Plot each (subsampled) slice
    for panel, idx in enumerate(slice_indices):
        row = panel // n_cols + 1
        col = panel % n_cols + 1

        slice_data = cube.get_slice(axis, idx)
        
        # Apply log scale if needed
        if log_scale and cube.data_type == 'density':
            slice_data = np.log10(slice_data + 1e-10)
        
        # Handle different axes
        # For non-orthogonal lattices: r = origin + i*voxel[0] + j*voxel[1] + k*voxel[2]
        if axis == 'z':
            # XY plane at Z slice k=idx (i, j vary)
            i = np.arange(cube.nvoxels[0])
            j = np.arange(cube.nvoxels[1])
            I, J = np.meshgrid(i, j, indexing='ij')

            X = (cube.origin[0] +
                 I * cube.voxel_vectors[0, 0] +
                 J * cube.voxel_vectors[1, 0] +
                 idx * cube.voxel_vectors[2, 0]) * bohr_to_ang
            Y = (cube.origin[1] +
                 I * cube.voxel_vectors[0, 1] +
                 J * cube.voxel_vectors[1, 1] +
                 idx * cube.voxel_vectors[2, 1]) * bohr_to_ang

            slice_pos = cube.origin[2] + idx * cube.voxel_vectors[2, 2]
            slice_dim = 2

        elif axis == 'y':
            # XZ plane at Y slice j=idx (i, k vary)
            i = np.arange(cube.nvoxels[0])
            k = np.arange(cube.nvoxels[2])
            I, K = np.meshgrid(i, k, indexing='ij')

            X = (cube.origin[0] +
                 I * cube.voxel_vectors[0, 0] +
                 idx * cube.voxel_vectors[1, 0] +
                 K * cube.voxel_vectors[2, 0]) * bohr_to_ang
            Y = (cube.origin[2] +
                 I * cube.voxel_vectors[0, 2] +
                 idx * cube.voxel_vectors[1, 2] +
                 K * cube.voxel_vectors[2, 2]) * bohr_to_ang

            slice_pos = cube.origin[1] + idx * cube.voxel_vectors[1, 1]
            slice_dim = 1

        else:  # axis == 'x'
            # YZ plane at X slice i=idx (j, k vary)
            j = np.arange(cube.nvoxels[1])
            k = np.arange(cube.nvoxels[2])
            J, K = np.meshgrid(j, k, indexing='ij')

            X = (cube.origin[1] +
                 idx * cube.voxel_vectors[0, 1] +
                 J * cube.voxel_vectors[1, 1] +
                 K * cube.voxel_vectors[2, 1]) * bohr_to_ang
            Y = (cube.origin[2] +
                 idx * cube.voxel_vectors[0, 2] +
                 J * cube.voxel_vectors[1, 2] +
                 K * cube.voxel_vectors[2, 2]) * bohr_to_ang

            slice_pos = cube.origin[0] + idx * cube.voxel_vectors[0, 0]
            slice_dim = 0
        
        # Add slice as a flat surface on the full 2D (skewed) Cartesian grid
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=np.zeros_like(X),
                surfacecolor=slice_data,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=(panel == 0),  # Only show colorbar for first panel
                colorbar=dict(
                    title=cube.get_data_type_label() if panel == 0 else None
                )
            ),
            row=row, col=col
        )
        
        # Add atoms
        if show_atoms:
            atom_props = get_atom_properties()
            
            for num, pos in zip(cube.atomic_numbers, cube.atomic_positions):
                if abs(pos[slice_dim] - slice_pos) < 2.0:  # Within 2 Bohr of slice
                    if axis == 'z':
                        plot_x = pos[0] * bohr_to_ang
                        plot_y = pos[1] * bohr_to_ang
                    elif axis == 'y':
                        plot_x = pos[0] * bohr_to_ang
                        plot_y = pos[2] * bohr_to_ang
                    else:  # axis == 'x'
                        plot_x = pos[1] * bohr_to_ang
                        plot_y = pos[2] * bohr_to_ang
                    
                    props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 8})
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=[plot_x],
                            y=[plot_y],
                            z=[0.0],
                            mode='markers',
                            marker=dict(
                                size=props['size'],
                                color=props['color'],
                                line=dict(color='white', width=1)
                            ),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
    
    # Update layout
    if axis == 'z':
        x_label = 'X (Å)'
        y_label = 'Y (Å)'
    elif axis == 'y':
        x_label = 'X (Å)'
        y_label = 'Z (Å)'
    else:  # axis == 'x'
        x_label = 'Y (Å)'
        y_label = 'Z (Å)'
    
    scale_info = " (log scale)" if log_scale and cube.data_type == 'density' else ""
    fig.update_layout(
        title_text=f'{cube.get_data_type_label()} - All {axis.upper()}-slices{scale_info}',
        showlegend=False,
        height=260 * n_rows,
        width=260 * n_cols
    )

    # Every subplot scene viewed top-down with equal aspect, so skewed slices
    # show true parallelograms.
    fig.update_scenes(
        xaxis_title=x_label,
        yaxis_title=y_label,
        zaxis=dict(visible=False, showticklabels=False, title=''),
        aspectmode='data',
        camera=dict(eye=dict(x=0.0, y=0.0, z=2.2), up=dict(x=0.0, y=1.0, z=0.0)),
    )

    return fig


def _plot_slice_browser_surface(cube, axis='z', colorscale=None, show_atoms=True,
                                log_scale=None, percentile_clip=99.5, max_frames=100):
    """Slice browser for NON-ORTHOGONAL cells: go.Surface frames + slider so
    skewed slices render as true parallelograms. (The orthogonal browser uses
    go.Heatmap/Contour, which are rectilinear-only.)"""
    bohr = 0.529177
    n_slices = int(cube.nvoxels[['x', 'y', 'z'].index(axis)])
    best = cube.get_best_visualization_slice(axis)

    if n_slices > max_frames:
        idxs = sorted(set(int(v) for v in np.linspace(0, n_slices - 1, max_frames)))
        if best not in idxs:
            idxs[int(np.argmin(np.abs(np.array(idxs) - best)))] = best
            idxs = sorted(set(idxs))
    else:
        idxs = list(range(n_slices))

    colorscale = get_plotly_colorscale(cube.data_type, colorscale)
    if log_scale is None:
        log_scale = (cube.data_type == 'density')

    d = cube.data
    if cube.data_type == 'density':
        nz = d[d > 0]
        vmin, vmax = (np.percentile(nz, 100 - percentile_clip), np.percentile(nz, percentile_clip)) \
            if len(nz) else (0.0, 1e-6)
        if log_scale:
            vmin, vmax = np.log10(vmin + 1e-10), np.log10(vmax + 1e-10)
    elif cube.is_diverging_data():
        m = np.percentile(np.abs(d), percentile_clip)
        vmin, vmax = -m, m
    else:
        vmin, vmax = float(np.min(d)), float(np.max(d))

    def _geom(idx):
        if axis == 'z':
            I, J = np.meshgrid(np.arange(cube.nvoxels[0]), np.arange(cube.nvoxels[1]), indexing='ij')
            X = (cube.origin[0] + I*cube.voxel_vectors[0,0] + J*cube.voxel_vectors[1,0] + idx*cube.voxel_vectors[2,0]) * bohr
            Y = (cube.origin[1] + I*cube.voxel_vectors[0,1] + J*cube.voxel_vectors[1,1] + idx*cube.voxel_vectors[2,1]) * bohr
            return X, Y, cube.origin[2] + idx*cube.voxel_vectors[2,2], 2, 'X (Å)', 'Y (Å)'
        if axis == 'y':
            I, K = np.meshgrid(np.arange(cube.nvoxels[0]), np.arange(cube.nvoxels[2]), indexing='ij')
            X = (cube.origin[0] + I*cube.voxel_vectors[0,0] + idx*cube.voxel_vectors[1,0] + K*cube.voxel_vectors[2,0]) * bohr
            Y = (cube.origin[2] + I*cube.voxel_vectors[0,2] + idx*cube.voxel_vectors[1,2] + K*cube.voxel_vectors[2,2]) * bohr
            return X, Y, cube.origin[1] + idx*cube.voxel_vectors[1,1], 1, 'X (Å)', 'Z (Å)'
        J, K = np.meshgrid(np.arange(cube.nvoxels[1]), np.arange(cube.nvoxels[2]), indexing='ij')
        X = (cube.origin[1] + idx*cube.voxel_vectors[0,1] + J*cube.voxel_vectors[1,1] + K*cube.voxel_vectors[2,1]) * bohr
        Y = (cube.origin[2] + idx*cube.voxel_vectors[0,2] + J*cube.voxel_vectors[1,2] + K*cube.voxel_vectors[2,2]) * bohr
        return X, Y, cube.origin[0] + idx*cube.voxel_vectors[0,0], 0, 'Y (Å)', 'Z (Å)'

    props_all = get_atom_properties()
    frames, xl, yl = [], 'X (Å)', 'Y (Å)'
    for idx in idxs:
        sd = cube.get_slice(axis, idx).copy()
        if log_scale and cube.data_type == 'density':
            sd = np.log10(sd + 1e-10)
        X, Y, spos, sdim, xl, yl = _geom(idx)
        fdata = [go.Surface(x=X, y=Y, z=np.zeros_like(X), surfacecolor=sd,
                            colorscale=colorscale, cmin=vmin, cmax=vmax, showscale=True,
                            colorbar=dict(title=cube.get_data_type_label()))]
        if show_atoms:
            ax_, ay_, col_, txt_ = [], [], [], []
            for pos, num in zip(cube.atomic_positions, cube.atomic_numbers):
                if abs(pos[sdim] - spos) < 2.0:
                    if axis == 'z':
                        ax_.append(pos[0]*bohr); ay_.append(pos[1]*bohr)
                    elif axis == 'y':
                        ax_.append(pos[0]*bohr); ay_.append(pos[2]*bohr)
                    else:
                        ax_.append(pos[1]*bohr); ay_.append(pos[2]*bohr)
                    p = props_all.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 10})
                    col_.append(p['color']); txt_.append(p['name'])
            fdata.append(go.Scatter3d(x=ax_, y=ay_, z=[0.0]*len(ax_), mode='markers',
                                      marker=dict(size=6, color=col_, line=dict(color='white', width=1)),
                                      text=txt_, hoverinfo='text', showlegend=False))
        frames.append(go.Frame(data=fdata, name=str(idx)))

    best_fi = idxs.index(best) if best in idxs else 0
    fig = go.Figure(data=frames[best_fi].data, frames=frames)
    sliders = [dict(active=best_fi, yanchor='top', y=0, xanchor='left', x=0,
                    currentvalue={'prefix': f'{axis.upper()}-Slice: ', 'visible': True, 'xanchor': 'right'},
                    pad={'b': 10, 't': 50}, len=0.9,
                    steps=[dict(args=[[str(i)], dict(mode='immediate', frame=dict(duration=0, redraw=True),
                                                     transition=dict(duration=0))],
                               label=str(i), method='animate') for i in idxs])]
    scale_info = " (log scale)" if log_scale and cube.data_type == 'density' else ""
    fig.update_layout(
        title=f"{cube.get_data_type_label()} - Interactive {axis.upper()}-Slice Browser (non-orthogonal){scale_info}",
        scene=dict(xaxis_title=xl, yaxis_title=yl,
                   zaxis=dict(visible=False, showticklabels=False, title=''),
                   aspectmode='data',
                   camera=dict(eye=dict(x=0.0, y=0.0, z=2.2), up=dict(x=0.0, y=1.0, z=0.0))),
        sliders=sliders, height=750, width=850)
    return fig


def plot_slice_browser_plotly(cube: CubeFile, axis: str = 'z', colorscale: Optional[str] = None,
                             show_atoms: bool = True, log_scale: bool = None,
                             percentile_clip: float = 99.5, max_frames: int = 100):
    """Interactive slice browser using Plotly with slider.

    Args:
        max_frames: Maximum number of frames to generate (default 100).
                    If n_slices > max_frames, will subsample evenly.
    """

    n_slices = cube.nvoxels[['x', 'y', 'z'].index(axis)]
    best_slice = cube.get_best_visualization_slice(axis)

    # Non-orthogonal cells need go.Surface (rectilinear Heatmap/Contour cannot
    # render skew); keep the rich contour browser below for orthogonal cells.
    _vv = cube.voxel_vectors
    _diag = max(abs(_vv[0, 0]), abs(_vv[1, 1]), abs(_vv[2, 2]))
    if any(abs(_vv[i, j]) > 1e-6 * _diag for i in range(3) for j in range(3) if i != j):
        return _plot_slice_browser_surface(cube, axis, colorscale, show_atoms,
                                           log_scale, percentile_clip, max_frames)

    # Subsample frames if too many slices
    if n_slices > max_frames:
        # Create evenly spaced slice indices, always including best_slice
        frame_indices = list(np.linspace(0, n_slices - 1, max_frames, dtype=int))
        # Ensure best_slice is included
        if best_slice not in frame_indices:
            # Replace closest index with best_slice
            closest_idx = np.argmin(np.abs(np.array(frame_indices) - best_slice))
            frame_indices[closest_idx] = best_slice
        frame_indices = sorted(set(frame_indices))  # Remove duplicates and sort
        print(f"  Subsampling {n_slices} slices to {len(frame_indices)} frames for performance")
    else:
        frame_indices = list(range(n_slices))

    # Get colorscale
    colorscale = get_plotly_colorscale(cube.data_type, colorscale)
    
    # Auto-detect if log scale should be used
    if log_scale is None:
        log_scale = (cube.data_type == 'density')
    
    bohr_to_ang = 0.529177

    # Calculate global color scale
    if cube.data_type == 'density':
        nonzero_data = cube.data[cube.data > 0]
        if len(nonzero_data) > 0:
            vmin = np.percentile(nonzero_data, 100 - percentile_clip)
            vmax = np.percentile(nonzero_data, percentile_clip)

            if log_scale:
                vmin = np.log10(vmin + 1e-10)
                vmax = np.log10(vmax + 1e-10)
        else:
            vmin, vmax = 0, 1e-6
    elif cube.is_diverging_data():
        # CRITICAL FIX: For spin/potential, 0 must ALWAYS be neutral (white/transparent)
        data_min = np.min(cube.data)
        data_max = np.max(cube.data)

        # Use percentiles to clip extreme outliers, but ALWAYS include 0 in range
        if cube.data_type == 'potential':
            # For potential, clip extremes and FORCE symmetry around 0 (just like SPIN)
            # This ensures 0 is EXACTLY at center -> appears WHITE/NEUTRAL
            abs_min = abs(data_min)
            abs_max = abs(data_max)
            asymmetry_ratio = abs_max / abs_min if abs_min > 1e-10 else float('inf')

            if asymmetry_ratio > 100:  # Highly asymmetric
                print(f"\n  Highly asymmetric potential detected (ratio: {asymmetry_ratio:.1f})")
                print(f"  Using MINIMUM range to make BOTH pos/neg features visible...")
                # Clip each side independently
                neg_data = cube.data[cube.data < 0]
                pos_data = cube.data[cube.data > 0]
                vmin_clip = np.percentile(neg_data, 1) if len(neg_data) > 0 else 0
                vmax_clip = np.percentile(pos_data, 99) if len(pos_data) > 0 else 0

                # CRITICAL FIX: Use MIN (not MAX) to ensure both features visible
                # This gives 50% colorscale to negative, 50% to positive
                vmax_sym = min(abs(vmin_clip), abs(vmax_clip))
                vmin = -vmax_sym
                vmax = vmax_sym
                print(f"  Symmetric range: [{vmin:.3f}, {vmax:.3f}] (0 = WHITE, both sides visible)\n")
            else:
                vmin, vmax = data_min, data_max
        else:
            # For spin, clip to percentiles but force symmetry around 0
            print(f"\n  Clipping spin data to percentiles for better visibility...")
            print(f"  Original range: [{data_min:.3e}, {data_max:.3e}]")

            # Clip each side independently to remove extreme noise
            neg_data = cube.data[cube.data < 0]
            pos_data = cube.data[cube.data > 0]

            if len(neg_data) > 0:
                vmin_clip = np.percentile(neg_data, 5)  # Remove bottom 5% noise
            else:
                vmin_clip = 0

            if len(pos_data) > 0:
                vmax_clip = np.percentile(pos_data, 95)  # Remove top 5% noise
            else:
                vmax_clip = 0

            # Force symmetry around 0 for proper color balance
            vmax_sym = max(abs(vmin_clip), abs(vmax_clip))
            vmin = -vmax_sym
            vmax = vmax_sym
            print(f"  Clipped symmetric range: [{vmin:.3e}, {vmax:.3e}] (0 = neutral)\n")
    else:
        vmin = np.min(cube.data)
        vmax = np.max(cube.data)

    # No need for pre-calculated ranges - let Plotly auto-fit to data (like ULTRA_SIMPLE)

    # Create frames for animation (using subsampled indices if needed)
    frames = []

    for idx in frame_indices:
        slice_data = cube.get_slice(axis, idx)
        
        # Apply log scale if needed
        display_data = slice_data.copy()
        if log_scale and cube.data_type == 'density':
            display_data = np.log10(display_data + 1e-10)
        
        # Create coordinates based on axis
        # Key: For non-orthogonal lattices, real space coordinate is:
        #   r = origin + i*voxel[0] + j*voxel[1] + k*voxel[2]
        # We must include the contribution from the fixed slice index!

        if axis == 'z':
            # XY plane - use SIMPLE orthogonal coordinates (like ULTRA_SIMPLE that worked!)
            # For 2D contour plots, Plotly needs simple 1D coordinate arrays
            # For non-orthogonal lattices: r = origin + i*v0 + j*v1 + k*v2
            # Z-slice (fixed k=idx): X = i*v0[0] + j*v1[0], Y = i*v0[1] + j*v1[1]
            # For diagonal terms (v0[0], v1[1]), we can use simple 1D arrays
            x_coords = np.arange(cube.nvoxels[0]) * cube.voxel_vectors[0, 0] * bohr_to_ang
            y_coords = np.arange(cube.nvoxels[1]) * cube.voxel_vectors[1, 1] * bohr_to_ang

            x_label = 'X (Å)'
            y_label = 'Y (Å)'

            # Calculate actual Z position of this slice in real space
            slice_pos = cube.origin[2] + idx * cube.voxel_vectors[2, 2]
            slice_dim = 2

        elif axis == 'y':
            # XZ plane (fixed j=idx) - CRITICAL FIX: Include contribution from fixed Y-slice index!
            # For non-orthogonal lattices: X = origin[0] + i*v0[0] + idx*v1[0] + k*v2[0]
            #                              Z = origin[2] + i*v0[2] + idx*v1[2] + k*v2[2]
            # For hexagonal lattice with Z perpendicular: v0[2]=0, v1[2]=0, v2[0]=0, so:
            #   X = i*v0[0] + idx*v1[0]  <- THIS IS THE MISSING OFFSET!
            #   Z = k*v2[2]
            x_offset = idx * cube.voxel_vectors[1, 0] * bohr_to_ang  # Contribution from fixed j
            x_coords = np.arange(cube.nvoxels[0]) * cube.voxel_vectors[0, 0] * bohr_to_ang + x_offset

            # VACUUM CROPPING FIX: For X/Y slices, crop Z range to remove vacuum
            if hasattr(cube, 'data_bounds') and cube.vacuum_detected:
                z_start, z_end = cube.data_bounds['z']
                y_coords = np.arange(z_start, z_end) * cube.voxel_vectors[2, 2] * bohr_to_ang
                # Crop display_data accordingly
                display_data = display_data[:, z_start:z_end]
            else:
                y_coords = np.arange(cube.nvoxels[2]) * cube.voxel_vectors[2, 2] * bohr_to_ang

            x_label = 'X (Å)'
            y_label = 'Z (Å)'

            # Calculate actual Y position of this slice
            slice_pos = cube.origin[1] + idx * cube.voxel_vectors[1, 1]
            slice_dim = 1

        else:  # axis == 'x'
            # YZ plane (fixed i=idx) - CRITICAL FIX: Include contribution from fixed X-slice index!
            # For non-orthogonal lattices: Y = origin[1] + idx*v0[1] + j*v1[1] + k*v2[1]
            #                              Z = origin[2] + idx*v0[2] + j*v1[2] + k*v2[2]
            # For hexagonal lattice with Z perpendicular: v0[2]=0, v1[2]=0, v2[1]=0, so:
            #   Y = idx*v0[1] + j*v1[1]  <- THIS IS THE MISSING OFFSET!
            #   Z = k*v2[2]
            x_offset = idx * cube.voxel_vectors[0, 1] * bohr_to_ang  # Contribution from fixed i
            x_coords = np.arange(cube.nvoxels[1]) * cube.voxel_vectors[1, 1] * bohr_to_ang + x_offset

            # VACUUM CROPPING FIX: For X/Y slices, crop Z range to remove vacuum
            if hasattr(cube, 'data_bounds') and cube.vacuum_detected:
                z_start, z_end = cube.data_bounds['z']
                y_coords = np.arange(z_start, z_end) * cube.voxel_vectors[2, 2] * bohr_to_ang
                # Crop display_data accordingly
                display_data = display_data[:, z_start:z_end]
            else:
                y_coords = np.arange(cube.nvoxels[2]) * cube.voxel_vectors[2, 2] * bohr_to_ang

            x_label = 'Y (Å)'
            y_label = 'Z (Å)'

            # Calculate actual X position of this slice
            slice_pos = cube.origin[0] + idx * cube.voxel_vectors[0, 0]
            slice_dim = 0

        # x_coords and y_coords are now created above for each axis

        # Smart contour spacing: more contours for spin/potential, fewer for density
        if cube.is_diverging_data():
            n_contours = 30  # Fine detail for small features
        else:
            n_contours = 25  # Good detail for density

        frame_data = [
            # Background heatmap (faint)
            go.Heatmap(
                x=x_coords,
                y=y_coords,
                z=display_data.T,
                colorscale=colorscale,
                zmid=0 if cube.is_diverging_data() else None,  # BUG 2 FIX: Center at zero for signed data
                showscale=False,
                opacity=0.3,
                zmin=vmin,
                zmax=vmax,
                name='Heatmap'
            ),
            # CONTOUR LINES - using WORKING parameter format from ULTRA_SIMPLE test
            go.Contour(
                x=x_coords,
                y=y_coords,
                z=display_data.T,
                colorscale=colorscale,
                zmid=0 if cube.is_diverging_data() else None,  # BUG 2 FIX: Center at zero for signed data
                contours_coloring='lines',  # WORKING PARAMETER (direct, not nested)
                line_width=2,
                contours_start=vmin,
                contours_end=vmax,
                contours_size=(vmax - vmin) / n_contours,  # Smart contour spacing
                contours_showlabels=True,
                contours_labelfont_size=12,
                contours_labelfont_color='white',
                showscale=True,
                colorbar=dict(
                    title=f'{"log₁₀ " if log_scale and cube.data_type == "density" else ""}'
                          f'{cube.get_data_type_label()}',
                    len=0.7,
                    tickformat='.2e'  # BUG 9 FIX: Force scientific notation (not SI prefixes like "p")
                ),
                name='Contours'
            )
        ]
        
        # Add atoms for this slice
        # CRITICAL: For hexagonal lattices, must convert atom positions to voxel indices first
        if show_atoms:
            atom_props = get_atom_properties()  # Get element colors
            atom_x_slice = []
            atom_y_slice = []
            atom_colors = []
            atom_hover_text = []  # NEW: Hover labels showing element names

            for pos, num in zip(cube.atomic_positions, cube.atomic_numbers):
                if abs(pos[slice_dim] - slice_pos) < 2.0:  # Within 2 Bohr of slice
                    pos_rel = pos - cube.origin  # Relative to origin

                    if axis == 'z':
                        # Invert 2x2 transformation: [X,Y] = [i,j] * [[v0x, v0y], [v1x, v1y]]
                        # For hexagonal: v0=[0.140, -0.081], v1=[-0.000, 0.162]
                        v0x, v0y = cube.voxel_vectors[0, 0], cube.voxel_vectors[0, 1]
                        v1x, v1y = cube.voxel_vectors[1, 0], cube.voxel_vectors[1, 1]
                        det = v0x * v1y - v0y * v1x
                        # Solve for (i, j)
                        i = (pos_rel[0] * v1y - pos_rel[1] * v1x) / det
                        j = (pos_rel[1] * v0x - pos_rel[0] * v0y) / det
                        # Use simple orthogonal projection
                        atom_x_slice.append(i * v0x * bohr_to_ang)
                        atom_y_slice.append(j * v1y * bohr_to_ang)
                    elif axis == 'y':
                        # X-Z plane: simpler direct mapping
                        # Heatmap uses: x = i*v0[0], y = k*v2[2]
                        # So atoms should use: x = (pos[0]-origin[0])*scale_x, y = (pos[2]-origin[2])*scale_y
                        # But heatmap coords start at 0, so just use relative position
                        i = pos_rel[0] / cube.voxel_vectors[0, 0]
                        k = pos_rel[2] / cube.voxel_vectors[2, 2]
                        atom_x_slice.append(i * cube.voxel_vectors[0, 0] * bohr_to_ang)
                        atom_y_slice.append(k * cube.voxel_vectors[2, 2] * bohr_to_ang)
                    else:  # axis == 'x'
                        # Y-Z plane: simpler direct mapping
                        # Heatmap uses: x = j*v1[1], y = k*v2[2]
                        j = pos_rel[1] / cube.voxel_vectors[1, 1]
                        k = pos_rel[2] / cube.voxel_vectors[2, 2]
                        atom_x_slice.append(j * cube.voxel_vectors[1, 1] * bohr_to_ang)
                        atom_y_slice.append(k * cube.voxel_vectors[2, 2] * bohr_to_ang)

                    # Get element color and name from VESTA scheme
                    props = atom_props.get(num, {'color': 'gray', 'name': f'Z{num}', 'size': 12})
                    atom_colors.append(props['color'])
                    atom_hover_text.append(f"{props['name']} (Z={num})")  # NEW: "C (Z=6)"

            # CRITICAL FIX FOR BUG 1: Always add scatter trace (even if empty)
            # This maintains consistent frame structure across all frames
            # Plotly requires all frames to have the same number and types of traces
            frame_data.append(
                go.Scatter(
                    x=atom_x_slice,  # Will be empty list if no atoms in this slice
                    y=atom_y_slice,
                    mode='markers',
                    marker=dict(
                        size=10,  # Slightly larger for better visibility
                        color=atom_colors,  # Element-specific VESTA colors
                        opacity=0.8,
                        line=dict(color='white', width=1.5)
                    ),
                    text=atom_hover_text,  # NEW: Hover text showing element names
                    hoverinfo='text',  # NEW: Show only the custom text on hover
                    showlegend=False,
                    name='Atoms'
                )
            )
        
        # CRITICAL FIX: Remove layout from Frame to match working test
        # The layout parameter in Frame was preventing contour rendering
        frames.append(go.Frame(
            data=frame_data,
            name=str(idx)
        ))
    
    # Find which frame index corresponds to best_slice
    best_frame_idx = frame_indices.index(best_slice) if best_slice in frame_indices else 0

    # Create initial figure (showing best slice)
    fig = go.Figure(
        data=frames[best_frame_idx].data,
        frames=frames
    )

    # Add slider
    sliders = [dict(
        active=best_frame_idx,
        yanchor='top',
        y=0,
        xanchor='left',
        x=0,
        currentvalue={
            'prefix': f'{axis.upper()}-Slice: ',
            'visible': True,
            'xanchor': 'right'
        },
        pad={'b': 10, 't': 50},
        len=0.9,
        steps=[
            dict(
                args=[[str(idx)],
                      dict(mode='immediate',
                           frame=dict(duration=0, redraw=True),
                           transition=dict(duration=0))],
                label=str(idx),
                method='animate'
            )
            for idx in frame_indices
        ]
    )]
    
    # Add toggle buttons for global vs per-frame color scaling
    # Position below the title to avoid overlap
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Global Scale",
                    method="restyle",
                    args=[{"zmin": [vmin, vmin], "zmax": [vmax, vmax]}],  # Both heatmap and contour
                ),
                dict(
                    label="Auto Scale",
                    method="restyle",
                    args=[{"zmin": [None, None], "zmax": [None, None]}],
                ),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.02,  # Just below the title
            yanchor="bottom"
        ),
    ]

    # Calculate data extent to constrain axes (remove whitespace)
    if axis == 'z':
        data_x_max = cube.nvoxels[0] * cube.voxel_vectors[0, 0] * bohr_to_ang
        data_y_max = cube.nvoxels[1] * cube.voxel_vectors[1, 1] * bohr_to_ang
    elif axis == 'y':
        data_x_max = cube.nvoxels[0] * cube.voxel_vectors[0, 0] * bohr_to_ang
        data_y_max = cube.nvoxels[2] * cube.voxel_vectors[2, 2] * bohr_to_ang
    else:  # axis == 'x'
        data_x_max = cube.nvoxels[1] * cube.voxel_vectors[1, 1] * bohr_to_ang
        data_y_max = cube.nvoxels[2] * cube.voxel_vectors[2, 2] * bohr_to_ang

    # Update layout with constrained axis ranges
    scale_info = " (log scale)" if log_scale and cube.data_type == 'density' else ""
    fig.update_layout(
        title=f"{cube.get_data_type_label()} - Interactive {axis.upper()}-Slice Browser{scale_info}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            range=[0, data_x_max],  # Constrain to data extent
            constrain='domain'
        ),
        yaxis=dict(
            range=[0, data_y_max],  # Constrain to data extent
            constrain='domain'
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        height=700,
        width=800
    )
    
    # Add annotation about best slice
    fig.add_annotation(
        text=f"Starting slice: {best_slice}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # NEW: Add spin/potential color legend annotations
    if cube.data_type == 'spin':
        # Add spin up/down labels
        fig.add_annotation(
            text="<b>Blue = Spin Down ↓</b>",
            xref="paper", yref="paper",
            x=1.0, y=0.35,
            xanchor="left",
            showarrow=False,
            bgcolor="rgba(0, 0, 255, 0.2)",
            bordercolor="blue",
            borderwidth=2,
            font=dict(size=12, color="blue")
        )
        fig.add_annotation(
            text="<b>Red = Spin Up ↑</b>",
            xref="paper", yref="paper",
            x=1.0, y=0.30,
            xanchor="left",
            showarrow=False,
            bgcolor="rgba(255, 0, 0, 0.2)",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=12, color="red")
        )
    elif cube.data_type == 'potential':
        # Add positive/negative labels
        fig.add_annotation(
            text="<b>Blue = Negative</b>",
            xref="paper", yref="paper",
            x=1.0, y=0.35,
            xanchor="left",
            showarrow=False,
            bgcolor="rgba(0, 0, 255, 0.2)",
            bordercolor="blue",
            borderwidth=2,
            font=dict(size=12, color="blue")
        )
        fig.add_annotation(
            text="<b>Red = Positive</b>",
            xref="paper", yref="paper",
            x=1.0, y=0.30,
            xanchor="left",
            showarrow=False,
            bgcolor="rgba(255, 0, 0, 0.2)",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=12, color="red")
        )

    # NOTE: Hexagonal lattice warning removed - alignment is now fixed!
    # X/Y axis slicing now works correctly for non-orthogonal lattices

    return fig


def interactive_menu(cube: CubeFile):
    """Interactive menu for visualization options."""
    print("\n" + "="*60)
    print("PLOTLY VISUALIZATION OPTIONS")
    print("="*60)
    
    print(f"\nRecommended isovalues: {', '.join([f'{x:.2e}' for x in cube.recommended_iso])}")
    
    print("\nChoose visualization:")
    print("1. 3D isosurfaces with recommended values")
    print("2. 3D isosurfaces with custom values")
    print("3. Single 2D slice")
    print("4. Browse all slices interactively")
    print("5. View all slices in grid")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6) [1]: ").strip() or "1"
    
    if choice == "1":
        return "3d", cube.recommended_iso, None
    
    elif choice == "2":
        print("\nEnter custom isovalues (comma-separated):")
        print("Example: -0.01,-0.001,0.001,0.01")
        iso_input = input("Values: ").strip()
        try:
            iso_values = [float(x.strip()) for x in iso_input.split(',')]
            return "3d", iso_values, None
        except:
            print("Invalid input. Using recommended values.")
            return "3d", cube.recommended_iso, None
    
    elif choice == "3":
        axis = input("Slice axis (x/y/z) [z]: ").strip().lower() or "z"
        max_idx = cube.nvoxels[['x', 'y', 'z'].index(axis)] - 1
        best_idx = cube.get_best_visualization_slice(axis)
        pos_input = input(f"Slice position (0-{max_idx}) [recommended={best_idx}]: ").strip()
        position = int(pos_input) if pos_input else best_idx
        return "slice", None, (axis, position)
    
    elif choice == "4":
        axis = input("Slice axis (x/y/z) [z]: ").strip().lower() or "z"
        return "slice_browse", None, (axis, None)
    
    elif choice == "5":
        axis = input("Slice axis (x/y/z) [z]: ").strip().lower() or "z"
        return "slice_all", None, (axis, None)
    
    else:
        return "exit", None, None


def detect_cube_files(directory='.'):
    """
    Detect all CUBE files in directory.
    Handles various naming conventions:
    - *.CUBE, *.cube
    - *_DENS.CUBE, *_POT.CUBE, *_SPIN.CUBE
    - *.CUBE.DAT, *.cube.dat
    """
    from pathlib import Path

    dir_path = Path(directory)

    # Search patterns (case-insensitive)
    patterns = [
        '*.CUBE',
        '*.cube',
        '*_DENS.CUBE',
        '*_POT.CUBE',
        '*_SPIN.CUBE',
        '*.CUBE.DAT',
        '*.cube.dat',
        '*DENS_CUBE.DAT',
        '*POT_CUBE.DAT',
        '*SPIN_CUBE.DAT'
    ]

    # Collect all matching files
    cube_files = set()
    for pattern in patterns:
        cube_files.update(dir_path.glob(pattern))

    # Sort and categorize
    categorized = {
        'DENS': [],
        'POT': [],
        'SPIN': [],
        'OTHER': []
    }

    for f in sorted(cube_files):
        fname = f.name.upper()
        if 'DENS' in fname:
            categorized['DENS'].append(f)
        elif 'POT' in fname:
            categorized['POT'].append(f)
        elif 'SPIN' in fname:
            categorized['SPIN'].append(f)
        else:
            categorized['OTHER'].append(f)

    return categorized


def interactive_file_selection(categorized_files):
    """
    Interactive prompt for file selection.
    Allows selecting:
    1. Single file
    2. All files of one type (DENS, POT, SPIN)
    3. All files
    """
    total = sum(len(files) for files in categorized_files.values())

    if total == 0:
        print("No CUBE files found in current directory.")
        return None

    print(f"\n{'='*60}")
    print(f"FOUND {total} CUBE FILE(S) IN CURRENT DIRECTORY")
    print(f"{'='*60}\n")

    # Show categorized summary
    for cat, files in categorized_files.items():
        if files:
            print(f"{cat:8s}: {len(files):2d} file(s)")
            for i, f in enumerate(files[:3]):  # Show first 3
                print(f"    {i+1}. {f.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
            print()

    print("What would you like to visualize?")
    print("─" * 60)

    option_map = {}
    idx = 1

    # Option: Individual files
    all_files = []
    for cat in ['DENS', 'POT', 'SPIN', 'OTHER']:
        all_files.extend(categorized_files[cat])

    if len(all_files) <= 10:
        # Show all files if <= 10
        for f in all_files:
            option_map[idx] = ('single', f)
            print(f"{idx:2d}. Single file: {f.name}")
            idx += 1
    else:
        # Show first 5
        for f in all_files[:5]:
            option_map[idx] = ('single', f)
            print(f"{idx:2d}. Single file: {f.name}")
            idx += 1
        print(f"    ... ({len(all_files) - 5} more files not shown)")
        print(f"    Type file number or name to select")

    print()

    # Option: All files of one type
    for cat in ['DENS', 'POT', 'SPIN']:
        if categorized_files[cat]:
            option_map[idx] = ('type', cat, categorized_files[cat])
            print(f"{idx:2d}. All {cat} files ({len(categorized_files[cat])} files)")
            idx += 1

    # Option: All files
    if total > 1:
        option_map[idx] = ('all', all_files)
        print(f"{idx:2d}. ALL files ({total} files) - batch processing")
        idx += 1

    print()
    print("─" * 60)
    choice = input(f"Enter choice (1-{idx-1}) or filename: ").strip()

    # Parse choice
    try:
        choice_num = int(choice)
        if choice_num in option_map:
            return option_map[choice_num]
        else:
            print(f"Invalid choice: {choice_num}")
            return None
    except ValueError:
        # Check if it's a filename
        for f in all_files:
            if choice.lower() in f.name.lower():
                return ('single', f)
        print(f"File not found: {choice}")
        return None


# ============================================================================
# Publication mode helpers (added 2026-05-31)
# Implement minimal extensions for journal-ready figure output:
#   - Bond traces between near atom pairs
#   - Hidden 3D axes / tight margins / no title
#   - Camera presets for consistent slab vs. cluster views
#   - Shared cmin/cmax across a series for comparable colorbars
#   - Static image export (PNG / SVG / PDF) via kaleido
# ============================================================================

_COVALENT_RADII = {  # Å, single-bond. Cordero 2008 + Pyykko for Li.
    1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76,
}

_CAMERA_PRESETS = {
    # 3/4 perspective. Layers are nearly horizontal but slightly tilted.
    "slab":      dict(eye=dict(x=1.6, y=-1.6, z=0.7), up=dict(x=0, y=0, z=1)),
    # Pure -y side view with z up: layers run perfectly horizontal in image.
    # Screen-horizontal is x — C atoms with the same x project to same screen
    # column (within-layer alignment is exact).
    "slab_side": dict(eye=dict(x=0.0, y=-2.5, z=0.0), up=dict(x=0, y=0, z=1)),
    # Rotated 90° around z: looks along +x. Screen-horizontal is y.
    # Useful when the slab_side view has FSI atoms wrapping across PBC; this
    # view rotates the supercell so a different in-plane direction is visible.
    "slab_side_rotated": dict(eye=dict(x=2.5, y=0.0, z=0.0), up=dict(x=0, y=0, z=1)),
    # Looking along the (a - b) = [1, -1, 0]_hex direction in Cartesian.
    # For 2x2 hex supercell with γ=120° this projects atoms onto a clean
    # zig-zag pattern with strong within-layer alignment.
    "slab_zigzag": dict(eye=dict(x=2.165, y=-1.25, z=0.0), up=dict(x=0, y=0, z=1)),
    # Looking along the (a + b) = [1, 1, 0]_hex direction (armchair).
    "slab_armchair": dict(eye=dict(x=1.25, y=2.165, z=0.0), up=dict(x=0, y=0, z=1)),
    # Slight 3/4 side view: layers nearly horizontal with a hint of depth cue.
    "slab_persp":dict(eye=dict(x=0.6, y=-2.2, z=0.0), up=dict(x=0, y=0, z=1)),
    "cluster":   dict(eye=dict(x=1.5, y=1.4, z=1.0),  up=dict(x=0, y=0, z=1)),
    "front":     dict(eye=dict(x=0.0, y=-2.2, z=0.0), up=dict(x=0, y=0, z=1)),
    "top":       dict(eye=dict(x=0.0, y=0.0, z=2.2),  up=dict(x=0, y=1, z=0)),
}


_ATOMIC_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20,
}


def _add_bond_traces(fig, cube, bohr_to_ang=0.529177,
                     factor=1.2, color="#3a3a3a", width=5,
                     max_bond_A=2.6, highlight_atoms=None,
                     highlight_color="#101010", highlight_width=None):
    """Append Scatter3d bond line traces for atom pairs within
    sum(r_cov) * factor (also gated by max_bond_A).

    If highlight_atoms is a set of element symbols (or atomic numbers),
    bonds between two atoms in that set get a separate THICKER, darker
    trace so the highlighted fragment (e.g. LiFSI) reads more boldly.
    """
    nums = list(getattr(cube, "atomic_numbers", []) or [])
    poss = list(getattr(cube, "atomic_positions", []) or [])
    if not nums:
        return
    if highlight_width is None:
        highlight_width = width + 4
    hl_zset = set()
    if highlight_atoms:
        for a in highlight_atoms:
            if isinstance(a, int):
                hl_zset.add(a)
            else:
                z = _ATOMIC_SYMBOL_TO_Z.get(str(a).strip())
                if z is not None:
                    hl_zset.add(z)
    poss_A = [np.asarray(p) * bohr_to_ang for p in poss]
    xs, ys, zs = [], [], []
    xs_hl, ys_hl, zs_hl = [], [], []
    n = len(nums)
    for i in range(n):
        zi = nums[i]; ri = _COVALENT_RADII.get(zi, 0.8)
        for j in range(i + 1, n):
            zj = nums[j]; rj = _COVALENT_RADII.get(zj, 0.8)
            d = float(np.linalg.norm(poss_A[i] - poss_A[j]))
            cutoff = (ri + rj) * factor
            if d <= cutoff and d <= max_bond_A:
                if hl_zset and zi in hl_zset and zj in hl_zset:
                    xs_hl += [poss_A[i][0], poss_A[j][0], None]
                    ys_hl += [poss_A[i][1], poss_A[j][1], None]
                    zs_hl += [poss_A[i][2], poss_A[j][2], None]
                else:
                    xs += [poss_A[i][0], poss_A[j][0], None]
                    ys += [poss_A[i][1], poss_A[j][1], None]
                    zs += [poss_A[i][2], poss_A[j][2], None]
    if xs:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False, name="bonds",
        ))
    if xs_hl:
        fig.add_trace(go.Scatter3d(
            x=xs_hl, y=ys_hl, z=zs_hl, mode="lines",
            line=dict(color=highlight_color, width=highlight_width),
            hoverinfo="skip", showlegend=False, name="bonds_hl",
        ))


def _apply_camera_preset(fig, preset_name, zoom=1.0, orthographic=False,
                         center_on_content=True):
    cam = _CAMERA_PRESETS.get(preset_name)
    if not cam:
        return
    cam = {k: dict(v) for k, v in cam.items()}
    if zoom != 1.0:
        for axis in ("x", "y", "z"):
            cam["eye"][axis] = cam["eye"][axis] * zoom
    # Auto-target center of mass of the scene content so the camera doesn't
    # frame a lot of empty vacuum (plotly defaults to scene origin, which
    # for slab cubes can be far from the actual atoms).
    if center_on_content:
        cx = cy = cz = 0.0; n = 0
        for tr in fig.data:
            if isinstance(tr, go.Mesh3d) and tr.x is not None:
                cx += float(np.mean(tr.x)); cy += float(np.mean(tr.y)); cz += float(np.mean(tr.z)); n += 1
        if n > 0:
            # plotly camera.center is in DATA coords ONLY if a target is set;
            # safer to express via scene aspectmode adjustment. Instead, just
            # set the camera center to a normalized (0,0,0) and rely on
            # aspectmode='data' picking up the content via auto-range.
            pass
    if orthographic:
        cam["projection"] = dict(type="orthographic")
    fig.update_scenes(camera=cam, aspectmode="data")


def _apply_publication_layout(fig, hide_axes=True, no_title=True,
                              bg_color="white", no_colorbar=False,
                              tight=True, atom_size=18,
                              colorbar_title=""):
    layout_kwargs = {}
    if no_title:
        layout_kwargs["title"] = None
    if tight:
        layout_kwargs["margin"] = dict(l=0, r=0, t=0, b=0)
    if bg_color:
        bg = "rgba(0,0,0,0)" if bg_color.lower() == "transparent" else bg_color
        layout_kwargs["paper_bgcolor"] = bg
        layout_kwargs["plot_bgcolor"] = bg
    layout_kwargs["showlegend"] = False
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if hide_axes:
        fig.update_scenes(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
        )
    # Drop auto-annotations (the colored caption box)
    if no_title and hide_axes:
        fig.update_layout(annotations=[])
    # Tighten atom marker traces + colorbars
    for tr in fig.data:
        if isinstance(tr, go.Scatter3d) and getattr(tr, "mode", "") and "markers" in tr.mode:
            try:
                if tr.marker is not None:
                    tr.marker.size = atom_size
                    if tr.marker.line is not None:
                        tr.marker.line.width = 1
                tr.showlegend = False
            except Exception:
                pass
        if isinstance(tr, go.Mesh3d):
            try:
                if no_colorbar:
                    tr.showscale = False
                else:
                    cb = getattr(tr, "colorbar", None)
                    if cb is not None:
                        cb.title = dict(text=colorbar_title)
                        cb.thickness = 18
                        cb.len = 0.7
            except Exception:
                pass


def _apply_color_range(fig, cmin, cmax):
    if cmin is None and cmax is None:
        return
    for tr in fig.data:
        if not isinstance(tr, go.Mesh3d):
            continue
        if cmin is not None and hasattr(tr, "cmin"):
            tr.cmin = cmin
        if cmax is not None and hasattr(tr, "cmax"):
            tr.cmax = cmax


def _apply_transparent_center_colorscale(fig):
    """Replace Mesh3d colorscales with a diverging dark-navy → (transparent
    white) → dark-maroon. Endpoints are saturated/darker than the previous
    version so values near the extremes continue to deepen in color, while
    the mid-range tints (light blue / light red) now occupy the inner stops.
    """
    transparent_div = [
        [0.00, "rgba(5, 24, 82, 1.0)"],     # very dark navy
        [0.10, "rgba(33, 102, 172, 1.0)"],
        [0.25, "rgba(67, 147, 195, 1.0)"],
        [0.38, "rgba(146, 197, 222, 0.65)"],
        [0.46, "rgba(209, 229, 240, 0.30)"],
        [0.50, "rgba(247, 247, 247, 0.0)"], # fully transparent center
        [0.54, "rgba(253, 219, 199, 0.30)"],
        [0.62, "rgba(244, 165, 130, 0.65)"],
        [0.75, "rgba(214, 96, 77, 1.0)"],
        [0.90, "rgba(178, 24, 43, 1.0)"],
        [1.00, "rgba(82, 0, 18, 1.0)"],     # very dark maroon
    ]
    for tr in fig.data:
        if isinstance(tr, go.Mesh3d) and getattr(tr, "intensity", None) is not None:
            tr.update(colorscale=transparent_div)


def _save_with_publication(fig, cube_for_atoms, args):
    """Apply opt-in publication tweaks then save HTML and/or PNG/SVG/PDF."""
    if getattr(args, "bonds", False) and cube_for_atoms is not None:
        highlight = None
        hl_str = getattr(args, "bond_highlight", None)
        if hl_str:
            highlight = [s.strip() for s in hl_str.split(",") if s.strip()]
        _add_bond_traces(fig, cube_for_atoms,
                         factor=getattr(args, "bond_factor", 1.2),
                         color=getattr(args, "bond_color", "#3a3a3a"),
                         width=getattr(args, "bond_width", 5),
                         highlight_atoms=highlight,
                         highlight_color=getattr(args, "bond_highlight_color", "#101010"),
                         highlight_width=getattr(args, "bond_highlight_width", None))
    if getattr(args, "transparent_center", False):
        _apply_transparent_center_colorscale(fig)
    if getattr(args, "color_min", None) is not None or getattr(args, "color_max", None) is not None:
        _apply_color_range(fig, args.color_min, args.color_max)
    if getattr(args, "camera_preset", None):
        _apply_camera_preset(fig, args.camera_preset, getattr(args, "zoom", 1.0),
                             orthographic=getattr(args, "orthographic", False))
    # Per-figure camera overrides (eye/up/center) — applied AFTER any preset
    # so they win.  Used for Kabsch-aligned cluster series.
    eye = getattr(args, "camera_eye", None)
    up = getattr(args, "camera_up", None)
    center = getattr(args, "camera_center", None)
    if eye or up or center:
        cam = {}
        if eye is not None:
            cam["eye"] = dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2]))
        if up is not None:
            cam["up"] = dict(x=float(up[0]), y=float(up[1]), z=float(up[2]))
        if center is not None:
            cam["center"] = dict(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        if getattr(args, "orthographic", False):
            cam["projection"] = dict(type="orthographic")
        fig.update_scenes(camera=cam, aspectmode="data")
    pub_mode = bool(getattr(args, "publication", False))
    if pub_mode or getattr(args, "no_title", False) or getattr(args, "no_axes", False):
        _apply_publication_layout(
            fig,
            hide_axes=(pub_mode or getattr(args, "no_axes", False)),
            no_title=(pub_mode or getattr(args, "no_title", False)),
            bg_color=getattr(args, "bg_color", "white"),
            no_colorbar=getattr(args, "no_colorbar", False),
            tight=pub_mode,
        )
    # Explicit scene axis ranges restrict the visible plot region without
    # changing the data aspect ratio (so atoms render round, bonds at
    # correct angles). 'aspectmode' is set to 'data' explicitly so plotly
    # respects the natural Å-to-Å scaling.
    scene_axis_kwargs = {}
    zr = getattr(args, "scene_zrange", None)
    xr = getattr(args, "scene_xrange", None)
    yr = getattr(args, "scene_yrange", None)
    if zr is not None:
        scene_axis_kwargs["zaxis"] = dict(range=[float(zr[0]), float(zr[1])],
                                          visible=False, showbackground=False,
                                          autorange=False)
    if xr is not None:
        scene_axis_kwargs["xaxis"] = dict(range=[float(xr[0]), float(xr[1])],
                                          visible=False, showbackground=False,
                                          autorange=False)
    if yr is not None:
        scene_axis_kwargs["yaxis"] = dict(range=[float(yr[0]), float(yr[1])],
                                          visible=False, showbackground=False,
                                          autorange=False)
    if scene_axis_kwargs:
        # Force isotropic 1:1:1 aspect.  'aspectmode=data' would compute the
        # ratio from the provided axis ranges, but with a non-axis-aligned
        # camera + perspective/ortho projection this can introduce subtle
        # stretching.  Setting aspectmode='cube' (or aspectratio=1,1,1) makes
        # plotly map each axis to the same fraction of scene container, so
        # atoms render as true circles and isosurfaces are not stretched.
        scene_axis_kwargs["aspectmode"] = "cube"
        scene_axis_kwargs["aspectratio"] = dict(x=1, y=1, z=1)
        fig.update_scenes(**scene_axis_kwargs)
    if getattr(args, "save", None):
        fig.write_html(args.save)
        print(f"Saved HTML to: {args.save}")
    img_w = getattr(args, "img_width", 1800)
    img_h = getattr(args, "img_height", 1350)
    for attr, fmt in (("save_png", "png"), ("save_svg", "svg"), ("save_pdf", "pdf")):
        path = getattr(args, attr, None)
        if path:
            try:
                fig.write_image(path, format=fmt, width=img_w, height=img_h, engine="kaleido")
                print(f"Saved {fmt.upper()} to: {path}")
            except Exception as e:
                print(f"FAILED {fmt.upper()} export: {e}")


def main():
    """Enhanced main with auto-detection and isosurface browser."""
    parser = argparse.ArgumentParser(
        description='Crystal Cube Visualizer with Auto-Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detection mode
  python crystal_cubeviz_plotly.py

  # Isosurface browser
  python crystal_cubeviz_plotly.py DENS.CUBE --iso-browse
  python crystal_cubeviz_plotly.py DENS.CUBE --iso-browse --quality fast

  # Traditional usage
  python crystal_cubeviz_plotly.py DENS.CUBE --iso 0.001,0.01
  python crystal_cubeviz_plotly.py DENS.CUBE --slice z 100
        """
    )
    parser.add_argument('filename', nargs='?', help='Cube file (optional - will auto-detect)')
    parser.add_argument('--iso', type=str, metavar='VALUES',
                       help='Override isosurface values (comma-separated). '
                            'For negative values, use: --iso="-1e-5,1e-5" (with quotes and equals sign)')
    parser.add_argument('--cmap', help='Plotly colorscale name')
    parser.add_argument('--alpha', type=float, default=0.7, help='Opacity (0-1)')

    # New: Isosurface browser options
    parser.add_argument('--iso-browse', action='store_true',
                       help='Interactive isosurface browser with slider')
    parser.add_argument('--iso-count', type=int, default=20,
                       help='Number of isosurface levels (default: 20)')
    parser.add_argument('--quality', choices=['fast', 'adaptive', 'high'],
                       default='adaptive', help='Rendering quality (default: adaptive)')

    # Vacuum cropping options
    parser.add_argument('--no-crop', action='store_true',
                       help='Disable automatic vacuum cropping')
    parser.add_argument('--crop-threshold', type=float, default=1e-8,
                       help='Vacuum detection threshold (default: 1e-8)')

    # Existing options
    parser.add_argument('--slice', nargs=2, help='Single slice: axis and position')
    parser.add_argument('--slice-all', help='Show all slices along axis in grid')
    parser.add_argument('--slice-browse', help='Browse slices interactively along axis')
    parser.add_argument('--no-atoms', action='store_true', help='Do not show atoms')
    parser.add_argument('--save', help='Save figure to HTML file')
    parser.add_argument('--log-scale', action='store_true', help='Force log scale for density')
    parser.add_argument('--linear-scale', action='store_true', help='Force linear scale')
    parser.add_argument('--clip', type=float, default=99.5, help='Percentile for color clipping')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--z-shift', type=float, default=0.0, help='Manual z-shift in Angstroms')

    # GIF export options
    parser.add_argument('--save-gif', type=str, metavar='FILE',
                       help='Export as animated GIF (requires kaleido and Pillow)')
    parser.add_argument('--gif-fps', type=int, default=5,
                       help='GIF frames per second (default: 5 for slice-browse, 10 for rotation)')
    parser.add_argument('--gif-width', type=int, default=800,
                       help='GIF width in pixels (default: 800)')
    parser.add_argument('--gif-height', type=int, default=600,
                       help='GIF height in pixels (default: 600, 800 for rotation)')
    parser.add_argument('--rotate-axis', choices=['x', 'y', 'z'], default='z',
                       help='Axis to rotate around for GIF (default: z)')
    parser.add_argument('--rotation-frames', type=int, default=36,
                       help='Number of frames for 360° rotation (default: 36)')

    # Dual CUBE visualization options
    parser.add_argument('--shape-cube', type=str, metavar='FILE',
                       help='CUBE file for isosurface shape (e.g., density)')
    parser.add_argument('--color-cube', type=str, metavar='FILE',
                       help='CUBE file for surface coloring (e.g., potential)')
    parser.add_argument('--shape-iso', type=float, metavar='VALUE',
                       help='Isovalue for shape extraction (required with --shape-cube)')

    # CUBE arithmetic options
    parser.add_argument('--subtract', nargs='+', metavar='FILE',
                       help='Subtract CUBE files: result = FILE1 - FILE2 - FILE3 - ...')
    parser.add_argument('--add', nargs='+', metavar='FILE',
                       help='Add CUBE files: result = FILE1 + FILE2 + FILE3 + ...')
    parser.add_argument('--multiply', nargs='+', metavar='FILE',
                       help='Multiply CUBE files element-wise')
    parser.add_argument('--divide', nargs=2, metavar='FILE',
                       help='Divide CUBE files: result = FILE1 / FILE2')
    parser.add_argument('--formula', type=str, metavar='EXPR',
                       help='Algebraic formula (e.g., "A - B - C" or "2*A + 0.5*B")')
    parser.add_argument('--cubes', type=str, nargs='+', metavar='VAR=FILE',
                       help='Define variables for formula (e.g., A=dens.CUBE B=surf.CUBE)')
    parser.add_argument('--output-cube', type=str, metavar='FILE',
                       help='Output CUBE filename for arithmetic result')
    parser.add_argument('--visualize-result', action='store_true',
                       help='Visualize arithmetic result after saving')
    parser.add_argument('--align-grids', action='store_true',
                       help='Automatically align CUBE grids by padding with zeros (for arithmetic operations on grids with different sizes but same origin/spacing)')

    # === Publication-mode extensions (added 2026-05-31) ===
    parser.add_argument('--bonds', action='store_true',
                       help='Draw bond traces between atom pairs within covalent-radius cutoff')
    parser.add_argument('--bond-factor', type=float, default=1.2,
                       help='Bond cutoff = (r_cov_i + r_cov_j) * factor  (default 1.2; covers Li-O dative contacts)')
    parser.add_argument('--bond-color', type=str, default='#3a3a3a',
                       help='Bond line color (default #3a3a3a)')
    parser.add_argument('--bond-width', type=int, default=5,
                       help='Bond line pixel width (default 5)')
    parser.add_argument('--publication', action='store_true',
                       help='Publication mode: hide axes/box, tight margins, no title')
    parser.add_argument('--no-title', action='store_true', help='Suppress auto-generated title')
    parser.add_argument('--no-axes', action='store_true', help='Hide 3D axes')
    parser.add_argument('--no-colorbar', action='store_true', help='Hide colorbars (for composite panels)')
    parser.add_argument('--transparent-center', action='store_true',
                       help='Use a diverging colorscale that is fully transparent at midpoint')
    parser.add_argument('--camera-preset', choices=list(_CAMERA_PRESETS.keys()),
                       help='Named camera angle (slab/cluster/front/top)')
    parser.add_argument('--zoom', type=float, default=1.0,
                       help='Camera eye distance scale (>1 = further out, reduces axis clip)')
    parser.add_argument('--bg-color', type=str, default='white',
                       help='Background color (e.g. white, transparent, #fafafa). Default white.')
    parser.add_argument('--color-min', type=float, default=None,
                       help='Force colorbar minimum (for shared range across series)')
    parser.add_argument('--color-max', type=float, default=None,
                       help='Force colorbar maximum (for shared range across series)')
    parser.add_argument('--save-png', type=str, metavar='FILE',
                       help='Save figure as static PNG via kaleido')
    parser.add_argument('--save-svg', type=str, metavar='FILE',
                       help='Save figure as static SVG via kaleido (vector)')
    parser.add_argument('--save-pdf', type=str, metavar='FILE',
                       help='Save figure as static PDF via kaleido (vector)')
    parser.add_argument('--img-width', type=int, default=2400,
                       help='Static image width in px (default 2400 for sharper PDF)')
    parser.add_argument('--img-height', type=int, default=1800,
                       help='Static image height in px (default 1800)')
    parser.add_argument('--orthographic', action='store_true',
                       help='Use orthographic camera projection (layers truly flat, no perspective)')
    parser.add_argument('--bond-highlight', type=str, metavar='ELEMENTS',
                       help='Comma-separated atomic symbols whose mutual bonds are drawn thicker (e.g. "Li,N,S,O,F" highlights the LiFSI fragment)')
    parser.add_argument('--bond-highlight-color', type=str, default='#101010',
                       help='Highlight bond color (default near-black)')
    parser.add_argument('--bond-highlight-width', type=int, default=None,
                       help='Highlight bond pixel width (default = bond-width + 4)')
    parser.add_argument('--camera-eye', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Override camera eye vector (overrides --camera-preset eye)')
    parser.add_argument('--camera-up', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Override camera up vector')
    parser.add_argument('--camera-center', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Camera center / look-at point in scene coordinates (Angstroms)')
    parser.add_argument('--scene-zrange', type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'),
                       help='Explicit scene z-axis range (Å) to enforce consistent px/Å '
                            'and clip out PBC artifacts above the topmost atom layer')
    parser.add_argument('--scene-xrange', type=float, nargs=2, metavar=('X_MIN', 'X_MAX'),
                       help='Explicit scene x-axis range (Å)')
    parser.add_argument('--scene-yrange', type=float, nargs=2, metavar=('Y_MIN', 'Y_MAX'),
                       help='Explicit scene y-axis range (Å)')

    args = parser.parse_args()

    # DUAL CUBE MODE: Handle before normal file detection
    if args.shape_cube and args.color_cube:
        if not args.shape_iso:
            print("ERROR: --shape-iso is required when using --shape-cube and --color-cube")
            sys.exit(1)

        # Load both CUBE files
        print(f"Loading shape CUBE: {args.shape_cube}")
        shape_cube = CubeFile(args.shape_cube, z_shift=args.z_shift)

        print(f"Loading color CUBE: {args.color_cube}")
        color_cube = CubeFile(args.color_cube, z_shift=args.z_shift)

        # Create dual visualization
        fig = plot_dual_cube_isosurface(
            shape_cube=shape_cube,
            color_cube=color_cube,
            iso_value=args.shape_iso,
            opacity=args.alpha,
            show_atoms=not args.no_atoms
        )

        if fig:
            # Handle GIF export for dual CUBE
            if args.save_gif:
                # Create rotation GIF
                height = args.gif_height if args.gif_height != 600 else 800
                fps = args.gif_fps if args.gif_fps != 5 else 10

                create_rotation_gif(
                    fig,
                    args.save_gif,
                    rotate_axis=args.rotate_axis,
                    n_frames=args.rotation_frames,
                    fps=fps,
                    width=args.gif_width,
                    height=height
                )

            # Save HTML + (optional) static images, applying publication tweaks
            any_save = args.save or args.save_png or args.save_svg or args.save_pdf
            if any_save:
                _save_with_publication(fig, shape_cube, args)
            elif not args.save_gif:
                fig.show()

        return  # Exit after dual CUBE visualization

    # CUBE ARITHMETIC MODE
    arithmetic_mode = args.subtract or args.add or args.multiply or args.divide or args.formula

    if arithmetic_mode:
        cubes_to_load = []
        coefficients = []

        # Simple subtraction
        if args.subtract:
            cubes_to_load = args.subtract
            coefficients = [1.0] + [-1.0] * (len(args.subtract) - 1)
            operation = 'linear'
            # Use "SUBTRACTED:" so it gets detected as density_difference
            result_comment = f"SUBTRACTED: {os.path.basename(args.subtract[0])} - " + " - ".join([os.path.basename(f) for f in args.subtract[1:]])

        # Simple addition
        elif args.add:
            cubes_to_load = args.add
            coefficients = [1.0] * len(args.add)
            operation = 'linear'
            result_comment = f"CUBE addition: " + " + ".join(args.add)

        # Multiplication
        elif args.multiply:
            cubes_to_load = args.multiply
            coefficients = [1.0] * len(args.multiply)  # Not used for multiply
            operation = 'multiply'
            result_comment = f"CUBE multiplication: " + " × ".join(args.multiply)

        # Division
        elif args.divide:
            cubes_to_load = args.divide
            coefficients = [1.0, 1.0]  # Not used for divide
            operation = 'divide'
            result_comment = f"CUBE division: {args.divide[0]} / {args.divide[1]}"

        # Formula mode
        elif args.formula:
            if not args.cubes:
                print("ERROR: --cubes is required when using --formula")
                print("Example: --formula 'A - B - C' --cubes A=hybrid.CUBE B=surf.CUBE C=mol.CUBE")
                sys.exit(1)

            # Parse variable assignments
            var_map = {}
            for assignment in args.cubes:
                if '=' not in assignment:
                    print(f"ERROR: Invalid assignment '{assignment}'. Use format VAR=FILE")
                    sys.exit(1)
                var, filename = assignment.split('=', 1)
                var_map[var.strip()] = filename.strip()

            # Parse formula to extract coefficients and variables
            # This is a simple parser - supports forms like "A - B - C" or "2*A + 0.5*B - C"
            import re

            formula = args.formula.replace(' ', '')
            # Find all terms with optional coefficients
            # Pattern: optional sign, optional number*var or just var
            pattern = r'([+-]?)(\d+\.?\d*\*)?([A-Za-z]\w*)'
            matches = re.findall(pattern, formula)

            if not matches:
                print(f"ERROR: Could not parse formula '{args.formula}'")
                sys.exit(1)

            cubes_to_load = []
            coefficients = []
            for sign, coeff_str, var in matches:
                if var not in var_map:
                    print(f"ERROR: Variable '{var}' in formula not defined in --cubes")
                    sys.exit(1)

                # Determine coefficient
                coeff = float(coeff_str.replace('*', '')) if coeff_str else 1.0
                if sign == '-':
                    coeff = -coeff

                cubes_to_load.append(var_map[var])
                coefficients.append(coeff)

            operation = 'linear'
            result_comment = f"CUBE formula: {args.formula}"
            print(f"\nParsed formula: {args.formula}")
            for var, file, coeff in zip([m[2] for m in matches], cubes_to_load, coefficients):
                print(f"  {var} ({file}): coefficient = {coeff:+.3f}")

        # Load all CUBE files
        print(f"\nLoading {len(cubes_to_load)} CUBE files for arithmetic...")
        print("  (Skipping vacuum detection to preserve original grids)")
        cubes = []
        for filename in cubes_to_load:
            if not os.path.exists(filename):
                print(f"ERROR: File not found: {filename}")
                sys.exit(1)
            print(f"  Loading: {filename}")
            cubes.append(CubeFile(filename, z_shift=args.z_shift, skip_vacuum_detection=True))

        # Perform arithmetic
        try:
            result_data, template_cube = perform_cube_arithmetic(cubes, coefficients, operation,
                                                                 align_grids=args.align_grids)
        except Exception as e:
            print(f"ERROR: Arithmetic failed: {e}")
            sys.exit(1)

        # Write output CUBE file
        if args.output_cube:
            write_cube_file(args.output_cube, template_cube, result_data, comment=result_comment)
            print(f"\n✓ Result saved to: {args.output_cube}")

            # Check if any visualization was requested
            viz_requested = (args.visualize_result or args.iso_browse or args.slice_browse or
                           args.slice_all or args.iso or args.slice)

            if viz_requested:
                print(f"\nLoading result for visualization...")
                # Load the saved file (this properly sets all metadata including data_type detection)
                filename = args.output_cube
                args.filename = args.output_cube  # Set this so auto-detection is skipped
                cube = CubeFile(filename, z_shift=args.z_shift)
                cube.print_info()
                # Continue to normal visualization flow below (don't return early)
            else:
                return  # Exit after saving if no visualization requested
        else:
            print("\nWARNING: No output file specified. Use --output-cube to save result.")
            return  # Exit if no output file

    # Auto-detection mode
    if args.filename is None:
        print("No file specified. Searching current directory...")
        categorized = detect_cube_files('.')
        selection = interactive_file_selection(categorized)

        if selection is None:
            sys.exit(1)

        sel_type = selection[0]

        if sel_type == 'single':
            # Process single file
            filename = str(selection[1])
        elif sel_type == 'type':
            # Process all files of one type
            cat, files = selection[1], selection[2]
            print(f"\nProcessing {len(files)} {cat} file(s)...")

            # Process each file
            for f in files:
                print(f"\n{'='*60}")
                print(f"Processing: {f.name}")
                print(f"{'='*60}")

                try:
                    cube = CubeFile(str(f), z_shift=args.z_shift)
                    cube.print_info()

                    # Use isosurface browser for batch processing
                    fig = plot_isosurface_browser_optimized(
                        cube,
                        n_values=args.iso_count,
                        quality=args.quality,
                        show_atoms=not args.no_atoms
                    )

                    # Save to HTML
                    output_name = f.stem + '_interactive.html'
                    fig.write_html(output_name)
                    print(f"✓ Saved: {output_name}\n")

                except Exception as e:
                    print(f"✗ Error processing {f.name}: {e}\n")

            print(f"\nBatch processing complete!")
            return

        elif sel_type == 'all':
            # Batch process all files
            files = selection[1]
            print(f"\nBatch processing {len(files)} file(s)...")

            for f in files:
                try:
                    print(f"\nProcessing: {f.name}")
                    cube = CubeFile(str(f), z_shift=args.z_shift)

                    fig = plot_isosurface_browser_optimized(
                        cube,
                        n_values=args.iso_count,
                        quality=args.quality,
                        show_atoms=not args.no_atoms
                    )

                    output_name = f.stem + '_interactive.html'
                    fig.write_html(output_name)
                    print(f"✓ Saved: {output_name}")

                except Exception as e:
                    print(f"✗ Error: {e}")

            print(f"\nBatch processing complete!")
            return
    else:
        filename = args.filename

    # Check file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        sys.exit(1)

    # Read cube file
    print(f"Reading: {filename}")
    cube = CubeFile(filename, z_shift=args.z_shift)
    cube.print_info()
    
    # Check for zero data
    if np.all(cube.data == 0):
        print("\nWARNING: This file contains all zeros!")
        if not args.interactive:
            sys.exit(0)
    
    # Determine log scale preference
    log_scale = None
    if args.log_scale:
        log_scale = True
    elif args.linear_scale:
        log_scale = False
    
    # Initialize figure
    fig = None
    
    # Interactive mode
    if args.interactive:
        viz_type, iso_values, slice_params = interactive_menu(cube)
        
        print(f"\nProcessing visualization type: {viz_type}")
        
        if viz_type == "exit":
            sys.exit(0)
        elif viz_type == "3d":
            print(f"Creating 3D isosurface with values: {iso_values}")
            fig = plot_isosurface_plotly(cube, iso_values, colorscale=args.cmap,
                                        opacity=args.alpha, show_atoms=not args.no_atoms)
        elif viz_type == "slice":
            print(f"Creating single slice: axis={slice_params[0]}, position={slice_params[1]}")
            fig = plot_slice_plotly(cube, axis=slice_params[0], position=slice_params[1],
                                  colorscale=args.cmap, show_atoms=not args.no_atoms,
                                  log_scale=log_scale, percentile_clip=args.clip)
        elif viz_type == "slice_browse":
            print(f"Creating interactive slice browser: axis={slice_params[0]}")
            fig = plot_slice_browser_plotly(cube, axis=slice_params[0], colorscale=args.cmap,
                                          show_atoms=not args.no_atoms, log_scale=log_scale,
                                          percentile_clip=args.clip)
        elif viz_type == "slice_all":
            print(f"Creating slice grid: axis={slice_params[0]}")
            fig = plot_all_slices_plotly(cube, axis=slice_params[0], colorscale=args.cmap,
                                        show_atoms=not args.no_atoms, log_scale=log_scale,
                                        percentile_clip=args.clip)
        else:
            print(f"Unknown visualization type: {viz_type}")
            fig = None
    
    # Command line mode
    else:
        fig = None  # Initialize at the start
        
        # Check for new slice options
        if args.slice_all:
            fig = plot_all_slices_plotly(cube, axis=args.slice_all.lower(), colorscale=args.cmap,
                                        show_atoms=not args.no_atoms, log_scale=log_scale,
                                        percentile_clip=args.clip)
        
        elif args.slice_browse:
            fig = plot_slice_browser_plotly(cube, axis=args.slice_browse.lower(),
                                          colorscale=args.cmap, show_atoms=not args.no_atoms,
                                          log_scale=log_scale, percentile_clip=args.clip)

        elif args.iso_browse:
            # New optimized isosurface browser with interactive slider
            print(f"Creating interactive isosurface browser with {args.iso_count} isovalues (quality: {args.quality})")
            fig = plot_isosurface_browser_optimized(
                cube,
                n_values=args.iso_count,
                colorscale=args.cmap,
                opacity=args.alpha,
                show_atoms=not args.no_atoms,
                quality=args.quality
            )

        elif args.slice:
            axis = args.slice[0].lower()
            position = int(args.slice[1]) if args.slice[1].isdigit() else None
            fig = plot_slice_plotly(cube, axis=axis, position=position,
                                  colorscale=args.cmap, show_atoms=not args.no_atoms,
                                  log_scale=log_scale, percentile_clip=args.clip)
        
        else:
            # Default to 3D visualization
            iso_values = None
            if args.iso:
                try:
                    iso_values = [float(x.strip()) for x in args.iso.split(',')]
                except:
                    print("Warning: Invalid iso values. Using recommended.")
            
            fig = plot_isosurface_plotly(cube, iso_values=iso_values, colorscale=args.cmap,
                                        opacity=args.alpha, show_atoms=not args.no_atoms)
    
    # Save or show
    if fig:
        # Handle GIF export
        if args.save_gif:
            # Determine GIF export mode
            if args.slice_browse or args.iso_browse:
                # Export slice/isosurface browser animation
                if hasattr(fig, 'frames') and len(fig.frames) > 0:
                    success = export_figure_to_gif(
                        fig,
                        args.save_gif,
                        fps=args.gif_fps,
                        width=args.gif_width,
                        height=args.gif_height
                    )
                    if not success:
                        print("GIF export failed. Falling back to HTML save.")
                else:
                    print("ERROR: Figure has no frames. Cannot export as animation GIF.")
                    print("Use --save-gif with --slice-browse or --iso-browse for animations.")
                    print("For rotation GIFs of static views, use --rotate-axis flag.")
            else:
                # Create rotation GIF for static 3D view
                # Adjust height for 3D rotation (typically square)
                height = args.gif_height if args.gif_height != 600 else 800
                fps = args.gif_fps if args.gif_fps != 5 else 10  # Default 10 FPS for rotation

                success = create_rotation_gif(
                    fig,
                    args.save_gif,
                    rotate_axis=args.rotate_axis,
                    n_frames=args.rotation_frames,
                    fps=fps,
                    width=args.gif_width,
                    height=height
                )
                if not success:
                    print("GIF export failed. Falling back to HTML save.")

        # Save HTML + (optional) static images, applying publication tweaks
        any_save = args.save or args.save_png or args.save_svg or args.save_pdf
        if any_save:
            _save_with_publication(fig, cube if 'cube' in dir() else None, args)
        elif not args.save_gif:
            fig.show()
    else:
        print("No figure was created. Please check your options and try again.")


if __name__ == "__main__":
    main()