#!/home/marcus/anaconda3/bin/python
"""
Vibrational Mode Viewer for CRYSTAL23 frequency calculations.

Parses .out files from CRYSTAL23 and creates interactive 3D visualizations
of vibrational normal modes using Plotly.

Usage:
    python vibmode_viewer.py <filename.out>
    python vibmode_viewer.py <filename.out> --mode 27
    python vibmode_viewer.py <filename.out> --list
    python vibmode_viewer.py <filename.out> --mode 27 --normalize
"""

import os
import sys
import re
import argparse
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# Toned-down VESTA-inspired colors (less saturated)
ELEMENT_COLORS = {
    'H':  '#E8E8E8',  # Light gray (not pure white)
    'C':  '#6E6E6E',  # Medium gray
    'N':  '#4E6EDB',  # Softer blue
    'O':  '#DB4E4E',  # Softer red
    'F':  '#7EC87E',  # Softer green
    'S':  '#D4C832',  # Softer yellow
    'LI': '#9E5DC8',  # Softer purple
    'NA': '#A87ED4',  # Softer light purple
    'K':  '#9E5DC8',  # Softer purple
    'CL': '#4EBF4E',  # Softer green
    'BR': '#8B4040',  # Softer dark red
    'I':  '#7A4E7A',  # Softer purple
    'P':  '#D48C3C',  # Softer orange
    'SI': '#D4B896',  # Softer beige
    'MG': '#78C850',  # Softer light green
    'CA': '#50C850',  # Softer green
    'AL': '#C89BD4',  # Aluminum - light purple
    'TI': '#7A9BB5',  # Titanium - blue-gray
    'SE': '#E8A33D',  # Selenium - orange
    'AG': '#B0B0B0',  # Silver - light gray
    'TE': '#9E7B5A',  # Tellurium - brown
    'PB': '#5A5A6E',  # Lead - dark slate
    'DEFAULT': '#C850A0'  # Softer pink
}

# Darker outline colors for depth
ELEMENT_OUTLINE_COLORS = {
    'H':  '#A0A0A0',
    'C':  '#404040',
    'N':  '#2A3A6E',
    'O':  '#6E2A2A',
    'F':  '#3E643E',
    'S':  '#6E6419',
    'LI': '#4E2E64',
    'NA': '#543F6E',
    'K':  '#4E2E64',
    'CL': '#275027',
    'BR': '#462020',
    'I':  '#3D273D',
    'P':  '#6E461E',
    'SI': '#6E5C4B',
    'MG': '#3C6428',
    'CA': '#286428',
    'AL': '#64506A',
    'TI': '#3E4E5A',
    'SE': '#74511E',
    'AG': '#585858',
    'TE': '#4F3D2D',
    'PB': '#2D2D37',
    'DEFAULT': '#642850'
}

# Covalent radii (Angstroms) - for bond detection
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'LI': 1.28, 'NA': 1.66, 'K': 2.03, 'CL': 1.02,
    'BR': 1.20, 'I': 1.39, 'P': 1.07, 'SI': 1.11, 'MG': 1.41,
    'CA': 1.76, 'AL': 1.21, 'TI': 1.60, 'SE': 1.20, 'AG': 1.45,
    'TE': 1.38, 'PB': 1.46, 'DEFAULT': 0.80
}

# Display sizes for Scatter3d markers (in pixels, will be scaled)
DISPLAY_SIZES = {
    'H': 14, 'C': 22, 'N': 20, 'O': 18, 'F': 16,
    'S': 28, 'LI': 20, 'NA': 26, 'K': 32, 'CL': 24,
    'BR': 28, 'I': 32, 'P': 24, 'SI': 26, 'MG': 22,
    'CA': 26, 'AL': 24, 'TI': 26, 'SE': 24, 'AG': 30,
    'TE': 30, 'PB': 36, 'DEFAULT': 18
}

ATOMIC_NUM_TO_ELEMENT = {
    1: 'H', 2: 'He', 3: 'LI', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'NA', 12: 'MG', 13: 'Al', 14: 'SI', 15: 'P',
    16: 'S', 17: 'CL', 18: 'Ar', 19: 'K', 20: 'CA', 35: 'BR', 53: 'I'
}

BOND_THRESHOLD = 1.3

# Signed int/float with optional E-notation. Shared by the modes-table and
# eigenvector-block parsers so imaginary (negative cm-1) and scientific-notation
# values are captured (the old fixed patterns rejected leading '-' and 'E').
_NUM = r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?'


class Crystal23FreqParser:
    """Parser for CRYSTAL23 frequency output files."""

    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.n_atoms = 0
        self.modes = []
        self.displacements = {}
        self._parse()

    def _parse(self):
        with open(self.filename, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\n')
        self._parse_coordinates(lines)
        self._parse_modes_table(lines)
        self._parse_raman_intensities(lines)
        self._parse_normal_modes(lines)

    def _parse_coordinates(self, lines):
        in_coord_section = False
        for i, line in enumerate(lines):
            if 'COORDINATES OF THE EQUIVALENT ATOMS' in line:
                in_coord_section = True
                continue
            if in_coord_section:
                if 'N. ATOM EQUIV' in line or line.strip() == '':
                    continue
                match = re.match(r'\s*(\d+)\s+\d+\s+\d+\s+(\d+)\s+(\w+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)', line)
                if match:
                    element = match.group(3).upper()
                    x, y, z = float(match.group(4)), float(match.group(5)), float(match.group(6))
                    self.atoms.append((element, x, y, z))
                elif len(self.atoms) > 0 and not line.strip().startswith(str(len(self.atoms) + 1)):
                    break
        self.n_atoms = len(self.atoms)

        if self.n_atoms == 0:
            in_input_section = False
            for i, line in enumerate(lines):
                if 'INPUT COORDINATES' in line:
                    in_input_section = True
                    continue
                if in_input_section and 'ATOM AT. N.' in line:
                    continue
                if in_input_section:
                    match = re.match(r'\s*(\d+)\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)', line)
                    if match:
                        atomic_num = int(match.group(2))
                        element = ATOMIC_NUM_TO_ELEMENT.get(atomic_num, 'X')
                        x, y, z = float(match.group(3)), float(match.group(4)), float(match.group(5))
                        self.atoms.append((element, x, y, z))
                    elif len(self.atoms) > 0 and line.strip() == '':
                        break
            self.n_atoms = len(self.atoms)

    # Modes-table row: "  1-   3   -0.5599E-06   -164.2299   -4.9235  (F1u)   A (     0.00)   I"
    #   start-end   eigv   freq_cm   freq_thz   (irrep)  IR (intens)  RAMAN
    # freq_cm/freq_thz use _NUM (capture imaginary/negative); irrep allows
    # primed/quoted labels (A', B").
    _MODE_ROW_RE = re.compile(
        r'\s*(\d+)-\s*(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')'
        r'\s+\(\s*([\w\'"+\-]+)\s*\)\s+(\w+)\s+\(\s*(' + _NUM + r')\)\s+(\w+)'
    )

    def _parse_modes_table(self, lines):
        in_modes_section = False
        for line in lines:
            if 'MODES' in line and 'EIGV' in line and 'FREQUENCIES' in line:
                in_modes_section = True
                continue
            if in_modes_section:
                match = self._MODE_ROW_RE.match(line)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    freq = float(match.group(4))
                    irrep = match.group(6)
                    ir_active = match.group(7) == 'A'
                    ir_intens = float(match.group(8))
                    raman_active = match.group(9) == 'A'
                    degenerate_with = list(range(start, end + 1))
                    # Expand a degenerate range (e.g. "7-  9") into one entry per
                    # mode so eigenvector columns map 1-to-1; partners share
                    # freq/irrep but keep their own mode number + displacement.
                    for mode_num in degenerate_with:
                        self.modes.append({
                            'mode': mode_num,
                            'freq': freq,
                            'irrep': irrep,
                            'imaginary': freq < -1.0,
                            'degeneracy': len(degenerate_with),
                            'degenerate_with': degenerate_with,
                            'ir_active': ir_active,
                            'ir_intens': ir_intens,
                            'raman_active': raman_active,
                            'raman_intens': 0.0,  # Will be filled by _parse_raman_intensities
                        })
                elif self.modes and 'HHHHH' in line:
                    break

    def _parse_raman_intensities(self, lines):
        """Parse Raman intensities from AVERAGED ISOTROPIC INTENSITIES section."""
        in_raman_section = False
        raman_data = {}

        for line in lines:
            if 'AVERAGED ISOTROPIC INTENSITIES' in line:
                in_raman_section = True
                continue
            if in_raman_section:
                if '-----' in line:
                    continue
                # Match: "   27-  27      330.7921 (A  )     30.21     22.63      7.58"
                match = re.match(r'\s*(\d+)-\s*\d+\s+([\d.]+)\s+\(\w+\s*\)\s+([\d.]+)', line)
                if match:
                    mode_num = int(match.group(1))
                    raman_intens = float(match.group(3))
                    raman_data[mode_num] = raman_intens
                elif raman_data and line.strip() == '':
                    # End of section
                    break

        # Update modes with Raman intensities
        for mode in self.modes:
            if mode['mode'] in raman_data:
                mode['raman_intens'] = raman_data[mode['mode']]

    def _parse_normal_modes(self, lines):
        in_normal_modes = False
        current_modes = []
        current_displacements = []
        # Eigenvector columns are emitted in the same mode order as the modes
        # table, so map column -> mode by a positional cursor. This is immune to
        # degeneracy (equal frequencies) and near-degenerate freq collisions,
        # unlike the old abs(Δfreq)<0.1 first-match lookup which overwrote
        # degenerate partners onto one mode.
        cursor = 0

        for line in lines:
            if 'NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES' in line:
                in_normal_modes = True
                continue
            if not in_normal_modes:
                continue

            if line.strip().startswith('FREQ(CM**-1)'):
                if current_modes and current_displacements:
                    self._save_displacement_block(current_modes, current_displacements)
                tail = line.split('FREQ(CM**-1)', 1)[-1]
                freqs = [float(f) for f in re.findall(_NUM, tail)]
                n = len(freqs)
                current_modes = []
                for k in range(n):
                    idx = cursor + k
                    if idx < len(self.modes):
                        current_modes.append(self.modes[idx]['mode'])
                    else:
                        current_modes.append(idx + 1)  # fallback synthetic index
                cursor += n
                current_displacements = [[] for _ in range(n)]
                continue

            # X row carries the atom prefix ("AT.  n EL  X  v1 v2 ...");
            # Y/Z continuation rows do not. Tokenize 1-6 values dynamically so
            # the trailing block (fewer than 6 modes) is no longer dropped, and
            # E-notation is accepted.
            x_match = re.match(r'\s*AT\.\s*\d+\s+\w+\s+X\s+(.*)', line)
            if x_match:
                x_vals = [float(v) for v in re.findall(_NUM, x_match.group(1))]
                for k in range(min(len(x_vals), len(current_displacements))):
                    current_displacements[k].append([x_vals[k], 0.0, 0.0])
                continue

            y_match = re.match(r'\s+Y\s+(' + _NUM + r'.*)', line)
            if y_match:
                y_vals = [float(v) for v in re.findall(_NUM, y_match.group(1))]
                for k in range(min(len(y_vals), len(current_displacements))):
                    if current_displacements[k]:
                        current_displacements[k][-1][1] = y_vals[k]
                continue

            z_match = re.match(r'\s+Z\s+(' + _NUM + r'.*)', line)
            if z_match:
                z_vals = [float(v) for v in re.findall(_NUM, z_match.group(1))]
                for k in range(min(len(z_vals), len(current_displacements))):
                    if current_displacements[k]:
                        current_displacements[k][-1][2] = z_vals[k]
                continue

            if 'VIBRATIONAL TEMPERATURES' in line or 'THERMODYNAMIC' in line:
                if current_modes and current_displacements:
                    self._save_displacement_block(current_modes, current_displacements)
                break

        if current_modes and current_displacements:
            self._save_displacement_block(current_modes, current_displacements)

    def _save_displacement_block(self, modes, displacements):
        for k, mode_idx in enumerate(modes):
            if k < len(displacements) and displacements[k]:
                self.displacements[mode_idx] = np.array(displacements[k])

    def get_coordinates(self):
        coords = np.array([[a[1], a[2], a[3]] for a in self.atoms])
        coords = coords - coords.mean(axis=0)
        return coords

    def get_elements(self):
        return [a[0] for a in self.atoms]

    def get_mode_info(self, mode_idx):
        for mode in self.modes:
            if mode['mode'] == mode_idx:
                return mode
        return None

    def get_displacement(self, mode_idx):
        return self.displacements.get(mode_idx, None)

    def list_modes(self):
        print("\n" + "="*80)
        print(f"{'Mode':>6} {'Freq (cm-1)':>12} {'IR Active':>10} {'IR Intens':>12} {'Raman':>8}")
        print("="*80)
        for mode in self.modes:
            # Show real modes (>1 cm-1) AND imaginary modes; only the near-zero
            # acoustic/translational residuals (~0) are hidden.
            if mode['freq'] > 1.0 or mode.get('imaginary'):
                ir_str = 'Yes' if mode['ir_active'] else 'No'
                raman_str = 'Yes' if mode['raman_active'] else 'No'
                tag = '  (imaginary)' if mode.get('imaginary') else ''
                print(f"{mode['mode']:>6} {mode['freq']:>12.2f} {ir_str:>10} {mode['ir_intens']:>12.2f} {raman_str:>8}{tag}")
        print("="*80 + "\n")


class VibModeAnimator:
    """Fast 3D animator for vibrational modes using optimized Scatter3d."""

    def __init__(self, parser):
        self.parser = parser
        self.coords = parser.get_coordinates()
        self.elements = parser.get_elements()
        self.n_atoms = len(self.elements)
        self.bonds = self._calculate_bonds()
        self.amplitude = 1.0
        self.n_frames = 30
        self.normalize = False
        self.show_arrows = True  # Show displacement arrows by default
        self.arrow_scale = 15.0  # Arrow length multiplier (displacements are in Bohr, need larger scale)

    def _calculate_bonds(self):
        bonds = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                elem_i = self.elements[i]
                elem_j = self.elements[j]
                r_i = COVALENT_RADII.get(elem_i.upper(), COVALENT_RADII['DEFAULT'])
                r_j = COVALENT_RADII.get(elem_j.upper(), COVALENT_RADII['DEFAULT'])
                max_dist = (r_i + r_j) * BOND_THRESHOLD
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                if dist < max_dist:
                    bonds.append((i, j))
        return bonds

    def _get_color(self, element):
        return ELEMENT_COLORS.get(element.upper(), ELEMENT_COLORS['DEFAULT'])

    def _get_outline_color(self, element):
        return ELEMENT_OUTLINE_COLORS.get(element.upper(), ELEMENT_OUTLINE_COLORS['DEFAULT'])

    def _get_size(self, element):
        return DISPLAY_SIZES.get(element.upper(), DISPLAY_SIZES['DEFAULT'])

    def _normalize_displacement(self, displacement, target_max=1.0):
        """
        Uniformly scale all displacement vectors so the maximum displacement
        magnitude equals target_max. This preserves relative intensities between
        atoms while making weak modes more visible.
        """
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        if max_disp > 1e-10:
            scale_factor = target_max / max_disp
            return displacement * scale_factor
        return displacement

    def _create_molecule_traces(self, coords, displacement=None, show_arrows=False, arrow_scale=2.0):
        """Create optimized traces using Scatter3d, optionally with displacement arrows."""
        # Refuse to render when the eigenvector atom count differs from the
        # parsed structure (e.g. eigenvectors span a super/conventional cell
        # while coordinates are the asymmetric/primitive unit). Without this the
        # per-atom loop silently uses the first n_atoms vectors -> misaligned
        # arrows on the wrong atoms.
        if displacement is not None and len(displacement) != self.n_atoms:
            raise ValueError(
                f"displacement has {len(displacement)} atoms but the parsed "
                f"structure has {self.n_atoms}; refusing to render misaligned "
                f"vibration vectors (eigenvectors likely span a larger cell than "
                f"the parsed coordinates)."
            )

        traces = []

        # Single trace for all atoms
        colors = [self._get_color(e) for e in self.elements]
        outline_colors = [self._get_outline_color(e) for e in self.elements]
        sizes = [self._get_size(e) for e in self.elements]

        hover_text = [f"{self.elements[i]}{i+1}" for i in range(self.n_atoms)]

        traces.append(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color=outline_colors, width=2),
                opacity=0.95
            ),
            text=hover_text,
            hoverinfo='text',
            name='Atoms'
        ))

        # Bonds as lines
        if self.bonds:
            bond_x, bond_y, bond_z = [], [], []
            for i, j in self.bonds:
                bond_x.extend([coords[i, 0], coords[j, 0], None])
                bond_y.extend([coords[i, 1], coords[j, 1], None])
                bond_z.extend([coords[i, 2], coords[j, 2], None])

            traces.append(go.Scatter3d(
                x=bond_x,
                y=bond_y,
                z=bond_z,
                mode='lines',
                line=dict(color='#555555', width=6),
                hoverinfo='skip',
                name='Bonds'
            ))

        # Displacement arrows (CRYSPLOT-style)
        if show_arrows and displacement is not None:
            # Arrow shafts (lines from atom to arrow tip)
            arrow_x, arrow_y, arrow_z = [], [], []
            cone_x, cone_y, cone_z = [], [], []
            cone_u, cone_v, cone_w = [], [], []

            for k in range(self.n_atoms):
                disp_mag = np.linalg.norm(displacement[k])
                if disp_mag > 0.001:  # Only show significant displacements
                    # Arrow shaft
                    end_point = coords[k] + displacement[k] * arrow_scale
                    arrow_x.extend([coords[k, 0], end_point[0], None])
                    arrow_y.extend([coords[k, 1], end_point[1], None])
                    arrow_z.extend([coords[k, 2], end_point[2], None])

                    # Cone (arrowhead) - position at end, pointing in displacement direction
                    cone_x.append(end_point[0])
                    cone_y.append(end_point[1])
                    cone_z.append(end_point[2])
                    # Direction unit vector for cone
                    disp_unit = displacement[k] / disp_mag
                    cone_length = 0.25  # Fixed cone size for visibility
                    cone_u.append(disp_unit[0] * cone_length)
                    cone_v.append(disp_unit[1] * cone_length)
                    cone_w.append(disp_unit[2] * cone_length)

            # Arrow shafts
            if arrow_x:
                traces.append(go.Scatter3d(
                    x=arrow_x,
                    y=arrow_y,
                    z=arrow_z,
                    mode='lines',
                    line=dict(color='#E63946', width=6),
                    hoverinfo='skip',
                    name='Arrows',
                    visible=True
                ))

            # Arrow heads (cones)
            if cone_x:
                traces.append(go.Cone(
                    x=cone_x, y=cone_y, z=cone_z,
                    u=cone_u, v=cone_v, w=cone_w,
                    colorscale=[[0, '#E63946'], [1, '#E63946']],
                    showscale=False,
                    sizemode='absolute',
                    sizeref=0.15,
                    anchor='tail',
                    hoverinfo='skip',
                    name='ArrowHeads',
                    visible=True
                ))

        return traces

    def _get_axis_range(self, coords, displacement=None):
        if displacement is not None:
            max_disp = np.abs(displacement).max() * self.amplitude
            margin = max(2.5, max_disp * 2)
        else:
            margin = 2.5

        x_range = [coords[:, 0].min() - margin, coords[:, 0].max() + margin]
        y_range = [coords[:, 1].min() - margin, coords[:, 1].max() + margin]
        z_range = [coords[:, 2].min() - margin, coords[:, 2].max() + margin]

        max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
        mid_x = (x_range[0] + x_range[1]) / 2
        mid_y = (y_range[0] + y_range[1]) / 2
        mid_z = (z_range[0] + z_range[1]) / 2

        return {
            'x': [mid_x - max_range/2, mid_x + max_range/2],
            'y': [mid_y - max_range/2, mid_y + max_range/2],
            'z': [mid_z - max_range/2, mid_z + max_range/2]
        }

    def _get_scene_layout(self, ranges):
        """Clean scene layout with subtle styling."""
        return dict(
            xaxis=dict(
                title=dict(text='X (Å)', font=dict(size=12, color='#444')),
                range=ranges['x'],
                showgrid=True,
                gridcolor='rgba(180, 180, 180, 0.4)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='#f5f5f5',
                tickfont=dict(size=10, color='#666'),
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text='Y (Å)', font=dict(size=12, color='#444')),
                range=ranges['y'],
                showgrid=True,
                gridcolor='rgba(180, 180, 180, 0.4)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='#f0f0f0',
                tickfont=dict(size=10, color='#666'),
                zeroline=False
            ),
            zaxis=dict(
                title=dict(text='Z (Å)', font=dict(size=12, color='#444')),
                range=ranges['z'],
                showgrid=True,
                gridcolor='rgba(180, 180, 180, 0.4)',
                gridwidth=1,
                showbackground=True,
                backgroundcolor='#ebebeb',
                tickfont=dict(size=10, color='#666'),
                zeroline=False
            ),
            aspectmode='cube',
            bgcolor='#fafafa',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        )

    def _validated_displacement(self, mode_idx, strict=False):
        """Fetch a mode's displacement, refusing (or skipping) when its atom
        count does not match the parsed structure.

        On high-symmetry / supercell FREQ outputs the parsed coordinates are the
        asymmetric/primitive cell while the eigenvectors span the full cell, so
        the per-atom arrays are misaligned. Rather than broadcasting the wrong
        vectors onto the wrong atoms we refuse (strict=True) or skip
        (strict=False, returns None) instead of crashing in numpy.
        """
        d = self.parser.get_displacement(mode_idx)
        if d is None:
            return None
        if len(d) != self.n_atoms:
            msg = (f"mode {mode_idx}: displacement has {len(d)} atoms but the "
                   f"parsed structure has {self.n_atoms}; refusing to render "
                   f"misaligned vibration vectors (eigenvectors likely span a "
                   f"larger cell than the parsed coordinates).")
            if strict:
                raise ValueError(msg)
            print(f"  Skipping {msg}")
            return None
        return d

    def create_static_view(self, mode_idx):
        """Create a static 3D view with displacement arrows."""
        displacement = self._validated_displacement(mode_idx)
        mode_info = self.parser.get_mode_info(mode_idx)

        if displacement is None:
            print(f"Error: No displacement data for mode {mode_idx}")
            return None

        if self.normalize:
            displacement = self._normalize_displacement(displacement)

        fig = go.Figure()

        # Use unified trace creation with arrows
        for trace in self._create_molecule_traces(
            self.coords,
            displacement=displacement,
            show_arrows=self.show_arrows,
            arrow_scale=self.arrow_scale
        ):
            fig.add_trace(trace)

        ranges = self._get_axis_range(self.coords, displacement)
        freq = mode_info['freq'] if mode_info else 0.0
        ir_str = "IR Active" if mode_info and mode_info['ir_active'] else "IR Inactive"
        raman_str = "Raman Active" if mode_info and mode_info['raman_active'] else "Raman Inactive"
        intens = mode_info['ir_intens'] if mode_info else 0.0
        norm_str = " (Normalized)" if self.normalize else ""

        fig.update_layout(
            title=dict(
                text=f"<b>Mode {mode_idx}: {freq:.1f} cm⁻¹</b>{norm_str}<br><span style='font-size:12px;color:#666'>{ir_str} ({intens:.1f} km/mol) | {raman_str}</span>",
                x=0.5,
                font=dict(size=15, color='#333')
            ),
            scene=self._get_scene_layout(ranges),
            showlegend=False,
            margin=dict(l=0, r=0, t=70, b=0),
            paper_bgcolor='#fafafa'
        )

        return fig

    def _generate_frames(self, displacement, prefix=""):
        """Generate animation frames for a given displacement."""
        frames = []
        for frame_idx in range(self.n_frames):
            # Offset by pi/2 so frame 0 starts at maximum displacement (cos instead of sin)
            # This ensures arrows are visible from the start
            phase = np.cos(2 * np.pi * frame_idx / self.n_frames)
            current_coords = self.coords + displacement * self.amplitude * phase
            # Arrows oscillate with the vibration - scale by phase
            # This makes arrows flip direction and vary in length as atoms move
            frame_data = self._create_molecule_traces(
                current_coords,
                displacement=displacement * phase,  # Oscillating arrows
                show_arrows=self.show_arrows,
                arrow_scale=self.arrow_scale
            )
            frames.append(go.Frame(data=frame_data, name=f"{prefix}{frame_idx}"))
        return frames

    def create_animation(self, mode_idx, speed=1.0, include_both_normalizations=False):
        """Create an animated 3D view with controls."""
        displacement_raw = self._validated_displacement(mode_idx)
        mode_info = self.parser.get_mode_info(mode_idx)

        if displacement_raw is None:
            print(f"Error: No displacement data for mode {mode_idx}")
            return None

        displacement_norm = self._normalize_displacement(displacement_raw.copy())

        # Use normalized or raw based on setting
        displacement = displacement_norm if self.normalize else displacement_raw

        base_duration = 60
        frame_duration = int(base_duration / speed)

        # Generate frames
        if include_both_normalizations:
            # For HTML export: generate both raw and normalized frame sets
            frames_raw = self._generate_frames(displacement_raw, "raw_")
            frames_norm = self._generate_frames(displacement_norm, "norm_")
            all_frames = frames_raw + frames_norm
            # Start with the appropriate frame set based on normalize setting
            initial_frames = frames_norm if self.normalize else frames_raw
            fig = go.Figure(data=initial_frames[0].data, frames=all_frames)
        else:
            # For display: just generate current mode frames
            frames = self._generate_frames(displacement, "")
            fig = go.Figure(data=frames[0].data, frames=frames)

        # Use larger range to accommodate both
        ranges = self._get_axis_range(self.coords, displacement_norm)
        freq = mode_info['freq'] if mode_info else 0.0
        ir_str = "IR Active" if mode_info and mode_info['ir_active'] else "IR Inactive"
        raman_str = "Raman Active" if mode_info and mode_info['raman_active'] else "Raman Inactive"
        intens = mode_info['ir_intens'] if mode_info else 0.0
        norm_str = " (Normalized)" if self.normalize else ""

        fig.update_layout(
            title=dict(
                text=f"<b>Mode {mode_idx}: {freq:.1f} cm⁻¹</b>{norm_str}<br><span style='font-size:12px;color:#666'>{ir_str} ({intens:.1f} km/mol) | {raman_str}</span>",
                x=0.5,
                font=dict(size=15, color='#333')
            ),
            scene=self._get_scene_layout(ranges),
            showlegend=False,
            updatemenus=[],  # Using custom JS controls instead
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=11, color='#555'),
                        prefix='Frame: ',
                        visible=True,
                        xanchor='right'
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.05,
                    y=0,
                    bgcolor='#eee',
                    bordercolor='#ccc',
                    steps=[
                        dict(
                            args=[[f"raw_{k}"], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))],
                            label=str(k),
                            method='animate'
                        )
                        for k in range(self.n_frames)
                    ]
                )
            ],
            margin=dict(l=0, r=0, t=70, b=90),
            paper_bgcolor='#fafafa'
        )

        return fig

    def export_gif(self, mode_idx, output_dir='.', fps=20, width=800, height=600):
        """Export vibrational mode animation as GIF."""
        try:
            from PIL import Image
        except ImportError:
            print("Error: Pillow is required for GIF export. Install with: pip install Pillow")
            return

        displacement = self.parser.get_displacement(mode_idx)
        mode_info = self.parser.get_mode_info(mode_idx)

        if displacement is None:
            print(f"Error: No displacement data for mode {mode_idx}")
            return

        if self.normalize:
            displacement = self._normalize_displacement(displacement)

        os.makedirs(output_dir, exist_ok=True)
        freq = mode_info['freq'] if mode_info else 0.0
        norm_suffix = '_norm' if self.normalize else ''
        gif_path = os.path.join(output_dir, f'mode_{mode_idx}_{freq:.0f}cm-1{norm_suffix}.gif')

        print(f"Rendering {self.n_frames} frames...")
        images = []

        for frame_idx in range(self.n_frames):
            phase = np.sin(2 * np.pi * frame_idx / self.n_frames)
            current_coords = self.coords + displacement * self.amplitude * phase

            # Create figure for this frame with arrows (full displacement for direction)
            fig = go.Figure()
            for trace in self._create_molecule_traces(
                current_coords,
                displacement=displacement,  # Full displacement for arrows
                show_arrows=self.show_arrows,
                arrow_scale=self.arrow_scale
            ):
                fig.add_trace(trace)

            ranges = self._get_axis_range(self.coords, displacement)
            ir_str = "IR Active" if mode_info and mode_info['ir_active'] else "IR Inactive"
            raman_str = "Raman Active" if mode_info and mode_info['raman_active'] else "Raman Inactive"
            intens = mode_info['ir_intens'] if mode_info else 0.0
            norm_str = " (Normalized)" if self.normalize else ""

            fig.update_layout(
                title=dict(
                    text=f"<b>Mode {mode_idx}: {freq:.1f} cm⁻¹</b>{norm_str}<br><span style='font-size:12px;color:#666'>{ir_str} ({intens:.1f} km/mol) | {raman_str}</span>",
                    x=0.5,
                    font=dict(size=15, color='#333')
                ),
                scene=self._get_scene_layout(ranges),
                showlegend=False,
                margin=dict(l=0, r=0, t=70, b=0),
                paper_bgcolor='#fafafa'
            )

            # Render to image
            img_bytes = fig.to_image(format='png', width=width, height=height)
            img = Image.open(BytesIO(img_bytes))
            images.append(img)

            # Progress indicator
            if (frame_idx + 1) % 10 == 0 or frame_idx == self.n_frames - 1:
                print(f"  Frame {frame_idx + 1}/{self.n_frames}")

        # Save as GIF
        duration = int(1000 / fps)  # ms per frame
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # 0 = infinite loop
        )
        print(f"Saved: {gif_path}")

    def show_mode(self, mode_idx, animate=True, speed=1.0, save_html=False, save_gif=False, output_dir='.', gif_fps=20):
        """Display a vibrational mode."""
        # Refuse early with a clear error when eigenvectors don't match the
        # parsed structure, instead of failing deep in a numpy broadcast.
        self._validated_displacement(mode_idx, strict=True)
        if animate:
            # Include both normalizations for HTML so user can toggle
            fig = self.create_animation(mode_idx, speed=speed, include_both_normalizations=save_html)
        else:
            fig = self.create_static_view(mode_idx)

        if fig is None:
            return

        if save_html:
            os.makedirs(output_dir, exist_ok=True)
            mode_info = self.parser.get_mode_info(mode_idx)
            freq = mode_info['freq'] if mode_info else 0.0
            norm_suffix = '_norm' if self.normalize else ''
            html_path = os.path.join(output_dir, f'mode_{mode_idx}_{freq:.0f}cm-1{norm_suffix}.html')

            # Generate HTML with custom looping JavaScript
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)

            # Inject JavaScript for controls
            show_arrows_js = 'true' if self.show_arrows else 'false'
            is_normalized_js = 'true' if self.normalize else 'false'
            loop_script = '''
<script>
(function() {
    var looping = false;
    var animationSpeed = 60;
    var numFrames = ''' + str(self.n_frames) + ''';
    var currentFrame = 0;
    var animationInterval = null;
    var showArrows = ''' + show_arrows_js + ''';
    var isNormalized = ''' + is_normalized_js + ''';

    function getFrameName(frameNum) {
        // Frame naming: raw_0, raw_1... or norm_0, norm_1...
        var prefix = isNormalized ? 'norm_' : 'raw_';
        return prefix + frameNum;
    }

    function updateSlider(frameNum) {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv && plotDiv.layout && plotDiv.layout.sliders && plotDiv.layout.sliders[0]) {
            Plotly.relayout(plotDiv, {'sliders[0].active': frameNum});
        }
    }

    function startLoop() {
        if (animationInterval) clearInterval(animationInterval);
        looping = true;
        animationInterval = setInterval(function() {
            if (!looping) return;
            currentFrame = (currentFrame + 1) % numFrames;
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {
                Plotly.animate(plotDiv, [getFrameName(currentFrame)], {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0},
                    mode: 'immediate'
                }).then(function() {
                    updateArrowVisibility();
                    updateSlider(currentFrame);
                });
            }
        }, animationSpeed);
        document.getElementById('loopBtn').textContent = '⏹ Stop';
        document.getElementById('loopBtn').style.backgroundColor = '#ffcccc';
    }

    function stopLoop() {
        looping = false;
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        document.getElementById('loopBtn').textContent = '🔄 Loop';
        document.getElementById('loopBtn').style.backgroundColor = '#ccffcc';
    }

    function toggleLoop() {
        if (looping) {
            stopLoop();
        } else {
            startLoop();
        }
    }

    function setSpeed(speed) {
        animationSpeed = speed;
        if (looping) {
            startLoop();
        }
    }

    function updateSliderSteps() {
        // Update slider steps to use correct frame prefix
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        var prefix = isNormalized ? 'norm_' : 'raw_';
        var newSteps = [];
        for (var i = 0; i < numFrames; i++) {
            newSteps.push({
                args: [[prefix + i], {frame: {duration: 0, redraw: true}, mode: 'immediate', transition: {duration: 0}}],
                label: String(i),
                method: 'animate'
            });
        }
        Plotly.relayout(plotDiv, {'sliders[0].steps': newSteps});
    }

    function toggleNormalize() {
        isNormalized = !isNormalized;
        var btn = document.getElementById('normBtn');
        if (isNormalized) {
            btn.textContent = '📏 Normalize: ON';
            btn.style.backgroundColor = '#d4edda';
        } else {
            btn.textContent = '📏 Normalize: OFF';
            btn.style.backgroundColor = '#f0f0f0';
        }
        // Update slider to use correct frame names
        updateSliderSteps();
        // Jump to current frame with new normalization
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            Plotly.animate(plotDiv, [getFrameName(currentFrame)], {
                frame: {duration: 0, redraw: true},
                transition: {duration: 0},
                mode: 'immediate'
            }).then(function() {
                updateArrowVisibility();
            });
        }
    }

    function updateArrowVisibility() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv || !plotDiv.data) return;

        // Find arrow traces (named 'Arrows' and 'ArrowHeads')
        var updates = [];
        var indices = [];
        for (var i = 0; i < plotDiv.data.length; i++) {
            var name = plotDiv.data[i].name;
            if (name === 'Arrows' || name === 'ArrowHeads') {
                indices.push(i);
                updates.push({visible: showArrows});
            }
        }
        if (indices.length > 0) {
            Plotly.restyle(plotDiv, {visible: showArrows}, indices);
        }
    }

    function toggleArrows() {
        showArrows = !showArrows;
        updateArrowVisibility();
        var btn = document.getElementById('arrowBtn');
        if (showArrows) {
            btn.textContent = '➡️ Arrows: ON';
            btn.style.backgroundColor = '#cce5ff';
        } else {
            btn.textContent = '➡️ Arrows: OFF';
            btn.style.backgroundColor = '#f0f0f0';
        }
    }

    // Wait for page load
    window.addEventListener('load', function() {
        var container = document.createElement('div');
        container.style.cssText = 'position:fixed; top:10px; right:10px; z-index:1000; background:white; padding:12px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.2); font-family:Arial,sans-serif; min-width:140px; max-height:calc(100vh - 20px); overflow-y:auto;';

        // Title
        var title = document.createElement('div');
        title.textContent = 'Controls';
        title.style.cssText = 'font-weight:bold; font-size:13px; margin-bottom:8px; color:#333; border-bottom:1px solid #ddd; padding-bottom:5px;';
        container.appendChild(title);

        // Loop button
        var loopBtn = document.createElement('button');
        loopBtn.id = 'loopBtn';
        loopBtn.textContent = '🔄 Loop';
        loopBtn.style.cssText = 'padding:6px 12px; margin:3px 0; cursor:pointer; border:1px solid #ccc; border-radius:4px; background:#ccffcc; font-size:13px; width:100%; text-align:left;';
        loopBtn.onclick = toggleLoop;
        container.appendChild(loopBtn);

        // Arrow toggle button
        var arrowBtn = document.createElement('button');
        arrowBtn.id = 'arrowBtn';
        arrowBtn.textContent = showArrows ? '➡️ Arrows: ON' : '➡️ Arrows: OFF';
        arrowBtn.style.cssText = 'padding:6px 12px; margin:3px 0; cursor:pointer; border:1px solid #ccc; border-radius:4px; background:' + (showArrows ? '#cce5ff' : '#f0f0f0') + '; font-size:13px; width:100%; text-align:left;';
        arrowBtn.onclick = toggleArrows;
        container.appendChild(arrowBtn);

        // Normalize toggle button
        var normBtn = document.createElement('button');
        normBtn.id = 'normBtn';
        normBtn.textContent = isNormalized ? '📏 Normalize: ON' : '📏 Normalize: OFF';
        normBtn.style.cssText = 'padding:6px 12px; margin:3px 0; cursor:pointer; border:1px solid #ccc; border-radius:4px; background:' + (isNormalized ? '#d4edda' : '#f0f0f0') + '; font-size:13px; width:100%; text-align:left;';
        normBtn.onclick = toggleNormalize;
        container.appendChild(normBtn);

        // Speed section
        var speedLabel = document.createElement('div');
        speedLabel.textContent = 'Speed:';
        speedLabel.style.cssText = 'font-size:11px; margin-top:10px; color:#666;';
        container.appendChild(speedLabel);

        var speedContainer = document.createElement('div');
        speedContainer.style.cssText = 'margin-top:5px;';
        var speeds = [{label:'0.5x', val:120}, {label:'1x', val:60}, {label:'2x', val:30}, {label:'4x', val:15}];
        speeds.forEach(function(s) {
            var btn = document.createElement('button');
            btn.textContent = s.label;
            btn.style.cssText = 'padding:4px 8px; margin:2px; cursor:pointer; border:1px solid #ccc; border-radius:3px; font-size:11px;';
            btn.onclick = function() { setSpeed(s.val); };
            speedContainer.appendChild(btn);
        });
        container.appendChild(speedContainer);

        // Divider
        var divider = document.createElement('div');
        divider.style.cssText = 'border-top:1px solid #ddd; margin:10px 0;';
        container.appendChild(divider);

        // GIF Resolution selector
        var resLabel = document.createElement('div');
        resLabel.textContent = 'GIF Resolution:';
        resLabel.style.cssText = 'font-size:11px; color:#666; margin-bottom:3px;';
        container.appendChild(resLabel);

        var resSelect = document.createElement('select');
        resSelect.id = 'gifResolution';
        resSelect.style.cssText = 'width:100%; padding:4px; margin-bottom:8px; border:1px solid #ccc; border-radius:4px; font-size:12px;';
        resSelect.innerHTML = '<option value="800x600">800x600</option><option value="1200x900">1200x900</option><option value="1600x1200" selected>1600x1200</option><option value="1920x1440">1920x1440</option>';
        container.appendChild(resSelect);

        // Include spectrum checkbox
        var specCheck = document.createElement('label');
        specCheck.style.cssText = 'display:flex; align-items:center; font-size:11px; color:#666; margin-bottom:8px; cursor:pointer;';
        specCheck.innerHTML = '<input type="checkbox" id="includeSpectrum" checked style="margin-right:6px;"> Include spectrum';
        container.appendChild(specCheck);

        // Save GIF button
        var gifBtn = document.createElement('button');
        gifBtn.id = 'gifBtn';
        gifBtn.textContent = '📷 Save GIF';
        gifBtn.style.cssText = 'padding:6px 12px; margin:3px 0; cursor:pointer; border:1px solid #ccc; border-radius:4px; background:#fff3cd; font-size:13px; width:100%; text-align:left;';
        gifBtn.onclick = captureGif;
        container.appendChild(gifBtn);

        // Status text
        var statusText = document.createElement('div');
        statusText.id = 'gifStatus';
        statusText.style.cssText = 'font-size:10px; color:#666; margin-top:5px; display:none;';
        container.appendChild(statusText);

        document.body.appendChild(container);

        // Initial arrow visibility
        setTimeout(updateArrowVisibility, 500);

        // Listen for slider changes to update currentFrame
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            plotDiv.on('plotly_sliderchange', function(e) {
                if (e && e.slider && typeof e.slider.active === 'number') {
                    currentFrame = e.slider.active;
                    updateArrowVisibility();
                }
            });
        }
    });

    // GIF capture using Plotly's toImage or html2canvas for full view
    async function captureGif() {
        var btn = document.getElementById('gifBtn');
        var status = document.getElementById('gifStatus');
        btn.disabled = true;
        btn.textContent = '⏳ Capturing...';
        btn.style.backgroundColor = '#e0e0e0';
        status.style.display = 'block';
        status.textContent = 'Loading libraries...';

        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) {
            status.textContent = 'Error: Plot not found';
            resetGifButton();
            return;
        }

        // Load gif.js and set up inline worker
        await loadGifJs();

        // Get resolution
        var resSelect = document.getElementById('gifResolution');
        var resolution = resSelect ? resSelect.value.split('x') : ['1600', '1200'];
        var gifWidth = parseInt(resolution[0]);
        var gifHeight = parseInt(resolution[1]);

        // Check if including spectrum
        var includeSpec = document.getElementById('includeSpectrum');
        var includeSpectrum = includeSpec ? includeSpec.checked : false;

        // Load html2canvas if needed
        if (includeSpectrum && !window.html2canvas) {
            var script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            document.head.appendChild(script);
            await new Promise(function(resolve) { script.onload = resolve; });
        }

        var capturedFrames = [];
        var gifDelay = Math.max(2, Math.round(animationSpeed / 10));

        // Hide control panel for cleaner capture if including spectrum
        var controlPanel = document.getElementById('modePanel');
        if (includeSpectrum && controlPanel) {
            controlPanel.style.display = 'none';
        }

        for (var i = 0; i < numFrames; i++) {
            status.textContent = 'Frame ' + (i + 1) + '/' + numFrames;

            try {
                var frameName = getFrameName(i);
                await Plotly.animate(plotDiv, [frameName], {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0},
                    mode: 'immediate'
                });

                updateArrowVisibility();
                await sleep(150);

                var dataUrl;
                if (includeSpectrum) {
                    // Capture full page including spectrum
                    var mainContainer = document.getElementById('mainContainer');
                    var canvas = await html2canvas(mainContainer || document.body, {
                        backgroundColor: '#fafafa',
                        scale: gifWidth / (mainContainer ? mainContainer.offsetWidth : window.innerWidth),
                        logging: false
                    });
                    dataUrl = canvas.toDataURL('image/png');
                } else {
                    // Just capture 3D plot
                    dataUrl = await Plotly.toImage(plotDiv, {format: 'png', width: gifWidth, height: gifHeight});
                }
                capturedFrames.push(dataUrl);
            } catch(e) {
                console.error('Frame capture error:', e);
                status.textContent = 'Error capturing frame ' + i + ': ' + e.message;
                if (controlPanel) controlPanel.style.display = '';
                resetGifButton();
                return;
            }
        }

        if (controlPanel) controlPanel.style.display = '';

        status.textContent = 'Creating GIF...';
        createGifFromFrames(capturedFrames, gifDelay, gifWidth, gifHeight, includeSpectrum);
    }

    // Inline gif.js worker to avoid CORS issues
    var gifWorkerBlob = null;
    function loadGifJs() {
        return new Promise(function(resolve, reject) {
            if (window.GIF && gifWorkerBlob) { resolve(); return; }
            var script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js';
            script.onload = function() {
                // Fetch worker script and create blob URL
                fetch('https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js')
                    .then(function(r) { return r.text(); })
                    .then(function(code) {
                        gifWorkerBlob = URL.createObjectURL(new Blob([code], {type: 'application/javascript'}));
                        resolve();
                    })
                    .catch(function(e) {
                        // Fallback: use simpler sync encoder
                        console.warn('Worker fetch failed, using fallback');
                        gifWorkerBlob = 'fallback';
                        resolve();
                    });
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    function sleep(ms) {
        return new Promise(function(resolve) { setTimeout(resolve, ms); });
    }

    function createGifFromFrames(frames, gifDelay, gifWidth, gifHeight, includeSpectrum) {
        var status = document.getElementById('gifStatus');
        var btn = document.getElementById('gifBtn');
        var frameDelay = gifDelay || 5;
        var width = gifWidth || 1600;
        var height = gifHeight || 1200;

        // Use the pre-loaded worker blob URL, or fallback
        if (gifWorkerBlob === 'fallback' || !gifWorkerBlob) {
            createGifFallback(frames);
            return;
        }

        // For full-page captures, calculate height from first frame
        var firstImg = new Image();
        firstImg.src = frames[0];

        firstImg.onload = function() {
            var actualWidth = includeSpectrum ? firstImg.width : width;
            var actualHeight = includeSpectrum ? firstImg.height : height;

            var gif = new GIF({
                workers: 2,
                quality: 10,
                width: actualWidth,
                height: actualHeight,
                workerScript: gifWorkerBlob
            });

            var loadedCount = 0;
            var images = [];

            frames.forEach(function(dataUrl, idx) {
                var img = new Image();
                img.onload = function() {
                    images[idx] = img;
                    loadedCount++;
                    if (loadedCount === frames.length) {
                        images.forEach(function(img) {
                            gif.addFrame(img, {delay: frameDelay * 10});
                        });
                        status.textContent = 'Encoding GIF...';
                        gif.render();
                    }
                };
                img.src = dataUrl;
            });

            gif.on('finished', function(blob) {
                var mode = modes.find(function(m) { return m.mode === currentMode; });
                var suffix = includeSpectrum ? '_with_spectrum' : '';
                var filename = 'mode_' + currentMode + '_' + Math.round(mode.freq) + 'cm-1' + suffix + '.gif';

                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                resetGifButton();
                status.textContent = 'GIF saved!';
                setTimeout(function() { status.style.display = 'none'; }, 2000);
            });

            gif.on('error', function(e) {
                console.error('GIF error:', e);
                status.textContent = 'Error creating GIF - trying fallback...';
                createGifFallback(frames);
            });
        };
    }

    // Fallback GIF creation using simple canvas approach
    function createGifFallback(frames) {
        var status = document.getElementById('gifStatus');
        status.textContent = 'Using fallback encoder...';

        // Simple approach: create downloadable frames or use basic encoder
        // For now, offer to download frames as a ZIP or individual images
        var downloadLinks = [];
        frames.forEach(function(dataUrl, idx) {
            downloadLinks.push({name: 'frame_' + idx.toString().padStart(3, '0') + '.png', data: dataUrl});
        });

        // Create a simple HTML page with all frames that user can save
        var html = '<!DOCTYPE html><html><head><title>Animation Frames</title></head><body>';
        html += '<h1>Animation Frames (save images and use external tool to create GIF)</h1>';
        html += '<p>Tip: Use ffmpeg or online tools like ezgif.com to combine these frames</p>';
        frames.forEach(function(dataUrl, idx) {
            html += '<div style="display:inline-block;margin:5px;">';
            html += '<img src="' + dataUrl + '" style="width:200px;border:1px solid #ccc;">';
            html += '<br><a href="' + dataUrl + '" download="frame_' + idx.toString().padStart(3, '0') + '.png">Download</a>';
            html += '</div>';
        });
        html += '</body></html>';

        var blob = new Blob([html], {type: 'text/html'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'animation_frames.html';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resetGifButton();
        status.textContent = 'Frames exported (use external tool for GIF)';
        setTimeout(function() { status.style.display = 'none'; }, 3000);
    }

    function resetGifButton() {
        var btn = document.getElementById('gifBtn');
        btn.disabled = false;
        btn.textContent = '📷 Save GIF';
        btn.style.backgroundColor = '#fff3cd';
    }
})();
</script>
'''
            # Insert script before closing body tag
            html_content = html_content.replace('</body>', loop_script + '</body>')

            with open(html_path, 'w') as f:
                f.write(html_content)
            print(f"Saved: {html_path}")

        if save_gif:
            self.export_gif(mode_idx, output_dir=output_dir, fps=gif_fps)

        if not save_html and not save_gif:
            fig.show()

    def create_all_modes_html(self, output_path='all_modes_viewer.html'):
        """Create a single HTML file with all vibrational modes."""
        import json

        print("Generating all-modes viewer...")

        # Get all modes with non-zero frequencies
        valid_modes = [m for m in self.parser.modes if m['freq'] > 1.0]
        print(f"Processing {len(valid_modes)} modes...")

        # Generate frames for all modes
        all_frames = []
        mode_info_list = []

        for i, mode in enumerate(valid_modes):
            mode_idx = mode['mode']
            displacement_raw = self._validated_displacement(mode_idx)

            if displacement_raw is None:
                continue

            displacement_norm = self._normalize_displacement(displacement_raw.copy())

            # Generate frames with mode-specific prefix
            frames_raw = self._generate_frames(displacement_raw, f"m{mode_idx}_raw_")
            frames_norm = self._generate_frames(displacement_norm, f"m{mode_idx}_norm_")
            all_frames.extend(frames_raw)
            all_frames.extend(frames_norm)

            mode_info_list.append({
                'mode': mode_idx,
                'freq': mode['freq'],
                'ir_active': mode['ir_active'],
                'ir_intens': mode['ir_intens'],
                'raman_active': mode['raman_active'],
                'raman_intens': mode.get('raman_intens', 0.0)
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(valid_modes)} modes...")

        print(f"Generated {len(all_frames)} total frames")

        # Use first mode's first frame as initial data
        if not mode_info_list:
            print("  No renderable modes (all eigenvectors mismatch the parsed "
                  "structure); skipping HTML generation.")
            return None
        first_mode = mode_info_list[0]['mode']
        initial_prefix = f"m{first_mode}_raw_" if not self.normalize else f"m{first_mode}_norm_"
        initial_frame_name = f"{initial_prefix}0"

        # Find initial frame data
        initial_data = None
        for frame in all_frames:
            if frame.name == initial_frame_name:
                initial_data = frame.data
                break

        if initial_data is None:
            initial_data = all_frames[0].data

        # Create figure
        fig = go.Figure(data=initial_data, frames=all_frames)

        # Get axis ranges (use largest displacement for range)
        max_disp = None
        for mode in mode_info_list[:5]:  # Sample first few modes
            disp = self._validated_displacement(mode['mode'])
            if disp is not None:
                disp_norm = self._normalize_displacement(disp)
                if max_disp is None or np.max(np.abs(disp_norm)) > np.max(np.abs(max_disp)):
                    max_disp = disp_norm

        ranges = self._get_axis_range(self.coords, max_disp)

        fig.update_layout(
            title=dict(
                text=f"<b>Vibrational Mode Viewer</b><br><span style='font-size:12px;color:#666'>{len(mode_info_list)} modes available</span>",
                x=0.5,
                font=dict(size=15, color='#333')
            ),
            scene=self._get_scene_layout(ranges),
            showlegend=False,
            updatemenus=[],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=11, color='#555'),
                        prefix='Frame: ',
                        visible=True,
                        xanchor='right'
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.05,
                    y=0,
                    bgcolor='#eee',
                    bordercolor='#ccc',
                    steps=[
                        dict(
                            args=[[f"m{first_mode}_raw_{k}"], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))],
                            label=str(k),
                            method='animate'
                        )
                        for k in range(self.n_frames)
                    ]
                )
            ],
            margin=dict(l=0, r=0, t=70, b=90),
            paper_bgcolor='#fafafa'
        )

        # Generate HTML
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)

        # Get unique elements for legend
        unique_elements = sorted(set(self.elements))
        element_legend = []
        for el in unique_elements:
            element_legend.append({
                'symbol': el,
                'color': ELEMENT_COLORS.get(el.upper(), ELEMENT_COLORS['DEFAULT']),
                'outline': ELEMENT_OUTLINE_COLORS.get(el.upper(), ELEMENT_OUTLINE_COLORS['DEFAULT'])
            })

        # Prepare mode data for JavaScript
        mode_data_js = json.dumps(mode_info_list)
        element_legend_js = json.dumps(element_legend)
        show_arrows_js = 'true' if self.show_arrows else 'false'
        is_normalized_js = 'true' if self.normalize else 'false'

        # Inject comprehensive JavaScript for all-modes control
        all_modes_script = '''
<style>
body {
    margin: 0;
    padding: 0;
}
#mainContainer {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}
#viewer3d {
    flex: 1;
    min-height: 50vh;
    position: relative;
}
#spectrumContainer {
    height: 280px;
    background: white;
    border-top: 2px solid #ddd;
    display: flex;
}
#spectrumPlot {
    flex: 1;
}
#spectrumControls {
    width: 120px;
    padding: 10px;
    background: #f8f8f8;
    border-left: 1px solid #ddd;
    font-family: Arial, sans-serif;
    font-size: 12px;
}
#spectrumControls .section-title {
    font-weight: bold;
    font-size: 11px;
    color: #333;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #ddd;
}
#spectrumControls button {
    width: 100%;
    padding: 6px 8px;
    margin: 3px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    text-align: left;
}
#spectrumControls button.active {
    background: #4a90d9;
    color: white;
    border-color: #3a7bc8;
}
#modePanel {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    background: white;
    padding: 12px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    font-family: Arial, sans-serif;
    max-height: calc(100vh - 320px);
    overflow-y: auto;
    min-width: 200px;
}
#modePanel .panel-title {
    font-weight: bold;
    font-size: 14px;
    margin-bottom: 10px;
    color: #333;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
}
#modePanel select, #modePanel button {
    width: 100%;
    padding: 6px 10px;
    margin: 3px 0;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
}
#modePanel select {
    background: white;
}
#modePanel .section-label {
    font-size: 11px;
    color: #666;
    margin-top: 10px;
    margin-bottom: 3px;
}
#modePanel .mode-info {
    font-size: 11px;
    color: #444;
    background: #f5f5f5;
    padding: 8px;
    border-radius: 4px;
    margin: 8px 0;
}
#modePanel .speed-btns {
    display: flex;
    gap: 3px;
}
#modePanel .speed-btns button {
    width: auto;
    flex: 1;
    padding: 4px 6px;
    font-size: 11px;
}
#elementLegend {
    position: fixed;
    top: 80px;
    right: 10px;
    z-index: 1000;
    background: white;
    padding: 10px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-size: 11px;
}
#elementLegend .legend-title {
    font-weight: bold;
    font-size: 11px;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #ddd;
    color: #333;
}
.element-item {
    display: flex;
    align-items: center;
    margin: 4px 0;
}
.element-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 8px;
    border: 2px solid;
}
.element-name {
    color: #444;
}
</style>
<script>
(function() {
    var modes = ''' + mode_data_js + ''';
    var elementLegend = ''' + element_legend_js + ''';
    var numFrames = ''' + str(self.n_frames) + ''';
    var currentMode = modes[0].mode;
    var currentFrame = 0;
    var looping = false;
    var animationSpeed = 60;
    var animationInterval = null;
    var showArrows = ''' + show_arrows_js + ''';
    var isNormalized = ''' + is_normalized_js + ''';

    function getFrameName(frameNum) {
        var prefix = isNormalized ? 'norm_' : 'raw_';
        return 'm' + currentMode + '_' + prefix + frameNum;
    }

    function updateSlider(frameNum) {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv && plotDiv.layout && plotDiv.layout.sliders && plotDiv.layout.sliders[0]) {
            Plotly.relayout(plotDiv, {'sliders[0].active': frameNum});
        }
    }

    function updateSliderSteps() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        var prefix = isNormalized ? 'norm_' : 'raw_';
        var newSteps = [];
        for (var i = 0; i < numFrames; i++) {
            newSteps.push({
                args: [['m' + currentMode + '_' + prefix + i], {frame: {duration: 0, redraw: true}, mode: 'immediate', transition: {duration: 0}}],
                label: String(i),
                method: 'animate'
            });
        }
        Plotly.relayout(plotDiv, {'sliders[0].steps': newSteps});
    }

    function updateArrowVisibility() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv || !plotDiv.data) return;
        var indices = [];
        for (var i = 0; i < plotDiv.data.length; i++) {
            var name = plotDiv.data[i].name;
            if (name === 'Arrows' || name === 'ArrowHeads') {
                indices.push(i);
            }
        }
        if (indices.length > 0) {
            Plotly.restyle(plotDiv, {visible: showArrows}, indices);
        }
    }

    function updateModeInfo() {
        var info = document.getElementById('modeInfo');
        var mode = modes.find(function(m) { return m.mode === currentMode; });
        if (mode && info) {
            var irStr = mode.ir_active ? 'Yes (' + mode.ir_intens.toFixed(1) + ' km/mol)' : 'No';
            var ramanStr = mode.raman_active ? 'Yes' : 'No';
            info.innerHTML = '<b>Mode ' + mode.mode + '</b><br>' +
                '<b>' + mode.freq.toFixed(1) + ' cm⁻¹</b><br>' +
                'IR: ' + irStr + '<br>' +
                'Raman: ' + ramanStr;
        }
    }

    function updateTitle() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        var mode = modes.find(function(m) { return m.mode === currentMode; });
        if (plotDiv && mode) {
            var irStr = mode.ir_active ? 'IR Active' : 'IR Inactive';
            var ramanStr = mode.raman_active ? 'Raman Active' : 'Raman Inactive';
            var normStr = isNormalized ? ' (Normalized)' : '';
            var title = '<b>Mode ' + mode.mode + ': ' + mode.freq.toFixed(1) + ' cm⁻¹</b>' + normStr +
                '<br><span style="font-size:12px;color:#666">' + irStr + ' (' + mode.ir_intens.toFixed(1) + ' km/mol) | ' + ramanStr + '</span>';
            Plotly.relayout(plotDiv, {'title.text': title});
        }
    }

    function changeMode(modeIdx) {
        currentMode = modeIdx;
        currentFrame = 0;
        updateSliderSteps();
        updateModeInfo();
        updateTitle();
        updateSpectrumHighlight();

        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            Plotly.animate(plotDiv, [getFrameName(0)], {
                frame: {duration: 0, redraw: true},
                transition: {duration: 0},
                mode: 'immediate'
            }).then(function() {
                updateArrowVisibility();
                updateSlider(0);
            });
        }
    }

    function startLoop() {
        if (animationInterval) clearInterval(animationInterval);
        looping = true;
        animationInterval = setInterval(function() {
            if (!looping) return;
            currentFrame = (currentFrame + 1) % numFrames;
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {
                Plotly.animate(plotDiv, [getFrameName(currentFrame)], {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0},
                    mode: 'immediate'
                }).then(function() {
                    updateArrowVisibility();
                    updateSlider(currentFrame);
                });
            }
        }, animationSpeed);
        document.getElementById('loopBtn').textContent = '⏹ Stop';
        document.getElementById('loopBtn').style.backgroundColor = '#ffcccc';
    }

    function stopLoop() {
        looping = false;
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        document.getElementById('loopBtn').textContent = '🔄 Loop';
        document.getElementById('loopBtn').style.backgroundColor = '#ccffcc';
    }

    function toggleLoop() {
        if (looping) stopLoop();
        else startLoop();
    }

    function setSpeed(speed) {
        animationSpeed = speed;
        if (looping) startLoop();
    }

    function toggleArrows() {
        showArrows = !showArrows;
        updateArrowVisibility();
        var btn = document.getElementById('arrowBtn');
        btn.textContent = showArrows ? '➡️ Arrows: ON' : '➡️ Arrows: OFF';
        btn.style.backgroundColor = showArrows ? '#cce5ff' : '#f0f0f0';
    }

    function toggleNormalize() {
        isNormalized = !isNormalized;
        var btn = document.getElementById('normBtn');
        btn.textContent = isNormalized ? '📏 Normalize: ON' : '📏 Normalize: OFF';
        btn.style.backgroundColor = isNormalized ? '#d4edda' : '#f0f0f0';
        updateSliderSteps();
        updateTitle();

        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            Plotly.animate(plotDiv, [getFrameName(currentFrame)], {
                frame: {duration: 0, redraw: true},
                transition: {duration: 0},
                mode: 'immediate'
            }).then(function() {
                updateArrowVisibility();
            });
        }
    }

    // GIF capture
    var gifWorkerBlob = null;

    function loadGifJs() {
        return new Promise(function(resolve, reject) {
            if (window.GIF && gifWorkerBlob) { resolve(); return; }
            var script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js';
            script.onload = function() {
                fetch('https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js')
                    .then(function(r) { return r.text(); })
                    .then(function(code) {
                        gifWorkerBlob = URL.createObjectURL(new Blob([code], {type: 'application/javascript'}));
                        resolve();
                    })
                    .catch(function(e) {
                        console.warn('Worker fetch failed, using fallback');
                        gifWorkerBlob = 'fallback';
                        resolve();
                    });
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    function sleep(ms) {
        return new Promise(function(resolve) { setTimeout(resolve, ms); });
    }

    async function captureGif() {
        var btn = document.getElementById('gifBtn');
        var status = document.getElementById('gifStatus');
        btn.disabled = true;
        btn.textContent = '⏳ Capturing...';
        btn.style.backgroundColor = '#e0e0e0';
        status.style.display = 'block';
        status.textContent = 'Loading GIF library...';

        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) {
            status.textContent = 'Error: Plot not found';
            resetGifButton();
            return;
        }

        await loadGifJs();

        // Get resolution settings
        var resSelect = document.getElementById('gifResolution');
        var resolution = resSelect ? resSelect.value.split('x') : ['1600', '1200'];
        var gifWidth = parseInt(resolution[0]);
        var gifHeight = parseInt(resolution[1]);

        // Check include spectrum
        var specCheckbox = document.getElementById('includeSpectrum');
        var includeSpectrum = specCheckbox ? specCheckbox.checked : false;
        var spectrumDiv = document.getElementById('spectrumPlot');
        var spectrumHeight = 200;
        var totalHeight = includeSpectrum && spectrumDiv ? gifHeight + spectrumHeight : gifHeight;

        var capturedFrames = [];
        var gifDelay = Math.max(2, Math.round(animationSpeed / 10));

        for (var i = 0; i < numFrames; i++) {
            status.textContent = 'Frame ' + (i + 1) + '/' + numFrames;

            try {
                var frameName = getFrameName(i);
                await Plotly.animate(plotDiv, [frameName], {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0},
                    mode: 'immediate'
                });
                updateArrowVisibility();
                await sleep(200);

                // Create composite canvas for all cases (to add element legend)
                var canvas = document.createElement('canvas');
                canvas.width = gifWidth;
                canvas.height = totalHeight;
                var ctx = canvas.getContext('2d');
                ctx.fillStyle = '#fafafa';
                ctx.fillRect(0, 0, gifWidth, totalHeight);

                // Capture 3D plot
                var plotImg = await Plotly.toImage(plotDiv, {format: 'png', width: gifWidth, height: gifHeight});
                var img3d = new Image();
                await new Promise(function(resolve) { img3d.onload = resolve; img3d.src = plotImg; });
                ctx.drawImage(img3d, 0, 0, gifWidth, gifHeight);

                if (includeSpectrum && spectrumDiv) {
                    // Capture spectrum
                    var specImg = await Plotly.toImage(spectrumDiv, {format: 'png', width: gifWidth, height: spectrumHeight});
                    var imgSpec = new Image();
                    await new Promise(function(resolve) { imgSpec.onload = resolve; imgSpec.src = specImg; });
                    ctx.drawImage(imgSpec, 0, gifHeight, gifWidth, spectrumHeight);
                }

                // Draw element legend on GIF
                var legendX = gifWidth - 90;
                var legendY = 10;
                var legendPadding = 8;
                var itemHeight = 18;
                var legendHeight = legendPadding * 2 + elementLegend.length * itemHeight + 20;

                ctx.fillStyle = 'rgba(255,255,255,0.95)';
                ctx.fillRect(legendX, legendY, 80, legendHeight);
                ctx.strokeStyle = '#ddd';
                ctx.strokeRect(legendX, legendY, 80, legendHeight);

                ctx.fillStyle = '#333';
                ctx.font = 'bold 11px Arial';
                ctx.fillText('Elements', legendX + legendPadding, legendY + 15);

                elementLegend.forEach(function(el, idx) {
                    var y = legendY + 25 + idx * itemHeight;
                    ctx.beginPath();
                    ctx.arc(legendX + legendPadding + 7, y + 5, 6, 0, Math.PI * 2);
                    ctx.fillStyle = el.color;
                    ctx.fill();
                    ctx.strokeStyle = el.outline;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.fillStyle = '#444';
                    ctx.font = '11px Arial';
                    ctx.fillText(el.symbol, legendX + legendPadding + 20, y + 9);
                });

                capturedFrames.push(canvas.toDataURL('image/png'));
            } catch(e) {
                console.error('Frame capture error:', e);
                status.textContent = 'Error capturing frame ' + i + ': ' + e.message;
                resetGifButton();
                return;
            }
        }

        status.textContent = 'Creating GIF...';
        createGifFromFrames(capturedFrames, gifDelay, gifWidth, totalHeight);
    }

    function createGifFromFrames(frames, gifDelay, gifWidth, gifHeight) {
        var status = document.getElementById('gifStatus');
        var frameDelay = gifDelay || 5;
        var width = gifWidth || 1600;
        var height = gifHeight || 1200;

        if (gifWorkerBlob === 'fallback' || !gifWorkerBlob) {
            createGifFallback(frames);
            return;
        }

        var gif = new GIF({
            workers: 2,
            quality: 10,
            width: width,
            height: height,
            workerScript: gifWorkerBlob
        });

        var loadedCount = 0;
        var images = [];

        frames.forEach(function(dataUrl, idx) {
            var img = new Image();
            img.onload = function() {
                images[idx] = img;
                loadedCount++;
                if (loadedCount === frames.length) {
                    images.forEach(function(img) {
                        gif.addFrame(img, {delay: frameDelay * 10});
                    });
                    status.textContent = 'Encoding GIF...';
                    gif.render();
                }
            };
            img.src = dataUrl;
        });

        gif.on('finished', function(blob) {
            var mode = modes.find(function(m) { return m.mode === currentMode; });
            var filename = 'mode_' + currentMode + '_' + Math.round(mode.freq) + 'cm-1.gif';

            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            resetGifButton();
            status.textContent = 'GIF saved!';
            setTimeout(function() { status.style.display = 'none'; }, 2000);
        });

        gif.on('error', function(e) {
            console.error('GIF error:', e);
            status.textContent = 'Error creating GIF - trying fallback...';
            createGifFallback(frames);
        });
    }

    function createGifFallback(frames) {
        var status = document.getElementById('gifStatus');
        status.textContent = 'Exporting frames...';

        var html = '<!DOCTYPE html><html><head><title>Animation Frames</title></head><body>';
        html += '<h1>Animation Frames for Mode ' + currentMode + '</h1>';
        html += '<p>Use ffmpeg or ezgif.com to combine these frames into a GIF</p>';
        frames.forEach(function(dataUrl, idx) {
            html += '<div style="display:inline-block;margin:5px;">';
            html += '<img src="' + dataUrl + '" style="width:200px;border:1px solid #ccc;">';
            html += '<br><a href="' + dataUrl + '" download="frame_' + idx.toString().padStart(3, '0') + '.png">Download</a>';
            html += '</div>';
        });
        html += '</body></html>';

        var blob = new Blob([html], {type: 'text/html'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'mode_' + currentMode + '_frames.html';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resetGifButton();
        status.textContent = 'Frames exported';
        setTimeout(function() { status.style.display = 'none'; }, 3000);
    }

    function resetGifButton() {
        var btn = document.getElementById('gifBtn');
        btn.disabled = false;
        btn.textContent = '📷 Save GIF';
        btn.style.backgroundColor = '#fff3cd';
    }

    // Spectrum display state
    var showIR = true;
    var showRaman = true;

    function renderSpectrum() {
        var spectrumDiv = document.getElementById('spectrumPlot');
        if (!spectrumDiv) return;

        var traces = [];
        var annotations = [];

        // Prepare data
        var irFreqs = [], irIntens = [], irModes = [];
        var ramanFreqs = [], ramanIntens = [], ramanModes = [];

        modes.forEach(function(m) {
            if (m.ir_active && m.ir_intens > 0) {
                irFreqs.push(m.freq);
                irIntens.push(m.ir_intens);
                irModes.push(m.mode);
            }
            if (m.raman_active && m.raman_intens > 0) {
                ramanFreqs.push(m.freq);
                ramanIntens.push(m.raman_intens);
                ramanModes.push(m.mode);
            }
        });

        // Normalize Raman intensities to similar scale as IR for display
        var maxIR = Math.max.apply(null, irIntens) || 1;
        var maxRaman = Math.max.apply(null, ramanIntens) || 1;
        var ramanScale = maxIR / maxRaman;

        if (showIR && irFreqs.length > 0) {
            // Create smooth IR spectrum using Lorentzian broadening
            var irSpectrum = createBroadenedSpectrum(irFreqs, irIntens, 8);
            traces.push({
                x: irSpectrum.x,
                y: irSpectrum.y,
                type: 'scatter',
                mode: 'lines',
                name: 'IR',
                line: {color: 'steelblue', width: 1.5},
                fill: 'tozeroy',
                fillcolor: 'rgba(70, 130, 180, 0.15)',
                hovertemplate: '%{x:.1f} cm⁻¹<extra>IR</extra>'
            });
            // Add markers for peaks (clickable)
            traces.push({
                x: irFreqs,
                y: irIntens,
                type: 'scatter',
                mode: 'markers',
                name: 'IR peaks',
                marker: {color: 'steelblue', size: 7, symbol: 'circle'},
                text: irModes.map(function(m) { return 'Mode ' + m; }),
                customdata: irModes,
                hovertemplate: '%{text}<br>%{x:.1f} cm⁻¹<br>%{y:.1f} km/mol<extra></extra>'
            });
        }

        if (showRaman && ramanFreqs.length > 0) {
            var scaledRamanIntens = ramanIntens.map(function(v) { return v * ramanScale; });
            var ramanSpectrum = createBroadenedSpectrum(ramanFreqs, scaledRamanIntens, 8);
            traces.push({
                x: ramanSpectrum.x,
                y: ramanSpectrum.y,
                type: 'scatter',
                mode: 'lines',
                name: 'Raman',
                line: {color: 'darkorange', width: 1.5},
                fill: 'tozeroy',
                fillcolor: 'rgba(255, 140, 0, 0.15)',
                hovertemplate: '%{x:.1f} cm⁻¹<extra>Raman</extra>'
            });
            traces.push({
                x: ramanFreqs,
                y: scaledRamanIntens,
                type: 'scatter',
                mode: 'markers',
                name: 'Raman peaks',
                marker: {color: 'darkorange', size: 7, symbol: 'diamond'},
                text: ramanModes.map(function(m) { return 'Mode ' + m; }),
                customdata: ramanModes,
                hovertemplate: '%{text}<br>%{x:.1f} cm⁻¹<br>Raman<extra></extra>'
            });
        }

        // Add vertical line for current mode
        var currentModeData = modes.find(function(m) { return m.mode === currentMode; });
        if (currentModeData) {
            traces.push({
                x: [currentModeData.freq, currentModeData.freq],
                y: [0, maxIR * 1.1],
                type: 'scatter',
                mode: 'lines',
                name: 'Current',
                line: {color: '#2a9d8f', width: 3, dash: 'dash'},
                hoverinfo: 'skip'
            });
        }

        var maxFreq = Math.max.apply(null, modes.map(function(m) { return m.freq; }));
        var layout = {
            margin: {l: 60, r: 120, t: 30, b: 50},
            xaxis: {
                title: {text: 'Wavenumber (cm⁻¹)', font: {size: 12}},
                range: [maxFreq + 100, 0],  // Inverted x-axis (standard for spectroscopy)
                gridcolor: 'rgba(0, 0, 0, 0.1)',
                showgrid: true,
                dtick: 500,  // Tick every 500 cm-1
                zeroline: false
            },
            yaxis: {
                title: {text: 'Intensity', font: {size: 12}},
                gridcolor: 'rgba(0, 0, 0, 0.1)',
                showgrid: true,
                zeroline: true,
                zerolinecolor: '#ccc'
            },
            showlegend: true,
            legend: {x: 1.02, xanchor: 'left', y: 1, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ddd', borderwidth: 1},
            paper_bgcolor: '#fafafa',
            plot_bgcolor: 'white',
            hovermode: 'closest'
        };

        Plotly.react(spectrumDiv, traces, layout).then(function() {
            // Add click handler for peaks
            spectrumDiv.on('plotly_click', function(data) {
                if (data.points && data.points[0] && data.points[0].customdata) {
                    var clickedMode = data.points[0].customdata;
                    changeMode(clickedMode);
                    document.getElementById('modeSelect').value = clickedMode;
                }
            });
        });
    }

    function createBroadenedSpectrum(freqs, intens, gamma) {
        // Create Lorentzian-broadened spectrum
        var minFreq = 0;
        var maxFreq = Math.max.apply(null, freqs) + 200;
        var nPoints = 500;
        var step = (maxFreq - minFreq) / nPoints;
        var x = [], y = [];

        for (var i = 0; i <= nPoints; i++) {
            var freq = minFreq + i * step;
            x.push(freq);
            var intensity = 0;
            for (var j = 0; j < freqs.length; j++) {
                // Lorentzian: I / (1 + ((x - x0) / gamma)^2)
                var diff = freq - freqs[j];
                intensity += intens[j] / (1 + Math.pow(diff / gamma, 2));
            }
            y.push(intensity);
        }
        return {x: x, y: y};
    }

    function updateSpectrumHighlight() {
        renderSpectrum();
    }

    function setSpectrumView(view) {
        showIR = (view === 'ir' || view === 'both');
        showRaman = (view === 'raman' || view === 'both');

        document.getElementById('irBtn').className = showIR && !showRaman ? 'active' : '';
        document.getElementById('ramanBtn').className = !showIR && showRaman ? 'active' : '';
        document.getElementById('bothBtn').className = showIR && showRaman ? 'active' : '';

        renderSpectrum();
    }

    // Build UI on load
    window.addEventListener('load', function() {
        // Restructure page layout
        var plotlyDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotlyDiv) {
            // Create main container
            var mainContainer = document.createElement('div');
            mainContainer.id = 'mainContainer';

            // Viewer container
            var viewer3d = document.createElement('div');
            viewer3d.id = 'viewer3d';

            // Move plotly div into viewer
            plotlyDiv.parentNode.insertBefore(mainContainer, plotlyDiv);
            viewer3d.appendChild(plotlyDiv);
            mainContainer.appendChild(viewer3d);

            // Spectrum container
            var spectrumContainer = document.createElement('div');
            spectrumContainer.id = 'spectrumContainer';

            var spectrumPlot = document.createElement('div');
            spectrumPlot.id = 'spectrumPlot';
            spectrumContainer.appendChild(spectrumPlot);

            // Spectrum controls
            var spectrumControls = document.createElement('div');
            spectrumControls.id = 'spectrumControls';

            var specTitle = document.createElement('div');
            specTitle.className = 'section-title';
            specTitle.textContent = 'Spectrum';
            spectrumControls.appendChild(specTitle);

            var irBtn = document.createElement('button');
            irBtn.id = 'irBtn';
            irBtn.textContent = '📊 IR Only';
            irBtn.onclick = function() { setSpectrumView('ir'); };
            spectrumControls.appendChild(irBtn);

            var ramanBtn = document.createElement('button');
            ramanBtn.id = 'ramanBtn';
            ramanBtn.textContent = '📊 Raman Only';
            ramanBtn.onclick = function() { setSpectrumView('raman'); };
            spectrumControls.appendChild(ramanBtn);

            var bothBtn = document.createElement('button');
            bothBtn.id = 'bothBtn';
            bothBtn.className = 'active';
            bothBtn.textContent = '📊 Both';
            bothBtn.onclick = function() { setSpectrumView('both'); };
            spectrumControls.appendChild(bothBtn);

            spectrumContainer.appendChild(spectrumControls);
            mainContainer.appendChild(spectrumContainer);

            // Resize plotly to fit
            Plotly.relayout(plotlyDiv, {autosize: true});
        }
        // Mode selection panel (left side)
        var modePanel = document.createElement('div');
        modePanel.id = 'modePanel';

        var title = document.createElement('div');
        title.className = 'panel-title';
        title.textContent = 'Mode Selection';
        modePanel.appendChild(title);

        // Mode dropdown
        var modeSelect = document.createElement('select');
        modeSelect.id = 'modeSelect';
        modes.forEach(function(m) {
            var opt = document.createElement('option');
            opt.value = m.mode;
            opt.textContent = 'Mode ' + m.mode + ': ' + m.freq.toFixed(1) + ' cm⁻¹';
            modeSelect.appendChild(opt);
        });
        modeSelect.onchange = function() { changeMode(parseInt(this.value)); };
        modePanel.appendChild(modeSelect);

        // Mode info display
        var modeInfo = document.createElement('div');
        modeInfo.id = 'modeInfo';
        modeInfo.className = 'mode-info';
        modePanel.appendChild(modeInfo);

        // Playback controls
        var playLabel = document.createElement('div');
        playLabel.className = 'section-label';
        playLabel.textContent = 'Playback';
        modePanel.appendChild(playLabel);

        var loopBtn = document.createElement('button');
        loopBtn.id = 'loopBtn';
        loopBtn.textContent = '🔄 Loop';
        loopBtn.style.backgroundColor = '#ccffcc';
        loopBtn.onclick = toggleLoop;
        modePanel.appendChild(loopBtn);

        // Speed buttons
        var speedLabel = document.createElement('div');
        speedLabel.className = 'section-label';
        speedLabel.textContent = 'Speed';
        modePanel.appendChild(speedLabel);

        var speedDiv = document.createElement('div');
        speedDiv.className = 'speed-btns';
        [{label:'0.5x', val:120}, {label:'1x', val:60}, {label:'2x', val:30}, {label:'4x', val:15}].forEach(function(s) {
            var btn = document.createElement('button');
            btn.textContent = s.label;
            btn.onclick = function() { setSpeed(s.val); };
            speedDiv.appendChild(btn);
        });
        modePanel.appendChild(speedDiv);

        // Display options
        var dispLabel = document.createElement('div');
        dispLabel.className = 'section-label';
        dispLabel.textContent = 'Display';
        modePanel.appendChild(dispLabel);

        var arrowBtn = document.createElement('button');
        arrowBtn.id = 'arrowBtn';
        arrowBtn.textContent = showArrows ? '➡️ Arrows: ON' : '➡️ Arrows: OFF';
        arrowBtn.style.backgroundColor = showArrows ? '#cce5ff' : '#f0f0f0';
        arrowBtn.onclick = toggleArrows;
        modePanel.appendChild(arrowBtn);

        var normBtn = document.createElement('button');
        normBtn.id = 'normBtn';
        normBtn.textContent = isNormalized ? '📏 Normalize: ON' : '📏 Normalize: OFF';
        normBtn.style.backgroundColor = isNormalized ? '#d4edda' : '#f0f0f0';
        normBtn.onclick = toggleNormalize;
        modePanel.appendChild(normBtn);

        // Export section
        var exportLabel = document.createElement('div');
        exportLabel.className = 'section-label';
        exportLabel.textContent = 'Export';
        modePanel.appendChild(exportLabel);

        // GIF Resolution selector
        var resLabel = document.createElement('div');
        resLabel.textContent = 'Resolution:';
        resLabel.style.cssText = 'font-size:10px; color:#666; margin-bottom:2px;';
        modePanel.appendChild(resLabel);

        var resSelect = document.createElement('select');
        resSelect.id = 'gifResolution';
        resSelect.style.cssText = 'width:100%; padding:4px; margin-bottom:6px; border:1px solid #ccc; border-radius:4px; font-size:11px;';
        resSelect.innerHTML = '<option value="800x600">800x600</option><option value="1200x900">1200x900</option><option value="1600x1200" selected>1600x1200</option><option value="1920x1440">1920x1440</option>';
        modePanel.appendChild(resSelect);

        // Include spectrum checkbox
        var specCheck = document.createElement('label');
        specCheck.style.cssText = 'display:flex; align-items:center; font-size:10px; color:#666; margin-bottom:6px; cursor:pointer;';
        specCheck.innerHTML = '<input type="checkbox" id="includeSpectrum" checked style="margin-right:5px;"> Include spectrum';
        modePanel.appendChild(specCheck);

        var gifBtn = document.createElement('button');
        gifBtn.id = 'gifBtn';
        gifBtn.textContent = '📷 Save GIF';
        gifBtn.style.backgroundColor = '#fff3cd';
        gifBtn.onclick = captureGif;
        modePanel.appendChild(gifBtn);

        var gifStatus = document.createElement('div');
        gifStatus.id = 'gifStatus';
        gifStatus.style.cssText = 'font-size:10px; color:#666; margin-top:5px; display:none;';
        modePanel.appendChild(gifStatus);

        document.body.appendChild(modePanel);

        // Element legend
        var elemLegendDiv = document.createElement('div');
        elemLegendDiv.id = 'elementLegend';
        var elemTitle = document.createElement('div');
        elemTitle.className = 'legend-title';
        elemTitle.textContent = 'Elements';
        elemLegendDiv.appendChild(elemTitle);

        elementLegend.forEach(function(el) {
            var item = document.createElement('div');
            item.className = 'element-item';
            item.innerHTML = '<div class="element-dot" style="background:' + el.color + '; border-color:' + el.outline + ';"></div>' +
                             '<span class="element-name">' + el.symbol + '</span>';
            elemLegendDiv.appendChild(item);
        });
        document.body.appendChild(elemLegendDiv);

        // Initialize
        setTimeout(function() {
            updateModeInfo();
            updateArrowVisibility();
            renderSpectrum();

            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {
                plotDiv.on('plotly_sliderchange', function(e) {
                    if (e && e.slider && typeof e.slider.active === 'number') {
                        currentFrame = e.slider.active;
                        updateArrowVisibility();
                    }
                });
            }
        }, 500);
    });
})();
</script>
'''
        # Insert script before closing body tag
        html_content = html_content.replace('</body>', all_modes_script + '</body>')

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Saved: {output_path}")
        return output_path

    def create_multipanel_html(self, output_path='multipanel_viewer.html'):
        """Create a flexible multi-panel HTML viewer for comparing modes."""
        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)

        print("Generating multi-panel viewer...")

        # Get all modes with non-zero frequencies
        valid_modes = [m for m in self.parser.modes if m['freq'] > 1.0]
        print(f"Processing {len(valid_modes)} modes...")

        # Generate frame data for all modes
        all_frames_data = {}
        mode_info_list = []

        for i, mode in enumerate(valid_modes):
            mode_idx = mode['mode']
            displacement_raw = self._validated_displacement(mode_idx)

            if displacement_raw is None:
                continue

            displacement_norm = self._normalize_displacement(displacement_raw.copy())

            # Store frame data for each mode (as serializable format)
            frames_raw = []
            frames_norm = []

            for frame_idx in range(self.n_frames):
                phase = np.cos(2 * np.pi * frame_idx / self.n_frames)

                # Raw frames
                current_coords = self.coords + displacement_raw * self.amplitude * phase
                frame_data = self._create_molecule_traces(
                    current_coords,
                    displacement=displacement_raw * phase,
                    show_arrows=self.show_arrows,
                    arrow_scale=self.arrow_scale
                )
                frames_raw.append([trace.to_plotly_json() for trace in frame_data])

                # Normalized frames
                current_coords_norm = self.coords + displacement_norm * self.amplitude * phase
                frame_data_norm = self._create_molecule_traces(
                    current_coords_norm,
                    displacement=displacement_norm * phase,
                    show_arrows=self.show_arrows,
                    arrow_scale=self.arrow_scale
                )
                frames_norm.append([trace.to_plotly_json() for trace in frame_data_norm])

            all_frames_data[mode_idx] = {
                'raw': frames_raw,
                'norm': frames_norm
            }

            mode_info_list.append({
                'mode': mode_idx,
                'freq': mode['freq'],
                'ir_active': mode['ir_active'],
                'ir_intens': mode['ir_intens'],
                'raman_active': mode['raman_active'],
                'raman_intens': mode.get('raman_intens', 0.0)
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(valid_modes)} modes...")

        print(f"Prepared data for {len(mode_info_list)} modes")

        # Get axis ranges
        max_disp = None
        for mode in mode_info_list[:5]:
            disp = self._validated_displacement(mode['mode'])
            if disp is not None:
                disp_norm = self._normalize_displacement(disp)
                if max_disp is None or np.max(np.abs(disp_norm)) > np.max(np.abs(max_disp)):
                    max_disp = disp_norm

        ranges = self._get_axis_range(self.coords, max_disp)
        scene_layout = self._get_scene_layout(ranges)

        # Get unique elements for legend
        unique_elements = sorted(set(self.elements))
        element_legend = []
        for el in unique_elements:
            element_legend.append({
                'symbol': el,
                'color': ELEMENT_COLORS.get(el.upper(), ELEMENT_COLORS['DEFAULT']),
                'outline': ELEMENT_OUTLINE_COLORS.get(el.upper(), ELEMENT_OUTLINE_COLORS['DEFAULT'])
            })

        # Prepare data for JavaScript
        mode_data_js = json.dumps(mode_info_list, cls=NumpyEncoder)
        frames_data_js = json.dumps(all_frames_data, cls=NumpyEncoder)
        scene_layout_js = json.dumps(scene_layout, cls=NumpyEncoder)
        element_legend_js = json.dumps(element_legend)
        show_arrows_js = 'true' if self.show_arrows else 'false'
        is_normalized_js = 'true' if self.normalize else 'false'
        n_frames_js = str(self.n_frames)
        if not mode_info_list:
            print("  No renderable modes (all eigenvectors mismatch the parsed "
                  "structure); skipping HTML generation.")
            return None
        first_mode = mode_info_list[0]['mode']

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi-Panel Vibrational Mode Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; background: #f0f0f0; }}

        #toolbar {{
            background: white;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        #toolbar button {{
            padding: 8px 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 13px;
        }}
        #toolbar button:hover {{ background: #f5f5f5; }}
        #toolbar button.active {{ background: #4a90d9; color: white; border-color: #3a7bc8; }}
        #toolbar .separator {{ width: 1px; height: 24px; background: #ddd; }}
        #toolbar .label {{ font-size: 12px; color: #666; }}

        #panelContainer {{
            display: grid;
            gap: 10px;
            padding: 10px;
            min-height: calc(100vh - 350px);
        }}
        .grid-1 {{ grid-template-columns: 1fr; }}
        .grid-2 {{ grid-template-columns: 1fr 1fr; }}
        .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
        .grid-4 {{ grid-template-columns: 1fr 1fr; }}
        .grid-6 {{ grid-template-columns: 1fr 1fr 1fr; }}

        .panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 350px;
        }}
        .panel-header {{
            padding: 8px 12px;
            background: #f8f8f8;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .panel-header select {{
            flex: 1;
            padding: 5px 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 12px;
        }}
        .panel-header .close-btn {{
            background: none;
            border: none;
            cursor: pointer;
            color: #999;
            font-size: 18px;
            padding: 0 5px;
        }}
        .panel-header .close-btn:hover {{ color: #e63946; }}
        .panel-viewer {{
            flex: 1;
            min-height: 300px;
            cursor: pointer;
        }}
        .panel-info {{
            padding: 6px 12px;
            background: #fafafa;
            border-top: 1px solid #eee;
            font-size: 11px;
            color: #666;
        }}
        .panel.active {{
            box-shadow: 0 0 0 3px #4a90d9, 0 2px 8px rgba(0,0,0,0.1);
        }}
        .panel.active .panel-header {{
            background: #e8f4fc;
        }}

        #spectrumContainer {{
            background: white;
            border-top: 2px solid #ddd;
            display: flex;
            height: 250px;
        }}
        #spectrumPlot {{ flex: 1; }}
        #spectrumControls {{
            width: 120px;
            padding: 10px;
            background: #f8f8f8;
            border-left: 1px solid #ddd;
            font-size: 12px;
        }}
        #spectrumControls .section-title {{
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #ddd;
        }}
        #spectrumControls button {{
            width: 100%;
            padding: 6px 8px;
            margin: 3px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
        }}
        #spectrumControls button.active {{
            background: #4a90d9;
            color: white;
        }}

        .add-panel-placeholder {{
            background: #f5f5f5;
            border: 2px dashed #ccc;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            min-height: 350px;
            transition: all 0.2s;
        }}
        .add-panel-placeholder:hover {{
            border-color: #4a90d9;
            background: #f0f7ff;
        }}
        .add-panel-placeholder span {{
            font-size: 48px;
            color: #ccc;
        }}
        .add-panel-placeholder:hover span {{ color: #4a90d9; }}

        #elementLegend {{
            position: fixed;
            top: 80px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            font-size: 11px;
        }}
        #elementLegend .legend-title {{
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #ddd;
            color: #333;
        }}
        .element-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .element-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid;
        }}
        .element-name {{
            color: #444;
        }}
    </style>
</head>
<body>
    <div id="toolbar">
        <button id="addPanelBtn" onclick="addPanel()">+ Add Panel</button>
        <div class="separator"></div>
        <span class="label">Sync:</span>
        <button id="syncBtn" class="active" onclick="toggleSync()">ON</button>
        <div class="separator"></div>
        <span class="label">Speed:</span>
        <button onclick="setSpeed(120)">0.5x</button>
        <button onclick="setSpeed(60)" class="active" id="speed1x">1x</button>
        <button onclick="setSpeed(30)">2x</button>
        <button onclick="setSpeed(15)">4x</button>
        <div class="separator"></div>
        <button id="arrowBtn" onclick="toggleArrows()">{{'➡️ Arrows: ON' if self.show_arrows else '➡️ Arrows: OFF'}}</button>
        <button id="normBtn" onclick="toggleNormalize()">{{'📏 Normalize: ON' if self.normalize else '📏 Normalize: OFF'}}</button>
        <div class="separator"></div>
        <button id="camSyncBtn" class="active" onclick="toggleCameraSync()">📷 Camera Sync: ON</button>
        <div class="separator"></div>
        <button id="loopBtn" onclick="toggleLoop()" style="background:#ccffcc;">🔄 Loop All</button>
        <div class="separator"></div>
        <span class="label">GIF:</span>
        <select id="gifResolution">
            <option value="800x600">800x600</option>
            <option value="1200x900">1200x900</option>
            <option value="1600x1200" selected>1600x1200</option>
            <option value="1920x1440">1920x1440</option>
        </select>
        <button id="gifBtn" onclick="captureGif()" style="background:#fff3cd;">📷 Save GIF</button>
        <span id="gifStatus" style="font-size:11px;color:#666;"></span>
    </div>

    <div id="elementLegend">
        <div class="legend-title">Elements</div>
        <div id="elementList"></div>
    </div>

    <div id="panelContainer" class="grid-1">
        <!-- Panels will be added here dynamically -->
    </div>

    <div id="spectrumContainer">
        <div id="spectrumPlot"></div>
        <div id="spectrumControls">
            <div class="section-title">Spectrum</div>
            <button id="irBtn" onclick="setSpectrumView('ir')">IR Only</button>
            <button id="ramanBtn" onclick="setSpectrumView('raman')">Raman Only</button>
            <button id="bothBtn" class="active" onclick="setSpectrumView('both')">Both</button>
        </div>
    </div>

<script>
(function() {{
    // Data
    var modes = {mode_data_js};
    var framesData = {frames_data_js};
    var sceneLayout = {scene_layout_js};
    var elementLegend = {element_legend_js};
    var numFrames = {n_frames_js};
    var firstMode = {first_mode};

    // Populate element legend
    var elementList = document.getElementById('elementList');
    elementLegend.forEach(function(el) {{
        var item = document.createElement('div');
        item.className = 'element-item';
        item.innerHTML = '<div class="element-dot" style="background:' + el.color + '; border-color:' + el.outline + ';"></div>' +
                         '<span class="element-name">' + el.symbol + '</span>';
        elementList.appendChild(item);
    }});

    // State
    var panels = [];
    var panelCounter = 0;
    var syncEnabled = true;
    var cameraSyncEnabled = true;
    var activePanel = null;
    var looping = false;
    var animationInterval = null;
    var animationSpeed = 60;
    var showArrows = {show_arrows_js};
    var isNormalized = {is_normalized_js};
    var showIR = true;
    var showRaman = true;

    // Panel class
    function Panel(id, modeIdx) {{
        this.id = id;
        this.modeIdx = modeIdx;
        this.currentFrame = 0;
        this.divId = 'panel-viewer-' + id;
    }}

    function getFrameData(modeIdx, frameIdx) {{
        var type = isNormalized ? 'norm' : 'raw';
        return framesData[modeIdx][type][frameIdx];
    }}

    function createPanelElement(panel) {{
        var div = document.createElement('div');
        div.className = 'panel';
        div.id = 'panel-' + panel.id;

        var mode = modes.find(function(m) {{ return m.mode === panel.modeIdx; }});

        div.innerHTML = `
            <div class="panel-header">
                <select onchange="changePanelMode(${{panel.id}}, parseInt(this.value))">
                    ${{modes.map(function(m) {{
                        var selected = m.mode === panel.modeIdx ? 'selected' : '';
                        return '<option value="' + m.mode + '" ' + selected + '>Mode ' + m.mode + ': ' + m.freq.toFixed(1) + ' cm⁻¹</option>';
                    }}).join('')}}
                </select>
                <button class="close-btn" onclick="removePanel(${{panel.id}})">&times;</button>
            </div>
            <div class="panel-viewer" id="${{panel.divId}}"></div>
            <div class="panel-info" id="panel-info-${{panel.id}}">
                ${{mode ? (mode.ir_active ? 'IR: ' + mode.ir_intens.toFixed(1) + ' km/mol' : 'IR: Inactive') + ' | ' + (mode.raman_active ? 'Raman: Active' : 'Raman: Inactive') : ''}}
            </div>
        `;

        return div;
    }}

    function initPanelPlot(panel) {{
        var frameData = getFrameData(panel.modeIdx, 0);
        var layout = {{
            scene: sceneLayout,
            margin: {{l: 0, r: 0, t: 0, b: 0}},
            paper_bgcolor: '#fafafa',
            showlegend: false
        }};

        Plotly.newPlot(panel.divId, frameData, layout, {{
            responsive: true,
            plotGlPixelRatio: 2
        }}).then(function() {{
            updatePanelArrows(panel);

            var plotDiv = document.getElementById(panel.divId);

            // Click to set active panel
            plotDiv.addEventListener('click', function() {{
                setActivePanel(panel.id);
            }});

            // Camera sync on relayout
            plotDiv.on('plotly_relayout', function(eventData) {{
                if (eventData && eventData['scene.camera']) {{
                    syncCameraToOthers(panel.id, eventData['scene.camera']);
                }}
            }});
        }});

        // Set first panel as active
        if (panels.length === 1) {{
            setActivePanel(panel.id);
        }}
    }}

    function updatePanelFrame(panel, frameIdx) {{
        panel.currentFrame = frameIdx;
        var frameData = getFrameData(panel.modeIdx, frameIdx);
        var plotDiv = document.getElementById(panel.divId);
        if (plotDiv) {{
            Plotly.react(plotDiv, frameData, plotDiv.layout).then(function() {{
                updatePanelArrows(panel);
            }});
        }}
    }}

    function updatePanelArrows(panel) {{
        var plotDiv = document.getElementById(panel.divId);
        if (!plotDiv || !plotDiv.data) return;
        var indices = [];
        for (var i = 0; i < plotDiv.data.length; i++) {{
            var name = plotDiv.data[i].name;
            if (name === 'Arrows' || name === 'ArrowHeads') {{
                indices.push(i);
            }}
        }}
        if (indices.length > 0) {{
            Plotly.restyle(plotDiv, {{visible: showArrows}}, indices);
        }}
    }}

    function updatePanelInfo(panel) {{
        var mode = modes.find(function(m) {{ return m.mode === panel.modeIdx; }});
        var infoDiv = document.getElementById('panel-info-' + panel.id);
        if (mode && infoDiv) {{
            infoDiv.innerHTML = (mode.ir_active ? 'IR: ' + mode.ir_intens.toFixed(1) + ' km/mol' : 'IR: Inactive') +
                ' | ' + (mode.raman_active ? 'Raman: Active' : 'Raman: Inactive');
        }}
    }}

    window.addPanel = function(modeIdx) {{
        if (panels.length >= 6) {{
            alert('Maximum 6 panels allowed');
            return;
        }}

        modeIdx = modeIdx || firstMode;
        var panel = new Panel(panelCounter++, modeIdx);
        panels.push(panel);

        var container = document.getElementById('panelContainer');
        var panelEl = createPanelElement(panel);

        // Remove add placeholder if exists
        var placeholder = container.querySelector('.add-panel-placeholder');
        if (placeholder) placeholder.remove();

        container.appendChild(panelEl);

        // Add new placeholder
        if (panels.length < 6) {{
            var newPlaceholder = document.createElement('div');
            newPlaceholder.className = 'add-panel-placeholder';
            newPlaceholder.onclick = function() {{ addPanel(); }};
            newPlaceholder.innerHTML = '<span>+</span>';
            container.appendChild(newPlaceholder);
        }}

        updateGridLayout();
        initPanelPlot(panel);
        renderSpectrum();
    }};

    window.removePanel = function(panelId) {{
        if (panels.length <= 1) {{
            alert('At least one panel required');
            return;
        }}

        var idx = panels.findIndex(function(p) {{ return p.id === panelId; }});
        if (idx > -1) {{
            panels.splice(idx, 1);
            var panelEl = document.getElementById('panel-' + panelId);
            if (panelEl) panelEl.remove();
            updateGridLayout();
            renderSpectrum();
        }}
    }};

    window.changePanelMode = function(panelId, modeIdx) {{
        var panel = panels.find(function(p) {{ return p.id === panelId; }});
        if (panel) {{
            panel.modeIdx = modeIdx;
            panel.currentFrame = 0;
            updatePanelFrame(panel, 0);
            updatePanelInfo(panel);
            renderSpectrum();
        }}
    }};

    function updateGridLayout() {{
        var container = document.getElementById('panelContainer');
        container.className = 'grid-' + Math.min(panels.length + 1, 6);

        // Resize all plots
        setTimeout(function() {{
            panels.forEach(function(panel) {{
                var plotDiv = document.getElementById(panel.divId);
                if (plotDiv) Plotly.Plots.resize(plotDiv);
            }});
        }}, 100);
    }}

    window.setActivePanel = function(panelId) {{
        activePanel = panelId;
        // Update visual indicator
        document.querySelectorAll('.panel').forEach(function(el) {{
            el.classList.remove('active');
        }});
        var panelEl = document.getElementById('panel-' + panelId);
        if (panelEl) panelEl.classList.add('active');
    }};

    window.toggleCameraSync = function() {{
        cameraSyncEnabled = !cameraSyncEnabled;
        var btn = document.getElementById('camSyncBtn');
        btn.textContent = cameraSyncEnabled ? '📷 Camera Sync: ON' : '📷 Camera Sync: OFF';
        btn.className = cameraSyncEnabled ? 'active' : '';
    }};

    var cameraSyncTimeout = null;
    var isSyncing = false;

    function syncCameraToOthers(sourcePanelId, camera) {{
        if (!cameraSyncEnabled || isSyncing) return;

        // Debounce: wait for user to stop moving camera
        if (cameraSyncTimeout) clearTimeout(cameraSyncTimeout);
        cameraSyncTimeout = setTimeout(function() {{
            isSyncing = true;
            panels.forEach(function(panel) {{
                if (panel.id !== sourcePanelId) {{
                    var plotDiv = document.getElementById(panel.divId);
                    if (plotDiv) {{
                        Plotly.relayout(plotDiv, {{'scene.camera': camera}});
                    }}
                }}
            }});
            setTimeout(function() {{ isSyncing = false; }}, 100);
        }}, 150);  // Wait 150ms after last camera move
    }}

    window.toggleSync = function() {{
        syncEnabled = !syncEnabled;
        document.getElementById('syncBtn').className = syncEnabled ? 'active' : '';
        document.getElementById('syncBtn').textContent = syncEnabled ? 'ON' : 'OFF';
    }};

    window.setSpeed = function(speed) {{
        animationSpeed = speed;
        document.querySelectorAll('#toolbar button').forEach(function(btn) {{
            if (['0.5x', '1x', '2x', '4x'].includes(btn.textContent)) {{
                btn.className = '';
            }}
        }});
        var speedMap = {{120: '0.5x', 60: '1x', 30: '2x', 15: '4x'}};
        document.querySelectorAll('#toolbar button').forEach(function(btn) {{
            if (btn.textContent === speedMap[speed]) btn.className = 'active';
        }});
        if (looping) startLoop();
    }};

    window.toggleArrows = function() {{
        showArrows = !showArrows;
        var btn = document.getElementById('arrowBtn');
        btn.textContent = showArrows ? '➡️ Arrows: ON' : '➡️ Arrows: OFF';
        panels.forEach(function(panel) {{
            updatePanelArrows(panel);
        }});
    }};

    window.toggleNormalize = function() {{
        isNormalized = !isNormalized;
        var btn = document.getElementById('normBtn');
        btn.textContent = isNormalized ? '📏 Normalize: ON' : '📏 Normalize: OFF';
        panels.forEach(function(panel) {{
            updatePanelFrame(panel, panel.currentFrame);
        }});
    }};

    function startLoop() {{
        if (animationInterval) clearInterval(animationInterval);
        looping = true;
        animationInterval = setInterval(function() {{
            panels.forEach(function(panel) {{
                if (syncEnabled) {{
                    panel.currentFrame = (panels[0].currentFrame + 1) % numFrames;
                }} else {{
                    panel.currentFrame = (panel.currentFrame + 1) % numFrames;
                }}
            }});
            if (syncEnabled && panels.length > 0) {{
                panels[0].currentFrame = (panels[0].currentFrame + 1) % numFrames;
            }}
            panels.forEach(function(panel) {{
                updatePanelFrame(panel, panel.currentFrame);
            }});
        }}, animationSpeed);
        document.getElementById('loopBtn').textContent = '⏹ Stop All';
        document.getElementById('loopBtn').style.backgroundColor = '#ffcccc';
    }}

    function stopLoop() {{
        looping = false;
        if (animationInterval) {{
            clearInterval(animationInterval);
            animationInterval = null;
        }}
        document.getElementById('loopBtn').textContent = '🔄 Loop All';
        document.getElementById('loopBtn').style.backgroundColor = '#ccffcc';
    }}

    window.toggleLoop = function() {{
        if (looping) stopLoop();
        else startLoop();
    }};

    // Spectrum functions
    function renderSpectrum() {{
        var spectrumDiv = document.getElementById('spectrumPlot');
        if (!spectrumDiv) return;

        var traces = [];
        var irFreqs = [], irIntens = [], irModes = [];
        var ramanFreqs = [], ramanIntens = [], ramanModes = [];

        modes.forEach(function(m) {{
            if (m.ir_active && m.ir_intens > 0) {{
                irFreqs.push(m.freq);
                irIntens.push(m.ir_intens);
                irModes.push(m.mode);
            }}
            if (m.raman_active && m.raman_intens > 0) {{
                ramanFreqs.push(m.freq);
                ramanIntens.push(m.raman_intens);
                ramanModes.push(m.mode);
            }}
        }});

        var maxIR = Math.max.apply(null, irIntens) || 1;
        var maxRaman = Math.max.apply(null, ramanIntens) || 1;
        var ramanScale = maxIR / maxRaman;

        if (showIR && irFreqs.length > 0) {{
            var irSpectrum = createBroadenedSpectrum(irFreqs, irIntens, 8);
            traces.push({{
                x: irSpectrum.x, y: irSpectrum.y,
                type: 'scatter', mode: 'lines', name: 'IR',
                line: {{color: 'steelblue', width: 1.5}},
                fill: 'tozeroy', fillcolor: 'rgba(70, 130, 180, 0.15)'
            }});
            traces.push({{
                x: irFreqs, y: irIntens,
                type: 'scatter', mode: 'markers', name: 'IR peaks',
                marker: {{color: 'steelblue', size: 6}},
                customdata: irModes,
                hovertemplate: 'Mode %{{customdata}}<br>%{{x:.1f}} cm⁻¹<br>%{{y:.1f}} km/mol<extra></extra>'
            }});
        }}

        if (showRaman && ramanFreqs.length > 0) {{
            var scaledRaman = ramanIntens.map(function(v) {{ return v * ramanScale; }});
            var ramanSpectrum = createBroadenedSpectrum(ramanFreqs, scaledRaman, 8);
            traces.push({{
                x: ramanSpectrum.x, y: ramanSpectrum.y,
                type: 'scatter', mode: 'lines', name: 'Raman',
                line: {{color: 'darkorange', width: 1.5}},
                fill: 'tozeroy', fillcolor: 'rgba(255, 140, 0, 0.15)'
            }});
            traces.push({{
                x: ramanFreqs, y: scaledRaman,
                type: 'scatter', mode: 'markers', name: 'Raman peaks',
                marker: {{color: 'darkorange', size: 6, symbol: 'diamond'}},
                customdata: ramanModes,
                hovertemplate: 'Mode %{{customdata}}<br>%{{x:.1f}} cm⁻¹<extra></extra>'
            }});
        }}

        // Add markers for selected modes in panels
        var panelColors = ['#2a9d8f', '#e63946', '#9b59b6', '#f39c12', '#1abc9c', '#e74c3c'];
        panels.forEach(function(panel, idx) {{
            var mode = modes.find(function(m) {{ return m.mode === panel.modeIdx; }});
            if (mode) {{
                traces.push({{
                    x: [mode.freq, mode.freq],
                    y: [0, maxIR * 1.1],
                    type: 'scatter', mode: 'lines',
                    name: 'Panel ' + (idx + 1),
                    line: {{color: panelColors[idx % panelColors.length], width: 2, dash: 'dash'}}
                }});
            }}
        }});

        var maxFreq = Math.max.apply(null, modes.map(function(m) {{ return m.freq; }}));
        var layout = {{
            margin: {{l: 60, r: 120, t: 20, b: 40}},
            xaxis: {{
                title: 'Wavenumber (cm⁻¹)',
                range: [maxFreq + 100, 0],
                gridcolor: 'rgba(0,0,0,0.1)',
                dtick: 500
            }},
            yaxis: {{
                title: 'Intensity',
                gridcolor: 'rgba(0,0,0,0.1)'
            }},
            showlegend: true,
            legend: {{x: 1.02, xanchor: 'left', y: 1, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ddd', borderwidth: 1}},
            paper_bgcolor: '#fafafa',
            plot_bgcolor: 'white',
            hovermode: 'closest'
        }};

        Plotly.react(spectrumDiv, traces, layout).then(function() {{
            spectrumDiv.on('plotly_click', function(data) {{
                if (data.points && data.points[0] && data.points[0].customdata) {{
                    var clickedMode = data.points[0].customdata;
                    // Change active panel's mode, or add new panel if none active
                    if (activePanel !== null) {{
                        changePanelMode(activePanel, clickedMode);
                        var selectEl = document.querySelector('#panel-' + activePanel + ' select');
                        if (selectEl) selectEl.value = clickedMode;
                    }} else if (panels.length < 6) {{
                        addPanel(clickedMode);
                    }} else {{
                        changePanelMode(panels[0].id, clickedMode);
                        document.querySelector('#panel-' + panels[0].id + ' select').value = clickedMode;
                    }}
                }}
            }});
        }});
    }}

    function createBroadenedSpectrum(freqs, intens, gamma) {{
        var maxFreq = Math.max.apply(null, freqs) + 200;
        var nPoints = 500;
        var step = maxFreq / nPoints;
        var x = [], y = [];

        for (var i = 0; i <= nPoints; i++) {{
            var freq = i * step;
            x.push(freq);
            var intensity = 0;
            for (var j = 0; j < freqs.length; j++) {{
                var diff = freq - freqs[j];
                intensity += intens[j] / (1 + Math.pow(diff / gamma, 2));
            }}
            y.push(intensity);
        }}
        return {{x: x, y: y}};
    }}

    window.setSpectrumView = function(view) {{
        showIR = (view === 'ir' || view === 'both');
        showRaman = (view === 'raman' || view === 'both');
        document.getElementById('irBtn').className = showIR && !showRaman ? 'active' : '';
        document.getElementById('ramanBtn').className = !showIR && showRaman ? 'active' : '';
        document.getElementById('bothBtn').className = showIR && showRaman ? 'active' : '';
        renderSpectrum();
    }};

    // GIF capture - full view including panels and spectrum
    var gifWorkerBlob = null;

    function loadGifJs() {{
        return new Promise(function(resolve, reject) {{
            if (window.GIF && gifWorkerBlob) {{ resolve(); return; }}
            var script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js';
            script.onload = function() {{
                fetch('https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js')
                    .then(function(r) {{ return r.text(); }})
                    .then(function(code) {{
                        gifWorkerBlob = URL.createObjectURL(new Blob([code], {{type: 'application/javascript'}}));
                        resolve();
                    }})
                    .catch(function() {{
                        gifWorkerBlob = 'fallback';
                        resolve();
                    }});
            }};
            script.onerror = reject;
            document.head.appendChild(script);
        }});
    }}

    function sleep(ms) {{
        return new Promise(function(resolve) {{ setTimeout(resolve, ms); }});
    }}

    window.captureGif = async function() {{
        var btn = document.getElementById('gifBtn');
        var status = document.getElementById('gifStatus');
        btn.disabled = true;
        btn.textContent = '⏳ Capturing...';
        status.textContent = 'Loading...';

        await loadGifJs();

        var resolution = document.getElementById('gifResolution').value.split('x');
        var gifWidth = parseInt(resolution[0]);

        var capturedFrames = [];
        var gifDelay = Math.max(2, Math.round(animationSpeed / 10));

        // Calculate layout dimensions
        var nPanels = panels.length;
        var cols = Math.min(nPanels, 3);
        var rows = Math.ceil(nPanels / cols);
        var panelWidth = Math.floor(gifWidth / cols);
        var panelHeight = Math.floor(panelWidth * 0.8);
        var spectrumHeight = 200;
        var totalHeight = (rows * panelHeight) + spectrumHeight;

        for (var i = 0; i < numFrames; i++) {{
            status.textContent = 'Frame ' + (i + 1) + '/' + numFrames;

            // Update all panels to this frame
            for (var p = 0; p < panels.length; p++) {{
                updatePanelFrame(panels[p], i);
            }}

            // Give WebGL time to render (critical for 3D capture)
            await sleep(200);

            // Force a redraw of all panels
            for (var p = 0; p < panels.length; p++) {{
                var plotDiv = document.getElementById(panels[p].divId);
                if (plotDiv && plotDiv._fullLayout) {{
                    Plotly.Plots.resize(plotDiv);
                }}
            }}
            await sleep(100);

            try {{
                // Create composite canvas
                var canvas = document.createElement('canvas');
                canvas.width = gifWidth;
                canvas.height = totalHeight;
                var ctx = canvas.getContext('2d');
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, gifWidth, totalHeight);

                // Capture each panel using Plotly.toImage
                for (var p = 0; p < panels.length; p++) {{
                    var panel = panels[p];
                    var plotDiv = document.getElementById(panel.divId);
                    if (plotDiv) {{
                        var panelImg = await Plotly.toImage(plotDiv, {{
                            format: 'png',
                            width: panelWidth - 10,
                            height: panelHeight - 40
                        }});

                        var img = new Image();
                        await new Promise(function(resolve) {{
                            img.onload = resolve;
                            img.src = panelImg;
                        }});

                        var col = p % cols;
                        var row = Math.floor(p / cols);
                        var x = col * panelWidth + 5;
                        var y = row * panelHeight + 5;

                        // Draw panel background
                        ctx.fillStyle = 'white';
                        ctx.fillRect(x, y, panelWidth - 10, panelHeight - 10);

                        // Draw mode label
                        ctx.fillStyle = '#333';
                        ctx.font = '12px Arial';
                        var mode = modes.find(function(m) {{ return m.mode === panel.modeIdx; }});
                        ctx.fillText('Mode ' + panel.modeIdx + ': ' + (mode ? mode.freq.toFixed(1) : '?') + ' cm⁻¹', x + 5, y + 15);

                        // Draw 3D image
                        ctx.drawImage(img, x, y + 20, panelWidth - 10, panelHeight - 40);
                    }}
                }}

                // Capture spectrum using Plotly.toImage
                var spectrumDiv = document.getElementById('spectrumPlot');
                if (spectrumDiv) {{
                    var specImg = await Plotly.toImage(spectrumDiv, {{
                        format: 'png',
                        width: gifWidth - 20,
                        height: spectrumHeight - 10
                    }});

                    var sImg = new Image();
                    await new Promise(function(resolve) {{
                        sImg.onload = resolve;
                        sImg.src = specImg;
                    }});

                    ctx.fillStyle = 'white';
                    ctx.fillRect(5, rows * panelHeight + 5, gifWidth - 10, spectrumHeight - 10);
                    ctx.drawImage(sImg, 10, rows * panelHeight + 5, gifWidth - 20, spectrumHeight - 10);
                }}

                // Draw element legend on GIF
                var legendX = gifWidth - 90;
                var legendY = 10;
                var legendPadding = 8;
                var itemHeight = 18;
                var legendHeight = legendPadding * 2 + elementLegend.length * itemHeight + 20;

                ctx.fillStyle = 'rgba(255,255,255,0.95)';
                ctx.fillRect(legendX, legendY, 80, legendHeight);
                ctx.strokeStyle = '#ddd';
                ctx.strokeRect(legendX, legendY, 80, legendHeight);

                ctx.fillStyle = '#333';
                ctx.font = 'bold 11px Arial';
                ctx.fillText('Elements', legendX + legendPadding, legendY + 15);

                elementLegend.forEach(function(el, idx) {{
                    var y = legendY + 25 + idx * itemHeight;
                    ctx.beginPath();
                    ctx.arc(legendX + legendPadding + 7, y + 5, 6, 0, Math.PI * 2);
                    ctx.fillStyle = el.color;
                    ctx.fill();
                    ctx.strokeStyle = el.outline;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.fillStyle = '#444';
                    ctx.font = '11px Arial';
                    ctx.fillText(el.symbol, legendX + legendPadding + 20, y + 9);
                }});

                capturedFrames.push(canvas.toDataURL('image/png'));
            }} catch(e) {{
                console.error('Capture error:', e);
                status.textContent = 'Error: ' + e.message;
            }}
        }}

        status.textContent = 'Creating GIF...';

        if (gifWorkerBlob === 'fallback' || !window.GIF) {{
            status.textContent = 'GIF.js not available';
            resetGifButton();
            return;
        }}

        var gif = new GIF({{
            workers: 2,
            quality: 10,
            width: gifWidth,
            height: totalHeight,
            workerScript: gifWorkerBlob
        }});

        var loadedCount = 0;
        capturedFrames.forEach(function(dataUrl, idx) {{
            var img = new Image();
            img.onload = function() {{
                gif.addFrame(img, {{delay: gifDelay * 10}});
                loadedCount++;
                if (loadedCount === capturedFrames.length) {{
                    gif.render();
                }}
            }};
            img.src = dataUrl;
        }});

        gif.on('finished', function(blob) {{
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'multipanel_animation.gif';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            resetGifButton();
            status.textContent = 'Saved!';
            setTimeout(function() {{ status.textContent = ''; }}, 2000);
        }});

        gif.on('error', function() {{
            status.textContent = 'Error creating GIF';
            resetGifButton();
        }});
    }};

    function resetGifButton() {{
        var btn = document.getElementById('gifBtn');
        btn.disabled = false;
        btn.textContent = '📷 Save GIF';
    }}

    // Initialize
    window.addEventListener('load', function() {{
        addPanel(firstMode);
        renderSpectrum();
    }});
}})();
</script>
</body>
</html>'''

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Saved: {output_path}")
        return output_path


def create_comparison_html(animators, labels, output_path='comparison_viewer.html', n_frames=30, show_arrows=True, normalize=False):
    """Create a multi-file comparison HTML viewer."""
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    print(f"Generating comparison viewer for {len(animators)} files...")

    all_files_data = []

    for file_idx, (animator, label) in enumerate(zip(animators, labels)):
        print(f"Processing: {label}...")

        valid_modes = [m for m in animator.parser.modes if m['freq'] > 1.0]

        # Generate frame data for all modes
        frames_data = {}
        mode_info_list = []

        for i, mode in enumerate(valid_modes):
            mode_idx = mode['mode']
            displacement_raw = animator._validated_displacement(mode_idx)

            if displacement_raw is None:
                continue

            displacement_norm = animator._normalize_displacement(displacement_raw.copy())

            frames_raw = []
            frames_norm = []

            for frame_idx in range(n_frames):
                phase = np.cos(2 * np.pi * frame_idx / n_frames)

                current_coords = animator.coords + displacement_raw * animator.amplitude * phase
                frame_data = animator._create_molecule_traces(
                    current_coords,
                    displacement=displacement_raw * phase,
                    show_arrows=show_arrows,
                    arrow_scale=animator.arrow_scale
                )
                frames_raw.append([trace.to_plotly_json() for trace in frame_data])

                current_coords_norm = animator.coords + displacement_norm * animator.amplitude * phase
                frame_data_norm = animator._create_molecule_traces(
                    current_coords_norm,
                    displacement=displacement_norm * phase,
                    show_arrows=show_arrows,
                    arrow_scale=animator.arrow_scale
                )
                frames_norm.append([trace.to_plotly_json() for trace in frame_data_norm])

            frames_data[mode_idx] = {'raw': frames_raw, 'norm': frames_norm}

            mode_info_list.append({
                'mode': mode_idx,
                'freq': mode['freq'],
                'ir_active': mode['ir_active'],
                'ir_intens': mode['ir_intens'],
                'raman_active': mode['raman_active'],
                'raman_intens': mode.get('raman_intens', 0.0)
            })

        # Get scene layout
        max_disp = None
        for mode in mode_info_list[:5]:
            disp = animator._validated_displacement(mode['mode'])
            if disp is not None:
                disp_norm = animator._normalize_displacement(disp)
                if max_disp is None or np.max(np.abs(disp_norm)) > np.max(np.abs(max_disp)):
                    max_disp = disp_norm

        ranges = animator._get_axis_range(animator.coords, max_disp)
        scene_layout = animator._get_scene_layout(ranges)

        all_files_data.append({
            'label': label,
            'modes': mode_info_list,
            'frames': frames_data,
            'scene': scene_layout,
            'firstMode': mode_info_list[0]['mode'] if mode_info_list else 1
        })

    # Get unique elements from all files for legend
    all_elements = set()
    for animator in animators:
        all_elements.update(animator.elements)
    unique_elements = sorted(all_elements)
    element_legend = []
    for el in unique_elements:
        element_legend.append({
            'symbol': el,
            'color': ELEMENT_COLORS.get(el.upper(), ELEMENT_COLORS['DEFAULT']),
            'outline': ELEMENT_OUTLINE_COLORS.get(el.upper(), ELEMENT_OUTLINE_COLORS['DEFAULT'])
        })

    # Color palette for different files
    file_colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson', 'goldenrod']

    # Generate HTML
    files_data_js = json.dumps(all_files_data, cls=NumpyEncoder)
    element_legend_js = json.dumps(element_legend)
    file_colors_js = json.dumps(file_colors[:len(animators)])
    show_arrows_js = 'true' if show_arrows else 'false'
    is_normalized_js = 'true' if normalize else 'false'
    n_frames_js = str(n_frames)

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi-File Comparison Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; background: #f0f0f0; }}

        #toolbar {{
            background: white;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        #toolbar button {{
            padding: 8px 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 13px;
        }}
        #toolbar button:hover {{ background: #f5f5f5; }}
        #toolbar button.active {{ background: #4a90d9; color: white; border-color: #3a7bc8; }}
        #toolbar .separator {{ width: 1px; height: 24px; background: #ddd; }}
        #toolbar .label {{ font-size: 12px; color: #666; }}

        #fileContainer {{
            display: grid;
            gap: 10px;
            padding: 10px;
            min-height: calc(100vh - 350px);
        }}

        .file-panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .file-header {{
            padding: 10px 15px;
            background: #f8f8f8;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .file-label {{
            font-weight: bold;
            font-size: 14px;
            padding: 4px 10px;
            border-radius: 4px;
            color: white;
        }}
        .file-header select {{
            flex: 1;
            padding: 6px 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        .file-viewer {{
            flex: 1;
            min-height: 350px;
            cursor: pointer;
        }}
        .file-panel.active {{
            box-shadow: 0 0 0 3px #4a90d9, 0 2px 8px rgba(0,0,0,0.1);
        }}
        .file-panel.active .file-header {{
            background: #e8f4fc;
        }}
        .file-info {{
            padding: 8px 15px;
            background: #fafafa;
            border-top: 1px solid #eee;
            font-size: 11px;
            color: #666;
        }}

        #spectrumContainer {{
            background: white;
            border-top: 2px solid #ddd;
            display: flex;
            height: 280px;
        }}
        #spectrumPlot {{ flex: 1; }}
        #spectrumControls {{
            width: 140px;
            padding: 10px;
            background: #f8f8f8;
            border-left: 1px solid #ddd;
            font-size: 12px;
        }}
        #spectrumControls .section-title {{
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #ddd;
        }}
        #spectrumControls button {{
            width: 100%;
            padding: 6px 8px;
            margin: 3px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
        }}
        #spectrumControls button.active {{
            background: #4a90d9;
            color: white;
        }}
        #spectrumControls .file-toggle {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 11px;
        }}
        #spectrumControls .file-toggle input {{
            margin-right: 6px;
        }}
        #spectrumControls .color-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }}

        #elementLegend {{
            position: fixed;
            top: 80px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            font-size: 11px;
        }}
        #elementLegend .legend-title {{
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #ddd;
            color: #333;
        }}
        .element-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .element-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid;
        }}
        .element-name {{
            color: #444;
        }}
    </style>
</head>
<body>
    <div id="toolbar">
        <span class="label">Speed:</span>
        <button onclick="setSpeed(120)">0.5x</button>
        <button onclick="setSpeed(60)" class="active" id="speed1x">1x</button>
        <button onclick="setSpeed(30)">2x</button>
        <button onclick="setSpeed(15)">4x</button>
        <div class="separator"></div>
        <button id="arrowBtn" onclick="toggleArrows()">➡️ Arrows: {'ON' if show_arrows else 'OFF'}</button>
        <button id="normBtn" onclick="toggleNormalize()">📏 Normalize: {'ON' if normalize else 'OFF'}</button>
        <div class="separator"></div>
        <button id="camSyncBtn" class="active" onclick="toggleCameraSync()">📷 Camera Sync: ON</button>
        <button id="loopBtn" onclick="toggleLoop()" style="background:#ccffcc;">🔄 Loop All</button>
        <div class="separator"></div>
        <span class="label">GIF:</span>
        <select id="gifResolution">
            <option value="800x600">800x600</option>
            <option value="1200x900">1200x900</option>
            <option value="1600x1200" selected>1600x1200</option>
            <option value="1920x1440">1920x1440</option>
        </select>
        <button id="gifBtn" onclick="captureGif()" style="background:#fff3cd;">📷 Save GIF</button>
        <span id="gifStatus" style="font-size:11px;color:#666;"></span>
    </div>

    <div id="fileContainer">
        <!-- File panels will be added here -->
    </div>

    <div id="elementLegend">
        <div class="legend-title">Elements</div>
        <div id="elementList"></div>
    </div>

    <div id="spectrumContainer">
        <div id="spectrumPlot"></div>
        <div id="spectrumControls">
            <div class="section-title">Spectrum Type</div>
            <button id="irBtn" onclick="setSpectrumView('ir')">IR Only</button>
            <button id="ramanBtn" onclick="setSpectrumView('raman')">Raman Only</button>
            <button id="bothBtn" class="active" onclick="setSpectrumView('both')">Both</button>
            <div class="section-title" style="margin-top:12px;">Files</div>
            <div id="fileToggles"></div>
        </div>
    </div>

<script>
(function() {{
    var filesData = {files_data_js};
    var elementLegend = {element_legend_js};
    var numFrames = {n_frames_js};
    var fileColors = {file_colors_js};

    // Populate element legend
    var elementList = document.getElementById('elementList');
    elementLegend.forEach(function(el) {{
        var item = document.createElement('div');
        item.className = 'element-item';
        item.innerHTML = '<div class="element-dot" style="background:' + el.color + '; border-color:' + el.outline + ';"></div>' +
                         '<span class="element-name">' + el.symbol + '</span>';
        elementList.appendChild(item);
    }});

    var fileStates = [];
    var syncEnabled = true;
    var looping = false;
    var animationInterval = null;
    var animationSpeed = 60;
    var showArrows = {show_arrows_js};
    var isNormalized = {is_normalized_js};
    var showIR = true;
    var showRaman = true;
    var fileVisibility = filesData.map(function() {{ return true; }});
    var cameraSyncEnabled = true;
    var cameraSyncTimeout = null;
    var isSyncing = false;
    var activeFileIdx = 0;

    function initFileState(fileIdx) {{
        return {{
            fileIdx: fileIdx,
            modeIdx: filesData[fileIdx].firstMode,
            currentFrame: 0,
            divId: 'file-viewer-' + fileIdx
        }};
    }}

    function getFrameData(fileIdx, modeIdx, frameIdx) {{
        var type = isNormalized ? 'norm' : 'raw';
        var frames = filesData[fileIdx].frames;
        // Handle both numeric and string keys (JSON converts int keys to strings)
        var modeFrames = frames[modeIdx] || frames[String(modeIdx)];
        if (!modeFrames) {{
            console.error('No frame data for file', fileIdx, 'mode', modeIdx);
            return [];
        }}
        return modeFrames[type][frameIdx] || [];
    }}

    function createFilePanels() {{
        var container = document.getElementById('fileContainer');
        var nFiles = filesData.length;
        container.style.gridTemplateColumns = 'repeat(' + Math.min(nFiles, 3) + ', 1fr)';

        filesData.forEach(function(file, idx) {{
            var panel = document.createElement('div');
            panel.className = 'file-panel';
            panel.id = 'file-panel-' + idx;

            var mode = file.modes.find(function(m) {{ return m.mode === file.firstMode; }});

            panel.innerHTML = `
                <div class="file-header">
                    <span class="file-label" style="background:${{fileColors[idx]}}">${{file.label}}</span>
                    <select onchange="changeFileMode(${{idx}}, parseInt(this.value))">
                        ${{file.modes.map(function(m) {{
                            var selected = m.mode === file.firstMode ? 'selected' : '';
                            return '<option value="' + m.mode + '" ' + selected + '>Mode ' + m.mode + ': ' + m.freq.toFixed(1) + ' cm⁻¹</option>';
                        }}).join('')}}
                    </select>
                </div>
                <div class="file-viewer" id="file-viewer-${{idx}}"></div>
                <div class="file-info" id="file-info-${{idx}}">
                    ${{mode ? (mode.ir_active ? 'IR: ' + mode.ir_intens.toFixed(1) + ' km/mol' : 'IR Inactive') + ' | ' + (mode.raman_active ? 'Raman Active' : 'Raman Inactive') : ''}}
                </div>
            `;

            container.appendChild(panel);
            fileStates.push(initFileState(idx));
        }});
    }}

    function initFilePlots() {{
        // Initialize plots one at a time with delay to avoid overwhelming browser
        var plotIndex = 0;
        console.log('initFilePlots: starting with', fileStates.length, 'files');

        function initNextPlot() {{
            if (plotIndex >= fileStates.length) {{
                console.log('initFilePlots: all plots initialized');
                setActiveFile(0);
                return;
            }}

            var state = fileStates[plotIndex];
            console.log('initFilePlots: initializing plot', plotIndex, 'fileIdx:', state.fileIdx, 'modeIdx:', state.modeIdx);
            var frameData = getFrameData(state.fileIdx, state.modeIdx, 0);
            console.log('initFilePlots: frameData has', frameData ? frameData.length : 0, 'traces');

            if (!frameData || frameData.length === 0) {{
                console.error('initFilePlots: no frame data for plot', plotIndex);
                plotIndex++;
                setTimeout(initNextPlot, 100);
                return;
            }}

            var layout = {{
                scene: filesData[state.fileIdx].scene,
                margin: {{l: 0, r: 0, t: 0, b: 0}},
                paper_bgcolor: '#fafafa',
                showlegend: false
            }};

            Plotly.newPlot(state.divId, frameData, layout, {{
                responsive: true,
                plotGlPixelRatio: 2
            }}).then(function() {{
                updateFileArrows(state);

                var plotDiv = document.getElementById(state.divId);

                // Click to set active file
                plotDiv.addEventListener('click', function() {{
                    setActiveFile(state.fileIdx);
                }});

                // Camera sync on relayout
                plotDiv.on('plotly_relayout', function(eventData) {{
                    if (eventData && eventData['scene.camera']) {{
                        syncCameraToOthers(state.fileIdx, eventData['scene.camera']);
                    }}
                }});

                plotIndex++;
                setTimeout(initNextPlot, 100);
            }});
        }}

        initNextPlot();
    }}

    window.setActiveFile = function(fileIdx) {{
        activeFileIdx = fileIdx;
        document.querySelectorAll('.file-panel').forEach(function(el) {{
            el.classList.remove('active');
        }});
        var panelEl = document.getElementById('file-panel-' + fileIdx);
        if (panelEl) panelEl.classList.add('active');
    }}

    function syncCameraToOthers(sourceFileIdx, camera) {{
        if (!cameraSyncEnabled || isSyncing) return;
        if (cameraSyncTimeout) clearTimeout(cameraSyncTimeout);
        cameraSyncTimeout = setTimeout(function() {{
            isSyncing = true;
            fileStates.forEach(function(state) {{
                if (state.fileIdx !== sourceFileIdx) {{
                    var plotDiv = document.getElementById(state.divId);
                    if (plotDiv) {{
                        Plotly.relayout(plotDiv, {{'scene.camera': camera}});
                    }}
                }}
            }});
            setTimeout(function() {{ isSyncing = false; }}, 100);
        }}, 150);
    }}

    window.toggleCameraSync = function() {{
        cameraSyncEnabled = !cameraSyncEnabled;
        var btn = document.getElementById('camSyncBtn');
        btn.textContent = cameraSyncEnabled ? '📷 Camera Sync: ON' : '📷 Camera Sync: OFF';
        btn.className = cameraSyncEnabled ? 'active' : '';
    }}

    function updateFileFrame(state, frameIdx) {{
        state.currentFrame = frameIdx;
        var frameData = getFrameData(state.fileIdx, state.modeIdx, frameIdx);
        var plotDiv = document.getElementById(state.divId);
        if (plotDiv) {{
            Plotly.react(plotDiv, frameData, plotDiv.layout).then(function() {{
                updateFileArrows(state);
            }});
        }}
    }}

    function updateFileArrows(state) {{
        var plotDiv = document.getElementById(state.divId);
        if (!plotDiv || !plotDiv.data) return;
        var indices = [];
        for (var i = 0; i < plotDiv.data.length; i++) {{
            var name = plotDiv.data[i].name;
            if (name === 'Arrows' || name === 'ArrowHeads') {{
                indices.push(i);
            }}
        }}
        if (indices.length > 0) {{
            Plotly.restyle(plotDiv, {{visible: showArrows}}, indices);
        }}
    }}

    function updateFileInfo(state) {{
        var file = filesData[state.fileIdx];
        var mode = file.modes.find(function(m) {{ return m.mode === state.modeIdx; }});
        var infoDiv = document.getElementById('file-info-' + state.fileIdx);
        if (mode && infoDiv) {{
            infoDiv.innerHTML = (mode.ir_active ? 'IR: ' + mode.ir_intens.toFixed(1) + ' km/mol' : 'IR Inactive') +
                ' | ' + (mode.raman_active ? 'Raman Active' : 'Raman Inactive');
        }}
    }}

    window.changeFileMode = function(fileIdx, modeIdx) {{
        var state = fileStates[fileIdx];
        state.modeIdx = modeIdx;
        state.currentFrame = 0;
        updateFileFrame(state, 0);
        updateFileInfo(state);
        renderSpectrum();
    }};

    window.setSpeed = function(speed) {{
        animationSpeed = speed;
        document.querySelectorAll('#toolbar button').forEach(function(btn) {{
            if (['0.5x', '1x', '2x', '4x'].includes(btn.textContent)) {{
                btn.className = '';
            }}
        }});
        var speedMap = {{120: '0.5x', 60: '1x', 30: '2x', 15: '4x'}};
        document.querySelectorAll('#toolbar button').forEach(function(btn) {{
            if (btn.textContent === speedMap[speed]) btn.className = 'active';
        }});
        if (looping) startLoop();
    }};

    window.toggleArrows = function() {{
        showArrows = !showArrows;
        document.getElementById('arrowBtn').textContent = showArrows ? '➡️ Arrows: ON' : '➡️ Arrows: OFF';
        fileStates.forEach(updateFileArrows);
    }};

    window.toggleNormalize = function() {{
        isNormalized = !isNormalized;
        document.getElementById('normBtn').textContent = isNormalized ? '📏 Normalize: ON' : '📏 Normalize: OFF';
        fileStates.forEach(function(state) {{
            updateFileFrame(state, state.currentFrame);
        }});
    }};

    window.toggleSync = function() {{
        syncEnabled = !syncEnabled;
        var btn = document.getElementById('syncBtn');
        btn.textContent = syncEnabled ? '🔗 Sync: ON' : '🔗 Sync: OFF';
        btn.className = syncEnabled ? 'active' : '';
    }};

    function startLoop() {{
        if (animationInterval) clearInterval(animationInterval);
        looping = true;
        animationInterval = setInterval(function() {{
            var nextFrame = (fileStates[0].currentFrame + 1) % numFrames;
            fileStates.forEach(function(state) {{
                if (syncEnabled) {{
                    updateFileFrame(state, nextFrame);
                }} else {{
                    updateFileFrame(state, (state.currentFrame + 1) % numFrames);
                }}
            }});
        }}, animationSpeed);
        document.getElementById('loopBtn').textContent = '⏹ Stop All';
        document.getElementById('loopBtn').style.backgroundColor = '#ffcccc';
    }}

    function stopLoop() {{
        looping = false;
        if (animationInterval) {{
            clearInterval(animationInterval);
            animationInterval = null;
        }}
        document.getElementById('loopBtn').textContent = '🔄 Loop All';
        document.getElementById('loopBtn').style.backgroundColor = '#ccffcc';
    }}

    window.toggleLoop = function() {{
        if (looping) stopLoop();
        else startLoop();
    }};

    // Spectrum
    function createFileToggles() {{
        var container = document.getElementById('fileToggles');
        filesData.forEach(function(file, idx) {{
            var div = document.createElement('div');
            div.className = 'file-toggle';
            div.innerHTML = `
                <input type="checkbox" checked onchange="toggleFileVisibility(${{idx}}, this.checked)">
                <span class="color-dot" style="background:${{fileColors[idx]}}"></span>
                <span>${{file.label}}</span>
            `;
            container.appendChild(div);
        }});
    }}

    window.toggleFileVisibility = function(idx, visible) {{
        fileVisibility[idx] = visible;
        renderSpectrum();
    }};

    function renderSpectrum() {{
        var spectrumDiv = document.getElementById('spectrumPlot');
        if (!spectrumDiv) return;

        var traces = [];
        var maxIntensity = 0;

        filesData.forEach(function(file, fileIdx) {{
            if (!fileVisibility[fileIdx]) return;

            var irFreqs = [], irIntens = [], ramanFreqs = [], ramanIntens = [];

            file.modes.forEach(function(m) {{
                if (m.ir_active && m.ir_intens > 0) {{
                    irFreqs.push(m.freq);
                    irIntens.push(m.ir_intens);
                    maxIntensity = Math.max(maxIntensity, m.ir_intens);
                }}
                if (m.raman_active && m.raman_intens > 0) {{
                    ramanFreqs.push(m.freq);
                    ramanIntens.push(m.raman_intens);
                }}
            }});

            var color = fileColors[fileIdx];

            if (showIR && irFreqs.length > 0) {{
                var irSpectrum = createBroadenedSpectrum(irFreqs, irIntens, 8);
                traces.push({{
                    x: irSpectrum.x, y: irSpectrum.y,
                    type: 'scatter', mode: 'lines',
                    name: file.label + ' (IR)',
                    line: {{color: color, width: 1.5}},
                    legendgroup: file.label
                }});
            }}

            if (showRaman && ramanFreqs.length > 0) {{
                var maxIR = Math.max.apply(null, irIntens) || 1;
                var maxRaman = Math.max.apply(null, ramanIntens) || 1;
                var scaled = ramanIntens.map(function(v) {{ return v * maxIR / maxRaman; }});
                var ramanSpectrum = createBroadenedSpectrum(ramanFreqs, scaled, 8);
                traces.push({{
                    x: ramanSpectrum.x, y: ramanSpectrum.y,
                    type: 'scatter', mode: 'lines',
                    name: file.label + ' (Raman)',
                    line: {{color: color, width: 1.5, dash: 'dash'}},
                    legendgroup: file.label
                }});
            }}
        }});

        // Add markers for selected modes
        fileStates.forEach(function(state) {{
            if (!fileVisibility[state.fileIdx]) return;
            var file = filesData[state.fileIdx];
            var mode = file.modes.find(function(m) {{ return m.mode === state.modeIdx; }});
            if (mode) {{
                traces.push({{
                    x: [mode.freq, mode.freq],
                    y: [0, maxIntensity * 1.1 || 100],
                    type: 'scatter', mode: 'lines',
                    name: file.label + ' selected',
                    line: {{color: fileColors[state.fileIdx], width: 2, dash: 'dot'}},
                    showlegend: false
                }});
            }}
        }});

        var allFreqs = [];
        filesData.forEach(function(file) {{
            file.modes.forEach(function(m) {{ allFreqs.push(m.freq); }});
        }});
        var maxFreq = Math.max.apply(null, allFreqs);

        var layout = {{
            margin: {{l: 60, r: 120, t: 20, b: 40}},
            xaxis: {{
                title: 'Wavenumber (cm⁻¹)',
                range: [maxFreq + 100, 0],
                gridcolor: 'rgba(0,0,0,0.1)',
                dtick: 500
            }},
            yaxis: {{
                title: 'Intensity',
                gridcolor: 'rgba(0,0,0,0.1)'
            }},
            showlegend: true,
            legend: {{x: 1.02, xanchor: 'left', y: 1, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ddd', borderwidth: 1}},
            paper_bgcolor: '#fafafa',
            plot_bgcolor: 'white',
            hovermode: 'closest'
        }};

        Plotly.react(spectrumDiv, traces, layout).then(function() {{
            // Attach click handler for mode selection
            spectrumDiv.removeAllListeners('plotly_click');
            spectrumDiv.on('plotly_click', function(data) {{
                if (!data.points || !data.points[0]) return;

                var clickedX = data.points[0].x;

                // Find closest mode from active file
                var activeFile = filesData[activeFileIdx];
                if (!activeFile || !activeFile.modes) return;

                var closestMode = null;
                var minDist = Infinity;
                activeFile.modes.forEach(function(m) {{
                    var dist = Math.abs(m.freq - clickedX);
                    if (dist < minDist) {{
                        minDist = dist;
                        closestMode = m.mode;
                    }}
                }});

                if (closestMode !== null && minDist < 50) {{
                    changeFileMode(activeFileIdx, closestMode);
                    // Update dropdown
                    var selectEl = document.querySelector('#file-panel-' + activeFileIdx + ' select');
                    if (selectEl) selectEl.value = closestMode;
                }}
            }});
        }});
    }}

    function createBroadenedSpectrum(freqs, intens, gamma) {{
        var maxFreq = Math.max.apply(null, freqs) + 200;
        var nPoints = 500;
        var step = maxFreq / nPoints;
        var x = [], y = [];

        for (var i = 0; i <= nPoints; i++) {{
            var freq = i * step;
            x.push(freq);
            var intensity = 0;
            for (var j = 0; j < freqs.length; j++) {{
                var diff = freq - freqs[j];
                intensity += intens[j] / (1 + Math.pow(diff / gamma, 2));
            }}
            y.push(intensity);
        }}
        return {{x: x, y: y}};
    }}

    window.setSpectrumView = function(view) {{
        showIR = (view === 'ir' || view === 'both');
        showRaman = (view === 'raman' || view === 'both');
        document.getElementById('irBtn').className = showIR && !showRaman ? 'active' : '';
        document.getElementById('ramanBtn').className = !showIR && showRaman ? 'active' : '';
        document.getElementById('bothBtn').className = showIR && showRaman ? 'active' : '';
        renderSpectrum();
    }};

    // GIF capture
    var gifWorkerBlob = null;

    function loadGifJs() {{
        return new Promise(function(resolve, reject) {{
            if (window.GIF && gifWorkerBlob) {{ resolve(); return; }}
            var script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js';
            script.onload = function() {{
                fetch('https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js')
                    .then(function(r) {{ return r.text(); }})
                    .then(function(code) {{
                        gifWorkerBlob = URL.createObjectURL(new Blob([code], {{type: 'application/javascript'}}));
                        resolve();
                    }})
                    .catch(function() {{
                        gifWorkerBlob = 'fallback';
                        resolve();
                    }});
            }};
            script.onerror = reject;
            document.head.appendChild(script);
        }});
    }}

    function sleep(ms) {{
        return new Promise(function(resolve) {{ setTimeout(resolve, ms); }});
    }}

    window.captureGif = async function() {{
        var btn = document.getElementById('gifBtn');
        var status = document.getElementById('gifStatus');
        btn.disabled = true;
        btn.textContent = '⏳ Capturing...';
        status.textContent = 'Loading...';

        await loadGifJs();

        var resolution = document.getElementById('gifResolution').value.split('x');
        var gifWidth = parseInt(resolution[0]);

        var capturedFrames = [];
        var gifDelay = Math.max(2, Math.round(animationSpeed / 10));

        // Calculate layout
        var nFiles = filesData.length;
        var cols = Math.min(nFiles, 3);
        var rows = Math.ceil(nFiles / cols);
        var panelWidth = Math.floor(gifWidth / cols);
        var panelHeight = Math.floor(panelWidth * 0.8);
        var spectrumHeight = 200;
        var totalHeight = (rows * panelHeight) + spectrumHeight;

        for (var i = 0; i < numFrames; i++) {{
            status.textContent = 'Frame ' + (i + 1) + '/' + numFrames;

            for (var f = 0; f < fileStates.length; f++) {{
                updateFileFrame(fileStates[f], i);
            }}

            // Give WebGL time to render (critical for 3D capture)
            await sleep(200);

            // Force a redraw of all file panels
            for (var f = 0; f < fileStates.length; f++) {{
                var plotDiv = document.getElementById(fileStates[f].divId);
                if (plotDiv && plotDiv._fullLayout) {{
                    Plotly.Plots.resize(plotDiv);
                }}
            }}
            await sleep(100);

            try {{
                var canvas = document.createElement('canvas');
                canvas.width = gifWidth;
                canvas.height = totalHeight;
                var ctx = canvas.getContext('2d');
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, gifWidth, totalHeight);

                // Capture each file panel
                for (var f = 0; f < fileStates.length; f++) {{
                    var state = fileStates[f];
                    var file = filesData[state.fileIdx];
                    var plotDiv = document.getElementById(state.divId);
                    if (plotDiv) {{
                        var panelImg = await Plotly.toImage(plotDiv, {{
                            format: 'png',
                            width: panelWidth - 10,
                            height: panelHeight - 50
                        }});

                        var img = new Image();
                        await new Promise(function(resolve) {{
                            img.onload = resolve;
                            img.src = panelImg;
                        }});

                        var col = f % cols;
                        var row = Math.floor(f / cols);
                        var x = col * panelWidth + 5;
                        var y = row * panelHeight + 5;

                        ctx.fillStyle = 'white';
                        ctx.fillRect(x, y, panelWidth - 10, panelHeight - 10);

                        // Draw file label
                        ctx.fillStyle = fileColors[f];
                        ctx.fillRect(x + 5, y + 5, 60, 18);
                        ctx.fillStyle = 'white';
                        ctx.font = 'bold 11px Arial';
                        ctx.fillText(file.label, x + 8, y + 17);

                        // Draw mode info
                        ctx.fillStyle = '#333';
                        ctx.font = '11px Arial';
                        var mode = file.modes.find(function(m) {{ return m.mode === state.modeIdx; }});
                        ctx.fillText('Mode ' + state.modeIdx + ': ' + (mode ? mode.freq.toFixed(1) : '?') + ' cm⁻¹', x + 70, y + 17);

                        ctx.drawImage(img, x, y + 25, panelWidth - 10, panelHeight - 50);
                    }}
                }}

                // Capture spectrum
                var spectrumDiv = document.getElementById('spectrumPlot');
                if (spectrumDiv) {{
                    var specImg = await Plotly.toImage(spectrumDiv, {{
                        format: 'png',
                        width: gifWidth - 20,
                        height: spectrumHeight - 10
                    }});

                    var sImg = new Image();
                    await new Promise(function(resolve) {{
                        sImg.onload = resolve;
                        sImg.src = specImg;
                    }});

                    ctx.fillStyle = 'white';
                    ctx.fillRect(5, rows * panelHeight + 5, gifWidth - 10, spectrumHeight - 10);
                    ctx.drawImage(sImg, 10, rows * panelHeight + 5, gifWidth - 20, spectrumHeight - 10);
                }}

                capturedFrames.push(canvas.toDataURL('image/png'));
            }} catch(e) {{
                console.error('Capture error:', e);
            }}
        }}

        status.textContent = 'Creating GIF...';

        if (gifWorkerBlob === 'fallback' || !window.GIF) {{
            status.textContent = 'GIF.js not available';
            resetGifButton();
            return;
        }}

        var gif = new GIF({{
            workers: 2,
            quality: 10,
            width: gifWidth,
            height: totalHeight,
            workerScript: gifWorkerBlob
        }});

        var loadedCount = 0;
        capturedFrames.forEach(function(dataUrl, idx) {{
            var img = new Image();
            img.onload = function() {{
                gif.addFrame(img, {{delay: gifDelay * 10}});
                loadedCount++;
                if (loadedCount === capturedFrames.length) {{
                    gif.render();
                }}
            }};
            img.src = dataUrl;
        }});

        gif.on('finished', function(blob) {{
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'comparison_animation.gif';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            resetGifButton();
            status.textContent = 'Saved!';
            setTimeout(function() {{ status.textContent = ''; }}, 2000);
        }});

        gif.on('error', function() {{
            status.textContent = 'Error creating GIF';
            resetGifButton();
        }});
    }}

    function resetGifButton() {{
        var btn = document.getElementById('gifBtn');
        btn.disabled = false;
        btn.textContent = '📷 Save GIF';
    }}

    // Initialize
    window.addEventListener('load', function() {{
        createFilePanels();
        initFilePlots();
        createFileToggles();
        renderSpectrum();

        setTimeout(function() {{
            fileStates.forEach(function(state) {{
                Plotly.Plots.resize(document.getElementById(state.divId));
            }});
        }}, 200);
    }});
}})();
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Vibrational Mode Viewer for CRYSTAL23')
    parser.add_argument('filename', nargs='+', help='CRYSTAL23 frequency output file(s) (.out)')
    parser.add_argument('--mode', '-m', type=int, help='Mode number to visualize')
    parser.add_argument('--list', '-l', action='store_true', help='List all vibrational modes')
    parser.add_argument('--save', '-s', action='store_true', help='Save as HTML file')
    parser.add_argument('--all', '-A', action='store_true', help='Generate single HTML with all modes')
    parser.add_argument('--multi', '-M', action='store_true', help='Generate multi-panel comparison viewer')
    parser.add_argument('--compare', '-C', action='store_true', help='Compare multiple files side-by-side')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison files (default: filenames)')
    parser.add_argument('--gif', '-g', action='store_true', help='Export as GIF')
    parser.add_argument('--gif-fps', type=int, default=20, help='GIF frames per second (default: 20)')
    parser.add_argument('--static', action='store_true', help='Static view with arrows')
    parser.add_argument('--output-dir', '-o', default='vibmode_html', help='Output directory')
    parser.add_argument('--amplitude', '-a', type=float, default=1.0, help='Displacement amplitude (default: 1.0)')
    parser.add_argument('--speed', type=float, default=1.0, help='Animation speed (default: 1.0)')
    parser.add_argument('--frames', '-f', type=int, default=30, help='Animation frames (default: 30)')
    parser.add_argument('--normalize', '-n', action='store_true', help='Normalize displacements for visibility')
    parser.add_argument('--arrows', action='store_true', default=True, help='Show displacement arrows (default: True)')
    parser.add_argument('--no-arrows', dest='arrows', action='store_false', help='Hide displacement arrows')
    parser.add_argument('--arrow-scale', type=float, default=15.0, help='Arrow length scale (default: 15.0)')

    args = parser.parse_args()

    # Handle comparison mode (multiple files)
    if args.compare:
        if len(args.filename) < 2:
            print("Error: --compare requires at least 2 files")
            sys.exit(1)

        # Check all files exist
        for f in args.filename:
            if not os.path.exists(f):
                print(f"Error: File not found: {f}")
                sys.exit(1)

        # Create animators for each file
        animators = []
        for f in args.filename:
            print(f"Parsing {f}...")
            freq_parser = Crystal23FreqParser(f)
            print(f"  Found {freq_parser.n_atoms} atoms and {len(freq_parser.modes)} modes")
            animator = VibModeAnimator(freq_parser)
            animator.amplitude = args.amplitude
            animator.n_frames = args.frames
            animator.normalize = args.normalize
            animator.show_arrows = args.arrows
            animator.arrow_scale = args.arrow_scale
            animators.append(animator)

        # Generate labels
        if args.labels:
            labels = args.labels
            if len(labels) < len(animators):
                labels.extend([os.path.basename(f).split('.')[0] for f in args.filename[len(labels):]])
        else:
            labels = [os.path.basename(f).split('.')[0] for f in args.filename]
            # Shorten labels if too long
            labels = [l[:20] if len(l) > 20 else l for l in labels]

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'comparison_viewer.html')
        create_comparison_html(animators, labels, output_path,
                               n_frames=args.frames, show_arrows=args.arrows, normalize=args.normalize)
        sys.exit(0)

    # Single file mode
    filename = args.filename[0]

    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    print(f"Parsing {filename}...")
    freq_parser = Crystal23FreqParser(filename)
    print(f"Found {freq_parser.n_atoms} atoms and {len(freq_parser.modes)} modes")

    animator = VibModeAnimator(freq_parser)
    print(f"Detected {len(animator.bonds)} bonds")

    if args.list:
        freq_parser.list_modes()
        sys.exit(0)

    animator.amplitude = args.amplitude
    animator.n_frames = args.frames
    animator.normalize = args.normalize
    animator.show_arrows = args.arrows
    animator.arrow_scale = args.arrow_scale

    if args.normalize:
        print("Normalization enabled: displacements scaled for visibility")
    if args.arrows:
        print("Displacement arrows enabled")

    if args.all:
        # Generate single HTML with all modes
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'all_modes_viewer.html')
        animator.create_all_modes_html(output_path)
        sys.exit(0)

    if args.multi:
        # Generate multi-panel comparison viewer
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'multipanel_viewer.html')
        animator.create_multipanel_html(output_path)
        sys.exit(0)

    if args.mode:
        animator.show_mode(args.mode, animate=not args.static, speed=args.speed,
                          save_html=args.save, save_gif=args.gif, output_dir=args.output_dir,
                          gif_fps=args.gif_fps)
    else:
        freq_parser.list_modes()
        while True:
            try:
                mode_input = input("Enter mode number (or 'q' to quit): ").strip()
                if mode_input.lower() == 'q':
                    break
                mode_idx = int(mode_input)
                animator.show_mode(mode_idx, animate=not args.static, speed=args.speed,
                                  save_html=args.save, save_gif=args.gif, output_dir=args.output_dir,
                                  gif_fps=args.gif_fps)
            except ValueError:
                print("Invalid input. Enter a mode number.")
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    main()
