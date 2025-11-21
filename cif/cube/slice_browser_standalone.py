#!/usr/bin/env python3
"""
Standalone slice browser using the EXACT working code from the successful test.
This bypasses the main visualization pipeline entirely.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path to import CubeFile
sys.path.insert(0, str(Path(__file__).parent))
from crystal_cubeviz_plotly import CubeFile

def create_slice_browser(cube_file_path, axis='z', output_file='slice_browser.html', max_frames=100):
    """Create slice browser using the exact working test code."""

    print(f"\nReading: {cube_file_path}")
    cube = CubeFile(cube_file_path)

    bohr_to_ang = 0.529177

    # Determine slicing parameters
    if axis == 'z':
        n_slices = cube.data.shape[2]
        slice_dim = 2
        x_label, y_label = 'X (Å)', 'Y (Å)'
    elif axis == 'y':
        n_slices = cube.data.shape[1]
        slice_dim = 1
        x_label, y_label = 'X (Å)', 'Z (Å)'
    else:  # x
        n_slices = cube.data.shape[0]
        slice_dim = 0
        x_label, y_label = 'Y (Å)', 'Z (Å)'

    print(f"Total slices: {n_slices}")

    # Frame subsampling
    if n_slices > max_frames:
        frame_indices = list(np.linspace(0, n_slices - 1, max_frames, dtype=int))
        frame_indices = sorted(set(frame_indices))
        print(f"Subsampling to {len(frame_indices)} frames")
    else:
        frame_indices = list(range(n_slices))

    # Best slice (middle of material region)
    if hasattr(cube, 'data_bounds') and cube.vacuum_detected:
        axis_key = ['x', 'y', 'z'][slice_dim]
        start, end = cube.data_bounds[axis_key]
        best_slice = (start + end) // 2
    else:
        best_slice = n_slices // 2

    best_frame_idx = frame_indices.index(best_slice) if best_slice in frame_indices else len(frame_indices) // 2

    print(f"Creating frames...")

    # NOTE: For non-orthogonal lattices (hexagonal), coordinates depend on slice index!
    # We'll calculate x_coords_1d and y_coords_1d inside the frame loop for each slice

    # CRITICAL FIX FOR BUG 2: Choose colorscale based on data type
    # For spin/potential (signed data), use diverging colorscale (RdBu)
    # For density (positive-only data), use sequential colorscale (Viridis)
    if cube.data_type in ['spin', 'potential']:
        colorscale = 'RdBu'  # Blue for negative, white for zero, red for positive
        colorscale_reversed = True  # RdBu_r: blue for negative
    else:
        colorscale = 'Viridis'  # Standard sequential for density
        colorscale_reversed = False

    # Collect all slice data for global vmin/vmax
    all_data = []
    for idx in frame_indices:
        if axis == 'z':
            slice_data = cube.data[:, :, idx]
        elif axis == 'y':
            slice_data = cube.data[:, idx, :]
        else:
            slice_data = cube.data[idx, :, :]

        # Apply log scale for density (NOT for spin/potential!)
        if cube.data_type == 'density':
            display_data = np.log10(np.abs(slice_data) + 1e-10)
        else:
            display_data = slice_data  # Keep signed values for spin/potential

        all_data.append(display_data)

    # Global color range
    all_data_flat = np.concatenate([d.flatten() for d in all_data])

    # CRITICAL FIX: For spin/potential, 0 must ALWAYS be neutral (white/transparent)
    if cube.data_type in ['spin', 'potential']:
        data_min = np.min(all_data_flat)
        data_max = np.max(all_data_flat)

        # Use percentiles to clip extreme outliers, but ALWAYS include 0 in range
        # This ensures 0 appears as white (neutral) in the colorscale
        if cube.data_type == 'potential':
            # For potential, clip extremes and FORCE symmetry around 0 (just like SPIN)
            # This ensures 0 is EXACTLY at center -> appears WHITE/NEUTRAL
            abs_min = abs(data_min)
            abs_max = abs(data_max)
            asymmetry_ratio = abs_max / abs_min if abs_min > 1e-10 else float('inf')

            if asymmetry_ratio > 100:  # Highly asymmetric
                print(f"  Highly asymmetric potential detected (ratio: {asymmetry_ratio:.1f})")
                print(f"  Using MINIMUM range to make BOTH pos/neg features visible...")
                # Clip each side independently
                neg_data = all_data_flat[all_data_flat < 0]
                pos_data = all_data_flat[all_data_flat > 0]
                vmin_clip = np.percentile(neg_data, 1) if len(neg_data) > 0 else 0
                vmax_clip = np.percentile(pos_data, 99) if len(pos_data) > 0 else 0

                # CRITICAL FIX: Use MIN (not MAX) to ensure both features visible
                # This gives 50% colorscale to negative, 50% to positive
                vmax_sym = min(abs(vmin_clip), abs(vmax_clip))
                vmin = -vmax_sym
                vmax = vmax_sym
                print(f"  Symmetric range: [{vmin:.3f}, {vmax:.3f}] (0 = WHITE, both sides visible)")
            else:
                vmin, vmax = data_min, data_max
        else:
            # For spin, clip to percentiles but force symmetry around 0
            # This ensures 0 is white and both pos/neg features are visible
            print(f"  Clipping spin data to percentiles for better visibility...")
            print(f"  Original range: [{data_min:.3e}, {data_max:.3e}]")

            # Clip each side independently to remove extreme noise
            neg_data = all_data_flat[all_data_flat < 0]
            pos_data = all_data_flat[all_data_flat > 0]

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
            print(f"  Clipped symmetric range: [{vmin:.3e}, {vmax:.3e}] (0 = neutral)")
    else:
        # For density, percentiles help avoid extreme outliers
        vmin = np.percentile(all_data_flat, 5)
        vmax = np.percentile(all_data_flat, 95)

    print(f"Global color range: [{vmin:.3e}, {vmax:.3e}]")

    # Create frames (EXACT same as working test)
    frames = []
    for idx in frame_indices:
        display_data = all_data[frame_indices.index(idx)]

        # CRITICAL FIX: Calculate coordinates for each slice (includes offset from fixed index)
        # For non-orthogonal lattices: r = origin + i*v0 + j*v1 + k*v2
        if axis == 'z':
            # Z-slice (XY plane, fixed k=idx): X = i*v0[0] + j*v1[0], Y = i*v0[1] + j*v1[1]
            # For diagonal-dominant lattices, can use simple 1D arrays
            x_coords_1d = np.arange(cube.data.shape[0]) * cube.voxel_vectors[0, 0] * bohr_to_ang
            y_coords_1d = np.arange(cube.data.shape[1]) * cube.voxel_vectors[1, 1] * bohr_to_ang
        elif axis == 'y':
            # Y-slice (XZ plane, fixed j=idx): X = i*v0[0] + idx*v1[0] + k*v2[0], Z = k*v2[2]
            # MISSING OFFSET: idx*v1[0]!
            x_offset = idx * cube.voxel_vectors[1, 0] * bohr_to_ang
            x_coords_1d = np.arange(cube.data.shape[0]) * cube.voxel_vectors[0, 0] * bohr_to_ang + x_offset
            y_coords_1d = np.arange(cube.data.shape[2]) * cube.voxel_vectors[2, 2] * bohr_to_ang
        else:  # axis == 'x'
            # X-slice (YZ plane, fixed i=idx): Y = idx*v0[1] + j*v1[1] + k*v2[1], Z = k*v2[2]
            # MISSING OFFSET: idx*v0[1]!
            x_offset = idx * cube.voxel_vectors[0, 1] * bohr_to_ang
            x_coords_1d = np.arange(cube.data.shape[1]) * cube.voxel_vectors[1, 1] * bohr_to_ang + x_offset
            y_coords_1d = np.arange(cube.data.shape[2]) * cube.voxel_vectors[2, 2] * bohr_to_ang

        # EXACT trace structure from working test
        # Use appropriate colorscale for data type
        heatmap_cmap = colorscale + '_r' if colorscale_reversed else colorscale
        contour_cmap = colorscale + '_r' if colorscale_reversed else colorscale

        # Smart contour spacing: more contours near meaningful values
        # For spin/potential, use finer spacing; for density, coarser spacing
        if cube.data_type in ['spin', 'potential']:
            # Use 20-25 contours for better detail
            contour_size = (vmax - vmin) / 25
        else:
            # For density, 10 contours is fine
            contour_size = (vmax - vmin) / 10

        trace_list = [
            go.Heatmap(
                x=x_coords_1d,
                y=y_coords_1d,
                z=display_data.T,
                colorscale=heatmap_cmap,
                zmid=0 if cube.data_type in ['spin', 'potential'] else None,  # Center at zero for signed data
                showscale=False,
                opacity=0.3,
                name='Heatmap'
            ),
            go.Contour(
                x=x_coords_1d,
                y=y_coords_1d,
                z=display_data.T,
                colorscale=contour_cmap,
                zmid=0 if cube.data_type in ['spin', 'potential'] else None,  # Center at zero for signed data
                zmin=vmin,
                zmax=vmax,
                contours=dict(
                    coloring='lines',
                    showlabels=True,
                    labelfont=dict(size=12, color='white'),
                    size=contour_size
                ),
                line=dict(width=2),
                showscale=True,
                colorbar=dict(
                    title=f'{"log₁₀ " if cube.data_type == "density" else ""}{cube.data_type.capitalize()}',
                    len=0.7,
                    tickformat='.2e'  # Force scientific notation (fixes "p" issue - BUG 9)
                ),
                name='Contours'
            )
        ]

        # Add atoms (simplified - just show all atoms)
        atom_x, atom_y, atom_hover = [], [], []
        # Simple atom properties lookup
        atom_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al',
                     14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 26: 'Fe',
                     29: 'Cu', 30: 'Zn', 34: 'Se', 50: 'Sn'}

        for pos, num in zip(cube.atomic_positions, cube.atomic_numbers):
            if axis == 'z':
                atom_x.append(pos[0] * bohr_to_ang)
                atom_y.append(pos[1] * bohr_to_ang)
            elif axis == 'y':
                atom_x.append(pos[0] * bohr_to_ang)
                atom_y.append(pos[2] * bohr_to_ang)
            else:
                atom_x.append(pos[1] * bohr_to_ang)
                atom_y.append(pos[2] * bohr_to_ang)

            # Add hover label with element name
            elem_name = atom_names.get(num, f'Z{num}')
            atom_hover.append(f"{elem_name} (Z={num})")

        # CRITICAL FIX FOR BUG 1: Always add scatter trace (even if empty)
        # This maintains consistent frame structure across all frames
        # If we only add scatter when atom_x is not empty, frames have different trace counts:
        #   - Frame with atoms: [heatmap, contour, scatter] (3 traces)
        #   - Frame without atoms: [heatmap, contour] (2 traces)
        # Plotly requires all frames to have the same number and types of traces
        trace_list.append(
            go.Scatter(
                x=atom_x,  # Will be empty list if no atoms in this slice
                y=atom_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='white',
                    opacity=0.6,
                    line=dict(color='black', width=1)
                ),
                text=atom_hover,  # NEW: Hover text with element names
                hoverinfo='text',  # NEW: Show only custom text on hover
                showlegend=False,
                name='Atoms'
            )
        )

        # CRITICAL: No layout parameter in Frame (just like working test)
        frames.append(go.Frame(data=trace_list, name=str(idx)))

    print(f"Creating figure with {len(frames)} frames...")

    # Create figure (EXACT same as working test)
    fig = go.Figure(data=frames[best_frame_idx].data, frames=frames)

    # Add slider
    sliders = [dict(
        active=best_frame_idx,
        yanchor='top',
        y=0,
        xanchor='left',
        x=0.1,
        currentvalue={
            'prefix': f'{axis.upper()}-Slice: ',
            'visible': True,
            'xanchor': 'right'
        },
        pad={'b': 10, 't': 50},
        len=0.8,
        steps=[
            dict(
                args=[
                    [str(frame_indices[i])],
                    {'frame': {'duration': 0, 'redraw': True},
                     'mode': 'immediate',
                     'transition': {'duration': 0}}
                ],
                label=str(frame_indices[i]),
                method='animate'
            )
            for i in range(len(frames))
        ]
    )]

    # Toggle buttons for global/auto scale
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Global Scale",
                    method="restyle",
                    args=[{"zmin": [vmin, vmin], "zmax": [vmax, vmax]}],
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
            y=1.02,
            yanchor="bottom"
        ),
    ]

    # Update layout
    fig.update_layout(
        title=f"{cube.data_type.capitalize()} - {axis.upper()}-Axis Slice Browser",
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        sliders=sliders,
        updatemenus=updatemenus,
        height=700,
        width=800
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

    print(f"Saving to: {output_file}")
    fig.write_html(output_file)
    print("Done!")

    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slice_browser_standalone.py <cube_file> [axis] [output_file]")
        print("  axis: x, y, or z (default: z)")
        print("  output_file: path to save HTML (default: slice_browser.html)")
        sys.exit(1)

    cube_file = sys.argv[1]
    axis = sys.argv[2] if len(sys.argv) > 2 else 'z'
    output = sys.argv[3] if len(sys.argv) > 3 else 'slice_browser_standalone.html'

    create_slice_browser(cube_file, axis=axis, output_file=output)
