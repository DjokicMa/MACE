"""
This script plots the DOS from a SQLite database containing CRYSTAL calculation results.

Usage:
python autoDOS.py database.db material_id [E lower limit] [E upper limit]

Or interactive mode:
python autoDOS.py database.db

Example:
python autoDOS.py crystal_data.db 2_dia2 -5 5
^ This plots DOS in a range of (+-)5 eV around the fermi level

Interactive example:
python autoDOS.py materials.db
^ This will prompt for material ID and energy range
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import re
from typing import Dict, List, Tuple, Optional

def parse_projection_labels(file_content_preview: str) -> List[str]:
    """
    Extract projection labels from the file_content_preview field.
    Labels are found after # symbols in the DOSS section.
    """
    labels = ['Energy (eV)']  # Start with energy label
    
    # Find lines with # symbols (these contain labels)
    lines = file_content_preview.split('\n')
    for line in lines:
        if '#' in line and not line.strip().startswith('#'):
            # Extract label after #
            label_part = line.split('#')[1].strip()
            
            # Format atom names (e.g., "C S" -> "C_s", "C all" -> "C_all")
            if len(label_part) > 0:
                # First, handle the special cases
                label_part = label_part.replace(' all', '_all')
                label_part = label_part.replace(' S', '_s')
                label_part = label_part.replace(' P', '_p')
                label_part = label_part.replace(' D', '_d')
                label_part = label_part.replace(' F', '_f')
                
                # Clean up any remaining spaces in atom labels
                parts = label_part.split()
                if len(parts) >= 2 and parts[0][0].isupper():
                    # This is an atom label
                    formatted_label = parts[0]
                    if len(parts) > 1:
                        formatted_label += '_' + parts[1].lower()
                    labels.append(formatted_label)
                else:
                    labels.append(label_part)
    
    # Add Total DOS at the end if not already present
    if 'Total DOS' not in labels:
        labels.append('Total DOS')
    
    return labels

def parse_dos_energy_range(file_content_preview: str) -> Optional[Tuple[float, float]]:
    """
    Parse the energy range from DOSS parameters in the file_content_preview.
    If the first two numbers after DOSS are negative, the next line contains the energy range in Hartree.
    Returns (E_min, E_max) in eV, or None if no restriction.
    """
    lines = file_content_preview.split('\n')
    
    for i, line in enumerate(lines):
        if line.strip().startswith('DOSS'):
            # Next line should have the parameters
            if i + 1 < len(lines):
                params_line = lines[i + 1].strip()
                params = params_line.split()
                
                if len(params) >= 4:
                    # Check if parameters at positions 2 and 3 are negative
                    try:
                        param1 = float(params[2])
                        param2 = float(params[3])
                        
                        if param1 < 0 and param2 < 0:
                            # Energy range is specified in the next line
                            if i + 2 < len(lines):
                                range_line = lines[i + 2].strip()
                                range_parts = range_line.split()
                                
                                if len(range_parts) >= 2:
                                    e_min_hartree = float(range_parts[0])
                                    e_max_hartree = float(range_parts[1])
                                    
                                    # Convert from Hartree to eV
                                    e_min_ev = e_min_hartree * 27.2114
                                    e_max_ev = e_max_hartree * 27.2114
                                    
                                    print(f"Detected energy range from DOSS input: {e_min_hartree:.2f} to {e_max_hartree:.2f} Hartree")
                                    print(f"Converted to eV: {e_min_ev:.2f} to {e_max_ev:.2f} eV")
                                    
                                    return (e_min_ev, e_max_ev)
                    except (ValueError, IndexError):
                        pass
    
    return None

def get_dos_data(db_path: str, material_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str], float, Optional[Tuple[float, float]]]:
    """
    Retrieve DOS data from the SQLite database.
    
    Returns:
        energy_points: numpy array of energy values
        total_dos: numpy array of total DOS values
        projections: dictionary of projection name -> DOS values
        labels: list of projection labels
        fermi_energy: Fermi energy in eV
        energy_range: Optional tuple of (E_min, E_max) in eV from DOSS input
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get the DOSS calculation for this material
        # First try to get any DOSS calculation, regardless of status
        cursor.execute("""
            SELECT calc_id, input_settings_json, status
            FROM calculations
            WHERE material_id = ? AND calc_type = 'DOSS'
            ORDER BY created_at DESC
            LIMIT 1
        """, (material_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No DOSS calculation found for material {material_id}")
        
        calc_id, input_settings_json, status = result
        print(f"Found DOSS calculation {calc_id} with status: {status}")
        
        # Debug: Show all available DOS properties
        cursor.execute("""
            SELECT DISTINCT property_name 
            FROM properties 
            WHERE calc_id = ? AND property_name LIKE '%dos%'
            ORDER BY property_name
        """, (calc_id,))
        
        dos_properties = cursor.fetchall()
        print(f"Available DOS properties: {[prop[0] for prop in dos_properties]}")
        
        input_settings = json.loads(input_settings_json)
        
        # Parse projection labels from the file content
        labels = parse_projection_labels(input_settings.get('file_content_preview', ''))
        
        # Parse energy range if specified in DOSS input
        energy_range = parse_dos_energy_range(input_settings.get('file_content_preview', ''))
        
        # Get energy points
        cursor.execute("""
            SELECT property_value_text
            FROM properties
            WHERE calc_id = ? AND property_name = 'energy_points'
        """, (calc_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No energy_points found for calculation {calc_id}")
        
        energy_points = np.array(json.loads(result[0]))
        
        # If energy range was specified in DOSS input, the energy points might be in Hartree
        # Check if the energy points match the Hartree range
        if energy_range is not None:
            e_min_hartree = energy_range[0] / 27.2114
            e_max_hartree = energy_range[1] / 27.2114
            
            # Check if energy points are in Hartree by comparing with expected range
            if (np.min(energy_points) >= e_min_hartree - 0.1 and 
                np.max(energy_points) <= e_max_hartree + 0.1):
                print(f"Energy points appear to be in Hartree. Converting to eV...")
                energy_points = energy_points * 27.2114
        
        # Get total DOS
        cursor.execute("""
            SELECT property_value_text
            FROM properties
            WHERE calc_id = ? AND property_name = 'total_dos'
        """, (calc_id,))
        
        result = cursor.fetchone()
        if not result:
            # Try alternative property names
            cursor.execute("""
                SELECT DISTINCT property_name 
                FROM properties 
                WHERE calc_id = ? AND property_name LIKE '%total%dos%'
            """, (calc_id,))
            
            alternatives = cursor.fetchall()
            if alternatives:
                print(f"No 'total_dos' found. Available alternatives: {[alt[0] for alt in alternatives]}")
            raise ValueError(f"No total_dos found for calculation {calc_id}")
        
        total_dos = np.array(json.loads(result[0]))
        print(f"Total DOS shape: {total_dos.shape}, min: {np.min(total_dos):.4f}, max: {np.max(total_dos):.4f}")
        
        # Additional diagnostics for total DOS
        n_positive = np.sum(total_dos > 0)
        n_negative = np.sum(total_dos < 0)
        n_zero = np.sum(total_dos == 0)
        print(f"Total DOS breakdown: {n_positive} positive, {n_negative} negative, {n_zero} zero values")
        
        total_dos = np.array(json.loads(result[0]))
        
        # Get projected DOS
        cursor.execute("""
            SELECT property_value_text
            FROM properties
            WHERE calc_id = ? AND property_name = 'projected_dos'
        """, (calc_id,))
        
        result = cursor.fetchone()
        if not result:
            print(f"Warning: No projected_dos found for calculation {calc_id}")
            projections = {}
        else:
            projected_dos_data = json.loads(result[0])
            
            # Map the generic projected_dos_N keys to actual labels
            projections = {}
            
            # Handle different formats of projected_dos_data
            if isinstance(projected_dos_data, dict):
                # Extract the numbers from keys like "projected_dos_2"
                dos_items = []
                for key, value in projected_dos_data.items():
                    if key.startswith('projected_dos_'):
                        # Extract the number
                        num = int(key.split('_')[-1])
                        dos_items.append((num, value))
                
                # Sort by number to maintain order
                dos_items.sort(key=lambda x: x[0])
                
                # Map to labels (skip first label which is 'Energy (eV)')
                for i, (num, dos_values) in enumerate(dos_items):
                    if i + 1 < len(labels):  # +1 because we skip 'Energy (eV)'
                        label = labels[i + 1]
                        projections[label] = np.array(dos_values)
                        print(f"Mapped projected_dos_{num} -> {label}")
                        
                        # Show stats for each projection
                        dos_array = np.array(dos_values)
                        n_pos = np.sum(dos_array > 0)
                        n_neg = np.sum(dos_array < 0)
                        print(f"  Stats: {n_pos} positive, {n_neg} negative values, max: {np.max(np.abs(dos_array)):.4f}")
            
            elif isinstance(projected_dos_data, list):
                # If it's a list, map directly to labels
                for i, dos_values in enumerate(projected_dos_data):
                    if i + 1 < len(labels):  # Skip energy label
                        projections[labels[i + 1]] = np.array(dos_values)
        
        # Get Fermi energy
        cursor.execute("""
            SELECT property_value
            FROM properties
            WHERE calc_id = ? AND property_name = 'fermi_energy'
        """, (calc_id,))
        
        result = cursor.fetchone()
        fermi_energy = result[0] if result else 0.0
        
        # Check for POTC data (vacuum reference)
        cursor.execute("""
            SELECT property_value
            FROM properties
            WHERE material_id = ? AND property_name = 'vacuum_potential'
            ORDER BY extracted_at DESC
            LIMIT 1
        """, (material_id,))
        
        result = cursor.fetchone()
        vacuum_ref = result[0] if result else 0.0
        
        # Apply vacuum reference if available
        if vacuum_ref != 0.0:
            energy_points = energy_points + vacuum_ref
        
        # Check for DOS analysis data
        cursor.execute("""
            SELECT property_value_text
            FROM properties
            WHERE calc_id = ? AND property_name = 'dos_analysis'
        """, (calc_id,))
        
        dos_analysis = cursor.fetchone()
        if dos_analysis:
            analysis_data = json.loads(dos_analysis[0])
            print(f"DOS analysis info: {list(analysis_data.keys()) if isinstance(analysis_data, dict) else 'Not a dict'}")
            if isinstance(analysis_data, dict):
                for key in ['total_states', 'dos_at_fermi']:
                    if key in analysis_data:
                        print(f"  {key}: {analysis_data[key]}")
        
        # Let's examine a specific energy point to understand the data
        # Find an energy point near 0 (Fermi level)
        fermi_idx = np.argmin(np.abs(energy_points))
        print(f"\nDOS values at Fermi level (E={energy_points[fermi_idx]:.4f} eV):")
        print(f"  Total DOS: {total_dos[fermi_idx]:.4f}")
        
        if 'C_s' in projections:
            print(f"  C_s: {projections['C_s'][fermi_idx]:.4f}")
        if 'C_p' in projections:
            print(f"  C_p: {projections['C_p'][fermi_idx]:.4f}")
        if 'C_d' in projections:
            print(f"  C_d: {projections['C_d'][fermi_idx]:.4f}")
        if 'C_all' in projections:
            print(f"  C_all: {projections['C_all'][fermi_idx]:.4f}")
            
        # Check if total DOS might be per spin
        if 'C_all' in projections:
            c_all_value = projections['C_all'][fermi_idx]
            total_value = total_dos[fermi_idx]
            if abs(c_all_value) > 0.01:
                print(f"  Ratio C_all/total: {c_all_value/total_value:.2f}")
        
        return energy_points, total_dos, projections, labels, fermi_energy, energy_range
        
    finally:
        conn.close()

def identify_atomic_projections(labels: List[str], projections: Dict[str, np.ndarray]) -> List[str]:
    """
    Identify atomic projections (those with '_all' suffix).
    """
    atomic_projections = []
    
    # Look for labels ending with '_all' as these are atomic projections
    for label in projections.keys():
        if label.endswith('_all'):
            atomic_projections.append(label)
    
    return atomic_projections

def plot_dos(material_id: str, energy_points: np.ndarray, total_dos: np.ndarray, 
             projections: Dict[str, np.ndarray], labels: List[str], 
             fermi_energy: float, E_l: float, E_u: float, dos_energy_range: Optional[Tuple[float, float]]):
    """
    Plot the DOS with user-selected or auto-selected projections.
    """
    # Identify atomic projections
    atomic_projections = identify_atomic_projections(labels, projections)
    
    # User input for projections
    search = []
    print('\nAvailable projections:')
    all_projections = [k for k in projections.keys() if k != 'Total DOS']
    print(all_projections)
    
    # Get unique atom types and orbital types
    atom_types = set()
    orbital_types = set()
    for proj in all_projections:
        parts = proj.split('_')
        if len(parts) >= 2:
            atom_types.add(parts[0])
            orbital_types.add('_' + parts[1])
    
    print(f'\nAtoms found: {sorted(atom_types)}')
    print(f'Orbital types found: {sorted(orbital_types)}')
    
    if atomic_projections:
        print(f'\nAtomic projections (_all) found: {atomic_projections}')
    
    print("\nPlotting options:")
    print("1. Plot atomic contributions only (_all projections)")
    print("2. Plot all available projections")
    print("3. Plot specific orbital type for all atoms (s, p, d, or f)")
    print("4. Manual selection of specific projections")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1' and atomic_projections:
        # Use atomic projections
        search = atomic_projections
        print(f"Plotting atomic contributions: {search}")
    
    elif choice == '2':
        # Plot all projections
        search = all_projections
        print(f"Plotting all projections: {search}")
    
    elif choice == '3':
        # Plot specific orbital type
        print("\nAvailable orbital types:", [o.replace('_', '') for o in sorted(orbital_types)])
        orbital_choice = input("Enter orbital type (s, p, d, f, or all): ").lower().strip()
        
        # Find all projections with this orbital type
        orbital_suffix = '_' + orbital_choice
        selected_projections = []
        for proj in all_projections:
            if proj.endswith(orbital_suffix):
                selected_projections.append(proj)
        
        if selected_projections:
            search = selected_projections
            print(f"Plotting all {orbital_choice} orbitals: {search}")
        else:
            print(f"No {orbital_choice} orbitals found. Proceeding with manual selection.")
            choice = '4'  # Fall back to manual selection
    
    if choice == '4' or (choice == '1' and not atomic_projections):
        # Manual selection
        num = input("\nEnter how many projections you want to plot: ")
        print('Enter string for projection: ')
        for i in range(int(num)):
            n = input("Projection "+str(i+1)+': ')
            search.append(str(n))
    
    elif choice not in ['1', '2', '3']:
        # Invalid choice, default to manual
        print("Invalid choice. Proceeding with manual selection.")
        num = input("\nEnter how many projections you want to plot: ")
        print('Enter string for projection: ')
        for i in range(int(num)):
            n = input("Projection "+str(i+1)+': ')
            search.append(str(n))
    
    # Apply energy range restrictions if detected from DOSS input
    if dos_energy_range is not None:
        dos_e_min, dos_e_max = dos_energy_range
        print(f"\nApplying DOSS energy range restriction: {dos_e_min:.2f} to {dos_e_max:.2f} eV")
        
        # If user-specified range is wider than DOS range, use DOS range
        E_l = max(E_l, dos_e_min - fermi_energy)  # Convert to E-Ef
        E_u = min(E_u, dos_e_max - fermi_energy)  # Convert to E-Ef
        
        print(f"Adjusted plotting range (E-Ef): {E_l:.2f} to {E_u:.2f} eV")
    
    # Also check against actual data range
    actual_min = np.min(energy_points)
    actual_max = np.max(energy_points)
    
    if E_l < actual_min or E_u > actual_max:
        print(f"\nRequested range ({E_l:.2f} to {E_u:.2f} eV) exceeds available data range ({actual_min:.2f} to {actual_max:.2f} eV)")
        E_l = max(E_l, actual_min)
        E_u = min(E_u, actual_max)
        print(f"Auto-adjusted to: {E_l:.2f} to {E_u:.2f} eV")
    
    # Filter data to energy range
    mask = (energy_points >= E_l) & (energy_points <= E_u)
    plot_energy = energy_points[mask]
    plot_total_dos = total_dos[mask]
    
    # Calculate energy step for padding
    energy_step = plot_energy[1] - plot_energy[0] if len(plot_energy) > 1 else 0.1
    n_pad = 5  # Number of padding points
    
    # Extend the plotting range to include padding
    E_l_extended = E_l - n_pad * energy_step
    E_u_extended = E_u + n_pad * energy_step
    
    # Define color schemes
    # Atom-specific base colors
    atom_colors = {
        'H': '#FF6B6B',   # Light red
        'C': '#4ECDC4',   # Teal
        'N': '#45B7D1',   # Sky blue
        'O': '#F7DC6F',   # Yellow
        'F': '#BB8FCE',   # Light purple
        'Si': '#F8C471',  # Light orange
        'P': '#85C1E2',   # Light blue
        'S': '#F9E79F',   # Pale yellow
        'Cl': '#82E0AA',  # Light green
        'Fe': '#EC7063',  # Coral
        'Co': '#AF7AC5',  # Purple
        'Ni': '#5DADE2',  # Blue
        'Cu': '#F5B041',  # Orange
        'Zn': '#58D68D',  # Green
        'Ga': '#E59866',  # Brown orange
        'Ge': '#AED6F1',  # Pale blue
        'As': '#F5CBA7',  # Peach
        'Se': '#A9DFBF',  # Mint
        'Br': '#F9EBEA',  # Very light pink
    }
    
    # Function to adjust color brightness
    def adjust_color_brightness(hex_color, factor):
        """Adjust the brightness of a hex color. Factor > 1 makes it lighter, < 1 makes it darker."""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        if factor > 1:  # Make lighter
            r = int(r + (255 - r) * (factor - 1))
            g = int(g + (255 - g) * (factor - 1))
            b = int(b + (255 - b) * (factor - 1))
        else:  # Make darker
            r = int(r * factor)
            g = int(g * factor)
            b = int(b * factor)
        
        # Ensure values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # Orbital-specific brightness factors and line styles
    orbital_styles = {
        '_s': {'brightness': 0.6, 'linestyle': '-', 'linewidth': 1.8},      # Darkest
        '_p': {'brightness': 0.8, 'linestyle': '-', 'linewidth': 2.0},      # Darker
        '_d': {'brightness': 1.0, 'linestyle': '-', 'linewidth': 2.2},      # Base color
        '_f': {'brightness': 1.2, 'linestyle': '-', 'linewidth': 2.4},      # Lighter
        '_all': {'brightness': 0.9, 'linestyle': '-', 'linewidth': 2.5}     # Slightly darker, thicker
    }
    
    # Default color for unknown atoms
    default_colors = ['#808080', '#A0522D', '#2F4F4F', '#8B4513', '#696969']
    
    # Create figure
    fig = plt.figure(figsize=(4.5, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Labels and limits
    ax.set_xlabel("DOS", size=18)
    ax.set_ylabel(r"$E-E_f$ (eV)", size=20)
    
    # Set limits - use extended range to include padding
    # Get maximum DOS value from total DOS and all selected projections
    max_dos_values = [np.abs(1.1*np.max(plot_total_dos)), np.abs(1.1*np.min(plot_total_dos))]
    
    # Check all selected projections for maximum values
    for projection_name in search:
        for key, dos_values in projections.items():
            if projection_name in key:
                plot_dos_values = dos_values[mask]
                max_dos_values.append(np.abs(1.1*np.max(plot_dos_values)))
                max_dos_values.append(np.abs(1.1*np.min(plot_dos_values)))
    
    xlimit = np.max(max_dos_values)
    ax.set(xlim=(-xlimit, xlimit), ylim=(E_l_extended, E_u_extended))
    ax.set_xticks([])
    plt.yticks(size=18)
    
    # Plot total DOS
    ax.fill_betweenx(plot_energy, plot_total_dos, label='Total DOS', alpha=0.3, color='black')
    ax.plot(plot_total_dos, plot_energy, color='black', linewidth=2.0, alpha=0.8)
    
    # Check and report if projections exceed total DOS
    # For spin-polarized calculations, check each spin channel separately
    total_dos_positive = plot_total_dos[plot_total_dos > 0]
    total_dos_negative = plot_total_dos[plot_total_dos < 0]
    
    max_total_up = np.max(total_dos_positive) if len(total_dos_positive) > 0 else 0
    max_total_down = np.abs(np.min(total_dos_negative)) if len(total_dos_negative) > 0 else 0
    
    for projection_name in search:
        for key, dos_values in projections.items():
            if projection_name in key:
                plot_dos_values = dos_values[mask]
                
                # Check spin-up (positive) and spin-down (negative) separately
                dos_positive = plot_dos_values[plot_dos_values > 0]
                dos_negative = plot_dos_values[plot_dos_values < 0]
                
                if len(dos_positive) > 0 and max_total_up > 0:
                    max_proj_up = np.max(dos_positive)
                    if max_proj_up > max_total_up * 1.1:
                        print(f"Warning: {key} spin-up has larger DOS than total (ratio: {max_proj_up/max_total_up:.2f})")
                
                if len(dos_negative) > 0 and max_total_down > 0:
                    max_proj_down = np.abs(np.min(dos_negative))
                    if max_proj_down > max_total_down * 1.1:
                        print(f"Warning: {key} spin-down has larger DOS than total (ratio: {max_proj_down/max_total_down:.2f})")
    
    # Check if projections sum to total (within the visible range)
    if 'C_all' in search or len(search) > 2:
        # Sum all C_s, C_p, C_d projections
        orbital_projections = []
        for proj_name in ['C_s', 'C_p', 'C_d']:
            if proj_name in projections:
                orbital_projections.append(projections[proj_name][mask])
        
        if len(orbital_projections) == 3:
            sum_orbitals = np.sum(orbital_projections, axis=0)
            
            # Compare with C_all if available
            if 'C_all' in projections:
                c_all = projections['C_all'][mask]
                diff = np.mean(np.abs(sum_orbitals - c_all))
                print(f"\nConsistency check: mean |sum(s,p,d) - all| = {diff:.6f}")
            
            # Compare with total DOS
            # Check a few points in the middle of the range
            mid_idx = len(plot_total_dos) // 2
            sample_indices = [mid_idx - 100, mid_idx, mid_idx + 100]
            
            print("\nSample DOS values at middle of range:")
            for idx in sample_indices:
                if 0 <= idx < len(plot_total_dos):
                    print(f"  Energy {plot_energy[idx]:.3f} eV: Total={plot_total_dos[idx]:.3f}, Sum(s,p,d)={sum_orbitals[idx]:.3f}")
                    if 'C_all' in projections:
                        print(f"    C_all={c_all[idx]:.3f}")
    
    # Plot selected projections with appropriate colors and styles
    used_colors = []
    for i, projection_name in enumerate(search):
        # Find matching projections
        for key, dos_values in projections.items():
            if projection_name in key:
                plot_dos_values = dos_values[mask]
                
                # Determine color based on atom type
                atom_type = key.split('_')[0]  # Extract atom symbol
                if atom_type in atom_colors:
                    base_color = atom_colors[atom_type]
                else:
                    # Use default color
                    base_color = default_colors[i % len(default_colors)]
                
                # Determine style based on orbital
                style_dict = {'brightness': 1.0, 'linestyle': '-', 'linewidth': 2.0}  # Default
                for orbital, style in orbital_styles.items():
                    if orbital in key:
                        style_dict = style
                        break
                
                # Adjust color brightness based on orbital
                color = adjust_color_brightness(base_color, style_dict['brightness'])
                
                # Adjust alpha for different orbitals
                alpha = 0.9 if '_all' in key else 0.8
                
                # Force first and last points to zero to prevent wrap-around
                # This is a definitive fix for the connecting line issue
                modified_dos = plot_dos_values.copy()
                if len(modified_dos) > 2:
                    # Set first and last points to exactly zero
                    modified_dos[0] = 0.0
                    modified_dos[-1] = 0.0
                    
                    # Optional: taper neighboring points
                    if len(modified_dos) > 4:
                        modified_dos[1] *= 0.5
                        modified_dos[-2] *= 0.5
                
                # Plot the modified DOS
                ax.plot(modified_dos, plot_energy, label=key, 
                       linewidth=style_dict['linewidth'], alpha=alpha, color=color, 
                       linestyle=style_dict['linestyle'])
    
    # Reference lines
    plt.axhline(0, color="black", linestyle='--', lw=1.5, alpha=1)  # Fermi level
    plt.axvline(0, color='black', lw=1.5)
    
    # Legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.25), fontsize=10)
    
    # Save and show
    fig.tight_layout()
    plt.tight_layout()
    
    output_dir = './'  # Current directory
    fig.savefig(f'{output_dir}{material_id}.DOSS.svg', format='svg', dpi=300)
    fig.savefig(f'{output_dir}{material_id}.DOSS.png', format='png', dpi=300)
    
    plt.show()
    plt.close('all')

def main():
    if len(sys.argv) < 2:
        print("Usage: python autoDOS.py database.db [material_id] [E_lower] [E_upper]")
        print("   Or: python autoDOS.py database.db  (interactive mode)")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # Check if we're in interactive mode (only database provided)
    if len(sys.argv) == 2:
        # Interactive mode - prompt for everything
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Show available materials with DOSS calculations
        cursor.execute("""
            SELECT DISTINCT material_id 
            FROM calculations 
            WHERE calc_type = 'DOSS'
            ORDER BY material_id
        """)
        
        materials = cursor.fetchall()
        conn.close()
        
        if not materials:
            print("No DOSS calculations found in the database.")
            sys.exit(1)
        
        print("\nAvailable materials with DOSS calculations:")
        for i, (mat_id,) in enumerate(materials):
            print(f"  {mat_id}")
        
        material_id = input("\nEnter material ID: ").strip()
        
        print("\nEnter energy range for plotting (in eV, relative to Fermi level)")
        E_l = float(input("Lower energy limit (e.g., -5): "))
        E_u = float(input("Upper energy limit (e.g., 5): "))
        
    else:
        # Command line mode - get parameters from arguments
        material_id = sys.argv[2]
        
        # Energy range (default: -5 to 5 eV)
        E_l = float(sys.argv[3]) if len(sys.argv) > 3 else -5.0
        E_u = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    
    print(f"\nProcessing DOS for material: {material_id}")
    print(f"Energy range: {E_l} to {E_u} eV")
    
    try:
        # Get DOS data from database
        energy_points, total_dos, projections, labels, fermi_energy, dos_energy_range = get_dos_data(db_path, material_id)
        
        print(f"Found {len(energy_points)} energy points")
        print(f"Found {len(projections)} projections")
        print(f"Fermi energy: {fermi_energy} eV")
        
        # Plot the DOS
        plot_dos(material_id, energy_points, total_dos, projections, labels, fermi_energy, E_l, E_u, dos_energy_range)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()