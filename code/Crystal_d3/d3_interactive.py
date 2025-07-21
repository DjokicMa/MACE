"""
D3 Interactive Configuration Module

This module contains:
- Interactive configuration functions for all D3 calculation types
- Output file parsing utilities
- Basis set parsing and orbital projection generation
- SHRINK extraction and modification utilities
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import sys
from pathlib import Path

# Import k-point utilities from the new module
from d3_kpoints import (
    get_crystal_system_from_space_group,
    get_band_path_from_symmetry,
    get_kpoint_coordinates_from_labels,
    get_literature_kpath_vectors,
    get_seekpath_full_kpath,
    extract_and_process_shrink,
    scale_kpoint_segments,
    BAND_TEMPLATES
)

# Add Crystal_d12 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
# Import yes_no_prompt function
def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """Prompt for yes/no input with default value."""
    if default.lower() == "yes":
        prompt_str = f"{prompt} [Y/n]: "
        default_val = True
    else:
        prompt_str = f"{prompt} [y/N]: "
        default_val = False
    
    response = input(prompt_str).strip().lower()
    if not response:
        return default_val
    return response in ['y', 'yes']
# Import generate_k_points function
def generate_k_points(system: str, n_points: int = 30) -> List[Tuple[str, List[float]]]:
    """Generate high-symmetry k-points for band structure calculations."""
    # For now, return a simple path
    return [
        ("G", [0.0, 0.0, 0.0]),
        ("X", [0.5, 0.0, 0.0]),
        ("M", [0.5, 0.5, 0.0]),
        ("G", [0.0, 0.0, 0.0])
    ]


def get_band_info_from_output(out_file: str) -> Dict[str, Any]:
    """Extract band structure information from CRYSTAL output file."""
    info = {
        'n_bands': 0,
        'valence_bands': 0,
        'conduction_bands': 0,
        'band_gap': None,
        'gap_type': None,
        'fermi_energy': None,
        'n_k_points': 0,
        'space_group': 1,
        'lattice_type': 'P',
        'lattice_params': None
    }
    
    if not os.path.exists(out_file):
        return info
    
    with open(out_file, 'r') as f:
        content = f.read()
    
    # Number of bands (AOs)
    ao_match = re.search(r'NUMBER OF AO\s+(\d+)', content)
    if ao_match:
        info['n_bands'] = int(ao_match.group(1))
    
    # Band gap information
    gap_match = re.search(r'(DIRECT|INDIRECT) ENERGY BAND GAP\s*:\s*([\d.]+)', content)
    if gap_match:
        info['gap_type'] = gap_match.group(1).lower()
        info['band_gap'] = float(gap_match.group(2))
    
    # Fermi energy - try multiple patterns
    fermi_energy = None
    
    # Pattern 1: Standard FERMI ENERGY line
    fermi_match = re.search(r'FERMI ENERGY\s*\[.*?\]\s*:\s*([-\d.]+)', content)
    if fermi_match:
        fermi_energy = float(fermi_match.group(1))
    
    # Pattern 2: EFERMI from final SCF cycle (metallic systems)
    if fermi_energy is None:
        # Look for SCF convergence marker and then find the last EFERMI before it
        scf_ended_match = re.search(r'== SCF ENDED[^\n]*\n', content)
        if scf_ended_match:
            # Get content up to SCF convergence
            content_before_convergence = content[:scf_ended_match.start()]
            # Find all EFERMI occurrences before convergence
            efermi_matches = re.findall(r'EFERMI\(AU\)\s*([-\d.E+]+)', content_before_convergence)
            if efermi_matches:
                # Take the last one (from final SCF cycle)
                fermi_energy = float(efermi_matches[-1])
        else:
            # Fallback: if no SCF ENDED marker, still try to find last EFERMI
            efermi_matches = re.findall(r'EFERMI\(AU\)\s*([-\d.E+]+)', content)
            if efermi_matches:
                fermi_energy = float(efermi_matches[-1])
    
    # Pattern 3: Use TOP OF VALENCE BANDS as approximation for insulators
    if fermi_energy is None:
        # Look for band structure info after SCF convergence
        scf_ended_match = re.search(r'== SCF ENDED[^\n]*\n', content)
        if scf_ended_match:
            # Get content after SCF convergence (where band info is printed)
            content_after_convergence = content[scf_ended_match.end():]
            # Find VBM energies after convergence
            vbm_matches = re.findall(r'TOP OF VALENCE BANDS.*?EIG\s*([-\d.E+]+)\s*AU', content_after_convergence)
            if vbm_matches:
                vbm_energies = [float(e) for e in vbm_matches]
                fermi_energy = max(vbm_energies)
        
        # Fallback: search entire content if no SCF marker found
        if fermi_energy is None:
            vbm_matches = re.findall(r'TOP OF VALENCE BANDS.*?EIG\s*([-\d.E+]+)\s*AU', content)
            if vbm_matches:
                vbm_energies = [float(e) for e in vbm_matches]
                fermi_energy = max(vbm_energies)
    
    if fermi_energy is not None:
        info['fermi_energy'] = fermi_energy
    
    # Valence and conduction bands
    vb_match = re.search(r'TOP OF VALENCE BANDS.*?BAND\s+(\d+)', content, re.DOTALL)
    if vb_match:
        info['valence_bands'] = int(vb_match.group(1))
    
    cb_match = re.search(r'BOTTOM OF CONDUCTION BANDS.*?BAND\s+(\d+)', content, re.DOTALL)
    if cb_match:
        info['conduction_bands'] = int(cb_match.group(1)) - info['valence_bands']
    
    # Space group
    sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
    if sg_match:
        info['space_group'] = int(sg_match.group(1))
    
    # Lattice type from space group symbol
    sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', content)
    if sg_symbol_match:
        symbol = sg_symbol_match.group(1).strip()
        if symbol:
            info['lattice_type'] = symbol[0]
    
    # Lattice parameters (A, B, C, ALPHA, BETA, GAMMA)
    lattice_match = re.search(r'LATTICE PARAMETERS.*?\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
    if lattice_match:
        info['lattice_params'] = [
            float(lattice_match.group(1)),  # A
            float(lattice_match.group(2)),  # B
            float(lattice_match.group(3)),  # C
            float(lattice_match.group(4)),  # ALPHA
            float(lattice_match.group(5)),  # BETA
            float(lattice_match.group(6))   # GAMMA
        ]
    
    return info


def extract_shrink_from_d12(d12_file: str, for_doss: bool = False) -> List[str]:
    """Extract SHRINK lines from D12 file and format for NEWK.
    
    Args:
        d12_file: Path to D12 file
        for_doss: If True, format for DOSS NEWK (same values, not doubled)
        
    Returns:
        List of strings representing the SHRINK/NEWK section
    """
    
    if not os.path.exists(d12_file):
        return ["48 48"]  # Default fallback
        
    shrink_lines = []
    found_shrink = False
    line_count = 0
    
    with open(d12_file, 'r') as f:
        for line in f:
            if found_shrink:
                shrink_lines.append(line.strip())
                line_count += 1
                if line_count >= 2:  # We need at most 2 lines after SHRINK
                    break
            elif 'SHRINK' in line:
                found_shrink = True
                
    if not shrink_lines:
        return ["48 48"]  # Default fallback
    
    # Parse the SHRINK format
    first_line = shrink_lines[0].split()
    
    if len(first_line) == 2:
        # Simple format: IS IPMG
        is_val = int(first_line[0])
        ipmg = int(first_line[1])
        
        if for_doss:
            # For DOSS, use same value for both parameters
            return [f"{is_val} {is_val}"]
        else:
            # For other calculations, keep original
            return [f"{is_val} {ipmg}"]
    
    elif len(first_line) == 1 and first_line[0] == "0":
        # Format with separate k-point specification
        if len(shrink_lines) > 1:
            # Second line has k1 k2 k3
            k_line = shrink_lines[1].split()
            if len(k_line) >= 3:
                k1, k2, k3 = int(k_line[0]), int(k_line[1]), int(k_line[2])
                
                if for_doss:
                    # For DOSS, use max value for both parameters
                    max_k = max(k1, k2, k3)
                    return [f"{max_k} {max_k}"]
                else:
                    # Keep original format
                    return shrink_lines[:2]
                
    # Return first line if we couldn't parse
    return [shrink_lines[0]]


def parse_basis_set_info(out_file: str) -> Tuple[List[List], int]:
    """Parse basis set information from CRYSTAL output file using alldos.py logic."""
    
    data_list = []
    n_ao = 0
    
    if not os.path.exists(out_file):
        return data_list, n_ao
        
    with open(out_file, 'r') as f:
        lines = f.readlines()
    
    # Extract number of AOs
    for line in lines:
        if "NUMBER OF AO" in line:
            clean_line = [x for x in line.split(" ") if x != ""]
            if len(clean_line) > 3:
                n_ao = int(clean_line[3])
            break
    
    # Find LOCAL ATOMIC FUNCTIONS BASIS SET section
    index_shells = 0
    for i, line in enumerate(lines):
        if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
            index_shells = i
            break
    
    if index_shells == 0:
        return data_list, n_ao
    
    # Extract shell data starting from 4 lines after the header (like alldos.py)
    unclean_shells = []
    for index in range(index_shells + 4, len(lines)):
        if "PROCESS" in lines[index]:
            continue
        if (
            "*******" in lines[index]
            or " INFORMATION " in lines[index]
            or lines[index] == "\n"
        ):
            break
        else:
            unclean_shells.append(lines[index].rstrip('\n'))
    
    # Convert to data_list format
    for line in unclean_shells:
        data_list.append([x for x in line.split(" ") if x != ""])
    
    return data_list, n_ao


def get_unique_atoms(atoms: List[str]) -> List[Tuple[str, int]]:
    """Get unique atoms and their first occurrence index."""
    seen = {}
    result = []
    for i, atom in enumerate(atoms):
        if atom not in seen:
            seen[atom] = i
            result.append((atom, i))
    return result

def get_atoms_and_shells(data_list: List[List]) -> Tuple[List[str], List[Dict]]:
    """Extract atoms and shell information from alldos.py style data_list."""
    
    # Find all atoms in the data (lines with 5 elements)
    atoms = []
    atom_list = []  # Unique atom types
    atoms_index = []
    
    for i, line in enumerate(data_list):
        if len(line) == 5:
            atoms.append(line[1])  # Element symbol
            if line[1] not in atom_list:
                atom_and_index = [line[1], i]
                atom_list.append(line[1])
                atoms_index.append(atom_and_index)
    
    # Count shells for each atom type (similar to alldos.py's number_shells)
    atoms_shells = {}
    for atom in atoms_index:
        S, P, SP, D, F = 0, 0, 0, 0, 0
        for i in range(int(atom[1]) + 1, len(data_list)):
            if len(data_list[i]) >= 5:
                break
            elif len(data_list[i]) > 0 and "S" in data_list[i][-1]:
                S += 1
            elif len(data_list[i]) > 0 and "SP" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    SP_shells = data_list[i][0].split("-")
                    for j in range(int(SP_shells[0]), int(SP_shells[1]) + 1):
                        SP += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        SP += 1
            elif len(data_list[i]) > 0 and "P" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    P_shells = data_list[i][0].split("-")
                    for j in range(int(P_shells[0]), int(P_shells[1]) + 1):
                        P += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        P += 1
            elif len(data_list[i]) > 0 and "D" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    D_shells = data_list[i][0].split("-")
                    for j in range(int(D_shells[0]), int(D_shells[1]) + 1):
                        D += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        D += 1
            elif len(data_list[i]) > 0 and "F" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    F_shells = data_list[i][0].split("-")
                    for j in range(int(F_shells[0]), int(F_shells[1]) + 1):
                        F += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        F += 1
        atoms_shells[atom[0]] = {"S": S, "SP": SP, "P": P, "D": D, "F": F}
        S, P, SP, D, F = 0, 0, 0, 0, 0
    
    return atoms, atoms_shells


def get_shells(atoms: List[str], atoms_shells: Dict) -> Dict[str, Dict[str, List[int]]]:
    """Generate shell information for each unique element (alldos.py style)."""
    # In this context, atoms_shells is already the shell count dictionary
    # We need to use generate_orbital_indices instead
    return generate_orbital_indices(atoms, atoms_shells)

def generate_orbital_indices(atoms: List[str], atoms_shells: List[Dict]) -> Dict[str, Dict[str, List[int]]]:
    """Generate orbital indices similar to alldos.py get_shells function."""
    
    AOindex = 1
    total_shells = {}
    
    for element in atoms_shells:
        total_shells[element] = {"S": [], "P": [], "D": [], "F": [], "SP": []}
    
    # Process all atoms (not just unique elements)
    for atom in atoms:
        for shell in range(0, int(atoms_shells[atom]["S"])):
            total_shells[atom]["S"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["SP"])):
            total_shells[atom]["SP"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["P"])):
            total_shells[atom]["P"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["D"])):
            total_shells[atom]["D"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["F"])):
            total_shells[atom]["F"].append(AOindex)
            AOindex += 1
    
    return total_shells


def create_doss_projections(total_shells: Dict, element_only: bool = False, include_totals: bool = True) -> List[str]:
    """Create DOSS projection specifications from orbital data (alldos.py style)."""
    
    projections = []
    
    if element_only:
        # Project by element only (option 2)
        for atom in total_shells:
            atom_total_indices = []
            
            # Collect all orbital indices for this element
            for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                atom_total_indices.extend(total_shells[atom][shell_type])
            
            if atom_total_indices:
                proj_line = f"{len(atom_total_indices)}  {' '.join(map(str, atom_total_indices))}  #{atom} all"
                projections.append(proj_line)
    else:
        # Project by element and orbital type (options 3 and 4)
        for atom in total_shells:
            # Add each shell type separately
            for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                if len(total_shells[atom][shell_type]) > 0:
                    indices = total_shells[atom][shell_type]
                    proj_line = f"{len(indices)}  {' '.join(map(str, indices))}  #{atom} {shell_type}"
                    projections.append(proj_line)
            
            # Add total for this element only if requested (option 3)
            if include_totals:
                atom_total_indices = []
                for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                    atom_total_indices.extend(total_shells[atom][shell_type])
                
                if atom_total_indices:
                    proj_line = f"{len(atom_total_indices)}  {' '.join(map(str, atom_total_indices))}  #{atom} all"
                    projections.append(proj_line)
    
    return projections


def configure_band_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure BAND calculation settings interactively."""
    print("\n=== BAND STRUCTURE CONFIGURATION ===")
    
    band_config = {}
    
    # Don't ask for title in shared mode - it will be set per material
    # Title will be automatically set in CRYSTALOptToD3.py to include material name
    
    # Path definition method
    print("\nBand path definition:")
    print("1: Automatic - Use standard path based on crystal symmetry from output file")
    print("2: Template selection - Choose from common band paths")
    print("3: Custom labels - Specify path using labels (G, X, M, etc.)")
    print("4: Fractional coordinates - Specify path using k-point vectors")
    
    path_method = input("Select method (1-4) [1]: ").strip() or "1"
    
    if path_method == "1":
        # True automatic path from output file
        if out_file:
            # Extract space group from output file
            import re
            space_group = 1
            lattice_type = 'P'
            
            try:
                with open(out_file, 'r') as f:
                    content = f.read()
                    
                # Find space group number
                sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
                if sg_match:
                    space_group = int(sg_match.group(1))
                
                # Find lattice type from space group symbol
                sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', content)
                if sg_symbol_match:
                    symbol = sg_symbol_match.group(1).strip()
                    if symbol:
                        lattice_type = symbol[0]
                
                print(f"\nDetected space group: {space_group} ({lattice_type} lattice)")
            except:
                print("\nCould not read space group from output file, using default P1")
        
        # Get appropriate path based on symmetry
        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
        
        # Now ask which format to use for the path
        print("\nBand path format:")
        print("1: High-symmetry labels (CRYSTAL-compatible subset)")
        print("2: K-point vectors (fractional coordinates)")
        print("3: Literature path with vectors (comprehensive, includes non-CRYSTAL points)")
        print("4: SeeK-path full paths (extended Bravais lattice notation)")
        format_choice = input("Select format (1-4) [1]: ").strip() or "1"
        
        if format_choice == "1":
            # Use labels - CRYSTAL-compatible subset
            band_config["path_method"] = "labels"
            band_config["shrink"] = 0
            band_config["path"] = path_labels
            band_config["auto_path"] = True  # Mark that this was auto-detected
        elif format_choice == "2":
            # Convert labels to k-point vectors
            band_config["path_method"] = "coordinates"
            
            # Calculate appropriate shrink based on lattice parameters
            # Try to extract lattice parameters from output file
            a, b, c = 10.0, 10.0, 10.0  # Default fallback
            if out_file:
                try:
                    with open(out_file, 'r') as f:
                        content = f.read()
                    # Look for lattice parameters
                    import re
                    # Pattern for PRIMITIVE CELL or LATTICE PARAMETERS
                    param_match = re.search(r'LATTICE PARAMETERS.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if not param_match:
                        param_match = re.search(r'PRIMITIVE CELL.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if param_match:
                        a = float(param_match.group(1))
                        b = float(param_match.group(2))
                        c = float(param_match.group(3))
                except:
                    pass
            
            # Extract and process shrink factor
            base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
            input_dir = Path(out_file).parent if out_file else Path.cwd()
            shrink = extract_and_process_shrink(out_file, base_name, input_dir, band_config)
            band_config["shrink"] = shrink
            band_config["auto_path"] = True  # Mark that this was auto-detected
            
            # Get k-point coordinates for the labels (returns fractional coordinates)
            frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
            
            # Scale fractional coordinates by shrink factor to get integers
            coord_segments = scale_kpoint_segments(frac_segments, shrink)
            band_config["segments"] = coord_segments
            print(f"\n✓ Converted path to k-point vectors (shrink={shrink})")
        else:
            # SeeK-path full paths with comprehensive k-points
            band_config["path_method"] = "coordinates"
            
            # Extract and process shrink factor
            base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
            input_dir = Path(out_file).parent if out_file else Path.cwd()
            shrink = extract_and_process_shrink(out_file, base_name, input_dir, band_config)
            band_config["shrink"] = shrink
            band_config["auto_path"] = True
            
            if format_choice == "3":
                # Literature path with comprehensive k-points
                band_config["literature_path"] = True
                
                # Get literature path based on space group and lattice type
                frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                if frac_segments:
                    # Scale fractional coordinates by shrink factor
                    coord_segments = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    print(f"✓ Using comprehensive literature k-path with {len(coord_segments)} segments")
                else:
                    # Fallback to standard path
                    print("\nLiterature path not available, using standard path")
                    coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                    band_config["segments"] = coord_segments
            else:
                # format_choice == "4" - SeeK-path
                band_config["seekpath_full"] = True
                
                # Get SeeK-path full path based on extended Bravais lattice (in fractional coordinates)
                frac_segments = get_seekpath_full_kpath(space_group, lattice_type, out_file)
                if frac_segments:
                    # Scale fractional coordinates by shrink factor
                    coord_segments = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    print(f"✓ Using SeeK-path full k-path with {len(coord_segments)} segments")
                else:
                    # Fallback to standard path
                    print("\nSeeK-path full path not available, using standard path")
                    coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                    band_config["segments"] = coord_segments
    
    elif path_method == "2":
        # Template selection
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        
        print("\nSelect crystal system:")
        print("1: Cubic")
        print("2: Hexagonal")
        print("3: Tetragonal")
        print("4: Orthorhombic")
        print("5: Monoclinic")
        print("6: Triclinic")
        
        system_choice = input("Select system (1-6) [1]: ").strip() or "1"
        
        template_map = {
            "1": ["cubic", "cubic2", "cubic3"],
            "2": ["hexagonal", "hexagonal2"],
            "3": ["tetragonal", "tetragonal2"],
            "4": ["orthorhombic", "orthorhombic2", "orthorhombic3", "orthorhombic4"],
            "5": ["monoclinic", "monoclinic2"],
            "6": ["triclinic"]
        }
        
        templates = template_map.get(system_choice, ["cubic"])
        
        if len(templates) > 1:
            print("\nAvailable templates:")
            for i, template in enumerate(templates):
                t = BAND_TEMPLATES[template]
                print(f"{i+1}: {t['description']}")
                print(f"   Path: {' → '.join(t['path'])}")
            
            template_idx = int(input(f"\nSelect template (1-{len(templates)}) [1]: ") or 1) - 1
            template_name = templates[template_idx]
        else:
            template_name = templates[0]
        
        template = BAND_TEMPLATES[template_name]
        band_config["path"] = template["path"]
        band_config["template"] = template_name
        print(f"\n✓ Using {template['description']} template")
    
    elif path_method == "3":
        # Custom labels
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        
        print("\nEnter band path as space-separated labels (e.g., G X M G):")
        print("Note: Use G for Gamma point")
        path_str = input("Path: ").strip()
        band_config["path"] = path_str.split() if path_str else ["G", "X", "M", "G"]
    
    else:
        # Manual k-point entry
        band_config["path_method"] = "manual"
        
        # Ask about mixed segments
        use_mixed = yes_no_prompt("\nUse mixed label/coordinate segments?", "no")
        
        if use_mixed:
            # Mixed path - each segment can be labels or coordinates
            n_segments = int(input("Number of path segments: "))
            segments = []
            band_config["shrink"] = 16  # Default, will be adjusted if needed
            
            for i in range(n_segments):
                print(f"\nSegment {i+1}:")
                seg_type = input("Type (L)abels or (C)oordinates [L]: ").strip().upper() or "L"
                
                if seg_type == "L":
                    labels = input("Enter two labels (e.g., G X): ").strip().upper()
                    segments.append(labels)
                    band_config["shrink"] = 0  # Labels mode
                else:
                    print("Enter coordinates (fractional or integer):")
                    coords = input("Six numbers (x1 y1 z1 x2 y2 z2): ").strip()
                    segments.append(coords)
                    # Check if we need shrink
                    try:
                        vals = [float(x) for x in coords.split()]
                        if all(v == int(v) for v in vals):
                            # Integer coordinates, need appropriate shrink
                            max_val = max(abs(int(v)) for v in vals)
                            if max_val > band_config.get("shrink", 16):
                                band_config["shrink"] = max_val * 2
                    except:
                        pass
            
            band_config["manual_segments"] = segments
            
        else:
            # All segments same type
            print("\nSegment format:")
            print("1: All segments use labels")
            print("2: All segments use fractional coordinates")
            format_choice = input("Select format (1-2) [1]: ").strip() or "1"
            
            n_segments = int(input("Number of path segments: "))
            segments = []
            
            if format_choice == "1":
                # All segments use labels
                band_config["shrink"] = 0  # Use labels mode
                for i in range(n_segments):
                    segment = input(f"Segment {i+1} (e.g., 'G X'): ").strip().upper()
                    segments.append(segment)
            else:
                # All segments use coordinates
                # Try to extract shrink from D12 file if available
                base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
                input_dir = Path(out_file).parent if out_file else Path.cwd()
                default_shrink = extract_and_process_shrink(out_file, base_name, input_dir, {})
                
                shrink_input = input(f"Shrink factor for k-points [{default_shrink}]: ").strip()
                shrink = int(shrink_input) if shrink_input else default_shrink
                
                # Ensure it's even for cleaner paths
                if shrink % 2 == 1:
                    shrink = shrink + 1
                    print(f"✓ Using shrink factor {shrink} (rounded to even number)")
                
                band_config["shrink"] = shrink
                
                print("\nEnter segments as fractional coordinates (will be scaled by shrink factor):")
                for i in range(n_segments):
                    segment_str = input(f"Segment {i+1}: ").strip()
                    try:
                        coords = [float(x) for x in segment_str.split()]
                        if len(coords) == 6:
                            # Scale fractional to integer using unified function
                            # Convert to segment format for scaling function
                            scaled_segment = scale_kpoint_segments([coords], shrink)[0]
                            segments.append(scaled_segment)
                        else:
                            print("Warning: Expected 6 numbers, skipping segment")
                    except ValueError:
                        print("Warning: Invalid input, skipping segment")
                
            band_config["segments"] = segments
    
    # Number of k-points
    n_points = int(input("\nTotal number of k-points along path [1000]: ") or 1000)
    band_config["n_points"] = n_points
    
    # Band range
    print("\nBand range selection:")
    print("1: All bands")
    print("2: Around Fermi level (~20 valence + ~30 conduction bands)")
    print("3: Custom range")
    
    band_range = input("Select option (1-3) [1]: ").strip() or "1"
    
    if band_range == "1":
        # This will be set per material in CRYSTALOptToD3.py based on n_ao
        band_config["bands"] = "auto"
        band_config["auto_bands"] = True
        band_config["first_band"] = 1
        band_config["last_band"] = None  # Will be set to n_ao
    elif band_range == "2":
        # Extract band info from output
        if out_file:
            band_info = get_band_info_from_output(out_file)
            if band_info['valence_bands'] > 0:
                # Around Fermi: ~20 valence + ~30 conduction
                first = max(1, band_info['valence_bands'] - 20)
                last = min(band_info['n_bands'], band_info['valence_bands'] + 30)
                band_config["first_band"] = first
                band_config["last_band"] = last
                print(f"\n✓ Selected bands {first} to {last} (around Fermi level)")
            else:
                # Fallback to all bands
                band_config["first_band"] = 1
                band_config["last_band"] = None
        else:
            # Default range around typical Fermi
            band_config["first_band"] = 1
            band_config["last_band"] = 50
    else:
        # Custom range
        first = int(input("First band [1]: ") or 1)
        last = int(input("Last band [all]: ") or 0)
        band_config["first_band"] = first
        band_config["last_band"] = last if last > 0 else None
    
    # Output options
    band_config["plot"] = True  # Always generate plot
    band_config["print_eigenvalues"] = yes_no_prompt("\nPrint eigenvalues to output?", "no")
    
    return band_config


def configure_doss_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure DOSS calculation settings interactively."""
    print("\n=== DENSITY OF STATES CONFIGURATION ===")
    
    doss_config = {}
    
    # Projection type
    print("\nDOS projection type:")
    print("1: Total DOS only")
    print("2: Projected DOS by element (all orbitals)")
    print("3: Projected DOS by element and orbital type (with totals)")
    print("4: Projected DOS by orbital type only (no element totals)")
    print("5: Projected DOS on specific atoms")
    print("6: Manual projections (custom combinations)")
    
    proj_type = input("Select projection type (1-6) [3]: ").strip() or "3"
    
    doss_config["projection_type"] = int(proj_type)
    
    if proj_type == "1":
        # Total DOS only
        doss_config["npro"] = 0
        doss_config["project_orbital_types"] = False
    elif proj_type == "2":
        # By element only
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = True
    elif proj_type == "3":
        # By element and orbital type with totals (default)
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = False
        doss_config["include_element_totals"] = True
    elif proj_type == "4":
        # By orbital type only (no element totals)
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = False
        doss_config["include_element_totals"] = False
    elif proj_type == "5":
        # Specific atoms
        doss_config["project_orbital_types"] = False
        print("\nAtom selection:")
        print("1: Project on all atoms")
        print("2: Project on specific atoms")
        
        atom_choice = input("Select (1-2) [1]: ").strip() or "1"
        
        if atom_choice == "1":
            doss_config["project_all_atoms"] = True
        else:
            atom_list = input("Enter atom numbers (e.g., 1 3 5): ").strip()
            if atom_list:
                doss_config["project_atoms"] = [int(a) for a in atom_list.split()]
            else:
                doss_config["project_atoms"] = [1]
    else:
        # Manual projections (option 6)
        doss_config["project_orbital_types"] = False
        doss_config["manual_projections"] = True
        
        # Get atom information from output file if available
        if out_file and os.path.exists(out_file):
            # Parse the output file for atom information
            n_atoms = 0
            atom_elements = []
            
            with open(out_file, 'r') as f:
                content = f.read()
            
            # Extract number of atoms
            atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s+(\d+)', content)
            if atoms_match:
                n_atoms = int(atoms_match.group(1))
            else:
                atoms_match = re.search(r'ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)', content)
                if atoms_match:
                    n_atoms = int(atoms_match.group(1))
            
            # Extract atom elements
            atom_pattern = re.compile(r'^\s*\d+\s+[TF]\s+\d+\s+([A-Z][a-z]?)\s+', re.MULTILINE)
            atom_elements = atom_pattern.findall(content)
            
            # Parse basis set to show available orbitals
            data_list, n_ao = parse_basis_set_info(out_file)
            total_shells = {}
            
            if data_list:
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                atoms_index = get_unique_atoms(atoms)
                total_shells = get_shells(atoms, atoms_shells)
            
            if n_atoms > 0:
                print(f"\nSystem contains {n_atoms} atoms:")
                if atom_elements:
                    for i, elem in enumerate(atom_elements[:n_atoms]):
                        print(f"  Atom {i+1}: {elem}")
                else:
                    print(f"  {n_atoms} atoms (element types not determined)")
                
                # Show available orbitals for each element
                if total_shells:
                    print("\nAvailable orbital projections by element:")
                    for element in sorted(set(atom_elements[:n_atoms])):
                        if element in total_shells:
                            available = []
                            for shell_type in ["S", "P", "SP", "D", "F"]:
                                if total_shells[element].get(shell_type):
                                    available.append(shell_type)
                            if available:
                                print(f"  {element}: {', '.join(available)}, all")
                
                print("\nInput format examples:")
                print("  - Atom projection: '1' or '1C' (for atom 1)")
                print("  - Element total: 'C all' or 'C' (all orbitals of element C)")
                print("  - Orbital-specific: 'C S', 'O P', etc.")
                print("\nNote: The script will convert your input to the proper DOSS format.")
        
        # Helper function to validate projection
        def validate_projection(spec: str, total_shells: Dict, atom_elements: List[str], n_atoms: int) -> Tuple[bool, str]:
            """Validate a projection specification and return (is_valid, error_message)."""
            spec = spec.strip()
            if not spec:
                return False, "Empty specification"
            
            # Check if it's an atom projection (starts with a number)
            if spec[0].isdigit():
                # Extract atom number
                atom_num_match = re.match(r'^(\d+)', spec)
                if atom_num_match:
                    atom_num = int(atom_num_match.group(1))
                    if atom_num < 1 or atom_num > n_atoms:
                        return False, f"Atom number {atom_num} out of range (1-{n_atoms})"
                    return True, ""
            
            # Otherwise, it's an element/orbital projection
            parts = spec.split()
            if not parts:
                return False, "Empty specification"
            
            element = parts[0].upper()
            
            # Check if element exists in the system
            if element not in total_shells:
                available_elements = sorted(total_shells.keys())
                return False, f"Element {element} not found in system. Available: {', '.join(available_elements)}"
            
            # If just element name or "element all", it's valid
            if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "all"):
                return True, ""
            
            # Specific orbital type
            if len(parts) == 2:
                orbital_type = parts[1].upper()
                
                # Check if orbital type exists for this element
                if orbital_type in total_shells[element] and total_shells[element][orbital_type]:
                    return True, ""
                else:
                    # Check for combined SP
                    if orbital_type == "SP" and "S" in total_shells[element] and "P" in total_shells[element]:
                        return True, ""
                    else:
                        available = [o for o in ["S", "P", "SP", "D", "F"] if o in total_shells[element] and total_shells[element][o]]
                        return False, f"Orbital type {orbital_type} not available for element {element}. Available: {', '.join(available)}"
            
            return False, f"Invalid projection format: '{spec}'"
        
        doss_config["manual_projection_specs"] = []
        valid_projections = []
        
        print("\nEnter projections (press Enter without input to stop):")
        i = 1
        while True:
            print(f"\nProjection {i}:")
            proj_spec = input("Enter projection specification: ").strip()
            
            if not proj_spec:
                if i == 1:
                    print("At least one projection is required!")
                    continue
                else:
                    break
            
            # Validate the projection
            is_valid, error_msg = validate_projection(proj_spec, total_shells, atom_elements, n_atoms)
            
            if is_valid:
                valid_projections.append(proj_spec)
                i += 1
            else:
                print(f"  ERROR: {error_msg}")
                print("  Please enter a valid projection.")
        
        doss_config["manual_projection_specs"] = valid_projections
        doss_config["npro"] = len(valid_projections)
        print(f"\nTotal valid projections: {len(valid_projections)}")
    
    # Energy window or band range
    print("\nEnergy/band range:")
    print("1: All bands")
    print("2: Energy window around Fermi level")
    print("3: Specific band range")
    
    range_choice = input("Select option (1-3) [1]: ").strip() or "1"
    
    if range_choice == "2":
        # Energy window - collect in eV but convert to Ha
        EV_TO_HA = 1.0 / 27.211386
        bmi_ev = float(input("Lower energy limit (eV below Fermi) [-10]: ") or -10)
        bma_ev = float(input("Upper energy limit (eV above Fermi) [20]: ") or 20)
        # Convert to Hartree
        bmi_ha = bmi_ev * EV_TO_HA
        bma_ha = bma_ev * EV_TO_HA
        doss_config["energy_window"] = (bmi_ha, bma_ha)
    elif range_choice == "3":
        # Band range
        first = int(input("First band [1]: ") or 1)
        last = int(input("Last band [all]: ") or -1)
        if last > 0:
            doss_config["band_range"] = (first, last)
    
    # Other parameters
    n_points = int(input("\nNumber of energy points [10000]: ") or 10000)
    doss_config["n_points"] = n_points
    
    # Output format
    print("\nOutput format:")
    print("1: CRYSTAL fort.25 format")
    print("2: DOSS.DAT format")
    
    out_format = input("Select format (1-2) [2]: ").strip() or "2"
    doss_config["output_format"] = int(out_format) if out_format in ["1", "2"] else 2
    
    # Polynomial order for DOS smoothing
    npol = int(input("\nPolynomial order for smoothing [14]: ") or 14)
    doss_config["npol"] = npol
    
    # Always print integrated DOS
    doss_config["print_integrated"] = True
    
    return doss_config


def configure_transport_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure BOLTZTRA transport calculation settings."""
    print("\n=== TRANSPORT PROPERTIES CONFIGURATION ===")
    
    transport_config = {}
    
    # Temperature range
    print("\nTemperature range:")
    t_min = float(input("Minimum temperature (K) [100]: ") or 100)
    t_max = float(input("Maximum temperature (K) [800]: ") or 800)
    t_step = float(input("Temperature step (K) [50]: ") or 50)
    transport_config["temperature_range"] = (t_min, t_max, t_step)
    
    # Chemical potential range
    print("\nChemical potential range:")
    
    # Try to get Fermi energy if output file provided
    fermi_energy_ev = None
    if out_file:
        band_info = get_band_info_from_output(out_file)
        if band_info.get('fermi_energy') is not None:
            # Convert from Hartree to eV
            fermi_energy_ev = band_info['fermi_energy'] * 27.211386
            print(f"Detected Fermi energy: {fermi_energy_ev:.3f} eV")
    
    print("\nChemical potential options:")
    print("1: Relative to Fermi energy (automatic)")
    print("2: Relative to VBM (manual)")
    print("3: Absolute values")
    
    mu_option = input("Select option (1-3) [1]: ").strip() or "1"
    
    if mu_option == "1" and fermi_energy_ev is not None:
        # Relative to Fermi energy
        print("\nRange relative to Fermi energy:")
        mu_min_rel = float(input("Minimum μ (eV below Fermi) [-2.0]: ") or -2.0)
        mu_max_rel = float(input("Maximum μ (eV above Fermi) [2.0]: ") or 2.0)
        mu_step = float(input("μ step (eV) [0.01]: ") or 0.01)
        
        # Store as absolute values
        transport_config["mu_range"] = (fermi_energy_ev + mu_min_rel, 
                                      fermi_energy_ev + mu_max_rel, 
                                      mu_step)
        transport_config["mu_reference"] = "fermi"
        transport_config["mu_relative_range"] = (mu_min_rel, mu_max_rel)
    else:
        # Manual entry (relative to VBM or absolute)
        if mu_option == "3":
            print("\nAbsolute chemical potential range:")
            reference = "absolute"
        else:
            print("\nChemical potential range (relative to VBM):")
            reference = "vbm"
            
        mu_min = float(input("Minimum μ (eV) [-2.0]: ") or -2.0)
        mu_max = float(input("Maximum μ (eV) [2.0]: ") or 2.0)
        mu_step = float(input("μ step (eV) [0.01]: ") or 0.01)
        transport_config["mu_range"] = (mu_min, mu_max, mu_step)
        transport_config["mu_reference"] = reference
    
    # Transport distribution function range
    print("\nTransport distribution function range:")
    tdf_min = float(input("Minimum energy (eV) [-5.0]: ") or -5.0)
    tdf_max = float(input("Maximum energy (eV) [5.0]: ") or 5.0)
    tdf_step = float(input("Energy step (eV) [0.01]: ") or 0.01)
    transport_config["tdf_range"] = (tdf_min, tdf_max, tdf_step)
    
    # Relaxation time
    tau = float(input("\nRelaxation time (fs) [10]: ") or 10)
    transport_config["relaxation_time"] = tau
    
    # Smearing
    use_smear = yes_no_prompt("\nUse smearing for metallic systems?", "no")
    if use_smear:
        smear = float(input("Smearing width (eV) [0.025]: ") or 0.025)
        transport_config["smearing"] = smear
        
        print("\nSmearing type:")
        print("0: Fermi-Dirac")
        print("1: Gaussian")
        print("2: Methfessel-Paxton")
        smear_type = int(input("Select type (0-2) [0]: ") or 0)
        transport_config["smearing_type"] = smear_type
    
    return transport_config


def configure_charge_density_calculation() -> Dict[str, Any]:
    """Configure charge density calculation (ECH3/ECHG)."""
    print("\n=== CHARGE DENSITY CONFIGURATION ===")
    
    charge_config = {}
    
    # Calculation type
    print("\nCharge density type:")
    print("1: ECH3 - 3D charge density grid")
    print("2: ECHG - 2D charge density map")
    
    calc_type = input("Select type (1-2) [1]: ").strip() or "1"
    
    if calc_type == "1":
        charge_config["type"] = "ECH3"
        
        # Number of grid points
        n_points = int(input("\nNumber of grid points per direction [100]: ") or 100)
        charge_config["n_points"] = n_points
        
        # For lower dimensional systems
        print("\nFor slabs/polymers/molecules, define non-periodic directions:")
        print("1: Use automatic SCALE")
        print("2: Define explicit RANGE")
        
        range_choice = input("Select option (1-2) [1]: ").strip() or "1"
        
        if range_choice == "2":
            charge_config["use_range"] = True
        else:
            charge_config["use_range"] = False
            scale = float(input("Scale factor for non-periodic directions [3.0]: ") or 3.0)
            charge_config["scale"] = scale
    
    else:
        charge_config["type"] = "ECHG"
        
        # Derivative order
        print("\nDerivative order:")
        print("0: Charge density")
        print("1: Gradient of charge density")
        print("2: Laplacian of charge density")
        
        deriv = int(input("Select order (0-2) [0]: ") or 0)
        charge_config["derivative_order"] = deriv
        
        # Map plane definition
        charge_config["need_map_points"] = yes_no_prompt("\nDefine charge density map plane?", "yes")
    
    return charge_config


def configure_potential_calculation() -> Dict[str, Any]:
    """Configure electrostatic potential calculation (POT3/POTC)."""
    print("\n=== ELECTROSTATIC POTENTIAL CONFIGURATION ===")
    
    potential_config = {}
    
    # Calculation type
    print("\nPotential calculation type:")
    print("1: POT3 - 3D potential grid")
    print("2: POTC - Potential at specific points")
    
    calc_type = input("Select type (1-2) [1]: ").strip() or "1"
    
    if calc_type == "1":
        potential_config["type"] = "POT3"
        
        # Grid settings
        n_points = int(input("\nNumber of grid points per direction [100]: ") or 100)
        potential_config["n_points"] = n_points
        
        itol = int(input("Bipolar expansion tolerance (5-8) [5]: ") or 5)
        potential_config["itol"] = itol
        
        # For lower dimensional systems
        print("\nFor slabs/polymers/molecules, define non-periodic directions:")
        print("1: Use automatic SCALE")
        print("2: Define explicit RANGE")
        
        range_choice = input("Select option (1-2) [1]: ").strip() or "1"
        
        if range_choice == "2":
            potential_config["use_range"] = True
        else:
            potential_config["use_range"] = False
            scale = float(input("Scale factor for non-periodic directions [3.0]: ") or 3.0)
            potential_config["scale"] = scale
    
    else:
        potential_config["type"] = "POTC"
        
        # ICA parameter
        print("\nCalculation mode:")
        print("0: Potential at specific points")
        print("1: Plane-averaged potential")
        print("2: Volume-averaged potential")
        
        ica = int(input("Select mode (0-2) [0]: ") or 0)
        potential_config["ica"] = ica
        
        if ica == 0:
            # Points specification
            print("\nDefine points:")
            print("1: At atomic positions")
            print("2: At custom points")
            print("3: Read from file")
            
            point_choice = input("Select option (1-3) [1]: ").strip() or "1"
            
            if point_choice == "1":
                potential_config["at_atoms"] = True
                potential_config["all_atoms"] = yes_no_prompt("Calculate at all atoms?", "yes")
                if not potential_config["all_atoms"]:
                    atom_list = input("Enter atom numbers: ").strip()
                    potential_config["atoms"] = [int(a) for a in atom_list.split()]
            elif point_choice == "2":
                n_points = int(input("Number of points: ") or 1)
                potential_config["n_points"] = n_points
                potential_config["custom_points"] = True
            else:
                potential_config["n_points"] = -int(input("Number of points in file: ") or 1)
                potential_config["read_file"] = True
        
        else:
            # Plane/volume averaging
            z_min = float(input("Z minimum (bohr) [0]: ") or 0)
            z_max = float(input("Z maximum (bohr) [10]: ") or 10)
            potential_config["z_range"] = (z_min, z_max)
            
            n_planes = int(input("Number of planes [100]: ") or 100)
            potential_config["n_planes"] = n_planes
            
            if ica == 2:
                thickness = float(input("Slice thickness (bohr) [0.5]: ") or 0.5)
                potential_config["slice_thickness"] = thickness
    
    return potential_config


def configure_d3_calculation(calc_type: str, out_file: Optional[str] = None, 
                           save_prompt: bool = False) -> Dict[str, Any]:
    """Main function to configure any D3 calculation type.
    
    Args:
        calc_type: Type of calculation (BAND, DOSS, etc.)
        out_file: Path to output file for context
        save_prompt: Whether to prompt user to save configuration
        
    Returns:
        Configuration dictionary
    """
    
    calc_type = calc_type.upper()
    
    if calc_type == "BAND":
        config = configure_band_calculation(out_file)
    elif calc_type == "DOSS":
        config = configure_doss_calculation(out_file)
    elif calc_type == "TRANSPORT":
        config = configure_transport_calculation(out_file)
    elif calc_type in ["CHARGE", "ECH3", "ECHG"]:
        config = configure_charge_density_calculation()
    elif calc_type in ["POTENTIAL", "POT3", "POTC"]:
        config = configure_potential_calculation()
    elif calc_type == "CHARGE+POTENTIAL":
        # Configure both charge and potential settings
        print("\n--- Charge Density Settings ---")
        charge_config = configure_charge_density_calculation()
        
        print("\n--- Electrostatic Potential Settings ---")
        potential_config = configure_potential_calculation()
        
        config = {
            "charge_config": charge_config,
            "potential_config": potential_config
        }
    else:
        raise ValueError(f"Unknown calculation type: {calc_type}")
    
    # Add calculation type to config if not present
    if "calculation_type" not in config:
        config["calculation_type"] = calc_type
    
    # Optional save prompt
    if save_prompt:
        from d3_config import save_d3_options_prompt
        save_d3_options_prompt(config)
    
    return config