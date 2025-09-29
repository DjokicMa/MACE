#!/usr/bin/env python3
"""
CRYSTALOptToD3.py - Generate D3 property input files from CRYSTAL calculations

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This script generates CRYSTAL D3 input files for property calculations from completed
CRYSTAL calculations (optimization or single point). It handles:
- BAND: Electronic band structure
- DOSS: Density of states
- TRANSPORT: Boltzmann transport properties  
- CHARGE: Charge density (ECH3/ECHG)
- POTENTIAL: Electrostatic potential (POT3/POTC)

The script automatically:
- Extracts necessary information from output files
- Copies required binary files (fort.9/fort.98)
- Generates properly formatted D3 input files
- Handles all calculation types interactively
"""

import os
import shutil
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import configuration modules
try:
    # Try absolute import first (for package usage)
    from Crystal_d3.d3_interactive import (configure_d3_calculation, get_band_info_from_output,
                               parse_basis_set_info, get_atoms_and_shells,
                               generate_orbital_indices, create_doss_projections,
                               extract_shrink_from_d12)
    from Crystal_d3.d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                           extract_and_process_shrink, scale_kpoint_segments, get_seekpath_labels,
                           get_seekpath_full_kpath, get_literature_kpath_vectors, unicode_to_ascii_kpoint,
                           validate_kpoint_labels_for_crystal23)
except ImportError:
    # Fall back to relative import (for script usage)
    from d3_interactive import (configure_d3_calculation, get_band_info_from_output,
                               parse_basis_set_info, get_atoms_and_shells,
                               generate_orbital_indices, create_doss_projections,
                               extract_shrink_from_d12)
    from d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                           extract_and_process_shrink, scale_kpoint_segments, get_seekpath_labels,
                           get_seekpath_full_kpath, get_literature_kpath_vectors, unicode_to_ascii_kpoint,
                           validate_kpoint_labels_for_crystal23)
from d3_config import (save_d3_config, load_d3_config, validate_d3_config,
                      print_d3_config_summary, save_d3_options_prompt,
                      list_available_d3_configs, select_d3_config_file)
import sys
from pathlib import Path
# Add Crystal_d12 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
from d12_interactive import yes_no_prompt
from d12_constants import (SPACEGROUP_SYMBOLS, SPACEGROUP_ALTERNATIVES, 
                          SPACEGROUP_SYMBOL_TO_NUMBER)


class D3Generator:
    """Handle D3 file generation from CRYSTAL output files."""
    
    def __init__(self, input_file: str, calc_type: str, output_dir: Optional[str] = None):
        self.input_file = Path(input_file).resolve()
        self.calc_type = calc_type.upper()
        self.base_name = self.input_file.stem
        
        # Remove common suffixes to get clean base name
        for suffix in ['_opt', '_sp', '_OPT', '_SP', '.out', '.log']:
            if self.base_name.endswith(suffix):
                self.base_name = self.base_name[:-len(suffix)]
                break
        
        self.input_dir = self.input_file.parent
        self.output_dir = Path(output_dir).resolve() if output_dir else self.input_dir
        
        # Parse output file for structure info
        self.structure_info = self._parse_output_file()
    
    def _parse_output_file(self) -> Dict[str, Any]:
        """Extract structure information from output file."""
        info = {
            'dimensionality': 3,
            'n_atoms': 0,
            'n_ao': 0,
            'n_electrons': 0,
            'space_group': 1,
            'lattice_type': 'P',
            'crystal_family': 'triclinic',
            'title': self.base_name,
            'atom_elements': []  # List of element symbols for each atom
        }
        
        if not self.input_file.exists():
            print(f"Warning: Output file not found: {self.input_file}")
            return info
        
        with open(self.input_file, 'r') as f:
            content = f.read()
        
        # Dimensionality
        if 'SLAB CALCULATION' in content or 'SLAB GROUP' in content:
            info['dimensionality'] = 2
        elif 'POLYMER CALCULATION' in content:
            info['dimensionality'] = 1
        elif 'MOLECULAR CALCULATION' in content:
            info['dimensionality'] = 0
        
        # Number of atoms - use total atoms in unit cell for DOSS projections
        atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s+(\d+)', content)
        if atoms_match:
            info['n_atoms'] = int(atoms_match.group(1))
        else:
            # Fallback to asymmetric unit if unit cell count not found
            atoms_match = re.search(r'ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)', content)
            if atoms_match:
                info['n_atoms'] = int(atoms_match.group(1))
        
        # Number of AOs
        ao_match = re.search(r'NUMBER OF AO\s+(\d+)', content)
        if ao_match:
            info['n_ao'] = int(ao_match.group(1))
        
        # Number of electrons
        elec_match = re.search(r'N\. OF ELECTRONS PER UNIT CELL\s+(\d+)', content)
        if elec_match:
            info['n_electrons'] = int(elec_match.group(1))
        
        # Space group symbol and lattice type
        # Updated regex to properly capture full space group symbols like "F D 3 M" or "P -4 M 2"
        # Capture everything after the colon that starts with a letter, then strip whitespace
        sg_symbol_match = re.search(r'SPACE GROUP[^:]*:\s*([A-Z][^$\n]+)', content)
        if sg_symbol_match:
            symbol = sg_symbol_match.group(1).strip()
            # Extract lattice type (first letter)
            if symbol:
                info['lattice_type'] = symbol[0]
                
                # Try to map symbol to space group number
                # First try the symbol as-is
                if symbol in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol]
                # Try alternatives mapping (includes spaced versions)
                elif symbol in SPACEGROUP_ALTERNATIVES:
                    info['space_group'] = SPACEGROUP_ALTERNATIVES[symbol]
                # Try removing spaces
                elif symbol.replace(' ', '') in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '')]
                # Try with dashes
                elif symbol.replace(' ', '-') in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '-')]
                
        # Space group number - try different patterns
        # Pattern 1: SPACE GROUP NUMBER: 123
        sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
        if sg_match:
            info['space_group'] = int(sg_match.group(1))
        
        # Crystal family
        if 'CRYSTAL FAMILY' in content:
            family_match = re.search(r'CRYSTAL FAMILY\s+:\s+(\w+)', content)
            if family_match:
                info['crystal_family'] = family_match.group(1).lower()
        
        # Title from output
        title_match = re.search(r'^\s*\*+\s+(.+?)\s+\*+', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1).strip()
        
        # Extract atom elements from ATOM listing
        # Look for pattern like: "1 T 6 C" where 6 is atomic number and C is element
        atom_elements = []
        atom_pattern = re.compile(r'^\s*\d+\s+[TF]\s+\d+\s+([A-Z][a-z]?)\s+', re.MULTILINE)
        atom_matches = atom_pattern.findall(content)
        if atom_matches:
            info['atom_elements'] = atom_matches
        
        return info
    
    def _copy_wavefunction(self) -> bool:
        """Copy fort.9 or fort.98 file to match D3 filename."""
        # Look for wavefunction files - try various naming patterns
        wf_files = [
            'fort.9', 'fort.98', 
            f'{self.base_name}.f9', f'{self.base_name}.f98',
        ]
        
        # Add numbered suffix patterns (sp, sp2, sp3...sp99, opt, opt2...opt99, etc.)
        for suffix in ['sp', 'opt', 'freq', 'soc']:
            # Base suffix without number
            wf_files.extend([
                f'{self.base_name}_{suffix}.f9', 
                f'{self.base_name}_{suffix}.f98'
            ])
            # Numbered suffixes from 2 to 99
            for i in range(2, 100):
                wf_files.extend([
                    f'{self.base_name}_{suffix}{i}.f9', 
                    f'{self.base_name}_{suffix}{i}.f98'
                ])
        
        # Also check if the input file stem has a different name
        input_stem = self.input_file.stem
        if input_stem != self.base_name:
            wf_files.extend([f'{input_stem}.f9', f'{input_stem}.f98'])
        
        source_wf = None
        for wf_file in wf_files:
            wf_path = self.input_dir / wf_file
            if wf_path.exists():
                source_wf = wf_path
                break
        
        if not source_wf:
            print("\nWarning: No wavefunction file (fort.9/fort.98) found!")
            print("The D3 calculation will fail without the wavefunction file.")
            cont = yes_no_prompt("Continue anyway?", "no")
            return cont
        
        # Determine target filename based on calculation type
        d3_name = f"{self.base_name}_{self.calc_type.lower()}"
        target_wf = self.output_dir / f"{d3_name}.f9"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy if needed
        if source_wf != target_wf:
            print(f"\nCopying wavefunction: {source_wf.name} -> {target_wf.name}")
            shutil.copy2(source_wf, target_wf)
        
        return True
    
    def _parse_manual_projection(self, spec: str, total_shells: Dict, atom_elements: List[str]) -> Optional[str]:
        """Parse manual projection specification and convert to DOSS format.
        
        Handles formats like:
        - "1" or "1C" -> "-1 1  #1C"
        - "C all" or "C" -> element total projection
        - "C S" -> element S orbital projection
        """
        spec = spec.strip()
        if not spec:
            return None
        
        # Check if it's an atom projection (starts with a number)
        if spec[0].isdigit():
            # Extract atom number
            atom_num_match = re.match(r'^(\d+)', spec)
            if atom_num_match:
                atom_num = int(atom_num_match.group(1))
                if 0 < atom_num <= len(atom_elements):
                    element = atom_elements[atom_num - 1]
                    return f"-1 {atom_num}  #{atom_num}{element}"
                else:
                    return None
        
        # Otherwise, it's an element/orbital projection
        parts = spec.split()
        if not parts:
            return None
        
        element = parts[0].upper()
        
        # Check if element exists in the system
        if element not in total_shells:
            return None
        
        # If just element name or "element all", return element total
        if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "all"):
            # Create element total projection - combine all orbitals
            all_indices = []
            for shell_type in ["S", "P", "SP", "D", "F"]:
                if shell_type in total_shells[element]:
                    all_indices.extend(total_shells[element][shell_type])
            
            if all_indices:
                return f"{len(all_indices)} {' '.join(map(str, all_indices))}  #{element} all"
            else:
                return None
        
        # Specific orbital type
        if len(parts) == 2:
            orbital_type = parts[1].upper()
            
            # Check if orbital type exists for this element
            if orbital_type in total_shells[element] and total_shells[element][orbital_type]:
                indices = total_shells[element][orbital_type]
                return f"{len(indices)} {' '.join(map(str, indices))}  #{element} {orbital_type}"
            else:
                # Check for combined SP
                if orbital_type == "SP" and "S" in total_shells[element] and "P" in total_shells[element]:
                    # Combine S and P
                    indices = total_shells[element]["S"] + total_shells[element]["P"]
                    return f"{len(indices)} {' '.join(map(str, indices))}  #{element} SP"
                else:
                    return None
        
        return None
    
    def _write_band_d3(self, config: Dict[str, Any]) -> str:
        """Write BAND calculation D3 file."""
        lines = ["BAND"]
        
        # Determine the path info for the title
        path_info = ""
        if config.get("path_method") == "labels":
            # Get the path labels
            if "path" in config and config["path"] != "auto":
                path = config["path"]
            else:
                # Auto-detect path
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path = get_band_path_from_symmetry(space_group, lattice_type)
            # Check if path contains discontinuous segments (indicated by |)
            path_str = []
            for label in path:
                if label == "|":
                    path_str.append("|")
                else:
                    # Convert Unicode to ASCII
                    path_str.append(unicode_to_ascii_kpoint(label))
            path_info = " - " + "-".join(path_str)
        elif config.get("seekpath_full"):
            # Get SeeK-path labels if available
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            seekpath_labels = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
            if len(seekpath_labels) > 1:
                # Join labels with hyphens, preserving | for discontinuities
                path_str = []
                for label in seekpath_labels:
                    if label == "|":
                        path_str.append("|")
                    else:
                        # Convert Unicode to ASCII (though these should already be ASCII)
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
            else:
                path_info = " - SeeK-path"
        elif config.get("literature_path"):
            # Get literature path labels
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            # Literature paths use the same labels as standard paths for now
            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
            path_str = []
            for label in path_labels:
                if label == "|":
                    path_str.append("|")
                else:
                    path_str.append(unicode_to_ascii_kpoint(label))
            path_info = " - " + "-".join(path_str)
        elif config.get("path_method") == "coordinates":
            # For vector paths, check if we have stored path labels
            if "path_labels" in config:
                path_str = []
                for label in config["path_labels"]:
                    if label == "|":
                        path_str.append("|")
                    else:
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
            else:
                # Try to get standard path labels
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                path_str = []
                for label in path_labels:
                    if label == "|":
                        path_str.append("|")
                    else:
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
        elif config.get("path_method") == "manual":
            path_info = " - Manual path"
        
        # Determine the k-path source for the title
        kpath_source = config.get("kpath_source", "default")
        if kpath_source == "seekpath_inv":
            source_info = " - SeeKPath (w.I)"
        elif kpath_source == "seekpath_noinv":
            source_info = " - SeeKPath (no.I)"
        elif kpath_source == "seekpath":
            source_info = " - SeeKPath"
        elif kpath_source == "literature":
            source_info = " - Literature"
        elif kpath_source == "manual":
            source_info = " - Manual"
        elif kpath_source == "template":
            source_info = " - Template"
        elif kpath_source == "fractional":
            source_info = " - Fractional"
        else:
            source_info = " - default"
        
        # Always include material name, source, and path in title
        title = f"{self.base_name} - Band Structure{source_info}{path_info}"
        if config.get("title") and config["title"] != "Band Structure":
            # If there's a custom title, include it
            title = f"{self.base_name} - {config['title']}{source_info}{path_info}"
        lines.append(title)
        
        if config.get("path_method") == "labels":
            # High-symmetry labels - use appropriate path based on space group
            if "path" not in config or config.get("path") == "auto":
                # Auto-detect path based on space group and lattice type
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path = get_band_path_from_symmetry(space_group, lattice_type)
            else:
                path = config.get("path", ["G", "X", "M", "G"])

            # Validate k-point labels for CRYSTAL23 compatibility
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            labels_valid, validated_path = validate_kpoint_labels_for_crystal23(path, space_group, lattice_type)

            if not labels_valid:
                # Switch to coordinate mode if any labels are invalid
                print("Switching to coordinate mode due to invalid k-point labels")
                # Convert to coordinate format
                segments = []
                current_segment = []

                for item in validated_path:
                    if item == "|":
                        if len(current_segment) >= 2:
                            for i in range(len(current_segment) - 1):
                                segments.append(f"{current_segment[i]} {current_segment[i+1]}")
                        current_segment = []
                    else:
                        current_segment.append(item)

                # Handle final segment
                if len(current_segment) >= 2:
                    for i in range(len(current_segment) - 1):
                        segments.append(f"{current_segment[i]} {current_segment[i+1]}")

                # Use coordinate format
                n_segments = len(segments)
                shrink = config.get("shrink", 16)  # Use non-zero shrink for coordinates
                n_points = config.get("n_points", 1000)

                first_band = config.get("first_band", 1)
                last_band = config.get("last_band")
                if last_band is None:
                    last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100

                plot = 1 if config.get("plot", True) else 0
                print_eig = 1 if config.get("print_eigenvalues", False) else 0

                lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")

                for segment in segments:
                    lines.append(segment)
            else:
                # Use original label format
                # Count actual segments (excluding | markers)
                n_segments = 0
                i = 0
                while i < len(validated_path) - 1:
                    if validated_path[i] == "|":
                        i += 1
                        continue
                    j = i + 1
                    while j < len(validated_path) and validated_path[j] == "|":
                        j += 1
                    if j < len(validated_path):
                        n_segments += 1
                        i = j
                    else:
                        break

                shrink = 0
                n_points = config.get("n_points", 1000)

                # Use band range from config
                first_band = config.get("first_band", 1)
                last_band = config.get("last_band")
                if last_band is None:
                    last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100

                plot = 1 if config.get("plot", True) else 0
                print_eig = 1 if config.get("print_eigenvalues", False) else 0

                lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")

                # Add path segments - handle discontinuous paths
                current_segment = []

                for label in validated_path:
                    if label == "|":
                        # End current segment
                        if len(current_segment) >= 2:
                            # Create segments from consecutive points
                            for i in range(len(current_segment) - 1):
                                lines.append(f"{current_segment[i]} {current_segment[i+1]}")
                        current_segment = []
                    else:
                        current_segment.append(label)

                # Handle final segment
                if len(current_segment) >= 2:
                    for i in range(len(current_segment) - 1):
                        lines.append(f"{current_segment[i]} {current_segment[i+1]}")
        
        elif config.get("path_method") == "manual":
            # Manual mixed path
            segments = config.get("manual_segments", [])
            n_segments = len(segments)
            shrink = config.get("shrink", 0)
            n_points = config.get("n_points", 1000)
            
            first_band = config.get("first_band", 1) 
            last_band = config.get("last_band")
            if last_band is None:
                last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100
            
            plot = 1 if config.get("plot", True) else 0
            print_eig = 1 if config.get("print_eigenvalues", False) else 0
            
            lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")
            
            # Add manual segments as-is
            for segment in segments:
                lines.append(segment)
                
        else:
            # Fractional coordinates
            segments = config.get("segments", [])
            n_segments = len(segments)
            shrink = config.get("shrink", 16)
            n_points = config.get("n_points", 1000)
            
            first_band = config.get("first_band", 1) 
            last_band = config.get("last_band")
            if last_band is None:
                last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100
            
            plot = 1 if config.get("plot", True) else 0
            print_eig = 1 if config.get("print_eigenvalues", False) else 0
            
            lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")
            
            # Add coordinate segments
            for segment in segments:
                lines.append(f"{segment[0]} {segment[1]} {segment[2]}  {segment[3]} {segment[4]} {segment[5]}")
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _write_doss_d3(self, config: Dict[str, Any]) -> str:
        """Write DOSS calculation D3 file with NEWK and orbital projections."""
        lines = []
        
        # First, we need NEWK
        lines.append("NEWK")
        
        # Extract SHRINK parameters from the corresponding D12 file
        d12_file = self.input_file.with_suffix('.d12')
        if not d12_file.exists():
            # Try to find D12 file in same directory
            d12_files = list(self.input_dir.glob(f"{self.base_name}*.d12"))
            if d12_files:
                d12_file = d12_files[0]
        
        if d12_file.exists():
            # Use for_doss=True to ensure ISP >= 2*IS
            dimensionality = self.structure_info.get('dimensionality', 3)
            shrink_lines = extract_shrink_from_d12(str(d12_file), for_doss=True, dimensionality=dimensionality)
            for line in shrink_lines:
                lines.append(line.strip())
        else:
            # Default fallback based on dimensionality
            dimensionality = self.structure_info.get('dimensionality', 3)
            is_val = 8 if dimensionality == 3 else 12
            isp = 2 * is_val
            lines.append(f"{is_val} {isp}")
        
        lines.append("1 0")  # IFE=1, IPLO=0
        
        # Now DOSS section
        lines.append("DOSS")
        
        # Debug output
        print(f"\nDOSS configuration:")
        print(f"  project_orbital_types: {config.get('project_orbital_types', True)}")
        print(f"  n_atoms: {self.structure_info.get('n_atoms', 0)}")
        print(f"  n_ao: {self.structure_info.get('n_ao', 0)}")
        
        # Determine projection type
        if config.get("manual_projections"):
            # Manual projections (option 6) - handle first
            manual_specs = config.get("manual_projection_specs", [])
            projections = []
            
            # Parse basis set info from output file for validation
            data_list, n_ao = parse_basis_set_info(str(self.input_file))
            total_shells = {}
            
            if data_list:
                # Get atoms and shells
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                # Generate orbital indices
                total_shells = generate_orbital_indices(atoms, atoms_shells)
            
            # Get atom information
            atom_elements = self.structure_info.get('atom_elements', [])
            
            # Parse each manual specification
            for spec in manual_specs:
                parsed_proj = self._parse_manual_projection(spec, total_shells, atom_elements)
                if parsed_proj:
                    projections.append(parsed_proj)
                else:
                    print(f"Warning: Skipping invalid projection specification: '{spec}'")
            
            # Update npro to match actual valid projections
            npro = len(projections)
            if npro == 0:
                print("Error: No valid projections found. Using total DOS only.")
                npro = 0
                projections = []
        elif config.get("project_orbital_types", True):
            # Generate orbital projections by element and type (default behavior)
            # Parse basis set info from output file
            data_list, n_ao = parse_basis_set_info(str(self.input_file))
            
            if data_list:
                # Get atoms and shells
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                
                # Generate orbital indices
                total_shells = generate_orbital_indices(atoms, atoms_shells)
                
                # Create projection lines
                element_only = config.get("element_only", False)
                include_totals = config.get("include_element_totals", True)
                projections = create_doss_projections(total_shells, element_only, include_totals)
                npro = len(projections)
                print(f"  Generated {npro} orbital projections")
            else:
                # Fallback when detailed basis set parsing fails
                print(f"Warning: Could not parse detailed basis set information from {self.input_file}")
                print("  Cannot create orbital-resolved projections without basis set info")
                print("  Using total DOS only")
                npro = 0
                projections = []
        elif config.get("project_all_atoms") or config.get("project_atoms"):
            # Manual atom projections
            atom_elements = self.structure_info.get('atom_elements', [])
            if config.get("project_all_atoms"):
                npro = self.structure_info['n_atoms']
                projections = []
                for i in range(npro):
                    if i < len(atom_elements):
                        element = atom_elements[i]
                        projections.append(f"-1 {i+1}  #{i+1}{element}")
                    else:
                        projections.append(f"-1 {i+1}  #Atom{i+1}")
            else:
                atoms = config.get("project_atoms", [])
                npro = len(atoms)
                projections = []
                for a in atoms:
                    atom_idx = a - 1  # atoms are 1-indexed
                    if atom_idx < len(atom_elements):
                        element = atom_elements[atom_idx]
                        projections.append(f"-1 {a}  #{a}{element}")
                    else:
                        projections.append(f"-1 {a}  #Atom{a}")
        else:
            # Total DOS only (option 1)
            npro = 0
            projections = []
        
        # Other parameters - use old alldos.py defaults
        n_points = config.get("n_points", 10000)  # Default from alldos.py
        
        # Handle band/energy range
        if config.get("energy_window"):
            # Energy window specified - negative band indices trigger BMI/BMA input
            first_band = -1
            last_band = -1
            bmi, bma = config["energy_window"]
        elif config.get("band_range"):
            # Specific band range
            first_band, last_band = config["band_range"]
            bmi, bma = None, None
        else:
            # All bands (default) - use actual band indices
            first_band = 1
            # Use n_ao as approximation for last band if available
            if self.structure_info.get('n_ao', 0) > 0:
                last_band = self.structure_info['n_ao']
            else:
                last_band = 200  # Safe default
            bmi, bma = None, None
            
        # IPLO values: 0=no output, 1=fort.25, 2=DOSS.DAT
        # Default to 2 (DOSS.DAT) as it's more commonly used
        output_format = config.get("output_format", 2)
        npol = config.get("npol", 14)  # Default from alldos.py
        
        # nprint: 0=no integrated DOS, 1=print integrated DOS
        if config.get("print_integrated", False):
            nprint = 1
        else:
            nprint = 0
        
        # First line after DOSS
        lines.append(f"{npro} {n_points} {first_band} {last_band} {output_format} {npol} {nprint}")
        
        # Add BMI/BMA if using energy window (only when first_band and last_band are negative)
        if first_band < 0 and last_band < 0 and bmi is not None:
            lines.append(f"{bmi} {bma}")
        
        # Add projection specifications
        for proj in projections:
            lines.append(proj)
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _write_transport_d3(self, config: Dict[str, Any]) -> str:
        """Write BOLTZTRA transport calculation D3 file."""
        lines = ["BOLTZTRA"]
        
        # Temperature range
        t_min, t_max, t_step = config.get("temperature_range", (100, 800, 50))
        lines.append("TRANGE")
        lines.append(f"{t_min} {t_max} {t_step}")
        
        # Chemical potential range
        mu_min, mu_max, mu_step = config.get("mu_range", (-2.0, 2.0, 0.01))
        lines.append("MURANGE")
        lines.append(f"{mu_min} {mu_max} {mu_step}")
        
        # Transport distribution function range
        tdf_min, tdf_max, tdf_step = config.get("tdf_range", (-5.0, 5.0, 0.01))
        lines.append("TDFRANGE")
        lines.append(f"{tdf_min} {tdf_max} {tdf_step}")
        
        # Optional parameters
        tau = config.get("relaxation_time", 10)
        if tau != 10:
            lines.append("RELAXTIM")
            lines.append(str(tau))
        
        smear = config.get("smearing", 0.0)
        if smear > 0:
            lines.append("SMEAR")
            lines.append(str(smear))
            
            smear_type = config.get("smearing_type", 0)
            if smear_type != 0:
                lines.append("SMEARTYP")
                lines.append(str(smear_type))
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _write_charge_d3(self, config: Dict[str, Any]) -> str:
        """Write charge density calculation D3 file."""
        lines = []
        
        if config.get("type") == "ECH3":
            lines.append("ECH3")
            lines.append(str(config.get("n_points", 100)))
            
            # Only for non-3D systems do we need SCALE or RANGE
            if self.structure_info['dimensionality'] < 3:
                if config.get("use_range", False):
                    lines.append("RANGE")
                    # Would need to get ranges interactively
                    print("\nDefine explicit ranges for non-periodic directions:")
                    if self.structure_info['dimensionality'] == 2:
                        z_min = float(input("Z min (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{z_min}")
                        lines.append(f"{z_max}")
                    elif self.structure_info['dimensionality'] == 1:
                        y_min = float(input("Y min (bohr): "))
                        z_min = float(input("Z min (bohr): "))
                        y_max = float(input("Y max (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{y_min} {z_min}")
                        lines.append(f"{y_max} {z_max}")
                    else:  # Molecule
                        x_min = float(input("X min (bohr): "))
                        y_min = float(input("Y min (bohr): "))
                        z_min = float(input("Z min (bohr): "))
                        x_max = float(input("X max (bohr): "))
                        y_max = float(input("Y max (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{x_min} {y_min} {z_min}")
                        lines.append(f"{x_max} {y_max} {z_max}")
                else:
                    lines.append("SCALE")
                    scale = config.get("scale", 3)
                    if self.structure_info['dimensionality'] == 2:
                        # 2D: only ZSCALE
                        lines.append(str(int(scale)))
                    elif self.structure_info['dimensionality'] == 1:
                        # 1D: YSCALE,ZSCALE
                        lines.append(f"{int(scale)} {int(scale)}")
                    else:  # 0D
                        # 0D: XSCALE,YSCALE,ZSCALE
                        lines.append(f"{int(scale)} {int(scale)} {int(scale)}")
        
        else:  # ECHG
            lines.append("ECHG")
            lines.append(str(config.get("derivative_order", 0)))
            
            # Need MAPNET input
            if config.get("need_map_points", False):
                print("\nDefine map plane by three points A, B, C")
                print("Enter coordinates in fractional (crystal) or Cartesian (Angstrom) units")
                
                coord_type = input("Coordinate type (F)ractional or (C)artesian [F]: ").upper() or "F"
                
                points = []
                for point in ["A", "B", "C"]:
                    coords = input(f"Point {point} (x y z): ").strip().split()
                    points.append([float(x) for x in coords])
                
                # MAPNET section
                lines.append("MAPNET")
                if coord_type == "C":
                    lines.append("CARTESIAN") 
                    lines.append("ANGSTROM")
                
                # Points
                for i, point in enumerate(points):
                    lines.append(f"{point[0]} {point[1]} {point[2]}")
                
                # Number of points
                n_points_ab = int(input("Number of points along AB [50]: ") or 50)
                n_points_bc = int(input("Number of points along BC [50]: ") or 50)
                lines.append(f"{n_points_ab} {n_points_bc}")
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _write_potential_d3(self, config: Dict[str, Any]) -> str:
        """Write electrostatic potential calculation D3 file."""
        lines = []
        
        if config.get("type") == "POT3":
            lines.append("POT3")
            lines.append(str(config.get("n_points", 100)))
            lines.append(str(config.get("itol", 5)))
            
            # Only for non-3D systems do we need SCALE or RANGE
            if self.structure_info['dimensionality'] < 3:
                if config.get("use_range", False):
                    lines.append("RANGE")
                    # Would need to get ranges interactively (same as charge)
                    print("\nDefine explicit ranges for non-periodic directions:")
                    if self.structure_info['dimensionality'] == 2:
                        z_min = float(input("Z min (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{z_min}")
                        lines.append(f"{z_max}")
                    elif self.structure_info['dimensionality'] == 1:
                        y_min = float(input("Y min (bohr): "))
                        z_min = float(input("Z min (bohr): "))
                        y_max = float(input("Y max (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{y_min} {z_min}")
                        lines.append(f"{y_max} {z_max}")
                    else:  # 0D
                        x_min = float(input("X min (bohr): "))
                        y_min = float(input("Y min (bohr): "))
                        z_min = float(input("Z min (bohr): "))
                        x_max = float(input("X max (bohr): "))
                        y_max = float(input("Y max (bohr): "))
                        z_max = float(input("Z max (bohr): "))
                        lines.append(f"{x_min} {y_min} {z_min}")
                        lines.append(f"{x_max} {y_max} {z_max}")
                else:
                    lines.append("SCALE")
                    scale = config.get("scale", 3)
                    if self.structure_info['dimensionality'] == 2:
                        # 2D: only ZSCALE
                        lines.append(str(int(scale)))
                    elif self.structure_info['dimensionality'] == 1:
                        # 1D: YSCALE,ZSCALE
                        lines.append(f"{int(scale)} {int(scale)}")
                    else:  # 0D
                        # 0D: XSCALE,YSCALE,ZSCALE
                        lines.append(f"{int(scale)} {int(scale)} {int(scale)}")
        
        else:  # POTC
            lines.append("POTC")
            ica = config.get("ica", 0)
            
            if ica == 0:  # At points
                if config.get("at_atoms", False):
                    npu = 0
                    ipa = 0 if config.get("all_atoms", True) else 1
                else:
                    npu = config.get("n_points", 0)
                    if config.get("read_file", False):
                        npu = -npu
                    ipa = 0
                
                lines.append(f"{ica} {npu} {ipa}")
                
                # Add custom points if needed
                if config.get("custom_points", False) and npu > 0:
                    print("\nEnter point coordinates (Cartesian, bohr):")
                    for i in range(npu):
                        coords = input(f"Point {i+1} (x y z): ").strip()
                        lines.append(coords)
            
            else:  # Plane/volume averaged
                z_min, z_max = config.get("z_range", (0, 10))
                n_planes = config.get("n_planes", 100)
                
                lines.append(f"{ica} {n_planes} 0")
                lines.append(f"{z_min} {z_max}")
                
                if ica == 2:  # Volume averaged
                    thickness = config.get("slice_thickness", 0.5)
                    lines.append(str(thickness))
        
        lines.append("END")
        return '\n'.join(lines)
    
    def generate_d3(self, shared_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate the D3 file based on calculation type.
        
        Returns:
            Configuration dictionary if successful, None otherwise
        """
        print(f"\n=== Generating {self.calc_type} D3 file ===")
        print(f"Input file: {self.input_file}")
        print(f"Base name: {self.base_name}")
        
        # Check if wavefunction exists and copy it
        # ALL calculation types need the wavefunction file
        if not self._copy_wavefunction():
            return None
        
        # Configure calculation - use shared config if provided
        if shared_config:
            config = shared_config.copy()  # Make a copy to allow modifications
            
            # For BAND calculations with automatic path, recalculate path for each material
            if self.calc_type == "BAND" and config.get("auto_path", False):
                # Re-extract space group for this specific material
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                
                if config.get("path_method") == "labels":
                    # Get appropriate band path for THIS material's symmetry
                    from d3_kpoints import get_band_path_from_symmetry
                    config["path"] = get_band_path_from_symmetry(space_group, lattice_type)
                    config["kpath_source"] = "default"  # Label-based paths use default source
                    print(f"  Using band path for space group {space_group}: {' → '.join(config['path'])}")
                elif config.get("path_method") == "coordinates":
                    # Convert labels to coordinates for THIS material's symmetry
                    from d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                                           get_seekpath_full_kpath, get_literature_kpath_vectors)
                    
                    if config.get("seekpath_full", False):
                        # Use SeeK-path full path
                        result = get_seekpath_full_kpath(space_group, lattice_type, str(self.input_file))
                        if result:
                            frac_segments, kpath_info = result
                            # Extract and process shrink factor
                            shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                               self.input_dir, config)
                            
                            # Scale fractional coordinates
                            coord_segments = scale_kpoint_segments(frac_segments, shrink)
                            config["segments"] = coord_segments
                            config["shrink"] = shrink
                            # Store SeeK-path labels
                            config["path_labels"] = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
                            # Store k-path source info
                            if kpath_info.get("source") == "literature":
                                config["kpath_source"] = "literature"
                            elif kpath_info.get("source") == "default":
                                config["kpath_source"] = "default"
                            elif kpath_info.get("has_inversion"):
                                config["kpath_source"] = "seekpath_inv"
                            else:
                                config["kpath_source"] = "seekpath_noinv"
                            print(f"  Using SeeK-path full k-path for space group {space_group} with shrink={shrink}")
                        else:
                            # Fallback - try literature path first
                            frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                            if frac_segments:
                                # Use literature path
                                shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                                   self.input_dir, config)
                                coord_segments = scale_kpoint_segments(frac_segments, shrink)
                                config["segments"] = coord_segments
                                config["shrink"] = shrink
                                config["path_labels"] = get_band_path_from_symmetry(space_group, lattice_type)
                                config["kpath_source"] = "literature"
                                print(f"  SeeK-path not available, using literature k-path for space group {space_group}")
                            else:
                                # Final fallback to standard path
                                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                                coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                                config["segments"] = coord_segments
                                config["path_labels"] = path_labels  # Store labels for title
                                config["kpath_source"] = "default"
                                print(f"  SeeK-path not available, using standard k-point vectors for space group {space_group}")
                    
                    elif config.get("literature_path", False):
                        # Use literature path
                        config["kpath_source"] = "literature"
                        frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                        if frac_segments:
                            # Extract and process shrink factor
                            shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                               self.input_dir, config)
                            
                            # Scale fractional coordinates
                            coord_segments = scale_kpoint_segments(frac_segments, shrink)
                            config["segments"] = coord_segments
                            config["shrink"] = shrink
                            # Store path labels (literature paths use standard labels for now)
                            config["path_labels"] = get_band_path_from_symmetry(space_group, lattice_type)
                            print(f"  Using literature k-path for space group {space_group} with shrink={shrink}")
                        else:
                            # Fallback
                            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                            coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                            config["segments"] = coord_segments
                            config["path_labels"] = path_labels  # Store labels for title
                            print(f"  Literature path not available, using standard k-point vectors for space group {space_group}")
                    
                    else:
                        # Standard path - need to scale coordinates
                        config["kpath_source"] = "default"
                        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                        frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                        
                        # Extract and process shrink factor
                        shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                           self.input_dir, config)
                        
                        # Scale fractional coordinates by shrink factor
                        coord_segments = scale_kpoint_segments(frac_segments, shrink)
                        
                        config["segments"] = coord_segments
                        config["path_labels"] = path_labels  # Store labels for title
                        config["shrink"] = shrink
                        print(f"  Using k-point vectors for space group {space_group} with shrink={shrink}")
            
            
            # For TRANSPORT calculations with auto Fermi reference, recalculate for each material
            if self.calc_type == "TRANSPORT" and config.get("mu_reference") == "fermi":
                # Extract Fermi energy for this material
                try:
                    from Crystal_d3.d3_interactive import get_band_info_from_output
                except ImportError:
                    from d3_interactive import get_band_info_from_output
                band_info = get_band_info_from_output(str(self.input_file))
                
                if band_info.get('fermi_energy') is not None:
                    # Convert from Hartree to eV
                    fermi_energy_ev = band_info['fermi_energy'] * 27.211386
                    
                    # Get relative range from config
                    if config.get("mu_range_type") == "auto_fermi":
                        # Config was cleaned, mu_range contains relative values
                        mu_min_rel, mu_max_rel, mu_step = config["mu_range"]
                    else:
                        # Use default relative range
                        mu_min_rel, mu_max_rel = -2.0, 2.0
                        mu_step = config.get("mu_range", (0, 0, 0.01))[2]
                    
                    # Calculate absolute values for this material
                    mu_min_abs = fermi_energy_ev + mu_min_rel
                    mu_max_abs = fermi_energy_ev + mu_max_rel
                    config["mu_range"] = (mu_min_abs, mu_max_abs, mu_step)
                    print(f"  Using Fermi energy {fermi_energy_ev:.3f} eV")
                    print(f"  Chemical potential range: {mu_min_abs:.3f} to {mu_max_abs:.3f} eV")
                else:
                    print("  Warning: Could not extract Fermi energy, using manual values")
        else:
            config = configure_d3_calculation(self.calc_type, str(self.input_file))
        
        # Generate D3 content
        if self.calc_type == "BAND":
            d3_content = self._write_band_d3(config)
        elif self.calc_type == "DOSS":
            d3_content = self._write_doss_d3(config)
        elif self.calc_type == "TRANSPORT":
            d3_content = self._write_transport_d3(config)
        elif self.calc_type in ["CHARGE", "ECH3", "ECHG"]:
            d3_content = self._write_charge_d3(config)
        elif self.calc_type in ["POTENTIAL", "POT3", "POTC"]:
            d3_content = self._write_potential_d3(config)
        elif self.calc_type == "CHARGE+POTENTIAL":
            # Combined ECH3+POT3 calculation
            # The config already contains charge_config and potential_config from configure_d3_calculation
            if "charge_config" in config and "potential_config" in config:
                # Using config (either from shared_config or just configured)
                charge_config = config["charge_config"]
                potential_config = config["potential_config"]
            else:
                # This shouldn't happen if configure_d3_calculation worked correctly
                print("\nError: CHARGE+POTENTIAL configuration missing charge_config or potential_config")
                return None
            
            # Write combined file
            d3_content = self._write_charge_d3(charge_config).rstrip('\nEND')
            d3_content += "\n" + self._write_potential_d3(potential_config)
            
        else:
            print(f"Error: Unknown calculation type {self.calc_type}")
            return False
        
        # Write D3 file
        d3_filename = f"{self.base_name}_{self.calc_type.lower()}.d3"
        d3_path = self.output_dir / d3_filename
        
        with open(d3_path, 'w') as f:
            f.write(d3_content)
        
        print(f"\n✓ D3 file written: {d3_path}")
        
        # Additional instructions
        print("\nTo run the calculation:")
        print(f"1. Make sure the wavefunction file is present: {self.base_name}_{self.calc_type.lower()}.f9")
        print(f"2. Submit the job with the D3 file: {d3_filename}")
        
        if self.calc_type == "BAND":
            print("\nOutput files:")
            print("  - BAND.DAT: Band structure data")
            print("  - fort.25: Band structure for plotting")
        elif self.calc_type == "DOSS":
            print("\nOutput files:")
            print("  - DOSS.DAT: Density of states data")
            print("  - fort.25: DOS for plotting (if requested)")
        elif self.calc_type == "TRANSPORT":
            print("\nOutput files:")
            print("  - SIGMA.DAT: Electrical conductivity")
            print("  - SEEBECK.DAT: Seebeck coefficient")
            print("  - SIGMAS.DAT: σS product")
            print("  - KAPPA.DAT: Thermal conductivity")
            print("  - TDF.DAT: Transport distribution function")
        
        # Return the configuration for potential saving
        return config


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate CRYSTAL D3 property calculation input files"
    )
    
    parser.add_argument(
        "--input", "-i",
        help="CRYSTAL output file (.out or .log)"
    )
    parser.add_argument(
        "--calc-type", "-t", "--calc_type",
        dest="calc_type",
        choices=["BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "CHARGE+POTENTIAL"],
        help="Type of property calculation"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all .out files in current directory"
    )
    parser.add_argument(
        "--shared-settings", "-s",
        action="store_true",
        help="Use shared settings for all files in batch mode"
    )
    parser.add_argument(
        "--config-file", "-c",
        help="Load D3 configuration from JSON file"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save configuration to JSON file after interactive setup"
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default=None,
        help="Filename for saving configuration (suppresses interactive prompt)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available D3 configuration files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for generated D3 files (default: same as input file)"
    )
    
    args = parser.parse_args()
    
    # Handle --list-configs
    if args.list_configs:
        configs = list_available_d3_configs()
        if configs:
            print("\nAvailable D3 configuration files:")
            for config in configs:
                print(f"  - {config}")
        else:
            print("\nNo D3 configuration files found in current directory.")
        return
    
    # If no input file and not batch mode, check if there are .out files in current dir
    if not args.input and not args.batch:
        out_files = list(Path.cwd().glob("*.out"))
        if out_files:
            # Automatically switch to batch mode if .out files found
            args.batch = True
            print(f"\nFound {len(out_files)} output file(s) in current directory:")
            for f in sorted(out_files)[:5]:  # Show first 5
                print(f"  - {f.name}")
            if len(out_files) > 5:
                print(f"  ... and {len(out_files) - 5} more")
    
    if args.batch:
        # Batch mode
        if hasattr(args, 'batch_dir') and args.batch_dir:
            # Use the directory specified by the user
            out_files = list(args.batch_dir.glob("*.out"))
            batch_dir = args.batch_dir
        else:
            # Use current working directory
            out_files = list(Path.cwd().glob("*.out"))
            batch_dir = Path.cwd()
            
        if not out_files:
            print(f"No .out files found in {batch_dir}")
            return
        
        print(f"Found {len(out_files)} output files")
        
        # Get calculation type
        if not args.calc_type:
            print("\nSelect calculation type for all files:")
            print("1: BAND - Electronic band structure")
            print("2: DOSS - Density of states")
            print("3: TRANSPORT - Transport properties")
            print("4: CHARGE - Charge density")
            print("5: POTENTIAL - Electrostatic potential")
            print("6: CHARGE+POTENTIAL - Combined calculation")
            
            choice = input("\nSelect type (1-6): ").strip()
            calc_types = {
                "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL"
            }
            calc_type = calc_types.get(choice, "BAND")
        else:
            calc_type = args.calc_type
        
        # Ask about shared settings if not specified and more than one file
        if not args.shared_settings and len(out_files) > 1:
            print("\nMultiple files detected. Use shared settings for all files?")
            use_shared = yes_no_prompt("Use same configuration for all files?", "yes")
            args.shared_settings = use_shared
        
        # Configure shared settings if requested
        shared_config = None
        
        # Load from config file if specified
        if args.config_file:
            shared_config = load_d3_config(args.config_file)
            if shared_config:
                print_d3_config_summary(shared_config)
                # Validate configuration
                is_valid, errors = validate_d3_config(shared_config)
                if not is_valid:
                    print("\nConfiguration validation errors:")
                    for error in errors:
                        print(f"  - {error}")
                    return
                # Override calc_type from config if present
                if "calculation_type" in shared_config:
                    calc_type = shared_config["calculation_type"]
            else:
                print(f"Failed to load configuration from {args.config_file}")
                return
        elif args.shared_settings:
            print("\n=== Configuring shared settings for all files ===")
            # Use first file as reference for getting structure info
            first_file = str(out_files[0])
            shared_config = configure_d3_calculation(calc_type, first_file)
            print("\n✓ Shared settings configured")
            
            # Option to save configuration
            if args.save_config or yes_no_prompt("\nSave this configuration for future use?", "no"):
                if args.options_file:
                    # Use specified filename without prompting
                    save_d3_config(shared_config, args.options_file)
                    print(f"Configuration saved to {args.options_file}")
                else:
                    save_d3_options_prompt(shared_config, skip_prompt=True)
        
        # Process each file
        for out_file in out_files:
            print(f"\n{'='*60}")
            print(f"Processing: {out_file}")
            generator = D3Generator(str(out_file), calc_type, args.output_dir)
            generator.generate_d3(shared_config)
    
    else:
        # Interactive mode
        if args.input:
            input_file = args.input
        else:
            input_file = input("Enter CRYSTAL output file path: ").strip()
        
        input_path = Path(input_file)
        
        # Check if the input exists
        if not input_path.exists():
            print(f"\nError: File or directory not found: {input_file}")
            return
        
        # Check if it's a directory
        if input_path.is_dir():
            # Look for .out files in the directory
            out_files = list(input_path.glob('*.out'))
            if not out_files:
                print(f"\nError: No CRYSTAL output files (.out) found in directory: {input_path}")
                print("Please specify a CRYSTAL output file (e.g., material.out) or a directory containing .out files.")
                return
            else:
                # Process all files in the directory
                print(f"\nFound {len(out_files)} output file(s) in {input_path.name}:")
                for f in sorted(out_files)[:10]:  # Show first 10
                    print(f"  - {f.name}")
                if len(out_files) > 10:
                    print(f"  ... and {len(out_files) - 10} more")
                
                # Get calculation type
                if not args.calc_type:
                    print("\nSelect calculation type for all files:")
                    print("1: BAND - Electronic band structure")
                    print("2: DOSS - Density of states")
                    print("3: TRANSPORT - Transport properties")
                    print("4: CHARGE - Charge density")
                    print("5: POTENTIAL - Electrostatic potential")
                    print("6: CHARGE+POTENTIAL - Combined calculation")
                    
                    choice = input("\nSelect type (1-6): ").strip()
                    calc_types = {
                        "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                        "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL"
                    }
                    calc_type = calc_types.get(choice, "BAND")
                else:
                    calc_type = args.calc_type
                
                # Ask about shared settings if more than one file
                shared_config = None
                if len(out_files) > 1:
                    print("\nMultiple files detected. Use shared settings for all files?")
                    use_shared = yes_no_prompt("Use same configuration for all files?", "yes")
                    
                    if use_shared:
                        print("\n=== Configuring shared settings for all files ===")
                        # Use first file as reference for getting structure info
                        first_file = str(out_files[0])
                        temp_generator = D3Generator(first_file, calc_type, args.output_dir)
                        shared_config = temp_generator.generate_d3()
                        if shared_config:
                            print("\n✓ Shared settings configured")
                            
                            # Option to save configuration
                            if yes_no_prompt("\nSave this configuration for future use?", "no"):
                                save_d3_options_prompt(shared_config, skip_prompt=True)
                
                # Process all files
                print(f"\nProcessing {len(out_files)} files...")
                for out_file in sorted(out_files):
                    print(f"\n{'='*60}")
                    print(f"Processing: {out_file.name}")
                    generator = D3Generator(str(out_file), calc_type, args.output_dir)
                    
                    if shared_config:
                        generator.generate_d3(shared_config)
                    else:
                        # Interactive configuration for each file
                        config = generator.generate_d3()
                
                print(f"\n{'='*60}")
                print(f"✓ Completed processing {len(out_files)} files")
                return
        elif not input_path.is_file():
            print(f"\nError: {input_path} is not a valid file.")
            print("Please specify a CRYSTAL output file (e.g., material.out).")
            return
        elif not input_path.suffix.lower() in ['.out', '.log']:
            print(f"\nWarning: {input_path.name} may not be a CRYSTAL output file.")
            print("CRYSTAL output files typically have .out or .log extensions.")
        
        # If we're in batch mode (directory was provided), skip single-file processing
        if not (hasattr(args, 'batch') and args.batch and input_file is None):
            # Single file processing
            if not args.calc_type:
                print("\nSelect calculation type:")
                print("1: BAND - Electronic band structure") 
                print("2: DOSS - Density of states")
                print("3: TRANSPORT - Transport properties (Boltzmann)")
                print("4: CHARGE - Charge density (3D or 2D)")
                print("5: POTENTIAL - Electrostatic potential")
                print("6: CHARGE+POTENTIAL - Combined calculation")
                
                choice = input("\nSelect type (1-6): ").strip()
                calc_types = {
                    "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                    "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL"
                }
                calc_type = calc_types.get(choice, "BAND")
            else:
                calc_type = args.calc_type
            
            # Load configuration if specified
            config = None
            if args.config_file:
                config = load_d3_config(args.config_file)
                if config:
                    print_d3_config_summary(config)
                    # Validate configuration
                    is_valid, errors = validate_d3_config(config)
                    if not is_valid:
                        print("\nConfiguration validation errors:")
                        for error in errors:
                            print(f"  - {error}")
                        return
                    # Override calc_type from config if present
                    if "calculation_type" in config:
                        calc_type = config["calculation_type"]
                else:
                    print(f"Failed to load configuration from {args.config_file}")
                    return
            
            generator = D3Generator(input_file, calc_type, args.output_dir)
            
            # Generate D3 file with config if available
            if config:
                generator.generate_d3(config)
            else:
                # Interactive configuration
                config = generator.generate_d3()
                
                # Option to save configuration
                if args.save_config and config:
                    if args.options_file:
                        # Use specified filename without prompting
                        save_d3_config(config, args.options_file)
                        print(f"Configuration saved to {args.options_file}")
                    else:
                        save_d3_options_prompt(config, skip_prompt=True)


if __name__ == "__main__":
    main()