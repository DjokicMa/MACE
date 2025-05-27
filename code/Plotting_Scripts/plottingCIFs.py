#!/usr/bin/python3
"""
Crystal Structure Visualizer and Analyzer

Overview:
This script provides functionality to visualize crystal structures from CIF files.
It generates 3D visualizations of crystalline materials including atoms, bonds,
and unit cells. The tool supports supercell creation, symmetry analysis,
bond calculation, and high-quality image generation.

Features:
- Loads and processes Crystallographic Information Files (CIF)
- Analyzes structural symmetry using ASE and pymatgen
- Visualizes crystal structures with pyvista in 3D
- Creates supercells for better visualization of periodicity
- Calculates interatomic bonds based on natural cutoffs
- Renders atoms with chemical element-specific colors

Usage:
1. Create an instance of CrystalVisualizer: visualizer = CrystalVisualizer()
2. Call process_cif_files() with input and output directories:
   visualizer.process_cif_files(input_directory, output_directory)

Arguments (for process_cif_files):
input_directory (str): Path to directory containing CIF files
output_directory (str): Output path for generated visualization images

Options:
- Automatically creates necessary output directory if it doesn't exist
- Generates multiple views of the structure from different angles
- Includes unit cell edges and bond representations in visualizations

Requirements:
Python 3.8+
numpy, pyvista, ase, pymatgen

Supported File Formats:
Input: CIF (Crystallographic Information Format)
Output: PNG images with transparent background option

Examples:
visualizer.process_cif_files("path/to/cifs", "path/to/output/pngs")

Notes:
- Adjust the scaling_factors in create_supercell() to change supercell size
- Modify element_colors dictionary to customize atom colors
- Bond lengths are calculated using natural cutoffs and adjusted for accuracy
"""

import os
from pathlib import Path
import numpy as np
import warnings
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
import pyvista as pv
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from typing import List, Tuple, Dict  # Import List and Tuple
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
import logging

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class CrystalVisualizer:
    def __init__(self):
        self.element_colors = {
            'H': '#FFFFFF', 'C': '#808080', 'N': '#0000FF',
            'O': '#FF0000', 'F': '#FFFF00', 'Cl': '#00FF00',
            'Br': '#A52A2A', 'I': '#940094', 'Fe': '#FFA500',
            'Au': '#FFD700', 'Ag': '#C0C0C0', 'Cu': '#FF7F50',
            'Si': '#DAA520'
        }

    def get_symmetrized_structure(self, cif_path: str):
        """Load structure using ASE and analyze symmetry."""
        try:
            # Use ASE to read the CIF file
            atoms = read(cif_path, format='cif')
            if atoms is None:
                raise ValueError(f"Could not read structure from {cif_path}")
            
            # Convert to pymatgen structure for symmetry analysis
            structure = AseAtomsAdaptor.get_structure(atoms)
            
            # Create symmetry analyzer with relaxed tolerance
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
            
            try:
                sym_structure = analyzer.get_symmetrized_structure()
                atoms = AseAtomsAdaptor.get_atoms(sym_structure)
            except Exception as e:
                print(f"Warning: Using original structure for {cif_path}: {e}")
            
            return atoms, analyzer
        except Exception as e:
            print(f"Error loading CIF file {cif_path}: {e}")
            raise

    def calculate_bonds(self, structure) -> List[Tuple]:
        """Calculate bonds between atoms."""
        try:
            # Get natural cutoffs with tolerance
            cutoffs = natural_cutoffs(structure, mult=1.15)  # Slightly increased multiplier
            
            # Create neighbor list
            nl = NeighborList(cutoffs, skin=0.3, self_interaction=False, bothways=True)
            nl.update(structure)
            
            bonds = []
            processed_pairs = set()
            
            for i in range(len(structure)):
                indices, offsets = nl.get_neighbors(i)
                cell = structure.get_cell()
                pos_i = structure.positions[i]
                
                for j, offset in zip(indices, offsets):
                    # Create unique identifier for bond
                    bond_id = tuple(sorted([i, j]))
                    
                    if bond_id not in processed_pairs:
                        pos_j = structure.positions[j] + np.dot(offset, cell)
                        
                        # Check for reasonable bond length
                        bond_length = np.linalg.norm(pos_j - pos_i)
                        if 0.5 < bond_length < 2.0:  # Adjusted for carbon-carbon bonds
                            bonds.append((pos_i, pos_j))
                            processed_pairs.add(bond_id)
            
            return bonds
        except Exception as e:
            print(f"Error in bond calculation: {e}")
            return []

    def create_bonds_visualization(self, bonds: List[Tuple]) -> pv.PolyData:
        """Create cylinder representations for bonds."""
        if not bonds:
            return None
        
        bond_mesh = pv.PolyData()
        
        for start, end in bonds:
            direction = end - start
            length = np.linalg.norm(direction)
            direction = direction / length
            
            cylinder = pv.Cylinder(
                center=(start + end) / 2,
                direction=direction,
                radius=0.1,  # Bond radius
                height=length,
                resolution=20
            )
            
            if bond_mesh.n_points == 0:
                bond_mesh = cylinder
            else:
                bond_mesh = bond_mesh.merge(cylinder)
                
        return bond_mesh

    def create_supercell(self, structure, scaling_factors: List[int]) -> np.ndarray:
        """Create a 4x4x4 supercell of the given structure."""
        try:
            supercell = structure * scaling_factors
            return supercell
        except Exception as e:
            print(f"Error in creating supercell: {e}")
            raise

    def process_cif_files(self, input_dir: str, output_dir: str):
        """Process CIF files with robust error handling."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for cif_file in Path(input_dir).glob('*.cif'):
            try:
                print(f"Processing {cif_file.name}...")
                
                # Load structure
                structure, analyzer = self.get_symmetrized_structure(str(cif_file))
                
                # Create 4x4x4 supercell
                scaling_factors = [2, 2, 2]
                supercell_structure = self.create_supercell(structure, scaling_factors)
                
                # Create plotter
                plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024])
                
                # Add atoms of supercell
                for atom in supercell_structure:
                    color = self.element_colors.get(atom.symbol, '#808080')
                    radius = covalent_radii[atom.number] * 0.5
                    
                    sphere = pv.Sphere(radius=radius, 
                                     center=atom.position, 
                                     phi_resolution=20, 
                                     theta_resolution=20)
                    
                    plotter.add_mesh(sphere,
                                   color=color,
                                   specular=0.5,
                                   specular_power=20,
                                   ambient=0.3,
                                   smooth_shading=True)
                
                # Add bonds in supercell
                bonds = self.calculate_bonds(supercell_structure)
                bonds_mesh = self.create_bonds_visualization(bonds)
                if bonds_mesh is not None:
                    plotter.add_mesh(bonds_mesh,
                                   color='lightgray',
                                   opacity=0.7,
                                   smooth_shading=True)
                
                # Add unit cell
                cell = supercell_structure.get_cell()
                vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
                vertices = np.dot(vertices, cell)
                
                # Calculate the geometric center of the unit cell
                unit_cell_center = vertices.mean(axis=0)
                
                # Create edges for unit cell (corrected version)
                edges = pv.PolyData()
                
                # Bottom face
                for i in range(4):
                    p1 = vertices[i]
                    p2 = vertices[(i + 1) % 4]
                    line = pv.Line(p1, p2)
                    if edges.n_points == 0:
                        edges = line
                    else:
                        edges = edges.merge(line)
                
                # Top face
                for i in range(4):
                    p1 = vertices[i + 4]
                    p2 = vertices[((i + 1) % 4) + 4]
                    line = pv.Line(p1, p2)
                    edges = edges.merge(line)
                
                # Vertical edges
                for i in range(4):
                    p1 = vertices[i]
                    p2 = vertices[i + 4]
                    line = pv.Line(p1, p2)
                    edges = edges.merge(line)
                
                plotter.add_mesh(edges, color='black', line_width=2, opacity=0.5)
                
                # Define camera positions relative to the unit cell center
                camera_positions = [
                    {
                        "position": [unit_cell_center[0], unit_cell_center[1], unit_cell_center[2] + 10],
                        "focal_point": unit_cell_center,
                        "up": [0, 1, 0]
                    },  # XY
                    {
                        "position": [unit_cell_center[0], unit_cell_center[1] + 10, unit_cell_center[2]],
                        "focal_point": unit_cell_center,
                        "up": [0, 0, 1]
                    },  # XZ
                    {
                        "position": [unit_cell_center[0] + 10, unit_cell_center[1], unit_cell_center[2]],
                        "focal_point": unit_cell_center,
                        "up": [0, 0, 1]
                    },  # YZ
                ]
                
                # Generate views
                for i, camera_pos in enumerate(camera_positions):
                    plotter.camera.position = camera_pos['position']  
                    plotter.camera.SetParallelProjection(True)
                    plotter.camera.focal_point = camera_pos['focal_point']
                    plotter.camera.up = camera_pos['up']
                    plotter.render()
                    
                    output_file = f"{output_dir}/{cif_file.stem}_view{i}.png"
                    plotter.screenshot(output_file, transparent_background=True)
                
                plotter.close()
                
            except Exception as e:
                print(f"Error processing {cif_file.name}: {e}")
                continue

# Usage
if __name__ == "__main__":
    visualizer = CrystalVisualizer()
    input_directory = "D:/CarbonAllotropes/CIFs/FirstOpt1/symm"
    output_directory = "D:/CarbonAllotropes/CIFs/FirstOpt1/symm/pngs"
    visualizer.process_cif_files(input_directory, output_directory)
