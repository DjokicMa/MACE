#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRYSTAL Output to CIF Converter
------------------------------
Convert CRYSTAL17/23 output files to CIF format with support for all
dimensionalities (3D, 2D, 1D, 0D) and calculation types (OPT, SP, FREQ).

DESCRIPTION:
    This script extracts optimized geometries from CRYSTAL17/23 output files
    and converts them to CIF format. It intelligently selects the best geometry
    based on calculation type and handles all dimensionalities properly.

USAGE:
    1. Single file conversion:
       python CrystalOutToCif.py material.out

    2. Batch conversion (current directory):
       python CrystalOutToCif.py .

    3. Batch conversion (specific directory):
       python CrystalOutToCif.py /path/to/calculations/

    4. With custom options:
       python CrystalOutToCif.py . --output-dir cifs/ --include-metadata

AUTHOR:
    Marcus Djokic
    Institution: Michigan State University, Mendoza Group
    Date: December 2024
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import from existing MACE infrastructure
from d12_parsers import CrystalOutputParser
from d12_constants import ATOMIC_NUMBER_TO_SYMBOL


class CrystalOutToCifConverter:
    """Main converter class for CRYSTAL output to CIF conversion"""

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the converter with options.

        Args:
            options: Dictionary of conversion options
        """
        self.parser = None  # Will be initialized per file
        self.options = options or {}
        self.converted_files = []
        self.failed_files = []

    def detect_calculation_type(self, content: str) -> str:
        """
        Detect the calculation type from output content.

        Args:
            content: CRYSTAL output file content

        Returns:
            str: Calculation type ('OPT', 'SP', 'FREQ', 'UNKNOWN')
        """
        # Look for optimization patterns
        if "OPTGEOM" in content or "FINAL OPTIMIZED GEOMETRY" in content:
            return "OPT"

        # Look for frequency patterns
        if "FREQCALC" in content or "FREQUENCY CALCULATION" in content:
            return "FREQ"

        # Look for single point indicators
        if "SINGLE POINT CALCULATION" in content or ("SCF" in content and "OPTGEOM" not in content):
            return "SP"

        return "UNKNOWN"

    def extract_best_geometry(self, output_data: Dict[str, Any], calc_type: str) -> Dict[str, Any]:
        """
        Extract the most appropriate geometry based on calculation type.

        Args:
            output_data: Parsed output data from CrystalOutputParser
            calc_type: Calculation type

        Returns:
            Dict containing the best geometry data
        """
        # For now, use the geometry from the parser
        # TODO: Implement priority-based geometry selection
        return output_data

    def validate_geometry(self, geometry_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that the extracted geometry is reasonable.

        Args:
            geometry_data: Geometry data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if we have coordinates
        if not geometry_data.get("coordinates"):
            return False, "No atomic coordinates found"

        # Check if we have cell parameters for periodic systems
        dimensionality = geometry_data.get("dimensionality", "CRYSTAL")
        if dimensionality != "MOLECULE" and not geometry_data.get("conventional_cell"):
            return False, f"No cell parameters found for {dimensionality} calculation"

        # Check for reasonable coordinate values
        coords = geometry_data["coordinates"]
        for coord in coords:
            try:
                x, y, z = float(coord["x"]), float(coord["y"]), float(coord["z"])
                if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                    return False, f"Unreasonable coordinate values found: {x}, {y}, {z}"
            except (ValueError, KeyError) as e:
                return False, f"Invalid coordinate data: {e}"

        return True, ""

    def format_cell_parameters(self, cell_params: List[float], dimensionality: str) -> Dict[str, float]:
        """
        Format cell parameters based on dimensionality.

        Args:
            cell_params: List of [a, b, c, alpha, beta, gamma]
            dimensionality: System dimensionality

        Returns:
            Dict with formatted cell parameters
        """
        # Convert to float in case they come as strings
        a, b, c, alpha, beta, gamma = [float(x) for x in cell_params[:6]]

        vacuum_thickness = float(self.options.get("vacuum_thickness", 20.0))

        if dimensionality == "SLAB":
            # For 2D materials, use vacuum thickness for c-axis
            c = vacuum_thickness
        elif dimensionality == "POLYMER":
            # For 1D materials, use vacuum for b and c
            b = vacuum_thickness
            c = vacuum_thickness
        elif dimensionality == "MOLECULE":
            # For 0D materials, use large cell
            a = max(a, vacuum_thickness)
            b = max(b, vacuum_thickness)
            c = max(c, vacuum_thickness)

        return {
            "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma
        }

    def write_cif_file(self, geometry_data: Dict[str, Any], output_path: str) -> None:
        """
        Write geometry data to CIF format.

        Args:
            geometry_data: Geometry data to write
            output_path: Output CIF file path
        """
        with open(output_path, 'w') as f:
            # CIF header
            base_name = Path(output_path).stem
            f.write(f"data_{base_name}\n\n")

            # Add metadata if requested
            if self.options.get("include_metadata", False):
                f.write(f"# Generated by MACE CrystalOutToCif.py\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Dimensionality: {geometry_data.get('dimensionality', 'UNKNOWN')}\n")
                f.write(f"# Space group: {geometry_data.get('spacegroup', 'Unknown')}\n\n")

            # Cell parameters
            dimensionality = geometry_data.get("dimensionality", "CRYSTAL")
            cell_params = geometry_data.get("conventional_cell", [10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

            cell = self.format_cell_parameters(cell_params, dimensionality)
            precision = int(self.options.get("precision", 6))

            f.write(f"_cell_length_a    {cell['a']:.{precision}f}\n")
            f.write(f"_cell_length_b    {cell['b']:.{precision}f}\n")
            f.write(f"_cell_length_c    {cell['c']:.{precision}f}\n")
            f.write(f"_cell_angle_alpha {cell['alpha']:.{precision}f}\n")
            f.write(f"_cell_angle_beta  {cell['beta']:.{precision}f}\n")
            f.write(f"_cell_angle_gamma {cell['gamma']:.{precision}f}\n\n")

            # Space group (default to P1 for simplicity)
            space_group = self.options.get("space_group", "P 1")
            f.write(f"_symmetry_space_group_name_H-M  '{space_group}'\n")
            f.write(f"_symmetry_Int_Tables_number     1\n\n")

            # Symmetry operations
            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")
            f.write("  'x, y, z'\n\n")

            # Atomic coordinates
            f.write("loop_\n")
            f.write("_atom_site_label\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")

            coordinates = geometry_data["coordinates"]
            atom_counts = {}

            for coord in coordinates:
                atom_num = int(coord["atom_number"])
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atom_num, f"X{atom_num}")

                # Create unique labels
                if symbol not in atom_counts:
                    atom_counts[symbol] = 0
                atom_counts[symbol] += 1
                label = f"{symbol}{atom_counts[symbol]}"

                # Handle coordinates based on dimensionality
                if dimensionality == "SLAB":
                    # For SLAB: x,y fractional, z Cartesian (convert to fractional)
                    x_fract = float(coord["x"])
                    y_fract = float(coord["y"])
                    z_cart = float(coord["z"])
                    z_fract = z_cart / cell["c"]  # Convert Cartesian z to fractional
                elif dimensionality == "POLYMER":
                    # For POLYMER: x fractional, y,z Cartesian
                    x_fract = float(coord["x"])
                    y_cart = float(coord["y"])
                    z_cart = float(coord["z"])
                    y_fract = y_cart / cell["b"]
                    z_fract = z_cart / cell["c"]
                elif dimensionality == "MOLECULE":
                    # For MOLECULE: all Cartesian (convert to fractional)
                    x_cart = float(coord["x"])
                    y_cart = float(coord["y"])
                    z_cart = float(coord["z"])
                    x_fract = x_cart / cell["a"]
                    y_fract = y_cart / cell["b"]
                    z_fract = z_cart / cell["c"]
                else:
                    # For CRYSTAL: all fractional
                    x_fract = float(coord["x"])
                    y_fract = float(coord["y"])
                    z_fract = float(coord["z"])

                f.write(f"{label:8s} {symbol:2s} {x_fract:12.{precision}f} {y_fract:12.{precision}f} {z_fract:12.{precision}f}\n")

    def convert_file(self, out_file: str, cif_file: Optional[str] = None) -> bool:
        """
        Convert a single CRYSTAL output file to CIF.

        Args:
            out_file: Path to CRYSTAL output file
            cif_file: Output CIF file path (auto-generated if None)

        Returns:
            bool: True if conversion successful
        """
        try:
            if not os.path.exists(out_file):
                print(f"Error: Input file {out_file} not found")
                return False

            if self.options.get("verbose", False):
                print(f"Processing: {out_file}")

            # Parse the output file
            self.parser = CrystalOutputParser(out_file)
            output_data = self.parser.parse()

            # Detect calculation type
            with open(out_file, 'r') as f:
                content = f.read()
            calc_type = self.detect_calculation_type(content)

            if self.options.get("verbose", False):
                print(f"  Detected calculation type: {calc_type}")
                print(f"  Dimensionality: {output_data.get('dimensionality', 'UNKNOWN')}")

            # Extract best geometry
            geometry_data = self.extract_best_geometry(output_data, calc_type)

            # Validate geometry
            is_valid, error_msg = self.validate_geometry(geometry_data)
            if not is_valid:
                print(f"Error: Invalid geometry in {out_file}: {error_msg}")
                self.failed_files.append((out_file, error_msg))
                return False

            # Generate output filename if not provided
            if cif_file is None:
                base_name = Path(out_file).stem
                output_dir = self.options.get("output_dir")
                if output_dir is None:
                    output_dir = Path(out_file).parent
                cif_file = os.path.join(str(output_dir), f"{base_name}.cif")

            # Create output directory if needed
            if cif_file:
                os.makedirs(os.path.dirname(cif_file), exist_ok=True)

            # Check for dry run
            if self.options.get("dry_run", False):
                print(f"  Would create: {cif_file}")
                return True

            # Write CIF file
            self.write_cif_file(geometry_data, cif_file)

            if self.options.get("verbose", False):
                print(f"  Created: {cif_file}")

            self.converted_files.append((out_file, cif_file))
            return True

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"Error processing {out_file}: {error_msg}")
            self.failed_files.append((out_file, error_msg))
            return False

    def find_crystal_outputs(self, directory: str) -> List[str]:
        """
        Find all CRYSTAL output files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of output file paths
        """
        out_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.out') and not file.startswith('slurm-'):
                    out_files.append(os.path.join(root, file))
        return sorted(out_files)

    def convert_directory(self, directory: str) -> None:
        """
        Batch convert all CRYSTAL output files in a directory.

        Args:
            directory: Directory containing output files
        """
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory")
            return

        out_files = self.find_crystal_outputs(directory)

        if not out_files:
            print(f"No CRYSTAL output files found in {directory}")
            return

        print(f"Found {len(out_files)} output file(s) to convert")

        if self.options.get("dry_run", False):
            print("DRY RUN - no files will be created\n")

        success_count = 0
        for i, out_file in enumerate(out_files, 1):
            print(f"[{i}/{len(out_files)}] ", end="")
            if self.convert_file(out_file):
                success_count += 1

        # Summary
        print(f"\nConversion completed:")
        print(f"  Successful: {success_count}/{len(out_files)}")
        print(f"  Failed: {len(self.failed_files)}")

        if self.failed_files and self.options.get("verbose", False):
            print("\nFailed conversions:")
            for out_file, error in self.failed_files:
                print(f"  {out_file}: {error}")

    def run(self, targets: List[str]) -> None:
        """
        Run the converter on the given targets.

        Args:
            targets: List of files or directories to convert
        """
        if not targets:
            targets = ["."]  # Default to current directory

        for target in targets:
            if os.path.isfile(target):
                self.convert_file(target)
            elif os.path.isdir(target):
                self.convert_directory(target)
            else:
                print(f"Error: {target} not found")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert CRYSTAL17/23 output files to CIF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s material.out                    # Convert single file
  %(prog)s .                               # Convert all files in current directory
  %(prog)s /path/to/calculations/          # Convert files in specific directory
  %(prog)s . --output-dir cifs/            # Save CIFs to specific directory
  %(prog)s . --include-metadata --verbose  # Include metadata and verbose output
  %(prog)s . --dry-run                     # Show what would be converted
        """
    )

    parser.add_argument(
        "targets",
        nargs="*",
        help="CRYSTAL output files or directories to convert"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for CIF files (default: same as input)"
    )
    parser.add_argument(
        "--force-dimension",
        choices=["3d", "2d", "1d", "0d"],
        help="Override detected dimensionality"
    )
    parser.add_argument(
        "--space-group",
        default="P 1",
        help="Space group for CIF (default: P 1)"
    )
    parser.add_argument(
        "--vacuum-thickness",
        type=float,
        default=20.0,
        help="Vacuum thickness for 2D/1D/0D materials (default: 20.0 Å)"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Coordinate precision (default: 6)"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include calculation metadata in CIF comments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without creating files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Convert arguments to options dictionary
    options = {
        "output_dir": args.output_dir,
        "force_dimension": args.force_dimension,
        "space_group": args.space_group,
        "vacuum_thickness": args.vacuum_thickness,
        "precision": args.precision,
        "include_metadata": args.include_metadata,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
    }

    # Create converter and run
    converter = CrystalOutToCifConverter(options)
    converter.run(args.targets)


if __name__ == "__main__":
    main()