#!/usr/bin/env python3
"""
CRYSTAL Workflow Planner
========================
Comprehensive workflow planning system that handles the complete calculation pipeline
from input preparation (CIFs or D12s) through all desired calculation steps.

Features:
- Flexible input handling (CIFs or D12s)
- Complete workflow pre-planning
- Integration with NewCifToD12.py and CRYSTALOptToD12.py
- JSON configuration management
- Custom workflow sequences (e.g., Opt1 → Opt2 → SP → Band/DOS)
- Resource planning and optimization

Author: Enhanced workflow planning system
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml

# Import MACE components
try:
    from mace.database.materials import create_material_id_from_file
    from mace.queue.manager import EnhancedCrystalQueueManager
    from mace.workflow.engine import WorkflowEngine
except ImportError:
    try:
        from .engine import WorkflowEngine
    except ImportError:
        WorkflowEngine = None

try:
    # Add the Crystal_d12 directory to path for importing
    parent_dir = Path(__file__).parent.parent.parent  # Go up to reorganization/
    sys.path.insert(0, str(parent_dir / "Crystal_d12"))
    # Import from the new modular structure
    from d12_constants import FREQ_TEMPLATES, ATOMIC_NUMBER_TO_SYMBOL, SPACEGROUP_SYMBOLS
    # Import succeeded, we should have access to these constants
    D12_CONSTANTS_AVAILABLE = True
    
    # Now try to import DummyFileCreator using relative import
    try:
        from .dummy_file_creator import DummyFileCreator
    except ImportError:
        try:
            from mace.workflow.dummy_file_creator import DummyFileCreator
        except ImportError:
            DummyFileCreator = None
except ImportError as e:
    print(f"Warning: Could not import d12_constants: {e}")
    D12_CONSTANTS_AVAILABLE = False
    # Provide minimal fallbacks
    FREQ_TEMPLATES = {}
    ATOMIC_NUMBER_TO_SYMBOL = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu',
        47: 'Ag', 79: 'Au'
    }
    SPACEGROUP_SYMBOLS = {
        1: 'P1', 2: 'P-1', 
        115: 'P -4 m 2',  # Tetragonal
        166: 'R -3 m',    # Rhombohedral/Trigonal
        227: 'Fd-3m'      # Cubic
    }
    # If d12_constants import failed, still try to get DummyFileCreator
    if 'DummyFileCreator' not in locals():
        try:
            from .dummy_file_creator import DummyFileCreator
        except ImportError:
            try:
                # Try absolute import
                from mace.workflow.dummy_file_creator import DummyFileCreator
            except ImportError:
                DummyFileCreator = None
                print("Warning: DummyFileCreator not available - will use fallback dummy files")


class WorkflowPlanner:
    """
    Comprehensive workflow planning system for CRYSTAL calculations.

    Handles the complete pipeline from input preparation through execution:
    1. Input handling (CIFs or D12s)
    2. Workflow planning and configuration
    3. JSON settings management
    4. Queue management and execution
    """

    def __init__(self, work_dir: str = ".", db_path: str = "materials.db"):
        self.work_dir = Path(work_dir).resolve()
        self.db_path = db_path
        # Database is not needed during planning phase - will be created by executor if needed
        # self.db = MaterialDatabase(db_path)  # Removed to prevent unnecessary database creation

        # Create necessary directories
        self.configs_dir = self.work_dir / "workflow_configs"
        self.outputs_dir = self.work_dir / "workflow_outputs"
        self.temp_dir = self.work_dir / "workflow_temp"
        self.scripts_dir = self.work_dir / "workflow_scripts"

        for dir_path in [
            self.configs_dir,
            self.outputs_dir,
            self.temp_dir,
            self.scripts_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Available calculation types and their dependencies
        # Note: This only includes base types - numbered versions are handled dynamically
        # Dependencies are now "soft" - they indicate preferred order but not strict requirements
        self.calc_types = {
            "OPT": {
                "name": "Geometry Optimization",
                "depends_on": [],
                "generates": ["optimized_geometry"],
            },
            "SP": {
                "name": "Single Point",
                "depends_on": [],
                "generates": ["electronic_structure", "wavefunction"],
            },
            "BAND": {
                "name": "Band Structure",
                "depends_on": ["SP", "OPT"],
                "generates": ["band_structure"],
            },
            "DOSS": {
                "name": "Density of States",
                "depends_on": ["SP", "OPT"],
                "generates": ["density_of_states"],
            },
            "FREQ": {
                "name": "Frequencies",
                "depends_on": ["OPT"],
                "generates": ["vibrational_modes"],
            },
            "TRANSPORT": {
                "name": "Transport Properties",
                "depends_on": ["SP", "OPT"],  # Requires wavefunction from SP or OPT
                "generates": ["conductivity", "seebeck_coefficient"],
                # Note: Runs sequentially after BAND/DOSS, not in parallel
            },
            "CHARGE+POTENTIAL": {
                "name": "Charge Density & Potential",
                "depends_on": ["SP", "OPT"],  # Requires wavefunction from SP or OPT
                "generates": ["charge_density", "electrostatic_potential"],
                # Note: Runs sequentially after BAND/DOSS, not in parallel
            },
        }

        # Numbered calculations (OPT2, SP2, etc.) are handled dynamically
        # They have the same dependencies as their base types

        # Predefined workflow templates
        self.workflow_templates = {
            "basic_opt": ["OPT"],
            "opt_sp": ["OPT", "SP"],
            "full_electronic": ["OPT", "SP", "BAND", "DOSS"],
            "double_opt": ["OPT", "OPT2", "SP"],
            "complete": ["OPT", "SP", "BAND", "DOSS", "FREQ"],
            "transport_analysis": ["OPT", "SP", "TRANSPORT"],
            "charge_analysis": ["OPT", "SP", "CHARGE+POTENTIAL"],
            "combined_analysis": ["OPT", "SP", "BAND", "DOSS", "TRANSPORT"],
            "custom": "user_defined",
        }

    def display_welcome(self):
        """Display welcome message and overview"""
        # Banner is shown by run_workflow.py, so we just show the title
        print("=" * 60)
        print("MACE Workflow Planner")
        print("=" * 60)
        print("Comprehensive workflow planning for CRYSTAL calculations")
        print()
        print("This planner will help you:")
        print("  1. Set up input files (from CIFs or existing D12s)")
        print("  2. Plan your complete calculation workflow")
        print("  3. Configure all calculation settings")
        print("  4. Save configurations for reproducibility")
        print("  5. Execute the planned workflow")
        print()

    def create_material_id_from_file(self, filepath: Path) -> str:
        """Wrapper for the imported create_material_id_from_file function"""
        return create_material_id_from_file(str(filepath))

    def get_input_type(self) -> str:
        """Determine input type and location"""
        print("Step 1: Input Configuration")
        print("-" * 40)

        input_options = {
            "1": "CIF files (will convert to D12s)",
            "2": "Existing D12 files",
            "3": "Mixed (some CIFs, some D12s)",
        }

        input_choice = get_user_input("Select input type", input_options, "1")
        input_type_map = {"1": "cif", "2": "d12", "3": "mixed"}

        return input_type_map[input_choice]

    def get_input_directory(self, input_type: str) -> Path:
        """Get input directory from user"""
        if input_type == "cif":
            default_prompt = "Directory containing CIF files"
        elif input_type == "d12":
            default_prompt = "Directory containing D12 files"
        else:
            default_prompt = "Directory containing input files"

        while True:
            input_dir = input(
                f"\n{default_prompt} (default: current directory): "
            ).strip()
            if not input_dir:
                input_dir = "."

            input_path = Path(input_dir).resolve()
            if input_path.exists() and input_path.is_dir():
                return input_path
            else:
                print(f"Directory {input_path} does not exist. Please try again.")

    def scan_input_files(
        self, input_dir: Path, input_type: str
    ) -> Dict[str, List[Path]]:
        """Scan directory for input files"""
        print(f"\nScanning {input_dir} for input files...")

        files = {"cif": [], "d12": []}

        if input_type in ["cif", "mixed"]:
            files["cif"] = list(input_dir.glob("*.cif"))

        if input_type in ["d12", "mixed"]:
            files["d12"] = list(input_dir.glob("*.d12"))

        total_files = len(files["cif"]) + len(files["d12"])
        print(f"Found {len(files['cif'])} CIF files and {len(files['d12'])} D12 files")
        print(f"Total: {total_files} input files")

        if total_files == 0:
            print("No input files found!")
            return files

        # Show sample files
        if files["cif"]:
            print(f"\nSample CIF files:")
            for i, f in enumerate(files["cif"][:3]):
                print(f"  {f.name}")
            if len(files["cif"]) > 3:
                print(f"  ... and {len(files['cif']) - 3} more")

        if files["d12"]:
            print(f"\nSample D12 files:")
            for i, f in enumerate(files["d12"][:3]):
                print(f"  {f.name}")
            if len(files["d12"]) > 3:
                print(f"  ... and {len(files['d12']) - 3} more")

        return files

    def setup_cif_conversion(
        self, cif_files: List[Path], first_calc_type: str = "OPT"
    ) -> Dict[str, Any]:
        """Set up CIF to D12 conversion using NewCifToD12.py"""
        print("\nStep 3: CIF Conversion Setup")
        print("-" * 40)
        print(f"Configuring CIF to D12 conversion for {first_calc_type} calculation")

        # Ask if user wants to use default settings or customize
        use_defaults = yes_no_prompt("Use default CIF conversion settings?", "yes")

        if use_defaults:
            # Use sensible defaults
            cif_config = {
                "symmetry_handling": "CIF",
                "write_only_unique": True,
                "dimensionality": "CRYSTAL",
                "calculation_type": "SP" if first_calc_type == "SP" else "OPT",
                "optimization_type": "FULLOPTG" if first_calc_type != "SP" else None,
                "method": "DFT",
                "dft_functional": "HSE06",
                "use_dispersion": True,
                "basis_set_type": "INTERNAL",
                "basis_set": "POB-TZVP-REV2",
                "dft_grid": "XLGRID",
                "is_spin_polarized": True,
                "use_smearing": False,
                "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
                "scf_method": "DIIS",
                "scf_maxcycle": 800,
                "fmixing": 30,
            }
            print("Using default settings for CIF conversion:")
            print(f"  Method: DFT/HSE06-D3")
            print(f"  Basis set: POB-TZVP-REV2 (internal)")
            print(
                f"  Calculation: {first_calc_type} ({'Geometry optimization' if first_calc_type != 'SP' else 'Single point energy'})"
            )
            print(f"  Symmetry: Use CIF symmetry information")
        else:
            print("Custom CIF conversion setup:")
            print("Choose customization level:")
            print("  1: Basic (functional + basis set)")
            print("     - Time to configure: ~30 seconds")
            print("     - Good for: Quick calculations with different methods")
            print("  2: Advanced (most common settings)")
            print("     - Time to configure: ~2-3 minutes")
            print("     - Good for: Production calculations, specific requirements")
            print("  3: Expert (full NewCifToD12.py integration)")
            print("     - Time to configure: ~5-10 minutes")
            print(
                "     - Good for: Complex systems, special basis sets, custom settings"
            )

            level_choice = self.get_safe_choice_input(
                "Customization level", valid_choices=["1", "2", "3"], default="1"
            )

            if level_choice == "1":
                cif_config = self.get_basic_customization(first_calc_type)
            elif level_choice == "2":
                cif_config = self.get_advanced_customization(first_calc_type)
            elif level_choice == "3":
                print("\nLaunching NewCifToD12.py for full configuration...")
                print(
                    "This will run the interactive configuration and save the results."
                )
                cif_config = self.run_full_cif_customization(
                    cif_files[0], first_calc_type
                )
            else:
                print("Invalid choice. Using basic customization.")
                cif_config = self.get_basic_customization(first_calc_type)

        # Save CIF configuration
        cif_config_file = self.configs_dir / "cif_conversion_config.json"
        with open(cif_config_file, "w") as f:
            json.dump(cif_config, f, indent=2)

        print(f"CIF conversion config saved to: {cif_config_file}")

        return cif_config

    def get_default_cif_config(self, calc_type: str = "OPT") -> Dict[str, Any]:
        """Return sensible default CIF configuration"""
        return {
            "symmetry_handling": "CIF",
            "write_only_unique": True,
            "dimensionality": "CRYSTAL",
            "calculation_type": "SP" if calc_type == "SP" else "OPT",
            "optimization_type": "FULLOPTG" if calc_type != "SP" else None,
            "optimization_settings": {
                "TOLDEG": 0.00003,
                "TOLDEX": 0.00012,
                "TOLDEE": 7,
                "MAXCYCLE": 800,
            },
            "method": "DFT",
            "dft_functional": "HSE06",
            "use_dispersion": True,
            "basis_set_type": "INTERNAL",
            "basis_set": "POB-TZVP-REV2",
            "dft_grid": "XLGRID",
            "is_spin_polarized": True,
            "use_smearing": False,
            "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
            "scf_method": "DIIS",
            "scf_maxcycle": 800,
            "fmixing": 30,
        }

    def get_basic_customization(self, first_calc_type: str = "OPT") -> Dict[str, Any]:
        """Get basic CIF customization options"""
        print("\nBasic customization options:")

        # Common functionals for quick selection
        print("Common DFT Functionals:")
        print("  1: HSE06 (hybrid, good for band gaps)")
        print("  2: PBE0 (hybrid)")
        print("  3: B3LYP (hybrid)")
        print("  4: PBE (GGA)")
        print("  5: PBESOL (GGA for solids)")
        print("  6: HF3C (low-cost with corrections)")
        print("  7: Custom (type any functional name)")

        func_choice = input("Select functional [1]: ").strip() or "1"
        func_map = {
            "1": "HSE06",
            "2": "PBE0",
            "3": "B3LYP",
            "4": "PBE",
            "5": "PBESOL",
            "6": "HF3C",
        }

        if func_choice == "7":
            functional_choice = input("Enter functional name: ").strip().upper()
        else:
            functional_choice = func_map.get(func_choice, "HSE06")

        # Map special functionals
        functional_keyword_map = {"PBESOL": "PBESOLXC", "SOGGA": "SOGGAXC"}
        crystal_functional = functional_keyword_map.get(
            functional_choice, functional_choice
        )

        # Basis set - handle 3c methods
        three_c_basis_map = {
            "HF3C": "MINIX",
            "HFSOL3C": "SOLMINIX",
            "PBEH3C": "def2-mSVP",
            "PBE03C": "def2-mSVP",
            "HSE3C": "def2-mSVP",
            "B973C": "mTZVP",
            "PBESOL03C": "SOLDEF2MSVP",
            "HSESOL3C": "SOLDEF2MSVP",
        }

        if functional_choice in three_c_basis_map:
            basis_choice = three_c_basis_map[functional_choice]
            print(f"{functional_choice} requires specific basis set: {basis_choice}")
        else:
            basis_choice = (
                input("Basis set [POB-TZVP-REV2]: ").strip() or "POB-TZVP-REV2"
            )

        cif_config = self.get_default_cif_config(first_calc_type)
        cif_config["dft_functional"] = crystal_functional
        cif_config["basis_set"] = basis_choice

        # Check D3 support
        d3_supported = functional_choice in [
            "BLYP",
            "PBE",
            "B97",
            "B3LYP",
            "PBE0",
            "mPW1PW91",
            "M06",
            "HSE06",
            "HSEsol",
            "LC-wPBE",
        ]
        if d3_supported and functional_choice not in three_c_basis_map:
            cif_config["use_dispersion"] = yes_no_prompt(
                f"Use D3 dispersion? ({functional_choice} supports it)", "yes"
            )
        elif functional_choice in three_c_basis_map:
            cif_config["use_dispersion"] = False  # Already included in 3c methods

        print(f"\nUsing customized settings:")
        print(f"  Method: DFT/{functional_choice}")
        print(f"  Basis set: {basis_choice}")
        print(f"  Dispersion: {'Yes' if cif_config.get('use_dispersion') else 'No'}")
        print(f"  Other settings: Using defaults")

        return cif_config

    def get_advanced_customization(
        self, first_calc_type: str = "OPT"
    ) -> Dict[str, Any]:
        """Get advanced CIF customization options"""
        print("\nAdvanced customization options:")

        # Method choice
        print("Method:")
        print("  1: DFT")
        print("  2: HF")
        method_choice = input("Method [1]: ").strip() or "1"
        method = "DFT" if method_choice == "1" else "HF"

        # DFT functional (if DFT)
        functional = "HSE06"
        if method == "DFT":
            print("\nDFT Functional Categories:")
            print("  1: Hybrid (HSE06, PBE0, B3LYP, etc.)")
            print("  2: GGA (PBE, PBESOL, BLYP, etc.)")
            print("  3: LDA (SVWN, PZ, PWLDA)")
            print("  4: meta-GGA (SCAN, M06, M06-L)")
            print("  5: 3c methods (HF-3c, B97-3c, PBE0-3c)")

            cat_choice = input("Category [1]: ").strip() or "1"

            if cat_choice == "1":  # Hybrid
                print("\nHybrid Functionals:")
                print("  1: HSE06 (recommended for band gaps)")
                print("  2: PBE0")
                print("  3: B3LYP")
                print("  4: PBESOL0")
                print("  5: HSEsol")
                print("  6: PBE0-13")
                func_choice = input("Select [1]: ").strip() or "1"
                functionals = {
                    "1": "HSE06",
                    "2": "PBE0",
                    "3": "B3LYP",
                    "4": "PBESOL0",
                    "5": "HSEsol",
                    "6": "PBE0-13",
                }
                functional = functionals.get(func_choice, "HSE06")

            elif cat_choice == "2":  # GGA
                print("\nGGA Functionals:")
                print("  1: PBE")
                print("  2: PBESOL (for solids)")
                print("  3: BLYP")
                print("  4: SOGGA")
                print("  5: WCGGA")
                func_choice = input("Select [1]: ").strip() or "1"
                functionals = {
                    "1": "PBE",
                    "2": "PBESOL",
                    "3": "BLYP",
                    "4": "SOGGA",
                    "5": "WCGGA",
                }
                functional = functionals.get(func_choice, "PBE")

            elif cat_choice == "3":  # LDA
                print("\nLDA Functionals:")
                print("  1: SVWN")
                print("  2: PZ")
                print("  3: PWLDA")
                func_choice = input("Select [1]: ").strip() or "1"
                functionals = {"1": "SVWN", "2": "PZ", "3": "PWLDA"}
                functional = functionals.get(func_choice, "SVWN")

            elif cat_choice == "4":  # meta-GGA
                print("\nmeta-GGA Functionals:")
                print("  1: SCAN")
                print("  2: M06")
                print("  3: M06-L")
                print("  4: M06-2X")
                func_choice = input("Select [1]: ").strip() or "1"
                functionals = {"1": "SCAN", "2": "M06", "3": "M06-L", "4": "M06-2X"}
                functional = functionals.get(func_choice, "SCAN")

            elif cat_choice == "5":  # 3c methods
                print("\n3c Methods (include dispersion and BSSE corrections):")
                print("  1: HF3C (uses MINIX basis)")
                print("  2: B973C (uses mTZVP basis)")
                print("  3: PBEH3C / PBE03C (uses def2-mSVP basis)")
                print("  4: HSE3C (uses def2-mSVP basis)")
                print("  5: HFSOL3C (for solids, uses SOLMINIX basis)")
                print("  6: PBESOL03C (for solids, uses SOLDEF2MSVP basis)")
                print("  7: HSESOL3C (for solids, uses SOLDEF2MSVP basis)")
                func_choice = input("Select [1]: ").strip() or "1"
                functionals = {
                    "1": "HF3C",
                    "2": "B973C",
                    "3": "PBEH3C",
                    "4": "HSE3C",
                    "5": "HFSOL3C",
                    "6": "PBESOL03C",
                    "7": "HSESOL3C",
                }
                functional = functionals.get(func_choice, "HF3C")

        # Basis set
        # Check if 3c method selected - they use specific basis sets
        three_c_basis_map = {
            "HF3C": "MINIX",
            "HFSOL3C": "SOLMINIX",
            "PBEH3C": "def2-mSVP",
            "PBE03C": "def2-mSVP",  # PBEH3C and PBE03C are the same
            "HSE3C": "def2-mSVP",
            "B973C": "mTZVP",
            "PBESOL03C": "SOLDEF2MSVP",
            "HSESOL3C": "SOLDEF2MSVP",
        }

        if functional in three_c_basis_map:
            basis_set = three_c_basis_map[functional]
            print(f"\n{functional} requires specific basis set: {basis_set}")
            basis_type = "INTERNAL"
            # 3c methods include dispersion by design
            dispersion = False
        else:
            print("\nBasis set:")
            print("  1: POB-TZVP-REV2 (internal)")
            print("  2: 6-31G* (internal)")
            print("  3: def2-TZVP (internal)")
            print("  4: Custom external")
            basis_choice = input("Basis set [1]: ").strip() or "1"
            basis_options = {
                "1": "POB-TZVP-REV2",
                "2": "6-31G*",
                "3": "def2-TZVP",
                "4": "EXTERNAL",
            }
            basis_set = basis_options.get(basis_choice, "POB-TZVP-REV2")
            basis_type = "EXTERNAL" if basis_choice == "4" else "INTERNAL"

            # Dispersion correction - check if functional supports D3
            d3_supported = functional in [
                "BLYP",
                "PBE",
                "B97",
                "B3LYP",
                "PBE0",
                "mPW1PW91",
                "M06",
                "HSE06",
                "HSEsol",
                "LC-wPBE",
            ]

            if d3_supported:
                dispersion = yes_no_prompt(
                    f"\nUse D3 dispersion correction? ({functional} supports D3)", "yes"
                )
            else:
                print(f"\nNote: {functional} does not support D3 dispersion correction")
                dispersion = False

        # Spin polarization
        spin_polarized = yes_no_prompt("Use spin polarization?", "yes")

        # Optimization type
        print("Optimization type:")
        print("  1: FULLOPTG (full optimization)")
        print("  2: CELLONLY (cell only)")
        print("  3: ATOMONLY (atomic positions only)")
        opt_choice = input("Optimization [1]: ").strip() or "1"
        opt_types = {"1": "FULLOPTG", "2": "CELLONLY", "3": "ATOMONLY"}
        opt_type = opt_types.get(opt_choice, "FULLOPTG")

        # Map functionals to their correct CRYSTAL keywords
        functional_keyword_map = {
            "PBESOL": "PBESOLXC",
            "SOGGA": "SOGGAXC",
        }

        # Apply functional mapping if needed
        crystal_functional = functional_keyword_map.get(functional, functional)

        # Build configuration
        cif_config = self.get_default_cif_config(first_calc_type)
        cif_config.update(
            {
                "method": method,
                "dft_functional": crystal_functional,
                "basis_set": basis_set,
                "basis_set_type": basis_type,
                "use_dispersion": dispersion,
                "is_spin_polarized": spin_polarized,
                "optimization_type": opt_type,
            }
        )

        print(f"\nAdvanced configuration:")
        print(f"  Method: {method}/{functional if method == 'DFT' else 'HF'}")
        print(f"  Basis set: {basis_set} ({basis_type})")
        print(f"  Dispersion: {'Yes' if dispersion else 'No'}")
        print(f"  Spin polarized: {'Yes' if spin_polarized else 'No'}")
        print(f"  Optimization: {opt_type}")

        return cif_config

    def run_full_cif_customization(
        self, sample_cif: Path, first_calc_type: str = "OPT"
    ) -> Dict[str, Any]:
        """Run full NewCifToD12.py customization"""
        # Find NewCifToD12.py - it's now in the root Crystal_d12 directory
        script_path = Path(__file__).parent.parent.parent / "Crystal_d12" / "NewCifToD12.py"

        if not script_path.exists():
            print(f"Error: NewCifToD12.py not found at {script_path}")
            print("Using advanced customization instead...")
            return self.get_advanced_customization(first_calc_type)

        print(f"\nRunning NewCifToD12.py with sample file: {sample_cif.name}")
        print("This will launch the full interactive configuration.")
        print("At the end, choose to SAVE the configuration for batch processing.")
        print("Press Enter to continue...")
        input()

        # Create temporary options file
        temp_options = self.temp_dir / "temp_cif_options.json"

        # Run NewCifToD12.py interactively
        cmd = [
            sys.executable,
            str(script_path),
            "--cif_dir",
            str(sample_cif.parent),
            "--save_options",
            "--options_file",
            str(temp_options),
        ]

        print("Launching NewCifToD12.py...")
        try:
            # Run interactively (not captured)
            result = subprocess.run(cmd, cwd=str(sample_cif.parent))

            if result.returncode == 0 and temp_options.exists():
                # Load the saved configuration
                with open(temp_options, "r") as f:
                    cif_config = json.load(f)
                print("Successfully loaded configuration from NewCifToD12.py")
                return cif_config
            else:
                print("NewCifToD12.py configuration failed or was cancelled.")
                print("Using default configuration...")
                return self.get_default_cif_config(first_calc_type)

        except Exception as e:
            print(f"Error running NewCifToD12.py: {e}")
            print("Using default configuration...")
            return self.get_default_cif_config(first_calc_type)

    def plan_workflow_sequence(self) -> List[str]:
        """Plan the complete workflow sequence"""
        print("\nStep 3: Workflow Sequence Planning")
        print("-" * 40)

        print("Available workflow templates:")
        for i, (key, sequence) in enumerate(self.workflow_templates.items(), 1):
            if key == "custom":
                print(f"{i}: Custom workflow (define your own sequence)")
            else:
                print(f"{i}: {key.replace('_', ' ').title()} - {' → '.join(sequence)}")

        template_options = {
            str(i): key for i, key in enumerate(self.workflow_templates.keys(), 1)
        }

        template_choice = get_user_input(
            "Select workflow template", template_options, "3"
        )
        selected_template = template_options[template_choice]
        print(f"Selected template: {selected_template}")

        if selected_template == "custom":
            return self.design_custom_workflow()
        else:
            sequence = self.workflow_templates[selected_template]
            print(f"\nSelected workflow: {' → '.join(sequence)}")

            # Ask if user wants to modify
            modify = yes_no_prompt("Modify this workflow?", "no")
            if modify:
                return self.modify_workflow_sequence(sequence)
            else:
                return sequence

    def design_custom_workflow(self) -> List[str]:
        """Design a custom workflow sequence"""
        print("\nCustom Workflow Designer")
        print("Available calculation types:")

        for calc_type, info in self.calc_types.items():
            deps = (
                " (depends on: " + ", ".join(info["depends_on"]) + ")"
                if info["depends_on"]
                else ""
            )
            print(f"  {calc_type}: {info['name']}{deps}")

        print(
            "\nNOTE: Numbered calculations (OPT2, SP2, BAND2, etc.) will be automatically assigned"
        )
        print("when you add multiple calculations of the same type.\n")

        print("Build your workflow sequence:")
        print("Enter calculation types in order (e.g., OPT SP BAND DOSS)")
        print("Type 'help' for workflow examples, 'list' to see available types again")

        sequence = []
        while True:
            # Display current sequence
            if sequence:
                self._display_workflow_sequence(sequence)
            else:
                print("\nCurrent sequence: Empty")

            user_input = input("\nNext calculation (or 'done'): ").strip().upper()

            if user_input == "DONE":
                break
            elif user_input == "HELP":
                self.show_workflow_help()
            elif user_input == "LIST" or user_input == "":
                # Show available types again
                print("\nAvailable calculation types:")
                for calc_type, info in self.calc_types.items():
                    deps = (
                        " (depends on: " + ", ".join(info["depends_on"]) + ")"
                        if info["depends_on"]
                        else ""
                    )
                    print(f"  {calc_type}: {info['name']}{deps}")
                print("\nType 'help' for workflow examples")
            elif user_input in self.calc_types:
                # Get properly numbered version
                numbered_calc = self._get_next_numbered_calc(sequence, user_input)
                if self._validate_numbered_calc_addition(sequence, numbered_calc):
                    sequence.append(numbered_calc)
                    print(f"✅ Added {numbered_calc}")
                else:
                    deps = ", ".join(self.calc_types[user_input]["depends_on"])
                    print(f"❌ Cannot add {user_input}. Missing dependencies: {deps}")
            else:
                print(f"Unknown calculation type: {user_input}")
                print("Available types:", ", ".join(self.calc_types.keys()))
                print("Type 'list' to see full descriptions")

        if not sequence:
            print("No calculations selected. Using basic optimization.")
            sequence = ["OPT"]

        return sequence

    def modify_workflow_sequence(self, sequence: List[str]) -> List[str]:
        """Modify an existing workflow sequence"""
        print(f"\nModifying workflow: {' → '.join(sequence)}")

        while True:
            print("\nOptions:")
            print("1: Add calculation")
            print("2: Remove calculation")
            print("3: Insert calculation")
            print("4: Show current sequence")
            print("5: Done")

            choice = input("Select option: ").strip()

            if choice == "1":
                # Show available calculation types with smart numbering
                print("\nAvailable calculation types to add:")
                available_types = self._get_available_calc_types(sequence)
                for i, calc_type in enumerate(available_types, 1):
                    base = calc_type.rstrip("0123456789")
                    desc = {
                        "OPT": "Geometry Optimization",
                        "SP": "Single Point Energy",
                        "BAND": "Band Structure",
                        "DOSS": "Density of States",
                        "FREQ": "Vibrational Frequencies",
                        "TRANSPORT": "Transport Properties",
                        "CHARGE+POTENTIAL": "Charge Density & Electrostatic Potential",
                    }.get(base, base)
                    print(f"  {i}. {calc_type} - {desc}")
                print("\nEnter number or type name directly")

                calc_input = input("\nAdd calculation: ").strip()

                # Handle numeric input
                if calc_input.isdigit():
                    idx = int(calc_input) - 1
                    if 0 <= idx < len(available_types):
                        calc = available_types[idx]
                    else:
                        print("Invalid number")
                        continue
                else:
                    calc = calc_input.upper()

                # Check if it's in the available types list (handles numbered versions like OPT2, SP3)
                if calc in available_types:
                    if self._validate_numbered_calc_addition(sequence, calc):
                        sequence.append(calc)
                        print(f"Added {calc}. Current: {' → '.join(sequence)}")
                    else:
                        print(f"Cannot add {calc} - check dependencies")
                # Handle base calculations (OPT, SP, etc.) - auto-number them
                elif calc in ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT", "CHARGE+POTENTIAL"]:
                    numbered_calc = self._get_next_numbered_calc(sequence, calc)
                    if self._validate_numbered_calc_addition(sequence, numbered_calc):
                        sequence.append(numbered_calc)
                        print(f"Added {numbered_calc}. Current: {' → '.join(sequence)}")
                    else:
                        print(f"Cannot add {numbered_calc} - check dependencies")
                else:
                    print(f"Unknown calculation type: {calc}")

            elif choice == "2":
                if sequence:
                    print(f"\nCurrent sequence: {' → '.join(sequence)}")
                    print(
                        "Enter the exact calculation to remove (e.g., OPT2, SP, BAND2):"
                    )
                    calc = input("Remove: ").strip().upper()
                    if calc in sequence:
                        sequence.remove(calc)
                        print(f"Removed {calc}. Current: {' → '.join(sequence)}")
                    else:
                        print(f"{calc} not found in sequence")

            elif choice == "3":
                print("\nAvailable calculation types to insert:")
                available_types = self._get_available_calc_types(sequence)
                for calc_type in available_types:
                    print(f"  {calc_type}")

                calc = input("\nInsert calculation type: ").strip().upper()

                # Handle numbered calculations
                if calc in ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT", "CHARGE+POTENTIAL"]:
                    numbered_calc = self._get_next_numbered_calc(sequence, calc)

                    print(f"\nCurrent sequence: {' → '.join(sequence)}")
                    print("Positions:")
                    for i in range(len(sequence) + 1):
                        if i == 0:
                            print(f"  1: Before {sequence[0]}")
                        elif i == len(sequence):
                            print(f"  {i + 1}: After {sequence[-1]}")
                        else:
                            print(
                                f"  {i + 1}: Between {sequence[i - 1]} and {sequence[i]}"
                            )

                    pos = input("Insert at position: ").strip()
                    try:
                        pos = int(pos) - 1
                        if 0 <= pos <= len(sequence):
                            if self._validate_numbered_calc_addition(
                                sequence[:pos] + sequence[pos:], numbered_calc
                            ):
                                sequence.insert(pos, numbered_calc)
                                print(
                                    f"Inserted {numbered_calc}. Current: {' → '.join(sequence)}"
                                )
                            else:
                                print(
                                    f"Cannot insert {numbered_calc} at this position - check dependencies"
                                )
                    except ValueError:
                        print("Invalid position")

            elif choice == "4":
                self._display_workflow_sequence(sequence)

            elif choice == "5":
                if not sequence:
                    print("\nError: Cannot have empty workflow!")
                    continue
                break

        return sequence

    def _get_available_calc_types(self, current_sequence: List[str]) -> List[str]:
        """Get list of available calculation types with proper numbering"""
        available = []
        base_types = ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT", "CHARGE+POTENTIAL"]

        for base_type in base_types:
            # Count how many of this type already exist
            count = sum(1 for calc in current_sequence if calc.startswith(base_type))

            if count == 0:
                # First instance doesn't need a number
                available.append(base_type)
            else:
                # Subsequent instances need numbers
                available.append(f"{base_type}{count + 1}")

        return available

    def _get_next_numbered_calc(self, sequence: List[str], base_type: str) -> str:
        """Get the next numbered version of a calculation type"""
        # Count existing instances of this base type
        count = 0
        for calc in sequence:
            if (
                calc == base_type
                or calc.startswith(base_type)
                and calc[len(base_type) :].isdigit()
            ):
                count += 1

        if count == 0:
            return base_type
        else:
            return f"{base_type}{count + 1}"

    def _display_workflow_sequence(self, sequence: List[str]):
        """Display the workflow sequence with details"""
        print(f"\nCurrent workflow sequence ({len(sequence)} steps):")
        print("─" * 70)

        if not sequence:
            print("  (Empty workflow)")
            return

        # Create visual representation
        for i, calc in enumerate(sequence):
            # Parse calculation type
            base = calc.rstrip("0123456789")
            num = calc[len(base) :] or "1"

            # Get description
            desc = {
                "OPT": "Geometry Optimization",
                "SP": "Single Point Energy",
                "BAND": "Band Structure",
                "DOSS": "Density of States",
                "FREQ": "Vibrational Frequencies",
            }.get(base, base)

            # Determine source
            if i == 0:
                source = "Input files"
            elif base == "OPT":
                source = f"From step {i}"
            elif base == "SP":
                # Find most recent OPT
                for j in range(i - 1, -1, -1):
                    if sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j + 1})"
                        break
                else:
                    source = "From previous step"
            elif base in ["BAND", "DOSS"]:
                # Find most recent SP or OPT
                for j in range(i - 1, -1, -1):
                    if sequence[j].startswith("SP") or sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j + 1})"
                        break
                else:
                    source = "From previous step"
            elif base == "FREQ":
                # Find most recent OPT
                for j in range(i - 1, -1, -1):
                    if sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j + 1})"
                        break
                else:
                    source = "From previous step"
            else:
                source = "From previous step"

            # Print step info
            print(f"  Step {i + 1}: {calc:<8} - {desc:<25} [{source}]")

            if i < len(sequence) - 1:
                print("    ↓")

        print("─" * 70)

    def _validate_numbered_calc_addition(
        self, sequence: List[str], new_calc: str
    ) -> bool:
        """Validate addition of numbered calculation types"""
        # Parse the calculation type
        import re

        match = re.match(r"^([A-Z+]+)(\d*)$", new_calc)
        if not match:
            return False

        base_type = match.group(1)

        # Check dependencies based on base type
        if base_type == "SP":
            # SP can be added at any time (no dependencies)
            return True
        elif base_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]:
            # BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL need at least one SP or OPT
            return any(
                calc.startswith("SP") or calc.startswith("OPT") for calc in sequence
            )
        elif base_type == "FREQ":
            # FREQ needs at least one OPT
            return any(calc.startswith("OPT") for calc in sequence)
        elif base_type == "OPT":
            # OPT can always be added
            return True

        return False

    def validate_calc_addition(
        self, current_sequence: List[str], new_calc: str
    ) -> bool:
        """Validate that a calculation can be added to the sequence"""
        dependencies = self.calc_types[new_calc]["depends_on"]

        for dep in dependencies:
            if dep not in current_sequence:
                return False

        return True

    def show_workflow_help(self):
        """Show workflow design help"""
        print("\nWorkflow Help:")
        print("Common patterns:")
        print("  Basic: OPT")
        print("  Single point: SP")
        print("  Electronic: OPT SP  or  SP BAND DOSS")
        print("  Analysis: OPT SP BAND DOSS")
        print("  Complete: OPT SP BAND DOSS FREQ")
        print("  Transport: OPT SP TRANSPORT")
        print("  Charge analysis: OPT SP CHARGE+POTENTIAL")
        print("  Double opt: OPT OPT2 SP")
        print("\nAdvanced patterns:")
        print("  SP-first: SP BAND DOSS")
        print("  Multi-stage: OPT SP BAND DOSS OPT2 OPT3 SP2 BAND2 DOSS2 FREQ")
        print("  Combined analysis: OPT SP BAND DOSS TRANSPORT CHARGE+POTENTIAL")
        print("  Iterative opt: OPT OPT2 OPT3 SP")
        print("  Multiple properties: OPT SP BAND DOSS BAND2 DOSS2")
        print("\nDependencies:")
        for calc, info in self.calc_types.items():
            if info["depends_on"]:
                print(f"  {calc} requires: {', '.join(info['depends_on'])}")
        print(
            "\nNOTE: Numbered calculations (OPT2, SP2, etc.) are automatically assigned"
        )
        print("      when you add multiple calculations of the same type.")

    def configure_workflow_steps(
        self, sequence: List[str], has_cifs: bool
    ) -> Dict[str, Dict[str, Any]]:
        """Configure settings for each workflow step"""
        print(f"\nStep 4: Configure Workflow Steps")
        print("-" * 40)

        step_configs = {}

        for i, calc_type in enumerate(sequence):
            # Parse calc_type to get base type for looking up info
            base_type = calc_type.rstrip("0123456789") or calc_type
            if base_type in self.calc_types:
                calc_name = self.calc_types[base_type]["name"]
            else:
                calc_name = calc_type
            print(f"\nConfiguring {calc_type} ({calc_name}):")

            if (calc_type == "OPT" or calc_type == "SP") and i == 0 and has_cifs:
                print(
                    f"  Using CIF conversion configuration for first {calc_type} step"
                )
                step_configs[f"{calc_type}_1"] = {"source": "cif_conversion"}

            elif (calc_type == "OPT" or calc_type == "SP") and i == 0 and not has_cifs:
                print(f"  Using existing D12 files for first {calc_type} step")
                print("    Settings from D12 files will be used as-is")
                print(
                    "    This includes: functional, basis set, tolerances, grid, etc."
                )
                step_configs[f"{calc_type}_1"] = {"source": "existing_d12"}

            elif calc_type == "OPT2":
                # OPT2 is always configured via CRYSTALOptToD12.py
                config = self.configure_optimization_step(calc_type, i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type == "OPT" and i > 0:
                # This shouldn't happen - subsequent OPTs should be OPT2, OPT3, etc.
                print(
                    f"  Warning: Found duplicate OPT at position {i + 1}. This should be OPT2."
                )
                config = self.configure_optimization_step(calc_type, i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("OPT") and calc_type[3:].isdigit():
                # Handle OPT3, OPT4, etc.
                config = self.configure_optimization_step(calc_type, i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type == "SP" or (
                calc_type.startswith("SP") and calc_type[2:].isdigit()
            ):
                # Single point calculation (SP, SP2, SP3, etc.)
                config = self.configure_single_point_step(calc_type, i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("BAND"):
                # Band structure calculations (BAND, BAND2, etc.)
                config = self.configure_analysis_step("BAND", i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("DOSS"):
                # DOS calculations (DOSS, DOSS2, etc.)
                config = self.configure_analysis_step("DOSS", i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("FREQ"):
                # Frequency calculations (FREQ, FREQ2, FREQ3, etc.)
                config = self.configure_frequency_step(calc_type, i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("TRANSPORT"):
                # Transport calculations (TRANSPORT, TRANSPORT2, etc.)
                config = self.configure_analysis_step("TRANSPORT", i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            elif calc_type.startswith("CHARGE+POTENTIAL"):
                # Charge+Potential calculations
                config = self.configure_analysis_step("CHARGE+POTENTIAL", i + 1)
                step_configs[f"{calc_type}_{i + 1}"] = config

            # Configure SLURM scripts for this step
            slurm_config = self.configure_slurm_scripts(calc_type, i + 1)
            step_configs[f"{calc_type}_{i + 1}"]["slurm_config"] = slurm_config

        return step_configs

    def configure_optimization_step(
        self, calc_type: str, step_num: int
    ) -> Dict[str, Any]:
        """Configure optimization calculation step"""
        print(f"  Configuring {calc_type} step {step_num}")

        print(f"    Choose {calc_type} customization level:")
        print(f"      0: Use sensible defaults")
        print(f"         - Type: FULLOPTG (optimize both atoms and cell)")
        print(f"         - TOLDEG: 3.0E-5, TOLDEX: 1.2E-4, TOLDEE: 7")
        print(f"         - MAXCYCLE: 800, Method/basis: inherited from previous step")
        print(f"      1: Basic (optimization type + tolerances)")
        print(f"         - Configure: FULLOPTG vs ATOMSONLY, convergence criteria")
        print(f"         - Time impact: Can reduce optimization time by 30-50%")
        print(f"      2: Advanced (method + basis set modifications)")
        print(f"         - Configure: Change functional/basis from initial calculation")
        print(f"         - Use case: Re-optimize with better method")
        print(f"      3: Expert (full CRYSTALOptToD12.py integration)")
        print(f"         - Configure: All CRYSTAL keywords interactively")
        print(f"         - Use case: Complex optimizations, constraints, special settings")

        while True:
            try:
                level = int(input("    Enter level (0-3): ").strip())
                if level in [0, 1, 2, 3]:
                    break
                print("    Please enter 0, 1, 2, or 3")
            except ValueError:
                print("    Please enter a valid number")

        if level == 0:
            # Use sensible defaults
            config = {
                "calculation_type": "OPT",
                "optimization_type": "FULLOPTG",
                "optimization_settings": {
                    "TOLDEG": 0.00003,
                    "TOLDEX": 0.00012,
                    "TOLDEE": 7,
                    "MAXCYCLE": 800,
                },
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True,
                "customization_level": 0,
            }
            print(f"\n    Using default {calc_type} configuration")
        else:
            # Get detailed configuration for levels 1-3
            config = {
                "calculation_type": "OPT",
                "source": "CRYSTALOptToD12.py",
                "customization_level": level,
            }
            
            if level == 1:
                # Basic configuration: optimization type and tolerances
                config.update(self._get_basic_opt_config())
            elif level == 2:
                # Advanced configuration: method/basis modifications
                config.update(self._get_advanced_opt_config())
            elif level == 3:
                # Expert configuration: full interactive setup
                config.update(self._get_expert_opt_config(calc_type, step_num))

            # Show summary of selected configuration
            if config.get("customization_level") == 1:
                print(f"\n    {calc_type} configuration summary:")
                print(f"      - Type: {config.get('optimization_type', 'FULLOPTG')}")
                opt_settings = config.get("optimization_settings", {})
                print(f"      - TOLDEG: {opt_settings.get('TOLDEG', 3e-5):.1E}")
                print(f"      - TOLDEX: {opt_settings.get('TOLDEX', 1.2e-4):.1E}")
                print(f"      - TOLDEE: {opt_settings.get('TOLDEE', 7)}")
                print(f"      - MAXCYCLE: {opt_settings.get('MAXCYCLE', 800)}")
                if config.get("custom_tolerances"):
                    tol = config["custom_tolerances"]
                    if tol.get("TOLINTEG"):
                        print(f"      - TOLINTEG: {tol['TOLINTEG']}")
                    if tol.get("TOLDEE"):
                        print(f"      - SCF TOLDEE: {tol['TOLDEE']}")

        return config

    def configure_single_point_step(
        self, calc_type: str = "SP", step_num: int = 2
    ) -> Dict[str, Any]:
        """Configure single point calculation with customization levels"""
        print(f"  Configuring {calc_type} calculation")

        print("    Choose SP customization level:")
        print("      0: Default (inherit all settings from previous OPT)")
        print("         - Best for: Energy comparison at same level of theory")
        print("      1: Basic (modify method/basis set)")
        print("         - Best for: Testing different functionals/basis sets")
        print("         - Example: OPT with PBE → SP with B3LYP for better energies")
        print("      2: Advanced (detailed SCF/convergence settings)")
        print("         - Best for: Difficult convergence, custom requirements")
        print("      3: Expert (full CRYSTALOptToD12.py integration)")
        print("         - Best for: Complete control over all parameters")

        while True:
            try:
                level = int(input("    Enter level (0-3): ").strip())
                if level in [0, 1, 2, 3]:
                    break
                print("    Please enter 0, 1, 2, or 3")
            except ValueError:
                print("    Please enter a valid number")

        config = {
            "calculation_type": "SP",
            "source": "CRYSTALOptToD12.py",
            "customization_level": level,
        }

        if level == 0:
            # Default: inherit all settings
            config["inherit_settings"] = True
        elif level == 1:
            # Basic: method/basis modifications
            config.update(self._get_basic_sp_config())
        elif level == 2:
            # Advanced: detailed settings
            config.update(self._get_advanced_sp_config())
        elif level == 3:
            # Expert: full customization
            config.update(self._get_expert_sp_config(calc_type, step_num))

        return config

    def _get_basic_sp_config(self) -> Dict[str, Any]:
        """Get basic SP configuration"""
        print("\n    Basic SP Setup:")
        print("    (Single point calculation with modified method/basis)")

        config = {"inherit_geometry": True, "inherit_settings": False}

        # Method modifications - streamlined
        config["method_modifications"] = self._get_method_modifications()

        # Basis set modifications - streamlined
        config["basis_modifications"] = self._get_basis_modifications()

        # Ask about tight convergence
        print("\n    Convergence settings for single point energy:")
        print("      Standard: Good for energy differences (~0.001 Ha accuracy)")
        print("      Tight: Required for accurate absolute energies (~0.00001 Ha)")
        print("      📊 Resource impact of tight convergence:")
        print("        - Time: +20-50% more SCF iterations")
        print("        - Memory: No significant change")
        print("        - Use for: Benchmark calculations, basis set comparisons")

        use_tight = yes_no_prompt("    Use tight convergence?", "no")
        if use_tight:
            print("      Applying tight convergence tolerances:")
            print("        - TOLINTEG: 9 9 9 11 38 (high accuracy integrals)")
            print("        - TOLDEE: 11 (SCF convergence to 10^-11 Ha)")
            config["tolerance_modifications"] = {
                "custom_tolerances": {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}
            }

        return config

    def _get_advanced_sp_config(self) -> Dict[str, Any]:
        """Get advanced SP configuration"""
        config = self._get_basic_sp_config()

        print("\n    Advanced SP Setup:")

        # SCF modifications
        modify_scf = yes_no_prompt("    Modify SCF convergence settings?", "no")
        if modify_scf:
            config["scf_modifications"] = self._get_scf_modifications()

        # Grid modifications
        modify_grid = yes_no_prompt("    Modify DFT integration grid?", "no")
        if modify_grid:
            config["grid_modifications"] = self._get_grid_modifications()

        return config

    def _get_expert_sp_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Get expert SP configuration"""
        print(f"\n    Expert {calc_type} Setup:")

        # Skip individual vs shared question for initial step (D12s already have settings)
        if step_num == 1:
            print("    Initial step - modifying existing D12 settings")
        else:
            # Ask if user wants per-material configs (to preserve symmetry)
            print("    Material Configuration Strategy:")
            print("")
            print("    1. Individual material handling (STRONGLY RECOMMENDED)")
            print("       Process: Interactive setup (1x) → Automatic per-material optimization")
            print("       Preserves: Symmetry, k-points, cell parameters, origin settings")
            print("       Generates: Unique configuration file per material")
            print("       ")
            print("    2. Batch uniform handling (USE WITH CAUTION)")
            print("       Process: Interactive setup (1x) → Same config for all")
            print("       WARNING: Forces all materials to use first material's symmetry")
            print("       Risk: Incorrect k-points, wrong space groups, failed calculations")

            config_choice = (
                input("    Choose configuration mode (1/2) [1]: ").strip() or "1"
            )

            if config_choice == "1":
                return self._get_per_material_expert_config(calc_type, step_num)

        # Copy required scripts early so we can use them
        self._copy_required_scripts_for_expert_mode()

        # Try to find a real D12 file from previous steps or current directory
        real_d12 = None
        
        # Search for D12 files in likely locations
        search_patterns = [
            f"workflow_outputs/*/step_{step_num-1:03d}_*/*/*.d12",
            f"workflow_inputs/step_{step_num:03d}_*/*.d12",
            "*.d12",
        ]
        
        for pattern in search_patterns:
            found_files = list(self.work_dir.glob(pattern))
            if found_files:
                real_d12 = found_files[0]
                print(f"    Found D12 file for configuration: {real_d12.name}")
                break
                
        # Run CRYSTALOptToD12.py interactively NOW during planning
        expert_config = self._run_interactive_crystal_opt_config("SP", real_d12)

        if expert_config:
            print(f"    ✅ Expert SP configuration completed successfully")
            return expert_config
        else:
            print(
                f"    ❌ Expert SP configuration failed, falling back to advanced mode"
            )
            return self._get_advanced_sp_config()

    def _get_basis_modifications(self) -> Dict[str, Any]:
        """Get basis set modification settings"""
        modifications = {}

        print("\n      Select basis set (will inherit from previous if unchanged):")
        print("        1: Keep current basis set")
        print("        2: POB-TZVP-REV2 (high quality triple-zeta, recommended)")
        print("        3: POB-TZVP (standard triple-zeta)")
        print("        4: def2-TZVP (alternative triple-zeta)")
        print("        5: POB-DZVP-REV2 (double-zeta, faster)")
        print("        6: STO-3G (minimal, very fast)")
        print("        7: Custom basis set")

        while True:
            basis_choice = input("      Choose basis set (1-7) [1]: ").strip() or "1"
            if basis_choice in ["1", "2", "3", "4", "5", "6", "7"]:
                break
            print("      Please enter a number from 1 to 7")

        if basis_choice == "1":
            modifications["inherit_basis"] = True
        else:
            basis_map = {
                "2": "POB-TZVP-REV2",
                "3": "POB-TZVP",
                "4": "def2-TZVP",
                "5": "POB-DZVP-REV2",
                "6": "STO-3G",
            }

            if basis_choice == "7":
                custom_basis = input("      Enter custom basis set name: ").strip()
                if custom_basis:
                    modifications["new_basis"] = custom_basis
            else:
                modifications["new_basis"] = basis_map[basis_choice]

        return modifications

    def _get_scf_modifications(self) -> Dict[str, Any]:
        """Get SCF modification settings"""
        modifications = {}

        # TOLDEE
        print("\n      SCF convergence threshold (TOLDEE):")
        print("        Current/inherited: 7 (default)")
        print(
            "        Common values: 7 (standard), 8 (tighter), 9 (very tight), 10+ (ultra-tight)"
        )
        toldee = input("      New TOLDEE value [keep current]: ").strip()
        if toldee:
            try:
                modifications["TOLDEE"] = int(toldee)
            except ValueError:
                print("      Invalid TOLDEE, keeping current")

        # FMIXING
        print("\n      SCF mixing factor (FMIXING):")
        print("        Current/inherited: 30 (default)")
        print("        Common values: 30 (standard), 20 (more stable), 50 (aggressive)")
        fmixing = input("      New FMIXING value [keep current]: ").strip()
        if fmixing:
            try:
                modifications["FMIXING"] = int(fmixing)
            except ValueError:
                print("      Invalid FMIXING, keeping current")

        return modifications

    def _get_grid_modifications(self) -> Dict[str, Any]:
        """Get DFT grid modification settings"""
        modifications = {}

        print("\n      DFT integration grid (accuracy vs speed):")
        print(
            "        Current/default: XLGRID (CRYSTAL23 default for most functionals)"
        )
        print("        1: Keep current grid")
        print("        2: XLGRID - Extra large grid (75,974 points/atom) [default]")
        print("        3: XXLGRID - Extra extra large (99,1454 points/atom)")
        print("        4: LGRID - Large grid (75,434 points/atom)")
        print("        5: DEFAULT - Standard CRYSTAL grid")
        print("        6: OLDGRID - Legacy grid from CRYSTAL09 (55,434 points/atom)")
        print(
            "        7: XXXLGRID - Ultra large (150,1454 points/atom) for high accuracy"
        )
        print(
            "        8: HUGEGRID - Huge grid (300,1454 points/atom) for SCAN functional"
        )

        while True:
            grid_choice = input("      Choose grid (1-8) [1]: ").strip() or "1"
            if grid_choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                break
            print("      Please enter a number from 1 to 8")

        if grid_choice != "1":
            grid_map = {
                "2": "XLGRID",
                "3": "XXLGRID",
                "4": "LGRID",
                "5": "DEFAULT",
                "6": "OLDGRID",
                "7": "XXXLGRID",
                "8": "HUGEGRID",
            }
            modifications["new_grid"] = grid_map[grid_choice]

        return modifications

    def configure_analysis_step(self, calc_type: str, step_num: int = None) -> Dict[str, Any]:
        """Configure D3 property calculations (BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL)"""
        print(f"  Configuring {calc_type} calculation")
        
        # All D3 calculations now use CRYSTALOptToD3.py
        config = {
            "calculation_type": calc_type,
            "source": "CRYSTALOptToD3.py",
            "requires_wavefunction": True,
            "d3_calculation": True,  # Flag to indicate this is a D3 calculation
        }
        
        # Ask for customization level
        print(f"\n  {calc_type} Calculation Customization Level:")
        print("    1. Basic (use sensible defaults)")
        print("    2. Advanced (customize key parameters)")
        print("    3. Expert (full CRYSTALOptToD3.py integration)")
        
        while True:
            try:
                level = int(input("\n  Select customization level (1-3): "))
                if 1 <= level <= 3:
                    break
                print("  Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("  Invalid input. Please enter a number.")
        
        if level == 1:
            # Basic - use default configurations
            config["d3_config_mode"] = "basic"
            config["d3_config"] = self._get_basic_d3_config(calc_type)
            
        elif level == 2:
            # Advanced - customize key parameters
            config["d3_config_mode"] = "advanced"
            config["d3_config"] = self._get_advanced_d3_config(calc_type)
            
        else:
            # Expert - full interactive configuration
            config["d3_config_mode"] = "expert"
            config["expert_mode"] = True
            config["interactive_setup"] = True
            print("\n  Expert mode: Full interactive configuration")
            
            # Ask whether to create individual configs per material
            print("\n  Expert Setup:")
            print("  Material Configuration Strategy:")
            print("")
            print("  1. Individual material handling (STRONGLY RECOMMENDED)")
            print("     Process: Interactive setup (1x) → Automatic per-material optimization")
            print("     Preserves: Symmetry, k-points, cell parameters, origin settings")
            print("     Generates: Unique configuration file per material")
            print("     ")
            print("  2. Batch uniform handling (USE WITH CAUTION)")
            print("     Process: Interactive setup (1x) → Same config for all")
            print("     WARNING: Forces all materials to use first material's symmetry")
            print("     Risk: Incorrect k-points, wrong space groups, failed calculations")
            
            config_mode = input("  Choose configuration mode (1/2) [1]: ").strip() or "1"
            
            if config_mode == "1":
                # Individual configs per material
                config["per_material_config"] = True
                print("\n  Individual configuration mode selected")
                
                # Create per-material expert configurations immediately
                per_material_config = self._get_per_material_expert_d3_config(calc_type, step_num)
                if per_material_config:
                    print(f"  ✅ Per-material {calc_type} configurations created successfully")
                    config.update(per_material_config)
                else:
                    print(f"  ❌ Per-material {calc_type} configuration failed, falling back to basic mode")
                    config["d3_config_mode"] = "basic"
                    config["d3_config"] = self._get_basic_d3_config(calc_type)
            else:
                # Single config for all materials
                config["per_material_config"] = False
                print("\n  Shared configuration mode selected")
                print("  Preparing to launch interactive configuration...")
                
                # Run CRYSTALOptToD3.py interactively NOW during planning
                expert_config = self._run_interactive_d3_config(calc_type, step_num)
                if expert_config:
                    print(f"  ✅ Expert {calc_type} configuration completed successfully")
                    config.update(expert_config)
                else:
                    print(f"  ❌ Expert {calc_type} configuration failed, falling back to basic mode")
                    config["d3_config_mode"] = "basic"
                    config["d3_config"] = self._get_basic_d3_config(calc_type)
        
        return config
    
    def _get_basic_d3_config(self, calc_type: str) -> Dict[str, Any]:
        """Get basic D3 configuration with sensible defaults"""
        configs = {
            "BAND": {
                "calculation_type": "BAND",
                "path": "auto",
                "bands": "auto", 
                "shrink": "auto",
                "labels": "auto",
                "auto_path": True
            },
            "DOSS": {
                "calculation_type": "DOSS",
                "npoints": 1000,
                "band": "all",
                "projection_type": 0,  # Total DOS only
                "e_range": [-20, 20]
            },
            "TRANSPORT": {
                "calculation_type": "TRANSPORT",
                "temperature_range": [100, 800, 50],
                "mu_range": [-2.0, 2.0, 0.01],
                "mu_reference": "fermi",
                "mu_range_type": "auto_fermi",
                "tdf_range": [-5.0, 5.0, 0.01],
                "relaxation_time": 10
            },
            "CHARGE+POTENTIAL": {
                "calculation_type": "CHARGE+POTENTIAL",
                "option_type": 6,  # 3D grid
                "mapnet": [100, 100, 100],
                "output_format": "GAUSSIAN"
            }
        }
        
        config = configs.get(calc_type, {})
        print(f"\n  Using basic {calc_type} configuration:")
        
        if calc_type == "BAND":
            print("    - Automatic k-path detection based on space group")
            print("    - All available bands included")
            print("    - Automatic SHRINK factor from parent calculation")
            
        elif calc_type == "DOSS":
            print("    - Total DOS only (no projections)")
            print("    - 1000 energy points")
            print("    - Energy range: -20 to 20 eV")
            
        elif calc_type == "TRANSPORT":
            print("    - Chemical potential range: -2 to +2 eV relative to Fermi")
            print("    - Temperature range: 100-800 K")
            print("    - Constant relaxation time: 10 fs")
            
        elif calc_type == "CHARGE+POTENTIAL":
            print("    - 3D charge density and electrostatic potential grids")
            print("    - 100x100x100 grid points")
            print("    - Gaussian-compatible output format (cube files)")
            
        return config
    
    def _get_advanced_d3_config(self, calc_type: str) -> Dict[str, Any]:
        """Get advanced D3 configuration with user customization"""
        config = {"calculation_type": calc_type}
        
        if calc_type == "BAND":
            print("\n  Advanced BAND configuration:")
            
            # Path configuration
            print("\n  K-point path:")
            print("    1. Automatic (SeeK-path based on space group)")
            print("    2. Custom path specification")
            
            path_choice = input("  Select path option [1]: ").strip() or "1"
            
            if path_choice == "1":
                config["path"] = "auto"
                config["auto_path"] = True
                
                # Path format selection
                print("\n  Path format for automatic k-path:")
                print("    1. High-symmetry labels (CRYSTAL-compatible subset)")
                print("    2. K-point vectors (fractional coordinates)")
                print("    3. Literature path with vectors (comprehensive)")
                print("    4. SeeK-path full paths (extended Bravais lattice notation)")
                
                format_choice = input("  Select format [1]: ").strip() or "1"
                format_map = {
                    "1": "labels",
                    "2": "vectors",
                    "3": "literature",
                    "4": "seekpath_full"
                }
                config["path_format"] = format_map.get(format_choice, "labels")
                config["labels"] = "auto" if config["path_format"] == "labels" else "none"
            else:
                print("  Custom path configuration selected")
                print("  You will specify the path during execution")
                config["custom_path"] = True
            
            # Band selection
            use_all = input("\n  Use all bands? [Y/n]: ").strip().lower()
            if use_all == 'n':
                config["band"] = input("  Band range (e.g., '1-50' or 'all'): ").strip()
            else:
                config["bands"] = "auto"
            
            # Points per segment
            npoints = input("  Points per k-path segment [200]: ").strip()
            config["npoints"] = int(npoints) if npoints else 200
            
        elif calc_type == "DOSS":
            print("\n  Advanced DOSS configuration:")
            
            # Projection type
            print("\n  DOS projection type:")
            print("    0. Total DOS only")
            print("    1. Atom contributions") 
            print("    2. Shell contributions")
            print("    3. Atomic orbital contributions")
            print("    4. Shell + AO contributions")
            
            proj_type = input("  Select projection type [4]: ").strip() or "4"
            config["projection_type"] = int(proj_type)
            
            if config["projection_type"] > 0:
                # Auto projections based on basis set
                auto_proj = input("  Auto-generate projections from basis set? [Y/n]: ").strip().lower()
                if auto_proj != 'n':
                    config["project_orbital_types"] = True
                    print("  ✓ Projections will be generated automatically")
                else:
                    print("  Manual projection configuration selected")
                    config["manual_projections"] = True
            
            # Energy range
            e_min = input("\n  Energy minimum (eV) [-20]: ").strip()
            e_max = input("  Energy maximum (eV) [20]: ").strip()
            config["e_range"] = [
                float(e_min) if e_min else -20,
                float(e_max) if e_max else 20
            ]
            
            # Number of points
            npoints = input("  Number of energy points [2000]: ").strip()
            config["npoints"] = int(npoints) if npoints else 2000
            
        elif calc_type == "TRANSPORT":
            print("\n  Advanced TRANSPORT configuration:")
            
            # Chemical potential reference
            print("\n  Chemical potential reference:")
            print("    1. Relative to Fermi energy (automatic)")
            print("    2. Relative to VBM (manual)")
            print("    3. Absolute values")
            
            mu_ref = input("  Select reference [1]: ").strip() or "1"
            
            if mu_ref == "1":
                config["mu_reference"] = "fermi"
                config["mu_range_type"] = "auto_fermi"
                mu_min = input("  Min μ relative to Fermi (eV) [-2]: ").strip()
                mu_max = input("  Max μ relative to Fermi (eV) [2]: ").strip()
                config["mu_range"] = [
                    float(mu_min) if mu_min else -2.0,
                    float(mu_max) if mu_max else 2.0,
                    0.01
                ]
            elif mu_ref == "2":
                config["mu_reference"] = "vbm"
                print("  You will need to specify VBM value during execution")
            else:
                config["mu_reference"] = "absolute"
                mu_min = input("  Min μ absolute (eV): ").strip()
                mu_max = input("  Max μ absolute (eV): ").strip()
                config["mu_range"] = [float(mu_min), float(mu_max), 0.01]
            
            # Temperature range
            t_min = input("\n  Min temperature (K) [100]: ").strip()
            t_max = input("  Max temperature (K) [800]: ").strip()
            t_step = input("  Temperature step (K) [50]: ").strip()
            config["temperature_range"] = [
                int(t_min) if t_min else 100,
                int(t_max) if t_max else 800,
                int(t_step) if t_step else 50
            ]
            
            # Relaxation time
            relax = input("  Relaxation time (fs) [10]: ").strip()
            config["relaxation_time"] = float(relax) if relax else 10
            
        elif calc_type == "CHARGE+POTENTIAL":
            print(f"\n  Advanced {calc_type} configuration:")
            
            # Output type
            print("\n  Output type:")
            print("    1. 3D grid (for visualization)")
            print("    2. 2D plane")
            print("    3. 1D line")
            print("    4. Points at atomic positions")
            
            out_type = input("  Select output type [1]: ").strip() or "1"
            
            type_map = {"1": 6, "2": 3, "3": 2, "4": 7}
            config["option_type"] = type_map[out_type]
            
            if out_type == "1":
                # 3D grid
                nx = input("  Grid points in x [100]: ").strip()
                ny = input("  Grid points in y [100]: ").strip()
                nz = input("  Grid points in z [100]: ").strip()
                config["mapnet"] = [
                    int(nx) if nx else 100,
                    int(ny) if ny else 100,
                    int(nz) if nz else 100
                ]
            elif out_type in ["2", "3"]:
                print("  Plane/line parameters will be configured during execution")
                config["requires_geometry_input"] = True
            
            # Output format
            print("\n  Output format:")
            print("    1. GAUSSIAN (cube file)")
            print("    2. XCRYSDEN")
            print("    3. Standard CRYSTAL")
            
            fmt_choice = input("  Select format [1]: ").strip() or "1"
            format_map = {"1": "GAUSSIAN", "2": "XCRYSDEN", "3": "CRYSTAL"}
            config["output_format"] = format_map[fmt_choice]
        
        return config

    def configure_frequency_step(
        self, calc_type: str = "FREQ", step_num: int = 2
    ) -> Dict[str, Any]:
        """Configure frequency calculation with customization levels"""
        print(f"  Configuring {calc_type} calculation")
        print("\n  ⚠️  Note: To calculate IR or Raman spectra, use Advanced or Expert mode")
        print("  Basic mode provides frequencies for thermodynamics only")

        # Ask for customization level
        print("\n  Frequency Calculation Customization Level:")
        print("    1. Basic (use sensible defaults)")
        print("    2. Advanced (customize key parameters)")
        print("    3. Expert (full control)")

        while True:
            try:
                level = int(input("\n  Select customization level (1-3): "))
                if 1 <= level <= 3:
                    break
                print("  Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("  Invalid input. Please enter a number.")

        if level == 1:
            # Basic - use defaults
            print("\n  Using simple frequency calculation defaults:")
            print("    - Mode: FREQCALC (gamma point vibrational analysis)")
            print("    - IR intensities: No (faster calculation)")
            print("    - Raman intensities: No")
            print("    - High accuracy tolerances for frequencies:")
            print("      - TOLINTEG: 9 9 9 11 38")
            print("      - TOLDEE: 11 (SCF convergence 10^-11 Ha)")
            print("\n  📊 Resource estimate:")
            print("    - Time: ~1-3x optimization time")
            print("    - Memory: Similar to optimization")
            print("    - Quality: Good for thermodynamics, no spectroscopy")
            print("    - Numerical derivatives: 2-point (two displacements per atom)")
            print("    - Method/basis: inherited from optimized geometry")

            config = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {
                    "mode": "GAMMA",
                    "numderiv": 2,
                    "intensities": False,
                    "temperatures": [298.15],
                    "custom_tolerances": {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11},
                },
            }

        elif level == 2:
            # Advanced - comprehensive frequency settings
            print("\n    Advanced frequency calculation setup:")

            config = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {},
            }

            # Frequency calculation mode
            print("\n  Frequency calculation mode:")
            print("    1. Gamma point only (REQUIRED for IR/Raman spectra)")
            print("       - Fastest option, suitable for molecules and large cells")
            print("       - Provides: frequencies, ZPE, thermal corrections, entropy")
            print("       - Enables: IR and Raman intensity calculations")
            print("       - Time: ~1-3x optimization time")
            print("       - Memory: Similar to optimization")
            print(
                "    2. Phonon calculations (band structure, DOS, or custom k-points)"
            )
            print(
                "       - For solid-state phonon properties (NOT for IR/Raman spectra)"
            )
            print(
                "       - Options: Full dispersion, specific k-points, or custom paths"
            )
            print("       - Time: Varies (4-60x optimization time)")
            print("       - Memory: Scales with calculation type")
            print("       - Note: Cannot calculate molecular IR/Raman intensities")

            mode_choice = input("  Select mode [1]: ").strip() or "1"
            if mode_choice == "1":
                config["frequency_settings"]["mode"] = "GAMMA"
                print(
                    "\n  ✓ Gamma point mode selected - IR/Raman intensities available"
                )
            elif mode_choice == "2":
                # Phonon calculations - ask for sub-option
                print("\n  Phonon calculation type:")
                print("    1. Full dispersion with supercell (recommended)")
                print("       - Complete phonon band structure and DOS")
                print("       - Uses SCELPHONO for supercell generation")
                print("       - Time: ~4-20x optimization (depends on supercell size)")
                print("    2. Custom k-points only")
                print("       - Calculate at specific k-points without full dispersion")
                print("       - Faster for targeted analysis")
                print("       - Time: Proportional to number of k-points")
                print("    3. High-symmetry points only")
                print("       - Just critical points (Gamma, X, M, etc.)")
                print("       - Quick check of key phonon frequencies")
                print("       - Time: ~2-4x optimization")

                phonon_type = input("  Select phonon type [1]: ").strip() or "1"

                if phonon_type == "1":
                    config["frequency_settings"]["mode"] = "DISPERSION"
                    config["frequency_settings"]["dispersion"] = True
                    print("\n  ✓ Full phonon dispersion mode selected")
                    print("  ⚠️  Note: IR/Raman intensities NOT available in this mode")
                    print("  For molecular spectra, use gamma point mode instead")

                    # Ask for output type
                    print("\n  Phonon calculation output type:")
                    print("    1. Phonon band structure")
                    print("    2. Phonon density of states")
                    print("    3. Both band structure and DOS")
                    
                    output_type = input("  Select output type [1]: ").strip() or "1"
                    
                    if output_type == "1":
                        config["frequency_settings"]["calculation_type"] = "bands"
                        print("  ✓ Phonon band structure selected")
                    elif output_type == "2":
                        config["frequency_settings"]["calculation_type"] = "dos"
                        print("  ✓ Phonon density of states selected")
                    elif output_type == "3":
                        config["frequency_settings"]["calculation_type"] = "both"
                        print("  ✓ Both phonon bands and DOS selected")
                    else:
                        config["frequency_settings"]["calculation_type"] = "bands"
                        print("  Invalid choice - defaulting to band structure")

                    # Supercell for phonon calculation
                    print("\n  Supercell for phonon calculation (SCELPHONO):")
                    print("    Larger supercells = better accuracy but much higher cost")
                    print("    1. Default 2x2x2 (8x more atoms)")
                    print("       - Time: ~4-8x optimization time")
                    print("       - Memory: ~8x optimization memory")
                    print("       - Quality: Good for most phonon properties")
                    print("    2. Custom expansion factors")
                    print(
                        "       - 3x3x3: ~10-20x time, ~27x memory, better for soft modes"
                    )
                    print("       - 4x4x4: ~30-60x time, ~64x memory, publication quality")
                    print("    Note: For 2D materials, use 2x2x1 or 3x3x1")
                    supercell_choice = input("  Select supercell [1]: ").strip() or "1"
                    if supercell_choice == "2":
                        nx = input("    Expansion in x [2]: ").strip()
                        ny = input("    Expansion in y [2]: ").strip()
                        nz = input("    Expansion in z [2]: ").strip()
                        config["frequency_settings"]["scelphono"] = [
                            int(nx) if nx else 2,
                            int(ny) if ny else 2,
                            int(nz) if nz else 2,
                        ]
                    else:
                        config["frequency_settings"]["scelphono"] = [2, 2, 2]

                    # Fourier interpolation
                    use_interphess = (
                        input("\n  Use Fourier interpolation (INTERPHESS)? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                    if use_interphess != "n":
                        config["frequency_settings"]["interphess"] = {
                            "expand": [2, 2, 2],  # Default values
                            "print": 0,
                        }

                elif phonon_type == "2":
                    # Custom k-points
                    config["frequency_settings"]["mode"] = "CUSTOM"
                    config["frequency_settings"]["custom_kpoints"] = True
                    print("\n  ✓ Custom k-point mode selected")
                    print("  You'll specify k-points manually in the configuration")
                    print("  Example: Calculate at specific q-points for testing")

                    # Ask how many k-points
                    nkpoints = input(
                        "\n  Number of k-points to calculate [4]: "
                    ).strip()
                    config["frequency_settings"]["num_kpoints"] = (
                        int(nkpoints) if nkpoints else 4
                    )

                elif phonon_type == "3":
                    # High-symmetry points only
                    config["frequency_settings"]["mode"] = "DISPERSION"
                    config["frequency_settings"]["high_symmetry_only"] = True
                    config["frequency_settings"]["scelphono"] = [1, 1, 1]
                    print("\n  ✓ High-symmetry points mode selected")
                    print("  Will calculate phonons at critical points only")
                    print("  Typical points: Gamma, X, M, K, etc.")
                    print("  ⚠️  Note: IR/Raman intensities NOT available in this mode")

            else:
                # Invalid choice - default to gamma
                config["frequency_settings"]["mode"] = "GAMMA"
                print("\n  Invalid choice - defaulting to gamma point mode")

            # Numerical derivative method
            print("\n  Numerical derivative method:")
            print("    1: One displacement per atom (faster, less accurate)")
            print("       - Forward difference: (g(x+t)-g(x))/t where t=0.001 Å")
            print("       - Time: N atoms × 1 SCF per atom")
            print("       - Error: O(t), suitable for quick estimates")
            print("       - Quality: May miss soft modes, less accurate frequencies")
            print("    2: Two displacements per atom (recommended)")
            print("       - Central difference: (g(x+t)-g(x-t))/2t where t=0.001 Å")
            print("       - Time: N atoms × 2 SCF per atom (2x slower)")
            print("       - Error: O(t²), much more accurate")
            print("       - Quality: Reliable frequencies, better for publication")
            numderiv = input("  Select method (1 or 2) [2]: ").strip() or "2"
            config["frequency_settings"]["numderiv"] = int(numderiv)

            # Gamma point spectroscopy options
            if config["frequency_settings"]["mode"] == "GAMMA":
                print("\n  Gamma point calculation type:")
                print("    1. Pure frequencies (thermodynamics only)")
                print("       - Fastest: Just vibrational frequencies")
                print("       - Provides: ZPE, thermal corrections, entropy")
                print("       - No spectroscopy data")
                print("    2. IR spectroscopy")
                print("       - Always calculates IR intensities")
                print("       - Optional: Generate spectrum plot (IRSPEC)")
                print("       - Multiple methods available (Berry phase, Wannier, CPHF)")
                print("    3. Raman spectroscopy")
                print("       - Always calculates Raman activities")
                print("       - Optional: Generate spectrum plot (RAMSPEC)")
                print("       - Requires CPHF calculation")
                print("       - Significantly more expensive than IR")
                print("    4. Both IR and Raman")
                print("       - Always calculates both IR intensities and Raman activities")
                print("       - Optional: Generate spectrum plots (IRSPEC/RAMSPEC)")
                print("       - Complete vibrational spectroscopy")
                print("       - Most expensive option")
                
                gamma_type = input("\n  Select calculation type [1]: ").strip() or "1"
                
                if gamma_type == "1":
                    # Pure frequencies - no intensities
                    config["frequency_settings"]["intensities"] = False
                    config["frequency_settings"]["raman"] = False
                    print("\n  ✓ Pure frequency calculation selected")
                    print("  No IR or Raman intensities will be calculated")
                    
                elif gamma_type == "2":
                    # IR only
                    config["frequency_settings"]["intensities"] = True
                    config["frequency_settings"]["raman"] = False
                    print("\n  ✓ IR spectroscopy selected")
                    print("  Note: IR intensities are ALWAYS calculated when IR is selected")
                    print("        Spectrum plot generation is optional (asked later)")
                    
                    print("\n  IR intensity calculation method:")
                    print("    1. Berry phase (INTPOL) - Default, efficient")
                    print("       - Best for: Periodic solids, semiconductors, insulators")
                    print("       - Works well: Covalent materials, MOFs, zeolites")
                    print("       - Limitations: Requires insulating state")
                    print("       - Time: +10-20% over base frequency")
                    print("       - Memory: Minimal overhead")
                    print("    2. Wannier functions (INTLOC) - Localized approach")
                    print("       - Best for: Molecular crystals, ionic solids")
                    print("       - Works well: Systems with localized bonds/charges")
                    print("       - Limitations: Requires insulating state, higher memory")
                    print("       - Time: +20-30% over base frequency")
                    print("       - Memory: +10-20% overhead")
                    print("    3. CPHF (INTCPHF) - Most accurate, analytical")
                    print("       - Best for: Any material (metals, semiconductors, insulators)")
                    print("       - Works well: Small unit cells, high accuracy needed")
                    print("       - Benefits: Also enables Raman, most reliable")
                    print("       - Time: +50-100% over base frequency")
                    print("       - Memory: ~2x base requirement")
                    
                    print("\n  Note: Berry phase (1) is the default as it works well for most")
                    print("        periodic systems and has the best speed/accuracy balance")
                    ir_method = input("\n  Select method [1]: ").strip() or "1"
                    ir_methods = {"1": "BERRY", "2": "WANNIER", "3": "CPHF"}
                    config["frequency_settings"]["ir_method"] = ir_methods.get(ir_method, "BERRY")
                    
                    # CPHF settings if selected
                    if config["frequency_settings"]["ir_method"] == "CPHF":
                        max_iter = input("  CPHF max iterations [30]: ").strip()
                        config["frequency_settings"]["cphf_max_iter"] = int(max_iter) if max_iter else 30
                        tol = input("  CPHF convergence (10^-x) [6]: ").strip()
                        config["frequency_settings"]["cphf_tolerance"] = int(tol) if tol else 6
                    
                    # Ask about minimal IR
                    print("\n  Spectrum plot generation:")
                    print("    By default: Intensities only (no spectrum plot)")
                    print("    Optional: Generate broadened IR spectrum plot (IRSPEC)")
                    print("\n  Do you want to skip spectrum plot generation?")
                    print("    - YES (minimal): Only calculate intensities")
                    print("    - NO: Calculate intensities AND generate spectrum plot")
                    print("\n  Computational cost of spectrum generation:")
                    print("    - Minimal overhead (<1% additional time)")
                    print("    - Can be generated later from .out file if needed")
                    minimal_ir = input("  Skip IR spectrum plot (minimal mode)? [Y/n]: ").strip().lower()
                    if minimal_ir != "n":
                        config["frequency_settings"]["minimal_ir"] = True
                        
                elif gamma_type == "3":
                    # Raman only
                    config["frequency_settings"]["intensities"] = True  # Needed for CPHF
                    config["frequency_settings"]["raman"] = True
                    config["frequency_settings"]["ir_method"] = "CPHF"  # Required for Raman
                    print("\n  ✓ Raman spectroscopy selected")
                    print("  Note: Raman activities are ALWAYS calculated when Raman is selected")
                    print("        Spectrum plot generation is optional (asked later)")
                    print("        CPHF will be used (required for Raman)")
                    
                    # CPHF settings
                    max_iter = input("\n  CPHF max iterations [30]: ").strip()
                    config["frequency_settings"]["cphf_max_iter"] = int(max_iter) if max_iter else 30
                    tol = input("  CPHF convergence (10^-x) [6]: ").strip()
                    config["frequency_settings"]["cphf_tolerance"] = int(tol) if tol else 6
                    
                    # Ask about minimal Raman
                    print("\n  Spectrum plot generation:")
                    print("    By default: Activities only (no spectrum plot)")
                    print("    Optional: Generate broadened Raman spectrum plot (RAMSPEC)")
                    print("\n  Do you want to skip spectrum plot generation?")
                    print("    - YES (minimal): Only calculate activities")
                    print("    - NO: Calculate activities AND generate spectrum plot")
                    print("\n  Computational cost of spectrum generation:")
                    print("    - Minimal overhead (<1% additional time)")
                    print("    - Can be generated later from .out file if needed")
                    minimal = input("  Skip Raman spectrum plot (minimal mode)? [Y/n]: ").strip().lower()
                    if minimal != "n":
                        config["frequency_settings"]["minimal_raman"] = True
                        
                elif gamma_type == "4":
                    # Both IR and Raman
                    config["frequency_settings"]["intensities"] = True
                    config["frequency_settings"]["raman"] = True
                    config["frequency_settings"]["ir_method"] = "CPHF"  # Required for Raman
                    print("\n  ✓ Both IR and Raman spectroscopy selected")
                    print("  Note: IR intensities AND Raman activities are ALWAYS calculated")
                    print("        Spectrum plot generation is optional (asked later)")
                    print("        CPHF will be used (required for Raman)")
                    
                    # CPHF settings
                    max_iter = input("\n  CPHF max iterations [30]: ").strip()
                    config["frequency_settings"]["cphf_max_iter"] = int(max_iter) if max_iter else 30
                    tol = input("  CPHF convergence (10^-x) [6]: ").strip()
                    config["frequency_settings"]["cphf_tolerance"] = int(tol) if tol else 6
                    
                    # Ask about minimal options
                    print("\n  Spectrum plot generation options:")
                    print("  Note: Intensities/activities are ALWAYS calculated")
                    print("        You're choosing whether to also generate spectrum plots")
                    
                    print("\n  IR spectrum plot:")
                    print("    - Skip plot (minimal): Only intensities in .out file")
                    print("    - Generate plot: Intensities + broadened spectrum (IRSPEC)")
                    print("    - Cost: <1% additional time, can be done later")
                    minimal_ir = input("  Skip IR spectrum plot (minimal mode)? [Y/n]: ").strip().lower()
                    if minimal_ir != "n":
                        config["frequency_settings"]["minimal_ir"] = True
                        
                    print("\n  Raman spectrum plot:")
                    print("    - Skip plot (minimal): Only activities in .out file")
                    print("    - Generate plot: Activities + broadened spectrum (RAMSPEC)")
                    print("    - Cost: <1% additional time, can be done later")
                    minimal_raman = input("  Skip Raman spectrum plot (minimal mode)? [Y/n]: ").strip().lower()
                    if minimal_raman != "n":
                        config["frequency_settings"]["minimal_raman"] = True
                else:
                    # Default to pure frequencies
                    config["frequency_settings"]["intensities"] = False
                    config["frequency_settings"]["raman"] = False
                    print("\n  Invalid choice - defaulting to pure frequencies")
                    
                # Spectral generation (only ask if not minimal)
                if (config["frequency_settings"].get("intensities") or config["frequency_settings"].get("raman")) and \
                   not (config["frequency_settings"].get("minimal_ir", False) and config["frequency_settings"].get("minimal_raman", False)):
                    
                    # Only ask about spectra if we have non-minimal IR or Raman
                    has_ir_spectra = config["frequency_settings"].get("intensities") and not config["frequency_settings"].get("minimal_ir", False)
                    has_raman_spectra = config["frequency_settings"].get("raman") and not config["frequency_settings"].get("minimal_raman", False)
                    
                    if has_ir_spectra or has_raman_spectra:
                        if has_ir_spectra and has_raman_spectra:
                            spectra_prompt = "\n  Generate IR and Raman spectra plots? [y/N]: "
                        elif has_ir_spectra:
                            spectra_prompt = "\n  Generate IR spectrum plot? [y/N]: "
                        else:
                            spectra_prompt = "\n  Generate Raman spectrum plot? [y/N]: "
                            
                        gen_spectra = input(spectra_prompt).strip().lower()
                        if gen_spectra == "y":
                            if has_ir_spectra:
                                config["frequency_settings"]["irspec"] = True
                                width = input("  IR peak width (cm^-1) [10]: ").strip()
                                config["frequency_settings"]["spec_dampfac"] = float(width) if width else 10
                            if has_raman_spectra:
                                config["frequency_settings"]["ramspec"] = True
                                if has_ir_spectra:
                                    # Ask for Raman width only if different from IR
                                    use_same = input("  Use same peak width for Raman? [Y/n]: ").strip().lower()
                                    if use_same == "n":
                                        width = input("  Raman peak width (cm^-1) [10]: ").strip()
                                        config["frequency_settings"]["raman_dampfac"] = float(width) if width else 10
                                else:
                                    width = input("  Raman peak width (cm^-1) [10]: ").strip()
                                    config["frequency_settings"]["spec_dampfac"] = float(width) if width else 10
            else:
                # Not gamma point mode - no intensities possible
                config["frequency_settings"]["intensities"] = False
                config["frequency_settings"]["raman"] = False

            # Anharmonic calculations
            calc_anharm = (
                input("\n  Include anharmonic corrections? [y/N]: ").strip().lower()
            )
            if calc_anharm == "y":
                config["frequency_settings"]["anharmonic"] = True
                print("\n  Anharmonic calculation type:")
                print("    1. ANHARM (basic X-H stretches)")
                print("       - Time: ~5-10x base frequency time")
                print("       - Quality: Good for X-H stretches only")
                print("       - Use for: H-bonded systems, OH/NH/CH groups")
                print("    2. VSCF (Vibrational SCF)")
                print("       - Time: ~10-20x base frequency time")
                print("       - Memory: ~2-3x base requirement")
                print("       - Quality: Good for fundamental transitions")
                print("       - Use for: Moderately anharmonic systems")
                print("    3. VCI (Vibrational CI)")
                print("       - Time: ~30-50x base frequency time")
                print("       - Memory: ~4-5x base requirement")
                print("       - Quality: Includes overtones and combinations")
                print("       - Use for: Highly anharmonic systems, complete spectra")
                anharm_type = input("  Select type [1]: ").strip() or "1"
                anharm_types = {"1": "ANHARM", "2": "VSCF", "3": "VCI"}
                config["frequency_settings"]["anharm_type"] = anharm_types.get(
                    anharm_type, "ANHARM"
                )

            # Temperature list
            if config["frequency_settings"].get("mode") == "GAMMA":
                temp_input = input(
                    "\n  Temperatures (K) space-separated [298.15]: "
                ).strip()
                if temp_input:
                    config["frequency_settings"]["temperatures"] = [
                        float(t) for t in temp_input.split()
                    ]
                else:
                    config["frequency_settings"]["temperatures"] = [298.15]

            # Always use high accuracy tolerances for FREQ
            print("\n  Using high accuracy tolerances for frequency calculations:")
            print("    TOLINTEG: 9 9 9 11 38")
            print("    TOLDEE: 11")
            config["frequency_settings"]["custom_tolerances"] = {
                "TOLINTEG": "9 9 9 11 38",
                "TOLDEE": 11,
            }

        else:
            # Expert - run CRYSTALOptToD12.py interactively
            print("\n  Expert mode: Full interactive configuration")

            # Ask whether to create individual configs per material
            print("\n  Expert Setup:")
            print("  Material Configuration Strategy:")
            print("")
            print("  1. Individual material handling (STRONGLY RECOMMENDED)")
            print("     Process: Interactive setup (1x) → Automatic per-material optimization")
            print("     Preserves: Symmetry, k-points, cell parameters, origin settings")
            print("     Generates: Unique configuration file per material")
            print("     ")
            print("  2. Batch uniform handling (USE WITH CAUTION)")
            print("     Process: Interactive setup (1x) → Same config for all")
            print("     WARNING: Forces all materials to use first material's symmetry")
            print("     Risk: Incorrect k-points, wrong space groups, failed calculations")

            config_choice = (
                input("  Choose configuration mode (1/2) [1]: ").strip() or "1"
            )

            if config_choice == "1":
                return self._get_per_material_expert_config(calc_type, step_num)

            # Copy required scripts early
            self._copy_required_scripts_for_expert_mode()

            print("\n    Expert mode: Full interactive configuration")

            # Try to find a real D12 file from previous steps or current directory
            real_d12 = None
            
            # Search for D12 files in likely locations
            search_patterns = [
                f"workflow_outputs/*/step_*/*/*.d12",
                f"workflow_inputs/step_*/*.d12",
                "*.d12",
            ]
            
            for pattern in search_patterns:
                found_files = list(self.work_dir.glob(pattern))
                if found_files:
                    real_d12 = found_files[0]
                    print(f"    Found D12 file for configuration: {real_d12.name}")
                    break
                    
            # Run CRYSTALOptToD12.py interactively for FREQ configuration
            expert_config = self._run_interactive_crystal_opt_config("FREQ", real_d12)

            if expert_config:
                print("  ✅ Expert FREQ configuration completed successfully")
                return expert_config
            else:
                print("  ❌ Expert FREQ configuration failed, using advanced settings")
                # Fall back to advanced settings
                config = {
                    "calculation_type": "FREQ",
                    "source": "CRYSTALOptToD12.py",
                    "inherit_base_settings": True,
                    "frequency_settings": {
                        "mode": "GAMMA",
                        "numderiv": 2,
                        "intensities": True,
                        "ir_method": "CPHF",
                        "raman": True,
                        "cphf_max_iter": 30,
                        "cphf_tolerance": 6,
                        "temperatures": [298.15],
                        "custom_tolerances": {
                            "TOLINTEG": "12 12 12 12 24",
                            "TOLDEE": 12,
                        },
                    },
                }
                return config

        return config


        return config

    def _get_basic_opt_config(self) -> Dict[str, Any]:
        """Get basic optimization configuration"""
        print("\n    Basic Optimization Setup:")

        # Optimization type
        print("    Optimization type:")
        print("      1: FULLOPTG (optimize atoms and cell)")
        print("      2: ATOMSONLY (optimize atoms only)")
        print("      3: CELLONLY (optimize cell only)")

        opt_choice = (
            input("    Choose optimization type (1-3, default 1): ").strip() or "1"
        )
        opt_types = {"1": "FULLOPTG", "2": "ATOMSONLY", "3": "CELLONLY"}
        opt_type = opt_types.get(opt_choice, "FULLOPTG")

        # Enhanced tolerances for subsequent optimizations
        print("\n    Convergence settings:")
        print("    Standard convergence:")
        print("      - TOLDEG: 3.0E-5 (RMS gradient threshold)")
        print("      - TOLDEX: 1.2E-4 (RMS displacement threshold)")
        print("      - TOLDEE: 7 (energy convergence 10^-7 Ha)")
        print("      - MAXCYCLE: 800 (maximum optimization steps)")
        print("    Tighter convergence (recommended for refined optimization):")
        print("      - TOLDEG: 1.5E-5 (2x tighter gradient)")
        print("      - TOLDEX: 6.0E-5 (2x tighter displacement)")
        print("      - TOLDEE: 8 (10x tighter energy, 10^-8 Ha)")
        print("      - MAXCYCLE: 1000 (25% more steps allowed)")

        use_tight = yes_no_prompt(
            "    Use tighter convergence for refined optimization?", "yes"
        )

        if use_tight:
            opt_settings = {
                "TOLDEG": 1.5e-5,  # Tighter than default 3e-5
                "TOLDEX": 6e-5,  # Tighter than default 1.2e-4
                "TOLDEE": 8,  # Tighter than default 7
                "MAXCYCLE": 1000,  # More cycles for convergence
            }
        else:
            opt_settings = {
                "TOLDEG": 3e-5,
                "TOLDEX": 1.2e-4,
                "TOLDEE": 7,
                "MAXCYCLE": 800,
            }

        return {
            "optimization_type": opt_type,
            "optimization_settings": opt_settings,
            "inherit_base_settings": True,
        }

    def _get_advanced_opt_config(self) -> Dict[str, Any]:
        """Get advanced optimization configuration"""
        config = self._get_basic_opt_config()

        print("\n    Advanced Optimization Setup:")

        # Method modifications - always ask
        config["method_settings"] = self._get_method_modifications()

        # Basis set modifications
        config["basis_settings"] = self._get_basis_modifications()

        # Custom tolerances
        print("\n    Convergence tolerances:")
        config["custom_tolerances"] = self._get_custom_tolerances()

        return config

    def _get_expert_opt_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Get expert optimization configuration with full CRYSTALOptToD12.py integration"""
        print(f"\n    Expert {calc_type} Setup:")

        # Skip individual vs shared question for initial step (D12s already have settings)
        if step_num == 1:
            print("    Initial step - modifying existing D12 settings")
        else:
            # For subsequent OPT steps, ask if user wants per-material configs
            print("    Material Configuration Strategy:")
            print("")
            print("    1. Individual material handling (STRONGLY RECOMMENDED)")
            print("       Process: Interactive setup (1x) → Automatic per-material optimization")
            print("       Preserves: Symmetry, k-points, cell parameters, origin settings")
            print("       Generates: Unique configuration file per material")
            print("       ")
            print("    2. Batch uniform handling (USE WITH CAUTION)")
            print("       Process: Interactive setup (1x) → Same config for all")
            print("       WARNING: Forces all materials to use first material's symmetry")
            print("       Risk: Incorrect k-points, wrong space groups, failed calculations")

            config_choice = (
                input("    Choose configuration mode (1/2) [1]: ").strip() or "1"
            )

            if config_choice == "1":
                return self._get_per_material_expert_config(calc_type, step_num)

        # Copy required scripts early so we can use them
        self._copy_required_scripts_for_expert_mode()

        # Try to find a real D12 file from previous steps or current directory
        real_d12 = None
        
        # Search for D12 files in likely locations
        search_patterns = [
            f"workflow_outputs/*/step_{step_num-1:03d}_*/*/*.d12",
            f"workflow_inputs/step_{step_num:03d}_*/*.d12",
            "*.d12",
        ]
        
        for pattern in search_patterns:
            found_files = list(self.work_dir.glob(pattern))
            if found_files:
                real_d12 = found_files[0]
                print(f"    Found D12 file for configuration: {real_d12.name}")
                break
                
        # Run CRYSTALOptToD12.py interactively NOW during planning
        # For OPT2, OPT3 etc., we pass "OPT" as calc type to CRYSTALOptToD12.py
        crystal_calc_type = "OPT" if calc_type.startswith("OPT") else calc_type
        # Pass the full calc_type (e.g., OPT2, OPT3) instead of just "OPT"
        expert_config = self._run_interactive_crystal_opt_config(calc_type, real_d12)

        if expert_config:
            # Add step-specific information
            expert_config["step_num"] = step_num
            expert_config["workflow_calc_type"] = (
                calc_type  # Keep original OPT2, OPT3 etc.
            )
            print(f"    ✅ Expert {calc_type} configuration completed successfully")
            return expert_config
        else:
            print(
                f"    ❌ Expert {calc_type} configuration failed, falling back to advanced mode"
            )
            return self._get_advanced_opt_config()

    def _get_method_modifications(self) -> Dict[str, Any]:
        """Get DFT method modification settings"""
        modifications = {}

        # Functional change - always ask which functional to use
        print("      Select DFT functional (will inherit from previous if unchanged):")
        print("        1: Keep current functional")
        print("        2: PBE-D3 (GGA, fast and reliable)")
        print("        3: B3LYP-D3 (hybrid, good for organics)")
        print("        4: HSE06 (hybrid, accurate band gaps)")
        print("        5: PBE0 (hybrid, general purpose)")
        print("        6: M06-2X (meta-GGA, for kinetics)")
        print("        7: Custom functional (select from full list)")

        while True:
            func_choice = input("      Choose functional (1-7) [1]: ").strip() or "1"
            if func_choice in ["1", "2", "3", "4", "5", "6", "7"]:
                break
            print("      Please enter a number from 1 to 7")

        if func_choice == "1":
            modifications["inherit_functional"] = True
        elif func_choice == "7":
            # Custom functional - show full selection
            selected_functional = self._select_custom_functional()
            modifications["custom_functional"] = selected_functional

            # Ask about dispersion for non-3C methods
            if selected_functional and "3C" not in selected_functional.upper():
                print(f"\n      Add dispersion correction to {selected_functional}?")
                print(
                    "        Dispersion correction (D3) improves description of van der Waals interactions"
                )
                use_d3 = yes_no_prompt("      Use D3 dispersion correction?", "yes")
                if use_d3:
                    modifications["custom_functional"] = f"{selected_functional}-D3"
        else:
            # Map choices to functionals
            functional_map = {
                "2": "PBE",
                "3": "B3LYP",
                "4": "HSE06",
                "5": "PBE0",
                "6": "M06-2X",
            }
            modifications["new_functional"] = functional_map[func_choice]
            # Check if D3 dispersion should be added
            if func_choice in ["2", "3", "4", "5"]:
                modifications["use_dispersion"] = True

        return modifications

    def _select_custom_functional(self) -> str:
        """Show full functional selection menu"""
        print("\n      Select functional category:")
        print("        1: Hartree-Fock methods")
        print("        2: LDA (Local Density Approximation)")
        print("        3: GGA (Generalized Gradient Approximation)")
        print("        4: Hybrid (mix of HF and DFT)")
        print("        5: Meta-GGA (includes kinetic energy density)")
        print("        6: 3C Composite methods")

        category_map = {
            "1": "HF",
            "2": "LDA",
            "3": "GGA",
            "4": "HYBRID",
            "5": "MGGA",
            "6": "3C",
        }

        while True:
            cat_choice = input("      Choose category (1-6): ").strip()
            if cat_choice in category_map:
                break
            print("      Please enter a number from 1 to 6")

        category = category_map[cat_choice]

        # Define functionals by category (matching d12creation.py)
        functionals = {
            "HF": ["HF", "UHF"],
            "LDA": ["SVWN", "VBH"],
            "GGA": ["PBE", "BLYP", "PBESOL", "PWGGA", "SOGGA", "WCGGA", "B97"],
            "HYBRID": [
                "B3LYP",
                "PBE0",
                "HSE06",
                "B3PW",
                "CAM-B3LYP",
                "LC-wPBE",
                "wB97X",
            ],
            "MGGA": [
                "M06",
                "M06-2X",
                "M06-L",
                "M06-HF",
                "SCAN",
                "r2SCAN",
                "MN15",
                "MN15L",
            ],
            "3C": ["HF-3C", "PBEh-3C", "HSE-3C", "B97-3C", "PBEsol0-3C", "HSEsol-3C"],
        }

        func_list = functionals.get(category, [])

        print(f"\n      Available {category} functionals:")
        for i, func in enumerate(func_list, 1):
            # Add descriptions for common functionals
            desc = ""
            if func == "PBE":
                desc = " - popular GGA, good general purpose"
            elif func == "B3LYP":
                desc = " - popular hybrid, good for organics"
            elif func == "HSE06":
                desc = " - screened hybrid, accurate band gaps"
            elif func == "PBE0":
                desc = " - 25% HF exchange, robust"
            elif func == "M06-2X":
                desc = " - 54% HF, good for kinetics"
            elif func == "PBESOL":
                desc = " - PBE revised for solids"
            elif func == "WCGGA":
                desc = " - Wu-Cohen GGA"
            elif func == "SCAN":
                desc = " - strongly constrained meta-GGA"
            elif func == "SVWN":
                desc = " - Slater exchange + VWN5 correlation"
            print(f"        {i}: {func}{desc}")

        while True:
            try:
                func_idx = int(
                    input(f"      Choose functional (1-{len(func_list)}): ").strip()
                )
                if 1 <= func_idx <= len(func_list):
                    return func_list[func_idx - 1]
                print(f"      Please enter a number from 1 to {len(func_list)}")
            except ValueError:
                print("      Please enter a valid number")

    def _get_custom_tolerances(self) -> Dict[str, Any]:
        """Get custom tolerance settings"""
        tolerances = {}

        # TOLINTEG
        print("      TOLINTEG (Coulomb/exchange integral tolerances):")
        print("        Current/default: 7 7 7 7 14")
        print("        Tighter: 8 8 8 8 16 or 9 9 9 9 18")
        custom_tolinteg = input("      New TOLINTEG [keep current]: ").strip()
        if custom_tolinteg:
            tolerances["TOLINTEG"] = custom_tolinteg

        # TOLDEE
        print("\n      TOLDEE (SCF energy convergence):")
        print("        Current/default: 7")
        print("        Tighter: 8, 9, or 10")
        custom_toldee = input("      New TOLDEE [keep current]: ").strip()
        if custom_toldee:
            try:
                tolerances["TOLDEE"] = int(custom_toldee)
            except ValueError:
                print("      Invalid TOLDEE, keeping current")

        return tolerances

    def _create_expert_opt_template(
        self, config_path: Path, calc_type: str, step_num: int
    ):
        """Create expert configuration template for CRYSTALOptToD12.py"""
        template = {
            "calculation_type": "OPT",
            "step_number": step_num,
            "optimization_type": "FULLOPTG",
            "description": f"Expert configuration for {calc_type} step {step_num}",
            "created": datetime.now().isoformat(),
            # Base settings to inherit or modify
            "inherit_geometry": True,
            "inherit_basis_set": True,
            "inherit_method": True,
            # Customizable parameters
            "optimization_settings": {
                "TOLDEG": 1.5e-5,
                "TOLDEX": 6e-5,
                "TOLDEE": 8,
                "MAXCYCLE": 1000,
            },
            # Advanced options (to be filled interactively)
            "method_modifications": {
                "change_functional": False,
                "new_functional": "",
                "change_basis": False,
                "new_basis": "",
                "change_grid": False,
                "new_grid": "",
            },
            "scf_modifications": {
                "change_tolerances": False,
                "TOLINTEG": "",
                "TOLDEE": "",
                "change_mixing": False,
                "FMIXING": "",
            },
            # Instructions for interactive use
            "_instructions": {
                "usage": "This file will be used by CRYSTALOptToD12.py for interactive configuration",
                "modify": "Set change_* flags to true and provide new values for customization",
                "inherit": "Set inherit_* flags to false to completely override settings",
            },
        }

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save template
        with open(config_path, "w") as f:
            json.dump(template, f, indent=2)

        return template

    def _copy_required_scripts_for_expert_mode(self):
        """Copy required scripts early for expert mode configuration"""
        # With MACE setup, all scripts are already in PATH via setup_mace.py
        # No need to copy scripts to the working directory
        pass
    
    def _get_per_material_expert_d3_config(
        self, calc_type: str, step_num: int
    ) -> Dict[str, Any]:
        """Create individual expert D3 configurations for each material"""
        print(f"\n    Creating per-material {calc_type} configurations...")
        
        # Find OUT files from SP step (or previous appropriate step)
        out_files = []
        prev_step = step_num - 1
        
        # Look for OUT files in likely locations
        search_patterns = [
            f"workflow_outputs/*/step_{prev_step:03d}_*/*/*.out",
            f"workflow_outputs/*/step_*_SP/*/*.out",  # Look for SP specifically
            "*.out",
        ]
        
        for pattern in search_patterns:
            found_files = list(self.work_dir.glob(pattern))
            if found_files:
                out_files.extend(found_files)
                break
                
        if not out_files:
            print("    No OUT files found. Using single configuration mode.")
            return self._run_interactive_d3_config(calc_type, step_num)
            
        print(f"    Found {len(out_files)} materials to configure")
        
        # Create config directory
        config_dir = (
            self.work_dir / "workflow_configs" / f"expert_{calc_type.lower()}_configs"
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy required scripts
        self._copy_required_scripts_for_expert_mode()
        
        # Find CRYSTALOptToD3.py
        local_script = self.work_dir / "CRYSTALOptToD3.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d3" / "CRYSTALOptToD3.py"
            )
            
        if not script_path.exists():
            print("    Error: CRYSTALOptToD3.py not found")
            return self._get_advanced_d3_config(calc_type)
            
        # Create individual configurations for each material
        material_configs = {}
        print(f"\n    Creating per-material {calc_type} configurations...")
        print(f"    Note: You will configure each material interactively")
        
        for i, out_file in enumerate(out_files):
            material_name = self.create_material_id_from_file(out_file)
            
            print(f"\n    Material {i+1}/{len(out_files)}: {material_name}")
            print(f"    Source file: {out_file.name}")
            
            # Create config file path
            config_file = (
                config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
            )
            
            # Strip instance numbers for CRYSTALOptToD3.py compatibility
            base_calc_type = re.sub(r'\d+$', '', calc_type)
            
            # Run CRYSTALOptToD3.py interactively for this material
            cmd = [
                sys.executable,
                str(script_path),
                "--input",
                str(out_file),
                "--calc-type",
                base_calc_type,
                "--output-dir",
                str(config_dir),  # Temporary output for config generation
                "--save-config",
                "--options-file",
                str(config_file),
            ]
            
            try:
                print(f"    Launching interactive configuration for {material_name}...")
                result = subprocess.run(cmd, cwd=str(self.work_dir))
                
                if result.returncode == 0 and config_file.exists():
                    material_configs[material_name] = {
                        "config_file": str(config_file),
                        "source_out": str(out_file),
                    }
                    print(f"      ✅ {material_name}: {calc_type} configuration saved")
                else:
                    print(f"      ❌ {material_name}: Configuration failed or cancelled")
                    
            except Exception as e:
                print(f"      ❌ Error configuring {material_name}: {e}")
            
        if not material_configs:
            print("    No configurations created successfully")
            return self._get_advanced_d3_config(calc_type)
            
        return {
            "expert_mode": True,
            "per_material_configs": True,
            "config_directory": str(config_dir),
            "material_configs": material_configs,
            "step_num": step_num,
            "workflow_calc_type": calc_type,
            "d3_config_mode": "expert",
        }
    
    def _create_base_expert_d3_config(
        self, calc_type: str, template_out: Path
    ) -> Optional[Dict[str, Any]]:
        """Create base expert D3 configuration using one OUT as template"""
        # Create temporary directory
        temp_dir = self.temp_dir / f"expert_d3_config_{calc_type.lower()}_base"
        temp_dir.mkdir(exist_ok=True)
        
        # Create sample files from template OUT
        sample_out = temp_dir / "sample.out"
        sample_f9 = temp_dir / "fort.9"
        
        # Copy the actual OUT file if it exists
        try:
            import shutil
            if template_out.exists():
                shutil.copy2(template_out, sample_out)
                print(f"      Using real output file as template: {template_out.name}")
            else:
                raise FileNotFoundError("Template OUT file not found")
        except Exception as e:
            print(f"      Warning: Could not copy template OUT: {e}")
            # Create minimal dummy output file
            with open(sample_out, "w") as f:
                f.write("CRYSTAL23 OUTPUT\n")
                f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
                f.write(
                    "LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n"
                )
                f.write(
                    "A              B              C           ALPHA      BETA       GAMMA\n"
                )
                f.write(
                    "5.0000         5.0000         5.0000      90.000     90.000     90.000\n"
                )
                
        # Create dummy fort.9 file
        with open(sample_f9, "w") as f:
            f.write("DUMMY WAVEFUNCTION FILE\n")
            
        # Config file
        base_config_file = temp_dir / f"{calc_type.lower()}_base_config.json"
        
        # Find CRYSTALOptToD3.py
        local_script = self.work_dir / "CRYSTALOptToD3.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d3" / "CRYSTALOptToD3.py"
            )
            
        # Strip any instance numbers for CRYSTALOptToD3.py compatibility
        base_calc_type = re.sub(r'\d+$', '', calc_type)
        
        print(f"\n      Launching CRYSTALOptToD3.py for {calc_type} base configuration...")
        print(
            f"      Note: This creates the base configuration for all materials"
        )
        print("")
        
        # Run CRYSTALOptToD3.py interactively
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(sample_out),
            "--calc-type",
            base_calc_type,  # Pass stripped calc type (BAND2 -> BAND)
            "--output-dir",
            str(temp_dir),
            "--save-config",
            "--options-file",
            str(base_config_file),
        ]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.work_dir))
            
            if result.returncode == 0 and base_config_file.exists():
                with open(base_config_file, "r") as f:
                    return json.load(f)
                    
        except Exception as e:
            print(f"      Error running CRYSTALOptToD3.py: {e}")
            
        return None

    def _run_interactive_d3_config(
        self, calc_type: str, step_num: Optional[int] = None, real_out_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Run CRYSTALOptToD3.py interactively for expert D3 configuration"""
        # Find CRYSTALOptToD3.py
        local_script = self.work_dir / "CRYSTALOptToD3.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d3" / "CRYSTALOptToD3.py"
            )

        if not script_path.exists():
            print(f"      Error: CRYSTALOptToD3.py not found")
            return None

        # Create a temporary directory for configuration
        temp_dir = self.temp_dir / f"expert_d3_{calc_type.lower()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create sample files for CRYSTALOptToD3.py to read
        sample_out = temp_dir / "sample.out"
        sample_f9 = temp_dir / "sample.f9"

        # Try to find a real .out file
        real_out_found = False
        
        # Use provided real .out if available
        if real_out_path and real_out_path.exists():
            source_out = real_out_path
            real_out_found = True
            print(f"      Using provided output file: {source_out.name}")
        else:
            # Search for .out files
            out_search_dirs = [
                self.work_dir,
                self.work_dir.parent if self.work_dir.parent.exists() else None,
                Path.cwd(),
            ]

            for search_dir in out_search_dirs:
                if search_dir and search_dir.exists():
                    out_files = list(search_dir.glob("*.out"))
                    if out_files:
                        source_out = out_files[0]
                        print(f"      Using real output file as template: {source_out.name}")
                        real_out_found = True
                        break
                        
        if real_out_found:
            try:
                # Copy the real .out file
                with open(source_out, "r") as f:
                    out_content = f.read()
                with open(sample_out, "w") as f:
                    f.write(out_content)
            except Exception as e:
                print(f"      Warning: Could not read output file: {e}")
                real_out_found = False
        
        # Also try to find a D12 file for extracting settings
        real_d12_found = False
        real_d12_path = None
        
        # Search for .d12 files
        d12_search_dirs = [
            self.work_dir,
            self.work_dir.parent if self.work_dir.parent.exists() else None,
            Path.cwd(),
        ]
        
        for search_dir in d12_search_dirs:
            if search_dir and search_dir.exists():
                d12_files = list(search_dir.glob("*.d12"))
                if d12_files:
                    real_d12_path = d12_files[0]
                    real_d12_found = True
                    break
                
        if not real_out_found:
            # Create a more informative dummy output file by extracting settings from D12
            if DummyFileCreator and real_d12_found and real_d12_path and real_d12_path.exists():
                creator = DummyFileCreator()
                d12_settings = creator.extract_d12_settings(real_d12_path)
                creator.create_dummy_out(sample_out, d12_settings)
            elif DummyFileCreator:
                # Create minimal dummy output file
                creator = DummyFileCreator()
                creator.create_minimal_dummy_out(sample_out)
            else:
                # Fallback minimal output file
                with open(sample_out, "w") as f:
                    f.write("CRYSTAL23 OUTPUT\n")
                    f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
                    f.write(
                        "LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - PRIMITIVE CELL\n"
                    )
                    f.write(
                        "         A              B              C           ALPHA      BETA       GAMMA \n"
                    )
                    f.write(
                        "     5.00000000     5.00000000     5.00000000    90.000000  90.000000  90.000000\n"
                    )
                    f.write("\n")
                    f.write("TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL\n")

        # Create a dummy fort.9 file
        with open(sample_f9, "w") as f:
            f.write("DUMMY WAVEFUNCTION FILE\n")

        print(f"\n      Launching CRYSTALOptToD3.py for {calc_type} configuration...")
        print(
            f"      Note: This is for configuration only - actual files will be processed during execution"
        )
        print("")

        # Generate config filename with step info and timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step_{step_num}_" if step_num else ""
        config_filename = f"d3_{calc_type.lower()}_{step_str}config_{timestamp}.json"
        
        # Save in workflow_configs directory instead of temp
        config_dir = self.configs_dir / "expert_d3_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_filepath = config_dir / config_filename
        
        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(sample_out),
            "--calc-type",
            calc_type,
            "--output-dir",
            str(temp_dir),
            "--save-config",
            "--options-file",
            str(config_filepath),  # Use full path
        ]

        try:
            # Run interactively (no capture_output so user can interact)
            result = subprocess.run(cmd, cwd=str(self.work_dir))

            if result.returncode == 0:
                # Look for the saved configuration file
                # First check for the exact file we specified
                config_file = config_filepath
                
                if not config_file.exists():
                    # Fallback: CRYSTALOptToD3.py might save configs with different patterns
                    config_files = []
                    
                    # Try different naming patterns in both temp_dir and config_dir
                    patterns = [
                        f"d3_config_{calc_type}_*.json",  # With timestamp
                        f"d3_{calc_type.lower()}_config.json",  # Default name
                        f"d3_config_CHARGE+POTENTIAL_*.json",  # Special case
                        f"d3_charge+potential_config.json",  # Special case default
                        "d3_*_config.json",  # Any d3 config
                    ]
                    
                    search_dirs = [temp_dir, config_dir]
                    for search_dir in search_dirs:
                        for pattern in patterns:
                            config_files.extend(list(search_dir.glob(pattern)))
                    
                    # Remove duplicates
                    config_files = list(set(config_files))
                    
                    if config_files:
                        # Use the most recent config file
                        config_file = max(config_files, key=lambda p: p.stat().st_mtime)
                    else:
                        config_file = None
                
                if config_file and config_file.exists():
                    
                    # Load the saved configuration
                    with open(config_file, "r") as f:
                        saved_config = json.load(f)

                    # Convert to our workflow config format
                    workflow_config = {
                        "expert_mode": True,
                        "expert_config_file": str(config_file),
                        "d3_settings": saved_config,
                        "calculation_type": calc_type,
                        "source": "CRYSTALOptToD3.py",
                    }

                    print(f"      ✓ Configuration saved: {config_file.name}")
                    return workflow_config
                else:
                    print(f"      Warning: No configuration file found after running CRYSTALOptToD3.py")
                    return None

        except Exception as e:
            print(f"      Error running CRYSTALOptToD3.py: {e}")

        return None


    def _create_base_expert_config(
        self, calc_type: str, template_d12: Path
    ) -> Optional[Dict[str, Any]]:
        """Create base expert configuration using one D12 as template"""
        # Create temporary directory
        temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}_base"
        temp_dir.mkdir(exist_ok=True)

        # Create sample files from template D12
        sample_out = temp_dir / "sample.out"
        sample_d12 = temp_dir / "sample.d12"

        # Copy the actual D12 file so all settings are preserved
        try:
            import shutil
            shutil.copy2(template_d12, sample_d12)
        except Exception as e:
            print(f"      Error copying template D12: {e}")
            return None

        # Use DummyFileCreator to create proper dummy output file
        if DummyFileCreator:
            creator = DummyFileCreator()
            d12_settings = creator.extract_d12_settings(template_d12)
            creator.create_dummy_out(sample_out, d12_settings)
        else:
            # Fallback to minimal dummy file
            with open(sample_out, "w") as f:
                f.write("CRYSTAL23 OUTPUT\n")
                f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
                f.write("LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - PRIMITIVE CELL\n")
                f.write("         A              B              C           ALPHA      BETA       GAMMA \n")
                f.write("     5.00000000     5.00000000     5.00000000    90.000000  90.000000  90.000000\n")
                f.write("\n")
                f.write("TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL\n")
                f.write("KOHN-SHAM HAMILTONIAN\n")
                f.write("\n")
                f.write("(EXCHANGE)[CORRELATION] FUNCTIONAL:(BECKE 88)[LEE-YANG-PARR]\n")
                f.write("\n")
                f.write("== SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -1.0000000000000E+02 CYCLES   1\n")

        # Config file
        base_config_file = temp_dir / f"{calc_type.lower()}_base_config.json"

        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d12" / "CRYSTALOptToD12.py"
            )

        # Strip any instance numbers for CRYSTALOptToD12.py compatibility
        # Handle up to 99 instances (OPT, OPT2, OPT3, ..., OPT99)
        base_calc_type = re.sub(r'\d+$', '', calc_type)
        
        # Run CRYSTALOptToD12.py with calc-type to skip the menu
        cmd = [
            sys.executable,
            str(script_path),
            "--out-file",
            str(sample_out),
            "--d12-file",
            str(sample_d12),
            "--output-dir",
            str(temp_dir),
            "--calc-type",
            base_calc_type,  # Pass stripped calc type to skip the menu (OPT2 -> OPT)
            "--save-options",
            "--options-file",
            str(base_config_file),
        ]

        try:
            result = subprocess.run(cmd, cwd=str(self.work_dir))

            if result.returncode == 0 and base_config_file.exists():
                with open(base_config_file, "r") as f:
                    return json.load(f)

        except Exception as e:
            print(f"      Error running CRYSTALOptToD12.py: {e}")

        return None

    def _get_per_material_expert_config(
        self, calc_type: str, step_num: int
    ) -> Dict[str, Any]:
        """Create individual expert configurations for each material"""
        print(f"\n    Creating per-material {calc_type} configurations...")

        # Find D12 files from previous step
        d12_files = []
        prev_step = step_num - 1

        # Look for D12 files in likely locations
        search_patterns = [
            f"workflow_outputs/*/step_{prev_step:03d}_*/*/*.d12",
            f"workflow_inputs/step_{prev_step:03d}_*/*.d12",
            "*.d12",
        ]

        for pattern in search_patterns:
            found_files = list(self.work_dir.glob(pattern))
            if found_files:
                d12_files.extend(found_files)
                break

        if not d12_files:
            print("    No D12 files found. Using single configuration mode.")
            return self._run_interactive_crystal_opt_config(calc_type)

        print(f"    Found {len(d12_files)} materials to configure")

        # Create config directory
        config_dir = (
            self.work_dir / "workflow_configs" / f"expert_{calc_type.lower()}_configs"
        )
        config_dir.mkdir(parents=True, exist_ok=True)

        # Copy required scripts
        self._copy_required_scripts_for_expert_mode()

        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d12" / "CRYSTALOptToD12.py"
            )

        if not script_path.exists():
            print("    Error: CRYSTALOptToD12.py not found")
            return self._get_advanced_opt_config()

        # First, create base configuration using first D12 as template
        print("\n    Creating base configuration...")
        base_config = self._create_base_expert_config(calc_type, d12_files[0])

        if not base_config:
            print("    Failed to create base configuration")
            return self._get_advanced_opt_config()

        print("    ✅ Base configuration created")

        # Now create per-material configs by modifying only symmetry fields
        material_configs = {}

        print(f"\n    Creating per-material symmetry configurations...")
        for i, d12_file in enumerate(d12_files):
            material_name = self.create_material_id_from_file(d12_file)

            # Extract symmetry settings from this D12
            if DummyFileCreator:
                creator = DummyFileCreator()
                symmetry_settings = creator.extract_d12_settings(d12_file)
            else:
                # Fallback to basic settings
                symmetry_settings = {
                    "spacegroup": 1,
                    "origin_setting": "0 0 0",
                    "dimensionality": "CRYSTAL"
                }

            # Create material-specific config by updating base config
            material_config = base_config.copy()
            material_config.update(
                {
                    "spacegroup": symmetry_settings.get("spacegroup", 1),
                    "origin_setting": symmetry_settings.get("origin_setting", "0 0 0"),
                    "dimensionality": symmetry_settings.get(
                        "dimensionality", "CRYSTAL"
                    ),
                    # Set write_only_unique based on space group
                    "write_only_unique": symmetry_settings.get("spacegroup", 1) != 1,
                    # Ensure correct calculation type
                    "calculation_type": calc_type,
                }
            )
            
            # Add calculated k-points if available
            if symmetry_settings.get("k_points"):
                material_config["k_points"] = symmetry_settings["k_points"]
                
            # Store cell parameters if extracted
            if symmetry_settings.get("cell_parameters"):
                material_config["cell_parameters"] = symmetry_settings["cell_parameters"]

            # Save material-specific config
            config_file = (
                config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
            )
            with open(config_file, "w") as f:
                json.dump(material_config, f, indent=2)

            material_configs[material_name] = {
                "config_file": str(config_file),
                "source_d12": str(d12_file),
            }

            kpoints_str = material_config.get('k_points', 'not calculated')
            print(
                f"      ✅ {material_name}: spacegroup={material_config['spacegroup']}, "
                + f"origin={material_config['origin_setting']}, k-points={kpoints_str}"
            )

        if not material_configs:
            print("    No configurations created successfully")
            return self._get_advanced_opt_config()

        return {
            "expert_mode": True,
            "per_material_configs": True,
            "config_directory": str(config_dir),
            "material_configs": material_configs,
            "step_num": step_num,
            "workflow_calc_type": calc_type,
        }

    def _run_interactive_crystal_opt_config(
        self, calc_type: str, real_d12_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Run CRYSTALOptToD12.py interactively for expert configuration"""
        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = (
                Path(__file__).parent.parent.parent / "Crystal_d12" / "CRYSTALOptToD12.py"
            )

        if not script_path.exists():
            print(f"      Error: CRYSTALOptToD12.py not found")
            return None

        # Create a temporary output and D12 file for configuration
        temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}"
        temp_dir.mkdir(exist_ok=True)

        # Create minimal sample files for CRYSTALOptToD12.py to read
        sample_out = temp_dir / "sample.out"
        sample_d12 = temp_dir / "sample.d12"

        # Try to find and copy a real D12 file
        real_d12_found = False
        
        # Use provided real D12 if available
        if real_d12_path and real_d12_path.exists():
            source_d12 = real_d12_path
            real_d12_found = True
            print(f"      Using provided D12 as template: {source_d12.name}")
        else:
            # Search for D12 files
            d12_search_dirs = [
                self.work_dir,
                self.work_dir.parent if self.work_dir.parent.exists() else None,
                Path.cwd(),
            ]

            for search_dir in d12_search_dirs:
                if search_dir and search_dir.exists():
                    d12_files = list(search_dir.glob("*.d12"))
                    if d12_files:
                        # Copy the first D12 file found
                        source_d12 = d12_files[0]
                        print(f"      Using real D12 as template: {source_d12.name}")
                        real_d12_found = True
                        break
                        
        if real_d12_found:
            try:
                with open(source_d12, "r") as f:
                    lines = f.readlines()

                # Copy the actual D12 file so all settings are preserved
                import shutil
                shutil.copy2(source_d12, sample_d12)

            except Exception as e:
                print(f"      Warning: Could not process D12 file: {e}")
                real_d12_found = False

        if not real_d12_found:
            print("      No real D12 found, using minimal template")
            # Fall back to minimal template
            with open(sample_d12, "w") as f:
                f.write("Sample D12 for configuration\n")
                f.write("CRYSTAL\n")
                f.write("0 0 0\n")
                f.write("1\n")
                f.write("5.0 5.0 5.0 90.0 90.0 90.0\n")
                f.write("1\n")
                f.write("6 0.0 0.0 0.0\n")
                f.write("OPTGEOM\n")
                f.write("FULLOPTG\n")
                f.write("ENDOPT\n")
                f.write("END\n")

        # Try to find corresponding .out file if we have a real D12
        real_out_found = False
        if real_d12_path and real_d12_path.exists():
            # Look for corresponding .out file
            out_path = real_d12_path.with_suffix('.out')
            if out_path.exists():
                try:
                    # Copy the real .out file
                    with open(out_path, "r") as f:
                        out_content = f.read()
                    with open(sample_out, "w") as f:
                        f.write(out_content)
                    real_out_found = True
                    print(f"      Using real output file: {out_path.name}")
                except Exception as e:
                    print(f"      Warning: Could not read output file: {e}")
                    
        if not real_out_found:
            # Write minimal output file using DummyFileCreator
            if DummyFileCreator:
                creator = DummyFileCreator()
                creator.create_minimal_dummy_out(sample_out)
            else:
                # Fallback
                with open(sample_out, "w") as f:
                    f.write("CRYSTAL23 OUTPUT\n")
                    f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
                    f.write(
                        "LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n"
                    )
                    f.write(
                        "A              B              C           ALPHA      BETA       GAMMA\n"
                    )
                    f.write(
                        "5.0000         5.0000         5.0000      90.000     90.000     90.000\n"
                    )
                    # Add Method line to avoid AttributeError
                    f.write("\nMETHOD SECTION\n")
                    f.write("Method: DFT (B3LYP)\n")
                    f.write("END OF METHOD SECTION\n")

        # Create a JSON config file to save the results
        config_file = temp_dir / f"{calc_type.lower()}_expert_config.json"

        # Strip any instance numbers for CRYSTALOptToD12.py compatibility
        # Handle up to 99 instances (OPT, OPT2, OPT3, ..., OPT99)
        base_calc_type = re.sub(r'\d+$', '', calc_type)
        
        # Display appropriate calc type in messages
        display_calc_type = calc_type
        if calc_type.startswith('OPT') and calc_type != 'OPT':
            display_calc_type = f"{calc_type} (Optimization)"
        
        print(f"\n      Launching CRYSTALOptToD12.py for Expert {display_calc_type} configuration...")

        # Build command with calc-type to pre-select calculation type
        # But still run interactively like the D3 scripts do
        cmd = [
            sys.executable,
            str(script_path),
            "--out-file",
            str(sample_out),
            "--d12-file",
            str(sample_d12),
            "--output-dir",
            str(temp_dir),
            "--calc-type",
            base_calc_type,  # Pre-select calc type but stay interactive (OPT2 -> OPT)
            "--save-options",
            "--options-file",
            str(config_file),
        ]

        try:
            # Run interactively (no capture_output so user can interact)
            result = subprocess.run(cmd, cwd=str(self.work_dir))

            if result.returncode == 0 and config_file.exists():
                # Load the saved configuration
                with open(config_file, "r") as f:
                    saved_config = json.load(f)

                # Convert to our workflow config format
                # Ensure the saved config has the correct calculation type
                # For FREQ calculations, always use "FREQ" not FREQ2, FREQ3, etc.
                if calc_type.startswith("FREQ"):
                    saved_config["calculation_type"] = "FREQ"
                else:
                    saved_config["calculation_type"] = calc_type

                # Save the corrected config back to file
                with open(config_file, "w") as f:
                    json.dump(saved_config, f, indent=2)

                expert_config = {
                    "expert_mode": True,
                    "interactive_setup": False,  # Already done
                    "crystal_opt_config": saved_config,
                    "source": "CRYSTALOptToD12.py",
                    "calculation_type": calc_type,
                    "inherit_geometry": True,
                    "config_file": str(config_file),
                    "options_file": str(config_file),  # Store path to saved options
                }

                return expert_config
            else:
                print(f"      CRYSTALOptToD12.py configuration failed or was cancelled")
                return None

        except Exception as e:
            print(f"      Error running CRYSTALOptToD12.py: {e}")
            return None

    def get_custom_sp_settings(self) -> Dict[str, Any]:
        """Get custom single point settings"""
        return {"modify_functional": False, "modify_basis": False, "modify_grid": False}

    def configure_slurm_scripts(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Configure and copy SLURM submission scripts for this calculation step"""
        print(f"    Configuring SLURM scripts for {calc_type} step {step_num}")

        # Determine which scripts are needed
        scripts_needed = self.get_required_scripts(calc_type)

        slurm_config = {"scripts": {}, "resources": {}, "modules": {}}

        for script_name in scripts_needed:
            print(f"      Setting up {script_name}...")
            script_config = self.setup_slurm_script(script_name, calc_type, step_num)
            slurm_config["scripts"][script_name] = script_config

        return slurm_config

    def get_required_scripts(self, calc_type: str) -> List[str]:
        """Determine which SLURM scripts are needed for a calculation type"""
        # Extract base calculation type (remove numbers)
        base_type = calc_type.rstrip("0123456789")

        if base_type in ["OPT", "SP", "FREQ"]:
            return ["submitcrystal23.sh"]
        elif base_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]:
            return ["submit_prop.sh"]
        else:
            return ["submitcrystal23.sh"]  # Default

    def setup_slurm_script(
        self, script_name: str, calc_type: str, step_num: int
    ) -> Dict[str, Any]:
        """Setup and customize a SLURM script for specific calculation"""

        # Get default resource settings based on script and calc type
        default_resources = self.get_default_resources(script_name, calc_type)

        print(f"        Default resources for {calc_type}:")
        print(f"          Cores: {default_resources['ntasks']}")

        # Add calculation-specific resource explanations
        if calc_type.startswith("FREQ"):
            print(f"          📊 FREQ resource notes:")
            print(
                f"            - Time scales with number of atoms (N_atoms × 2 displacements)"
            )
            print(f"            - Memory increases for CPHF/Raman calculations (~2-3x)")
            print(
                f"            - Phonon dispersion needs more time (supercell calculations)"
            )
        elif calc_type.startswith("SP"):
            print(f"          📊 SP resource notes:")
            print(f"            - Single SCF calculation, typically faster than OPT")
            print(
                f"            - Tight convergence may require +20-50% more iterations"
            )

        # Display memory with clear indication of per-cpu vs total
        # This addresses user confusion about memory specifications in SLURM
        # submitcrystal23.sh uses --mem-per-cpu while submit_prop.sh uses --mem (total)
        if "memory_per_cpu" in default_resources:
            print(f"          Memory per CPU: {default_resources['memory_per_cpu']}")
            # Try to calculate total memory
            mem_str = default_resources["memory_per_cpu"].upper()
            if mem_str.endswith("G") or mem_str.endswith("GB"):
                mem_val = int(mem_str.rstrip("GB"))
                total_mem_gb = mem_val * default_resources["ntasks"]
                print(
                    f"          Total Memory: {total_mem_gb}G ({default_resources['ntasks']} cores × {default_resources['memory_per_cpu']})"
                )
            elif mem_str.endswith("M") or mem_str.endswith("MB"):
                mem_val = int(mem_str.rstrip("MB"))
                total_mem_mb = mem_val * default_resources["ntasks"]
                total_mem_gb = total_mem_mb // 1000
                print(
                    f"          Total Memory: ~{total_mem_gb}G ({default_resources['ntasks']} cores × {default_resources['memory_per_cpu']})"
                )
        elif "memory" in default_resources:
            print(f"          Total Memory: {default_resources['memory']}")
            # Try to calculate per-cpu memory
            mem_str = default_resources["memory"].upper()
            if mem_str.endswith("G") or mem_str.endswith("GB"):
                mem_val = int(mem_str.rstrip("GB"))
                per_cpu_gb = mem_val // default_resources["ntasks"]
                print(
                    f"          Memory per CPU: ~{per_cpu_gb}G ({default_resources['memory']} ÷ {default_resources['ntasks']} cores)"
                )
            elif mem_str.endswith("M") or mem_str.endswith("MB"):
                mem_val = int(mem_str.rstrip("MB"))
                per_cpu_mb = mem_val // default_resources["ntasks"]
                per_cpu_gb = per_cpu_mb // 1000
                if per_cpu_gb > 0:
                    print(
                        f"          Memory per CPU: ~{per_cpu_gb}G ({default_resources['memory']} ÷ {default_resources['ntasks']} cores)"
                    )
                else:
                    print(
                        f"          Memory per CPU: ~{per_cpu_mb}M ({default_resources['memory']} ÷ {default_resources['ntasks']} cores)"
                    )
        else:
            print(f"          Memory: N/A")

        print(f"          Walltime: {default_resources['walltime']}")
        print(f"          Account: {default_resources.get('account', 'mendoza_q')}")

        # Ask user if they want to customize
        customize = yes_no_prompt(
            f"        Customize resources for {calc_type} step {step_num}?", "no"
        )

        if customize:
            resources = self.get_custom_resources(default_resources, calc_type)
        else:
            resources = default_resources

        # Create script configuration
        script_config = {
            "source_script": script_name,
            "step_specific_name": f"{script_name.replace('.sh', '')}_{calc_type.lower()}_{step_num}.sh",
            "resources": resources,
            "customizations": [],
            "copy_to_workdir": True,
        }

        # Ask about additional customizations
        additional_custom = yes_no_prompt(
            f"        Add custom SLURM directives for {calc_type}?", "no"
        )
        if additional_custom:
            customizations = self.get_additional_customizations(calc_type)
            script_config["customizations"] = customizations

        return script_config

    def get_default_resources(self, script_name: str, calc_type: str) -> Dict[str, Any]:
        """Get default resource settings for script and calculation type"""

        if script_name == "submitcrystal23.sh":
            # From the script analysis: 32 cores, 7 hours, 5G per core
            base_resources = {
                "ntasks": 32,
                "nodes": 1,
                "walltime": "7-00:00:00",
                "memory_per_cpu": "5G",
                "account": "mendoza_q",
                "module": "CRYSTAL/23-intel-2023a",
                "scratch_dir": "$SCRATCH/crys23",
            }
        elif script_name == "submit_prop.sh":
            # From the script analysis: 28 cores, 2 hours, 80G total
            base_resources = {
                "ntasks": 28,
                "nodes": 1,
                "walltime": "2:00:00",
                "memory": "80G",
                "account": "mendoza_q",
                "module": "CRYSTAL/23-intel-2023a",
                "scratch_dir": "$SCRATCH/crys23/prop",
            }
        else:
            # Default fallback
            base_resources = {
                "ntasks": 16,
                "nodes": 1,
                "walltime": "4:00:00",
                "memory_per_cpu": "4G",
                "account": "general",
            }

        # Apply calculation-specific scaling from workflows.yaml
        scaled_resources = self.apply_calc_type_scaling(base_resources, calc_type)

        return scaled_resources

    def apply_calc_type_scaling(
        self, resources: Dict[str, Any], calc_type: str
    ) -> Dict[str, Any]:
        """Apply calculation-type specific resource scaling"""

        # Extract base calculation type for scaling (handle numbered variants)
        base_type = calc_type.rstrip("0123456789")

        # Resource scaling based on workflows.yaml analysis
        scaling_rules = {
            "OPT": {"walltime_factor": 1.0, "memory_factor": 1.0},  # Standard - 7 days
            "SP": {
                "walltime_factor": 0.43,
                "memory_factor": 0.8,
            },  # 3 days (3/7 ≈ 0.43)
            "FREQ": {
                "walltime_factor": 1.0,
                "memory_factor": 1.5,
            },  # 7 days (N_atoms × 2 × OPT time)
            "BAND": {
                "walltime_factor": 1.0,
                "memory_factor": 0.6,
            },  # 1 day (base is already 1 day)
            "DOSS": {
                "walltime_factor": 1.0,
                "memory_factor": 0.6,
            },  # 1 day (base is already 1 day)
        }

        # Use base type for lookup (so BAND2, BAND3 etc use BAND scaling)
        if base_type in scaling_rules:
            scaling = scaling_rules[base_type]
        elif (
            calc_type in scaling_rules
        ):  # Fallback for exact match (like OPT2 if explicitly defined)
            scaling = scaling_rules[calc_type]
        else:
            # No scaling if type not found
            return resources

        # Apply walltime scaling
        if "walltime" in resources:
            current_walltime = resources["walltime"]
            if "-" in current_walltime:
                # Format: D-HH:MM:SS
                days, time_part = current_walltime.split("-")
                hours = int(time_part.split(":")[0])
                total_hours = int(days) * 24 + hours
                new_total_hours = max(1, int(total_hours * scaling["walltime_factor"]))
                new_days = new_total_hours // 24
                new_hours = new_total_hours % 24
                resources["walltime"] = f"{new_days}-{new_hours:02d}:00:00"
            else:
                # Format: H:MM:SS (legacy)
                hours = int(current_walltime.split(":")[0])
                new_hours = max(1, int(hours * scaling["walltime_factor"]))
                resources["walltime"] = f"{new_hours}:00:00"

        # Apply memory scaling
        memory_factor = scaling["memory_factor"]
        if "memory_per_cpu" in resources:
            current_mem = resources["memory_per_cpu"]
            if current_mem.endswith("G"):
                mem_val = int(current_mem[:-1])
                new_mem = max(1, int(mem_val * memory_factor))
                resources["memory_per_cpu"] = f"{new_mem}G"
        elif "memory" in resources:
            current_mem = resources["memory"]
            if current_mem.endswith("G"):
                mem_val = int(current_mem[:-1])
                new_mem = max(1, int(mem_val * memory_factor))
                resources["memory"] = f"{new_mem}G"

        return resources

    def get_safe_choice_input(
        self, prompt: str, valid_choices: list, default: str = None
    ) -> str:
        """Get user choice with validation against a list of valid options"""
        while True:
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip() or default
            else:
                user_input = input(f"{prompt}: ").strip()

            if user_input in valid_choices:
                return user_input
            else:
                print(
                    f"⚠️  Invalid choice '{user_input}'. Valid options: {', '.join(valid_choices)}"
                )
                continue

    def get_safe_integer_input(
        self, prompt: str, default: int, min_val: int = 1, max_val: int = None
    ) -> int:
        """Get integer input with validation and re-prompting on error"""
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                return default

            try:
                value = int(user_input)
                if value < min_val:
                    print(
                        f"          ⚠️  Value must be at least {min_val}. Please try again."
                    )
                    continue
                if max_val and value > max_val:
                    print(
                        f"          ⚠️  Value must be at most {max_val}. Please try again."
                    )
                    continue
                return value
            except ValueError:
                print(
                    f"          ⚠️  Invalid input '{user_input}'. Please enter a number."
                )
                continue

    def get_safe_memory_input(self, prompt: str, default: str) -> str:
        """Get memory input with validation"""
        import re

        while True:
            user_input = input(prompt).strip()
            if not user_input:
                return default

            # Valid memory formats: number + optional unit (G, GB, M, MB)
            pattern = r"^\d+([GMK]B?)?$"

            if re.match(pattern, user_input, re.IGNORECASE):
                # Normalize the format
                if user_input.isdigit():
                    return f"{user_input}G"  # Default to GB if no unit
                return user_input.upper()
            else:
                print(f"          ⚠️  Invalid memory format '{user_input}'.")
                print(f"          Valid formats: 4G, 4GB, 4000M, 4000MB")
                print(f"          Examples: 5G, 48G, 4000M")
                continue

    def get_safe_walltime_input(self, prompt: str, default: str) -> str:
        """Get walltime input with validation"""
        import re

        while True:
            user_input = input(prompt).strip()
            if not user_input:
                return default

            # Valid walltime formats: HH:MM:SS, D-HH:MM:SS, DD-HH:MM:SS
            patterns = [
                r"^\d{1,2}:\d{2}:\d{2}$",  # HH:MM:SS
                r"^\d-\d{1,2}:\d{2}:\d{2}$",  # D-HH:MM:SS
                r"^\d{1,2}-\d{1,2}:\d{2}:\d{2}$",  # DD-HH:MM:SS
            ]

            if any(re.match(pattern, user_input) for pattern in patterns):
                return user_input
            else:
                print(f"          ⚠️  Invalid walltime format '{user_input}'.")
                print(f"          Valid formats: HH:MM:SS, D-HH:MM:SS, or DD-HH:MM:SS")
                print(f"          Examples: 24:00:00, 3-00:00:00, 7-00:00:00")
                continue

    def get_custom_resources(
        self, default_resources: Dict[str, Any], calc_type: str
    ) -> Dict[str, Any]:
        """Get custom resource settings from user"""
        resources = default_resources.copy()

        print(f"        Customize resources for {calc_type}:")

        # Cores
        resources["ntasks"] = self.get_safe_integer_input(
            f"          Cores [{resources['ntasks']}]: ",
            default=resources["ntasks"],
            min_val=1,
            max_val=128,  # Adjust based on your cluster
        )

        # Memory
        if "memory_per_cpu" in resources:
            resources["memory_per_cpu"] = self.get_safe_memory_input(
                f"          Memory per CPU [{resources['memory_per_cpu']}]: ",
                default=resources["memory_per_cpu"],
            )
        elif "memory" in resources:
            resources["memory"] = self.get_safe_memory_input(
                f"          Total memory [{resources['memory']}]: ",
                default=resources["memory"],
            )

        # Walltime
        resources["walltime"] = self.get_safe_walltime_input(
            f"          Walltime [{resources['walltime']}]: ",
            default=resources["walltime"],
        )

        # Account
        new_account = input(
            f"          Account [{resources.get('account', 'mendoza_q')}]: "
        ).strip()
        if new_account:
            resources["account"] = new_account

        return resources

    def get_additional_customizations(self, calc_type: str) -> List[Dict[str, str]]:
        """Get additional SLURM customizations from user"""
        customizations = []

        print(f"        Additional SLURM directives for {calc_type}:")
        print(
            "        Enter SLURM directives (without #SBATCH). Press Enter when done."
        )

        common_options = {
            "1": "--constraint=intel18",
            "2": "--exclude=node1,node2",
            "3": "--partition=gpu",
            "4": "--gres=gpu:1",
            "5": "--array=1-10",
            "6": "Custom directive",
        }

        print("        Common options:")
        for key, option in common_options.items():
            print(f"          {key}: {option}")
        print("          Enter: Finish")

        while True:
            choice = input("        Select option or enter custom directive: ").strip()

            if not choice:
                break
            elif choice in common_options:
                if choice == "6":
                    custom = input("        Enter custom directive: ").strip()
                    if custom:
                        customizations.append(
                            {"directive": custom, "description": "Custom"}
                        )
                else:
                    directive = common_options[choice]
                    customizations.append(
                        {"directive": directive, "description": f"Common: {directive}"}
                    )
            else:
                # Treat as custom directive
                customizations.append({"directive": choice, "description": "Custom"})

        return customizations

    def copy_and_customize_scripts(
        self, step_configs: Dict[str, Dict[str, Any]], workflow_id: str, queue_config: Dict[str, Any]
    ):
        """Copy and customize all SLURM scripts for the workflow"""
        print("\nStep 4.5: Setting up SLURM scripts and configuration files")
        print("-" * 40)

        scripts_dir = self.work_dir / "workflow_scripts"
        scripts_dir.mkdir(exist_ok=True)

        bin_scripts_dir = Path(__file__).parent

        # Copy SLURM scripts
        for step_key, config in step_configs.items():
            if "slurm_config" in config:
                slurm_config = config["slurm_config"]

                for script_name, script_config in slurm_config["scripts"].items():
                    # Add workflow_id to script_config for context environment variables
                    script_config['workflow_id'] = workflow_id
                    self.create_customized_script(
                        bin_scripts_dir, scripts_dir, script_config, step_key, queue_config
                    )

        # Copy additional required files
        self.copy_additional_files(bin_scripts_dir, workflow_id)

    def create_customized_script(
        self,
        bin_dir: Path,
        scripts_dir: Path,
        script_config: Dict[str, Any],
        step_key: str,
        queue_config: Dict[str, Any],
    ):
        """Create a customized version of a SLURM script"""

        source_script = bin_dir / script_config["source_script"]
        target_script = scripts_dir / script_config["step_specific_name"]

        print(f"    Creating {target_script.name} for {step_key}")

        if not source_script.exists():
            print(f"      Warning: Source script {source_script} not found")
            return

        # Read source script
        with open(source_script, "r") as f:
            content = f.read()

        # Apply customizations
        modified_content = self.apply_script_customizations(content, script_config, queue_config)

        # Write customized script
        with open(target_script, "w") as f:
            f.write(modified_content)

        # Make executable
        target_script.chmod(0o755)

        print(f"      Created: {target_script}")
        print(
            f"      Resources: {script_config['resources']['ntasks']} cores, "
            f"{script_config['resources']['walltime']} walltime"
        )

    def apply_script_customizations(
        self, content: str, script_config: Dict[str, Any], queue_config: Dict[str, Any]
    ) -> str:
        """Apply resource and directive customizations to script content"""

        resources = script_config["resources"]
        customizations = script_config.get("customizations", [])

        lines = content.split("\n")
        modified_lines = []

        for line in lines:
            # Modify resource directives
            if line.startswith("echo '#SBATCH --ntasks="):
                modified_lines.append(
                    f"echo '#SBATCH --ntasks={resources['ntasks']}' >> $1.sh"
                )
            elif line.startswith("echo '#SBATCH -t "):
                modified_lines.append(
                    f"echo '#SBATCH -t {resources['walltime']}' >> $1.sh"
                )
            elif (
                line.startswith("echo '#SBATCH --mem-per-cpu=")
                and "memory_per_cpu" in resources
            ):
                modified_lines.append(
                    f"echo '#SBATCH --mem-per-cpu={resources['memory_per_cpu']}' >> $1.sh"
                )
            elif line.startswith("echo '#SBATCH --mem=") and "memory" in resources:
                modified_lines.append(
                    f"echo '#SBATCH --mem={resources['memory']}' >> $1.sh"
                )
            elif line.startswith("echo '#SBATCH -A ") and "account" in resources:
                modified_lines.append(
                    f"echo '#SBATCH -A {resources['account']}' >> $1.sh"
                )
            elif (
                line.startswith("echo '#SBATCH --constraint=")
                and "constraint" in resources
            ):
                modified_lines.append(
                    f"echo '#SBATCH --constraint={resources['constraint']}' >> $1.sh"
                )
            # Add workflow context environment variables after export JOB line
            elif line.startswith("echo 'export JOB="):
                modified_lines.append(line)
                # Add workflow context environment variables
                workflow_id = script_config.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                # Use absolute path to workflow context directory
                context_dir = str(Path(self.work_dir).resolve() / f".mace_context_{workflow_id}")
                modified_lines.append(f"echo '# Workflow context for queue manager' >> $1.sh")
                modified_lines.append(f"echo 'export MACE_WORKFLOW_ID=\"{workflow_id}\"' >> $1.sh")
                modified_lines.append(f"echo 'export MACE_CONTEXT_DIR=\"{context_dir}\"' >> $1.sh")
                modified_lines.append(f"echo 'export MACE_ISOLATION_MODE=\"isolated\"' >> $1.sh")
                continue  # Skip the normal append since we already added the line
            # CRYSTAL module loading (Python module already in base template)
            elif "module load CRYSTAL" in line and ">> $1.sh" in line:
                modified_lines.append(line)
                # Note: Python module loading is already in the base template
            else:
                modified_lines.append(line)

        # Add custom directives after the main SBATCH directives
        if customizations:
            # Find where to insert (after last #SBATCH directive)
            insert_pos = -1
            for i, line in enumerate(modified_lines):
                if line.startswith("echo '#SBATCH"):
                    insert_pos = i

            if insert_pos >= 0:
                for custom in customizations:
                    directive = custom["directive"]
                    custom_line = f"echo '#SBATCH {directive}' >> $1.sh"
                    modified_lines.insert(insert_pos + 1, custom_line)
                    insert_pos += 1

        # Apply queue management configuration to callback parameters
        content_str = "\n".join(modified_lines)
        
        # Replace hardcoded callback parameters with configured values
        # Pattern: --max-jobs 250 --reserve 30 --max-submit 5
        import re
        callback_pattern = r'--max-jobs\s+\d+\s+--reserve\s+\d+\s+--max-submit\s+\d+'
        replacement = f"--max-jobs {queue_config['max_jobs']} --reserve {queue_config['reserve_slots']} --max-submit {queue_config['max_submit_batch']}"
        
        content_str = re.sub(callback_pattern, replacement, content_str)
        
        # Also handle --max-recovery-attempts if present  
        recovery_pattern = r'--max-recovery-attempts\s+\d+'
        recovery_replacement = f"--max-recovery-attempts {queue_config['max_recovery_attempts']}"
        content_str = re.sub(recovery_pattern, recovery_replacement, content_str)

        return content_str

    def copy_additional_files(self, bin_dir: Path, workflow_id: str):
        """Copy additional required files (minimal with centralized installation)"""
        print("    Setting up workflow directory...")

        # With centralized MACE installation, we only need to ensure directories exist
        # All scripts are accessible via PATH or $MACE_HOME
        
        # Create essential directories
        essential_dirs = [
            self.work_dir / "workflow_configs",
            self.work_dir / "workflow_scripts",  # For generated SLURM scripts
            self.work_dir / "workflow_inputs",   # For input files
            self.work_dir / "workflow_outputs"   # For output files
        ]
        
        for dir_path in essential_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"    ✓ Created essential workflow directories")
        print(f"    Note: Using centralized MACE installation from $MACE_HOME")
        
        # With centralized installation, scripts are accessed via:
        # - Direct execution: enhanced_queue_manager.py (via PATH)
        # - Module imports: from mace.database.materials import MaterialDatabase
        # - Environment paths: $MACE_HOME/mace/config/recovery_config.yaml
        
        return 0, 0  # No files copied, no files missing

    def should_update_file(self, source_file: Path, dest_file: Path) -> bool:
        """Check if we should update the destination file"""
        if not dest_file.exists():
            return True

        # Compare modification times
        source_mtime = source_file.stat().st_mtime
        dest_mtime = dest_file.stat().st_mtime

        return source_mtime > dest_mtime

    def save_workflow_plan(self, workflow_plan: Dict[str, Any]) -> Path:
        """Save complete workflow plan to JSON"""
        # Use the workflow_id timestamp to ensure the plan file matches
        workflow_id = workflow_plan.get(
            "workflow_id", f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        timestamp = workflow_id.replace("workflow_", "")
        plan_file = self.configs_dir / f"workflow_plan_{timestamp}.json"

        with open(plan_file, "w") as f:
            json.dump(workflow_plan, f, indent=2, default=str)

        print(f"\n✅ Workflow plan saved to: {plan_file}")
        print(f"\n📁 Configuration locations:")
        print(f"   Main plan: {plan_file}")
        print(f"   CIF config: {self.configs_dir}/cif_conversion_config.json")
        print(f"   Step configs: {self.configs_dir}/step_configs/workflow_*_step_*.json")
        print(f"   All configs: {self.configs_dir}/")
        return plan_file

    def execute_workflow_plan(self, plan_file: Path):
        """Execute the planned workflow using WorkflowExecutor for proper isolation support"""
        print(f"\nStep 5: Execute Workflow")
        print("-" * 40)

        with open(plan_file, "r") as f:
            plan = json.load(f)

        print("Workflow execution summary:")
        print(
            f"  Input files: {len(plan['input_files']['cif'])} CIFs, {len(plan['input_files']['d12'])} D12s"
        )
        print(f"  Workflow sequence: {' → '.join(plan['workflow_sequence'])}")
        print(
            f"  Total materials: {len(plan['input_files']['cif']) + len(plan['input_files']['d12'])}"
        )
        
        # Show isolation mode if set
        isolation_mode = plan.get('isolation_mode', 'shared')
        if isolation_mode != 'shared':
            print(f"  Isolation mode: {isolation_mode}")
            post_action = plan.get('post_completion_action', 'keep')
            print(f"  Post-completion: {post_action}")

        proceed = yes_no_prompt("Proceed with workflow execution?", "yes")
        if not proceed:
            print("Workflow execution cancelled.")
            return

        # Use WorkflowExecutor for proper isolation support
        try:
            from mace.workflow.executor import WorkflowExecutor
            
            print("\nStarting workflow execution...")
            executor = WorkflowExecutor(str(self.work_dir), self.db_path)
            
            # Execute through the proper workflow executor which handles contexts
            executor.execute_workflow_plan(plan_file)
            
        except Exception as e:
            print(f"\nError during workflow execution: {e}")
            import traceback
            traceback.print_exc()
            
            # Ask if user wants to try fallback execution
            use_fallback = yes_no_prompt("\nTry fallback execution without isolation?", "no")
            if use_fallback:
                print("\nUsing fallback execution (no isolation support)...")
                # Initialize queue manager for fallback
                queue_manager = EnhancedCrystalQueueManager(
                    d12_dir=str(Path.cwd()),
                    max_jobs=200,
                    enable_tracking=True,
                    db_path=self.db_path,
                    enable_error_recovery=True,
                )
                self.run_workflow_execution(plan, queue_manager)

    def run_workflow_execution(self, plan: Dict[str, Any], queue_manager):
        """Run the actual workflow execution"""
        print("\nStarting workflow execution...")

        # Phase 1: Convert CIFs to D12s if needed
        if plan["input_files"]["cif"]:
            print("Phase 1: Converting CIF files to D12 format...")
            self.convert_cifs_to_d12s(plan)

        # Phase 2: Execute planned calculation sequence using WorkflowExecutor
        print("Phase 2: Executing calculation sequence...")
        try:
            from mace.workflow.executor import WorkflowExecutor

            executor = WorkflowExecutor(str(self.work_dir), self.db_path)

            # Use the workflow ID from the plan - CRITICAL for workflow progression!
            workflow_id = plan.get("workflow_id")
            if not workflow_id:
                # Only generate a new one if missing (shouldn't happen)
                workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(
                    f"WARNING: No workflow_id in plan, generated new one: {workflow_id}"
                )

            # Prepare the workflow directory structure for the executor
            workflow_dir = executor.outputs_dir / workflow_id
            workflow_dir.mkdir(exist_ok=True)

            # Set up step_001_OPT directory with D12 files
            step_001_dir = workflow_dir / "step_001_OPT"
            step_001_dir.mkdir(exist_ok=True)

            # Copy D12 files from input directory to step directory with clean naming
            input_dir = Path(plan["input_directory"])
            d12_files = list(input_dir.glob("*.d12"))

            print(
                f"  Found {len(d12_files)} D12 files for workflow execution"
            )
            # Don't copy files here - let the executor handle it to avoid duplication
            # The executor will create individual material folders and copy files there

            # Execute the workflow using the proper executor
            executor.execute_workflow_steps(plan, workflow_id)

        except Exception as e:
            print(f"Error executing workflow: {e}")
            print("Falling back to basic execution...")
            self.execute_calculation_sequence(plan, queue_manager)

    def convert_cifs_to_d12s(self, plan: Dict[str, Any]):
        """Convert CIF files to D12 format using saved configuration"""
        # Check if D12 files already exist (skip if already done)
        input_dir = Path(plan["input_directory"])
        existing_d12s = list(input_dir.glob("*.d12"))
        cif_files = list(input_dir.glob("*.cif"))

        if len(existing_d12s) >= len(cif_files):
            print(
                f"Found {len(existing_d12s)} D12 files already exist for {len(cif_files)} CIF files."
            )
            print("Skipping CIF conversion - using existing D12 files.")

            # D12 files already exist in input directory, no need to copy to separate input directory
            # They will be used directly by the workflow executor
            print(f"  Using existing D12 files from: {input_dir}")

            return

        # If we need to convert, use the WorkflowExecutor
        print("Converting CIF files to D12 format...")
        try:
            from mace.workflow.executor import WorkflowExecutor

            executor = WorkflowExecutor(str(self.work_dir), self.db_path)

            # Create a temporary workflow ID for this conversion
            workflow_id = f"temp_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Run the conversion using the proper executor
            executor.convert_cifs_with_config(plan, workflow_id)
            print("CIF conversion completed successfully!")

        except Exception as e:
            print(f"CIF conversion failed: {e}")
            print("Please check the configuration and try again.")

    def execute_calculation_sequence(self, plan: Dict[str, Any], queue_manager):
        """Execute the planned calculation sequence"""
        print("Starting calculation sequence execution...")

        # Get the input files for the first step from the input directory
        input_dir = Path(plan["input_directory"])
        d12_files = list(input_dir.glob("*.d12"))

        if not d12_files:
            print("Error: No D12 files found for job submission!")
            return

        print(f"Found {len(d12_files)} D12 files to submit")

        # Submit the initial OPT calculations using the enhanced queue manager
        try:
            # Copy D12 files to working directory for submission with clean names
            for d12_file in d12_files:
                clean_material_id = self.create_clean_material_id(d12_file)
                clean_filename = f"{clean_material_id}_opt.d12"
                dest_file = Path.cwd() / clean_filename
                if not dest_file.exists():
                    shutil.copy2(d12_file, dest_file)
                    print(f"  Prepared: {d12_file.name} → {clean_filename}")

            # Submit jobs using the queue manager
            print(f"\nSubmitting {len(d12_files)} OPT jobs...")
            queue_manager.process_new_d12_files()

            print("Job submission completed!")
            print("Monitor progress with:")
            print(f"  mace monitor --status")
            print(f"  Database: {self.db_path}")

        except Exception as e:
            print(f"Error during job submission: {e}")
            print("You can manually submit jobs using:")
            print(f"  cd {input_dir}")
            print("  mace submit --max-jobs 200")

    def configure_queue_management(self) -> Dict[str, Any]:
        """Configure SLURM queue management settings with streamlined interface"""
        print("\nStep 4.8: Queue Management Configuration")
        print("-" * 40)
        
        # Default settings optimized for 1000-job SLURM limit
        default_config = {
            "max_jobs": 950,           # Buffer below 1000 limit
            "reserve_slots": 50,       # Safety buffer
            "max_submit_batch": 10,    # Jobs to submit per callback
            "max_recovery_attempts": 3 # Error recovery attempts
        }
        
        print("Default queue management settings:")
        print(f"  • Maximum total jobs: {default_config['max_jobs']}")
        print(f"  • Reserve slots: {default_config['reserve_slots']}")
        print(f"  • Max submit per callback: {default_config['max_submit_batch']}")
        print(f"  • Recovery attempts: {default_config['max_recovery_attempts']}")
        print("\nThese settings prevent hitting SLURM job limits and are optimized")
        print("for most HPC clusters with 1000-job limits.")
        
        modify = yes_no_prompt("\nModify queue management settings?", "no")
        
        if not modify:
            return default_config
            
        print("\nCustomizing queue management settings:")
        config = {}
        
        # Max total jobs
        print(f"\n1. Maximum total SLURM jobs allowed:")
        print(f"   Current default: {default_config['max_jobs']}")
        print(f"   This prevents exceeding your cluster's job limit")
        print(f"   Recommended: 950 for clusters with 1000-job limits")
        max_jobs_input = input(f"   Enter value [{default_config['max_jobs']}]: ").strip()
        config['max_jobs'] = int(max_jobs_input) if max_jobs_input else default_config['max_jobs']
        
        # Reserve slots
        print(f"\n2. SLURM slots to keep in reserve:")
        print(f"   Current default: {default_config['reserve_slots']}")
        print(f"   Safety buffer to prevent submitting too close to limit")
        reserve_input = input(f"   Enter value [{default_config['reserve_slots']}]: ").strip()
        config['reserve_slots'] = int(reserve_input) if reserve_input else default_config['reserve_slots']
        
        # Max submit per callback
        print(f"\n3. Maximum jobs to submit per callback:")
        print(f"   Current default: {default_config['max_submit_batch']}")
        print(f"   Controls batch size when completed jobs trigger new submissions")
        print(f"   Higher values = faster throughput, lower values = more stability")
        submit_input = input(f"   Enter value [{default_config['max_submit_batch']}]: ").strip()
        config['max_submit_batch'] = int(submit_input) if submit_input else default_config['max_submit_batch']
        
        # Recovery attempts
        print(f"\n4. Maximum error recovery attempts:")
        print(f"   Current default: {default_config['max_recovery_attempts']}")
        print(f"   How many times to retry failed calculations with fixes")
        recovery_input = input(f"   Enter value [{default_config['max_recovery_attempts']}]: ").strip()
        config['max_recovery_attempts'] = int(recovery_input) if recovery_input else default_config['max_recovery_attempts']
        
        print(f"\nQueue management configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        return config

    def main_interactive_workflow(self):
        """Main interactive workflow planning interface"""
        self.display_welcome()

        # Step 1: Determine input type and location
        input_type = self.get_input_type()
        input_dir = self.get_input_directory(input_type)
        input_files = self.scan_input_files(input_dir, input_type)

        if not input_files["cif"] and not input_files["d12"]:
            print("No input files found. Exiting.")
            return

        # Step 2: Plan workflow sequence first to know what type of calculation we're starting with
        workflow_sequence = self.plan_workflow_sequence()

        # Step 3: CIF conversion setup if needed (now we know the first calculation type)
        cif_config = None
        if input_files["cif"]:
            # Pass the first calculation type to setup_cif_conversion
            first_calc_type = workflow_sequence[0] if workflow_sequence else "OPT"
            cif_config = self.setup_cif_conversion(input_files["cif"], first_calc_type)

        # Step 4: Configure workflow steps
        step_configs = self.configure_workflow_steps(
            workflow_sequence, bool(input_files["cif"])
        )
        
        # Step 4.5: Choose workflow isolation mode
        print("\nStep 4.5: Workflow Isolation Settings")
        print("-" * 40)
        print("Choose how this workflow should handle its data:")
        print("  1. Isolated (recommended) - Separate database per workflow")
        print("     • No conflicts with other workflows")
        print("     • Clean separation of data")
        print("     • Easy to archive/delete")
        print("  2. Shared - Use shared database (legacy behavior)")
        print("     • All workflows share same database")
        print("     • Traditional MACE behavior")
        print("     • Risk of conflicts with concurrent workflows")
        print("  3. Hybrid - Shared schema, isolated data")
        print("     • Uses shared database structure")
        print("     • But keeps data separate")
        
        isolation_choice = input("\nSelect isolation mode [1]: ").strip() or "1"
        isolation_map = {"1": "isolated", "2": "shared", "3": "hybrid"}
        isolation_mode = isolation_map.get(isolation_choice, "isolated")
        
        # Ask about post-completion actions if using isolation
        post_completion_action = "keep"
        if isolation_mode != "shared":
            print("\nPost-completion actions:")
            print("  1. Keep workflow context active (default)")
            print("  2. Archive workflow context after completion")
            print("  3. Export results and delete context after completion")
            
            action_choice = input("\nSelect action [1]: ").strip() or "1"
            action_map = {"1": "keep", "2": "archive", "3": "export_and_delete"}
            post_completion_action = action_map.get(action_choice, "keep")

        # Step 4.8: Configure queue management settings
        queue_config = self.configure_queue_management()

        # Step 5: Create comprehensive workflow plan
        workflow_plan = {
            "created": datetime.now().isoformat(),
            "input_type": input_type,
            "input_directory": str(input_dir),
            "input_files": {k: [str(f) for f in v] for k, v in input_files.items()},
            "workflow_sequence": workflow_sequence,
            "step_configurations": step_configs,
            "cif_conversion_config": cif_config,
            "isolation_mode": isolation_mode,
            "post_completion_action": post_completion_action,
            "queue_management": queue_config,
            "execution_settings": {
                "max_concurrent_jobs": 200,
                "enable_material_tracking": True,
                "auto_progression": True,
            },
        }

        # Step 5: Copy and customize SLURM scripts
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.copy_and_customize_scripts(step_configs, workflow_id, queue_config)

        # Add script information to workflow plan
        workflow_plan["workflow_id"] = workflow_id
        workflow_plan["scripts_directory"] = str(self.work_dir / "workflow_scripts")

        # Step 6: Save and execute
        plan_file = self.save_workflow_plan(workflow_plan)

        execute_now = yes_no_prompt("Execute workflow now?", "yes")
        if execute_now:
            self.execute_workflow_plan(plan_file)
        else:
            print(f"\nWorkflow plan saved. Execute later with:")
            print(f"  python workflow_planner.py --execute {plan_file}")

    def create_clean_material_id(self, file_path: Path) -> str:
        """Create a clean material ID from file path using smart suffix removal"""
        name = file_path.stem

        # First, extract the core material identifier (before the first technical suffix)
        # Look for pattern like "materialname_opt_BULK_..." or "materialname_BULK_..."
        parts = name.split("_")

        # Find the first part that looks like a technical suffix
        core_parts = []
        for i, part in enumerate(parts):
            # Special case: if this is "opt" and we only have one core part so far,
            # and the core part ends with a number, this might be a calc type rather than material identifier
            if (
                part.upper() == "OPT"
                and len(core_parts) == 1
                and core_parts[0]
                and core_parts[0][-1].isdigit()
            ):
                # This looks like "test1_opt" - the opt is a calc type, not part of material name
                break
            # Check if this part is a technical suffix (removed OPT from this list since we handle it specially)
            elif part.upper() in [
                "SP",
                "FREQ",
                "BAND",
                "DOSS",
                "BULK",
                "OPTGEOM",
                "CRYSTAL",
                "SLAB",
                "POLYMER",
                "MOLECULE",
                "SYMM",
                "TZ",
                "DZ",
                "SZ",
            ]:
                break
            # Check if this part is a DFT functional
            elif part.upper() in [
                "PBE",
                "B3LYP",
                "HSE06",
                "PBE0",
                "SCAN",
                "BLYP",
                "BP86",
            ]:
                break
            # Check if this part contains basis set info
            elif (
                "POB" in part.upper()
                or "TZVP" in part.upper()
                or "DZVP" in part.upper()
            ):
                break
            # Check if this part is a dispersion correction
            elif "D3" in part.upper():
                break
            else:
                core_parts.append(part)

        # If we found core parts, use them
        if core_parts:
            clean_name = "_".join(core_parts)
        else:
            # Fallback: just use the first part
            clean_name = parts[0] if parts else name

        # Handle special characters that might need preservation
        # Don't remove things like numbers, hyphens in material names

        return clean_name


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MACE Workflow Planner")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    parser.add_argument("--execute", help="Execute saved workflow plan")

    args = parser.parse_args()

    planner = WorkflowPlanner(args.work_dir, args.db_path)

    if args.execute:
        plan_file = Path(args.execute)
        if plan_file.exists():
            planner.execute_workflow_plan(plan_file)
        else:
            print(f"Workflow plan file not found: {plan_file}")
    else:
        planner.main_interactive_workflow()


def get_user_input(prompt: str, options: Dict[str, str], default: str = "") -> str:
    """Get user input with options"""
    print(f"\n{prompt}:")
    for key, desc in options.items():
        print(f"  {key}: {desc}")

    while True:
        choice = input(f"Enter choice [{default}]: ").strip()
        if not choice and default:
            return default
        elif choice in options:
            return choice
        else:
            print(f"Invalid choice. Please select from: {', '.join(options.keys())}")


def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """Get yes/no response from user with validation"""
    default_char = "Y/n" if default.lower() == "yes" else "y/N"

    while True:
        response = input(f"{prompt} [{default_char}]: ").strip().lower()

        if not response:
            return default.lower() == "yes"

        if response in ["y", "yes", "true", "1"]:
            return True
        elif response in ["n", "no", "false", "0"]:
            return False
        else:
            print(f"⚠️  Invalid response '{response}'. Please enter yes/no (y/n).")
            continue


if __name__ == "__main__":
    main()
