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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml

# Import our components
try:
    from material_database import MaterialDatabase, create_material_id_from_file
    from enhanced_queue_manager import EnhancedCrystalQueueManager
    from workflow_engine import WorkflowEngine
    # Add the Crystal_To_CIF directory to path for importing
    sys.path.append(str(Path(__file__).parent.parent / "Crystal_To_CIF"))
    from d12creation import *
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


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
        self.db = MaterialDatabase(db_path)
        
        # Create necessary directories
        self.configs_dir = self.work_dir / "workflow_configs"
        self.outputs_dir = self.work_dir / "workflow_outputs"
        self.temp_dir = self.work_dir / "workflow_temp"
        self.scripts_dir = self.work_dir / "workflow_scripts"
        
        for dir_path in [self.configs_dir, self.outputs_dir, self.temp_dir, self.scripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Available calculation types and their dependencies
        # Note: This only includes base types - numbered versions are handled dynamically
        self.calc_types = {
            "OPT": {"name": "Geometry Optimization", "depends_on": [], "generates": ["optimized_geometry"]},
            "SP": {"name": "Single Point", "depends_on": ["OPT"], "generates": ["electronic_structure", "wavefunction"]},
            "BAND": {"name": "Band Structure", "depends_on": ["SP"], "generates": ["band_structure"]},
            "DOSS": {"name": "Density of States", "depends_on": ["SP"], "generates": ["density_of_states"]},
            "FREQ": {"name": "Frequencies", "depends_on": ["OPT"], "generates": ["vibrational_modes"]},
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
            "custom": "user_defined"
        }
        
    def display_welcome(self):
        """Display welcome message and overview"""
        print("=" * 80)
        print("CRYSTAL Workflow Planner")
        print("=" * 80)
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
            "3": "Mixed (some CIFs, some D12s)"
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
            input_dir = input(f"\n{default_prompt} (default: current directory): ").strip()
            if not input_dir:
                input_dir = "."
                
            input_path = Path(input_dir).resolve()
            if input_path.exists() and input_path.is_dir():
                return input_path
            else:
                print(f"Directory {input_path} does not exist. Please try again.")
                
    def scan_input_files(self, input_dir: Path, input_type: str) -> Dict[str, List[Path]]:
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
        
    def setup_cif_conversion(self, cif_files: List[Path]) -> Dict[str, Any]:
        """Set up CIF to D12 conversion using NewCifToD12.py"""
        print("\nStep 2: CIF Conversion Setup")
        print("-" * 40)
        print("Configuring CIF to D12 conversion using NewCifToD12.py")
        
        # Ask if user wants to use default settings or customize
        use_defaults = yes_no_prompt("Use default CIF conversion settings?", "yes")
        
        if use_defaults:
            # Use sensible defaults
            cif_config = {
                "symmetry_handling": "CIF",
                "write_only_unique": True,
                "dimensionality": "CRYSTAL",
                "calculation_type": "OPT",
                "optimization_type": "FULLOPTG",
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
                "fmixing": 30
            }
            print("Using default settings for CIF conversion:")
            print(f"  Method: DFT/HSE06-D3")
            print(f"  Basis set: POB-TZVP-REV2 (internal)")
            print(f"  Calculation: Geometry optimization")
            print(f"  Symmetry: Use CIF symmetry information")
        else:
            print("Custom CIF conversion setup:")
            print("Choose customization level:")
            print("  1: Basic (functional + basis set)")
            print("  2: Advanced (most common settings)")
            print("  3: Expert (full NewCifToD12.py integration)")
            
            level_choice = self.get_safe_choice_input(
                "Customization level", 
                valid_choices=["1", "2", "3"],
                default="1"
            )
            
            if level_choice == "1":
                cif_config = self.get_basic_customization()
            elif level_choice == "2":
                cif_config = self.get_advanced_customization()
            elif level_choice == "3":
                print("\nLaunching NewCifToD12.py for full configuration...")
                print("This will run the interactive configuration and save the results.")
                cif_config = self.run_full_cif_customization(cif_files[0])
            else:
                print("Invalid choice. Using basic customization.")
                cif_config = self.get_basic_customization()
        
        # Save CIF configuration
        cif_config_file = self.configs_dir / "cif_conversion_config.json"
        with open(cif_config_file, 'w') as f:
            json.dump(cif_config, f, indent=2)
            
        print(f"CIF conversion config saved to: {cif_config_file}")
        
        return cif_config
        
    def get_default_cif_config(self) -> Dict[str, Any]:
        """Return sensible default CIF configuration"""
        return {
            "symmetry_handling": "CIF",
            "write_only_unique": True,
            "dimensionality": "CRYSTAL",
            "calculation_type": "OPT",
            "optimization_type": "FULLOPTG",
            "optimization_settings": {
                "TOLDEG": 0.00003,
                "TOLDEX": 0.00012,
                "TOLDEE": 7,
                "MAXCYCLE": 800
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
            "fmixing": 30
        }
        
    def get_basic_customization(self) -> Dict[str, Any]:
        """Get basic CIF customization options"""
        print("\nBasic customization options:")
        functional_choice = input("DFT Functional [HSE06]: ").strip() or "HSE06"
        basis_choice = input("Basis set [POB-TZVP-REV2]: ").strip() or "POB-TZVP-REV2"
        
        cif_config = self.get_default_cif_config()
        cif_config["dft_functional"] = functional_choice
        cif_config["basis_set"] = basis_choice
        
        print(f"Using customized settings:")
        print(f"  Method: DFT/{functional_choice}")
        print(f"  Basis set: {basis_choice}")
        print(f"  Other settings: Using defaults")
        
        return cif_config
        
    def get_advanced_customization(self) -> Dict[str, Any]:
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
            print("DFT Functional:")
            print("  1: HSE06")
            print("  2: B3LYP")
            print("  3: PBE0")
            print("  4: PBE")
            func_choice = input("Functional [1]: ").strip() or "1"
            functionals = {"1": "HSE06", "2": "B3LYP", "3": "PBE0", "4": "PBE"}
            functional = functionals.get(func_choice, "HSE06")
        
        # Basis set
        print("Basis set:")
        print("  1: POB-TZVP-REV2 (internal)")
        print("  2: 6-31G* (internal)")
        print("  3: def2-TZVP (internal)")
        print("  4: Custom external")
        basis_choice = input("Basis set [1]: ").strip() or "1"
        basis_options = {
            "1": "POB-TZVP-REV2",
            "2": "6-31G*", 
            "3": "def2-TZVP",
            "4": "EXTERNAL"
        }
        basis_set = basis_options.get(basis_choice, "POB-TZVP-REV2")
        basis_type = "EXTERNAL" if basis_choice == "4" else "INTERNAL"
        
        # Dispersion correction
        dispersion = yes_no_prompt("Use dispersion correction (D3)?", "yes")
        
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
        
        # Build configuration
        cif_config = self.get_default_cif_config()
        cif_config.update({
            "method": method,
            "dft_functional": functional,
            "basis_set": basis_set,
            "basis_set_type": basis_type,
            "use_dispersion": dispersion,
            "is_spin_polarized": spin_polarized,
            "optimization_type": opt_type
        })
        
        print(f"\nAdvanced configuration:")
        print(f"  Method: {method}/{functional if method=='DFT' else 'HF'}")
        print(f"  Basis set: {basis_set} ({basis_type})")
        print(f"  Dispersion: {'Yes' if dispersion else 'No'}")
        print(f"  Spin polarized: {'Yes' if spin_polarized else 'No'}")
        print(f"  Optimization: {opt_type}")
        
        return cif_config
        
    def run_full_cif_customization(self, sample_cif: Path) -> Dict[str, Any]:
        """Run full NewCifToD12.py customization"""
        # Find NewCifToD12.py
        script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "NewCifToD12.py"
        
        if not script_path.exists():
            print(f"Error: NewCifToD12.py not found at {script_path}")
            print("Using advanced customization instead...")
            return self.get_advanced_customization()
        
        print(f"\nRunning NewCifToD12.py with sample file: {sample_cif.name}")
        print("This will launch the full interactive configuration.")
        print("At the end, choose to SAVE the configuration for batch processing.")
        print("Press Enter to continue...")
        input()
        
        # Create temporary options file
        temp_options = self.temp_dir / "temp_cif_options.json"
        
        # Run NewCifToD12.py interactively
        cmd = [
            sys.executable, str(script_path),
            "--cif_dir", str(sample_cif.parent),
            "--save_options",
            "--options_file", str(temp_options)
        ]
        
        print("Launching NewCifToD12.py...")
        try:
            # Run interactively (not captured)
            result = subprocess.run(cmd, cwd=str(sample_cif.parent))
            
            if result.returncode == 0 and temp_options.exists():
                # Load the saved configuration
                with open(temp_options, 'r') as f:
                    cif_config = json.load(f)
                print("Successfully loaded configuration from NewCifToD12.py")
                return cif_config
            else:
                print("NewCifToD12.py configuration failed or was cancelled.")
                print("Using default configuration...")
                return self.get_default_cif_config()
                
        except Exception as e:
            print(f"Error running NewCifToD12.py: {e}")
            print("Using default configuration...")
            return self.get_default_cif_config()
        
        
        
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
                
        template_options = {str(i): key for i, key in enumerate(self.workflow_templates.keys(), 1)}
        
        template_choice = get_user_input("Select workflow template", template_options, "4")
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
            deps = " (depends on: " + ", ".join(info["depends_on"]) + ")" if info["depends_on"] else ""
            print(f"  {calc_type}: {info['name']}{deps}")
            
        print("\nNOTE: Numbered calculations (OPT2, SP2, BAND2, etc.) will be automatically assigned")
        print("when you add multiple calculations of the same type.\n")
        
        print("Build your workflow sequence:")
        print("Enter calculation types in order (e.g., OPT SP BAND DOSS)")
        print("Type 'help' for more information")
        
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
                    base = calc_type.rstrip('0123456789')
                    desc = {
                        'OPT': 'Geometry Optimization',
                        'SP': 'Single Point Energy',
                        'BAND': 'Band Structure',
                        'DOSS': 'Density of States',
                        'FREQ': 'Vibrational Frequencies'
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
                elif calc in ["OPT", "SP", "BAND", "DOSS", "FREQ"]:
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
                    print("Enter the exact calculation to remove (e.g., OPT2, SP, BAND2):")
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
                if calc in ["OPT", "SP", "BAND", "DOSS", "FREQ"]:
                    numbered_calc = self._get_next_numbered_calc(sequence, calc)
                    
                    print(f"\nCurrent sequence: {' → '.join(sequence)}")
                    print("Positions:")
                    for i in range(len(sequence) + 1):
                        if i == 0:
                            print(f"  1: Before {sequence[0]}")
                        elif i == len(sequence):
                            print(f"  {i+1}: After {sequence[-1]}")
                        else:
                            print(f"  {i+1}: Between {sequence[i-1]} and {sequence[i]}")
                            
                    pos = input("Insert at position: ").strip()
                    try:
                        pos = int(pos) - 1
                        if 0 <= pos <= len(sequence):
                            if self._validate_numbered_calc_addition(sequence[:pos] + sequence[pos:], numbered_calc):
                                sequence.insert(pos, numbered_calc)
                                print(f"Inserted {numbered_calc}. Current: {' → '.join(sequence)}")
                            else:
                                print(f"Cannot insert {numbered_calc} at this position - check dependencies")
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
        base_types = ["OPT", "SP", "BAND", "DOSS", "FREQ"]
        
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
            if calc == base_type or calc.startswith(base_type) and calc[len(base_type):].isdigit():
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
            base = calc.rstrip('0123456789')
            num = calc[len(base):] or '1'
            
            # Get description
            desc = {
                'OPT': 'Geometry Optimization',
                'SP': 'Single Point Energy',
                'BAND': 'Band Structure',
                'DOSS': 'Density of States', 
                'FREQ': 'Vibrational Frequencies'
            }.get(base, base)
            
            # Determine source
            if i == 0:
                source = "Input files"
            elif base == "OPT":
                source = f"From step {i}"
            elif base == "SP":
                # Find most recent OPT
                for j in range(i-1, -1, -1):
                    if sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j+1})"
                        break
                else:
                    source = "From previous step"
            elif base in ["BAND", "DOSS"]:
                # Find most recent SP or OPT
                for j in range(i-1, -1, -1):
                    if sequence[j].startswith("SP") or sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j+1})"
                        break
                else:
                    source = "From previous step"
            elif base == "FREQ":
                # Find most recent OPT
                for j in range(i-1, -1, -1):
                    if sequence[j].startswith("OPT"):
                        source = f"From {sequence[j]} (step {j+1})"
                        break
                else:
                    source = "From previous step"
            else:
                source = "From previous step"
                
            # Print step info
            print(f"  Step {i+1}: {calc:<8} - {desc:<25} [{source}]")
            
            if i < len(sequence) - 1:
                print("    ↓")
                
        print("─" * 70)
        
    def _validate_numbered_calc_addition(self, sequence: List[str], new_calc: str) -> bool:
        """Validate addition of numbered calculation types"""
        # Parse the calculation type
        import re
        match = re.match(r'^([A-Z]+)(\d*)$', new_calc)
        if not match:
            return False
            
        base_type = match.group(1)
        
        # Check dependencies based on base type
        if base_type == "SP":
            # SP needs at least one OPT
            return any(calc.startswith("OPT") for calc in sequence)
        elif base_type in ["BAND", "DOSS"]:
            # BAND/DOSS need at least one SP or OPT
            return any(calc.startswith("SP") or calc.startswith("OPT") for calc in sequence)
        elif base_type == "FREQ":
            # FREQ needs at least one OPT
            return any(calc.startswith("OPT") for calc in sequence)
        elif base_type == "OPT":
            # OPT can always be added
            return True
            
        return False
        
    def validate_calc_addition(self, current_sequence: List[str], new_calc: str) -> bool:
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
        print("  Electronic: OPT SP")
        print("  Analysis: OPT SP BAND DOSS")
        print("  Complete: OPT SP BAND DOSS FREQ")
        print("  Double opt: OPT OPT2 SP")
        print("\nAdvanced patterns:")
        print("  Multi-stage: OPT SP BAND DOSS OPT2 OPT3 SP2 BAND2 DOSS2 FREQ")
        print("  Iterative opt: OPT OPT2 OPT3 SP")
        print("  Multiple properties: OPT SP BAND DOSS BAND2 DOSS2")
        print("\nDependencies:")
        for calc, info in self.calc_types.items():
            if info["depends_on"]:
                print(f"  {calc} requires: {', '.join(info['depends_on'])}")
        print("\nNOTE: Numbered calculations (OPT2, SP2, etc.) are automatically assigned")
        print("      when you add multiple calculations of the same type.")
                
    def configure_workflow_steps(self, sequence: List[str], has_cifs: bool) -> Dict[str, Dict[str, Any]]:
        """Configure settings for each workflow step"""
        print(f"\nStep 4: Configure Workflow Steps")
        print("-" * 40)
        
        step_configs = {}
        
        for i, calc_type in enumerate(sequence):
            # Parse calc_type to get base type for looking up info
            base_type = calc_type.rstrip('0123456789') or calc_type
            if base_type in self.calc_types:
                calc_name = self.calc_types[base_type]['name']
            else:
                calc_name = calc_type
            print(f"\nConfiguring {calc_type} ({calc_name}):")
            
            if calc_type == "OPT" and i == 0 and has_cifs:
                print("  Using CIF conversion configuration for first OPT step")
                step_configs[f"{calc_type}_1"] = {"source": "cif_conversion"}
                
            elif calc_type == "OPT" and i == 0 and not has_cifs:
                print("  Using existing D12 files for first OPT step")
                step_configs[f"{calc_type}_1"] = {"source": "existing_d12"}
                
            elif calc_type == "OPT2":
                # OPT2 is always configured via CRYSTALOptToD12.py
                config = self.configure_optimization_step(calc_type, i+1)
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type == "OPT" and i > 0:
                # This shouldn't happen - subsequent OPTs should be OPT2, OPT3, etc.
                print(f"  Warning: Found duplicate OPT at position {i+1}. This should be OPT2.")
                config = self.configure_optimization_step(calc_type, i+1)
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type.startswith("OPT") and calc_type[3:].isdigit():
                # Handle OPT3, OPT4, etc.
                config = self.configure_optimization_step(calc_type, i+1)
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type == "SP" or (calc_type.startswith("SP") and calc_type[2:].isdigit()):
                # Single point calculation (SP, SP2, SP3, etc.)
                config = self.configure_single_point_step()
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type.startswith("BAND"):
                # Band structure calculations (BAND, BAND2, etc.)
                config = self.configure_analysis_step("BAND")
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type.startswith("DOSS"):
                # DOS calculations (DOSS, DOSS2, etc.)
                config = self.configure_analysis_step("DOSS")
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type == "FREQ":
                # Frequency calculation
                config = self.configure_frequency_step()
                step_configs[f"{calc_type}_{i+1}"] = config
            
            # Configure SLURM scripts for this step
            slurm_config = self.configure_slurm_scripts(calc_type, i+1)
            step_configs[f"{calc_type}_{i+1}"]["slurm_config"] = slurm_config
                
        return step_configs
        
    def configure_optimization_step(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Configure optimization calculation step"""
        print(f"  Configuring {calc_type} step {step_num}")
        
        use_defaults = yes_no_prompt("    Use default optimization settings?", "yes")
        
        if use_defaults:
            config = {
                "calculation_type": "OPT",
                "optimization_type": "FULLOPTG", 
                "optimization_settings": {
                    "TOLDEG": 0.00003,
                    "TOLDEX": 0.00012,
                    "TOLDEE": 7,
                    "MAXCYCLE": 800
                },
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True
            }
        else:
            # Would run CRYSTALOptToD12.py configuration
            config = self.get_detailed_opt_config(calc_type, step_num)
            
        return config
        
    def configure_single_point_step(self) -> Dict[str, Any]:
        """Configure single point calculation with customization levels"""
        print("  Configuring single point calculation")
        
        print("    Choose SP customization level:")
        print("      0: Default (inherit all settings from previous OPT)")
        print("      1: Basic (modify method/basis set)")
        print("      2: Advanced (detailed SCF/convergence settings)")
        print("      3: Expert (full CRYSTALOptToD12.py integration)")
        
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
            "customization_level": level
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
            config.update(self._get_expert_sp_config())
            
        return config
        
    def _get_basic_sp_config(self) -> Dict[str, Any]:
        """Get basic SP configuration"""
        print("\n    Basic SP Setup:")
        
        config = {
            "inherit_geometry": True,
            "inherit_settings": False
        }
        
        # Method modifications
        modify_method = yes_no_prompt("    Change DFT functional?", "no")
        if modify_method:
            config["method_modifications"] = self._get_method_modifications()
        else:
            config["inherit_method"] = True
            
        # Basis set modifications
        modify_basis = yes_no_prompt("    Change basis set?", "no")
        if modify_basis:
            config["basis_modifications"] = self._get_basis_modifications()
        else:
            config["inherit_basis"] = True
            
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
        
    def _get_expert_sp_config(self) -> Dict[str, Any]:
        """Get expert SP configuration"""
        print("\n    Expert SP Setup:")
        print("    This will run CRYSTALOptToD12.py interactively for full customization")
        
        proceed = yes_no_prompt("    Proceed with expert SP configuration?", "yes")
        if not proceed:
            return self._get_advanced_sp_config()
        
        # Copy required scripts early so we can use them
        self._copy_required_scripts_for_expert_mode()
        
        # Run CRYSTALOptToD12.py interactively NOW during planning
        expert_config = self._run_interactive_crystal_opt_config("SP")
        
        if expert_config:
            print(f"    ✅ Expert SP configuration completed successfully")
            return expert_config
        else:
            print(f"    ❌ Expert SP configuration failed, falling back to advanced mode")
            return self._get_advanced_sp_config()
        
    def _get_basis_modifications(self) -> Dict[str, Any]:
        """Get basis set modification settings"""
        modifications = {}
        
        print("      Basis set options:")
        print("        1: POB-TZVP-REV2 (high quality triple-zeta)")
        print("        2: def2-TZVP (balanced triple-zeta)")
        print("        3: Custom basis set")
        
        basis_choice = input("      Choose (1-3): ").strip()
        basis_map = {
            "1": "POB-TZVP-REV2",
            "2": "def2-TZVP", 
            "3": "custom"
        }
        
        if basis_choice in basis_map:
            modifications["new_basis"] = basis_map[basis_choice]
            if basis_choice == "3":
                custom_basis = input("      Enter custom basis set name: ").strip()
                if custom_basis:
                    modifications["new_basis"] = custom_basis
                    
        return modifications
        
    def _get_scf_modifications(self) -> Dict[str, Any]:
        """Get SCF modification settings"""
        modifications = {}
        
        # TOLDEE
        modify_toldee = yes_no_prompt("      Change TOLDEE (SCF convergence)?", "no")
        if modify_toldee:
            toldee = input("      New TOLDEE value (current: from OPT): ").strip()
            if toldee:
                try:
                    modifications["TOLDEE"] = int(toldee)
                except ValueError:
                    print("      Invalid TOLDEE, keeping default")
                    
        # FMIXING
        modify_fmixing = yes_no_prompt("      Change FMIXING (SCF mixing)?", "no")
        if modify_fmixing:
            fmixing = input("      New FMIXING value (current: from OPT): ").strip()
            if fmixing:
                try:
                    modifications["FMIXING"] = int(fmixing)
                except ValueError:
                    print("      Invalid FMIXING, keeping default")
                    
        return modifications
        
    def _get_grid_modifications(self) -> Dict[str, Any]:
        """Get DFT grid modification settings"""
        modifications = {}
        
        print("      DFT integration grid options:")
        print("        1: XLGRID (extra large, high accuracy)")
        print("        2: LGRID (large, good accuracy)")
        print("        3: MGRID (medium, balanced)")
        
        grid_choice = input("      Choose grid (1-3): ").strip()
        grid_map = {
            "1": "XLGRID",
            "2": "LGRID",
            "3": "MGRID"
        }
        
        if grid_choice in grid_map:
            modifications["new_grid"] = grid_map[grid_choice]
            
        return modifications
        
    def configure_analysis_step(self, calc_type: str) -> Dict[str, Any]:
        """Configure band structure or DOS calculation"""
        print(f"  Configuring {calc_type} calculation")
        
        if calc_type == "BAND":
            script = "create_band_d3.py"
        else:  # DOSS
            script = "alldos.py"
            
        config = {
            "calculation_type": calc_type,
            "source": script,
            "requires_wavefunction": True,
            "isolated_directory": True
        }
        
        return config
        
    def configure_frequency_step(self) -> Dict[str, Any]:
        """Configure frequency calculation with customization levels"""
        print("  Configuring frequency calculation")
        
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
            config = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {
                    "mode": "FREQCALC",
                    "intensities": True,
                    "raman": False,
                    "custom_tolerances": {
                        "TOLINTEG": "12 12 12 12 24",
                        "TOLDEE": 12
                    }
                }
            }
            
        elif level == 2:
            # Advanced - customize key parameters
            config = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {}
            }
            
            # Ask about intensities
            calc_intensities = input("\n  Calculate IR intensities? [Y/n]: ").strip().lower()
            config["frequency_settings"]["intensities"] = calc_intensities != 'n'
            
            # Ask about Raman
            calc_raman = input("  Calculate Raman intensities? [y/N]: ").strip().lower()
            config["frequency_settings"]["raman"] = calc_raman == 'y'
            
            # Ask about modes
            print("\n  Frequency calculation mode:")
            print("    1. Full frequency calculation (FREQCALC)")
            print("    2. Partial frequencies (FREQRANGE)")
            print("    3. Numerical frequencies (NUMFREQ)")
            
            mode_choice = input("  Select mode [1]: ").strip() or "1"
            modes = {
                "1": "FREQCALC",
                "2": "FREQRANGE",
                "3": "NUMFREQ"
            }
            config["frequency_settings"]["mode"] = modes.get(mode_choice, "FREQCALC")
            
            # Custom tolerances
            print("\n  Use enhanced tolerances for frequency calculations?")
            enhanced = input("  [Y/n]: ").strip().lower()
            if enhanced != 'n':
                config["frequency_settings"]["custom_tolerances"] = {
                    "TOLINTEG": "12 12 12 12 24",
                    "TOLDEE": 12
                }
                
        else:
            # Expert - run CRYSTALOptToD12.py interactively
            print("\n  Expert mode: Full interactive configuration")
            
            # Copy required scripts early
            self._copy_required_scripts_for_expert_mode()
            
            # Run CRYSTALOptToD12.py interactively for FREQ configuration
            expert_config = self._run_interactive_crystal_opt_config("FREQ")
            
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
                        "mode": "FREQCALC",
                        "intensities": True,
                        "raman": True,
                        "custom_tolerances": {
                            "TOLINTEG": "12 12 12 12 24",
                            "TOLDEE": 12
                        }
                    }
                }
                return config
        
        return config
        
    def get_detailed_opt_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Get detailed optimization configuration with expert mode"""
        print(f"    Custom {calc_type} configuration:")
        print(f"    Choose customization level:")
        print(f"      1: Basic (optimization type + tolerances)")
        print(f"      2: Advanced (method + basis set modifications)")
        print(f"      3: Expert (full CRYSTALOptToD12.py integration)")
        
        while True:
            try:
                level = int(input("    Enter level (1-3): ").strip())
                if level in [1, 2, 3]:
                    break
                print("    Please enter 1, 2, or 3")
            except ValueError:
                print("    Please enter a valid number")
        
        config = {
            "calculation_type": "OPT",
            "source": "CRYSTALOptToD12.py",
            "customization_level": level
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
            
        return config
    
    def _get_basic_opt_config(self) -> Dict[str, Any]:
        """Get basic optimization configuration"""
        print("\n    Basic Optimization Setup:")
        
        # Optimization type
        print("    Optimization type:")
        print("      1: FULLOPTG (optimize atoms and cell)")
        print("      2: ATOMSONLY (optimize atoms only)")
        print("      3: CELLONLY (optimize cell only)")
        
        opt_choice = input("    Choose optimization type (1-3, default 1): ").strip() or "1"
        opt_types = {"1": "FULLOPTG", "2": "ATOMSONLY", "3": "CELLONLY"}
        opt_type = opt_types.get(opt_choice, "FULLOPTG")
        
        # Enhanced tolerances for subsequent optimizations
        use_tight = yes_no_prompt("    Use tighter convergence for refined optimization?", "yes")
        
        if use_tight:
            opt_settings = {
                "TOLDEG": 1.5e-5,   # Tighter than default 3e-5
                "TOLDEX": 6e-5,     # Tighter than default 1.2e-4  
                "TOLDEE": 8,        # Tighter than default 7
                "MAXCYCLE": 1000    # More cycles for convergence
            }
        else:
            opt_settings = {
                "TOLDEG": 3e-5,
                "TOLDEX": 1.2e-4,
                "TOLDEE": 7,
                "MAXCYCLE": 800
            }
        
        return {
            "optimization_type": opt_type,
            "optimization_settings": opt_settings,
            "inherit_base_settings": True
        }
    
    def _get_advanced_opt_config(self) -> Dict[str, Any]:
        """Get advanced optimization configuration"""
        config = self._get_basic_opt_config()
        
        print("\n    Advanced Optimization Setup:")
        
        # Method modifications
        modify_method = yes_no_prompt("    Modify DFT method from previous step?", "no")
        if modify_method:
            config["modify_method"] = True
            config["method_settings"] = self._get_method_modifications()
        
        # Custom tolerances
        custom_tolerances = yes_no_prompt("    Set custom TOLINTEG/SCF tolerances?", "no")
        if custom_tolerances:
            config["custom_tolerances"] = self._get_custom_tolerances()
            
        return config
    
    def _get_expert_opt_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Get expert optimization configuration with full CRYSTALOptToD12.py integration"""
        print(f"\n    Expert {calc_type} Setup:")
        
        # For OPT2/OPT3, ask if user wants per-material configs
        if calc_type.startswith("OPT") and calc_type != "OPT":
            print("    Options:")
            print("    1. Create individual configuration for each material (preserves exact symmetry)")
            print("    2. Create one configuration for all materials")
            
            config_choice = input("    Choose configuration mode (1/2) [1]: ").strip() or "1"
            
            if config_choice == "1":
                return self._get_per_material_expert_config(calc_type, step_num)
        
        print(f"    This will run CRYSTALOptToD12.py interactively for full customization")
        
        proceed = yes_no_prompt("    Proceed with expert configuration?", "yes")
        if not proceed:
            return self._get_advanced_opt_config()
        
        # Copy required scripts early so we can use them
        self._copy_required_scripts_for_expert_mode()
        
        # Run CRYSTALOptToD12.py interactively NOW during planning
        # For OPT2, OPT3 etc., we pass "OPT" as calc type to CRYSTALOptToD12.py
        crystal_calc_type = "OPT" if calc_type.startswith("OPT") else calc_type
        # Pass the full calc_type (e.g., OPT2, OPT3) instead of just "OPT"
        expert_config = self._run_interactive_crystal_opt_config(calc_type)
        
        if expert_config:
            # Add step-specific information
            expert_config["step_num"] = step_num
            expert_config["workflow_calc_type"] = calc_type  # Keep original OPT2, OPT3 etc.
            print(f"    ✅ Expert {calc_type} configuration completed successfully")
            return expert_config
        else:
            print(f"    ❌ Expert {calc_type} configuration failed, falling back to advanced mode")
            return self._get_advanced_opt_config()
    
    def _get_method_modifications(self) -> Dict[str, Any]:
        """Get DFT method modification settings"""
        modifications = {}
        
        # Functional change
        change_functional = yes_no_prompt("      Change DFT functional?", "no")
        if change_functional:
            print("      Common functional changes for refinement:")
            print("        1: PBE → HSE06 (hybrid for better band gaps)")
            print("        2: B3LYP → PBE0 (different hybrid)")
            print("        3: Custom functional")
            
            func_choice = input("      Choose (1-3 or 'n' to skip): ").strip()
            if func_choice in ["1", "2", "3"]:
                modifications["change_functional"] = func_choice
        
        # Basis set change
        change_basis = yes_no_prompt("      Change basis set?", "no")
        if change_basis:
            modifications["change_basis"] = True
            
        return modifications
    
    def _get_custom_tolerances(self) -> Dict[str, Any]:
        """Get custom tolerance settings"""
        tolerances = {}
        
        print("      Custom tolerance settings:")
        
        # TOLINTEG
        custom_tolinteg = input("      TOLINTEG (e.g., '8 8 8 8 16', default: use previous): ").strip()
        if custom_tolinteg:
            tolerances["TOLINTEG"] = custom_tolinteg
            
        # TOLDEE
        custom_toldee = input("      TOLDEE (default: use previous): ").strip()
        if custom_toldee:
            try:
                tolerances["TOLDEE"] = int(custom_toldee)
            except ValueError:
                print("      Invalid TOLDEE, using previous value")
        
        return tolerances
    
    def _create_expert_opt_template(self, config_path: Path, calc_type: str, step_num: int):
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
                "MAXCYCLE": 1000
            },
            
            # Advanced options (to be filled interactively)
            "method_modifications": {
                "change_functional": False,
                "new_functional": "",
                "change_basis": False,
                "new_basis": "",
                "change_grid": False,
                "new_grid": ""
            },
            
            "scf_modifications": {
                "change_tolerances": False,
                "TOLINTEG": "",
                "TOLDEE": "",
                "change_mixing": False,
                "FMIXING": ""
            },
            
            # Instructions for interactive use
            "_instructions": {
                "usage": "This file will be used by CRYSTALOptToD12.py for interactive configuration",
                "modify": "Set change_* flags to true and provide new values for customization",
                "inherit": "Set inherit_* flags to false to completely override settings"
            }
        }
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save template
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        return template
        
    def _copy_required_scripts_for_expert_mode(self):
        """Copy required scripts early for expert mode configuration"""
        print("      Copying required scripts for expert configuration...")
        
        # Scripts we need for expert mode
        required_scripts = [
            "CRYSTALOptToD12.py",
            "d12creation.py",
            "StructureClass.py"
        ]
        
        # Source directory
        source_dir = Path(__file__).parent.parent / "Crystal_To_CIF"
        
        # Copy scripts to working directory
        for script_name in required_scripts:
            source_file = source_dir / script_name
            dest_file = self.work_dir / script_name
            
            if source_file.exists() and not dest_file.exists():
                shutil.copy2(source_file, dest_file)
                print(f"        Copied: {script_name}")
            elif dest_file.exists():
                print(f"        Already exists: {script_name}")
            else:
                print(f"        Warning: Source not found: {script_name}")
    
    def _extract_d12_settings(self, d12_file: Path) -> Dict[str, Any]:
        """Extract basis set, functional, and symmetry settings from a D12 file"""
        settings = {
            'basis_set': 'POB-TZVP-REV2',  # Default recommendation when external basis is used
            'functional': 'PBE-D3',  # Default
            'dft_grid': 'XLGRID',  # Default
            'spin': False,
            'spacegroup': 1,  # Default P1
            'origin_setting': '0 0 0',  # Default origin
            'dimensionality': 'CRYSTAL'  # Default
        }
        
        if not d12_file.exists():
            return settings
            
        try:
            with open(d12_file, 'r') as f:
                lines = f.readlines()
                
            # Extract structure information (skip title line)
            for i in range(1, len(lines)):
                line = lines[i].strip()
                
                # Check for dimensionality
                if line in ['CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE']:
                    settings['dimensionality'] = line
                    
                    # For CRYSTAL, next lines are origin and space group
                    if line == 'CRYSTAL' and i + 2 < len(lines):
                        settings['origin_setting'] = lines[i + 1].strip()
                        try:
                            settings['spacegroup'] = int(lines[i + 2].strip())
                        except ValueError:
                            pass
                    break
                    
            # Extract basis set - only if BASISSET keyword exists
            # If it doesn't exist, it means external basis sets are used, so keep the default
            content = ''.join(lines)
            if 'BASISSET' in content:
                for i, line in enumerate(lines):
                    if line.strip() == 'BASISSET':
                        if i + 1 < len(lines):
                            settings['basis_set'] = lines[i + 1].strip()
                            break
            
            # Extract DFT settings
            if 'DFT' in content and 'ENDDFT' in content:
                dft_start = content.find('DFT')
                dft_end = content.find('ENDDFT')
                if dft_start != -1 and dft_end != -1:
                    dft_section = content[dft_start:dft_end]
                    dft_lines = dft_section.split('\n')
                    
                    for line in dft_lines[1:]:  # Skip 'DFT' line
                        line = line.strip()
                        if line and not line.startswith('END'):
                            # Check for SPIN
                            if line == 'SPIN':
                                settings['spin'] = True
                            # Check for grid settings
                            elif 'GRID' in line:
                                settings['dft_grid'] = line
                            # Check for functional (exclude GRID lines)
                            elif line and 'GRID' not in line and line != 'SPIN':
                                settings['functional'] = line
                                
        except Exception as e:
            print(f"      Warning: Could not extract settings from D12: {e}")
            
        return settings
    
    def _create_base_expert_config(self, calc_type: str, template_d12: Path) -> Optional[Dict[str, Any]]:
        """Create base expert configuration using one D12 as template"""
        # Create temporary directory
        temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}_base"
        temp_dir.mkdir(exist_ok=True)
        
        # Create sample files from template D12
        sample_out = temp_dir / "sample.out"
        sample_d12 = temp_dir / "sample.d12"
        
        # Copy template D12 with single atom
        try:
            with open(template_d12, 'r') as f:
                lines = f.readlines()
            
            with open(sample_d12, 'w') as f:
                f.write("Sample D12 for configuration\n")
                
                i = 1
                while i < len(lines):
                    line = lines[i].strip()
                    f.write(lines[i])
                    
                    # After cell parameters, modify atom section
                    try:
                        parts = line.split()
                        if len(parts) >= 6:
                            [float(p) for p in parts[:6]]
                            if i + 1 < len(lines):
                                atom_count = int(lines[i + 1].strip())
                                f.write("1\n")
                                if i + 2 < len(lines):
                                    atom_parts = lines[i + 2].strip().split()
                                    if atom_parts:
                                        f.write(f"{atom_parts[0]} 0.0 0.0 0.0\n")
                                i += 2 + atom_count
                                while i < len(lines):
                                    f.write(lines[i])
                                    i += 1
                                break
                    except (ValueError, IndexError):
                        pass
                    i += 1
                    
        except Exception as e:
            print(f"      Error processing template D12: {e}")
            return None
        
        # Create output file
        with open(sample_out, 'w') as f:
            f.write("CRYSTAL23 OUTPUT\n")
            f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
            f.write("LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
            f.write("A              B              C           ALPHA      BETA       GAMMA\n")
            f.write("5.0000         5.0000         5.0000      90.000     90.000     90.000\n")
        
        # Config file
        base_config_file = temp_dir / f"{calc_type.lower()}_base_config.json"
        
        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "CRYSTALOptToD12.py"
        
        # Run CRYSTALOptToD12.py
        cmd = [
            sys.executable, str(script_path),
            "--out-file", str(sample_out),
            "--d12-file", str(sample_d12),
            "--output-dir", str(temp_dir),
            "--save-options",
            "--options-file", str(base_config_file)
        ]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.work_dir))
            
            if result.returncode == 0 and base_config_file.exists():
                with open(base_config_file, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            print(f"      Error running CRYSTALOptToD12.py: {e}")
            
        return None
    
    def _get_per_material_expert_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Create individual expert configurations for each material"""
        print(f"\n    Creating per-material {calc_type} configurations...")
        
        # Find D12 files from previous step
        d12_files = []
        prev_step = step_num - 1
        
        # Look for D12 files in likely locations
        search_patterns = [
            f"workflow_outputs/*/step_{prev_step:03d}_*/*/*.d12",
            f"workflow_inputs/step_{prev_step:03d}_*/*.d12",
            "*.d12"
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
        config_dir = self.work_dir / "workflow_configs" / f"expert_{calc_type.lower()}_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy required scripts
        self._copy_required_scripts_for_expert_mode()
        
        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "CRYSTALOptToD12.py"
        
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
            symmetry_settings = self._extract_d12_settings(d12_file)
            
            # Create material-specific config by updating base config
            material_config = base_config.copy()
            material_config.update({
                'spacegroup': symmetry_settings.get('spacegroup', 1),
                'origin_setting': symmetry_settings.get('origin_setting', '0 0 0'),
                'dimensionality': symmetry_settings.get('dimensionality', 'CRYSTAL'),
                # Set write_only_unique based on space group
                'write_only_unique': symmetry_settings.get('spacegroup', 1) != 1
            })
            
            # Save material-specific config
            config_file = config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
            with open(config_file, 'w') as f:
                json.dump(material_config, f, indent=2)
            
            material_configs[material_name] = {
                "config_file": str(config_file),
                "source_d12": str(d12_file)
            }
            
            print(f"      ✅ {material_name}: spacegroup={material_config['spacegroup']}, " + 
                  f"origin={material_config['origin_setting']}")
        
        if not material_configs:
            print("    No configurations created successfully")
            return self._get_advanced_opt_config()
        
        return {
            "expert_mode": True,
            "per_material_configs": True,
            "config_directory": str(config_dir),
            "material_configs": material_configs,
            "step_num": step_num,
            "workflow_calc_type": calc_type
        }
    
    def _run_interactive_crystal_opt_config(self, calc_type: str) -> Optional[Dict[str, Any]]:
        """Run CRYSTALOptToD12.py interactively for expert configuration"""
        # Find CRYSTALOptToD12.py
        local_script = self.work_dir / "CRYSTALOptToD12.py"
        if local_script.exists():
            script_path = local_script
        else:
            script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "CRYSTALOptToD12.py"
            
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
        d12_search_dirs = [
            self.work_dir,
            self.work_dir.parent if self.work_dir.parent.exists() else None,
            Path.cwd()
        ]
        
        for search_dir in d12_search_dirs:
            if search_dir and search_dir.exists():
                d12_files = list(search_dir.glob("*.d12"))
                if d12_files:
                    # Copy the first D12 file found
                    source_d12 = d12_files[0]
                    print(f"      Using real D12 as template: {source_d12.name}")
                    
                    try:
                        with open(source_d12, 'r') as f:
                            lines = f.readlines()
                        
                        # Modify the D12 to have just one atom at origin
                        with open(sample_d12, 'w') as f:
                            # Keep title
                            f.write("Sample D12 for configuration\n")
                            
                            # Copy everything up to the atom count
                            i = 1
                            while i < len(lines):
                                line = lines[i].strip()
                                f.write(lines[i])
                                
                                # After cell parameters, change atom count to 1
                                # Check if this line contains cell parameters (6 numbers)
                                try:
                                    parts = line.split()
                                    if len(parts) >= 6:
                                        # Try to parse as floats
                                        [float(p) for p in parts[:6]]
                                        # This is cell parameters, next line should be atom count
                                        if i + 1 < len(lines):
                                            f.write("1\n")  # Write 1 atom
                                            # Find where atoms section ends
                                            atom_count = int(lines[i + 1].strip())
                                            # Write one atom at origin
                                            if i + 2 < len(lines):
                                                # Parse first atom line to get atom number
                                                atom_parts = lines[i + 2].strip().split()
                                                if atom_parts:
                                                    atom_num = atom_parts[0]
                                                    f.write(f"{atom_num} 0.0 0.0 0.0\n")
                                            
                                            # Skip original atoms section
                                            i += 2 + atom_count
                                            
                                            # Copy rest of file
                                            while i < len(lines):
                                                f.write(lines[i])
                                                i += 1
                                            break
                                except (ValueError, IndexError):
                                    pass
                                    
                                i += 1
                                
                        real_d12_found = True
                        break
                        
                    except Exception as e:
                        print(f"      Warning: Could not process D12 file: {e}")
                        real_d12_found = False
        
        if not real_d12_found:
            print("      No real D12 found, using minimal template")
            # Fall back to minimal template
            with open(sample_d12, 'w') as f:
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
        
        # Write minimal output file
        with open(sample_out, 'w') as f:
            f.write("CRYSTAL23 OUTPUT\n")
            f.write("FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n")
            f.write("LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n")
            f.write("A              B              C           ALPHA      BETA       GAMMA\n")
            f.write("5.0000         5.0000         5.0000      90.000     90.000     90.000\n")
            
        # Create a JSON config file to save the results
        config_file = temp_dir / f"{calc_type.lower()}_expert_config.json"
        
        print(f"\n      Launching CRYSTALOptToD12.py for {calc_type} configuration...")
        print(f"      Note: This is for configuration only - actual files will be processed during execution")
        print("")
        
        # Build command
        cmd = [
            sys.executable, str(script_path),
            "--out-file", str(sample_out),
            "--d12-file", str(sample_d12),
            "--output-dir", str(temp_dir),
            "--save-options",
            "--options-file", str(config_file)
        ]
        
        try:
            # Run interactively (no capture_output so user can interact)
            result = subprocess.run(cmd, cwd=str(self.work_dir))
            
            if result.returncode == 0 and config_file.exists():
                # Load the saved configuration
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    
                # Convert to our workflow config format
                expert_config = {
                    "expert_mode": True,
                    "interactive_setup": False,  # Already done
                    "crystal_opt_config": saved_config,
                    "source": "CRYSTALOptToD12.py",
                    "calculation_type": calc_type,
                    "inherit_geometry": True,
                    "config_file": str(config_file),
                    "options_file": str(config_file)  # Store path to saved options
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
        return {
            "modify_functional": False,
            "modify_basis": False,
            "modify_grid": False
        }
        
    def configure_slurm_scripts(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Configure and copy SLURM submission scripts for this calculation step"""
        print(f"    Configuring SLURM scripts for {calc_type} step {step_num}")
        
        # Determine which scripts are needed
        scripts_needed = self.get_required_scripts(calc_type)
        
        slurm_config = {
            "scripts": {},
            "resources": {},
            "modules": {}
        }
        
        for script_name in scripts_needed:
            print(f"      Setting up {script_name}...")
            script_config = self.setup_slurm_script(script_name, calc_type, step_num)
            slurm_config["scripts"][script_name] = script_config
            
        return slurm_config
        
    def get_required_scripts(self, calc_type: str) -> List[str]:
        """Determine which SLURM scripts are needed for a calculation type"""
        if calc_type in ["OPT", "OPT2", "SP", "FREQ"]:
            return ["submitcrystal23.sh"]
        elif calc_type in ["BAND", "DOSS", "TRANSPORT"]:
            return ["submit_prop.sh"]
        else:
            return ["submitcrystal23.sh"]  # Default
            
    def setup_slurm_script(self, script_name: str, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Setup and customize a SLURM script for specific calculation"""
        
        # Get default resource settings based on script and calc type
        default_resources = self.get_default_resources(script_name, calc_type)
        
        print(f"        Default resources for {calc_type}:")
        print(f"          Cores: {default_resources['ntasks']}")
        memory_str = default_resources.get('memory_per_cpu', default_resources.get('memory', 'N/A'))
        print(f"          Memory: {memory_str}")
        print(f"          Walltime: {default_resources['walltime']}")
        print(f"          Account: {default_resources.get('account', 'mendoza_q')}")
        
        # Ask user if they want to customize
        customize = yes_no_prompt(f"        Customize resources for {calc_type} step {step_num}?", "no")
        
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
            "copy_to_workdir": True
        }
        
        # Ask about additional customizations
        additional_custom = yes_no_prompt(f"        Add custom SLURM directives for {calc_type}?", "no")
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
                "scratch_dir": "$SCRATCH/crys23"
            }
        elif script_name == "submit_prop.sh":
            # From the script analysis: 28 cores, 2 hours, 80G total
            base_resources = {
                "ntasks": 28,
                "nodes": 1,
                "walltime": "1-00:00:00",
                "memory": "80G",
                "account": "mendoza_q",
                "module": "CRYSTAL/23-intel-2023a",
                "scratch_dir": "$SCRATCH/crys23/prop"
            }
        else:
            # Default fallback
            base_resources = {
                "ntasks": 16,
                "nodes": 1,
                "walltime": "4:00:00",
                "memory_per_cpu": "4G",
                "account": "general"
            }
            
        # Apply calculation-specific scaling from workflows.yaml
        scaled_resources = self.apply_calc_type_scaling(base_resources, calc_type)
        
        return scaled_resources
        
    def apply_calc_type_scaling(self, resources: Dict[str, Any], calc_type: str) -> Dict[str, Any]:
        """Apply calculation-type specific resource scaling"""
        
        # Resource scaling based on workflows.yaml analysis
        scaling_rules = {
            "OPT": {"walltime_factor": 1.0, "memory_factor": 1.0},  # Standard - 7 days
            "OPT2": {"walltime_factor": 1.0, "memory_factor": 1.0}, # Second opt - 7 days
            "SP": {"walltime_factor": 0.43, "memory_factor": 0.8}, # 3 days (3/7 ≈ 0.43)
            "FREQ": {"walltime_factor": 1.0, "memory_factor": 1.5}, # 7 days (same as OPT)
            "BAND": {"walltime_factor": 1.0, "memory_factor": 0.6}, # 1 day (base is already 1 day)
            "DOSS": {"walltime_factor": 1.0, "memory_factor": 0.6}  # 1 day (base is already 1 day)
        }
        
        if calc_type in scaling_rules:
            scaling = scaling_rules[calc_type]
            
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
        
    def get_safe_choice_input(self, prompt: str, valid_choices: list, default: str = None) -> str:
        """Get user choice with validation against a list of valid options"""
        while True:
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip() or default
            else:
                user_input = input(f"{prompt}: ").strip()
                
            if user_input in valid_choices:
                return user_input
            else:
                print(f"⚠️  Invalid choice '{user_input}'. Valid options: {', '.join(valid_choices)}")
                continue
    
    def get_safe_integer_input(self, prompt: str, default: int, min_val: int = 1, max_val: int = None) -> int:
        """Get integer input with validation and re-prompting on error"""
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                return default
            
            try:
                value = int(user_input)
                if value < min_val:
                    print(f"          ⚠️  Value must be at least {min_val}. Please try again.")
                    continue
                if max_val and value > max_val:
                    print(f"          ⚠️  Value must be at most {max_val}. Please try again.")
                    continue
                return value
            except ValueError:
                print(f"          ⚠️  Invalid input '{user_input}'. Please enter a number.")
                continue
    
    def get_safe_memory_input(self, prompt: str, default: str) -> str:
        """Get memory input with validation"""
        import re
        
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                return default
            
            # Valid memory formats: number + optional unit (G, GB, M, MB)
            pattern = r'^\d+([GMK]B?)?$'
            
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
                r'^\d{1,2}:\d{2}:\d{2}$',  # HH:MM:SS
                r'^\d-\d{1,2}:\d{2}:\d{2}$',  # D-HH:MM:SS
                r'^\d{1,2}-\d{1,2}:\d{2}:\d{2}$',  # DD-HH:MM:SS
            ]
            
            if any(re.match(pattern, user_input) for pattern in patterns):
                return user_input
            else:
                print(f"          ⚠️  Invalid walltime format '{user_input}'.")
                print(f"          Valid formats: HH:MM:SS, D-HH:MM:SS, or DD-HH:MM:SS")
                print(f"          Examples: 24:00:00, 3-00:00:00, 7-00:00:00")
                continue
    
    def get_custom_resources(self, default_resources: Dict[str, Any], calc_type: str) -> Dict[str, Any]:
        """Get custom resource settings from user"""
        resources = default_resources.copy()
        
        print(f"        Customize resources for {calc_type}:")
        
        # Cores
        resources['ntasks'] = self.get_safe_integer_input(
            f"          Cores [{resources['ntasks']}]: ",
            default=resources['ntasks'],
            min_val=1,
            max_val=128  # Adjust based on your cluster
        )
            
        # Memory
        if 'memory_per_cpu' in resources:
            resources['memory_per_cpu'] = self.get_safe_memory_input(
                f"          Memory per CPU [{resources['memory_per_cpu']}]: ",
                default=resources['memory_per_cpu']
            )
        elif 'memory' in resources:
            resources['memory'] = self.get_safe_memory_input(
                f"          Total memory [{resources['memory']}]: ",
                default=resources['memory']
            )
                
        # Walltime
        resources['walltime'] = self.get_safe_walltime_input(
            f"          Walltime [{resources['walltime']}]: ",
            default=resources['walltime']
        )
            
        # Account
        new_account = input(f"          Account [{resources.get('account', 'mendoza_q')}]: ").strip()
        if new_account:
            resources['account'] = new_account
            
        return resources
        
    def get_additional_customizations(self, calc_type: str) -> List[Dict[str, str]]:
        """Get additional SLURM customizations from user"""
        customizations = []
        
        print(f"        Additional SLURM directives for {calc_type}:")
        print("        Enter SLURM directives (without #SBATCH). Press Enter when done.")
        
        common_options = {
            "1": "--constraint=intel18",
            "2": "--exclude=node1,node2", 
            "3": "--partition=gpu",
            "4": "--gres=gpu:1",
            "5": "--array=1-10",
            "6": "Custom directive"
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
                        customizations.append({"directive": custom, "description": "Custom"})
                else:
                    directive = common_options[choice]
                    customizations.append({"directive": directive, "description": f"Common: {directive}"})
            else:
                # Treat as custom directive
                customizations.append({"directive": choice, "description": "Custom"})
                
        return customizations
        
    def copy_and_customize_scripts(self, step_configs: Dict[str, Dict[str, Any]], workflow_id: str):
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
                    self.create_customized_script(
                        bin_scripts_dir, scripts_dir, script_config, step_key
                    )
                    
        # Copy additional required files
        self.copy_additional_files(bin_scripts_dir, workflow_id)
                    
    def create_customized_script(self, bin_dir: Path, scripts_dir: Path, 
                                script_config: Dict[str, Any], step_key: str):
        """Create a customized version of a SLURM script"""
        
        source_script = bin_dir / script_config["source_script"]
        target_script = scripts_dir / script_config["step_specific_name"]
        
        print(f"    Creating {target_script.name} for {step_key}")
        
        if not source_script.exists():
            print(f"      Warning: Source script {source_script} not found")
            return
            
        # Read source script
        with open(source_script, 'r') as f:
            content = f.read()
            
        # Apply customizations
        modified_content = self.apply_script_customizations(content, script_config)
        
        # Write customized script
        with open(target_script, 'w') as f:
            f.write(modified_content)
            
        # Make executable
        target_script.chmod(0o755)
        
        print(f"      Created: {target_script}")
        print(f"      Resources: {script_config['resources']['ntasks']} cores, "
              f"{script_config['resources']['walltime']} walltime")
              
    def apply_script_customizations(self, content: str, script_config: Dict[str, Any]) -> str:
        """Apply resource and directive customizations to script content"""
        
        resources = script_config["resources"]
        customizations = script_config.get("customizations", [])
        
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # Modify resource directives
            if line.startswith("echo '#SBATCH --ntasks="):
                modified_lines.append(f"echo '#SBATCH --ntasks={resources['ntasks']}' >> $1.sh")
            elif line.startswith("echo '#SBATCH -t "):
                modified_lines.append(f"echo '#SBATCH -t {resources['walltime']}' >> $1.sh")
            elif line.startswith("echo '#SBATCH --mem-per-cpu=") and "memory_per_cpu" in resources:
                modified_lines.append(f"echo '#SBATCH --mem-per-cpu={resources['memory_per_cpu']}' >> $1.sh")
            elif line.startswith("echo '#SBATCH --mem=") and "memory" in resources:
                modified_lines.append(f"echo '#SBATCH --mem={resources['memory']}' >> $1.sh")
            elif line.startswith("echo '#SBATCH -A ") and "account" in resources:
                modified_lines.append(f"echo '#SBATCH -A {resources['account']}' >> $1.sh")
            elif line.startswith("echo '#SBATCH --constraint=") and "constraint" in resources:
                modified_lines.append(f"echo '#SBATCH --constraint={resources['constraint']}' >> $1.sh")
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
                    
        return '\n'.join(modified_lines)
        
    def copy_additional_files(self, bin_dir: Path, workflow_id: str):
        """Copy additional required files using the copy_dependencies system"""
        print("    Copying additional workflow files...")
        
        try:
            # Import and use the copy_dependencies function
            sys.path.append(str(bin_dir))
            from copy_dependencies import copy_dependencies
            
            # Use the comprehensive copy_dependencies function
            copied_count, missing_count = copy_dependencies(str(self.work_dir))
            
            if missing_count > 0:
                print(f"    Warning: {missing_count} dependencies could not be copied")
            else:
                print(f"    ✓ Successfully copied all {copied_count} dependencies")
                
        except Exception as e:
            print(f"    Error using copy_dependencies: {e}")
            print("    Falling back to essential files only...")
            
            # Fallback to basic essential files only
            essential_files = [
                "enhanced_queue_manager.py",
                "material_database.py",
                "error_recovery.py",
                "workflows.yaml",
                "recovery_config.yaml"
            ]
            
            copied_count = 0
            for filename in essential_files:
                source_file = bin_dir / filename
                dest_file = self.work_dir / filename
                
                if source_file.exists():
                    if not dest_file.exists() or self.should_update_file(source_file, dest_file):
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                        print(f"      Copied: {filename}")
                        
            print(f"    Copied {copied_count} essential files (fallback mode)")
                
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_file = self.configs_dir / f"workflow_plan_{timestamp}.json"
        
        with open(plan_file, 'w') as f:
            json.dump(workflow_plan, f, indent=2, default=str)
            
        print(f"\n✅ Workflow plan saved to: {plan_file}")
        print(f"\n📁 Configuration locations:")
        print(f"   Main plan: {plan_file}")
        print(f"   CIF config: {self.configs_dir}/cif_conversion_config.json")
        print(f"   Step configs: {self.temp_dir}/workflow_*_step_*.json")
        print(f"   All configs: {self.configs_dir}/")
        return plan_file
        
    def execute_workflow_plan(self, plan_file: Path):
        """Execute the planned workflow"""
        print(f"\nStep 5: Execute Workflow")
        print("-" * 40)
        
        with open(plan_file, 'r') as f:
            plan = json.load(f)
            
        print("Workflow execution summary:")
        print(f"  Input files: {len(plan['input_files']['cif'])} CIFs, {len(plan['input_files']['d12'])} D12s")
        print(f"  Workflow sequence: {' → '.join(plan['workflow_sequence'])}")
        print(f"  Total materials: {len(plan['input_files']['cif']) + len(plan['input_files']['d12'])}")
        
        proceed = yes_no_prompt("Proceed with workflow execution?", "yes")
        if not proceed:
            print("Workflow execution cancelled.")
            return
            
        # Initialize queue manager
        queue_manager = EnhancedCrystalQueueManager(
            d12_dir=str(self.outputs_dir),
            max_jobs=200,
            enable_tracking=True,
            db_path=self.db_path
        )
        
        # Execute workflow
        self.run_workflow_execution(plan, queue_manager)
        
    def run_workflow_execution(self, plan: Dict[str, Any], queue_manager):
        """Run the actual workflow execution"""
        print("\nStarting workflow execution...")
        
        # Phase 1: Convert CIFs to D12s if needed
        if plan['input_files']['cif']:
            print("Phase 1: Converting CIF files to D12 format...")
            self.convert_cifs_to_d12s(plan)
            
        # Phase 2: Execute planned calculation sequence using WorkflowExecutor
        print("Phase 2: Executing calculation sequence...")
        try:
            from workflow_executor import WorkflowExecutor
            executor = WorkflowExecutor(str(self.work_dir), self.db_path)
            
            # Use the workflow ID from the plan - CRITICAL for workflow progression!
            workflow_id = plan.get('workflow_id')
            if not workflow_id:
                # Only generate a new one if missing (shouldn't happen)
                workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"WARNING: No workflow_id in plan, generated new one: {workflow_id}")
            
            # Prepare the workflow directory structure for the executor
            workflow_dir = executor.outputs_dir / workflow_id
            workflow_dir.mkdir(exist_ok=True)
            
            # Set up step_001_OPT directory with D12 files
            step_001_dir = workflow_dir / "step_001_OPT"
            step_001_dir.mkdir(exist_ok=True)
            
            # Copy D12 files from input directory to step directory with clean naming
            input_dir = Path(plan['input_directory'])
            d12_files = list(input_dir.glob("*.d12"))
            
            print(f"  Copying {len(d12_files)} D12 files to workflow step directory with clean names...")
            for d12_file in d12_files:
                # Create clean material ID from dirty filename
                clean_material_id = self.create_clean_material_id(d12_file)
                
                # Only add _opt if it doesn't already end with _opt (to avoid _opt_opt)
                if clean_material_id.endswith('_opt'):
                    clean_filename = f"{clean_material_id}.d12"
                else:
                    clean_filename = f"{clean_material_id}_opt.d12"
                    
                dest_file = step_001_dir / clean_filename
                
                if not dest_file.exists():
                    shutil.copy2(d12_file, dest_file)
                    print(f"    Copied: {d12_file.name} → {clean_filename}")
                else:
                    print(f"    Exists: {clean_filename}")
                        
            # Execute the workflow using the proper executor
            executor.execute_workflow_steps(plan, workflow_id)
            
        except Exception as e:
            print(f"Error executing workflow: {e}")
            print("Falling back to basic execution...")
            self.execute_calculation_sequence(plan, queue_manager)
        
    def convert_cifs_to_d12s(self, plan: Dict[str, Any]):
        """Convert CIF files to D12 format using saved configuration"""
        # Check if D12 files already exist (skip if already done)
        input_dir = Path(plan['input_directory'])
        existing_d12s = list(input_dir.glob("*.d12"))
        cif_files = list(input_dir.glob("*.cif"))
        
        if len(existing_d12s) >= len(cif_files):
            print(f"Found {len(existing_d12s)} D12 files already exist for {len(cif_files)} CIF files.")
            print("Skipping CIF conversion - using existing D12 files.")
            
            # D12 files already exist in input directory, no need to copy to separate input directory
            # They will be used directly by the workflow executor
            print(f"  Using existing D12 files from: {input_dir}")
                    
            return
            
        # If we need to convert, use the WorkflowExecutor
        print("Converting CIF files to D12 format...")
        try:
            from workflow_executor import WorkflowExecutor
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
        input_dir = Path(plan['input_directory'])
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
            print(f"  python enhanced_queue_manager.py --status")
            print(f"  Database: {self.db_path}")
            
        except Exception as e:
            print(f"Error during job submission: {e}")
            print("You can manually submit jobs using:")
            print(f"  cd {input_dir}")
            print("  python enhanced_queue_manager.py --max-jobs 200")
        
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
            
        # Step 2: CIF conversion setup if needed
        cif_config = None
        if input_files["cif"]:
            cif_config = self.setup_cif_conversion(input_files["cif"])
            
        # Step 3: Plan workflow sequence
        workflow_sequence = self.plan_workflow_sequence()
        
        # Step 4: Configure workflow steps
        step_configs = self.configure_workflow_steps(workflow_sequence, bool(input_files["cif"]))
        
        # Step 5: Create comprehensive workflow plan
        workflow_plan = {
            "created": datetime.now().isoformat(),
            "input_type": input_type,
            "input_directory": str(input_dir),
            "input_files": {k: [str(f) for f in v] for k, v in input_files.items()},
            "workflow_sequence": workflow_sequence,
            "step_configurations": step_configs,
            "cif_conversion_config": cif_config,
            "execution_settings": {
                "max_concurrent_jobs": 200,
                "enable_material_tracking": True,
                "auto_progression": True
            }
        }
        
        # Step 5: Copy and customize SLURM scripts
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.copy_and_customize_scripts(step_configs, workflow_id)
        
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
        parts = name.split('_')
        
        # Find the first part that looks like a technical suffix
        core_parts = []
        for i, part in enumerate(parts):
            # Special case: if this is "opt" and we only have one core part so far,
            # and the core part ends with a number, this might be a calc type rather than material identifier
            if (part.upper() == 'OPT' and len(core_parts) == 1 and 
                core_parts[0] and core_parts[0][-1].isdigit()):
                # This looks like "test1_opt" - the opt is a calc type, not part of material name
                break
            # Check if this part is a technical suffix (removed OPT from this list since we handle it specially)
            elif part.upper() in ['SP', 'FREQ', 'BAND', 'DOSS', 'BULK', 'OPTGEOM', 
                                'CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE', 'SYMM', 'TZ', 'DZ', 'SZ']:
                break
            # Check if this part is a DFT functional
            elif part.upper() in ['PBE', 'B3LYP', 'HSE06', 'PBE0', 'SCAN', 'BLYP', 'BP86']:
                break
            # Check if this part contains basis set info  
            elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():
                break
            # Check if this part is a dispersion correction
            elif 'D3' in part.upper():
                break
            else:
                core_parts.append(part)
        
        # If we found core parts, use them
        if core_parts:
            clean_name = '_'.join(core_parts)
        else:
            # Fallback: just use the first part
            clean_name = parts[0] if parts else name
            
        # Handle special characters that might need preservation
        # Don't remove things like numbers, hyphens in material names
        
        return clean_name


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CRYSTAL Workflow Planner")
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