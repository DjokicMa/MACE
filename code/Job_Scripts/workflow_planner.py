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
    from material_database import MaterialDatabase
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
        self.inputs_dir = self.work_dir / "workflow_inputs" 
        self.outputs_dir = self.work_dir / "workflow_outputs"
        self.temp_dir = self.work_dir / "temp"
        
        for dir_path in [self.configs_dir, self.inputs_dir, self.outputs_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Available calculation types and their dependencies
        self.calc_types = {
            "OPT": {"name": "Geometry Optimization", "depends_on": [], "generates": ["optimized_geometry"]},
            "OPT2": {"name": "Second Optimization", "depends_on": ["OPT"], "generates": ["refined_geometry"]},
            "SP": {"name": "Single Point", "depends_on": ["OPT"], "generates": ["electronic_structure", "wavefunction"]},
            "BAND": {"name": "Band Structure", "depends_on": ["SP"], "generates": ["band_structure"]},
            "DOSS": {"name": "Density of States", "depends_on": ["SP"], "generates": ["density_of_states"]},
            "FREQ": {"name": "Frequencies", "depends_on": ["OPT"], "generates": ["vibrational_modes"]},
            "TRANSPORT": {"name": "Transport Properties", "depends_on": ["SP", "BAND"], "generates": ["transport"]},
        }
        
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
            
            level_choice = input("Customization level [1]: ").strip() or "1"
            
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
            
        print("\nBuild your workflow sequence:")
        print("Enter calculation types in order (e.g., OPT SP BAND DOSS)")
        print("Type 'help' for more information")
        
        sequence = []
        while True:
            current = " → ".join(sequence) if sequence else "Empty"
            user_input = input(f"\nCurrent sequence: {current}\nNext calculation (or 'done'): ").strip().upper()
            
            if user_input == "DONE":
                break
            elif user_input == "HELP":
                self.show_workflow_help()
            elif user_input in self.calc_types:
                if self.validate_calc_addition(sequence, user_input):
                    sequence.append(user_input)
                    print(f"Added {user_input}. Current: {' → '.join(sequence)}")
                else:
                    deps = ", ".join(self.calc_types[user_input]["depends_on"])
                    print(f"Cannot add {user_input}. Missing dependencies: {deps}")
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
            print("4: Done")
            
            choice = input("Select option: ").strip()
            
            if choice == "1":
                calc = input("Add calculation type: ").strip().upper()
                if calc in self.calc_types and self.validate_calc_addition(sequence, calc):
                    sequence.append(calc)
                    print(f"Added {calc}. Current: {' → '.join(sequence)}")
                    
            elif choice == "2":
                if sequence:
                    calc = input("Remove calculation type: ").strip().upper()
                    if calc in sequence:
                        sequence.remove(calc)
                        print(f"Removed {calc}. Current: {' → '.join(sequence)}")
                        
            elif choice == "3":
                calc = input("Insert calculation type: ").strip().upper()
                pos = input("Insert at position (1-based): ").strip()
                try:
                    pos = int(pos) - 1
                    if 0 <= pos <= len(sequence) and calc in self.calc_types:
                        sequence.insert(pos, calc)
                        print(f"Inserted {calc}. Current: {' → '.join(sequence)}")
                except ValueError:
                    print("Invalid position")
                    
            elif choice == "4":
                break
                
        return sequence
        
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
        print("\nDependencies:")
        for calc, info in self.calc_types.items():
            if info["depends_on"]:
                print(f"  {calc} requires: {', '.join(info['depends_on'])}")
                
    def configure_workflow_steps(self, sequence: List[str], has_cifs: bool) -> Dict[str, Dict[str, Any]]:
        """Configure settings for each workflow step"""
        print(f"\nStep 4: Configure Workflow Steps")
        print("-" * 40)
        
        step_configs = {}
        
        for i, calc_type in enumerate(sequence):
            print(f"\nConfiguring {calc_type} ({self.calc_types[calc_type]['name']}):")
            
            if calc_type == "OPT" and i == 0 and has_cifs:
                print("  Using CIF conversion configuration for first OPT step")
                step_configs[f"{calc_type}_1"] = {"source": "cif_conversion"}
                
            elif calc_type in ["OPT", "OPT2"] and i > 0:
                # Subsequent optimization - configure via CRYSTALOptToD12.py
                config = self.configure_optimization_step(calc_type, i+1)
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type == "SP":
                # Single point calculation
                config = self.configure_single_point_step()
                step_configs[f"{calc_type}_{i+1}"] = config
                
            elif calc_type in ["BAND", "DOSS"]:
                # Analysis calculations
                config = self.configure_analysis_step(calc_type)
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
        """Configure single point calculation"""
        print("  Configuring single point calculation")
        
        use_defaults = yes_no_prompt("    Use settings from optimization?", "yes")
        
        config = {
            "calculation_type": "SP",
            "source": "CRYSTALOptToD12.py",
            "inherit_settings": use_defaults
        }
        
        if not use_defaults:
            # Allow modification of method/basis set
            modify_method = yes_no_prompt("    Modify DFT method/basis set?", "no")
            if modify_method:
                config["custom_settings"] = self.get_custom_sp_settings()
                
        return config
        
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
        """Configure frequency calculation"""
        print("  Configuring frequency calculation")
        
        config = {
            "calculation_type": "FREQ",
            "source": "CRYSTALOptToD12.py",
            "inherit_base_settings": True,
            "custom_tolerances": {
                "TOLINTEG": "12 12 12 12 24",
                "TOLDEE": 12
            }
        }
        
        return config
        
    def get_detailed_opt_config(self, calc_type: str, step_num: int) -> Dict[str, Any]:
        """Get detailed optimization configuration"""
        # This would integrate with CRYSTALOptToD12.py configuration
        # For now, return reasonable defaults
        return {
            "calculation_type": "OPT",
            "optimization_type": "FULLOPTG",
            "optimization_settings": {
                "TOLDEG": 0.00003,
                "TOLDEX": 0.00012,
                "TOLDEE": 7,
                "MAXCYCLE": 800
            },
            "source": "CRYSTALOptToD12.py"
        }
        
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
        if calc_type in ["OPT", "SP", "FREQ"]:
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
        print(f"          Account: {default_resources.get('account', 'general')}")
        
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
                "account": "general",
                "constraint": "intel18",
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
        
    def get_custom_resources(self, default_resources: Dict[str, Any], calc_type: str) -> Dict[str, Any]:
        """Get custom resource settings from user"""
        resources = default_resources.copy()
        
        print(f"        Customize resources for {calc_type}:")
        
        # Cores
        new_cores = input(f"          Cores [{resources['ntasks']}]: ").strip()
        if new_cores:
            resources['ntasks'] = int(new_cores)
            
        # Memory
        if 'memory_per_cpu' in resources:
            new_mem = input(f"          Memory per CPU [{resources['memory_per_cpu']}]: ").strip()
            if new_mem:
                resources['memory_per_cpu'] = new_mem
        elif 'memory' in resources:
            new_mem = input(f"          Total memory [{resources['memory']}]: ").strip()
            if new_mem:
                resources['memory'] = new_mem
                
        # Walltime
        new_walltime = input(f"          Walltime [{resources['walltime']}]: ").strip()
        if new_walltime:
            resources['walltime'] = new_walltime
            
        # Account
        new_account = input(f"          Account [{resources.get('account', 'general')}]: ").strip()
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
        """Copy additional required files from bin directory"""
        print("    Copying additional workflow files...")
        
        # Files to copy from bin directory
        additional_files = [
            "workflows.yaml",
            "recovery_config.yaml", 
            "enhanced_queue_manager.py",
            "crystal_queue_manager.py",  # Backup
            "material_database.py",
            "workflow_engine.py",
            "error_recovery.py",
            "material_monitor.py",
            "crystal_file_manager.py",
            "populate_completed_jobs.py"
        ]
        
        # Core dependency scripts from Crystal_To_CIF directory
        crystal_to_cif_files = [
            "NewCifToD12.py",
            "CRYSTALOptToD12.py",
            "d12creation.py"
        ]
        
        # Analysis scripts from Creation_Scripts directory
        creation_scripts_files = [
            "alldos.py",
            "create_band_d3.py"
        ]
        
        copied_files = []
        
        # Copy workflow management files from bin directory
        for filename in additional_files:
            source_file = bin_dir / filename
            dest_file = self.work_dir / filename
            
            if source_file.exists():
                if not dest_file.exists() or self.should_update_file(source_file, dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(filename)
                    print(f"      Copied: {filename}")
                else:
                    print(f"      Exists: {filename}")
            else:
                print(f"      Missing: {filename} (not found in bin)")
        
        # Copy Crystal_To_CIF dependency scripts
        crystal_to_cif_dir = bin_dir.parent / "Crystal_To_CIF"
        for filename in crystal_to_cif_files:
            source_file = crystal_to_cif_dir / filename
            dest_file = self.work_dir / filename
            
            if source_file.exists():
                if not dest_file.exists() or self.should_update_file(source_file, dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(filename)
                    print(f"      Copied: {filename}")
                else:
                    print(f"      Exists: {filename}")
            else:
                print(f"      Missing: {filename} (not found in Crystal_To_CIF)")
        
        # Copy Creation_Scripts dependency scripts
        creation_scripts_dir = bin_dir.parent / "Creation_Scripts"
        for filename in creation_scripts_files:
            source_file = creation_scripts_dir / filename
            dest_file = self.work_dir / filename
            
            if source_file.exists():
                if not dest_file.exists() or self.should_update_file(source_file, dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(filename)
                    print(f"      Copied: {filename}")
                else:
                    print(f"      Exists: {filename}")
            else:
                print(f"      Missing: {filename} (not found in Creation_Scripts)")
                
        if copied_files:
            print(f"    Copied {len(copied_files)} additional files")
        
        # Make Python scripts executable
        for filename in copied_files:
            if filename.endswith('.py'):
                (self.work_dir / filename).chmod(0o755)
                
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
            
        print(f"\nWorkflow plan saved to: {plan_file}")
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
            
            # Create a workflow ID for this execution
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare the workflow directory structure for the executor
            workflow_dir = executor.outputs_dir / workflow_id
            workflow_dir.mkdir(exist_ok=True)
            
            # Set up step_001_OPT directory with D12 files
            step_001_dir = workflow_dir / "step_001_OPT"
            step_001_dir.mkdir(exist_ok=True)
            
            # Copy D12 files to the workflow directory
            input_dir = self.inputs_dir / "step_001_OPT"
            if input_dir.exists():
                for d12_file in input_dir.glob("*.d12"):
                    dest_file = step_001_dir / d12_file.name
                    if not dest_file.exists():
                        shutil.copy2(d12_file, dest_file)
                        
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
            
            # Copy existing D12s to workflow directory if needed
            workflow_input_dir = self.inputs_dir / "step_001_OPT"
            workflow_input_dir.mkdir(exist_ok=True)
            
            for d12_file in existing_d12s:
                dest_file = workflow_input_dir / d12_file.name
                if not dest_file.exists():
                    shutil.copy2(d12_file, dest_file)
                    print(f"  Copied: {d12_file.name}")
                    
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
        
        # Get the input files for the first step
        input_dir = self.inputs_dir / "step_001_OPT"
        d12_files = list(input_dir.glob("*.d12"))
        
        if not d12_files:
            print("Error: No D12 files found for job submission!")
            return
            
        print(f"Found {len(d12_files)} D12 files to submit")
        
        # Submit the initial OPT calculations using the enhanced queue manager
        try:
            # Copy D12 files to working directory for submission
            for d12_file in d12_files:
                dest_file = Path.cwd() / d12_file.name
                if not dest_file.exists():
                    shutil.copy2(d12_file, dest_file)
                    print(f"  Prepared: {d12_file.name}")
            
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
    """Get yes/no response from user"""
    default_char = "Y/n" if default.lower() == "yes" else "y/N"
    response = input(f"{prompt} [{default_char}]: ").strip().lower()
    
    if not response:
        return default.lower() == "yes"
    return response in ["y", "yes", "1", "true"]


if __name__ == "__main__":
    main()