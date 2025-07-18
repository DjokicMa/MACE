#!/usr/bin/env python3
"""
Run CRYSTAL Workflow
===================
Simple interface to launch the comprehensive CRYSTAL workflow planning and execution system.

Usage Examples:
  # Interactive workflow planning
  python run_workflow.py --interactive

  # Execute a saved workflow plan
  python run_workflow.py --execute workflow_plan_20241216_143022.json

  # Quick start with CIFs using defaults
  python run_workflow.py --quick-start --cif-dir /path/to/cifs --workflow full_electronic

  # Status monitoring
  python run_workflow.py --status

Author: Workflow integration script
"""

import os
import sys
import argparse
from pathlib import Path

# Add our modules to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "Crystal_To_CIF"))

# Auto-copy dependencies if needed
def ensure_dependencies():
    """Ensure all required dependencies are available locally"""
    required_files = [
        "enhanced_queue_manager.py",
        "material_database.py", 
        "workflow_engine.py",
        "error_recovery.py",
        "error_detector.py",
        "crystal_property_extractor.py",
        "formula_extractor.py",
        "input_settings_extractor.py",
        "query_input_settings.py"
    ]
    
    missing_files = []
    for filename in required_files:
        if not (current_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"Missing required dependencies: {missing_files}")
        print("Auto-copying dependencies...")
        try:
            # Import and run copy_dependencies
            sys.path.append(str(current_dir))
            from copy_dependencies import copy_dependencies
            copied, missing = copy_dependencies(str(current_dir))
            if missing > 0:
                print(f"Warning: {missing} dependencies could not be copied")
            else:
                print("✓ All dependencies copied successfully")
        except Exception as e:
            print(f"Error copying dependencies: {e}")
            print("Please manually run: python copy_dependencies.py")
            sys.exit(1)

# Ensure dependencies before importing
ensure_dependencies()

try:
    from workflow_planner import WorkflowPlanner
    from workflow_executor import WorkflowExecutor
    from enhanced_queue_manager import EnhancedCrystalQueueManager
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing workflow modules: {e}")
    print("Please ensure all required modules are available")
    print("Try running: python copy_dependencies.py")
    sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CRYSTAL Workflow Planning and Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive workflow planning (recommended for first use)
  python run_workflow.py --interactive

  # Quick start with CIFs and predefined workflow
  python run_workflow.py --quick-start --cif-dir ./my_cifs --workflow full_electronic

  # Execute a previously saved workflow plan
  python run_workflow.py --execute workflow_plan_20241216_143022.json

  # Check status of running workflows
  python run_workflow.py --status

  # Show available workflow templates
  python run_workflow.py --show-templates

Workflow Templates:
  basic_opt         : OPT
  opt_sp           : OPT → SP
  full_electronic  : OPT → SP → BAND → DOSS
  double_opt       : OPT → OPT2 → SP
  complete         : OPT → SP → BAND → DOSS → FREQ
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--interactive", action="store_true",
                           help="Run interactive workflow planner")
    mode_group.add_argument("--execute", metavar="PLAN_FILE",
                           help="Execute a saved workflow plan")
    mode_group.add_argument("--quick-start", action="store_true",
                           help="Quick start with minimal configuration")
    mode_group.add_argument("--status", action="store_true",
                           help="Show status of running workflows")
    mode_group.add_argument("--show-templates", action="store_true",
                           help="Show available workflow templates")
    
    # Quick start options
    parser.add_argument("--cif-dir", help="Directory containing CIF files (for quick start)")
    parser.add_argument("--d12-dir", help="Directory containing D12 files (for quick start)")
    parser.add_argument("--workflow", default="full_electronic",
                       choices=["basic_opt", "opt_sp", "full_electronic", "double_opt", "complete"],
                       help="Workflow template for quick start")
    
    # Common options
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    parser.add_argument("--max-jobs", type=int, default=200, help="Maximum concurrent jobs")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode(args)
    elif args.execute:
        run_execute_mode(args)
    elif args.quick_start:
        run_quick_start_mode(args)
    elif args.status:
        run_status_mode(args)
    elif args.show_templates:
        show_workflow_templates()
    

def run_interactive_mode(args):
    """Run interactive workflow planning"""
    print("Starting Interactive Workflow Planner")
    print("=" * 50)
    
    planner = WorkflowPlanner(args.work_dir, args.db_path)
    planner.main_interactive_workflow()


def run_execute_mode(args):
    """Execute a saved workflow plan"""
    plan_file = Path(args.execute)
    
    if not plan_file.exists():
        print(f"Error: Workflow plan file not found: {plan_file}")
        sys.exit(1)
        
    print(f"Executing Workflow Plan: {plan_file}")
    print("=" * 50)
    
    executor = WorkflowExecutor(args.work_dir, args.db_path)
    executor.execute_workflow_plan(plan_file)


def run_quick_start_mode(args):
    """Run quick start mode with minimal configuration"""
    print("Quick Start Workflow Setup")
    print("=" * 50)
    
    # Validate input directories
    input_dir = None
    input_type = None
    
    if args.cif_dir:
        input_dir = Path(args.cif_dir)
        input_type = "cif"
    elif args.d12_dir:
        input_dir = Path(args.d12_dir)
        input_type = "d12"
    else:
        print("Error: Must specify either --cif-dir or --d12-dir for quick start")
        sys.exit(1)
        
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
        
    # Find input files
    if input_type == "cif":
        input_files = list(input_dir.glob("*.cif"))
        file_type_name = "CIF"
    else:
        input_files = list(input_dir.glob("*.d12"))
        file_type_name = "D12"
        
    if not input_files:
        print(f"Error: No {file_type_name} files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(input_files)} {file_type_name} files in {input_dir}")
    print(f"Selected workflow: {args.workflow}")
    
    # Create quick workflow plan
    workflow_templates = {
        "basic_opt": ["OPT"],
        "opt_sp": ["OPT", "SP"],
        "full_electronic": ["OPT", "SP", "BAND", "DOSS"],
        "double_opt": ["OPT", "OPT2", "SP"],
        "complete": ["OPT", "SP", "BAND", "DOSS", "FREQ"]
    }
    
    sequence = workflow_templates[args.workflow]
    
    # Generate quick plan
    quick_plan = create_quick_workflow_plan(
        input_dir, input_files, input_type, sequence, args
    )
    
    # Save and execute
    planner = WorkflowPlanner(args.work_dir, args.db_path)
    plan_file = planner.save_workflow_plan(quick_plan)
    
    print(f"\nQuick workflow plan created: {plan_file}")
    print("Starting execution...")
    
    executor = WorkflowExecutor(args.work_dir, args.db_path)
    executor.execute_workflow_plan(plan_file)


def create_quick_workflow_plan(input_dir, input_files, input_type, sequence, args):
    """Create a quick workflow plan with default settings"""
    from datetime import datetime
    
    # Default CIF conversion config (adapt to first calculation type)
    first_calc_type = sequence[0] if sequence else "OPT"
    default_cif_config = {
        "symmetry_handling": "CIF",
        "write_only_unique": True,
        "dimensionality": "CRYSTAL",
        "calculation_type": "SP" if first_calc_type == "SP" else "OPT",
        "optimization_type": "FULLOPTG" if first_calc_type != "SP" else None,
        "optimization_settings": {
            "TOLDEG": 0.00003,
            "TOLDEX": 0.00012,
            "TOLDEE": 7,
            "MAXCYCLE": 800
        },
        "method": "DFT",
        "dft_functional": "B3LYP",
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
    
    # Default step configurations
    step_configs = {}
    for i, calc_type in enumerate(sequence):
        step_num = i + 1
        
        if (calc_type == "OPT" or calc_type == "SP") and i == 0 and input_type == "cif":
            step_configs[f"{calc_type}_{step_num}"] = {"source": "cif_conversion"}
        elif calc_type in ["OPT", "OPT2"]:
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "OPT",
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True
            }
        elif calc_type == "SP":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "SP",
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True
            }
        elif calc_type == "BAND":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "BAND",
                "source": "create_band_d3.py",
                "requires_wavefunction": True
            }
        elif calc_type == "DOSS":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "DOSS", 
                "source": "alldos.py",
                "requires_wavefunction": True
            }
        elif calc_type == "FREQ":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {
                    "mode": "GAMMA",
                    "numderiv": 2,
                    "intensities": False,
                    "temperatures": [298.15],
                    "custom_tolerances": {
                        "TOLINTEG": "9 9 9 11 38",
                        "TOLDEE": 11
                    }
                }
            }
    
    # Create plan
    plan = {
        "created": datetime.now().isoformat(),
        "mode": "quick_start",
        "input_type": input_type,
        "input_directory": str(input_dir),
        "input_files": {
            input_type: [str(f) for f in input_files],
            "cif" if input_type == "d12" else "d12": []
        },
        "workflow_sequence": sequence,
        "step_configurations": step_configs,
        "cif_conversion_config": default_cif_config if input_type == "cif" else None,
        "execution_settings": {
            "max_concurrent_jobs": args.max_jobs,
            "enable_material_tracking": True,
            "auto_progression": True
        }
    }
    
    return plan


def run_status_mode(args):
    """Show status of running workflows and jobs"""
    print("CRYSTAL Workflow Status")
    print("=" * 50)
    
    # Check database for active workflows
    db = MaterialDatabase(args.db_path)
    
    # Get recent calculations
    print("Recent calculations:")
    recent_calcs = db.get_recent_calculations(limit=20)
    
    if recent_calcs:
        print(f"{'ID':<10} {'Material':<20} {'Type':<8} {'Status':<12} {'Started':<20}")
        print("-" * 80)
        for calc in recent_calcs:
            calc_id = calc.get('calc_id', 'N/A')[:8]
            material_id = calc.get('material_id', 'N/A')[:18]
            calc_type = calc.get('calc_type', 'N/A')
            status = calc.get('status', 'N/A')
            started = calc.get('start_time', 'N/A')[:19] if calc.get('start_time') else 'N/A'
            
            print(f"{calc_id:<10} {material_id:<20} {calc_type:<8} {status:<12} {started:<20}")
    else:
        print("No recent calculations found")
    
    # Check queue manager status
    print(f"\nFor detailed queue status, run:")
    print(f"  python enhanced_queue_manager.py --status")


def show_workflow_templates():
    """Show available workflow templates"""
    print("Available Workflow Templates")
    print("=" * 50)
    
    templates = {
        "basic_opt": {
            "sequence": ["OPT"],
            "description": "Basic geometry optimization only"
        },
        "opt_sp": {
            "sequence": ["OPT", "SP"],
            "description": "Geometry optimization followed by single point calculation"
        },
        "full_electronic": {
            "sequence": ["OPT", "SP", "BAND", "DOSS"],
            "description": "Complete electronic structure characterization"
        },
        "double_opt": {
            "sequence": ["OPT", "OPT2", "SP"],
            "description": "Two-stage optimization for difficult convergence cases"
        },
        "complete": {
            "sequence": ["OPT", "SP", "BAND", "DOSS", "FREQ"],
            "description": "Complete characterization including vibrational analysis"
        }
    }
    
    for name, info in templates.items():
        sequence_str = " → ".join(info["sequence"])
        print(f"{name:<20} : {sequence_str}")
        print(f"{'':>20}   {info['description']}")
        print()
    
    print("Calculation Types:")
    calc_types = {
        "OPT": "Geometry optimization",
        "OPT2": "Second/refined geometry optimization", 
        "SP": "Single point electronic structure calculation",
        "BAND": "Band structure calculation",
        "DOSS": "Density of states calculation",
        "FREQ": "Vibrational frequency calculation"
    }
    
    for calc_type, description in calc_types.items():
        print(f"  {calc_type:<6} : {description}")


if __name__ == "__main__":
    main()