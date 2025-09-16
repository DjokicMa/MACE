#!/usr/bin/env python3
"""
MACE Workflow Manager
====================
Part of MACE (Mendoza Automated CRYSTAL Engine)

Simple interface to launch the comprehensive CRYSTAL workflow planning and execution system.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Usage Examples:
  # Interactive workflow planning
  mace.py --interactive

  # Execute a saved workflow plan
  mace.py --execute workflow_plan_20241216_143022.json

  # Quick start with CIFs using defaults
  mace.py --quick-start --cif-dir /path/to/cifs --workflow full_electronic

  # Status monitoring
  mace.py --status

Developed by: Marcus Djokic
Contributors: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey
Advisor: Prof. Jose Luis Mendoza-Cortes
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path for Crystal_d12/d3 access
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "Crystal_d12"))
sys.path.insert(0, str(parent_dir / "Crystal_d3"))

# Import MACE modules
try:
    from mace.workflow.planner import WorkflowPlanner
    from mace.workflow.executor import WorkflowExecutor
    from mace.queue.manager import EnhancedCrystalQueueManager
    from mace.database.materials import MaterialDatabase
except ImportError as e:
    print(f"Error importing MACE modules: {e}")
    print("Please ensure all MACE modules are properly installed")
    sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MACE Workflow Planning and Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive workflow planning (recommended for first use)
  python run_mace.py --interactive

  # Quick start with CIFs and predefined workflow
  python run_mace.py --quick-start --cif-dir ./my_cifs --workflow full_electronic

  # Execute a previously saved workflow plan
  python run_mace.py --execute workflow_plan_20241216_143022.json

  # Check status of running workflows
  python run_mace.py --status

  # Show available workflow templates
  python run_mace.py --show-templates

Workflow Templates:
  basic_opt         : OPT
  opt_sp           : OPT → SP
  full_electronic  : OPT → SP → BAND → DOSS
  double_opt       : OPT → OPT2 → SP
  complete         : OPT → SP → BAND → DOSS → FREQ
  transport_analysis : OPT → SP → TRANSPORT
  charge_analysis  : OPT → SP → CHARGE+POTENTIAL
  combined_analysis : OPT → SP → BAND → DOSS → TRANSPORT
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
                       choices=["basic_opt", "opt_sp", "full_electronic", "double_opt", "complete", 
                               "transport_analysis", "charge_analysis", "combined_analysis"],
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
    # Banner and animation already shown by mace_cli
    print("\nStarting Interactive Workflow Planner")
    print("=" * 60)
    
    planner = WorkflowPlanner(args.work_dir, args.db_path)
    planner.main_interactive_workflow()


def run_execute_mode(args):
    """Execute a saved workflow plan"""
    plan_file = Path(args.execute)
    
    if not plan_file.exists():
        print(f"Error: Workflow plan file not found: {plan_file}")
        sys.exit(1)
        
    print(f"Executing Workflow Plan: {plan_file}")
    print("=" * 60)
    
    executor = WorkflowExecutor(args.work_dir, args.db_path)
    executor.execute_workflow_plan(plan_file)


def run_quick_start_mode(args):
    """Run quick start mode with minimal configuration"""
    print("Quick Start Workflow Setup")
    print("=" * 60)
    
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
        "complete": ["OPT", "SP", "BAND", "DOSS", "FREQ"],
        "transport_analysis": ["OPT", "SP", "TRANSPORT"],
        "charge_analysis": ["OPT", "SP", "CHARGE+POTENTIAL"],
        "combined_analysis": ["OPT", "SP", "BAND", "DOSS", "TRANSPORT"]
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


def _get_quick_slurm_config(calc_type: str, step_num: int) -> Dict[str, Any]:
    """Get SLURM configuration for quick start mode"""
    import os
    from pathlib import Path
    
    # Determine script type and resources based on calculation
    base_type = calc_type.rstrip("0123456789")
    
    # Get the job scripts directory - correctly handle the path
    mace_dir = Path(__file__).parent.parent
    job_scripts_dir = mace_dir / "mace" / "submission"
    
    if base_type in ["OPT", "SP", "FREQ"]:
        script_name = "submitcrystal23.sh"
        default_resources = {
            "OPT": {"ntasks": 32, "memory_per_cpu": "5G", "walltime": "7-00:00:00"},
            "SP": {"ntasks": 32, "memory_per_cpu": "4G", "walltime": "3-00:00:00"},
            "FREQ": {"ntasks": 32, "memory_per_cpu": "5G", "walltime": "7-00:00:00"}
        }
    else:  # D3 calculations
        script_name = "submit_prop.sh"
        default_resources = {
            "BAND": {"ntasks": 28, "memory_per_cpu": "80G", "walltime": "2:00:00"},
            "DOSS": {"ntasks": 28, "memory_per_cpu": "80G", "walltime": "2:00:00"},
            "TRANSPORT": {"ntasks": 28, "memory_per_cpu": "80G", "walltime": "2:00:00"},
            "CHARGE+POTENTIAL": {"ntasks": 28, "memory_per_cpu": "80G", "walltime": "2:00:00"}
        }
    
    resources = default_resources.get(base_type, default_resources.get("OPT"))
    
    return {
        "scripts": {
            script_name: {
                "source_script": script_name,  # Just use the script name, executor will find it
                "step_specific_name": f"{script_name.replace('.sh', '')}_{calc_type.lower()}_{step_num}.sh",
                "workflow_id": None,  # Will be set during execution
                "resources": resources,
                "calculation_type": calc_type
            }
        },
        "resources": resources,
        "modules": {}
    }


def _get_basic_d3_config_for_quick_start(calc_type: str) -> Dict[str, Any]:
    """Get basic D3 configuration for quick start mode"""
    configs = {
        "BAND": {
            "calculation_type": "BAND",
            "path": "auto",
            "bands": "auto", 
            "shrink": "auto",
            "labels": "auto",
            "auto_path": True,
            "n_points": 10000,
            "path_method": "coordinates"
        },
        "DOSS": {
            "calculation_type": "DOSS",
            "n_points": 10000,
            "bands": "all",
            "projection_type": 0,  # Total DOS only
            "energy_range": [-20, 20]
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
            "charge_config": {
                "type": "ECH3",
                "n_points": 1000,
                "scale": 3.0,
                "use_range": False
            },
            "potential_config": {
                "type": "POT3",
                "n_points": 1000,
                "scale": 3.0,
                "use_range": False
            }
        }
    }
    return configs.get(calc_type, {})


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
        elif calc_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]:
            # All D3 calculations now use CRYSTALOptToD3.py
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": calc_type,
                "source": "CRYSTALOptToD3.py",
                "requires_wavefunction": True,
                "d3_calculation": True,
                "d3_config_mode": "basic",
                "d3_config": _get_basic_d3_config_for_quick_start(calc_type)
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
        
        # Add SLURM configuration for each step
        slurm_config = _get_quick_slurm_config(calc_type, step_num)
        step_configs[f"{calc_type}_{step_num}"]["slurm_config"] = slurm_config
    
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
    print("MACE Workflow Status")
    print("=" * 60)
    
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
    print(f"  python mace.py queue --status")


def show_workflow_templates():
    """Show available workflow templates"""
    print("Available Workflow Templates")
    print("=" * 60)
    
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
        },
        "transport_analysis": {
            "sequence": ["OPT", "SP", "TRANSPORT"],
            "description": "Transport properties calculation (conductivity, Seebeck)"
        },
        "charge_analysis": {
            "sequence": ["OPT", "SP", "CHARGE+POTENTIAL"],
            "description": "Charge density and electrostatic potential analysis"
        },
        "combined_analysis": {
            "sequence": ["OPT", "SP", "BAND", "DOSS", "TRANSPORT"],
            "description": "Electronic structure with transport properties"
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
        "FREQ": "Vibrational frequency calculation",
        "TRANSPORT": "Transport properties (conductivity, Seebeck coefficient)",
        "CHARGE+POTENTIAL": "Charge density and electrostatic potential"
    }
    
    for calc_type, description in calc_types.items():
        print(f"  {calc_type:<6} : {description}")


if __name__ == "__main__":
    main()