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
    from mace.workflow.common.constants import WORKFLOW_TEMPLATES, WORKFLOW_TEMPLATE_DESCRIPTIONS
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

  # First step as SLURM job arrays, at most 10 tasks of each running at once
  python run_mace.py --quick-start --d12-dir ./decks --workflow opt_sp --array --max-running 10

Workflow Templates:
  basic_opt         : OPT
  opt_sp           : OPT → SP
  opt_sp_freq      : OPT → SP → FREQ
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
    mode_group.add_argument("--interactive", "-i", action="store_true",
                           help="Run interactive workflow planner")
    mode_group.add_argument("--execute", "-e", metavar="PLAN_FILE",
                           help="Execute a saved workflow plan")
    mode_group.add_argument("--quick-start", "-q", action="store_true",
                           help="Quick start with minimal configuration")
    mode_group.add_argument("--status", "-s", nargs="?", const=True, default=False, metavar="ID",
                           help="List the workflows of the folder and their state (as --list); with a workflow ID, that "
                                "workflow's steps per material (add --json for JSON)")
    mode_group.add_argument("--list", action="store_true",
                           help="List the workflows planned in the folder (--work-dir) and their state")
    mode_group.add_argument("--spec", metavar="SPEC",
                           help="A workflow spec (JSON): check it, or write its plan with --write-plan")
    mode_group.add_argument("--pause", metavar="ID",
                           help="Pause a workflow: completions are recorded, nothing new is submitted")
    mode_group.add_argument("--resume", metavar="ID",
                           help="Resume a paused workflow and start the steps held meanwhile")
    mode_group.add_argument("--cancel", metavar="ID",
                           help="Cancel a workflow: stop its progression and scancel its own "
                                "queued/running jobs (lists them and asks first; --yes skips the question)")
    mode_group.add_argument("--process", metavar="ID",
                           help="Start the next steps of a workflow whose completion step did not run "
                                "(what the job callback would have done), for this workflow only")
    mode_group.add_argument("--show-templates", "-T", action="store_true",
                           help="Show available workflow templates")

    # Quick start options
    parser.add_argument("--cif-dir", "-c", help="Directory containing CIF files (for quick start)")
    parser.add_argument("--d12-dir", "-d", help="Directory containing D12 files (for quick start)")
    parser.add_argument("--workflow", "-w", default="full_electronic",
                       choices=list(WORKFLOW_TEMPLATES),
                       help="Workflow template for quick start")

    # Common options
    parser.add_argument("--work-dir", "-W", default=None, help="Working directory (default: the current "
                        "directory; for --spec of decks, the decks' folder)")
    parser.add_argument("--db-path", "-D", default="materials.db", help="Database path")
    parser.add_argument("--max-jobs", "-j", type=int, default=None,
                        help="At most N of this workflow's jobs queued at once (default: none; all "
                             "workflows share your queue budget)")
    parser.add_argument("--array", action="store_true",
                        help="Submit the first step as SLURM job arrays: materials whose job scripts ask for exactly the same resources share one array, one task per material. Later steps are submitted per material as each job completes, as always.")
    parser.add_argument("--max-running", metavar="N", type=int,
                        help="With --array: at most N tasks of each array run at once (--array=0-M%%N). The limit is per array; recovery resubmissions are normal jobs and are not limited.")
    
    parser.add_argument("--write-plan", action="store_true",
                        help="With --spec: write the plan (and, for decks, its job scripts)")
    parser.add_argument("--json", action="store_true",
                        help="With --status ID, --list, --spec, --pause/--resume/--cancel/--process: one JSON document on stdout")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="With --cancel: do not ask before canceling the listed jobs")

    args = parser.parse_args()
    work_dir_given = args.work_dir is not None
    if args.work_dir is None:
        args.work_dir = "."
    if args.write_plan and not args.spec:
        parser.error("--write-plan goes with --spec")
    if args.yes and not args.cancel:
        parser.error("--yes goes with --cancel")
    if args.json and not (args.list or args.spec or args.pause or args.resume or args.cancel or args.process
                          or args.status):
        parser.error("--json goes with --status, --list, --spec, --pause, --resume, --cancel or --process")
    if args.max_running is not None and not args.array:
        parser.error("--max-running limits the tasks of a job array; use it with --array")
    if args.max_running is not None and args.max_running < 1:
        parser.error("--max-running needs a number of 1 or more")
    if args.array and not (args.interactive or args.execute or args.quick_start):
        parser.error("--array goes with --interactive, --quick-start or --execute")
    
    if args.interactive:
        run_interactive_mode(args)
    elif args.execute:
        run_execute_mode(args)
    elif args.quick_start:
        run_quick_start_mode(args)
    elif isinstance(args.status, str) or args.list or args.spec or args.pause or args.resume \
            or args.cancel or args.process:
        from mace.workflow import commands
        folder = Path(args.work_dir).resolve()
        if isinstance(args.status, str):
            rc = commands.cmd_status(folder, args.status, args.json)
        elif args.list:
            rc = commands.cmd_list(folder, args.json)
        elif args.spec:
            rc = commands.cmd_spec(args.spec, folder, args.db_path, args.write_plan, args.json,
                                   work_dir_given=work_dir_given)
        elif args.pause:
            rc = commands.cmd_pause(folder, args.pause, args.json)
        elif args.resume:
            rc = commands.cmd_resume(folder, args.resume, args.json)
        elif args.process:
            rc = commands.cmd_process(folder, args.process, args.json)
        else:
            rc = commands.cmd_cancel(folder, args.cancel, args.json, args.yes)
        sys.exit(rc)
    elif args.status:
        sys.exit(run_status_mode(args))
    elif args.show_templates:
        show_workflow_templates()
    

def run_interactive_mode(args):
    """Run interactive workflow planning"""
    # Banner and animation already shown by mace_cli
    print("\nStarting Interactive Workflow Planner")
    print("=" * 60)
    
    planner = WorkflowPlanner(args.work_dir, args.db_path)
    # --array: written into the plan this session saves (no new prompt, so
    # piped answers stay as they are)
    planner.job_arrays = job_arrays_setting(args)
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
    if getattr(args, "array", False):
        executor.job_arrays_override = job_arrays_setting(args)
    executor.execute_workflow_plan(plan_file)


def job_arrays_setting(args):
    """The plan's job_arrays setting for --array [--max-running N], or None."""
    if not getattr(args, "array", False):
        return None
    return {"first_step": True, "max_running": getattr(args, "max_running", None)}


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
    workflow_templates = WORKFLOW_TEMPLATES
    
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
    # The plan carries no workflow_id; its file is named by the one minted
    # for it, and the execution must use that same id (one plan file).
    executor.default_workflow_id = "workflow_" + plan_file.stem[len("workflow_plan_"):]
    executor.execute_workflow_plan(plan_file)


def _get_quick_slurm_config(calc_type: str, step_num: int, inputs=None, basis=None) -> Dict[str, Any]:
    """Get SLURM configuration for quick start mode (``inputs``: the plan's
    decks or CIFs, which size the job when the site turned size-aware
    suggestions on: [resources] auto)"""
    import os
    from pathlib import Path
    
    # Determine script type and resources based on calculation
    base_type = calc_type.rstrip("0123456789")
    # a TAU step's strained single points: the SP's job, 1.5x its walltime
    tau = base_type == "TAU"
    if tau:
        base_type = "SP"
    
    # Get the job scripts directory - correctly handle the path
    mace_dir = Path(__file__).parent.parent
    job_scripts_dir = mace_dir / "mace" / "submission"
    
    from mace.workflow.common.constants import MODULE_STEP_TYPES, SLURM_RESOURCE_DEFAULTS
    if base_type in ["OPT", "SP", "FREQ"] or base_type in MODULE_STEP_TYPES:
        script_name = "submitcrystal23.sh"
        default_resources = {
            "OPT": {"ntasks": 32, "memory_per_cpu": "5G", "walltime": "7-00:00:00"},
            "SP": {"ntasks": 32, "memory_per_cpu": "4G", "walltime": "3-00:00:00"},
            "FREQ": {"ntasks": 32, "memory_per_cpu": "5G", "walltime": "7-00:00:00"}
        }
        # ELASTIC, EOS, QHA, CPHF: the workflow's built-in sizes of these jobs
        default_resources.update({t: {k: SLURM_RESOURCE_DEFAULTS[t][k]
                                      for k in ("ntasks", "memory_per_cpu", "walltime")}
                                  for t in MODULE_STEP_TYPES})
    else:  # D3 calculations
        from mace.workflow.common.constants import PROPERTIES_MEMORY, PROPERTIES_NTASKS
        script_name = "submit_prop.sh"
        default_resources = {
            "BAND": {"ntasks": PROPERTIES_NTASKS, "memory": PROPERTIES_MEMORY, "walltime": "2:00:00"},
            "DOSS": {"ntasks": PROPERTIES_NTASKS, "memory": PROPERTIES_MEMORY, "walltime": "2:00:00"},
            "TRANSPORT": {"ntasks": PROPERTIES_NTASKS, "memory": PROPERTIES_MEMORY, "walltime": "2:00:00"},
            "CHARGE+POTENTIAL": {"ntasks": PROPERTIES_NTASKS, "memory": PROPERTIES_MEMORY, "walltime": "2:00:00"},
            "ORBITALS": {"ntasks": PROPERTIES_NTASKS, "memory": PROPERTIES_MEMORY, "walltime": "1:00:00"}
        }
    
    resources = default_resources.get(base_type, default_resources.get("OPT"))
    # the site's account and CRYSTAL module (and partition / Python modules
    # when set), stored in the plan once: mace/submission/site.py
    from mace.submission import site
    try:
        stored = site.plan_resources(calc_type=base_type)
    except site.SiteConfigError as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)
    # built-in values stay implicit: the template's own lines, byte for byte
    if stored.get("account") == site.DEFAULTS["account"]:
        stored.pop("account")
    if [stored.get("module")] == site.DEFAULTS["crystal"] and "modules" not in stored:
        stored.pop("module")
    resources = {**resources, **stored}
    # the site's [resources] job size, when set (memory in the site's form)
    kind = "scf" if script_name == "submitcrystal23.sh" else "prop"
    auto = site.auto_for_inputs(inputs, kind, base_type, basis=basis) if inputs else None
    if auto:
        print(f"  {calc_type}: suggested from the inputs' size ([resources] auto): {auto['reason']}")
    resources = site.overlay_resources(resources, site.load_resolved(), kind, base_type, auto)
    if tau:
        from mace.workflow.tau_step import scale_walltime
        resources = {**resources, "walltime": scale_walltime(resources["walltime"])}
    
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
    from mace.workflow.common.constants import opt2d3_default_band_config, opt2d3_default_doss_config, opt2d3_default_transport_config
    configs = {
        # opt2d3's answers when Enter is pressed through it, as the engine
        # default and the planner's Basic BAND use: the same parent gets the
        # same deck by every route
        "BAND": opt2d3_default_band_config(),
        "DOSS": opt2d3_default_doss_config(),
        "TRANSPORT": opt2d3_default_transport_config(),
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
        },
        # opt2d3's answer when Enter is pressed: the band-edge k points
        "ORBITALS": {
            "calculation_type": "ORBITALS",
            "kmesh": "band_edges"
        }
    }
    return configs.get(calc_type, {})


def create_quick_workflow_plan(input_dir, input_files, input_type, sequence, args):
    """Create a quick workflow plan with default settings"""
    from datetime import datetime
    from d12_constants import opt_convergence, scf_tolerances
    
    # Default CIF conversion config (adapt to first calculation type)
    first_calc_type = sequence[0] if sequence else "OPT"
    default_cif_config = {
        "symmetry_handling": "CIF",
        "write_only_unique": True,
        "dimensionality": "CRYSTAL",
        "calculation_type": "SP" if first_calc_type == "SP" else "OPT",
        "optimization_type": "FULLOPTG" if first_calc_type != "SP" else None,
        # opt2d12's Standard convergence (the one presets table)
        "optimization_settings": opt_convergence("1", upper=True),
        "method": "DFT",
        "dft_functional": "B3LYP",
        "use_dispersion": True,
        "basis_set_type": "INTERNAL",
        "basis_set": "POB-TZVP-REV2",
        "dft_grid": "XLGRID",
        "is_spin_polarized": True,
        "use_smearing": False,
        "tolerances": scf_tolerances("1"),
        "scf_method": "DIIS",
        "scf_maxcycle": 800,
        "fmixing": 30
    }
    
    from mace.workflow.common.constants import MODULE_STEP_TYPES
    # Default step configurations
    step_configs = {}
    for i, calc_type in enumerate(sequence):
        step_num = i + 1
        # a numbered repeat (OPT2, OPT3, SP2, BAND2, ...) is configured as
        # its base type
        base_type = calc_type.rstrip("0123456789") or calc_type
        
        if (calc_type == "OPT" or calc_type == "SP") and i == 0 and input_type == "cif":
            step_configs[f"{calc_type}_{step_num}"] = {"source": "cif_conversion"}
        elif base_type == "OPT":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "OPT",
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True
            }
        elif base_type == "SP":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "SP",
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True
            }
        elif base_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS"]:
            # All D3 calculations now use CRYSTALOptToD3.py
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": base_type,
                "source": "CRYSTALOptToD3.py",
                "requires_wavefunction": True,
                "d3_calculation": True,
                "d3_config_mode": "basic",
                "d3_config": _get_basic_d3_config_for_quick_start(base_type)
            }
        elif base_type == "TAU":
            # the TRANSPORT step's relaxation time from strained single points
            # of the SP (mace/workflow/tau_step.py)
            from mace.workflow.tau_step import check_options
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "TAU",
                "source": "mace.transport.strain",
                "tau": check_options({}),
            }
        elif base_type in MODULE_STEP_TYPES:
            # ELASTIC, EOS, QHA, CPHF: opt2d12's module deck at the parent's
            # geometry and method; no elastic_settings is CRYSTAL's defaults
            # (an elastic parent's own block for the same module)
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": base_type,
                "source": "CRYSTALOptToD12.py",
                "inherit_settings": True,
            }
        elif base_type == "FREQ":
            step_configs[f"{calc_type}_{step_num}"] = {
                "calculation_type": "FREQ",
                "source": "CRYSTALOptToD12.py",
                "inherit_base_settings": True,
                "frequency_settings": {
                    "mode": "GAMMA",
                    "temperatures": [298.15],
                    # No tolerances or intensities: opt2d12's FREQ defaults
                    # apply (Very tight and none, or a FREQ parent's own)
                }
            }
        
        # Add SLURM configuration for each step
        slurm_config = _get_quick_slurm_config(calc_type, step_num, input_files,
                                               default_cif_config.get("basis_set") if input_type == "cif" else None)
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
            "auto_progression": True,
            # a cap somebody chose (--max-jobs, a spec's execution.max_jobs) is kept even when it is 200
            **({"max_jobs_explicit": True} if args.max_jobs is not None else {}),
        }
    }
    job_arrays = job_arrays_setting(args)
    if job_arrays:
        plan["job_arrays"] = job_arrays
    
    return plan


def run_status_mode(args):
    """``mace workflow --status`` without an ID: the workflows planned in
    the folder and their state (as ``--list``), read-only (no database is
    created or changed)."""
    from mace.workflow import commands
    rc = commands.cmd_list(Path(args.work_dir).resolve(), args.json)
    if not args.json:
        print("\nOne workflow's steps per material: mace workflow --status <ID>"
              "\nAll your jobs and their ETA:      mace status")
    return rc


def show_workflow_templates():
    """Show available workflow templates"""
    print("Available Workflow Templates")
    print("=" * 60)
    
    templates = {name: {"sequence": seq, "description": WORKFLOW_TEMPLATE_DESCRIPTIONS.get(name, "")}
                 for name, seq in WORKFLOW_TEMPLATES.items()}
    
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
        "CHARGE+POTENTIAL": "Charge density and electrostatic potential",
        "ORBITALS": "Orbitals as Molden files (NEWK + ORBITALS)",
        "ELASTIC": "Elastic, piezoelectric or photoelastic constants (after OPT)",
        "EOS": "Equation of state (after OPT)",
        "QHA": "Quasi-harmonic approximation (after OPT)",
        "CPHF": "Dielectric response, CPHF/KS (after OPT)",
    }
    
    for calc_type, description in calc_types.items():
        print(f"  {calc_type:<6} : {description}")
    print("\n  Finite-displacement phonons with phonopy are not a workflow step: "
          "mace opt2d12 --out-file PARENT.out --calc-type PHONOPY")


if __name__ == "__main__":
    main()