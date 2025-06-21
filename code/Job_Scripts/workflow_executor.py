#!/usr/bin/env python3
"""
CRYSTAL Workflow Executor
=========================
Enhanced execution engine that integrates with the workflow planner to execute
complex calculation sequences with full configuration management.

Features:
- JSON-based configuration management
- Integration with NewCifToD12.py and CRYSTALOptToD12.py
- Intelligent job submission and dependency management
- Progress tracking and error recovery
- Resource optimization

Author: Workflow execution system
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import threading
import queue

# Import our components
try:
    from material_database import MaterialDatabase
    from enhanced_queue_manager import EnhancedCrystalQueueManager
    sys.path.append(str(Path(__file__).parent.parent / "Crystal_To_CIF"))
    from d12creation import *
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class WorkflowExecutor:
    """
    Executes planned workflows with full configuration management and error handling.
    
    Integrates deeply with NewCifToD12.py and CRYSTALOptToD12.py to ensure
    consistent settings across all calculation steps.
    """
    
    def __init__(self, work_dir: str = ".", db_path: str = "materials.db"):
        self.work_dir = Path(work_dir).resolve()
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        
        # Set up directories
        self.configs_dir = self.work_dir / "workflow_configs"
        self.outputs_dir = self.work_dir / "workflow_outputs"
        self.temp_dir = self.work_dir / "workflow_temp"
        
        for dir_path in [self.configs_dir, self.outputs_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize queue manager
        self.queue_manager = EnhancedCrystalQueueManager(
            d12_dir=str(self.outputs_dir),
            max_jobs=200,
            enable_tracking=True,
            db_path=self.db_path
        )
        
        # Track active workflows
        self.active_workflows = {}
        self.execution_lock = threading.RLock()
        
    def execute_workflow_plan(self, plan_file: Path):
        """Execute a complete workflow plan"""
        print(f"Executing workflow plan: {plan_file}")
        
        with open(plan_file, 'r') as f:
            plan = json.load(f)
            
        # Use workflow_id from plan, or generate new one if missing
        workflow_id = plan.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"Using workflow_id: {workflow_id}")
        
        try:
            # Phase 1: Prepare input files
            self.prepare_input_files(plan, workflow_id)
            
            # Phase 2: Execute calculation sequence
            self.execute_workflow_steps(plan, workflow_id)
            
            # Phase 3: Monitor and manage execution
            if workflow_id in self.active_workflows:
                self.monitor_workflow_execution(workflow_id)
            else:
                print("Phase 3: Workflow monitoring skipped (workflow completed synchronously)")
            
        except Exception as e:
            print(f"Workflow execution failed: '{workflow_id}'")
            self.cleanup_workflow(workflow_id)
            raise
            
    def prepare_input_files(self, plan: Dict[str, Any], workflow_id: str):
        """Prepare all input files according to the plan"""
        print("Phase 1: Preparing input files...")
        
        # Create workflow-specific directories
        workflow_dir = self.outputs_dir / workflow_id
        workflow_dir.mkdir(exist_ok=True)
        
        # Phase 1a: Convert CIFs if needed
        if plan['input_files']['cif']:
            print("  Converting CIF files to D12 format...")
            self.convert_cifs_with_config(plan, workflow_id)
            
        # Phase 1b: Organize existing D12s
        if plan['input_files']['d12']:
            print("  Organizing existing D12 files...")
            self.organize_existing_d12s(plan, workflow_id)
            
        # Phase 1c: Generate subsequent calculation inputs
        print("  Pre-generating calculation configurations...")
        self.generate_calculation_configs(plan, workflow_id)
        
    def execute_workflow_steps(self, plan: Dict[str, Any], workflow_id: str):
        """Execute the complete workflow with proper calculation folder structure"""
        print("Phase 2: Executing workflow calculations...")
        
        workflow_dir = self.outputs_dir / workflow_id
        workflow_sequence = plan['workflow_sequence']
        
        # Get initial D12 files (from CIF conversion or existing)
        step_001_dir = workflow_dir / "step_001_OPT"
        d12_files = list(step_001_dir.glob("*.d12"))
        
        if not d12_files:
            print("Error: No D12 files found for workflow execution!")
            return
            
        print(f"Found {len(d12_files)} materials to process")
        print(f"Workflow sequence: {' → '.join(workflow_sequence)}")
        
        # Execute first step (always OPT)
        first_step = workflow_sequence[0]
        step_dir = workflow_dir / f"step_001_{first_step}"
        
        print(f"\nExecuting Step 1: {first_step}")
        submitted_jobs = self.execute_step(plan, workflow_id, first_step, 1, d12_files, step_dir)
        
        if submitted_jobs:
            print(f"Successfully submitted {len(submitted_jobs)} {first_step} calculations")
            print("Monitor progress with:")
            print(f"  python enhanced_queue_manager.py --status")
            print(f"  squeue -u $USER")
        else:
            print(f"Warning: No {first_step} jobs were submitted")
            
    def execute_step(self, plan: Dict[str, Any], workflow_id: str, calc_type: str, 
                    step_num: int, d12_files: List[Path], step_dir: Path) -> List[str]:
        """Execute a single workflow step with individual calculation folders"""
        submitted_jobs = []
        
        for d12_file in d12_files:
            # Extract material name from D12 file
            material_name = self.extract_material_name(d12_file)
            
            # Create individual calculation folder
            calc_dir = step_dir / material_name
            calc_dir.mkdir(exist_ok=True)
            
            # Move (not copy) D12 file to calculation folder to avoid redundancy
            calc_d12_file = calc_dir / f"{material_name}.d12"
            if not calc_d12_file.exists():
                shutil.move(str(d12_file), str(calc_d12_file))
            
            # Generate individual SLURM script
            slurm_script = self.generate_slurm_script(
                plan, workflow_id, calc_type, step_num, material_name, calc_dir
            )
            
            # Submit job
            job_id = self.submit_slurm_job(slurm_script, calc_dir)
            if job_id:
                submitted_jobs.append(job_id)
                print(f"  Submitted {material_name}: Job ID {job_id}")
            else:
                print(f"  Failed to submit {material_name}")
                
        return submitted_jobs
        
    def extract_core_material_name(self, d12_file: Path) -> str:
        """Extract the core material name by removing ALL workflow-related suffixes"""
        name = d12_file.stem
        
        # ACTUAL suffixes from NewCifToD12.py, CRYSTALOptToD12.py, and d12creation.py
        # Based on real filename construction patterns found in the code
        suffixes_to_remove = [
            # === BASIS SETS (from NewCifToD12.py and d12creation.py) ===
            '_POB-TZVP-REV2', '_POB-DZVP-REV2', '_POB-TZVP', '_POB-DZVP',
            '_STO-3G', '_3-21G', '_6-31G', '_6-311G', '_def2-SVP', '_def2-TZVP',
            '_DZVP-REV2', '_TZVP-REV2',  # External basis directory names
            
            # === DFT FUNCTIONALS WITH DISPERSION (from NewCifToD12.py) ===
            '_HSE06-D3', '_PBE-D3', '_B3LYP-D3', '_PBE0-D3', '_SCAN-D3',
            '_BLYP-D3', '_BP86-D3', '_wB97X-D3',
            
            # === DFT FUNCTIONALS WITHOUT DISPERSION ===
            '_HSE06', '_PBE', '_B3LYP', '_PBE0', '_SCAN', '_BLYP', '_BP86', '_wB97X',
            '_LDA', '_VWN', '_PWGGA', '_PW91',
            
            # === HARTREE-FOCK METHODS ===
            '_RHF', '_UHF', '_HF',
            
            # === CRYSTALOptToD12.py SPECIFIC SUFFIXES ===
            '_optimized',  # Always added by CRYSTALOptToD12.py
            
            # === CALCULATION TYPES (from NewCifToD12.py pattern) ===
            '_OPT', '_SP', '_FREQ',
            
            # === DIMENSIONALITY (from NewCifToD12.py) ===
            '_CRYSTAL', '_SLAB', '_POLYMER', '_MOLECULE',
            
            # === SYMMETRY SETTINGS (from NewCifToD12.py) ===
            '_symm', '_P1',
            
            # === LOWERCASE CALC TYPES (from CRYSTALOptToD12.py) ===
            '_opt', '_sp', '_freq',
            
            # === WORKFLOW-ADDED SUFFIXES ===
            '_band', '_doss', '_BAND', '_DOSS'
        ]
        
        # Remove suffixes iteratively until no more matches
        original_name = name
        while True:
            name_before = name
            for suffix in suffixes_to_remove:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            # If no suffix was removed, we're done
            if name == name_before:
                break
        
        # Final cleanup - remove any trailing underscores or numbers
        name = name.rstrip('_0123456789')
        
        # Only replace truly problematic characters for filesystem compatibility
        # Preserve ^ and , as they are commonly used in chemical notation
        name = name.replace(' ', '_')  # Spaces to underscores
        name = name.replace('/', '_').replace('\\', '_')  # Path separators
        # Keep other characters like ^, ,, ., - as they are common in chemical names
        
        # Ensure we have a valid name
        if not name or len(name) < 2:
            # Fallback to a basic version if cleaning removed too much
            name = original_name.split('_')[0]
            
        # Ensure it starts with a letter (required for some systems)
        if name and not name[0].isalpha():
            name = f"mat_{name}"
            
        return name or "unknown_material"
        
    def extract_material_name(self, d12_file: Path) -> str:
        """Extract material name with appropriate calculation type suffix"""
        core_name = self.extract_core_material_name(d12_file)
        
        # For OPT calculations, add _opt suffix
        # This distinguishes from the raw core name
        return f"{core_name}_opt"
        
    def generate_slurm_script(self, plan: Dict[str, Any], workflow_id: str, 
                             calc_type: str, step_num: int, material_name: str, 
                             calc_dir: Path) -> Path:
        """Generate individual SLURM script for a calculation"""
        
        # Get step configuration
        step_key = f"{calc_type}_{step_num}"
        step_config = plan.get('step_configurations', {}).get(step_key, {})
        
        # Determine template script name
        if calc_type in ['OPT', 'OPT2', 'SP', 'FREQ']:
            template_name = f"submitcrystal23_{calc_type.lower()}_{step_num}.sh"
        elif calc_type in ['BAND', 'DOSS']:
            template_name = f"submit_prop_{calc_type.lower()}_{step_num}.sh"
        else:
            template_name = f"submit_{calc_type.lower()}_{step_num}.sh"
            
        # Find template script in work directory
        template_script = self.work_dir / "workflow_scripts" / template_name
        
        if not template_script.exists():
            # Fall back to basic template
            if calc_type in ['OPT', 'OPT2', 'SP', 'FREQ']:
                template_script = self.work_dir / "workflow_scripts" / "submitcrystal23_opt_1.sh"
            else:
                template_script = self.work_dir / "workflow_scripts" / "submit_prop_band_3.sh"
                
        if not template_script.exists():
            raise FileNotFoundError(f"No SLURM template found for {calc_type}")
            
        # Read template content
        with open(template_script, 'r') as f:
            template_content = f.read()
            
        # Update template with specific values
        script_content = self.customize_slurm_script(
            template_content, workflow_id, calc_type, step_num, material_name, calc_dir
        )
        
        # Write individual script
        script_file = calc_dir / f"{material_name}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        # Make executable
        script_file.chmod(0o755)
        
        return script_file
        
    def customize_slurm_script(self, template_content: str, workflow_id: str, 
                              calc_type: str, step_num: int, material_name: str, 
                              calc_dir: Path) -> str:
        """Customize SLURM script template for specific calculation"""
        
        # Define scratch directory structure for workflow isolation
        # Set scratch to parent directory so that mkdir -p $scratch/$JOB works correctly
        if calc_type in ['BAND', 'DOSS']:
            scratch_base_dir = f"$SCRATCH/{workflow_id}/step_{step_num:03d}_{calc_type}"
        else:
            scratch_base_dir = f"$SCRATCH/{workflow_id}/step_{step_num:03d}_{calc_type}"
        
        # Replace placeholders in template following the existing pattern
        script_content = template_content
        
        # Replace $1 placeholders with material name
        script_content = script_content.replace('$1', material_name)
        
        # Update scratch directory paths to use base directory
        # This preserves the original template pattern of mkdir -p $scratch/$JOB
        # where $JOB will be the material_name, creating the full path
        script_content = script_content.replace(
            'export scratch=$SCRATCH/crys23',
            f'export scratch={scratch_base_dir}'
        )
        script_content = script_content.replace(
            'export scratch=$SCRATCH/crys23/prop',
            f'export scratch={scratch_base_dir}'
        )
        
        # Add Python module loading for the callback scripts
        # Find the line with "module load CRYSTAL" and add Python module after it
        # Note: Python module loading is already in the base template, no need to add it again
        
        # Fix callback mechanism - check multiple possible locations for queue managers
        enhanced_callback = '''# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers
if [ -f $DIR/enhanced_queue_manager.py ]; then
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/crystal_queue_manager.py ]; then
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
fi'''
        
        # Replace the existing callback section
        import re
        callback_pattern = r'# ADDED: Auto-submit new jobs when this one completes.*?fi'
        script_content = re.sub(callback_pattern, enhanced_callback, script_content, flags=re.DOTALL)
        
        # Add workflow metadata as comments at the top
        metadata_lines = [
            f"# Generated by CRYSTAL Workflow Manager",
            f"# Workflow ID: {workflow_id}",
            f"# Calculation Type: {calc_type}",
            f"# Step: {step_num}",
            f"# Material: {material_name}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Insert metadata after shebang line
        lines = script_content.split('\n')
        if lines[0].startswith('#!/bin/bash'):
            lines = [lines[0]] + metadata_lines + lines[1:]
        else:
            lines = metadata_lines + lines
            
        return '\n'.join(lines)
        
    def submit_slurm_job(self, script_file: Path, calc_dir: Path) -> Optional[str]:
        """Submit SLURM job and return job ID"""
        try:
            # Change to calculation directory
            original_cwd = os.getcwd()
            os.chdir(calc_dir)
            
            # Check if this is a script generator (template) or actual SLURM script
            job_name = script_file.stem
            
            # Check if the script contains script generation logic
            with open(script_file, 'r') as f:
                script_content = f.read()
            
            if 'echo \'#!/bin/bash --login\' >' in script_content or 'echo "#SBATCH' in script_content:
                # This is a script generator template - run locally to generate actual script
                print(f"  Running script generator locally: {script_file.name}")
                result = subprocess.run(
                    ['bash', script_file.name, job_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output (the template runs sbatch at the end)
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        if 'Submitted batch job' in line:
                            job_id = line.split()[-1]
                            return job_id
                    
                    # Maybe the template just generated the script but didn't submit it
                    # Look for generated script and submit it manually
                    generated_script = calc_dir / f"{job_name}.sh"
                    if generated_script.exists():
                        print(f"  Found generated script: {generated_script.name}")
                        result = subprocess.run(
                            ['sbatch', generated_script.name],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            output_lines = result.stdout.strip().split('\n')
                            for line in output_lines:
                                if 'Submitted batch job' in line:
                                    job_id = line.split()[-1]
                                    return job_id
                else:
                    print(f"Error running script generator: {result.stderr}")
                    
            else:
                # This is a regular SLURM script - submit directly
                print(f"  Submitting SLURM script directly: {script_file.name}")
                result = subprocess.run(
                    ['sbatch', script_file.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        if 'Submitted batch job' in line:
                            job_id = line.split()[-1]
                            return job_id
                else:
                    print(f"SLURM submission failed: {result.stderr}")
                
        except Exception as e:
            print(f"Error submitting job: {e}")
            
        finally:
            os.chdir(original_cwd)
            
        return None
        
    def convert_cifs_with_config(self, plan: Dict[str, Any], workflow_id: str):
        """Convert CIF files using saved configuration"""
        cif_config = plan.get('cif_conversion_config')
        if not cif_config:
            raise ValueError("CIF conversion config not found in plan")
            
        workflow_dir = self.outputs_dir / workflow_id
        cif_output_dir = workflow_dir / "step_001_OPT"
        cif_output_dir.mkdir(exist_ok=True)
        
        # Save CIF config for this workflow
        cif_config_file = self.temp_dir / f"{workflow_id}_cif_config.json"
        with open(cif_config_file, 'w') as f:
            json.dump(cif_config, f, indent=2)
            
        # Run NewCifToD12.py in batch mode
        # Check for local copy first
        local_script_path = self.work_dir / "NewCifToD12.py"
        if local_script_path.exists():
            script_path = local_script_path
        else:
            script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "NewCifToD12.py"
        
        conversion_cmd = [
            sys.executable, str(script_path),
            "--batch",
            "--options_file", str(cif_config_file),
            "--cif_dir", plan['input_directory'],
            "--output_dir", str(cif_output_dir)
        ]
        
        print(f"    Running: {' '.join(conversion_cmd)}")
        print(f"    Working directory: {os.getcwd()}")
        print(f"    Script path exists: {script_path.exists()}")
        print(f"    Config file exists: {cif_config_file.exists()}")
        print(f"    Input directory exists: {Path(plan['input_directory']).exists()}")
        
        # Add timeout to prevent hanging
        try:
            result = subprocess.run(conversion_cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            print("    CIF conversion timed out after 5 minutes")
            raise RuntimeError("CIF conversion timed out")
        
        print(f"    Return code: {result.returncode}")
        if result.stdout:
            print(f"    STDOUT: {result.stdout}")
        if result.stderr:
            print(f"    STDERR: {result.stderr}")
        
        if result.returncode != 0:
            print(f"CIF conversion failed:")
            raise RuntimeError("CIF conversion failed")
            
        print(f"    CIF conversion completed. Files in: {cif_output_dir}")
        
        # Update plan with generated D12 files
        generated_d12s = list(cif_output_dir.glob("*.d12"))
        plan['generated_d12s'] = [str(f) for f in generated_d12s]
        print(f"    Generated {len(generated_d12s)} D12 files")
        
    def organize_existing_d12s(self, plan: Dict[str, Any], workflow_id: str):
        """Organize existing D12 files for workflow execution"""
        workflow_dir = self.outputs_dir / workflow_id
        initial_dir = workflow_dir / "step_001_initial"
        initial_dir.mkdir(exist_ok=True)
        
        # Copy existing D12s to workflow directory
        existing_d12s = []
        for d12_path in plan['input_files']['d12']:
            d12_file = Path(d12_path)
            dest_file = initial_dir / d12_file.name
            shutil.copy2(d12_file, dest_file)
            existing_d12s.append(str(dest_file))
            
        plan['organized_d12s'] = existing_d12s
        print(f"    Organized {len(existing_d12s)} existing D12 files")
        
    def generate_calculation_configs(self, plan: Dict[str, Any], workflow_id: str):
        """Pre-generate all calculation step configurations"""
        print("  Generating calculation step configurations...")
        
        step_configs = plan['step_configurations']
        workflow_sequence = plan['workflow_sequence']
        
        for i, calc_type in enumerate(workflow_sequence):
            step_num = i + 1
            step_key = f"{calc_type}_{step_num}"
            
            if step_key in step_configs:
                config = step_configs[step_key]
                self.prepare_step_configuration(config, calc_type, step_num, workflow_id)
                
    def prepare_step_configuration(self, config: Dict[str, Any], calc_type: str, 
                                 step_num: int, workflow_id: str):
        """Prepare configuration for a specific calculation step"""
        config_file = self.temp_dir / f"{workflow_id}_step_{step_num:03d}_{calc_type}_config.json"
        
        # Enhance config with workflow-specific information
        enhanced_config = config.copy()
        enhanced_config.update({
            "workflow_id": workflow_id,
            "step_number": step_num,
            "calculation_type": calc_type,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(config_file, 'w') as f:
            json.dump(enhanced_config, f, indent=2)
            
        print(f"    Generated config for step {step_num}: {calc_type}")
        
    def execute_calculation_sequence(self, plan: Dict[str, Any], workflow_id: str):
        """Execute the planned calculation sequence"""
        print("Phase 2: Executing calculation sequence...")
        
        workflow_sequence = plan['workflow_sequence']
        workflow_dir = self.outputs_dir / workflow_id
        
        # Track workflow state
        self.active_workflows[workflow_id] = {
            "plan": plan,
            "current_step": 0,
            "total_steps": len(workflow_sequence),
            "step_status": {},
            "submitted_jobs": {},
            "start_time": datetime.now()
        }
        
        # Submit initial calculations
        self.submit_initial_calculations(workflow_id)
        
    def submit_initial_calculations(self, workflow_id: str):
        """Submit the first step calculations"""
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        sequence = plan['workflow_sequence']
        
        if not sequence:
            print("No calculations in workflow sequence")
            return
            
        first_calc_type = sequence[0]
        print(f"  Submitting initial {first_calc_type} calculations...")
        
        # Get input files for first step
        input_files = []
        
        # First check if there are D12 files in the step directory (created by workflow planner)
        step_num = 1  # First step is always step 1
        step_dir = self.outputs_dir / workflow_id / f"step_{step_num:03d}_{first_calc_type}"
        if step_dir.exists():
            step_d12_files = list(step_dir.glob("*.d12"))
            if step_d12_files:
                input_files = step_d12_files
                print(f"  Found {len(input_files)} D12 files in step directory: {step_dir}")
        
        # If no files in step directory, check the plan for input file locations
        if not input_files:
            if 'generated_d12s' in plan:
                input_files = [Path(f) for f in plan['generated_d12s']]
                print(f"  Using generated D12 files: {len(input_files)} files")
            elif 'organized_d12s' in plan:
                input_files = [Path(f) for f in plan['organized_d12s']]
                print(f"  Using organized D12 files: {len(input_files)} files")
            elif 'input_files' in plan and 'd12' in plan['input_files']:
                # This is the correct key for workflow planner generated plans
                input_files = [Path(f) for f in plan['input_files']['d12']]
                print(f"  Using input D12 files from plan: {len(input_files)} files")
                
                # Copy these files to the step directory for organization
                step_dir.mkdir(parents=True, exist_ok=True)
                copied_files = []
                for d12_file in input_files:
                    if d12_file.exists():
                        dest_file = step_dir / d12_file.name
                        if not dest_file.exists():
                            shutil.copy2(d12_file, dest_file)
                            print(f"    Copied: {d12_file.name} -> {dest_file}")
                        copied_files.append(dest_file)
                    else:
                        print(f"    Warning: Source file not found: {d12_file}")
                        
                # Use the copied files as the input files
                input_files = [f for f in copied_files if f.exists()]
                        
            elif 'input_directory' in plan:
                # Look for D12 files in the input directory
                input_dir = Path(plan['input_directory'])
                input_files = list(input_dir.glob("*.d12"))
                print(f"  Found {len(input_files)} D12 files in input directory: {input_dir}")
                
                if input_files:
                    # Copy files to step directory for better organization
                    step_dir.mkdir(parents=True, exist_ok=True)
                    copied_files = []
                    for d12_file in input_files:
                        dest_file = step_dir / d12_file.name
                        if not dest_file.exists():
                            shutil.copy2(d12_file, dest_file)
                            print(f"    Copied: {d12_file.name} -> {dest_file}")
                        copied_files.append(dest_file)
                    input_files = copied_files
                else:
                    print(f"Error: No D12 files found for workflow execution in {input_dir}!")
                    return
        
        if not input_files:
            print("Error: No input files found for workflow execution!")
            return
            
        # Submit jobs for each input file
        submitted_jobs = []
        for d12_file in input_files:
            job_id = self.submit_single_calculation(
                d12_file, first_calc_type, workflow_id, 1
            )
            if job_id:
                submitted_jobs.append(job_id)
                
        workflow_info["submitted_jobs"][f"step_1_{first_calc_type}"] = submitted_jobs
        workflow_info["step_status"][f"step_1_{first_calc_type}"] = "submitted"
        
        print(f"  Submitted {len(submitted_jobs)} {first_calc_type} calculations")
        
    def submit_single_calculation(self, input_file: Path, calc_type: str, 
                                workflow_id: str, step_num: int) -> Optional[str]:
        """Submit a single calculation"""
        try:
            # Create material ID from file
            material_id = self.create_material_id_from_file(input_file)
            print(f"  Created material_id: {material_id} from {input_file.name}")
            
            # Submit via queue manager
            calc_id = self.queue_manager.submit_calculation(
                d12_file=input_file,
                calc_type=calc_type,
                material_id=material_id
            )
            print(f"  Queue manager returned calc_id: {calc_id}")
            
            if calc_id:
                # Store workflow context in database
                print(f"  Updating calculation settings with workflow_id: {workflow_id}")
                self.db.update_calculation_settings(calc_id, {
                    "workflow_id": workflow_id,
                    "workflow_step": step_num,
                    "workflow_calc_type": calc_type
                })
                print(f"  Successfully updated workflow metadata for calc_id: {calc_id}")
            else:
                print(f"  Warning: No calc_id returned from queue manager for {material_id}")
                
            return calc_id
            
        except Exception as e:
            print(f"Failed to submit calculation for {input_file}: {e}")
            return None
            
    def create_material_id_from_file(self, file_path: Path) -> str:
        """Create a material ID from file path using comprehensive suffix removal"""
        name = file_path.stem
        
        # COMPREHENSIVE suffix removal based on actual d12creation patterns
        suffixes_to_remove = [
            # === BASIS SETS ===
            '_POB-TZVP-REV2', '_POB-DZVP-REV2', '_POB-TZVP', '_POB-DZVP',
            '_STO-3G', '_3-21G', '_6-31G', '_6-311G', '_def2-SVP', '_def2-TZVP',
            '_DZVP-REV2', '_TZVP-REV2',
            
            # === DFT FUNCTIONALS WITH DISPERSION ===
            '_HSE06-D3', '_PBE-D3', '_B3LYP-D3', '_PBE0-D3', '_SCAN-D3',
            '_BLYP-D3', '_BP86-D3', '_wB97X-D3',
            
            # === DFT FUNCTIONALS WITHOUT DISPERSION ===
            '_HSE06', '_PBE', '_B3LYP', '_PBE0', '_SCAN', '_BLYP', '_BP86', '_wB97X',
            '_LDA', '_VWN', '_PWGGA', '_PW91',
            
            # === HARTREE-FOCK METHODS ===
            '_RHF', '_UHF', '_HF',
            
            # === CALCULATION TYPES ===
            '_OPT', '_SP', '_FREQ', '_BAND', '_DOSS',
            '_opt', '_sp', '_freq', '_band', '_doss',
            '_optimized', '_single_point',
            
            # === DIMENSIONALITY ===
            '_CRYSTAL', '_SLAB', '_POLYMER', '_MOLECULE',
            
            # === SYMMETRY ===
            '_symm', '_P1', '_nosymm',
            
            # === CALCULATION MODES ===
            '_OPTGEOM', '_SCFDIR', '_FREQCALC',
            
            # === BULK/SURFACE DESCRIPTORS ===
            '_BULK', '_SURFACE', '_SLAB',
            
            # === BASIS SET DESCRIPTORS ===
            '_TZ', '_DZ', '_SZ',  # Triple/Double/Single zeta
            
            # === ADDITIONAL DESCRIPTORS ===
            '_CA', '-CA',  # Often used in topology names
        ]
        
        # Apply suffix removal iteratively (keep removing until no more matches)
        # Sort by length (longest first) to avoid partial matches
        sorted_suffixes = sorted(suffixes_to_remove, key=len, reverse=True)
        
        changed = True
        while changed:
            changed = False
            for suffix in sorted_suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    changed = True
                    break  # Start over with the shortened name
                
        # Clean name - DO NOT add mat_ prefix here for cleaner names
        # The enhanced queue manager will handle directory/file naming consistently
        return name
        
    def monitor_workflow_execution(self, workflow_id: str):
        """Monitor and manage workflow execution"""
        print("Phase 3: Monitoring workflow execution...")
        
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        sequence = plan['workflow_sequence']
        
        print(f"  Workflow: {' → '.join(sequence)}")
        print(f"  Monitoring workflow {workflow_id}...")
        print(f"  Use 'python enhanced_queue_manager.py --status' for detailed job status")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self.workflow_monitor_loop,
            args=(workflow_id,),
            daemon=True
        )
        monitor_thread.start()
        
        return monitor_thread
        
    def workflow_monitor_loop(self, workflow_id: str):
        """Main monitoring loop for workflow execution"""
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        sequence = plan['workflow_sequence']
        
        check_interval = 60  # Check every minute
        
        while workflow_id in self.active_workflows:
            try:
                # Check current step status
                current_step = workflow_info["current_step"]
                
                if current_step < len(sequence):
                    calc_type = sequence[current_step]
                    step_key = f"step_{current_step + 1}_{calc_type}"
                    
                    if self.check_step_completion(workflow_id, step_key):
                        # Current step completed, advance to next
                        self.advance_workflow_step(workflow_id)
                        
                else:
                    # All steps completed
                    self.complete_workflow(workflow_id)
                    break
                    
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in workflow monitoring: {e}")
                time.sleep(check_interval)
                
    def check_step_completion(self, workflow_id: str, step_key: str) -> bool:
        """Check if a workflow step is completed"""
        workflow_info = self.active_workflows[workflow_id]
        
        if step_key not in workflow_info["submitted_jobs"]:
            return False
            
        job_ids = workflow_info["submitted_jobs"][step_key]
        
        # Check status of all jobs in this step
        completed_jobs = 0
        failed_jobs = 0
        
        for job_id in job_ids:
            calc = self.db.get_calculation(job_id)
            if calc:
                status = calc.get('status', 'unknown')
                if status == 'completed':
                    completed_jobs += 1
                elif status in ['failed', 'cancelled']:
                    failed_jobs += 1
                    
        total_jobs = len(job_ids)
        
        if completed_jobs == total_jobs:
            print(f"Step {step_key} completed successfully ({completed_jobs}/{total_jobs})")
            return True
        elif failed_jobs > 0:
            print(f"Step {step_key} has failures ({failed_jobs}/{total_jobs} failed)")
            # Could implement error recovery here
            
        return False
        
    def advance_workflow_step(self, workflow_id: str):
        """Advance workflow to the next step"""
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        sequence = plan['workflow_sequence']
        
        current_step = workflow_info["current_step"]
        next_step = current_step + 1
        
        if next_step >= len(sequence):
            self.complete_workflow(workflow_id)
            return
            
        next_calc_type = sequence[next_step]
        print(f"Advancing to step {next_step + 1}: {next_calc_type}")
        
        # Generate input files for next step
        self.generate_next_step_inputs(workflow_id, next_step, next_calc_type)
        
        # Submit next step calculations
        self.submit_next_step_calculations(workflow_id, next_step, next_calc_type)
        
        workflow_info["current_step"] = next_step
        
    def generate_next_step_inputs(self, workflow_id: str, step_num: int, calc_type: str):
        """Generate input files for the next calculation step"""
        print(f"  Generating inputs for step {step_num + 1}: {calc_type}")
        
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        
        # Get configuration for this step
        config_file = self.temp_dir / f"{workflow_id}_step_{step_num + 1:03d}_{calc_type}_config.json"
        
        if not config_file.exists():
            print(f"    Warning: No config file found for step {step_num + 1}")
            return
            
        with open(config_file, 'r') as f:
            step_config = json.load(f)
            
        source = step_config.get('source', 'unknown')
        
        if source == "CRYSTALOptToD12.py":
            self.generate_inputs_with_crystal_opt(workflow_id, step_num, calc_type, step_config)
        elif source in ["create_band_d3.py", "alldos.py"]:
            self.generate_inputs_with_analysis_script(workflow_id, step_num, calc_type, step_config)
        else:
            print(f"    Unknown input generation method: {source}")
            
    def generate_inputs_with_crystal_opt(self, workflow_id: str, step_num: int, 
                                       calc_type: str, config: Dict[str, Any]):
        """Generate inputs using CRYSTALOptToD12.py"""
        print(f"    Using CRYSTALOptToD12.py for {calc_type} inputs")
        
        # Get completed calculations from previous step
        prev_step_key = f"step_{step_num}_{self.active_workflows[workflow_id]['plan']['workflow_sequence'][step_num-1]}"
        prev_job_ids = self.active_workflows[workflow_id]["submitted_jobs"].get(prev_step_key, [])
        
        workflow_dir = self.outputs_dir / workflow_id
        next_step_dir = workflow_dir / f"step_{step_num + 1:03d}_{calc_type}"
        next_step_dir.mkdir(exist_ok=True)
        
        # Generate inputs for each completed calculation
        for job_id in prev_job_ids:
            calc = self.db.get_calculation(job_id)
            if calc and calc.get('status') == 'completed':
                output_file = calc.get('output_file')
                input_file = calc.get('input_file')
                
                if output_file and input_file and Path(output_file).exists():
                    self.run_crystal_opt_conversion(
                        output_file, input_file, next_step_dir, calc_type, config
                    )
                    
    def run_crystal_opt_conversion(self, output_file: str, input_file: str, 
                                 output_dir: Path, calc_type: str, config: Dict[str, Any]):
        """Run CRYSTALOptToD12.py conversion"""
        # Check for local copy first
        local_script_path = self.work_dir / "CRYSTALOptToD12.py"
        if local_script_path.exists():
            script_path = local_script_path
        else:
            script_path = Path(__file__).parent.parent / "Crystal_To_CIF" / "CRYSTALOptToD12.py"
        
        # Create temporary config for CRYSTALOptToD12.py
        temp_config = self.temp_dir / f"temp_crystal_opt_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        crystal_opt_config = {
            "calculation_type": calc_type,
            "keep_current_settings": config.get("inherit_settings", True)
        }
        
        with open(temp_config, 'w') as f:
            json.dump(crystal_opt_config, f)
            
        # Run CRYSTALOptToD12.py
        cmd = [
            sys.executable, str(script_path),
            "--out-file", output_file,
            "--d12-file", input_file,
            "--output-dir", str(output_dir),
            "--options-file", str(temp_config)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"      Generated {calc_type} input from {Path(output_file).name}")
            else:
                print(f"      Failed to generate {calc_type} input: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"      Timeout generating {calc_type} input")
        finally:
            if temp_config.exists():
                temp_config.unlink()
                
    def generate_inputs_with_analysis_script(self, workflow_id: str, step_num: int,
                                            calc_type: str, config: Dict[str, Any]):
        """Generate inputs using analysis scripts (create_band_d3.py, alldos.py)"""
        print(f"    Using {config['source']} for {calc_type} inputs")
        
        # These scripts typically run in isolated directories with specific file requirements
        # Implementation would depend on the specific script requirements
        pass
        
    def submit_next_step_calculations(self, workflow_id: str, step_num: int, calc_type: str):
        """Submit calculations for the next step"""
        workflow_dir = self.outputs_dir / workflow_id
        step_dir = workflow_dir / f"step_{step_num + 1:03d}_{calc_type}"
        
        if not step_dir.exists():
            print(f"    No step directory found: {step_dir}")
            return
            
        # Find input files for this step
        input_files = list(step_dir.glob("*.d12"))
        
        if not input_files:
            print(f"    No input files found in {step_dir}")
            return
            
        print(f"    Submitting {len(input_files)} {calc_type} calculations")
        
        submitted_jobs = []
        for d12_file in input_files:
            job_id = self.submit_single_calculation(d12_file, calc_type, workflow_id, step_num + 1)
            if job_id:
                submitted_jobs.append(job_id)
                
        step_key = f"step_{step_num + 1}_{calc_type}"
        self.active_workflows[workflow_id]["submitted_jobs"][step_key] = submitted_jobs
        self.active_workflows[workflow_id]["step_status"][step_key] = "submitted"
        
    def complete_workflow(self, workflow_id: str):
        """Complete workflow execution"""
        print(f"Workflow {workflow_id} completed!")
        
        workflow_info = self.active_workflows[workflow_id]
        
        # Generate completion report
        self.generate_workflow_report(workflow_id)
        
        # Clean up
        del self.active_workflows[workflow_id]
        
    def generate_workflow_report(self, workflow_id: str):
        """Generate a completion report for the workflow"""
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        
        report_file = self.outputs_dir / workflow_id / "workflow_report.json"
        
        report = {
            "workflow_id": workflow_id,
            "completion_time": datetime.now().isoformat(),
            "start_time": workflow_info["start_time"].isoformat(),
            "sequence": plan['workflow_sequence'],
            "step_status": workflow_info["step_status"],
            "submitted_jobs": workflow_info["submitted_jobs"],
            "total_steps": workflow_info["total_steps"]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Workflow report saved: {report_file}")
        
    def cleanup_workflow(self, workflow_id: str):
        """Clean up workflow resources"""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            
        # Clean up temporary files
        temp_files = self.temp_dir.glob(f"{workflow_id}_*")
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except:
                pass


def main():
    """Main entry point for workflow executor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL Workflow Executor")
    parser.add_argument("plan_file", help="Workflow plan file to execute")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    plan_file = Path(args.plan_file)
    if not plan_file.exists():
        print(f"Plan file not found: {plan_file}")
        sys.exit(1)
        
    executor = WorkflowExecutor(args.work_dir, args.db_path)
    executor.execute_workflow_plan(plan_file)


if __name__ == "__main__":
    main()