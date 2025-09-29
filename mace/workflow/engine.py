#!/usr/bin/env python3
"""
CRYSTAL Workflow Engine
-----------------------
Orchestrates the complete CRYSTAL calculation workflow while maintaining
material tracking and handling the real file naming conventions and
directory requirements of existing scripts.

Key Features:
- Maintains material ID consistency across complex file naming
- Creates isolated directories for each calculation step
- Handles the directory requirements of D3 calculations
- Integrates with CRYSTALOptToD12.py for geometry extraction
- Preserves all existing script behavior and file naming

Author: Based on implementation plan for material tracking system
"""

import os
import sys
import subprocess
import shutil
import tempfile
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import threading

# Define which calculations are optional (can fail without blocking workflow)
# These calculations can fail without preventing the workflow from continuing
OPTIONAL_CALC_TYPES = {'BAND', 'DOSS', 'FREQ', 'TRANSPORT', 'CHARGE+POTENTIAL'}

# Import MACE components
from mace.database.materials import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12
from mace.database.materials_contextual import ContextualMaterialDatabase
from mace.workflow.context import get_current_context
from mace.utils.settings_extractor import extract_input_settings


class WorkflowEngine:
    """
    Orchestrates CRYSTAL calculation workflows with material tracking.
    
    Handles the complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
    while maintaining material ID consistency and directory isolation for
    scripts like CRYSTALOptToD3.py.
    """
    
    def __init__(self, db_path: str = "materials.db", base_work_dir: str = None, auto_submit: bool = True):
        # Check for active workflow context
        ctx = get_current_context()
        if ctx:
            # Use contextual database which automatically uses workflow-specific paths
            self.db = ContextualMaterialDatabase(db_path=db_path if db_path != "materials.db" else None)
            print(f"Using workflow context: {ctx.workflow_id}")
        else:
            # Initialize traditional database connection
            self.db = MaterialDatabase(db_path)
        self.base_work_dir = Path(base_work_dir or os.getcwd())
        self.script_paths = self.get_script_paths()
        self.lock = threading.RLock()
        self.auto_submit = auto_submit  # Enable automatic submission by default
        
        # Create workflow working directories
        self.workflow_dir = self.base_work_dir / "workflow_staging"
        self.workflow_dir.mkdir(exist_ok=True)
        
        # Clean up old workflow staging directories (older than 7 days)
        self._cleanup_old_workflow_dirs()
        
    def get_workflow_sequence(self, workflow_id: str) -> Optional[List[str]]:
        """Get the planned workflow sequence for a workflow ID"""
        if not workflow_id:
            return None
            
        # Search multiple locations for workflow_configs directory
        search_paths = [
            self.base_work_dir / "workflow_configs",
            Path.cwd() / "workflow_configs",
            Path.cwd().parent / "workflow_configs",
            Path.cwd().parent.parent / "workflow_configs",
            Path.cwd().parent.parent.parent / "workflow_configs",
            Path.cwd().parent.parent.parent.parent / "workflow_configs"
        ]
        
        workflow_plan_file = None
        for search_dir in search_paths:
            if search_dir.exists():
                candidate = search_dir / f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"
                if candidate.exists():
                    workflow_plan_file = candidate
                    break
        
        if not workflow_plan_file:
            print(f"DEBUG: Could not find workflow plan file for {workflow_id}")
            print(f"DEBUG: Searched in: {[str(p) for p in search_paths if p.exists()]}")
            return None
            
        try:
            with open(workflow_plan_file, 'r') as f:
                plan = json.load(f)
                sequence = plan.get('workflow_sequence', [])
                print(f"DEBUG: Found workflow sequence: {sequence}")
                return sequence
        except Exception as e:
            print(f"DEBUG: Error reading workflow plan: {e}")
            return None
    
    def get_workflow_step_number(self, workflow_id: str, calc_type: str) -> int:
        """Get the correct step number for a calculation type in the workflow"""
        workflow_sequence = self.get_workflow_sequence(workflow_id)
        if not workflow_sequence:
            # Fallback to hardcoded values if no workflow plan
            default_steps = {"OPT": 1, "SP": 2, "BAND": 3, "DOSS": 4, "FREQ": 5}
            base_type = calc_type.rstrip('0123456789')
            return default_steps.get(base_type, 1)
            
        # Find the step number in the workflow sequence
        for i, step in enumerate(workflow_sequence, 1):
            if step == calc_type:
                return i
                
        # If not found, return next available step
        return len(workflow_sequence) + 1
    
    def get_workflow_step_config(self, workflow_id: str, calc_type: str) -> Optional[Dict[str, Any]]:
        """Get workflow step configuration for a specific calculation type"""
        if not workflow_id:
            return None
            
        # Search multiple locations for workflow_configs directory
        search_paths = [
            self.base_work_dir / "workflow_configs",
            Path.cwd() / "workflow_configs",
            Path.cwd().parent / "workflow_configs",
            Path.cwd().parent.parent / "workflow_configs",
            Path.cwd().parent.parent.parent / "workflow_configs",
            Path.cwd().parent.parent.parent.parent / "workflow_configs"
        ]
        
        workflow_plan_file = None
        for search_dir in search_paths:
            if search_dir.exists():
                candidate = search_dir / f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"
                if candidate.exists():
                    workflow_plan_file = candidate
                    break
        
        if not workflow_plan_file:
            return None
            
        try:
            with open(workflow_plan_file, 'r') as f:
                plan = json.load(f)
                step_configs = plan.get('step_configurations', {})
                
                # Find configuration for this calc type
                for step_key, config in step_configs.items():
                    if calc_type in step_key:
                        return config
                        
                return None
        except Exception:
            return None
    
    def retry_failed_calculation(self, calc_id: str, max_retries: int = 3) -> Optional[str]:
        """
        Retry a failed calculation in the same directory.
        
        Args:
            calc_id: ID of the failed calculation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Job ID if resubmitted successfully, None otherwise
        """
        calc = self.db.get_calculation(calc_id)
        if not calc:
            print(f"Calculation {calc_id} not found")
            return None
            
        # Check retry count
        retry_count = calc.get('retry_count', 0)
        if retry_count >= max_retries:
            print(f"Max retries ({max_retries}) reached for {calc_id}")
            return None
            
        calc_type = calc['calc_type']
        work_dir = Path(calc['work_dir'])
        
        # Apply error recovery fixes if available
        if hasattr(self, 'error_recovery') and self.error_recovery:
            try:
                fixed = self.error_recovery.fix_calculation_errors(calc_id)
                if not fixed:
                    print(f"Could not fix errors for {calc_id}")
                    return None
            except Exception as e:
                print(f"Error recovery failed: {e}")
                
        # Find SLURM script in the directory
        slurm_scripts = list(work_dir.glob("*.sh"))
        if not slurm_scripts:
            print(f"No SLURM script found in {work_dir}")
            return None
            
        slurm_script = slurm_scripts[0]
        
        # Re-submit the job
        job_id = self._submit_calculation_to_slurm(slurm_script, work_dir)
        if job_id:
            # Update database with new submission
            self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=job_id)
            # Note: update_calculation_retry_count should be added to MaterialDatabase
            # For now, update the settings
            settings = json.loads(calc.get('settings_json', '{}'))
            settings['retry_count'] = retry_count + 1
            self.db.update_calculation_settings(calc_id, settings)
            print(f"Retried {calc_type} calculation as job {job_id} (attempt {retry_count + 1}/{max_retries})")
            return job_id
        else:
            print(f"Failed to resubmit {calc_type} calculation")
            return None
        
    def _cleanup_old_workflow_dirs(self):
        """Clean up old workflow staging directories to prevent accumulation"""
        if not self.workflow_dir.exists():
            return
            
        # Clean up directories older than 7 days
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 60 * 60)
        
        for item in self.workflow_dir.iterdir():
            if item.is_dir():
                # Check directory age
                try:
                    dir_mtime = item.stat().st_mtime
                    if dir_mtime < cutoff_time:
                        # Remove old directory
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"Cleaned up old workflow directory: {item.name}")
                except Exception as e:
                    # Skip if we can't access the directory
                    pass
                    
        # Also clean up failed workflow directories
        self._cleanup_failed_workflow_dirs()
                    
    def _cleanup_failed_workflow_dirs(self):
        """Clean up workflow directories from failed generation attempts"""
        workflow_outputs_dir = self.base_work_dir / "workflow_outputs"
        if not workflow_outputs_dir.exists():
            return
            
        # Look for workflow directories
        for workflow_dir in workflow_outputs_dir.glob("workflow_*"):
            if workflow_dir.is_dir():
                # Check if directory has any successful calculations
                has_output = any(workflow_dir.rglob("*.out"))
                has_slurm_output = any(workflow_dir.rglob("*.o*"))
                
                # Check age - clean up failed dirs after 1 day
                try:
                    dir_mtime = workflow_dir.stat().st_mtime
                    age_hours = (datetime.now().timestamp() - dir_mtime) / 3600
                    
                    if not has_output and not has_slurm_output and age_hours > 24:
                        shutil.rmtree(workflow_dir, ignore_errors=True)
                        print(f"Removed failed workflow directory: {workflow_dir.name}")
                except Exception as e:
                    # Skip if we can't access the directory
                    pass
                    
    def get_workflow_id_from_calculation(self, calc_dir: Path) -> Optional[str]:
        """Get workflow ID from calculation directory metadata"""
        metadata_file = calc_dir / '.workflow_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('workflow_id')
            except Exception as e:
                print(f"Error reading workflow metadata: {e}")
        return None
    
    
    def get_script_paths(self) -> Dict[str, Path]:
        """Get paths to all the CRYSTAL workflow scripts."""
        current_dir = Path.cwd()
        # Get the reorganization directory (parent of mace)
        base_path = Path(__file__).parent.parent.parent
        
        # Check for local copies first (in current working directory)
        scripts = {
            'crystal_to_d12': "CRYSTALOptToD12.py",
            'newcif_to_d12': "NewCifToD12.py",
            'crystal_to_d3': "CRYSTALOptToD3.py",
            'd3_interactive': "d3_interactive.py",
            'd3_config': "d3_config.py",
            # Legacy scripts removed - use CRYSTALOptToD3.py instead
            # 'alldos': "alldos.py",  # Deprecated - use CRYSTALOptToD3.py
            # 'create_band': "create_band_d3.py",  # Deprecated - use CRYSTALOptToD3.py
            'd12creation': "d12creation.py"
        }
        
        script_paths = {}
        
        for key, script_name in scripts.items():
            # First check local working directory
            local_path = current_dir / script_name
            if local_path.exists():
                script_paths[key] = local_path
            else:
                # Fall back to repository location
                if key in ['crystal_to_d12', 'newcif_to_d12', 'd12creation']:
                    script_paths[key] = base_path / "Crystal_d12" / script_name
                elif key in ['crystal_to_d3', 'd3_interactive', 'd3_config']:
                    script_paths[key] = base_path / "Crystal_d3" / script_name
        
        return script_paths
    
    def _create_slurm_script_for_calculation(self, calc_dir: Path, material_name: str, 
                                           calc_type: str, step_num: int, workflow_id: str) -> Path:
        """Create SLURM script for a calculation"""
        import os
        
        # Find appropriate template script
        # Use the base_work_dir from the workflow engine, not current directory
        base_dir = self.base_work_dir
        workflow_scripts_dir = base_dir / "workflow_scripts"
        
        # Look for templates with flexible numbering
        template_script = None
        base_type, type_num = self._parse_calc_type(calc_type)
        
        if base_type == "SP":
            # Try numbered templates first (for SP2, SP3, etc.)
            if type_num > 1:
                # Try specific numbered templates
                numbered_patterns = [
                    f"submitcrystal23_sp{type_num}_*.sh",
                    f"submitcrystal23_sp{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general SP templates
            if not template_script:
                for name in ["submitcrystal23_sp_*.sh", "submitcrystal23_sp.sh"]:
                    candidates = list(workflow_scripts_dir.glob(name))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "FREQ":
            # Try numbered templates first (for FREQ2, FREQ3, etc.)
            if type_num > 1:
                # Try specific numbered templates
                numbered_patterns = [
                    f"submitcrystal23_freq{type_num}_*.sh",
                    f"submitcrystal23_freq{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general FREQ templates
            if not template_script:
                for pattern in ["submitcrystal23_freq_*.sh", "submitcrystal23_freq.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "BAND":
            # Try numbered templates first (for BAND2, BAND3, etc.)
            if type_num > 1:
                numbered_patterns = [
                    f"submit_prop_band{type_num}_*.sh",
                    f"submit_prop_band{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general BAND templates
            if not template_script:
                for pattern in ["submit_prop_band_*.sh", "submit_prop_band.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "DOSS":
            # Try numbered templates first (for DOSS2, DOSS3, etc.)
            if type_num > 1:
                numbered_patterns = [
                    f"submit_prop_doss{type_num}_*.sh",
                    f"submit_prop_doss{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general DOSS templates
            if not template_script:
                for pattern in ["submit_prop_doss_*.sh", "submit_prop_doss.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        else:
            # For other calculation types (OPT, OPTn, etc.)
            if base_type == "OPT":
                # Try numbered templates first (for OPT2, OPT3, etc.)
                if type_num > 1:
                    # Try specific numbered templates
                    numbered_patterns = [
                        f"submitcrystal23_opt{type_num}_*.sh",
                        f"submitcrystal23_opt{type_num}.sh"
                    ]
                    for pattern in numbered_patterns:
                        candidates = list(workflow_scripts_dir.glob(pattern))
                        if candidates:
                            template_script = candidates[0]
                            break
                    
                # If no specific template found, try general OPT templates
                if not template_script:
                    for pattern in ["submitcrystal23_opt_*.sh", "submitcrystal23_opt.sh", "submitcrystal23.sh"]:
                        candidates = list(workflow_scripts_dir.glob(pattern))
                        if candidates:
                            template_script = candidates[0]
                            break
        
        # If still no template found, use the first available one that matches the pattern
        if not template_script:
            if calc_type in ["OPT", "SP", "FREQ"]:
                pattern = f"submitcrystal23_*{calc_type.lower()}*.sh"
            else:
                pattern = f"submit_prop_*{calc_type.lower()}*.sh"
            
            templates = list(workflow_scripts_dir.glob(pattern))
            if templates:
                template_script = templates[0]
        
        # Debug: Print template path being used
        print(f"  Using template: {template_script}")
        if template_script:
            print(f"  Template exists: {template_script.exists()}")
        
        if not template_script or not template_script.exists():
            # Fall back to basic template
            if calc_type in ["OPT", "SP", "FREQ"]:
                template_script = base_dir / "submitcrystal23.sh"
                print(f"  Fallback template: {template_script}")
                print(f"  Fallback exists: {template_script.exists()}")
            else:
                template_script = base_dir / "submit_prop.sh"
        
        if not template_script.exists():
            raise FileNotFoundError(f"No SLURM template found for {calc_type}")
        
        # Read template content
        with open(template_script, 'r') as f:
            template_content = f.read()
            
        # Check if this is an old-style script generator (should be avoided)
        if 'echo \'#!/bin/bash --login\' >' in template_content:
            print(f"  WARNING: Template {template_script} appears to be a script generator, not a direct SLURM script")
            print(f"  This will cause submission issues. Consider using a direct SLURM template.")
        
        print(f"  Template content starts with: {template_content[:100]}...")
        
        # Create individual script for this material  
        # Use clean naming without mat_ prefix throughout
        material_script_name = f"{material_name}.sh"
        script_path = calc_dir / material_script_name
        
        # Customize script content using material name for file references
        customized_content = self._customize_slurm_script(
            template_content, material_name, calc_type, workflow_id, step_num
        )
        
        # Write script
        try:
            with open(script_path, 'w') as f:
                f.write(customized_content)
            
            # Make executable
            script_path.chmod(0o755)
            
            print(f"  Successfully created script: {script_path}")
            print(f"  Script exists: {script_path.exists()}")
            
            return script_path
        except Exception as e:
            print(f"  Error creating script {script_path}: {e}")
            raise
    
    def _customize_slurm_script(self, template_content: str, material_name: str, 
                              calc_type: str, workflow_id: str, step_num: int) -> str:
        """Customize SLURM script template for specific calculation"""
        import re
        
        customized = template_content
        
        # Replace $1 placeholders with actual material name
        customized = customized.replace("$1", material_name)
        
        # Fix any incorrect MACE_CONTEXT_DIR settings from templates
        # Remove lines setting MACE_CONTEXT_DIR to SLURM_SUBMIT_DIR
        customized = re.sub(
            r'^export MACE_CONTEXT_DIR="\$\{SLURM_SUBMIT_DIR\}/\.mace_context_[^"]+"\s*$',
            '',
            customized,
            flags=re.MULTILINE
        )
        
        # Add workflow context environment variables with absolute paths
        # Check if we already have MACE_WORKFLOW_ID exported
        if 'export MACE_WORKFLOW_ID=' not in customized:
            # Find where to insert - after export JOB= line
            lines = customized.split('\n')
            insert_index = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('export JOB='):
                    insert_index = i + 1
                    break
                    
            if insert_index > 0:
                # Determine the absolute context directory path
                # Get the workflow root directory
                ctx = get_current_context()
                if ctx:
                    context_dir_path = str(ctx.context_dir)
                else:
                    # Fallback - try to determine from current structure
                    # We're in workflow_outputs/workflow_ID/step_XXX/material/
                    # Need to go up to find the base directory
                    current = Path.cwd()
                    for _ in range(10):
                        if current.name.startswith('workflow_') and current.parent.name == 'workflow_outputs':
                            # Found the workflow directory
                            base_dir = current.parent.parent
                            context_dir_path = str(base_dir / f'.mace_context_{workflow_id}')
                            break
                        current = current.parent
                    else:
                        # Last resort - use self.base_work_dir
                        context_dir_path = str(self.base_work_dir / f'.mace_context_{workflow_id}')
                
                context_exports = f'''# Workflow context for queue manager
export MACE_WORKFLOW_ID="{workflow_id}"
export MACE_CONTEXT_DIR="{context_dir_path}"
export MACE_ISOLATION_MODE="isolated"'''
                
                lines.insert(insert_index, context_exports)
                customized = '\n'.join(lines)
        
        # Update scratch directory to be workflow-specific
        if "export scratch=" in customized:
            scratch_dir = f"$SCRATCH/{workflow_id}/step_{step_num:03d}_{calc_type}/{material_name}"
            customized = re.sub(
                r'export scratch=\$SCRATCH/[\w/]+',
                f'export scratch={scratch_dir}',
                customized
            )
        
        # Enhance queue manager path resolution to handle workflow directory structure
        if "# ADDED: Auto-submit new jobs when this one completes" in customized:
            # Replace the existing queue manager logic with enhanced path resolution
            queue_manager_logic = '''
# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers
cd $DIR

# First check if MACE_HOME is set and use it
if [ ! -z "$MACE_HOME" ]; then
    if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
    elif [ -f "$MACE_HOME/enhanced_queue_manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/enhanced_queue_manager.py"
    fi
else
    # MACE_HOME not set, try to find in PATH or relative locations
    # Try using which to find mace_cli (which we know works)
    MACE_CLI=$(which mace_cli 2>/dev/null)
    if [ ! -z "$MACE_CLI" ]; then
        # Found mace_cli, derive MACE_HOME from it
        MACE_HOME=$(dirname "$MACE_CLI")
        if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
            QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
        fi
    fi
fi

# If still not found, check standard relative locations
if [ -z "$QUEUE_MANAGER" ]; then
    if [ -f $DIR/mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/mace/queue/manager.py"
    elif [ -f $DIR/../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../mace/queue/manager.py"
    elif [ -f $DIR/../../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../../mace/queue/manager.py"
    elif [ -f $DIR/enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
    elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
    elif [ -f $DIR/crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/crystal_queue_manager.py"
    elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../crystal_queue_manager.py"
    fi
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    echo "Found queue manager at: $QUEUE_MANAGER"
    # Use the workflow context database if available
    if [ ! -z "$MACE_CONTEXT_DIR" ] && [ -f "$MACE_CONTEXT_DIR/materials.db" ]; then
        echo "Using workflow context database: $MACE_CONTEXT_DIR/materials.db"
        python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3 --db-path "$MACE_CONTEXT_DIR/materials.db"
    else
        python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
    fi
else
    echo "Warning: Queue manager not found. Checked:"
    echo "  - \\$MACE_HOME/mace/queue/manager.py"
    echo "  - Various relative paths from $DIR"
    echo "  Workflow progression may not continue automatically"
fi'''
            
            # Replace the entire queue manager section
            import re
            # First try to find and replace the complete queue manager section
            # Look for the pattern that includes potentially duplicated logic
            pattern = r'# ADDED: Auto-submit new jobs when this one completes.*?(?:fi\s*){1,2}'
            if re.search(pattern, customized, re.DOTALL):
                customized = re.sub(
                    pattern,
                    queue_manager_logic.strip(),
                    customized,
                    flags=re.DOTALL
                )
            else:
                # If not found, append it
                customized = customized.rstrip() + '\n\n' + queue_manager_logic
        
        # Ensure memory reporting is correct
        customized = self._fix_memory_reporting(customized)
        
        return customized
    
    def _fix_memory_reporting(self, script_content: str) -> str:
        """
        Fix memory reporting to handle both --mem and --mem-per-cpu formats.
        Ensure consistent reporting of expected memory usage.
        """
        import re
        
        # Check which memory format is used
        mem_per_cpu_match = re.search(r'#SBATCH\s+--mem-per-cpu[=\s]+(\d+)([GMK]?)B?', script_content)
        mem_total_match = re.search(r'#SBATCH\s+--mem[=\s]+(\d+)([GMK]?)B?', script_content)
        
        if mem_per_cpu_match:
            # Using per-CPU memory
            value = int(mem_per_cpu_match.group(1))
            unit = mem_per_cpu_match.group(2) or 'G'
            
            # Find number of CPUs/tasks
            ntasks_match = re.search(r'#SBATCH\s+--ntasks[=\s]+(\d+)', script_content)
            cpus_match = re.search(r'#SBATCH\s+--cpus-per-task[=\s]+(\d+)', script_content)
            
            num_cpus = 1
            if ntasks_match:
                num_cpus = int(ntasks_match.group(1))
            elif cpus_match:
                num_cpus = int(cpus_match.group(1))
                
            # Calculate total memory
            total_gb = self._convert_to_gb(value * num_cpus, unit)
            
            # Add comment with total memory calculation
            comment = f"\n# Total memory: {value}{unit} per CPU × {num_cpus} CPUs = {total_gb:.1f}GB total\n"
            script_content = re.sub(
                r'(#SBATCH\s+--mem-per-cpu[=\s]+\d+[GMK]?B?)',
                r'\1' + comment,
                script_content
            )
            
        elif mem_total_match:
            # Using total memory - add clarifying comment
            value = int(mem_total_match.group(1))
            unit = mem_total_match.group(2) or 'G'
            total_gb = self._convert_to_gb(value, unit)
            
            comment = f"\n# Total memory: {total_gb:.1f}GB\n"
            script_content = re.sub(
                r'(#SBATCH\s+--mem[=\s]+\d+[GMK]?B?)',
                r'\1' + comment,
                script_content
            )
            
        return script_content
    
    def _convert_to_gb(self, value: int, unit: str) -> float:
        """Convert memory value to GB"""
        if unit == 'M':
            return value / 1024
        elif unit == 'K':
            return value / (1024 * 1024)
        else:  # G or no unit
            return float(value)
    
    def _submit_calculation_to_slurm(self, script_path: Path, work_dir: Path) -> Optional[str]:
        """Submit calculation to SLURM and return job ID"""
        import subprocess
        import re
        import os
        
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Extract job name from the script path (using clean naming throughout)
            job_name = script_path.stem
            
            # Use the script file name relative to work_dir since we changed to work_dir
            script_filename = script_path.name
            
            # Check if the script contains script generation logic
            print(f"  Reading script file: {script_filename} (from work_dir: {work_dir})")
            with open(script_filename, 'r') as f:
                script_content = f.read()
            
            if 'echo \'#!/bin/bash --login\' >' in script_content or 'echo "#SBATCH' in script_content:
                # This is a script generator template - run locally to generate actual script
                print(f"  Running script generator locally: {script_filename} with job name: {job_name}")
                result = subprocess.run(
                    ['bash', script_filename, job_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output (the template runs sbatch at the end)
                    output = result.stdout.strip()
                    job_id_match = re.search(r'Submitted batch job (\d+)', output)
                    if job_id_match:
                        return job_id_match.group(1)
                    
                    # Maybe the template just generated the script but didn't submit it
                    # Look for generated script and submit it manually
                    generated_script = work_dir / f"{full_job_name}.sh"
                    if generated_script.exists():
                        print(f"  Found generated script: {generated_script.name}")
                        result = subprocess.run(
                            ['sbatch', generated_script.name],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
                            if job_id_match:
                                return job_id_match.group(1)
                else:
                    print(f"Error running script generator: {result.stderr}")
                    
            else:
                # This is a regular SLURM script - submit directly
                print(f"  Submitting SLURM script directly: {script_path.name}")
                result = subprocess.run(
                    ['sbatch', str(script_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output
                    output = result.stdout.strip()
                    job_id_match = re.search(r'Submitted batch job (\d+)', output)
                    if job_id_match:
                        return job_id_match.group(1)
                    else:
                        print(f"Could not extract job ID from: {output}")
                        return None
                else:
                    print(f"Error submitting job: {result.stderr}")
                    return None
                
        finally:
            os.chdir(original_cwd)
        
    def extract_core_material_id_from_complex_filename(self, filename: str) -> str:
        """
        Extract core material ID from complex CRYSTAL filenames.
        
        Handles the real naming patterns from NewCifToD12.py:
        'material_CRYSTAL_OPTGEOM_symm_PBE-D3_POB-TZVP-REV2.d12'
        'material_CRYSTAL_SCFDIR_P1_HSE06_POB-TZVP-REV2.d12'
        
        Returns the core material identifier that remains consistent.
        """
        # Use the smart suffix removal function - no additional cleanup needed
        return create_material_id_from_file(filename)
        
    def get_material_id_from_any_file(self, file_path: Path) -> str:
        """Get material ID from any calculation file, handling complex naming."""
        # First try the database to see if this file is already tracked
        existing_calcs = self.db.get_all_calculations()
        
        file_str = str(file_path)
        for calc in existing_calcs:
            if (calc.get('input_file') == file_str or 
                calc.get('output_file') == file_str or
                file_path.stem in calc.get('input_file', '')):
                return calc['material_id']
        
        # If not found, extract from filename
        return self.extract_core_material_id_from_complex_filename(file_path.name)
        
    def ensure_material_exists(self, file_path: Path, source_type: str = "unknown") -> str:
        """Ensure material exists in database, create if needed."""
        material_id = self.get_material_id_from_any_file(file_path)
        
        # Check if material already exists
        existing_material = self.db.get_material(material_id)
        if existing_material:
            return material_id
            
        # Create new material
        if file_path.suffix == '.d12':
            formula = extract_formula_from_d12(str(file_path))
        else:
            formula = "Unknown"
            
        self.db.create_material(
            material_id=material_id,
            formula=formula,
            source_type=source_type,
            source_file=str(file_path),
            metadata={
                'workflow_created': True,
                'original_filename': file_path.name,
                'creation_timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"Created material record: {material_id}")
        return material_id
        
    def create_isolated_calculation_directory(self, material_id: str, calc_type: str, 
                                            source_files: List[Path]) -> Path:
        """
        Create isolated directory for calculation with only required files.
        
        This is critical for scripts like CRYSTALOptToD3.py which may
        expect to run in a clean directory with only the relevant files.
        """
        # Create unique directory name with timestamp and UUID to prevent collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        calc_dir_name = f"{material_id}_{calc_type}_{timestamp}_{unique_id}"
        calc_dir = self.workflow_dir / calc_dir_name
        calc_dir.mkdir(exist_ok=True)
        
        # Copy only the required files
        copied_files = []
        for source_file in source_files:
            if source_file.exists():
                dest_file = calc_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                copied_files.append(dest_file)
                print(f"Copied {source_file.name} to isolated directory")
            else:
                print(f"Warning: Source file not found: {source_file}")
                
        return calc_dir
        
    def run_script_in_isolated_directory(self, script_path: Path, work_dir: Path, 
                                       args: List[str] = None, input_data: str = None) -> Tuple[bool, str, str]:
        """
        Run a script in an isolated directory and capture output.
        
        Args:
            script_path: Path to the script to run
            work_dir: Directory to run the script in
            args: Additional command line arguments
            input_data: Input data to provide to the script (for interactive scripts)
            
        Returns:
            (success, stdout, stderr)
        """
        script_path = Path(script_path)
        if not script_path.exists():
            return False, "", f"Script not found: {script_path}"
            
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
            
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Script execution timed out"
        except Exception as e:
            return False, "", f"Error running script: {e}"
    
    def find_workflow_context(self, input_file_path: str) -> Optional[Tuple[str, str]]:
        """
        Find the workflow ID and step from a file path.
        
        Args:
            input_file_path: Path to an input file
            
        Returns:
            Tuple of (workflow_id, step_dir) if found, None otherwise
        """
        path = Path(input_file_path)
        
        # Look for workflow_outputs/workflow_YYYYMMDD_HHMMSS/step_XXX_TYPE pattern
        for parent in path.parents:
            if parent.name.startswith("workflow_") and len(parent.name) > 15:
                # Check if parent's parent is workflow_outputs
                if parent.parent.name == "workflow_outputs":
                    workflow_id = parent.name
                    # Find the step directory
                    for part in path.parts:
                        if part.startswith("step_") and "_" in part[5:]:
                            return workflow_id, part
        return None
    
    def get_workflow_output_base(self, opt_calc: Dict) -> Path:
        """
        Get the base workflow output directory for placing new calculations.
        
        Args:
            opt_calc: OPT calculation record
            
        Returns:
            Path to workflow outputs directory
        """
        # First try to get workflow_id from calculation settings
        settings = json.loads(opt_calc.get('settings_json', '{}'))
        workflow_id = settings.get('workflow_id')
        
        if workflow_id:
            return self.base_work_dir / "workflow_outputs" / workflow_id
        
        # Try to find from file path context
        opt_input_file = opt_calc.get('input_file', '')
        workflow_context = self.find_workflow_context(opt_input_file)
        
        if workflow_context:
            workflow_id, _ = workflow_context
            return self.base_work_dir / "workflow_outputs" / workflow_id
        
        # Try to find most recent workflow directory for this material
        material_id = opt_calc.get('material_id')
        if material_id and (self.base_work_dir / "workflow_outputs").exists():
            # Look for workflows containing this material
            workflow_dirs = sorted(
                [d for d in (self.base_work_dir / "workflow_outputs").iterdir() 
                 if d.is_dir() and d.name.startswith("workflow_")],
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            
            for wf_dir in workflow_dirs:
                # Check if this workflow contains our material
                for step_dir in wf_dir.iterdir():
                    if step_dir.is_dir() and step_dir.name.startswith("step_"):
                        for mat_dir in step_dir.iterdir():
                            if mat_dir.is_dir() and self.extract_core_material_name(material_id) in mat_dir.name:
                                print(f"DEBUG: Found existing workflow {wf_dir.name} for material {material_id}")
                                return wf_dir
        
        # Last resort - create new workflow (this should rarely happen)
        print(f"WARNING: Creating new workflow directory - could not find existing workflow context")
        print(f"  opt_calc: {opt_calc}")
        from datetime import datetime
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.base_work_dir / "workflow_outputs" / workflow_id
    
    def extract_core_material_name(self, material_id: str) -> str:
        """Extract the core material name with proper handling of numbered materials"""
        # Handle both full filenames and stems
        from pathlib import Path
        if material_id.endswith('.d12') or material_id.endswith('.d3'):
            name = Path(material_id).stem
        else:
            name = material_id
        
        # Strategy: Remove known calculation suffixes from the end
        # IMPORTANT: Check numbered suffixes BEFORE base suffixes to avoid 
        # stripping numbers that are part of the material name (e.g., test2_sp)
        
        # Generate suffixes dynamically for any number (up to 99 seems reasonable)
        calc_suffixes = []
        base_types = ['opt', 'sp', 'freq', 'band', 'doss']
        
        # Add numbered suffixes in reverse order (higher numbers first)
        for num in range(99, 1, -1):
            for base in base_types:
                calc_suffixes.append(f'_{base}{num}')
        
        # Then add base suffixes
        for base in base_types:
            calc_suffixes.append(f'_{base}')
        
        # Remove calc suffix if present
        clean_name = name
        for suffix in calc_suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # If name still contains technical keywords, extract the core
        if any(keyword in clean_name.upper() for keyword in ['BULK', 'OPTGEOM', 'CRYSTAL', 'PBE', 'B3LYP']):
            # Use the original logic for complex names
            parts = clean_name.split('_')
            core_parts = []
            
            for part in parts:
                # Stop at technical keywords
                if part.upper() in ['BULK', 'OPTGEOM', 'CRYSTAL', 'SLAB', 'POLYMER', 
                                   'MOLECULE', 'SYMM', 'TZ', 'DZ', 'SZ', 'PBE', 'B3LYP', 
                                   'HSE06', 'PBE0', 'SCAN', 'BLYP', 'BP86']:
                    break
                elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():
                    break
                elif 'D3' in part.upper():
                    break
                else:
                    core_parts.append(part)
                    
            if core_parts:
                clean_name = '_'.join(core_parts)
        
        return clean_name
        # Only replace truly problematic characters for filesystem compatibility
        # Preserve ^ and , as they are commonly used in chemical notation
        clean_name = clean_name.replace(' ', '_')  # Spaces to underscores
        clean_name = clean_name.replace('/', '_').replace('\\', '_')  # Path separators
        # Keep other characters like ^, ,, ., - as they are common in chemical names
        
        return clean_name
    
    def clean_material_name(self, material_id: str) -> str:
        """
        Clean material name for use in directory names - wrapper for backward compatibility
        """
        return self.extract_core_material_name(material_id)
        
    def get_next_calc_suffix(self, core_name: str, calc_type: str, workflow_base: Path) -> str:
        """
        Get the next available suffix for a calculation type (e.g., _sp, _sp2, _sp3)
        
        Args:
            core_name: Core material name (e.g., "1_dia")
            calc_type: Calculation type ("SP", "OPT", "BAND", "DOSS", "FREQ")
            workflow_base: Base workflow directory to check for existing calculations
            
        Returns:
            Clean suffix like "_sp", "_sp2", "_opt", "_opt2", etc.
        """
        type_suffix = calc_type.lower()
        
        # Check what calculation directories already exist
        existing_dirs = []
        for step_dir in workflow_base.glob(f"step_*_{calc_type}"):
            if step_dir.is_dir():
                for material_dir in step_dir.glob(f"{core_name}*"):
                    existing_dirs.append(material_dir.name)
        
        # Determine the next number
        if not existing_dirs:
            # First calculation of this type
            return f"_{type_suffix}"
        
        # Find highest existing number
        max_num = 0
        for dir_name in existing_dirs:
            # Look for pattern like 1_dia_sp2 or 1_dia_sp  
            if f"_{type_suffix}" in dir_name:
                parts = dir_name.split(f"_{type_suffix}")
                if len(parts) > 1 and parts[1]:
                    # Has a number after the type
                    try:
                        num = int(parts[1])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
                else:
                    # No number means it's the first one
                    max_num = max(max_num, 1)
        
        # Return next available suffix
        next_num = max_num + 1
        if next_num == 1:
            return f"_{type_suffix}"
        else:
            return f"_{type_suffix}{next_num}"
            
    def generate_sp_from_opt(self, opt_calc_id: str) -> Optional[str]:
        """
        Generate SP calculation from completed OPT using CRYSTALOptToD12.py.
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            
        Returns:
            New SP calculation ID if successful, None otherwise
        """
        print(f"Generating SP calculation from OPT {opt_calc_id}")
        
        # Get OPT calculation details
        opt_calc = self.db.get_calculation(opt_calc_id)
        if not opt_calc or opt_calc['status'] != 'completed':
            print(f"OPT calculation {opt_calc_id} not completed")
            return None
            
        material_id = opt_calc['material_id']
        opt_output_file = Path(opt_calc['output_file'])
        opt_input_file = Path(opt_calc['input_file'])
        
        if not opt_output_file.exists():
            print(f"OPT output file not found: {opt_output_file}")
            return None
            
        # Create isolated directory for CRYSTALOptToD12.py
        work_dir = self.create_isolated_calculation_directory(
            material_id, "SP_generation", [opt_output_file, opt_input_file]
        )
        
        try:
            # Run CRYSTALOptToD12.py in isolated directory
            crystal_to_d12_script = self.script_paths['crystal_to_d12']
            
            # Find the specific out and d12 files in the work directory
            out_files = list(work_dir.glob("*.out"))
            d12_files = list(work_dir.glob("*.d12"))
            
            if not out_files:
                print(f"No .out file found in {work_dir}")
                return None
                
            out_file = out_files[0]
            d12_file = d12_files[0] if d12_files else None
            
            # Use single file mode with command-line arguments for non-interactive execution
            # Use just the filename, not the full path, since we'll run in the work directory
            args = [
                "--out-file", out_file.name,
                "--output-dir", ".",
                "--non-interactive",
                "--calc-type", "SP"
            ]
            
            if d12_file:
                args.extend(["--d12-file", d12_file.name])
            
            # Still need input responses even in non-interactive mode
            # The script asks for confirmation and some settings
            input_responses = "n\n2\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"CRYSTALOptToD12.py failed: {stderr}")
                return None
                
            # Debug output
            print(f"CRYSTALOptToD12.py output (first 500 chars): {stdout[:500]}...")
            print(f"Files in work_dir after script: {list(work_dir.glob('*.d12'))}")
                
            # Find generated SP .d12 file
            sp_files = list(work_dir.glob("*SP*.d12"))
            if not sp_files:
                # Try other patterns
                sp_files = list(work_dir.glob("*_sp*.d12")) + list(work_dir.glob("*SCFDIR*.d12"))
                
            if not sp_files:
                print("No SP input file generated by CRYSTALOptToD12.py")
                return None
                
            sp_input_file = sp_files[0]
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(opt_calc)
            
            # Get core material name and determine SP suffix
            core_name = self.extract_core_material_name(material_id)
            sp_suffix = self.get_next_calc_suffix(core_name, "SP", workflow_base)
            
            # Use core name directly (already clean)
            clean_job_name = core_name
            
            # Create material-specific directory for SP calculation
            dir_name = f"{core_name}{sp_suffix}"
            sp_step_dir = workflow_base / "step_002_SP" / dir_name
            sp_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Create workflow metadata file for this calculation
            metadata = {
                'workflow_id': workflow_base.name,
                'step_num': 2,
                'calc_type': 'SP',
                'material_id': material_id
            }
            metadata_file = sp_step_dir / '.workflow_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Move SP file to material's directory with consistent naming
            clean_sp_name = f"{core_name}{sp_suffix}.d12"
            sp_final_location = sp_step_dir / clean_sp_name
            shutil.move(sp_input_file, sp_final_location)
            
            # Create SLURM script for SP calculation  
            # Use the clean material name for the job
            sp_job_name = f"{clean_job_name}{sp_suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                sp_step_dir, sp_job_name, "SP", 2, workflow_base.name
            )
            
            # Create SP calculation record
            # Extract workflow_id from parent calculation if it exists
            workflow_id = None
            workflow_step = None
            if opt_calc.get('settings_json'):
                try:
                    parent_settings = json.loads(opt_calc['settings_json'])
                    workflow_id = parent_settings.get('workflow_id')
                    workflow_step = parent_settings.get('workflow_step')
                except json.JSONDecodeError:
                    pass
            
            # Build settings with workflow_id propagation
            settings = {
                'generated_from_opt': opt_calc_id,
                'generation_method': 'CRYSTALOptToD12.py',
                'workflow_step': True,
                'slurm_script': str(slurm_script_path)
            }
            
            # Add workflow_id if it exists
            if workflow_id:
                settings['workflow_id'] = workflow_id
                if workflow_step is not None:
                    settings['workflow_step'] = workflow_step + 1
            
            sp_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="SP",
                input_file=str(sp_final_location),
                work_dir=str(sp_step_dir),
                settings=settings
            )
            
            # Submit SP calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, sp_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(sp_calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted SP calculation {sp_calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated SP calculation {sp_calc_id} but submission failed: {sp_final_location}")
            
            return sp_calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def generate_doss_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate DOSS calculation from completed SP using CRYSTALOptToD3.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New DOSS calculation ID if successful, None otherwise
        """
        # Use new D3 generation if available, fall back to legacy
        if self._use_new_d3_generation():
            return self.generate_d3_calculation_new(sp_calc_id, "DOSS")
        else:
            return self.generate_property_calculation(sp_calc_id, "DOSS")
            
    def generate_band_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate BAND calculation from completed SP using CRYSTALOptToD3.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New BAND calculation ID if successful, None otherwise
        """
        # Use new D3 generation if available, fall back to legacy
        if self._use_new_d3_generation():
            return self.generate_d3_calculation_new(sp_calc_id, "BAND")
        else:
            return self.generate_property_calculation(sp_calc_id, "BAND")
    
    def generate_transport_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """Generate TRANSPORT calculation from completed SP"""
        if self._use_new_d3_generation():
            return self.generate_d3_calculation_new(sp_calc_id, "TRANSPORT")
        else:
            print("TRANSPORT calculations require new CRYSTALOptToD3.py")
            return None
    
    def generate_charge_potential_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """Generate CHARGE+POTENTIAL calculation from completed SP"""
        if self._use_new_d3_generation():
            return self.generate_d3_calculation_new(sp_calc_id, "CHARGE+POTENTIAL")
        else:
            print("CHARGE+POTENTIAL calculations require new CRYSTALOptToD3.py")
            return None
    
    def _use_new_d3_generation(self) -> bool:
        """Check if we should use new CRYSTALOptToD3.py for D3 generation"""
        # Check if CRYSTALOptToD3.py is available
        crystal_to_d3 = self.script_paths.get('crystal_to_d3')
        if crystal_to_d3 and Path(crystal_to_d3).exists():
            return True
        return False
    
    def generate_d3_calculation_new(self, source_calc_id: str, target_calc_type: str) -> Optional[str]:
        """
        Generate D3 calculation using new CRYSTALOptToD3.py script.
        
        Supports BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL calculations.
        
        Args:
            source_calc_id: ID of source calculation (SP or OPT with wavefunction)
            target_calc_type: Target D3 calculation type
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {target_calc_type} calculation using CRYSTALOptToD3.py")
        
        # Get source calculation details
        source_calc = self.db.get_calculation(source_calc_id)
        if not source_calc:
            print(f"Source calculation {source_calc_id} not found")
            return None
            
        material_id = source_calc['material_id']
        
        # Find the most recent calculation with a wavefunction
        wavefunction_calc_id = self._find_most_recent_wavefunction_calc(material_id)
        if not wavefunction_calc_id:
            print(f"No completed calculation with wavefunction found for material {material_id}")
            return None
            
        wf_calc = self.db.get_calculation(wavefunction_calc_id)
        if not wf_calc or wf_calc['status'] != 'completed':
            print(f"Wavefunction calculation {wavefunction_calc_id} not completed")
            return None
            
        wf_output_file = Path(wf_calc['output_file'])

        # Get workflow base directory and determine proper step location
        workflow_base = self.get_workflow_output_base(source_calc)

        # Find the step number for this calculation type
        step_num = self._get_next_step_number(workflow_base, target_calc_type)

        # Create work directory in proper workflow step location
        core_name = self.extract_core_material_name(material_id)
        calc_suffix = self.get_next_calc_suffix(core_name, target_calc_type, workflow_base)
        dir_name = f"{core_name}{calc_suffix}"

        step_dir = workflow_base / f"step_{step_num:03d}_{target_calc_type}" / dir_name
        work_dir = step_dir / "tmp_d3_generation"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get CRYSTALOptToD3.py script path
            script_path = self.script_paths.get('crystal_to_d3')
            if not script_path or not Path(script_path).exists():
                print("CRYSTALOptToD3.py not found")
                return None
                
            # Create basic D3 configuration based on calc type
            d3_config = self._get_default_d3_config(target_calc_type)
            
            # Save configuration to temp file
            config_file = work_dir / f"{target_calc_type.lower()}_config.json"
            import json
            with open(config_file, 'w') as f:
                json_config = {
                    "version": "1.0",
                    "type": "d3_configuration",
                    "calculation_type": target_calc_type,
                    "configuration": d3_config
                }
                json.dump(json_config, f, indent=2)
            
            # Run CRYSTALOptToD3.py
            cmd = [
                sys.executable, str(script_path),
                "--input", str(wf_output_file),
                "--calc-type", target_calc_type,
                "--output-dir", str(work_dir),
                "--config-file", str(config_file)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"CRYSTALOptToD3.py failed: {result.stderr}")
                return None
                
            # Find generated D3 file
            # With --output-dir, files will be in work_dir
            d3_files = list(work_dir.glob(f"*_{target_calc_type.lower()}.d3"))
            if not d3_files:
                print(f"No D3 file generated in {work_dir}")
                return None
                
            # Get the generated D3 file
            d3_file = d3_files[0]
            
            # Create final directories in proper workflow step location
            base_type, calc_num = self._parse_calc_type(target_calc_type)
            final_dir = step_dir / f"{base_type}{calc_num}"
            final_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files to final location
            final_d3 = final_dir / f"{material_id}_{target_calc_type.lower()}.d3"
            shutil.copy2(d3_file, final_d3)
            
            # Also copy the wavefunction file
            # CRYSTALOptToD3.py creates a wavefunction file with matching name
            wf_file = d3_file.with_suffix('.f9')
            if wf_file.exists():
                final_wf = final_dir / f"{material_id}_{target_calc_type.lower()}.f9"
                shutil.copy2(wf_file, final_wf)
            else:
                print(f"Warning: Wavefunction file not found: {wf_file}")
            
            # Create and submit calculation
            calc_id = self._create_and_submit_d3_calculation(
                material_id, target_calc_type, final_d3, final_dir, wavefunction_calc_id
            )
            
            return calc_id
            
        except Exception as e:
            print(f"Error generating {target_calc_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temp directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _get_default_d3_config(self, calc_type: str) -> Dict[str, Any]:
        """Get default D3 configuration for different calculation types"""
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
                "projection_type": 0,
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
                    "n_points": 100,
                    "scale": 3,
                    "use_range": False
                },
                "potential_config": {
                    "type": "POT3",
                    "n_points": 100,
                    "scale": 3,
                    "use_range": False
                }
            }
        }
        
        # Extract base type for numbered calculations (BAND2, DOSS3, etc)
        base_type, _ = self._parse_calc_type(calc_type)
        return configs.get(base_type, {})
    
    def _create_and_submit_d3_calculation(self, material_id: str, calc_type: str, 
                                         d3_file: Path, work_dir: Path, 
                                         parent_calc_id: str) -> Optional[str]:
        """Create and submit a D3 calculation"""
        # Parse calculation type
        base_type, calc_num = self._parse_calc_type(calc_type)
        
        # Determine step number based on calculation type
        default_steps = {"BAND": 3, "DOSS": 4, "TRANSPORT": 5, "CHARGE+POTENTIAL": 6}
        step_num = default_steps.get(base_type, 3)
        if calc_num > 1:
            step_num += (calc_num - 1) * 10
            
        # Get workflow ID from parent calculation
        parent_calc = self.db.get_calculation(parent_calc_id)
        workflow_id = parent_calc.get('workflow_id', 'manual')
        
        # Generate SLURM script
        slurm_script = self._create_slurm_script_for_calculation(
            work_dir, material_id, calc_type, step_num, workflow_id
        )
        
        if not slurm_script:
            print(f"Failed to create SLURM script for {calc_type}")
            return None
            
        # Create calculation record
        calc_id = self.db.create_calculation(
            material_id=material_id,
            calc_type=calc_type,
            input_file=str(d3_file),
            work_dir=str(work_dir),
            prerequisite_calc_id=parent_calc_id,
            settings={'workflow_id': workflow_id, 'step_number': step_num}
        )
        
        # Submit job
        job_id = self._submit_slurm_job(slurm_script, work_dir)
        if job_id:
            self.db.update_calculation(calc_id, slurm_job_id=job_id, status='submitted')
            print(f"Submitted {calc_type} calculation: Job ID {job_id}, Calc ID {calc_id}")
            return calc_id
        else:
            self.db.update_calculation(calc_id, status='failed')
            return None
        
    def generate_property_calculation(self, source_calc_id: str, target_calc_type: str) -> Optional[str]:
        """
        Generate property calculation (BAND, DOSS) from a source calculation with wavefunction.
        
        For BAND/DOSS, this will use the most recent SP or OPT calculation with a wavefunction.
        
        Args:
            source_calc_id: ID of source calculation (used as hint, may find more recent)
            target_calc_type: Target calculation type (BAND, DOSS, BAND2, DOSS2, etc.)
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {target_calc_type} calculation")
        
        # Validate templates exist before attempting generation
        if not self._validate_property_templates(target_calc_type):
            print(f"ERROR: Required templates for {target_calc_type} generation not found")
            return None
        
        # Parse target calculation type to get base type and number
        base_type, calc_num = self._parse_calc_type(target_calc_type)
        
        # Get source calculation details
        source_calc = self.db.get_calculation(source_calc_id)
        if not source_calc:
            print(f"Source calculation {source_calc_id} not found")
            return None
            
        material_id = source_calc['material_id']
        
        # Find the most recent calculation with a wavefunction (SP or OPT)
        wavefunction_calc_id = self._find_most_recent_wavefunction_calc(material_id)
        if not wavefunction_calc_id:
            print(f"No completed calculation with wavefunction found for material {material_id}")
            return None
            
        # Get the wavefunction calculation details
        wf_calc = self.db.get_calculation(wavefunction_calc_id)
        if not wf_calc or wf_calc['status'] != 'completed':
            print(f"Wavefunction calculation {wavefunction_calc_id} not completed")
            return None
            
        wf_output_file = Path(wf_calc['output_file'])
        wf_input_file = Path(wf_calc['input_file'])
        
        # Find associated .f9 file
        wf_f9_file = wf_output_file.with_suffix('.f9')
        if not wf_f9_file.exists():
            work_dir = Path(wf_calc.get('work_dir', wf_output_file.parent))
            wf_f9_file = work_dir / f"{wf_output_file.stem}.f9"
            
        if not wf_f9_file.exists():
            print(f"Wavefunction .f9 file not found for {wavefunction_calc_id}")
            return None
            
        # Determine which script to use based on target type
        if base_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]:
            # Use new CRYSTALOptToD3.py for all D3 calculation types
            return self.generate_d3_calculation_new(wavefunction_calc_id, target_calc_type)
        else:
            print(f"Unsupported property calculation type: {target_calc_type}")
            return None
            
    def generate_freq_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate FREQ calculation from completed SP using CRYSTALOptToD12.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New FREQ calculation ID if successful, None otherwise
        """
        return self.generate_numbered_calculation(sp_calc_id, "FREQ")
        
    def generate_freq_from_opt(self, opt_calc_id: str, target_calc_type: str = "FREQ") -> Optional[str]:
        """
        Generate FREQ calculation from completed OPT using CRYSTALOptToD12.py.
        Supports numbered variants (FREQ, FREQ2, FREQ3, etc.)
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            target_calc_type: Target calculation type (FREQ, FREQ2, etc.)
            
        Returns:
            New FREQ calculation ID if successful, None otherwise
        """
        # Simply delegate to generate_numbered_calculation which handles all types
        return self.generate_numbered_calculation(opt_calc_id, target_calc_type)
            
    def _check_and_trigger_pending_calculations(self, material_id: str, planned_sequence: List[str]) -> List[str]:
        """
        Check for any calculations in the planned sequence that should have been triggered
        but haven't been yet. This handles cases where workflow steps might have been missed.
        
        Returns:
            List of new calculation IDs created
        """
        new_calc_ids = []
        
        if not planned_sequence:
            return new_calc_ids
            
        # Get all calculations for this material
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Build a map of completed calculations by type
        completed_by_type = {}
        for calc in all_calcs:
            if calc['status'] == 'completed':
                calc_type = calc['calc_type']
                if calc_type not in completed_by_type:
                    completed_by_type[calc_type] = []
                completed_by_type[calc_type].append(calc)
        
        # Check if any pending calculations can now be started
        for planned_type in planned_sequence:
            base_type, type_num = self._parse_calc_type(planned_type)
            
            # Skip if already completed or in progress
            if self._calculation_already_exists(material_id, planned_type):
                existing_status = [c['status'] for c in all_calcs if c['calc_type'] == planned_type]
                print(f"Calculation {planned_type} already exists (status: {existing_status}), skipping...")
                continue
                
            # Check dependencies
            can_start = False
            source_calc_id = None
            
            if base_type == "SP":
                # SP calculations depend on their previous step in the workflow sequence
                # Find what step comes before this SP in the planned sequence
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    can_start = True
                    source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                elif not prev_step:
                    # SP is the first calculation - needs CIF source
                    can_start = True
                    source_calc_id = 'CIF'  # Special marker for CIF generation
                    
            elif base_type == "FREQ":
                # FREQ calculations need an optimized geometry from an OPT calculation
                # Wait for the dependency step to complete, but use the highest numbered OPT
                # that has been COMPLETED up to this point (not future OPTs in the sequence)
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    # Find the highest numbered OPT calculation that's been completed
                    # This correctly handles cases like: OPT → OPT2 → SP → OPT3 → FREQ → OPT4
                    # where FREQ uses OPT3 (not OPT4 which hasn't run yet)
                    opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                    if opt_source:
                        can_start = True
                        source_calc_id = opt_source  # Use highest completed OPT for FREQ generation
                    
            elif base_type in ["BAND", "DOSS"]:
                # BAND/DOSS calculations depend on their previous step in the workflow sequence
                # They typically need a wavefunction from SP or OPT
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    can_start = True
                    source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                    
            elif base_type == "OPT":
                # OPT calculations need appropriate source geometry
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    # Check if previous step can provide geometry
                    prev_base, _ = self._parse_calc_type(prev_step)
                    
                    if prev_base in ['OPT', 'SP']:  # These can provide geometry
                        can_start = True
                        source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                    else:
                        # Previous step can't provide geometry (e.g., FREQ, BAND, DOSS)
                        # Find the highest numbered OPT completed so far
                        opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        if opt_source:
                            can_start = True
                            source_calc_id = opt_source  # Use highest completed OPT
                        elif type_num == 1:
                            # First OPT with no prior OPT - need CIF source
                            can_start = True
                            source_calc_id = 'CIF'  # Special marker for CIF generation
            
            # Trigger the calculation if dependencies are met
            if can_start and source_calc_id:
                print(f"Triggering pending {planned_type} calculation...")
                
                if base_type == "SP":
                    if source_calc_id == 'CIF':
                        # Generate from CIF
                        # Find material_id from context or use planned_type to derive it
                        material_id = None
                        for calcs in completed_by_type.values():
                            if calcs:
                                material_id = calcs[0]['material_id']
                                break
                        # If no completed calcs, we need to get material_id from somewhere else
                        # This would typically come from the workflow context
                        if material_id:
                            calc_id = self.generate_calculation_from_cif(material_id, planned_type)
                        else:
                            print(f"Cannot determine material_id for CIF generation")
                            calc_id = None
                    else:
                        calc_id = self.generate_numbered_calculation(source_calc_id, planned_type)
                elif base_type == "FREQ":
                    # FREQ always uses generate_freq_from_opt with an OPT calculation
                    # source_calc_id should already be from an OPT due to fixed dependency logic
                    calc_id = self.generate_freq_from_opt(source_calc_id, planned_type)
                elif base_type in ["BAND", "DOSS"]:
                    calc_id = self.generate_property_calculation(source_calc_id, planned_type)
                elif base_type == "OPT":
                    if source_calc_id == 'CIF':
                        # Generate from CIF
                        # Find material_id from any completed calculation
                        material_id = None
                        for calcs in completed_by_type.values():
                            if calcs:
                                material_id = calcs[0]['material_id']
                                break
                        if material_id:
                            calc_id = self.generate_calculation_from_cif(material_id, planned_type)
                        else:
                            print(f"Cannot determine material_id for CIF generation")
                            calc_id = None
                    else:
                        calc_id = self.generate_numbered_calculation(source_calc_id, planned_type)
                else:
                    calc_id = None
                    
                if calc_id:
                    new_calc_ids.append(calc_id)
                    
        return new_calc_ids

    def execute_workflow_step(self, material_id: str, completed_calc_id: str) -> List[str]:
        """
        Execute the next workflow step(s) for a material.
        
        Args:
            material_id: Material identifier
            completed_calc_id: ID of just-completed calculation
            
        Returns:
            List of new calculation IDs created
        """
        new_calc_ids = []
        
        # Clean up any failed workflow directories proactively
        self._cleanup_failed_workflow_dirs()
        
        # Get the completed calculation
        completed_calc = self.db.get_calculation(completed_calc_id)
        if not completed_calc:
            print(f"Completed calculation not found: {completed_calc_id}")
            return new_calc_ids
            
        calc_type = completed_calc['calc_type']
        
        print(f"Executing workflow step after {calc_type} completion for {material_id}")
        
        # Determine next steps based on completed calculation type and workflow plan
        # Check if workflow metadata exists to determine planned sequence
        # Look for workflow_id in settings (where it's stored) or metadata
        settings_json = completed_calc.get('settings_json')
        if settings_json:
            try:
                settings = json.loads(settings_json)
            except (json.JSONDecodeError, TypeError):
                settings = {}
        else:
            settings = {}
            
        # Also check for workflow_id from environment when in workflow context
        workflow_id = settings.get('workflow_id') or completed_calc.get('metadata', {}).get('workflow_id')
        if not workflow_id and os.environ.get('MACE_WORKFLOW_ID'):
            workflow_id = os.environ.get('MACE_WORKFLOW_ID')
        
        print(f"DEBUG: Extracted workflow_id: {workflow_id}")
        print(f"DEBUG: Settings: {settings}")
        
        planned_sequence = self.get_workflow_sequence(workflow_id) if workflow_id else None
        print(f"DEBUG: Planned sequence: {planned_sequence}")
        
        # Parse current calculation type to handle numbered types
        base_type, type_num = self._parse_calc_type(calc_type)
        
        if base_type == "OPT":
            # Generate next steps based on workflow plan or default behavior
            if planned_sequence:
                # Find current position in sequence and get next step
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                print(f"DEBUG: Current index in sequence: {current_index}")
                next_steps = self._get_next_steps_from_sequence(current_index, planned_sequence, calc_type)
                print(f"DEBUG: Next steps: {next_steps}")
                
                # Track completed and failed calculations for dependency checking
                all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                completed_calcs = {calc['calc_type'] for calc in all_calcs if calc['status'] == 'completed'}
                failed_generations = set()  # Track failures in this execution
                
                for next_calc_type in next_steps:
                    # Check if calculation already exists (submitted, running, or completed)
                    if self._calculation_already_exists(material_id, next_calc_type):
                        print(f"Calculation {next_calc_type} already exists for {material_id}, skipping...")
                        continue
                    
                    # Check dependencies first
                    deps_met, blocking_calc = self._check_dependencies_met(
                        next_calc_type, material_id, planned_sequence, 
                        completed_calcs, failed_generations
                    )
                    
                    if not deps_met:
                        print(f"Skipping {next_calc_type} - dependency {blocking_calc} not met")
                        continue
                    
                    next_base_type, next_num = self._parse_calc_type(next_calc_type)
                    
                    if next_base_type == "OPT":
                        # Generate another optimization (OPT2, OPT3, etc.)
                        opt_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                        if opt_calc_id:
                            new_calc_ids.append(opt_calc_id)
                        else:
                            failed_generations.add(next_calc_type)
                            # OPT is usually critical
                            print(f"CRITICAL: Failed to generate {next_calc_type}")
                    elif next_base_type == "SP":
                        sp_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                        if sp_calc_id:
                            new_calc_ids.append(sp_calc_id)
                        else:
                            failed_generations.add(next_calc_type)
                            # SP is usually critical if BAND/DOSS follow
                            print(f"CRITICAL: Failed to generate {next_calc_type}")
                    elif next_base_type == "FREQ":
                        # FREQ needs optimized geometry from the highest numbered OPT calculation
                        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                        
                        # Build completed_by_type dict for finding highest OPT
                        completed_by_type = {}
                        for calc in all_calcs:
                            if calc['status'] == 'completed':
                                calc_type = calc['calc_type']
                                if calc_type not in completed_by_type:
                                    completed_by_type[calc_type] = []
                                completed_by_type[calc_type].append(calc)
                        
                        # Find the highest numbered OPT
                        opt_calc_id = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        
                        if opt_calc_id:
                            freq_calc_id = self.generate_freq_from_opt(opt_calc_id, next_calc_type)
                            if freq_calc_id:
                                new_calc_ids.append(freq_calc_id)
                            else:
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        else:
                            print(f"No completed OPT calculation found for {next_calc_type} generation")
                            failed_generations.add(next_calc_type)
            else:
                # Default behavior: generate SP only (FREQ should be explicitly requested in workflow)
                sp_calc_id = self.generate_sp_from_opt(completed_calc_id)
                if sp_calc_id:
                    new_calc_ids.append(sp_calc_id)
                # Note: FREQ generation removed from default behavior - should be explicitly in workflow plan
        
        elif base_type == "SP":
            # Generate next steps based on workflow plan or default behavior
            print(f"SP completed. Planned sequence: {planned_sequence}")
            if planned_sequence:
                # Find current position and get next steps
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                next_steps = self._get_next_steps_from_sequence(current_index, planned_sequence, calc_type)
                
                # Track completed and failed calculations for dependency checking
                all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                completed_calcs = {calc['calc_type'] for calc in all_calcs if calc['status'] == 'completed'}
                failed_generations = set()  # Track failures in this execution
                
                # Generate calculations for all next steps (which may be parallel)
                for next_calc_type in next_steps:
                    # Check if calculation already exists (submitted, running, or completed)
                    if self._calculation_already_exists(material_id, next_calc_type):
                        print(f"Calculation {next_calc_type} already exists for {material_id}, skipping...")
                        continue
                    
                    # Check dependencies first
                    deps_met, blocking_calc = self._check_dependencies_met(
                        next_calc_type, material_id, planned_sequence, 
                        completed_calcs, failed_generations
                    )
                    
                    if not deps_met:
                        print(f"Skipping {next_calc_type} - dependency {blocking_calc} not met")
                        continue
                    
                    try:
                        next_base_type, next_num = self._parse_calc_type(next_calc_type)
                        
                        if next_base_type == "DOSS":
                            print(f"Generating {next_calc_type} from planned sequence...")
                            doss_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                            if doss_calc_id:
                                new_calc_ids.append(doss_calc_id)
                            else:
                                # Generation failed
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        elif next_base_type == "BAND":
                            print(f"Generating {next_calc_type} from planned sequence...")
                            band_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                            if band_calc_id:
                                new_calc_ids.append(band_calc_id)
                            else:
                                # Generation failed
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        elif next_base_type == "TRANSPORT":
                            print(f"Generating {next_calc_type} from planned sequence...")
                            transport_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                            if transport_calc_id:
                                new_calc_ids.append(transport_calc_id)
                            else:
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        elif next_base_type == "CHARGE+POTENTIAL":
                            print(f"Generating {next_calc_type} from planned sequence...")
                            charge_potential_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                            if charge_potential_calc_id:
                                new_calc_ids.append(charge_potential_calc_id)
                            else:
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        elif next_base_type == "OPT":
                            # Check if we have a previous OPT to use
                            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                            completed_by_type = {}
                            for calc in all_calcs:
                                if calc['status'] == 'completed':
                                    ct = calc['calc_type']
                                    if ct not in completed_by_type:
                                        completed_by_type[ct] = []
                                    completed_by_type[ct].append(calc)
                            
                            # Find any completed OPT
                            opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                            
                            if opt_source:
                                # Use existing OPT as source
                                print(f"Generating {next_calc_type} from previous OPT...")
                                opt_calc_id = self.generate_numbered_calculation(opt_source, next_calc_type)
                            else:
                                # No OPT exists, generate from CIF
                                print(f"No previous OPT found. Generating {next_calc_type} from CIF...")
                                opt_calc_id = self.generate_calculation_from_cif(material_id, next_calc_type)
                            
                            if opt_calc_id:
                                new_calc_ids.append(opt_calc_id)
                            else:
                                # Generation failed
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                        elif next_base_type == "SP":
                            # Generate another SP from current SP
                            print(f"Generating {next_calc_type} from SP...")
                            sp_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                            if sp_calc_id:
                                new_calc_ids.append(sp_calc_id)
                        elif next_base_type == "FREQ":
                            # FREQ needs optimized geometry from the highest numbered OPT calculation
                            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                            
                            # Build completed_by_type dict for finding highest OPT
                            completed_by_type = {}
                            for calc in all_calcs:
                                if calc['status'] == 'completed':
                                    calc_type_temp = calc['calc_type']
                                    if calc_type_temp not in completed_by_type:
                                        completed_by_type[calc_type_temp] = []
                                    completed_by_type[calc_type_temp].append(calc)
                            
                            # Find the highest numbered OPT
                            opt_calc_id = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                            
                            if opt_calc_id:
                                print(f"Generating {next_calc_type} from OPT...")
                                freq_calc_id = self.generate_freq_from_opt(opt_calc_id, next_calc_type)
                                if freq_calc_id:
                                    new_calc_ids.append(freq_calc_id)
                                else:
                                    print(f"Failed to generate {next_calc_type}")
                                    failed_generations.add(next_calc_type)
                            else:
                                print(f"No completed OPT found for {next_calc_type} generation")
                                failed_generations.add(next_calc_type)
                                if self._is_calculation_optional(next_calc_type):
                                    print(f"Failed to generate optional {next_calc_type}, continuing...")
                                else:
                                    print(f"CRITICAL: Failed to generate {next_calc_type}")
                    except Exception as e:
                        print(f"Exception generating {next_calc_type}: {e}")
                        failed_generations.add(next_calc_type)
                        
                        # Check if this is a critical calculation
                        if self._is_calculation_optional(next_calc_type):
                            print(f"Optional calculation {next_calc_type} failed, continuing...")
                        else:
                            print(f"CRITICAL: Required calculation {next_calc_type} failed!")
                            print(f"This may block dependent calculations.")
                        continue
            else:
                # Default behavior: generate both DOSS and BAND
                print("No planned sequence found. Using default: generating both DOSS and BAND...")
                doss_calc_id = self.generate_doss_from_sp(completed_calc_id)
                if doss_calc_id:
                    new_calc_ids.append(doss_calc_id)
                    
                band_calc_id = self.generate_band_from_sp(completed_calc_id)
                if band_calc_id:
                    new_calc_ids.append(band_calc_id)
                
        elif base_type in ["FREQ", "BAND", "DOSS"]:
            # These calculations are often terminal, but sometimes workflow continues
            if planned_sequence:
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                # Use the new function that skips already-existing calculations
                next_steps = self._get_next_unstarted_steps(current_index, planned_sequence, material_id)
                
                # Only generate next steps if we have them in the plan
                if next_steps:
                    for next_calc_type in next_steps:
                        next_base_type, next_num = self._parse_calc_type(next_calc_type)
                        
                        if next_base_type == "OPT":
                            # OPT after FREQ/BAND/DOSS needs geometry from highest completed OPT
                            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                            completed_by_type = {}
                            for calc in all_calcs:
                                if calc['status'] == 'completed':
                                    ct = calc['calc_type']
                                    if ct not in completed_by_type:
                                        completed_by_type[ct] = []
                                    completed_by_type[ct].append(calc)
                            
                            # Find highest completed OPT
                            opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                            if opt_source:
                                opt_calc_id = self.generate_numbered_calculation(opt_source, next_calc_type)
                                if opt_calc_id:
                                    new_calc_ids.append(opt_calc_id)
                            else:
                                print(f"No completed OPT found to use as source for {next_calc_type}")
                        elif next_base_type == "FREQ":
                            # Another FREQ calculation - also needs OPT geometry
                            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                            completed_by_type = {}
                            for calc in all_calcs:
                                if calc['status'] == 'completed':
                                    ct = calc['calc_type']
                                    if ct not in completed_by_type:
                                        completed_by_type[ct] = []
                                    completed_by_type[ct].append(calc)
                            
                            opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                            if opt_source:
                                freq_calc_id = self.generate_freq_from_opt(opt_source, next_calc_type)
                                if freq_calc_id:
                                    new_calc_ids.append(freq_calc_id)
                        # Add other calculation types as needed
        
        # Note: We do NOT check for all pending calculations here anymore
        # The workflow should progress step by step based on actual dependencies
        # This prevents premature triggering of later steps
        
        # Update workflow state if we have a workflow_id
        if workflow_id:
            try:
                self.db.update_workflow_state(workflow_id, completed_step=calc_type)
            except Exception as e:
                print(f"Failed to update workflow state: {e}")
        
        if new_calc_ids:
            print(f"Generated {len(new_calc_ids)} new calculations: {new_calc_ids}")
        else:
            print(f"No follow-up calculations generated for {calc_type}")
            
        return new_calc_ids
        
    def _find_calc_position_in_sequence(self, calc_type: str, completed_calc: Dict, planned_sequence: List[str]) -> int:
        """
        Find the position of the current calculation in the planned sequence.
        
        Handles numbered calculation types (OPT2, OPT3, SP2, BAND2, etc.)
        """
        # Extract base type and number from calc_type (e.g., OPT2 -> OPT, 2)
        base_type, type_num = self._parse_calc_type(calc_type)
        
        # Count how many calculations of this base type have been completed for this material
        material_id = completed_calc['material_id']
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Count completed calculations of this base type
        completed_count = 0
        for calc in all_calcs:
            calc_base, _ = self._parse_calc_type(calc['calc_type'])
            if calc_base == base_type and calc['status'] == 'completed':
                completed_count += 1
        
        # Find all positions of this base type in the sequence
        type_positions = []
        for i, seq_type in enumerate(planned_sequence):
            seq_base, _ = self._parse_calc_type(seq_type)
            if seq_base == base_type:
                type_positions.append(i)
        
        print(f"DEBUG _find_calc_position: calc_type={calc_type}, base_type={base_type}")
        print(f"DEBUG _find_calc_position: completed_count={completed_count}, type_positions={type_positions}")
        
        # Return the position corresponding to the just-completed calculation
        # completed_count = 1 means we just finished the first OPT, so we're at position 0
        if completed_count > 0 and completed_count <= len(type_positions):
            position = type_positions[completed_count - 1]
            print(f"DEBUG _find_calc_position: returning position {position}")
            return position
        elif type_positions:
            # We've done more than planned, return last position of this type
            position = type_positions[-1]
            print(f"DEBUG _find_calc_position: returning last position {position}")
            return position
        else:
            # Not found in sequence, return end
            position = len(planned_sequence) - 1
            print(f"DEBUG _find_calc_position: returning end position {position}")
            return position
                
    def _get_next_steps_from_sequence(self, current_index: int, planned_sequence: List[str], 
                                     completed_calc_type: str) -> List[str]:
        """
        Get the next calculation steps from the planned sequence.
        
        Returns the immediate next step(s) in the sequence. For parallel calculations
        (e.g., BAND and DOSS which both read from the same SP wavefunction), returns
        both if they're adjacent in the sequence.
        
        Args:
            current_index: Current position in the sequence
            planned_sequence: The full planned calculation sequence
            completed_calc_type: The type of calculation that just completed
            
        Returns:
            List of calculation types that should be started next
        """
        if not planned_sequence or current_index >= len(planned_sequence) - 1:
            return []
        
        # Get the immediate next step
        next_index = current_index + 1
        next_steps = []
        
        if next_index < len(planned_sequence):
            next_calc = planned_sequence[next_index]
            next_steps.append(next_calc)
            
            # Only handle truly parallel cases where calculations can share the same input
            next_base, _ = self._parse_calc_type(next_calc)
            
            # BAND and DOSS can run in parallel as they both just read the wavefunction
            # TRANSPORT and CHARGE+POTENTIAL should run sequentially after BAND/DOSS
            if next_base in ["BAND", "DOSS"]:
                # Check if the other property calculation is also in the sequence
                other_base = "DOSS" if next_base == "BAND" else "BAND"
                # Look for the other property calculation nearby in the sequence
                for i in range(max(0, next_index - 1), min(len(planned_sequence), next_index + 2)):
                    if i != next_index:
                        # Parse the calculation type at this position
                        check_base, check_num = self._parse_calc_type(planned_sequence[i])
                        # If it's the complementary calculation type (BAND/DOSS only)
                        if check_base == other_base:
                            # Check if it hasn't been started yet
                            if planned_sequence[i] not in next_steps:
                                next_steps.append(planned_sequence[i])
                            break
            # Note: TRANSPORT and CHARGE+POTENTIAL are not included in parallel execution
            # They will run sequentially as defined in the workflow sequence
        
        return next_steps
    
    def _get_next_unstarted_steps(self, current_index: int, planned_sequence: List[str], 
                                  material_id: str) -> List[str]:
        """
        Get the next unstarted calculation steps from the planned sequence.
        
        This function looks ahead in the sequence to find calculations that haven't
        been started yet, skipping over any that already exist.
        
        Args:
            current_index: Current position in the sequence
            planned_sequence: The full planned calculation sequence
            material_id: Material ID to check for existing calculations
            
        Returns:
            List of calculation types that should be started next
        """
        if not planned_sequence or current_index >= len(planned_sequence) - 1:
            return []
        
        next_steps = []
        
        # Look ahead in the sequence for unstarted calculations
        for i in range(current_index + 1, len(planned_sequence)):
            calc_type = planned_sequence[i]
            
            # Check if this calculation already exists
            if self._calculation_already_exists(material_id, calc_type):
                continue
                
            # Add this unstarted calculation
            next_steps.append(calc_type)
            
            # Check if we should also include parallel calculations
            base_type, _ = self._parse_calc_type(calc_type)
            
            # For BAND/DOSS, check if the complementary calculation should also be included
            if base_type in ["BAND", "DOSS"] and i + 1 < len(planned_sequence):
                next_calc = planned_sequence[i + 1]
                next_base, _ = self._parse_calc_type(next_calc)
                
                # If the next calculation is the complementary property calculation
                if (base_type == "BAND" and next_base == "DOSS") or \
                   (base_type == "DOSS" and next_base == "BAND"):
                    if not self._calculation_already_exists(material_id, next_calc):
                        next_steps.append(next_calc)
                    # Skip the next iteration since we've already processed it
                    i += 1
            
            # For most cases, only return the immediate next unstarted step(s)
            # Exception: BAND/DOSS parallel execution handled above
            if next_steps:
                break
        
        return next_steps
        
    def _validate_property_templates(self, calc_type: str) -> bool:
        """
        Validate that required scripts exist for property calculations.
        
        Args:
            calc_type: Calculation type (BAND, DOSS, etc.)
            
        Returns:
            True if templates/scripts exist, False otherwise
        """
        base_type, _ = self._parse_calc_type(calc_type)
        
        if base_type in ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"]:
            # Check for CRYSTALOptToD3.py script
            if 'crystal_to_d3' not in self.script_paths:
                print(f"CRYSTALOptToD3.py script not found")
                return False
                
            script_path = self.script_paths['crystal_to_d3']
            if not script_path.exists():
                print(f"CRYSTALOptToD3.py script not found at: {script_path}")
                return False
            return True
            
        # Other calculation types don't need special templates
        return True
    
    def _parse_calc_type(self, calc_type: str) -> Tuple[str, int]:
        """
        Parse calculation type to extract base type and number.
        
        Examples:
            OPT -> (OPT, 1)
            OPT2 -> (OPT, 2)
            OPT_1 -> (OPT, 1)
            OPT_2 -> (OPT, 2)
            SP -> (SP, 1)
            SP2 -> (SP, 2)
            SP_2 -> (SP, 2)
            BAND3 -> (BAND, 3)
            BAND_3 -> (BAND, 3)
            CHARGE+POTENTIAL -> (CHARGE+POTENTIAL, 1)
            CHARGE+POTENTIAL_2 -> (CHARGE+POTENTIAL, 2)
        """
        import re
        # Handle both formats: TYPE_N and TYPEN
        match = re.match(r'^([A-Z]+(?:\+[A-Z]+)?)(?:_|)(\d*)$', calc_type)
        if match:
            base_type = match.group(1)
            num_str = match.group(2)
            num = int(num_str) if num_str else 1
            return base_type, num
        else:
            # Fallback for unexpected formats
            return calc_type, 1
    
    def _find_dependency_in_sequence(self, calc_type: str, planned_sequence: List[str]) -> Optional[str]:
        """
        Find what calculation type this step depends on based on the planned sequence.
        
        This understands the actual computational dependencies, not just sequence order.
        For example, BAND and DOSS both depend on SP/OPT (for wavefunction), not on each other.
        
        Args:
            calc_type: The calculation type we're checking dependencies for
            planned_sequence: The full planned calculation sequence
            
        Returns:
            The calculation type this depends on, or None if not found
        """
        if not planned_sequence or calc_type not in planned_sequence:
            return None
            
        # Find the position of this calc type in the sequence
        try:
            calc_index = planned_sequence.index(calc_type)
        except ValueError:
            return None
            
        # If it's the first step, it has no dependencies
        if calc_index == 0:
            return None
            
        base_type, type_num = self._parse_calc_type(calc_type)
        
        # Define actual computational dependencies
        if base_type == "SP":
            # SP depends on the previous OPT (or previous numbered OPT)
            # Look backwards for the most recent OPT
            for i in range(calc_index - 1, -1, -1):
                prev_base, _ = self._parse_calc_type(planned_sequence[i])
                if prev_base == "OPT":
                    return planned_sequence[i]
            return None
            
        elif base_type in ["BAND", "DOSS"]:
            # BAND and DOSS depend on SP or OPT (for wavefunction)
            # Look backwards for the most recent SP or OPT
            for i in range(calc_index - 1, -1, -1):
                prev_base, _ = self._parse_calc_type(planned_sequence[i])
                if prev_base in ["SP", "OPT"]:
                    return planned_sequence[i]
            return None
            
        elif base_type == "FREQ":
            # FREQ depends on the previous OPT (needs optimized geometry)
            # Look backwards for the most recent OPT
            for i in range(calc_index - 1, -1, -1):
                prev_base, _ = self._parse_calc_type(planned_sequence[i])
                if prev_base == "OPT":
                    return planned_sequence[i]
            return None
            
        elif base_type == "OPT":
            # OPT can depend on:
            # - Previous OPT (for multi-stage optimization)
            # - Previous calculation that can provide geometry
            # - Nothing (if starting from CIF)
            
            # For OPT2, OPT3, etc., look for the previous OPT
            if type_num > 1:
                # Look for OPT with number = type_num - 1
                target_type = f"OPT{type_num - 1}" if type_num > 2 else "OPT"
                if target_type in planned_sequence:
                    return target_type
                    
            # For any OPT after FREQ/BAND/DOSS, it depends on the previous OPT
            # (since FREQ/BAND/DOSS don't produce new geometries)
            for i in range(calc_index - 1, -1, -1):
                prev_base, _ = self._parse_calc_type(planned_sequence[i])
                if prev_base == "OPT":
                    return planned_sequence[i]
                elif prev_base == "SP":
                    # SP can provide geometry too
                    return planned_sequence[i]
                elif prev_base in ["FREQ", "BAND", "DOSS"]:
                    # These don't provide geometry, keep looking
                    continue
                    
            # No dependency found - this OPT starts from CIF
            return None
            
        else:
            # For any other calculation type, use the previous step
            return planned_sequence[calc_index - 1]
    
    def _find_highest_numbered_calc_of_type(self, completed_by_type: Dict[str, List[Dict]], base_type: str) -> Optional[str]:
        """
        Find the highest numbered completed calculation of a given base type.
        
        Args:
            completed_by_type: Dictionary mapping calc_type to list of completed calculations
            base_type: Base calculation type (e.g., 'OPT', 'SP')
            
        Returns:
            Calculation ID of the highest numbered calc of this type, or None
        """
        highest_calc_id = None
        highest_num = 0
        
        # Check all completed calculations
        for calc_type, calcs in completed_by_type.items():
            calc_base, calc_num = self._parse_calc_type(calc_type)
            if calc_base == base_type and calc_num > highest_num:
                highest_num = calc_num
                if calcs:  # Make sure there are calculations
                    highest_calc_id = calcs[-1]['calc_id']  # Take the most recent
        
        return highest_calc_id
    
    def _is_calculation_optional(self, calc_type: str) -> bool:
        """
        Check if a calculation type is optional (can fail without blocking workflow).
        
        Args:
            calc_type: Calculation type (e.g., 'BAND', 'BAND2', 'DOSS', 'FREQ2')
            
        Returns:
            True if the calculation is optional
        """
        base_type, _ = self._parse_calc_type(calc_type)
        return base_type in OPTIONAL_CALC_TYPES
    
    def _calculation_already_exists(self, material_id: str, calc_type: str) -> bool:
        """
        Check if a calculation of this type already exists for the material
        (in any state: submitted, running, completed, failed).
        
        Args:
            material_id: Material ID
            calc_type: Calculation type (e.g., 'BAND2', 'OPT3')
            
        Returns:
            True if calculation already exists
        """
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        for calc in all_calcs:
            if calc['calc_type'] == calc_type:
                # Check if it's not in a terminal failed state that needs retry
                if calc['status'] in ['submitted', 'running', 'completed']:
                    return True
                elif calc['status'] == 'failed':
                    # Could check if it's eligible for retry, but for now just say it exists
                    return True
        return False
    
    def _check_dependencies_met(self, calc_type: str, material_id: str, 
                               planned_sequence: List[str], 
                               completed_calcs: Set[str],
                               failed_calcs: Set[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if dependencies for a calculation are met based on the workflow sequence.
        
        Args:
            calc_type: The calculation type to check
            material_id: Material ID
            planned_sequence: The planned workflow sequence
            completed_calcs: Set of completed calculation types
            failed_calcs: Set of failed calculation types in this execution
            
        Returns:
            (dependencies_met, blocking_calc_type)
        """
        # Find what this calculation depends on in the sequence
        dependency = self._find_dependency_in_sequence(calc_type, planned_sequence)
        
        if not dependency:
            # No dependency - this is the first step
            return True, None
            
        # Check if the dependency is completed
        # Need to check both exact match and base type match
        if dependency in completed_calcs:
            return True, None
            
        # Also check if base type matches (e.g., OPT_1 dependency satisfied by OPT)
        dep_base, dep_num = self._parse_calc_type(dependency)
        for completed in completed_calcs:
            comp_base, comp_num = self._parse_calc_type(completed)
            if comp_base == dep_base:
                # For numbered dependencies, check if we have the right one
                if dep_num == 1 or comp_num == dep_num:
                    return True, None
            
        # Check if the dependency failed in this execution
        if dependency in failed_calcs:
            # Check if the failed dependency was optional
            if self._is_calculation_optional(dependency):
                # Optional dependency failed - we might still continue
                # For BAND/DOSS, we need SP; for others, check sequence
                base_type, _ = self._parse_calc_type(calc_type)
                dep_base, _ = self._parse_calc_type(dependency)
                
                # Special case: D3 calculations really need wavefunction
                if base_type in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL'] and dep_base == 'SP':
                    return False, dependency
                    
                # For other cases, try to find alternative source
                # This will be handled by the existing logic in execute_workflow_step
                return True, None
            else:
                # Critical dependency failed
                return False, dependency
                
        # Dependency not completed yet
        return False, dependency
        
    def generate_opt2_from_opt(self, opt_calc_id: str) -> Optional[str]:
        """
        Generate OPT2 calculation from completed OPT using CRYSTALOptToD12.py.
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            
        Returns:
            New OPT2 calculation ID if successful, None otherwise
        """
        return self.generate_numbered_calculation(opt_calc_id, "OPT2")
    
    def find_original_cif_source(self, material_id: str) -> Optional[Path]:
        """
        Find the original CIF file that was used to create this material.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Path to CIF file if found, None otherwise
        """
        # Look for CIF in workflow_inputs or workflow configuration
        workflow_base = self.base_work_dir
        
        # Check workflow_inputs directory
        workflow_inputs = workflow_base / "workflow_inputs"
        if workflow_inputs.exists():
            # Look for CIF files that match the material name
            core_name = self.extract_core_material_name(material_id)
            cif_patterns = [f"{core_name}.cif", f"*{core_name}*.cif"]
            
            for pattern in cif_patterns:
                cif_files = list(workflow_inputs.glob(pattern))
                if cif_files:
                    return cif_files[0]
        
        # Check workflow config for CIF directory
        config_dir = workflow_base / "workflow_configs"
        if config_dir.exists():
            # Look for workflow plan files
            for plan_file in config_dir.glob("workflow_plan_*.json"):
                try:
                    with open(plan_file, 'r') as f:
                        plan = json.load(f)
                        
                    if plan.get('input_type') == 'cif' and plan.get('input_directory'):
                        cif_dir = Path(plan['input_directory'])
                        if cif_dir.exists():
                            # Look for matching CIF file
                            core_name = self.extract_core_material_name(material_id)
                            for cif_file in cif_dir.glob("*.cif"):
                                if core_name in cif_file.stem:
                                    return cif_file
                except Exception as e:
                    print(f"Error reading workflow plan {plan_file}: {e}")
        
        return None
    
    def generate_calculation_from_cif(self, material_id: str, calc_type: str) -> Optional[str]:
        """
        Generate a calculation (OPT or SP) from the original CIF file.
        
        Args:
            material_id: Material identifier
            calc_type: Calculation type to generate (e.g., 'OPT', 'SP')
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {calc_type} from CIF for {material_id}")
        
        # Find the original CIF file
        cif_file = self.find_original_cif_source(material_id)
        if not cif_file:
            print(f"Could not find original CIF file for {material_id}")
            return None
        
        print(f"  Found CIF file: {cif_file}")
        
        # Create working directory for CIF conversion
        work_dir = self.create_isolated_calculation_directory(
            material_id, f"{calc_type}_from_cif_generation", [cif_file]
        )
        
        try:
            # Get NewCifToD12.py script
            newcif_script = self.script_paths.get('newcif_to_d12')
            if not newcif_script:
                print("NewCifToD12.py script not found")
                return None
            
            # Prepare arguments for NewCifToD12.py
            args = [
                "--cif-file", str(work_dir / cif_file.name),
                "--output-dir", str(work_dir)
            ]
            
            # Check for CIF conversion config
            config_file = self.base_work_dir / "workflow_configs" / "cif_conversion_config.json"
            if config_file.exists():
                args.extend(["--config-file", str(config_file)])
                print(f"  Using CIF conversion config: {config_file}")
            
            # Determine calculation type for NewCifToD12.py
            # 1 = OPT, 2 = SP
            calc_type_num = "1" if calc_type.startswith("OPT") else "2"
            
            # Prepare input responses for non-interactive mode
            # This assumes using config file or defaults
            input_responses = f"{calc_type_num}\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                newcif_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"NewCifToD12.py failed: {stderr}")
                return None
            
            # Find generated D12 file
            d12_files = list(work_dir.glob("*.d12"))
            if not d12_files:
                print("No D12 file generated by NewCifToD12.py")
                return None
            
            generated_d12 = d12_files[0]
            
            # Get workflow output directory
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            workflow_base = self.base_work_dir / "workflow_outputs" / workflow_id
            
            # Determine step number and create directory
            step_num = self._get_next_step_number(workflow_base, calc_type)
            core_name = self.extract_core_material_name(material_id)
            
            # Parse calc type for numbered calculations
            base_type, type_num = self._parse_calc_type(calc_type)
            if type_num > 1:
                dir_suffix = f"_{base_type.lower()}{type_num}"
            else:
                dir_suffix = f"_{base_type.lower()}"
            
            dir_name = f"{core_name}{dir_suffix}"
            calc_dir = workflow_base / f"step_{step_num:03d}_{calc_type}" / dir_name
            calc_dir.mkdir(parents=True, exist_ok=True)
            
            # Move D12 file to calculation directory
            final_d12_name = f"{core_name}{dir_suffix}.d12"
            final_d12_path = calc_dir / final_d12_name
            shutil.move(generated_d12, final_d12_path)
            
            # Create workflow metadata file for this calculation
            metadata = {
                'workflow_id': workflow_id,
                'step_num': step_num,
                'calc_type': calc_type,
                'material_id': material_id
            }
            metadata_file = calc_dir / '.workflow_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create SLURM script
            job_name = f"{core_name}{dir_suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                calc_dir, job_name, base_type, step_num, workflow_id
            )
            
            # Create calculation record
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=calc_type,
                input_file=str(final_d12_path),
                work_dir=str(calc_dir),
                settings={
                    'generated_from': 'CIF',
                    'generation_method': 'NewCifToD12.py',
                    'cif_source': str(cif_file),
                    'workflow_id': workflow_id,
                    'step_number': step_num
                }
            )
            
            # Submit if auto-submit is enabled
            if hasattr(self, 'auto_submit') and self.auto_submit:
                job_id = self._submit_calculation_to_slurm(slurm_script_path, calc_dir)
                if job_id:
                    self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=job_id)
                    print(f"Submitted {calc_type} calculation as job {job_id}: {final_d12_path}")
                else:
                    print(f"Generated {calc_type} calculation but submission failed: {final_d12_path}")
            else:
                print(f"Generated {calc_type} calculation (pending submission): {final_d12_path}")
            
            return calc_id
            
        finally:
            # Clean up working directory
            shutil.rmtree(work_dir, ignore_errors=True)
    
    def generate_numbered_calculation(self, source_calc_id: str, target_calc_type: str) -> Optional[str]:
        """
        Generate a numbered calculation (OPT2, OPT3, SP2, etc.) from a source calculation.
        
        Args:
            source_calc_id: ID of source calculation
            target_calc_type: Target calculation type (e.g., OPT2, SP2)
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {target_calc_type} calculation from {source_calc_id}")
        
        # Get source calculation details
        source_calc = self.db.get_calculation(source_calc_id)
        if not source_calc or source_calc['status'] != 'completed':
            print(f"Source calculation {source_calc_id} not completed")
            return None
            
        material_id = source_calc['material_id']
        
        # Get output file path - try both direct column and settings
        output_file_path = source_calc.get('output_file')
        if not output_file_path:
            # Try to get from settings
            try:
                settings = json.loads(source_calc.get('settings_json') or '{}')
                output_file_path = settings.get('output_file')
            except:
                pass
                
        if not output_file_path:
            # Try to infer from input file
            input_file_path = source_calc.get('input_file')
            if input_file_path:
                input_path = Path(input_file_path)
                output_file_path = str(input_path.with_suffix('.out'))
                
        if not output_file_path:
            print(f"Could not determine output file path for {source_calc_id}")
            return None
            
        source_output_file = Path(output_file_path)
        source_input_file = Path(source_calc.get('input_file', ''))
        
        if not source_output_file.exists():
            print(f"Source output file not found: {source_output_file}")
            return None
            
        # Parse target calculation type
        target_base_type, target_num = self._parse_calc_type(target_calc_type)
        
        # Extract workflow_id from parent calculation early so we can use it
        workflow_id = None
        workflow_step = None
        if source_calc.get('settings_json'):
            try:
                parent_settings = json.loads(source_calc['settings_json'])
                workflow_id = parent_settings.get('workflow_id')
                workflow_step = parent_settings.get('workflow_step')
            except json.JSONDecodeError:
                pass
        
        # Check for expert config file for numbered calculations (OPT2/OPT3/SP2/FREQ)
        expert_config_file = None
        if (target_base_type in ["OPT", "SP", "FREQ"]) and (target_num > 1 or target_base_type == "FREQ"):
            # First check for material-specific config
            config_search_paths = [
                # Material-specific config in workflow_configs
                self.base_work_dir / "workflow_configs" / f"expert_{target_calc_type.lower()}_configs" / f"{material_id}_{target_calc_type.lower()}_expert_config.json",
                # Legacy location in workflow_temp (general config)
                self.base_work_dir / "workflow_temp" / f"expert_config_{target_calc_type.lower()}" / f"{target_calc_type.lower()}_expert_config.json"
            ]
            
            for config_path in config_search_paths:
                if config_path.exists():
                    expert_config_file = config_path
                    print(f"  Found expert config for {target_calc_type}: {expert_config_file}")
                    break
            
            # If no material-specific config, look for any config in the directories
            if not expert_config_file:
                for config_dir in [p.parent for p in config_search_paths if p.parent.exists()]:
                    config_files = list(config_dir.glob("*_expert_config.json"))
                    if config_files:
                        expert_config_file = config_files[0]
                        print(f"  Found expert config for {target_calc_type}: {expert_config_file}")
                        break
            
        # For OPT2/OPT3, we need to find the ORIGINAL input file (from CIF or initial OPT)
        # to preserve symmetry information
        original_input_file = None
        if target_base_type == "OPT" and target_num > 1:
            # Try to find the original OPT input file
            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
            for calc in all_calcs:
                if calc['calc_type'] == 'OPT' and calc['status'] in ['completed', 'running', 'submitted']:
                    original_opt_input = Path(calc['input_file'])
                    if original_opt_input.exists() and original_opt_input != source_input_file:
                        original_input_file = original_opt_input
                        print(f"  Found original OPT input for symmetry: {original_input_file.name}")
                        break
        
        # Create isolated directory for CRYSTALOptToD12.py
        files_to_copy = [source_output_file, source_input_file]
        if original_input_file and original_input_file.exists():
            files_to_copy.append(original_input_file)
        
        work_dir = self.create_isolated_calculation_directory(
            material_id, f"{target_calc_type}_generation", files_to_copy
        )
        
        try:
            # Run CRYSTALOptToD12.py in isolated directory
            crystal_to_d12_script = self.script_paths['crystal_to_d12']
            
            # Find the specific out and d12 files in the work directory
            out_files = list(work_dir.glob("*.out"))
            d12_files = list(work_dir.glob("*.d12"))
            
            if not out_files:
                print(f"No .out file found in {work_dir}")
                return None
                
            out_file = out_files[0]
            d12_file = d12_files[0] if d12_files else None
            
            # Use single file mode with automatic input responses
            args = [
                "--out-file", str(out_file),
                "--output-dir", str(work_dir)
            ]
            
            if d12_file:
                args.extend(["--d12-file", str(d12_file)])
            
            # For SP/FREQ/OPT generation without expert config, extract functional info
            if target_base_type in ["SP", "FREQ", "OPT"] and not expert_config_file and d12_file:
                print(f"  Extracting functional info from source d12 for {target_base_type} generation")
                try:
                    # Extract settings from the OPT d12 file
                    settings = extract_input_settings(Path(d12_file))
                    functional_info = settings.get('functional_info', {})
                    
                    # Determine the functional from the extracted info
                    functional = None
                    if functional_info.get('method') == 'DFT':
                        # Check for common functionals
                        exchange = functional_info.get('exchange', '').upper()
                        correlation = functional_info.get('correlation', '').upper()
                        dispersion = functional_info.get('dispersion', '')
                        
                        # Import the comprehensive functional list from d12_constants
                        try:
                            from Crystal_d12.d12_constants import FUNCTIONAL_CATEGORIES
                            
                            # Get all functionals from all categories
                            all_functionals = []
                            for category_data in FUNCTIONAL_CATEGORIES.values():
                                all_functionals.extend(category_data.get('functionals', []))
                            
                            # Check if the exchange matches any known functional
                            for func in all_functionals:
                                if func.upper() in exchange.upper():
                                    functional = func
                                    break
                                    
                            # If not found in exchange, check the full content
                            if not functional:
                                for func in all_functionals:
                                    if func.upper() in settings.get('basis_set', '').upper():
                                        # Sometimes functional is in basis set name
                                        functional = func
                                        break
                                        
                        except ImportError:
                            # Fallback to hardcoded list if import fails
                            print("    Warning: Could not import d12_constants, using fallback functional list")
                            if 'B3LYP' in exchange or ('B3' in exchange and 'LYP' in correlation):
                                functional = 'B3LYP'
                            elif 'PBESOL' in exchange or 'PBSOL' in exchange:
                                functional = 'PBESOL'
                            elif 'PBE0' in exchange:
                                functional = 'PBE0'
                            elif 'PBE' in exchange:
                                functional = 'PBE'
                            elif 'HSE06' in exchange:
                                functional = 'HSE06'
                            elif 'BLYP' in exchange or ('B' in exchange and 'LYP' in correlation):
                                functional = 'BLYP'
                        
                        # Add dispersion correction
                        if dispersion == 'D3' and functional:
                            functional += '-D3'
                    elif functional_info.get('method') == 'HF':
                        functional = 'RHF'
                    
                    if functional:
                        # Create a temporary config file for SP/FREQ generation
                        temp_config = work_dir / f"{target_base_type.lower()}_config.json"
                        config_data = {
                            "functional": functional,
                            "calculation_type": target_base_type
                        }
                        with open(temp_config, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        
                        args.extend(["--config-file", str(temp_config)])
                        print(f"    Created config with functional: {functional}")
                        expert_config_file = temp_config  # Mark that we have a config
                    else:
                        print(f"    Could not determine functional from d12 settings: {functional_info}")
                except Exception as e:
                    print(f"    Error extracting functional info: {e}")
            
            # If we have an expert config file for OPT2/OPT3, use it
            if expert_config_file and expert_config_file.exists():
                args.extend(["--config-file", str(expert_config_file)])
                print(f"  Using expert config file for {target_calc_type}")
                print(f"  Config file path: {expert_config_file}")
                # Read and display config contents for debugging
                try:
                    with open(expert_config_file, 'r') as f:
                        config_content = json.load(f)
                        print(f"  Config functional: {config_content.get('functional', 'N/A')}")
                        print(f"  Config dispersion: {config_content.get('dispersion', 'N/A')}")
                except Exception as e:
                    print(f"  Error reading config file: {e}")
                
                # With expert config file, we need different responses based on calc type
                if target_base_type == "FREQ":
                    # For FREQ with config file:
                    # 1. Apply config? → y (yes)
                    # Additional newlines for any prompts CRYSTALOptToD12 might have
                    input_responses = "y\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                else:
                    # For other types with config file:
                    # 1. Apply config? → y (yes)
                    input_responses = "y\n"
            elif not expert_config_file:
                # Check if workflow configuration has settings for this calculation type
                workflow_config = self.get_workflow_step_config(workflow_id, target_calc_type)
                if workflow_config:
                    # Create temporary expert config based on calculation type
                    temp_config = {
                        "calculation_type": target_base_type,
                        "inherit_settings": workflow_config.get('inherit_settings', workflow_config.get('inherit_base_settings', True))
                    }
                    
                    # Add optimization-specific settings
                    if target_base_type == "OPT" and 'optimization_type' in workflow_config:
                        temp_config["optimization_type"] = workflow_config.get('optimization_type', 'FULLOPTG')
                        temp_config["optimization_settings"] = workflow_config.get('optimization_settings', {})
                    
                    # Add frequency-specific settings
                    # Check both possible keys for frequency settings
                    if target_base_type == "FREQ":
                        freq_settings = workflow_config.get('frequency_settings') or workflow_config.get('freq_settings')
                        if freq_settings:
                            print(f"  DEBUG: Found frequency settings: {freq_settings}")
                            # Check if it's the new comprehensive format or old format
                            if isinstance(freq_settings, dict):
                                # New comprehensive format - pass through directly
                                temp_config['freq_settings'] = freq_settings
                                # Also add as frequency_settings for compatibility
                                temp_config['frequency_settings'] = freq_settings
                            else:
                                # Old format compatibility
                                if 'mode' in freq_settings:
                                    temp_config['freq_mode'] = freq_settings['mode']
                                if 'intensities' in freq_settings:
                                    temp_config['ir_intensities'] = freq_settings['intensities']
                                if 'raman' in freq_settings:
                                    temp_config['raman_intensities'] = freq_settings['raman']
                    
                    # Add method settings if present
                    if 'method_settings' in workflow_config:
                        method_settings = workflow_config['method_settings']
                        if 'custom_functional' in method_settings:
                            temp_config['functional'] = method_settings['custom_functional']
                        elif 'new_functional' in method_settings:
                            temp_config['functional'] = method_settings['new_functional']
                            if method_settings.get('use_dispersion'):
                                temp_config['dispersion'] = True
                    
                    # Add basis settings if present
                    if 'basis_settings' in workflow_config:
                        basis_settings = workflow_config['basis_settings']
                        if 'new_basis' in basis_settings:
                            temp_config['basis_set'] = basis_settings['new_basis']
                            temp_config['basis_set_type'] = 'INTERNAL'
                    
                    # Add custom tolerances if present
                    if 'custom_tolerances' in workflow_config:
                        custom_tol = workflow_config['custom_tolerances']
                        if 'TOLINTEG' in custom_tol:
                            temp_config['tolinteg'] = custom_tol['TOLINTEG']
                        if 'TOLDEE' in custom_tol:
                            temp_config['scf_toldee'] = custom_tol['TOLDEE']
                    # Write to temporary file in work directory
                    temp_config_file = work_dir / f"{target_calc_type}_temp_config.json"
                    with open(temp_config_file, 'w') as f:
                        json.dump(temp_config, f, indent=2)
                    args.extend(["--config-file", str(temp_config_file)])
                    print(f"  Created temporary config file for {target_calc_type}")
                    print(f"  Config contents:")
                    for key, value in temp_config.items():
                        if key == 'optimization_settings':
                            print(f"    {key}:")
                            for k, v in value.items():
                                print(f"      {k}: {v}")
                        elif key in ['freq_settings', 'frequency_settings'] and isinstance(value, dict):
                            print(f"    {key}:")
                            for k, v in value.items():
                                print(f"      {k}: {v}")
                        else:
                            print(f"    {key}: {value}")
                # With config file, we need different responses based on calc type
                if target_base_type == "FREQ":
                    # For FREQ with config file:
                    # 1. Apply config? → y (yes)
                    # Additional newlines for any prompts CRYSTALOptToD12 might have
                    input_responses = "y\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                else:
                    # For other types with config file:
                    # 1. Apply config? → y (yes)
                    input_responses = "y\n"
            else:
                # Prepare input responses for non-interactive execution based on target type
                # 1. Keep settings? → y (yes, keep original settings)
                if target_base_type == "OPT":
                    # 2. Calc type → 2 (OPT)
                    # 3. Symmetry choice → 1 (Write only unique atoms)
                    input_responses = "y\n2\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                elif target_base_type == "SP":
                    # 2. Calc type → 1 (SP)
                    # 3. Symmetry choice → 1 (Write only unique atoms)
                    input_responses = "y\n1\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                elif target_base_type == "FREQ":
                    # 2. Calc type → 3 (FREQ)
                    input_responses = "y\n3\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                else:
                    # Default to SP for unknown types
                    input_responses = "y\n1\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"CRYSTALOptToD12.py failed: {stderr}")
                return None
            
            # Debug: show some output to verify config was applied
            if expert_config_file and expert_config_file.exists():
                print(f"  CRYSTALOptToD12.py output (last 10 lines):")
                output_lines = stdout.strip().split('\n')
                for line in output_lines[-10:]:
                    print(f"    {line}")
                
            # Get the original input file name to exclude it from results
            original_d12_name = source_input_file.name
            
            # Find generated .d12 file based on target type
            if target_base_type == "OPT":
                generated_files = list(work_dir.glob("*OPT*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_opt*.d12")) + list(work_dir.glob("*OPTGEOM*.d12"))
            elif target_base_type == "SP":
                generated_files = list(work_dir.glob("*SP*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_sp*.d12")) + list(work_dir.glob("*SCFDIR*.d12"))
            elif target_base_type == "FREQ":
                generated_files = list(work_dir.glob("*FREQ*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_freq*.d12"))
            else:
                # Generic pattern
                generated_files = list(work_dir.glob("*.d12"))
            
            # Filter out the original input file and look for files with optimized/B3LYP/modified in name
            filtered_files = []
            for f in generated_files:
                if f.name != original_d12_name:
                    # Prefer files with these indicators that they were modified
                    if any(x in f.name.lower() for x in ['b3lyp', 'optimized', 'modified', f'_{target_base_type.lower()}_']):
                        filtered_files.insert(0, f)  # Put at beginning
                    else:
                        filtered_files.append(f)
            
            if filtered_files:
                generated_files = filtered_files
            else:
                # If no filtered files, exclude just the original
                generated_files = [f for f in generated_files if f.name != original_d12_name]
                
            if not generated_files:
                print(f"No {target_base_type} input file generated by CRYSTALOptToD12.py")
                return None
                
            generated_input_file = generated_files[0]
            print(f"  Selected generated file: {generated_input_file.name}")
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(source_calc)
            
            # Get core material name
            core_name = self.extract_core_material_name(material_id)
            
            # Create directory naming based on target type and number
            if target_num > 1:
                suffix = f"_{target_base_type.lower()}{target_num}"
            else:
                suffix = f"_{target_base_type.lower()}"
            
            # Find the step number for this calculation type
            step_num = self._get_next_step_number(workflow_base, target_calc_type)
            
            dir_name = f"{core_name}{suffix}"
            # Use the full calculation type in the step directory name
            step_dir = workflow_base / f"step_{step_num:03d}_{target_calc_type}" / dir_name
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move generated file to material's directory with proper naming
            clean_name = f"{core_name}{suffix}.d12"
            final_location = step_dir / clean_name
            shutil.move(generated_input_file, final_location)
            
            # Fix naming for numbered calculations if needed
            if target_num > 1:
                self._fix_numbered_naming(step_dir, core_name, target_base_type, target_num)
            
            # Create SLURM script for calculation
            job_name = f"{core_name}{suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                step_dir, job_name, target_base_type, step_num, workflow_base.name
            )
            
            # Create calculation record
            # workflow_id was already extracted at the beginning of the method
            
            # Use extracted workflow_id or fallback to workflow_base.name
            if not workflow_id:
                workflow_id = workflow_base.name
            
            # Build settings with workflow_id propagation
            settings = {
                'workflow_id': workflow_id,
                'step_number': step_num,
                'generated_from': source_calc_id,
                'generation_method': 'CRYSTALOptToD12.py',
                'parent_calc_id': source_calc_id
            }
            
            # Add workflow_step if it exists
            if workflow_step is not None:
                settings['workflow_step'] = workflow_step + 1
            
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=target_calc_type,
                input_file=str(final_location),
                work_dir=str(step_dir),
                settings=settings
            )
            
            # Create workflow metadata file for callback tracking
            metadata = {
                'workflow_id': workflow_id,  # Use the extracted/propagated workflow_id
                'step_num': step_num,
                'calc_type': target_calc_type,
                'material_id': material_id,
                'calc_id': calc_id,
                'generated_from': source_calc_id
            }
            metadata_file = step_dir / '.workflow_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Submit if auto-submission is enabled
            if hasattr(self, 'auto_submit') and self.auto_submit:
                job_id = self._submit_calculation_to_slurm(slurm_script_path, step_dir)
                if job_id:
                    self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=job_id)
                    print(f"Submitted {target_calc_type} calculation as job {job_id}: {final_location}")
                else:
                    print(f"Generated {target_calc_type} calculation {calc_id} but submission failed: {final_location}")
            else:
                print(f"Generated {target_calc_type} calculation {calc_id} (pending submission): {final_location}")
            
            return calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def _get_next_step_number(self, workflow_base: Path, calc_type: str) -> int:
        """Get the next available step number for a calculation type."""
        # First try to get from workflow plan
        workflow_id = workflow_base.name
        step_num = self.get_workflow_step_number(workflow_id, calc_type)
        if step_num:
            return step_num
            
        # Fallback to the original logic if no workflow plan
        # For numbered calculations (OPT2, OPT3, SP2, etc.), we need to check the base type
        # but create separate steps for each numbered variant
        base_type, type_num = self._parse_calc_type(calc_type)
        
        # Check if a step already exists for this EXACT calculation type (including number)
        for step_dir in workflow_base.glob("step_*"):
            if step_dir.is_dir():
                # Check if the directory name contains this calc_type
                dir_parts = step_dir.name.split('_')
                if len(dir_parts) >= 3:
                    # For numbered types, match exact type (OPT2, not just OPT)
                    if type_num > 1:
                        step_calc_type = '_'.join(dir_parts[2:])  # Handle names like step_002_OPT2
                        if step_calc_type == calc_type or step_calc_type == base_type + str(type_num):
                            try:
                                return int(dir_parts[1])
                            except ValueError:
                                pass
                    else:
                        # For base types (OPT, SP, etc.), only match if it's the base type
                        if dir_parts[2] == calc_type:
                            try:
                                return int(dir_parts[1])
                            except ValueError:
                                pass
        
        # No existing step for this calc type, find the next available number
        existing_steps = []
        for step_dir in workflow_base.glob("step_*"):
            if step_dir.is_dir():
                # Extract step number from directory name
                try:
                    step_num = int(step_dir.name.split('_')[1])
                    existing_steps.append(step_num)
                except (IndexError, ValueError):
                    pass
                    
        # Return next available number
        return max(existing_steps) + 1 if existing_steps else 1
        
        
    def _fix_numbered_naming(self, output_dir: Path, core_name: str, base_type: str, type_num: int):
        """Fix naming for numbered calculation files generated by CRYSTALOptToD12.py"""
        # CRYSTALOptToD12.py might generate files with duplicate type suffixes
        # e.g., material_sp_sp.d12 should be material_sp2.d12
        
        type_lower = base_type.lower()
        for file_path in output_dir.glob("*.d12"):
            # Check for duplicate suffixes like _sp_sp or _opt_opt
            if f"_{type_lower}_{type_lower}" in file_path.name:
                # Create new name with number
                new_name = f"{core_name}_{type_lower}{type_num}.d12"
                new_path = output_dir / new_name
                
                if not new_path.exists():
                    file_path.rename(new_path)
                    print(f"        Renamed: {file_path.name} → {new_name}")
                    
    def _find_most_recent_wavefunction_calc(self, material_id: str) -> Optional[str]:
        """
        Find the most recent completed calculation with a wavefunction (SP or OPT).
        
        BAND and DOSS calculations need the wavefunction from the most recent SP or OPT.
        
        Returns:
            Calculation ID of the most recent wavefunction calculation, or None if not found
        """
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Filter for completed calculations that produce wavefunctions
        wf_calcs = []
        for calc in all_calcs:
            if calc['status'] == 'completed':
                calc_base, _ = self._parse_calc_type(calc['calc_type'])
                if calc_base in ['SP', 'OPT']:
                    # Check if output file exists and wavefunction file exists
                    output_file = Path(calc.get('output_file', ''))
                    if output_file.exists():
                        wf_file = output_file.parent / 'fort.9'
                        f9_file = output_file.with_suffix('.f9')
                        if wf_file.exists() or f9_file.exists():
                            wf_calcs.append(calc)
        
        # Sort by completion time and return most recent
        if wf_calcs:
            wf_calcs.sort(key=lambda c: c.get('end_time', c.get('start_time', '')), reverse=True)
            return wf_calcs[0]['calc_id']
        
        return None
        
    def get_workflow_status(self, material_id: str) -> Dict:
        """Get comprehensive workflow status for a material."""
        material = self.db.get_material(material_id)
        if not material:
            return {"error": f"Material {material_id} not found"}
            
        # Get all calculations for this material
        calculations = self.db.get_calculations_by_status(material_id=material_id)
        
        # Organize by calculation type and status
        workflow_status = {
            "material_id": material_id,
            "formula": material.get("formula", "Unknown"),
            "total_calculations": len(calculations),
            "by_type": {},
            "completed_workflow_steps": [],
            "pending_workflow_steps": [],
            "failed_calculations": [],
            "workflow_complete": False
        }
        
        calc_types = ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT"]
        
        for calc_type in calc_types:
            type_calcs = [c for c in calculations if c['calc_type'] == calc_type]
            workflow_status["by_type"][calc_type] = {
                "total": len(type_calcs),
                "completed": len([c for c in type_calcs if c['status'] == 'completed']),
                "running": len([c for c in type_calcs if c['status'] == 'running']),
                "failed": len([c for c in type_calcs if c['status'] == 'failed']),
                "pending": len([c for c in type_calcs if c['status'] == 'pending'])
            }
            
            # Track completed and failed
            if workflow_status["by_type"][calc_type]["completed"] > 0:
                workflow_status["completed_workflow_steps"].append(calc_type)
                
            if workflow_status["by_type"][calc_type]["failed"] > 0:
                failed_calcs = [c for c in type_calcs if c['status'] == 'failed']
                workflow_status["failed_calculations"].extend([
                    {
                        "calc_id": c['calc_id'],
                        "calc_type": c['calc_type'],
                        "error_type": c.get('error_type', 'unknown'),
                        "error_message": c.get('error_message', '')
                    } for c in failed_calcs
                ])
        
        # Determine next steps
        if "OPT" in workflow_status["completed_workflow_steps"]:
            if "SP" not in workflow_status["completed_workflow_steps"]:
                workflow_status["pending_workflow_steps"].append("SP")
            if "FREQ" not in workflow_status["completed_workflow_steps"]:
                workflow_status["pending_workflow_steps"].append("FREQ")
            if "SP" in workflow_status["completed_workflow_steps"]:
                if "BAND" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("BAND")
                if "DOSS" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("DOSS")
        else:
            workflow_status["pending_workflow_steps"].append("OPT")
            
        # Check if complete workflow is finished (OPT -> SP/FREQ -> BAND/DOSS)
        if all(step in workflow_status["completed_workflow_steps"] for step in ["OPT", "SP", "BAND", "DOSS", "FREQ"]):
            workflow_status["workflow_complete"] = True
            
        return workflow_status
        
    def process_completed_calculations(self) -> int:
        """
        Process all recently completed calculations and trigger workflow steps.
        
        Returns:
            Number of new workflow steps initiated
        """
        # Get recently completed calculations that haven't been processed
        completed_calcs = self.db.get_calculations_by_status('completed')
        
        new_steps = 0
        for calc in completed_calcs:
            # Check if this calculation has already triggered workflow steps
            settings_json = calc.get('settings_json')
            if settings_json:
                try:
                    settings = json.loads(settings_json)
                except (json.JSONDecodeError, TypeError):
                    settings = {}
            else:
                settings = {}
                
            if settings.get('workflow_processed'):
                continue
                
            # Execute workflow step
            new_calc_ids = self.execute_workflow_step(calc['material_id'], calc['calc_id'])
            
            if new_calc_ids:
                new_steps += len(new_calc_ids)
            
            # Always mark this calculation as workflow processed, even if no new calculations were generated
            # This prevents re-processing the same calculation multiple times
            settings['workflow_processed'] = True
            settings['workflow_process_timestamp'] = datetime.now().isoformat()
            settings['workflow_steps_generated'] = len(new_calc_ids) if new_calc_ids else 0
            self.db.update_calculation_settings(calc['calc_id'], settings)
                
        return new_steps


def main():
    """CLI interface for workflow engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MACE Workflow Engine")
    parser.add_argument("--action", choices=['status', 'process', 'workflow'], 
                       default='process', help="Action to perform")
    parser.add_argument("--material-id", help="Material ID for status checking")
    parser.add_argument("--db", default="materials.db", help="Path to materials database")
    parser.add_argument("--work-dir", default=".", help="Base working directory")
    
    args = parser.parse_args()
    
    # Initialize workflow engine
    workflow_engine = WorkflowEngine(args.db, args.work_dir)
    
    if args.action == 'status':
        if args.material_id:
            status = workflow_engine.get_workflow_status(args.material_id)
            print(f"\n=== Workflow Status for {args.material_id} ===")
            print(f"Formula: {status['formula']}")
            print(f"Total calculations: {status['total_calculations']}")
            print(f"Completed steps: {', '.join(status['completed_workflow_steps'])}")
            print(f"Pending steps: {', '.join(status['pending_workflow_steps'])}")
            print(f"Workflow complete: {status['workflow_complete']}")
            
            if status['failed_calculations']:
                print(f"\nFailed calculations:")
                for failed in status['failed_calculations']:
                    print(f"  - {failed['calc_id']} ({failed['calc_type']}): {failed['error_type']}")
        else:
            print("Please specify --material-id for status checking")
            
    elif args.action == 'process':
        print("Processing completed calculations...")
        new_steps = workflow_engine.process_completed_calculations()
        print(f"Initiated {new_steps} new workflow steps")
        
    elif args.action == 'workflow':
        # Show workflow status for all materials
        all_materials = workflow_engine.db.get_all_materials()
        print(f"\n=== Workflow Status Summary ({len(all_materials)} materials) ===")
        
        for material in all_materials:
            status = workflow_engine.get_workflow_status(material['material_id'])
            completed = len(status['completed_workflow_steps'])
            total_steps = 4  # OPT, SP, BAND, DOSS
            progress = f"{completed}/{total_steps}"
            
            print(f"{material['material_id']}: {progress} steps completed")
            if status['failed_calculations']:
                print(f"  ⚠️  {len(status['failed_calculations'])} failed calculations")


if __name__ == "__main__":
    main()