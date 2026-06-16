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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import threading
import queue

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
    from mace.queue.manager import EnhancedCrystalQueueManager
    from mace.workflow.context import WorkflowContext, workflow_context, get_current_context
    # Crystal_d12 modules no longer needed here - handled by subprocess calls
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Visual layer facade. rich-OPTIONAL and adaptive: plain text when non-TTY (the
# unattended HPCC path), styled when interactive. The guarded import + shim below
# is ESSENTIAL — if mace.utils.ui ever fails to import on a compute node, this
# module must fall back to plain print() and keep the unattended pipeline working.
try:
    from mace.utils import ui
except Exception:
    import sys as _sys, re as _re
    class _UIShim:
        _TAG = _re.compile(r"\[/?[a-z#][^\[\]]*\]")
        def _p(self, m): return self._TAG.sub("", str(m))
        def ok(self, m): print(self._p(m))
        def info(self, m): print(self._p(m))
        def print(self, m): print(self._p(m))
        def warn(self, m): print(self._p(m), file=_sys.stderr)
        def err(self, m): print(self._p(m), file=_sys.stderr)
        def rule(self, t=""): print(self._p(t))
        def table(self, cols, rows, title=None):
            if title: print(self._p(title))
            for r in rows: print("  ".join(str(c) for c in r))
        def progress(self, it, **k): return it
        def badge(self, s): return str(s).upper()
    ui = _UIShim()


class WorkflowExecutor:
    """
    Executes planned workflows with full configuration management and error handling.
    
    Integrates deeply with NewCifToD12.py and CRYSTALOptToD12.py to ensure
    consistent settings across all calculation steps.
    """
    
    def __init__(self, work_dir: str = ".", db_path: str = "materials.db"):
        self.work_dir = Path(work_dir).resolve()
        
        # Check for active context and use its paths if available
        ctx = get_current_context()
        if ctx:
            # Use context-specific database path
            self.db_path = str(ctx.get_database_path())
            ui.info(f"Using context database: {self.db_path}")
            # In isolated mode, don't create database here - it will be created by context
            self.db = None  # Will be set when context is activated
        else:
            # Only create database in shared mode
            self.db_path = db_path
            if db_path == "materials.db" and not Path(db_path).exists():
                # Delay database creation until isolation mode is determined
                # This prevents creating materials.db before we know if we're in isolation mode
                self.db = None
            else:
                # Create database for non-default paths or existing databases
                self.db = MaterialDatabase(self.db_path, auto_initialize=True)
        
        # Set up directories
        self.configs_dir = self.work_dir / "workflow_configs"
        self.outputs_dir = self.work_dir / "workflow_outputs"
        self.temp_dir = self.work_dir / "workflow_temp"
        
        for dir_path in [self.configs_dir, self.outputs_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize queue manager - delay if in isolated mode or if no database exists yet
        if ctx:
            # In isolated mode, don't create queue manager here
            self.queue_manager = None  # Will be set when context is activated
        elif db_path == "materials.db" and not Path(db_path).exists():
            # If using default path and database doesn't exist, delay initialization
            # This prevents creating materials.db before we know the isolation mode
            self.queue_manager = None
            # Queue manager will be created later when isolation mode is determined
        else:
            # Only create queue manager in shared mode with existing database
            # Use default settings for now - will be updated from plan during execution
            self.queue_manager = EnhancedCrystalQueueManager(
                d12_dir=str(self.outputs_dir),
                max_jobs=200,  # Will be updated from workflow plan
                enable_tracking=True,
                enable_error_recovery=True,
                max_recovery_attempts=3,  # Will be updated from workflow plan
                db_path=self.db_path
            )
        
        # Track active workflows
        self.active_workflows = {}
        self.execution_lock = threading.RLock()
        
    def recreate_workflow_scripts(self, plan: Dict[str, Any]):
        """Recreate workflow_scripts directory from JSON configuration"""
        ui.rule("Phase 0: Recreating workflow scripts from configuration...")
        
        scripts_dir = self.work_dir / "workflow_scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Get the base MACE directory
        mace_dir = Path(__file__).parent.parent
        
        # Get workflow_id from plan
        workflow_id = plan.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Iterate through all step configurations
        for step_key, step_config in plan.get('step_configurations', {}).items():
            if 'slurm_config' in step_config and 'scripts' in step_config['slurm_config']:
                for script_name, script_config in step_config['slurm_config']['scripts'].items():
                    # Add workflow_id to script config
                    script_config['workflow_id'] = workflow_id
                    
                    # Try to get source script - handle both absolute and relative paths
                    source_script_path = script_config.get('source_script', '')
                    if source_script_path.startswith('/'):
                        # Absolute path provided - use it directly
                        source_script = Path(source_script_path)
                    else:
                        # Relative path - use standard location
                        source_script = mace_dir / "submission" / script_name
                    
                    target_script = scripts_dir / script_config['step_specific_name']
                    
                    if source_script.exists():
                        # Read source script as-is (keep as script generators)
                        with open(source_script, 'r') as f:
                            content = f.read()
                        
                        # Apply customizations from config
                        content = self.apply_script_customizations(content, script_config)
                        
                        # Write customized script
                        with open(target_script, 'w') as f:
                            f.write(content)
                        
                        # Make executable
                        target_script.chmod(0o755)
                        
                        ui.ok(f"  Created {target_script.name} for {step_key}")
                    else:
                        ui.warn(f"  Warning: Source script {source_script} not found")
        
        print("")
        
    def apply_script_customizations(self, content: str, script_config: Dict[str, Any]) -> str:
        """Apply customizations to SLURM script content"""
        # Apply resource customizations
        if 'resources' in script_config:
            resources = script_config['resources']

            # SLURM directive mappings - handle both short and long forms
            # Format: (short_flag, long_flag, new_value)
            slurm_mappings = [
                ('-n ', '--ntasks=', str(resources.get('ntasks', 32))),
                ('--ntasks=', '--ntasks=', str(resources.get('ntasks', 32))),
                ('-N ', '--nodes=', str(resources.get('nodes', 1))),
                ('--nodes=', '--nodes=', str(resources.get('nodes', 1))),
                ('-t ', '--time=', resources.get('walltime', '7-00:00:00')),
                ('--time=', '--time=', resources.get('walltime', '7-00:00:00')),
                ('-A ', '-A ', resources.get('account', 'mendoza_q')),
                ('--account=', '--account=', resources.get('account', 'mendoza_q')),
                ('--mem-per-cpu=', '--mem-per-cpu=', resources.get('memory_per_cpu', '5G')),
                ('--mem=', '--mem=', resources.get('memory', '48G')),
            ]

            # Apply replacements line by line
            lines = content.split('\n')
            modified_lines = []

            for line in lines:
                # Check for SLURM directive replacements
                replaced = False

                # Handle script generator format: echo '#SBATCH ...' >> $1.sh
                if "echo '#SBATCH" in line or 'echo "#SBATCH' in line:
                    for short_flag, long_flag, new_value in slurm_mappings:
                        # Check if this line contains this SLURM flag
                        flag_pattern = f"#SBATCH {short_flag}" if not short_flag.startswith('--') else f"#SBATCH {short_flag}"
                        if flag_pattern in line:
                            # Preserve the echo format and quotes
                            if "echo '#SBATCH" in line:
                                # Single-quoted echo
                                new_line = f"echo '#SBATCH {long_flag}{new_value}' >> $1.sh"
                            else:
                                # Double-quoted echo
                                new_line = f'echo "#SBATCH {long_flag}{new_value}" >> $1.sh'
                            modified_lines.append(new_line)
                            replaced = True
                            break

                # Handle direct SLURM directive format: #SBATCH ...
                elif line.strip().startswith('#SBATCH'):
                    for short_flag, long_flag, new_value in slurm_mappings:
                        flag_pattern = f"#SBATCH {short_flag}"
                        if flag_pattern in line:
                            new_line = f"#SBATCH {long_flag}{new_value}"
                            modified_lines.append(new_line)
                            replaced = True
                            break

                if not replaced:
                    # Add workflow context environment variables after 'export JOB=' line
                    if line.startswith("echo 'export JOB="):
                        modified_lines.append(line)
                        # Add workflow context environment variables for script generators
                        workflow_id = script_config.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        modified_lines.append(f"echo '# Workflow context for queue manager' >> $1.sh")
                        modified_lines.append(f"echo 'export MACE_WORKFLOW_ID=\"{workflow_id}\"' >> $1.sh")
                        # Use absolute path for MACE_CONTEXT_DIR
                        context_dir = str(self.work_dir / f".mace_context_{workflow_id}")
                        modified_lines.append(f"echo 'export MACE_CONTEXT_DIR=\"{context_dir}\"' >> $1.sh")
                        modified_lines.append(f"echo 'export MACE_ISOLATION_MODE=\"{getattr(self, 'isolation_mode', 'isolated')}\"' >> $1.sh")
                    else:
                        modified_lines.append(line)
            
            content = '\n'.join(modified_lines)
        
        # Apply any additional customizations
        if 'customizations' in script_config:
            for customization in script_config['customizations']:
                if 'find' in customization and 'replace' in customization:
                    content = content.replace(customization['find'], customization['replace'])
        
        return content
        
    def copy_config_files(self, plan: Dict[str, Any], workflow_id: str):
        """Copy configuration files from temp to workflow_configs directory"""
        # Only copy if there are config files to copy
        if not self.temp_dir.exists():
            return
            
        config_files = list(self.temp_dir.glob(f"workflow_{workflow_id}_*.json"))
        if config_files:
            ui.info("  Copying configuration files...")
            for config_file in config_files:
                dest_file = self.configs_dir / config_file.name
                if not dest_file.exists():
                    shutil.copy2(config_file, dest_file)
                    ui.ok(f"  Copied {config_file.name}")
                    
        # Also save the CIF conversion config if present
        if plan.get('cif_conversion_config'):
            cif_config_file = self.configs_dir / "cif_conversion_config.json"
            if not cif_config_file.exists():
                with open(cif_config_file, 'w') as f:
                    json.dump(plan['cif_conversion_config'], f, indent=2)
                ui.ok("  Saved cif_conversion_config.json")
                
    def execute_workflow_plan(self, plan_file: Path):
        """Execute a complete workflow plan"""
        ui.info(f"Executing workflow plan: {plan_file}")
        
        with open(plan_file, 'r') as f:
            plan = json.load(f)
            
        # Use workflow_id from plan, or generate new one if missing
        workflow_id = plan.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ui.info(f"Using workflow_id: {workflow_id}")
        
        # Get isolation mode from plan (default to 'isolated' if not specified)
        isolation_mode = plan.get('isolation_mode', 'isolated')
        post_completion_action = plan.get('post_completion_action', 'keep')
        
        # Check if we should use workflow isolation
        if isolation_mode == 'shared':
            # Traditional behavior - no context needed
            ui.info(f"Using shared mode (no workflow isolation)")
            self._execute_workflow_plan_impl(plan, workflow_id, isolation_mode)
        else:
            # Use workflow context for isolation
            ui.info(f"Using {isolation_mode} isolation mode")
            ui.info(f"Post-completion action: {post_completion_action}")
            
            with workflow_context(workflow_id, base_dir=self.work_dir, isolation_mode=isolation_mode) as ctx:
                # Update database and queue manager paths to use context-specific ones
                original_db_path = self.db_path
                original_db = self.db
                original_queue_manager = self.queue_manager
                
                try:
                    # Create context-aware instances
                    self.db_path = str(ctx.get_database_path())
                    self.db = MaterialDatabase(self.db_path)
                    
                    # Get queue config from pending config or use defaults
                    queue_config = getattr(self, '_pending_queue_config', {
                        'max_jobs': 200,
                        'reserve_slots': 30,
                        'max_submit_batch': 5,
                        'max_recovery_attempts': 3
                    })

                    # Recreate queue manager with context paths and proper configuration
                    self.queue_manager = EnhancedCrystalQueueManager(
                        d12_dir=str(self.outputs_dir),
                        max_jobs=queue_config['max_jobs'],
                        reserve_slots=queue_config['reserve_slots'],
                        enable_tracking=True,
                        enable_error_recovery=True,
                        max_recovery_attempts=queue_config['max_recovery_attempts'],
                        db_path=self.db_path
                    )
                    self.queue_manager.max_submit_per_callback = queue_config['max_submit_batch']

                    ui.info(f"  Created isolated queue manager with max_jobs={queue_config['max_jobs']}")
                    
                    # Execute workflow within context
                    self._execute_workflow_plan_impl(plan, workflow_id, isolation_mode)
                    
                    # Handle post-completion actions
                    if post_completion_action == 'archive':
                        print(f"\nArchiving workflow context to: {self.work_dir / 'archived_workflows'}")
                        ctx.archive(self.work_dir / 'archived_workflows')
                    elif post_completion_action == 'export_and_delete':
                        print(f"\nExporting results and deleting context")
                        export_dir = self.work_dir / 'workflow_results' / workflow_id
                        ctx.export_results(export_dir)
                        # Context will be cleaned up automatically on exit
                    # 'keep' action requires no special handling
                    
                finally:
                    # Restore original instances
                    self.db_path = original_db_path
                    self.db = original_db
                    self.queue_manager = original_queue_manager

    def _configure_queue_manager_from_plan(self, plan: Dict[str, Any]):
        """Extract queue configuration from workflow plan and update queue manager"""
        # Extract queue configuration from plan — the planner writes
        # queue_management at the plan top level; keep the old nested
        # location as a fallback
        execution_settings = plan.get('execution_settings', {})
        queue_config = plan.get('queue_management') or execution_settings.get('queue_management', {})

        # Default values (conservative fallback)
        max_jobs = queue_config.get('max_jobs', 200)
        reserve_slots = queue_config.get('reserve_slots', 30)
        max_submit_batch = queue_config.get('max_submit_batch', 5)
        max_recovery_attempts = queue_config.get('max_recovery_attempts', 3)

        ui.info(f"Queue Configuration from Plan:")
        ui.info(f"  • Max jobs: {max_jobs}")
        ui.info(f"  • Reserve slots: {reserve_slots}")
        ui.info(f"  • Max submit batch: {max_submit_batch}")
        ui.info(f"  • Recovery attempts: {max_recovery_attempts}")

        # Recreate queue manager with correct settings if it exists
        if self.queue_manager:
            ui.info(f"  Updating existing queue manager configuration...")
            # Update the existing queue manager parameters
            self.queue_manager.max_jobs = max_jobs
            self.queue_manager.reserve_slots = reserve_slots
            self.queue_manager.max_submit_per_callback = max_submit_batch
            if hasattr(self.queue_manager, 'max_recovery_attempts'):
                self.queue_manager.max_recovery_attempts = max_recovery_attempts
        else:
            # If queue manager doesn't exist yet, store config for later creation
            self._pending_queue_config = {
                'max_jobs': max_jobs,
                'reserve_slots': reserve_slots,
                'max_submit_batch': max_submit_batch,
                'max_recovery_attempts': max_recovery_attempts
            }
            ui.info(f"  Queue manager not yet created, storing configuration for later...")

    def _validate_workflow_plan(self, plan: Dict[str, Any]) -> bool:
        """Comprehensive validation of workflow plan before execution"""
        ui.info("🔍 Validating workflow plan...")
        validation_errors = []

        # 1. Check for required plan components
        required_fields = ['workflow_sequence', 'input_directory', 'input_type']
        for field in required_fields:
            if field not in plan:
                validation_errors.append(f"Missing required field: {field}")

        # 2. Validate input directory and files
        input_dir = Path(plan.get('input_directory', ''))
        if not input_dir.exists():
            validation_errors.append(f"Input directory does not exist: {input_dir}")
        else:
            input_type = plan.get('input_type', 'cif')
            input_files = plan.get('input_files', {})

            if input_type == 'cif' and not input_files.get('cif', []):
                # Check if CIF files exist in directory
                cif_files = list(input_dir.glob("*.cif"))
                if not cif_files:
                    validation_errors.append(f"No CIF files found in {input_dir}")

            elif input_type == 'd12' and not input_files.get('d12', []):
                # Check if D12 files exist in directory
                d12_files = list(input_dir.glob("*.d12"))
                if not d12_files:
                    validation_errors.append(f"No D12 files found in {input_dir}")

        # 3. Validate workflow sequence
        workflow_sequence = plan.get('workflow_sequence', [])
        if not workflow_sequence:
            validation_errors.append("Empty workflow sequence")
        else:
            valid_calc_types = ['OPT', 'SP', 'FREQ', 'BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']
            for calc_type in workflow_sequence:
                # Planner legitimately emits numbered types (OPT2, SP2, BAND3...)
                base_type = calc_type.rstrip('0123456789')
                if base_type not in valid_calc_types:
                    validation_errors.append(f"Invalid calculation type: {calc_type}")

        # 4. Validate step configurations
        step_configs = plan.get('step_configurations', {})
        for step_key, config in step_configs.items():
            if not isinstance(config, dict):
                validation_errors.append(f"Invalid step configuration for {step_key}")

        # 5. Check execution settings
        execution_settings = plan.get('execution_settings', {})
        if 'max_concurrent_jobs' in execution_settings:
            max_jobs = execution_settings['max_concurrent_jobs']
            if not isinstance(max_jobs, int) or max_jobs <= 0:
                validation_errors.append(f"Invalid max_concurrent_jobs: {max_jobs}")

        # 6. Validate queue management configuration
        queue_config = plan.get('queue_management', {})
        if queue_config:
            max_jobs = queue_config.get('max_jobs', 200)
            if not isinstance(max_jobs, int) or max_jobs <= 0 or max_jobs > 1000:
                validation_errors.append(f"Invalid queue max_jobs: {max_jobs} (must be 1-1000)")

        # Report validation results
        if validation_errors:
            ui.err("❌ Validation failed with the following errors:")
            for error in validation_errors:
                ui.err(f"   • {error}")
            return False
        else:
            ui.ok("Workflow plan validation passed")
            return True

    def _apply_execution_settings(self, plan: Dict[str, Any]):
        """Apply execution settings from workflow plan"""
        execution_settings = plan.get('execution_settings', {})

        ui.info("🔧 Applying execution settings...")

        # Material tracking
        enable_tracking = execution_settings.get('enable_material_tracking', True)
        if hasattr(self, 'queue_manager') and self.queue_manager:
            # Update queue manager tracking settings
            ui.info(f"   • Material tracking: {'enabled' if enable_tracking else 'disabled'}")

        # Auto progression
        auto_progression = execution_settings.get('auto_progression', True)
        ui.info(f"   • Auto progression: {'enabled' if auto_progression else 'disabled'}")

        # Max concurrent jobs
        max_concurrent = execution_settings.get('max_concurrent_jobs', 200)
        ui.info(f"   • Max concurrent jobs: {max_concurrent}")

    def _execute_workflow_plan_impl(self, plan: Dict[str, Any], workflow_id: str, isolation_mode: str = 'isolated'):
        """Internal implementation of workflow execution"""
        # Store isolation mode for use in customize_slurm_script
        self.isolation_mode = isolation_mode

        # CRITICAL: Extract and apply queue configuration from workflow plan
        self._configure_queue_manager_from_plan(plan)

        # CRITICAL: Validate workflow plan and inputs before execution
        if not self._validate_workflow_plan(plan):
            ui.err("❌ Workflow validation failed. Aborting execution.")
            return

        # Apply execution settings from the plan
        self._apply_execution_settings(plan)

        try:
            # Phase 0: Recreate workflow scripts from configuration
            self.recreate_workflow_scripts(plan)
            
            # Phase 0.5: Copy configuration files
            self.copy_config_files(plan, workflow_id)
            
            # Phase 1: Prepare input files
            self.prepare_input_files(plan, workflow_id)
            
            # Phase 2: Execute calculation sequence
            # Always use the same method as interactive mode
            self.execute_workflow_steps(plan, workflow_id)
            
            # Phase 3: Monitor and manage execution
            if workflow_id in self.active_workflows:
                self.monitor_workflow_execution(workflow_id)
            else:
                ui.info("Phase 3: Workflow monitoring skipped (workflow completed synchronously)")
            
        except Exception as e:
            ui.err(f"Workflow execution failed: '{workflow_id}'")
            self.cleanup_workflow(workflow_id)
            raise
            
    def prepare_input_files(self, plan: Dict[str, Any], workflow_id: str):
        """Prepare all input files according to the plan"""
        ui.info("Phase 1: Preparing input files...")
        
        # Create workflow-specific directories
        workflow_dir = self.outputs_dir / workflow_id
        workflow_dir.mkdir(exist_ok=True)
        
        # Copy monitoring scripts to current working directory (where D12 files are)
        # This matches the interactive mode behavior
        self.setup_workflow_monitoring(self.work_dir)
        
        # Phase 1a: Convert CIFs if needed
        if plan['input_files']['cif']:
            ui.info("  Converting CIF files to D12 format...")
            self.convert_cifs_with_config(plan, workflow_id)
            
        # Phase 1b: Organize existing D12s
        if plan['input_files']['d12']:
            ui.info("  Organizing existing D12 files...")
            self.organize_existing_d12s(plan, workflow_id)
            
        # Phase 1c: Generate subsequent calculation inputs
        ui.info("  Pre-generating calculation configurations...")
        self.generate_calculation_configs(plan, workflow_id)
        
    def organize_existing_d12s(self, plan: Dict[str, Any], workflow_id: str):
        """Create workflow directory structure without copying D12 files"""
        workflow_dir = self.outputs_dir / workflow_id
        first_calc_type = plan.get('workflow_sequence', ['OPT'])[0]
        step_001_dir = workflow_dir / f"step_001_{first_calc_type}"
        step_001_dir.mkdir(parents=True, exist_ok=True)
        
        # Just create the directory structure - don't copy files
        # The execute_step method will copy files directly to individual material folders
        ui.info(f"    Created workflow directory structure for {len(plan.get('input_files', {}).get('d12', []))} D12 files")
        
    def generate_calculation_configs(self, plan: Dict[str, Any], workflow_id: str):
        """Generate configuration files for each workflow step"""
        step_configs = plan.get('step_configurations', {})
        
        for step_key, step_config in step_configs.items():
            # Create a configuration file for each step in workflow_configs
            step_configs_dir = self.configs_dir / "step_configs"
            step_configs_dir.mkdir(exist_ok=True)
            config_file = step_configs_dir / f"{workflow_id}_{step_key}_config.json"
            with open(config_file, 'w') as f:
                json.dump(step_config, f, indent=2)
            ui.info(f"    Generated config for {step_key}: {config_file.name}")
        
    def execute_workflow_steps(self, plan: Dict[str, Any], workflow_id: str):
        """Execute the complete workflow with proper calculation folder structure"""
        ui.info("Phase 2: Executing workflow calculations...")
        
        workflow_dir = self.outputs_dir / workflow_id
        workflow_sequence = plan['workflow_sequence']
        
        # Get initial D12 files (from CIF conversion or existing)
        # Check for the first calculation type in the sequence
        first_calc_type = workflow_sequence[0] if workflow_sequence else 'OPT'
        step_001_dir = workflow_dir / f"step_001_{first_calc_type}"
        
        # Get D12 files from their original location (execute mode) or step directory (interactive mode)
        d12_files = []
        
        # First check if files are in step directory (from CIF conversion or interactive mode)
        step_d12_files = list(step_001_dir.glob("*.d12"))
        if step_d12_files:
            d12_files = step_d12_files
            ui.info(f"  Using D12 files from step directory")
        # Check for generated D12s from CIF conversion
        elif 'generated_d12s' in plan:
            for d12_path in plan['generated_d12s']:
                d12_file = Path(d12_path)
                if d12_file.exists():
                    d12_files.append(d12_file)
            ui.info(f"  Using D12 files from CIF conversion")
        # Otherwise get them from original location (execute mode with existing D12s)
        elif 'input_files' in plan and 'd12' in plan['input_files']:
            for d12_path in plan['input_files']['d12']:
                d12_file = Path(d12_path)
                if d12_file.exists():
                    d12_files.append(d12_file)
                else:
                    # Try filename only in current directory
                    local_file = self.work_dir / d12_file.name
                    if local_file.exists():
                        d12_files.append(local_file)
            ui.info(f"  Using D12 files from original location")
        
        # Final fallback: For CIF workflows, check if D12s exist in the input directory
        # This handles the case where CIF conversion happened but generated_d12s wasn't saved
        if not d12_files and plan.get('input_type') == 'cif':
            input_dir = Path(plan.get('input_directory', self.work_dir))
            d12_files = list(input_dir.glob("*.d12"))
            if d12_files:
                ui.info(f"  Found D12 files from CIF conversion in input directory")
        
        if not d12_files:
            ui.err("Error: No D12 files found for workflow execution!")
            return
            
        ui.info(f"Found {len(d12_files)} materials to process")
        ui.info(f"Workflow sequence: {' → '.join(workflow_sequence)}")
        
        # Execute first step (always OPT)
        first_step = workflow_sequence[0]
        step_dir = workflow_dir / f"step_001_{first_step}"
        
        print(f"\nExecuting Step 1: {first_step}")
        submitted_jobs = self.execute_step(plan, workflow_id, first_step, 1, d12_files, step_dir)
        
        if submitted_jobs:
            ui.ok(f"Successfully submitted {len(submitted_jobs)} {first_step} calculations")
            ui.info("Monitor progress with:")
            print(f"  mace monitor --status")
            print(f"  squeue -u $USER")
        else:
            ui.warn(f"Warning: No {first_step} jobs were submitted")
            
    def execute_step(self, plan: Dict[str, Any], workflow_id: str, calc_type: str, 
                    step_num: int, d12_files: List[Path], step_dir: Path) -> List[str]:
        """Execute a single workflow step with individual calculation folders and database tracking"""
        submitted_jobs = []
        existing_materials = []  # Track material IDs already used in this workflow
        
        for d12_file in d12_files:
            # Extract core material ID for database consistency
            base_material_id = self.create_material_id_from_file(d12_file)
            
            # Make unique if duplicate
            material_id = self.make_unique_material_id(base_material_id, d12_file, existing_materials)
            existing_materials.append(material_id)
            
            if material_id != base_material_id:
                ui.info(f"  Note: Using unique ID '{material_id}' for duplicate material '{base_material_id}'")
            
            # Create individual calculation folder using material_id
            calc_dir = step_dir / material_id
            calc_dir.mkdir(exist_ok=True)
            
            # Copy D12 file to calculation folder with clean name (matching interactive mode behavior)
            # Create clean filename: material_id.d12
            clean_filename = f"{material_id}.d12"
            calc_d12_file = calc_dir / clean_filename
            if not calc_d12_file.exists():
                shutil.copy2(str(d12_file), str(calc_d12_file))
            
            # Submit via workflow-specific queue manager that respects our directory structure
            ui.info(f"  Submitting {material_id} via workflow queue manager...")
            calc_id = self.submit_workflow_calculation(
                d12_file=calc_d12_file,
                calc_type=calc_type,
                material_id=material_id,
                workflow_id=workflow_id,
                step_num=step_num,
                calc_dir=calc_dir
            )
            
            if calc_id:
                submitted_jobs.append(calc_id)
                ui.ok(f"  Submitted {material_id}: Job ID {calc_id}")
            else:
                ui.err(f"  Failed to submit {material_id}")
                
        return submitted_jobs
        
    def submit_workflow_calculation(self, d12_file: Path, calc_type: str, material_id: str,
                                   workflow_id: str, step_num: int, calc_dir: Path) -> str:
        """Submit calculation using workflow-specific directory structure and database tracking"""
        # Ensure database is initialized
        if self.db is None:
            # Initialize database if not already done (happens when default path was used)
            # In isolation mode, the context should have created the database by now
            if hasattr(self, 'isolation_mode') and self.isolation_mode == 'isolated':
                # In isolated mode, use the context database path
                context_dir = self.work_dir / f".mace_context_{workflow_id}"
                if context_dir.exists():
                    self.db_path = str(context_dir / "materials.db")
                    self.db = MaterialDatabase(self.db_path, auto_initialize=True)
                else:
                    raise RuntimeError(f"Context directory not found: {context_dir}")
            else:
                # In shared mode, create the default database
                self.db = MaterialDatabase(self.db_path, auto_initialize=True)
        
        # Create calculation record in database first
        calc_id = self.db.create_calculation(
            material_id=material_id,
            calc_type=calc_type,
            calc_subtype=None,
            input_file=str(d12_file),
            work_dir=str(calc_dir),
            settings={
                "workflow_id": workflow_id,
                "workflow_step": step_num,
                "workflow_calc_type": calc_type
            }
        )
        
        
        # Create workflow metadata file for callback tracking
        metadata = {
            'workflow_id': workflow_id,
            'step_num': step_num,
            'calc_type': calc_type,
            'material_id': material_id,
            'calc_id': calc_id
        }
        metadata_file = calc_dir / '.workflow_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        # Generate SLURM script in the calculation directory
        script_file = self.generate_individual_slurm_script(
            d12_file, calc_type, material_id, calc_dir, workflow_id, step_num
        )
        
        # Submit job
        job_id = self.submit_slurm_job(script_file, calc_dir)
        
        if job_id:
            # Update calculation with SLURM job ID
            self.db.update_calculation_status(
                calc_id, 'submitted', 
                slurm_job_id=job_id,
                output_file=str(calc_dir / f"{material_id}.out")
            )
            return job_id
        else:
            # Mark as failed
            self.db.update_calculation_status(calc_id, 'failed', error_message="SLURM submission failed")
            return None
            
    def generate_individual_slurm_script(self, d12_file: Path, calc_type: str, 
                                       material_id: str, calc_dir: Path, 
                                       workflow_id: str, step_num: int) -> Path:
        """Generate individual SLURM script for workflow calculation"""
        # Look for template scripts - first try workflow_scripts, then fallback to Job_Scripts
        scripts_dir = self.work_dir / "workflow_scripts"
        job_scripts_dir = Path(__file__).parent  # Job_Scripts directory
        
        # Determine template names
        base_type = calc_type.rstrip('0123456789')  # Remove trailing numbers
        
        if calc_type == 'OPT2' or base_type == 'OPT':
            specific_template = f"submitcrystal23_opt_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
        elif base_type in ['SP', 'FREQ']:
            specific_template = f"submitcrystal23_{base_type.lower()}_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
        elif base_type in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']:
            # All D3 property calculations use the same template
            specific_template = f"submit_prop_{calc_type.lower()}_{step_num}.sh"
            generic_template = "submit_prop.sh"
        else:
            specific_template = f"submit_{calc_type.lower()}_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
            
        # Try to find the template script
        template_script = None
        
        # First try specific template in workflow_scripts
        if scripts_dir.exists():
            specific_path = scripts_dir / specific_template
            if specific_path.exists():
                template_script = specific_path
        
        # If not found, try generic template in Job_Scripts
        if not template_script:
            generic_path = job_scripts_dir / generic_template
            if generic_path.exists():
                template_script = generic_path
        
        if not template_script or not template_script.exists():
            raise FileNotFoundError(f"Template script not found. Looked for {specific_template} in {scripts_dir} and {generic_template} in {job_scripts_dir}")
            
        # Read and customize template
        with open(template_script, 'r') as f:
            template_content = f.read()
            
        # Customize for this specific calculation
        script_content = self.customize_workflow_slurm_script(
            template_content, d12_file, calc_type, material_id, 
            calc_dir, workflow_id, step_num
        )
        
        # Write individual script
        script_file = calc_dir / f"{material_id}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        script_file.chmod(0o755)
        return script_file
        
    def customize_workflow_slurm_script(self, template_content: str, d12_file: Path,
                                      calc_type: str, material_id: str, calc_dir: Path,
                                      workflow_id: str, step_num: int) -> str:
        """Customize SLURM script for workflow calculation"""
        script_content = template_content
        
        # Replace job name
        script_content = script_content.replace(
            f'--job-name={calc_type.lower()}',
            f'--job-name={material_id}_{calc_type.lower()}'
        )
        
        # Replace output file
        script_content = script_content.replace(
            f'--output={calc_type.lower()}.o%j',
            f'--output={material_id}_{calc_type.lower()}.o%j'
        )
        
        # Set working directory to our calculation directory
        script_content = script_content.replace(
            'cd ${SLURM_SUBMIT_DIR}',
            f'cd {calc_dir}'
        )
        
        # Set up scratch directory
        scratch_dir = f"$SCRATCH/{workflow_id}/step_{step_num:03d}_{calc_type}/{material_id}"
        script_content = script_content.replace(
            'export scratch=$SCRATCH/crys23',
            f'export scratch={scratch_dir}'
        )
        script_content = script_content.replace(
            'export scratch=$SCRATCH/crys23/prop',
            f'export scratch={scratch_dir}'
        )
        
        # Add workflow context environment variables
        # Find the line with module loads and add after it
        module_pattern = r'(module\s+load[^\n]+\n)'
        context_dir = str(self.work_dir / f".mace_context_{workflow_id}")
        
        # Create the export statements
        context_exports = (
            '\n# Workflow context for queue manager callback\n'
            f'export MACE_WORKFLOW_ID="{workflow_id}"\n'
            f'export MACE_CONTEXT_DIR="{context_dir}"\n'
            f'export MACE_ISOLATION_MODE="isolated"\n\n'
        )
        
        if re.search(module_pattern, script_content):
            # Add after last module load
            script_content = re.sub(
                r'((?:module\s+load[^\n]+\n)+)',
                r'\1' + context_exports,
                script_content
            )
        else:
            # If no module loads found, add after shebang
            script_content = script_content.replace(
                '#!/bin/bash\n',
                '#!/bin/bash\n' + context_exports
            )
        
        # Replace file references
        script_content = script_content.replace('$1', material_id)
        
        # Fix the queue manager path for workflow structure
        # From material directory: ~/test/workflow_outputs/workflow_ID/step_XXX_TYPE/material_name/
        # To base directory: ~/test/ (4 levels up: ../../../../)
        #
        # Path breakdown:
        # 1. material_name/ → step_XXX_TYPE/ (1 level: ../)
        # 2. step_XXX_TYPE/ → workflow_ID/ (2 levels: ../../)  
        # 3. workflow_ID/ → workflow_outputs/ (3 levels: ../../../)
        # 4. workflow_outputs/ → test/ (4 levels: ../../../../)
        
        # The workflow templates already have correct multi-location callback logic
        # Add max-recovery-attempts parameter if not already present
        if '--max-recovery-attempts' not in script_content:
            script_content = script_content.replace(
                '--callback-mode completion',
                '--callback-mode completion --max-recovery-attempts 3'
            )
        
        return script_content
        
    def submit_slurm_job(self, script_file: Path, calc_dir: Path) -> str:
        """Submit SLURM job and return job ID"""
        original_cwd = os.getcwd()
        
        try:
            os.chdir(calc_dir)
            
            result = subprocess.run(
                ['sbatch', script_file.name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                for line in result.stdout.strip().split('\n'):
                    if 'Submitted batch job' in line:
                        return line.split()[-1]
            else:
                ui.err(f"SLURM submission failed: {result.stderr}")

        except Exception as e:
            ui.err(f"Error submitting job: {e}")

        finally:
            os.chdir(original_cwd)
            
        return None
        
    def extract_core_material_name(self, d12_file: Path) -> str:
        """Extract the core material name using smart suffix removal"""
        # Use the same logic as create_material_id_from_file for consistency
        return self.create_material_id_from_file(d12_file)
        
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
        
        # Look for template scripts - first try workflow_scripts, then fallback to Job_Scripts
        scripts_dir = self.work_dir / "workflow_scripts"
        job_scripts_dir = Path(__file__).parent  # Job_Scripts directory
        
        # Determine template names
        base_type = calc_type.rstrip('0123456789')  # Remove trailing numbers
        
        if calc_type == 'OPT2' or base_type == 'OPT':
            specific_template = f"submitcrystal23_opt_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
        elif base_type in ['SP', 'FREQ']:
            specific_template = f"submitcrystal23_{base_type.lower()}_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
        elif base_type in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']:
            # All D3 property calculations use the same template
            specific_template = f"submit_prop_{calc_type.lower()}_{step_num}.sh"
            generic_template = "submit_prop.sh"
        else:
            specific_template = f"submit_{calc_type.lower()}_{step_num}.sh"
            generic_template = "submitcrystal23.sh"
            
        # Try to find the template script
        template_script = None
        
        # First try specific template in workflow_scripts
        if scripts_dir.exists():
            specific_path = scripts_dir / specific_template
            if specific_path.exists():
                template_script = specific_path
        
        # If not found, try generic template in Job_Scripts
        if not template_script:
            generic_path = job_scripts_dir / generic_template
            if generic_path.exists():
                template_script = generic_path
        
        if not template_script or not template_script.exists():
            raise FileNotFoundError(f"No SLURM template found for {calc_type}. Looked for {specific_template} in {scripts_dir} and {generic_template} in {job_scripts_dir}")
            
        # Read template content
        with open(template_script, 'r') as f:
            template_content = f.read()

        # Apply SLURM resource customizations from step config if available
        if 'slurm_config' in step_config and 'scripts' in step_config['slurm_config']:
            # Find the matching script config for this calc type
            for script_name, script_config in step_config['slurm_config']['scripts'].items():
                if 'resources' in script_config:
                    script_config['workflow_id'] = workflow_id
                    template_content = self.apply_script_customizations(template_content, script_config)
                    break

        # Update template with specific values (workflow context, scratch dirs, etc.)
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
        
        # Add workflow context environment variables for queue manager context detection
        # Since these are script generators, we need to add echo commands
        context_export_commands = f'''echo '# Workflow context for queue manager' >> $1.sh
echo 'export MACE_WORKFLOW_ID="{workflow_id}"' >> $1.sh
echo 'export MACE_CONTEXT_DIR="{self.work_dir}/.mace_context_{workflow_id}"' >> $1.sh
echo 'export MACE_ISOLATION_MODE="{getattr(self, 'isolation_mode', 'isolated')}"' >> $1.sh
'''
        
        # For script generators, insert after the echo 'export JOB=' line
        if "echo 'export JOB=" in script_content:
            lines = script_content.split('\n')
            for i, line in enumerate(lines):
                if "echo 'export JOB=" in line:
                    # Insert context export echo commands after this line
                    lines.insert(i + 1, context_export_commands)
                    script_content = '\n'.join(lines)
                    break
        # For direct scripts (non-generators), use the original approach
        elif 'export JOB=' in script_content and 'echo' not in script_content:
            context_exports = f'''
# Workflow context for queue manager
export MACE_WORKFLOW_ID="{workflow_id}"
export MACE_CONTEXT_DIR="{self.work_dir}/.mace_context_{workflow_id}"
export MACE_ISOLATION_MODE="{getattr(self, 'isolation_mode', 'isolated')}"
'''
            lines = script_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('export JOB='):
                    lines.insert(i + 1, context_exports)
                    script_content = '\n'.join(lines)
                    break
        
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
    
    # Check if we're in a workflow context by looking for the context directory
    if [ ! -z "$MACE_CONTEXT_DIR" ] && [ -d "$MACE_CONTEXT_DIR" ]; then
        # In workflow context - pass the context database path
        CONTEXT_DB="$MACE_CONTEXT_DIR/materials.db"
        if [ -f "$CONTEXT_DB" ]; then
            echo "Using workflow context database: $CONTEXT_DB"
            python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3 --db-path "$CONTEXT_DB"
        else
            echo "Warning: Context database not found at $CONTEXT_DB, using default"
            python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
        fi
    else
        # Not in workflow context - use default behavior
        python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
    fi
else
    echo "Warning: Queue manager not found. Checked:"
    echo "  - \$MACE_HOME/mace/queue/manager.py"
    echo "  - Various relative paths from $DIR"
    echo "  Workflow progression may not continue automatically"
fi'''
        
        # Replace the existing callback section - handle both old and new formats
        import re
        
        # Check if this is a template script that generates another script
        if "echo" in script_content and ">> $1.sh" in script_content:
            # This is a template script like submitcrystal23.sh
            # We need to replace the callback line with proper echo commands
            
            # Convert our enhanced callback to echo format
            echo_callback_lines = []
            for line in enhanced_callback.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    # Escape special characters for echo
                    escaped_line = line.replace('$', '\\$').replace('"', '\\"').replace('`', '\\`')
                    echo_callback_lines.append(f"echo '{escaped_line}' >> $1.sh")
            
            enhanced_callback_echo = '\n'.join(echo_callback_lines)
            
            # Pattern for the original callback in template format
            callback_pattern = r"echo '# ADDED: Auto-submit new jobs when this one completes.*?enhanced_queue_manager\.py.*?' >> \$1\.sh"
            if re.search(callback_pattern, script_content, flags=re.DOTALL):
                script_content = re.sub(callback_pattern, enhanced_callback_echo, script_content, flags=re.DOTALL)
            else:
                # Try simpler pattern without echo
                callback_pattern2 = r"# ADDED: Auto-submit new jobs when this one completes.*?enhanced_queue_manager\.py.*"
                if re.search(callback_pattern2, script_content, flags=re.DOTALL):
                    # Replace with echo version
                    script_content = re.sub(callback_pattern2, enhanced_callback_echo, script_content, flags=re.DOTALL)
        else:
            # This is a regular script, not a template
            # First try the pattern with 'fi' at the end (for scripts that already have if-else logic)
            callback_pattern1 = r'# ADDED: Auto-submit new jobs when this one completes.*?fi'
            if re.search(callback_pattern1, script_content, flags=re.DOTALL):
                script_content = re.sub(callback_pattern1, enhanced_callback, script_content, flags=re.DOTALL)
            else:
                # Try simpler pattern
                callback_pattern2 = r"cd \$DIR\s*\n\s*enhanced_queue_manager\.py.*"
                if re.search(callback_pattern2, script_content):
                    script_content = re.sub(callback_pattern2, enhanced_callback.strip(), script_content)
        
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
                ui.info(f"  Running script generator locally: {script_file.name}")
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
                        ui.info(f"  Found generated script: {generated_script.name}")
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
                    ui.err(f"Error running script generator: {result.stderr}")
                    
            else:
                # This is a regular SLURM script - submit directly
                ui.info(f"  Submitting SLURM script directly: {script_file.name}")
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
                    ui.err(f"SLURM submission failed: {result.stderr}")

        except Exception as e:
            ui.err(f"Error submitting job: {e}")

        finally:
            os.chdir(original_cwd)
            
        return None
        
    def convert_cifs_with_config(self, plan: Dict[str, Any], workflow_id: str):
        """Convert CIF files using saved configuration"""
        cif_config = plan.get('cif_conversion_config')
        if not cif_config:
            raise ValueError("CIF conversion config not found in plan")
            
        workflow_dir = self.outputs_dir / workflow_id
        
        # Determine first calculation type from workflow sequence
        first_calc_type = plan.get('workflow_sequence', ['OPT'])[0]
        cif_output_dir = workflow_dir / f"step_001_{first_calc_type}"
        cif_output_dir.mkdir(parents=True, exist_ok=True)
        
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
            script_path = Path(__file__).parent.parent.parent / "Crystal_d12" / "NewCifToD12.py"
        
        conversion_cmd = [
            sys.executable, str(script_path),
            "--batch",
            "--options_file", str(cif_config_file),
            "--cif_dir", plan['input_directory'],
            "--output_dir", str(cif_output_dir)
        ]
        
        ui.info(f"    Running: {' '.join(conversion_cmd)}")
        ui.info(f"    Working directory: {os.getcwd()}")
        ui.info(f"    Script path exists: {script_path.exists()}")
        ui.info(f"    Config file exists: {cif_config_file.exists()}")
        ui.info(f"    Input directory exists: {Path(plan['input_directory']).exists()}")
        
        # Add timeout to prevent hanging
        try:
            result = subprocess.run(conversion_cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            ui.err("    CIF conversion timed out after 5 minutes")
            raise RuntimeError("CIF conversion timed out")

        ui.info(f"    Return code: {result.returncode}")
        if result.stdout:
            ui.info(f"    STDOUT: {result.stdout}")
        if result.stderr:
            ui.info(f"    STDERR: {result.stderr}")

        if result.returncode != 0:
            ui.err(f"CIF conversion failed:")
            raise RuntimeError("CIF conversion failed")

        ui.ok(f"    CIF conversion completed. Files in: {cif_output_dir}")
        
        # Update plan with generated D12 files
        generated_d12s = list(cif_output_dir.glob("*.d12"))
        plan['generated_d12s'] = [str(f) for f in generated_d12s]
        ui.info(f"    Generated {len(generated_d12s)} D12 files")
        
        # Save the updated plan back to disk so generated_d12s is persisted
        if 'workflow_id' in plan:
            updated_plan_file = self.configs_dir / f"workflow_plan_{plan['workflow_id']}_updated.json"
            with open(updated_plan_file, 'w') as f:
                json.dump(plan, f, indent=2)
            ui.info(f"    Updated plan saved with D12 paths")
        
    # DEPRECATED: Removed in refactoring - organize_existing_d12s_old()
    # This old method copied D12 files but is no longer used.
    # The new organize_existing_d12s() method keeps files in their original location.
        
    def generate_calculation_configs(self, plan: Dict[str, Any], workflow_id: str):
        """Pre-generate all calculation step configurations"""
        ui.info("  Generating calculation step configurations...")
        
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
        step_configs_dir = self.configs_dir / "step_configs"
        step_configs_dir.mkdir(exist_ok=True)
        config_file = step_configs_dir / f"{workflow_id}_step_{step_num:03d}_{calc_type}_config.json"
        
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
            
        ui.info(f"    Generated config for step {step_num}: {calc_type}")
        
    def execute_calculation_sequence(self, plan: Dict[str, Any], workflow_id: str):
        """Execute the planned calculation sequence"""
        ui.info("Phase 2: Executing calculation sequence...")
        
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
            ui.warn("No calculations in workflow sequence")
            return
            
        first_calc_type = sequence[0]
        ui.info(f"  Submitting initial {first_calc_type} calculations...")
        
        # Get input files for first step
        input_files = []
        
        # First check if there are D12 files in the step directory (created by workflow planner)
        step_num = 1  # First step is always step 1
        step_dir = self.outputs_dir / workflow_id / f"step_{step_num:03d}_{first_calc_type}"
        if step_dir.exists():
            step_d12_files = list(step_dir.glob("*.d12"))
            if step_d12_files:
                input_files = step_d12_files
                ui.info(f"  Found {len(input_files)} D12 files in step directory: {step_dir}")
        
        # If no files in step directory, check the plan for input file locations
        if not input_files:
            if 'generated_d12s' in plan:
                input_files = [Path(f) for f in plan['generated_d12s']]
                ui.info(f"  Using generated D12 files: {len(input_files)} files")
            elif 'organized_d12s' in plan:
                input_files = [Path(f) for f in plan['organized_d12s']]
                ui.info(f"  Using organized D12 files: {len(input_files)} files")
            elif 'input_files' in plan and 'd12' in plan['input_files']:
                # This is the correct key for workflow planner generated plans
                # Don't copy files - use them from their original location
                raw_files = [Path(f) for f in plan['input_files']['d12']]
                input_files = []
                for d12_file in raw_files:
                    if d12_file.exists():
                        input_files.append(d12_file)
                    else:
                        # Try filename only in current directory
                        local_file = self.work_dir / d12_file.name
                        if local_file.exists():
                            input_files.append(local_file)
                        else:
                            ui.warn(f"    Warning: Source file not found: {d12_file}")
                
                ui.info(f"  Using input D12 files from plan: {len(input_files)} files")
                        
            elif 'input_directory' in plan:
                # Look for D12 files in the input directory
                input_dir = Path(plan['input_directory'])
                input_files = list(input_dir.glob("*.d12"))
                ui.info(f"  Found {len(input_files)} D12 files in input directory: {input_dir}")

                if not input_files:
                    ui.err(f"Error: No D12 files found for workflow execution in {input_dir}!")
                    return
        
        if not input_files:
            ui.err("Error: No input files found for workflow execution!")
            return
        
        # Don't rename files - work with them as they are
        # The execute_step method will handle copying to individual material folders
            
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
        
        ui.ok(f"  Submitted {len(submitted_jobs)} {first_calc_type} calculations")
        
    def submit_single_calculation(self, input_file: Path, calc_type: str, 
                                workflow_id: str, step_num: int) -> Optional[str]:
        """Submit a single calculation"""
        try:
            # Create material ID from file
            material_id = self.create_material_id_from_file(input_file)
            ui.info(f"  Created material_id: {material_id} from {input_file.name}")
            
            # Check if this is a D3 calculation
            is_d3_calc = calc_type.rstrip('0123456789') in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']
            
            if is_d3_calc:
                # For D3 calculations, we need to handle submission differently
                # The queue manager expects d12_file, but we can pass the d3 file
                # The submit_prop.sh script will handle it correctly
                calc_id = self.queue_manager.submit_calculation(
                    d12_file=input_file,  # Pass D3 file but keep parameter name for compatibility
                    calc_type=calc_type,
                    material_id=material_id
                )
            else:
                # Submit via queue manager normally for D12 files
                calc_id = self.queue_manager.submit_calculation(
                    d12_file=input_file,
                    calc_type=calc_type,
                    material_id=material_id
                )
            ui.info(f"  Queue manager returned calc_id: {calc_id}")
            
            if calc_id:
                # Store workflow context in database
                ui.info(f"  Updating calculation settings with workflow_id: {workflow_id}")
                self.db.update_calculation_settings(calc_id, {
                    "workflow_id": workflow_id,
                    "workflow_step": step_num,
                    "workflow_calc_type": calc_type
                })
                ui.ok(f"  Successfully updated workflow metadata for calc_id: {calc_id}")
            else:
                ui.warn(f"  Warning: No calc_id returned from queue manager for {material_id}")
                
            return calc_id
            
        except Exception as e:
            ui.err(f"Failed to submit calculation for {input_file}: {e}")
            return None
            
    def create_material_id_from_file(self, file_path: Path) -> str:
        """Create a material ID using the canonical database derivation.

        The executor previously kept workflow filenames as-is ("1_dia_opt"),
        while the queue manager, populate_completed_jobs scan, and the
        workflow engine all derive the canonical ID ("1_dia"). That split the
        same material across two IDs: the submitted record never matched the
        completion scan, every completed job was re-registered as a duplicate
        material, and follow-up steps hung off the duplicate.
        """
        from mace.database.materials import create_material_id_from_file as canonical_material_id
        return canonical_material_id(file_path)
        
    def extract_functional_from_filename(self, file_path: Path) -> str:
        """Extract DFT functional from filename for duplicate differentiation"""
        name = file_path.stem
        parts = name.split('_')
        
        # Look for functional names in the filename
        functionals = ['PBE', 'B3LYP', 'HSE06', 'PBE0', 'SCAN', 'BLYP', 'BP86', 'M06', 'TPSS', 'LDA']
        
        for part in parts:
            # Check direct functional match
            if part.upper() in functionals:
                # Check if it has dispersion correction
                if 'D3' in name.upper():
                    return f"{part.upper()}-D3"
                elif 'D2' in name.upper():
                    return f"{part.upper()}-D2"
                else:
                    return part.upper()
            # Check for M06 variants
            elif part.upper().startswith('M06'):
                if 'D3' in name.upper():
                    return f"{part.upper()}-D3"
                else:
                    return part.upper()
                    
        # No functional found in filename
        return None
        
    def make_unique_material_id(self, base_material_id: str, d12_file: Path, 
                               existing_materials: List[str]) -> str:
        """
        Create a unique material ID by adding functional-based suffix if needed.
        
        Args:
            base_material_id: The base material identifier
            d12_file: The D12 file path to extract functional info from
            existing_materials: List of already used material IDs
            
        Returns:
            Unique material ID, possibly with functional suffix
        """
        # If base ID is not in existing materials, use it as-is
        if base_material_id not in existing_materials:
            return base_material_id
            
        # Try to extract functional from filename
        functional = self.extract_functional_from_filename(d12_file)
        
        if functional:
            # Try functional-based suffix first
            unique_id = f"{base_material_id}_{functional}"
            if unique_id not in existing_materials:
                return unique_id
                
            # If that's taken too, add a counter
            counter = 2
            while f"{unique_id}_{counter}" in existing_materials:
                counter += 1
            return f"{unique_id}_{counter}"
        else:
            # No functional found, use generic counter
            counter = 2
            while f"{base_material_id}_{counter}" in existing_materials:
                counter += 1
            return f"{base_material_id}_{counter}"
        
    def monitor_workflow_execution(self, workflow_id: str):
        """Monitor and manage workflow execution"""
        ui.info("Phase 3: Monitoring workflow execution...")
        
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        sequence = plan['workflow_sequence']
        
        ui.info(f"  Workflow: {' → '.join(sequence)}")
        ui.info(f"  Monitoring workflow {workflow_id}...")
        ui.info(f"  Use 'mace monitor --status' for detailed job status")
        
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
                ui.err(f"Error in workflow monitoring: {e}")
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
            ui.ok(f"Step {step_key} completed successfully ({completed_jobs}/{total_jobs})")
            return True
        elif failed_jobs > 0:
            ui.warn(f"Step {step_key} has failures ({failed_jobs}/{total_jobs} failed)")
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
        ui.info(f"Advancing to step {next_step + 1}: {next_calc_type}")
        
        # Generate input files for next step
        self.generate_next_step_inputs(workflow_id, next_step, next_calc_type)
        
        # Submit next step calculations
        self.submit_next_step_calculations(workflow_id, next_step, next_calc_type)
        
        workflow_info["current_step"] = next_step
        
    def generate_next_step_inputs(self, workflow_id: str, step_num: int, calc_type: str):
        """Generate input files for the next calculation step"""
        ui.info(f"  Generating inputs for step {step_num + 1}: {calc_type}")
        
        workflow_info = self.active_workflows[workflow_id]
        plan = workflow_info["plan"]
        
        # Get configuration for this step
        step_configs_dir = self.configs_dir / "step_configs"
        config_file = step_configs_dir / f"{workflow_id}_step_{step_num + 1:03d}_{calc_type}_config.json"
        
        if not config_file.exists():
            ui.warn(f"    Warning: No config file found for step {step_num + 1}")
            return
            
        with open(config_file, 'r') as f:
            step_config = json.load(f)
            
        source = step_config.get('source', 'unknown')
        
        if source == "CRYSTALOptToD12.py":
            if step_config.get("d3_calculation", False):
                # This is a D3 calculation using CRYSTALOptToD3.py
                self.generate_inputs_with_crystal_opt_d3(workflow_id, step_num, calc_type, step_config)
            else:
                # Regular D12 calculation
                self.generate_inputs_with_crystal_opt(workflow_id, step_num, calc_type, step_config)
        elif source in ["create_band_d3.py", "alldos.py"]:
            # Legacy scripts - no longer supported, use CRYSTALOptToD3.py instead
            ui.err(f"    Error: Legacy script {source} is no longer supported.")
            ui.err(f"    Please use CRYSTALOptToD3.py for {calc_type} calculations.")
            # Try to use CRYSTALOptToD3.py as a fallback
            if calc_type.rstrip('0123456789') in ['BAND', 'DOSS']:
                ui.info(f"    Attempting to use CRYSTALOptToD3.py instead...")
                step_config['d3_calculation'] = True
                self.generate_inputs_with_crystal_opt_d3(workflow_id, step_num, calc_type, step_config)
        else:
            ui.err(f"    Unknown input generation method: {source}")
            
    def generate_inputs_with_crystal_opt(self, workflow_id: str, step_num: int, 
                                       calc_type: str, config: Dict[str, Any]):
        """Generate inputs using CRYSTALOptToD12.py"""
        ui.info(f"    Using CRYSTALOptToD12.py for {calc_type} inputs")
        
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
            script_path = Path(__file__).parent.parent.parent / "Crystal_d12" / "CRYSTALOptToD12.py"
        
        # Check if expert mode was already configured during planning
        if config.get("expert_mode", False) and config.get("crystal_opt_config"):
            ui.info(f"      Using expert configuration from planning phase for {calc_type}")
            # Use the saved configuration from planning phase
            saved_config = config.get("crystal_opt_config", {})
            
            # Create a temporary config file with the saved settings
            temp_config = self.temp_dir / f"expert_{calc_type.lower()}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(temp_config, 'w') as f:
                json.dump(saved_config, f, indent=2)
            
            # Run CRYSTALOptToD12.py with the saved configuration
            cmd = [
                sys.executable, str(script_path),
                "--out-file", output_file,
                "--d12-file", input_file,
                "--output-dir", str(output_dir),
                "--options-file", str(temp_config)
            ]
            
            try:
                ui.info(f"      Running CRYSTALOptToD12.py with expert configuration...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    ui.ok(f"      Successfully generated {calc_type} input with expert settings")
                    
                    # Fix naming for OPT2 files
                    if calc_type == "OPT2":
                        self._fix_opt2_naming(output_dir, Path(output_file).stem)
                else:
                    ui.err(f"      Error generating {calc_type} input: {result.stderr}")
            except Exception as e:
                ui.err(f"      Error running with expert config: {e}")
            
            # Clean up temp config
            if temp_config.exists():
                temp_config.unlink()
            return
            
        # Check if this should run interactively (fallback for old configs)
        elif config.get("run_interactive", False) or config.get("interactive", False):
            ui.info(f"      Running CRYSTALOptToD12.py interactively for expert {calc_type} configuration")
            
            # Run CRYSTALOptToD12.py interactively
            cmd = [
                sys.executable, str(script_path),
                "--out-file", output_file,
                "--d12-file", input_file,
                "--output-dir", str(output_dir),
                "--calc-type", calc_type if calc_type != "OPT2" else "OPT"
            ]
            
            try:
                # Run interactively (no capture_output so user can interact)
                ui.info(f"      Launching interactive configuration...")
                ui.info(f"      Command: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    ui.ok(f"      Successfully generated {calc_type} input interactively")
                    
                    # Fix naming for OPT2 files
                    if calc_type == "OPT2":
                        self._fix_opt2_naming(output_dir, Path(output_file).stem)
                else:
                    ui.err(f"      Interactive {calc_type} generation failed or was cancelled")
            except Exception as e:
                ui.err(f"      Error running interactive mode: {e}")
            return
        
        # Non-interactive mode (batch with config file)
        # Create temporary config for CRYSTALOptToD12.py
        temp_config = self.temp_dir / f"temp_crystal_opt_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Ensure we pass the correct calculation type (OPT for OPT2, not SP)
        actual_calc_type = "OPT" if calc_type == "OPT2" else calc_type
        
        crystal_opt_config = {
            "calculation_type": actual_calc_type,
            "keep_current_settings": config.get("inherit_settings", True)
        }

        # CRITICAL: Extract and preserve external basis set information from input file
        if Path(input_file).exists():
            try:
                # Import the D12 parser to extract basis set info
                sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
                from d12_parsers import CrystalInputParser

                parser = CrystalInputParser(input_file)
                d12_data = parser.parse()

                # Check if this was an external basis set calculation
                if d12_data.get("basis_set_type") == "EXTERNAL" or self._has_external_basis(input_file):
                    ui.info(f"      Detected external basis set in {Path(input_file).name}")
                    crystal_opt_config["basis_set_type"] = "EXTERNAL"

                    # Try to determine the basis set path from the input file
                    basis_path = self._extract_basis_path_from_d12(input_file)
                    if basis_path == "USE_ORIGINAL_EXTERNAL_BASIS":
                        # Use the original external basis data from the D12 file
                        crystal_opt_config["use_original_external_basis"] = True
                        ui.info(f"      Using original external basis data from {Path(input_file).name}")
                    elif basis_path:
                        crystal_opt_config["basis_set_path"] = basis_path
                        crystal_opt_config["use_original_external_basis"] = False
                        ui.info(f"      Using external basis from: {basis_path}")
                    else:
                        # Fallback: try to use original basis data if possible
                        crystal_opt_config["use_original_external_basis"] = True
                        ui.warn(f"      Warning: External basis detected but path not found, using original basis data")

                elif d12_data.get("basis_set"):
                    # Internal basis set - preserve the specific basis set name
                    crystal_opt_config["basis_set_type"] = "INTERNAL"
                    crystal_opt_config["basis_set"] = d12_data.get("basis_set")
                    ui.info(f"      Preserving internal basis set: {d12_data.get('basis_set')}")

            except Exception as e:
                ui.warn(f"      Warning: Could not extract basis set info from {input_file}: {e}")
                ui.warn(f"      Falling back to keep_current_settings")
        
        # Add optimization-specific settings if provided
        if calc_type in ["OPT", "OPT2"] and "optimization_settings" in config:
            crystal_opt_config["optimization_settings"] = config["optimization_settings"]
        if calc_type in ["OPT", "OPT2"] and "optimization_type" in config:
            crystal_opt_config["optimization_type"] = config["optimization_type"]
            
        # Add frequency-specific settings if provided
        if calc_type == "FREQ" and "frequency_settings" in config:
            crystal_opt_config["frequency_settings"] = config["frequency_settings"]
            # Pass custom tolerances if defined
            if "custom_tolerances" in config.get("frequency_settings", {}):
                crystal_opt_config["custom_tolerances"] = config["frequency_settings"]["custom_tolerances"]
        
        # Add modifications for SP and other calculation types
        if "tolerance_modifications" in config:
            crystal_opt_config["tolerance_modifications"] = config["tolerance_modifications"]
        if "method_modifications" in config:
            crystal_opt_config["method_modifications"] = config["method_modifications"]
        if "basis_modifications" in config:
            crystal_opt_config["basis_modifications"] = config["basis_modifications"]
        
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
                ui.info(f"      Generated {calc_type} input from {Path(output_file).name}")

                # Fix naming for OPT2 files (CRYSTALOptToD12.py generates files with _opt suffix)
                if calc_type == "OPT2":
                    self._fix_opt2_naming(output_dir, Path(output_file).stem)
            else:
                ui.err(f"      Failed to generate {calc_type} input: {result.stderr}")
        except subprocess.TimeoutExpired:
            ui.err(f"      Timeout generating {calc_type} input")
        finally:
            if temp_config.exists():
                temp_config.unlink()

    def _has_external_basis(self, d12_file: str) -> bool:
        """Check if a D12 file contains external basis set data"""
        try:
            with open(d12_file, 'r') as f:
                content = f.read()
                # Look for typical external basis indicators
                # External basis sets have numeric data patterns and "99 0" terminator
                return "99 0" in content and any(
                    line.strip().startswith(('0 0 ', '0 2 ', '0 3 '))
                    for line in content.split('\n')
                )
        except Exception:
            return False

    def _extract_basis_path_from_d12(self, d12_file: str) -> Optional[str]:
        """Extract basis set path from D12 filename for external basis sets"""
        # Extract basis set path from filename patterns like:
        # material_CRYSTAL_OPT_symm_HSE06-D3_full.basis.triplezeta.d12
        filename = Path(d12_file).name

        # Explicit per-site override wins (set by the user); avoids hardcoding any
        # one institution's basis directory. MACE_TZ_BASIS_PATH / MACE_DZ_BASIS_PATH
        # point directly at a tz/dz dir; MACE_BASIS_DIR is a parent holding both.
        basis_dir = os.environ.get("MACE_BASIS_DIR")
        if "full.basis.triplezeta" in filename:
            for cand in (os.environ.get("MACE_TZ_BASIS_PATH"),
                         os.path.join(basis_dir, "full.basis.triplezeta") if basis_dir else None):
                if cand and Path(cand).exists():
                    return cand
        elif "full.basis.doublezeta" in filename:
            for cand in (os.environ.get("MACE_DZ_BASIS_PATH"),
                         os.path.join(basis_dir, "full.basis.doublezeta") if basis_dir else None):
                if cand and Path(cand).exists():
                    return cand

        # Next, try MACE config for bundled basis sets that ship with the repo.
        try:
            # mace_config.py lives at the repo root: mace/workflow/ -> mace/ ->
            # repo root is three parents up (the old parent.parent pointed at
            # mace/ so this block silently never ran -> bundled sets were unused).
            config_path = Path(__file__).parent.parent.parent / "mace_config.py"
            if config_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("mace_config", config_path)
                mace_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mace_config)

                # Check for bundled basis set indicators in filename
                if any(tz_indicator in filename.lower() for tz_indicator in ['full.basis.triplezeta', 'triplezeta', 'tz', 'tzvp']):
                    bundled_tz_path = getattr(mace_config, 'DEFAULT_TZ_PATH', None)
                    if bundled_tz_path and Path(bundled_tz_path).exists():
                        return bundled_tz_path

                elif any(dz_indicator in filename.lower() for dz_indicator in ['full.basis.doublezeta', 'doublezeta', 'dz', 'dzvp']):
                    bundled_dz_path = getattr(mace_config, 'DEFAULT_DZ_PATH', None)
                    if bundled_dz_path and Path(bundled_dz_path).exists():
                        return bundled_dz_path
        except Exception:
            pass

        # Fallback: Look for common basis set directory patterns in filename
        # This handles cases where custom paths were used

        # Legacy HPC paths (for backward compatibility). The institutional path
        # below is MSU/Mendoza-group specific and simply won't exist off-cluster
        # (set MACE_*_BASIS_PATH above for other sites); ./ and ../ are last-resort.
        if "full.basis.triplezeta" in filename:
            legacy_paths = [
                "/mnt/research/mendozacortes_group/bin/helpscripts/code/full.basis.triplezeta/",
                "./full.basis.triplezeta/",
                "../full.basis.triplezeta/"
            ]
            for path in legacy_paths:
                if Path(path).exists():
                    return path
        elif "full.basis.doublezeta" in filename:
            legacy_paths = [
                "/mnt/research/mendozacortes_group/bin/helpscripts/code/full.basis.doublezeta/",
                "./full.basis.doublezeta/",
                "../full.basis.doublezeta/"
            ]
            for path in legacy_paths:
                if Path(path).exists():
                    return path

        # If we can't determine the exact path, try to parse it from the D12 file itself
        # by looking for basis set content and trying to match it to known locations
        try:
            with open(d12_file, 'r') as f:
                content = f.read()

            # If it has external basis content, we should preserve it even if we can't
            # determine the exact original path. CRYSTALOptToD12.py can use the
            # original external basis data from the file itself.
            if "99 0" in content and any(line.strip().startswith(('0 0 ', '0 2 ', '0 3 ')) for line in content.split('\n')):
                # Return a special marker indicating we should use original basis data
                return "USE_ORIGINAL_EXTERNAL_BASIS"
        except Exception:
            pass

        return None

    def _fix_opt2_naming(self, output_dir: Path, base_name: str):
        """Fix naming for OPT2 files generated by CRYSTALOptToD12.py"""
        # CRYSTALOptToD12.py might generate files with patterns like material_opt_opt.d12
        # We want to rename them to material_opt2.d12
        
        for file_path in output_dir.glob("*.d12"):
            if "_opt_opt" in file_path.name or "_opt.d12" in file_path.name:
                # Extract material name
                material_name = file_path.stem
                if material_name.endswith("_opt"):
                    material_name = material_name[:-4]  # Remove _opt suffix
                    
                # Create new name with opt2
                new_name = f"{material_name}_opt2.d12"
                new_path = output_dir / new_name
                
                if not new_path.exists():
                    file_path.rename(new_path)
                    ui.info(f"        Renamed: {file_path.name} → {new_name}")
    
    def _fix_d3_numbered_naming(self, output_dir: Path, base_name: str, calc_type: str):
        """Fix naming for numbered D3 files (BAND2, DOSS2, etc.) generated by CRYSTALOptToD3.py"""
        # Extract the base calculation type and number
        import re
        match = re.match(r'^([A-Z+]+)(\d*)$', calc_type)
        if not match:
            return
            
        base_calc_type = match.group(1)
        instance_num = match.group(2)
        
        # If no instance number (first instance), no renaming needed
        if not instance_num:
            return
            
        # CRYSTALOptToD3.py generates files like material_band.d3, material_doss.d3
        # We need to rename them to material_band2.d3, material_doss2.d3, etc.
        
        # Handle CHARGE+POTENTIAL special case
        if base_calc_type == "CHARGE+POTENTIAL":
            file_suffix = "charge+potential"
        else:
            file_suffix = base_calc_type.lower()
        
        # Find and rename D3 files
        for file_path in output_dir.glob("*.d3"):
            if f"_{file_suffix}.d3" in file_path.name:
                # Extract the base name from the file
                file_base = file_path.stem.replace(f"_{file_suffix}", "")
                
                # Create new name with instance number
                new_name = f"{file_base}_{file_suffix}{instance_num}.d3"
                new_path = output_dir / new_name
                
                if not new_path.exists():
                    file_path.rename(new_path)
                    ui.info(f"        Renamed: {file_path.name} → {new_name}")
                    
        # Also rename corresponding f9 files
        for file_path in output_dir.glob("*.f9"):
            if f"_{file_suffix}.f9" in file_path.name:
                # Extract the base name from the file
                file_base = file_path.stem.replace(f"_{file_suffix}", "")
                
                # Create new name with instance number
                new_name = f"{file_base}_{file_suffix}{instance_num}.f9"
                new_path = output_dir / new_name
                
                if not new_path.exists():
                    file_path.rename(new_path)
                    ui.info(f"        Renamed: {file_path.name} → {new_name}")
    
    def generate_inputs_with_crystal_opt_d3(self, workflow_id: str, step_num: int,
                                           calc_type: str, config: Dict[str, Any]):
        """Generate D3 inputs using CRYSTALOptToD3.py"""
        ui.info(f"    Using CRYSTALOptToD3.py for {calc_type} D3 inputs")
        
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
                
                if output_file and Path(output_file).exists():
                    self.run_crystal_opt_d3_conversion(
                        output_file, next_step_dir, calc_type, config
                    )
    
    def run_crystal_opt_d3_conversion(self, output_file: str, output_dir: Path,
                                     calc_type: str, config: Dict[str, Any]):
        """Run CRYSTALOptToD3.py conversion"""
        # Check for local copy first
        local_script_path = self.work_dir / "CRYSTALOptToD3.py"
        if local_script_path.exists():
            script_path = local_script_path
        else:
            script_path = Path(__file__).parent.parent.parent / "Crystal_d3" / "CRYSTALOptToD3.py"
        
        # Handle different configuration modes
        d3_config_mode = config.get("d3_config_mode", "basic")
        
        if d3_config_mode == "expert" and config.get("interactive_setup", False):
            # Expert mode - run interactively and save configuration
            ui.info(f"      Running CRYSTALOptToD3.py interactively for expert {calc_type} configuration")
            
            # Strip instance numbers for CRYSTALOptToD3.py compatibility (BAND2 -> BAND)
            base_calc_type = re.sub(r'\d+$', '', calc_type)
            
            # Get material name from output file
            material_name = Path(output_file).stem
            if material_name.endswith('_opt'):
                material_name = material_name[:-4]
            elif material_name.endswith('_sp'):
                material_name = material_name[:-3]
                
            # Create per-material config file path
            config_dir = self.work_dir / "workflow_configs" / f"expert_{calc_type.lower()}_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            material_config_file = config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
            
            cmd = [
                sys.executable, str(script_path),
                "--input", output_file,
                "--output-dir", str(output_dir),
                "--calc-type", base_calc_type,
                "--save-config",  # Save the configuration
                "--options-file", str(material_config_file)  # Save to per-material file
            ]
            
            try:
                # Run interactively (no capture_output so user can interact)
                ui.info(f"      Launching interactive configuration...")
                ui.info(f"      Configuration will be saved to: {material_config_file.name}")
                ui.info(f"      Command: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    ui.ok(f"      Successfully generated {calc_type} D3 input interactively")
                    if material_config_file.exists():
                        ui.ok(f"      Saved configuration to {material_config_file}")
                else:
                    ui.err(f"      Interactive {calc_type} generation failed or was cancelled")
            except Exception as e:
                ui.err(f"      Error running interactive mode: {e}")
                
        else:
            # Basic or Advanced mode - use configuration file
            d3_config = config.get("d3_config", {})
            
            # Create temporary config file
            temp_config = self.temp_dir / f"temp_d3_{calc_type.lower()}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare configuration for CRYSTALOptToD3.py
            # Strip instance numbers for compatibility (BAND2 -> BAND)
            base_calc_type = re.sub(r'\d+$', '', calc_type)
            
            d3_json_config = {
                "version": "1.0",
                "type": "d3_configuration",
                "calculation_type": base_calc_type,
                "configuration": d3_config
            }
            
            with open(temp_config, 'w') as f:
                json.dump(d3_json_config, f, indent=2)
            
            # Run CRYSTALOptToD3.py with config file
            cmd = [
                sys.executable, str(script_path),
                "--input", output_file,
                "--output-dir", str(output_dir),
                "--config-file", str(temp_config)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    ui.info(f"      Generated {calc_type} D3 input from {Path(output_file).name}")
                    
                    # Fix D3 file naming for numbered instances (BAND2, DOSS2, etc.)
                    base_name = Path(output_file).stem
                    self._fix_d3_numbered_naming(output_dir, base_name, calc_type)
                else:
                    ui.err(f"      Failed to generate {calc_type} D3 input: {result.stderr}")
            except subprocess.TimeoutExpired:
                ui.err(f"      Timeout generating {calc_type} D3 input")
            except Exception as e:
                ui.err(f"      Error running CRYSTALOptToD3.py: {e}")
            finally:
                if temp_config.exists():
                    temp_config.unlink()
                    
    # DEPRECATED: Removed in refactoring - generate_inputs_with_analysis_script()
    # This stub function was never implemented and referenced old scripts (create_band_d3.py, alldos.py)
    # that have been replaced by CRYSTALOptToD3.py
        
    def submit_next_step_calculations(self, workflow_id: str, step_num: int, calc_type: str):
        """Submit calculations for the next step"""
        workflow_dir = self.outputs_dir / workflow_id
        step_dir = workflow_dir / f"step_{step_num + 1:03d}_{calc_type}"
        
        if not step_dir.exists():
            ui.warn(f"    No step directory found: {step_dir}")
            return
            
        # Find input files for this step
        # D3 calculations (BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL) use .d3 files
        if calc_type.rstrip('0123456789') in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']:
            input_files = list(step_dir.glob("*.d3"))
            file_type = "D3"
        else:
            input_files = list(step_dir.glob("*.d12"))
            file_type = "D12"
        
        if not input_files:
            ui.warn(f"    No {file_type} input files found in {step_dir}")
            return

        ui.info(f"    Submitting {len(input_files)} {calc_type} calculations")
        
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
        ui.ok(f"Workflow {workflow_id} completed!")
        
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
            
        ui.ok(f"Workflow report saved: {report_file}")
        
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
                
    def setup_workflow_monitoring(self, workflow_dir: Path):
        """Create monitoring documentation for workflow directory."""
        try:
            ui.info("  Creating workflow monitoring guide...")
            
            # Create monitoring documentation instead of copying scripts
            readme_content = """# Workflow Monitoring Guide

This workflow is managed by MACE. Use these commands to monitor your workflow:

## Quick Commands

### Check Workflow Status
```bash
# View status of all calculations
mace monitor --status

# Live monitoring dashboard
mace monitor
# or
mace monitor --dashboard
# Press Ctrl+C to stop
```

### Check SLURM Queue
```bash
# View your running jobs
squeue -u $USER

# View jobs for this workflow
squeue -u $USER | grep workflow_
```

### Direct Script Access

If you need more detailed control, you can run the scripts directly:

### Material Monitor
```bash
cd {}
python $MACE_HOME/mace/material_monitor.py --action stats
python $MACE_HOME/mace/material_monitor.py --action dashboard
```

### Queue Manager
```bash
cd {}
python $MACE_HOME/mace/queue/manager.py --status
```

### Error Recovery
```bash
cd {}
python $MACE_HOME/mace/recovery/recovery.py --action stats
python $MACE_HOME/mace/recovery/recovery.py --action recover
```

### Workflow Engine
```bash
cd {}
python $MACE_HOME/mace/workflow/engine.py --action status
```

## Notes
- The primary interface is through `mace monitor` for checking status
- All other functionality requires running scripts directly from MACE installation
- The workflow uses an isolated database in .mace_context_workflow_*/
- If MACE_HOME is not set, replace $MACE_HOME with the path to your MACE installation
""".format(workflow_dir, workflow_dir, workflow_dir, workflow_dir)
            
            readme_path = workflow_dir / "WORKFLOW_MONITORING.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
                
            ui.ok("    Created workflow monitoring guide")

        except Exception as e:
            ui.warn(f"    Warning: Could not setup monitoring scripts: {e}")
            ui.warn(f"    Error details: {type(e).__name__}: {str(e)}")
            ui.warn("    You can manually run: python setup_workflow_monitoring.py")


def main():
    """Main entry point for workflow executor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MACE Workflow Executor")
    parser.add_argument("plan_file", help="Workflow plan file to execute")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    plan_file = Path(args.plan_file)
    if not plan_file.exists():
        ui.err(f"Plan file not found: {plan_file}")
        sys.exit(1)
        
    executor = WorkflowExecutor(args.work_dir, args.db_path)
    executor.execute_workflow_plan(plan_file)


if __name__ == "__main__":
    main()