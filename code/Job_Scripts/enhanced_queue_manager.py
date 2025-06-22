#!/usr/bin/env python3
"""
Enhanced CRYSTAL Queue Manager with Material Tracking
----------------------------------------------------
Extends the existing crystal_queue_manager.py with comprehensive material tracking,
early failure detection, and automated workflow progression.

Key Features:
- Material tracking database integration
- Early job failure detection and cancellation
- Automated workflow progression (OPT -> SP -> BAND/DOSS)
- Separate calculation folders for organization
- Integration with existing SLURM scripts

Author: Based on implementation plan for material tracking system
"""

import os
import sys
import subprocess
import time
import argparse
import json
import tempfile
import shutil
import re
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

# Import our material database
from material_database import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12, find_material_by_similarity


class EnhancedCrystalQueueManager:
    """
    Enhanced queue manager with material tracking and workflow automation.
    
    Maintains compatibility with existing crystal_queue_manager.py while adding:
    - Material tracking database
    - Early failure detection
    - Automated workflow progression  
    - Organized calculation folders
    - Integration with analysis scripts
    """
    
    def __init__(self, d12_dir, max_jobs=250, reserve_slots=30, 
                 db_path="materials.db", enable_tracking=True, 
                 enable_error_recovery=True, max_recovery_attempts=3):
        self.d12_dir = Path(d12_dir).resolve()
        self.max_jobs = max_jobs
        self.reserve_slots = reserve_slots
        self.enable_tracking = enable_tracking
        self.enable_error_recovery = enable_error_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.db_path = db_path
        
        # Detect workflow context and setup script paths
        self.is_workflow_context = self._detect_workflow_context()
        self.script_paths = self._setup_script_paths()
        
        # Initialize material tracking database
        if self.enable_tracking:
            self.db = MaterialDatabase(db_path)
        else:
            self.db = None
            
        # Initialize error recovery system
        self.error_recovery_engine = None
        if self.enable_error_recovery and self.enable_tracking:
            try:
                from error_recovery import ErrorRecoveryEngine
                self.error_recovery_engine = ErrorRecoveryEngine(db_path)
                print(f"Error recovery enabled with max {self.max_recovery_attempts} attempts per job")
            except ImportError as e:
                print(f"Warning: Error recovery disabled - could not import ErrorRecoveryEngine: {e}")
                self.enable_error_recovery = False
        
        # Input settings extraction is integrated directly into database storage
            
        # Legacy job status for compatibility
        self.legacy_status_file = self.d12_dir / "crystal_job_status.json"
        self.legacy_job_status = self.load_legacy_status()
        
        # Job monitoring
        self.early_failure_checks = 5  # Number of checks before considering early failure
        self.min_job_runtime = 300  # Minimum seconds before checking for early failure
        self.max_submit_per_callback = 5  # Maximum jobs to submit per callback
        
        # Workflow settings
        self.workflow_enabled = True
        self.auto_submit_followups = True
        
    def _detect_workflow_context(self) -> bool:
        """Detect if we're running in a workflow context."""
        cwd = Path.cwd()
        
        # Check for workflow indicators
        workflow_indicators = [
            cwd / "workflow_scripts",
            cwd / "workflow_configs", 
            cwd / "workflow_outputs",
            cwd / "workflow_inputs"
        ]
        
        return any(indicator.exists() for indicator in workflow_indicators)
        
    def _setup_script_paths(self) -> dict:
        """Setup script paths based on context (workflow vs repository)."""
        script_paths = {}
        
        if self.is_workflow_context:
            # In workflow context - look for workflow-specific scripts first
            workflow_scripts_dir = Path.cwd() / "workflow_scripts"
            if workflow_scripts_dir.exists():
                script_paths.update({
                    'submitcrystal23_opt': workflow_scripts_dir / "submitcrystal23_opt_1.sh",
                    'submitcrystal23_sp': workflow_scripts_dir / "submitcrystal23_sp_2.sh", 
                    'submit_prop_band': workflow_scripts_dir / "submit_prop_band_3.sh",
                    'submit_prop_doss': workflow_scripts_dir / "submit_prop_doss_4.sh",
                    'submitcrystal23_freq': workflow_scripts_dir / "submitcrystal23_freq_5.sh"
                })
            
            # Fallback to repository scripts if workflow scripts don't exist
            repo_scripts_dir = Path(__file__).parent
            script_paths.setdefault('submitcrystal23', repo_scripts_dir / "submitcrystal23.sh")
            script_paths.setdefault('submit_prop', repo_scripts_dir / "submit_prop.sh")
        else:
            # In repository context - use original script directory
            script_dir = Path(__file__).parent
            script_paths.update({
                'submitcrystal23': script_dir / "submitcrystal23.sh",
                'submit_prop': script_dir / "submit_prop.sh"
            })
        
        return script_paths
        
    def _get_submit_script_for_calc_type(self, calc_type: str) -> Optional[str]:
        """Get the appropriate submit script for a calculation type."""
        if self.is_workflow_context:
            # In workflow context, use specific workflow scripts
            if calc_type == 'OPT':
                return str(self.script_paths.get('submitcrystal23_opt', 
                          self.script_paths.get('submitcrystal23')))
            elif calc_type == 'SP':
                return str(self.script_paths.get('submitcrystal23_sp',
                          self.script_paths.get('submitcrystal23')))
            elif calc_type == 'FREQ':
                return str(self.script_paths.get('submitcrystal23_freq',
                          self.script_paths.get('submitcrystal23')))
            elif calc_type == 'BAND':
                return str(self.script_paths.get('submit_prop_band',
                          self.script_paths.get('submit_prop')))
            elif calc_type == 'DOSS':
                return str(self.script_paths.get('submit_prop_doss',
                          self.script_paths.get('submit_prop')))
            elif calc_type in ['TRANSPORT']:
                return str(self.script_paths.get('submit_prop'))
        else:
            # In repository context, use general scripts
            if calc_type in ['OPT', 'SP', 'FREQ']:
                return str(self.script_paths.get('submitcrystal23'))
            elif calc_type in ['BAND', 'DOSS', 'TRANSPORT']:
                return str(self.script_paths.get('submit_prop'))
        
        return None
        
    def _populate_completed_jobs_from_outputs(self):
        """Populate database with completed jobs found in workflow outputs."""
        if not self.enable_tracking:
            return
            
        try:
            # Import the population script functionality
            from populate_completed_jobs import scan_for_completed_calculations, populate_database
            
            print("  Scanning for completed calculations...")
            completed_calcs = scan_for_completed_calculations(Path.cwd())
            
            if completed_calcs:
                print(f"  Found {len(completed_calcs)} completed calculations")
                added_count = populate_database(completed_calcs, self.db)
                if added_count > 0:
                    print(f"  Added {added_count} new calculations to database")
                
                # Extract properties for all completed calculations (new and existing without properties)
                print("  Checking property extraction for completed calculations...")
                self._extract_properties_for_completed_jobs(completed_calcs)
            
        except ImportError:
            print("  Warning: Could not import populate_completed_jobs module")
        except Exception as e:
            print(f"  Error populating completed jobs: {e}")
    
    def _extract_properties_for_completed_jobs(self, completed_calcs: List[Dict]):
        """Extract and store properties for completed calculations."""
        for calc_info in completed_calcs:
            try:
                # Find the calculation in the database by matching output file
                output_file = calc_info.get('output_file')
                if not output_file:
                    continue
                
                # Find database calculation by output file
                calc = self._find_calculation_by_output_file(output_file)
                if not calc:
                    print(f"  ⚠️  No database record found for {Path(output_file).name}")
                    continue
                
                calc_id = calc['calc_id']
                
                # Check if this calculation already has properties extracted
                has_properties = self._calculation_has_properties(calc_id)
                
                if not has_properties:
                    print(f"  🔍 Processing completed calculation: {calc_id}")
                    
                    # Extract and store properties
                    self.extract_and_store_properties(calc)
                    
                    # Update material information 
                    self.update_material_information(calc)
                else:
                    print(f"  ✅ Skipping {calc_id} - properties already extracted")
                
            except Exception as e:
                material_id = calc_info.get('material_id', 'unknown')
                calc_type = calc_info.get('calc_type', 'unknown')
                print(f"  ❌ Error processing {material_id}_{calc_type}: {e}")
    
    def _find_calculation_by_output_file(self, output_file: str) -> Optional[Dict]:
        """Find a calculation in the database by its output file path."""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM calculations WHERE output_file = ?",
                    (output_file,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except:
            return None
    
    def _calculation_has_properties(self, calc_id: str) -> bool:
        """Check if a calculation already has properties extracted."""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM properties WHERE calc_id = ?",
                    (calc_id,)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except:
            return False
            
    def _trigger_workflow_progression(self):
        """Trigger workflow progression using the workflow engine."""
        if not self.enable_tracking:
            return
            
        try:
            print("  Triggering workflow progression...")
            
            # Import and use WorkflowEngine for proper workflow handling
            from workflow_engine import WorkflowEngine
            
            # Initialize workflow engine with same database
            workflow_engine = WorkflowEngine(self.db_path, str(self.d12_dir))
            
            # Process completed calculations and generate next steps
            new_calc_ids = workflow_engine.process_completed_calculations()
            
            if new_calc_ids > 0:
                print(f"  Workflow engine initiated {new_calc_ids} new workflow steps")
                print("  Automatic progression to next calculation type initiated")
            else:
                print("  No new workflow steps needed at this time")
                
        except ImportError as e:
            print(f"  Could not import workflow_engine: {e}")
            print("  Falling back to basic queue processing")
            self.process_new_d12_files()
        except Exception as e:
            print(f"  Error in workflow progression: {e}")
            print("  Check workflow engine and database integrity")
        
    def load_legacy_status(self):
        """Load legacy job status for backward compatibility."""
        default_status = {"submitted": {}, "pending": [], "completed": []}
        
        status_paths = [
            self.legacy_status_file,
            Path.home() / self.legacy_status_file.name,
            Path(tempfile.gettempdir()) / self.legacy_status_file.name
        ]
        
        for status_path in status_paths:
            if status_path.exists():
                try:
                    with open(status_path, 'r') as f:
                        data = json.load(f)
                        print(f"Loaded legacy status from {status_path}")
                        return data
                except Exception as e:
                    print(f"Error reading legacy status file {status_path}: {e}")
                    
        return default_status
        
    def save_legacy_status(self):
        """Save legacy status for backward compatibility."""
        try:
            with open(self.legacy_status_file, 'w') as f:
                json.dump(self.legacy_job_status, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save legacy status: {e}")
            
    def create_calculation_folder(self, material_id: str, calc_type: str) -> Path:
        """
        Create organized folder structure for calculations.
        
        Structure: base_dir/calc_type/material_id/
        Each material gets its own directory within the calculation type.
        """
        calc_type_dir = self.d12_dir / calc_type.lower()
        material_dir = calc_type_dir / material_id
        material_dir.mkdir(parents=True, exist_ok=True)
        return material_dir
        
    def extract_material_info_from_d12(self, d12_file: Path) -> Tuple[str, str, Dict]:
        """Extract material information from .d12 file."""
        material_id = create_material_id_from_file(d12_file)
        formula = extract_formula_from_d12(d12_file)
        
        # Extract additional info from d12 file
        metadata = {
            'original_file': str(d12_file),
            'file_size': d12_file.stat().st_size,
            'created_time': datetime.fromtimestamp(d12_file.stat().st_ctime).isoformat()
        }
        
        # Try to determine calculation type from filename or content
        calc_type = self.determine_calc_type_from_file(d12_file)
        metadata['detected_calc_type'] = calc_type
        
        return material_id, formula, metadata
        
    def determine_calc_type_from_file(self, d12_file: Path) -> str:
        """Determine calculation type from filename or file content."""
        filename = d12_file.name.lower()
        
        # Check filename for type indicators
        if '_opt' in filename or 'optim' in filename:
            return 'OPT'
        elif '_sp' in filename or 'single' in filename:
            return 'SP'
        elif '_band' in filename or 'band' in filename:
            return 'BAND'
        elif '_dos' in filename or 'doss' in filename:
            return 'DOSS'
        elif '_freq' in filename or 'frequency' in filename:
            return 'FREQ'
            
        # Check file content for OPTGEOM keyword
        try:
            with open(d12_file, 'r') as f:
                content = f.read().upper()
                if 'OPTGEOM' in content:
                    return 'OPT'
                elif 'FREQCALC' in content:
                    return 'FREQ'
                else:
                    return 'SP'  # Default assumption
        except:
            return 'SP'  # Default fallback
            
    def submit_calculation(self, d12_file: Path, calc_type: str = None, 
                          material_id: str = None, prerequisite_calc_id: str = None) -> Optional[str]:
        """
        Submit a calculation with material tracking.
        
        Args:
            d12_file: Path to .d12 input file
            calc_type: Type of calculation (OPT, SP, BAND, DOSS)
            material_id: Material ID (generated if None)
            prerequisite_calc_id: Calculation this depends on
            
        Returns:
            calc_id if successful, None if failed
        """
        # Extract material information
        if material_id is None:
            material_id, formula, metadata = self.extract_material_info_from_d12(d12_file)
            print(f"    Enhanced QM: extracted material_id='{material_id}' from {d12_file.name}")
        else:
            formula = extract_formula_from_d12(d12_file)
            metadata = {}
            print(f"    Enhanced QM: using provided material_id='{material_id}' for {d12_file.name}")
            
        if calc_type is None:
            calc_type = self.determine_calc_type_from_file(d12_file)
            
        # Create material record if it doesn't exist
        if self.enable_tracking:
            existing_material = self.db.get_material(material_id)
            if not existing_material:
                self.db.create_material(
                    material_id=material_id,
                    formula=formula,
                    source_type='d12',
                    source_file=str(d12_file),
                    metadata=metadata
                )
                
        # Create calculation folder and copy input file
        calc_dir = self.create_calculation_folder(material_id, calc_type)
        input_filename = f"{material_id}_{calc_type.lower()}.d12"
        calc_input_file = calc_dir / input_filename
        
        # Copy input file to calculation directory
        shutil.copy2(d12_file, calc_input_file)
        
        # Create calculation record
        calc_id = None
        if self.enable_tracking:
            print(f"    Enhanced QM: creating calculation record for {material_id} {calc_type}")
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=calc_type,
                input_file=str(calc_input_file),
                work_dir=str(calc_dir),
                prerequisite_calc_id=prerequisite_calc_id
            )
            print(f"    Enhanced QM: created calc_id='{calc_id}'")
            
        # Submit to SLURM
        slurm_job_id = self.submit_to_slurm(calc_input_file, calc_dir, calc_type)
        
        if slurm_job_id:
            # Update tracking database
            if self.enable_tracking and calc_id:
                self.db.update_calculation_status(
                    calc_id, 
                    'submitted', 
                    slurm_job_id=slurm_job_id
                )
                
            # Update legacy tracking
            self.legacy_job_status["submitted"][slurm_job_id] = {
                "file": str(calc_input_file),
                "calc_id": calc_id,
                "material_id": material_id,
                "calc_type": calc_type,
                "submitted_time": datetime.now().isoformat()
            }
            self.save_legacy_status()
            
            print(f"Submitted {calc_type} calculation for {material_id}: Job {slurm_job_id}")
            return calc_id
        else:
            print(f"Failed to submit calculation for {material_id}")
            return None
            
    def submit_to_slurm(self, input_file: Path, work_dir: Path, calc_type: str) -> Optional[str]:
        """
        Submit job to SLURM using appropriate submission script.
        
        Args:
            input_file: Path to .d12 input file
            work_dir: Working directory for calculation
            calc_type: Type of calculation (determines which script to use)
            
        Returns:
            SLURM job ID if successful, None if failed
        """
        # Determine which submission script to use based on context
        submit_script = self._get_submit_script_for_calc_type(calc_type)
        if not submit_script:
            print(f"Unknown calculation type: {calc_type}")
            return None
            
        if not Path(submit_script).exists():
            print(f"Submit script not found: {submit_script}")
            return None
            
        # Change to working directory
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Check if this is a script generator (template) or actual SLURM script
            script_path = Path(submit_script)
            job_name = input_file.stem  # Remove .d12 extension
            
            # Check if the script contains script generation logic
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            if 'echo \'#!/bin/bash --login\' >' in script_content or 'echo "#SBATCH' in script_content:
                # This is a script generator template - run locally to generate actual script
                print(f"  Running script generator: {script_path.name}")
                cmd = ['bash', str(script_path), job_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output (the template runs sbatch at the end)
                    output = result.stdout.strip()
                    job_id_match = re.search(r'Submitted batch job (\d+)', output)
                    if job_id_match:
                        return job_id_match.group(1)
                    else:
                        print(f"Could not extract job ID from template output: {output}")
                        # Maybe the template just generated the script but didn't submit it
                        # Look for generated script and submit it manually
                        generated_script = work_dir / f"{job_name}.sh"
                        if generated_script.exists():
                            print(f"  Found generated script: {generated_script}")
                            cmd = ['sbatch', str(generated_script)]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
                                if job_id_match:
                                    return job_id_match.group(1)
                        return None
                else:
                    print(f"Error running script generator: {result.stderr}")
                    return None
                    
            else:
                # This is a regular SLURM script - submit directly
                print(f"  Submitting SLURM script: {script_path.name}")
                cmd = [str(script_path), job_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
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
            
    def check_queue_status(self):
        """Check SLURM queue and update calculation statuses."""
        # Get current queue status
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'unknown'), '-o', '%i,%T,%S'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                print(f"Error checking queue: {result.stderr}")
                return
                
            # Parse squeue output
            queue_jobs = {}
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        job_id, state = parts[0], parts[1]
                        queue_jobs[job_id] = state
                        
        except Exception as e:
            print(f"Error checking queue status: {e}")
            return
            
        # Update calculation statuses
        if self.enable_tracking:
            running_calcs = self.db.get_calculations_by_status('submitted') + \
                           self.db.get_calculations_by_status('running')
                           
            for calc in running_calcs:
                slurm_job_id = calc['slurm_job_id']
                if not slurm_job_id:
                    continue
                    
                if slurm_job_id in queue_jobs:
                    slurm_state = queue_jobs[slurm_job_id]
                    
                    # Map SLURM state to our status
                    if slurm_state in ['PENDING', 'CONFIGURING']:
                        status = 'submitted'
                    elif slurm_state in ['RUNNING']:
                        status = 'running'
                    elif slurm_state in ['COMPLETED']:
                        status = 'completed'
                        self.handle_completed_calculation(calc['calc_id'])
                    elif slurm_state in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
                        status = 'failed'
                        self.handle_failed_calculation(calc['calc_id'], slurm_state)
                    else:
                        continue  # Unknown state, don't update
                        
                    # Update database
                    if calc['status'] != status:
                        self.db.update_calculation_status(
                            calc['calc_id'], status, slurm_state=slurm_state
                        )
                        
                else:
                    # Job not in queue - check if it completed or failed
                    self.check_completed_or_failed_job(calc)
                    
    def check_early_job_failure(self):
        """Check for jobs that are failing early and cancel them if needed."""
        if not self.enable_tracking:
            return
            
        # Get jobs that have been running for a while
        cutoff_time = datetime.now() - timedelta(seconds=self.min_job_runtime)
        
        running_calcs = self.db.get_calculations_by_status('running')
        
        for calc in running_calcs:
            if not calc['started_at']:
                continue
                
            started_time = datetime.fromisoformat(calc['started_at'])
            if started_time > cutoff_time:
                continue  # Too recent to check
                
            # Check if output file shows signs of early failure
            if self.is_job_failing_early(calc):
                print(f"Detected early failure for {calc['calc_id']}, cancelling job")
                self.cancel_job(calc['slurm_job_id'], calc['calc_id'])
                
    def is_job_failing_early(self, calc: Dict) -> bool:
        """
        Check if a job is failing early by examining output files.
        
        Args:
            calc: Calculation record dictionary
            
        Returns:
            True if job appears to be failing early
        """
        output_file = calc.get('output_file')
        if not output_file or not os.path.exists(output_file):
            # No output file yet, check for common output name
            work_dir = Path(calc['work_dir'])
            material_id = calc['material_id']
            calc_type = calc['calc_type']
            
            # Try common output file patterns
            possible_outputs = [
                work_dir / f"{material_id}_{calc_type.lower()}.out",
                work_dir / f"{Path(calc['input_file']).stem}.out"
            ]
            
            for possible_output in possible_outputs:
                if possible_output.exists():
                    output_file = str(possible_output)
                    break
            else:
                return False  # No output file found
                
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Check for early failure indicators
            early_failure_patterns = [
                "CRYSTAL STOPS",  # Fatal CRYSTAL error
                "FORTRAN STOP",   # Fortran runtime error
                "segmentation fault",  # Segfault
                "killed by signal",    # Process killed
                "out of memory",       # Memory error
                "disk full",           # Disk space error
                "SLURMSTEPD: error",   # SLURM error
                "DUE TO TIME LIMIT",   # Time limit exceeded early
            ]
            
            content_upper = content.upper()
            for pattern in early_failure_patterns:
                if pattern.upper() in content_upper:
                    return True
                    
            # Check if file is too small for runtime (might indicate immediate crash)
            if len(content) < 1000 and calc.get('calc_type') == 'OPT':
                # OPT jobs should produce more output
                return True
                
        except Exception as e:
            print(f"Error checking output file {output_file}: {e}")
            
        return False
        
    def cancel_job(self, slurm_job_id: str, calc_id: str):
        """Cancel a SLURM job and update tracking."""
        try:
            result = subprocess.run(['scancel', slurm_job_id], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Cancelled job {slurm_job_id}")
                if self.enable_tracking:
                    self.db.update_calculation_status(
                        calc_id, 'cancelled', 
                        error_type='early_failure',
                        error_message='Job cancelled due to early failure detection'
                    )
            else:
                print(f"Error cancelling job {slurm_job_id}: {result.stderr}")
                
        except Exception as e:
            print(f"Error cancelling job {slurm_job_id}: {e}")
            
    def handle_completed_calculation(self, calc_id: str):
        """Handle a completed calculation - extract properties and plan next steps."""
        if not self.enable_tracking:
            return
            
        calc = self.db.get_calculation_by_slurm_id(calc_id) or \
               next((c for c in self.db.get_calculations_by_status('completed') 
                    if c['calc_id'] == calc_id), None)
               
        if not calc:
            return
            
        print(f"Handling completed calculation: {calc_id}")
        
        # Extract and store input settings directly in database
        self.extract_and_store_input_settings(calc)
        
        # Update file records
        self.update_file_records(calc)
        
        # Extract and store properties from completed calculation
        self.extract_and_store_properties(calc)
        
        # Update material information with formula and space group
        self.update_material_information(calc)
            
        # Plan next calculation in workflow
        if self.workflow_enabled and self.auto_submit_followups:
            self.plan_next_calculation(calc['material_id'], calc['calc_id'])
            
    def handle_failed_calculation(self, calc_id: str, slurm_state: str):
        """Handle a failed calculation - analyze error and attempt recovery."""
        if not self.enable_tracking:
            return
            
        calc = self.db.get_calculation_by_slurm_id(calc_id) or \
               next((c for c in self.db.get_calculations_by_status('failed') 
                    if c['calc_id'] == calc_id), None)
               
        if not calc:
            return
            
        print(f"Handling failed calculation: {calc_id} (SLURM state: {slurm_state})")
        
        # Analyze error type from output file
        error_type, error_message = self.analyze_calculation_error(calc)
        
        # Update database with error information
        self.db.update_calculation_status(
            calc_id, 'failed',
            error_type=error_type,
            error_message=error_message
        )
        
        # Attempt automatic error recovery
        if self.enable_error_recovery and self.error_recovery_engine:
            recovery_success = self.attempt_error_recovery(calc, error_type, error_message)
            if recovery_success:
                print(f"✅ Error recovery successful for {calc_id} - job resubmitted")
            else:
                print(f"❌ Error recovery failed or not applicable for {calc_id}")
        else:
            print(f"Error analysis: {error_type} - {error_message}")
        
    def analyze_calculation_error(self, calc: Dict) -> Tuple[str, str]:
        """
        Analyze the error in a failed calculation.
        
        Returns:
            Tuple of (error_type, error_message)
        """
        # Try to find and read output file
        output_file = calc.get('output_file')
        if not output_file or not os.path.exists(output_file):
            return "no_output", "No output file found"
            
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Common CRYSTAL error patterns (from updatelists2.py logic)
            error_patterns = {
                'shrink_error': [
                    "SHRINK FACTOR TOO SMALL",
                    "TOO SMALL SHRINK FACTOR"
                ],
                'memory_error': [
                    "INSUFFICIENT MEMORY",
                    "OUT OF MEMORY",
                    "MEMORY ALLOCATION",
                    "SEGMENTATION FAULT"
                ],
                'convergence_error': [
                    "SCF NOT CONVERGED",
                    "CONVERGENCE NOT ACHIEVED",
                    "TOO MANY SCF CYCLES"
                ],
                'geometry_error': [
                    "ATOMS TOO CLOSE",
                    "GEOMETRY OPTIMIZATION FAILED",
                    "SMALL DISTANCE BETWEEN ATOMS"
                ],
                'time_limit': [
                    "DUE TO TIME LIMIT",
                    "TIME LIMIT EXCEEDED"
                ],
                'io_error': [
                    "I/O ERROR",
                    "DISK FULL",
                    "PERMISSION DENIED"
                ]
            }
            
            content_upper = content.upper()
            
            for error_type, patterns in error_patterns.items():
                for pattern in patterns:
                    if pattern in content_upper:
                        return error_type, f"Detected: {pattern}"
                        
            # If no specific error found, return generic
            return "unknown_error", "Calculation failed with unknown error"
            
        except Exception as e:
            return "file_error", f"Error reading output file: {e}"
            
    def attempt_error_recovery(self, calc: Dict, error_type: str, error_message: str) -> bool:
        """
        Attempt automatic error recovery for a failed calculation.
        
        Args:
            calc: Calculation record from database
            error_type: Type of error detected
            error_message: Error message details
            
        Returns:
            bool: True if recovery was successful and job resubmitted, False otherwise
        """
        calc_id = calc['calc_id']
        
        # Check if error type is recoverable
        recoverable_errors = ['shrink_error', 'memory_error', 'convergence_error', 'timeout_error', 'scf_error']
        if error_type not in recoverable_errors:
            print(f"⚠️  Error type '{error_type}' is not recoverable for {calc_id}")
            return False
        
        # Check recovery attempt limits
        recovery_count = self.get_recovery_attempt_count(calc_id)
        if recovery_count >= self.max_recovery_attempts:
            print(f"⚠️  Max recovery attempts ({self.max_recovery_attempts}) reached for {calc_id}")
            return False
        
        print(f"🔧 Attempting error recovery for {calc_id} (attempt {recovery_count + 1}/{self.max_recovery_attempts})")
        print(f"   Error: {error_type} - {error_message}")
        
        try:
            # Use ErrorRecoveryEngine to attempt recovery
            recovered = self.error_recovery_engine.attempt_recovery(calc)
            
            if recovered:
                # Increment recovery attempt count
                self.increment_recovery_attempt_count(calc_id)
                
                # Resubmit the job
                work_dir = Path(calc['work_dir'])
                if self.resubmit_fixed_calculation(calc):
                    print(f"🚀 Successfully resubmitted recovered job for {calc_id}")
                    return True
                else:
                    print(f"❌ Failed to resubmit recovered job for {calc_id}")
                    return False
            else:
                print(f"🔧 Recovery not successful for {calc_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error during recovery attempt for {calc_id}: {e}")
            return False
    
    def get_recovery_attempt_count(self, calc_id: str) -> int:
        """Get the number of recovery attempts for a calculation."""
        if not self.db:
            return 0
        try:
            # Query database for recovery attempts
            result = self.db.execute_query(
                "SELECT recovery_attempts FROM calculations WHERE calc_id = ?", 
                (calc_id,)
            )
            return result[0]['recovery_attempts'] if result else 0
        except:
            return 0
    
    def increment_recovery_attempt_count(self, calc_id: str):
        """Increment the recovery attempt count for a calculation."""
        if not self.db:
            return
        try:
            current_count = self.get_recovery_attempt_count(calc_id)
            self.db.execute_query(
                "UPDATE calculations SET recovery_attempts = ? WHERE calc_id = ?",
                (current_count + 1, calc_id)
            )
        except Exception as e:
            print(f"Warning: Could not update recovery attempt count for {calc_id}: {e}")
    
    def resubmit_fixed_calculation(self, calc: Dict) -> bool:
        """Resubmit a calculation after error recovery."""
        try:
            work_dir = Path(calc['work_dir'])
            calc_id = calc['calc_id']
            
            # Find the D12 file (should be fixed by error recovery)
            d12_files = list(work_dir.glob("*.d12"))
            if not d12_files:
                print(f"❌ No D12 file found in {work_dir} for resubmission")
                return False
            
            d12_file = d12_files[0]
            
            # Update database status to 'resubmitted'
            self.db.update_calculation_status(calc_id, 'resubmitted', 
                                            error_type=None, error_message="Recovered and resubmitted")
            
            # Mark this as a recovery resubmission
            if hasattr(self.db, 'execute_query'):
                try:
                    self.db.execute_query(
                        "UPDATE calculations SET completion_type = 'recovery_attempt' WHERE calc_id = ?",
                        (calc_id,)
                    )
                except:
                    pass  # Column might not exist in older databases
            
            # Submit the calculation using existing submit logic
            return self.submit_single_calculation(d12_file, calc['calc_type'])
            
        except Exception as e:
            print(f"❌ Error resubmitting calculation {calc['calc_id']}: {e}")
            return False
    
    def extract_and_store_properties(self, calc: Dict):
        """Extract properties from completed calculation and store in database."""
        try:
            # Import property extractor
            from crystal_property_extractor import CrystalPropertyExtractor
            
            output_file = calc.get('output_file')
            if not output_file or not Path(output_file).exists():
                print(f"  ⚠️  No output file found for property extraction: {calc['calc_id']}")
                return
            
            print(f"  🔍 Extracting properties from {Path(output_file).name}")
            
            # Initialize property extractor with same database
            extractor = CrystalPropertyExtractor(self.db_path)
            
            # Extract properties
            properties = extractor.extract_all_properties(
                Path(output_file),
                material_id=calc['material_id'],
                calc_id=calc['calc_id']
            )
            
            if properties:
                # Save properties to database
                saved_count = extractor.save_properties_to_database(properties)
                print(f"  ✅ Extracted and saved {saved_count} properties")
            else:
                print(f"  ⚠️  No properties extracted from {Path(output_file).name}")
                
        except ImportError:
            print(f"  ⚠️  Property extractor not available - skipping property extraction")
        except Exception as e:
            print(f"  ❌ Error during property extraction for {calc['calc_id']}: {e}")
    
    def update_material_information(self, calc: Dict):
        """Update material information with formula and space group from files."""
        try:
            # Import formula extractor
            from formula_extractor import update_materials_table_info
            
            material_id = calc['material_id']
            input_file = calc.get('input_file')
            output_file = calc.get('output_file')
            
            # Find associated CIF file if available
            work_dir = Path(calc['work_dir'])
            cif_files = list(work_dir.glob("*.cif"))
            cif_file = cif_files[0] if cif_files else None
            
            # Update material information
            update_materials_table_info(
                self.db,
                material_id,
                d12_file=Path(input_file) if input_file else None,
                cif_file=cif_file,
                output_file=Path(output_file) if output_file else None
            )
            
        except ImportError:
            print(f"  ⚠️  Formula extractor not available - skipping material info update")
        except Exception as e:
            print(f"  ⚠️  Error updating material information for {calc['calc_id']}: {e}")
            
    def update_file_records(self, calc: Dict):
        """Update file records for a completed calculation."""
        if not self.enable_tracking:
            return
            
        work_dir = Path(calc['work_dir'])
        calc_id = calc['calc_id']
        
        # Common file patterns to track
        file_patterns = {
            'output': ['*.out'],
            'log': ['*.log', '*.err'],
            'property': ['*.dat', '*.csv'],
            'wavefunction': ['*.f9', 'fort.9'],
            'plot': ['*.png', '*.pdf']
        }
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                for file_path in work_dir.glob(pattern):
                    self.db.add_file_record(
                        calc_id=calc_id,
                        file_type=file_type,
                        file_name=file_path.name,
                        file_path=str(file_path)
                    )
                    
    def extract_and_store_input_settings(self, calc: Dict):
        """Extract input settings and store directly in materials database."""
        if not self.enable_tracking:
            return
            
        try:
            from input_settings_extractor import extract_and_store_input_settings
            
            calc_id = calc['calc_id']
            input_file = calc.get('input_file')
            
            if not input_file:
                print(f"  ⚠️  No input file found for settings extraction: {calc_id}")
                return
            
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"  ⚠️  Input file not found: {input_path}")
                return
            
            print(f"  ⚙️  Extracting input settings from {input_path.name}")
            
            # Extract and store settings directly in materials.db
            success = extract_and_store_input_settings(calc_id, input_path, self.db_path)
            
            if success:
                print(f"  ✅ Input settings stored in materials.db for {calc_id}")
            else:
                print(f"  ⚠️  Failed to extract input settings for {calc_id}")
                
        except ImportError:
            print(f"  ⚠️  Input settings extractor not available")
        except Exception as e:
            print(f"  ❌ Error extracting input settings for {calc_id}: {e}")
        
    def extract_properties(self, calc: Dict):
        """Extract properties from completed calculation."""
        # This will be implemented in Phase 3
        # For now, just placeholder
        print(f"TODO: Extract properties from {calc['calc_id']}")
        
    def plan_next_calculation(self, material_id: str, completed_calc_id: str):
        """Plan and submit the next calculation in the workflow using WorkflowEngine."""
        if not self.enable_tracking:
            return
            
        print(f"Triggering workflow progression for material {material_id}")
        
        try:
            # Import and use WorkflowEngine for proper workflow handling
            from workflow_engine import WorkflowEngine
            
            # Initialize workflow engine with same database
            workflow_engine = WorkflowEngine(self.db_path, str(self.base_dir))
            
            # Process completed calculations and generate next steps
            new_calc_ids = workflow_engine.execute_workflow_step(material_id, completed_calc_id)
            
            if new_calc_ids:
                print(f"Workflow engine initiated {len(new_calc_ids)} new calculations for {material_id}")
                
                # If auto-submission is enabled, submit the new calculations
                if self.auto_submit_followups:
                    for calc_id in new_calc_ids:
                        calc = next((c for c in self.db.get_all_calculations() 
                                   if c['calc_id'] == calc_id), None)
                        if calc and calc.get('input_file'):
                            print(f"Auto-submitting generated calculation: {calc_id}")
                            slurm_job_id = self.submit_to_slurm(
                                Path(calc['input_file']), 
                                Path(calc['input_file']).parent,
                                calc['calc_type']
                            )
                            if slurm_job_id:
                                self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=slurm_job_id)
                                print(f"Submitted {calc_id} as SLURM job {slurm_job_id}")
            else:
                print(f"No new workflow steps needed for {material_id}")
                
        except ImportError as e:
            print(f"Could not import workflow_engine: {e}")
            print("Falling back to basic workflow progression")
            # Fallback to basic next step determination if workflow_engine not available
            next_calc_type = self.db.get_next_calculation_in_workflow(material_id)
            if next_calc_type:
                print(f"Next step needed: {next_calc_type} (manual generation required)")
            else:
                print(f"Workflow complete for material {material_id}")
        except Exception as e:
            print(f"Error in workflow progression: {e}")
            print("Workflow progression failed - check logs for details")
            
    def generate_followup_input_file(self, completed_calc: Dict, next_calc_type: str) -> Optional[Path]:
        """
        Generate input file for follow-up calculation.
        
        This is a placeholder - full implementation will use:
        - CRYSTALOptToD12.py for OPT -> SP
        - alldos.py for SP -> DOSS  
        - create_band_d3.py for SP -> BAND
        """
        print(f"TODO: Generate {next_calc_type} input from {completed_calc['calc_id']}")
        return None
        
    def check_completed_or_failed_job(self, calc: Dict):
        """Check if a job that's not in queue has completed or failed."""
        # Check for output files to determine completion status
        work_dir = Path(calc['work_dir'])
        
        # Look for output files
        output_files = list(work_dir.glob("*.out"))
        
        if output_files:
            # Found output file, check if calculation completed successfully
            output_file = output_files[0]
            
            try:
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                # Check for successful completion indicators
                if "CRYSTAL ENDS" in content.upper() or "CALCULATION TERMINATED" in content.upper():
                    # Successful completion
                    self.db.update_calculation_status(
                        calc['calc_id'], 'completed',
                        output_file=str(output_file)
                    )
                    self.handle_completed_calculation(calc['calc_id'])
                else:
                    # Failed calculation
                    error_type, error_message = self.analyze_calculation_error(calc)
                    self.db.update_calculation_status(
                        calc['calc_id'], 'failed',
                        output_file=str(output_file),
                        error_type=error_type,
                        error_message=error_message
                    )
                    
            except Exception as e:
                print(f"Error checking output file {output_file}: {e}")
                
    def process_new_d12_files(self):
        """Process new .d12 files in the directory for submission."""
        # Find .d12 files that haven't been submitted yet
        # Search both directly in d12_dir and in workflow subdirectories
        d12_files = list(self.d12_dir.glob("*.d12"))  # Direct files
        d12_files.extend(list(self.d12_dir.glob("**/*.d12")))  # Recursive search in subdirectories
        
        # Remove duplicates (in case a file appears in both searches)
        d12_files = list(set(d12_files))
        
        submitted_count = 0
        
        for d12_file in d12_files:
            # Check if we've reached the submission limit for this callback
            if submitted_count >= self.max_submit_per_callback:
                print(f"Reached max submissions per callback ({self.max_submit_per_callback})")
                break
            # Check if this file has already been submitted
            if self.enable_tracking:
                material_id = create_material_id_from_file(d12_file)
                existing_calcs = self.db.get_calculations_by_status(
                    material_id=material_id
                )
                
                # Skip if already has calculations
                if existing_calcs:
                    continue
                    
            # Check queue capacity
            current_jobs = len(self.legacy_job_status["submitted"])
            if current_jobs >= (self.max_jobs - self.reserve_slots):
                print(f"Queue nearly full ({current_jobs}/{self.max_jobs}), skipping new submissions")
                break
                
            # Submit the calculation
            calc_id = self.submit_calculation(d12_file)
            if calc_id:
                submitted_count += 1
            
    def run_monitoring_cycle(self):
        """Run one cycle of queue monitoring and management."""
        print(f"\n=== Queue Monitoring Cycle - {datetime.now()} ===")
        
        # Check queue status and update calculations
        self.check_queue_status()
        
        # Check for early job failures
        self.check_early_job_failure()
        
        # Process new .d12 files for submission
        self.process_new_d12_files()
        
        # Print status summary
        if self.enable_tracking:
            stats = self.db.get_database_stats()
            print(f"Database Stats: {stats['total_materials']} materials, "
                  f"{sum(stats.get('calculations_by_status', {}).values())} calculations")
                  
        print("=== End Monitoring Cycle ===\n")
        
    def run_callback_check(self, mode='completion'):
        """Run a single callback check cycle based on trigger mode."""
        print(f"\n=== Queue Manager Callback ({mode}) - {datetime.now()} ===")
        
        if mode == 'completion':
            # Job completion callback - check status and trigger workflow progression
            
            # First, populate database with any completed jobs not yet tracked
            if self.is_workflow_context:
                self._populate_completed_jobs_from_outputs()
            
            self.check_queue_status()
            
            # In workflow context, use workflow engine for progression instead of basic D12 processing
            if self.is_workflow_context and self.workflow_enabled:
                self._trigger_workflow_progression()
            else:
                # Fallback to basic D12 file processing
                self.process_new_d12_files()
            
        elif mode == 'early_failure':
            # Early failure detection
            self.check_early_job_failure()
            
        elif mode == 'status_check':
            # General status check
            self.check_queue_status()
            
        elif mode == 'submit_new':
            # Submit new jobs if capacity available
            self.process_new_d12_files()
            
        elif mode == 'full_check':
            # Full monitoring cycle
            self.run_monitoring_cycle()
            
        # Print status summary
        if self.enable_tracking:
            stats = self.db.get_database_stats()
            print(f"Database Stats: {stats['total_materials']} materials, "
                  f"{sum(stats.get('calculations_by_status', {}).values())} calculations")
                  
        print("=== Callback Complete ===\n")
            
    def get_status_report(self) -> Dict:
        """Generate a comprehensive status report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'queue_info': {},
            'database_stats': {}
        }
        
        # Legacy queue info
        report['queue_info'] = {
            'submitted_jobs': len(self.legacy_job_status.get("submitted", {})),
            'max_jobs': self.max_jobs,
            'reserve_slots': self.reserve_slots
        }
        
        # Database statistics
        if self.enable_tracking:
            report['database_stats'] = self.db.get_database_stats()
            
        return report


    def store_workflow_configuration_as_template(self, workflow_config_file: Path = None):
        """Store current workflow configuration as a template in the database."""
        if not self.enable_tracking:
            return
            
        try:
            # Find workflow configuration file if not provided
            if not workflow_config_file:
                config_dir = Path.cwd() / "workflow_configs"
                config_files = list(config_dir.glob("workflow_plan_*.json"))
                if not config_files:
                    print("  ⚠️  No workflow configuration files found")
                    return
                workflow_config_file = sorted(config_files)[-1]  # Use most recent
            
            if not workflow_config_file.exists():
                print(f"  ⚠️  Workflow config file not found: {workflow_config_file}")
                return
                
            print(f"  📋 Storing workflow configuration as template: {workflow_config_file.name}")
            
            # Load workflow configuration
            with open(workflow_config_file, 'r') as f:
                config = json.load(f)
            
            # Extract template information
            template_id = f"template_{config['created'].replace(':', '').replace('-', '').replace('.', '_')}"
            template_name = f"{config['input_type'].upper()} → {' → '.join(config['workflow_sequence'])}"
            description = f"Auto-generated from {workflow_config_file.name}"
            
            # Convert workflow steps to template format
            workflow_steps = []
            for step_num, calc_type in enumerate(config['workflow_sequence'], 1):
                step_key = f"{calc_type}_{step_num}"
                step_config = config['step_configurations'].get(step_key, {})
                
                workflow_steps.append({
                    'step_number': step_num,
                    'calc_type': calc_type,
                    'source': step_config.get('source', 'unknown'),
                    'slurm_config': step_config.get('slurm_config', {}),
                    'dependencies': [step_num - 1] if step_num > 1 else []
                })
            
            # Store template in database
            self.db.create_workflow_template(
                template_id=template_id,
                template_name=template_name,
                workflow_steps=workflow_steps,
                description=description
            )
            
            print(f"  ✅ Stored workflow template: {template_id}")
            return template_id
            
        except Exception as e:
            print(f"  ❌ Error storing workflow template: {e}")
            return None
    
    def create_workflow_instance_for_material(self, material_id: str, template_id: str = None):
        """Create a workflow instance for a material."""
        if not self.enable_tracking:
            return None
            
        try:
            # Use most recent template if not specified
            if not template_id:
                templates = self.db.get_all_workflow_templates()
                if not templates:
                    print(f"  ⚠️  No workflow templates found")
                    return None
                template_id = templates[0]['template_id']
            
            # Create workflow instance
            instance_id = self.db.create_workflow_instance(material_id, template_id)
            print(f"  📋 Created workflow instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            print(f"  ❌ Error creating workflow instance for {material_id}: {e}")
            return None


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced CRYSTAL Queue Manager with Material Tracking"
    )
    parser.add_argument(
        "--d12-dir", 
        default=".", 
        help="Directory containing .d12 files (default: current directory)"
    )
    parser.add_argument(
        "--max-jobs", 
        type=int, 
        default=250, 
        help="Maximum number of jobs to maintain (default: 250)"
    )
    parser.add_argument(
        "--reserve", 
        type=int, 
        default=30, 
        help="Number of job slots to reserve (default: 30)"
    )
    parser.add_argument(
        "--db-path", 
        default="materials.db", 
        help="Path to materials database (default: materials.db)"
    )
    parser.add_argument(
        "--callback-mode", 
        choices=['completion', 'early_failure', 'status_check', 'submit_new', 'full_check'],
        default='completion',
        help="Callback mode (default: completion)"
    )
    parser.add_argument(
        "--disable-tracking", 
        action="store_true", 
        help="Disable material tracking (legacy mode)"
    )
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Show status report and exit"
    )
    parser.add_argument(
        "--submit-file", 
        help="Submit a specific .d12 file and exit"
    )
    parser.add_argument(
        "--max-submit", 
        type=int, 
        default=5, 
        help="Maximum number of new jobs to submit in one callback (default: 5)"
    )
    parser.add_argument(
        "--disable-error-recovery", 
        action="store_true", 
        help="Disable automatic error recovery"
    )
    parser.add_argument(
        "--max-recovery-attempts", 
        type=int, 
        default=3, 
        help="Maximum recovery attempts per job (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Create queue manager
    manager = EnhancedCrystalQueueManager(
        d12_dir=args.d12_dir,
        max_jobs=args.max_jobs,
        reserve_slots=args.reserve,
        db_path=args.db_path,
        enable_tracking=not args.disable_tracking,
        enable_error_recovery=not args.disable_error_recovery,
        max_recovery_attempts=args.max_recovery_attempts
    )
    
    manager.max_submit_per_callback = args.max_submit
    
    if args.status:
        # Print status report and exit
        report = manager.get_status_report()
        print(json.dumps(report, indent=2))
        
    elif args.submit_file:
        # Submit specific file and exit
        d12_file = Path(args.submit_file)
        if not d12_file.exists():
            print(f"Error: File {d12_file} not found")
            sys.exit(1)
            
        calc_id = manager.submit_calculation(d12_file)
        if calc_id:
            print(f"Successfully submitted calculation: {calc_id}")
        else:
            print("Failed to submit calculation")
            sys.exit(1)
            
    else:
        # Run callback check
        manager.run_callback_check(args.callback_mode)


if __name__ == "__main__":
    main()