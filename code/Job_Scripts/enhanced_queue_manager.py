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
                 db_path="materials.db", enable_tracking=True):
        self.d12_dir = Path(d12_dir).resolve()
        self.max_jobs = max_jobs
        self.reserve_slots = reserve_slots
        self.enable_tracking = enable_tracking
        
        # Initialize material tracking database
        if self.enable_tracking:
            self.db = MaterialDatabase(db_path)
        else:
            self.db = None
            
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
        
        Structure: base_dir/calc_type/
        All materials of the same calculation type stored together.
        This is more efficient for file quotas and workflow analysis.
        """
        calc_dir = self.d12_dir / calc_type.lower()
        calc_dir.mkdir(parents=True, exist_ok=True)
        return calc_dir
        
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
        else:
            formula = extract_formula_from_d12(d12_file)
            metadata = {}
            
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
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=calc_type,
                input_file=str(calc_input_file),
                work_dir=str(calc_dir),
                prerequisite_calc_id=prerequisite_calc_id
            )
            
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
        # Determine which submission script to use
        if calc_type in ['OPT', 'SP', 'FREQ']:
            submit_script = "submitcrystal23.sh"
        elif calc_type in ['BAND', 'DOSS', 'TRANSPORT']:
            submit_script = "submit_prop.sh"
        else:
            print(f"Unknown calculation type: {calc_type}")
            return None
            
        # Change to working directory
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Submit job
            cmd = [submit_script, input_file.name]
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
        
        # Update file records
        self.update_file_records(calc)
        
        # Extract properties if this is a property-generating calculation
        if calc['calc_type'] in ['OPT', 'SP', 'BAND', 'DOSS']:
            self.extract_properties(calc)
            
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
        
        # TODO: Implement error recovery in Phase 2
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
                    
    def extract_properties(self, calc: Dict):
        """Extract properties from completed calculation."""
        # This will be implemented in Phase 3
        # For now, just placeholder
        print(f"TODO: Extract properties from {calc['calc_id']}")
        
    def plan_next_calculation(self, material_id: str, completed_calc_id: str):
        """Plan and submit the next calculation in the workflow."""
        if not self.enable_tracking:
            return
            
        # Get next calculation type needed
        next_calc_type = self.db.get_next_calculation_in_workflow(material_id)
        
        if not next_calc_type:
            print(f"Workflow complete for material {material_id}")
            return
            
        print(f"Planning next calculation for {material_id}: {next_calc_type}")
        
        # Get the completed calculation details
        completed_calc = next(
            (c for c in self.db.get_calculations_by_status('completed') 
             if c['calc_id'] == completed_calc_id), None
        )
        
        if not completed_calc:
            print(f"Could not find completed calculation {completed_calc_id}")
            return
            
        # Generate input file for next calculation
        next_input_file = self.generate_followup_input_file(
            completed_calc, next_calc_type
        )
        
        if next_input_file:
            # Submit the next calculation
            self.submit_calculation(
                d12_file=next_input_file,
                calc_type=next_calc_type,
                material_id=material_id,
                prerequisite_calc_id=completed_calc_id
            )
        else:
            print(f"Failed to generate input file for {next_calc_type}")
            
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
        d12_files = list(self.d12_dir.glob("*.d12"))
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
            # Job completion callback - check status and submit new jobs
            self.check_queue_status()
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
    
    args = parser.parse_args()
    
    # Create queue manager
    manager = EnhancedCrystalQueueManager(
        d12_dir=args.d12_dir,
        max_jobs=args.max_jobs,
        reserve_slots=args.reserve,
        db_path=args.db_path,
        enable_tracking=not args.disable_tracking
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