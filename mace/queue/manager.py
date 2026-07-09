#!/usr/bin/env python3
"""
Enhanced CRYSTAL Queue Manager with Material Tracking
----------------------------------------------------
Extends the existing crystal_queue_manager.py with comprehensive material tracking,
early failure detection, and automated workflow progression.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

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

# Import MACE components
from mace.database.materials import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12, find_material_by_similarity
from mace.database.materials_contextual import ContextualMaterialDatabase
from mace.workflow.context import get_current_context

# Import lock manager for race condition prevention
try:
    from mace.queue.queue_lock_manager import QueueLockManager, CallbackThrottler
    LOCKING_AVAILABLE = True
except ImportError:
    try:
        # Fallback to importing from current directory
        from .queue_lock_manager import QueueLockManager, CallbackThrottler
        LOCKING_AVAILABLE = True
    except ImportError:
        LOCKING_AVAILABLE = False
        print("Warning: Queue locking not available - race conditions possible with simultaneous callbacks")


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
                 enable_error_recovery=True, max_recovery_attempts=3,
                 organize_outputs=True):
        self.d12_dir = Path(d12_dir).resolve()
        self.max_jobs = max_jobs
        self.reserve_slots = reserve_slots
        self.enable_tracking = enable_tracking
        # When False (manual `mace manager` default), inputs are submitted in place
        # instead of being copied into a <calc_type>/<material_id>/ tree. Workflow runs
        # keep organize_outputs=True (and use their own workflow dirs anyway).
        self.organize_outputs = organize_outputs
        self.enable_error_recovery = enable_error_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.db_path = db_path
        
        # Detect workflow context and setup script paths
        self.is_workflow_context = self._detect_workflow_context()
        self.script_paths = self._setup_script_paths()
        
        # Initialize material tracking database
        if self.enable_tracking:
            # Check for active workflow context
            ctx = get_current_context()
            if ctx:
                # Use contextual database with explicit db_path if provided
                # This ensures we use the context's database path
                self.db = ContextualMaterialDatabase(db_path=db_path if db_path != "materials.db" else None)
                print(f"Using workflow context database: {self.db.get_context_info()['db_path']}")
            else:
                # Traditional database
                self.db = MaterialDatabase(db_path)
        else:
            self.db = None
            
        # Initialize error recovery system
        self.error_recovery_engine = None
        if self.enable_error_recovery and self.enable_tracking:
            try:
                # Lazy import to avoid pandas/pyarrow conflicts
                from mace.recovery.recovery import ErrorRecoveryEngine
                self.error_recovery_engine = ErrorRecoveryEngine(db_path)
                print(f"Error recovery enabled with max {self.max_recovery_attempts} attempts per job")
            except ImportError as e:
                print(f"Warning: Error recovery disabled - could not import ErrorRecoveryEngine: {e}")
                self.enable_error_recovery = False
            except Exception as e:
                print(f"Warning: Error recovery disabled due to error: {e}")
                self.enable_error_recovery = False
                self.error_recovery_engine = None
        
        # Input settings extraction is integrated directly into database storage
            
        # Legacy job status for compatibility
        # Use context-specific status file if available
        ctx = get_current_context()
        if ctx:
            self.legacy_status_file = ctx.get_storage_path() / "crystal_job_status.json"
        else:
            self.legacy_status_file = self.d12_dir / "crystal_job_status.json"
        self.legacy_job_status = self.load_legacy_status()
        
        # Job monitoring
        self.early_failure_checks = 5  # Number of checks before considering early failure
        self.min_job_runtime = 300  # Minimum seconds before checking for early failure
        self.max_submit_per_callback = 5  # Maximum jobs to submit per callback
        
        # Workflow settings
        self.workflow_enabled = True
        self.auto_submit_followups = True
        
        # Initialize lock manager for race condition prevention
        self.lock_manager = None
        self.throttler = None
        if LOCKING_AVAILABLE:
            try:
                # Use context-specific lock directory if available
                ctx = get_current_context()
                if ctx:
                    lock_dir = ctx.get_lock_dir()
                else:
                    # Anchor the lock to the DATABASE being protected, not the
                    # callback's cwd: follow-up jobs run in their own dirs, so
                    # cwd-anchored locks gave every concurrent completion
                    # callback its own lock dir — no mutual exclusion at all,
                    # and near-simultaneous completions duplicated the same
                    # follow-up calculations (sp2, 4x band/doss).
                    lock_dir = Path(self.db_path).resolve().parent / ".queue_locks"
                    
                self.lock_manager = QueueLockManager(lock_dir=lock_dir, lock_timeout=300)
                self.throttler = CallbackThrottler(min_delay=0.5, max_delay=2.0)
                print(f"Queue locking enabled - lock directory: {lock_dir}")
            except Exception as e:
                print(f"Warning: Could not initialize lock manager: {e}")
                self.lock_manager = None
                self.throttler = None
        
    def _detect_workflow_context(self) -> bool:
        """Detect if we're running in a workflow context."""
        cwd = Path.cwd()
        
        # First, check if we have MACE_WORKFLOW_ID environment variable set
        # This is a more reliable indicator when running from SLURM scripts
        if os.environ.get('MACE_WORKFLOW_ID'):
            # Try to find the workflow root based on the workflow ID
            workflow_id = os.environ.get('MACE_WORKFLOW_ID')
            
            # Check current directory and up to 10 parent directories for the workflow root
            current = cwd
            for _ in range(10):
                # Look for workflow indicators at this level
                if (current / "workflow_outputs" / workflow_id).exists():
                    self.workflow_root = current
                    return True
                if current.parent == current:
                    break
                current = current.parent
        
        # Fallback to directory-based detection
        # Check current directory and up to 7 parent directories
        check_dirs = [cwd]
        current = cwd
        for _ in range(7):
            if current.parent != current:
                current = current.parent
                check_dirs.append(current)
            else:
                break
        
        # Check for workflow indicators in any of these directories
        workflow_root_candidate = None
        for check_dir in check_dirs:
            workflow_indicators = [
                check_dir / "workflow_scripts",
                check_dir / "workflow_configs", 
                check_dir / "workflow_outputs",
                check_dir / "workflow_inputs"
            ]
            
            if any(indicator.exists() for indicator in workflow_indicators):
                # Store the first (deepest) candidate
                if workflow_root_candidate is None:
                    workflow_root_candidate = check_dir
                
                # Check if this is the true workflow root by verifying it has multiple indicators
                indicators_found = sum(1 for ind in workflow_indicators if ind.exists())
                if indicators_found >= 3:  # Strong indication this is the root
                    self.workflow_root = check_dir
                    return True
        
        # If we found a candidate but not a strong match, use the highest level candidate
        if workflow_root_candidate:
            # Find the highest level directory with workflow indicators
            for check_dir in reversed(check_dirs):
                workflow_indicators = [
                    check_dir / "workflow_scripts",
                    check_dir / "workflow_configs", 
                    check_dir / "workflow_outputs",
                    check_dir / "workflow_inputs"
                ]
                if any(indicator.exists() for indicator in workflow_indicators):
                    self.workflow_root = check_dir
                    return True
        
        self.workflow_root = None
        return False
        
    def _setup_script_paths(self) -> dict:
        """Setup script paths based on context (workflow vs repository)."""
        script_paths = {}
        
        if self.is_workflow_context and hasattr(self, 'workflow_root'):
            # In workflow context - look for workflow-specific scripts first
            workflow_scripts_dir = self.workflow_root / "workflow_scripts"
            if workflow_scripts_dir.exists():
                script_paths.update({
                    'submitcrystal23_opt': workflow_scripts_dir / "submitcrystal23_opt_1.sh",
                    'submitcrystal23_sp': workflow_scripts_dir / "submitcrystal23_sp_2.sh", 
                    'submit_prop_band': workflow_scripts_dir / "submit_prop_band_3.sh",
                    'submit_prop_doss': workflow_scripts_dir / "submit_prop_doss_4.sh",
                    'submitcrystal23_freq': workflow_scripts_dir / "submitcrystal23_freq_5.sh"
                })
            
            # Fallback to repository scripts if workflow scripts don't exist
            # (templates live in mace/submission, not mace/queue)
            repo_scripts_dir = Path(__file__).parent.parent / "submission"
            script_paths.setdefault('submitcrystal23', repo_scripts_dir / "submitcrystal23.sh")
            script_paths.setdefault('submit_prop', repo_scripts_dir / "submit_prop.sh")
        else:
            # In repository context - use the submission script directory
            script_dir = Path(__file__).parent.parent / "submission"
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
            elif calc_type in ['TRANSPORT', 'CHARGE+POTENTIAL']:
                return str(self.script_paths.get('submit_prop'))
        else:
            # In repository context, use general scripts
            if calc_type in ['OPT', 'SP', 'FREQ']:
                return str(self.script_paths.get('submitcrystal23'))
            elif calc_type in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']:
                return str(self.script_paths.get('submit_prop'))
        
        return None
        
    def _populate_completed_jobs_from_outputs(self):
        """Populate database with completed jobs found in workflow outputs."""
        if not self.enable_tracking:
            return
            
        try:
            # Import the population script functionality
            from mace.database.populate_completed_jobs import scan_for_completed_calculations, populate_database
            
            print("  Scanning for completed calculations...")
            
            # If in workflow context, scan the entire workflow directory
            scan_dir = Path.cwd()
            if self.is_workflow_context:
                # Try to find the workflow root directory
                workflow_id = os.environ.get('MACE_WORKFLOW_ID')
                if workflow_id:
                    # Look for the workflow directory in parent paths
                    current = Path.cwd()
                    for _ in range(10):  # Check up to 10 levels up
                        # Check if current directory name matches workflow_id
                        if current.name == workflow_id and current.parent.name == "workflow_outputs":
                            scan_dir = current
                            print(f"  Scanning entire workflow directory: {scan_dir}")
                            break
                        # Also check for workflow_outputs/workflow_id pattern
                        potential_workflow = current / "workflow_outputs" / workflow_id
                        if potential_workflow.exists():
                            scan_dir = potential_workflow
                            print(f"  Scanning entire workflow directory: {scan_dir}")
                            break
                        current = current.parent
                        if current == current.parent:  # Reached root
                            break
            
            completed_calcs = scan_for_completed_calculations(scan_dir)
            
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
            from mace.workflow.engine import WorkflowEngine
            
            # Determine the correct base directory for workflow engine
            if self.is_workflow_context and hasattr(self, 'workflow_root'):
                # Use the workflow root directory when in workflow context
                base_dir = str(self.workflow_root)
            else:
                # Use the d12_dir for non-workflow contexts
                base_dir = str(self.d12_dir)
            
            # Initialize workflow engine with same database and correct base directory
            workflow_engine = WorkflowEngine(self.db_path, base_dir)
            
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
            
    def create_calculation_folder(self, material_id: str, calc_type: str,
                                  source_file=None) -> Path:
        """
        Create organized folder structure for calculations.

        Structure:
        - Workflow: Uses existing workflow structure (no new folders created)
        - Standard (organize_outputs=True): base_dir/calc_type/material_id/
        - In-place (organize_outputs=False): the source file's own directory, with no
          new folders created (manual `mace manager` default). Requires source_file.
        """
        if self.is_workflow_context and hasattr(self, 'workflow_root'):
            # In workflow context, don't create new folders
            # The workflow executor has already created the proper structure
            # Just return the current directory if we're already in the right place
            cwd = Path.cwd()
            
            # Check if we're in a material-specific directory already
            if cwd.name == material_id or cwd.parent.name.startswith('step_'):
                return cwd
            
            # Otherwise, try to find the material directory in workflow outputs
            workflow_outputs = self.workflow_root / "workflow_outputs"
            if workflow_outputs.exists():
                # Look for the material directory in any workflow/step
                for workflow_dir in workflow_outputs.iterdir():
                    if workflow_dir.is_dir() and workflow_dir.name.startswith('workflow_'):
                        for step_dir in workflow_dir.iterdir():
                            if step_dir.is_dir() and step_dir.name.startswith('step_'):
                                material_dir = step_dir / material_id
                                if material_dir.exists():
                                    return material_dir
            
            # If not found, fall back to creating in current directory
            # This shouldn't happen in normal workflow operation
            print(f"Warning: Could not find workflow directory for {material_id}, using current directory")
            return Path.cwd()
        else:
            if not self.organize_outputs and source_file is not None:
                # In-place: submit the file where it already lives; no copying, no
                # <calc_type>/<material_id>/ tree. The DB still records this as work_dir.
                return Path(source_file).resolve().parent
            # Standard behavior - create folder structure
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
                          material_id: str = None, prerequisite_calc_id: str = None,
                          job_script_override: Path = None) -> Optional[str]:
        """
        Submit a calculation with material tracking.

        Args:
            d12_file: Path to .d12 input file
            calc_type: Type of calculation (OPT, SP, BAND, DOSS)
            material_id: Material ID (generated if None)
            prerequisite_calc_id: Calculation this depends on
            job_script_override: A ready-made SLURM script to submit directly
                instead of regenerating one from the template. Used by error
                recovery to honor a bumped --mem/--time script.

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
                
        # Create calculation folder (organized) or resolve in-place directory.
        calc_dir = self.create_calculation_folder(material_id, calc_type, source_file=d12_file)

        # Determine file extension based on calculation type
        is_d3_calc = calc_type.rstrip('0123456789') in ['BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']
        file_extension = '.d3' if is_d3_calc else '.d12'

        in_place = (not self.organize_outputs) and (not self.is_workflow_context)
        if in_place:
            # Submit the original file as-is: no copy, no rename, no extra subfolder.
            calc_input_file = Path(d12_file).resolve()
        else:
            input_filename = f"{material_id}_{calc_type.lower()}{file_extension}"
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
        slurm_job_id = self.submit_to_slurm(calc_input_file, calc_dir, calc_type,
                                            submit_script_override=job_script_override)
        
        if slurm_job_id:
            # Update tracking database
            if self.enable_tracking and calc_id:
                self.db.update_calculation_status(
                    calc_id,
                    'submitted',
                    slurm_job_id=slurm_job_id
                )
                # Record the SLURM script that was ACTUALLY submitted so the
                # memory/timeout recovery handlers (which edit resource
                # directives) can find it. On a recovered resubmission that is
                # the bumped *_recovery_N.sh override — recording the original
                # generated script instead made every subsequent bump restart
                # from the original resources (escalation plateaued at one
                # bump instead of compounding).
                try:
                    if job_script_override:
                        submitted_script = Path(job_script_override)
                    else:
                        submitted_script = calc_dir / f"{calc_input_file.stem}.sh"
                    if submitted_script.exists():
                        with self.db._get_connection() as conn:
                            conn.execute(
                                "UPDATE calculations SET job_script = ? WHERE calc_id = ?",
                                (str(submitted_script), calc_id)
                            )
                except Exception:
                    pass
                
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
            
    def submit_to_slurm(self, input_file: Path, work_dir: Path, calc_type: str,
                        submit_script_override: Path = None) -> Optional[str]:
        """
        Submit job to SLURM using appropriate submission script.

        Args:
            input_file: Path to .d12 input file
            work_dir: Working directory for calculation
            calc_type: Type of calculation (determines which script to use)
            submit_script_override: A ready-made SLURM script to submit directly,
                bypassing template selection. Used by error recovery so a bumped
                --mem/--time script is honored instead of being regenerated away.

        Returns:
            SLURM job ID if successful, None if failed
        """
        # An error-recovery override (a concrete, already-bumped SLURM script)
        # takes precedence over the per-calc-type template.
        if submit_script_override and Path(submit_script_override).exists():
            submit_script = str(Path(submit_script_override).resolve())
        else:
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
                    
            elif re.search(r'(?m)^\s*#SBATCH\b', script_content):
                # A ready-made SLURM batch file (e.g. an error-recovery bumped
                # --mem/--time script returned via submit_script_override). It must
                # be handed to sbatch, NOT executed directly: executing the batch
                # body would run the payload on the login node and never print a
                # 'Submitted batch job N' line, and such files are typically not +x
                # (raising PermissionError that gets swallowed upstream, so recovery
                # silently never resubmits). Submit via sbatch and parse the job id.
                print(f"  Submitting SLURM batch file via sbatch: {script_path.name}")
                cmd = ['sbatch', str(script_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
                    if job_id_match:
                        return job_id_match.group(1)
                    else:
                        print(f"Could not extract job ID from: {result.stdout.strip()}")
                        return None
                else:
                    print(f"Error submitting batch file: {result.stderr}")
                    return None

            else:
                # A self-submitting executable script (legacy path) - run it directly.
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
            
    def check_queue(self) -> Tuple[int, int]:
        """Check SLURM queue and return (running, pending) job counts.
        
        Returns:
            Tuple of (running_jobs, pending_jobs)
        """
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'unknown'), '-o', '%i,%T,%S'],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode != 0:
                raise Exception(f"squeue error: {result.stderr}")
                
            # Count jobs by state
            running = 0
            pending = 0
            
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        state = parts[1]
                        if state == 'RUNNING':
                            running += 1
                        elif state in ['PENDING', 'CONFIGURING']:
                            pending += 1
                            
            return running, pending
            
        except FileNotFoundError:
            raise Exception("SLURM (squeue) not found - this command requires SLURM to be available")
        except Exception as e:
            raise Exception(f"Error checking queue: {e}")
    
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
                        
        except FileNotFoundError:
            # SLURM not available
            if self.enable_tracking:
                print("  SLURM not available - skipping queue status check")
            return
        except Exception as e:
            print(f"Error checking queue status: {e}")
            return
            
        # Update calculation statuses
        if self.enable_tracking:
            # A completion callback runs as the final line of the job script
            # itself, so squeue still lists the invoking job as RUNNING even
            # though its CRYSTAL run has finished. Classify that calc from its
            # output file instead of queue state — a lone manual job has no
            # later callback to sweep it up and would stay 'running' forever.
            own_job_id = os.environ.get('SLURM_JOB_ID')

            running_calcs = self.db.get_calculations_by_status('submitted') + \
                           self.db.get_calculations_by_status('running')

            for calc in running_calcs:
                slurm_job_id = calc['slurm_job_id']
                if not slurm_job_id:
                    continue

                if own_job_id and str(slurm_job_id) == own_job_id:
                    self.check_completed_or_failed_job(calc)
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
                    elif slurm_state in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
                        status = 'failed'
                    else:
                        continue  # Unknown state, don't update

                    # Update database BEFORE invoking handlers so they see
                    # the final status (handlers previously ran first and
                    # their status-based lookups found nothing)
                    if calc['status'] != status:
                        self.db.update_calculation_status(
                            calc['calc_id'], status, slurm_state=slurm_state
                        )

                    if status == 'completed':
                        self.handle_completed_calculation(calc['calc_id'])
                    elif status == 'failed':
                        self.handle_failed_calculation(calc['calc_id'], slurm_state)
                        
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
            
        # calc_id is a calculation ID; look it up directly (slurm-id lookup
        # kept as fallback for legacy callers that pass a SLURM job id)
        calc = self.db.get_calculation(calc_id) or \
               self.db.get_calculation_by_slurm_id(calc_id)

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
            
        # calc_id is a calculation ID; look it up directly (slurm-id lookup
        # kept as fallback for legacy callers that pass a SLURM job id)
        calc = self.db.get_calculation(calc_id) or \
               self.db.get_calculation_by_slurm_id(calc_id)

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
                    # CRYSTAL prints "TOO MANY CYCLES" (matches detector.py);
                    # the old "TOO MANY SCF CYCLES" never matched real output.
                    "TOO MANY CYCLES"
                ],
                'geometry_error': [
                    "ATOMS TOO CLOSE",
                    "GEOMETRY OPTIMIZATION FAILED",
                    "SMALL DISTANCE BETWEEN ATOMS"
                ],
                'timeout_error': [
                    "DUE TO TIME LIMIT",
                    "TIME LIMIT EXCEEDED"
                ],
                'disk_space_error': [
                    "DISK FULL",
                    "NO SPACE LEFT",
                    "DISK QUOTA EXCEEDED"
                ],
                'io_error': [
                    "I/O ERROR",
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
        
        # Check if error type is recoverable. disk_space_error routes to the
        # engine's cleanup_handler (frees scratch >100MB, protecting wavefunction
        # files) -- previously 'DISK FULL' was lumped into io_error and excluded
        # here, so the advertised disk-space recovery was unreachable.
        recoverable_errors = ['shrink_error', 'memory_error', 'convergence_error',
                              'timeout_error', 'scf_error', 'disk_space_error']
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
            # Use ErrorRecoveryEngine to attempt recovery. The calc dict was
            # fetched before the error info was written, so inject the
            # analyzed error type/message the engine keys its handlers on.
            calc = dict(calc)
            calc['error_type'] = error_type
            calc['error_message'] = error_message
            # create_record=False: we create the single submission row below via
            # resubmit, so the engine should not also create an (unsubmitted,
            # orphaned) recovery row.
            recovery = self.error_recovery_engine.attempt_recovery(calc, create_record=False)

            if recovery:
                # Increment recovery attempt count
                self.increment_recovery_attempt_count(calc_id)

                # Resubmit the FIXED input the handler produced — NOT the
                # original failing input — honoring any bumped job script.
                if self.resubmit_fixed_calculation(
                    calc,
                    fixed_input=recovery.get('fixed_input_file'),
                    fixed_job_script=recovery.get('fixed_job_script'),
                ):
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
            # MaterialDatabase has no execute_query method — the old call
            # raised AttributeError, the bare except returned 0, and
            # max_recovery_attempts was never enforced
            with self.db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT recovery_attempts FROM calculations WHERE calc_id = ?",
                    (calc_id,)
                )
                row = cursor.fetchone()
                return (row[0] or 0) if row else 0
        except Exception:
            return 0

    def increment_recovery_attempt_count(self, calc_id: str):
        """Increment the recovery attempt count for a calculation."""
        if not self.db:
            return
        try:
            with self.db._get_connection() as conn:
                conn.execute(
                    "UPDATE calculations SET recovery_attempts = COALESCE(recovery_attempts, 0) + 1 "
                    "WHERE calc_id = ?",
                    (calc_id,)
                )
        except Exception as e:
            print(f"Warning: Could not update recovery attempt count for {calc_id}: {e}")
    
    def resubmit_fixed_calculation(self, calc: Dict, fixed_input=None, fixed_job_script=None) -> bool:
        """Resubmit a calculation after error recovery.

        Args:
            calc: the original failed calculation record.
            fixed_input: the recovery-generated input (e.g. a ``*_recovery_*.d12``
                with a bumped MAXCYCLE). When given we submit THIS — submitting
                the original failing input instead was the core recovery bug.
            fixed_job_script: a recovery-generated SLURM script carrying bumped
                ``--mem``/``--time`` (resource handlers). When given it is
                submitted directly rather than regenerating a default script
                from the template, which would discard the resource bump.
        """
        try:
            work_dir = Path(calc['work_dir'])
            calc_id = calc['calc_id']

            d12_file = None
            # Prefer the recovery-generated fix.
            if fixed_input:
                fixed_input = Path(fixed_input)
                if fixed_input.exists():
                    d12_file = fixed_input
                else:
                    print(f"⚠️  Recovery input {fixed_input} missing; falling back to work_dir input")

            if d12_file is None:
                # Fallback: locate an input in the work_dir. Prefer the one whose
                # stem matches this calc's recorded input_file: in in-place mode
                # several calculations can share one work_dir, so a bare glob()[0]
                # could resubmit the wrong input. Then recorded path, then first match.
                d12_files = list(work_dir.glob("*.d12"))
                if not d12_files:
                    print(f"❌ No D12 file found in {work_dir} for resubmission")
                    return False
                recorded = calc.get('input_file')
                if recorded:
                    recorded_path = Path(recorded)
                    if recorded_path.exists():
                        d12_file = recorded_path
                    else:
                        for f in d12_files:
                            if f.stem == recorded_path.stem:
                                d12_file = f
                                break
                if d12_file is None:
                    d12_file = d12_files[0]

            # Update database status to 'resubmitted'
            self.db.update_calculation_status(calc_id, 'resubmitted', 
                                            error_type=None, error_message="Recovered and resubmitted")
            
            # Mark this as a recovery resubmission
            try:
                with self.db._get_connection() as conn:
                    conn.execute(
                        "UPDATE calculations SET completion_type = 'recovery_attempt' WHERE calc_id = ?",
                        (calc_id,)
                    )
            except Exception:
                pass  # Column might not exist in older databases

            # Submit the calculation using existing submit logic
            # (submit_single_calculation never existed on this class —
            # recovered jobs were silently never resubmitted)
            new_calc_id = self.submit_calculation(
                d12_file, calc_type=calc.get('calc_type'),
                material_id=calc.get('material_id'),
                job_script_override=fixed_job_script,
            )
            return new_calc_id is not None
            
        except Exception as e:
            print(f"❌ Error resubmitting calculation {calc['calc_id']}: {e}")
            return False
    
    def extract_and_store_properties(self, calc: Dict):
        """Extract properties from completed calculation and store in database."""
        try:
            # Import property extractor
            from mace.utils.property_extractor import CrystalPropertyExtractor
            
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
            from mace.utils.formula_extractor import update_materials_table_info
            
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
            from mace.utils.settings_extractor import extract_and_store_input_settings

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
            from mace.workflow.engine import WorkflowEngine
            
            # Determine the correct base directory for workflow engine
            if self.is_workflow_context and hasattr(self, 'workflow_root'):
                # Use the workflow root directory when in workflow context
                base_dir = str(self.workflow_root)
            else:
                # Use the d12_dir for non-workflow contexts
                base_dir = str(self.d12_dir)
            
            # Initialize workflow engine with same database and correct base directory
            workflow_engine = WorkflowEngine(self.db_path, base_dir)
            
            # Process completed calculations and generate next steps
            new_calc_ids = workflow_engine.execute_workflow_step(material_id, completed_calc_id)
            
            if new_calc_ids:
                print(f"Workflow engine initiated {len(new_calc_ids)} new calculations for {material_id}")
                
                # If auto-submission is enabled, submit the new calculations
                if self.auto_submit_followups:
                    for calc_id in new_calc_ids:
                        calc = next((c for c in self.db.get_all_calculations()
                                   if c['calc_id'] == calc_id), None)
                        if calc and calc.get('slurm_job_id'):
                            # execute_workflow_step already submitted it (and
                            # its script carries the workflow context exports).
                            # Re-submitting here launched a duplicate job AND
                            # re-ran the raw script generator in place, wiping
                            # those exports — so the duplicate's callbacks
                            # opened a fresh cwd-local DB and fanned out again.
                            continue
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
        - CRYSTALOptToD3.py for SP -> DOSS  
        - CRYSTALOptToD3.py for SP -> BAND
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
            # Prefer the output matching this calculation's input file name
            input_stem = Path(calc['input_file']).stem if calc.get('input_file') else None
            output_file = next(
                (f for f in output_files if f.stem == input_stem), output_files[0]
            )

            try:
                # Reuse the validated completion detection from completion_checker:
                # real CRYSTAL outputs signal success via "OPT END" / "TOTAL CPU
                # TIME =" — the old "CRYSTAL ENDS"/"CALCULATION TERMINATED"
                # markers match zero real outputs, so every finished job that had
                # left the queue was marked failed.
                from mace.completion_checker import categorize_output_file
                category, _ = categorize_output_file(output_file)

                if category.startswith('complete'):
                    # Successful completion
                    self.db.update_calculation_status(
                        calc['calc_id'], 'completed',
                        output_file=str(output_file)
                    )
                    self.handle_completed_calculation(calc['calc_id'])
                else:
                    # Job left the queue without a completion signal -> failed.
                    # Record the output file BEFORE invoking the handler so
                    # error analysis reads the real output (the stale in-memory
                    # record had output_file=None -> "no_output"), and route
                    # through handle_failed_calculation so error recovery runs
                    # for this path too.
                    self.db.update_calculation_status(
                        calc['calc_id'], 'failed',
                        output_file=str(output_file)
                    )
                    self.handle_failed_calculation(calc['calc_id'], 'NOT_IN_QUEUE')

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
        # Apply throttling to reduce simultaneous callbacks
        if self.throttler:
            self.throttler.throttle(f"callback_{mode}")
            
        # Acquire distributed lock
        if self.lock_manager:
            lock_name = f"queue_manager_{mode}"
            try:
                # Execute callback with lock
                self.lock_manager.with_lock(
                    lock_name,
                    self._run_callback_check_locked,
                    mode,
                    timeout=60
                )
            except TimeoutError:
                print(f"⚠️  Could not acquire lock for {mode} callback - another instance may be running")
                return
            except Exception as e:
                print(f"❌ Error in callback with locking: {e}")
                # Fall back to running without lock
                self._run_callback_check_locked(mode)
        else:
            # No lock manager available, run directly
            self._run_callback_check_locked(mode)
            
    def _run_callback_check_locked(self, mode='completion'):
        """Internal callback implementation with lock protection."""
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
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Copy each input into a <calc_type>/<material_id>/ folder (organized). "
             "Default is in-place: submit files where they are, no copies."
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
        max_recovery_attempts=args.max_recovery_attempts,
        organize_outputs=args.organize
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