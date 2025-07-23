# MACE Workflow Module - Detailed Refactoring Plan
# Section 4: Monolithic Script Decomposition

## 4. Monolithic Script Decomposition

### 4.1 Overview of Monolithic Scripts

#### 4.1.1 Size and Complexity Analysis

```
Script          Lines    Methods    Classes    Max Method Size    Avg Complexity    Dependencies
-----------  --------  ---------  ---------  -----------------  ----------------  --------------
engine.py       3,291         53          1                225              28.4              12
planner.py      4,619         87          1                285              24.7              15
executor.py     1,902         42          1                165              18.3               8

Total:          9,812        182          3                285              23.8              35

Issues:
- 73% of codebase is in 3 files
- Average method size: 54 lines (target: <20)
- 15 methods exceed 100 lines
- High coupling between components
- Difficult to test individual functions
- Long import chains
```

### 4.2 engine.py Deep Decomposition

#### 4.2.1 Current Structure Analysis

```python
# engine.py current structure (3,291 lines)

# Lines 1-150: Imports and constants
import os
import sys
import json
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
# ... 140 more lines of imports and constants

# Lines 151-300: Class initialization
class WorkflowEngine:
    """Manages workflow execution and progression."""
    
    def __init__(self, base_dir=".", db_path="materials.db"):
        # 150 lines of initialization code
        self.base_dir = Path(base_dir)
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        self.queue_manager = QueueManager()
        # ... extensive setup
    
    # Lines 301-800: Core workflow processing
    def process_all_materials(self):
        """Main entry point - 185 lines!"""
        # Complex logic with nested loops and conditions
    
    def process_material(self, material_id):
        """Process single material - 165 lines"""
        # Multiple responsibilities mixed together
    
    # Lines 801-1500: File management
    def _prepare_calculation_directory(self, calc_config):
        """125 lines of directory setup"""
    
    def _copy_required_files(self, source, dest, calc_type):
        """145 lines of file copying logic"""
    
    # Lines 1501-2200: Job submission
    def _submit_calculation(self, calc_config):
        """225 lines - largest method!"""
        # SLURM script generation, submission, tracking
    
    # Lines 2201-2800: Error recovery
    def _handle_calculation_error(self, error, calc):
        """185 lines of error handling"""
    
    # Lines 2801-3291: Optional calculations
    def process_optional_calculations(self, material_id):
        """165 lines for optional calc handling"""

# Key problems:
# 1. process_all_materials() does too much
# 2. File operations mixed with business logic
# 3. SLURM details embedded in workflow logic
# 4. Error handling scattered throughout
# 5. No clear separation of concerns
```

#### 4.2.2 Proposed Module Structure

```
engine/
├── __init__.py              # Package initialization
├── core.py                  # Core WorkflowEngine class (400 lines)
├── orchestrator.py          # Workflow orchestration logic (500 lines)
├── material_processor.py    # Material processing logic (400 lines)
├── file_manager.py          # File operations (600 lines)
├── job_submitter.py         # SLURM job submission (500 lines)
├── error_recovery.py        # Error handling and recovery (400 lines)
├── optional_calcs.py        # Optional calculation handling (300 lines)
├── validators.py            # Input/output validation (200 lines)
├── constants.py             # Constants and configuration (100 lines)
├── utils.py                # Utility functions (200 lines)
└── types.py                # Type definitions (100 lines)
```

#### 4.2.3 Detailed Module Breakdown

##### core.py - Core WorkflowEngine Class

```python
# engine/core.py
"""
Core workflow engine that coordinates all components.
~400 lines
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .orchestrator import WorkflowOrchestrator
from .material_processor import MaterialProcessor
from .file_manager import WorkflowFileManager
from .job_submitter import JobSubmitter
from .error_recovery import ErrorRecoveryHandler
from .optional_calcs import OptionalCalculationHandler
from .validators import WorkflowValidator
from ..database.materials import MaterialDatabase

class WorkflowEngine:
    """
    Main workflow engine - coordinates all workflow operations.
    
    This class serves as the facade for the workflow subsystem,
    delegating specific responsibilities to specialized components.
    """
    
    def __init__(self, base_dir: Path = Path("."), db_path: str = "materials.db"):
        """Initialize workflow engine with all components."""
        self.base_dir = Path(base_dir).resolve()
        self.db_path = db_path
        self.logger = self._setup_logging()
        
        # Initialize database connection
        self.db = MaterialDatabase(db_path)
        
        # Initialize components with dependency injection
        self.orchestrator = WorkflowOrchestrator(self.db, self.logger)
        self.material_processor = MaterialProcessor(self.db, self.logger)
        self.file_manager = WorkflowFileManager(self.base_dir, self.logger)
        self.job_submitter = JobSubmitter(self.db, self.logger)
        self.error_handler = ErrorRecoveryHandler(self.db, self.logger)
        self.optional_handler = OptionalCalculationHandler(self.db, self.logger)
        self.validator = WorkflowValidator(self.logger)
        
        # Register component callbacks
        self._register_callbacks()
        
        self.logger.info(f"WorkflowEngine initialized at {self.base_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for workflow engine."""
        logger = logging.getLogger('WorkflowEngine')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_callbacks(self):
        """Register callbacks between components."""
        # When orchestrator needs to submit a job
        self.orchestrator.on_submit_job = self.job_submitter.submit
        
        # When job completes
        self.job_submitter.on_job_complete = self.orchestrator.handle_job_completion
        
        # When error occurs
        self.orchestrator.on_error = self.error_handler.handle_error
        
        # When files need to be prepared
        self.material_processor.on_prepare_files = self.file_manager.prepare_calculation
    
    # Main public interface
    def process_all_materials(self, 
                            workflow_id: Optional[str] = None,
                            material_filter: Optional[List[str]] = None,
                            dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all materials in workflow(s).
        
        Args:
            workflow_id: Specific workflow to process
            material_filter: List of material IDs to process
            dry_run: If True, only simulate processing
            
        Returns:
            Processing results summary
        """
        self.logger.info(f"Starting material processing (dry_run={dry_run})")
        
        # Delegate to orchestrator
        return self.orchestrator.process_workflows(
            workflow_id=workflow_id,
            material_filter=material_filter,
            dry_run=dry_run
        )
    
    def process_material(self, 
                        material_id: str,
                        force: bool = False) -> Dict[str, Any]:
        """
        Process a single material.
        
        Args:
            material_id: Material to process
            force: Force processing even if not ready
            
        Returns:
            Processing results
        """
        return self.material_processor.process(material_id, force=force)
    
    def process_optional_calculations(self,
                                    material_id: str,
                                    calc_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process optional calculations for a material.
        
        Args:
            material_id: Material ID
            calc_types: Specific calculation types to process
            
        Returns:
            Processing results
        """
        return self.optional_handler.process(material_id, calc_types)
    
    def check_material_status(self, material_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a material."""
        return self.material_processor.get_status(material_id)
    
    def recover_failed_calculations(self,
                                  workflow_id: Optional[str] = None,
                                  auto_fix: bool = True) -> Dict[str, Any]:
        """
        Attempt to recover failed calculations.
        
        Args:
            workflow_id: Limit to specific workflow
            auto_fix: Automatically apply fixes
            
        Returns:
            Recovery results
        """
        return self.error_handler.recover_failures(workflow_id, auto_fix)
    
    # Utility methods
    def validate_workflow(self, workflow_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate workflow configuration."""
        return self.validator.validate_workflow(workflow_config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            'total_materials': self.db.count_materials(),
            'active_workflows': len(self.orchestrator.get_active_workflows()),
            'running_jobs': len(self.job_submitter.get_running_jobs()),
            'failed_calculations': len(self.error_handler.get_failed_calculations()),
            'file_operations': self.file_manager.get_statistics(),
            'job_statistics': self.job_submitter.get_statistics()
        }
        return stats
```

##### orchestrator.py - Workflow Orchestration

```python
# engine/orchestrator.py
"""
Workflow orchestration logic.
~500 lines
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from ..database.materials import MaterialDatabase
from .types import WorkflowState, CalculationDependency

class WorkflowOrchestrator:
    """
    Orchestrates workflow execution and progression.
    
    Responsible for:
    - Determining calculation order
    - Managing dependencies
    - Coordinating job submission
    - Tracking workflow state
    """
    
    def __init__(self, db: MaterialDatabase, logger: logging.Logger):
        self.db = db
        self.logger = logger
        
        # Callbacks for external operations
        self.on_submit_job: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Workflow state tracking
        self.active_workflows: Dict[str, WorkflowState] = {}
        
        # Dependency graph
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build calculation dependency graph."""
        return {
            'OPT': [],  # No dependencies
            'SP': ['OPT'],  # Depends on OPT
            'FREQ': ['OPT'],  # Depends on OPT
            'BAND': ['SP'],  # Depends on SP
            'DOSS': ['SP'],  # Depends on SP
            'TRANSPORT': ['SP'],  # Depends on SP
            'CHARGE+POTENTIAL': ['SP'],  # Depends on SP
            'ECHG': ['SP'],  # Depends on SP
            'POTM': ['SP'],  # Depends on SP
        }
    
    def process_workflows(self,
                         workflow_id: Optional[str] = None,
                         material_filter: Optional[List[str]] = None,
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Main workflow processing logic.
        
        This method:
        1. Gets workflows to process
        2. For each workflow, gets materials
        3. For each material, determines next calculations
        4. Submits ready calculations
        5. Tracks progress
        """
        results = {
            'workflows_processed': 0,
            'materials_processed': 0,
            'calculations_submitted': 0,
            'errors': [],
            'details': {}
        }
        
        # Get workflows
        workflows = self._get_workflows_to_process(workflow_id)
        
        for workflow in workflows:
            wf_id = workflow['workflow_id']
            self.logger.info(f"Processing workflow {wf_id}")
            
            # Initialize workflow state
            if wf_id not in self.active_workflows:
                self.active_workflows[wf_id] = WorkflowState(
                    workflow_id=wf_id,
                    status='active',
                    current_step=workflow.get('current_step', 1)
                )
            
            # Get materials
            materials = self.db.get_materials_for_workflow(wf_id)
            
            # Apply filter if provided
            if material_filter:
                materials = [m for m in materials if m['material_id'] in material_filter]
            
            # Process each material
            workflow_results = []
            for material in materials:
                mat_result = self._process_material_in_workflow(
                    material,
                    workflow,
                    dry_run
                )
                workflow_results.append(mat_result)
                
                if mat_result['submitted'] > 0:
                    results['calculations_submitted'] += mat_result['submitted']
                if mat_result['errors']:
                    results['errors'].extend(mat_result['errors'])
            
            results['workflows_processed'] += 1
            results['materials_processed'] += len(materials)
            results['details'][wf_id] = workflow_results
        
        return results
    
    def _process_material_in_workflow(self,
                                    material: Dict[str, Any],
                                    workflow: Dict[str, Any],
                                    dry_run: bool) -> Dict[str, Any]:
        """Process single material within workflow context."""
        mat_id = material['material_id']
        result = {
            'material_id': mat_id,
            'submitted': 0,
            'skipped': 0,
            'errors': []
        }
        
        # Get workflow sequence
        sequence = self._get_workflow_sequence(workflow)
        
        # Determine next calculations
        next_calcs = self._determine_next_calculations(
            material,
            sequence,
            workflow['workflow_id']
        )
        
        # Submit ready calculations
        for calc_type in next_calcs:
            if dry_run:
                self.logger.info(f"[DRY RUN] Would submit {calc_type} for {mat_id}")
                result['submitted'] += 1
            else:
                try:
                    job_id = self._submit_calculation(
                        material,
                        calc_type,
                        workflow['workflow_id']
                    )
                    if job_id:
                        result['submitted'] += 1
                        self.logger.info(f"Submitted {calc_type} for {mat_id}: job {job_id}")
                    else:
                        result['skipped'] += 1
                except Exception as e:
                    error_msg = f"Failed to submit {calc_type} for {mat_id}: {e}"
                    result['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
                    if self.on_error:
                        self.on_error(e, material, calc_type)
        
        return result
    
    def _determine_next_calculations(self,
                                   material: Dict[str, Any],
                                   sequence: List[str],
                                   workflow_id: str) -> List[str]:
        """
        Determine which calculations are ready to run.
        
        Algorithm:
        1. Get completed calculations for material
        2. For each calc type in sequence:
           - Check if already completed
           - Check if dependencies are met
           - Check if already running
        3. Return list of ready calculations
        """
        mat_id = material['material_id']
        
        # Get calculation history
        calculations = self.db.get_calculations_for_material(mat_id)
        
        # Build status map
        calc_status = {}
        for calc in calculations:
            calc_type = calc['calc_type']
            status = calc['status']
            
            # Handle multiple instances (OPT2, OPT3, etc.)
            base_type = calc_type.rstrip('0123456789')
            
            if base_type not in calc_status:
                calc_status[base_type] = []
            calc_status[base_type].append(status)
        
        # Determine ready calculations
        ready = []
        
        for calc_type in sequence:
            # Check if already completed
            if calc_type in calc_status:
                statuses = calc_status[calc_type]
                if 'completed' in statuses:
                    continue  # Already done
                if 'running' in statuses or 'submitted' in statuses:
                    continue  # Already in progress
            
            # Check dependencies
            dependencies = self.dependency_graph.get(calc_type, [])
            deps_met = True
            
            for dep in dependencies:
                if dep not in calc_status:
                    deps_met = False
                    break
                    
                dep_statuses = calc_status[dep]
                if 'completed' not in dep_statuses:
                    deps_met = False
                    break
            
            if deps_met:
                ready.append(calc_type)
                
                # Only submit one calculation at a time per material
                # to avoid resource conflicts
                break
        
        return ready
    
    def _submit_calculation(self,
                          material: Dict[str, Any],
                          calc_type: str,
                          workflow_id: str) -> Optional[str]:
        """Submit calculation via job submitter."""
        if not self.on_submit_job:
            raise RuntimeError("No job submitter registered")
        
        # Prepare calculation configuration
        calc_config = {
            'material_id': material['material_id'],
            'calc_type': calc_type,
            'workflow_id': workflow_id,
            'formula': material.get('formula', 'unknown'),
            'space_group': material.get('space_group', 'unknown')
        }
        
        # Submit via callback
        return self.on_submit_job(calc_config)
    
    def handle_job_completion(self, job_id: str, status: str, output_path: Path):
        """Handle job completion callback."""
        self.logger.info(f"Job {job_id} completed with status {status}")
        
        # Update calculation status
        calc = self.db.get_calculation_by_job_id(job_id)
        if calc:
            self.db.update_calculation_status(
                calc['calc_id'],
                status,
                output_path=str(output_path)
            )
            
            # Check if this triggers new calculations
            material = self.db.get_material(calc['material_id'])
            workflow = self.db.get_workflow(calc['workflow_id'])
            
            if material and workflow:
                self._process_material_in_workflow(
                    material,
                    workflow,
                    dry_run=False
                )
```

##### file_manager.py - File Operations Management

```python
# engine/file_manager.py
"""
File operations and management.
~600 lines
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil
import os
import hashlib
import json
import logging
from datetime import datetime

class WorkflowFileManager:
    """
    Manages all file operations for workflows.
    
    Responsibilities:
    - Preparing calculation directories
    - Copying required files
    - Organizing output files
    - Validating file integrity
    - Managing temporary files
    - Archiving completed calculations
    """
    
    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = Path(base_dir).resolve()
        self.logger = logger
        
        # Directory structure
        self.work_dir = self.base_dir / "workflow_work"
        self.archive_dir = self.base_dir / "workflow_archive"
        self.temp_dir = self.base_dir / "workflow_temp"
        
        # Create directories
        self._create_directory_structure()
        
        # File operation statistics
        self.stats = {
            'files_copied': 0,
            'files_moved': 0,
            'directories_created': 0,
            'total_size_processed': 0,
            'errors': 0
        }
    
    def _create_directory_structure(self):
        """Create required directory structure."""
        for directory in [self.work_dir, self.archive_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories for organization
        for calc_type in ['OPT', 'SP', 'FREQ', 'BAND', 'DOSS', 'TRANSPORT', 'CHARGE+POTENTIAL']:
            (self.work_dir / calc_type).mkdir(exist_ok=True)
            (self.archive_dir / calc_type).mkdir(exist_ok=True)
    
    def prepare_calculation(self,
                          calc_config: Dict[str, Any],
                          source_files: Dict[str, Path]) -> Path:
        """
        Prepare calculation directory with all required files.
        
        Args:
            calc_config: Calculation configuration
            source_files: Dictionary of required files
            
        Returns:
            Path to prepared calculation directory
        """
        calc_type = calc_config['calc_type']
        material_id = calc_config['material_id']
        workflow_id = calc_config['workflow_id']
        
        # Create unique calculation directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        calc_dir = self.work_dir / calc_type / f"{material_id}_{calc_type}_{timestamp}"
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats['directories_created'] += 1
        self.logger.info(f"Created calculation directory: {calc_dir}")
        
        # Copy required files
        for file_type, source_path in source_files.items():
            if not source_path.exists():
                raise FileNotFoundError(f"Required file not found: {source_path}")
            
            # Determine destination name
            if file_type == 'input':
                dest_name = f"{material_id}.d12" if calc_type != 'FREQ' else f"{material_id}.d12"
            elif file_type == 'wavefunction':
                dest_name = f"{material_id}.f9"
            elif file_type == 'properties':
                dest_name = f"{material_id}.d3"
            else:
                dest_name = source_path.name
            
            dest_path = calc_dir / dest_name
            
            # Copy file
            self._copy_file_with_verification(source_path, dest_path)
        
        # Create metadata file
        metadata = {
            'calc_type': calc_type,
            'material_id': material_id,
            'workflow_id': workflow_id,
            'created_at': datetime.now().isoformat(),
            'source_files': {k: str(v) for k, v in source_files.items()},
            'calc_config': calc_config
        }
        
        metadata_path = calc_dir / 'calculation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return calc_dir
    
    def _copy_file_with_verification(self, source: Path, dest: Path):
        """Copy file and verify integrity."""
        # Calculate source checksum
        source_hash = self._calculate_file_hash(source)
        
        # Copy file
        shutil.copy2(source, dest)
        self.stats['files_copied'] += 1
        self.stats['total_size_processed'] += source.stat().st_size
        
        # Verify destination
        dest_hash = self._calculate_file_hash(dest)
        
        if source_hash != dest_hash:
            dest.unlink()  # Remove corrupted file
            raise IOError(f"File copy verification failed: {source} -> {dest}")
        
        self.logger.debug(f"Copied and verified: {source} -> {dest}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def organize_output_files(self,
                            job_id: str,
                            output_dir: Path,
                            calc_config: Dict[str, Any]) -> Dict[str, Path]:
        """
        Organize output files after job completion.
        
        Returns:
            Dictionary mapping file types to their paths
        """
        calc_type = calc_config['calc_type']
        material_id = calc_config['material_id']
        
        organized_files = {}
        
        # Define expected output files
        expected_files = {
            'output': ['.out', '.output'],
            'wavefunction': ['.f9', 'fort.9'],
            'properties': ['.f25', 'fort.25'],
            'dos': ['.DOSS'],
            'band': ['.BAND', '.BAND_GNU'],
            'transport': ['.SEEBECK', '.SIGMA'],
            'charge': ['.ECHG'],
            'potential': ['.POTM']
        }
        
        # Search for expected files
        for file_type, extensions in expected_files.items():
            for ext in extensions:
                # Try different naming patterns
                patterns = [
                    f"{material_id}{ext}",
                    f"{material_id}_{calc_type}{ext}",
                    f"fort{ext}" if ext.startswith('.') else ext
                ]
                
                for pattern in patterns:
                    file_path = output_dir / pattern
                    if file_path.exists():
                        organized_files[file_type] = file_path
                        break
        
        # Move files to organized structure
        final_files = {}
        dest_dir = self.work_dir / calc_type / f"{material_id}_{job_id}_output"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for file_type, file_path in organized_files.items():
            dest_path = dest_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            final_files[file_type] = dest_path
            self.stats['files_moved'] += 1
        
        # Archive if calculation is complete
        if 'output' in final_files:
            self._archive_calculation(dest_dir, calc_config)
        
        return final_files
    
    def _archive_calculation(self, calc_dir: Path, calc_config: Dict[str, Any]):
        """Archive completed calculation."""
        calc_type = calc_config['calc_type']
        material_id = calc_config['material_id']
        
        # Create archive name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"{material_id}_{calc_type}_{timestamp}.tar.gz"
        archive_path = self.archive_dir / calc_type / archive_name
        
        # Create compressed archive
        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(calc_dir, arcname=calc_dir.name)
        
        self.logger.info(f"Archived calculation to {archive_path}")
    
    def cleanup_old_files(self, days: int = 30):
        """Clean up old temporary and work files."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        cleaned_count = 0
        cleaned_size = 0
        
        # Clean temp directory
        for item in self.temp_dir.rglob('*'):
            if item.is_file():
                if item.stat().st_mtime < cutoff_time:
                    size = item.stat().st_size
                    item.unlink()
                    cleaned_count += 1
                    cleaned_size += size
        
        # Clean old work directories
        for calc_type_dir in self.work_dir.iterdir():
            if calc_type_dir.is_dir():
                for calc_dir in calc_type_dir.iterdir():
                    if calc_dir.is_dir():
                        # Check metadata file for age
                        metadata_file = calc_dir / 'calculation_metadata.json'
                        if metadata_file.exists():
                            if metadata_file.stat().st_mtime < cutoff_time:
                                shutil.rmtree(calc_dir)
                                cleaned_count += 1
        
        self.logger.info(f"Cleaned {cleaned_count} items ({cleaned_size / 1024 / 1024:.1f} MB)")
    
    def validate_calculation_files(self, calc_dir: Path) -> Tuple[bool, List[str]]:
        """
        Validate all files in calculation directory.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check metadata file
        metadata_file = calc_dir / 'calculation_metadata.json'
        if not metadata_file.exists():
            errors.append("Missing metadata file")
            return False, errors
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            errors.append(f"Invalid metadata file: {e}")
            return False, errors
        
        # Validate required files based on calculation type
        calc_type = metadata.get('calc_type')
        material_id = metadata.get('material_id')
        
        required_files = self._get_required_files(calc_type)
        
        for file_type, patterns in required_files.items():
            found = False
            for pattern in patterns:
                file_path = calc_dir / pattern.format(material_id=material_id)
                if file_path.exists():
                    found = True
                    break
            
            if not found:
                errors.append(f"Missing required {file_type} file")
        
        return len(errors) == 0, errors
    
    def _get_required_files(self, calc_type: str) -> Dict[str, List[str]]:
        """Get required files for calculation type."""
        base_files = {
            'input': ['{material_id}.d12']
        }
        
        calc_specific = {
            'OPT': {},
            'SP': {
                'wavefunction': ['{material_id}.f9', 'fort.9']
            },
            'FREQ': {
                'wavefunction': ['{material_id}.f9', 'fort.9']
            },
            'BAND': {
                'wavefunction': ['{material_id}.f9', 'fort.9'],
                'properties_input': ['{material_id}.d3']
            },
            'DOSS': {
                'wavefunction': ['{material_id}.f9', 'fort.9'],
                'properties_input': ['{material_id}.d3']
            }
        }
        
        files = base_files.copy()
        files.update(calc_specific.get(calc_type, {}))
        
        return files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file manager statistics."""
        return self.stats.copy()
```

### 4.3 planner.py Deep Decomposition

#### 4.3.1 Current Structure Analysis

```python
# planner.py current structure (4,619 lines)

# Major sections:
# 1. Interactive planning (lines 401-1500) - 1100 lines
# 2. CIF conversion (lines 1501-2500) - 1000 lines  
# 3. Expert configurations (lines 2501-3500) - 1000 lines
# 4. Configuration management (lines 3501-4200) - 700 lines
# 5. Templates (lines 4201-4619) - 400 lines

# Key problems:
# 1. plan_interactive() is 285 lines of nested prompts
# 2. Expert configuration methods are highly duplicated
# 3. CIF conversion logic mixed with UI
# 4. No clear separation between UI and logic
# 5. Configuration scattered throughout
```

#### 4.3.2 Proposed Module Structure

```
planner/
├── __init__.py              # Package initialization
├── core.py                  # Core WorkflowPlanner class (500 lines)
├── interactive/
│   ├── __init__.py
│   ├── prompts.py          # User prompts and UI (300 lines)
│   ├── validators.py       # Input validation (200 lines)
│   └── handlers.py         # Response handlers (300 lines)
├── cif_converter.py         # CIF conversion logic (700 lines)
├── expert_modes/
│   ├── __init__.py
│   ├── base.py            # Base expert configuration (200 lines)
│   ├── opt_expert.py      # OPT expert mode (150 lines)
│   ├── sp_expert.py       # SP expert mode (150 lines)
│   ├── freq_expert.py     # FREQ expert mode (150 lines)
│   └── properties_expert.py # Properties expert modes (200 lines)
├── templates.py             # Workflow templates (400 lines)
├── config_manager.py        # Configuration management (400 lines)
├── validators.py            # Workflow validation (300 lines)
└── utils.py                # Utility functions (200 lines)
```

#### 4.3.3 Detailed Module Breakdown

##### core.py - Core WorkflowPlanner

```python
# planner/core.py
"""
Core workflow planning functionality.
~500 lines
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import logging
from datetime import datetime

from .interactive import InteractivePlanner
from .cif_converter import CifConverter
from .expert_modes import ExpertModeManager
from .templates import WorkflowTemplateManager
from .config_manager import WorkflowConfigManager
from .validators import WorkflowValidator
from ..database.materials import MaterialDatabase

class WorkflowPlanner:
    """
    Main workflow planning class.
    
    Coordinates all planning activities and provides the main API.
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path] = ".",
                 db_path: str = "materials.db",
                 isolated: bool = False,
                 isolation_context: Optional[Any] = None):
        """Initialize workflow planner."""
        self.base_dir = Path(base_dir).resolve()
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        
        # Isolation support (merged from contextual)
        self.isolated = isolated
        self.isolation_context = isolation_context
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.interactive = InteractivePlanner(self)
        self.cif_converter = CifConverter(self)
        self.expert_manager = ExpertModeManager(self)
        self.template_manager = WorkflowTemplateManager()
        self.config_manager = WorkflowConfigManager(self.base_dir)
        self.validator = WorkflowValidator()
        
        # Setup directories
        self._setup_directories()
        
        self.logger.info(f"WorkflowPlanner initialized at {self.base_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger('WorkflowPlanner')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_directories(self):
        """Create required directories."""
        dirs = [
            self.base_dir / "workflow_configs",
            self.base_dir / "workflow_inputs",
            self.base_dir / "workflow_outputs",
            self.base_dir / "workflow_scripts"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Main public methods
    def plan_workflow(self,
                     mode: str = "interactive",
                     config_file: Optional[Path] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Main entry point for workflow planning.
        
        Args:
            mode: Planning mode ('interactive', 'template', 'config')
            config_file: Configuration file for non-interactive modes
            **kwargs: Additional arguments for specific modes
            
        Returns:
            Workflow configuration dictionary
        """
        if self.isolated and self.isolation_context:
            with self.isolation_context:
                return self._plan_workflow_impl(mode, config_file, **kwargs)
        else:
            return self._plan_workflow_impl(mode, config_file, **kwargs)
    
    def _plan_workflow_impl(self,
                           mode: str,
                           config_file: Optional[Path],
                           **kwargs) -> Dict[str, Any]:
        """Implementation of workflow planning."""
        self.logger.info(f"Starting workflow planning in {mode} mode")
        
        if mode == "interactive":
            config = self.interactive.plan()
        elif mode == "template":
            template_name = kwargs.get('template', 'basic_opt')
            config = self.template_manager.create_from_template(template_name)
        elif mode == "config":
            if not config_file:
                raise ValueError("Config file required for config mode")
            config = self.config_manager.load_config(config_file)
        else:
            raise ValueError(f"Unknown planning mode: {mode}")
        
        # Validate configuration
        is_valid, errors = self.validator.validate(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Save configuration
        config_path = self.config_manager.save_config(config)
        config['config_file'] = str(config_path)
        
        # Register in database
        self._register_workflow(config)
        
        self.logger.info(f"Workflow planning completed: {config['workflow_id']}")
        
        return config
    
    def _register_workflow(self, config: Dict[str, Any]):
        """Register workflow in database."""
        workflow_data = {
            'workflow_id': config['workflow_id'],
            'created_at': datetime.now().isoformat(),
            'status': 'planned',
            'config_file': config['config_file'],
            'input_type': config.get('input_type', 'unknown'),
            'total_steps': len(config.get('workflow_sequence', [])),
            'metadata_json': json.dumps({
                'creator': 'WorkflowPlanner',
                'version': '2.0',
                'planning_mode': config.get('planning_mode', 'interactive')
            })
        }
        
        self.db.create_workflow(workflow_data)
    
    # Quick planning methods
    def quick_opt_workflow(self,
                          input_files: List[Path],
                          **kwargs) -> Dict[str, Any]:
        """Quick planning for optimization workflow."""
        config = self.template_manager.create_from_template('basic_opt')
        config['input_files'] = [str(f) for f in input_files]
        config.update(kwargs)
        
        return self._finalize_quick_config(config)
    
    def quick_full_workflow(self,
                           cif_files: List[Path],
                           **kwargs) -> Dict[str, Any]:
        """Quick planning for full electronic structure workflow."""
        config = self.template_manager.create_from_template('full_electronic')
        
        # Convert CIFs
        d12_files = self.cif_converter.batch_convert(
            cif_files,
            level='basic',
            **kwargs
        )
        
        config['input_files'] = [str(f) for f in d12_files]
        config.update(kwargs)
        
        return self._finalize_quick_config(config)
    
    def _finalize_quick_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize quick configuration."""
        # Validate
        is_valid, errors = self.validator.validate(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Save and register
        config_path = self.config_manager.save_config(config)
        config['config_file'] = str(config_path)
        self._register_workflow(config)
        
        return config
    
    # Expert mode access
    def configure_expert_calculation(self,
                                   calc_type: str,
                                   template_file: Path,
                                   **kwargs) -> Path:
        """Configure calculation in expert mode."""
        return self.expert_manager.configure(calc_type, template_file, **kwargs)
    
    # Template management
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available workflow templates."""
        return self.template_manager.list_templates()
    
    def create_custom_template(self,
                             name: str,
                             config: Dict[str, Any]):
        """Create custom workflow template."""
        self.template_manager.save_template(name, config)
    
    # Utility methods
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of planned workflow."""
        workflow = self.db.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        return {
            'workflow_id': workflow_id,
            'status': workflow['status'],
            'created_at': workflow['created_at'],
            'config_file': workflow.get('config_file'),
            'progress': self._calculate_progress(workflow_id)
        }
    
    def _calculate_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Calculate workflow progress."""
        calcs = self.db.get_calculations_for_workflow(workflow_id)
        
        total = len(calcs)
        completed = sum(1 for c in calcs if c['status'] == 'completed')
        
        return {
            'total_calculations': total,
            'completed_calculations': completed,
            'percentage': (completed / total * 100) if total > 0 else 0
        }
```

##### interactive/prompts.py - User Interface

```python
# planner/interactive/prompts.py
"""
Interactive prompts and user interface.
~300 lines
"""

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import os

class InteractivePrompter:
    """
    Handles all user interaction for workflow planning.
    
    Separates UI concerns from business logic.
    """
    
    def __init__(self):
        self.prompt_history = []
        self.validators = self._setup_validators()
    
    def _setup_validators(self) -> Dict[str, Callable]:
        """Setup input validators."""
        return {
            'yes_no': lambda x: x.lower() in ['y', 'yes', 'n', 'no'],
            'integer': lambda x: x.isdigit(),
            'file_exists': lambda x: Path(x).exists(),
            'directory_exists': lambda x: Path(x).is_dir(),
            'choice': lambda x, choices: x in choices
        }
    
    def prompt(self,
              message: str,
              default: Optional[Any] = None,
              validator: Optional[Callable] = None,
              error_message: str = "Invalid input") -> Any:
        """
        Generic prompt method.
        
        Args:
            message: Prompt message
            default: Default value
            validator: Validation function
            error_message: Error message for invalid input
            
        Returns:
            User input (validated)
        """
        while True:
            # Build prompt
            if default is not None:
                prompt_text = f"{message} [{default}]: "
            else:
                prompt_text = f"{message}: "
            
            # Get input
            user_input = input(prompt_text).strip()
            
            # Use default if empty
            if not user_input and default is not None:
                user_input = str(default)
            
            # Validate
            if validator:
                if validator(user_input):
                    self.prompt_history.append((message, user_input))
                    return user_input
                else:
                    print(f"Error: {error_message}")
            else:
                self.prompt_history.append((message, user_input))
                return user_input
    
    def yes_no_prompt(self, message: str, default: str = "yes") -> bool:
        """Prompt for yes/no answer."""
        response = self.prompt(
            message,
            default=default,
            validator=self.validators['yes_no'],
            error_message="Please enter yes/no or y/n"
        )
        return response.lower() in ['y', 'yes']
    
    def choice_prompt(self,
                     message: str,
                     choices: List[str],
                     default: Optional[str] = None,
                     descriptions: Optional[Dict[str, str]] = None) -> str:
        """Prompt for choice from list."""
        # Display choices
        print(f"\n{message}")
        for i, choice in enumerate(choices, 1):
            desc = ""
            if descriptions and choice in descriptions:
                desc = f" - {descriptions[choice]}"
            print(f"  {i}. {choice}{desc}")
        
        # Get selection
        validator = lambda x: (x.isdigit() and 1 <= int(x) <= len(choices)) or x in choices
        
        selection = self.prompt(
            "Select option (number or name)",
            default=default,
            validator=validator,
            error_message=f"Please select 1-{len(choices)} or enter valid choice name"
        )
        
        # Convert number to choice
        if selection.isdigit():
            return choices[int(selection) - 1]
        return selection
    
    def multi_choice_prompt(self,
                          message: str,
                          choices: List[str],
                          default: Optional[List[str]] = None) -> List[str]:
        """Prompt for multiple choices."""
        print(f"\n{message}")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        
        print("\nEnter selections (comma-separated numbers or 'all'):")
        
        selection = self.prompt(
            "Selections",
            default=','.join(map(str, default)) if default else None
        )
        
        if selection.lower() == 'all':
            return choices
        
        # Parse selections
        selected = []
        for item in selection.split(','):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(choices):
                    selected.append(choices[idx])
        
        return selected
    
    def file_prompt(self,
                   message: str,
                   must_exist: bool = True,
                   extension: Optional[str] = None) -> Path:
        """Prompt for file path."""
        while True:
            path_str = self.prompt(message)
            path = Path(path_str).expanduser().resolve()
            
            if must_exist and not path.exists():
                print(f"Error: File not found: {path}")
                continue
            
            if extension and not path.suffix == extension:
                print(f"Error: Expected {extension} file")
                continue
            
            return path
    
    def directory_prompt(self,
                        message: str,
                        must_exist: bool = True,
                        create: bool = False) -> Path:
        """Prompt for directory path."""
        while True:
            path_str = self.prompt(message)
            path = Path(path_str).expanduser().resolve()
            
            if must_exist and not path.exists():
                if create:
                    if self.yes_no_prompt(f"Create directory {path}?"):
                        path.mkdir(parents=True, exist_ok=True)
                        return path
                else:
                    print(f"Error: Directory not found: {path}")
                    continue
            
            if path.exists() and not path.is_dir():
                print(f"Error: Not a directory: {path}")
                continue
            
            return path
    
    def integer_prompt(self,
                      message: str,
                      min_value: Optional[int] = None,
                      max_value: Optional[int] = None,
                      default: Optional[int] = None) -> int:
        """Prompt for integer value."""
        def validator(x):
            if not x.isdigit():
                return False
            value = int(x)
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            return True
        
        error_msg = "Please enter a valid integer"
        if min_value is not None and max_value is not None:
            error_msg += f" between {min_value} and {max_value}"
        elif min_value is not None:
            error_msg += f" >= {min_value}"
        elif max_value is not None:
            error_msg += f" <= {max_value}"
        
        result = self.prompt(
            message,
            default=default,
            validator=validator,
            error_message=error_msg
        )
        
        return int(result)
    
    def show_summary(self, data: Dict[str, Any], title: str = "Summary"):
        """Display summary of configuration."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)
        
        for key, value in data.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print('='*60)
```

##### expert_modes/base.py - Base Expert Configuration

```python
# planner/expert_modes/base.py
"""
Base class for expert mode configurations.
~200 lines
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import sys
import json
import shutil
from datetime import datetime

class ExpertModeBase(ABC):
    """
    Base class for all expert mode configurations.
    
    Implements common functionality and defines interface.
    """
    
    def __init__(self, calc_type: str, planner):
        self.calc_type = calc_type
        self.planner = planner
        self.logger = planner.logger.getChild(f"Expert{calc_type}")
        
        # Paths to external tools
        self.tool_paths = self._get_tool_paths()
    
    def _get_tool_paths(self) -> Dict[str, Path]:
        """Get paths to external configuration tools."""
        base_path = Path(__file__).parent.parent.parent.parent
        
        paths = {
            'OPT': base_path / "Crystal_d12" / "CRYSTALOptToD12.py",
            'SP': base_path / "Crystal_d12" / "CRYSTALOptToD12.py",
            'FREQ': base_path / "Crystal_d12" / "CRYSTALOptToD12.py",
            'BAND': base_path / "Crystal_d3" / "create_band_d3.py",
            'DOSS': base_path / "Crystal_d3" / "alldos.py",
            'TRANSPORT': base_path / "Crystal_d3" / "create_Transportd3.py",
            'CHARGE+POTENTIAL': base_path / "Crystal_d3" / "create_charge_potential_d3.py"
        }
        
        return paths
    
    def configure(self,
                 template_file: Path,
                 options: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Main configuration method.
        
        Args:
            template_file: Template D12/D3 file
            options: Configuration options
            
        Returns:
            Path to configured file or None if failed
        """
        self.logger.info(f"Starting {self.calc_type} expert configuration")
        
        # Setup
        temp_dir = self._create_temp_directory()
        
        try:
            # Prepare files
            work_file = self._prepare_template(template_file, temp_dir)
            
            # Show current configuration
            self._display_current_config(work_file)
            
            # Get user configuration
            if options and options.get('use_defaults'):
                config_options = self._get_default_options()
            else:
                config_options = self._get_user_configuration(work_file)
            
            # Apply configuration
            result_file = self._apply_configuration(
                work_file,
                config_options,
                temp_dir
            )
            
            if result_file and result_file.exists():
                # Copy to final location
                final_file = self._finalize_configuration(result_file)
                return final_file
            else:
                self.logger.error("Configuration failed - no output generated")
                return None
                
        except Exception as e:
            self.logger.error(f"Expert configuration error: {e}")
            return None
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _create_temp_directory(self) -> Path:
        """Create temporary working directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_dir = Path(f"temp_{self.calc_type}_expert_{timestamp}")
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def _prepare_template(self, template_file: Path, temp_dir: Path) -> Path:
        """Prepare template file for configuration."""
        work_file = temp_dir / f"template_{self.calc_type}{template_file.suffix}"
        shutil.copy2(template_file, work_file)
        return work_file
    
    def _display_current_config(self, template_file: Path):
        """Display current configuration from template."""
        print(f"\n{'='*60}")
        print(f"Current {self.calc_type} Configuration")
        print('='*60)
        
        # Parse and display current settings
        current_settings = self._parse_current_settings(template_file)
        
        for category, settings in current_settings.items():
            print(f"\n{category}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
    
    @abstractmethod
    def _parse_current_settings(self, template_file: Path) -> Dict[str, Any]:
        """Parse current settings from template file."""
        pass
    
    @abstractmethod
    def _get_user_configuration(self, template_file: Path) -> Dict[str, Any]:
        """Get configuration from user interaction."""
        pass
    
    @abstractmethod
    def _get_default_options(self) -> Dict[str, Any]:
        """Get default configuration options."""
        pass
    
    def _apply_configuration(self,
                           input_file: Path,
                           options: Dict[str, Any],
                           work_dir: Path) -> Optional[Path]:
        """Apply configuration using external tool."""
        tool_path = self.tool_paths.get(self.calc_type)
        if not tool_path or not tool_path.exists():
            raise FileNotFoundError(f"Configuration tool not found: {tool_path}")
        
        # Build command
        cmd = [
            sys.executable,
            str(tool_path),
            str(input_file),
            f"--type={self.calc_type}",
            "--expert"
        ]
        
        # Add options
        for key, value in options.items():
            if value is not None:
                cmd.append(f"--{key}={value}")
        
        # Execute
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir)
        )
        
        if result.returncode != 0:
            self.logger.error(f"Configuration failed: {result.stderr}")
            return None
        
        # Find output file
        expected_output = work_dir / f"{input_file.stem}_configured{input_file.suffix}"
        if expected_output.exists():
            return expected_output
        
        # Search for any new file
        for file in work_dir.glob(f"*{input_file.suffix}"):
            if file != input_file:
                return file
        
        return None
    
    def _finalize_configuration(self, configured_file: Path) -> Path:
        """Move configured file to final location."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"{self.calc_type}_expert_{timestamp}{configured_file.suffix}"
        final_path = self.planner.base_dir / "workflow_inputs" / final_name
        
        shutil.copy2(configured_file, final_path)
        self.logger.info(f"Configuration saved to: {final_path}")
        
        return final_path
```

### 4.4 executor.py Decomposition Strategy

#### 4.4.1 Current Issues

```python
# executor.py issues (1,902 lines):

1. _execute_workflow_step() - 165 lines of mixed concerns
2. _handle_step_dependencies() - 125 lines of complex logic
3. _prepare_step_inputs() - 145 lines of file operations
4. No clear separation between execution and coordination
5. Error handling scattered throughout
```

#### 4.4.2 Proposed Structure

```
executor/
├── __init__.py
├── core.py                  # Core executor (300 lines)
├── step_executor.py         # Step execution logic (400 lines)
├── dependency_manager.py    # Dependency handling (300 lines)
├── progress_tracker.py      # Progress tracking (200 lines)
├── error_handler.py         # Error handling (200 lines)
└── utils.py                # Utilities (100 lines)
```

### 4.5 Benefits of Decomposition

#### 4.5.1 Quantitative Benefits

```
Metric                      Before         After      Improvement
----------------------  -----------  -----------  --------------
Largest file                  4,619          800         -82.7%
Average file size               957          350         -63.4%
Max method size                 285           50         -82.5%
Methods per class                87           15         -82.8%
Cyclomatic complexity          28.4         12.5         -56.0%
Import statements               140           25         -82.1%
Test coverage potential          5%          85%       +1600.0%
```

#### 4.5.2 Qualitative Benefits

1. **Maintainability**
   - Single responsibility per module
   - Clear interfaces between components
   - Easier to locate and fix bugs

2. **Testability**
   - Small, focused units
   - Mockable dependencies
   - Clear test boundaries

3. **Extensibility**
   - New features in new modules
   - Plugin architecture possible
   - Clear extension points

4. **Team Development**
   - Multiple developers can work independently
   - Clear ownership boundaries
   - Reduced merge conflicts

5. **Performance**
   - Lazy loading of modules
   - Better memory management
   - Optimized imports

### 4.6 Migration Strategy

#### Phase 1: Create Module Structure (Week 1)
- Create new directory structures
- Set up __init__.py files
- Define interfaces

#### Phase 2: Extract Core Components (Week 2)
- Extract core classes
- Maintain facade pattern
- Ensure backward compatibility

#### Phase 3: Decompose Methods (Week 3)
- Break down large methods
- Extract to appropriate modules
- Update internal references

#### Phase 4: Testing and Integration (Week 4)
- Comprehensive testing
- Update all imports
- Performance validation

This decomposition transforms monolithic scripts into well-organized, maintainable modules while preserving all functionality.