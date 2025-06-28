# Detailed Analysis of Duplicate Functions in Workflow System

## 1. SLURM Submission Duplicates

### Multiple Implementations Found:

#### A. **workflow_engine.py**
```python
def _submit_calculation_to_slurm(self, script_path: Path, work_dir: Path) -> Optional[str]:
    # Lines 489-567
    # Handles both template scripts and direct submission
    # Changes to work_dir before submission
    # Checks for script generator templates
```

#### B. **workflow_executor.py**
```python
def submit_jobs_for_step(self, step_dir: Path, step_config: Dict) -> List[str]:
    # Different approach - calls subprocess for SLURM scripts
    # Has different error handling
```

#### C. **enhanced_queue_manager.py**
```python
def submit_job(self, job_file: str) -> Optional[str]:
    # Lines ~800-850
    # Another implementation with different logic
    # Integrated with material tracking
```

#### D. **crystal_queue_manager.py** (legacy)
```python
def submit_job(self, d12_file: str) -> bool:
    # Old implementation, still referenced in some places
```

### Issues with Multiple SLURM Submissions:
- **Inconsistent error handling** across implementations
- **Different working directory handling** (some change dir, some don't)
- **Template detection logic** varies between implementations
- **Job ID extraction** patterns differ
- **Logging and reporting** inconsistent

## 2. Material ID Extraction Duplicates

### At Least 6 Different Implementations:

#### A. **material_database.py** (authoritative)
```python
def create_material_id_from_file(filename: str) -> str:
    """Smart extraction of material ID from complex filenames"""
    # This should be the single source of truth
```

#### B. **workflow_engine.py**
```python
def extract_core_material_id_from_complex_filename(self, filename: str) -> str:
    # Lines 568-580
    # Just calls create_material_id_from_file but adds wrapper
    
def get_material_id_from_any_file(self, file_path: Path) -> str:
    # Lines 581-595
    # Adds database lookup before extraction
```

#### C. **workflow_executor.py**
```python
def extract_material_name(self, filename: str) -> str:
    # Different implementation with different logic
```

#### D. **enhanced_queue_manager.py**
```python
def extract_material_id(self, filename: str) -> str:
    # Yet another implementation
```

#### E. **crystal_property_extractor.py**
```python
# Has its own material ID extraction logic
```

#### F. **Various scripts in Archived/**
- Multiple fix scripts each with their own extraction logic

### Issues with Material ID Extraction:
- **No single source of truth** despite material_database.py having the best implementation
- **Different algorithms** produce different results for edge cases
- **Complex filenames** from NewCifToD12.py handled differently
- **Special characters** handled inconsistently

## 3. Cleanup Method Duplicates

### Multiple Cleanup Implementations:

#### A. **workflow_engine.py**
```python
def _cleanup_old_workflow_dirs(self):
    # Lines 96-110
    # Cleans workflow_staging directories
    
def _cleanup_failed_workflow_dirs(self):
    # Added in recent fixes
    # Cleans failed workflow output directories
```

#### B. **workflow_executor.py**
```python
def cleanup_temp_files(self):
    # Different approach, cleans temp directory
```

#### C. **material_database.py**
```python
def cleanup_old_calculations(self, days: int = 30):
    # Database-level cleanup
```

#### D. **enhanced_queue_manager.py**
```python
def cleanup_completed_jobs(self):
    # Cleans up after job completion
```

### Issues with Cleanup Methods:
- **No coordination** between different cleanup strategies
- **Different retention policies** 
- **Potential for premature deletion** if multiple cleanups run
- **No unified logging** of what was cleaned

## 4. Script Location Resolution Issues

### Inconsistent Path Resolution:

#### A. **workflow_engine.py**
```python
def get_script_paths(self):
    script_locations = {
        'NewCifToD12.py': [
            self.base_work_dir.parent / "Crystal_To_CIF",
            Path("/path/to/scripts/Crystal_To_CIF")
        ],
        # Multiple fallback paths
    }
```

#### B. **workflow_executor.py**
```python
# Uses different path resolution:
sys.path.append(str(Path(__file__).parent.parent / "Crystal_To_CIF"))
```

#### C. **Enhanced callback mechanism**
```bash
# In SLURM scripts, checks multiple levels:
if [ -f $DIR/enhanced_queue_manager.py ]; then
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
# Up to 5 levels deep
```

### Issues with Script Resolution:
- **No central registry** of script locations
- **Hardcoded paths** that break on different systems
- **Multiple fallback strategies** make debugging difficult
- **sys.path pollution** from multiple appends

## 5. Configuration Sprawl Examples

### Hardcoded Values Found Throughout:

#### A. **SLURM Defaults**
- **workflow_planner.py**: Lines ~2500-2600
```python
"cores": 32, "memory": "5G", "walltime": "7-00:00:00"  # For OPT
"cores": 28, "memory": "48G", "walltime": "1-00:00:00"  # For BAND
```

- **workflow_engine.py**: Different defaults
```python
"--mem": "4GB", "--time": "24:00:00"  # Different values!
```

- **submitcrystal23.sh**: Yet another set
```bash
#SBATCH --mem=5GB
#SBATCH --time=7-00:00:00
```

#### B. **Calculation Settings**
- **Tolerances** hardcoded in multiple places with different values
- **Convergence criteria** varies between scripts
- **Grid settings** inconsistent

#### C. **File Patterns**
- **Output file extensions** defined differently
- **Template locations** hardcoded in multiple places
- **Basis set paths** repeated throughout

### Issues with Configuration:
- **No single configuration file** for defaults
- **Changes require updating multiple files**
- **Risk of inconsistency** between components
- **Hard to customize** for different clusters

## Recommendations for Fixing Duplicates

### 1. **Create Service Classes**
```python
# slurm_service.py
class SlurmService:
    """Single service for all SLURM operations"""
    def submit(self, script_path: Path, **kwargs) -> str:
        # One implementation for all SLURM submissions
        
# material_id_service.py  
class MaterialIdService:
    """Single service for material ID operations"""
    def extract(self, filename: str) -> str:
        # Just use material_database.create_material_id_from_file
        
# cleanup_service.py
class CleanupService:
    """Coordinated cleanup operations"""
    def cleanup(self, age_days: int = 7):
        # Unified cleanup with proper coordination
```

### 2. **Centralize Configuration**
```yaml
# workflow_config.yaml
slurm_defaults:
  OPT:
    cores: 32
    memory: "5G"
    walltime: "7-00:00:00"
    account: "mendoza_q"
  BAND:
    cores: 28
    memory: "48G"
    walltime: "1-00:00:00"
    account: "general"
    
script_locations:
  NewCifToD12.py: "../Crystal_To_CIF"
  CRYSTALOptToD12.py: "../Crystal_To_CIF"
  create_band_d3.py: "../Creation_Scripts"
```

### 3. **Use Dependency Injection**
```python
class WorkflowManager:
    def __init__(self, slurm_service, material_service, cleanup_service):
        self.slurm = slurm_service
        self.materials = material_service
        self.cleanup = cleanup_service
```

This approach would eliminate duplicates and create a much more maintainable system.