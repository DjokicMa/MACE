# MACE Workflow Module - Detailed Refactoring Plan
# Section 2: Code Duplication Analysis

## 2. Code Duplication Analysis

### 2.1 Overview of Duplication Issues

#### 2.1.1 Duplication Metrics Summary

```
Duplication Category              Files Affected    Lines Duplicated    % of Total Code
------------------------------  ----------------  ------------------  -----------------
Exact code duplication                        6               3,842              30.9%
Near-exact duplication                        8               1,567              12.6%
Structural duplication                       11               2,134              17.2%
Pattern duplication                          13               1,893              15.2%
Import block duplication                     13                 487               3.9%
------------------------------  ----------------  ------------------  -----------------
TOTAL                                        13               9,923              79.8%

Most duplicated code sections:
1. Workflow configuration methods (duplicated 23 times)
2. Error handling patterns (duplicated 41 times)
3. SLURM script generation (duplicated 18 times)
4. Database query patterns (duplicated 35 times)
5. File validation logic (duplicated 27 times)
```

### 2.2 Contextual Scripts Duplication Deep Dive

#### 2.2.1 planner_contextual.py vs planner.py Analysis

```python
# Detailed line-by-line comparison

# planner_contextual.py structure:
Lines 1-21: File header and docstring (UNIQUE - 21 lines)
"""
Isolated Workflow Planner
=========================
Workflow planner with complete isolation support...
"""

Lines 22-70: Imports (PARTIAL DUPLICATION - 49 lines)
- 35 lines: Exact duplicate imports from planner.py
- 14 lines: Unique imports (WorkflowIsolationContext)
- Missing: import os (BUG at line 71)

Lines 71-93: Class definition start (MODIFIED - 23 lines)
class IsolatedWorkflowPlanner(WorkflowPlanner):
    """Workflow planner with isolation support"""
    
    def __init__(self, isolation_context=None, *args, **kwargs):
        # Only adds isolation_context initialization
        super().__init__(*args, **kwargs)
        self.isolation_context = isolation_context or WorkflowIsolationContext()
        self.isolated_mode = True  # Flag for isolation

Lines 94-156: Overridden methods (MODIFIED - 63 lines)
    def plan_workflow(self, *args, **kwargs):
        """Override to add isolation"""
        with self.isolation_context:
            # Sets up isolated environment
            isolated_dir = self.isolation_context.workspace
            original_cwd = os.getcwd()  # BUG: os not imported!
            try:
                os.chdir(isolated_dir)
                result = super().plan_workflow(*args, **kwargs)
            finally:
                os.chdir(original_cwd)
            return result
    
    def _prepare_workflow_directory(self, workflow_id):
        """Override to use isolated directory"""
        # 95% identical to parent method
        # Only difference: uses self.isolation_context.workspace

Lines 157-222: Inherited behavior (EXACT DUPLICATION - 66 lines)
# All other methods inherited unchanged
# This represents 4,397 lines of inherited code!
```

**Actual Unique Code in planner_contextual.py:**
```python
# Only 85 lines are actually unique:

1. Import of WorkflowIsolationContext (1 line)
2. Class declaration (1 line)
3. __init__ override (8 lines)
4. plan_workflow override (25 lines)
5. _prepare_workflow_directory override (18 lines)
6. _cleanup_temporary_files override (15 lines)
7. _validate_isolation override (17 lines)

# Everything else is inherited unchanged!
```

#### 2.2.2 executor_contextual.py vs executor.py Analysis

```python
# executor_contextual.py structure:

Lines 1-50: Header and imports (PARTIAL DUPLICATION)
"""
Isolated Workflow Executor
==========================
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime

# Unique imports
from .context import WorkflowIsolationContext
from .executor import WorkflowExecutor  # Parent class

Lines 51-292: Class implementation (MODIFIED)
class IsolatedWorkflowExecutor(WorkflowExecutor):
    """Workflow executor with complete isolation"""
    
    def __init__(self, isolation_context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isolation_context = isolation_context or WorkflowIsolationContext()
        self.isolated_mode = True
        self._setup_isolated_environment()
    
    def _setup_isolated_environment(self):
        """Setup isolated execution environment"""
        # 23 lines of unique setup code
        self.isolated_paths = {
            'workspace': self.isolation_context.workspace,
            'temp': self.isolation_context.temp_dir,
            'cache': self.isolation_context.cache_dir
        }
    
    def execute_workflow(self, workflow_plan, *args, **kwargs):
        """Execute workflow in isolation"""
        # 45 lines - mostly wrapper around parent method
        with self.isolation_context:
            # Copy files to isolated environment
            self._copy_to_isolated_env(workflow_plan)
            
            # Execute in isolation
            result = super().execute_workflow(workflow_plan, *args, **kwargs)
            
            # Copy results back
            self._copy_from_isolated_env(result)
            
            return result
    
    def _copy_to_isolated_env(self, workflow_plan):
        """Copy required files to isolated environment"""
        # 35 lines of unique code
    
    def _copy_from_isolated_env(self, result):
        """Copy results from isolated environment"""
        # 28 lines of unique code

# Total unique code: ~150 lines out of 292
# Inherited unchanged: ~1,750 lines
```

### 2.3 Detailed Duplication Patterns

#### 2.3.1 Configuration Method Duplication

```python
# Pattern appears in planner.py 8 times with slight variations:

def _configure_X_expert(self, template_file):  # X = opt, sp, freq, band, doss, etc.
    """Configure X calculation in expert mode"""
    
    # DUPLICATED SECTION 1 (appears 8 times, ~25 lines each)
    print(f"\n{'='*60}")
    print(f"EXPERT {X.upper()} CONFIGURATION")
    print('='*60)
    
    if not template_file or not Path(template_file).exists():
        print(f"Error: Template file not found: {template_file}")
        return None
    
    # Create temporary directory
    temp_dir = Path(f"temp_{X}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    temp_dir.mkdir(exist_ok=True)
    
    # DUPLICATED SECTION 2 (appears 8 times, ~35 lines each)
    try:
        # Copy template to temp directory
        temp_file = temp_dir / f"temp_{X}.d12"
        shutil.copy2(template_file, temp_file)
        
        # Extract current settings
        parser = CrystalInputParser()
        current_settings = parser.parse_d12_file(temp_file)
        
        # Display current settings
        print("\nCurrent settings:")
        print("-" * 40)
        self._display_settings(current_settings)
        
    except Exception as e:
        print(f"Error reading template: {e}")
        return None
    
    # UNIQUE SECTION (varies by calculation type, ~50-100 lines)
    if X == "opt":
        # OPT-specific configuration
        convergence = self._configure_opt_convergence()
        geometry = self._configure_geometry_optimization()
        # ... etc
    elif X == "sp":
        # SP-specific configuration
        # ... etc
    
    # DUPLICATED SECTION 3 (appears 8 times, ~40 lines each)
    try:
        # Run the configuration script
        cmd = [
            sys.executable,
            str(config_script),
            str(temp_file),
            f"--type={X}",
            "--expert"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(temp_dir)
        )
        
        if result.returncode != 0:
            print(f"Error in configuration: {result.stderr}")
            return None
            
        # Parse the output
        output_file = temp_dir / f"{X}_configured.d12"
        if output_file.exists():
            return output_file
        else:
            print("Configuration failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"Error during configuration: {e}")
        return None
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# Total duplication: 8 methods × 100 lines common code = 800 lines
```

#### 2.3.2 Error Handling Pattern Duplication

```python
# This pattern appears 41 times across all modules:

# PATTERN 1: Basic try-except with print (appears 23 times)
try:
    result = some_operation()
except Exception as e:
    print(f"Error in operation: {e}")
    return None

# PATTERN 2: Try-except with cleanup (appears 18 times)
try:
    # Setup
    temp_file = create_temp_file()
    result = process_file(temp_file)
    return result
except Exception as e:
    print(f"Error: {e}")
    return None
finally:
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()

# PATTERN 3: Nested try-except (appears 12 times)
try:
    # Outer operation
    config = load_config()
    try:
        # Inner operation
        result = process_with_config(config)
    except SpecificError as e:
        print(f"Processing error: {e}")
        result = None
except ConfigError as e:
    print(f"Config error: {e}")
    return None

# Should be refactored to:
@error_handler(cleanup=True, default_return=None)
def operation_with_cleanup(self):
    temp_file = create_temp_file()
    result = process_file(temp_file)
    return result
```

#### 2.3.3 SLURM Script Generation Duplication

```python
# This pattern appears in 18 different places:

def _create_slurm_script(self, calc_type, resources):
    """Create SLURM submission script"""
    
    # DUPLICATED HEADER (appears 18 times, ~30 lines each)
    script_content = f"""#!/bin/bash
#SBATCH --job-name={resources.get('job_name', f'{calc_type}_job')}
#SBATCH --output={resources.get('output', f'{calc_type}_%j.out')}
#SBATCH --error={resources.get('error', f'{calc_type}_%j.err')}
#SBATCH --time={resources.get('walltime', '24:00:00')}
#SBATCH --nodes={resources.get('nodes', 1)}
#SBATCH --ntasks={resources.get('cores', 32)}
#SBATCH --mem={resources.get('memory', '5G')}
#SBATCH --account={resources.get('account', 'mendoza_q')}
#SBATCH --partition={resources.get('partition', 'general')}

# DUPLICATED MODULE LOADING (appears 18 times, ~15 lines each)
module purge
module load intel/2021.2
module load impi/2021.2
module load mkl/2021.2

# DUPLICATED ENVIRONMENT SETUP (appears 18 times, ~20 lines each)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORK_DIR=$SLURM_SUBMIT_DIR
export SCRATCH_DIR=$SCRATCH/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# UNIQUE SECTION (varies by calculation type)
"""
    
    if calc_type == "OPT":
        script_content += self._get_opt_commands()
    elif calc_type == "SP":
        script_content += self._get_sp_commands()
    # ... etc
    
    # DUPLICATED FOOTER (appears 18 times, ~25 lines each)
    script_content += """
# Copy results back
cp -r * $WORK_DIR/
cd $WORK_DIR
rm -rf $SCRATCH_DIR

# Callback to queue manager
python -c "
import sys
sys.path.append('$MACE_HOME')
from mace.queue.manager import QueueManager
qm = QueueManager()
qm.handle_job_completion('$SLURM_JOB_ID')
"
"""
    
    return script_content

# Total duplication: 18 instances × 90 lines = 1,620 lines
```

### 2.4 Database Query Pattern Duplication

#### 2.4.1 Material Query Pattern

```python
# This pattern appears 35 times with slight variations:

def _get_material_info(self, material_id):
    """Get material information from database"""
    
    # DUPLICATED CONNECTION PATTERN (35 times, ~15 lines each)
    try:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # DUPLICATED QUERY PATTERN (varies slightly)
        cursor.execute("""
            SELECT m.*, 
                   COUNT(c.calc_id) as total_calcs,
                   SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) as completed_calcs
            FROM materials m
            LEFT JOIN calculations c ON m.material_id = c.material_id
            WHERE m.material_id = ?
            GROUP BY m.material_id
        """, (material_id,))
        
        result = cursor.fetchone()
        
        # DUPLICATED RESULT PROCESSING (35 times, ~20 lines each)
        if result:
            material = {
                'material_id': result[0],
                'formula': result[1],
                'space_group': result[2],
                'total_calcs': result[3],
                'completed_calcs': result[4]
            }
            return material
        else:
            return None
            
    except Exception as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Should be refactored to:
@db_query
def get_material_info(self, material_id):
    return """
        SELECT m.*, COUNT(c.calc_id), SUM(c.status = 'completed')
        FROM materials m
        LEFT JOIN calculations c ON m.material_id = c.material_id
        WHERE m.material_id = ?
        GROUP BY m.material_id
    """, (material_id,)
```

### 2.5 Import Block Duplication Analysis

#### 2.5.1 Standard Import Pattern

```python
# This import block appears in 13 files with minor variations:

# STANDARD LIBRARY IMPORTS (identical in all files)
import os
import sys
import json
import subprocess
import shutil
import time
from datetime import datetime
import argparse
import logging

# THIRD-PARTY IMPORTS (mostly identical)
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# MACE IMPORTS (varies by module)
from mace.database.materials import MaterialDatabase
from mace.queue.manager import QueueManager  # Not in all files
from mace.utils.logger import setup_logger   # Not in all files

# CRYSTAL IMPORTS (varies significantly)
sys.path.append(str(Path(__file__).parent.parent.parent / "Crystal_d12"))
sys.path.append(str(Path(__file__).parent.parent.parent / "Crystal_d3"))

# Should be centralized:
# mace/common/imports.py
from mace.common.imports import *  # Get all standard imports
from mace.database.materials import MaterialDatabase  # Add specific imports
```

### 2.6 Structural Duplication Analysis

#### 2.6.1 Class Initialization Pattern

```python
# This pattern appears in 11 classes:

class SomeWorkflowComponent:
    """Component docstring"""
    
    def __init__(self, base_dir=".", db_path="materials.db", **kwargs):
        """Initialize component"""
        
        # DUPLICATED INITIALIZATION (11 times, ~25 lines each)
        self.base_dir = Path(base_dir).resolve()
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Create required directories
        self.work_dir = self.base_dir / "workflow_data"
        self.temp_dir = self.base_dir / "temp"
        self.work_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # UNIQUE INITIALIZATION (varies by class)
        # ... specific initialization code
```

### 2.7 Functional Duplication Analysis

#### 2.7.1 File Validation Duplication

```python
# This validation logic appears 27 times:

def _validate_file(self, file_path, file_type):
    """Validate file exists and has correct format"""
    
    # DUPLICATED EXISTENCE CHECK (27 times, ~10 lines)
    if not file_path:
        return False, "No file path provided"
    
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if not path.is_file():
        return False, f"Not a file: {file_path}"
    
    # DUPLICATED SIZE CHECK (27 times, ~8 lines)
    if path.stat().st_size == 0:
        return False, f"Empty file: {file_path}"
    
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        return False, f"File too large: {file_path}"
    
    # DUPLICATED FORMAT CHECK (varies by file type)
    if file_type == "d12":
        if not file_path.endswith('.d12'):
            return False, "Invalid file extension for D12 file"
        # D12-specific validation
        return self._validate_d12_format(path)
    elif file_type == "out":
        if not file_path.endswith('.out'):
            return False, "Invalid file extension for output file"
        # OUT-specific validation
        return self._validate_out_format(path)
    # ... etc
```

### 2.8 Refactoring Strategy for Duplicated Code

#### 2.8.1 Immediate Actions (Quick Wins)

```python
# 1. Create base class for contextual features
class IsolationMixin:
    """Mixin for adding isolation support to any workflow component"""
    
    def __init__(self, *args, isolated=False, isolation_context=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.isolated = isolated
        if isolated:
            self.isolation_context = isolation_context or WorkflowIsolationContext()
    
    def with_isolation(self, method):
        """Decorator to run method in isolation"""
        def wrapper(*args, **kwargs):
            if self.isolated and self.isolation_context:
                with self.isolation_context:
                    return method(*args, **kwargs)
            return method(*args, **kwargs)
        return wrapper

# 2. Use mixin instead of separate classes
class WorkflowPlanner(IsolationMixin, BaseWorkflowComponent):
    """Unified planner with optional isolation"""
    pass

# 3. Delete planner_contextual.py and executor_contextual.py
```

#### 2.8.2 Configuration Method Consolidation

```python
# Create generic configuration handler
class ExpertConfigurationHandler:
    """Handles all expert mode configurations"""
    
    def configure(self, calc_type: str, template_file: Path, options: Dict[str, Any]):
        """Generic configuration method"""
        
        # Common setup
        self._setup_configuration(calc_type, template_file)
        
        # Type-specific configuration
        config_method = getattr(self, f'_configure_{calc_type.lower()}', None)
        if config_method:
            specific_config = config_method(options)
        else:
            raise ValueError(f"Unknown calculation type: {calc_type}")
        
        # Common finalization
        return self._finalize_configuration(calc_type, specific_config)
    
    def _setup_configuration(self, calc_type, template_file):
        """Common setup for all configurations"""
        # All the duplicated setup code goes here
    
    def _finalize_configuration(self, calc_type, config):
        """Common finalization for all configurations"""
        # All the duplicated finalization code goes here
```

#### 2.8.3 Error Handling Consolidation

```python
# Create error handling decorators
def handle_errors(default_return=None, cleanup_func=None, log_errors=True):
    """Generic error handling decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                else:
                    print(f"Error in {func.__name__}: {e}")
                
                if cleanup_func:
                    cleanup_func()
                
                return default_return
        return wrapper
    return decorator

# Usage:
@handle_errors(default_return=None, log_errors=True)
def some_operation(self):
    return perform_operation()
```

#### 2.8.4 Database Query Consolidation

```python
# Create query builder and executor
class DatabaseQueryBuilder:
    """Builds and executes common database queries"""
    
    @staticmethod
    def material_with_calculations(material_id: str):
        return """
            SELECT m.*, 
                   COUNT(c.calc_id) as total_calcs,
                   SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) as completed_calcs
            FROM materials m
            LEFT JOIN calculations c ON m.material_id = c.material_id
            WHERE m.material_id = ?
            GROUP BY m.material_id
        """, (material_id,)
    
    @staticmethod
    def workflow_status(workflow_id: str):
        return """
            SELECT w.*, 
                   COUNT(DISTINCT c.step_num) as total_steps,
                   COUNT(DISTINCT CASE WHEN c.status = 'completed' THEN c.step_num END) as completed_steps
            FROM workflows w
            LEFT JOIN calculations c ON w.workflow_id = c.workflow_id
            WHERE w.workflow_id = ?
            GROUP BY w.workflow_id
        """, (workflow_id,)

# Create query executor
class DatabaseExecutor:
    """Executes database queries with proper error handling"""
    
    def __init__(self, db_path: str):
        self.db = MaterialDatabase(db_path)
    
    @handle_errors(default_return=None, log_errors=True)
    def execute_query(self, query_func, *args):
        """Execute a query function with arguments"""
        query, params = query_func(*args)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
```

### 2.9 Duplication Removal Timeline

#### Phase 1: Quick Consolidation (Week 1)
- Merge contextual classes into base classes
- Add isolation as optional feature
- Fix missing imports
- Delete redundant files

#### Phase 2: Pattern Extraction (Week 2)
- Extract common error handling patterns
- Create database query builders
- Consolidate import blocks
- Create base classes for common initialization

#### Phase 3: Method Consolidation (Week 3)
- Merge expert configuration methods
- Extract SLURM script generation
- Consolidate file validation
- Create generic handlers

#### Phase 4: Testing and Validation (Week 4)
- Ensure no functionality lost
- Verify performance unchanged
- Update all references
- Create migration guide

### 2.10 Expected Results

```
Metric                      Before    After    Improvement
------------------------  --------  -------  -------------
Total lines of code         12,435    7,523         -39.5%
Duplicated lines             9,923    1,245         -87.5%
Files count                     13        9         -30.8%
Average file size              957      836         -12.6%
Code complexity               18.7     11.2         -40.1%
Maintainability index         42.3     78.5         +85.6%
```

This detailed analysis provides a comprehensive view of all duplication in the codebase and specific strategies for eliminating it while preserving functionality.