# MACE Workflow Module Refactoring Plan

## Current State Analysis

### 1. Script Inventory and Classification

#### Production Scripts (Core Functionality)
- **planner.py** (4,619 lines) - Main workflow planning system
  - Handles interactive and automated workflow configuration
  - Creates workflow plans and step configurations
  - Manages CIF conversion and calculation setup
  
- **executor.py** (1,902 lines) - Workflow execution engine
  - Executes workflow plans created by planner
  - Manages file dependencies and job submission
  - Handles error recovery and progress tracking

- **engine.py** (3,291 lines) - Legacy workflow orchestration
  - Original workflow automation system
  - Handles OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL progression
  - Tightly coupled with material database

- **context.py** (368 lines) - Workflow isolation framework
  - Provides isolated execution environments
  - Manages temporary directories and cleanup
  - Thread-safe execution support

- **dummy_file_creator.py** (762 lines) - Utility for creating dummy D12/OUT files
  - Creates placeholder files for CRYSTALOptToD12.py compatibility
  - Extracts settings from existing D12 files

#### Contextual Extensions (Duplicated Code)
- **planner_contextual.py** (222 lines) - Isolated planner wrapper
- **executor_contextual.py** (292 lines) - Isolated executor wrapper

#### Monitoring/Status Scripts (Overlapping Functionality)
- **status.py** (268 lines) - Primary workflow status checker
- **monitor_workflow.py** (84 lines) - Quick monitoring helper
- **check_workflows.py** (100 lines) - Manual workflow progression tool

#### Support/Utility Scripts
- **callback.py** (74 lines) - SLURM job completion callback
- **run_workflow_animated.py** (23 lines) - Animation wrapper
- **run_workflow_isolated.py** (125 lines) - Example isolation usage

#### Test Scripts
- **test_isolation.py** (305 lines) - Isolation feature testing

### 2. Major Issues Identified

#### A. Code Duplication
- **planner_contextual.py** duplicates 95% of planner.py code
- **executor_contextual.py** duplicates 95% of executor.py code
- Contextual versions only add isolation through inheritance
- **Missing import**: `planner_contextual.py` line 71 missing `import os`

#### B. Overlapping Functionality
- Three separate scripts for workflow monitoring (status, monitor_workflow, check_workflows)
- Each implements similar database queries and display logic
- No clear separation of concerns

#### C. Monolithic Scripts
- **engine.py** (3,291 lines) handles too many responsibilities:
  - Workflow orchestration
  - File management
  - Job submission
  - Database operations
  - Error recovery
- **planner.py** (4,619 lines) could be split into:
  - Interactive configuration
  - Workflow templates
  - CIF conversion management
  - Expert mode handlers

#### D. Integration Issues
- Isolation features partially integrated but not fully utilized
- Hard-coded configuration values throughout
- Inconsistent error handling patterns

### 3. Refactoring Plan

#### Phase 1: Consolidate Duplicated Code (Week 1)

**1.1 Merge Contextual Features**
```python
# In planner.py, add isolation support:
class WorkflowPlanner:
    def __init__(self, base_dir=".", db_path="materials.db", isolated=False):
        self.isolated = isolated
        if isolated:
            self.context = WorkflowIsolationContext(base_dir)
        # ... existing code
```

**1.2 Remove Duplicate Files**
- Delete `planner_contextual.py`
- Delete `executor_contextual.py`
- Update imports in dependent scripts

#### Phase 2: Consolidate Monitoring Tools (Week 2)

**2.1 Create Unified Monitor**
```python
# New file: workflow_monitor.py
class WorkflowMonitor:
    def status(self, workflow_id=None, format="summary"):
        """Show workflow status"""
    
    def monitor(self, interval=30, active_only=True):
        """Real-time monitoring"""
    
    def check_progression(self, manual=False):
        """Check and optionally progress workflows"""
```

**2.2 Deprecate Old Scripts**
- Move functionality from status.py, monitor_workflow.py, check_workflows.py
- Create compatibility wrappers during transition

#### Phase 3: Refactor Monolithic Scripts (Weeks 3-4)

**3.1 Split engine.py**
```
engine/
├── __init__.py
├── orchestrator.py      # Core workflow logic (800 lines)
├── file_manager.py      # File operations (600 lines)
├── job_submitter.py     # SLURM integration (500 lines)
├── error_recovery.py    # Error handling (400 lines)
├── database_ops.py      # DB operations (400 lines)
└── config.py           # Configuration (200 lines)
```

**3.2 Split planner.py**
```
planner/
├── __init__.py
├── core.py             # Core planning logic (1000 lines)
├── interactive.py      # Interactive configuration (800 lines)
├── templates.py        # Workflow templates (500 lines)
├── cif_handler.py      # CIF conversion (700 lines)
├── expert_modes.py     # Expert configurations (800 lines)
└── validators.py       # Input validation (300 lines)
```

#### Phase 4: Standardization (Week 5)

**4.1 Configuration Management**
```yaml
# workflow_config.yaml
workflow:
  optional_calculations:
    - CHARGE+POTENTIAL
    - TRANSPORT
  
  resource_defaults:
    OPT:
      cores: 32
      memory: "5G"
      walltime: "7-00:00:00"
    SP:
      cores: 32
      memory: "4G"
      walltime: "3-00:00:00"
```

**4.2 Error Handling Framework**
```python
# workflow/errors.py
class WorkflowError(Exception):
    """Base workflow exception"""

class ConfigurationError(WorkflowError):
    """Configuration-related errors"""

class ExecutionError(WorkflowError):
    """Execution-related errors"""
```

**4.3 Logging Integration**
```python
# workflow/logging.py
import logging

def setup_workflow_logger(name, level=logging.INFO):
    """Configure consistent logging"""
    logger = logging.getLogger(name)
    handler = logging.FileHandler(f"workflow_{name}.log")
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

### 4. Migration Strategy

#### Step 1: Create New Structure (Don't Delete Old)
- Implement refactored modules alongside existing code
- Ensure backward compatibility

#### Step 2: Gradual Migration
- Update imports one module at a time
- Run parallel testing
- Maintain fallback options

#### Step 3: Deprecation Notices
```python
import warnings

def old_function():
    warnings.warn(
        "This function is deprecated. Use new_module.new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Call new implementation
    return new_module.new_function()
```

#### Step 4: Final Cleanup
- Remove deprecated code after transition period
- Update documentation
- Update all dependent scripts

### 5. Testing Requirements

#### Unit Tests
```python
# tests/test_planner_core.py
def test_workflow_creation():
    """Test basic workflow creation"""

def test_cif_conversion():
    """Test CIF to D12 conversion"""

def test_expert_mode_configuration():
    """Test expert mode settings"""
```

#### Integration Tests
```python
# tests/test_workflow_integration.py
def test_full_workflow_execution():
    """Test complete workflow from CIF to properties"""

def test_error_recovery():
    """Test workflow recovery mechanisms"""
```

### 6. Documentation Updates

#### API Documentation
- Document all public interfaces
- Provide migration guides
- Include usage examples

#### Architecture Documentation
- Update module dependency diagrams
- Document data flow
- Explain design decisions

### 7. Benefits of Refactoring

1. **Improved Maintainability**
   - Smaller, focused modules
   - Clear separation of concerns
   - Easier to test and debug

2. **Better Performance**
   - Reduced code duplication
   - Optimized imports
   - Efficient resource usage

3. **Enhanced Flexibility**
   - Configuration-driven behavior
   - Pluggable components
   - Easier to extend

4. **Simplified Development**
   - Clear module boundaries
   - Consistent patterns
   - Better error messages

### 8. Risk Mitigation

1. **Backward Compatibility**
   - Maintain old interfaces during transition
   - Provide clear migration paths
   - Test extensively before removing old code

2. **User Impact**
   - Communicate changes clearly
   - Provide transition period
   - Maintain documentation for both versions

3. **Data Integrity**
   - Ensure database schema compatibility
   - Test data migration thoroughly
   - Provide rollback procedures

## Implementation Timeline

- **Week 1**: Consolidate contextual features
- **Week 2**: Unify monitoring tools
- **Week 3-4**: Refactor monolithic scripts
- **Week 5**: Standardization and testing
- **Week 6**: Documentation and deployment

## Success Metrics

1. **Code Metrics**
   - Reduce total lines of code by 30%
   - No file exceeds 1000 lines
   - Test coverage > 80%

2. **Performance Metrics**
   - Workflow execution time unchanged
   - Memory usage reduced by 20%
   - Import time reduced by 50%

3. **Developer Metrics**
   - Time to add new features reduced by 40%
   - Bug fix time reduced by 30%
   - Onboarding time for new developers reduced by 50%