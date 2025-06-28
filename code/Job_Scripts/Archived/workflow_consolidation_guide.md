# Workflow Consolidation Implementation Guide

## Priority 1: Consolidate Workflow Execution

### Current State Analysis

We have two parallel workflow execution systems:

1. **workflow_engine.py** (713 lines)
   - Original implementation
   - Focuses on material tracking integration
   - Has execute_workflow_step() method
   - Handles file naming complexity
   - Contains the recent fixes for FREQ generation, template validation, memory handling

2. **workflow_executor.py** (>1000 lines)
   - Newer implementation for workflow planner
   - More comprehensive execution phases
   - Better progress tracking
   - Integrates with JSON workflow plans

### Consolidation Strategy

#### Step 1: Create Unified Interface

```python
# unified_workflow_manager.py

class UnifiedWorkflowManager:
    """
    Unified workflow management combining best practices from both implementations.
    """
    
    def __init__(self, work_dir: str = ".", db_path: str = "materials.db"):
        self.work_dir = Path(work_dir).resolve()
        self.db = MaterialDatabase(db_path)
        self.queue_manager = EnhancedCrystalQueueManager(
            d12_dir=str(self.work_dir),
            max_jobs=200,
            enable_tracking=True,
            enable_error_recovery=True,
            db_path=db_path
        )
        
        # Best of both: workflow tracking from executor, material focus from engine
        self.active_workflows = {}
        self.workflow_states = {}  # From the new persistence work
        
    def execute_workflow(self, plan_file: Path = None, workflow_dict: Dict = None):
        """
        Execute workflow from either JSON file or dict.
        Combines both execution patterns.
        """
        if plan_file:
            with open(plan_file, 'r') as f:
                plan = json.load(f)
        else:
            plan = workflow_dict
            
        workflow_id = plan.get('workflow_id', self._generate_workflow_id())
        
        # Use workflow state persistence
        if self.db.has_workflow_states_table():
            self.db.create_workflow_state(
                workflow_id=workflow_id,
                material_id="multiple",  # Will be updated per material
                planned_sequence=plan['workflow_sequence']
            )
        
        # Execute with unified logic
        self._execute_unified(plan, workflow_id)
```

#### Step 2: Merge Execution Logic

Key methods to consolidate:

1. **SLURM Submission** - Currently duplicated
```python
def submit_to_slurm(self, script_path: Path, work_dir: Path, 
                    material_id: str = None) -> Optional[str]:
    """
    Unified SLURM submission handling both script types.
    Combines logic from both implementations.
    """
    # From workflow_executor: handle template scripts
    # From workflow_engine: handle direct scripts
    # Add: better error reporting and logging
```

2. **Material ID Extraction** - Multiple implementations
```python
def get_material_id(self, file_path: Path) -> str:
    """
    Single source of truth for material ID extraction.
    Uses material_database.create_material_id_from_file()
    """
    return create_material_id_from_file(file_path.name)
```

3. **Cleanup Methods** - Consolidate multiple versions
```python
def cleanup_failed_workflows(self, age_days: int = 7):
    """
    Unified cleanup combining both implementations.
    """
    # From workflow_engine: cleanup old staging dirs
    # From workflow_executor: cleanup temp dirs
    # Add: cleanup orphaned database records
```

### Step 3: Preserve Critical Fixes

Ensure we keep the recent fixes from workflow_engine.py:

1. **FREQ Generation Fix**
```python
# Only generate next steps if they exist in plan
if next_steps:
    for next_calc_type in next_steps:
        # Generate next calculation
```

2. **Template Validation**
```python
def _validate_property_templates(self, calc_type: str) -> bool:
    """Keep the template validation logic"""
    # Check for required templates before generation
```

3. **Memory Format Handling**
```python
def _fix_memory_reporting(self, script_content: str) -> str:
    """Preserve the memory format detection"""
    # Handle both --mem and --mem-per-cpu
```

### Step 4: Migration Path

1. **Phase 1: Create Adapter**
```python
# workflow_engine.py (modified)
from unified_workflow_manager import UnifiedWorkflowManager

class WorkflowEngine(UnifiedWorkflowManager):
    """Compatibility wrapper - deprecate over time"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("WorkflowEngine is deprecated, use UnifiedWorkflowManager", 
                     DeprecationWarning)
```

2. **Phase 2: Update Imports**
- Change imports in enhanced_queue_manager.py
- Update workflow_planner.py to use unified manager
- Modify run_workflow.py to import new module

3. **Phase 3: Remove Old Code**
- After testing period, remove workflow_engine.py
- Merge any remaining unique functionality

### Step 5: Testing Strategy

Create comprehensive tests for the unified implementation:

```python
# test_unified_workflow.py

def test_workflow_execution():
    """Test basic workflow execution"""
    manager = UnifiedWorkflowManager()
    # Test with sample workflow plan
    
def test_slurm_submission_types():
    """Test both template and direct script submission"""
    # Test template scripts (from executor)
    # Test direct scripts (from engine)
    
def test_material_id_consistency():
    """Ensure material ID extraction is consistent"""
    # Test with various filename patterns
    
def test_error_recovery_integration():
    """Test that error recovery still works"""
    # Simulate failures and recovery
```

## Benefits of Consolidation

1. **Code Reduction**: ~40% less code to maintain
2. **Single Source of Truth**: One place for workflow logic
3. **Better Testing**: Easier to test single implementation
4. **Improved Performance**: Eliminate redundant operations
5. **Clearer Architecture**: Easier for new developers

## Risk Mitigation

1. **Compatibility Layer**: Keep old interfaces working
2. **Gradual Migration**: Phase approach over 2-3 weeks
3. **Comprehensive Testing**: Test all workflow patterns
4. **Rollback Plan**: Keep old code in Archived/ folder

## Implementation Timeline

- **Week 1**: Create unified manager, add compatibility layer
- **Week 2**: Migrate all callers, test extensively
- **Week 3**: Monitor for issues, remove deprecated code

This consolidation will significantly improve maintainability while preserving all current functionality and recent fixes.