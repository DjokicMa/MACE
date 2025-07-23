# Workflow Isolation Design Document

## Problem Summary

When running multiple MACE workflows in the same directory, severe conflicts arise:

### 1. **Database Contamination**
- Single `materials.db` file shared by all workflows
- Material IDs collide between workflows (e.g., both have "diamond")
- Workflow progression triggers actions on wrong materials
- Error recovery attempts to "fix" calculations from other workflows

### 2. **Queue Manager Conflicts**
- All workflows share the same queue manager instance
- Job submission limits apply globally, not per-workflow
- Callbacks don't distinguish between workflows
- One workflow can exhaust job slots for all others

### 3. **File Storage Overlap**
- Calculation files stored without workflow context
- Property extraction overwrites data between workflows
- Binary files (fort.9, fort.25) can get mixed up
- No namespace separation in storage directories

### 4. **Material ID Collisions**
```python
# Both workflows create same ID from filename
"diamond_opt.d12" → material_id: "diamond"
# Workflow A and B share this material!
```

## Solution Architecture

### Core Component: WorkflowContext

The `WorkflowContext` class (already implemented in `mace/workflow/context.py`) provides:

1. **Isolated Resources**
   - Separate databases per workflow
   - Workflow-specific storage directories
   - Isolated lock files and queue states
   - Context-aware configuration

2. **Three Isolation Modes**
   - **isolated**: Complete separation (recommended)
   - **shared**: Traditional behavior (backward compatible)
   - **hybrid**: Shared schema, isolated data

3. **Automatic Context Management**
   - Thread-safe context switching
   - Environment variable propagation
   - Child process inheritance
   - Cleanup and archival support

## Implementation Plan

### Phase 1: Add Context Support to Workflow Planner

```python
# In workflow_planner.py main_interactive_workflow():

# Add isolation mode selection
print("\nWorkflow Isolation Mode:")
print("  1. Isolated (recommended) - Separate database per workflow")
print("  2. Shared - Use shared database (legacy behavior)")
print("  3. Hybrid - Shared schema, isolated data")

isolation_mode = input("Select mode [1]: ").strip() or "1"
isolation_map = {"1": "isolated", "2": "shared", "3": "hybrid"}

# Save in workflow plan
workflow_plan["isolation_mode"] = isolation_map.get(isolation_mode, "isolated")
```

### Phase 2: Context-Aware Database Access

Create wrapper classes that automatically use the correct database based on context:

```python
# mace/database/materials_contextual.py
from mace.workflow.context import get_current_context, require_context
from mace.database.materials import MaterialDatabase

class ContextualMaterialDatabase(MaterialDatabase):
    """Context-aware material database that uses workflow-specific databases"""
    
    def __init__(self, db_path: Optional[str] = None):
        # Get context and use its database path
        ctx = require_context()
        if db_path is None:
            db_path = str(ctx.get_database_path())
        super().__init__(db_path)
    
    @classmethod
    def from_context(cls) -> 'ContextualMaterialDatabase':
        """Create database instance from current context"""
        return cls()
```

### Phase 3: Update Workflow Executor

```python
# In workflow_executor.py execute_workflow_plan():

# Load workflow plan
with open(plan_file, 'r') as f:
    plan = json.load(f)

# Extract workflow ID and isolation mode
workflow_id = plan.get("workflow_id", f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
isolation_mode = plan.get("isolation_mode", "isolated")

# Create and activate context
with workflow_context(workflow_id, isolation_mode=isolation_mode) as ctx:
    # All workflow execution happens within context
    # Database access will automatically use isolated resources
    self._execute_workflow_steps(plan)
    
    # Export results before context cleanup
    ctx.export_results(self.work_dir / "workflow_results")
```

### Phase 4: Update Queue Manager

```python
# In enhanced_queue_manager.py:

def __init__(self, ...):
    # Check for active context
    ctx = get_current_context()
    if ctx:
        # Use context-specific paths
        self.db_path = str(ctx.get_database_path())
        self.lock_dir = ctx.get_lock_dir()
        self.status_file = ctx.get_storage_path() / "queue_status.json"
    else:
        # Fallback to traditional paths
        self.db_path = db_path or "materials.db"
        # ...
```

## Migration Strategy

### 1. **Backward Compatibility**
- Default to shared mode if no context specified
- Existing scripts continue to work unchanged
- Gradual opt-in to isolation features

### 2. **Simple Adoption**
```bash
# Old way (still works)
python run_workflow.py --interactive

# New way with isolation
export MACE_ISOLATION_MODE=isolated
python run_workflow.py --interactive
```

### 3. **Configuration File Support**
```json
// .mace_config.json
{
  "default_isolation_mode": "isolated",
  "archive_completed_workflows": true,
  "cleanup_delay_days": 7
}
```

## Benefits

### 1. **Complete Workflow Isolation**
- No database conflicts between workflows
- Independent job queues per workflow
- Clean material ID namespacing

### 2. **Improved Debugging**
- Each workflow has its own database for inspection
- Clear separation of logs and outputs
- Easy to identify which files belong to which workflow

### 3. **Better Resource Management**
- Per-workflow job limits
- Independent error recovery
- Parallel workflow execution without conflicts

### 4. **Easy Cleanup**
- Archive or delete entire workflow contexts
- No orphaned files in shared locations
- Clear workflow boundaries

## Directory Structure with Isolation

```
working_directory/
├── .mace_context_workflow_20250723_120000/
│   ├── materials.db              # Isolated database
│   ├── structures.db             # Isolated ASE database
│   ├── calculation_storage/      # Isolated file storage
│   ├── .queue_locks/            # Isolated locks
│   └── context_config.json      # Context metadata
├── .mace_context_workflow_20250723_140000/
│   └── ... (another isolated workflow)
├── workflow_configs/            # Shared config directory
├── workflow_results/           # Exported results
└── archived_workflows/         # Completed workflow archives
```

## Example Usage

### Running Multiple Workflows Safely

```bash
# Terminal 1: Workflow for project A
cd /shared/calculations
python mace_cli workflow --interactive
# Select: Isolated mode
# Configure: CIF files from project_a/

# Terminal 2: Workflow for project B  
cd /shared/calculations  # Same directory!
python mace_cli workflow --interactive
# Select: Isolated mode
# Configure: CIF files from project_b/

# Both workflows run independently without conflicts
```

### Accessing Isolated Data

```python
# Script to analyze specific workflow results
from mace.workflow.context import workflow_context
from mace.database.materials import MaterialDatabase

# Access specific workflow's data
with workflow_context("workflow_20250723_120000", isolation_mode="isolated") as ctx:
    db = MaterialDatabase(str(ctx.get_database_path()))
    
    # Query materials from this workflow only
    materials = db.get_all_materials()
    for mat in materials:
        print(f"{mat.material_id}: {mat.formula}")
```

## Implementation Priority

1. **High Priority**: WorkflowContext class (✅ Complete)
2. **High Priority**: Update workflow planner to set isolation mode
3. **Medium Priority**: Context-aware database wrappers
4. **Medium Priority**: Update workflow executor for context activation
5. **Low Priority**: Update all components for context awareness
6. **Low Priority**: Add cleanup and archival utilities

## Testing Plan

1. **Unit Tests**: Context creation, activation, isolation
2. **Integration Tests**: Multiple concurrent workflows
3. **Stress Tests**: Resource cleanup, database locking
4. **Migration Tests**: Backward compatibility verification

## Next Steps

1. Add isolation mode selection to workflow planner
2. Create context-aware database wrapper
3. Update workflow executor to use contexts
4. Test with multiple concurrent workflows
5. Document usage patterns and best practices