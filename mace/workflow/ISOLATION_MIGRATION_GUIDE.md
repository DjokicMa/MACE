# MACE Workflow Isolation Migration Guide

## Overview

The workflow isolation feature allows MACE to run workflows in isolated environments with separate databases and resources. This prevents conflicts between concurrent workflows and enables better resource management.

## Migration Strategy

### Phase 1: Opt-in Migration (Recommended)

1. **No changes required for existing code** - All existing scripts continue to work with shared resources
2. **New workflows can opt-in** to isolation using configuration or environment variables
3. **Gradual adoption** - Test isolation with non-critical workflows first

### Phase 2: Gradual Adoption

1. Update import statements in your scripts:
```python
# Old
from mace.database.materials import MaterialDatabase
from mace.workflow.planner import WorkflowPlanner
from mace.workflow.executor import WorkflowExecutor

# New (with isolation support)
from mace.database.materials_contextual import ContextualMaterialDatabase as MaterialDatabase
from mace.workflow.planner_contextual import ContextualWorkflowPlanner as WorkflowPlanner
from mace.workflow.executor_contextual import ContextualWorkflowExecutor as WorkflowExecutor
```

2. Or use the compatibility layer:
```python
# Enable contextual databases globally
from mace.database.materials_contextual import enable_contextual_databases
enable_contextual_databases()
```

### Phase 3: Full Migration

Update `run_workflow.py` and other entry points to use contextual versions by default.

## Configuration Options

### 1. Environment Variables

```bash
# Set isolation mode
export MACE_ISOLATION_MODE=isolated  # or 'shared'

# Auto cleanup after workflow
export MACE_AUTO_CLEANUP=true

# Specific workflow ID (for resuming)
export MACE_WORKFLOW_ID=my_workflow_001
```

### 2. Configuration File

Create `.mace_config.json` in your working directory:

```json
{
  "isolation_mode": "isolated",
  "auto_cleanup": false,
  "context_settings": {
    "cleanup_on_exit": true,
    "archive_results": true
  }
}
```

### 3. Command Line (Future)

```bash
# Run with isolation
mace workflow --isolation-mode isolated --execute plan.json

# Run with shared resources (default)
mace workflow --isolation-mode shared --execute plan.json
```

## Usage Examples

### Example 1: Isolated Workflow Execution

```python
from mace.workflow.executor_contextual import IsolatedWorkflowExecution

# Execute workflow in isolated environment
with IsolatedWorkflowExecution("my_workflow_001") as executor:
    result = executor.execute_workflow_plan("workflow_plan.json")
# Resources automatically cleaned up on exit
```

### Example 2: Manual Context Management

```python
from mace.workflow.context import WorkflowContext
from mace.database.materials_contextual import ContextualMaterialDatabase

# Create isolated context
with WorkflowContext("my_workflow", isolation_mode="isolated") as ctx:
    # Get isolated database
    db = ctx.get_database()
    
    # Create materials and calculations
    material_id = db.create_material(
        material_id="test_material",
        formula="Al2O3",
        space_group=167
    )
    
    # Export results when done
    export_file = ctx.export_results()
```

### Example 3: Interactive Workflow Planning with Isolation

```python
from mace.workflow.planner_contextual import ContextualWorkflowPlanner

# Create planner with isolation
planner = ContextualWorkflowPlanner(isolation_mode="isolated")

# Plan workflow (creates isolated context automatically)
plan = planner.plan_workflow()

# Save plan (saves to isolated config directory)
planner.save_workflow_plan(plan)
```

## Backward Compatibility

### What Works Without Changes

1. **All existing scripts** using `MaterialDatabase` directly
2. **Queue managers** continue to use shared database by default
3. **Direct script execution** (not through workflow system)
4. **Analysis scripts** that read from shared database

### What Needs Updates

1. **Workflow planners** - Use `ContextualWorkflowPlanner` for isolation support
2. **Workflow executors** - Use `ContextualWorkflowExecutor` for isolation support
3. **Custom scripts** that need isolation - Import contextual versions

## Best Practices

### 1. When to Use Isolation

- **Parameter sweeps** - Each parameter set gets its own database
- **Concurrent workflows** - Prevent database conflicts
- **Testing** - Isolated test runs without affecting production
- **Reproducibility** - Each workflow has complete provenance

### 2. When to Use Shared Mode

- **Single user** systems with sequential workflows
- **Small calculations** where overhead isn't justified
- **Legacy compatibility** requirements
- **Shared material libraries** that multiple workflows access

### 3. Resource Management

```python
# Always export results before cleanup in isolated mode
with WorkflowContext("my_workflow", isolation_mode="isolated") as ctx:
    # ... do work ...
    
    # Export results before context closes
    export_file = ctx.export_results("shared_materials.db")
```

### 4. Debugging Isolated Workflows

```python
# List active contexts
from mace.workflow.context import WorkflowContext
active = WorkflowContext.list_active_contexts()
print(f"Active contexts: {active}")

# Access specific context
ctx = WorkflowContext.get_active_context("workflow_001")
if ctx:
    print(f"Context status: {ctx.get_status()}")
```

## Migration Checklist

- [ ] Test isolation with a simple workflow
- [ ] Update workflow planning scripts to use contextual planner
- [ ] Update workflow execution scripts to use contextual executor
- [ ] Configure default isolation mode (environment or config file)
- [ ] Update documentation for your workflows
- [ ] Train users on isolation options
- [ ] Set up cleanup policies (manual vs automatic)
- [ ] Implement result archiving if needed

## Troubleshooting

### Issue: "Database locked" errors
**Solution**: This usually means multiple processes are accessing the same database. Enable isolation mode to give each workflow its own database.

### Issue: Can't find results after workflow
**Solution**: Check if workflow used isolation. Results may be in the workflow context directory or exported to a JSON file.

### Issue: Disk space usage
**Solution**: Enable auto-cleanup or implement a cleanup policy for old workflow contexts.

### Issue: Need to resume failed workflow
**Solution**: Keep cleanup_on_exit=False and use the workflow ID to resume:
```python
# Resume with existing context
ctx = WorkflowContext("failed_workflow_001", cleanup_on_exit=False)
```

## Future Enhancements

1. **Automatic result merging** from isolated to shared database
2. **Workflow context templates** for common scenarios  
3. **Web UI** for managing isolated workflows
4. **Cloud storage** integration for context archiving
5. **Distributed execution** with context synchronization