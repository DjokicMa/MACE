# Workflow Isolation Implementation Summary

## Overview

This document summarizes the implementation of workflow isolation features for MACE (Mendoza Automated CRYSTAL Engine) to prevent conflicts when running multiple workflows in the same directory.

## Problem Solved

When running multiple MACE workflows in the same folder, the following conflicts occurred:
- **Database contamination**: Single `materials.db` shared by all workflows
- **Material ID collisions**: Multiple workflows creating same material IDs
- **Queue manager conflicts**: Job limits applied globally across all workflows
- **File storage overlap**: Calculation tracking mixed between workflows

## Implementation Components

### 1. WorkflowContext Class (`mace/workflow/context.py`)

Core isolation system providing:
- **Three isolation modes**:
  - `isolated`: Complete separation (recommended)
  - `shared`: Traditional behavior (backward compatible)
  - `hybrid`: Shared schema, isolated data

- **Features**:
  - Thread-safe context management
  - Environment variable propagation
  - Automatic cleanup and archival
  - Hidden directory structure (`.mace_context_*`)

### 2. Workflow Planner Updates (`mace/workflow/planner.py`)

Added interactive isolation configuration:
- Step 4.5: Isolation mode selection
- Step 4.6: Post-completion action selection
- Options saved in workflow plan JSON

### 3. Workflow Executor Updates (`mace/workflow/executor.py`)

Enhanced to use workflow contexts:
- Checks isolation mode from workflow plan
- Creates and activates WorkflowContext for non-shared modes
- Updates database and queue manager paths dynamically
- Handles post-completion actions (archive/export)

### 4. Context-Aware Database (`mace/database/materials_contextual.py`)

Updated existing contextual database to work with new WorkflowContext:
- Automatic context detection via `get_current_context()`
- Seamless path resolution for isolated databases
- Workflow-specific query methods
- Full backward compatibility

### 5. Context-Aware Queue Manager (`mace/queue/manager.py`)

Modified to support workflow contexts:
- Uses ContextualMaterialDatabase when context is active
- Context-specific lock directories
- Isolated status files per workflow

## Usage Examples

### Running Multiple Isolated Workflows

```bash
# Terminal 1
cd /shared/calculations
python mace_cli workflow --interactive
# Select: Isolated mode
# Configure: CIF files from project_a/

# Terminal 2 (same directory!)
cd /shared/calculations  
python mace_cli workflow --interactive
# Select: Isolated mode
# Configure: CIF files from project_b/

# Both workflows run independently without conflicts
```

### Programmatic Usage

```python
from mace.workflow.context import workflow_context
from mace.database.materials_contextual import ContextualMaterialDatabase

# Execute workflow with isolation
with workflow_context("my_workflow", isolation_mode="isolated") as ctx:
    db = ContextualMaterialDatabase()
    # All database operations are isolated
    materials = db.get_all_materials()
```

## Directory Structure

```
working_directory/
├── .mace_context_workflow_123/       # Hidden context directory
│   ├── materials.db                  # Isolated database
│   ├── structures.db                 # Isolated ASE database
│   ├── calculation_storage/          # Isolated file storage
│   ├── .queue_locks/                # Isolated locks
│   └── context_config.json          # Context metadata
├── workflow_configs/                # Shared configurations
└── workflow_outputs/                # Calculation outputs
    └── workflow_123/
        └── step_001_OPT/
            ├── mat_1_dia/           # Individual material folders
            └── mat_2_graphene/
```

## Key Benefits

1. **Complete Workflow Isolation**: No database conflicts between workflows
2. **Material ID Namespacing**: Each workflow has its own material ID space
3. **Independent Job Queues**: Per-workflow job limits and queue management
4. **Easy Cleanup**: Archive or delete entire workflow contexts
5. **Backward Compatibility**: Existing scripts work unchanged in shared mode

## Testing

A comprehensive test script (`test_workflow_isolation.py`) demonstrates:
- Running multiple isolated workflows simultaneously
- Shared mode compatibility
- Manual context switching
- Database isolation verification

## Migration Path

1. **Existing workflows**: Continue to work in shared mode by default
2. **New workflows**: Can opt into isolation during interactive planning
3. **Environment control**: Set `MACE_ISOLATION_MODE=isolated` for default isolation

## Future Enhancements

- Configuration file support for default isolation settings
- Workflow migration tools (shared → isolated)
- Context visualization and management utilities
- Cross-context data sharing mechanisms