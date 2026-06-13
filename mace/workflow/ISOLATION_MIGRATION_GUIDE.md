# MACE Workflow Isolation

## Status

Workflow isolation is **built into the base classes** — there is no separate
"contextual" API to migrate to. Use the standard classes; isolation is selected
via `isolation_mode` and the active workflow context.

> Superseded: the standalone `ContextualWorkflowPlanner`
> (`mace/workflow/planner_contextual.py`), `ContextualWorkflowExecutor`
> (`mace/workflow/executor_contextual.py`), and the `run_workflow_isolated.py`
> example were **removed** — they were an unused early prototype that nothing
> imported. Their functionality now lives in the base classes. See git history
> if you need the old prototype.

## How isolation works now

- **Execution / planning isolation** is implemented directly in
  `mace.workflow.executor.WorkflowExecutor` and
  `mace.workflow.planner.WorkflowPlanner` via `isolation_mode`
  (`shared` | `isolated` | `hybrid`), backed by `mace.workflow.context`
  (`get_current_context()` / the `WorkflowContext`). No subclass swap is needed.

- **Database isolation** uses
  `mace.database.materials_contextual.ContextualMaterialDatabase`, which is
  still available and is used automatically when a workflow context is active
  (e.g. `EnhancedCrystalQueueManager` and the recovery engine pick it up).

## Usage

No code changes are required for existing scripts — they run with shared
resources by default. To run isolated, set the workflow's `isolation_mode` (or
run within an isolated `WorkflowContext`); the base planner/executor and the
contextual database handle the rest.
