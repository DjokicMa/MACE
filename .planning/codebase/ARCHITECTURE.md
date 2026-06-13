<!-- refreshed: 2026-06-13 -->
# Architecture

**Analysis Date:** 2026-06-13

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Points                            │
│  `mace_cli` (compiled)   `mace/run_mace.py`   `mace/run_workflow.py`│
└──────────────┬──────────────────┬──────────────────┬───────────────┘
               │                  │                  │
               ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Workflow Layer                                 │
│  planner.py (WorkflowPlanner)  │  executor.py (WorkflowExecutor)   │
│  engine.py  (WorkflowEngine)   │  context.py  (WorkflowContext)    │
│  callback.py                   │  status.py / monitor_workflow.py  │
│                `mace/workflow/`                                      │
└──────────────┬──────────────────┬──────────────────────────────────┘
               │                  │
       ┌───────▼──────┐   ┌───────▼──────────┐
       │  Queue Layer │   │  Recovery Layer  │
       │  manager.py  │   │  detector.py     │
       │  monitor.py  │   │  recovery.py     │
       │  lock_mgr.py │   │  pandas_utils.py │
       │`mace/queue/` │   │`mace/recovery/`  │
       └───────┬──────┘   └───────┬──────────┘
               │                  │
               ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Generation Layer                            │
│  Crystal_d12/NewCifToD12.py     Crystal_d3/CRYSTALOptToD3.py       │
│  Crystal_d12/CRYSTALOptToD12.py Crystal_d3/d3_kpoints.py           │
│  Crystal_d12/d12_from_config.py Crystal_d3/seekpath_interface.py   │
│  Crystal_d12/d12_config.py      Crystal_d3/d3_config.py            │
│  Crystal_d12/d12_parsers.py     Crystal_d3/d3_interactive.py       │
│  Crystal_d12/d12_writer.py                                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SLURM / HPC                                    │
│  mace/submission/crystal.py   mace/submission/properties.py         │
│  mace/submission/portable_slurm_generator.py                        │
│  mace/submission/templates/                                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ (jobs complete → callback)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Database / Persistence Layer                        │
│  MaterialDatabase (SQLite + ASE)  — `mace/database/materials.py`   │
│  ContextualMaterialDatabase       — `mace/database/materials_contextual.py` │
│  analysis/, query/, export/, interactive/                            │
│  Property Extractor  — `mace/utils/property_extractor.py`           │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| WorkflowPlanner | Pre-plans full calculation sequence from CIFs/D12s; produces JSON plan | `mace/workflow/planner.py` |
| WorkflowExecutor | Executes JSON plan step-by-step; drives D12/D3 generation via subprocess | `mace/workflow/executor.py` |
| WorkflowEngine | Orchestrates individual calculation steps; manages workflow staging dirs | `mace/workflow/engine.py` |
| WorkflowContext | Provides isolated per-workflow database, lock, and storage directories | `mace/workflow/context.py` |
| EnhancedCrystalQueueManager | Monitors SLURM queue; detects job completion; fires callbacks; submits next step | `mace/queue/manager.py` |
| QueueLockManager | File-based + thread-based distributed locking to prevent callback races | `mace/queue/queue_lock_manager.py` |
| CrystalErrorDetector | Parses `.out`/`.o` files to classify CRYSTAL errors (shrink, SCF, timeout, etc.) | `mace/recovery/detector.py` |
| ErrorRecoveryEngine | Applies YAML-configured recovery strategies; resubmits fixed jobs | `mace/recovery/recovery.py` |
| MaterialDatabase | Thread-safe SQLite + ASE database for materials, calculations, properties | `mace/database/materials.py` |
| ContextualMaterialDatabase | Workflow-scoped wrapper around MaterialDatabase | `mace/database/materials_contextual.py` |
| CrystalPropertyExtractor | Regex-based extraction of electronic/structural properties from `.out` files | `mace/utils/property_extractor.py` |
| NewCifToD12 | Converts CIF structures to CRYSTAL D12 input files (uses ASE + spglib) | `Crystal_d12/NewCifToD12.py` |
| CRYSTALOptToD12 | Extracts optimized geometry from `.out` and generates next-step D12 | `Crystal_d12/CRYSTALOptToD12.py` |
| d12_from_config | Unified dispatcher for JSON-config-driven D12 generation | `Crystal_d12/d12_from_config.py` |
| CRYSTALOptToD3 | Generates D3 property calculation inputs from completed SP/OPT outputs | `Crystal_d3/CRYSTALOptToD3.py` |
| seekpath_interface | Wraps seekpath library to generate high-symmetry k-point paths for BAND | `Crystal_d3/seekpath_interface.py` |
| submission/crystal.py | Generates and submits SLURM scripts for D12 jobs | `mace/submission/crystal.py` |
| submission/properties.py | Generates and submits SLURM scripts for D3/property jobs | `mace/submission/properties.py` |

## Pattern Overview

**Overall:** Pipeline Automation with Event-Driven Callbacks

**Key Characteristics:**
- Each calculation step runs as an isolated SLURM job; MACE acts as a workflow coordinator
- JSON workflow plans decouple planning from execution
- SLURM completion fires `callback.py`, which advances the workflow to the next step
- SQLite database tracks every material, calculation state, and extracted property
- File-based locking (`mace/queue/queue_lock_manager.py`) guards against simultaneous callback races
- Optional calculations (BAND, DOSS, FREQ, TRANSPORT, CHARGE+POTENTIAL) do not block the main chain if they fail
- Workflow isolation via `WorkflowContext` allows multiple concurrent workflows in the same directory

## Layers

**CLI / Entry Layer:**
- Purpose: User-facing command dispatch
- Location: `mace_cli` (compiled binary), `mace/run_mace.py`, `mace/run_workflow.py`
- Contains: Argument parsing, mode selection (interactive, execute, status, quick-start)
- Depends on: Workflow layer
- Used by: End user / shell

**Workflow Layer:**
- Purpose: Plans, sequences, and tracks multi-step calculation pipelines
- Location: `mace/workflow/`
- Contains: Planner, executor, engine, context, callback, status, monitor, dummy file creator
- Depends on: Queue layer, Input Generation layer, Database layer
- Used by: CLI layer

**Queue / SLURM Layer:**
- Purpose: Job submission, monitoring, completion detection, and locking
- Location: `mace/queue/`
- Contains: `manager.py`, `monitor.py`, `queue_lock_manager.py`, `legacy_manager.py`
- Depends on: Database layer, Recovery layer
- Used by: Workflow layer, SLURM job epilog/callback

**Recovery Layer:**
- Purpose: Error classification and automated resubmission with applied fixes
- Location: `mace/recovery/`
- Contains: `detector.py`, `recovery.py`, `pandas_utils.py`
- Config: `mace/config/recovery_config.yaml`
- Depends on: Database layer, `code/Check_Scripts/fixk.py` (legacy)
- Used by: Queue layer

**Input Generation Layer:**
- Purpose: Produces CRYSTAL D12 and D3 input files from structures or prior outputs
- Location: `Crystal_d12/`, `Crystal_d3/`
- Contains: Converters, config managers, parsers, writers, k-point generators
- Depends on: ASE, spglib, seekpath (Anaconda Python only); basis set data files
- Used by: Workflow layer (via subprocess calls, not direct import)

**Submission Layer:**
- Purpose: Writes and submits SLURM job scripts
- Location: `mace/submission/`
- Contains: `crystal.py`, `properties.py`, `portable_slurm_generator.py`
- Depends on: `mace/utils/node_exclusion.py` for AMD20 exclusion
- Used by: Workflow layer, Queue layer

**Database / Persistence Layer:**
- Purpose: Durable tracking of materials, calculation states, and extracted properties
- Location: `mace/database/`
- Contains: `materials.py`, `materials_contextual.py`, `populate_completed_jobs.py`
- Sub-packages: `analysis/`, `query/`, `export/`, `interactive/`, `utils/`
- Storage: `materials.db` (SQLite), `structures.db` (ASE)
- Used by: All other layers

**Utilities Layer:**
- Purpose: Cross-cutting helpers: property extraction, file management, formatting
- Location: `mace/utils/`
- Contains: `property_extractor.py`, `file_manager.py`, `formula_extractor.py`, `settings_extractor.py`, `scf_settings_extractor.py`, `dat_file_processor.py`, `node_exclusion.py`, `animation.py`, `banner.py`
- Used by: Workflow, Queue, Recovery, and Database layers

**Plotting / Post-Processing Layer:**
- Purpose: Standalone scripts for visualizing completed calculation outputs
- Location: `code/NewPlotting_Scripts/`, `code/Plotting_Scripts/`, `Plotting/`, `code/Band_Alignment/`
- Contains: Band structure, DOS, phonon band, charge density plotters
- Used by: End user independently of MACE workflow

## Data Flow

### Primary Workflow Path

1. User invokes `mace_cli` or `mace/run_mace.py --interactive` / `--execute plan.json`
2. `WorkflowPlanner` reads CIFs or existing D12 files from `--cif-dir`; generates JSON workflow plan with ordered calculation steps (e.g., OPT → SP → BAND → DOSS)
3. `WorkflowExecutor` loads JSON plan; for each step:
   a. Calls `Crystal_d12/NewCifToD12.py` or `Crystal_d12/CRYSTALOptToD12.py` via subprocess to produce `.d12`
   b. Calls `mace/submission/crystal.py` to generate and submit SLURM script
   c. Records calculation in `MaterialDatabase` with status `SUBMITTED`
4. SLURM job runs CRYSTAL23 on the HPC cluster
5. On job completion, SLURM epilog or poll calls `mace/workflow/callback.py`
6. `EnhancedCrystalQueueManager` (in `mace/queue/manager.py`) detects completion, updates DB to `COMPLETED`
7. `WorkflowEngine` advances to next step: generates D3 input via `Crystal_d3/CRYSTALOptToD3.py`, submits property job
8. `CrystalPropertyExtractor` (`mace/utils/property_extractor.py`) parses `.out` file and populates properties table in `materials.db`

### Error Recovery Path

1. `CrystalErrorDetector` (`mace/recovery/detector.py`) classifies error from `.out` / SLURM `.o` file
2. `canonical_error_type()` (`mace/recovery/recovery.py`) maps detector names to YAML config keys
3. `ErrorRecoveryEngine` looks up recovery strategy in `mace/config/recovery_config.yaml`
4. Engine applies fix (e.g., increases SHRINK via `fixk.py`, increases walltime, adjusts FMIXING) and resubmits
5. Recovery attempts tracked in DB; escalates to manual review after `max_retries`

### D12/D3 Config-Driven Generation

1. User selects or creates JSON config from `Crystal_d12/example_configs/` or `Crystal_d3/example_configs/`
2. `Crystal_d12/d12_from_config.py` detects input file type (CIF vs. CRYSTAL `.out`)
3. Dispatches to `NewCifToD12.py` (CIF) or `CRYSTALOptToD12.py` (prior output)
4. Config applied to override defaults; output `.d12` written to working directory
5. Analogously, `Crystal_d3/d3_config.py` manages D3 configs for BAND, DOSS, ECH3, POT3, TRANSPORT

**State Management:**
- All calculation state persists in `materials.db` (SQLite, thread-safe with `threading.RLock`)
- Atomic structures persist in `structures.db` (ASE database)
- Per-workflow isolation uses `WorkflowContext` with separate `.mace_context_<id>/` directories
- Queue lock files live in `test/ECH3POT3/.queue_locks/` or workflow context lock dir

## Key Abstractions

**Material ID:**
- Purpose: Stable identifier linking a physical material across all calculation steps and file naming variations
- Created by: `create_material_id_from_file()` in `mace/database/materials.py`
- Used throughout: DB records, workflow plan JSON, file naming

**Workflow Plan (JSON):**
- Purpose: Serialized, replayable specification of all steps, materials, configs, and resources
- Produced by: `WorkflowPlanner` (`mace/workflow/planner.py`)
- Consumed by: `WorkflowExecutor` (`mace/workflow/executor.py`)
- Format: JSON file named `workflow_plan_<YYYYMMDD_HHMMSS>.json`

**WorkflowContext:**
- Purpose: Isolates databases, locks, and storage for a single workflow run
- Location: `mace/workflow/context.py`
- Modes: `isolated` (separate DBs), `shared`, `hybrid`
- Context dir: `.mace_context_<workflow_id>/` in working directory

**Optional Calculation Types:**
- Defined as: `OPTIONAL_CALC_TYPES = {'BAND', 'DOSS', 'FREQ', 'TRANSPORT', 'CHARGE+POTENTIAL'}` in `mace/workflow/engine.py`
- Behavior: Failure does not block workflow progression; OPT and SP are blocking

**Recovery Config (YAML):**
- Location: `mace/config/recovery_config.yaml`
- Structure: Per-error-type handler, retry limits, parameter adjustment factors
- Error types: `shrink_error`, `memory_error`, `convergence_error`, `timeout_error`, `disk_space_error`

## Entry Points

**Interactive Planning:**
- Location: `mace/run_mace.py` (also `mace_cli` compiled binary)
- Triggers: `--interactive` flag
- Responsibilities: Prompts user for CIF dir, workflow type, resources; writes JSON plan

**Execute Saved Plan:**
- Location: `mace/run_mace.py`
- Triggers: `--execute <plan.json>`
- Responsibilities: Loads plan, instantiates `WorkflowExecutor`, drives step-by-step execution

**Job Completion Callback:**
- Location: `mace/workflow/callback.py`
- Triggers: Called by SLURM epilog or queue monitor on job finish
- Responsibilities: Updates DB state, fires `WorkflowEngine` to submit next step

**Queue Monitor (daemon mode):**
- Location: `mace/queue/monitor.py`, `mace/material_monitor.py`
- Triggers: Run as background daemon or cron
- Responsibilities: Polls SLURM queue, detects completions/failures, fires callbacks

**Status Check:**
- Location: `mace/workflow/status.py`, `mace/run_mace.py --status`
- Triggers: User request
- Responsibilities: Queries DB, prints workflow/calculation status summary

## Architectural Constraints

- **Python runtime:** Must use `/home/marcus/anaconda3/bin/python` — ASE, spglib, and seekpath are only installed there; system Python lacks these packages
- **Subprocess boundary:** `Crystal_d12/` and `Crystal_d3/` scripts are always invoked via subprocess from the workflow layer, not imported directly (except `d12_constants.py` which is imported). This preserves their standalone usability.
- **Threading:** Each of `MaterialDatabase`, `WorkflowEngine`, `EnhancedCrystalQueueManager`, and `CrystalErrorDetector` carries its own `threading.RLock`. No shared global lock hierarchy — concurrent callback invocations are guarded per-instance.
- **Global state:** `WorkflowContext._active_contexts` is a class-level dict (global to process); `WorkflowContext._thread_local` is thread-local. These are the only module-level mutable singletons.
- **SLURM dependency:** All job submission and monitoring assumes a SLURM HPC environment with `sbatch`, `squeue`, `sacct`. No local execution mode for actual CRYSTAL runs.
- **Circular imports:** `mace/workflow/planner.py` and `mace/workflow/engine.py` both import from `mace/database/materials.py` and `mace/queue/manager.py`; neither imports the other, avoiding direct circularity.
- **File naming convention:** CRYSTAL output filenames encode the full calculation chain history (e.g., `1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out`). The `create_material_id_from_file()` function must parse these to extract the stable material identity.

## Anti-Patterns

### Direct import of Crystal_d12 scripts (except constants)

**What happens:** Workflow code occasionally attempts `from d12_constants import ...` by inserting the `Crystal_d12` directory into `sys.path`.
**Why it's wrong:** The converter scripts (`NewCifToD12.py`, `CRYSTALOptToD12.py`) import `ase`, `spglib` at module level — importing them under the wrong Python interpreter silently degrades behavior or raises ImportError.
**Do this instead:** Invoke converters as subprocesses using `/home/marcus/anaconda3/bin/python Crystal_d12/NewCifToD12.py ...`. Only import `d12_constants.py` directly (it has no external deps).

### Running workflow with system Python

**What happens:** Calling `python mace/run_mace.py` with system Python instead of Anaconda Python.
**Why it's wrong:** ASE, spglib, and seekpath are absent; the D12/D3 generation subprocess calls will fail.
**Do this instead:** Always activate the mace environment (`source activate_mace.sh`) or invoke via `/home/marcus/anaconda3/bin/python`.

### Relying on shared `materials.db` for concurrent workflows

**What happens:** Two workflows run simultaneously in the same directory without using `WorkflowContext`.
**Why it's wrong:** Callback events from workflow A update material state for workflow B; DB corruption is possible under concurrent writes.
**Do this instead:** Use `WorkflowContext` with `isolation_mode="isolated"` so each workflow gets its own `.mace_context_<id>/materials.db`.

## Error Handling

**Strategy:** Classify errors at the CRYSTAL output level; apply configured recovery; escalate to manual review.

**Patterns:**
- `CrystalErrorDetector` classifies errors via regex on `.out` and SLURM `.o` files
- `ERROR_TYPE_ALIASES` dict in `recovery.py` normalizes detector names to YAML config keys
- Each recovery strategy specifies `max_retries`; on exhaustion, status is set to `FAILED_UNRECOVERABLE` in DB
- Optional calc types (BAND, DOSS, etc.) catch their own exceptions and set status without raising
- `QueueLockManager` uses `fcntl` file locks + randomized backoff to prevent race conditions on simultaneous callbacks

## Cross-Cutting Concerns

**Logging:** `print()` statements throughout; no structured logging framework. Output goes to stdout/stderr captured by SLURM.
**Validation:** Input validation is interactive (prompts with defaults); JSON configs validated on load via `load_d12_config()` / `load_d3_config()`.
**Authentication:** None — relies on HPC account (`mendoza_q`) configured in SLURM scripts.
**Path resolution:** `mace_config.py` at repo root provides canonical paths for basis sets, RCSR data, and template directories; supports `MACE_HOME` environment variable override.

---

*Architecture analysis: 2026-06-13*
