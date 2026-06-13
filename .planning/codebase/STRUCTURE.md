# Codebase Structure

**Analysis Date:** 2026-06-13

## Directory Layout

```
reorganization/                    # Repo root (MACE_HOME)
├── mace_cli                       # Compiled CLI binary (entry point)
├── mace_config.py                 # Central path configuration for all components
├── mace/                          # Core MACE Python package
│   ├── __init__.py
│   ├── run_mace.py                # Main CLI script (interactive/execute/status)
│   ├── run_workflow.py            # Alternate workflow runner
│   ├── completion_checker.py      # Standalone completion checker
│   ├── enhanced_queue_manager.py  # Legacy top-level queue manager shim
│   ├── material_monitor.py        # Background daemon monitor
│   ├── config/
│   │   ├── __init__.py
│   │   └── recovery_config.yaml   # Error recovery strategies (per error type)
│   ├── workflow/
│   │   ├── engine.py              # WorkflowEngine — step orchestration (~3587 lines)
│   │   ├── planner.py             # WorkflowPlanner — plan generation (~5006 lines)
│   │   ├── executor.py            # WorkflowExecutor — plan execution (~2446 lines)
│   │   ├── context.py             # WorkflowContext — isolation management
│   │   ├── callback.py            # Job completion callback handler
│   │   ├── status.py              # Status query and display
│   │   ├── monitor_workflow.py    # Workflow-level monitor
│   │   ├── check_workflows.py     # Workflow health checks
│   │   ├── dummy_file_creator.py  # Creates placeholder files for dry-run planning
│   │   ├── run_workflow_animated.py  # Animated TUI runner
│   │   ├── run_workflow_isolated.py  # Isolated-context runner
│   │   └── common/
│   │       └── constants.py       # Shared workflow constants
│   ├── queue/
│   │   ├── manager.py             # EnhancedCrystalQueueManager (~1810 lines)
│   │   ├── monitor.py             # SLURM queue polling daemon
│   │   ├── queue_lock_manager.py  # File + thread distributed locking
│   │   └── legacy_manager.py      # Original queue manager (preserved for reference)
│   ├── recovery/
│   │   ├── detector.py            # CrystalErrorDetector — error classification
│   │   ├── recovery.py            # ErrorRecoveryEngine — fix and resubmit
│   │   └── pandas_utils.py        # Optional pandas wrapper (graceful degradation)
│   ├── database/
│   │   ├── materials.py           # MaterialDatabase (SQLite + ASE, ~1508 lines)
│   │   ├── materials_contextual.py  # ContextualMaterialDatabase
│   │   ├── populate_completed_jobs.py  # Backfill DB from existing output files
│   │   ├── analysis/              # Statistical analysis of DB contents
│   │   │   ├── aggregation.py
│   │   │   ├── comparison.py
│   │   │   ├── correlation.py
│   │   │   ├── distribution.py
│   │   │   ├── missing_data.py
│   │   │   └── workflow_progress.py
│   │   ├── query/                 # Query builder utilities
│   │   │   ├── filters.py
│   │   │   ├── advanced_filters.py
│   │   │   └── queries.py
│   │   ├── export/                # Export DB data to various formats
│   │   │   ├── formats.py
│   │   │   └── visualization.py
│   │   └── interactive/           # Interactive DB browser
│   │       └── interactive.py
│   ├── submission/
│   │   ├── crystal.py             # SLURM script generator for D12 jobs
│   │   ├── properties.py          # SLURM script generator for D3/property jobs
│   │   ├── portable_slurm_generator.py  # Portable SLURM script builder
│   │   └── templates/             # SLURM template files (if any)
│   ├── plotting/
│   │   └── main.py                # MACE-integrated plotting dispatcher
│   └── utils/
│       ├── property_extractor.py  # CrystalPropertyExtractor — regex parsing of .out
│       ├── settings_extractor.py  # Extract calc settings from output files
│       ├── scf_settings_extractor.py  # SCF-specific settings extraction
│       ├── formula_extractor.py   # Chemical formula parsing
│       ├── dat_file_processor.py  # .DAT / .f25 file processing
│       ├── file_manager.py        # CrystalFileManager — file ops utilities
│       ├── node_exclusion.py      # SLURM node exclusion (AMD20 blacklist)
│       ├── advanced_electronic_analyzer.py
│       ├── population_analysis_processor.py
│       ├── check_property_units.py
│       ├── copy_dependencies.py
│       ├── installer.py
│       ├── mace_env_helper.py
│       ├── animation.py           # TUI animation helpers
│       ├── banner.py              # CLI banner display
│       ├── mace_quick_animation.py
│       ├── show_properties.py
│       └── analyze_script_dependencies.py
├── Crystal_d12/                   # D12 input generation (standalone + importable)
│   ├── NewCifToD12.py             # CIF → D12 converter (uses ASE/spglib, ~1381 lines)
│   ├── CRYSTALOptToD12.py         # Prior output → next D12 (geometry extraction)
│   ├── d12_from_config.py         # Unified JSON-config-driven D12 dispatcher
│   ├── d12_config.py              # D12 JSON config management
│   ├── d12_constants.py           # Atomic numbers, space groups, FREQ templates
│   ├── d12_parsers.py             # Output file parsers (~1359 lines)
│   ├── d12_writer.py              # D12 file writer
│   ├── d12_calc_basic.py          # Basic calculation builders
│   ├── d12_calc_freq.py           # FREQ calculation builders
│   ├── d12_interactive.py         # Interactive D12 builder
│   ├── CrystalOutToCif.py         # CRYSTAL output → CIF converter
│   ├── CRYSTALOptToD12.py         # (see above)
│   ├── check_geometry_mismatch.py # Geometry consistency checker
│   ├── NewCifToD12.py             # (see above)
│   ├── basis_sets/
│   │   ├── full.basis.doublezeta/ # DZVP-REV2 basis set files
│   │   ├── full.basis.triplezeta/ # TZVP-REV2 basis set files
│   │   └── stuttgart/             # Stuttgart ECP basis set files
│   ├── Ghosts/                    # Ghost-atom D12 generation scripts
│   ├── example_configs/           # Named JSON config templates for common calc types
│   │   ├── standard_dft_opt.json
│   │   ├── high_accuracy_sp.json
│   │   ├── quick_screen.json
│   │   ├── freq_analysis.json
│   │   ├── metallic_system.json
│   │   ├── 3c_composite.json
│   │   ├── phonon_bands.json
│   │   └── surface_slab.json
│   └── Archived/                  # Superseded converter scripts (reference only)
├── Crystal_d3/                    # D3 input generation (standalone + importable)
│   ├── CRYSTALOptToD3.py          # SP/OPT output → D3 property input generator
│   ├── d3_config.py               # D3 JSON config management
│   ├── d3_interactive.py          # Interactive D3 builder
│   ├── d3_kpoints.py              # K-point mesh generation for D3
│   ├── seekpath_interface.py      # seekpath wrapper for high-symmetry k-paths
│   ├── test_seekpath_interface.py # Seekpath unit tests
│   ├── RCSR/                      # RCSR net topology data (2P, 3P periodic nets)
│   │   ├── 2P/                    # 2-periodic net data
│   │   └── 3P/                    # 3-periodic net data
│   ├── example_configs/           # Named JSON config templates for D3 calc types
│   │   ├── band_auto_everything.json
│   │   ├── band_high_symmetry.json
│   │   ├── doss_total_only.json
│   │   ├── doss_element_orbital_auto.json
│   │   ├── doss_orbital_projections.json
│   │   ├── charge_density_3d.json
│   │   └── transport_auto_fermi.json
│   └── Archived/                  # Superseded D3 scripts (reference only)
├── code/                          # Standalone utility scripts (not in MACE package)
│   ├── Check_Scripts/             # Completion/error checking scripts
│   │   ├── check_completedV2.py
│   │   ├── check_erroredV2.py
│   │   ├── updatelists2.py        # Error list management (used by recovery layer)
│   │   └── fixk.py                # SHRINK/k-point fix script (used by recovery layer)
│   ├── Post_Processing_Scripts/
│   │   └── grab_properties.py
│   ├── Band_Alignment/            # Band alignment analysis scripts
│   │   ├── CBM_VBM.py
│   │   └── getWF.py
│   ├── NewPlotting_Scripts/       # Current plotting scripts
│   │   ├── AutoBands/autoBands.py
│   │   ├── AutoDOS/autoDOS.py
│   │   ├── AutoChargeDens/format_and_plot_f25_new.py
│   │   └── AutoPhononBands/autoPhononBands.py
│   ├── Plotting_Scripts/          # Older plotting scripts (partially superseded)
│   └── OldSLURMTemplates/         # Legacy SLURM submission scripts
├── Plotting/                      # Additional interactive plotting scripts
│   ├── ipBANDS_V2.py
│   ├── ipDOS_V2.py
│   └── plottingCIFs.py
├── test/                          # Real CRYSTAL output files for parser verification
│   ├── OPT/                       # Geometry optimization outputs (.d12, .out, .f9)
│   ├── SP/                        # Single-point calculation outputs
│   ├── BAND/                      # Band structure outputs (.BAND.DAT, .f25, .d3)
│   ├── DOSS/                      # Density of states outputs
│   ├── FREQ/                      # Frequency/phonon calculation outputs
│   ├── TRANSPORT/                 # Transport property outputs
│   ├── ECH3POT3/                  # Charge density + potential outputs
│   │   └── .queue_locks/          # Runtime queue lock files
│   └── CIFs/                      # Input CIF structure files
├── activate_mace.sh               # Environment activation script
├── setup_mace.py                  # Installation/setup script
├── mace_examples.sh               # Example usage commands
├── requirements.txt               # Python package requirements
├── mace_config.py                 # Central path config (see above)
├── README.md
├── DOCUMENTATION.md
├── INSTALLATION.md
├── AUTHORSHIP.md
├── CHANGELOG.md
├── CODEBASE_AUDIT.md
└── .planning/
    └── codebase/                  # GSD planning documents
```

## Directory Purposes

**`mace/`:**
- Purpose: The core MACE Python package — workflow orchestration, queue management, error recovery, database, submission
- Key files: `run_mace.py` (entry), `workflow/engine.py`, `workflow/planner.py`, `queue/manager.py`, `database/materials.py`
- Importable as package: Yes (has `__init__.py`)

**`Crystal_d12/`:**
- Purpose: All logic for generating CRYSTAL D12 input files from CIF structures or prior CRYSTAL outputs
- Key files: `NewCifToD12.py`, `CRYSTALOptToD12.py`, `d12_from_config.py`, `d12_constants.py`, `d12_parsers.py`
- Note: Scripts require Anaconda Python (ASE, spglib deps). Invoked as subprocess by workflow layer.
- Only `d12_constants.py` is safe to import directly (no external deps)

**`Crystal_d3/`:**
- Purpose: All logic for generating CRYSTAL D3 property input files (BAND, DOSS, ECH3, POT3, TRANSPORT)
- Key files: `CRYSTALOptToD3.py`, `d3_config.py`, `seekpath_interface.py`, `d3_kpoints.py`
- Note: `seekpath_interface.py` requires seekpath (Anaconda only)

**`Crystal_d12/basis_sets/`:**
- Purpose: Stored basis set definition files for DZVP-REV2, TZVP-REV2, and Stuttgart ECPs
- Generated: No (static data files)
- Committed: Yes

**`Crystal_d3/RCSR/`:**
- Purpose: Reticular Chemistry Structure Resource net topology data used for periodic net analysis
- Generated: No (static reference data)
- Committed: Yes

**`Crystal_d12/example_configs/` and `Crystal_d3/example_configs/`:**
- Purpose: Named, reusable JSON configuration templates for standard calculation setups
- Usage: Pass to `d12_from_config.py --config <name>.json` or load via `load_d12_config()`
- Add new configs here when a recurring calculation type needs a standard preset

**`code/`:**
- Purpose: Standalone utility scripts that predate MACE or operate independently of the workflow
- Key scripts: `Check_Scripts/fixk.py` (used by error recovery), `Check_Scripts/updatelists2.py`
- Not part of the `mace` package; not importable as a package

**`test/`:**
- Purpose: Real CRYSTAL output files from actual HPC runs; used to verify parsers and property extractors
- Structure: One subdirectory per calculation type (OPT, SP, BAND, DOSS, FREQ, TRANSPORT, ECH3POT3, CIFs)
- Do NOT add synthetic fixtures here — use real `.out` outputs for parser testing

**`mace/config/`:**
- Purpose: Runtime configuration files for MACE behavior
- Key file: `recovery_config.yaml` — defines all automated error recovery strategies

**`Archived/` subdirectories:**
- Purpose: Superseded scripts preserved for reference; not part of active workflow
- Located under: `Crystal_d12/Archived/`, `Crystal_d3/Archived/`, `code/Check_Scripts/Archived/`, `code/Plotting_Scripts/Archived/`
- Do NOT modify or import from Archived unless explicitly restoring functionality

## Key File Locations

**Entry Points:**
- `mace_cli`: Compiled binary CLI entry point (root)
- `mace/run_mace.py`: Primary Python CLI script
- `mace/run_workflow.py`: Alternate workflow runner

**Configuration:**
- `mace_config.py`: All canonical directory paths for the project
- `mace/config/recovery_config.yaml`: Error recovery strategies and parameters
- `Crystal_d12/example_configs/*.json`: D12 calculation presets
- `Crystal_d3/example_configs/*.json`: D3 calculation presets

**Core Logic:**
- `mace/workflow/planner.py`: Workflow planning (~5006 lines)
- `mace/workflow/engine.py`: Step orchestration (~3587 lines)
- `mace/workflow/executor.py`: Plan execution (~2446 lines)
- `mace/queue/manager.py`: SLURM queue management (~1810 lines)
- `mace/database/materials.py`: Material/calculation database (~1508 lines)
- `Crystal_d12/NewCifToD12.py`: CIF-to-D12 conversion (~1381 lines)
- `Crystal_d12/d12_parsers.py`: Output file parsing (~1359 lines)

**Error Recovery:**
- `mace/recovery/detector.py`: Error classification
- `mace/recovery/recovery.py`: Recovery strategies and resubmission
- `code/Check_Scripts/fixk.py`: SHRINK/k-point fix (called by recovery engine)

**Testing Data:**
- `test/OPT/*.out`: Optimization calculation outputs for parser testing
- `test/SP/*.out`: Single-point outputs
- `test/BAND/*.BAND.DAT`, `test/BAND/*.f25`: Band structure data files
- `test/DOSS/`: DOS output files

## Naming Conventions

**Files:**
- Python scripts: `snake_case.py` for modules/utilities, `PascalCase.py` for standalone converter scripts (e.g., `NewCifToD12.py`, `CRYSTALOptToD12.py`)
- SLURM job scripts: `<material_name>.sh`
- CRYSTAL input files: `<material_name>.d12` (D12), `<material_name>.d3` (D3)
- CRYSTAL output files: `<material_name>.out`, SLURM output: `<material_name>-<jobid>.o`
- Basis set files: `*.basis` or named by element in subdirectory

**CRYSTAL File Naming (important for parsers):**
- Full chain encoded in filename: `<material>_<calc1>_<result>_<calc2>_<result>_<calc3>.ext`
- Example: `1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out`
- The `create_material_id_from_file()` function in `mace/database/materials.py` must parse this chain

**Directories:**
- Calculation type directories in `test/`: ALL_CAPS (OPT, SP, BAND, DOSS, FREQ)
- Package subdirectories: `snake_case` (workflow, queue, recovery, database, utils)
- Archived material: `Archived/` (PascalCase subdirectory)

**Classes:**
- PascalCase: `WorkflowEngine`, `MaterialDatabase`, `CrystalErrorDetector`, `EnhancedCrystalQueueManager`

**Functions/methods:**
- `snake_case`: `create_material_id_from_file()`, `load_d12_config()`, `canonical_error_type()`

## Where to Add New Code

**New workflow step type (new CRYSTAL calc type):**
- Add to `OPTIONAL_CALC_TYPES` or blocking set in `mace/workflow/engine.py`
- Add submission logic in `mace/submission/properties.py` or `mace/submission/crystal.py`
- Add D3 config template to `Crystal_d3/example_configs/`
- Add test outputs to `test/<CALC_TYPE>/`

**New error recovery strategy:**
- Add handler entry to `mace/config/recovery_config.yaml`
- Implement handler method in `mace/recovery/recovery.py`
- Add error classification pattern to `mace/recovery/detector.py`
- Add alias mapping if needed to `ERROR_TYPE_ALIASES` in `mace/recovery/recovery.py`

**New property extracted from CRYSTAL output:**
- Add regex pattern + extraction logic to `mace/utils/property_extractor.py` (`CrystalPropertyExtractor`)
- Add DB column if needed to `mace/database/materials.py` (`_initialize_database`)
- Verify against real output in `test/` directories — do not use synthetic fixtures

**New D12 calculation preset:**
- Add JSON file to `Crystal_d12/example_configs/`
- Add default dict entry in `Crystal_d12/d12_config.py` (`get_default_d12_configs()`)

**New D3 calculation preset:**
- Add JSON file to `Crystal_d3/example_configs/`
- Add config support in `Crystal_d3/d3_config.py`

**New plotting script:**
- Add to `code/NewPlotting_Scripts/<CalcType>/` with auto-detection of input format
- Update `mace/plotting/main.py` dispatcher if integrating with MACE CLI

**New database query or analysis:**
- Queries: `mace/database/query/queries.py` or `filters.py`
- Analysis: `mace/database/analysis/` (appropriate submodule)
- Export: `mace/database/export/formats.py`

**New utility/helper shared across layers:**
- Location: `mace/utils/<descriptive_name>.py`
- Import in consuming modules; do not add to `Crystal_d12/` or `Crystal_d3/`

## Special Directories

**`.mace_context_<workflow_id>/`:**
- Purpose: Per-workflow isolated resources (DB, locks, storage) created at runtime
- Generated: Yes (by `WorkflowContext` at runtime)
- Committed: No (in `.gitignore`)

**`workflow_staging/`:**
- Purpose: Temporary working directories for each workflow step's file staging
- Generated: Yes (by `WorkflowEngine`)
- Committed: No

**`Crystal_d12/basis_sets/`:**
- Purpose: Static basis set data referenced at D12 generation time
- Generated: No
- Committed: Yes

**`test/`:**
- Purpose: Canonical real-world output files for parser verification
- Generated: No (captured from real HPC runs)
- Committed: Yes — treat as ground truth for parser behavior

**`Archived/` (multiple locations):**
- Purpose: Superseded scripts kept for historical reference
- Generated: No
- Committed: Yes — do not remove; do not add new code here

---

*Structure analysis: 2026-06-13*
