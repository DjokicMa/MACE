# MACE Codebase Authorship and Attribution

This document provides authorship attribution for the MACE (Mendoza Automated CRYSTAL Engine) codebase.

## Primary Developer
**Marcus Djokic** (PhD Student) - Primary developer and architect of the MACE system

## Contributors
- **Daniel Maldonado Lopez** (PhD Student)
- **Brandon Lewis** (Undergraduate)
- **William Comaskey**
- **Kevin Lucht**
- **Wangwei Lan**

## Principal Investigator
**Prof. Jose Luis Mendoza-Cortes** - Michigan State University, Mendoza Group

---

## Detailed Script Attribution

### Crystal_d12/

#### Marcus Djokic
**Core Modules:**
- `d12_calc_basic.py` - Basic calculation configuration module
- `d12_calc_freq.py` - Frequency calculation module
- `d12_config.py` - JSON configuration management
- `d12_constants.py` - Constants and defaults
- `d12_from_config.py` - Unified configuration interface
- `d12_interactive.py` - Interactive prompts and utilities
- `d12_parsers.py` - File parsing utilities
- `d12_writer.py` - D12 file writing utilities
- `NewCifToD12.py` - Primary CIF to D12 conversion script
- `CRYSTALOptToD12.py` - Extract optimized geometry to new D12 (reworked version)

**Archived Scripts:**
- `CRYSTALtoCIF-V2.py` - Version 2 of CRYSTAL to CIF converter

#### Daniel Maldonado Lopez
**Core Scripts:**
- `create_d12_w-ghosts.py` - Automatic ghost atom insertion
- `manual_create_d12_w-ghosts.py` - Manual ghost atom insertion

**Archived Scripts:**
- `CRYSTAL2cif.py` - CRYSTAL to CIF conversion
- `CRYSTAL2cif_slab.py` - CRYSTAL to CIF conversion for slabs
- `crystal_to_cif_python3.py` - Python 3 compatible CIF converter

#### William Comaskey
**Archived Scripts:**
- `crys22.py` - CRYSTAL22 related utilities

#### Mixed Attribution
- `CRYSTALOptToD12.py` (original) - Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic

### Crystal_d3/

#### Marcus Djokic
- `CRYSTALOptToD3.py` - Main D3 generation script
- `d3_config.py` - Configuration management for D3
- `d3_interactive.py` - Interactive configuration for D3
- `d3_kpoints.py` - K-point path generation
- `create_Transportd3.py` (archived) - Transport property calculations
- `alldos_old.py` (archived)
- `d3_config_old.py` (archived)

#### William Comaskey
- `d3_matdump.py` - MATDUMP: the CRYSTAL `properties` matrix-dump calculation
  type, which prints H(R) and S(R) in the direct-lattice representation

**The LCAO->Wannier90 method, and the `lcao2wannier` package that implements
it, are William Comaskey's work.** MACE generates the CRYSTAL input and
orchestrates the run. It implements no part of the conversion - not the Fourier
transform to H(k)/S(k), not the generalized eigenproblem in the non-orthogonal
AO basis, not the SCDM projections for `.amn`, and not the analytic
momentum-shifted GTO overlaps for `.mmn`. That is the entire scientific content
of the feature, and it is his.

`lcao2wannier` is consumed as an optional external dependency and is never
vendored into this repository, so its LICENSE and copyright stay with the
author.

**CITATION: TODO** - ask William Comaskey which citation he wants (the package,
a paper, or both) before any citation string is published in MACE. Do not
invent one.

#### Mixed Attribution
- `alldos.py` (archived) - Density of states generation (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)
- `create_band_d3.py` (archived) - Band structure generation (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)

### mace/ Package

#### Marcus Djokic
**Root Scripts:**
- `enhanced_queue_manager.py` - Enhanced queue management with material tracking
- `material_monitor.py` - Material monitoring dashboard
- `run_workflow.py` - Main workflow runner

**Database Module:**
- `utils/database_status_report.py` - Database status reporting
- `populate_completed_jobs.py` - Populate completed jobs in database
- `query/queries.py` - Database query utilities

**Queue Module:**
- `manager.py` - Enhanced tracking queue manager (EnhancedCrystalQueueManager)

**Wannier Module (orchestration only - see William Comaskey below):**
- `wannier/driver.py` - Invokes the `lcao2wannier` package and surfaces its
  diagnostics
- `wannier/cli.py` - `mace wannier` front-end

**Recovery Module:**
- `pandas_utils.py` - Pandas utility functions

**Submission Module:**
- `__init__.py`
- `check_submitted.py` - Check submitted jobs
- `submit_d12.py` - D12 job submission
- `submit_d3.py` - D3 job submission
- `submit_frequency.py` - Frequency calculation submission

#### William Comaskey
- The LCAO->Wannier90 method and the `lcao2wannier` package driven by
  `mace/wannier/`. MACE's modules there only build an argument list, run his
  package, and report what it says; every scientific step of the conversion is
  his. `lcao2wannier` is an optional external dependency and is never vendored.
  **CITATION: TODO** - ask him which citation he wants; do not invent one.

#### Marcus Djokic (continued)

**Utils Module:**
- `animation.py` - Loading animation utilities
- `animation_simple.py` - Simple animation utilities
- `basis_analysis.py` - Basis set analysis
- `cif_parser.py` - CIF file parsing
- `formula.py` - Chemical formula utilities
- `lattice_utils.py` - Lattice parameter utilities
- `plotting_utils.py` - Plotting utilities
- `property_aggregator.py` - Property aggregation
- `property_analysis.py` - Property analysis utilities
- `slurm_utils.py` - SLURM job utilities
- `structure_analyzer.py` - Structure analysis
- `symmetry_utils.py` - Symmetry utilities

**Workflow Module:**
- `interactive_monitor.py` - Interactive monitoring
- `monitor.py` - Workflow monitoring
- `progress_tracker.py` - Progress tracking
- `resource_optimizer.py` - Resource optimization
- `status_analyzer.py` - Status analysis

### code/ Directory

#### Marcus Djokic
**Check_Scripts:**
- `check_completedV2.py` - Check completed calculations
- `check_erroredV2.py` - Check errored calculations
- `updatelists2.py` - Update job lists
- `fixk.py` - Fix k-point issues

**Plotting_Scripts:**
- `overview.py` (archived) - Legacy overview script

#### Daniel Maldonado Lopez
**Band_Alignment:**
- `getWF.py` - Work function extraction

#### Kevin Lucht
**Plotting_Scripts:**
- `plotting.py` - General plotting utilities (edited by William Comaskey 03/23/2022)

#### Mixed Attribution
**SLURM Submission Scripts:**
- `submitcrystal23.py` - CRYSTAL job submission (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)
- `submit_prop.py` - Properties job submission (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)
- `submitcrystal23.sh` - SLURM template script (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)
- `submit_prop.sh` - SLURM template script (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)

**OldSLURMTemplates:** (7 files) 
- All submission template scripts (Prior contributions from Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic)

#### Unknown/To Be Determined
**code/NewPlotting_Scripts:**
- `autoBands.py`
- `autoPhononBands.py`

(Other previously listed unattributed scripts — crystalOutputToJson.py,
getOptimizationProgress.py, getPropertyData.py, getSpecialKPoints.py,
makelatexTable.py, makePlot.py, plotResultsParallel.py, createTodoLists.py,
check_erroredJobs.py, autoDOS_new.py, plotResults.py, result_pdf.py — were
removed from the repository in the v1.1.0 cleanup; see git history.)

---

## Notes

1. Scripts marked as "Unknown/To Be Determined" require additional investigation to determine proper attribution.
2. This document should be updated as new scripts are added or attribution information is discovered.
3. All contributors are affiliated with Michigan State University, Mendoza Group unless otherwise noted.
4. The MACE system represents a collaborative effort with Marcus Djokic as the primary architect and developer.

## Recommended Action

For scripts without clear attribution, it is recommended to:
1. Add appropriate header comments with author information
2. Include creation date and modification history
3. Reference this document for attribution

Last Updated: July 12, 2026
