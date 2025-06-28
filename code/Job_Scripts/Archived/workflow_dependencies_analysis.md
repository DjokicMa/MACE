# CRYSTAL Workflow System Dependencies Analysis

## Overview
The workflow system (workflow_planner.py, workflow_executor.py, run_workflow.py) orchestrates complex CRYSTAL calculation sequences. This document maps all dependencies and integration points.

## Core Python Module Dependencies

### Internal Job_Scripts Dependencies
1. **material_database.py** - SQLite database management
2. **enhanced_queue_manager.py** - SLURM job queue management  
3. **workflow_engine.py** - Workflow state and progression
4. **error_recovery.py** - Error detection and recovery
5. **error_detector.py** - Error pattern recognition
6. **crystal_property_extractor.py** - Extract properties from output files
7. **formula_extractor.py** - Extract chemical formulas and space groups
8. **input_settings_extractor.py** - Extract settings from D12/D3 files
9. **query_input_settings.py** - Query stored settings

### External Script Dependencies

#### Crystal_To_CIF Directory
1. **NewCifToD12.py** - Convert CIF files to D12 format
   - Called by: workflow_executor.py for CIF conversion
   - Location: `../Crystal_To_CIF/NewCifToD12.py`
   
2. **CRYSTALOptToD12.py** - Generate SP/FREQ inputs from optimized structures
   - Called by: workflow_executor.py for post-optimization steps
   - Location: `../Crystal_To_CIF/CRYSTALOptToD12.py`
   
3. **d12creation.py** - Shared utilities and constants
   - Imported by: workflow_planner.py, workflow_executor.py
   - Location: `../Crystal_To_CIF/d12creation.py`

#### Creation_Scripts Directory  
1. **create_band_d3.py** - Generate band structure input files
   - Called by: workflow_executor.py (referenced but not fully implemented)
   - Location: `../Creation_Scripts/create_band_d3.py`
   
2. **alldos.py** - Generate density of states input files
   - Called by: workflow_executor.py (referenced but not fully implemented)
   - Location: `../Creation_Scripts/alldos.py`

### SLURM Script Templates
1. **submitcrystal23.sh** - Main CRYSTAL calculation submission
2. **submit_prop.sh** - Properties calculation submission
3. Template generation for workflow-specific scripts:
   - `submitcrystal23_opt_{step_num}.sh`
   - `submitcrystal23_{calc_type}_{step_num}.sh`
   - `submit_prop_{calc_type}_{step_num}.sh`

## Directory Structure Dependencies

### Required Directories
```
working_directory/
├── workflow_configs/          # JSON configuration files
├── workflow_scripts/          # Generated SLURM scripts
├── workflow_inputs/           # Initial input files
├── workflow_outputs/          # Calculation outputs
└── workflow_temp/             # Temporary files
```

### Basis Set Dependencies
```
Creation_Scripts/basis/
├── full.basis.doublezeta/    # Double-zeta basis sets (by atomic number)
├── full.basis.triplezeta/    # Triple-zeta basis sets (by atomic number)
├── stuttgart/                # Stuttgart pseudopotentials
└── sopseud/                  # SOC pseudopotentials (.mol files)
```

### RCSR Database Integration
```
Creation_Scripts/RCSR/
├── 2P/                       # 2-periodic structure templates
├── RCSR-2P.csv              # 2P structure metadata
└── RCSR-3P.csv              # 3P structure metadata
```

## Integration Architecture

### Workflow Planning Phase
1. **Input Detection**: CIF files or existing D12 files
2. **Configuration**: 
   - Level 1-3 CIF customization via NewCifToD12.py integration
   - Expert mode runs CRYSTALOptToD12.py interactively
3. **JSON Plan Generation**: Complete workflow configuration saved

### Workflow Execution Phase
1. **CIF Conversion** (if needed):
   - Calls `NewCifToD12.py` with JSON configuration
   - Timeout protection (5 minutes)
   - Batch processing support

2. **Step Progression**:
   - OPT calculations submitted via generated SLURM scripts
   - Post-OPT steps use `CRYSTALOptToD12.py`
   - BAND/DOSS use respective Creation_Scripts tools

3. **File Management**:
   - Individual folders per material: `mat_1_dia/`, `mat_2_dia2/`
   - Individual SLURM scripts per material
   - Scratch directory isolation: `$SCRATCH/workflow_ID/step_N/material/`

### Callback Integration
All generated SLURM scripts include enhanced callback mechanism:
```bash
# Multi-location queue manager detection
# Checks local directory first, then parent directories
# Supports both enhanced and legacy queue managers
```

## Key Integration Points

### subprocess.run() Calls
1. **NewCifToD12.py**: Batch CIF conversion with timeout
2. **CRYSTALOptToD12.py**: Interactive and batch modes
3. **sbatch**: Direct SLURM job submission

### Path Resolution Strategy
1. Check for local script copy first
2. Fall back to parent directory: `Path(__file__).parent.parent / "Crystal_To_CIF"`
3. Dynamic sys.path modification for imports

### Material ID Management
- Handles complex naming from NewCifToD12.py output
- Cleans material names for folder/script generation
- Maintains database consistency across workflow steps

## Configuration Persistence
- JSON files store complete workflow state
- Enables workflow reproduction and sharing
- Integration with material database for provenance

## Error Handling
- Timeout protection for external script calls
- File existence verification
- Graceful fallback for missing dependencies
- Integration with error_recovery.py for automated fixes

## SLURM Resource Management
- Dynamic resource allocation per calculation type
- Template-based script generation
- Individual job submission per material
- Callback-based workflow progression