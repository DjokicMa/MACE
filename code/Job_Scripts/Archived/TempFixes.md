# Workflow System Fixes Plan

## Critical Issues Identified

### 1. Database Creation Issues
- **AttributeError**: `create_fresh_database.py` references non-existent `self.d12_dir`
- **Status**: FIXED - Changed to `self.workflow_dir`

### 2. Input Settings Extraction Problems
- **D3 Files Incorrectly Showing HF**: BAND/DOSS d3 files are being assigned HF method when they inherit from SP calculations
- **Missing k_path_labels**: BAND d3 files should extract k-point path labels but aren't
- **Solution**: Fix `input_settings_extractor.py` to properly handle D3 inheritance and extract k-path data

### 3. Property Extraction Issues
- **Wrong Units**: Angles showing "Å" instead of "degrees", volumes showing "Å" instead of "Å³"
- **Missing Advanced Properties**: Effective mass, mobility, transport properties not being calculated
- **Formula Corruption**: Materials showing "Be" instead of correct formulas like "C"
- **Solution**: Fix `crystal_property_extractor.py` unit assignment logic and add advanced calculations

### 4. Workflow Progression Problems
- **OPT2 Mislabeling**: Second OPT is being labeled as SP instead of OPT2
- **Missing Script Copying**: SLURM scripts not copied to calculation directories
- **Hardcoded Workflow Order**: System assumes OPT→SP→BAND→DOSS instead of flexible sequence
- **Solution**: Fix `workflow_executor.py` to properly handle custom sequences

### 5. Database Schema Differences
- **Live vs Created**: Live production database missing `workflow_templates`, `workflow_instances` tables
- **Missing Columns**: Some tables missing metadata columns
- **Solution**: Ensure schema consistency between all database creation methods

### 6. File Dependencies
- **Copy Dependencies**: `copy_dependencies.py` missing files needed for live workflow
- **Solution**: Update file list to include all workflow components

## Advanced Properties Theory & Implementation

### Effective Mass Calculation
- **Theory**: Second derivative of energy with respect to momentum: `1/m* = d²E/dk²/ℏ²`
- **Implementation**: Estimate from band structure curvature near band edges
- **Classification**: Semimetal if effective mass below threshold with small/zero gap

### Transport Properties
- **Mobility**: Based on effective mass and scattering mechanisms
- **Conductivity**: Electronic and thermal transport coefficients
- **Seebeck Coefficient**: Thermoelectric transport property

### Material Classification
- **Metal**: Finite DOS at Fermi level above threshold
- **Semimetal**: Near-zero gap with low DOS at Fermi level  
- **Semiconductor**: Clear band gap with zero DOS at Fermi level

## Implementation Order

1. **Fix create_fresh_database.py**: Complete property extraction with correct units
2. **Fix input_settings_extractor.py**: Proper D3 handling and k-path extraction
3. **Fix crystal_property_extractor.py**: Unit assignment and advanced properties
4. **Fix workflow system**: OPT2 handling, script copying, flexible sequences
5. **Update documentation**: Theory explanations and workflow guides
6. **Update copy_dependencies.py**: Complete file list

## Files to Modify

### Core Fixes
- `create_fresh_database.py` - Database creation with fixed references
- `input_settings_extractor.py` - D3 file handling and k-path extraction
- `crystal_property_extractor.py` - Unit fixes and advanced properties
- `formula_extractor.py` - Prevent formula corruption

### Workflow Fixes  
- `workflow_executor.py` - OPT2 handling and script copying
- `workflow_planner.py` - Flexible sequence support
- `enhanced_queue_manager.py` - Proper workflow progression

### Supporting Files
- `copy_dependencies.py` - Complete file dependencies
- Documentation files for theory and usage