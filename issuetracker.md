# MACE Workflow Issue Tracker

## Issue Summary
This document tracks issues identified during MACE workflow execution on 2025-07-22.

**Update 2025-07-22**: Deep analysis completed. Issue #1 has been fixed, which should resolve several related issues.

**Update 2025-07-22 (Final)**: All high and medium priority issues have been fixed:
- Issue #1: FIXED - D12 file discovery (Critical)
- Issue #12: FIXED - AttributeError in d12_interactive.py (High)  
- Issue #4: FIXED - Expert mode sample files (High)
- Issue #8: FIXED - CHARGE+POTENTIAL validation (Medium)
- Issue #2: FIXED - Script copying warnings (Medium)
- Issue #6: FIXED - DOSS default value (Low)
- Issue #5: FIXED - FREQ phonon options (Low)
- Issue #7: FIXED - BAND format selection (Low)

**Update 2025-07-23**: Issues #3 and #10 have been fixed:
- Issue #3: FIXED - D3 expert mode now launches interactive configuration (Medium)
- Issue #10: FIXED - Workflow database conflicts resolved with comprehensive isolation system (High)

---

## Issue #1: D12 File Discovery and Execution Failure
**Status:** FIXED ✓  
**Severity:** Critical  
**Component:** workflow_executor.py

### Description
Workflow fails to execute after CIF to D12 conversion with error "No D12 files found for workflow execution!" despite D12 files being successfully created.

### Symptoms
- D12 files are created in the working directory
- Workflow detects existing D12 files and skips conversion
- Execution phase fails to find D12 files

### Root Cause Analysis
The workflow plan wasn't persisting the `generated_d12s` field after CIF conversion:
1. Plan is saved to JSON before CIF conversion (without D12 paths)
2. During execution, CIFs are converted and `plan['generated_d12s']` is updated in memory only
3. The updated plan is never saved back to disk
4. Later runs can't find the D12 files because the plan doesn't know where they are

### Fix Applied
1. **Save updated plan after CIF conversion** (workflow_executor.py line 850-855)
   - After generating D12s, save the updated plan with `generated_d12s` field
   
2. **Add fallback check for CIF workflows** (workflow_executor.py line 294-300)
   - If D12s not found through normal means, check input directory for CIF-converted files

### Impact on Other Issues
Fixing this issue should also resolve:
- **Issue #4** (partially) - Expert mode will have access to real D12 files
- **Issue #12** (potentially) - Real files mean proper functional extraction

---

## Issue #2: Expert Mode Script Copying Failures
**Status:** FIXED ✓  
**Severity:** Medium  
**Component:** workflow_planner.py

### Description
Expert mode configuration shows warnings about missing script files when trying to copy required scripts.

### Symptoms
```
Warning: Source not found: CRYSTALOptToD12.py
Warning: Source not found: d12creation.py
Warning: Source not found: StructureClass.py
Warning: Source not found: CRYSTALOptToD3.py
Warning: Source not found: d3_interactive.py
Warning: Source not found: d3_config.py
```

### Root Cause Analysis
The `_copy_required_scripts_for_expert_mode` method has multiple issues:
1. **Wrong path calculation**: Uses `Path(__file__).parent.parent / "Crystal_d12"` but scripts are at `Path(__file__).parent.parent.parent / "Crystal_d12"`
2. **Non-existent scripts**: References `d12creation.py` and `StructureClass.py` which don't exist in new structure
3. **Unnecessary with MACE**: All scripts are in PATH after `setup_mace.py --add-to-path`
4. **Silent failure**: Uses `if source_file.exists()` without error handling

### Actual Impact
- **Not critical**: Expert mode still works because scripts are found later via different path resolution
- **Confusing warnings**: Users see warnings but functionality works

### Fix Applied (2025-07-22)
Replaced the entire method body with a simple `pass` statement since:
1. With MACE setup, all scripts are already in PATH via `setup_mace.py`
2. No need to copy scripts to the working directory
3. Eliminates confusing warnings about missing scripts

---

## Issue #3: Expert Mode D3 Calculations Skip Configuration
**Status:** FIXED ✓  
**Severity:** Medium  
**Component:** workflow_planner.py

### Description
Expert mode for TRANSPORT, BAND, and DOSS calculations doesn't launch interactive configuration, jumping directly to SLURM setup.

### Symptoms
- Selecting expert mode (option 3) for D3 calculations shows message but no configuration
- No interactive CRYSTALOptToD3.py session launched
- Proceeds directly to SLURM resource configuration

### Root Cause Analysis
The expert mode handler for D3 calculations was:
1. Deferring configuration to execution time instead of planning time
2. Not launching CRYSTALOptToD3.py interactively during workflow planning
3. Missing the ability to find and use saved configuration files

### Fix Applied (2025-07-23)
1. **Created `_run_interactive_d3_config` method**
   - Launches CRYSTALOptToD3.py interactively during workflow planning
   - Uses correct command-line arguments: `--input`, `--calc-type`, `--save-config`
   - Creates temporary workspace for script execution
   
2. **Fixed CRYSTALOptToD3.py command arguments**
   - Removed incorrect arguments like `--out-file`, `--wf-file`, `--options-file`
   - Used proper argument format that the script expects
   
3. **Enhanced config file search patterns**
   - Searches for multiple naming patterns since CRYSTALOptToD3.py uses different defaults
   - Patterns include: `d3_config_BAND_*.json`, `d3_band_config.json`, etc.
   - Special handling for CHARGE+POTENTIAL naming variations

### Implementation Details
```python
# New method to run D3 expert configuration
def _run_interactive_d3_config(
    self, calc_type: str, real_out_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Run CRYSTALOptToD3.py interactively for expert D3 configuration"""
    # Creates temporary workspace
    # Copies real .out file if available
    # Runs CRYSTALOptToD3.py with correct arguments
    # Searches for saved config with flexible patterns
```

The fix ensures D3 expert mode behaves consistently with D12 expert mode, running interactive configuration during planning rather than deferring to execution time.

---

## Issue #4: Expert Mode Uses Generic Sample Files Instead of Actual Workflow Files
**Status:** FIXED ✓  
**Severity:** High  
**Component:** workflow_planner.py

### Description
Expert mode for subsequent steps (FREQ2, SP3) uses generic sample files instead of actual files from previous workflow steps, causing AttributeError and incorrect settings.

### Symptoms
```
AttributeError: 'NoneType' object has no attribute 'startswith'
File "/mnt/ffs24/home/jmendoza/bin/reorganization/Crystal_d12/d12_interactive.py", line 491
```
- Sample files don't contain actual calculation settings from workflow
- Functional is None because sample.out has no real DFT/HF settings
- Expert mode shows wrong calculation type (e.g., FREQ instead of SP)

### Root Cause Analysis
Expert mode creates minimal sample files with hardcoded content:
1. Creates `sample.out` with generic "Hartree-Fock (RHF)" method
2. Creates `sample.d12` with single hydrogen atom
3. These don't reflect actual materials or settings from the workflow
4. When CRYSTALOptToD12.py tries to extract settings, functional is None
5. The `.startswith()` call fails on None value

### Actual Code Issue
In `_run_interactive_crystal_opt_config` method, expert mode creates:
```python
sample_out.write_text("""CRYSTAL23 OUTPUT
...
Method: Hartree-Fock (RHF)  # Hardcoded, not from actual calculation
""")
```

### Impact of Issue #1 Fix
- **Partially helps**: With Issue #1 fixed, workflow has access to real D12 files
- **Still needs fixing**: Expert mode still creates sample files instead of using real ones

### Fix Applied (2025-07-22)
1. **Modified `_run_interactive_crystal_opt_config` to accept real D12 files**
   - Added optional `real_d12_path` parameter
   - Uses provided D12 file instead of creating minimal sample
   
2. **Updated all expert mode methods to find and pass real D12 files**
   - `_get_expert_sp_config`: Searches for D12 files from previous steps
   - `_get_expert_opt_config`: Same search pattern implementation
   - Expert FREQ configuration: Also searches for real D12 files
   
3. **Enhanced .out file handling**
   - Looks for corresponding .out file when real D12 is found
   - Copies real .out content instead of minimal template
   - Added default Method line to prevent AttributeError

This ensures expert mode uses actual workflow files with correct settings.

---

## Issue #5: Limited FREQ Advanced Options
**Status:** FIXED ✓  
**Severity:** Low  
**Component:** workflow_planner.py

### Description
Advanced FREQ mode in workflow planner offers only basic phonon options compared to full d12_calc_freq.py capabilities.

### Symptoms
- When selecting phonon mode (#2), only offers:
  1. Full dispersion with supercell
  2. Custom k-points only
  3. High-symmetry points only
- Missing the choice between phonon bands vs phonon DOS
- No access to the 8 templates available in d12_calc_freq.py

### Root Cause Analysis
The workflow planner's `configure_freq_advanced` method simplifies the d12 frequency options:
1. Doesn't show the full template menu (8 options in d12_calc_freq.py)
2. For phonon calculations, doesn't ask whether user wants bands or DOS
3. Missing distinction between calculation types after selecting "Full dispersion"

### What d12_calc_freq.py Actually Offers
```
1: Basic frequencies only
2: IR spectrum  
3: Raman spectrum
4: IR + Raman spectra
5: Thermodynamic properties
6: Phonon band structure    # Missing in workflow planner
7: Phonon density of states # Missing in workflow planner
8: Custom settings
```

### Fix Applied (2025-07-22)
Added phonon output type selection after "Full dispersion with supercell":
- Option 1: Phonon band structure
- Option 2: Phonon density of states  
- Option 3: Both band structure and DOS

The selection is stored in `config["frequency_settings"]["calculation_type"]` with values:
- "bands" for band structure
- "dos" for density of states
- "both" for both outputs

This matches the functionality available in d12_calc_freq.py templates 6 & 7.

---

## Issue #6: DOSS Advanced Mode Default
**Status:** FIXED ✓  
**Severity:** Low  
**Component:** workflow_planner.py

### Description
Advanced DOSS configuration should default to option 4 (Shell + AO contributions) to match D3 script behavior.

### Symptoms
- Currently defaults to option 0 (Total DOS only)
- User must manually select option 4 each time

### Root Cause Analysis
In `configure_doss_advanced` method (line 1627):
```python
proj_type = input("  Select projection type [0]: ").strip() or "0"
```
Default is hardcoded to "0" instead of "4".

### Code Location
- File: workflow_planner.py
- Method: `configure_doss_advanced`
- Line: 1627

### Fix Applied (2025-07-22)
Changed the default from "0" to "4" in line 1644:
```python
proj_type = input("  Select projection type [4]: ").strip() or "4"
```

This makes option 4 (Shell + AO contributions) the default, which is more useful for most analyses as it provides detailed DOS projections by both shell and atomic orbital contributions.

---

## Issue #7: Limited BAND Advanced Options
**Status:** FIXED ✓  
**Severity:** Low  
**Component:** workflow_planner.py

### Description
Advanced BAND configuration only offers automatic path, missing format options available in d3_interactive.py.

### Symptoms
- Only asks automatic vs custom path
- Missing format selection (labels/vectors/literature/SeeK-path)
- No template selection option

### Root Cause Analysis
The workflow planner's `configure_band_advanced` method oversimplifies compared to d3_interactive.py:
1. Only offers path choice (automatic/custom)
2. Doesn't ask about output format
3. Missing the 4 format options from d3_interactive.py

### What d3_interactive.py Offers
After choosing automatic path, it asks for format:
1. High-symmetry labels (CRYSTAL-compatible subset)
2. K-point vectors (fractional coordinates)
3. Literature path with vectors (comprehensive)
4. SeeK-path full paths (extended Bravais lattice notation)

### Code Location
- File: workflow_planner.py
- Method: `configure_band_advanced`
- Lines: ~1569-1591

### Fix Applied (2025-07-22)
Added format selection after choosing automatic path (lines 1617-1632):
```python
# Path format selection
print("\n  Path format for automatic k-path:")
print("    1. High-symmetry labels (CRYSTAL-compatible subset)")
print("    2. K-point vectors (fractional coordinates)")
print("    3. Literature path with vectors (comprehensive)")
print("    4. SeeK-path full paths (extended Bravais lattice notation)")

format_choice = input("  Select format [1]: ").strip() or "1"
```

The format is stored in `config["path_format"]` with values:
- "labels" for high-symmetry labels
- "vectors" for k-point vectors
- "literature" for literature paths
- "seekpath_full" for SeeK-path full notation

This matches the options available in d3_interactive.py.

---

## Issue #8: CHARGE+POTENTIAL Cannot Be Added
**Status:** FIXED ✓  
**Severity:** Medium  
**Component:** workflow_planner.py

### Description
CHARGE+POTENTIAL calculation cannot be added to workflow, showing "Cannot add CHARGE+POTENTIAL - check dependencies".

### Symptoms
- Repeatedly fails to add despite having SP calculations
- No clear dependency information provided
- Works in predefined templates but not custom workflows

### Root Cause Analysis
The regex pattern in `_validate_numbered_calc_addition` method doesn't handle the `+` character:
```python
# Line 1023 in workflow_planner.py
match = re.match(r"^([A-Z]+)(\d*)$", new_calc)
```
This pattern only matches uppercase letters followed by optional digits. When it encounters `CHARGE+POTENTIAL`, the regex returns None, causing validation to fail.

### Why Templates Work
Predefined templates bypass this validation since they're loaded directly with `CHARGE+POTENTIAL` already included.

### Code Location
- File: workflow_planner.py
- Method: `_validate_numbered_calc_addition`
- Line: 1023

### Fix Applied (2025-07-22)
Fixed the regex pattern in line 1023:
```python
# Changed from:
match = re.match(r"^([A-Z]+)(\d*)$", new_calc)
# To:
match = re.match(r"^([A-Z+]+)(\d*)$", new_calc)
```

The updated regex `[A-Z+]+` now matches uppercase letters AND the plus sign, allowing "CHARGE+POTENTIAL" to be properly validated and added to workflows.

---

## Issue #12: AttributeError in d12_interactive.py when functional is None
**Status:** FIXED ✓  
**Severity:** High  
**Component:** d12_interactive.py

### Description
AttributeError occurs when functional extraction returns None and code tries to call .startswith() on it.

### Symptoms
```
AttributeError: 'NoneType' object has no attribute 'startswith'
File "/mnt/ffs24/home/jmendoza/bin/reorganization/Crystal_d12/d12_interactive.py", line 491
```

### Root Cause Analysis
In `get_calculation_options_from_current` method:
```python
# Line 491
if functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] or functional.startswith("HF"):
```
When functional is None (due to extraction failure or invalid sample files), the `.startswith()` call fails.

### Impact of Issue #1 Fix
- **May help**: With real files available, functional extraction more likely to succeed
- **Still needs fixing**: None check should be added for safety

### Code Location
- File: d12_interactive.py
- Function: `get_calculation_options_from_current`
- Line: 491

### Fix Applied (2025-07-22)
Added None check in line 491 of d12_interactive.py:
```python
# Changed from:
if functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] or functional.startswith("HF"):
# To:
if functional and (functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] or functional.startswith("HF")):
```

This prevents the AttributeError when functional extraction returns None, allowing the code to fall through to the else clause which sets method_type to "DFT".

---

## Implementation Priority

### All Issues Resolved ✅

**Critical Issues (Fixed)**
1. **Issue #1** - ✅ FIXED - Workflow execution blocker (2025-07-22)

**High Priority Issues (Fixed)**
2. **Issue #12** - ✅ FIXED - AttributeError blocking expert mode (2025-07-22)
3. **Issue #4** - ✅ FIXED - Expert mode sample files (2025-07-22)

**Medium Priority Issues (Fixed)**
4. **Issue #8** - ✅ FIXED - CHARGE+POTENTIAL validation (2025-07-22)
5. **Issue #2** - ✅ FIXED - Script copying warnings (2025-07-22)
6. **Issue #3** - ✅ FIXED - D3 expert mode behavior (2025-07-23)

**Low Priority Issues (Fixed)**
7. **Issue #5** - ✅ FIXED - FREQ option expansion (2025-07-22)
8. **Issue #6** - ✅ FIXED - DOSS default value (2025-07-22)
9. **Issue #7** - ✅ FIXED - BAND option expansion (2025-07-22)

---

## Testing Requirements

After fixes are implemented:
1. ✅ Test full workflow with CIF to D12 conversion (Issue #1 fixed)
2. Test expert mode works without AttributeError
3. Ensure CHARGE+POTENTIAL can be added to workflows
4. Verify expert mode uses real workflow files
5. Test advanced options show all expected choices
6. Validate with both single and multiple materials

---

## Quick Fixes Summary

These issues can be fixed with minimal code changes:
- **Issue #6**: Change default from "0" to "4" (line 1627)
- **Issue #8**: Change regex from `[A-Z]+` to `[A-Z+]+` (line 1023)
- **Issue #12**: Add `if functional and` check (line 491)

---

## Issue #10: Workflow Database Conflicts
**Status:** FIXED ✓  
**Severity:** High  
**Component:** Multiple (workflow system, database, queue manager)

### Description
Running multiple workflows in the same directory causes database contamination and material ID conflicts.

### Symptoms
- Single `materials.db` shared by all workflows
- Material IDs collide between workflows
- Workflow progression triggers actions on wrong materials
- Error recovery attempts to "fix" calculations from other workflows

### Root Cause Analysis
1. All workflows share a single materials.db file
2. No namespace separation for material IDs
3. Queue manager operates globally across all workflows
4. No isolation between workflow contexts

### Fix Applied (2025-07-23)
Implemented comprehensive workflow isolation system:

1. **Created WorkflowContext class** (`mace/workflow/context.py`)
   - Three isolation modes: isolated, shared, hybrid
   - Thread-safe context management
   - Automatic cleanup and archival
   - Hidden directory structure (.mace_context_*)

2. **Updated Workflow Planner** 
   - Added isolation mode selection (Step 4.5)
   - Added post-completion action selection (Step 4.6)
   - Saves settings in workflow plan JSON

3. **Enhanced Workflow Executor**
   - Checks isolation mode from plan
   - Creates and activates WorkflowContext
   - Dynamic database/queue manager path updates
   - Handles post-completion actions

4. **Updated Context-Aware Database**
   - Automatic context detection
   - Seamless path resolution
   - Workflow-specific queries
   - Full backward compatibility

5. **Modified Queue Manager**
   - Uses ContextualMaterialDatabase when context active
   - Context-specific lock directories
   - Isolated status files

### Testing
Created comprehensive test script (`test_workflow_isolation.py`) that demonstrates:
- Running multiple isolated workflows simultaneously
- Shared mode compatibility
- Manual context switching
- Database isolation verification