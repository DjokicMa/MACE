# CRYSTAL Job Status Analysis and Error Handling

This folder contains CRYSTAL-specific utilities for automatically analyzing calculation results, classifying job completion status, and implementing targeted error fixes for common CRYSTAL calculation issues.

## Included Scripts

### 1. updatelists2.py

* Language: Python 3
* Required Libraries: `os`, `glob`, `csv`
* Purpose: Scans all `.out` files and categorizes job status automatically.
* Output: Generates multiple `.csv` files:

  * `complete_list.csv`
  * `completesp_list.csv`
  * `too_many_scf_list.csv`
  * `memory_list.csv`
  * `shrink_error_list.csv`
  * `geometry_small_dist_list.csv`
  * `potential_list.csv`
  * `unknown_list.csv`
* Logic: Uses CRYSTAL-specific error messages to classify jobs.

### 2. check\_completedV2.py

* Language: Python 3
* Required Libraries: `os`, `shutil`, `csv`
* Purpose: Moves all successfully completed jobs to a `done/` folder.
* Input: `complete_list.csv` or `completesp_list.csv`
* Moves: `.sh`, `.out`, `.d12`, `.f9` (matching job names)

### 3. check\_erroredV2.py

* Language: Python 3
* Required Libraries: `os`, `shutil`, `csv`
* Purpose: Moves errored jobs (e.g., SCF cycle exceeded) to an `errored/` folder.
* Input: `too_many_scf_list.csv` or similar
* Moves: `.sh`, `.out`, `.d12`, `.f9`
* Tip: Can create subfolders for different error types for easier bulk-fix workflows.

### 4. fixk.py

* Language: Python 3
* Required Libraries: `os`, `glob`
* Purpose: Automatically fixes problematic `SHRINK` lines in `.d12` files.
* Use Case: Apply to files caught by `shrink_error_list.csv`
* Behavior: Replaces the SHRINK k-point mesh with the smallest value found.


## Integration with Enhanced Queue Management

These scripts are **automatically integrated** with the enhanced queue manager system:

- **`enhanced_queue_manager.py`** uses the error classification logic from `updatelists2.py`
- **`error_recovery.py`** incorporates `fixk.py` functionality for automated SHRINK parameter fixes
- **`workflow_engine.py`** automatically triggers appropriate fixes based on error classifications

## Manual Workflow (Legacy Usage)

1. **Analyze Results**: Run `updatelists2.py` on a batch folder to classify all job statuses
2. **Organize Completed**: Use `check_completedV2.py` to move successful jobs to `done/` folder
3. **Organize Errors**: Use `check_erroredV2.py` to sort errored jobs by error type
4. **Apply Fixes**: Use `fixk.py` for SHRINK errors and other targeted fixes
5. **Extract Geometries**: Use `CRYSTALOptToD12.py` to extract optimized structures
6. **Continue Workflow**: Submit follow-up SP, BAND, or DOSS calculations

## Error Classification

The scripts recognize specific CRYSTAL error patterns:

- **Complete**: `ENDED - TOTAL CPU TIME` or `FINAL OPTIMIZED GEOMETRY`
- **SCF Convergence**: `TOO MANY CYCLES IN SCF`
- **Memory Issues**: `INSUFFICIENT MEMORY`, `ALLOCATION ERROR`
- **SHRINK Errors**: `SHRINK FACTORS LESS THAN`, `SHRINK VALUE TOO SMALL`
- **Geometry Issues**: `SMALL INTERATOMIC DISTANCE`, `ATOMS TOO CLOSE`
- **Potential Problems**: Warning patterns that may indicate issues

## Requirements

- **Python 3.x** with standard libraries (`os`, `glob`, `csv`, `shutil`)
- **CRYSTAL output files** (`.out`) for analysis
- **Associated input files** (`.d12`) for error fixing

## Notes

- Scripts are designed for **batch processing** of hundreds of calculations
- Error classifications are based on **CRYSTAL-specific output patterns**
- Integration with modern workflow management provides **automated error recovery**
- Manual usage is maintained for **specialized workflows** and **debugging**

For automated usage, see `enhanced_queue_manager.py` and `error_recovery.py` documentation.
