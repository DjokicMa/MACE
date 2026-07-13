# CRYSTAL Job Status Analysis and Error Handling

This folder contains CRYSTAL-specific utilities for automatically analyzing calculation results, classifying job completion status, and implementing targeted error fixes for common CRYSTAL calculation issues.

## Included Scripts

### 1. updatelists2.py

* Language: Python 3
* Required Libraries: `os`, `pandas`
* Purpose: Scans all `.out` files and categorizes job status automatically.
* Output: Generates multiple `.csv` files:

  * `complete_list.csv`
  * `completesp_list.csv`
  * `too_many_scf_list.csv`
  * `memory_list.csv`
  * `quota_list.csv`
  * `time_list.csv`
  * `shrink_error_list.csv`
  * `geometry_small_dist_list.csv`
  * `linear_basis_list.csv`
  * `potential_list.csv`
  * `unknown_list.csv`
  * `ongoing_list.csv`
* Logic: Uses CRYSTAL-specific error messages to classify jobs.

### 2. check\_completedV2.py

* Language: Python 3
* Required Libraries: `os`, `shutil`, `pandas`
* Purpose: Moves all successfully completed jobs to a `completed/` folder.
* Input: `complete_list.csv` or `completesp_list.csv`
* Moves: `.sh`, `.out`, `.d12`, `.f9` (matching job names)

### 3. check\_erroredV2.py

* Language: Python 3
* Required Libraries: `os`, `shutil`, `pandas`
* Purpose: Moves errored jobs (e.g., SCF cycle exceeded) to categorized subdirectories under `errored/`.
* Input: `too_many_scf_list.csv` or similar
* Moves: `.sh`, `.out`, `.d12`, `.f9`
* Files are sorted into one subfolder per error type for easier bulk-fix workflows.

### 4. fixk.py

* Language: Python 3
* Required Libraries: `os`, `pandas`
* Purpose: Automatically fixes problematic `SHRINK` lines in `.d12` files.
* Use Case: Apply to files caught by `shrink_error_list.csv`
* Behavior: Replaces the SHRINK k-point mesh with the smallest value found.


## Integration with Enhanced Queue Management

The MACE workflow system implements the same job-status checks:

- Error classification lives in `mace/queue/manager.py` and `mace/recovery/detector.py`
- **`mace/recovery/recovery.py`** incorporates `fixk.py` functionality for automated SHRINK parameter fixes

## Manual Workflow (Legacy Usage)

1. **Analyze Results**: Run `updatelists2.py` on a batch folder to classify all job statuses
2. **Organize Completed**: Use `check_completedV2.py` to move successful jobs to `completed/` folder
3. **Organize Errors**: Use `check_erroredV2.py` to sort errored jobs by error type
4. **Apply Fixes**: Use `fixk.py` for SHRINK errors and other targeted fixes
5. **Extract Geometries**: Use `CRYSTALOptToD12.py` to extract optimized structures
6. **Continue Workflow**: Submit follow-up SP, BAND, or DOSS calculations

## Error Classification

The scripts recognize specific CRYSTAL error patterns:

- **Complete (OPT)**: `OPT END`
- **Complete (SP)**: `TOTAL CPU TIME =` (with no `OPT END`)
- **SCF Convergence**: `TOO MANY CYCLES`
- **Memory Issues**: `out-of-memory handler`
- **Disk Quota**: `error during write`
- **Time Limit**: `DUE TO TIME LIMIT`
- **SHRINK Errors**: `ANISOTROPIC SHRINKING FACTOR`
- **Geometry Issues**: `**** NEIGHB ****` (small interatomic distances)
- **Linear Dependence**: `BASIS SET LINEARLY DEPENDENT`
- **Potential Problems**: crash patterns (`segmentation fault`, `bad termination of`, `srun: error:`, `slurmstepd: error:`)
- **Unknown / Ongoing**: any other line containing `error`; otherwise the job is treated as still running

## Requirements

- **Python 3.x** with `pandas` (plus standard libraries `os`, `shutil`)
- **CRYSTAL output files** (`.out`) for analysis
- **Associated input files** (`.d12`) for error fixing

## Notes

- Scripts are designed for **batch processing** of hundreds of calculations
- Error classifications are based on **CRYSTAL-specific output patterns**
- Integration with modern workflow management provides **automated error recovery**
- Manual usage is maintained for **specialized workflows** and **debugging**

For automated usage, see `mace/enhanced_queue_manager.py` and `mace/recovery/recovery.py`.
