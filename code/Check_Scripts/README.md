# Check\_Scripts Folder Documentation

This folder contains CRYSTAL-specific utilities for checking, classifying, fixing, and preparing outputs from geometry optimization or single-point CRYSTAL calculations.

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


## Suggested Workflow

1. Run `updatelists2.py` on a batch folder.
2. Use `check_completed2.py` and `check_errored2.py` to sort jobs.
3. Fix errors (e.g. using `fixk.py` for shrink errors).
4. Extract optimized `.d12` files with `get_optimized2.py` or `CRYSTALOptToD12.py`.
5. Use the cleaned outputs for follow-up SP or postprocessing.

For any questions, contact the maintainer or refer to the CRYSTAL documentation for error string meanings.
