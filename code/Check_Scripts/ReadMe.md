# Check Scripts

This folder contains Python scripts used for managing, validating, and updating simulation jobs based on output files. These scripts help ensure consistency, detect errors, and automate post-processing cleanup or relaunching.

---

## Scripts Overview

### `check_completed2.py`
**Purpose**: Identifies completed simulation jobs.

- Checks for required files: `POSCAR`, `OUTCAR`, `CONTCAR`
- Flags incomplete or corrupted runs

**Requirements**:
- Python 3.x
- Libraries: `os`, `glob`
- ```updatelist.py``` must exist beforehand
- `completesp_list.csv` or `complete_list.csv` must exist
- `done/` directory must be created beforehand

**I/O**:
- **Input**: Directory of simulation job folders
- **Output**: Console log of successful completions

---

### `check_errored2.py`
**Purpose**: Detects errored or non-converging jobs.

- Scans for known error strings or convergence issues in `OUTCAR`

**Requirements**:
- Python 3.x
- Libraries: `os`, `glob`

**I/O**:
- **Input**: Job output folders
- **Output**: Printed report of failures

---

### `fixk.py`
**Purpose**: Repairs faulty `KPOINTS` files.

- Edits k-point settings in-place
- Ensures valid formatting for restarts

**Requirements**:
- Python 3.x
- Libraries: `os`

**I/O**:
- **Input**: Job folders with malformed `KPOINTS`
- **Output**: Fixed `KPOINTS` file

---

### `get_optimized2.py`
**Purpose**: Extracts relaxed structure from completed jobs.

- Converts `CONTCAR` → `POSCAR`
- Prepares structures for next job stage

**Requirements**:
- Python 3.x
- Libraries: `os`, `shutil`

**I/O**:
- **Input**: Optimized job folders
- **Output**: Updated `POSCAR` files

---

### `updatelists2.py`
**Purpose**: Maintains job status tracking files.

- Updates lists that record which jobs are done/pending
- Automates batch job tracking

**Requirements**:
- Python 3.x
- Libraries: `os`

**I/O**:
- **Input**: Text-based tracking files
- **Output**: Updated job lists

---

## Usage

Ensure all scripts are in the base folder where your jobs are located. Then run:

```bash
python check_completed2.py
python check_errored2.py
python fixk.py
python get_optimized2.py
python updatelists2.py
