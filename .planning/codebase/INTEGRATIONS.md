# External Integrations

**Analysis Date:** 2026-06-13

## APIs & External Services

**HPC Job Scheduler (SLURM):**
- SLURM workload manager — job submission, queue monitoring, cancellation
  - Commands invoked: `sbatch` (submit), `squeue` (query), `scancel` (cancel)
  - Invocation: `subprocess.run(['squeue', ...])` in `mace/queue/manager.py` and `mace/queue/legacy_manager.py`; `os.system(f"sbatch {sh_script}")` in `mace/submission/crystal.py` and `mace/submission/properties.py`
  - Auth: inherited from HPC cluster user session (no explicit credentials)
  - Cluster: Michigan State University HPCC; account `mendoza_q` (hardcoded default in `mace/submission/submitcrystal23.sh`)

**CRYSTAL23 DFT Solver:**
- CRYSTAL23 quantum chemistry code — the core scientific calculation engine
  - Not called via Python API; invoked as MPI binary by SLURM jobs
  - Binary path: `$EBROOTCRYSTAL/bin/Pcrystal` (parallel SCF/optimization), `$EBROOTCRYSTAL/bin/Pproperties` (band structure, DOS, transport)
  - Input files: `.d12` (SCF/geometry), `.d3` (properties)
  - Output files: `.out` (text output), `fort.9` (wavefunction binary, renamed to `.f9`), `fort.25` (formatted property data, `.f25`)
  - Module: `CRYSTAL/23-intel-2023a` (loaded via HPC environment modules)
  - Integration point: `mace/submission/submitcrystal23.sh`, `mace/submission/submit_prop.sh`, `mace/submission/portable_slurm_generator.py`

## Data Storage

**Databases:**
- SQLite (embedded, no server)
  - File: `materials.db` (created in working directory per-project)
  - Client: raw `sqlite3` stdlib — no ORM; manual schema management
  - Access: `mace/database/materials.py` (`MaterialDatabase` class with thread-safe connection pooling via `threading.Lock`)
  - Contextual variant: `mace/database/materials_contextual.py` (`ContextualMaterialDatabase`)
  - Stores: material records, calculation states (pending/running/completed/failed), workflow step tracking, file records, job IDs, extracted properties
  - Backup: uses SQLite backup API (`mace/database/materials.py` line ~1229) for safe live backup

**File Storage:**
- Local filesystem only — all input files (`.d12`, `.d3`, `.cif`), output files (`.out`, `.f9`, `.f25`), and database reside on local/network disk
- HPC scratch: `$SCRATCH/crys23/<jobname>/` — SLURM scripts copy inputs to scratch, copy outputs back to submission directory
- No cloud storage (S3, GCS, Azure) integration

**Caching:**
- None — no Redis, Memcached, or in-memory caching layer
- File-based state: job status persisted in `materials.db`; completed job outputs read directly from filesystem

## Authentication & Identity

**Auth Provider:**
- None — no authentication system
- HPC access controlled entirely by cluster-level SSH/user accounts
- `MACE_HOME` env var used to locate installation; `USER`/`LOGNAME` env vars used when querying SLURM (`mace/queue/legacy_manager.py` line 113)

## Scientific Library Integrations

**ASE (Atomic Simulation Environment) — `ase>=3.22.0`:**
- CIF file parsing: `ase.io.read` used in `Crystal_d12/NewCifToD12.py` and `code/Plotting_Scripts/plottingCIFs.py`
- Structure manipulation: `ase.neighborlist.NeighborList`, `ase.data.chemical_symbols`, `ase.data.covalent_radii`
- Optional ASE database connector: `ase.db.connect` imported conditionally in `mace/database/materials.py`
- CIF visualization: `ase.spacegroup.crystal` in `Plotting/plottingCIFs.py`

**spglib — `spglib>=1.16.0`:**
- Space group symmetry analysis; conditional import in `Crystal_d12/CrystalOutToCif.py` and `Crystal_d12/NewCifToD12.py`
- Pattern: `try: import spglib; except ImportError: ...` (graceful fallback)
- Only available in Anaconda environment (`/home/marcus/anaconda3/`)

**seekpath (optional):**
- HPKOT k-path generation for band structure calculations (Hinuma et al., Comp. Mat. Sci. 128, 140, 2017)
- Referenced from: `Crystal_d3/seekpath_interface.py` (conditional import), `Crystal_d3/d3_kpoints.py` (conditional import)
- Commented out in `requirements.txt` — must be installed manually
- Only available in Anaconda environment
- Fallback: manual k-path definition when seekpath unavailable

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, Rollbar, or external error tracking
- Errors logged to stdout/stderr; recovery events written to `recovery_logs/` directory (configured in `mace/config/recovery_config.yaml`)

**Logs:**
- `print()` statements throughout — no structured logging framework (no `logging` module)
- Recovery logs: flat files in `recovery_logs/` directory, retained 30 days per config
- Failed inputs archived to `failed_inputs/` directory per recovery config

## CI/CD & Deployment

**Hosting:**
- Local HPC cluster installation; no container deployment
- Installation via `python setup_mace.py` which configures shell rc files

**CI Pipeline:**
- None — no GitHub Actions, Jenkins, or any CI/CD pipeline

## Environment Configuration

**Required env vars (HPC production):**
- `MACE_HOME` — repo root path; set by `activate_mace.sh` or user shell rc
- `SCRATCH` — HPC scratch filesystem root (cluster-provided)
- `SLURM_SUBMIT_DIR`, `SLURM_NTASKS` — SLURM-injected job environment variables (used inside batch scripts)

**Optional env vars:**
- `USER` or `LOGNAME` — used to filter `squeue` output to current user's jobs

**Secrets location:**
- None — no API keys, passwords, or secrets required

## Webhooks & Callbacks

**Incoming:**
- None — MACE is not a server and receives no inbound HTTP requests

**Outgoing:**
- None — no HTTP webhooks sent to external services
- SLURM job completion callback: SLURM scripts call back into `mace/queue/manager.py` at job end (via shell command in SLURM script tail) to trigger queue advancement — this is a local process call, not HTTP

## Data Export Formats

**Visualization output (via `mace/database/export/visualization.py`):**
- Vega-Lite JSON schema (`https://vega.github.io/schema/vega-lite/v5.json`) — referenced as inline spec, no live API call
- Plotly HTML (`https://cdn.plot.ly/plotly-latest.min.js`) — CDN reference embedded in generated HTML reports; requires internet to render in browser

**File formats read/written:**
- `.cif` — Crystallographic Information File (read via ASE, written by `Crystal_d12/CrystalOutToCif.py`)
- `.d12` — CRYSTAL23 SCF input format (written by `Crystal_d12/` modules)
- `.d3` — CRYSTAL23 properties input format (written by `Crystal_d3/` modules)
- `.out` — CRYSTAL23 plain-text output (parsed by `mace/utils/property_extractor.py`, `mace/recovery/detector.py`)
- `fort.9` / `.f9` — binary wavefunction file (copied, not parsed)
- `fort.25` / `.f25` — formatted band/DOS data (parsed by plotting scripts)
- `.json` — calculation configs (`Crystal_d12/example_configs/`, `Crystal_d3/example_configs/`)
- `.yaml` — recovery configuration (`mace/config/recovery_config.yaml`)
- `.db` — SQLite materials database

---

*Integration audit: 2026-06-13*
