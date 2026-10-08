# MACE Installation Guide

## Quick Start

1. **Clone or download the MACE repository**
   ```bash
   git clone https://github.com/DjokicMa/MACE.git
   cd MACE
   ```

2. **Install the Python packages** (Python 3.11 or newer)
   ```bash
   pip install -r requirements.txt
   ```
   On a cluster, also into the Python the job scripts load
   ([HPC / SLURM clusters](#hpc--slurm-clusters-important-install-into-the-module-python)).

3. **Run the setup script** (optional: it puts `mace` on your PATH;
   [no shell startup file is needed](#what-mace-needs))
   ```bash
   # Interactive setup (recommended)
   python setup_mace.py --add-to-path
   
   # Or specify your shell directly
   python setup_mace.py --shell zsh --add-to-path
   python setup_mace.py --shell bash --add-to-path
   python setup_mace.py --shell both --add-to-path  # Configure both shells
   ```

4. **Apply the configuration**
   ```bash
   source ~/.zshrc   # For zsh users
   source ~/.bashrc  # For bash users
   # Or start a new terminal session
   ```

5. **Verify installation**
   ```bash
   echo $MACE_HOME  # Should show the repository path
   mace --help  # Should show MACE commands
   ```

## What MACE needs

Everything below the first table is optional: a missing package or program
turns off only the feature that needs it, and that feature says what to
install.

| Needed for | What | How MACE finds it |
|---|---|---|
| Everything | Python 3.11 or newer with `requirements.txt` (numpy, matplotlib, ase, spglib, **seekpath 2.1+**, rich, PyYAML, pandas, plotly, scipy, kaleido 0.2.1) | the Python that runs `mace` |
| Calculations | CRYSTAL17/23 (`Pcrystal`, `Pproperties`) and an MPI runtime | site settings (`MACE_CRYSTAL_MODULE`, `[modules] crystal`, `[crystal] bin_dir`), else the built-in module when it loads, else the newest CRYSTAL 23.x module that has both programs |
| Submitting jobs | SLURM (`sbatch`, `squeue`, `sacct`) | PATH; `mace site` sets account, partition and the rest. `mace run` needs no SLURM |
| Job completion steps | the same `requirements.txt` in the Python the job scripts load | `[modules] python`, else the Python that ran `mace` |

| Optional feature | What | How MACE finds it |
|---|---|---|
| `mace preflight` | the serial `crystal` binary | `--crystal-bin`, `MACE_CRYSTAL_BIN`, `[crystal] bin_dir`, `$EBROOTCRYSTAL/bin`, PATH, then the job scripts' CRYSTAL module |
| `mace phonopy` | `pip install 'phonopy<5'` | the Python that runs `mace` |
| `mace plotting --cube` isosurfaces | `pip install scikit-image` | the Python that runs `mace` |
| GFN-FF in `mace preopt` and Clean | the xtb program, 6.7 or newer (LGPL, not shipped) | `MACE_XTB`, else `xtb` on PATH, else the newest `xtb` module 6.7 or newer (`MACE_XTB_MODULE` pins one, `none` turns the search off) |
| ML potentials in `mace preopt` and Clean | a separate Python environment with mace-torch or chgnet, and the model weights | `mace mlip setup --python PATH` (or `MACE_MLIP_PYTHON`); weights from `mace mlip fetch` in `$XDG_DATA_HOME/mace-tool/mlip/` (`MACE_MLIP_DIR`) |
| `mace cluster` | SLURM's `sinfo`, `squeue`, `sacct`, `scontrol`, `sshare`, `sstat`; a quota command if your site has one | PATH; `[quota]` in site.toml names a site's own tool |
| The MACE GUI | nothing beyond `requirements.txt` on the machine with the data; a current browser with WebGL2 on your computer, and ssh to reach a cluster | `mace gui --open`, or the link it prints |
| `mace gui connect` | Python 3 and an OpenSSH client on your computer (no MACE packages); `scripts/` has bash and PowerShell versions for a computer without MACE | PATH |
| The GUI's browser tests | `pip install -r requirements-gui-test.txt` (selenium, axe-selenium-python) and Firefox | only for running the tests |

**No shell startup file is needed.** `setup_mace.py --add-to-path` adds
`MACE_HOME` and PATH lines to `~/.bashrc` or `~/.zshrc` for convenience, but
MACE runs without them: call `<checkout>/mace_cli` directly, link it into a
folder on your PATH (`ln -s <checkout>/mace_cli ~/bin/mace`), or `source
activate_mace.sh` for one session. Job scripts pin the MACE install that
submitted them, so they do not depend on `MACE_HOME` either. Every new setting
of 1.1.5 lives in a file under `~/.config/mace/` or in an optional environment
variable.

**Files MACE keeps for you** (all optional, made on first use):
`~/.config/mace/site.toml` (site settings for job scripts), `gui.json` (the
GUI profile), `connect.toml` (`mace gui connect` profiles), `mlip.json` (the
ML environment) and `config.json` (the terminal theme); caches under
`~/.cache/mace/` (`modules.json`, `cluster/`, `gui/`, `preopt/`); the GUI's
File History in `~/.local/state/mace/gui/history/`.

### What changed since 1.1.4

| | 1.1.4 | 1.1.5 |
|---|---|---|
| seekpath | optional | **required** (2.1 or newer); `--static-kpath` or `MACE_STATIC_KPATH=1` writes the static path without it |
| Other required packages | numpy, matplotlib, ase, spglib, rich, PyYAML, pandas, plotly, scipy, kaleido | unchanged |
| Optional packages | - | phonopy (`mace phonopy`); selenium and axe-selenium-python (GUI browser tests only) |
| External programs | CRYSTAL, SLURM, MPI | the same, plus optional xtb (GFN-FF) and an optional ML Python environment |
| CRYSTAL module | the module written in the job-script templates | site settings, else the built-in module when it loads, else the newest CRYSTAL 23.x module with `Pcrystal` and `Pproperties` |
| Configuration files | `~/.config/mace/config.json` (theme) | also `site.toml`, `gui.json`, `connect.toml`, `mlip.json`, all optional |
| Shell startup file | `MACE_HOME` and PATH from `setup_mace.py` | not needed (see above); existing lines keep working |

New environment variables, all optional:

- Site settings: `MACE_SITE_CONFIG`, `MACE_SITE_GROUP_CONFIG`,
  `MACE_SLURM_ACCOUNT`, `MACE_SLURM_PARTITION`, `MACE_SLURM_QOS`,
  `MACE_SLURM_CONSTRAINT`, `MACE_SLURM_EXTRA`, `MACE_SLURM_NTASKS`,
  `MACE_SLURM_NODES`, `MACE_SLURM_CPUS_PER_TASK`, `MACE_SLURM_MEM`,
  `MACE_SLURM_MEM_PER_CPU`, `MACE_CRYSTAL_MODULE`, `MACE_CRYSTAL_BIN_DIR`,
  `MACE_MPI_LAUNCHER`, `MACE_MODULE_PURGE`, `MACE_SCRATCH_BASE`,
  `MACE_RUN_MAX_TIME` ([Using MACE on another cluster](#using-mace-on-another-cluster))
- Band paths: `MACE_STATIC_KPATH`
- Database: `MACE_DB_JOURNAL`, `MACE_DB_LOCKING` (CHANGELOG, "Upgrading")
- The GUI: `MACE_GUI_PORT`, `MACE_GUI_TOKEN`, `MACE_GUI_GATEWAY`,
  `MACE_GUI_ROOTS`, `MACE_GUI_GPU`, `MACE_GUI_HISTORY_DIR`,
  `MACE_GUI_SESSION_IDLE_HOURS`, `MACE_GUI_SESSION_MAX_DAYS`,
  `MACE_GUI_IDLE_STOP_HOURS`, `MACE_GUI_MANAGED_TIMEOUT`,
  `MACE_GUI_MANAGED_GRACE`, `MACE_GUI_MAX_WORKERS`, `MACE_GUI_CACHE_MB`,
  `MACE_GUI_STATE_DIR`, `MACE_GUI_PRINT_TOKEN` ([docs/gui.md](docs/gui.md))
- Starting geometries: `MACE_XTB`, `MACE_XTB_MODULE`, `MACE_MLIP_PYTHON`,
  `MACE_MLIP_DIR`, `MACE_MLIP_NO_DOWNLOAD` ([docs/preopt.md](docs/preopt.md#environment-variables))
- Charged systems: `MACE_FRAGMENT_CHARGES` (CHANGELOG)

## Detailed Installation

### Prerequisites

- Python 3.11 or newer (CI tests 3.11 and 3.12; the MSU HPCC module Python is 3.11)
- SLURM workload manager (for HPC environments)
- CRYSTAL17/23 quantum chemistry software

### Step 1: Install Python Dependencies

```bash
# Core dependencies (required; or: pip install -r requirements.txt)
pip install numpy matplotlib ase spglib seekpath pyyaml pandas plotly

# Recommended: themed terminal UI (MACE falls back to plain text without it)
pip install rich

# Optional: interactive cube/FREQ plotting engines (mace plotting --cube / --freq)
pip install scipy kaleido==0.2.1 scikit-image

# Verify installation
python -c "import numpy, matplotlib, ase, spglib, seekpath, yaml, pandas, plotly; print('All dependencies installed successfully')"
```

#### HPC / SLURM clusters (IMPORTANT: install into the *module* Python)

On an HPC cluster the job-completion callback baked into every generated `.sh`
(`python mace/queue/manager.py --callback-mode completion ...`) runs under the
**module-loaded Python on the compute node** — NOT whatever Python you used to
install MACE interactively (e.g. an Anaconda env). So the dependencies must be
visible to that module Python, or the callback can't track / scrape / recover /
progress workflows.

Install with the SAME modules loaded that the job scripts use, using `--user`
(lands in `~/.local`, which is on the shared home filesystem and therefore visible
from every compute node). Run these on a development or compute node: a
login gateway may not serve software modules (on MSU HPCC the gateway answers
`module load Python/3.11.3-GCCcore-12.3.0` with "could not be loaded"; the same
load works after `ssh dev-amd20`).

```bash
# Match the modules the generated job scripts load (`mace site` lists them;
# these are MSU HPCC's - on another cluster see "Using MACE on another cluster")
module purge
module load CRYSTAL/23-intel-2023a Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0

# Many sites' Python-bundle already ships numpy/scipy/pandas/ase/spglib/matplotlib/pyyaml.
# Install whatever is missing into the module Python (seekpath is not in the bundle):
pip install --user seekpath

# Verify in this exact module env (and ideally on a compute node via: srun --pty bash):
python -c "import numpy, scipy, pandas, ase, spglib, seekpath, yaml; print('core OK')"
python "$MACE_HOME/mace/queue/manager.py" --callback-mode status_check   # read-only sanity check
```

**The MACE GUI on a dev node.** `mace gui` runs under the same module Python
and needs nothing beyond `requirements.txt` (ase, spglib and plotly; a
feature whose package is missing shows the `pip install --user ...` line). It
listens on 127.0.0.1 of the node; your laptop reaches it through an ssh
tunnel, which `mace gui` prints. Choose a personal port on shared nodes
(`export MACE_GUI_PORT=8xxx`). From your own computer, `mace gui connect`
starts it on the node and opens it in your browser in one step. See
[docs/gui.md](docs/gui.md#first-run-from-your-computer), "First run from your
computer", for Linux, macOS, Windows, MobaXterm and VS Code.

On a development node (on MSU HPCC: `ssh dev-amd20`, `dev-amd24` or
`dev-intel18` from the gateway) the same module Python also runs `mace` itself, and `mace run` runs
small decks there directly, without SLURM - see DOCUMENTATION.md, "Run on a dev
node". The gateway has no CRYSTAL module and no `mpirun`, and `mace run` refuses
to start there.

Notes:
- **seekpath** (2.1 or newer) is required. It computes the HPKOT band k-path of a 3D cell from
  its structure (Hinuma et al., Comp. Mat. Sci. 128, 140 (2017)) and MACE writes
  it in CRYSTAL's reciprocal basis, with the lattice-dependent points of that
  cell. The built-in tables hold SeeK-path's numbers in SeeK-path's standard cell,
  which for many lattices is not CRYSTAL's (axis order, reduction, centering
  vectors), so a path written from them can visit or name other points. Without
  seekpath, a 3D SeeK-path BAND deck (opt2d3 format 4, a plan's SeeK-path BAND
  step) or SeeK-path phonon dispersion (opt2d12) stops with the install command;
  a workflow step fails the same way, since the SLURM callback generates its
  decks with the module Python. To write the table path anyway, pass
  `--static-kpath` to `mace opt2d3` / `mace opt2d12`, or set
  `MACE_STATIC_KPATH=1` (in a job's environment, e.g. `~/.bashrc`, for a
  workflow). Slabs and polymers do not need seekpath, and the label, k-point
  vector, literature and custom paths do not use it.
- **pyarrow** is NOT required (pandas runs without it); don't bother installing it.
- `--user` installs are keyed to the Python X.Y version (e.g. `~/.local/lib/python3.11`),
  so reinstall if you switch the Python module's minor version.

### Step 2: Configure Environment

The `setup_mace.py` script handles environment configuration:

#### Options:
- `--shell [bash|zsh|both]`: Specify which shell to configure
- `--add-to-path`: Add all MACE directories to PATH
- `--force`: Reconfigure even if already set up
- `--no-shell`: Skip shell configuration
- `--skip-deps`: Skip dependency checking

#### What it configures:
1. **MACE_HOME**: Environment variable pointing to the repository
2. **PATH** (with --add-to-path): Adds these directories:
   - `$MACE_HOME` (for mace_cli command)
   - `$MACE_HOME/mace` (new structure)
   - `$MACE_HOME/Crystal_d12`
   - `$MACE_HOME/Crystal_d3`
   - `$MACE_HOME/code/Check_Scripts`
   - `$MACE_HOME/code/Post_Processing_Scripts`
3. **Alias**: `mace` → `mace_cli` for convenient command access

### Step 3: Test Installation

```bash
# Test MACE command
mace --help

# Test workflow system
mace workflow --interactive

# Test direct script access (if PATH configured)
NewCifToD12.py --help
CRYSTALOptToD3.py --help
```

### Alternative: Manual Installation

If you prefer manual setup:

1. **Set environment variables in your shell config:**
   ```bash
   # Add to ~/.zshrc or ~/.bashrc
   export MACE_HOME="/path/to/MACE"
   export PATH="$MACE_HOME:$PATH"
   export PATH="$MACE_HOME/mace:$PATH"
   export PATH="$MACE_HOME/Crystal_d12:$PATH"
   export PATH="$MACE_HOME/Crystal_d3:$PATH"
   alias mace="mace_cli"
   ```

2. **Source the configuration:**
   ```bash
   source ~/.zshrc  # or ~/.bashrc
   ```

### Using the Activation Script

For temporary use without modifying shell config:
```bash
source activate_mace.sh
```

This sets up the environment for the current session only.

## Using MACE on another cluster

MACE submits jobs with SLURM only. PBS, Torque and LSF are planned for MACE
1.2 or later; until then, `mace run` runs decks on the node you are on.

MACE runs on any SLURM cluster. Its job-script templates carry built-in
values from the cluster it was developed on (MSU HPCC): a SLURM account,
Python modules, a CRYSTAL module and queue limits. With no setting of your
own, MACE checks each one against your cluster before it writes a job script,
and replaces what does not apply. What it detects on its own:

| Built-in value | On your cluster, with no setting |
|---|---|
| SLURM account (`#SBATCH -A`) | Written only if you are a member of it (`sacctmgr show assoc user=$USER`); otherwise left out, so SLURM charges your default account. MACE says once which. |
| CRYSTAL module | Used if it loads and has `Pcrystal`/`Pproperties`; otherwise the best `CRYSTAL/*` module of the same major version that does. |
| Python modules (completion step) | Loaded only if they load here; otherwise the job calls the Python that runs MACE (its full path). MACE says so once. |
| Queue limits (`--max-jobs 950 --reserve 50`) | Your MaxSubmitJobs (`scontrol show assoc_mgr`) minus a 5% reserve, when SLURM reports one. |
| Default walltime (7 days, 2 hours for properties) | Lowered to your partition's MaxTime when that is shorter. |
| Development-node names in messages, node-exclusion menus | MSU HPCC's only on MSU HPCC; elsewhere generic wording and your cluster's node types. |

The answers are cached for a day in `~/.cache/mace/modules.json`. Without
SLURM or a module system (a workstation), nothing is checked and the
built-in values are written as they are.

To set your own values once:

```bash
mace site --init                  # writes ~/.config/mace/site.toml, every line commented out
$EDITOR ~/.config/mace/site.toml  # uncomment and set what you need
mace site                         # what MACE will write, and where each value comes from
mace site --refresh               # check the cluster again (after you join an account, say)
mace site check                   # before your first real job: account, partition, job size,
                                  # CRYSTAL, MPI, Python, scratch, and sbatch --test-only
```

A complete `site.toml` (every key is optional; use your own names):

```toml
[slurm]
account   = "my_account"            # #SBATCH -A my_account
partition = "my_partition"          # #SBATCH -p my_partition ("a,b": SLURM picks one)
[modules]
crystal = "CRYSTAL/23-foss-2023a"   # or a list loaded in order: ["foss/2023a", "CRYSTAL/23-foss-2023a"]
crystal_minimum = "23"              # the fallback window (default: the major version of crystal)
crystal_maximum = "23"
python = ["Python/3.11"]            # the completion step's Python module(s); [] loads none
[queue]
max_jobs = 450                      # jobs MACE keeps queued
reserve  = 50                       # queue slots it leaves free for you
[local_run]
gateway_message = "This is a login node ({reason}): log in to a compute node and rerun mace run."
module_message  = "the {module} module cannot be loaded on {host}"
```

The environment overrides the file for one shell:

| Variable | Sets |
|---|---|
| `MACE_SLURM_ACCOUNT` | the account |
| `MACE_SLURM_PARTITION` | the partition (comma-separated list allowed) |
| `MACE_CRYSTAL_MODULE` | the CRYSTAL module, exactly (comma-separated to load several in order; `none` keeps the template's line) |
| `MACE_SITE_CONFIG` | another `site.toml` (`none` ignores it and every group file) |
| `MACE_SITE_GROUP_CONFIG` | the group files to read instead of `/etc/mace/site.toml` and `$XDG_CONFIG_DIRS/mace/site.toml` (`none`: no group file) |
| `MACE_SLURM_QOS`, `MACE_SLURM_CONSTRAINT`, `MACE_SLURM_EXTRA` | `[slurm] qos`, `constraint`, `extra` |
| `MACE_SLURM_NTASKS`, `MACE_SLURM_NODES`, `MACE_SLURM_CPUS_PER_TASK`, `MACE_SLURM_MEM`, `MACE_SLURM_MEM_PER_CPU` | `[resources]`, for both job kinds |
| `MACE_MPI_LAUNCHER` | `[mpi] launcher` |
| `MACE_CRYSTAL_BIN_DIR` | `[crystal] bin_dir` |
| `MACE_SCRATCH_BASE` | `[paths] scratch` |
| `MACE_MODULE_PURGE` | `[modules] purge` (0 or 1) |
| `MACE_RUN_MAX_TIME` | `[local_run] max_time` |

docs/cli.md, "Site settings", describes every key and check.

**The completion step's Python.** Every job ends by running MACE's queue
manager with `python`. On a cluster where the built-in Python modules do not
load, that is the Python you ran `mace` with, so install MACE's requirements
into it (`pip install -r requirements.txt`), and make sure the compute nodes
can see it (a home or project filesystem). To load your cluster's Python
module instead, set `[modules] python`.

**The rest of the job script.** These keys change the other parts of the
templates; each is optional and changes only its own lines:

| What | Built-in | Key (environment) |
|---|---|---|
| Job size | `--ntasks=32`, `-N 1`, `--cpus-per-task=1`, `--mem-per-cpu=5G` (properties: 28 tasks, `--mem=80G`) | `[resources] ntasks, nodes, cpus_per_task, mem_per_cpu` or `mem`; `[resources.properties]` for Pproperties jobs (`MACE_SLURM_NTASKS`, ...) |
| Starting CRYSTAL | `I_MPI_HYDRA_BOOTSTRAP=ssh mpirun -n $SLURM_NTASKS` | `[mpi] launcher = "srun"`, `srun_mpi`, `env` (`MACE_MPI_LAUNCHER`) |
| CRYSTAL itself | the CRYSTAL module's `Pcrystal`/`Pproperties` | `[crystal] bin_dir`, `scf`, `properties` (`MACE_CRYSTAL_BIN_DIR`) |
| `module purge` | yes | `[modules] purge = false` (`MACE_MODULE_PURGE=0`) |
| More `#SBATCH` lines | none | `[slurm] qos`, `constraint`, `extra` (`MACE_SLURM_QOS`, ...) |
| Scratch | `$SCRATCH/crys23`, then `/mnt/scratch/$USER`, the submit directory, `$TMPDIR` | `[paths] scratch`, tried right after `$SCRATCH` (`MACE_SCRATCH_BASE`) |
| `mace run` time limit | `ulimit -t` minus 5 minutes; 1h55m on MSU HPCC | `[local_run] max_time` (`MACE_RUN_MAX_TIME`) |

`mace submit -i`, `--walltime` and the workflow planner offer the
`[resources]` values as their defaults; what you type is written as typed.
MACE uses copies of a template (a workflow's `workflow_scripts/`, your own
`<job>.sh`) as written.

- **MPI.** `launcher = "mpirun"` (the default) runs Intel MPI builds as
  today. With an OpenMPI build it normally works when OpenMPI was built with
  SLURM support (it reads the allocation itself and ignores
  `I_MPI_HYDRA_BOOTSTRAP`); if it does not, use `srun`: OpenMPI with PMIx,
  `srun_mpi = "pmix"`; Intel MPI, `srun_mpi = "pmi2"` and
  `env = ["I_MPI_PMI_LIBRARY=<your libpmi2.so>"]`.
- **Scratch.** Set `[paths] scratch` only to a directory every compute node
  shares (often /scratch, /work or /gpfs, but on some clusters
  /scratch is a node's local disk). MACE never guesses it; `mace site check`
  suggests what it finds.
- **Several nodes.** `[resources] nodes` above 1 needs shared scratch and an
  MPI launch that spans nodes.
- **MSU-like node names.** MSU HPCC's own defaults (the 1h55m `mace run`
  limit, its node-exclusion menus) also apply on another cluster whose node
  names look like MSU's (`skl-12`, `dev-amd20`); set the values in site.toml
  there.

A site.toml for a generic cluster (OpenMPI with srun, a group install):

```toml
[slurm]
account    = "my_lab"
partition  = "compute"
qos        = "normal"
constraint = "epyc"
extra      = ["--exclusive"]
[resources]
ntasks      = 64
nodes       = 1
mem_per_cpu = "3900M"
[resources.properties]
ntasks = 32
mem    = "120G"
[mpi]
launcher = "srun"
srun_mpi = "pmix"
[modules]
crystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]
python  = ["Python/3.11.5"]
# purge = false     # Cray PE: keep the base PrgEnv modules
[crystal]
# bin_dir = "/project/my_lab/crystal23/bin"   # instead of the module
[paths]
scratch = "/scratch/$USER"     # only if all nodes share it
[local_run]
max_time = "4h"
```

**Settings for a whole cluster or lab.** Admins: `/etc/mace/site.toml`
(owned by root). A lab: a file in a directory only its owner can write,
named by `MACE_SITE_GROUP_CONFIG` in the group's shell profile; MACE refuses
a group file that others can modify (`chmod 644 site.toml; chmod 755 <dir>`).
Each person's `~/.config/mace/site.toml` overrides it per key, and `mace site`
shows where every value comes from (and who owns the group file). Everyone
who uses a group file runs what it names (modules, `bin_dir`), so only use
one you trust.

Before your first real job: `mace site check`.

## Troubleshooting

### Common Issues

1. **"MACE_HOME not set"**
   - Run: `source ~/.zshrc` or `source ~/.bashrc`
   - Or start a new terminal

2. **"Module not found" errors**
   - Install missing dependencies: `pip install [module_name]`
   - Check Python version: `python --version` (needs 3.11 or newer)

3. **Scripts not found in PATH**
   - Ensure you used `--add-to-path` during setup
   - Or use full paths: `python $MACE_HOME/mace/submission/crystal.py`

4. **Permission denied**
   - Make scripts executable: `chmod +x mace_cli`

### Verification Commands

```bash
# Check environment
echo $MACE_HOME
which python
python --version

# Check MACE installation
python -c "import sys; print('\n'.join(sys.path))"
ls -la $MACE_HOME/mace/

# Test imports
python -c "from mace.database.materials import MaterialDatabase; print('✓ Database module')"
python -c "from mace.workflow.planner import WorkflowPlanner; print('✓ Workflow module')"
```

## Updating MACE

To update your installation:

1. **Pull latest changes** (if using git)
   ```bash
   cd $MACE_HOME
   git pull
   ```

2. **Re-run setup** (if configuration changed)
   ```bash
   python setup_mace.py --force
   ```

3. **Update dependencies**
   ```bash
   pip install --upgrade numpy matplotlib ase spglib pyyaml pandas
   ```

## Usage After Installation

After completing installation, you can use MACE commands globally:
```bash
# Using the unified mace command (if PATH configured)
mace workflow --interactive
mace submit calculation.d12
mace monitor
mace analyze --extract-properties .
mace convert --cif_dir ./cifs
mace plotting --all
mace gui --dir . --open

# Or with full path
python mace_cli workflow --interactive
python mace_cli submit calculation.d12

# Or use scripts directly if added to PATH
NewCifToD12.py --cif_dir ./cifs
CRYSTALOptToD12.py --input material.out
```

## Uninstalling

To remove MACE configuration:

1. Remove environment variables from your shell config
2. Remove the lines between `# MACE Environment` markers in ~/.zshrc or ~/.bashrc
3. Delete the repository directory (optional)