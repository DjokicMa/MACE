# MACE Installation Guide

## Quick Start

1. **Clone or download the MACE repository**
   ```bash
   git clone <repository-url>
   cd reorganization
   ```

2. **Run the setup script**
   ```bash
   # Interactive setup (recommended)
   python setup_mace.py --add-to-path
   
   # Or specify your shell directly
   python setup_mace.py --shell zsh --add-to-path
   python setup_mace.py --shell bash --add-to-path
   python setup_mace.py --shell both --add-to-path  # Configure both shells
   ```

3. **Apply the configuration**
   ```bash
   source ~/.zshrc   # For zsh users
   source ~/.bashrc  # For bash users
   # Or start a new terminal session
   ```

4. **Verify installation**
   ```bash
   echo $MACE_HOME  # Should show the repository path
   mace --help  # Should show MACE commands
   ```

## Detailed Installation

### Prerequisites

- Python 3.7 or higher
- SLURM workload manager (for HPC environments)
- CRYSTAL17/23 quantum chemistry software

### Step 1: Install Python Dependencies

```bash
# Core dependencies (required)
pip install numpy matplotlib ase spglib pyyaml pandas

# Recommended: accurate band structure k-paths for all crystal systems
pip install seekpath

# Optional dependencies
pip install scipy scikit-learn

# Verify installation
python -c "import numpy, matplotlib, ase, spglib, yaml, pandas; print('All dependencies installed successfully')"
python -c "import seekpath; print('seekpath available for accurate band paths')"
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
from every compute node):

```bash
# Match the modules the generated job scripts load (see mace/submission/submitcrystal23.sh)
module purge
module load CRYSTAL/23-intel-2023a Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0

# Many sites' Python-bundle already ships numpy/scipy/pandas/ase/spglib/matplotlib/pyyaml.
# Install whatever is missing into the module Python (e.g. seekpath):
pip install --user seekpath

# Verify in this exact module env (and ideally on a compute node via: srun --pty bash):
python -c "import numpy, scipy, pandas, ase, spglib, yaml; print('core OK')"
python -c "import seekpath; print('seekpath OK -> library-computed band k-paths')"
python "$MACE_HOME/mace/queue/manager.py" --callback-mode status_check   # read-only sanity check
```

Notes:
- **seekpath** is optional. Without it, BAND generation still works — it falls back
  to the built-in extended-Bravais k-path tables. Installing it (into the module
  Python) upgrades to live, geometry-computed paths; the SLURM callback picks it up
  automatically.
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
submitcrystal23.py --help
NewCifToD12.py --help
```

### Alternative: Manual Installation

If you prefer manual setup:

1. **Set environment variables in your shell config:**
   ```bash
   # Add to ~/.zshrc or ~/.bashrc
   export MACE_HOME="/path/to/mace/reorganization"
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

## Troubleshooting

### Common Issues

1. **"MACE_HOME not set"**
   - Run: `source ~/.zshrc` or `source ~/.bashrc`
   - Or start a new terminal

2. **"Module not found" errors**
   - Install missing dependencies: `pip install [module_name]`
   - Check Python version: `python --version` (needs 3.7+)

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
mace monitor --dashboard
mace analyze --extract-properties .
mace convert --from-cif *.cif

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