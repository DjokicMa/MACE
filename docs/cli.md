# MACE command line

Every `mace` command by group, then a [reference](#command-reference) for
each command with all of its options (kept consistent with `mace <command>
--help` by a test), then two guides: [site settings](#site-settings-mace-site-sitetoml)
for job scripts on any cluster, and [mace cluster](#mace-cluster).

## Commands by group

`mace --help` lists every command in eight groups; the tables below follow
the same order. `mace <command> --help` gives each command's options, and
`mace help GROUP` prints one group. On a new cluster, start with `mace site`.

### Automated workflows

Plan the steps once; MACE submits each next step. `mace help workflows` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace workflow` | Plan and run multi-step workflows - OPT, SP, BAND/DOSS, FREQ... | [Workflow manager](../DOCUMENTATION.md#workflow-manager-phase-3) |
| `mace engine` | Advance workflows: process finished runs, submit the next step | [Workflow manager](../DOCUMENTATION.md#workflow-manager-phase-3) |
| `mace manager` | Tracking queue manager - materials.db records, submission cycles | [Material tracking](../DOCUMENTATION.md#material-tracking-system) |
| `mace recover` | Prepare fixes for failed tracked runs (SHRINK, memory, SCF) | [Error handling](../DOCUMENTATION.md#error-handling-and-recovery) |

### Monitoring

Your calculations and the cluster. `mace help monitoring` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace monitor` | What MACE tracks (materials.db) - live dashboard, or --status | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace completion` | Outputs in a folder, no database - finished, failed, running | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace cluster` | The cluster from SLURM - nodes, limits, quotas, a job's progress | [mace cluster](#mace-cluster) |

**Which command shows my jobs?** `mace monitor` shows the calculations MACE
tracks (workflows, `submit --track`); `mace completion` any CRYSTAL outputs in
a folder, tracked or not; `mace cluster job ID` one SLURM job (state, progress,
ETA, walltime risk); `mace manager --status` the SLURM queue. You seldom run
`engine` or `manager` yourself: when a tracked job ends, its script runs the
manager's completion step, which records the result, recovers a failure and
submits the workflow's next step through the engine.

### Manual calculations

Build decks, check them, submit or run them. `mace help manual` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace convert` | CIF to D12 - supports all CRYSTAL basis sets and functionals | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace opt2d12` | D12 decks from a finished run - SP, FREQ and more, settings kept | [One template, many structures](../DOCUMENTATION.md#one-d12-template-for-many-structures-mace-opt2d12) |
| `mace opt2d3` | D3 property decks from a finished run - BAND, DOSS, TRANSPORT... | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace opt2cif` | CIF from a finished run - any dimensionality or calculation type | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace preflight` | Check D12 decks with CRYSTAL's TESTPDIM before you spend a job | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace submit` | Submit D12/D3 decks to SLURM as written (--array, --track, ...) | [Job arrays](../DOCUMENTATION.md#job-arrays-mace-submit---array) |
| `mace run` | Run decks on this node now, no SLURM - small SP, BAND/DOSS... | [Run on a dev node](../DOCUMENTATION.md#run-on-a-dev-node-mace-run) |

### Results

Extract, analyze, plot and query. `mace help results` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace analyze` | Outputs into materials.db (gaps, energies); --convergence FILE | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace plotting` | Plots - bands, DOS, structures, cube volumetrics, FREQ, IR/Raman | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace database` | Query materials.db - search, compare, export (CSV, Excel, ...) | [Material tracking](../DOCUMENTATION.md#material-tracking-system) |

### Special systems and structure tools

`mace help special` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace preopt` | A better starting geometry for a CIF/D12 (springs, GFN-FF, ML) | [preopt.md](preopt.md) |
| `mace mlip` | ML potentials for preopt - list, set up the environment, fetch | [preopt.md](preopt.md#ml-potentials) |
| `mace defects` | Point-defect deck sets (charge states, calibration) and analysis | [defects/README.md](defects/README.md) |
| `mace ions` | Charged systems - CHEMOD/DOPING/SPIN for ions and charged cells | [MACE commands](../DOCUMENTATION.md#mace-commands-primary-interface) |
| `mace phonopy` | Phonons with phonopy - displaced supercells, then bands and DOS | [Phonons with phonopy](../DOCUMENTATION.md#phonons-with-phonopy-mace-phonopy) |

### GUI

`mace help gui` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace gui` | The MACE GUI in your browser - structures, plots, database, setup; `--slurm`: run it as a SLURM batch job on a compute node ([details](gui.md#run-it-as-a-slurm-job)); `connect`: start it on a cluster node from your computer (with `--slurm` too); `login-url`: a new one-time sign-in link for a running GUI; `diagnostics`: the server's part of a bug report (no GUI needed) | [gui.md](gui.md) |

### Setup and information

Run `mace site` first on a new cluster. `mace help setup` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace site` | Show the site settings for job scripts (--init, check) | [Site settings](#site-settings-mace-site-sitetoml) |
| `mace help` | This list, one group (mace help monitoring) or one command | `mace help --help` |
| `mace version` | Show the MACE version (same as --version) | `mace version --help` |
| `mace credits` | Show the development team (same as --credits) | `mace credits --help` |

### Aliases and deprecated

Old names; they keep working. `mace help aliases` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace status` | Same as 'mace monitor --status' | `mace status --help` |
| `mace queue` | Deprecated - use 'mace manager' | `mace queue --help` |

## Command reference

One section per command, in the order of `mace --help`. Each lists every
option `mace <command> --help` shows (a test keeps the two in step); the
help adds the full wording and more examples. Paths in the examples are
placeholders.

### mace workflow

Plan a multi-step calculation (OPT, SP, BAND/DOSS, FREQ, ...) once; MACE
submits each next step when the one before it finishes.

```
mace workflow [--interactive | --quick-start | --execute FILE | --status | --show-templates] [options]
```

| Option | What it does |
|---|---|
| `--interactive` | Plan step by step in the interactive planner |
| `--quick-start` | Start from a predefined template, with `--cif-dir` or `--d12-dir` |
| `--execute FILE` | Run a saved plan (`workflow_configs/workflow_plan_*.json`) |
| `--status` | Show the status of the workflows here |
| `--show-templates` | List the templates |
| `--cif-dir DIR` | CIF files for `--quick-start` |
| `--d12-dir DIR` | D12 decks for `--quick-start` |
| `--workflow TYPE` | Template: basic_opt, opt_sp, opt_sp_freq, full_electronic (default), double_opt, complete, transport_analysis, charge_analysis, combined_analysis |
| `--work-dir DIR` | Working directory (default: here) |
| `--db-path PATH` | Database (default: materials.db) |
| `--max-jobs N` | Concurrent jobs at most (default 200) |
| `--array` | Submit the first step as SLURM job arrays (later steps go one by one as before) |
| `--max-running N` | With `--array`: at most N tasks of each array at once |

```bash
mace workflow --interactive
mace workflow --quick-start --cif-dir ./cifs --workflow full_electronic
mace workflow --quick-start --d12-dir ./decks --workflow opt_sp --array --max-running 10
```

More: [Workflow manager](../DOCUMENTATION.md#workflow-manager-phase-3),
[Job arrays](../DOCUMENTATION.md#job-arrays-mace-submit---array).

### mace engine

Advance workflows by hand: process finished runs and submit their next
steps. A tracked job's script normally does this for you.

```
mace engine [--action status|process|workflow] [--material-id ID] [--db PATH] [--work-dir DIR]
```

| Option | What it does |
|---|---|
| `--action ACTION` | `status` (default; needs `--material-id`), `process` (finished runs, then next steps) or `workflow` (per-material summary) |
| `--material-id ID` | The material for `--action status` |
| `--db PATH` | Database (default: materials.db) |
| `--work-dir DIR` | Base working directory (default: here) |

There is no `--dry-run` or `--max-submit`.

```bash
mace engine --action process
mace engine --action status --material-id diamond
```

### mace manager

The tracking queue manager: materials.db records and submission cycles.
Job scripts run its completion step when a tracked job ends.

```
mace manager [--status] [--d12-dir DIR] [--max-jobs N] [--reserve N] [--organize] [--callback-mode MODE]
```

| Option | What it does |
|---|---|
| `--status` | The SLURM queue: running, pending, free slots |
| `--d12-dir DIR` | Decks to manage (default: here) |
| `--max-jobs N` | Concurrent jobs at most (default 250) |
| `--reserve N` | Slots kept free (default 30) |
| `--organize` | Copy inputs into `<calc_type>/<material_id>/` (default: submit in place) |
| `--callback-mode MODE` | One cycle of `full_check` (default), `submit_new`, `status_check`, `early_failure` or `completion` |
| `--dry-run` | Not implemented: ignored with a warning, and a real submission follows. Use `--status` |

```bash
mace manager --status
mace manager --callback-mode completion
```

More: [Material tracking](../DOCUMENTATION.md#material-tracking-system).

### mace recover

Prepares fixed inputs and job scripts for failed tracked runs (SHRINK,
memory, SCF, ...); it submits nothing. Submit them with `mace submit` (the
action ends with the exact command for each one), or let the queue manager
(`mace manager`) resubmit tracked runs. A recovery that was prepared but
never submitted does not count toward `max_retries`, and running `mace
recover` again reuses it instead of preparing another copy.

```
mace recover [--action recover|stats|config] [--max-recoveries N] [--config FILE] [--db PATH] [--create-config]
```

| Option | What it does |
|---|---|
| `--action ACTION` | `recover` (default), `stats` or `config` |
| `--max-recoveries N` | At most N recoveries (default 10) |
| `--config FILE` | Recovery rules (default: `recovery_config.yaml` here, else MACE's own) |
| `--db PATH` | Database |
| `--create-config` | Write a default `recovery_config.yaml` |

```bash
mace recover
mace recover --action stats
```

More: [Error handling](../DOCUMENTATION.md#error-handling-and-recovery).

### mace monitor

What MACE tracks in materials.db: a live dashboard by default (there is no
`--dashboard` flag), or a status summary.

```
mace monitor [--status [--summary | --detailed]] [--interval N] [--action ACTION] [options]
```

| Option | What it does |
|---|---|
| `--status` | A quick status summary instead of the dashboard |
| `--summary` | With `--status`: materials and calculation totals only |
| `--detailed` | With `--status`: calculation types and recent activity |
| `--interval N` | Dashboard refresh in seconds (default 30) |
| `--action ACTION` | `dashboard` (default), `status`, `report` or `stats` |
| `--output FILE` | The file for `--action report` |
| `--workflow-id ID` | One workflow only |
| `--db-path PATH` | Database (default: materials.db) |

`--material-id` is not honored in `--status` mode.

```bash
mace monitor
mace monitor --status --detailed
mace monitor --action report --output report.json
```

### mace completion

Which CRYSTAL outputs in a folder finished, failed or are still running; no
database needed.

```
mace completion [-d DIR] [-r] [--detailed] [--move-completed | --organize] [options]
```

| Option | What it does |
|---|---|
| `-d`, `--directory DIR` | Folder to scan (default: here) |
| `-r`, `--recursive` | Scan subfolders too (workflow folders) |
| `--detailed` | List the files of each category |
| `--move-completed` | Move finished runs into `completed/` |
| `--organize` | Sort every file into folders by status and calculation type |
| `--output-dir DIR` | Where `--move-completed`/`--organize` put them |
| `--extensions EXT ...` | The file suffixes to move (default: inputs, outputs, .f9/.f25, .DAT and cube files) |
| `--remove-zombie-jobs` | Cancel your SLURM jobs whose output already shows a failure |

```bash
mace completion -d ./cifs
mace completion --recursive --detailed
```

### mace cluster (summary)

The cluster from SLURM, read-only and rate-limited: partitions, nodes, your
limits, quotas, where your files are, one job's progress and walltime risk,
runtime estimates and the least-loaded development node.

```
mace cluster [overview|nodes|fit|reservations|quota|files|limits|job ID|estimate DECK|devnodes] [options]
```

| Option | Subcommand | What it does |
|---|---|---|
| `--json`, `--refresh` | all | Machine-readable output; ask SLURM again (at most every 15 s) |
| `--partition P`, `--feature F`, `--free`, `--all-partitions`, `--group-by feature\|partition` | nodes, fit | Which nodes to show and how to group them |
| `--cpus N`, `--nodes N`, `--mem-per-cpu SIZE`, `--mem SIZE`, `--gpus [TYPE:]N`, `--constraint EXPR`, `--time D-HH:MM:SS` | fit | The request to fit |
| `--predict`, `--script JOB.sh` | fit | Ask `sbatch --test-only` for a start time (of a request, or of a job script) |
| `--rediscover` | quota | Look up the site's quota command again |
| `--top N`, `--max-seconds S`, `--sizes top\|all\|off` | files | Scan size and depth |
| `--check-submit N` (with `--cpus C`) | limits | Warn before adding N jobs of C cores |
| `--follow`, `--live` | job | Refresh every 60 s; one `sstat` call for live memory and CPU |
| `--cores N`, `--history DIR`, `--db materials.db` | estimate | The run to estimate and where to look for similar ones |
| `--pick` | devnodes | Print only the node name |

```bash
mace cluster
mace cluster fit --cpus 32 --mem 64G --time 1-00:00:00 --predict
mace cluster job 12345678 --follow
```

More: [mace cluster](#mace-cluster), the full guide below.

### mace convert

CIF files to D12 decks, with any CRYSTAL basis set and functional, asked
interactively or loaded from saved answers.

```
mace convert [--cif_dir DIR] [--output_dir DIR] [--batch] [--save_options] [--options_file FILE] [charged-system options]
```

| Option | What it does |
|---|---|
| `--cif_dir DIR` | The folder of .cif files (default: here; not file arguments) |
| `--output_dir DIR` | Where the decks go (default: the CIF folder) |
| `--batch` | Use saved answers; ask nothing |
| `--save_options` | Also save the answers (`cif2d12_options.json`) |
| `--options_file FILE` | The answers file to save or load |
| `--ion-charges auto` | Find known ions (Li+, FSI-, ...) by connectivity and give them their formal charges |
| `--net-charge Q` | Net charge (default: the sum of the ions found) |
| `--site-states JSON` | Per-atom shell charges, as JSON or a JSON file |
| `--fragment-bond-scale S` | Bonded when closer than S x the covalent radii sum (default 1.15) |
| `--chemod auto\|yes\|no` | Write CHEMOD (default auto) |
| `--spinlock N` | n_alpha - n_beta for an odd electron count |
| `--spinlock-cycles N` | SCF cycles the SPINLOCK holds |

```bash
mace convert --cif_dir cifs/ --output_dir decks/ --save_options
mace convert --cif_dir more_cifs/ --batch
```

More: [Crystal_d12/README.md](../Crystal_d12/README.md).

### mace opt2d12

New D12 decks (SP, OPT, FREQ, ...) from finished runs, with the parent's
settings kept; one saved template can be applied to many structures.

```
mace opt2d12 --out-file OUT [OUT ...] | --directory DIR [options]
```

| Option | What it does |
|---|---|
| `--out-file OUT ...` | Output(s) to start from; several or a glob form one batch |
| `--d12-file FILE` | The parent deck for a single output (default: the same name beside it) |
| `--directory DIR` | Every output in DIR (default: here) |
| `--output-dir DIR` | Where the new decks go (default: beside the input) |
| `--shared-settings` | One set of answers for every file |
| `--save-options` | Save the answers |
| `--options-file FILE` | The answers file to save or load |
| `--config-file FILE` | Apply a saved configuration to every file (each keeps its own geometry and symmetry) |
| `-y`, `--yes` | Apply `--config-file` without asking |
| `--non-interactive` | Ask nothing; needs `--calc-type` |
| `--calc-type SP\|OPT\|FREQ` | The calculation type for `--non-interactive` |
| `--opt-type TYPE` | FULLOPTG, ATOMONLY, CELLONLY, ITATOCEL, CVOLOPT or INTREDUN for a non-interactive OPT |
| `--static-kpath` | Without seekpath, write the static HPKOT path for a 3D phonon dispersion (same as `MACE_STATIC_KPATH=1`) |
| `--origin-setting S` | Origin setting: `auto` (default), `0 0 1`, ... |
| `--ion-charges`, `--net-charge`, `--site-states`, `--chemod`, `--fragment-bond-scale`, `--spinlock`, `--spinlock-cycles` | Charged systems, recomputed for the new geometry (as for `mace convert`) |

```bash
mace opt2d12 --out-file optimized.out --calc-type SP --non-interactive
mace opt2d12 --directory opts --config-file sp_template.json --yes
```

More: [One template, many structures](../DOCUMENTATION.md#one-d12-template-for-many-structures-mace-opt2d12).

### mace opt2d3

D3 property decks (BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL) from a
finished run.

```
mace opt2d3 --input OUT [--calc-type TYPE] [options]   |   mace opt2d3 --batch [options]
```

| Option | What it does |
|---|---|
| `-i`, `--input OUT` | The CRYSTAL output (.out or .log) |
| `-t`, `--calc-type TYPE` (also `--calc_type`) | BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL or CHARGE+POTENTIAL |
| `-b`, `--batch` | Every .out here |
| `-s`, `--shared-settings` | One set of answers for the batch |
| `-c`, `--config-file FILE` | Load a saved D3 configuration |
| `--save-config` | Save the configuration after the questions |
| `--options-file FILE` | The file name for `--save-config` |
| `--list-configs` | List the saved configurations |
| `-o`, `--output-dir DIR` | Where the decks go (default: beside the input) |
| `--static-kpath` | Without seekpath, write the static HPKOT path for a 3D BAND (same as `MACE_STATIC_KPATH=1`) |

```bash
mace opt2d3 -i optimized.out -t BAND
mace opt2d3 --batch --calc-type DOSS
```

More: [Crystal_d3/README.md](../Crystal_d3/README.md).

### mace opt2cif

CIF files from finished runs of any dimensionality (OPT, SP or FREQ).

```
mace opt2cif [TARGETS ...] [options]
```

| Option | What it does |
|---|---|
| `--output-dir DIR` | Where the CIFs go (default: beside the input) |
| `--force-dimension 3d\|2d\|1d\|0d` | Override the detected dimensionality |
| `--space-group SG` | Override the space group |
| `--preserve-symmetry` | Write the unique atoms with the space group |
| `--vacuum-thickness A` | Vacuum for 2D/1D (default 20 Å); the minimum box for 0D |
| `--precision N` | Coordinate digits (default 6) |
| `--include-metadata` | Calculation details as CIF comments |
| `--dry-run` | Show what would be converted |
| `--verbose` | More output |
| `--verify` | Check the symmetrized CIF against P1 with spglib |
| `--verify-tolerance T` | Tolerance for `--verify` (default 0.01, fractional) |

```bash
mace opt2cif . --output-dir cifs/
mace opt2cif . --preserve-symmetry --verify
```

### mace preflight

Let CRYSTAL read a copy of each D12 deck (TESTPDIM) before you spend a job
on it. Your files are never changed.

```
mace preflight [FILES/DIRS ...] [options]
```

| Option | What it does |
|---|---|
| `-j`, `--jobs N` | Decks checked at once (default 1) |
| `-t`, `--timeout SECS` | Per deck (default 120) |
| `-k`, `--keep DIR` | Keep each run directory in DIR |
| `-r`, `--recursive` | Look in subfolders |
| `--static-only` | Never run CRYSTAL; structure checks only |
| `--crystal-bin PATH` | The serial `crystal` binary (else `MACE_CRYSTAL_BIN`, `$EBROOTCRYSTAL/bin/crystal`, PATH, or the job scripts' CRYSTAL module) |

Exit codes: 0 accepted, 1 a deck rejected, 2 a deck could not be checked.

```bash
mace preflight decks/ -r -j 4 --keep preflight_failures/
```

### mace submit

Submit D12/D3 decks to SLURM exactly as written; nothing follows unless you
ask for it.

```
mace submit [FILE_OR_DIR ...] [options]
```

| Option | What it does |
|---|---|
| `-i`, `--interactive` | Choose cores, memory and walltime |
| `--nosubmit` | Write the job scripts only |
| `--overwrite-sh` | Write new job scripts instead of reusing existing `.sh` files |
| `--walltime TIME` | Walltime for these jobs (e.g. `2:00:00`) |
| `--track` | Record the jobs in materials.db (monitor, analysis, recovery) |
| `--progress TEMPLATE` | Plan the steps that follow and run them as each job ends (a template from `mace workflow --show-templates`, or `interactive`); implies `--track` |
| `--array` | Decks with identical resource requests share one SLURM job array |
| `--max-running N` | With `--array`: at most N tasks of each array at once |
| `--exclude-type TYPE` | Exclude nodes named TYPE-<number> |
| `--exclude-nodes LIST` | Exclude these nodes (`node-042,node-050` or `node-[042-050]`) |
| `--exclude-amd20` | MSU HPCC only: exclude the AMD20 nodes |
| `--exclude-mendoza` | MSU HPCC only: exclude the Mendoza group's nodes |

The exclusion options are mutually exclusive. The job scripts follow your
[site settings](#site-settings-mace-site-sitetoml).

```bash
mace submit my_calculations/
mace submit --track --walltime 4:00:00 mat_opt.d12
mace submit --array --max-running 10 my_calculations/
```

More: [Job arrays](../DOCUMENTATION.md#job-arrays-mace-submit---array).

### mace run

Run decks now on the node you are on (an interactive or development node),
without SLURM, staged exactly as a SLURM job.

```
mace run [FILES/DIRS ...] [-n N] [--timeout T] [--track] [--dry-run]
```

| Option | What it does |
|---|---|
| `-n`, `--cores N` | MPI ranks (default 4, 1 to 16) |
| `--timeout T` | Wall time per deck: `90s`, `30m`, `1h` or `H:MM:SS` (default 30m) |
| `--track` | Record the runs in ./materials.db |
| `--dry-run` | Print the order, scratch folders and .f9 hand-offs; run nothing |

Every .d12 runs first, then each .d3 after the SCF that feeds it. Refused on
a login node and inside a SLURM job. Exit codes: 0 all completed, 1 a deck
failed, 2 refused, 124 timeout.

```bash
mace run dia_sp.d12 dia_band.d3 dia_doss.d3
mace run --dry-run dia/
```

More: [Run on a dev node](../DOCUMENTATION.md#run-on-a-dev-node-mace-run).

### mace analyze

Extract properties from outputs into materials.db, build a database from a
folder, or show how one run converged.

```
mace analyze --extract-properties DIR [options]
mace analyze --convergence FILE.out [--json]
```

| Option | What it does |
|---|---|
| `--extract-properties DIR` | The folder of outputs |
| `--output-file FILE` | One output instead of a folder |
| `--material-id ID` | Filter by, or override, the material id |
| `--calc-id ID` | Override the calculation id |
| `--calc-type TYPE` | Filter by calculation type (OPT, SP, FREQ, BAND, DOSS) |
| `--filter-material ID` | Only this material |
| `--filter-type TYPE` | Only this calculation type |
| `--db-path PATH` | Database (default: materials.db) |
| `--fragments` | Per-fragment Mulliken charges for neutral runs too |
| `--register` | First record the folder's completed calculations |
| `--dry-run` | Write nothing; list each output as new, changed, unchanged, ... |
| `--json` | With `--dry-run` or `--convergence`: JSON |
| `--skip-unchanged` | Only new and changed outputs of tracked calculations |
| `--backup` | Back up the database into `<db dir>/backups/` first |
| `--keep-backups N` | With `--backup`: keep N (default 3) |
| `--progress-json` | One JSON line per file on stdout |
| `--convergence FILE` | Per optimization point: SCF cycles, energy change, gradients and displacements against their thresholds |

```bash
mace analyze --extract-properties runs/ --register --skip-unchanged --backup
mace analyze --convergence mat1_opt.out
```

### mace plotting

Plots from CRYSTAL outputs: band structures, DOS, structures, cube
volumetrics, FREQ normal modes and IR/Raman spectra. With no mode flag it
asks.

```
mace plotting [MODE] [FILE ...] [-d DIR] [-o DIR] [options]
```

| Option | What it does |
|---|---|
| `--band`, `--dos`, `--structure`, `--cube`, `--freq` (or `--vibmodes`), `--ir`, `--raman`, `--spectra`, `--all` | What to plot (`--spectra`: IR and Raman; `--all`: everything found) |
| `-d`, `--directory DIR` | Where to look (default: here) |
| `-o`, `--output DIR` | Where the plots go |
| `--format FMT` | Cube/FREQ output: html (default), png, svg or pdf |
| `--e-lower E`, `--e-upper E` | Band energy window in eV (default -8, 8) |
| `--alpha A` | Band stacking transparency (default 0.3) |
| `--no-gaps` | Remove the gaps at k-path discontinuities |
| `--projection TYPE` | DOS: both, total, orbital or tm_orb |
| `--no-composition` | DOS: no VB/CB composition labels |
| `--supercell NX NY NZ` | Structure: supercell (default 2 2 2) |
| `--bond-cutoff A` | Structure: bond cutoff in Å (default 1.9) |
| `--color-by-coord` | Structure: color atoms by coordination |
| `--density`, `--esp`, `--spin` | Cube: the quantity (each implies `--cube`) |
| `--diff A B` | Cube: density difference A - B |
| `--iso V[,V2...]` | Cube: isovalue(s) |
| `--view iso\|slice\|slice-all` | Cube: view (default iso) |
| `--slice AXIS POS` | Cube: one 2D slice |
| `--colorscale NAME`, `--no-atoms`, `--bonds`, `--publication` | Cube: color scale and display |
| `--engine-args "..."` | Cube: extra flags for the cube engine |
| `--list-modes` | FREQ: the mode table only |
| `--mode N` | FREQ: this mode (default: the lowest real one) |
| `--all-modes` | FREQ: every mode in one HTML |
| `--gif` | FREQ: an animated GIF |
| `--amplitude F`, `--normalize`, `--static` | FREQ: displacement scale, normalized vectors, no animation |
| `--raman-mode MODE` | Raman: total, par_perp or all (default) |
| `--average` | Average conformer (`-confN_`) spectra per material |
| `--ir-column N` | IR intensity column (default 2, absorbance) |

```bash
mace plotting
mace plotting --band -d ./band_calcs -o ./plots
mace plotting --freq mol_freq.out --mode 7 --gif
```

### mace database

Query materials.db: statistics, searches, properties, exports, comparisons
and checks.

```
mace database [--action ACTION] [options]
```

| Option | What it does |
|---|---|
| `--action ACTION` | stats (default), query, properties, export, compare, missing, correlate, distribution, workflow, validate, history, interactive or aggregate |
| `--material-id ID` | One material |
| `--filter "EXPR"` | Property filter for query/export, e.g. `"band_gap > 3.0"` (repeatable) |
| `--logic AND\|OR` | How filters combine (default AND) |
| `--format FORMAT` | Export format: CSV, JSON, Excel, LaTeX or HTML |
| `--output FILE` | Export file |
| `--materials LIST` | Material ids to compare |
| `--material-ids LIST` | Material ids for missing/workflow/validate |
| `--category NAME` | properties: one category (e.g. electronic) |
| `--detail-level LEVEL` | missing: how much to show (e.g. full) |
| `--properties LIST` | correlate/distribution: these properties |
| `--bins N` | distribution: histogram bins |
| `--workflow NAME`, `--detailed` | workflow: one template, more detail |
| `--fix`, `--check-integrity` | validate: repair what it can; check the database itself |
| `--operation recent` | history: recent changes |

```bash
mace database --action query --filter "band_gap > 3.0"
mace database --action export --format excel --output results.xlsx
```

More: [Material tracking](../DOCUMENTATION.md#material-tracking-system).

### mace preopt

A better starting geometry for a CIF or D12 (springs, GFN-FF through xtb,
or an ML potential). CRYSTAL still optimizes it; only the geometry is
written.

```
mace preopt INPUT.cif|INPUT.d12 [--method springs|gfnff|mlip:ID|auto] [-o OUTDIR] [options]
```

| Option | What it does |
|---|---|
| `--method M` | springs (default), gfnff, mlip:ID or auto |
| `-o`, `--output-dir D` | Where to write (default: beside the input) |
| `--force` | Overwrite `<stem>_preopt.cif/.json` |
| `--relax-cell` | Relax the cell too |
| `--no-symmetry` | Do not keep the space group |
| `--symprec TOL` | Symmetry tolerance in Å (default 1e-3) |
| `--symmetrize TOL` | Symmetrize first (it can raise the space group) |
| `--charge Q` | Net charge (GFN-FF) |
| `--uhf N` | Unpaired electrons, recorded (GFN-FF ignores it) |
| `--fmax F` | Force threshold in eV/Å |
| `--steps N` | Step limit |
| `--time-limit S` | Seconds |
| `--threads K` | ML worker threads |
| `--device cpu\|cuda:N` | ML device (default cpu; a GPU only when you name it) |
| `--no-fallback` | Write nothing when the method fails |
| `--json` | Print the provenance JSON |

```bash
mace preopt LiFSI_box.cif --method gfnff --charge 0
mace preopt MgO.cif --method auto -o preopt/
```

More: [preopt.md](preopt.md).

### mace mlip

The ML potentials `mace preopt` and the GUI's Clean can use: list them,
record the Python environment they run in, fetch their weights.

```
mace mlip list [--json] [--recheck]
mace mlip setup [--python PATH]
mace mlip fetch ID [--yes] [--dir DIR]
```

| Option | What it does |
|---|---|
| `--json`, `--recheck` | list: JSON; probe the environment again |
| `--python PATH` | setup: probe and record this interpreter (without it, print how to make one) |
| `--yes`, `--dir DIR` | fetch: do not ask; where the weights go |

`mace mlip fetch` is the only MACE command that downloads anything.

```bash
mace mlip setup --python ~/mlenv/bin/python
mace mlip fetch acesuit-mace-mpa-0-medium
```

More: [preopt.md](preopt.md#ml-potentials).

### mace defects

Point defects in a periodic cell: a pristine supercell and one single-point
deck per charge state from your SP host deck, then formation energies.

```
mace defects build HOST.d12 (--vacancy EL | --ghost EL | --substitute OLD=NEW | --insert EL@x,y,z) (--supercell | --supercon | --supercell-matrix | --sizes) [options]
mace defects build --resume DIR
mace defects potc DIR [--jobs a,b] [--planar ZM ZP]
mace defects analyze DIR [options]
```

| Option | What it does |
|---|---|
| `--vacancy EL`, `--ghost EL`, `--substitute OLD=NEW`, `--insert EL@x,y,z` | build: the defect (ATOMREMO, GHOSTS, ATOMSUBS, ATOMINSE) |
| `--cart` | build: `--insert` coordinates in Cartesian Å |
| `--site-label N`, `--site-frac x,y,z` | build: which site |
| `--supercell NxNxN`, `--supercon NxNxN`, `--supercell-matrix 'a ... i'`, `--sizes 3,4,5` | build: the supercell (`--sizes`: 2D size series) |
| `--charges 0,1,2` | build: charge states (`--charges=-1,0,1` when the list starts with a minus) |
| `--spin-state q=NSPIN`, `--spinlock-cycles N`, `--host-spin refuse\|keep` | build: spin per charge state, SPINLOCK cycles, the host's own spin records |
| `--calibrate [LADDER]`, `--calibrate-on defect\|pristine`, `--calibrate-endpoints` | build: fractional-DOPING calibration decks (3D) |
| `--pristine-out FILE`, `--shrink '...'`, `--name PREFIX`, `-o`, `--output DIR`, `--force`, `--resume DIR` | build: pristine output, SHRINK, job names, output folder, overwrite, finish a waiting set |
| `--jobs a,b`, `--planar ZM ZP` | potc: only these jobs; 2D planar-averaged potential |
| `--mu EL=val`, `--mu-file FILE`, `--vbm val` | analyze: chemical potentials, VBM override |
| `--eps VALUE`, `--eps-from FILE`, `--eps-static` | analyze: dielectric constant for the image-charge term |
| `--offset spheropole\|calibration`, `--apply-alignment` | analyze: charged-cell offset; POTC alignment (opt-in) |
| `--json FILE`, `--csv FILE` | analyze: write the results |

```bash
mace defects build MgO_host.d12 --vacancy O --supercell 2x2x2 --charges 0,1,2 -o MgO_VO
mace submit MgO_VO
mace defects analyze MgO_VO --mu O=-75.1Ha
```

More: [defects/README.md](defects/README.md).

### mace ions

Add CHEMOD / DOPING / CHARGED (and SPIN + SPINLOCK for an odd electron
count) to decks of ion clusters or charged cells. Ions are found by
connectivity.

```
mace ions DECKS ... (-o DIR | --in-place) [options]
```

| Option | What it does |
|---|---|
| `--ion-charges auto` | Find known ions and give them their formal charges |
| `--net-charge Q` | Net charge (default: the sum of the ions) |
| `--site-states JSON` | Per-atom shell charges (needs `--net-charge`) |
| `--fragment-bond-scale S` | Bond cutoff scale (default 1.15) |
| `--chemod auto\|yes\|no` | Write CHEMOD (default auto) |
| `--spinlock N`, `--spinlock-cycles N` | SPINLOCK for an odd electron count |
| `--parent-out OUT` | An output with the same basis, to check the shell counts |
| `--replace` | Remove the deck's own CHEMOD/CHARGED/DOPING first |
| `-o`, `--output-dir DIR` | Write the new decks here |
| `--in-place` | Overwrite the decks |
| `--non-interactive` | Never ask |

```bash
mace ions d12s/li_ec2.d12 --ion-charges auto -o charged/
mace ions cell.d12 --net-charge -1 -o charged/
```

More: [Crystal_d12/README.md](../Crystal_d12/README.md) ("Charged systems
and ion clusters").

### mace phonopy

Finite-displacement phonons through phonopy (optional: `pip install
'phonopy<5'`): displaced supercells from a finished 3D OPT or SP parent,
then force sets, band structure, DOS and thermal properties.

```
mace phonopy generate PARENT.out --dim "N N N" [options]
mace phonopy collect DIR [options]
```

| Option | What it does |
|---|---|
| `--d12 FILE` | generate: the parent deck (default: same stem) |
| `--dim "N N N"` | generate: the supercell (or 9 integers) |
| `--output-dir DIR`, `--force` | generate: output folder (default `phonopy_<stem>_<dim>/`); reuse a non-empty one |
| `--amplitude A` | generate: displacement in Å (default 0.01) |
| `--pm auto\|yes\|no` | generate: plus/minus displacements |
| `--toldee N` | generate: TOLDEE (raised only, at least 10) |
| `--cry-sym` | generate: each displaced cell's symmetry in its .ext |
| `--keep-shrink` | generate: keep the parent's SHRINK |
| `--check` | collect: only check the outputs |
| `--band auto\|"q ..."\|none`, `--band-labels`, `--band-points N` | collect: the band path (default: SeeK-path) |
| `--mesh 100.0\|"a b c"` | collect: the DOS mesh |
| `--tmin`, `--tmax`, `--tstep` | collect: temperature range (K) |
| `--nac-from FREQ.out` | collect: Born charges and dielectric tensor from a Gamma FREQCALC with INTENS + INTCPHF |
| `--plot` | collect: write band_dos.pdf |

Submit the displaced decks with plain `mace submit DIR --walltime <printed>`,
never with `--track` or `--progress`.

```bash
mace phonopy generate mat_opt.out --dim "2 2 2"
mace submit phonopy_mat_opt_222 --walltime 2:00:00
mace phonopy collect phonopy_mat_opt_222 --plot
```

More: [Phonons with phonopy](../DOCUMENTATION.md#phonons-with-phonopy-mace-phonopy).

### mace gui

The MACE GUI in your browser. [gui.md](gui.md) is its guide; this lists the
options.

```
mace gui [--dir DIR] [--port P] [--open] [options]
mace gui --slurm [--dir DIR] [--time T] [--cpus N] [--mem M] [options]
mace gui connect [PROFILE] [--gateway G] [--node N] [--dir D] [options]
mace gui login-url [--port P] [--open]
mace gui diagnostics [--dir DIR] [--json]
```

`mace gui`:

| Option | What it does |
|---|---|
| `-d`, `--dir DIR` | The folder to serve (default: here) |
| `-p`, `--port P` | Port (default `MACE_GUI_PORT`, else 8765; 0 picks a free one) |
| `--open` | Open a browser on this machine |
| `--gpu N` | With `--open`: the NVIDIA GPU the browser renders on (default 0) |
| `--host H` | Interface to bind (default 127.0.0.1) |
| `--insecure-network` | Needed for any `--host` but loopback |
| `--allow-host NAME` | Also accept this Host header |
| `--token T` | This session token (`MACE_GUI_TOKEN` does it privately) |
| `--gateway G` | The login host named in the tunnel hint |
| `--root DIR` | A folder File > Open Folder may switch within (repeatable; `none` turns it off) |
| `--managed`, `--http-control`, `--ready-file FILE` | Used by `mace gui connect` and `mace gui --slurm` |
| `--slurm` | Run the GUI as a small SLURM batch job and forward a port to it |
| `-v`, `--verbose` | Log requests |

`mace gui --slurm` adds `--time T` (default 4:00:00), `--cpus N` (2),
`--mem M` (4G), `--partition`, `--account`, `--qos`, `--constraint`,
`--gpus`, `--sbatch-arg ARG` (one more sbatch option as `--name=value`,
repeatable, e.g. `--sbatch-arg=--reservation=r1`), `--setup LINE` and `--dry-run`; see
[Run it as a SLURM job](gui.md#run-it-as-a-slurm-job).

`mace gui connect` (from your computer):

| Option | What it does |
|---|---|
| `PROFILE` | A saved profile (default: the `[defaults]` profile) |
| `--gateway G`, `--jump HOSTS`, `--node N`, `--hop auto\|jump\|nested` | How to reach the node |
| `--dir DIR`, `--mace CMD`, `--setup LINE`, `--conda ENV` | What runs on the node |
| `--local-port P`, `--ssh-option OPT`, `--root DIR`, `--no-open` | This computer's side |
| `--slurm` and its `--time`, `--cpus`, `--mem`, `--partition`, `--account`, `--qos`, `--constraint`, `--gpus`, `--sbatch-arg` | Run the GUI as a SLURM job |
| `--save [NAME]`, `--list`, `--forget NAME` | Saved profiles |
| `--dry-run` | Print the ssh command(s) |
| `--export-script bash\|powershell`, `-o`, `--output FILE` | A standalone connect script for a computer without MACE |

`mace gui login-url` prints a new one-time sign-in link (`--port`,
`--open`); for a server started by `mace gui connect`, run it on the computer
where `mace gui connect` runs (the link uses the forwarded port); `mace gui
diagnostics` prints the server's part of a bug report
without paths or names (`--dir`, `--json`).

```bash
mace gui --dir . --open
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs
mace gui connect --slurm --gateway me@hpcc.example.edu --node dev-node-01
```

More: [gui.md](gui.md), [First run from your computer](gui.md#first-run-from-your-computer).

### mace site

The site settings MACE writes into job scripts (account, partition, CRYSTAL
module, MPI launcher, scratch, ...), and a check of them on this cluster.

```
mace site [show|check] [--json] [--refresh] [--init] [--partition P] [--account A] [--kind scf|prop|both] [--no-sbatch]
```

| Option | What it does |
|---|---|
| `show` (default), `check` | Show the settings and where each comes from; or test them here, with `sbatch --test-only` |
| `--json` | JSON |
| `--refresh` | Forget the cached module and SLURM answers |
| `--init` | Write `~/.config/mace/site.toml` with every setting commented out |
| `--partition P`, `--account A` | check: test these instead |
| `--kind scf\|prop\|both` | check: which job scripts (default both) |
| `--no-sbatch` | check: do not ask sbatch |

More: [Site settings](#site-settings-mace-site-sitetoml).

### mace help

`mace help` prints the grouped command list, `mace help GROUP` one group
(workflows, monitoring, manual, results, special, gui, setup, aliases), and
`mace help COMMAND` that command's `--help`.

### mace version

Prints the MACE version, as `mace --version`.

### mace credits

Prints the development team, as `mace --credits`.

### mace status

The same as `mace monitor --status`; its arguments (e.g. `--detailed`) are
passed on.

### mace queue

Deprecated; use `mace manager`.

## Site settings (`mace site`, site.toml)

MACE submits jobs with SLURM only. PBS, Torque and LSF are planned for MACE
1.2 or later; until then, `mace run` runs decks on the node you are on.

MACE's job-script templates carry built-in values from the cluster MACE was
developed on (MSU HPCC): a SLURM account, a CRYSTAL module, Python modules
and queue limits. Site settings replace those values in every job script MACE
generates on your cluster, without editing the templates. With no settings,
MACE checks the built-in values against the cluster it runs on (see "What
MACE detects" below); where they all apply, or where nothing can be checked
(no SLURM, no module system), every script keeps its bytes. See also
INSTALLATION.md, "Using MACE on another cluster".

Where the values come from, per key, first match wins:

1. the environment: `MACE_SLURM_ACCOUNT`, `MACE_SLURM_PARTITION`,
   `MACE_CRYSTAL_MODULE` and the others in the table below;
2. `~/.config/mace/site.toml` (`$XDG_CONFIG_HOME/mace/site.toml`;
   `MACE_SITE_CONFIG=<path>` reads another file, `MACE_SITE_CONFIG=none`
   ignores it and every group file);
3. group files, for everyone on a cluster or in a lab: `/etc/mace/site.toml`,
   then `<dir>/mace/site.toml` for each directory of `$XDG_CONFIG_DIRS`
   (default `/etc/xdg`; the first directory wins), or instead the files that
   `MACE_SITE_GROUP_CONFIG` names (a colon-separated list, the first wins;
   `MACE_SITE_GROUP_CONFIG=none` ignores group files);
4. MACE's built-in values, checked against this cluster.

A group file is read only when nobody but its owner can change it (neither
the file nor its directory writable by group or others) and, in `/etc/mace`
or `$XDG_CONFIG_DIRS`, only when it belongs to root or to you; otherwise MACE
stops with exit code 2 and says why (`chmod 644 site.toml; chmod 755 <dir>`).
A file you name in `MACE_SITE_GROUP_CONFIG` may belong to someone else (a
lab's PI): naming it is your decision to trust it, and `mace site` shows its
owner. Everyone who uses a group file runs what it names (modules,
`bin_dir`). Your own `site.toml` that others can write gets a warning only.
Reading settings needs Python 3.11 or newer: on an older Python with a
settings file present, MACE stops with exit code 2 rather than use the
built-in values.

```toml
[slurm]
account   = "my_account"            # #SBATCH -A my_account
partition = "general"               # #SBATCH -p general ("a,b": SLURM picks one)
qos        = "normal"               # #SBATCH --qos=normal
constraint = "epyc"                 # #SBATCH --constraint=epyc (node features: sinfo -o "%P %f")
extra      = ["--exclusive"]        # more #SBATCH options, at most 16 (see below)
[resources]
ntasks        = 64                  # #SBATCH --ntasks=64 (built-in: 32; properties 28)
nodes         = 1                   # #SBATCH -N 1 (above 1: see "Several nodes" below)
cpus_per_task = 1                   # #SBATCH --cpus-per-task=1
mem_per_cpu   = "3900M"             # #SBATCH --mem-per-cpu=3900M; or mem = "250G" (#SBATCH --mem), not both
[resources.properties]
ntasks = 32                         # the same keys, for Pproperties jobs only
mem    = "120G"
[mpi]
launcher = "srun"                   # "mpirun" (default: the template's own line) or "srun"
srun_mpi = "pmix"                   # srun --mpi=pmix (pmix, pmi2, pmix_v3, pmix_v4, none)
env = ["OMPI_MCA_btl=^openib"]      # exported before CRYSTAL starts
[modules]
crystal = "CRYSTAL/23-foss-2023a"   # (or crystal/23.1) or an ordered list for Lmod:
# crystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]
crystal_minimum = "23"              # the fallback window (default: the major of crystal)
crystal_maximum = "23"
python = ["Python/3.11"]            # [] loads none
purge = false                       # keep the modules loaded at login (Cray PE); default true
[crystal]
bin_dir    = "/opt/crystal23/bin"   # CRYSTAL without a module: Pcrystal, Pproperties are here
scf        = "Pcrystal"             # or "MPPcrystal"
properties = "Pproperties"
[paths]
scratch = "/scratch/$USER"          # tried right after $SCRATCH ($USER, $HOME, $SLURM_JOB_ID allowed)
[queue]
max_jobs = 450                      # jobs MACE keeps queued (--max-jobs of the completion step)
reserve  = 50                       # queue slots it leaves free for you (--reserve)
[local_run]
max_time = "4h"                     # longest `mace run --timeout` (90s, 30m, 4h or H:MM:SS)
gateway_message = "This is a login node ({reason}): log in to a compute node and rerun mace run."
module_message  = "the {module} module cannot be loaded on {host}"
```

Each key changes only its own line(s) of the job script; with none of them
set, the scripts keep their bytes:

| Key | Environment | In the job script |
|---|---|---|
| `[slurm] qos`, `constraint` | `MACE_SLURM_QOS`, `MACE_SLURM_CONSTRAINT` | `#SBATCH --qos=`, `#SBATCH --constraint=` after `-A`/`-p` |
| `[slurm] extra` | `MACE_SLURM_EXTRA` (comma-separated) | one `#SBATCH` line each; `--ntasks`, `--mem`, `--time`, `--account`, `--partition`, `--qos`, `--constraint`, `--job-name`, `--output`, `--error`, `--array`, `--exclude`, `--chdir` and their abbreviations (`--part=`) are refused (MACE writes them; the message names the key to use) |
| `[resources] ntasks`, `nodes`, `cpus_per_task`, `mem`, `mem_per_cpu` | `MACE_SLURM_NTASKS`, `MACE_SLURM_NODES`, `MACE_SLURM_CPUS_PER_TASK`, `MACE_SLURM_MEM`, `MACE_SLURM_MEM_PER_CPU` (both job kinds; one memory form clears the other) | `--ntasks`, `-N`, `--cpus-per-task`, the memory line in its form; `cpus_per_task` above 1 also exports `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` |
| `[mpi] launcher`, `srun_mpi`, `env` | `MACE_MPI_LAUNCHER` | the line that starts CRYSTAL; `srun` keeps `mpirun` for `mace run` |
| `[modules] purge` | `MACE_MODULE_PURGE` (0 or 1) | `module purge` becomes a comment |
| `[crystal] bin_dir`, `scf`, `properties` | `MACE_CRYSTAL_BIN_DIR` | the executables; with no CRYSTAL module of yours, the module line becomes a comment (and `EBROOTCRYSTAL` is set when `bin_dir` ends in `/bin`) |
| `[paths] scratch` | `MACE_SCRATCH_BASE` | one scratch candidate after `$SCRATCH` |
| `[local_run] max_time` | `MACE_RUN_MAX_TIME` | (no script change: the longest `mace run --timeout`) |

`mace submit -i`, `--walltime` and the workflow planner write cores and
memory themselves: the `[resources]` values are their defaults (the prompt
shows the site's memory form, "Memory per CPU" or "Memory per node
(total)"), and a value you type is written as typed.

Values that land on a line bash runs are data only: `[mpi] env` items are
`NAME=value` with a value of letters, digits and `. _ / : , = + @ % ^ -`
(no spaces, quotes or shell characters); names that change how the job or
MACE runs (`PATH`, `LD_*`, `PYTHON*`, `BASH_*`, `MACE_*`, `SLURM_*`, `HOME`,
`TMPDIR`, ...) are refused; the launcher is `mpirun` or `srun` (your own
launcher command is planned for MACE 1.2); `bin_dir` and `scratch` are paths
without `..`, quotes or `$(...)` (`scratch` may use `$USER`, `$HOME` and
`$SLURM_JOB_ID`).

**MPI.** `launcher = "mpirun"` (the default) starts Intel MPI builds as
before. An OpenMPI build normally works with it too when OpenMPI was built
with SLURM support (it reads the allocation itself and ignores
`I_MPI_HYDRA_BOOTSTRAP`); if it does not, use `srun`: OpenMPI with PMIx,
`srun_mpi = "pmix"`; Intel MPI, `srun_mpi = "pmi2"` and
`env = ["I_MPI_PMI_LIBRARY=<your libpmi2.so>"]`. `mace site check` asks
`srun --mpi=list` whether the type exists.

**Scratch.** Set `[paths] scratch` only to a directory every compute node
shares (often /scratch, /work or /gpfs, but on some clusters
/scratch is a node's own disk). MACE never guesses it; `mace site check`
and `mace site --init` suggest what they find, as a comment.

**Several nodes.** `[resources] nodes` above 1 needs a scratch directory all
nodes share and an MPI launch that spans nodes (`$TMPDIR` is node-local);
`mace site` warns about it.

Every key is optional. Values end up inside the job scripts, so they are
checked before anything is generated: an account or partition of letters,
digits, `.`, `_`, `-` (a partition list separated by commas), module names
without spaces or quotes, a CRYSTAL list whose last entry is the `CRYSTAL/`
module, messages on one line with only `{reason}`, `{module}` and `{host}`.
An invalid value (in the file or the environment) stops `mace submit`, `mace
run` and the planner with exit code 2 and the source of the value; nothing
is written. An unknown key is only a warning.

**What MACE detects** when a value is not set (each check is cached for a
day in `~/.cache/mace/modules.json`; `mace site --refresh` checks again):

- **Account.** The built-in account is written only when you are a member of
  it (`sacctmgr show assoc user=$USER`), or when that cannot be checked (no
  `sacctmgr`: a workstation). Otherwise the `#SBATCH -A` line is left out, so
  SLURM charges your default account, and MACE says so once.
- **Python modules.** The built-in Python modules are loaded only where they
  load (a module probe), or where there is no module system. Otherwise the
  job's completion step runs the Python that runs MACE (its full path, as
  `python() { "/path/to/python" "$@"; }` in the job script), and MACE says so
  once. Install MACE's requirements into that Python, or set `[modules]
  python` to your cluster's Python module.
- **Queue limits.** `--max-jobs 950 --reserve 50` (a 1000-job MaxSubmitJobs)
  becomes your association's MaxSubmitJobs minus a 5% reserve when `scontrol
  show assoc_mgr` reports one; `[queue]` sets them.
- **Walltime.** A template's default walltime (7 days for decks, 2 hours for
  properties) is lowered to the MaxTime of your partition (`[slurm]
  partition`, else the cluster's default partition) when that is shorter.
  A walltime you ask for (`--walltime`, `-i`) is written as given.
- **On MSU HPCC** (the cluster the built-in values come from) the Python
  modules and the walltime are written as they are: its login gateway serves
  no modules, and a submit plugin routes its jobs by walltime.
- **Messages and node exclusion.** MSU HPCC's development-node names and its
  node-exclusion choices (AMD20 nodes, the Mendoza group's nodes) appear only
  on MSU HPCC (its host names, its domain); elsewhere the menus offer "by
  node type" and "a node list" of your cluster.

**The CRYSTAL module.**

- `MACE_CRYSTAL_MODULE=<module>` pins it exactly (several, comma-separated,
  are loaded in order); `MACE_CRYSTAL_MODULE=none` keeps the template's line.
- Otherwise the preferred module (`[modules] crystal`, else the built-in) is
  used when it loads here and provides `Pcrystal` and `Pproperties` (under
  `$EBROOTCRYSTAL/bin`, or on `PATH` in one folder for modules that set no
  root; MACE then sets `$EBROOTCRYSTAL` in the script).
- Only when it does not, MACE uses the best other `CRYSTAL/*` module (or,
  when there is none, a lower-case `crystal/*` one) of the same major
  version (plain builds before MPP and binary builds, then the same
  toolchain, then the newest), says so once, and writes a `# MACE:`
  comment into the script. It never switches to another CRYSTAL major version
  on its own: set `crystal_maximum` (or `crystal`) to allow one.
- Without a module system (a workstation) nothing is probed. The choice is
  cached for 7 days in `~/.cache/mace/modules.json`, checked against the
  module's binaries on every use.
- A completion callback (inside a SLURM job) never probes: a follow-up loads
  the module its parent job loaded, so one chain keeps one CRYSTAL build.
  Tracked calculations record it as `crystal_module` in their settings.
- `mace run` and `mace preflight` probe the same module the job script loads.

**Workflows** store the account (or that none is written), the module, and
the Python and queue values when they differ from the built-in ones, when
they are planned; later steps use the stored values. Plans made before site settings (including
1.1.4's, which store `mendoza_q`) keep theirs: plan again to use new ones.
Copies of a template (`workflow_scripts/`, your own `<job>.sh`) are used as
written.

**`mace site`** shows the file it reads and whether it exists, the group
files with their owners, each value with its source (`env MACE_...`,
`site.toml: <path>`, `group: <path>` or `built-in`), the job size (and,
where `sinfo` answers, whether it fits a node of the partition: physical
cores, memory, a suggestion), the MPI launch, the CRYSTAL module with the
reason and any rejected candidates, the executables, the scratch order, the
Python and queue limits the completion step uses, the `mace run` time limit,
and, where SLURM accounting exists, whether the account is one of your SLURM
accounts.

- `mace site --json` prints the same as JSON.
- `mace site --refresh` forgets the cached CRYSTAL choice, Python module
  check and SLURM answers, and checks again.
- `mace site --init` writes `site.toml` with every setting commented out
  (only when it does not exist); edit a line, then check with `mace site`.

**`mace site check`** tests the settings on this cluster before your first
real job, one line per item with a `Fix:` line for each problem: the
settings files, SLURM itself (on a PBS/Torque or LSF cluster it says those
are planned for MACE 1.2), the account, the partition (exists, up, admits
the account and QOS), QOS and constraint, the job size against the nodes,
CRYSTAL (module or `bin_dir`), the MPI launcher, the completion step's
Python (`import tomllib, mace.queue.manager`), scratch, and a dry
submission: the real generator writes job scripts for a tiny MgO deck in a
temporary folder, `bash -n` reads them and `sbatch --test-only` asks SLURM
whether it would accept them (no job is created; some sites log such
requests). The folder is removed afterwards. Exit code 0 when all is well,
1 with warnings, 2 when something failed. On a login node that serves no
modules (MSU HPCC's gateway, for example) the CRYSTAL and Python modules
cannot be tried, so those two items are warnings; run `mace site check` on
a compute or development node to test them.

- `mace site check --json` prints the same as JSON.
- `mace site check --partition P` / `mace site check --account A` test
  another partition or account than the settings name.
- `mace site check --kind scf` (or `prop`, `both`, the default) chooses the
  job scripts to test.
- `mace site check --no-sbatch` stops after writing and reading the scripts
  and keeps them (the folder is printed).

When `sbatch` refuses a job for a reason a setting can fix, `mace submit`
(and the queue manager) adds one line, once per run, naming the key:
"Invalid account or account/partition combination" and "You must specify an
account" (`[slurm] account`), "Invalid qos specification" and "Job violates
accounting/QOS policy" (`[slurm] qos`), "Requested node configuration is not
available" and "Memory required by task is not available" (`[resources]`),
"Invalid feature specification" (`[slurm] constraint`), "Requested time
limit is invalid" (`--walltime` or `[slurm] partition`).

**Workflows** made with these settings store `[resources]` (the lines no
planner prompt writes), `[mpi]`, `[paths]`, `[crystal]`, `[modules] purge`
and `[slurm] qos`, `constraint` and `extra` in the plan; every later step
uses the stored values, checked again when its script is written (a plan
edited by hand cannot put shell code into a job script). At plan time the
workflow's queue limits are lowered to fit your MaxSubmitJobs when SLURM
reports a smaller one.

**`mace run`'s time limit.** At most the node's CPU-time limit
(`ulimit -t`) minus 5 minutes; on MSU HPCC, 1h55m (its development nodes stop
a process at 2 hours of CPU time); `[local_run] max_time` sets it. MSU HPCC is
recognized by its domain, its profile script and its node names, so a
cluster with MSU-like node names (`skl-12`) also gets 1h55m: set `max_time`
there.

## mace cluster

Cluster status, limits, quotas and the progress of your own jobs, read from
SLURM. It works on any SLURM cluster (nothing site-specific is built in) and
is **read-only**: it never submits a job (the only `sbatch` it runs is
`sbatch --test-only`, which SLURM answers without queuing anything), never
cancels, holds or changes a job, and never deletes or moves a file.

It runs on a login or development node of the cluster you are logged in to,
with nothing but the standard library of Python 3.11 or newer (numpy, PyYAML
and the rest are not needed). On a machine without SLURM it says
"No SLURM on this machine (sinfo not found)" and exits with code 3;
`mace cluster quota`, `files`, `estimate` and `devnodes` still work there.

```
mace cluster                      overview: usable partitions, your jobs vs limits,
                                  quota warnings, reservations soon
mace cluster nodes [--partition P] [--feature F] [--free] [--all-partitions]
                   [--group-by feature|partition]
mace cluster fit --cpus N [--nodes N] [--mem-per-cpu SIZE | --mem SIZE] [--gpus [TYPE:]N]
                 [--constraint EXPR] [--partition P] [--time D-HH:MM:SS] [--predict]
                 [--script JOB.sh]
mace cluster reservations
mace cluster quota [--rediscover]
mace cluster files [PATH] [--top N] [--max-seconds S] [--sizes top|all|off]
mace cluster limits [--check-submit N --cpus C]
mace cluster job ID [--follow] [--live]
mace cluster estimate DECK.d12 [--cores N] [--history DIR] [--db materials.db]
mace cluster devnodes [--pick]
```

Every subcommand takes `--json` (the same data, machine-readable) and
`--refresh`. Exit codes: 0 ok (warnings included), 2 bad arguments, 3 no
SLURM here, 4 job not found or not yours, 5 a file scan is already running.

### How often it asks SLURM

Every query is cached in `~/.cache/mace/cluster/<cluster>/` (one small file
per query, readable only by you) and shared by all your MACE shells and GUI
windows:

| What | Asked at most every | `--refresh` / Refresh at most every |
|---|---|---|
| your jobs (`squeue -u`), partition summary (`sinfo`) | 60 s | 15 s |
| nodes (`scontrol show nodes`) | 60 s | 60 s |
| partitions (`scontrol show partition`) | 1 h | 5 min |
| reservations | 5 min | 1 min |
| limits and fairshare (`sacctmgr`, `sshare`) | 10 min | 2 min |
| quotas | 10 min | 2 min |
| one job (`scontrol show job`) | 60 s | 15 s |

So with the GUI open there is at most one `squeue` and one `sinfo` per
minute, and one node query per minute while the Cluster view is visible. A
query that fails is retried after 30 s, then 60, 120 ... up to 5 minutes,
showing the last answer marked stale meanwhile. Two MACE processes never
ask the same thing at once (the second waits at most 35 s, then shows the
cached answer). Inside a SLURM allocation (an `salloc` shell, a batch job,
a GUI started with `--slurm`) the allocation's `SLURM_*` variables are left
out of every query, so the answers are those of a login node.

### Nodes and "will it start?"

`mace cluster nodes` sums the nodes by CPU type (a node feature such as
`amd24` or `intel18`, picked by `[cluster] group_feature_regex`) or by
partition: idle, mixed, allocated, down or drained, free cores, free memory
and free GPUs. Only the partitions your accounts, groups and QOS may use are
counted (hidden group partitions included); `--all-partitions` shows the rest,
marked "not usable by your accounts".

`mace cluster fit` lists, per usable partition, the nodes that could take
the request **now**: free cores and memory per node, free GPUs of the type,
the `--constraint` features (`&`, `|` and plain names; brackets and counts are
left to SLURM) and a MaxTime that covers `--time`. Free now is a snapshot:
priority, limits and backfill decide the real start. `--predict` asks SLURM
itself with `sbatch --test-only` and prints the predicted start, nodes and
partition, or SLURM's refusal ("Requested time limit is invalid"). Maintenance
reservations that a job of that walltime would run into are warned about.

### Quotas

`mace cluster quota` shows space and file counts for your home, scratch and
project spaces, with a warning from 80 % and critical from 90 % (both
`[quota]` settings). It reads, first that answers:

1. a `[quota] command` you give in site.toml;
2. your site's own `quota` command, when your login profile defines it as an
   alias. MACE looks it up once a day (`bash -lc 'alias quota'`, 20 s at
   most; `--rediscover` looks again now) and then runs it directly. It runs it
   only when it is an absolute path to a file outside your home folder that you
   cannot change, with no shell syntax; of the variables the alias sets, only
   `LANG`, `LC_*`, `TERM`, `COLUMNS`, `NO_COLOR`, `FORCE_COLOR` and the names in
   `[quota] alias_env_allow` are passed (clearing `LD_LIBRARY_PATH` is allowed;
   setting `LD_*`, `PYTHON*`, `PATH` and the like never is);
3. `lfs quota`, `mmlsquota` (GPFS), `quota -s -w`;
4. `df`, labeled "Filesystem (not your quota)" when nothing else answered.

### Your limits

`mace cluster limits` shows each association (account, partition, QOS) with
its running, submitted and per-user core limits next to your current use
(job array tasks count one by one), and your fairshare.
`--check-submit 300 --cpus 32` says what adding 300 jobs of 32 cores would
run into, for example "This adds 300 jobs; 812 of 1000 submit slots are in
use, so 112 more are refused by SLURM." It is advice (exit 0); `mace submit`
is unchanged.

### One job: progress, ETA and walltime

`mace cluster job ID` works for your own jobs only. It shows the state,
reason, node, elapsed time against the limit, cores and memory, and the
files (work folder, output, job log, scratch folder), for an array task the
other tasks of its array still in the queue, then the progress read from the
live output:

The estimate comes as nested levels: the job's total, the stages of the run in
the order CRYSTAL runs them, and the loop running inside the current stage.
`--json` carries the same levels (`progress.levels`, the total first; each with
its sentence in `text`, the counts, the time left as a low/high pair and the
reason behind it in `basis`):

```
Progress (opt): Point 5 of about 7-42 (+ final restart 1-7)
  Total about 2 min - 1 h 50 min left (rough)
    Optimization point 5 of about 7-42 (+ final restart 1-7)
      Thresholds: SCF energy change 1e-9 (tighter than the optimization default 1e-7)
      Next point's SCF: about 15-28 cycles (from earlier points); running 59 s of about 2 min per point
  Why: about 2 min - 1 h 50 min left in total: Optimization: about 3-44 more points (...)
Note: Could exceed the walltime if it runs to the long end of the range (1 h 50 min left in total, 46 min of walltime remaining).
```

"Point 5" is the last point CRYSTAL has printed; the SCF running now
belongs to the next one.

- **Optimizations:** the point reached, the four criteria against their
  thresholds, and the points left from CRYSTAL's own stop rules: the run ends
  at the first point where all four criteria are met, or both gradients are
  met with the energy change below TOLDEE, or MAXCYCLE is reached. Each
  criterion's trend over the last 6 points of the current segment gives the
  range; the high end is at least 6x the low end, and where only MAXCYCLE
  would bound it, the 90th percentile of points left in the corpus half the
  rule was chosen on (44) is used and the range reads "rough". CRYSTAL23
  checks a converged optimization again with the integrals of the final
  geometry and restarts it when that check fails (FINALRUN 4 by default), so
  "+ final restart 1-7" is added unless the deck sets FINALRUN 0-2. With
  fewer than 3 points, a flat or rising trend, or just after a FINALRUN
  restart the points left are "unknown". While the first point runs, its SCF
  is in the output itself, and the inner line gives its cycle and |DE| against
  the threshold (no cycles-left estimate there). After that the
  inner line is the next point's SCF: after the first point CRYSTAL writes the
  SCF to `SCFOUT.LOG` in the scratch folder, so the running cycle is read from
  there when MACE may read that folder (the last 2 MB, read-only); otherwise
  the line gives the cycles the last points took.
- **Single points:** setup and integrals, the SCF ("SCF cycle 6 of about 7-14,
  ~4 s per cycle (|DE| 5.5e-9, threshold 1e-9)"), then the rest after the SCF.
  CRYSTAL ends the SCF after two cycles in a row with |DE| below the
  threshold, so once it is below, one more cycle is expected.
  A restart from a previous density (GUESSP, in the deck or in the output) is
  flagged: the estimate was fit on runs without it, which usually need more
  cycles. The deck's SCF MAXCYCLE caps the high end.
- **Frequencies:** an optional preoptimization (named, with its point; levels
  for it come later); for IR/Raman through CPHF the reference SCF, the CPKS
  field directions X/Y/Z and, for Raman, the six second-order directions with
  the current cycle, then the IR/Raman tensors; the central point; the
  displacements ("Displacement 3 of 5 (60 %), about 15 min - 30 min each";
  the output does not change while a displacement runs, so the time since its
  last change is the running displacement's, and it comes off the low end);
  the frequencies and output. CRYSTAL runs the CPHF stages before the central
  point, and the central point's time (its SCF and gradient) leaves them and
  the setup out. These stages have no time estimate yet, so a running one says
  how long it has run ("IR/Raman tensors: running (14 min so far)"): the IR
  tensors print three lines and then nothing for many minutes. A FREQINFO.DAT restart
  shows the central point as restored and takes the time per displacement
  only from the rows computed in the restarted run. No time is extrapolated
  from the central point: displaced cells have lower symmetry and cost from a
  fraction to many times as much (6x on one corpus run), so before the first
  displacement of the run finishes the total is "unknown".
- **The total** adds the time left of the stages still to run. When a stage
  has no upper end (the CPHF stages have no time estimate yet), it reads "at
  least ... left" and names that stage; when nothing beyond a few seconds is
  known, "unknown". "fair" means a range from the trend that is at most 3x
  wide; "rough" a wider one or one whose high end comes from typical runs.
- **Thresholds** are shown against CRYSTAL23's default for the context (an
  optimization's gradients, displacements and energy change; the SCF energy
  change in an optimization 1e-7, in a single point 1e-6, in FREQCALC 1e-9
  as section 8.1.3 of the manual gives it, though its TOLDEE table says
  1e-10), only where one differs. MACE's usual decks set a tighter SCF
  threshold (1e-9 or below), so an optimization from MACE usually shows that
  line. They do not change the estimate: the corpus has 115 runs with the
  default gradient threshold and only 3 with a tighter one (12, 12 and 20
  points against a median of 15), no evidence that tighter thresholds need
  more points.
- **Walltime risk:** "likely to exceed" when even the low end of the total
  does not fit in the walltime left, "tight" when the central estimate (the
  geometric mean of the range) needs 80 % of it or more. When only the long
  end would not fit, a softer note says so ("Could exceed the walltime if it
  runs to the long end of the range ..."). The sentence says what happens
  then, from the job folder's own `recovery_config.yaml` (or MACE's) and from
  whether another job of the same folder is still queued: a job killed at its
  walltime does not run MACE's completion step, so when it is the last one,
  nothing notices it until you run `mace manager --callback-mode completion`
  in its folder.

Times are anchored at CRYSTAL's own start (the output's `STARTING DATE ...
TIME ...` line) when it agrees with SLURM's start within 10 minutes, else at
SLURM's start; module loading and file staging are left out that way.
SLURM's elapsed time comes from a query cached for up to a minute; for a
running job the cache's age is added, so the time moves on between refreshes.
Once a job has left the queue, its state and elapsed time come from SLURM's
accounting (scontrol forgets an ended job within minutes on many clusters).
When SLURM ended it (TIMEOUT, CANCELLED, ...) before CRYSTAL finished, the
stage it was in reads "stopped here" and the later ones "not reached": a
killed CRYSTAL writes no last line, so the output alone looks unfinished. A
copied or finished output has no clock, so it gets no time per displacement.
The same output read again unchanged (and less than 30 s later) is answered
from a small in-memory cache.

#### How the estimates were tested

Every rule was backtested on the real CRYSTAL23 outputs of MACE's test corpus,
cut at many points of each run and compared with what happened next. The
corpus holds conformers and repeated geometries (an SP of an optimized
structure repeats the optimization's first SCF), so the runs were split into
two halves by family (material name plus identical opening SCF energies):
choices were made on one half by cross-validation over its families and
scored on the other, with 95 % intervals from resampling families. The
held-out results, which the test suite gates
(`tests/cluster/test_cluster_eta.py`):

| Level | Held-out runs (families) | Inside the range |
|---|---|---|
| Optimization points left | 23 (17) | 84 % (75-93); time left 91 % (82-100); no estimate in 32 % of the cuts |
| Next point's SCF cycles | 23 (17) | 82 % (78-87) |
| Single-point SCF cycles left | 34 (19) | 72 % (61-91) when first scored, 80 % (70-100) as shipped; time left 73 % (62-94), then 80 % (69-100); no estimate in 40 % |
| FREQ displacements left (in SCF cycles) | 11 (11) | 89 % (63-100) |

How blind these are: the optimization rule was first worked out in a study
that looked at the whole corpus, so its parameters were chosen again from the
dev half alone, with the selection rule fixed before the run; the earlier
study had still shaped which parameters were tried. The window of the next
point's SCF cycles was also first measured on the whole corpus. For the
single points, a fix made after the first held-out scoring (once |DE| is below
the threshold, one more cycle is left) was found on the dev half; both scores
are given, and the test floors use the first. The FREQ factors (0.75-1.5) were
not tuned.

The optimization ranges are wide (median high/low 9x) and about 1 in 3 take
their high end from typical runs; the rule they replace had 70 % (61-78) on
the same split, with 59 % of its ranges capped only by MAXCYCLE. The walltime
labels, on the same cuts with synthetic walltimes: "likely to exceed" was
wrong 4 % (0-7) of the time, "tight" showed on a job with twice the time it
needed 10 % (8-14), and an overrun got neither a label nor the note 6 %
(0-10). One refresh costs about as much as before: 7.9 s against 7.4 s for
the earlier passes on a 50 MB output, and 1.03-1.12x the earlier passes on
120 MB optimization, frequency and single-point outputs (10-20 s on this
workstation; an unchanged output read again within 30 s comes from the cache).

Limits: every corpus run converged (none hit MAXCYCLE, a SCF cycle limit or
the walltime), only 2 used GUESSP, none used SMEAR, and about half of the
single points are small molecular clusters. Estimates can therefore be too
short for an SCF that stalls or a metallic system, and too long for a GUESSP
restart (flagged). The single-point misses are almost all too short, in long
slab SCFs of 200+ cycles. FREQ rows carry no time, so the displacement rule
was tested in SCF cycles. The SCFOUT.LOG reader was checked on a live
optimization on a SLURM cluster (CRYSTAL23 writes the same `CYC` lines there,
and starts the file anew for each point); when its format is not recognized
the inner line falls back to the earlier points.

Planned for 1.1.6: a time estimate for the CPHF/CPKS stages (per cycle and
per direction) and the tensors; levels inside a FREQ preoptimization; a
progress column in the Cluster view's Overview; scaling the typical-run
priors by the thresholds once there are tight-threshold runs to fit; a
GUESSP- and SMEAR-aware SCF prior; reading only the new part of a very large
output; totals for job arrays and workflows; and `mace cluster progress
FILE.out` for runs started with `mace run`.

CPU and memory efficiency come from SLURM's accounting and are shown only
when they are plausible. MACE's job script starts CRYSTAL's MPI ranks through
mpirun, outside SLURM's job steps, so SLURM often measures nothing for them;
the job then says "Not measured by SLURM for this job" rather than a wrong
number. `--live` asks `sstat` once; `--follow` prints again every 60 s from the
shared cache.

### Estimates from history

`mace cluster estimate DECK.d12` looks at finished runs in the deck's folder
(or `--history DIR`, and a `materials.db` read-only): runs of the same
calculation type and functional family, scaled by atomic orbitals, k points
and cores, give a median and a 10-90 % range, and a suggested walltime (the
90 % value x 1.3, rounded up to the hour). With fewer than 3 similar runs it
says "Not enough similar runs (2 found; 3 needed)"; a calculation type whose
estimates missed too often in a leave-one-out check is hidden. Nothing is
changed in your scripts.

### Where are my files

`mace cluster files [PATH]` scans one folder tree at low priority and lists
the folders with the most files (and the largest, when sizes are read). Load
on a parallel file system: about 2000 metadata operations per second for at
most 10 minutes (`[scan]` settings); counts need no per-file stat, sizes do,
so `--sizes top` (the default) reads sizes only in the 20 busiest folders. One
scan per user runs at a time.

Cleanup suggestions are commands to copy, one quoted path each; nothing is
ever deleted. Each rule matches by MACE's own markers or by provenance, never
by guessing from a name:

- **R1** a CRYSTAL scratch folder of a finished MACE job, found only through
  the `.<JOB>.scratch` file the job script leaves in the job folder, when the
  job's output ended normally and no job of yours with that name is queued.
  Never matched: a scratch folder without that pointer, or of a job still
  running or ended early (an OPTGEOM RESTART may need it), or a pointer that
  names the job folder, the scanned folder, your home, a scratch root or a
  folder holding any of them.
- **R2** the same for the `.mace_scratch` fallback folder next to a deck.
- **R3** Python's own temporary folders (`pymp-XXXXXXXX`, `tmpXXXXXXXX`)
  directly inside `/tmp`, `/var/tmp` or the system temp folder, owned by you
  and older than 2 days. Never matched: anything in your home, the scanned
  folder or a scratch folder, or a name like `tmp_results`.
- **R4** `core.<pid>` files that are core dumps (checked by their first 4
  bytes), `__pycache__` folders holding only `.pyc` files (not inside an
  installed package or virtual environment; at most 10 are listed), and
  `.mace_arrays/<tag>` folders of job arrays no longer in the queue.
- **R5** folders marked with a `.mace_test_dir` file, or listed in
  `[scan] leftover_paths`.

### Development nodes

`mace cluster devnodes` probes the hosts in `[devnodes] hosts` over ssh
(batch mode, 5 s timeout) and ranks them by load per core (or free memory);
`--pick` prints only the chosen name, e.g. for
`mace gui connect --node "$(ssh <gateway> mace cluster devnodes --pick)"`.
Without hosts configured it says how to add them.

### Settings (site.toml)

All optional, in `~/.config/mace/site.toml` (`MACE_SITE_CONFIG=<path>`
replaces it, `MACE_SITE_CONFIG=none` ignores it):

```toml
[cluster]
group_feature_regex = "^(intel|amd|arm|xeon|epyc|zen|skylake|cascade|icelake|sapphire|grace)[-_]?[0-9a-z]*$"
node_query = "auto"           # auto | scontrol | sinfo

[quota]
# command = ["/path/to/site/quota/tool"]   # format: table | lfs | quota | json
warn_percent = 80             # 50..99
critical_percent = 90         # above warn_percent, at most 99
# paths = ["/project/mygroup"]
# alias_env_allow = []

[scan]
entries_per_second = 2000     # 500..10000
max_seconds = 600             # 60..3600
max_entries = 1200000         # at most entries_per_second x max_seconds
sizes = "top"                 # top | all | off
# leftover_paths = []

[devnodes]
# hosts = ["dev-node-1", "dev-node-2"]
pick_by = "load"              # load | memory
```

A bad value is named with its line ("site.toml:12: [quota] warn_percent =
120: must be a whole number from 50 to 99") and keeps the built-in value.

> **Example: MSU ICER HPCC.** The site quota tool is an alias that the login
> profile defines (`quota` runs `/usr/local/hpcc/bin/quota-scripts/userquota.py`
> with `LD_LIBRARY_PATH` cleared); MACE finds and uses it with no setting. Node
> types are features such as `amd20`, `amd22`, `amd24` and `intel18`, matched
> by the default `group_feature_regex`. Development nodes can be listed as
> `[devnodes] hosts = ["dev-amd20", "dev-amd24", "dev-intel18"]`.

### Privacy

Only your own jobs are asked about in detail (`squeue -u`, `sacct -u`); a job
id that is not yours is refused. Node, partition and reservation data are
public on a cluster; MACE never lists other users' jobs.
