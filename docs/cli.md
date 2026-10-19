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
| `mace recover` | Fix failed tracked runs and resubmit them (memory, SCF, walltime, ...) | [Error handling](../DOCUMENTATION.md#error-handling-and-recovery) |
| `mace queue` | Your queue budget: submit held jobs (drain) or show it (status) | [mace queue](#mace-queue) |

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
| `mace site` | Show job-script site settings (--init, check, set/unset, suggest) | [Site settings](#site-settings-mace-site-sitetoml) |
| `mace help` | This list, one group (mace help monitoring) or one command | `mace help --help` |
| `mace version` | Show the MACE version (same as --version) | `mace version --help` |
| `mace update` | Update MACE (and a cluster's) to a release, preview or main (docs/update.md) | `mace update --help` |
| `mace credits` | Show the development team (same as --credits) | `mace credits --help` |

### Aliases and deprecated

Old names; they keep working. `mace help aliases` prints this group.

| Command | What it is for | More |
|---|---|---|
| `mace status` | Same as 'mace monitor --status' | `mace status --help` |

## Command reference

One section per command, in the order of `mace --help`. Each lists every
option `mace <command> --help` shows (a test keeps the two in step); the
help adds the full wording and more examples. Paths in the examples are
placeholders.

### mace workflow

Plan a multi-step calculation (OPT, SP, BAND/DOSS, FREQ, ...) once; MACE
submits each next step when the one before it finishes.

```
mace workflow [--interactive | --quick-start | --execute FILE | --status [ID] | --list | --show-templates] [options]
mace workflow --spec SPEC [--write-plan] [--json]
mace workflow --pause ID | --resume ID | --cancel ID [--yes] | --process ID [--json]
```

| Option | What it does |
|---|---|
| `--interactive` | Plan step by step in the interactive planner |
| `--quick-start` | Start from a predefined template, with `--cif-dir` or `--d12-dir` |
| `--execute FILE` | Run a saved plan (`workflow_configs/workflow_plan_*.json`) |
| `--status` | The workflows planned here and their state (as `--list`; add `--json` for JSON); read-only, creates no database |
| `--status ID` | One workflow: the state of every step of every material (planned, waiting, queued, running, completed, failed, blocked, paused, ...), read from its plan, its database (read-only), `squeue` and its control file; a running step also shows its time left (read from its `.out`, as the Jobs view estimates it) |
| `--list` | The workflows planned in the folder (`--work-dir`), newest first, with their state; CLI- and GUI-started ones and plans from MACE 1.1.4/1.1.5 |
| `--spec SPEC` | Check a workflow spec and show each step's settings and resources with where each value comes from; nothing is written |
| `--write-plan` | With `--spec`: write the plan (and, for decks, its job scripts) and print the command that launches it |
| `--pause ID` | Completions are still recorded; no next step and no recovery job is submitted |
| `--resume ID` | Lift a pause (or a cancel) and start the steps held meanwhile |
| `--cancel ID` | Stop the workflow and `scancel` its own queued and running jobs (those SLURM lists among your jobs); lists them and asks first |
| `--process ID` | Start the steps whose completion step did not run (a job finished and nothing was submitted after it), as that callback would have; this workflow only, and not while it is paused or canceled |
| `--yes` | With `--cancel`: do not ask (with `--json` or without a terminal, `--cancel` only lists the jobs unless `--yes` is given) |
| `--json` | With `--status ID`, `--list`, `--spec`, `--pause`, `--resume`, `--cancel`, `--process`: one JSON document on stdout |
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

#### Workflow specs

A spec is one JSON file that describes a whole workflow with its per-step
settings; the GUI's workflow builder writes the same file. `mace workflow
--spec SPEC --write-plan` turns it into an ordinary plan and prints the
command that launches it: `mace workflow --execute PLAN` for CIFs, `mace
submit` with the plan for decks (see mace submit):

```json
{"version": 1, "label": "perovskites",
 "inputs": {"type": "cif", "directory": "inputs"},
 "template": "full_electronic",
 "steps": {
   "OPT_1":  {"settings_file": "opt_cif2d12.json", "resources": {"ntasks": 32, "walltime": "2-00:00:00"}},
   "SP_2":   {"overrides": {"method_modifications": {"new_functional": "HSE06"}}},
   "BAND_3": {"settings_file": "band_opt2d3.json", "resources": {"memory": "48G"}}},
 "execution": {"max_jobs": 100, "job_arrays": {"first_step": true, "max_running": 20}}}
```

- `inputs`: `type` `cif` or `d12`, and `files` (a list) or `directory`.
  The decks of a d12 start must be in one folder, and `--write-plan` writes
  their plan there (they are submitted from it; a `--work-dir` elsewhere is
  refused). Paths are relative to the spec.
- `template` (a template name) or `sequence` (a list such as
  `["OPT", "OPT2", "SP", "FREQ"]`).
- `steps` (keyed `<TYPE>_<position>`; every key optional):
  `settings_file` is a Calculation Setup JSON: the cif2d12 options of the
  first step of a CIF start, else the opt2d12/opt2d3 config the step's deck
  is generated with (copied to `workflow_configs/spec_<id>/`); an opt2d3
  config that names another calculation type than its step is refused. `overrides`
  are merged into the step: `method_modifications` (`new_functional`,
  `use_dispersion`; without `use_dispersion` the step keeps its parent's D3
  when the new functional has D3 parameters, so HSE06 on a B3LYP-D3 parent is
  HSE06-D3, and `"use_dispersion": false` drops it, as in the GUI's workflow
  builder), `optimization_settings`, `optimization_type`,
  `frequency_settings` for d12 steps, `d3_config`/`d3_settings` for
  BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL/ORBITALS, and on SP steps the SOC keys
  (`soc`, `soc_ecp`, `soc_basis`, `soc_guess`, `smear`, `soc_fmixing`; see
  [SOC decks](#spin-orbit-coupling-soc-decks)); other keys are refused. The
  SOC keys are checked as `opt2d12` checks them, need `"soc": true`
  (`"soc": false` makes a scalar step after a SOC one), and are refused on
  OPT, FREQ and properties steps (manual p. 166). `overrides` next to a
  `settings_file` on the same step are refused (the step is generated from
  that file alone: put them in it), except on the first step of a CIF start,
  where they are merged into the conversion settings. On a CIF start
  whose first step is SP they go into the conversion settings (no
  `soc_guess`: a CIF has no parent run). `--json` lists, per step, the
  later steps that cannot start from a SOC run. `resources`:
  `ntasks`, `walltime`, `memory` or `memory_per_cpu`, `account`,
  `partition`, `constraint` (node features: written as the SLURM constraint line
  of every job of the step, recoveries included, in place of the site's
  `[slurm] constraint`), `exclude`. The first step of a d12 start
  runs the decks as they are (resources only).
- `execution`: `max_jobs`, `walltime` (every step), `auto_progression`,
  `job_arrays`. By default a workflow has no `max_jobs` of its own: every
  workflow and manual submission shares your queue budget (see
  [mace queue](#mace-queue)). A `max_jobs` you set is a hard cap on the
  workflow's jobs queued or running at once, first step and every BAND or
  DOSS job included: a job over it is held (reason: the workflow's max_jobs)
  and submitted when one of the workflow's jobs ended. A plan of MACE 1.1.5
  or earlier stores 200 whether or not anyone chose it; 200 counts as not set.

A step the spec does not mention is the template's step exactly: the plan,
job scripts and decks are byte for byte what a `mace submit` progress
template (decks) or `mace workflow --quick-start` (CIFs) write, and a d12
follow-up inherits its parent's settings as always.

```bash
mace workflow --spec spec.json                   # check; shows every value's origin
mace workflow --spec spec.json --write-plan      # writes the plan, prints the launch command
mace workflow --execute workflow_configs/workflow_plan_<id>.json              # CIF start
mace workflow --list
mace workflow --status workflow_<id>
mace workflow --pause workflow_<id>
mace workflow --resume workflow_<id>
mace workflow --cancel workflow_<id>
mace workflow --process workflow_<id>
```

Pause, resume and cancel write `workflow_configs/workflow_control_<id>.json`
next to the plan; the plan itself never changes. Every job callback reads it:
while a workflow is paused or canceled its completions and failures are
recorded, and nothing is submitted. `mace recover` leaves its failed steps
alone too (listed as "resume it first"); resume the workflow, then recover
them. `--resume` starts the next
steps of the runs that completed meanwhile. `--cancel` sets the state first,
so a job ending at that moment submits nothing, then runs `scancel` for the
workflow's jobs that a fresh `squeue` lists among yours; a bare job-array id
is never canceled. `--process` is for a step `--status` shows as stale: its
parent finished but the job's completion step never submitted it (a node
crash, a callback that failed); it processes this workflow's completed
calculations the way the callback does, and leaves other workflows sharing
the database alone. `--process` and `--resume` wait for the queue lock the
job callbacks hold, so they never submit a step a callback is submitting.
When `scancel` fails for some of the jobs, `--cancel` asks `squeue` again
and reports each job as canceled or not.

`mace workflow --spec SPEC --json` names each step's `account` as its job
script gets it: a string, or `null` when the script has no `#SBATCH -A` line
(SLURM charges your default account).

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
| `--max-jobs N` | At most N - reserve of your running and pending jobs in squeue (each array task counted, jobs MACE did not submit too); default: your queue budget ([mace queue](#mace-queue)), which also caps a smaller N |
| `--reserve N` | The slots `--max-jobs` leaves free (default 0); without `--max-jobs` it is ignored with a warning: set `[queue] reserve` in site.toml instead |
| `--organize` | Copy inputs into `<calc_type>/<material_id>/` (default: submit in place) |
| `--callback-mode MODE` | One cycle of `full_check` (default), `submit_new`, `status_check`, `early_failure` or `completion` |
| `--dry-run` | Not implemented: ignored with a warning, and a real submission follows. Use `--status` |

```bash
mace manager --status
mace manager --callback-mode completion
```

More: [Material tracking](../DOCUMENTATION.md#material-tracking-system).

### mace recover

Finds the failed calculations in the tracked database, classifies each
failure (the same classification the queue manager records), shows a table
with the calculation, its error class, the proposed fix and the retries used
of the limit, and after a confirmation applies the fixes and resubmits the
calculations. The resubmission is the queue manager's own recovery path, the
one a job's completion callback runs: the calculation keeps its tracking, a
workflow step keeps its `workflow_id` and step (the workflow goes on when the
new run completes), and the retry limits of `recovery_config.yaml`
(`max_retries` per error type) and of the lineage (3 recoveries in all) are
the same.

Completion callbacks do not recover a job that ended more than
`MACE_RECOVER_MAX_AGE_DAYS` (default 3) days ago on their own; run `mace
recover --db <db>` for those.

```
mace recover [--dry-run] [--yes] [--calc-id ID] [--dir PATH] [--errors CLASS,...]
             [--include-stale] [--overwrite-sh] [--max-recoveries N] [--json]
             [--db PATH] [--config FILE]
mace recover --action stats|config
mace recover --prepare-only
```

| Option | What it does |
|---|---|
| `--dry-run` | Show the diff of every file each fix would write or change and the `sbatch` command; nothing is changed or submitted |
| `--yes`, `-y` | Apply the fixes and resubmit without asking |
| `--calc-id ID` | Only this calculation (repeat, or comma-separate several) |
| `--dir PATH` | Only calculations whose work folder is inside PATH (without `--db`: the databases in PATH, `materials.db` and `.mace_context_*/materials.db`) |
| `--errors CLASS,...` | Only these error classes (`oom`, `scf`, `time_limit`, `shrink`, `disk`, `opt_unconverged`, ...) or error types (`memory_error`, ...) |
| `--include-stale` | With `--yes`: also resubmit jobs that ended more than `MACE_RECOVER_MAX_AGE_DAYS` (default 3) days ago |
| `--overwrite-sh` | Write the job script again from the template when the fix leaves it alone (default: the existing `<job>.sh` is submitted as it is) |
| `--max-recoveries N` | Resubmit at most N calculations in one run (default 10) |
| `--max-recovery-attempts N` | Recoveries per calculation lineage, all errors together (default 3) |
| `--max-jobs N`, `--reserve N` | No resubmission when the queue holds `N - reserve` jobs, as `mace manager` (default: your queue budget, [mace queue](#mace-queue); `--reserve` defaults to 0 and needs `--max-jobs`) |
| `--json` | The plan (and the results) as one JSON document on stdout; the GUI reads this. It asks nothing, so it needs `--yes` or `--dry-run` (without either it exits 2) |
| `--prepare-only` | Only write the fixed inputs and job scripts and print the command that submits each one (the 1.1.5 behavior) |
| `--action ACTION` | `recover` (default), `stats` or `config` |
| `--config FILE` | Recovery rules (default: `recovery_config.yaml` in the job's folder, else MACE's own) |
| `--db PATH` | Database (default: `materials.db` and the workflow databases `.mace_context_*/materials.db` here, or in `--dir`) |
| `--create-config` | Write a default `recovery_config.yaml` |

Without `--yes` it asks before it changes anything; with no terminal and no
`--yes` the answer is No. Before each resubmission it takes the queue lock
the job callbacks take and checks again that the calculation is still failed,
that no newer run of it is submitted or running, that `squeue` lists no job of
its deck (its own job, or a recovery of it, in its folder), and that the
queue is not full; so two `mace recover` runs, or a run and a callback, never
resubmit one calculation twice. Calculations without an automatic recovery
(atoms too close, a linearly dependent basis, no writable scratch, ...) are
listed with the reason. A job that ended more than `MACE_RECOVER_MAX_AGE_DAYS`
days ago is never resubmitted by the callbacks; `mace recover` asks about
those separately. A database it cannot write (a read-only file or folder)
is reported and skipped. The exit code is 1 when a resubmission failed, when
the queue lock stayed busy (run it again), or when no database could be
opened; otherwise 0.

A dry run on an out-of-memory OPT and an SCF stop:

```
$ mace recover --dry-run
Failed calculations in /scratch/me/run/materials.db:
  #  Calculation                    Error                    Retries         Proposed fix  [Action]
  1  3_dia3_OPT_20261014_231525168  oom (memory_error)       0/2, total 0/3  new job script: memory x1.5 (at most 200GB)  [resubmit]
  2  1_dia_SP_20261014_231525180    scf (convergence_error)  0/2, total 0/3  new deck: SCF MAXCYCLE +1000, FMIXING -10 (not below 10)  [resubmit]
2 to resubmit.

Dry run: nothing is changed or submitted.
[1] 3_dia3_OPT_20261014_231525168 - new job script: memory x1.5 (at most 200GB)
    error: Detected: KILLED BY SIGNAL: 9
    fix: Memory increased from 5GB to 7GB
    would write /scratch/me/run/3_dia3_opt_rev1_recovery_20261014_231525.sh
      --- /scratch/me/run/3_dia3_opt_rev1.sh
      +++ /scratch/me/run/3_dia3_opt_rev1_recovery_20261014_231525.sh
      (the diff: the memory line, 5G per CPU raised to 7GB per CPU)
    would run: cd /scratch/me/run && sbatch /scratch/me/run/3_dia3_opt_rev1_recovery_20261014_231525.sh
[2] 1_dia_SP_20261014_231525180 - new deck: SCF MAXCYCLE +1000, FMIXING -10 (not below 10)
    error: Detected: FERMI ENERGY NOT IN INTERVAL
    would write /scratch/me/run/1_dia_sp_recovery_20261014_231525.d12
      --- /scratch/me/run/1_dia_sp.d12
      +++ /scratch/me/run/1_dia_sp_recovery_20261014_231525.d12
       MAXCYCLE
      -1600
      +2600
       FMIXING
      -30
      +20
    would run: cd /scratch/me/run && bash .../mace/submission/submitcrystal23.sh 1_dia_sp_recovery_20261014_231525   (writes 1_dia_sp_recovery_20261014_231525.sh)
    would run: cd /scratch/me/run && sbatch /scratch/me/run/1_dia_sp_recovery_20261014_231525.sh
```

(Shortened. The `_recovery_<date>_<time>` names get the time of the real run.)

```bash
mace recover --dry-run                  # see the changes
mace recover                            # the same table, then: Apply the fixes and resubmit 2 calculation(s)? [y/N]
mace recover --errors oom,time_limit --yes
mace recover --calc-id 3_dia3_OPT_20261014_231525168
mace recover --action stats
```

More: [Error handling](../DOCUMENTATION.md#error-handling-and-recovery).

### mace queue

Your queue budget. MACE keeps at most a budget of your jobs queued (running
and pending, every array task counted, all workflows, manual submissions and
recoveries together):

```
budget = your SLURM submit limit - reserve        ([queue] in site.toml)
```

The limit is the smallest of `scontrol show assoc_mgr` MaxSubmitJobs and
`sacctmgr`'s association and QOS MaxSubmit (each read once a day); when none
can be read MACE keeps at most 900 queued and says so once. The reserve
(default 100) stays free for jobs of your own. On MSU HPCC (1040) the budget
is 940.

Every MACE sbatch is admitted job by job under one lock all your MACE
processes share (a file in `~/.cache/mace/queue`, or `MACE_QUEUE_STATE_DIR`):
the first step of a workflow, every next step (BAND and DOSS each on its
own), recovery resubmissions, job arrays and `mace submit`. A job that does
not fit is **held**: its row shows `held` (`mace workflow --status`, `mace
monitor --status`, the GUI's Workflows and Jobs views, `--json` outputs) with
a reason: the queue budget is full, the workflow's own `max_jobs` is reached,
or SLURM refused it at its submit limit. A SLURM refusal (QOSMaxSubmitJobPerUserLimit,
AssocMaxSubmitJobLimit, "job submit limit") holds the job, stops submitting
for that run and lowers the budget to the limit SLURM showed for an hour; any
other sbatch error leaves the job failed, with SLURM's message.

Held jobs are submitted as slots free up: after every completion step, by
`mace workflow --process ID`, by the GUI while it is open, and by `mace queue
drain`. When work is held and no MACE job is queued (your queue is full of
other jobs), one small job named `mace-drain` (1 core, 5 minutes, starting 15
minutes later, your site's account and partition) runs `mace queue drain`; it
comes back only while work is still held and no MACE job is queued, and there
is never more than one. When squeue cannot be read (it is tried twice), the
job is submitted anyway, at most once an hour. `[queue] sentinel = false`
turns it off.

The order (`[queue] order`): `depth` (default) submits the next steps of
materials already started first, then the oldest first steps; `breadth` the
first steps first; `mixed` offers `followup_share` of the free slots to next
steps and the rest (at least one) to first steps. Within a step the oldest
held job goes first. (SLURM itself starts pending jobs by its own priority;
a lower SLURM priority (nice value) for first steps, which would let next
steps start sooner, is a possible later option, to be measured on a cluster
first.)

```
mace queue drain [--dry-run] [--json] [--db DB] [--quiet]
mace queue status [--json]
```

| Option | What it does |
|---|---|
| `drain` | Submit held jobs of every active workflow while the budget has room, in the `[queue]` order; exits 1 when another process holds the budget lock or a held job could not be submitted |
| `--dry-run` | Show what would be submitted; submit nothing |
| `--json` | The report as JSON: released, held by reason, budget, use, the sentinel |
| `--db DB` | Also drain this database (repeatable); active workflows are found on their own |
| `--quiet` | Print nothing but errors |
| `status` | The budget and where it comes from, your queued jobs, held work by reason, the sentinel |

Jobs held under 1.1.5 or 1.1.6 rules are converted on the first drain: a row
`pending` with no job id for more than 10 minutes (and no job of its name
queued), or `failed` with no job id after a limit refusal, becomes held when
its job script exists. Jobs already queued by 1.1.5 or 1.1.6 keep their old
completion-step line (950 and 50): those completion steps keep at most the
smaller of 900 and your budget queued. An untracked `mace submit` has no
database to hold jobs in: it lists the decks that did not fit and asks
before it uses the reserve for them (see [mace submit](#mace-submit)). Any
other use of `mace queue` (a bare `mace queue`, `mace queue --status`) prints
this help and exits 2; the old queue manager is `mace manager`.

```bash
mace queue status
mace queue drain --dry-run
mace workflow --process workflow_20261017_120000   # also releases held jobs
```

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

`--status` (not `--summary`) also lists each active tracked job with its time
left, from its output, as `mace cluster job` computes it: "~2.1 h (1.4-3.2 h)"
(the geometric middle and the range), ">= 1.4 h" when only a lower end is
known, "rough" for a wide or thin-data estimate, "possibly stalled" when the
output has been silent for 10x a step, "starts in ~3.0 h (SLURM)" for a
pending job (SLURM's own start estimate), and a blank cell when nothing is
known. A footer adds when each listed array's last task should end and the
rest of each workflow. SLURM is asked once for the whole table (the cached
`squeue` the Cluster view uses, plus one `sacct` when an array is listed).

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
| `--remove-zombie-jobs` | Cancel a calculation's own SLURM job (found from its own `<name>-<id>.o` log and confirmed with SLURM) whose output already shows a failure |

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
mace opt2d12 --out-file OUT [OUT ...] | --d12-file DECK | --directory DIR [options]
```

| Option | What it does |
|---|---|
| `--out-file OUT ...` | Output(s) to start from; several or a glob form one batch |
| `--d12-file FILE` | The parent deck for a single output (default: the same name beside it). Without `--out-file` the deck is the parent: the new deck has its geometry records and its settings, named `<deck>_<calc>_<functional>.d12` (refused for EXTERNAL decks and decks that edit their geometry, e.g. SUPERCEL: use their run's `.out`) |
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
| `--ion-charges`, `--net-charge`, `--site-states`, `--chemod`, `--fragment-bond-scale`, `--spinlock`, `--spinlock-cycles` | Charged systems, recomputed for the new geometry (as for `mace convert`). A requested SPINLOCK replaces the parent's, and a net charge that changes the electron-count parity needs `--spinlock` when the parent carries one. `--spinlock` / `--spinlock-cycles` alone set the spin state of the parent's charge state (checked against its electron count); `--chemod` and `--fragment-bond-scale` need a charge change and are refused without one |

```bash
mace opt2d12 --out-file optimized.out --calc-type SP --non-interactive
mace opt2d12 --directory opts --config-file sp_template.json --yes
mace opt2d12 --d12-file deck.d12 --calc-type OPT          # a deck as the parent
```

More: [One template, many structures](../DOCUMENTATION.md#one-d12-template-for-many-structures-mace-opt2d12),
and [Spin-orbit coupling (SOC) decks](#spin-orbit-coupling-soc-decks).

### mace opt2d3

D3 property decks (BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL, ORBITALS) from a
finished run. ORBITALS writes the parent's orbitals as Molden files: see
[Orbitals (ORBITALS)](#orbitals-orbitals).

```
mace opt2d3 --input OUT [--calc-type TYPE] [options]   |   mace opt2d3 --batch [options]
```

| Option | What it does |
|---|---|
| `-i`, `--input OUT` | The CRYSTAL output (.out or .log) |
| `-t`, `--calc-type TYPE` (also `--calc_type`) | BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL, CHARGE+POTENTIAL or ORBITALS |
| `-b`, `--batch` | Every .out here |
| `-s`, `--shared-settings` | One set of answers for the batch |
| `-c`, `--config-file FILE` | Load a saved D3 configuration |
| `--save-config` | Save the configuration after the questions |
| `--options-file FILE` | The file name for `--save-config` |
| `--list-configs` | List the saved configurations |
| `-o`, `--output-dir DIR` | Where the decks go (default: beside the input) |
| `--static-kpath` | Without seekpath, write the static HPKOT path for a 3D BAND (same as `MACE_STATIC_KPATH=1`) |
| `--edit-kpath` | BAND: ask "Edit the band k-path?" when the answers are piped in too (at a terminal it is always asked) |

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
| `--overwrite-sh` | Write new job scripts instead of reusing existing `.sh` files; also required for `-i`/`--walltime`/`--exclude-*` to apply to a script edited by hand or written before 1.1.5 |
| `--walltime TIME` | Walltime for these jobs (e.g. `2:00:00`) |
| `--track` | Record the jobs in materials.db (monitor, analysis, recovery) |
| `--progress TEMPLATE` | Plan the steps that follow and run them as each job ends (a template from `mace workflow --show-templates`, or `interactive`); implies `--track` |
| `--progress-plan PLAN` | Submit the decks as the first step of a plan written by `mace workflow --spec SPEC --write-plan` (decks the plan does not list are left out of a folder); they then progress as the plan says; implies `--track`, and the plan's `job_arrays` setting applies |
| `--array` | Decks with identical resource requests share one SLURM job array |
| `--max-running N` | With `--array`: at most N tasks of each array at once |
| `-y`, `--yes` | Without `--track`: submit the decks that do not fit in your queue budget from the reserve without asking |
| `--exclude-type TYPE` | Exclude nodes named TYPE-<number> |
| `--exclude-nodes LIST` | Exclude these nodes (`node-042,node-050` or `node-[042-050]`) |
| `--exclude-amd20` | MSU HPCC only: exclude the AMD20 nodes |
| `--exclude-mendoza` | MSU HPCC only: exclude the Mendoza group's nodes |

The exclusion options are mutually exclusive, and the MSU HPCC rows are
listed in `--help` only on MSU HPCC. The job scripts follow your
[site settings](#site-settings-mace-site-sitetoml).

A deck whose name has spaces or shell characters is refused, and so is a
`.d3` whose `.f9` is missing or empty.

Every job counts against your queue budget ([mace queue](#mace-queue)).
With `--track` (and `--progress`) a deck that does not fit is held and
submitted as slots free up. Without `--track` there is nothing to hold it in:
MACE lists the decks it did not submit and asks before it uses the reserve
for them (`--yes` uses it without asking; with no terminal and no `--yes` it
does not). With `--array` the tasks that fit are submitted as the array and
the rest held (or listed).

**Exit status:** 0 when every job was submitted; 1 when any job was not
submitted or was refused (also with `--nosubmit` when a script cannot be
written).

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
mace analyze --convergence FILE.out --energy [--segments continuous|separate] [--relative-to REF] [--unit UNIT]
mace analyze --convergence FILE.out --scf [--step latest|first|N|point:N] [--unit UNIT]
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
| `--register` | First record the folder's completed calculations (materials and calculations rows), so `mace database` and its exports list them. On by default for a folder; not for `--output-file` |
| `--no-register` | Store the properties only: outputs without a calculation record share one set of rows per material, and `mace database` lists no material for them |
| `--dry-run` | Write nothing; list each output as new, changed, unchanged, ... |
| `--json` | With `--dry-run` or `--convergence`: JSON |
| `--skip-unchanged` | Only new and changed outputs of tracked calculations |
| `--backup` | Back up the database into `<db dir>/backups/` first |
| `--keep-backups N` | With `--backup`: keep N (default 3) |
| `--progress-json` | One JSON line per file on stdout (a build that continues one that stopped starts with `{"resumed": {"already": N, "total": M}}`) |
| `--kappa-lattice VALUE\|FILE` | TRANSPORT runs: the lattice thermal conductivity κ_L in W/(m·K) (one value, or a CSV of `T,kappa_L` rows) for the total ZT |
| `--tau-fs FS` | With `--kappa-lattice`: the relaxation time to evaluate the total ZT at (default: the run's RELAXTIM) |
| `--convergence FILE` | Per optimization point: SCF cycles, energy change, gradients and displacements against their thresholds |
| `--energy` | With `--convergence`: the energy along the run as the GUI's energy card copies it (point, FINALRUN segment, frame, E in Ha, ΔE) |
| `--segments MODE` | With `--energy`: `continuous` (default: one reference across the FINALRUN restarts) or `separate` (one per segment) |
| `--relative-to REF` | With `--energy`: `final` (default), `first`, `minimum` or `absolute` |
| `--unit UNIT` | With `--energy`: `meV` (default), `eV`, `Ha`, `kJ/mol` or `kcal/mol`; with `--scf`: `Ha` (default) or the same |
| `--scf` | With `--convergence`: the SCF convergence cycle by cycle (energy, its change, the spin density, the notable cycles: Fock/KS mixing, DIIS, the spin lock's release, a conducting state), the TOLDEE it ran against, and an optimization's cycles per point; the GUI's SCF card copies the same columns |
| `--step STEP` | With `--scf`: `latest` (default), `first`, a block number, or `point:N` |

For a TRANSPORT run the database gets the peak Seebeck coefficient, power
factor PF = S²σ and **electronic** figure of merit ZTₑ along each axis and
of the averaged tensors. ZTₑ = PF·T/κₑ counts the electrons only: the
lattice thermal conductivity is not in it, so it is an upper bound on the
material's ZT, and it grows without limit where T·σ·S² approaches κ₀ (values
of 10-20 are common near a band edge). κ₀ (KAPPA.dat; the manual's κₑ of
Eq. 14.3 is this zero-field coefficient) and the electronic
κₑ = κ₀ − T·σ·S² at the averaged peak are stored as
`transport_kappa0_at_zt_max` and `transport_kappa_electronic_at_zt_max`.

S, σ and κ₀ are tensors (CRYSTAL23 manual pp. 314-315: SEEBECK.dat has the
nine components, SIGMA.dat and KAPPA.dat the six of a symmetric tensor; a
slab's files the in-plane ones, `sigma_xx sigma_xy sigma_yy`). PF and ZTₑ
of tensors averaged over their diagonal are no physical quantity of an
anisotropic cell, so each axis gets its own peak, from that diagonal element
alone:

| Key | Meaning |
|---|---|
| `transport_power_factor_max_xx`, `_yy`, `_zz` | peak PF along each axis (W/(m·K²)) |
| `transport_zt_electronic_max_xx`, `_yy`, `_zz` | peak ZTₑ along each axis |
| `transport_power_factor_best_axis`, `transport_zt_electronic_best_axis` | the axis of the highest value |
| `transport_power_factor_max_avg`, `transport_zt_electronic_max_avg` | the peaks of the averaged tensors (an average, not an axis; `transport_zt_electronic_max` and `transport_zt_max` before 1.1.6) |
| `transport_tensor_average` | the average those are: `(xx+yy+zz)/3`, or `(xx+yy)/2, in plane (a slab)` |

A slab has xx and yy only (its zz is along the vacuum). Each peak is found
with the same rule: points where |σ| is at least 1e-3 of that quantity's
peak |σ| (BoltzTraP's determinant spikes are left out) and κₑ > 0. The afi
framework of the corpus gives ZTₑ 19.55 for the averaged tensors and 15.44,
15.44 and 20.46 along xx, yy and zz. `mace analyze --extract-properties`
prints the axes of each TRANSPORT run:

```
   PF (W/(m·K²)) peak: xx 0.007034, yy 0.007034, zz 0.01161 (best zz); of the averaged tensors 0.007067
   ZTₑ (no lattice κ) peak: xx 15.44, yy 15.44, zz 20.46 (best zz); of the averaged tensors 19.55
   total ZT peak: xx 1.948, yy 1.948, zz 2.9 (best zz); of the averaged tensors 2.025
```

(the last line with `--kappa-lattice 1.0`).

A build that stops before its end (a CPU-time limit, a kill, Ctrl+C) keeps
every output it finished. The next build with `--skip-unchanged` skips them
and says so: "Resumed: 3 of 10 already done" (in its summary too).

**Total ZT.** With `--kappa-lattice`, MACE also stores the total figure of merit
ZT = S²σT/(κₑ + κ_L), with κ_L the lattice thermal conductivity you supply in
W/(m·K). Give one value, or a CSV file of `T,kappa_L` rows (an optional
`T,kappa_L` header and `#` comment lines are allowed). MACE interpolates the table
linearly between its rows and never extrapolates. If the table does not reach
every temperature of a run, MACE stores no total for that run, names the missing
temperatures, removes any total an earlier extraction stored for it, and leaves
the electronic keys as they are. One κ_L applies to every TRANSPORT run of the
extraction; when the runs are of more than one material, MACE warns at the end
and names them.

For an open-shell (spin-polarized) run, CRYSTAL writes SIGMA.dat, SIGMAS.dat
and KAPPA.dat as an alpha block and a beta block (`# 1 [Alpha ...]`,
`# 1 [Beta ...]`; CRYSTAL23 manual, BOLTZTRA), and SEEBECK.dat once, as it is
not spin dependent. The two spin channels conduct in parallel, so MACE uses
σ = σα + σβ and κ₀ = κ₀α + κ₀β for every transport key, the electronic ones
included (the `transport` record says `"spin_channels": "alpha+beta"`).

The relaxation time matters here. CRYSTAL's transport distribution function
carries the constant relaxation time τ of RELAXTIM (CRYSTAL23 manual Eq. 14.4,
default 10 fs), and SIGMA.dat and KAPPA.dat are written in SI units, so they
hold σ and κ₀ at that τ, not per unit τ. ZTₑ is a ratio in which τ cancels, but
the total does not, because κ_L has no τ. MACE reads τ from the run's `.out`
(`RELAXATION TIME (fs)`). `--tau-fs` evaluates the total at another τ instead,
scaling σ and κ₀ by τ/τ_run; S does not depend on τ. If the `.out` does not give
τ, MACE stores no total (and removes an earlier one).

The total adds these keys and never changes the electronic ones:

| Key | Meaning |
|---|---|
| `transport_zt_total_max_xx`, `_yy`, `_zz`, `transport_zt_total_best_axis` | the highest ZT along each axis, and the best axis |
| `transport_zt_total_max_avg` | the highest ZT of the averaged tensors over the run's (T, μ) points (`transport_zt_total_max` before 1.1.6) |
| `transport_zt_total_max_temperature_k`, `transport_zt_total_max_mu_ev` | where that peak is |
| `transport_zt_total_kappa_lattice_at_max`, `transport_zt_total_kappa_electronic_at_max` | κ_L and κₑ there (W/(m·K)) |
| `transport_zt_total_tau_fs`, `transport_zt_total_tau_source` | the τ used, and whether it was the run's or given |
| `transport_zt_total_lattice_source` | where κ_L came from: `constant 1.5 W/(m·K)`, or the table's path, sha256, rows and range |
| `transport_zt_total_inputs` | the size and first/last 64 KiB sha1 of the `.out` and `.dat` files the total was computed from |
| `transport_zt_total` | all of the above but the last as one JSON record |

A later extraction without `--kappa-lattice` keeps these rows while the run's
`.out` and `.dat` files are the ones they were computed from (the database
build's change detection: a `touch` changes nothing). If the files have
changed, it removes them and says so. Extract again with `--kappa-lattice` to
replace them. The GUI's transport plot shows the same
numbers (its ZT tab).

```bash
mace analyze --extract-properties runs/ --register --skip-unchanged --backup
mace analyze --extract-properties runs/ --kappa-lattice 1.5
mace analyze --extract-properties runs/ --kappa-lattice kappa_L.csv --tau-fs 10
mace analyze --convergence mat1_opt.out
mace analyze --convergence mat1_opt.out --energy --segments separate
mace analyze --convergence mat1_sp.out --scf --unit meV
```

### mace plotting

Plots from CRYSTAL outputs: band structures, DOS, structures, cube
volumetrics, orbitals (ORBITALS), FREQ normal modes and IR/Raman spectra.
With no mode flag it asks.

**Exit status:** 0 when every requested figure was drawn; 1 when a requested
kind has no files or any figure was not drawn (a file error, a band + DOS
file without its partner, `--energy-reference vacuum` without the POTC
files). One bad file does not stop the others; the run lists the figures it
could not draw at the end, on stderr.

```
mace plotting [MODE] [FILE ...] [-d DIR] [-o DIR] [options]
```

| Option | What it does |
|---|---|
| `--band`, `--dos`, `--band-dos`, `--structure`, `--cube`, `--orbitals`, `--freq` (or `--vibmodes`), `--ir`, `--raman`, `--spectra`, `--bz`, `--all` | What to plot (`--band-dos`: each band structure with its run's DOS in one figure on a shared energy axis, the GUI's band + DOS view, written as `<material>.BAND_DOS.png`; its FILE is either file of the run; `--spectra`: IR and Raman; `--bz`: the first Brillouin zone with the k-path; `--all`: everything found, band + DOS figures too) |
| `-d`, `--directory DIR` | Where to look (default: here) |
| `-o`, `--output DIR` | Where the plots go (band and DOS: beside the data when not given) |
| `--backend plotly\|matplotlib` | Band, DOS and band + DOS renderer. `plotly` (default when plotly and kaleido can export here) draws the GUI's figure: the same code, colors, line widths, legends and k labels as the GUI's plot pane at a light plot background, written by kaleido (PNG at 300 dpi, `--dpi`; the size in the PNG's metadata). `matplotlib` draws the figures of MACE 1.1.5 and earlier (with the defaults below aligned to the GUI's); MACE falls back to it, saying so once with the reason and the install line, where plotly or kaleido is missing (an HPCC's plain `python3`) or cannot export (plotly 7 needs kaleido 1, which needs Chrome; MACE pins `plotly>=5.0.0,<6.2` with `kaleido==0.2.1`). Also `mace-folder.toml` / `site.toml` `[plots] backend` and `MACE_PLOT_BACKEND` |
| `--style NAME` | Plot style, as the GUI's Settings › Appearance › Plot style: `default`, `publication` (Okabe–Ito after black, black frame, outside ticks, no grid), `vesta` (cube maps and isosurfaces: VESTA's look; other plots draw `default`), `presentation` (large text, thick lines), `grayscale` (print: black and grays, line styles and markers) or `colorblind` (Okabe–Ito, cividis maps). Band, DOS, gallery, IR/Raman and cube figures |
| `--spin-colors auto\|mace\|blue-red\|black-vermilion\|custom` | Band and DOS: the spin color scheme, as the GUI's Spin colors. `auto` (default): the style's: blue/red (#1f4e9c / #c0392b, dark plot #7aa7f0 / #f08a7c) in `default` and `presentation`, black/vermilion in `publication`, the style's own pair in `grayscale` and `colorblind`; `mace` (MACE pink/cyan, #e0218a / #00a0a4); `custom`: `--spin-up-color`/`--spin-down-color` |
| `--spin-up-color HEX`, `--spin-down-color HEX` | Band and DOS: a custom spin pair (#rrggbb; given alone, they replace that color of the style's pair). The DOS total is drawn in the pair too |
| `--band-color HEX` | Band and DOS: the color of a run that is not spin-polarized (closed shell, or α = β): its bands and one-sided DOS total, with no spin legend. Default: the style's, the plot's text color (black on a light plot). A spin-polarized run keeps the spin colors. Also `[plots] band_color` |
| `--spin-down-style dash\|dot\|dashdot\|solid` (or `--spin-style`), `--spin-down-solid` | Band and DOS: spin down's (β's) line style (default dash, for colorblind safety; spin up solid). With both spins on a DOS the dash means spin down only: projections are solid, told apart by color |
| `--dos-colors MODE` | DOS: which projections and their colors: `auto` (default: one element by orbital, two or three element × orbital, more by element), `total`, `element` (the element's color from the palette, darkened where too light), `orbital` (s, p, d, f hues), `element-orbital` (shades of each element's color, the strongest for the element's largest projection) |
| `--no-dos-total`, `--dos-total` | DOS: without the total (an ink line over a light fill of its color; with both spins the spin pair, each side filled lightly in its spin's color), or with it (the default; `--dos-total` overrides a `mace-folder.toml`'s `dos_total = false`) |
| `--dos-smooth EV` | DOS: Gaussian smoothing of the drawn curves, sigma in eV (display only, labeled in the plot; e.g. `0.05`, which takes the spikes off and keeps the structure). `0`: off (the default). `[plots] dos_smooth`, the GUI's DOS smoothing |
| `--dos-spin auto\|mirrored\|overlaid` (or `--spin-dos`) | DOS spin layout: `auto` (default) mirrors a spin-polarized run and draws a closed-shell or α = β run one-sided (an α = β run as the α + β total), as the GUI |
| `--units eV\|meV\|Ha\|kJ/mol\|kcal/mol` | Band and DOS energy unit (plotly; default eV, or the folder's `[units] energy`) |
| `--width IN`, `--height IN` | Band, DOS and band + DOS figure size in inches. Default: plotly the GUI pane's size (band and band + DOS 7.9 x 4.4 in, DOS 5.4 x 5.4 in, its text 9 pt); matplotlib 3.5 x 5 in, a band figure wider with many k labels. matplotlib widens a figure on the right when its legend needs room |
| `--dpi N` | PNG resolution in dots per inch, written into the PNG so LaTeX and Word place it at its size: band, DOS and band + DOS (default 300; `--backend matplotlib` band and DOS 600) and `--legend` (default 300) |
| `--format FMT` | Output format. Cube/FREQ: html (default), png, svg or pdf. Band, DOS, band + DOS: png, svg, pdf or html (plotly; default png and svg) |
| `--e-lower E`, `--e-upper E` | Band and DOS energy window in eV around the reference (default -6, 6, as the GUI) |
| `--energy-reference auto\|fermi\|vacuum` | Band and DOS: the energy zero. `auto` (default): the vacuum scale when the run's POTC files (`<material>_POTC.POTC.DAT` and `<material>_POTC.out`) are beside the data, else E − E<sub>F</sub>; `fermi`: E − E<sub>F</sub> even then; `vacuum`: the vacuum scale (a file without the POTC files is skipped with an error). The axis title names the reference |
| `--fermi-shift EV` | Band and DOS: move the energies by −EV, so the energy EV above E<sub>F</sub> is the reference (the GUI's Fermi shift: the dotted line and the window stay where they are). The axis title reads E − (E<sub>F</sub> + EV) (or E<sub>vac</sub>); default 0 |
| `--alpha A` | Band line opacity, matplotlib; below 1 overlapping bands stack darker (default 1) |
| `--no-gaps` | matplotlib: remove the gaps at k-path discontinuities (a jump's two labels share one tick, "H\|P", either way) |
| `--band-path FILE` | Band and `--bz`: plot along an edited k-path (JSON `band_path`, as `mace gui`'s k-path editor and `mace opt2d3` write it: Crystal_d3/README.md) |
| `--projection TYPE` | DOS, matplotlib: both, total, orbital or tm_orb (default: the projections of `--dos-colors`, as the GUI) |
| `--no-composition` | DOS: no VB/CB composition labels |
| `--supercell NX NY NZ` | Structure: supercell (default 2 2 2) |
| `--bond-cutoff A` | Structure: bond cutoff in Å (default 1.9) |
| `--color-by-coord` | Structure: color atoms by coordination |
| `--legend [FILE\|FOLDER ...]` | An element legend for figures: the elements of the files, or the union over files and folders (`-d` when none), as SVG (default), PNG or a TSV/CSV/JSON table (`--format`); written to `-o` (a folder: `element_legend.<ext>`, or a file name) |
| `--legend-layout vertical\|horizontal\|grid`, `--legend-columns N`, `--legend-shape circle\|square`, `--legend-swatch PX`, `--legend-font-size PX`, `--legend-label symbol\|name\|both`, `--legend-order z\|first\|alpha`, `--legend-text dark\|light\|#rrggbb`, `--legend-background none\|white\|black\|#rrggbb`, `--legend-no-outline`, `--legend-recursive` | Legend: layout, swatches, labels, sort order, colors and subfolders (`--dpi`: its PNG resolution) |
| `--palette NAME` | Element colors for atoms (structure SVGs, FREQ modes, cube overlays): a preset (`vesta`, `jmol`, `cvd`, `legacy`) or a palette saved in the GUI (View › Element Colors). Default: the default palette saved there, else VESTA |
| `--density`, `--esp`, `--spin` | Cube: the quantity (each implies `--cube`) |
| `--diff A B` | Cube: density difference A - B |
| `--iso V[,V2...]` | Cube: isovalue(s). Default: one surface at the GUI's default level (a ± pair for signed data) |
| `--view iso\|slice\|slice-all` | Cube: view (default iso) |
| `--slice AXIS POS` | Cube: one 2D slice, drawn as a flat map on Cartesian in-plane axes (a skewed cell keeps its parallelogram), equal x/y scale, the atoms within 2 bohr of the plane and the unit on the colorbar |
| `--colorscale NAME`, `--no-atoms`, `--bonds`, `--publication` | Cube: color scale and display |
| `--potential-colors chemistry\|physics` | Cube: electrostatic potential colors in isosurfaces, slices and potential-colored surfaces. `chemistry` (default, as GaussView and VMD): red = negative (electron-rich), blue = positive (electron-poor), the same pair as spin (+ blue); `physics`: red = positive (MACE 1.1.6 and earlier). The GUI's Settings › Viewport › Potential colors; also `[plots] potential_colors`. Spin and differences stay + blue, − red |
| `--clip P` | Cube slices: the color range runs from the (100 − P)th to the Pth percentile of the values, symmetric about 0 for spin and potential, so the nuclear spikes do not take the whole scale (default 99) |
| `--color-range LO,HI` | Cube slices: the color range outright (log10 units for a log-scale density; a negative low end needs `--color-range=-0.01,0.01`) |
| `--engine-args "..."` | Cube: extra flags for the cube engine |
| `--mo LIST` | Orbitals: comma-separated `HOMO`, `LUMO`, `HOMO-1`, `LUMO+2` or MO numbers (default `HOMO,LUMO`); one cube and plot each |
| `--k N` | Orbitals: the k point, numbered as in the ORBITALS `.out` (default: the k point of the valence-band top, else Γ) |
| `--orb-spin alpha\|beta` | Orbitals: the spin channel of an open-shell run (default alpha) |
| `--part re\|im\|abs2` | Orbitals: Re ψ (default, ± lobes), Im ψ (complex k points only) or \|ψ\|² |
| `--degenerate-sum` | Orbitals: sum \|ψ\|² over each degenerate set |
| `--cells NxMxL` | Orbitals: cells drawn along the periodic axes (default 3 for a polymer, 2x2 for a slab, 1x1x1 for a crystal) |
| `--quality preview\|normal\|fine` | Orbitals: grid spacing 0.40, 0.25 (default) or 0.15 bohr |
| `--list-orbitals` | Orbitals: the k points, the levels and the self-check; nothing is drawn |
| `--list-modes` | FREQ: the mode table only |
| `--mode N` | FREQ: this mode (default: the lowest real one) |
| `--all-modes` | FREQ: every mode in one HTML |
| `--gif` | FREQ: an animated GIF |
| `--amplitude F`, `--normalize`, `--static` | FREQ: displacement scale, normalized vectors, no animation |
| `--raman-mode MODE` | Raman: total, par_perp or all (default) |
| `--average` | Average conformer (`-confN_`) spectra per material |
| `--ir-column N` | IR intensity column (default 2, absorbance) |
| `--gallery KIND` | Many outputs in one grid figure: `band`, `dos`, `bandos` (band + DOS), `ir`, `raman`, `opt` (energy along each optimization), `scf` (SCF convergence) or `structure` (thumbnails). The FILEs and folders (searched recursively) make the set; at most 200 panes |
| `--cols N` | Gallery: columns (default 3, at most 8) |
| `--independent-axes` | Gallery: each pane its own ranges (default: shared; band and DOS panes always use `--e-lower`/`--e-upper`, and `--energy-reference`/`--fermi-shift`) |
| `--caption-csv PATH` | Gallery: also write the caption table (material id, formula, band gap, energy from the `materials.db` beside the files) |
| `--max N` | Gallery: at most N panes (default 200) |
| `--db PATH` | Gallery: the database for the captions (default: the `materials.db` beside the files, or up to three folders above) |

```bash
mace plotting
mace plotting --band -d ./band_calcs -o ./plots
mace plotting --dos run_doss.DOSS.DAT --style publication --dos-colors element-orbital
mace plotting --band-dos                     # each band structure with its DOS, one figure
mace plotting --band --spin-colors blue-red --spin-down-solid
mace plotting --band --backend matplotlib    # the figures of MACE 1.1.5
mace plotting --freq mol_freq.out --mode 7 --gif
mace plotting --band Si_band.BAND.DAT --band-path edited_path.json
mace plotting --bz Si_band.BAND.DAT     # the zone of Si_band.out's cell, the path of Si_band.d3
mace plotting --gallery band BAND/ --cols 4 -o bands.svg
mace plotting --gallery bandos runs/ --caption-csv captions.csv
```

Atoms are drawn in the same element colors as the GUI: the default palette
saved in `~/.config/mace/element_colors.json` (`$XDG_CONFIG_HOME/mace/`),
which the GUI's View › Element Colors writes, or the one `--palette` names.
With no saved default it is VESTA, as in the GUI.

`--legend` writes the key for such a figure: one swatch and label per
element, in the same palette. The SVG keeps each element as its own group
with real text, so it stays editable in Inkscape, Illustrator or
PowerPoint; the table formats give symbol, name, Z, hex color and RGB for
other tools. The GUI's Export Element Legend makes the same file.

```bash
mace plotting --legend MgO.cif
mace plotting --legend runs/ "extra/*.cif" --legend-layout grid --legend-label both \
    --format png --dpi 600 --legend-background white -o figure_key.png
```

`--palette legacy` (or
the Legacy preset set as the default in the GUI) draws each plot in the
colors `mace plotting` used before 1.1.6.

#### Orbitals (ORBITALS)

An ORBITALS calculation (a properties step like BAND or DOSS: `mace opt2d3`,
the workflow planner, or the GUI's Calculation Setup, Batch Setup and
workflow builder) makes CRYSTAL23 write the orbitals of the parent SCF as
Molden files: one `<job>_orb.molden` for a molecule, and one
`<job>_orb_<nnnnn>_K<ijk>_real.molden` per k point for a polymer, slab or
crystal (plus a `_complex.molden` with the imaginary part at complex k
points). Every band is in every file. The deck is NEWK + ORBITALS:

```
NEWK
2 8          IS ISP (no line for a molecule)
1 0
ORBITALS
mgo_orbitals_orb   the file root (<deck stem>_orb)
1            ICAR, on its own line
0            ILOC, on its own line: always 0
END
```

ICAR and ILOC must be on separate lines: on one line CRYSTAL stops with
FORMAT ERROR and still exits 0, so MACE checks that the `.molden` files
exist before it calls the step complete. ORBITALS has no END of its own.

**Which k points.** NEWK cannot pick single k points, so MACE picks the
smallest net that holds the band edges, read from the parent's SCF output
(TOP OF VALENCE BANDS / BOTTOM OF VIRTUAL BANDS): IS 1 (Γ only, one file)
when both edges are at Γ, else IS 2 (Γ and the zone-boundary points, which
are all real). When an edge sits elsewhere (Si's conduction-band minimum
near X) the nearest real points are used and MACE says so. "All real k
points" asks for IS 2 directly; a custom IS is possible, but complex k
points of slabs and crystals cannot be drawn (below).

`mace plotting --orbitals` takes a run's `.molden` files (or finds them in
`-d`), shows the k points, the levels and the self-check, and draws the
orbitals you pick. Each orbital becomes a Gaussian cube in `<job>_orbitals/`
next to the job (`<material>_orbitals/` for MACE's own `<material>_orbitals`
step) (made once, reused afterwards), drawn by the cube engine
with the same formats and `--publication` styling as ECH3/POT3 cubes. Re ψ
and Im ψ are drawn as two surfaces, + blue and − red (as in the GUI), at ± the isovalue that
encloses 90% of \|ψ\|² (`--iso` sets your own); \|ψ\|² is drawn as a
density. The `.molden` files are never changed.

```bash
mace plotting --orbitals                                  # HOMO and LUMO, asking k and spin
mace plotting --orbitals --list-orbitals
mace plotting --orbitals si_orb_00001_K000_real.molden --mo HOMO --degenerate-sum
mace plotting --orbitals --k 2 --mo HOMO --part im --cells 4    # a polymer's complex k point
mace plotting --orbitals --mo HOMO --orb-spin beta --format png --publication
```

How MACE reads the files: the k vector comes from the k list of the
ORBITALS `.out` (the file names hold only labels), the cell from
`[CellAxes]` (`[Cell]` is unusable), and the atoms are moved to CRYSTAL's own
images from the `.out`, which the Bloch phases at zone-boundary k points
need. Each file is checked on load: the orbitals must be orthonormal under
the analytic overlap S(k) (C<sup>H</sup> S C = 1). Good files pass to
10<sup>−6</sup>–10<sup>−4</sup>; a file that fails is not drawn.

| | Molecule (0D) | Polymer (1D) | Slab (2D) | Crystal (3D) |
|---|---|---|---|---|
| Orbitals | real MOs | Bloch ψ<sub>nk</sub> | Bloch ψ<sub>nk</sub> | Bloch ψ<sub>nk</sub> |
| k points drawn | – | Γ, real and complex | Γ and real only | Γ and real only |
| What to view | Re ψ (± lobes) | Re ψ at Γ and real k; Re, Im or \|ψ\|² at complex k | Re ψ | Re ψ |
| Default box | atoms ± 5 bohr | 3 cells, vacuum ± 5 bohr | 2×2 cells, vacuum ± 5 bohr | 1×1×1 cell |

Caveats:

- **Complex k points of slabs and crystals** are not offered: CRYSTAL23's
  `_complex` coefficients there fail the orthonormality check (bcc Fe's are
  all zeros), so no picture would be right. Polymers' complex k points pass.
- **Degenerate levels** (Si's valence-band top, O<sub>2</sub>'s π\*) mix
  arbitrarily inside their set; one member is only one choice of many.
  `--degenerate-sum` draws the summed \|ψ\|², which matches CRYSTAL's own
  band density (PBAN). Single members are gauge-dependent; MACE makes the
  largest coefficient positive so the colors repeat.
- **Spin.** Open-shell runs hold both spins in each file; pick one with
  `--orb-spin`. A closed-shell periodic file lists 1.0 per band; that is
  2 electrons. Metals have fractional occupations: the levels are counted
  from the Fermi level.
- **ECPs.** The file gives the full nuclear charge and no core; electron
  counts come from the `.out`.
- **Two-component (SOC) runs are refused.** CRYSTAL23 cannot write
  two-component orbitals: NEWK and ORBITALS crash on a TWOCOMPON
  wavefunction (on the SCF k net ORBITALS writes an empty file and dies).
  Run the same basis one-component instead: the SOC basis sets on STUTSC
  give valid ORBITALS files in a 1c run.
- **Wannier functions** are not made through ORBITALS (CRYSTAL's ILOC 1
  output is unusable); MACE always writes ILOC 0.
- **File count and size.** Each periodic run writes (real points + 2 × complex
  points) files, each about nbf² × 22 bytes (nbf 36: 33 KB; nbf 1000: 22 MB;
  nbf 3000: 200 MB). A cube is about 13 bytes per grid point (1-3 MB at the
  default quality). With HPCC's 1M-file quota, keep IS small; MACE warns
  before a run or a request that writes many or large files, and copies back
  only the `.molden` files (not the `fort.*.peN` files Pproperties leaves).
- **Basis note.** POB-DZVP-REV2 diverged at B3LYP on the polyyne chain used to
  test this (energies of +3·10<sup>4</sup> Ha and more in the first cycle, even
  with TOLINTEG 8 8 8 8 20) while POB-DZVP converged; this is an SCF issue of
  that basis on that system, not of ORBITALS.

### mace database

Query materials.db: statistics, searches, properties, exports, comparisons
and checks.

```
mace database [--action ACTION] [options]
```

| Option | What it does |
|---|---|
| `--action ACTION` | stats (default), query, properties, export, compare, missing, correlate, distribution, workflow, validate, history, interactive or aggregate |
| `--db-path PATH` | The database file (default: the workflow context's, else `materials.db`); it must exist |
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

Excel export needs openpyxl (`pip install 'openpyxl<4'`, listed in
`requirements-optional.txt`); CSV, JSON, LaTeX and HTML need nothing extra.

#### Database property keys

Every property name the extractor writes is listed, with its category, unit,
value type and meaning, in one registry: `mace/utils/property_keys.py`
(`KEYS`). The `--category` names are lattice, structural, crystallographic,
energy, electronic, band_structure, density_of_states, population_analysis,
optimization, frequency, thermodynamics, transport, charge_potential,
orbitals, scf_settings and computational. Some keys to know:

| Key | Meaning |
|---|---|
| `total_energy_corrected_au` / `_ev` | total energy + D3 + gCP: the energy to compare |
| `band_gap`, `band_gap_type`, `direct_band_gap`, `indirect_band_gap` | the last SCF's gap (eV) |
| `band_fundamental_gap_ev`, `band_direct_gap_ev`, `band_gap_is_direct` | the gap along a BAND run's path |
| `electronic_classification`, `classification_gap_source` | metal (gap < 0.01 eV), semiconductor (up to 4 eV) or insulator, from `band_gap` (scf) or the band structure; semimetal needs a band structure and a DOS |
| `spin_polarized`, `magnetic_classification` | an unrestricted run; one with a net or atomic spin |
| `fermi_energy_au` | the SCF's Fermi energy, else the one a BAND or DOSS output prints (Hartree) |
| `dos_at_fermi_total`, `dos_at_fermi_alpha`, `dos_at_fermi_beta` | DOS at E_F (states/eV per cell) |
| `primitive_a` ... `primitive_gamma`, `primitive_cell_volume` / `_area` / `_length` | the final geometry's cell (an OPT's optimized one); `initial_*` the input's; none along a non-periodic direction |
| `space_group_number`, `layer_group_number` | none for a molecule; a slab's layer group as well |
| `lowest_frequency_cm` | the lowest vibration without the rigid-body modes (negative: imaginary) |
| `cpu_time_total_s`, `wall_time_s` | TOTAL CPU TIME (all processes) and the last TELAPSE |
| `scf_tolinteg` | the five TOLINTEG exponents as text (`"8 8 8 9 24"`) when the deck sets TOLINTEG |
| `calculation_type` | OPT, SP, FREQ, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL or ORBITALS |
| `mulliken_alpha_plus_beta` / `_minus_beta` | per atom `mulliken_population` (electrons, not the net charge) and orbital populations |

The large JSON values (`neighbor_analysis`, `final_neighbor_analysis`,
`overlap_population*`, `processed_*`) are kept in the `property_blobs` table;
`mace database --action properties` and the GUI's inspector show them with
the rest, and an export writes them only when named in its property list.

**Renamed keys (1.1.6).** A key renamed or merged in 1.1.6 is still read
under its old name until 1.1.7: a filter, query, export property list or the
GUI's property filter that names it gets the new key, with a note on
stderr, and a database not yet extracted again shows its old rows under the
new names. `property_keys.ALIASES` has the full map; the main ones:

| Old name | Now |
|---|---|
| `total_cpu_time` | `cpu_time_total_s` |
| `fermi_energy`, `fermi_energy_band`, `fermi_energy_dos` | `fermi_energy_au` |
| `is_spin_polarized`, `chargepot_spin_polarized` | `spin_polarized` |
| `classification_band_gap` | `band_gap` |
| `band_dat_band_gap_ev`, `band_indirect_gap_ev`, `band_gap_advanced_ev` | `band_fundamental_gap_ev` |
| `doss_dat_band_gap_ev` | `dos_band_gap_from_dos_ev` |
| `dos_total_at_fermi`, `dos_at_fermi_both_spins_ev` | `dos_at_fermi_total` |
| `doss_dat_metallic` | `dos_is_metallic` |
| `electronic_classification_advanced` | `electronic_classification` |
| `final_primitive_*`, `final_crystallographic_*`, `final_density`, `final_atomic_positions` | the plain names |
| `final_atomic_position_axes`, `final_atoms_count` | `atomic_position_axes`, `atoms_in_unit_cell` |
| `scf_diis_enabled` | `scf_diis` |
| `transport_zt_electronic_max`, `transport_zt_max` | `transport_zt_electronic_max_avg` |
| `transport_power_factor_max`, `transport_zt_total_max` | `transport_power_factor_max_avg`, `transport_zt_total_max_avg` |

No longer stored (no replacement): `conductivity_type` (use
`electronic_classification`), `insulator_type`, `semiconductor_type`,
`magnetic_type`, `conductivity_type_advanced`,
`conductivity_estimate_advanced`, `is_semimetal_advanced`,
`has_dielectric_data`, `dielectric_extraction_note`,
`dos_analysis_gcrit_factor`, `band_structure_k_range_min`,
`advanced_analysis_available`, `advanced_analysis_method`,
`real_electron_mobility`, `real_hole_mobility`, `band_gap_advanced_ha`,
`dos_at_fermi_level` and `doss_dat_at_fermi` (both per Hartree; use
`dos_at_fermi_total`), and `band_start` when it is 1.

**The 1.1.6 schema.** The first time MACE 1.1.6 opens a database an earlier
version built, it copies it to `<name>_pre116_backup_<time>.db` beside it
(never pruned), removes duplicate property rows (keeping the newest), adds a
unique (material, calculation, property) index and an index on `calc_id`,
and moves the large JSON values to `property_blobs`, in one transaction
under the database's write lock (`PRAGMA user_version` 1 records it). If the
backup fails, nothing changes and MACE tries again next time. MACE 1.1.5
still reads a migrated database, without the values moved to
`property_blobs`; write it only with 1.1.6 (restore the backup to go back).

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
| `--check` | Check ssh, its config, keys, agent and permissions for this profile and try each hop with only `true` (read-only); OK/WARN/FAIL lines, each problem with its fix |
| `--no-net` | With `--check`: no ssh logins or DNS lookups |
| `--fix-config` | Print a suggested `~/.ssh/config` block for this profile (never written) |
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
mace site set KEY VALUE | unset KEY
mace site set --folder [DIR] KEY VALUE | unset --folder [DIR] KEY
mace site suggest DECK... [--type T] [--json]
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
| `--folder [DIR]` | set / unset: a project folder's `mace-folder.toml` (DIR, default here) instead of your `site.toml`; see "Folder settings from the shell" |
| `set KEY VALUE`, `unset KEY` | Change one setting in `~/.config/mace/site.toml` (e.g. `resources.FREQ.ntasks 64`) |
| `suggest DECK...` | The size-aware job size for these decks, and why ([resources] auto) |
| `--type T` | suggest: size the decks as this calculation type (OPT, SP, FREQ, BAND, DOSS, ...) |

More: [Site settings](#site-settings-mace-site-sitetoml).

### mace help

`mace help` prints the grouped command list, `mace help GROUP` one group
(workflows, monitoring, manual, results, special, gui, setup, aliases), and
`mace help COMMAND` that command's `--help`.

### mace version

Prints the MACE version, as `mace --version`.

### mace update

Moves this MACE git checkout (and, with `--remote`, a cluster's) to the
newest release, preview or main, safely: no forced moves, no resets, refused
with local changes or local commits. [update.md](update.md) is its guide.

```
mace update [--channel release|preview|latest] [--check [--json]] [--dry-run] [--yes] [--pip|--no-pip] [--local] [--remote [PROFILE]] [--remote-only] [--home HOME]
```

| Option | What it does |
|---|---|
| `--channel C` | `release` (default: the newest vX.Y.Z tag), `preview` (the newest vX.Y.Z-preview or vX.Y.Z tag) or `latest` (origin/main) |
| `--check` | Only report the current and the available versions |
| `--json` | With `--check`: one JSON object for `--channel` on stdout (what MACE GUI's update notice reads; [update.md](update.md#from-mace-gui)) |
| `--dry-run` | Show what would change; change nothing |
| `--yes`, `-y` | Do not ask before updating (and never run pip unless `--pip`) |
| `--pip`, `--no-pip` | When requirements changed: run pip without asking (`--user` for a system or module Python), or only print the pip command |
| `--local` | Update this computer's checkout (the default without `--remote`; with it, both) |
| `--remote [PROFILE]`, `--remote-only` | Also (or only) the MACE on the cluster of this `mace gui connect` profile |
| `--home HOME` | The checkout to update (default: the one this mace runs from) |

### mace credits

Prints the development team, as `mace --credits`.

### mace status

The same as `mace monitor --status`; its arguments (e.g. `--detailed`) are
passed on.

## Spin-orbit coupling (SOC) decks

*New in 1.1.6.* A two-component spin-orbit single point (CRYSTAL23 manual
ch. 6) is written only when you ask for it, with keys in a saved-answers
file:

```bash
# a new structure: "soc": true in the cif2d12 options file
mace convert --cif_dir cifs/ --batch --options_file soc_options.json
# a finished scalar run: "soc": true in an opt2d12 template, an SP deck
mace opt2d12 --out-file PbTe_opt.out --config-file soc_sp.json --yes
```

A `mace convert` options file needs the whole method, since a CIF has no
parent to take it from. This one works as written (an explicit ECP basis and
a functional the two-component SCF accepts; `"basis_set"` takes a bundled
library by name):

```json
{"method": "DFT", "dft_functional": "PBE0", "use_dispersion": true,
 "basis_set_type": "EXTERNAL", "basis_set": "full.basis.triplezeta",
 "calculation_type": "SP", "soc": true, "smear": 0.005}
```

An `opt2d12` template keeps everything else from the parent, so it needs only
the SOC keys (the parent must have an explicit ECP basis and a 2c functional):

```json
{"calculation_type": "SP", "soc": true, "smear": 0.005}
```

When a deck cannot be made SOC, one refusal lists every change it needs with
the keys to set (for the default settings: an EXTERNAL basis instead of
BASISSET, and a functional other than HSE06).

`mace workflow` offers it too: the SP step's settings menu has "4: Spin-orbit
coupling (two-component SOC single point)", and a workflow spec takes the
same keys as an SP step's `overrides` (`"steps": {"SP_2": {"overrides":
{"soc": true, "smear": 0.005}}}`). The GUI sets them in Calculation Setup,
Batch Setup and the workflow builder ([gui.md](gui.md#calculation-setup)):
it writes these keys into the settings file and runs the command above on
it, so its deck is the command's deck, byte for byte.

**What MACE writes.** The single-point deck the command would write anyway,
then: `TWOCOMPON` / `SOC` / `END` before `SCFDIR`; each ECP atom's `INPUT`
potential replaced by the same potential with its spin-orbit part (`INPSOC`,
manual pp. 84-87), valence shells unchanged; DIIS and HISTDIIS left out (a
DIIS record stops a 2c run); the SHRINK Gilat net set equal to the Monkhorst
net; FMIXING 50 and at least 200 SCF cycles. The log names every change.
The deck is named apart from the scalar deck of the same structure, so the two
never overwrite each other: `opt2d12` writes `<stem>_sp_<functional>_soc_optimized.d12`
(the scalar SP is `<stem>_sp_<functional>_optimized.d12`), and `mace convert`
puts `_SOC` after the functional (`Au_CRYSTAL_SP_symm_PBE0_SOC_<basis>.d12`).
Either tool says so when it replaces a deck that is already there.

**Single points only.** CRYSTAL23 has no geometry optimization, frequencies or
response properties in the two-component basis (manual p. 166), and accepts
only the keywords chapter 6 lists. MACE refuses, writing nothing:

- an OPT or FREQ deck with `"soc": true`, and an OPT or FREQ child of a SOC
  run (optimize and run frequencies on the scalar structure first, then make
  the SOC single point from it). Measured on HPCC: a TWOCOMPON deck with
  OPTGEOM stops at input, and the reason is printed only in `fort.87`
  ("TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM"), not in the `.out`;
- ANDERSON and BROYDEN (not in the chapter 6 list), a DFTD3 block, range-
  separated hybrids (HSE06), meta-GGAs and the 3c methods (pure and global
  hybrid LDA/GGA functionals only; the `-D3` forms of BLYP, PBE, B3LYP, PBE0
  and PW1PW work). CRYSTAL itself does not refuse ANDERSON in a 2c deck:
  measured on HPCC, a PbTe SOC deck with ANDERSON diverged and stopped with
  "FERMI ENERGY NOT IN INTERVAL", where the same deck without it converged in
  9 cycles;
- `PRINTOUT` with `EIGENVAL`: in a 2c run on HPCC it printed garbage symmetry
  labels, split the eigenvalues into alpha and beta sets and died with a
  segfault after the SCF. The band structure of a SOC run comes from a BAND
  properties run;
- GUESSPSO or GUESSPNOSO outside the TWOCOMPON block: CRYSTAL stops on either
  at input ("KEYWORD GUESSPSO NOT ALLOWED", measured). MACE writes them
  inside the block itself (`"soc_guess"`, below); `mace preflight` fails and
  `mace submit` (also with `--nosubmit`) refuses a deck that has one outside;
- BASISSET (an internal basis): SOC needs the explicit ECP records;
- GHOSTS, CHARGED and other basis-input records after `99 0` (CHEMOD is kept:
  the manual pairs it with the 2c guess, p. 172);
- DOPING and SPINLOCK, which chapter 6 does not list either. The
  charged-system options (`--ion-charges`, `--net-charge`, `--site-states`,
  `--chemod`, `--spinlock`) add their records after the SOC deck is written,
  and these rules apply to them too: `mace convert` keeps a neutral CHEMOD
  and refuses a deck that would get CHARGED, DOPING or SPINLOCK (or CHEMOD
  with `"soc_basis"`). `mace opt2d12` refuses the charged-system options on a
  SOC deck.

**CHEMOD works in 2c.** Measured on HPCC: with CHEMOD the initial guess
changes, and the converged energy matches the deck without it within 6e-7
Ha, with the default guess and with GUESSPATNC alike.

**Which basis sets and elements work.** CRYSTAL puts SOC only in the ECP
(manual p. 175), so an atom gets a spin-orbit term only when its basis has
an ECP that is, number for number, the scalar part of one of the Stuttgart
spin-orbit ECPs MACE carries. MACE never pairs a valence basis with a
potential it was not published with: any other ECP is refused.

| Basis family | Atoms that get a spin-orbit term |
|---|---|
| `full.basis.triplezeta`, `full.basis.doublezeta` | Rb-Mo, Ru-I, Cs-Tm, Lu-Bi. Yb is refused (no spin-orbit ECP for its core). |
| `full.basis.triplezeta.rev2022` | The same, except La (its 46-electron core is not the spin-orbit ECP's) and Yb. |
| `stuttgart` | Rb-Sb, I, La-Tm, Lu, Bi, Po. Te, Cs, Ba and Hf-Pb have other potentials there and are refused. Below Rb the writers take the all-electron records. |
| `pob-tzvp-rev2-soc` (`"soc_basis"`, below) | Rb-Mo, Ru-I, Cs, Ba, Hf-Bi. |

Actinides are refused in every family (no spin-orbit ECP).

**All-electron atoms get no spin-orbit term.** H-Br and Tc are all-electron in
the `full.basis` families, so they keep their basis and their spin-orbit
splitting is simply missing. For light atoms that matters little, and MACE
notes them. From Ga on it is a warning naming the atoms: the 4p splitting of
Ga-Br is about 0.3-0.5 eV, so InAs, GaAs, CsPbBr3 or Bi2Se3 lose part of their
SOC. MACE's `stuttgart` library also holds ECP10 records for Ga-Se (231-234)
that do carry the term (their provenance is not verified). No writer picks
them, since below Rb the writers take the all-electron record, so enter one
by hand: atom number 231 (Ga) in the geometry and that record as its basis
block, then convert the deck with `opt2d12`. MACE has no such ECP for Br or
Tc.

How much this matters was measured on HPCC with InAs: the valence-band
spin-orbit splitting (the split-off gap Delta0) was about 0.07 eV with the
all-electron As of `full.basis.triplezeta` and about 0.35 eV with the
stuttgart ECP10 As (233) and INPSOC; experiment gives 0.39 eV. The
all-electron atom loses about 80% of the splitting, while the total energy
moves by only about 0.6 mHa, so the energy does not show it. When the band
structure matters, use the ECP10 records for Ga-Se; MACE's warning says so.

**INPSOC, STUTSC or the official SOC basis.**

| Choice | Key | What it is |
|---|---|---|
| INPSOC (default) | none, or `"soc_ecp": "inpsoc"` | MACE writes the spin-orbit ECP inline, INTERNAL convention (the manual's own Eu example, p. 87). The deck's valence basis is unchanged, so a scalar run in the same basis is a clean reference. |
| STUTSC | `"soc_ecp": "stutsc"` | CRYSTAL's internal Stuttgart spin-orbit library, one keyword per atom (manual p. 84). Only for the elements where it was measured to be the same potential as INPSOC: Rb-Mo, Ru-I, Cs, Ba, Ta-Bi. Not Hf (below), not La-Lu (CRYSTAL stops). No SOSCALE. |
| Official SOC basis | `"soc_basis": "pob-tzvp-rev2-soc"` | The X_pob_TZVP_rev2_SOC sets from crystal.unito.it: STUTSC with the pob-TZVP-rev2 valence shells. For Cs, Ba and Hf-Bi these are the shells `full.basis.triplezeta` has; for Rb-I, those of `full.basis.triplezeta.rev2022`. Hf gets the official shells with MACE's INPSOC instead of STUTSC (below). Not with CHEMOD. |

**Hf: INPSOC, never STUTSC.** MACE's Hf potential (INPUT, and INPSOC for
SOC) is the published ECP60MDF (Figgen et al., J. Chem. Phys. 130 (2009)
164108). CRYSTAL's internal STUTSC for Hf gives the r² term of the local
potential the exponent of the first term, 10.2488560 where ECP60MDF has
10.7291270; measured on HPCC (HfO2), that costs +9.67e-5 Ha. So
`"soc_ecp": "stutsc"` is refused for Hf, and since the official
Hf_pob_TZVP_rev2_SOC record names STUTSC, `"soc_basis"` writes Hf with the
official valence shells on MACE's INPSOC, and says so. A deck derived from
one with the official Hf STUTSC block gets INPSOC the same way.

**Validated on HPCC (CRYSTAL/23-intel-2023a).** STUTSC and MACE's INPSOC are
the same potential for Bi (bitwise identical energies) and for Pb and Te
(2.8e-7 Ha apart). The PbTe sanity deck exactly as MACE writes it (INPSOC,
PBE0) converges to -461.17736583 Ha in 9 cycles, 1.1e-7 Ha from the
reference.

**f elements (La-Lu, actinides).** An open-shell 4f 2c-SCF did not converge
on HPCC: EuO from a cold start stopped with "FERMI ENERGY NOT IN INTERVAL",
and with SMEAR (0.005 or 0.01) its energy swung by about 8 Ha from cycle to
cycle until MAXCYCLE. A SOC deck with an f element gets a warning. A route
that may work, untested: a one-component SPIN parent run without symmetry,
then the SOC deck from it with `"soc_guess": "guesspnoso"`.

The energy a SOC deck gives is compared with a two-component run without
SOC in the same basis (manual p. 174): copy the deck and delete the `SOC`
line of its TWOCOMPON block.

**Metals and small gaps: k-mesh and SMEAR.** Measured on HPCC (FMIXING
85, TOLDEE 7): fcc Au aborted in the first SCF cycles on MACE's default
SHRINK 10 10 and on 11 11 ("POSSIBLY CONDUCTING STATE", then "ERROR **** ZERO
**** FERMI ENERGY NOT IN INTERVAL"); it converged on 12 12, and on 10 10 with
SMEAR 0.005. PbTe (7 7) and a Bi2 bilayer (0 10 / 10 10 1) converged without
SMEAR. So for a metal or a small-gap system:

- add `"smear": 0.005` (hartree) to the options file or template; MACE never
  adds SMEAR by itself;
- or use a denser mesh than the scalar run's;
- `opt2d12` warns when the parent `.out` looks metallic (POSSIBLY CONDUCTING
  STATE, or a gap below 0.1 eV) and, at a terminal, asks for a width (Enter =
  none). The `"smear"` key answers that question without a terminal, with
  the same result: a terminal answer of 0.005 and `"smear": 0.005` write the
  same deck. Without a terminal and without the key (`--yes`, a workflow)
  it writes the deck without SMEAR and lists it in the closing summary; the
  GUI shows the warning in its SOC section and offers the width there.
  `mace convert` cannot tell from a CIF and prints one warning per run.

Other keys: `"soc_fmixing": N` (default 50; 80 for a GUESSPSO restart),
`"soc_guess": "guesspso" | "guesspnoso"` (a 2c restart from the parent's
`.f9`, opt2d12 only). SPIN is accepted and has no effect in a 2c run. The
full reference is [Crystal_d12/README.md, soc_ecp.py](../Crystal_d12/README.md#soc_ecppy).

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
ntasks        = 64                  # #SBATCH --ntasks=64 (built-in: 32; properties 8)
nodes         = 1                   # #SBATCH -N 1 (above 1: see "Several nodes" below)
cpus_per_task = 1                   # #SBATCH --cpus-per-task=1
mem_per_cpu   = "3900M"             # #SBATCH --mem-per-cpu=3900M; or mem = "250G" (#SBATCH --mem), not both
walltime      = "24:00:00"          # #SBATCH -t 24:00:00 (built-in: 7-00:00:00; properties 2:00:00)
[resources.properties]
ntasks = 32                         # the same keys, for Pproperties jobs only
mem    = "120G"
[resources.FREQ]                    # the same keys, for one calculation type (see below)
ntasks   = 64
walltime = "3-00:00:00"
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
[queue]                             # the queue budget: see mace queue
reserve  = 100                      # queue slots MACE never uses (default 100, a tenth of a limit below 1000)
budget   = 940                      # the most jobs MACE keeps queued, all workflows together (default: limit - reserve)
limit    = 1040                     # your submit limit, when SLURM does not say (default: read from SLURM)
order    = "depth"                  # held jobs start: depth (default), breadth or mixed
followup_share = 0.75               # order = "mixed": the share of free slots offered to next steps first
sentinel = true                     # false: no mace-drain job restarts held work when no MACE job is queued
max_jobs = 950                      # older form: budget = max_jobs - reserve (budget replaces it)
[local_run]
max_time = "4h"                     # longest `mace run --timeout` (90s, 30m, 4h or H:MM:SS)
gateway_message = "This is a login node ({reason}): log in to a compute node and rerun mace run."
module_message  = "the {module} module cannot be loaded on {host}"
[folders]                           # what a project folder's mace-folder.toml [submit] may set
allow = ["ntasks", "walltime", "partition"]   # the default; [] lets a folder set none
partitions = ["general", "bigmem"]  # the partitions a folder may choose (default: none)
max_ntasks = 128                    # a folder's ntasks above this is ignored
max_walltime = "7-00:00:00"         # likewise its walltime
```

Each key changes only its own line(s) of the job script; with none of them
set, the scripts keep their bytes:

| Key | Environment | In the job script |
|---|---|---|
| `[slurm] qos`, `constraint` | `MACE_SLURM_QOS`, `MACE_SLURM_CONSTRAINT` | `#SBATCH --qos=`, `#SBATCH --constraint=` after `-A`/`-p` |
| `[slurm] extra` | `MACE_SLURM_EXTRA` (comma-separated) | one `#SBATCH` line each; `--ntasks`, `--mem`, `--time`, `--account`, `--partition`, `--qos`, `--constraint`, `--job-name`, `--output`, `--error`, `--array`, `--exclude`, `--chdir` and their abbreviations (`--part=`) are refused (MACE writes them; the message names the key to use) |
| `[resources] ntasks`, `nodes`, `cpus_per_task`, `mem`, `mem_per_cpu` | `MACE_SLURM_NTASKS`, `MACE_SLURM_NODES`, `MACE_SLURM_CPUS_PER_TASK`, `MACE_SLURM_MEM`, `MACE_SLURM_MEM_PER_CPU` (both job kinds and every calculation type; one memory form clears the other) | `--ntasks`, `-N`, `--cpus-per-task`, the memory line in its form; `cpus_per_task` above 1 also exports `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` |
| `[resources] walltime` | `MACE_SLURM_WALLTIME` (every job) | `#SBATCH -t`, written as set (`D-HH:MM:SS`, `D-HH` or `HH:MM:SS`) |
| `[resources.properties]`, `[resources.OPT]`, ... | (the variables above win over them) | the same lines, for properties jobs or one calculation type |
| `[mpi] launcher`, `srun_mpi`, `env` | `MACE_MPI_LAUNCHER` | the line that starts CRYSTAL; `srun` keeps `mpirun` for `mace run` |
| `[modules] purge` | `MACE_MODULE_PURGE` (0 or 1) | `module purge` becomes a comment |
| `[crystal] bin_dir`, `scf`, `properties` | `MACE_CRYSTAL_BIN_DIR` | the executables; with no CRYSTAL module of yours, the module line becomes a comment (and `EBROOTCRYSTAL` is set when `bin_dir` ends in `/bin`) |
| `[paths] scratch` | `MACE_SCRATCH_BASE` | one scratch candidate after `$SCRATCH` |
| `[local_run] max_time` | `MACE_RUN_MAX_TIME` | (no script change: the longest `mace run --timeout`) |
| `[folders] allow`, `partitions`, `max_ntasks`, `max_walltime` | | (no script change: what a project folder's `mace-folder.toml` `[submit]` table may set; see docs/gui.md, "Folder settings") |

`mace submit -i`, `--walltime` and the workflow planner write cores,
memory and walltime themselves: the `[resources]` values are their defaults
(the prompt shows the site's memory form, "Memory per CPU" or "Memory per
node (total)"), and a value you type is written as typed.

### Change the default cores per job

The built-in job size is 32 cores for CRYSTAL jobs (OPT, SP, FREQ) and 8
for properties jobs (BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL; 28 before
1.1.5, but measured BAND and DOSS runs kept only about 1 of 28 cores busy;
`[resources.properties] ntasks = 28` keeps the old size). To use, for
example, 16 cores for every job and fewer for properties jobs, set it once in
`~/.config/mace/site.toml` (`mace site --init` creates the file with every
key commented out):

```toml
[resources]
ntasks = 16            # every job: OPT, SP, FREQ, and properties jobs too

[resources.properties]
ntasks = 4             # optional: properties jobs only (BAND, DOSS, ...)
```

or from the shell, which edits the same file (and creates it when it is
absent):

```bash
mace site set resources.ntasks 16
mace site set resources.properties.ntasks 4
```

or for one shell session (or in `~/.bashrc`): `export MACE_SLURM_NTASKS=16`.
Check it with `mace site`: the `Resources:` and `Properties:` lines show
`--ntasks=16` and the file it comes from.

The value is then the default everywhere MACE offers or writes a job size:
the workflow planner's "Cores" offer, `mace submit` (also `--array`,
`--track`, `--progress` and `-i`), `mace workflow --quick-start`, the GUI's
Submit dialog, and the next steps of a workflow (also the follow-up jobs of
`mace submit --track` in a plain folder). A number you choose for one job
still wins: the answer to a planner or `-i` prompt, and
`MACE_SLURM_NTASKS` over the file. A workflow planned before you changed the
setting keeps the sizes stored in its plan. Memory (`mem_per_cpu` or `mem`)
works the same way.

**One calculation type.** A `[resources.<TYPE>]` table sets the same keys
for one calculation type: `OPT`, `SP`, `FREQ`, `BAND`, `DOSS`, `TRANSPORT`,
`"CHARGE+POTENTIAL"` (quoted, for the `+`) or `ORBITALS`. For example, frequencies on
64 cores for 3 days and a smaller band structure:

```toml
[resources.FREQ]
ntasks   = 64
mem_per_cpu = "7G"
walltime = "3-00:00:00"

[resources.BAND]
ntasks = 4
```

(`mace site set resources.FREQ.ntasks 64`, `mace site set
resources.FREQ.walltime 3-00:00:00`, ...). `mace submit` tells each deck's
type from its keywords (OPTGEOM: OPT, FREQCALC: FREQ, else SP; BAND, DOSS,
BOLTZTRA, ECH3/POT3 in a .d3), so a folder of mixed decks gets each type's
size; `mace submit -i` then asks once per type that has a table. On a pipe
(answers fed in, not typed) `mace submit -i` refuses when the decks split
into more than one group: it lists the groups in the order they are asked
and submits nothing. A script that answers each group in that order sets
`MACE_SUBMIT_PIPED_GROUPS=1`. Per key, the first of these wins:

1. what you choose for one job (a planner or `-i` answer, `--walltime`);
2. the environment (`MACE_SLURM_NTASKS`, `MACE_SLURM_WALLTIME`, ...);
3. `[resources.<TYPE>]`;
4. a project folder's `mace-folder.toml` `[submit.properties]` (properties
   types only; `ntasks` and `walltime`, as site.toml's `[folders]` allows);
5. `[resources.properties]` (properties types only);
6. the folder's plain `[submit]` (`ntasks` and `walltime`; its walltime
   covers properties jobs only when neither 4 nor 5 sets one);
7. `[resources]`;
8. the built-in values.

`mace submit DIR` and `mace submit DIR/deck` read the `mace-folder.toml`
of the decks' folder (or its nearest parent below your home), wherever you
run them from, as the completion callback that later runs in that folder
does. `MACE_FOLDER_SETTINGS` still wins.

A folder's walltime and a size-aware suggestion's are never above the
partition's MaxTime: MACE asks for the MaxTime instead (`mace site` says so).

Each table is read from your `site.toml` and the group files in the usual
order (your file wins over a group file per key). `mace site` lists the
size and walltime of every type, with the table each value comes from, once
any `[resources.<TYPE>]` table is set.

**Walltime.** `[resources] walltime = "24:00:00"` (or `D-HH:MM:SS`, `D-HH`;
`MACE_SLURM_WALLTIME` for one session) is the default walltime of every job
in place of the built-in 7 days (CRYSTAL) and 2 hours (properties); it goes
in `[resources.properties]` and `[resources.<TYPE>]` like the job size. It is
the default the planner and `mace submit -i` offer, and the `#SBATCH -t` of
every job script MACE writes; `mace submit --walltime` and a typed answer
still win. A walltime above your partition's MaxTime that you set (site.toml,
the environment, `--walltime`) is written as set (SLURM then refuses the job;
`mace site` warns about it, whatever the MaxTime), while the built-in one, a
project folder's and a size-aware suggestion's are lowered to the MaxTime.

**Size-aware suggestions.** With `[resources] auto = true` (`mace site set
resources.auto true`), MACE sizes each job from its deck instead of using
one size for every job: a band structure of a small cell asks for a few
cores and a few GB, a FREQ of a 1,800-AO cell a full node for days. It is
off by default (every job script and planner offer stays as it was), and it
can be turned on for some types only:

```toml
[resources]
auto = true            # every type ...

[resources.OPT]
auto = false           # ... but OPT (or only [resources.BAND] auto = true)
```

Per key,
the first of these wins: what you choose for one job, the environment,
the values set in `site.toml` (any table), the suggestion, the built-in
values. So `[resources] ntasks = 16` with `auto = true` keeps 16 cores and
takes the suggested memory and walltime, computed for 16 cores.

`mace site suggest DECK...` prints the suggestion and the reason, also when
`auto` is off, and the job size the deck's job script gets now:

```text
$ mace site suggest mof_opt_sp.d12
mof_opt_sp.d12  (SP)
  Suggested:  --ntasks=32 --mem-per-cpu=2G -t 0-07:00:00
  Why:        ~1,800 AOs, 24 k-points x 2 spins -> 32 cores (48 cores useful); ~2G per core; walltime 7.0 h (estimate ~33 min)
  Sized from: out: mof_opt.out
  Job script: --ntasks=32 --mem-per-cpu=5G -t 7-00:00:00  ([resources] auto off: ...)
```

(`mace site suggest --type FREQ DECK` sizes a deck as another type, `mace
site suggest --json DECK` prints the same as JSON.) The inputs are the atomic
orbitals, irreducible k-points, symmetry operators and spins, read from the
CRYSTAL output of the deck's parent run when one is in the folder (the
longest `.out` name the deck's name starts with, or the deck's own), else
worked out from the deck: the basis shell by shell (explicit basis) or per
element (an internal `BASISSET`, from MACE's bundled basis libraries), the
cell expanded by its space group and SUPERCEL, and the k-points from SHRINK.
A FREQ also counts its displacements (3 per atom of the asymmetric unit).
The rules, fitted to about 680 completed CRYSTAL23 runs and checked on a
held-out 30 %:

* work: one SCF cycle costs about `exp(-4.63) x AO^1.66 x (k-points x
  spins)^0.52` core-seconds; an SP is ~14 cycles, an OPT ~18, a FREQ ~39
  per displacement; a properties job `exp(a) x AO^b` core-seconds per type.
  Half of the held-out estimates are within a factor of 1.5 to 2 of the
  measured run time (DOSS 3);
* cores: Pcrystal spreads the integrals over every process and the k-space
  work over the k-points, so more cores help up to `max(k-points x spins,
  AO/16)` (properties: AO/100); within that, the fewest cores that finish the
  estimate in an hour (properties: 15 minutes), never more than a node has
  (`mace site` caches the node size; without it the built-in 32 / 8);
* memory: per process `428 + 151 x AO^2/symmetry/10^6 + 863 x k-points x
  AO/10^6` MB times 2.6 (FREQ counts no symmetry), at least 2 GB per core;
  properties `276 + 87 x AO^2/10^6` MB times 1.35. No held-out job peaked
  above its suggestion (the smallest margin was 1.3x);
* walltime: the estimate times the 95th percentile of measured/estimated for
  the type, times 1.5, rounded up to the hour, at least 2 hours (properties
  1 hour), at most the partition's MaxTime. It covered 90 to 100 % of the
  held-out runs per type; the misses were SCF runs that took hundreds of
  cycles (metallic slabs) and repeated properties runs that took 10 times
  their siblings. Set `walltime` (it wins over the suggestion) where your
  systems converge slowly.

Where it applies: `mace submit` (a folder gets one job size per group of
decks with the same suggestion; `-i` offers it as the default; `--array`,
`--track`), the next steps of `--track` runs in a plain folder, and the
planner, `--progress` plans and `mace workflow --quick-start`, which store
one size per step: what fits the largest of the plan's inputs. Each says the
reason as it goes. The GUI's Submit dialog shows the reason under each deck
(the same function, `GET /api/submit/suggest?path=DECK` for scripts).

**Change a setting from the shell.** `mace site set KEY VALUE` and
`mace site unset KEY` edit your own `site.toml` (never a group file: whoever
owns it edits it by hand). KEY is `table.key`, for example `resources.ntasks`,
`resources.properties.mem`, `resources.OPT.walltime`, `slurm.account` or
`mpi.launcher`; the type in `resources.<type>.<key>` may be lower case. The
value is checked as `mace submit` checks it before anything is written,
the file keeps its comments, blank lines and other values (a commented-out
example of the key, as `--init` writes them, is uncommented), and MACE
prints the value the job scripts now get, with a note when an environment
variable wins over the file. Setting `mem` removes the table's
`mem_per_cpu` and the other way round. A list takes `["a", "b"]` or, for
`slurm.extra` and `modules.python`, a comma-separated list (`mace site set
slurm.extra --exclusive`). `unset` removes the line, or comments it out when
it carries a comment of its own. Exit codes: 0 done, 1 nothing to unset, 2
an unknown key, an invalid value or a file it cannot edit safely (an inline
table or a dotted key: edit those by hand).

**Folder settings from the shell.** `mace site set --folder [DIR] KEY
VALUE` and `mace site unset --folder [DIR] KEY` set or remove one key of a
project folder's `mace-folder.toml` (DIR, default the folder you are in;
the file is created from the commented template when the folder has none).
They are the GUI's Settings > Save to this folder from the shell, with the
same writer and the same checks (`mace/utils/folder_settings.py`):

```
mace site set --folder plots.energy_window -3,5      # here
mace site set --folder ~/proj units.energy meV
mace site set --folder ~/proj submit.walltime 1-00
mace site unset --folder ~/proj plots.energy_window
```

KEY is `table.key` from the file's keys (docs/gui.md, "Folder settings"),
including the ones the GUI does not save: `setup.opt2d12_config`,
`setup.opt2d3_config`, `setup.cif2d12_options` and the `[submit]` keys. A
value is checked before anything is written: `true`/`false`, a number, a
list as `-3,5` or `[-3, 5]` (`database.columns material_id,formula`), a name
or a path inside the folder. The file keeps its comments and other values.
Refused, with nothing written: a file that is a symbolic link or that
everyone can change, a folder everyone can write to (a scratch root: MACE
would ignore the file there), your home folder, a file MACE already ignores
(fix it by hand), and a file above DIR (it applies to DIR, but is edited
from its own folder). A `[submit]` value your site's `[folders]` rules
would ignore (a key the site does not let a folder set, a partition not in
its list, ntasks or a walltime above its maximum) is refused with the
site's reason. Exit codes: 0 done, 1 nothing to unset, 2 refused.

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
  it (`sacctmgr show assoc user=$USER`). Where that cannot be checked (no
  `sacctmgr`, or one that lists nothing for you) it is still written on MSU
  HPCC and on a host without SLURM (a workstation). Otherwise the `#SBATCH
  -A` line is left out, so SLURM charges your default account, and MACE says
  so once: "no account line (SLURM's default account); set yours with
  MACE_SLURM_ACCOUNT or site.toml".
- **Python modules.** The built-in Python modules are loaded only where they
  load (a module probe), or where there is no module system. Otherwise the
  job's completion step runs the Python that runs MACE (its full path, as
  `python() { "/path/to/python" "$@"; }` in the job script), and MACE says so
  once. Install MACE's requirements into that Python, or set `[modules]
  python` to your cluster's Python module.
- **Queue limits.** `--max-jobs 950 --reserve 50` (the completion step keeps
  at most 900 jobs queued) becomes your MaxSubmitJobs and the reserve
  (`--max-jobs 1040 --reserve 100` for 1040) when `scontrol show assoc_mgr`
  reports one; `[queue]` sets them (see [mace queue](#mace-queue)).
- **Walltime.** A template's default walltime (7 days for decks, 2 hours for
  properties) is lowered to the MaxTime of your partition (`[slurm]
  partition`, else the cluster's default partition) when that is shorter.
  A walltime you ask for (`--walltime`, `-i`, `[resources] walltime`) is
  written as given.
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
  the setup out. A CPHF/CPKS stage is estimated from its own directions
  (cycles per direction left from the directions already done, [min - 1,
  max + 2], or the corpus range 3-31 before the first finishes; time from the
  last cycles x 0.75-1.5), a pending second-order stage from the first-order
  one, the tensors are bounded by the CPHF stages' time (they print three
  lines and then nothing for many minutes, so a running one also says how
  long it has run), and the central point that follows takes 0.75-1.75 x
  the reference SCF. A FREQINFO.DAT restart
  shows the central point as restored and takes the time per displacement
  only from the rows computed in the restarted run. No time is extrapolated
  from the central point: displaced cells have lower symmetry and cost from a
  fraction to many times as much (6x on one corpus run), so before the first
  displacement of the run finishes the total is "unknown".
- **Properties runs** (1.1.6): one level per block of the `.d3` (NEWK, BAND,
  DOSS, ECH3, POT3, BOLTZTRA, ...), done when CRYSTAL prints its timing line;
  DOSS, POT3 and BOLTZTRA are scaled from the block before them (NEWK, ECH3,
  NEWK) with the ratio's range over half the corpus; a run still in its first
  block has nothing to scale from and says so.
- **EOS, ELASTCON, ANHARM** (1.1.6): the volume, strained configuration or
  X-H distance that runs, of how many (from RANGE, DEFORM x NUMDERIV, POINTS26
  or CRYSTAL23's defaults; ELASTCON's strain count is CRYSTAL's symmetry
  analysis, unknown without DEFORM), and the time left from the units done.
  No run of these kinds was available to test against, so they are "rough".
- **Metals and stalls** (1.1.6): an SCF that printed "POSSIBLY CONDUCTING
  STATE" gets a 1.5x higher upper end; a running job silent for 10x its
  stage's time per step (at least 30 min) reads "possibly stalled".
- **The total** adds the time left of the stages still to run. When a stage
  has no upper end (for example a FREQ before its first displacement), it reads "at
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
| CPHF/CPKS cycles left (1.1.6, leave-one-family-out) | 9 (9) | 79 % (73-84) of 187 cuts; time left 84 % (78-89) |
| IR/Raman tensors bound / central point after CPHF (1.1.6) | 9 (9) | 90 % (85-93) / 86 % (80-90) |
| Properties block from the one before (1.1.6) | DOSS 36, POT3 28 | DOSS 100 % (90-100), POT3 93 % (77-98); BOLTZTRA 4 runs, not gated |
| SCF cycles left, metals (1.1.6) | 22 (9) | 89 % (85-100) with the 1.5x high end, 74 % before; insulators 95 % |

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

Beyond the job (1.1.6), `mace cluster job` and the Job document add:

- **Queue wait** for a pending job: SLURM's own start estimate (what `squeue
  --start` shows), labeled "queue wait estimate from SLURM"; when SLURM has
  none yet, it says so. MACE does not model the queue itself.
- **Array**: when the array's last task should end: time per task from the
  array's finished tasks (sacct), before any finished from this task's own
  estimate; a pending task starts at its SLURM estimate.
- **Workflow**: the steps of the workflow plan after this job, each from the
  history model (the nearest finished runs of that calc type, scaled to this
  system), plus this job's own time left; the queue wait of steps not
  submitted yet is not estimated.

These totals are "rough": no array or workflow history was available to test
them against. The inspector's run card in the GUI shows the same levels for
an output still being written.

GUESSP: the corpus's two GUESSP SCFs are 1000-cycle slab runs, so no
narrower GUESSP range was fit; the flag stays.

Still planned: levels inside a FREQ preoptimization; a progress column in the
Cluster view's Overview; scaling the typical-run priors by the thresholds once
there are tight-threshold runs to fit; reading only the new part of a very
large output; and `mace cluster progress FILE.out` for runs started with `mace
run`.

CPU and memory efficiency are shown in plain numbers, for example "CPU
efficiency 62% (used about 20 of 32 cores on average)" and "Memory peak 9.4 GiB
of 64 GiB requested (15%)", with a short hint when a job clearly asked for too
much. One note follows the CPU line: MPI libraries such as Intel MPI keep
waiting ranks busy, so CPU efficiency near 100% does not show that every core
did useful work; compare run times at different core counts to find the useful
size. CPU efficiency is the CPU time of all the job's steps over elapsed time x
cores; memory is the highest MaxRSS of its steps (x tasks for a multi-task step)
over the requested memory (AllocTRES `mem=`, else ReqMem per CPU, per node or in
total). A running job has no CPU time or peak memory in `sacct` until its steps
end, so `--live` asks `sstat` once for its running steps and uses those numbers.
A figure SLURM did not record says "Not measured by SLURM for this job" and why
(for example processes started outside SLURM, such as by mpirun's ssh launcher on
other nodes, are not counted). `--follow` prints again every 60 s from the
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
