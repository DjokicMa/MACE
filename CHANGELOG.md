# Changelog

All notable changes to MACE (Mendoza Automated CRYSTAL Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.5] - 2026-10-15

A preview of this release was tagged `v1.1.5-preview` on 2026-10-08. The
final 1.1.5 is that preview plus the fixes listed first, under "Since the
preview".

### Since the preview

#### Upgrading from the preview

- **Upgrade in place while jobs are running.** The queue lock is now
  `.queue_locks/queue_state.lock`, one lock for every callback mode. Jobs
  still pinned to an older separate install take the old per-mode locks,
  which do not exclude the new one, until they finish; upgrading the
  install they use (in place) avoids that. Old `queue_manager_<mode>.lock`
  files are ignored and safe to delete.
- **Completion callbacks wait for `mace manager` sweeps.** A job's
  completion callback now takes the same queue lock as `mace manager`
  (every mode, interactive sweeps included) and gives up after 60 s: it
  then records only its own job's result and leaves the follow-up steps to
  the next sweep or callback (it says so). A long interactive `mace
  manager` session can therefore delay follow-ups until the next sweep;
  job array tasks keep retrying for up to 15 minutes.
- **`mace completion --remove-zombie-jobs` is stricter.** Only a
  calculation's own SLURM logs (`<name>-<id>.o`, `<name>_<id>.o`,
  `<name>.o<id>`, `<name>-<A>_<T>.o`) are read, and a job is canceled only
  when SLURM confirms its name and folder; anything else is listed as not
  canceled. Cancel those by hand if they really are stuck.
- **The max-jobs limit counts all your queued jobs.** `--max-jobs` and
  `--reserve` now apply to every running and pending job you have in the
  queue (each array task counted), including jobs MACE did not submit; an
  old `crystal_job_status.json` in your home or temp folder no longer
  matters.
- **Calculations stuck "submitted" by long-ended jobs.** The next sweep
  records them failed with the job's real end state. Those whose job ended
  more than 3 days ago (`MACE_RECOVER_MAX_AGE_DAYS`) are not resubmitted
  automatically; MACE prints one line for each, and `mace recover` prepares
  their retry (their error is classified as for any failed job; a job
  SLURM ended OUT_OF_MEMORY or TIMEOUT before writing its `.out` counts as
  a memory or walltime error).
- **Job scripts an older MACE wrote are written again.** A `<name>.sh`
  that MACE 1.0.0 to the 1.1.5 preview wrote with its default resources
  and nobody changed is now recognized: `mace submit` (with or without
  `--track`, `--walltime`, `-i`, `--progress`, `--array`) says "<name>.sh
  was written by an older MACE; regenerating it with 1.1.5's fixes" and
  writes it again, instead of reusing it without the 1.1.5 job-script
  fixes (or, with those options, stopping as if it had been edited). For a
  properties script that means 8 cores, not 28 (see below). A script
  edited by hand, or written with other resources (`-i`, a site setting),
  is still yours: it is reused as written, or stops a submission with
  resource options until you add `--overwrite-sh`.
- **A workflow step's job script keeps the plan's cores and memory.**
  `mace submit --walltime` (or node exclusion, or `--track` without
  options) on a step folder used to write its `<name>.sh` again from the
  plain template, dropping the cores and memory the plan had set. Now
  only the walltime (or exclusion) is set in that script; use `-i` or
  `--overwrite-sh` to size it again.
- **Queue lock in a folder shared by several users.** The lock file
  `.queue_locks/queue_state.lock` was created readable only (0644) for
  everyone but its creator, whose sweeps then failed to take it. It is now
  created 0666 less your umask (group-writable under the usual shared-folder
  umask 002). A sweep that cannot write an existing one prints one line and
  uses the owner-token lock instead; since that does not exclude a sweep
  holding the file, have its owner run `chmod g+w` on it.
- **Properties jobs ask for 8 cores, not 28.** BAND, DOSS, TRANSPORT and
  CHARGE+POTENTIAL scripts now carry `--ntasks=8` (memory unchanged:
  `--mem` is a total). To keep 28, set `ntasks = 28` under
  `[resources.properties]` in `~/.config/mace/site.toml`; a `[resources]`
  ntasks, `MACE_SLURM_NTASKS`, a typed answer and the sizes stored in an
  existing workflow plan still win. New planner and quick-start BAND/DOSS
  steps ask for `--mem=80G` (was 48G), the same as `mace submit`.
- **Edited or older job scripts.** With `-i`, `--walltime` or an
  `--exclude-*` option, `mace submit` rewrites a script MACE wrote; one
  edited by hand or written before 1.1.5 now stops the submission. Add
  `--overwrite-sh` to replace it. `--track` keeps an edited `.sh` and
  honors `-i`.
- **`mace submit` exits 1 when any job was not submitted** (also with
  `--nosubmit` when a script cannot be written). Check wrappers that test
  its exit status.
- **Some decks are refused.** Deck names with spaces or shell characters
  are refused, and so is a `.d3` whose `.f9` is missing or empty.
- **Generated scripts quote their paths,** so they differ byte for byte from
  older ones; jobs already queued keep their old scripts.
- **Copy-back is narrower.** fort.9 and properties outputs are copied back
  only when this run wrote them.
- **`-i` walltimes are capped** at the partition's MaxTime.
- **Lattice vectors of 40 Å or more get k = 1.** Regenerate decks for
  molecules or clusters in a box.
- **opt2d3 changes.** It refuses BAND, DOSS and TRANSPORT for a molecule
  (workflow plans with such a step stop there), picks collision-safe
  names, uses the output's own `.f9` first, and the static oC k-path
  changed.
- **Re-extract your database.** Several stored values changed (see Fixed:
  slab areas, TRANSPORT ZT and κ, final-geometry keys, summary counts, and
  others). Run `mace analyze --extract-properties <folder> --db-path <db>`
  without `--skip-unchanged` (or untick "Only new and changed files" in the
  GUI). A full run replaces stale rows of tracked calculations; rows filed
  under old untracked ids and rows of failed runs stay.
- **Workflows.** Workflow ids gain a 4-hex suffix
  (`workflow_20250618_160317_3fa9`; check scripts that parse ids),
  `auto_progression: false` is obeyed, d12 steps use the plan's
  `expert_config_file` when a plan carries one (the planner writes it only
  for D3 steps), isolated workflows copy their templates into
  `.mace_context_<id>/workflow_scripts/`, and a plan's archive or
  export-and-delete action runs when its last step finishes.
- **Error type names changed:** `convergence_error`, `timeout_error`,
  `disk_space_error`, `geometry_error` (old names are mapped).
  Out-of-memory and Fermi-level errors are now recoverable.
- **MSU-only flags are hidden off MSU HPCC.**
- **GUI Setup and Batch Setup stop on an unrecognized CLI question**
  instead of sending Enter (a chosen VBH functional is now really
  applied). Batch Setup prefixes same-named parents (`A_`, `B_`).
- **`mace/workflow/submitcrystal23.sh` and `mace/workflow/submit_prop.sh`
  are gone.** They were symbolic links to the generators in
  `mace/submission/`, which every MACE reader now uses directly. A script
  or alias of your own that names the `mace/workflow/` path must name
  `mace/submission/submitcrystal23.sh` (or `submit_prop.sh`) instead; the
  generators on your PATH are unchanged, and edits you made through the
  links were made to the `mace/submission/` files already.
- **No `-A mendoza_q` where SLURM accounting cannot be read, off MSU
  HPCC.** On a cluster other than MSU HPCC whose SLURM accounting MACE
  cannot read (no `sacctmgr`, or one that lists nothing for you), new job
  scripts now carry no `#SBATCH -A` line, so SLURM charges your default
  account. If such a cluster accepted the built-in `mendoza_q` and you want
  to keep writing it, set `MACE_SLURM_ACCOUNT=mendoza_q` or `[slurm]
  account` in site.toml. Job scripts already written (`<job>.sh` beside a
  deck, reused unless you regenerate them) and workflows planned before the
  upgrade (the account stored in `workflow_configs/*.json`, the copies in
  `workflow_scripts/`) keep the line they have; regenerate the scripts or
  plan the workflow again to drop it. MSU HPCC and hosts without SLURM are
  unchanged.
- **The scripts' dry runs print comment lines.** `mace-gui-connect.sh
  --dry-run` and `mace-gui-connect.ps1 -DryRun` now start with a `# hop ...`
  line, and the `.ps1` adds `ssh --% ...` (the same command, to paste into
  PowerShell) after a `#` line. A script that reads their output one
  argument per line should skip lines starting with `# `.

#### Added

- **Docs: how to set the default cores per job.** `docs/cli.md` ("Change the
  default cores per job") shows that `ntasks` under `[resources]` in the site
  config (or `MACE_SLURM_NTASKS`) sets the job size everywhere MACE offers or
  writes one: the planner, `mace submit`, job arrays, `--progress` plans and
  the GUI.
- **Docs: SSH key setup for `mace gui connect`.** `docs/ssh-setup.md` walks
  through a key with a passphrase, loading it into an agent (Linux, the macOS
  Keychain, the Windows OpenSSH Authentication Agent service, MobaXterm's
  MobAgent, VS Code), an `~/.ssh/config` for the gateway and the dev nodes
  (with connection sharing for 2FA on Linux and macOS), how each maps to
  `--gateway`, `--hop` and `--ssh-option`, and a troubleshooting table.
  Linked from docs/gui.md and the README.

#### Changed

- **Built-in cores for properties jobs: 8 (was 28).** On MSU HPCC, real
  BAND and DOSS jobs on 28 cores kept 0.4-0.9 cores busy (1-3%), and a
  work-based sizing of 160+ measured properties runs (BAND, DOSS,
  CHARGE+POTENTIAL) asked for 8 or fewer for most of them (12-19 only for
  the largest slabs). Pproperties does spread k-points over its ranks, so
  the default is not 1. The number now lives in one place
  (`PROPERTIES_NTASKS` in `mace/workflow/common/constants.py`): the
  `submit_prop.sh` template, `mace site` and `mace site --init`, the
  planner's offer, `mace submit`, quick start and workflow plans all take
  it from there. The total memory stays `--mem=80G` (the planner and quick
  start now use it too; see Fixed); measured properties peaks stay under
  0.6 GB per rank. CRYSTAL jobs (OPT, SP, FREQ) keep 32 cores, byte for
  byte.

#### Fixed

- **`mace submit -i`, `--walltime` and `--exclude` work from a read-only
  install.** Their temporary job-script generator
  (`submitcrystal23_custom_<pid>.sh`, `submit_prop_custom_<pid>.sh`) was
  written into the install's `mace/submission/` folder, so on a group or
  site-wide install the submit stopped with "Permission denied", and a
  checkout collected the files of runs running side by side. It now goes to
  a private folder in your temporary folder (`$TMPDIR`, else `/tmp`), one
  per generator, removed when the run ends. The job scripts it writes are
  unchanged, still pinned to the install.
- **The VS Code Remote-SSH notes cover the integrated browser and
  stopping.** Ctrl+click on the sign-in link can open VS Code's own browser:
  the docs name the setting that sends `localhost` links to your default
  browser (`workbench.browser.openLocalhostLinks`) and how to get a fresh
  link, and say that closing the tab leaves the server running until Sign
  Out and Stop Server or Ctrl+C.
- **A SLAB, POLYMER or MOLECULE deck opens its run's output.** The
  viewer draws 3D decks only, so clicking such a `.d12` (the example
  folder's `graphene_slab_sp.d12` on first click) gave an error. When the
  `.out` of the same name sits next to it, the GUI now opens that output
  and says why; without one the message is as before.
- **The built-in account stays at MSU HPCC.** Where SLURM accounting
  could not be read (no `sacctmgr`, or one that lists nothing for you),
  MACE kept MSU HPCC's `mendoza_q`, so on any other cluster with SLURM
  every job script (`mace submit`, workflow steps and their
  `workflow_scripts/` copies, array tasks, recovery resubmissions, the
  GUI's Submit, `mace run`) got `#SBATCH -A mendoza_q # or general`, which
  that cluster may reject. The built-in account (and the planner's MSU
  fallback `general`) is now kept there only on MSU HPCC and on a host
  without SLURM (a workstation); elsewhere the scripts get no account line
  and SLURM charges your default account. `mace site`, `mace site check`,
  the warning MACE prints once and the GUI's Submit preview say "no account
  line (SLURM's default account); set yours with MACE_SLURM_ACCOUNT or
  site.toml". An account you set yourself is always written, members of
  `mendoza_q` still get it, and MSU HPCC's job scripts are unchanged byte
  for byte.
- **A recovered workflow step is resubmitted from its own folder.** The
  queue manager looked for a workflow step's folder by the material's name
  (`step_*/A`), but a later step's folder is named after its calculation
  (`step_002_SP/A_sp`). A recovered step after the first was copied into
  and run from the OPT step's folder (`step_001_OPT/A`), or, from a
  completion callback, from the folder of the job whose callback ran, or
  from the current folder with "Could not find workflow directory for A,
  using current directory". Recovery now uses the folder recorded for the
  failed calculation, and the warning, where no folder is found, names the
  folder it submits from.
- **`mace-gui-connect.ps1` runs on Windows PowerShell 5.1.** It read
  `ssh -V` with `2>&1` while `$ErrorActionPreference` was `Stop`, which
  PowerShell 5.1 turns into a terminating error (ssh prints its version on
  stderr), so every real run stopped at once. ssh now runs as a process
  whose output the script reads itself, nothing in the script redirects a
  native command's stderr, and an error message or a failed browser start
  no longer ends the script early.
- **Profile paths with backslashes read correctly in the connect
  scripts.** Both scripts decoded TOML escapes one after the other, so a
  saved `C:\\Users\\me\\tmp` came back as `C:\Users\me`, a tab, and
  `mp`. They now decode every escape in one pass, as Python does, and the
  PowerShell script reads `connect.toml` as UTF-8.
- **A `--dir` that does not exist on the node is named.** `mace gui
  connect` (and both scripts) said the node "did not find the mace
  command" when the remote `cd` failed; they now say "The folder <dir> does
  not exist on <node>". The stderr reader no longer writes to the console
  after the session ends, which could abort Python at exit ("could not
  acquire lock for <stderr> at interpreter shutdown").
- **First use of a node through the gateway explains the host-key
  question.** With `--hop jump`, ssh asks once whether to trust the node's
  key; the waiting note now mentions that question, and when ssh reports
  "Host key verification failed" MACE prints the line to run once (`ssh -J
  <gateway> <node> true`) or suggests `--hop nested`. Keys are never
  trusted automatically, and `--hop auto` does not switch hops on a
  host-key failure. With `-J` the gateway's key is checked on your computer
  too: the message names the gateway (`ssh <gateway> true`) or the node
  when ssh's own message names it, and otherwise gives both lines, the
  gateway's first. ssh prompts written to stderr now show before you
  answer them.
- **`mace gui connect --dry-run` prints a PowerShell line on Windows.**
  Its POSIX-quoted line cannot be pasted into PowerShell; on Windows each
  command now also comes as `ssh --% ...`, and each form says which shell
  it is for.
- **Killing `mace gui connect` on Windows ends its ssh.** A Python killed
  from Task Manager left `ssh.exe` holding the tunnel; ssh now runs in a
  Windows job object that closes with Python.
- **Stopping MACE GUI no longer fails on a cut-off reply.** A stop shuts
  the server down while it answers, and on a busy machine the reply could
  end early. `mace gui --slurm` then exited with code 1 (an IncompleteRead
  traceback) instead of reporting that MACE GUI stopped. The control calls
  of `mace gui --managed`/`--slurm`, `mace gui connect` and `mace gui
  login-url` now treat a cut-off reply as no answer.
- **`mace-gui-connect.sh` shows the link at once in MobaXterm.** Under
  MobaXterm the link appeared only after Enter was pressed. The script now
  takes ssh's output from a file it follows instead of a FIFO, and its
  check that the tunnel answers gives up on a silent connection after 3 s
  (20 s in all, as `mace gui connect`) instead of waiting for good.
- **No "stty: standard input: Inappropriate ioctl for device" before the
  link.** A node's login profile prints it when `bash -l` runs without a
  terminal; `mace gui connect` and `mace-gui-connect.sh` drop exactly that
  line until the server is ready (it stays in their failure diagnosis).
- **No "channel N: open failed: connect failed: Connection refused"
  before the link.** With the jump hop the tunnel's local port listens as
  soon as ssh connects, so a connection that arrives before the server
  listens (a browser tab left from an earlier session) made ssh print it.
  `mace gui connect` and `mace-gui-connect.sh` hide exactly that line until
  the tunnel answers, and show it when the tunnel never does.
- **The scripts' `--dry-run` names the hop.** `mace-gui-connect.sh
  --dry-run` and `mace-gui-connect.ps1 -DryRun` start with a `# hop jump`,
  `# hop nested` or `# no hop` line, as `mace gui connect --dry-run` does;
  the arguments follow one per line.
- **The MobaXterm notes cover its own keys.** MobaXterm asks the host-key
  questions again and, with the jump hop, the node's password, until a key
  is set up in its own `~/.ssh`; docs/gui.md shows how to use the Windows
  key there.
- **The MobaSSHTunnel recipe works.** docs/gui.md told MobaXterm users to
  make one tunnel from the gateway to the node's port, but `mace gui`
  listens only on the node's `127.0.0.1`, so the browser kept loading. The
  section now uses the `ssh -N -J ...` line `mace gui` prints (in
  MobaXterm's local terminal) or two chained MobaSSHTunnel tunnels ending
  at `127.0.0.1`, and explains a browser that keeps loading. The printed
  hint says the tunnel must end at the node's `127.0.0.1`.
- **The tunnel hint can be pasted.** `mace gui` printed `you@<gateway>`,
  which bash reads as a redirect ("bash: gateway: No such file or
  directory"). It now prints `LOGIN_HOST` and says to replace it, or the
  real login host from `--gateway`, `MACE_GUI_GATEWAY` or the new `[gui]`
  `gateway` key of `site.toml` (which a cluster's `/etc/mace/site.toml`
  can set for everyone). It also gives the line that exports the
  standalone script for a computer without MACE (MobaXterm, Git Bash,
  PowerShell), and the GUI's Connecting dialog shows the same.
- **`mace gui` never draws through X11 forwarding.** MobaXterm sets
  `DISPLAY` on the node. The server and its workers now always use
  matplotlib's Agg backend (jobs submitted from the GUI keep your login
  environment), and `--open` starts no browser on the node through a
  forwarded display; it says to use the tunnel.
- **A profile saved on Windows reads in MobaXterm.** Python on Windows
  wrote `connect.toml` with CRLF line ends, and `mace-gui-connect.sh` then
  never found its `[defaults]` profile ("pass --gateway ... or save a
  profile"). The script now ignores a line's trailing CR everywhere, and
  `mace gui connect --save` writes LF line ends on every system (Python and
  the PowerShell script read both).
- **MobaXterm, Cygwin and Git Bash find a profile saved on Windows.**
  `mace-gui-connect.sh` reads `%APPDATA%\mace\connect.toml` when its own
  `~/.config/mace/connect.toml` is missing, and says so; under Git Bash it
  turns off MSYS path conversion, so `//` and `/c/` paths reach ssh as
  written.
- **Windows checkouts and zip files no longer break workflow plans.** The
  workflow planner and executor read their job-script templates through
  two symbolic links, which Windows extraction (or git without symlink
  support) turns into one-line text files that were copied as the
  template. They now read `mace/submission/` directly.
- **The connect documentation covers Windows.** `mace gui connect` runs
  natively on Windows (`python <MACE>\mace_cli gui connect`, since
  Anaconda has no `py` launcher); the profile's permissions on Windows, the
  dry-run quoting per shell, first-use host keys, MobaXterm's own profile
  and `~/.ssh`, picking a personal port on a shared dev node, and the
  PowerShell script's differences (no `--slurm`, `-Hop auto` works as
  `jump`) are described.

- **Submit no longer reports a false preview mismatch.** It could say the
  run's job scripts differ from the preview when the scripts' trailing
  checksum happened to look like a Unix time, because only one side was
  normalized. The comparison now ignores that checksum.

- **FREQ "Custom settings" writes the IR/Raman spectrum it asks for.** A
  Yes to "Generate IR/Raman spectrum plot?" wrote no IRSPEC or RAMSPEC
  block, and the spectrum questions were never asked. They now are, and
  each answer becomes its CRYSTAL23 keyword: the spectral range (RANGE),
  the step size (LENSTEP, replacing a "resolution" nothing read), a
  Gaussian line shape (GAUSS for IRSPEC, VOIGT 0 for RAMSPEC), the FWHM
  (DAMPFAC), and the Raman temperature and laser (RAMANEXP T, nm). An
  answer at CRYSTAL's default writes nothing. With IR and Raman both
  chosen, both plots are offered. The questions no CRYSTAL input or MACE
  tool reads (a Raman laser and temperature outside RAMANEXP, absolute
  Raman units, CSV/JCAMP-DX output) are gone, and the RAMANEXP temperature
  is kept apart from THERMO's TEMP. The predefined FREQ templates write the
  same decks as before.
- **Setup and Batch Setup accept a CIF spglib cannot reduce.** A P1 CIF, or
  a slab or molecule in a box, makes `mace cif2d12` ask "Use the 'reduced'
  structure anyway?", a question Setup did not know, so it stopped and
  wrote no deck. Setup now keeps the CLI's default (Yes; every atom is kept
  either way).
- **TRANSPORT Setup works on an output with a Fermi energy.** `mace opt2d3`
  then asks for the chemical potential range relative to it ("Minimum μ
  (eV below Fermi)"), which Setup did not recognize, so Setup and Batch
  Setup stopped without a deck. The CLI's defaults (-2 to 2 eV) are kept.
- **Setup knows the rest of the CLIs' questions on its routes.** A check of
  every question in the CLIs' source found more that Setup could reach but
  did not recognize, each stopping the run: a FREQ template's "Other
  questions" (spectral and temperature ranges, supercell, phonon band path
  and k-point density, INS) and Custom settings (supercell expansion,
  INTERPHESS, phonon bands and DOS, CPHF, restart checkpoint); cif2d12's
  tolerance search after "Use original CIF space group"; a basis set
  missing an element (Setup now shows the CLI's refusal); opt2d12 on a
  parent whose functional MACE cannot identify (Enter keeps the CLI's
  HSE06); and opt2d3's POTC point count. The FREQ template answer no longer
  also answers the phonon band path's "Select template" menu.
- **An EXTERNAL deck redraws when its geometry file changes.** The GUI's
  cache was keyed on the deck file alone, so after its `.ext` (or the
  run's `.out`) was added or changed the viewer kept the old drawing or the
  old error until the deck itself changed.
- **A deck with a non-standard origin (IFSO > 1) is drawn as CRYSTAL builds
  it.** The GUI (and `mace preopt`) ignored the IX IY IZ origin-shift
  record, so the space group's operations were applied about the wrong
  point and a different structure was drawn. They now act about the shifted
  origin (IX IY IZ / 24 of the cell vectors), matching CRYSTAL23's TESTGEOM
  output; the deck's own frame is kept.
- **An EXTERNAL deck opens in `mace gui`, and `mace submit` / `mace
  preflight` say when its fort.34 is missing.** A deck whose geometry is
  `EXTERNAL` (CRYSTAL reads fort.34 at run time) showed "not found:
  'cell_parameters'". The GUI now draws it from the file its job would stage
  as fort.34 (`<deck>.ext`, or what the deck's own `.sh` copies there), else
  from the run's `.out`, labeled "geometry from <file>"; with neither it says
  what to put where. `mace submit` (and the GUI's submit preview) warns
  before such a deck goes to the queue without that file, and `mace
  preflight` names it and stages what a hand-written job script copies.
  A deck the reader cannot follow now gets a plain reason (a 400, not "not
  found"); decks with the space group as a symbol (IFLAG 1) or in
  rhombohedral axes (IFHR 1) draw correctly, and a geometry edited after the
  atoms (SUPERCEL, ATOMSUBS, ...) is labeled as drawn before the edits.
- **Properties jobs get one memory default on every path.** Workflow quick
  start stored the properties memory as `memory_per_cpu: "80G"`, which the
  executor ignores for `submit_prop.sh`, so those jobs got `--mem=48G`; the
  planner offered 48G for BAND and DOSS (80G x 0.6, also applied to a
  site's own `[resources.properties] mem`), while `mace submit` and the
  template use 80G. Quick start, the planner, workflow steps and `mace
  submit` now all write the same total, `--mem=80G` (`PROPERTIES_MEMORY` in
  `mace/workflow/common/constants.py`), and a site's `[resources.properties]`
  `mem` or `mem_per_cpu` is written as set.
- **The job-efficiency report says what CPU efficiency cannot show.** MPI
  libraries such as Intel MPI keep waiting ranks busy, so CPU efficiency near
  100% does not show that every core did useful work. `mace cluster job` and
  the GUI's Job document now say so once, right after the CPU line (once per
  `--follow`, not every refresh), and suggest comparing run times at different
  core counts; the numbers and the "You could request fewer cores" hint are
  unchanged.
- **The run view's ENERGY ALONG THE RUN is one continuous curve across
  FINALRUN restarts.** An optimization CRYSTAL restarted with FINALRUN was
  drawn as separate pieces, each relative to its own segment's point, with
  the line broken at every restart. By default the card now connects every
  optimization point and takes ΔE relative to the run's first or final
  point; the restarts stay marked by dotted lines. **Segments: Separate**
  (the card, Run Analysis › Plot Options, and Settings › Plots) brings back
  the per-segment view, remembered per document. Copy TSV gains a segment
  column and, like Open as Plot, follows the choice. The same data in the
  terminal: `mace analyze --convergence FILE.out --energy [--segments
  separate] [--relative-to first] [--unit eV]`.
- **The OPT compare view's Displacement map and arrows show something when
  only the cell relaxes.** Displacements were measured relative to the cell
  (the change of fractional coordinates), so an optimization whose atoms sit
  on special positions (AB graphite, rock salt) while the cell shrinks drew
  no arrows and a 0.0000 Å to 0.0000 Å scale. A new **Δr** choice offers
  **With cell change** (r_final − r_initial in Å, the cell change included,
  the mean shift removed, the nearest periodic image), the default when the
  cell changed, and **Relative to the cell** (the old measure), remembered
  in this browser. When the atoms did not move the legend says so and gives
  the cell's change (a, b, c, V in Å and %); the scale reads in Å or pm
  with three significant figures (one value when every atom moved the
  same), the arrow note says how far 1 Å drawn moved, the Arrow × slider
  reaches past ×200 for small relaxations, and arrows are skipped only when
  their drawn length is under 0.01 Å.
- **A click on an IR or Raman peak opens the mode under the pointer.** The
  plot took the mode from plotly's last hover, which plotly computes at most
  every 50 ms, so a quick click right after moving could open a neighbor the
  pointer passed on the way (on a LiFSI-DEC spectrum, mode 55 instead of the
  strong mode 54 1.5 cm⁻¹ away). The mode now comes from the marker nearest
  the click itself (within 10 px).
- **`mace gui connect` no longer asks for the node's password when the
  gateway's key is set in `~/.ssh/config`.** With the jump hop (`ssh -J`),
  the node hop logs in from your computer with the settings ssh has for the
  node's name, so an `IdentityFile` in the gateway's `Host` block (e.g.
  `Host hpcc` with `~/.ssh/id_ed25519_hpcc`) did not reach it: without an
  agent holding that key, the node got only the default key files, and ssh
  asked for the node's password. The gateway's `IdentityFile`,
  `IdentityAgent` and `IdentitiesOnly` are now passed for the node hop too
  (`--dry-run` shows them), and a refused node points to the new
  troubleshooting entry (docs/gui.md, "the node asks for a password"). The
  standalone scripts do the same.
- **`mace gui connect --mace ~/...` finds MACE on the node.** An unquoted
  `~` is expanded by your own shell, so `--mace ~/mace/mace_cli` reached the
  node as `/home/you/mace/mace_cli`, which is not there when the node's home
  is elsewhere ("did not find the mace command"). A `--mace`, `--dir`,
  `--root` or `--conda` value (or a saved profile value) under this
  computer's home folder is now sent as `~/...` and expanded on the node,
  with a note; quote the `~` to pass a path as is. When the mace command is
  still not found, the message names the rewritten path. `--conda` takes a
  `~/...` path too. The standalone scripts do the same.
- **Stop Server in the GUI tells every open tab that its running tasks were
  canceled.** The server now waits (5 s at most) for each canceled task to end
  and for its event to reach the tabs before it exits; before, the tabs could
  keep showing the task as running.
- **`mace gui --slurm` no longer leaves its ssh tunnel running.** The tunnel
  (and anything it started) now ends with the session however it ends: Stop,
  Ctrl+C, SIGTERM, a hang-up, an error, or, on Linux, the session being
  killed. Once the GUI is ready, a hang-up or SIGTERM leaves the job
  running, to reattach to; before that, the session cancels the job it
  submitted.
- **`mace gui connect` explains a key that ssh could not use.** When ssh
  ends with "Permission denied (publickey)" or cannot unlock a key with a
  passphrase, it now says to load the key with `ssh-agent` and `ssh-add` (on
  Windows the OpenSSH Authentication Agent service), or to use an
  `~/.ssh/config` Host alias as the gateway. A gateway that looks like an
  email address (`netid@university.edu`) gets a warning naming the profile
  file and the `--gateway ... --save` fix; it still connects. The standalone
  scripts do the same.
- **GUI: a middle click in a 3D viewer no longer pastes on Linux.** The
  primary-selection paste it also sends opened Import from Clipboard with
  whatever text was last selected (or pasted copied atoms into the open
  structure); the middle button now only pans or zooms there. Ctrl+V and a
  middle click on a text field paste as before.
- **GUI: no drop-down, menu or button cut off or hidden at everyday zoom
  levels.** Drop-downs with a fixed height cut their text's descenders (the
  vertical padding left too little room for Linux's DejaVu Sans), segmented
  buttons wrapped inside their fixed height ("Upper Left"), the Inspector's
  Supercell/Transform/Edit Cell row pushed its first button out of sight, and
  Calculation Setup squeezed its drop-downs in the narrow Inspector. Menus,
  plot menus, the Plot Options popover and the periodic table now open in one
  overlay layer above every panel (a plot's popover was clipped by its panel
  and covered by the bottom panel), stay inside the window, and follow one
  z-index scale (`--z-*` in `css/tokens.css`), with notices below menus. At
  narrow widths the main toolbar scrolls sideways rather than wrapping: at
  110% on a 1280 px wide window, for example, Setup is reached by scrolling
  the toolbar.
- **Stopping the GUI no longer leaves a half-started worker behind.** A task
  dispatched while the server stopped could start a pool worker that the
  exiting server never finished starting, which printed an `EOFError`
  traceback; the pool now takes nothing new once it is stopping.
- **Auto pre-optimization classifies a structure about ten times faster.**
  The dimensionality scan now checks the bond network once per bond length
  instead of after every bond: 0.4 s instead of 5 s for a 128-atom slab.
- **A middle click on a toolbar button or tab no longer pastes either.** As
  in the 3D viewer, the primary selection a middle click brings on Linux is
  dropped anywhere outside a text field.
- **All-caps element symbols in CIFs are read as the right element.** `BI`
  was read as boron, `CL` as carbon and `SN4+` as sulfur, and a lower-case
  `bi` failed. cif2d12, the GUI and `mace preopt` now take the two-letter
  element when the first two letters name one, and prefer the type symbol over
  the label.
- **POSCAR species lines are read in any case.** `BI S` or `bi s` no longer
  stops the GUI with a `KeyError`.
- **cif2d12's manual CIF reader works again.** When ase cannot read a CIF,
  the fallback reader stopped with `NameError`; it now reads the atoms (a
  label such as `C1A` is carbon) and drops uncertainties from cell values.
- **cif2d12 writes the CIF's asymmetric unit when it keeps the CIF's space
  group over spglib's.** When spglib found a different group (an R-3c or
  P63/mmc CIF written to 6 decimals reads as a lower one) and the CIF's group
  was kept, as batch mode always does, the deck listed every atom of the cell
  under that group (30 for R-3c Ag2Br3), so each atom CRYSTAL generated was a
  copy it had to fold back, and the GUI could not open the deck. It now lists
  the CIF's own atom records (2), which CRYSTAL expands to the same 30 atoms.
- **The GUI shows the same Fd-3m structure cif2d12 writes.** A CIF of
  diamond at (1/8, 1/8, 1/8) with no symmetry operators listed opened as 16
  overlapping atoms (ase's origin choice 1); it now opens in origin choice 2,
  as cif2d12 writes the deck, with 8.
- **`mace preopt` reads such an Fd-3m CIF the same way.** It pre-optimized
  the 16 overlapping atoms and wrote them back under Fd-3m; it now starts
  from the 8-atom diamond cif2d12 and the GUI read.
- **Job efficiency in plain numbers.** The Job view and `mace cluster job`
  now say, for example, "CPU efficiency 62% (used about 20 of 32 cores on
  average)" and "Memory peak 9.4 GiB of 64 GiB requested (15%)", also for a
  running job (from `sstat`) instead of "Not measured by SLURM". CPU time is
  summed over all job steps, not taken from the largest one, and a per-node
  `ReqMem` is counted correctly. A running job's `extern` step, for which
  sstat reports an impossible CPU time, is left out (and listed in the
  details), and an OUT_OF_MEMORY job is advised to ask for more memory.
- **GUI: clicks and selections land under the pointer at any UI scale.** At a
  scale other than 100 %, picking atoms, box and lasso selections, right-click
  menus, plot hover and splitter drags landed off from the pointer in current
  Chrome, Edge, Brave and Firefox.
- **GUI: a clear message instead of a blank 3D view without WebGL.** The
  message now says first that the 3D view needs WebGL and to turn on hardware
  acceleration in the browser's settings, then reload. It also shows when the
  browser passes the WebGL check but gives the 3D view no context, which left
  the panel blank before.
- **GUI: the 3D view is sharp at a UI scale above 100 %.** The 3D canvas was
  drawn at 100 % and stretched, so atoms and labels looked soft at 110-150 %.
  It now draws at the screen's full resolution; clicks and selections still
  land under the pointer.
- **Two workflows in one folder no longer share job-script templates.** The
  templates each later step is built from were kept only in the folder's
  `workflow_scripts/`, named by step (`submitcrystal23_sp_2.sh`), so a second
  workflow planned in the same folder overwrote the first one's, and the first
  workflow's later steps ran with the second one's cores, memory and walltime.
  An isolated (or hybrid) workflow now also keeps its own copy in
  `.mace_context_<id>/workflow_scripts/` and its steps are built from that.
  A workflow planned before this keeps using `workflow_scripts/`; when a new
  workflow is planned in its folder it first gets a copy of the templates it
  uses, so it keeps its resources. Shared-mode workflows are unchanged.
- **"Archive workflow context after completion" waits for the last step.**
  The planner's archive (and export-and-delete) option acted right after the
  first jobs were submitted: the running jobs' `MACE_CONTEXT_DIR` pointed at
  a folder that was gone, their callbacks fell back to a database in the job
  folder, and the workflow stopped after its first step. The job callback
  now carries out the action, under the queue lock, once every step of the
  workflow has finished (completed, or failed with nothing resubmitted) and
  the completed ones have been through progression. Exported databases are
  copied through SQLite, so the latest results are in them.
- **An incomplete copy of a workflow's own templates costs no step its
  resources.** A copy into `.mace_context_<id>/workflow_scripts/` that failed
  part way (over quota, say) left the folder with templates missing, and the
  engine then looked nowhere else: those steps got the generic job script
  instead of the plan's cores, memory and walltime. The copy made for an
  older workflow is now made aside and renamed into place whole, and a step
  whose template is missing from the workflow's own copy uses the folder's
  `workflow_scripts/` one (with a line saying so) before the generic script.
- **A d12 step's own expert configuration in the plan is used.** BAND, DOSS
  and the other D3 steps built their deck from the plan step's
  `expert_config_file`; the SP (the first one too), OPTn, SPn and FREQ steps
  ignored it and ran with the parent's method or the step's other settings.
  The planner writes `expert_config_file` only for D3 steps, so plans made
  by the planner are unchanged; this affects plans whose d12 steps carry the
  key (written by hand, or by tools to come such as the GUI planner). Those
  steps now take the same order as the D3 steps: the step's
  `expert_config_file` first (a path from another machine is looked up by
  name in `workflow_configs/expert_<type>_configs/`), then what they used
  before. For OPT2/SP2/FREQ-style steps the key, when present, now takes
  precedence over the per-material
  `{material}_{type}_expert_config.json` lookup. A step without one is
  unchanged and keeps the parent's settings.
- **A plan's `auto_progression: false` is followed.** `mace workflow` printed
  "Auto progression: disabled" and every job's completion callback submitted
  the next steps anyway. With it off, the callback now records the completed
  step, submits nothing and prints how to continue: `mace engine --action
  process` (with the workflow's database), which submits the next steps.
- **A workflow started from CIFs keeps one plan.** The CIF conversion saves
  the plan with the generated decks as `workflow_plan_<id>_updated.json`,
  which also matched `workflow_plan_*.json`: the workflow progress report then
  found two plans and showed no planned sequence, and the template store took
  the copy instead of the plan. Both, the GUI's batch view and the database
  rebuild now skip the copy.
- **Workflows started in the same second no longer share an id.** A
  workflow id was the start time to the second (`workflow_20261014_120000`),
  so two launches in one second (two `mace submit --progress` runs, a script
  starting several plans) wrote the same plan file and used one context. Ids
  now end in four random hex digits (`workflow_20261014_120000_3fa9`), and
  each new id is checked before use: a suffix drawn again in the same second,
  or an id whose plan file or context already exists in the folder, is
  replaced by a new draw, and the id is reserved in the workflow folder
  (`workflow_configs/.workflow_ids/<id>`, created atomically) so two
  processes starting at the same moment never get the same one. The plan
  file, context folder and output folder names follow the id as before.
  `mace workflow --quick-start` names its plan file and runs the workflow with
  one id (they matched only when both were made in the same second, otherwise
  the folder got a second plan file). Existing workflows keep their ids.
- **The 3D views turn about the unit cell's center, and show it while they
  turn.** A view turns about the point at its center, and the default view
  centered the projected outline of the drawing, so a crystal turned about a
  point that moved with the atoms and boundary atoms drawn. The default view
  (and Reset View) now centers the cell: its middle for a crystal (the middle
  of what is drawn for a larger display range), the in-plane cell center at
  the slab's mid-height for a slab, and the middle of the atoms for a
  molecule. This holds in the structure view, the comparison, the mode and
  volume views and the slab preview; edits keep the view as before, and a pan
  carries the center along. While a drag turns a view (right-drag in the
  structure view, left-drag in the others) a small crosshair marks the center
  and goes away on release. View › Center Rotation on Unit Cell and Center
  Rotation on Selection (also in the viewport's menu and the command palette)
  move the center without changing the zoom or the orientation.
- **The toolbar's display options work in the mode view of a FREQ output.**
  Show Unit Cell, Show Bonds, labels, boundary atoms, the display range
  (supercells), the atom style, the projection, View Along, the 90° turns and
  Reset View were gray there. Every periodic image of an atom now moves as the
  atom does (the modes are at Γ); for a phonon-dispersion output the view keeps
  one cell and the range says why. Bonds come from the rest geometry in every
  frame, so they no longer flicker as atoms swing past the bond cutoff; labels
  sit at the rest positions. The Inspector's 2×2×2 box follows the display
  range. Every 3D view now starts in the projection Settings › Viewport set
  (the mode, volume and comparison views drew in perspective until it was
  changed), and a refit of the mode or comparison view lands where the first
  one did. The comparison of a run's steps follows Show Unit Cell, Show Bonds
  and the atom style too. Options that cannot apply say why in their tooltip
  (editing tools in the mode view, labels and ranges in the comparison, the
  structure options in a volume, which has its own in the Inspector).
- **The mode view no longer shows scrollbars or zooms in and out while it
  plays or turns.** The structure views' 280 px minimum height applied to the
  mode view's 3D view too, so where less room was left under its head (a
  bottom panel, a UI scale of 110 % or more) its panel scrolled; a scrollbar
  took room from the 3D view, which resized it, which took the scrollbar away
  again, and the view pumped on every frame. The mode and volume views now
  fill exactly the room they have, their panels never scroll, the mode view's
  head wraps instead of widening the panel, and every 3D canvas is a block
  clipped to its viewer.
- **Band plots no longer join the two sides of a k-path break.** At a
  discontinuity such as `A|L` or `M|H`, the `mace gui` band plot drew a
  near-vertical line from the last band energy at A to the first at L.
  CRYSTAL does not repeat the k-point at a break (the tick's point ends one
  panel and the next point is already one step along the path), and the gap
  was only inserted where an x value repeated, so real BAND.DAT files never
  got one. The gap now goes after the tick's point in both layouts; the
  `A|L` label is unchanged. `mace plotting --band` already drew each segment
  on its own and is unaffected.
- **A non-magnetic spin-polarized band structure is drawn once, not as a
  solid red line over the blue one.** With SPIN on, CRYSTAL writes beta bands
  that differ from alpha only in the last printed digit (graphite: at most
  2.7 meV), and `mace gui` drew them dotted over the solid alpha set. Where
  bands are degenerate (graphite's L-H valence pair, the flat Γ-A bands) the
  dotted copies run out of phase and fill each other's gaps, so those regions
  looked solid pink. Bands that agree within 10 meV everywhere are now drawn
  once, labeled "α = β (non-magnetic)" with the largest difference in the
  note under the title; a magnetic run keeps both sets, now with beta drawn
  under alpha. `mace plotting --band` draws such a run (and a closed-shell
  run, which also writes two equal blocks) as one set too, with the legend
  "Spin ↑ = ↓ (non-magnetic)" or "Bands" instead of a dashed copy on top.
- **Both spins of a spin-polarized DOS are filled and listed in the
  legend.** In the `mace gui` DOS panel only the spin-up total had a shaded
  fill; the mirrored spin-down total (dotted, on the left) had none, and the
  legend listed only the "↑" entries. The spin-down total is now filled too,
  lighter and hatched so the two sides read apart in light and dark themes,
  the legend lists each projection once (a click hides both its spins) plus
  a key, "↑ solid, right" and "↓ dotted, left", and the note under the title
  says which side is which. `mace
  plotting --dos` already filled both sides.
- **A wide energy range no longer collapses the DOS panel.** When the DOSS
  energy window cuts through bands, CRYSTAL puts a large value in the
  second record of each spin block (graphite: 133 states/eV in one bin, under
  2 elsewhere; about four states, those below the window). With the plot's
  energy range widened past the window edge (for example -10 to +10 eV
  around a window starting at -7 eV), that bin drew a long horizontal line
  across the DOS panel and set its axis to about ±130 states/eV, squashing
  every curve onto zero. `mace gui` and `mace plotting --dos` now leave that
  edge bin out (only when it is a clear spike: three times its neighbor and
  above every value inside the window), the GUI shades the energies outside
  the computed DOS window on the DOS panel and says so in the note under the
  title, and the DOS axis keeps fitting the data within the visible range.
- **The GUI's Open Editors list no longer squeezes the file tree.** Every
  open document added a row to the explorer's Open Editors section with no
  limit, so with many documents open the file tree shrank to nothing. Each
  explorer section (Open Editors, Files, Materials.db, Recent) now scrolls
  inside its own share of the explorer, the file tree keeps at least about
  five rows (a third of a short explorer), and the active editor's row stays
  in view. Open Editors and Files now fold from their header like Recent and
  Materials.db (click, Enter or Space), and the folds are remembered with
  your profile. The section is also no longer redrawn under the pointer or
  keyboard focus every time the profile syncs.
- **k-path tick labels no longer overlap, and `mace plotting --band` puts
  each one at its own k.** Seekpath's path for a base-centered orthorhombic
  cell (oC2/oA2, e.g. slipped AB graphite: Γ-Y-F₀|Δ₀-Γ-Z-B₀|G₀-T-Y|Γ-S-R-Z-T)
  has segments much shorter than the rest when ζ is close to 1/2 (Y-F₀ and
  B₀|G₀-T, 14 times shorter than Γ-Y here). That is the standard path, not a
  generation error, but the labels of those ticks ran into each other: the
  `mace gui` band plot printed "YF₀|Δ₀" and "B₀|G₀T" on top of each other and
  turned the axis labels sideways. A label too close to its neighbor now
  drops to a second row (the plot makes room below), the labels stay
  upright, ticks at the same k merge into one "A|B" label, and the layout is
  redone when you zoom along k. `mace plotting --band` spread each segment's
  labels evenly along it, so on Γ-Y-F₀ the Y label sat half-way instead of
  next to F₀ (and on graphite's Γ-M-K-Γ-A-L-H-A every inner label was off);
  labels now sit at the BAND.DAT tick positions, a crowded label drops a row
  the same way, and the "Wave Vector" title is left out when it does (the
  axes keep their fixed place so band and DOS figures still line up).
  The three breaks of this path get gaps like any other.
- **Two queue callbacks no longer run at once.** The queue lock counted as
  expired after 5 minutes even while its holder was still running, so a
  waiting callback broke in, and the first holder's release then deleted the
  new holder's lock file, letting a third one in. It is now an OS lock (a
  POSIX record lock, held until its holder finishes or exits, with no
  expiry) on a lock file that is never deleted; it works across the nodes of
  a cluster on NFS and GPFS (checked on MSU HPCC's home and scratch). The
  lock is also one lock for every callback mode: before, completion
  callbacks and `mace manager --callback-mode full_check` / `status_check`
  each took their own lock and both could resubmit the same failed job
  through recovery. On a filesystem without working locks MACE falls back
  to an owner-token file with a heartbeat that is deleted only by its owner;
  `MACE_QUEUE_LOCK=token` forces that (for Lustre mounted with <!-- british-ok: Lustre is a file system -->
  `localflock`). The lock file is now `.queue_locks/queue_state.lock`; old
  `queue_manager_<mode>.lock` files are ignored and can be deleted.
- **`mace completion --remove-zombie-jobs` no longer cancels another
  calculation's job.** The job id was read from any `*<name>*.o` log in the
  folder, so a finished `si` took `si_sp-19000002.o` (a healthy job still
  running) as its own and canceled it; with several logs the first one the
  folder listed won. Now only the calculation's own logs count
  (`<name>-<id>.o`, as MACE's job scripts write them, `<name>_<id>.o`,
  `<name>.o<id>` and a job array task's `<name>-<A>_<T>.o`), the newest
  wins, and before anything is canceled SLURM (squeue, else scontrol) must
  confirm the job: its name is the calculation's and it runs in the
  calculation's folder, or, for an array task, MACE's record of that array
  lists this calculation as that task. A job SLURM does not confirm is
  listed as not canceled, with the reason. The prompts are unchanged.
- **The max-jobs limit counts the jobs really in the queue.** `mace manager`
  (`--callback-mode submit_new` and `full_check`) counted the entries of
  `crystal_job_status.json`, which are never removed and were also read
  from your home folder and the temp folder: an old file with 220 entries
  stopped every new submission ("Queue nearly full (220/250)") with an
  empty queue, while a really full queue did not stop anything. It now
  counts your running and pending jobs in squeue (each job array task
  counted, all your jobs, as SLURM counts them against its limit), or, when
  squeue does not answer, the database's submitted and running
  calculations. The status file is still written, and `--status` still
  shows it, but it no longer limits submission.
- **A job killed for memory before writing its output is recorded as
  failed.** The queue check asked sacct for a job's state without `-P`, and
  sacct cuts the State column at 10 characters: OUT_OF_MEMORY came back as
  `OUT_OF_ME+`, which MACE did not take as an end state, so the calculation
  stayed "submitted" forever and its recovery never ran. sacct is now asked
  for parsable output (as job array tasks already were), and the state is
  read whole (`CANCELLED by 1234` reads CANCELLED). OUT_OF_MEMORY,
  NODE_FAIL, PREEMPTED, BOOT_FAIL, DEADLINE and TIMEOUT all count as end
  states.
- **Jobs that ended days ago are not resubmitted on their own.** With the
  sacct fix above, a calculation left "submitted" by a job that ended long
  ago (OUT_OF_MEMORY, NODE_FAIL, ...) is now recorded failed on the next
  sweep. When sacct says that job ended more than 3 days ago, MACE records
  the failure with the real state and classifies its error (so `mace
  recover` finds it) but does not start its recovery; it
  prints one line per job ("job 123 (name) ended OUT_OF_MEMORY on <date>;
  not resubmitted automatically - run `mace recover --db <db>` to retry").
  Jobs that ended more recently are recovered as before. Set
  `MACE_RECOVER_MAX_AGE_DAYS` to change the 3 days.
- **Batch Setup keeps parents with the same file name apart.** Two parents
  named alike in different subfolders (`A/mat.out` and `B/mat.out`, or two
  `mat.cif`) wrote the same settings file and the same deck in the batch
  folder, so the second overwrote the first. Their settings files, CIF copies
  and decks now start with their subfolder's name (`A_...`, `B_...`), and
  `commands.sh` renames the follow-up decks the same way. Files whose names
  would still clash (`x.out` and `x.log` in one folder) are refused before
  anything is written.
- **IR spectra from IRSPEC.DAT have the right column names.** The GUI called
  the file's second column "transmittance"; it is the wavelength in nm
  (CRYSTAL23 manual, p.453). The third column is the raw absorbance, and a
  periodic run's file has the classical and two Rayleigh absorbances and a
  reflectance per inequivalent direction after it, which the plot now offers
  as components (the y axis says Reflectance when one is shown). The data
  export names every column and adds `wavelength_nm`. RAMSPEC.DAT's columns
  were already right.
- **Comparing molecules no longer shows atoms that did not move.** A
  molecule's cell is only its bounding box, and Compare measured each atom
  in fractions of each structure's own box (a CIF of the EC molecule in a
  40 Å box against its MOLECULE output: 2.4 Å "displacements" and a −93 %
  cell). A molecule, or two structures that differ in periodicity, is now
  compared in Cartesian coordinates once the rigid translation and rotation
  are taken out, with no cell table; the atom pairing works about each
  structure's center too. Crystals, slabs and polymers compare as before.
- **Build › Supercell works on slabs and polymers.** It refused every
  structure that is not periodic in 3D ("Supercells are built from 3D
  cells"), so a SLAB or POLYMER output could not be repeated. It now repeats
  along the periodic directions: the dialog keeps n at 1 along the others,
  and a matrix that would repeat or mix in a non-periodic direction is
  refused with the reason (a molecule has nothing to repeat). Atoms are no
  longer wrapped along a non-periodic direction, which would have split a
  slab whose atoms sit outside its nominal cell. Opening the Matrix tab now
  carries the n1 × n2 × n3 values over; it left the matrix empty.
- **No BAND, DOSS or TRANSPORT deck for a molecule.** `mace opt2d3` (and so
  Setup and Batch Setup) wrote a BAND deck with a triclinic k-path for a
  MOLECULE output, which CRYSTAL does not run: the CRYSTAL23 manual says BAND
  "does not run for molecules" (p.309) and DOSS "do not run for molecules"
  (p.320), and BOLTZTRA's transport comes from the bands. `opt2d3` now stops
  before asking anything, says why and writes no deck; Batch Setup's scan
  marks a molecule not eligible for these three, and Setup shows the reason.
  CHARGE and POTENTIAL still work for molecules. Slabs and polymers keep
  their in-plane and along-chain k-paths.
- **A keybinding row's Reset no longer binds a key twice.** Reset gave the
  command its default keys back even when another command had been given one
  of them since, so one key ran two commands with no warning. It now asks
  first, as recording a key does: Replace takes the key back from the other
  command, Cancel leaves both as they are.
- **A crashed or out-of-memory-killed run no longer reads as "still
  running" for 10 minutes.** An output that ended with CRYSTAL's fatal
  `ERROR ****` banner, or with the MPI launcher's "BAD TERMINATION … KILLED
  BY SIGNAL: 9" (an out-of-memory kill) or an `MPI_Abort`, had none of the
  end markers MACE looked for, so the GUI showed it live (and Batch Setup
  "still running") until it was 10 minutes old. These lines now end a run
  at once, everywhere MACE decides whether an output is live. None of them
  appears mid-run in any output of the test corpus.
- **A volume slice outside the grid is refused with the valid range.**
  `api/volume/slice` with a plane index past the grid's last plane failed
  as a server error (500); a negative or non-numeric index did too. The
  index is now checked against the cube's header before anything runs, and
  answered with 400 and "index runs from 0 to N". The volume view resets a
  stored plane that does not fit the grid it opens.
- **A molecule's output is known by CRYSTAL's own banner.** CRYSTAL23 prints
  "MOLECULAR CALCULATION" (every molecule output in the test corpus), but the
  GUI looked for "MOLECULE CALCULATION". Molecules were still found through
  the later "DIMENSIONALITY OF THE SYSTEM 0" line, so a finished output read
  the same, but a molecule's output still being written, cut before that
  line, read as a 3D crystal. The workflow planner's stand-in outputs wrote
  the same wrong banner, so `opt2d3` took a planned molecule for a crystal;
  now it refuses a BAND, DOSS or TRANSPORT step for it, as it does for a
  real molecule output.
- **Decks of the same name in different folders no longer share a scratch
  directory.** The job scripts ran CRYSTAL in `$SCRATCH/crys23/<deck name>`
  (properties: `crys23/prop/<deck name>`), so `x.d12` in two folders used one
  directory: at the same time they overwrote each other's `INPUT` and
  `fort.9`, and one after the other the second restarted from the first
  one's `OPTINFO.DAT` or `fort.20`. The directory now records the folder whose
  deck uses it (`.mace_submit_dir`), and a deck of the same name from another
  folder runs in `$SCRATCH/crys23/.mace_by_dir/<checksum>/<deck name>`. A deck
  rerun from its own folder keeps its directory, so OPTGEOM RESTART and GUESSP
  restarts work as before; a directory without a record (an earlier MACE, or a
  `fort.20` you staged by hand) belongs to the first deck that runs in it.
  The `OPTINFO.DAT`, `fort.20` and `fort.9` a rerun brings over from the
  directory its last run used are taken only when that directory is still
  this folder's (or has no record), never from one a same-named deck in
  another folder has claimed since; each run writes its claim again, so a
  purged scratch is reclaimed by the next run.
  Jobs already queued run their old script and are unchanged.
- **A properties job runs on its own `.f9` and copies back only what it
  wrote.** Its scratch directory was never emptied, the `.f9` was copied
  there unchecked and every result file was copied back unconditionally: with
  `<name>.f9` missing, Pproperties ran on the `fort.9` an earlier run left,
  and a run that failed copied the earlier run's `BAND.DAT`, `fort.25`,
  `DOSS.DAT`, ... back as its own result. The directory is now emptied before
  staging, the job stops at once with the reason in `<name>.out` when
  `<name>.f9` is missing or empty, and only files this run wrote come back
  (`fort.9` only when Pproperties rewrote it). The SCF job script likewise
  copies `fort.9` back only when CRYSTAL wrote it in this run, so an old one
  left in scratch no longer replaces `<name>.f9`. `mace submit` refuses a
  `.d3` whose `.f9` is missing or empty and says which (`--nosubmit` still
  writes its job script).
- **`mace submit` exits 1 when a job was not submitted.** Without
  `--track` it printed "Submission complete" and exited 0 even when every
  `sbatch` call failed. It now ends with how many jobs were not submitted
  ("none of the 2 job(s) was submitted", "1 of 3 job(s) not submitted") and
  exits 1, also when only some failed or a job script could not be written
  (with `--nosubmit`: "Script generation incomplete"), so scripts that run
  `mace submit` can tell. A folder with both kinds of decks submits both and
  exits 1 if either part failed. `--track` already exited 1.
- **Job folders and scratch directories with a space in their path work.**
  The job scripts expanded `$DIR` and `$scratch` unquoted (`cp $DIR/$JOB.d12
  $scratch/$JOB/INPUT`, `cd $scratch/$JOB`, `>& $DIR/${JOB}.out`), so a deck
  in `my calcs/` staged nothing, CRYSTAL wrote no output and the copy-back
  failed. Every expansion in the generated scripts is quoted now, the CRYSTAL
  launch line included (`mpirun -n "$SLURM_NTASKS"
  "$EBROOTCRYSTAL/bin/Pcrystal"`; shellcheck reports no SC2086 or SC2164),
  and the job stops with a message when it cannot enter its scratch
  directory instead of running CRYSTAL elsewhere (after CRYSTAL, a failed
  `cd` back to the submit folder is reported and the steps that follow use
  full paths). The site settings' launcher still replaces the older
  unquoted launch line in a template copied by an earlier MACE. A deck *name* with
  a space or a shell character (`si (copy).d12`) cannot be a SLURM job name
  or the script's `JOB` variable: `mace submit` now refuses it up front, says
  to rename it and exits 1 (it wrote a broken script and still reported
  success), and the generators refuse such a name too. The site settings'
  `[mpi]`/`[crystal]` launch lines still apply to a template copied into a
  workflow folder by an earlier MACE.
- **`mace submit -i` keeps to the partition's time limit it announces.**
  Where the default partition allows less than the template's 7 days, MACE
  said "partition short allows at most 2-00:00:00" but only the plain path
  applied it: `-i` (and `--exclude-nodes`, `--walltime`, which write the
  `-t` line themselves) still asked for 7 days, and SLURM rejected or held
  the job. The default walltime, the one `-i` offers and the one you accept
  there are now capped to the partition's MaxTime on every path. A walltime
  you give (`--walltime`, or typed at the `-i` prompt) is written as given,
  with a warning when the partition allows less, also with `--track`.
- **`--walltime`, `-i` and the node-exclusion options no longer miss an
  existing job script.** `mace submit` reuses an existing `<name>.sh` unless
  `--overwrite-sh` is given, and the options went only into a newly written
  script: a reused one ran with its old resources and nothing said so. Job
  scripts now end with a line holding the `cksum` of the lines above it. With
  resource options, a script MACE wrote and nobody changed since is written
  again with them; a script that was edited (or written before 1.1.5) stops
  the submission before anything is queued (and before the `-i` prompts),
  naming the scripts and `--overwrite-sh`. Without resource options existing
  scripts are reused as before.
  again with them; a script that was edited (or that an older MACE wrote
  with other than its default resources) stops the submission before
  anything is queued (and before the `-i` prompts), naming the scripts and
  `--overwrite-sh`. A script an older MACE (1.0.0 to the 1.1.5 preview)
  wrote with its default resources, unchanged since, is recognized and
  written again (with or without options). A script MACE wrote from a
  customized template (a workflow plan's `workflow_scripts/` copy with the
  plan's cores and memory, or `-i`) is not written again from the plain
  template under `--walltime` or node exclusion: only the walltime or
  exclusion asked for is set in it, so it keeps its cores and memory (`-i`
  still writes it again with the resources you type). Without resource
  options other existing scripts are reused as before.

- **`mace submit --track` honors `-i` and keeps an edited job script.** The
  tracked path regenerated `<name>.sh` from the template every time: `-i` was
  ignored and a hand-edited script was silently replaced. It now follows the
  untracked path's rules: `-i` asks for the resources and the tracked jobs
  get them; an edited `<name>.sh` is submitted as written (MACE adds only the
  line that points its completion step at the database); a script MACE wrote
  and nobody changed is written again as before (so is one an older MACE
  wrote with its default resources), except one written from a customized
  template (a workflow plan's copy, `-i`), which is submitted with its own
  cores and memory (`--walltime` set in it); and with `-i`, `--walltime`,
  `--progress` or `--array` an edited script stops the submission before
  anything is recorded or queued. `--overwrite-sh` replaces any script.
  (`-i` with `--progress` says it is not applied: the plan sets the
  resources.)
- **The final energies of a multi-point geometry optimization count D3 and
  gCP once.** From the second point of an OPT on (and from the first point
  of a restarted one), CRYSTAL's `TOTAL ENERGY(DFT)` line already includes
  the dispersion and gCP corrections, and the extractor added them again:
  `total_energy_corrected_au` was low by D3 + gCP (Ti11O14 by 0.63 AU, a
  4LG 3c OPT by 0.094 AU), `total_energy_au` held the corrected energy, and
  `total_energy_plus_d3_au` was point 1's. The corrected total is now the
  `OPT END` energy, and the plain and D3-only totals are it minus the final
  corrections. 52 outputs of the test corpus change (every multi-point OPT
  with D3 or gCP); one-point OPTs and SPs do not.
- **A multi-point geometry optimization no longer stores its first
  geometry's SCF values as the run's.** CRYSTAL prints an OPT's SCF in full
  only at point 1; the extractor took the gap kind (`direct_band_gap`,
  `indirect_band_gap`, `band_gap_type`), `fermi_energy` from the EFERMI
  lines and the `+++ ENERGIES IN A.U. +++` components (kinetic, e-e, e-n,
  n-n, exchange, correlation) from there (Ti8O13: 0.76 eV indirect at point
  1 against 1.04 eV at the end). Only what follows the last optimization
  point counts now: the gap and Fermi energy CRYSTAL prints after OPT END,
  and a final geometry printed as conducting (FERMI ENERGY after OPT END)
  after gapped cycles at point 1 has `band_gap` 0. Point-1-only values are
  no longer stored for these runs. 51 multi-point OPTs of the test corpus
  change; one-point OPTs and SPs do not.
- **A spin-polarized geometry optimization stores its `band_gap`.** With
  CRYSTAL's `ALPHA BAND GAP` / `BETA BAND GAP` summary after OPT END, only
  the two channel gaps were stored, and the classification took the alpha
  one. `band_gap` is now the gap the rest of the extractor already used
  (the fundamental gap of the final geometry's SCF when CRYSTAL printed its
  band edges, else the smaller channel), and the classification uses it
  (Ti11O14: 0.1403 eV, not alpha's 0.1826). 87 OPTs of the test corpus gain
  a `band_gap`; 6 change `classification_band_gap`.
- **The final geometry of a slab or molecule optimization is stored.** The
  final cell was read only from a PRIMITIVE CELL line with `VOLUME=` and a
  density, and the final atoms only under an `X/A Y/B Z/C` header, as a 3D
  cell prints them. A slab prints no volume and `X/A Y/B Z(ANGSTROM)`, a
  molecule `X(ANGSTROM) Y(ANGSTROM) Z(ANGSTROM)`, so neither got
  `final_primitive_*` or `final_atomic_positions`, and a slab's
  `primitive_*` kept the input cell (with its area as
  `primitive_cell_volume`). Both are read now, and the new
  `final_atomic_position_axes` names the Cartesian columns of such a table.
  9 slab and 66 molecule OPTs of the test corpus change; 3D runs do not.
- **A slab's cell area is stored as `primitive_cell_area` (Å²), not as a
  volume.** CRYSTAL heads the cell measure VOLUME in every output, but for
  a slab it is the area of the 2D cell (and for a polymer the length of
  its cell), and the extractor stored it as `primitive_cell_volume` and
  `initial_primitive_cell_volume` in Å³. For dimensionality 2 these are now
  `primitive_cell_area` / `initial_primitive_cell_area` (Å²), for
  dimensionality 1 `..._cell_length` (Å); `primitive_cell_volume` is kept
  for 3D only. A slab OPT's final cell, which CRYSTAL prints without its
  area, gets `final_primitive_cell_area` = a·b·sin γ (a polymer's
  `final_primitive_cell_length` = a), and `primitive_cell_area` is the final
  one, as the other `primitive_*` values are. The completeness check and the
  unit tables know the new names. 27 slab outputs of the test corpus change
  (8 OPT, 19 SP).
- **`mace database --db-path PATH` uses PATH.** The option was in the help
  but never read: every action opened the workflow context's database or
  `materials.db` in the current folder (creating an empty one there). Every
  action now reads (or, for `clean`, changes) the given file, and a path with
  no database is an error ("No database at ...") rather than a new empty
  database.
- **No more `final_final_*` and `initial_initial_*` properties.** The
  extractor prefixed the final and initial geometry's keys again when they
  already carried the prefix, so every OPT stored its atoms twice
  (`final_atomic_positions` and `final_final_atomic_positions`, and
  `initial_initial_atomic_positions` in place of `initial_atomic_positions`),
  and an export carried the doubled tables (13,500-character cells). The
  keys keep their one prefix now; an OPT's initial atoms are
  `initial_atomic_positions`. An existing database's doubled rows are left
  out of exports like the other structure tables. 193 outputs of the test
  corpus lose the doubled keys, with the same values under the single names.
- **The electronic ZT of a TRANSPORT run uses the electronic thermal
  conductivity, and is named as electrons-only.** KAPPA.dat holds
  CRYSTAL's zero-field coefficient κ₀ (its header's "K coefficient",
  CRYSTAL23 manual Eq. 14.3), not κₑ; κₑ = κ₀ −
  T·σ·S². `transport_zt_max` was PF·T/κ₀, which can never reach 1; it is
  conductivity, and is named as electrons-only.** KAPPA.dat holds what the
  CRYSTAL23 manual calls κₑ (Eq. 14.3; the file's header says "K
  coefficient"): the zero-field coefficient κ₀, measured with no electric
  field. MACE derives the electronic thermal conductivity at zero current,
  κₑ = κ₀ − T·σ·S², from it. `transport_zt_max` was PF·T/κ₀, which can never reach 1; it is
  now PF·T/κₑ = x/(1 − x) with x = PF·T/κ₀ (C1-RCSR-bcu-f: 0.935 → 14.42;
  afi 0.951 → 19.55; ana 0.928 → 12.88). κ₀ and κₑ at the ZT peak are
  stored as `transport_kappa0_at_zt_max` and
  `transport_kappa_electronic_at_zt_max` (W/(m·K)). A point where T·σ·S² ≥
  κ₀ is left out with a warning, and a run with no usable KAPPA.dat point
  stores no `transport_zt_max` (was 0). This ZTₑ counts the electrons
  only: the lattice thermal conductivity is left out, so it is an upper
  bound on the material's ZT that grows without limit as T·σ·S² approaches
  κ₀ (values of 10-20 near a band edge are expected, not an error). It is
  now also stored as `transport_zt_electronic_max`; `transport_zt_max`
  keeps the same value under its old name. The Database view explains these
  properties on hover, `transport_power_factor_max` shows its unit
  (W/(m·K²), was "dimensionless"), and the GUI's transport plot labels
  KAPPA.dat as κ₀. docs/cli.md (`mace analyze`) describes them. 3
  TRANSPORT outputs of the test corpus change.
- **`num_imaginary_frequencies` counts imaginary modes as the GUI's run
  analysis does.** It counted every printed row with a negative frequency:
  a molecule's rigid-body modes (-0.007 cm⁻¹), noise of a few cm⁻¹, and a
  degenerate row once. Both now use one rule
  (`mace/utils/imaginary_modes.py`): without the 3 (periodic), 6
  (molecule) or 5 (linear molecule) modes closest to zero, each mode of a
  degenerate row, and only those of at least 20 cm⁻¹. The run analysis'
  own fallback when the dimensionality is unknown now also recognizes a
  molecule from its `DIMENSIONALITY OF THE SYSTEM    0` line. 65 FREQ
  outputs of the test corpus change (e.g. 2 to 0 for a molecule with a
  -11 cm⁻¹ mode, 1 to 3 for a triply degenerate -164 cm⁻¹ mode).
- **The advanced BAND/DOSS analysis reads wrapped DOSS.DAT rows and gives
  a BAND.DAT's gap.** It read each physical line of a DOSS.DAT as an energy
  point, but CRYSTAL wraps a record with many projections over several
  lines and repeats the energy grid per spin: a 1002-point file read as
  4008 points, and a molecule's `band_gap_advanced_ev` came out as
  68.76 eV (true 7.84). It now uses the extractor's own DOSS.DAT parser.
  A gap read off a DOS is not the material's (CRYSTAL puts E_F on a
  molecule's broadened HOMO peak, and a DOSS band range leaves states out),
  so a DOSS.DAT alone stores no `band_gap_advanced_*` and no
  `electronic_classification_advanced`. A BAND.DAT alone stored a gap of 0
  and "unknown"; its gap now comes from the band edges, taken by band
  rather than by the sign of each eigenvalue (a valence-band top a few
  micro-Hartree above E_F made a 7.8 eV gap 0.16 eV): it matches
  `band_dat_band_gap_ev` within 0.01 eV for all 72 corpus outputs that
  have one. The effective masses, which start from these band edges, now
  take the hole mass from the valence band and the electron mass from the
  conduction band. A hole mass without an electron mass no longer fails
  the analysis. 156 BAND/DOSS outputs of the test corpus change.
- **A database build records the formula and dimensionality the output
  shows.** Every material the build registered was stored as a CRYSTAL (a
  molecule too), and its formula was counted from the .d12's atom lines, which
  hold the asymmetric unit only: TiPbO3 was stored as TiPbO2, TiO (space group
  189, 3 Ti + 3 O in the cell) as Ti2O, and a material with no .d12 beside its
  output as "Unknown". The formula is now the reduced formula of the cell
  CRYSTAL prints (a molecule's whole formula), read with the same structure
  reader as the GUI's structure view, the cell formula is kept in the
  material's metadata (`cell_formula`), and the dimensionality is MOLECULE,
  POLYMER, SLAB or CRYSTAL as the output says. A deck is only the fallback, for
  an output with no geometry, and is expanded by its space group first. In an
  existing database the next build corrects a formula MACE took from a deck's
  atom lines and a CRYSTAL it wrote for a molecule, polymer or slab; a formula
  set any other way (typed by hand) is left as it is.
- **A database build no longer files properties under materials it has no
  row for, and its summary counts what it did.** With Register on, the build
  still extracted every `.out` it saw: failed runs (which it rightly does not
  register) and MACE's `X_temp.out` copy of a run whose `X.out` sits beside
  it got their properties under a material id taken from the file name, with
  no materials row, so the Database view could not open them (the material
  page answered 404) and an export left them out; the `_temp` copy was also
  registered in place of the run's own output. Such outputs are now listed as
  Skipped (a new class in the preview and in `--dry-run`) and not extracted,
  the populate scan registers `X.out` rather than its copy (with the
  `X_temp.d12` deck), an older record of the copy is matched to `X.out`, and
  an output a build extracts with Register off gets a materials row (formula
  and dimensionality from the output). With Register off a failed or
  unfinished output is still extracted (as `mace analyze --extract-properties`
  without the build options does), under its own materials row; only a build
  that registers skips it. A skipped output's line says why ("a copy of
  X.out", "the same run as ...", "a failed or unfinished run with no
  calculation record"), as `--dry-run` does. Outputs extracted again
  although unchanged (a build without `--skip-unchanged`) are counted as
  "N re-extracted" in the summary, which read "0 new, 0 changed" after
  re-extracting 11. The CLI's last line and the GUI's
  notification are now the same sentence from one function, for example
  "Database updated: 11 new, 0 changed, 0 failed outputs; 8 materials (8
  new); 5 skipped"; the notification used to count files as "new" (11 for a
  folder of 5 materials) and the CLI counted the orphaned ids as materials.
  Rows an earlier build filed for failed runs stay in an existing database
  (a build deletes nothing); they are not shown, as before.
- **A CIF with no `_chemical_formula_sum` gets its formula from its atom
  sites.** The fallback read the word after the `_atom_site_type_symbol`
  header, which is the next loop tag, so such a CIF got the formula
  "_atom_site_fract_x". The atom-site loop is now read row by row (the type
  symbol, else the label; charges such as `O2-` dropped).
- **The Database view's CSV and JSON export keeps text-valued properties.**
  The GUI's export read only the numeric column, so a property stored as text
  (e.g. `electronic_classification`) came out empty, where `mace database
  --action export` wrote it; the GUI's export also kept the large structure
  fields the CLI leaves out, and where two calculations of a material gave
  one property the two exports could pick different values. Both now take
  their values from one function, and a test compares every cell of the two
  CSV files, not only their headers.
- **The Database view finds a calculation's files after the project folder
  was moved or copied.** A database keeps the absolute paths of the folder it
  was built in; after a move (or with a copy opened from another folder) the
  material page said "Not in this folder" for every input and output. When
  the recorded path is not in the served folder, the file is now looked for
  under the database's folder by the end of the recorded path
  (`/old/project/OPT/x.out` -> `OPT/x.out` beside the database), and a bare
  file name under the recorded work folder found the same way. The database
  is not changed.
- **The Create Database dialog's command shows the backup it will run.**
  For a new database the command at the bottom of the dialog left out
  `--backup --keep-backups N` while the build ran with them when Back up the
  database first was ticked (the default), so the copied command was not the
  one that ran. The dialog now sends the tick and the command follows it.
- **Registering a copy of a run whose SLURM job id is already recorded no
  longer leaves a half-made calculation.** The same run in two folders (its
  `.out`, `.d12` and `-<job>.o`) made the second registration stop with
  "UNIQUE constraint failed: calculations.slurm_job_id" after the
  calculation row was written: the row stayed `pending` with no output file
  and the build filed the copy's properties under it. An identical copy
  (same job id, same size and content) is now the same calculation: not
  registered again and not extracted (Skipped); a different output with the
  job id is registered without it, with a warning; and a registration that
  fails part-way takes its row back out. In an existing database the next
  build completes such a left-behind row with its output (no job id); no row
  is deleted.
- **Extracting without Register files a material's outputs under one id,
  the one a registration gives.** Without `--register`, an output's rows went
  under its file name with one suffix stripped: `1_dia_opt_rev1` for the OPT,
  `1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized` for its SP and
  `2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized` for a FREQ, where `--register`
  (and the workflow) file them under `1_dia` and `2_dia2`. Untracked rows now
  use the same function as the registration. In an existing database, rows
  an earlier untracked extraction filed under the old ids stay where they are
  (nothing is deleted); a new extraction writes under the new id.
- **Two different structures whose outputs have the same name in different
  folders are two materials.** `A/mat_opt.out` (diamond) and `B/mat_opt.out`
  (TiO) both map to the material id "mat", so a database build (CLI or GUI)
  filed both runs under one material. When a new calculation's structure
  clearly differs from the material its id names (another reduced formula,
  or the same formula in another space group, read from the outputs), it now
  gets the id plus its folder, as Batch Setup names same-name parents
  ("mat_B"; a short hash of the folder is added if that id holds yet another
  structure), and the build says so. The same structure in another folder,
  such as a copied project, stays the same material. A calculation already
  recorded keeps its material: rows an earlier build merged stay where they
  are.
- **`mace submit` lists MSU HPCC's node options only on MSU HPCC.** The
  help of every cluster showed `--exclude-amd20` and `--exclude-mendoza`.
  Where MSU HPCC's node choices do not apply (another SLURM cluster), the
  help no longer lists them, and `--exclude-amd20` is still accepted but
  warns and excludes nothing, as `--exclude-mendoza` already did. On MSU
  HPCC the help and the job scripts are unchanged.
- **No developer machine paths in code that runs for users.**
  `setup_mace.py --update-templates` looked for (and would have rewritten)
  one developer's checkout path in the job-script templates; the templates
  find `MACE_HOME` at run time, so it now only checks that they are there.
  The unused `code/OldSLURMTemplates/submitcrystal23_env_aware.sh` names a
  placeholder instead of that path. A test keeps such paths out.
- **Setup no longer answers a CLI question it does not know.** Calculation
  Setup (and Batch Setup) runs `mace convert`, `opt2d12` and `opt2d3` and
  answers their questions; a question no answer named got Enter, the CLI's
  default. So a reworded prompt turned a GUI answer into a default without
  notice, and the deck changed. Enter now goes only to the questions Setup
  knows are meant to keep their default; any other question stops Setup
  with "did not ask what Setup expected" and quotes it. A question with no
  default to keep (opt2d3's custom band "Path:", for one) stops the same
  way instead of being sent empty answers. Choosing an LDA functional (VBH)
  in Setup now sets it; the question went unrecognized and SVWN was kept. A
  new test runs every Setup route through the real CLI and fails on any
  question nothing matched.
- **`mace gui --slurm` leaves no tunnel behind when it is killed.** The ssh
  tunnel to the job's node runs in a process group of its own, so killing
  the session with SIGKILL (`kill -9`, an out-of-memory kill) skipped its
  cleanup: on Linux only the ssh itself was told to stop, and only when the
  tunnel was opened from the main thread; an ssh wrapper's other processes
  (a ProxyCommand, for one) kept running, and on other systems nothing
  stopped. The tunnel now runs under a small guard that ends its whole
  process group within a second of the session's death, from any thread
  and on any POSIX system.
- **The Jobs table's header no longer lets rows show above it.** Scrolled,
  the sticky header stopped at the panel's top padding, so a strip of rows
  stayed visible above it, and its bottom line scrolled away with the
  first row. The header now sticks to the panel's very top, keeps its line,
  takes the theme's surface color (light, dark or any theme built on the
  tokens) and paints above the rows, the pinned actions column too, while
  staying below menus.
- **The Explorer lists new files without a reload.** A file that appeared
  in the folder from outside the GUI (a job writing its `.out`, a copy from
  a shell) showed only after reloading the page, or for a page with an open
  document, after switching back to it. The server now notices a file added
  to or removed from any folder the Explorer lists within about 2 seconds,
  with one stat of each folder (up to 500 folders, within the Explorer's
  depth; no file is read), and the page lists the folder again. Polling
  rather than inotify, so jobs writing from other nodes of a cluster's
  shared filesystem are seen too.
- **A lattice vector of 40 Å or longer gets one k-point, not 12.**
  The k-point rule gives each direction the smallest shrinking factor k
  with 40 < k·L < 80 Å, but its search started at k = 2, so a vector that
  already reaches 40 Å never matched and fell back to 12: a cluster in a
  44 × 44 × 48 Å box (`mace convert`, an `opt2d12` deck without a parent
  SHRINK, the workflow dummy files) was written with `SHRINK 0 24 /
  12 12 12`. Such a vector now gets k = 1, and the factor never grows with
  the length. In the test corpus 131 of 261 CIFs, all molecules or
  clusters in a vacuum box, change from `0 24 / 12 12 12` to `0 2 / 1 1 1`
  when converted as CRYSTAL; every deck of a cell under 40 Å is
  byte-identical. The dimensionality is not changed for you: convert such a
  structure as MOLECULE to drop the periodic images altogether.
- **The completion checker, the recovery detector and the queue manager
  now agree on why a run failed.** Each had its own error patterns:
  the completion checker had no OOM or Fermi patterns and filed an OOM kill
  (`KILLED BY SIGNAL: 9`) and a diverged SCF (`ERROR **** ZERO **** FERMI
  ENERGY NOT IN INTERVAL`) as "Segmentation fault/runtime error"; the
  detector took the first matching line, so mpirun's BAD TERMINATION banner
  printed above `KILLED BY SIGNAL: 9` made an OOM kill a `system_error`
  with no recovery, and it called a NEIGHB stop (atoms too close) a basis
  set linear dependence. All three now read one table
  (`mace/utils/crystal_errors.py`) in one priority order: an OOM kill
  first, then CRYSTAL's own errors, then the generic MPI/SLURM lines. On
  the 707 corpus outputs, the 4 failed runs now classify the same way
  everywhere (OOM kill and segfault-then-kill: memory; Fermi stop: SCF
  convergence; NEIGHB: geometry), and the other 703 classify as before.
  The detector now reports the queue manager's error type names
  (`convergence_error`, `timeout_error`, `disk_space_error`,
  `geometry_error`; older names stored in the database are still mapped),
  `ANISOTROPIC SHRINKING FACTOR` now reaches the SHRINK recovery, and an
  error with no automatic recovery says why.
- **`mace opt2d3` takes the output's own wavefunction and no longer lets
  two outputs share a deck name.** `X_opt_sp.out` is named `X_opt_*`, and
  the wavefunction search tried `fort.9` and `X_opt.f9` (the
  optimization's) before `X_opt_sp.f9`, so its BAND/DOSS deck could run on
  the wrong wavefunction; the output's own `.f9`/`.f98` is now tried first.
  `X.out` and `X_sp.out` side by side both wrote `X_band.d3` (and
  `X_band.f9`), the second over the first (the `X_sp`/`X_opt` pair was
  already kept apart); and `X_opt_sp.out` took the `X_opt_*` name that
  `X_opt.out` keeps when it collides. Beside another output with (or
  making) the same name, each output keeps its whole name; alone, the name
  is as before.
- **The static k-path of a C-centered orthorhombic cell is in CRYSTAL's
  basis and for that cell.** With `--static-kpath` (or `MACE_STATIC_KPATH=1`)
  every C-centered orthorhombic cell got SeeK-path's oC1 path ("oS1") with
  the numbers SeeK-path writes in its own standard cell and a C₀/Σ₀
  parameter for some other cell, behind a warning. Read in CRYSTAL's
  primitive basis those points name other points and can lie outside the
  first Brillouin zone. The variant is now SeeK-path's (oC1 when a ≤ b,
  else oC2; oC1 for the groups that do not tell a from b, as spglib
  standardizes them), and its points are evaluated for the cell and moved
  to CRYSTAL's basis, as for the other centered lattices: the static deck
  is the deck the seekpath library writes. The oC2 entries' labels now
  name their segments. A corpus test checks that every point of the path
  opt2d3 writes, library and static route, lies in the first zone of the
  cell CRYSTAL runs. (BAND decks written before 1.1.4 also had the library
  route's points in SeeK-path's cell, for any lattice whose standard cell
  differs from CRYSTAL's, graphite AB slip's Y, F₀ and Δ₀ among them; 1.1.4
  fixed that route: write such decks again.)
- **The band + DOS plot's legends no longer cover the data.** One legend
  spanned both panels in two rows inside the plot, over the top bands, the
  y tick labels and the DOS curves, and listed every projection twice
  (↑ and ↓). The band legend (α/β, or "α = β (non-magnetic)") now sits
  above the band panel and the DOS legend to the right of the DOS panel, in
  margins sized to the legends as drawn at any window size and UI scale
  (PNG/SVG exports too); each projection is listed once, with one spin key
  ("↑ solid, right", "↓ dotted, left") for a spin-polarized DOS. The
  subtitle is one short line, the rest of its guidance in a help tooltip.
  `mace plotting --band/--dos` (and so the Publication Figure) puts its
  legend right of the axes and their labels, the figure widening when it
  needs the room, the axes keeping their size so band and DOS figures still
  line up; the DOS band-gap note heads the legend instead of a box over the
  curves.
- GUI split view: when a document was slow to open to the side and you
  moved the focus to the left group meanwhile, the right group took the
  focus (and the Inspector) back once the document was in. The focus now
  stays where you put it.
- GUI: dragging a document tab selected the tab text. In Chromium, the next
  drag from that tab then started the browser's own drag, which canceled
  the drag onto the right half that splits the editor. Tabs no longer
  select text or start a native drag.
- **A task started as the GUI stops no longer starts a worker.** When the
  first pool task was dispatched while the server was stopping, it built a
  fresh process pool and started a worker the exiting server then left half
  started. The stop now marks the tasks stopping first; a pool is never
  built after that and a new task is refused ("The server is stopping.").
- **A slab compared with its own CIF no longer shows a wrapped atom as a
  move.** Two ends that differ in periodicity were compared in Cartesian
  coordinates with no periodic image at all, so an atom wrapped by +a in
  one of them showed up to 7 Å of displacement (and RMSD). Along the axes
  periodic in both ends each atom is now taken at its nearest image.
- **`mace gui connect` takes a literal absolute path under your home
  folder.** A path under this computer's home folder was always sent as
  `~/...`, so a node that has the very same absolute path (a shared file
  system mounted at the same place) could not be given it. A leading `//`
  now marks a literal path (`--dir //home/you/runs` is sent as
  `/home/you/runs`, never rewritten); the bash and PowerShell connect scripts
  do the same, and docs/ssh-setup.md says so.
- **GUI: a click on an IR or Raman peak opens that peak's mode at any UI
  scale.** Above 100 % the click was measured in screen px against the
  plot's own px, so it opened the mode of a marker up and to the left of
  the pointer (or none).
- **GUI: menus, a plot's Units/Export menus and Plot Options, and the
  periodic table follow the UI scale.** They open in a layer outside the
  zoomed window shell, so above 100 % they stayed at 100 % (the plot's menus
  had been scaled before they moved there). The layer now takes the UI
  scale; each still opens next to its button and flips to stay inside the
  window.
- **GUI: the Compare view of an optimization fits a 1280x800 window again.**
  Its longer legend (the arrows' scale and the cell's change, each on a row
  of its own) and a 240 px floor under the 3D view made the tab scroll by
  about 90 px at 100 % UI scale (over 200 px at 110 %), hiding the Frame
  slider and Edit This Step. The legend is now one wrapping line with the
  cell's change cut to one line (its whole text as the tooltip), the
  toolbar is tighter, and the 3D view gives way first, down to 150 px. At
  110 % the tab may still scroll a little at that window size; the Frame
  slider and Edit This Step are reached by scrolling it.
- **GUI: Run Analysis' Cell plot keeps its data area at high UI scale.** At
  125 % in a 900 px window, the legend's eight entries became one entry per
  row and then moved to a column right of the volume axis. That left the
  plots 86 x 220 px of a 463 x 300 px figure (seen with plotly 7.1). Plotly
  wraps a legend that does not fit on one row into a grid as wide as its
  widest entry, here the FINALRUN restart key. Each entry is now its own legend group, so the rows wrap as wide as
  the entries are: four rows above the plot, and the plots take 265 x 112 px.
  The legend fit also judges rows at the right margin they will have, not
  the wider one a column left behind.
- **GUI: the Submit dialog's site details pass the `dlitem` accessibility
  check.** Each term and value sat in a `<div>` inside the `<dl>`, which
  axe-core (as bundled with axe-selenium-python) reports as a serious
  violation. The `<dt>`/`<dd>` pairs are now direct children of the `<dl>`,
  which is the two-column grid itself. No other GUI markup uses `<dt>`/`<dd>`.

#### Testing

- A symlinked `test/` corpus is ignored by git (`/test` in `.gitignore`).
- The GUI database-build browser tests (`tests/gui/test_db_build_e2e.py`)
  pass alone and in any order; each builds the database it needs.
- The database-build CLI tests (`tests/test_db_build_cli.py`) compare
  against the base tree knowing that the doubled geometry keys
  (`final_final_*`, `initial_initial_*`) are now stored once under their
  single prefix. They run only with the `test/` corpus, so CI never saw
  them fail.
- The corpus deck test (`test_every_corpus_deck_opens_or_says_why`)
  accepts a SLAB, POLYMER or MOLECULE deck that now opens the `.out` of
  its run, and checks that the `.out` it names exists.
- `tests/fake_slurm.py`'s sacct cuts the State column as SLURM's does, and
  its fake jobs no longer leave a GUI server running.
- **No plot's legend, label or note covers its data any more.** The same
  rule now holds for every plot, GUI and CLI. The IR/Raman spectra drew
  their legend in two rows over the curves and the peak wavenumbers over
  the peaks: the legend is now a column right of the plot and the peak
  labels sit on an axis above it (a weaker peak too close to a stronger one
  goes unlabeled). Run Analysis wrote the thresholds, TOLDEE, SCF
  MAXCYCLE, the FINALRUN restarts and the reason the optimization stopped
  as text on the plot; each is now a legend entry with its line style.
  Transport and Compare Convergence legends sit above the plot. A legend
  too tall for the pane moves to a scrolling column, and the margins follow
  the legends at any window size and UI scale. The 2D map's corner labels
  stay inside the plane, clear of the tick labels, and every plot's
  subtitle is one line, its full text in the help tooltip. `mace plotting
  --ir/--raman` and the DOS's band-edge composition notes moved their
  legend and notes right of the axes too.
- **The GUI's Submit preview of a properties (.d3) job matches the real
  submit again.** The preview copies a folder into a scratch mirror with an
  empty stand-in for each .f9, and since `mace submit` refuses a .d3 whose
  .f9 is empty, the preview reported "b.f9 is empty ... Not submitted" for
  a job the real submit sends. The stand-in now keeps the real .f9 empty or
  not, so the preview refuses exactly what the real run refuses.
- **A job recovered from an SCF error keeps the resources set in its job
  script.** Recovery writes a fixed `*_recovery_*` deck and generated that
  deck's job script from the template, so the ntasks, memory, walltime,
  partition, exclude and account changed in the original `<job>.sh` were
  lost on resubmission. The new script now takes the original's `#SBATCH`
  resource lines (job name and output stay the recovery deck's); an
  out-of-memory recovery still raises the memory from the value set there.
  When the original is not a job script MACE can read, the template's
  resources are used and one line says so.
- **The GUI's Setup shows the wavefunction opt2d3 copies.** Since opt2d3
  looks for the output's own `<stem>.f9` first, and names an output's files
  after its whole stem when another output beside it has the same name, the
  Setup panel could name a different .f9 (`X.f9` for `X_sp.out`) than the
  command it prints would copy. It now follows the same search.
- The mode view's size test (`tests/gui/test_freq_view_size_e2e.py`) also
  plays in a headless Chromium that never gives the page focus (Chrome for
  Testing 140).

### The 1.1.5 preview (2026-10-08)

Everything since 1.1.4. New: **the MACE GUI (`mace gui`)**, the folder's
structures, optimizations, plots and `materials.db` in your browser, with a
structure editor, builders and Calculation Setup that saves the JSON the CLI
reads, on your machine or on a cluster node through an ssh tunnel; pre-optimized
starting geometries (`mace preopt`, `mace mlip`; CRYSTAL still optimizes); a new
logo; charged systems and ion clusters (fragments found
by connectivity, CHEMOD/DOPING/spin planned for you, per-fragment Mulliken
charges), `mace defects` (defect deck sets and formation energies, with the 3D
charged-cell offset and image-charge corrections measured on CRYSTAL23),
`mace phonopy` (finite-displacement phonons through phonopy 4.x), SLURM job
arrays for many decks at once (`mace submit --array`, `mace workflow --array`),
and one basis file per family with an opt-in TZVP-REV2 2022 update and the SOC
basis records.
**Band paths from an OPT parent now use the optimized cell** (they used the
initial cell), static band paths compute their lattice-dependent points for the
cell, and long-axis cells keep the band-path ISS at most 999; regenerate BAND
decks made from OPT parents. Also fixed: the bundled Pu basis record, the BAND
gap of doped runs, ECP Mulliken net charges, odd-electron and charged-parent
decks, 1.1.1 plans' functional names, piped answers to interactive runs, and
several menu and job-script fixes. Workflow BAND steps now write the deck
opt2d3 writes for the same choices (NSUB 1000; a default step is opt2d3's
Enter-through label path), opt2d12 children of a
SUPERCEL/SUPERCON parent are the supercell in P1, and the bundled La-Lu and
Ac-Es basis records are neutral. Several of these change generated decks -
see Fixed below. Coming from 1.1.1-1.1.4, read Upgrading first.

### Upgrading from 1.1.1-1.1.4

- **Refresh the band gaps of existing databases.** Earlier versions could
  store `indirect_band_gap`, `direct_band_gap` and `band_gap_type` from an
  earlier SCF cycle (cycle 0 of an SP) instead of the converged one, a
  `band_gap` from the last gapped cycle of a run that ended conducting, and,
  for a spin-polarized run, one channel's gap instead of the fundamental gap
  across both channels (see Fixed). Run `mace analyze --extract-properties
  <folder> --db-path <materials.db>` again on the folders of a database to
  replace them; the rows of an extraction are rewritten, so a stale gap is
  removed.
- **What to install.** seekpath 2.1 or newer is now required; everything
  else new is optional (phonopy for `mace phonopy`, xtb and an ML Python
  environment for `mace preopt` and Clean, a browser on your computer for the
  GUI). No shell startup file is needed: every new setting lives in an
  optional file under `~/.config/mace/` or an optional environment variable.
  INSTALLATION.md, "What MACE needs", lists each requirement, how MACE finds
  it and what changed since 1.1.4, including every new environment variable.
- **`mace --help` lists the commands in groups.** The usage line and the
  positional entry now read `COMMAND` instead of listing every command; a
  mistyped command's error now reads `argument COMMAND: invalid choice: 'x'
  ...` (was `argument command: ...`; the `invalid choice: 'x'` part and exit
  code 2 are unchanged). `mace help` (an error before) now prints the help;
  `mace help gui` prints the GUI group, not `mace gui --help`. No command,
  option or per-command help changed.
- **`mace cluster` is a new command; no existing command changes.** It keeps a
  small cache in `~/.cache/mace/cluster/<cluster>/` (about 20 files, counted
  against a home file quota) and reads optional `[cluster]`, `[quota]`,
  `[scan]` and `[devnodes]` tables of `~/.config/mace/site.toml`. Once a day
  it looks up a `quota` alias in your login profile (`bash -lc`, 20 s at most).
  With the GUI open on a cluster it asks SLURM for your jobs once per
  `cluster.pollSeconds` (60 s) while a cluster view is visible or a job of
  the folder is queued.
- **Site settings for job scripts.** With no `~/.config/mace/site.toml` and
  no `MACE_SLURM_*`/`MACE_CRYSTAL_MODULE` in the environment, every job script
  keeps its bytes on MSU HPCC for a member of `mendoza_q`, and wherever SLURM
  and modules cannot be asked (a workstation); a `site.toml` written by `mace
  site --init` changes nothing until you uncomment a line. On any other
  cluster the built-in values are checked first: an account you are not a
  member of is left out (SLURM charges your default account), Python modules
  that do not load are replaced by the Python that runs MACE, the completion
  step's `--max-jobs/--reserve` follow your MaxSubmitJobs, and the default
  walltime is lowered to your partition's MaxTime; MACE says each once, and
  `mace site` lists them. MSU's development-node names and node-exclusion
  choices appear only on MSU HPCC. If the CRYSTAL module MACE was built for
  is not available, job scripts now load the best CRYSTAL 23.x module that has
  Pcrystal and Pproperties (it used to fail with `bin/Pcrystal: No such file
  or directory`), a lower-case `crystal/...` modulefile too; a notice and a
  `# MACE:` line in the script say so. MACE
  never switches to another CRYSTAL major version on its own: set
  `crystal_maximum` (or `crystal`) in `~/.config/mace/site.toml` to allow one.
  Pin a module with `MACE_CRYSTAL_MODULE=<module>`, or keep the old line with
  `MACE_CRYSTAL_MODULE=none`. An invalid value in the file or in those
  variables now stops generation with exit code 2 instead of writing a broken
  script. Completion callbacks reuse the module their parent job loaded and
  never probe modules.
- **Site settings need Python 3.11 to be read.** On an older Python with a
  site settings file present, MACE now stops with exit code 2 (it ignored the
  file and wrote the built-in values); MACE already requires 3.11. A
  `~/.config/mace/site.toml` that group or others can write gets a warning
  (its values still apply); `MACE_SITE_CONFIG=none` now also ignores group
  files. A `[resources]`, `[mpi]`, `[paths]` or `[crystal]` table you put in
  site.toml by hand earlier ("unknown keys ignored") now applies.
- **`mace run` off MSU HPCC is no longer capped at 1h55m per deck** where the
  node sets no CPU-time limit (`ulimit -t`); the 30-minute default per deck
  stays. On MSU HPCC the cap is unchanged; `[local_run] max_time` sets it.
- **Workflows on a cluster with a MaxSubmitJobs below 1000** get queue limits
  that fit it at plan time (`--max-jobs 190 --reserve 10` for 200), said once.
- **Workflows planned before you set an account or modules in site.toml keep
  the account and module stored in their plan** (`workflow_configs/*.json`),
  including plans made by MACE 1.1.4, which store `mendoza_q` explicitly. Plan
  again to use the new settings. The planner's account prompt now defaults to
  the site account. A quick-start plan without a stored account (1.1.4's
  `mace submit --progress`) gets the site account when its steps are built.
- **Submitting from the GUI writes only what `mace submit` writes**, plus, for
  a tracked submit, one backup of the folder's `materials.db` in `backups/`
  (the newest three are kept) and File History copies of the files it
  replaces. Batch Setup writes a `batch_*/` folder only when you click
  Write. Jobs the GUI submits get the environment `mace gui` was started
  with, as from that shell.

- **`mace gui` is new; nothing else changes.** It needs no new packages: it
  uses ase, spglib and plotly from `requirements.txt`, and shows an install
  hint if your install is missing one. Plain `mace` commands do not load it.
- **It writes only what docs/gui.md lists:** files you save in `--dir`
  (structures, CIF, settings JSON, Calculation Setup's files and the decks you
  ask for), `~/.config/mace/gui.json`, `~/.cache/mace/gui`, temporary files
  under `$TMPDIR` (removed at exit), a 0600 state file holding its token while
  it runs (`$XDG_RUNTIME_DIR/mace-gui-<port>.json`, removed at exit), and,
  only when you use them, `~/.config/mace/connect.toml` (`mace gui connect`
  profiles) and `~/.cache/mace/modules.json` (the xtb module search).
  `materials.db` is opened read-only unless you run Create or Update Database
  from a Folder (below); no schema changes.
- **It listens only on 127.0.0.1.** Another address needs `--host` together
  with `--insecure-network`, because the token then travels unencrypted; the
  ssh tunnel it prints is the supported way to reach it from a laptop. The
  token is random unless you set one (`MACE_GUI_TOKEN` or `--token`), which
  must be at least 16 characters, each a letter, a digit or `. _ ~ -`.
- **Choose your own port on shared nodes** (`export MACE_GUI_PORT=8xxx`):
  browser drafts are kept per port (everything else in the GUI profile follows
  you to any port).
- **GUI defaults to GPU 0** for its GPU features (none yet) and for
  `--open --gpu`'s help; a pre-release GUI's stored GPU 1 falls back to GPU 0
  once: pick it again in Settings › Advanced or set `MACE_GUI_GPU=1`. Shift+A,
  Shift+P and the Boundary Atoms checkbox now change the open document only
  (the defaults stay in Settings › Viewport). A pre-release GUI's
  `gui.json` becomes the version-2 profile on its first save (a copy is kept
  as `gui.json.v1.bak`); a pre-release GUI writing it afterwards keeps only the
  settings.
- **The printed link signs in once** (`?login=`): the same URL from the
  browser's history signs nobody in. `mace gui login-url` prints a new link,
  and the sign-in page takes the token; the token works only there and in an
  `X-Mace-Token` header, never in a URL or a cookie. A pre-release `?token=`
  link or bookmark shows the sign-in page. A session ends after 12 hours
  without keyboard or mouse input (Settings › Server › Sign out after: Never
  restores the old behavior); tasks are not affected. When stdout is not a
  terminal, the token is in the state file (`.json`, not the earlier `.url`
  file): read `base` from it and use `mace gui login-url`.
- **`mace gui --open` no longer starts a browser on an ssh node without a
  display** (a text browser would take over the terminal); it says so.
- **MACE GUI never reads credential or secret files** (`.ssh/...`,
  `.config/gh/...`, `.config/gcloud/...`, `.netrc`, `.gnupg`, `.aws`,
  `.azure`, `.kube`, `.docker`, `.git-credentials`, `.pgpass`, `.pypirc`,
  `.npmrc`, `.vault-token`, `.env` and `.env.*`, shell and interpreter
  histories, ssh private keys `id_rsa`/`id_ed25519`/...), even when it serves
  your home folder: they are not listed and show as "hidden for safety". It
  never writes into a hidden folder.
- **xtb may now be found through your cluster's module system** where GFN-FF
  said "not found": when xtb is not on PATH, MACE loads the newest xtb module
  6.7 or newer for xtb's own process (`MACE_XTB` still wins;
  `MACE_XTB_MODULE=<module>` pins one, `MACE_XTB_MODULE=none` turns the search
  off). The wrapper of the earlier HPCC notes keeps working.
- **New environment variables:** `MACE_GUI_ROOTS`, `MACE_GUI_MANAGED_TIMEOUT`,
  `MACE_GUI_MANAGED_GRACE`, `MACE_GUI_SESSION_IDLE_HOURS`,
  `MACE_GUI_SESSION_MAX_DAYS`, `MACE_GUI_IDLE_STOP_HOURS` and
  `MACE_XTB_MODULE`; new gui.json keys
  `session.idleHours`, `session.maxDays` and `server.idleStopHours`.
- **The README header is now an SVG logo;** the ANSI banner in the terminal is
  unchanged.
- **`mace preopt` and `mace mlip` are new.** xtb and the ML environment are
  optional and external; nothing new is required. `mace mlip fetch` is the
  only MACE command that downloads; the weights go to
  `$XDG_DATA_HOME/mace-tool/mlip/` (or `MACE_MLIP_DIR`), apart from
  mace-torch's own `~/.cache/mace`. A pre-optimized geometry is a starting
  point only: run a CRYSTAL OPT from it before SP/FREQ/BAND/DOSS.
  `mace preopt` refuses a .d12 whose records edit its geometry (SUPERCEL,
  ATOMREMO, ATOMSUBS, GHOSTS, ...): pre-optimize the expanded structure (the
  .out of a run, or a CIF exported from it).
- **A CIF from `mace preopt`** (or saved from the GUI after a GFN-FF/ML
  Clean) carries one `# mace-preopt:` comment line; every other save is
  unchanged. `mace convert` prints one stderr warning for such a CIF when the
  deck is not an OPT; its prompts, defaults and decks are unchanged.
- **`mace gui` checks in the background at start what Clean can offer:** a
  short `xtb --version` when xtb is found, and an import probe of the ML
  interpreter only when one is recorded (up to 20 s, cached for 7 days in
  `$XDG_CACHE_HOME/mace/preopt/`). Startup does not wait for it. Its
  temporary files, and those of every GFN-FF/ML run, go under
  `<tmp>/mace-preopt-<uid>/`, which stays as an empty directory.

- **Install seekpath 2.1 or newer; it is now required** (`pip install -r
  requirements.txt`, or `pip install --upgrade seekpath`). seekpath 2.0.x
  lacked the rotation matrix the CRYSTAL-basis path needs and silently gave
  the static path; it now counts as missing. Without it, a SeeK-path band path of a 3D cell
  (opt2d3 format 4, a configuration or plan step with `seekpath_full`,
  including Basic and quick-start BAND steps planned before 1.1.5 that stored
  the SeeK-path format) and an opt2d12 SeeK-path phonon dispersion now stop
  with the install command and no deck, where they used to write the static
  HPKOT table path with a note. On MSU HPCC the workflow callback generates
  decks with the module Python, so install it there: on a dev node (e.g.
  `ssh dev-amd20`), `module load Python/3.11.3-GCCcore-12.3.0
  Python-bundle-PyPI/2023.06-GCCcore-12.3.0`, then `pip install --user
  seekpath` (INSTALLATION.md, "HPC / SLURM clusters"). To keep the static
  path, pass `--static-kpath` to `mace opt2d3` / `mace opt2d12`, or set
  `MACE_STATIC_KPATH=1` (for a workflow, in the environment its jobs
  inherit). Slab and polymer paths, the default label path and the k-point
  vector, literature and custom paths are unaffected.
- **Workflows planned before 1.1.2 produce mixed decks.** The plan's SP
  basis, tolerance and functional overrides and its FREQ settings
  (tolerances, NUMDERIV, IR/Raman) now apply, but only to steps generated
  after the upgrade. Finish such workflows on the old version or re-plan
  them.
- **1.1.1 plans with a custom functional** stored prose names (HSEsol-3C,
  HF-3C, M06-2X, HF, ...). They are now rewritten to the CRYSTAL23 keyword
  when the plan is loaded, with a warning. A -D3 suffix CRYSTAL23 has no
  parameters for is dropped, and an unknown name stops that step. Nothing to
  do unless a step is refused; then re-plan it.
- **`workflow_scripts/` is a snapshot** taken when the workflow is planned.
  A template copied before 1.1.3 that the plan names is now rebuilt from the
  plan (current generator, the plan's resources and directives) when the
  engine uses it; the old copy is kept as `<name>.pre-1.1.3`. A template the
  plan does not name is left alone, with a warning: re-plan, or copy the
  current `mace/submission/submitcrystal23.sh` and `submit_prop.sh` into it,
  to get the empty-`$SCRATCH` guard and `.f20` staging.
- **`mace submit` reuses existing `<job>.sh` files.** Scripts generated
  before 1.1.3 lack the scratch guard and GUESSP staging; regenerate them
  with `mace submit --overwrite-sh` (MACE now warns when it reuses one).
- **Job scripts and `workflow_scripts/` copies written by 1.1.3-1.1.4 are
  reused without a warning** and lack this release's job-script changes:
  two-component GUESSP handling (a 2c job's own `.f9` is no longer staged as
  fort.20), keeping a hand-staged `fort.20`, staging `<job>.ext` as fort.34
  (`mace phonopy` decks) and the `.mace_scratch` cleanup. For TWOCOMPON
  GUESSP restarts, hand-staged densities and phonopy decks, regenerate them
  with `mace submit --overwrite-sh`, or re-plan the workflow.
- **GUESSP from a hand-staged density** can go next to the deck as
  `<job>.f20`, or into the scratch directory as `fort.20`.
- **`<submit dir>/.mace_scratch` is now removed** after a run that ended
  normally. Do not keep files of your own there.
- **Piped answer files must be regenerated.** opt2d12 now asks for the
  integration grid, asks "Keep the basis set from the original D12? [Y/n]"
  before the basis menus when the parent deck has an external (EXTERNAL)
  basis and the functional is not a 3c method (the menus follow only on
  "n"; an internal-basis parent, a 3c functional and --shared-settings get
  the basis menus as before, with no new question), takes LEVSHIFT as one "LEVSHIFT ISHIFT
  ILOCK" answer (was value, then cycles; a single value is now asked again),
  and asks for the MAXTRADIUS value and the HISTDIIS size. opt2d3 ECHG
  offers order 0-1 (was 0-2) and adds "Number of points along the B-A
  segment"; a `--config-file` run without a terminal needs
  `map_points`/`map_coord_type` in the config. The planner's SP "Use tight
  convergence?" and OPT "Use tighter convergence?" questions are 0-3 level
  menus (y/yes and n/no are still taken, with a note). Shifted answers can
  land in other questions. Three invalid answers in a row to any question
  now stop a piped run, and a piped opt2d12 run lists every setting that
  differs from the parent deck, but a shifted answer that is valid for the
  question it lands on cannot be caught; use `--save-options` and
  `--config-file` instead of piping.
- **Planner and quick-start FREQ steps no longer write NUMDERIV 2.** New
  plans use CRYSTAL's default; plans saved before 1.1.2 keep 2. Set NUMDERIV
  2 to compare with older runs.
- **FREQ children of a FREQ parent with IR/Raman intensities.** A piped
  opt2d12 answer file for such a child (the parent's FREQCALC block has
  intensity or spectrum settings) now meets "Select template (0-8) [0]": a
  blank answer keeps the parent's settings (was Basic), "1" still picks
  Basic, and the questions that followed the template are not asked after
  "0". A parent with only NUMDERIV gets the menu as before. In the planner's
  Advanced FREQ level, a blank "Select calculation type" now keeps the
  parent's intensities ("1" is an explicit none).
- **Planner phonon dispersion steps.** "Use Fourier interpolation
  (INTERPHESS)?" now defaults to no (blank was yes), and a band structure
  step stores opt2d12's automatic BANDS path. Plans saved before 1.1.5 keep
  the INTERPHESS 2 2 2 and the deck with no BANDS path they stored; re-plan
  such steps.
- **opt2d12 templates saved by 1.1.1** do not record a "no" to MAXTRADIUS or
  SPINLOCK, so they now pick up the parent's values. Re-save the template
  if that is not wanted.
- **Re-organizing sorts previously misfiled SP/FREQ outputs** into
  `completesp/` and `completefreq/` (1.1.1 put them in `complete/`).
  The same scan corrects the `calc_type` of a record an older MACE typed
  from its file name (an SP named `*_OPT_*` recorded as OPT); workflow types
  (OPT2, SP2, ...) are left as they are, and `calc_id` keeps its old name.
- **`mace/database/utils/populate_completed_jobs.py` was removed.** Use
  `mace/database/populate_completed_jobs.py`.
- **Stale property rows.** Re-extracting a calculation now removes the rows
  it no longer produces (1.1.1's band fields on DOSS runs, optimization rows
  on OPTGEOM-titled SP runs, doped BAND gaps). Rows stored without a
  calc_id cannot be attributed to one output: `mace analyze
  --extract-properties` removes those of a material when it re-extracted
  every output of that material it can see. With `--material-id`, or when
  `--filter-type`/`--calc-type` left one of the material's outputs out, they
  are kept; rebuild the database to clean those.
- **Workflow BAND steps planned before 1.1.5.** Plans store the BAND
  settings they were made with: a planner BAND step keeps the NSUB 10000 it
  stored (new plans and the engine default use opt2d3's 1000), and a step set
  in the planner's Advanced menu keeps keys opt2d3 does not read ("labels"
  writes k-point vectors; literature, SeeK-path and custom paths write a deck
  with no path, NLINE 0). Basic and quick-start BAND steps keep the k-point
  vector or SeeK-path format they stored (new plans and the engine default
  write opt2d3's Enter-through label path), and Basic, Advanced and
  quick-start FREQ steps keep the Very tight tolerances they stored, which
  replace a FREQ parent's own. Re-plan such BAND and FREQ steps.
- **Regenerate opt2d12 children of SUPERCEL/SUPERCON parents** (e.g. defect
  supercells). They were the host's space group with the supercell's cell
  and asymmetric atoms, not the parent's crystal.
- **Regenerate decks with La-Lu on the bundled triplezeta/doublezeta records,
  or Ac-Es on any bundled family.** Their shell charges did not add up to a
  neutral atom, and CRYSTAL stopped with INPBAS "UNIT CELL NOT NEUTRAL".
- **Basis sets are one file per family.** Old `basis_sets/<family>/` paths
  still work, including at the custom-path prompt. Outside tools that read
  the per-element files must use `basis_library.read_record`.
- **Job arrays change nothing unless you ask for them** (`--array`, or a
  plan's `job_arrays` key): without them the job scripts, the sbatch calls and
  the database rows are what 1.1.4 wrote. What does change for everyone:
  - The planner's custom SLURM directive menu no longer has option 5
    (`--array=1-10`), and an `--array`/`-a` directive is refused; the other
    options keep their numbers. A piped answer stream that chose 5 now stops
    as out of step. A plan that holds an `--array` directive has it dropped
    with a warning: it made every job of that step ten copies of one
    material in one scratch directory.
  - Queue counts (`mace manager --status`, the monitor's queue health) count
    each pending task of any job array, yours from other tools included, as
    SLURM counts them against MaxSubmitJobs, so they can be higher than
    before.
  - MACE's own squeue and sacct calls set `SLURM_BITSTR_LEN=0` (in their own
    environment only) so array task lists print in full.
  - `mace run` also refuses a deck that is a queued or running task of a MACE
    job array, as it refuses a same-named job.
  - A database written by 1.1.5 with job array rows (`A_T` job ids) should be
    read by 1.1.5: an older MACE does not expand the pending task lists of
    squeue and keeps such rows as they are.
- **materials.db on a cluster file system uses SQLite's rollback journal.**
  SQLite's WAL mode is not safe when processes on different nodes share a
  database on NFS, GPFS, Lustre or BeeGFS, which is how completion callbacks, <!-- british-ok: Lustre is a file system -->
  the login node and the GUI use it. The first write by 1.1.5 to such a
  database switches it out of WAL when nothing else has it open (the `-wal`
  and `-shm` files go away); on a local disk nothing changes. A 1.1.4 process
  switches it back on its next write, without losing data: upgrade every MACE
  install that writes the same database. On mounts without cluster-wide locks
  (Lustre `localflock`/`noflock`, NFS `nolock`, BeeGFS by default) MACE's <!-- british-ok: Lustre is a file system -->
  writers also take turns through `<db>.mace-writer.lock` and say so once.
  `MACE_DB_JOURNAL=wal|delete` and `MACE_DB_LOCKING=trust|degraded` override
  this; set them in `~/.bashrc` (jobs inherit them) or not at all. FUSE
  mounts count as network file systems too (sshfs, automounted research
  storage, and `fuseblk` disks such as NTFS or exFAT USB drives): a database
  there gets the rollback journal and the writer lock file.
- **A busy database no longer re-runs a completion callback.** A "database is
  locked" error after the callback had submitted a follow-up used to re-run
  the whole callback (without its lock on the plain path), which could submit
  the follow-up twice. Now the job records its own row and leaves the rest to
  the next sweep (`mace manager --callback-mode completion`). Any other error
  raised after a submission no longer falls back to an unlocked re-run either;
  an error before any submission keeps that fallback. A callback that
  waited 60 s for the queue lock also records its own row instead of nothing.
- **`mace analyze --extract-properties` files a tracked calculation's rows
  under that calculation's material.** Rows an earlier extraction filed under
  a material id guessed from the file name are moved (by UPDATE, never
  deleted) on the next extraction, and an "Unknown" placeholder material gets
  its formula and space group. A query by the file-name id for a tracked
  calculation then finds nothing. Untracked outputs keep their file-name ids.
  The new flags `--register`, `--dry-run [--json]`, `--skip-unchanged`,
  `--backup [--keep-backups N]` and `--progress-json` are opt-in; `--backup`
  writes `<db dir>/backups/materials_backup_YYYYMMDD_HHMMSS.db` (and the
  structures.db beside it) and keeps 3.
- **The GUI's energy card now plots point 1 with CRYSTAL's D3/gCP correction**
  (see Fixed); runs with a dispersion or gCP correction show different (correct)
  first steps. Its run cache is rebuilt once.
- **The GUI never writes over outputs and keeps what it replaces.** Save,
  Save As and Calculation Setup refuse a calculation output, a wavefunction, a
  data file, a database, a MACE job script or the deck of a queued job (409,
  with the reason); before any other replace the current version goes to File
  History in `~/.local/state/mace/gui/history/` (up to 500 MB of inputs and
  1000 MB of output copies by default; `MACE_GUI_HISTORY_DIR` moves it). A
  write is refused when that copy cannot be made.
- The Settings "Energy unit" (Units and numbers) is now used by plots.

### Added

- **Two editor groups side by side in the GUI** (View › Editor Groups › Split
  Editor Right, Ctrl+\\; Open to the Side; drag a tab across): any two
  documents, every kind, either group editable; Link Cameras turns two 3D
  views together; a spectrum's peak opens its mode in the other group, and
  Color Density by Potential, Volume Difference and Compare the Two Groups'
  Structures take the other group's document. The Inspector follows the
  focused group and each group has its own bottom panel (its atoms, timeline
  or modes). The split comes back per folder.
- **MACE GUI as a SLURM job: `mace gui --slurm` and `mace gui connect
  --slurm`.** One small batch job (`mace-gui`; 2 CPUs, 4G and 4 hours by
  default, partition/account/QOS/constraint from the flags, the connect
  profile or your site settings, never one cluster's built-in values) runs
  the server on a compute node; MACE waits while it is pending (Ctrl+C
  cancels that job only), forwards a port to the job's node, and reattaches
  to it later. The job ends, and its allocation is released, when the server
  stops: Ctrl+C, Sign Out and Stop Server, 1 hour without use, or 60 s before
  the time limit (warnings at 15 and 5 minutes; the status bar shows the
  job). Every SLURM read goes through the shared rate-limited cache. See
  docs/gui.md#run-it-as-a-slurm-job.
- **The Submit dialog's Site row shows every resolved site value** (job size,
  MPI launch, scratch, QOS, constraint, executables, each with its source) and
  has a **Check** button that runs `mace site check` (sbatch --test-only only).
- **`mace --help` lists the commands in groups** - automated workflows,
  monitoring, manual calculations, results, special systems and structure
  tools, GUI, setup - with a note on which command shows your jobs and a
  Quick start that begins with the GUI and includes `mace site` for a new
  cluster. `mace help GROUP` prints one group and `mace help COMMAND` a
  command's help; docs/cli.md starts with the same groups.
- **`mace cluster`: cluster status and job progress (SLURM), read-only.**
  Nodes by CPU type or partition with free cores, memory and GPUs in the
  partitions your accounts may use; `fit` (where a request could start now)
  and `--predict` (`sbatch --test-only`, nothing submitted); reservations;
  storage and file-count quotas from the site's own tool, lfs, mmlsquota,
  quota or df, warned at 80/90 %; `files`, a throttled scan of where your
  files are with cleanup commands to copy (never deletes); your association
  and QOS limits, use and fairshare, and `--check-submit` advice before a big
  batch; `job ID`, with efficiency only when SLURM measured it, progress and
  a nested ETA (total, stage, inner loop) from the live output for
  optimizations, single points and frequencies (CRYSTAL's stop rules and its
  default FINALRUN restart for optimizations; the IR/Raman CPHF stages and
  FREQINFO.DAT restarts for frequencies, never extrapolated from the central
  point), backtested on the test corpus with family-grouped held-out scores,
  and what happens if it times out; `estimate`, a runtime range and walltime
  suggestion from similar finished runs; `devnodes --pick`. Every query is
  cached and rate-limited (one node query per minute, shared by all your MACE
  windows); standard library only, so it runs on a cluster's bare module
  Python. See docs/cli.md#mace-cluster.
- **The GUI's Cluster view and Job documents** (the same data): Overview,
  Nodes, Will It Start?, Storage (with the scan), Limits and, where a module
  system exists, Software (the CRYSTAL and xtb modules MACE would load) tabs; a Job
  document that follows a running job's progress; a status-bar chip when
  storage is nearly full; optional browser notifications when your jobs
  finish (Settings › Cluster). One server-side poller asks SLURM for your
  jobs, at most once a minute.
- **MACE on any SLURM cluster: the rest of the job script in site.toml, group
  settings, and `mace site check`.** New optional keys change only their own
  lines: the job size (`[resources] ntasks, nodes, cpus_per_task, mem_per_cpu`
  or `mem`, `[resources.properties]` for Pproperties jobs; `mace submit -i`,
  `--walltime` and the planner offer them as defaults), how CRYSTAL starts
  (`[mpi] launcher = "srun"`, `srun_mpi`, `env`), CRYSTAL from a directory
  instead of a module (`[crystal] bin_dir`, `scf = "MPPcrystal"`), `[modules]
  purge = false`, `[slurm] qos`, `constraint` and `extra` directives, a shared
  scratch candidate (`[paths] scratch`, never guessed) and `[local_run]
  max_time`, each with a `MACE_*` variable. Admins and labs can share settings
  in `/etc/mace/site.toml`, `$XDG_CONFIG_DIRS/mace/site.toml` or the files
  `MACE_SITE_GROUP_CONFIG` names; each person's site.toml overrides them per
  key, and a group file others can modify is refused. Values that reach a line
  bash runs are checked to be data only, in the file and again when a
  workflow plan is replayed. `mace site` shows every value with its file (and
  a group file's owner) and whether the job size fits a node (`sinfo`);
  `mace site check` tests account, partition, QOS, job size, CRYSTAL, MPI,
  the completion Python and scratch, and asks `sbatch --test-only` about a
  generated job script. sbatch refusals for QOS, node size, features and time
  limits now get a one-line hint naming the key. With none of the new keys
  set, every job script keeps its bytes. MACE submits with SLURM only; PBS,
  Torque and LSF are planned for MACE 1.2 or later.
- **Site settings for the job scripts MACE generates (account, partition,
  modules), `mace site`, and a CRYSTAL module fallback within the same major
  version.** `~/.config/mace/site.toml` (or `MACE_SLURM_ACCOUNT`,
  `MACE_SLURM_PARTITION`, `MACE_CRYSTAL_MODULE`) sets what `mace submit`,
  `mace run`, job arrays, workflows and completion callbacks write into job
  scripts; `mace site` shows each value and its source, checks the account
  against your SLURM associations, and `--init` writes a commented template.
  `mace run` and `mace preflight` probe the module the scripts load. With
  no setting, MACE works out of the box on any SLURM cluster: it checks the
  built-in account (`sacctmgr`), Python modules (a module probe), queue
  limits (`scontrol show assoc_mgr`) and the partition's MaxTime, caches the
  answers for a day, and writes only what applies; `[queue] max_jobs,
  reserve` set the limits. See docs/cli.md, "Site settings", and
  INSTALLATION.md, "Using MACE on another cluster".
- **Submit, Run on This Node, Jobs and Batch Setup in the GUI.** Submit…
  previews the exact `mace submit` command in a copy of the folder (read-only
  SLURM queries reach the cluster, state changes are only recorded), shows
  the resources, the files it and the jobs will write, the site settings and
  every conflict (a finished output, a deck already queued, an account that
  is not yours), and submits only on its own button; Run on This Node does
  the same with `mace run`. Jobs lists your own jobs (arrays grouped, the
  folder or all), follows them, reports jobs that leave the queue, and
  cancels single jobs and array tasks only. Batch Setup sets up decks for a
  folder's CIFs or for completed runs that need a follow-up, each follow-up
  keeping its parent's settings, and writes a `batch_*/` folder with the
  CLI's settings files, the decks and a `commands.sh` that re-creates them.
  See docs/gui.md, "Submit and Jobs" and "Batch Setup".
- **The MACE GUI: `mace gui [--dir DIR] [--open]`.** A local browser app,
  served by a standard-library server bound to 127.0.0.1 with a session
  token, for the calculations in a folder: an explorer, document tabs, a
  structure viewer and editor with symmetry (Sym/P1, undo, `.bak` on the
  first overwrite, drafts), builders (supercell, slab, point defects,
  transform, symmetrize), Clean Geometry (springs, GFN-FF, ML potentials or
  Auto, run out of process, the spring result kept as the fallback), the
  OPT trajectory before/after, BAND/DOS, IR/Raman and FREQ-mode documents,
  Compare, coordination polyhedra, SVG/PNG export, the `materials.db`
  browser, and Calculation Setup, which asks the CLI's own questions and
  saves the JSON settings file `mace convert`/`opt2d12`/`opt2d3` read (the
  same deck as the manual route). On a cluster it runs on a dev node and
  prints the `ssh -N -J ... -L` line for your laptop (`export
  MACE_GUI_GATEWAY=<login host>` fills in its gateway); the 3D views draw on
  the laptop's GPU. No new dependencies. See docs/gui.md and try
  `examples/gui`.
- **GUI viewer: per-document Range & boundary and Rendering.** Each structure
  document has its own display range (fractional from/to per axis, e.g. Cell
  + 10% = −0.1 to 1.1), boundary atoms, bonded atoms outside the range (one
  shell, or whole discrete molecules; chains, layers and 3D networks are
  found by the rank of their bond cycles and never grown), polyhedra, atom
  size and bond radius, and its own look (presets Default, Flat, Publication
  and Presentation; depth cueing, light strength and direction, ambient
  light, ambient occlusion, outline, background). Settings › Viewport holds
  the defaults, which every document follows until it changes a value; Save
  as Default and Reset per group. View only: nothing reaches decks, CIFs or
  the settings JSON.
- **The GUI profile on the server.** Layout, Recent files, remembered choices
  and each document's view options are kept in `~/.config/mace/gui.json` and
  `gui-docprefs.json` next to it (not only in the browser), so they come back
  on a new tunnel port, in another browser and on another computer sharing
  your home; the first paint uses them too, and a change made just before
  closing the tab is saved. Only unsaved drafts, Zen Mode, plot options and
  the overwrite question stay per browser. Two servers saving at once keep both
  changes (a file lock and a read-back check).
- **GUI: Settings › Advanced exports, imports and resets the whole profile**
  (you choose the parts; Recent files and per-document options are off by
  default when exporting; a reset keeps a copy first, the newest 3), and a
  browser without WebGL gets an explanation in the viewport instead of a blank
  canvas, with the rest of the GUI working.
- **GUI: a way out of Zen Mode** (a chip at the top right, Esc, the viewport
  menu), and the browser tab names the folder ("MgO.cif — project — MACE");
  the menu bar shortens paths under `$SCRATCH`, `$WORK` or `$PSCRATCH` too.
- **CI: a headless browser smoke job** (`gui-smoke`: Firefox and Chromium with
  software WebGL on the example folder), which fails, never skips, without a
  browser.
- **`mace gui connect`: start MACE GUI on a cluster node and open it in your
  computer's browser with one ssh session.** It logs in through the gateway
  (`--hop jump`, `--hop nested` for nodes that accept logins only from the
  login node, or `auto`), starts `mace gui --managed` in the folder you name,
  forwards a free local port and opens a one-time sign-in link; Ctrl+C stops
  the server, and a dropped connection stops it unless GUI tasks still run, in
  which case `mace gui connect` again reattaches. Profiles in
  `~/.config/mace/connect.toml`; it needs only Python 3 and ssh on your
  computer. `scripts/mace-gui-connect.sh` and `scripts/mace-gui-connect.ps1`
  do the same without MACE (Linux, macOS, MobaXterm, Windows PowerShell), read
  the same profiles, and `--export-script bash|powershell` fills one in. The
  server's own hint now names the exact node and prints the connect line; in
  a VS Code terminal it says to Ctrl+click the link.
- **File › Open Folder…** switches the served folder within your home and
  scratch folders (`--root`, `MACE_GUI_ROOTS`; `--root none` turns it off),
  refusing hidden and credential folders; other tabs follow.
- **File › Sign Out and Stop Server…** ends the session and stops the server
  cleanly (Stop When Tasks Finish waits for running tasks, Cancel Stop undoes
  it); **one-time sign-in links**, session expiry after a time without input,
  an optional idle stop, and `mace gui login-url` (for a server started by
  `mace gui connect`, run it where `mace gui connect` runs: the link uses the
  forwarded port; on the node it says so).
- **Help › Copy Diagnostics and Report a Problem…** (and `mace gui
  diagnostics`): a bug report with versions, OS, browser, packages, settings,
  recent errors and log lines, with paths, file and folder names, tokens, user
  and host names removed.
- **CPU-time limits per process** (`ulimit -t`, common on dev nodes) are
  detected: MACE GUI restarts its ML worker and pool workers before they reach
  the limit, a worker the kernel kills anyway fails only its own task, and the
  server warns and stops itself cleanly near the limit.
- **GFN-FF finds an xtb module by itself** (`mace/utils/envmodules.py`): the
  newest suitable version of the machine's Lmod or Tcl modules, loaded for
  xtb's process only and remembered for 7 days; provenance names the module.
- **The GUI opens everything the CLI reads or writes.** Files are recognized
  by content: CRYSTAL and properties outputs (calculation type from the CLI's
  own classifier: all 161 FREQ and 118 OPT outputs of the test corpus right),
  SLURM job logs (opened as text linked to their calculation, never parsed for
  geometry), molecule outputs (no PRIMITIVE CELL block: 132 FREQ, 66 OPT and 67
  SP runs that did not open before), cube files including CRYSTAL's raw
  `*_CUBE.DAT` and `_WAN_CUBE.DAT` names, fort.25, BOLTZTRA data, restart data,
  wave functions, scripts and molecular formats. New documents: a properties
  output's summary with its data files; a **Volume** viewer for cubes
  (isosurfaces, ± lobes, slices, integrated density, a density colored by its
  potential with the CLI's range and colors, differences of grids and Save
  Result As); transport plots; 2D charge-density and potential maps from
  fort.25 as contour plots; Compare Convergence of several runs. FREQ
  outputs open as Normal modes, with arrows that move with the atoms while the
  mode plays; IR/Raman peaks are clickable and open their mode, and a FREQ
  output with no SPEC.DAT gets a spectrum broadened from its modes with
  CRYSTAL's DAMPFAC width (FWHM 8 cm⁻¹). Notices close on their own (info) or
  stay with a close button (errors), count repeats, and land in a notification
  history. Export Image takes a printed size at a DPI (written into the PNG)
  and SVG shading. Setup writes SLAB, POLYMER and MOLECULE decks, orienting the
  cell first so the CLI's coordinate writer gets the orientation it handles.
  The CLI is unchanged. See docs/gui.md, "What each file opens as", "Volumes"
  and "Molecules, slabs and polymers".
- **Pre-optimized starting geometries: `mace preopt INPUT.cif|INPUT.d12
  --method springs|gfnff|mlip:<id>|auto`**, written as `<stem>_preopt.cif` in
  the input's own space group and setting, with a provenance
  `<stem>_preopt.json`. Springs are the default; GFN-FF runs through an
  external xtb 6.7+; the ML potentials (MACE-MPA-0 from ACEsuit's mace-torch,
  which is unrelated to this MACE, and CHGNet 0.3.0) run out of process in
  their own environment. Every run makes the spring result first and keeps it
  when GFN-FF/ML is refused, fails, or is rejected (energy rose, volume moved
  over 10 %, a close contact, a space-group change, not converged). A raised
  space group is never accepted, since CRYSTAL OPT would keep it;
  `--symmetrize TOL` is the explicit way to raise it. The result is a starting
  geometry only: the method of the calculation is untouched and CRYSTAL still
  optimizes. See docs/preopt.md.
- **`mace mlip list|setup|fetch`:** the ML models, their licenses and status;
  instructions for an ML environment (MACE creates none) and recording one;
  the weights download, shown and confirmed first, sha256-verified.
- `mace convert` and the GUI's Setup warn (and change nothing else) when a
  geometry pre-optimized with GFN-FF or an ML potential is used for anything
  but an OPT.
- **A MACE logo** (docs/logo/, with its usage guide), drawn from the
  terminal banner: the README header and the GUI's favicon, menu bar, About
  and welcome screen use it.
- **SLURM job arrays (opt-in): `mace submit --array [--max-running N]`.**
  Many decks at once go to SLURM as job arrays instead of one job each. Every
  deck keeps the job script the default path writes; decks whose scripts ask
  for exactly the same resources share one array (one task per deck), and a
  deck with a request of its own goes in as a normal job (nothing is rounded
  up or down; a `#SBATCH` line that also names the job keeps its resources in
  the array's request). `--max-running N` limits the running tasks of each array
  (`--array=0-M%N`). Works with `--track`, `--progress`, `--walltime`, `-i`,
  `--nosubmit` (writes the array and prints its sbatch line) and mixed
  .d12/.d3 folders. `mace workflow --quick-start/--execute/--interactive
  --array [--max-running N]` and `mace submit --progress ... --array` write
  `"job_arrays": {"first_step": true, "max_running": N}` into the plan: the
  first wave goes in as arrays and each task's own callback chains that
  material's later steps as normal jobs. Tasks are `<array id>_<task>`;
  status, completion, timeout restart, FINALRUN-loop recovery, early-failure
  cancel, zombie removal and the database all address the single task, and a
  bare array id is never canceled. The arrays of a submission are submitted
  held and released after their rows are written; a refused array removes the
  rest; a submission larger than the free MaxSubmitJobs slots is refused up
  front. A task canceled while pending (SLURM keeps no record of it) is marked
  canceled after about 10 minutes of finding nothing of it on two status
  checks. Each array's manifest, wrapper and per-task logs are in
  `.mace_arrays/` (folder names with spaces work). Callbacks of array tasks
  wait up to 15 minutes (inside their walltime) for the queue lock, and record
  their own result if they cannot get it. See DOCUMENTATION.md, "Job arrays".
- **`mace run`: run decks on the node you are on, without SLURM.** For a
  MATDUMP, BAND/DOSS/POTC, a small SP or a quick rerun on an HPCC dev node.
  Each deck runs through the job script `mace submit` would submit (written to
  `.mace_local/<job>.sh`, run under `bash --login`), so staging, the per-job
  scratch directory and the copy-back names are a SLURM job's; the log is
  `<job>-local.o` and no completion callback runs. `.d12` decks run first, then
  each `.d3` after the SCF opt2d3 named it after, with that SCF's new `.f9`
  copied in. `-n` ranks (default 4, at most 16), `--timeout`
  (default 30m, below the dev nodes' 2 h CPU-time limit), `--dry-run` and
  `--track`. Ctrl-C, a hang-up or the timeout stops every CRYSTAL process of
  the run. Refused on the login gateway, inside a SLURM job, without a loadable
  CRYSTAL module, and for a job name another run already has - on any node,
  through a claim file in `$SCRATCH/.mace_run/` (on HPCC the same name started
  on dev-amd20 and dev-intel18 shared one scratch directory, and one run
  died). A stopped deck's `.out` ends with a `mace run: STOPPED` line, so
  `mace completion` lists it as timed out or stopped, not as a crash. Measured on
  dev-amd20: the diamond SP energy matched the SLURM run's to every printed
  digit. See DOCUMENTATION.md, "Run on a dev node".
- **Opt-in TZVP-REV2 2022 update: `full.basis.triplezeta.rev2022`.** The
  bundled triplezeta set carries pob-TZVP 2018 for Rb-I and ERD small-core
  2017 for La-Lu. The new family replaces those with the newer sets from
  crystal.unito.it: X_POB_TZVP_rev2 for Rb-Mo and Ru-I (Laun & Bredow, JCC
  2022, DOI 10.1002/jcc.26839) and X_pob_TZVP_rev2 for La-Lu (Seidler, Laun &
  Bredow, JCC 2023, DOI 10.1002/jcc.27097). Tc keeps the ERD 2017 all-electron
  record, because the site has no Tc_POB_TZVP_rev2. Every other record is
  byte-identical to `full.basis.triplezeta`.
  - **Core treatment.** Rb-I keep the ECPs of pob-TZVP 2018 (the same ZNUC).
    Ce-Lu keep a 28-electron core with 4f in the valence, as ERD has.
    **La changes:** its rev2 ECP has ZNUC 11, a 46-electron core, where ERD
    has ZNUC 29. With this family each La atom carries 18 fewer electrons,
    which shows in the Mulliken `NO. OF ELECTRONS`.
  - **How to choose it.** Pick external basis menu option `4: TZVP-REV2 (2022
    update: Rb-I, La-Lu)`, or put `"external_basis_family": "tz-rev2022"` in a
    configuration file. In code, use `mace_config.DEFAULT_TZ_REV2022_PATH` or
    `get_basis_path("tz-rev2022")`. Options 1-3 are unchanged and 2 is still
    the default. Decks are named `..._full.basis.triplezeta.rev2022.d12`, and
    the workflow executor keeps such a child on this family even when
    `MACE_TZ_BASIS_PATH` or `MACE_BASIS_DIR` is set.
  - **Provenance.** Each record keeps its URL, retrieval date, page checksum,
    DOI as printed and core. `basis_sets/fetch_unito.py` and `build_library.py
    --fetch` rebuild the family.
  - **Status.** Untested in CRYSTAL. The Rb-I records should reproduce
    CRYSTAL23's internal `BASISSET POB-TZVP-REV2` energies exactly.
- **POB-TZVP-REV2 SOC basis records (data only).** X_pob_TZVP_rev2_SOC for
  Rb-Mo, Ru-I, Cs, Ba and Hf-Bi, from crystal.unito.it. Each is the rev2
  valence shells (checked numerically) on CRYSTAL's internal STUTSC library
  (manual p.84), for two-component SOC runs (pp.169-175).
  - Read them with `basis_library.soc_basis_record(z)`. They are never offered
    as an ordinary external basis: there is no menu entry, path or
    configuration key for them.
  - Untested in CRYSTAL.
- **Charged systems and ion clusters** (opt-in; see "Charged systems and ion
  clusters" in `Crystal_d12/README.md`). `cif2d12` (`--ion-charges auto`,
  `--net-charge`, `--site-states`, `--chemod`, `--fragment-bond-scale`,
  `--spinlock`, `--spinlock-cycles`, or the same keys in an options/config
  file), `opt2d12` (the same flags, recomputed from the optimized geometry)
  and a new `mace ions` command for existing decks write `CHEMOD` (starting
  shell charges), `DOPING` (the net charge), `CHARGED` and, for an odd
  electron count, `SPIN` + `SPINLOCK`. Ions are found by connectivity
  (covalent radii), so atom order does not matter: Li+, Na+, K+, Mg2+, Ca2+,
  Zn2+, FSI-, TFSI-, OTf-, ClO4-, NO3-, PF6-, BF4-; anything else is neutral,
  except that a molecule that cannot be closed-shell neutral by its bonds
  (CO3, SO4, OH, a carboxylate: an anion outside the table) stops auto mode
  until `net_charge` is given, so Li2CO3 is not written with `DOPING -2`.
  On the 65 Li/FSI/solvent clusters of the test corpus it finds the same
  ions in any atom order (the group's script assumed a fixed order and left
  two clusters without records). The records follow what was measured on
  CRYSTAL23: with an internal BASISSET the count comes from `DOPING` and
  `CHEMOD` only shapes the guess; with an explicit basis a non-neutral
  `CHEMOD` + `CHARGED` sets the count and `DOPING` is not added (both
  together counted the charge twice). An odd count in a molecule is never
  written closed shell (a periodic cell may be, with a warning: a half-filled
  band); RHF/HF-3c and `GHOSTS` + `DOPING` with an odd count are refused; an
  odd-electron open-shell deck with an HSE-type screened hybrid gets a
  warning in every dimensionality, also in opt2d12 children (measured:
  HSEsol-3c, HSE06 and HSEsol crash at SCF cycle 1-3 on any odd-electron
  open-shell system, 0D and 3D, while PBE0/PBEsol0 converge; the functional
  is never changed); and
  `CHEMOD` needs a deck without symmetry (CRYSTAL stops: "SHELL CHARGES NOT
  COMPATIBLE WITH SYMMETRY"). Shell charges for SOLDEF2MSVP, POB-TZVP-REV2
  and POB-DZVP-REV2 (H B C N O F Na Mg P S Cl K Ca Zn Li) were measured on
  HPCC; other internal libraries need explicit `site_states`. New example
  config `ion_cluster_molecule.json` (P1, every atom written, so a cluster
  CIF that declares a space group is not reduced).
- **Per-fragment Mulliken charges of charged runs.** Property extraction adds
  `electron_count` and `fragment_charges` (each molecule/ion's charge, spin,
  formal charge and deviation, an ion more than 0.5 e from its formal charge
  flagged) when a run was charged (Mulliken count different from the sum of
  valence charges, a renormalized guess other than after a GUESSP restart,
  or "UNIT CELL NOT NEUTRAL"), and the
  extractor prints them as a Fragments section;
  `mace analyze --extract-properties <dir> --fragments` (or
  `MACE_FRAGMENT_CHARGES=1`) computes them for every run. Neutral runs get
  no new properties otherwise.
- **`mace defects`: point-defect deck sets and their analysis (new command;
  nothing existing changes).** `mace defects build HOST.d12` writes a pristine
  supercell deck (SUPERCEL/SUPERCON, 3D or 2D including CRYSTAL + SLABCUT
  hosts) and one single-point deck per charge state for a vacancy (ATOMREMO),
  ghost vacancy (GHOSTS), substitution (ATOMSUBS) or interstitial (ATOMINSE,
  untested on HPCC), by editing the host deck's text: method, basis, grids and
  tolerances stay byte for byte. Charge is `DOPING = -q`; odd electron counts
  (or `--spin-state q=NSPIN`, parity-checked, p.128) get SPIN/UHF and
  `SPINLOCK NSPIN 30`, and every choice is printed. The records sit where the
  decks run on HPCC had them, and those decks (MgO V_O q=0/+1/+2, ghost q=0/+2,
  the explicit-basis odd-ghost recipe, fractional-DOPING decks, hBN V_N 3x3 to
  6x6) are reproduced byte for byte. Refused with the reason: OPT/FREQ hosts
  (build from an SP), KEEPSYMM in block 1, charged hosts, host spin locks
  (unless `--host-spin keep`), odd ghost states with an internal basis
  (measured to abort or collapse), GHOSTS on ECP atoms (p.82), and more (see
  `docs/defects/README.md`). Supercell labels come from CRYSTAL's output: a set
  without one writes the pristine deck and finishes with `--resume DIR` as soon
  as the running pristine output shows its geometry. `mace defects potc DIR`
  stages POTC runs (with copies of the `.f9`) into `DIR/potc`. `mace defects
  analyze DIR` checks the Mulliken electron count (DOPING applied) and the
  final spin, and reports formation energies and transition levels, all
  unrelaxed. 3D charged states get the offset measured on CRYSTAL23 (MgO, LiF,
  Si): `E(q) + q*(C(q)+C(0))/2` with `C = 2*E_sph/N` from the output's
  `EXT EL-SPHEROPOLE` term and N = header electron count + DOPING, and, with
  `--eps VALUE` or `--eps-from` a CPHF (eps_inf) or FREQCALC INTCPHF output
  (`--eps-static`: eps_0), the Lany-Zunger-scaled Makov-Payne image-charge
  term with the Ewald Madelung potential of the actual supercell. On the
  measured MgO V_O runs (53 to 249 atoms, eps_inf 2.865) this gives eps(+/0)
  2.85-2.88 eV and eps(2+/+) -0.09 to -0.19 eV. Without a dielectric constant
  charged states are labeled "uncorrected for image charges"; every report
  names the corrections it applied. `--calibrate` adds restricted
  fractional-DOPING decks as an optional cross-check (`analyze --offset
  calibration` uses them instead). 2D sets use an A + B/L fit over the sizes.
  Energies are SCF + D3 + gCP for -D3 and 3c methods (CRYSTAL's
  `TOTAL ENERGY + DISP + GCP` line, not the bare `SCF ENDED` value): those
  terms do not cancel between a defect cell and the pristine cell, so chemical
  potentials must include them too, and a job whose terms differ from the
  pristine's is excluded. Potential alignment is reported, and applied only
  with `--apply-alignment` (measured to worsen the agreement between charged
  cells of different size). A worked MgO V_O deck set is in
  `docs/defects/MgO_VO` (not run).
- **`mace phonopy generate` / `mace phonopy collect`: finite-displacement phonons with
  phonopy** (optional dependency, `phonopy<5`). `generate` reads a finished 3D OPT or
  SP parent and writes one EXTERNAL deck per displaced supercell, named after the
  parent and the supercell, with its geometry in `<job>.ext`. The deck has the layout
  phonopy's own CRYSTAL writer gives it, measured to give forces on CRYSTAL23: the
  parent's basis and SCF block (phonopy's TEMPLATE, also written to `TEMPLATE` for an
  explicit basis; never with an END, which stops CRYSTAL before GRADCAL with no
  forces), then TOLDEE (at least 10), GRADCAL and END. The basis and SCF block are
  kept byte for byte apart from the printed edits (GUESSP dropped, buffers for large
  cells, SHRINK divided by the supercell size - rounded up where it does not divide -
  unless `--keep-shrink` or a non-diagonal `--dim`). `--cry-sym` (opt-in) writes each
  displaced cell's symmetry into its `.ext` as phonopy's CRY_SYM does - the same
  frequencies for diamond at about a third of the core hours, measured on CRYSTAL23 -
  with phonopy's caveat that some space groups (e.g. R-3m) do not work. Parents it
  cannot reproduce in a P1 supercell (non-3D, charged, spin-locked, magnetic without a
  lock, ghosts, unknown SCF keywords, unconverged) are refused with the reason.
  `collect` checks every output - including that CRYSTAL ran the displaced geometry it
  was given - names any output whose forces phonopy cannot read, and stops unless
  FORCE_SETS was really written (phonopy's `phonopy-init -f` exits 0 when it fails)
  before writing band, mesh, DOS and thermal properties, with optional non-analytical
  correction from a Gamma FREQCALC with INTENS + INTCPHF. Submit the decks with a
  plain `mace submit DIR`, never `--track`, with the walltime `generate` prints: it
  grows with k-points x atoms^3 and carries a 1.5x margin (3 h for a 2x2x2 hybrid-DFT
  diamond, 30 h for 3x3x3, which measured on CRYSTAL23 did not finish its SCF in 2 h).
- The CRYSTAL job script copies `<job>.ext` to `fort.34` when that file exists next to
  the deck (EXTERNAL geometry). No existing deck has one, so their jobs are unchanged.
- `mace preflight` checks an EXTERNAL deck when its geometry sits next to it as
  `<stem>.ext`, staging it as fort.34 the way the job script does.

- **Run Analysis in the GUI**: a run card (how the run stopped, in the words
  of `mace check` and the CRYSTAL23 manual, and the run's details), and a plot
  document with the energy per CRYSTAL optimization point by FINALRUN segment,
  the four convergence criteria against their thresholds (with where each
  threshold came from), SCF cycles, the cell along the run and distances or
  angles tracked across frames; Compare Runs; Problems rows from the output
  (warning stops, SCF not converged, CRYSTAL warnings, non-default tolerances,
  imaginary modes beyond numerical noise). See docs/gui.md › Analyzing a run.
- **`mace analyze --convergence FILE.out [--json]`**: the same per-point table
  in a terminal (SCF cycles, printed DE, the four criteria against their
  thresholds, FINALRUN segments, how the run stopped, threshold sources).
- **Plot options on every GUI plot**: Log Y/X, units (eV, meV, Ha, kJ/mol,
  kcal/mol; cm⁻¹, meV, THz), relative to, per cell/atom/formula unit, ranges,
  line width, Export (PNG, SVG, CSV, Copy as TSV), remembered per plot.
- **Create or Update Database from a Folder** (GUI): the
  `mace analyze --extract-properties … --register` command it shows, with a
  preview of new, changed, unchanged, untracked, unreadable and incomplete
  outputs, a backup first, progress and Cancel. Raman intensities of FREQ
  runs (`raman_intensities`) and the space-group number CRYSTAL23 prints as a
  symbol (`space_group_number`) are now extracted too.
- **Live runs and live databases in the GUI**: outputs a job is still writing
  update every 15 s (never shown as broken), the text view follows the tail,
  and the Database view reloads when jobs record results, keeping its sort
  and selection; a busy database says so and retries.
- **File History** (Explorer › Show File History…, File: Show History): Open,
  Compare with Current and Restore for every file the GUI replaced, and Undo
  on the save toast.
- **The Atoms, Coordination and Measurements panels of a run** show the frame
  on screen, read-only (Edit This Step to change it).

### Changed

- **The documentation is reorganized.** The README is a short front page:
  what MACE is, how to install it, a GUI quick start, then the command line,
  a feature tour and links. docs/cli.md now has a reference section for every
  command with all of its options (a test keeps it in step with each
  command's `--help`); docs/gui.md is the GUI guide; docs/citations.md lists
  what to cite (CRYSTAL23, SeeK-path, spglib, ASE, phonopy, the defect
  corrections, 3Dmol.js and the pre-optimization methods); INSTALLATION.md
  lists every requirement. The repository tree and workflow isolation moved
  from the README to DOCUMENTATION.md. A test checks every relative link and
  anchor in the Markdown files.
- **seekpath is a required dependency** (`requirements.txt`, `seekpath>=2.1,<3`;
  an older seekpath counts as missing, and `setup_mace.py` checks for it).
  A SeeK-path band path of a 3D cell is
  computed from the structure and written in CRYSTAL's reciprocal basis; the
  static HPKOT tables hold SeeK-path's numbers in SeeK-path's standard cell,
  which for many lattices is not CRYSTAL's (axis order, reduction, centering
  vectors), so a deck written from them can visit or name other points.
  - Without seekpath, a 3D SeeK-path BAND deck (opt2d3, workflow `seekpath_full`
    configurations) or SeeK-path phonon dispersion (opt2d12) stops with the
    install command, the MSU HPCC module-Python instructions and the opt-in;
    opt2d3 and opt2d12 exit non-zero when no deck was written.
  - `--static-kpath` (opt2d3, opt2d12) or `MACE_STATIC_KPATH=1` writes the
    static path as before, with a note.
  - With seekpath installed, a library error for a 3D cell (or an output it
    cannot read the cell from) stops that deck the same way, with the error
    and the opt-in, and opt2d12 checks before writing; it used to print "seekpath library failed (...), falling back to
    static data" and write the static path. With the opt-in it still falls
    back (with that warning), and a polymer's path, which does not need
    seekpath, still falls back.
  - Slabs and polymers do not need seekpath and print no install message (a
    polymer printed "Install 'seekpath'"); label, k-point vector, literature
    and custom paths do not use it. Importing the k-path modules stays silent.
  - CI installs seekpath in every job (`requirements-test.txt`).

- **The bundled external basis sets are one file per family.**
  - `Crystal_d12/basis_sets/full.basis.triplezeta/`, `full.basis.doublezeta/`
    and `stuttgart/` (about 300 per-element files) are now
    `full.basis.triplezeta.lib`, `full.basis.doublezeta.lib` and
    `stuttgart.lib`, read by `Crystal_d12/basis_library.py`.
  - The old directory paths still work everywhere MACE takes a basis path,
    including `mace_config.DEFAULT_TZ_PATH`, configuration files that name
    them and the custom-path prompt (menu option 3). Any other directory is
    still read from disk.
  - Every record is returned byte for byte: tests pin all records and the
    decks written from them. No deck changes, apart from the Pu, La-Lu and
    Ac-Es fixes below; five stuttgart records CRYSTAL cannot read are refused.
  - **Custom tooling** that opened `basis_sets/<family>/<n>` directly should
    call `basis_library.read_record(path, n)` instead, or
    `basis_library.load_family(name)`.
  - Each record now states where it came from (`basis_sets/PROVENANCE.md`).
    The records were matched exactly against crystal.unito.it and checked
    against basissetexchange.org: H-Br, Cs, Ba and Hf-Bi are pob-TZVP-rev2,
    Rb-I are pob-TZVP 2018, and DZ H-Br and Rb-I are pob-DZVP-rev2 and
    pob-DZVP 2018. Unmatched records say "unverified".
  - `python mace_config.py` (`ensure_paths_exist`) reports the three `.lib`
    files under the same names. This is diagnostic output only.
- **Workflow BAND steps use NSUB 1000, as opt2d3** (engine default, planner
  Basic and Advanced, quick-start), instead of 10000. DOSS keeps its 10000
  energy points. Deck change: NSUB only.
- **Default workflow BAND steps write opt2d3's Enter-through deck**: the
  automatic path as high-symmetry labels (format 1, ISS 0). The engine
  default and the planner's Basic BAND wrote k-point vectors, and
  quick-start / `--progress` the SeeK-path vectors, so the same parent got a
  different path, title and NLINE record by route; the decks now match opt2d3
  byte for byte (checked on 13 lattices and a slab). Deck change: path
  records, NLINE, ISS and title. Vectors, literature and SeeK-path paths are
  formats 2-4 in the planner's Advanced BAND menu.
- opt2d3's k-point-vector and literature BAND titles name the path
  ("V-Y-GAMMA-X" instead of "in-plane path" on slabs) and the "Literature"
  source, as the same configuration replayed from a file (a workflow step or
  a saved Expert config) titles it. Only the title line changes; a
  literature request with no literature path is titled "default".
- opt2d12 runs answered through a pipe list every setting that differs from
  the parent deck (functional, basis, grid, TOLINTEG, TOLDEE, LEVSHIFT,
  HISTDIIS, MAXTRADIUS), so an answer file written for an older MACE cannot
  change a deck silently. The deck is unchanged.
- The phonopy walltime suggestion carries a 1.5x margin (diamond 2x2x2
  hybrid 2 h -> 3 h, 3x3x3 20 h -> 30 h); lighter methods and `--cry-sym`
  are not shortened, since a job killed at its walltime leaves no forces.
- The workflow engine no longer prints raw DEBUG dumps (FREQ settings,
  workflow id/settings/sequence, sequence position).
- The defects README states the POTC sign convention the alignment assumes
  (unverified), and the phonopy docs explain the phonopy/CRYSTAL mass offset
  (about 1 cm-1 at 540 cm-1 for Si) and why an internal-basis set has no
  TEMPLATE (phonopy's writer puts ENDGEOM before it).
- INSTALLATION.md says to install the module-Python dependencies on a
  development or compute node: an MSU HPCC gateway node refuses
  `module load Python/3.11.3-GCCcore-12.3.0` (the module the job scripts
  load), while the same load works on dev-amd20. Job scripts are unchanged;
  they run on compute nodes.
- The stuttgart library's description says which records are unverified (117
  of 119) and which CRYSTAL cannot read.
- Tests: the bundled MgO defect outputs
  `tests/data/supercell_parents/rg_q0.out.gz` and `rs_qm1.out.gz` are now
  trimmed excerpts (about 500 lines each) with no MPI start-up lines,
  process counts, run dates, timings or per-node CPU times;
  `tests/test_supercell_parent_excerpts.py` keeps such lines out. So is
  the rod group 4 polymer parent `tests/data/low_dim_groups/polyyne_rg4.out`
  (now without its SCF cycles and timings, about 410 lines).

- **MACE uses American spelling throughout**: messages, prompts, help text,
  comments, docstrings and documentation now read "optimization", "canceled",
  "centered", "neighbors", "behavior" and so on, and a test keeps British
  spellings from coming back. Unchanged on purpose: the stored job status
  `cancelled`, the saved-config keys `unrecognised_functional` and
  `unrecognised_functional_source`, CRYSTAL and SLURM text (`CANCELLED`), and
  the UniTO basis-set URLs named after British element names. Decks are
  byte-identical.

### Fixed

- **`mace` works through a symbolic link to `mace_cli`** (for example
  `~/bin/mace`) without `MACE_HOME`: `convert`, `opt2d12`, `opt2d3`, `opt2cif`
  and `ions` looked for their tools next to the link and stopped with "No
  module named ...". Since 1.0.0.
- **GUI: the activity bar and the theme, settings and search buttons work.**
  They had no click handler. Every visible command button is now tested with
  real clicks.
- **GUI: no atom is cut off by the clipping planes** after a range change, a
  pan or a zoom (the planes followed only the first fit: 637 of 900 atoms of
  a 3×3×3 AgTe cell were clipped), and depth cueing is subtle (15%) instead of
  3Dmol's default fog over most of the cell.
- **`mace preopt --device cuda:N` stays within the job's GPUs.** Inside a job
  with `CUDA_VISIBLE_DEVICES` set (SLURM, a site wrapper), N now counts within
  that list (`3,5` and `cuda:1` give GPU 5); an N outside it, or a cuda device
  in a job without GPUs, is refused before the worker starts.
- **The GUI's energy card put an optimization's first point on the wrong
  energy surface.** CRYSTAL prints point 1's `TOTAL ENERGY(DFT)` without its
  D3/gCP correction and compares later points with the corrected value, so
  most real runs (117 of 118 test optimizations) showed a spurious drop of
  hundreds of meV at step 1. The card now uses CRYSTAL's corrected value
  (and the OPT END energy for the final frame).
- **The GUI's Database view could read a database mid-write.** It fell back to
  an "immutable" read on any open error, including a busy database and the
  hot journal a killed writer leaves; it now decides by SQLite's error code,
  answers "busy" and retries, and reads a snapshot only of a database in a
  folder it cannot write.
- **Extracted properties of tracked calculations were invisible in the
  database** (filed under file-name material ids no materials row had).
- **MACE never cancels a whole job array when its database cannot be read.**
  The check that refuses `scancel <array id>` gave up silently on a database
  error, so a bare array id that no nearby `.mace_arrays/*/submitted.json`
  named went through and canceled every task. Now SLURM is asked instead
  (`squeue -r`, then `scontrol show job`): an array id is refused, a plain
  job SLURM lists cancels as before, and when SLURM cannot answer either the
  cancel is refused with the reason ("could not check whether <id> is a job
  array ..."). Affects the queue manager's cancels and the GUI's Jobs panel.
- **Security: MACE GUI could read secret files outside its credential
  list.** Serving your home folder, the read, preview and download routes
  blocked only `.ssh`, `.gnupg`, `.aws`, `.kube`, `.docker`, `.config/gh`,
  `.netrc`, `.git-credentials` and `.pgpass`, so `api/text?path=.pypirc` (or
  `.npmrc`, `.env`, `.bash_history`, an `id_rsa` outside `.ssh`, ...) returned
  the file. Those are now hidden for safety too, wherever they are and also
  through a symbolic link (the list is in the Security section of
  docs/gui.md).

- **squeue lines of job arrays are read whole.** squeue folds an array's
  pending tasks into one token, and once the pending set has a gap the
  brackets hold commas (`A_[2,4-5%2]`, measured). Every queue check split the
  line on each comma, so the token was cut in two and the state read from the
  wrong field; such lines (from any array in your queue) were miscounted.
  They are split outside the brackets now, and a task list SLURM printed cut
  short changes nothing.
- **A task number is no longer read as a job id.** The `.o` lookup of
  `mace completion --remove-zombie-jobs` took `X-<array>_<task>.o` for job
  `<task>` when the task number had six digits; it now reads `<array>_<task>`,
  and the scan of finished outputs records it too.
- **A job that wrote no output never takes another deck's.** When a tracked
  job left the queue without its own `.out`, the check took the first `.out`
  of its directory, so in a folder of many decks (`mace submit DIR --array`,
  whose recovery resubmissions are normal jobs, or `--track DIR`) a restart
  canceled while pending was recorded as completed with another material's
  output and properties (found in the HPCC array test). An output that has a
  deck of its own next to it (`X.out` beside `X.d12`/`X.d3`) is now never
  used for another calculation; the job is recorded as ended without output.
- **The planner's `--array=1-10` directive preset is gone** (see Upgrading).
  It turned every job of the step into ten copies of one material sharing a
  scratch directory, on the `mace submit --progress interactive` route.
- **A job script run outside SLURM no longer runs in `/`.** `bash job.sh`
  left `DIR` (`$SLURM_SUBMIT_DIR`) empty: the script copied from and to `/`,
  never ran CRYSTAL, exited 0 and ran the completion callback from the home
  directory. It now stops with exit 1 and says to use sbatch or `mace run`.
- **`mace convert --help` works without ase.** It stopped with
  ModuleNotFoundError before printing anything; a conversion without ase now
  says to install it.
- **A workflow FREQ step keeps a FREQ parent's SCF tolerances.** The
  planner's Basic, Advanced and Expert-fallback FREQ steps and the
  quick-start FREQ step wrote Very tight (9 9 9 11 38 / 11) into the plan as
  if the user had chosen it, replacing a FREQ parent's tolerances. They now
  leave the choice to opt2d12: Very tight after an OPT parent (as before, and
  as the manual opt2d12 route gives), the parent's own after a FREQ parent.
  Tolerances set in the plan still win.
- **A FREQ child of a FREQ parent keeps the parent's IR/Raman
  intensities** (INTENS, INTCPHF, INTRAMAN, IRSPEC/RAMSPEC) on every route,
  when it runs in the parent's Gamma-point mode. opt2d12's interactive FREQ
  template menu now defaults to "0: Keep the parent's FREQCALC settings"
  when the parent has intensity or spectrum settings (it defaulted to "Basic
  frequencies only", writing NOINTENS); a parent with only NUMDERIV gets the
  usual questions, its NUMDERIV the default answer. The workflow's Basic and
  quick-start FREQ steps no longer write "no intensities" into the plan, the
  Advanced level's blank answer keeps the parent's, and a config that names
  no intensity setting keeps the parent's when its FREQ mode is the
  parent's. A config that names NUMDERIV alone therefore no longer drops the
  parent's intensities, and a phonon dispersion config gets none (NOINTENS),
  as opt2d12's phonon templates write.
- **The workflow's phonon band structure step writes opt2d12's deck.** The
  planner's Advanced FREQ phonon route (full dispersion, band structure) on
  default answers wrote INTERPHESS 2 2 2 and no BANDS path. It now writes
  what opt2d12's "6: Phonon band structure" template writes on default
  answers: SCELPHONO 2x2x2, NOINTENS, DISPERSION and the automatic
  high-symmetry BANDS label path, with no INTERPHESS (checked byte for byte
  on the corpus AgCl FREQ parent). Deck change: BANDS records instead of
  INTERPHESS, and the title names the path.
- **The workflow's phonon DOS steps write opt2d12's deck.** The planner's
  Advanced FREQ phonon route with output type "2: Phonon density of
  states" or "3: Both band structure and DOS" stored only a label nothing
  reads, so the deck was `FREQCALC / NOINTENS / DISPERSION / ENDFREQ`: no
  PDOS, and no BANDS for "both". On default answers DOS now writes what
  opt2d12's "7: Phonon density of states" writes (PDOS `2000.0 200`,
  atom-projected), and "both" what opt2d12's custom dispersion route
  without INTERPHESS writes: template 6's BANDS path, then that PDOS. An
  output type not on the menu, which says it defaults to the band
  structure, now writes its BANDS path too. Deck change: re-plan phonon
  DOS steps.
- The planner's Expert FREQ fallback (used when the interactive opt2d12 run
  fails) wrote IR + Raman intensities through CPHF while saying "using
  advanced settings". It now writes what opt2d12's interactive FREQ route
  gives on default answers (Gamma-point frequencies, no intensities, or a
  FREQ parent's own) and says so.
- **The default BAND deck for a triclinic parent or any slab** (opt2d3
  Enter-through, and so the workflow default) is now integer k-point
  coordinates, with ISS taken from the parent's shrink. This is the deck
  opt2d3's k-point vector format writes. CRYSTAL23 rejects label input
  there: triclinic stopped with "K-POINTS NOT CODIFIED" and slabs with
  "FORMAT ERROR IN INPUT DECK". A label path that MACE cannot express for
  the lattice now writes no deck, where it used to write placeholder
  records.
- **The phonon band structure deck for a triclinic (P-1/P1) parent**
  (opt2d12's "6: Phonon band structure" on default answers, and so the
  planner's phonon band structure step) writes its BANDS path as integer
  coordinates (`BANDS / 16 100 / 9 / 8 8 0 0 8 0 ...`) instead of the
  ISS 0 label records (`V Y`, `Y GAMMA`, ...) CRYSTAL23 does not codify for
  a triclinic lattice. Deck change: regenerate triclinic phonon band decks.
- **A polymer's BAND deck runs along the chain** (opt2d3 Enter-through, and
  so the workflow default): Gamma -> X, `1 2 NSUB ... / 0 0 0  1 0 0`,
  whatever space group the polymer's symbol maps to. A polymer whose
  corresponding space group MACE maps (rod groups 28, 51) got an ISS = 0
  label deck, which CRYSTAL23 stops with "BANDE **** FORMAT ERROR IN INPUT
  DECK" (a polymer takes no label records). One whose symbol it does not map
  (rod groups 1 and 4) got the triclinic 3D path with I2/I3 != 0, against
  manual p.310 note 3: CRYSTAL ran it as five copies of Gamma-X under
  meaningless labels. The k-point vector, literature and SeeK-path formats
  keep their segments along the chain (I2 = I3 = J2 = J3 = 0), or fall back
  to Gamma -> X. Measured on CRYSTAL23 for rod groups 1, 4 and 28. Deck
  change: regenerate polymer BAND decks.
- **opt2d12 writes the phonon band structure (template 6) for a slab.** It
  asked every question and then refused, saying a slab has no space group
  to take the path from. A slab now gets SCELPHONO as a 2x2 matrix (`2 0 /
  0 2`, manual p.73; the 3x3 matrix aborts) and BANDS records of 4 integers
  (I1 I2 J1 J2), the in-plane part of the corresponding space group's path
  as opt2d3 cuts it. CRYSTAL23 takes no BANDS labels for a slab (INPFREQ
  FORMAT ERROR) and reads only 4 integers of each record, so 6-integer
  records went to other k points. Measured on an hBN monolayer: `12 100 / 2
  / 6 0 0 0 / 0 0 4 4` ran as M-Gamma-K and the P1 slab path as
  V-Y-Gamma-X. WANG (3D only, manual p.234), INTERPHESS, PDOS and INS are
  still refused for a slab (not run on CRYSTAL23), as is a slab from
  cif2d12 (no output to read the corresponding space group from) and a
  polymer dispersion (not run). opt2d12 now refuses a dispersion as soon as
  the FREQ template is chosen, before the method, basis and SCF questions.
- opt2d12's "6: Phonon band structure" FREQ template no longer carries
  one run's answers into the next run in the same process: the
  template's BANDS settings were shared, so after a default label-path run
  a later run lost the k-point density question and wrote the earlier
  run's segment count. Each run now starts from the template itself.
- The advanced electronic analysis in property extraction reads only the
  output's own BAND.DAT / DOSS.DAT: `<stem>.BAND.DAT`, else the same
  parent's `_band` / `_doss` run. It took the first such file in the
  output's directory, which could belong to another calculation. Two equally
  good matches are now skipped and named in `advanced_analysis_note`, not
  guessed between.
- **opt2d12 `--config-file` reads legacy functional spellings** ("HSEsol-3C",
  "PBEh-3C", "M06-2X", "HF", ...) as workflow plans do: the CRYSTAL23 keyword
  is written and a 3c method gets its basis. This covers the planner's Expert
  configs under `workflow_configs/expert_*_configs`, which the engine hands to
  opt2d12. "M06-2X" reached the deck, which CRYSTAL23 rejects, and
  "HSEsol-3C" kept the parent's explicit basis as "BASISSET / EXTERNAL". An
  unknown name now writes nothing and says why.
- The workflow planner's free-text questions (the sequence builder's "Next
  calculation", the Advanced BAND custom path and "Use all bands?", and the
  SLURM directives) refuse answers that cannot be meant for them (an unknown
  calculation type, a path that is not k-point labels, a directive that is
  not an sbatch option, anything but yes/no) and stop a piped run after
  three in a row. These and the planner's shared yes/no, menu and number
  questions stop with a message, not an EOFError traceback, when piped
  answers run out.
- opt2d12's FREQ menus ask again on an answer that is not on the menu. The
  template menu took any other answer (e.g. "9") as "Custom settings", the
  FREQCALC/ANHARM, mode and supercell questions took it as their first or
  last entry, and the phonon band-path crystal-system and template menus fell
  back to their first entry without saying so.
- The workflow planner's number prompts (queue limits, BAND/DOSS points,
  DOSS projection type and energy range, TRANSPORT chemical potentials,
  temperatures and relaxation time, CHARGE+POTENTIAL grid, FREQ supercell,
  k-point count, CPHF settings, peak widths, dielectric constant and the
  temperature list) ask again on an answer that is not a number instead of
  ending `mace workflow` with a ValueError. Counts must now be at least 1
  (0 allowed for reserve slots, recovery attempts and DOSS projection type
  0-4); absolute chemical potentials have no default and are asked again
  when left blank.
- Band and phonon paths for a slab no longer print "Install 'seekpath'":
  the library is not used for a slab on purpose (it would standardize the
  vacuum axis; manual p.310 note 3), so the advice changed nothing. A slab
  now gets one line saying the static in-plane path is used; a 3D cell
  without the library still gets the install hint. The path is unchanged.
- opt2d12's log names the basis the deck is written with. From an
  external-basis parent it printed "Using external basis set from original
  D12 file" (and "Preserving external basis settings") before the config
  chose the method, so HF-3c children logged that and then "Basis set:
  MINIX", and DFT-3c children (PBEH3C, HSESOL3C, ...) logged it although the
  deck says BASISSET <3c basis>. The line is now printed once, only when the
  deck carries the parent's external basis. The legacy workflow executor
  likewise logs "Basis set: <3c basis> (required by <functional>; the
  parent's basis is not used)" for a 3c step. Log only: decks are unchanged.
- **opt2d12 no longer writes SP or OPT decks from a phonon-supercell
  (SCELPHONO) parent** (a dispersion or a Gamma-only SCELPHONO FREQ; the
  refusal says "phonon supercell (SCELPHONO)"). The parent's output geometry
  is the SCELPHONO supercell (manual sec. 4.21 p.73-74); measured on HPCC
  (MgO, SCELPHONO 2x2x2 + DISPERSION) the SP deck written from it was a
  space-group-225 cell with a = 5.95 A and 5 asymmetric atoms instead of
  MgO. Such children are now refused like FREQ children, with the advice to
  make them from the OPT or SP output the dispersion run started from. Deck
  change: these decks are no longer written.
- The job script no longer stages a job's own two-component `.f9` for a
  GUESSP inside a TWOCOMPON block. There GUESSP means GUESSPNOSO (manual
  p.173): the 2c-SCF starts from a one-component matrix of an unrestricted
  run without symmetry. GUESSP, GUESSPNOSO and 2NDVARIAT inside TWOCOMPON now
  use only an explicitly staged `$JOB.f20` (with a note of the p.173
  requirement); with none, a bare GUESSP is dropped from the scratch copy
  (cold start), and GUESSPNOSO/2NDVARIAT are left as written with a warning
  (CRYSTAL stops without fort.20, measured). A GUESSP at the top level of a
  deck with a TWOCOMPON block gets the same handling, as CRYSTAL reads it like
  GUESSPNOSO too (measured on HPCC, Bi2 slab, PBE0+SOC: the deck's own 2c
  `.f9` as fort.20 stops with "io error Read_real_1d", a 1c matrix computed
  with symmetry with "SYMMETRY INCOMPATIBILITY"); the OPTGEOM RESTART of such
  a deck no longer keeps its own 2c fort.20 either. A one-component
  `$JOB.f20` staged for a top-level GUESSP is used with a warning: that
  measured run read it, then diverged where the cold start converged in 12
  cycles. A GUESSP in a one-component deck is handled as before.
- `opt2d3 --config-file` finds the shipped example configs by bare name
  (`band_auto_everything.json`) from any directory, and a relative
  `.../Crystal_d3/example_configs/<name>` written against another working
  directory, and prints the file it loaded. A configuration that cannot be
  loaded now ends opt2d3 with exit status 1 instead of 0 with no deck.
- **SeeK-path band paths use the computed geometry: the optimized cell after
  an optimization, the supercell after SUPERCEL.** With the seekpath library,
  `opt2d3` (and the workflow BAND step, and `opt2d12` phonon-dispersion paths)
  read the cell from the first "PRIMITIVE CELL" lattice parameters and the
  atoms from the first atom list of the parent output. For an OPT parent that
  is the initial geometry, so the path's lattice-dependent points (and, when
  the cell changed enough, the lattice variant, tI1 or tI2 say) were those of
  the starting cell. After SUPERCEL the cell was the input primitive cell
  while the atoms were the supercell's (FREQ `..._supercel222.out`: 16 atoms
  in a 2.507 A cell instead of the 5.013 A supercell). Cell and atoms now come
  from one computed geometry block: the one after the last FINAL OPTIMIZED
  GEOMETRY, otherwise the first (GEOMETRY FOR WAVE FUNCTION, the cell fort.9
  holds). **K-paths are in the basis of the computed cell**: a BAND deck from
  a SUPERCEL output is in the supercell's reciprocal basis (folded bands).
  Outputs whose geometry input edits the cell (SUPERCEL, ATOMSUBS, ATOMREMO,
  ELASTIC, ...) change the same way. An atom count that does not match the
  one CRYSTAL prints, or an output without such a block, is read as before
  with a warning. An optimization that ended unconverged (OPT END - FAILED)
  prints no final geometry; its last printed point is used, the geometry its
  fort.9 holds (a killed run keeps the first).
- **Static band paths (without the seekpath library) use the computed cell
  too.** The static route took the conventional cell of the input echo: an
  optimization's starting cell chose the lattice variant (tI1 or tI2, oF1 or
  oF3, hR1 or hR2) and the oA parameter. It now takes the CRYSTALLOGRAPHIC
  CELL printed with the final optimized geometry. A SUPERCEL output's static
  path is mapped to the supercell's reciprocal basis, f_super = E f_prim with
  E the printed EXPANSION MATRIX OF PRIMITIVE CELL (rows), as the library
  route gives it; before, it was written in the primitive cell's basis while
  BAND ran on the supercell. An R cell given in rhombohedral axes (CRYSTAL
  echoes it as a = b = c, alpha = beta = gamma) is read in hexagonal axes.
  `opt2d12` now writes the child of a SUPERCEL parent as the supercell in P1
  (below), and its phonon-dispersion path is that cell's own. Single-point
  outputs without SUPERCEL or geometry editing are unchanged.
- **Static band paths compute their lattice-dependent points for the cell.**
  Without the seekpath library the static paths of body-centered tetragonal
  (tI1, tI2), rhombohedral (hR1, hR2), face-centered orthorhombic (oF1) and
  C-centered monoclinic (mS1, SeeK-path's mC1) lattices held the
  lattice-dependent points of one particular cell (tI1's Z at 0.341718 for
  every crystal), so their segments ended off the special points of any
  other cell. Those points are now evaluated for the cell CRYSTAL prints
  with SeeK-path's HPKOT formulas (new `Crystal_d3/hpkot_static.py`, numpy
  only; transcribed from SeeK-path, MIT, Hinuma et al. 2017) and given in
  CRYSTAL's reciprocal basis, with the ISS the library route chooses: the
  static and library routes write the same BAND path for the 30 centered
  corpus outputs; for about 1% of random cells the ISS chosen can differ.
  The warning and the "install seekpath" note are gone for these paths.
  Cells not in SeeK-path's setting (oF axes not in a < b < c, a C-centered
  monoclinic cell whose a-c mesh SeeK-path reduces) or of another variant
  (oF2/oF3, mC2/mC3) keep the old entry with a warning that gives the
  reason; aP, mP, oP, oI, oS and oC paths still warn as before. Paths
  written without an output file are unchanged.
- **Band-path ISS stays at most 999 for long-axis cells.** When no ISS up to
  999 placed the lattice-dependent points within half a k-point spacing (at
  10000 points), both the seekpath library route and the new static
  computed-point route wrote the points exactly, with ISS up to ~10**6 (a
  C2/m cell of 9.7 x 3.7 x 20.8 A: ISS 959108 with the library, 588382
  statically). fort.25 writes the segment ends as fixed-width integers (6I3
  in App. D p.447; 6I5 measured), so such ends overflowed there. ISS is now
  the one up to 999 that moves the points least (990 for that cell, a move
  of 8.6e-4 1/A, 0.13 of the k-point spacing at NSUB 1000), with a warning
  giving the move. Deck change: the ISS and segment-end integers of BAND
  paths for such cells only; cells that fitted before are unchanged.
  Untested on CRYSTAL.
- **The bundled Pu basis record (`294`, triplezeta and doublezeta) declared
  11 shells but holds 9.** CRYSTAL would have read the next atom's records,
  or `99 0`, as Pu's missing shells. It now says `294 9`, as the source set
  does. That alone did not make Pu decks run: its f-shell charge was also 3
  short of neutral (see the Ac-Es fix below). **Deck change:** any deck with
  Pu and a bundled external basis. Untested in CRYSTAL.
- **The BAND gap of a doped run used the wrong band.** Without CRYSTAL's
  "TOP OF VALENCE BANDS" line the occupied-band count was the header's
  "N. OF ELECTRONS PER CELL" // 2. It now comes from the Mulliken
  "NO. OF ELECTRONS" count (never the ALPHA-BETA print, which is the spin) of
  the BAND .out or its SCF parent beside it; an odd or fractional count
  leaves the BAND gap unset instead of wrong.
- **Mulliken net charges of pseudopotential (ECP) atoms.** The net charge
  was the Mulliken Z column minus the population; for an ECP atom that
  column holds CRYSTAL's label (282 for Pb) and the population counts valence
  electrons only, so Pb came out near +261. It is now measured against the
  pseudopotential's NUCLEAR CHARGE. An ECP atom (Mulliken Z 200 + Z) whose
  valence charge the .out does not give is flagged with no charge, rather
  than measured against Z (the 21 ECP outputs of the test corpus all give
  it). Newly extracted rows change for ECP atoms; rows already in a database
  are not migrated (re-extract to update).
- **Derived decks keep a charged parent's electron count when the basis set
  changes.** An explicit-basis parent carries its charge in `CHEMOD` +
  `CHARGED`; a child with an internal BASISSET got the same records, which
  there only reshape the guess, so it ran neutral. The charge is now written
  as `DOPING`. `CHEMOD` is copied only when the child loads the same basis
  set (it was copied whenever the atom counts matched, although its shell
  charges are per shell of the parent's basis). When the basis changes and
  the parent's records are exactly what MACE's ion charges write for it, the
  equivalent `CHEMOD` is worked out again for the child's basis (net charge
  as `DOPING`, or `CHARGED` in an explicit basis); any other `CHEMOD` is
  dropped with a warning that gives the reason. `DOPING` is always kept,
  with a warning when the atom count changed. Deck change for such children.
- **A derived deck with an odd electron count is no longer written closed
  shell.** It gets the parent's `SPIN`/`SPINLOCK`, or opt2d12 refuses it
  with the reason.
- **opt2d12 kept no charge records when the parent deck failed to parse,**
  although the file was there; they are now carried from its text. **With
  no parent deck at all**, a charged parent's `DOPING` is recovered from the
  .out's Mulliken count; without a Mulliken print a warning says the charge
  state cannot be recovered when the .out shows a sign of a charged run
  (CHEMOD, CHARGED, "UNIT CELL NOT NEUTRAL", a renormalized guess), and an
  informational note says so otherwise (a neutral FREQ parent is not warned
  about). Deck change for such children.
- **A failed run's message says what CRYSTAL printed.** The queue manager
  reported "Calculation failed with unknown error" for any error it had no
  pattern for; it now quotes CRYSTAL's `ERROR ****` line, and for an .out
  that ends in `MPI_Abort` with no error text says that Pcrystal can hide an
  input error such as INPBAS "UNIT CELL NOT NEUTRAL" (measured on CRYSTAL23:
  only a serial run prints it). The error type, and so recovery, is
  unchanged; such outputs were and are never counted as finished (pinned on
  the real outputs).
- **An END after `CHEMOD` in a library-basis deck is caught.** After
  `BASISSET <library>` the lines are already block 3, so an `END` closing
  `CHEMOD` ends the input there and CRYSTAL23 silently drops the DFT block
  (the run falls back to HF; measured on CRYSTAL23). MACE never wrote it;
  `mace ions` now refuses a deck that has one (also with `--replace`, which
  used to strip `CHEMOD` and keep the stray `END`), and opt2d12 warns that
  such a parent ran HF. A `CHEMOD` record whose atom labels stand on lines of
  their own, the shell charges on the next line (as CRYSTAL also reads it),
  is now read, carried and stripped whole; it used to be cut after one line
  per atom, which hid that `END` from the check.
- `mace submit DIR` with a relative directory (or `sub/deck.d12`) submitted nothing:
  every deck failed with "Failed to generate/find script". Absolute paths were fine.
- **cif2d12 / `mace convert` custom setup shows the functional menus again.**
  Since 1.1.2 every CIF was treated as a parent whose functional could not be
  identified. The category and functional menus were replaced by a free-text
  prompt that rejected menu numbers, and blank answers wrote HSE06 without
  D3. Blank answers write HSE06-D3 again, as in 1.1.1 and as the defaults
  banner says. opt2d12 parents keep the replacement prompt.
- **Workflow planner convergence menus accept yes/no again.** The SP and OPT
  convergence questions became 0-3 menus, and any other answer was quietly
  taken as 0 (keep the previous step's values). y/yes and n/no now mean
  what they did in 1.1.1. SP: y/yes is Very tight (TOLINTEG 9 9 9 11 38,
  TOLDEE 11), n/no keeps the previous step's tolerances. OPT: y/yes is
  TOLDEG 1.5e-5, TOLDEX 6e-5, TOLDEE 8, MAXCYCLE 1000; n/no is TOLDEG 3e-5,
  TOLDEX 1.2e-4, TOLDEE 7, MAXCYCLE 800. A note is printed for both, and
  any other answer is asked again.
- **Answers piped into interactive cif2d12 and opt2d3 runs are read again.**
  cif2d12 ignored a piped answer to the space-group mismatch question, and
  opt2d3 ECHG refused with "no terminal to ask". If the pipe runs out, the
  question takes the unattended choice (cif2d12) or the run writes no deck
  and exits 1 (opt2d3), with no traceback. `--batch` and `--config-file` runs
  are unchanged.
- **Out-of-step piped answers stop the run.** With no terminal, three invalid
  answers in a row to any question (yes/no, numbered menus, numbers,
  opt2d3, the planner and plotting) now stop the run with an error; at a
  terminal the question still asks until answered.
  Before, the next blank answer took the default and the run carried on with
  answers out of step. In the planner this covers its number, memory,
  walltime and level menus too, and an off-menu Advanced BAND path option or
  path format is asked again (it was taken as a custom path, or as labels).
- **A fort.20 staged by hand for GUESSP is kept.** Copying a density matrix
  into the job's scratch directory as fort.20 was how GUESSP restarts were
  done in 1.1.1. Since 1.1.2 the job script stripped GUESSP when there was
  no `<job>.f20`/`.f9` next to the deck, or copied the job's own old `.f9`
  over it. A non-empty fort.20 in scratch is now used unless the job's `.f9`
  is newer (then it was left by an earlier run of the same job). The same
  applies in the TWOCOMPON branch.
- **The `.mace_scratch` fallback is cleaned up.** When `$SCRATCH` is
  unusable, CRYSTAL runs in `<submit dir>/.mace_scratch`, which was never
  removed. Its job directory is now removed after a run that reached
  CRYSTAL's normal end; a run that did not keeps it for an OPTGEOM restart.
  Any scratch fallback is now also noted in `<job>.out`.
- **A job script from before 1.1.3 is no longer reused silently.** `mace
  submit` warns when a MACE-written `<job>.sh` has no scratch guard, with the
  `--overwrite-sh` hint; the script is not changed. A workflow's
  `workflow_scripts/` template from before 1.1.3 that the plan names is
  rebuilt from the plan, the way a fresh execution of the plan writes it
  (the old copy kept as `<name>.pre-1.1.3`), so later steps get the scratch
  guard and RESTART/GUESSP staging; one the plan does not name is warned
  about, with the re-plan hint.
- **Decks are written through a symlink and keep their permissions.** The
  atomic deck write replaced a symlinked deck with a regular file (the link
  target kept the old deck) and gave every deck the umask's mode. The link
  target is now written and the deck's mode kept, as in 1.1.1; `mace ions`
  uses the same writer.
- `mace <command> -h` (and a help flag after other arguments, or after
  `--no-banner`) shows that command's help instead of the general usage.
- `mace submit DIR` on a mixed .d12/.d3 directory stops with a message (exit
  1) instead of an EOFError traceback when stdin has no answer.
- `mace convert` without `--batch` stops with a message pointing at
  `--batch`, instead of an EOFError traceback, when stdin ends before the
  setup questions are answered.
- opt2d12's free-text LEVSHIFT ("ISHIFT ILOCK") and HISTDIIS-size answers
  are asked again on bad input instead of quietly taking the default (which
  pushed the rest of a piped answer file one more question out of step).
  Deck change: an ILOCK other than 0/1, a single LEVSHIFT value or a
  non-positive HISTDIIS size is no longer written or replaced.
- The planner's optimization-type menu asks again on an off-menu answer
  instead of silently keeping the previous type, and its CHARGE+POTENTIAL
  output-type question asks again instead of crashing with a KeyError.
- `mace submit --interactive` accepts walltimes like 12:00:00, 2:00:00 and
  12:00, which its own prompt gives as examples (they were refused and the
  default used).
- opt2d3 runs answered through a pipe read their later questions there too
  (missing-wavefunction "Continue anyway?", each file's ECHG map plane and
  POTC points with shared settings), as in 1.1.1, instead of refusing for
  want of a terminal. Runs given a `--config-file` still never ask.
- `opt2d3 --batch --config-file` takes the calculation type from the config
  instead of asking for it first; with none anywhere it exits 1 naming
  `--calc-type`.
- opt2d12's grid menu marks the parent's grid as the default instead of
  always labeling XLGRID. The default itself is unchanged.
- **The HSE-type odd-electron crash is no longer resubmitted with more
  memory.** A DIIS NaN at SCF cycle 1-3, then ranks killed by signal 11/6/9,
  on an odd-electron open-shell run with HSEsol-3c, HSE06 or HSEsol is
  reported as `screened_hybrid_odd_error`, which names PBE0/PBEsol0 as what
  converged and keeps the functional. A closed-shell out-of-memory kill is
  still `memory_error`.
- A job squeue still lists as COMPLETED is classified from its .out; a run
  whose ranks were killed (SLURM records OOM and segfault kills as
  COMPLETED) is no longer marked completed.
- Re-extracting untracked outputs (no calculation record) also removes the
  rows an older extractor wrote and the current one no longer produces (e.g.
  1.1.1's band fields of DOSS runs), when the run re-extracts every output
  of that material it can see; a database rebuild is no longer needed for
  that.
- Re-organizing (the populate scan) corrects the calc type of a record an
  older MACE typed from its file name (e.g. an SP named `*_OPT_*` recorded
  as OPT); workflow-numbered types are left as they are.
- **Workflow BAND steps write the deck opt2d3 writes for the same choices.**
  The planner's Advanced BAND menu stored keys opt2d3 does not read:
  "labels" gave k-point vectors, and literature, SeeK-path and custom paths
  gave a deck with no path (NLINE 0). It now writes the keys opt2d3 saves,
  asks for a custom path at planning time instead of promising it "during
  execution", and stores a band range as first and last band. Checked on the
  59 periodic SP parents of the test corpus, with and without seekpath: the
  engine default, planner Basic, quick-start and Advanced formats 1-4 are
  byte-identical to opt2d3's decks for the same choices.
- Interactive opt2d3 reads a slab's "CORRESPONDING SPACE GROUP N." line. It
  wrote the P1 (triclinic) path for slab outputs that print only that line;
  the config route already read it.
- Workflow D3 steps use a plan's per-material Expert configurations; the
  engine ignored them and ran its own default deck. A config file planned on
  another machine is found again under this run's `workflow_configs`.
- Workflow FREQ steps keep a FREQ parent's NUMDERIV when the plan names
  none, as opt2d12 does. At the planner prompt a blank answer means "the
  parent's, else none" and 0 means no keyword.
- The legacy WorkflowExecutor step generator passes its step config (and an
  Expert plan's saved config) as `--config-file` with `--non-interactive`;
  it used `--options-file`, which opt2d12 only writes to, so nothing was
  applied and no deck was written.
- **opt2d12 writes the child of a SUPERCEL/SUPERCON parent as the supercell
  it computed**, every atom in P1 in CRYSTAL's order, instead of the host's
  space group with the supercell's cell and asymmetric atoms (the diamond
  2x2x2 SP child was SG 227, a = 5.01 A, 5 atoms). The parent's DOPING,
  CHEMOD and GHOSTS are kept, and SHRINK is written in the P1 form (same
  mesh). The workflow engine goes through the same code.
- Decks derived from a GHOSTS parent keep its ghost atoms: opt2d12 copies
  GHOSTS when the child lists the same atoms (ghosts are not counted in the
  electron parity), and opt2d3 DOSS projects the ghosts' basis functions on
  their own (`<El>_GHOST`) instead of under their element.
- opt2d12 `--calc-type` (and `--opt-type`) win over a `--config-file`: a
  config without `calculation_type` no longer turns `--calc-type SP` into an
  OPT deck.
- Without the seekpath library, centrosymmetric C-centered groups (C2/m,
  C2/c, Cmcm, ...) get SeeK-path's path (9 and 11 segments) instead of the
  primed Cc / Cmc2_1 path (18 and 22); the static k-path table has no
  duplicate keys, and aP1's labels name its segments. An orthorhombic F cell
  gets the oF variant SeeK-path takes (1/a^2 vs 1/b^2 + 1/c^2), not the
  shortest-axis guess, and a centrosymmetric oF2 cell the time-reversal
  path.
- **The bundled La-Lu (ERD, full.basis.triplezeta/doublezeta) and Ac-Es (all
  three families) basis records have neutral formal shell charges;** every
  La-Lu or Ac-Es deck on them stopped with INPBAS "UNIT CELL NOT NEUTRAL".
  La-Lu: +2 on the first empty s/sp shell and +1 on the first empty d,
  keeping the published 4f count (the site prints an Ln3+ guess); Ac-Es: the
  site's f charge restored. Exponents and coefficients are unchanged.
  Untested in CRYSTAL.
- Basis records CRYSTAL cannot read (stuttgart 204, 219, 220, 227, 294) are
  refused on every deck path, naming the record, instead of being copied
  into decks.
- A walltime-killed two-component OPT whose TWOCOMPON block asks for
  GUESSP/GUESSPNOSO/2NDVARIAT is no longer restarted from its own
  two-component fort.20 (io error Read_real_1d), and the job script drops
  GUESSROTM (and its angles) with the TWOCOMPON GUESSP it drops for want of
  a one-component matrix. Refreshed old job scripts get the same staging.
- CHEMOD records with the label on its own line and integer charges are read
  correctly (labels, and the charge an explicit-basis CHEMOD sets).
- The static k-path of an optimized SUPERCEL cell is the host's path in the
  supercell basis, as for its SP.
- The custom-labels BAND menu lists the labels for the output's own lattice
  (it always showed cubic P), and the shrink fallback (no SHRINK in deck or
  output) uses the computed cell.
- Computed geometry blocks of SLAB and POLYMER outputs are recognized.
- opt2d3 no longer overwrites X_sp's deck with X_opt's when both are in one
  directory: they are written as `X_sp_<type>.d3` and `X_opt_<type>.d3`; a
  lone output keeps the short name.
- The seekpath-missing note is no longer printed on every import.
- An opt2d12 `--config-file` whose top-level `functional` is a 3c method
  (HSESOL3C, PBEH3C, ...) and that names no basis gets the method's own
  basis. On a parent with an explicit basis the deck said "BASISSET /
  EXTERNAL (from original D12)", which CRYSTAL cannot read. A functional
  under `method_modifications`, and a config that names its basis, already
  worked.
- Charged-system setup refuses when the parent .out lists other shells than
  MACE's table for an internal library, and the CHEMOD-with-symmetry refusal
  names the orbit-wide CHEMOD and MODISYMM alternatives.
- opt2d3 reads the electron count CRYSTAL prints.
- `mace defects analyze` says when a `--site-label` sits on the wrong
  element instead of reporting DOPING not applied; an ECP site's electron
  count is given as its parity only; and a warning is printed when the
  image-charge term is applied to a far-from-cubic supercell.
- A Mulliken atom with a second all-electron basis (label Z + 1000) is
  measured against Z instead of being flagged as an ECP of unknown valence;
  CRYSTAL takes an ECP only on Z + 200 or Z + 300 (manual p.25).
- A SPINLOCK, ATOMSPIN, BETALOCK or DOPING phonopy parent is refused once,
  not twice.
- `basis_library` loads a family from `MACE_HOME` in a `copy_dependencies`
  job directory.
- The test suite writes no bytecode, so it never writes into the test
  corpus.
- The opt2d12 tests that answer prompts through a runner import the
  checkout under test; they imported whichever `mace` PYTHONPATH named.
- **A job's completion callback runs the MACE that submitted it.** Job
  scripts run under `bash --login`, which re-reads `~/.bashrc`; an
  `export MACE_HOME=...` there (with PYTHONPATH, as `setup_mace.py
  --add-to-path` writes it) decided which install's queue manager and
  `mace` package ran the job's follow-up and recovery steps, so a second
  install (a side-by-side test, a shared group copy) had its callbacks run
  by the one the dotfile names, silently. Job scripts now set MACE_HOME and
  PYTHONPATH to the generating install after the login shell has started,
  and say so in the job log when the login shell named another; `mace` and
  the queue manager export their own install to the workflow's generator
  copies. Job scripts and `workflow_scripts/` copies written before 1.1.5
  are not pinned; regenerate them with `mace submit --overwrite-sh`, or
  re-plan the workflow.
- The input-settings queries (`python -m mace.database.query.queries`) and
  `settings_extractor.query_calculation_settings` read the extracted deck
  settings from `input_settings_json`, where they are stored; they read
  `settings_json` (the job's workflow metadata), so a stored extraction was
  never shown and calculations with only workflow metadata were listed.
  Databases from before that column still have theirs read from
  `settings_json`.
- Importing `mace.utils.show_properties` no longer opens (and creates) a
  `materials.db` in the current directory; the query runs only when the
  script is run.
- `setup_mace.py --update-templates` checks the job-script templates in
  `mace/submission/` and reports each one; it looked in the removed
  `code/Job_Scripts/` and silently did nothing. Its `--help` examples name
  `setup_mace.py` (they named `setup_crystal_tools.py`).
- The install docs (README, INSTALLATION, the Crystal_d12 and Crystal_d3
  READMEs) require Python 3.11 or newer, the oldest version CI tests; they
  said 3.6+ or 3.7+, which nothing tests.
- `pytest.ini` no longer says the Crystal_d12 menu tests run separately;
  every test is under `tests/`.
- **Automatic workflow steps after the first ran nothing.** The engine kept
  a step's script generator as `<job>.sh` and ran `bash <job>.sh <job>`,
  so the generator overwrote the file bash was reading. The submitted
  script stopped after the module loads (no CRYSTAL run, no completion
  callback) and each follow-up ended in seconds with no `.out`; the first
  step of `mace workflow --execute` and quick-start did the same. The
  generator now runs from a separate copy (broken since 1.1.2).
- A job that left the queue without writing its `.out` (canceled while
  pending, killed before CRYSTAL started) stayed `submitted` for good, so
  its workflow could neither finish nor recover. Once sacct reports the
  job ended, the record is marked failed with that reason (`no_output`).
  `mace monitor --status` also counts submitted calculations, which it
  left out of the totals.
- **`mace submit --progress` split a workflow over two databases.** It
  recorded the submissions in `./materials.db`, but the plan's job scripts
  export the workflow context, so the completion callback recorded the step
  again in `.mace_context_<id>/materials.db`, filed its properties in
  `./materials.db` under the second record's id, and left the first record
  `submitted`. The submissions are now recorded in the context database,
  as `mace workflow --execute` records its first step, and the queue
  manager gives property extraction, settings extraction, the workflow
  engine and error recovery the database it opened rather than the
  `materials.db` default.
- `mace manager --d12-dir DIR` submitted nothing: with no `--callback-mode`
  it ran the job-script completion callback, which outside a workflow only
  records status. It now runs a full cycle (update the records, catch early
  failures, submit the decks in `DIR`), and the manager is built once.
- Jobs the queue manager submitted outside its database's directory
  (`mace manager --organize`, or `--d12-dir` elsewhere) had completion
  callbacks that opened a new `materials.db` in the job directory, so the
  manager's records stayed `submitted` with no properties. The job script
  now exports `MACE_DB_PATH`, and the callback passes it as `--db-path`.
  Job scripts written before this release do not carry it; regenerate
  them, or run `mace manager --callback-mode status_check` from the
  database's directory to bring the records up to date.
- `mace submit DIR` and `mace submit DIR/x.d3` with a relative path
  failed for D3 decks ("Failed to generate/find script") and submitted
  nothing, as the D12 path did before 1.1.5; the D3 path now resolves the
  directory first too.
- `mace analyze --extract-properties` filed every property a second time
  with no calculation id: it matched an output to its calculation by the
  exact path string, and a scan passes relative paths where the database
  holds resolved ones. It now compares the resolved paths and, in an
  in-place directory with several calculations, picks the one whose deck
  has the output's name. In a workflow directory it stores into the
  context database `mace database` and `mace monitor` read (not
  `./materials.db`), and "Materials processed" counts the materials (it
  always printed 0).
- `mace database --action export` (CSV, JSON) left out every property
  unless `--include-property` named some; it now includes them all by
  default, as intended.
- `mace database --action export --format excel` without openpyxl wrote
  CSV text into the `.xlsx` file and reported success; it now stops with
  the install hint and writes nothing.
- The DOSS step a quick-start, `--progress` or Basic plan chooses (and the
  engine's default) was described as total DOS in a -20..20 eV window, but
  the writer reads neither of those keys and wrote opt2d3's Enter-through
  deck (projected by element and orbital type, all bands). The plan now
  carries opt2d3's Enter-through answers, so it says what the deck holds.
  Decks are unchanged.
- Messages: the engine no longer warns that every shipped job-script
  template "will cause submission issues", the queue manager no longer
  says "Could not extract job ID" before submitting the generated script,
  workflow execution no longer says the workflow "completed synchronously"
  when only its first step is submitted, `mace database --action stats`
  suggests `--action export --format csv --output FILE` (not
  `--export-csv`), and `mace submit --help` lists `--walltime`.
- With the whole script submitted (above), follow-up steps ran CRYSTAL and
  then stopped with "syntax error: unexpected end of file" before their
  completion callback: the engine replaced the template's callback section
  from `# ADDED` up to the `fi` in the word "file" of a comment. A template
  that already carries the current callback is now left as it is.
- **An optimization FINALRUN keeps re-starting is recognized and
  recovered.** Measured on CRYSTAL23 (Si, B3LYP-D3, TOLINTEG 7 7 7 7 14):
  after each FINALRUN re-start the energy jumped alternately down and up by
  about 2.2E-4 Ha and the run converged to the other of two states, forever
  within one job (SPIN plays no part; FINALRUN 3 loops too, FINALRUN 0/2
  converge falsely). Three or more such alternating re-starts are now a
  `finalrun_loop` error, whether the job reached MAXCYCLE, was killed at its
  walltime (it is not sent to the timeout recovery: more time only loops
  longer) or is still running. The recovery starts a fresh optimization from
  the lowest-energy geometry reached with TOLINTEG one preset level up
  (Standard 7 7 7 7 14 -> Tight 8 8 8 9 24 -> Very tight 9 9 9 11 38; no
  value is ever lowered) and every other setting kept; the deck is kept as
  `<JOB>.d12.orig`, the output as `<JOB>.out.finalrunloop<N>`
  (`recovery_config.yaml`: `finalrun_loop`, max_retries 2). A deck already at
  Very tight is not resubmitted.
  A job still running in such a loop is canceled and recovered at once by
  the next check of running jobs: a `mace manager` cycle, a status or
  early-failure check, or the completion callback of any other job in the
  database. A lone job has no such check while it runs; killed at its
  walltime it runs no completion callback, and the next `mace manager` or
  callback recovers it.
- **"* OPT END - FAILED *" is an unconverged optimization**, not a
  segmentation fault/runtime error. `mace completion` sorted an optimization that
  reached its OPTGEOM MAXCYCLE into `potential` because of the MPI_Abort that
  follows; it now has its own `opt_unconverged` category, and the queue
  manager recovers it (`opt_unconverged_error`) with a fresh optimization
  from the best geometry reached instead of leaving it as an unknown error.
- `export` in the interactive database explorer (`mace database --action
  interactive`) failed every time with "unexpected keyword argument
  'material_ids'"; it now writes the materials of the last query.
- `mace manager` (a full_check cycle) printed its "Database Stats" line
  twice; it prints it once.
- `d12_writer.write_optimization_block` stopped with ModuleNotFoundError: it
  imported its OPTGEOM writer from `d12_calc_opt`, which does not exist. It
  now uses the one in `d12_calc_basic`. No generator calls it, so no deck
  changes.
- **`mace opt2d3 --help` names `mace opt2d3`.** The help of the commands
  handed to a script's own options (convert, opt2d12, opt2d3, opt2cif,
  completion) began `usage: CRYSTALOptToD3.py ...`, the script's file name;
  it now reads `usage: mace opt2d3 ...`, and opt2cif's examples say
  `mace opt2cif`. Run directly, a script keeps its own name.
- **`mace convert|opt2d12|opt2d3|opt2cif|completion --help` show the
  examples.** mace_cli held hand-written help for these commands that was
  never printed, since `--help` goes to the script's own options; the
  examples and notes in it (convert reads a directory, opt2d12/opt2d3 take
  no positional file, completion's status categories) are now in the
  scripts' help, and the unreachable copies are gone. A command now has
  either mace_cli's help or the script's, never both.
- **`mace ions --help` has examples**, as `mace phonopy` and `mace defects`
  do: an ion cluster from cif2d12 decks with `--ion-charges auto -o DIR`, an
  explicit `--net-charge`, `--site-states`, and `--parent-out` to check the
  shell counts, plus a pointer to "Charged systems and ion clusters" in
  `Crystal_d12/README.md`.
- **opt2d3 BAND path method 4 with all segments of one kind writes the
  path.** "Fractional coordinates", not mixed, all coordinates stopped with
  UnboundLocalError ('Path') before asking for the segments; all coordinates
  and all labels both stored their segments where the writer of a manual path
  does not read, so the deck would have been an empty BAND (0 segments). The
  segments given are now written, as the mixed route writes them, and a shrink
  raised to reach a point (1/4 needs a multiple of 4) is the shrink of every
  segment and of the BAND line.
- **`mace completion` no longer lists every properties run it cannot type as
  a band structure.** A finished run whose `.d3` drives none of BAND, DOSS,
  TRANSPORT or CHARGE+POTENTIAL (a matrix dump, BASISSET with print codes
  60/64, for one) was listed as "Band structure (D3 BAND)" and organized into
  `completeband/`; it is now "Other properties (D3)", in
  `completeproperties/`, and the queue manager's scan of finished outputs
  records its type as PROPERTIES. A `.d3` whose file name names a properties
  type keeps that type.
- **A BAND or DOSS run no longer rewrites the material's formula.** Its
  completion read the `.d3` as a geometry: a BAND header such as
  `4 0 1000 1 37 1 0` read as atomic number 4 plus three coordinates, so the
  formula became `Be` (`NeBO` for a DOSS deck), and any formula of one or two
  characters (Si, C2, Cu) set by the OPT or SP was replaced by it, on
  `mace submit --progress` and `mace run --track` alike. A formula is now
  read only from a deck's CRYSTAL/SLAB/POLYMER/MOLECULE geometry block, a
  short formula is no longer taken as a wrong one, and a material first
  recorded by a properties run (formula `Unknown`) gets its formula from the
  next SCF deck.
- **opt2d3 DOSS and TRANSPORT keep the parent's k-mesh without its
  `.d12`.** With only the `.out` and `.f9` in the directory, the NEWK line
  fell back to `8 16` (`12 24` for slabs and polymers) without a word, though
  the `.out` prints the SCF mesh (`SHRINK. FACT.(MONKH.) 10 10 10`, Gilat
  20): an SP on SHRINK 10 20 got a DOSS deck on 8 16, where the deck with the
  `.d12` beside it, and the workflow's, had `10 20`. The mesh is now read
  from the `.out` when there is no `.d12`, giving the same deck; only when
  neither has one is the default used, and opt2d3 says so. Decks made with
  the `.d12` present do not change.
- **`mace submit --track` names only this run's jobs in its `Job ID(s):`
  line.** It listed every job the folder's `crystal_job_status.json`
  remembers, so a second tracked submission in a folder also named the
  earlier jobs (copied into `scancel`, it would cancel them too).
- **GUI: Submit refuses a walltime that is not a SLURM time** (`30`,
  `2:00:00`, `1-00`, `1-00:00:00` and the like) before anything runs, and a
  deck whose earlier submission sbatch refused is no longer called "already
  queued".
- **GUI: a server keeps answering after the terminal that started it closed**
  (a plain `ssh node mace gui ...` without a tty): File > Open Folder lost its
  reply on the dead pipe.
- **GUI: the Jobs panel opens a job's Job document** (progress, ETA) with
  Details or a double-click; an ended job's document shows its name, size,
  node and output from `sacct`, and `mace gui connect --slurm` names the
  job's node instead of the node ssh reached.
- **GUI: Run Analysis Export > PNG/SVG saves the file, and Copy Image to
  Clipboard copies.** Both turned the image into a file with a request the
  page's security policy refuses, so nothing was saved and the copy always
  failed.
- **GUI: Calculation Setup ignores views that hold no file of their own**
  (a file opened as text, Run Analysis, Compare with Input). It treated them
  as an unsaved structure or an output to follow and showed "no path given".
- **GUI: the FREQ modes view stays inside its editor group.** A long output
  name widened it past its panel: the molecule sat off center, and in a split
  group it was off screen.
- **opt2d3 copies the wavefunction only for a deck it writes.** A run that
  ended without a deck (`< /dev/null`, "The input ended before every question
  was answered") still left `<base>_band.f9` / `<base>_doss.f9`, a full copy
  of the `.f9`, in the directory. The copy is now made after the deck is
  written.
- **`mace ions` (and the CHEMOD carry of opt2d12) refuses decks that reorder
  or rebuild the atoms.** It kept its own list of 17 geometry records and
  missed ATOMORDE, which reorders the atoms after the geometry is read, so
  CHEMOD labels taken from the deck order pointed at other atoms; FULLE,
  NANOMULTI, HYDROSUB, CLUSTSIZE, NANORODS, NANOCRYSTAL and WULFF were
  missed too, and PRIMITIV (a properties keyword) was refused. It now reads
  the one table `mace preopt` uses, on the first word of each line; records
  that keep every atom in place (ELASTIC, BREAKELAS, ATOMROT, MOLEXP,
  MOLSPLIT, USESAED, POINTCHG, GHOSTS) are still accepted. Decks without
  these records are written as before.
- **Decks with a non-standard origin (IFSO > 1) work in `mace ions` and
  opt2d12.** The origin-shift record after the space group was read as the
  cell, so `mace ions` refused valid decks with confusing errors and the
  opt2d12 child of a CHEMOD parent lost its CHEMOD; the child also kept
  `0 0 2` without the shift record, a deck CRYSTAL misreads. The record is
  now skipped where the atoms are counted and carried into the child.
- **`mace preflight` tests the CRYSTAL build the job scripts use.** Its
  module probe always ran `module purge` and ignored `[modules] purge =
  false`, so it could load another build than `mace run` and the job
  scripts; it now uses the same probe as `mace run`, and runs none when
  CRYSTAL comes from `[crystal] bin_dir`. When `bin_dir` has no serial
  `crystal`, preflight and the GUI say so instead of using a `crystal`
  found elsewhere.
- **A whitespace-only `SLURM_JOB_ID` no longer counts as a SLURM
  allocation.** A second definition of `outside_allocation_env` replaced the
  first and dropped every `SLURM_*` variable for it.
- **GUI pre-optimizations in two threads with the same method keep their own
  calculator.** The tier's factory went through a shared table that each run
  set and cleared, so one run could use the other's (another charge or
  backend) or silently fall back to the springs.
- **`mace recover` says what it does: it prepares fixed inputs and job
  scripts and submits nothing.** The command list, `docs/cli.md` and the
  help promised a resubmission, and the action printed a `resubmit_delay`
  that nothing applies. It now ends with the exact `mace submit` (or
  `sbatch`, for a bumped job script) command for each prepared recovery;
  a recovery prepared but never submitted is reused when the action runs
  again and no longer counts toward `max_retries`. The queue manager still
  resubmits tracked runs itself.
- **The workflow planner's "Memory per CPU" takes `<n>G` or `<n>M` (optional
  B) only.** A bare number became gigabytes per CPU (`4096` wrote
  `--mem-per-cpu=4096G`, a job SLURM can never schedule) and `4K` was
  accepted; both are now asked again, as `mace submit -i` does. Valid answers
  give the same `#SBATCH` line as before.
- **GUI: Submit loads the database safety helpers without the database
  package**, as every other GUI page does (no change in behavior).
- **Stored band gaps come from the converged SCF (a correction to stored
  data).** The extractor took the last DIRECT and the last INDIRECT gap line
  separately, so an SP whose cycle 0 printed an indirect gap of 3.4768 eV and
  that converged to a direct gap of 4.3946 eV was stored with
  `indirect_band_gap` 3.4768 and `band_gap_type` "indirect"; 24 of the 244
  SP/OPT outputs of the test corpus had a wrong type or a stale gap. Only
  the last SCF counts now: its gap kind and value per spin channel, the
  last ALPHA/BETA summary rather than the first, and a run that ended
  conducting stores `band_gap` 0 and no gap kind instead of the last gapped
  cycle's value. A spin-polarized run's `band_gap` (without CRYSTAL's
  ALPHA/BETA summary) and its direct/indirect gap are now the fundamental
  gap of the converged SCF, min(CBM) - max(VBM) over both spin channels
  from its TOP OF VALENCE / BOTTOM OF VIRTUAL BANDS lines, instead of the
  last channel's gap (1.1.4) or the smaller channel gap: when the
  valence-band top is in one channel and the conduction-band bottom in the
  other, the gap is smaller than both (the FSI radical: 3.0356 eV, beta
  HOMO to alpha LUMO, not 3.3875 or 8.7278), and edges that overlap store
  0 with no gap kind. Direct or indirect follows the k points of the VBM
  and CBM. `alpha_band_gap` and `beta_band_gap` keep CRYSTAL's per-channel
  values; when CRYSTAL prints that summary, the gap used by the derived
  estimates is the same fundamental gap. A two-channel run without the
  summary is now stored with `spin_polarized` true (it was false).
  Re-extract existing databases (see Upgrading).
- **`mace cluster job` and the Job document show a started job as running.**
  scontrol and squeue answers are cached for a minute, so about a minute
  after a job started it showed "PENDING (Priority), Elapsed 0 s" next to
  its live SCF progress. When the job's output was written after SLURM was
  asked, MACE asks again; within the 15 s refresh floor the output wins and
  the job is shown as running with a note instead of an elapsed time.
- **The Copy Diagnostics / Sign Out smoke test no longer fails at random**
  (tests only): a menu closes when the browser window loses focus, which a
  busy machine with several headless browsers caused between the click and
  the wait; the test opens the menu again and runs in Firefox too.
- **Tests that compare with an older tree use the 1.1.5 base commit**, so
  they run in a public clone instead of skipping (tests only).

### Known limitations (planned for MACE 1.1.6)

- The plot gallery (many outputs in panes, cycled, with a caption table) is
  not in 1.1.5; the split view compares two documents side by side, and
  Compare Runs overlays convergence.
- A database update that a dev node's CPU-time limit stops does not continue
  by itself; start it again with "Only new and changed files" (or `mace
  analyze ... --skip-unchanged`) and it continues where it stopped.
- GUI preferences are per user (`~/.config/mace/gui.json`); per-folder
  overrides (`.mace/gui.json` in a project) are not there yet.
- `mace gui connect --slurm` is not in the standalone `scripts/` connect
  scripts.
- The job ETA (`mace cluster job`, the GUI's Job document) covers
  optimizations, single points and frequencies. The IR/Raman CPHF/CPKS stages
  and tensors have no time estimate yet (a running one says how long it has
  run, and the total reads "at least ... left"); a FREQ preoptimization has
  no inner levels; properties runs (band, DOS, transport and the other
  properties blocks) show which block runs without a time; EOS, ELASTCON,
  anharmonic and phonon-dispersion runs get no dedicated estimate; job
  arrays and workflows get no total. The estimate was backtested on a corpus
  where every run converged and none used SMEAR, so it can be too short for
  an SCF that stalls or a metallic system and too long for a GUESSP restart
  (flagged). Planned for 1.1.6: CPHF stage timing, properties blocks,
  EOS/ELASTCON, anharmonic and dispersion runs, and totals for arrays and
  workflows (see `docs/cli.md`).
- `mace recover` prepares fixed inputs and job scripts but does not submit
  them; only the queue manager resubmits tracked runs, and `resubmit_delay`
  is read but not applied. A standalone resubmission from `mace recover` is
  planned for 1.1.6.
- The FREQ templates are defined twice (`Crystal_d12/d12_constants.py` and
  `Crystal_d12/d12_calc_freq.py`); 1.1.6 keeps one.
- The consolidation proposed in `PATHFINDER-2026-10-08/03-unified-proposal.md`
  (U0-U8: dead code, one SLURM client, one site/CRYSTAL resolution, shared
  prompt validators, one deck-records toolkit, fast `.out` facts, library
  code out of `mace/gui`, workflow/queue control flow, CLI/GUI hygiene) is
  planned for 1.1.6.

### Testing

- 7752 tests (3322 in 1.1.4). On the final tree: 7683 passed (67 skipped,
  2 xfailed) on Python 3.12 with the test/ corpus, ase, plotly and selenium;
  6857 passed (891 skipped) on Python 3.12 without the corpus; 5841 passed
  (927 skipped) on Python 3.11 with CI's packages plus selenium; 6414 passed
  (786 skipped) on Python 3.12 with ase, spglib and phonopy but neither plotly
  nor selenium; the GUI smoke tests (`-m guismoke`) pass with the browsers
  required. Run in parallel groups on a busy workstation, a few browser and
  process tests (a keyboard activation, a pool worker check after a hangup
  or Ctrl+C, a local-run cancel) missed their deadline and passed when rerun
  alone. A test runner that starts pytest in the background of a script must
  give it the default Ctrl+C handling (a shell's background job ignores
  SIGINT, and the children inherit that), or the Ctrl+C tests fail. No test
  writes into the read-only test/ corpus (a WAL database there is read
  through a copy).
- The GUI's browser tests (`pytest -m gui`) drive headless Firefox through
  `mace_cli gui`, drawing WebGL in software (llvmpipe), never on a GPU. They
  are opt-in: `pip install -r requirements-gui-test.txt` (selenium,
  axe-selenium-python); without selenium they skip, as in CI, where the GUI's
  API and CLI tests run. Both CI jobs now stop after 30 minutes.
- Release gates through the real entry point: the access rules (token, Host,
  session path and cookie scope, POST header, no test route without the test
  hooks), the start-up rules (`--host`, `--token`, `MACE_GUI_TOKEN`,
  `MACE_GUI_PORT`, the URL file), the connection limits, SIGTERM and a
  hangup (also of a real terminal), writes that refuse symbolic links, the
  help, that plain commands import no GUI code, the Python children's
  environment, no pool worker left after SIGTERM, Ctrl+C, a hangup or
  SIGKILL of the server, and a deck written with a single pool worker.
- Accessibility: axe-core finds no serious or critical WCAG 2 A/AA violation
  on the main screens and dialogs in light and dark; a keyboard walkthrough,
  Esc and focus return on every dialog, reduced motion, and the tokens'
  contrast are tested.
- Wall time with CI's packages plus selenium (Python 3.11, nice 10): 4 min
  59 s before the GUI, 6 min 47 s after.
- The performance budgets stay opt-in (`MACE_GUI_BENCH=1 pytest -m guibench
  tests/gui/bench`). Measured by hand while the GUI was developed: over an ssh
  tunnel from a workstation to an MSU HPCC dev node (2026-10-04, the module
  Python 3.11.3 with plotly 6.1.2 and ase 3.25), API calls took 5-75 ms, a
  deck preview about 0.9 s, 3Dmol.js (0.5 MB) about 60 ms and plotly.js (4.6
  MB) about 100 ms, and the server stopped leaving nothing running; on a
  two-GPU workstation (GPU0 driving the display), `mace gui --open --gpu 1`
  moved Firefox 156's processes from GPU0 to GPU1 (nvidia-smi). The browser
  keys that only real key presses reach (Alt+Left/Right, F5 with a modified
  document, the developer-tool keys) are checked under Xvfb
  (`tests/gui/test_browser_keys_xvfb.py`, opt-in); a check on a desktop
  session is still to do.

*Fittingly, 1.1.5 shares its number with element 115, moscovium (Mc). Any
resemblance to a certain meteorite-powered Pack-a-Punch machine is purely
coincidental.*

## [1.1.4] - 2026-10-02

Everything since 1.1.3. **Band and phonon-dispersion paths made with the SeeK-path
library were written in SeeK-path's cell, not CRYSTAL's, since 1.0.0:**
C-centered (oC) and A-centered (oA) orthorhombic paths went through wrong
k-points, and triclinic, some monoclinic and non-standard orthorhombic paths had
wrong labels or segments. They are now converted to CRYSTAL's reciprocal basis;
regenerate affected BAND and phonon paths. Decks derived from a charged or
ionic parent now keep its `CHEMOD`, `CHARGED` and `DOPING`; before, SP and FREQ
children ran a different charge state. Also: SeeK-path-consistent static paths
for more lattices, in-plane band paths for slabs, smaller band-path ISS, HF-3c
and HFsol-3c configs write their MINIX/SOLMINIX basis, and several menu,
workflow-state and job-script fixes. Several of these change generated decks -
see Fixed below.

### Fixed

- **SeeK-path band and phonon-dispersion paths are written in CRYSTAL's
  reciprocal basis.** Since 1.0.0 the seekpath-library route (BAND `.d3`
  decks from `opt2d3`, phonon-dispersion paths in FREQ `.d12` decks from
  `opt2d12`, and the workflow steps that run them) copied SeeK-path's point
  coordinates unchanged. Those are fractions of the reciprocal vectors of
  SeeK-path's standardized primitive cell; BAND reads them as fractions of
  the reciprocal vectors of CRYSTAL's primitive cell (manual p.309), which is
  often a different cell. The points are now converted (f = N^-1 f_SeeK-path,
  with SeeK-path's cell = N times CRYSTAL's), keeping their labels and
  Cartesian positions; SeeK-path's own numbers are kept where every segment
  is already a symmetry image of the right one, so those decks are
  unchanged. Affected: **C-centered orthorhombic (oC)** and **A-centered
  orthorhombic (oA, groups 38-41)** cells, whose points and segments were
  wrong; **triclinic (aP)** cells, whose labels named other points;
  **primitive orthorhombic (oP)** cells of groups whose setting does not fix
  the axes (Pmmm, for one) when not given in a < b < c order; and any other
  cell CRYSTAL holds in a setting SeeK-path standardizes differently (in the
  test corpus a cubic cF1 cell given in a non-standard primitive cell, and
  two P2/m and P2_1/m cells whose Gamma-A and E-Z segments ran to another
  copy of A and E). Without the library, the static oA paths (added in this
  release cycle) are converted the same way and take the cell's own
  lattice-dependent parameter; the static paths of other lattices whose
  standard cell CRYSTAL's can differ from (aP, mP, mC, oP, oC, oF, oI) now
  warn. Band-structure and phonon-dispersion decks made for such cells
  should be regenerated.

- **A deck derived from a charged or ionic parent keeps its charge state.**
  `opt2d12` wrote SP, FREQ and other children without the parent's `CHEMOD`
  (per-atom starting shell charges, e.g. Li+ and FSI-), `CHARGED` and `DOPING`
  (net electrons added or removed) records, so the child of a +1 Li/FSI/EC
  cluster ran as a neutral one with a different electron count. They are now
  copied from the parent deck: `CHEMOD`/`CHARGED` after the basis set and
  `DOPING` in the SCF block, as in decks CRYSTAL23 runs. `CHEMOD` names atoms by
  position, so it is only copied when the child lists the same number of atoms;
  otherwise the deck is written with a warning to add it by hand.

- **An opt2d12 config with `"method": "HF"` and HF3C or HFSOL3C writes MINIX
  or SOLMINIX.** It kept the parent's basis: an external-basis parent gave
  `BASISSET / EXTERNAL (from original D12) / HF3C / END`, which is not input,
  and an internal one `BASISSET / POB-TZVP-REV2 / HF3C`. The manual defines
  HF-3c for a pure HF calculation in MINIX and HFSOL-3c in SOLMINIX (sec.
  5.3.1 p. 158, 5.4.1 p. 162); such decks now read `BASISSET / MINIX / HF3C /
  END` (layout p. 159, run on CRYSTAL23), with plain atomic numbers (47, not
  the external basis's 247).
- **HF-3c on a structure with metal atoms prints a caution** (opt2d12 and
  cif2d12). The manual (p. 158) says HF-3c was designed for organic systems,
  is carefully tested for molecular crystals, and that inorganic crystals and
  metals are less well tested; an AgCl HF3C/MINIX run diverged on CRYSTAL23.
  The deck is still written.
- **opt2d12 refuses a FREQ deck from a phonon-dispersion (SCELPHONO) FREQ
  run.** The child repeated the parent's SCELPHONO on the parent's geometry,
  which is already the SCELPHONO supercell (manual pp. 73-74), so the
  supercell would be expanded again.
- **Out-of-range answers are asked again** at the OPT convergence-level menu
  (it fell through to Custom), the FREQ IR-method menu (KeyError) and the FREQ
  NUMDERIV menu (ValueError on a non-number; 3 was written although NUMDERIV
  takes 1 or 2, manual p. 218).
- **The planner's BAND prompt asks for the total number of k-points along
  the path** (NSUB, manual p. 309), not "points per k-path segment". The
  default (10000) is unchanged.
- **The workflow state records the step that completed.** After an OPT whose
  planned next step was FREQ it could record another calculation's type.
- **The job script does not stage an empty `<job>.f20` for GUESSP;** it
  falls back to the job's own non-empty `.f9`, or drops GUESSP as before.
- **`guessp_restart` is documented as applied by the workflow engine only;**
  the executor's older step generator, which normal workflows no longer
  reach, now says it writes its decks without GUESSP.
- **Primitive hexagonal groups get SeeK-path's hP1/hP2 split.** Without the
  seekpath library every primitive trigonal and hexagonal group got the hP1
  path, whose last segment K-H_2 only hP1 groups need. SeeK-path (2.2.2,
  HPKOT) uses hP2 for 150, 152, 154, 156, 158, 164, 165 and 168-194; those
  groups (P321, P-3m1, P6_3/mmc, P6/mmm, ...) now get the hP2 path, one
  segment shorter. 143-145, 147, 149, 151, 153, 157, 159, 162 and 163 keep
  hP1.
- **The static face-centered cubic SeeK-path paths are SeeK-path's.** Without
  the seekpath library, the cF1 entry held Setyawan-Curtarolo's 9-segment
  path (Gamma-X-W-K-Gamma-L-U-W|L-K|U-X) and the cF2 entry a 9-segment path
  with a primed K' in a centrosymmetric group, both titled "SeeKPath". They
  now hold SeeK-path's cF1 (Gamma-X-U|K-Gamma-L-W-X-W_2, 7 segments) and cF2
  (Gamma-X-U|K-Gamma-L-W-X, 6 segments), and the cF1_noinv title gets the
  "|" between U and K that its segments already had.
- **Face-centered cubic groups get SeeK-path's cF1/cF2 split.** Without the
  seekpath library the split was reversed for 202, 203 and 225-228: Fm-3m,
  Fm-3c, Fd-3m (diamond, Si) and Fd-3c got cF1, and Fm-3 and Fd-3 got cF2.
  SeeK-path uses cF1 below 207 (196, 202, 203) and cF2 from 207 (209, 210,
  216, 219, 225-228), so Fm-3m and Fd-3m decks lose the X-W_2 segment and
  F23 decks gain it.
- **Body-centered cubic groups are all cI1.** The static lookup called 199,
  204, 206, 211, 214, 217 and 220 "cI2", a lattice SeeK-path (HPKOT) does not
  have, backed by table entries copied from cI1's. Those entries are removed
  and every body-centered cubic group is cI1. The decks are unchanged (the
  paths were identical).
- **The static rhombohedral SeeK-path paths are SeeK-path's.** Without the
  seekpath library, R-3, R-3m and R-3c (148, 166, 167) got Setyawan-
  Curtarolo's rhombohedral paths under a "SeeKPath" title, with points
  (B, B1, Q, P1, X, Z) SeeK-path does not have: 11 segments for hR1, 10 for
  hR2. They now get SeeK-path's hR1 path (Gamma-T-H_2|H_0-L-Gamma-S_0|S_2-F-
  Gamma, 7 segments) and hR2 path (Gamma-L-T-P_0|P_2-Gamma-F, 5 segments),
  with the same representative values for the lattice-dependent points
  H, S and P as the non-centrosymmetric entries.
- **Rhombohedral groups get SeeK-path's hR1/hR2 choice.** Without the
  seekpath library the static lookup took hR1 for c/a < sqrt(6) (hexagonal
  axes), so typical rhombohedral crystals (c/a 2.5-7; AgBr R-3c, Bi2Se3,
  corundum) got the hR2 path and flat cells got hR1. SeeK-path (HPKOT) takes
  hR1 for c/a > sqrt(3/2), i.e. a rhombohedral angle below 90 degrees, and
  hR2 otherwise; a cell given in rhombohedral axes is now read by its angle.
  Rhombohedral decks from outputs with c/a above sqrt(6) or below sqrt(3/2)
  change path.
- **A-centered orthorhombic groups get SeeK-path's oA paths.** Without the
  seekpath library, Amm2, Aem2, Ama2 and Aea2 (38-41) fell through to the
  primitive orthorhombic oP1 path. They now get SeeK-path's oA1 (b < c) or
  oA2 (b > c) path with its primed copy, and the title marks the jumps
  C_0|SIGMA_0 and A_0|E_0 (F_0|DELTA_0, B_0|G_0) that the segments make.
- **`opt2d3` reads a slab's symmetry.** A SLAB output names its symmetry as
  a plane group and the corresponding space group ("TWO-SIDED PLANE GROUP
  N. 80 : P 6/M M M", "CORRESPONDING SPACE GROUP N. 191"; manual App. A.2,
  p.421), not on the "SPACE GROUP ... :" line `opt2d3` reads, so every slab
  was taken as space group 1 (P1) and, without the seekpath library, got the
  triclinic band path. It now takes the corresponding space group and the
  plane group's lattice letter.
- **A slab's band path stays in the plane of the slab.** `opt2d3` BAND wrote
  a 3D path for a SLAB parent: with the seekpath library, SeeK-path's path
  for the slab's 500 A box, e.g. graphene's hP2 path with nine segments, six
  of them at or towards kz = 1/2. A slab is periodic in a and b only, and
  BAND takes I3 = J3 = 0 in 2D (manual p.310, note 3). The path of the
  corresponding space group is now cut to its segments in the kz = 0 plane
  (graphene: Gamma-M-K-Gamma), in coordinate and label mode, and the title
  names the segments kept. For a slab the seekpath library is not used:
  it standardizes the 3D box and can turn the vacuum axis into a or b (an
  oblique slab becomes mP with b along the normal), so its coordinates would
  not be in CRYSTAL's slab cell; the static path is. If no segment of a path
  lies in the plane (the literature path of a C-centered rectangular slab),
  the in-plane SeeK-path path is written instead. The `opt2d12` SeeK-path
  phonon band path of a slab is cut the same way. Polymers (1D) are not
  changed. Untested on CRYSTAL.
- **SeeK-path band paths keep ISS within three digits.** With the seekpath
  library, ISS was the largest denominator of every SeeK-path point - also
  points not on the path - after rounding lattice-dependent coordinates to a
  denominator below 10**6, so decks carried ISS up to ~10**6 (median about
  6*10**4 over representative cells of all 230 groups). CRYSTAL runs them
  (the k-point density depends only on NSUB, manual p.309 note 4), but
  fort.25 writes the segment ends with format 6I3 (App. D, p.447), so ends
  of 1000 and more printed as asterisks. Now the points SeeK-path fixes by
  fractions (its points.txt) set ISS to their lowest common denominator,
  from 4 (cubic P 4, cI 4, cF 8, hexagonal 6, previously 16 or 18), and are
  written exactly; lattice-dependent points are written as k/ISS with ISS at
  most 999, a multiple of that denominator, choosing the ISS that moves them
  least, provided no point moves by more than half the k-point spacing of
  the path at 10000 points (the workflow's NSUB). Over all 230 groups and
  about 4800 representative cells the largest move is 1.2e-4 of a reciprocal
  lattice vector (3.7e-4 1/A with 2*pi, at most 0.42 of the k-point spacing
  at 10000 points); MACE's old round trip of these decks already moved points
  by up to 1.2e-4. Whether a point is lattice-dependent now comes from
  SeeK-path's definitions, not from its value being within 1e-6 of a small
  fraction. NSUB is unchanged. The `opt2d12` SeeK-path phonon path now
  starts from the library's ISS, as `opt2d3` does, instead of rescaling the
  library's points to the band shrink. Untested on CRYSTAL.

## [1.1.3] - 2026-10-01

Everything since 1.1.2. Band paths are written with exact high-symmetry points
and follow SeeK-path for more lattices (cubic, hexagonal, monoclinic, centered);
`opt2d12` FREQ decks repeat their FREQ parent's FREQCALC settings and
Hartree-Fock parents stay Hartree-Fock; the DOSS example configs now give decks
CRYSTAL runs; a P1 deck from a CIF without symmetry operators holds the right
atoms; and workflow steps can optionally restart their SCF from the previous
step's density matrix (off by default). Several of these change the content of
generated decks - see Fixed below.

### Added

- **Workflow steps can restart their SCF from the previous step's density
  matrix (opt-in).** With `"guessp_restart": true` under `execution_settings`
  in the workflow plan JSON, an SP, FREQ or OPTn step the engine builds from a
  completed OPT or SP gets the predecessor's `.f9` copied to `<job>.f20` in its
  directory (the job script stages that as fort.20) and a GUESSP record before
  SCFDIR. The manual (GUESSP, pp. 114-115) requires the same symmetry and the
  same atoms, basis functions and shells in the same order, and CRYSTAL does
  not check this, so the engine only does it when both decks have the same
  symmetry records, atom list, basis set and spin treatment (UHF/ROHF/SPIN);
  the lattice, coordinates and functional may differ. It is not done when the
  follow-up sets ATOMSPIN (p. 99: "does not work with GUESSP") or either deck
  is two-component, when the predecessor left no non-empty `.f9`, or when the
  step's job script predates the `.f20` staging; the engine prints why. The
  setting is off by default, and without it the engine writes exactly what it
  wrote before. Untested on CRYSTAL.

### Fixed

- **The Crystal_d12 README says where the phonon SeeK-path path gets its
  inversion.** It still said the phonon path had only the space-group number,
  and kept a known issue about the output text being taken for a file name.
  Since that fix the parent output reaches the SeeK-path helpers as a file
  (`d12_calc_freq._output_as_file`), and inversion and cell parameters come
  from it (`d3_kpoints.detect_inversion_from_crystal_output`); the number is
  the fallback when there is no output.
- **`opt2d12 --config-file` writes Hartree-Fock decks from an HF config.** A
  config with `"method": "HF"` and a null functional (the form
  `quick_screen.json` had) stopped every file with `argument of type
  'NoneType' is not iterable`, and one naming only the method, or its flavor
  as `hf_method`, kept the parent's DFT functional. opt2d12 now reads the
  flavor from `hf_method` or `functional`, RHF when neither names one, as
  cif2d12 does, and refuses an HF config that names a DFT functional.
- **"Proceed with spglib space group" writes spglib's cell with spglib's
  group.** When a CIF's group and spglib's disagree, option 2 declared
  spglib's group number but kept the CIF's cell and atoms, so CRYSTAL applied
  the group's standard operators to a cell in another setting: rock salt given
  as its 60-degree primitive cell became an Fm-3m cube with a = 3.99 A instead
  of 5.64 A. The deck now carries spglib's standardized conventional cell and
  its asymmetric unit, in the origin the deck declares for two-origin groups.
- **A P1 deck from an Fd-3m CIF holds the structure the CIF describes.** A
  CIF with no symmetry-operator loop is expanded by ASE in origin choice 1, but
  the converter reads Fd-3m atoms at (1/8, 1/8, 1/8) as origin choice 2 - as
  the symmetrized deck ("0 0 0") and CRYSTAL's own diamond example do. The
  P1 deck of such a diamond CIF therefore held 16 atoms 1.26 A apart instead
  of diamond's 8. The P1 expansion now uses the origin the symmetrized deck
  declares for Fd-3m, and for any two-origin group when origin_setting is
  STANDARD. CIFs that list their operators are expanded as before.
- **P2/c and P2_1/c keep the primitive monoclinic SeeK-path.** Without the
  seekpath library, a band or phonon path built from a CRYSTAL output reads
  the cell parameters, and with them space groups 13 and 14 were sent to the
  C-centered monoclinic path (mS1), whose points belong to another Brillouin
  zone; the split between P and C was taken at group 11. They now get mP1,
  as SeeK-path does and as they already did without cell parameters.
- **Simple cubic groups 207-230 get SeeK-path's cP2 path.** Without the
  seekpath library every primitive cubic group took the cP1 path, whose last
  segment M-X_1 belongs to groups 195-206 only. P432, P-43m, Pm-3m and the
  other groups 207-230 now take cP2 (cP2_noinv when non-centrosymmetric), as
  SeeK-path does.
- **Static SeeK-path titles name the segments the deck writes.** Without the
  seekpath library, the title labels of the P-1 (aP3), C-centered monoclinic
  (mS1: C2, Cm, C2/m, ...), F-orthorhombic (oF1-oF3: Fmmm, Fddd), I-tetragonal
  (tI1/tI2: I4/m, I4/mmm, ...) and non-centrosymmetric F-cubic (F-43m, F23, ...)
  paths ran on where SeeK-path's path jumps, so the title had more edges than
  the deck had segments and band-plot nodes were misnamed. They now follow
  SeeK-path's path, with a "|" at each jump. The P-1 aP2 path also called
  (1/2, 1/2, 0) and (1/2, 0, 1/2) N and M; SeeK-path names them V and U. The
  coordinates are unchanged.
- **Static SeeK-path titles of the other centered lattices name their
  segments too.** The same run-on labels were in the paths used for the
  non-centrosymmetric rhombohedral groups (R3, R32, R3m, R3c: hR1/hR2),
  body-centered tetragonal groups (I4, I-4, I422, I4mm, I-42d, ...: tI1/tI2)
  and face-centered orthorhombic groups (F222, Fmm2, Fdd2: oF1/oF3), and for
  every body-centered orthorhombic group (Immm, Imm2, I222, ...: oI1) and
  every C-centered orthorhombic group (Cmcm, Cmc2_1, Cmmm, ...: oS1). They
  now follow SeeK-path's path with a "|" at each jump; the tI2 path without
  inversion also named its point G as Gamma. The coordinates are unchanged.
- **Hexagonal K and H points are written exactly in BAND and phonon paths.**
  Coordinate paths are written in units of 1/shrink, and K = (1/3, 1/3, 0) is
  only a whole number of steps when the shrink is a multiple of 3; at the usual
  16 it was rounded to 5/16. Quarter points had the same problem at a shrink
  such as 6 or 990. The shrink is now raised to the next common multiple of the
  points' denominators (16 becomes 18 for hexagonal paths), the way a shrink
  too small for the path was already raised.
- **The DOSS example configs give decks CRYSTAL runs.** Decks written from
  `doss_total_only.json`, `doss_orbital_projections.json` and
  `doss_element_orbital_auto.json` (and any DOSS deck with "print integrated
  DOS" switched on) stopped in CRYSTAL with "DOSS FORMAT ERROR IN INPUT DECK":
  the printing-option count was 1 but no printing-option record followed. That
  record (`105 -1`) is now written. `doss_total_only.json` also asked for bands
  0 to 999, which CRYSTAL refuses ("BAND RANGE NOT ALLOWED"); a band range is
  now kept within 1 to the number of AOs. The examples now write `DOSS.DAT`
  (they asked for no file at all), and "Total DOS only" gives a total DOS
  instead of per-orbital projections. All three examples were run on HPCC.
- **An `opt2d12` phonon deck with a SeeK-path band path is written.** The
  SeeK-path helpers read a `.out` file, but `opt2d12` passed them the parent's
  output text, so every such deck stopped with `OSError: File name too long`.
  They now read the same text from a temporary file, and the path matches what
  the `.out` itself gives.
- **The DAT-file and population-analysis processors open the materials
  database.** `process_calculation_dat_files` and
  `process_material_population_analysis` (`mace/utils`) imported
  `MaterialDatabase` from `material_database`, a module that no longer exists,
  and stopped with ModuleNotFoundError; they now use `mace.database.materials`.
- **Space-group tables match the International Tables and CRYSTAL.** Pbcn
  (60) was listed with two origin choices, and its "alternate origin" code
  `0 1 0` is CRYSTAL's rhombohedral-axes flag: a cif2d12 deck asked for that
  origin carried it. It now writes `0 0 0` for Pbcn with any origin option;
  every other group's origin handling is unchanged, and the list is now
  exactly the ITA two-origin groups. The spelling table maps group 90 as
  CRYSTAL prints it ("P 4 21 2", not "P 42 1 2"), maps P-42c to 112 (not
  114), and has no duplicate key; every lookup already reached the right
  number through another table, so no other deck changes.
- **`copy_dependencies.py` copies the files it lists.** Its list still used
  the names from before the scripts moved into the `mace` package, and it
  looked for `Crystal_d12/` and `Crystal_d3/` inside `mace/`, so it reported
  most files missing and copied none of the converters. It now copies each
  file from where it lives, under its current name, together with the
  modules the copied converters import (`d12_config.py`, `menu_nav.py`,
  `spglib_compat.py`, `seekpath_interface.py`). It no longer tries to copy
  `recovery_config.yaml`, which the recovery engine reads from `mace/config/`.
- **Documentation examples and paths that did not work.** The D3 batch
  example passed `--config-file doss_orbital_projections.json`, which is
  opened as given and not looked up in `example_configs/`; it now gives the
  full `$MACE_HOME/Crystal_d3/example_configs/` path. DOCUMENTATION.md listed
  three of the six `--calc-type` values of `opt2d3`, pointed at
  `Band_Alignment/` and `Post_Processing_Scripts/` without their `code/`
  prefix, and referred to a `d12creation.py` that no longer exists. It now
  also documents applying one `opt2d12` template to many structures.
- **The standard-library `queue` module is no longer replaced by
  `mace/queue`.** `mace_cli` puts `mace/` itself on `sys.path`, where the
  `mace/queue` package shadows the standard module, so the first bare
  `import queue` in the process registered `mace/queue` under that name.
  Removing an unused `import queue` from `mace/workflow/executor.py` did not
  stop this, because `pyarrow`, which `mace.database` imports, runs its own
  `import queue`. Importing `mace` now loads the standard `queue` with
  `mace/` hidden from `sys.path`, before anything else can, and `mace_cli`
  imports the job-queue package as `mace.queue` rather than `queue`.
- **A cif2d12 SLAB deck lists each atom once, at its own height.** A SLAB
  record takes the symmetry-unique atoms, with z in Angstrom from the layer
  group's origin (CRYSTAL23 manual p. 21). Handed a whole cell - which batch
  mode keeps when spglib and the CIF disagree, or with `write_only_unique`
  off - the converter wrote every atom with z = c times fractional z: a Bi2
  bilayer (P-3m1, layer group 72) got its inversion image as a second Bi at
  z = 19.10 A instead of -0.90 A, and CRYSTAL built a four-atom slab 38 A
  thick. Atoms the layer group generates from an earlier one are now left
  out, using the CIF's own operators when they match the named group's in
  number and kind (otherwise the atoms are written as given, with a
  warning). For a layer group with an operation that reverses z, the whole
  layer is moved by one offset so that the operation's plane - at
  fractional z = 0 or 1/2 in the cell - is at z = 0; the layer stays in one
  piece, even when it is centered on z = 1/2 or straddles z = 0/1, and every
  z distance in it is the cell's own. A polar-layer-group slab keeps its
  c times z heights; when it straddles z = 0/1, the atoms just below z = 1
  are written one cell lower, next to the rest of the layer.
- **The static orthorhombic-P path without inversion names its segments
  in order.** Without the seekpath library, `oP1_noinv` listed the labels
  Y-T before X-U (and Y'-T' before X'-U') while its coordinates, like
  SeeK-path's path, run X-U first, so those segments carried each other's
  labels.
- **Non-centrosymmetric structures with a 2-fold axis get the
  non-centrosymmetric SeeK-path path.** Without the seekpath library, the
  band and phonon paths read inversion from the CRYSTAL output, and the
  SYMMOPS check matched a run of six matrix elements that a 2-fold axis
  along z (and, in the hexagonal frame, an in-plane 2-fold or -6) shares
  with the inversion. Every such group - in the
  primitive lattices 16-19, 25-34, 75-78, 81, 89-96, 99-106, 111-118,
  149-154, 168-174, 177-190, 195, 198, 207, 208, 212, 213, 215 and 218 - was
  called centrosymmetric and got the path without primed points. An operator
  now counts as an inversion only when its whole rotation part is -I, and a
  SYMMOPS table without one means no inversion. The space-group-number check
  no longer reads a digit out of a symbol ("P N N 2" as group 2).
- **The static SeeK-path band paths go through the zone-boundary points.**
  Without the seekpath library, band and phonon paths come from a table in
  `d3_kpoints` that defines several lattices twice; the definitions that
  win had every coordinate doubled (body-centered cubic x4, hexagonal x6), so
  simple cubic X was (0, 1, 0) - a reciprocal lattice vector, the same point
  as Gamma - where SeeK-path and CRYSTAL23 (manual Tables 14.1-14.2,
  pp. 311-312) have (0, 1/2, 0). The paths for aP, cP, cI, hP, mP, oP, tP,
  tI and mC2 (and the aP and mC2 non-centrosymmetric ones) now use SeeK-path's
  coordinates: e.g. every centrosymmetric primitive cubic, tetragonal,
  orthorhombic, monoclinic and hexagonal group, Im-3m, Ia-3d and I4/mmm.
- **`get_auto_phonon_path` returns a path in the labels and vectors
  formats.** Both stopped with NameError, because `d12_calc_freq` checked
  `get_crystal_system_from_space_group` without importing it from
  `d3_kpoints`. Nothing calls the function today, so no deck changes.
- **An `opt2d12` SeeK-path phonon deck's title names the path it writes.**
  The title took its labels from the space-group number alone, while the
  BANDS path is chosen from the parent's output, whose lattice parameters
  and symmetry operators pick the SeeK-path variant. For AgBr in R-3c the
  title listed an 11-segment path over the 10-segment one written, and
  every such title called its path "default". The title now reads the same
  output, so it lists the written path's labels and says "SeeKPath (w.I)",
  "SeeKPath (no.I)" or "Literature" as the path was chosen.
- **The workflow planner loads without `d12_constants`.** When
  `d12_constants` or `DummyFileCreator` could not be imported, the planner
  reported it with `ui.warn` before `ui` was defined, so the import stopped
  with NameError instead of using its built-in fallbacks. `ui` is now set up
  first.
- **A mistyped answer to a D3 yes/no question is asked again.** In the
  interactive `opt2d3` setup, an answer other than y/yes/n/no (or Enter)
  counted as "no", even where the default was "yes", and the question moved
  on. It now prints "Please respond with 'yes' or 'no' (or 'y' or 'n')." and
  asks again, as the D12 tools do. Every recognized answer, and end of input,
  behave as before.
- **An `opt2d12` FREQ deck repeats its FREQ parent's FREQCALC settings.** Only
  NUMDERIV was read from the parent's FREQCALC block, so the child of a parent
  that asked for IR intensities (`INTENS / INTCPHF / ENDCPHF`), Raman
  intensities or IR/Raman spectra was written with `NOINTENS`. The whole block
  is read now, and a FREQ child written without questions (`--non-interactive`,
  or a `--config-file` without FREQ settings of its own) repeats it; RESTART
  is not repeated, as it restarts the parent's own run from the FREQINFO.DAT
  that run wrote (manual sec. 8.2, p. 219). A record the writer never writes
  (e.g. NOUSESYMM) is reported in the parse as `freq_unparsed`. The
  interactive questions still default only NUMDERIV to the parent's.
- **A Hartree-Fock parent deck reads back as Hartree-Fock.** `CrystalInputParser`
  gave no functional for a deck with no Hamiltonian keyword (CRYSTAL's default
  RHF, manual p. 123) or with `HF3C` / `HFSOL3C` (manual sec. 5.3.1, pp. 158,
  162), so such a deck was written back with a DFT block. It now reads them as
  the functionals `RHF`, `HF3C` and `HFSOL3C` (method HF), as the deck writer
  takes them, and reads SMEAR under the writer's `smearing` key as well. For
  `opt2d12`: an RHF parent's `.out` already names RHF, and SMEAR still comes
  from the `.out`, so those children are unchanged; an HF-3c or HFsol-3c
  parent's child now gets that functional, where the `.out` could give RHF and
  drop the correction (untested: no HF-3c run to hand).
- **A deck MACE wrote reads back as the same deck.** `CrystalInputParser`,
  which reads the parent deck for `opt2d12`, kept coordinates only as floats
  (`1.250000000000E-01` came back as `0.125`) and stripped the spacing of an
  EXTERNAL basis' records. It now also keeps the title, each atom record's
  coordinates and last word, and the EXTERNAL basis records as written.
  Parsing a deck and writing it back with `opt2d12`'s writer now gives the
  same deck byte for byte for all 811 decks `cif2d12` writes in the new
  round-trip test (CRYSTAL, SLAB, POLYMER and MOLECULE; internal and EXTERNAL
  basis; SP, OPT and FREQ; 80 before), for `opt2d12`'s own children of the
  real runs in `tests/data`, and for 324 of the 401 real decks of the test
  corpus apart from their title, which is `./<name>` (an older MACE took it
  from the path). The other 77 hold records the writer never writes, or in
  another order: they were edited after MACE wrote them (SUPERCEL, a RESTART
  that recovery adds to OPTGEOM, GUESSP below MAXCYCLE, RAMSPEC ahead of
  IRSPEC) or written by hand. The new keys never reach `opt2d12`'s settings,
  so its decks are unchanged.

### Removed

- Five standalone scripts that nothing runs, imports or documents:
  `mace/workflow/status.py`, `mace/workflow/callback.py` and
  `mace/workflow/check_workflows.py` (superseded by `mace status` and the
  queue manager's completion callback), `mace/utils/scf_settings_extractor.py`
  and `mace/utils/analyze_script_dependencies.py`.
- The phonon fallback k-path tables `SPACEGROUP_TO_PATH` and
  `HIGH_SYMMETRY_PATHS` (`Crystal_d12/d12_constants.py`). They gave one path
  per crystal system whatever the lattice centering (C2/m the primitive
  monoclinic path, Fmmm and Immm the primitive orthorhombic one, I4/mmm the
  primitive tetragonal one, R-3m the hexagonal one), and their only reader was
  a fallback in `d12_calc_freq.get_auto_phonon_path`, a function nothing in
  MACE calls. Its coordinate paths now all come from the centering-aware
  band-path code that its "vectors" format already used. No deck changes.

### Testing

- Tests for paths that had none: the workflow plan JSON from planner to
  executor to a job's callback (save, load, validation, queue settings),
  `mace status` and `mace completion` on real outputs (categories, zombie
  jobs, moving files, the workflow's isolated database), 'b' to go back
  through the real OPT/SP/FREQ and opt2d3 questionnaires, and the timeout
  RESTART recovery on the committed PbTiO3 and Ag2Br3 runs, end to end with
  sbatch/squeue/sacct/scontrol stand-ins on PATH.
- The corpus sweeps that skip without `test/` (deck geometry rebuild,
  EXTERNAL basis blocks, TESTPDIM insertion, outputs typed by their deck,
  settings extraction, aggregation keys, timeout RESTART) also run over the
  real decks and outputs committed under `tests/data`, which now include the
  two original ECP OPT decks (`tests/data/samples/ecp_decks`).
- Three known bugs are pinned as expected failures: at the OPT
  convergence-level menu an answer outside 1-4 is taken as Custom, and in
  the FREQ menu an answer outside 1-3 for the IR method (KeyError) or a
  non-number for NUMDERIV (ValueError) stops the questionnaire.
- CI has a second job with ase, spglib and seekpath installed
  (`requirements-optional.txt`), so the tests that need them run there.

## [1.1.2] - 2026-09-29

Everything since 1.1.1. Decks MACE derives from a finished calculation now keep
the parent's method and settings; `opt2d12` prompts default to the parent's
values and a saved template can be applied to many structures at once; OPT jobs
that run out of walltime continue from their last step (`OPTGEOM RESTART`, with
a fallback when the restart itself aborts); calculation types are read from a
file's records rather than its name; and a series of CRYSTAL23-conformance,
job-script and recovery fixes. Several of these change the content of generated
decks - see Fixed and Changed below.

Conformance work traced from a user report: every hybrid lead perovskite died
with `ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL` under HSE-3c, on cells that
are neutral. Auditing that one bug against the CRYSTAL23 manual turned up a
family of others. Every behavioral change below was run through
CRYSTAL/23-intel-2023a on real hardware, not just reasoned from the manual.

### Fixed

- **A POLYMER or SLAB deck made from a `.out` alone keeps the parent's rod or
  layer group.** `opt2d12` wrote whatever space group it had as the group
  record. With the parent `.d12` present that was the parent's own record, so
  every existing child was right (checked: all 162 SP/OPT/FREQ children of the
  27 slab decks in `test/`). From the `.out` alone it was not: a POLYMER
  output's "CORRESPONDING SPACE GROUP : P 65" line was read as 3D space group
  170 and written as the rod group, which CRYSTAL refuses (rod groups run
  1-99, layer groups 1-80), and one inside the range names a different group.
  A SLAB fell back to P1 instead of its layer group. The child now takes the
  group from the parent deck or the `.out`'s own "TWO-SIDED PLANE GROUP N." /
  "POLYMER GROUP N." line, never from a 3D analysis, and a group outside the
  range is refused before any deck is written.
- **A SLAB deck's k-point mesh keeps its layer group's symmetry.** A mesh
  `opt2d12` generates (a child made from the `.out` alone) or `cif2d12` writes
  comes from the conventional a and b, and was only equalized for 3D decks. In
  a centered-rectangular layer group the two primitive directions are
  equivalent, so graphene in layer group 47 got `SHRINK 0 36 / 18 10 1` and
  CRYSTAL stopped with "SHRINK BREAKS SYMMETRY". Square, hexagonal and
  centered-rectangular layer groups (CRYSTAL numbers the last 10, 13, 16, 22,
  26, 35, 36, 47, 48) now get one factor for both in-plane directions; oblique
  and primitive rectangular groups and POLYMER meshes are unchanged, and a
  child made with the parent `.d12` still takes the parent's own SHRINK.
  Checked with real SCF runs of CRYSTAL/23 on HPCC: layer group 47 from
  `opt2d12` and from `cif2d12` (refused before, converged now), 37 (18 10 1
  kept) and 80, and rod groups 28 and 51.
- **The example configs load in `mace convert --batch --options_file`.** None
  of the eight `Crystal_d12/example_configs/*.json` did: they wrap their
  settings in `{"version", "type", "configuration"}` and spell them the
  opt2d12 way (`functional`, `dispersion`, `scf_settings`), and every run
  stopped with `KeyError: 'dimensionality'`. The loader now unwraps the file
  and maps those names onto cif2d12's. What a config leaves out gets the
  interactive defaults (3D `CRYSTAL`, the CIF's space group written as its
  asymmetric unit, INTERNAL basis), and a file that cannot describe a deck
  (no functional, an unknown method, calculation type or symmetry setting)
  is refused with the reason and a non-zero exit. Each config was converted
  from real CIFs and checked by CRYSTAL23 (`mace preflight`).
  A flat options file saved by `--save_options` or the workflow planner
  writes the same decks as before - checked byte for byte on all 168 corpus
  CIFs with 18 such files - with one deliberate exception: a lowercase
  `"calculation_type": "opt"` used to fail the comparison with `"OPT"` and
  silently write a single point with `_opt_` in its name; it is now read as
  OPT, so the deck gets its OPTGEOM block and the name reads `_OPT_`. The
  basis set is written exactly as the file spells it (`DEF2-MSVP` stays
  `DEF2-MSVP`), and a 3c method paired with another basis is written as
  asked with a warning: CRYSTAL23 runs HSE-3c on POB-TZVP-REV2, which the
  lead perovskites need because its internal def2-mSVP has no Pb.
  `phonon_bands.json` now names the 2x2x2 `SCELPHONO` supercell that
  `FREQCALC DISPERSION` needs (without it CRYSTAL23 stops with
  `MAKE SUPERCELL WITH SCELPHONO`) and writes its phonon band path as
  k-point coordinates, since CRYSTAL23 rejects the label form there
  (`FORMAT ERROR IN FREQCALC INPUT DECK`).
- **`mace opt2d12 --config-file` reads the example configs too.** It also
  took the wrapper for the settings and crashed. `quick_screen.json` now names
  its HF flavor (`"functional": "RHF"`) so both converters read it the same
  way.
- **`d12_from_config.py` runs.** It passed the input file as an argument
  neither converter accepts, and did not recognize a CRYSTAL output that
  starts with MPI start-up lines. It also reports a file for which no deck
  was written as a failure (and exits non-zero) instead of "Successfully".
- **`mace convert --batch` exits non-zero when it writes nothing.** A refused
  options file, or a batch in which every CIF is refused (e.g. HF-3c's MINIX
  on a Pb structure), printed the reason and exited 0, so the workflow
  executor - which only checks the exit status - carried on with no decks.
- **The built-in `3c_composite` template uses def2-mSVP**, the basis PBEh-3c
  is defined on, instead of MINIX (HF-3c's). The basis now comes from the
  `basis_requirements` table rather than a second copy; a configuration that
  names a 3c method and no basis gets that basis.
- **cif2d12 FREQ decks follow the lattice centering for the phonon path.** The
  automatic phonon-dispersion path read the centering only from a CRYSTAL
  output, so every deck made from a CIF got the primitive path: Fd-3m
  diamond got the simple-cubic M-G-R-X-G where opt2d12 writes X-G-L-W-G for
  the same structure. cif2d12 now takes the letter from the space group
  symbol and writes the same path as opt2d12.
- **Phonon dispersion is refused off a 3D crystal, and no partial deck is
  left.** `opt2d12 --config-file phonon_bands.json` on a SLAB parent crashed
  in the title (no space group to take the path from) and left a 0-byte
  deck; on a MOLECULE it wrote `SCELPHONO` + `DISPERSION`, which CRYSTAL23
  stops with `SUPERCELL OPTION NOT ALLOWED FOR MOLECULES`. The 3x3
  `SCELPHONO` matrix both converters write is also rejected for a SLAB
  (`THE SUPERCELL IS OF ZERO VOLUME`; a slab takes 2x2), and the automatic
  path needs a 3D space group. Both converters now refuse dispersion for a
  MOLECULE, SLAB or POLYMER with the reason, and
  write every deck to a hidden part file that is renamed into place only when
  complete, so an abort or a crash never leaves a 0-byte or truncated deck.
- **`opt2d12 --output-dir` holds the deck when `--out-file` is an absolute
  path.** The new name was built from the parent's full path, and joining an
  absolute path onto the output directory discards the directory, so the
  deck was written next to the parent (into the test corpus, for one run).
- **`mace convert --batch` never reads the terminal.** On a structure whose
  atoms are all symmetry-unique (every P1 CIF), the symmetry step asked "Use
  the 'reduced' structure anyway?", hit end of input and reported "Error
  during symmetry analysis" before carrying on - 85 of the 168 corpus CIFs.
  Batch mode now takes symmetry from the options file (`symmetry_handling`,
  `write_only_unique`, `symmetry_tolerance`, now documented) and gives every
  question an interactive run would ask the prompt's own default, so the
  decks are byte-for-byte the ones written before, without the error. A CIF
  with no space group is skipped with a message instead of prompting, and a
  batch run without spglib says so once at the start.
- **A walltime-killed geometry optimization continues where it stopped.**
  When an OPT runs out of time after at least one optimization step, the
  recovery adds `RESTART` to the OPTGEOM block of the same deck, so the new job
  picks up the earlier steps from OPTINFO.DAT in the same scratch directory
  (CRYSTAL23 manual 7.4.3) and uses the killed run's last density matrix as
  the SCF guess. A job killed in its first SCF has no OPTINFO.DAT and starts
  over as before. The job script checks for OPTINFO.DAT again on the compute
  node and falls back to a fresh start if it is missing. The killed run's
  output is kept as `<job>.out.timeout1`, `.timeout2`, ...
- **A RESTART rerun that aborts continues as a fresh optimization from the
  best geometry reached.** Measured on HPCC: the rerun re-evaluates the
  lowest-energy point, sees almost no energy change, shrinks the trust radius
  to zero and dies in `MPI_Abort` (`ERROR **** BFGS_ **** PXK TOO SMALL` only
  in the scratch `fort.87`; SLURM says COMPLETED). This is now recognized
  (error type `opt_trust_radius_error`, up to `max_retries` 2) and the same
  deck is resubmitted without `RESTART`, starting from the lowest-energy
  point reached by the runs of that deck (energies from runs that started
  from another geometry are not comparable: CRYSTAL fixes its integral
  screening at the starting geometry). Only the cell parameters and atom coordinates
  change - read from the geometry CRYSTAL prints at every optimization point,
  in the deck's own frame - and the deck as it was is kept first as
  `<job>.d12.orig` (then `.orig2`, ...; a backup is never overwritten). The
  aborted output is kept as `<job>.out.optabort1`, ... A timed-out OPT whose
  scratch directory has lost OPTINFO.DAT starts from its best point the same
  way instead of from the beginning. Only 3D decks given by space group are
  rewritten; for anything else the job is not resubmitted.
- **CRYSTAL's own error is read from the scratch `fort.87`** when the `.out`
  ends without CRYSTAL's normal end, so such aborts get a real error type (or
  `crystal_error` with CRYSTAL's message) instead of `unknown_error`.
- **Per-error `max_retries` is enforced** across the job's recovery lineage,
  on top of the overall limit of recovery attempts (the built-in default for
  timeouts is now 2, as in `recovery_config.yaml`).
- **Job scripts written before the RESTART staging existed get it on
  recovery.** A workflow copies the job-script generator into
  `workflow_scripts/` when it is planned, so the job scripts of a workflow
  planned before this release run a RESTART deck without checking for
  OPTINFO.DAT. The timeout recovery adds the current staging to the copy of
  the script it resubmits (the original is left alone), unless the script
  handles `fort.20` in some other way. Such a workflow's later steps are still
  first submitted with the old scripts; re-plan it to update them.
- **An empty `<job>.f9` is no longer staged as the GUESSP guess.** A run
  CRYSTAL aborts during an optimization still copies its (empty) fort.9 back.
- **The doubled walltime stays within the queue's limit.** It is checked with
  SLURM when the job is resubmitted (`sbatch --test-only`, plus the MaxTime of
  the partition the job runs in): 7 days on the general partitions and
  mendoza_q, 14 days for jobs submitted with `-A mendoza_q_long`. A job is
  never moved to another partition or account. A job already at the limit is
  resubmitted at the same walltime only when it can get further than last
  time - an OPT continuing with RESTART or from its best point. Anything else
  (an SP or FREQ, an OPT killed in its first SCF) would time out again, so it
  is not resubmitted and the log says so. `max_walltime` is used only when
  SLURM cannot be reached.
- **Timeouts are recognized at all.** SLURM writes the time-limit notice to
  the job's `-o` log and leaves the `.out` cut off mid-line, so a timed-out job
  was classified as an unknown error and never recovered. SLURM's TIMEOUT
  state, or that notice in the job's log, now marks it as a timeout.

- **The calculation type is read from a file's records, not its name or
  title.** `_opt` matched `_optimized`, so the queue manager typed every
  chained deck (`X_opt_..._optimized_sp_..._optimized`) submitted without an
  explicit type as an OPT; on the corpus that was every SP and FREQ deck and
  most BAND, DOSS, CHARGE+POTENTIAL and TRANSPORT decks. Keywords in a deck or
  output title (`..._BULK_OPTGEOM_...`) no longer make a run look like an OPT,
  CHARGE+POTENTIAL decks (ECH3/POT3) are no longer typed BAND, the recovery
  detector no longer reports FREQ and properties outputs as SP, DOSS outputs
  are no longer stored with band-structure fields, and `mace analyze
  --filter-type` matches outputs again. A deck's own records decide (OPTGEOM,
  FREQCALC, the properties keywords); a name is only a tie-break.
- **Generated SP, FREQ and OPT2 decks keep the parent's settings.** Decks the
  workflow engine derives from a finished calculation were quietly changing
  the method. Found by regenerating decks from real parents through every
  engine path and comparing them keyword by keyword with the parent:
  - the functional was guessed by substring, so PBE0, PBEsol, PBESOL0, PBEh-3c
    and LC-wPBE parents became PBE, UHF became RHF, and the no-plan SP path
    turned HF parents into HSE06-D3;
  - an HSEsol-3c SP parent produced FREQ/SP/OPT2 decks labeled `HSESOL3C-D3`
    (3c methods already contain D3 and gCP), or crashed before writing one;
  - the two-number `SHRINK IS ISP` form, the most common in practice, was
    regenerated from the cell (`5 10` became `7 14`, some meshes coarser), and
    anisotropic meshes such as `30 30 10` were flattened;
  - FMIXING, MAXCYCLE, BROYDEN, LEVSHIFT, SPINLOCK and BIPOSIZE/EXCHSIZE were
    reset to defaults, and OPT2 decks lost MAXTRADIUS or gained tolerances the
    parent never set;
  - a plan's FREQ tolerances and SP overrides were never applied;
  - with no plan, SP decks reset tolerances to Standard, added or dropped D3,
    and replaced the parent's own EXTERNAL basis with the library copy.

  The tolerance, basis, D3, SCF and SPINLOCK prompts now default to the
  parent's value; an explicit answer or plan setting still wins.
  A saved `--config-file` applied to a different material still gets a
  k-mesh from its own cell.

- **`opt2d12` prompts default to the parent's settings.** Pressing Enter now
  keeps what the parent had: OPT type and convergence (including tolerances
  that match no preset), MAXTRADIUS, level shifting, HISTDIIS, GUESSP, the
  DFT grid (now actually asked), CVOLOPT, and ITATOCEL/INTREDUN types.
  Answering "yes" to level shifting on a parent without it now takes effect,
  and a "no" to MAXTRADIUS survives `--save-options`/`--config-file`.
- **Parent functionals are kept exactly or flagged.** Functionals written as
  EXCHANGE/CORRELAT/HYBRID records were reduced to the exchange name, so a
  PBE0 written that way became PBE. Those records, range-separation records
  (SR-OMEGA, SR-HYB, ...), LSRSH-PBE's value line and a separate DFTD3 or
  GRIMME block are now carried over verbatim. A parent functional CRYSTAL23
  does not accept (e.g. B2PLYP, SCAN-D3) now prompts for a replacement, with
  HSE06 as the default; paths that cannot ask print a warning.
- **Deck titles are no longer read as keywords.** A title such as
  `..._BULK_OPTGEOM_...` made an SP parent look like an OPT and moved its SCF
  settings into the child's OPTGEOM block.
- **An OPT in an SP-first plan is built from the SP.** The engine's CIF
  re-conversion for this case passed flags NewCifToD12 has never accepted
  and always failed; it has been removed. An OPT planned after BAND, DOSS, TRANSPORT or
  CHARGE+POTENTIAL is built from the last OPT or SP; after TRANSPORT and
  CHARGE+POTENTIAL the workflow used to stop.
- **A FREQ parent's NUMDERIV is kept** in a follow-up FREQ deck, and is the
  default at the NUMDERIV prompt.

- **Basis-set coverage is measured, not assumed.** CRYSTAL counts neutrality as
  basis-set shell charges against nuclear charge, and its internal def2-mSVP
  library is broken for Pb. MACE emitted those decks silently because the
  compatibility check guarded on `basis_set in INTERNAL_BASIS_SETS` and the 3c
  sets have no entry there, so the element loop never ran. Coverage for every
  element 1-99 across all 12 internal basis sets was measured by running
  CRYSTAL23 on one-atom cells; raw results and the scanner live in
  `tests/basis_coverage/`. The hand-written ranges were wrong in the dangerous
  direction - POB-TZVP-REV2 claimed He, Ne, Ar and Tc. External sets are now
  checked for the file the run would actually read, since `read_basis_file`
  returns `""` for a missing one and silently drops that basis block.
- **An EXTERNAL basis is no longer truncated.** The basis block is terminated by
  its own `99 0` record, but the parser matched that as a *substring*. Basis and
  ECP data are columns of numbers and `99 0` occurs inside them constantly:
  lead's ECP line `12.296303 281.285499 0` ends `...499 0`, so the parser stopped
  mid-ECP and dropped Pb and every element after it. CRYSTAL rejected the result
  with `ERROR **** INPBAS **** FORMAT ERROR IN INPUT DECK`. Measured over the
  corpus, 13 of 106 EXTERNAL decks truncated this way, every one Pb-bearing;
  all 106 now extract byte-for-byte. This is why OPT steps succeeded while every
  SP failed - OPT decks come from a CIF, SP decks are regenerated from the OPT's
  D12 through this parser.
- **Deck-breaking keyword errors.** `functional_keyword_map` rewrote three
  correct CRYSTAL keywords into strings that appear nowhere in the manual: VBH,
  PWGGA and WCGGA became VBHLYP, PW91GGA and WCGGAPBE. BROYDEN was written bare,
  without the W0/IMIX/ISTART record it requires. `OPT_TYPES` offered INTONLY and
  ITATOCELL, neither a CRYSTAL keyword; both are now ATOMONLY. The planner
  emitted prose spellings (HF-3C, M06-2X) where the input keywords are HF3C and
  M062X, offered "HF" which is not a Hamiltonian, and appended `-D3` to any
  functional rather than the ten the manual parametrizes.
- **Silently wrong geometry.** Rhombohedral cells written with IFHR=1 emitted
  `a c` where the manual requires `a alpha`, so alpha was read as 5.13 degrees
  instead of 55.29 - a collapsed cell, no error raised. SLAB decks always wrote
  three cell parameters instead of the minimal set, so CRYSTAL consumed `b` as
  the atom count.
- **SLAB and POLYMER group numbers.** The record after SLAB is a *layer* group
  (Appendix A.2, IGR 1-80) and after POLYMER a *rod* group (A.3, IGR 1-99), but
  the converter wrote the 3D space-group number into both. A hexagonal slab
  (space group 191) made CRYSTAL MPI_Abort having written no `fort.87` at all,
  so MACE's own error classification saw nothing. POLYMER failed more quietly,
  since a wrong number is usually inside 1-99 and simply builds a different
  chain. The reverse map is deliberately partial - 45 of 230 space groups for
  layer groups, 75 for rod groups - and refuses rather than guesses where the
  appendix row is not in the International Tables first setting or where several
  candidate orientations exist.
- **Slabs whose symmetry axis is off the in-plane origin are refused.** Every
  layer group the map can produce except P1 carries a rotation axis along z that
  CRYSTAL places at the in-plane origin before expanding the asymmetric unit
  about it. A structure whose axis sits elsewhere expands into a different slab
  and runs to completion looking correct. This is a necessary condition, not a
  sufficient one; `mace preflight` remains the authoritative check.
- **Silently wrong properties.** Four BAND fallback paths wrote fractional
  k-points where CRYSTAL expects integers in units of 1/ISS, aborting the read or
  collapsing the path to GAMMA. IRSPEC was suppressed whenever no dielectric
  tensor was supplied, so IR spectra were never generated at all. SCELPHONO was
  written inside FREQCALC rather than the geometry block, and with three values
  where a nine-real expansion matrix is required. POTC's ICA was off by one.
- **HF plus an external basis never closed the geometry block**, so every
  RHF/UHF deck built from a CIF aborted immediately.
- **The workflow dropped a configured node exclusion.** The planner collects one
  and writes it into the scripts it generates, but Phase 0 regenerates every
  script from the plan at execution time and discarded it - nothing in the
  resource mapping matched `--exclude`. Workflow jobs therefore ran with no
  exclusion while `mace submit`, which honors the same setting, kept it. Only an
  explicitly configured exclusion is written; with none, the template is
  unchanged.
- **Unattended runs no longer die on a prompt.** Three `stdin.isatty()` sites
  were unguarded, including the missing-element prompt - which, now that the
  compatibility check actually fires, would have hit `EOFError` mid-sweep. The
  batch symmetry-mismatch path likewise reaches an explicit logged decision
  instead of surfacing as "Error during symmetry analysis".
- **Documentation version drift.** The README banner still said 1.1.0. The
  single-source version test only scanned `.py` files and `mace_cli`, so a
  version written in prose had nothing holding it to the package; it is now
  pinned. Historical "NEW in v1.1.0" notes and the v1.0.0 Zenodo citation are
  deliberately left alone.
- **A job no longer runs CRYSTAL on an empty INPUT when `$SCRATCH` is empty.**
  On some nodes (agx-000) `$SCRATCH` is empty inside the job even under
  `bash --login`, so the scratch directory became `/crys23`, nothing could be
  staged there and CRYSTAL stopped with `END OF DATA IN INPUT DECK` (reproduced
  on HPCC with the old script). The OPT/SP/FREQ and properties job scripts now
  check the scratch directory before staging into it and otherwise use, in
  order, `/mnt/scratch/$USER` (what HPCC sets `$SCRATCH` to, so the same
  directory), `.mace_scratch/` in the submit directory, then `$TMPDIR`, saying
  which in the job log. With none writable the job stops before CRYSTAL runs,
  and `<job>.out` then holds only that error (the previous output is kept as
  `<job>.out.prev<N>`, or overwritten where it cannot be moved), so an earlier
  run's results or error are never read as this one's. The stop is classified
  as `scratch_error` (`scratch` in `mace check`) and is not recovered
  automatically: every fallback, the shared submit directory included, was
  unwritable, so a resubmission would stop the same way. The directory used is
  recorded as `.<job>.scratch` in the submit directory; the recovery reads it
  to find OPTINFO.DAT and fort.87, and a RESTART rerun that lands in a
  different directory takes OPTINFO.DAT, fort.20 and fort.9 together from
  whichever directory has the newer OPTINFO.DAT.
  When `$SCRATCH` is empty where the recovery runs, it now rebuilds it the same
  way instead of treating the scratch directory as unknown.
- **`recovery_config.yaml` is read.** The recovery engine only looked for the
  file in the directory it ran in, so the one shipped in `mace/config` never
  applied. It now uses an explicit `--config` file, else `recovery_config.yaml`
  in the current directory, else the shipped one. A file is merged into the
  built-in defaults key by key, where it used to replace the whole
  `error_recovery` section (so a partial file silently dropped every recovery it
  did not mention). An entry naming a handler that does not exist is reported;
  for an error type MACE recovers, the built-in handler is used with the
  entry's other settings, so its `max_retries` still counts and
  `manual_escalation` means `max_retries: 0` - a file can never switch on a
  recovery it switched off. The shipped file named five handlers that do not
  exist; it now lists exactly the recoveries MACE performs, with the built-in
  values. With no `recovery_config.yaml` in the job directory (the usual case)
  every recovery runs as before. An older copy of the shipped file in a job
  directory (the legacy `copy_dependencies` put one there) still keeps
  disk-space clean-up off (`max_retries: 0`), but is now merged instead of
  replacing the section, so the recoveries it does not mention - the newer
  optimization fresh start for a collapsed step size - are no longer dropped.
- **`opt2d12 --directory DIR --config-file T` no longer asks, or stops, per
  file.** It asked "Apply these settings from config file? [Y/n]" for every
  file; with nothing on stdin each file logged "EOF when reading a line", no
  deck was written, and the run exited 0. A single `--out-file` with nothing on
  stdin (or a closed one) failed the same way. See Added for the batch run.
- **A saved `opt2d12` template applies to other structures.** Settings saved
  with `--save-options` from a parent with an EXTERNAL basis stored the marker
  "EXTERNAL (from original D12)" as the basis; applied with `--config-file` to
  any internal-basis parent the deck failed with "External basis set path not
  configured properly" (78 of the 118 `test/OPT` parents with the report's
  Ag1Br1 template). That marker now means "each structure's own parent basis":
  an external-basis parent keeps its own basis records, an internal one its
  named basis. A template that names an internal basis (POB-TZVP-REV2) or an
  external basis directory still applies it to every file. Templates saved by
  earlier versions load and follow the same rule.
- **A deck that moves an external-basis parent to an internal basis writes
  plain atomic numbers.** The parent numbers ECP atoms by its basis (silver
  247); under BASISSET the deck now writes 47, and the basis-coverage check,
  which skipped every such deck, runs on it (an HSESOL3C template on the five
  lead parents in `test/OPT` is now refused: SOLDEF2MSVP has no Pb).
- **A template from a P1 structure no longer writes every atom of a symmetric
  one.** Its "write all atoms" was applied as is, so an HSESOL3C template saved
  from a molecule wrote diamond's two carbons under space group 227 (31 of 118
  `test/OPT` decks). A structure with symmetry is written as its asymmetric
  unit.
- **A lone `opt2d12 --out-file X.out` uses the `X.d12` beside it.** Without
  `--d12-file` it ran from the `.out` alone, with or without `--config-file`
  (a shell glob that matched one file did the same), so the parent deck's
  settings were silently replaced by what the `.out` shows: Ag1Br1's external
  ECP basis became POB-TZVP-REV2, while the log said the parent's basis was
  kept. It is now paired like each file of a batch; a file with no `.d12`
  beside it says it is converted from the `.out` alone.
- **`opt2d12` never answers the basis-coverage question for you.** With
  `--yes`, `--non-interactive` or a `--config-file` batch, a basis that lacks
  an element (SOLDEF2MSVP and Pb) still asked "Do you want to continue
  anyway?" at a terminal, and Enter wrote a deck CRYSTAL rejects. That file
  now fails with the reason, the same with a terminal as without. With stdin
  closed (`<&-`) opt2d12 crashed with AttributeError where it checked for a
  terminal; it is treated as nobody to ask.

### Added

- **The `.d12` parser reads the whole geometry input**: the space, layer, rod
  or point group record, CRYSTAL's IFHR (rhombohedral axes) flag, the cell
  record expanded to a, b, c, alpha, beta, gamma, and every atom (atomic
  number and coordinates). It used to stop at the space group. Every CRYSTAL
  and SLAB deck in `test/` (137) is rebuilt from the parse, record for record,
  by MACE's own deck writer. `opt2d12` still takes a child's geometry from the
  `.out`: its decks are byte-identical to before for all 1458 SP/OPT/FREQ
  children of the `test/OPT` and `test/SP` parents. A deck that names its
  space group by symbol (IFLAG = 1) is read too, in the spaced form CRYSTAL
  takes ("F M 3 M", "P 21/C", "P -4 21 M", "R -3 C"); a spelling CRYSTAL23
  refuses or misreads ("FM3M", "F M -3 M", "P -4 2 1 M") is reported in the
  parse as `geometry_unparsed` instead of silently leaving the geometry out.
- **`mace preflight`** - runs CRYSTAL over a copy of a deck with a TESTPDIM
  record inserted. TESTPDIM stops after the whole input is read and symmetry
  analyzed, which is late enough to catch a bad group, a bad lattice record, or a
  basis set with no functions for one of the elements. Measured at about a second
  per deck, against a queue wait to learn the same thing. A run that produced no
  `fort.87` *and* no success marker counts as a failure, because that is exactly
  how the hexagonal slab died.
- **One `opt2d12` template for many structures.** `--config-file` applies to
  every file of `--directory DIR` or of several `--out-file` paths (a shell
  glob such as `opts/*.out`; each `.out` is paired with the `.d12` of the same
  name beside it). Before any file it prints the plan (config, file count,
  output directory, the settings it applies) and, at a terminal only, asks
  once "Apply to all N files? [Y/n]"; `-y`/`--yes` skips that, and with no
  terminal nothing is asked. A single file still asks at a terminal, and a
  piped "y" works as before. Each file gets a status line and the run ends
  with "N written, M failed" and each failure's reason; it exits non-zero if
  any file failed or none was written (as does any `--directory` run now,
  with or without a config; files are taken in name order). Nothing is asked
  per file: a question a file would need (a basis without one of its
  elements) fails that file instead. A path given twice is converted once,
  and two inputs whose decks would have the same name in the same directory
  (`dupA/X.out` and `dupB/X.out` with one `--output-dir`) both fail rather
  than one overwriting the other. A parent optimization CRYSTAL never finished
  (no OPT END: killed, out of time) or stopped unconverged at the cycle limit
  ("OPT END - FAILED", which prints no final geometry) is still converted, from
  its starting geometry, and flagged on its line and in the summary: "N
  written (K from unfinished or failed optimizations), M failed". On the 118 `test/OPT` parents the batch
  decks are byte-identical to the single-file config path run once per file,
  for templates saved from an internal-basis, an external-basis and an
  HSESOL3C parent.
- **SLABCUT**, opt-in via `options['slabcut']`. The manual builds a slab from the
  3D structure and derives the layer group itself, sidestepping the orientation
  ambiguity that forces a refusal above. It needs two runs - a SLABINFO probe to
  number the atomic layers, then the real cut.
- **GUESSP SCF restart.** Every run already saved its converged density matrix as
  `$JOB.f9` and nothing ever read one back. CRYSTAL reads the guess from
  `fort.20`, so the job script now stages one. Measured on the same MgO cell,
  functional and basis: cold start -275.14904353964 AU in 11 cycles against
  -275.14904354956 AU in 5. It is opt-in - a staged matrix is only a valid guess
  for the same geometry *and* basis set. Where no matrix exists the script strips
  the GUESSP record from the scratch copy, because CRYSTAL does not fall back to
  the atomic guess; it stops with
  `ERROR **** GUESSP **** COPY OF WAVEFUNCTION FILE fort.20 CAN NOT BE FOUND`.
- **A stale node-exclusion warning.** `MENDOZA_NODES` is a hardcoded list and the
  partition is not static. A retired node leaves a harmless dead entry, but a
  *renamed* node silently stops being excluded while the menu still reports that
  it is. The exclusion menu now checks the list against `sinfo` and says
  something only when there is drift, staying silent off-cluster or when the
  check fails.
- **ECHG map planes can come from a config** instead of only interactive
  prompts, so an ECHG deck is reachable from a workflow at all.

### Changed

- **One set of convergence presets.** The planner, `opt2d12`, `cif2d12` and
  the quick-start plan share `opt2d12`'s Standard / Tight / Very tight tiers
  for OPT convergence and SCF tolerances. The planner's default OPT is now
  Standard (TOLDEG 0.0003, TOLDEX 0.0012), which it previously set 10 times
  tighter; planner steps with no settings of their own inherit the parent's.
  The planner's ATOMSONLY is corrected to ATOMONLY.
- **FREQ decks default to Very tight SCF tolerances** (TOLINTEG 9 9 9 11 38,
  TOLDEE 11) on every path unless a value is given.
- **NUMDERIV is written only when asked for.**
- **The basis-set menu asks the measured tables.** For any structure with max
  Z > 36 it previously offered only POB-DZVP and POB-TZVP, neither of which
  carries Pb, while filtering out POB-TZVP-REV2, which does - the menu was itself
  a cause of the reported failure. It now falls back to an external set when
  nothing internal covers the structure.
- **VBH, PWGGA and WCGGA are emitted as EXCHANGE/CORRELAT pairs**, which is how
  CRYSTAL exposes them; there is no standalone keyword. The pairings are the
  manual's own. The input parser reads such pairs back to the single name MACE
  uses - the same gap already existed for PBESOL and SOGGA, whose decks never
  round-tripped either.
- **The SPINLOCK prompt no longer claims -1 gives an antiferromagnetic guess.**
  It does not: SPINLOCK locks the cell total n_alpha - n_beta. The manual's AFM
  recipe needs ATOMSPIN, which MACE does not implement.
- **`FUNCTIONAL_KEYWORD_MAP` is deleted** rather than corrected. It was imported
  in one place and never read there, and half its entries were wrong in ways that
  would have been silent - LC-wPBE and LC-BLYP, both range-separated hybrids,
  were mapped onto Wu-Cohen exchange records.
- **`opt2d12 --save-options` stores settings, not a structure.** The template
  no longer holds the parent's atoms, optimization log, external basis records
  or k-point mesh (the report's template drops from 90 KB to about 1 KB). No
  opt2d12 question sets the mesh, so the stored one was always the parent's,
  and it made every other structure's mesh be regenerated from its cell
  (Ag1Cl3's SHRINK 8 16 became 9 18). Each structure now keeps its own
  parent's SHRINK, also with a template saved earlier: a config's `k_points`
  is not applied. To give every file one mesh on purpose, write
  `"k_points_for_all_files": "8 16"` (or `"8 8 8"`, or one number) into the
  config by hand; it is written exactly as given.
- The interactive CIF-converter banner credits its author only; the trailing
  tool-attribution clause is gone, and a test keeps it that way.

### Testing

771 tests at the start of this work, 1074 now, with the regression tests
asserting against real decks under `test/` rather than fixtures wherever one
exists.

## [1.1.1] - 2026-08-18

Verified end-to-end on MSU HPCC (SLURM + CRYSTAL23): a bare submission runs only
what was submitted, and a `--progress full_electronic` run drove
OPT → SP → BAND + DOSS to completion inside its own plan directory.

### Changed

- **A manual submission no longer starts a workflow.** Every job script MACE
  writes ends with the queue-manager completion callback; outside a workflow
  that callback used to adopt every untracked `.d12`/`.d3` in the directory
  tree and then progress the completed job from built-in defaults (OPT → SP,
  SP → BAND + DOSS) inside a synthesized `workflow_outputs/workflow_<time>/`
  directory. A hand-run `mace submit` now runs exactly what was submitted;
  progression requires a plan. `MACE_PLANLESS_PROGRESSION=1` restores the old
  behavior. `mace manager` and `--callback-mode submit_new` are unchanged —
  keeping the queue fed is what they are for.

### Added

- `mace submit --progress TEMPLATE|interactive` — plan the steps that should
  follow decks you built by hand, then run them as each job completes. Writes
  a real workflow plan (no CIF conversion: the existing decks are the starting
  point) and stamps each submission with its workflow ID, so progression
  follows the plan rather than defaults. `interactive` uses the planner's own
  step prompts. Implies `--track`. A deck may enter the sequence at any step.
- `mace submit --walltime TIME` — override the SLURM time limit for a
  submission instead of taking the per-calc-type defaults (OPT 7 days, SP 3
  days). Short jobs queue sooner, which matters for test and QA runs. Applies
  to the manual D12/D3 submitters, the tracked path, and every step of a
  `--progress` plan.
- Short flags for `mace workflow`: `-i/--interactive`, `-e/--execute`,
  `-q/--quick-start`, `-s/--status`, `-T/--show-templates`, `-c/--cif-dir`,
  `-d/--d12-dir`, `-w/--workflow`, `-W/--work-dir`, `-D/--db-path`,
  `-j/--max-jobs`.

### Fixed

- **`mace opt2d12 --calc-type X --non-interactive` no longer dies with EOFError**
  when nothing is on stdin - the form MACE's own help documents. It walked the
  interactive settings flow (the workflow engine drives that flow with scripted
  answers) and crashed at the first prompt. When no answers are available it
  now keeps the settings extracted from the source calculation. The engine's
  generated decks are byte-identical before and after.
- **`mace opt2d12 --output-dir` is honored.** It was parsed and the directory
  created, but the path never reached the writer, so decks always landed in the
  current directory. The deck's CRYSTAL title line still carries only the file
  name.

- **Follow-up steps could be orphaned from their workflow.** In an isolated
  context the completion callback re-registers a finished job from its output
  file, and that scan recorded no workflow metadata. Progression still found
  the plan (it falls back to `$MACE_WORKFLOW_ID`), but the output-directory
  resolver read the same empty settings and minted a fresh
  `workflow_<timestamp>` directory with no plan file — so the step it generated
  was stamped with a dead id, and when that step finished no plan could be
  found and the chain stopped. The resolver now uses the same environment
  fallback, and the scan stamps the workflow id onto the records it creates.
  Previously masked: the old no-plan branch emitted default BAND + DOSS
  regardless, so the chain looked correct while the id was already wrong.
- `WORKFLOW_TEMPLATES` was missing `opt_sp_freq`, which the CLI already
  accepted.
- **Template BAND steps used a lower-quality k-path than the interactive
  planner.** `seekpath_full` is the only flag that makes the D3 generator take
  the SeeK-path (HPKOT) branch, and the quick-start / `--progress <template>`
  BAND config never set it — so those BANDs fell back to the built-in
  extended-Bravais tables while the planner's expert config got the full path.
  Wrong paths only for non-cubic lattices, and silent either way. The flag is
  now requested unconditionally: plans are built on a login node while the D3
  is generated later on a compute node, so probing for the library at plan time
  would test the wrong machine, and `get_seekpath_full_kpath` already degrades
  seekpath → literature → static on its own. Verified on HPCC (diamond):
  `default - X-GAMMA-L-W-GAMMA` (4 segments) became
  `SeeKPath (w.I) - GAMMA-X-U-|-K-GAMMA-L-W-X` (6 segments).

## [1.1.0] - 2026-07-12

Changes since 1.0.5.

### Added

#### Themed terminal UI (visual layer)
- `mace/utils/ui.py` rich-based facade: status lines, tables, progress bars,
  spinners, live dashboards, themed startup banner
- Selectable color themes (`--theme <name>`, `--save-theme`, `MACE_THEME`)
- Fully optional: degrades to plain text without `rich`; honors `NO_COLOR` and
  `TERM=dumb`; user text is never interpreted as markup (injection-safe)

### Fixed
- **d12/d3 generation** — an aborted D12 creation no longer leaves a truncated deck reported as success; TOLINTEG extraction preserves custom tolerances on pure-DFT outputs; batch mode re-prompts on invalid input.
- **Database** — TRANSPORT and CHARGE+POTENTIAL outputs use the canonical material ID (no duplicate rows), and transport statistics count Seebeck entries only.
- **Plotting/UX polish** — missing/unreadable spectra files give clean errors instead of tracebacks; `mace plotting` propagates its exit status; malformed `--iso` is a usage error; plain-mode (no-rich) output preserves bracketed text verbatim.
- **HPC QA campaign (release wave)** — ~18 workflow/queue/recovery fixes from an end-to-end SLURM test campaign. Highlights: one workflow mints ONE workflow id (fixes the nested-DB split-brain between engine and queue manager); job-state checks confirm via `sacct` before failing jobs missing from `squeue`; recovery attempt caps count the whole lineage, so resubmission chains stay bounded; TRANSPORT d3 decks emit `NEWK` before `BOLTZTRA` and get their properties terminator `END`; CIF conversion without spglib writes the CIF's asymmetric unit instead of the expanded cell; plotting pins a headless matplotlib backend (`Agg`) for compute nodes.

### Changed
- **Repo hygiene** — internal planning/audit docs untracked (kept on disk), generated artifacts gitignored, unused legacy modules removed (legacy queue manager, portable SLURM generator, contextual executor/planner variants, installer/env-helper utilities); PyPDF2 dropped as a dependency (nothing imports it).
- **Formula ordering** for newly extracted formulas follows a revised element convention (e.g. `TiPbO3` vs the older `PbO3Ti`); previously stored rows are unaffected. `fermi_energy` rows written by v1.1.0 carry the correct `Hartree` unit label (older rows said `eV` while storing Hartree values).

## [1.0.5] - 2026-06-14

Changes since 1.0.0. Tagged retroactively: 1.0.5 was the version `mace --version`
reported at this point, but it was never separately announced, so its changes
were first described in the 1.1.0 notes. 1.0.1 through 1.0.4 were never released.

### Added

#### Plotting subsystem (`mace plotting`)
One command for publication-ready plots from CRYSTAL outputs, with content-based
file detection, per-kind flags (`--band --dos --structure --cube --freq --ir
--raman --all`), an interactive menu, and `-o` output routing:

#### Deep property extraction
The materials database now stores the full scientific results of each calculation
type, not just scalar summaries — as compact JSON plus flat, queryable scalar rows
in the existing `properties` table (no schema change):

#### Queue / submission
- In-place submission for manual `mace submit` (no forced reorganization);
  `--organize` restores the copy-into-folders layout
- `completion` command surfaced in `mace --help`; all command help audited
  against the real parsers (fabricated flags removed)

### Fixed
- **FREQ extractor** previously parsed vibrational data into a discarded local variable; frequencies/IR/Raman now actually persist (fixed the units-anchor `(CM**-1)` and mode-range parse bugs).
- **Error-recovery chain** — previously-dead paths called nonexistent DB/manager APIs (errors swallowed): max-recovery-attempts, recovered-job resubmission, and workflow-engine step submission now work end-to-end; the timeout handler parses the `-t 7-00:00:00` day form and never shrinks walltime; the memory handler preserves `--mem-per-cpu` vs `--mem`; recovered resubmissions record the bumped script so repeated failures escalate cumulatively.
- **d12/d3 generation correctness** — SPINLOCK parse + round-trip (a configured spin lock survives OPT continuation and JSON-config reuse); origin-setting preservation; k-point table fixes (C-/I-centered orthorhombic assignments, duplicate table key); DOSS Fermi-window unit consistency.
- **JSON config save/apply round-trips** for `opt2d12` / `opt2d3` (settings no longer drift through a save→load cycle); invalid interactive calc-type choices re-prompt instead of silently defaulting to BAND.
- **Database correctness** — canonical material-ID derivation, NULL-safe dedup on re-extraction, pressure unit-conversion table (kbar/Mbar swap, atm factor), enthalpy H = G + TS, full-precision Hartree↔eV constants (single source: `mace/constants.py`), pyarrow import-order crash guard.

### Changed
- **Repo hygiene** — development and internal artifacts kept out of the repository; ad-hoc validation scripts centralized under `tests/`.
- **CI** — GitHub Actions runs the self-contained test suite on a fresh clone (data-dependent tests skip without the local `test/` corpus).

## [1.0.0] - 2026-02-12

### Added

#### Core Framework
- **MACE CLI** (`mace_cli`) - Unified command-line interface for all MACE functionality
- **Workflow Manager** - Complete end-to-end workflow planning and execution system
- **Material Tracking Database** - SQLite + ASE integration for calculation history and provenance
- **Enhanced Queue Manager** - Intelligent SLURM job scheduling with material tracking
- **Error Recovery System** - Automated detection and fixing of common CRYSTAL errors

#### Input Generation
- **NewCifToD12.py** - CIF to CRYSTAL D12 input file conversion
- **CRYSTALOptToD12.py** - Generate inputs from optimized structures
- **CRYSTALOptToD3.py** - Unified D3 generation with basic/advanced/expert modes

#### Band Structure
- **Seekpath Integration** - Accurate k-path generation using the seekpath library (HPKOT methodology)
- Support for all 26 extended Bravais lattice types (cF1, cF2, hR1, hR2, mC1, etc.)
- Proper handling of parametric k-points for non-cubic lattices
- Automatic SHRINK factor calculation for exact integer k-point coordinates

#### Property Calculations
- Band structure (BAND) input generation with automatic k-path detection
- Density of states (DOSS) with orbital resolution
- Transport properties (Boltzmann transport calculations)
- Charge density and electrostatic potential analysis

#### Workflow Features
- Interactive workflow planning with three customization levels (Basic/Advanced/Expert)
- Pre-defined workflow templates (basic_opt, opt_sp, full_electronic, double_opt, complete)
- Workflow isolation for running multiple workflows in the same directory
- JSON-based configuration persistence for reproducibility

#### File Management
- Complete file storage with settings extraction from D12/D3 files
- SHA256 checksums for file integrity verification
- Organized storage by calculation ID with metadata preservation

#### Monitoring
- Real-time calculation monitoring dashboard
- Completion status checking with `mace completion`
- Zombie job detection and cleanup

### Dependencies
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- ase >= 3.22.0
- spglib >= 1.16.0
- PyPDF2 >= 2.0.0
- pyyaml >= 6.0
- pandas >= 1.3.0
- seekpath (optional, recommended for accurate band structure k-paths)

---

## Roadmap (not yet scheduled)

### Planned
- PyPI package distribution
- Additional workflow templates
- Enhanced visualization tools
