# Changelog

All notable changes to MACE (Mendoza Automated CRYSTAL Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
- **The phonon fallback k-path table names only points it has.** Most F- and
  I-centred cubic groups (196, 197, 199, 202-204, 206, 209-211, 214, 217, 219,
  220, 226) got the simple-cubic path, and the monoclinic and triclinic paths
  named points (M1, X, V, W, ...) with no coordinates, so their segments were
  dropped. F and I cubic now get the fcc and bcc paths, and monoclinic and
  triclinic use the points MACE's band-path code uses (CRYSTAL23 manual Table
  14.1 for P monoclinic). The table is read only by `get_auto_phonon_path`,
  which nothing calls today, so no deck changes.
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
  piece, even when it is centred on z = 1/2 or straddles z = 0/1, and every
  z distance in it is the cell's own. A polar-layer-group slab keeps its
  c times z heights; when it straddles z = 0/1, the atoms just below z = 1
  are written one cell lower, next to the rest of the layer.
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
  win had every coordinate doubled (body-centred cubic x4, hexagonal x6), so
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
  asks again, as the D12 tools do. Every recognised answer, and end of input,
  behave as before.

### Removed

- Five standalone scripts that nothing runs, imports or documents:
  `mace/workflow/status.py`, `mace/workflow/callback.py` and
  `mace/workflow/check_workflows.py` (superseded by `mace status` and the
  queue manager's completion callback), `mace/utils/scf_settings_extractor.py`
  and `mace/utils/analyze_script_dependencies.py`.

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
family of others. Every behavioural change below was run through
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
  comes from the conventional a and b, and was only equalised for 3D decks. In
  a centred-rectangular layer group the two primitive directions are
  equivalent, so graphene in layer group 47 got `SHRINK 0 36 / 18 10 1` and
  CRYSTAL stopped with "SHRINK BREAKS SYMMETRY". Square, hexagonal and
  centred-rectangular layer groups (CRYSTAL numbers the last 10, 13, 16, 22,
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
  its HF flavour (`"functional": "RHF"`) so both converters read it the same
  way.
- **`d12_from_config.py` runs.** It passed the input file as an argument
  neither converter accepts, and did not recognise a CRYSTAL output that
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
- **cif2d12 FREQ decks follow the lattice centring for the phonon path.** The
  automatic phonon-dispersion path read the centring only from a CRYSTAL
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
  in the scratch `fort.87`; SLURM says COMPLETED). This is now recognised
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
- **Timeouts are recognised at all.** SLURM writes the time-limit notice to
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
  - an HSEsol-3c SP parent produced FREQ/SP/OPT2 decks labelled `HSESOL3C-D3`
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
  exclusion while `mace submit`, which honours the same setting, kept it. Only an
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
  analysed, which is late enough to catch a bad group, a bad lattice record, or a
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
  than one overwriting the other. A parent optimisation CRYSTAL never finished
  (no OPT END: killed, out of time) or stopped unconverged at the cycle limit
  ("OPT END - FAILED", which prints no final geometry) is still converted, from
  its starting geometry, and flagged on its line and in the summary: "N
  written (K from unfinished or failed optimisations), M failed". On the 118 `test/OPT` parents the batch
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
  no longer holds the parent's atoms, optimisation log, external basis records
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
- **`mace opt2d12 --output-dir` is honoured.** It was parsed and the directory
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
