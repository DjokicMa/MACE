# MACE Codebase Audit — 2026-06-12

Produced by 8 parallel review agents, one per module group. Every finding below was
verified against the code (and where possible reproduced live against the real CRYSTAL
outputs in `test/`) — none are speculative. Severity: **CRITICAL** = feature always broken
or data corrupted; **MAJOR** = broken under common usage; **MINOR** = edge case or cosmetic.

**Environment note:** the scientific stack (`ase` 3.25, `spglib` 2.6, `seekpath` 2.2.1,
`matplotlib`) exists only in the Anaconda base python (`/home/marcus/anaconda3/bin/python`),
not the system python. Several findings are conditional on which interpreter runs MACE:
under system python the extractor loses ASE structure storage and BAND generation falls
back from SeeK-path to the built-in k-point tables (hitting Crystal_d3 finding #1).
Generated SLURM/callback scripts that invoke bare `python` inherit whatever is on the
node's PATH — worth pinning to the conda interpreter.

## Already fixed this session

- `mace analyze --extract-properties <dir>` always crashed (flag not defined in
  `property_extractor.py` argparse + argument-order scrambling) — fixed in `mace_cli`
  by rebuilding args from `original_argv` and translating to `--scan-directory`.
- Neighbor/bond analysis only read the **first** neighbor table (initial geometry).
  Now also extracts the **last** table (optimized geometry) as `final_*` properties
  (`final_min_bond_distance_ang`, `final_max_bond_distance_ang`,
  `final_neighbor_analysis`, `final_max_coordination_number`, ...).
- 5-token wrapped continuation lines in the neighbor table (e.g. `2 C    0 0 1`) were
  misparsed as shell headers, producing fake 1.0 Å `min_bond_distance_ang` values.
  Fixed by requiring decimal points in the R/ANG, R/AU columns.

### Workflow-chain fixes (2026-06-12, verified end-to-end with mocked SLURM)

The full `mace workflow --quick-start` chain (opt_sp and full_electronic) was
verified locally: OPT submit → completion callback → SP generation/submit →
SP completion → BAND+DOSS generation/submit → BAND completion, with a single
material record, correct statuses, and no duplicates across repeated callbacks.
Fixes applied:

- CLI §1.1 (workflow branch only): `mace workflow` now rebuilds argv from
  `original_argv` (and drops `--no-banner`), so `--execute/--quick-start` work.
- Workflow §3.1/§3.2: engine D3 submission — `_submit_slurm_job` →
  `_submit_calculation_to_slurm`, `db.update_calculation` →
  `update_calculation_status` (BAND/DOSS/TRANSPORT submission was dead).
- Workflow §3.3: executor plan validation accepts numbered types (OPT2, SP2...).
- Workflow §3.4: executor reads `queue_management` from the plan top level.
- Workflow §3.5: numbered calcs select their numbered SLURM template.
- Workflow §3.6: D3 calcs read workflow_id from parent's settings_json.
- Workflow §3.8: `full_job_name` NameError in the submit fallback.
- Workflow §3.9: monitoring-guide format-args/NameError (guide now written).
- Workflow §3.11: step-config lookup is exact-type match ("SP" no longer
  matches "TRANSPORT_3").
- Queue §4 (new finding, root cause of dead progression): the
  job-left-queue completion check used markers ("CRYSTAL ENDS",
  "CALCULATION TERMINATED") that match **0 of 637** real outputs — every
  finished job was marked failed. Now reuses the validated
  `completion_checker.categorize_output_file` (error patterns → OPT END /
  TOTAL CPU TIME =), and prefers the .out matching the calc's input stem.
- Queue §4.3: completed/failed handlers now look up by calc id (the slurm-id
  lookup made them no-ops), and the status update happens before handlers run.
- Database §2.1/§2.2: populate_completed_jobs stores output_file on records,
  dedups by output_file OR work_dir+calc_type (engine-created records have
  output_file=NULL), and detects BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL types.
- New finding (root cause of duplicate materials): the **executor** had its
  own `create_material_id_from_file` that kept workflow stems as-is
  ("1_dia_opt") while the queue manager/engine/scan all derive "1_dia" — the
  submitted record never matched the completion scan, so every completed job
  re-registered as a duplicate material and follow-ups hung off the duplicate.
  The executor now delegates to the canonical database derivation.

### Progression matrix verification (2026-06-12, 19 sandbox workflows)

Materials spanning bulk (sg 14, 62, 99, 167, 221, 227 + carbon frameworks),
a 4-layer-graphene+FSI slab, and LiFSI-DEC molecules (HSESOL3C), through
templates complete / full_electronic / opt_sp / charge_analysis plus a custom
OPT→SP→FREQ sequence driven through the real planner/executor. FREQ
generation verified against real FREQ inputs for 9 families (diamonds, Ag
halides, TiPbO3 ×2, TiSe2O6, TiSe, the LiFSI molecule); two families without
a FREQ .d12 on disk were checked against their SP lineage + FREQCALC block. Generated follow-ups compared
structurally against the real files in test/ (dimensionality, space group,
atom count, functional, basis mode, SHRINK; d3 headers/projections/paths):
SP, BAND, FREQ, and CHARGE+POTENTIAL all match; DOSS matches projections and
NEWK exactly, differing only in quick-start's explicit 1..N band range vs the
manual energy-window (-1 -1 + BMI/BMA) form. Two more bugs found and fixed:

- `Crystal_d12/CRYSTALOptToD12.py` (write_d12_file): the basis-compatibility
  check tested the default INTERNAL basis even when the source d12 supplied
  external basis records, then blocked on an interactive prompt — in workflow
  callbacks (no stdin) this killed SP/FREQ generation for any material with
  elements outside POB-TZVP-REV2 (TiPbO3, Ag2Br3...). The check is now skipped
  when external basis data is present, and non-interactive runs abort cleanly
  instead of crashing at the prompt.
- `Crystal_d3/d3_kpoints.py` (extract_and_process_shrink): configs use
  `"shrink": "auto"` (a string); when every numeric extraction fails (MOLECULE
  outputs: no SHRINK, no lattice) the string reached `shrink <= 0` →
  TypeError, killing BAND generation for molecules. Non-numeric markers now
  coerce to the default.

Caveats: TRANSPORT progression has no OPT→SP chain in test/ to drive
(standalone C1-RCSR-afi files only); molecule BAND in the real flow uses a
separate periodic-embedded SP (`_CRYSTAL_SP_symm`) — the workflow generates a
molecule-dim BAND d3 instead, and the engine correctly treats BAND as
optional if it fails; FREQ progression verified on the diamonds (refs match),
not driven for molecules.

### Second fix wave (2026-06-12 afternoon, committed)

- **CLI §1 closed out**: all command branches rebuild args from the original
  command line; database history import, status/queue redirects, --dry-run
  forwarding, and the engine default action (now status) fixed. (§1.1-1.6)
- **Utils §5 data-quality closed out**: package imports (population/dat/advanced
  processing now actually runs), INDIRECT-gap false match, first-vs-last
  occurrences, Fermi exponents + Hartree unit, Mulliken section anchoring and
  A.O.-table-only parsing, doubled final positions, point-1 convergence, and
  NULL-calc_id duplicate property rows. (§5.1-5.11) Verified against the 637
  real outputs.
- **JSON config round-trips verified and fixed**: opt2d3 applies saved configs
  non-interactively (calc type read from config; DOSS/TRANSPORT validation
  aligned with what configurators emit); opt2d12 configs no longer transplant
  the source material's spacegroup/dimensionality onto the target. convert
  --save_options/--batch verified working as-is.
- **Recovery §4 closed out**: full chain verified with mocked SLURM
  (detection → d12 fix → recovery record → resubmission). Includes the .f9
  protection, settings kwarg, resubmission method, attempt counters, taxonomy
  aliases, next-line MAXCYCLE/FMIXING parsing, Check_Scripts paths, NULL
  settings_json, job-left-queue routing through the failure handler, submit
  script location, and job_script recording. (§4.1-4.14)
- **d12/d3 correctness (part of §6/§7)**: non-interactive origin preservation,
  hybrid functional detection (B3LYP no longer parsed as BLYP), TOLINTEG
  positive-exponent handling, label-mode NLINE counting + GAMMA→G mapping,
  orthorhombic centering map, Fermi-centered DOSS energy windows.
- **opt_sp_freq** added as a built-in workflow template.

### Third fix wave (2026-06-12 night, committed) — data-quality batch

- **DAT post-processing rewritten (§5.13, finding #1 follow-on)**: `dat_file_processor`
  now parses the real CRYSTAL BAND.DAT (`# NKPT NBND NSPIN` header, `<abscissa>
  E1..E_NBND` rows, absolute-Hartree energies) and DOSS.DAT (wrapped multi-line
  records reshaped by `1+NPROJ`, `E-EFermi` energies, total DOS = last column).
  Band metal/insulator now requires the .out Fermi level (returns None instead
  of "metallic" when absent); DOS metallicity uses a gap-existence test robust
  to edge spikes and semimetals. property_extractor merges only compact
  `band_dat_*`/`doss_dat_*` scalars (no more JSON-dumping raw arrays). Verified:
  4LG graphene → metallic, 9 molecular electrolytes → ~7.5-8.4 eV insulators.
  (505f5ead)
- **§7.9 fixed**: `CRYSTALOptToD3.py:176` element regex `[A-Z][a-z]?` →
  `[A-Z][A-Za-z]?`; all-caps "TI"/"PB"/"SI" now captured (was returning an empty
  list, which made every manual atom projection fail the bound check). The
  `d3_interactive.py:1025` variant and geometry double-counting remain open.
  (11a158dc)
- **§2.6 fixed**: `utils/units.py` pressure factors — kbar/Mbar were swapped and
  atm ~1000× off; recomputed from the GPa base and round-trip verified.
  (1b7c064f)
- **§2.9 fixed**: `analysis/missing_data.py` `CALC_TYPE_PROPERTIES` realigned to
  the extractor's real vocabulary (`total_energy_au`, `final_primitive_a`,
  `zero_point_energy_au`, `total_kpoints`, `doss_dat_*`...); required props now
  only those emitted for any system of a type, so completeness is meaningful
  (was ~0% / "everything missing"). TRANSPORT/CHARGE+POTENTIAL require nothing
  (extractor doesn't parse those yet). (05c7d91c)
- **§2.10 fixed**: chemical formula extraction — materials.py delegates to
  formula_extractor (fixes queue manager + engine); formula_extractor now
  matches scientific-notation coordinates, decodes the Z+200 ECP offset, bounds
  its fallback to the geometry block, and orders metals-first. `1LiFSI*` →
  `LiC5NH10S2F2O7` (was `225810`), `Ti19O30` → `Ti19O30`, `Ag2Br3` → `AgBr`.
  (28e64fbf)

### Fourth fix wave (2026-06-13, committed) — calc-type generation batch

- **§7.9 sibling fixed**: `d3_interactive.py:1025` element regex `[A-Z][a-z]?`
  → `[A-Z][A-Za-z]?` (same all-caps two-letter bug as CRYSTALOptToD3:176).
  (de9c40f8)
- **§7-bundle `-D3-D3` fixed**: the continuation filename method-name guard
  only excluded 3C methods, not `-D3` already present, so a carried-over
  `B3LYP-D3` doubled to `B3LYP-D3-D3`. Added the missing guard (+ the same on
  the print_summary line). (ee56e9e1)
- **§7.1 (FREQ) fixed**: phonon BANDS deck wrote the `ISS NSUB` line before the
  path was resolved; label paths force shrink=0 and coordinate paths rescale it,
  so the written ISS didn't match the emitted segments. Header now written after
  resolution. (eb23aa59)
- **§2/§6 SPIN-into-closed-shell fixed**: the "set all defaults" branch derives
  spin from the source instead of hardcoding True, and CrystalInputParser inits
  spin_polarized=False; closed-shell sources (no SPIN keyword) stay closed-shell
  while SPIN-bearing sources are unchanged. (c0844e3d)
- **§8.13 fixed**: `ipDOS_V2.py` dropped every second row of unwrapped
  17+-column DOS files; now reads continuations only while the record is short.
  (fdf9cf8e)

**Still open (highest-value remainder):** transport-coefficient and
charge/potential-grid extraction (not implemented — missing-data now treats
them as having no required props); memory/timeout recovery resource edits not
carried into resubmitted scripts; configured SPINLOCK never written by active
writers (§6.13 remainder); geometry double-counting in d3 atom projections
(§7.9 remainder); monoclinic unique-axis detection
(§6.14); duplicate pre-scaled seekpath fallback tables (§7.1 — only triggers
when the seekpath library is missing; present in the anaconda env);
contextual planner/executor variants (§3.7); plotting branch internals (§8.2+);
materials_contextual drift (§2.4-2.5); and the defunct code/ cleanup.

## Cross-cutting root causes

These five patterns account for the majority of the critical findings:

1. **CLI argument scrambling** — `args.args + remaining` from `parse_known_args`
   separates flags from their values. Nearly every valued flag of `mace workflow`,
   `submit`, `monitor`, `manager`, `recover`, `engine`, `plotting`, `queue`, and
   `status` crashes or silently misbehaves. The `analyze` and `database` branches
   already use the correct `original_argv` rebuild — applying the same pattern to the
   other branches fixes ~10 findings at once.
2. **Bare imports that fail under the package layout** — modules import siblings as
   top-level names (`from population_analysis_processor import ...`). These fail in
   every production context and are silently swallowed by `try/except`, permanently
   disabling whole features (population analysis post-processing, BAND/DOSS .DAT
   processing, advanced electronic analysis, property history).
3. **Divergent material-ID derivation** — at least four implementations
   (`create_material_id_from_file`, `database/populate_completed_jobs.py`,
   `file_manager._infer_material_id_from_filename`,
   `create_fresh_database._extract_material_id`) produce different IDs for the same
   file, splitting one material across multiple DB identities.
4. **First-vs-last occurrence parsing** — same class as the neighbor-table bug:
   several extractors take the first regex match in a file that prints the quantity
   repeatedly during optimization (direct/indirect gap, gradient norm, D3 dispersion,
   seekpath geometry), storing pre-optimization values.
5. **Writer/reader key mismatches** — one module writes a key/column/property name,
   another reads a different one (`queue_management` placement, `input_settings_json`
   vs `settings_json`, expected property names in missing-data analysis, recovery
   error-type taxonomy).

---

## 1. CLI layer (`mace_cli`, `run_mace.py`, `mace_config.py`, `setup_mace.py`)

1. **CRITICAL** `mace_cli:1122` (also 1155, 1348, 1493, 1584, 2775, 2793) — `args.args + remaining` scrambles flag/value pairing, so virtually every documented value-taking flag crashes in the workflow, submit, monitor, manager, recover, engine, and plotting branches. Reproduced live: `workflow --execute plan.json`, `recover --action stats`, `engine --action status`, `monitor --interval 60`, `plotting --band -d .`, `manager --max-jobs 200 --reserve 20` (ValueError at `int('--reserve')`), `submit --exclude-nodes node1 file.d12`. Fix: rebuild each branch's args from `original_argv` exactly as the analyze (1441-1455) and database branches do.
2. **CRITICAL** `mace_cli:2627` — `mace database --action history` always broken: imports nonexistent `mace.database.history` (PropertyHistory lives in `mace/database/utils/history.py`).
3. **MAJOR** `mace/workflow/engine.py:3527` — engine's default `--action` is `process` (submits jobs) while the help documents `status`; bare `mace engine` silently submits the next workflow calculations.
4. **MAJOR** `mace_cli:1576` — deprecated `queue` redirect forwards only `args.args` (flags land in `remaining`), so `mace queue --status` silently launches the full job-submitting queue manager instead of showing status.
5. **MAJOR** `mace_cli:1564` — manager branch forwards `--dry-run`, which `queue/manager.py` argparse doesn't define → "unrecognized arguments" after the manager/db were already created; yet the help documents the flag.
6. **MAJOR** `mace_cli:1123` — appends `--no-banner` to the rebuilt argv, but `run_mace.py` argparse doesn't define it → `mace workflow ... --no-banner` exits. Rely on the `MACE_NO_BANNER` env var already set at line 1105.
7. **MAJOR** `mace_cli:668` — monitor help documents `--dashboard` but `queue/monitor.py` doesn't define it (it's the default action anyway).
8. **MINOR** `mace_cli:1363-1369` — `monitor --status --material-id X` parses the filter then never uses it; shows global stats.
9. **MINOR** `mace_cli:379-390` — monitor help documents action `health` and `--max-materials`; neither exists.
10. **MINOR** `mace_cli:453-528` — `show_workflow_action_help` documents an `--action` interface and ~9 flags that don't exist in `run_mace.py`.
11. **MINOR** `mace_cli:402-441, 533-575` — engine and recover action help document numerous nonexistent flags.
12. **MINOR** `mace_cli:1615` — calls undefined `show_action_help()` (NameError if reached); should be `show_database_action_help(action)`.
13. **MINOR** `mace_cli:2512, 2553, 2604, 2727` — database actions workflow/validate/history/aggregate parse options from `remaining` instead of the rebuilt `all_args` like the other actions.
14. **MINOR** `mace_cli:1483-1485` — `status` alias drops `remaining` (so `mace status --detailed` ignores the flag) and prepends `--status` twice.
15. **MINOR** `setup_mace.py:239-242, 369` — `--update-templates` targets `code/Job_Scripts/` which doesn't exist (live templates are `mace/submission/*.sh`); silently no-ops.

## 2. Database (`mace/database/`)

1. **CRITICAL** `populate_completed_jobs.py:163-196` — every queue-manager scan duplicates completed calculations: `output_file` is never stored (fallback calls nonexistent `db.update_calculation_files`, AttributeError silently passed), and dedup + `_find_calculation_by_output_file` both compare that NULL column. Verified: two runs over one .out → 2 records.
2. **CRITICAL** `populate_completed_jobs.py:40-46` — material-ID derivation diverges from `create_material_id_from_file` (strips only exact `_opt/_sp/_freq` endings): `mat_opt2.out` → `mat_opt2` vs `mat`; same material gets two IDs. The richer, correct copy in `database/utils/` is NOT the one production imports.
3. **CRITICAL** `materials.py:1119-1142` — `store_material_property` always fails: inserts a `uuid4()` string into `property_id INTEGER PRIMARY KEY AUTOINCREMENT` → `sqlite3.IntegrityError: datatype mismatch` (reproduced on a fresh DB).
4. **MAJOR** `materials_contextual.py:166, 208-215` — `copy_to_context` always crashes: calls nonexistent `create_or_update_material` and passes kwargs `unit=`/`conditions=` that `store_material_property` doesn't accept.
5. **MAJOR** `materials_contextual.py:257, 293` — `get_workflow_materials`/`get_workflow_calculations` call `self.get_connection()`; the method is `_get_connection` → AttributeError whenever a workflow_id is set.
6. **MAJOR** ✅ FIXED (1b7c064f) `utils/units.py:48-53` — kbar/Mbar pressure conversion factors are swapped and `atm` is 1000× too small (`convert_units(10,'gpa','kbar')` → 0.1 instead of 100).
7. **MAJOR** `materials.py:635` — `cleanup_old_records` uses `datetime.now().replace(day=day-30)` → ValueError on ~every invocation; use `timedelta`.
8. **MAJOR** `utils/validation.py:626-647` — `validate_materials(material_ids=[...])` is a `pass` stub whose report dict then KeyErrors in the formatter; the interactive explorer's `do_validate` always hits this.
9. **MAJOR** ✅ FIXED (05c7d91c) `analysis/missing_data.py:16-48` — expected property names don't match anything the extractor writes (`total_energy` vs `total_energy_au`, `final_a` vs `final_primitive_a`, ...) → every completed calc reported "missing data".
10. **MAJOR** ✅ FIXED (28e64fbf) `materials.py:1439-1476` — `extract_formula_from_d12` returns garbage (e.g. `'2'` for diamond; joins counts without symbols) and the queue manager imports THIS version, not the correct `mace.utils.formula_extractor`. (Both versions fixed: materials.py delegates; formula_extractor handles sci-notation coords + ECP offset.)
11. **MINOR** `analysis/aggregation.py:149, 185` — `group_by='conductivity_type'/'energy_range'` look up property names the extractor never writes; everything lands in "Unknown".
12. **MINOR** `materials_contextual.py:173, 188` — passes already-encoded JSON strings into functions that `json.dumps` again → double-encoded JSON (latent until #4 fixed).
13. **MINOR** `materials.py:984-1001` — `capture_workflow_execution_data` can `return instance_id` before assignment (NameError) and only returns the last material's instance.
14. **MINOR** `utils/history.py:582` — `changed_at.split('T')` never splits SQLite's space-separated timestamps; report groups by full timestamp instead of date.
15. **MINOR** `utils/create_fresh_database.py:311-331` — a third divergent material-ID variant (`name_opt_BULK_OPTGEOM_*` → `name_opt` instead of `name`).
16. Below threshold: `query/queries.py:335-352` `query_materials` silently ignores its `filters` argument; SELECT-then-UPDATE on `slurm_job_id` (materials.py:386-390) is not atomic across processes.

## 3. Workflow (`mace/workflow/`)

1. **CRITICAL** `engine.py:1574` — `_create_and_submit_d3_calculation` calls `self._submit_slurm_job(...)`, which doesn't exist (real name `_submit_calculation_to_slurm`) → every engine-driven BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL submission raises AttributeError, swallowed at 1477-1481; D3 steps are never submitted and orphaned `pending` records re-trigger each callback.
2. **CRITICAL** `engine.py:1576, 1580` — same function calls nonexistent `self.db.update_calculation(...)` (real: `update_calculation_status`).
3. **CRITICAL** `executor.py:414-417` — plan validation whitelists only base calc types but the planner legitimately emits numbered ones (`OPT2`, `SP2`, `BAND2`) → "Invalid calculation type: OPT2" aborts valid plans. Validate `calc_type.rstrip('0123456789')`.
4. **MAJOR** `executor.py:344` vs `planner.py:4847` — executor reads `execution_settings.queue_management` but planner writes top-level `queue_management`; user's queue limits silently replaced by hardcoded 200/30/5.
5. **MAJOR** `engine.py:3252` — numbered calcs pass `target_base_type` into script creation, so OPT2/SP2/FREQ2 reuse the step-1 template instead of the planner's `_opt2_2.sh` with custom resources.
6. **MAJOR** `engine.py:1552` — `parent_calc.get('workflow_id', 'manual')` always yields `'manual'` (no such column; it lives in `settings_json`) → D3 calcs break status grouping and plan lookup.
7. **MAJOR** `planner_contextual.py` / `executor_contextual.py` — written against a WorkflowContext API that doesn't exist (`get_active_context` vs `get_active`, `cleanup_on_exit=`, `ctx.get_database()`, etc., plus `os` used without import) — every contextual/isolated path fails at runtime.
8. **MINOR** `engine.py:763` — fallback branch references undefined `full_job_name` → NameError when a generator succeeds without printing "Submitted batch job".
9. **MINOR** `executor.py:2376-2438` — monitoring README template: 6 placeholders, 2 format args, out-of-scope variable; exception swallowed, file never created.
10. **MINOR** `engine.py:1873-2175` — no follow-up branch for completed TRANSPORT/CHARGE+POTENTIAL; custom sequences with steps after them stall permanently.
11. **MINOR** `engine.py:162` — substring match `calc_type in step_key` makes "SP" match "TRANSPORT_3"; wrong step config returned.
12. **MINOR** `engine.py:1545, 1453` — D3 step numbers hardcoded (BAND=3/DOSS=4/+10) disagree with plan-derived directory numbering; extra nested `BAND1` subdir with no `.workflow_metadata.json`.
13. **MINOR** `callback.py:37-66` — `--db-path` accepted but `"materials.db"` hardcoded for both database and engine.
14. **MINOR** `monitor_workflow.py:12-20` — imports `material_monitor`/`material_database` as top-level modules; neither resolves → tool always exits "modules not found".
15. **MINOR** `engine.py:1060` — early return makes the filesystem-sanitization block unreachable; names with spaces/slashes produce broken paths.
16. Note: duplicate method definitions in executor.py (`generate_calculation_configs` :547/:1354, `submit_slurm_job` :889/:1202) — shadowed dead code; the live versions happen to be correct.

## 4. Queue / submission / recovery

1. **CRITICAL** `recovery/recovery.py:573-584` — `create_recovery_calculation` passes `settings_json=` to `db.create_calculation`, which only accepts `settings:` → every recovery producing a fixed input dies with TypeError (swallowed, returns False).
2. **CRITICAL** `queue/manager.py:1176` — `resubmit_fixed_calculation` calls nonexistent `self.submit_single_calculation(...)` → recovered jobs never resubmitted (AttributeError swallowed).
3. **MAJOR** `queue/manager.py:818-823, 946-978` — completed/failed handlers receive `calc['calc_id']` but look it up by SLURM id, and the fallback scans pre-update status → both handlers no-op from the squeue path; error recovery is dead in normal operation.
4. **MAJOR** `queue/manager.py:1126, 1140` — recovery attempt counters call nonexistent `self.db.execute_query` → attempts always 0; `max_recovery_attempts` never enforced.
5. **MAJOR** `recovery/recovery.py:269`, `detector.py:472` — fix scripts resolved to `mace/Check_Scripts/` which doesn't exist (they're at `code/Check_Scripts/`) → shrink_error/updatelists handlers always fail.
6. **MAJOR** `submission/portable_slurm_generator.py:80-83` — generated callback runs `manager.py --work-dir .` but manager argparse only has `--d12-dir` → completion callback exits with argparse error code 2, never runs.
7. **MAJOR** error-type taxonomy mismatch — manager emits `time_limit`, recoverable list/config use `timeout_error`; detector emits `scf_convergence`/`disk_quota` vs config `convergence_error`/`disk_space_error` → those failures never recoverable.
8. **MAJOR** `recovery/recovery.py:188-194` — retry-count checks read keys that aren't columns (`parent_calc_id`, `is_recovery_attempt` live inside settings_json) → per-error `max_retries` never enforced.
9. **MAJOR** `recovery/recovery.py:628` — `json.loads(calc.get('settings_json', '{}'))` crashes on NULL column (returns None, not '{}'); `mace recover --action stats` TypeErrors on queue-manager-created calcs.
10. **MAJOR** `recovery/recovery.py:544-547` — disk-cleanup handler deletes `*.f*`/`fort.*` files >100MB, which includes the `.f9` wavefunctions needed for SP/BAND/DOSS restarts — recovery destroys restart data.
11. **MINOR** `submission/crystal.py:76`, `properties.py:77` — walltime validation regex rejects every documented example (`12:00:00` → False); interactive users trapped in a reprompt loop.
12. **MINOR** `queue/manager.py:1279` — `from input_settings_extractor import ...`: module exists nowhere in the repo; feature permanently dead.
13. **MINOR** `database/populate_completed_jobs.py:43-55` — BAND/DOSS outputs typed as "OPT" under `_band`/`_doss` material IDs (the unused `database/utils/` copy handles this correctly).
14. **MINOR** `recovery/recovery.py:317-371, 469-524` — memory/timeout handlers depend on `calc['job_script']` which nothing ever writes; recovery scripts they create are never submitted.
15. **MINOR** `queue/legacy_manager.py:166-186` — requires "Submitted batch job" from `submitcrystal23.sh` whose sbatch line is commented out; always returns False (module has no in-repo importers — candidate for retirement).

## 5. Utils (`mace/utils/`)

1. **CRITICAL** `property_extractor.py:99, 1211, 1294, 1953` — bare sibling imports (`population_analysis_processor`, `dat_file_processor`, `advanced_electronic_analyzer`) fail in every production context and are silently swallowed → population post-processing, BAND/DOSS .DAT processing, and advanced analysis never run. Fix: `from mace.utils.… import …` (with fallback).
2. **CRITICAL** `property_extractor.py:297-301` — `DIRECT ENERGY BAND GAP` regex matches inside "**IN**DIRECT ENERGY BAND GAP" → spurious `direct_band_gap` for every indirect-gap material (verified on 1_dia).
3. **MAJOR** `property_extractor.py:298, 304` — direct/indirect gaps take the FIRST match (initial SCF) while `band_gap` takes the last; classification then prefers the stale value (3.4^2T137_rev1: 3.22 eV used vs final 4.04).
4. **MAJOR** `property_extractor.py:775` — alpha-beta Mulliken section anchor matches the SPINLOCK echo line, storing full electron populations (~6 e/atom) as spin densities in all spin-locked outputs.
5. **MAJOR** `property_extractor.py:786` — Mulliken atom regex scans the whole section: 40 "atoms" for a 20-atom cell, 28 for diamond's 2 (A.O. + SHELL tables, repeated PPAN prints all counted).
6. **MAJOR** `property_extractor.py:705-707` — final-positions regex spans primitive + crystallographic tables → every atom doubled (`final_atoms_count` 4 for diamond's 2).
7. **MAJOR** `property_extractor.py:515-517` — `final_gradient_norm` takes the FIRST "GRADIENT NORM" (0.0678 stored where the real final is 0.00015).
8. **MAJOR** `property_extractor.py:406-420` — D3 dispersion and "TOTAL ENERGY + DISP" take the first occurrence while total energy takes the last → mutually inconsistent stored energies.
9. **MAJOR** `property_extractor.py:507-512` — `optimization_converged=False` for runs converging at point 1 (no "CONVERGENCE TESTS SATISFIED" printed; `OPT END - CONVERGED` should also count).
10. **MAJOR** `property_extractor.py:1136, 1794` — Fermi regex truncates scientific exponents (`-1.137E-01` → -1.137, 8.8× off) and the unit map labels Hartree values as eV.
11. **MAJOR** `property_extractor.py:1076-1081` — with `calc_id=None` (untracked folders) the dedup `WHERE calc_id = ?` never matches NULL → every re-extraction inserts a full duplicate property set.
12. **MAJOR** `file_manager.py:265-275` — `_infer_material_id_from_filename` corrupts names via mid-string `replace` (`..._optimized_rev1` → `...imized_rev1`) and is a fourth divergent ID scheme.
13. **MAJOR** ✅ FIXED (505f5ead) `dat_file_processor.py:232-242` — band analysis sets `max_occupied = max(all eigenvalues)` → every band structure classified metallic with `band_gap=None` (latent while finding #1 keeps this module unloadable). (Full rewrite: real BAND/DOSS formats, Fermi-referenced gap, robust DOS metallicity; verified on real outputs.)
14. **MINOR** `property_extractor.py:320, 361, 530, 1150`; `scf_settings_extractor.py:170` — confirmed never-match patterns (literal `'band_gap'`, `SPACE GROUP NUMBER`, `SCF FIELD CONVERGENCE`) — dead extraction paths.
15. **MINOR** — `show_properties.py:7` runs DB-creating code at import (no `__main__` guard); `material_database` imports in 3 files always fail; population processor expects `mulliken_population` keys but extractor emits `*_alpha_plus_beta` (bonding analysis never produced; 0.29 overlap diamond C–C would be labeled 'ionic'); FREQ enthalpy checks `zero_point_energy_au` before it's set; `initial_initial_*`/`final_final_*` junk keys from double-prefixing; settings_extractor writes `input_settings_json` but its own query reads `settings_json`.

## 6. Crystal_d12 (input generators)

1. **CRITICAL** ✅ FIXED (eb23aa59) `d12_calc_freq.py:1922` — phonon BANDS writes the `ISS NSUB` line before path resolution; later label-paths zero/rescale `shrink`, so the emitted deck pairs ISS=16 with label segments → CRYSTAL misparses. (BANDS header now emitted after the path/shrink is resolved.)
2. **CRITICAL** `CRYSTALOptToD12.py:907-918` — `--non-interactive` "auto" origin overrides the preserved origin ("0 0 1" for most space groups, "0 1 0" for 143–194); diamond Fd-3m re-emitted with origin-2 coords → wrong structure. Default to `settings["origin_setting"]`.
3. **CRITICAL** `NewCifToD12.py:1060-1069` — HF method + EXTERNAL basis never writes the geometry-closing `END` (DFT path does) → malformed deck.
4. **CRITICAL** `NewCifToD12.py:945-948` + `d12_constants.py:1707` — rhombohedral-axes trigonal CIFs: IFHR=1 directive emitted but cell line is `a c` instead of `a alpha` → alpha dropped, wrong cell.
5. **MAJOR** `d12_parsers.py:600-665` — functional extraction ignores the hybrid-exchange line: all B3LYP test outputs parse as "BLYP" (PBE0→PBE, etc.) → wrong functional re-emitted when no .d12 accompanies the .out.
6. **MAJOR** `d12_parsers.py:870-881` — TOLINTEG extraction `abs()`s the `10** 20` screening-disabled exponents → "7 7 20 20 20" re-emitted for pure-DFT runs.
7. **MAJOR** `d12_calc_basic.py:95,141` + `d12_constants.py:866` — OPT menu option 3 emits invalid keyword "ITATOCELL" (and constants copy has invalid "INTONLY"); CRYSTAL rejects the deck. Should be ATOMONLY.
8. **MAJOR** `NewCifToD12.py:359` — fallback CIF parser references undefined `Element` → NameError (uncaught) for every CIF ASE can't read.
9. **MAJOR** `CRYSTALOptToD12.py:619` — reads `settings["smearing"]` but the parser stores `use_smearing` → SMEAR silently dropped.
10. **MAJOR** `CRYSTALOptToD12.py:311-331` — ANHARM path emits a malformed block (geometry END placement inconsistent with the FREQCALC path).
11. **MAJOR** `NewCifToD12.py:1230-1250` — symmetry "CIF" + write-all-atoms keeps the non-P1 space group → CRYSTAL regenerates orbits → coincident atoms (fatal).
12. **MAJOR** `d12_constants.py:1934-1939` — EXTERNAL basis compatibility check misses absent elements (He, Ne, Ar, Kr, Xe, Po–Ra have no files) → deck written with NO basis for those elements.
13. **MAJOR** ⚠ PARTIAL FIX (c0844e3d) `d12_interactive.py:1837` (also 710, 1803) — "default advanced settings" hardcodes `spin_polarized=True` (SPIN injected into closed-shell re-runs — visible in `test/1_dia_opt_rev1.d12`); configured SPINLOCK is never written by any active writer. (SPIN injection fixed: defaults branch now derives from the source, CrystalInputParser inits spin_polarized=False; verified diamond→False, electrolytes/Ti→True. The SPINLOCK-not-written issue remains open.)
14. **MAJOR** `d12_constants.py:1701-1702` — monoclinic always emits `a b c beta` with no unique-axis detection; c-unique CIFs silently produce a wrong lattice.
15. **MINOR** bundle — `-D3-D3` doubled filename suffix (live artifact in test/) ✅ FIXED (ee56e9e1, added the missing "-D3" not-in-name guard); simplified `SHRINK k n` overwritten with raw string; `--output-dir` created but unused; `d12_from_config.py` wrapper passes a positional no target accepts (always argparse error); spurious MULTI_ORIGIN entry for SG 60; `Ghosts/create_d12_w-ghosts.py:66` ECP fixup writes wrong index.

## 7. Crystal_d3 (properties-input generators)

1. **CRITICAL** `d3_kpoints.py:1526-2628` — ~16 duplicate keys in `seekpath_data`; the later PRE-SCALED entries override the fractional ones and then get scaled AGAIN by the shrink factor (SG 191: M emitted as (36,0,0) with shrink 12 = 3× outside the BZ → band path folds 6×) whenever the seekpath library isn't installed. Delete the pre-scaled duplicates.
2. **MAJOR** `d3_kpoints.py:864-873` — orthorhombic centering map wrong: C-centered → body-centered table, I-centered → simple table (validated archived script and the labels function both agree on C/A→AB, I→BC).
3. **MAJOR** `d3_interactive.py:575-607` — interactive path passes descriptive strings ('cubic_fc') to a function matching single letters → FCC treated as simple cubic, rhombohedral as hexagonal; wrong-family paths in single-file mode.
4. **MAJOR** `CRYSTALOptToD3.py:464-478` — labels-mode NLINE counts segments across `|` discontinuities but emission doesn't → header claims one more segment than written; CRYSTAL reads END as a segment.
5. **MAJOR** `d3_kpoints.py:838-850` — invalid-label fallback emits fractional coordinate strings under an integer-shrink header (collapse to ~Γ), and unknown labels become `0 0 0` silently.
6. **MAJOR** `d3_kpoints.py:601-629` (commit 772d8495) — BAND_PATHS labels changed "G"→"GAMMA" and are written verbatim into label-mode .d3 segment lines; CRYSTAL's tables define single-letter labels (all validated archived templates use "G"). Map GAMMA→G at write time.
7. **MAJOR** `d3_interactive.py:1151` + `CRYSTALOptToD3.py:687-722` — DOSS energy window prompts "eV below/above Fermi" but writes BMI/BMA as absolute energies without adding E_F → window centered on 0, not the Fermi level.
8. **MAJOR** `d3_config.py:231-277` — `validate_d3_config` requires keys the configurators never produce → every saved DOSS/TRANSPORT config rejected on `--config-file` reload.
9. **MAJOR** ⚠ PARTIAL FIX (11a158dc, de9c40f8) `CRYSTALOptToD3.py:176`, `d3_interactive.py:1025` — atom-element regex can't match CRYSTAL's all-caps two-letter symbols ("SI", "TI") → atom projections rejected for any system with 2-letter elements; also double-counts atoms from repeated geometry blocks. (Both regex sites now fixed + verified on Ti13Pb3; only the geometry double-count remains open.)
10. **MINOR** `CRYSTALOptToD3.py:128` — electron-count regex says "PER UNIT CELL"; real outputs print "PER CELL" → always 0 (currently unused).
11. **MINOR** `d3_interactive.py:146-153` — "BOTTOM OF CONDUCTION BANDS" (real: "VIRTUAL BANDS") and `SPACE GROUP.*?NUMBER:` never match → dead-but-misleading fields.
12. **MINOR** `d3_kpoints.py:996-1058` — literature k-path references labels with no coordinates (rhombohedral Z/X, monoclinic C/D/E/Z…) → up to 6 of 13 segments silently dropped.
13. **MINOR** `seekpath_interface.py:126-213`, `d3_kpoints.py:2649` — structure parsed from the FIRST geometry in the .out (pre-optimization) → c/a-dependent variant selection and shrink rule can use the wrong cell.
14. **MINOR** `d3_interactive.py:1399` + `CRYSTALOptToD3.py:909-911` — POTC n_points double-negated (positive NPU emitted, CRYSTAL then expects inline coords that aren't there); atoms list collected but never written.
15. **MINOR** `d3_interactive.py:539-540, 615, 1130` — NameErrors when no output file is supplied; missing `|` markers in tI2/mC3 paths misrepresent emitted segments.

## 8. Plotting (`mace/plotting/`, `Plotting/`) and `code/`

1. **CRITICAL** `mace_cli:2792-2794` — same argument-scrambling bug: `mace plotting --dos --projection orbital` → `['orbital','--dos','--projection']`; only bare-flag invocations survive.
2. **MAJOR** `mace/plotting/main.py:382,767` — offers projection `tm_orb` which `ipDOS_V2.py:1488` rejects with `sys.exit(1)` (it's an element_mode there, not a proj_type).
3. **MAJOR** `mace/plotting/main.py:326,475` — `except Exception` doesn't catch the `SystemExit` the legacy plotters raise → one bad file aborts the whole batch and the mace process.
4. **MAJOR** `mace/plotting/main.py:111, 106` — discovery globs broader than what the plotters accept (`*.DOSS.DAT` vs required `*_doss.DOSS.DAT`; `*.BAND.DAT` vs `[._]band.band.dat`) → "No files found" + exit mid-batch.
5. **MAJOR** `mace/plotting/main.py:297` — `--alpha` forwarded only when ≠0.3, but the plotter's default is 1.0 → documented default renders opaque.
6. **MINOR** `main.py:290-315` — `sys.argv` restore not in `finally`; exception leaves argv clobbered process-wide.
7. **MINOR** `main.py:106` — case-sensitive globs miss real files like `*_SLAB_BAND.BAND.dat`.
8. **MINOR** `main.py:318` vs `ipBANDS_V2.py:950` — output-name round-trip mismatch (`mat_band.BANDS.svg` expected vs `mat.BANDS.svg` written) → "Generated:" never reported.
9. **MINOR** `main.py:280-313` — per-file loop × plotter-replots-everything → N files plotted N times each.
10. **MINOR** `main.py:499,875` — `parallel_jobs` config key never read.
11. **MINOR** `main.py:827-858` — CLI-mode configs omit `formats` → PNG outputs missing from reported file lists.
12. **MINOR** `main.py:244` — transparency prompt inverts matplotlib alpha semantics.
13. **MINOR** ✅ FIXED (fdf9cf8e) `Plotting/ipDOS_V2.py:960-965` — 17+-column unwrapped DOS data: every second row silently discarded by an unconditional `next(f)`. (Now pulls continuations only while the record is short; verified 50→100 rows on an unwrapped file, wrapped files unchanged. Stale `code/` copy left for the plotting-tree cleanup.)

### Defunct in `code/` (candidates for archiving/removal)
- `code/Plotting_Scripts/{ipBANDS_V2,ipDOS_V2,plottingCIFs,autoBands,autoPhononBands}.py` — stale older copies of `Plotting/` and `code/NewPlotting_Scripts/` versions.
- `code/Plotting_Scripts/OverviewPDF.py`, `code/Post_Processing_Scripts/grab_properties.py` — hardcoded per-user absolute paths; dead off those machines.
- `code/Plotting_Scripts/Archived/`, `code/Check_Scripts/Archived/`, `code/OldSLURMTemplates/` — superseded.
- `code/NewPlotting_Scripts/AutoDOS/` — stray run artifacts (`-100`, `materials.db-journal`).
- `mace/queue/legacy_manager.py` — no in-repo importers; submission path can't work with the shipped script.

## What's working well

- **Core single-process DB lifecycle**: schema, create/update calculation, workflow-state
  CRUD, WAL + locking, migrations — all verified sound. The extractor's own property
  INSERT path is correct.
- **Energy/structure extraction**: total energy (last-value), energy components, Ha→eV
  factors, initial/final lattice params, density, volumes, SHRINK, crystal system,
  space-group extraction — all verified numerically correct against the 11 real outputs.
- **The live workflow execution chain**: plan-file naming, `step_NNN_TYPE` conventions,
  expert-config paths, step-config keys, SLURM template globs — writer and reader agree.
  Completion detection (recently validated) untouched and consistent.
- **SLURM interaction**: sbatch/squeue parsing, state mapping, special characters in real
  material names (`^`, `,`, `.`) survive the generated scripts.
- **DOSS AO-projection pipeline**: verified exactly correct end-to-end on real output
  (36 AOs, index ranges, NPRO counts), and FCC/hexagonal/tetragonal k-point tables match
  CRYSTAL's documented tables.
- **d12 generation main paths**: element/space-group tables complete and consistent;
  FINAL OPTIMIZED GEOMETRY extraction exact on all test outputs; keyword ordering on the
  main DFT paths matches validated decks; settings merge prefers .d12 correctly.
- **Plotting physics**: Ha→eV, Fermi/vacuum alignment, DOS scaling all consistent across
  the band/DOS plotters; mace→plotter flag wiring is complete (once the argv scrambling
  is fixed).
