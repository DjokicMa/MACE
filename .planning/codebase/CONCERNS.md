# Codebase Concerns

**Analysis Date:** 2026-06-13

> **Source of truth:** This document synthesizes the live `CODEBASE_AUDIT.md` (340 lines, 2026-06-12,
> produced by 8 parallel agents against real `test/` outputs) plus direct code inspection.
> Severity ratings from the audit: **CRITICAL** = feature always broken or data corrupted;
> **MAJOR** = broken under common usage; **MINOR** = edge case or cosmetic.
> Items marked **FIXED** were resolved in the two fix waves committed on 2026-06-12.

---

## Cross-Cutting Root Causes

These five patterns drive the majority of critical findings across all modules:

**1. Bare sibling imports that fail under package layout:**
- Modules inside `mace/utils/` import siblings as top-level names (e.g. `from population_analysis_processor import ...`) — these fail in every production context and are silently swallowed by `try/except`, permanently disabling whole features.
- Files: `mace/utils/property_extractor.py:99`, `mace/utils/property_extractor.py:1211`, `mace/utils/property_extractor.py:1294`, `mace/utils/property_extractor.py:1953`

**2. Divergent material-ID derivation (at least 4 implementations):**
- `mace/database/materials.py` (`create_material_id_from_file`) — canonical
- `mace/database/populate_completed_jobs.py:40-46` — strips only exact `_opt/_sp/_freq`; `mat_opt2.out` → `mat_opt2` vs canonical `mat`
- `mace/utils/file_manager.py:265-275` (`_infer_material_id_from_filename`) — mid-string `replace` corrupts names
- `mace/database/utils/create_fresh_database.py:311-331` — `name_opt_BULK_OPTGEOM_*` → `name_opt`
- Impact: one material splits across multiple DB identities; completion scans never match submitted records.

**3. First-vs-last occurrence parsing in property extractor:**
- Several extractors take the first regex match in a file that prints a quantity repeatedly during optimization (direct/indirect gap, gradient norm, D3 dispersion, seekpath geometry), storing pre-optimization values.
- Files: `mace/utils/property_extractor.py:298,304,515-517,406-420`

**4. Writer/reader key mismatches:**
- One module writes a key/column name; another reads a different one.
- Known pairs: `queue_management` (planner writes top-level, executor reads `execution_settings.queue_management`); `input_settings_json` vs `settings_json` (settings extractor); expected property names in missing-data analysis; recovery error-type taxonomy aliases.

**5. Pervasive bare `except:` clauses swallowing real errors:**
- 48 bare `except:` instances and 255 broad `except Exception:` instances across the codebase.
- Key files: `mace/utils/property_extractor.py:1072`, `mace/queue/manager.py:382,395`, `mace/workflow/engine.py:2853`, `mace/workflow/executor.py:2341`, `Crystal_d12/d12_parsers.py:384,448,1005,1012,1031,1037`
- Impact: AttributeErrors and TypeErrors from broken API calls are silently swallowed, making debugging near-impossible.

---

## Tech Debt

**Non-package module layout (Crystal_d12, Crystal_d3, Plotting):**
- Issue: `Crystal_d12/`, `Crystal_d3/`, and `Plotting/` have no root `__init__.py`, making them not importable as Python packages. All cross-module imports require `sys.path.insert(0, ...)` workarounds scattered across 20+ files.
- Files: `mace/run_mace.py:39-40`, `Crystal_d12/d12_calc_freq.py:24,417,815`, `mace/workflow/executor.py:1892`, `Crystal_d3/CRYSTALOptToD3.py:57`, `mace/workflow/dummy_file_creator.py:24`, `mace/plotting/main.py:264,419,536`
- Impact: Import order is fragile; adding `Crystal_d12` to `sys.path` at module level can shadow stdlib modules; subtle bugs when scripts are run from different working directories.
- Fix approach: Add `__init__.py` to `Crystal_d12/` and `Crystal_d3/`; convert all internal imports to `from Crystal_d12.d12_constants import ...` style; eliminate `sys.path` manipulation.

**Duplicate plotting scripts (diverged copies):**
- Issue: `Plotting/ipBANDS_V2.py` and `code/Plotting_Scripts/ipBANDS_V2.py` have 1452-line diff; `Plotting/ipDOS_V2.py` and `code/Plotting_Scripts/ipDOS_V2.py` have 1796-line diff. They are no longer the same file.
- Files: `Plotting/ipBANDS_V2.py`, `Plotting/ipDOS_V2.py`, `code/Plotting_Scripts/ipBANDS_V2.py`, `code/Plotting_Scripts/ipDOS_V2.py`
- Impact: Bug fixes applied to `Plotting/` are not reflected in `code/Plotting_Scripts/` and vice versa. The canonical version is unclear.
- Fix approach: Delete `code/Plotting_Scripts/{ipBANDS_V2,ipDOS_V2}.py` and treat `Plotting/` as the single source of truth.

**Legacy manager with no in-repo importers:**
- Issue: `mace/queue/legacy_manager.py` is a dead module — no file in the repo imports it. Its submission path requires "Submitted batch job" from `submitcrystal23.sh` whose sbatch line is commented out, so it always returns False anyway.
- Files: `mace/queue/legacy_manager.py`
- Fix approach: Archive or delete.

**Contextual/isolated workflow variants not wired to real APIs:**
- Issue: `mace/workflow/executor_contextual.py` (292 lines) and `mace/workflow/planner_contextual.py` (222 lines) were written against a `WorkflowContext` API that doesn't exist (`get_active_context` vs `get_active`, `cleanup_on_exit=`, `ctx.get_database()`, etc.); `os` used without import.
- Files: `mace/workflow/executor_contextual.py`, `mace/workflow/planner_contextual.py`
- Impact: Every contextual/isolated workflow path fails at import or runtime.
- Fix approach: Either implement the `WorkflowContext` API to match these callers, or rewrite callers to use the real API.

**`code/` directory is largely dead:**
- Issue: `code/Plotting_Scripts/`, `code/Post_Processing_Scripts/grab_properties.py` (hardcoded `/home/daniel/` paths), `code/Plotting_Scripts/Archived/`, `code/OldSLURMTemplates/` are all superseded.
- Files: `code/Plotting_Scripts/`, `code/Post_Processing_Scripts/grab_properties.py:22`, `code/OldSLURMTemplates/`
- Fix approach: Archive entire `code/` subtree or move `code/NewPlotting_Scripts/` contents elsewhere and delete `code/`.

**`setup_mace.py` --update-templates targets non-existent directory:**
- Issue: `--update-templates` targets `code/Job_Scripts/` which doesn't exist (live templates live in `mace/submission/*.sh`).
- Files: `setup_mace.py:239-242,369`

**`__pycache__/` files committed in Archived subdirectory:**
- Issue: `Crystal_d12/Archived/__pycache__/*.pyc` are tracked in git, matching Python 3.14 bytecode. They will break on any other Python version and should never be committed.
- Files: `Crystal_d12/Archived/__pycache__/` (5 `.pyc` files)
- Fix approach: `git rm -r Crystal_d12/Archived/__pycache__/`; `.gitignore` already excludes `__pycache__/` for non-Archived dirs.

**Embedded debug `print()` statements in production code:**
- Issue: 16 `print(f"DEBUG: ...")` calls and 5363 total `print()` calls throughout production modules. Debug prints in hot paths slow execution and obscure real output.
- Files: `mace/workflow/engine.py:98,99,106,109,997,1872,1873,1876,1886,2230,2231,2237,2242,2247,3084`
- Fix approach: Replace with `logging.debug()`; adopt `logging` framework project-wide.

**Unimplemented stubs with TODO comments that silently return defaults:**
- Issue: Two Brillouin-zone variant selectors always return the same answer regardless of input (simplified logic placeholders). Two TODO functions in `queue/manager.py` only print a message.
- Files: `Crystal_d3/d3_kpoints.py:274` (`determine_orthorhombic_i_variant` → always `"oI1"`), `Crystal_d3/d3_kpoints.py:292` (`determine_orthorhombic_s_variant` → always `"oS1"`), `mace/queue/manager.py:1349,1419`
- Impact: Wrong k-path variant selection for oI and oS structures; queue auto-progression and property extraction hooks do nothing.

---

## Known Bugs

> Items marked **FIXED** were resolved in fix waves committed 2026-06-12. Remaining open items are listed below.

**BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL submissions were dead (FIXED):**
- `engine.py:1574` called `self._submit_slurm_job(...)` (wrong name); and `self.db.update_calculation(...)` (wrong name). AttributeErrors silently swallowed — D3 steps never submitted.
- Fixed: renamed to `_submit_calculation_to_slurm` and `update_calculation_status`.

**OPEN — Duplicate pre-scaled seekpath fallback k-point table (§7.1):**
- Symptoms: For some space groups (e.g. SG 191), when `seekpath` library is absent, band-path k-points are emitted 3× outside the Brillouin zone.
- Files: `Crystal_d3/d3_kpoints.py:1526-2628`
- Trigger: Running without the `seekpath` library installed (system Python, not Anaconda).
- Note: `seekpath` IS present in `/home/marcus/anaconda3/`, so this only fires on the compute nodes if they inherit the wrong Python.

**OPEN — Monoclinic unique-axis detection missing (§6.14):**
- Symptoms: Monoclinic CIFs always emit `a b c beta` cell format with no unique-axis check; c-unique monoclinic CIFs produce a silently wrong lattice parameter ordering.
- Files: `Crystal_d12/d12_constants.py:1701-1702`, `Crystal_d12/NewCifToD12.py`

**OPEN — Plotting batch aborts on `SystemExit` (§8.3):**
- Symptoms: One bad file during `mace plotting --batch` calls `sys.exit(1)` inside `ipDOS_V2.py`; `except Exception` in the caller does not catch `SystemExit`, aborting the entire batch.
- Files: `mace/plotting/main.py:326,475`

**OPEN — Plotting file glob mismatches (§8.4):**
- Symptoms: Discovery globs `*.DOSS.DAT` / `*.BAND.DAT` don't match the actual file naming patterns `*_doss.DOSS.DAT` / `[._]band.band.dat` → "No files found" mid-batch.
- Files: `mace/plotting/main.py:111,106`

**OPEN — `sys.argv` not restored in finally block (§8.6):**
- Symptoms: An exception inside the plotting dispatch path leaves `sys.argv` permanently clobbered for the rest of the process.
- Files: `mace/plotting/main.py:290-315`

**OPEN — `materials_contextual.py` crashes on every call (§2.4-2.5):**
- `copy_to_context` calls nonexistent `create_or_update_material`; `get_workflow_materials`/`get_workflow_calculations` call `self.get_connection()` (real: `_get_connection`).
- Files: `mace/database/materials_contextual.py:166,208-215,257,293`

**OPEN — `monitor_workflow.py` always exits (§3.14):**
- Imports `material_monitor`/`material_database` as top-level modules; neither resolves → tool always exits "modules not found" before doing anything.
- Files: `mace/workflow/monitor_workflow.py:12-20`

**OPEN — DOSS energy window uses absolute energies instead of Fermi-relative (§7.7, PARTIALLY FIXED):**
- `d3_interactive.py:1151` + `CRYSTALOptToD3.py:687-722` prompt "eV below/above Fermi" but in some paths write BMI/BMA as absolute energies without adding E_F.
- Files: `Crystal_d3/d3_interactive.py:1151`, `Crystal_d3/CRYSTALOptToD3.py:687-722`

**OPEN — POTC block double-negated and atoms never written (§7.14):**
- Positive NPU emitted (CRYSTAL expects negative for number of points); atoms list collected but never written to the block.
- Files: `Crystal_d3/d3_interactive.py:1399`, `Crystal_d3/CRYSTALOptToD3.py:909-911`

**OPEN — No follow-up branch for TRANSPORT/CHARGE+POTENTIAL in engine (§3.10):**
- Custom sequences with steps after TRANSPORT or CHARGE+POTENTIAL stall permanently (no completion handler advances the chain).
- Files: `mace/workflow/engine.py:1873-2175`

**OPEN — `validate_materials` is a pass stub (§2.8):**
- `materials.py:validate_materials(material_ids=[...])` body is `pass`; the interactive explorer's `do_validate` always hits a KeyError in the formatter.
- Files: `mace/database/utils/validation.py:626-647`

**OPEN — `cleanup_old_records` always raises ValueError (§2.7):**
- `datetime.now().replace(day=day-30)` crashes on most invocations. Should use `timedelta`.
- Files: `mace/database/materials.py:635`

**OPEN — `analysis/missing_data.py` reports all calcs as missing (§2.9):**
- Expected property names (`total_energy`, `final_a`) don't match what the extractor writes (`total_energy_au`, `final_primitive_a`).
- Files: `mace/database/analysis/missing_data.py:16-48`

**OPEN — `analysis/aggregation.py` groups nothing (§2.11):**
- `group_by='conductivity_type'/'energy_range'` look up property names the extractor never writes; everything lands in "Unknown".
- Files: `mace/database/analysis/aggregation.py:149,185`

**OPEN — `ipDOS_V2.py` silently discards every other row for wide DOS data (§8.13):**
- 17+-column unwrapped DOS data: an unconditional `next(f)` skips every second row.
- Files: `Plotting/ipDOS_V2.py:960-965`

---

## Security Considerations

**`shell=True` subprocess calls with externally-derived input:**
- Risk: Both shell=True calls pass strings constructed from external data (a node-type argument from config, and a job submission script path). If those strings contain shell metacharacters, command injection is possible.
- Files: `mace/utils/node_exclusion.py:135` (node type name in scontrol query), `mace/queue/legacy_manager.py:169` (job submission command string)
- Current mitigation: Neither input is validated or sanitized before shell interpolation.
- Recommendation: Replace with list-form `subprocess.run([...])` without `shell=True`.

**Hardcoded institutional paths in production workflow code:**
- Risk: If the repository is used outside the Mendoza group HPC cluster, fallback basis-set paths silently fail to find basis files and may write incomplete `.d12` decks.
- Files: `mace/workflow/executor.py:2026,2035` (`/mnt/research/mendozacortes_group/...`), `mace/workflow/submitcrystal23.sh:12` (`#SBATCH -A mendoza_q`), `mace/workflow/submit_prop.sh:13` (`#SBATCH -A mendoza_q`)
- Current mitigation: Fallback chain checks `./` and `../` after the hardcoded path.
- Recommendation: Make the basis-set search path configurable via `mace_config.py` or `~/.mace/config.yaml`.

**252 `open()` calls without `encoding=` specification:**
- Risk: On systems where `locale.getpreferredencoding()` is not UTF-8, reading CRYSTAL output files (which contain non-ASCII symbols in symmetry labels etc.) will silently corrupt or raise UnicodeDecodeError.
- Files: Widespread across `Crystal_d12/`, `Crystal_d3/`, `mace/utils/property_extractor.py`
- Recommendation: Add `encoding='utf-8'` to all file opens that read CRYSTAL output.

---

## Performance Bottlenecks

**Plotting batch runs N files × N times each (§8.9):**
- Problem: `mace/plotting/main.py:280-313` outer loop iterates over files, but each inner plotter call re-plots every file it discovers in the directory. N files → N² plot operations.
- Files: `mace/plotting/main.py:280-313`
- Cause: Plotters (`ipBANDS_V2.py`, `ipDOS_V2.py`) do their own directory scan via `sys.argv` manipulation rather than accepting a single target file.
- Improvement path: Refactor plotters to accept explicit file paths; call each once.

**Property extractor re-reads entire file for each property section:**
- Problem: `mace/utils/property_extractor.py` applies dozens of independent `re.search()`/`re.findall()` calls over the full file content, each O(N) in file size. A 100MB FREQ output is scanned ~50 times.
- Files: `mace/utils/property_extractor.py`
- Improvement path: Single-pass extraction using a state machine or compiled master regex.

**SQLite WAL mode is set but threading discipline is mixed:**
- Problem: `mace/database/materials.py:253-269` correctly opens WAL + 30s busy timeout. However, `mace/workflow/context.py:129,141` opens a second per-workflow SQLite connection (no WAL pragma set there) and `code/NewPlotting_Scripts/AutoDOS/autoDOS.py:123,693` opens SQLite connections with no WAL/timeout.
- Files: `mace/workflow/context.py:129,141`, `code/NewPlotting_Scripts/AutoDOS/autoDOS.py`
- Improvement path: Centralize all DB access through `materials.py`'s `_get_connection()` context manager.

---

## Fragile Areas

**Property extractor regex patterns (validated core is sound; edges are fragile):**
- Files: `mace/utils/property_extractor.py`
- Why fragile: Many patterns match the first occurrence in a file that prints values repeatedly during optimization (FIXED for the main cases in fix wave 2). Several patterns are literal strings that never match real output (`'band_gap'`, `SPACE GROUP NUMBER`, `SCF FIELD CONVERGENCE`).
- Safe modification: Always test against real outputs in `test/*.out` (11 families covering OPT, SP, BAND, DOSS, FREQ, TRANSPORT). Never add patterns without verifying on at least one real file.
- Test coverage: No automated test suite — all verification is manual against `test/` outputs.

**`mace/workflow/engine.py` (3587 lines) — workflow state machine:**
- Files: `mace/workflow/engine.py`
- Why fragile: Step-config lookup uses substring matching (`calc_type in step_key` at line 162) that causes "SP" to match "TRANSPORT_3". Callback completion detection uses hardcoded step-number arithmetic for D3 positions (line 1545). Threading lock is RLock, but the lock does not cover DB calls.
- Safe modification: Any change to step naming or numbering requires checking all four: `_find_calc_position`, `_create_and_submit_d3_calculation`, the step-key lookup, and the plan-file reader.

**`mace/workflow/planner.py` (5006 lines) — largest file:**
- Files: `mace/workflow/planner.py`
- Why fragile: Three deferred `from mace.workflow.executor import WorkflowExecutor` calls at lines 4553, 4592, 4653 — if that import fails, the error surfaces only at runtime when those branches are reached. Import fallback chain (`try: from .engine import WorkflowEngine` / `except: from mace.workflow.engine import ...`) masks import errors.
- Safe modification: Any refactor of `executor.py`'s public API must update all three import sites.

**`Crystal_d12/d12_parsers.py` (1359 lines) — validated mace detection/parsing:**
- Files: `Crystal_d12/d12_parsers.py`
- Why fragile: Six bare `except:` clauses at lines 384, 448, 1005, 1012, 1031, 1037. The core mace detection and geometry parsing logic is validated — do not rewrite. Layer new parsing behavior on top.
- Test coverage: Verified against `test/*.out` real outputs. Add new patterns as non-destructive additions.

**SLURM submission templates (hardcoded for one HPC cluster):**
- Files: `mace/workflow/submitcrystal23.sh`, `mace/workflow/submit_prop.sh`
- Why fragile: Account (`mendoza_q`), module names (`CRYSTAL/23-intel-2023a`, `Python/3.11.3-GCCcore-12.3.0`), and scratch path (`$SCRATCH/crys23`) are hardcoded. Any cluster migration requires editing both scripts.
- Safe modification: Parameterize via the SLURM config in `mace/config/` rather than editing templates directly.

---

## Scaling Limits

**Single SQLite database for all workflow state:**
- Current capacity: Adequate for ~thousands of calculations (WAL mode, 30s timeout).
- Limit: With many concurrent queue-manager polling loops all writing to the same SQLite WAL file, lock contention will increase. SQLite has an effective write-concurrency limit of ~1 writer/second.
- Scaling path: Partition by workflow ID into separate DB files, or migrate to PostgreSQL for large parallel campaigns.

**No test infrastructure — all verification is manual:**
- Current capacity: 11 real output families in `test/`, verified manually.
- Limit: Any regression in parsing or workflow logic goes undetected until a live run fails.
- Scaling path: Add pytest suite using `test/*.out` files as fixtures (already mandated by project memory: "verify parsers against test/*.out, not synthetic fixtures").

---

## Dependencies at Risk

**`seekpath` commented out in `requirements.txt`:**
- Risk: The requirement line is present but commented: `# seekpath`. Any environment built from `requirements.txt` will lack `seekpath`, triggering the broken fallback k-point table (§7.1 above).
- Impact: Wrong k-paths for all structures when running with the system Python (not Anaconda).
- Migration plan: Uncomment the `seekpath` line; pin version (`seekpath>=2.2.1`).

**Scientific stack only available in Anaconda environment:**
- Risk: `ase`, `spglib`, `seekpath`, `matplotlib` exist only at `/home/marcus/anaconda3/bin/python`. SLURM callback scripts that call bare `python` will inherit the compute node's system Python and silently lose all scientific functionality.
- Files: `mace/workflow/submitcrystal23.sh`, `mace/workflow/submit_prop.sh`, `mace/workflow/callback.py`
- Migration plan: Pin callback and submission scripts to the Anaconda Python path explicitly (per project memory: always use `/home/marcus/anaconda3/bin/python`).

**`PyPDF2` in requirements.txt:**
- Risk: `PyPDF2` is deprecated upstream (superseded by `pypdf`). API breakage expected in future pip upgrades.
- Files: `requirements.txt`
- Migration plan: Replace with `pypdf>=3.0.0`.

**No version pins on any requirement:**
- Risk: `requirements.txt` uses only `>=` lower bounds. A breaking change in any dependency (numpy, matplotlib, ase) will silently affect all environments.
- Files: `requirements.txt`
- Migration plan: Pin to tested upper bounds (`numpy>=1.21.0,<3.0`); or use `pip-tools`/`conda lock` files.

---

## Missing Critical Features

**No automated test suite:**
- Problem: Only one test file exists in the active codebase (`Crystal_d3/test_seekpath_interface.py`). All correctness verification is done manually against `test/*.out` real outputs.
- Blocks: Safe refactoring of parsers, database, and workflow engine.
- Priority: High — any regression is invisible until a live computation fails.

**No centralized logging framework:**
- Problem: 5363 `print()` calls serve as the only diagnostic output. There is no log level control, no log file, and no structured output.
- Blocks: Debugging silent failures caused by swallowed exceptions.
- Priority: Medium.

**`show_properties.py` executes DB-creating code at import:**
- Problem: `mace/database/interactive/show_properties.py:7` runs database creation code at module import time (no `if __name__ == '__main__':` guard). Any import of this module in a test or library context will create/modify the database.
- Files: `mace/database/interactive/show_properties.py:7`
- Priority: Medium.

**CrystalOutToCif priority-based geometry selection not implemented:**
- Problem: `Crystal_d12/CrystalOutToCif.py:244` has `# TODO: Implement priority-based geometry selection` — the function returns the raw parser geometry with no selection logic.
- Files: `Crystal_d12/CrystalOutToCif.py:242-245`
- Priority: Low (current behavior is functional for single-geometry outputs).

---

## Test Coverage Gaps

**Workflow end-to-end regression:**
- What's not tested: The full `mace workflow --quick-start` chain (OPT → SP → BAND/DOSS) with a real CRYSTAL output, without mocked SLURM. The 19-sandbox progression matrix was done with mocked SLURM.
- Files: `mace/workflow/engine.py`, `mace/workflow/executor.py`, `mace/workflow/planner.py`
- Risk: Any change to step naming, numbering, or SLURM template selection breaks the chain silently.
- Priority: High.

**Property extractor parsing correctness:**
- What's not tested: All 637 `test/*.out` outputs against expected property values. Current verification is manual spot-check.
- Files: `mace/utils/property_extractor.py`
- Risk: Regex regressions go undetected; stored energies/geometries corrupted silently.
- Priority: High.

**Database round-trips:**
- What's not tested: `create_material_id_from_file` consistency across all four implementations; `store_material_property` under concurrent writes; `cleanup_old_records` with dates near month boundaries.
- Files: `mace/database/materials.py`, `mace/database/populate_completed_jobs.py`, `mace/utils/file_manager.py`, `mace/database/utils/create_fresh_database.py`
- Priority: High.

**Crystal_d12 / Crystal_d3 input generation:**
- What's not tested: Generated `.d12`/`.d3` decks validated against CRYSTAL's parser. Only a few families in `test/` provide reference inputs.
- Files: `Crystal_d12/*.py`, `Crystal_d3/*.py`
- Risk: Malformed decks (wrong keyword ordering, missing END, wrong cell format) are submitted to HPC and fail silently after hours of queue time.
- Priority: High.

---

*Concerns audit: 2026-06-13*
