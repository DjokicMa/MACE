# MACE Remediation Plan & Resume Anchor

Single source of truth for outstanding work. Updated 2026-06-13. If context is
compacted, **start here**, then `git log --oneline -40` to see what's committed.

Constraints (always): verify every fix against **real `test/` outputs** (never synthetic);
use `/home/marcus/anaconda3/bin/python`; "don't fix what works" (layer, don't rewrite);
`test/` is gitignored — never commit it; commit but don't push unless asked.

Verification habit (a prior-commit lesson): several past commits shipped "verified" claims
that did NOT reproduce. Re-run each fix against real data and don't overclaim.

---

## NEXT UP — remaining stragglers in priority order (start here after compaction)

The campaign's bug-fix + dead-code waves are DONE (see "DONE" + the dated commit log). Full
evidence for each item below is in "PERSISTING ISSUES" near the bottom of this file. When the
user opts into a panel, analyze with the Workflow tool first (it caught 2 wrong audit claims).
Add/extend a pytest test in `tests/` for every fix. Suggested order:

1. ContextualMaterialDatabase trio (LATENT, cheap, 1 commit) — materials_contextual.py:
   `copy_to_context` -> rename `create_or_update_material` to `create_material`;
   `get_workflow_materials`/`get_workflow_calculations` (L257/293) -> `_get_connection`;
   and `store_material_property` (materials.py:1114-1145) -> drop the explicit `property_id`
   so the INTEGER PK autoincrements (it currently inserts str(uuid4) -> datatype mismatch).
   All dormant today (no live callers), so no behavior change risk; fix before those features
   are wired up.
2. Dead `input_settings_extractor` import (HIGH, quick) — queue/manager.py:1390: remove it or
   implement the module; feature silently never runs (swallowed ImportError).
3. `mace submit` job-id capture (HIGH) — submission/crystal.py:246,362 + properties.py:243,359
   use os.system; switch to subprocess.run + "Submitted batch job (\d+)" regex (mirror
   manager.submit_to_slurm) so untracked submissions are trackable/recoverable.
4. node_exclusion.py:132-135 shell=True injection (MEDIUM) — split the `| grep` pipe into two
   subprocess calls (or filter in Python); node_type is unsanitized on a live path.
5. MEDIUM cleanups: aggregation.py:149 conductivity_type grouping; crystal-system/space-group
   dup (d3_kpoints.py:862-899 vs d12_constants.py); hardcoded basis paths executor.py:2026,2035.
6. STRATEGIC (larger, panel + incremental, test-first): shared CRYSTAL `.out` parser / Fermi /
   constants module; coverage for engine.py + planner.py; delete the remaining dead files
   (mace_config.py, portable_slurm_generator.py, Crystal_d12/Archived, Crystal_d3/Archived,
   code/Check_Scripts/Archived) — KEEP enhanced_queue_manager.py (required shim).

Do NOT re-fix the "Already FIXED but docs are stale" list at the bottom (CONCERNS.md is stale).

---

## DONE (committed this campaign)

- Sixth wave: monoclinic unique-axis detect (`ba6f4664`), SPINLOCK writer (`8d370021`),
  d3 geometry double-count (`a876d9f0`).
- Version 1.0.5 + `mace submit` defaults to cwd (`d498488f`).
- Back-nav: menu_nav controller (`5cac5be0`), opt2d12 (`f44c01b7`), opt2d3 shared (`11bbc6e2`);
  `mace manager` in-place opt-out (`b040648b`).
- Wave A (from the full-commit multi-agent review): `-D3-D3` guard in CRYSTALOptToD12
  (`733c3efb`), formula_extractor BASISSET reset (`43820ded`), d12_config Tuple import
  (`098228cb`), missing-data electronic_classification→optional + aggregation TypeError
  (`0d52a696`).
- Fix 6 (back-aware crash-safe prompts): nav_int/nav_float/nav_choice + opt2d12 OPT config
  (`a59e5007`), d12_calc_freq ~95 prompts (`2e115b15`), d3_interactive ~59 prompts (`d6b7eb93`).
  Planner inherits via the wrapped per-step configs. Tests: `Crystal_d12/test_menu_nav.py`,
  `Crystal_d12/test_back_integration.py`.
- gCP + corrected total + molecular FREQ thermo + enthalpy (`fb173e82`), all in
  `mace/utils/property_extractor.py`, verified against real `test/` outputs:
  - `gcp_energy_au/ev` added; `total_energy_corrected_au/ev` = total + D3 + gCP from the
    FINAL-geometry components. NOTE: we do NOT prefer CRYSTAL's printed
    `TOTAL ENERGY + DISP + GCP` line — in an OPT it is the INITIAL geometry and never
    re-printed after OPT END (4LG OPT was stale by 0.094 AU). Verified on 226 HSESOL3C
    outputs: corrected ≡ total+D3+gCP exactly; SP matches printed line to ~1e-10.
    `total_energy_plus_d3_au` unchanged for back-compat (only `missing_data.py` reads it).
  - BONUS BUG FOUND + FIXED: molecular FREQ runs extracted ZERO thermodynamics — CRYSTAL
    labels the block `...TAKING INTO ACCOUNT MOLECULAR ...` for molecules vs
    `...WITH VIBRATIONAL CONTRIBUTIONS` for periodic; regex matched only the latter.
    Broadened → Gibbs/ET/PV/TS now extracted on all 95 FREQ outputs (were missing on every
    molecular one — the bulk of the electrolyte set).
  - `enthalpy_au/ev/kj_mol` = Gibbs + TS (== EL+E0+ET+PV); `electronic_energy_au/ev/kj_mol`
    (EL line). Removed prior enthalpy block (dead code: gated on `zero_point_energy_au`
    before it was set; mislabeled the thermal correction as enthalpy). H verified exact
    (==Gibbs+TS==EL+E0+ET+PV, 0 error over 95 files).

(minor, still open) OPT energy *components* (`_extract_energy_components`) use `re.search` (first
`+++ ENERGIES IN A.U. +++` block); for multi-step OPT they can be from an early cycle while total
is from the last. Take the last block. Low priority — components are diagnostic, not the total.

---

## WAVE B — DONE (both committed; diagnoses partly corrected during the fix)

### B1 — Error-recovery chain now resubmits the FIX (CRITICAL) — commit `acf632c9`
Confirmed by inspection: recovery "succeeded" but `resubmit_fixed_calculation` submitted the
ORIGINAL `calc['input_file']`, and `submit_to_slurm` regenerated the job script from the template
(discarding any bumped `--mem`/`--time`). Both fix classes were dropped.
- Handlers now return `(fixed_input, fixed_job_script)`; `attempt_recovery` returns the artifacts
  dict (+ `create_record` so the manager path makes no orphaned recovery row).
  `resubmit_fixed_calculation(calc, fixed_input, fixed_job_script)` submits the fix;
  `submit_calculation`/`submit_to_slurm` gained an override to honor a bumped script.
- `convergence_handler` is now OPTGEOM-aware (bumps only the SCF MAXCYCLE; verified on a real
  dual-MAXCYCLE OPT d12). Taxonomy unified: `TOO MANY SCF CYCLES` → `TOO MANY CYCLES`.
- Verified with mocked SLURM: resubmission carries SCF MAXCYCLE=1800 (the fix), not 800; memory
  bump threads the 120GB script; no-fix path still falls back to the recorded input.
- CAVEAT (unchanged): `test/` has ZERO failed CRYSTAL outputs, so the real-failure taxonomy
  remains unverified against actual failures. The organized-mode resubmit (copy to a new
  calc_dir) + bumped-script-cwd interaction is also untested against real SLURM.

### B2 — BAND.DAT gap fixed via band-index method (HIGH) — commit `4a1b6194`
The review's stated cause (NBND line-wrapping) did NOT reproduce — every record is one physical
line of 1+NBND tokens. The REAL cause: BAND.DAT eigenvalues are Fermi-REFERENCED (E - E_Fermi,
Fermi at 0), but `_analyze_band_structure` treated them as absolute and split at the .out's
absolute Fermi → gap ~6x too small (1.45 eV vs true 7.8). Fixed by computing the gap by BAND
INDEX (VBM=max_k E[N-1], CBM=min_k E[N]); N from "TOP OF VALENCE BANDS - BAND N" or
electrons//2. Referencing-independent; matches each band .out's own gap to <0.01 eV. Without N,
gap/VBM/CBM are left None (never guessed). Verified on 78 BAND.DAT (insulators 7.2-9.2 eV,
4LG metals 0.0, semiconductors 2-4 eV). Note: this is the BAND-PATH gap (matches the band .out);
it can legitimately differ from the SP-mesh gap for near-zero-gap systems.

---

## BACKLOG (lower priority, from the review — fix opportunistically)

- ~~D3 workflow continuation halts on the `'manual'` workflow_id fallback~~ — DONE (`7f82f01f`):
  derives the id from `get_workflow_output_base(parent_calc).name`; also hardened that helper's
  NULL `settings_json` crash. Tests added.
- ~~OPT continuation drops a configured non-zero SPINLOCK~~ — DONE (`7260581a`):
  `_extract_spinlock_settings` parses `SPINLOCK\n<n> [<cycles>]` (both forms) and sets
  spin_polarized for a non-zero lock; verified round-trip + tests.
- ~~Public d12 wrappers lose their signature~~ — DONE (`593a09e4`): `functools.update_wrapper`
  on `configure_single_point/optimization` + `get_frequency_configuration` (sets `__wrapped__`,
  keeps the public name).
- `mace manager` in-place default flips silently: print a one-line notice; add `--organize`
  to `mace manager --help` (mace_cli help block ~767-773); confirm before bare `mace submit`
  on a populated dir. LOW.
- ~~`mace submit --track` opt-in DB tracking~~ — DONE (`0c650a27`): routes through
  EnhancedCrystalQueueManager(enable_tracking=True, organize_outputs=False), in-place; default
  unchanged (manager/DB built only under --track); guards --nosubmit/node-exclusion; tests added.
  Panel-analyzed before implementation.
- Full cross-step planner orchestrator back (risky — interleaves JSON writes / subprocess);
  `b` at the rare explicit-coordinate-entry prompts (freq/d3 manual k-path coords).
- ~~`formula_extractor.extract_formula_from_d12` str/Path crash~~ — DONE (`d8f1140f`): all four
  public entry points coerce str→Path; missing paths return None; regression tests added.
- ~~`CrystalPropertyExtractor.__init__` unconditionally creates `materials.db`~~ — DONE
  (`00f2efb0`): DB is now lazy + `enable_tracking=False` for offline single-file extraction;
  pure parsing never creates a file. Tests added.

## STRATEGIC IMPROVEMENTS (from the review)

- **Regression test suite over the `test/` corpus + CI** — DONE (commit `013b8b61`).
  `tests/` (pytest) pins gCP/corrected-total/molecular-thermo/enthalpy, BAND.DAT band-index gap,
  formula BASISSET-reset, and the B1 recovery chain against real `test/` outputs; `find_data()`
  skips cleanly when the gitignored corpus is absent so CI stays green (recovery tests are
  self-contained and always run). `pytest.ini` + `.github/workflows/tests.yml`. Now 33 tests
  green, also covering: SPINLOCK round-trip, D3 workflow_id fallback, offline extraction, and
  the SP missing-data classifier (`8622cb52`). Still uncovered: the `-D3-D3` guard and d3
  geometry double-count are inline in large interactive flows (`CRYSTALOptToD12.write_*`,
  `CRYSTALOptToD3.generate_d3`); pinning them cleanly needs a small pure-helper extraction —
  deferred to avoid refactoring validated interactive code. Both were verified against real
  outputs at fix time.
  Update: the `-D3-D3` issue is a FILENAME-only nuisance (content always correct; confirmed by
  user — results in `test/` unaffected). Now fully removed (`7175e10b`) via the testable
  `dedupe_dispersion_suffix` helper applied to the continuation filename; 9 tests pin it.
  (d3 geometry double-count remains the only unpinned Wave A fix.)
- Delete confirmed-dead/duplicate trees: `code/Plotting_Scripts` (1733 lines, stale vs live
  `Plotting/` — fdf9cf8e fixed `Plotting/` but the stale copy still has the bug),
  `executor_contextual`/`planner_contextual`/`run_workflow_isolated` (imported by nothing),
  `Archived/`. De-track committed artifacts (root `materials.db`, `.DS_Store`) via .gitignore.
- ~~Single-source `__version__`~~ — DONE (`4387049b`): mace_cli, formats.py, animation.py all
  derive from `mace/__init__.py`; verified a bump propagates; tests pin it. Panel-analyzed.
  STILL TODO: extract a shared crystal-parsing/constants module (one HARTREE_TO_EV, one geometry
  parser, one element/ECP map) — do incrementally behind tests to preserve validated detection.

## DEAD-CODE DELETION — DONE (user-confirmed; panel-verified)

Multi-agent panel (analysis + adversarial verify, run wf_c1f92500-571) confirmed all candidates
dead with zero live callers; the adversarial verifiers could not refute "safe to delete" and
debunked a competing PATHFINDER "ACTIVE" claim. User confirmed deletion. Each removed as its own
revertable commit; recoverable from git history.
- Workflow isolation cluster (`853670bd`): deleted
  `mace/workflow/{executor_contextual,planner_contextual,run_workflow_isolated}.py`; rewrote
  `ISOLATION_MIGRATION_GUIDE.md` to describe the base-class isolation that supersedes them.
- `code/Plotting_Scripts/` (`12683bc0`): deleted (stale dup of live `Plotting/`); removed orphaned
  PATH exports (activate_mace.sh, setup_mace.py x2) + dead `mace_env_helper.py` entry; pointed
  example/help + README/INSTALLATION/DOCUMENTATION at `mace plotting`. AUTHORSHIP.md kept (credit).
- KEPT (live): `mace/database/materials_contextual.py` (ContextualMaterialDatabase, mace_cli:1379,1630).
- Verified after each: `mace plotting`/`--version` work, no live refs remain, 51-test suite green.

---

## PERSISTING ISSUES — synthesis of ALL prior analysis (verified 2026-06-13)

Cross-referenced CODEBASE_AUDIT.md / PATHFINDER-2026-06-12 / .planning/codebase/CONCERNS.md
against what's fixed; each VERIFIED against live code + call-sites (audit severities were often
overstated — checking callers downgraded the "critical" trio to dormant).

LATENT (real defects, but in methods with NO live callers — fix before relying on those features):
- `ContextualMaterialDatabase` (mace/database/materials_contextual.py) has 3 broken methods:
  (a) `copy_to_context` calls `create_or_update_material` which does not exist (L166);
  (b) it then calls `store_material_property` (materials.py:1114) which INSERTs `str(uuid4())`
      into `property_id INTEGER PRIMARY KEY` (materials.py:130) → datatype-mismatch IntegrityError;
  (c) `get_workflow_materials`/`get_workflow_calculations` (L257/293) call `self.get_connection()`
      but only `_get_connection` exists → AttributeError.
  ALL THREE ARE DORMANT: `copy_to_context` has zero callers; the two workflow-query methods have
  zero callers; `store_material_property`'s only caller is `copy_to_context`. The live extractor
  writes via its own autoincrement INSERT (property_extractor.save_properties_to_database), which
  is why tracking works today. Cheap to fix (rename to create_material/_get_connection; drop the
  explicit property_id so it autoincrements).

HIGH (live):
- Default `mace submit` discards the SLURM job_id: submission/crystal.py:246,362 +
  properties.py:243,359 use `os.system(...)`, so untracked submissions can't be
  monitored/recovered (the new `--track`/`mace manager` path captures the id; this is the
  untracked path). Fix: switch to subprocess.run + job-id regex (mirror manager.submit_to_slurm).
- Dead import disables input-settings capture: queue/manager.py:1390
  `from input_settings_extractor import ...` — no such module exists → ImportError (likely
  swallowed); the feature silently never runs. Remove or implement.

MEDIUM:
- `shell=True` with interpolated `node_type` in the LIVE util mace/utils/node_exclusion.py:132-135
  (`scontrol show nodes | grep 'NodeName={node_type}'`) — injection surface; split the pipe.
- Duplicated crystal-system/space-group logic: d3_kpoints.py:862-899 vs inline chains in
  d12_constants.py (no cross-import) — drift landmine.
- aggregation.py:149 `conductivity_type` grouping may bucket all "Unknown" (the energy_range/
  band_gap_range groupings were already fixed in c51c1c94; this sub-grouping was not).
- Hardcoded institutional basis paths executor.py:2026,2035 (has ./ + ../ fallbacks → degraded,
  not fatal, off-cluster).

LOW:
- Plotting batch is O(N^2) (mace/plotting/main.py ~L280-313, each plotter re-globs) — perf only.
- capture_workflow_execution_data (materials.py ~L1004) returns last material only; NameError on
  present-but-empty input lists.
- legacy_manager.py:166 shell=True — dead/reference-only (no live import); near-zero exposure.

STRATEGIC (debt; no active bug):
- Shared crystal-`.out` parser duplicated across >=6 sites (property_extractor is most complete;
  detector/d12_parsers/seekpath/CBM_VBM re-parse). Fermi extraction in ~5 places. d12_config vs
  d3_config parallel save/load/validate I/O. analysis/* query boilerplate (no base class). Two
  submit_slurm_job impls in executor.py (L893,L1206) + dual script generators (executor.py:747 vs
  engine.py:327). HIGH_SYMMETRY_PATHS dup (d12_constants copy vestigial). → extract shared modules
  incrementally behind tests (preserve validated detection).
- Untested core: engine.py (~3587 LoC), planner.py (~5006 LoC) have no automated tests; the
  51-test suite pins campaign fixes only. No coverage gate.
- More dead files NOT yet removed (panel/PATHFINDER dead-code verification): top-level
  `mace_config.py` (unimported by mace_cli/run_mace), `mace/submission/portable_slurm_generator.py`
  (example; emits `--work-dir .` the manager argparse rejects), `Crystal_d12/Archived/` (8),
  `Crystal_d3/Archived/` (11), `code/Check_Scripts/Archived/`. KEEP `enhanced_queue_manager.py`
  (REQUIRED shim — referenced by copy_dependencies/templates/docs).

Already FIXED but docs (esp. CONCERNS.md) are STALE — do NOT re-fix: missing_data classifier
(0d52a696), ipDOS_V2 row-skip (fdf9cf8e) + stale copy deleted (12683bc0), aggregation
energy/band_gap groupings (c51c1c94), walltime regex, recovery fix-script paths, engine
step-config substring->exact match, plotting glob breadth, validate_materials stub (85cae68c),
history.py timestamp split (c51c1c94), AND recovery `.f9`/`.f98`/`fort.9` deletion guard
(recovery.py:637-642 — VERIFIED present). CONCERNS.md is the stalest artifact; several audit
paths are pre-reorg (now under mace/database/utils/).

## Where the full review output lives
Ephemeral (will not survive): `/tmp/claude-1000/.../tasks/wmcn5j90p.output` (43-agent review,
per-commit verdicts). The verified high/critical items are captured above. Re-run the review
workflow if the raw per-commit detail is needed again.
