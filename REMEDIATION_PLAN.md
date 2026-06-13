# MACE Remediation Plan & Resume Anchor

Single source of truth for outstanding work. Updated 2026-06-13. If context is
compacted, **start here**, then `git log --oneline -40` to see what's committed.

Constraints (always): verify every fix against **real `test/` outputs** (never synthetic);
use `/home/marcus/anaconda3/bin/python`; "don't fix what works" (layer, don't rewrite);
`test/` is gitignored — never commit it; commit but don't push unless asked.

Verification habit (a prior-commit lesson): several past commits shipped "verified" claims
that did NOT reproduce. Re-run each fix against real data and don't overclaim.

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

## DEAD-CODE DELETION — panel-verified, AWAITING USER CONFIRMATION (destructive)

Multi-agent panel (analysis + adversarial verify, run wf_c1f92500-571) confirmed all candidates
are dead with zero live callers; the adversarial verifiers could not refute "safe to delete" and
debunked a competing PATHFINDER "ACTIVE" claim. Pending user go-ahead:
- `code/Plotting_Scripts/` (stale duplicate of live `Plotting/`; `mace plotting` resolves to
  `Plotting/` via sys.path.insert). deletion_risk low. Optional collateral: orphaned PATH exports
  in activate_mace.sh:23 / setup_mace.py:128,211 and echo-help lines.
- `mace/workflow/executor_contextual.py`, `planner_contextual.py`, `run_workflow_isolated.py`
  — one orphaned isolation-EXAMPLE cluster (run_workflow_isolated self-labels as "Example"; the
  live isolation feature lives in the BASE executor/planner/context, unaffected). Delete together
  + remove/​update `mace/workflow/ISOLATION_MIGRATION_GUIDE.md`.
- DO NOT DELETE `mace/database/materials_contextual.py` (ContextualMaterialDatabase) — it is LIVE
  (mace_cli:1379,1630). `Archived/` does not exist (moot).

---

## Where the full review output lives
Ephemeral (will not survive): `/tmp/claude-1000/.../tasks/wmcn5j90p.output` (43-agent review,
per-commit verdicts). The verified high/critical items are captured above. Re-run the review
workflow if the raw per-commit detail is needed again.
