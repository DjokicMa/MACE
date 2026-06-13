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

## WAVE B (confirmed high/critical from the full review)

### B1 — Error-recovery chain is broken end-to-end (CRITICAL, commit e24cc154)
Recovery "succeeds" but resubmits the ORIGINAL failing input, so SCF/OOM/timeout jobs re-run
with the exact parameters that already failed.
- `mace/recovery/recovery.py`: SCF handler writes the fix to `<stem>_recovery_<ts>.d12`
  (~L308-310) but only records it in a separate DB row; memory/timeout handlers literally
  `return Path(calc['input_file'])` (~L396), discarding the bumped `--mem`/`--time` script.
- `mace/queue/manager.py` `resubmit_fixed_calculation` (~L1213) prefers `recorded_path.exists()`
  → the original unfixed input, and submits THAT (verified: MAXCYCLE stays at the failed value).
- Fix: pass the handler's `_recovery_<ts>` path into resubmit (or update `calc['input_file']`);
  make memory/timeout handlers return their bumped `_recovery_<ts>.sh` (or have submit_to_slurm
  honor it); dispatch or remove the orphaned pending recovery row.
- Also (same commit): `convergence_handler` bumps BOTH OPTGEOM and SCF `MAXCYCLE` (no block
  awareness — only the SCF cap should change); double-bookkeeping creates 2 DB rows per recovery;
  error-string taxonomy diverges: `manager.analyze_calculation_error` uses `TOO MANY SCF CYCLES`
  (manager.py:~1073) vs detector `TOO MANY CYCLES` (detector.py:~67) — unify.
- CAVEAT: `test/` has ZERO failed CRYSTAL outputs, so the whole recovery taxonomy is UNVERIFIED
  against real failures. Need at least one real failed `.out` to validate before trusting this.

### B2 — BAND.DAT electronic gap is wrong and persisted (HIGH, commit 505f5ead)
`mace/utils/dat_file_processor.py` `_analyze_band_structure`: BAND.DAT records line-wrap when
NBND is large (e.g. NBND=259 → many physical lines per k-point) and are NOT de-wrapped, and the
gap is split at the absolute `.out` Fermi rather than the file's own VBM. Result: ~10x-wrong gap
(e.g. BAND 0.8 eV vs DOSS 7.6 eV vs .out 8.2 eV) written via `property_extractor.py:~1282-1287`
(`band_dat_band_gap_ev/vbm/cbm/metallic`).
- Fix: de-wrap BAND records (flatten tokens, reshape by width `1 + NBND`, mirror the DOSS
  token-flatten already in the same module); reference eigenvalues to the file's own VBM (E≈0)
  / detect the HOMO-LUMO crossing instead of the absolute Fermi.
- Until fixed: stop persisting `band_dat_band_gap_ev/vbm/cbm` (DOSS already gives the correct gap).
- Verify on a wrapped BAND.DAT (e.g. `test/BAND/1LiFSI-1EMS-conf1_*band.BAND.DAT`, NBND=259)
  and a metal (4LG) vs the DOSS/.out gap.

---

## BACKLOG (lower priority, from the review — fix opportunistically)

- D3 workflow continuation halts on the `'manual'` workflow_id fallback (`engine.py:~1558`
  → use `workflow_base.name`, matching the numbered/SP paths). MEDIUM.
- OPT continuation drops a configured non-zero SPINLOCK: `d12_parsers` never parses the
  SPINLOCK keyword, so the round-trip loses it. LOW.
- Public d12 wrappers `def f(*args, **kwargs)` lose their signature — add `functools.wraps`
  to `configure_single_point/optimization`, `get_frequency_configuration`. LOW.
- `mace manager` in-place default flips silently: print a one-line notice; add `--organize`
  to `mace manager --help` (mace_cli help block ~767-773); confirm before bare `mace submit`
  on a populated dir. LOW.
- `mace submit --track` opt-in DB tracking (route through EnhancedCrystalQueueManager in-place).
- Full cross-step planner orchestrator back (risky — interleaves JSON writes / subprocess);
  `b` at the rare explicit-coordinate-entry prompts (freq/d3 manual k-path coords).
- `formula_extractor.extract_formula_from_d12` expects a `Path` but some callers pass a `str`
  → `AttributeError: 'str' object has no attribute 'exists'`. Make it accept both. (Found in review.)
- `CrystalPropertyExtractor.__init__` unconditionally creates `materials.db`, blocking offline
  single-file extraction — make tracking/db optional.

## STRATEGIC IMPROVEMENTS (from the review)

- **Regression test suite over the `test/` corpus + CI** — highest leverage; nearly every
  STILL_INCORRECT finding (formula +1 atom, BAND.DAT reference, -D3-D3, recovery resubmit, gCP)
  plus the "claimed-but-doesn't-reproduce" pattern would have been caught by parser-output tests.
- Delete confirmed-dead/duplicate trees: `code/Plotting_Scripts` (1733 lines, stale vs live
  `Plotting/` — fdf9cf8e fixed `Plotting/` but the stale copy still has the bug),
  `executor_contextual`/`planner_contextual`/`run_workflow_isolated` (imported by nothing),
  `Archived/`. De-track committed artifacts (root `materials.db`, `.DS_Store`) via .gitignore.
- Single-source `__version__` (import from `mace/__init__.py`); extract a shared
  crystal-parsing/constants module (one HARTREE_TO_EV, one geometry parser, one element/ECP map)
  — do incrementally behind tests to preserve validated detection behavior.

---

## Where the full review output lives
Ephemeral (will not survive): `/tmp/claude-1000/.../tasks/wmcn5j90p.output` (43-agent review,
per-commit verdicts). The verified high/critical items are captured above. Re-run the review
workflow if the raw per-commit detail is needed again.
