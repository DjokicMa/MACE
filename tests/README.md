# MACE regression tests

Parser/handler regression tests that pin the behavior fixed during the
remediation campaign so it cannot silently regress (the recurring
"claimed-but-doesn't-reproduce" failure mode).

## Running

```bash
python -m pytest                       # from the repo root
```

Install the pinned test dependencies with `pip install -r requirements-test.txt`
(pytest, pyyaml, numpy, rich, matplotlib). `ase`/`spglib`/`scipy`/`plotly` unlock
more of the suite; missing extras skip their tests.

## Data dependency

Per project policy these tests verify against the **real** CRYSTAL outputs in
`test/` — no synthetic CRYSTAL output. That corpus is ~12 GB and gitignored, so:

- On a developer machine (with `test/` present) the full suite runs.
- In a data-less CI it **skips** the data-dependent tests cleanly; the
  self-contained logic tests (`test_recovery.py`) still run.

`conftest.find_data(pattern)` locates a real file by glob or skips.

## Coverage

The suite has grown well beyond this founding set — 60+ test files now also
cover the plotting handlers, terminal UI, queue/engine logic, and database
extraction. The table below documents the original remediation wave:

| File | Locks in |
|------|----------|
| `test_energy_extraction.py` | gCP term, corrected total (final-geometry, not stale OPT line), molecular FREQ thermo header, enthalpy = G + TS, electronic energy |
| `test_band_dat.py` | BAND.DAT gap by band index (Fermi-referenced), metal detection, None-without-n_occ |
| `test_formula_extraction.py` | BASISSET-section reset (Sulfolane, EC, combined electrolyte) |
| `test_recovery.py` | recovery resubmits the FIX not the original; OPTGEOM-aware MAXCYCLE; bumped-script threading |

When adding a parser fix, add a test here with the ground-truth value read from
the real output, and reference the fixing commit.
