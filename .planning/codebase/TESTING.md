# Testing Patterns

**Analysis Date:** 2026-06-13

## Test Framework

**Runner:**
- No pytest, unittest, or tox configuration found (no `pytest.ini`, `setup.cfg`, `pyproject.toml`, `tox.ini`)
- The single formal test suite (`Crystal_d3/test_seekpath_interface.py`) uses a **custom hand-rolled test runner** — not pytest or unittest
- Tests are executed by running the script directly with Python

**Assertion Library:**
- No assertion library — pass/fail logic is implemented manually via a `TestResult` class
- Numeric comparisons use `numpy.allclose(array, expected, atol=1e-10)` for floating-point arrays
- String comparisons use direct `==` equality
- Conditional pass/fail using `if condition: results.ok(name) else: results.fail(name, msg)`

**Run Commands:**
```bash
# Run all unit tests
/home/marcus/anaconda3/bin/python Crystal_d3/test_seekpath_interface.py

# Run unit tests + integration test against a specific output file
/home/marcus/anaconda3/bin/python Crystal_d3/test_seekpath_interface.py --file test/BAND/somefile.out

# Compare new vs old implementation
/home/marcus/anaconda3/bin/python Crystal_d3/test_seekpath_interface.py --compare --file test/BAND/somefile.out

# Print detailed band path info
/home/marcus/anaconda3/bin/python Crystal_d3/test_seekpath_interface.py --info --file test/BAND/somefile.out
```

## Test File Organization

**Location:**
- One formal test file co-located with the module it tests: `Crystal_d3/test_seekpath_interface.py` tests `Crystal_d3/seekpath_interface.py`
- `test/` directory at repo root contains **real CRYSTAL output fixtures** (not test code), organized by calculation type

**Naming:**
- Test script: `test_<module_name>.py`
- Test functions: `test_<thing_being_tested>(results: TestResult)`

**`test/` directory structure:**
```
test/
├── BAND/     # Real .out, .f25, .f9, .d3, .BAND.DAT files from band structure calculations
├── CIFs/     # Real .cif crystal structure files
├── DOSS/     # Real .out, .f25, .DOSS.DAT files from DOS calculations
├── ECH3POT3/ # Real charge+potential output files
├── FREQ/     # Real .out, .d12, .f9 files from frequency calculations
├── OPT/      # Real .out, .d12 files from geometry optimization calculations
├── SP/       # Real .out, .d12 files from single-point calculations
└── TRANSPORT/# Real .out, .KAPPA.dat, .SEEBECK.dat files from transport calculations
```

These files are the canonical verification corpus. All parser changes must be verified against them.

## Test Structure

**Custom TestResult class** (defined in `Crystal_d3/test_seekpath_interface.py:43`):
```python
class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, msg=""):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  [FAIL] {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} tests passed")
        return self.failed == 0
```

**Test function signature:**
```python
def test_<name>(results: TestResult):
    """Docstring describing what is tested."""
    print("\n--- Testing <name> ---")
    # ... test cases
    if condition:
        results.ok("Case description")
    else:
        results.fail("Case description", f"Got: {actual}")
```

**Test runner entry in `run_all_tests()`:**
```python
def run_all_tests():
    if not MODULE_AVAILABLE:
        return False
    results = TestResult()
    test_cell_params_to_vectors(results)
    test_label_conversion(results)
    test_minimum_shrink(results)
    test_seekpath_available(results)
    test_discontinuity_detection(results)
    return results.summary()
```

**Sections within `test_seekpath_interface.py`:**
- Unit Tests section (`# === Unit Tests ===`): Pure function tests with known inputs/outputs
- Integration Tests section (`# === Integration Tests ===`): Tests against real `.out` files from `test/`
- Main section: `argparse`-based CLI to choose between unit tests, integration tests, or comparison mode

## Mocking

**Framework:** None — no `unittest.mock`, `pytest-mock`, or `MagicMock` used

**Mock data pattern:**
- Inline dictionaries constructed within test functions to simulate module outputs:
  ```python
  mock_result = {
      'point_coords': {
          'GAMMA': [0.0, 0.0, 0.0],
          'X': [0.5, 0.0, 0.5],
          ...
      },
      'path': [('GAMMA', 'X'), ('X', 'U'), ('K', 'GAMMA'), ('GAMMA', 'L')],
      'has_inversion_symmetry': True,
      'bravais_lattice': 'cF',
      'bravais_lattice_extended': 'cF1'
  }
  ```
- The mock dict is passed directly to the function under test (`convert_to_mace_format`)
- No patching of external calls

**What to mock:**
- Complex external library outputs (e.g., seekpath result dicts) when testing downstream processing logic

**What NOT to mock:**
- The underlying physics/chemistry parsers — always test against real CRYSTAL `.out` files in `test/`
- The `seekpath` library itself when testing that integration works end-to-end

## Fixtures and Factories

**Test Data:**
- Real CRYSTAL output files in `test/` subdirectories serve as fixtures
- Parser tests pass the file path string to the function under test:
  ```python
  structure = parse_crystal_structure(output_file)  # output_file = path to test/*.out
  ```
- Synthetic numeric fixtures constructed inline in unit tests:
  ```python
  cell = cell_params_to_vectors(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
  expected = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
  ```

**Location:**
- All real fixtures: `test/BAND/`, `test/OPT/`, `test/SP/`, `test/FREQ/`, `test/DOSS/`, `test/ECH3POT3/`, `test/TRANSPORT/`, `test/CIFs/`
- Synthetic numeric fixtures: defined inline within the test function body

## Coverage

**Requirements:** None enforced — no coverage configuration or CI gate

**Coverage approach:** Manual: developer runs the test script and verifies `[PASS]` / `[FAIL]` output. Exit code is `0` for all-pass, `1` for any failure.

## Test Types

**Unit Tests:**
- Scope: individual pure functions with known math (cell parameter conversion, label string conversion, shrink factor calculation)
- Approach: construct specific inputs, compare to hand-calculated expected values
- Location: `Crystal_d3/test_seekpath_interface.py` functions `test_cell_params_to_vectors`, `test_label_conversion`, `test_minimum_shrink`, `test_discontinuity_detection`

**Integration Tests:**
- Scope: full pipeline from file path → parsed structure → seekpath → output format
- Approach: pass a real CRYSTAL `.out` file from `test/`, assert non-empty output and correct counts
- Location: `Crystal_d3/test_seekpath_interface.py` function `test_with_file`
- Run with: `--file test/BAND/<file>.out`

**Comparison Tests:**
- Scope: new implementation vs old implementation for regression detection
- Approach: run both code paths on the same `.out` file, print differences
- Location: `Crystal_d3/test_seekpath_interface.py` function `compare_implementations`
- Run with: `--compare --file test/BAND/<file>.out`

**E2E Tests:** Not formally present — the `test/` fixture files serve as an informal E2E corpus for manual developer verification

## Common Patterns

**Floating-point testing:**
```python
if np.allclose(cell, expected, atol=1e-10):
    results.ok("Cubic cell")
else:
    results.fail("Cubic cell", f"Got:\n{cell}")
```

**Integer threshold testing:**
```python
if shrink >= 6:
    results.ok("Thirds (1/3)")
else:
    results.fail("Thirds", f"Got shrink={shrink}, need >=6")
```

**Count testing:**
```python
n_discontinuities = labels.count('|')
if n_discontinuities == 1:
    results.ok("Discontinuity count")
else:
    results.fail("Discontinuity count", f"Expected 1, got {n_discontinuities}")
```

**Optional dependency guard:**
```python
if not SEEKPATH_AVAILABLE:
    results.fail("Discontinuity detection", "seekpath not available")
    return
```

**File existence guard:**
```python
if not Path(output_file).exists():
    results.fail("File exists", f"Not found: {output_file}")
    return
```

**Exception safety in integration tests:**
```python
try:
    structure = parse_crystal_structure(output_file)
    if structure is not None:
        results.ok(f"Structure parsing ({len(structure['numbers'])} atoms)")
    else:
        results.fail("Structure parsing", "Returned None")
        return
except Exception as e:
    results.fail("Structure parsing", str(e))
    return
```

## Verification Against Real Outputs (Project Rule)

Per project memory: **always verify parsers against `test/*.out` files, not synthetic fixtures**. When modifying any parser in `Crystal_d12/` or `Crystal_d3/`, run it manually against files in the matching `test/` subdirectory:

```bash
# Example: verify d12 parser against OPT output
/home/marcus/anaconda3/bin/python -c "
from Crystal_d12.d12_parsers import CrystalOutputParser
p = CrystalOutputParser('test/OPT/1_dia_opt_rev1.out')
print(p.parse())
"
```

---

*Testing analysis: 2026-06-13*
