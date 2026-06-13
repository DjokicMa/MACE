# Coding Conventions

**Analysis Date:** 2026-06-13

## Naming Patterns

**Files:**
- Modules use `snake_case` with descriptive prefixes that indicate their domain: `d12_parsers.py`, `d12_config.py`, `d12_constants.py`, `d12_writer.py`
- Top-level scripts that are entry points use `PascalCase` or `CamelCase` stems: `CRYSTALOptToD12.py`, `NewCifToD12.py`, `CrystalOutToCif.py`
- Test files are prefixed with `test_`: `test_seekpath_interface.py`
- Subpackage modules use `snake_case`: `file_manager.py`, `property_extractor.py`, `queue_lock_manager.py`

**Functions:**
- All functions use `snake_case`: `rotation_matrix_to_xyz`, `parse_crystal_structure`, `get_accurate_bandpath`, `calculate_minimum_shrink`
- Private/internal methods on classes use a single leading underscore: `_extract_geometry`, `_extract_spacegroup`, `_extract_cell_parameters`, `_extract_functional`
- Entry-point functions are named `main()` in every script

**Variables:**
- Local variables use `snake_case`: `space_group`, `shrink_factor`, `output_file`, `calc_id`
- Loop variables: short meaningful names (`i`, `j` for numeric indexing; `line`, `lines`, `term` for text processing)

**Types / Classes:**
- Classes use `PascalCase`: `CrystalOutputParser`, `CrystalInputParser`, `MaterialDatabase`, `WorkflowEngine`, `QueueLockManager`, `TestResult`, `AtomicCharge`
- Contextual subclasses suffix with `Contextual`: `ContextualWorkflowExecutor`, `ContextualWorkflowPlanner`, `ContextualMaterialDatabase`

**Constants:**
- Module-level constants use `SCREAMING_SNAKE_CASE`: `ELEMENT_SYMBOLS`, `SPACEGROUP_SYMBOLS`, `DEFAULT_TOLERANCES`, `RHOMBOHEDRAL_SPACEGROUPS`, `FUNCTIONAL_CATEGORIES`, `ECP_ELEMENTS_EXTERNAL`
- Defined in dedicated constants modules: `Crystal_d12/d12_constants.py`, `mace/workflow/common/constants.py`

## Code Style

**Formatting:**
- No automated formatter is configured (no `.prettierrc`, `pyproject.toml`, `setup.cfg`, or `biome.json` found)
- Indentation: 4 spaces (consistent throughout)
- Line separators within long modules: `# =============================================================================` (78 `=` chars) used to delineate major sections (Unit Tests, Integration Tests, Main)
- Shorter section dividers: `# ---` prefix style (e.g., `# --- Testing cell_params_to_vectors ---`)

**Linting:**
- No linter configured (no `.flake8`, `pylint.ini`, `ruff.toml`)
- Code relies on author discipline rather than automated enforcement

**String formatting:**
- f-strings used throughout for interpolation: `f"Got: {convert_seekpath_label('GAMMA')}"`, `f"  [PASS] {name}"`
- f-strings preferred over `.format()` or `%`

**Encoding declaration:**
- Core Crystal_d12 modules include explicit encoding declaration: `# -*- coding: utf-8 -*-`
- mace/ modules omit the encoding declaration (Python 3 default UTF-8 assumed)

## File Headers

Every non-trivial module starts with:
1. Shebang: `#!/usr/bin/env python3` (occasionally `#!/usr/bin/python3`)
2. Optional encoding: `# -*- coding: utf-8 -*-` (Crystal_d12 modules)
3. Module-level docstring with:
   - Title and dashes/equals underline
   - Short description paragraph
   - `Author:` and `Institution:` fields (Marcus Djokic, Michigan State University, Mendoza Group)
   - `Features:`, `Classes:`, or `Usage:` section as applicable

Example from `Crystal_d12/d12_config.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 Configuration Management Module
============================================================
...
Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
```

## Import Organization

**Order (observed pattern):**
1. Standard library: `os`, `sys`, `re`, `json`, `argparse`, `subprocess`, `pathlib`, `typing`, `datetime`, `threading`, `collections`
2. Third-party: `numpy as np`, `pandas`, `seekpath`
3. Local/relative: same-package imports using bare module names (no dots in Crystal_d12), package-relative imports (`from mace.recovery.pandas_utils import ...`) in mace/

**Path manipulation for local imports:**
- Crystal_d12 scripts insert their own directory into `sys.path` at load time:
  ```python
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  ```
- mace/ uses proper Python package imports (relative or absolute)

**Path handling — mixed style:**
- `pathlib.Path` used in newer code (`mace/`, `Crystal_d3/`, `Crystal_d12/d12_from_config.py`)
- `os.path.*` still present in older and mixed code: `os.path.dirname`, `os.path.exists`, `os.path.join`
- Both styles coexist; prefer `pathlib.Path` for new code

## Type Annotations

- Used consistently on function signatures in newer modules: `Crystal_d12/d12_parsers.py`, `Crystal_d12/d12_writer.py`, `mace/database/utils/units.py`
- Imports from `typing`: `Dict`, `List`, `Any`, `Optional`, `Tuple`, `Union`
- Return types annotated with `-> ReturnType`
- `dataclass` used for small data holders: `AtomicCharge`, `BondInformation`, `CoordinationEnvironment` in `mace/utils/population_analysis_processor.py`, `BasisSetInfo` in `Crystal_d12/d12_constants.py`
- Older scripts (in `code/`) lack annotations entirely

## Docstrings

**Function docstrings:**
- Google/NumPy-adjacent style: description paragraph, then `Args:` / `Returns:` sections
- Short methods may use single-line docstrings: `"""Extract unit cell parameters"""`
- Longer functions include full Args/Returns: `rotation_matrix_to_xyz` in `Crystal_d12/d12_parsers.py`

**Class docstrings:**
- One-liner descriptions on simpler classes; multi-line with feature lists on complex classes (`CrystalErrorDetector`, `MaterialDatabase`)

## Error Handling

**Patterns:**
- `try/except (ValueError, IndexError)` used heavily in parsers to silently skip malformed lines
- Specific exceptions caught where the type is known; `except Exception as e` used for broader fallback logging in mace/ components
- `raise ValueError("message")` used when parsing cannot recover: `raise ValueError("Could not find geometry in output file")`
- Graceful degradation: optional dependencies wrapped in try/except at import time with a boolean flag:
  ```python
  try:
      from seekpath_interface import (...)
      MODULE_AVAILABLE = True
  except ImportError as e:
      MODULE_AVAILABLE = False
  ```
- `PANDAS_AVAILABLE` pattern in `mace/recovery/detector.py` / `mace/recovery/pandas_utils.py`

**What NOT to do:**
- Do not use bare `except:` without specifying exception type
- Do not swallow exceptions silently without at least a boolean flag or print

## Logging

**Framework:** `print()` statements — no `logging` module in use anywhere

**Patterns:**
- Progress/status messages go to stdout via `print()`
- Indented output uses `"  "` prefix (2 spaces) for sub-items: `print(f"  [PASS] {name}")`, `print(f"  Bravais: ...")`
- Warnings use `"Warning:"` prefix in message string
- Error messages use `"Error:"` prefix
- No structured logging, no log levels

## Comments

**When to Comment:**
- Inline comments explain non-obvious numeric logic (floating-point rounding, fraction identification)
- Section dividers `# ===...===` used to separate logical blocks within large files
- TODO/FIXME used for known incomplete implementations:
  - `# TODO: Implement priority-based geometry selection` (`Crystal_d12/CrystalOutToCif.py:244`)
  - `# TODO: Implement full SeeK-path criteria` (`Crystal_d3/d3_kpoints.py:274`)
  - `# TODO: Extract properties from ...` (`mace/queue/manager.py:1349`)

## Function Design

**Size:** Functions range from 10-line helpers to 200+ line parsers. Prefer extracting parse sub-steps into `_extract_*` methods.

**Parameters:**
- File paths accepted as `str` (not `Path`) at function boundaries; converted to `Path` internally when needed
- Configuration data passed as `Dict[str, Any]`
- Output files (CRYSTAL `.out`) accepted by path string throughout

**Return Values:**
- Parsers return `Dict[str, Any]` for structured results
- Functions that may fail return `None` rather than raising (integration test helpers, optional lookups)
- Multiple return values use plain tuples: `(segments, labels, kpath_info)`

## Module Design

**Exports:**
- No `__all__` defined in most modules; public API is implicit
- `mace/__init__.py` and subpackage `__init__.py` files present but may be minimal

**Barrel Files:**
- Not used in Crystal_d12 (flat module directory with `sys.path` manipulation)
- Present in mace/ subpackages (`mace/__init__.py`, `mace/recovery/__init__.py`, etc.) but contents are typically sparse

## Entry Point Pattern

Every script that can be run directly follows this pattern:
```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    # logic

if __name__ == "__main__":
    main()
```

---

*Convention analysis: 2026-06-13*
