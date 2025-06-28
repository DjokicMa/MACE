# YAML Files Analysis in Job_Scripts

## Overview

There are two YAML configuration files in the Job_Scripts directory:

1. **recovery_config.yaml** (10,743 bytes)
2. **workflows.yaml** (13,121 bytes)

## 1. recovery_config.yaml

### Purpose
Defines automated recovery strategies for common CRYSTAL calculation errors.

### Actually Used By
- **error_recovery.py** - ACTIVELY LOADED AND USED
  - Loads the configuration in `__init__` method
  - Default path: "recovery_config.yaml"
  - Uses `yaml.safe_load()` to parse the configuration
  - Falls back to hardcoded defaults if file not found

### Contents
- Error recovery handlers for:
  - SHRINK errors (k-point mesh issues)
  - Memory errors (insufficient allocation)
  - Convergence errors (SCF not converging)
  - Timeout errors (walltime exceeded)
  - Disk space errors (manual intervention required)
  - Basis set errors (linear dependence)
  - Geometry optimization failures
  - Symmetry errors

- Global settings:
  - Max concurrent recoveries
  - Recovery log retention
  - Notification settings
  - Safety limits

### Status: **ACTIVELY USED** ✅

## 2. workflows.yaml

### Purpose
Intended to define calculation workflows, dependencies, and resource requirements.

### Actually Used By
- **NONE** - NOT ACTIVELY LOADED ❌

### Investigation Results
- Referenced in comments in `workflow_planner.py` but never actually loaded
- Listed in `copy_dependencies.py` as a file to copy
- The comment in workflow_planner.py says "Apply calculation-specific scaling from workflows.yaml" but the scaling is actually hardcoded
- No actual YAML loading code found for this file

### Contents
Defines comprehensive workflow configurations including:
- Full characterization workflow (OPT → SP → BAND/DOSS)
- Quick optimization workflow
- High accuracy workflow
- Transport properties workflow
- Phonon analysis workflow
- Resource requirements for each step
- Dependency specifications

### Status: **NOT USED** ❌

## Detailed Analysis

### recovery_config.yaml - Implementation
```python
# In error_recovery.py:
def __init__(self, db_path: str = "materials.db", config_path: str = "recovery_config.yaml"):
    self.config_path = Path(config_path)
    self.config = self.load_recovery_config()

def load_recovery_config(self) -> Dict:
    if self.config_path.exists():
        with open(self.config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            default_config.update(user_config)
            return default_config
```

### workflows.yaml - No Implementation Found
Despite comprehensive workflow definitions, this file is never loaded. The workflow logic is instead:
- Hardcoded in `workflow_planner.py` as dictionaries
- Templates defined directly in Python code
- Resource scaling defined as inline dictionaries

## Recommendations

### For recovery_config.yaml
- **KEEP** - Actively used by error_recovery.py
- Well-structured and provides valuable configuration
- Allows customization without code changes

### For workflows.yaml
- **CONSIDER REMOVING** or **IMPLEMENT LOADING**
- Option 1: Remove since it's not used and workflow logic is hardcoded
- Option 2: Implement actual loading in workflow_planner.py to use these definitions
- The file contains good workflow definitions that could replace hardcoded values

## Code That Should Load workflows.yaml

In `workflow_planner.py`, there's a comment suggesting it should use workflows.yaml:
```python
# Apply calculation-specific scaling from workflows.yaml
scaled_resources = self.apply_calc_type_scaling(base_resources, calc_type)
```

But the actual implementation uses hardcoded values:
```python
scaling_rules = {
    "OPT": {"walltime_factor": 1.0, "memory_factor": 1.0},
    "SP": {"walltime_factor": 0.43, "memory_factor": 0.8},
    # ... etc
}
```

## Conclusion

- **recovery_config.yaml**: Essential and actively used ✅
- **workflows.yaml**: Good content but not implemented - either implement or remove ⚠️