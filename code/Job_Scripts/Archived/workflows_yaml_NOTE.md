# Note on workflows.yaml

## Archive Date: June 27, 2025

## Reason for Archiving
This YAML file was never actually loaded or used by any script in the workflow system. Investigation revealed:

1. **Not Loaded**: No code actually loads this file with `yaml.load()` or `yaml.safe_load()`
2. **Hardcoded Instead**: All workflow definitions are hardcoded in `workflow_planner.py`
3. **Only Referenced**: Only mentioned in comments suggesting it should be used

## File Contents
The file contains well-structured workflow definitions including:
- Full characterization workflow (OPT → SP → BAND/DOSS)
- Quick optimization workflow  
- High accuracy workflow
- Transport properties workflow
- Phonon analysis workflow
- Resource requirements and dependencies

## Future Consideration
The workflow definitions in this file are actually quite good. If someone wants to reduce hardcoding in the future, they could:

1. Implement YAML loading in `workflow_planner.py`
2. Replace hardcoded workflow templates with these definitions
3. Allow users to customize workflows via external configuration

## Related Code
In `workflow_planner.py` around line 2500:
```python
# Apply calculation-specific scaling from workflows.yaml
scaled_resources = self.apply_calc_type_scaling(base_resources, calc_type)
```

But the actual implementation uses hardcoded dictionaries instead of loading this file.