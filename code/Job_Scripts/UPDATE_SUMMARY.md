# Job_Scripts Directory Reference Updates

## Summary
Updated all Python files in the Job_Scripts directory and its subdirectories to use the new directory names:
- `Crystal_To_CIF` → `Crystal_d12`
- `Creation_Scripts` → `Crystal_d3`

## Files Updated

### Main Job_Scripts Directory
1. **workflow_planner.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`
   - Updated script paths for NewCifToD12.py and CRYSTALOptToD12.py

2. **workflow_executor.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`
   - Updated script paths for NewCifToD12.py and CRYSTALOptToD12.py

3. **workflow_engine.py**
   - Updated script paths for Crystal_d12 directory
   - Updated script paths for Crystal_d3 directory

4. **run_workflow.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`

5. **copy_dependencies.py**
   - Updated source directory from `Crystal_To_CIF` to `Crystal_d12`
   - Updated source directory from `Creation_Scripts` to `Crystal_d3`

### test_2d_materials Subdirectory
1. **test_freq_verification.py**
   - Updated script paths for CRYSTALOptToD12.py and NewCifToD12.py

2. **test_freq_simple.py**
   - Updated script path for CRYSTALOptToD12.py

3. **test_freq_cli.py**
   - Updated script paths for NewCifToD12.py and CRYSTALOptToD12.py

4. **test_freq_generation.py**
   - Updated script paths for CRYSTALOptToD12.py and NewCifToD12.py

5. **test_all_freq_templates.py**
   - Updated import path and script path for CRYSTALOptToD12.py

6. **test_raman_template_flow.py**
   - Updated script path and import path for CRYSTALOptToD12.py

7. **freq_test_cli/newcif_basic_freq_(thermodynamics)/test_freq_interactive.py**
   - Updated test file path and script path references

### ExampleFreqWorkflow Subdirectory
1. **workflow_planner.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`
   - Updated script paths for NewCifToD12.py and CRYSTALOptToD12.py

2. **workflow_executor.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`
   - Updated script paths for NewCifToD12.py and CRYSTALOptToD12.py

3. **workflow_engine.py**
   - Updated script paths for Crystal_d12 directory
   - Updated script paths for Crystal_d3 directory

4. **run_workflow.py**
   - Updated import path from `Crystal_To_CIF` to `Crystal_d12`

5. **copy_dependencies.py**
   - Updated source directory from `Crystal_To_CIF` to `Crystal_d12`
   - Updated source directory from `Creation_Scripts` to `Crystal_d3`

## Total Files Updated: 17

All references have been successfully updated to use the new directory structure.