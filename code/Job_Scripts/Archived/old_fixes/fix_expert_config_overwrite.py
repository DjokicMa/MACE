#!/usr/bin/env python3
"""
Fix Expert Config Overwrite Issue
=================================
Fixes the issue where OPT2 and OPT3 expert configurations overwrite each other
because they use the same directory name.
"""

def fix_workflow_planner():
    """
    The fix needed in workflow_planner.py to prevent config overwrites.
    
    The issue: When configuring OPT2 and OPT3 in expert mode, both use 
    calc_type="OPT" which creates the same directory "expert_config_opt",
    causing OPT3 to overwrite OPT2's configuration.
    
    The solution: Include the step number or full calc type in the directory name.
    """
    
    print("Fix for workflow_planner.py:")
    print("=" * 60)
    print("""
In the _get_expert_opt_config method (around line 1266), change:

OLD:
    expert_config = self._run_interactive_crystal_opt_config(crystal_calc_type)

NEW:
    # Pass the full calc_type (e.g., OPT2, OPT3) instead of just "OPT"
    expert_config = self._run_interactive_crystal_opt_config(calc_type)

---

In the _run_interactive_crystal_opt_config method (around line 1420), change:

OLD:
    temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}"

NEW:
    # Use full calc_type to create unique directories for OPT2, OPT3, etc.
    temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}"
    
    # This will create:
    # - expert_config_opt for OPT
    # - expert_config_opt2 for OPT2
    # - expert_config_opt3 for OPT3
    # - expert_config_sp for SP
    # - expert_config_sp2 for SP2
    # etc.

---

Also in _run_interactive_crystal_opt_config, update the CRYSTALOptToD12.py call
to use the correct base calculation type:

OLD (around line 1462):
    "--calc-type", calc_type  # This would pass OPT2, OPT3 which CRYSTALOptToD12.py doesn't understand

NEW:
    # Extract base calc type for CRYSTALOptToD12.py
    import re
    base_calc_type = re.match(r'^([A-Z]+)', calc_type).group(1) if calc_type else "SP"
    ...
    "--calc-type", base_calc_type  # Pass OPT, SP, FREQ (without numbers)
""")
    print("=" * 60)
    
    # Check what's currently in the workflow temp directory
    import os
    from pathlib import Path
    
    # Look for expert config directories
    if Path("workflow_temp").exists():
        print("\nCurrent expert config directories:")
        for item in Path("workflow_temp").iterdir():
            if item.is_dir() and item.name.startswith("expert_config_"):
                print(f"  {item.name}/")
                # List JSON files
                for json_file in item.glob("*.json"):
                    print(f"    - {json_file.name}")
                    
    print("\nThe issue: Both OPT2 and OPT3 are using 'expert_config_opt' directory!")
    print("This causes OPT3 configuration to overwrite OPT2 configuration.")
    
    # Temporary workaround
    print("\n" + "=" * 60)
    print("TEMPORARY WORKAROUND:")
    print("=" * 60)
    print("""
Since your workflow is already running, here's what will happen:

1. When OPT completes and OPT2 needs to be generated:
   - It will use the configuration in expert_config_opt/opt_expert_config.json
   - This is actually the OPT3 configuration (since it was configured last)
   
2. When OPT2 completes and OPT3 needs to be generated:
   - It will use the same configuration file
   - This is correct for OPT3

So both OPT2 and OPT3 will use the OPT3 configuration settings.

To work around this for future workflows:
1. Configure OPT2 and OPT3 with the same settings, OR
2. After configuring OPT2, manually copy the config before configuring OPT3:
   cp workflow_temp/expert_config_opt/opt_expert_config.json workflow_temp/opt2_config_backup.json
   
3. After workflow planning completes, restore the OPT2 config:
   cp workflow_temp/opt2_config_backup.json workflow_temp/expert_config_opt2/opt_expert_config.json
""")


if __name__ == "__main__":
    fix_workflow_planner()