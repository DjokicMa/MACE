# 🔧 Fix Workflow Progression Issue

## Problem
Your OPT calculations completed, but they're not progressing to SP step because the SLURM scripts can't find `enhanced_queue_manager.py` in the workflow structure.

## Quick Fix

Run these commands from your `~/test` directory:

### 1. Copy Queue Manager to Workflow Directory
```bash
cd ~/test
cp enhanced_queue_manager.py workflow_outputs/workflow_20250621_001019/
cp crystal_queue_manager.py workflow_outputs/workflow_20250621_001019/
cp material_database.py workflow_outputs/workflow_20250621_001019/
cp error_recovery.py workflow_outputs/workflow_20250621_001019/
cp recovery_config.yaml workflow_outputs/workflow_20250621_001019/
```

### 2. Manually Trigger SP Generation
```bash
cd workflow_outputs/workflow_20250621_001019/
python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
```

### 3. Check Progress
```bash
# From workflow directory
python enhanced_queue_manager.py --status

# Or from base directory  
cd ~/test
python material_monitor.py --action stats
```

## What This Does

1. **Copies Queue Manager**: Places `enhanced_queue_manager.py` in the workflow directory where SLURM scripts can find it
2. **Processes Completions**: Scans for completed OPT jobs and generates SP inputs
3. **Submits SP Jobs**: Automatically submits the next step calculations

## Verify SP Jobs Started

After running the fix, check:
```bash
cd ~/test
squeue -u $USER  # Should show new SP jobs
tree workflow_outputs/  # Should show step_002_SP directory
```

## For Future Workflows

The workflow executor has been updated to:
- ✅ Automatically copy queue manager to workflow directories
- ✅ Update SLURM scripts to find queue manager in workflow hierarchy
- ✅ Enable proper progression: OPT → SP → BAND → DOSS

Your next `python run_workflow.py --interactive` will work correctly without manual intervention.

## Current Status Check

To see what materials completed OPT:
```bash
cd ~/test/workflow_outputs/workflow_20250621_001019/step_001_OPT/
for dir in */; do
    material=${dir%/}
    if [[ -f "$dir/$material.out" && -f "$dir/$material.f9" ]]; then
        echo "✅ $material - OPT completed"
    else
        echo "⏳ $material - OPT in progress"
    fi
done
```

## Database Status

Check what's in the database:
```bash
cd ~/test
python -c "
from material_database import MaterialDatabase
db = MaterialDatabase()
calcs = db.get_recent_calculations(20)
for calc in calcs:
    print(f'{calc[\"material_id\"]} - {calc[\"calc_type\"]} - {calc[\"status\"]}')
"
```

This should show your 8 OPT calculations and their status. After running the fix, you should see SP calculations appearing.