# Workflow Monitoring Guide

This workflow is managed by MACE. Use these commands to monitor your workflow:

## Quick Commands

### Check Workflow Status
```bash
mace status
# or
mace workflow --status
```

### Live Monitoring Dashboard
```bash
mace monitor --dashboard
# Press Ctrl+C to stop
```

### Check Material Database
```bash
mace database --stats
```

### View Queue Status
```bash
mace queue --status
```

### Material Monitor
```bash
# Quick stats
python -m mace.material_monitor --action stats

# Live dashboard
python -m mace.material_monitor --action dashboard
```

### Error Recovery
```bash
# Check for recoverable errors
python -m mace.recovery.recover --action stats

# Attempt recovery
python -m mace.recovery.recover --action recover
```

### Workflow Engine
```bash
# Process workflow
python -m mace.workflow.engine --action process

# Check workflow status
python -m mace.workflow.engine --action status
```

## Notes
- All MACE commands are available via your PATH after running setup_mace.py
- The workflow uses an isolated database in .mace_context_{workflow_id}/
- Monitor commands automatically detect the active workflow context
