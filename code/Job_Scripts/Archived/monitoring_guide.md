# CRYSTAL Workflow Monitoring Guide

## Quick Setup

If you're in a workflow directory and need monitoring capabilities, run:
```bash
python setup_workflow_monitoring.py
```

This copies all necessary monitoring scripts to your current directory.

## Monitoring Commands

### 1. Quick Status Check
```bash
python material_monitor.py --action stats
python monitor_workflow.py --action status
```

### 2. Live Dashboard (Continuous Monitoring)
```bash
python material_monitor.py --action dashboard --interval 30
```
Press Ctrl+C to stop. Refreshes every 30 seconds.

### 3. One-Time Status Snapshot
```bash
python material_monitor.py --action status
```

### 4. Detailed Report Generation
```bash
python material_monitor.py --action report --output my_report.json
```

## Database Queries

### Python Database Queries
```python
from material_database import MaterialDatabase

# Initialize database connection
db = MaterialDatabase("materials.db")

# Get all materials
materials = db.get_all_materials()
for mat in materials:
    print(f"{mat['material_id']}: {mat['formula']} ({mat['status']})")

# Get calculations by status
pending_calcs = db.get_calculations_by_status("pending")
running_calcs = db.get_calculations_by_status("running") 
completed_calcs = db.get_calculations_by_status("completed")
failed_calcs = db.get_calculations_by_status("failed")

print(f"Pending: {len(pending_calcs)}, Running: {len(running_calcs)}")
print(f"Completed: {len(completed_calcs)}, Failed: {len(failed_calcs)}")

# Get calculations for specific material
material_calcs = db.get_calculations_by_status(material_id="test1")
for calc in material_calcs:
    print(f"{calc['calc_type']}: {calc['status']}")

# Get recent calculations (last 20)
recent = db.get_recent_calculations(20)
for calc in recent:
    print(f"{calc['material_id']} - {calc['calc_type']} - {calc['status']}")

# Database statistics
stats = db.get_database_stats()
print(f"Total materials: {stats['total_materials']}")
print(f"Calculations by status: {stats['calculations_by_status']}")
print(f"Calculations by type: {stats['calculations_by_type']}")
print(f"Database size: {stats['db_size_mb']:.2f} MB")
```

### Helper Script Queries
```bash
# Show all materials
python monitor_workflow.py --action materials

# Show recent calculations  
python monitor_workflow.py --action calculations

# Show database statistics
python monitor_workflow.py --action stats
```

## Checking Workflow Progress

### For Specific Materials
```python
from material_database import MaterialDatabase

db = MaterialDatabase()

# Check progress for a specific material
material_id = "test1"
material = db.get_material(material_id)
if material:
    print(f"Material: {material['formula']}")
    print(f"Created: {material['created_at']}")
    
    # Get all calculations for this material
    calcs = db.get_calculations_by_status(material_id=material_id)
    
    # Group by calculation type and status
    by_type = {}
    for calc in calcs:
        calc_type = calc['calc_type']
        if calc_type not in by_type:
            by_type[calc_type] = {}
        status = calc['status']
        by_type[calc_type][status] = by_type[calc_type].get(status, 0) + 1
    
    for calc_type, statuses in by_type.items():
        print(f"{calc_type}: {statuses}")
```

### Workflow Completion Status
```python
from material_database import MaterialDatabase

db = MaterialDatabase()

def check_workflow_completion(material_id):
    """Check which steps of the workflow are complete."""
    calcs = db.get_calculations_by_status(material_id=material_id)
    
    completed_types = set()
    for calc in calcs:
        if calc['status'] == 'completed':
            completed_types.add(calc['calc_type'])
    
    workflow_steps = ['OPT', 'SP', 'BAND', 'DOSS', 'FREQ']
    
    print(f"Workflow progress for {material_id}:")
    for step in workflow_steps:
        status = "✓" if step in completed_types else "✗"
        print(f"  {status} {step}")
    
    return completed_types

# Check specific material
check_workflow_completion("test1")

# Check all materials
materials = db.get_all_materials()
for mat in materials:
    check_workflow_completion(mat['material_id'])
    print()
```

## Job Queue Monitoring

### SLURM Queue Check
```bash
# Check your jobs
squeue -u $USER

# Check job details
scontrol show job JOB_ID

# Check completed jobs
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode
```

### Database Job Tracking
```python
from material_database import MaterialDatabase

db = MaterialDatabase()

# Get calculations with SLURM job IDs
all_calcs = db.get_all_calculations()
with_slurm = [c for c in all_calcs if c.get('slurm_job_id')]

print(f"Calculations with SLURM job IDs: {len(with_slurm)}")

# Show active jobs
active = [c for c in with_slurm if c['status'] in ['submitted', 'running']]
for calc in active:
    print(f"Job {calc['slurm_job_id']}: {calc['material_id']} ({calc['calc_type']}) - {calc['status']}")
```

## Error Analysis

### Recent Errors
```python
from error_detector import CrystalErrorDetector

detector = CrystalErrorDetector()

# Generate error report for last 7 days
report = detector.generate_error_report(days_back=7)

print(f"Errors in last 7 days: {sum(report['error_summary'].values())}")
print(f"Error types: {report['error_summary']}")

# Show materials with errors
for mat_id, details in report['material_details'].items():
    if details['errors_found'] > 0:
        print(f"{mat_id}: {details['errors_found']} errors")
        print(f"  Most common: {details['most_common_error']}")
```

### Analyze Specific Output File
```python
from error_detector import CrystalErrorDetector
from pathlib import Path

detector = CrystalErrorDetector()

# Analyze a specific output file
output_file = Path("material_name.out")
if output_file.exists():
    analysis = detector.analyze_output_file(output_file)
    
    print(f"Status: {analysis['status']}")
    if analysis['error_type']:
        print(f"Error: {analysis['error_type']}")
        print(f"Recoverable: {analysis['recoverable']}")
        if analysis['recovery_hints']:
            print("Recovery suggestions:")
            for hint in analysis['recovery_hints']:
                print(f"  - {hint}")
```

## Common Monitoring Tasks

### 1. Check if Database is Being Updated
```python
from material_database import MaterialDatabase
from datetime import datetime, timedelta

db = MaterialDatabase()
recent = db.get_recent_calculations(10)

if recent:
    latest = recent[0]
    latest_time = datetime.fromisoformat(latest['created_at'])
    age = datetime.now() - latest_time
    
    print(f"Latest calculation: {age.total_seconds()/3600:.1f} hours ago")
    if age > timedelta(hours=6):
        print("Warning: No recent activity - check if workflow is running")
else:
    print("No calculations in database")
```

### 2. Monitor Job Success Rate
```python
from material_database import MaterialDatabase
from datetime import datetime, timedelta

db = MaterialDatabase()

# Get calculations from last 7 days
cutoff = datetime.now() - timedelta(days=7)
all_calcs = db.get_all_calculations()

recent_calcs = []
for calc in all_calcs:
    if calc.get('created_at'):
        created = datetime.fromisoformat(calc['created_at'])
        if created > cutoff:
            recent_calcs.append(calc)

if recent_calcs:
    completed = len([c for c in recent_calcs if c['status'] == 'completed'])
    failed = len([c for c in recent_calcs if c['status'] == 'failed'])
    total = len(recent_calcs)
    
    success_rate = (completed / total) * 100 if total > 0 else 0
    
    print(f"Last 7 days: {total} calculations")
    print(f"Success rate: {success_rate:.1f}% ({completed} completed, {failed} failed)")
```

### 3. Find Stuck Jobs
```python
from material_database import MaterialDatabase
from datetime import datetime, timedelta

db = MaterialDatabase()

# Find jobs submitted more than 24 hours ago but still pending/running
cutoff = datetime.now() - timedelta(hours=24)
all_calcs = db.get_all_calculations()

stuck_jobs = []
for calc in all_calcs:
    if calc['status'] in ['submitted', 'running'] and calc.get('created_at'):
        created = datetime.fromisoformat(calc['created_at'])
        if created < cutoff:
            stuck_jobs.append(calc)

if stuck_jobs:
    print(f"Found {len(stuck_jobs)} potentially stuck jobs:")
    for calc in stuck_jobs:
        hours_old = (datetime.now() - datetime.fromisoformat(calc['created_at'])).total_seconds() / 3600
        print(f"  {calc['material_id']} ({calc['calc_type']}) - {hours_old:.1f} hours old")
        if calc.get('slurm_job_id'):
            print(f"    SLURM job: {calc['slurm_job_id']}")
```

## Troubleshooting

### Database Connection Issues
```python
from material_database import MaterialDatabase

try:
    db = MaterialDatabase("materials.db")
    stats = db.get_database_stats()
    print("Database connection: OK")
    print(f"Database size: {stats['db_size_mb']:.2f} MB")
except Exception as e:
    print(f"Database connection failed: {e}")
```

### Missing Monitoring Scripts
If you get import errors, run:
```bash
python setup_workflow_monitoring.py
```

### Performance Issues
If monitoring is slow:
```python
from material_database import MaterialDatabase

db = MaterialDatabase()
stats = db.get_database_stats()

# Check database size
if stats['db_size_mb'] > 1000:  # >1GB
    print("Large database detected - consider cleanup")
    
    # Clean up old failed calculations
    db.cleanup_old_records(days_old=30)
    print("Cleaned up old records")
```

## Integration with Workflow Execution

### Check Workflow Configuration
If using the workflow system, you can check saved configurations:
```python
import json
from pathlib import Path

# Find workflow configuration files
config_dir = Path("workflow_configs")
if config_dir.exists():
    for config_file in config_dir.glob("*.json"):
        with open(config_file) as f:
            config = json.load(f)
            print(f"Workflow: {config.get('workflow_id', 'unknown')}")
            print(f"  Sequence: {config.get('workflow_sequence', [])}")
            print(f"  Created: {config.get('created', 'unknown')}")
```

This guide provides comprehensive monitoring capabilities for your CRYSTAL workflow system. Use these commands and scripts to track progress, identify issues, and ensure smooth operation of your calculations.