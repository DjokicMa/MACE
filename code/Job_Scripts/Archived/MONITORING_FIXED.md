# ✅ Monitoring Issues Fixed

All monitoring script issues have been resolved! Here's what was fixed and how to use the monitoring system:

## 🔧 Issues Fixed

1. **Import Error for `error_detector` module** - Fixed with graceful fallback
2. **Syntax Errors in f-string queries** - Replaced with proper Python syntax
3. **Missing monitoring scripts in workflow directories** - Auto-setup now included

## 🚀 Quick Start

### Automatic Setup (Recommended)
When you run workflows, monitoring scripts are automatically copied to the workflow directory.

### Manual Setup
If you need monitoring in any directory:
```bash
python setup_workflow_monitoring.py
```

## 📊 Monitoring Commands

### 1. Quick Status Check
```bash
# Database and queue overview
python material_monitor.py --action stats
python monitor_workflow.py --action status
```

### 2. Live Dashboard
```bash
# Continuous monitoring (press Ctrl+C to stop)
python material_monitor.py --action dashboard --interval 30
```

### 3. Database Queries

#### Using Helper Script
```bash
# Show all materials
python monitor_workflow.py --action materials

# Show recent calculations
python monitor_workflow.py --action calculations

# Show database statistics
python monitor_workflow.py --action stats
```

#### Using Python Directly
```python
from material_database import MaterialDatabase

# Initialize database
db = MaterialDatabase("materials.db")

# Get all materials
materials = db.get_all_materials()
for mat in materials:
    print(f"{mat['material_id']}: {mat['formula']} ({mat['status']})")

# Get calculations by status
pending = db.get_calculations_by_status("pending")
running = db.get_calculations_by_status("running")
completed = db.get_calculations_by_status("completed")
failed = db.get_calculations_by_status("failed")

print(f"Pending: {len(pending)}, Running: {len(running)}")
print(f"Completed: {len(completed)}, Failed: {len(failed)}")

# Get calculations for specific material
material_calcs = db.get_calculations_by_status(material_id="test1")
for calc in material_calcs:
    print(f"{calc['calc_type']}: {calc['status']}")

# Database statistics
stats = db.get_database_stats()
print(f"Total materials: {stats['total_materials']}")
print(f"Calculations by status: {stats['calculations_by_status']}")
```

### 4. Workflow Progress Tracking

#### Check Individual Materials
```python
from material_database import MaterialDatabase

db = MaterialDatabase()

def check_workflow_progress(material_id):
    """Check which workflow steps are complete for a material."""
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
check_workflow_progress("test1")

# Check all materials
materials = db.get_all_materials()
for mat in materials:
    check_workflow_progress(mat['material_id'])
    print()
```

### 5. Error Analysis (When Available)
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

## 📈 Current Status

Based on your database:
- **Materials**: 0 (database is ready for new workflows)
- **Calculations**: 0 
- **Database Size**: 0.09 MB (fresh database)
- **Queue Jobs**: N/A (no jobs currently running)

## 🎯 Next Steps

1. **Database Population Confirmed**: The workflow system now properly populates the database when jobs are submitted (not just when they finish)

2. **Monitoring Ready**: All monitoring scripts are available and working

3. **Start a Workflow**: When you run `python run_workflow.py --interactive`, the database will be populated immediately upon job submission

4. **Monitor Progress**: Use the commands above to track your calculations

## 📚 Documentation

- **Complete Guide**: `monitoring_guide.md` - Comprehensive monitoring documentation
- **Quick Reference**: `MONITORING_README.md` - Created in each workflow directory
- **Setup Script**: `setup_workflow_monitoring.py` - Copies all monitoring tools

## ✅ Test Verification

All monitoring components have been tested and verified:
- ✅ Database connection working
- ✅ Import errors resolved  
- ✅ Monitoring scripts functional
- ✅ Helper scripts operational
- ✅ Workflow integration complete

You can now confidently use the monitoring system to track your CRYSTAL calculations!