#!/usr/bin/env python3
"""
Test Automatic Error Recovery Integration
=========================================
This script demonstrates how the enhanced queue manager now automatically
detects, analyzes, and recovers from common CRYSTAL calculation errors.

Usage:
  python test_error_recovery.py [--test-mode]
  
The test mode simulates a failed calculation to show the recovery process.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_queue_manager import EnhancedCrystalQueueManager
    from material_database import MaterialDatabase
    from error_recovery import ErrorRecoveryEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory with all dependencies.")
    sys.exit(1)

def create_test_failed_calculation():
    """Create a mock failed calculation for testing."""
    print("🧪 Creating test failed calculation...")
    
    # Create temporary directory
    test_dir = Path("test_error_recovery_demo")
    test_dir.mkdir(exist_ok=True)
    
    # Create a mock D12 file with SHRINK error
    d12_content = """CRYSTAL
EXTERNAL
OPTGEOM
SHRINK
8 16
END
"""
    
    d12_file = test_dir / "test_material.d12"
    with open(d12_file, 'w') as f:
        f.write(d12_content)
    
    # Create a mock output file with SHRINK error
    output_content = """
CRYSTAL17 calculation started
Loading basis set...
Setting up calculation...

EEEEEE
TTTTTT   ERROR ERROR ERROR ERROR ERROR ERROR
EEEEEE
SHRINK PARAMETERS TOO HIGH FOR THIS SYSTEM
PLEASE REDUCE SHRINK VALUES
CALCULATION TERMINATED

Program ended with exit code 1
"""
    
    output_file = test_dir / "test_material.out"
    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"    Created test files in: {test_dir}")
    return test_dir, d12_file, output_file

def simulate_error_recovery():
    """Simulate the error recovery process."""
    print("\n🔧 Simulating Automatic Error Recovery Process")
    print("=" * 60)
    
    # Create test environment
    test_dir, d12_file, output_file = create_test_failed_calculation()
    
    try:
        # Initialize queue manager with error recovery enabled
        print("\n1️⃣ Initializing Enhanced Queue Manager with Error Recovery...")
        queue_manager = EnhancedCrystalQueueManager(
            d12_dir=str(test_dir),
            max_jobs=10,
            enable_tracking=True,
            enable_error_recovery=True,
            max_recovery_attempts=3,
            db_path=str(test_dir / "test_materials.db")
        )
        
        print(f"    ✅ Queue manager initialized")
        print(f"    📊 Error recovery: {'ENABLED' if queue_manager.enable_error_recovery else 'DISABLED'}")
        print(f"    🔄 Max recovery attempts: {queue_manager.max_recovery_attempts}")
        
        # Create a mock calculation record
        print("\n2️⃣ Creating mock calculation record...")
        material_id = "test_material"
        calc_id = f"{material_id}_opt_001"
        
        if queue_manager.db:
            # Add material to database
            queue_manager.db.create_material(
                material_id=material_id,
                formula="C2H4",  # Simple test formula
                source_type="test"
            )
            
            # Add calculation to database
            queue_manager.db.create_calculation(
                material_id=material_id,
                calc_type="OPT",
                input_file=str(d12_file),
                work_dir=str(test_dir),
                calc_id=calc_id
            )
            
            print(f"    ✅ Created calculation record: {calc_id}")
        
        # Simulate error analysis
        print("\n3️⃣ Analyzing calculation error...")
        mock_calc = {
            'calc_id': calc_id,
            'material_id': material_id,
            'calc_type': 'OPT',
            'work_dir': str(test_dir),
            'input_file': str(d12_file),
            'output_file': str(output_file)
        }
        
        error_type, error_message = queue_manager.analyze_calculation_error(mock_calc)
        print(f"    🔍 Detected error type: {error_type}")
        print(f"    📝 Error message: {error_message}")
        
        # Test error recovery
        print("\n4️⃣ Attempting automatic error recovery...")
        if queue_manager.enable_error_recovery and queue_manager.error_recovery_engine:
            # This would normally be called automatically when a job fails
            recovery_success = queue_manager.attempt_error_recovery(mock_calc, error_type, error_message)
            
            if recovery_success:
                print(f"    ✅ Error recovery SUCCESSFUL!")
                print(f"    🚀 Job would be automatically resubmitted")
            else:
                print(f"    ⚠️  Error recovery not successful or not applicable")
        else:
            print(f"    ❌ Error recovery not available")
        
        # Show database state
        print("\n5️⃣ Database state after recovery:")
        if queue_manager.db:
            calcs = queue_manager.db.get_calculations_by_material(material_id)
            for calc in calcs:
                print(f"    📊 {calc['calc_id']}: {calc['status']} (recovery attempts: {calc.get('recovery_attempts', 0)})")
        
        print("\n✅ Error Recovery Demonstration Complete!")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if test_dir.exists():
            try:
                shutil.rmtree(test_dir)
                print(f"\n🧹 Cleaned up test directory: {test_dir}")
            except:
                print(f"\n⚠️  Could not clean up test directory: {test_dir}")

def show_integration_summary():
    """Show how error recovery is integrated into the workflow."""
    print("\n🎯 Automatic Error Recovery Integration Summary")
    print("=" * 60)
    
    print("""
🔄 **How It Works:**

1️⃣ **Job Completion Monitoring**
   - SLURM scripts automatically call enhanced_queue_manager.py when jobs complete
   - Command: python enhanced_queue_manager.py --callback-mode completion --max-recovery-attempts 3

2️⃣ **Automatic Error Detection**
   - Failed jobs trigger handle_failed_calculation()
   - Errors are analyzed using analyze_calculation_error()
   - Common error patterns detected: SHRINK, memory, convergence, timeout, SCF

3️⃣ **Intelligent Recovery**
   - attempt_error_recovery() calls ErrorRecoveryEngine
   - Applies appropriate fixes based on error type
   - Tracks recovery attempts to prevent infinite loops

4️⃣ **Automatic Resubmission**
   - Fixed calculations are automatically resubmitted
   - Database tracks recovery attempts and status
   - Workflow progression continues normally

🛠️  **Supported Error Types:**
   ✅ SHRINK errors → Reduce SHRINK parameters
   ✅ Memory errors → Increase memory allocation
   ✅ Convergence errors → Adjust convergence criteria
   ✅ Timeout errors → Increase walltime
   ✅ SCF errors → Modify SCF settings

🎚️  **Configuration Options:**
   --max-recovery-attempts 3     # Maximum attempts per job
   --disable-error-recovery      # Turn off automatic recovery

📊 **Monitoring:**
   python material_monitor.py --action stats      # Recovery statistics
   python enhanced_queue_manager.py --status      # Current job states
""")

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Test automatic error recovery integration")
    parser.add_argument("--test-mode", action="store_true", help="Run simulation test")
    
    args = parser.parse_args()
    
    print("🔧 CRYSTAL Automatic Error Recovery System")
    print("=" * 50)
    
    if args.test_mode:
        simulate_error_recovery()
    
    show_integration_summary()
    
    print("\n🚀 **Next Steps:**")
    print("1. Run your workflows normally with: python run_workflow.py --interactive")
    print("2. Error recovery will happen automatically when jobs fail")
    print("3. Monitor progress with: python material_monitor.py --action stats")
    print("4. Check recovery details with: python enhanced_queue_manager.py --status")

if __name__ == "__main__":
    main()