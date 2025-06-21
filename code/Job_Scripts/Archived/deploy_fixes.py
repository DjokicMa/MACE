#!/usr/bin/env python3
"""
Deploy Callback Fixes
----------------------
Deploy the fixed callback mechanism files to a target directory.
"""

import os
import sys
import shutil
from pathlib import Path

def deploy_fixes_to_directory(target_dir):
    """Deploy the fixed files to the target directory."""
    target_dir = Path(target_dir).resolve()
    source_dir = Path(__file__).parent
    
    # Files to deploy
    files_to_deploy = [
        'enhanced_queue_manager.py',
        'material_database.py', 
        'workflow_engine.py',
        'workflow_executor.py',
        'populate_completed_jobs.py',
        'test_callback_fix.py'
    ]
    
    print(f"Deploying callback fixes to: {target_dir}")
    
    deployed_count = 0
    for filename in files_to_deploy:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"  ✓ Deployed: {filename}")
                deployed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to deploy {filename}: {e}")
        else:
            print(f"  ! Source file not found: {filename}")
    
    print(f"\nDeployment complete: {deployed_count}/{len(files_to_deploy)} files deployed")
    
    # Test the deployment
    print("\nTesting deployment...")
    test_imports(target_dir)
    
    return deployed_count == len(files_to_deploy)


def test_imports(target_dir):
    """Test that the deployed files can be imported correctly."""
    original_path = sys.path.copy()
    
    try:
        # Add target directory to path
        sys.path.insert(0, str(target_dir))
        
        # Test imports
        try:
            from enhanced_queue_manager import EnhancedQueueManager
            print("  ✓ Enhanced queue manager import successful")
        except Exception as e:
            print(f"  ✗ Enhanced queue manager import failed: {e}")
            
        try:
            from workflow_engine import WorkflowEngine
            print("  ✓ Workflow engine import successful")
        except Exception as e:
            print(f"  ✗ Workflow engine import failed: {e}")
            
        try:
            from populate_completed_jobs import scan_for_completed_calculations
            print("  ✓ Populate completed jobs import successful")
        except Exception as e:
            print(f"  ✗ Populate completed jobs import failed: {e}")
            
    finally:
        # Restore original path
        sys.path = original_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy callback mechanism fixes")
    parser.add_argument("target_dir", help="Target directory to deploy fixes to")
    parser.add_argument("--test-only", action="store_true", help="Only test imports, don't deploy")
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing imports only...")
        test_imports(args.target_dir)
    else:
        success = deploy_fixes_to_directory(args.target_dir)
        if success:
            print("\n🎉 All fixes deployed successfully!")
            print("\nNext steps:")
            print(f"1. cd {args.target_dir}")
            print("2. python test_callback_fix.py")
            print("3. python enhanced_queue_manager.py --callback-mode completion")
        else:
            print("\n⚠️  Some files failed to deploy. Check the errors above.")


if __name__ == "__main__":
    main()