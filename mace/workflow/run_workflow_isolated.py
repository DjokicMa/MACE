#!/usr/bin/env python3
"""
Example: Minimal changes to enable isolation in run_workflow.py
===============================================================
This shows how to add isolation support with minimal code changes.
"""

import sys
import os
from pathlib import Path

# Minimal change: Import contextual versions with same names
try:
    # Try importing contextual versions first
    from mace.workflow.planner_contextual import ContextualWorkflowPlanner as WorkflowPlanner
    from mace.workflow.executor_contextual import ContextualWorkflowExecutor as WorkflowExecutor
    ISOLATION_AVAILABLE = True
except ImportError:
    # Fall back to original versions
    from mace.workflow.planner import WorkflowPlanner
    from mace.workflow.executor import WorkflowExecutor
    ISOLATION_AVAILABLE = False


def main():
    """Example main function with isolation support."""
    
    # Check for isolation mode preference
    isolation_mode = os.environ.get('MACE_ISOLATION_MODE', 'shared')
    
    if ISOLATION_AVAILABLE and isolation_mode == 'isolated':
        print("🔒 Running in ISOLATED mode - each workflow gets its own database")
    else:
        print("🔗 Running in SHARED mode - using common database")
    
    # Create planner with isolation support (if available)
    if ISOLATION_AVAILABLE:
        planner = WorkflowPlanner(isolation_mode=isolation_mode)
    else:
        planner = WorkflowPlanner()
    
    # Rest of the workflow planning code remains the same
    planner.display_welcome()
    
    # ... planning logic ...
    
    # Create executor with isolation support (if available)
    if ISOLATION_AVAILABLE:
        executor = WorkflowExecutor(auto_activate_context=True)
    else:
        executor = WorkflowExecutor()
    
    # ... execution logic ...


# Alternative: Monkey-patch approach for zero code changes
def enable_isolation_globally():
    """
    Enable isolation support globally without changing any import statements.
    This approach modifies the module namespace directly.
    """
    try:
        # Import contextual versions
        from mace.workflow.planner_contextual import ContextualWorkflowPlanner
        from mace.workflow.executor_contextual import ContextualWorkflowExecutor
        from mace.database.materials_contextual import ContextualMaterialDatabase
        
        # Replace in mace modules
        import mace.workflow.planner
        import mace.workflow.executor
        import mace.database.materials
        
        mace.workflow.planner.WorkflowPlanner = ContextualWorkflowPlanner
        mace.workflow.executor.WorkflowExecutor = ContextualWorkflowExecutor
        mace.database.materials.MaterialDatabase = ContextualMaterialDatabase
        
        print("✅ Workflow isolation support enabled globally")
        return True
        
    except ImportError as e:
        print(f"⚠️  Could not enable isolation support: {e}")
        return False


# Example: Minimal wrapper for existing run_workflow.py
def run_workflow_with_optional_isolation():
    """
    Wrapper that adds isolation support to existing run_workflow.py
    without modifying the original file.
    """
    # Check if user wants isolation
    if os.environ.get('MACE_ENABLE_ISOLATION', '').lower() in ['true', '1', 'yes']:
        if enable_isolation_globally():
            print("Isolation support activated")
    
    # Import and run the original run_workflow
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from run_workflow import main as original_main
    
    # Run with isolation support (if enabled)
    original_main()


if __name__ == "__main__":
    # Example 1: Direct usage with conditional imports
    main()
    
    # Example 2: Wrapper approach (uncomment to use)
    # run_workflow_with_optional_isolation()
    
    # Example 3: Quick test of isolation
    if '--test-isolation' in sys.argv and ISOLATION_AVAILABLE:
        print("\nTesting isolation features...")
        
        from mace.workflow.context import workflow_context
        
        with workflow_context("test_workflow", isolation_mode="isolated") as ctx:
            print(f"✓ Created isolated context: {ctx.workflow_id}")
            print(f"✓ Database path: {ctx.db_path}")
            print(f"✓ Working directory: {ctx.workflow_root}")
            
            db = ctx.get_database()
            stats = db.get_database_stats()
            print(f"✓ Database initialized: {stats}")
        
        print("✓ Context cleaned up successfully")