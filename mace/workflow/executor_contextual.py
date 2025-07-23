#!/usr/bin/env python3
"""
Context-Aware Workflow Executor Extension
=========================================
Extensions to the workflow executor to support isolated execution contexts.

Author: Workflow isolation enhancement
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from mace.workflow.executor import WorkflowExecutor
from mace.workflow.context import WorkflowContext, workflow_context
from mace.database.materials_contextual import ContextualMaterialDatabase
from mace.queue.manager import EnhancedCrystalQueueManager


class ContextualWorkflowExecutor(WorkflowExecutor):
    """
    Extended workflow executor with context isolation support.
    
    Features:
    - Automatic context activation from workflow plans
    - Isolated resource management during execution
    - Context-aware component initialization
    - Proper cleanup and result export
    """
    
    def __init__(self, work_dir: str = ".", db_path: str = None,
                 auto_activate_context: bool = True):
        """
        Initialize contextual workflow executor.
        
        Args:
            work_dir: Working directory
            db_path: Database path (auto-resolved if None)
            auto_activate_context: Whether to auto-activate contexts from plans
        """
        self.auto_activate_context = auto_activate_context
        self.active_context = None
        
        # Check for existing active context
        current_context = WorkflowContext.get_active_context()
        if current_context:
            self.active_context = current_context
            if db_path is None:
                db_path = str(current_context.db_path)
                
        # Initialize parent
        super().__init__(work_dir=work_dir, db_path=db_path or "materials.db")
        
        # Replace components with contextual versions if we have a context
        if self.active_context:
            self._setup_contextual_components()
    
    def _setup_contextual_components(self):
        """Set up context-aware components."""
        if not self.active_context:
            return
            
        # Replace database with contextual version
        self.db = self.active_context.get_database()
        
        # Replace queue manager with contextual version
        self.queue_manager = self.active_context.get_queue_manager(
            max_jobs=self.queue_manager.max_jobs,
            reserve_slots=self.queue_manager.reserve_slots,
            enable_tracking=True,
            enable_error_recovery=True
        )
        
        # Update directories to use context paths
        self.configs_dir = self.active_context.config_dir
        self.outputs_dir = self.active_context.output_dir
        self.temp_dir = self.active_context.temp_dir
    
    def execute_workflow_plan(self, plan_file: Path) -> Dict[str, Any]:
        """
        Execute workflow plan with automatic context handling.
        
        Extends parent method to handle context activation and cleanup.
        """
        print(f"Executing workflow plan: {plan_file}")
        
        # Load plan to check for context requirements
        import json
        with open(plan_file, 'r') as f:
            plan = json.load(f)
            
        # Check if plan requires context
        context_settings = plan.get('context_settings', {})
        requires_context = context_settings.get('requires_context', False)
        isolation_mode = context_settings.get('isolation_mode', 'shared')
        
        # Set up context if needed and not already active
        context_created_here = False
        if requires_context and isolation_mode == 'isolated' and not self.active_context:
            if self.auto_activate_context:
                # Create and activate context
                workflow_id = plan.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                print(f"\nActivating isolated context for workflow: {workflow_id}")
                
                self.active_context = WorkflowContext(
                    workflow_id=workflow_id,
                    base_dir=self.work_dir,
                    isolation_mode='isolated',
                    cleanup_on_exit=False  # We'll handle cleanup manually
                )
                
                # Enter context
                self.active_context.__enter__()
                context_created_here = True
                
                # Set up contextual components
                self._setup_contextual_components()
                
                # Update plan paths to use context directories
                self._update_plan_paths(plan)
            else:
                print("Warning: Plan requires isolated context but auto_activate_context is False")
        
        try:
            # Execute with parent method
            result = super().execute_workflow_plan(plan_file)
            
            # If we created a context, export results
            if context_created_here and self.active_context:
                self._export_and_cleanup_context(result)
                
            return result
            
        except Exception as e:
            # Ensure cleanup on error
            if context_created_here and self.active_context:
                print(f"\nError during execution, cleaning up context...")
                self.active_context.__exit__(type(e), e, None)
            raise
    
    def _update_plan_paths(self, plan: Dict[str, Any]):
        """Update plan paths to use context directories."""
        if not self.active_context:
            return
            
        # Update execution settings
        if 'execution_settings' in plan:
            plan['execution_settings']['output_directory'] = str(self.active_context.output_dir)
            plan['execution_settings']['script_directory'] = str(self.active_context.script_dir)
            
        # Update step configurations
        for step_config in plan.get('step_configurations', {}).values():
            if 'output_dir' in step_config:
                # Adjust output directory to be within context
                original_dir = Path(step_config['output_dir'])
                step_config['output_dir'] = str(self.active_context.output_dir / original_dir.name)
    
    def _export_and_cleanup_context(self, result: Dict[str, Any]):
        """Export results and optionally clean up context."""
        print("\n" + "=" * 60)
        print("Workflow Execution Complete - Context Management")
        print("=" * 60)
        
        # Export results to shared database
        export_file = self.active_context.export_results()
        print(f"✓ Exported results to: {export_file}")
        
        # Get user preference for cleanup
        if self._should_cleanup():
            print("\nCleaning up isolated resources...")
            self.active_context.cleanup()
            print("✓ Cleanup complete")
        else:
            print(f"\nIsolated resources preserved at: {self.active_context.workflow_root}")
            print("You can manually clean up later with:")
            print(f"  rm -rf {self.active_context.workflow_root}")
            
        # Exit context
        self.active_context.__exit__(None, None, None)
        self.active_context = None
    
    def _should_cleanup(self) -> bool:
        """Prompt user for cleanup preference."""
        # Check environment variable
        auto_cleanup = os.environ.get('MACE_AUTO_CLEANUP', '').lower()
        if auto_cleanup in ['true', '1', 'yes']:
            return True
        elif auto_cleanup in ['false', '0', 'no']:
            return False
            
        # Interactive prompt
        print("\nDo you want to clean up isolated workflow resources?")
        print("(This will remove the workflow-specific database and files)")
        response = input("Clean up? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    
    def execute_with_context(self, plan_file: Path, workflow_id: str = None,
                           isolation_mode: str = 'isolated') -> Dict[str, Any]:
        """
        Execute workflow with explicit context management.
        
        This method provides more control over context lifecycle.
        """
        if workflow_id is None:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        with workflow_context(workflow_id, base_dir=self.work_dir, 
                            isolation_mode=isolation_mode) as ctx:
            # Update components to use context
            self.active_context = ctx
            self._setup_contextual_components()
            
            # Execute workflow
            result = super().execute_workflow_plan(plan_file)
            
            # Export results before context closes
            if isolation_mode == 'isolated':
                export_file = ctx.export_results()
                print(f"Exported results to: {export_file}")
                
            return result
    
    def get_context_status(self) -> Dict[str, Any]:
        """Get current context status."""
        if self.active_context:
            return self.active_context.get_status()
        else:
            return {
                'has_context': False,
                'isolation_mode': 'shared',
                'message': 'No active workflow context'
            }


# Convenience function for creating contextual executor
def create_contextual_executor(**kwargs) -> ContextualWorkflowExecutor:
    """Create a contextual workflow executor with auto-detection."""
    return ContextualWorkflowExecutor(**kwargs)


# Context manager for isolated workflow execution
class IsolatedWorkflowExecution:
    """Context manager for isolated workflow execution."""
    
    def __init__(self, workflow_id: str, work_dir: str = ".", 
                 cleanup_on_exit: bool = True):
        self.workflow_id = workflow_id
        self.work_dir = Path(work_dir)
        self.cleanup_on_exit = cleanup_on_exit
        self.context = None
        self.executor = None
        
    def __enter__(self):
        # Create and enter context
        self.context = WorkflowContext(
            workflow_id=self.workflow_id,
            base_dir=self.work_dir,
            isolation_mode='isolated',
            cleanup_on_exit=self.cleanup_on_exit
        )
        self.context.__enter__()
        
        # Create executor using context
        self.executor = ContextualWorkflowExecutor(
            work_dir=str(self.work_dir),
            db_path=str(self.context.db_path)
        )
        self.executor.active_context = self.context
        self.executor._setup_contextual_components()
        
        return self.executor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Export results if no exception
        if exc_type is None and self.context:
            self.context.export_results()
            
        # Exit context (handles cleanup)
        if self.context:
            self.context.__exit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    # Example usage
    print("Testing ContextualWorkflowExecutor...")
    
    # Test isolated execution
    with IsolatedWorkflowExecution("test_workflow_002") as executor:
        print(f"Executor context status: {executor.get_context_status()}")
        print(f"Database path: {executor.db.db_path}")
        
    print("\nIsolated execution complete")