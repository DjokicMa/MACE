#!/usr/bin/env python3
"""
Context-Aware Workflow Planner Extension
========================================
Extensions to the workflow planner to support isolated execution contexts.

Author: Workflow isolation enhancement
"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from mace.workflow.planner import WorkflowPlanner
from mace.workflow.context import WorkflowContext, workflow_context
from mace.database.materials_contextual import ContextualMaterialDatabase


class ContextualWorkflowPlanner(WorkflowPlanner):
    """
    Extended workflow planner with context isolation support.
    
    Features:
    - Automatic workflow context creation
    - Isolated resource management
    - Configuration-based isolation mode selection
    - Backward compatible with existing workflows
    """
    
    def __init__(self, work_dir: str = ".", db_path: str = None, 
                 isolation_mode: str = None, workflow_context: WorkflowContext = None):
        """
        Initialize contextual workflow planner.
        
        Args:
            work_dir: Working directory
            db_path: Database path (auto-resolved if None)
            isolation_mode: "isolated", "shared", or None (auto-detect)
            workflow_context: Existing workflow context to use
        """
        # Determine isolation mode
        if isolation_mode is None:
            isolation_mode = self._determine_isolation_mode()
            
        self.isolation_mode = isolation_mode
        self.workflow_context = workflow_context
        
        # If using isolation and no context provided, we'll create one later
        if self.isolation_mode == "isolated" and self.workflow_context is None:
            # Don't create context yet - wait for workflow planning
            self.pending_context = True
            # Use temporary shared database for initial setup
            if db_path is None:
                db_path = "materials.db"
        else:
            self.pending_context = False
            # Use contextual database
            if db_path is None and self.workflow_context:
                db_path = str(self.workflow_context.db_path)
                
        # Initialize parent with resolved paths
        super().__init__(work_dir=work_dir, db_path=db_path)
        
        # Replace database with contextual version if we have a context
        if self.workflow_context:
            self.db = ContextualMaterialDatabase.from_context(self.workflow_context)
    
    def _determine_isolation_mode(self) -> str:
        """Determine isolation mode from environment or config."""
        # Check environment variable
        env_mode = os.environ.get('MACE_ISOLATION_MODE')
        if env_mode in ['isolated', 'shared']:
            return env_mode
            
        # Check for config file
        config_file = Path.cwd() / ".mace_config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                mode = config.get('isolation_mode', 'shared')
                if mode in ['isolated', 'shared']:
                    return mode
                    
        # Default to shared for backward compatibility
        return 'shared'
    
    def create_workflow_context(self, workflow_id: str) -> WorkflowContext:
        """Create a new workflow context."""
        context = WorkflowContext(
            workflow_id=workflow_id,
            base_dir=self.work_dir,
            isolation_mode=self.isolation_mode,
            cleanup_on_exit=False  # Don't auto-cleanup, let executor handle it
        )
        
        # Copy relevant data from shared database if in isolated mode
        if self.isolation_mode == "isolated":
            context.copy_from_shared(self.db_path)
            
        return context
    
    def plan_workflow(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Plan workflow with optional context creation.
        
        Extends parent plan_workflow to add context information.
        """
        # Get the base workflow plan
        plan = super().plan_workflow(*args, **kwargs)
        
        # Add context information
        plan['context_settings'] = {
            'isolation_mode': self.isolation_mode,
            'requires_context': self.isolation_mode == 'isolated',
            'context_created': False
        }
        
        # If we need isolation, create context now
        if self.pending_context and self.isolation_mode == 'isolated':
            workflow_id = plan.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.workflow_context = self.create_workflow_context(workflow_id)
            
            # Update to use isolated database
            self.db = self.workflow_context.get_database()
            
            # Update plan with context info
            plan['context_settings']['context_created'] = True
            plan['context_settings']['context_id'] = workflow_id
            plan['context_settings']['context_root'] = str(self.workflow_context.workflow_root)
            
            # Update directory paths in plan to use isolated directories
            if 'execution_settings' in plan:
                plan['execution_settings']['output_directory'] = str(self.workflow_context.output_dir)
                plan['execution_settings']['script_directory'] = str(self.workflow_context.script_dir)
                
        return plan
    
    def save_workflow_plan(self, plan: Dict[str, Any], plan_file: Path = None) -> Path:
        """Save workflow plan with context information."""
        # If we have a context, save to context directory
        if self.workflow_context:
            if plan_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plan_file = self.workflow_context.config_dir / f"workflow_plan_{timestamp}.json"
            else:
                # Ensure plan is saved in context directory
                plan_file = self.workflow_context.config_dir / plan_file.name
                
        # Use parent method to actually save
        return super().save_workflow_plan(plan, plan_file)
    
    def get_isolation_options(self) -> Dict[str, str]:
        """Get available isolation options for user selection."""
        return {
            "1": "Shared resources (default) - Use common database and directories",
            "2": "Isolated resources - Create workflow-specific database and directories",
            "3": "Auto-detect from environment/config"
        }
    
    def prompt_isolation_mode(self) -> str:
        """Prompt user for isolation mode selection."""
        print("\nStep 0: Resource Isolation Mode")
        print("-" * 40)
        
        options = self.get_isolation_options()
        choice = get_user_input("Select resource isolation mode", options, "1")
        
        mode_map = {
            "1": "shared",
            "2": "isolated", 
            "3": self._determine_isolation_mode()
        }
        
        selected_mode = mode_map[choice]
        print(f"Selected mode: {selected_mode}")
        
        return selected_mode
    
    def cleanup_workflow_context(self):
        """Clean up workflow context if it was created."""
        if self.workflow_context and self.isolation_mode == "isolated":
            # Export results before cleanup
            export_file = self.workflow_context.export_results()
            print(f"Exported workflow results to: {export_file}")
            
            # Perform cleanup
            self.workflow_context.cleanup()


# Helper function for backward compatibility
def get_user_input(prompt: str, options: Dict[str, str], default: str) -> str:
    """Get user input with options display."""
    print(f"\n{prompt}:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    
    choice = input(f"Choice [{default}]: ").strip()
    return choice if choice else default


# Convenience function for creating contextual planner
def create_contextual_planner(**kwargs) -> ContextualWorkflowPlanner:
    """Create a contextual workflow planner with auto-detection."""
    # Check if we're in an active workflow context
    current_context = WorkflowContext.get_active_context()
    if current_context:
        kwargs['workflow_context'] = current_context
        kwargs['isolation_mode'] = current_context.isolation_mode
        
    return ContextualWorkflowPlanner(**kwargs)


if __name__ == "__main__":
    # Test contextual planner
    print("Testing ContextualWorkflowPlanner...")
    
    # Test with different isolation modes
    for mode in ['shared', 'isolated']:
        print(f"\nTesting with {mode} mode:")
        planner = ContextualWorkflowPlanner(isolation_mode=mode)
        print(f"  Database path: {planner.db.resolved_db_path}")
        print(f"  Isolation mode: {planner.isolation_mode}")