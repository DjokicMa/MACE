#!/usr/bin/env python3
"""
Context-Aware Material Database for MACE
========================================
Extension of MaterialDatabase that supports workflow-specific contexts
while maintaining backward compatibility.

Author: Workflow isolation enhancement
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

# Import the original MaterialDatabase
from mace.database.materials import MaterialDatabase

# Import context management
sys.path.insert(0, str(Path(__file__).parent.parent))
from mace.workflow.context import get_current_context, require_context


class ContextualMaterialDatabase(MaterialDatabase):
    """
    Extension of MaterialDatabase that supports workflow contexts.
    
    Features:
    - Automatic context detection from environment
    - Seamless switching between isolated and shared databases
    - Full backward compatibility with existing code
    - Context-aware path resolution
    """
    
    def __init__(self, db_path: str = None, ase_db_path: str = None, 
                 workflow_context: 'WorkflowContext' = None, 
                 auto_detect_context: bool = True):
        """
        Initialize context-aware material database.
        
        Args:
            db_path: Database path (overrides context if provided)
            ase_db_path: ASE database path (overrides context if provided)
            workflow_context: Explicit workflow context to use
            auto_detect_context: Whether to auto-detect context from environment
        """
        # Try to detect context if not provided
        if workflow_context is None and auto_detect_context:
            workflow_context = get_current_context()
            
        self.workflow_context = workflow_context
        
        # Resolve paths based on context
        if workflow_context and db_path is None:
            # Use context's database path
            db_path = str(workflow_context.get_database_path())
            if workflow_context.isolation_mode != 'shared':
                print(f"Using context database: {db_path}")
        elif db_path is None:
            # Default to standard path
            db_path = "materials.db"
            
        if workflow_context and ase_db_path is None:
            # Use context's ASE database path
            ase_db_path = str(workflow_context.get_ase_database_path())
        elif ase_db_path is None:
            # Default to standard path
            ase_db_path = "structures.db"
            
        # Initialize parent class with resolved paths
        # If we have a workflow context and it's isolated, delay initialization
        # to prevent creating the root materials.db
        auto_init = True
        if workflow_context and workflow_context.isolation_mode != 'shared':
            # In isolated mode, let the context create the database
            auto_init = False
        elif db_path == "materials.db" and not Path(db_path).exists():
            # If using default path and it doesn't exist, check if we might get a context later
            auto_init = False
            
        super().__init__(db_path=db_path, ase_db_path=ase_db_path, auto_initialize=auto_init)
        
        # Store resolved paths for reference
        self.resolved_db_path = db_path
        self.resolved_ase_db_path = ase_db_path
    
    def _detect_workflow_context(self) -> Optional['WorkflowContext']:
        """Detect workflow context from environment."""
        # This method is no longer needed as we use get_current_context()
        return get_current_context()
    
    def _resolve_db_path(self) -> str:
        """Resolve database path based on context."""
        # Check environment variable first
        env_path = os.environ.get('MACE_DB_PATH')
        if env_path:
            return env_path
            
        # Use workflow context if available
        if self.workflow_context:
            return str(self.workflow_context.get_database_path())
            
        # Default to standard path
        return "materials.db"
    
    def _resolve_ase_db_path(self) -> str:
        """Resolve ASE database path based on context."""
        # If we have a workflow context, use its ASE database path
        if self.workflow_context:
            return str(self.workflow_context.get_ase_database_path())
            
        # Default to standard path
        return "structures.db"
    
    def get_context_info(self) -> dict:
        """Get information about the current database context."""
        return {
            'has_context': self.workflow_context is not None,
            'workflow_id': self.workflow_context.workflow_id if self.workflow_context else None,
            'isolation_mode': self.workflow_context.isolation_mode if self.workflow_context else 'shared',
            'db_path': self.resolved_db_path,
            'ase_db_path': self.resolved_ase_db_path,
            'context_dir': str(self.workflow_context.context_dir) if self.workflow_context else None,
            'is_active': self.workflow_context.is_active if self.workflow_context and hasattr(self.workflow_context, 'is_active') else True if self.workflow_context else False
        }
    
    def is_isolated(self) -> bool:
        """Check if database is running in isolated mode."""
        return (self.workflow_context is not None and 
                self.workflow_context.isolation_mode != 'shared')
    
    def copy_to_context(self, target_context: 'WorkflowContext', 
                       material_ids: list = None, 
                       include_calculations: bool = True,
                       include_properties: bool = True):
        """
        Copy data to another workflow context.
        
        Args:
            target_context: Target workflow context
            material_ids: Specific materials to copy (None for all)
            include_calculations: Whether to copy calculations
            include_properties: Whether to copy properties
        """
        # Create a new database instance for the target context
        target_db = ContextualMaterialDatabase(workflow_context=target_context)
        
        # Get materials to copy
        if material_ids:
            materials = [self.get_material(mid) for mid in material_ids if self.get_material(mid)]
        else:
            materials = self.get_all_materials()
            
        # Copy materials
        for material in materials:
            if not target_db.get_material(material['material_id']):
                target_db.create_or_update_material(
                    material_id=material['material_id'],
                    formula=material['formula'],
                    space_group=material.get('space_group'),
                    dimensionality=material.get('dimensionality', 'CRYSTAL'),
                    source_type=material.get('source_type'),
                    source_file=material.get('source_file'),
                    metadata=material.get('metadata_json')
                )
                
            # Copy calculations if requested
            if include_calculations:
                calcs = self.get_calculations_by_material(material['material_id'])
                for calc in calcs:
                    if not target_db.get_calculation(calc['calc_id']):
                        # Create calculation in target
                        new_calc_id = target_db.create_calculation(
                            material_id=calc['material_id'],
                            calc_type=calc['calc_type'],
                            calc_subtype=calc.get('calc_subtype'),
                            input_file=calc.get('input_file'),
                            work_dir=calc.get('work_dir'),
                            settings=calc.get('settings_json')
                        )
                        
                        # Update status if calculation was completed
                        if calc.get('status') in ['completed', 'failed']:
                            target_db.update_calculation_status(
                                calc_id=new_calc_id,
                                status=calc['status'],
                                slurm_job_id=calc.get('slurm_job_id'),
                                slurm_state=calc.get('slurm_state'),
                                output_file=calc.get('output_file'),
                                exit_code=calc.get('exit_code'),
                                error_type=calc.get('error_type'),
                                error_message=calc.get('error_message')
                            )
                            
            # Copy properties if requested
            if include_properties:
                props = self.get_material_properties(material['material_id'])
                for prop in props:
                    target_db.store_material_property(
                        material_id=prop['material_id'],
                        calc_id=prop.get('calc_id'),
                        property_name=prop['property_name'],
                        property_value=prop['property_value'],
                        unit=prop.get('unit'),
                        conditions=prop.get('conditions_json')
                    )
    
    @classmethod
    def from_context(cls, workflow_context: 'WorkflowContext' = None, require: bool = False, **kwargs):
        """
        Create database instance from workflow context.
        
        Args:
            workflow_context: Workflow context to use (auto-detect if None)
            require: If True, raise error if no context is active
            **kwargs: Additional arguments for MaterialDatabase
        """
        if workflow_context is None:
            if require:
                workflow_context = require_context()
            else:
                workflow_context = get_current_context()
        return cls(workflow_context=workflow_context, **kwargs)
    
    def get_workflow_materials(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all materials for a specific workflow.
        
        Args:
            workflow_id: Workflow ID to filter by. If None and context is active,
                        uses current context's workflow ID.
                        
        Returns:
            List of material records
        """
        if workflow_id is None and self.workflow_context:
            workflow_id = self.workflow_context.workflow_id
            
        if workflow_id:
            # Filter calculations by workflow_id in settings
            query = """
                SELECT DISTINCT m.*
                FROM materials m
                JOIN calculations c ON m.material_id = c.material_id
                WHERE json_extract(c.settings_json, '$.workflow_id') = ?
            """
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (workflow_id,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results
        else:
            # No workflow filter - return all materials
            return self.get_all_materials()
            
    def get_workflow_calculations(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all calculations for a specific workflow.
        
        Args:
            workflow_id: Workflow ID to filter by. If None and context is active,
                        uses current context's workflow ID.
                        
        Returns:
            List of calculation records
        """
        if workflow_id is None and self.workflow_context:
            workflow_id = self.workflow_context.workflow_id
            
        if workflow_id:
            query = """
                SELECT *
                FROM calculations
                WHERE json_extract(settings_json, '$.workflow_id') = ?
                ORDER BY started_at DESC
            """
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (workflow_id,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                    
                return results
        else:
            # No workflow filter - return recent calculations
            return self.get_recent_calculations(limit=100)


# Convenience function for backward compatibility
def get_contextual_database(db_path: str = None, **kwargs) -> ContextualMaterialDatabase:
    """
    Get a context-aware database instance.
    
    This function provides backward compatibility while enabling context awareness.
    """
    return ContextualMaterialDatabase(db_path=db_path, **kwargs)


# Monkey-patch option for complete backward compatibility
def enable_contextual_databases():
    """
    Enable contextual databases globally by replacing MaterialDatabase.
    
    WARNING: This modifies the module-level MaterialDatabase class.
    Use with caution in production code.
    """
    import mace.database.materials
    mace.database.materials.MaterialDatabase = ContextualMaterialDatabase
    print("Contextual databases enabled globally")


if __name__ == "__main__":
    # Test contextual database
    print("Testing ContextualMaterialDatabase...")
    
    # Test without context (should work like normal MaterialDatabase)
    db1 = ContextualMaterialDatabase()
    print(f"Context info (no context): {db1.get_context_info()}")
    
    # Test with simulated context
    os.environ['MACE_WORKFLOW_ID'] = 'test_workflow'
    os.environ['MACE_DB_PATH'] = '/tmp/test_workflow.db'
    
    db2 = ContextualMaterialDatabase()
    print(f"Context info (with env): {db2.get_context_info()}")
    
    # Clean up
    os.environ.pop('MACE_WORKFLOW_ID', None)
    os.environ.pop('MACE_DB_PATH', None)