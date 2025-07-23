#!/usr/bin/env python3
"""
Test Script for Workflow Isolation Features
===========================================
Demonstrates and tests the workflow isolation capabilities.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mace.workflow.context import WorkflowContext, workflow_context
from mace.database.materials_contextual import ContextualMaterialDatabase
from mace.workflow.planner_contextual import ContextualWorkflowPlanner
from mace.workflow.executor_contextual import ContextualWorkflowExecutor


def test_basic_context():
    """Test basic context creation and cleanup."""
    print("\n" + "="*60)
    print("Test 1: Basic Context Creation")
    print("="*60)
    
    workflow_id = f"test_basic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with workflow_context(workflow_id, isolation_mode="isolated") as ctx:
        print(f"✓ Created context: {workflow_id}")
        print(f"  - Root directory: {ctx.workflow_root}")
        print(f"  - Database path: {ctx.db_path}")
        print(f"  - Config directory: {ctx.config_dir}")
        
        # Test database creation
        db = ctx.get_database()
        print(f"✓ Database created successfully")
        
        # Create test material
        material_id = db.create_material(
            material_id="test_material",
            formula="Al2O3",
            space_group=167,
            metadata={"test": True}
        )
        print(f"✓ Created test material: {material_id}")
        
        # Verify material exists
        material = db.get_material(material_id)
        assert material is not None
        print(f"✓ Material retrieved successfully")
        
    print("✓ Context closed successfully")
    

def test_context_isolation():
    """Test that contexts are truly isolated."""
    print("\n" + "="*60)
    print("Test 2: Context Isolation")
    print("="*60)
    
    # Create two isolated contexts
    ctx1_id = "test_isolation_1"
    ctx2_id = "test_isolation_2"
    
    # Create material in context 1
    with workflow_context(ctx1_id, isolation_mode="isolated", cleanup_on_exit=False) as ctx1:
        db1 = ctx1.get_database()
        db1.create_material("material_1", "H2O", metadata={"context": 1})
        print(f"✓ Created material in context 1")
        
    # Create material in context 2
    with workflow_context(ctx2_id, isolation_mode="isolated", cleanup_on_exit=False) as ctx2:
        db2 = ctx2.get_database()
        db2.create_material("material_2", "CO2", metadata={"context": 2})
        print(f"✓ Created material in context 2")
        
    # Verify isolation
    with workflow_context(ctx1_id, isolation_mode="isolated", cleanup_on_exit=False) as ctx1:
        db1 = ctx1.get_database()
        
        # Should find material_1 but not material_2
        mat1 = db1.get_material("material_1")
        mat2 = db1.get_material("material_2")
        
        assert mat1 is not None, "Material 1 should exist in context 1"
        assert mat2 is None, "Material 2 should NOT exist in context 1"
        print(f"✓ Context 1 isolation verified")
        
    # Cleanup
    for ctx_id in [ctx1_id, ctx2_id]:
        ctx = WorkflowContext(ctx_id, isolation_mode="isolated")
        ctx.cleanup(force=True)
        
    print("✓ Cleanup completed")


def test_shared_mode():
    """Test shared mode compatibility."""
    print("\n" + "="*60)
    print("Test 3: Shared Mode")
    print("="*60)
    
    # Create context in shared mode
    with workflow_context("test_shared", isolation_mode="shared") as ctx:
        print(f"✓ Created shared context")
        print(f"  - Database path: {ctx.db_path}")
        
        db = ctx.get_database()
        
        # In shared mode, should use the default database
        assert "materials.db" in str(ctx.db_path)
        print(f"✓ Using shared database correctly")
        
        # Create a test material
        material_id = f"shared_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.create_material(material_id, "NaCl", metadata={"shared": True})
        print(f"✓ Created material in shared database: {material_id}")
        
    # Verify material persists (since it's in shared database)
    db_direct = ContextualMaterialDatabase()
    material = db_direct.get_material(material_id)
    assert material is not None
    print(f"✓ Material persists in shared database")


def test_workflow_planning_with_isolation():
    """Test workflow planning with isolation."""
    print("\n" + "="*60)
    print("Test 4: Workflow Planning with Isolation")
    print("="*60)
    
    # Create temporary directory with test CIF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a dummy CIF file
        cif_path = temp_path / "test.cif"
        cif_path.write_text("""
data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al 0.0 0.0 0.0
""")
        
        # Create planner with isolation
        planner = ContextualWorkflowPlanner(
            work_dir=temp_dir,
            isolation_mode="isolated"
        )
        
        print(f"✓ Created isolated workflow planner")
        
        # Simulate planning (simplified)
        plan = {
            "workflow_id": "test_planning",
            "input_type": "cif",
            "input_files": {"cif": [str(cif_path)]},
            "workflow_sequence": ["OPT", "SP"],
            "context_settings": {
                "isolation_mode": "isolated",
                "requires_context": True
            }
        }
        
        # Save plan
        plan_file = planner.save_workflow_plan(plan)
        print(f"✓ Saved workflow plan: {plan_file}")
        
        # Verify plan was saved to context directory
        if planner.workflow_context:
            assert plan_file.parent == planner.workflow_context.config_dir
            print(f"✓ Plan saved to isolated config directory")


def test_contextual_database():
    """Test contextual database features."""
    print("\n" + "="*60)
    print("Test 5: Contextual Database")
    print("="*60)
    
    # Test auto-detection
    os.environ['MACE_WORKFLOW_ID'] = 'test_auto_detect'
    os.environ['MACE_DB_PATH'] = '/tmp/test_auto.db'
    os.environ['MACE_ISOLATION_MODE'] = 'isolated'
    
    db = ContextualMaterialDatabase()
    info = db.get_context_info()
    
    print(f"✓ Auto-detected context:")
    print(f"  - Workflow ID: {info['workflow_id']}")
    print(f"  - Database path: {info['db_path']}")
    print(f"  - Isolation mode: {info['isolation_mode']}")
    
    # Cleanup environment
    for key in ['MACE_WORKFLOW_ID', 'MACE_DB_PATH', 'MACE_ISOLATION_MODE']:
        os.environ.pop(key, None)
        
    print(f"✓ Environment cleaned up")


def test_result_export():
    """Test exporting results from isolated context."""
    print("\n" + "="*60)
    print("Test 6: Result Export")
    print("="*60)
    
    workflow_id = "test_export"
    
    # Create and populate isolated context
    with workflow_context(workflow_id, isolation_mode="isolated", cleanup_on_exit=False) as ctx:
        db = ctx.get_database()
        
        # Create some test data
        for i in range(3):
            material_id = f"export_test_{i}"
            db.create_material(
                material_id=material_id,
                formula=f"Test{i}",
                metadata={"index": i}
            )
            
            # Create calculation
            calc_id = db.create_calculation(
                material_id=material_id,
                calc_type="OPT",
                settings={"test": True}
            )
            
            # Mark as completed
            db.update_calculation_status(calc_id, "completed")
            
        print(f"✓ Created test data in isolated context")
        
        # Export results
        export_file = ctx.export_results()
        print(f"✓ Exported results to: {export_file}")
        
        # Verify export
        with open(export_file, 'r') as f:
            export_data = json.load(f)
            
        assert len(export_data['materials']) == 3
        assert len(export_data['calculations']) == 3
        print(f"✓ Export contains correct data")
        
    # Cleanup
    ctx = WorkflowContext(workflow_id, isolation_mode="isolated")
    ctx.cleanup(force=True)
    print(f"✓ Cleanup completed")


def run_all_tests():
    """Run all isolation tests."""
    print("\n" + "="*80)
    print("MACE Workflow Isolation Test Suite")
    print("="*80)
    
    tests = [
        test_basic_context,
        test_context_isolation,
        test_shared_mode,
        test_workflow_planning_with_isolation,
        test_contextual_database,
        test_result_export
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)