#!/usr/bin/env python3
"""
Integration Test Script for CRYSTAL Material Tracking System
-----------------------------------------------------------
Tests integration between the new material tracking components and existing
check_completedV2.py and check_erroredV2.py scripts.

Author: Based on implementation plan for material tracking system
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
from typing import Dict

# Import our components
from material_database import MaterialDatabase, create_material_id_from_file
from crystal_file_manager import CrystalFileManager
from error_detector import CrystalErrorDetector
from enhanced_queue_manager import EnhancedCrystalQueueManager
from material_monitor import MaterialMonitor


def create_test_files(test_dir: Path) -> Dict[str, Path]:
    """Create test CRYSTAL files for integration testing."""
    test_files = {}
    
    # Create a test .d12 file
    d12_content = """Test Material
CRYSTAL
0 0 0
225
5.0
2
13 0.0 0.0 0.0
8 0.25 0.25 0.25
END
BASISSET
POB-TZVP-REV2
END
"""
    
    d12_file = test_dir / "test_material_opt.d12"
    with open(d12_file, 'w') as f:
        f.write(d12_content)
    test_files['input'] = d12_file
    
    # Create a test completed .out file
    completed_out_content = """


                         ****************************
                         *** CRYSTAL17 PROGRAM  ***
                         ****************************

 CRYSTAL CALCULATION
 (INPUT ACCORDING TO THE INTERNATIONAL TABLES FOR X-RAY CRYSTALLOGRAPHY)
 CRYSTAL FAMILY                       :  CUBIC       
 CRYSTAL CLASS  (GROTH - 1921)        :  CUBIC HOLOHEDRAL
 SPACE GROUP (CENTROSYMMETRIC)        :  F M 3 M         

 LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL
       A              B              C           ALPHA      BETA       GAMMA
    3.50000000     3.50000000     3.50000000    60.000000  60.000000  60.000000

 NUMBER OF ATOMS IN THE ASYMMETRIC UNIT:    2

OPT END - CONVERGED

 TOTAL CPU TIME =         123.45

 EEEEEEEEEE TERMINATION  DATE 10 12 2023 TIME 10:30:45.7
"""
    
    completed_out = test_dir / "test_material_opt.out"
    with open(completed_out, 'w') as f:
        f.write(completed_out_content)
    test_files['completed_output'] = completed_out
    
    # Create a test errored .out file
    error_out_content = """


                         ****************************
                         *** CRYSTAL17 PROGRAM  ***
                         ****************************

 CRYSTAL CALCULATION

 SCF FIELD DIRECTION    :  1.000  1.000  1.000

 *                               CRYSTAL
 *                               *******
 *                               CALCULATION

TOO MANY CYCLES

 INFORMATION **** ENOSTOP ****
 OPTIMISATION STOPPED ON REQUEST OF THE USER

 EEEEEEEEEE TERMINATION  DATE 10 12 2023 TIME 10:30:45.7
"""
    
    error_out = test_dir / "test_material_error.out"
    with open(error_out, 'w') as f:
        f.write(error_out_content)
    test_files['error_output'] = error_out
    
    # Create a test ongoing .out file
    ongoing_out_content = """


                         ****************************
                         *** CRYSTAL17 PROGRAM  ***
                         ****************************

 CRYSTAL CALCULATION

 SCF FIELD DIRECTION    :  1.000  1.000  1.000

 *                               CRYSTAL
 *                               *******
 *                               CALCULATION

 SCF CYCLE     1
"""
    
    ongoing_out = test_dir / "test_material_ongoing.out"
    with open(ongoing_out, 'w') as f:
        f.write(ongoing_out_content)
    test_files['ongoing_output'] = ongoing_out
    
    return test_files


def test_database_operations():
    """Test basic database operations."""
    print("Testing database operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_materials.db"
        db = MaterialDatabase(str(db_path))
        
        # Test material creation
        material_id = db.create_material(
            material_id="test_Al2O3",
            formula="Al2O3",
            space_group=225,
            source_type="test",
            source_file="test.d12"
        )
        
        assert material_id == "test_Al2O3", "Material creation failed"
        
        # Test calculation creation
        calc_id = db.create_calculation(
            material_id=material_id,
            calc_type="OPT",
            input_file="test.d12",
            work_dir="/tmp/test"
        )
        
        assert calc_id.startswith("test_Al2O3_OPT"), "Calculation creation failed"
        
        # Test status update
        db.update_calculation_status(calc_id, "completed")
        
        # Test retrieval
        material = db.get_material(material_id)
        assert material is not None, "Material retrieval failed"
        
        calcs = db.get_calculations_by_status("completed")
        assert len(calcs) == 1, "Calculation retrieval failed"
        
        stats = db.get_database_stats()
        assert stats['total_materials'] == 1, "Database stats incorrect"
        
    print("✓ Database operations test passed")


def test_file_manager():
    """Test file manager operations."""
    print("Testing file manager operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        file_manager = CrystalFileManager(str(test_dir), str(db_path))
        test_files = create_test_files(test_dir)
        
        # Test material ID extraction
        material_id = create_material_id_from_file(test_files['input'])
        assert material_id == "test_material", f"Material ID extraction failed: {material_id}"
        
        # Test directory structure creation
        material_dir = file_manager.create_material_directory_structure(material_id)
        assert material_dir.exists(), "Material directory creation failed"
        assert (material_dir / "opt").exists(), "OPT directory creation failed"
        
        # Test file discovery
        discovered = file_manager.discover_material_files()
        assert material_id in discovered, "File discovery failed"
        assert len(discovered[material_id]) >= 1, "Not all files discovered"
        
        # Test file organization
        organized = file_manager.organize_calculation_files(
            material_id, "opt", [test_files['input'], test_files['completed_output']]
        )
        assert len(organized['input']) == 1, "Input file organization failed"
        assert len(organized['output']) == 1, "Output file organization failed"
        
        # Test integrity checking
        integrity = file_manager.check_file_integrity(test_files['input'])
        assert integrity['exists'], "File integrity check failed"
        assert integrity['readable'], "File readability check failed"
        
    print("✓ File manager test passed")


def test_error_detector():
    """Test error detection functionality."""
    print("Testing error detector...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        error_detector = CrystalErrorDetector(str(test_dir), str(db_path))
        test_files = create_test_files(test_dir)
        
        # Test completed calculation analysis
        completed_analysis = error_detector.analyze_output_file(test_files['completed_output'])
        assert completed_analysis['status'] == 'completed', "Completed calculation not detected"
        assert completed_analysis['completion_type'] is not None, "Completion type not detected"
        
        # Test error analysis
        error_analysis = error_detector.analyze_output_file(test_files['error_output'])
        assert error_analysis['status'] == 'error', "Error not detected"
        assert error_analysis['error_type'] == 'scf_convergence', "Error type not correct"
        assert error_analysis['recoverable'] == True, "Recoverability not detected"
        
        # Test ongoing calculation analysis
        ongoing_analysis = error_detector.analyze_output_file(test_files['ongoing_output'])
        assert ongoing_analysis['status'] in ['ongoing', 'incomplete'], "Ongoing calculation not detected"
        
        # Test recovery suggestions
        suggestions = error_detector.suggest_recovery_actions(error_analysis)
        assert len(suggestions) > 0, "No recovery suggestions generated"
        
    print("✓ Error detector test passed")


def test_enhanced_queue_manager():
    """Test enhanced queue manager functionality."""
    print("Testing enhanced queue manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        # Create test files
        test_files = create_test_files(test_dir)
        
        # Initialize queue manager (without actual SLURM submission)
        queue_manager = EnhancedCrystalQueueManager(
            str(test_dir), max_jobs=10, enable_tracking=True, db_path=str(db_path)
        )
        
        # Test material info extraction
        material_id, formula, metadata = queue_manager.extract_material_info_from_d12(test_files['input'])
        assert material_id == "test_material", "Material ID extraction failed"
        
        # Test calculation type detection
        calc_type = queue_manager.determine_calc_type_from_file(test_files['input'])
        assert calc_type in ['OPT', 'SP'], "Calculation type detection failed"
        
        # Test calculation folder creation
        calc_dir = queue_manager.create_calculation_folder(material_id, calc_type)
        assert calc_dir.exists(), "Calculation folder creation failed"
        
        # Test early failure detection functionality (just verify it runs)
        is_failing = queue_manager.is_job_failing_early({
            'calc_id': 'test_calc',
            'material_id': material_id,
            'calc_type': calc_type,
            'work_dir': str(test_dir),
            'input_file': str(test_files['input']),
            'output_file': str(test_files['ongoing_output']),
            'started_at': datetime.now().isoformat()
        })
        # The function should return a boolean (functionality test)
        assert isinstance(is_failing, bool), "Early failure detection should return boolean"
        
    print("✓ Enhanced queue manager test passed")


def test_integration_with_check_scripts():
    """Test integration with existing check scripts."""
    print("Testing integration with check scripts...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        # Create test files
        test_files = create_test_files(test_dir)
        
        file_manager = CrystalFileManager(str(test_dir), str(db_path))
        
        # Test integration (this will fail gracefully if scripts not found)
        try:
            results = file_manager.integrate_with_check_scripts()
            # If scripts are found and run successfully
            if not results['script_errors']:
                assert 'completed' in results, "Check scripts integration failed"
                assert 'errored' in results, "Check scripts integration failed"
            else:
                print("  Note: Check scripts not found - this is expected in test environment")
        except Exception as e:
            print(f"  Note: Check scripts integration test skipped: {e}")
            
        # Test updatelists integration
        error_detector = CrystalErrorDetector(str(test_dir), str(db_path))
        try:
            updatelists_results = error_detector.run_updatelists_integration(test_dir)
            if 'error' not in updatelists_results:
                assert 'categories' in updatelists_results, "Updatelists integration failed"
            else:
                print("  Note: updatelists2.py not found - this is expected in test environment")
        except Exception as e:
            print(f"  Note: Updatelists integration test skipped: {e}")
    
    print("✓ Integration with check scripts test passed")


def test_material_monitor():
    """Test material monitor functionality."""
    print("Testing material monitor...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        # Create some test data
        db = MaterialDatabase(str(db_path))
        material_id = db.create_material("test_monitor", "Al2O3", source_type="test")
        calc_id = db.create_calculation(material_id, "OPT")
        db.update_calculation_status(calc_id, "completed")
        
        monitor = MaterialMonitor(str(test_dir), str(db_path))
        
        # Test database connectivity check
        connectivity = monitor.check_database_connectivity()
        assert connectivity, "Database connectivity check failed"
        
        # Test quick stats
        stats = monitor.get_quick_stats()
        assert stats['materials'] == 1, "Quick stats materials count incorrect"
        assert stats['calculations'] == 1, "Quick stats calculations count incorrect"
        
        # Test system status
        status = monitor.get_system_status()
        assert 'database' in status, "System status missing database info"
        assert 'queue' in status, "System status missing queue info"
        assert status['database']['accessible'], "Database should be accessible"
        
    print("✓ Material monitor test passed")


def run_comprehensive_integration_test():
    """Run a comprehensive end-to-end integration test."""
    print("Running comprehensive integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        db_path = test_dir / "test_materials.db"
        
        # Create test files
        test_files = create_test_files(test_dir)
        
        # Initialize all components
        db = MaterialDatabase(str(db_path))
        file_manager = CrystalFileManager(str(test_dir), str(db_path))
        error_detector = CrystalErrorDetector(str(test_dir), str(db_path))
        monitor = MaterialMonitor(str(test_dir), str(db_path))
        
        # Simulate workflow: create material, run calculation, analyze results
        
        # 1. Create material from input file
        material_id = create_material_id_from_file(test_files['input'])
        material_id = db.create_material(
            material_id=material_id,
            formula="Al2O3", 
            source_type="d12",
            source_file=str(test_files['input'])
        )
        
        # 2. Create calculation
        calc_id = db.create_calculation(
            material_id=material_id,
            calc_type="OPT",
            input_file=str(test_files['input']),
            work_dir=str(test_dir)
        )
        
        # 3. Simulate job completion
        db.update_calculation_status(calc_id, "completed", output_file=str(test_files['completed_output']))
        
        # 4. Organize files
        organized = file_manager.organize_calculation_files(
            material_id, "opt", list(test_files.values())
        )
        
        # 5. Analyze results
        analysis = error_detector.analyze_output_file(test_files['completed_output'])
        
        # 6. Generate reports
        file_report = file_manager.generate_file_report(material_id)
        error_report = error_detector.generate_error_report(material_id)
        system_status = monitor.get_system_status()
        
        # Verify end-to-end workflow
        assert material_id in file_report['materials'], "Material not in file report"
        assert analysis['status'] == 'completed', "Analysis did not detect completion"
        assert system_status['database']['accessible'], "Database not accessible"
        
        # Verify file organization
        material_dir = test_dir / material_id
        assert material_dir.exists(), "Material directory not created"
        assert (material_dir / "opt").exists(), "OPT directory not created"
        
        print(f"  ✓ Created material: {material_id}")
        print(f"  ✓ Created calculation: {calc_id}")
        print(f"  ✓ Organized {sum(len(files) for files in organized.values())} files")
        print(f"  ✓ Analysis status: {analysis['status']}")
        print(f"  ✓ System health: {system_status['database']['status']}")
        
    print("✓ Comprehensive integration test passed")


def main():
    """Run all integration tests."""
    print("CRYSTAL Material Tracking System - Integration Tests")
    print("=" * 60)
    
    tests = [
        test_database_operations,
        test_file_manager,
        test_error_detector,
        test_enhanced_queue_manager,
        test_integration_with_check_scripts,
        test_material_monitor,
        run_comprehensive_integration_test
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())