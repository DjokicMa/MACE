#!/usr/bin/env python3
"""
Workflow Progression Simulation Script
Tests the robustness of workflow plan tracking and step progression
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import time
from typing import List, Dict, Optional, Tuple

# Add the Job_Scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from workflow_engine import WorkflowEngine
from material_database import MaterialDatabase

class WorkflowSimulator:
    """Simulates workflow progression on HPCC with various success/failure scenarios"""
    
    def __init__(self, temp_dir: str = None):
        """Initialize the simulator with a temporary working directory"""
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="workflow_sim_"))
        
        # Create necessary subdirectories
        (self.temp_dir / "workflow_configs").mkdir(exist_ok=True)
        (self.temp_dir / "workflow_outputs").mkdir(exist_ok=True)
        
        # Initialize workflow engine and database
        self.db_path = self.temp_dir / "materials.db"
        self.db = MaterialDatabase(str(self.db_path))
        self.engine = WorkflowEngine(str(self.db_path), str(self.temp_dir))
        
        print(f"Simulation workspace: {self.temp_dir}")
        
    def create_workflow_plan(self, workflow_id: str, sequence: List[str]) -> Path:
        """Create a workflow plan JSON file"""
        plan = {
            "workflow_id": workflow_id,
            "workflow_sequence": sequence,
            "created": datetime.now().isoformat(),
            "execution_settings": {
                "max_concurrent_jobs": 200,
                "enable_material_tracking": True,
                "auto_progression": True
            }
        }
        
        plan_file = self.temp_dir / "workflow_configs" / f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
            
        return plan_file
        
    def create_mock_calculation_files(self, material_id: str, calc_type: str, 
                                    workflow_id: str, step_num: int) -> Dict[str, Path]:
        """Create mock calculation files for testing"""
        # Create step directory
        step_dir = self.temp_dir / "workflow_outputs" / workflow_id / f"step_{step_num:03d}_{calc_type}" / material_id
        step_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Create mock input file
        if calc_type in ["BAND", "DOSS"]:
            input_file = step_dir / f"{material_id}.d3"
            input_file.write_text(f"Mock {calc_type} input for {material_id}")
        else:
            input_file = step_dir / f"{material_id}.d12"
            # Create more realistic d12 file to avoid parser errors
            d12_content = """Diamond
CRYSTAL
0 0 0
227
3.56683
1
6 0.0 0.0 0.0
END
OPTGEOM
END
"""
            input_file.write_text(d12_content)
        files['input'] = input_file
        
        # Create mock output file with required information
        output_file = step_dir / f"{material_id}.out"
        output_content = f"""Mock {calc_type} output
SPACE GROUP : F d -3 m
NUMBER OF AO  200
TERMINATED
"""
        output_file.write_text(output_content)
        files['output'] = output_file
        
        # Create mock f9 file for calculations that need it
        if calc_type in ["OPT", "SP"]:
            f9_file = step_dir / f"{material_id}.f9"
            f9_file.write_text("Mock wavefunction")
            files['f9'] = f9_file
            
        return files
        
    def simulate_calculation_completion(self, material_id: str, calc_type: str, 
                                      workflow_id: str, step_num: int, 
                                      success: bool = True) -> str:
        """Simulate a calculation completing (successfully or with failure)"""
        print(f"\n{'='*60}")
        print(f"Simulating {calc_type} completion for {material_id} (step {step_num})")
        print(f"Success: {success}")
        
        # Create mock files
        files = self.create_mock_calculation_files(material_id, calc_type, workflow_id, step_num)
        
        # Create calculation record
        settings = {
            'workflow_id': workflow_id,
            'workflow_step': step_num
        }
        
        calc_id = self.db.create_calculation(
            material_id=material_id,
            calc_type=calc_type,
            input_file=str(files['input']),
            work_dir=str(files['input'].parent),
            settings=settings
        )
        
        # Update status and output file
        if success:
            self.db.update_calculation_status(calc_id, 'completed', output_file=str(files['output']))
            print(f"✓ {calc_type} completed successfully")
            
            # Trigger workflow progression
            print("Triggering workflow progression...")
            try:
                new_calc_ids = self.engine.execute_workflow_step(material_id, calc_id)
                if new_calc_ids:
                    print(f"Generated next calculations: {new_calc_ids}")
                    for new_id in new_calc_ids:
                        new_calc = self.db.get_calculation(new_id)
                        if new_calc:
                            print(f"  - {new_calc['calc_type']} ({new_calc['calc_id']})")
                else:
                    print("No follow-up calculations generated")
            except Exception as e:
                print(f"Error in workflow progression: {e}")
        else:
            self.db.update_calculation_status(calc_id, 'failed')
            print(f"✗ {calc_type} failed")
            
        return calc_id
        
    def run_workflow_scenario(self, scenario_name: str, sequence: List[str], 
                            failure_points: List[int] = None) -> None:
        """Run a complete workflow scenario with optional failure points"""
        print(f"\n{'#'*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"Sequence: {' → '.join(sequence)}")
        if failure_points:
            print(f"Failure points: steps {failure_points}")
        print(f"{'#'*70}")
        
        # Create workflow plan with unique timestamp
        time.sleep(0.1)  # Small delay to ensure unique timestamps
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        self.create_workflow_plan(workflow_id, sequence)
        
        # Create material with unique ID for each scenario
        material_id = f"test_material_{workflow_id.replace('workflow_', '')}"
        self.db.create_material(material_id, formula="C", space_group=227)
        
        # Track completed steps
        completed_steps = []
        
        # Execute workflow steps
        for i, calc_type in enumerate(sequence, 1):
            # Check if this step should fail
            should_fail = failure_points and i in failure_points
            
            # For the first step or after failures, create manually
            if i == 1 or (failure_points and (i-1) in failure_points):
                calc_id = self.simulate_calculation_completion(
                    material_id, calc_type, workflow_id, i, 
                    success=not should_fail
                )
                if not should_fail:
                    completed_steps.append(calc_type)
            else:
                # Check if this step was auto-generated
                calcs = self.db.get_calculations_by_material(material_id)
                pending_calc = None
                for calc in calcs:
                    if calc['calc_type'] == calc_type and calc['status'] == 'pending':
                        pending_calc = calc
                        break
                        
                if pending_calc:
                    print(f"\n{'='*60}")
                    print(f"Found auto-generated {calc_type} calculation")
                    # Simulate its completion
                    if not should_fail:
                        self.db.update_calculation_status(pending_calc['calc_id'], 'completed')
                        completed_steps.append(calc_type)
                        print(f"✓ {calc_type} completed successfully")
                        
                        # Trigger next steps
                        print("Triggering workflow progression...")
                        try:
                            new_calc_ids = self.engine.execute_workflow_step(material_id, pending_calc['calc_id'])
                            if new_calc_ids:
                                print(f"Generated next calculations: {new_calc_ids}")
                            else:
                                print("No follow-up calculations generated")
                        except Exception as e:
                            print(f"Error in workflow progression: {e}")
                    else:
                        self.db.update_calculation_status(pending_calc['calc_id'], 'failed')
                        print(f"✗ {calc_type} failed")
                else:
                    print(f"\n⚠ Warning: {calc_type} was not auto-generated!")
                    # Manual creation as fallback
                    calc_id = self.simulate_calculation_completion(
                        material_id, calc_type, workflow_id, i, 
                        success=not should_fail
                    )
                    if not should_fail:
                        completed_steps.append(calc_type)
                        
        # Summary
        print(f"\n{'='*60}")
        print("SCENARIO SUMMARY:")
        print(f"Planned sequence: {' → '.join(sequence)}")
        print(f"Completed steps: {' → '.join(completed_steps)}")
        print(f"Completion rate: {len(completed_steps)}/{len(sequence)} ({100*len(completed_steps)/len(sequence):.1f}%)")
        
        # Show final calculation status
        all_calcs = self.db.get_calculations_by_material(material_id)
        print(f"\nFinal calculation status:")
        # Sort by created_at since workflow_step might not be available in all calculations
        for calc in sorted(all_calcs, key=lambda x: x.get('created_at', '')):
            # Parse settings_json if it exists
            settings_json_str = calc.get('settings_json', '{}')
            if settings_json_str:
                try:
                    settings = json.loads(settings_json_str)
                    step = settings.get('workflow_step', '?')
                except (json.JSONDecodeError, TypeError):
                    step = '?'
            else:
                step = '?'
            print(f"  Step {step}: {calc['calc_type']} - {calc['status']}")
            
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.name.startswith("workflow_sim_"):
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up temporary directory: {self.temp_dir}")


def main():
    """Run workflow progression simulations"""
    
    # Create simulator
    sim = WorkflowSimulator()
    
    try:
        # Scenario 1: Simple workflow - all success
        sim.run_workflow_scenario(
            "Simple Workflow - All Success",
            ["OPT", "SP", "BAND", "DOSS"]
        )
        
        # Scenario 2: Simple workflow - BAND fails
        sim.run_workflow_scenario(
            "Simple Workflow - BAND Failure",
            ["OPT", "SP", "BAND", "DOSS"],
            failure_points=[3]  # BAND fails
        )
        
        # Scenario 3: Complex workflow - all success  
        sim.run_workflow_scenario(
            "Complex Workflow - All Success",
            ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2", "SP2"]
        )
        
        # Scenario 4: Complex workflow - multiple failures
        sim.run_workflow_scenario(
            "Complex Workflow - Multiple Failures",
            ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2", "SP2"],
            failure_points=[3, 6]  # BAND and OPT2 fail
        )
        
        # Scenario 5: Multi-OPT workflow - all success
        sim.run_workflow_scenario(
            "Multi-OPT Workflow - All Success",
            ["OPT", "OPT2", "OPT3", "SP", "BAND", "DOSS", "FREQ", "OPT4"]
        )
        
        # Scenario 6: Multi-OPT workflow - early SP failure
        sim.run_workflow_scenario(
            "Multi-OPT Workflow - SP Failure",
            ["OPT", "OPT2", "OPT3", "SP", "BAND", "DOSS", "FREQ", "OPT4"],
            failure_points=[4]  # SP fails
        )
        
    finally:
        # Cleanup
        sim.cleanup()


if __name__ == "__main__":
    main()