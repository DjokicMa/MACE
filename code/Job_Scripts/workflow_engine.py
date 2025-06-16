#!/usr/bin/env python3
"""
CRYSTAL Workflow Engine
-----------------------
Orchestrates the complete CRYSTAL calculation workflow while maintaining
material tracking and handling the real file naming conventions and
directory requirements of existing scripts.

Key Features:
- Maintains material ID consistency across complex file naming
- Creates isolated directories for each calculation step
- Handles the directory requirements of alldos.py and create_band_d3.py
- Integrates with CRYSTALOptToD12.py for geometry extraction
- Preserves all existing script behavior and file naming

Author: Based on implementation plan for material tracking system
"""

import os
import sys
import subprocess
import shutil
import tempfile
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

# Import our material database and other components
from material_database import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12


class WorkflowEngine:
    """
    Orchestrates CRYSTAL calculation workflows with material tracking.
    
    Handles the complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
    while maintaining material ID consistency and directory isolation for
    scripts like alldos.py and create_band_d3.py.
    """
    
    def __init__(self, db_path: str = "materials.db", base_work_dir: str = None):
        self.db = MaterialDatabase(db_path)
        self.base_work_dir = Path(base_work_dir or os.getcwd())
        self.script_paths = self.get_script_paths()
        self.lock = threading.RLock()
        
        # Create workflow working directories
        self.workflow_dir = self.base_work_dir / "workflow_staging"
        self.workflow_dir.mkdir(exist_ok=True)
        
    def get_script_paths(self) -> Dict[str, Path]:
        """Get paths to all the CRYSTAL workflow scripts."""
        base_path = Path(__file__).parent.parent
        
        return {
            'crystal_to_d12': base_path / "Crystal_To_CIF" / "CRYSTALOptToD12.py",
            'newcif_to_d12': base_path / "Crystal_To_CIF" / "NewCifToD12.py",
            'alldos': base_path / "Creation_Scripts" / "alldos.py",
            'create_band': base_path / "Creation_Scripts" / "create_band_d3.py",
            'd12creation': base_path / "Crystal_To_CIF" / "d12creation.py"
        }
        
    def extract_core_material_id_from_complex_filename(self, filename: str) -> str:
        """
        Extract core material ID from complex CRYSTAL filenames.
        
        Handles the real naming patterns from NewCifToD12.py:
        'material_CRYSTAL_OPTGEOM_symm_PBE-D3_POB-TZVP-REV2.d12'
        'material_CRYSTAL_SCFDIR_P1_HSE06_POB-TZVP-REV2.d12'
        
        Returns the core material identifier that remains consistent.
        """
        # Use existing function but enhance it for workflow tracking
        core_id = create_material_id_from_file(filename)
        
        # Additional cleanup for workflow consistency
        workflow_suffixes = ['_OPT', '_SP', '_BAND', '_DOSS', '_FREQ']
        for suffix in workflow_suffixes:
            if core_id.endswith(suffix):
                core_id = core_id[:-len(suffix)]
                break
                
        return core_id
        
    def get_material_id_from_any_file(self, file_path: Path) -> str:
        """Get material ID from any calculation file, handling complex naming."""
        # First try the database to see if this file is already tracked
        existing_calcs = self.db.get_all_calculations()
        
        file_str = str(file_path)
        for calc in existing_calcs:
            if (calc.get('input_file') == file_str or 
                calc.get('output_file') == file_str or
                file_path.stem in calc.get('input_file', '')):
                return calc['material_id']
        
        # If not found, extract from filename
        return self.extract_core_material_id_from_complex_filename(file_path.name)
        
    def ensure_material_exists(self, file_path: Path, source_type: str = "unknown") -> str:
        """Ensure material exists in database, create if needed."""
        material_id = self.get_material_id_from_any_file(file_path)
        
        # Check if material already exists
        existing_material = self.db.get_material(material_id)
        if existing_material:
            return material_id
            
        # Create new material
        if file_path.suffix == '.d12':
            formula = extract_formula_from_d12(str(file_path))
        else:
            formula = "Unknown"
            
        self.db.create_material(
            material_id=material_id,
            formula=formula,
            source_type=source_type,
            source_file=str(file_path),
            metadata={
                'workflow_created': True,
                'original_filename': file_path.name,
                'creation_timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"Created material record: {material_id}")
        return material_id
        
    def create_isolated_calculation_directory(self, material_id: str, calc_type: str, 
                                            source_files: List[Path]) -> Path:
        """
        Create isolated directory for calculation with only required files.
        
        This is critical for scripts like alldos.py and create_band_d3.py which
        expect to run in a clean directory with only the relevant files.
        """
        # Create unique directory name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calc_dir_name = f"{material_id}_{calc_type}_{timestamp}"
        calc_dir = self.workflow_dir / calc_dir_name
        calc_dir.mkdir(exist_ok=True)
        
        # Copy only the required files
        copied_files = []
        for source_file in source_files:
            if source_file.exists():
                dest_file = calc_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                copied_files.append(dest_file)
                print(f"Copied {source_file.name} to isolated directory")
            else:
                print(f"Warning: Source file not found: {source_file}")
                
        return calc_dir
        
    def run_script_in_isolated_directory(self, script_path: Path, work_dir: Path, 
                                       args: List[str] = None) -> Tuple[bool, str, str]:
        """
        Run a script in an isolated directory and capture output.
        
        Args:
            script_path: Path to the script to run
            work_dir: Directory to run the script in
            args: Additional command line arguments
            
        Returns:
            (success, stdout, stderr)
        """
        if not script_path.exists():
            return False, "", f"Script not found: {script_path}"
            
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
            
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Script execution timed out"
        except Exception as e:
            return False, "", f"Error running script: {e}"
            
    def generate_sp_from_opt(self, opt_calc_id: str) -> Optional[str]:
        """
        Generate SP calculation from completed OPT using CRYSTALOptToD12.py.
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            
        Returns:
            New SP calculation ID if successful, None otherwise
        """
        print(f"Generating SP calculation from OPT {opt_calc_id}")
        
        # Get OPT calculation details
        opt_calc = self.db.get_calculation(opt_calc_id)
        if not opt_calc or opt_calc['status'] != 'completed':
            print(f"OPT calculation {opt_calc_id} not completed")
            return None
            
        material_id = opt_calc['material_id']
        opt_output_file = Path(opt_calc['output_file'])
        opt_input_file = Path(opt_calc['input_file'])
        
        if not opt_output_file.exists():
            print(f"OPT output file not found: {opt_output_file}")
            return None
            
        # Create isolated directory for CRYSTALOptToD12.py
        work_dir = self.create_isolated_calculation_directory(
            material_id, "SP_generation", [opt_output_file, opt_input_file]
        )
        
        try:
            # Run CRYSTALOptToD12.py in isolated directory
            crystal_to_d12_script = self.script_paths['crystal_to_d12']
            
            # CRYSTALOptToD12.py arguments for batch processing
            args = [
                "--directory", str(work_dir),
                "--output-dir", str(work_dir),
                "--batch"  # Use batch mode to avoid interactive prompts
            ]
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args
            )
            
            if not success:
                print(f"CRYSTALOptToD12.py failed: {stderr}")
                return None
                
            # Find generated SP .d12 file
            sp_files = list(work_dir.glob("*SP*.d12"))
            if not sp_files:
                # Try other patterns
                sp_files = list(work_dir.glob("*_sp*.d12")) + list(work_dir.glob("*SCFDIR*.d12"))
                
            if not sp_files:
                print("No SP input file generated by CRYSTALOptToD12.py")
                return None
                
            sp_input_file = sp_files[0]
            
            # Move SP file to appropriate location
            sp_final_location = self.base_work_dir / "sp" / sp_input_file.name
            sp_final_location.parent.mkdir(exist_ok=True)
            shutil.move(sp_input_file, sp_final_location)
            
            # Create SP calculation record
            sp_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="SP",
                input_file=str(sp_final_location),
                settings={
                    'generated_from_opt': opt_calc_id,
                    'generation_method': 'CRYSTALOptToD12.py',
                    'workflow_step': True
                }
            )
            
            print(f"Generated SP calculation {sp_calc_id}: {sp_final_location.name}")
            return sp_calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def generate_doss_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate DOSS calculation from completed SP using alldos.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New DOSS calculation ID if successful, None otherwise
        """
        print(f"Generating DOSS calculation from SP {sp_calc_id}")
        
        # Get SP calculation details
        sp_calc = self.db.get_calculation(sp_calc_id)
        if not sp_calc or sp_calc['status'] != 'completed':
            print(f"SP calculation {sp_calc_id} not completed")
            return None
            
        material_id = sp_calc['material_id']
        sp_output_file = Path(sp_calc['output_file'])
        sp_input_file = Path(sp_calc['input_file'])
        
        # Find associated .f9 file
        sp_f9_file = sp_output_file.with_suffix('.f9')
        if not sp_f9_file.exists():
            # Try in work directory
            work_dir = Path(sp_calc.get('work_dir', sp_output_file.parent))
            sp_f9_file = work_dir / f"{sp_output_file.stem}.f9"
            
        if not sp_f9_file.exists():
            print(f"SP .f9 file not found for {sp_calc_id}")
            return None
            
        # Create isolated directory for alldos.py (CRITICAL for this script)
        required_files = [sp_input_file, sp_output_file, sp_f9_file]
        work_dir = self.create_isolated_calculation_directory(
            material_id, "DOSS_generation", required_files
        )
        
        try:
            # Run alldos.py in isolated directory
            alldos_script = self.script_paths['alldos']
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                alldos_script, work_dir
            )
            
            if not success:
                print(f"alldos.py failed: {stderr}")
                return None
                
            # Find generated DOSS .d3 file
            doss_files = list(work_dir.glob("*DOSS*.d3"))
            if not doss_files:
                print("No DOSS input file generated by alldos.py")
                return None
                
            doss_input_file = doss_files[0]
            
            # Also check for renamed .f9 file
            doss_f9_files = list(work_dir.glob("*DOSS*.f9"))
            
            # Move DOSS files to appropriate location
            doss_dir = self.base_work_dir / "doss"
            doss_dir.mkdir(exist_ok=True)
            
            doss_final_location = doss_dir / doss_input_file.name
            shutil.move(doss_input_file, doss_final_location)
            
            if doss_f9_files:
                doss_f9_final = doss_dir / doss_f9_files[0].name
                shutil.move(doss_f9_files[0], doss_f9_final)
            
            # Create DOSS calculation record
            doss_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="DOSS",
                input_file=str(doss_final_location),
                settings={
                    'generated_from_sp': sp_calc_id,
                    'generation_method': 'alldos.py',
                    'workflow_step': True,
                    'has_f9_file': bool(doss_f9_files)
                }
            )
            
            print(f"Generated DOSS calculation {doss_calc_id}: {doss_final_location.name}")
            return doss_calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def generate_band_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate BAND calculation from completed SP using create_band_d3.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New BAND calculation ID if successful, None otherwise
        """
        print(f"Generating BAND calculation from SP {sp_calc_id}")
        
        # Get SP calculation details
        sp_calc = self.db.get_calculation(sp_calc_id)
        if not sp_calc or sp_calc['status'] != 'completed':
            print(f"SP calculation {sp_calc_id} not completed")
            return None
            
        material_id = sp_calc['material_id']
        sp_output_file = Path(sp_calc['output_file'])
        sp_input_file = Path(sp_calc['input_file'])
        
        # Find associated .f9 file
        sp_f9_file = sp_output_file.with_suffix('.f9')
        if not sp_f9_file.exists():
            work_dir = Path(sp_calc.get('work_dir', sp_output_file.parent))
            sp_f9_file = work_dir / f"{sp_output_file.stem}.f9"
            
        if not sp_f9_file.exists():
            print(f"SP .f9 file not found for {sp_calc_id}")
            return None
            
        # Create isolated directory for create_band_d3.py (CRITICAL for this script)
        required_files = [sp_input_file, sp_output_file, sp_f9_file]
        work_dir = self.create_isolated_calculation_directory(
            material_id, "BAND_generation", required_files
        )
        
        try:
            # Run create_band_d3.py in isolated directory
            create_band_script = self.script_paths['create_band']
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                create_band_script, work_dir
            )
            
            if not success:
                print(f"create_band_d3.py failed: {stderr}")
                return None
                
            # Find generated BAND .d3 file
            band_files = list(work_dir.glob("*BAND*.d3")) + list(work_dir.glob("*band*.d3"))
            if not band_files:
                print("No BAND input file generated by create_band_d3.py")
                return None
                
            band_input_file = band_files[0]
            
            # Also check for renamed .f9 file
            band_f9_files = list(work_dir.glob("*BAND*.f9")) + list(work_dir.glob("*band*.f9"))
            
            # Move BAND files to appropriate location
            band_dir = self.base_work_dir / "band"
            band_dir.mkdir(exist_ok=True)
            
            band_final_location = band_dir / band_input_file.name
            shutil.move(band_input_file, band_final_location)
            
            if band_f9_files:
                band_f9_final = band_dir / band_f9_files[0].name
                shutil.move(band_f9_files[0], band_f9_final)
            
            # Create BAND calculation record
            band_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="BAND",
                input_file=str(band_final_location),
                settings={
                    'generated_from_sp': sp_calc_id,
                    'generation_method': 'create_band_d3.py',
                    'workflow_step': True,
                    'has_f9_file': bool(band_f9_files)
                }
            )
            
            print(f"Generated BAND calculation {band_calc_id}: {band_final_location.name}")
            return band_calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def execute_workflow_step(self, material_id: str, completed_calc_id: str) -> List[str]:
        """
        Execute the next workflow step(s) for a material.
        
        Args:
            material_id: Material identifier
            completed_calc_id: ID of just-completed calculation
            
        Returns:
            List of new calculation IDs created
        """
        new_calc_ids = []
        
        # Get the completed calculation
        completed_calc = self.db.get_calculation(completed_calc_id)
        if not completed_calc:
            print(f"Completed calculation not found: {completed_calc_id}")
            return new_calc_ids
            
        calc_type = completed_calc['calc_type']
        
        print(f"Executing workflow step after {calc_type} completion for {material_id}")
        
        # Determine next steps based on completed calculation type
        if calc_type == "OPT":
            # OPT completed -> generate SP
            sp_calc_id = self.generate_sp_from_opt(completed_calc_id)
            if sp_calc_id:
                new_calc_ids.append(sp_calc_id)
                
        elif calc_type == "SP":
            # SP completed -> generate both DOSS and BAND
            doss_calc_id = self.generate_doss_from_sp(completed_calc_id)
            if doss_calc_id:
                new_calc_ids.append(doss_calc_id)
                
            band_calc_id = self.generate_band_from_sp(completed_calc_id)
            if band_calc_id:
                new_calc_ids.append(band_calc_id)
                
        # Note: DOSS and BAND are typically terminal calculations in the workflow
        
        if new_calc_ids:
            print(f"Generated {len(new_calc_ids)} new calculations: {new_calc_ids}")
        else:
            print(f"No follow-up calculations generated for {calc_type}")
            
        return new_calc_ids
        
    def get_workflow_status(self, material_id: str) -> Dict:
        """Get comprehensive workflow status for a material."""
        material = self.db.get_material(material_id)
        if not material:
            return {"error": f"Material {material_id} not found"}
            
        # Get all calculations for this material
        calculations = self.db.get_calculations_by_status(material_id=material_id)
        
        # Organize by calculation type and status
        workflow_status = {
            "material_id": material_id,
            "formula": material.get("formula", "Unknown"),
            "total_calculations": len(calculations),
            "by_type": {},
            "completed_workflow_steps": [],
            "pending_workflow_steps": [],
            "failed_calculations": [],
            "workflow_complete": False
        }
        
        calc_types = ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT"]
        
        for calc_type in calc_types:
            type_calcs = [c for c in calculations if c['calc_type'] == calc_type]
            workflow_status["by_type"][calc_type] = {
                "total": len(type_calcs),
                "completed": len([c for c in type_calcs if c['status'] == 'completed']),
                "running": len([c for c in type_calcs if c['status'] == 'running']),
                "failed": len([c for c in type_calcs if c['status'] == 'failed']),
                "pending": len([c for c in type_calcs if c['status'] == 'pending'])
            }
            
            # Track completed and failed
            if workflow_status["by_type"][calc_type]["completed"] > 0:
                workflow_status["completed_workflow_steps"].append(calc_type)
                
            if workflow_status["by_type"][calc_type]["failed"] > 0:
                failed_calcs = [c for c in type_calcs if c['status'] == 'failed']
                workflow_status["failed_calculations"].extend([
                    {
                        "calc_id": c['calc_id'],
                        "calc_type": c['calc_type'],
                        "error_type": c.get('error_type', 'unknown'),
                        "error_message": c.get('error_message', '')
                    } for c in failed_calcs
                ])
        
        # Determine next steps
        if "OPT" in workflow_status["completed_workflow_steps"]:
            if "SP" not in workflow_status["completed_workflow_steps"]:
                workflow_status["pending_workflow_steps"].append("SP")
            elif "SP" in workflow_status["completed_workflow_steps"]:
                if "BAND" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("BAND")
                if "DOSS" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("DOSS")
        else:
            workflow_status["pending_workflow_steps"].append("OPT")
            
        # Check if basic workflow is complete (OPT -> SP -> BAND/DOSS)
        if all(step in workflow_status["completed_workflow_steps"] for step in ["OPT", "SP", "BAND", "DOSS"]):
            workflow_status["workflow_complete"] = True
            
        return workflow_status
        
    def process_completed_calculations(self) -> int:
        """
        Process all recently completed calculations and trigger workflow steps.
        
        Returns:
            Number of new workflow steps initiated
        """
        # Get recently completed calculations that haven't been processed
        completed_calcs = self.db.get_calculations_by_status('completed')
        
        new_steps = 0
        for calc in completed_calcs:
            # Check if this calculation has already triggered workflow steps
            settings = json.loads(calc.get('settings_json', '{}'))
            if settings.get('workflow_processed'):
                continue
                
            # Execute workflow step
            new_calc_ids = self.execute_workflow_step(calc['material_id'], calc['calc_id'])
            
            if new_calc_ids:
                new_steps += len(new_calc_ids)
                
                # Mark this calculation as workflow processed
                settings['workflow_processed'] = True
                settings['workflow_process_timestamp'] = datetime.now().isoformat()
                self.db.update_calculation_status(
                    calc['calc_id'], 
                    calc['status'],
                    settings_json=json.dumps(settings)
                )
                
        return new_steps


def main():
    """CLI interface for workflow engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL Workflow Engine")
    parser.add_argument("--action", choices=['status', 'process', 'workflow'], 
                       default='process', help="Action to perform")
    parser.add_argument("--material-id", help="Material ID for status checking")
    parser.add_argument("--db", default="materials.db", help="Path to materials database")
    parser.add_argument("--work-dir", default=".", help="Base working directory")
    
    args = parser.parse_args()
    
    # Initialize workflow engine
    workflow_engine = WorkflowEngine(args.db, args.work_dir)
    
    if args.action == 'status':
        if args.material_id:
            status = workflow_engine.get_workflow_status(args.material_id)
            print(f"\n=== Workflow Status for {args.material_id} ===")
            print(f"Formula: {status['formula']}")
            print(f"Total calculations: {status['total_calculations']}")
            print(f"Completed steps: {', '.join(status['completed_workflow_steps'])}")
            print(f"Pending steps: {', '.join(status['pending_workflow_steps'])}")
            print(f"Workflow complete: {status['workflow_complete']}")
            
            if status['failed_calculations']:
                print(f"\nFailed calculations:")
                for failed in status['failed_calculations']:
                    print(f"  - {failed['calc_id']} ({failed['calc_type']}): {failed['error_type']}")
        else:
            print("Please specify --material-id for status checking")
            
    elif args.action == 'process':
        print("Processing completed calculations...")
        new_steps = workflow_engine.process_completed_calculations()
        print(f"Initiated {new_steps} new workflow steps")
        
    elif args.action == 'workflow':
        # Show workflow status for all materials
        all_materials = workflow_engine.db.get_all_materials()
        print(f"\n=== Workflow Status Summary ({len(all_materials)} materials) ===")
        
        for material in all_materials:
            status = workflow_engine.get_workflow_status(material['material_id'])
            completed = len(status['completed_workflow_steps'])
            total_steps = 4  # OPT, SP, BAND, DOSS
            progress = f"{completed}/{total_steps}"
            
            print(f"{material['material_id']}: {progress} steps completed")
            if status['failed_calculations']:
                print(f"  ⚠️  {len(status['failed_calculations'])} failed calculations")


if __name__ == "__main__":
    main()