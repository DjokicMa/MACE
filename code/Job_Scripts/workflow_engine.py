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
        
    def get_workflow_sequence(self, workflow_id: str) -> Optional[List[str]]:
        """Get the planned workflow sequence for a workflow ID"""
        if not workflow_id:
            return None
            
        # Try to find the workflow plan file
        workflow_configs_dir = self.base_work_dir / "workflow_configs"
        
        # Look for plan files that match this workflow ID
        for plan_file in workflow_configs_dir.glob("workflow_plan_*.json"):
            try:
                import json
                with open(plan_file, 'r') as f:
                    plan = json.load(f)
                    
                if plan.get('workflow_id') == workflow_id:
                    return plan.get('workflow_sequence', [])
            except Exception as e:
                print(f"Error reading workflow plan {plan_file}: {e}")
                continue
                
        return None
    
    def get_script_paths(self) -> Dict[str, Path]:
        """Get paths to all the CRYSTAL workflow scripts."""
        current_dir = Path.cwd()
        base_path = Path(__file__).parent.parent
        
        # Check for local copies first (in current working directory)
        scripts = {
            'crystal_to_d12': "CRYSTALOptToD12.py",
            'newcif_to_d12': "NewCifToD12.py",
            'alldos': "alldos.py",
            'create_band': "create_band_d3.py",
            'd12creation': "d12creation.py"
        }
        
        script_paths = {}
        
        for key, script_name in scripts.items():
            # First check local working directory
            local_path = current_dir / script_name
            if local_path.exists():
                script_paths[key] = local_path
            else:
                # Fall back to repository location
                if key in ['crystal_to_d12', 'newcif_to_d12', 'd12creation']:
                    script_paths[key] = base_path / "Crystal_To_CIF" / script_name
                elif key in ['alldos', 'create_band']:
                    script_paths[key] = base_path / "Creation_Scripts" / script_name
        
        return script_paths
    
    def _create_slurm_script_for_calculation(self, calc_dir: Path, material_name: str, 
                                           calc_type: str, step_num: int, workflow_id: str) -> Path:
        """Create SLURM script for a calculation"""
        import os
        
        # Find appropriate template script
        base_dir = Path.cwd()
        if calc_type == "SP":
            template_script = base_dir / "workflow_scripts" / "submitcrystal23_sp_2.sh"
        elif calc_type == "FREQ":
            template_script = base_dir / "workflow_scripts" / "submitcrystal23_freq_5.sh"
        elif calc_type in ["BAND", "DOSS"]:
            template_script = base_dir / "workflow_scripts" / f"submit_prop_{calc_type.lower()}_{step_num + 2}.sh"
        else:
            template_script = base_dir / "workflow_scripts" / "submitcrystal23_opt_1.sh"
        
        if not template_script.exists():
            # Fall back to basic template
            if calc_type in ["OPT", "SP", "FREQ"]:
                template_script = base_dir / "submitcrystal23.sh"
            else:
                template_script = base_dir / "submit_prop.sh"
        
        if not template_script.exists():
            raise FileNotFoundError(f"No SLURM template found for {calc_type}")
        
        # Read template content
        with open(template_script, 'r') as f:
            template_content = f.read()
        
        # Create individual script for this material
        material_script_name = f"mat_{material_name}.sh"
        script_path = calc_dir / material_script_name
        
        # Customize script content
        customized_content = self._customize_slurm_script(
            template_content, material_name, calc_type, workflow_id, step_num
        )
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(customized_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        return script_path
    
    def _customize_slurm_script(self, template_content: str, material_name: str, 
                              calc_type: str, workflow_id: str, step_num: int) -> str:
        """Customize SLURM script template for specific calculation"""
        import re
        
        customized = template_content
        
        # Replace $1 placeholders with actual material name
        customized = customized.replace("$1", material_name)
        
        # Update scratch directory to be workflow-specific
        if "export scratch=" in customized:
            scratch_dir = f"$SCRATCH/{workflow_id}/step_{step_num:03d}_{calc_type}"
            customized = re.sub(
                r'export scratch=\$SCRATCH/[\w/]+',
                f'export scratch={scratch_dir}',
                customized
            )
        
        return customized
    
    def _submit_calculation_to_slurm(self, script_path: Path, work_dir: Path) -> Optional[str]:
        """Submit calculation to SLURM and return job ID"""
        import subprocess
        import re
        import os
        
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Submit job using sbatch
            result = subprocess.run(
                ['sbatch', str(script_path)], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                output = result.stdout.strip()
                job_id_match = re.search(r'Submitted batch job (\d+)', output)
                if job_id_match:
                    return job_id_match.group(1)
                else:
                    print(f"Could not extract job ID from: {output}")
                    return None
            else:
                print(f"Error submitting job: {result.stderr}")
                return None
                
        finally:
            os.chdir(original_cwd)
        
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
                                       args: List[str] = None, input_data: str = None) -> Tuple[bool, str, str]:
        """
        Run a script in an isolated directory and capture output.
        
        Args:
            script_path: Path to the script to run
            work_dir: Directory to run the script in
            args: Additional command line arguments
            input_data: Input data to provide to the script (for interactive scripts)
            
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
                input=input_data,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Script execution timed out"
        except Exception as e:
            return False, "", f"Error running script: {e}"
    
    def find_workflow_context(self, input_file_path: str) -> Optional[Tuple[str, str]]:
        """
        Find the workflow ID and step from a file path.
        
        Args:
            input_file_path: Path to an input file
            
        Returns:
            Tuple of (workflow_id, step_dir) if found, None otherwise
        """
        path = Path(input_file_path)
        
        # Look for workflow_outputs/workflow_YYYYMMDD_HHMMSS/step_XXX_TYPE pattern
        for parent in path.parents:
            if parent.name.startswith("workflow_") and len(parent.name) > 15:
                # Check if parent's parent is workflow_outputs
                if parent.parent.name == "workflow_outputs":
                    workflow_id = parent.name
                    # Find the step directory
                    for part in path.parts:
                        if part.startswith("step_") and "_" in part[5:]:
                            return workflow_id, part
        return None
    
    def get_workflow_output_base(self, opt_calc: Dict) -> Path:
        """
        Get the base workflow output directory for placing new calculations.
        
        Args:
            opt_calc: OPT calculation record
            
        Returns:
            Path to workflow outputs directory
        """
        opt_input_file = opt_calc.get('input_file', '')
        workflow_context = self.find_workflow_context(opt_input_file)
        
        if workflow_context:
            workflow_id, _ = workflow_context
            return self.base_work_dir / "workflow_outputs" / workflow_id
        else:
            # Fall back to creating a new workflow if context not found
            from datetime import datetime
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self.base_work_dir / "workflow_outputs" / workflow_id
    
    def clean_material_name(self, material_id: str) -> str:
        """
        Clean material name for use in directory names.
        
        Args:
            material_id: Raw material ID
            
        Returns:
            Cleaned material name safe for directories
        """
        # Replace problematic characters with underscores
        clean_name = material_id.replace(',', '_').replace('^', '_').replace(' ', '_')
        clean_name = clean_name.replace('/', '_').replace('\\', '_')
        return clean_name
            
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
            
            # Find the specific out and d12 files in the work directory
            out_files = list(work_dir.glob("*.out"))
            d12_files = list(work_dir.glob("*.d12"))
            
            if not out_files:
                print(f"No .out file found in {work_dir}")
                return None
                
            out_file = out_files[0]
            d12_file = d12_files[0] if d12_files else None
            
            # Use single file mode with automatic input responses
            args = [
                "--out-file", str(out_file),
                "--output-dir", str(work_dir)
            ]
            
            if d12_file:
                args.extend(["--d12-file", str(d12_file)])
            
            # Prepare input responses for non-interactive execution
            # 1. Keep settings? → y (yes, keep original DFT/PBE-D3 settings from d12)
            # 2. Calc type → 1 (SP)
            # 3. Symmetry choice → 1 (Write only unique atoms)
            # 4. Additional defaults for any other prompts
            input_responses = "y\n1\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args, input_data=input_responses
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
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(opt_calc)
            
            # Create material-specific directory for SP calculation
            material_clean = self.clean_material_name(material_id)
            # Avoid double mat_ prefix
            if material_clean.startswith("mat_"):
                dir_name = material_clean
            else:
                dir_name = f"mat_{material_clean}"
            sp_step_dir = workflow_base / "step_002_SP" / dir_name
            sp_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move SP file to material's directory
            sp_final_location = sp_step_dir / sp_input_file.name
            shutil.move(sp_input_file, sp_final_location)
            
            # Create SLURM script for SP calculation  
            # Use the actual D12 file stem for JOB variable to ensure consistency
            sp_job_name = sp_final_location.stem
            slurm_script_path = self._create_slurm_script_for_calculation(
                sp_step_dir, sp_job_name, "SP", 2, workflow_base.name
            )
            
            # Create SP calculation record
            sp_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="SP",
                input_file=str(sp_final_location),
                work_dir=str(sp_step_dir),
                settings={
                    'generated_from_opt': opt_calc_id,
                    'generation_method': 'CRYSTALOptToD12.py',
                    'workflow_step': True,
                    'slurm_script': str(slurm_script_path)
                }
            )
            
            # Submit SP calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, sp_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(sp_calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted SP calculation {sp_calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated SP calculation {sp_calc_id} but submission failed: {sp_final_location}")
            
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
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(sp_calc)
            
            # Create material-specific directory for DOSS calculation
            material_clean = self.clean_material_name(material_id)
            # Avoid double mat_ prefix
            if material_clean.startswith("mat_"):
                dir_name = material_clean
            else:
                dir_name = f"mat_{material_clean}"
            doss_step_dir = workflow_base / "step_004_DOSS" / dir_name
            doss_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move DOSS files to material's directory
            doss_final_location = doss_step_dir / doss_input_file.name
            shutil.move(doss_input_file, doss_final_location)
            
            if doss_f9_files:
                doss_f9_final = doss_step_dir / doss_f9_files[0].name
                shutil.move(doss_f9_files[0], doss_f9_final)
            
            # Create SLURM script for DOSS calculation
            # Use the actual D12 file stem for JOB variable to ensure consistency
            doss_job_name = doss_final_location.stem
            slurm_script_path = self._create_slurm_script_for_calculation(
                doss_step_dir, doss_job_name, "DOSS", 4, workflow_base.name
            )
            
            # Create DOSS calculation record
            doss_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="DOSS",
                input_file=str(doss_final_location),
                work_dir=str(doss_step_dir),
                settings={
                    'generated_from_sp': sp_calc_id,
                    'generation_method': 'alldos.py',
                    'workflow_step': True,
                    'has_f9_file': bool(doss_f9_files),
                    'slurm_script': str(slurm_script_path)
                }
            )
            
            # Submit DOSS calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, doss_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(doss_calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted DOSS calculation {doss_calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated DOSS calculation {doss_calc_id} but submission failed: {doss_final_location.name}")
            
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
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(sp_calc)
            
            # Create material-specific directory for BAND calculation
            material_clean = self.clean_material_name(material_id)
            # Avoid double mat_ prefix
            if material_clean.startswith("mat_"):
                dir_name = material_clean
            else:
                dir_name = f"mat_{material_clean}"
            band_step_dir = workflow_base / "step_003_BAND" / dir_name
            band_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move BAND files to material's directory
            band_final_location = band_step_dir / band_input_file.name
            shutil.move(band_input_file, band_final_location)
            
            if band_f9_files:
                band_f9_final = band_step_dir / band_f9_files[0].name
                shutil.move(band_f9_files[0], band_f9_final)
            
            # Create SLURM script for BAND calculation
            # Use the actual D12 file stem for JOB variable to ensure consistency
            band_job_name = band_final_location.stem
            slurm_script_path = self._create_slurm_script_for_calculation(
                band_step_dir, band_job_name, "BAND", 3, workflow_base.name
            )
            
            # Create BAND calculation record
            band_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="BAND",
                input_file=str(band_final_location),
                work_dir=str(band_step_dir),
                settings={
                    'generated_from_sp': sp_calc_id,
                    'generation_method': 'create_band_d3.py',
                    'workflow_step': True,
                    'has_f9_file': bool(band_f9_files),
                    'slurm_script': str(slurm_script_path)
                }
            )
            
            # Submit BAND calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, band_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(band_calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted BAND calculation {band_calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated BAND calculation {band_calc_id} but submission failed: {band_final_location.name}")
            
            return band_calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def generate_freq_from_opt(self, opt_calc_id: str) -> Optional[str]:
        """
        Generate FREQ calculation from completed OPT using CRYSTALOptToD12.py.
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            
        Returns:
            New FREQ calculation ID if successful, None otherwise
        """
        print(f"Generating FREQ calculation from OPT {opt_calc_id}")
        
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
            material_id, "FREQ_generation", [opt_output_file, opt_input_file]
        )
        
        try:
            # Run CRYSTALOptToD12.py in isolated directory
            crystal_to_d12_script = self.script_paths['crystal_to_d12']
            
            # Find the specific out and d12 files in the work directory
            out_files = list(work_dir.glob("*.out"))
            d12_files = list(work_dir.glob("*.d12"))
            
            if not out_files:
                print(f"No .out file found in {work_dir}")
                return None
                
            out_file = out_files[0]
            d12_file = d12_files[0] if d12_files else None
            
            # Use single file mode with automatic input responses for FREQ
            args = [
                "--out-file", str(out_file),
                "--output-dir", str(work_dir)
            ]
            
            if d12_file:
                args.extend(["--d12-file", str(d12_file)])
            
            # Prepare input responses for FREQ calculation
            # 1. Keep settings? → y (yes, keep original settings from d12)
            # 2. Calc type → 3 (FREQ)
            # 3. Symmetry choice → 1 (Write only unique atoms)
            # 4. Additional defaults for any other prompts
            input_responses = "y\n3\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"CRYSTALOptToD12.py failed for FREQ: {stderr}")
                return None
                
            # Find generated FREQ .d12 file
            freq_files = list(work_dir.glob("*FREQ*.d12"))
            if not freq_files:
                # Try other patterns
                freq_files = list(work_dir.glob("*_freq*.d12")) + list(work_dir.glob("*FREQCALC*.d12"))
                
            if not freq_files:
                print("No FREQ input file generated by CRYSTALOptToD12.py")
                return None
                
            freq_input_file = freq_files[0]
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(opt_calc)
            
            # Create material-specific directory for FREQ calculation
            material_clean = self.clean_material_name(material_id)
            # Avoid double mat_ prefix
            if material_clean.startswith("mat_"):
                dir_name = material_clean
            else:
                dir_name = f"mat_{material_clean}"
            freq_step_dir = workflow_base / "step_005_FREQ" / dir_name
            freq_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move FREQ file to material's directory
            freq_final_location = freq_step_dir / freq_input_file.name
            shutil.move(freq_input_file, freq_final_location)
            
            # Create SLURM script for FREQ calculation
            # Use the actual D12 file stem for JOB variable to ensure consistency
            freq_job_name = freq_final_location.stem
            slurm_script_path = self._create_slurm_script_for_calculation(
                freq_step_dir, freq_job_name, "FREQ", 5, workflow_base.name
            )
            
            # Create FREQ calculation record
            freq_calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type="FREQ",
                input_file=str(freq_final_location),
                work_dir=str(freq_step_dir),
                settings={
                    'generated_from_opt': opt_calc_id,
                    'generation_method': 'CRYSTALOptToD12.py',
                    'workflow_step': True,
                    'slurm_script': str(slurm_script_path)
                }
            )
            
            # Submit FREQ calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, freq_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(freq_calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted FREQ calculation {freq_calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated FREQ calculation {freq_calc_id} but submission failed: {freq_final_location}")
            
            return freq_calc_id
            
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
        
        # Determine next steps based on completed calculation type and workflow plan
        # Check if workflow metadata exists to determine planned sequence
        workflow_id = completed_calc.get('metadata', {}).get('workflow_id')
        planned_sequence = self.get_workflow_sequence(workflow_id) if workflow_id else None
        
        if calc_type == "OPT":
            # Generate next steps based on workflow plan or default behavior
            if planned_sequence:
                # Follow the planned workflow sequence
                if "SP" in planned_sequence:
                    sp_calc_id = self.generate_sp_from_opt(completed_calc_id)
                    if sp_calc_id:
                        new_calc_ids.append(sp_calc_id)
                
                if "FREQ" in planned_sequence:
                    freq_calc_id = self.generate_freq_from_opt(completed_calc_id)
                    if freq_calc_id:
                        new_calc_ids.append(freq_calc_id)
            else:
                # Default behavior: generate SP only
                sp_calc_id = self.generate_sp_from_opt(completed_calc_id)
                if sp_calc_id:
                    new_calc_ids.append(sp_calc_id)
                
        elif calc_type == "SP":
            # Generate next steps based on workflow plan or default behavior
            if planned_sequence:
                # Follow the planned workflow sequence
                if "DOSS" in planned_sequence:
                    doss_calc_id = self.generate_doss_from_sp(completed_calc_id)
                    if doss_calc_id:
                        new_calc_ids.append(doss_calc_id)
                        
                if "BAND" in planned_sequence:
                    band_calc_id = self.generate_band_from_sp(completed_calc_id)
                    if band_calc_id:
                        new_calc_ids.append(band_calc_id)
            else:
                # Default behavior: generate both DOSS and BAND
                doss_calc_id = self.generate_doss_from_sp(completed_calc_id)
                if doss_calc_id:
                    new_calc_ids.append(doss_calc_id)
                    
                band_calc_id = self.generate_band_from_sp(completed_calc_id)
                if band_calc_id:
                    new_calc_ids.append(band_calc_id)
                
        # Note: DOSS, BAND, and FREQ are typically terminal calculations in the workflow
        
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
            if "FREQ" not in workflow_status["completed_workflow_steps"]:
                workflow_status["pending_workflow_steps"].append("FREQ")
            if "SP" in workflow_status["completed_workflow_steps"]:
                if "BAND" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("BAND")
                if "DOSS" not in workflow_status["completed_workflow_steps"]:
                    workflow_status["pending_workflow_steps"].append("DOSS")
        else:
            workflow_status["pending_workflow_steps"].append("OPT")
            
        # Check if complete workflow is finished (OPT -> SP/FREQ -> BAND/DOSS)
        if all(step in workflow_status["completed_workflow_steps"] for step in ["OPT", "SP", "BAND", "DOSS", "FREQ"]):
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
                self.db.update_calculation_settings(calc['calc_id'], settings)
                
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