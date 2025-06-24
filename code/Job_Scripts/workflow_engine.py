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
    
    def __init__(self, db_path: str = "materials.db", base_work_dir: str = None, auto_submit: bool = True):
        self.db = MaterialDatabase(db_path)
        self.base_work_dir = Path(base_work_dir or os.getcwd())
        self.script_paths = self.get_script_paths()
        self.lock = threading.RLock()
        self.auto_submit = auto_submit  # Enable automatic submission by default
        
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
    def get_workflow_id_from_calculation(self, calc_dir: Path) -> Optional[str]:
        """Get workflow ID from calculation directory metadata"""
        metadata_file = calc_dir / '.workflow_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('workflow_id')
            except Exception as e:
                print(f"Error reading workflow metadata: {e}")
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
        workflow_scripts_dir = base_dir / "workflow_scripts"
        
        # Look for templates with flexible numbering
        template_script = None
        base_type, type_num = self._parse_calc_type(calc_type)
        
        if base_type == "SP":
            # Try numbered templates first (for SP2, SP3, etc.)
            if type_num > 1:
                # Try specific numbered templates
                numbered_patterns = [
                    f"submitcrystal23_sp{type_num}_*.sh",
                    f"submitcrystal23_sp{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general SP templates
            if not template_script:
                for name in ["submitcrystal23_sp_*.sh", "submitcrystal23_sp.sh"]:
                    candidates = list(workflow_scripts_dir.glob(name))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "FREQ":
            # Try numbered templates first (for FREQ2, FREQ3, etc.)
            if type_num > 1:
                # Try specific numbered templates
                numbered_patterns = [
                    f"submitcrystal23_freq{type_num}_*.sh",
                    f"submitcrystal23_freq{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general FREQ templates
            if not template_script:
                for pattern in ["submitcrystal23_freq_*.sh", "submitcrystal23_freq.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "BAND":
            # Try numbered templates first (for BAND2, BAND3, etc.)
            if type_num > 1:
                numbered_patterns = [
                    f"submit_prop_band{type_num}_*.sh",
                    f"submit_prop_band{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general BAND templates
            if not template_script:
                for pattern in ["submit_prop_band_*.sh", "submit_prop_band.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        elif base_type == "DOSS":
            # Try numbered templates first (for DOSS2, DOSS3, etc.)
            if type_num > 1:
                numbered_patterns = [
                    f"submit_prop_doss{type_num}_*.sh",
                    f"submit_prop_doss{type_num}.sh"
                ]
                for pattern in numbered_patterns:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
            
            # If no specific template found, try general DOSS templates
            if not template_script:
                for pattern in ["submit_prop_doss_*.sh", "submit_prop_doss.sh"]:
                    candidates = list(workflow_scripts_dir.glob(pattern))
                    if candidates:
                        template_script = candidates[0]
                        break
        else:
            # For other calculation types (OPT, OPTn, etc.)
            if base_type == "OPT":
                # Try numbered templates first (for OPT2, OPT3, etc.)
                if type_num > 1:
                    # Try specific numbered templates
                    numbered_patterns = [
                        f"submitcrystal23_opt{type_num}_*.sh",
                        f"submitcrystal23_opt{type_num}.sh"
                    ]
                    for pattern in numbered_patterns:
                        candidates = list(workflow_scripts_dir.glob(pattern))
                        if candidates:
                            template_script = candidates[0]
                            break
                    
                # If no specific template found, try general OPT templates
                if not template_script:
                    for pattern in ["submitcrystal23_opt_*.sh", "submitcrystal23_opt.sh", "submitcrystal23.sh"]:
                        candidates = list(workflow_scripts_dir.glob(pattern))
                        if candidates:
                            template_script = candidates[0]
                            break
        
        # If still no template found, use the first available one that matches the pattern
        if not template_script:
            if calc_type in ["OPT", "SP", "FREQ"]:
                pattern = f"submitcrystal23_*{calc_type.lower()}*.sh"
            else:
                pattern = f"submit_prop_*{calc_type.lower()}*.sh"
            
            templates = list(workflow_scripts_dir.glob(pattern))
            if templates:
                template_script = templates[0]
        
        # Debug: Print template path being used
        print(f"  Using template: {template_script}")
        print(f"  Template exists: {template_script.exists()}")
        
        if not template_script.exists():
            # Fall back to basic template
            if calc_type in ["OPT", "SP", "FREQ"]:
                template_script = base_dir / "submitcrystal23.sh"
                print(f"  Fallback template: {template_script}")
                print(f"  Fallback exists: {template_script.exists()}")
            else:
                template_script = base_dir / "submit_prop.sh"
        
        if not template_script.exists():
            raise FileNotFoundError(f"No SLURM template found for {calc_type}")
        
        # Read template content
        with open(template_script, 'r') as f:
            template_content = f.read()
            
        # Check if this is an old-style script generator (should be avoided)
        if 'echo \'#!/bin/bash --login\' >' in template_content:
            print(f"  WARNING: Template {template_script} appears to be a script generator, not a direct SLURM script")
            print(f"  This will cause submission issues. Consider using a direct SLURM template.")
        
        print(f"  Template content starts with: {template_content[:100]}...")
        
        # Create individual script for this material  
        # Use clean naming without mat_ prefix throughout
        material_script_name = f"{material_name}.sh"
        script_path = calc_dir / material_script_name
        
        # Customize script content using material name for file references
        customized_content = self._customize_slurm_script(
            template_content, material_name, calc_type, workflow_id, step_num
        )
        
        # Write script
        try:
            with open(script_path, 'w') as f:
                f.write(customized_content)
            
            # Make executable
            script_path.chmod(0o755)
            
            print(f"  Successfully created script: {script_path}")
            print(f"  Script exists: {script_path.exists()}")
            
            return script_path
        except Exception as e:
            print(f"  Error creating script {script_path}: {e}")
            raise
    
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
        
        # Enhance queue manager path resolution to handle workflow directory structure
        if "# ADDED: Auto-submit new jobs when this one completes" in customized:
            # Replace the existing queue manager logic with enhanced path resolution
            queue_manager_logic = '''
# ADDED: Auto-submit new jobs when this one completes
# Enhanced path resolution for workflow directory structure
QUEUE_MANAGER=""
if [ -f $DIR/enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
elif [ -f $DIR/../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../enhanced_queue_manager.py"
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
elif [ -f $DIR/../../../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../../../enhanced_queue_manager.py"
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    cd $(dirname "$QUEUE_MANAGER")
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
else
    echo "Warning: enhanced_queue_manager.py not found - workflow progression may not continue automatically"
fi'''
            
            # Replace the entire queue manager section
            import re
            customized = re.sub(
                r'# ADDED: Auto-submit new jobs when this one completes.*?fi',
                queue_manager_logic.strip(),
                customized,
                flags=re.DOTALL
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
            
            # Extract job name from the script path (using clean naming throughout)
            job_name = script_path.stem
            
            # Use the script file name relative to work_dir since we changed to work_dir
            script_filename = script_path.name
            
            # Check if the script contains script generation logic
            print(f"  Reading script file: {script_filename} (from work_dir: {work_dir})")
            with open(script_filename, 'r') as f:
                script_content = f.read()
            
            if 'echo \'#!/bin/bash --login\' >' in script_content or 'echo "#SBATCH' in script_content:
                # This is a script generator template - run locally to generate actual script
                print(f"  Running script generator locally: {script_filename} with job name: {job_name}")
                result = subprocess.run(
                    ['bash', script_filename, job_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract job ID from sbatch output (the template runs sbatch at the end)
                    output = result.stdout.strip()
                    job_id_match = re.search(r'Submitted batch job (\d+)', output)
                    if job_id_match:
                        return job_id_match.group(1)
                    
                    # Maybe the template just generated the script but didn't submit it
                    # Look for generated script and submit it manually
                    generated_script = work_dir / f"{full_job_name}.sh"
                    if generated_script.exists():
                        print(f"  Found generated script: {generated_script.name}")
                        result = subprocess.run(
                            ['sbatch', generated_script.name],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
                            if job_id_match:
                                return job_id_match.group(1)
                else:
                    print(f"Error running script generator: {result.stderr}")
                    
            else:
                # This is a regular SLURM script - submit directly
                print(f"  Submitting SLURM script directly: {script_path.name}")
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
        # Use the smart suffix removal function - no additional cleanup needed
        return create_material_id_from_file(filename)
        
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
    
    def extract_core_material_name(self, material_id: str) -> str:
        """Extract the core material name with proper handling of numbered materials"""
        # Handle both full filenames and stems
        from pathlib import Path
        if material_id.endswith('.d12') or material_id.endswith('.d3'):
            name = Path(material_id).stem
        else:
            name = material_id
        
        # Strategy: Remove known calculation suffixes from the end
        # IMPORTANT: Check numbered suffixes BEFORE base suffixes to avoid 
        # stripping numbers that are part of the material name (e.g., test2_sp)
        
        # Generate suffixes dynamically for any number (up to 99 seems reasonable)
        calc_suffixes = []
        base_types = ['opt', 'sp', 'freq', 'band', 'doss']
        
        # Add numbered suffixes in reverse order (higher numbers first)
        for num in range(99, 1, -1):
            for base in base_types:
                calc_suffixes.append(f'_{base}{num}')
        
        # Then add base suffixes
        for base in base_types:
            calc_suffixes.append(f'_{base}')
        
        # Remove calc suffix if present
        clean_name = name
        for suffix in calc_suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # If name still contains technical keywords, extract the core
        if any(keyword in clean_name.upper() for keyword in ['BULK', 'OPTGEOM', 'CRYSTAL', 'PBE', 'B3LYP']):
            # Use the original logic for complex names
            parts = clean_name.split('_')
            core_parts = []
            
            for part in parts:
                # Stop at technical keywords
                if part.upper() in ['BULK', 'OPTGEOM', 'CRYSTAL', 'SLAB', 'POLYMER', 
                                   'MOLECULE', 'SYMM', 'TZ', 'DZ', 'SZ', 'PBE', 'B3LYP', 
                                   'HSE06', 'PBE0', 'SCAN', 'BLYP', 'BP86']:
                    break
                elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():
                    break
                elif 'D3' in part.upper():
                    break
                else:
                    core_parts.append(part)
                    
            if core_parts:
                clean_name = '_'.join(core_parts)
        
        return clean_name
        # Only replace truly problematic characters for filesystem compatibility
        # Preserve ^ and , as they are commonly used in chemical notation
        clean_name = clean_name.replace(' ', '_')  # Spaces to underscores
        clean_name = clean_name.replace('/', '_').replace('\\', '_')  # Path separators
        # Keep other characters like ^, ,, ., - as they are common in chemical names
        
        return clean_name
    
    def clean_material_name(self, material_id: str) -> str:
        """
        Clean material name for use in directory names - wrapper for backward compatibility
        """
        return self.extract_core_material_name(material_id)
        
    def get_next_calc_suffix(self, core_name: str, calc_type: str, workflow_base: Path) -> str:
        """
        Get the next available suffix for a calculation type (e.g., _sp, _sp2, _sp3)
        
        Args:
            core_name: Core material name (e.g., "1_dia")
            calc_type: Calculation type ("SP", "OPT", "BAND", "DOSS", "FREQ")
            workflow_base: Base workflow directory to check for existing calculations
            
        Returns:
            Clean suffix like "_sp", "_sp2", "_opt", "_opt2", etc.
        """
        type_suffix = calc_type.lower()
        
        # Check what calculation directories already exist
        existing_dirs = []
        for step_dir in workflow_base.glob(f"step_*_{calc_type}"):
            if step_dir.is_dir():
                for material_dir in step_dir.glob(f"{core_name}*"):
                    existing_dirs.append(material_dir.name)
        
        # Determine the next number
        if not existing_dirs:
            # First calculation of this type
            return f"_{type_suffix}"
        
        # Find highest existing number
        max_num = 0
        for dir_name in existing_dirs:
            # Look for pattern like 1_dia_sp2 or 1_dia_sp  
            if f"_{type_suffix}" in dir_name:
                parts = dir_name.split(f"_{type_suffix}")
                if len(parts) > 1 and parts[1]:
                    # Has a number after the type
                    try:
                        num = int(parts[1])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
                else:
                    # No number means it's the first one
                    max_num = max(max_num, 1)
        
        # Return next available suffix
        next_num = max_num + 1
        if next_num == 1:
            return f"_{type_suffix}"
        else:
            return f"_{type_suffix}{next_num}"
            
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
            
            # Get core material name and determine SP suffix
            core_name = self.extract_core_material_name(material_id)
            sp_suffix = self.get_next_calc_suffix(core_name, "SP", workflow_base)
            
            # Use core name directly (already clean)
            clean_job_name = core_name
            
            # Create material-specific directory for SP calculation
            dir_name = f"{core_name}{sp_suffix}"
            sp_step_dir = workflow_base / "step_002_SP" / dir_name
            sp_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move SP file to material's directory with consistent naming
            clean_sp_name = f"{core_name}{sp_suffix}.d12"
            sp_final_location = sp_step_dir / clean_sp_name
            shutil.move(sp_input_file, sp_final_location)
            
            # Create SLURM script for SP calculation  
            # Use the clean material name for the job
            sp_job_name = f"{clean_job_name}{sp_suffix}"
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
        return self.generate_property_calculation(sp_calc_id, "DOSS")
            
    def generate_band_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate BAND calculation from completed SP using create_band_d3.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New BAND calculation ID if successful, None otherwise
        """
        return self.generate_property_calculation(sp_calc_id, "BAND")
        
    def generate_property_calculation(self, source_calc_id: str, target_calc_type: str) -> Optional[str]:
        """
        Generate property calculation (BAND, DOSS) from a source calculation with wavefunction.
        
        For BAND/DOSS, this will use the most recent SP or OPT calculation with a wavefunction.
        
        Args:
            source_calc_id: ID of source calculation (used as hint, may find more recent)
            target_calc_type: Target calculation type (BAND, DOSS, BAND2, DOSS2, etc.)
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {target_calc_type} calculation")
        
        # Parse target calculation type to get base type and number
        base_type, calc_num = self._parse_calc_type(target_calc_type)
        
        # Get source calculation details
        source_calc = self.db.get_calculation(source_calc_id)
        if not source_calc:
            print(f"Source calculation {source_calc_id} not found")
            return None
            
        material_id = source_calc['material_id']
        
        # Find the most recent calculation with a wavefunction (SP or OPT)
        wavefunction_calc_id = self._find_most_recent_wavefunction_calc(material_id)
        if not wavefunction_calc_id:
            print(f"No completed calculation with wavefunction found for material {material_id}")
            return None
            
        # Get the wavefunction calculation details
        wf_calc = self.db.get_calculation(wavefunction_calc_id)
        if not wf_calc or wf_calc['status'] != 'completed':
            print(f"Wavefunction calculation {wavefunction_calc_id} not completed")
            return None
            
        wf_output_file = Path(wf_calc['output_file'])
        wf_input_file = Path(wf_calc['input_file'])
        
        # Find associated .f9 file
        wf_f9_file = wf_output_file.with_suffix('.f9')
        if not wf_f9_file.exists():
            work_dir = Path(wf_calc.get('work_dir', wf_output_file.parent))
            wf_f9_file = work_dir / f"{wf_output_file.stem}.f9"
            
        if not wf_f9_file.exists():
            print(f"Wavefunction .f9 file not found for {wavefunction_calc_id}")
            return None
            
        # Determine which script to use based on target type
        if base_type == "BAND":
            script_key = 'create_band'
            generation_label = f"{target_calc_type}_generation"
            file_pattern = "*BAND*.d3"
            file_pattern2 = "*band*.d3"
        elif base_type == "DOSS":
            script_key = 'alldos'
            generation_label = f"{target_calc_type}_generation"
            file_pattern = "*DOSS*.d3"
            file_pattern2 = "*doss*.d3"
        else:
            print(f"Unsupported property calculation type: {target_calc_type}")
            return None
            
        # Create isolated directory for the script (CRITICAL for these scripts)
        required_files = [wf_input_file, wf_output_file, wf_f9_file]
        work_dir = self.create_isolated_calculation_directory(
            material_id, generation_label, required_files
        )
        
        try:
            # Run the appropriate script in isolated directory
            script = self.script_paths[script_key]
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                script, work_dir
            )
            
            if not success:
                print(f"{script_key} script failed: {stderr}")
                return None
                
            # Find generated .d3 file
            output_files = list(work_dir.glob(file_pattern)) + list(work_dir.glob(file_pattern2))
            if not output_files:
                print(f"No {base_type} input file generated by {script_key}")
                return None
                
            output_input_file = output_files[0]
            
            # Also check for renamed .f9 file
            f9_files = list(work_dir.glob(f"*{base_type}*.f9")) + list(work_dir.glob(f"*{base_type.lower()}*.f9"))
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(source_calc)
            
            # Get core material name and determine suffix
            core_name = self.extract_core_material_name(material_id)
            calc_suffix = self.get_next_calc_suffix(core_name, base_type, workflow_base)
            
            # Use core name directly (already clean)
            clean_job_name = core_name
            
            # Determine step number based on base type
            step_numbers = {"BAND": 3, "DOSS": 4}
            step_num = step_numbers.get(base_type, 3)
            
            # Create material-specific directory for calculation
            dir_name = f"{core_name}{calc_suffix}"
            calc_step_dir = workflow_base / f"step_{step_num:03d}_{base_type}" / dir_name
            calc_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files to material's directory with consistent naming
            clean_name = f"{core_name}{calc_suffix}.d3"
            final_location = calc_step_dir / clean_name
            shutil.move(output_input_file, final_location)
            
            if f9_files:
                clean_f9_name = f"{core_name}{calc_suffix}.f9"
                f9_final = calc_step_dir / clean_f9_name
                shutil.move(f9_files[0], f9_final)
            
            # Create SLURM script for calculation
            # Use the clean material name for the job
            job_name = f"{clean_job_name}{calc_suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                calc_step_dir, job_name, base_type, step_num, workflow_base.name
            )
            
            # Create calculation record
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=target_calc_type,
                input_file=str(final_location),
                work_dir=str(calc_step_dir),
                settings={
                    'generated_from_wavefunction': wavefunction_calc_id,
                    'generation_method': script_key,
                    'workflow_step': True,
                    'has_f9_file': bool(f9_files),
                    'slurm_script': str(slurm_script_path)
                }
            )
            
            # Submit calculation
            slurm_job_id = self._submit_calculation_to_slurm(slurm_script_path, calc_step_dir)
            if slurm_job_id:
                self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=slurm_job_id)
                print(f"Generated and submitted {target_calc_type} calculation {calc_id}: Job {slurm_job_id}")
            else:
                print(f"Generated {target_calc_type} calculation {calc_id} but submission failed: {final_location.name}")
            
            return calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def generate_freq_from_sp(self, sp_calc_id: str) -> Optional[str]:
        """
        Generate FREQ calculation from completed SP using CRYSTALOptToD12.py.
        
        Args:
            sp_calc_id: ID of completed SP calculation
            
        Returns:
            New FREQ calculation ID if successful, None otherwise
        """
        return self.generate_numbered_calculation(sp_calc_id, "FREQ")
        
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
            
            # Get core material name and determine FREQ suffix
            core_name = self.extract_core_material_name(material_id)
            freq_suffix = self.get_next_calc_suffix(core_name, "FREQ", workflow_base)
            
            # Use core name directly (already clean)
            clean_job_name = core_name
            
            # Create material-specific directory for FREQ calculation
            dir_name = f"{core_name}{freq_suffix}"
            freq_step_dir = workflow_base / "step_005_FREQ" / dir_name
            freq_step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move FREQ file to material's directory with consistent naming
            clean_freq_name = f"{core_name}{freq_suffix}.d12"
            freq_final_location = freq_step_dir / clean_freq_name
            shutil.move(freq_input_file, freq_final_location)
            
            # Create SLURM script for FREQ calculation
            # Use the clean material name for the job
            freq_job_name = f"{clean_job_name}{freq_suffix}"
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
            
    def _check_and_trigger_pending_calculations(self, material_id: str, planned_sequence: List[str]) -> List[str]:
        """
        Check for any calculations in the planned sequence that should have been triggered
        but haven't been yet. This handles cases where workflow steps might have been missed.
        
        Returns:
            List of new calculation IDs created
        """
        new_calc_ids = []
        
        if not planned_sequence:
            return new_calc_ids
            
        # Get all calculations for this material
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Build a map of completed calculations by type
        completed_by_type = {}
        for calc in all_calcs:
            if calc['status'] == 'completed':
                calc_type = calc['calc_type']
                if calc_type not in completed_by_type:
                    completed_by_type[calc_type] = []
                completed_by_type[calc_type].append(calc)
        
        # Check if any pending calculations can now be started
        for planned_type in planned_sequence:
            base_type, type_num = self._parse_calc_type(planned_type)
            
            # Skip if already completed or in progress
            existing = [c for c in all_calcs if c['calc_type'] == planned_type]
            if existing:
                continue
                
            # Check dependencies
            can_start = False
            source_calc_id = None
            
            if base_type == "SP":
                # SP calculations depend on their previous step in the workflow sequence
                # Find what step comes before this SP in the planned sequence
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    can_start = True
                    source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                elif not prev_step:
                    # SP is the first calculation - needs CIF source
                    can_start = True
                    source_calc_id = 'CIF'  # Special marker for CIF generation
                    
            elif base_type == "FREQ":
                # FREQ calculations need an optimized geometry from an OPT calculation
                # Wait for the dependency step to complete, but use the highest numbered OPT
                # that has been COMPLETED up to this point (not future OPTs in the sequence)
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    # Find the highest numbered OPT calculation that's been completed
                    # This correctly handles cases like: OPT → OPT2 → SP → OPT3 → FREQ → OPT4
                    # where FREQ uses OPT3 (not OPT4 which hasn't run yet)
                    opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                    if opt_source:
                        can_start = True
                        source_calc_id = opt_source  # Use highest completed OPT for FREQ generation
                    
            elif base_type in ["BAND", "DOSS"]:
                # BAND/DOSS calculations depend on their previous step in the workflow sequence
                # They typically need a wavefunction from SP or OPT
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    can_start = True
                    source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                    
            elif base_type == "OPT":
                # OPT calculations need appropriate source geometry
                prev_step = self._find_dependency_in_sequence(planned_type, planned_sequence)
                if prev_step and prev_step in completed_by_type:
                    # Check if previous step can provide geometry
                    prev_base, _ = self._parse_calc_type(prev_step)
                    
                    if prev_base in ['OPT', 'SP']:  # These can provide geometry
                        can_start = True
                        source_calc_id = completed_by_type[prev_step][-1]['calc_id']
                    else:
                        # Previous step can't provide geometry (e.g., FREQ, BAND, DOSS)
                        # Find the highest numbered OPT completed so far
                        opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        if opt_source:
                            can_start = True
                            source_calc_id = opt_source  # Use highest completed OPT
                        elif type_num == 1:
                            # First OPT with no prior OPT - need CIF source
                            can_start = True
                            source_calc_id = 'CIF'  # Special marker for CIF generation
            
            # Trigger the calculation if dependencies are met
            if can_start and source_calc_id:
                print(f"Triggering pending {planned_type} calculation...")
                
                if base_type == "SP":
                    if source_calc_id == 'CIF':
                        # Generate from CIF
                        # Find material_id from context or use planned_type to derive it
                        material_id = None
                        for calcs in completed_by_type.values():
                            if calcs:
                                material_id = calcs[0]['material_id']
                                break
                        # If no completed calcs, we need to get material_id from somewhere else
                        # This would typically come from the workflow context
                        if material_id:
                            calc_id = self.generate_calculation_from_cif(material_id, planned_type)
                        else:
                            print(f"Cannot determine material_id for CIF generation")
                            calc_id = None
                    else:
                        calc_id = self.generate_numbered_calculation(source_calc_id, planned_type)
                elif base_type == "FREQ":
                    # FREQ always uses generate_freq_from_opt with an OPT calculation
                    # source_calc_id should already be from an OPT due to fixed dependency logic
                    calc_id = self.generate_freq_from_opt(source_calc_id)
                elif base_type in ["BAND", "DOSS"]:
                    calc_id = self.generate_property_calculation(source_calc_id, planned_type)
                elif base_type == "OPT":
                    if source_calc_id == 'CIF':
                        # Generate from CIF
                        # Find material_id from any completed calculation
                        material_id = None
                        for calcs in completed_by_type.values():
                            if calcs:
                                material_id = calcs[0]['material_id']
                                break
                        if material_id:
                            calc_id = self.generate_calculation_from_cif(material_id, planned_type)
                        else:
                            print(f"Cannot determine material_id for CIF generation")
                            calc_id = None
                    else:
                        calc_id = self.generate_numbered_calculation(source_calc_id, planned_type)
                else:
                    calc_id = None
                    
                if calc_id:
                    new_calc_ids.append(calc_id)
                    
        return new_calc_ids

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
        # Look for workflow_id in settings (where it's stored) or metadata
        settings = json.loads(completed_calc.get('settings_json', '{}'))
        workflow_id = settings.get('workflow_id') or completed_calc.get('metadata', {}).get('workflow_id')
        
        print(f"DEBUG: Extracted workflow_id: {workflow_id}")
        print(f"DEBUG: Settings: {settings}")
        
        planned_sequence = self.get_workflow_sequence(workflow_id) if workflow_id else None
        print(f"DEBUG: Planned sequence: {planned_sequence}")
        
        # Parse current calculation type to handle numbered types
        base_type, type_num = self._parse_calc_type(calc_type)
        
        if base_type == "OPT":
            # Generate next steps based on workflow plan or default behavior
            if planned_sequence:
                # Find current position in sequence and get next step
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                next_steps = self._get_next_steps_from_sequence(current_index, planned_sequence, calc_type)
                
                for next_calc_type in next_steps:
                    next_base_type, next_num = self._parse_calc_type(next_calc_type)
                    
                    if next_base_type == "OPT":
                        # Generate another optimization (OPT2, OPT3, etc.)
                        opt_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                        if opt_calc_id:
                            new_calc_ids.append(opt_calc_id)
                    elif next_base_type == "SP":
                        sp_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                        if sp_calc_id:
                            new_calc_ids.append(sp_calc_id)
                    elif next_base_type == "FREQ":
                        # FREQ needs optimized geometry from the highest numbered OPT calculation
                        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                        
                        # Build completed_by_type dict for finding highest OPT
                        completed_by_type = {}
                        for calc in all_calcs:
                            if calc['status'] == 'completed':
                                calc_type = calc['calc_type']
                                if calc_type not in completed_by_type:
                                    completed_by_type[calc_type] = []
                                completed_by_type[calc_type].append(calc)
                        
                        # Find the highest numbered OPT
                        opt_calc_id = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        
                        if opt_calc_id:
                            freq_calc_id = self.generate_freq_from_opt(opt_calc_id)
                            if freq_calc_id:
                                new_calc_ids.append(freq_calc_id)
                        else:
                            print(f"No completed OPT calculation found for FREQ generation")
            else:
                # Default behavior: generate SP and FREQ in parallel
                sp_calc_id = self.generate_sp_from_opt(completed_calc_id)
                if sp_calc_id:
                    new_calc_ids.append(sp_calc_id)
                # Also generate FREQ from OPT (runs in parallel with SP)
                freq_calc_id = self.generate_freq_from_opt(completed_calc_id)
                if freq_calc_id:
                    new_calc_ids.append(freq_calc_id)
        
        elif base_type == "SP":
            # Generate next steps based on workflow plan or default behavior
            print(f"SP completed. Planned sequence: {planned_sequence}")
            if planned_sequence:
                # Find current position and get next steps
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                next_steps = self._get_next_steps_from_sequence(current_index, planned_sequence, calc_type)
                
                # Generate calculations for all next steps (which may be parallel)
                for next_calc_type in next_steps:
                    next_base_type, next_num = self._parse_calc_type(next_calc_type)
                    
                    if next_base_type == "DOSS":
                        print(f"Generating {next_calc_type} from planned sequence...")
                        doss_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                        if doss_calc_id:
                            new_calc_ids.append(doss_calc_id)
                    elif next_base_type == "BAND":
                        print(f"Generating {next_calc_type} from planned sequence...")
                        band_calc_id = self.generate_property_calculation(completed_calc_id, next_calc_type)
                        if band_calc_id:
                            new_calc_ids.append(band_calc_id)
                    elif next_base_type == "OPT":
                        # Check if we have a previous OPT to use
                        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                        completed_by_type = {}
                        for calc in all_calcs:
                            if calc['status'] == 'completed':
                                ct = calc['calc_type']
                                if ct not in completed_by_type:
                                    completed_by_type[ct] = []
                                completed_by_type[ct].append(calc)
                        
                        # Find any completed OPT
                        opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        
                        if opt_source:
                            # Use existing OPT as source
                            print(f"Generating {next_calc_type} from previous OPT...")
                            opt_calc_id = self.generate_numbered_calculation(opt_source, next_calc_type)
                        else:
                            # No OPT exists, generate from CIF
                            print(f"No previous OPT found. Generating {next_calc_type} from CIF...")
                            opt_calc_id = self.generate_calculation_from_cif(material_id, next_calc_type)
                        
                        if opt_calc_id:
                            new_calc_ids.append(opt_calc_id)
                    elif next_base_type == "SP":
                        # Generate another SP from current SP
                        print(f"Generating {next_calc_type} from SP...")
                        sp_calc_id = self.generate_numbered_calculation(completed_calc_id, next_calc_type)
                        if sp_calc_id:
                            new_calc_ids.append(sp_calc_id)
            else:
                # Default behavior: generate both DOSS and BAND
                print("No planned sequence found. Using default: generating both DOSS and BAND...")
                doss_calc_id = self.generate_doss_from_sp(completed_calc_id)
                if doss_calc_id:
                    new_calc_ids.append(doss_calc_id)
                    
                band_calc_id = self.generate_band_from_sp(completed_calc_id)
                if band_calc_id:
                    new_calc_ids.append(band_calc_id)
                
        elif base_type in ["FREQ", "BAND", "DOSS"]:
            # These calculations are often terminal, but sometimes workflow continues
            if planned_sequence:
                current_index = self._find_calc_position_in_sequence(calc_type, completed_calc, planned_sequence)
                next_steps = self._get_next_steps_from_sequence(current_index, planned_sequence, calc_type)
                
                for next_calc_type in next_steps:
                    next_base_type, next_num = self._parse_calc_type(next_calc_type)
                    
                    if next_base_type == "OPT":
                        # OPT after FREQ/BAND/DOSS needs geometry from highest completed OPT
                        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                        completed_by_type = {}
                        for calc in all_calcs:
                            if calc['status'] == 'completed':
                                ct = calc['calc_type']
                                if ct not in completed_by_type:
                                    completed_by_type[ct] = []
                                completed_by_type[ct].append(calc)
                        
                        # Find highest completed OPT
                        opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        if opt_source:
                            opt_calc_id = self.generate_numbered_calculation(opt_source, next_calc_type)
                            if opt_calc_id:
                                new_calc_ids.append(opt_calc_id)
                        else:
                            print(f"No completed OPT found to use as source for {next_calc_type}")
                    elif next_base_type == "FREQ":
                        # Another FREQ calculation - also needs OPT geometry
                        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
                        completed_by_type = {}
                        for calc in all_calcs:
                            if calc['status'] == 'completed':
                                ct = calc['calc_type']
                                if ct not in completed_by_type:
                                    completed_by_type[ct] = []
                                completed_by_type[ct].append(calc)
                        
                        opt_source = self._find_highest_numbered_calc_of_type(completed_by_type, 'OPT')
                        if opt_source:
                            freq_calc_id = self.generate_freq_from_opt(opt_source)
                            if freq_calc_id:
                                new_calc_ids.append(freq_calc_id)
                    # Add other calculation types as needed
        
        # Note: We do NOT check for all pending calculations here anymore
        # The workflow should progress step by step based on actual dependencies
        # This prevents premature triggering of later steps
        
        if new_calc_ids:
            print(f"Generated {len(new_calc_ids)} new calculations: {new_calc_ids}")
        else:
            print(f"No follow-up calculations generated for {calc_type}")
            
        return new_calc_ids
        
    def _find_calc_position_in_sequence(self, calc_type: str, completed_calc: Dict, planned_sequence: List[str]) -> int:
        """
        Find the position of the current calculation in the planned sequence.
        
        Handles numbered calculation types (OPT2, OPT3, SP2, BAND2, etc.)
        """
        # Extract base type and number from calc_type (e.g., OPT2 -> OPT, 2)
        base_type, type_num = self._parse_calc_type(calc_type)
        
        # Count how many calculations of this base type have been completed for this material
        material_id = completed_calc['material_id']
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Count completed calculations of this base type
        completed_count = 0
        for calc in all_calcs:
            calc_base, _ = self._parse_calc_type(calc['calc_type'])
            if calc_base == base_type and calc['status'] == 'completed':
                completed_count += 1
        
        # Find all positions of this base type in the sequence
        type_positions = []
        for i, seq_type in enumerate(planned_sequence):
            seq_base, _ = self._parse_calc_type(seq_type)
            if seq_base == base_type:
                type_positions.append(i)
        
        # Return the position corresponding to the just-completed calculation
        # completed_count = 1 means we just finished the first OPT, so we're at position 0
        if completed_count > 0 and completed_count <= len(type_positions):
            return type_positions[completed_count - 1]
        elif type_positions:
            # We've done more than planned, return last position of this type
            return type_positions[-1]
        else:
            # Not found in sequence, return end
            return len(planned_sequence) - 1
                
    def _get_next_steps_from_sequence(self, current_index: int, planned_sequence: List[str], 
                                     completed_calc_type: str) -> List[str]:
        """
        Get the next calculation steps from the planned sequence.
        
        Returns only the immediate next step(s) in the sequence. The only exception
        is for truly parallel calculations that can use the same wavefunction
        (e.g., BAND and DOSS which both read from the same SP wavefunction).
        
        Args:
            current_index: Current position in the sequence
            planned_sequence: The full planned calculation sequence
            completed_calc_type: The type of calculation that just completed
            
        Returns:
            List of calculation types that should be started next
        """
        if not planned_sequence or current_index >= len(planned_sequence) - 1:
            return []
        
        # Get the immediate next step
        next_index = current_index + 1
        next_steps = []
        
        if next_index < len(planned_sequence):
            next_calc = planned_sequence[next_index]
            next_steps.append(next_calc)
            
            # Only handle truly parallel cases where calculations can share the same input
            next_base, _ = self._parse_calc_type(next_calc)
            
            # BAND and DOSS can run in parallel as they both just read the wavefunction
            if next_base in ["BAND", "DOSS"]:
                # Check if the other property calculation is also in the sequence
                other_prop = "DOSS" if next_base == "BAND" else "BAND"
                # Look for the other property calculation nearby in the sequence
                for i in range(max(0, next_index - 1), min(len(planned_sequence), next_index + 2)):
                    if i != next_index and planned_sequence[i] == other_prop:
                        # Check if it hasn't been started yet
                        if other_prop not in next_steps:
                            next_steps.append(other_prop)
                        break
        
        return next_steps
        
    def _parse_calc_type(self, calc_type: str) -> Tuple[str, int]:
        """
        Parse calculation type to extract base type and number.
        
        Examples:
            OPT -> (OPT, 1)
            OPT2 -> (OPT, 2)
            SP -> (SP, 1)
            SP2 -> (SP, 2)
            BAND3 -> (BAND, 3)
        """
        import re
        match = re.match(r'^([A-Z]+)(\d*)$', calc_type)
        if match:
            base_type = match.group(1)
            num_str = match.group(2)
            num = int(num_str) if num_str else 1
            return base_type, num
        else:
            # Fallback for unexpected formats
            return calc_type, 1
    
    def _find_dependency_in_sequence(self, calc_type: str, planned_sequence: List[str]) -> Optional[str]:
        """
        Find what calculation type this step depends on based on the planned sequence.
        
        This looks at the workflow sequence to determine the actual dependency,
        not making assumptions about what depends on what.
        
        Args:
            calc_type: The calculation type we're checking dependencies for
            planned_sequence: The full planned calculation sequence
            
        Returns:
            The calculation type this depends on, or None if not found
        """
        if not planned_sequence or calc_type not in planned_sequence:
            return None
            
        # Find the position of this calc type in the sequence
        try:
            calc_index = planned_sequence.index(calc_type)
        except ValueError:
            return None
            
        # If it's the first step, it has no dependencies
        if calc_index == 0:
            return None
            
        # The dependency is the previous completed step
        # (not necessarily the immediate previous in sequence)
        return planned_sequence[calc_index - 1]
    
    def _find_highest_numbered_calc_of_type(self, completed_by_type: Dict[str, List[Dict]], base_type: str) -> Optional[str]:
        """
        Find the highest numbered completed calculation of a given base type.
        
        Args:
            completed_by_type: Dictionary mapping calc_type to list of completed calculations
            base_type: Base calculation type (e.g., 'OPT', 'SP')
            
        Returns:
            Calculation ID of the highest numbered calc of this type, or None
        """
        highest_calc_id = None
        highest_num = 0
        
        # Check all completed calculations
        for calc_type, calcs in completed_by_type.items():
            calc_base, calc_num = self._parse_calc_type(calc_type)
            if calc_base == base_type and calc_num > highest_num:
                highest_num = calc_num
                if calcs:  # Make sure there are calculations
                    highest_calc_id = calcs[-1]['calc_id']  # Take the most recent
        
        return highest_calc_id
        
    def generate_opt2_from_opt(self, opt_calc_id: str) -> Optional[str]:
        """
        Generate OPT2 calculation from completed OPT using CRYSTALOptToD12.py.
        
        Args:
            opt_calc_id: ID of completed OPT calculation
            
        Returns:
            New OPT2 calculation ID if successful, None otherwise
        """
        return self.generate_numbered_calculation(opt_calc_id, "OPT2")
    
    def find_original_cif_source(self, material_id: str) -> Optional[Path]:
        """
        Find the original CIF file that was used to create this material.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Path to CIF file if found, None otherwise
        """
        # Look for CIF in workflow_inputs or workflow configuration
        workflow_base = self.base_work_dir
        
        # Check workflow_inputs directory
        workflow_inputs = workflow_base / "workflow_inputs"
        if workflow_inputs.exists():
            # Look for CIF files that match the material name
            core_name = self.extract_core_material_name(material_id)
            cif_patterns = [f"{core_name}.cif", f"*{core_name}*.cif"]
            
            for pattern in cif_patterns:
                cif_files = list(workflow_inputs.glob(pattern))
                if cif_files:
                    return cif_files[0]
        
        # Check workflow config for CIF directory
        config_dir = workflow_base / "workflow_configs"
        if config_dir.exists():
            # Look for workflow plan files
            for plan_file in config_dir.glob("workflow_plan_*.json"):
                try:
                    with open(plan_file, 'r') as f:
                        plan = json.load(f)
                        
                    if plan.get('input_type') == 'cif' and plan.get('input_directory'):
                        cif_dir = Path(plan['input_directory'])
                        if cif_dir.exists():
                            # Look for matching CIF file
                            core_name = self.extract_core_material_name(material_id)
                            for cif_file in cif_dir.glob("*.cif"):
                                if core_name in cif_file.stem:
                                    return cif_file
                except Exception as e:
                    print(f"Error reading workflow plan {plan_file}: {e}")
        
        return None
    
    def generate_calculation_from_cif(self, material_id: str, calc_type: str) -> Optional[str]:
        """
        Generate a calculation (OPT or SP) from the original CIF file.
        
        Args:
            material_id: Material identifier
            calc_type: Calculation type to generate (e.g., 'OPT', 'SP')
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {calc_type} from CIF for {material_id}")
        
        # Find the original CIF file
        cif_file = self.find_original_cif_source(material_id)
        if not cif_file:
            print(f"Could not find original CIF file for {material_id}")
            return None
        
        print(f"  Found CIF file: {cif_file}")
        
        # Create working directory for CIF conversion
        work_dir = self.create_isolated_calculation_directory(
            material_id, f"{calc_type}_from_cif_generation", [cif_file]
        )
        
        try:
            # Get NewCifToD12.py script
            newcif_script = self.script_paths.get('newcif_to_d12')
            if not newcif_script:
                print("NewCifToD12.py script not found")
                return None
            
            # Prepare arguments for NewCifToD12.py
            args = [
                "--cif-file", str(work_dir / cif_file.name),
                "--output-dir", str(work_dir)
            ]
            
            # Check for CIF conversion config
            config_file = self.base_work_dir / "workflow_configs" / "cif_conversion_config.json"
            if config_file.exists():
                args.extend(["--config-file", str(config_file)])
                print(f"  Using CIF conversion config: {config_file}")
            
            # Determine calculation type for NewCifToD12.py
            # 1 = OPT, 2 = SP
            calc_type_num = "1" if calc_type.startswith("OPT") else "2"
            
            # Prepare input responses for non-interactive mode
            # This assumes using config file or defaults
            input_responses = f"{calc_type_num}\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                newcif_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"NewCifToD12.py failed: {stderr}")
                return None
            
            # Find generated D12 file
            d12_files = list(work_dir.glob("*.d12"))
            if not d12_files:
                print("No D12 file generated by NewCifToD12.py")
                return None
            
            generated_d12 = d12_files[0]
            
            # Get workflow output directory
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            workflow_base = self.base_work_dir / "workflow_outputs" / workflow_id
            
            # Determine step number and create directory
            step_num = self._get_next_step_number(workflow_base, calc_type)
            core_name = self.extract_core_material_name(material_id)
            
            # Parse calc type for numbered calculations
            base_type, type_num = self._parse_calc_type(calc_type)
            if type_num > 1:
                dir_suffix = f"_{base_type.lower()}{type_num}"
            else:
                dir_suffix = f"_{base_type.lower()}"
            
            dir_name = f"{core_name}{dir_suffix}"
            calc_dir = workflow_base / f"step_{step_num:03d}_{calc_type}" / dir_name
            calc_dir.mkdir(parents=True, exist_ok=True)
            
            # Move D12 file to calculation directory
            final_d12_name = f"{core_name}{dir_suffix}.d12"
            final_d12_path = calc_dir / final_d12_name
            shutil.move(generated_d12, final_d12_path)
            
            # Create SLURM script
            job_name = f"{core_name}{dir_suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                calc_dir, job_name, base_type, step_num, workflow_id
            )
            
            # Create calculation record
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=calc_type,
                input_file=str(final_d12_path),
                work_dir=str(calc_dir),
                settings={
                    'generated_from': 'CIF',
                    'generation_method': 'NewCifToD12.py',
                    'cif_source': str(cif_file),
                    'workflow_id': workflow_id,
                    'step_number': step_num
                }
            )
            
            # Submit if auto-submit is enabled
            if hasattr(self, 'auto_submit') and self.auto_submit:
                job_id = self._submit_calculation_to_slurm(slurm_script_path, calc_dir)
                if job_id:
                    self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=job_id)
                    print(f"Submitted {calc_type} calculation as job {job_id}: {final_d12_path}")
                else:
                    print(f"Generated {calc_type} calculation but submission failed: {final_d12_path}")
            else:
                print(f"Generated {calc_type} calculation (pending submission): {final_d12_path}")
            
            return calc_id
            
        finally:
            # Clean up working directory
            shutil.rmtree(work_dir, ignore_errors=True)
    
    def generate_numbered_calculation(self, source_calc_id: str, target_calc_type: str) -> Optional[str]:
        """
        Generate a numbered calculation (OPT2, OPT3, SP2, etc.) from a source calculation.
        
        Args:
            source_calc_id: ID of source calculation
            target_calc_type: Target calculation type (e.g., OPT2, SP2)
            
        Returns:
            New calculation ID if successful, None otherwise
        """
        print(f"Generating {target_calc_type} calculation from {source_calc_id}")
        
        # Get source calculation details
        source_calc = self.db.get_calculation(source_calc_id)
        if not source_calc or source_calc['status'] != 'completed':
            print(f"Source calculation {source_calc_id} not completed")
            return None
            
        material_id = source_calc['material_id']
        source_output_file = Path(source_calc['output_file'])
        source_input_file = Path(source_calc['input_file'])
        
        if not source_output_file.exists():
            print(f"Source output file not found: {source_output_file}")
            return None
            
        # Parse target calculation type
        target_base_type, target_num = self._parse_calc_type(target_calc_type)
        
        # Check for expert config file for numbered calculations (OPT2/OPT3/SP2/FREQ)
        expert_config_file = None
        if (target_base_type in ["OPT", "SP", "FREQ"]) and (target_num > 1 or target_base_type == "FREQ"):
            # Check for expert config in workflow_temp directory
            config_dir = self.base_work_dir / "workflow_temp" / f"expert_config_{target_calc_type.lower()}"
            if config_dir.exists():
                config_files = list(config_dir.glob("*_expert_config.json"))
                if config_files:
                    expert_config_file = config_files[0]
                    print(f"  Found expert config for {target_calc_type}: {expert_config_file}")
            
        # For OPT2/OPT3, we need to find the ORIGINAL input file (from CIF or initial OPT)
        # to preserve symmetry information
        original_input_file = None
        if target_base_type == "OPT" and target_num > 1:
            # Try to find the original OPT input file
            all_calcs = self.db.get_calculations_by_status(material_id=material_id)
            for calc in all_calcs:
                if calc['calc_type'] == 'OPT' and calc['status'] in ['completed', 'running', 'submitted']:
                    original_opt_input = Path(calc['input_file'])
                    if original_opt_input.exists() and original_opt_input != source_input_file:
                        original_input_file = original_opt_input
                        print(f"  Found original OPT input for symmetry: {original_input_file.name}")
                        break
        
        # Create isolated directory for CRYSTALOptToD12.py
        files_to_copy = [source_output_file, source_input_file]
        if original_input_file and original_input_file.exists():
            files_to_copy.append(original_input_file)
        
        work_dir = self.create_isolated_calculation_directory(
            material_id, f"{target_calc_type}_generation", files_to_copy
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
            
            # If we have an expert config file for OPT2/OPT3, use it
            if expert_config_file and expert_config_file.exists():
                args.extend(["--config-file", str(expert_config_file)])
                print(f"  Using expert config file for {target_calc_type}")
                print(f"  Config file path: {expert_config_file}")
                # Read and display config contents for debugging
                try:
                    with open(expert_config_file, 'r') as f:
                        config_content = json.load(f)
                        print(f"  Config functional: {config_content.get('functional', 'N/A')}")
                        print(f"  Config dispersion: {config_content.get('dispersion', 'N/A')}")
                except Exception as e:
                    print(f"  Warning: Could not read config file: {e}")
                # With config file, we just need to confirm applying it
                # 1. Apply config? → y (yes)
                input_responses = "y\n"
            # Prepare input responses for non-interactive execution based on target type
            # 1. Keep settings? → y (yes, keep original settings)
            elif target_base_type == "OPT":
                # 2. Calc type → 2 (OPT)
                # 3. Symmetry choice → 1 (Write only unique atoms)
                input_responses = "y\n2\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            elif target_base_type == "SP":
                # 2. Calc type → 1 (SP)
                # 3. Symmetry choice → 1 (Write only unique atoms)
                input_responses = "y\n1\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            elif target_base_type == "FREQ":
                # 2. Calc type → 3 (FREQ)
                input_responses = "y\n3\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            else:
                # Default to SP for unknown types
                input_responses = "y\n1\n1\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            
            success, stdout, stderr = self.run_script_in_isolated_directory(
                crystal_to_d12_script, work_dir, args, input_data=input_responses
            )
            
            if not success:
                print(f"CRYSTALOptToD12.py failed: {stderr}")
                return None
            
            # Debug: show some output to verify config was applied
            if expert_config_file and expert_config_file.exists():
                print(f"  CRYSTALOptToD12.py output (last 10 lines):")
                output_lines = stdout.strip().split('\n')
                for line in output_lines[-10:]:
                    print(f"    {line}")
                
            # Get the original input file name to exclude it from results
            original_d12_name = source_input_file.name
            
            # Find generated .d12 file based on target type
            if target_base_type == "OPT":
                generated_files = list(work_dir.glob("*OPT*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_opt*.d12")) + list(work_dir.glob("*OPTGEOM*.d12"))
            elif target_base_type == "SP":
                generated_files = list(work_dir.glob("*SP*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_sp*.d12")) + list(work_dir.glob("*SCFDIR*.d12"))
            elif target_base_type == "FREQ":
                generated_files = list(work_dir.glob("*FREQ*.d12"))
                if not generated_files:
                    generated_files = list(work_dir.glob("*_freq*.d12"))
            else:
                # Generic pattern
                generated_files = list(work_dir.glob("*.d12"))
            
            # Filter out the original input file and look for files with optimized/B3LYP/modified in name
            filtered_files = []
            for f in generated_files:
                if f.name != original_d12_name:
                    # Prefer files with these indicators that they were modified
                    if any(x in f.name.lower() for x in ['b3lyp', 'optimized', 'modified', f'_{target_base_type.lower()}_']):
                        filtered_files.insert(0, f)  # Put at beginning
                    else:
                        filtered_files.append(f)
            
            if filtered_files:
                generated_files = filtered_files
            else:
                # If no filtered files, exclude just the original
                generated_files = [f for f in generated_files if f.name != original_d12_name]
                
            if not generated_files:
                print(f"No {target_base_type} input file generated by CRYSTALOptToD12.py")
                return None
                
            generated_input_file = generated_files[0]
            print(f"  Selected generated file: {generated_input_file.name}")
            
            # Get workflow output directory and create material-specific directory
            workflow_base = self.get_workflow_output_base(source_calc)
            
            # Get core material name
            core_name = self.extract_core_material_name(material_id)
            
            # Create directory naming based on target type and number
            if target_num > 1:
                suffix = f"_{target_base_type.lower()}{target_num}"
            else:
                suffix = f"_{target_base_type.lower()}"
            
            # Find the step number for this calculation type
            step_num = self._get_next_step_number(workflow_base, target_calc_type)
            
            dir_name = f"{core_name}{suffix}"
            # Use the full calculation type in the step directory name
            step_dir = workflow_base / f"step_{step_num:03d}_{target_calc_type}" / dir_name
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Move generated file to material's directory with proper naming
            clean_name = f"{core_name}{suffix}.d12"
            final_location = step_dir / clean_name
            shutil.move(generated_input_file, final_location)
            
            # Fix naming for numbered calculations if needed
            if target_num > 1:
                self._fix_numbered_naming(step_dir, core_name, target_base_type, target_num)
            
            # Create SLURM script for calculation
            job_name = f"{core_name}{suffix}"
            slurm_script_path = self._create_slurm_script_for_calculation(
                step_dir, job_name, target_base_type, step_num, workflow_base.name
            )
            
            # Create calculation record
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=target_calc_type,
                input_file=str(final_location),
                work_dir=str(step_dir),
                settings={
                    'workflow_id': workflow_base.name,
                    'step_number': step_num,
                    'generated_from': source_calc_id,
                    'generation_method': 'CRYSTALOptToD12.py',
                    'parent_calc_id': source_calc_id
                }
            )
            
            # Create workflow metadata file for callback tracking
            metadata = {
                'workflow_id': workflow_base.name,
                'step_num': step_num,
                'calc_type': target_calc_type,
                'material_id': material_id,
                'calc_id': calc_id,
                'generated_from': source_calc_id
            }
            metadata_file = step_dir / '.workflow_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Submit if auto-submission is enabled
            if hasattr(self, 'auto_submit') and self.auto_submit:
                job_id = self._submit_calculation_to_slurm(slurm_script_path, step_dir)
                if job_id:
                    self.db.update_calculation_status(calc_id, 'submitted', slurm_job_id=job_id)
                    print(f"Submitted {target_calc_type} calculation as job {job_id}: {final_location}")
                else:
                    print(f"Generated {target_calc_type} calculation {calc_id} but submission failed: {final_location}")
            else:
                print(f"Generated {target_calc_type} calculation {calc_id} (pending submission): {final_location}")
            
            return calc_id
            
        finally:
            # Clean up isolated directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
    def _get_next_step_number(self, workflow_base: Path, calc_type: str) -> int:
        """Get the next available step number for a calculation type."""
        # For numbered calculations (OPT2, OPT3, SP2, etc.), we need to check the base type
        # but create separate steps for each numbered variant
        base_type, type_num = self._parse_calc_type(calc_type)
        
        # Check if a step already exists for this EXACT calculation type (including number)
        for step_dir in workflow_base.glob("step_*"):
            if step_dir.is_dir():
                # Check if the directory name contains this calc_type
                dir_parts = step_dir.name.split('_')
                if len(dir_parts) >= 3:
                    # For numbered types, match exact type (OPT2, not just OPT)
                    if type_num > 1:
                        step_calc_type = '_'.join(dir_parts[2:])  # Handle names like step_002_OPT2
                        if step_calc_type == calc_type or step_calc_type == base_type + str(type_num):
                            try:
                                return int(dir_parts[1])
                            except ValueError:
                                pass
                    else:
                        # For base types (OPT, SP, etc.), only match if it's the base type
                        if dir_parts[2] == calc_type:
                            try:
                                return int(dir_parts[1])
                            except ValueError:
                                pass
        
        # No existing step for this calc type, find the next available number
        existing_steps = []
        for step_dir in workflow_base.glob("step_*"):
            if step_dir.is_dir():
                # Extract step number from directory name
                try:
                    step_num = int(step_dir.name.split('_')[1])
                    existing_steps.append(step_num)
                except (IndexError, ValueError):
                    pass
                    
        # Return next available number
        return max(existing_steps) + 1 if existing_steps else 1
        
        
    def _fix_numbered_naming(self, output_dir: Path, core_name: str, base_type: str, type_num: int):
        """Fix naming for numbered calculation files generated by CRYSTALOptToD12.py"""
        # CRYSTALOptToD12.py might generate files with duplicate type suffixes
        # e.g., material_sp_sp.d12 should be material_sp2.d12
        
        type_lower = base_type.lower()
        for file_path in output_dir.glob("*.d12"):
            # Check for duplicate suffixes like _sp_sp or _opt_opt
            if f"_{type_lower}_{type_lower}" in file_path.name:
                # Create new name with number
                new_name = f"{core_name}_{type_lower}{type_num}.d12"
                new_path = output_dir / new_name
                
                if not new_path.exists():
                    file_path.rename(new_path)
                    print(f"        Renamed: {file_path.name} → {new_name}")
                    
    def _find_most_recent_wavefunction_calc(self, material_id: str) -> Optional[str]:
        """
        Find the most recent completed calculation with a wavefunction (SP or OPT).
        
        BAND and DOSS calculations need the wavefunction from the most recent SP or OPT.
        
        Returns:
            Calculation ID of the most recent wavefunction calculation, or None if not found
        """
        all_calcs = self.db.get_calculations_by_status(material_id=material_id)
        
        # Filter for completed calculations that produce wavefunctions
        wf_calcs = []
        for calc in all_calcs:
            if calc['status'] == 'completed':
                calc_base, _ = self._parse_calc_type(calc['calc_type'])
                if calc_base in ['SP', 'OPT']:
                    # Check if output file exists and wavefunction file exists
                    output_file = Path(calc.get('output_file', ''))
                    if output_file.exists():
                        wf_file = output_file.parent / 'fort.9'
                        f9_file = output_file.with_suffix('.f9')
                        if wf_file.exists() or f9_file.exists():
                            wf_calcs.append(calc)
        
        # Sort by completion time and return most recent
        if wf_calcs:
            wf_calcs.sort(key=lambda c: c.get('end_time', c.get('start_time', '')), reverse=True)
            return wf_calcs[0]['calc_id']
        
        return None
        
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