#!/usr/bin/env python3
"""
CRYSTAL Error Recovery Engine
-----------------------------
Automated error detection and recovery for CRYSTAL calculations.
Integrates with existing error detection scripts and provides
configurable recovery strategies.

Key Features:
- Integration with existing fixk.py and updatelists2.py
- YAML-based configuration for recovery rules
- Automatic job resubmission with fixes applied
- Recovery attempt tracking and escalation
- Support for common CRYSTAL error patterns

Author: Based on implementation plan for material tracking system
"""

import os
import sys
import yaml
import subprocess
import shutil
import re
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
    from mace.recovery.detector import CrystalErrorDetector
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Make sure all required Python files are in the same directory as {__file__}")
    sys.exit(1)


class ErrorRecoveryEngine:
    """
    Automated error recovery system for CRYSTAL calculations.
    
    Handles common calculation failures using configurable recovery strategies
    and integrates with existing error detection and fixing scripts.
    """
    
    def __init__(self, db_path: str = "materials.db", config_path: str = "recovery_config.yaml"):
        self.db = MaterialDatabase(db_path)
        self.config_path = Path(config_path)
        self.config = self.load_recovery_config()
        self.error_detector = CrystalErrorDetector(db_path=db_path)
        self.lock = threading.RLock()
        
        # Track recovery attempts to prevent infinite loops
        self.recovery_attempts = {}
        
    def load_recovery_config(self) -> Dict:
        """Load recovery configuration from YAML file."""
        default_config = {
            "error_recovery": {
                "shrink_error": {
                    "handler": "fixk_handler",
                    "max_retries": 3,
                    "resubmit_delay": 300,  # seconds
                    "escalate_on_failure": True
                },
                "memory_error": {
                    "handler": "memory_handler", 
                    "memory_factor": 1.5,
                    "max_memory": "200GB",
                    "max_retries": 2,
                    "resubmit_delay": 600
                },
                "convergence_error": {
                    "handler": "convergence_handler",
                    "max_cycles_increase": 1000,
                    "fmixing_adjustment": 10,
                    "max_retries": 2,
                    "resubmit_delay": 300
                },
                "timeout_error": {
                    "handler": "timeout_handler",
                    "walltime_factor": 2.0,
                    "max_walltime": "48:00:00",
                    "max_retries": 1,
                    "resubmit_delay": 900
                },
                "disk_space_error": {
                    "handler": "cleanup_handler",
                    "cleanup_scratch": True,
                    "max_retries": 1,
                    "resubmit_delay": 1800
                }
            },
            "global_settings": {
                "max_concurrent_recoveries": 10,
                "recovery_log_retention_days": 30,
                "enable_auto_escalation": True,
                "notification_enabled": False
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(user_config)
                    return default_config
            except Exception as e:
                print(f"Error loading recovery config {self.config_path}: {e}")
                print("Using default configuration")
                
        return default_config
        
    def save_default_config(self):
        """Save default configuration to file for user customization."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.load_recovery_config(), f, default_flow_style=False, indent=2)
        print(f"Saved default recovery configuration to {self.config_path}")
        
    def detect_and_recover_errors(self, max_recoveries: int = None) -> int:
        """
        Main entry point - detect failed calculations and attempt recovery.
        
        Args:
            max_recoveries: Maximum number of recoveries to attempt in this run
            
        Returns:
            Number of recovery attempts made
        """
        if max_recoveries is None:
            max_recoveries = self.config["global_settings"].get("max_concurrent_recoveries", 10)
            
        # Get failed calculations that haven't exceeded retry limits
        failed_calcs = self.get_recoverable_calculations()
        
        recovery_count = 0
        for calc in failed_calcs[:max_recoveries]:
            if self.attempt_recovery(calc):
                recovery_count += 1
                
        return recovery_count
        
    def get_recoverable_calculations(self) -> List[Dict]:
        """Get calculations that failed but can still be recovered."""
        failed_calcs = self.db.get_calculations_by_status('failed')
        recoverable = []
        
        for calc in failed_calcs:
            calc_id = calc['calc_id']
            error_type = calc.get('error_type', 'unknown')
            
            # Check if this error type is recoverable
            if error_type not in self.config["error_recovery"]:
                continue
                
            # Check retry limits
            retry_count = self.get_retry_count(calc_id)
            max_retries = self.config["error_recovery"][error_type].get("max_retries", 3)
            
            if retry_count < max_retries:
                recoverable.append(calc)
                
        return recoverable
        
    def get_retry_count(self, calc_id: str) -> int:
        """Get number of recovery attempts for a calculation."""
        # Check database for recovery attempts
        recovery_calcs = self.db.get_calculations_by_status(
            material_id=self.db.get_calculation(calc_id)['material_id']
        )
        
        # Count calculations that are recovery attempts for this calc
        retry_count = 0
        for calc in recovery_calcs:
            if calc.get('parent_calc_id') == calc_id and calc.get('is_recovery_attempt'):
                retry_count += 1
                
        return retry_count
        
    def attempt_recovery(self, calc: Dict) -> bool:
        """
        Attempt to recover a failed calculation.
        
        Args:
            calc: Failed calculation record
            
        Returns:
            True if recovery was attempted, False if skipped
        """
        calc_id = calc['calc_id']
        error_type = calc.get('error_type', 'unknown')
        
        print(f"Attempting recovery for calculation {calc_id} (error: {error_type})")
        
        # Get recovery configuration for this error type
        error_config = self.config["error_recovery"].get(error_type, {})
        handler_name = error_config.get("handler")
        
        if not handler_name:
            print(f"No handler configured for error type: {error_type}")
            return False
            
        # Get the handler method
        handler_method = getattr(self, handler_name, None)
        if not handler_method:
            print(f"Handler method not found: {handler_name}")
            return False
            
        try:
            # Apply recovery fix
            fixed_input_file = handler_method(calc, error_config)
            
            if fixed_input_file:
                # Create new calculation with recovery attempt
                self.create_recovery_calculation(calc, fixed_input_file, error_config)
                return True
            else:
                print(f"Recovery handler failed to generate fixed input for {calc_id}")
                return False
                
        except Exception as e:
            print(f"Error during recovery attempt for {calc_id}: {e}")
            return False
            
    def fixk_handler(self, calc: Dict, config: Dict) -> Optional[Path]:
        """
        Handler for SHRINK parameter errors using existing fixk.py script.
        
        Args:
            calc: Failed calculation record
            config: Recovery configuration for this error type
            
        Returns:
            Path to fixed input file, or None if fix failed
        """
        print(f"Applying fixk.py fix for SHRINK error in {calc['calc_id']}")
        
        # Get original input file
        original_input = Path(calc['input_file'])
        if not original_input.exists():
            print(f"Original input file not found: {original_input}")
            return None
            
        # Create temporary directory for fix
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Copy input file to temp directory
            temp_input = temp_dir / original_input.name
            shutil.copy2(original_input, temp_input)
            
            # Run fixk.py in temp directory
            fixk_script = Path(__file__).parent.parent / "Check_Scripts" / "fixk.py"
            
            try:
                # Change to temp directory and run fixk.py
                result = subprocess.run(
                    [sys.executable, str(fixk_script)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # Create fixed input file with recovery suffix
                    recovery_suffix = f"_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    fixed_name = original_input.stem + recovery_suffix + original_input.suffix
                    fixed_path = original_input.parent / fixed_name
                    
                    # Copy fixed file back
                    shutil.copy2(temp_input, fixed_path)
                    
                    print(f"SHRINK fix applied successfully: {fixed_path}")
                    return fixed_path
                else:
                    print(f"fixk.py failed: {result.stderr}")
                    return None
                    
            except subprocess.TimeoutExpired:
                print("fixk.py timed out")
                return None
            except Exception as e:
                print(f"Error running fixk.py: {e}")
                return None
                
    def memory_handler(self, calc: Dict, config: Dict) -> Optional[Path]:
        """
        Handler for memory-related errors - increases memory allocation.
        
        Args:
            calc: Failed calculation record
            config: Recovery configuration for this error type
            
        Returns:
            Path to fixed input file, or None if fix failed
        """
        print(f"Applying memory fix for {calc['calc_id']}")
        
        # Parse original job script to get current memory
        job_script = Path(calc.get('job_script', ''))
        if not job_script.exists():
            print(f"Job script not found: {job_script}")
            return None
            
        try:
            with open(job_script, 'r') as f:
                script_content = f.read()
                
            # Find current memory allocation
            memory_match = re.search(r'#SBATCH\s+--mem(?:-per-cpu)?[=\s]+(\d+)([GMK]?)B?', script_content)
            
            if memory_match:
                current_memory = int(memory_match.group(1))
                unit = memory_match.group(2) or 'G'
                
                # Convert to GB for calculations
                if unit == 'M':
                    current_memory_gb = current_memory / 1024
                elif unit == 'K':
                    current_memory_gb = current_memory / (1024 * 1024)
                else:  # G or no unit
                    current_memory_gb = current_memory
                    
                # Apply memory factor
                memory_factor = config.get('memory_factor', 1.5)
                new_memory_gb = int(current_memory_gb * memory_factor)
                
                # Check against maximum
                max_memory_str = config.get('max_memory', '200GB')
                max_memory_gb = int(re.search(r'(\d+)', max_memory_str).group(1))
                
                if new_memory_gb > max_memory_gb:
                    new_memory_gb = max_memory_gb
                    
                # Update job script
                new_memory_line = f"#SBATCH --mem={new_memory_gb}GB"
                updated_script = re.sub(
                    r'#SBATCH\s+--mem(?:-per-cpu)?[=\s]+\d+[GMK]?B?',
                    new_memory_line,
                    script_content
                )
                
                # Create new job script
                recovery_suffix = f"_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                new_script_name = job_script.stem + recovery_suffix + job_script.suffix
                new_script_path = job_script.parent / new_script_name
                
                with open(new_script_path, 'w') as f:
                    f.write(updated_script)
                    
                print(f"Memory increased from {current_memory_gb}GB to {new_memory_gb}GB")
                
                # Return original input file (no changes needed)
                return Path(calc['input_file'])
                
            else:
                print("Could not find memory allocation in job script")
                return None
                
        except Exception as e:
            print(f"Error applying memory fix: {e}")
            return None
            
    def convergence_handler(self, calc: Dict, config: Dict) -> Optional[Path]:
        """
        Handler for SCF convergence errors - adjusts convergence parameters.
        
        Args:
            calc: Failed calculation record
            config: Recovery configuration for this error type
            
        Returns:
            Path to fixed input file, or None if fix failed
        """
        print(f"Applying convergence fix for {calc['calc_id']}")
        
        original_input = Path(calc['input_file'])
        if not original_input.exists():
            print(f"Original input file not found: {original_input}")
            return None
            
        try:
            with open(original_input, 'r') as f:
                lines = f.readlines()
                
            # Find and update SCF parameters
            updated_lines = []
            found_maxcycle = False
            found_fmixing = False
            
            for line in lines:
                if 'MAXCYCLE' in line.upper():
                    # Increase MAXCYCLE
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.upper() == 'MAXCYCLE' and i + 1 < len(parts):
                            try:
                                current_cycles = int(parts[i + 1])
                                max_increase = config.get('max_cycles_increase', 1000)
                                new_cycles = current_cycles + max_increase
                                parts[i + 1] = str(new_cycles)
                                found_maxcycle = True
                                print(f"Increased MAXCYCLE from {current_cycles} to {new_cycles}")
                                break
                            except ValueError:
                                pass
                    updated_lines.append(' '.join(parts) + '\n')
                    
                elif 'FMIXING' in line.upper():
                    # Adjust FMIXING
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.upper() == 'FMIXING' and i + 1 < len(parts):
                            try:
                                current_fmix = int(parts[i + 1])
                                adjustment = config.get('fmixing_adjustment', 10)
                                new_fmix = max(10, current_fmix - adjustment)  # Don't go below 10
                                parts[i + 1] = str(new_fmix)
                                found_fmixing = True
                                print(f"Adjusted FMIXING from {current_fmix} to {new_fmix}")
                                break
                            except ValueError:
                                pass
                    updated_lines.append(' '.join(parts) + '\n')
                else:
                    updated_lines.append(line)
                    
            # Create recovery input file
            recovery_suffix = f"_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            fixed_name = original_input.stem + recovery_suffix + original_input.suffix
            fixed_path = original_input.parent / fixed_name
            
            with open(fixed_path, 'w') as f:
                f.writelines(updated_lines)
                
            if found_maxcycle or found_fmixing:
                print(f"Convergence parameters updated: {fixed_path}")
                return fixed_path
            else:
                print("No convergence parameters found to modify")
                return None
                
        except Exception as e:
            print(f"Error applying convergence fix: {e}")
            return None
            
    def timeout_handler(self, calc: Dict, config: Dict) -> Optional[Path]:
        """Handler for timeout errors - increases walltime."""
        print(f"Applying timeout fix for {calc['calc_id']}")
        
        # Similar to memory_handler but adjusts walltime
        job_script = Path(calc.get('job_script', ''))
        if not job_script.exists():
            print(f"Job script not found: {job_script}")
            return None
            
        try:
            with open(job_script, 'r') as f:
                script_content = f.read()
                
            # Find current walltime
            time_match = re.search(r'#SBATCH\s+--time[=\s]+(\d+):(\d+):(\d+)', script_content)
            
            if time_match:
                hours = int(time_match.group(1))
                minutes = int(time_match.group(2))
                seconds = int(time_match.group(3))
                
                total_seconds = hours * 3600 + minutes * 60 + seconds
                
                # Apply time factor
                time_factor = config.get('walltime_factor', 2.0)
                new_total_seconds = int(total_seconds * time_factor)
                
                # Convert back to HH:MM:SS
                new_hours = new_total_seconds // 3600
                new_minutes = (new_total_seconds % 3600) // 60
                new_seconds = new_total_seconds % 60
                
                # Check against maximum
                max_time_str = config.get('max_walltime', '48:00:00')
                max_hours = int(max_time_str.split(':')[0])
                if new_hours > max_hours:
                    new_hours = max_hours
                    new_minutes = 0
                    new_seconds = 0
                    
                # Update job script
                new_time_line = f"#SBATCH --time={new_hours:02d}:{new_minutes:02d}:{new_seconds:02d}"
                updated_script = re.sub(
                    r'#SBATCH\s+--time[=\s]+\d+:\d+:\d+',
                    new_time_line,
                    script_content
                )
                
                # Create new job script
                recovery_suffix = f"_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                new_script_name = job_script.stem + recovery_suffix + job_script.suffix
                new_script_path = job_script.parent / new_script_name
                
                with open(new_script_path, 'w') as f:
                    f.write(updated_script)
                    
                print(f"Walltime increased from {hours:02d}:{minutes:02d}:{seconds:02d} to {new_hours:02d}:{new_minutes:02d}:{new_seconds:02d}")
                
                # Return original input file
                return Path(calc['input_file'])
                
            else:
                print("Could not find walltime in job script")
                return None
                
        except Exception as e:
            print(f"Error applying timeout fix: {e}")
            return None
            
    def cleanup_handler(self, calc: Dict, config: Dict) -> Optional[Path]:
        """Handler for disk space errors - cleans up scratch space."""
        print(f"Applying cleanup fix for {calc['calc_id']}")
        
        # Clean up scratch space if enabled
        if config.get('cleanup_scratch', True):
            work_dir = Path(calc.get('work_dir', ''))
            if work_dir.exists():
                try:
                    # Remove large temporary files
                    for pattern in ['*.tmp', '*.scratch', 'fort.*', '*.f*']:
                        for file_path in work_dir.glob(pattern):
                            if file_path.is_file() and file_path.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                                file_path.unlink()
                                print(f"Removed large file: {file_path}")
                                
                except Exception as e:
                    print(f"Error during cleanup: {e}")
                    
        # Return original input file (no changes needed)
        return Path(calc['input_file'])
        
    def create_recovery_calculation(self, original_calc: Dict, fixed_input_file: Path, 
                                  recovery_config: Dict) -> str:
        """
        Create a new calculation record for the recovery attempt.
        
        Args:
            original_calc: Original failed calculation
            fixed_input_file: Path to fixed input file
            recovery_config: Recovery configuration used
            
        Returns:
            New calculation ID
        """
        material_id = original_calc['material_id']
        calc_type = original_calc['calc_type']
        
        # Generate new calculation ID
        recovery_calc_id = self.db.create_calculation(
            material_id=material_id,
            calc_type=calc_type,
            input_file=str(fixed_input_file),
            priority=original_calc.get('priority', 0) + 1,  # Higher priority for recovery
            settings_json=json.dumps({
                'is_recovery_attempt': True,
                'parent_calc_id': original_calc['calc_id'],
                'recovery_strategy': recovery_config.get('handler'),
                'recovery_timestamp': datetime.now().isoformat()
            })
        )
        
        print(f"Created recovery calculation {recovery_calc_id} for {original_calc['calc_id']}")
        
        # Add delay before resubmission if configured
        resubmit_delay = recovery_config.get('resubmit_delay', 0)
        if resubmit_delay > 0:
            print(f"Recovery will be delayed by {resubmit_delay} seconds")
            
        return recovery_calc_id
        
    def get_recovery_statistics(self) -> Dict:
        """Get statistics on recovery attempts and success rates."""
        stats = {
            'total_failed_calculations': len(self.db.get_calculations_by_status('failed')),
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'error_type_breakdown': {},
            'recovery_success_rate': 0.0
        }
        
        # Get all calculations to analyze recovery patterns
        all_calcs = self.db.get_calculations_by_status()
        
        for calc in all_calcs:
            settings = json.loads(calc.get('settings_json', '{}'))
            if settings.get('is_recovery_attempt'):
                stats['recovery_attempts'] += 1
                if calc['status'] == 'completed':
                    stats['successful_recoveries'] += 1
                    
                # Track by error type
                parent_calc = self.db.get_calculation(settings.get('parent_calc_id'))
                if parent_calc:
                    error_type = parent_calc.get('error_type', 'unknown')
                    if error_type not in stats['error_type_breakdown']:
                        stats['error_type_breakdown'][error_type] = {'attempts': 0, 'successes': 0}
                    stats['error_type_breakdown'][error_type]['attempts'] += 1
                    if calc['status'] == 'completed':
                        stats['error_type_breakdown'][error_type]['successes'] += 1
                        
        # Calculate success rate
        if stats['recovery_attempts'] > 0:
            stats['recovery_success_rate'] = stats['successful_recoveries'] / stats['recovery_attempts']
            
        return stats


def main():
    """CLI interface for error recovery engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL Error Recovery Engine")
    parser.add_argument("--action", choices=['recover', 'stats', 'config'], 
                       default='recover', help="Action to perform")
    parser.add_argument("--config", default="recovery_config.yaml",
                       help="Path to recovery configuration file")
    parser.add_argument("--db", default="materials.db",
                       help="Path to materials database")
    parser.add_argument("--max-recoveries", type=int, default=10,
                       help="Maximum number of recoveries to attempt")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file")
    
    args = parser.parse_args()
    
    # Initialize recovery engine
    recovery_engine = ErrorRecoveryEngine(args.db, args.config)
    
    if args.create_config or args.action == 'config':
        recovery_engine.save_default_config()
        print(f"Configuration saved to {args.config}")
        return
        
    elif args.action == 'stats':
        stats = recovery_engine.get_recovery_statistics()
        print("\n=== Error Recovery Statistics ===")
        print(f"Total failed calculations: {stats['total_failed_calculations']}")
        print(f"Recovery attempts: {stats['recovery_attempts']}")
        print(f"Successful recoveries: {stats['successful_recoveries']}")
        print(f"Success rate: {stats['recovery_success_rate']:.1%}")
        
        if stats['error_type_breakdown']:
            print("\nBreakdown by error type:")
            for error_type, data in stats['error_type_breakdown'].items():
                success_rate = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
                print(f"  {error_type}: {data['successes']}/{data['attempts']} ({success_rate:.1%})")
                
    elif args.action == 'recover':
        print("Starting error recovery process...")
        recovered = recovery_engine.detect_and_recover_errors(args.max_recoveries)
        print(f"Attempted recovery for {recovered} calculations")


if __name__ == "__main__":
    main()