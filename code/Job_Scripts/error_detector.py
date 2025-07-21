#!/usr/bin/env python3
"""
Error Detection Engine for CRYSTAL Material Tracking System
-----------------------------------------------------------
Integrates with existing updatelists2.py logic to detect and classify errors
in CRYSTAL calculations. Provides enhanced error analysis and recovery suggestions.

Key Features:
- Integration with existing updatelists2.py error patterns
- Enhanced error classification and analysis
- Error statistics and reporting
- Recovery recommendation system
- Database integration for error tracking

Author: Based on implementation plan for material tracking system
"""

import os
import re
import json
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import threading
from collections import defaultdict, Counter

# Import our material database and file manager
from material_database import MaterialDatabase
from crystal_file_manager import CrystalFileManager


class CrystalErrorDetector:
    """
    Advanced error detection and analysis for CRYSTAL calculations.
    
    Integrates with existing updatelists2.py logic while providing enhanced
    functionality for error recovery and tracking.
    """
    
    def __init__(self, base_dir: str = ".", db_path: str = "materials.db", 
                 enable_tracking: bool = True):
        self.base_dir = Path(base_dir).resolve()
        self.enable_tracking = enable_tracking
        self.lock = threading.RLock()
        
        # Initialize database connection
        if self.enable_tracking:
            self.db = MaterialDatabase(db_path)
        else:
            self.db = None
            
        # Initialize file manager for integration
        self.file_manager = CrystalFileManager(base_dir, db_path, enable_tracking)
        
        # Enhanced error patterns based on updatelists2.py with additional context
        self.error_patterns = {
            'scf_convergence': {
                'patterns': ["TOO MANY CYCLES", "SCF NOT CONVERGED", "CONVERGENCE NOT ACHIEVED"],
                'severity': 'medium',
                'recoverable': True,
                'description': 'SCF convergence failure',
                'recovery_hints': [
                    'Increase MAXCYCLE for SCF',
                    'Adjust FMIXING percentage',
                    'Try different SCF method (ANDERSON, BROYDEN)',
                    'Use better initial guess',
                    'Reduce integration tolerances'
                ]
            },
            'memory_error': {
                'patterns': [
                    "out-of-memory handler", "INSUFFICIENT MEMORY", "OUT OF MEMORY",
                    "MEMORY ALLOCATION", "segmentation fault", "Stack trace terminated abnormally"
                ],
                'severity': 'high',
                'recoverable': True,
                'description': 'Memory allocation failure',
                'recovery_hints': [
                    'Request more memory in job script',
                    'Reduce basis set size',
                    'Use smaller k-point mesh',
                    'Enable direct SCF if available',
                    'Split calculation into smaller parts'
                ]
            },
            'disk_quota': {
                'patterns': ["error during write", "disk full", "No space left on device"],
                'severity': 'high',
                'recoverable': True,
                'description': 'Disk space or quota exceeded',
                'recovery_hints': [
                    'Clean up old calculation files',
                    'Request more disk quota',
                    'Use scratch directory for temporary files',
                    'Compress or archive old results'
                ]
            },
            'time_limit': {
                'patterns': ["DUE TO TIME LIMIT", "TIME LIMIT EXCEEDED", "killed by signal 15"],
                'severity': 'medium',
                'recoverable': True,
                'description': 'Job time limit exceeded',
                'recovery_hints': [
                    'Request longer wall time',
                    'Restart from checkpoint if available',
                    'Reduce calculation complexity',
                    'Use more efficient settings'
                ]
            },
            'geometry_error': {
                'patterns': [
                    "GEOMETRY OPTIMIZATION FAILED", "NEGATIVE FREQUENCY",
                    "OPTIMIZATION NOT CONVERGED"
                ],
                'severity': 'medium',
                'recoverable': True,
                'description': 'Geometry optimization convergence problems',
                'recovery_hints': [
                    'Check initial geometry for reasonable bond lengths',
                    'Use smaller optimization steps (MAXTRADIUS)',
                    'Try different optimization algorithm',
                    'Manually adjust problematic atoms',
                    'Use constraints for problematic coordinates'
                ]
            },
            'basis_linear_dependence': {
                'patterns': [
                    "**** NEIGHB ****", "ATOMS TOO CLOSE", "SMALL DISTANCE BETWEEN ATOMS"
                ],
                'severity': 'high',
                'recoverable': True,
                'description': 'Basis set linear dependence from close atomic distances',
                'recovery_hints': [
                    'Scale unit cell to separate close atoms',
                    'Use EIGS keyword to diagnose overlap matrix',
                    'Remove diffuse basis functions',
                    'Increase basis set contraction',
                    'Consider different basis set choice'
                ]
            },
            'shrink_error': {
                'patterns': [
                    "ANISOTROPIC SHRINKING FACTOR", "SHRINK FACTOR TOO SMALL",
                    "TOO SMALL SHRINK FACTOR"
                ],
                'severity': 'medium',
                'recoverable': True,
                'description': 'K-point mesh shrinking factor issues',
                'recovery_hints': [
                    'Increase shrinking factors',
                    'Use isotropic k-point mesh',
                    'Check cell parameters for reasonableness',
                    'Use automatic k-point generation'
                ]
            },
            'basis_error': {
                'patterns': [
                    "BASIS SET LINEARLY DEPENDENT", "LINEAR DEPENDENCE",
                    "BASIS SET ERROR", "GHOST BASIS OVERLAP"
                ],
                'severity': 'high',
                'recoverable': True,
                'description': 'Basis set linear dependence or errors',
                'recovery_hints': [
                    'Use different basis set',
                    'Remove redundant basis functions',
                    'Check for ghost atoms',
                    'Use basis set optimization tools'
                ]
            },
            'system_error': {
                'patterns': [
                    "=   bad termination of", "abort(1) on node", "srun: error:",
                    "slurmstepd: error: ***", "forrtl: error (78):",
                    "CRYSTAL STOPS", "FORTRAN STOP"
                ],
                'severity': 'high',
                'recoverable': False,
                'description': 'System or runtime error',
                'recovery_hints': [
                    'Check system resources and health',
                    'Verify input file format',
                    'Try running on different node',
                    'Contact system administrator',
                    'Check for corrupted files'
                ]
            },
            'io_error': {
                'patterns': [
                    "I/O ERROR", "PERMISSION DENIED", "FILE NOT FOUND",
                    "CANNOT OPEN FILE", "READ ERROR", "WRITE ERROR"
                ],
                'severity': 'medium',
                'recoverable': True,
                'description': 'Input/output file errors',
                'recovery_hints': [
                    'Check file permissions',
                    'Verify file paths exist',
                    'Check disk space',
                    'Ensure files are not corrupted',
                    'Use absolute paths if needed'
                ]
            }
        }
        
        # Completion patterns from updatelists2.py
        self.completion_patterns = {
            'optimization_complete': {
                'patterns': ["OPT END", "OPTIMIZATION CONVERGED"],
                'calc_type': 'OPT'
            },
            'single_point_complete': {
                'patterns': ["    TOTAL CPU TIME =", "CRYSTAL ENDS"],
                'calc_type': 'SP'
            },
            'frequency_complete': {
                'patterns': ["FREQUENCY CALCULATION", "VIBRATIONAL FREQUENCIES"],
                'calc_type': 'FREQ'
            },
            'band_complete': {
                'patterns': ["BAND STRUCTURE", "BAND CALCULATION COMPLETED"],
                'calc_type': 'BAND'
            },
            'dos_complete': {
                'patterns': ["DENSITY OF STATES", "DOS CALCULATION COMPLETED"],
                'calc_type': 'DOSS'
            },
            'transport_complete': {
                'patterns': ["TRANSPORT PROPERTIES", "BOLTZTRA CALCULATION", "SEEBECK COEFFICIENT"],
                'calc_type': 'TRANSPORT'
            },
            'charge_potential_complete': {
                'patterns': ["CHARGE DENSITY", "ELECTROSTATIC POTENTIAL", "ECHG CALCULATION", "POTC CALCULATION"],
                'calc_type': 'CHARGE+POTENTIAL'
            }
        }
        
    def analyze_output_file(self, output_file: Path) -> Dict[str, any]:
        """
        Analyze a single CRYSTAL output file for errors and completion status.
        
        Args:
            output_file: Path to .out file to analyze
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'file': str(output_file),
            'analyzed_at': datetime.now().isoformat(),
            'status': 'unknown',
            'error_type': None,
            'error_details': [],
            'completion_type': None,
            'calc_type': None,
            'recoverable': None,
            'recovery_hints': [],
            'file_size': 0,
            'last_modified': None,
            'runtime_info': {},
            'performance_metrics': {}
        }
        
        if not output_file.exists():
            result['status'] = 'file_not_found'
            return result
            
        try:
            stat = output_file.stat()
            result['file_size'] = stat.st_size
            result['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Read file content
            with open(output_file, 'r', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
        except Exception as e:
            result['status'] = 'read_error'
            result['error_details'].append(f"Could not read file: {e}")
            return result
            
        # Extract runtime and performance information
        self._extract_runtime_info(lines, result)
        
        # Check for errors first (following updatelists2.py logic)
        error_found = self._check_for_errors(lines, result)
        
        if not error_found:
            # Check for completion patterns
            self._check_for_completion(lines, result)
            
        # Additional analysis
        self._analyze_performance_issues(lines, result)
        self._extract_calculation_details(lines, result)
        
        return result
        
    def _check_for_errors(self, lines: List[str], result: Dict) -> bool:
        """Check for error patterns in output file lines."""
        for line in lines:
            line_lower = line.lower()
            
            for error_type, error_info in self.error_patterns.items():
                for pattern in error_info['patterns']:
                    if pattern.lower() in line_lower:
                        result['status'] = 'error'
                        result['error_type'] = error_type
                        result['error_details'].append({
                            'pattern': pattern,
                            'line': line.strip(),
                            'severity': error_info['severity'],
                            'description': error_info['description']
                        })
                        result['recoverable'] = error_info['recoverable']
                        result['recovery_hints'] = error_info['recovery_hints']
                        return True
                        
        # Check for generic errors
        for line in lines:
            if 'error' in line.lower() and not any(skip in line.lower() 
                                                  for skip in ['no error', 'error correction', 'stderr']):
                result['status'] = 'unknown_error'
                result['error_details'].append({
                    'pattern': 'generic_error',
                    'line': line.strip(),
                    'severity': 'unknown',
                    'description': 'Unclassified error message'
                })
                return True
                
        return False
        
    def _check_for_completion(self, lines: List[str], result: Dict):
        """Check for completion patterns in output file lines."""
        for completion_type, completion_info in self.completion_patterns.items():
            for pattern in completion_info['patterns']:
                if any(pattern in line for line in lines):
                    result['status'] = 'completed'
                    result['completion_type'] = completion_type
                    result['calc_type'] = completion_info['calc_type']
                    return
                    
        # If no completion found, check if calculation is still running
        if any('CRYSTAL' in line and 'CALCULATION' in line for line in lines[:20]):
            result['status'] = 'ongoing'
        else:
            result['status'] = 'incomplete'
            
    def _extract_runtime_info(self, lines: List[str], result: Dict):
        """Extract runtime and timing information from output."""
        runtime_info = {}
        
        for line in lines:
            # Extract CPU time
            if "TOTAL CPU TIME =" in line:
                try:
                    time_match = re.search(r'(\d+\.\d+)', line)
                    if time_match:
                        runtime_info['total_cpu_time'] = float(time_match.group(1))
                except:
                    pass
                    
            # Extract wall time
            elif "ELAPSED TIME =" in line:
                try:
                    time_match = re.search(r'(\d+\.\d+)', line)
                    if time_match:
                        runtime_info['wall_time'] = float(time_match.group(1))
                except:
                    pass
                    
            # Extract SCF cycles
            elif "SCF CYCLE" in line:
                try:
                    cycle_match = re.search(r'CYCLE\s+(\d+)', line)
                    if cycle_match:
                        runtime_info['scf_cycles'] = int(cycle_match.group(1))
                except:
                    pass
                    
            # Extract memory usage
            elif "MEMORY" in line and "MB" in line:
                try:
                    mem_match = re.search(r'(\d+)\s*MB', line)
                    if mem_match:
                        runtime_info['memory_mb'] = int(mem_match.group(1))
                except:
                    pass
                    
        result['runtime_info'] = runtime_info
        
    def _analyze_performance_issues(self, lines: List[str], result: Dict):
        """Analyze performance-related issues and bottlenecks."""
        performance = {}
        
        # Check for slow convergence
        scf_cycles = result['runtime_info'].get('scf_cycles', 0)
        if scf_cycles > 100:
            performance['slow_convergence'] = {
                'issue': 'High number of SCF cycles',
                'cycles': scf_cycles,
                'suggestion': 'Consider adjusting SCF parameters or initial guess'
            }
            
        # Check for excessive runtime
        cpu_time = result['runtime_info'].get('total_cpu_time', 0)
        if cpu_time > 24 * 3600:  # More than 24 hours
            performance['long_runtime'] = {
                'issue': 'Very long calculation time',
                'time_hours': cpu_time / 3600,
                'suggestion': 'Consider reducing computational complexity'
            }
            
        # Check for memory issues (not failures, but high usage)
        memory_mb = result['runtime_info'].get('memory_mb', 0)
        if memory_mb > 50000:  # More than 50GB
            performance['high_memory'] = {
                'issue': 'High memory usage',
                'memory_gb': memory_mb / 1024,
                'suggestion': 'Monitor memory usage and consider optimization'
            }
            
        result['performance_metrics'] = performance
        
    def _extract_calculation_details(self, lines: List[str], result: Dict):
        """Extract calculation-specific details from output."""
        details = {}
        
        # Extract geometry information
        for line in lines:
            if "PRIMITIVE CELL" in line and "VOLUME" in line:
                volume_match = re.search(r'VOLUME\s*=\s*(\d+\.\d+)', line)
                if volume_match:
                    details['cell_volume'] = float(volume_match.group(1))
                    
            elif "NUMBER OF ATOMS" in line:
                atom_match = re.search(r'(\d+)', line)
                if atom_match:
                    details['num_atoms'] = int(atom_match.group(1))
                    
            elif "SPACE GROUP" in line:
                sg_match = re.search(r'(\d+)', line)
                if sg_match:
                    details['space_group'] = int(sg_match.group(1))
                    
        result['calculation_details'] = details
        
    def run_updatelists_integration(self, directory: Path = None) -> Dict[str, any]:
        """
        Run analysis using existing updatelists2.py integration.
        
        Args:
            directory: Directory to analyze (default: base_dir)
            
        Returns:
            Dictionary with categorized results
        """
        if directory is None:
            directory = self.base_dir
            
        # Find updatelists2.py script
        updatelists_script = Path(__file__).parent.parent / "Check_Scripts" / "updatelists2.py"
        
        if not updatelists_script.exists():
            return {
                'error': f'updatelists2.py not found at {updatelists_script}',
                'categories': {}
            }
            
        results = {
            'analyzed_at': datetime.now().isoformat(),
            'directory': str(directory),
            'categories': {},
            'script_output': '',
            'csv_files': [],
            'errors': []
        }
        
        # Change to the target directory and run updatelists2.py
        original_cwd = os.getcwd()
        try:
            os.chdir(directory)
            
            # Run the script
            result = subprocess.run(
                ['python', str(updatelists_script)],
                capture_output=True, text=True, timeout=300
            )
            
            results['script_output'] = result.stdout
            
            if result.returncode != 0:
                results['errors'].append(f"Script failed with code {result.returncode}: {result.stderr}")
                return results
                
            # Parse the CSV files created by updatelists2.py
            csv_patterns = [
                'complete_list.csv', 'completesp_list.csv', 'ongoing_list.csv',
                'too_many_scf_list.csv', 'memory_list.csv', 'quota_list.csv',
                'time_list.csv', 'geometry_small_dist_list.csv', 'shrink_error_list.csv',
                'linear_basis_list.csv', 'potential_list.csv', 'unknown_list.csv'
            ]
            
            for csv_file in csv_patterns:
                csv_path = Path(csv_file)
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        category = csv_file.replace('_list.csv', '')
                        results['categories'][category] = df['data_files'].tolist()
                        results['csv_files'].append(str(csv_path))
                    except Exception as e:
                        results['errors'].append(f"Error reading {csv_file}: {e}")
                        
        except subprocess.TimeoutExpired:
            results['errors'].append("updatelists2.py timed out")
        except Exception as e:
            results['errors'].append(f"Error running updatelists2.py: {e}")
        finally:
            os.chdir(original_cwd)
            
        return results
        
    def generate_error_report(self, material_id: str = None, 
                            days_back: int = 7) -> Dict[str, any]:
        """
        Generate comprehensive error report for materials.
        
        Args:
            material_id: Specific material to analyze (None for all)
            days_back: Number of days to look back for recent errors
            
        Returns:
            Dictionary with error report
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days_back,
            'materials_analyzed': 0,
            'total_files_analyzed': 0,
            'error_summary': defaultdict(int),
            'recovery_recommendations': defaultdict(list),
            'material_details': {},
            'trending_errors': {},
            'system_health': {}
        }
        
        # Determine materials to analyze
        if material_id:
            material_dirs = [self.base_dir / material_id]
        else:
            material_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            
        all_errors = []
        
        for material_dir in material_dirs:
            if not material_dir.is_dir():
                continue
                
            mat_id = material_dir.name
            material_info = {
                'total_calculations': 0,
                'errors_found': 0,
                'error_types': defaultdict(int),
                'recent_errors': [],
                'oldest_error': None,
                'most_common_error': None
            }
            
            # Analyze all .out files in material directory
            out_files = list(material_dir.rglob("*.out"))
            
            for out_file in out_files:
                report['total_files_analyzed'] += 1
                material_info['total_calculations'] += 1
                
                # Check if file is recent enough
                if out_file.stat().st_mtime < cutoff_date.timestamp():
                    continue
                    
                analysis = self.analyze_output_file(out_file)
                
                if analysis['status'] == 'error':
                    material_info['errors_found'] += 1
                    error_type = analysis['error_type']
                    material_info['error_types'][error_type] += 1
                    
                    error_record = {
                        'file': str(out_file),
                        'error_type': error_type,
                        'timestamp': analysis['last_modified'],
                        'recoverable': analysis['recoverable'],
                        'recovery_hints': analysis['recovery_hints']
                    }
                    
                    material_info['recent_errors'].append(error_record)
                    all_errors.append(error_record)
                    
                    # Update global counters
                    report['error_summary'][error_type] += 1
                    
                    # Collect recovery recommendations
                    if analysis['recoverable']:
                        for hint in analysis['recovery_hints']:
                            if hint not in report['recovery_recommendations'][error_type]:
                                report['recovery_recommendations'][error_type].append(hint)
                                
            # Calculate material-specific statistics
            if material_info['error_types']:
                most_common = max(material_info['error_types'].items(), key=lambda x: x[1])
                material_info['most_common_error'] = most_common[0]
                
                # Find oldest error
                if material_info['recent_errors']:
                    oldest = min(material_info['recent_errors'], 
                               key=lambda x: x['timestamp'])
                    material_info['oldest_error'] = oldest['timestamp']
                    
            if material_info['total_calculations'] > 0:
                report['material_details'][mat_id] = material_info
                report['materials_analyzed'] += 1
                
        # Calculate trending errors (errors increasing over time)
        self._calculate_error_trends(all_errors, report)
        
        # Generate system health assessment
        self._assess_system_health(report)
        
        return report
        
    def _calculate_error_trends(self, all_errors: List[Dict], report: Dict):
        """Calculate trending errors over time."""
        from collections import Counter
        import pandas as pd
        
        if not all_errors:
            return
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_errors)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by day and error type
        daily_errors = df.groupby([df['timestamp'].dt.date, 'error_type']).size().reset_index(name='count')
        
        # Calculate trends (simple approach - comparing recent vs older periods)
        recent_date = daily_errors['timestamp'].max()
        cutoff = recent_date - timedelta(days=3)
        
        recent_errors = daily_errors[daily_errors['timestamp'] > cutoff]
        older_errors = daily_errors[daily_errors['timestamp'] <= cutoff]
        
        recent_counts = recent_errors.groupby('error_type')['count'].sum()
        older_counts = older_errors.groupby('error_type')['count'].sum()
        
        trending = {}
        for error_type in recent_counts.index:
            recent_count = recent_counts[error_type]
            older_count = older_counts.get(error_type, 0)
            
            if older_count > 0:
                trend = (recent_count - older_count) / older_count * 100
                if abs(trend) > 20:  # Only report significant trends
                    trending[error_type] = {
                        'trend_percent': trend,
                        'recent_count': recent_count,
                        'older_count': older_count,
                        'direction': 'increasing' if trend > 0 else 'decreasing'
                    }
                    
        report['trending_errors'] = trending
        
    def _assess_system_health(self, report: Dict):
        """Assess overall system health based on error patterns."""
        health = {
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        total_errors = sum(report['error_summary'].values())
        total_files = report['total_files_analyzed']
        
        if total_files > 0:
            error_rate = total_errors / total_files * 100
            
            if error_rate > 50:
                health['overall_status'] = 'critical'
                health['issues'].append(f'High error rate: {error_rate:.1f}%')
                health['recommendations'].append('Investigate system configuration and resources')
            elif error_rate > 25:
                health['overall_status'] = 'warning'
                health['issues'].append(f'Elevated error rate: {error_rate:.1f}%')
                health['recommendations'].append('Monitor system performance and resource usage')
                
        # Check for specific problematic patterns
        if report['error_summary'].get('memory_error', 0) > 5:
            health['issues'].append('Multiple memory errors detected')
            health['recommendations'].append('Review memory allocation and job requirements')
            
        if report['error_summary'].get('system_error', 0) > 3:
            health['issues'].append('System errors detected')
            health['recommendations'].append('Contact system administrator')
            
        # Check trending errors
        for error_type, trend_info in report['trending_errors'].items():
            if trend_info['direction'] == 'increasing' and trend_info['trend_percent'] > 50:
                health['issues'].append(f'Increasing {error_type} errors')
                health['recommendations'].append(f'Address {error_type} error causes proactively')
                
        report['system_health'] = health
        
    def suggest_recovery_actions(self, analysis_result: Dict) -> List[Dict]:
        """
        Suggest specific recovery actions based on error analysis.
        
        Args:
            analysis_result: Result from analyze_output_file()
            
        Returns:
            List of recovery action dictionaries
        """
        if analysis_result['status'] != 'error' or not analysis_result['recoverable']:
            return []
            
        error_type = analysis_result['error_type']
        actions = []
        
        # Generate specific actions based on error type and context
        if error_type == 'scf_convergence':
            runtime_info = analysis_result.get('runtime_info', {})
            cycles = runtime_info.get('scf_cycles', 0)
            
            if cycles > 200:
                actions.append({
                    'action': 'increase_maxcycle',
                    'description': 'Increase SCF MAXCYCLE significantly',
                    'priority': 'high',
                    'parameters': {'maxcycle': cycles * 2}
                })
            else:
                actions.append({
                    'action': 'adjust_scf_parameters',
                    'description': 'Adjust SCF mixing and method',
                    'priority': 'medium',
                    'parameters': {'fmixing': 10, 'method': 'ANDERSON'}
                })
                
        elif error_type == 'memory_error':
            runtime_info = analysis_result.get('runtime_info', {})
            memory_mb = runtime_info.get('memory_mb', 0)
            
            if memory_mb > 0:
                recommended_memory = int(memory_mb * 1.5)
                actions.append({
                    'action': 'increase_memory',
                    'description': f'Increase job memory allocation',
                    'priority': 'high',
                    'parameters': {'memory_gb': recommended_memory // 1024 + 1}
                })
            else:
                actions.append({
                    'action': 'optimize_memory_usage',
                    'description': 'Use more memory-efficient settings',
                    'priority': 'medium',
                    'parameters': {'direct_scf': True, 'smaller_basis': True}
                })
                
        elif error_type == 'time_limit':
            runtime_info = analysis_result.get('runtime_info', {})
            cpu_time = runtime_info.get('total_cpu_time', 0)
            
            if cpu_time > 0:
                recommended_time = int(cpu_time * 1.5 / 3600) + 1  # Convert to hours and add buffer
                actions.append({
                    'action': 'increase_walltime',
                    'description': f'Increase job wall time',
                    'priority': 'high',
                    'parameters': {'walltime_hours': recommended_time}
                })
                
        # Add general recovery hints as actions
        for hint in analysis_result.get('recovery_hints', []):
            actions.append({
                'action': 'manual_intervention',
                'description': hint,
                'priority': 'medium',
                'parameters': {}
            })
            
        return actions
        
    def update_database_with_errors(self, material_id: str = None) -> Dict[str, int]:
        """
        Update database with error information from analysis.
        
        Args:
            material_id: Specific material to update (None for all)
            
        Returns:
            Dictionary with update statistics
        """
        if not self.enable_tracking or not self.db:
            return {'error': 'Database tracking not enabled'}
            
        stats = {
            'calculations_updated': 0,
            'errors_recorded': 0,
            'materials_processed': 0
        }
        
        # Get materials to process
        if material_id:
            materials = [self.db.get_material(material_id)] if self.db.get_material(material_id) else []
        else:
            materials = self.db.get_materials_by_status('active')
            
        for material in materials:
            if not material:
                continue
                
            mat_id = material['material_id']
            stats['materials_processed'] += 1
            
            # Get calculations for this material
            calculations = self.db.get_calculations_by_status(material_id=mat_id)
            
            for calc in calculations:
                if calc['status'] not in ['failed', 'error']:
                    continue
                    
                output_file = calc.get('output_file')
                if not output_file or not Path(output_file).exists():
                    continue
                    
                # Analyze the output file
                analysis = self.analyze_output_file(Path(output_file))
                
                if analysis['status'] == 'error':
                    # Update calculation with error information
                    self.db.update_calculation_status(
                        calc['calc_id'], 'failed',
                        error_type=analysis['error_type'],
                        error_message=f"Error: {analysis['error_type']} - {analysis['error_details'][0].get('description', 'Unknown error') if analysis['error_details'] else 'No details'}"
                    )
                    
                    stats['calculations_updated'] += 1
                    stats['errors_recorded'] += 1
                    
        return stats


def main():
    """Main function for CLI operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL Error Detection and Analysis")
    parser.add_argument("--base-dir", default=".", help="Base directory for analysis")
    parser.add_argument("--material-id", help="Specific material ID to analyze")
    parser.add_argument("--action", choices=['analyze', 'report', 'updatelists', 'recovery'], 
                       default='analyze', help="Action to perform")
    parser.add_argument("--output-file", help="Output file for analysis")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for error analysis")
    
    args = parser.parse_args()
    
    # Create error detector
    detector = CrystalErrorDetector(args.base_dir, args.db_path)
    
    if args.action == 'analyze':
        if args.output_file:
            print(f"Analyzing output file: {args.output_file}")
            result = detector.analyze_output_file(Path(args.output_file))
            
            print(f"\nAnalysis Result:")
            print(f"Status: {result['status']}")
            if result['error_type']:
                print(f"Error Type: {result['error_type']}")
                print(f"Recoverable: {result['recoverable']}")
                if result['recovery_hints']:
                    print("Recovery Hints:")
                    for hint in result['recovery_hints']:
                        print(f"  - {hint}")
        else:
            print("Output file required for analyze action")
            
    elif args.action == 'report':
        print("Generating error report...")
        report = detector.generate_error_report(args.material_id, args.days_back)
        
        print(f"\nError Report Generated at {report['generated_at']}")
        print(f"Period: {report['period_days']} days")
        print(f"Materials analyzed: {report['materials_analyzed']}")
        print(f"Files analyzed: {report['total_files_analyzed']}")
        
        if report['error_summary']:
            print(f"\nError Summary:")
            for error_type, count in report['error_summary'].items():
                print(f"  {error_type}: {count}")
                
        print(f"System Health: {report['system_health']['overall_status']}")
        
        # Save detailed report
        report_file = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved to {report_file}")
        
    elif args.action == 'updatelists':
        print("Running updatelists2.py integration...")
        result = detector.run_updatelists_integration()
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Analysis completed for {result['directory']}")
            print(f"Categories found: {len(result['categories'])}")
            for category, files in result['categories'].items():
                print(f"  {category}: {len(files)} files")
                
    elif args.action == 'recovery':
        if args.output_file:
            print(f"Generating recovery suggestions for: {args.output_file}")
            analysis = detector.analyze_output_file(Path(args.output_file))
            suggestions = detector.suggest_recovery_actions(analysis)
            
            if suggestions:
                print("\nRecovery Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion['description']} (Priority: {suggestion['priority']})")
                    if suggestion['parameters']:
                        print(f"   Parameters: {suggestion['parameters']}")
            else:
                print("No recovery suggestions available")
        else:
            print("Output file required for recovery action")


if __name__ == "__main__":
    main()