#!/usr/bin/env python3
"""
Crystal File Manager for Material Tracking System
-------------------------------------------------
Manages organized directory structure and file operations for CRYSTAL calculations.
Integrates with existing check_completedV2.py and check_erroredV2.py scripts.

Key Features:
- Organized directory structure by material ID and calculation type
- Integration with existing file checking scripts
- Automatic file discovery and cataloging
- File integrity checking and validation
- Cleanup and archival operations

Author: Based on implementation plan for material tracking system
"""

import os
import shutil
import json
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import threading
import subprocess
import glob
import re

# Import our material database
from material_database import MaterialDatabase, create_material_id_from_file, extract_formula_from_d12


class CrystalFileManager:
    """
    Manages file organization and operations for CRYSTAL calculations.
    
    Provides organized directory structure, file discovery, integrity checking,
    and integration with existing analysis scripts.
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
            
        # File type patterns for automatic discovery
        self.file_patterns = {
            'input': ['*.d12', '*.inp'],
            'output': ['*.out', '*.outp'],
            'log': ['*.log'],
            'error': ['*.err'],
            'wavefunction': ['*.f9', 'fort.9', '*.f98', 'fort.98'],
            'density': ['*.f25', 'fort.25'],
            'property': ['*.dat', '*.csv', '*.properties'],
            'plot': ['*.png', '*.pdf', '*.ps', '*.eps'],
            'backup': ['*.bak', '*.backup'],
            'temporary': ['*.tmp', '*.temp', 'fort.*']
        }
        
        # Known CRYSTAL output file extensions
        self.crystal_outputs = {
            'f9': 'Wavefunction file',
            'f25': 'Density and potential file', 
            'f98': 'Formatted wavefunction',
            'f80': 'Band structure file',
            'f25': 'Electron density file',
            'BAND.DAT': 'Band structure data',
            'DOSS.DAT': 'Density of states data',
            'FREQINFO.DAT': 'Frequency information',
            'OPTINFO.DAT': 'Optimization information'
        }
        
    def create_material_directory_structure(self, material_id: str) -> Path:
        """
        Create organized directory structure for a material.
        
        Structure: base_dir/material_id/
                  ├── opt/          # Geometry optimization
                  ├── sp/           # Single point calculations  
                  ├── band/         # Band structure calculations
                  ├── doss/         # Density of states
                  ├── freq/         # Frequency calculations
                  ├── transport/    # Transport properties
                  ├── analysis/     # Analysis results
                  └── archive/      # Archived files
                  
        Args:
            material_id: Unique material identifier
            
        Returns:
            Path to the material directory
        """
        material_dir = self.base_dir / material_id
        
        # Standard calculation type directories
        calc_dirs = ['opt', 'sp', 'band', 'doss', 'freq', 'transport', 'analysis', 'archive']
        
        for calc_type in calc_dirs:
            calc_dir = material_dir / calc_type
            calc_dir.mkdir(parents=True, exist_ok=True)
            
        return material_dir
        
    def organize_calculation_files(self, material_id: str, calc_type: str, 
                                 source_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Organize calculation files into appropriate directory structure.
        
        Args:
            material_id: Material identifier
            calc_type: Type of calculation (opt, sp, band, etc.)
            source_files: List of files to organize
            
        Returns:
            Dictionary mapping file types to organized file paths
        """
        calc_dir = self.base_dir / material_id / calc_type.lower()
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        organized_files = {
            'input': [],
            'output': [], 
            'property': [],
            'plot': [],
            'log': [],
            'error': [],
            'other': []
        }
        
        for source_file in source_files:
            if not source_file.exists():
                continue
                
            file_type = self._classify_file(source_file)
            
            # Generate organized filename
            if file_type in ['input', 'output']:
                # Use standard naming: material_id_calc_type.ext
                if file_type == 'input':
                    target_name = f"{material_id}_{calc_type.lower()}.d12"
                else:
                    target_name = f"{material_id}_{calc_type.lower()}.out"
            else:
                # Keep original name but add material prefix if not present
                original_name = source_file.name
                if not original_name.startswith(material_id):
                    target_name = f"{material_id}_{calc_type.lower()}_{original_name}"
                else:
                    target_name = original_name
                    
            target_path = calc_dir / target_name
            
            # Copy or move file
            try:
                if source_file.parent != calc_dir:
                    shutil.copy2(source_file, target_path)
                    print(f"Organized: {source_file} -> {target_path}")
                else:
                    target_path = source_file  # Already in correct location
                    
                organized_files[file_type].append(target_path)
                
                # Update database file records if tracking enabled
                if self.enable_tracking and self.db:
                    # Find calculation record
                    calcs = self.db.get_calculations_by_status(
                        material_id=material_id, calc_type=calc_type.upper()
                    )
                    if calcs:
                        calc_id = calcs[0]['calc_id']  # Use most recent
                        self.db.add_file_record(
                            calc_id=calc_id,
                            file_type=file_type,
                            file_name=target_path.name,
                            file_path=str(target_path)
                        )
                        
            except Exception as e:
                print(f"Error organizing file {source_file}: {e}")
                organized_files['other'].append(source_file)
                
        return organized_files
        
    def _classify_file(self, file_path: Path) -> str:
        """Classify file type based on extension and content."""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        # Check by extension first
        if suffix in ['.d12', '.inp']:
            return 'input'
        elif suffix in ['.out', '.outp']:
            return 'output'
        elif suffix in ['.log']:
            return 'log'
        elif suffix in ['.err']:
            return 'error'
        elif suffix in ['.dat', '.csv']:
            return 'property'
        elif suffix in ['.png', '.pdf', '.ps', '.eps']:
            return 'plot'
        elif suffix in ['.f9', '.f25', '.f98', '.f80']:
            return 'property'
        elif name.startswith('fort.'):
            return 'property'
        else:
            return 'other'
            
    def discover_material_files(self, search_dir: Path = None) -> Dict[str, List[Path]]:
        """
        Discover CRYSTAL files in directory and group by potential material.
        
        Args:
            search_dir: Directory to search (default: base_dir)
            
        Returns:
            Dictionary mapping material_id to list of associated files
        """
        if search_dir is None:
            search_dir = self.base_dir
            
        material_files = {}
        
        # Find all potential CRYSTAL files
        all_files = []
        for pattern_type, patterns in self.file_patterns.items():
            for pattern in patterns:
                all_files.extend(search_dir.glob(pattern))
                
        # Group files by material ID
        for file_path in all_files:
            try:
                # Try to extract material ID from filename or content
                if file_path.suffix.lower() == '.d12':
                    material_id = create_material_id_from_file(file_path)
                else:
                    # Try to infer from filename
                    material_id = self._infer_material_id_from_filename(file_path)
                    
                if material_id:
                    if material_id not in material_files:
                        material_files[material_id] = []
                    material_files[material_id].append(file_path)
                    
            except Exception as e:
                print(f"Warning: Could not classify file {file_path}: {e}")
                
        return material_files
        
    def _infer_material_id_from_filename(self, file_path: Path) -> Optional[str]:
        """Infer material ID from filename patterns."""
        name = file_path.stem
        
        # Remove common calculation type suffixes
        for suffix in ['_opt', '_sp', '_band', '_doss', '_freq', '_transport']:
            name = name.replace(suffix, '')
            
        # Remove common functional suffixes  
        for suffix in ['_pbe', '_hse06', '_b3lyp', '_pbe0']:
            name = name.replace(suffix, '')
            
        # Remove basis set suffixes
        for suffix in ['_dzvp', '_tzvp', '_pob']:
            name = name.replace(suffix, '')
            
        return name if name else None
        
    def check_file_integrity(self, file_path: Path) -> Dict[str, any]:
        """
        Check file integrity and extract basic information.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Dictionary with integrity information
        """
        result = {
            'exists': file_path.exists(),
            'size': 0,
            'checksum': None,
            'readable': False,
            'last_modified': None,
            'file_type': None,
            'errors': []
        }
        
        if not result['exists']:
            result['errors'].append('File does not exist')
            return result
            
        try:
            stat = file_path.stat()
            result['size'] = stat.st_size
            result['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            result['file_type'] = self._classify_file(file_path)
            
            # Check if file is readable
            try:
                with open(file_path, 'r') as f:
                    f.read(1)  # Try to read first character
                result['readable'] = True
            except:
                result['readable'] = False
                result['errors'].append('File is not readable')
                
            # Calculate checksum for small files
            if result['size'] < 100 * 1024 * 1024:  # Less than 100MB
                try:
                    result['checksum'] = self._calculate_checksum(file_path)
                except Exception as e:
                    result['errors'].append(f'Could not calculate checksum: {e}')
                    
            # Check for zero-size files
            if result['size'] == 0:
                result['errors'].append('File is empty')
                
            # Check for very old files (potential orphans)
            age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            if age > timedelta(days=30):
                result['warnings'] = result.get('warnings', [])
                result['warnings'].append(f'File is {age.days} days old')
                
        except Exception as e:
            result['errors'].append(f'Error checking file: {e}')
            
        return result
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def integrate_with_check_scripts(self, material_id: str = None) -> Dict[str, any]:
        """
        Integrate with existing check_completedV2.py and check_erroredV2.py scripts.
        
        Args:
            material_id: Specific material to check (None for all)
            
        Returns:
            Dictionary with check results
        """
        results = {
            'completed': [],
            'errored': [],
            'script_errors': []
        }
        
        # Find the check scripts
        script_dir = Path(__file__).parent.parent / "Check_Scripts"
        completed_script = script_dir / "check_completedV2.py"
        errored_script = script_dir / "check_erroredV2.py"
        
        if not completed_script.exists():
            results['script_errors'].append(f"check_completedV2.py not found at {completed_script}")
            
        if not errored_script.exists():
            results['script_errors'].append(f"check_erroredV2.py not found at {errored_script}")
            
        if results['script_errors']:
            return results
            
        # Determine directories to check
        if material_id:
            check_dirs = [self.base_dir / material_id]
        else:
            # Find all material directories
            check_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            
        for check_dir in check_dirs:
            if not check_dir.exists():
                continue
                
            # Run check_completedV2.py
            try:
                result = subprocess.run(
                    ['python', str(completed_script), str(check_dir)],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    # Parse output for completed calculations
                    completed_files = self._parse_check_script_output(result.stdout, 'completed')
                    results['completed'].extend(completed_files)
                else:
                    results['script_errors'].append(f"check_completedV2.py failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                results['script_errors'].append("check_completedV2.py timed out")
            except Exception as e:
                results['script_errors'].append(f"Error running check_completedV2.py: {e}")
                
            # Run check_erroredV2.py
            try:
                result = subprocess.run(
                    ['python', str(errored_script), str(check_dir)],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    # Parse output for errored calculations
                    errored_files = self._parse_check_script_output(result.stdout, 'errored')
                    results['errored'].extend(errored_files)
                else:
                    results['script_errors'].append(f"check_erroredV2.py failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                results['script_errors'].append("check_erroredV2.py timed out")
            except Exception as e:
                results['script_errors'].append(f"Error running check_erroredV2.py: {e}")
                
        return results
        
    def _parse_check_script_output(self, output: str, script_type: str) -> List[Dict]:
        """Parse output from check scripts to extract file information."""
        files = []
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Look for file paths in the output
            if '.out' in line or '.d12' in line:
                # Try to extract file path and any status information
                parts = line.split()
                for part in parts:
                    if part.endswith('.out') or part.endswith('.d12'):
                        file_path = Path(part)
                        if file_path.exists():
                            files.append({
                                'file_path': str(file_path),
                                'status': script_type,
                                'detected_at': datetime.now().isoformat()
                            })
                            
        return files
        
    def cleanup_old_files(self, days_old: int = 30, file_types: List[str] = None) -> Dict[str, int]:
        """
        Clean up old files based on age and type.
        
        Args:
            days_old: Remove files older than this many days
            file_types: List of file types to clean up (None for all temporary types)
            
        Returns:
            Dictionary with cleanup statistics
        """
        if file_types is None:
            file_types = ['temporary', 'backup', 'log']
            
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleanup_stats = {
            'files_removed': 0,
            'files_archived': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        # Find files to clean up
        for material_dir in self.base_dir.iterdir():
            if not material_dir.is_dir():
                continue
                
            for calc_dir in material_dir.iterdir():
                if not calc_dir.is_dir():
                    continue
                    
                # Find old files of specified types
                for file_path in calc_dir.iterdir():
                    if not file_path.is_file():
                        continue
                        
                    file_type = self._classify_file(file_path)
                    if file_type not in file_types:
                        continue
                        
                    try:
                        file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_age < cutoff_date:
                            file_size = file_path.stat().st_size
                            
                            # Archive important files, delete temporary ones
                            if file_type in ['temporary']:
                                file_path.unlink()
                                cleanup_stats['files_removed'] += 1
                            else:
                                # Move to archive directory
                                archive_dir = material_dir / 'archive'
                                archive_dir.mkdir(exist_ok=True)
                                archive_path = archive_dir / file_path.name
                                shutil.move(str(file_path), str(archive_path))
                                cleanup_stats['files_archived'] += 1
                                
                            cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
                            
                    except Exception as e:
                        cleanup_stats['errors'].append(f"Error cleaning {file_path}: {e}")
                        
        return cleanup_stats
        
    def generate_file_report(self, material_id: str = None) -> Dict[str, any]:
        """
        Generate comprehensive file report for materials.
        
        Args:
            material_id: Specific material to report on (None for all)
            
        Returns:
            Dictionary with file report information
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'materials': {},
            'summary': {
                'total_materials': 0,
                'total_files': 0,
                'total_size_mb': 0,
                'file_types': {},
                'integrity_issues': 0
            }
        }
        
        # Determine materials to report on
        if material_id:
            material_dirs = [self.base_dir / material_id]
        else:
            material_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            
        for material_dir in material_dirs:
            if not material_dir.is_dir():
                continue
                
            mat_id = material_dir.name
            material_info = {
                'path': str(material_dir),
                'calculation_types': {},
                'total_files': 0,
                'total_size_mb': 0,
                'last_activity': None,
                'integrity_issues': []
            }
            
            # Check each calculation type directory
            for calc_dir in material_dir.iterdir():
                if not calc_dir.is_dir():
                    continue
                    
                calc_type = calc_dir.name
                calc_info = {
                    'files': [],
                    'file_count': 0,
                    'size_mb': 0
                }
                
                # Check all files in calculation directory
                for file_path in calc_dir.iterdir():
                    if not file_path.is_file():
                        continue
                        
                    integrity = self.check_file_integrity(file_path)
                    file_info = {
                        'name': file_path.name,
                        'type': integrity['file_type'],
                        'size_mb': integrity['size'] / (1024 * 1024),
                        'last_modified': integrity['last_modified'],
                        'integrity_ok': len(integrity['errors']) == 0
                    }
                    
                    if not file_info['integrity_ok']:
                        material_info['integrity_issues'].extend(integrity['errors'])
                        report['summary']['integrity_issues'] += 1
                        
                    calc_info['files'].append(file_info)
                    calc_info['file_count'] += 1
                    calc_info['size_mb'] += file_info['size_mb']
                    
                    # Update summary
                    report['summary']['total_files'] += 1
                    report['summary']['total_size_mb'] += file_info['size_mb']
                    
                    file_type = file_info['type']
                    if file_type not in report['summary']['file_types']:
                        report['summary']['file_types'][file_type] = 0
                    report['summary']['file_types'][file_type] += 1
                    
                    # Track most recent activity
                    if integrity['last_modified']:
                        mod_time = integrity['last_modified']
                        if (material_info['last_activity'] is None or 
                            mod_time > material_info['last_activity']):
                            material_info['last_activity'] = mod_time
                            
                material_info['calculation_types'][calc_type] = calc_info
                material_info['total_files'] += calc_info['file_count']
                material_info['total_size_mb'] += calc_info['size_mb']
                
            if material_info['total_files'] > 0:
                report['materials'][mat_id] = material_info
                report['summary']['total_materials'] += 1
                
        return report
        
    def sync_with_database(self) -> Dict[str, int]:
        """
        Synchronize file system state with database records.
        
        Returns:
            Dictionary with sync statistics
        """
        if not self.enable_tracking or not self.db:
            return {'error': 'Database tracking not enabled'}
            
        stats = {
            'files_added': 0,
            'files_updated': 0,
            'files_removed': 0,
            'discrepancies': 0
        }
        
        # Get all materials from database
        materials = self.db.get_materials_by_status('active')
        
        for material in materials:
            material_id = material['material_id']
            material_dir = self.base_dir / material_id
            
            if not material_dir.exists():
                continue
                
            # Get calculations for this material
            calculations = self.db.get_calculations_by_status(material_id=material_id)
            
            for calc in calculations:
                calc_id = calc['calc_id']
                calc_type = calc['calc_type']
                work_dir = Path(calc['work_dir']) if calc['work_dir'] else material_dir / calc_type.lower()
                
                if not work_dir.exists():
                    continue
                    
                # Check all files in the calculation directory
                for file_path in work_dir.iterdir():
                    if not file_path.is_file():
                        continue
                        
                    # Check if file is already in database
                    # (This would require a get_files_by_calc_id method in the database)
                    # For now, just add the file record
                    file_type = self._classify_file(file_path)
                    self.db.add_file_record(
                        calc_id=calc_id,
                        file_type=file_type,
                        file_name=file_path.name,
                        file_path=str(file_path)
                    )
                    stats['files_added'] += 1
                    
        return stats


def main():
    """Main function for testing and CLI operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL File Manager")
    parser.add_argument("--base-dir", default=".", help="Base directory for file operations")
    parser.add_argument("--material-id", help="Specific material ID to work with")
    parser.add_argument("--action", choices=['discover', 'organize', 'check', 'report', 'cleanup'], 
                       default='discover', help="Action to perform")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    
    args = parser.parse_args()
    
    # Create file manager
    manager = CrystalFileManager(args.base_dir, args.db_path)
    
    if args.action == 'discover':
        print("Discovering CRYSTAL files...")
        material_files = manager.discover_material_files()
        
        print(f"\nFound files for {len(material_files)} materials:")
        for material_id, files in material_files.items():
            print(f"\n{material_id}: {len(files)} files")
            for file_path in files[:5]:  # Show first 5 files
                print(f"  {file_path}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
                
    elif args.action == 'organize':
        if not args.material_id:
            print("Material ID required for organize action")
            return
            
        print(f"Organizing files for material {args.material_id}...")
        # This would need specific source files to organize
        print("Note: organize action requires source files to be specified")
        
    elif args.action == 'check':
        print("Running integration with check scripts...")
        results = manager.integrate_with_check_scripts(args.material_id)
        
        print(f"\nCompleted calculations: {len(results['completed'])}")
        print(f"Errored calculations: {len(results['errored'])}")
        if results['script_errors']:
            print(f"Script errors: {results['script_errors']}")
            
    elif args.action == 'report':
        print("Generating file report...")
        report = manager.generate_file_report(args.material_id)
        
        print(f"\nFile Report Generated at {report['generated_at']}")
        print(f"Total materials: {report['summary']['total_materials']}")
        print(f"Total files: {report['summary']['total_files']}")
        print(f"Total size: {report['summary']['total_size_mb']:.2f} MB")
        print(f"Integrity issues: {report['summary']['integrity_issues']}")
        
        # Save detailed report
        report_file = f"file_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to {report_file}")
        
    elif args.action == 'cleanup':
        print("Cleaning up old files...")
        stats = manager.cleanup_old_files()
        
        print(f"\nCleanup completed:")
        print(f"Files removed: {stats['files_removed']}")
        print(f"Files archived: {stats['files_archived']}")
        print(f"Space freed: {stats['space_freed_mb']:.2f} MB")
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")


if __name__ == "__main__":
    main()