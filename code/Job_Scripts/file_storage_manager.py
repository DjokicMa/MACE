#!/usr/bin/env python3
"""
File Storage Manager for CRYSTAL Calculations
=============================================
Comprehensive file storage system for preserving complete calculation provenance
including input files, settings, intermediate files, and output data.

This system addresses the user's requirement to store:
- D12/D3 input files with all settings
- Complete calculation provenance 
- Intermediate files (fort.9, fort.25, etc.)
- Configuration files and parameters
- Output files and extracted data

Author: Enhanced property extraction system
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)


class FileStorageManager:
    """
    Manages comprehensive file storage for CRYSTAL calculations.
    
    Features:
    - Preserves complete calculation provenance
    - Stores D12/D3 input files with settings extraction
    - Archives intermediate files (fort.9, fort.25, etc.)
    - Maintains file integrity with checksums
    - Organizes files by material and calculation type
    - Supports file retrieval and reconstruction
    """
    
    def __init__(self, db_path: str = "materials.db", storage_root: str = "calculation_storage"):
        self.db = MaterialDatabase(db_path)
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(exist_ok=True)
        
        # File type categorization
        self.file_types = {
            'input': ['.d12', '.d3', '.input'],
            'output': ['.out', '.output', '.log'],
            'binary': ['.f9', '.f25', '.fort.9', '.fort.25', '.wf', '.prop'],
            'property': ['.BAND', '.DOSS', '.OPTC', '.ELPH'],
            'script': ['.sh', '.slurm', '.job'],
            'plot': ['.png', '.pdf', '.eps', '.svg'],
            'data': ['.csv', '.json', '.yaml', '.xml'],
            'config': ['.conf', '.cfg', '.ini', '.param']
        }
        
        # Initialize storage structure
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage directory structure."""
        # Create main storage directories
        for subdir in ['materials', 'calculations', 'templates', 'archives']:
            (self.storage_root / subdir).mkdir(exist_ok=True)
        
        print(f"📁 File storage initialized at: {self.storage_root}")
    
    def store_calculation_files(self, calc_id: str, material_id: str, 
                               calc_type: str, source_directory: Path,
                               preserve_original: bool = True) -> Dict[str, Any]:
        """
        Store all files associated with a calculation.
        
        Args:
            calc_id: Unique calculation identifier
            material_id: Material identifier
            calc_type: Type of calculation (OPT, SP, BAND, etc.)
            source_directory: Directory containing calculation files
            preserve_original: Whether to keep original files
            
        Returns:
            Dictionary with storage information
        """
        print(f"📦 Storing files for calculation {calc_id} ({calc_type})")
        
        # Create calculation storage directory
        calc_storage_dir = self.storage_root / "calculations" / calc_id
        calc_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and categorize files
        files_found = self._find_calculation_files(source_directory, material_id, calc_type)
        
        storage_info = {
            'calc_id': calc_id,
            'material_id': material_id,
            'calc_type': calc_type,
            'storage_directory': str(calc_storage_dir),
            'files_stored': {},
            'settings_extracted': {},
            'storage_timestamp': datetime.now().isoformat()
        }
        
        # Store each file
        for file_path, file_info in files_found.items():
            stored_info = self._store_single_file(
                file_path, calc_storage_dir, file_info, 
                calc_id, preserve_original
            )
            
            if stored_info:
                storage_info['files_stored'][file_path.name] = stored_info
                
                # Extract settings from input files
                if file_info['category'] == 'input':
                    settings = self._extract_input_settings(file_path)
                    if settings:
                        storage_info['settings_extracted'][file_path.name] = settings
        
        # Store metadata
        metadata_file = calc_storage_dir / f"{calc_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(storage_info, f, indent=2)
        
        # Update database
        self._update_database_file_records(calc_id, storage_info)
        
        print(f"   ✅ Stored {len(storage_info['files_stored'])} files")
        return storage_info
    
    def _find_calculation_files(self, source_directory: Path, 
                               material_id: str, calc_type: str) -> Dict[Path, Dict[str, Any]]:
        """Find all files related to a calculation."""
        files_found = {}
        
        if not source_directory.exists():
            print(f"   ⚠️  Source directory not found: {source_directory}")
            return files_found
        
        # Look for files matching the material pattern
        patterns = [
            f"{material_id}.*",
            f"*{material_id}*",
            f"*.{calc_type.lower()}",
            f"*{calc_type.lower()}*"
        ]
        
        for pattern in patterns:
            for file_path in source_directory.glob(pattern):
                if file_path.is_file():
                    file_info = self._analyze_file(file_path, calc_type)
                    files_found[file_path] = file_info
        
        # Also look for common CRYSTAL files
        common_files = [
            'fort.9', 'fort.25', 'fort.20', 'fort.21', 'fort.32', 'fort.33',
            'INPUT', 'OUTPUT', 'crystal.out', 'properties.out'
        ]
        
        for filename in common_files:
            file_path = source_directory / filename
            if file_path.exists():
                file_info = self._analyze_file(file_path, calc_type)
                files_found[file_path] = file_info
        
        return files_found
    
    def _analyze_file(self, file_path: Path, calc_type: str) -> Dict[str, Any]:
        """Analyze a file to determine its type and importance."""
        file_info = {
            'original_path': str(file_path),
            'filename': file_path.name,
            'size': file_path.stat().st_size,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'category': 'other',
            'importance': 'medium',
            'checksum': self._calculate_checksum(file_path)
        }
        
        # Categorize by extension
        suffix = file_path.suffix.lower()
        for category, extensions in self.file_types.items():
            if suffix in extensions:
                file_info['category'] = category
                break
        
        # Determine importance
        if suffix in ['.d12', '.d3']:
            file_info['importance'] = 'critical'  # Input files are critical
        elif suffix in ['.out', '.output']:
            file_info['importance'] = 'critical'  # Output files are critical
        elif suffix in ['.f9', '.f25']:
            file_info['importance'] = 'high'  # Binary files are high importance
        elif 'fort.' in file_path.name:
            file_info['importance'] = 'high'  # CRYSTAL fort files
        elif suffix in ['.sh', '.slurm']:
            file_info['importance'] = 'medium'  # Scripts are medium importance
        
        return file_info
    
    def _store_single_file(self, source_path: Path, storage_dir: Path, 
                          file_info: Dict[str, Any], calc_id: str,
                          preserve_original: bool = True) -> Optional[Dict[str, Any]]:
        """Store a single file with metadata."""
        try:
            # Determine storage filename (preserve original name)
            storage_filename = file_info['filename']
            storage_path = storage_dir / storage_filename
            
            # Copy the file
            if preserve_original:
                shutil.copy2(source_path, storage_path)
            else:
                shutil.move(str(source_path), str(storage_path))
            
            # Verify checksum
            stored_checksum = self._calculate_checksum(storage_path)
            if stored_checksum != file_info['checksum']:
                print(f"   ⚠️  Checksum mismatch for {storage_filename}")
                return None
            
            stored_info = {
                'stored_path': str(storage_path),
                'stored_filename': storage_filename,
                'original_checksum': file_info['checksum'],
                'stored_checksum': stored_checksum,
                'storage_timestamp': datetime.now().isoformat(),
                'category': file_info['category'],
                'importance': file_info['importance'],
                'size': file_info['size']
            }
            
            return stored_info
            
        except Exception as e:
            print(f"   ❌ Error storing {source_path}: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return ""
    
    def _extract_input_settings(self, input_file: Path) -> Dict[str, Any]:
        """Extract calculation settings from D12/D3 input files."""
        settings = {
            'file_type': input_file.suffix,
            'extraction_timestamp': datetime.now().isoformat(),
            'crystal_keywords': [],
            'basis_set_info': {},
            'calculation_parameters': {},
            'geometry_info': {}
        }
        
        try:
            with open(input_file, 'r') as f:
                content = f.read()
        except:
            return settings
        
        # Extract CRYSTAL keywords
        crystal_keywords = [
            'CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE', 'EXTERNAL',
            'OPTGEOM', 'FREQCALC', 'SINGLEPOINT', 'RESTART',
            'DFT', 'HYBRID', 'EXCHANGE', 'CORRELAT', 'NONLOCAL',
            'SHRINK', 'TOLINTEG', 'TOLDEE', 'SCFDIR', 'FMIXING',
            'MAXCYCLE', 'ANDERSON', 'DIIS', 'BROYDEN'
        ]
        
        found_keywords = []
        for keyword in crystal_keywords:
            if keyword in content.upper():
                found_keywords.append(keyword)
        
        settings['crystal_keywords'] = found_keywords
        
        # Extract specific parameters
        parameter_patterns = {
            'shrink_factor': r'SHRINK\s+(\d+)\s+(\d+)',
            'tolinteg_values': r'TOLINTEG\s+([\d\s]+)',
            'toldee_value': r'TOLDEE\s+(\d+)',
            'maxcycle_value': r'MAXCYCLE\s+(\d+)',
            'exchange_functional': r'EXCHANGE\s+(\w+)',
            'correlation_functional': r'CORRELAT\s+(\w+)'
        }
        
        for param_name, pattern in parameter_patterns.items():
            match = re.search(pattern, content.upper())
            if match:
                settings['calculation_parameters'][param_name] = match.groups()
        
        # Extract basis set information
        if 'EXTERNAL' in content.upper():
            settings['basis_set_info']['type'] = 'external'
            # Look for basis set file references
            basis_match = re.search(r'EXTERNAL\s*\n\s*(\S+)', content, re.IGNORECASE)
            if basis_match:
                settings['basis_set_info']['file'] = basis_match.group(1)
        else:
            settings['basis_set_info']['type'] = 'internal'
        
        # Extract geometry information
        if 'OPTGEOM' in content.upper():
            settings['geometry_info']['optimization'] = True
            # Look for optimization parameters
            optgeom_match = re.search(r'OPTGEOM\s*\n((?:.*\n)*?)END', content, re.IGNORECASE)
            if optgeom_match:
                settings['geometry_info']['optimization_parameters'] = optgeom_match.group(1).strip()
        
        return settings
    
    def _update_database_file_records(self, calc_id: str, storage_info: Dict[str, Any]):
        """Update database with file storage information."""
        try:
            with self.db._get_connection() as conn:
                # Update calculation record with storage info
                conn.execute("""
                    UPDATE calculations 
                    SET settings_json = ?
                    WHERE calc_id = ?
                """, (json.dumps(storage_info['settings_extracted']), calc_id))
                
                # Insert file records
                for filename, file_info in storage_info['files_stored'].items():
                    conn.execute("""
                        INSERT OR REPLACE INTO files 
                        (calc_id, file_type, file_name, file_path, file_size, 
                         created_at, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        calc_id,
                        file_info['category'],
                        filename,
                        file_info['stored_path'],
                        file_info['size'],
                        file_info['storage_timestamp'],
                        file_info['stored_checksum']
                    ))
                
                print(f"   📊 Updated database records for {calc_id}")
                
        except Exception as e:
            print(f"   ⚠️  Error updating database: {e}")
    
    def retrieve_calculation_files(self, calc_id: str, target_directory: Path) -> bool:
        """Retrieve all files for a calculation to a target directory."""
        print(f"📂 Retrieving files for calculation {calc_id}")
        
        calc_storage_dir = self.storage_root / "calculations" / calc_id
        if not calc_storage_dir.exists():
            print(f"   ❌ Storage directory not found: {calc_storage_dir}")
            return False
        
        target_directory.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        files_copied = 0
        for file_path in calc_storage_dir.glob("*"):
            if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                target_file = target_directory / file_path.name
                shutil.copy2(file_path, target_file)
                files_copied += 1
        
        print(f"   ✅ Retrieved {files_copied} files to {target_directory}")
        return True
    
    def get_calculation_settings(self, calc_id: str) -> Dict[str, Any]:
        """Get extracted settings for a calculation."""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT settings_json FROM calculations WHERE calc_id = ?",
                    (calc_id,)
                )
                result = cursor.fetchone()
                
                if result and result[0]:
                    return json.loads(result[0])
                else:
                    return {}
        except:
            return {}
    
    def list_stored_files(self, calc_id: str) -> List[Dict[str, Any]]:
        """List all files stored for a calculation."""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT file_type, file_name, file_path, file_size, 
                           created_at, checksum
                    FROM files 
                    WHERE calc_id = ?
                    ORDER BY file_type, file_name
                """, (calc_id,))
                
                files = []
                for row in cursor:
                    files.append({
                        'file_type': row[0],
                        'file_name': row[1],
                        'file_path': row[2],
                        'file_size': row[3],
                        'created_at': row[4],
                        'checksum': row[5]
                    })
                
                return files
        except:
            return []
    
    def verify_file_integrity(self, calc_id: str) -> Dict[str, bool]:
        """Verify integrity of all stored files for a calculation."""
        print(f"🔍 Verifying file integrity for calculation {calc_id}")
        
        stored_files = self.list_stored_files(calc_id)
        integrity_status = {}
        
        for file_info in stored_files:
            file_path = Path(file_info['file_path'])
            filename = file_info['file_name']
            expected_checksum = file_info['checksum']
            
            if file_path.exists():
                current_checksum = self._calculate_checksum(file_path)
                integrity_status[filename] = (current_checksum == expected_checksum)
            else:
                integrity_status[filename] = False
        
        return integrity_status
    
    def cleanup_storage(self, older_than_days: int = 30, dry_run: bool = True):
        """Clean up old storage directories."""
        print(f"🧹 Cleaning up storage older than {older_than_days} days (dry_run={dry_run})")
        
        cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 3600)
        cleaned_count = 0
        
        for calc_dir in (self.storage_root / "calculations").iterdir():
            if calc_dir.is_dir():
                # Check if calculation is still active
                try:
                    metadata_file = calc_dir / f"{calc_dir.name}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        storage_timestamp = datetime.fromisoformat(metadata['storage_timestamp']).timestamp()
                        
                        if storage_timestamp < cutoff_date:
                            print(f"   📁 Would clean: {calc_dir.name}")
                            
                            if not dry_run:
                                shutil.rmtree(calc_dir)
                                cleaned_count += 1
                except:
                    continue
        
        print(f"   ✅ {'Would clean' if dry_run else 'Cleaned'} {cleaned_count} directories")


def main():
    """Main function for testing file storage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL File Storage Manager")
    parser.add_argument("--store", help="Store files from directory")
    parser.add_argument("--calc-id", help="Calculation ID")
    parser.add_argument("--material-id", help="Material ID")
    parser.add_argument("--calc-type", help="Calculation type")
    parser.add_argument("--retrieve", help="Retrieve files to directory")
    parser.add_argument("--list-files", action="store_true", help="List stored files")
    parser.add_argument("--verify", action="store_true", help="Verify file integrity")
    parser.add_argument("--settings", action="store_true", help="Show extracted settings")
    
    args = parser.parse_args()
    
    if not any([args.store, args.retrieve, args.list_files, args.verify, args.settings]):
        print("❌ Please specify an action (--store, --retrieve, --list-files, --verify, --settings)")
        return
    
    storage_manager = FileStorageManager()
    
    if args.store:
        if not all([args.calc_id, args.material_id, args.calc_type]):
            print("❌ --store requires --calc-id, --material-id, and --calc-type")
            return
        
        source_dir = Path(args.store)
        storage_info = storage_manager.store_calculation_files(
            args.calc_id, args.material_id, args.calc_type, source_dir
        )
        
        print(f"📦 Storage completed:")
        print(f"   Files stored: {len(storage_info['files_stored'])}")
        print(f"   Settings extracted: {len(storage_info['settings_extracted'])}")
    
    elif args.retrieve:
        if not args.calc_id:
            print("❌ --retrieve requires --calc-id")
            return
        
        target_dir = Path(args.retrieve)
        success = storage_manager.retrieve_calculation_files(args.calc_id, target_dir)
        
        if success:
            print(f"✅ Files retrieved to: {target_dir}")
        else:
            print(f"❌ Failed to retrieve files")
    
    elif args.list_files:
        if not args.calc_id:
            print("❌ --list-files requires --calc-id")
            return
        
        files = storage_manager.list_stored_files(args.calc_id)
        
        if files:
            print(f"📁 Files stored for calculation {args.calc_id}:")
            for file_info in files:
                print(f"   {file_info['file_type']:10} | {file_info['file_name']:30} | {file_info['file_size']:8} bytes")
        else:
            print(f"❌ No files found for calculation {args.calc_id}")
    
    elif args.verify:
        if not args.calc_id:
            print("❌ --verify requires --calc-id")
            return
        
        integrity = storage_manager.verify_file_integrity(args.calc_id)
        
        print(f"🔍 File integrity for calculation {args.calc_id}:")
        for filename, is_valid in integrity.items():
            status = "✅" if is_valid else "❌"
            print(f"   {status} {filename}")
    
    elif args.settings:
        if not args.calc_id:
            print("❌ --settings requires --calc-id")
            return
        
        settings = storage_manager.get_calculation_settings(args.calc_id)
        
        if settings:
            print(f"⚙️  Settings for calculation {args.calc_id}:")
            print(json.dumps(settings, indent=2))
        else:
            print(f"❌ No settings found for calculation {args.calc_id}")


if __name__ == "__main__":
    main()