#!/usr/bin/env python3
"""
Create Fresh Materials Database from Workflow Files
===================================================
Comprehensive script to create a new materials database by processing all
workflow output files with complete property extraction and input settings.

This script:
1. Creates a fresh materials database
2. Discovers all workflow files (.out, .d12, .d3)
3. Extracts material IDs and creates material records
4. Creates calculation records for each step
5. Extracts comprehensive properties from output files
6. Extracts input settings from D12/D3 files
7. Creates file associations for complete provenance

Author: Generated for materials database project
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from material_database import MaterialDatabase
from crystal_property_extractor import CrystalPropertyExtractor
from input_settings_extractor import extract_and_store_input_settings
from formula_extractor import extract_formula_from_d12, extract_space_group_from_output


class WorkflowDatabaseCreator:
    """Create comprehensive materials database from workflow files."""
    
    def __init__(self, workflow_dir: str = "workflow_outputs", db_path: str = "materials.db"):
        self.workflow_dir = Path(workflow_dir)
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        self.property_extractor = CrystalPropertyExtractor(db_path)
        
        # Calculation type mapping
        self.calc_types = {
            'step_001_OPT': 'OPT',
            'step_002_SP': 'SP', 
            'step_003_BAND': 'BAND',
            'step_004_DOSS': 'DOSS'
        }
        
        # Statistics
        self.stats = {
            'materials_created': 0,
            'calculations_created': 0,
            'properties_extracted': 0,
            'input_settings_extracted': 0,
            'files_processed': 0
        }
        
    def create_fresh_database(self):
        """Create a completely fresh database from workflow files."""
        print("🏗️  Creating fresh materials database from workflow files...")
        print(f"   Workflow directory: {self.workflow_dir}")
        print(f"   Database path: {self.db_path}")
        
        # Remove existing database if it exists
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
            print(f"   ✅ Removed existing database")
        
        # Initialize fresh database
        self.db = MaterialDatabase(self.db_path)
        self.property_extractor = CrystalPropertyExtractor(self.db_path)
        
        # Ensure enhanced database schema compatibility
        with self.db._get_connection() as conn:
            # Add input_settings_json column if not exists
            try:
                conn.execute('ALTER TABLE calculations ADD COLUMN input_settings_json TEXT')
                print(f"   ✅ Added input_settings_json column")
            except Exception:
                pass  # Column already exists
                
            # Add last_updated column if not exists
            try:
                conn.execute('ALTER TABLE calculations ADD COLUMN last_updated TEXT')
                print(f"   ✅ Added last_updated column")
            except Exception:
                pass  # Column already exists
                
            # Add job_status column if not exists
            try:
                conn.execute('ALTER TABLE calculations ADD COLUMN job_status TEXT DEFAULT "unknown"')
                print(f"   ✅ Added job_status column")
            except Exception:
                pass  # Column already exists
                
            # Add metadata column to files table if not exists
            try:
                conn.execute('ALTER TABLE files ADD COLUMN metadata TEXT')
                print(f"   ✅ Added files.metadata column")
            except Exception:
                pass  # Column already exists
                
            # Check if materials table has metadata column, if not skip workflow queries
            try:
                cursor = conn.execute("PRAGMA table_info(materials)")
                columns = [row[1] for row in cursor.fetchall()]
                self.has_metadata_column = 'metadata' in columns
                if not self.has_metadata_column:
                    print(f"   ⚠️  Materials table missing metadata column - workflow integration stats disabled")
            except Exception:
                self.has_metadata_column = False
        
        print(f"   ✅ Created fresh database")
        
        # Discover all workflow files
        workflow_files = self._discover_workflow_files()
        print(f"   📂 Discovered {len(workflow_files)} workflow calculations")
        
        # Process each workflow calculation
        for workflow_data in workflow_files:
            self._process_workflow_calculation(workflow_data)
        
        # Process workflow configuration files
        self._process_workflow_configs()
        
        # Print final statistics
        self._print_final_statistics()
        
    def _discover_workflow_files(self) -> List[Dict]:
        """Discover all workflow calculation files."""
        workflow_files = []
        
        # Find all workflow directories
        for workflow_path in self.workflow_dir.glob("workflow_*"):
            if not workflow_path.is_dir():
                continue
                
            workflow_id = workflow_path.name
            
            # Process each step directory
            for step_path in sorted(workflow_path.glob("step_*")):
                if not step_path.is_dir():
                    continue
                    
                step_name = step_path.name
                calc_type = self.calc_types.get(step_name, step_name)
                
                # Process each material directory in this step
                for material_path in step_path.glob("*"):
                    if not material_path.is_dir():
                        continue
                        
                    material_name = material_path.name
                    
                    # Find associated files
                    out_files = list(material_path.glob("*.out"))
                    d12_files = list(material_path.glob("*.d12"))
                    d3_files = list(material_path.glob("*.d3"))
                    
                    # Extract clean material ID
                    material_id = self._extract_material_id(material_name)
                    
                    workflow_data = {
                        'workflow_id': workflow_id,
                        'step_name': step_name,
                        'calc_type': calc_type,
                        'material_id': material_id,
                        'material_name': material_name,
                        'material_path': material_path,
                        'output_files': out_files,
                        'input_d12_files': d12_files,
                        'input_d3_files': d3_files
                    }
                    
                    workflow_files.append(workflow_data)
        
        return workflow_files
    
    def _process_workflow_configs(self):
        """Process workflow configuration and template files."""
        print(f"\n📋 Processing workflow configuration files...")
        
        # Check for workflow_configs directory
        config_dir = self.workflow_dir.parent / "workflow_configs"
        if config_dir.exists():
            config_files = list(config_dir.glob("*.json"))
            print(f"   Found {len(config_files)} configuration files")
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Store config data in database if workflow_instances table exists
                    workflow_id = config_data.get('workflow_id', config_file.stem)
                    
                    with self.db._get_connection() as conn:
                        # Check if workflow_instances table exists
                        cursor = conn.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name='workflow_instances'
                        """)
                        if cursor.fetchone():
                            conn.execute("""
                                INSERT OR REPLACE INTO workflow_instances 
                                (workflow_id, status, workflow_config_json, created_at)
                                VALUES (?, ?, ?, ?)
                            """, (
                                workflow_id,
                                'discovered',
                                json.dumps(config_data),
                                datetime.now().isoformat()
                            ))
                            print(f"   ✅ Stored config: {config_file.name}")
                        else:
                            print(f"   ℹ️  No workflow_instances table - config stored as file reference only")
                            
                except Exception as e:
                    print(f"   ⚠️  Failed to process {config_file.name}: {e}")
        else:
            print(f"   ⚠️  No workflow_configs directory found")
        
        # Check for workflow_scripts directory
        scripts_dir = self.workflow_dir.parent / "workflow_scripts"
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob("*.sh"))
            print(f"   Found {len(script_files)} template script files")
            
            for script_file in script_files:
                try:
                    # Create file record for template scripts
                    with self.db._get_connection() as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO files 
                            (calc_id, file_type, file_name, file_path, file_size, 
                             created_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            'workflow_template',
                            'template',
                            script_file.name,
                            str(script_file),
                            script_file.stat().st_size if script_file.exists() else 0,
                            datetime.now().isoformat(),
                            json.dumps({
                                'file_category': 'template',
                                'source': 'workflow_scripts',
                                'discovered_at': datetime.now().isoformat()
                            })
                        ))
                    
                    self.stats['files_processed'] += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to process {script_file.name}: {e}")
        else:
            print(f"   ⚠️  No workflow_scripts directory found")
    
    def _extract_formula_basic(self, d12_file: Path) -> Optional[str]:
        """Basic formula extraction fallback."""
        try:
            with open(d12_file, 'r') as f:
                content = f.read()
            # Look for atomic number patterns in geometry section
            import re
            atomic_nums = re.findall(r'^\s*(\d+)\s+', content, re.MULTILINE)
            if atomic_nums:
                return f"Material_{len(set(atomic_nums))}elements"
        except Exception:
            pass
        return None
    
    def _extract_space_group_basic(self, output_file: Path) -> Optional[str]:
        """Basic space group extraction fallback."""
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if 'SPACE GROUP' in line or 'CRYSTAL FAMILY' in line:
                        return line.strip().split()[-1]
        except Exception:
            pass
        return None
    
    def _categorize_file_importance(self, file_path: Path) -> str:
        """Categorize file importance for storage management."""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        # Critical files for calculation reproduction
        if suffix in ['.d12', '.d3', '.out']:
            return 'critical'
        
        # Important binary/result files
        if suffix in ['.f9', '.f25'] or 'fort.' in name:
            return 'important'
        
        # Useful auxiliary files
        if suffix in ['.sh', '.log']:
            return 'useful'
        
        # Critical data files
        if suffix == '.DAT' or 'DAT' in file_path.name:
            return 'critical'
        
        # Archive candidates
        return 'auxiliary'
    
    def _extract_material_id(self, material_name: str) -> str:
        """Extract clean material ID from workflow material name."""
        # Remove common suffixes and workflow-specific naming
        material_id = material_name
        
        # Remove calculation type suffixes iteratively (handle cascaded suffixes)
        suffixes = ['_opt', '_sp', '_band', '_doss', '_freq']
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if material_id.endswith(suffix):
                    material_id = material_id[:-len(suffix)]
                    changed = True
                    break
        
        # Remove workflow-specific naming patterns
        material_id = re.sub(r'_BULK_OPTGEOM.*', '', material_id)
        material_id = re.sub(r'_CRYSTAL_.*', '', material_id)
        
        return material_id
    
    def _process_workflow_calculation(self, workflow_data: Dict):
        """Process a single workflow calculation."""
        material_id = workflow_data['material_id']
        calc_type = workflow_data['calc_type']
        workflow_id = workflow_data['workflow_id']
        
        print(f"\n📊 Processing: {material_id} - {calc_type}")
        
        # Create or get material record
        self._ensure_material_exists(workflow_data)
        
        # Create calculation record
        calc_id = self._create_calculation_record(workflow_data)
        
        # Process output files for property extraction
        if workflow_data['output_files']:
            for output_file in workflow_data['output_files']:
                self._extract_properties_from_output(output_file, material_id, calc_id)
        
        # Process input files for settings extraction
        input_files = workflow_data['input_d12_files'] + workflow_data['input_d3_files']
        for input_file in input_files:
            self._extract_input_settings(input_file, calc_id)
        
        # Update file associations
        self._update_file_associations(workflow_data, calc_id)
        
    def _ensure_material_exists(self, workflow_data: Dict):
        """Ensure material record exists in database."""
        material_id = workflow_data['material_id']
        
        # Check if material already exists
        with self.db._get_connection() as conn:
            cursor = conn.execute("SELECT material_id FROM materials WHERE material_id = ?", (material_id,))
            existing = cursor.fetchone()
            if existing:
                # Update metadata to include workflow information (if column exists)
                try:
                    # Check if metadata column exists
                    cursor = conn.execute("PRAGMA table_info(materials)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'metadata' in columns:
                        cursor = conn.execute("SELECT metadata FROM materials WHERE material_id = ?", (material_id,))
                        existing_metadata = cursor.fetchone()[0]
                        if existing_metadata:
                            metadata = json.loads(existing_metadata)
                        else:
                            metadata = {}
                        
                        # Add workflow tracking to metadata
                        if 'workflows' not in metadata:
                            metadata['workflows'] = []
                        if workflow_data['workflow_id'] not in metadata['workflows']:
                            metadata['workflows'].append(workflow_data['workflow_id'])
                            metadata['last_updated'] = datetime.now().isoformat()
                            
                            conn.execute(
                                "UPDATE materials SET metadata = ? WHERE material_id = ?",
                                (json.dumps(metadata), material_id)
                            )
                    else:
                        print(f"   ℹ️  Metadata column not available - using legacy schema")
                except Exception as e:
                    print(f"   ⚠️  Failed to update material metadata: {e}")
                return  # Material already exists
        
        # Extract formula and space group if we have input/output files
        formula = None
        space_group = None
        
        # Try to get formula from D12 file using enhanced extraction
        if workflow_data['input_d12_files']:
            d12_file = workflow_data['input_d12_files'][0]
            try:
                from formula_extractor import extract_formula_from_d12
                formula = extract_formula_from_d12(d12_file)
            except ImportError:
                # Fallback to basic extraction
                formula = self._extract_formula_basic(d12_file)
        
        # Try to get space group from output file using enhanced extraction
        if workflow_data['output_files']:
            output_file = workflow_data['output_files'][0]
            try:
                from formula_extractor import extract_space_group_from_output
                space_group = extract_space_group_from_output(output_file)
            except ImportError:
                # Fallback to basic extraction
                space_group = self._extract_space_group_basic(output_file)
        
        # Create material record with enhanced metadata (if supported)
        try:
            # Check if create_material supports metadata parameter
            import inspect
            sig = inspect.signature(self.db.create_material)
            supports_metadata = 'metadata' in sig.parameters
            
            if supports_metadata:
                metadata = {
                    'workflow_id': workflow_data['workflow_id'],
                    'workflows': [workflow_data['workflow_id']],
                    'source': 'workflow_extraction',
                    'created_at': datetime.now().isoformat(),
                    'extraction_method': 'create_fresh_database',
                    'workflow_context': True
                }
                
                self.db.create_material(
                    material_id=material_id,
                    formula=formula,
                    space_group=space_group,
                    metadata=metadata
                )
            else:
                # Legacy schema - create without metadata
                self.db.create_material(
                    material_id=material_id,
                    formula=formula,
                    space_group=space_group
                )
            self.stats['materials_created'] += 1
            print(f"   ✅ Created material: {material_id} (formula: {formula}, space group: {space_group})")
        except Exception as e:
            print(f"   ⚠️  Failed to create material {material_id}: {e}")
    
    def _create_calculation_record(self, workflow_data: Dict) -> str:
        """Create calculation record and return calc_id."""
        material_id = workflow_data['material_id']
        calc_type = workflow_data['calc_type']
        workflow_id = workflow_data['workflow_id']
        
        # Get file paths
        output_file = str(workflow_data['output_files'][0]) if workflow_data['output_files'] else None
        input_file = None
        if workflow_data['input_d12_files']:
            input_file = str(workflow_data['input_d12_files'][0])
        elif workflow_data['input_d3_files']:
            input_file = str(workflow_data['input_d3_files'][0])
        
        work_dir = str(workflow_data['material_path'])
        
        # Create settings dictionary compatible with enhanced_queue_manager
        settings = {
            'workflow_id': workflow_id,
            'step_name': workflow_data['step_name'],
            'material_name': workflow_data['material_name'],
            'extracted_at': datetime.now().isoformat(),
            'source': 'workflow_database_creation',
            'workflow_context': True
        }
        
        try:
            calc_id = self.db.create_calculation(
                material_id=material_id,
                calc_type=calc_type,
                input_file=input_file,
                work_dir=work_dir,
                settings=settings
            )
            
            # Update with output file path and status if file exists
            if output_file:
                # Determine job status based on file existence and content
                job_status = 'completed' if Path(output_file).exists() else 'unknown'
                
                with self.db._get_connection() as conn:
                    conn.execute(
                        "UPDATE calculations SET output_file = ?, job_status = ? WHERE calc_id = ?",
                        (output_file, job_status, calc_id)
                    )
                    
                    # Also update last_updated timestamp
                    conn.execute(
                        "UPDATE calculations SET last_updated = datetime('now') WHERE calc_id = ?",
                        (calc_id,)
                    )
            
            self.stats['calculations_created'] += 1
            print(f"   ✅ Created calculation: {calc_id}")
            return calc_id
        except Exception as e:
            print(f"   ⚠️  Failed to create calculation: {e}")
            # Generate a fallback calc_id for property extraction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{material_id}_{calc_type}_{timestamp}"
    
    def _extract_properties_from_output(self, output_file: Path, material_id: str, calc_id: str):
        """Extract properties from output file."""
        try:
            properties = self.property_extractor.extract_all_properties(
                output_file, 
                material_id=material_id, 
                calc_id=calc_id
            )
            
            if properties:
                saved_count = self.property_extractor.save_properties_to_database(properties)
                self.stats['properties_extracted'] += saved_count
                print(f"      📊 Extracted {saved_count} properties from {output_file.name}")
            else:
                print(f"      ⚠️  No properties extracted from {output_file.name}")
                
        except Exception as e:
            print(f"      ❌ Failed to extract properties from {output_file.name}: {e}")
    
    def _extract_input_settings(self, input_file: Path, calc_id: str):
        """Extract input settings from D12/D3 file with enhanced error handling."""
        try:
            # Use enhanced input settings extractor if available
            from input_settings_extractor import extract_and_store_input_settings
            success = extract_and_store_input_settings(calc_id, input_file, self.db_path)
            if success:
                self.stats['input_settings_extracted'] += 1
                print(f"      ⚙️  Extracted input settings from {input_file.name}")
            else:
                # Try basic extraction if enhanced fails
                settings = self._extract_basic_settings(input_file)
                if settings:
                    with self.db._get_connection() as conn:
                        conn.execute(
                            "UPDATE calculations SET input_settings_json = ? WHERE calc_id = ?",
                            (json.dumps(settings), calc_id)
                        )
                    self.stats['input_settings_extracted'] += 1
                    print(f"      ⚙️  Extracted basic settings from {input_file.name}")
                else:
                    print(f"      ⚠️  Failed to extract input settings from {input_file.name}")
        except ImportError:
            # Fallback to basic extraction if module not available
            settings = self._extract_basic_settings(input_file)
            if settings:
                with self.db._get_connection() as conn:
                    conn.execute(
                        "UPDATE calculations SET input_settings_json = ? WHERE calc_id = ?",
                        (json.dumps(settings), calc_id)
                    )
                self.stats['input_settings_extracted'] += 1
                print(f"      ⚙️  Extracted basic settings from {input_file.name}")
            else:
                print(f"      ⚠️  Could not extract settings from {input_file.name}")
        except Exception as e:
            print(f"      ❌ Error extracting input settings from {input_file.name}: {e}")
    
    def _extract_basic_settings(self, input_file: Path) -> Optional[Dict]:
        """Basic settings extraction for compatibility."""
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            settings = {
                'filename': input_file.name,
                'file_type': input_file.suffix,
                'extraction_method': 'basic_fallback',
                'extracted_at': datetime.now().isoformat()
            }
            
            # Handle D3 files (BAND/DOSS inputs) differently
            if input_file.suffix == '.d3':
                settings['calculation_type'] = 'properties'
                # Don't assume DFT settings for D3 files - they inherit from previous SP
                lines = content.strip().split('\n')
                if len(lines) > 0:
                    first_line = lines[0].strip()
                    if first_line in ['BAND', 'DOSS', 'NEWK']:
                        settings['property_type'] = first_line
                
                # Extract k-path labels from BAND d3 files
                if 'BAND' in content and len(lines) > 3:
                    k_path_labels = []
                    k_path_segments = []
                    for i, line in enumerate(lines):
                        if i < 3:  # Skip header lines
                            continue
                        if line.strip() == 'END':
                            break
                        line_clean = line.strip()
                        if ' ' in line_clean and not line_clean.startswith('#'):
                            k_points = line_clean.split()
                            if len(k_points) >= 2:
                                # Extract k-point labels (typically first two items)
                                start_point = k_points[0]
                                end_point = k_points[1]
                                k_path_labels.extend([start_point, end_point])
                                k_path_segments.append(f"{start_point} {end_point}")
                    
                    if k_path_labels:
                        # Store both individual labels and path segments
                        unique_labels = list(dict.fromkeys(k_path_labels))  # Preserve order, remove duplicates
                        settings['k_path_labels'] = unique_labels
                        settings['k_path_segments'] = k_path_segments
                        
                        # Create condensed k-path format with proper continuity handling
                        # Example: 'X G', 'G L', 'L W', 'W G' -> 'X G L W G'
                        # Example: 'X G', 'G L', 'G W', 'W G' -> 'X G L|G W G'
                        condensed_segments = []
                        current_path = []
                        
                        for segment in k_path_segments:
                            points = segment.split()
                            if len(points) == 2:
                                start_point, end_point = points
                                
                                # If this is the first segment or continues from previous
                                if not current_path:
                                    current_path = [start_point, end_point]
                                elif current_path[-1] == start_point:
                                    # Continuous path - just add the end point
                                    current_path.append(end_point)
                                else:
                                    # Discontinuous path - finish current and start new
                                    condensed_segments.append(' '.join(current_path))
                                    current_path = [start_point, end_point]
                        
                        # Add the final path segment
                        if current_path:
                            condensed_segments.append(' '.join(current_path))
                        
                        # Join with | for discontinuous segments
                        settings['k_path_condensed'] = '|'.join(condensed_segments)
                
                # For D3 files, mark that settings should be inherited from SP
                settings['settings_inherited'] = True
                settings['note'] = 'Settings inherited from previous SP calculation'
                
            else:
                # For D12 files, extract CRYSTAL keywords
                keywords = ['OPTGEOM', 'DFT', 'EXCHANGE', 'CORRELAT', 'SHRINK', 'TOLINTEG']
                for keyword in keywords:
                    if keyword in content:
                        settings[keyword.lower()] = True
                
                # Attempt to identify functional type from D12 content
                if 'DFT' in content:
                    settings['method'] = 'DFT'
                    # Look for common functionals
                    functionals = ['PBE', 'B3LYP', 'HSE06', 'PBE0', 'LDA']
                    for func in functionals:
                        if func in content:
                            settings['functional'] = func
                            break
                elif any(kw in content for kw in ['RHF', 'UHF']):
                    settings['method'] = 'HF'
            
            return settings
            
        except Exception:
            return None
    
    def _update_file_associations(self, workflow_data: Dict, calc_id: str):
        """Update file associations in database with enhanced metadata."""
        all_files = (workflow_data['output_files'] + 
                    workflow_data['input_d12_files'] + 
                    workflow_data['input_d3_files'])
        
        # Also check for additional common files in the directory
        material_path = workflow_data['material_path']
        if material_path.exists():
            additional_files = []
            # Include DAT files, fort files, and other calculation outputs
            # DAT files follow patterns: {material}.{BAND|DOSS}.DAT or just {BAND|DOSS}.DAT
            for pattern in ['*.sh', '*.f9', '*.f25', '*.log', '*.err', '*.BAND.DAT', '*.DOSS.DAT', 'BAND.DAT', 'DOSS.DAT', 'fort.*', 'INPUT']:
                additional_files.extend(material_path.glob(pattern))
            all_files.extend(additional_files)
        
        for file_path in all_files:
            try:
                # Enhanced file type determination
                suffix = file_path.suffix.lower()
                if suffix == '.out':
                    file_type = 'output'
                elif suffix in ['.d12', '.d3', '.input']:
                    file_type = 'input'
                elif suffix == '.sh':
                    file_type = 'script'
                elif suffix in ['.f9', '.f25']:
                    file_type = 'binary'
                elif suffix in ['.log', '.err']:
                    file_type = 'log'
                elif suffix == '.DAT' or '.BAND.DAT' in file_path.name or '.DOSS.DAT' in file_path.name:
                    file_type = 'data'
                elif file_path.name.startswith('fort.') or file_path.name == 'INPUT':
                    file_type = 'binary'
                else:
                    file_type = 'auxiliary'
                
                # Enhanced file metadata
                file_metadata = {
                    'workflow_id': workflow_data['workflow_id'],
                    'step_name': workflow_data['step_name'],
                    'material_name': workflow_data['material_name'],
                    'discovered_at': datetime.now().isoformat(),
                    'file_category': self._categorize_file_importance(file_path)
                }
                
                # Create file record with enhanced information
                with self.db._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO files 
                        (calc_id, file_type, file_name, file_path, file_size, 
                         created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        calc_id,
                        file_type,
                        file_path.name,
                        str(file_path),
                        file_path.stat().st_size if file_path.exists() else 0,
                        datetime.now().isoformat(),
                        json.dumps(file_metadata)
                    ))
                
                self.stats['files_processed'] += 1
                
            except Exception as e:
                print(f"      ⚠️  Failed to create file record for {file_path.name}: {e}")
    
    def _print_final_statistics(self):
        """Print final database creation statistics."""
        print("\n" + "="*60)
        print("🎉 FRESH DATABASE CREATION COMPLETE!")
        print("="*60)
        print(f"📊 Final Statistics:")
        print(f"   Materials created: {self.stats['materials_created']}")
        print(f"   Calculations created: {self.stats['calculations_created']}")
        print(f"   Properties extracted: {self.stats['properties_extracted']}")
        print(f"   Input settings extracted: {self.stats['input_settings_extracted']}")
        print(f"   Files processed: {self.stats['files_processed']}")
        
        # Show database summary
        with self.db._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM materials")
            materials_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM calculations")
            calculations_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM properties")
            properties_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM files")
            files_count = cursor.fetchone()[0]
        
        print(f"\n📋 Database Contents:")
        print(f"   Materials in database: {materials_count}")
        print(f"   Calculations in database: {calculations_count}")
        print(f"   Properties in database: {properties_count}")
        print(f"   Files in database: {files_count}")
        
        # Show calculation type breakdown
        print(f"\n🔬 Calculation Types:")
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT calc_type, COUNT(*) as count 
                FROM calculations 
                GROUP BY calc_type 
                ORDER BY count DESC
            """)
            for calc_type, count in cursor.fetchall():
                print(f"   {calc_type}: {count} calculations")
        
        # Show property category breakdown
        print(f"\n📊 Property Categories:")
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT property_category, COUNT(*) as count 
                FROM properties 
                GROUP BY property_category 
                ORDER BY count DESC
            """)
            for category, count in cursor.fetchall():
                print(f"   {category}: {count} properties")
        
        # Show workflow completion status (only if metadata column exists)
        print(f"\n🔄 Workflow Integration Status:")
        with self.db._get_connection() as conn:
            if hasattr(self, 'has_metadata_column') and self.has_metadata_column:
                # Show materials with workflow metadata
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM materials 
                    WHERE metadata LIKE '%workflow%'
                """)
                workflow_materials = cursor.fetchone()[0]
                print(f"   Materials with workflow tracking: {workflow_materials}")
            else:
                print(f"   Materials with workflow tracking: N/A (legacy schema)")
            
            # Show calculations with input settings
            cursor = conn.execute("""
                SELECT COUNT(*) FROM calculations 
                WHERE input_settings_json IS NOT NULL
            """)
            settings_count = cursor.fetchone()[0]
            print(f"   Calculations with input settings: {settings_count}")
            
            # Show file type distribution
            cursor = conn.execute("""
                SELECT file_type, COUNT(*) as count 
                FROM files 
                GROUP BY file_type 
                ORDER BY count DESC
            """)
            print(f"\n📁 File Type Distribution:")
            for file_type, count in cursor.fetchall():
                print(f"   {file_type}: {count} files")


def main():
    """Main function to create fresh database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create fresh materials database from workflow files")
    parser.add_argument("--workflow-dir", default="workflow_outputs", 
                       help="Directory containing workflow files")
    parser.add_argument("--db-path", default="materials.db", 
                       help="Path for new database")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing database")
    parser.add_argument("--enable-recovery", action="store_true",
                       help="Enable enhanced error recovery during extraction")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output for debugging")
    
    args = parser.parse_args()
    
    # Check if database exists and force flag
    if Path(args.db_path).exists() and not args.force:
        response = input(f"Database {args.db_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Create fresh database with options
    creator = WorkflowDatabaseCreator(args.workflow_dir, args.db_path)
    
    # Set verbosity if requested
    if args.verbose:
        print(f"Verbose mode enabled")
        print(f"Working directory: {os.getcwd()}")
        print(f"Script location: {__file__}")
    
    creator.create_fresh_database()


if __name__ == "__main__":
    main()