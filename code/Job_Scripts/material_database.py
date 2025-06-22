#!/usr/bin/env python3
"""
Material Database System for CRYSTAL Workflow Tracking
------------------------------------------------------
SQLite-based database with ASE integration for comprehensive material tracking.
Designed to work with existing CRYSTAL workflow scripts.

Author: Based on implementation plan for material tracking system
"""

import sqlite3
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import shutil

# ASE integration for structure storage
try:
    from ase.db import connect as ase_connect
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("Warning: ASE not available. Structure storage will be limited.")


class MaterialDatabase:
    """
    Thread-safe database for tracking CRYSTAL calculations and materials.
    
    Combines SQLite for structured data with ASE database for atomic structures.
    Handles concurrent access from multiple queue manager instances.
    """
    
    def __init__(self, db_path: str = "materials.db", ase_db_path: str = "structures.db"):
        self.db_path = Path(db_path).resolve()
        self.ase_db_path = Path(ase_db_path).resolve()
        self.lock = threading.RLock()
        self._initialize_database()
        
        # Initialize ASE database if available
        if HAS_ASE:
            self.ase_db = ase_connect(str(self.ase_db_path))
        else:
            self.ase_db = None
            
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Materials table: Core material information
                CREATE TABLE IF NOT EXISTS materials (
                    material_id TEXT PRIMARY KEY,
                    formula TEXT NOT NULL,
                    space_group INTEGER,
                    dimensionality TEXT DEFAULT 'CRYSTAL',  -- CRYSTAL, SLAB, POLYMER, MOLECULE
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_type TEXT,  -- cif, d12, manual
                    source_file TEXT,
                    status TEXT DEFAULT 'active',  -- active, archived, error
                    metadata_json TEXT,  -- Flexible JSON storage for additional data
                    notes TEXT
                );
                
                -- Calculations table: Individual calculation tracking
                CREATE TABLE IF NOT EXISTS calculations (
                    calc_id TEXT PRIMARY KEY,
                    material_id TEXT NOT NULL,
                    calc_type TEXT NOT NULL,  -- OPT, SP, BAND, DOSS, FREQ, TRANSPORT
                    calc_subtype TEXT,  -- e.g., for different functionals
                    status TEXT DEFAULT 'pending',  -- pending, submitted, running, completed, failed, cancelled
                    priority INTEGER DEFAULT 0,
                    
                    -- SLURM integration
                    slurm_job_id TEXT UNIQUE,
                    slurm_state TEXT,  -- PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT
                    
                    -- Timing information
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    
                    -- File tracking
                    input_file TEXT,
                    output_file TEXT,
                    job_script TEXT,
                    work_dir TEXT,
                    
                    -- Calculation settings
                    settings_json TEXT,  -- JSON blob for calculation parameters
                    
                    -- Results and errors
                    exit_code INTEGER,
                    error_type TEXT,
                    error_message TEXT,
                    recovery_attempts INTEGER DEFAULT 0,  -- Number of recovery attempts
                    completion_type TEXT DEFAULT 'first_try',  -- first_try, recovered, manual
                    
                    -- Dependencies
                    prerequisite_calc_id TEXT,  -- Which calculation this depends on
                    
                    FOREIGN KEY (material_id) REFERENCES materials (material_id),
                    FOREIGN KEY (prerequisite_calc_id) REFERENCES calculations (calc_id)
                );
                
                -- Properties table: Extracted material properties
                CREATE TABLE IF NOT EXISTS properties (
                    property_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    calc_id TEXT,  -- Which calculation produced this property
                    property_category TEXT NOT NULL,  -- electronic, structural, thermodynamic
                    property_name TEXT NOT NULL,  -- band_gap, work_function, lattice_a, etc.
                    property_value REAL,
                    property_value_text TEXT,  -- For non-numeric properties
                    property_unit TEXT,
                    confidence REAL,  -- Quality metric for the property
                    extracted_at TEXT NOT NULL,
                    extractor_script TEXT,  -- Which script extracted this property
                    
                    FOREIGN KEY (material_id) REFERENCES materials (material_id),
                    FOREIGN KEY (calc_id) REFERENCES calculations (calc_id)
                );
                
                -- Files table: Track all files associated with calculations
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calc_id TEXT NOT NULL,
                    file_type TEXT NOT NULL,  -- input, output, log, property, plot
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    created_at TEXT NOT NULL,
                    checksum TEXT,  -- For integrity checking
                    
                    FOREIGN KEY (calc_id) REFERENCES calculations (calc_id)
                );
                
                -- Workflow templates for common calculation sequences
                CREATE TABLE IF NOT EXISTS workflow_templates (
                    template_id TEXT PRIMARY KEY,
                    template_name TEXT NOT NULL,
                    description TEXT,
                    workflow_steps_json TEXT NOT NULL,  -- JSON array of calculation types and settings
                    created_at TEXT NOT NULL
                );
                
                -- Workflow instances: Track progress through calculation sequences
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    instance_id TEXT PRIMARY KEY,
                    material_id TEXT NOT NULL,
                    template_id TEXT,  -- Optional: NULL for runtime workflows
                    status TEXT DEFAULT 'active',  -- active, completed, failed, paused
                    current_step INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    workflow_config_json TEXT,  -- Full workflow configuration JSON
                    workflow_scripts_json TEXT,  -- Generated SLURM scripts and metadata
                    
                    FOREIGN KEY (material_id) REFERENCES materials (material_id),
                    FOREIGN KEY (template_id) REFERENCES workflow_templates (template_id)
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_materials_formula ON materials (formula);
                CREATE INDEX IF NOT EXISTS idx_materials_status ON materials (status);
                CREATE INDEX IF NOT EXISTS idx_calculations_material ON calculations (material_id);
                CREATE INDEX IF NOT EXISTS idx_calculations_status ON calculations (status);
                CREATE INDEX IF NOT EXISTS idx_calculations_type ON calculations (calc_type);
                CREATE INDEX IF NOT EXISTS idx_calculations_slurm ON calculations (slurm_job_id);
                CREATE INDEX IF NOT EXISTS idx_properties_material ON properties (material_id);
                CREATE INDEX IF NOT EXISTS idx_properties_name ON properties (property_name);
                CREATE INDEX IF NOT EXISTS idx_files_calc ON files (calc_id);
            """)
            
            # Apply any necessary migrations
            self._apply_migrations()
            
    def _apply_migrations(self):
        """Apply database schema migrations for existing databases."""
        with self._get_connection() as conn:
            # Check if recovery_attempts column exists in calculations table
            cursor = conn.execute("PRAGMA table_info(calculations)")
            calc_columns = [row[1] for row in cursor.fetchall()]
            
            if 'recovery_attempts' not in calc_columns:
                print("Adding recovery_attempts column to calculations table...")
                conn.execute("ALTER TABLE calculations ADD COLUMN recovery_attempts INTEGER DEFAULT 0")
                
            if 'completion_type' not in calc_columns:
                print("Adding completion_type column to calculations table...")
                conn.execute("ALTER TABLE calculations ADD COLUMN completion_type TEXT DEFAULT 'first_try'")
            
            if 'input_settings_json' not in calc_columns:
                print("Adding input_settings_json column to calculations table...")
                conn.execute("ALTER TABLE calculations ADD COLUMN input_settings_json TEXT")
            
            # Check if workflow config columns exist in workflow_instances table
            cursor = conn.execute("PRAGMA table_info(workflow_instances)")
            workflow_columns = [row[1] for row in cursor.fetchall()]
            
            if 'workflow_config_json' not in workflow_columns:
                print("Adding workflow_config_json column to workflow_instances table...")
                conn.execute("ALTER TABLE workflow_instances ADD COLUMN workflow_config_json TEXT")
                
            if 'workflow_scripts_json' not in workflow_columns:
                print("Adding workflow_scripts_json column to workflow_instances table...")
                conn.execute("ALTER TABLE workflow_instances ADD COLUMN workflow_scripts_json TEXT")
            
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        with self.lock:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # 30 second timeout for database locks
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
                
    def create_material(self, material_id: str, formula: str, space_group: int = None,
                       dimensionality: str = 'CRYSTAL', source_type: str = None,
                       source_file: str = None, metadata: Dict = None) -> str:
        """
        Create a new material record.
        
        Args:
            material_id: Unique identifier for the material
            formula: Chemical formula
            space_group: Crystallographic space group number
            dimensionality: CRYSTAL, SLAB, POLYMER, or MOLECULE
            source_type: How the material was created (cif, d12, manual)
            source_file: Original source file path
            metadata: Additional metadata as dictionary
            
        Returns:
            material_id of the created material
        """
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO materials (
                    material_id, formula, space_group, dimensionality,
                    created_at, updated_at, source_type, source_file, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (material_id, formula, space_group, dimensionality,
                  now, now, source_type, source_file, metadata_json))
                  
        return material_id
        
    def create_calculation(self, material_id: str, calc_type: str, calc_subtype: str = None,
                          input_file: str = None, work_dir: str = None,
                          settings: Dict = None, prerequisite_calc_id: str = None,
                          priority: int = 0) -> str:
        """
        Create a new calculation record.
        
        Args:
            material_id: Material this calculation belongs to
            calc_type: Type of calculation (OPT, SP, BAND, DOSS, etc.)
            calc_subtype: Subtype for organization (e.g., functional name)
            input_file: Path to input file (.d12)
            work_dir: Working directory for calculation
            settings: Calculation settings as dictionary
            prerequisite_calc_id: Calculation this depends on
            priority: Priority for queue scheduling
            
        Returns:
            calc_id of the created calculation
        """
        # Generate unique calculation ID
        calc_id = f"{material_id}_{calc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if calc_subtype:
            calc_id += f"_{calc_subtype}"
            
        now = datetime.now().isoformat()
        settings_json = json.dumps(settings) if settings else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO calculations (
                    calc_id, material_id, calc_type, calc_subtype, priority,
                    created_at, input_file, work_dir, settings_json, prerequisite_calc_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (calc_id, material_id, calc_type, calc_subtype, priority,
                  now, input_file, work_dir, settings_json, prerequisite_calc_id))
                  
        return calc_id
        
    def update_calculation_status(self, calc_id: str, status: str, slurm_job_id: str = None,
                                 slurm_state: str = None, output_file: str = None,
                                 exit_code: int = None, error_type: str = None,
                                 error_message: str = None):
        """
        Update calculation status and related information.
        
        Args:
            calc_id: Calculation to update
            status: New status (pending, submitted, running, completed, failed, cancelled)
            slurm_job_id: SLURM job ID
            slurm_state: SLURM job state
            output_file: Path to output file
            exit_code: Process exit code
            error_type: Type of error if failed
            error_message: Error message details
        """
        now = datetime.now().isoformat()
        
        # Determine which timestamp to update based on status
        timestamp_updates = []
        if status == 'submitted' and slurm_job_id:
            timestamp_updates.append(('submitted_at', now))
        elif status == 'running':
            timestamp_updates.append(('started_at', now))
        elif status in ['completed', 'failed', 'cancelled']:
            timestamp_updates.append(('completed_at', now))
            
        with self._get_connection() as conn:
            # Build dynamic update query
            update_fields = ['status = ?']
            update_values = [status]
            
            if slurm_job_id:
                update_fields.append('slurm_job_id = ?')
                update_values.append(slurm_job_id)
                
            if slurm_state:
                update_fields.append('slurm_state = ?')
                update_values.append(slurm_state)
                
            if output_file:
                update_fields.append('output_file = ?')
                update_values.append(output_file)
                
            if exit_code is not None:
                update_fields.append('exit_code = ?')
                update_values.append(exit_code)
                
            if error_type:
                update_fields.append('error_type = ?')
                update_values.append(error_type)
                
            if error_message:
                update_fields.append('error_message = ?')
                update_values.append(error_message)
                
            # Add timestamp updates
            for field, value in timestamp_updates:
                update_fields.append(f'{field} = ?')
                update_values.append(value)
                
            update_values.append(calc_id)
            
            query = f"UPDATE calculations SET {', '.join(update_fields)} WHERE calc_id = ?"
            conn.execute(query, update_values)
            
    def update_calculation_settings(self, calc_id: str, settings: Dict[str, Any], merge: bool = False):
        """Update calculation settings."""
        if merge:
            # Get existing settings and merge with new ones
            existing_calc = self.get_calculation(calc_id)
            if existing_calc and existing_calc.get('settings_json'):
                try:
                    existing_settings = json.loads(existing_calc['settings_json'])
                    merged_settings = existing_settings.copy()
                    merged_settings.update(settings)
                    settings = merged_settings
                except json.JSONDecodeError:
                    # If existing settings are invalid JSON, just use new settings
                    pass
        
        settings_json = json.dumps(settings)
        
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE calculations SET settings_json = ? WHERE calc_id = ?",
                (settings_json, calc_id)
            )
            
    def get_calculation_by_slurm_id(self, slurm_job_id: str) -> Optional[Dict]:
        """Get calculation record by SLURM job ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations WHERE slurm_job_id = ?
            """, (slurm_job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
    def get_calculations_by_status(self, status: str = None, calc_type: str = None,
                                  material_id: str = None) -> List[Dict]:
        """Get calculations filtered by status, type, or material."""
        conditions = []
        params = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
            
        if calc_type:
            conditions.append("calc_type = ?")
            params.append(calc_type)
            
        if material_id:
            conditions.append("material_id = ?")
            params.append(material_id)
            
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        with self._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT * FROM calculations{where_clause} ORDER BY created_at DESC
            """, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def get_material(self, material_id: str) -> Optional[Dict]:
        """Get material record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM materials WHERE material_id = ?
            """, (material_id,))
            row = cursor.fetchone()
            if row:
                material = dict(row)
                # Parse metadata JSON
                if material['metadata_json']:
                    material['metadata'] = json.loads(material['metadata_json'])
                return material
            return None
            
    def get_materials_by_status(self, status: str = 'active') -> List[Dict]:
        """Get all materials with given status."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM materials WHERE status = ? ORDER BY created_at DESC
            """, (status,))
            materials = []
            for row in cursor.fetchall():
                material = dict(row)
                if material['metadata_json']:
                    material['metadata'] = json.loads(material['metadata_json'])
                materials.append(material)
            return materials
            
    def get_all_materials(self) -> List[Dict]:
        """Get all materials in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM materials ORDER BY created_at DESC
            """)
            materials = []
            for row in cursor.fetchall():
                material = dict(row)
                if material['metadata_json']:
                    material['metadata'] = json.loads(material['metadata_json'])
                materials.append(material)
            return materials
            
    def get_all_calculations(self) -> List[Dict]:
        """Get all calculations in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_calculations(self, limit: int = 20) -> List[Dict]:
        """Get recent calculations with detailed information."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def get_calculation(self, calc_id: str) -> Optional[Dict]:
        """Get calculation record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations WHERE calc_id = ?
            """, (calc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
    def get_calculations_by_material(self, material_id: str) -> List[Dict]:
        """Get all calculations for a specific material."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations WHERE material_id = ? ORDER BY created_at DESC
            """, (material_id,))
            return [dict(row) for row in cursor.fetchall()]
            
    def add_file_record(self, calc_id: str, file_type: str, file_name: str,
                       file_path: str, checksum: str = None):
        """Add a file record associated with a calculation."""
        now = datetime.now().isoformat()
        file_size = None
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO files (calc_id, file_type, file_name, file_path,
                                 file_size, created_at, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (calc_id, file_type, file_name, file_path, file_size, now, checksum))
            
    def get_next_calculation_in_workflow(self, material_id: str) -> Optional[str]:
        """
        Determine the next calculation type needed for a material based on completed work.
        Returns the next calc_type or None if workflow is complete.
        """
        # Standard workflow: OPT -> SP -> (BAND + DOSS in parallel)
        completed_calcs = set()
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT calc_type FROM calculations 
                WHERE material_id = ? AND status = 'completed'
            """, (material_id,))
            completed_calcs = {row[0] for row in cursor.fetchall()}
            
        # Workflow logic
        if 'OPT' not in completed_calcs:
            return 'OPT'
        elif 'SP' not in completed_calcs:
            return 'SP'
        elif 'BAND' not in completed_calcs:
            return 'BAND'
        elif 'DOSS' not in completed_calcs:
            return 'DOSS'
        else:
            return None  # Workflow complete
            
    def cleanup_old_records(self, days_old: int = 30):
        """Clean up old failed/cancelled calculations to prevent database bloat."""
        cutoff_date = datetime.now().replace(day=datetime.now().day - days_old).isoformat()
        
        with self._get_connection() as conn:
            # Remove old failed calculations
            conn.execute("""
                DELETE FROM calculations 
                WHERE status IN ('failed', 'cancelled') AND created_at < ?
            """, (cutoff_date,))
            
            # Remove orphaned file records
            conn.execute("""
                DELETE FROM files WHERE calc_id NOT IN (
                    SELECT calc_id FROM calculations
                )
            """)
            
    # ============================================================================
    # Workflow Templates and Instances Management
    # ============================================================================
    
    def create_workflow_template(self, template_id: str, template_name: str, 
                                workflow_steps: List[Dict], description: str = None) -> str:
        """
        Create a new workflow template.
        
        Args:
            template_id: Unique identifier for the template
            template_name: Human-readable name
            workflow_steps: List of calculation steps (type, settings, dependencies)
            description: Optional description
            
        Returns:
            template_id of the created template
        """
        now = datetime.now().isoformat()
        workflow_steps_json = json.dumps(workflow_steps)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO workflow_templates (
                    template_id, template_name, description, workflow_steps_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (template_id, template_name, description, workflow_steps_json, now))
            
        return template_id
    
    def get_workflow_template(self, template_id: str) -> Optional[Dict]:
        """Get a workflow template by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_templates WHERE template_id = ?",
                (template_id,)
            )
            row = cursor.fetchone()
            if row:
                template = dict(row)
                template['workflow_steps'] = json.loads(template['workflow_steps_json'])
                return template
            return None
    
    def get_all_workflow_templates(self) -> List[Dict]:
        """Get all workflow templates."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM workflow_templates ORDER BY created_at DESC")
            templates = []
            for row in cursor.fetchall():
                template = dict(row)
                template['workflow_steps'] = json.loads(template['workflow_steps_json'])
                templates.append(template)
            return templates
    
    def create_workflow_instance(self, material_id: str, template_id: str = None, 
                                workflow_config: Dict = None, workflow_scripts: Dict = None) -> str:
        """
        Create a new workflow instance for a material.
        
        Args:
            material_id: Material to run workflow on
            template_id: Template to use (optional)
            workflow_config: Complete workflow configuration from workflow planner
            workflow_scripts: Generated SLURM scripts and metadata
            
        Returns:
            instance_id of the created instance
        """
        instance_id = f"{material_id}_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now().isoformat()
        
        workflow_config_json = json.dumps(workflow_config) if workflow_config else None
        workflow_scripts_json = json.dumps(workflow_scripts) if workflow_scripts else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO workflow_instances (
                    instance_id, material_id, template_id, status, 
                    current_step, started_at, workflow_config_json, workflow_scripts_json
                ) VALUES (?, ?, ?, 'active', 0, ?, ?, ?)
            """, (instance_id, material_id, template_id, now, workflow_config_json, workflow_scripts_json))
            
        return instance_id
    
    def update_workflow_instance_status(self, instance_id: str, status: str, 
                                       current_step: int = None, completed_at: str = None,
                                       workflow_config: Dict = None, workflow_scripts: Dict = None):
        """Update workflow instance status, progress, and configuration data."""
        with self._get_connection() as conn:
            update_fields = ['status = ?']
            update_values = [status]
            
            if current_step is not None:
                update_fields.append('current_step = ?')
                update_values.append(current_step)
                
            if completed_at:
                update_fields.append('completed_at = ?')
                update_values.append(completed_at)
                
            if workflow_config:
                update_fields.append('workflow_config_json = ?')
                update_values.append(json.dumps(workflow_config))
                
            if workflow_scripts:
                update_fields.append('workflow_scripts_json = ?')
                update_values.append(json.dumps(workflow_scripts))
                
            update_values.append(instance_id)
            
            query = f"UPDATE workflow_instances SET {', '.join(update_fields)} WHERE instance_id = ?"
            conn.execute(query, update_values)
    
    def get_workflow_instance(self, instance_id: str) -> Optional[Dict]:
        """Get a workflow instance by ID with parsed JSON fields."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_instances WHERE instance_id = ?",
                (instance_id,)
            )
            row = cursor.fetchone()
            if row:
                instance = dict(row)
                # Parse JSON fields
                if instance.get('workflow_config_json'):
                    try:
                        instance['workflow_config'] = json.loads(instance['workflow_config_json'])
                    except json.JSONDecodeError:
                        instance['workflow_config'] = None
                        
                if instance.get('workflow_scripts_json'):
                    try:
                        instance['workflow_scripts'] = json.loads(instance['workflow_scripts_json'])
                    except json.JSONDecodeError:
                        instance['workflow_scripts'] = None
                        
                return instance
            return None
    
    def get_workflow_instances_by_material(self, material_id: str) -> List[Dict]:
        """Get all workflow instances for a material."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_instances WHERE material_id = ? ORDER BY started_at DESC",
                (material_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_active_workflow_instances(self) -> List[Dict]:
        """Get all active workflow instances."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_instances WHERE status = 'active' ORDER BY started_at"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_workflow_instances(self) -> List[Dict]:
        """Get all workflow instances with parsed JSON fields."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM workflow_instances ORDER BY started_at DESC")
            instances = []
            for row in cursor.fetchall():
                instance = dict(row)
                # Parse JSON fields
                if instance.get('workflow_config_json'):
                    try:
                        instance['workflow_config'] = json.loads(instance['workflow_config_json'])
                    except json.JSONDecodeError:
                        instance['workflow_config'] = None
                        
                if instance.get('workflow_scripts_json'):
                    try:
                        instance['workflow_scripts'] = json.loads(instance['workflow_scripts_json'])
                    except json.JSONDecodeError:
                        instance['workflow_scripts'] = None
                        
                instances.append(instance)
            return instances
    
    def capture_workflow_execution_data(self, workflow_id: str, workflow_configs_dir: str = "workflow_configs", 
                                       workflow_scripts_dir: str = "workflow_scripts") -> Optional[str]:
        """
        Capture workflow configuration and scripts from filesystem and store in database.
        
        Args:
            workflow_id: ID of the workflow (e.g., 'workflow_20250621_170313')
            workflow_configs_dir: Directory containing workflow config JSON files
            workflow_scripts_dir: Directory containing generated SLURM scripts
            
        Returns:
            instance_id if workflow data was captured successfully, None otherwise
        """
        from pathlib import Path
        
        # Look for workflow config file
        config_dir = Path(workflow_configs_dir)
        script_dir = Path(workflow_scripts_dir)
        
        workflow_config = None
        workflow_scripts = {}
        
        # Find and load workflow config JSON
        config_pattern = f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"
        config_files = list(config_dir.glob(config_pattern))
        
        if config_files:
            config_file = config_files[0]
            try:
                with open(config_file, 'r') as f:
                    workflow_config = json.load(f)
                print(f"📋 Loaded workflow config: {config_file.name}")
            except Exception as e:
                print(f"❌ Error loading workflow config {config_file}: {e}")
                return None
        else:
            print(f"⚠️ No workflow config found for pattern: {config_pattern}")
            return None
        
        # Collect all workflow scripts
        if script_dir.exists():
            for script_file in script_dir.glob("*.sh"):
                try:
                    with open(script_file, 'r') as f:
                        script_content = f.read()
                    
                    workflow_scripts[script_file.name] = {
                        'file_path': str(script_file),
                        'content': script_content,
                        'size': script_file.stat().st_size,
                        'created': script_file.stat().st_mtime
                    }
                    print(f"📝 Captured script: {script_file.name}")
                except Exception as e:
                    print(f"❌ Error reading script {script_file}: {e}")
        
        # Extract material IDs from workflow config
        if workflow_config and 'input_files' in workflow_config:
            input_files = workflow_config['input_files']
            
            # Process materials from the workflow
            for file_type, file_list in input_files.items():
                for file_path in file_list:
                    material_id = create_material_id_from_file(file_path)
                    
                    # Create or update workflow instance for this material
                    instance_id = self.create_workflow_instance(
                        material_id=material_id,
                        template_id=None,  # No predefined template
                        workflow_config=workflow_config,
                        workflow_scripts=workflow_scripts
                    )
                    print(f"💾 Stored workflow data for material: {material_id} (instance: {instance_id})")
                    
            return instance_id
        else:
            print("❌ No input files found in workflow config")
            return None

    def get_material_calculation_summary(self, material_id: str = None) -> Dict:
        """
        Get calculation type tracking for materials.
        
        Args:
            material_id: Specific material ID, or None for all materials
            
        Returns:
            Dictionary with calculation type summaries per material
        """
        with self._get_connection() as conn:
            if material_id:
                # Get calculation types for specific material
                cursor = conn.execute("""
                    SELECT calc_type, status, COUNT(*) as count 
                    FROM calculations 
                    WHERE material_id = ?
                    GROUP BY calc_type, status
                    ORDER BY calc_type, status
                """, (material_id,))
                
                results = cursor.fetchall()
                summary = {
                    'material_id': material_id,
                    'calculation_types': {},
                    'total_calculations': 0
                }
                
                for calc_type, status, count in results:
                    if calc_type not in summary['calculation_types']:
                        summary['calculation_types'][calc_type] = {}
                    summary['calculation_types'][calc_type][status] = count
                    summary['total_calculations'] += count
                    
                # Add completed calculation types list
                completed_types = []
                for calc_type, statuses in summary['calculation_types'].items():
                    if statuses.get('completed', 0) > 0:
                        completed_types.append(calc_type)
                summary['completed_types'] = sorted(completed_types)
                
                return summary
            else:
                # Get calculation types for all materials
                cursor = conn.execute("""
                    SELECT material_id, calc_type, status, COUNT(*) as count 
                    FROM calculations 
                    GROUP BY material_id, calc_type, status
                    ORDER BY material_id, calc_type, status
                """)
                
                results = cursor.fetchall()
                summaries = {}
                
                for material_id, calc_type, status, count in results:
                    if material_id not in summaries:
                        summaries[material_id] = {
                            'material_id': material_id,
                            'calculation_types': {},
                            'total_calculations': 0,
                            'completed_types': []
                        }
                    
                    if calc_type not in summaries[material_id]['calculation_types']:
                        summaries[material_id]['calculation_types'][calc_type] = {}
                    
                    summaries[material_id]['calculation_types'][calc_type][status] = count
                    summaries[material_id]['total_calculations'] += count
                
                # Calculate completed types for each material
                for material_id, summary in summaries.items():
                    completed_types = []
                    for calc_type, statuses in summary['calculation_types'].items():
                        if statuses.get('completed', 0) > 0:
                            completed_types.append(calc_type)
                    summary['completed_types'] = sorted(completed_types)
                
                return summaries

    def get_database_stats(self) -> Dict:
        """Get statistics about the database contents."""
        with self._get_connection() as conn:
            stats = {}
            
            # Material counts
            cursor = conn.execute("SELECT COUNT(*) FROM materials")
            stats['total_materials'] = cursor.fetchone()[0]
            
            # Calculation counts by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM calculations GROUP BY status
            """)
            stats['calculations_by_status'] = dict(cursor.fetchall())
            
            # Calculation counts by type
            cursor = conn.execute("""
                SELECT calc_type, COUNT(*) FROM calculations GROUP BY calc_type
            """)
            stats['calculations_by_type'] = dict(cursor.fetchall())
            
            # Database size
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
        return stats
        
    def backup_database(self, backup_dir: str = None):
        """Create a backup of the database."""
        if backup_dir is None:
            backup_dir = Path.cwd() / "backups"
            
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"materials_backup_{timestamp}.db"
        
        # Use SQLite backup API for safe backup of active database
        with self._get_connection() as conn:
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
            
        # Also backup ASE database if it exists
        if self.ase_db and os.path.exists(self.ase_db_path):
            ase_backup_path = backup_dir / f"structures_backup_{timestamp}.db"
            shutil.copy2(self.ase_db_path, ase_backup_path)
            
        return backup_path


# Convenience functions for common operations
def create_material_id_from_file(file_path: str) -> str:
    """
    Generate a consistent material ID from input file path using smart suffix removal.
    
    Handles complex CRYSTAL naming conventions by extracting the core material identifier
    before technical suffixes, preserving unique identifiers and special characters.
    
    Examples:
    - 'test1_opt_BULK_OPTGEOM_symm_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2.d12' → 'test1'
    - 'test3-RCSR-ums_BULK_OPTGEOM_TZ_symm_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2.d12' → 'test3-RCSR-ums'
    """
    file_path = Path(file_path)
    name = file_path.stem
    
    # First, extract the core material identifier (before the first technical suffix)
    # Look for pattern like "materialname_opt_BULK_..." or "materialname_BULK_..."
    parts = name.split('_')
    
    # Find the first part that looks like a technical suffix
    core_parts = []
    for i, part in enumerate(parts):
        # Special case: if this is "opt" and we only have one core part so far,
        # and the core part ends with a number, this might be a calc type rather than material identifier
        if (part.upper() == 'OPT' and len(core_parts) == 1 and 
            core_parts[0] and core_parts[0][-1].isdigit()):
            # This looks like "test1_opt" - the opt is a calc type, not part of material name
            break
        # Check if this part is a technical suffix (removed OPT from this list since we handle it specially)
        elif part.upper() in ['SP', 'FREQ', 'BAND', 'DOSS', 'BULK', 'OPTGEOM', 
                            'CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE', 'SYMM', 'TZ', 'DZ', 'SZ']:
            break
        # Check if this part is a DFT functional
        elif part.upper() in ['PBE', 'B3LYP', 'HSE06', 'PBE0', 'SCAN', 'BLYP', 'BP86']:
            break
        # Check if this part contains basis set info  
        elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():
            break
        # Check if this part is a dispersion correction
        elif 'D3' in part.upper():
            break
        else:
            core_parts.append(part)
    
    # If we found core parts, use them
    if core_parts:
        clean_name = '_'.join(core_parts)
    else:
        # Fallback: just use the first part
        clean_name = parts[0] if parts else name
        
    return clean_name


def find_material_by_similarity(db: 'MaterialDatabase', potential_material_id: str) -> Optional[str]:
    """
    Find existing material by similarity matching.
    
    Handles cases where file naming variations create slightly different material IDs
    for the same material (e.g., 'material_symm' vs 'material').
    
    Args:
        db: MaterialDatabase instance
        potential_material_id: The material ID we're trying to match
        
    Returns:
        Existing material ID if found, None otherwise
    """
    try:
        # Get all existing materials
        existing_materials = db.get_all_materials()
        
        # Direct match first
        for material in existing_materials:
            if material['material_id'] == potential_material_id:
                return potential_material_id
        
        # Similarity matching - check for base name matches
        potential_base = potential_material_id.replace('_symm', '').replace('_opt', '').replace('_sp', '')
        
        for material in existing_materials:
            existing_id = material['material_id']
            existing_base = existing_id.replace('_symm', '').replace('_opt', '').replace('_sp', '')
            
            # If base names match, use the existing material ID
            if potential_base == existing_base or existing_base == potential_base:
                return existing_id
                
            # Check if one is a prefix of the other (handle _symm variations)
            if (potential_base.startswith(existing_base) or 
                existing_base.startswith(potential_base)):
                return existing_id
                
    except Exception as e:
        print(f"Warning: Error in material similarity matching: {e}")
        
    return None


def extract_formula_from_d12(d12_file: str) -> str:
    """Extract chemical formula from .d12 file."""
    try:
        with open(d12_file, 'r') as f:
            lines = f.readlines()
            
        # Look for atomic coordinates section
        in_coords = False
        elements = []
        
        for line in lines:
            line = line.strip()
            if line.isdigit() and not in_coords:
                # Number of atoms line
                in_coords = True
                continue
                
            if in_coords and line:
                parts = line.split()
                if len(parts) >= 4:
                    atomic_num = int(parts[0])
                    if atomic_num > 200:
                        atomic_num -= 200  # Remove ECP offset
                    if atomic_num <= 118:  # Valid atomic number
                        elements.append(atomic_num)
                        
        # Convert to formula (simplified)
        if elements:
            from collections import Counter
            element_counts = Counter(elements)
            # This is a very basic formula generation
            return ''.join(f"{num}" if num > 1 else "" for num in sorted(element_counts.values()))
        else:
            return "Unknown"
            
    except Exception as e:
        print(f"Error extracting formula from {d12_file}: {e}")
        return "Unknown"


if __name__ == "__main__":
    # Test the database functionality
    db = MaterialDatabase("test_materials.db")
    
    # Create a test material
    material_id = db.create_material(
        material_id="Al2O3_test",
        formula="Al2O3",
        space_group=167,
        dimensionality="CRYSTAL",
        source_type="cif",
        source_file="Al2O3.cif"
    )
    
    # Create a test calculation
    calc_id = db.create_calculation(
        material_id=material_id,
        calc_type="OPT",
        input_file="Al2O3_opt.d12",
        work_dir="/path/to/work"
    )
    
    # Update calculation status
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="12345")
    db.update_calculation_status(calc_id, "completed")
    
    # Print statistics
    stats = db.get_database_stats()
    print("Database Statistics:", stats)
    
    print("Test completed successfully!")