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
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import shutil

from . import dbsafety        # the SQLite journal/lock/backup policy (stdlib only)
from mace.utils import property_keys  # the key registry and the old-name aliases (stdlib only)

# The status stored in the database for a canceled job. The value keeps its
# original spelling because existing databases already hold it.
STATUS_CANCELED_DB = "cancelled"  # british-ok: stored database value

# ASE integration for structure storage
try:
    from ase.db import connect as ase_connect
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    # a warning, so stderr: stdout may be a JSON stream (--progress-json, --dry-run --json)
    print("Warning: ASE not available. Structure storage will be limited.", file=sys.stderr)


# PRAGMA user_version of the schema this MACE writes: 1 = the 1.1.6
# properties schema (unique (material, calculation, property) rows, an index
# on calc_id, the property_blobs side table)
SCHEMA_VERSION = 1


class MaterialDatabase:
    """
    Thread-safe database for tracking CRYSTAL calculations and materials.
    
    Combines SQLite for structured data with ASE database for atomic structures.
    Handles concurrent access from multiple queue manager instances.
    """
    
    def __init__(self, db_path: str = "materials.db", ase_db_path: str = "structures.db", auto_initialize: bool = True):
        self.db_path = Path(db_path).resolve()
        self.ase_db_path = Path(ase_db_path).resolve()
        self.lock = threading.RLock()
        self._initialized = False
        
        # Only initialize if auto_initialize is True
        if auto_initialize:
            self._initialize_database()
            
            # Initialize ASE database if available
            if HAS_ASE:
                self.ase_db = ase_connect(str(self.ase_db_path))
            else:
                self.ase_db = None
        else:
            self.ase_db = None
            
    def _ensure_initialized(self):
        """Ensure database is initialized before use."""
        if not self._initialized:
            self._initialize_database()
            if HAS_ASE and self.ase_db is None:
                self.ase_db = ase_connect(str(self.ase_db_path))
            self._initialized = True
            
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self._initialized = True
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
                    confidence REAL,  -- unused (always NULL); kept so MACE 1.1.5 and earlier can read the database
                    extracted_at TEXT NOT NULL,
                    extractor_script TEXT,  -- Which script extracted this property
                    
                    FOREIGN KEY (material_id) REFERENCES materials (material_id),
                    FOREIGN KEY (calc_id) REFERENCES calculations (calc_id)
                );
                
                -- Large JSON properties (neighbor tables, overlap populations,
                -- processed population analysis: property_keys.BLOB_KEYS), kept
                -- out of the properties table so its scans stay small (1.1.6)
                CREATE TABLE IF NOT EXISTS property_blobs (
                    blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    calc_id TEXT,
                    property_category TEXT NOT NULL,
                    property_name TEXT NOT NULL,
                    property_value_text TEXT,
                    property_unit TEXT,
                    extracted_at TEXT NOT NULL,
                    extractor_script TEXT,

                    FOREIGN KEY (material_id) REFERENCES materials (material_id),
                    FOREIGN KEY (calc_id) REFERENCES calculations (calc_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_property_blobs_key
                    ON property_blobs (material_id, ifnull(calc_id, ''), property_name);
                CREATE INDEX IF NOT EXISTS idx_property_blobs_calc ON property_blobs (calc_id);
                
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
                
                -- Workflow states: Independent workflow state tracking
                CREATE TABLE IF NOT EXISTS workflow_states (
                    workflow_id TEXT PRIMARY KEY,
                    material_id TEXT NOT NULL,
                    planned_sequence TEXT NOT NULL,  -- JSON array of planned steps
                    completed_steps TEXT NOT NULL,   -- JSON array of completed steps
                    failed_steps TEXT,               -- JSON array of failed steps
                    current_step INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',    -- active, completed, failed, paused
                    last_updated TEXT NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    recovery_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    notes TEXT,
                    
                    FOREIGN KEY (material_id) REFERENCES materials (material_id)
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
                CREATE INDEX IF NOT EXISTS idx_workflow_states_material ON workflow_states (material_id);
                CREATE INDEX IF NOT EXISTS idx_workflow_states_status ON workflow_states (status);
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

            version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < SCHEMA_VERSION:
            self._migrate_properties_116()

    def _migrate_properties_116(self) -> bool:
        """The 1.1.6 properties schema on a database an earlier MACE built:
        one row per (material, calculation, property) - duplicates removed,
        the newest row kept - enforced by a unique index, an index on
        calc_id, and the large JSON properties (property_keys.BLOB_KEYS)
        moved to the property_blobs table. A database with properties is
        first backed up beside itself (<name>_pre116_backup_<time>.db, never
        pruned); if the backup fails nothing is changed and MACE works on
        without the new indexes, trying again next time. Idempotent (PRAGMA
        user_version records it), and done in one transaction under the
        write lock, as every write is. Returns whether it ran."""
        blob_names = sorted(property_keys.BLOB_KEYS)
        marks = ','.join('?' * len(blob_names))
        with self._get_connection() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
                return False
        # the backup is taken inside the migration's write transaction: a
        # process opening the database at the same time waits for it here and
        # then finds it migrated, rather than writing a full backup of its own
        # (readers go on meanwhile; the database cannot change under the copy)
        with self.transaction() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
                return False  # another process did it meanwhile
            has_rows = conn.execute("SELECT EXISTS (SELECT 1 FROM properties)").fetchone()[0]
            if has_rows:
                try:
                    backup = dbsafety.snapshot_backup(self.db_path, self.db_path.parent,
                                                      f"{self.db_path.stem}_pre116_backup")
                except Exception as e:
                    print(f"⚠️  {self.db_path.name}: not updated to the MACE 1.1.6 schema, as its backup "
                          f"failed ({e}); it is tried again the next time MACE opens it", file=sys.stderr)
                    return False
                print(f"💾 {self.db_path.name}: updating to the MACE 1.1.6 schema; a copy of it as it was "
                      f"is {backup}", file=sys.stderr)
            # one row per (material, calculation, property): the newest
            # (extracted_at, then the higher row id) of a duplicate set. One
            # pass over the table: a correlated EXISTS was quadratic in the
            # rows of each property name (90 s for 215 k rows).
            dropped = conn.execute("""
                DELETE FROM properties WHERE property_id IN (
                    SELECT property_id FROM (
                        SELECT property_id, ROW_NUMBER() OVER (
                            PARTITION BY material_id, ifnull(calc_id, ''), property_name
                            ORDER BY ifnull(extracted_at, '') DESC, property_id DESC) AS rn
                        FROM properties)
                    WHERE rn > 1)""").rowcount
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_properties_key "
                         "ON properties (material_id, ifnull(calc_id, ''), property_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_calc ON properties (calc_id)")
            moved = conn.execute(f"""
                INSERT OR REPLACE INTO property_blobs
                    (material_id, calc_id, property_category, property_name, property_value_text,
                     property_unit, extracted_at, extractor_script)
                SELECT material_id, calc_id, property_category, property_name, property_value_text,
                       property_unit, extracted_at, extractor_script
                FROM properties WHERE property_name IN ({marks})""", blob_names).rowcount
            conn.execute(f"DELETE FROM properties WHERE property_name IN ({marks})", blob_names)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if has_rows:
            print(f"   {dropped} duplicate property rows removed, {moved} large JSON values moved to "
                  f"property_blobs", file=sys.stderr)
        return True
            
    def _open_connection(self) -> sqlite3.Connection:
        """A configured connection (the caller holds self.lock and the write guard)."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,  # 30 second timeout for database locks
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Journal mode by file system (dbsafety): WAL on a local disk as
        # before, the rollback journal on a network/cluster file system
        # (WAL is not safe across nodes); busy_timeout 30 s either way
        dbsafety.configure(conn, self.db_path)
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return conn

    def _shared_connection(self):
        """The connection of the transaction() this thread is inside, or None."""
        local = getattr(self, '_tx_local', None)
        return getattr(local, 'conn', None) if local is not None else None

    @contextmanager
    def transaction(self):
        """One connection and one transaction (BEGIN IMMEDIATE ... COMMIT)
        for a block of writes, e.g. all the property rows of one output.

        Every ``_get_connection()`` this thread opens inside the block uses
        the same connection, as a SAVEPOINT: what it wrote is undone alone
        when it raises, as its own connection's rollback did, and the block
        commits once, under the same lock, write guard and lock retries as
        ``_get_connection``. An error leaving the block rolls all of it back.
        A transaction() inside another one joins it."""
        if self._shared_connection() is not None:
            with self._get_connection() as conn:
                yield conn
            return
        self._ensure_initialized()
        if getattr(self, '_tx_local', None) is None:
            self._tx_local = threading.local()
        with self.lock, dbsafety.write_guard(self.db_path):
            conn = self._open_connection()
            try:
                # nothing is done yet, so taking the write lock may be retried
                dbsafety.retry_locked(lambda: conn.execute("BEGIN IMMEDIATE"))
                self._tx_local.conn = conn
                self._tx_local.depth = 0
                yield conn
                if conn.in_transaction:
                    dbsafety.retry_locked(conn.commit)
            except BaseException:
                conn.rollback()
                raise
            finally:
                self._tx_local.conn = None
                conn.close()

    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager with WAL mode for concurrency."""
        shared = self._shared_connection()
        if shared is not None:
            # inside transaction(): a savepoint on its connection
            self._tx_local.depth += 1
            name = f"mace_sp_{self._tx_local.depth}"
            shared.execute(f"SAVEPOINT {name}")
            try:
                yield shared
            except BaseException:
                if shared.in_transaction:
                    shared.execute(f"ROLLBACK TO {name}")
                    shared.execute(f"RELEASE {name}")
                raise
            else:
                if shared.in_transaction:      # an explicit commit() inside already ended it
                    shared.execute(f"RELEASE {name}")
            finally:
                self._tx_local.depth -= 1
            return

        # Ensure database is initialized before connecting
        self._ensure_initialized()

        with self.lock, dbsafety.write_guard(self.db_path):
            conn = self._open_connection()

            try:
                yield conn
                # a commit is idempotent (the transaction already holds its
                # lock): retried while another process holds the database
                dbsafety.retry_locked(conn.commit)
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
        # Generate unique calculation ID with microsecond precision to avoid collisions
        calc_id = f"{material_id}_{calc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}"
        if calc_subtype:
            calc_id += f"_{calc_subtype}"
            
        now = datetime.now().isoformat()
        settings_json = json.dumps(settings) if settings else None

        # The timestamp in calc_id only has millisecond resolution ([:-3]
        # truncates %f), so two rapid same-type creations can collide on the
        # PRIMARY KEY — uniquify with a suffix instead of raising.
        base_calc_id = calc_id
        with self._get_connection() as conn:
            for suffix in range(2, 100):
                try:
                    conn.execute("""
                        INSERT INTO calculations (
                            calc_id, material_id, calc_type, calc_subtype, priority,
                            created_at, input_file, work_dir, settings_json, prerequisite_calc_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (calc_id, material_id, calc_type, calc_subtype, priority,
                          now, input_file, work_dir, settings_json, prerequisite_calc_id))
                    break
                except sqlite3.IntegrityError:
                    calc_id = f"{base_calc_id}_{suffix}"
            else:
                raise sqlite3.IntegrityError(
                    f"could not create a unique calc_id from {base_calc_id}")

        return calc_id
        
    def update_calculation_status(self, calc_id: str, status: str, slurm_job_id: str = None,
                                 slurm_state: str = None, output_file: str = None,
                                 exit_code: int = None, error_type: str = None,
                                 error_message: str = None):
        """
        Update calculation status and related information.
        
        Args:
            calc_id: Calculation to update
            status: New status (pending, submitted, running, completed, failed, or
                STATUS_CANCELED_DB for a canceled job)
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
        elif status in ['completed', 'failed', STATUS_CANCELED_DB]:
            timestamp_updates.append(('completed_at', now))
            
        with self._get_connection() as conn:
            # Build dynamic update query
            update_fields = ['status = ?']
            update_values = [status]
            
            if slurm_job_id:
                # Only update slurm_job_id if it's not already set to avoid UNIQUE constraint violations
                cursor = conn.execute('SELECT slurm_job_id FROM calculations WHERE calc_id = ?', (calc_id,))
                current_row = cursor.fetchone()
                if current_row and not current_row[0]:  # Only if current slurm_job_id is NULL
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
            
    def update_array_rows(self, rows: List[Dict[str, Any]]):
        """Record the tasks of a job array submission, all rows in one
        transaction: slurm_job_id (A_T, only where none is set yet),
        job_script, settings.slurm_array, submitted_at, and status
        'submitted' unless the row has already moved on."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            for r in rows:
                cur = conn.execute(
                    "SELECT status, settings_json FROM calculations WHERE calc_id = ?",
                    (r['calc_id'],)).fetchone()
                if cur is None:
                    raise KeyError(f"no calculation {r['calc_id']}")
                try:
                    settings = json.loads(cur['settings_json'] or '{}')
                except (json.JSONDecodeError, TypeError):
                    settings = {}
                settings['slurm_array'] = r['slurm_array']
                status = cur['status'] if cur['status'] not in (None, '', 'pending') \
                    else 'submitted'
                conn.execute(
                    "UPDATE calculations SET status = ?, "
                    "slurm_job_id = COALESCE(slurm_job_id, ?), job_script = ?, "
                    "settings_json = ?, submitted_at = ?, "
                    "output_file = COALESCE(?, output_file) WHERE calc_id = ?",
                    (status, r['slurm_job_id'], r.get('job_script'), json.dumps(settings),
                     now, r.get('output_file'), r['calc_id']))

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
            
    def analyze_missing_data(self, material_ids: List[str] = None,
                           target_properties: List[str] = None) -> Dict[str, Any]:
        """
        Analyze missing properties across materials.
        
        Args:
            material_ids: List of materials to analyze (None = all)
            target_properties: Specific properties to check (None = all known)
            
        Returns:
            Analysis results with missing data summary and recommendations
        """
        from mace.database.analysis.missing_data import MissingDataAnalyzer
        analyzer = MissingDataAnalyzer(self)
        return analyzer.analyze_missing_data(material_ids, target_properties)
        
    def get_material_calculations(self, material_id: str) -> List[Dict]:
        """Get all calculations for a specific material."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM calculations 
                WHERE material_id = ? 
                ORDER BY created_at DESC
            """, (material_id,))
            
            calculations = []
            for row in cursor.fetchall():
                calc = dict(row)
                if calc.get('settings_json'):
                    calc['settings'] = json.loads(calc['settings_json'])
                calculations.append(calc)
            
            return calculations
            
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
        """Clean up old failed/canceled calculations to prevent database bloat."""
        # Use timedelta, not replace(day=...): subtracting days_old from the
        # current day-of-month gives a non-positive day for most of the month
        # (e.g. the 13th - 30 = -17) and raised ValueError on nearly every call.
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        with self._get_connection() as conn:
            # Remove old failed calculations
            conn.execute("""
                DELETE FROM calculations 
                WHERE status IN ('failed', ?) AND created_at < ?
            """, (STATUS_CANCELED_DB, cutoff_date))
            
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
            
    def create_workflow_state(self, workflow_id: str, material_id: str, 
                            planned_sequence: List[str]) -> str:
        """
        Create a new workflow state record.
        
        Args:
            workflow_id: Unique workflow identifier
            material_id: Material this workflow is for
            planned_sequence: List of planned calculation steps
            
        Returns:
            workflow_id
        """
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO workflow_states (
                    workflow_id, material_id, planned_sequence, completed_steps,
                    current_step, status, last_updated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (workflow_id, material_id, json.dumps(planned_sequence), 
                  json.dumps([]), 0, 'active', now, now))
                  
        return workflow_id
        
    def update_workflow_state(self, workflow_id: str, completed_step: str = None,
                            failed_step: str = None, status: str = None):
        """Update workflow state after step completion or failure"""
        with self._get_connection() as conn:
            # Get current state
            cursor = conn.execute("""
                SELECT completed_steps, failed_steps, planned_sequence, current_step
                FROM workflow_states WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()
            
            if not row:
                return
                
            completed_steps = json.loads(row[0])
            failed_steps = json.loads(row[1]) if row[1] else []
            planned_sequence = json.loads(row[2])
            current_step = row[3]
            
            # Update lists
            if completed_step and completed_step not in completed_steps:
                completed_steps.append(completed_step)
                # Update current step index
                if completed_step in planned_sequence:
                    current_step = planned_sequence.index(completed_step) + 1
                    
            if failed_step and failed_step not in failed_steps:
                failed_steps.append(failed_step)
                
            # Determine status if not provided
            if not status:
                if len(completed_steps) == len(planned_sequence):
                    status = 'completed'
                elif failed_steps:
                    status = 'failed'
                else:
                    status = 'active'
                    
            # Update record
            now = datetime.now().isoformat()
            conn.execute("""
                UPDATE workflow_states SET
                    completed_steps = ?,
                    failed_steps = ?,
                    current_step = ?,
                    status = ?,
                    last_updated = ?,
                    error_count = error_count + ?
                WHERE workflow_id = ?
            """, (json.dumps(completed_steps), json.dumps(failed_steps),
                  current_step, status, now, 1 if failed_step else 0, workflow_id))
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow state record"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM workflow_states WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()
            
            if row:
                state = dict(row)
                # Parse JSON fields
                state['planned_sequence'] = json.loads(state['planned_sequence'])
                state['completed_steps'] = json.loads(state['completed_steps'])
                state['failed_steps'] = json.loads(state['failed_steps']) if state['failed_steps'] else []
                return state
                
            return None
    
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
        
    def store_material_property(self, material_id: str, property_name: str,
                               property_value: any, property_category: str = 'General',
                               calc_id: str = None, property_unit: str = None,
                               confidence: float = None, extractor_script: str = None,
                               blob: bool = False) -> int:
        """Store a material property in the database.

        Returns the autoincremented integer property_id. The ``property_id``
        column is ``INTEGER PRIMARY KEY AUTOINCREMENT``, so it must NOT be set
        explicitly (a string UUID raises a SQLite datatype mismatch).

        A dict or list value is stored as JSON text. ``blob=True`` (a large
        JSON value, such as a derived property's curve) puts the row in the
        property_blobs table instead, one row per (material, calculation,
        property) there too; the return value is then its blob_id.
        """
        if blob:
            return self._store_property_blob(material_id, property_name, property_value, property_category,
                                             calc_id, property_unit, extractor_script)
        with self._get_connection() as conn:
            # Handle both numeric and text values
            property_value_num = None
            property_value_text = None

            if isinstance(property_value, (dict, list, tuple)):
                property_value_text = json.dumps(property_value)
            else:
                try:
                    property_value_num = float(property_value)
                except (ValueError, TypeError):
                    property_value_text = str(property_value)

            # one row per (material, calculation, property) (the unique
            # index of the 1.1.6 schema): a stored one is replaced
            conn.execute(
                "DELETE FROM properties WHERE material_id = ? AND property_name = ? "
                "AND (calc_id = ? OR (calc_id IS NULL AND ? IS NULL))",
                (material_id, property_name, calc_id, calc_id))
            cursor = conn.execute("""
                INSERT INTO properties (
                    material_id, calc_id, property_category,
                    property_name, property_value, property_value_text,
                    property_unit, confidence, extractor_script,
                    extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                material_id, calc_id, property_category,
                property_name, property_value_num, property_value_text,
                property_unit, confidence, extractor_script,
                # isoformat string (matches every other timestamp in this module
                # and avoids the deprecated sqlite3 default datetime adapter on 3.12+)
                datetime.now().isoformat()
            ))

            if self._shared_connection() is None:
                # inside transaction() the block commits once, at its end
                conn.commit()
            return cursor.lastrowid

    def _store_property_blob(self, material_id: str, property_name: str, property_value, property_category,
                             calc_id, property_unit, extractor_script) -> int:
        """store_material_property(blob=True): the property_blobs row (a
        copy of the name in the properties table goes)."""
        text = property_value if isinstance(property_value, str) else json.dumps(property_value)
        key = (material_id, property_name, calc_id, calc_id)
        where = "material_id = ? AND property_name = ? AND (calc_id = ? OR (calc_id IS NULL AND ? IS NULL))"
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM properties WHERE {where}", key)
            conn.execute(f"DELETE FROM property_blobs WHERE {where}", key)
            cursor = conn.execute("""
                INSERT INTO property_blobs (material_id, calc_id, property_category, property_name,
                                            property_value_text, property_unit, extracted_at, extractor_script)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (material_id, calc_id, property_category, property_name, text, property_unit,
                  datetime.now().isoformat(), extractor_script))
            if self._shared_connection() is None:
                conn.commit()
            return cursor.lastrowid

    def get_material_properties(self, material_id: str, include_blobs: bool = True) -> List[Dict]:
        """Get all properties for a specific material: the large JSON values
        of the property_blobs table too, unless ``include_blobs`` is False
        (their property_id is None)."""
        with self._get_connection() as conn:
            blobs = ""
            if include_blobs and conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'property_blobs'").fetchone():
                blobs = """
                UNION ALL
                SELECT NULL, material_id, calc_id, property_category,
                       property_name, NULL, property_value_text,
                       property_unit, NULL, extractor_script,
                       extracted_at
                FROM property_blobs
                WHERE material_id = ?"""
            cursor = conn.execute(f"""
                SELECT property_id, material_id, calc_id, property_category,
                       property_name, property_value, property_value_text,
                       property_unit, confidence, extractor_script,
                       extracted_at
                FROM properties
                WHERE material_id = ?{blobs}
                ORDER BY property_category, property_name
            """, (material_id, material_id) if blobs else (material_id,))
            
            columns = [desc[0] for desc in cursor.description]
            # names renamed in 1.1.6 shown under their current names
            # (a database built earlier: property_keys.normalize_rows)
            results = property_keys.normalize_rows(dict(zip(columns, row)) for row in cursor.fetchall())
            for prop in results:
                # Use text value if numeric value is null
                if prop['property_value'] is None and prop['property_value_text']:
                    prop['property_value'] = prop['property_value_text']
                
            return results
            
    def get_all_properties(self) -> List[Dict]:
        """Get all properties from the database, without the large JSON
        values of the property_blobs table (get_material_properties has
        them)."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT property_id, material_id, calc_id, property_category,
                       property_name, property_value, property_value_text,
                       property_unit, confidence, extractor_script,
                       extracted_at
                FROM properties
                ORDER BY material_id, property_category, property_name
            """)
            
            columns = [desc[0] for desc in cursor.description]
            # names renamed in 1.1.6 shown under their current names
            # (a database built earlier: property_keys.normalize_rows)
            results = property_keys.normalize_rows(dict(zip(columns, row)) for row in cursor.fetchall())
            for prop in results:
                # Use text value if numeric value is null
                if prop['property_value'] is None and prop['property_value_text']:
                    prop['property_value'] = prop['property_value_text']
                
            return results
            
    def get_properties_by_name(self, property_name: str) -> List[Dict]:
        """Get all values of a specific property across materials (a name
        renamed in 1.1.6 also finds the rows of its old names)."""
        names = property_keys.query_names(property_name)
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT m.material_id, m.formula, p.property_name, p.property_value,
                       p.property_value_text, p.property_unit, p.calc_id,
                       p.extracted_at
                FROM properties p
                JOIN materials m ON p.material_id = m.material_id
                WHERE p.property_name IN ({})
                ORDER BY p.property_value
            """.format(','.join('?' * len(names))), names)
            
            columns = [desc[0] for desc in cursor.description]
            # names renamed in 1.1.6 shown under their current names
            # (a database built earlier: property_keys.normalize_rows)
            results = property_keys.normalize_rows(dict(zip(columns, row)) for row in cursor.fetchall())
            for prop in results:
                # Use text value if numeric value is null
                if prop['property_value'] is None and prop['property_value_text']:
                    prop['property_value'] = prop['property_value_text']
                
            return results
        
    def backup_database(self, backup_dir: str = None):
        """Create a backup of the database."""
        if backup_dir is None:
            backup_dir = Path.cwd() / "backups"
            
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        
        # A consistent copy through the SQLite backup API from a read-only
        # connection (dbsafety.snapshot_backup), for the ASE database too: it
        # is SQLite as well, and a plain file copy can tear while it is written
        self._ensure_initialized()
        backup_path = dbsafety.snapshot_backup(self.db_path, backup_dir, "materials_backup")
        if self.ase_db and os.path.exists(self.ase_db_path):
            dbsafety.snapshot_backup(self.ase_db_path, backup_dir, "structures_backup")
            
        return backup_path
    
    def filter_materials_by_properties(self, filter_strings: List[str], logic: str = 'AND') -> List[Dict]:
        """
        Filter materials by property value ranges.
        
        Args:
            filter_strings: List of filter strings like ["band_gap > 3.0", "total_energy < -1000"]
            logic: Logic to combine filters ('AND' or 'OR')
            
        Returns:
            List of materials that match the filters
        """
        from mace.database.query.filters import create_filter_from_strings
        
        # Get all materials and properties
        all_materials = self.get_all_materials()
        all_properties = self.get_all_properties()
        
        # Create and apply filter
        filter_obj = create_filter_from_strings(filter_strings, logic)
        matching_material_ids = filter_obj.apply_to_materials(all_materials, all_properties)
        
        # Return full material records for matching IDs
        return [m for m in all_materials if m['material_id'] in matching_material_ids]
        
    def filter_materials_advanced(self, expression: str) -> List[Dict]:
        """
        Filter materials using advanced SQL-like syntax.
        
        Supports:
        - Parentheses: (band_gap > 3 AND space_group = 227) OR total_energy < -1000
        - LIKE operator: formula LIKE 'C%'
        - IN operator: space_group IN (225, 227, 229)
        - IS NULL/IS NOT NULL: transport_seebeck IS NOT NULL
        
        Args:
            expression: Advanced filter expression
            
        Returns:
            List of materials that match the filter
        """
        from mace.database.query.advanced_filters import evaluate_advanced_filter
        
        # Get all materials and their properties
        all_materials = self.get_all_materials()
        all_properties = self.get_all_properties()
        
        # Group properties by material
        props_by_material = {}
        for prop in all_properties:
            mat_id = prop['material_id']
            if mat_id not in props_by_material:
                props_by_material[mat_id] = []
            props_by_material[mat_id].append(prop)
            
        # Apply filter to each material
        matching_materials = []
        for material in all_materials:
            mat_id = material['material_id']
            mat_props = props_by_material.get(mat_id, [])
            
            try:
                if evaluate_advanced_filter(expression, material, mat_props):
                    matching_materials.append(material)
            except Exception as e:
                # Log error but continue filtering
                print(f"Warning: Error filtering material {mat_id}: {e}")
                
        return matching_materials
        
    def filter_properties(self, filter_strings: List[str], material_id: str = None) -> List[Dict]:
        """
        Filter properties by value ranges.
        
        Args:
            filter_strings: List of filter strings
            material_id: Optional material ID to filter by
            
        Returns:
            List of properties that match the filters
        """
        from mace.database.query.filters import create_filter_from_strings
        
        # Get properties
        if material_id:
            properties = self.get_material_properties(material_id)
        else:
            properties = self.get_all_properties()
            
        # Create and apply filter
        filter_obj = create_filter_from_strings(filter_strings, 'OR')  # OR logic for properties
        return filter_obj.apply_to_properties(properties)


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
    import re
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
        # Check if this part matches a calculation type with optional number (opt, opt2, opt10, etc.)
        # transport/charge/potential/charge+potential/orbitals are calc types too — without them,
        # simple-named TRANSPORT / CHARGE+POTENTIAL outputs registered as brand-new
        # (duplicate) materials, the exact bug the canonical-ID routing exists to fix.
        if re.match(r'^(opt|sp|band|doss|freq|transport|charge\+potential|charge|potential|orbitals)\d*$', part.lower()):
            # This looks like opt, opt2, sp10, band3, transport2, etc. - stop here
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
    """Extract chemical formula from a .d12 file.

    Delegates to mace.utils.formula_extractor, which maps atomic numbers to
    element symbols and builds an ordered formula (e.g. 'TiPbO3'). The previous
    local implementation joined the *count values* without ever converting
    atomic numbers to symbols, so it returned a meaningless digit string; the
    queue manager and workflow engine import this function, so every material
    they registered got a garbage formula.
    """
    from pathlib import Path
    try:
        from mace.utils.formula_extractor import extract_formula_from_d12 as _extract
    except ImportError:  # pragma: no cover - fallback when run without the package
        from formula_extractor import extract_formula_from_d12 as _extract
    try:
        formula = _extract(Path(d12_file))
    except Exception as e:
        print(f"Error extracting formula from {d12_file}: {e}")
        formula = None
    return formula or "Unknown"


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