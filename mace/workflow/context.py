#!/usr/bin/env python3
"""
Workflow Context Manager for MACE
==================================

Provides isolated execution contexts for MACE workflows to prevent
conflicts when running multiple workflows in the same directory.

Features:
- Isolated databases per workflow
- Separate file storage directories
- Context-aware resource management
- Backward compatible design
"""

import os
import json
import secrets
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from datetime import datetime


class WorkflowContext:
    """Manages isolated resources for a specific workflow."""
    
    _thread_local = threading.local()
    _active_contexts: Dict[str, 'WorkflowContext'] = {}
    _lock = threading.RLock()
    
    def __init__(self, workflow_id: str, base_dir: Optional[Path] = None,
                 isolation_mode: str = "isolated"):
        """
        Initialize workflow context.
        
        Args:
            workflow_id: Unique identifier for this workflow
            base_dir: Base directory for workflow resources
            isolation_mode: 'isolated', 'shared', or 'hybrid'
        """
        self.workflow_id = workflow_id
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.isolation_mode = isolation_mode
        self.context_dir = self.base_dir / f".mace_context_{workflow_id}"
        
        # Resource paths
        self.db_path = self._get_db_path()
        self.structures_db_path = self._get_structures_db_path()
        self.storage_dir = self._get_storage_dir()
        self.lock_dir = self._get_lock_dir()
        self.config_file = self.context_dir / "context_config.json"
        
        # Context state
        self.is_active = False
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
        
    def _get_db_path(self) -> Path:
        """Get path to workflow-specific database."""
        if self.isolation_mode == "shared":
            return self.base_dir / "materials.db"
        else:
            return self.context_dir / "materials.db"
    
    def _get_structures_db_path(self) -> Path:
        """Get path to workflow-specific structures database."""
        if self.isolation_mode == "shared":
            return self.base_dir / "structures.db"
        else:
            return self.context_dir / "structures.db"
    
    def _get_storage_dir(self) -> Path:
        """Get path to workflow-specific storage directory."""
        if self.isolation_mode == "shared":
            return self.base_dir / "calculation_storage"
        else:
            return self.context_dir / "calculation_storage"
    
    def _get_lock_dir(self) -> Path:
        """Get path to workflow-specific lock directory."""
        if self.isolation_mode == "shared":
            return self.base_dir / ".queue_locks"
        else:
            return self.context_dir / ".queue_locks"
    
    def initialize(self) -> None:
        """Initialize workflow context resources."""
        if self.isolation_mode != "shared":
            # Create context directory
            self.context_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.lock_dir.mkdir(parents=True, exist_ok=True)
            
            # Save context configuration
            self._save_config()
            
            # Copy shared database schema if needed
            if self.isolation_mode == "hybrid":
                self._copy_shared_schema()
    
    def _save_config(self) -> None:
        """Save context configuration."""
        config = {
            "workflow_id": self.workflow_id,
            "isolation_mode": self.isolation_mode,
            "created_at": self.created_at.isoformat(),
            "base_dir": str(self.base_dir),
            "db_path": str(self.db_path),
            "structures_db_path": str(self.structures_db_path),
            "storage_dir": str(self.storage_dir),
            "lock_dir": str(self.lock_dir),
            "metadata": self.metadata
        }
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _copy_shared_schema(self) -> None:
        """Copy schema from shared database for hybrid mode."""
        shared_db = self.base_dir / "materials.db"
        if shared_db.exists() and not self.db_path.exists():
            try:
                from mace.database import dbsafety
            except ImportError:          # mace/ on sys.path
                from database import dbsafety
            # Connect to shared database
            conn_shared = dbsafety.connect_rw(shared_db)
            
            # Get schema
            schema_sql = []
            cursor = conn_shared.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                if row[0]:
                    schema_sql.append(row[0])
            conn_shared.close()
            
            # Create isolated database with same schema
            conn_isolated = dbsafety.connect_rw(self.db_path)
            for sql in schema_sql:
                conn_isolated.execute(sql)
            conn_isolated.commit()
            conn_isolated.close()
    
    def activate(self) -> None:
        """Activate this context for the current thread."""
        with self._lock:
            self.initialize()
            self.is_active = True
            
            # Set as active context
            self._active_contexts[self.workflow_id] = self
            
            # Set thread-local context
            self._thread_local.context = self
            
            # Set environment variables for child processes
            os.environ['MACE_WORKFLOW_ID'] = self.workflow_id
            os.environ['MACE_CONTEXT_DIR'] = str(self.context_dir)
            os.environ['MACE_ISOLATION_MODE'] = self.isolation_mode
    
    def deactivate(self) -> None:
        """Deactivate this context."""
        with self._lock:
            self.is_active = False
            
            # Remove from active contexts
            self._active_contexts.pop(self.workflow_id, None)
            
            # Clear thread-local if this is the current context
            if getattr(self._thread_local, 'context', None) == self:
                self._thread_local.context = None
            
            # Clear environment variables
            for key in ['MACE_WORKFLOW_ID', 'MACE_CONTEXT_DIR', 'MACE_ISOLATION_MODE']:
                os.environ.pop(key, None)
    
    def cleanup(self, archive: bool = True) -> None:
        """Clean up context resources."""
        self.deactivate()
        
        if self.isolation_mode != "shared" and self.context_dir.exists():
            if archive:
                # Archive context directory
                archive_name = f"{self.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                archive_path = self.base_dir / "archived_workflows" / archive_name
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(self.context_dir), str(archive_path))
            else:
                # Remove context directory
                shutil.rmtree(self.context_dir)
    
    def archive(self, archive_dir: Optional[Path] = None) -> Path:
        """Archive the workflow context to specified directory.
        
        Args:
            archive_dir: Directory to archive to. If None, uses base_dir/archived_workflows
            
        Returns:
            Path to archived context
        """
        if self.isolation_mode == "shared":
            raise ValueError("Cannot archive shared context")
            
        if not self.context_dir.exists():
            raise ValueError(f"Context directory does not exist: {self.context_dir}")
            
        # Determine archive location
        if archive_dir is None:
            archive_dir = self.base_dir / "archived_workflows"
        else:
            archive_dir = Path(archive_dir)
            
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Create archive name with timestamp
        archive_name = f"{self.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_path = archive_dir / archive_name
        
        # Ensure we're not overwriting an existing archive
        if archive_path.exists():
            raise ValueError(f"Archive already exists: {archive_path}")
            
        # Move context directory to archive
        shutil.move(str(self.context_dir), str(archive_path))
        
        # Update context directory reference
        self.context_dir = archive_path
        
        # The paths will be automatically updated because they use getter methods
        # that reference self.context_dir
        
        print(f"✓ Workflow context archived to: {archive_path}")
        return archive_path
    
    def get_database_path(self) -> Path:
        """Get the path to the materials database for this context."""
        return self.db_path
    
    def get_structures_database_path(self) -> Path:
        """Get the path to the structures database for this context."""
        return self.structures_db_path
    
    def get_ase_database_path(self) -> Path:
        """Get the path to the ASE database for this context (alias for structures database)."""
        return self.structures_db_path
    
    def get_storage_path(self, subdir: Optional[str] = None) -> Path:
        """Get storage path for this context."""
        path = self.storage_dir
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_lock_path(self, lock_name: str) -> Path:
        """Get lock file path for this context."""
        return self.lock_dir / f"{lock_name}.lock"
    
    def get_lock_dir(self) -> Path:
        """Get the lock directory for this context."""
        return self.lock_dir
    
    @classmethod
    def get_active(cls) -> Optional['WorkflowContext']:
        """Get the currently active context for this thread."""
        # First check thread-local storage
        ctx = getattr(cls._thread_local, 'context', None)
        if ctx:
            return ctx
            
        # Fallback to environment variables
        workflow_id = os.environ.get('MACE_WORKFLOW_ID')
        context_dir = os.environ.get('MACE_CONTEXT_DIR')
        isolation_mode = os.environ.get('MACE_ISOLATION_MODE', 'isolated')
        
        if workflow_id and context_dir:
            # Create context from environment
            try:
                ctx = cls.from_environment(workflow_id, context_dir, isolation_mode)
                ctx.activate()
                return ctx
            except Exception:
                # Failed to create from environment
                pass
                
        return None
    
    @classmethod
    def from_environment(cls, workflow_id: str, context_dir: str, isolation_mode: str = 'isolated') -> 'WorkflowContext':
        """Create a workflow context from environment variables."""
        # Check if context already exists
        if workflow_id in cls._active_contexts:
            return cls._active_contexts[workflow_id]
            
        # Parse the context_dir to get base_dir
        context_path = Path(context_dir)
        if context_path.name.startswith(".mace_context_"):
            base_dir = context_path.parent
        else:
            # Context dir doesn't follow expected pattern, use parent
            base_dir = context_path.parent
            
        # Create new context with correct base_dir
        ctx = cls(workflow_id, base_dir=base_dir, isolation_mode=isolation_mode)
        
        # Override context_dir if it's different from the calculated one
        if ctx.context_dir != context_path:
            ctx.context_dir = context_path
            # Update all dependent paths
            ctx.db_path = ctx._get_db_path()
            ctx.structures_db_path = ctx._get_structures_db_path()
            ctx.storage_dir = ctx._get_storage_dir()
            ctx.lock_dir = ctx._get_lock_dir()
            ctx.config_file = ctx.context_dir / "context_config.json"
        
        # Ensure context directory exists
        if ctx.context_dir.exists():
            # Load context configuration if it exists
            config_file = ctx.context_dir / "context_config.json"
            if config_file.exists():
                try:
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        ctx.metadata.update(config.get('metadata', {}))
                        # Update base_dir from config if available
                        if 'base_dir' in config:
                            ctx.base_dir = Path(config['base_dir'])
                except Exception:
                    pass
                    
        return ctx
    
    @classmethod
    def get_or_create_shared(cls) -> 'WorkflowContext':
        """Get or create a shared context."""
        shared_id = "shared_context"
        if shared_id not in cls._active_contexts:
            ctx = cls(shared_id, isolation_mode="shared")
            ctx.activate()
        return cls._active_contexts[shared_id]
    
    def __enter__(self):
        """Context manager entry."""
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.deactivate()
        return False
    
    def export_results(self, export_dir: Path) -> None:
        """Export results from isolated context to shared location."""
        if self.isolation_mode == "shared":
            return  # Nothing to export
        
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export databases (through SQLite: a file copy misses what is
        # still in the write-ahead log)
        if self.db_path.exists():
            _copy_sqlite(self.db_path, export_dir / f"{self.workflow_id}_materials.db")
        
        if self.structures_db_path.exists():
            _copy_sqlite(self.structures_db_path, export_dir / f"{self.workflow_id}_structures.db")
        
        # Export calculation files
        if self.storage_dir.exists():
            storage_export = export_dir / f"{self.workflow_id}_calculations"
            shutil.copytree(self.storage_dir, storage_export, dirs_exist_ok=True)
        
        # Export metadata
        metadata_file = export_dir / f"{self.workflow_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "workflow_id": self.workflow_id,
                "created_at": self.created_at.isoformat(),
                "exported_at": datetime.now().isoformat(),
                "isolation_mode": self.isolation_mode,
                "metadata": self.metadata
            }, f, indent=2)


def _copy_sqlite(source: Path, target: Path) -> None:
    """A consistent copy of a SQLite database, write-ahead log included."""
    import sqlite3
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    except sqlite3.Error:
        shutil.copy2(source, target)
    finally:
        src.close()


@contextmanager
def workflow_context(workflow_id: str, **kwargs):
    """
    Context manager for workflow isolation.
    
    Args:
        workflow_id: Unique workflow identifier
        **kwargs: Additional arguments for WorkflowContext
        
    Yields:
        WorkflowContext: Active context for the workflow
    """
    ctx = WorkflowContext(workflow_id, **kwargs)
    try:
        ctx.activate()
        yield ctx
    finally:
        ctx.deactivate()


# Ids this process has handed out (several may be minted in one second).
_ISSUED_IDS: set = set()
_ISSUED_LOCK = threading.Lock()
# Reservations of minted ids in a folder: not a plan file, so no reader of
# workflow_plan_*.json ever sees one.
ID_RESERVATIONS = Path("workflow_configs") / ".workflow_ids"


def _id_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _id_suffix() -> str:
    return secrets.token_hex(2)


def _id_taken(base: Path, workflow_id: str) -> bool:
    """A plan file or a context of this id already exists in ``base``."""
    plan = base / "workflow_configs" / f"workflow_plan_{workflow_id[len('workflow_'):]}.json"
    return plan.exists() or (base / f".mace_context_{workflow_id}").exists()


def _reserve_id(base: Path, workflow_id: str) -> bool:
    """Claim the id in ``base`` atomically (O_EXCL); False when another
    process has it. A folder that is no workflow folder yet (no
    ``workflow_configs/``) or cannot be written is checked, not claimed."""
    if _id_taken(base, workflow_id):
        return False
    folder = base / ID_RESERVATIONS
    if not folder.parent.is_dir():
        return True
    try:
        folder.mkdir(exist_ok=True)
        fd = os.open(str(folder / workflow_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        return True
    os.close(fd)
    return True


def new_workflow_id(base_dir: Optional[Union[str, Path]] = None) -> str:
    """A new workflow id: ``workflow_<YYYYmmdd_HHMMSS>_<4 hex>``, unique.

    The time alone has one-second resolution, and a random suffix alone
    still repeats now and then, so the id is also checked: not handed out
    before by this process, no plan file or context of it in the folder
    (``base_dir``, else the current one), and, when ``base_dir`` is given,
    reserved there atomically (``workflow_configs/.workflow_ids/<id>``,
    O_EXCL) so another process minting at the same moment gets another.
    On any clash a new suffix is drawn. Everything that reads an id takes it
    whole (the plan file is ``workflow_plan_<id without "workflow_">.json``),
    so existing ids keep resolving.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    while True:
        workflow_id = f"workflow_{_id_stamp()}_{_id_suffix()}"
        with _ISSUED_LOCK:
            if workflow_id in _ISSUED_IDS:
                continue
            try:
                free = (_reserve_id(base, workflow_id) if base_dir is not None
                        else not _id_taken(base, workflow_id))
            except OSError:
                free = True
            if not free:
                continue
            _ISSUED_IDS.add(workflow_id)
            return workflow_id


def workflow_scripts_dir(base_dir: Union[str, Path], workflow_id: str) -> Path:
    """A workflow's own copy of its job-script templates.

    ``workflow_scripts/`` in the folder is shared: the templates there are
    named by step (``submitcrystal23_sp_2.sh``), so a second workflow planned
    in the same folder overwrote the first one's, and the first workflow's
    later steps ran with the second one's resources. Each isolated workflow
    keeps its templates here as well, and the engine builds its steps from
    this copy.
    """
    return Path(base_dir) / f".mace_context_{workflow_id}" / "workflow_scripts"


def keep_workflow_scripts(base_dir: Union[str, Path], workflow_id: str, names) -> None:
    """Copy the named templates from the folder's ``workflow_scripts/`` into
    the workflow's own copy (``workflow_scripts_dir``)."""
    shared = Path(base_dir) / "workflow_scripts"
    own = workflow_scripts_dir(base_dir, workflow_id)
    for name in names:
        src = shared / name
        if src.is_file():
            own.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, own / name)


def freeze_shared_scripts(base_dir: Union[str, Path], workflow_id: Optional[str] = None) -> None:
    """Before ``workflow_scripts/`` is rewritten for a new workflow, give
    every other workflow in the folder that has no copy of its own (planned
    before workflows kept one) a copy of the templates it builds its steps
    from today, so the new workflow's templates never reach it."""
    base = Path(base_dir)
    shared = base / "workflow_scripts"
    try:
        templates = [p for p in shared.glob("*.sh") if p.is_file()]
        contexts = list(base.glob(".mace_context_*"))
    except OSError:
        return
    if not templates:
        return
    for ctx_dir in contexts:
        wid = ctx_dir.name[len(".mace_context_"):]
        if not ctx_dir.is_dir() or wid == workflow_id:
            continue
        own = ctx_dir / "workflow_scripts"
        if own.exists():
            continue
        # Copied whole or not at all: the copy is made aside and renamed
        # into place, so a copy that fails part way (over quota, say) never
        # leaves an own folder missing templates; the workflow then keeps
        # using the shared ones.
        tmp = ctx_dir / f".workflow_scripts.tmp-{secrets.token_hex(4)}"
        try:
            tmp.mkdir()
            for p in templates:
                shutil.copy2(p, tmp / p.name)
            os.rename(tmp, own)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)


def get_current_context() -> Optional[WorkflowContext]:
    """Get the currently active workflow context."""
    return WorkflowContext.get_active()


def auto_activate_context(base_dir: Optional[str] = None,
                          workflow_id: Optional[str] = None) -> Optional[WorkflowContext]:
    """Auto-detect and activate an isolated workflow context if none is active yet.

    Non-interactive counterpart to the monitor dashboard's context detection: scans
    ``base_dir`` (default: cwd) for ``.mace_context_*`` directories and activates one
    when the choice is unambiguous -- either the given ``workflow_id`` or exactly one
    context present. Returns the active context (existing or newly activated), or
    ``None`` when there is nothing to activate or the choice is ambiguous. Never
    prompts and never raises.

    This is what lets ``mace monitor --status`` / ``mace database`` read the
    workflow's isolated database instead of the empty default ``materials.db``.
    """
    existing = get_current_context()
    if existing is not None:
        return existing
    try:
        base = Path(base_dir) if base_dir else Path.cwd()
        if workflow_id:
            target = base / f".mace_context_{workflow_id}"
            dirs = [target] if target.is_dir() else []
        else:
            dirs = sorted(base.glob(".mace_context_*"))
        if len(dirs) == 1:
            wid = dirs[0].name.replace(".mace_context_", "")
            ctx = WorkflowContext(wid, base_dir=str(base))
            ctx.activate()
            return ctx
    except Exception:
        return None
    return None


def require_context(isolation_mode: Optional[str] = None) -> WorkflowContext:
    """
    Get current context or create a default one.
    
    Args:
        isolation_mode: Required isolation mode (optional)
        
    Returns:
        WorkflowContext: Active context
        
    Raises:
        RuntimeError: If no context is active and creation fails
    """
    ctx = get_current_context()
    
    if ctx is None:
        # Try to detect from environment
        workflow_id = os.environ.get('MACE_WORKFLOW_ID')
        if workflow_id:
            context_dir = os.environ.get('MACE_CONTEXT_DIR')
            mode = os.environ.get('MACE_ISOLATION_MODE', 'isolated')
            ctx = WorkflowContext(workflow_id, base_dir=Path(context_dir).parent if context_dir else None,
                                isolation_mode=mode)
            ctx.activate()
        else:
            # Create shared context as fallback
            ctx = WorkflowContext.get_or_create_shared()
    
    if isolation_mode and ctx.isolation_mode != isolation_mode:
        raise RuntimeError(f"Context isolation mode mismatch: required '{isolation_mode}', got '{ctx.isolation_mode}'")
    
    return ctx