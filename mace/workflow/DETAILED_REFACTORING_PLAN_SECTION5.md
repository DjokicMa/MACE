# MACE Workflow Module - Detailed Refactoring Plan
# Section 5: Architecture Improvements

## 5. Architecture Improvements

### 5.1 Configuration Management System

#### 5.1.1 Current Configuration Issues

```
Issue                                  Occurrences    Example
----------------------------------  -------------  -----------------------------------------
Hard-coded SLURM parameters                   234  account='mendoza_q'
Hard-coded timeouts                            89  timeout=300  # seconds
Hard-coded file paths                         156  '/path/to/crystal/bin'
Magic numbers in code                         178  if retries > 5:
Environment-specific values                    67  module load intel/2021.2
Scattered configuration logic                 412  Various config checks throughout
No configuration validation                   N/A  Values used without checking
No configuration versioning                   N/A  No way to track config changes
```

#### 5.1.2 Comprehensive Configuration Solution

##### Directory Structure
```
config/
├── __init__.py
├── loader.py                    # Configuration loading system
├── validator.py                 # Configuration validation
├── schema.py                   # Configuration schema definitions
├── defaults/
│   ├── workflow.yaml           # Default workflow settings
│   ├── resources.yaml          # Default resource allocations
│   ├── paths.yaml             # Default paths and directories
│   └── advanced.yaml          # Advanced settings
├── environments/
│   ├── development.yaml       # Development environment
│   ├── production.yaml        # Production environment
│   └── testing.yaml          # Testing environment
└── user/
    └── .gitignore            # Ignore user configurations
```

##### Master Configuration Schema (config/schema.py)
```python
"""
Configuration schema definitions using pydantic for validation.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
from enum import Enum

class ResourceAllocation(BaseModel):
    """Resource allocation for SLURM jobs."""
    cores: int = Field(32, ge=1, le=128, description="Number of cores")
    memory: str = Field("5G", regex=r'^\d+[GMK]$', description="Memory allocation")
    walltime: str = Field("24:00:00", regex=r'^\d+-?\d+:\d+:\d+$', description="Wall time")
    account: str = Field("mendoza_q", description="SLURM account")
    partition: Optional[str] = Field("general", description="SLURM partition")
    qos: Optional[str] = Field(None, description="Quality of service")
    
    @validator('memory')
    def validate_memory(cls, v):
        """Ensure memory is in valid format."""
        import re
        if not re.match(r'^\d+[GMK]$', v):
            raise ValueError("Memory must be in format: 123G, 456M, or 789K")
        return v

class CalculationType(str, Enum):
    """Valid calculation types."""
    OPT = "OPT"
    SP = "SP"
    FREQ = "FREQ"
    BAND = "BAND"
    DOSS = "DOSS"
    TRANSPORT = "TRANSPORT"
    CHARGE_POTENTIAL = "CHARGE+POTENTIAL"
    ECHG = "ECHG"
    POTM = "POTM"

class WorkflowConfig(BaseModel):
    """Workflow configuration schema."""
    
    # Workflow settings
    name: str = Field(..., description="Workflow name")
    version: str = Field("1.0", description="Configuration version")
    
    # Optional calculations
    optional_calculations: List[CalculationType] = Field(
        default_factory=lambda: [
            CalculationType.CHARGE_POTENTIAL,
            CalculationType.TRANSPORT,
            CalculationType.ECHG,
            CalculationType.POTM
        ],
        description="Calculations considered optional"
    )
    
    # Resource defaults by calculation type
    resource_defaults: Dict[CalculationType, ResourceAllocation] = Field(
        default_factory=dict,
        description="Default resources for each calculation type"
    )
    
    # Timeouts
    timeouts: Dict[str, int] = Field(
        default_factory=lambda: {
            "cif_conversion": 300,
            "file_check": 30,
            "job_submission": 60,
            "database_operation": 10
        },
        description="Operation timeouts in seconds"
    )
    
    # File patterns
    file_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "output": "*.out",
            "wavefunction": "*.f9",
            "properties": "*.f25",
            "input": "*.d12",
            "properties_input": "*.d3"
        },
        description="File patterns for different file types"
    )
    
    # Retry policies
    retry_policies: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "job_submission": {"max_retries": 3, "backoff": 60},
            "file_operations": {"max_retries": 5, "backoff": 10},
            "database_operations": {"max_retries": 3, "backoff": 1}
        },
        description="Retry policies for different operations"
    )

class PathConfig(BaseModel):
    """Path configuration schema."""
    
    # Base directories
    base_dir: Path = Field(Path("."), description="Base directory")
    work_dir: Path = Field(Path("workflow_work"), description="Working directory")
    archive_dir: Path = Field(Path("workflow_archive"), description="Archive directory")
    temp_dir: Path = Field(Path("workflow_temp"), description="Temporary directory")
    
    # Tool paths
    crystal_bin: Path = Field(
        Path("/opt/crystal23/bin"),
        description="CRYSTAL binary directory"
    )
    mace_tools: Path = Field(
        Path.home() / "mace",
        description="MACE tools directory"
    )
    
    # Script paths
    d12_scripts: Path = Field(
        Path("Crystal_d12"),
        description="D12 script directory"
    )
    d3_scripts: Path = Field(
        Path("Crystal_d3"),
        description="D3 script directory"
    )
    
    @validator('*', pre=True)
    def expand_paths(cls, v):
        """Expand user paths and make absolute."""
        if isinstance(v, str):
            v = Path(v)
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

class EnvironmentConfig(BaseModel):
    """Environment-specific configuration."""
    
    # Module system
    modules_enabled: bool = Field(True, description="Use module system")
    modules_to_load: List[str] = Field(
        default_factory=lambda: [
            "intel/2021.2",
            "impi/2021.2",
            "mkl/2021.2"
        ],
        description="Modules to load"
    )
    
    # Environment variables
    environment_vars: Dict[str, str] = Field(
        default_factory=lambda: {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CRYSTAL23_ROOT": "/opt/crystal23"
        },
        description="Environment variables to set"
    )
    
    # SLURM settings
    slurm_enabled: bool = Field(True, description="Use SLURM")
    slurm_constraints: Optional[str] = Field(None, description="SLURM constraints")
    
    # Scratch settings
    use_scratch: bool = Field(True, description="Use scratch directory")
    scratch_base: Path = Field(Path("$SCRATCH"), description="Scratch base directory")

class AdvancedConfig(BaseModel):
    """Advanced configuration options."""
    
    # Performance settings
    parallel_submissions: int = Field(5, ge=1, le=20, description="Max parallel job submissions")
    queue_check_interval: int = Field(60, ge=10, description="Queue check interval in seconds")
    
    # Database settings
    database_pool_size: int = Field(5, ge=1, le=20, description="Database connection pool size")
    database_timeout: int = Field(30, ge=5, description="Database timeout in seconds")
    
    # Logging settings
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_rotation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        },
        description="Log rotation settings"
    )
    
    # Feature flags
    feature_flags: Dict[str, bool] = Field(
        default_factory=lambda: {
            "use_new_monitor": True,
            "enable_auto_recovery": True,
            "use_caching": True,
            "enable_notifications": False
        },
        description="Feature flags"
    )

class MasterConfig(BaseModel):
    """Master configuration combining all sections."""
    
    workflow: WorkflowConfig
    paths: PathConfig
    environment: EnvironmentConfig
    advanced: AdvancedConfig
    
    # Metadata
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    modified_at: Optional[str] = Field(None, description="Last modification timestamp")
    schema_version: str = Field("2.0", description="Schema version")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            Path: str
        }
```

##### Configuration Loader (config/loader.py)
```python
"""
Configuration loading and management system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .schema import MasterConfig, WorkflowConfig, PathConfig, EnvironmentConfig, AdvancedConfig
from .validator import ConfigValidator

class ConfigurationManager:
    """
    Centralized configuration management.
    
    Features:
    - Hierarchical configuration loading
    - Environment-based overrides
    - Validation with schema
    - Hot-reloading support
    - Configuration caching
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.logger = logging.getLogger('ConfigurationManager')
        
        # Configuration cache
        self._config: Optional[MasterConfig] = None
        self._raw_config: Dict[str, Any] = {}
        self._load_time: Optional[datetime] = None
        
        # Validator
        self.validator = ConfigValidator()
        
        # Load configuration
        self.reload()
    
    def reload(self) -> MasterConfig:
        """Reload configuration from files."""
        self.logger.info("Reloading configuration")
        
        # Load base configuration
        raw_config = self._load_hierarchical_config()
        
        # Apply environment overrides
        raw_config = self._apply_environment_overrides(raw_config)
        
        # Apply user overrides
        raw_config = self._apply_user_overrides(raw_config)
        
        # Validate and create typed configuration
        self._config = self._create_typed_config(raw_config)
        self._raw_config = raw_config
        self._load_time = datetime.now()
        
        self.logger.info("Configuration loaded successfully")
        return self._config
    
    def _load_hierarchical_config(self) -> Dict[str, Any]:
        """Load configuration files in hierarchical order."""
        config = {}
        
        # Load default configurations
        defaults_dir = self.config_dir / "defaults"
        for config_file in defaults_dir.glob("*.yaml"):
            section_name = config_file.stem
            section_config = self._load_yaml_file(config_file)
            config[section_name] = section_config
        
        # Load environment-specific configuration
        env = os.environ.get('MACE_ENV', 'development')
        env_file = self.config_dir / "environments" / f"{env}.yaml"
        if env_file.exists():
            env_config = self._load_yaml_file(env_file)
            config = self._deep_merge(config, env_config)
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse YAML file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Environment variables follow pattern: MACE_SECTION_KEY
        # Example: MACE_WORKFLOW_NAME=my_workflow
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith('MACE_'):
                # Parse environment variable
                parts = env_key[5:].lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    key = '_'.join(parts[1:])
                    
                    # Apply override
                    if section not in config:
                        config[section] = {}
                    
                    # Handle nested keys (MACE_SECTION_NESTED_KEY)
                    current = config[section]
                    key_parts = key.split('_')
                    
                    for part in key_parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Convert value type
                    current[key_parts[-1]] = self._convert_env_value(env_value)
        
        return config
    
    def _apply_user_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user-specific overrides."""
        user_config_file = self.config_dir / "user" / "config.yaml"
        
        if user_config_file.exists():
            user_config = self._load_yaml_file(user_config_file)
            config = self._deep_merge(config, user_config)
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]
        
        # String
        return value
    
    def _create_typed_config(self, raw_config: Dict[str, Any]) -> MasterConfig:
        """Create typed configuration from raw dictionary."""
        # Add metadata
        raw_config['created_at'] = raw_config.get('created_at', datetime.now().isoformat())
        raw_config['modified_at'] = datetime.now().isoformat()
        
        # Create typed configuration
        return MasterConfig(**raw_config)
    
    @property
    def config(self) -> MasterConfig:
        """Get current configuration."""
        if self._config is None:
            self.reload()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Examples:
            config.get('workflow.name')
            config.get('paths.base_dir')
            config.get('advanced.feature_flags.use_caching')
        """
        keys = key.split('.')
        value = self._raw_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (in memory only)."""
        keys = key.split('.')
        config = self._raw_config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        
        # Revalidate
        self._config = self._create_typed_config(self._raw_config)
    
    def save_user_override(self, overrides: Dict[str, Any]) -> None:
        """Save user overrides to file."""
        user_config_file = self.config_dir / "user" / "config.yaml"
        user_config_file.parent.mkdir(exist_ok=True)
        
        # Load existing user config
        existing = {}
        if user_config_file.exists():
            existing = self._load_yaml_file(user_config_file)
        
        # Merge overrides
        updated = self._deep_merge(existing, overrides)
        
        # Save
        with open(user_config_file, 'w') as f:
            yaml.dump(updated, f, default_flow_style=False, sort_keys=True)
        
        # Reload
        self.reload()

# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None

def get_config() -> MasterConfig:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager.config

def reload_config() -> MasterConfig:
    """Reload global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager.reload()
```

### 5.2 Error Handling Framework

#### 5.2.1 Current Error Handling Issues

```python
# Current problematic patterns found in codebase:

# Pattern 1: Bare except (89 occurrences)
try:
    result = some_operation()
except:
    print("Error occurred")
    result = None

# Pattern 2: Generic exception catching (156 occurrences)
try:
    result = some_operation()
except Exception as e:
    print(f"Error: {e}")
    return None

# Pattern 3: No error context (234 occurrences)
try:
    process_material(material_id)
except Exception as e:
    print(f"Failed: {e}")  # No context about what failed

# Pattern 4: Silent failures (45 occurrences)
try:
    critical_operation()
except:
    pass  # Silent failure!

# Pattern 5: Inconsistent error handling (178 occurrences)
# Some methods raise, some return None, some print
```

#### 5.2.2 Comprehensive Error Handling Solution

##### Error Hierarchy (errors/exceptions.py)
```python
"""
Comprehensive error hierarchy for workflow system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

class WorkflowError(Exception):
    """
    Base exception for all workflow errors.
    
    Features:
    - Error context preservation
    - Structured error information
    - Recovery hints
    - Error tracking
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None,
                 recovery_hints: Optional[List[str]] = None):
        """
        Initialize workflow error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
            cause: Original exception that caused this error
            recovery_hints: Suggestions for recovery
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.recovery_hints = recovery_hints or []
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None,
            'recovery_hints': self.recovery_hints,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback
        }

# Configuration Errors
class ConfigurationError(WorkflowError):
    """Configuration-related errors."""
    pass

class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Missing required configuration: {config_key}",
            error_code="CONFIG_MISSING",
            details={'config_key': config_key},
            recovery_hints=[
                f"Add '{config_key}' to your configuration file",
                f"Set environment variable MACE_{config_key.upper()}",
                "Check configuration documentation"
            ],
            **kwargs
        )

class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""
    def __init__(self, config_key: str, value: Any, reason: str, **kwargs):
        super().__init__(
            f"Invalid configuration for {config_key}: {reason}",
            error_code="CONFIG_INVALID",
            details={
                'config_key': config_key,
                'value': str(value),
                'reason': reason
            },
            recovery_hints=[
                f"Check valid values for {config_key}",
                "Refer to configuration schema",
                f"Current value: {value}"
            ],
            **kwargs
        )

# File Operation Errors
class FileOperationError(WorkflowError):
    """File operation errors."""
    pass

class FileNotFoundError(FileOperationError):
    """Required file not found."""
    def __init__(self, file_path: str, file_type: str = "file", **kwargs):
        super().__init__(
            f"{file_type.capitalize()} not found: {file_path}",
            error_code="FILE_NOT_FOUND",
            details={
                'file_path': file_path,
                'file_type': file_type
            },
            recovery_hints=[
                f"Check if {file_path} exists",
                f"Verify {file_type} was created successfully",
                "Check file permissions"
            ],
            **kwargs
        )

class FilePermissionError(FileOperationError):
    """Permission denied for file operation."""
    pass

class FileCorruptionError(FileOperationError):
    """File integrity check failed."""
    def __init__(self, file_path: str, expected_hash: str, actual_hash: str, **kwargs):
        super().__init__(
            f"File corruption detected: {file_path}",
            error_code="FILE_CORRUPTED",
            details={
                'file_path': file_path,
                'expected_hash': expected_hash,
                'actual_hash': actual_hash
            },
            recovery_hints=[
                "Re-copy or re-generate the file",
                "Check disk integrity",
                "Verify source file is not corrupted"
            ],
            **kwargs
        )

# Job Submission Errors
class JobSubmissionError(WorkflowError):
    """Job submission errors."""
    pass

class SLURMError(JobSubmissionError):
    """SLURM-specific errors."""
    def __init__(self, command: str, exit_code: int, stderr: str, **kwargs):
        super().__init__(
            f"SLURM command failed: {command}",
            error_code="SLURM_ERROR",
            details={
                'command': command,
                'exit_code': exit_code,
                'stderr': stderr
            },
            recovery_hints=[
                "Check SLURM queue status",
                "Verify account and partition access",
                "Check resource availability",
                f"SLURM error: {stderr}"
            ],
            **kwargs
        )

class ResourceLimitError(JobSubmissionError):
    """Resource limits exceeded."""
    pass

# Calculation Errors
class CalculationError(WorkflowError):
    """Calculation-related errors."""
    pass

class ConvergenceError(CalculationError):
    """Calculation failed to converge."""
    def __init__(self, calc_type: str, material_id: str, cycles: int, **kwargs):
        super().__init__(
            f"{calc_type} calculation failed to converge for {material_id}",
            error_code="CONVERGENCE_FAILED",
            details={
                'calc_type': calc_type,
                'material_id': material_id,
                'cycles': cycles
            },
            recovery_hints=[
                "Increase MAXCYCLE parameter",
                "Adjust mixing parameters (FMIXING)",
                "Check initial geometry",
                "Enable level shifter (LEVSHIFT)"
            ],
            **kwargs
        )

class DependencyError(CalculationError):
    """Required dependency not satisfied."""
    def __init__(self, calc_type: str, dependency: str, **kwargs):
        super().__init__(
            f"{calc_type} requires {dependency} to be completed first",
            error_code="DEPENDENCY_ERROR",
            details={
                'calc_type': calc_type,
                'required_dependency': dependency
            },
            recovery_hints=[
                f"Complete {dependency} calculation first",
                "Check workflow sequence",
                "Verify previous calculations succeeded"
            ],
            **kwargs
        )

# Database Errors
class DatabaseError(WorkflowError):
    """Database-related errors."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Cannot connect to database."""
    pass

class DatabaseLockError(DatabaseError):
    """Database is locked."""
    def __init__(self, db_path: str, **kwargs):
        super().__init__(
            f"Database is locked: {db_path}",
            error_code="DATABASE_LOCKED",
            details={'db_path': db_path},
            recovery_hints=[
                "Wait for other operations to complete",
                "Check for stuck processes",
                "Consider increasing database timeout"
            ],
            **kwargs
        )

# Validation Errors
class ValidationError(WorkflowError):
    """Input validation errors."""
    pass

class InvalidInputError(ValidationError):
    """Invalid input provided."""
    def __init__(self, input_name: str, value: Any, expected: str, **kwargs):
        super().__init__(
            f"Invalid {input_name}: expected {expected}, got {type(value).__name__}",
            error_code="INVALID_INPUT",
            details={
                'input_name': input_name,
                'value': str(value),
                'expected': expected
            },
            **kwargs
        )

# Timeout Errors
class TimeoutError(WorkflowError):
    """Operation timed out."""
    def __init__(self, operation: str, timeout: int, **kwargs):
        super().__init__(
            f"{operation} timed out after {timeout} seconds",
            error_code="TIMEOUT",
            details={
                'operation': operation,
                'timeout': timeout
            },
            recovery_hints=[
                f"Increase timeout for {operation}",
                "Check if operation is stuck",
                "Verify system resources"
            ],
            **kwargs
        )
```

##### Error Handler (errors/handler.py)
```python
"""
Centralized error handling and recovery.
"""

import logging
from typing import Dict, Any, Optional, Callable, Type, List
from functools import wraps
import asyncio
from datetime import datetime

from .exceptions import WorkflowError
from .recovery import ErrorRecoveryStrategy

class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    
    Features:
    - Automatic error categorization
    - Recovery strategy selection
    - Error tracking and metrics
    - Notification system
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('ErrorHandler')
        self.recovery_strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        self.error_metrics: Dict[str, int] = {}
        self.notification_handlers: List[Callable] = []
        
        # Register default strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        from .recovery import (
            RetryStrategy, ExponentialBackoffStrategy,
            FailoverStrategy, CircuitBreakerStrategy
        )
        
        # File errors: retry with backoff
        self.register_strategy(
            FileOperationError,
            ExponentialBackoffStrategy(max_retries=5, base_delay=1)
        )
        
        # Database errors: circuit breaker
        self.register_strategy(
            DatabaseError,
            CircuitBreakerStrategy(failure_threshold=5, recovery_timeout=60)
        )
        
        # Job submission: retry with delay
        self.register_strategy(
            JobSubmissionError,
            RetryStrategy(max_retries=3, delay=30)
        )
    
    def register_strategy(self, 
                         error_type: Type[Exception],
                         strategy: ErrorRecoveryStrategy):
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = strategy
    
    def register_notification(self, handler: Callable[[WorkflowError], None]):
        """Register notification handler."""
        self.notification_handlers.append(handler)
    
    def handle_error(self, 
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    can_recover: bool = True) -> Optional[Any]:
        """
        Handle error with appropriate strategy.
        
        Args:
            error: The exception to handle
            context: Additional error context
            can_recover: Whether recovery should be attempted
            
        Returns:
            Recovery result or None
        """
        # Track error metrics
        error_type = type(error).__name__
        self.error_metrics[error_type] = self.error_metrics.get(error_type, 0) + 1
        
        # Create workflow error if needed
        if not isinstance(error, WorkflowError):
            error = WorkflowError(
                message=str(error),
                cause=error,
                details=context or {}
            )
        
        # Log error
        self.logger.error(
            f"{error.error_code}: {error.message}",
            extra={
                'error_details': error.to_dict(),
                'context': context
            }
        )
        
        # Notify handlers
        for handler in self.notification_handlers:
            try:
                handler(error)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {e}")
        
        # Attempt recovery
        if can_recover:
            strategy = self._get_recovery_strategy(error)
            if strategy:
                try:
                    return strategy.recover(error, context)
                except Exception as recovery_error:
                    self.logger.error(
                        f"Recovery failed: {recovery_error}",
                        exc_info=True
                    )
        
        # Re-raise if no recovery
        raise error
    
    def _get_recovery_strategy(self, 
                              error: Exception) -> Optional[ErrorRecoveryStrategy]:
        """Get appropriate recovery strategy for error."""
        # Check exact type match
        error_type = type(error)
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # Check parent classes
        for base_type, strategy in self.recovery_strategies.items():
            if isinstance(error, base_type):
                return strategy
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error metrics."""
        return {
            'error_counts': self.error_metrics.copy(),
            'total_errors': sum(self.error_metrics.values()),
            'error_types': list(self.error_metrics.keys())
        }

# Decorator for automatic error handling
def handle_errors(
    default_return: Any = None,
    can_recover: bool = True,
    log_errors: bool = True,
    error_handler: Optional[ErrorHandler] = None
):
    """
    Decorator for automatic error handling.
    
    Usage:
        @handle_errors(default_return=None)
        def risky_operation():
            # Code that might fail
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                try:
                    return handler.handle_error(e, context, can_recover)
                except Exception:
                    if log_errors:
                        handler.logger.error(
                            f"Unrecoverable error in {func.__name__}",
                            exc_info=True
                        )
                    return default_return
        
        return wrapper
    return decorator

# Async version
def handle_errors_async(
    default_return: Any = None,
    can_recover: bool = True,
    log_errors: bool = True,
    error_handler: Optional[ErrorHandler] = None
):
    """Async version of error handling decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                try:
                    result = handler.handle_error(e, context, can_recover)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                except Exception:
                    if log_errors:
                        handler.logger.error(
                            f"Unrecoverable error in {func.__name__}",
                            exc_info=True
                        )
                    return default_return
        
        return wrapper
    return decorator
```

### 5.3 Logging Integration

#### 5.3.1 Current Logging Issues

```
Issue                              Count    Example
-------------------------------  -------  -----------------------------------------
Print statements                      523  print(f"Error: {e}")
No logging configuration              N/A  No centralized logging
Inconsistent log formatting          234  Various formats used
No log rotation                      N/A  Logs grow indefinitely
No structured logging                N/A  Hard to parse logs
No log aggregation                   N/A  Logs scattered across files
Missing context in logs              412  No request IDs or correlation
```

#### 5.3.2 Comprehensive Logging Solution

##### Logging Configuration (logging/config.py)
```python
"""
Centralized logging configuration.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self.request_id = None
        self.user_id = None
        self.workflow_id = None
    
    def filter(self, record):
        """Add context to log record."""
        record.request_id = self.request_id or 'no-request-id'
        record.user_id = self.user_id or 'system'
        record.workflow_id = self.workflow_id or 'no-workflow'
        record.hostname = getattr(record, 'hostname', 'localhost')
        return True

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'request_id': getattr(record, 'request_id', None),
            'user_id': getattr(record, 'user_id', None),
            'workflow_id': getattr(record, 'workflow_id', None),
            'hostname': getattr(record, 'hostname', None)
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'getMessage']:
                log_data[key] = value
        
        return json.dumps(log_data)

class LogManager:
    """
    Centralized log management.
    
    Features:
    - Multiple output handlers (file, console, syslog)
    - Log rotation
    - Structured logging
    - Context injection
    - Performance logging
    """
    
    def __init__(self, 
                 app_name: str = "mace_workflow",
                 log_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.app_name = app_name
        self.log_dir = log_dir or Path("logs")
        self.config = config or {}
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Context filter
        self.context_filter = ContextFilter()
        
        # Configure root logger
        self._configure_root_logger()
        
        # Configure module loggers
        self._configure_module_loggers()
    
    def _configure_root_logger(self):
        """Configure root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Add handlers
        root_logger.addHandler(self._create_console_handler())
        root_logger.addHandler(self._create_file_handler())
        root_logger.addHandler(self._create_error_file_handler())
        
        # Add context filter
        for handler in root_logger.handlers:
            handler.addFilter(self.context_filter)
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Use colored output if available
        try:
            import colorlog
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        except ImportError:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> logging.Handler:
        """Create rotating file handler."""
        log_file = self.log_dir / f"{self.app_name}.log"
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        handler.setLevel(logging.DEBUG)
        
        # Use JSON formatter for structured logging
        if self.config.get('structured_logging', True):
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '[%(request_id)s] - %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_error_file_handler(self) -> logging.Handler:
        """Create error-only file handler."""
        error_file = self.log_dir / f"{self.app_name}_errors.log"
        
        handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        handler.setLevel(logging.ERROR)
        handler.setFormatter(JSONFormatter())
        
        return handler
    
    def _configure_module_loggers(self):
        """Configure specific module loggers."""
        # Database logger
        db_logger = logging.getLogger('mace.database')
        db_logger.setLevel(logging.WARNING)
        
        # Workflow logger
        workflow_logger = logging.getLogger('mace.workflow')
        workflow_logger.setLevel(logging.INFO)
        
        # Performance logger
        perf_logger = logging.getLogger('mace.performance')
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'performance.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        perf_handler.setFormatter(JSONFormatter())
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
    
    def set_context(self, **kwargs):
        """Set logging context."""
        for key, value in kwargs.items():
            setattr(self.context_filter, key, value)
    
    def clear_context(self):
        """Clear logging context."""
        self.context_filter.request_id = None
        self.context_filter.user_id = None
        self.context_filter.workflow_id = None
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger with context."""
        logger = logging.getLogger(name)
        return logger

# Performance logging utilities
class PerformanceLogger:
    """Log performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger('mace.performance')
    
    def log_operation(self, 
                     operation: str,
                     duration: float,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log operation performance."""
        self.logger.info(
            f"Performance: {operation}",
            extra={
                'operation': operation,
                'duration_ms': duration * 1000,
                'metadata': metadata or {}
            }
        )
    
    def log_metric(self,
                  metric_name: str,
                  value: float,
                  unit: str = 'count',
                  tags: Optional[Dict[str, str]] = None):
        """Log a metric."""
        self.logger.info(
            f"Metric: {metric_name}",
            extra={
                'metric_name': metric_name,
                'value': value,
                'unit': unit,
                'tags': tags or {}
            }
        )

# Context manager for logging context
class LogContext:
    """Context manager for logging context."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.log_manager = LogManager()
    
    def __enter__(self):
        """Enter context."""
        self.log_manager.set_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.log_manager.clear_context()

# Usage examples:
# with LogContext(request_id=str(uuid.uuid4()), workflow_id='wf-123'):
#     logger.info("Processing workflow")
```

### 5.4 Testing Framework

#### 5.4.1 Current Testing Issues

```
Issue                              Status    Coverage
-------------------------------  --------  ----------
No unit tests                       ❌          0%
No integration tests                ❌          0%
No test fixtures                    ❌         N/A
No mocking framework                ❌         N/A
No test automation                  ❌         N/A
No performance tests                ❌         N/A
No test documentation               ❌         N/A
```

#### 5.4.2 Comprehensive Testing Solution

##### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # pytest configuration and fixtures
├── unit/
│   ├── __init__.py
│   ├── test_planner/
│   │   ├── test_core.py
│   │   ├── test_interactive.py
│   │   ├── test_cif_converter.py
│   │   └── test_expert_modes.py
│   ├── test_engine/
│   │   ├── test_orchestrator.py
│   │   ├── test_file_manager.py
│   │   └── test_job_submitter.py
│   ├── test_executor/
│   │   └── test_step_executor.py
│   └── test_monitor/
│       └── test_workflow_monitor.py
├── integration/
│   ├── __init__.py
│   ├── test_workflow_execution.py
│   ├── test_cif_to_properties.py
│   └── test_error_recovery.py
├── performance/
│   ├── __init__.py
│   ├── test_large_workflows.py
│   └── test_database_performance.py
├── fixtures/
│   ├── sample_cifs/
│   ├── sample_d12s/
│   ├── sample_outputs/
│   └── mock_data.py
└── utils/
    ├── __init__.py
    ├── test_helpers.py
    └── assertions.py
```

##### Test Configuration (tests/conftest.py)
```python
"""
pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sqlite3
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mace.workflow.planner import WorkflowPlanner
from mace.workflow.engine import WorkflowEngine
from mace.workflow.executor import WorkflowExecutor
from mace.database.materials import MaterialDatabase

# Test configuration
TEST_DB = ":memory:"
TEST_TIMEOUT = 30

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_db():
    """Create mock database."""
    db = Mock(spec=MaterialDatabase)
    
    # Mock common methods
    db.get_material.return_value = {
        'material_id': 'test_material',
        'formula': 'C',
        'space_group': 'Fm-3m'
    }
    
    db.get_workflow.return_value = {
        'workflow_id': 'test_workflow',
        'status': 'active',
        'created_at': datetime.now().isoformat()
    }
    
    db.get_calculations_for_material.return_value = []
    
    return db

@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    
    # Create schema
    conn.executescript("""
        CREATE TABLE materials (
            material_id TEXT PRIMARY KEY,
            formula TEXT,
            space_group TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE workflows (
            workflow_id TEXT PRIMARY KEY,
            status TEXT,
            config_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE calculations (
            calc_id TEXT PRIMARY KEY,
            material_id TEXT,
            workflow_id TEXT,
            calc_type TEXT,
            status TEXT,
            job_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materials(material_id),
            FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
        );
        
        CREATE TABLE files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            calc_id TEXT,
            file_name TEXT,
            file_path TEXT,
            file_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (calc_id) REFERENCES calculations(calc_id)
        );
    """)
    
    yield conn
    conn.close()

@pytest.fixture
def workflow_planner(temp_dir, mock_db):
    """Create WorkflowPlanner instance."""
    planner = WorkflowPlanner(base_dir=temp_dir, db_path=":memory:")
    planner.db = mock_db
    return planner

@pytest.fixture
def workflow_engine(temp_dir, mock_db):
    """Create WorkflowEngine instance."""
    engine = WorkflowEngine(base_dir=temp_dir, db_path=":memory:")
    engine.db = mock_db
    return engine

@pytest.fixture
def sample_cif_content():
    """Sample CIF file content."""
    return """
data_diamond
_chemical_formula_structural     C
_chemical_formula_sum            "C 1"
_cell_length_a                   3.567
_cell_length_b                   3.567
_cell_length_c                   3.567
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_space_group_name_H-M_alt        "F d -3 m"
_space_group_IT_number           227

loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
C1 C 0.00000 0.00000 0.00000 1.0
"""

@pytest.fixture
def sample_d12_content():
    """Sample D12 file content."""
    return """DIAMOND
CRYSTAL
0 0 0
227
3.567
1
6 0.0 0.0 0.0
END
99 0
END
DFT
B3LYP
END
SHRINK
8 8
TOLINTEG
7 7 7 7 14
TOLDEE
7
MAXCYCLE
100
FMIXING
30
END
"""

@pytest.fixture
def mock_slurm_commands(monkeypatch):
    """Mock SLURM commands."""
    def mock_sbatch(args, **kwargs):
        """Mock sbatch command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Submitted batch job 12345\n"
        mock_result.stderr = ""
        return mock_result
    
    def mock_squeue(args, **kwargs):
        """Mock squeue command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "12345 general job_name user R 0:01 1 node001\n"
        mock_result.stderr = ""
        return mock_result
    
    monkeypatch.setattr("subprocess.run", lambda args, **kwargs: {
        'sbatch': mock_sbatch,
        'squeue': mock_squeue
    }.get(args[0], Mock())(args, **kwargs))

# Performance testing fixtures
@pytest.fixture
def large_workflow_config():
    """Configuration for large workflow testing."""
    return {
        'workflow_id': 'perf_test_workflow',
        'input_type': 'cif',
        'workflow_sequence': ['OPT', 'SP', 'BAND', 'DOSS'],
        'materials': [f'material_{i}' for i in range(100)]
    }

# Custom markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
```

##### Example Unit Test (tests/unit/test_planner/test_core.py)
```python
"""
Unit tests for WorkflowPlanner core functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mace.workflow.planner import WorkflowPlanner

class TestWorkflowPlanner:
    """Test WorkflowPlanner class."""
    
    def test_initialization(self, temp_dir):
        """Test planner initialization."""
        planner = WorkflowPlanner(base_dir=temp_dir)
        
        assert planner.base_dir == temp_dir
        assert planner.db is not None
        assert (temp_dir / "workflow_configs").exists()
        assert (temp_dir / "workflow_inputs").exists()
    
    def test_plan_workflow_interactive(self, workflow_planner, monkeypatch):
        """Test interactive workflow planning."""
        # Mock user inputs
        inputs = iter([
            "1",  # Input type: CIF
            "1",  # CIF level: Basic
            "1",  # Workflow template: basic_opt
            "y",  # Confirm
        ])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        
        # Mock CIF files
        workflow_planner.cif_converter.find_cif_files = Mock(
            return_value=[Path("test.cif")]
        )
        workflow_planner.cif_converter.batch_convert = Mock(
            return_value=[Path("test.d12")]
        )
        
        # Execute
        config = workflow_planner.plan_workflow(mode="interactive")
        
        # Verify
        assert config['input_type'] == 'cif'
        assert config['workflow_sequence'] == ['OPT']
        assert 'workflow_id' in config
        assert 'config_file' in config
    
    def test_plan_workflow_template(self, workflow_planner):
        """Test template-based workflow planning."""
        config = workflow_planner.plan_workflow(
            mode="template",
            template="full_electronic"
        )
        
        assert config['workflow_sequence'] == ['OPT', 'SP', 'BAND', 'DOSS']
    
    def test_quick_opt_workflow(self, workflow_planner, temp_dir):
        """Test quick optimization workflow."""
        # Create dummy D12 file
        d12_file = temp_dir / "test.d12"
        d12_file.write_text("DUMMY D12 CONTENT")
        
        config = workflow_planner.quick_opt_workflow([d12_file])
        
        assert config['workflow_sequence'] == ['OPT']
        assert str(d12_file) in config['input_files']
    
    @pytest.mark.parametrize("calc_type", ["OPT", "SP", "FREQ", "BAND", "DOSS"])
    def test_configure_expert_calculation(self, workflow_planner, calc_type, temp_dir):
        """Test expert mode configuration for different calculation types."""
        # Create template file
        template = temp_dir / f"template.{'d12' if calc_type in ['OPT', 'SP', 'FREQ'] else 'd3'}"
        template.write_text("TEMPLATE CONTENT")
        
        # Mock expert manager
        workflow_planner.expert_manager.configure = Mock(
            return_value=temp_dir / f"configured.{template.suffix}"
        )
        
        result = workflow_planner.configure_expert_calculation(
            calc_type,
            template,
            use_defaults=True
        )
        
        assert result is not None
        workflow_planner.expert_manager.configure.assert_called_once()
    
    def test_workflow_validation(self, workflow_planner):
        """Test workflow configuration validation."""
        # Valid configuration
        valid_config = {
            'workflow_id': 'test',
            'workflow_sequence': ['OPT', 'SP'],
            'input_files': ['test.d12']
        }
        
        is_valid, errors = workflow_planner.validator.validate(valid_config)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = {
            'workflow_sequence': ['SP', 'OPT'],  # Wrong order
            'input_files': []  # No inputs
        }
        
        is_valid, errors = workflow_planner.validator.validate(invalid_config)
        assert not is_valid
        assert len(errors) > 0
    
    def test_isolated_mode(self, temp_dir):
        """Test planner with isolation enabled."""
        from mace.workflow.context import WorkflowIsolationContext
        
        isolation_context = WorkflowIsolationContext(temp_dir)
        planner = WorkflowPlanner(
            base_dir=temp_dir,
            isolated=True,
            isolation_context=isolation_context
        )
        
        assert planner.isolated is True
        assert planner.isolation_context is not None
    
    @patch('mace.workflow.planner.MaterialDatabase')
    def test_database_error_handling(self, mock_db_class, temp_dir):
        """Test database error handling."""
        # Mock database to raise error
        mock_db = Mock()
        mock_db.create_workflow.side_effect = Exception("Database error")
        mock_db_class.return_value = mock_db
        
        planner = WorkflowPlanner(base_dir=temp_dir)
        
        with pytest.raises(Exception) as exc_info:
            planner._register_workflow({'workflow_id': 'test'})
        
        assert "Database error" in str(exc_info.value)
```

This comprehensive architecture improvement plan addresses all major issues in the workflow module, providing robust solutions for configuration management, error handling, logging, and testing.