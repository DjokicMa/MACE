# MACE Workflow Module - Detailed Refactoring Plan
# Section 7: Risk Analysis and Mitigation

## 7. Risk Analysis and Mitigation

### 7.1 Comprehensive Risk Assessment

#### 7.1.1 Risk Matrix Overview

```
Risk Category         Count    High Impact    Medium Impact    Low Impact
------------------  -------  -------------  ---------------  ------------
Technical Risks          15              6                5             4
Operational Risks        12              4                6             2
Resource Risks            8              3                4             1
User Impact Risks        10              5                3             2
Integration Risks         9              4                4             1
------------------  -------  -------------  ---------------  ------------
Total Risks              54             22               22            10

Risk Severity Distribution:
- Critical (High Probability + High Impact): 8 risks (14.8%)
- Major (Medium/High combination): 18 risks (33.3%)
- Moderate (Medium/Medium): 20 risks (37.0%)
- Minor (Low combinations): 8 risks (14.8%)
```

### 7.2 Technical Risks

#### 7.2.1 Breaking Existing Workflows

**Risk Description:**
Refactoring may break existing workflow configurations and running calculations.

**Detailed Analysis:**
```python
# Potential breaking changes:
1. Import path changes:
   OLD: from mace.workflow import WorkflowPlanner
   NEW: from mace.workflow.planner.core import WorkflowPlanner

2. Method signature changes:
   OLD: planner.configure_expert(calc_type, template)
   NEW: planner.expert_manager.configure(calc_type, template, options)

3. Configuration format changes:
   OLD: Hard-coded values throughout
   NEW: Configuration-driven approach

4. Database schema changes:
   OLD: Direct SQL queries
   NEW: ORM-based approach
```

**Impact Assessment:**
- Probability: High (80%)
- Impact: High (affects all users)
- Risk Score: 8/10

**Mitigation Strategies:**

1. **Compatibility Layer**
```python
# mace/workflow/compat.py
"""Compatibility layer for smooth migration."""

import warnings
from typing import Any, Dict, Optional
from pathlib import Path

# Import new modules
from mace.workflow.planner.core import WorkflowPlanner as NewWorkflowPlanner
from mace.workflow.engine.core import WorkflowEngine as NewWorkflowEngine

class WorkflowPlanner(NewWorkflowPlanner):
    """Compatibility wrapper for WorkflowPlanner."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Direct import of WorkflowPlanner is deprecated. "
            "Use 'from mace.workflow.planner import WorkflowPlanner' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
    
    def configure_expert(self, calc_type: str, template: Path, **kwargs):
        """Legacy method for expert configuration."""
        warnings.warn(
            "configure_expert is deprecated. "
            "Use expert_manager.configure() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.expert_manager.configure(calc_type, template, kwargs)

# Maintain old import paths
__all__ = ['WorkflowPlanner', 'WorkflowEngine', 'WorkflowExecutor']
```

2. **Version Detection**
```python
# mace/workflow/version.py
"""Version management for backward compatibility."""

import json
from pathlib import Path
from typing import Dict, Any

CURRENT_VERSION = "2.0"
COMPATIBLE_VERSIONS = ["1.0", "1.1", "1.2"]

def detect_workflow_version(config: Dict[str, Any]) -> str:
    """Detect workflow configuration version."""
    # Check explicit version
    if 'version' in config:
        return config['version']
    
    # Check for v1 indicators
    if 'hard_coded_account' in config:
        return "1.0"
    
    # Check for v1.1 indicators
    if 'workflow_templates' in config and 'version' not in config:
        return "1.1"
    
    # Default to current
    return CURRENT_VERSION

def migrate_configuration(config: Dict[str, Any], 
                         from_version: str,
                         to_version: str = CURRENT_VERSION) -> Dict[str, Any]:
    """Migrate configuration between versions."""
    if from_version == to_version:
        return config
    
    # Apply migrations sequentially
    migrations = {
        ("1.0", "1.1"): migrate_v10_to_v11,
        ("1.1", "1.2"): migrate_v11_to_v12,
        ("1.2", "2.0"): migrate_v12_to_v20
    }
    
    current = from_version
    migrated = config.copy()
    
    while current != to_version:
        next_version = get_next_version(current)
        migration_key = (current, next_version)
        
        if migration_key in migrations:
            migrated = migrations[migration_key](migrated)
        
        current = next_version
    
    return migrated
```

3. **Gradual Migration Path**
```python
# mace/workflow/migration.py
"""Migration utilities for workflow refactoring."""

class WorkflowMigrator:
    """Handles migration of workflows to new structure."""
    
    def __init__(self, db_path: str):
        self.db = MaterialDatabase(db_path)
        self.logger = logging.getLogger('WorkflowMigrator')
    
    def analyze_impact(self) -> Dict[str, Any]:
        """Analyze impact of migration on existing workflows."""
        active_workflows = self.db.get_active_workflows()
        
        impact = {
            'total_workflows': len(active_workflows),
            'requires_migration': 0,
            'auto_migratable': 0,
            'manual_intervention': 0,
            'affected_calculations': 0
        }
        
        for workflow in active_workflows:
            version = detect_workflow_version(workflow)
            if version != CURRENT_VERSION:
                impact['requires_migration'] += 1
                
                if self._can_auto_migrate(workflow):
                    impact['auto_migratable'] += 1
                else:
                    impact['manual_intervention'] += 1
                
                # Count affected calculations
                calcs = self.db.get_calculations_for_workflow(workflow['workflow_id'])
                impact['affected_calculations'] += len(calcs)
        
        return impact
    
    def migrate_workflow(self, workflow_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Migrate a single workflow to new structure."""
        workflow = self.db.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # Detect version
        current_version = detect_workflow_version(workflow)
        
        # Create migration plan
        plan = {
            'workflow_id': workflow_id,
            'current_version': current_version,
            'target_version': CURRENT_VERSION,
            'steps': []
        }
        
        if current_version == CURRENT_VERSION:
            plan['status'] = 'up_to_date'
            return plan
        
        # Migrate configuration
        try:
            new_config = migrate_configuration(
                workflow['config'],
                current_version,
                CURRENT_VERSION
            )
            
            plan['steps'].append({
                'action': 'update_configuration',
                'status': 'planned' if dry_run else 'executed'
            })
            
            if not dry_run:
                self.db.update_workflow_config(workflow_id, new_config)
            
            plan['status'] = 'success'
            
        except Exception as e:
            plan['status'] = 'failed'
            plan['error'] = str(e)
        
        return plan
```

#### 7.2.2 Performance Degradation

**Risk Description:**
Modularization might introduce performance overhead.

**Impact Assessment:**
- Probability: Medium (40%)
- Impact: Medium
- Risk Score: 5/10

**Mitigation Strategies:**

1. **Performance Benchmarking Suite**
```python
# tests/performance/benchmark.py
"""Performance benchmarking for workflow operations."""

import time
import memory_profiler
import cProfile
from typing import Dict, Any, List
import statistics

class PerformanceBenchmark:
    """Benchmark workflow performance."""
    
    def __init__(self):
        self.results = {}
        self.baselines = self._load_baselines()
    
    def benchmark_import_time(self) -> Dict[str, float]:
        """Benchmark module import times."""
        import_times = {}
        
        modules = [
            'mace.workflow.planner',
            'mace.workflow.engine',
            'mace.workflow.executor',
            'mace.workflow.monitor'
        ]
        
        for module in modules:
            start = time.perf_counter()
            __import__(module)
            end = time.perf_counter()
            import_times[module] = (end - start) * 1000  # ms
        
        return import_times
    
    def benchmark_workflow_creation(self, n_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark workflow creation performance."""
        from mace.workflow.planner import WorkflowPlanner
        
        planner = WorkflowPlanner()
        times = []
        
        for i in range(n_iterations):
            start = time.perf_counter()
            config = planner.quick_opt_workflow([f"test_{i}.d12"])
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
    
    @memory_profiler.profile
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        # Import all modules
        from mace.workflow import WorkflowPlanner, WorkflowEngine, WorkflowExecutor
        
        # Create instances
        planner = WorkflowPlanner()
        engine = WorkflowEngine()
        executor = WorkflowExecutor()
        
        # Perform operations
        config = planner.quick_opt_workflow(["test.d12"])
        
        # Get memory usage
        return memory_profiler.memory_usage()
    
    def compare_with_baseline(self) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        current = {
            'import_time': self.benchmark_import_time(),
            'workflow_creation': self.benchmark_workflow_creation(),
            'memory_usage': self.benchmark_memory_usage()
        }
        
        comparison = {}
        
        # Compare import times
        for module, time in current['import_time'].items():
            baseline = self.baselines.get('import_time', {}).get(module, time)
            comparison[f"{module}_import"] = {
                'current': time,
                'baseline': baseline,
                'regression': time > baseline * 1.1  # 10% threshold
            }
        
        return comparison
```

2. **Lazy Loading Implementation**
```python
# mace/workflow/utils/lazy_loader.py
"""Lazy loading utilities for performance optimization."""

import importlib
import sys
from typing import Any, Optional

class LazyModule:
    """Lazy load module on first access."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Optional[Any] = None
    
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
            # Replace self in sys.modules
            sys.modules[self.module_name] = self._module
        return getattr(self._module, name)
    
    def __dir__(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return dir(self._module)

# Usage in __init__.py
def _lazy_import(name: str, module_path: str):
    """Create lazy import."""
    globals()[name] = LazyModule(module_path)

# Lazy load heavy modules
_lazy_import('numpy', 'numpy')
_lazy_import('matplotlib', 'matplotlib')
_lazy_import('ase', 'ase')
```

#### 7.2.3 Data Corruption Risk

**Risk Description:**
Refactoring database interactions might corrupt existing data.

**Impact Assessment:**
- Probability: Low (20%)
- Impact: Critical
- Risk Score: 7/10

**Mitigation Strategies:**

1. **Database Backup Strategy**
```python
# mace/workflow/utils/backup.py
"""Database backup utilities."""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
import hashlib

class DatabaseBackup:
    """Handle database backups before risky operations."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.backup_dir = self.db_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, description: str = "") -> Path:
        """Create timestamped backup of database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}_{description}.db"
        backup_path = self.backup_dir / backup_name
        
        # Create backup
        shutil.copy2(self.db_path, backup_path)
        
        # Verify backup integrity
        if not self._verify_backup(backup_path):
            raise RuntimeError(f"Backup verification failed: {backup_path}")
        
        # Create metadata
        self._create_backup_metadata(backup_path, description)
        
        return backup_path
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        # Check file exists and size matches
        if not backup_path.exists():
            return False
        
        if backup_path.stat().st_size != self.db_path.stat().st_size:
            return False
        
        # Verify can open and query
        try:
            conn = sqlite3.connect(backup_path)
            conn.execute("SELECT COUNT(*) FROM sqlite_master")
            conn.close()
            return True
        except:
            return False
    
    def restore_backup(self, backup_path: Path) -> None:
        """Restore database from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Create safety backup of current state
        self.create_backup("pre_restore")
        
        # Restore
        shutil.copy2(backup_path, self.db_path)
        
        # Verify restore
        if not self._verify_backup(self.db_path):
            raise RuntimeError("Restore verification failed")
```

2. **Transaction Safety**
```python
# mace/workflow/database/safe_operations.py
"""Safe database operations with automatic rollback."""

from contextlib import contextmanager
import sqlite3
from typing import Any, Generator

class SafeDatabaseOperations:
    """Ensure database operations are atomic and safe."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup = DatabaseBackup(db_path)
    
    @contextmanager
    def safe_transaction(self, 
                        backup_description: str = "") -> Generator[sqlite3.Connection, None, None]:
        """Create safe transaction with automatic rollback."""
        # Create backup before transaction
        backup_path = self.backup.create_backup(backup_description)
        
        conn = sqlite3.connect(self.db_path)
        conn.isolation_level = None  # Manual transaction control
        
        try:
            conn.execute("BEGIN EXCLUSIVE")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            # Restore from backup if critical error
            if self._is_critical_error(e):
                self.backup.restore_backup(backup_path)
            raise
        finally:
            conn.close()
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if error requires backup restore."""
        critical_errors = [
            "database disk image is malformed",
            "database corruption",
            "constraint failed"
        ]
        return any(err in str(error).lower() for err in critical_errors)
```

### 7.3 Operational Risks

#### 7.3.1 Resource Availability

**Risk Description:**
Key developers might not be available throughout the refactoring.

**Impact Assessment:**
- Probability: Medium (50%)
- Impact: High
- Risk Score: 6/10

**Mitigation Strategies:**

1. **Knowledge Documentation**
```markdown
# Developer Handover Document

## Critical Knowledge Areas

### 1. Workflow State Machine
The workflow progression follows a specific state machine...

### 2. Database Schema Relationships
Key relationships between tables...

### 3. SLURM Integration Points
Critical integration points with SLURM...

### 4. Error Recovery Mechanisms
How error recovery works...

## Code Navigation Guide

### Key Files and Their Purpose
- `planner.py`: Main workflow planning logic
- `engine.py`: Workflow execution engine
- `executor.py`: Step-by-step execution

### Common Debugging Scenarios
1. Workflow stuck in pending state
2. Calculation files not found
3. Database lock issues

## Testing Procedures

### Running Test Suite
```bash
pytest tests/unit/
pytest tests/integration/
```

### Manual Testing Checklist
- [ ] Create new workflow
- [ ] Execute full workflow
- [ ] Test error recovery
- [ ] Verify monitoring
```

2. **Pair Programming Schedule**
```python
# development_schedule.py
"""Development schedule for knowledge sharing."""

schedule = {
    "week_1": {
        "primary": "Developer A",
        "secondary": "Developer B",
        "tasks": ["Configuration management", "Error handling"]
    },
    "week_2": {
        "primary": "Developer B", 
        "secondary": "Developer C",
        "tasks": ["Monitoring consolidation", "Testing"]
    },
    # Rotate developers for knowledge sharing
}
```

#### 7.3.2 Deployment Failures

**Risk Description:**
Production deployment might fail or cause outages.

**Impact Assessment:**
- Probability: Medium (40%)
- Impact: High
- Risk Score: 6/10

**Mitigation Strategies:**

1. **Canary Deployment**
```python
# deployment/canary.py
"""Canary deployment strategy."""

class CanaryDeployment:
    """Manage canary deployment of workflow changes."""
    
    def __init__(self, load_balancer):
        self.load_balancer = load_balancer
        self.metrics_collector = MetricsCollector()
    
    def deploy_canary(self, 
                     new_version: str,
                     canary_percentage: float = 0.1) -> bool:
        """Deploy new version to small percentage of traffic."""
        # Route 10% traffic to new version
        self.load_balancer.set_routing({
            'v1': 1 - canary_percentage,
            'v2': canary_percentage
        })
        
        # Monitor for 1 hour
        metrics = self.metrics_collector.collect_metrics(duration=3600)
        
        # Check success criteria
        if self._check_canary_health(metrics):
            return True
        else:
            # Rollback
            self.load_balancer.set_routing({'v1': 1.0, 'v2': 0.0})
            return False
    
    def _check_canary_health(self, metrics: Dict[str, Any]) -> bool:
        """Check if canary deployment is healthy."""
        # Error rate threshold
        if metrics['error_rate'] > 0.01:  # 1%
            return False
        
        # Performance threshold
        if metrics['p95_latency'] > metrics['baseline_p95'] * 1.2:  # 20% degradation
            return False
        
        # Success rate threshold
        if metrics['success_rate'] < 0.99:  # 99%
            return False
        
        return True
```

2. **Rollback Automation**
```bash
#!/bin/bash
# automated_rollback.sh

set -e

DEPLOYMENT_ID=$1
HEALTH_CHECK_URL="http://localhost:8080/health"
MAX_RETRIES=5
RETRY_DELAY=10

echo "Starting deployment health check for: $DEPLOYMENT_ID"

# Health check function
health_check() {
    response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_CHECK_URL)
    if [ $response -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Attempt health checks
for i in $(seq 1 $MAX_RETRIES); do
    echo "Health check attempt $i..."
    
    if health_check; then
        echo "✓ Health check passed"
        exit 0
    else
        echo "✗ Health check failed"
        
        if [ $i -eq $MAX_RETRIES ]; then
            echo "Maximum retries reached. Initiating rollback..."
            ./rollback.sh $DEPLOYMENT_ID
            exit 1
        fi
        
        sleep $RETRY_DELAY
    fi
done
```

### 7.4 User Impact Risks

#### 7.4.1 Learning Curve

**Risk Description:**
Users need to learn new interfaces and patterns.

**Impact Assessment:**
- Probability: High (90%)
- Impact: Medium
- Risk Score: 6/10

**Mitigation Strategies:**

1. **Comprehensive Training Materials**
```python
# training/interactive_tutorial.py
"""Interactive tutorial for new workflow system."""

class WorkflowTutorial:
    """Interactive tutorial system."""
    
    def __init__(self):
        self.lessons = [
            self.lesson_1_basic_workflow,
            self.lesson_2_configuration,
            self.lesson_3_monitoring,
            self.lesson_4_error_handling
        ]
        self.progress = {}
    
    def start_tutorial(self, user_id: str):
        """Start interactive tutorial."""
        print("Welcome to MACE Workflow v2 Tutorial!")
        print("="*50)
        
        for i, lesson in enumerate(self.lessons, 1):
            print(f"\nLesson {i}: {lesson.__name__}")
            
            try:
                lesson()
                self.progress[user_id] = i
                
                if not self._continue_prompt():
                    break
                    
            except Exception as e:
                print(f"Error in lesson: {e}")
                if not self._retry_prompt():
                    break
    
    def lesson_1_basic_workflow(self):
        """Lesson 1: Creating a basic workflow."""
        print("\nIn this lesson, you'll learn to create a basic workflow.")
        print("Let's start with a simple optimization workflow...")
        
        # Interactive demo
        from mace.workflow.planner import WorkflowPlanner
        planner = WorkflowPlanner()
        
        print("\nStep 1: Initialize the planner")
        print(">>> planner = WorkflowPlanner()")
        
        print("\nStep 2: Create a quick optimization workflow")
        print(">>> config = planner.quick_opt_workflow(['diamond.d12'])")
        
        # Show result
        config = planner.quick_opt_workflow(['example.d12'])
        print(f"\nWorkflow created: {config['workflow_id']}")
```

2. **Migration Assistant**
```python
# migration/assistant.py
"""Interactive migration assistant."""

class MigrationAssistant:
    """Help users migrate from v1 to v2."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.suggestions = []
    
    def analyze_user_code(self, code_path: Path) -> List[Dict[str, Any]]:
        """Analyze user code for migration issues."""
        issues = []
        
        for file_path in code_path.rglob("*.py"):
            content = file_path.read_text()
            
            # Check for old imports
            old_imports = self._find_old_imports(content)
            for imp in old_imports:
                issues.append({
                    'file': str(file_path),
                    'line': imp['line'],
                    'issue': 'deprecated_import',
                    'old': imp['import'],
                    'new': self._get_new_import(imp['import']),
                    'severity': 'warning'
                })
            
            # Check for deprecated methods
            deprecated_calls = self._find_deprecated_calls(content)
            for call in deprecated_calls:
                issues.append({
                    'file': str(file_path),
                    'line': call['line'],
                    'issue': 'deprecated_method',
                    'old': call['method'],
                    'new': self._get_new_method(call['method']),
                    'severity': 'error'
                })
        
        return issues
    
    def generate_migration_script(self, issues: List[Dict[str, Any]]) -> str:
        """Generate automated migration script."""
        script = """#!/usr/bin/env python3
\"\"\"
Auto-generated migration script for MACE Workflow v2
Generated: {timestamp}
\"\"\"

import re
from pathlib import Path

# Migration mappings
IMPORT_MAPPINGS = {{
    'from mace.workflow import WorkflowPlanner': 'from mace.workflow.planner import WorkflowPlanner',
    'from mace.workflow import WorkflowEngine': 'from mace.workflow.engine import WorkflowEngine',
}}

METHOD_MAPPINGS = {{
    'configure_expert': 'expert_manager.configure',
    'show_workflow_status': 'WorkflowMonitor().status',
}}

def migrate_file(file_path: Path):
    \"\"\"Migrate a single file.\"\"\"
    content = file_path.read_text()
    original = content
    
    # Update imports
    for old, new in IMPORT_MAPPINGS.items():
        content = content.replace(old, new)
    
    # Update method calls
    for old, new in METHOD_MAPPINGS.items():
        pattern = rf'\\b{old}\\b'
        content = re.sub(pattern, new, content)
    
    # Write back if changed
    if content != original:
        file_path.write_text(content)
        print(f"✓ Migrated: {{file_path}}")
        return True
    return False

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    
    migrated = 0
    for py_file in path.rglob("*.py"):
        if migrate_file(py_file):
            migrated += 1
    
    print(f"\\nMigration complete: {{migrated}} files updated")
""".format(timestamp=datetime.now().isoformat())
        
        return script
```

### 7.5 Integration Risks

#### 7.5.1 External Tool Compatibility

**Risk Description:**
Changes might break integration with Crystal tools.

**Impact Assessment:**
- Probability: Medium (50%)
- Impact: High
- Risk Score: 6/10

**Mitigation Strategies:**

1. **Integration Test Suite**
```python
# tests/integration/test_crystal_tools.py
"""Integration tests for Crystal tools."""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock

class TestCrystalIntegration:
    """Test integration with Crystal tools."""
    
    @pytest.fixture
    def crystal_tools(self):
        """Mock Crystal tool paths."""
        return {
            'NewCifToD12': Path('Crystal_d12/NewCifToD12.py'),
            'CRYSTALOptToD12': Path('Crystal_d12/CRYSTALOptToD12.py'),
            'alldos': Path('Crystal_d3/alldos.py'),
            'create_band_d3': Path('Crystal_d3/create_band_d3.py')
        }
    
    def test_cif_conversion_compatibility(self, crystal_tools, tmp_path):
        """Test CIF conversion tool compatibility."""
        # Create test CIF
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(SAMPLE_CIF_CONTENT)
        
        # Test with new wrapper
        from mace.workflow.planner.cif_converter import CifConverter
        converter = CifConverter()
        
        # Mock subprocess call
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="SUCCESS: Created test.d12"
            )
            
            result = converter.convert_single(cif_file, level='basic')
            
            # Verify correct tool was called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert str(crystal_tools['NewCifToD12']) in ' '.join(args)
    
    def test_optimization_to_sp_compatibility(self, crystal_tools, tmp_path):
        """Test OPT to SP conversion compatibility."""
        # Create test files
        opt_out = tmp_path / "opt.out"
        opt_out.write_text(SAMPLE_OPT_OUTPUT)
        
        gui_file = tmp_path / "opt.gui"
        gui_file.write_text(SAMPLE_GUI_CONTENT)
        
        # Test with new system
        from mace.workflow.planner.expert_modes import OptExpertMode
        expert = OptExpertMode('SP', Mock())
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            result = expert.configure(opt_out)
            
            # Verify compatibility
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert 'CRYSTALOptToD12.py' in ' '.join(call_args)
```

2. **Version Compatibility Matrix**
```python
# mace/workflow/compatibility/crystal_versions.py
"""Crystal tool version compatibility."""

from typing import Dict, List, Tuple
import subprocess
import re

class CrystalCompatibility:
    """Check and ensure Crystal tool compatibility."""
    
    # Compatibility matrix
    COMPATIBILITY_MATRIX = {
        'NewCifToD12.py': {
            '1.0': ['CRYSTAL17', 'CRYSTAL23'],
            '2.0': ['CRYSTAL23'],
        },
        'CRYSTALOptToD12.py': {
            '1.5': ['CRYSTAL17', 'CRYSTAL23'],
            '2.0': ['CRYSTAL23'],
        }
    }
    
    def check_tool_version(self, tool_path: Path) -> Tuple[str, str]:
        """Check version of Crystal tool."""
        try:
            result = subprocess.run(
                ['python', str(tool_path), '--version'],
                capture_output=True,
                text=True
            )
            
            # Parse version from output
            version_match = re.search(r'version\s+(\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1), 'OK'
            else:
                return 'unknown', 'WARNING'
                
        except Exception as e:
            return 'error', str(e)
    
    def verify_compatibility(self) -> Dict[str, Any]:
        """Verify all tool compatibility."""
        results = {}
        
        for tool_name, versions in self.COMPATIBILITY_MATRIX.items():
            tool_path = self._find_tool_path(tool_name)
            if not tool_path:
                results[tool_name] = {
                    'status': 'NOT_FOUND',
                    'compatible': False
                }
                continue
            
            version, status = self.check_tool_version(tool_path)
            
            results[tool_name] = {
                'path': str(tool_path),
                'version': version,
                'status': status,
                'compatible': self._is_compatible(tool_name, version)
            }
        
        return results
```

### 7.6 Risk Monitoring and Response

#### 7.6.1 Risk Dashboard

```python
# monitoring/risk_dashboard.py
"""Real-time risk monitoring dashboard."""

class RiskMonitor:
    """Monitor risks during refactoring."""
    
    def __init__(self):
        self.metrics = {}
        self.thresholds = self._load_thresholds()
        self.alerts = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current risk metrics."""
        return {
            'code_coverage': self._get_test_coverage(),
            'import_time': self._measure_import_time(),
            'error_rate': self._get_error_rate(),
            'performance_regression': self._check_performance(),
            'compatibility_issues': self._check_compatibility(),
            'user_feedback': self._get_user_feedback()
        }
    
    def evaluate_risks(self) -> List[Dict[str, Any]]:
        """Evaluate current risk levels."""
        metrics = self.collect_metrics()
        risks = []
        
        # Check each metric against thresholds
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold and value > threshold:
                risks.append({
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'severity': self._calculate_severity(metric, value, threshold)
                })
        
        return risks
    
    def generate_report(self) -> str:
        """Generate risk assessment report."""
        risks = self.evaluate_risks()
        
        report = f"""
Risk Assessment Report
Generated: {datetime.now().isoformat()}
{'='*50}

Overall Risk Level: {self._calculate_overall_risk(risks)}

Identified Risks:
"""
        
        for risk in sorted(risks, key=lambda x: x['severity'], reverse=True):
            report += f"""
{risk['metric']}:
  Current: {risk['value']}
  Threshold: {risk['threshold']}
  Severity: {risk['severity']}
  Action: {self._get_mitigation_action(risk)}
"""
        
        return report
```

#### 7.6.2 Automated Response System

```python
# monitoring/automated_response.py
"""Automated risk response system."""

class AutomatedRiskResponse:
    """Automatically respond to detected risks."""
    
    def __init__(self):
        self.responses = {
            'high_error_rate': self.handle_high_error_rate,
            'performance_regression': self.handle_performance_regression,
            'compatibility_failure': self.handle_compatibility_failure,
            'test_failure': self.handle_test_failure
        }
    
    def respond_to_risk(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically respond to identified risk."""
        risk_type = risk['type']
        
        if risk_type in self.responses:
            response = self.responses[risk_type](risk)
            
            # Log response
            self._log_response(risk, response)
            
            # Notify stakeholders
            self._notify_stakeholders(risk, response)
            
            return response
        else:
            return self.default_response(risk)
    
    def handle_high_error_rate(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Handle high error rate risk."""
        error_rate = risk['value']
        
        if error_rate > 0.05:  # 5%
            # Immediate rollback
            return {
                'action': 'rollback',
                'reason': f'Error rate {error_rate:.1%} exceeds 5% threshold',
                'automated': True
            }
        elif error_rate > 0.02:  # 2%
            # Reduce traffic
            return {
                'action': 'reduce_traffic',
                'percentage': 50,
                'reason': f'Error rate {error_rate:.1%} exceeds 2% threshold',
                'automated': True
            }
```

### 7.7 Risk Summary and Priorities

#### 7.7.1 Risk Priority Matrix

| Risk | Probability | Impact | Score | Priority | Owner |
|------|------------|--------|-------|----------|-------|
| Breaking workflows | High | High | 8 | P0 | Tech Lead |
| Data corruption | Low | Critical | 7 | P0 | DBA |
| Resource availability | Medium | High | 6 | P1 | PM |
| Deployment failure | Medium | High | 6 | P1 | DevOps |
| Performance regression | Medium | Medium | 5 | P2 | Dev Team |
| Learning curve | High | Medium | 6 | P1 | Training |
| Tool compatibility | Medium | High | 6 | P1 | Integration |

#### 7.7.2 Risk Response Summary

1. **P0 Risks**: Require immediate mitigation before proceeding
2. **P1 Risks**: Must have mitigation plan in place
3. **P2 Risks**: Monitor and respond as needed

This comprehensive risk analysis ensures all potential issues are identified and addressed proactively during the refactoring process.