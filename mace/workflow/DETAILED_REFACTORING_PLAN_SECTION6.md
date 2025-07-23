# MACE Workflow Module - Detailed Refactoring Plan
# Section 6: Implementation Roadmap

## 6. Implementation Roadmap

### 6.1 Overview and Timeline

#### 6.1.1 Project Timeline Summary

```
Phase                         Duration    Start Date    End Date      Dependencies
--------------------------  ----------  ------------  ------------  ----------------
Phase 1: Foundation            2 weeks    Week 1        Week 2      None
Phase 2: Consolidation         2 weeks    Week 3        Week 4      Phase 1
Phase 3: Modularization        2 weeks    Week 5        Week 6      Phase 2
Phase 4: Deployment            2 weeks    Week 7        Week 8      Phase 3
--------------------------  ----------  ------------  ------------  ----------------
Total Duration                 8 weeks

Critical Path:
Foundation → Consolidation → Modularization → Deployment

Resource Requirements:
- 2-3 developers full-time
- 1 DevOps engineer (Weeks 6-8)
- Code review from senior developer
- Testing resources (Weeks 4-8)
```

### 6.2 Phase 1: Foundation (Weeks 1-2)

#### 6.2.1 Week 1: Setup and Configuration

##### Day 1-2: Project Setup
```bash
# Create refactoring branch
git checkout -b refactor/workflow-module-v2

# Create new directory structure
mkdir -p mace/workflow/{config,errors,logging}
mkdir -p mace/workflow/planner/{core,interactive,expert_modes}
mkdir -p mace/workflow/engine/{core,orchestration,file_ops,job_mgmt}
mkdir -p mace/workflow/executor/{core,steps,dependencies}
mkdir -p tests/{unit,integration,performance,fixtures}

# Set up configuration files
touch mace/workflow/config/{__init__,loader,schema,validator}.py
touch mace/workflow/errors/{__init__,exceptions,handler,recovery}.py
touch mace/workflow/logging/{__init__,config,performance}.py

# Initialize test structure
touch tests/conftest.py
touch tests/unit/test_{planner,engine,executor,monitor}.py
```

##### Day 3-4: Configuration Management Implementation
```python
# mace/workflow/config/defaults/workflow.yaml
workflow:
  name: "mace_workflow"
  version: "2.0"
  
  optional_calculations:
    - CHARGE+POTENTIAL
    - TRANSPORT
    - ECHG
    - POTM
  
  resource_defaults:
    OPT:
      cores: 32
      memory: "5G"
      walltime: "7-00:00:00"
      account: "mendoza_q"
    SP:
      cores: 32
      memory: "4G"
      walltime: "3-00:00:00"
      account: "mendoza_q"
    BAND:
      cores: 28
      memory: "80G"
      walltime: "24:00:00"
      account: "mendoza_q"
    DOSS:
      cores: 28
      memory: "80G"
      walltime: "24:00:00"
      account: "mendoza_q"
    FREQ:
      cores: 32
      memory: "5G"
      walltime: "7-00:00:00"
      account: "mendoza_q"
  
  timeouts:
    cif_conversion: 300
    file_check: 30
    job_submission: 60
    database_operation: 10
```

Implementation tasks:
- [ ] Implement ConfigurationManager class
- [ ] Create configuration schema with pydantic
- [ ] Implement configuration validation
- [ ] Create environment-specific configs
- [ ] Add configuration hot-reloading
- [ ] Write configuration tests

##### Day 5: Logging Framework
```python
# Setup logging configuration
from mace.workflow.logging import LogManager, LogContext

# Initialize logging
log_manager = LogManager(
    app_name="mace_workflow",
    config={
        'structured_logging': True,
        'log_level': 'INFO',
        'rotate_size': '10MB',
        'backup_count': 5
    }
)

# Usage in modules
logger = log_manager.get_logger(__name__)

# Context-aware logging
with LogContext(workflow_id='wf-123', material_id='mat-456'):
    logger.info("Processing material")
```

Tasks:
- [ ] Implement LogManager class
- [ ] Create JSON formatter for structured logging
- [ ] Setup log rotation handlers
- [ ] Implement context filter
- [ ] Create performance logger
- [ ] Replace all print statements with logging

#### 6.2.2 Week 2: Core Refactoring

##### Day 6-7: Error Handling Framework
```python
# Implement error hierarchy
from mace.workflow.errors import (
    WorkflowError, ConfigurationError, FileOperationError,
    JobSubmissionError, CalculationError, DatabaseError
)

# Implement error handler
from mace.workflow.errors import ErrorHandler, handle_errors

# Setup recovery strategies
error_handler = ErrorHandler()
error_handler.register_strategy(
    FileOperationError,
    ExponentialBackoffStrategy(max_retries=5)
)

# Apply decorator
@handle_errors(default_return=None, can_recover=True)
def risky_operation():
    # Operation that might fail
    pass
```

Tasks:
- [ ] Create error exception hierarchy
- [ ] Implement ErrorHandler class
- [ ] Create recovery strategies
- [ ] Implement error decorators
- [ ] Add error metrics tracking
- [ ] Write error handling tests

##### Day 8-9: Merge Contextual Features
```python
# Update planner.py
class WorkflowPlanner:
    def __init__(self, base_dir=".", db_path="materials.db", 
                 isolated=False, isolation_context=None):
        self.base_dir = Path(base_dir)
        self.db_path = db_path
        self.isolated = isolated
        self.isolation_context = isolation_context
        
        # Rest of initialization
```

Tasks:
- [ ] Add isolation parameters to base classes
- [ ] Merge contextual functionality
- [ ] Update all method calls
- [ ] Test isolation mode
- [ ] Remove contextual files
- [ ] Update imports

##### Day 10: Documentation and Review
- [ ] Document new configuration system
- [ ] Document error handling patterns
- [ ] Document logging usage
- [ ] Code review with team
- [ ] Address review feedback
- [ ] Prepare for Phase 2

### 6.3 Phase 2: Consolidation (Weeks 3-4)

#### 6.3.1 Week 3: Monitoring Unification

##### Day 11-12: Create Unified Monitor
```python
# mace/workflow/monitor.py
class WorkflowMonitor:
    """Unified workflow monitoring system."""
    
    def __init__(self, db_path="materials.db"):
        self.db = MaterialDatabase(db_path)
        self.engine = WorkflowEngine(db_path=db_path)
        self.display_handlers = self._setup_display_handlers()
        self.metrics_cache = {}
    
    def status(self, workflow_id=None, display_mode='summary', **kwargs):
        """Get and display workflow status."""
        # Implementation combining all three scripts
    
    def monitor(self, interval=30, mode='continuous', **kwargs):
        """Monitor workflows with auto-refresh."""
        # Real-time monitoring implementation
    
    def check_progression(self, workflow_id=None, auto_progress=False, **kwargs):
        """Check and optionally progress workflows."""
        # Progression checking implementation
```

Tasks:
- [ ] Create WorkflowMonitor class
- [ ] Implement all display modes
- [ ] Add monitoring modes (once, continuous, watch)
- [ ] Implement progression checking
- [ ] Create CLI interface
- [ ] Add compatibility wrappers

##### Day 13-14: Deprecate Old Scripts
```python
# compatibility.py
import warnings
from mace.workflow.monitor import WorkflowMonitor

def show_workflow_status(**kwargs):
    warnings.warn(
        "show_workflow_status is deprecated. Use WorkflowMonitor.status()",
        DeprecationWarning,
        stacklevel=2
    )
    monitor = WorkflowMonitor()
    return monitor.status(**kwargs)
```

Tasks:
- [ ] Create compatibility module
- [ ] Add deprecation warnings
- [ ] Update all references
- [ ] Test compatibility
- [ ] Document migration path
- [ ] Update user documentation

##### Day 15: Initial Decomposition - Engine
```
# Create engine module structure
engine/
├── __init__.py
├── core.py              # Core WorkflowEngine
├── orchestrator.py      # Workflow orchestration
├── file_manager.py      # File operations
├── job_submitter.py     # Job submission
├── error_recovery.py    # Error handling
└── utils.py            # Utilities
```

Tasks:
- [ ] Create engine package structure
- [ ] Extract WorkflowEngine core
- [ ] Move orchestration logic
- [ ] Extract file operations
- [ ] Move job submission code
- [ ] Update imports progressively

#### 6.3.2 Week 4: Complete Decomposition

##### Day 16-17: Decompose Planner
```python
# planner/core.py - Core planner class (~500 lines)
class WorkflowPlanner:
    def __init__(self, base_dir=".", db_path="materials.db"):
        self.interactive = InteractivePlanner(self)
        self.cif_converter = CifConverter(self)
        self.expert_manager = ExpertModeManager(self)
        self.template_manager = TemplateManager()
        self.config_manager = ConfigurationManager()
```

Tasks:
- [ ] Create planner package structure
- [ ] Extract interactive planning
- [ ] Move CIF conversion logic
- [ ] Extract expert modes
- [ ] Move template management
- [ ] Test decomposed modules

##### Day 18-19: Decompose Executor
```
executor/
├── __init__.py
├── core.py              # Core executor
├── step_executor.py     # Step execution
├── dependency_mgr.py    # Dependencies
├── progress_tracker.py  # Progress tracking
└── utils.py            # Utilities
```

Tasks:
- [ ] Create executor package structure
- [ ] Extract core executor logic
- [ ] Move step execution
- [ ] Extract dependency management
- [ ] Add progress tracking
- [ ] Update all references

##### Day 20: Testing and Integration
- [ ] Run all existing tests
- [ ] Fix broken imports
- [ ] Test decomposed modules
- [ ] Performance benchmarking
- [ ] Integration testing
- [ ] Document changes

### 6.4 Phase 3: Modularization (Weeks 5-6)

#### 6.4.1 Week 5: Complete Modularization

##### Day 21-22: Standardize Interfaces
```python
# Define standard interfaces
from abc import ABC, abstractmethod

class WorkflowComponent(ABC):
    """Base interface for workflow components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        pass
    
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate component state."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass
```

Tasks:
- [ ] Define component interfaces
- [ ] Implement base classes
- [ ] Standardize method signatures
- [ ] Add type hints everywhere
- [ ] Create component registry
- [ ] Document interfaces

##### Day 23-24: Plugin Architecture
```python
# plugin_manager.py
class PluginManager:
    """Manage workflow plugins."""
    
    def __init__(self):
        self.plugins = {}
        self._discover_plugins()
    
    def register_plugin(self, name: str, plugin: WorkflowPlugin):
        """Register a workflow plugin."""
        self.plugins[name] = plugin
    
    def get_plugin(self, name: str) -> WorkflowPlugin:
        """Get plugin by name."""
        return self.plugins.get(name)
```

Tasks:
- [ ] Design plugin interface
- [ ] Implement plugin manager
- [ ] Create plugin discovery
- [ ] Add plugin configuration
- [ ] Test plugin loading
- [ ] Document plugin API

##### Day 25: Performance Optimization
```python
# Implement lazy loading
class LazyLoader:
    """Lazy load heavy modules."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)

# Usage
numpy = LazyLoader('numpy')  # Only loads when accessed
```

Tasks:
- [ ] Profile import times
- [ ] Implement lazy loading
- [ ] Optimize hot paths
- [ ] Add caching where needed
- [ ] Reduce memory usage
- [ ] Benchmark improvements

#### 6.4.2 Week 6: Testing and Documentation

##### Day 26-27: Comprehensive Testing
```python
# Test suite structure
tests/
├── unit/
│   ├── test_config_manager.py
│   ├── test_error_handler.py
│   ├── test_workflow_monitor.py
│   └── test_all_components.py
├── integration/
│   ├── test_full_workflow.py
│   ├── test_error_recovery.py
│   └── test_plugin_system.py
└── performance/
    ├── test_import_times.py
    ├── test_memory_usage.py
    └── test_large_workflows.py
```

Tasks:
- [ ] Write unit tests (>80% coverage)
- [ ] Create integration tests
- [ ] Add performance tests
- [ ] Test error scenarios
- [ ] Test edge cases
- [ ] Fix discovered issues

##### Day 28-29: Documentation Update
```markdown
# Workflow Module Documentation

## Architecture Overview
The refactored workflow module follows a modular architecture...

## Configuration Guide
Configuration is managed through YAML files...

## Error Handling
Comprehensive error handling with recovery strategies...

## Plugin Development
Create custom workflow plugins...

## Migration Guide
Migrating from v1 to v2...
```

Tasks:
- [ ] Update architecture docs
- [ ] Write configuration guide
- [ ] Document error handling
- [ ] Create plugin guide
- [ ] Write migration guide
- [ ] Add code examples

##### Day 30: Final Review
- [ ] Code review with team
- [ ] Performance review
- [ ] Security review
- [ ] Documentation review
- [ ] Address feedback
- [ ] Prepare for deployment

### 6.5 Phase 4: Deployment (Weeks 7-8)

#### 6.5.1 Week 7: Gradual Rollout

##### Day 31-32: Development Environment
```bash
# Deploy to development
./deploy.sh development

# Run smoke tests
pytest tests/smoke/

# Monitor for issues
tail -f logs/mace_workflow.log
```

Tasks:
- [ ] Deploy to dev environment
- [ ] Run smoke tests
- [ ] Monitor logs
- [ ] Gather feedback
- [ ] Fix critical issues
- [ ] Update deployment scripts

##### Day 33-34: Staging Environment
```bash
# Deploy to staging
./deploy.sh staging

# Run full test suite
pytest tests/

# Performance testing
python tests/performance/benchmark.py
```

Tasks:
- [ ] Deploy to staging
- [ ] Run full test suite
- [ ] Performance testing
- [ ] Load testing
- [ ] User acceptance testing
- [ ] Fix discovered issues

##### Day 35: Production Preparation
```bash
# Production checklist
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Rollback plan ready
- [ ] Monitoring configured
- [ ] Team trained
```

Tasks:
- [ ] Final security review
- [ ] Update production configs
- [ ] Prepare rollback plan
- [ ] Configure monitoring
- [ ] Brief support team
- [ ] Schedule deployment

#### 6.5.2 Week 8: Production Migration

##### Day 36-37: Production Deployment
```bash
# Production deployment plan
1. Backup current system
2. Deploy v2 in parallel
3. Route 10% traffic to v2
4. Monitor for 24 hours
5. Gradually increase traffic
6. Full cutover
```

Tasks:
- [ ] Execute deployment plan
- [ ] Monitor system health
- [ ] Track error rates
- [ ] Monitor performance
- [ ] Gather user feedback
- [ ] Address issues quickly

##### Day 38-39: Post-Deployment
```python
# Monitoring dashboard
from mace.workflow.monitor import SystemMonitor

monitor = SystemMonitor()
monitor.track_metrics([
    'workflow_creation_time',
    'calculation_submission_rate',
    'error_rate',
    'recovery_success_rate',
    'average_completion_time'
])
```

Tasks:
- [ ] Monitor all metrics
- [ ] Analyze performance
- [ ] Document lessons learned
- [ ] Plan future improvements
- [ ] Update runbooks
- [ ] Celebrate success!

##### Day 40: Project Closure
- [ ] Final documentation
- [ ] Knowledge transfer
- [ ] Archive old code
- [ ] Update CI/CD pipelines
- [ ] Plan maintenance schedule
- [ ] Project retrospective

### 6.6 Risk Mitigation Strategies

#### 6.6.1 Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| Breaking changes | Medium | High | Feature flags, gradual rollout |
| Performance regression | Low | Medium | Continuous benchmarking |
| Data corruption | Low | High | Comprehensive testing, backups |
| Integration failures | Medium | Medium | Integration test suite |
| User adoption issues | Medium | Low | Training, documentation |

#### 6.6.2 Contingency Plans

##### Rollback Procedure
```bash
#!/bin/bash
# rollback.sh - Emergency rollback script

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./rollback.sh <version>"
    exit 1
fi

# Stop new workflow creation
touch /tmp/workflow.maintenance

# Backup current state
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz mace/workflow/

# Rollback to previous version
git checkout $VERSION -- mace/workflow/

# Restart services
systemctl restart mace-workflow

# Remove maintenance flag
rm /tmp/workflow.maintenance

echo "Rolled back to version $VERSION"
```

### 6.7 Success Criteria

#### 6.7.1 Quantitative Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Code reduction | >30% | Line count analysis |
| Test coverage | >80% | Coverage.py report |
| Import time | <1s | Performance profiling |
| Memory usage | <200MB | Memory profiler |
| Error rate | <0.5% | Log analysis |
| Bug fix time | <2hr | Issue tracking |

#### 6.7.2 Qualitative Metrics

- [ ] Improved developer satisfaction
- [ ] Easier onboarding for new developers
- [ ] Reduced time to implement new features
- [ ] Better system maintainability
- [ ] Clearer code organization
- [ ] Enhanced debugging capability

### 6.8 Long-term Maintenance Plan

#### 6.8.1 Regular Maintenance Schedule

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Dependency updates | Monthly | DevOps |
| Performance review | Quarterly | Tech Lead |
| Security audit | Quarterly | Security Team |
| Code quality review | Monthly | Development Team |
| Documentation update | Bi-weekly | Documentation Team |

#### 6.8.2 Future Enhancements

1. **Version 2.1** (Q2)
   - Add GraphQL API
   - Implement workflow templates marketplace
   - Add real-time notifications

2. **Version 2.2** (Q3)
   - Machine learning for resource prediction
   - Advanced workflow optimization
   - Multi-cloud support

3. **Version 3.0** (Q4)
   - Complete microservices architecture
   - Kubernetes native deployment
   - Advanced workflow orchestration

This detailed implementation roadmap provides a clear path from the current monolithic structure to a modern, maintainable, and extensible workflow system.