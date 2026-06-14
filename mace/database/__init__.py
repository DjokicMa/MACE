"""
MACE Database Module
====================
Material tracking database with workflow isolation support.
"""

# --- PyArrow import-order guard (do not remove without re-testing) ------------
# Importing this package's dependency chain registers the Arrow C++ extension type
# such that a LATER `import pandas` (e.g. the lazy `import pandas` in
# export/formats.py:_export_excel, or anything pulling xarray/plotly.express)
# raises `pyarrow.lib.ArrowKeyError: A type extension with name
# arrow.py_extension_type already defined`. Repro of the bug this prevents:
#     python -c "import mace.database; import pandas"                 # crash
#     python -c "import pyarrow; import mace.database; import pandas" # ok
# Importing pyarrow FIRST makes pyarrow.lib register the extension cleanly once, so
# the later pandas import is a no-op re-use. The test suite never caught this
# because no test imports pandas after mace.database (the real CLI does, e.g. Excel
# export); tests/test_pyarrow_import_order.py now pins the real invocation order.
# Guarded so environments without pyarrow (where the conflict can't arise) are fine.
try:  # pragma: no cover - environment dependent
    import pyarrow as _pyarrow  # noqa: F401
except Exception:
    pass
# -----------------------------------------------------------------------------

# Core database classes
from .materials import MaterialDatabase
from .materials_contextual import ContextualMaterialDatabase, get_contextual_database

# Query functionality
from .query import (
    PropertyFilter, parse_filter_string,
    AdvancedFilterParser, parse_advanced_filter, evaluate_advanced_filter,
    query_materials, execute_custom_query
)

# Analysis tools
from .analysis import (
    MaterialComparison, compare_materials,
    MissingDataAnalyzer, analyze_missing_data,
    PropertyCorrelation, calculate_property_correlations,
    PropertyDistribution, analyze_property_distributions,
    WorkflowProgress, track_workflow_progress,
    PropertyAggregator, aggregate_by_groups
)

# Export functionality
from .export import ExportFormatter, export_materials, VisualizationExporter

# Utilities
from .utils import (
    UnitConverter, convert_units, get_property_units, get_default_unit,
    parse_value_with_unit, format_value_with_unit,
    PropertyValidator, DatabaseValidator, validate_materials,
    PropertyHistory
)

# Interactive explorer
from .interactive import DatabaseExplorer, run_interactive_explorer

__all__ = [
    # Core database
    'MaterialDatabase',
    'ContextualMaterialDatabase', 
    'get_contextual_database',
    # Query
    'PropertyFilter', 'parse_filter_string',
    'AdvancedFilterParser', 'parse_advanced_filter', 'evaluate_advanced_filter',
    'query_materials', 'execute_custom_query',
    # Analysis
    'MaterialComparison', 'compare_materials',
    'MissingDataAnalyzer', 'analyze_missing_data',
    'PropertyCorrelation', 'calculate_property_correlations',
    'PropertyDistribution', 'analyze_property_distributions',
    'WorkflowProgress', 'track_workflow_progress',
    'PropertyAggregator', 'aggregate_by_groups',
    # Export
    'ExportFormatter', 'export_materials', 'VisualizationExporter',
    # Utils
    'UnitConverter', 'convert_units', 'get_property_units', 'get_default_unit',
    'parse_value_with_unit', 'format_value_with_unit',
    'PropertyValidator', 'DatabaseValidator', 'validate_materials',
    'PropertyHistory',
    # Interactive
    'DatabaseExplorer', 'run_interactive_explorer'
]