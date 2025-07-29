"""
MACE Database Module
====================
Material tracking database with workflow isolation support.
"""

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