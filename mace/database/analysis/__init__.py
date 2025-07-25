"""
Analysis module for MACE database
==================================
Provides advanced analysis capabilities for materials data.
"""

from .comparison import MaterialComparison, compare_materials
from .missing_data import MissingDataAnalyzer, analyze_missing_data
from .correlation import PropertyCorrelation, calculate_property_correlations
from .distribution import PropertyDistribution, analyze_property_distributions

__all__ = ['MaterialComparison', 'compare_materials', 
           'MissingDataAnalyzer', 'analyze_missing_data',
           'PropertyCorrelation', 'calculate_property_correlations',
           'PropertyDistribution', 'analyze_property_distributions']