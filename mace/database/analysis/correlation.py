"""
Property Correlation Analysis
=============================
Analyze correlations between material properties.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import math
from collections import defaultdict


class PropertyCorrelation:
    """Analyzes correlations between material properties."""
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def calculate_correlations(self, property_pairs: List[Tuple[str, str]] = None,
                             min_samples: int = 3,
                             material_ids: List[str] = None) -> Dict[str, Any]:
        """
        Calculate correlations between properties.
        
        Args:
            property_pairs: List of (prop1, prop2) tuples to analyze (None = all numeric pairs)
            min_samples: Minimum number of materials with both properties
            material_ids: Specific materials to analyze (None = all)
            
        Returns:
            Correlation analysis results
        """
        # Get all properties
        all_properties = self.db.get_all_properties()
        
        # Filter by materials if specified
        if material_ids:
            material_set = set(material_ids)
            all_properties = [p for p in all_properties if p['material_id'] in material_set]
            
        # Group properties by material
        props_by_material = defaultdict(dict)
        for prop in all_properties:
            mat_id = prop['material_id']
            prop_name = prop['property_name']
            prop_value = prop['property_value']
            
            # Try to convert to numeric
            try:
                numeric_value = float(prop_value)
                props_by_material[mat_id][prop_name] = numeric_value
            except (ValueError, TypeError):
                # Skip non-numeric properties
                pass
                
        # Get all numeric property names
        all_prop_names = set()
        for mat_props in props_by_material.values():
            all_prop_names.update(mat_props.keys())
            
        # Determine property pairs to analyze
        if property_pairs:
            pairs_to_analyze = property_pairs
        else:
            # Generate all unique pairs
            prop_list = sorted(all_prop_names)
            pairs_to_analyze = []
            for i in range(len(prop_list)):
                for j in range(i + 1, len(prop_list)):
                    pairs_to_analyze.append((prop_list[i], prop_list[j]))
                    
        # Calculate correlations
        results = {
            'correlations': [],
            'summary': {
                'total_materials': len(props_by_material),
                'total_properties': len(all_prop_names),
                'pairs_analyzed': len(pairs_to_analyze)
            }
        }
        
        for prop1, prop2 in pairs_to_analyze:
            # Collect paired values
            x_values = []
            y_values = []
            materials = []
            
            for mat_id, mat_props in props_by_material.items():
                if prop1 in mat_props and prop2 in mat_props:
                    x_values.append(mat_props[prop1])
                    y_values.append(mat_props[prop2])
                    materials.append(mat_id)
                    
            # Skip if not enough samples
            if len(x_values) < min_samples:
                continue
                
            # Calculate correlation
            correlation = self._pearson_correlation(x_values, y_values)
            
            # Calculate linear regression
            slope, intercept, r_squared = self._linear_regression(x_values, y_values)
            
            # Store results
            result = {
                'property_1': prop1,
                'property_2': prop2,
                'correlation': correlation,
                'r_squared': r_squared,
                'slope': slope,
                'intercept': intercept,
                'sample_count': len(x_values),
                'materials': materials,
                'x_range': [min(x_values), max(x_values)] if x_values else [None, None],
                'y_range': [min(y_values), max(y_values)] if y_values else [None, None]
            }
            
            results['correlations'].append(result)
            
        # Sort by absolute correlation
        results['correlations'].sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Add summary statistics
        self._add_summary_stats(results)
        
        return results
        
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0
            
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        # Calculate covariance and standard deviations
        covariance = 0.0
        x_variance = 0.0
        y_variance = 0.0
        
        for i in range(n):
            x_diff = x[i] - x_mean
            y_diff = y[i] - y_mean
            covariance += x_diff * y_diff
            x_variance += x_diff * x_diff
            y_variance += y_diff * y_diff
            
        # Avoid division by zero
        if x_variance == 0 or y_variance == 0:
            return 0.0
            
        # Calculate correlation
        correlation = covariance / math.sqrt(x_variance * y_variance)
        return correlation
        
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        Calculate linear regression parameters.
        
        Returns:
            (slope, intercept, r_squared)
        """
        n = len(x)
        if n == 0:
            return 0.0, 0.0, 0.0
            
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        # Calculate slope
        numerator = 0.0
        denominator = 0.0
        
        for i in range(n):
            numerator += (x[i] - x_mean) * (y[i] - y_mean)
            denominator += (x[i] - x_mean) ** 2
            
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
            
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        ss_tot = 0.0
        ss_res = 0.0
        
        for i in range(n):
            y_pred = slope * x[i] + intercept
            ss_tot += (y[i] - y_mean) ** 2
            ss_res += (y[i] - y_pred) ** 2
            
        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            
        return slope, intercept, r_squared
        
    def _add_summary_stats(self, results: Dict):
        """Add summary statistics to results."""
        correlations = results['correlations']
        
        if not correlations:
            results['summary']['strong_correlations'] = []
            results['summary']['strongest_positive'] = None
            results['summary']['strongest_negative'] = None
            return
            
        # Find strong correlations (|r| > 0.7)
        strong = []
        for corr in correlations:
            if abs(corr['correlation']) > 0.7:
                strong.append({
                    'properties': f"{corr['property_1']} vs {corr['property_2']}",
                    'correlation': corr['correlation'],
                    'r_squared': corr['r_squared'],
                    'samples': corr['sample_count']
                })
                
        results['summary']['strong_correlations'] = strong
        
        # Find strongest positive and negative
        positive = [c for c in correlations if c['correlation'] > 0]
        negative = [c for c in correlations if c['correlation'] < 0]
        
        if positive:
            strongest_pos = max(positive, key=lambda x: x['correlation'])
            results['summary']['strongest_positive'] = {
                'properties': f"{strongest_pos['property_1']} vs {strongest_pos['property_2']}",
                'correlation': strongest_pos['correlation'],
                'r_squared': strongest_pos['r_squared']
            }
        else:
            results['summary']['strongest_positive'] = None
            
        if negative:
            strongest_neg = min(negative, key=lambda x: x['correlation'])
            results['summary']['strongest_negative'] = {
                'properties': f"{strongest_neg['property_1']} vs {strongest_neg['property_2']}",
                'correlation': strongest_neg['correlation'],
                'r_squared': strongest_neg['r_squared']
            }
        else:
            results['summary']['strongest_negative'] = None
            
    def format_correlation_report(self, results: Dict, top_n: int = 20,
                                min_correlation: float = 0.0) -> str:
        """
        Format correlation results as readable report.
        
        Args:
            results: Results from calculate_correlations()
            top_n: Number of top correlations to show
            min_correlation: Minimum absolute correlation to display
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        lines.append("=== Property Correlation Analysis ===")
        lines.append(f"Materials analyzed: {results['summary']['total_materials']}")
        lines.append(f"Properties analyzed: {results['summary']['total_properties']}")
        lines.append(f"Property pairs analyzed: {results['summary']['pairs_analyzed']}")
        lines.append("")
        
        # Strong correlations summary
        strong = results['summary']['strong_correlations']
        if strong:
            lines.append(f"=== Strong Correlations (|r| > 0.7) ===")
            lines.append(f"Found {len(strong)} strong correlations:")
            for s in strong[:10]:  # Show top 10
                lines.append(f"  {s['properties']}: r = {s['correlation']:.3f} (R² = {s['r_squared']:.3f}, n = {s['samples']})")
            if len(strong) > 10:
                lines.append(f"  ... and {len(strong) - 10} more")
            lines.append("")
            
        # Strongest positive/negative
        if results['summary']['strongest_positive']:
            pos = results['summary']['strongest_positive']
            lines.append(f"Strongest positive correlation:")
            lines.append(f"  {pos['properties']}: r = {pos['correlation']:.3f} (R² = {pos['r_squared']:.3f})")
            
        if results['summary']['strongest_negative']:
            neg = results['summary']['strongest_negative']
            lines.append(f"Strongest negative correlation:")
            lines.append(f"  {neg['properties']}: r = {neg['correlation']:.3f} (R² = {neg['r_squared']:.3f})")
            
        lines.append("")
        
        # Detailed correlations
        lines.append("=== Top Correlations ===")
        lines.append(f"{'Property 1':30} {'Property 2':30} {'r':>8} {'R²':>8} {'n':>5}")
        lines.append("-" * 85)
        
        shown = 0
        for corr in results['correlations']:
            if abs(corr['correlation']) >= min_correlation:
                prop1 = corr['property_1'][:30]
                prop2 = corr['property_2'][:30]
                r = corr['correlation']
                r2 = corr['r_squared']
                n = corr['sample_count']
                
                lines.append(f"{prop1:30} {prop2:30} {r:8.3f} {r2:8.3f} {n:5d}")
                
                shown += 1
                if shown >= top_n:
                    break
                    
        if shown == 0:
            lines.append("No correlations found meeting criteria.")
            
        return "\n".join(lines)
        
    def get_scatter_plot_data(self, property_1: str, property_2: str,
                            material_ids: List[str] = None) -> Dict[str, Any]:
        """
        Get data for scatter plot of two properties.
        
        Args:
            property_1: X-axis property
            property_2: Y-axis property  
            material_ids: Specific materials to include
            
        Returns:
            Dictionary with plot data
        """
        # Get properties
        all_properties = self.db.get_all_properties()
        
        # Filter by materials if specified
        if material_ids:
            material_set = set(material_ids)
            all_properties = [p for p in all_properties if p['material_id'] in material_set]
            
        # Group by material
        props_by_material = defaultdict(dict)
        for prop in all_properties:
            if prop['property_name'] in [property_1, property_2]:
                mat_id = prop['material_id']
                prop_name = prop['property_name']
                try:
                    props_by_material[mat_id][prop_name] = float(prop['property_value'])
                except (ValueError, TypeError):
                    pass
                    
        # Collect plot data
        plot_data = {
            'x_property': property_1,
            'y_property': property_2,
            'points': []
        }
        
        for mat_id, props in props_by_material.items():
            if property_1 in props and property_2 in props:
                # Get material info
                material = self.db.get_material(mat_id)
                
                plot_data['points'].append({
                    'material_id': mat_id,
                    'formula': material.get('formula', 'Unknown') if material else 'Unknown',
                    'x': props[property_1],
                    'y': props[property_2]
                })
                
        # Calculate correlation if we have data
        if plot_data['points']:
            x_values = [p['x'] for p in plot_data['points']]
            y_values = [p['y'] for p in plot_data['points']]
            
            correlation = self._pearson_correlation(x_values, y_values)
            slope, intercept, r_squared = self._linear_regression(x_values, y_values)
            
            plot_data['statistics'] = {
                'correlation': correlation,
                'r_squared': r_squared,
                'slope': slope,
                'intercept': intercept,
                'sample_count': len(plot_data['points'])
            }
            
        return plot_data


def calculate_property_correlations(db, property_pairs: List[Tuple[str, str]] = None,
                                  min_samples: int = 3,
                                  output_format: str = 'report',
                                  top_n: int = 20) -> str:
    """
    Convenience function to calculate property correlations.
    
    Args:
        db: MaterialDatabase instance
        property_pairs: Specific pairs to analyze (None = all)
        min_samples: Minimum samples required
        output_format: 'report', 'json', or 'dict'
        top_n: Number of top correlations to show in report
        
    Returns:
        Formatted results
    """
    analyzer = PropertyCorrelation(db)
    results = analyzer.calculate_correlations(property_pairs, min_samples)
    
    if output_format == 'report':
        return analyzer.format_correlation_report(results, top_n)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    else:  # dict
        return results