"""
Property Distribution Analysis
==============================
Analyze distributions and histograms of material properties.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import math
from collections import defaultdict, Counter


class PropertyDistribution:
    """Analyzes distributions of material properties."""
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def analyze_distributions(self, properties: List[str] = None,
                            n_bins: int = 10,
                            material_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analyze property distributions.
        
        Args:
            properties: List of properties to analyze (None = all numeric)
            n_bins: Number of histogram bins
            material_ids: Specific materials to analyze (None = all)
            
        Returns:
            Distribution analysis results
        """
        # Get all properties
        all_properties = self.db.get_all_properties()
        
        # Filter by materials if specified
        if material_ids:
            material_set = set(material_ids)
            all_properties = [p for p in all_properties if p['material_id'] in material_set]
            
        # Group properties by name
        props_by_name = defaultdict(list)
        for prop in all_properties:
            prop_name = prop['property_name']
            prop_value = prop['property_value']
            
            # Skip if specific properties requested and this isn't one
            if properties and prop_name not in properties:
                continue
                
            # Try to convert to numeric
            try:
                numeric_value = float(prop_value)
                props_by_name[prop_name].append({
                    'value': numeric_value,
                    'material_id': prop['material_id']
                })
            except (ValueError, TypeError):
                # Handle non-numeric properties separately
                props_by_name[prop_name].append({
                    'value': str(prop_value),
                    'material_id': prop['material_id'],
                    'is_categorical': True
                })
                
        # Analyze each property
        results = {
            'distributions': {},
            'summary': {
                'total_properties': len(props_by_name),
                'total_materials': len(set(p['material_id'] for props in all_properties for p in [props]))
            }
        }
        
        for prop_name, values in props_by_name.items():
            if not values:
                continue
                
            # Check if categorical
            if any(v.get('is_categorical', False) for v in values):
                # Categorical distribution
                results['distributions'][prop_name] = self._analyze_categorical(prop_name, values)
            else:
                # Numeric distribution
                results['distributions'][prop_name] = self._analyze_numeric(prop_name, values, n_bins)
                
        return results
        
    def _analyze_numeric(self, prop_name: str, values: List[Dict], n_bins: int) -> Dict:
        """Analyze numeric property distribution."""
        numeric_values = [v['value'] for v in values]
        
        if not numeric_values:
            return {'type': 'numeric', 'error': 'No numeric values'}
            
        # Basic statistics
        min_val = min(numeric_values)
        max_val = max(numeric_values)
        mean_val = sum(numeric_values) / len(numeric_values)
        
        # Calculate median
        sorted_values = sorted(numeric_values)
        n = len(sorted_values)
        if n % 2 == 0:
            median_val = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            median_val = sorted_values[n//2]
            
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in numeric_values) / n
        std_dev = math.sqrt(variance)
        
        # Calculate histogram
        histogram = self._calculate_histogram(numeric_values, n_bins, min_val, max_val)
        
        # Calculate percentiles
        percentiles = {}
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            percentiles[f'p{p}'] = sorted_values[idx]
            
        # Find outliers (values beyond 1.5 * IQR)
        q1 = percentiles['p25']
        q3 = percentiles['p75']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for v in values:
            if v['value'] < lower_bound or v['value'] > upper_bound:
                outliers.append({
                    'material_id': v['material_id'],
                    'value': v['value']
                })
                
        return {
            'type': 'numeric',
            'property': prop_name,
            'count': len(numeric_values),
            'statistics': {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'median': median_val,
                'std_dev': std_dev,
                'range': max_val - min_val,
                'cv': std_dev / mean_val if mean_val != 0 else float('inf')  # Coefficient of variation
            },
            'percentiles': percentiles,
            'histogram': histogram,
            'outliers': {
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'values': outliers[:10]  # Limit to top 10
            }
        }
        
    def _analyze_categorical(self, prop_name: str, values: List[Dict]) -> Dict:
        """Analyze categorical property distribution."""
        # Count occurrences
        value_counts = Counter(v['value'] for v in values)
        
        # Sort by frequency
        sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate mode
        mode_value, mode_count = sorted_counts[0] if sorted_counts else (None, 0)
        
        return {
            'type': 'categorical',
            'property': prop_name,
            'count': len(values),
            'unique_values': len(value_counts),
            'mode': {
                'value': mode_value,
                'count': mode_count,
                'frequency': mode_count / len(values) if values else 0
            },
            'value_counts': dict(sorted_counts[:20]),  # Top 20
            'all_values': len(sorted_counts) > 20  # Flag if there are more
        }
        
    def _calculate_histogram(self, values: List[float], n_bins: int, 
                           min_val: float, max_val: float) -> Dict:
        """Calculate histogram data."""
        if min_val == max_val:
            # All values are the same
            return {
                'bins': [min_val],
                'counts': [len(values)],
                'bin_edges': [min_val, min_val]
            }
            
        # Calculate bin edges
        bin_width = (max_val - min_val) / n_bins
        bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
        
        # Count values in each bin
        counts = [0] * n_bins
        for value in values:
            # Find bin index
            bin_idx = int((value - min_val) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            counts[bin_idx] += 1
            
        # Calculate bin centers
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        
        return {
            'bins': bin_centers,
            'counts': counts,
            'bin_edges': bin_edges,
            'bin_width': bin_width
        }
        
    def format_distribution_report(self, results: Dict, top_n: int = 10) -> str:
        """
        Format distribution analysis as readable report.
        
        Args:
            results: Results from analyze_distributions()
            top_n: Number of properties to show details for
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        lines.append("=== Property Distribution Analysis ===")
        lines.append(f"Properties analyzed: {results['summary']['total_properties']}")
        lines.append(f"Materials included: {results['summary']['total_materials']}")
        lines.append("")
        
        # Separate numeric and categorical
        numeric_props = []
        categorical_props = []
        
        for prop_name, dist in results['distributions'].items():
            if dist['type'] == 'numeric':
                numeric_props.append((prop_name, dist))
            else:
                categorical_props.append((prop_name, dist))
                
        # Numeric properties
        if numeric_props:
            lines.append("=== Numeric Properties ===")
            lines.append(f"{'Property':30} {'Count':>6} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std Dev':>12}")
            lines.append("-" * 90)
            
            # Sort by coefficient of variation (most variable first)
            numeric_props.sort(key=lambda x: abs(x[1]['statistics']['cv']), reverse=True)
            
            for i, (prop_name, dist) in enumerate(numeric_props[:top_n]):
                stats = dist['statistics']
                prop_display = prop_name[:30]
                count = dist['count']
                
                # Format values
                def fmt(val):
                    if abs(val) < 0.001 or abs(val) > 10000:
                        return f"{val:.2e}"
                    else:
                        return f"{val:.4f}"
                        
                lines.append(f"{prop_display:30} {count:>6} {fmt(stats['min']):>12} "
                           f"{fmt(stats['max']):>12} {fmt(stats['mean']):>12} {fmt(stats['std_dev']):>12}")
                           
                # Show histogram for first few
                if i < 3:
                    lines.append(f"  Histogram: {self._format_mini_histogram(dist['histogram'])}")
                    
                # Show outliers if any
                if dist['outliers']['count'] > 0:
                    lines.append(f"  Outliers: {dist['outliers']['count']} values beyond "
                               f"[{fmt(dist['outliers']['lower_bound'])}, {fmt(dist['outliers']['upper_bound'])}]")
                    
            if len(numeric_props) > top_n:
                lines.append(f"\n... and {len(numeric_props) - top_n} more numeric properties")
                
        # Categorical properties
        if categorical_props:
            lines.append("\n=== Categorical Properties ===")
            lines.append(f"{'Property':30} {'Count':>6} {'Unique':>8} {'Mode':>20} {'Frequency':>10}")
            lines.append("-" * 75)
            
            for prop_name, dist in categorical_props[:top_n]:
                prop_display = prop_name[:30]
                count = dist['count']
                unique = dist['unique_values']
                mode = str(dist['mode']['value'])[:20]
                freq = dist['mode']['frequency']
                
                lines.append(f"{prop_display:30} {count:>6} {unique:>8} {mode:>20} {freq:>10.1%}")
                
                # Show top values for first few
                if unique <= 5:
                    value_str = ", ".join(f"{v}({c})" for v, c in 
                                        list(dist['value_counts'].items())[:5])
                    lines.append(f"  Values: {value_str}")
                    
            if len(categorical_props) > top_n:
                lines.append(f"\n... and {len(categorical_props) - top_n} more categorical properties")
                
        return "\n".join(lines)
        
    def _format_mini_histogram(self, histogram: Dict) -> str:
        """Format a mini text histogram."""
        counts = histogram['counts']
        if not counts:
            return "[]"
            
        max_count = max(counts)
        if max_count == 0:
            return "[]"
            
        # Normalize to 10 characters
        bar_chars = "▁▂▃▄▅▆▇█"
        result = ""
        
        for count in counts:
            level = int((count / max_count) * 7)
            result += bar_chars[level]
            
        return result


def analyze_property_distributions(db, properties: List[str] = None,
                                 n_bins: int = 10,
                                 output_format: str = 'report',
                                 top_n: int = 10) -> str:
    """
    Convenience function to analyze property distributions.
    
    Args:
        db: MaterialDatabase instance
        properties: Specific properties to analyze (None = all)
        n_bins: Number of histogram bins
        output_format: 'report', 'json', or 'dict'
        top_n: Number of properties to show in report
        
    Returns:
        Formatted results
    """
    analyzer = PropertyDistribution(db)
    results = analyzer.analyze_distributions(properties, n_bins)
    
    if output_format == 'report':
        return analyzer.format_distribution_report(results, top_n)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    else:  # dict
        return results