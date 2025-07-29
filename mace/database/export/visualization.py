"""
Visualization Data Export
=========================
Export data formatted for visualization tools and plotting libraries.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import csv
from datetime import datetime
from collections import defaultdict
import numpy as np


class VisualizationExporter:
    """Export data formatted for various visualization tools."""
    
    # Supported visualization formats
    FORMATS = {
        'plotly': 'Plotly JSON format',
        'matplotlib': 'Matplotlib-ready arrays',
        'bokeh': 'Bokeh ColumnDataSource format',
        'vega': 'Vega-Lite specification',
        'd3': 'D3.js compatible JSON',
        'gnuplot': 'Gnuplot data format',
        'origin': 'Origin-compatible format'
    }
    
    def __init__(self, db):
        """
        Initialize visualization exporter.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def export_scatter_data(self, x_property: str, y_property: str,
                          material_ids: List[str] = None,
                          color_by: Optional[str] = None,
                          size_by: Optional[str] = None,
                          format: str = 'plotly') -> Dict[str, Any]:
        """
        Export scatter plot data.
        
        Args:
            x_property: Property for x-axis
            y_property: Property for y-axis
            material_ids: Specific materials to include
            color_by: Property to color points by
            size_by: Property to size points by
            format: Export format
            
        Returns:
            Formatted visualization data
        """
        # Collect data
        data_points = []
        
        materials = self.db.get_all_materials()
        if material_ids:
            materials = [m for m in materials if m['material_id'] in material_ids]
            
        for material in materials:
            mat_id = material['material_id']
            props = self.db.get_material_properties(mat_id)
            
            # Get property values
            prop_dict = {p['property_name']: p['property_value'] for p in props}
            
            if x_property in prop_dict and y_property in prop_dict:
                try:
                    point = {
                        'material_id': mat_id,
                        'formula': material.get('formula', 'Unknown'),
                        'x': float(prop_dict[x_property]),
                        'y': float(prop_dict[y_property])
                    }
                    
                    # Add color property if requested
                    if color_by and color_by in prop_dict:
                        try:
                            point['color'] = float(prop_dict[color_by])
                        except ValueError:
                            point['color'] = prop_dict[color_by]
                            
                    # Add size property if requested
                    if size_by and size_by in prop_dict:
                        try:
                            point['size'] = float(prop_dict[size_by])
                        except ValueError:
                            pass
                            
                    data_points.append(point)
                    
                except (ValueError, TypeError):
                    pass
                    
        # Format based on requested format
        if format == 'plotly':
            return self._format_plotly_scatter(data_points, x_property, y_property, color_by, size_by)
        elif format == 'matplotlib':
            return self._format_matplotlib_scatter(data_points)
        elif format == 'bokeh':
            return self._format_bokeh_scatter(data_points)
        elif format == 'vega':
            return self._format_vega_scatter(data_points, x_property, y_property, color_by)
        elif format == 'd3':
            return self._format_d3_scatter(data_points)
        elif format == 'gnuplot':
            return self._format_gnuplot_scatter(data_points)
        else:
            return {'error': f'Unsupported format: {format}'}
            
    def _format_plotly_scatter(self, data_points: List[Dict], x_prop: str, y_prop: str,
                             color_by: str = None, size_by: str = None) -> Dict:
        """Format data for Plotly."""
        trace = {
            'type': 'scatter',
            'mode': 'markers',
            'x': [p['x'] for p in data_points],
            'y': [p['y'] for p in data_points],
            'text': [f"{p['material_id']}<br>{p['formula']}" for p in data_points],
            'hovertemplate': '%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>',
            'marker': {}
        }
        
        if color_by and any('color' in p for p in data_points):
            trace['marker']['color'] = [p.get('color', 0) for p in data_points]
            trace['marker']['colorscale'] = 'Viridis'
            trace['marker']['showscale'] = True
            trace['marker']['colorbar'] = {'title': color_by}
            
        if size_by and any('size' in p for p in data_points):
            sizes = [p.get('size', 1) for p in data_points]
            # Normalize sizes
            min_size = min(sizes)
            max_size = max(sizes)
            if max_size > min_size:
                normalized_sizes = [(s - min_size) / (max_size - min_size) * 30 + 5 for s in sizes]
            else:
                normalized_sizes = [15] * len(sizes)
            trace['marker']['size'] = normalized_sizes
            
        layout = {
            'title': f'{y_prop} vs {x_prop}',
            'xaxis': {'title': x_prop},
            'yaxis': {'title': y_prop},
            'hovermode': 'closest'
        }
        
        return {
            'data': [trace],
            'layout': layout
        }
        
    def _format_matplotlib_scatter(self, data_points: List[Dict]) -> Dict:
        """Format data for Matplotlib."""
        return {
            'x': np.array([p['x'] for p in data_points]),
            'y': np.array([p['y'] for p in data_points]),
            'labels': [p['material_id'] for p in data_points],
            'colors': np.array([p.get('color', 0) for p in data_points]) if any('color' in p for p in data_points) else None,
            'sizes': np.array([p.get('size', 50) for p in data_points]) if any('size' in p for p in data_points) else None
        }
        
    def _format_bokeh_scatter(self, data_points: List[Dict]) -> Dict:
        """Format data for Bokeh."""
        source_data = {
            'x': [p['x'] for p in data_points],
            'y': [p['y'] for p in data_points],
            'material_id': [p['material_id'] for p in data_points],
            'formula': [p['formula'] for p in data_points]
        }
        
        if any('color' in p for p in data_points):
            source_data['color'] = [p.get('color', 0) for p in data_points]
            
        if any('size' in p for p in data_points):
            source_data['size'] = [p.get('size', 10) for p in data_points]
            
        return {'source': source_data}
        
    def _format_vega_scatter(self, data_points: List[Dict], x_prop: str, y_prop: str,
                           color_by: str = None) -> Dict:
        """Format as Vega-Lite specification."""
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": f"Scatter plot of {y_prop} vs {x_prop}",
            "data": {"values": data_points},
            "mark": {"type": "circle", "tooltip": True},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": x_prop},
                "y": {"field": "y", "type": "quantitative", "title": y_prop},
                "tooltip": [
                    {"field": "material_id", "type": "nominal"},
                    {"field": "formula", "type": "nominal"},
                    {"field": "x", "type": "quantitative", "title": x_prop},
                    {"field": "y", "type": "quantitative", "title": y_prop}
                ]
            }
        }
        
        if color_by and any('color' in p for p in data_points):
            spec["encoding"]["color"] = {
                "field": "color",
                "type": "quantitative",
                "title": color_by,
                "scale": {"scheme": "viridis"}
            }
            
        return spec
        
    def _format_d3_scatter(self, data_points: List[Dict]) -> Dict:
        """Format for D3.js."""
        return {
            'data': data_points,
            'metadata': {
                'count': len(data_points),
                'x_range': [min(p['x'] for p in data_points), max(p['x'] for p in data_points)] if data_points else [0, 1],
                'y_range': [min(p['y'] for p in data_points), max(p['y'] for p in data_points)] if data_points else [0, 1]
            }
        }
        
    def _format_gnuplot_scatter(self, data_points: List[Dict]) -> str:
        """Format for Gnuplot."""
        lines = ["# x y material_id formula"]
        for p in data_points:
            lines.append(f"{p['x']} {p['y']} # {p['material_id']} {p['formula']}")
        return '\n'.join(lines)
        
    def export_histogram_data(self, property_name: str,
                            n_bins: int = 20,
                            material_ids: List[str] = None,
                            format: str = 'plotly') -> Dict[str, Any]:
        """
        Export histogram data.
        
        Args:
            property_name: Property to create histogram for
            n_bins: Number of bins
            material_ids: Specific materials to include
            format: Export format
            
        Returns:
            Formatted histogram data
        """
        # Collect values
        values = []
        
        materials = self.db.get_all_materials()
        if material_ids:
            materials = [m for m in materials if m['material_id'] in material_ids]
            
        for material in materials:
            props = self.db.get_material_properties(material['material_id'])
            for prop in props:
                if prop['property_name'] == property_name:
                    try:
                        values.append(float(prop['property_value']))
                    except (ValueError, TypeError):
                        pass
                    break
                    
        if not values:
            return {'error': f'No numeric values found for {property_name}'}
            
        # Calculate histogram
        hist, bin_edges = np.histogram(values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Format based on requested format
        if format == 'plotly':
            return self._format_plotly_histogram(values, property_name)
        elif format == 'matplotlib':
            return {
                'values': np.array(values),
                'bins': n_bins,
                'range': (min(values), max(values))
            }
        elif format == 'vega':
            return self._format_vega_histogram(values, property_name, n_bins)
        else:
            return {
                'bin_centers': bin_centers.tolist(),
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
            
    def _format_plotly_histogram(self, values: List[float], property_name: str) -> Dict:
        """Format histogram for Plotly."""
        return {
            'data': [{
                'type': 'histogram',
                'x': values,
                'name': property_name,
                'marker': {'color': 'rgba(0, 119, 190, 0.7)'}
            }],
            'layout': {
                'title': f'Distribution of {property_name}',
                'xaxis': {'title': property_name},
                'yaxis': {'title': 'Count'},
                'bargap': 0.05
            }
        }
        
    def _format_vega_histogram(self, values: List[float], property_name: str, n_bins: int) -> Dict:
        """Format histogram as Vega-Lite specification."""
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": f"Histogram of {property_name}",
            "data": {"values": [{"value": v} for v in values]},
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "value",
                    "type": "quantitative",
                    "bin": {"maxbins": n_bins},
                    "title": property_name
                },
                "y": {
                    "aggregate": "count",
                    "type": "quantitative",
                    "title": "Count"
                }
            }
        }
        
    def export_heatmap_data(self, properties: List[str],
                          material_ids: List[str] = None,
                          normalize: bool = True,
                          format: str = 'plotly') -> Dict[str, Any]:
        """
        Export correlation heatmap data.
        
        Args:
            properties: Properties to correlate
            material_ids: Specific materials to include
            normalize: Whether to normalize values
            format: Export format
            
        Returns:
            Formatted heatmap data
        """
        # Collect property values
        prop_values = defaultdict(list)
        
        materials = self.db.get_all_materials()
        if material_ids:
            materials = [m for m in materials if m['material_id'] in material_ids]
            
        for material in materials:
            mat_props = self.db.get_material_properties(material['material_id'])
            prop_dict = {p['property_name']: p['property_value'] for p in mat_props}
            
            # Check if material has all requested properties
            has_all = True
            values = {}
            for prop in properties:
                if prop in prop_dict:
                    try:
                        values[prop] = float(prop_dict[prop])
                    except (ValueError, TypeError):
                        has_all = False
                        break
                else:
                    has_all = False
                    break
                    
            if has_all:
                for prop, value in values.items():
                    prop_values[prop].append(value)
                    
        # Calculate correlation matrix
        n_props = len(properties)
        corr_matrix = np.zeros((n_props, n_props))
        
        for i, prop1 in enumerate(properties):
            for j, prop2 in enumerate(properties):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    if prop_values[prop1] and prop_values[prop2]:
                        corr = np.corrcoef(prop_values[prop1], prop_values[prop2])[0, 1]
                        corr_matrix[i, j] = corr
                        
        # Format based on requested format
        if format == 'plotly':
            return self._format_plotly_heatmap(corr_matrix, properties)
        elif format == 'matplotlib':
            return {
                'matrix': corr_matrix,
                'labels': properties
            }
        else:
            return {
                'matrix': corr_matrix.tolist(),
                'labels': properties
            }
            
    def _format_plotly_heatmap(self, matrix: np.ndarray, labels: List[str]) -> Dict:
        """Format heatmap for Plotly."""
        return {
            'data': [{
                'type': 'heatmap',
                'z': matrix.tolist(),
                'x': labels,
                'y': labels,
                'colorscale': 'RdBu',
                'zmid': 0,
                'text': [[f'{val:.2f}' for val in row] for row in matrix],
                'texttemplate': '%{text}',
                'textfont': {'size': 10}
            }],
            'layout': {
                'title': 'Property Correlation Heatmap',
                'xaxis': {'tickangle': -45},
                'yaxis': {'autorange': 'reversed'}
            }
        }
        
    def export_3d_scatter(self, x_property: str, y_property: str, z_property: str,
                         material_ids: List[str] = None,
                         color_by: Optional[str] = None,
                         format: str = 'plotly') -> Dict[str, Any]:
        """
        Export 3D scatter plot data.
        
        Args:
            x_property: Property for x-axis
            y_property: Property for y-axis
            z_property: Property for z-axis
            material_ids: Specific materials to include
            color_by: Property to color points by
            format: Export format
            
        Returns:
            Formatted 3D visualization data
        """
        # Collect data
        data_points = []
        
        materials = self.db.get_all_materials()
        if material_ids:
            materials = [m for m in materials if m['material_id'] in material_ids]
            
        for material in materials:
            mat_id = material['material_id']
            props = self.db.get_material_properties(mat_id)
            prop_dict = {p['property_name']: p['property_value'] for p in props}
            
            if all(p in prop_dict for p in [x_property, y_property, z_property]):
                try:
                    point = {
                        'material_id': mat_id,
                        'formula': material.get('formula', 'Unknown'),
                        'x': float(prop_dict[x_property]),
                        'y': float(prop_dict[y_property]),
                        'z': float(prop_dict[z_property])
                    }
                    
                    if color_by and color_by in prop_dict:
                        try:
                            point['color'] = float(prop_dict[color_by])
                        except ValueError:
                            point['color'] = prop_dict[color_by]
                            
                    data_points.append(point)
                    
                except (ValueError, TypeError):
                    pass
                    
        # Format based on requested format
        if format == 'plotly':
            return self._format_plotly_3d_scatter(data_points, x_property, y_property, z_property, color_by)
        else:
            return {
                'points': data_points,
                'axes': {
                    'x': x_property,
                    'y': y_property,
                    'z': z_property
                }
            }
            
    def _format_plotly_3d_scatter(self, data_points: List[Dict], x_prop: str, y_prop: str,
                                z_prop: str, color_by: str = None) -> Dict:
        """Format 3D scatter for Plotly."""
        trace = {
            'type': 'scatter3d',
            'mode': 'markers',
            'x': [p['x'] for p in data_points],
            'y': [p['y'] for p in data_points],
            'z': [p['z'] for p in data_points],
            'text': [f"{p['material_id']}<br>{p['formula']}" for p in data_points],
            'hovertemplate': '%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<br>%{zaxis.title.text}: %{z}<extra></extra>',
            'marker': {'size': 5}
        }
        
        if color_by and any('color' in p for p in data_points):
            trace['marker']['color'] = [p.get('color', 0) for p in data_points]
            trace['marker']['colorscale'] = 'Viridis'
            trace['marker']['showscale'] = True
            trace['marker']['colorbar'] = {'title': color_by}
            
        layout = {
            'title': f'{z_prop} vs {y_prop} vs {x_prop}',
            'scene': {
                'xaxis': {'title': x_prop},
                'yaxis': {'title': y_prop},
                'zaxis': {'title': z_prop}
            }
        }
        
        return {
            'data': [trace],
            'layout': layout
        }
        
    def save_visualization(self, viz_data: Dict[str, Any], filename: str,
                         format: str = 'json') -> str:
        """
        Save visualization data to file.
        
        Args:
            viz_data: Visualization data
            filename: Output filename
            format: File format (json, html, csv)
            
        Returns:
            Path to saved file
        """
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(viz_data, f, indent=2)
                
        elif format == 'html' and 'data' in viz_data and 'layout' in viz_data:
            # Create simple HTML with Plotly
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <title>MACE Visualization</title>
</head>
<body>
    <div id="plot" style="width: 100%; height: 600px;"></div>
    <script>
        var data = {data};
        var layout = {layout};
        Plotly.newPlot('plot', data, layout);
    </script>
</body>
</html>
"""
            html_content = html_template.format(
                data=json.dumps(viz_data['data']),
                layout=json.dumps(viz_data['layout'])
            )
            
            with open(filename, 'w') as f:
                f.write(html_content)
                
        elif format == 'csv' and isinstance(viz_data, str):
            # For gnuplot format
            with open(filename, 'w') as f:
                f.write(viz_data)
                
        else:
            raise ValueError(f"Unsupported save format: {format}")
            
        return filename