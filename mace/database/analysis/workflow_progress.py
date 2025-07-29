"""
Workflow Progress Tracking
==========================
Track and visualize the completion status of multi-step workflows.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from collections import defaultdict, OrderedDict


class WorkflowProgress:
    """Tracks and analyzes workflow progress for materials."""
    
    # Standard workflow sequences
    WORKFLOWS = {
        'basic_opt': ['OPT'],
        'opt_sp': ['OPT', 'SP'],
        'full_electronic': ['OPT', 'SP', 'BAND', 'DOSS'],
        'transport_analysis': ['OPT', 'SP', 'TRANSPORT'],
        'charge_analysis': ['OPT', 'SP', 'CHARGE+POTENTIAL'],
        'combined_analysis': ['OPT', 'SP', 'BAND', 'DOSS', 'TRANSPORT'],
        'complete': ['OPT', 'SP', 'BAND', 'DOSS', 'FREQ']
    }
    
    def __init__(self, db):
        """
        Initialize with database connection.
        
        Args:
            db: MaterialDatabase instance
        """
        self.db = db
        
    def track_progress(self, material_ids: List[str] = None,
                      workflow: str = None,
                      custom_sequence: List[str] = None) -> Dict[str, Any]:
        """
        Track workflow progress for materials.
        
        Args:
            material_ids: Specific materials to track (None = all)
            workflow: Predefined workflow name (from WORKFLOWS)
            custom_sequence: Custom calculation sequence to track
            
        Returns:
            Progress tracking results
        """
        # Determine workflow sequence
        if custom_sequence:
            sequence = custom_sequence
            workflow_name = 'custom'
        elif workflow and workflow in self.WORKFLOWS:
            sequence = self.WORKFLOWS[workflow]
            workflow_name = workflow
        else:
            # Default to full electronic workflow
            sequence = self.WORKFLOWS['full_electronic']
            workflow_name = 'full_electronic'
            
        # Get materials
        if material_ids:
            materials = [self.db.get_material(mat_id) for mat_id in material_ids]
            materials = [m for m in materials if m]  # Filter None
        else:
            materials = self.db.get_all_materials()
            
        # Track progress for each material
        results = {
            'workflow': workflow_name,
            'sequence': sequence,
            'materials': {},
            'summary': {
                'total_materials': len(materials),
                'completed': 0,
                'in_progress': 0,
                'not_started': 0,
                'failed': 0,
                'average_completion': 0.0
            }
        }
        
        total_completion = 0.0
        
        for material in materials:
            mat_id = material['material_id']
            progress = self._track_material_progress(mat_id, sequence)
            results['materials'][mat_id] = progress
            
            # Update summary
            total_completion += progress['completion_percentage']
            
            if progress['status'] == 'completed':
                results['summary']['completed'] += 1
            elif progress['status'] == 'in_progress':
                results['summary']['in_progress'] += 1
            elif progress['status'] == 'not_started':
                results['summary']['not_started'] += 1
            elif progress['status'] == 'failed':
                results['summary']['failed'] += 1
                
        # Calculate average completion
        if materials:
            results['summary']['average_completion'] = total_completion / len(materials)
            
        # Add workflow insights
        self._add_workflow_insights(results)
        
        return results
        
    def _track_material_progress(self, material_id: str, sequence: List[str]) -> Dict[str, Any]:
        """Track progress for a single material."""
        # Get all calculations for this material
        calculations = self.db.get_calculations_for_material(material_id)
        
        # Group by calculation type
        calc_by_type = defaultdict(list)
        for calc in calculations:
            calc_type = calc.get('calculation_type', 'UNKNOWN')
            calc_by_type[calc_type].append(calc)
            
        # Track each step in sequence
        steps = []
        completed_count = 0
        current_step = None
        overall_status = 'not_started'
        
        for i, calc_type in enumerate(sequence):
            step_info = {
                'step': i + 1,
                'type': calc_type,
                'status': 'pending',
                'calculation_id': None,
                'job_id': None,
                'started_at': None,
                'completed_at': None,
                'runtime_hours': None
            }
            
            # Check if this calculation exists
            if calc_type in calc_by_type:
                calcs = calc_by_type[calc_type]
                
                # Find the most recent calculation
                latest_calc = max(calcs, key=lambda c: c.get('started_at', ''))
                
                step_info['calculation_id'] = latest_calc.get('calculation_id')
                step_info['job_id'] = latest_calc.get('job_id')
                step_info['started_at'] = latest_calc.get('started_at')
                step_info['completed_at'] = latest_calc.get('completed_at')
                
                # Determine status
                status = latest_calc.get('status', 'unknown')
                if status == 'completed':
                    step_info['status'] = 'completed'
                    completed_count += 1
                    
                    # Calculate runtime
                    if step_info['started_at'] and step_info['completed_at']:
                        try:
                            start = datetime.fromisoformat(step_info['started_at'])
                            end = datetime.fromisoformat(step_info['completed_at'])
                            runtime = end - start
                            step_info['runtime_hours'] = runtime.total_seconds() / 3600
                        except:
                            pass
                            
                elif status in ['running', 'submitted', 'pending']:
                    step_info['status'] = 'running'
                    if not current_step:
                        current_step = calc_type
                        overall_status = 'in_progress'
                elif status == 'failed':
                    step_info['status'] = 'failed'
                    overall_status = 'failed'
                    
                    # Check for recovery attempts
                    recovery_count = sum(1 for c in calcs if 'recovery' in c.get('notes', ''))
                    if recovery_count > 0:
                        step_info['recovery_attempts'] = recovery_count
                        
            # Check dependencies
            if i > 0 and steps[i-1]['status'] != 'completed':
                step_info['blocked_by'] = sequence[i-1]
                
            steps.append(step_info)
            
        # Calculate completion percentage
        completion = (completed_count / len(sequence)) * 100 if sequence else 0
        
        # Determine overall status
        if completed_count == len(sequence):
            overall_status = 'completed'
        elif completed_count == 0 and not current_step:
            overall_status = 'not_started'
            
        return {
            'material_id': material_id,
            'workflow_steps': steps,
            'completed_steps': completed_count,
            'total_steps': len(sequence),
            'completion_percentage': completion,
            'status': overall_status,
            'current_step': current_step,
            'total_runtime_hours': sum(s['runtime_hours'] or 0 for s in steps)
        }
        
    def _add_workflow_insights(self, results: Dict):
        """Add insights about workflow execution."""
        insights = []
        
        # Identify bottlenecks
        step_times = defaultdict(list)
        step_failures = defaultdict(int)
        
        for mat_progress in results['materials'].values():
            for step in mat_progress['workflow_steps']:
                if step['runtime_hours']:
                    step_times[step['type']].append(step['runtime_hours'])
                if step['status'] == 'failed':
                    step_failures[step['type']] += 1
                    
        # Find slowest steps
        if step_times:
            avg_times = {
                step: sum(times) / len(times) 
                for step, times in step_times.items()
            }
            slowest_step = max(avg_times.items(), key=lambda x: x[1])
            insights.append({
                'type': 'performance',
                'message': f"Slowest step: {slowest_step[0]} (avg {slowest_step[1]:.1f} hours)"
            })
            
        # Find most error-prone steps
        if step_failures:
            worst_step = max(step_failures.items(), key=lambda x: x[1])
            if worst_step[1] > 0:
                insights.append({
                    'type': 'reliability',
                    'message': f"Most failures: {worst_step[0]} ({worst_step[1]} failures)"
                })
                
        # Check for stuck workflows
        stuck_count = sum(
            1 for m in results['materials'].values()
            if m['status'] == 'in_progress' and 
            any(s['status'] == 'running' and s['runtime_hours'] and s['runtime_hours'] > 168
                for s in m['workflow_steps'])
        )
        
        if stuck_count > 0:
            insights.append({
                'type': 'warning',
                'message': f"{stuck_count} materials have jobs running > 7 days"
            })
            
        results['insights'] = insights
        
    def get_workflow_summary(self, workflow: str = None) -> Dict[str, Any]:
        """
        Get summary statistics for a specific workflow.
        
        Args:
            workflow: Workflow name (None = analyze all)
            
        Returns:
            Summary statistics
        """
        if workflow and workflow not in self.WORKFLOWS:
            return {'error': f"Unknown workflow: {workflow}"}
            
        # Track all workflows or specific one
        workflow_stats = {}
        
        workflows_to_check = [workflow] if workflow else list(self.WORKFLOWS.keys())
        
        for wf_name in workflows_to_check:
            progress = self.track_progress(workflow=wf_name)
            
            workflow_stats[wf_name] = {
                'sequence': self.WORKFLOWS[wf_name],
                'total_materials': progress['summary']['total_materials'],
                'completed': progress['summary']['completed'],
                'in_progress': progress['summary']['in_progress'],
                'failed': progress['summary']['failed'],
                'average_completion': progress['summary']['average_completion'],
                'insights': progress.get('insights', [])
            }
            
        return workflow_stats
        
    def format_progress_report(self, results: Dict, detailed: bool = False) -> str:
        """
        Format workflow progress as readable report.
        
        Args:
            results: Results from track_progress()
            detailed: Include per-material details
            
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        lines.append("=== Workflow Progress Report ===")
        lines.append(f"Workflow: {results['workflow']}")
        lines.append(f"Sequence: {' → '.join(results['sequence'])}")
        lines.append("")
        
        # Summary
        summary = results['summary']
        lines.append("=== Summary ===")
        lines.append(f"Total materials: {summary['total_materials']}")
        lines.append(f"Completed: {summary['completed']} ({summary['completed']/summary['total_materials']*100:.1f}%)")
        lines.append(f"In progress: {summary['in_progress']}")
        lines.append(f"Not started: {summary['not_started']}")
        lines.append(f"Failed: {summary['failed']}")
        lines.append(f"Average completion: {summary['average_completion']:.1f}%")
        lines.append("")
        
        # Progress visualization
        if summary['total_materials'] > 0:
            lines.append("=== Progress Distribution ===")
            
            # Create histogram
            bins = [0, 25, 50, 75, 100]
            bin_counts = [0] * (len(bins) - 1)
            
            for mat_progress in results['materials'].values():
                pct = mat_progress['completion_percentage']
                for i in range(len(bins) - 1):
                    if bins[i] <= pct < bins[i+1] or (i == len(bins) - 2 and pct == 100):
                        bin_counts[i] += 1
                        break
                        
            for i in range(len(bins) - 1):
                label = f"{bins[i]:3d}-{bins[i+1]:3d}%"
                count = bin_counts[i]
                bar = "█" * int(count * 30 / max(summary['total_materials'], 1))
                lines.append(f"{label}: {bar} {count}")
                
            lines.append("")
            
        # Insights
        if 'insights' in results and results['insights']:
            lines.append("=== Insights ===")
            for insight in results['insights']:
                lines.append(f"[{insight['type'].upper()}] {insight['message']}")
            lines.append("")
            
        # Detailed material progress
        if detailed:
            lines.append("=== Material Details ===")
            
            # Sort materials by completion percentage
            sorted_materials = sorted(
                results['materials'].items(),
                key=lambda x: (-x[1]['completion_percentage'], x[0])
            )
            
            for mat_id, progress in sorted_materials[:20]:  # Show top 20
                status_symbol = {
                    'completed': '✓',
                    'in_progress': '→',
                    'failed': '✗',
                    'not_started': '·'
                }.get(progress['status'], '?')
                
                lines.append(f"\n{mat_id} [{status_symbol}] {progress['completion_percentage']:.0f}% complete")
                
                # Show step details
                for step in progress['workflow_steps']:
                    step_symbol = {
                        'completed': '✓',
                        'running': '→',
                        'failed': '✗',
                        'pending': '·'
                    }.get(step['status'], '?')
                    
                    step_line = f"  {step['step']}. {step['type']:20} [{step_symbol}]"
                    
                    if step['runtime_hours']:
                        step_line += f" ({step['runtime_hours']:.1f}h)"
                        
                    if step.get('blocked_by'):
                        step_line += f" - blocked by {step['blocked_by']}"
                        
                    if step.get('recovery_attempts'):
                        step_line += f" - {step['recovery_attempts']} recovery attempts"
                        
                    lines.append(step_line)
                    
            if len(results['materials']) > 20:
                lines.append(f"\n... and {len(results['materials']) - 20} more materials")
                
        return "\n".join(lines)
        
    def export_progress_data(self, results: Dict, format: str = 'json') -> str:
        """
        Export progress data in various formats.
        
        Args:
            results: Progress tracking results
            format: Export format (json, csv)
            
        Returns:
            Formatted data string
        """
        if format == 'json':
            return json.dumps(results, indent=2, default=str)
            
        elif format == 'csv':
            lines = []
            
            # Header
            lines.append("material_id,workflow,total_steps,completed_steps,completion_percentage,status,current_step,runtime_hours")
            
            # Data rows
            for mat_id, progress in results['materials'].items():
                lines.append(','.join([
                    mat_id,
                    results['workflow'],
                    str(progress['total_steps']),
                    str(progress['completed_steps']),
                    f"{progress['completion_percentage']:.1f}",
                    progress['status'],
                    progress['current_step'] or '',
                    f"{progress['total_runtime_hours']:.1f}"
                ]))
                
            return '\n'.join(lines)
            
        else:
            return f"Unsupported format: {format}"


def track_workflow_progress(db, material_ids: List[str] = None,
                          workflow: str = None,
                          custom_sequence: List[str] = None,
                          output_format: str = 'report',
                          detailed: bool = False) -> str:
    """
    Convenience function to track workflow progress.
    
    Args:
        db: MaterialDatabase instance
        material_ids: Specific materials to track
        workflow: Predefined workflow name
        custom_sequence: Custom calculation sequence
        output_format: 'report', 'json', 'csv', or 'dict'
        detailed: Include detailed per-material info
        
    Returns:
        Formatted results
    """
    tracker = WorkflowProgress(db)
    results = tracker.track_progress(material_ids, workflow, custom_sequence)
    
    if output_format == 'report':
        return tracker.format_progress_report(results, detailed)
    elif output_format == 'json':
        return json.dumps(results, indent=2, default=str)
    elif output_format == 'csv':
        return tracker.export_progress_data(results, 'csv')
    else:  # dict
        return results