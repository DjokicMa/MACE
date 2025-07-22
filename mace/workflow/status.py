#!/usr/bin/env python3
"""
Enhanced Workflow Status Checking
==================================
Provides detailed status information for active workflows.

Usage:
  python workflow_status.py [workflow_id]
  python workflow_status.py --all
  python workflow_status.py --active
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import MACE components
try:
    from database.materials import MaterialDatabase
    from workflow.engine import WorkflowEngine
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class WorkflowStatusChecker:
    """Enhanced workflow status checking"""
    
    def __init__(self, work_dir: str = ".", db_path: str = "materials.db"):
        self.work_dir = Path(work_dir).resolve()
        self.db_path = db_path
        self.db = MaterialDatabase(db_path)
        self.engine = WorkflowEngine(db_path, str(work_dir))
        
        self.configs_dir = self.work_dir / "workflow_configs"
        self.outputs_dir = self.work_dir / "workflow_outputs"
        
    def find_all_workflows(self) -> List[Dict[str, Any]]:
        """Find all workflow plans"""
        workflows = []
        
        if not self.configs_dir.exists():
            return workflows
            
        for plan_file in self.configs_dir.glob("workflow_plan_*.json"):
            try:
                with open(plan_file, 'r') as f:
                    plan = json.load(f)
                    plan['plan_file'] = str(plan_file)
                    workflows.append(plan)
            except Exception as e:
                print(f"Error reading {plan_file}: {e}")
                
        return workflows
        
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed status for a specific workflow"""
        # Find workflow plan
        workflow_plan = None
        for plan in self.find_all_workflows():
            if plan.get('workflow_id') == workflow_id:
                workflow_plan = plan
                break
                
        if not workflow_plan:
            return {"error": f"Workflow {workflow_id} not found"}
            
        status = {
            "workflow_id": workflow_id,
            "created": workflow_plan.get('created', 'Unknown'),
            "sequence": workflow_plan.get('workflow_sequence', []),
            "input_type": workflow_plan.get('input_type', 'Unknown'),
            "materials": {},
            "overall_progress": {},
            "active_jobs": [],
            "failed_jobs": [],
            "completed_steps": []
        }
        
        # Get all calculations with this workflow_id
        all_calcs = self.db.get_all_calculations()
        workflow_calcs = []
        
        for calc in all_calcs:
            settings = json.loads(calc.get('settings_json', '{}'))
            if settings.get('workflow_id') == workflow_id:
                workflow_calcs.append(calc)
                
        # Group by material
        for calc in workflow_calcs:
            material_id = calc['material_id']
            if material_id not in status['materials']:
                material = self.db.get_material(material_id)
                status['materials'][material_id] = {
                    'formula': material.get('formula', 'Unknown') if material else 'Unknown',
                    'calculations': {},
                    'progress': []
                }
                
            calc_type = calc['calc_type']
            status['materials'][material_id]['calculations'][calc_type] = {
                'calc_id': calc['calc_id'],
                'status': calc['status'],
                'start_time': calc.get('start_time', 'Not started'),
                'end_time': calc.get('end_time', 'Not completed'),
                'slurm_job_id': calc.get('slurm_job_id', 'None')
            }
            
            # Track active and failed jobs
            if calc['status'] == 'running':
                status['active_jobs'].append({
                    'material': material_id,
                    'calc_type': calc_type,
                    'job_id': calc.get('slurm_job_id', 'Unknown')
                })
            elif calc['status'] == 'failed':
                status['failed_jobs'].append({
                    'material': material_id,
                    'calc_type': calc_type,
                    'error': calc.get('error_message', 'Unknown error')
                })
                
        # Calculate overall progress
        total_steps = len(workflow_plan.get('workflow_sequence', [])) * len(status['materials'])
        completed_steps = sum(
            1 for mat_data in status['materials'].values()
            for calc_data in mat_data['calculations'].values()
            if calc_data['status'] == 'completed'
        )
        
        status['overall_progress'] = {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'percentage': (completed_steps / total_steps * 100) if total_steps > 0 else 0
        }
        
        return status
        
    def print_workflow_status(self, workflow_id: str):
        """Print formatted workflow status"""
        status = self.get_workflow_status(workflow_id)
        
        if 'error' in status:
            print(f"❌ {status['error']}")
            return
            
        print(f"\n{'='*80}")
        print(f"Workflow Status: {workflow_id}")
        print(f"{'='*80}")
        print(f"Created: {status['created']}")
        print(f"Sequence: {' → '.join(status['sequence'])}")
        print(f"Progress: {status['overall_progress']['completed_steps']}/{status['overall_progress']['total_steps']} ({status['overall_progress']['percentage']:.1f}%)")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * status['overall_progress']['percentage'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"[{bar}]")
        
        print(f"\nMaterials ({len(status['materials'])}):")
        print("-" * 80)
        
        for material_id, mat_data in status['materials'].items():
            print(f"\n{material_id} ({mat_data['formula']})")
            
            # Show calculation progress for this material
            for calc_type in status['sequence']:
                if calc_type in mat_data['calculations']:
                    calc_info = mat_data['calculations'][calc_type]
                    status_symbol = {
                        'completed': '✅',
                        'running': '🔄',
                        'failed': '❌',
                        'submitted': '📤',
                        'pending': '⏸️'
                    }.get(calc_info['status'], '❓')
                    
                    print(f"  {status_symbol} {calc_type}: {calc_info['status']}", end="")
                    if calc_info['slurm_job_id'] != 'None':
                        print(f" (Job: {calc_info['slurm_job_id']})", end="")
                    print()
                else:
                    print(f"  ⏸️ {calc_type}: Not started")
                    
        if status['active_jobs']:
            print(f"\nActive Jobs ({len(status['active_jobs'])}):")
            for job in status['active_jobs']:
                print(f"  - {job['material']} {job['calc_type']} (Job: {job['job_id']})")
                
        if status['failed_jobs']:
            print(f"\n⚠️ Failed Jobs ({len(status['failed_jobs'])}):")
            for job in status['failed_jobs']:
                print(f"  - {job['material']} {job['calc_type']}: {job['error']}")
                
    def print_all_workflows(self, active_only: bool = False):
        """Print summary of all workflows"""
        workflows = self.find_all_workflows()
        
        if not workflows:
            print("No workflows found.")
            return
            
        print(f"\n{'='*80}")
        print(f"All Workflows ({len(workflows)} total)")
        print(f"{'='*80}")
        
        for workflow in workflows:
            workflow_id = workflow.get('workflow_id', 'Unknown')
            status = self.get_workflow_status(workflow_id)
            
            if 'error' in status:
                continue
                
            progress = status['overall_progress']
            is_active = len(status['active_jobs']) > 0
            
            if active_only and not is_active:
                continue
                
            # Status icon
            if progress['percentage'] == 100:
                icon = "✅"
            elif is_active:
                icon = "🔄"
            elif len(status['failed_jobs']) > 0:
                icon = "⚠️"
            else:
                icon = "⏸️"
                
            print(f"\n{icon} {workflow_id}")
            print(f"   Created: {workflow.get('created', 'Unknown')}")
            print(f"   Progress: {progress['completed_steps']}/{progress['total_steps']} ({progress['percentage']:.1f}%)")
            print(f"   Materials: {len(status['materials'])}")
            
            if is_active:
                print(f"   Active: {len(status['active_jobs'])} jobs running")
            if status['failed_jobs']:
                print(f"   Failed: {len(status['failed_jobs'])} jobs failed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Check workflow status")
    parser.add_argument("workflow_id", nargs="?", help="Specific workflow ID to check")
    parser.add_argument("--all", action="store_true", help="Show all workflows")
    parser.add_argument("--active", action="store_true", help="Show only active workflows")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db-path", default="materials.db", help="Database path")
    
    args = parser.parse_args()
    
    checker = WorkflowStatusChecker(args.work_dir, args.db_path)
    
    if args.all or args.active:
        checker.print_all_workflows(active_only=args.active)
    elif args.workflow_id:
        checker.print_workflow_status(args.workflow_id)
    else:
        # Show active workflows by default
        print("Showing active workflows. Use --all to see all workflows.")
        checker.print_all_workflows(active_only=True)


if __name__ == "__main__":
    main()