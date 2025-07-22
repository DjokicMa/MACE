#!/usr/bin/env python3
"""
Material Monitor - CLI Tools for CRYSTAL Material Tracking System
----------------------------------------------------------------
Provides command-line tools for monitoring material status, database health,
and workflow progress in the CRYSTAL material tracking system.

Key Features:
- Real-time status monitoring
- Database health checks
- Progress reporting and statistics
- Interactive CLI dashboard
- Alert system for critical issues

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import subprocess
from collections import defaultdict, Counter
import signal

# Import MACE components
try:
    from database.materials import MaterialDatabase
    from utils.file_manager import CrystalFileManager
    try:
        from recovery.detector import CrystalErrorDetector
        HAS_ERROR_DETECTOR = True
    except ImportError:
        HAS_ERROR_DETECTOR = False
        print("Warning: error_detector module not available. Error analysis will be limited.")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Make sure all required Python files are in the same directory as {__file__}")
    sys.exit(1)


class MaterialMonitor:
    """
    CLI monitoring tools for the CRYSTAL material tracking system.
    
    Provides various monitoring capabilities including real-time status,
    database health, workflow progress, and alert systems.
    """
    
    def __init__(self, base_dir: str = ".", db_path: str = "materials.db"):
        self.base_dir = Path(base_dir).resolve()
        self.db_path = db_path
        self.running = False
        self.refresh_interval = 30  # seconds
        
        # Initialize components
        self.db = MaterialDatabase(db_path)
        self.file_manager = CrystalFileManager(base_dir, db_path)
        if HAS_ERROR_DETECTOR:
            self.error_detector = CrystalErrorDetector(base_dir, db_path)
        else:
            self.error_detector = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nShutdown signal received. Stopping monitor...")
        self.running = False
        
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'database': self._check_database_health(),
            'queue': self._check_queue_status(),
            'files': self._check_file_system_health(),
            'errors': self._check_recent_errors(),
            'performance': self._check_performance_metrics()
        }
        
        return status
        
    def _check_database_health(self) -> Dict[str, any]:
        """Check database health and connectivity."""
        health = {
            'status': 'unknown',
            'accessible': False,
            'size_mb': 0,
            'last_backup': None,
            'stats': {},
            'issues': []
        }
        
        try:
            # Test database connectivity
            stats = self.db.get_database_stats()
            health['accessible'] = True
            health['stats'] = stats
            health['size_mb'] = stats.get('db_size_mb', 0)
            
            # Check for potential issues
            if stats.get('total_materials', 0) == 0:
                health['issues'].append('No materials in database')
            
            if health['size_mb'] > 1000:  # Larger than 1GB
                health['issues'].append('Large database size - consider cleanup')
                
            # Check database file age
            db_file = Path(self.db_path)
            if db_file.exists():
                mod_time = datetime.fromtimestamp(db_file.stat().st_mtime)
                age = datetime.now() - mod_time
                if age > timedelta(hours=24):
                    health['issues'].append('Database not updated in 24+ hours')
                    
            health['status'] = 'healthy' if not health['issues'] else 'warning'
            
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Database connection failed: {e}")
            
        return health
        
    def _check_queue_status(self) -> Dict[str, any]:
        """Check SLURM queue status and job distribution."""
        queue = {
            'status': 'unknown',
            'total_jobs': 0,
            'by_status': {},
            'by_user': {},
            'recent_submissions': 0,
            'issues': []
        }
        
        try:
            # Get SLURM queue information
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'unknown'), '-o', '%i,%T,%S,%j'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                queue['total_jobs'] = len(lines)
                
                status_counts = Counter()
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            status_counts[parts[1]] += 1
                            
                queue['by_status'] = dict(status_counts)
                
                # Check for potential issues
                pending_jobs = status_counts.get('PENDING', 0)
                if pending_jobs > queue['total_jobs'] * 0.5:
                    queue['issues'].append(f'High number of pending jobs: {pending_jobs}')
                    
                failed_jobs = status_counts.get('FAILED', 0)
                if failed_jobs > 5:
                    queue['issues'].append(f'Multiple failed jobs: {failed_jobs}')
                    
                queue['status'] = 'healthy' if not queue['issues'] else 'warning'
                
            else:
                queue['status'] = 'error'
                queue['issues'].append('Cannot access SLURM queue')
                
        except subprocess.TimeoutExpired:
            queue['status'] = 'error'
            queue['issues'].append('SLURM queue check timed out')
        except Exception as e:
            queue['status'] = 'error'
            queue['issues'].append(f'Queue check failed: {e}')
            
        return queue
        
    def _check_file_system_health(self) -> Dict[str, any]:
        """Check file system health and organization."""
        files = {
            'status': 'unknown',
            'total_materials': 0,
            'total_files': 0,
            'total_size_mb': 0,
            'organization_score': 0,
            'orphaned_files': 0,
            'issues': []
        }
        
        try:
            # Generate file report
            report = self.file_manager.generate_file_report()
            
            files['total_materials'] = report['summary']['total_materials']
            files['total_files'] = report['summary']['total_files']
            files['total_size_mb'] = report['summary']['total_size_mb']
            
            # Calculate organization score (percentage of files in organized structure)
            organized_materials = len([m for m in report['materials'].values() 
                                     if m['total_files'] > 0])
            if files['total_materials'] > 0:
                files['organization_score'] = (organized_materials / files['total_materials']) * 100
                
            # Check for issues
            integrity_issues = report['summary']['integrity_issues']
            if integrity_issues > 0:
                files['issues'].append(f'File integrity issues: {integrity_issues}')
                
            if files['total_size_mb'] > 100000:  # More than 100GB
                files['issues'].append('Large total file size - consider cleanup')
                
            if files['organization_score'] < 80:
                files['issues'].append('Poor file organization - consider reorganizing')
                
            files['status'] = 'healthy' if not files['issues'] else 'warning'
            
        except Exception as e:
            files['status'] = 'error'
            files['issues'].append(f'File system check failed: {e}')
            
        return files
        
    def _check_recent_errors(self) -> Dict[str, any]:
        """Check for recent errors and trends."""
        errors = {
            'status': 'unknown',
            'recent_count': 0,
            'error_rate': 0,
            'trending_up': [],
            'critical_errors': 0,
            'issues': []
        }
        
        try:
            # Generate error report for last 24 hours
            if self.error_detector:
                report = self.error_detector.generate_error_report(days_back=1)
            else:
                report = {'error_summary': {}, 'trending_errors': {}, 'total_files_analyzed': 0}
            
            errors['recent_count'] = sum(report['error_summary'].values())
            
            if report['total_files_analyzed'] > 0:
                errors['error_rate'] = (errors['recent_count'] / report['total_files_analyzed']) * 100
                
            # Check for trending errors
            for error_type, trend in report['trending_errors'].items():
                if trend['direction'] == 'increasing':
                    errors['trending_up'].append(error_type)
                    
            # Count critical errors
            critical_types = ['memory_error', 'system_error', 'basis_error']
            errors['critical_errors'] = sum(report['error_summary'].get(et, 0) for et in critical_types)
            
            # Check for issues
            if errors['error_rate'] > 25:
                errors['issues'].append(f'High error rate: {errors["error_rate"]:.1f}%')
                
            if errors['critical_errors'] > 3:
                errors['issues'].append(f'Multiple critical errors: {errors["critical_errors"]}')
                
            if errors['trending_up']:
                errors['issues'].append(f'Increasing error types: {", ".join(errors["trending_up"])}')
                
            errors['status'] = 'healthy' if not errors['issues'] else 'warning'
            
        except Exception as e:
            errors['status'] = 'error'
            errors['issues'].append(f'Error check failed: {e}')
            
        return errors
        
    def _check_performance_metrics(self) -> Dict[str, any]:
        """Check system performance metrics."""
        performance = {
            'status': 'unknown',
            'avg_job_time': 0,
            'queue_throughput': 0,
            'success_rate': 0,
            'resource_efficiency': 0,
            'issues': []
        }
        
        try:
            # Get recent calculations from database
            recent_calcs = []
            for status in ['completed', 'failed']:
                calcs = self.db.get_calculations_by_status(status)
                # Filter to recent calculations (last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                for calc in calcs:
                    if calc.get('completed_at'):
                        completed_time = datetime.fromisoformat(calc['completed_at'])
                        if completed_time > cutoff:
                            recent_calcs.append(calc)
                            
            if recent_calcs:
                # Calculate success rate
                completed = len([c for c in recent_calcs if c['status'] == 'completed'])
                performance['success_rate'] = (completed / len(recent_calcs)) * 100
                
                # Calculate average job time (for completed jobs)
                completed_calcs = [c for c in recent_calcs if c['status'] == 'completed']
                if completed_calcs:
                    total_time = 0
                    valid_times = 0
                    
                    for calc in completed_calcs:
                        if calc.get('started_at') and calc.get('completed_at'):
                            start = datetime.fromisoformat(calc['started_at'])
                            end = datetime.fromisoformat(calc['completed_at'])
                            duration = (end - start).total_seconds() / 3600  # hours
                            total_time += duration
                            valid_times += 1
                            
                    if valid_times > 0:
                        performance['avg_job_time'] = total_time / valid_times
                        
                # Calculate throughput (jobs per day)
                performance['queue_throughput'] = len(recent_calcs) / 7
                
            # Check for performance issues
            if performance['success_rate'] < 75:
                performance['issues'].append(f'Low success rate: {performance["success_rate"]:.1f}%')
                
            if performance['avg_job_time'] > 24:
                performance['issues'].append(f'Long average job time: {performance["avg_job_time"]:.1f} hours')
                
            if performance['queue_throughput'] < 1:
                performance['issues'].append('Low queue throughput')
                
            performance['status'] = 'healthy' if not performance['issues'] else 'warning'
            
        except Exception as e:
            performance['status'] = 'error'
            performance['issues'].append(f'Performance check failed: {e}')
            
        return performance
        
    def print_status_dashboard(self, status: Dict[str, any]):
        """Print a formatted status dashboard to console."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print("CRYSTAL MATERIAL TRACKING SYSTEM - STATUS DASHBOARD")
        print("=" * 60)
        print(f"Last Updated: {status['timestamp']}")
        print()
        
        # Database Status
        db = status['database']
        db_status_color = self._get_status_color(db['status'])
        print(f"DATABASE: {db_status_color}{db['status'].upper()}\033[0m")
        if db['accessible']:
            print(f"  Materials: {db['stats'].get('total_materials', 0)}")
            print(f"  Size: {db['size_mb']:.1f} MB")
            print(f"  Calculations by status: {db['stats'].get('calculations_by_status', {})}")
        if db['issues']:
            for issue in db['issues']:
                print(f"  ⚠️  {issue}")
        print()
        
        # Queue Status  
        queue = status['queue']
        queue_status_color = self._get_status_color(queue['status'])
        print(f"QUEUE: {queue_status_color}{queue['status'].upper()}\033[0m")
        print(f"  Total jobs: {queue['total_jobs']}")
        print(f"  By status: {queue['by_status']}")
        if queue['issues']:
            for issue in queue['issues']:
                print(f"  ⚠️  {issue}")
        print()
        
        # File System Status
        files = status['files']
        files_status_color = self._get_status_color(files['status'])
        print(f"FILES: {files_status_color}{files['status'].upper()}\033[0m")
        print(f"  Materials: {files['total_materials']}")
        print(f"  Total files: {files['total_files']}")
        print(f"  Total size: {files['total_size_mb']:.1f} MB")
        print(f"  Organization: {files['organization_score']:.1f}%")
        if files['issues']:
            for issue in files['issues']:
                print(f"  ⚠️  {issue}")
        print()
        
        # Error Status
        errors = status['errors']
        errors_status_color = self._get_status_color(errors['status'])
        print(f"ERRORS: {errors_status_color}{errors['status'].upper()}\033[0m")
        print(f"  Recent errors (24h): {errors['recent_count']}")
        print(f"  Error rate: {errors['error_rate']:.1f}%")
        print(f"  Critical errors: {errors['critical_errors']}")
        if errors['trending_up']:
            print(f"  Trending up: {', '.join(errors['trending_up'])}")
        if errors['issues']:
            for issue in errors['issues']:
                print(f"  ⚠️  {issue}")
        print()
        
        # Performance Status
        perf = status['performance']
        perf_status_color = self._get_status_color(perf['status'])
        print(f"PERFORMANCE: {perf_status_color}{perf['status'].upper()}\033[0m")
        print(f"  Success rate: {perf['success_rate']:.1f}%")
        print(f"  Avg job time: {perf['avg_job_time']:.1f} hours")
        print(f"  Throughput: {perf['queue_throughput']:.1f} jobs/day")
        if perf['issues']:
            for issue in perf['issues']:
                print(f"  ⚠️  {issue}")
        print()
        
        # Overall system health
        all_statuses = [
            status['database']['status'],
            status['queue']['status'], 
            status['files']['status'],
            status['errors']['status'],
            status['performance']['status']
        ]
        
        if 'error' in all_statuses:
            overall = 'CRITICAL'
            color = '\033[91m'  # Red
        elif 'warning' in all_statuses:
            overall = 'WARNING'
            color = '\033[93m'  # Yellow
        else:
            overall = 'HEALTHY'
            color = '\033[92m'  # Green
            
        print("=" * 60)
        print(f"OVERALL SYSTEM STATUS: {color}{overall}\033[0m")
        print("=" * 60)
        print(f"Press Ctrl+C to stop monitoring")
        
    def _get_status_color(self, status: str) -> str:
        """Get ANSI color code for status."""
        if status == 'healthy':
            return '\033[92m'  # Green
        elif status == 'warning':
            return '\033[93m'  # Yellow
        elif status == 'error':
            return '\033[91m'  # Red
        else:
            return '\033[0m'   # Default
            
    def run_continuous_monitoring(self, interval: int = 30):
        """Run continuous monitoring with dashboard updates."""
        self.refresh_interval = interval
        self.running = True
        
        print(f"Starting continuous monitoring (refresh every {interval} seconds)...")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                status = self.get_system_status()
                self.print_status_dashboard(status)
                
                # Sleep with interruption check
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retry
                
        print("\nMonitoring stopped.")
        
    def generate_detailed_report(self, output_file: str = None) -> Dict[str, any]:
        """Generate detailed system report."""
        print("Generating detailed system report...")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'material_details': {},
            'error_analysis': {},
            'recommendations': []
        }
        
        # Get detailed material information
        try:
            materials = self.db.get_materials_by_status('active')
            for material in materials:
                mat_id = material['material_id']
                
                # Get calculations for this material
                calcs = self.db.get_calculations_by_status(material_id=mat_id)
                
                material_info = {
                    'formula': material['formula'],
                    'created_at': material['created_at'],
                    'total_calculations': len(calcs),
                    'by_status': Counter(c['status'] for c in calcs),
                    'by_type': Counter(c['calc_type'] for c in calcs),
                    'last_activity': max((c.get('completed_at', c.get('created_at', '')) 
                                        for c in calcs), default=None)
                }
                
                report['material_details'][mat_id] = material_info
                
        except Exception as e:
            report['material_details'] = {'error': str(e)}
            
        # Get detailed error analysis
        try:
            if self.error_detector:
                error_report = self.error_detector.generate_error_report(days_back=7)
                report['error_analysis'] = error_report
            else:
                report['error_analysis'] = {'error': 'Error detector not available'}
        except Exception as e:
            report['error_analysis'] = {'error': str(e)}
            
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Detailed report saved to {output_file}")
            
        return report
        
    def _generate_recommendations(self, report: Dict[str, any]) -> List[str]:
        """Generate system recommendations based on report data."""
        recommendations = []
        
        status = report['system_status']
        
        # Database recommendations
        if status['database']['status'] != 'healthy':
            for issue in status['database']['issues']:
                if 'Large database size' in issue:
                    recommendations.append("Consider running database cleanup to remove old failed calculations")
                elif 'not updated' in issue:
                    recommendations.append("Check if material tracking is running properly")
                    
        # Queue recommendations
        if status['queue']['status'] != 'healthy':
            for issue in status['queue']['issues']:
                if 'pending jobs' in issue:
                    recommendations.append("Check cluster resources and job scheduling policies")
                elif 'failed jobs' in issue:
                    recommendations.append("Investigate common failure causes and implement error recovery")
                    
        # File system recommendations  
        if status['files']['status'] != 'healthy':
            for issue in status['files']['issues']:
                if 'integrity issues' in issue:
                    recommendations.append("Run file integrity checks and repair corrupted files")
                elif 'file size' in issue:
                    recommendations.append("Implement automated file cleanup and archival")
                elif 'organization' in issue:
                    recommendations.append("Use file manager tools to reorganize calculation files")
                    
        # Error recommendations
        if status['errors']['status'] != 'healthy':
            for issue in status['errors']['issues']:
                if 'error rate' in issue:
                    recommendations.append("Review and optimize default calculation settings")
                elif 'critical errors' in issue:
                    recommendations.append("Address critical system errors immediately")
                elif 'Increasing error' in issue:
                    recommendations.append("Monitor trending errors and implement proactive fixes")
                    
        # Performance recommendations
        if status['performance']['status'] != 'healthy':
            for issue in status['performance']['issues']:
                if 'success rate' in issue:
                    recommendations.append("Implement better error recovery and job retry mechanisms")
                elif 'job time' in issue:
                    recommendations.append("Optimize calculation settings for better performance")
                elif 'throughput' in issue:
                    recommendations.append("Consider increasing job submission rate or parallelization")
                    
        # Add general recommendations if no specific issues
        if not recommendations:
            recommendations.extend([
                "System appears healthy - continue regular monitoring",
                "Consider implementing automated backups if not already in place",
                "Review and update error recovery strategies periodically"
            ])
            
        return recommendations
        
    def check_database_connectivity(self) -> bool:
        """Quick database connectivity check."""
        try:
            stats = self.db.get_database_stats()
            return True
        except:
            return False
            
    def get_quick_stats(self) -> Dict[str, any]:
        """Get quick statistics for command-line display."""
        stats = {}
        
        try:
            db_stats = self.db.get_database_stats()
            stats['materials'] = db_stats.get('total_materials', 0)
            stats['calculations'] = sum(db_stats.get('calculations_by_status', {}).values())
            stats['db_size_mb'] = db_stats.get('db_size_mb', 0)
        except:
            stats['materials'] = 'N/A'
            stats['calculations'] = 'N/A' 
            stats['db_size_mb'] = 'N/A'
            
        try:
            # Quick queue check
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', 'unknown')],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                stats['queue_jobs'] = len(lines)
            else:
                stats['queue_jobs'] = 'N/A'
        except:
            stats['queue_jobs'] = 'N/A'
            
        return stats


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="CRYSTAL Material Monitor")
    parser.add_argument("--base-dir", default=".", help="Base directory for monitoring")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--action", choices=['dashboard', 'status', 'report', 'stats'], 
                       default='dashboard', help="Monitoring action to perform")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Refresh interval for dashboard (seconds)")
    parser.add_argument("--output", help="Output file for reports")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = MaterialMonitor(args.base_dir, args.db_path)
    
    # Check database connectivity first
    if not monitor.check_database_connectivity():
        print("Warning: Cannot connect to database. Some features may not work.")
        
    if args.action == 'dashboard':
        monitor.run_continuous_monitoring(args.interval)
        
    elif args.action == 'status':
        status = monitor.get_system_status()
        monitor.print_status_dashboard(status)
        
    elif args.action == 'report':
        output_file = args.output or f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = monitor.generate_detailed_report(output_file)
        
        print(f"\nSystem Report Summary:")
        print(f"Materials: {len(report.get('material_details', {}))}")
        print(f"Overall Status: {report['system_status']}")
        if report['recommendations']:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(report['recommendations'][:3], 1):
                print(f"{i}. {rec}")
                
    elif args.action == 'stats':
        stats = monitor.get_quick_stats()
        print("Quick Statistics:")
        print(f"  Materials: {stats['materials']}")
        print(f"  Calculations: {stats['calculations']}")
        print(f"  Database size: {stats['db_size_mb']} MB")
        print(f"  Queue jobs: {stats['queue_jobs']}")


if __name__ == "__main__":
    main()