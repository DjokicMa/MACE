#!/usr/bin/env python3
"""
CRYSTAL23 SLURM Job Queue Manager with robust error handling
"""

import os
import sys
import subprocess
import time
import argparse
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

class JobQueueManager:
    def __init__(self, d12_dir, status_file, max_jobs=250, reserve_slots=30):
        self.d12_dir = os.path.abspath(d12_dir)
        self.status_file = status_file
        self.max_jobs = max_jobs
        self.reserve_slots = reserve_slots
        self.job_status = self.load_status()
        self.save_failed = False
        
    def load_status(self):
        """Load job status from file, or create new if doesn't exist."""
        default_status = {"submitted": {}, "pending": [], "completed": []}
        
        # Try multiple locations for the status file
        for status_path in [
            self.status_file,  # Original location
            os.path.join(os.path.expanduser("~"), os.path.basename(self.status_file)),  # Home dir
            os.path.join(tempfile.gettempdir(), os.path.basename(self.status_file))  # Temp dir
        ]:
            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as f:
                        data = json.load(f)
                        print(f"Loaded status from {status_path}")
                        # Update our status file location if we loaded from an alternative
                        if status_path != self.status_file:
                            self.status_file = status_path
                        return data
                except (json.JSONDecodeError, IOError, OSError) as e:
                    print(f"Error reading status file {status_path}: {e}")
        
        print(f"Creating new status file")
        return default_status
    
    def save_status(self):
        """Save job status to file with robust error handling."""
        # If previous saves failed, try alternate locations
        if self.save_failed:
            alt_locations = [
                os.path.join(os.path.expanduser("~"), os.path.basename(self.status_file)),  # Home dir
                os.path.join(tempfile.gettempdir(), os.path.basename(self.status_file))  # Temp dir
            ]
            for loc in alt_locations:
                if self._try_save_status(loc):
                    self.status_file = loc  # Update to new working location
                    self.save_failed = False
                    print(f"Successfully saved status to alternate location: {loc}")
                    return
            print("WARNING: Failed to save status to any location")
            return
            
        # Try normal save
        if not self._try_save_status(self.status_file):
            self.save_failed = True
            print(f"WARNING: Failed to save status file to {self.status_file}")
            # Immediately try alternate locations
            self.save_status()
    
    def _try_save_status(self, filepath):
        """Try to save status to a specific file with atomic write."""
        try:
            # Write to temp file first then rename for atomicity
            temp_file = f"{filepath}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.job_status, f, indent=2)
            
            # Atomic rename
            shutil.move(temp_file, filepath)
            return True
        except (IOError, OSError) as e:
            print(f"Error saving to {filepath}: {e}")
            try:
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            return False
    
    def find_d12_files(self):
        """Find all .d12 files in directory and subdirectories."""
        d12_files = []
        try:
            for root, _, files in os.walk(self.d12_dir):
                for file in files:
                    if file.endswith('.d12'):
                        d12_path = os.path.join(root, file)
                        job_name = os.path.splitext(file)[0]
                        d12_files.append((job_name, d12_path))
        except (IOError, OSError) as e:
            print(f"Error scanning for d12 files: {e}")
        return d12_files
    
    def get_current_jobs(self):
        """Get number of currently running/queued jobs."""
        try:
            result = subprocess.run(['squeue', '-u', os.environ.get('USER', os.environ.get('LOGNAME')), '-h'], 
                                  stdout=subprocess.PIPE, text=True, timeout=30)
            job_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            return job_count
        except subprocess.TimeoutExpired:
            print("Warning: squeue command timed out")
            return 0
        except Exception as e:
            print(f"Error checking job queue: {e}")
            return 0
    
    def update_job_status(self):
        """Update status of all d12 files."""
        try:
            all_d12 = self.find_d12_files()
            
            # Check for new d12 files
            for job_name, d12_path in all_d12:
                if (job_name not in self.job_status["submitted"] and 
                    job_name not in self.job_status["pending"] and
                    job_name not in self.job_status["completed"]):
                    self.job_status["pending"].append(job_name)
                    
            # Get list of all submitted job IDs
            submitted_ids = list(self.job_status["submitted"].values())
            
            if submitted_ids:
                # Check which jobs have completed
                running_jobs = []
                try:
                    result = subprocess.run(['squeue', '-u', os.environ.get('USER', os.environ.get('LOGNAME')), 
                                          '--format=%i', '-h'], 
                                          stdout=subprocess.PIPE, text=True, timeout=30)
                    running_jobs = result.stdout.strip().split('\n') if result.stdout.strip() else []
                except subprocess.TimeoutExpired:
                    print("Warning: squeue check timed out")
                except Exception as e:
                    print(f"Error checking job status: {e}")
                
                # Update completed jobs
                for job_name, job_id in list(self.job_status["submitted"].items()):
                    if job_id not in running_jobs:
                        self.job_status["submitted"].pop(job_name)
                        if job_name not in self.job_status["completed"]:
                            self.job_status["completed"].append(job_name)
            
            self.save_status()
        except Exception as e:
            print(f"Error updating job status: {e}")
    
    def submit_job(self, job_name, d12_path):
        """Submit a single job to SLURM."""
        job_dir = os.path.dirname(d12_path)
        cmd = f"{job_dir}/submitcrystal23.sh {job_name}"
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=60)
            
            # Extract job ID from sbatch output
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                self.job_status["submitted"][job_name] = job_id
                if job_name in self.job_status["pending"]:
                    self.job_status["pending"].remove(job_name)
                print(f"Submitted job {job_name} with ID {job_id}")
                return True
            else:
                print(f"Failed to submit job {job_name}: Unexpected output: {output}")
                if result.stderr:
                    print(f"stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"Timeout submitting job {job_name}")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error submitting job {job_name}: {e}")
            return False
    
    def process_queue(self, max_submit=None):
        """Process the queue, submitting jobs as space allows."""
        try:
            self.update_job_status()
            
            current_jobs = self.get_current_jobs()
            available_slots = max(0, self.max_jobs - current_jobs)
            
            if available_slots <= self.reserve_slots:
                print(f"Only {available_slots} slots available, reserving {self.reserve_slots} for critical jobs.")
                return 0
            
            available_slots -= self.reserve_slots
            
            if max_submit is not None:
                available_slots = min(available_slots, max_submit)
            
            print(f"Current jobs: {current_jobs}, Available slots: {available_slots}")
            
            # Find pending jobs with d12 files
            pending_jobs = []
            for job_name in self.job_status["pending"]:
                d12_path = None
                # Find the d12 path for this job name
                for name, path in self.find_d12_files():
                    if name == job_name:
                        d12_path = path
                        break
                
                if d12_path and os.path.exists(d12_path):
                    pending_jobs.append((job_name, d12_path))
            
            # Submit jobs up to available slots
            submitted_count = 0
            for job_name, d12_path in pending_jobs[:available_slots]:
                if self.submit_job(job_name, d12_path):
                    submitted_count += 1
                    try:
                        self.save_status()  # Save after each submission
                    except Exception as e:
                        print(f"Warning: Failed to save status: {e}")
            
            print(f"Submitted {submitted_count} new jobs")
            return submitted_count
        except Exception as e:
            print(f"Error processing queue: {e}")
            return 0
    
    def print_status(self):
        """Print the current status of all jobs."""
        try:
            self.update_job_status()
            
            print("\n===== JOB QUEUE STATUS =====")
            print(f"Pending jobs: {len(self.job_status['pending'])}")
            print(f"Running jobs: {len(self.job_status['submitted'])}")
            print(f"Completed jobs: {len(self.job_status['completed'])}")
            total = len(self.job_status['pending']) + len(self.job_status['submitted']) + len(self.job_status['completed'])
            print(f"Total jobs: {total}")
            print(f"Status file: {self.status_file}")
            print("============================\n")
        except Exception as e:
            print(f"Error printing status: {e}")

def main():
    parser = argparse.ArgumentParser(description="CRYSTAL23 SLURM Job Queue Manager")
    parser.add_argument("--d12-dir", default=".", help="Directory containing .d12 files")
    parser.add_argument("--status-file", default="crystal_job_status.json", help="File to track job status")
    parser.add_argument("--max-jobs", type=int, default=250, help="Maximum number of jobs in queue")
    parser.add_argument("--reserve", type=int, default=30, help="Reserved slots for critical jobs")
    parser.add_argument("--max-submit", type=int, help="Maximum jobs to submit in this run")
    parser.add_argument("--status", action="store_true", help="Only print status, don't submit")
    
    args = parser.parse_args()
    
    try:
        manager = JobQueueManager(
            d12_dir=args.d12_dir,
            status_file=args.status_file,
            max_jobs=args.max_jobs,
            reserve_slots=args.reserve
        )
        
        if args.status:
            manager.print_status()
        else:
            manager.process_queue(max_submit=args.max_submit)
            manager.print_status()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
