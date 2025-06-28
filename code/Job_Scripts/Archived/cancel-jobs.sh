#!/bin/sh
"""
This script cancels all SLURM jobs for the current user that have job IDs greater than a specified minimum job number.
Usage: ./script.sh <minimum_job_number>
"""

if [ -z "$1" ] ; then
    echo "Minimum Job Number argument is required.  Run as '$0 jobnum'"
    exit 1
fi
minjobnum="$1"
myself="$(id -u -n)"
for j in $(squeue --user="$myself" --noheader --format='%i') ; do
  if [ "$j" -gt "$minjobnum" ] ; then
    scancel "$j"
  fi
done
