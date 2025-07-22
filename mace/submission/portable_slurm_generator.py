#!/usr/bin/env python3
"""
Example of generating portable SLURM scripts that work for any user.
"""

import os
from pathlib import Path

def get_mace_home():
    """Find MACE_HOME using various methods."""
    
    # Method 1: Check environment variable
    if 'MACE_HOME' in os.environ:
        return os.environ['MACE_HOME']
    
    # Method 2: Check relative to this script
    current_file = Path(__file__).resolve()
    
    # If we're in code/Job_Scripts/
    if current_file.parent.name == 'Job_Scripts' and current_file.parent.parent.name == 'code':
        return str(current_file.parent.parent.parent)
    
    # Method 3: Search upward for marker files
    search_dir = current_file.parent
    for _ in range(5):  # Check up to 5 levels up
        if (search_dir / 'code' / 'Job_Scripts' / 'enhanced_queue_manager.py').exists():
            return str(search_dir)
        search_dir = search_dir.parent
    
    # Method 4: Ask user to set it
    return None

def generate_slurm_script(job_name, calc_type='OPT'):
    """Generate a portable SLURM script."""
    
    mace_home = get_mace_home()
    
    if not mace_home:
        print("ERROR: Could not find MACE_HOME")
        print("Please either:")
        print("  1. Set environment variable: export MACE_HOME=/path/to/mace")
        print("  2. Run this script from within the mace repository")
        return None
    
    script_content = f'''#!/bin/bash --login
#SBATCH -J {job_name}
#SBATCH -o {job_name}-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=5G

export JOB={job_name}
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

# MACE path (detected when script was generated)
export MACE_HOME={mace_home}

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "MACE: $MACE_HOME"

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0

mkdir -p $scratch/$JOB
cp $DIR/$JOB.d12 $scratch/$JOB/INPUT
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal 2>&1 >& $DIR/${{JOB}}.out
cp fort.9 ${{DIR}}/${{JOB}}.f9

# Callback using MACE
if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
    cd $DIR
    python $MACE_HOME/mace/queue/manager.py \\
        --db-path ./materials.db \\
        --work-dir . \\
        --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
else
    echo "Warning: Queue manager not found at $MACE_HOME/mace/queue/manager.py"
fi
'''
    
    return script_content

def main():
    """Example usage."""
    mace_home = get_mace_home()
    
    if mace_home:
        print(f"Found MACE_HOME: {mace_home}")
        
        # Generate example SLURM script
        script = generate_slurm_script("test_job")
        if script:
            with open("test_job.sh", "w") as f:
                f.write(script)
            print("Generated test_job.sh")
    else:
        print("Could not detect MACE_HOME")

if __name__ == "__main__":
    main()