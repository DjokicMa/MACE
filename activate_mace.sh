#!/bin/bash
# MACE (Mendoza Automated CRYSTAL Engine) Activation Script
# Source this file to temporarily set up MACE environment
#
# Developed by: Marcus Djokic (Primary Developer)
# Contributors: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey
# Mendoza Group, Michigan State University

export MACE_HOME="/mnt/iscsi/UsefulScripts/Codebase/reorganization"

# Add MACE scripts to PATH - using reorganized structure
export PATH="$MACE_HOME:$PATH"  # For mace.py and mace_cli
export PATH="$MACE_HOME/mace/submission:$PATH"
export PATH="$MACE_HOME/mace/queue:$PATH"
export PATH="$MACE_HOME/mace/database:$PATH"
export PATH="$MACE_HOME/mace/recovery:$PATH"
export PATH="$MACE_HOME/mace/utils:$PATH"
export PATH="$MACE_HOME/Crystal_d12:$PATH"
export PATH="$MACE_HOME/Crystal_d3:$PATH"

# Legacy paths for scripts not yet migrated
export PATH="$MACE_HOME/code/Check_Scripts:$PATH"
export PATH="$MACE_HOME/code/Plotting_Scripts:$PATH"
export PATH="$MACE_HOME/code/Post_Processing_Scripts:$PATH"
export PATH="$MACE_HOME/code/Band_Alignment:$PATH"

# Add convenient alias
alias mace="mace_cli"

echo "MACE (Mendoza Automated CRYSTAL Engine) environment activated!"
echo "  MACE_HOME: $MACE_HOME"
echo "  Added to PATH:"
echo "    - mace/ (MACE core modules)"
echo "    - Crystal_d12 (D12 creation tools)"
echo "    - Crystal_d3 (D3 creation tools)"
echo "    - Legacy scripts (Check, Plotting, Post-Processing)"
