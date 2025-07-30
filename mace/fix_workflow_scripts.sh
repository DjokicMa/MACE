#!/bin/bash
# Script to fix MACE_CONTEXT_DIR in existing SLURM scripts

# Usage: ./fix_workflow_scripts.sh /path/to/test6

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/test6"
    exit 1
fi

TEST_DIR="$1"
WORKFLOW_DIR=$(find "$TEST_DIR/workflow_outputs" -name "workflow_*" -type d | head -1)

if [ -z "$WORKFLOW_DIR" ]; then
    echo "No workflow directory found in $TEST_DIR/workflow_outputs"
    exit 1
fi

WORKFLOW_ID=$(basename "$WORKFLOW_DIR")
CORRECT_CONTEXT_DIR="$TEST_DIR/.mace_context_$WORKFLOW_ID"

echo "Fixing SLURM scripts for workflow: $WORKFLOW_ID"
echo "Correct context directory: $CORRECT_CONTEXT_DIR"
echo

# Find all .sh files in the workflow outputs
find "$WORKFLOW_DIR" -name "*.sh" -type f | while read script; do
    echo "Fixing: $script"
    
    # Remove the incorrect MACE_CONTEXT_DIR line (the one with SLURM_SUBMIT_DIR)
    sed -i '/export MACE_CONTEXT_DIR="${SLURM_SUBMIT_DIR}\/.mace_context_/d' "$script"
    
    # Update the correct MACE_CONTEXT_DIR to use absolute path if it exists
    # This handles the second occurrence which should have the correct path
    sed -i "s|export MACE_CONTEXT_DIR=\".*\.mace_context_${WORKFLOW_ID}\"|export MACE_CONTEXT_DIR=\"${CORRECT_CONTEXT_DIR}\"|g" "$script"
    
    # If no MACE_CONTEXT_DIR remains, add it after MACE_WORKFLOW_ID
    if ! grep -q "export MACE_CONTEXT_DIR=" "$script"; then
        sed -i "/export MACE_WORKFLOW_ID=\"$WORKFLOW_ID\"/a export MACE_CONTEXT_DIR=\"$CORRECT_CONTEXT_DIR\"" "$script"
    fi
done

echo
echo "✅ Script fixes complete!"
echo
echo "Now you need to:"
echo "1. Populate the shared database with completed calculations"
echo "2. Trigger workflow progression"