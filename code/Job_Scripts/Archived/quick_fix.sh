#!/bin/bash
# Quick Fix Deployment Script
# Copies the fixed callback mechanism files to your test directory

# Get the target directory (first argument or current directory)
TARGET_DIR=${1:-.}

echo "Deploying callback fixes to: $TARGET_DIR"

# Copy the fixed files
cp enhanced_queue_manager.py "$TARGET_DIR/"
cp material_database.py "$TARGET_DIR/"
cp workflow_engine.py "$TARGET_DIR/"
cp workflow_executor.py "$TARGET_DIR/"
cp populate_completed_jobs.py "$TARGET_DIR/"
cp test_callback_fix.py "$TARGET_DIR/"

echo "Files deployed. Testing..."

# Test the callback mechanism
cd "$TARGET_DIR"
echo "Testing enhanced queue manager..."
python enhanced_queue_manager.py --callback-mode completion

echo ""
echo "Deployment complete!"
echo ""
echo "The callback mechanism should now:"
echo "1. Detect completed workflow jobs in workflow_outputs/"
echo "2. Add them to the materials database"
echo "3. Use workflow_engine.py to generate next step (OPT → SP)"
echo "4. Automatically submit SP calculations"
echo ""
echo "Instead of creating the 'opt/' folder with duplicate calculations."