#!/bin/bash

# Deploy Fixed Files Script
# Copies the corrected callback mechanism files

TARGET_DIR=${1:-/mnt/ffs24/home/djokicma/test}

echo "Deploying fixed callback mechanism files to: $TARGET_DIR"

# Copy the fixed files
cp enhanced_queue_manager.py "$TARGET_DIR/"
cp populate_completed_jobs.py "$TARGET_DIR/"
cp material_database.py "$TARGET_DIR/"
cp workflow_engine.py "$TARGET_DIR/"

echo "Files deployed successfully!"
echo ""
echo "Fixed issues:"
echo "✓ Changed add_material → create_material"
echo "✓ Changed add_calculation → create_calculation"  
echo "✓ Fixed structure_file → source_file parameter"
echo "✓ Added missing db_path attribute"
echo "✓ Enhanced callback to use workflow progression"
echo ""
echo "Now test with:"
echo "cd $TARGET_DIR"
echo "python enhanced_queue_manager.py --callback-mode completion"