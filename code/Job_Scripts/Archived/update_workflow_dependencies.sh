#!/bin/bash
# Update Workflow Dependencies
# ============================
# Quick script to update all missing dependencies in a workflow directory

WORKFLOW_DIR="$1"
if [ -z "$WORKFLOW_DIR" ]; then
    WORKFLOW_DIR="."
fi

cd "$WORKFLOW_DIR"
echo "Updating workflow dependencies in: $(pwd)"

# Get the script directory (Job_Scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Copying missing dependencies..."

# Core dependencies that are often missing
CORE_FILES=(
    "error_detector.py"
    "material_database.py"
    "enhanced_queue_manager.py"
    "crystal_property_extractor.py"
    "formula_extractor.py"
    "input_settings_extractor.py"
    "query_input_settings.py"
    "error_recovery.py"
    "recovery_config.yaml"
)

# Copy missing files
for file in "${CORE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        if [ -f "$SCRIPT_DIR/$file" ]; then
            cp "$SCRIPT_DIR/$file" .
            echo "  ✓ Copied: $file"
        else
            echo "  ✗ Missing source: $file"
        fi
    else
        # Update existing file if source is newer
        if [ "$SCRIPT_DIR/$file" -nt "$file" ]; then
            cp "$SCRIPT_DIR/$file" .
            echo "  ↻ Updated: $file"
        fi
    fi
done

echo "Dependencies updated!"
echo ""
echo "Test the enhanced queue manager:"
echo "  python enhanced_queue_manager.py --callback-mode completion"