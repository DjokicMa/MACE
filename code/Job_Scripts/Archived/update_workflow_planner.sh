#!/bin/bash
# Update workflow planner with fixed default accounts

WORKFLOW_DIR="$1"
if [ -z "$WORKFLOW_DIR" ]; then
    WORKFLOW_DIR="."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Updating workflow planner in: $WORKFLOW_DIR"
echo "Source: $SCRIPT_DIR/workflow_planner.py"

# Copy the updated workflow planner
if [ -f "$SCRIPT_DIR/workflow_planner.py" ]; then
    cp "$SCRIPT_DIR/workflow_planner.py" "$WORKFLOW_DIR/"
    echo "✓ Updated workflow_planner.py"
    echo ""
    echo "Fixed defaults:"
    echo "  ✓ BAND/DOSS account: general → mendoza_q"
    echo "  ✓ Removed intel18 constraint"
    echo "  ✓ All calculation types now default to mendoza_q"
else
    echo "✗ Source file not found: $SCRIPT_DIR/workflow_planner.py"
fi