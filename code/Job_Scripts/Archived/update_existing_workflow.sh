#!/bin/bash
# Script to update existing workflow job scripts with fixed callback paths

WORKFLOW_DIR="$1"
if [ -z "$WORKFLOW_DIR" ]; then
    echo "Usage: $0 <workflow_directory>"
    echo "Example: $0 ~/test/workflow_outputs/workflow_20250621_154017"
    exit 1
fi

echo "Updating job scripts in: $WORKFLOW_DIR"

# Find all .sh files in material directories
find "$WORKFLOW_DIR" -name "*.sh" -type f | while read script_file; do
    echo "Updating: $script_file"
    
    # Create backup
    cp "$script_file" "$script_file.backup"
    
    # Update the callback section
    sed -i '
    # Remove old single-location callback
    /# ADDED: Auto-submit new jobs when this one completes/,/^fi$/{
        /# ADDED: Auto-submit new jobs when this one completes/!{
            /^fi$/!d
        }
    }
    
    # Add the new multi-location callback after the marker
    /# ADDED: Auto-submit new jobs when this one completes/a\
# Check multiple possible locations for queue managers\
if [ -f $DIR/enhanced_queue_manager.py ]; then\
    cd $DIR\
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3\
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then\
    cd $DIR/../../../../\
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3\
elif [ -f $DIR/crystal_queue_manager.py ]; then\
    cd $DIR\
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5\
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then\
    cd $DIR/../../../../\
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5\
fi
    ' "$script_file"
    
    echo "Updated: $script_file (backup saved as $script_file.backup)"
done

echo "All job scripts updated!"
echo "Backups saved with .backup extension"