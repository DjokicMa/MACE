#!/bin/bash

###############################################################################
# Simple GIF Generator - No vacuum cropping, pure visualization
#
# This version does NOT use --z-shift to avoid any potential alignment issues
# between atoms and isosurfaces.
#
# Usage:
#   ./generate_gifs_simple.sh file.CUBE          # Generate all 3 types
#   ./generate_gifs_simple.sh *DENS*.CUBE        # Batch process
###############################################################################

PYTHON="/home/marcus/anaconda3/bin/python"
VIZ_SCRIPT="/mnt/iscsi/UsefulScripts/Codebase/reorganization/cif/cube/crystal_cubeviz_plotly.py"

# Check if files were provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 file1.CUBE [file2.CUBE ...]"
    echo "Example: $0 *DENS*.CUBE"
    exit 1
fi

echo "========================================================================"
echo "  Simple GIF Generator (No Z-Shift)"
echo "========================================================================"
echo ""

for cube_file in "$@"; do
    [ -e "$cube_file" ] || continue

    base_name="${cube_file%.CUBE}"

    echo "Processing: $cube_file"
    echo "----------------------------------------"

    # Determine if we need --show-both
    SHOW_BOTH=""
    if [[ "$cube_file" == *"POT"* ]] || [[ "$cube_file" == *"SPIN"* ]] || [[ "$cube_file" == *"SUBTRACTED"* ]] || [[ "$cube_file" == *"difference"* ]]; then
        SHOW_BOTH="--show-both"
    fi

    # 1. Rotation GIF
    echo "[1/3] Rotation GIF..."
    mkdir -p rotation_gifs_simple
    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --save-gif "rotation_gifs_simple/${base_name}_rotation.gif" \
        --rotation-frames 36 \
        --gif-fps 10 \
        --gif-width 1000 \
        --gif-height 1000 \
        $SHOW_BOTH \
        2>&1 | grep -E "Saved:|ERROR"

    # 2. Iso-Browse GIF
    echo "[2/3] Iso-Browse GIF..."
    mkdir -p iso_browse_gifs_simple
    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --iso-browse \
        --iso-count 20 \
        --save-gif "iso_browse_gifs_simple/${base_name}_iso_browse.gif" \
        --gif-fps 8 \
        --gif-width 1000 \
        --gif-height 800 \
        $SHOW_BOTH \
        2>&1 | grep -E "Saved:|ERROR"

    # 3. Slice-Browse GIF
    echo "[3/3] Slice-Browse GIF..."
    mkdir -p slice_browse_gifs_simple
    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --slice-browse z \
        --save-gif "slice_browse_gifs_simple/${base_name}_slice_browse.gif" \
        --gif-fps 10 \
        --gif-width 1000 \
        --gif-height 800 \
        2>&1 | grep -E "Saved:|ERROR"

    echo ""
done

echo "========================================================================"
echo "  Complete!"
echo "========================================================================"
echo "Check: rotation_gifs_simple/, iso_browse_gifs_simple/, slice_browse_gifs_simple/"
