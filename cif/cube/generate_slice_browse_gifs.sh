#!/bin/bash

###############################################################################
# Generate High-Quality Slice-Browse GIFs for all CUBE files
#
# This script creates animations that slice through the Z-axis, showing
# 2D cross-sections of the data at different heights. Perfect for understanding
# layer-by-layer structure of 2D materials and molecular systems.
#
# Output: Creates *_slice_browse.gif for each CUBE file in the slice_browse_gifs/ folder
###############################################################################

# Configuration for high-quality output
FPS=10                  # 10 fps = smooth scrolling through slices
WIDTH=1200              # High resolution
HEIGHT=900              # 4:3 aspect ratio (includes slider UI)

# Output directory
OUTPUT_DIR="slice_browse_gifs"
mkdir -p "$OUTPUT_DIR"

# Python interpreter
PYTHON="/home/marcus/anaconda3/bin/python"

# Visualization script
VIZ_SCRIPT="/mnt/iscsi/UsefulScripts/Codebase/reorganization/cif/cube/crystal_cubeviz_plotly.py"

echo "========================================================================"
echo "  Generating High-Quality Slice-Browse GIFs"
echo "========================================================================"
echo "  Settings:"
echo "    - FPS: $FPS"
echo "    - Resolution: ${WIDTH}x${HEIGHT}"
echo "    - Output directory: $OUTPUT_DIR/"
echo "========================================================================"
echo ""

# Process all CUBE files
count=0
for cube_file in *.CUBE; do
    [ -e "$cube_file" ] || continue

    count=$((count + 1))
    base_name="${cube_file%.CUBE}"
    output_gif="$OUTPUT_DIR/${base_name}_slice_browse.gif"

    echo "[$count] Processing: $cube_file"
    echo "    Output: $output_gif"

    # All file types use the same slice-browse command
    # The colormap is automatically selected based on data type
    # Use Z-axis for 2D materials (most informative for layer structures)
    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --slice-browse z \
        --save-gif "$output_gif" \
        --gif-fps $FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT \
        --z-shift 20

    if [ $? -eq 0 ]; then
        echo "    ✓ Success!"
    else
        echo "    ✗ Failed"
    fi
    echo ""
done

echo "========================================================================"
echo "  Completed! Generated $count slice-browse GIFs in $OUTPUT_DIR/"
echo "========================================================================"
