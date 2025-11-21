#!/bin/bash

###############################################################################
# Simple GIF Generator - Uses default visualization (no modifications)
#
# This just adds --save-gif to the command that already works perfectly
#
# Usage:
#   ./make_gifs.sh file.CUBE                # Generate all 3 types for one file
#   ./make_gifs.sh *DENS*.CUBE              # Process all density files
#   ./make_gifs.sh *.CUBE                   # Process all CUBE files
###############################################################################

PYTHON="/home/marcus/anaconda3/bin/python"
VIZ_SCRIPT="./crystal_cubeviz_plotly.py"

# Settings for nice-looking GIFs
ROTATION_FRAMES=60      # 60 frames = 6° per frame (smooth)
ROTATION_FPS=12         # 12 fps (smooth playback, ~5 second loop)
ISO_COUNT=25            # 25 isovalue levels
ISO_FPS=8               # 8 fps (readable)
SLICE_FPS=10            # 10 fps (smooth scrolling)
WIDTH=1200              # High quality
HEIGHT=1200             # Square for rotation
HEIGHT_BROWSER=900      # 4:3 for browser modes

# Output directories
mkdir -p gif_rotation gif_iso_browse gif_slice_browse

# Check if files provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 file1.CUBE [file2.CUBE ...]"
    echo ""
    echo "Examples:"
    echo "  $0 my_file.CUBE"
    echo "  $0 *DENS*.CUBE"
    echo "  $0 *.CUBE"
    exit 1
fi

echo "========================================================================"
echo "  Generating High-Quality GIFs"
echo "========================================================================"
echo "  Settings:"
echo "    Rotation: ${ROTATION_FRAMES} frames, ${ROTATION_FPS} fps, ${WIDTH}x${HEIGHT} px"
echo "    Iso-browse: ${ISO_COUNT} levels, ${ISO_FPS} fps, ${WIDTH}x${HEIGHT_BROWSER} px"
echo "    Slice-browse: ${SLICE_FPS} fps, ${WIDTH}x${HEIGHT_BROWSER} px"
echo "========================================================================"
echo ""

count=0
for cube_file in "$@"; do
    [ -e "$cube_file" ] || continue

    count=$((count + 1))
    base_name="${cube_file%.CUBE}"

    echo "[$count] Processing: $cube_file"

    # Auto-detect data type (script handles POT/SPIN/SUBTRACTED automatically)
    if [[ "$cube_file" == *"POT"* ]] || [[ "$cube_file" == *"SPIN"* ]] || \
       [[ "$cube_file" == *"SUBTRACTED"* ]] || [[ "$cube_file" == *"difference"* ]]; then
        echo "    (Auto-detected: potential/spin/difference data)"
    fi

    # 1. Rotation GIF
    echo "    [1/3] Rotation GIF..."
    $PYTHON $VIZ_SCRIPT "$cube_file" \
        --save-gif "gif_rotation/${base_name}_rotation.gif" \
        --rotation-frames $ROTATION_FRAMES \
        --gif-fps $ROTATION_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "gif_rotation/${base_name}_rotation.gif" | cut -f1)
        echo "          ✓ Saved: gif_rotation/${base_name}_rotation.gif ($size)"
    else
        echo "          ✗ Failed"
    fi

    # 2. Iso-Browse GIF
    echo "    [2/3] Iso-Browse GIF..."
    $PYTHON $VIZ_SCRIPT "$cube_file" \
        --iso-browse \
        --iso-count $ISO_COUNT \
        --save-gif "gif_iso_browse/${base_name}_iso_browse.gif" \
        --gif-fps $ISO_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT_BROWSER \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "gif_iso_browse/${base_name}_iso_browse.gif" | cut -f1)
        echo "          ✓ Saved: gif_iso_browse/${base_name}_iso_browse.gif ($size)"
    else
        echo "          ✗ Failed"
    fi

    # 3. Slice-Browse GIF
    echo "    [3/3] Slice-Browse GIF..."
    $PYTHON $VIZ_SCRIPT "$cube_file" \
        --slice-browse z \
        --save-gif "gif_slice_browse/${base_name}_slice_browse.gif" \
        --gif-fps $SLICE_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT_BROWSER \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "gif_slice_browse/${base_name}_slice_browse.gif" | cut -f1)
        echo "          ✓ Saved: gif_slice_browse/${base_name}_slice_browse.gif ($size)"
    else
        echo "          ✗ Failed"
    fi

    echo ""
done

echo "========================================================================"
echo "  Complete! Processed $count file(s)"
echo "========================================================================"
echo "  Output directories:"
echo "    gif_rotation/      - 360° rotation animations"
echo "    gif_iso_browse/    - Isovalue slider animations"
echo "    gif_slice_browse/  - Z-slice animations"
echo "========================================================================"
