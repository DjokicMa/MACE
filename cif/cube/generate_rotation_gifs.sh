#!/bin/bash

###############################################################################
# Generate High-Quality 360° Rotation GIFs for all CUBE files
#
# This script creates smooth rotating isosurface visualizations showing the
# 3D structure from all angles. Perfect for presentations and publications.
#
# Output: Creates *_rotation.gif for each CUBE file in the rotation_gifs/ folder
###############################################################################

# Configuration for high-quality output
ROTATION_FRAMES=72      # 72 frames = 5° per frame (smooth rotation)
FPS=15                  # 15 fps = smooth playback
WIDTH=1200              # High resolution
HEIGHT=1200             # Square aspect ratio for rotation
ROTATE_AXIS="z"         # Rotate around Z-axis (vertical)

# Output directory
OUTPUT_DIR="rotation_gifs"
mkdir -p "$OUTPUT_DIR"

# Python interpreter
PYTHON="/home/marcus/anaconda3/bin/python"

# Visualization script
VIZ_SCRIPT="/mnt/iscsi/UsefulScripts/Codebase/reorganization/cif/cube/crystal_cubeviz_plotly.py"

echo "========================================================================"
echo "  Generating High-Quality Rotation GIFs"
echo "========================================================================"
echo "  Settings:"
echo "    - Frames: $ROTATION_FRAMES (360° / $ROTATION_FRAMES = $((360 / ROTATION_FRAMES))° per frame)"
echo "    - FPS: $FPS"
echo "    - Resolution: ${WIDTH}x${HEIGHT}"
echo "    - Rotation axis: $ROTATE_AXIS"
echo "    - Output directory: $OUTPUT_DIR/"
echo "========================================================================"
echo ""

# Process all CUBE files
count=0
for cube_file in *.CUBE; do
    [ -e "$cube_file" ] || continue

    count=$((count + 1))
    base_name="${cube_file%.CUBE}"
    output_gif="$OUTPUT_DIR/${base_name}_rotation.gif"

    echo "[$count] Processing: $cube_file"
    echo "    Output: $output_gif"

    # Determine visualization type based on filename
    if [[ "$cube_file" == *"DENS"* ]] || [[ "$cube_file" == *"density"* ]]; then
        # Density files - use default isosurface
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --save-gif "$output_gif" \
            --rotation-frames $ROTATION_FRAMES \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --rotate-axis $ROTATE_AXIS \
            --z-shift 20

    elif [[ "$cube_file" == *"POT"* ]] || [[ "$cube_file" == *"potential"* ]]; then
        # Potential files - show both positive and negative
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --save-gif "$output_gif" \
            --rotation-frames $ROTATION_FRAMES \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --rotate-axis $ROTATE_AXIS \
            --show-both \
            --z-shift 20

    elif [[ "$cube_file" == *"SPIN"* ]] || [[ "$cube_file" == *"spin"* ]]; then
        # Spin files - show both alpha and beta
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --save-gif "$output_gif" \
            --rotation-frames $ROTATION_FRAMES \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --rotate-axis $ROTATE_AXIS \
            --show-both \
            --z-shift 20

    elif [[ "$cube_file" == *"SUBTRACTED"* ]] || [[ "$cube_file" == *"difference"* ]]; then
        # Difference files - show both positive and negative with symmetric values
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --save-gif "$output_gif" \
            --rotation-frames $ROTATION_FRAMES \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --rotate-axis $ROTATE_AXIS \
            --show-both \
            --z-shift 20
    else
        # Default: regular isosurface
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --save-gif "$output_gif" \
            --rotation-frames $ROTATION_FRAMES \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --rotate-axis $ROTATE_AXIS \
            --z-shift 20
    fi

    if [ $? -eq 0 ]; then
        echo "    ✓ Success!"
    else
        echo "    ✗ Failed"
    fi
    echo ""
done

echo "========================================================================"
echo "  Completed! Generated $count rotation GIFs in $OUTPUT_DIR/"
echo "========================================================================"
