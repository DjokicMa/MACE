#!/bin/bash

###############################################################################
# Generate High-Quality Iso-Browse GIFs for all CUBE files
#
# This script creates animations that smoothly transition through different
# isovalue levels, showing how the isosurface shape changes with threshold.
# Excellent for understanding electron density distribution gradients.
#
# Output: Creates *_iso_browse.gif for each CUBE file in the iso_browse_gifs/ folder
###############################################################################

# Configuration for high-quality output
ISO_COUNT=30            # Number of different isovalues to animate through
FPS=8                   # 8 fps = smooth but not too fast
WIDTH=1200              # High resolution
HEIGHT=900              # 4:3 aspect ratio (includes slider UI)

# Output directory
OUTPUT_DIR="iso_browse_gifs"
mkdir -p "$OUTPUT_DIR"

# Python interpreter
PYTHON="/home/marcus/anaconda3/bin/python"

# Visualization script
VIZ_SCRIPT="/mnt/iscsi/UsefulScripts/Codebase/reorganization/cif/cube/crystal_cubeviz_plotly.py"

echo "========================================================================"
echo "  Generating High-Quality Iso-Browse GIFs"
echo "========================================================================"
echo "  Settings:"
echo "    - Isovalues: $ISO_COUNT levels (logarithmic spacing)"
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
    output_gif="$OUTPUT_DIR/${base_name}_iso_browse.gif"

    echo "[$count] Processing: $cube_file"
    echo "    Output: $output_gif"

    # Determine visualization type based on filename
    if [[ "$cube_file" == *"DENS"* ]] || [[ "$cube_file" == *"density"* ]]; then
        # Density files - use default isovalues (good range for electron density)
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --iso-browse \
            --iso-count $ISO_COUNT \
            --save-gif "$output_gif" \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --z-shift 20

    elif [[ "$cube_file" == *"POT"* ]] || [[ "$cube_file" == *"potential"* ]]; then
        # Potential files - show positive surface evolution
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --iso-browse \
            --iso-count $ISO_COUNT \
            --save-gif "$output_gif" \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --z-shift 20

    elif [[ "$cube_file" == *"SPIN"* ]] || [[ "$cube_file" == *"spin"* ]]; then
        # Spin files - show positive surface evolution
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --iso-browse \
            --iso-count $ISO_COUNT \
            --save-gif "$output_gif" \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --z-shift 20

    elif [[ "$cube_file" == *"SUBTRACTED"* ]] || [[ "$cube_file" == *"difference"* ]]; then
        # Difference files - show both positive and negative evolution with symmetric values
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --iso-browse \
            --iso-count $ISO_COUNT \
            --show-both \
            --save-gif "$output_gif" \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
            --z-shift 20
    else
        # Default: regular iso-browse
        $PYTHON "$VIZ_SCRIPT" "$cube_file" \
            --iso-browse \
            --iso-count $ISO_COUNT \
            --save-gif "$output_gif" \
            --gif-fps $FPS \
            --gif-width $WIDTH \
            --gif-height $HEIGHT \
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
echo "  Completed! Generated $count iso-browse GIFs in $OUTPUT_DIR/"
echo "========================================================================"
