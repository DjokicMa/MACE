#!/bin/bash

###############################################################################
# Electrostatic Potential (ESP) GIF Generator
#
# Creates dual-CUBE visualizations where:
#   - Shape = electron density isosurface (DENS.CUBE)
#   - Color = electrostatic potential (POT.CUBE)
#
# Automatically matches DENS and POT files from the same calculation.
#
# Usage:
#   ./make_esp_gifs.sh                      # Process all matched pairs
#   ./make_esp_gifs.sh base_name            # Process specific calculation
###############################################################################

PYTHON="/home/marcus/anaconda3/bin/python"
VIZ_SCRIPT="./crystal_cubeviz_plotly.py"

# GIF settings
ROTATION_FRAMES=60
ROTATION_FPS=12
WIDTH=1200
HEIGHT=1200
HEIGHT_BROWSER=900

# Output directory
OUTPUT_DIR="gif_esp"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "  Electrostatic Potential (ESP) GIF Generator"
echo "========================================================================"
echo "  Visualization: Density isosurface colored by potential"
echo "  Settings: ${ROTATION_FRAMES} frames, ${ROTATION_FPS} fps, ${WIDTH}x${HEIGHT} px"
echo "========================================================================"
echo ""

# Function to get base name (remove _DENS.CUBE or _POT.CUBE suffix)
get_base_name() {
    local file="$1"
    echo "$file" | sed -E 's/_(DENS|POT|SPIN)\.CUBE$//'
}

# Find all DENS files
declare -A processed
count=0

for dens_file in *_DENS.CUBE; do
    [ -e "$dens_file" ] || continue

    base_name=$(get_base_name "$dens_file")

    # Skip if already processed
    [ -n "${processed[$base_name]}" ] && continue
    processed[$base_name]=1

    # Look for matching POT file
    pot_file="${base_name}_POT.CUBE"

    if [ ! -e "$pot_file" ]; then
        echo "⚠ Skipping: $dens_file (no matching POT file)"
        continue
    fi

    count=$((count + 1))

    echo "[$count] Processing ESP for: $base_name"
    echo "    DENS: $dens_file"
    echo "    POT:  $pot_file"

    # Determine appropriate isovalue for density shape
    # Use 0.01 as a good default for electron density
    ISO_VALUE=0.01

    # Create short name for output
    output_base=$(echo "$base_name" | sed -E 's/charge\+potential/ESP/' | sed -E 's/_opt_HSESOL3C_optimized//')

    # 1. Rotation GIF
    echo "    [1/3] Rotation GIF..."
    $PYTHON $VIZ_SCRIPT \
        --shape-cube "$dens_file" \
        --color-cube "$pot_file" \
        --shape-iso $ISO_VALUE \
        --save-gif "${OUTPUT_DIR}/${output_base}_ESP_rotation.gif" \
        --rotation-frames $ROTATION_FRAMES \
        --gif-fps $ROTATION_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "${OUTPUT_DIR}/${output_base}_ESP_rotation.gif" | cut -f1)
        echo "          ✓ Saved: ${OUTPUT_DIR}/${output_base}_ESP_rotation.gif ($size)"
    else
        echo "          ✗ Failed"
    fi

    # Note: iso-browse and slice-browse don't work well with dual-cube mode
    # They're designed for single CUBE files
    # So we only do rotation GIFs for ESP

    echo ""
done

if [ $count -eq 0 ]; then
    echo "No matching DENS/POT pairs found."
    echo ""
    echo "Expected file pattern:"
    echo "  basename_DENS.CUBE"
    echo "  basename_POT.CUBE"
else
    echo "========================================================================"
    echo "  Complete! Processed $count ESP visualization(s)"
    echo "========================================================================"
    echo "  Output: ${OUTPUT_DIR}/"
    echo ""
    echo "Note: Only rotation GIFs are created for ESP plots."
    echo "      Iso-browse and slice-browse require single CUBE files."
    echo "========================================================================"
fi
