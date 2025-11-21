#!/bin/bash

###############################################################################
# Master Script: Generate ALL High-Quality GIFs for CUBE files
#
# This script generates all three types of animations:
#   1. Rotation GIFs     - 360° rotating isosurface views
#   2. Iso-Browse GIFs   - Animations through different isovalue levels
#   3. Slice-Browse GIFs - Animations through Z-axis slices
#
# Usage:
#   ./generate_all_gifs.sh              # Process ALL .CUBE files (careful!)
#   ./generate_all_gifs.sh file1.CUBE   # Process specific file
#   ./generate_all_gifs.sh *DENS*.CUBE  # Process files matching pattern
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================================================"
echo "  MASTER GIF GENERATOR - All Visualization Modes"
echo "========================================================================"
echo ""

# Check if specific files were provided
if [ $# -eq 0 ]; then
    echo "WARNING: No files specified. This will process ALL .CUBE files!"
    echo ""
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    # Process all CUBE files
    files=(*.CUBE)
else
    # Process specified files
    files=("$@")
fi

echo "Will process ${#files[@]} file(s)"
echo ""

# Temporary modify the generation scripts to only process specified files
# by creating a temporary marker file

for cube_file in "${files[@]}"; do
    [ -e "$cube_file" ] || continue

    base_name="${cube_file%.CUBE}"

    echo "========================================================================"
    echo "Processing: $cube_file"
    echo "========================================================================"

    # 1. Generate Rotation GIF
    echo ""
    echo "[1/3] Generating 360° Rotation GIF..."
    output_dir="rotation_gifs"
    mkdir -p "$output_dir"
    output_gif="$output_dir/${base_name}_rotation.gif"

    PYTHON="/home/marcus/anaconda3/bin/python"
    VIZ_SCRIPT="$SCRIPT_DIR/crystal_cubeviz_plotly.py"

    # Determine file type and set appropriate flags
    EXTRA_FLAGS=""
    if [[ "$cube_file" == *"POT"* ]] || [[ "$cube_file" == *"SPIN"* ]] || [[ "$cube_file" == *"SUBTRACTED"* ]]; then
        EXTRA_FLAGS="--show-both"
    fi

    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --save-gif "$output_gif" \
        --rotation-frames 72 \
        --gif-fps 15 \
        --gif-width 1200 \
        --gif-height 1200 \
        --rotate-axis z \
        --z-shift 20 \
        $EXTRA_FLAGS

    [ $? -eq 0 ] && echo "  ✓ Rotation GIF: $output_gif" || echo "  ✗ Failed"

    # 2. Generate Iso-Browse GIF
    echo ""
    echo "[2/3] Generating Iso-Browse GIF..."
    output_dir="iso_browse_gifs"
    mkdir -p "$output_dir"
    output_gif="$output_dir/${base_name}_iso_browse.gif"

    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --iso-browse \
        --iso-count 30 \
        --save-gif "$output_gif" \
        --gif-fps 8 \
        --gif-width 1200 \
        --gif-height 900 \
        --z-shift 20 \
        $EXTRA_FLAGS

    [ $? -eq 0 ] && echo "  ✓ Iso-Browse GIF: $output_gif" || echo "  ✗ Failed"

    # 3. Generate Slice-Browse GIF
    echo ""
    echo "[3/3] Generating Slice-Browse GIF..."
    output_dir="slice_browse_gifs"
    mkdir -p "$output_dir"
    output_gif="$output_dir/${base_name}_slice_browse.gif"

    $PYTHON "$VIZ_SCRIPT" "$cube_file" \
        --slice-browse z \
        --save-gif "$output_gif" \
        --gif-fps 10 \
        --gif-width 1200 \
        --gif-height 900 \
        --z-shift 20

    [ $? -eq 0 ] && echo "  ✓ Slice-Browse GIF: $output_gif" || echo "  ✗ Failed"

    echo ""
done

echo "========================================================================"
echo "  ALL GIFs GENERATED!"
echo "========================================================================"
echo ""
echo "Output directories:"
echo "  - rotation_gifs/      (360° rotation animations)"
echo "  - iso_browse_gifs/    (isovalue slider animations)"
echo "  - slice_browse_gifs/  (Z-slice animations)"
echo ""
