#!/bin/bash

###############################################################################
# Density Difference GIF Generator
#
# Automatically finds matching triplets:
#   - *_opt_*_DENS.CUBE (hybrid system)
#   - *_FSI_opt_*_DENS.CUBE (isolated molecule)
#   - *_graphene_opt_*_DENS.CUBE (bare surface)
#
# Performs: Δρ = ρ(hybrid) - ρ(surface) - ρ(molecule)
#
# Then generates all 3 GIF types for the difference.
#
# Usage:
#   ./make_difference_gifs.sh              # Process all matched triplets
###############################################################################

PYTHON="/home/marcus/anaconda3/bin/python"
VIZ_SCRIPT="./crystal_cubeviz_plotly.py"

# GIF settings
ROTATION_FRAMES=60
ROTATION_FPS=12
ISO_COUNT=25
ISO_FPS=8
SLICE_FPS=10
WIDTH=1200
HEIGHT=1200
HEIGHT_BROWSER=900

# Output directories
DIFF_DIR="difference_cubes"
GIF_DIR="gif_difference"
mkdir -p "$DIFF_DIR" "$GIF_DIR"

echo "========================================================================"
echo "  Density Difference (Δρ) GIF Generator"
echo "========================================================================"
echo "  Calculates: Δρ = ρ(hybrid) - ρ(surface) - ρ(molecule)"
echo "  Then generates all 3 GIF types for visualization"
echo "========================================================================"
echo ""

# Function to extract base name
# Removes _opt_*, _FSI_opt_*, or _graphene_opt_* and everything after
get_base_pattern() {
    local file="$1"
    echo "$file" | sed -E 's/_(opt|FSI_opt|graphene_opt)_.*$//'
}

# Function to get the suffix after _opt, _FSI_opt, or _graphene_opt
get_suffix() {
    local file="$1"
    echo "$file" | sed -E 's/^.*_(opt|FSI_opt|graphene_opt)_//'
}

# Find all _opt_ files (but exclude _FSI_opt and _graphene_opt)
count=0
declare -A processed

for opt_file in *_opt_*_DENS.CUBE; do
    [ -e "$opt_file" ] || continue

    # Skip if this is actually a _FSI_opt or _graphene_opt file
    [[ "$opt_file" == *"_FSI_opt_"* ]] && continue
    [[ "$opt_file" == *"_graphene_opt_"* ]] && continue

    # Extract base pattern and suffix
    base=$(get_base_pattern "$opt_file")
    suffix=$(get_suffix "$opt_file")

    # Skip if already processed
    key="${base}_${suffix}"
    [ -n "${processed[$key]}" ] && continue
    processed[$key]=1

    # Construct expected FSI and graphene filenames
    fsi_file="${base}_FSI_opt_${suffix}"
    graphene_file="${base}_graphene_opt_${suffix}"

    # Check if all three exist
    if [ ! -e "$fsi_file" ]; then
        echo "⚠ Skipping: $opt_file (missing $fsi_file)"
        continue
    fi

    if [ ! -e "$graphene_file" ]; then
        echo "⚠ Skipping: $opt_file (missing $graphene_file)"
        continue
    fi

    count=$((count + 1))

    echo "========================================================================"
    echo "[$count] Processing Density Difference"
    echo "========================================================================"
    echo "  Hybrid:   $opt_file"
    echo "  Molecule: $fsi_file"
    echo "  Surface:  $graphene_file"
    echo ""

    # Create output filename
    # Use base + descriptive name
    short_base=$(echo "$base" | sed -E 's/4LG_FSI_//' | sed -E 's/_2x2_ABAB//' | sed -E 's/_charge\+potential//')
    output_name="${short_base}_delta_rho"
    output_cube="${DIFF_DIR}/${output_name}.CUBE"

    # Perform subtraction with grid alignment
    echo "  [1/4] Calculating Δρ = hybrid - surface - molecule..."
    $PYTHON $VIZ_SCRIPT \
        --subtract "$opt_file" "$graphene_file" "$fsi_file" \
        --align-grids \
        --output-cube "$output_cube" \
        > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "        ✗ Subtraction failed"
        continue
    fi

    size=$(du -h "$output_cube" | cut -f1)
    echo "        ✓ Created: $output_cube ($size)"

    # Now generate GIFs for the difference CUBE
    echo ""
    echo "  [2/4] Rotation GIF..."
    $PYTHON $VIZ_SCRIPT "$output_cube" \
        --save-gif "${GIF_DIR}/${output_name}_rotation.gif" \
        --rotation-frames $ROTATION_FRAMES \
        --gif-fps $ROTATION_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "${GIF_DIR}/${output_name}_rotation.gif" | cut -f1)
        echo "        ✓ Saved: ${GIF_DIR}/${output_name}_rotation.gif ($size)"
    else
        echo "        ✗ Failed"
    fi

    echo ""
    echo "  [3/4] Iso-Browse GIF..."
    $PYTHON $VIZ_SCRIPT "$output_cube" \
        --iso-browse \
        --iso-count $ISO_COUNT \
        --save-gif "${GIF_DIR}/${output_name}_iso_browse.gif" \
        --gif-fps $ISO_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT_BROWSER \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "${GIF_DIR}/${output_name}_iso_browse.gif" | cut -f1)
        echo "        ✓ Saved: ${GIF_DIR}/${output_name}_iso_browse.gif ($size)"
    else
        echo "        ✗ Failed"
    fi

    echo ""
    echo "  [4/4] Slice-Browse GIF..."
    $PYTHON $VIZ_SCRIPT "$output_cube" \
        --slice-browse z \
        --save-gif "${GIF_DIR}/${output_name}_slice_browse.gif" \
        --gif-fps $SLICE_FPS \
        --gif-width $WIDTH \
        --gif-height $HEIGHT_BROWSER \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        size=$(du -h "${GIF_DIR}/${output_name}_slice_browse.gif" | cut -f1)
        echo "        ✓ Saved: ${GIF_DIR}/${output_name}_slice_browse.gif ($size)"
    else
        echo "        ✗ Failed"
    fi

    echo ""
done

if [ $count -eq 0 ]; then
    echo "No matching triplets found."
    echo ""
    echo "Expected pattern:"
    echo "  basename_opt_suffix_DENS.CUBE"
    echo "  basename_FSI_opt_suffix_DENS.CUBE"
    echo "  basename_graphene_opt_suffix_DENS.CUBE"
else
    echo "========================================================================"
    echo "  Complete! Processed $count density difference(s)"
    echo "========================================================================"
    echo "  CUBE files: ${DIFF_DIR}/"
    echo "  GIF files:  ${GIF_DIR}/"
    echo ""
    echo "  Each difference includes:"
    echo "    - *_rotation.gif (360° rotation)"
    echo "    - *_iso_browse.gif (isovalue slider)"
    echo "    - *_slice_browse.gif (Z-slice animation)"
    echo "========================================================================"
fi
