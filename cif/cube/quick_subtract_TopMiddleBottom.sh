#!/bin/bash
# Quick subtraction script for TopMiddleBottom system
# Usage: ./quick_subtract_TopMiddleBottom.sh

HYBRID="4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential"
FSI="4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential"
GRAPHENE="4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential"

echo "========================================"
echo "DENSITY DIFFERENCE (Δρ)"
echo "========================================"
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract ${HYBRID}_DENS.CUBE ${FSI}_DENS.CUBE ${GRAPHENE}_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE

echo ""
echo "========================================"
echo "POTENTIAL DIFFERENCE (ΔV)"
echo "========================================"
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract ${HYBRID}_POT.CUBE ${FSI}_POT.CUBE ${GRAPHENE}_POT.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_POT.CUBE

echo ""
echo "========================================"
echo "SPIN DIFFERENCE (Δσ)"
echo "========================================"
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract ${HYBRID}_SPIN.CUBE ${FSI}_SPIN.CUBE ${GRAPHENE}_SPIN.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_SPIN.CUBE

echo ""
echo "========================================"
echo "COMPLETE! Results saved:"
echo "  TopMiddleBottom_DELTA_RHO.CUBE  (density difference)"
echo "  TopMiddleBottom_DELTA_POT.CUBE  (potential difference)"
echo "  TopMiddleBottom_DELTA_SPIN.CUBE (spin difference)"
echo ""
echo "Visualize with:"
echo "  python crystal_cubeviz_plotly.py TopMiddleBottom_DELTA_RHO.CUBE --iso-browse"
echo "  python crystal_cubeviz_plotly.py TopMiddleBottom_DELTA_POT.CUBE --iso-browse"
echo "  python crystal_cubeviz_plotly.py TopMiddleBottom_DELTA_SPIN.CUBE --iso-browse"
echo "========================================"
