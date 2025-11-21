# One-Step CUBE Subtraction Commands

## ✨ **All-in-One Subtraction** (Recommended!)

The `crystal_cubeviz_plotly.py` script has built-in subtraction with automatic grid alignment!

---

## 🚀 **Single Command: Subtract & Visualize**

### **Density Difference (Δρ) - Most Important!**

```bash
# One command: hybrid - FSI - graphene → visualize
# NOTE: All files go in --subtract (first is positive, rest are subtracted)
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE \
    --iso-browse
```

**This single command:**
1. ✅ Loads hybrid CUBE
2. ✅ Subtracts FSI CUBE (with interpolation!)
3. ✅ Subtracts graphene CUBE (with interpolation!)
4. ✅ Saves result to `TopMiddleBottom_DELTA_RHO.CUBE`
5. ✅ Opens interactive browser immediately!

---

### **Potential Difference (ΔV)**

```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_POT.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_POT.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_POT.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_POT.CUBE \
    --iso-browse
```

---

### **Spin Difference (Δσ)**

```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_SPIN.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_SPIN.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_SPIN.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_SPIN.CUBE \
    --iso-browse
```

---

## 📁 **Save File Only (No Visualization)**

If you just want to create the difference file:

```bash
# Create difference file without visualizing
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE
```

Then visualize later:
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    TopMiddleBottom_DELTA_RHO.CUBE --iso-browse
```

---

## 🎛️ **Different Visualization Modes**

### **Slice Browser (Better for Interfaces)**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE \
    --slice-browse z
```

### **Static Isosurface**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE \
    --iso 0.001
```

---

## 🧮 **Algebraic Formula Mode (Advanced)**

For more complex operations:

```bash
# Using formula notation
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --formula "H - F - G" \
    --cubes \
        H=4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        F=4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
        G=4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE \
    --iso-browse
```

**With coefficients:**
```bash
# Example: 2*A - 0.5*B - C
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --formula "2*H - 0.5*F - G" \
    --cubes H=hybrid.CUBE F=FSI.CUBE G=graphene.CUBE \
    --align-grids \
    --output-cube custom_diff.CUBE \
    --iso-browse
```

---

## 🔍 **How It Works**

### **`--align-grids` Flag**
This flag enables **automatic grid alignment with interpolation**:

1. **Detects misaligned grids**:
   - Different origins
   - Different voxel sizes
   - Different dimensions

2. **Applies interpolation**:
   - Uses scipy.interpolate if available
   - Maps smaller grid onto larger grid
   - Handles real-space coordinate transformation

3. **Performs subtraction**:
   - Element-wise on aligned grids
   - Preserves larger grid as template

### **What Gets Saved**
The output CUBE file contains:
- **Title**: "CUBE subtraction: file1 - file2 - file3"
- **Grid**: From first file (hybrid)
- **Data**: Difference values (positive and negative!)
- **Metadata**: Proper data type detection

---

## ⚠️ **Important Notes**

### **Requires scipy**
The `--align-grids` feature requires scipy for interpolation. You have anaconda3, so this is included:
```bash
/home/marcus/anaconda3/bin/python -c "import scipy; print('scipy available!')"
```

### **Order Matters**
```bash
--subtract FILE1 FILE2 FILE3
# Result: FILE1 - FILE2 - FILE3
```

**IMPORTANT**: The first file in `--subtract` is positive, all subsequent files are subtracted from it.
NO positional argument is needed - all files go in the `--subtract` list!

### **Alternative: Use subtract_cubes.py**
If you prefer the standalone tool:
```bash
# Step 1: hybrid - FSI
python subtract_cubes.py \
    4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
    4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
    temp.CUBE

# Step 2: temp - graphene
python subtract_cubes.py \
    temp.CUBE \
    4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE \
    TopMiddleBottom_DELTA_RHO.CUBE

# Step 3: Visualize
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    TopMiddleBottom_DELTA_RHO.CUBE --iso-browse
```

---

## 📊 **Expected Results**

After running the density difference command:

```
Data type: density_difference
  ✓ Data contains both positive and negative values
  - Red surfaces: Electron accumulation
  - Blue surfaces: Electron depletion
  - Label: "Density Difference (Δρ)"
```

### **Typical Isovalues**:
- ±0.0001 e/Bohr³ (very sensitive)
- ±0.001 e/Bohr³ (good for physisorption)
- ±0.01 e/Bohr³ (for stronger interactions)

---

## 🎯 **Quick Test**

Try this simple test first:

```bash
# Just hybrid - FSI (simpler, 2 files only)
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE \
        4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
    --align-grids \
    --output-cube TEST_substrate.CUBE \
    --iso-browse
```

If that works well, add the graphene subtraction!

---

## 🚀 **Batch Script for All Three**

Save as `quick_subtract.sh`:

```bash
#!/bin/bash

HYBRID="4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential"
FSI="4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential"
GRAPHENE="4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential"

# Density difference
echo "Creating density difference..."
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    ${HYBRID}_DENS.CUBE \
    --subtract ${FSI}_DENS.CUBE ${GRAPHENE}_DENS.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_RHO.CUBE

# Potential difference
echo "Creating potential difference..."
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    ${HYBRID}_POT.CUBE \
    --subtract ${FSI}_POT.CUBE ${GRAPHENE}_POT.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_POT.CUBE

# Spin difference
echo "Creating spin difference..."
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    ${HYBRID}_SPIN.CUBE \
    --subtract ${FSI}_SPIN.CUBE ${GRAPHENE}_SPIN.CUBE \
    --align-grids \
    --output-cube TopMiddleBottom_DELTA_SPIN.CUBE

echo "Done! Visualize with:"
echo "  python crystal_cubeviz_plotly.py TopMiddleBottom_DELTA_RHO.CUBE --iso-browse"
```

Run it:
```bash
chmod +x quick_subtract.sh
./quick_subtract.sh
```

---

## ✨ **Summary**

**One command does it all:**
```bash
python crystal_cubeviz_plotly.py --subtract FILE1 FILE2 FILE3 --align-grids --iso-browse
```
(First file is positive, rest are subtracted)

**That's it!** No intermediate files, automatic grid alignment, immediate visualization! 🎉
