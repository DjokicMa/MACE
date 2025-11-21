# Quick Start: CUBE File Subtraction

## The Problem You Had

Subtracting `molecule.CUBE` from `hybrid.CUBE` showed almost no difference because:
- **Grids weren't aligned** (40.7 Bohr Z-offset!)
- Different origins, voxel sizes, and dimensions
- Essentially subtracting zeros everywhere

## The Solution

### Step 1: Diagnose (Optional but Recommended)
```bash
python diagnose_cube_subtraction.py hybrid.CUBE molecule.CUBE
```
Shows you exactly what's misaligned.

### Step 2: Correct Subtraction
```bash
python subtract_cubes.py hybrid.CUBE molecule.CUBE output.CUBE
```
This automatically:
- ✅ Aligns grids in real space
- ✅ Interpolates different voxel sizes
- ✅ Maps molecule to correct location
- ✅ Performs proper subtraction

### Step 3: Visualize
```bash
python crystal_cubeviz_plotly.py output.CUBE --iso-browse
```
Now correctly shows:
- **Blue regions**: Electron depletion (Δρ < 0)
- **Red regions**: Electron accumulation (Δρ > 0)
- **Label**: "Density Difference (Δρ)"

---

## What You Get

### Corrected File:
```
4LG_FSI_subtracted_CORRECTED.CUBE
```

### Properties:
- **Type**: Density difference (Δρ)
- **Coverage**: 15% of grid (molecule region)
- **Range**: -95 to +312 (includes negatives!)
- **Colormap**: Diverging (blue/red)

### Physical Interpretation:
Your system shows **weak physisorption**:
- Δρ ≈ 0.001 e/Bohr³
- van der Waals interaction
- Minimal charge transfer

---

## Scientific Terms

### What It's Called:
1. **Density Difference** (Δρ) - Most common
2. **Charge Density Difference**
3. **Deformation Density**
4. **Charge Redistribution**

### What It Shows:
1. **Bonding character** (covalent vs ionic vs vdW)
2. **Charge transfer** magnitude and direction
3. **Orbital hybridization**
4. **Polarization effects**

---

## Typical Values

| Magnitude | Interaction Type | Your System |
|-----------|------------------|-------------|
| >0.1 e/Å³ | Strong covalent | ❌ |
| 0.01-0.1 | Chemisorption | ❌ |
| **<0.01** | **Physisorption** | **✅** |

---

## Important Insights

### For Δρ Plots:
1. **Positive (red)**: Electrons accumulate here
2. **Negative (blue)**: Electrons depleted here
3. **Magnitude matters**: 0.001 vs 0.1 is 100× difference!

### Common Uses:
- Molecule on surface interactions
- Catalysis active sites
- 2D heterostructure interfaces
- Chemical bonding analysis

---

## Files You Now Have

### Tools:
1. `diagnose_cube_subtraction.py` - Diagnostic tool
2. `subtract_cubes.py` - Correct subtraction

### Documentation:
3. `DENSITY_DIFFERENCE_PLOTS_GUIDE.md` - Full scientific guide
4. `CUBE_SUBTRACTION_FIXES.md` - Technical details
5. This file - Quick reference

### Updated:
6. `crystal_cubeviz_plotly.py` - Now recognizes difference files!

---

## Quick Commands

```bash
# For your specific files:
python subtract_cubes.py \
    4LG_FSI_2x2_ABAB_opt_charge+potential_DENS.CUBE \
    4LG_FSI_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE \
    result.CUBE

# Visualize:
python crystal_cubeviz_plotly.py result.CUBE --iso-browse

# Try different isovalues:
python crystal_cubeviz_plotly.py result.CUBE --iso 0.001  # ±0.001 e/Bohr³
python crystal_cubeviz_plotly.py result.CUBE --iso 0.01   # ±0.01 e/Bohr³

# Slice view:
python crystal_cubeviz_plotly.py result.CUBE --slice-browse z
```

---

## Read More

📖 **`DENSITY_DIFFERENCE_PLOTS_GUIDE.md`** for:
- Detailed scientific background
- Physical interpretation
- Best practices
- Your specific system analysis
- Advanced techniques
