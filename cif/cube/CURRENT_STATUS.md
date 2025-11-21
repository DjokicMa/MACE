# CRYSTAL CUBE Visualizer - Current Status & Bug Fix Plan

**Date**: 2025-10-24
**Script**: `crystal_cubeviz_plotly.py`
**Version**: Post-Phase 1 Implementation

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [What Works Well](#what-works-well)
3. [Critical Bugs](#critical-bugs)
4. [High Priority Issues](#high-priority-issues)
5. [Medium Priority Issues](#medium-priority-issues)
6. [Feature Requests](#feature-requests)
7. [Bug Fix Plan](#bug-fix-plan)
8. [Recent Changes](#recent-changes)

---

## Executive Summary

### Current State
The CRYSTAL CUBE visualizer has successfully implemented:
- ✅ 3D isosurface browser with slider control
- ✅ Slice browser for Z-axis visualization
- ✅ Automatic vacuum detection and cropping
- ✅ Smart isosurface value generation
- ✅ Symmetric isosurface display (spin up/down, potential ±)
- ✅ Toggle controls for showing/hiding isosurfaces
- ✅ Element-specific atom colors (VESTA scheme)

### What Needs Immediate Attention
Several critical bugs have been identified that affect the accuracy and usability:

**CRITICAL (Breaks functionality)**:
1. Atom persistence in slice-browse (atoms stick from previous frames)
2. POT slice-browse only shows positive values (wrong colorscale)
3. --iso flag fails with negative scientific notation
4. Spin up/down isosurfaces at different scales

**HIGH PRIORITY (Incorrect visualization)**:
5. X/Y slice alignment off (only Z works)
6. Missing spin up/down and pos/neg labels
7. Global/Auto scale toggle breaks after first use
8. Auto scale doesn't affect contours

---

## What Works Well

### Standard 3D Isosurface View
```bash
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE
python crystal_cubeviz_plotly.py 4LG_FSI_POT.CUBE
python crystal_cubeviz_plotly.py 4LG_FSI_SPIN.CUBE
```
**Status**: ✅ **WORKS PERFECTLY**
- Automatic isovalue detection
- Clean visualization
- Atoms displayed correctly
- Vacuum cropping working

### 3D Isosurface Browser
```bash
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --iso-browse
```
**Status**: ✅ **MOSTLY WORKS**
- Slider control works
- Symmetric display for spin/potential (red/blue surfaces)
- Toggle buttons for spin up/down and potential ±
- Minor issues: button positioning, axis scaling changes when toggling

### Z-Axis Slice Browser
```bash
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --slice-browse z
```
**Status**: ⚠️ **WORKS FOR DENSITY, BROKEN FOR POTENTIAL/SPIN**
- Density: Shows heatmap + contours + atoms correctly
- Potential: Only shows positive values (wrong colorscale)
- Spin: Shows pos/neg but scale bar uses "p" notation
- Atom persistence bug (atoms stick from previous frames)

### Automatic Vacuum Detection
**Status**: ✅ **WORKS PERFECTLY**
- Correctly detects 2D materials with vacuum
- Reduces memory by ~50% for typical 2D materials
- Prints clear diagnostic information

---

## Critical Bugs

### 🔴 BUG 1: Atom Persistence in Slice-Browse
**File**: `slice_browser_standalone.py` lines 153-184
**Severity**: CRITICAL - Breaks visualization accuracy

**Problem**:
When browsing slices, atoms from previous frames "stick" and appear on slices where they shouldn't be. This happens because frames have inconsistent trace counts:
- Frame with atoms: `[heatmap, contour, scatter]` (3 traces)
- Frame without atoms: `[heatmap, contour]` (2 traces)

Plotly requires **all frames to have the same number of traces**.

**Example**:
- User views center slice (shows atoms correctly)
- User slides to edge slice (no atoms in this region)
- BUG: Center slice atoms still visible on edge slice

**Root Cause**:
```python
# CURRENT (BROKEN):
if atom_x:  # Only add scatter trace if atoms present
    trace_list.append(go.Scatter(...))
```

**Fix**:
```python
# CORRECT:
# Always add scatter trace (even if empty) to maintain frame structure
trace_list.append(go.Scatter(
    x=atom_x,  # Will be empty list if no atoms
    y=atom_y,
    ...
))
```

**User Quote**: *"atoms sometime linger if i go to extremes of the unit cell, it shows the atoms well beyond where it should"*

---

### 🔴 BUG 2: POT Slice-Browse Only Shows Positive Values
**File**: `slice_browser_standalone.py` lines 121-150
**Severity**: CRITICAL - Shows incorrect data

**Problem**:
Potential (POT) slice browser only displays positive values, even though the data contains negative values. The heatmap and contours are incorrectly using 'Viridis' colorscale (sequential) instead of 'RdBu' (diverging).

**Example**:
- POT file has range: [-0.127, 1035.1]
- Slice browser shows: Only positive region visible
- Expected: Blue (negative), white (zero), red (positive)

**Root Cause**:
```python
# CURRENT (BROKEN):
go.Heatmap(
    z=display_data.T,
    colorscale='Viridis',  # WRONG for signed data!
    ...
)
go.Contour(
    z=display_data.T,
    colorscale='Viridis',  # WRONG for signed data!
    ...
)
```

**Fix**:
```python
# CORRECT:
# Detect data type and choose appropriate colorscale
if cube.data_type in ['spin', 'potential']:
    colorscale = 'RdBu'  # Diverging for signed data
    zmid = 0  # Center colorscale at zero
else:
    colorscale = 'Viridis'  # Sequential for density

go.Heatmap(
    z=display_data.T,
    colorscale=colorscale,
    zmid=zmid if cube.data_type in ['spin', 'potential'] else None,
    ...
)
```

**User Quote**: *"With slice-browse on pot files, it wierdly only shows pos values even though I know some of these values are absolutely negative"*

---

### 🔴 BUG 3: --iso Flag Fails with Negative Scientific Notation
**File**: `crystal_cubeviz_plotly.py` line 2381
**Severity**: CRITICAL - Breaks command-line usage

**Problem**:
Parser fails when user provides negative isovalues in scientific notation:
```bash
python crystal_cubeviz_plotly.py file.CUBE --iso -1.00e-012,-1.00e-013,1.00e-013,1.00e-012
# ERROR: argument --iso: expected one argument
```

The leading `-` is interpreted as a new flag.

**Root Cause**:
```python
# CURRENT (BROKEN):
parser.add_argument('--iso', type=str, help='...')
# When user types: --iso -1.00e-012,...
# Argparse sees: --iso (flag) followed by -1.00e-012 (new flag)
```

**Fix**:
```python
# CORRECT (use action with special handling):
parser.add_argument('--iso', type=str, action='store', metavar='VALUES',
                   help='Isosurface values (use quotes for negative: "--iso=-1e-5,1e-5")')
```

Or instruct users to use quotes:
```bash
python crystal_cubeviz_plotly.py file.CUBE --iso="-1.00e-012,-1.00e-013,1.00e-013,1.00e-012"
```

**User Quote**: *"Another issue i found is that --iso flag seems to fail when i use neg values and scientific notation"*

---

### 🔴 BUG 4: Spin Up/Down Isosurfaces at Different Scales
**File**: `crystal_cubeviz_plotly.py` lines 707-727
**Severity**: CRITICAL - Shows incorrect relative magnitudes

**Problem**:
When generating isovalues for spin data, we only check `np.max(np.abs(data))` which might come from either spin-up or spin-down. If spin-up maximum is 1e-9 and spin-down maximum is 5e-11, the isovalue range will be biased toward spin-up, making spin-down surfaces appear too small or disappear.

**Root Cause**:
```python
# CURRENT (BROKEN):
abs_max = np.max(np.abs(data))  # Might be from either sign
isovalues = np.logspace(log_min, log_max, n_values)
```

This doesn't ensure both positive and negative sides are properly represented.

**Fix**:
```python
# CORRECT:
# Check both positive and negative maxima separately
data_max_pos = np.max(data[data > 0]) if np.any(data > 0) else 0
data_max_neg = abs(np.min(data[data < 0])) if np.any(data < 0) else 0

# Use the larger of the two to ensure both sides are visible
abs_max = max(data_max_pos, data_max_neg)

# Generate symmetric isovalues
isovalues = np.logspace(log_min, log_max, n_values)
```

**VESTA Confirmation**: User confirmed in VESTA that spin up and spin down should appear at similar scales.

**User Quote**: *"I can confirm in VESTA I properly see the spin up and spin down isosurfaces which should be around the same scaling"*

---

## High Priority Issues

### ⚠️ ISSUE 5: X/Y Slice Alignment Off (Z Works)
**File**: `slice_browser_standalone.py` lines 68-83
**Severity**: HIGH - X/Y slices show misaligned atoms

**Problem**:
Z-axis slicing works perfectly (atoms align with density), but X and Y axis slicing show atoms in incorrect positions relative to the density/potential data.

**Root Cause**:
Coordinate transformation for X and Y slices may not be accounting for the slice position offset correctly.

**Status**: Partially fixed in Phase 1B but needs verification

**User Quote**: *"keep in mind the position when we project on z slices line up well, the x and y ones are still off like we previously discussed"*

---

### ⚠️ ISSUE 6: Missing Spin Up/Down and Pos/Neg Labels
**Severity**: HIGH - Lack of clarity in interpretation

**Problem**:
- Standard 3D view: No indication of what colors mean (red=up/pos, blue=down/neg)
- Slice-browse: No indication of what colors mean
- Only iso-browse has labels (via toggle buttons)

**Fix Needed**:
Add text annotations or subtitle to all visualization modes:
- Spin: "Red = Spin Up | Blue = Spin Down"
- Potential: "Red = Positive | Blue = Negative"

**User Quote**: *"Please consider some thing about the standard run with no flags, calculations like the spin and pot do not currently show what the colors are spin up or down / pos or neg potential"*

---

### ⚠️ ISSUE 7: Global/Auto Scale Toggle Breaks After First Use
**File**: `slice_browser_standalone.py` lines 220-244
**Severity**: HIGH - Toggle becomes unresponsive

**Problem**:
- First toggle: Global → Auto (works)
- Second toggle: Auto → Global (works)
- Third toggle: Global → Auto (BREAKS - doesn't respond)

Likely a Plotly state issue or incorrect restyle arguments.

**User Quote**: *"It works the first time going to auto and back to global. But if i am still on the same slice frame, it seemingly refuses to go back to auto"*

---

### ⚠️ ISSUE 8: Auto Scale Doesn't Affect Contours
**File**: `slice_browser_standalone.py` lines 220-244
**Severity**: HIGH - Incomplete functionality

**Problem**:
The Global/Auto scale toggle only affects the heatmap trace, not the contour trace. When user selects "Auto Scale", they expect both heatmap AND contours to rescale to the current frame's data range.

**Current Behavior**:
```python
args=[{"zmin": [None, None], "zmax": [None, None]}]
# Only affects first two traces (heatmap + contour)
# But doesn't update contour line spacing
```

**Expected Behavior**:
- Auto Scale: Contours calculated per-frame with optimal spacing
- Global Scale: Contours use consistent values across all frames

**User Quote**: *"currently the auto vs global thing works well the first time and adjusts the heatmap, but it does not do anything for the contour"*

---

## Medium Priority Issues

### 📋 ISSUE 9: Scale Bar Shows "p" Instead of Scientific Notation
**File**: `slice_browser_standalone.py` lines 121-150
**Severity**: MEDIUM - Cosmetic, but confusing

**Problem**:
Slice browser colorbar shows values like "8.32p - -8.32p" instead of proper scientific notation "8.32×10⁻¹²".

The "p" stands for "pico" (10⁻¹²), which is Plotly's default SI prefix formatting.

**Fix**:
Force scientific notation in colorbar:
```python
colorbar=dict(
    title=f'{cube.data_type.capitalize()}',
    len=0.7,
    tickformat='.2e'  # Force scientific notation
)
```

**User Quote**: *"the scale bar is strange it goes from 8.32p - -8.32p , im not sure what p is"*

---

### 📋 ISSUE 10: Missing Atom Hover Labels in Slice-Browse
**File**: `slice_browser_standalone.py` lines 167-181
**Severity**: MEDIUM - Usability issue

**Problem**:
In slice browser, hovering over atoms doesn't show what element they are.

**Current Code**:
```python
go.Scatter(
    x=atom_x,
    y=atom_y,
    mode='markers',
    name='Atoms'  # Generic name
)
```

**Fix**:
```python
# Add hover information per atom
atom_elements = []
atom_numbers = []
for atom_idx, pos in enumerate(cube.atomic_positions):
    # Check if atom is in this slice
    if in_slice:
        z_num = cube.atomic_numbers[atom_idx]
        props = get_atom_properties()
        element = props.get(z_num, {}).get('name', f'Z{z_num}')
        atom_elements.append(element)
        atom_numbers.append(atom_idx)

go.Scatter(
    x=atom_x,
    y=atom_y,
    mode='markers',
    text=[f"{elem} (atom {idx})" for elem, idx in zip(atom_elements, atom_numbers)],
    hovertemplate='%{text}<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<extra></extra>',
    name='Atoms'
)
```

**User Quote**: *"Why does slice-browse not tell me the what the atom is when i highlight over it"*

---

### 📋 ISSUE 11: Incomplete Periodic Table Coverage
**File**: `crystal_cubeviz_plotly.py` lines 747-771
**Severity**: MEDIUM - Some elements show as "Z##"

**Problem**:
Currently only ~20 common elements defined. Need all 118 elements.

**Current State**:
- Have: H, Li, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Fe, Cu, Zn, Se, Sn
- Missing: All lanthanides, actinides, and many transition metals

**Fix**:
Add complete periodic table with VESTA colors.

**User Quote**: *"We should aim to capture all elements in the periodic table"*

---

### 📋 ISSUE 12: Toggle Button Positioning Overlaps Title (3D View)
**File**: `crystal_cubeviz_plotly.py` lines 1320-1348
**Severity**: MEDIUM - Cosmetic issue

**Problem**:
Toggle buttons in 3D isosurface browser overlap with plot title.

**Current Position**:
```python
y=1.15,  # Too high, overlaps title
```

**Fix**:
```python
y=1.08,  # Move down slightly
```

**User Quote**: *"The toggle overlaps with the titles"*

---

### 📋 ISSUE 13: Axis Scaling Changes When Toggling (3D View)
**File**: `crystal_cubeviz_plotly.py` lines 1287-1348
**Severity**: MEDIUM - Makes comparison difficult

**Problem**:
When using toggle buttons (Show Both / Up Only / Down Only), the 3D view automatically zooms/rescales to fit visible traces. This makes it hard to compare where features are located.

**Expected**: Axes should maintain constant range regardless of which traces are visible.

**Fix**:
Set fixed axis ranges in layout:
```python
scene=dict(
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    zaxis=dict(range=[z_min, z_max]),
    aspectmode='data'
)
```

**User Quote**: *"I dont like that the scaling/crop changes when we scroll through the toggles. ideally they should stay at the same zoom or scale as the both option"*

---

## Feature Requests

### 💡 FEATURE 1: GIF Export
**Severity**: LOW - Nice to have

**Request**:
- Export 3D plot with rotation as GIF
- Export slice/iso browser progression as GIF

**Implementation**:
Would require `kaleido` or `plotly.io.write_image` with frame iteration.

**User Quote**: *"another cool feature would be to allow for use to save a gif of the 3d plots rotating. And for the browse like isobrowse and slicebrowse, maybe we should be able to use a flag to save a gif"*

---

## Bug Fix Plan

### Phase 1: Critical Bugs (IMMEDIATE)
**Timeline**: Today (2025-10-24)
**Priority**: CRITICAL - Must be fixed before further use

1. ✅ **BUG 1: Atom Persistence** (30 min)
   - Modify slice_browser_standalone.py lines 153-184
   - Always add scatter trace (even if empty)
   - Test on edge slices

2. ✅ **BUG 2: POT Colorscale** (20 min)
   - Modify slice_browser_standalone.py lines 121-150
   - Detect data type and use RdBu for spin/potential
   - Add zmid=0 for diverging colorscales

3. ✅ **BUG 3: --iso Parser** (15 min)
   - Modify crystal_cubeviz_plotly.py line 2381
   - Add documentation for proper usage
   - Test with negative values

4. ✅ **BUG 4: Spin Scaling** (25 min)
   - Modify crystal_cubeviz_plotly.py lines 707-727
   - Check both positive and negative maxima
   - Use max of both for symmetric range

**Total Time**: ~90 minutes

---

### Phase 2: High Priority Issues (NEXT)
**Timeline**: Today (2025-10-24 afternoon)
**Priority**: HIGH - Affects interpretation accuracy

5. ⚠️ **ISSUE 6: Add Spin/Potential Labels** (45 min)
   - Modify all plotting functions
   - Add subtitle or annotation
   - Test all modes (standard, iso-browse, slice-browse)

6. ⚠️ **ISSUE 7-8: Fix Scale Toggle** (60 min)
   - Debug why toggle breaks after multiple uses
   - Make auto scale affect both heatmap and contours
   - Add per-frame contour recalculation for auto mode

7. ⚠️ **ISSUE 5: Verify X/Y Alignment** (30 min)
   - Test current implementation
   - Fix if still broken
   - Add visual verification

**Total Time**: ~135 minutes

---

### Phase 3: Medium Priority Issues (LATER)
**Timeline**: Tomorrow (2025-10-25)
**Priority**: MEDIUM - Usability improvements

8. 📋 **ISSUE 9: Scale Bar Formatting** (10 min)
9. 📋 **ISSUE 10: Atom Hover Labels** (30 min)
10. 📋 **ISSUE 11: Complete Periodic Table** (45 min)
11. 📋 **ISSUE 12: Button Positioning** (10 min)
12. 📋 **ISSUE 13: Axis Scaling Consistency** (30 min)

**Total Time**: ~125 minutes

---

### Phase 4: Cleanup & Documentation
**Timeline**: Tomorrow (2025-10-25)
**Priority**: LOW - Housekeeping

13. 🧹 **Cleanup HTML/MD Files** (15 min)
    - Remove obsolete test files
    - Keep only: CURRENT_STATUS.md, README.md (if exists)
    - Archive old plan.md

14. 📝 **Update Documentation** (30 min)
    - Create usage guide
    - Document all command-line flags
    - Add troubleshooting section

**Total Time**: ~45 minutes

---

## Recent Changes

### 2025-10-24 Morning
1. ✅ Added F and other common elements to `get_atom_properties()`
2. ✅ Fixed toggle button atom visibility (always show atoms)
3. ✅ Added decimation factor to fix isosurface scaling
4. ✅ Fixed isovalue range to use actual data maximum
5. ✅ Added toggle buttons for spin up/down and pot pos/neg
6. ✅ Added warnings for extreme isovalues

### Issues Discovered During Testing
- Atom persistence in slice-browse
- POT colorscale wrong
- --iso parser broken
- Spin scaling asymmetric
- Toggle button positioning
- Axis scaling changes when toggling
- Scale toggle breaks after multiple uses
- Auto scale doesn't affect contours
- Missing labels in standard/slice modes

---

## Testing Strategy

### Critical Bug Verification
After each bug fix, test:
1. **Atom Persistence**: Browse from center to edge slices
2. **POT Colorscale**: Verify negative values visible in blue
3. **--iso Parser**: Test with negative scientific notation
4. **Spin Scaling**: Compare with VESTA visualization

### Regression Testing
Ensure existing functionality still works:
- Standard 3D view (no flags)
- Iso-browse mode
- Slice-browse for Z-axis with density
- Vacuum detection and cropping

---

## Command-Line Examples

### Working Commands
```bash
# Standard 3D view (WORKS)
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE
python crystal_cubeviz_plotly.py 4LG_FSI_POT.CUBE
python crystal_cubeviz_plotly.py 4LG_FSI_SPIN.CUBE

# 3D isosurface browser (MOSTLY WORKS)
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --iso-browse

# Z-axis slice browser density (WORKS)
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --slice-browse z
```

### Broken Commands (Need Fix)
```bash
# BROKEN: --iso with negative values
python crystal_cubeviz_plotly.py 4LG_FSI_SPIN.CUBE --iso -1e-12,-1e-13,1e-13,1e-12
# FIX: Use quotes
python crystal_cubeviz_plotly.py 4LG_FSI_SPIN.CUBE --iso="-1e-12,-1e-13,1e-13,1e-12"

# BROKEN: POT slice browser
python crystal_cubeviz_plotly.py 4LG_FSI_POT.CUBE --slice-browse z
# Shows only positive values

# BROKEN: X/Y slice browsers
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --slice-browse x
python crystal_cubeviz_plotly.py 4LG_FSI_DENS.CUBE --slice-browse y
# Atoms misaligned
```

---

## Summary

The CRYSTAL CUBE visualizer has made excellent progress with Phase 1 implementation (vacuum detection, smart isovalues, isosurface browser). However, several critical bugs have been discovered during user testing that need immediate attention.

**Top Priority**: Fix the 4 critical bugs that affect data accuracy (atom persistence, POT colorscale, --iso parser, spin scaling).

**Next Steps**: Add labels for spin/potential interpretation, fix scale toggle issues, verify X/Y slice alignment.

**Long Term**: Complete periodic table, add GIF export, improve documentation.

---

**End of Status Document**
