# High-Quality GIF Generation Guide for CUBE Files

## 🎬 Quick Start

### Generate ALL animation types for specific files:
```bash
# Single file - all three GIF types
./generate_all_gifs.sh my_density.CUBE

# Multiple specific files
./generate_all_gifs.sh file1.CUBE file2.CUBE file3.CUBE

# All density files
./generate_all_gifs.sh *DENS*.CUBE

# All files (WARNING: can take a long time!)
./generate_all_gifs.sh
```

### Generate specific animation types:
```bash
# Only rotation GIFs (360° spinning views)
./generate_rotation_gifs.sh

# Only iso-browse GIFs (isovalue slider animations)
./generate_iso_browse_gifs.sh

# Only slice-browse GIFs (Z-slice animations)
./generate_slice_browse_gifs.sh
```

---

## 📊 Animation Types

### 1. **Rotation GIFs** (`rotation_gifs/`)
**What it shows:** 360° rotation of 3D isosurface around vertical (Z) axis

**Best for:**
- Presentations and talks
- Understanding 3D molecular/surface structure
- Publication supplementary materials
- Quick visual overview of structure

**Default Settings:**
- **Frames:** 72 (5° per frame = very smooth)
- **FPS:** 15 (smooth playback)
- **Resolution:** 1200×1200 px (square, high quality)
- **Duration:** ~4.8 seconds per loop

**Output example:** `4LG_2x2_AA_opt_charge+potential_DENS_rotation.gif`

---

### 2. **Iso-Browse GIFs** (`iso_browse_gifs/`)
**What it shows:** Animation through different isovalue thresholds (electron density levels)

**Best for:**
- Understanding density distribution gradients
- Showing how structure changes with different cutoff values
- Comparing core vs valence electron regions
- Educational demonstrations of isosurface concepts

**Default Settings:**
- **Isovalues:** 30 levels (logarithmic spacing)
- **FPS:** 8 (readable slider values)
- **Resolution:** 1200×900 px (includes slider UI)
- **Duration:** ~3.75 seconds per loop

**Special handling:**
- **Density difference files:** Uses small symmetric values (1e-4 to 1.0 e/Bohr³)
- **Spin/Potential files:** Shows positive surface evolution
- **Regular density:** Standard logarithmic spacing

**Output example:** `4LG_2x2_AA_opt_charge+potential_DENS_iso_browse.gif`

---

### 3. **Slice-Browse GIFs** (`slice_browse_gifs/`)
**What it shows:** 2D cross-sections sliding through Z-axis (height)

**Best for:**
- 2D materials showing layer-by-layer structure
- Understanding charge distribution in planes
- Interfacial systems and surface adsorption
- Comparing different Z-heights

**Default Settings:**
- **FPS:** 10 (smooth scrolling)
- **Resolution:** 1200×900 px (includes slider UI)
- **Colormap:** Automatic based on data type
  - `viridis` for density
  - `RdBu_r` for differences/spin/potential

**Output example:** `4LG_2x2_AA_opt_charge+potential_DENS_slice_browse.gif`

---

## ⚙️ Advanced Customization

### Modify settings in the scripts

Edit the script files to change quality/performance tradeoffs:

#### **For smoother (but larger) rotation GIFs:**
```bash
ROTATION_FRAMES=120     # 3° per frame (ultra smooth)
FPS=20                  # 20 fps (very smooth playback)
WIDTH=1920              # Full HD width
HEIGHT=1920             # Full HD height
```

#### **For faster rendering (smaller files):**
```bash
ROTATION_FRAMES=36      # 10° per frame (standard)
FPS=10                  # 10 fps (standard playback)
WIDTH=800               # Standard resolution
HEIGHT=800
```

#### **For presentation quality (balanced):**
```bash
ROTATION_FRAMES=72      # 5° per frame (smooth)
FPS=15                  # 15 fps (smooth)
WIDTH=1200              # High resolution
HEIGHT=1200
```

---

## 🎨 File Type Auto-Detection

The scripts automatically detect file types and apply appropriate settings:

| File Pattern | Detection | Special Handling |
|--------------|-----------|------------------|
| `*DENS*` | Electron density | Standard isosurface |
| `*POT*` | Electrostatic potential | `--show-both` (±surfaces) |
| `*SPIN*` | Spin density | `--show-both` (α/β) |
| `*SUBTRACTED*` | Density difference (Δρ) | `--show-both`, symmetric isovalues |
| `*difference*` | General difference | `--show-both`, diverging colormap |

---

## 📐 Dependencies

Required packages (already installed):
- ✓ **kaleido** (v0.2.1) - For static image export from Plotly
- ✓ **Pillow** (v11.1.0) - For GIF assembly
- ✓ **plotly** - For 3D visualization
- ✓ **numpy** - For numerical operations

---

## 💾 File Size Considerations

Typical GIF file sizes (depends on complexity):

| Animation Type | Typical Size | Range |
|----------------|--------------|-------|
| Rotation (72 frames) | 2-5 MB | 1-10 MB |
| Iso-Browse (30 frames) | 1-3 MB | 0.5-5 MB |
| Slice-Browse (auto frames) | 3-8 MB | 2-15 MB |

**Tips to reduce file size:**
1. Lower resolution (`WIDTH`, `HEIGHT`)
2. Fewer frames (`ROTATION_FRAMES`, `N_ISOVALUES`)
3. Lower FPS (8-10 instead of 15)

**Tips to increase quality:**
1. More frames (120+ for rotation)
2. Higher resolution (1920×1920)
3. Higher FPS (20-30)

---

## 🎯 Use Case Examples

### **For a presentation slide:**
```bash
# Single high-quality rotation GIF for one key structure
./generate_all_gifs.sh most_important_structure_DENS.CUBE

# Use: rotation_gifs/most_important_structure_DENS_rotation.gif
```

### **For supplementary materials:**
```bash
# All three animation types for all density files
./generate_all_gifs.sh *DENS*.CUBE

# Provides comprehensive visualization suite
```

### **For density difference analysis:**
```bash
# First, create subtracted file with proper detection
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract hybrid.CUBE surface.CUBE molecule.CUBE \
    --align-grids \
    --output-cube SUBTRACTED_result.CUBE

# Then generate GIFs showing charge redistribution
./generate_all_gifs.sh SUBTRACTED_result.CUBE
```

### **For comparison of multiple systems:**
```bash
# Generate rotation GIFs for all density files to compare side-by-side
./generate_rotation_gifs.sh
```

---

## 🔧 Manual Command Examples

For fine control, use the visualization script directly:

### **Custom rotation with specific isovalue:**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    my_density.CUBE \
    --iso 0.01 \
    --save-gif custom_rotation.gif \
    --rotation-frames 90 \
    --gif-fps 15 \
    --gif-width 1600 \
    --gif-height 1600 \
    --rotate-axis z
```

### **Iso-browse with specific range:**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    my_density.CUBE \
    --iso-browse \
    --iso-count 50 \
    --save-gif detailed_iso_browse.gif \
    --gif-fps 10 \
    --gif-width 1400 \
    --gif-height 1000
```

### **Slice-browse with custom FPS:**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    my_density.CUBE \
    --slice-browse z \
    --save-gif smooth_slices.gif \
    --gif-fps 15 \
    --gif-width 1400 \
    --gif-height 1000
```

### **Density difference with both surfaces:**
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    SUBTRACTED_delta_rho.CUBE \
    --iso-browse \
    --show-both \
    --iso-count 40 \
    --save-gif delta_rho_iso_browse.gif \
    --gif-fps 8
```

---

## 📂 Output Directory Structure

After running the scripts, you'll have:

```
current_directory/
├── rotation_gifs/
│   ├── file1_rotation.gif
│   ├── file2_rotation.gif
│   └── ...
├── iso_browse_gifs/
│   ├── file1_iso_browse.gif
│   ├── file2_iso_browse.gif
│   └── ...
└── slice_browse_gifs/
    ├── file1_slice_browse.gif
    ├── file2_slice_browse.gif
    └── ...
```

---

## 🚀 Performance Notes

**Time estimates** (per CUBE file on typical HPC system):
- **Rotation GIF (72 frames):** ~15-30 seconds
- **Iso-Browse GIF (30 frames):** ~10-20 seconds
- **Slice-Browse GIF (auto frames):** ~15-30 seconds

**Processing 10 files with all three modes:** ~5-10 minutes total

**Large batch processing:**
For 50+ CUBE files, consider:
1. Running on compute node (not login node)
2. Processing in batches
3. Running overnight for very large datasets

---

## ✨ Tips for Best Results

### **For publications:**
- Use 1200×1200 or higher resolution
- 72+ frames for rotation
- 15+ FPS for smooth playback
- Save source CUBE files with GIFs for reproducibility

### **For presentations:**
- 1200×1200 resolution is ideal for slides
- 15 FPS provides smooth playback
- Consider adding text annotations in PowerPoint/Keynote

### **For density differences (Δρ):**
- Always use `--show-both` to show accumulation AND depletion
- Iso-browse mode is especially useful for density differences
- The scripts automatically detect and handle SUBTRACTED files

### **For 2D materials:**
- Slice-browse is particularly informative
- Shows layer-by-layer structure
- Use `--z-shift 20` to remove vacuum (automatic in scripts)

---

## 🐛 Troubleshooting

### **"kaleido is required for GIF export"**
```bash
/home/marcus/anaconda3/bin/pip install kaleido
```

### **"Pillow is required for GIF creation"**
```bash
/home/marcus/anaconda3/bin/pip install Pillow
```

### **GIFs are too large**
Reduce resolution or frames in script settings:
```bash
WIDTH=800
HEIGHT=800
ROTATION_FRAMES=36
```

### **Animation is choppy**
Increase FPS in script settings:
```bash
FPS=20  # or higher
```

### **Script is taking too long**
Process fewer files at once:
```bash
./generate_all_gifs.sh *DENS*.CUBE  # Instead of all files
```

---

## 📚 Related Documentation

- `QUICK_START_CUBE_SUBTRACTION.md` - Creating density difference files
- `DENSITY_DIFFERENCE_PLOTS_GUIDE.md` - Scientific background on Δρ plots
- `CUBE_SUBTRACTION_FIXES.md` - Technical details of grid alignment
- `crystal_cubeviz_plotly.py --help` - Full command-line reference

---

## 🎓 Citation

If you use these visualizations in publications, consider citing the MACE toolkit:

> Djokic, M. et al. "MACE: Mendoza Automated CRYSTAL Engine"
> Michigan State University, Mendoza Group (2025)

---

**Generated:** 2025-10-27
**Script Version:** 1.0
**Compatible with:** crystal_cubeviz_plotly.py (latest)
